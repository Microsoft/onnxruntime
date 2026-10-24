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

#include "cuda_hint.cuh"
#include "defines.h"
#if !(IS_MLA)
#include "ldgsts.cuh"
#include "mha.h"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifndef GENERATE_CUBIN
#include "hostUtils.h"
#include <cuda_runtime.h>
#include <string>
#ifndef NDEBUG
#include <cstdio>
#endif
#endif

// There are 4 ways to pass ctaRowMax backward from gemm1 warps to gemm0 warps:
//  1. Protect with xFwdBarriers+xBwdBarriers. This way, ctaRowMax is available to gemm0 warps together with x tiles and
//  warpRowMax/warpRowSum. But ctaRowMax is required before warp tile online softmax, while the other buffers is needed
//  only after online softmax. So xBwdBarriers wait will need to be moved before online softmax.
//  2. Similar to approach 1, but we add an additional register copy of ctaRowMax in gemm0 warps. It's loaded from smem
//  ctaRowMax after warp tile online softmax, so the current warp tile can't use it. But we can pass it to next
//  iteration so softmax of next tile can use it. The update will be delayed by 1 more iteration and we need one or two
//  more registers. Alternatively, put the extra copy in shared memory, so we have double buffer for ctaRowMax.
//  3. Protected with dedicated backward barriers (xFwdBarriers + ctaRowmaxBwdBarriers). Then we don't have drawbacks of
//  1 or 2, but we need extra smem barriers and extra arrive/wait instructions.
//  4. No protection, just use volatile read/write. This approach gives most timely update and has lowest cost, but the
//  result is non-deterministic up to an small numeric error.
// #define CTA_ROW_MAX_BACKWARD_METHOD 4
// 1 is 8% slower than 4. 2/3 are 10% slower than 4.
#define CTA_ROW_MAX_BACKWARD_METHOD 1

static_assert(inputElemSize >= cacheElemSize);

constexpr uint32_t cacheElemsPerGrain = exactDiv(grainBytes, cacheElemSize);
constexpr uint32_t inputElemsPerGrain = exactDiv(grainBytes, inputElemSize);
constexpr bool enableMicroFastPath = false;

// x: horizontal stacking for cta horizontal tile size
// y: vertical stacking for cta vertical tile size
// z: must be 2 for warp specialization.
constexpr uint3 ctaShapeInWarps = {4, 1, 2};

static_assert(ctaShapeInWarps.z == 2);  // for warp specialization
constexpr uint32_t nbWarpsPerCta = ctaShapeInWarps.x * ctaShapeInWarps.y * ctaShapeInWarps.z;
constexpr uint32_t ctaSize = warp_size * nbWarpsPerCta;

#if SPEC_DEC
// Use 32 row size
constexpr uint32_t nbValidRows = rowsPerBlock;
static_assert(nbValidRows <= 32u);
#else
constexpr uint32_t nbValidRows = headGrpSize * beamWidth;
#endif
constexpr uint2 warpTile = {64, roundUp(nbValidRows, 16U)};
static_assert(nbValidRows <= warpTile.y);

constexpr uint32_t gemm1WarpsPerGrp = exactDiv(headElems, warpTile.x);
constexpr uint32_t gemm1NbWarpGrps = exactDiv(ctaShapeInWarps.x, gemm1WarpsPerGrp);  // warp groups split along seqLen dim.

constexpr uint2 ctaTile = {warpTile.x * ctaShapeInWarps.x,  // if .x is greater than headSize, then gemm1 uses split-K
                           warpTile.y* ctaShapeInWarps.y};

constexpr uint32_t cvtExpansion = exactDiv(inputElemSize, cacheElemSize);

#ifndef __CUDA_ARCH__
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#else
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 900
constexpr uint32_t preferedKHeadPartBytes = 128;
__constant__ constexpr uint32_t cacheVTileSeqLen = 64;
#else
// Safe default for older or unknown architectures
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#endif
#endif
constexpr uint32_t kHeadPartBytes = mha::min(preferedKHeadPartBytes, paddedCacheHeadBytes);
// constexpr uint32_t cacheElemsPerKHeadPart = exactDiv(kHeadPartBytes, cacheElemSize);

constexpr bool persistentQ = paddedInputHeadBytes * ctaTile.y <= (16u << 10);
static_assert(persistentQ);
constexpr uint32_t qHeadPartBytes = persistentQ ? paddedInputHeadBytes : kHeadPartBytes;
[[maybe_unused]] constexpr uint32_t qHeadPartElems = exactDiv(qHeadPartBytes, inputElemSize);

constexpr uint32_t nbPartsPerCacheKHead = exactDiv(paddedCacheHeadBytes, kHeadPartBytes);
[[maybe_unused]] constexpr uint32_t nbPartsPerInputKHead = exactDiv(paddedInputHeadBytes, kHeadPartBytes);
constexpr uint32_t nbPartsPerInputQHead = exactDiv(paddedInputHeadBytes, qHeadPartBytes);

// false - each warp load V tiles independent of each other; true - all warps in a warp group load V tiles together.
// @fixme: when true, and nbVBuffers is only 2, we need to sync all warps in a group after finishing using a buffer and
// before refill it with prefetch data. We may need at least 3.
constexpr bool grpLoadV = GRP_LOAD_V;

// number of shared memory buffers for latency hiding
constexpr uint32_t nbQBuffers = mha::min(nbPartsPerInputQHead, 2u);  // for latency hiding
constexpr uint32_t nbKBuffers = 2;                                   // for latency hiding
constexpr uint32_t nbVBuffers = 2;                                   // @fixme: H100 SXM need more in-flight requests. may need to increase this.
constexpr uint32_t nbXBuffers = 1;

__device__ inline uint3 getWarpIdx(const Warp& warp = this_warp()) {
  return uint3{ctaShapeInWarps.x == 1 ? 0 : makeWarpUniform(warp, threadIdx.x / warp_size),
               ctaShapeInWarps.y == 1 ? 0 : makeWarpUniform(warp, threadIdx.y),
               ctaShapeInWarps.z == 1 ? 0 : makeWarpUniform(warp, threadIdx.z)};
}

__device__ inline uint32_t gemm1WarpGrpIdx(uint32_t warpIdxX) {
  return gemm1NbWarpGrps == 1 ? 0 : warpIdxX / gemm1WarpsPerGrp;
}

__device__ inline uint32_t gemm1WarpIdxInGrp(uint32_t warpIdxX) {
  return gemm1WarpsPerGrp == 1 ? 0 : (gemm1NbWarpGrps == 1 ? warpIdxX : warpIdxX % gemm1WarpsPerGrp);
}

constexpr uint32_t instM = 16;
[[maybe_unused]] constexpr uint32_t instN = 8;
// constexpr uint32_t instK = 16;

using QuadRegRowMax = QuadRegRowMaxT<warpTile.y>;            // data is replicated across 4 threads in a MMA quad.
using ThrdRegRowMax = ThrdRegRowMaxT<warpTile.y>;            // unlike QuadRegRowMax, not replicated.
using UniformRescaleMask = UniformRescaleMaskT<warpTile.y>;  // uniform and stored in UR

__device__ inline bool any(const UniformRescaleMask& x) {
  uint32_t val = 0U;
#pragma unroll
  for (uint32_t i = 0; i < x.size; i++) {
    uint32_t word = x[i];
    constexpr uint32_t wordBits = 32;
    if (warpTile.y % wordBits != 0 && i + 1 == x.size) {
      constexpr uint32_t validBits = warpTile.y % wordBits;
      word &= ((1U << validBits) - 1);
    }
    val |= word;
  }
  return val != 0;
}

#ifndef NDEBUG
__device__ inline void printRowMax(const ThrdRegRowMax& src) {
  for (uint32_t i = 0; i < warp_size * src.size; i++) {
    if (laneId() == i % warp_size) {
      printf("%f%s", src[i / warp_size], i == 31 ? "\n" : " ");
    }
    __syncwarp();
  }
}

__device__ inline void printRowMax(const QuadRegRowMax& src) {
  for (uint32_t i = 0; i < src.size / 4; i++) {
    for (uint32_t j = 0; j < 8; j++) {
      if (laneId() == 4 * j) {
        for (uint32_t k = 0; k < 4; k++) {
          printf("%f%s", src[i * 4 + k], i == 31 ? "\n" : " ");
        }
      }
      __syncwarp();
    }
  }
}
#endif

struct alignas(16) SMemWarpRowMax {
  const __device__ inline float& operator[](uint32_t idxRow) const {
    assert(idxRow < ThrdRegRowMax::size * warp_size);
    const uint32_t idxInstM8 = idxRow / quadPerWarp;
    return data[ThrdRegRowMax::size == 1 ? 0 : idxInstM8 / 4][idxRow % quadPerWarp][idxInstM8 % 4];
  }

  __device__ inline float& operator[](uint32_t idxRow) {
    return const_cast<float&>(static_cast<const SMemWarpRowMax&>(*this)[idxRow]);
  }

  // When data is register, data is replicate across 4 threads in a quad.
  template <bool asVolatile>
  __device__ inline const QuadRegRowMax loadToRegForQuad(const Warp& warp) const {
    const uint32_t idxQuad = laneId() / 4;
    QuadRegRowMax result;
#pragma unroll
    for (uint32_t i = 0; i < divUp(warpTile.y, quadPerWarp * 4); i++) {
      const auto& src = data[i][idxQuad];
      auto& dst = reinterpret_cast<float (&)[4]>(result[4 * i]);
      if constexpr (asVolatile) {
        asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                     : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
                     : "l"(__cvta_generic_to_shared(&src)));
      } else {
        reinterpret_cast<float4&>(dst) = reinterpret_cast<const float4&>(src);
      }
    }
    return result;
  }

  template <bool asVolatile>
  __device__ inline const ThrdRegRowMax loadToReg(const Warp& warp) const {
    ThrdRegRowMax result;
#pragma unroll
    for (uint32_t i = 0; i < result.size; i++) {
      const auto& src = this->operator[](warp_size * i + laneId());
      float& dst = result[i];
      if constexpr (asVolatile) {
        dst = static_cast<const float volatile&>(src);
        // asm volatile("ld.volatile.shared.f32 %0, [%1];\n"
        //     : "=f"(dst) : "l"(__cvta_generic_to_shared(&src)));
      } else {
        dst = src;
      }
    }
    return result;
  }

  template <bool asVolatile>
  __device__ inline void storeFromReg(const Warp& warp, const QuadRegRowMax& regData) {
    for (uint32_t i = 0; i < regData.size; i++) {
      assert(regData[i] == __shfl_sync(0xFU << (laneId() / 4 * 4), regData[i], 0, 4));
    }
    if (laneId() % 4 != 0) {
      return;
    }
    const uint32_t idxQuad = laneId() / 4;
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = data[i][idxQuad];
      const auto& src = reinterpret_cast<const float (&)[4]>(regData[4 * i]);
      if constexpr (asVolatile) {
        asm volatile(
            "st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(__cvta_generic_to_shared(&dst)),
            "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
      } else {
        reinterpret_cast<float4&>(dst) = reinterpret_cast<const float4&>(src);
      }
    }
  }

  template <bool asVolatile>
  __device__ inline void storeFromReg(const Warp& warp, const ThrdRegRowMax& regData) {
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = this->operator[](warp_size * i + laneId());
      assert(!hasBankConflict(&dst));
      const float src = regData[i];
      if constexpr (asVolatile) {
        static_cast<float volatile&>(dst) = src;
      } else {
        dst = src;
      }
    }
  }

  __device__ inline void atomicMaxUpdate(const Warp& warp, const ThrdRegRowMax& regData) {
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = this->operator[](warp_size * i + laneId());
      assert(!hasBankConflict(&dst));
      const float src = regData[i];
      atomicMax(&dst, src);
    }
  }

  float data[ThrdRegRowMax::size][quadPerWarp][4];
};

// cacheVTileSeqLen may be smaller than x cols, so we need multiple v tiles per X tile.
constexpr uint32_t nbCacheVTilesPerXTile = exactDiv(warpTile.x, cacheVTileSeqLen);

[[maybe_unused]] constexpr uint32_t nbWarpGrpsPerXTile = mha::min(nbCacheVTilesPerXTile, gemm1NbWarpGrps);

#if USE_PAGED_KV_CACHE
constexpr uint32_t nbPagesPerWarpTile = (warpTile.x <= tokensPerPage ? 1U : exactDiv(warpTile.x, tokensPerPage));
using KCachePageIndices = Vec<KVCachePageIndex, nbPagesPerWarpTile>;
constexpr uint32_t nbPagesPerVTile = (cacheVTileSeqLen <= tokensPerPage ? 1 : exactDiv(cacheVTileSeqLen, tokensPerPage));
using VCachePageIndices = Vec<KVCachePageIndex, nbPagesPerVTile>;
#endif

static_assert(ctaShapeInWarps.y == 1);

struct alignas(128) SharedMem {
  using QSmemBuffer = Array2D<LdGrain, warpTile.y, exactDiv(qHeadPartBytes, grainBytes)>;
  using KSmemBuffer = Array2D<LdGrain, warpTile.x, exactDiv(kHeadPartBytes, grainBytes)>;
  using XSmemBuffer = Array2D<LdGrain, warpTile.y, exactDiv(inputElemSize* warpTile.x, grainBytes)>;
  using VSmemBuffer = Array2D<LdGrain, cacheVTileSeqLen, exactDiv(grpLoadV ? headElems : warpTile.x, cacheElemsPerGrain)>;

  QSmemBuffer q[ctaShapeInWarps.y][nbQBuffers];
  KSmemBuffer k[ctaShapeInWarps.x][nbKBuffers];
  XSmemBuffer x[ctaShapeInWarps.y][ctaShapeInWarps.x];
  static_assert(nbXBuffers == 1);
  VSmemBuffer v[gemm1NbWarpGrps][grpLoadV ? 1 : gemm1WarpsPerGrp][nbVBuffers];

  SMemWarpRowMax warpRowMax[ctaShapeInWarps.y][ctaShapeInWarps.x];  // the max used when computing this->x
  SMemWarpRowMax warpRowSum[ctaShapeInWarps.y][ctaShapeInWarps.x];  // the row sum of gemm0 output

#if CTA_ROW_MAX_BACKWARD_METHOD == 1 || CTA_ROW_MAX_BACKWARD_METHOD == 2 || CTA_ROW_MAX_BACKWARD_METHOD == 3
  // protected with xFwdBarriers+xBwdBarriers for CTA_ROW_MAX_BACKWARD_METHOD 1 or 2, and with
  // xFwdBarriers+ctaRowMaxBwdBarriers for 3. Cannot reuse warpRowMax because a gemm1 warp is not sure whether other
  // gemm1 warps have finished using it, unless we want to pay extra sync.
  SMemWarpRowMax ctaRowMax[ctaShapeInWarps.y][ctaShapeInWarps.x];
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
  SMemWarpRowMax ctaRowMax[ctaShapeInWarps.y];  // just a hint, no strict protection required if you don't care about
                                                // non-deterministic output (up to a small numeric error)
#endif

#if BEAM_WIDTH > 1
  Vec<uint32_t, warpTile.x> gemm0CacheIndir[ctaShapeInWarps.x];
  Vec<uint32_t, cacheVTileSeqLen> gemm1CacheIndir[grpLoadV ? gemm1NbWarpGrps : ctaShapeInWarps.x];
#if USE_PAGED_KV_CACHE
  Vec<KCachePageIndices, beamWidth> kCachePages[ctaShapeInWarps.x];
  Vec<VCachePageIndices, beamWidth> vCachePages[grpLoadV ? gemm1NbWarpGrps : ctaShapeInWarps.x];
#endif
#endif

  using Barrier = CtaBarrier;

  Barrier qBarrier[ctaShapeInWarps.y];
  // Beside X buffers, also protects warpRowMax and warpRowSum. For CTA_ROW_MAX_BACKWARD_METHOD==1 or 2, also
  // ctaRowMax.
  CtaBarrierPair xBarriers[ctaShapeInWarps.y][ctaShapeInWarps.x];
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
  Barrier ctaRowMaxBwdBarriers[ctaShapeInWarps.y]
                              [ctaShapeInWarps.x];  // xFwdBarriers+ctaRowMaxBwdBarriers protects ctaRowMax
#endif

#if GRP_LOAD_V
  static constexpr uint32_t nbOtherBarriers = nbVBuffers * gemm1NbWarpGrps + gemm1NbWarpGrps;
  Barrier otherBarriers[nbOtherBarriers];
#endif
  __device__ inline Barrier* vBarrier(uint32_t warpGrpIdx, uint32_t idxBuf) {
#if GRP_LOAD_V
    return &reinterpret_cast<Barrier(&)[gemm1NbWarpGrps][nbVBuffers]>(otherBarriers)[warpGrpIdx][idxBuf];
#else
    return nullptr;
#endif
  }

  __device__ inline Barrier* warpGrpBar(uint32_t warpGrpIdx) {
#if GRP_LOAD_V
    return &otherBarriers[nbVBuffers * gemm1NbWarpGrps + warpGrpIdx];
#else
    return nullptr;
#endif
  }
};

CUBIN_EXPORT __device__ constexpr uint32_t smemSize = sizeof(SharedMem);
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static_assert(smemSize < kMAX_SMEM_SIZE);
#endif

#if 0
template <bool swizzled, uint32_t rows, uint32_t cols>
__device__ inline void smemRotateInplace(const Warp& Warp, Array2D<LdGrain, rows, cols>& data, uint32_t idxPart, uint32_t idxToken) {
    static_assert(inputSeqLen == 1);
    constexpr uint32_t rowElems = inputElemsPerGrain * cols;
    constexpr uint32_t nbParts = exactDiv(headElems, idxPart);
    static_assert(nbParts % 2 == 0);
    const bool isFirstHalf = (idxPart < nbParts / 2);
    static_assert(mha::is_same_v<InputElem, half>, "not implemented");
    if constexpr (cols <= warp_size) {
        static_assert(warp_size % cols == 0);
        constexpr uint32_t thrdGrpSize = LdGrain::size * cols;
        const uint32_t idxThrdGrp = laneId() / thrdGrpSize;
        const uint32_t thrdGrpLane = laneId() % thrdGrpSize;
        constexpr uint32_t nbThrdGrps = warp_size / thrdGrpSize;
        static_assert(warp_size % thrdGrpSize == 0);
        constexpr uint32_t nbElemsPerWord = exactDiv(sizeof(LdGrain::Elem), inputElemSize);
        Vec<float, nbElemsPerWord> cosAngles;
        Vec<float, nbElemsPerWord> sinAngles;
#pragma unroll
        for (uint32_t i = 0; i < angles.size; i++) {
            const uint32_t n = rowElems * (idxPart % (nbParts / 2)) + angles.size * thrdGrpLane + i;
            const float angle = powf(1E-4f, n * (2.f / headElems)) * idxToken;
            sincosf(angle, &sinAngles[i], &cosAngles[i]);
        }

        constexpr uint32_t nbIters = exactDiv(rows, nbThrdGrps);
#pragma unroll
        for (uint32_t i = 0; i < nbIters; i++) {
            const auto word = data.template at<swizzled>(nbThrdGrps * i + idxThrdGrp, thrdGrpLane / LdGrain::size)[thrdGrpLane % LdGrain::size];
            const float2 val = __half22float2(reinterpret_cast<const InputElem2&>(word));
            Vec<float, nbElemsPerWord> result;
#pragma unroll
            for (uint32_t j = 0; j < nbElemsPerWord; j++) {
                if (isFirstHalf) {
                    result[j] = cosAngles[j] * ;
                }
            }
        }
    }
    else {
        static_assert(cols <= warp_size, "not implemented");
    }
}
#endif

using WarpAcc = WarpAccT<warpTile.y, warpTile.x>;

#if SPEC_DEC
#define MMAS_N_PER_MASK 2

__device__ inline void applyMaskFromInput(const Warp& warp, WarpAcc& acc, const MaskType* mask, uint32_t rowOffset,
                                          uint32_t nbValidCols, uint32_t qSeqLen, uint32_t actualQSeqLen, uint32_t headGrpSize
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                                          ,
                                          int32_t tok0WinBeg, uint32_t seqIter, const uint32_t cacheSeqLen, const uint32_t warpTileTokenBeg
#endif
) {
  const uint32_t idxInQuad = laneId() % 4;
  const uint32_t idxQuad = laneId() / 4;
  // Packed mask is aligned with 32 bits (2 uint16_t).
  const uint32_t nbPackedMasksPerRow = divUp(qSeqLen, 32u) * 2u;
  const uint16_t* uint16Mask = reinterpret_cast<const uint16_t*>(mask);
  constexpr uint64_t fullMask = ~uint64_t{0};
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
  const Range tileRange = {warpTileTokenBeg, warpTileTokenBeg + warpTile.x};
  const Range maxMaskOutRange = {0, mha::max(0, tok0WinBeg) + (nbValidRows / MMAS_N_PER_MASK - 1)};
  const bool ctaNeedBegMask = tileRange.beg < maxMaskOutRange.end;
  assert(ctaNeedBegMask == overlap(tileRange, maxMaskOutRange));
  const int32_t tok0NbMaskOut = int32_t(tok0WinBeg) - int32_t(warpTileTokenBeg);
  const uint32_t nbSeqItersWithoutSpecDecMask = (cacheSeqLen - actualQSeqLen) / ctaTile.x;
  const bool ctaNeedSpecDecMask = (seqIter >= nbSeqItersWithoutSpecDecMask);
#else
  constexpr bool ctaNeedBegMask = false;
  const bool ctaNeedSpecDecMask = true;
  const int32_t tok0NbMaskOut = -2147483648;
#endif
  const bool needMask = ctaNeedBegMask || ctaNeedSpecDecMask;

  if (!needMask) {
    return;
  }
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      const uint32_t idxQTokInCta = (rowOffset + instM * m + idxQuad + i * 8) / headGrpSize;
      const uint32_t tokenRow = min(idxQTokInCta, actualQSeqLen - 1);
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
      const int32_t begNbMaskOut = tok0NbMaskOut + int32_t(idxQTokInCta);
      const uint64_t begMask = (begNbMaskOut > 0 ? fullMask << begNbMaskOut : fullMask);
#else
      const uint64_t begMask = fullMask;
#endif

#pragma unroll
      for (uint32_t mask_n = 0; mask_n < acc.cols / MMAS_N_PER_MASK; mask_n++) {
        const uint32_t firstCol = instN * mask_n * MMAS_N_PER_MASK + InstAcc::cols * idxInQuad;
        const uint32_t lastCol = firstCol + instN * (MMAS_N_PER_MASK - 1) + InstAcc::cols - 1;
        const uint32_t maskPos0 = firstCol + actualQSeqLen < nbValidCols
                                      ? 0u
                                      : min(firstCol + actualQSeqLen - nbValidCols, actualQSeqLen - 1);
        const uint32_t maskPos1 = lastCol + actualQSeqLen < nbValidCols
                                      ? 0u
                                      : min(lastCol + actualQSeqLen - nbValidCols, actualQSeqLen - 1);
        const uint32_t maskPosStart = (maskPos0 / 16) * 16;
        uint32_t packedMask = ~uint32_t{0};
        if (ctaNeedSpecDecMask) {
          reinterpret_cast<uint16_t*>(&packedMask)[0] = uint16Mask[tokenRow * nbPackedMasksPerRow + (maskPos0 / 16)];
          reinterpret_cast<uint16_t*>(&packedMask)[1] = uint16Mask[tokenRow * nbPackedMasksPerRow + (maskPos1 / 16)];
        }
#pragma unroll
        for (uint32_t nj = 0; nj < MMAS_N_PER_MASK; nj++) {
#pragma unroll
          for (uint32_t j = 0; j < InstAcc::cols; j++) {
            const uint32_t n = (mask_n * MMAS_N_PER_MASK + nj);
            const uint32_t col = instN * n + InstAcc::cols * idxInQuad + j;
            // const bool maskFlag = col + qSeqLen < nbValidCols ? true : mask[tokenRow * qSeqLen + (col +
            // qSeqLen - nbValidCols)];
            const bool maskFlag = col + actualQSeqLen < nbValidCols
                                      ? true
                                      : packedMask & (1u << ((col + actualQSeqLen - nbValidCols) - maskPosStart));

            const bool begMaskFlag = ctaNeedBegMask ? (begMask & (1ULL << col)) : true;

            acc(m, n)(i, j) = maskFlag && begMaskFlag && col < nbValidCols ? acc(m, n)(i, j) : SAFE_INIT_ROW_MAX;
          }
        }
      }
    }
  }
}
#endif

__device__ inline QuadRegRowMax warpTileOnlineSoftmax(const Warp& warp, const QuadRegRowMax& rowMaxHint, WarpAcc& acc) {
  QuadRegRowMax rowMax = rowMaxHint;
// compute per-thread row max
#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++) {
          rowMax[m * InstAcc::rows + i] = fmaxf(rowMax[m * InstAcc::rows + i], acc(m, n)(i, j));
        }
      }
    }
  }
// compute warp row max
#pragma unroll
  for (uint32_t xorMask = 2; xorMask != 0; xorMask /= 2) {
#pragma unroll
    for (uint32_t i = 0; i < rowMax.size; i++) {
      rowMax[i] = fmaxf(rowMax[i], __shfl_xor_sync(~0U, rowMax[i], xorMask));
    }
  }
// update acc and rowMax
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      const float maxVal = rowMax[m * InstAcc::rows + i];
      const float bias = maxVal * log2e;
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          float& elem = acc(m, n)(i, j);
          assert(maxVal >= elem);
          elem = exp2f(elem * log2e - bias);
        }
      }
    }
  }
  return rowMax;
}

using GemmOutRegTile = Array2D<InputElem2, WarpAcc::rows * InstAcc::rows, WarpAcc::cols * exactDiv(InstAcc::cols, 2)>;

__device__ inline GemmOutRegTile toFp16(const WarpAcc& acc) {
  GemmOutRegTile dst;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j += 2) {
#if INPUT_FP16
          dst(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2) = __floats2half2_rn(acc(m, n)(i, j), acc(m, n)(i, j + 1));
#else
          dst(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2) = __floats2bfloat162_rn(acc(m, n)(i, j), acc(m, n)(i, j + 1));
#endif
        }
      }
    }
  }
  return dst;
}

__device__ inline WarpAcc toWarpAcc(const GemmOutRegTile& outTile) {
  WarpAcc acc;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j += 2) {
#if INPUT_FP16
          const float2 fp32Vals = __half22float2(outTile(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2));
#else
          const float2 fp32Vals = __bfloat1622float2(outTile(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2));
#endif
          acc(m, n)(i, j) = fp32Vals.x;
          acc(m, n)(i, j + 1) = fp32Vals.y;
        }
      }
    }
  }
  return acc;
}

__device__ inline QuadRegRowMax computeRowSum(const Warp& warp, const GemmOutRegTile& src) {
  Vec<InstAcc, exactDiv(GemmOutRegTile::rows, InstAcc::rows)> acc{};
#if INPUT_FP16
  const InputElem2 b[2][1] = {__floats2half2_rn(1, 1), __floats2half2_rn(1, 1)};
#else
  const InputElem2 b[2][1] = {__floats2bfloat162_rn(1, 1), __floats2bfloat162_rn(1, 1)};
#endif
#pragma unroll
  for (uint32_t n = 0; n < exactDiv(GemmOutRegTile::cols, 2); n++) {
#pragma unroll
    for (uint32_t m = 0; m < exactDiv(GemmOutRegTile::rows, 2); m++) {
      const InputElem2 a[2 /*kEx*/][2 /*mEx*/] = {src(m * 2, n * 2), src(m * 2 + 1, n * 2), src(m * 2, n * 2 + 1), src(m * 2 + 1, n * 2 + 1)};
      mma<InputElem>(acc[m].data, reinterpret_cast<const uint32_t (&)[2][2]>(a),
                     reinterpret_cast<const uint32_t (&)[2][1]>(b));
    }
  }
  QuadRegRowMax rowSum;
#pragma unroll
  for (uint32_t i = 0; i < acc.size; i++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::rows; j++) {
      rowSum[i * InstAcc::rows + j] = acc[i](j, 0);
#pragma unroll
      for (uint32_t k = 0; k < InstAcc::cols; k++) {
        assert(acc[i](j, k) == acc[i](j, 0));
      }
    }
    rowSum[i * 2] = acc[i](0, 0);
    rowSum[i * 2 + 1] = acc[i](1, 0);
  }
// Sometimes there are errors in sum and they mismatch inside a quad. Force broadcast from lane 0 of each quad to
// eliminate mismatch. This has no visible impact on final result and can be removed.
#pragma unroll
  for (uint32_t i = 0; i < QuadRegRowMax::size; i++) {
    const auto lane0Val = __shfl_sync(0xFU << (laneId() / 4 * 4), rowSum[i], 0, 4);
    // Disable the assert, sometimes it triggers because of different orders of accumulation.
    // assert(fabs(rowSum[i] - lane0Val) < 1E-4f);
    rowSum[i] = lane0Val;
  }
  return rowSum;
}

__device__ inline void storeOrderedGemmOutTile(const Warp& warp, SharedMem::XSmemBuffer& dst, const GemmOutRegTile& src) {
  static_assert(sizeof(dst) == sizeof(src) * warp_size);
  const uint32_t lane = laneId();
#if __CUDA_ARCH__ >= 900
  constexpr uint2 storeUnits = {4, 1};  // in 8x8 b16 matrices.
  static_assert(storeUnits.x * storeUnits.y == 4);
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8 * storeUnits.y); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8 * storeUnits.x); n++) {
      const uint32_t idxRowLocal = lane % 8;
      const uint32_t flatIdxMatLocal = lane / 8;
      const uint2 idxMatLocal = {flatIdxMatLocal % storeUnits.x, flatIdxMatLocal / storeUnits.x};
      LdGrain* const p = &dst.template at<true>(
          8 * (storeUnits.y * m + idxMatLocal.y) + idxRowLocal, storeUnits.x * n + idxMatLocal.x);

      LdGrain data;
#pragma unroll
      for (uint32_t i = 0; i < storeUnits.y; i++) {
#pragma unroll
        for (uint32_t j = 0; j < storeUnits.x; j++) {
          data[i * storeUnits.x + j] = reinterpret_cast<const uint32_t&>(src(m * storeUnits.y + i, n * storeUnits.x + j));
        }
      }
      stmatrix_4x<false>(warp, p, data);
    }
  }
#else
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8); n++) {
      const uint32_t idxRowLocal = laneId() / 4;
      const uint32_t idxWordLocal = laneId() % 4;
      dst.template at<true>(8 * m + idxRowLocal, n)[idxWordLocal] = reinterpret_cast<const uint32_t&>(src(m, n));
    }
  }
#endif
}

// Reorder to compensate the reorder caused by V cache load+conversion.
__device__ inline void reorderAndStoreGemmOutTile(
    const Warp& warp, SharedMem::XSmemBuffer& dst, const GemmOutRegTile& src) {
  static_assert(sizeof(dst) == sizeof(src) * warp_size);
  const uint32_t lane = laneId();
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8 * 2); n++) {
      const uint32_t idxRowLocal = laneId() / 4;
      const uint32_t idxSegLocal = laneId() % 4;
      Vec<InputElem2, cvtExpansion> seg;
#pragma unroll
      for (uint32_t e = 0; e < cvtExpansion; e++) {
        seg[e] = src(m, n * cvtExpansion + e);
      }
      // reorder
      // Ideally compiler should be able to fuse this into toFp16() and just reorder input registers of F2FP
      // instructions.
      Vec<InputElem, cvtExpansion * 2> reorderedSeg;
#pragma unroll
      for (uint32_t e = 0; e < cvtExpansion; e++) {
        reorderedSeg[e] = seg[e].x;
        reorderedSeg[cvtExpansion + e] = seg[e].y;
      }
      static_assert(cvtExpansion <= LdGrain::size);
      constexpr uint32_t nbSegPerGrain = exactDiv(grainBytes, sizeof(seg));
      reinterpret_cast<Vec<uint32_t, cvtExpansion>&>(dst.template at<true>(8 * m + idxRowLocal,
                                                                           n * cvtExpansion + idxSegLocal / nbSegPerGrain)[idxSegLocal % nbSegPerGrain * cvtExpansion]) = reinterpret_cast<Vec<uint32_t, cvtExpansion>&>(reorderedSeg);
    }
  }
}

__device__ inline void storeGemmOutTile(
    const Warp& warp, SharedMem::XSmemBuffer& dst, const GemmOutRegTile& src, bool reorder) {
  if (reorder) {
    reorderAndStoreGemmOutTile(warp, dst, src);
  } else {
    storeOrderedGemmOutTile(warp, dst, src);
  }
}

__device__ inline GemmOutRegTile loadGemmOutTile(const Warp& warp, const SharedMem::XSmemBuffer& src) {
  const uint32_t lane = laneId();
  GemmOutRegTile dst;
  static_assert(sizeof(src) == sizeof(dst) * warp_size);
#if __CUDA_ARCH__ >= 900
  constexpr uint2 storeUnits = {4, 1};  // in 8x8 b16 matrices.
  static_assert(storeUnits.x * storeUnits.y == 4);
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(SharedMem::XSmemBuffer::rows, 8 * storeUnits.y); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(SharedMem::XSmemBuffer::cols * grainBytes / inputElemSize, 8 * storeUnits.x);
         n++) {
      const uint32_t idxRowLocal = lane % 8;
      const uint32_t flatIdxMatLocal = lane / 8;
      const uint2 idxMatLocal = {flatIdxMatLocal % storeUnits.x, flatIdxMatLocal / storeUnits.x};
      const LdGrain* const p = &src.template at<true>(
          8 * (storeUnits.y * m + idxMatLocal.y) + idxRowLocal, storeUnits.x * n + idxMatLocal.x);

      LdGrain data = ldmatrix_4x<false>(warp, p);
#pragma unroll
      for (uint32_t i = 0; i < storeUnits.y; i++) {
#pragma unroll
        for (uint32_t j = 0; j < storeUnits.x; j++) {
          reinterpret_cast<uint32_t&>(dst(m * storeUnits.y + i, n * storeUnits.x + j)) = data[i * storeUnits.x + j];
        }
      }
    }
  }
#else
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(SharedMem::XSmemBuffer::rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(SharedMem::XSmemBuffer::cols * grainBytes / inputElemSize, 8); n++) {
      const uint32_t idxRowLocal = laneId() / 4;
      const uint32_t idxWordLocal = laneId() % 4;
      reinterpret_cast<uint32_t&>(dst(m, n)) = src.template at<true>(8 * m + idxRowLocal, n)[idxWordLocal];
    }
  }
#endif
  return dst;
}
// only the first nbValidRows rows are copied, to allow padding.
__device__ inline void copyOutputToGlobalMem(const Warp& warp, OutputHead* dst, uint32_t nbQHeads,
#if SPEC_DEC
                                             uint32_t headGrpSize, uint32_t idxHeadGrpOffset, uint32_t nbValidHeadTokens,
                                             uint32_t actualQSeqLen,
#else
                                             uint32_t idxHeadGrp,
#endif
                                             uint2 dstOffset, const SharedMem::XSmemBuffer& src) {
  static_assert(sizeof(PaddedInputHead) == grainBytes * SharedMem::XSmemBuffer::cols * gemm1WarpsPerGrp);
#if SPEC_DEC
  static_assert(warpTile.y <= SharedMem::XSmemBuffer::rows);
  unused(actualQSeqLen);
#else
  static_assert(nbValidRows <= SharedMem::XSmemBuffer::rows);
#endif
  constexpr uint32_t nbIters = divUp(nbValidRows * SharedMem::XSmemBuffer::cols, warp_size);
#pragma unroll
  for (uint32_t i = 0; i < nbIters; i++) {
    const uint32_t flatIdx = warp_size * i + laneId();
    const uint32_t r = flatIdx / SharedMem::XSmemBuffer::cols;
    const uint32_t c = flatIdx % SharedMem::XSmemBuffer::cols;
    assert(r < SharedMem::XSmemBuffer::rows);
    const LdGrain data = src.template at<true>(r, c);

    const uint32_t m = dstOffset.y + r;
    const uint32_t n = exactDiv(dstOffset.x, grainBytes / inputElemSize) + c;
#if SPEC_DEC
    if (r >= nbValidHeadTokens) {
#else
    if (nbValidRows * SharedMem::XSmemBuffer::cols % warp_size != 0 && m >= nbValidRows) {
#endif
      break;
    }
#if SPEC_DEC
    // m is a request-wide flattened (token, head) row, so it is not bounded by one tile's height.
    const uint32_t idxBeam = 0;
    const uint32_t idxInGrp = m;
    const uint32_t tokenIdx = idxInGrp / headGrpSize;
    const uint32_t headIdx = idxInGrp % headGrpSize;
    assert(idxBeam < beamWidth);
    const uint32_t idxHead = idxHeadGrpOffset + tokenIdx * nbQHeads + headIdx;
    assert(idxHead < actualQSeqLen * nbQHeads);
#else
    assert(m < nbValidRows);
    const uint32_t idxBeam = m / headGrpSize;
    const uint32_t idxInGrp = m % headGrpSize;
    assert(idxBeam < beamWidth);
    const uint32_t idxHead = headGrpSize * idxHeadGrp + idxInGrp;
    assert(idxHead < nbQHeads);
#endif
    assert(n < paddedInputHeadBytes / grainBytes);
    if (!isHeadPadded || n < ioHeadBytes / grainBytes) {
      const auto outVec = convert<OutputHead::Elem>(reinterpret_cast<const Vec<InputElem, inputElemsPerGrain>&>(data));
      reinterpret_cast<Vec<mha::decay_t<decltype(outVec)>, exactDiv(ioHeadBytes, grainBytes)>&>(
          dst[nbQHeads * idxBeam + idxHead])[n] = outVec;
    }
  }
}

// MMA instruction expansion in GEMM k-dim and m/n-dim, with b16 8x8 as baseline
template <uint32_t kEx_, uint32_t mnEx_>
struct InstInMat {
  static constexpr uint32_t kEx = kEx_;
  static constexpr uint32_t mnEx = mnEx_;
  uint32_t data[kEx][mnEx];
};

template <uint32_t kEx, uint32_t mnEx, bool transOuter>
using InstInMatWTrans = InstInMat<transOuter ? mnEx : kEx, transOuter ? kEx : mnEx>;

//@fixme: for B-mat, use InstInMat<2, 1>[2] instead.

// kEx is for srcCol and mnEx is for srcRow, before transpose.
// rowBeg/colBeg are in src indices
// note that grainBytes-byte swizzling per 128-byte or per row(>=128byte) is applied when loading to avoid bank
// conflict. transOuter: transpose InstInMat with 8x8 b16 matrices as elements unchanged. transInner: transpose the
// elements, i.e. the 8x8 b16 matrices. transOuter=true and transInner=false is for B matrix of 16816. It actually loads
// two 8x16 B matrices for two instructions. transOuter=false and transInner=false is for A matrix of 16816.
template <uint32_t kEx, uint32_t mnEx, bool transOuter, bool transInner, uint32_t srcRows, uint32_t srcCols>
__device__ inline InstInMatWTrans<kEx, mnEx, transOuter> loadInstInMat(
    const Warp& warp, const Array2D<LdGrain, srcRows, srcCols>& src, uint32_t rowOffset, uint32_t colOffset) {
  static_assert(kEx * mnEx == 4, "implemented only for ldmatrix.x4 for now");
  using Dst = InstInMatWTrans<kEx, mnEx, transOuter>;
  assert(rowOffset % (8 * mnEx) == 0 && colOffset % kEx == 0);
  const uint32_t idx = laneId() / 8;
  const uint32_t idxKEx = idx / Dst::mnEx;
  const uint32_t idxMNEx = idx % Dst::mnEx;
  const uint32_t srcIdxKEx = (transOuter ? idxMNEx : idxKEx);
  const uint32_t srcIdxMNEx = (transOuter ? idxKEx : idxMNEx);

  const LdGrain* const ptr = &src.template at<true>(rowOffset + 8 * srcIdxMNEx + laneId() % 8, colOffset + srcIdxKEx);

  const Vec<uint32_t, 4> data = ldmatrix_4x<transInner>(warp, ptr);
  static_assert(sizeof(Dst) == sizeof(data));
  Dst dst;
#pragma unroll
  for (int i = 0; i < data.size; i++) {
    (&dst.data[0][0])[i] = data[i];
  }
  return dst;
}

template <typename T, uint32_t rows, uint32_t cols, bool transpose>
using Array2DWTrans = Array2D<T, transpose ? cols : rows, transpose ? rows : cols>;

// src rows/cols are in src indices
// dst rows/cols are in InstInMatWTrans
// row is contiguous and gemm-K dim.
// kEx combines with dstCols and mnEx combines with dstRows.
template <uint32_t kEx, uint32_t mnEx, uint32_t dstRows, uint32_t dstCols, bool transArr2D, bool transInstInMatOuter,
          bool transInstInMatInner, uint32_t srcRows, uint32_t srcCols /*in LdGrain*/>
__device__ inline Array2DWTrans<InstInMatWTrans<kEx, mnEx, transInstInMatOuter>, dstRows, dstCols, transArr2D>
loadMatrix(const Warp& warp, const Array2D<LdGrain, srcRows, srcCols>& src, uint32_t rowBeg, uint32_t colBeg) {
  assert(rowBeg % (8 * mnEx * dstRows) == 0 && colBeg % (kEx * dstCols) == 0);
  Array2DWTrans<InstInMatWTrans<kEx, mnEx, transInstInMatOuter>, dstRows, dstCols, transArr2D> dst;
#pragma unroll
  for (uint32_t i = 0; i < dstRows; i++) {
#pragma unroll
    for (uint32_t j = 0; j < dstCols; j++) {
      (transArr2D ? dst(j, i) : dst(i, j)) = loadInstInMat<kEx, mnEx, transInstInMatOuter, transInstInMatInner>(
          warp, src, rowBeg + (mnEx * 8) * i, colBeg + kEx * j);
    }
  }
  return dst;
}

// acc is used as both input and output
// qColBeg is in the unit of LdGrain
// using KElemType = int8_t;
template <typename KElemType>
__device__ inline void smemQKPartGemm(
    const Warp& warp, WarpAcc& acc, const SharedMem::QSmemBuffer& q, uint32_t qColBeg, const SharedMem::KSmemBuffer& k) {
  assert(qColBeg % (SharedMem::KSmemBuffer::cols) == 0);
  constexpr uint32_t kEx = 2;
  constexpr uint32_t mnEx = 2;
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert((mha::is_same_v<KElemType, half> || mha::is_same_v<KElemType, __nv_bfloat16> || mha::is_same_v<KElemType, int8_t> || mha::is_same_v<KElemType, __nv_fp8_e4m3>),
                "not implemented");
  constexpr uint32_t nbInstInMatPerSliceInGemmKDim = 1;
  constexpr uint32_t kElemSize = sizeof(KElemType);
  constexpr uint32_t elemsPerKHeadPart = exactDiv(kHeadPartBytes, kElemSize);
  constexpr uint32_t gemmKSplit = exactDiv(elemsPerKHeadPart, 8 * kEx * nbInstInMatPerSliceInGemmKDim);

  // @fixme: check if compiler mixes LDS+HMMA and does prefetch properly. We are not doing prefetch explicitly. But we
  // do fully unroll and expect compiler to do that for us.
  constexpr uint32_t nbUnroll = cacheElemSize == 2 ? gemmKSplit : 2;
#pragma unroll(nbUnroll)
  for (uint32_t s = 0; s < gemmKSplit; s++) {
    // load q
    constexpr uint32_t qSliceRows = exactDiv(warpTile.y, 8 * mnEx);  // in InstInMat
    constexpr uint32_t qSliceCols = nbInstInMatPerSliceInGemmKDim;
    const Array2D<InstInMat<kEx, mnEx>, qSliceRows, qSliceCols> qSlice = loadMatrix<kEx, mnEx, qSliceRows, qSliceCols, false, false, false>(
        warp, q, 0, qColBeg + kEx * qSliceCols * s);
    // load k
    constexpr uint32_t cvtExp = exactDiv(inputElemSize, kElemSize);
    constexpr uint32_t mnExK = mnEx * cvtExp;
    constexpr uint32_t kExK = exactDiv(kEx, cvtExp);
    constexpr uint32_t kSliceRows = exactDiv(warpTile.x, 8 * mnExK);  // in InstInMat
    constexpr uint32_t kSliceCols = nbInstInMatPerSliceInGemmKDim;
    const Array2D<InstInMat<mnExK, kExK>, kSliceRows, kSliceCols> kSliceOrig = loadMatrix<kExK, mnExK, kSliceRows, kSliceCols, false, true, false>(warp, k, 0, kExK * kSliceCols * s);
    const auto kSlice = [&]() -> Array2D<InstInMat<mnExK, kEx>, kSliceRows, kSliceCols> {
      if constexpr (mha::is_same_v<InputElem, KElemType>) {
        return kSliceOrig;
      } else if constexpr ((mha::is_same_v<KElemType, int8_t> || mha::is_same_v<KElemType, __nv_fp8_e4m3>)) {
        Array2D<InstInMat<mnExK, kEx>, kSliceRows, kSliceCols> ret;
#pragma unroll
        for (uint32_t m = 0; m < kSliceRows; m++) {
#pragma unroll
          for (uint32_t n = 0; n < kSliceCols; n++) {
#pragma unroll
            for (uint32_t i = 0; i < mnExK; i++) {
#pragma unroll
              for (uint32_t j = 0; j < kExK; j++) {
                const auto data = convertKCacheWordToF16<InputElem, KElemType>(kSliceOrig(m, n).data[i][j]);
                ret(m, n).data[i][j * cvtExp] = data[0];
                ret(m, n).data[i][j * cvtExp + 1] = data[1];
              }
            }
          }
        }
        return ret;
      } else {
        assert(!"not implemented");
        trap();
      }
    }();
// compute
#pragma unroll
    for (uint32_t i = 0; i < qSliceRows; i++) {
#pragma unroll
      for (uint32_t j = 0; j < kSliceRows; j++) {
        const InstInMat<kEx, mnEx> matrixA = qSlice(i, 0);
        const InstInMat<mnExK, kEx> matrixB = kSlice(j, 0);
#pragma unroll
        for (uint32_t n = 0; n < mnExK; n++) {
          const uint32_t b[2][1] = {matrixB.data[n][0], matrixB.data[n][1]};
          mma<InputElem>(acc(i, j * mnExK + n).data, matrixA.data, b);
        }
      }
    }
  }
}

// acc is used as both input and output
// v needs transpose
template <typename VElemType>
__device__ inline void smemXVPartGemm(const Warp& warp, WarpAcc& acc, bool skipXRowRescale,
                                      UniformRescaleMask xRowNeedRescaleMask, ThrdRegRowMax xRowScales, const SharedMem::XSmemBuffer& x,
                                      uint32_t idxVTilePerXTile, const SharedMem::VSmemBuffer& vt, uint32_t idxNSplit) {
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert((mha::is_same_v<VElemType, half> || mha::is_same_v<VElemType, __nv_bfloat16> || mha::is_same_v<VElemType, int8_t> || mha::is_same_v<VElemType, __nv_fp8_e4m3>),
                "not implemented");
  constexpr uint32_t kEx = 2;
  constexpr uint32_t mnEx = 2;
  constexpr uint32_t nbInstInMatPerSliceInGemmKDim = 1;
  static_assert(SharedMem::XSmemBuffer::rows == 8 * InstAcc::rows * WarpAcc::rows);
  static_assert(
      grpLoadV || sizeof(SharedMem::VSmemBuffer::Elem) / cacheElemSize * SharedMem::VSmemBuffer::cols == warpTile.x);
  static_assert(
      !grpLoadV || sizeof(SharedMem::VSmemBuffer::Elem) / cacheElemSize * SharedMem::VSmemBuffer::cols == headElems);
  if (grpLoadV) {
    assert(idxNSplit < gemm1WarpsPerGrp);
  } else {
    assert(idxNSplit == 0);
  }
  constexpr uint32_t gemmKSplit = exactDiv(SharedMem::VSmemBuffer::rows, 8 * kEx * nbInstInMatPerSliceInGemmKDim);

  Vec<InputElem2, QuadRegRowMax::size> xRowScalesQuad;
  if (!enableMicroFastPath || !skipXRowRescale) {
    assertWarpConverged();
#if INPUT_FP16
    const Vec<InputElem2, ThrdRegRowMax::size> xRowScalesF16 = __float2half2_rn(xRowScales);
#else
    const Vec<InputElem2, ThrdRegRowMax::size> xRowScalesF16 = __float2bfloat162_rn(xRowScales);
#endif
    static_assert(sizeof(xRowScalesF16) == sizeof(ThrdRegRowMax));
    reinterpret_cast<QuadRegRowMax&>(xRowScalesQuad) = replicateForQuad(warp, reinterpret_cast<const ThrdRegRowMax&>(xRowScalesF16));
  }

// @fixme: check if compiler mixes LDS+HMMA and does prefetch properly. We are not doing prefetch explicitly. But we do
// fully unroll and expect compiler to do that for us.
#pragma unroll
  for (uint32_t s = 0; s < gemmKSplit; s++) {
    // load x
    constexpr uint32_t xSliceRows = exactDiv(warpTile.y, 8 * mnEx);  // in InstInMat
    constexpr uint32_t xSliceCols = nbInstInMatPerSliceInGemmKDim;
    const uint32_t colBeg = SharedMem::XSmemBuffer::cols / nbCacheVTilesPerXTile * idxVTilePerXTile + exactDiv(inputElemSize * 8 * kEx * nbInstInMatPerSliceInGemmKDim, grainBytes) * s;
    Array2D<InstInMat<kEx, mnEx>, xSliceRows, xSliceCols> xSlice = loadMatrix<kEx, mnEx, xSliceRows, xSliceCols, false, false, false>(warp, x, 0u, colBeg);
    if (!enableMicroFastPath || !skipXRowRescale) {
#pragma unroll
      for (uint32_t m = 0; m < xSliceRows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < mnEx; i++) {
          const uint32_t r = m * mnEx + i;
#pragma unroll
          for (uint32_t n = 0; n < xSliceCols; n++) {
#pragma unroll
            for (uint32_t j = 0; j < kEx; j++) {
              InputElem2& elem = reinterpret_cast<InputElem2&>(xSlice(m, n).data[j][i]);
              elem = skipXRowRescale ? elem : elem * xRowScalesQuad[r];
            }
          }
        }
      }
    }
    // load v slice. rows and cols here are before transpose
    constexpr uint32_t mnExV = mnEx * cvtExpansion;
    constexpr uint32_t vSliceCols = exactDiv(warpTile.x, 8 * mnExV);  // in InstInMat
    constexpr uint32_t vSliceRows = nbInstInMatPerSliceInGemmKDim;
    const uint32_t rowBeg = 8 * kEx * nbInstInMatPerSliceInGemmKDim * s;
    const Array2D<InstInMat<mnEx, kEx>, vSliceCols, vSliceRows> vSliceOrig = loadMatrix<mnEx, kEx, vSliceRows, vSliceCols, true, false, true>(
        warp, vt, rowBeg, mnEx * vSliceCols * idxNSplit);
    const Array2D<InstInMat<mnExV, kEx>, vSliceCols, vSliceRows> vSlice = [&]() {
      if constexpr (mha::is_same_v<InputElem, VElemType>) {
        return vSliceOrig;
      } else if constexpr ((mha::is_same_v<VElemType, int8_t> || mha::is_same_v<VElemType, __nv_fp8_e4m3>)) {
        Array2D<InstInMat<mnExV, kEx>, vSliceCols, vSliceRows> ret;
#pragma unroll
        for (uint32_t m = 0; m < ret.rows; m++) {
#pragma unroll
          for (uint32_t n = 0; n < ret.cols; n++) {
            const auto& src = vSliceOrig(m, n);
            auto& dst = ret(m, n);
#pragma unroll
            for (uint32_t i = 0; i < mnEx; i++) {
#pragma unroll
              for (uint32_t j = 0; j < kEx; j++) {
                const auto data = convertVCacheWordToF16<InputElem, VElemType>(src.data[i][j]);
#pragma unroll
                for (uint32_t e = 0; e < cvtExpansion; e++) {
                  dst.data[i * cvtExpansion + e][j] = data[e];
                }
              }
            }
          }
        }
        return ret;
      } else {
        assert(!"not implemented");
        trap();
      }
    }();
// compute
#pragma unroll
    for (uint32_t i = 0; i < xSliceRows; i++) {
#pragma unroll
      for (uint32_t j = 0; j < vSliceCols; j++) {
        const auto& vInMat = vSlice(j, 0);
#pragma unroll
        for (uint32_t n = 0; n < mnExV; n++) {
          mma<InputElem>(acc(i, j * mnExV + n).data, xSlice(i, 0).data,
                         reinterpret_cast<const uint32_t (&)[2][1]>(vInMat.data[n]));
        }
      }
    }
  }
}

__device__ inline void pickAccRowsForBeamSearch(const Warp& warp, WarpAcc& dst, const WarpAcc& src, bool isCtxTile,
                                                uint32_t idxBeam, void (*func)(float& d, float s)) {
  const uint32_t idxQuad = laneId() / 4;
  constexpr uint32_t nbQuads = warp_size / 4;
#pragma unroll
  for (uint32_t m = 0; m < WarpAcc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < WarpAcc::cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          const uint32_t idxRow = instM * m + nbQuads * i + idxQuad;
          if (isCtxTile || (idxRow >= headGrpSize * idxBeam && idxRow < headGrpSize * idxBeam + headGrpSize)) {
            func(dst(m, n)(i, j), src(m, n)(i, j));
          }
        }
      }
    }
  }
}

__device__ inline void rescaleAcc(
    const Warp& warp, WarpAcc& acc, const UniformRescaleMask& rescaleMask, const ThrdRegRowMax& rowScales) {
  static_assert(WarpAcc::rows * InstAcc::rows * 8 <= ThrdRegRowMax::size * warp_size);
// const QuadRegRowMax quadRowScales = replicateForQuad(warp, rowScales);
#pragma unroll
  for (uint32_t m = 0; m < WarpAcc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      const uint32_t r = m * InstAcc::rows + i;  // in 8-row unit.
      const bool skip = enableMicroFastPath && ((rescaleMask[r / 4] & (0xFFU << 8 * r)) == 0);
      if (skip) {  // @fixme: do we need this?
        continue;
      }
      // const float scale = quadRowScales[r]; // @fixme: see if this is faster than the line below.
      const float scale = replicateValForQuad(warp, rowScales, r);
#pragma unroll
      for (uint32_t n = 0; n < WarpAcc::cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          acc(m, n)(i, j) *= scale;
        }
      }
    }
  }
}

__device__ inline void rescaleAcc(const Warp& warp, WarpAcc& acc, float scale) {
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          acc(m, n)(i, j) *= scale;
        }
      }
    }
  }
}

template <bool useFp32Acc, uint32_t nbWarps, uint32_t nbTiles, uint32_t rows, uint32_t cols>
__device__ inline void smemFp16ArraySum(
    uint32_t idxWarp, Array2D<LdGrain, rows, cols>& dst, const Array2D<LdGrain, rows, cols> tiles[nbTiles]) {
  constexpr uint32_t nbThrds = warp_size * nbWarps;
  const uint32_t tid = warp_size * idxWarp + laneId();
  constexpr uint32_t nbGrains = SharedMem::XSmemBuffer::rows * SharedMem::XSmemBuffer::cols;
  constexpr uint32_t nbGrainsPerThrd = exactDiv(nbGrains, nbThrds);
  using AccType = mha::conditional_t<useFp32Acc, float2, InputElem2>;

#pragma unroll
  for (uint32_t i = 0; i < nbGrainsPerThrd; i++) {
    Vec<AccType, LdGrain::size> result;
    result.fill(AccType{0, 0});
    const uint32_t idx = nbThrds * i + tid;
#pragma unroll
    for (uint32_t j = 0; j < nbTiles; j++) {
      const auto data = reinterpret_cast<const Vec<InputElem2, LdGrain::size>(&)[nbGrains]>(tiles[j])[idx];
      if constexpr (useFp32Acc) {
#if INPUT_FP16
        result = addFloat2(result, __half22float2(data));
#else
        result = addFloat2(result, __bfloat1622float2(data));
#endif
      } else {
        result = __hadd2_rn(result, data);
      }
    }
    auto& dstGrain = reinterpret_cast<Vec<InputElem2, LdGrain::size>(&)[nbGrains]>(dst)[idx];
    if constexpr (useFp32Acc) {
#if INPUT_FP16
      dstGrain = __float22half2_rn(result);
#else
      PRAGMA_UNROLL_FP16_ONLY
      for (uint32_t k = 0; k < LdGrain::size; ++k) {
        dstGrain[k] = __floats2bfloat162_rn(result[k].x, result[k].y);
      }
#endif
    } else {
      dstGrain = result;
    }
  }
}

template <uint32_t nbBuffers>
__device__ inline ThrdRegRowMax mergeRowMax(
    const Warp& warp, const TinyPtr<SMemWarpRowMax> rowMaxBuffers, uint32_t nbSubSeqPerSeq) {
  ThrdRegRowMax regBuffers[nbBuffers];
  auto load = [&](uint32_t n) {
    assert(n < nbSubSeqPerSeq);
    regBuffers[n % nbBuffers] = rowMaxBuffers[n].loadToReg<false>(warp);
  };
#pragma unroll
  for (uint32_t i = 0; i < nbBuffers; i++) {
    if (i >= nbSubSeqPerSeq) {
      break;
    }
    load(i);
  }
  ThrdRegRowMax mergedRowMax = regBuffers[0];
  for (uint32_t n = 0; n < divUp(nbSubSeqPerSeq, nbBuffers); n++) {
#pragma unroll
    for (uint32_t i = 0; i < nbBuffers; i++) {
      const uint32_t idx = nbBuffers * n + i;
      if (idx >= nbSubSeqPerSeq) {
        break;
      }
      mergedRowMax = fmaxf(mergedRowMax, regBuffers[i]);
      const uint32_t idxNext = idx + nbBuffers;
      if (idxNext < nbSubSeqPerSeq) {
        load(idxNext);
      }
    }
  }
  return mergedRowMax;
}

__device__ inline void addAttentionSinks(
    ThrdRegRowMax& globalRowSum, const ThrdRegRowMax globalRowMax, const float* attentionSinks
#if SPEC_DEC
    ,
    uint32_t rowOffset, uint32_t nbValidHeadTokens
#endif
) {
  for (uint32_t i = 0; i < globalRowSum.size; i++) {
    uint32_t srcOffset = warp_size * i + laneId();
#if SPEC_DEC
    // Rows are flattened (token, head) pairs, so every token reuses its own head's sink.
    if (srcOffset < nbValidHeadTokens) {
      globalRowSum[i] += expf(attentionSinks[(rowOffset + srcOffset) % headGrpSize] - globalRowMax[i]);
    }
#else
    if (srcOffset < headGrpSize) {
      globalRowSum[i] += expf(attentionSinks[srcOffset] - globalRowMax[i]);
    }
#endif
  }
}

#ifdef NDEBUG
__device__ __forceinline__
#else
CUBIN_EXPORT __global__
#endif
    void
    kernel_mha_impl(
#if SPEC_DEC
        const uint32_t qSeqLen, const uint32_t nbKHeads, const uint32_t headGrpSize,
        const SeqLenDataType* __restrict__ qCuSeqLens,  // [nbReq + 1]
#else
        const uint32_t nbKHeads,
#endif
#if SLIDING_WINDOW
        uint32_t slidingWinSize,
#endif
        float qScale,
        OutputHead* const __restrict__ output,  // [nbReq][beamWidth][nbQHeads]
#if LOW_PREC_OUTPUT
        const float* rcpOutScale,
#endif
        // NOTE: the input is actually Q buffer when integrated to TRT-LLM.
        const IOHead* const __restrict__ q,  // [nbReq][beamWidth][nbQHeads],
#if SPEC_DEC
        const MaskType* __restrict__ mask,  // [qSeqLen, divUp(qSeqLen, 32)].
#endif
        const float* attentionSinks,  // [headGrpSize]
#ifdef NDEBUG
        const KVCacheList<usePagedKVCache>& cacheList,
#if BEAM_WIDTH > 1
        const BeamSearchParams& beamSearchParams,
#endif
#else
        const KVCacheList<usePagedKVCache> cacheList,
#if BEAM_WIDTH > 1
        const BeamSearchParams beamSearchParams,
#endif
#endif
        const uint32_t batchSize,
        // Device memory scalars for quantized KV cache. K and V have independent scales:
        // kCacheScale is folded into qkScale (applied to Q*K.T before softmax) and vCacheScale into
        // voScale (applied to the P*V accumulator). Both are read once per CTA, outside the K/V loop.
        // Either may be null, meaning "scale is 1": the caller has already folded a non-scalar
        // (per-channel) scale into Q / into the output, which is exact because dequantization is
        // linear and the channel dim is contracted for K and free for V.
        const float* __restrict__ kCacheScale,
        const float* __restrict__ vCacheScale,
        uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr) {
  assert(allowMultiBlockMode || gridDim.x == 1);
  const bool isMultiBlock = allowMultiBlockMode && (gridDim.x != 1);
  const uint32_t nbSubSeqPerSeq = allowMultiBlockMode ? gridDim.x : 1;
  const uint32_t idxSubSeqInSeq = allowMultiBlockMode ? blockIdx.x : 0;
  assert(!isMultiBlock || (semaphores != nullptr && scratch != nullptr));

  // gridDim: x - K/V sequence-dim split; y - number of K or V heads per token; z - number of requests
#if SPEC_DEC
  // In speculative mode gridDim.y also fans out over the token tiles of each head group.
  assert(gridDim.z == batchSize && gridDim.y % nbKHeads == 0);
#else
  assert(gridDim.z == batchSize && gridDim.y == nbKHeads);
#endif
  extern __shared__ char smemByteBuf[];
  SharedMem& smem = *reinterpret_cast<SharedMem*>(&smemByteBuf[0]);

  const uint32_t idxReq = blockIdx.z;
#if SPEC_DEC
  // Variable query sequence length support.
  const bool variableQSeqLen = qCuSeqLens != nullptr;
  const uint32_t actualQSeqLen = variableQSeqLen ? uint32_t(qCuSeqLens[idxReq + 1] - qCuSeqLens[idxReq]) : qSeqLen;
  if (actualQSeqLen == 0) {
    return;
  }
  // Same as idxReq * qSeqLen if all sequences all the same.
  // Take different beams as different requests/sequences currently.
  const uint32_t reqSeqOffset = variableQSeqLen ? uint32_t(qCuSeqLens[idxReq]) : (qSeqLen * idxReq);

  const uint32_t nbVHeads = nbKHeads;
  const uint32_t nbQHeads = nbKHeads * headGrpSize;
  const uint32_t nbQHeadTokens = nbQHeads * actualQSeqLen;
  const uint32_t nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;

  const uint32_t nbTokenBlocksPerGrp = gridDim.y / nbKHeads;
  const uint32_t idxHeadGrp = blockIdx.y / nbTokenBlocksPerGrp;  // inside one request
  const uint32_t idxHeadTokenInGrp = (blockIdx.y % nbTokenBlocksPerGrp) * warpTile.y;
  const uint32_t totalNbHeadTokensInGrp = actualQSeqLen * headGrpSize;
  const uint32_t nbValidHeadTokens = idxHeadTokenInGrp > totalNbHeadTokensInGrp
                                         ? 0u
                                         : mha::min(totalNbHeadTokensInGrp - idxHeadTokenInGrp, rowsPerBlock);
  // Shift the mask ptr by batch_idx.
  mask += reqSeqOffset * divUp(qSeqLen, 32u);
#else
  const uint32_t nbQHeads = nbKHeads * headGrpSize;

  const uint32_t idxHeadGrp = blockIdx.y;  // inside one request
#endif

  const auto ctaThrdId = threadIdx.x + warp_size * ctaShapeInWarps.x * (threadIdx.y + ctaShapeInWarps.y * threadIdx.z);
  assert(blockDim.x == ctaShapeInWarps.x * warp_size && blockDim.y == ctaShapeInWarps.y && blockDim.z == ctaShapeInWarps.z);
  const auto warp = this_warp();
  const uint3 warpIdx = getWarpIdx(warp);  // @fixme: use BoundedVal
  assert(warpIdx.x < ctaShapeInWarps.x && warpIdx.y < ctaShapeInWarps.y && warpIdx.z < ctaShapeInWarps.z);
  const uint32_t flatWarpIdPerRow = warpIdx.z * ctaShapeInWarps.x + warpIdx.x;  // per ctaShapeInWarps.y value
  unused(flatWarpIdPerRow);

  // initialize shared memory
  static_assert(persistentQ && ctaShapeInWarps.y == 1);
  if (ctaThrdId < ctaShapeInWarps.y) {
    init(&smem.qBarrier[ctaThrdId], warp_size * ctaShapeInWarps.x);  // be sure to use .noinc
  }
  constexpr uint32_t cacheVTileSeqStride = cacheVTileSeqLen * gemm1NbWarpGrps;
  constexpr uint32_t nbXTilesPerXIter = cacheVTileSeqStride < warpTile.x ? 1 : exactDiv(cacheVTileSeqStride, warpTile.x);
  constexpr uint32_t nbXItersPerCtaTile = exactDiv(ctaShapeInWarps.x, nbXTilesPerXIter);
  constexpr uint32_t nbVItersPerXIter = exactDiv(warpTile.x * nbXTilesPerXIter, cacheVTileSeqStride);
  constexpr uint32_t nbWarpGrpsPerXTile = mha::min(nbCacheVTilesPerXTile, gemm1NbWarpGrps);
  unused(nbWarpGrpsPerXTile);
  static_assert(warpTile.x >= cacheVTileSeqLen, "not implemented yet");
  static_assert(ctaSize >= uint32_t(sizeof(smem.xBarriers) / sizeof(CtaBarrierPair)));
  if (ctaThrdId < uint32_t(sizeof(smem.xBarriers) / sizeof(CtaBarrierPair))) {
    (&smem.xBarriers[0][0])[ctaThrdId].initialize(warp_size, warp_size * gemm1WarpsPerGrp * nbWarpGrpsPerXTile);
  }
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
  static_assert(ctaSize >= sizeof(smem.ctaRowMaxBwdBarriers) / sizeof(SharedMem::Barrier));
  if (ctaThrdId < sizeof(smem.ctaRowMaxBwdBarriers) / sizeof(SharedMem::Barrier)) {
    init(&smem.ctaRowMaxBwdBarriers[0][0] + ctaThrdId, warp_size);
  }
#endif
#if CTA_ROW_MAX_BACKWARD_METHOD != 0
  static_assert(ctaSize >= sizeof(smem.ctaRowMax) / sizeof(float));
  if (ctaThrdId < sizeof(smem.ctaRowMax) / sizeof(float)) {
    reinterpret_cast<float*>(&smem.ctaRowMax[0])[ctaThrdId] = SAFE_INIT_ROW_MAX;
  }
#endif
#if GRP_LOAD_V
  static_assert(ctaSize >= gemm1NbWarpGrps * nbVBuffers);
  if (ctaThrdId < gemm1NbWarpGrps * nbVBuffers) {
    init(smem.vBarrier(0, 0) + ctaThrdId, warp_size * gemm1WarpsPerGrp);
  }
  if (ctaThrdId < gemm1NbWarpGrps) {
    init(smem.warpGrpBar(ctaThrdId), warp_size * gemm1WarpsPerGrp);
  }
#endif
  __syncthreads();

#if ENABLE_PDL
  preExit();
  acqBulk();
#endif

  constexpr bool qkSwizzle = true;
  // load whole Q heads into shared memory
#if SPEC_DEC
  if (warpIdx.z == 0) {
    // map from idxQHead to idxHead in q input.
    const auto localQHeadTokenIdxMap = [nbQHeads, headGrpSize, reqSeqOffset, idxReq, idxHeadTokenInGrp](uint32_t idxHeadTokenLocal) -> uint32_t {
      assert(idxHeadTokenLocal < warpTile.y);  // may be larger than nbValidRows, then the output does not matter.
      if constexpr (beamWidth == 1) {
        idxHeadTokenLocal += idxHeadTokenInGrp;
        const uint32_t tokenIdx = (idxHeadTokenLocal / headGrpSize);
        const uint32_t headIdx = idxHeadTokenLocal % headGrpSize;
        return tokenIdx * nbQHeads + headIdx;
      }
    };
    static_assert(nbValidRows <= warpTile.y);
    const auto srcBase = q;
    const uint32_t idxHeadTokenBeg = nbQHeads * reqSeqOffset + (idxHeadGrp * headGrpSize);
    const TinyPtr<IOHead const> src{srcBase, idxHeadTokenBeg};

    const bool isFullTile = (nbValidHeadTokens == warpTile.y);
    static_assert(nbQBuffers == 1);
    if (isFullTile) {
      copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, true, warpTile.y>(
          warpIdx.x, smem.q[warpIdx.y][0], src, nbValidHeadTokens, localQHeadTokenIdxMap);
    } else {
      copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, false, warpTile.y>(
          warpIdx.x, smem.q[warpIdx.y][0], src, nbValidHeadTokens, localQHeadTokenIdxMap);
    }

    ldgsts::barArrive(smem.qBarrier[warpIdx.y], true);
  }
#else
  if (warpIdx.z == 0) {
    // map from idxQHead to idxHead in q input.
    const auto localQHeadIdxMap = [nbQHeads, idxReq, idxHeadGrp](uint32_t idxHeadLocal) -> uint32_t {
      assert(idxHeadLocal < warpTile.y);  // may be larger than nbValidRows, then the output does not matter.
      if constexpr (beamWidth == 1) {
        return idxHeadLocal;
      }
      const uint32_t idxBeam = idxHeadLocal / headGrpSize;
      const uint32_t result = idxHeadLocal + idxBeam * (nbQHeads - headGrpSize);
      const uint32_t idxQHeadInGrp = idxHeadLocal % headGrpSize;
      const uint32_t ref = nbQHeads * idxBeam + idxQHeadInGrp;
      assert(result == ref);
      unused(ref);
      return result;
    };
    static_assert(nbValidRows <= warpTile.y);
    const auto srcBase = q;
    // NOTE: read from Q buffer directly.
    const uint32_t idxHeadBeg = nbQHeads * beamWidth * idxReq + headGrpSize * idxHeadGrp;
    const TinyPtr<IOHead const> src{srcBase, idxHeadBeg};

    constexpr bool isFullTile = (nbValidRows == warpTile.y);
    static_assert(nbQBuffers == 1);
    copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, isFullTile, warpTile.y>(
        warpIdx.x, smem.q[warpIdx.y][0], src, nbValidRows, localQHeadIdxMap);
    ldgsts::barArrive(smem.qBarrier[warpIdx.y], true);
  }
#endif

  const uint32_t cacheSeqLen = getCacheSeqLen<usePagedKVCache>(cacheList, idxReq)
#if SPEC_DEC
                               + (actualQSeqLen > 0 ? actualQSeqLen - 1 : 0)
#endif
      ;
#if SLIDING_WINDOW && SPEC_DEC && !IS_SPEC_DEC_TREE
  // Position of the request's first query token. applyMaskFromInput() adds the per-row query-token
  // index (derived from the flattened row offset) on top of this, so no tile offset belongs here.
  const uint32_t tok0SeqLen = cacheSeqLen - actualQSeqLen + 1;
  const int32_t tok0WinBeg = int32_t(tok0SeqLen) - int32_t(slidingWinSize);
  const uint32_t nbTotalSkipTokens = mha::max(0, tok0WinBeg);

#elif SLIDING_WINDOW
  const bool rtIsReallySliding = (cacheSeqLen > slidingWinSize);
  assert(!SPEC_DEC || !rtIsReallySliding);
  const uint32_t nbTotalSkipTokens = rtIsReallySliding ? cacheSeqLen - slidingWinSize : 0;
#else
  constexpr bool rtIsReallySliding = false;
  constexpr uint32_t nbTotalSkipTokens = 0;
#endif
  const uint32_t nbSkipLeadingTiles = nbTotalSkipTokens / ctaTile.x;
  const uint32_t tile0NbSkipTokens = nbTotalSkipTokens % ctaTile.x;
  unused(tile0NbSkipTokens);
#if USE_PAGED_KV_CACHE
  const uint32_t nbPages = divUp(cacheSeqLen, tokensPerPage);
  constexpr uint32_t nbPagesPerCtaTile = exactDiv(ctaTile.x, tokensPerPage);
#endif

  const uint32_t nbSeqIters = useKVCache ? divUp(cacheSeqLen, ctaTile.x) : 0;
#if SLIDING_WINDOW && SPEC_DEC && !IS_SPEC_DEC_TREE
  const uint32_t nbSeqItersWithoutMask = nbSkipLeadingTiles;
#elif SPEC_DEC
  const uint32_t nbSeqItersWithoutMask = (cacheSeqLen - actualQSeqLen) / ctaTile.x;
#endif

  const uint32_t seqStrideIters = nbSubSeqPerSeq;
  constexpr bool isKVCacheQuantized = (cacheElemSize < 2);
#if defined(XQA_PAGED_INT4)
  // INT4 dequantizes into FP16 shared memory, so cacheElemSize is 2 and isKVCacheQuantized is false,
  // yet the packed codes still carry the caller's scalar dequant factor.
  constexpr bool hasScalarCacheScale = true;
#else
  constexpr bool hasScalarCacheScale = isKVCacheQuantized;
#endif
  const uint32_t seqIterInit = nbSkipLeadingTiles + idxSubSeqInSeq;
#if BEAM_WIDTH > 1
  const uint32_t nbCtxCtaTiles = beamSearchParams.ctxLenList[idxReq * beamWidth] / ctaTile.x;
#endif
  auto isConvergedTile = [&](uint32_t seqIter) {
#if BEAM_WIDTH == 1
    return true;
#else
    return seqIter < nbCtxCtaTiles;
#endif
  };
  if (warpIdx.z == 0) {
    // qkScale is applied onto Q*K.T before softmax. A null kCacheScale means the scale is already in Q.
    const float qkScale = qScale * ((hasScalarCacheScale && kCacheScale != nullptr) ? kCacheScale[0] : 1.f);
    CircIdx<nbKBuffers> idxCurrSMemKBuf{nbKBuffers - 1};
    const auto getSMemKTile = [&](uint32_t idx) -> SharedMem::KSmemBuffer& { return smem.k[warpIdx.x][idx]; };
#if BEAM_WIDTH > 1
    auto loadCacheIndir = [&](uint32_t seqIter, uint32_t idxBeam) mutable {
      auto& dst = smem.gemm0CacheIndir[warpIdx.x];
      const uint32_t offset = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
      loadIndicesForBeamSearchAsync<1, warpTile.x>(
          0, dst, beamSearchParams, idxReq, idxBeam, offset, cacheSeqLen);
    };
    loadCacheIndir(seqIterInit, 0U);
#endif
#if USE_PAGED_KV_CACHE
#if BEAM_WIDTH == 1
    KCachePageIndices pageIdx = KCachePageIndices::filled(kBAD_PAGE_INDEX);
#endif
    auto loadPages = [&](uint32_t idxPage) mutable {
#if BEAM_WIDTH == 1
      const uint32_t idxBeam = 0;
      pageIdx = getPage<KCachePageIndices::size>(cacheList, true, idxReq, idxBeam, idxPage, nbPages);
#else
      auto& dst = smem.kCachePages[warpIdx.x];
      loadPagesForBeamSearchAsync<1>(0U, dst, cacheList, true, idxReq, idxPage, nbPages);
#endif
    };
    uint32_t idxPageBeg = nbPagesPerCtaTile * seqIterInit + warpIdx.x * warpTile.x / tokensPerPage;
    loadPages(idxPageBeg);
#else
    constexpr uint32_t idxBeamBase = 0U;
    const uint32_t cacheKBaseBatch = cacheList.capacity * nbKHeads * (idxBeamBase + beamWidth * idxReq);
    const uint32_t cacheKSeqBaseOffset = cacheList.isBSNH
                                             ? (cacheKBaseBatch + idxHeadGrp)
                                             : (cacheKBaseBatch + cacheList.capacity * idxHeadGrp);
#endif
    auto loadKTilePart = [&](uint32_t seqIter, uint32_t idxBeam, uint32_t idxPart) mutable {
      assert(idxBeam < beamWidth);
      assert(seqIter % nbSubSeqPerSeq == seqIterInit % nbSubSeqPerSeq);
      const auto idxNextSMemKBuf = idxCurrSMemKBuf.next();
      auto& dst = getSMemKTile(idxNextSMemKBuf);
      const uint32_t dstHeadOffset = 0;
      const uint32_t seqOffset = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
      const uint32_t idxHeadBeg = (seqOffset % tokensPerPage) * nbKHeads + idxHeadGrp;

#else
      const uint32_t idxHeadBeg = tokensPerPage * idxHeadGrp + seqOffset % tokensPerPage;
#endif
#if BEAM_WIDTH == 1
#if PAGED_KV_CACHE_LAYOUT == 1
      const HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> src{
          cacheList.kCacheVLLM, pageIdx, nbKHeads, idxHeadBeg};
#else
      const HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> src{
          cacheList.pool, pageIdx, nbKHeads, idxHeadBeg};
#endif
#else
      const IndexedHeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> src{
          /*indices=*/smem.gemm0CacheIndir[warpIdx.x].data,
#if PAGED_KV_CACHE_LAYOUT == 1
          /*pool=*/cacheList.kCacheVLLM,
#else
          /*pool=*/cacheList.pool,
#endif
          /*pageIndices=*/smem.kCachePages[warpIdx.x].data,
          /*nbKHeads=*/nbKHeads,
          /*offset=*/idxHeadBeg};
#endif
#else
      const uint32_t idxHeadBeg = cacheList.isBSNH
                                      ? (cacheKSeqBaseOffset + seqOffset * nbKHeads)
                                      : (cacheKSeqBaseOffset + seqOffset);
#if BEAM_WIDTH == 1
      const TinyPtr<GMemCacheHead const> src{cacheList.kData, idxHeadBeg};
#else
      const IndexedHeadPtr<GMemCacheHead const, 0, 0> src{/*indices=*/smem.gemm0CacheIndir[warpIdx.x].data,
                                                          /*pointer=*/cacheList.data,
                                                          /*offset=*/idxHeadBeg,
                                                          /*beamStride=*/cacheList.capacity * nbKHeads * 2};
      // trap();
      // assert("not implemented");
#endif
#endif
      // if (threadIdx.x == dbgPrintTid) {
      //     printf("K: seqIter=%u, idxBeam=%u, idxPart=%u: pointers={%p, %p}, indices={", seqIter, idxBeam,
      //     idxPart, src.pointers[0], src.pointers[1]); const uint32_t nbHeadsAvail = mha::min((seqOffset <
      //     cacheSeqLen ? cacheSeqLen - seqOffset : 0U), warpTile.x); for (int i = 0; i < nbHeadsAvail; i++) {
      //         printf("%u, ", src.indices[i]);
      //     }
      //     printf("}\n");
      // }
      const bool isFullTile = (seqIter + 1 < nbSeqIters);
      if (isFullTile) {
        copyPartialHeadsAsync<PaddedCacheHead, warpTile.x, nbPartsPerCacheKHead, qkSwizzle, true>(
            warp, dst, dstHeadOffset, src, idxPart);
      } else {
        const uint32_t nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                               : 0U);  // may also be full but it can be handled correctly anyway
        copyPartialHeadsAsync<PaddedCacheHead, warpTile.x, nbPartsPerCacheKHead, qkSwizzle, false>(
            warp, dst, dstHeadOffset, src, idxPart, nbHeadsAvail);
      }
#if BEAM_WIDTH > 1
      // to make sure all threads has finished usage of cache indir and pages
      __syncwarp();
#endif
      if (idxPart + 1 == nbPartsPerCacheKHead) {
#if USE_PAGED_KV_CACHE
        const bool isForNextSeqIter = isConvergedTile(seqIter) || idxBeam == beamWidth - 1;
        if (isForNextSeqIter) {
          idxPageBeg += nbPagesPerCtaTile * nbSubSeqPerSeq;
          loadPages(idxPageBeg);
        }
#endif
#if BEAM_WIDTH > 1
        uint32_t idxBeamNext, seqIterDelta;
        mha::tie(idxBeamNext, seqIterDelta) = isConvergedTile(seqIter)
                                                  ? mha::tuple<uint32_t, uint32_t>(0U, 1U)
                                                  : carryLE<beamWidth>(idxBeam + 1, 0);  // optimize for context cache
        loadCacheIndir(seqIter + seqStrideIters * seqIterDelta, idxBeamNext);
#endif
      }
    };

#if BEAM_WIDTH > 1
    ldgsts::commitGroup();
    ldgsts::waitGroup<0>();
    __syncwarp();
#endif
    loadKTilePart(seqIterInit, 0, 0);
    ldgsts::commitGroup();  // @fixme: do prefetch for next iter tile if last part
    idxCurrSMemKBuf++;

    auto& xBar = smem.xBarriers[warpIdx.y][warpIdx.x];
    bool xBarConsumedParityNext = false;

    bool qBarParityNext = false;
    auto& qBar = smem.qBarrier[warpIdx.y];
    qBar.wait_parity(qBarParityNext);
    qBarParityNext = !qBarParityNext;
    constexpr bool reorderForKCache = (useKVCache && inputElemSize == 2 && cacheElemSize == 1);
    if constexpr (reorderForKCache) {
      reorder16bQHeadsToMatch8bKCache<ctaShapeInWarps.x, qkSwizzle, true>(warpIdx.x, smem.q[warpIdx.y][0]);
      unused(qBar.arrive());
      qBar.wait_parity(qBarParityNext);
      qBarParityNext = !qBarParityNext;
      assertWarpConverged();
    }
#if CTA_ROW_MAX_BACKWARD_METHOD == 2
    ThrdRegRowMax initRowMax;
    initRowMax.fill(safeInitRowMax);
#endif
    for (uint32_t seqIter = seqIterInit; seqIter < nbSeqIters; seqIter += seqStrideIters) {
#if SHORT_SEQ_OPT
      if (ctaTile.x * seqIter + warpTile.x * warpIdx.x >= cacheSeqLen) {
        break;
      }
#endif
      auto runGemm0 = [&](auto elemK, uint32_t idxBeam) {
        assert(idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth));
        using KElemType = mha::decay_t<decltype(elemK)>;
        constexpr uint32_t elemsPerKHeadPart = exactDiv(kHeadPartBytes, sizeof(KElemType));
        constexpr uint32_t nbPartsPerKHead = exactDiv(headElems, elemsPerKHeadPart);
        // the accumulator
        WarpAcc acc{};
        constexpr uint32_t nbUnroll = (cacheElemSize == 2 ? nbPartsPerKHead : 1);
#pragma unroll(nbUnroll)
        for (uint32_t p = 0; p < nbPartsPerKHead; p++) {
          constexpr bool syncKTileEarly = (beamWidth > 1);  // alternative is to use double buffer for cacheIndir and pages
          if constexpr (syncKTileEarly) {
            // synchronize gemm0CacheIndir for the next loadKTilePart. the last loaded K tile is also
            // sync'ed at the same time.
            ldgsts::waitGroup<0>();
            __syncwarp();
          }
          // prefetch next part into shared memory
          uint32_t idxPartNext, idxBeamNext, nNextBias;
          mha::tie(idxPartNext, idxBeamNext, nNextBias) = isConvergedTile(seqIter)
                                                              ? carryLE<nbPartsPerKHead, 1U>(p + 1, idxBeam, 0U)
                                                              : carryLE<nbPartsPerKHead, beamWidth>(p + 1, idxBeam, 0U);

          loadKTilePart(seqIter + seqStrideIters * nNextBias, idxBeamNext, idxPartNext);
          ldgsts::commitGroup();
          // @fixme: do L2 cache prefetch for next iter tile if last part

          // q is already synchronized
          if constexpr (!syncKTileEarly) {
            // synchronize k
            ldgsts::waitGroup<1>();
          }
          const SharedMem::QSmemBuffer& smemQ = smem.q[warpIdx.y][0];
          constexpr uint32_t qOffsetPerPart = exactDiv(elemsPerKHeadPart, inputElemsPerGrain);
          const uint32_t smemQOffset = qOffsetPerPart * p;
          const SharedMem::KSmemBuffer& smemKPart = getSMemKTile(idxCurrSMemKBuf);
          // #ifndef NDEGBUG
          //                     for (uint32_t i = 0; i < exactDiv(smemKPart.rows * smemKPart.cols,
          //                     warp_size); i++) {
          //                         const uint32_t idx = warp_size * i + laneId();
          //                         const uint32_t r = idx / smemKPart.cols;
          //                         const uint32_t c = idx % smemKPart.cols;

          //                         assert(smemKPart(r, c) == );
          //                     }
          // #endif
          // do computation.
          smemQKPartGemm<KElemType>(warp, acc, smemQ, smemQOffset, smemKPart);
          idxCurrSMemKBuf++;
        }
        return acc;
      };
      WarpAcc acc;
      //@fixme: alternative is to use separate inner loop, which results in larger but maybe faster code.
      for (uint32_t idxBeam = 0; idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth); idxBeam++) {
        WarpAcc tmp;
        if constexpr (mha::is_same_v<CacheElem, InputElem>) {
          tmp = runGemm0(CacheElem{}, idxBeam);
        } else {
          tmp = runGemm0(CacheElem{}, idxBeam);
        }
        pickAccRowsForBeamSearch(
            warp, acc, tmp, isConvergedTile(seqIter), idxBeam, [](float& d, float s) { d = s; });
      }
      // apply qkScale
      rescaleAcc(warp, acc, qkScale);
#if CTA_ROW_MAX_BACKWARD_METHOD == 0
      QuadRegRowMax initRowMaxQuad;
      initRowMaxQuad.fill(safeInitRowMax);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 1
      // load hint
      xBar.consumed.wait_parity(getAndFlip(xBarConsumedParityNext));
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToRegForQuad<false>(warp);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 2
      QuadRegRowMax initRowMaxQuad = replicateForQuad(warp, initRowMax);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 3
      // load hint
      smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].wait_parity(xBarConsumedParityNext);
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToRegForQuad<false>(warp);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
      // load hint
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y].loadToRegForQuad<true>(warp);
#endif
      // masking
      const uint32_t warpTileTokenBeg = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
#if SPEC_DEC
      if (seqIter >= nbSeqItersWithoutMask) {
        const uint32_t nbValidCols = (warpTileTokenBeg < cacheSeqLen ? cacheSeqLen - warpTileTokenBeg : 0U);
        applyMaskFromInput(warp, acc, mask, idxHeadTokenInGrp, nbValidCols, qSeqLen, actualQSeqLen, headGrpSize
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                           ,
                           tok0WinBeg, seqIter, cacheSeqLen, warpTileTokenBeg
#endif
        );
      }
#else
      const bool isFirstIter = (seqIter == nbSkipLeadingTiles);
      const bool needMaskLeading = (rtIsReallySliding && isFirstIter);
      const bool isLastIter = (seqIter + 1 == nbSeqIters);
      const bool needMaskTrailing = isLastIter && cacheSeqLen % ctaTile.x != 0;
      if (needMaskLeading || needMaskTrailing) {
        const uint32_t validTokenBeg = (!needMaskLeading || nbTotalSkipTokens < warpTileTokenBeg)
                                           ? 0
                                           : nbTotalSkipTokens - warpTileTokenBeg;
        const uint32_t validTokenEnd = (warpTileTokenBeg < cacheSeqLen ? cacheSeqLen - warpTileTokenBeg : 0U);
        if (validTokenBeg > 0 || validTokenEnd < warpTile.x) {
          applyMask(warp, acc, validTokenBeg, validTokenEnd);
        }
      }
#endif

      // find max and update acc into exp(acc-max).
      const QuadRegRowMax regRowMax = warpTileOnlineSoftmax(warp, initRowMaxQuad, acc);

      // store result and max to shared memory.
      const GemmOutRegTile fp16Acc = toFp16(acc);
      const QuadRegRowMax regRowSum = computeRowSum(warp, fp16Acc);
#if CTA_ROW_MAX_BACKWARD_METHOD != 1
      xBar.consumed.wait_parity(getAndFlip(xBarConsumedParityNext));
#if CTA_ROW_MAX_BACKWARD_METHOD == 2
      initRowMax = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToReg<false>(warp);
#endif
#endif
      storeOrderedGemmOutTile(warp, smem.x[warpIdx.y][warpIdx.x], fp16Acc);
      smem.warpRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, regRowMax);
      smem.warpRowSum[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, regRowSum);
      unused(xBar.produced.arrive());
    }
#if defined(XQA_PAGED_INT4)
    ldgsts::waitGroup<0>();
    __syncthreads();
#endif
  } else {
    assert(warpIdx.z == 1);
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
    unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
#endif
    const uint32_t warpIdxInGrp = gemm1WarpIdxInGrp(warpIdx.x);  // @fixme: use BoundedVal
    const uint32_t warpGrpIdx = gemm1WarpGrpIdx(warpIdx.x);      // @fixme: use BoundedVal
    auto* const pWarpGrpBar = smem.warpGrpBar(warpGrpIdx);
    ParityOrNone<grpLoadV> warpGrpBarParityNext{};
#if BEAM_WIDTH > 1
    auto loadCacheIndir = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter, uint32_t idxBeam) mutable {
      const uint32_t seqOffset = ctaTile.x * seqIter + warpTile.x * nbXTilesPerXIter * xIter + cacheVTileSeqStride * vIter + cacheVTileSeqLen * warpGrpIdx;
      auto& dst = smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x];
      loadIndicesForBeamSearchAsync<grpLoadV ? gemm1WarpsPerGrp : 1U, cacheVTileSeqLen>(
          grpLoadV ? warpIdxInGrp : 0U, dst, beamSearchParams, idxReq, idxBeam, seqOffset, cacheSeqLen);
    };
    loadCacheIndir(seqIterInit, 0, 0, 0);
#endif
    unused(smem.xBarriers[warpIdx.y][warpIdx.x].consumed.arrive(gemm1WarpsPerGrp * nbWarpGrpsPerXTile));
    CircIdx<nbVBuffers> idxCurrSMemVBuf{nbVBuffers - 1};
    const auto getSmemVTile = [&](uint32_t idx) -> SharedMem::VSmemBuffer& { return smem.v[warpGrpIdx][grpLoadV ? 0 : warpIdxInGrp][idx]; };
    const auto getSmemVBar = [&](uint32_t idx) -> SharedMem::Barrier* { return smem.vBarrier(warpGrpIdx, idx); };
#if USE_PAGED_KV_CACHE
#if BEAM_WIDTH == 1
    VCachePageIndices pageIdx = VCachePageIndices::filled(kBAD_PAGE_INDEX);
#endif
    auto loadPages = [&](uint32_t idxPageBeg) mutable {
#if BEAM_WIDTH == 1
      const uint32_t idxBeam = 0;
      pageIdx = getPage<VCachePageIndices::size>(cacheList, false, idxReq, idxBeam, idxPageBeg, nbPages);
#else
      auto& dst = smem.vCachePages[grpLoadV ? warpGrpIdx : warpIdx.x];
      loadPagesForBeamSearchAsync<grpLoadV ? gemm1WarpsPerGrp : 1U>(
          grpLoadV ? warpIdxInGrp : 0U, dst, cacheList, false, idxReq, idxPageBeg, nbPages);
#endif
    };
    uint32_t idxPageBeg = nbPagesPerCtaTile * seqIterInit + cacheVTileSeqLen * warpGrpIdx / tokensPerPage;
    loadPages(idxPageBeg);
#else
    const uint32_t idxBeamBase = 0;
    const uint32_t cacheVBaseBatch = cacheList.capacity * nbKHeads * (idxBeamBase + beamWidth * idxReq);
    const uint32_t cacheVSeqBaseOffset = cacheList.isBSNH
                                             ? (cacheVBaseBatch + idxHeadGrp)
                                             : (cacheVBaseBatch + cacheList.capacity * idxHeadGrp);
#endif
    auto nextStep = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter, uint32_t idxBeam) {
      uint32_t vIterNext, isNextBeam;
      mha::tie(vIterNext, isNextBeam) = carryLE<nbVItersPerXIter>(vIter + 1, 0);

      uint32_t idxBeamNext, xIterNext, nNextBias;
      mha::tie(idxBeamNext, xIterNext, nNextBias) = isConvergedTile(seqIter)
                                                        ? carryLE<1, nbXItersPerCtaTile>(idxBeam + isNextBeam, xIter, 0)
                                                        : carryLE<beamWidth, nbXItersPerCtaTile>(idxBeam + isNextBeam, xIter, 0);

      const uint32_t seqIterNext = seqIter + seqStrideIters * nNextBias;
      return mha::tuple<uint32_t, uint32_t, uint32_t, uint32_t>(seqIterNext, xIterNext, vIterNext, idxBeamNext);
    };
    auto loadVTilePart = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter,
                             uint32_t idxBeam) mutable {  // @fixme: merge three iteration parameters into idxVTileGlb.
      assert(idxBeam < beamWidth);
      assert(seqIter % nbSubSeqPerSeq == seqIterInit % nbSubSeqPerSeq);
      const auto idxNextSMemVBuf = idxCurrSMemVBuf.next();
      auto& dst = getSmemVTile(idxNextSMemVBuf);
      const uint32_t dstHeadOffset = 0;
      constexpr bool vSwizzle = true;

      const uint32_t seqOffset = ctaTile.x * seqIter + warpTile.x * nbXTilesPerXIter * xIter + cacheVTileSeqStride * vIter + cacheVTileSeqLen * warpGrpIdx;
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
      const uint32_t idxHeadBeg = (seqOffset % tokensPerPage) * nbKHeads + idxHeadGrp;

#else
      const uint32_t idxHeadBeg = tokensPerPage * idxHeadGrp + seqOffset % tokensPerPage;
#endif
#if BEAM_WIDTH == 1
#if PAGED_KV_CACHE_LAYOUT == 1
      const HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> src{
          cacheList.vCacheVLLM, pageIdx, nbKHeads, idxHeadBeg};
#else
      const HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> src{
          cacheList.pool, pageIdx, nbKHeads, idxHeadBeg};
#endif
#else
      const IndexedHeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> src{
          /*indices=*/smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x].data,
#if PAGED_KV_CACHE_LAYOUT == 1
          /*pool=*/cacheList.vCacheVLLM,
#else
          /*pool=*/cacheList.pool,
#endif
          /*pageIndices=*/smem.vCachePages[grpLoadV ? warpGrpIdx : warpIdx.x].data,
          /*nbKHeads=*/nbKHeads,
          /*offset=*/idxHeadBeg};
#endif
#else
      const uint32_t idxHeadBeg = cacheList.isBSNH
                                      ? (cacheVSeqBaseOffset + seqOffset * nbKHeads)
                                      : (cacheVSeqBaseOffset + seqOffset);
#if BEAM_WIDTH == 1
      const TinyPtr<GMemCacheHead const> src{cacheList.vData, idxHeadBeg};
#else
      const IndexedHeadPtr<GMemCacheHead const, 0, 0> src{
          /*indices=*/smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x].data,
          /*pointer=*/cacheList.data,
          /*offset=*/idxHeadBeg,
          /*beamStride=*/cacheList.capacity * nbKHeads * 2};
#endif
#endif
      // if (threadIdx.x == dbgPrintTid) {
      //     printf("V: seqIter=%u, xIter=%u, idxBeam=%u, vIter=%u: pointers={%p, %p}, indices={", seqIter, xIter,
      //     idxBeam, vIter, src.pointers[0], src.pointers[1]); const uint32_t nbHeadsAvail = mha::min((seqOffset
      //     < cacheSeqLen ? cacheSeqLen - seqOffset : 0U), cacheVTileSeqLen); for (int i = 0; i < nbHeadsAvail;
      //     i++) {
      //         printf("%u, ", src.indices[i]);
      //     }
      //     printf("}\n");
      // }

#if GRP_LOAD_V
      const uint32_t nbHeadsAvail = (seqIter + 1 < nbSeqIters)
                                        ? cacheVTileSeqLen
                                        : (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                                   : 0U);  // may also be full but it can be handled correctly anyway
      copyHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, false>(
          warpIdxInGrp, dst, src, nbHeadsAvail);
#else
      const uint32_t nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                             : 0U);  // may also be full but it can be handled correctly anyway
      unused(nbHeadsAvail);
      const bool isFullTile = (seqIter + 1 < nbSeqIters);
      if (isFullTile) {
        copyPartialHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, true>(
            warp, dst, dstHeadOffset, src, warpIdxInGrp);
      } else {
        const uint32_t nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                               : 0U);  // may also be full but it can be handled correctly anyway
        copyPartialHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, false>(
            warp, dst, dstHeadOffset, src, warpIdxInGrp, mha::min(nbHeadsAvail, cacheVTileSeqLen));
      }
#endif

#if BEAM_WIDTH > 1
      // to make sure all threads has finished usage of cache indir and pages
      unused(arrive<grpLoadV>(pWarpGrpBar));
      wait_parity<grpLoadV>(pWarpGrpBar, getAndFlip<grpLoadV>(warpGrpBarParityNext));
#endif
#if USE_PAGED_KV_CACHE
      constexpr uint32_t xIterSeqStride = cacheVTileSeqStride * nbVItersPerXIter;
      // `if constexpr` inside a non-template function still type-checks the discarded branch, so
      // both divisors below must stay non-zero for every instantiation even though only one branch
      // is ever live (ORT builds XQA with -Werror all-warnings, which turns a constant-folded
      // "right operand of % is zero" in the dead branch into a build failure).
      constexpr uint32_t nbXItersPerPage =
          (xIterSeqStride <= tokensPerPage ? exactDiv(tokensPerPage, xIterSeqStride) : 1U);
      constexpr uint32_t nbPagesPerXIter =
          (xIterSeqStride <= tokensPerPage ? 1U : exactDiv(xIterSeqStride, tokensPerPage));
      if constexpr (xIterSeqStride <= tokensPerPage) {
        static_assert(nbXItersPerPage <= nbXItersPerCtaTile);
        if (xIter % nbXItersPerPage == nbXItersPerPage - 1 && vIter == nbVItersPerXIter - 1 && (idxBeam == beamWidth - 1 || isConvergedTile(seqIter))) {
          const auto step = 1;  // cacheVTileSeqLen * gemm1NbWarpGrps / tokensPerPage;
          idxPageBeg += (idxPageBeg % nbPagesPerCtaTile == nbPagesPerCtaTile - 1
                             ? nbPagesPerCtaTile * (nbSubSeqPerSeq - 1) + step
                             : step);
          assert(beamWidth == 1 || cacheVTileSeqStride <= tokensPerPage && "todo: need to substrate from idxPageBeg for beam switching");
          loadPages(idxPageBeg);
        }
      } else {
        assert(nbVItersPerXIter == 1);
        if ((idxBeam == beamWidth - 1 || isConvergedTile(seqIter)) && vIter == nbVItersPerXIter - 1) {
          const auto step = nbPagesPerXIter;
          idxPageBeg += (idxPageBeg % nbPagesPerCtaTile + step >= nbPagesPerCtaTile
                             ? nbPagesPerCtaTile * (nbSubSeqPerSeq - 1) + step
                             : step);
          loadPages(idxPageBeg);
        }
      }
#endif
#if BEAM_WIDTH > 1
      uint32_t seqIterNext, xIterNext, vIterNext, idxBeamNext;
      mha::tie(seqIterNext, xIterNext, vIterNext, idxBeamNext) = nextStep(seqIter, xIter, vIter, idxBeam);
      loadCacheIndir(seqIterNext, xIterNext, vIterNext, idxBeamNext);
#endif
    };
    auto commitVTileLoad = [&](uint32_t idxVBar) {
#if GRP_LOAD_V
      auto& bar = *getSmemVBar(idxVBar);
      ldgsts::barArrive(bar, true);
#else
      ldgsts::commitGroup();
#endif
    };
    auto syncVTileLoad = [&](uint32_t idxVBar, ParityOrNone<grpLoadV> parity, bool alreadyComplete) {
#if GRP_LOAD_V
      if (alreadyComplete) {
        return;
      }
      SharedMem::Barrier& bar = *getSmemVBar(idxVBar);
      bar.wait_parity(parity);
#else
      assert(!alreadyComplete);
      ldgsts::waitGroup<nbVBuffers - 1>();
#endif
    };
    auto testVTileLoad = [&](uint32_t idxVBar, ParityOrNone<grpLoadV> parity) { return test_wait_parity<grpLoadV>(getSmemVBar(idxVBar), parity); };

#if BEAM_WIDTH > 1
    // synchronize first page/cacheIndir loading to shared memory
    ldgsts::commitGroup();
    ldgsts::waitGroup<0>();
    unused(arrive<grpLoadV>(pWarpGrpBar));
    wait_parity<grpLoadV>(pWarpGrpBar, getAndFlip<grpLoadV>(warpGrpBarParityNext));
#endif

    loadVTilePart(seqIterInit, 0, 0, 0);
    commitVTileLoad(idxCurrSMemVBuf.next());
    idxCurrSMemVBuf++;
    ParityOrNone<grpLoadV> vBarParity{};
    // @fixme: do prefetch for next iter tile if last part

    ThrdRegRowMax globalRowMax;
    globalRowMax.fill(SAFE_INIT_ROW_MAX);
    ThrdRegRowMax globalRowSum;
    globalRowSum.fill(0);
    // the accumulator
    WarpAcc acc{};
    if (grpLoadV) {
      unused(pWarpGrpBar->arrive());
    }
    bool xBarProducedParityNext = false;
    for (uint32_t seqIter = seqIterInit; seqIter < nbSeqIters; seqIter += seqStrideIters) {
#pragma unroll
      for (uint32_t xIter = 0; xIter < nbXItersPerCtaTile; xIter++) {
        const uint32_t idxXTile = xIter * nbXTilesPerXIter + warpGrpIdx / nbCacheVTilesPerXTile;
        assert(idxXTile < ctaShapeInWarps.x);
#if SHORT_SEQ_OPT
        if (ctaTile.x * seqIter + warpTile.x * idxXTile >= cacheSeqLen) {
          break;
        }
#endif
        const auto& smemXTile = smem.x[warpIdx.y][idxXTile];
        auto& xBar = smem.xBarriers[warpIdx.y][idxXTile];
        ThrdRegRowMax xRowScales;
        UniformRescaleMask xRowNeedRescaleMask;  // expect storage in UR
        bool skipXRowRescale;
        for (uint32_t idxBeam = 0; idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth); idxBeam++) {
#pragma unroll
          for (uint32_t vIter = 0; vIter < nbVItersPerXIter; vIter++) {
            const bool vTestConsumed = test_wait_parity<grpLoadV>(pWarpGrpBar, warpGrpBarParityNext);
            constexpr bool syncVTileEarly = (beamWidth > 1);  // alternative is to use double buffer for cacheIndir and pages
            bool vTestProduced = syncVTileEarly && testVTileLoad(idxCurrSMemVBuf, vBarParity);
            auto isLastVBuf = [&] { return (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1); };
            unused(isLastVBuf);
            const uint32_t idxVTileInsideXIter = gemm1NbWarpGrps * vIter + warpGrpIdx;
            const uint32_t idxVTile = idxVTileInsideXIter % nbCacheVTilesPerXTile;  // inside XTile.
            assert(idxVTile < nbCacheVTilesPerXTile);
            uint32_t nNext, xIterNext, vIterNext, idxBeamNext;
            mha::tie(nNext, xIterNext, vIterNext, idxBeamNext) = nextStep(seqIter, xIter, vIter, idxBeam);
            if constexpr (syncVTileEarly) {
              // sync early to make sure that cacheIndir and pages has been loaded. The last loaded V tile
              // is also sync'ed at the same time.
              syncVTileLoad(idxCurrSMemVBuf, vBarParity, vTestProduced);
              if (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1) {
                flip<grpLoadV>(vBarParity);
              }
            }
            if (!vTestConsumed) {
              wait_parity<grpLoadV>(pWarpGrpBar, warpGrpBarParityNext);
            }
            flip<grpLoadV>(warpGrpBarParityNext);
            loadVTilePart(nNext, xIterNext, vIterNext, idxBeamNext);
            commitVTileLoad(idxCurrSMemVBuf.next());
            // @fixme: do L2 cache prefetch for next iter tile

            if constexpr (!syncVTileEarly) {
              vTestProduced = testVTileLoad(idxCurrSMemVBuf, vBarParity);
            }

            if (idxBeam == 0 && vIter == 0) {
              xBar.produced.wait_parity(xBarProducedParityNext);
              const auto& smemRowMax = smem.warpRowMax[warpIdx.y][idxXTile];
              const auto& smemRowSum = smem.warpRowSum[warpIdx.y][idxXTile];
              // update globalRowMax
              ThrdRegRowMax xTileRowMax;
              ThrdRegRowMax xTileRowSum;
              UniformRescaleMask needRescaleMask;
#pragma unroll
              for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
                xTileRowMax[i] = smemRowMax[warp_size * i + laneId()];
                xTileRowSum[i] = smemRowSum[warp_size * i + laneId()];
                assert(__ballot_sync(~0U, laneId() == 0) == 1U);
                assert(__ballot_sync(~0U, laneId() == 0) == 1U);
                needRescaleMask[i] = __ballot_sync(~0U, xTileRowMax[i] != globalRowMax[i]);
              }
              const bool skipAllRescale = !any(needRescaleMask);
              if (skipAllRescale) {
                skipXRowRescale = true;
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
                if (idxXTile == warpIdx.x) {
                  unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
                }
#endif
              } else {
                const ThrdRegRowMax globalRowMaxOld = globalRowMax;
                UniformRescaleMask accRowNeedRescaleMask;
#pragma unroll
                for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
                  accRowNeedRescaleMask[i] = __ballot_sync(~0U, xTileRowMax[i] > globalRowMaxOld[i]);
                  xRowNeedRescaleMask[i] = (needRescaleMask[i] & ~accRowNeedRescaleMask[i]);
                  assert(xRowNeedRescaleMask[i] == __ballot_sync(~0U, xTileRowMax[i] < globalRowMaxOld[i]));
                  globalRowMax[i] = fmaxf(globalRowMaxOld[i], xTileRowMax[i]);
                }
                skipXRowRescale = !any(xRowNeedRescaleMask);

#if CTA_ROW_MAX_BACKWARD_METHOD == 1 || CTA_ROW_MAX_BACKWARD_METHOD == 2 || CTA_ROW_MAX_BACKWARD_METHOD == 3
                // update smem.ctaRowMax.
                if (idxXTile == warpIdx.x) {
                  smem.ctaRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, globalRowMax);
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
                  unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
#endif
                }
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
                // update smem.ctaRowMax.
                // smem.ctaRowMax[warpIdx.y].storeFromReg<true>(warp, globalRowMax);
                smem.ctaRowMax[warpIdx.y].atomicMaxUpdate(warp, globalRowMax);
#endif
                // update row sum and acc
                if (!enableMicroFastPath || any(accRowNeedRescaleMask)) {
                  const ThrdRegRowMax accRowScales = expf(globalRowMaxOld - globalRowMax);
                  globalRowSum = globalRowSum * accRowScales;
                  // @fixme: when tmpAcc is used, this can be delayed.
                  rescaleAcc(warp, acc, accRowNeedRescaleMask, accRowScales);
                }
                if (!enableMicroFastPath || !skipXRowRescale) {
                  xRowScales = skipXRowRescale ? xRowScales : expf(xTileRowMax - globalRowMax);
                  xTileRowSum = skipXRowRescale ? xTileRowSum : xTileRowSum * xRowScales;
                }
              }
              globalRowSum = globalRowSum + xTileRowSum;
            }
            if constexpr (!syncVTileEarly) {
              syncVTileLoad(idxCurrSMemVBuf, vBarParity, vTestProduced);
              if (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1) {
                flip<grpLoadV>(vBarParity);
              }
            }
            const auto& smemVTile = getSmemVTile(idxCurrSMemVBuf);
            // do computation from shared memory X and V tiles
#if BEAM_WIDTH == 1
            smemXVPartGemm<CacheElem>(warp, acc, skipXRowRescale, xRowNeedRescaleMask, xRowScales,
                                      smemXTile, idxVTile, smemVTile, grpLoadV ? warpIdxInGrp : 0);
#else
            WarpAcc tmpAcc{};
            smemXVPartGemm<CacheElem>(warp, tmpAcc, skipXRowRescale, xRowNeedRescaleMask, xRowScales,
                                      smemXTile, idxVTile, smemVTile, grpLoadV ? warpIdxInGrp : 0);
            pickAccRowsForBeamSearch(
                warp, acc, tmpAcc, isConvergedTile(seqIter), idxBeam, [](float& d, float s) { d += s; });
#endif
            if (grpLoadV) {
              unused(pWarpGrpBar->arrive());
            }
            idxCurrSMemVBuf++;
          }
        }  // idxBeam
        xBar.consumed.arrive();
      }  // xIter
      flip(xBarProducedParityNext);
    }  // seqIter

    const auto fullRescaleMask = UniformRescaleMask::filled(~0U);

    constexpr bool needMergeGlobal = (gemm1NbWarpGrps > 1 && nbXTilesPerXIter > 1);
    if constexpr (needMergeGlobal) {
      assert(gemm1NbWarpGrps != 1);
      __syncthreads();
      smem.warpRowMax[warpIdx.y][warpIdx.x].template storeFromReg<false>(warp, globalRowMax);
      smem.warpRowSum[warpIdx.y][warpIdx.x].template storeFromReg<false>(warp, globalRowSum);
      __syncthreads();
      for (uint32_t i = 1; i < nbXTilesPerXIter; i++) {  // i = 0 is for self and we can skip
        static_assert(nbXTilesPerXIter * nbWarpGrpsPerXTile == gemm1NbWarpGrps);
        const uint32_t otherWarpGrpIdx = (warpGrpIdx + nbWarpGrpsPerXTile * i) % gemm1NbWarpGrps;
        const uint32_t otherWarpIdx = warpIdxInGrp + gemm1WarpsPerGrp * otherWarpGrpIdx;
#ifndef NDEBUG
        {
          const auto v1 = smem.warpRowMax[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
          const auto v2 = smem.warpRowMax[warpIdx.y][otherWarpIdx - warpIdxInGrp].template loadToReg<false>(warp);
#pragma unroll
          for (uint32_t k = 0; k < ThrdRegRowMax::size; k++) {
            assert(__float_as_int(v1[k]) == __float_as_int(v2[k]));
          }
        }
#endif
        const auto otherRowMax = smem.warpRowMax[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
        const auto otherRowSum = smem.warpRowSum[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
        const auto globalRowMaxNew = fmaxf(globalRowMax, otherRowMax);
        const auto scaleForThis = expf(globalRowMax - globalRowMaxNew);
        const auto scaleForOther = expf(otherRowMax - globalRowMaxNew);
        rescaleAcc(warp, acc, fullRescaleMask, scaleForThis);
        globalRowSum = globalRowSum * scaleForThis + otherRowSum * scaleForOther;
        globalRowMax = globalRowMaxNew;
      }
    }

    // A null vCacheScale means the caller rescales the output itself (per-channel V scale).
    float voScale = ((hasScalarCacheScale && vCacheScale != nullptr) ? vCacheScale[0] : 1.F);
    if (seqIterInit < nbSeqIters) {  // otherwise rcpRowSum will be NAN.
      // The attention sinks are moved to the multi-block reduction part if the multi-block is enabled.
      if (!isMultiBlock && attentionSinks != nullptr) {
        // Attention sinks are per head.
        addAttentionSinks(globalRowSum, globalRowMax, attentionSinks + headGrpSize * idxHeadGrp
#if SPEC_DEC
                          ,
                          idxHeadTokenInGrp, nbValidHeadTokens
#endif
        );
      }
      const ThrdRegRowMax rcpRowSum = __frcp_rn(globalRowSum);
#if LOW_PREC_OUTPUT
      voScale *= rcpOutScale[0];
#endif
      rescaleAcc(warp, acc, fullRescaleMask, rcpRowSum * ThrdRegRowMax::filled(voScale));
    }
    const GemmOutRegTile outTile = toFp16(acc);

#if defined(XQA_PAGED_INT4)
    ldgsts::waitGroup<0>();
    __syncwarp();
#endif
    auto mergeAndSaveOutTile = [&](const GemmOutRegTile& tile, bool reorder) {
      if constexpr (gemm1NbWarpGrps == 1) {
        // swizzle in shared memory and write output global memory
        auto& outSwizzleBuffer = smem.x[warpIdx.y][warpIdx.x];
        __syncthreads();
        storeGemmOutTile(warp, outSwizzleBuffer, tile, reorder);
        __syncwarp();
        return &outSwizzleBuffer;
      } else {
        __syncthreads();
        // store to shared memory, then merge groups.
        using PostProcSMem = SharedMem::XSmemBuffer[ctaShapeInWarps.y][gemm1WarpsPerGrp][gemm1NbWarpGrps];
        static_assert(sizeof(PostProcSMem) <= smemSize);
        SharedMem::XSmemBuffer(&postSMem)[gemm1NbWarpGrps] = reinterpret_cast<PostProcSMem&>(smem)[warpIdx.y][warpIdxInGrp];
        storeGemmOutTile(warp, postSMem[warpGrpIdx], tile, reorder);
        __syncthreads();
        smemFp16ArraySum<true, gemm1NbWarpGrps, gemm1NbWarpGrps>(warpGrpIdx, postSMem[0], postSMem);
        __syncthreads();
        return &postSMem[0];
      }
    };

    // merge results from different warp groups
    SharedMem::XSmemBuffer* smemOutTile = mergeAndSaveOutTile(outTile, inputElemSize == 2 && cacheElemSize == 1);
    if (isMultiBlock) {
      static_assert(ctaShapeInWarps.y == 1, "not implemented");
#if SPEC_DEC
      // Includes both kHeads and qTokens.
      const uint32_t nbIndepHeadTokens = gridDim.y;
      const uint32_t indepHeadTokenIdx = blockIdx.y;
      const uint32_t nbSeq = nbIndepHeadTokens * batchSize;
#else
      const uint32_t nbSeq = nbKHeads * batchSize;
#endif
      const uint32_t nbSubSeq = nbSubSeqPerSeq * nbSeq;
      MemSegmenter<false> segmenter{scratch};

#if SPEC_DEC
      const uint32_t idxSeq = nbIndepHeadTokens * idxReq + indepHeadTokenIdx;
#else
      const uint32_t idxSeq = nbKHeads * idxReq + idxHeadGrp;
#endif
      const uint32_t idxBufBase = nbSubSeqPerSeq * idxSeq;
      const uint32_t idxBuf = idxBufBase + idxSubSeqInSeq;
      // copy row max/sum
      const TinyPtr<SMemWarpRowMax> rowMaxBuffers = segmenter.newSeg<SMemWarpRowMax>(nbSubSeq);
      const TinyPtr<SMemWarpRowMax> rowSumBuffers = segmenter.newSeg<SMemWarpRowMax>(nbSubSeq);
      if (warpGrpIdx == 0 && warpIdxInGrp == 0) {
        rowMaxBuffers[idxBuf].storeFromReg<false>(warp, globalRowMax);
        rowSumBuffers[idxBuf].storeFromReg<false>(warp, globalRowSum);
      }
      using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
      const TinyPtr<Vec<ScratchBuf, gemm1WarpsPerGrp>> scratchBuffers = segmenter.newSeg<Vec<ScratchBuf, gemm1WarpsPerGrp>>(nbSubSeq);
      // copy output to scratch
      copyGrains<false, nbValidRows * ScratchBuf::cols, gemm1NbWarpGrps>(
          warpGrpIdx, &scratchBuffers[idxBuf][warpIdxInGrp](0, 0), &(*smemOutTile)(0, 0));
      __syncthreads();
      constexpr uint32_t nbTileBuffers = 2;

      struct MultiBlockSMem {
        bool isLastCta;

        struct MBBuf {
          SMemWarpRowMax rowMax;
          SMemWarpRowMax rowSum;
          SharedMem::XSmemBuffer tiles[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax tileRowMax[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax tileRowSums[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax mergedRowSum[gemm1NbWarpGrps];
        };

        MBBuf storage[ctaShapeInWarps.y];
      };

      static_assert(sizeof(MultiBlockSMem) <= smemSize);
      MultiBlockSMem& mbsmem = reinterpret_cast<MultiBlockSMem&>(smem);
      // increase the semaphore by 1
      if (warpIdx.y == 0 && warpGrpIdx == 0 && warpIdxInGrp == 0 && laneId() == 0) {
        uint32_t old;
        const uint32_t lastOld = nbSubSeqPerSeq - 1;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                     : "=r"(old)
                     : "l"(&semaphores[idxSeq]), "r"(lastOld));
        assert(old < nbSubSeqPerSeq);
        mbsmem.isLastCta = (old == lastOld);
      }
      __syncthreads();

      // merge if we are the last CTA.
      const bool isLastCta = mbsmem.isLastCta;
#if defined(XQA_PAGED_INT4)
      __syncthreads();
#endif
      if (isLastCta) {
        MultiBlockSMem::MBBuf& mbbuf = mbsmem.storage[warpIdx.y];
        SMemWarpRowMax& smemRowMax = reinterpret_cast<SMemWarpRowMax&>(smem);
        // get row max.
        if (warpIdx.x == 0) {
          const ThrdRegRowMax mergedRowMax = mergeRowMax<8>(warp, rowMaxBuffers + idxBufBase, nbSubSeqPerSeq);
          smemRowMax.storeFromReg<false>(warp, mergedRowMax);
        }
        __syncthreads();
        const ThrdRegRowMax mergedRowMax = smemRowMax.loadToReg<false>(warp);

        // rescale and accumulate
        auto getTileBuf = [&](auto& buffers, uint32_t d) -> decltype(buffers[0][0][0])& { return buffers[warpGrpIdx][warpIdxInGrp][d]; };
        auto loadBufAsync = [&](uint32_t n) {
#if defined(XQA_PAGED_INT4)
          __syncwarp();
#endif
          const uint32_t d = n / gemm1NbWarpGrps % nbTileBuffers;
          SharedMem::XSmemBuffer& dstTile = getTileBuf(mbbuf.tiles, d);
          SMemWarpRowMax& dstRowSum = getTileBuf(mbbuf.tileRowSums, d);
          SMemWarpRowMax& dstRowMax = getTileBuf(mbbuf.tileRowMax, d);
          copyGrains<true, sizeof(ScratchBuf) / grainBytes, 1, true>(
              0, &dstTile(0, 0), &scratchBuffers[idxBufBase + n][warpIdxInGrp](0, 0));
          constexpr uint32_t nbGrainsPerRowMaxBuf = exactDiv(sizeof(SMemWarpRowMax), grainBytes);
          copyGrains<true, roundUp(nbGrainsPerRowMaxBuf, 32u), 1, nbGrainsPerRowMaxBuf % 32 == 0>(0,
                                                                                                  reinterpret_cast<LdGrain*>(&dstRowSum),
                                                                                                  reinterpret_cast<const LdGrain*>(&rowSumBuffers[idxBufBase + n]), nbGrainsPerRowMaxBuf);
          copyGrains<true, roundUp(nbGrainsPerRowMaxBuf, 32u), 1, nbGrainsPerRowMaxBuf % 32 == 0>(0,
                                                                                                  reinterpret_cast<LdGrain*>(&dstRowMax),
                                                                                                  reinterpret_cast<const LdGrain*>(&rowMaxBuffers[idxBufBase + n]), nbGrainsPerRowMaxBuf);
        };
        loadBufAsync(warpGrpIdx);
        ldgsts::commitGroup();
        WarpAcc sumAcc{};
        ThrdRegRowMax partialMergedRowSum{};
        for (uint32_t n = warpGrpIdx; n < nbSubSeqPerSeq; n += gemm1NbWarpGrps) {
          if (n + gemm1NbWarpGrps < nbSubSeqPerSeq) {
            loadBufAsync(n + gemm1NbWarpGrps);
          }
          ldgsts::commitGroup();
          ldgsts::waitGroup<1>();
#if defined(XQA_PAGED_INT4)
          __syncwarp();
#endif
          const uint32_t d = n / gemm1NbWarpGrps % nbTileBuffers;
          WarpAcc tile = toWarpAcc(loadGemmOutTile(warp, mbbuf.tiles[warpGrpIdx][warpIdxInGrp][d]));
          const ThrdRegRowMax tileRowMax = getTileBuf(mbbuf.tileRowMax, d).loadToReg<false>(warp);
          const ThrdRegRowMax tileRowSum = getTileBuf(mbbuf.tileRowSums, d).loadToReg<false>(warp);
          const ThrdRegRowMax tileRowScales = expf(tileRowMax - mergedRowMax);
          const ThrdRegRowMax scaledTileRowSum = tileRowSum * tileRowScales;
          partialMergedRowSum = partialMergedRowSum + scaledTileRowSum;
          assert(isfinite(partialMergedRowSum[0]));
          rescaleAcc(warp, tile, fullRescaleMask, scaledTileRowSum);
          sumAcc = sumAcc + tile;
        }

        ThrdRegRowMax mergedRowSum{};
        if (gemm1NbWarpGrps == 1) {
          mergedRowSum = partialMergedRowSum;
        } else {
          if (warpIdxInGrp == 0) {
            mbbuf.mergedRowSum[warpGrpIdx].storeFromReg<false>(warp, partialMergedRowSum);
          }
          __syncthreads();
#ifndef NDEBUG
#pragma unroll
          for (uint32_t k = 0; k < ThrdRegRowMax::size; k++) {
            assert(__float_as_int(mbbuf.mergedRowSum[warpGrpIdx].loadToReg<false>(warp)[k]) == __float_as_int(partialMergedRowSum[k]));
          }
          __syncthreads();
#endif
#pragma unroll
          for (uint32_t i = 0; i < gemm1NbWarpGrps; i++) {
            mergedRowSum = mergedRowSum + mbbuf.mergedRowSum[i].loadToReg<false>(warp);
            assert(isfinite(mergedRowSum[0]));
          }
        }
        if (attentionSinks != nullptr) {
          // Attention sinks are per head.
          addAttentionSinks(mergedRowSum, mergedRowMax, attentionSinks + headGrpSize * idxHeadGrp
#if SPEC_DEC
                            ,
                            idxHeadTokenInGrp, nbValidHeadTokens
#endif
          );
        }
        __syncthreads();
        rescaleAcc(warp, sumAcc, fullRescaleMask, __frcp_rn(mergedRowSum));
        const GemmOutRegTile mergedOutTile = toFp16(sumAcc);
        smemOutTile = mergeAndSaveOutTile(mergedOutTile, false);
      }
    }
    if (warpGrpIdx == 0) {
#if SPEC_DEC
      copyOutputToGlobalMem(warp, &output[reqSeqOffset * nbQHeads], nbQHeads, headGrpSize,
                            (idxHeadGrp * headGrpSize), nbValidHeadTokens, actualQSeqLen,
                            uint2{warpTile.x * warpIdxInGrp, nbValidRows * warpIdx.y + idxHeadTokenInGrp}, *smemOutTile);
#else
      copyOutputToGlobalMem(warp, &output[nbQHeads * beamWidth * idxReq], nbQHeads, idxHeadGrp,
                            uint2{warpTile.x * warpIdxInGrp, nbValidRows * warpIdx.y}, *smemOutTile);
#endif
    }
  }
}

#if SPEC_DEC
#if __CUDA_ARCH__ == 900 && M_TILESIZE == 16
constexpr uint32_t nbCtaPerSM = 2;
#else
constexpr uint32_t nbCtaPerSM = 1;
#endif
#else
#if __CUDA_ARCH__ == 900
constexpr uint32_t nbCtaPerSM = 2;
#else
constexpr uint32_t nbCtaPerSM = 1;
#endif
#endif

[[maybe_unused]] CUBIN_EXPORT __device__ constexpr XQAKernelType kernelType = XQAKernelType::kAMPERE_WARP_SPECIALIZED;

#ifdef NDEBUG
CUBIN_EXPORT __global__ __launch_bounds__(256, nbCtaPerSM) void kernel_mha(
#if SPEC_DEC
    const uint32_t qSeqLen, const uint32_t nbKHeads, const uint32_t headGrpSize, const SeqLenDataType* qCuSeqLens,
#else
    const uint32_t nbKHeads,
#endif
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale,
    OutputHead* const __restrict__ output,  // [nbReq][beamWidth][nbQHeads]
#if LOW_PREC_OUTPUT
    const float* rcpOutScale,
#endif
    const IOHead* const __restrict__ q,  // [nbReq][beamWidth][nbQHeads],
#if SPEC_DEC
    const MaskType* __restrict__ mask,  // [qSeqLen, divUp(qSeqLen, 32))] uint2 (each bit represents mask for one col
                                        // position).
#endif
    const float* attentionSinks,  // [headGrpSize]
    const KVCacheList<usePagedKVCache> cacheList,
#if BEAM_WIDTH > 1
    const BeamSearchParams beamSearchParams,
#endif
    const uint32_t batchSize,
    // Device memory scalars for quantized KV cache. See kernel_mha_impl.
    const float* __restrict__ kCacheScale,
    const float* __restrict__ vCacheScale,
    uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr) {
#if SPEC_DEC
  kernel_mha_impl(qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
  kernel_mha_impl(nbKHeads,
#endif
#if SLIDING_WINDOW
                  slidingWinSize,
#endif
                  qScale, output,
#if LOW_PREC_OUTPUT
                  rcpOutScale,
#endif
                  q,
#if SPEC_DEC
                  mask,
#endif
                  attentionSinks, cacheList,
#if BEAM_WIDTH > 1
                  beamSearchParams,
#endif
                  batchSize, kCacheScale, vCacheScale, semaphores, scratch);
}
#else
static constexpr auto kernel_mha = kernel_mha_impl;
#endif

#ifndef GENERATE_CUBIN
uint32_t computeNbSubSeqPerSeqMHA(const cudaDeviceProp& prop, uint32_t batchSize, uint32_t nbKHeads, uint32_t maxSeqLen) {
  if (!allowMultiBlockMode) {
    return 1;
  }
  const auto env = std::getenv("XQA_NB_SUB_SEQ");
  if (env != nullptr) {
    const int32_t val = std::stoi(env);
    if (val > 0) {
      return val;
    }
  }
  return std::min<uint32_t>(
      std::max<uint32_t>(1U, prop.multiProcessorCount / (batchSize * nbKHeads)), divUp(maxSeqLen, ctaTile.x));
}

void launchMHA(const cudaDeviceProp& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
               uint32_t slidingWinSize,
#endif
               float qScale, OutputHead* output,
#if LOW_PREC_OUTPUT
               const float* rcpOutScale,
#endif
#if USE_INPUT_KV
               const InputHead* qkv,
#if ROPE_STYLE != 0
               const Vec<float, validElemsPerHead>* ropeCosSin,
#endif
#else
               const InputHead* q,
#endif
               const float* attentionSinks,  // [headGrpSize]
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
               GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#else
               GMemCacheHead* pool,  // global pool of pages
#endif
               const KVCachePageIndex*
                   kvCachePageList,  // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
#else
               GMemKVCacheHead* kCacheData,
               GMemKVCacheHead* vCacheData,
               bool isBSNH,
#endif
               uint32_t maxSeqLen, const uint32_t* seqLen,
#if BEAM_WIDTH > 1
               const BeamSearchParams& beamSearchParams,
#endif
               uint32_t batchSize,
               // Device memory scalars for quantized KV cache. K and V may have different scales;
               // each is either a per-tensor scale or a normalizer for a folded per-channel scale.
               const float* __restrict__ kCacheScale,
               const float* __restrict__ vCacheScale,
#if SPEC_DEC
               const SpecDecParams& specDecParams,
#endif
#if SKIP_SOFTMAX_ATTN
               const float skipSoftmaxThresholdScaleFactor,  // for compatibility with mha_sm90.cu only
#if SKIP_SOFTMAX_ATTN_BLOCK_STATS
               uint32_t* __restrict__ skippedBlockCount,  // for compatibility with mha_sm90.cu only
               uint32_t* __restrict__ totalBlockCount,    // for compatibility with mha_sm90.cu only
#endif
#endif
               uint32_t* semaphores, void* scratch, cudaStream_t stream) {
#if SPEC_DEC
  const auto qSeqLen = specDecParams.qSeqLen;
  const auto qCuSeqLens = specDecParams.qCuSeqLens;
  const auto mask = specDecParams.mask;
#endif
#if USE_INPUT_KV
  throw std::runtime_error("not implemented");
#else
  static const uint32_t hostSmemSize = [&]() {
    uint32_t size;
    checkCuda(cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)));
    // Defensive backstop: the kernel's shared-memory footprint is fixed at compile time for its
    // target SM (sm_80/sm_90 use a larger K/V-tile layout than sm_86/sm_89/sm_120). When such a
    // kernel is JIT-compiled from PTX onto a device with a smaller per-block opt-in limit (e.g.
    // consumer Blackwell sm_120, ~99 KB), cudaFuncSetAttribute below returns cudaErrorInvalidValue.
    // The GQA dispatcher already skips XQA in that case (see GetXQARequiredSharedMemoryBytes); this
    // guard turns any remaining mismatch into an actionable message instead of an opaque CUDA error.
    if (size > prop.sharedMemPerBlockOptin) {
      throw std::runtime_error(
          "XQA kernel requires " + std::to_string(size) +
          " bytes of shared memory per block but this GPU allows only " +
          std::to_string(prop.sharedMemPerBlockOptin) +
          " bytes. Build ONNX Runtime with the device's native CUDA architecture (e.g. add 120 to "
          "CMAKE_CUDA_ARCHITECTURES for sm_120 / RTX 50-series) or disable XQA (ORT_ENABLE_XQA=0).");
    }
    checkCuda(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
    return size;
  }();
  const uint32_t nbVHeads = nbKHeads;
  unused(nbVHeads);
  const uint32_t nbQHeads = nbKHeads * headGrpSize;
  unused(nbQHeads);

  // const uint32_t nbSubSeqPerSeq = allowMultiBlockMode ? DBG_NB_CTAS_PER_SEQ : 1;
  const uint32_t nbSubSeqPerSeq = computeNbSubSeqPerSeqMHA(prop, batchSize, nbKHeads, maxSeqLen);
  // printf("DEBUG: launchMHA: batch=%u, nbKHeads=%u, maxSeq=%u, nbSubSeqPerSeq=%u\n", batchSize, nbKHeads, maxSeqLen, nbSubSeqPerSeq);
  // gridDim.z == batchSize && gridDim.y == nbKHeads && gridDim.x == nbSubSeqPerSeq
#if SPEC_DEC
  const uint32_t nbTokenBlocksPerGrp = divUp(qSeqLen * headGrpSize, rowsPerBlock);
  const dim3 dimGrid{nbSubSeqPerSeq, nbKHeads * nbTokenBlocksPerGrp, batchSize};
#else
  const dim3 dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize};
#endif
  const dim3 dimCta{warp_size * ctaShapeInWarps.x, ctaShapeInWarps.y, ctaShapeInWarps.z};
#if defined(NDEBUG) || USE_PAGED_KV_CACHE
  const auto launchCfg = makeLaunchConfig(dimGrid, dimCta, hostSmemSize, stream, ENABLE_PDL != 0);
#endif
#if USE_PAGED_KV_CACHE
  const uint32_t maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
#if PAGED_KV_CACHE_LAYOUT == 1
  const KVCacheList<true> cacheList{kCacheVLLM, vCacheVLLM, kvCachePageList, seqLen, maxNbPagesPerSeq, 1};
#else
  const KVCacheList<true> cacheList{pool, kvCachePageList, seqLen, maxNbPagesPerSeq, 1};
#endif
  cudaLaunchKernelEx(&launchCfg, kernel_mha,
#if SPEC_DEC
                     qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
                     nbKHeads,
#endif
#if SLIDING_WINDOW
                     slidingWinSize,
#endif
                     qScale, output,
#if LOW_PREC_OUTPUT
                     rcpOutScale,
#endif
                     q,
#if SPEC_DEC
                     mask,
#endif
                     attentionSinks, cacheList,
#if BEAM_WIDTH > 1
                     beamSearchParams,
#endif
                     batchSize, kCacheScale, vCacheScale, semaphores, scratch);
#else
  const KVCacheList<false> cacheList{kCacheData, vCacheData, seqLen, maxSeqLen, isBSNH, 1};
#ifndef NDEBUG
  kernel_mha<<<dimGrid, dimCta, hostSmemSize, stream>>>(
#else
  cudaLaunchKernelEx(&launchCfg, kernel_mha,
#endif
#if SPEC_DEC
      qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
                     nbKHeads,
#endif
#if SLIDING_WINDOW
      slidingWinSize,
#endif
      qScale, output,
#if LOW_PREC_OUTPUT
      rcpOutScale,
#endif
      q,
#if SPEC_DEC
      mask,
#endif
      attentionSinks, cacheList,
#if BEAM_WIDTH > 1
      beamSearchParams,
#endif
      batchSize, kCacheScale, vCacheScale, semaphores, scratch);
#endif
  checkCuda(cudaPeekAtLastError());
#endif  // USE_INPUT_KV
}
#endif
#endif

__device__ __host__ inline size_t GetScratchSize(uint32_t nbSeq, uint32_t nbSubSeqPerSeq) {
  const uint32_t nbSubSeq = nbSubSeqPerSeq * nbSeq;
  size_t offset = 0;

  // 1. rowMax
  offset = roundUp<size_t>(offset, sizeof(SMemWarpRowMax));
  offset += sizeof(SMemWarpRowMax) * nbSubSeq;

  // 2. rowSum
  offset = roundUp<size_t>(offset, sizeof(SMemWarpRowMax));
  offset += sizeof(SMemWarpRowMax) * nbSubSeq;

  // 3. scratchBuffers
  using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
  using VecT = Vec<ScratchBuf, gemm1WarpsPerGrp>;

  // size_t sem_size = roundUp<size_t>(nbSeq * sizeof(uint32_t), 128);
  // if (nbSubSeqPerSeq > 1) {
  //   printf("[MHA_IMPL] GetScratchSize: nbSeq=%u, nbSubSeqPerSeq=%u, sizeof(SMemWarpRowMax)=%zu, sizeof(VecT)=%zu, nbValidRows=%u, XS_cols=%u\n",
  //           nbSeq, nbSubSeqPerSeq, (size_t)sizeof(SMemWarpRowMax), (size_t)sizeof(VecT), (uint32_t)nbValidRows, (uint32_t)SharedMem::XSmemBuffer::cols);
  // }

  offset = roundUp<size_t>(offset, sizeof(VecT));
  offset += sizeof(VecT) * nbSubSeq;

  return offset;
}
