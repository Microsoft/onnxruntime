// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"
#include "contrib_ops/cuda/math/matmul_block_scaled_fp8_tiling.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

// cuda_fp8.h only ships with CUDA 11.8+. Guard the include so older toolkits (or
// DISABLE_FLOAT8_TYPES builds) still compile. CUDA_VERSION is provided by <cuda.h>,
// which is pulled in by common.cuh above.
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

namespace {

// to_float / from_float / to_float2 / Vec2 come from core/providers/cuda/cu_inc/cuda_type_helper.cuh.

// Dequantizes FP8 E4M3 weights with per-block FP32 scales into FP16/BF16.
// b_fp8 is [N, K] row-major FP8 E4M3, weight_scale is [N, k_blocks] fp32. Scalar per-element
// fallback used when K is not a multiple of 16 (the vectorized kernel below handles K % 16 == 0).
template <typename T>
__global__ void DequantizeBlockScaledFp8Kernel(T* __restrict__ out,
                                               const __nv_fp8_e4m3* __restrict__ b_fp8,
                                               const float* __restrict__ weight_scale,
                                               int n,
                                               int k,
                                               int k_blocks,
                                               int block_size,
                                               bool transpose) {
  const long long total = static_cast<long long>(n) * k;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / k);
  const int col = static_cast<int>(idx - static_cast<long long>(row) * k);
  const int blk = col / block_size;
  const float scale = weight_scale[row * k_blocks + blk];
  out[transpose ? static_cast<long long>(col) * n + row : idx] =
      from_float<T>(static_cast<float>(b_fp8[idx]) * scale);
}

// Vectorized dequantization for K % 16 == 0. Each thread converts one 16-element K chunk of a
// single row: a coalesced 16-byte FP8 load and a coalesced 32-byte (2 x uint4) FP16/BF16 store.
// The row index comes from a 2D grid (blockIdx.y), avoiding the expensive 64-bit idx / k division
// of the scalar kernel. When block_size is a multiple of 16 the whole chunk shares one scale, so a
// single scale value is loaded per chunk instead of per element.
template <typename T>
__global__ void DequantizeBlockScaledFp8Vec16Kernel(T* __restrict__ out,
                                                    const __nv_fp8_e4m3* __restrict__ b_fp8,
                                                    const float* __restrict__ weight_scale,
                                                    int n,
                                                    int k,
                                                    int k_blocks,
                                                    int block_size,
                                                    int kv,
                                                    bool block_aligned16) {
  const int g = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;  // 16-element chunk in a row
  if (g >= kv) {
    return;
  }
  const int col0 = g * 16;
  for (int row = blockIdx.y; row < n; row += gridDim.y) {
    const size_t base = static_cast<size_t>(row) * k + col0;
    const uint4 raw = *reinterpret_cast<const uint4*>(b_fp8 + base);
    const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
    const float* srow = weight_scale + static_cast<size_t>(row) * k_blocks;

    T outv[16];
    if (block_aligned16) {
      const float scale = srow[col0 / block_size];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        outv[i] = from_float<T>(static_cast<float>(bp[i]) * scale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        outv[i] = from_float<T>(static_cast<float>(bp[i]) * srow[(col0 + i) / block_size]);
      }
    }

    uint4* op = reinterpret_cast<uint4*>(out + base);
    op[0] = *reinterpret_cast<const uint4*>(&outv[0]);
    op[1] = *reinterpret_cast<const uint4*>(&outv[8]);
  }
}

template <typename T>
__global__ void AddBiasKernel(T* __restrict__ y, const T* __restrict__ bias, int m, int n) {
  const long long total = static_cast<long long>(m) * n;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % n);
  y[idx] = from_float<T>(to_float<T>(y[idx]) + to_float<T>(bias[col]));
}

// Statically quantizes a FP16/BF16 activation to FP8 E4M3 using a single per-tensor scale and then
// dequantizes it back to the activation type. This realizes W8A8 activation numerics: the FP8
// rounding error is intentionally introduced so the result matches native W8A8 execution, while the
// downstream GEMM stays in the activation type (architecture independent). a_scale is the dequant
// scale, so the quantized value is fp8_e4m3(a / a_scale) and the emitted activation is
// fp8_e4m3(a / a_scale) * a_scale.
template <typename T>
__global__ void QuantizeDequantizeActivationFp8Kernel(T* __restrict__ out,
                                                      const T* __restrict__ in,
                                                      const float* __restrict__ a_scale,
                                                      long long total) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float scale = *a_scale;
  const float inv_scale = scale != 0.f ? 1.0f / scale : 0.f;
  const float x = to_float<T>(in[idx]);
  const float q = static_cast<float>(__nv_fp8_e4m3(x * inv_scale));
  out[idx] = from_float<T>(q * scale);
}

// In-register form of QuantizeDequantizeActivationFp8Kernel for the 8 activations packed in one
// uint4. Bit-identical to the standalone kernel, so the GEMV can absorb the W8A8 activation
// rounding instead of round-tripping A through a scratch buffer.
template <typename AType>
__device__ __forceinline__ void Fp8ActQdq16(uint4& raw, float inv_scale, float scale) {
  AType* v = reinterpret_cast<AType*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float q = static_cast<float>(__nv_fp8_e4m3(to_float<AType>(v[i]) * inv_scale));
    v[i] = from_float<AType>(q * scale);
  }
}

// -----------------------------------------------------------------------------
// Fused FP8 weight-only GEMV fast path for the decode phase (small M).
//
// Each warp computes RowsPerWarp x ColsPerWarp output elements Y[row, col]. The 32
// lanes of the warp cooperatively reduce over K using 16-wide vectorized loads, so
// the FP8 weight B is streamed exactly once with fully coalesced warp transactions.
// Block scales are applied once per K-block (not per element): a lane's 16-element
// chunk is guaranteed to lie inside a single K-block whenever block_size is a
// multiple of 16, so weight_scale is folded in per chunk. Runs on SM80+.
//
// ColsPerWarp / Unroll exist because the decode GEMV is bound by the number of
// *outstanding* loads, not by bandwidth or by instruction count. Nsight Compute on
// an 8192x2048 decode GEMV (H200) reports only 1.0 full waves, L1 hit rate 79.6%
// (the A row is already resident, so staging it in shared memory buys nothing), and
// a third of the warp stall cycles waiting on L1TEX. With one 16-element B chunk in
// flight per thread, each thread only moves K/32 bytes of B and eats the full L1
// latency every iteration.
//
//   * Unroll      issues `Unroll` independent B/A loads before consuming any of them.
//   * ColsPerWarp lets one A load feed `ColsPerWarp` independent FMA chains, which
//                 both raises arithmetic intensity and adds independent B streams.
//
// Both trade occupancy (already register-capped at 8 blocks/SM) for per-thread
// memory-level parallelism, which is the right trade when there is only one wave.
// <RowsPerWarp, 1, 1> reproduces the original one-column-per-warp geometry.

// Vector-2 companion type and float2 widening for the accumulate type come from
// core/providers/cuda/cu_inc/cuda_type_helper.cuh (Vec2 / to_float2), shared with
// the NVFP4 GEMV in matmul_block_scaled_fp4.cu.

// Convert 16 packed FP8 bytes into 8 half2 (one `cvt.rn.f16x2.e4m3x2` per pair
// instead of 16 scalar converts). FP8 E4M3 is exactly representable in FP16, so
// this is lossless for both the half and the bfloat16 instantiation.
//
// Written as a macro rather than a helper taking `__half2 (&)[8]`: an array passed
// by reference is placed in local memory, which costs ~2x here and much more at
// RowsPerWarp > 1 where the spill lands in the inner loop.
#define ORT_FP8_GEMV_CVT16(raw, bh)                                                         \
  do {                                                                                      \
    const __nv_fp8x2_storage_t* _p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&(raw)); \
    _Pragma("unroll") for (int _i = 0; _i < 8; ++_i) {                                      \
      (bh)[_i] = __half2(__nv_cvt_fp8x2_to_halfraw2(_p[_i], __NV_E4M3));                    \
    }                                                                                       \
  } while (0)

// acc += dot(A[0:16], B[0:16]) with FP32 accumulation, matching the scalar version bit for bit.
//
// `bh` is always __half2: ORT_FP8_GEMV_CVT16 decodes FP8 E4M3 through __nv_cvt_fp8x2_to_halfraw2
// for both the half and the bfloat16 instantiation (E4M3 is exactly representable in FP16), so
// __half22float2 is the correct widening here regardless of AType. If ORT_FP8_GEMV_CVT16 is ever
// changed to emit __nv_bfloat162 for the bfloat16 path, this macro must be updated in lockstep.
#define ORT_FP8_GEMV_DOT16(a0, a1, bh, acc)                   \
  do {                                                        \
    const AVec2* _a0 = reinterpret_cast<const AVec2*>(&(a0)); \
    const AVec2* _a1 = reinterpret_cast<const AVec2*>(&(a1)); \
    _Pragma("unroll") for (int _i = 0; _i < 4; ++_i) {        \
      const float2 _av = to_float2(_a0[_i]);                  \
      const float2 _bv = __half22float2((bh)[_i]);            \
      (acc) = fmaf(_av.x, _bv.x, (acc));                      \
      (acc) = fmaf(_av.y, _bv.y, (acc));                      \
    }                                                         \
    _Pragma("unroll") for (int _i = 0; _i < 4; ++_i) {        \
      const float2 _av = to_float2(_a1[_i]);                  \
      const float2 _bv = __half22float2((bh)[4 + _i]);        \
      (acc) = fmaf(_av.x, _bv.x, (acc));                      \
      (acc) = fmaf(_av.y, _bv.y, (acc));                      \
    }                                                         \
  } while (0)

template <int RowsPerWarp, int ColsPerWarp, int Unroll, typename AType>
__global__ void MatMulBlockScaledFp8GemvKernel(AType* __restrict__ output,
                                               const AType* __restrict__ input_a,
                                               const __nv_fp8_e4m3* __restrict__ input_b,
                                               const float* __restrict__ weight_scale,
                                               const AType* __restrict__ bias,
                                               const float* __restrict__ act_scale,
                                               int m,
                                               int n,
                                               int k,
                                               int block_size,
                                               int k_blocks) {
  using AVec2 = Vec2<AType>;

  constexpr int kElemsPerLane = 16;
  constexpr int kStride = 32 * kElemsPerLane;  // 512 elements per warp sub-chunk

  const int lane = threadIdx.x;  // 0..31
  const int col_base = (blockIdx.x * blockDim.y + threadIdx.y) * ColsPerWarp;
  const int row_base = blockIdx.y * RowsPerWarp;
  if (row_base >= m || col_base >= n) {
    return;
  }

  const bool act_qdq = act_scale != nullptr;
  const float a_scale = act_qdq ? *act_scale : 0.f;
  const float a_inv_scale = a_scale != 0.f ? 1.0f / a_scale : 0.f;

  float acc[RowsPerWarp][ColsPerWarp] = {};

  for (int base = 0; base < k; base += kStride * Unroll) {
    // Phase 1: issue every load for this iteration back to back. Nothing here consumes a
    // loaded value, so all Unroll * (ColsPerWarp + 2 * RowsPerWarp) requests are in flight
    // at once instead of one at a time.
    uint4 b_raw[Unroll][ColsPerWarp];
    uint4 a_lo[Unroll][RowsPerWarp];
    uint4 a_hi[Unroll][RowsPerWarp];
    int koff[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      koff[u] = base + u * kStride + lane * kElemsPerLane;
      const bool k_ok = koff[u] < k;
#pragma unroll
      for (int c = 0; c < ColsPerWarp; ++c) {
        const int col = col_base + c;
        b_raw[u][c] = (k_ok && col < n)
                          ? *reinterpret_cast<const uint4*>(input_b + static_cast<size_t>(col) * k + koff[u])
                          : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
        if (k_ok && row < m) {
          const uint4* ap = reinterpret_cast<const uint4*>(input_a + static_cast<size_t>(row) * k + koff[u]);
          a_lo[u][r] = ap[0];
          a_hi[u][r] = ap[1];
          if (act_qdq) {
            Fp8ActQdq16<AType>(a_lo[u][r], a_inv_scale, a_scale);
            Fp8ActQdq16<AType>(a_hi[u][r], a_inv_scale, a_scale);
          }
        } else {
          a_lo[u][r] = make_uint4(0, 0, 0, 0);
          a_hi[u][r] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // Phase 2: consume.
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      if (koff[u] >= k) {
        continue;
      }
      const int kb = koff[u] / block_size;
      if constexpr (RowsPerWarp == 1) {
        // One row: nothing to amortize, and the extra live fp32 values of the hoisted
        // form measurably cost occupancy at the wide M == 1 tiles (<1,4,2>).
#pragma unroll
        for (int c = 0; c < ColsPerWarp; ++c) {
          const int col = col_base + c;
          if (col >= n) {
            continue;
          }
          __half2 b_half[8];
          ORT_FP8_GEMV_CVT16(b_raw[u][c], b_half);
          const float b_scale = weight_scale[static_cast<size_t>(col) * k_blocks + kb];
          float partial = 0.0f;
          ORT_FP8_GEMV_DOT16(a_lo[u][0], a_hi[u][0], b_half, partial);
          acc[0][c] += partial * b_scale;
        }
      } else {
        // M > 1 (speculative decode / MTP verify): the naive form widens B to fp32 once
        // per row and A once per column, so a lane spends 8 + 48 * RowsPerWarp
        // instructions per 16 weight bytes and the kernel goes ALU-bound long before it
        // saturates HBM (1.25 TB/s at M == 4 vs 2.35 TB/s at M == 1 on H200). Hoisting
        // both widenings out of the inner loop drops that to
        // 8 * C + 16 * R + 16 * C + 16 * R * C, i.e. 200 -> 120 per column at R=4, C=2.
        // The fma order is unchanged, so the result is bit-identical to the scalar form.
        // The 16-element chunk is consumed in two halves to keep only RowsPerWarp * 8
        // widened A values live at a time.
        __half2 b_half[ColsPerWarp][8];
#pragma unroll
        for (int c = 0; c < ColsPerWarp; ++c) {
          ORT_FP8_GEMV_CVT16(b_raw[u][c], b_half[c]);
        }
        float b_scale[ColsPerWarp];
#pragma unroll
        for (int c = 0; c < ColsPerWarp; ++c) {
          const int col = col_base + c;
          b_scale[c] = (col < n) ? weight_scale[static_cast<size_t>(col) * k_blocks + kb] : 0.0f;
        }
        float partial[RowsPerWarp][ColsPerWarp] = {};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a_f[RowsPerWarp][8];
#pragma unroll
          for (int r = 0; r < RowsPerWarp; ++r) {
            const AVec2* av = reinterpret_cast<const AVec2*>(h == 0 ? &a_lo[u][r] : &a_hi[u][r]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 v = to_float2(av[i]);
              a_f[r][2 * i] = v.x;
              a_f[r][2 * i + 1] = v.y;
            }
          }
#pragma unroll
          for (int c = 0; c < ColsPerWarp; ++c) {
            float b_f[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 v = __half22float2(b_half[c][4 * h + i]);
              b_f[2 * i] = v.x;
              b_f[2 * i + 1] = v.y;
            }
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
              for (int j = 0; j < 8; ++j) {
                partial[r][c] = fmaf(a_f[r][j], b_f[j], partial[r][c]);
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
          if (row_base + r >= m) {
            continue;
          }
#pragma unroll
          for (int c = 0; c < ColsPerWarp; ++c) {
            if (col_base + c >= n) {
              continue;
            }
            acc[r][c] += partial[r][c] * b_scale[c];
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
    for (int c = 0; c < ColsPerWarp; ++c) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][c] += __shfl_down_sync(0xffffffffu, acc[r][c], offset);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      const int row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (int c = 0; c < ColsPerWarp; ++c) {
        const int col = col_base + c;
        if (col >= n) {
          continue;
        }
        float result = acc[r][c];
        if (bias != nullptr) {
          result += to_float<AType>(bias[col]);
        }
        output[static_cast<size_t>(row) * n + col] = from_float<AType>(result);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Tensor-core decode GEMV (SM80+): mma.m16n8k16 with FP32 accumulation.
//
// The fp32-FMA kernel above is ALU bound once M > 1 (it re-widens A and B to fp32 for every
// row/column pair), so more ILP cannot help. The fix is to stop doing the dot products on the
// FMA pipe. Operand mapping -- the WEIGHT goes into the mma A slot and the ACTIVATION into the
// mma B slot, which makes both fragments match the tensors' natural layouts:
//
//   mma A[16, 16] row-major  <- weight[16 output columns][16 k]   (B is [N, K] row-major)
//   mma B[16,  8] col-major  <- activation[16 k][8 rows]          (A is [M, K] row-major)
//   mma D[16,  8]            -> y[16 output columns][8 rows]
//
// The mma "M" extent is therefore our N (16 output columns per warp) and the mma "N" extent is
// our M (up to 8 rows, which is why the caller caps the GEMV at M <= 8). At M == 4 half the mma N
// lanes are idle, but the instruction count per weight byte still drops about 10x, which is what
// the ALU-bound path needs.
//
// Per-lane fragment ownership (PTX m16n8k16 layout, g = lane >> 2, t = lane & 3). A lane does NOT
// compute a whole dot product on its own: the tensor core exchanges partial products across the
// warp, so the A rows, the B column and the D rows a lane holds are three independent slices.
//
//   ra[0] = (mma row g,     mma k 2t,   2t+1)  <- weight column col_lo = 16 * blockIdx.x + g
//   ra[1] = (mma row g + 8, mma k 2t,   2t+1)  <- weight column col_hi = col_lo + 8
//   ra[2] = (mma row g,     mma k 2t+8, 2t+9)  <- weight column col_lo
//   ra[3] = (mma row g + 8, mma k 2t+8, 2t+9)  <- weight column col_hi
//   rb[0] = (mma k 2t,   2t+1, mma col g)      <- activation row g
//   rb[1] = (mma k 2t+8, 2t+9, mma col g)      <- activation row g
//   acc[0], acc[1] = (mma row g,     mma col 2t, 2t+1) -> y[2t][col_lo], y[2t+1][col_lo]
//   acc[2], acc[3] = (mma row g + 8, mma col 2t, 2t+1) -> y[2t][col_hi], y[2t+1][col_hi]
//
// So `g` selects both the output-column pair (through the A fragment) and the activation row
// (through the B fragment), while the accumulator the lane ends up owning covers output rows 2t
// and 2t + 1. Loads keyed off `g` and stores keyed off `t` are therefore both correct and are
// deliberately different; see the GemvTensorCoreLaneOwnership* tests for a one-hot probe of this
// mapping.
//
// Fragment loads are made fully coalesced by PERMUTING the K axis. K is a reduction axis, so any
// permutation applied to BOTH operands leaves the result unchanged. Within a 64-element K window,
// lane (g = lane >> 2, t = lane & 3) loads one contiguous uint4 of weight bytes [16t, 16t + 16)
// and the matching 32 contiguous activation bytes; mma step j (0..3) then consumes bytes
// 4j..4j+3 of them. Four lanes cover 64 contiguous bytes of one weight row (no sector
// over-fetch) and a single uint4 load feeds four mma instructions.
//
// KSplit warps per block each take a strided share of the K windows and are reduced through
// shared memory. Without it, 16 columns per warp gives ~8x fewer warps than the fp32 kernel and
// the GPU runs out of memory-level parallelism long before it runs out of bandwidth.
//
// Requires k % 64 == 0 and block_size % 64 == 0 (so a 64-element window lies in one K block).
// Accuracy is unchanged: FP8 E4M3 -> FP16/BF16 is lossless, products are exact, and the mma
// accumulates in FP32 exactly like the FMA path. Measured against an FP64 reference on the Qwen
// shapes, max error is identical to the FMA kernel (~2-4e-4, i.e. pure FP16 output rounding).
template <typename T>
struct Fp8GemvMma;

template <>
struct Fp8GemvMma<half> {
  // 16 packed FP8 bytes -> 8 b32 registers, each holding 2 halves.
  __device__ __forceinline__ static void Cvt16(const uint4& raw, uint32_t (&out)[8]) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const __half2 v = __half2(__nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3));
      out[i] = *reinterpret_cast<const uint32_t*>(&v);
    }
  }

  __device__ __forceinline__ static void Mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    ORT_UNUSED_PARAMETER(d);
    ORT_UNUSED_PARAMETER(a);
    ORT_UNUSED_PARAMETER(b);
#endif
  }
};

template <>
struct Fp8GemvMma<__nv_bfloat16> {
  // No direct FP8 -> BF16 intrinsic, so go through the (lossless) FP8 -> FP16 converter and then
  // FP16 -> FP32 -> BF16. Every step is exact for E4M3: 3 mantissa bits fit in BF16's 7 and the
  // E4M3 exponent range is a strict subset of BF16's.
  __device__ __forceinline__ static void Cvt16(const uint4& raw, uint32_t (&out)[8]) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3));
      const __nv_bfloat162 v = __float22bfloat162_rn(__half22float2(h));
      out[i] = *reinterpret_cast<const uint32_t*>(&v);
    }
  }

  __device__ __forceinline__ static void Mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    ORT_UNUSED_PARAMETER(d);
    ORT_UNUSED_PARAMETER(a);
    ORT_UNUSED_PARAMETER(b);
#endif
  }
};

template <typename T>
__device__ __forceinline__ T* Fp8OutputColumn(T* output, int col, int& stride,
                                              const Fp8GemvOutputSplit& split) {
  for (int i = 0; i < split.count; ++i) {
    if (col < split.widths[i]) {
      stride = split.widths[i];
      return static_cast<T*>(split.outputs[i]) + col;
    }
    col -= split.widths[i];
  }
  return output + col;
}

template <typename T, bool SharedExpert>
__device__ __forceinline__ T Fp8RoundOutput(float value, int row, int col, int n,
                                            const Fp8SharedExpertEpilogue& epilogue) {
  T rounded = from_float<T>(value);
  if constexpr (SharedExpert) {
    rounded = from_float<T>(to_float<T>(rounded) * epilogue.row_gates[row]);
    rounded = from_float<T>(to_float<T>(rounded) +
                            to_float<T>(static_cast<const T*>(epilogue.residual)[static_cast<size_t>(row) * n + col]));
  }
  return rounded;
}

template <int KSplit, int MTiles, bool SplitOutput, typename AType, bool PerColumnS2 = false,
          bool SharedExpert = false>
__device__ __forceinline__ void Fp8MmaGemvBody(AType* __restrict__ output,
                                               const AType* __restrict__ input_a,
                                               const __nv_fp8_e4m3* __restrict__ input_b,
                                               const float* __restrict__ weight_scale,
                                               const AType* __restrict__ bias,
                                               const float* __restrict__ act_scale,
                                               int m,
                                               int n,
                                               int k,
                                               int block_size,
                                               int k_blocks,
                                               const Fp8GemvOutputSplit& split,
                                               const Fp8SharedExpertEpilogue& epilogue = {}) {
  using Mma = Fp8GemvMma<AType>;

  // Keep the same MMA/KSplit reduction, but remove per-block scaling and QDQ
  // bookkeeping for weight-only, two-row projections with one scale per column.
  if constexpr (PerColumnS2) {
    m = 2;
    k_blocks = 1;
  }
  const bool act_qdq = !PerColumnS2 && act_scale != nullptr;
  const float a_scale = act_qdq ? *act_scale : 0.f;
  const float a_inv_scale = a_scale != 0.f ? 1.0f / a_scale : 0.f;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  // See the fragment-ownership table above: `g` indexes the A rows (output columns col_lo/col_hi)
  // and the B column (activation row); `t` indexes the k sub-slice and, in the accumulator, the
  // output rows 2t / 2t + 1 this lane stores.
  const int g = lane >> 2;
  const int t = lane & 3;
  const int windows = k >> 6;

  const int col_lo = blockIdx.x * 16 + g;
  const int col_hi = col_lo + 8;
  const bool lo_ok = col_lo < n;
  const bool hi_ok = col_hi < n;

  // Each M tile is a different group of 8 activation rows sharing one pass over the FP8 weight.
  bool a_ok[MTiles];
  size_t a_off[MTiles];
#pragma unroll
  for (int mt = 0; mt < MTiles; ++mt) {
    const int row = g + (mt << 3);
    a_ok[mt] = row < m;
    a_off[mt] = static_cast<size_t>(a_ok[mt] ? row : 0) * k + (t << 4);
  }

  float acc[MTiles][4] = {};   // scaled, summed over K blocks
  float accb[MTiles][4] = {};  // unscaled, within the current K block
  int cur_kb = -1;

#pragma unroll 2
  for (int wi = warp; wi < windows; wi += KSplit) {
    const int k0 = wi << 6;
    const int kb = PerColumnS2 ? 0 : k0 / block_size;
    if (kb != cur_kb) {
      if (cur_kb >= 0) {
        const float s_lo = lo_ok ? weight_scale[static_cast<size_t>(col_lo) * k_blocks + cur_kb] : 0.f;
        const float s_hi = hi_ok ? weight_scale[static_cast<size_t>(col_hi) * k_blocks + cur_kb] : 0.f;
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
          acc[mt][0] += accb[mt][0] * s_lo;
          acc[mt][1] += accb[mt][1] * s_lo;
          acc[mt][2] += accb[mt][2] * s_hi;
          acc[mt][3] += accb[mt][3] * s_hi;
          accb[mt][0] = accb[mt][1] = accb[mt][2] = accb[mt][3] = 0.f;
        }
      }
      cur_kb = kb;
    }

    // Activation: 32 contiguous bytes of each tile's row (rows >= m read as zero).
    uint4 a_raw[MTiles][2];
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
      if (a_ok[mt]) {
        const uint4* ap = reinterpret_cast<const uint4*>(input_a + a_off[mt] + k0);
        a_raw[mt][0] = ap[0];
        a_raw[mt][1] = ap[1];
        if (act_qdq) {
          Fp8ActQdq16<AType>(a_raw[mt][0], a_inv_scale, a_scale);
          Fp8ActQdq16<AType>(a_raw[mt][1], a_inv_scale, a_scale);
        }
      } else {
        a_raw[mt][0] = make_uint4(0, 0, 0, 0);
        a_raw[mt][1] = make_uint4(0, 0, 0, 0);
      }
    }

    // Weight: one uint4 per half of the 16-column tile.
    const uint4 w_lo = lo_ok
                           ? *reinterpret_cast<const uint4*>(input_b + static_cast<size_t>(col_lo) * k + k0 + (t << 4))
                           : make_uint4(0, 0, 0, 0);
    const uint4 w_hi = hi_ok
                           ? *reinterpret_cast<const uint4*>(input_b + static_cast<size_t>(col_hi) * k + k0 + (t << 4))
                           : make_uint4(0, 0, 0, 0);

    uint32_t b_lo[8], b_hi[8];
    Mma::Cvt16(w_lo, b_lo);
    Mma::Cvt16(w_hi, b_hi);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t ra[4] = {b_lo[2 * j], b_hi[2 * j], b_lo[2 * j + 1], b_hi[2 * j + 1]};
#pragma unroll
      for (int mt = 0; mt < MTiles; ++mt) {
        const uint32_t* av = reinterpret_cast<const uint32_t*>(a_raw[mt]);
        const uint32_t rb[2] = {av[2 * j], av[2 * j + 1]};
        Mma::Mma(accb[mt], ra, rb);
      }
    }
  }

  if (cur_kb >= 0) {
    const float s_lo = lo_ok ? weight_scale[static_cast<size_t>(col_lo) * k_blocks + cur_kb] : 0.f;
    const float s_hi = hi_ok ? weight_scale[static_cast<size_t>(col_hi) * k_blocks + cur_kb] : 0.f;
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
      acc[mt][0] += accb[mt][0] * s_lo;
      acc[mt][1] += accb[mt][1] * s_lo;
      acc[mt][2] += accb[mt][2] * s_hi;
      acc[mt][3] += accb[mt][3] * s_hi;
    }
  }

  if constexpr (KSplit > 1) {
    __shared__ float red[KSplit * 32 * 4 * MTiles];
    float* slot = red + (warp * 32 + lane) * (4 * MTiles);
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        slot[mt * 4 + i] = acc[mt][i];
      }
    }
    __syncthreads();
    if (warp != 0) {
      return;
    }
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[mt][i] = 0.f;
      }
    }
    for (int ws = 0; ws < KSplit; ++ws) {
      const float* p = red + (ws * 32 + lane) * (4 * MTiles);
#pragma unroll
      for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[mt][i] += p[mt * 4 + i];
        }
      }
    }
  }

  // Lane (g, t) owns y[2t, col_lo], y[2t + 1, col_lo], y[2t, col_hi], y[2t + 1, col_hi] of each
  // M tile.
  const float bias_lo = (bias != nullptr && lo_ok) ? to_float<AType>(bias[col_lo]) : 0.f;
  const float bias_hi = (bias != nullptr && hi_ok) ? to_float<AType>(bias[col_hi]) : 0.f;
  int stride_lo = n;
  int stride_hi = n;
  AType* output_lo = output + col_lo;
  AType* output_hi = output + col_hi;
  if constexpr (SplitOutput) {
    output_lo = lo_ok ? Fp8OutputColumn(output, col_lo, stride_lo, split) : nullptr;
    output_hi = hi_ok ? Fp8OutputColumn(output, col_hi, stride_hi, split) : nullptr;
  }
#pragma unroll
  for (int mt = 0; mt < MTiles; ++mt) {
    const int row = (t << 1) + (mt << 3);
    if (row < m) {
      if (lo_ok) {
        output_lo[static_cast<size_t>(row) * stride_lo] =
            Fp8RoundOutput<AType, SharedExpert>(acc[mt][0] + bias_lo, row, col_lo, n, epilogue);
      }
      if (hi_ok) {
        output_hi[static_cast<size_t>(row) * stride_hi] =
            Fp8RoundOutput<AType, SharedExpert>(acc[mt][2] + bias_hi, row, col_hi, n, epilogue);
      }
    }
    if (row + 1 < m) {
      if (lo_ok) {
        output_lo[static_cast<size_t>(row + 1) * stride_lo] =
            Fp8RoundOutput<AType, SharedExpert>(acc[mt][1] + bias_lo, row + 1, col_lo, n, epilogue);
      }
      if (hi_ok) {
        output_hi[static_cast<size_t>(row + 1) * stride_hi] =
            Fp8RoundOutput<AType, SharedExpert>(acc[mt][3] + bias_hi, row + 1, col_hi, n, epilogue);
      }
    }
  }
}

// Two entry points over one body. The pinned one carries a residency hint; see
// `Fp8MmaGemvPinsResidency` for when the launcher picks it and why the plain one has to stay.
// clang-format off
#if CUDA_VERSION >= 12000
// Avoid a per-thread local copy when the inlined store helper references this descriptor.
#define ORT_FP8_SPLIT_PARAM const __grid_constant__ Fp8GemvOutputSplit split
#else
#define ORT_FP8_SPLIT_PARAM const Fp8GemvOutputSplit split
#endif
#define ORT_FP8_MMA_GEMV_PARAMS                  \
  AType* __restrict__ output,                    \
      const AType* __restrict__ input_a,         \
      const __nv_fp8_e4m3* __restrict__ input_b, \
      const float* __restrict__ weight_scale,    \
      const AType* __restrict__ bias,            \
      const float* __restrict__ act_scale,       \
      int m, int n, int k, int block_size, int k_blocks, ORT_FP8_SPLIT_PARAM

#define ORT_FP8_MMA_GEMV_ARGS \
  output, input_a, input_b, weight_scale, bias, act_scale, m, n, k, block_size, k_blocks, split
// clang-format on

template <int KSplit, int MTiles, bool SplitOutput, typename AType>
__global__ void MatMulBlockScaledFp8MmaGemvKernel(ORT_FP8_MMA_GEMV_PARAMS) {
  Fp8MmaGemvBody<KSplit, MTiles, SplitOutput, AType>(ORT_FP8_MMA_GEMV_ARGS);
}

template <int KSplit, int MTiles, bool SplitOutput, typename AType>
__global__ void MatMulBlockScaledFp8MmaGemvPerColumnS2Kernel(ORT_FP8_MMA_GEMV_PARAMS) {
  Fp8MmaGemvBody<KSplit, MTiles, SplitOutput, AType, true>(ORT_FP8_MMA_GEMV_ARGS);
}

template <int KSplit, int MTiles, bool SplitOutput, typename AType>
__global__ __launch_bounds__(32 * KSplit, 3) void MatMulBlockScaledFp8MmaGemvKernelPinned(ORT_FP8_MMA_GEMV_PARAMS) {
  Fp8MmaGemvBody<KSplit, MTiles, SplitOutput, AType>(ORT_FP8_MMA_GEMV_ARGS);
}

#undef ORT_FP8_MMA_GEMV_ARGS
#undef ORT_FP8_MMA_GEMV_PARAMS
#undef ORT_FP8_SPLIT_PARAM

template <int KSplit, bool PerColumnS2>
__global__ void MatMulFp8SharedExpertKernel(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_fp8_e4m3* b,
    const float* scales, int m, int n, int k, int block_size, int k_blocks,
    Fp8SharedExpertEpilogue epilogue) {
  Fp8MmaGemvBody<KSplit, 1, false, __nv_bfloat16, PerColumnS2, true>(
      output, a, b, scales, nullptr, nullptr, m, n, k, block_size, k_blocks, {}, epilogue);
}

__global__ void Fp8SharedExpertEpilogueKernel(__nv_bfloat16* output, Fp8SharedExpertEpilogue epilogue,
                                              int64_t count, int n) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = Fp8RoundOutput<__nv_bfloat16, true>(
        __bfloat162float(output[index]), static_cast<int>(index / n), static_cast<int>(index % n), n, epilogue);
  }
}

// Kill switch for A/B testing the tensor-core path against the FMA path in the same binary.
bool Fp8GemvMmaEnabled() {
  static bool const enabled = onnxruntime::ParseEnvironmentVariableWithDefault<bool>("ORT_FP8_GEMV_MMA", true);
  return enabled;
}

// Largest M each sub-path accepts. One mma launch unrolls 4 tiles of the mma's 8-row N extent.
// Larger speculative batches are split into two launches so they keep the same per-row arithmetic
// instead of switching to the dequantize + cuBLAS path.
constexpr int kFp8MmaGemvTileM = 32;
constexpr int kFp8MmaGemvSupportedMaxM = 64;
constexpr int kFp8ScalarGemvMaxM = 8;

int Fp8MmaGemvDispatchMaxM() {
  static const int max_m = onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_MAX_M", 32);
  ORT_ENFORCE(max_m >= 1 && max_m <= kFp8MmaGemvSupportedMaxM,
              "ORT_FP8_GEMV_MAX_M must be in [1, ", kFp8MmaGemvSupportedMaxM, "], got ", max_m, ".");
  return max_m;
}

}  // namespace

#endif  // !DISABLE_FLOAT8_TYPES && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

int ApplyFp8MmaKSplitOverride(int k_split, int m, int n, int k);

Status LaunchFp8SharedExpertEpilogue(void* y, const Fp8SharedExpertEpilogue& epilogue,
                                     int m, int n, cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const int64_t count = static_cast<int64_t>(m) * n;
  ORT_RETURN_IF_NOT(m >= 0 && n > 0 && epilogue.row_gates && epilogue.residual,
                    "Shared expert epilogue requires valid dimensions, gates and residual");
  if (count == 0) return Status::OK();
  Fp8SharedExpertEpilogueKernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(y), epilogue, count, n);
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(epilogue);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Shared expert requires FP8 support.");
#endif
}

Status LaunchMatMulFp8SharedExpert(void* y, const void* a, const void* weight,
                                   const float* scales, int m, int n, int k, int block_size,
                                   const Fp8SharedExpertEpilogue& epilogue,
                                   const cudaDeviceProp& device_prop, cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  ORT_RETURN_IF_NOT(m >= 1 && m <= 2 && k >= 256 && k % 64 == 0 && block_size % 64 == 0 &&
                        device_prop.major >= 8 && epilogue.row_gates && epilogue.residual,
                    "Shared expert fused GEMV requires SM80+, M <= 2 and aligned K/block size");
  const int selected = PickGenericFp8MmaKSplit(n, k / 64);
  const int k_split = ApplyFp8MmaKSplitOverride(selected, m, n, k);
  static const int per_column_s2 =
      ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_PER_COLUMN_S2", 0);
  ORT_RETURN_IF_NOT(per_column_s2 == 0 || per_column_s2 == 1,
                    "ORT_FP8_GEMV_PER_COLUMN_S2 must be 0 or 1");
  const auto launch = [&]<int KSplit>() {
    const auto variant = [&]<bool S2>() {
      MatMulFp8SharedExpertKernel<KSplit, S2><<<static_cast<unsigned int>((n + 15) / 16),
                                                dim3(32, KSplit), 0, stream>>>(
          static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(a),
          static_cast<const __nv_fp8_e4m3*>(weight), scales, m, n, k, block_size,
          (k + block_size - 1) / block_size, epilogue);
    };
    if (per_column_s2 && m == 2 && block_size == k) {
      variant.template operator()<true>();
    } else {
      variant.template operator()<false>();
    }
  };
  switch (k_split) {
    case 4:
      launch.template operator()<4>();
      break;
    case 8:
      launch.template operator()<8>();
      break;
    case 16:
      launch.template operator()<16>();
      break;
    case 32:
      launch.template operator()<32>();
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported shared expert K split: ", k_split);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(weight);
  ORT_UNUSED_PARAMETER(scales);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(epilogue);
  ORT_UNUSED_PARAMETER(device_prop);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Shared expert requires FP8 support.");
#endif
}

Status LaunchDequantizeBlockScaledFp8(void* b_dequant,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream,
                                      bool transpose) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(n) * k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(b_fp8);

  // Fast path: K is a multiple of 16, so each thread owns one aligned 16-element chunk and both the
  // FP8 load and the FP16/BF16 store are fully coalesced. This is the common prefill layout.
  if (!transpose && k % 16 == 0) {
    const int kv = k / 16;  // 16-element chunks per row
    constexpr int kThreads = 256;
    const unsigned int grid_x = static_cast<unsigned int>((kv + kThreads - 1) / kThreads);
    const unsigned int grid_y = static_cast<unsigned int>(n < 65535 ? n : 65535);
    const dim3 blocks{grid_x, grid_y};
    const bool block_aligned16 = (block_size % 16 == 0);
    if (is_bf16) {
      DequantizeBlockScaledFp8Vec16Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, kv,
          block_aligned16);
    } else {
      DequantizeBlockScaledFp8Vec16Kernel<half><<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<half*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, kv, block_aligned16);
    }
    return CUDA_CALL(cudaGetLastError());
  }

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    DequantizeBlockScaledFp8Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, transpose);
  } else {
    DequantizeBlockScaledFp8Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, transpose);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(b_dequant);
  ORT_UNUSED_PARAMETER(b_fp8);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(transpose);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}

Status LaunchAddBiasBlockScaledFp8(void* y,
                                   const void* bias,
                                   int m,
                                   int n,
                                   bool is_bf16,
                                   cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(m) * n;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    AddBiasKernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(bias), m, n);
  } else {
    AddBiasKernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(y), reinterpret_cast<const half*>(bias), m, n);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}
Status LaunchQuantizeDequantizeActivationFp8(void* a_out,
                                             const void* a_in,
                                             const float* a_scale,
                                             int m,
                                             int k,
                                             bool is_bf16,
                                             cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(m) * k;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    QuantizeDequantizeActivationFp8Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(a_out), reinterpret_cast<const __nv_bfloat16*>(a_in), a_scale, total);
  } else {
    QuantizeDequantizeActivationFp8Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(a_out), reinterpret_cast<const half*>(a_in), a_scale, total);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(a_out);
  ORT_UNUSED_PARAMETER(a_in);
  ORT_UNUSED_PARAMETER(a_scale);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}
int MatMulBlockScaledFp8GemvMaxM(int k, int block_size, const cudaDeviceProp& device_prop) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  if (device_prop.major >= 8 && k % 64 == 0 && k >= 256 && block_size % 64 == 0 && Fp8GemvMmaEnabled()) {
    return Fp8MmaGemvDispatchMaxM();
  }
  return kFp8ScalarGemvMaxM;
#else
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(device_prop);
  return 8;
#endif
}

int ApplyFp8MmaKSplitOverride(int k_split, int m, int n, int k) {
  static int const override_k_split =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_KSPLIT", 0);
  static int const match_n =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_MATCH_N", 0);
  static int const match_k =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_MATCH_K", 0);
  ORT_ENFORCE(override_k_split == 0 || override_k_split == 4 || override_k_split == 8 ||
                  override_k_split == 16 || override_k_split == 32,
              "ORT_FP8_GEMV_KSPLIT must be 0, 4, 8, 16, or 32.");
  ORT_ENFORCE(match_n >= 0 && match_k >= 0,
              "ORT_FP8_GEMV_MATCH_N and ORT_FP8_GEMV_MATCH_K must be non-negative.");

  if ((match_n != 0 && n != match_n) || (match_k != 0 && k != match_k) ||
      override_k_split == 0) {
    return k_split;
  }
  ORT_ENFORCE(override_k_split != 32 || m <= 8,
              "ORT_FP8_GEMV_KSPLIT=32 supports M up to 8, got M=", m, ".");
  return override_k_split;
}

bool Fp8MmaGb10TuningEnabled() {
  static bool const enabled = [] {
    const int disable_tuning =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_DISABLE_GB10_TUNING", 0);
    ORT_ENFORCE(disable_tuning == 0 || disable_tuning == 1,
                "ORT_FP8_GEMV_DISABLE_GB10_TUNING must be 0 or 1.");
    return disable_tuning == 0;
  }();
  return enabled;
}

bool MatMulBlockScaledFp8GemvSupportsSplitOutput(int m, int k, int block_size,
                                                 const cudaDeviceProp& device_prop) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  return device_prop.major >= 8 && m > 0 && m <= kFp8MmaGemvTileM &&
         k >= 256 && k % 64 == 0 && block_size % 64 == 0 && Fp8GemvMmaEnabled();
#else
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(device_prop);
  return false;
#endif
}

static Status LaunchMatMulBlockScaledFp8GemvImpl(void* y,
                                                 const void* a,
                                                 const void* b_fp8,
                                                 const float* weight_scale,
                                                 const void* bias,
                                                 const float* act_scale,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 int block_size,
                                                 bool is_bf16,
                                                 const cudaDeviceProp& device_prop,
                                                 cudaStream_t stream,
                                                 bool enable_gb10_ksplit32,
                                                 const Fp8GemvOutputSplit& split) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  if (m <= 0 || n <= 0 || k <= 0) {
    return Status::OK();
  }
  // This kernel assumes K is a multiple of 16 (each lane loads a full 16-element slice) and that
  // block_size is a multiple of 16 (a lane's slice lies inside a single K-block). Guard against
  // misuse if this helper is ever reused outside the callers that already check these.
  ORT_RETURN_IF_NOT(k % 16 == 0, "MatMulBlockQuantizedFp8Weight GEMV requires K divisible by 16, got ", k, ".");
  ORT_RETURN_IF_NOT(block_size % 16 == 0,
                    "MatMulBlockQuantizedFp8Weight GEMV requires block_size divisible by 16, got ", block_size, ".");
  const int k_blocks = (k + block_size - 1) / block_size;
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(b_fp8);
  ORT_RETURN_IF_NOT(split.count >= 0 && split.count <= Fp8GemvOutputSplit::kMaxOutputs,
                    "FP8 GEMV split output count must be between zero and eight.");
  if (split.count != 0) {
    int64_t total = 0;
    for (int i = 0; i < split.count; ++i) {
      ORT_RETURN_IF_NOT(split.widths[i] > 0 && split.outputs[i] != nullptr,
                        "FP8 GEMV split outputs require positive widths and non-null buffers.");
      total += split.widths[i];
    }
    ORT_RETURN_IF_NOT(total == n, "FP8 GEMV split widths must sum to N.");
  }
  ORT_RETURN_IF_NOT(split.count == 0 || MatMulBlockScaledFp8GemvSupportsSplitOutput(m, k, block_size, device_prop),
                    "Direct split output requires the tensor-core GEMV path with M <= 32.");

  if (m > kFp8MmaGemvTileM) {
    ORT_RETURN_IF_NOT(device_prop.major >= 8 && k % 64 == 0 && k >= 256 &&
                          block_size % 64 == 0 && m <= kFp8MmaGemvSupportedMaxM && Fp8GemvMmaEnabled(),
                      "MatMulBlockQuantizedFp8Weight GEMV supports M above ", kFp8MmaGemvTileM,
                      " only on the mma sub-path, got M=", m, ".");
    const size_t element_size = is_bf16 ? sizeof(__nv_bfloat16) : sizeof(half);
    ORT_RETURN_IF_ERROR(LaunchMatMulBlockScaledFp8GemvImpl(
        y, a, b_fp8, weight_scale, bias, act_scale, kFp8MmaGemvTileM, n, k, block_size,
        is_bf16, device_prop, stream, false, {}));
    return LaunchMatMulBlockScaledFp8GemvImpl(
        static_cast<uint8_t*>(y) + static_cast<size_t>(kFp8MmaGemvTileM) * n * element_size,
        static_cast<const uint8_t*>(a) + static_cast<size_t>(kFp8MmaGemvTileM) * k * element_size,
        b_fp8, weight_scale, bias, act_scale, m - kFp8MmaGemvTileM, n, k, block_size,
        is_bf16, device_prop, stream, false, {});
  }

  // Tensor-core path (SM80+). Beats the FMA kernel at every M on H200: 1.06-1.23x at M == 1 and
  // 1.4-1.87x at M == 4, where the FMA kernel is ALU bound. Needs 64-element K windows, and at
  // least 4 of them so KSplit warps have something to do.
  //
  // One warp owns a 16-column x 8-row output tile, where lane (g = lane >> 2, t = lane & 3) reads
  // activation row g + 8 * mt, covers weight columns 16 * blockIdx.x + g and + 8, and stores output
  // rows 2t and 2t + 1 of those two columns. MTiles such groups share one pass over the weight, so
  // M is capped at 8 * MTiles; past that the rows have nowhere to live.
  if (device_prop.major >= 8 && m <= kFp8MmaGemvTileM && k % 64 == 0 && k >= 256 &&
      block_size % 64 == 0 && Fp8GemvMmaEnabled()) {
    const int windows = k / 64;
    // Preserve the generic schedule for recursive tiles from requests above the qualified M range.
    const int selected_k_split =
        enable_gb10_ksplit32 && Fp8MmaGb10TuningEnabled()
            ? PickFp8MmaKSplit(n, m, windows, device_prop.multiProcessorCount,
                               device_prop.major, device_prop.minor)
            : PickGenericFp8MmaKSplit(n, windows);
    const int k_split = ApplyFp8MmaKSplitOverride(selected_k_split, m, n, k);
    const int mtiles = (m > 16) ? 4 : ((m > 8) ? 2 : 1);
    const dim3 mma_blocks{static_cast<unsigned int>((n + 15) / 16)};
    const bool pin_residency = Fp8MmaGemvPinsResidency(
        n, k_split, mtiles, device_prop.multiProcessorCount, device_prop.major, device_prop.minor);
    static const int per_column_s2 =
        ParseEnvironmentVariableWithDefault<int>("ORT_FP8_GEMV_PER_COLUMN_S2", 0);
    ORT_ENFORCE(per_column_s2 == 0 || per_column_s2 == 1,
                "ORT_FP8_GEMV_PER_COLUMN_S2 must be 0 or 1.");
    const auto launch_mma = [&]<int KSplit, int MTiles>() {
      const dim3 mma_threads{32, KSplit};
      const auto launch_variant = [&]<bool SplitOutput>() {
#define ORT_FP8_LAUNCH_MMA(kernel_name)                                                      \
  do {                                                                                       \
    if (is_bf16) {                                                                           \
      kernel_name<KSplit, MTiles, SplitOutput><<<mma_blocks, mma_threads, 0, stream>>>(      \
          reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(a), b, \
          weight_scale, reinterpret_cast<const __nv_bfloat16*>(bias), act_scale, m, n, k,    \
          block_size, k_blocks, split);                                                      \
    } else {                                                                                 \
      kernel_name<KSplit, MTiles, SplitOutput><<<mma_blocks, mma_threads, 0, stream>>>(      \
          reinterpret_cast<half*>(y), reinterpret_cast<const half*>(a), b,                   \
          weight_scale, reinterpret_cast<const half*>(bias), act_scale, m, n, k,             \
          block_size, k_blocks, split);                                                      \
    }                                                                                        \
  } while (0)
        if constexpr (MTiles == 1) {
          if (per_column_s2 && m == 2 && block_size == k && act_scale == nullptr && !pin_residency) {
            ORT_FP8_LAUNCH_MMA(MatMulBlockScaledFp8MmaGemvPerColumnS2Kernel);
            return;
          }
        }
        if constexpr (KSplit == 16 && MTiles == 1) {
          if (pin_residency) {
            ORT_FP8_LAUNCH_MMA(MatMulBlockScaledFp8MmaGemvKernelPinned);
            return;
          }
        }
        ORT_FP8_LAUNCH_MMA(MatMulBlockScaledFp8MmaGemvKernel);
#undef ORT_FP8_LAUNCH_MMA
      };
      if (split.count) {
        launch_variant.template operator()<true>();
      } else {
        launch_variant.template operator()<false>();
      }
    };
    // Only 1, 2 and 4 row tiles are instantiated; an M of 17..24 rounds up to 4 and masks the
    // remainder, which costs nothing next to the weight traffic it shares.
    const auto launch_for_ksplit = [&]<int KSplit>() {
      if (mtiles == 1) {
        launch_mma.template operator()<KSplit, 1>();
      } else if (mtiles == 2) {
        launch_mma.template operator()<KSplit, 2>();
      } else {
        launch_mma.template operator()<KSplit, 4>();
      }
    };
    if (k_split == 32) {
      ORT_ENFORCE(mtiles == 1, "FP8 GEMV KSplit32 supports only M up to 8.");
      launch_mma.template operator()<32, 1>();
    } else if (k_split == 16) {
      launch_for_ksplit.template operator()<16>();
    } else if (k_split == 8) {
      launch_for_ksplit.template operator()<8>();
    } else {
      launch_for_ksplit.template operator()<4>();
    }
    return CUDA_CALL(cudaGetLastError());
  }

  ORT_RETURN_IF_NOT(m <= kFp8ScalarGemvMaxM,
                    "MatMulBlockQuantizedFp8Weight GEMV without the mma sub-path supports M up to ",
                    kFp8ScalarGemvMaxM, ", got ", m, ".");

  constexpr int kWarpsPerBlock = 8;
  const dim3 threads{32, kWarpsPerBlock};
  const auto launch = [&]<int RowsPerWarp, int ColsPerWarp, int Unroll>() {
    const int cols_per_block = kWarpsPerBlock * ColsPerWarp;
    const dim3 blocks{static_cast<unsigned int>((n + cols_per_block - 1) / cols_per_block),
                      static_cast<unsigned int>((m + RowsPerWarp - 1) / RowsPerWarp)};
    if (is_bf16) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp, ColsPerWarp, Unroll><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(a), b,
          weight_scale, reinterpret_cast<const __nv_bfloat16*>(bias), act_scale, m, n, k, block_size, k_blocks);
    } else {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp, ColsPerWarp, Unroll><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<half*>(y), reinterpret_cast<const half*>(a), b,
          weight_scale, reinterpret_cast<const half*>(bias), act_scale, m, n, k, block_size, k_blocks);
    }
  };

  // M == 1 is the decode-with-batch-1 case and by far the hottest. There the grid is a
  // single wave, so widening each warp (ColsPerWarp) and pre-issuing loads (Unroll) buys
  // memory-level parallelism at no real occupancy cost. It only pays once N is large
  // enough that dividing the column count still leaves the GPU full -- below N == 4096
  // the wider tiles measured slower than one column per warp on H200.
  //
  // M in [2, 4] is speculative decode (the MTP verify forward is N+1 tokens wide). There
  // RowsPerWarp > 1 already reads the weight once for every row, so the kernel is ALU
  // bound rather than bandwidth bound, and ColsPerWarp pays for a second reason: it
  // amortizes the fp32 widening of A across columns (see the hoisted path in the kernel).
  // ColsPerWarp is capped so that gridDim.x stays >= ~128 blocks (H200 has 132 SMs) --
  // below that the lost column parallelism costs more than the saved instructions.
  // Measured on H200 (us, M == 4, vs the previous <R,1,1> and vs cuBLAS fp16):
  //   8192x2048  10.9 cublas | 13.7 <4,1,1> | 9.7 <4,4,1>
  //   4096x2048   8.3 cublas |  8.1 <4,1,1> | 6.9 <4,4,1>
  //   2048x4096   8.3 cublas |  9.0 <4,1,1> | 7.9 <4,2,2>
  //    512x2048   7.3 cublas |  5.0 <4,1,1> | 4.6 <4,1,2>
  if (m == 1) {
    if (n >= 8192) {
      launch.template operator()<1, 4, 2>();
    } else if (n >= 4096) {
      launch.template operator()<1, 2, 2>();
    } else {
      launch.template operator()<1, 1, 1>();
    }
  } else if (m <= 2) {
    if (n >= 8192) {
      launch.template operator()<2, 4, 1>();
    } else if (n >= 2048) {
      launch.template operator()<2, 2, 1>();
    } else {
      launch.template operator()<2, 1, 2>();
    }
  } else if (m <= 4) {
    if (n >= 4096) {
      launch.template operator()<4, 4, 1>();
    } else if (n >= 2048) {
      launch.template operator()<4, 2, 2>();
    } else {
      launch.template operator()<4, 1, 2>();
    }
  } else {
    launch.template operator()<8, 1, 1>();
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(b_fp8);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(device_prop);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(enable_gb10_ksplit32);
  ORT_UNUSED_PARAMETER(split);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}

Status LaunchMatMulBlockScaledFp8Gemv(void* y,
                                      const void* a,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      const void* bias,
                                      const float* act_scale,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      const cudaDeviceProp& device_prop,
                                      cudaStream_t stream,
                                      const Fp8GemvOutputSplit& split) {
  return LaunchMatMulBlockScaledFp8GemvImpl(
      y, a, b_fp8, weight_scale, bias, act_scale, m, n, k, block_size,
      is_bf16, device_prop, stream, true, split);
}

}  // namespace onnxruntime::contrib::cuda
