// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

#include <cuda_fp16.h>
#include <type_traits>

// Shared INT/FP4 profiled-shape thresholds.
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
// Shared device-side kernels + launch/dispatch helpers (fpA_intB_gemv namespace).
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_device.cuh"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/platform/env_var_utils.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// Tiling knobs swept by the FP4 GEMV autotuner. These mirror the INT path's default m-tile
// config (kDefaultCtaN/kDefaultThreads) plus two alternative tilings (kCtaN16/kThreads64).
static constexpr int kDefaultCtaN = 8;
static constexpr int kDefaultThreads = 128;
static constexpr int kCtaN16 = 16;
static constexpr int kThreads64 = 64;

int CtaNForConfig(MoeGemvConfig config) {
  return config == MoeGemvConfig::kCtaN16 ? kCtaN16 : kDefaultCtaN;
}

// Opt-in interleaved MXFP4 GEMV path (env ORT_FP4_GEMV_INTERLEAVED=1). See moe_gemv_fp4.h.
// Parsed once via ORT's environment helper for consistent parsing/thread-safety. Off by
// default so the shipped single-pass ColumnMajor path stays byte-for-byte unchanged.
bool Fp4MoeGemvUseInterleaved() {
  const static bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED", 0) == 1;
  return enabled;
}

// Override for the interleaved path (env ORT_FP4_GEMV_INTERLEAVED_HALFACC=1). When set, the
// interleaved GEMV forces 16-bit (AccT=TypeA) accumulation for BOTH fp16 and bf16, overriding the
// default dtype-conditional Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum). Forcing
// 16-bit on bf16 regresses bf16 accuracy, so this override is off by default.
bool Fp4MoeGemvInterleavedHalfAccum() {
  const static bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED_HALFACC", 0) == 1;
  return enabled;
}

// Interleaved-path tiling: a smaller CtaN than the default (8) to recover the occupancy the
// interleaved layout + fp32 accumulation cost. CtaN must be even (kernel static_assert) and a
// divisor that keeps n % (CtaN*kInterleave) == 0 for the target shapes. kInterleave = 4 here.
static constexpr int kInterleavedCtaN = 4;
static constexpr int kInterleavedThreads = 128;

// Opt-out for the shape-derived default tiling (env ORT_FP4_GEMV_DEFAULT_TILING=0), which
// restores the fixed kDefaultCtaN/kDefaultThreads tiling for every shape.
static bool Fp4MoeGemvUseDefaultTilingHeuristic() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_DEFAULT_TILING", 1) == 1;
  return enabled;
}

// Blocks per SM required before a 64-thread block is worth it. The grid is fixed by
// (expanded_num_rows, n / CtaN) and does not depend on Threads, so halving the block size halves
// the threads each block contributes to residency. These kernels are register-limited (about 72
// registers/thread on sm_90, so roughly 900 resident threads/SM), which a 128-thread block reaches
// with ~7 resident blocks and a 64-thread block only with ~14. Requiring 16 blocks/SM therefore
// keeps the SM just as full while halving the epilogue's reduction width.
static constexpr int64_t kMinBlocksPerSmForThreads64 = 16;

MoeGemvConfig Fp4MoeGemvDefaultConfig(int64_t expanded_num_rows, int64_t n, int64_t k,
                                      int multi_processor_count) {
  // The interleaved path pins its own CtaN/Threads and ignores `config`.
  if (Fp4MoeGemvUseInterleaved() || !Fp4MoeGemvUseDefaultTilingHeuristic()) {
    return MoeGemvConfig::kDefault;
  }
  // Each block walks K in strides of CtaK = StepK * Threads, with StepK = 128 / activation_bits.
  constexpr int64_t kStepK = 128 / 16;
  constexpr int64_t kDefaultCtaK = kStepK * kDefaultThreads;

  // (a) Idle threads. When CtaK > k the tail of every block never enters the K loop at all, so a
  //     128-thread block leaves half its threads doing nothing (NVFP4 fc2 has k = inter_size,
  //     e.g. 512 against CtaK = 1024). Narrowing the block is a pure win here.
  if (kDefaultCtaK > k) {
    return MoeGemvConfig::kThreads64;
  }

  // (b) Epilogue cost. The MAC work per block is fixed by (CtaN, k) regardless of Threads, but the
  //     epilogue reduces partial sums across Threads/32 warps through shared memory. A narrower
  //     block does the same math with half the barriers and a shallower reduction tree, so prefer
  //     it whenever the grid is large enough that the SM still fills up (see the constant above).
  const int64_t blocks = expanded_num_rows * (n / kDefaultCtaN);
  if (blocks >= kMinBlocksPerSmForThreads64 * multi_processor_count) {
    return MoeGemvConfig::kThreads64;
  }
  return MoeGemvConfig::kDefault;
}

// --- MXFP4 (e2m1) GEMV: non-interleaved ColumnMajor layout (kInterleave = 1) ---
// Weights are the QMoERepackFP4ColToRow output ([experts, n, k/2] row-major, two e2m1
// codes per byte, even-K in the low nibble). Block scales are the
// LaunchQMoECombineFp4ScalesForGemv output ([experts, k/32, n] in TypeA, already folded with
// the per-expert global scale). The kernel decodes e2m1 in-register via Fp4I2FConverter.
template <typename T>
struct Fp4ADetails;
template <>
struct Fp4ADetails<half> {
  using Type = fiv::FP16DetailsA;
};
#ifdef ENABLE_BF16
template <>
struct Fp4ADetails<__nv_bfloat16> {
  using Type = fiv::BF16DetailsA;
};
#endif

// TileSizeK is unused by the ColumnMajor (kInterleave = 1) indexing/reduction beyond the
// shmem-write lane gating (only lane 0 of each warp writes), so 64 matches the INT convention.
static constexpr int kTileSizeKFp4 = 64;
template <typename T>
using Fp4KernelDetails =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajor, false, kTileSizeKFp4>;

// Interleaved Details. ColumnMajorInterleaved with TileSizeK = 64 and kElemBits = 4
// gives kInterleave = 128*8/(64*4) = 4, kStepK = 128/4 = 32, kThreadsPerInterleavedTile =
// 64/32 = 2. The linear Fp4I2FConverter is reused (UseInterleavedConverter = false): the
// preprocessor's layout-only steps 1-3 produce exactly the nibble order the linear converter
// expects. Weights for this layout are produced by the interleaved branch of
// QMoE::PrePackRepackFP4Weights (CUTLASS fpA_intB W4_A16 preprocessor, apply_bias_interleave=false).
template <typename T>
using Fp4KernelDetailsInterleaved =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajorInterleaved, false,
                       kTileSizeKFp4>;

// Same interleaved Details, but with UseInterleavedConverter = true so the e2m1 decoder inverts
// the `[e0,e2,e4,e6,e1,e3,e5,e7]` nibble pair-interleave on the fly. That is preprocessor step 4
// (interleave_without_bias), i.e. exactly the layout the SM80 grouped GEMM consumes. Selecting
// this variant lets the decode GEMV read the *same* pre-packed buffer the SM80 prefill uses, so
// the MXFP4 expert weights are stored once instead of twice (~9 GiB saved for gpt-oss-20b).
// Each word is un-permuted with a few register ops before the shared packed decode.
template <typename T>
using Fp4KernelDetailsSm80Pair =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajorInterleaved, true,
                       kTileSizeKFp4>;

// Carries a Details type into a generic lambda so the interleaved and pair-interleaved launch
// bodies can be shared instead of duplicated.
template <typename U>
struct TypeTag {
  using type = U;
};

// Interleaved-path accumulation policy (dtype-conditional). fp16 has a 10-bit mantissa, so 16-bit
// (half) accumulation over the interleaved kStepK=32 chains stays within tolerance and keeps
// register use low. bf16 has only 7 mantissa bits, so 16-bit accumulation loses too much precision
// (bf16 fails tolerance) and must accumulate in fp32. The ORT_FP4_GEMV_INTERLEAVED_HALFACC override
// forces 16-bit accum for BOTH dtypes.
template <typename T>
using Fp4GemvAccT = std::conditional_t<std::is_same<T, half>::value, half, float>;

template <bool NativeDecode = false>
__device__ __forceinline__ float DecodeE4M3Fn(uint8_t code) {
#if defined(ORT_CUDA_TYPE_HELPER_HAS_FP8) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (NativeDecode) {
    return onnxruntime::contrib::cuda::e4m3_to_float(code);
  }
#endif
  const int sign = code & 0x80;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if ((code & 0x7f) == 0) {
    return sign ? -0.0f : 0.0f;
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return __int_as_float(0x7fffffff);
  }
  const float value = exponent == 0
                          ? static_cast<float>(mantissa) * 0.001953125f
                          : __int_as_float(((exponent + 120) << 23) | (mantissa << 20));
  return sign ? -value : value;
}

template <typename T, bool FusedSwiGlu, bool EnableBias>
__device__ __forceinline__ void StoreMoeGemvFp4Pair(
    float acc0, float acc1, const T* expert_bias, T* out, int row, int column, int output_columns,
    int expert, cutlass_kernels::ActivationParams activation_params) {
  if constexpr (EnableBias) {
    acc0 += static_cast<float>(expert_bias[column]);
    acc1 += static_cast<float>(expert_bias[column + 1]);
  }
  if constexpr (FusedSwiGlu) {
    const float* alpha = activation_params.swiglu_alpha;
    const float* beta = activation_params.swiglu_beta;
    const float* limit = activation_params.swiglu_limit;
    const float activation_alpha = alpha ? alpha[expert] : activation_params.alpha;
    const float activation_beta = beta ? beta[expert] : activation_params.beta;
    const float activation_limit = limit ? limit[expert] : activation_params.limit;
    if (isfinite(activation_limit)) {
      acc0 = fminf(acc0, activation_limit);
      acc1 = fminf(fmaxf(acc1, -activation_limit), activation_limit);
    }
    acc1 += activation_beta;
    const float sigmoid = 1.0f / (1.0f + expf(-activation_alpha * acc0));
    out[static_cast<int64_t>(row) * (output_columns / 2) + column / 2] = static_cast<T>(acc0 * sigmoid * acc1);
  } else {
    out[static_cast<int64_t>(row) * output_columns + column] = static_cast<T>(acc0);
    out[static_cast<int64_t>(row) * output_columns + column + 1] = static_cast<T>(acc1);
  }
}

__device__ __forceinline__ float DecodeE8M0(uint8_t code) {
  if (code == 0xff) {
    return __int_as_float(0x7fffffff);
  }
  return ldexpf(1.0f, static_cast<int>(code) - 127);
}

// NVFP4/MXFP4 schema weights are [E, K, N/2], with adjacent output columns in the two nibbles of
// each byte. Transpose an N16/K1024 tile in shared memory so 64 independent K groups can
// reuse each block scale for 16 values (or 32 values for MXFP4). Padding distributes compute-time reads across banks.
template <typename T, bool FusedSwiGlu, bool EnableBias, bool IsMxfp4 = false>
__global__ void MoeGemvFp4RawNPackedKernel(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kColsPerThread = 8;
  constexpr int kNLanes = 2;
  constexpr int kKLanes = kWarpsPerBlock * kWarpSize / kNLanes;
  constexpr unsigned kFullMask = 0xffffffffu;

  const int row = static_cast<int>(blockIdx.x);
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int n_lane = lane % kNLanes;
  const int k_lane = threadIdx.x / kNLanes;
  const int n_base = (static_cast<int>(blockIdx.y) * kNLanes + n_lane) * kColsPerThread;
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;

  const T* row_act = act + static_cast<int64_t>(source_row) * k;
  const uint8_t* expert_weight = weight + static_cast<int64_t>(expert) * weight_expert_stride;
  const uint8_t* expert_scales = block_scales + static_cast<int64_t>(expert) * scale_expert_stride;
  const float global_scale = global_scales[expert];
  const T* expert_bias = EnableBias ? bias + static_cast<int64_t>(expert) * n : nullptr;

  __shared__ uint32_t weight_tile[kKLanes][kNLanes][17];
  float accumulators[kColsPerThread] = {};
  for (int k_tile = 0; k_tile < k; k_tile += kKLanes * 16) {
    for (int tile_row = threadIdx.x; tile_row < kKLanes * 16; tile_row += kWarpSize * kWarpsPerBlock) {
      const int tile_n = static_cast<int>(blockIdx.y) * kNLanes * kColsPerThread;
      uint2 packed = {};
      if (k_tile + tile_row < k) {
        const int reduction_column = k_tile + tile_row;
        const uint8_t* weights = expert_weight + static_cast<int64_t>(reduction_column) * (n / 2);
        if (n % (kNLanes * kColsPerThread) == 0) {
          packed = *reinterpret_cast<const uint2*>(weights + tile_n / 2);
        } else {
#pragma unroll
          for (int vector = 0; vector < kNLanes; ++vector) {
            uint32_t word = 0;
#pragma unroll
            for (int pair = 0; pair < kColsPerThread / 2; ++pair) {
              const int column = tile_n + vector * kColsPerThread + pair * 2;
              if (column < n) {
                word |= static_cast<uint32_t>(weights[column / 2]) << (pair * 8);
              }
            }
            reinterpret_cast<uint32_t*>(&packed)[vector] = word;
          }
        }
      }
      weight_tile[tile_row / 16][0][tile_row % 16] = packed.x;
      weight_tile[tile_row / 16][1][tile_row % 16] = packed.y;
    }
    __syncthreads();
    const int k_base = k_tile + k_lane * 16;
    if (k_base < k) {
      alignas(4) T scales[kColsPerThread];
#pragma unroll
      for (int col = 0; col < kColsPerThread; ++col) {
        if constexpr (IsMxfp4) {
          scales[col] = n_base + col < n
                            ? static_cast<T>(DecodeE8M0(expert_scales[static_cast<int64_t>(n_base + col) *
                                                                          (k / 32) +
                                                                      k_base / 32]) *
                                             global_scale)
                            : static_cast<T>(0.0f);
        } else {
          scales[col] = n_base + col < n
                            ? static_cast<T>(DecodeE4M3Fn(expert_scales[static_cast<int64_t>(n_base + col) *
                                                                            (k / 16) +
                                                                        k_base / 16]) *
                                             global_scale)
                            : static_cast<T>(0.0f);
        }
      }
      alignas(16) T activations[16];
      reinterpret_cast<uint4*>(activations)[0] = reinterpret_cast<const uint4*>(row_act + k_base)[0];
      reinterpret_cast<uint4*>(activations)[1] = reinterpret_cast<const uint4*>(row_act + k_base)[1];
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        uint32_t packed = weight_tile[k_lane][n_lane][element];
        const float activation = static_cast<float>(activations[element]);
        alignas(4) T decoded[kColsPerThread];
        fiv::Fp4I2FConverter<T>::template convert<kColsPerThread>(&packed, decoded);
        using PackedT = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
#pragma unroll
        for (int pair = 0; pair < kColsPerThread / 2; ++pair) {
          reinterpret_cast<PackedT*>(decoded)[pair] =
              __hmul2(reinterpret_cast<const PackedT*>(decoded)[pair],
                      reinterpret_cast<const PackedT*>(scales)[pair]);
        }
#pragma unroll
        for (int col = 0; col < kColsPerThread; ++col) {
          accumulators[col] += static_cast<float>(decoded[col]) * activation;
        }
      }
    }
    __syncthreads();
  }

  __shared__ float partials[kWarpsPerBlock][kColsPerThread][kNLanes];
#pragma unroll
  for (int col = 0; col < kColsPerThread; ++col) {
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 2);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 4);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 8);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 16);
    if (lane < kNLanes) {
      partials[warp][col][n_lane] = accumulators[col];
    }
  }
  __syncthreads();
  if (k_lane != 0 || n_base >= n) {
    return;
  }

#pragma unroll
  for (int col = 0; col < kColsPerThread; col += 2) {
    const int n0 = n_base + col;
    if (n0 >= n) {
      continue;
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
#pragma unroll
    for (int partial = 0; partial < kWarpsPerBlock; ++partial) {
      acc0 += partials[partial][col][n_lane];
      acc1 += partials[partial][col + 1][n_lane];
    }
    StoreMoeGemvFp4Pair<T, FusedSwiGlu, EnableBias>(
        acc0, acc1, expert_bias, out, row, n0, n, expert, activation_params);
  }
#endif
}

template <typename T, bool FusedSwiGlu, bool EnableBias, int KLanes = 8, bool ReuseExpertRows = false, bool NativeScale = false, bool IsMxfp4 = false>
__global__ void MoeGemvFp4RawKPackedKernel(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int output_columns, int reduction_size,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kKLanes = KLanes;
  constexpr int kBlockSize = 16;
  constexpr int kColsPerBlock = 128 / kKLanes;
  static_assert(kKLanes == 4 || kKLanes == 8);
  constexpr unsigned kFullMask = 0xffffffffu;
  using PackedT = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;

  const int row = static_cast<int>(blockIdx.x);
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int expert_index = 0; expert_index < num_experts && permuted_row_to_expert == nullptr; ++expert_index) {
    if (row >= static_cast<int>(expert_first_token_offset[expert_index + 1])) {
      expert = expert_index + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int column = static_cast<int>(blockIdx.y) * kColsPerBlock + threadIdx.x / kKLanes;
  bool second_row = false;
  if constexpr (ReuseExpertRows) {
    const int first = static_cast<int>(expert_first_token_offset[expert]);
    const int end = static_cast<int>(expert_first_token_offset[expert + 1]);
    if (end - first >= 4) {
      if ((row - first) % 2 != 0) {
        return;
      }
      second_row = row + 1 < end;
    }
  }
  const int k_lane = threadIdx.x % kKLanes;
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;
  const T* row_act = act + static_cast<int64_t>(source_row) * reduction_size;
  const int second_source_row = second_row
                                    ? (permuted_row_to_source_row ? permuted_row_to_source_row[row + 1] % num_rows : row + 1)
                                    : source_row;
  const T* second_act = act + static_cast<int64_t>(second_source_row) * reduction_size;
  const uint8_t* expert_weight = weight + static_cast<int64_t>(expert) * weight_expert_stride;
  const uint8_t* expert_scales = block_scales + static_cast<int64_t>(expert) * scale_expert_stride;
  const float global_scale = global_scales[expert];
  const T* expert_bias = EnableBias ? bias + static_cast<int64_t>(expert) * output_columns : nullptr;

  float accumulators[ReuseExpertRows ? 2 : 1][2] = {};
  for (int reduction_column = k_lane * kBlockSize;
       column < output_columns && reduction_column < reduction_size;
       reduction_column += kKLanes * kBlockSize) {
    uint2 packed = *reinterpret_cast<const uint2*>(
        expert_weight + static_cast<int64_t>(column) * (reduction_size / 2) + reduction_column / 2);
    constexpr int kScaleGroupSize = IsMxfp4 ? 32 : kBlockSize;
    const uint8_t scale_code =
        expert_scales[static_cast<int64_t>(column) * (reduction_size / kScaleGroupSize) +
                      reduction_column / kScaleGroupSize];
    const T scale = static_cast<T>((IsMxfp4 ? DecodeE8M0(scale_code) : DecodeE4M3Fn<NativeScale>(scale_code)) *
                                   global_scale);
    alignas(4) T scale_pair[2] = {scale, scale};
    alignas(16) T activations[kBlockSize];
    reinterpret_cast<uint4*>(activations)[0] = reinterpret_cast<const uint4*>(row_act + reduction_column)[0];
    reinterpret_cast<uint4*>(activations)[1] = reinterpret_cast<const uint4*>(row_act + reduction_column)[1];
    alignas(16) T second_activations[kBlockSize];
    if constexpr (ReuseExpertRows) {
      if (second_row) {
        reinterpret_cast<uint4*>(second_activations)[0] = reinterpret_cast<const uint4*>(second_act + reduction_column)[0];
        reinterpret_cast<uint4*>(second_activations)[1] = reinterpret_cast<const uint4*>(second_act + reduction_column)[1];
      }
    }
    alignas(4) T decoded[kBlockSize];
    fiv::Fp4I2FConverter<T>::template convert<kBlockSize>(&packed, decoded);
#pragma unroll
    for (int pair = 0; pair < kBlockSize / 2; ++pair) {
      reinterpret_cast<PackedT*>(decoded)[pair] =
          __hmul2(reinterpret_cast<const PackedT*>(decoded)[pair], *reinterpret_cast<const PackedT*>(scale_pair));
      accumulators[0][0] += static_cast<float>(decoded[pair * 2]) * static_cast<float>(activations[pair * 2]);
      accumulators[0][1] += static_cast<float>(decoded[pair * 2 + 1]) * static_cast<float>(activations[pair * 2 + 1]);
      if constexpr (ReuseExpertRows) {
        if (second_row) {
          accumulators[1][0] += static_cast<float>(decoded[pair * 2]) * static_cast<float>(second_activations[pair * 2]);
          accumulators[1][1] += static_cast<float>(decoded[pair * 2 + 1]) * static_cast<float>(second_activations[pair * 2 + 1]);
        }
      }
    }
  }

  float accumulator = accumulators[0][0] + accumulators[0][1];
#pragma unroll
  for (int offset = kKLanes / 2; offset > 0; offset /= 2) {
    accumulator += __shfl_xor_sync(kFullMask, accumulator, offset, kKLanes);
  }
  const float paired_accumulator = __shfl_xor_sync(kFullMask, accumulator, kKLanes);
  if (k_lane == 0 && column < output_columns && column % 2 == 0) {
    StoreMoeGemvFp4Pair<T, FusedSwiGlu, EnableBias>(
        accumulator, paired_accumulator, expert_bias, out, row, column, output_columns, expert, activation_params);
  }
  if constexpr (ReuseExpertRows) {
    if (second_row) {
      float second_accumulator = accumulators[1][0] + accumulators[1][1];
#pragma unroll
      for (int offset = kKLanes / 2; offset > 0; offset /= 2) {
        second_accumulator += __shfl_xor_sync(kFullMask, second_accumulator, offset, kKLanes);
      }
      const float second_paired = __shfl_xor_sync(kFullMask, second_accumulator, kKLanes);
      if (k_lane == 0 && column < output_columns && column % 2 == 0) {
        StoreMoeGemvFp4Pair<T, FusedSwiGlu, EnableBias>(
            second_accumulator, second_paired, expert_bias, out, row + 1, column, output_columns, expert, activation_params);
      }
    }
  }
#endif
}

bool Fp4MoeGemvSupportsNativeScale() {
#if defined(ORT_CUDA_TYPE_HELPER_HAS_FP8)
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  int sm_major = 0;
  if (cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || sm_major < 9) {
    (void)cudaGetLastError();
    return false;
  }
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(
          &attributes, MoeGemvFp4RawKPackedKernel<half, false, false, 4, false, true>) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attributes.binaryVersion >= 90 && attributes.ptxVersion >= 90;
#else
  return false;
#endif
}

bool Fp4MoeGemvKernelUsesNativeScale(const void* kernel) {
  return kernel == reinterpret_cast<const void*>(MoeGemvFp4RawKPackedKernel<half, false, false, 4, false, true>) ||
         kernel == reinterpret_cast<const void*>(MoeGemvFp4RawKPackedKernel<half, false, false, 8, false, true>) ||
         kernel == reinterpret_cast<const void*>(MoeGemvFp4RawKPackedKernel<__nv_bfloat16, false, false, 4, false, true>) ||
         kernel == reinterpret_cast<const void*>(MoeGemvFp4RawKPackedKernel<__nv_bfloat16, false, false, 8, false, true>);
}

// ---------------------------------------------------------------------------------------------
// Decode-specialised NVFP4 GEMV for the row-major [E, N, K/2] layout (small expanded row counts).
//
// MoeGemvFp4RawKPackedKernel above issues 8-byte weight loads and re-reads/decodes the activation row
// for every 16-element block, which leaves the decode shapes issue/latency bound. This kernel streams the
// weights straight from global memory into registers with 128-bit loads, keeps several K tiles in flight,
// decodes FP4 in registers and stages the activation row once per CTA in shared memory as FP32.
//
// It reproduces MoeGemvFp4RawKPackedKernel's floating-point summation order exactly, so the results are
// bit-identical: K lane L = block (16 elements) % KLanes keeps separate even/odd-element accumulators over
// its blocks, a lane total is acc_even + acc_odd, and lane totals are combined by an xor tree over
// KLanes / 2, ..., 1. Here one thread owns the lane pair (2j, 2j + 1) of two adjacent columns: one 16-byte
// load covers the two blocks, the xor-tree steps over lane distances KLanes / 2 .. 2 become shuffles
// between the KLanes / 2 threads of a column, and the last step (lane distance 1) is an in-thread add.
template <typename T>
__device__ __forceinline__ float PackedLowToFloat(uint32_t word) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(word & 0xffffu)));
  } else {
    return __uint_as_float(word << 16);
  }
}

template <typename T>
__device__ __forceinline__ float PackedHighToFloat(uint32_t word) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(word >> 16)));
  } else {
    return __uint_as_float(word & 0xffff0000u);
  }
}

__device__ __forceinline__ uint4 LoadWeightStreaming(const void* ptr) {
  uint4 value;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(ptr));
  return value;
}

// The activation row is staged as FP32 in shared memory and must fit the default 48 KiB limit.
static constexpr int kDecodeMaxK = 8192;
// Default enablement (ORT_FP4_GEMV_DECODE_KERNEL=1) is restricted to the H200-measured decode envelope:
// SM90, up to 20 expanded rows, N and K up to 2560, and verified table-free scale eligibility
// (|global_scale| * 448 < 3.98). Set ORT_FP4_GEMV_DECODE_KERNEL=2 to opt into the full kernel envelope
// (SM90+, up to 80 expanded rows, K <= 8192, any even N, and the generic Fp4I2FConverter scale fallback).
static constexpr int kDecodeDefaultMaxN = 2560;
static constexpr int kDecodeDefaultMaxK = 2560;
static constexpr int64_t kDecodeDefaultMaxExpandedRows = 20;
// Tiling of the two KLanes variants (KLanes = 8 for K > 1024, 4 otherwise): warps per CTA, K tiles per
// prefetch batch and the min-blocks-per-SM hint that caps registers. Tuned on H200 for the Qwen3.8 FC1
// (N=1280, K=2560, fused SwiGLU) and FC2 (N=2560, K=640) shapes at 10 expanded rows.
static constexpr int kDecodeK8Warps = 2;
static constexpr int kDecodeK8Batch = 2;
static constexpr int kDecodeK8MinBlocks = 2;
static constexpr int kDecodeK4Warps = 2;
static constexpr int kDecodeK4Batch = 2;
static constexpr int kDecodeK4MinBlocks = 4;
// Rows beyond this use the KPacked kernel, which amortizes less per-row work as the grid grows.
static constexpr int64_t kDecodeMaxExpandedRows = 80;

// Table-free decode: place the three e2m1 magnitude bits (e1 e0 m) at the lowest exponent bits and the
// top mantissa bit of T and the sign at bit 15. The result is fp4_value * 2^-126 for bf16 and
// fp4_value * 2^-14 for fp16 (denormal for e == 0), so multiplying by block_scale * 2^126 (resp. 2^14)
// gives exactly T(fp4_value * block_scale) with a single rounding. Returns the pairs {nibble j, nibble
// j + 4} for j = 0..3.
template <typename T>
__device__ __forceinline__ void Fp4WordToRaw(uint32_t word, uint32_t (&raw)[4]) {
  if constexpr (std::is_same_v<T, half>) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t nibbles = word >> (4 * j);
      raw[j] = ((nibbles & 0x00070007u) << 9) | ((nibbles & 0x00080008u) << 12);
    }
  } else {
    const uint32_t upper = word >> 8;
    raw[0] = ((word & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
    raw[1] = ((word & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
    raw[2] = ((upper & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
    raw[3] = ((upper & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
  }
}

// {2^126, 2^126} in bf16 and {2^14, 2^14} in fp16.
// A function rather than a variable template: Windows nvcc rejects __device__ variable templates with const type.
template <typename T>
__device__ __forceinline__ constexpr uint32_t Fp4RawRescale() {
  return std::is_same_v<T, half> ? 0x74007400u : 0x7E807E80u;
}

// Activation unit (32 elements = one lane pair's K slice of a tile) stride in shared memory: 36 floats =
// 9 float4 chunks, so the threads of a warp that read different units hit distinct banks.
static constexpr int kDecodeActUnitStride = 36;
// Every thread owns two adjacent columns, so the SwiGLU pair (n, n + 1) stays inside a thread.
static constexpr int kDecodeColsPerThread = 2;

template <typename T>
using PackedPairT = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;

// Accumulates the 32 products of one weight vector per column. chain[c][lane * 2 + parity] is the
// accumulator of the lane (0: elements 0..15, 1: elements 16..31) and element parity. `decode(c, w, pairs)`
// yields the four packed pairs {element j, element j + 4} of 32-bit weight word w of column c; each
// activation chunk is read from shared memory once for all columns.
template <typename T, typename DecodeFn>
__device__ __forceinline__ void FmaWeightVector(const float* act, float (&chain)[kDecodeColsPerThread][4],
                                                DecodeFn decode) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t decoded_pairs[kDecodeColsPerThread][4];
#pragma unroll
    for (int c = 0; c < kDecodeColsPerThread; ++c) {
      decode(c, w, decoded_pairs[c]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      // Elements 8w + 4h .. 8w + 4h + 3 are float4 chunk 2w + h of the unit.
      const float4 a = *reinterpret_cast<const float4*>(act + (2 * w + h) * 4);
      const float activations[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int c = 0; c < kDecodeColsPerThread; ++c) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float decoded = h == 0 ? PackedLowToFloat<T>(decoded_pairs[c][i])
                                       : PackedHighToFloat<T>(decoded_pairs[c][i]);
          chain[c][(w / 2) * 2 + (i & 1)] += decoded * activations[i];
        }
      }
    }
  }
}

// Block scales of the two blocks of one vector: scale_rounded = T(e4m3 * global_scale), packed
// {block A, block B} in T.
template <typename T>
__device__ __forceinline__ uint32_t RoundedScalePair(uint32_t scale_code, float global_scale) {
  uint32_t half_pair;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half_pair) : "h"(static_cast<uint16_t>(scale_code)));
  const float scale_a = __half2float(__ushort_as_half(static_cast<uint16_t>(half_pair & 0xffffu))) * global_scale;
  const float scale_b = __half2float(__ushort_as_half(static_cast<uint16_t>(half_pair >> 16))) * global_scale;
  if constexpr (std::is_same_v<T, half>) {
    const half2 rounded = __floats2half2_rn(scale_a, scale_b);
    return *reinterpret_cast<const uint32_t*>(&rounded);
  } else {
    const __nv_bfloat162 rounded = __floats2bfloat162_rn(scale_a, scale_b);
    return *reinterpret_cast<const uint32_t*>(&rounded);
  }
}

// Operands of one weight vector per column, passed by value to the out-of-line generic decode so the
// caller's register-resident arrays are never address-taken.
struct DecodeVectorState {
  uint4 packed[kDecodeColsPerThread];
  uint32_t scale_codes[kDecodeColsPerThread];
  float chain[kDecodeColsPerThread][4];
};

// Generic decode (any T, any block scale): the Fp4I2FConverter path of the KPacked kernel.
template <typename T>
__device__ __noinline__ DecodeVectorState AccumulateWeightVectorGeneric(DecodeVectorState state, float global_scale,
                                                                        const float* act) {
  uint32_t scale_rounded[kDecodeColsPerThread];
#pragma unroll
  for (int c = 0; c < kDecodeColsPerThread; ++c) {
    scale_rounded[c] = RoundedScalePair<T>(state.scale_codes[c], global_scale);
  }
  FmaWeightVector<T>(act, state.chain, [&](int c, int w, uint32_t (&decoded_pairs)[4]) {
    uint32_t word_copy = (&state.packed[c].x)[w];
    alignas(4) T decoded[8];
    fiv::Fp4I2FConverter<T>::template convert<8>(&word_copy, decoded);
    const uint32_t scale_splat = __byte_perm(scale_rounded[c], scale_rounded[c], (w / 2) ? 0x3232 : 0x1010);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      reinterpret_cast<PackedPairT<T>*>(decoded)[p] = __hmul2(reinterpret_cast<const PackedPairT<T>*>(decoded)[p],
                                                              *reinterpret_cast<const PackedPairT<T>*>(&scale_splat));
    }
    const uint16_t* bits = reinterpret_cast<const uint16_t*>(decoded);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      decoded_pairs[j] = static_cast<uint32_t>(bits[j]) | (static_cast<uint32_t>(bits[j + 4]) << 16);
    }
  });
  return state;
}

// Fast path: valid when every block scale is below 4 (|global_scale| * 448 < 3.98), so that the rescaled
// block scale (block_scale * 2^126 in bf16, * 2^14 in fp16) stays finite.
template <typename T>
__device__ __forceinline__ void AccumulateWeightVector(
    const uint4 (&packed)[kDecodeColsPerThread], const uint32_t (&scale_codes)[kDecodeColsPerThread],
    float global_scale, bool fast_scales, const float* act, float (&chain)[kDecodeColsPerThread][4]) {
  if (fast_scales) {
    uint32_t scale_raw[kDecodeColsPerThread][2];
#pragma unroll
    for (int c = 0; c < kDecodeColsPerThread; ++c) {
      const uint32_t rounded_bits = RoundedScalePair<T>(scale_codes[c], global_scale);
      const uint32_t raw_rescale = Fp4RawRescale<T>();
      const PackedPairT<T> shifted = __hmul2(*reinterpret_cast<const PackedPairT<T>*>(&rounded_bits),
                                             *reinterpret_cast<const PackedPairT<T>*>(&raw_rescale));
      const uint32_t shifted_bits = *reinterpret_cast<const uint32_t*>(&shifted);
      scale_raw[c][0] = __byte_perm(shifted_bits, shifted_bits, 0x1010);
      scale_raw[c][1] = __byte_perm(shifted_bits, shifted_bits, 0x3232);
    }
    FmaWeightVector<T>(act, chain, [&](int c, int w, uint32_t (&decoded_pairs)[4]) {
      uint32_t raw[4];
      Fp4WordToRaw<T>((&packed[c].x)[w], raw);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const PackedPairT<T> product = __hmul2(*reinterpret_cast<const PackedPairT<T>*>(&raw[j]),
                                               *reinterpret_cast<const PackedPairT<T>*>(&scale_raw[c][w / 2]));
        decoded_pairs[j] = *reinterpret_cast<const uint32_t*>(&product);
      }
    });
    return;
  }
  DecodeVectorState state;
#pragma unroll
  for (int c = 0; c < kDecodeColsPerThread; ++c) {
    state.packed[c] = packed[c];
    state.scale_codes[c] = scale_codes[c];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      state.chain[c][q] = chain[c][q];
    }
  }
  state = AccumulateWeightVectorGeneric<T>(state, global_scale, act);
#pragma unroll
  for (int c = 0; c < kDecodeColsPerThread; ++c) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      chain[c][q] = state.chain[c][q];
    }
  }
}

// A CTA covers Warps * 32 / (KLanes / 2) * 2 adjacent columns of one expanded row. Each thread loads
// `Batch` K tiles (KLanes blocks of 16 elements per column) at a time and prefetches the next batch while
// it accumulates the current one.
template <typename T, bool FusedSwiGlu, bool EnableBias, int KLanes, int Warps, int Batch, int MinBlocks>
__global__ void __launch_bounds__(Warps * 32, MinBlocks) MoeGemvFp4DecodeKernel(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int kThreadsPerColumn = KLanes / 2;
  constexpr int kThreads = Warps * 32;
  constexpr int kColsPerCta = kThreads / kThreadsPerColumn * kDecodeColsPerThread;
  constexpr int kTileBytes = KLanes * 8;  // weight bytes of one column and tile
  constexpr unsigned kFullMask = 0xffffffffu;
  static_assert(KLanes == 4 || KLanes == 8);
  static_assert(Batch >= 1);

  // Programmatic dependent launch (only active when launched with the PDL attribute, otherwise no-ops): let the next
  // kernel of the MoE chain start its prologue while this one runs. FC2 reads only data of kernels that completed
  // before FC1 passed its wait (expert maps, weights) before its own wait and the FC1 output after it.
  // FC1 triggers only after its own wait (the routing kernel may be its primary grid), so FC2 and the finalize
  // kernel, which read routing metadata before their waits, never launch before that metadata is complete.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (FusedSwiGlu) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  const int row = static_cast<int>(blockIdx.x);
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int expert_index = 0; expert_index < num_experts && permuted_row_to_expert == nullptr; ++expert_index) {
    if (row >= static_cast<int>(expert_first_token_offset[expert_index + 1])) {
      expert = expert_index + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int j = threadIdx.x % kThreadsPerColumn;
  const int first_column = static_cast<int>(blockIdx.y) * kColsPerCta +
                           (static_cast<int>(threadIdx.x) / kThreadsPerColumn) * kDecodeColsPerThread;
  const bool column_valid = first_column < n;  // n is even, so both columns of a thread are valid or neither
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;
  const int blocks = k / 16;
  const int tiles = (blocks + KLanes - 1) / KLanes;

  const uint8_t* weight_row = weight + static_cast<int64_t>(expert) * weight_expert_stride +
                              static_cast<int64_t>(first_column) * (k / 2) + j * 16;
  const uint8_t* scale_row = block_scales + static_cast<int64_t>(expert) * scale_expert_stride +
                             static_cast<int64_t>(first_column) * blocks + j * 2;

  // Threads whose lane pair lies past the end of K in a tile load nothing and skip the tile.
  const auto tile_valid = [&](int tile) { return column_valid && tile * KLanes + j * 2 < blocks; };
  const auto load_batch = [&](int first_tile, uint4(&packed)[Batch][kDecodeColsPerThread],
                              uint32_t (&scale_codes)[Batch][kDecodeColsPerThread]) {
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const int tile = first_tile + b;
      const bool valid = tile < tiles && tile_valid(tile);
#pragma unroll
      for (int c = 0; c < kDecodeColsPerThread; ++c) {
        packed[b][c] = valid ? LoadWeightStreaming(weight_row + static_cast<int64_t>(c) * (k / 2) + tile * kTileBytes)
                             : make_uint4(0, 0, 0, 0);
        scale_codes[b][c] =
            valid ? *reinterpret_cast<const uint16_t*>(scale_row + static_cast<int64_t>(c) * blocks + tile * KLanes)
                  : 0u;
      }
    }
  };

  uint4 packed[Batch][kDecodeColsPerThread];
  uint32_t scale_codes[Batch][kDecodeColsPerThread];
  load_batch(0, packed, scale_codes);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (!FusedSwiGlu) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // the activation row is the FC1 output
  }
#endif
  // Stage the activation row as FP32, one padded 32-element unit per lane pair.
  extern __shared__ __align__(16) float act_smem[];
  {
    const uint4* source = reinterpret_cast<const uint4*>(act + static_cast<int64_t>(source_row) * k);
    for (int index = threadIdx.x; index < k / 8; index += kThreads) {
      const uint4 raw = source[index];
      const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
      float values[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[2 * i] = PackedLowToFloat<T>(words[i]);
        values[2 * i + 1] = PackedHighToFloat<T>(words[i]);
      }
      float* destination = act_smem + (index / 4) * kDecodeActUnitStride + (index % 4) * 8;
      *reinterpret_cast<float4*>(destination) = make_float4(values[0], values[1], values[2], values[3]);
      *reinterpret_cast<float4*>(destination + 4) = make_float4(values[4], values[5], values[6], values[7]);
    }
  }
  const float global_scale = global_scales[expert];
  const bool fast_scales = fabsf(global_scale) * 448.0f < 3.98f;
  __syncthreads();

  float chain[kDecodeColsPerThread][4] = {};
#pragma unroll 1
  for (int first_tile = 0; first_tile < tiles; first_tile += Batch) {
    uint4 current_packed[Batch][kDecodeColsPerThread];
    uint32_t current_codes[Batch][kDecodeColsPerThread];
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
#pragma unroll
      for (int c = 0; c < kDecodeColsPerThread; ++c) {
        current_packed[b][c] = packed[b][c];
        current_codes[b][c] = scale_codes[b][c];
      }
    }
    if (first_tile + Batch < tiles) {
      load_batch(first_tile + Batch, packed, scale_codes);
    }
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const int tile = first_tile + b;
      if (tile < tiles && tile_valid(tile)) {
        const float* unit_act = act_smem + (tile * (KLanes / 2) + j) * kDecodeActUnitStride;
        AccumulateWeightVector<T>(current_packed[b], current_codes[b], global_scale, fast_scales, unit_act, chain);
      }
    }
  }

  // Lane totals (acc_even + acc_odd) of the two lanes of the pair, then the xor tree over lane distances
  // KLanes / 2 .. 2 as shuffles between the threads of a column.
  float lane_total[kDecodeColsPerThread][2];
#pragma unroll
  for (int c = 0; c < kDecodeColsPerThread; ++c) {
    lane_total[c][0] = chain[c][0] + chain[c][1];
    lane_total[c][1] = chain[c][2] + chain[c][3];
#pragma unroll
    for (int offset = kThreadsPerColumn / 2; offset > 0; offset /= 2) {
      lane_total[c][0] += __shfl_xor_sync(kFullMask, lane_total[c][0], offset);
      lane_total[c][1] += __shfl_xor_sync(kFullMask, lane_total[c][1], offset);
    }
  }
  if (j == 0 && column_valid) {
    const T* expert_bias = EnableBias ? bias + static_cast<int64_t>(expert) * n : nullptr;
    StoreMoeGemvFp4Pair<T, FusedSwiGlu, EnableBias>(
        lane_total[0][0] + lane_total[0][1], lane_total[1][0] + lane_total[1][1], expert_bias, out, row,
        first_column, n, expert, activation_params);
  }
#endif
}

// Opt-out for the programmatic dependent launch of FC2 (env ORT_FP4_GEMV_PDL=0).
static bool Fp4MoeGemvUsePdl() {
  static const bool enabled = onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_PDL", 1) != 0;
  return enabled;
}

template <typename KernelPtr>
static bool QuerySm90DecodeKernelCaps(KernelPtr kernel, bool* pdl_supported = nullptr) {
  if (pdl_supported != nullptr) {
    *pdl_supported = false;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  int sm_major = 0;
  if (cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || sm_major < 9) {
    (void)cudaGetLastError();
    return false;
  }
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  const bool image_available = attributes.ptxVersion >= 90 && attributes.binaryVersion >= 90;
  if (!image_available) {
    return false;
  }
  if (pdl_supported != nullptr) {
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
      *pdl_supported = driver_version >= 12000;
    } else {
      (void)cudaGetLastError();
    }
  }
  return true;
}

template <typename T, bool FusedSwiGlu>
void LaunchMoeGemvFp4RawNPacked(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t n, int64_t k, cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream, bool weights_row_major, int sm,
    bool decode_scale_eligible = false, bool* decode_kernel_launched = nullptr,
    bool programmatic_dependent_launch = false, int group_size = 16) {
  if (decode_kernel_launched != nullptr) {
    *decode_kernel_launched = false;
  }
  constexpr int kThreads = 128;
  constexpr int kColsPerBlock = 16;
  const int64_t weight_expert_stride = n * k / 2;
  const int64_t scale_expert_stride = n * (k / group_size);
  const dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>((n + kColsPerBlock - 1) / kColsPerBlock));
  if (weights_row_major) {
    static const bool enable_expert_reuse =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_NVFP4_FC1_EXPERT_REUSE", 0) == 1;
    // Decode kernel mode (env ORT_FP4_GEMV_DECODE_KERNEL):
    //   0: disabled (always use MoeGemvFp4RawKPackedKernel)
    //   1: default (measured SM90 envelope with verified table-free scale eligibility)
    //   2: opt-in full kernel envelope (SM90+, up to 80 expanded rows, K <= 8192, including generic scale fallback)
    static const int decode_mode =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_DECODE_KERNEL", 1);
    const bool reuse_experts = enable_expert_reuse && FusedSwiGlu && n == 1280 && k == 2560 &&
                               expanded_num_rows >= 40 && expert_first_token_offset != nullptr &&
                               permuted_row_to_expert != nullptr;
    static const bool native_scale =
        onnxruntime::ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false);
    static const bool native_scale_supported = native_scale && Fp4MoeGemvSupportsNativeScale();
    const bool allow_extended_decode_envelope = decode_mode >= 2;
    // When ORT_NVFP4_GEMV_NATIVE_SCALE=1 is enabled on SM90, the 4-lane native-scale KPacked kernel
    // (MoeGemvFp4RawKPackedKernel<half, ..., 4, false, true>) is slightly faster than MoeGemvFp4DecodeKernel
    // for FP16 at K <= 1024 (e.g. Qwen3.8 FC2 K=640: 8.03 vs 8.41 us at top-10, 14.17 vs 14.23 us at top-20)
    // while producing bit-identical results.
    const bool prefer_native_kpacked_fp16 =
        std::is_same_v<T, half> && native_scale_supported && k <= 1024;
    const bool decode_envelope_ok =
        group_size == 16 &&
        (allow_extended_decode_envelope
             ? (sm >= 90)
             : (sm == 90 && decode_scale_eligible && !prefer_native_kpacked_fp16 &&
                expanded_num_rows <= kDecodeDefaultMaxExpandedRows &&
                n <= kDecodeDefaultMaxN && k <= kDecodeDefaultMaxK));
    // Decode kernel: same summation order and KLanes dispatch as the KPacked kernel below (bit-identical).
    // It steps aside for the expert-reuse variant and for shapes/alignments/scales outside the active envelope.
    if (!reuse_experts && decode_mode != 0 && decode_envelope_ok && expanded_num_rows <= kDecodeMaxExpandedRows &&
        k % 32 == 0 && k <= kDecodeMaxK && n % 2 == 0 && reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(block_scales) % 2 == 0 && reinterpret_cast<uintptr_t>(act) % 16 == 0) {
      const size_t decode_smem = static_cast<size_t>(k / 32) * kDecodeActUnitStride * sizeof(float);
      const auto launch_decode = [&](auto k_lanes, auto warps_tag, auto batch_tag, auto min_blocks_tag) {
        constexpr int kKLanes = decltype(k_lanes)::value;
        constexpr int kWarps = decltype(warps_tag)::value;
        constexpr int kBatch = decltype(batch_tag)::value;
        constexpr int kMinBlocks = decltype(min_blocks_tag)::value;
        constexpr int kBlockCols = kWarps * 32 / (kKLanes / 2) * kDecodeColsPerThread;
        const dim3 decode_grid(static_cast<unsigned>(expanded_num_rows),
                               static_cast<unsigned>((n + kBlockCols - 1) / kBlockCols));
        const auto launch_bias = [&](auto bias_tag) {
          constexpr bool kEnableBias = decltype(bias_tag)::value;
          auto kernel = MoeGemvFp4DecodeKernel<T, FusedSwiGlu, kEnableBias, kKLanes, kWarps, kBatch, kMinBlocks>;
          // PTX compiled below SM90 has an empty decode body, even when JIT-loaded on SM90+.
          // Image selection depends on the device SM; refresh when a thread switches architectures.
          static thread_local int cached_sm = -1;
          static thread_local bool available = false;
          static thread_local bool pdl_supported = false;
          if (cached_sm != sm) {
            available = QuerySm90DecodeKernelCaps(kernel, &pdl_supported);
            cached_sm = sm;
          }
          if (!available) {
            return false;
          }
          if (programmatic_dependent_launch && Fp4MoeGemvUsePdl() && pdl_supported) {
            cudaLaunchConfig_t config{};
            config.gridDim = decode_grid;
            config.blockDim = dim3(kWarps * 32);
            config.dynamicSmemBytes = decode_smem;
            config.stream = stream;
            cudaLaunchAttribute attribute[1]{};
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            config.attrs = attribute;
            config.numAttrs = 1;
            const cudaError_t status = cudaLaunchKernelEx(
                &config, kernel, act, weight, block_scales, global_scales, bias, out, expert_first_token_offset,
                permuted_row_to_expert, num_experts, weight_expert_stride, scale_expert_stride, static_cast<int>(n),
                static_cast<int>(k), activation_params, permuted_row_to_source_row, static_cast<int>(num_rows));
            if (status == cudaSuccess) {
              return true;
            }
            if (status != cudaErrorInvalidValue && status != cudaErrorNotSupported) {
              ORT_ENFORCE(status == cudaSuccess, "FP4 GEMV decode launch failed: ", cudaGetErrorString(status));
            }
            (void)cudaGetLastError();
            pdl_supported = false;
          }
          kernel<<<decode_grid, kWarps * 32, decode_smem, stream>>>(
              act, weight, block_scales, global_scales, bias, out,
              expert_first_token_offset, permuted_row_to_expert, num_experts,
              weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k),
              activation_params, permuted_row_to_source_row, static_cast<int>(num_rows));
          CUDA_CALL_THROW(cudaGetLastError());
          return true;
        };
        if (bias != nullptr) {
          return launch_bias(std::true_type{});
        } else {
          return launch_bias(std::false_type{});
        }
      };
      bool launched;
      if (k <= 1024) {
        launched = launch_decode(std::integral_constant<int, 4>{}, std::integral_constant<int, kDecodeK4Warps>{},
                                 std::integral_constant<int, kDecodeK4Batch>{}, std::integral_constant<int, kDecodeK4MinBlocks>{});
      } else {
        launched = launch_decode(std::integral_constant<int, 8>{}, std::integral_constant<int, kDecodeK8Warps>{},
                                 std::integral_constant<int, kDecodeK8Batch>{}, std::integral_constant<int, kDecodeK8MinBlocks>{});
      }
      if (launched) {
        if (decode_kernel_launched != nullptr) {
          *decode_kernel_launched = true;
        }
        return;
      }
    }
    const auto launch_k_packed = [&](auto k_lanes, auto reuse_rows, auto native_tag, auto mxfp4_tag) {
      constexpr int kKLanes = decltype(k_lanes)::value;
      constexpr bool kReuseRows = decltype(reuse_rows)::value;
      constexpr bool kNativeScale = decltype(native_tag)::value;
      constexpr bool kIsMxfp4 = decltype(mxfp4_tag)::value;
      constexpr int kNativeColsPerBlock = kThreads / kKLanes;
      const dim3 native_grid(static_cast<unsigned>(expanded_num_rows),
                             static_cast<unsigned>((n + kNativeColsPerBlock - 1) / kNativeColsPerBlock));
      if (bias != nullptr) {
        MoeGemvFp4RawKPackedKernel<T, FusedSwiGlu, true, kKLanes, kReuseRows, kNativeScale, kIsMxfp4><<<native_grid, kThreads, 0, stream>>>(
            act, weight, block_scales, global_scales, bias, out,
            expert_first_token_offset, permuted_row_to_expert, num_experts,
            weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
            permuted_row_to_source_row, static_cast<int>(num_rows));
      } else {
        MoeGemvFp4RawKPackedKernel<T, FusedSwiGlu, false, kKLanes, kReuseRows, kNativeScale, kIsMxfp4><<<native_grid, kThreads, 0, stream>>>(
            act, weight, block_scales, global_scales, bias, out,
            expert_first_token_offset, permuted_row_to_expert, num_experts,
            weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
            permuted_row_to_source_row, static_cast<int>(num_rows));
      }
    };
    const auto launch = [&](auto native_tag, auto mxfp4_tag) {
      if (k <= 1024) {
        launch_k_packed(std::integral_constant<int, 4>{}, std::false_type{}, native_tag, mxfp4_tag);
      } else if (reuse_experts) {
        launch_k_packed(std::integral_constant<int, 8>{}, std::true_type{}, native_tag, mxfp4_tag);
      } else {
        launch_k_packed(std::integral_constant<int, 8>{}, std::false_type{}, native_tag, mxfp4_tag);
      }
    };
    if (group_size == 32) {
      launch(std::false_type{}, std::true_type{});
    } else if (native_scale) {
      launch(std::true_type{}, std::false_type{});
    } else {
      launch(std::false_type{}, std::false_type{});
    }
    return;
  }
  const auto launch_n_packed = [&](auto mxfp4_tag) {
    constexpr bool kIsMxfp4 = decltype(mxfp4_tag)::value;
    if (bias != nullptr) {
      MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, true, kIsMxfp4><<<grid, kThreads, 0, stream>>>(
          act, weight, block_scales, global_scales, bias, out,
          expert_first_token_offset, permuted_row_to_expert, num_experts,
          weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
          permuted_row_to_source_row, static_cast<int>(num_rows));
    } else {
      MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, false, kIsMxfp4><<<grid, kThreads, 0, stream>>>(
          act, weight, block_scales, global_scales, bias, out,
          expert_first_token_offset, permuted_row_to_expert, num_experts,
          weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
          permuted_row_to_source_row, static_cast<int>(num_rows));
    }
  };
  if (group_size == 32) {
    launch_n_packed(std::true_type{});
  } else {
    launch_n_packed(std::false_type{});
  }
}

// MXFP4 GEMV shape support. Mirrors is_moe_gemv_supported but for the non-interleaved
// ColumnMajor layout: kInterleave = 1, so n need only be divisible by the CtaN tile width
// selected by `config`, and the per-thread step is StepK = 128 / activation_bits = 8
// (not 128 / weight_bits).
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed,
                               bool weights_row_major) {
  if (sm < 80) {
    return false;
  }
  if (group_size != 16 && group_size != 32) {  // 32 = MXFP4 block size, 16 = NVFP4 block size
    return false;
  }
  if (k % group_size != 0) {
    return false;
  }
  const int64_t max_expanded_rows =
      raw_n_packed && weights_row_major ? kMaxProfiledExpandedRowsNvfp4KPacked : kMaxProfiledExpandedRowsFp4;
  if (expanded_num_rows <= 0 || expanded_num_rows > max_expanded_rows) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  if (raw_n_packed) {
    return group_size == 16 && n % 2 == 0;
  }
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
    // Interleaved path: ColumnMajorInterleaved (kInterleave = 4, kStepK = 32), fixed CtaN =
    // kInterleavedCtaN. Each block covers CtaN*kInterleave columns, and a complete interleaved
    // K-tile is kStepK*kThreadsPerInterleavedTile = 32*2 = 64 wide, so require
    // n % (CtaN*4) == 0 and k % 64 == 0. (gpt-oss-20b fc1 n=5760/k=2880 and fc2 n=2880/k=2880
    // both satisfy this.) `config` is ignored in this mode: CtaN/Threads are pinned to keep the
    // prepacked weight layout and the kernel dispatch in agreement.
    // The interleaved kStepK=32 tile is tied to the MXFP4 block-32 scale layout, so it only supports
    // group_size == 32; NVFP4 (block 16) must use the non-interleaved ColumnMajor path below.
    if (group_size != 32) {
      return false;
    }
    if (n % (kInterleavedCtaN * 4) != 0) {
      return false;
    }
    if (k % 64 != 0) {
      return false;
    }
    return true;
  }
  if (n % CtaNForConfig(config) != 0) {  // kInterleave = 1
    return false;
  }
  // StepK = 128 / activation_bits = 8; is_moe_gemv_fp4_supported requires k % group_size == 0 with
  // group_size >= 16 (16 for NVFP4, 32 for MXFP4), so k % 8 == 0 always holds.
  if (k % (128 / 16) != 0) {
    return false;
  }
  return true;
}

bool is_moe_gemv_fp4_sm80_layout_supported(int64_t n, int64_t k, int group_size) {
  // Same constraints the interleaved branch of is_moe_gemv_fp4_supported enforces, minus the
  // row-count bounds (which PrePack cannot know): the interleaved kStepK=32 tile is tied to the
  // MXFP4 block-32 scale layout, each block covers kInterleavedCtaN*kInterleave columns, and a
  // complete interleaved K-tile is kStepK*kThreadsPerInterleavedTile = 64 wide.
  return group_size == 32 && n % (kInterleavedCtaN * 4) == 0 && k % 64 == 0;
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, config,
                                   /*sm80_pair_interleaved=*/false, /*raw_n_packed=*/false);
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, MoeGemvConfig::kDefault);
}

bool is_moe_gemv_fp4_decode_image_available(int sm, bool is_fp16) {
  if (sm < 90) {
    return false;
  }
  const auto check_type = [&](auto type_tag) {
    using T = typename decltype(type_tag)::type;
    return QuerySm90DecodeKernelCaps(
        MoeGemvFp4DecodeKernel<T, true, false, 8, kDecodeK8Warps, kDecodeK8Batch, kDecodeK8MinBlocks>);
  };
  if (is_fp16) {
    return check_type(TypeTag<half>{});
  }
#ifdef ENABLE_BF16
  return check_type(TypeTag<__nv_bfloat16>{});
#else
  return false;
#endif
}

template <typename T>
void launch_moe_gemv_fp4_symmetric(const T* act, const uint8_t* weight, const T* scales,
                                   const uint8_t* raw_block_scales, const float* raw_global_scales,
                                   const T* bias, T* out,
                                   const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                   int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                                   int sm, MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed,
                                   cudaStream_t stream, bool weights_row_major, bool decode_scale_eligible,
                                   bool* decode_kernel_launched, bool programmatic_dependent_launch) {
  if (decode_kernel_launched != nullptr) {
    *decode_kernel_launched = false;
  }
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16 || group_size == 32, "Raw FP4 GEMV requires group_size 16 (NVFP4) or 32 (MXFP4).");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw FP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, false>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, cutlass_kernels::ActivationParams{}, nullptr, expanded_num_rows, stream,
        weights_row_major, sm, decode_scale_eligible, decode_kernel_launched, programmatic_dependent_launch,
        group_size);
    return;
  }
  // Interleaved path: ColumnMajorInterleaved layout + dtype-conditional accumulation + smaller
  // CtaN. Taken either via the opt-in env knob or because the caller pre-packed a single
  // SM80-grouped-GEMM buffer (sm80_pair_interleaved) that this kernel un-permutes while decoding.
  // The prepacked fc2 weights are in the interleaved layout, so the kernel must match.
  // CtaN/Threads are pinned (config ignored) so weights and kernel always agree. AccT follows the
  // Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum); HALFACC forces 16-bit for both.
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
    auto launch_interleaved = [&](auto details_tag) {
      using DetailsI = typename decltype(details_tag)::type;
      if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
        fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size,
            stream);
      } else {
        fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, Fp4GemvAccT<T>>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size,
            stream);
      }
    };
    if (sm80_pair_interleaved) {
      launch_interleaved(TypeTag<Fp4KernelDetailsSm80Pair<T>>{});
    } else {
      launch_interleaved(TypeTag<Fp4KernelDetailsInterleaved<T>>{});
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum): bf16 has only 7
  // mantissa bits, so 16-bit accumulation over K loses too much precision and fails tolerance
  // (e.g. NVFP4 block-16 decode at k=512). CtaN/Threads remain pure parallelization/tiling knobs
  // and the accumulation dtype is identical for every config, so every config computes the same
  // dot products; Threads additionally sets the K partition, so it perturbs the summation order.
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_group_size<Details, cta_n(), threads(), T, Fp4GemvAccT<T>>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    const T* act, const uint8_t* weight, const T* scales, const uint8_t* raw_block_scales,
    const float* raw_global_scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, bool sm80_pair_interleaved,
    bool raw_n_packed,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream, bool weights_row_major,
    bool decode_scale_eligible, bool* decode_kernel_launched, bool programmatic_dependent_launch) {
  if (decode_kernel_launched != nullptr) {
    *decode_kernel_launched = false;
  }
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16 || group_size == 32, "Raw FP4 GEMV requires group_size 16 (NVFP4) or 32 (MXFP4).");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw FP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, true>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size * 2, k, activation_params, permuted_row_to_source_row, num_rows, stream,
        weights_row_major, sm, decode_scale_eligible, decode_kernel_launched, programmatic_dependent_launch,
        group_size);
    return;
  }
  // Interleaved path: ColumnMajorInterleaved layout + dtype-conditional accumulation + smaller
  // CtaN, fusing SwiGLU. Taken either via the opt-in env knob or because the caller pre-packed a
  // single SM80-grouped-GEMM buffer (sm80_pair_interleaved). SwiGLU fusion is orthogonal to the
  // weight layout: it only concerns the fc1 output-column (gate/value) order, so both variants
  // fuse it identically. Takes precedence over the split-K path so the kernel matches the
  // prepacked fc1 weights. CtaN/Threads pinned (config ignored). AccT follows the Fp4GemvAccT
  // policy: fp16->fp16 accum since fp16's mantissa tolerates it; bf16->fp32 accum since 16-bit
  // accum fails bf16 tolerance. HALFACC forces 16-bit for both dtypes.
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
    auto launch_interleaved = [&](auto details_tag) {
      using DetailsI = typename decltype(details_tag)::type;
      if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
        fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k,
            group_size, activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
      } else {
        fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T,
                                                             Fp4GemvAccT<T>>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k,
            group_size, activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
      }
    };
    if (sm80_pair_interleaved) {
      launch_interleaved(TypeTag<Fp4KernelDetailsSm80Pair<T>>{});
    } else {
      launch_interleaved(TypeTag<Fp4KernelDetailsInterleaved<T>>{});
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16, bf16->fp32); see launch_moe_gemv_fp4_symmetric.
  // The accumulation dtype is fixed across the CtaN/Threads sweep; Threads still changes the K
  // partition, so it is a tiling knob, not a bit-exact one.
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), T, Fp4GemvAccT<T>>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template void launch_moe_gemv_fp4_symmetric<half>(
    const half*, const uint8_t*, const half*, const uint8_t*, const float*, const half*, half*,
    const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t, bool, bool, bool*, bool);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
    const half*, const uint8_t*, const half*, const uint8_t*, const float*, const half*, half*,
    const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, bool, bool,
    const int*, int64_t, cudaStream_t, bool, bool, bool*, bool);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t,
    bool, bool, bool*, bool);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, bool, bool, const int*, int64_t, cudaStream_t, bool, bool, bool*, bool);
#endif

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
