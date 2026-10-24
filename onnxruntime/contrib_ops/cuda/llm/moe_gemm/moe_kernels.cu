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

#include <algorithm>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <cmath>
#include <type_traits>
#include <limits>
#include <random>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/detail/collective/mixed_input_utils.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue/thread/fused_activations.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cu_inc/cub.cuh"
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/data_type.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/kernels/pre_quant_scale_kernel.h"
#include "contrib_ops/cuda/llm/kernels/quantization.cuh"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_activation_kernels.cuh"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_utils.cuh"
#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
#include "contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.h"
#endif

#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

using namespace onnxruntime::llm::kernels;
using namespace onnxruntime::llm::common;

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Master switch for the symmetric INT MoE GEMV fast path. Enabled by default;
// set ORT_DISABLE_MOE_GEMV=1 to fall back to the CUTLASS grouped GEMM.
inline bool MoeGemvDisabledByEnv() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV", 0) == 1;
  return disabled;
}

inline bool MoeGemvSplitK2SwiGLUDisabledByEnv() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU", 0) == 1;
  return disabled;
}

// Kill switch for folding finalizeMoeRouting into the FC2 GEMV epilogue. Enabled by default;
// set ORT_DISABLE_MOE_GEMV_FUSED_FINALIZE=1 to run FC2 GEMV + a separate finalize kernel.
inline bool MoeGemvFusedFinalizeDisabledByEnv() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV_FUSED_FINALIZE", 0) == 1;
  return disabled;
}

// Kill switch for skipping expandInputRows when the FC1 GEMV can read the unexpanded activations
// through permuted_row_to_unpermuted_row. Enabled by default; set ORT_DISABLE_MOE_GEMV_SKIP_EXPAND=1
// to always materialize the expanded activation buffer.
inline bool MoeGemvSkipExpandDisabledByEnv() {
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV_SKIP_EXPAND", 0) == 1;
  return disabled;
}

// Kill switch for folding softmax + top-k routing into the expert-map prologue. Enabled by default;
// set ORT_DISABLE_MOE_FUSED_ROUTING=1 to run a standalone SoftmaxTopK kernel before runMoe.
inline bool MoeFusedRoutingDisabledByEnv() {
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_FUSED_ROUTING", 0) == 1;
  return disabled;
}

template <typename T, typename WeightType, typename ScaleBiasType>
constexpr bool MoeGemvSplitK2SwiGLUSupported() {
  return std::is_same_v<T, half> && std::is_same_v<WeightType, cutlass::uint4b_t> &&
         std::is_same_v<ScaleBiasType, T>;
}

inline bool MoeGemvRejectedByProfiledInterSize(int64_t expanded_num_rows, int64_t inter_size) {
  return expanded_num_rows > onnxruntime::llm::kernels::moe_gemv::kMaxProfiledExpandedRowsForSmallProblemDim &&
         inter_size < onnxruntime::llm::kernels::moe_gemv::kMinProfiledProblemDimForExpandedRowsAbove4;
}

// Single source of truth for the caller-supplied `disabled` term of the FC1 MoE GEMV.
//
// gemm1 uses it to decide whether to launch the GEMV, and runMoe must predict that same decision
// before gemm1 runs in order to skip expandInputRows. Two hand-written copies of this expression
// had already drifted (runMoe was missing the !bias_is_broadcast term), and a drift here turns the
// ORT_ENFORCE in gemm1 into a hard runtime failure rather than a lost optimization.
inline bool MoeGemvFc1Disabled(int ep_size, bool use_ampere_activation_fusion, bool bias_is_broadcast,
                               int64_t expanded_num_rows, int64_t inter_size) {
  return ep_size > 1 || use_ampere_activation_fusion || !bias_is_broadcast ||
         MoeGemvRejectedByProfiledInterSize(expanded_num_rows, inter_size);
}

template <typename WeightType>
constexpr int MoeGemvWeightBits() {
  if constexpr (std::is_same_v<WeightType, cutlass::uint4b_t>) {
    return 4;
  } else if constexpr (std::is_same_v<WeightType, uint8_t>) {
    return 8;
  } else {
    return 0;
  }
}

// Attempts the batched symmetric INT MoE GEMV. Returns true if it ran (output
// written), false if the configuration is unsupported and the caller must fall
// back to the grouped GEMM. Compiles to a no-op (returns false) for type
// combinations other than (half activations, int4/int8 weights, ScaleBias==T).
template <typename T, typename WeightType, typename ScaleBiasType>
bool tryLaunchMoeGemvIntSymmetric(const T* input, const WeightType* weights, const ScaleBiasType* scales,
                                  const ScaleBiasType* weight_zeros, const ScaleBiasType* biases, T* output,
                                  const int64_t* expert_first_token_offset, int num_experts_per_node,
                                  const int* permuted_row_to_expert, int64_t expanded_num_rows,
                                  int64_t n, int64_t k, int sm, int group_size,
                                  bool disabled, cudaStream_t stream) {
  if constexpr ((std::is_same_v<WeightType, cutlass::uint4b_t> || std::is_same_v<WeightType, uint8_t>) &&
                (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && std::is_same_v<ScaleBiasType, T>) {
    const bool env_disabled = MoeGemvDisabledByEnv();
    const bool has_block_zeros = group_size > 0 && weight_zeros != nullptr;
    constexpr int weight_bits = MoeGemvWeightBits<WeightType>();
    if (disabled || env_disabled || has_block_zeros) {
      return false;
    }
    if (!onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_supported(
            sm, expanded_num_rows, n, k, weight_bits, group_size)) {
      return false;
    }
    onnxruntime::llm::kernels::moe_gemv::launch_moe_gemv_int_symmetric<T, WeightType>(
        input, weights, scales, biases, output, expert_first_token_offset, permuted_row_to_expert,
        num_experts_per_node, expanded_num_rows, n, k, group_size, sm, stream);
    return true;
  } else {
    (void)input;
    (void)weights;
    (void)scales;
    (void)weight_zeros;
    (void)biases;
    (void)output;
    (void)expert_first_token_offset;
    (void)num_experts_per_node;
    (void)permuted_row_to_expert;
    (void)expanded_num_rows;
    (void)n;
    (void)k;
    (void)sm;
    (void)group_size;
    (void)disabled;
    (void)stream;
    return false;
  }
}

// Attempts the FC2 symmetric INT MoE GEMV with finalizeMoeRouting fused into its epilogue.
// Returns true if it ran, in which case `final_output` already holds the reduced MoE output and
// the caller must skip finalizeMoeRoutingKernelLauncher. Returns false if the configuration is
// unsupported, leaving the caller to run the unfused GEMV/GEMM plus a separate finalize.
template <typename T, typename WeightType, typename ScaleBiasType, typename OutputType>
bool tryLaunchMoeGemvIntSymmetricFusedFinalize(
    const T* input, const WeightType* weights, const ScaleBiasType* scales, const ScaleBiasType* weight_zeros,
    const ScaleBiasType* biases, OutputType* final_output, const int* unpermuted_row_to_permuted_row,
    const int* permuted_row_to_expert, const float* unpermuted_final_scales, int num_experts_per_node,
    int64_t num_rows, int64_t experts_per_token, int64_t expanded_num_rows, int64_t n, int64_t k, int sm,
    int group_size, bool disabled, cudaStream_t stream) {
  if constexpr ((std::is_same_v<WeightType, cutlass::uint4b_t> || std::is_same_v<WeightType, uint8_t>) &&
                (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && std::is_same_v<ScaleBiasType, T> &&
                std::is_same_v<OutputType, T>) {
    const bool has_block_zeros = group_size > 0 && weight_zeros != nullptr;
    constexpr int weight_bits = MoeGemvWeightBits<WeightType>();
    if (disabled || MoeGemvDisabledByEnv() || MoeGemvFusedFinalizeDisabledByEnv() || has_block_zeros) {
      return false;
    }
    // The fused epilogue resolves each partial's expert through permuted_row_to_expert; there is no
    // expert_first_token_offset scan fallback in that kernel.
    if (permuted_row_to_expert == nullptr) {
      return false;
    }
    if (!onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_fused_finalize_supported(
            sm, num_rows, experts_per_token, expanded_num_rows, n, k, weight_bits, group_size)) {
      return false;
    }
    onnxruntime::llm::kernels::moe_gemv::launch_moe_gemv_int_symmetric_fused_finalize<T, WeightType>(
        input, weights, scales, biases, final_output, unpermuted_row_to_permuted_row, permuted_row_to_expert,
        unpermuted_final_scales, num_experts_per_node, num_rows, experts_per_token, n, k, group_size, sm, stream);
    return true;
  } else {
    (void)input;
    (void)weights;
    (void)scales;
    (void)weight_zeros;
    (void)biases;
    (void)final_output;
    (void)unpermuted_row_to_permuted_row;
    (void)permuted_row_to_expert;
    (void)unpermuted_final_scales;
    (void)num_experts_per_node;
    (void)num_rows;
    (void)experts_per_token;
    (void)expanded_num_rows;
    (void)n;
    (void)k;
    (void)sm;
    (void)group_size;
    (void)disabled;
    (void)stream;
    return false;
  }
}

// Single source of truth for "will the FC1 interleaved-SwiGLU MoE GEMV run?".
//
// tryLaunchMoeGemvIntSymmetricInterleavedSwiGLU uses it to decide whether to launch, and runMoe
// uses it *before* gemm1 to decide whether expandInputRows can be skipped (the GEMV can read the
// unexpanded activations through permuted_row_to_unpermuted_row, the grouped GEMM cannot). Keeping
// one predicate makes it impossible for the two decisions to drift apart.
template <typename T, typename WeightType, typename ScaleBiasType>
bool moeGemvInterleavedSwiGLUWillRun(const ScaleBiasType* weight_zeros, int64_t expanded_num_rows,
                                     int64_t inter_size, int64_t k, int sm, int group_size, bool disabled,
                                     cutlass_kernels::ActivationParams activation_params) {
  if constexpr ((std::is_same_v<WeightType, cutlass::uint4b_t> || std::is_same_v<WeightType, uint8_t>) &&
                (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && std::is_same_v<ScaleBiasType, T>) {
    const bool has_block_zeros = group_size > 0 && weight_zeros != nullptr;
    if (disabled || MoeGemvDisabledByEnv() || has_block_zeros) {
      return false;
    }
    if (activation_params.swiglu_fusion != 1) {
      return false;
    }
    if (activation_params.activation_type != ActivationType::Swiglu &&
        activation_params.activation_type != ActivationType::SwigluBias) {
      return false;
    }
    constexpr int weight_bits = MoeGemvWeightBits<WeightType>();
    return onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_supported(sm, expanded_num_rows, inter_size * 2, k,
                                                                      weight_bits, group_size);
  } else {
    (void)weight_zeros;
    (void)expanded_num_rows;
    (void)inter_size;
    (void)k;
    (void)sm;
    (void)group_size;
    (void)disabled;
    (void)activation_params;
    return false;
  }
}

template <typename T, typename WeightType, typename ScaleBiasType>
bool tryLaunchMoeGemvIntSymmetricInterleavedSwiGLU(
    const T* input, const WeightType* weights, const ScaleBiasType* scales, const ScaleBiasType* weight_zeros,
    const ScaleBiasType* biases, T* output,
    const int64_t* expert_first_token_offset, int num_experts_per_node, const int* permuted_row_to_expert,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int sm, int group_size,
    bool disabled, cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, float* splitk_partials, cudaStream_t stream) {
  if constexpr ((std::is_same_v<WeightType, cutlass::uint4b_t> || std::is_same_v<WeightType, uint8_t>) &&
                (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && std::is_same_v<ScaleBiasType, T>) {
    if (!moeGemvInterleavedSwiGLUWillRun<T, WeightType, ScaleBiasType>(
            weight_zeros, expanded_num_rows, inter_size, k, sm, group_size, disabled, activation_params)) {
      return false;
    }
    onnxruntime::llm::kernels::moe_gemv::launch_moe_gemv_int_symmetric_interleaved_swiglu<T, WeightType>(
        input, weights, scales, biases, output, expert_first_token_offset, permuted_row_to_expert,
        num_experts_per_node, expanded_num_rows, inter_size, k, group_size, sm, activation_params,
        permuted_row_to_source_row, num_rows, splitk_partials, stream);
    return true;
  } else {
    (void)input;
    (void)weights;
    (void)scales;
    (void)weight_zeros;
    (void)biases;
    (void)output;
    (void)expert_first_token_offset;
    (void)num_experts_per_node;
    (void)permuted_row_to_expert;
    (void)expanded_num_rows;
    (void)inter_size;
    (void)k;
    (void)sm;
    (void)group_size;
    (void)disabled;
    (void)activation_params;
    (void)permuted_row_to_source_row;
    (void)num_rows;
    (void)splitk_partials;
    (void)stream;
    return false;
  }
}

// Ranker used to group the (token, top-k slot) pairs by selected expert.
//
// cub::BlockRadixRank keeps a private set of packed digit counters per thread, so its
// shared memory and its per-thread scan both grow with 2^LOG2_NUM_EXPERTS. At 256
// experts (LOG2_NUM_EXPERTS == 9) and a decode-sized block of 32 threads that is 512
// bins spread over a single warp, which costs ~16KB of shared memory, 255 registers per
// thread and enough spilling that the kernel is dominated by counter bookkeeping rather
// than by the handful of assignments it actually ranks. The match-based ranker ranks
// with warp ballots instead of per-thread counters, so its cost tracks the number of
// keys rather than the number of bins.
template <int BLOCK_SIZE, int LOG2_NUM_EXPERTS>
using MoeExpertRadixRank = cub::BlockRadixRankMatch<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(const int* const token_selected_experts,
                                                         int* const permuted_row_to_unpermuted_row, int* const unpermuted_row_to_permuted_row,
                                                         int* const permuted_token_selected_experts,
                                                         int64_t* const expert_first_token_offset, const int64_t num_tokens, const int experts_per_token,
                                                         const int start_expert, const int end_expert, const int num_experts_per_node) {
  // Only using block wise collective so we can only have one block
  assert(gridDim.x == 1);

  assert(start_expert <= end_expert);
  assert(num_experts_per_node == (end_expert - start_expert));
  assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

  const int token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  bool is_valid_token = token < num_tokens;

  // This is the masked expert id for this token
  int local_token_selected_experts[EXPERTS_PER_TOKEN];
  // This is the final permuted rank of this token (ranked by selected expert)
  int local_token_permuted_indices[EXPERTS_PER_TOKEN];

  // Wait PDL before reading token_selected_experts
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

// build expert map
// we need to populate expert ids for all threads, even if there are
// fewer tokens
#pragma unroll
  for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
    const int expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i] : num_experts_per_node;

    // If the token is not valid, set the expert id to num_experts_per_node + 1
    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0, num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    local_token_selected_experts[i] = !is_valid_token   ? num_experts_per_node + 1
                                      : is_valid_expert ? (expert - start_expert)
                                                        : num_experts_per_node;
  }

  // TODO: decompose cub's sort to expose the bucket starts, and just return
  // that to elide the binary search

  // sort the expert map
  using BlockRadixRank = MoeExpertRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS>;
  extern __shared__ unsigned char temp_storage[];
  auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

  // Sanity check that the number of bins do correspond to the number of experts
  static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
  assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

  int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

  cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
  BlockRadixRank(sort_temp).RankKeys(
      local_token_selected_experts, local_token_permuted_indices, extractor, local_expert_first_token_offset);

// We are done with compute, launch the dependent kernels while the stores are in flight
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // write to shared memory and global memory
  if (is_valid_token) {
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
      const int unpermuted_row = i * num_tokens + token;
      const int permuted_row = local_token_permuted_indices[i];
      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
      permuted_token_selected_experts[permuted_row] = local_token_selected_experts[i];
    }
  }

#pragma unroll
  for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++) {
    int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
    if (out_expert_id < num_experts_per_node + 1) {
      expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
    }
  }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenDispatch(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                                cudaStream_t stream) {
  ORT_ENFORCE(num_experts_per_node == (end_expert - start_expert),
              "num_experts_per_node must be equal to end_expert - start_expert");
  const int threads = BLOCK_SIZE;
  const int blocks = (num_tokens + threads - 1) / threads;
  ORT_ENFORCE(blocks == 1, "Current implementation requires single block");

  using BlockRadixRank = MoeExpertRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS>;
  size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = shared_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  auto kernel = &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

  int device = 0;
  int max_smem_per_block = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (shared_size >= static_cast<size_t>(max_smem_per_block)) {
    // This should mean that
    // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
    // wouldn't work.
    return false;
  }

  CUDA_CALL_THROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
  CUDA_CALL_THROW(cudaLaunchKernelEx(&config, kernel, token_selected_experts, permuted_row_to_unpermuted_row,
                                     unpermuted_row_to_permuted_row, permuted_token_selected_experts,
                                     expert_first_token_offset, num_tokens, experts_per_token, start_expert, end_expert,
                                     num_experts_per_node));

  return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                 int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                 const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                                 cudaStream_t stream) {
  const int block_size = num_tokens;
  if (num_tokens > 256) {
    ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Number of tokens ", num_tokens, " is greater than 256, which is not supported for fused moe prologues"));
    return false;
  }

  auto func = &fusedBuildExpertMapsSortFirstTokenDispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  if (block_size > 32 && block_size <= 64) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 64 && block_size <= 128) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 128 && block_size <= 256) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  }

  return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
              permuted_token_selected_experts, expert_first_token_offset, num_tokens, num_experts_per_node,
              experts_per_token, start_expert, end_expert, stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                                 int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                                 int64_t* expert_first_token_offset, const int64_t num_tokens,
                                                 const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                                 cudaStream_t stream) {
  auto func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
  switch (experts_per_token) {
    case 1: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
      break;
    }
    case 2: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<2, LOG2_NUM_EXPERTS>;
      break;
    }
    case 4: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<4, LOG2_NUM_EXPERTS>;
      break;
    }
    case 6: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<6, LOG2_NUM_EXPERTS>;
      break;
    }
    case 8: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<8, LOG2_NUM_EXPERTS>;
      break;
    }
    case 10: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<10, LOG2_NUM_EXPERTS>;
      break;
    }
    default: {
      ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Top-K value ", experts_per_token, " does not have supported fused moe prologues"));
      return false;
    }
  }
  return func(token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
              permuted_token_selected_experts, expert_first_token_offset, num_tokens, num_experts_per_node,
              experts_per_token, start_expert, end_expert, stream);
}

bool fusedBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                        int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                        int64_t* expert_first_token_offset, const int64_t num_tokens,
                                        const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                        cudaStream_t stream) {
  // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e. num_experts_per_node + 2 values
  // This is floor(log2(num_experts_per_node+1)) + 1
  int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
  if (expert_log <= 10) {
    auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>, &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>, &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>, &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>, &fusedBuildExpertMapsSortFirstTokenBlockSize<9>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<10>};

    return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
                                 unpermuted_row_to_permuted_row, permuted_token_selected_experts,
                                 expert_first_token_offset, num_tokens, num_experts_per_node, experts_per_token,
                                 start_expert, end_expert, stream);
  }
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Experts per node ", num_experts_per_node, " does not have supported fused moe prologues"));
  return false;
}

// ---------------------------------------------------------------------------
// Optimization C: fused routing prologue (single token).
//
// At decode the MoE op launches a standalone softmax/top-k kernel and then the expert-map prologue.
// Both are single-block kernels whose cost is launch + drain latency rather than work, so the
// second launch is nearly pure overhead. This kernel does both in one launch using a single warp:
// lane e holds expert e's logit, a warp bitonic sort produces the top-k, and for a single token the
// permutation is simply "the k selected experts ordered by expert id", which every lane can derive
// with shuffles - no CUB block rank and no shared memory.
//
// The softmax/top-k arithmetic uses the very same helpers and warp bitonic sort as
// SoftmaxTopKWarpBitonicKernel in contrib_ops/cuda/moe/qmoe_kernels.cu (shared through
// core/providers/cuda/cu_inc/topk_warp_sort.cuh), so the emitted expert scales are bit-identical.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fusedRoutingBuildExpertMapsSingleTokenKernel(
    const T* router_logits, int* token_selected_experts, float* token_final_scales,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int* permuted_token_selected_experts, int64_t* expert_first_token_offset, const int num_experts,
    const int experts_per_token, const bool normalize_routing_weights) {
  namespace topk = onnxruntime::cuda::topk;
  const int lane = static_cast<int>(threadIdx.x);
  const int k = experts_per_token;

  // Wait PDL before reading router_logits
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const float logit =
      (lane < num_experts) ? static_cast<float>(router_logits[lane]) : topk::kNegativeInfinity;
  const float max_val = topk::WarpReduceMax(logit);
  const float inv_sum =
      topk::SafeInvSum(topk::WarpReduceSum((lane < num_experts) ? expf(logit - max_val) : 0.0f));

  // Sorting by logit is equivalent to sorting by softmax probability (monotonic). After the sort
  // lane r holds the rank-r (score, expert), with ties broken toward the lower expert index.
  float score = logit;
  int index = (lane < num_experts) ? lane : INT_MAX;
  topk::WarpBitonicSortDescending(score, index);

  float prob = (lane < k) ? topk::SoftmaxScale(score, max_val, inv_sum) : 0.0f;
  if (normalize_routing_weights) {
    prob /= topk::TopKNormalizeDenom(normalize_routing_weights, topk::WarpReduceSum(prob));
  }

  // Expanded row i of this token is the rank-i selection, so unpermuted_row == rank (num_tokens is
  // 1 and unpermuted_row = i * num_tokens + token). The permuted order sorts those k rows by expert
  // id, ties broken by rank - the same order cub::BlockRadixRank produces in the general kernel.
  int permuted_row = 0;
  int selected_below_lane = 0;  // number of selected experts with an id below this lane's id
  for (int r = 0; r < k; ++r) {
    const int other = __shfl_sync(0xFFFFFFFF, index, r);
    permuted_row += (other < index || (other == index && r < lane)) ? 1 : 0;
    selected_below_lane += (other < lane) ? 1 : 0;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  if (lane < k) {
    token_selected_experts[lane] = index;
    token_final_scales[lane] = prob;
    permuted_row_to_unpermuted_row[permuted_row] = lane;
    unpermuted_row_to_permuted_row[lane] = permuted_row;
    permuted_token_selected_experts[permuted_row] = index;
  }

  // expert_first_token_offset[e] is the number of selected experts with an id below e, for
  // e in [0, num_experts]. The last entry is the total number of expanded rows.
  if (lane < num_experts) {
    expert_first_token_offset[lane] = selected_below_lane;
  }
  if (lane == 0) {
    expert_first_token_offset[num_experts] = k;
  }
}

bool isFusedMoeRoutingSupported(const int64_t num_tokens, const int num_experts,
                                const int num_experts_per_node, const int experts_per_token,
                                const int ep_size) {
  // One warp, one token: lane e owns expert e, so the whole expert count must fit in a warp, and
  // the permutation shortcut above assumes a single token. Expert parallelism would additionally
  // need the out-of-node masking that the general prologue does.
  return !MoeFusedRoutingDisabledByEnv() && num_tokens == 1 && ep_size == 1 &&
         num_experts == num_experts_per_node && num_experts >= 1 &&
         num_experts <= onnxruntime::cuda::topk::kWarpBitonicMaxSize && experts_per_token >= 1 &&
         experts_per_token <= num_experts;
}

template <typename T>
void launchFusedRoutingBuildExpertMaps(const T* router_logits, int* token_selected_experts,
                                       float* token_final_scales, int* permuted_row_to_unpermuted_row,
                                       int* unpermuted_row_to_permuted_row,
                                       int* permuted_token_selected_experts,
                                       int64_t* expert_first_token_offset, const int num_experts,
                                       const int experts_per_token, const bool normalize_routing_weights,
                                       cudaStream_t stream) {
  // The runner is also instantiated for narrow activation types (fp8/fp4) that can never be a
  // router logit dtype; reject those here so the caller does not have to.
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) {
    cudaLaunchConfig_t config;
    config.gridDim = 1;
    config.blockDim = onnxruntime::cuda::topk::kWarpSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    CUDA_CALL_THROW(cudaLaunchKernelEx(&config, &fusedRoutingBuildExpertMapsSingleTokenKernel<T>, router_logits,
                                       token_selected_experts, token_final_scales, permuted_row_to_unpermuted_row,
                                       unpermuted_row_to_permuted_row, permuted_token_selected_experts,
                                       expert_first_token_offset, num_experts, experts_per_token,
                                       normalize_routing_weights));
  } else {
    (void)router_logits;
    (void)token_selected_experts;
    (void)token_final_scales;
    (void)permuted_row_to_unpermuted_row;
    (void)unpermuted_row_to_permuted_row;
    (void)permuted_token_selected_experts;
    (void)expert_first_token_offset;
    (void)num_experts;
    (void)experts_per_token;
    (void)normalize_routing_weights;
    (void)stream;
    ORT_THROW("Fused MoE routing requires float, half or bfloat16 router logits");
  }
}

int64_t computeNumTokensPerBlock(const int64_t num_tokens, const int64_t num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(const int* token_selected_experts, int* blocked_expert_counts,
                                           int* blocked_row_to_unpermuted_row, const int64_t num_tokens, const int64_t num_experts_per_token,
                                           const int start_expert_id) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // target_expert_id and expert_id are offset by start_expert_id
  const int target_expert_id = blockIdx.x;
  const int block_id = blockIdx.y;
  const int num_blocks_per_seq = gridDim.y;
  const int token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      const int expert_id = token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  const int has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index;
  BlockScan(temp_storage).ExclusiveSum(has_matched, index);

  if (has_matched) {
    blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock + index] = expanded_token_id;
  }
  if (threadIdx.x == kNumTokensPerBlock - 1) {
    blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void blockExpertPrefixSum(const int* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, const int64_t num_tokens, const int64_t num_experts_per_node,
                          const int64_t num_experts_per_token, const int64_t num_tokens_per_block, const int64_t num_blocks_per_seq,
                          const int start_expert_id, cudaStream_t stream) {
  const dim3 blocks(num_experts_per_node, num_blocks_per_seq);
  const dim3 threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = blockExpertPrefixSumKernel<1024>;
  if (num_tokens_per_block <= 32) {
    func = blockExpertPrefixSumKernel<32>;
  } else if (num_tokens_per_block <= 64) {
    func = blockExpertPrefixSumKernel<64>;
  } else if (num_tokens_per_block <= 128) {
    func = blockExpertPrefixSumKernel<128>;
  } else if (num_tokens_per_block <= 256) {
    func = blockExpertPrefixSumKernel<256>;
  } else if (num_tokens_per_block <= 512) {
    func = blockExpertPrefixSumKernel<512>;
  }
  cudaLaunchKernelEx(&config, func, token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                     num_tokens, num_experts_per_token, start_expert_id);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                                 int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_blocks_per_seq,
                                                 const int64_t num_elem_per_thread) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = threadIdx.x * num_elem_per_thread;
  int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Note: Because of limited registers, cannot store thread-level prefix sum or enable #pragma unroll
  for (int i = 0; i < num_elem_per_thread; i++) {
    // TODO(enweiz): Fix uncoalesced access with shared memory.
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      cnt += blocked_expert_counts[offset + i];
    }
  }

  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  for (int i = 0; i < num_elem_per_thread; i++) {
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      blocked_expert_counts_cumsum[offset + i] = cumsum;
      if ((offset + i) % num_blocks_per_seq == 0) {
        expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
      }
      cumsum += blocked_expert_counts[offset + i];
      if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum;
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                                            int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_blocks_per_seq) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq ? blocked_expert_counts[threadIdx.x] : 0;
  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  if (threadIdx.x < num_experts_per_node * num_blocks_per_seq) {
    blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
    if (threadIdx.x % num_blocks_per_seq == 0) {
      expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
    }
    if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1) {
      expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void globalExpertPrefixSum(const int* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                           int64_t* expert_first_token_offset, const int64_t num_experts_per_node, const int64_t num_tokens_per_block,
                           const int64_t num_blocks_per_seq, cudaStream_t stream) {
  const int64_t num_elements = num_experts_per_node * num_blocks_per_seq;

  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 1024;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  if (num_elements <= 1024) {
    auto func = globalExpertPrefixSumKernel<1024>;
    if (num_elements <= 32) {
      func = globalExpertPrefixSumKernel<32>;
      config.blockDim = 32;
    } else if (num_elements <= 64) {
      func = globalExpertPrefixSumKernel<64>;
      config.blockDim = 64;
    } else if (num_elements <= 128) {
      func = globalExpertPrefixSumKernel<128>;
      config.blockDim = 128;
    } else if (num_elements <= 256) {
      func = globalExpertPrefixSumKernel<256>;
      config.blockDim = 256;
    } else if (num_elements <= 512) {
      func = globalExpertPrefixSumKernel<512>;
      config.blockDim = 512;
    }
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
  } else {
    auto func = globalExpertPrefixSumLargeKernel<1024>;
    const int64_t num_elem_per_thread = onnxruntime::llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq, num_elem_per_thread);
  }
}

__global__ void mergeExpertPrefixSumKernel(const int* blocked_expert_counts, const int* blocked_expert_counts_cumsum,
                                           const int* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                                           int* unpermuted_row_to_permuted_row, const int num_tokens) {
  const int target_expert_id = blockIdx.x;
  const int block_id = blockIdx.y;
  const int num_blocks_per_seq = gridDim.y;
  const int token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
  const int offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
  if (threadIdx.x < cnt) {
    const int unpermuted_row = blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
    const int permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void mergeExpertPrefixSum(const int* blocked_expert_counts, const int* blocked_expert_counts_cumsum,
                          const int* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, const int64_t num_tokens, const int64_t num_experts_per_node,
                          const int64_t num_tokens_per_block, const int64_t num_blocks_per_seq, cudaStream_t stream) {
  const dim3 blocks(num_experts_per_node, num_blocks_per_seq);
  const dim3 threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, mergeExpertPrefixSumKernel, blocked_expert_counts, blocked_expert_counts_cumsum,
                     blocked_row_to_unpermuted_row, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
                     unpermuted_row_to_permuted_row, static_cast<int>(num_tokens));
}

// threeStepBuildExpertMapsSortFirstToken uses three kernels to achieve the sort of token_selected_experts

// 1. blockExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA has
// num_tokens_per_block threads. blocked_row_to_unpermuted_row points to a 2D buffer of size [num_experts_per_node,
// num_tokens], which can be viewed as [num_experts_per_node, num_blocks_per_seq] blocks, and each block has
// num_tokens_per_block tokens. Note that each CTA corresponds to a block in blocked_row_to_unpermuted_row. Within each
// CTA, the threads leverage cub::BlockScan to compute the offsets of tokens that activate the target expert. If a
// thread's token activates the target expert, the thread stores its unpermuted_row to the buffer block with the offset.
// In addition, the kernel also stores the expert counts for each block to another 2D buffer blocked_expert_counts of
// size [num_experts_per_node, num_blocks_per_seq].

// 2. globalExpertPrefixSumKernel launches 1 CTA; that CTA has num_experts_per_node * num_blocks_per_seq threads.
// The kernel views blocked_expert_counts as a 1D buffer, and leverages cub::BlockScan to compute the prefix sum of the
// expert counts for each block. The prefix sum is stored to blocked_expert_counts_cumsum.

// 3. mergeExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA has
// num_tokens_per_block threads. Each CTA obtains the block-level offset from blocked_expert_counts_cumsum, and thus
// compacts blocked_row_to_unpermuted_row to permuted_row_to_unpermuted_row. In addition, with the block-level offsets,
// the kernel fills permuted_token_selected_experts.

// computeNumTokensPerBlock decides num_tokens_per_block. Note that both blockExpertPrefixSumKernel and
// globalExpertPrefixSumKernel leverage cub::BlockScan, and their CTA sizes are num_tokens_per_block and
// num_experts_per_node * num_blocks_per_seq, respectively. computeNumTokensPerBlock tries to find a minimum CTA size
// for both kernels, so that the block-leval cub::BlockScan can be efficient.

void threeStepBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
                                            int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            const int64_t num_tokens, const int64_t num_experts_per_node, const int64_t num_experts_per_token,
                                            const int start_expert_id, cudaStream_t stream) {
  const int64_t num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_tokens, num_tokens_per_block);

  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row, num_tokens,
                       num_experts_per_node, num_experts_per_token, num_tokens_per_block, num_blocks_per_seq, start_expert_id, stream);
  sync_check_cuda_error(stream);

  globalExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum, expert_first_token_offset,
                        num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, stream);
  sync_check_cuda_error(stream);

  mergeExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row,
                       permuted_token_selected_experts, permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, num_tokens,
                       num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, stream);
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(
    const float** alpha_scale_ptr_array, const int64_t num_experts_per_node, const float* fp8_dequant) {
  // First, compute the global tid. We only need 1 thread per expert.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

  assert(fp8_dequant != nullptr);
  alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

const float** computeFP8DequantScale(
    const float** alpha_scale_ptr_array, const int num_experts_per_node, const float* fp8_dequant, cudaStream_t stream) {
  if (!fp8_dequant) {
    return nullptr;
  }

  const int threads = std::min(1024, num_experts_per_node);
  const int blocks = (num_experts_per_node + threads - 1) / threads;

  computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(
      alpha_scale_ptr_array, num_experts_per_node, fp8_dequant);

  return alpha_scale_ptr_array;
}

template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int gemm_m,
                                            int gemm_n, int gemm_k, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat,
                                            const TmaWarpSpecializedGroupedGemmInput::ElementSF* weight_block_scale, int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_A);
  assert(layout_info.fpX_block_scaling_factors_stride_B);

  // M & N swapped for transpose
  auto stride_a_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(layout_info.fpX_block_scaling_factors_stride_A);
  auto stride_b_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(layout_info.fpX_block_scaling_factors_stride_B);
  stride_a_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  stride_b_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));

  // This assert validates our current assumption that A&B can be safely transposed without needing to modify
  assert(BSConfig::tile_atom_to_shape_SFB(cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) == BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type = std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
                          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_A[expert] = fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_B[expert] = weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}

__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx,
    int groupwise_scale_group_size) {
  layout_info.stride_a[out_idx] = cutlass::make_cute_packed_stride(
      TmaWarpSpecializedGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
  int stride_b_n = gemm_n;
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION) {
    stride_b_n *= 2;
  }
  layout_info.stride_b[out_idx] = cutlass::make_cute_packed_stride(
      TmaWarpSpecializedGroupedGemmInput::StrideB{}, cute::make_shape(stride_b_n, gemm_k, 1));
  if (layout_info.stride_c) {
    assert(false && "CUTLASS does not support a 1xN bias");
    //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
    layout_info.stride_c[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, 1));
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE ||
      layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION) {
    int stride_d_n = gemm_n;
    if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION) {
      stride_d_n /= 2;
    }
    layout_info.default_epilogue.stride_d[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::DefaultEpilogue::StrideD{}, cute::make_shape(stride_d_n, gemm_m, 1));
  }
  if (layout_info.int4_groupwise_params.enabled) {
    assert(groupwise_scale_group_size > 0);
    layout_info.int4_groupwise_params.stride_s_a[out_idx] = cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::StrideSFA{},
                                                                                             cute::make_shape(gemm_n, gemm_k / groupwise_scale_group_size, 1));
  }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeTmaWarpSpecializedInputPointers(TmaWarpSpecializedGroupedGemmInput& layout_info, int64_t gemm_m,
                                                       int64_t gemm_n, int64_t gemm_k, int num_tokens_before_expert, int64_t expert, const T* in,
                                                       const WeightType* weights, const TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA* w4a8_weight_scale,
                                                       const TmaWarpSpecializedGroupedGemmInput::ElementSF* mxfp4_weight_scale,
                                                       int groupwise_scale_group_size,
                                                       const ScaleBiasType* bias, OutputType* output, const int64_t out_idx) {
  // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
  layout_info.ptr_a[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);

  // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
  layout_info.ptr_b[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));

  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE ||
      layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION) {
    // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
    int64_t ptr_d_n = gemm_n;
    if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION) {
      ptr_d_n /= 2;
    }
    layout_info.default_epilogue.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * ptr_d_n);
  }
  if (layout_info.int4_groupwise_params.enabled) {
    assert(groupwise_scale_group_size > 0);
    assert(mxfp4_weight_scale || w4a8_weight_scale);
    if (mxfp4_weight_scale) {
      constexpr bool use_wfp4a16_scale_layout =
#if defined(ENABLE_FP4) && defined(ENABLE_BF16)
          std::is_same_v<WeightType, __nv_fp4_e2m1> &&
          (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
#elif defined(ENABLE_FP4)
          std::is_same_v<WeightType, __nv_fp4_e2m1> && std::is_same_v<T, half>;
#else
          false;
#endif
      int64_t scale_offset = 0;
      if constexpr (use_wfp4a16_scale_layout) {
        // The WFP4A16 scale buffer is packed in groups of 8 k-blocks (PackedScalesNum =
        // CTA_K(256) / group_size(32)). PrePackFp4ScalesForTmaWs zero-pads each expert's
        // k-block count up to a multiple of 8 so the GEMM's last partial CTA-K-tile can read
        // a full packed group. Stride experts by the padded count to match the packed buffer.
        // (For 8-aligned k-block counts this is a no-op, preserving existing 256-aligned shapes.)
        constexpr int64_t kPackedScalesPerKTile = 8;
        const int64_t k_blocks = gemm_k / groupwise_scale_group_size;
        const int64_t k_blocks_padded =
            ((k_blocks + kPackedScalesPerKTile - 1) / kPackedScalesPerKTile) * kPackedScalesPerKTile;
        scale_offset = expert * (gemm_n * k_blocks_padded);
      } else {
        constexpr int scale_cols_alignment = 4;
        const auto scale_rows = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
            gemm_n, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX);
        const auto scale_cols = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
            gemm_k / groupwise_scale_group_size, scale_cols_alignment);
        scale_offset = expert * scale_rows * scale_cols;
      }
      layout_info.int4_groupwise_params.ptr_s_a[out_idx] =
          reinterpret_cast<const TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA*>(
              safe_inc_ptr(mxfp4_weight_scale, scale_offset));
    } else {
      constexpr int scale_element_size_adjustment =
#if defined(ENABLE_FP4)
          std::is_same_v<WeightType, __nv_fp4_e2m1> ? 2 :
#endif
                                                    1;
      const auto scale_offset = expert * (gemm_n * gemm_k / (groupwise_scale_group_size * scale_element_size_adjustment));
      layout_info.int4_groupwise_params.ptr_s_a[out_idx] = safe_inc_ptr(w4a8_weight_scale, scale_offset);
    }
  }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(const int64_t* expert_first_token_offset,
                                                       TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                                       int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
                                                       const int64_t num_experts_per_node, const T* gemm1_in, const T* gemm2_in, const WeightType* weights1,
                                                       const WeightType* weights2, const float* alpha_scale_flat1, const float* alpha_scale_flat2,
                                                       const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat1,
                                                       const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat2, QuantParams quant_params,
                                                       const ScaleBiasType* bias1, const ScaleBiasType* bias2, OutputType* gemm1_output, OutputType* gemm2_output) {
  // First, compute the global tid. We only need 1 thread per expert.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Both gemms use the same token offset
  const auto num_tokens_before_expert = expert_first_token_offset[expert];
  const auto num_tokens_including_expert = expert_first_token_offset[expert + 1];
  const auto num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
  const auto gemm_m = num_tokens_to_expert;

  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm1_n, gemm_m, gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm2_n, gemm_m, gemm2_k);

  if (layout_info1.int4_groupwise_params.enabled) {
    layout_info1.int4_groupwise_params.shape.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
        gemm1_n, gemm_m, gemm1_k);
  }

  if (layout_info2.int4_groupwise_params.enabled) {
    layout_info2.int4_groupwise_params.shape.problem_shapes[expert] = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape(
        gemm2_n, gemm_m, gemm2_k);
  }

  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;
    layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + expert;
  }

  constexpr int groupwise_scale_group_size =
#if defined(ENABLE_FP4)
      std::is_same_v<WeightType, __nv_fp4_e2m1> ? cutlass::gemm::collective::detail::mxfp4_group_size :
#endif
                                                cutlass::gemm::collective::detail::int4_group_size;

  auto setupIfSelected = [&](auto bs_config, auto quant_type) {
    if (quant_type.fc1.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(layout_info1, expert, gemm_m, gemm1_n, gemm1_k,
                                                       fp4_act_flat1, quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(layout_info2, expert, gemm_m, gemm2_n, gemm2_k,
                                                       fp4_act_flat2, quant_type.fc2.weight_block_scale, num_tokens_before_expert);
    }
  };

  // NVFP4 native FP4xFP4 sets quant_params.fp4 (WeightType == __nv_fp4_e2m1); the WFP4AFP8 and
  // MXFP4 fp4-weight paths leave it null and use mxfp8_mxfp4 instead, so this stays a runtime no-op
  // for them. Run unconditionally (matching TRT-LLM) so the fp4-weight NVFP4 path is wired.
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{}, quant_params.fp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{}, quant_params.fp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{}, quant_params.mxfp8_mxfp4);

  assert(gemm_m <= INT32_MAX);
  assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
  assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
  assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
  assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert, groupwise_scale_group_size);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert, groupwise_scale_group_size);

  const auto* fc1_weight_block_scale = quant_params.mxfp8_mxfp4.fc1.weight_block_scale
                                           ? quant_params.mxfp8_mxfp4.fc1.weight_block_scale
                                       : quant_params.fp8_mxfp4.fc1.weight_block_scale
                                           ? quant_params.fp8_mxfp4.fc1.weight_block_scale
                                           : quant_params.fp4.fc1.weight_block_scale;
  const auto* fc2_weight_block_scale = quant_params.mxfp8_mxfp4.fc2.weight_block_scale
                                           ? quant_params.mxfp8_mxfp4.fc2.weight_block_scale
                                       : quant_params.fp8_mxfp4.fc2.weight_block_scale
                                           ? quant_params.fp8_mxfp4.fc2.weight_block_scale
                                           : quant_params.fp4.fc2.weight_block_scale;

  computeTmaWarpSpecializedInputPointers(layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert,
                                         gemm1_in, weights1,
                                         reinterpret_cast<const TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA*>(
                                             quant_params.groupwise.fc1.weight_scales),
                                         fc1_weight_block_scale,
                                         groupwise_scale_group_size,
                                         bias1, gemm1_output, expert);
  computeTmaWarpSpecializedInputPointers(layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert,
                                         gemm2_in, weights2,
                                         reinterpret_cast<const TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA*>(
                                             quant_params.groupwise.fc2.weight_scales),
                                         fc2_weight_block_scale,
                                         groupwise_scale_group_size,
                                         bias2, gemm2_output, expert);

  // Backport TRT-LLM 603ec03f: move launch_dependents to after the kernel work to avoid
  // illegal memory access on SM120 when dependents are launched before this kernel finishes
  // populating its outputs.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the permuted_row_to_unpermuted_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <class InputActivationsType, class ExpandedActivationsType,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType, bool PRE_QUANT_AWQ>
__global__ void expandInputRowsKernel(const InputActivationsType* unpermuted_input,
                                      ExpandedActivationsType* permuted_output, const float* unpermuted_scales, float* permuted_scales,
                                      const int* permuted_row_to_unpermuted_row, const int64_t num_tokens, const int64_t hidden_size, const int64_t k,
                                      const float* fc1_act_global_scale, bool use_per_expert_act_scale, const int64_t* expert_first_token_offset,
                                      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                      const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, const int64_t num_experts_per_node,
                                      const InputActivationsType* prequant_scales = nullptr) {
  static_assert(BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE || !PRE_QUANT_AWQ,
                "AWQ and Block Scaling are mutually exclusive");
#ifdef ENABLE_FP4
  constexpr bool is_mxfp8 = std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX && !PRE_QUANT_AWQ;
  constexpr bool is_mxfp8_input = is_mxfp8 && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>;
  constexpr bool need_mxfp8_quant = is_mxfp8 && !is_mxfp8_input;
  constexpr bool is_nvfp4 = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4 && !PRE_QUANT_AWQ;
  constexpr bool is_nvfp4_input = is_nvfp4 && std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
  constexpr bool need_nvfp4_quant = is_nvfp4 && !is_nvfp4_input;
#else
  constexpr bool is_mxfp8 = false;
  constexpr bool is_mxfp8_input = false;
  constexpr bool need_mxfp8_quant = false;
  constexpr bool is_nvfp4 = false;
  constexpr bool is_nvfp4_input = false;
  constexpr bool need_nvfp4_quant = false;
#endif

  static_assert(need_nvfp4_quant || need_mxfp8_quant || PRE_QUANT_AWQ || std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                "Only NVFP4, MXFP8 and WINT4_AFP8 supports outputting a different format as part of the expansion");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  constexpr int VecSize = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                   : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;

  constexpr int64_t ELEM_PER_THREAD = (is_nvfp4 || is_mxfp8) ? CVT_FP4_ELTS_PER_THREAD : (128 / sizeof_bits<InputActivationsType>::value);

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  constexpr int64_t min_k_dim_alignment = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  const int64_t padded_hidden_size = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(hidden_size, min_k_dim_alignment);

  const int64_t num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens; permuted_row += gridDim.x) {
    const int64_t unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];

    // Load 128-bits per thread

    constexpr int64_t ELEM_PER_BYTE = is_nvfp4_input ? 2 : 1;
    using DataElem = std::conditional_t<is_nvfp4_input, uint32_t,
                                        std::conditional_t<is_mxfp8_input, uint64_t, cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>>;
    using OutputElem = std::conditional_t<is_nvfp4, uint32_t,
                                          std::conditional_t<is_mxfp8, uint64_t, cutlass::Array<ExpandedActivationsType, ELEM_PER_THREAD>>>;

    // Duplicate and permute rows
    const int64_t source_k_rank = unpermuted_row / num_tokens;
    const int64_t source_row = unpermuted_row % num_tokens;

    const auto* source_row_ptr = reinterpret_cast<const DataElem*>(unpermuted_input + source_row * hidden_size / ELEM_PER_BYTE);
    // Cast first to handle when this is FP4
    auto* dest_row_ptr = reinterpret_cast<OutputElem*>(permuted_output) + permuted_row * hidden_size / ELEM_PER_THREAD;

    const int64_t start_offset = threadIdx.x;
    const int64_t stride = EXPAND_THREADS_PER_BLOCK;
    const int64_t num_elems_in_col = hidden_size / ELEM_PER_THREAD;
    assert(hidden_size % ELEM_PER_THREAD == 0);
    assert(hidden_size % VecSize == 0);

    if constexpr (is_nvfp4 || is_mxfp8) {
      static_assert(ELEM_PER_THREAD == 8, "Expecting 8 elements per thread for quantized types");
      int64_t expert = findTotalEltsLessThanTarget(
                           expert_first_token_offset, num_experts_per_node, (int64_t)permuted_row + 1) -
                       1;

      assert(!fc1_act_global_scale || is_nvfp4 && "Global scale is only supported for NVFP4");
      size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
      float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[act_scale_idx] : 1.0f;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];

      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto in_vec = source_row_ptr[elem_index];
        if constexpr (need_nvfp4_quant || need_mxfp8_quant) {
          auto res = quantizePackedFPXValue<InputActivationsType, ExpandedActivationsType, DataElem, VecSize>(
              in_vec, global_scale_val, num_tokens_before_expert, expert, permuted_row, elem_index,
              padded_hidden_size, fc1_act_sf_flat,
              is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                       : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
          static_assert(
              sizeof(res) == sizeof(*dest_row_ptr), "Quantized value must be the same size as the output");
          dest_row_ptr[elem_index] = res;
        } else {
          assert(act_scale_idx == 0 && "Cannot use per-expert act scale for pre-quantized activations");
          writeSF<VecSize, ELEM_PER_THREAD>(num_tokens_before_expert, expert, source_row, permuted_row,
                                            elem_index, padded_hidden_size, fc1_act_sf_flat, input_sf);
          dest_row_ptr[elem_index] = in_vec;
        }
      }

      // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan values in the
      // padded SF atom Use VecSize per thread since we are just writing out zeros so every thread can process a
      // whole vector
      size_t padding_start_offset = hidden_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_hidden_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col; elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, permuted_row, elem_index,
                                  padded_hidden_size, fc1_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    } else if constexpr (PRE_QUANT_AWQ) {
      static_assert(!is_nvfp4 && !is_mxfp8, "NVFP4 and MXFP8 are not supported for AWQ");
      static_assert(!std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                    "Input and output types must be different for AWQ");
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto frag_elems = source_row_ptr[elem_index];

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEM_PER_THREAD; e++) {
          frag_elems[e] = frag_elems[e] * prequant_scales[elem_index * ELEM_PER_THREAD + e];
        }

        dest_row_ptr[elem_index] = arrayConvert<DataElem, OutputElem>(frag_elems);
      }
    } else {
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
      }
    }

    if (permuted_scales && threadIdx.x == 0) {
      const int64_t source_k_idx = source_row * k + source_k_rank;
      permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // Pad zeros in the extra SFs along the N dimension, we do this to ensure there are no nan values in the padded SF
  // atom
  if constexpr (is_nvfp4 || is_mxfp8) {
    const int64_t start_offset = threadIdx.x;
    const int64_t stride = EXPAND_THREADS_PER_BLOCK;
    // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
    const int64_t padded_num_elems_in_col = padded_hidden_size / VecSize;
    assert(padded_hidden_size % VecSize == 0);

    constexpr int min_num_tokens_alignment = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                      : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
    static_assert((min_num_tokens_alignment & (min_num_tokens_alignment - 1)) == 0,
                  "Min num tokens alignment must be a power of two");
    // Since we don't know a priori how much padding is needed we assume the max per expert
    // NOTE: we don't use (min_num_tokens_alignment-1) to be able to do power of two divisions
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    for (int64_t padding_token = blockIdx.x; padding_token < num_padding_tokens; padding_token += gridDim.x) {
      int64_t expert = padding_token / min_num_tokens_alignment;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];
      int64_t num_tokens_after_expert = expert_first_token_offset[expert + 1];
      int64_t tokens_to_expert = num_tokens_after_expert - num_tokens_before_expert;
      int64_t padding_to_expert = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(tokens_to_expert, min_num_tokens_alignment) - tokens_to_expert;
      int64_t expert_pad_idx = padding_token % min_num_tokens_alignment;
      if (expert_pad_idx < padding_to_expert) {
        for (int64_t elem_index = start_offset; elem_index < padded_num_elems_in_col; elem_index += stride) {
          writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1,
                                    num_tokens_after_expert + expert_pad_idx, elem_index, padded_hidden_size, fc1_act_sf_flat,
                                    /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
        }
      }
    }
  }
}

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(const InputActivationsType* unpermuted_input,
                                   ExpandedActivationsType* permuted_output, const float* unpermuted_scales, float* permuted_scales,
                                   const int* permuted_row_to_unpermuted_row, const int64_t num_rows, const int64_t hidden_size, const int k,
                                   const int num_experts_per_node, const QuantParams& quant_params, bool use_per_expert_act_scale,
                                   int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                   const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, const void* prequant_scales, cudaStream_t stream) {
#ifdef ENABLE_FP4
  ORT_ENFORCE(
      (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> && fc1_act_sf_flat) || !use_per_expert_act_scale,
      "Per-expert act scale for FC1 is only supported for NVFP4 activations");
  constexpr int64_t min_num_tokens_alignment = std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>
                                                   ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
  int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;
#else
  int64_t num_padding_tokens = 0;
#endif

  static const int64_t smCount = onnxruntime::llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  const int64_t blocks = std::min(smCount * 8, std::max(num_rows * k, num_padding_tokens));
  const int64_t threads = EXPAND_THREADS_PER_BLOCK;

  auto func = [&]() {
#ifdef ENABLE_FP8
    // Always MXFP8
    if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && !std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      ORT_ENFORCE(quant_params.mxfp8_mxfp4.fc1.weight_block_scale || prequant_scales,
                  "MXFP8xMXFP4 block scaling or prequant_scales or prequant_scales parameters not provided");
      return prequant_scales ? &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                                      TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, true>
                             : &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                                      TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>;
    }
    // Could be either regular FP8 or MXFP8
    else if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      ORT_ENFORCE(!prequant_scales, "NVFP4 is not supported for AWQ");
      return quant_params.mxfp8_mxfp4.fc1.weight_block_scale
                 ? &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>
                 : &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    } else
#endif
#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>) {
      ORT_ENFORCE(
          quant_params.fp4.fc1.weight_block_scale, "NVFP4 block scaling is expected for FP4xFP4");
      ORT_ENFORCE(!prequant_scales, "NVFP4 is not supported for AWQ");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4, false>;
    } else
#endif
    {
      ORT_ENFORCE(!prequant_scales, "w4afp8 Prequant scales provided for non-FP8 data type");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, func, unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
                     permuted_row_to_unpermuted_row, num_rows, hidden_size, k, quant_params.fp4.fc1.act_global_scale,
                     use_per_expert_act_scale, expert_first_token_offset, fc1_act_sf_flat, input_sf, num_experts_per_node,
                     reinterpret_cast<const InputActivationsType*>(prequant_scales));
}

#define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)                      \
  template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(             \
      const InputActivationsType* unpermuted_input, ExpandedActivationsType* permuted_output,             \
      const float* unpermuted_scales, float* permuted_scales, const int* permuted_row_to_unpermuted_row,  \
      const int64_t num_rows, const int64_t hidden_size, const int k, const int num_experts_per_node,     \
      const QuantParams& quant_params, bool use_per_expert_act_scale, int64_t* expert_first_token_offset, \
      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                                     \
      const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, const void* prequant_scales,         \
      cudaStream_t stream)

// Instantiate the data types that are used by the external pytorch op
INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
#ifdef ENABLE_BF16
INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
#endif
#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
// W4A8 (WFP4AFP8) native path: BF16/FP16 input is quantized to MXFP8 inside the expansion kernel
// using the existing MXFP8 branch (gated by quant_params.mxfp8_mxfp4.fc1.weight_block_scale).
INSTANTIATE_EXPAND_INPUT_ROWS(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

enum class ScaleMode : int {
  NO_SCALE = 0,
  DEFAULT = 1,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// The routing metadata (selected expert, permuted row, routing scale) only varies with the
// top-k slot, not with the column, yet the naive reduction loop re-reads it from global memory
// for every column tile through a chain of two dependent loads (expert map -> permuted row ->
// expanded row data). With a small grid (one block per token, i.e. a decode step) there is not
// enough occupancy to hide that chain and the kernel becomes long-scoreboard bound. Staging the
// metadata in shared memory once per block turns the k expanded-row loads into independent
// accesses that can all be in flight at the same time.
constexpr static int FINALIZE_MAX_STAGED_TOPK = FINALIZE_THREADS_PER_BLOCK;

// Expert rows whose loads are issued together by the staged finalize kernel (top-k <= 16 takes one round).
constexpr static int FINALIZE_LOAD_BATCH = 16;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE, bool STAGE_ROUTING>
__global__ void finalizeMoeRoutingKernel(const GemmOutputType* expanded_permuted_rows,
                                         OutputType* reduced_unpermuted_output, const ScaleBiasType* bias, const float* scales,
                                         const int* unpermuted_row_to_permuted_row, const int* token_selected_experts, const int64_t orig_cols,
                                         const int64_t experts_per_token, const int num_experts_per_node, const int start_expert_id,
                                         const bool early_routing_loads) {
  const int64_t original_row = blockIdx.x;
  const int64_t num_rows = gridDim.x;
  const auto offset = original_row * orig_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
  assert(orig_cols % FINALIZE_ELEM_PER_THREAD == 0);

  const int64_t start_offset = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.y;
  const int64_t num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  const auto* bias_v = reinterpret_cast<const BiasElem*>(bias);
  const auto* expanded_permuted_rows_v = reinterpret_cast<const InputElem*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (!(STAGE_ROUTING && early_routing_loads)) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

  __shared__ int s_permuted_row[STAGE_ROUTING ? FINALIZE_MAX_STAGED_TOPK : 1];
  __shared__ int s_expert_id[STAGE_ROUTING ? FINALIZE_MAX_STAGED_TOPK : 1];
  __shared__ float s_row_scale[STAGE_ROUTING ? FINALIZE_MAX_STAGED_TOPK : 1];
  if constexpr (STAGE_ROUTING) {
    for (int64_t k_idx = threadIdx.x; k_idx < experts_per_token; k_idx += blockDim.x) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int const expert_id = token_selected_experts[k_offset] - start_expert_id;
      bool const is_local_expert = expert_id >= 0 && expert_id < num_experts_per_node;
      s_expert_id[k_idx] = expert_id;
      // -1 marks a slot routed to an expert owned by another rank, which is skipped below.
      s_permuted_row[k_idx] = is_local_expert ? unpermuted_row_to_permuted_row[original_row + k_idx * num_rows] : -1;
      s_row_scale[k_idx] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
    }
    __syncthreads();
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (STAGE_ROUTING && early_routing_loads) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    if constexpr (STAGE_ROUTING) {
      // Issue the loads of up to FINALIZE_LOAD_BATCH expert rows together (they are independent), then
      // accumulate them in top-k order: the sums are the same as with one load per iteration.
      for (int k_base = 0; k_base < experts_per_token; k_base += FINALIZE_LOAD_BATCH) {
        InputElem batch[FINALIZE_LOAD_BATCH];
#pragma unroll
        for (int i = 0; i < FINALIZE_LOAD_BATCH; ++i) {
          const int k_idx = k_base + i;
          if (k_idx < experts_per_token && s_permuted_row[k_idx] >= 0) {
            batch[i] = expanded_permuted_rows_v[static_cast<int64_t>(s_permuted_row[k_idx]) * num_elems_in_col + elem_index];
          }
        }
#pragma unroll
        for (int i = 0; i < FINALIZE_LOAD_BATCH; ++i) {
          const int k_idx = k_base + i;
          if (k_idx >= experts_per_token || s_permuted_row[k_idx] < 0) {
            continue;
          }
          const int64_t expert_id = s_expert_id[k_idx];
          const float row_scale = s_row_scale[k_idx];
          ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(batch[i]);
          if (bias) {
            const auto* bias_ptr = bias_v + expert_id * num_elems_in_col;
            expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
          }

          thread_output = thread_output + row_scale * expert_result;
        }
      }
    } else {
#pragma unroll 4
      for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
        const int64_t k_offset = original_row * experts_per_token + k_idx;
        const int64_t expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }
        const int64_t expanded_permuted_row = unpermuted_row_to_permuted_row[original_row + k_idx * num_rows];
        const float row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

        const auto* expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

        ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
        if (bias) {
          const auto* bias_ptr = bias_v + expert_id * num_elems_in_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }

        thread_output = thread_output + row_scale * expert_result;
      }
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE, int ExpertsPerToken>
__global__ void finalizeMoeRoutingOneRowKernel(const GemmOutputType* expanded_permuted_rows,
                                               OutputType* reduced_unpermuted_output, const ScaleBiasType* bias, const float* scales,
                                               const int* unpermuted_row_to_permuted_row, const int* token_selected_experts,
                                               const int64_t orig_cols, const int num_experts_per_node,
                                               const int start_expert_id) {
  static_assert(ExpertsPerToken > 0 && ExpertsPerToken <= 4);

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
  assert(orig_cols % FINALIZE_ELEM_PER_THREAD == 0);

  __shared__ int expanded_permuted_rows_for_topk[ExpertsPerToken];
  __shared__ int expert_ids_for_topk[ExpertsPerToken];
  __shared__ float row_scales_for_topk[ExpertsPerToken];

  const int tid = threadIdx.x;
  if (tid < ExpertsPerToken) {
    const int expert_id = token_selected_experts[tid] - start_expert_id;
    expert_ids_for_topk[tid] = expert_id;
    expanded_permuted_rows_for_topk[tid] =
        (expert_id >= 0 && expert_id < num_experts_per_node) ? unpermuted_row_to_permuted_row[tid] : -1;
    row_scales_for_topk[tid] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[tid];
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int64_t start_offset = tid;
  const int64_t stride = FINALIZE_THREADS_PER_BLOCK;
  const int64_t num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  const auto* bias_v = reinterpret_cast<const BiasElem*>(bias);
  const auto* expanded_permuted_rows_v = reinterpret_cast<const InputElem*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_unpermuted_output);

  for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
#pragma unroll
    for (int k_idx = 0; k_idx < ExpertsPerToken; ++k_idx) {
      const int expanded_permuted_row = expanded_permuted_rows_for_topk[k_idx];
      if (expanded_permuted_row < 0) {
        continue;
      }

      const auto* expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (bias) {
        const auto* bias_ptr = bias_v + expert_ids_for_topk[k_idx] * num_elems_in_col;
        expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      }

      thread_output = thread_output + row_scales_for_topk[k_idx] * expert_result;
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingNoFillingKernel(const GemmOutputType* expanded_permuted_rows,
                                                  OutputType* reduced_unpermuted_output, const ScaleBiasType* bias, const float* scales,
                                                  const int* const unpermuted_row_to_permuted_row, const int* permuted_row_to_unpermuted_row,
                                                  const int* token_selected_experts, const int64_t* expert_first_token_offset, const int64_t num_rows,
                                                  const int64_t orig_cols, const int64_t experts_per_token, const int num_experts_per_node, const int start_expert_id) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int64_t num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t expanded_permuted_row = blockIdx.x; expanded_permuted_row < num_valid_tokens;
       expanded_permuted_row += gridDim.x) {
    int64_t unpermuted_row = permuted_row_to_unpermuted_row[expanded_permuted_row];

    // Duplicate and permute rows
    const int64_t source_k_rank = unpermuted_row / num_rows;
    const int64_t source_row = unpermuted_row % num_rows;

    // If the expert is the first selected (valid) one of the corresponding token on the current EP rank, do
    // reduction; otherwise, skip.
    bool is_first_selected_expert = true;
    for (int k_idx = 0; k_idx < source_k_rank; ++k_idx) {
      const int expert_id = token_selected_experts[source_row * experts_per_token + k_idx] - start_expert_id;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        is_first_selected_expert = false;
        break;
      }
    }
    if (!is_first_selected_expert) {
      continue;
    }

    OutputType* reduced_row_ptr = reduced_unpermuted_output + source_row * orig_cols;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
    assert(orig_cols % FINALIZE_ELEM_PER_THREAD == 0);

    const int64_t start_offset = threadIdx.x;
    const int64_t stride = FINALIZE_THREADS_PER_BLOCK;
    const int64_t num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    const auto* bias_v = reinterpret_cast<const BiasElem*>(bias);
    const auto* expanded_permuted_rows_v = reinterpret_cast<const InputElem*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      ComputeElem thread_output;
      thread_output.fill(0);
      for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
        const int64_t k_offset = source_row * experts_per_token + k_idx;
        const int64_t expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        const int64_t expanded_permuted_row_from_k_idx = unpermuted_row_to_permuted_row[source_row + k_idx * num_rows];

        const float row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

        const auto* expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row_from_k_idx * num_elems_in_col;

        ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

        if (bias) {
          const auto* bias_ptr = bias_v + expert_id * num_elems_in_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }

        thread_output = thread_output + row_scale * expert_result;
      }
      OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
      reduced_row_ptr_v[elem_index] = output_elem;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename KernelPtr>
struct FinalizeKernelLaunchCaps {
  int max_threads_per_block;
  bool sm90_pdl_available;
};

template <typename KernelPtr>
FinalizeKernelLaunchCaps<KernelPtr> getFinalizeKernelLaunchCaps(KernelPtr func) {
  static thread_local int cached_device = -1;
  static thread_local KernelPtr cached_func = nullptr;
  static thread_local FinalizeKernelLaunchCaps<KernelPtr> cached_caps{FINALIZE_THREADS_PER_BLOCK, false};
  int device = -1;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  if (device != cached_device || func != cached_func) {
    cudaFuncAttributes attr{};
    CUDA_CALL_THROW(cudaFuncGetAttributes(&attr, func));
    int driver_version = 0;
    int sm_major = 0;
    const bool driver_and_arch_ok =
        cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version >= 12000 &&
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && sm_major >= 9;
    if (!driver_and_arch_ok) {
      (void)cudaGetLastError();
    }
    const int max_threads = std::max(32, std::min(1024, (attr.maxThreadsPerBlock / 32) * 32));
    cached_caps = {max_threads,
                   driver_and_arch_ok && attr.ptxVersion >= 90 && attr.binaryVersion >= 90};
    cached_device = device;
    cached_func = func;
  }
  return cached_caps;
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(const GemmOutputType* expanded_permuted_rows,
                                      OutputType* reduced_unpermuted_output, const ScaleBiasType* bias, const float* final_scales,
                                      const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_unpermuted_row,
                                      const int* token_selected_experts, const int64_t* expert_first_token_offset, const int64_t num_rows,
                                      const int64_t cols, const int64_t experts_per_token, const int64_t num_experts_per_node,
                                      MOEParallelismConfig parallelism_config, const bool enable_alltoall, cudaStream_t stream,
                                      const bool early_routing_loads) {
  // Only add bias on rank 0 for tensor parallelism
  const bool is_rank_0 = parallelism_config.tp_rank == 0;
  const ScaleBiasType* bias_ptr = is_rank_0 ? bias : nullptr;
  constexpr int64_t kFinalizeElemPerThread = 128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
  ORT_ENFORCE(cols % kFinalizeElemPerThread == 0,
              "MoE finalize requires cols to be divisible by ", kFinalizeElemPerThread,
              " for vectorized 128-bit loads, got ", cols, ".");
  int num_experts_per_node_int = SafeInt<int>(num_experts_per_node);
  const int start_expert_id = num_experts_per_node_int * parallelism_config.ep_rank;

  const bool env_pdl = onnxruntime::llm::common::getEnvEnablePDL();
  cudaLaunchConfig_t config{};
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = env_pdl ? 1 : 0;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (parallelism_config.ep_size > 1 && enable_alltoall) {
    // If all-to-all comm is enabled, finalizeMoeRouting doesn't need to fill the invalid output tokens with zeros.
    static const int smCount = onnxruntime::llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    const int64_t blocks = smCount * 8;
    const int64_t threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales
                    ? &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT>
                    : &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
    CUDA_CALL_THROW(cudaLaunchKernelEx(
        &config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
        unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
        expert_first_token_offset, num_rows, cols, experts_per_token, num_experts_per_node_int, start_expert_id));
  } else {
    if (num_rows <= 0) {
      return;
    }
    // If all-gather reduce-scatter is used, finalizeMoeRouting must fill invalid output tokens with zeros.
    const int64_t blocks = num_rows;
    // One thread per 128-bit column vector (clamped to the kernel's register-limited maxThreadsPerBlock) so
    // each block reduces its row in as few passes as possible; the one-row kernel below keeps the fixed block size.
    const int64_t column_vectors = cols / kFinalizeElemPerThread;
    config.gridDim = blocks;
    if (num_rows == 1 && experts_per_token > 0 && experts_per_token <= 4) {
      config.blockDim = FINALIZE_THREADS_PER_BLOCK;
#define LAUNCH_FINALIZE_ONE_ROW(EXPERTS_PER_TOKEN)                                                                                        \
  do {                                                                                                                                    \
    auto func = final_scales                                                                                                              \
                    ? &finalizeMoeRoutingOneRowKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, EXPERTS_PER_TOKEN>   \
                    : &finalizeMoeRoutingOneRowKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, EXPERTS_PER_TOKEN>; \
    CUDA_CALL_THROW(cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,          \
                                       unpermuted_row_to_permuted_row, token_selected_experts, cols, num_experts_per_node_int,            \
                                       start_expert_id));                                                                                 \
  } while (0)

      switch (experts_per_token) {
        case 1:
          LAUNCH_FINALIZE_ONE_ROW(1);
          break;
        case 2:
          LAUNCH_FINALIZE_ONE_ROW(2);
          break;
        case 3:
          LAUNCH_FINALIZE_ONE_ROW(3);
          break;
        default:
          LAUNCH_FINALIZE_ONE_ROW(4);
          break;
      }
#undef LAUNCH_FINALIZE_ONE_ROW
    } else if (experts_per_token <= FINALIZE_MAX_STAGED_TOPK) {
      auto func = final_scales
                      ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, true>
                      : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, true>;
      const auto caps = getFinalizeKernelLaunchCaps(func);
      config.blockDim = std::min<int64_t>(
          caps.max_threads_per_block,
          std::max<int64_t>(FINALIZE_THREADS_PER_BLOCK / 2, (column_vectors + 31) / 32 * 32));
      if (num_rows <= 2) {
        const int decode_threads = std::min(32, caps.max_threads_per_block);
        config.blockDim = dim3(decode_threads);
        config.gridDim = dim3(static_cast<unsigned>(num_rows),
                              static_cast<unsigned>((column_vectors + decode_threads - 1) / decode_threads));
      }
      const bool use_early_routing_loads = early_routing_loads && caps.sm90_pdl_available;
      if (use_early_routing_loads) {
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        const cudaError_t status = cudaLaunchKernelEx(
            &config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
            unpermuted_row_to_permuted_row, token_selected_experts, cols, experts_per_token, num_experts_per_node_int,
            start_expert_id, true);
        if (status == cudaSuccess) {
          return;
        }
        if (status != cudaErrorInvalidValue && status != cudaErrorNotSupported) {
          CUDA_CALL_THROW(status);
        }
        (void)cudaGetLastError();
        attrs[0].val.programmaticStreamSerializationAllowed = 0;
      }
      CUDA_CALL_THROW(cudaLaunchKernelEx(
          &config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
          unpermuted_row_to_permuted_row, token_selected_experts, cols, experts_per_token, num_experts_per_node_int,
          start_expert_id, false));
    } else {
      auto func = final_scales
                      ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>
                      : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, false>;
      const auto caps = getFinalizeKernelLaunchCaps(func);
      config.blockDim = std::min<int64_t>(
          caps.max_threads_per_block,
          std::max<int64_t>(FINALIZE_THREADS_PER_BLOCK / 2, (column_vectors + 31) / 32 * 32));
      CUDA_CALL_THROW(cudaLaunchKernelEx(
          &config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, final_scales,
          unpermuted_row_to_permuted_row, token_selected_experts, cols, experts_per_token, num_experts_per_node_int,
          start_expert_id, false));
    }
  }
}

#define INSTANTIATE_FINALIZE_MOE_ROUTING(OutputT, GemmOutputT, ScaleBiasT)                                          \
  template void finalizeMoeRoutingKernelLauncher<OutputT, GemmOutputT, ScaleBiasT>(                                 \
      const GemmOutputT* expanded_permuted_rows, OutputT* reduced_unpermuted_output, const ScaleBiasT* bias,        \
      const float* final_scales, const int* expanded_source_row_to_expanded_dest_row,                               \
      const int* expanded_dest_row_to_expanded_source_row, const int* expert_for_source_row,                        \
      const int64_t* expert_first_token_offset, const int64_t num_rows, const int64_t cols,                         \
      const int64_t experts_per_token, const int64_t num_experts_per_node, MOEParallelismConfig parallelism_config, \
      const bool enable_alltoall, cudaStream_t stream, const bool early_routing_loads);

// Instantiate the data types that are used by the external pytorch op
INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
#ifdef ENABLE_BF16
INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#endif

// ============================== Gated Activation =================================

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, const LoraType* lora_result, const ScaleBiasType* bias,
                                  const int64_t* num_valid_tokens_ptr, int* permuted_token_selected_experts, int64_t inter_size) {
  const int64_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  const LoraType* lora_result_1 = lora_result + token * inter_size;
  int expert_id = permuted_token_selected_experts[token];
  if constexpr (IsGated) {
    output = output + token * inter_size * 2;
    bias = bias + expert_id * inter_size * 2;
  } else {
    output = output + token * inter_size;
    bias = bias + expert_id * inter_size;
  }

  constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / sizeof_bits<LoraType>::value;

  using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<const DataElem*>(lora_result_1);
  auto bias_vec = reinterpret_cast<const BiasElem*>(bias);
  auto output_vec = reinterpret_cast<BiasElem*>(output);

  const int64_t start_offset = tid;
  const int64_t stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
  const int64_t num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    auto bias_value = bias_vec[elem_index];
    output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
  }

  if constexpr (IsGated) {
    auto lora_result_2_vec = reinterpret_cast<const DataElem*>(lora_result_1 + num_tokens * inter_size);
    const int64_t inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto lora_value = lora_result_2_vec[elem_index];
      auto bias_value = bias_vec[elem_index + inter_size_vec];
      output_vec[elem_index + inter_size_vec] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }
  }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, const LoraType* lora_result, const ScaleBiasType* bias,
                 const int64_t* num_valid_tokens_ptr, int64_t inter_size, int* permuted_token_selected_experts, int64_t num_tokens,
                 bool is_gated_activation, cudaStream_t stream) {
  const int64_t blocks = num_tokens;
  const int64_t threads = LORA_KERNELS_THREADS_PER_BLOCK;

  auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                         : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
  selected_fn<<<blocks, threads, 0, stream>>>(
      output, lora_result, bias, num_valid_tokens_ptr, permuted_token_selected_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(
    T* output, const T* lora_result, const int64_t* num_valid_tokens_ptr, int64_t inter_size) {
  const int64_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  const int64_t num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  const T* lora_result_1 = lora_result + token * inter_size;
  output = output + token * inter_size * 2;

  constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;

  using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<const DataElem*>(lora_result_1);
  auto output_vec = reinterpret_cast<DataElem*>(output);

  const int64_t start_offset = tid;
  const int64_t stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
  const int64_t num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    output_vec[elem_index] = lora_value;
  }

  auto lora_result_2_vec = reinterpret_cast<const DataElem*>(lora_result_1 + num_tokens * inter_size);
  const int64_t inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_2_vec[elem_index];
    output_vec[elem_index + inter_size_vec] = lora_value;
  }
}

template <class T>
void loraReorder(T* output, const T* lora_result, const int64_t* num_valid_tokens_ptr, int64_t inter_size,
                 int64_t num_tokens, cudaStream_t stream) {
  const int64_t blocks = num_tokens;
  const int64_t threads = LORA_KERNELS_THREADS_PER_BLOCK;

  loraReorderKernel<T><<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, const InputType* input, const int64_t* num_valid_tokens_ptr,
                                 int64_t inter_size, const float* scale, bool scale_is_dequant) {
  const int64_t tid = threadIdx.x;
  const int64_t token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  input = input + token * inter_size;

  constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / sizeof_bits<InputType>::value;

  using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
  auto input_vec = reinterpret_cast<const DataElem*>(input);
  auto output_vec = reinterpret_cast<OutputElem*>(output);

  const int64_t start_offset = tid;
  const int64_t stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
  const int64_t num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

  ComputeElem deqaunt_scale_value;
  float dequant_scale = scale[0];
  if (!scale_is_dequant) {
    dequant_scale = 1.f / dequant_scale;
  }
  deqaunt_scale_value.fill(dequant_scale);

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
  }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, const InputType* input, const int64_t* num_valid_tokens_ptr, int64_t inter_size,
                int64_t num_tokens, const float* scale, bool scale_is_dequant, cudaStream_t stream) {
  const int64_t blocks = num_tokens;
  const int64_t threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

  dequantFP8Kernel<OutputType, InputType>
      <<<blocks, threads, 0, stream>>>(output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::CutlassMoeFCRunner(
    int sm_version,
    ActivationType activation_type,
    bool normalize_routing_weights,
    bool use_sparse_mixer,
    bool use_sm80_fp4)
    : sm_(sm_version),
      activation_type_(activation_type),
      normalize_routing_weights_(normalize_routing_weights),
      use_sparse_mixer_(use_sparse_mixer) {
  // Apply the routing decision before querying tactics; a provisional TMA query can fail on Ampere.
  setUseSm80Fp4(use_sm80_fp4);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::getWorkspaceDeviceBufferSizes(
    const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
    const int experts_per_token, ActivationType activation_type,
    bool use_awq, int swiglu_fusion) {
  size_t num_moe_inputs = experts_per_token * num_rows;
  const size_t permuted_elems = num_moe_inputs * hidden_size;
  const size_t interbuf_elems = num_moe_inputs * inter_size;
  size_t glu_inter_elems = 0;
  bool is_gated_activation = isGatedActivation(activation_type);
  if (is_gated_activation) {
    glu_inter_elems = interbuf_elems * 2;
  } else if (mayHaveDifferentGEMMOutputType()) {
    // In this case we are using activation quantization, and some intermediate buffers will be unquantized
    // We need to have separate memory for these as we can no longer alias the output buffer for reuse
    glu_inter_elems = interbuf_elems;
  }

  bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

  const size_t gemm_output_dtype = sizeof(UnfusedGemmOutputType);

  constexpr float dtype_size = act_fp4 ? 0.5f : (use_w4afp8 ? 2.0f : sizeof(T));

  const size_t permuted_row_to_unpermuted_row_size = num_moe_inputs * sizeof(int);
  const size_t permuted_token_selected_experts_size = num_moe_inputs * sizeof(int);

  const int64_t num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_rows, num_tokens_per_block);
  const size_t blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * sizeof(int);
  const size_t blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  const size_t blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * sizeof(int);

  const size_t permuted_data_size = permuted_elems * dtype_size;
  const size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  const size_t permuted_token_final_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
  const size_t glu_inter_size = glu_inter_elems * gemm_output_dtype;                // May be an intermediate type for quantization
  const size_t fc1_result_size = interbuf_elems * dtype_size;                       // Activation quantizes so back to dtype_size
  const size_t fc2_result_size = num_moe_inputs * hidden_size * gemm_output_dtype;  // May be an intermediate type for quantization

  // If topk is greater than num_experts_per_node (i.e. large EP value), then we don't need to allocate for the whole
  // tokens*topk
  auto act_sf_rows = std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts_per_node));
  const size_t sf_size = getScalingType() == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
                             ? sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF)
                             : sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

  const size_t fc1_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, getScalingType()) * sf_size;
  const size_t fc2_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, getScalingType()) * sf_size;
  const size_t fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  const size_t tma_ws_size = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, getScalingType()) : 0;

  const size_t gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

  // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
  // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
  // in the case of fused activation we overlap permuted_data and fc2_result
  // in the case of unfused activation we overlap permuted_data and fc1_result
  // we need to calculate the max possible size, so use the max of all three
  size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
  // In the gated (glu) case fc1_result_ used to alias overlapped_gemm1_gemm2_inputs. That made the
  // fused-SwiGLU GEMV read and write the same buffer with mismatched row strides (input hidden_size,
  // output inter_size), corrupting any launch that spanned more than one wave. fc1_result_ now uses
  // its own dedicated buffer (fc1_result_dedicated below), so inputs no longer needs to fit fc1_result_size.

  const size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

  // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
  size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_outputs_size = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
  }

  // Dedicated buffer for the gated FC1 result so the fused-SwiGLU GEMV never aliases its own input
  // (see comment above). Only the gated / has-glu path uses it; zero otherwise.
  const size_t fc1_result_dedicated_size = (glu_inter_elems > 0) ? fc1_result_size : 0;

  static constexpr int kMoeGemvSplitK = 2;
  const size_t moe_gemv_splitk_partials_size = is_gated_activation && !MoeGemvSplitK2SwiGLUDisabledByEnv() &&
                                                       MoeGemvSplitK2SwiGLUSupported<T, WeightType, ScaleBiasType>()
                                                   ? kMoeGemvSplitK * glu_inter_elems * sizeof(float)
                                                   : 0;

  size_t smoothed_act_size = use_awq ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
                                     : 0;  // Extra workspace required by AWQ for smoothing activations
  size_t fp4_deep_gemm_workspace_size = 0;
#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
  if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<WeightType, __nv_bfloat16> &&
                std::is_same_v<OutputType, __nv_bfloat16> && std::is_same_v<InputType, __nv_bfloat16>) {
    if (use_fp4_deep_gemm_ && num_rows > 0 && num_rows <= deep_gemm_sm90::kMaxTokensPerExpert &&
        hidden_size == deep_gemm_sm90::kHiddenSize && inter_size == deep_gemm_sm90::kInterSize &&
        deep_gemm_sm90::NumExpertsSupported(num_experts_per_node) && experts_per_token == 6 &&
        activation_type == ActivationType::Swiglu && swiglu_fusion == 1 && !use_awq) {
      fp4_deep_gemm_workspace_size = deep_gemm_sm90::GetWorkspaceSize(num_experts_per_node);
    }
  }
#endif

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                                                \
  do {                                                                                                      \
    auto aligned_size = onnxruntime::llm::common::alignSize(size, onnxruntime::llm::common::kCudaMemAlign); \
    out_map[#name] = std::pair{aligned_size, map_offset};                                                   \
    map_offset += aligned_size;                                                                             \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD(overlapped_gemm1_gemm2_inputs);
  ADD(overlapped_gemm1_gemm2_outputs);
  ADD(fc1_result_dedicated);
  ADD(moe_gemv_splitk_partials);
  ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
  ADD_NAME(alpha_scale_ptr_array_fc2, alpha_scale_ptr_array_size);
  ADD(fp4_act_scale);
  ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
  ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
  ADD(gemm_workspace);
  ADD(smoothed_act);
  ADD(fp4_deep_gemm_workspace);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::getWorkspaceSize(
    const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size, const int num_experts,
    const int experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config,
    bool use_awq, int swiglu_fusion) {
  const int ep_size = parallelism_config.ep_size;
  ORT_ENFORCE(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
  auto sizes_map = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts / ep_size,
                                                 experts_per_token, activation_type, use_awq, swiglu_fusion);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
  size_t size = onnxruntime::llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Mixture Of Experts Plugin requires workspace of ", size / 1024.f / 1024.f, " MiB"));
  return size;
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::configureWsPtrs(char* ws_ptr,
                                                                                                      const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
                                                                                                      const int experts_per_token, ActivationType activation_type, MOEParallelismConfig parallelism_config,
                                                                                                      bool use_awq, int swiglu_fusion) {
  auto workspaces = getWorkspaceDeviceBufferSizes(num_rows, hidden_size, inter_size, num_experts_per_node,
                                                  experts_per_token, activation_type, use_awq, swiglu_fusion);

  auto getWsPtr = [&](auto type, const std::string& name) {
    return workspaces.at(name).first ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
                                     : nullptr;
  };

  permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");
  blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");

  expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");

  // We check if the provided config uses fused finalize and disable it if it does not
  const bool gemm2_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_);
  permuted_token_final_scales_ = (gemm2_using_tma_ws && mayHaveFinalizeFused()) ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

  const bool is_gated_activation = isGatedActivation(activation_type);
  const bool gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);
  const bool gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
  const bool tma_ws_has_glu = gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
  // We always use fused path if we can
  const bool non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
  const bool has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu || use_fp8;

  // Always same value, but overlapped with either fc1_result_ or fc2_result_
  permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  // Always same value, ignored if not needed
  glu_inter_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

  // fc2 aliases one of the overlapped pointers depending on whether the activation is fused/unfused.
  // NOTE: It is important to get the overlapped pointers correct as the wrong order will cause the buffer to be used
  // as an input and output for the same gemm, which will cause corruption.
  // fc1_result_ uses a dedicated buffer in the gated case so the fused-SwiGLU GEMV does not write its
  // inter_size-strided output over its own hidden_size-strided input, which corrupted multi-wave launches.
  fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "fc1_result_dedicated")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
  fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  moe_gemv_splitk_partials_ = getWsPtr(float{}, "moe_gemv_splitk_partials");

  alpha_scale_ptr_array_fc1_ = getWsPtr((const float*)(nullptr), "alpha_scale_ptr_array_fc1");
  alpha_scale_ptr_array_fc2_ = getWsPtr((const float*)(nullptr), "alpha_scale_ptr_array_fc2");

  // NOTE: We alias these, but if we fuse the quantization for GEMM2 into GEMM1 they will need separated
  fc1_fp4_act_scale_ = nullptr;
  fc2_fp4_act_scale_ = nullptr;
  if (use_block_scaling) {
    fc1_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    fc2_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    ORT_ENFORCE(fc1_fp4_act_scale_ != nullptr);
    ORT_ENFORCE(fc2_fp4_act_scale_ != nullptr);
  }

  tma_ws_grouped_gemm1_input_ = {};
  tma_ws_grouped_gemm2_input_ = {};
  if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    tma_ws_grouped_gemm1_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"),
                                                   num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
                                                   getScalingType());
    tma_ws_grouped_gemm2_input_.configureWorkspace(getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"),
                                                   num_experts_per_node, getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
                                                   getScalingType());
  }

  if (use_awq) {
    smoothed_act_ = getWsPtr(int8_t{}, "smoothed_act");
  }
  fp4_deep_gemm_workspace_ = getWsPtr(int8_t{}, "fp4_deep_gemm_workspace");
}
template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
const T* CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::applyPrequantScale(
    void* smoothed_act, const void* permuted_data, const void* prequant_scales, const int64_t* num_valid_tokens_ptr,
    const int64_t expanded_num_rows, const int64_t seq_len, const bool use_awq, cudaStream_t stream) {
  const T* gemm_input;
  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  if (use_prequant_scale_kernel) {
    ORT_ENFORCE(
        (!std::is_same_v<T, WeightType>), "Prequant scales are only used for different weight/activation type!");
    if constexpr (!std::is_same_v<T, WeightType>) {
      onnxruntime::llm::kernels::apply_per_channel_scale_kernel_launcher<UnfusedGemmOutputType, T>(
          reinterpret_cast<T*>(smoothed_act), reinterpret_cast<const UnfusedGemmOutputType*>(permuted_data),
          reinterpret_cast<const UnfusedGemmOutputType*>(prequant_scales), expanded_num_rows, seq_len,
          num_valid_tokens_ptr, stream);
    }
    gemm_input = reinterpret_cast<const T*>(smoothed_act);
  } else {
    gemm_input = reinterpret_cast<const T*>(permuted_data);
  }
  sync_check_cuda_error(stream);
  return gemm_input;
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    const T* const input, T* const output,
    void* const intermediate_result, const int64_t* const expert_first_token_offset,
    const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template, const WeightType* const fc1_expert_weights,
    const ScaleBiasType* const fc1_expert_biases, const int64_t* const num_valid_tokens_ptr,
    const ScaleBiasType* const fc1_int_scales, const float* const fc1_fp8_dequant, const float* const fc2_fp8_quant,
    const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params, const int64_t num_rows,
    const int64_t expanded_num_rows, const int64_t hidden_size, const int64_t inter_size,
    const int num_experts_per_node, ActivationType fc1_activation_type, const float** alpha_scale_ptr_array,
    const int* permuted_row_to_expert, float* moe_gemv_splitk_partials, const int* permuted_row_to_source_row,
    bool bias_is_broadcast,
    cudaStream_t stream, MOEParallelismConfig parallelism_config,
    cutlass_extensions::CutlassGemmConfig config,
    ActivationParameters activation_params) {
  const bool using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
  const bool is_gated_activation = isGatedActivation(fc1_activation_type);
  const bool use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(config, is_gated_activation, inter_size, hidden_size);
  const size_t fc1_out_size = ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

#if ORT_LLM_VERBOSE
  printf("DEBUG HOST (v5): gemm1 use_ampere_activation_fusion=%d, is_gated_activation=%d\n", use_ampere_activation_fusion, is_gated_activation);
#endif

  const int64_t* total_tokens_including_expert = expert_first_token_offset + 1;

  if (using_tma_ws_gemm1) {
    ORT_ENFORCE(config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_ampere_activation_fusion);

    ORT_ENFORCE(!use_fp4 || fc1_fp4_act_flat);
    ORT_ENFORCE(!use_fp4 || fc2_fp4_act_flat);

    bool has_different_gemm_output_type = using_tma_ws_gemm1 && !std::is_same_v<T, OutputType>;
    const bool has_intermediate = has_different_gemm_output_type || is_gated_activation;
    ORT_ENFORCE(has_intermediate || input != output, "Input and output buffers are overlapping");
    auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

    auto tma_ws_input = tma_ws_input_template;

    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    } else if constexpr (use_wfp8a16) {
      // W8A16-FP8: apply per-expert global scale via alpha in the epilogue
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.fp8.dequant_fc1, stream);
    }

    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
                                                                                   /*weights*/ nullptr, /*scales*/ nullptr, /*zeros*/ nullptr, /*biases*/ nullptr, /*C*/ nullptr,
                                                                                   alpha_scale_ptr_array, /*occupancy*/ nullptr, fc1_activation_type, num_rows,
                                                                                   /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
                                                                                   /*use_fused_moe*/ false, stream, activation_params, config};
    gemm_runner.moeGemm(universal_input, tma_ws_input);

    sync_check_cuda_error(stream);

    // TODO: when bias_is_broadcast is false, fuse bias to gemm
    using GatedActOutputType = std::conditional_t<use_w4afp8, ScaleBiasType, T>;
    bool use_per_expert_act_scale = use_fp4        ? quant_params.fp4.fc2.use_per_expert_act_scale
                                    : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.use_per_expert_act_scale
                                    : use_fp8      ? quant_params.fp8.fc2_use_per_expert_act_scale
                                                   : false;

    doActivation<GatedActOutputType, UnfusedGemmOutputType>(reinterpret_cast<GatedActOutputType*>(output),
                                                            static_cast<const UnfusedGemmOutputType*>(gemm_output),
                                                            quant_params.fp8.quant_fc2, fc1_expert_biases, bias_is_broadcast,
                                                            expert_first_token_offset, num_experts_per_node, inter_size,
                                                            expanded_num_rows, fc1_activation_type, quant_params,
                                                            use_per_expert_act_scale, fc2_fp4_act_flat, stream,
                                                            activation_params);

    sync_check_cuda_error(stream);
  } else if (use_fp8) {
    ORT_ENFORCE(!use_ampere_activation_fusion);
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_block_scaling);

    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, quant_params.fp8.dequant_fc1, stream);

    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                   total_tokens_including_expert, fc1_expert_weights, /*scales*/ nullptr, /*zeros*/ nullptr,
                                                                                   /*biases*/ nullptr, reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result), alpha_scale_ptr_array,
                                                                                   /*occupancy*/ nullptr, fc1_activation_type, expanded_num_rows, /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, /*bias_is_broadcast*/ true,
                                                                                   /*use_fused_moe*/ false, stream, activation_params, config};
    gemm_runner.moeGemm(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    bool use_per_expert_act_scale = use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale : false;
    doActivation<T, UnfusedGemmOutputType>(output, static_cast<const UnfusedGemmOutputType*>(intermediate_result),
                                           quant_params.fp8.quant_fc2, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset,
                                           num_experts_per_node, inter_size, expanded_num_rows, fc1_activation_type,
                                           quant_params, use_per_expert_act_scale, nullptr, stream, activation_params);

    sync_check_cuda_error(stream);
  } else if (!is_gated_activation) {
    ORT_ENFORCE(!use_ampere_activation_fusion);
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    }
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                   total_tokens_including_expert, fc1_expert_weights,
                                                                                   /*scales*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_scales)
                                                                                       : fc1_int_scales,
                                                                                   /*zeros*/ quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_zeros)
                                                                                       : nullptr,
                                                                                   fc1_expert_biases, reinterpret_cast<OutputType*>(output), alpha_scale_ptr_array, /*occupancy*/ nullptr,
                                                                                   fc1_activation_type == ActivationType::Relu2
                                                                                       ? ActivationType::Identity
                                                                                       : fc1_activation_type,
                                                                                   expanded_num_rows, /*N*/ int64_t(fc1_out_size),
                                                                                   /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
                                                                                   /*use_fused_moe*/ false, stream, activation_params, config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    if (fc1_activation_type == ActivationType::Relu2) {
      // Preserve the FC1 output-type rounding before squaring; the GEMM already added its bias.
      doActivation<T, OutputType>(
          output, reinterpret_cast<const OutputType*>(output), nullptr,
          static_cast<const OutputType*>(nullptr), bias_is_broadcast,
          expert_first_token_offset, num_experts_per_node, inter_size, expanded_num_rows,
          fc1_activation_type, quant_params, false, nullptr, stream, activation_params);
    }

    sync_check_cuda_error(stream);
  } else {
    ORT_ENFORCE(!config.is_tma_warp_specialized);
    ORT_ENFORCE(is_gated_activation);
    ORT_ENFORCE(
        !use_ampere_activation_fusion || input != output, "Input and output buffers are overlapping");
    ORT_ENFORCE(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(
          alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc1.alpha, stream);
    }
    const bool use_splitk_swiglu_gemv = !MoeGemvSplitK2SwiGLUDisabledByEnv() &&
                                        MoeGemvSplitK2SwiGLUSupported<T, WeightType, ScaleBiasType>();
    ORT_ENFORCE(!use_splitk_swiglu_gemv || moe_gemv_splitk_partials != nullptr,
                "Split-K2 SwiGLU GEMV requires split-K GEMV workspace");
    const bool fc1_did_fused_gemv = tryLaunchMoeGemvIntSymmetricInterleavedSwiGLU<T, WeightType, ScaleBiasType>(
        input, fc1_expert_weights,
        quant_params.groupwise.group_size > 0
            ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_scales)
            : fc1_int_scales,
        quant_params.groupwise.group_size > 0
            ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_zeros)
            : nullptr,
        fc1_expert_biases, output, expert_first_token_offset, num_experts_per_node,
        permuted_row_to_expert, expanded_num_rows, inter_size, hidden_size,
        onnxruntime::llm::common::getSMVersion(), quant_params.groupwise.group_size,
        /*disabled=*/
        MoeGemvFc1Disabled(parallelism_config.ep_size, use_ampere_activation_fusion, bias_is_broadcast,
                           expanded_num_rows, inter_size),
        activation_params,
        permuted_row_to_source_row, num_rows,
        use_splitk_swiglu_gemv ? moe_gemv_splitk_partials : nullptr,
        stream);
    // One-sided guard for the skip-expand path only. On the normal path permuted_row_to_source_row
    // is nullptr and this check is trivially satisfied. When it is non-null the caller skipped
    // expandInputRows and handed us the unexpanded activations plus the permuted-row map; only the
    // GEMV understands that layout, so bail loudly rather than silently feeding a wrongly-shaped
    // buffer to the grouped GEMM below.
    ORT_ENFORCE(permuted_row_to_source_row == nullptr || fc1_did_fused_gemv,
                "FC1 activations were not expanded but the MoE GEMV did not run");

    // Run the GEMM with activation function overridden with `Identity`, we do the activation separately.
    // Fast path: symmetric INT4/INT8 (per-column or block-wise) MoE GEMV for small expanded-row counts
    // (e.g. batch-1 decode).
    const bool fc1_did_gemv = fc1_did_fused_gemv || tryLaunchMoeGemvIntSymmetric<T, WeightType, ScaleBiasType>(
                                                        input, fc1_expert_weights,
                                                        quant_params.groupwise.group_size > 0
                                                            ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_scales)
                                                            : fc1_int_scales,
                                                        quant_params.groupwise.group_size > 0
                                                            ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_zeros)
                                                            : nullptr,
                                                        fc1_expert_biases, static_cast<T*>(intermediate_result), expert_first_token_offset, num_experts_per_node,
                                                        permuted_row_to_expert, expanded_num_rows, /*n=*/static_cast<int64_t>(fc1_out_size), /*k=*/hidden_size,
                                                        onnxruntime::llm::common::getSMVersion(),
                                                        quant_params.groupwise.group_size,
                                                        /*disabled=*/
                                                        MoeGemvFc1Disabled(parallelism_config.ep_size,
                                                                           use_ampere_activation_fusion, bias_is_broadcast,
                                                                           expanded_num_rows, inter_size),
                                                        stream);
    if (!fc1_did_gemv) {
      auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                                     total_tokens_including_expert, fc1_expert_weights,
                                                                                     /*scales*/ quant_params.groupwise.group_size > 0
                                                                                         ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_scales)
                                                                                         : fc1_int_scales,
                                                                                     /*zeros*/ quant_params.groupwise.group_size > 0
                                                                                         ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_zeros)
                                                                                         : nullptr,
                                                                                     fc1_expert_biases, static_cast<OutputType*>(use_ampere_activation_fusion ? output : intermediate_result),
                                                                                     alpha_scale_ptr_array, /*occupancy*/ nullptr,
                                                                                     use_ampere_activation_fusion ? fc1_activation_type : ActivationType::Identity, expanded_num_rows,
                                                                                     /*N*/ int64_t(fc1_out_size),
                                                                                     /*K*/ hidden_size, num_experts_per_node, quant_params.groupwise.group_size, bias_is_broadcast,
                                                                                     use_ampere_activation_fusion, stream, activation_params, config};
      gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});
    }

    sync_check_cuda_error(stream);

    if (!use_ampere_activation_fusion && !fc1_did_fused_gemv) {
      using GatedActOutputType = std::conditional_t<use_w4afp8, ScaleBiasType, T>;
      if (is_gated_activation) {
        doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(
            reinterpret_cast<GatedActOutputType*>(output),
            static_cast<const UnfusedGemmOutputType*>(intermediate_result),
            num_valid_tokens_ptr, inter_size, expanded_num_rows, fc1_activation_type, stream, activation_params);
      } else {
        doActivation<GatedActOutputType, UnfusedGemmOutputType>(
            reinterpret_cast<GatedActOutputType*>(output),
            static_cast<const UnfusedGemmOutputType*>(intermediate_result),
            nullptr, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset,
            num_experts_per_node, inter_size, expanded_num_rows, fc1_activation_type, quant_params,
            false, nullptr, stream, activation_params);
      }

      sync_check_cuda_error(stream);
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
    const T* const input, void* const gemm_output,
    OutputType* const final_output, const int64_t* const expert_first_token_offset,
    const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template, const WeightType* const fc2_expert_weights,
    const ScaleBiasType* const fc2_expert_biases, const ScaleBiasType* const fc2_int_scales,
    const float* const fc2_fp8_dequant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
    QuantParams quant_params, const float* const unpermuted_final_scales, const float* const permuted_final_scales,
    const int* const unpermuted_row_to_permuted_row, const int* permuted_row_to_unpermuted_row,
    const int* const token_selected_experts, const int64_t* const num_valid_tokens_ptr, const int64_t num_rows,
    const int64_t expanded_num_rows, const int64_t hidden_size, const int64_t inter_size,
    const int num_experts_per_node, const int64_t k, const float** alpha_scale_ptr_array,
    const int* permuted_row_to_expert,
    cudaStream_t stream, MOEParallelismConfig parallelism_config,
    cutlass_extensions::CutlassGemmConfig config) {
  const int64_t* total_tokens_including_expert = expert_first_token_offset + 1;

  const bool using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

  TmaWarpSpecializedGroupedGemmInput tma_ws_input{};
  if (using_tma_ws_gemm2) {
    tma_ws_input = tma_ws_input_template;
    if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
      // TODO For some reason this has to be done here, it should not overlap with anything else, but
      // doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want this to run on a
      // second stream and overlap with everything else
      //
      // This also means it is included in the timing for the profiler, which is probably more representative
      // until we can overlap it
      CUDA_CALL_THROW(cudaMemsetAsync(final_output, 0x0, sizeof(OutputType) * num_rows * hidden_size, stream));
    }
  } else if (use_fp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
  }
  if (use_w4afp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(
        alpha_scale_ptr_array, num_experts_per_node, quant_params.groupwise.fc2.alpha, stream);
  } else if constexpr (use_wfp8a16) {
    // W8A16-FP8: apply per-expert fc2 global scale via alpha in the epilogue
    alpha_scale_ptr_array = computeFP8DequantScale(
        alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
  }

  ActivationParameters activation_params;  // Here assume gemm2 has no activation
                                           // Note: expanded_num_rows, to check this value, it's greater than num_rows * num_experts_per_node
                                           // Fast path: symmetric INT4/INT8 (per-column or block-wise) MoE GEMV
                                           // (no bias here; fc2 bias applied in finalizeMoeRouting).
                                           // Keep the decode GEMV path single-EP until token-drop/all-to-all
                                           // cases are profiled and validated.
  const auto fc2_gemv_scales = quant_params.groupwise.group_size > 0
                                   ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc2.weight_scales)
                                   : fc2_int_scales;
  const auto fc2_gemv_zeros = quant_params.groupwise.group_size > 0
                                  ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc2.weight_zeros)
                                  : nullptr;
  const bool fc2_gemv_disabled = parallelism_config.ep_size > 1 || using_tma_ws_gemm2 ||
                                 MoeGemvRejectedByProfiledInterSize(expanded_num_rows, inter_size);

  // Preferred fast path: the same GEMV, but with the top_k finalize reduction folded into its
  // epilogue so `final_output` is written directly and no separate finalize kernel is launched.
  // Restricted to single-TP as well as single-EP: the fused epilogue always adds the FC2 bias,
  // whereas finalizeMoeRouting only adds it on tp_rank 0.
  const bool fc2_did_fused_finalize =
      tryLaunchMoeGemvIntSymmetricFusedFinalize<T, WeightType, ScaleBiasType, OutputType>(
          input, fc2_expert_weights, fc2_gemv_scales, fc2_gemv_zeros, fc2_expert_biases, final_output,
          unpermuted_row_to_permuted_row, permuted_row_to_expert, unpermuted_final_scales, num_experts_per_node,
          num_rows, /*experts_per_token=*/k, expanded_num_rows, /*n=*/hidden_size, /*k=*/inter_size,
          onnxruntime::llm::common::getSMVersion(), quant_params.groupwise.group_size,
          /*disabled=*/fc2_gemv_disabled || parallelism_config.tp_size > 1, stream);

  const bool fc2_did_gemv = fc2_did_fused_finalize ||
                            tryLaunchMoeGemvIntSymmetric<T, WeightType, ScaleBiasType>(
                                input, fc2_expert_weights, fc2_gemv_scales, fc2_gemv_zeros,
                                /*biases*/ nullptr, static_cast<T*>(gemm_output), expert_first_token_offset,
                                num_experts_per_node, permuted_row_to_expert, expanded_num_rows,
                                /*n=*/hidden_size, /*k=*/inter_size, onnxruntime::llm::common::getSMVersion(),
                                quant_params.groupwise.group_size,
                                /*disabled=*/fc2_gemv_disabled, stream);
  if (!fc2_did_gemv) {
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{input, total_tokens_including_expert,
                                                                                   fc2_expert_weights,
                                                                                   quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc2.weight_scales)
                                                                                       : fc2_int_scales,
                                                                                   quant_params.groupwise.group_size > 0
                                                                                       ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc2.weight_zeros)
                                                                                       : nullptr,
                                                                                   nullptr, static_cast<OutputType*>(gemm_output),
                                                                                   alpha_scale_ptr_array, /*occupancy*/ nullptr, ActivationType::Identity, expanded_num_rows,
                                                                                   /*N*/ hidden_size,
                                                                                   /*K*/ inter_size,
                                                                                   num_experts_per_node,
                                                                                   quant_params.groupwise.group_size,
                                                                                   /*bias_is_broadcast*/ false,
                                                                                   /*use_fused_moe*/ false,
                                                                                   stream,
                                                                                   activation_params,
                                                                                   config};
    gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
  }
  sync_check_cuda_error(stream);

  bool has_different_output_type_ampere = (use_w4afp8 || use_fp8) && !using_tma_ws_gemm2;
  bool using_hopper_fused_finalize = tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool has_different_output_type_tma_ws = !using_hopper_fused_finalize && using_tma_ws_gemm2;

  if (fc2_did_fused_finalize) {
    // final_output was produced by the fused GEMV epilogue; nothing left to reduce.
  } else if (has_different_output_type_ampere || has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<const UnfusedGemmOutputType*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
        parallelism_config, /*enable_alltoall=*/false, stream);
  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(static_cast<const T*>(gemm_output), final_output,
                                                    fc2_expert_biases, unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
                                                    token_selected_experts, expert_first_token_offset, num_rows, hidden_size, k, num_experts_per_node,
                                                    parallelism_config, /*enable_alltoall=*/false, stream);
  }
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::runMoe(
    const void* input_activations_void, const void* input_sf_void, const int* token_selected_experts,
    const float* token_final_scales, const void* fc1_expert_weights_void, const void* fc1_expert_biases_void,
    ActivationType fc1_activation_type, const void* fc2_expert_weights_void, const void* fc2_expert_biases_void,
    QuantParams quant_params, const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size,
    const int full_num_experts, const int experts_per_token, char* workspace_ptr, void* final_output_void,
    int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
    ActivationParameters activation_params, FusedRoutingParams fused_routing,
    cudaStream_t stream) {
  static constexpr bool int_scales_required = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
  static constexpr bool fp8_scales_required = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

  const auto* input_activations = static_cast<const InputType*>(input_activations_void);
  const auto* input_sf = input_sf_void
                             ? reinterpret_cast<const TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf_void)
                             : nullptr;
  const auto* fc1_expert_weights = static_cast<const WeightType*>(fc1_expert_weights_void);
  const auto* fc1_expert_biases = reinterpret_cast<const ScaleBiasType*>(fc1_expert_biases_void);
  const auto* fc2_expert_weights = static_cast<const WeightType*>(fc2_expert_weights_void);
  const auto* fc1_int_scales = reinterpret_cast<const ScaleBiasType*>(quant_params.wo.fc1_weight_scales);
  const auto* fc2_int_scales = reinterpret_cast<const ScaleBiasType*>(quant_params.wo.fc2_weight_scales);

  const auto* fc1_fp8_dequant = quant_params.fp8.dequant_fc1;
  const auto* fc2_fp8_quant = quant_params.fp8.quant_fc2;
  const auto* fc2_fp8_dequant = quant_params.fp8.dequant_fc2;

  const auto* fc2_wfp4afp8_quant_scale = quant_params.fp8_mxfp4.fc2.act_global_scale;

  const auto* fc2_expert_biases = reinterpret_cast<const ScaleBiasType*>(fc2_expert_biases_void);
  auto* final_output = static_cast<OutputType*>(final_output_void);
  const float* token_topk_unpermuted_scales = token_final_scales;

  ORT_ENFORCE(input_activations);
  ORT_ENFORCE(token_selected_experts);
  ORT_ENFORCE(fc1_expert_weights);
  ORT_ENFORCE(fc2_expert_weights);
  ORT_ENFORCE(workspace_ptr);
  // ORT_ENFORCE(token_topk_unpermuted_scales);
  ORT_ENFORCE(unpermuted_row_to_permuted_row);
  ORT_ENFORCE(full_num_experts % parallelism_config.ep_size == 0);
  ORT_ENFORCE(full_num_experts % parallelism_config.cluster_size == 0);

  if (quant_params.mxfp8_mxfp4.fc1.weight_block_scale) {
    // 64-element (32-byte) K alignment for the FP4 weight. The original guard required
    // 128 elements (64 bytes), but the WFP4A16 SM90 grouped GEMM only needs the weight K to
    // satisfy the FP4 block-scale vector (32) and TMA (16-byte) alignment; the offline scale
    // prepack (PrePackFp4ScalesForTmaWs) zero-pads each expert's k-block count up to a multiple
    // of the packed K-tile (8) so the last partial CTA-K-tile can read a full packed group.
    // Relaxing 128->64 admits gpt-oss-20b (hidden=inter=2880, 64-aligned but not 128-aligned);
    // validated correct (matches dequant fallback) and clean under compute-sanitizer memcheck.
    ORT_ENFORCE(hidden_size % (32 * 8 / sizeof_bits<WeightType>::value) == 0,
                "Hidden size %d does not meet minimum alignment requirements for MXFP8_MXFP4 MOE GEMM %d",
                (int)hidden_size, (int)(32 * 8 / sizeof_bits<WeightType>::value));
    ORT_ENFORCE(inter_size % (32 * 8 / sizeof_bits<WeightType>::value) == 0,
                "Inter size %d does not meet minimum alignment requirements for MXFP8_MXFP4 MOE GEMM %d", (int)inter_size,
                (int)(32 * 8 / sizeof_bits<WeightType>::value));
  } else {
    // Require at least 128 bits of alignment for MOE GEMM
    ORT_ENFORCE(hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
                "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d", (int)hidden_size,
                (int)(128 / sizeof_bits<WeightType>::value));
    ORT_ENFORCE(inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
                "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d", (int)inter_size,
                (int)(128 / sizeof_bits<WeightType>::value));
  }

  // These values must fit into an int for building the source maps
  ORT_ENFORCE(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
  ORT_ENFORCE(
      num_rows * full_num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
  ORT_ENFORCE(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
              "experts_per_token * num_experts is too large");

  ORT_ENFORCE(gemm1_config_, "MOE GEMM1 Config is not set");
  ORT_ENFORCE(gemm2_config_, "MOE GEMM2 Config is not set");

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  const bool use_ampere_activation_fusion = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);

  if (int_scales_required) {
    if (!(quant_params.groupwise.fc1.weight_scales && quant_params.groupwise.fc2.weight_scales)) {
      ORT_ENFORCE(
          fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
      ORT_ENFORCE(
          fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");
    }
    ORT_ENFORCE(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
                "FP8 scales are provided for integer quantization");
  } else if (fp8_scales_required) {
    ORT_ENFORCE(
        fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
    if constexpr (!use_wfp8a16) {
      // Pure FP8 (T == WeightType == FP8) needs quant_fc2 to quantize intermediate activations.
      // W8A16-FP8 does NOT need quant_fc2 since activations stay in FP16/BF16.
      ORT_ENFORCE(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
    }
    ORT_ENFORCE(
        fc2_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC2 is a null pointer");

    ORT_ENFORCE(
        fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
  } else {
    ORT_ENFORCE(
        fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
    ORT_ENFORCE(
        fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
    ORT_ENFORCE(
        fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
    ORT_ENFORCE(
        fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    ORT_ENFORCE(
        fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;
  const int num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token,
                  fc1_activation_type, parallelism_config, use_awq, activation_params.swiglu_fusion);

  int start_expert = num_experts_per_node * parallelism_config.ep_rank;
  int end_expert = start_expert + num_experts_per_node;

  const bool needs_num_valid = parallelism_config.ep_size > 1;
  const int64_t* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  {
    // Optimization C: when the caller handed us the raw router logits, softmax + top-k has not run
    // yet and we fold it into the expert-map prologue, saving a kernel launch per MoE layer. The
    // caller gates on the same predicate, so the two cannot disagree about who computed the routing.
    bool const use_fused_routing = fused_routing.router_logits != nullptr;
    bool fused_prologue_result = use_fused_routing;
    if (use_fused_routing) {
      ORT_ENFORCE(!use_w4afp8, "Fused MoE routing is not supported for W4AFP8");
      ORT_ENFORCE(isFusedMoeRoutingSupported(num_rows, full_num_experts, num_experts_per_node,
                                             experts_per_token, parallelism_config.ep_size),
                  "Fused MoE routing was requested for an unsupported configuration");
      ORT_ENFORCE(fused_routing.token_selected_experts && fused_routing.token_final_scales);
      launchFusedRoutingBuildExpertMaps<InputType>(
          static_cast<InputType const*>(fused_routing.router_logits), fused_routing.token_selected_experts,
          fused_routing.token_final_scales, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
          permuted_token_selected_experts_, expert_first_token_offset_, full_num_experts, experts_per_token,
          fused_routing.normalize_routing_weights, stream);
      token_selected_experts = fused_routing.token_selected_experts;
      token_topk_unpermuted_scales = fused_routing.token_final_scales;
      if (fused_routing.on_routing_ready != nullptr) {
        fused_routing.on_routing_ready(fused_routing.routing_context, token_selected_experts,
                                       static_cast<size_t>(expanded_num_rows), stream);
      }
    } else if (!use_w4afp8) {
      // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for W4AFP8
      fused_prologue_result = fusedBuildExpertMapsSortFirstToken(token_selected_experts,
                                                                 permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
                                                                 permuted_token_selected_experts_, expert_first_token_offset_, num_rows,
                                                                 num_experts_per_node, experts_per_token, start_expert, end_expert,
                                                                 stream);
    }

    if (!fused_prologue_result) {
      ORT_LLM_LOG_DEBUG("Falling back to unfused prologue");
      threeStepBuildExpertMapsSortFirstToken(token_selected_experts, permuted_token_selected_experts_,
                                             permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row, expert_first_token_offset_,
                                             blocked_expert_counts_, blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
                                             num_experts_per_node, experts_per_token, start_expert, stream);
    }

    sync_check_cuda_error(stream);

    bool is_gated_activation = isGatedActivation(fc1_activation_type);

    // Only NVFP4xNVFP4 supports FC1 per-expert act scale
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc1.use_per_expert_act_scale : false;
    T* gemm1_input_expand = use_w4afp8 ? reinterpret_cast<T*>(smoothed_act_) : reinterpret_cast<T*>(permuted_data_);

    // Optimization B: when FC1 is served by the interleaved-SwiGLU MoE GEMV, the GEMV can read the
    // unexpanded activations directly through permuted_row_to_unpermuted_row_, so replicating every
    // token experts_per_token times is pure overhead (an extra kernel plus a redundant write/read of
    // expanded_num_rows * hidden_size elements). The GEMV path only runs for tiny expanded-row counts
    // (decode), which is exactly where that fixed overhead dominates.
    //
    // expandInputRows also converts InputType -> T, applies smoothing/act-scale generation and fills
    // permuted_token_final_scales_ for the Hopper fused finalize, so the skip is only legal when none
    // of those side effects are needed.
    const bool expand_is_pure_permutation =
        std::is_same_v<InputType, T> && !use_w4afp8 && !use_fp4 && !use_fp8 && !use_wfp4a16 && !use_wfp4afp8 &&
        !use_awq && permuted_token_final_scales_ == nullptr && input_sf == nullptr && !use_per_expert_act_scale;
    const bool fc1_gemv_will_run =
        !moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) && !use_fp8 && is_gated_activation &&
        moeGemvInterleavedSwiGLUWillRun<T, WeightType, ScaleBiasType>(
            quant_params.groupwise.group_size > 0
                ? static_cast<const ScaleBiasType*>(quant_params.groupwise.fc1.weight_zeros)
                : nullptr,
            expanded_num_rows, inter_size, hidden_size, onnxruntime::llm::common::getSMVersion(),
            quant_params.groupwise.group_size,
            // Must match the term gemm1 passes below, including the bias_is_broadcast literal.
            /*disabled=*/
            MoeGemvFc1Disabled(parallelism_config.ep_size, use_ampere_activation_fusion,
                               /*bias_is_broadcast=*/true, expanded_num_rows, inter_size),
            activation_params);
    const bool skip_expand_input_rows =
        !MoeGemvSkipExpandDisabledByEnv() && expand_is_pure_permutation && fc1_gemv_will_run;

    // nullptr means "the activations have already been expanded by expandInputRows", which is the
    // layout every FC1 consumer understands. It stays nullptr on the non-skip path; only when
    // skip_expand_input_rows is true do we hand down the permutation map, and then the fused GEMV
    // is the only kernel that can consume the unexpanded activations.
    const int* fc1_permuted_row_to_source_row = nullptr;
    if (skip_expand_input_rows) {
      fc1_permuted_row_to_source_row = permuted_row_to_unpermuted_row_;
    } else {
      expandInputRowsKernelLauncher(input_activations, gemm1_input_expand, token_topk_unpermuted_scales,
                                    permuted_token_final_scales_, permuted_row_to_unpermuted_row_, num_rows, hidden_size, experts_per_token,
                                    num_experts_per_node, quant_params, use_per_expert_act_scale, expert_first_token_offset_,
                                    fc1_fp4_act_scale_, input_sf, use_w4afp8 ? quant_params.groupwise.fc1.act_scales : nullptr, stream);
    }
    const auto* gemm1_input = skip_expand_input_rows ? reinterpret_cast<const T*>(input_activations)
                                                     : static_cast<const T*>(gemm1_input_expand);

    sync_check_cuda_error(stream);

#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
    if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<WeightType, __nv_bfloat16> &&
                  std::is_same_v<OutputType, __nv_bfloat16> && std::is_same_v<InputType, __nv_bfloat16>) {
      const bool use_fp4_deep_gemm =
          use_fp4_deep_gemm_ && fp4_deep_gemm_workspace_ != nullptr && num_rows > 0 &&
          fp4_deep_gemm_fc1_weight_scales_ != nullptr && fp4_deep_gemm_fc2_weight_scales_ != nullptr &&
          num_rows <= deep_gemm_sm90::kMaxTokensPerExpert &&
          hidden_size == deep_gemm_sm90::kHiddenSize && inter_size == deep_gemm_sm90::kInterSize &&
          deep_gemm_sm90::NumExpertsSupported(num_experts_per_node) && experts_per_token == 6 &&
          fc1_activation_type == ActivationType::Swiglu && activation_params.swiglu_fusion == 1 &&
          fc1_expert_biases == nullptr && fc2_expert_biases == nullptr && !use_awq && input_sf == nullptr &&
          parallelism_config.tp_size == 1 && parallelism_config.ep_size == 1 &&
          parallelism_config.cluster_size == 1 && quant_params.groupwise.group_size <= 0 &&
          quant_params.wo.fc1_weight_scales == nullptr && quant_params.wo.fc2_weight_scales == nullptr;
      if (use_fp4_deep_gemm) {
        // WeightType stays bf16 at the ORT level; the prepacked DeepGEMM buffers are e4m3 bytes.
        deep_gemm_sm90::Run(
            reinterpret_cast<const __nv_bfloat16*>(permuted_data_), expert_first_token_offset_,
            reinterpret_cast<const __nv_fp8_e4m3*>(fc1_expert_weights), fp4_deep_gemm_fc1_weight_scales_,
            reinterpret_cast<const __nv_fp8_e4m3*>(fc2_expert_weights), fp4_deep_gemm_fc2_weight_scales_,
            reinterpret_cast<__nv_bfloat16*>(fc2_result_), num_experts_per_node, activation_params.alpha,
            activation_params.beta, activation_params.limit, fp4_deep_gemm_workspace_, stream);
        finalizeMoeRoutingKernelLauncher<OutputType, T, ScaleBiasType>(
            static_cast<const T*>(fc2_result_), final_output, nullptr, token_topk_unpermuted_scales,
            unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_, token_selected_experts,
            expert_first_token_offset_, num_rows, hidden_size, experts_per_token, num_experts_per_node,
            parallelism_config, /*enable_alltoall=*/false, stream);
        sync_check_cuda_error(stream);
        return;
      }
    }
#endif

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(num_rows, expanded_num_rows,
                                                                                  fc1_activation_type, use_ampere_activation_fusion, hidden_size, inter_size, num_experts_per_node, input_activations_void, input_sf,
                                                                                  final_output, fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
                                                                                  start_expert, parallelism_config, stream);

    if constexpr (!use_w4afp8) {
      // With the expansion skipped there is nothing to smooth in place: the skip already requires
      // use_awq == false, so applyPrequantScale would just echo its second argument back.
      if (!skip_expand_input_rows) {
        gemm1_input = applyPrequantScale(smoothed_act_, permuted_data_, quant_params.groupwise.fc1.act_scales,
                                         num_valid_tokens_ptr, expanded_num_rows, hidden_size, use_awq, stream);
      }
    }
    sync_check_cuda_error(stream);
    Self::gemm1(moe_gemm_runner_, gemm1_input, fc1_result_, glu_inter_result_,
                expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr,
                fc1_int_scales, fc1_fp8_dequant, use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant,
                fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_fc1_, permuted_token_selected_experts_,
                moe_gemv_splitk_partials_, fc1_permuted_row_to_source_row, /*bias_is_broadcast=*/true, stream,
                parallelism_config, *gemm1_config_, activation_params);
    sync_check_cuda_error(stream);

    auto gemm2_input = applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                                          num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    sync_check_cuda_error(stream);
    Self::gemm2(moe_gemm_runner_, gemm2_input, fc2_result_, final_output,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
                fc2_fp8_dequant, fc2_fp4_act_scale_, quant_params, token_topk_unpermuted_scales,
                permuted_token_final_scales_, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, experts_per_token, alpha_scale_ptr_array_fc2_, permuted_token_selected_experts_, stream,
                parallelism_config, *gemm2_config_);
    sync_check_cuda_error(stream);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::computeStridesTmaWarpSpecialized(
    const int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n,
    int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, const int num_experts_per_node, const T* gemm1_in,
    const T* gemm2_in, const WeightType* weights1, const WeightType* weights2, const float* fp8_dequant1,
    const float* fp8_dequant2, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat1,
    const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat2, QuantParams quant_params,
    const ScaleBiasType* bias1, const ScaleBiasType* bias2, UnfusedGemmOutputType* gemm1_output,
    UnfusedGemmOutputType* gemm2_output, cudaStream_t stream) {
  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;

  auto alpha_scale_flat1 = use_fp4        ? quant_params.fp4.fc1.global_scale
                           : use_wfp4a16  ? (quant_params.mxfp8_mxfp4.fc1.global_scale
                                                 ? quant_params.mxfp8_mxfp4.fc1.global_scale
                                                 : quant_params.fp8_mxfp4.fc1.global_scale)
                           : use_wfp4afp8 ? (quant_params.fp8_mxfp4.fc1.global_scale
                                                 ? quant_params.fp8_mxfp4.fc1.global_scale
                                                 : quant_params.mxfp8_mxfp4.fc1.global_scale)
                           : use_fp8      ? fp8_dequant1
                                          : nullptr;
  auto alpha_scale_flat2 = use_fp4        ? quant_params.fp4.fc2.global_scale
                           : use_wfp4a16  ? (quant_params.mxfp8_mxfp4.fc2.global_scale
                                                 ? quant_params.mxfp8_mxfp4.fc2.global_scale
                                                 : quant_params.fp8_mxfp4.fc2.global_scale)
                           : use_wfp4afp8 ? (quant_params.fp8_mxfp4.fc2.global_scale
                                                 ? quant_params.fp8_mxfp4.fc2.global_scale
                                                 : quant_params.mxfp8_mxfp4.fc2.global_scale)
                           : use_fp8      ? fp8_dequant2
                                          : nullptr;
  if (!alpha_scale_flat1 && !alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array = nullptr;
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = use_w4afp8 || use_wfp4a16 || quant_params.groupwise.group_size > 0;
  layout_info2.int4_groupwise_params.enabled = use_w4afp8 || use_wfp4a16 || quant_params.groupwise.group_size > 0;
  layout_info1.int4_groupwise_params.use_wfp4a16 = use_wfp4a16;
  layout_info2.int4_groupwise_params.use_wfp4a16 = use_wfp4a16;

  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  const int threads = std::min(1024, num_experts_per_node);
  const int blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance = &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1, layout_info2, num_tokens,
                     expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1,
                     weights2, alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params, bias1, bias2,
                     gemm1_output, gemm2_output);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, Enable>::setupTmaWarpSpecializedInputs(
    int64_t num_rows, int64_t expanded_num_rows, ActivationType fc1_activation_type, bool use_fused_gated_activation,
    int64_t hidden_size, int64_t inter_size, int64_t num_experts_per_node, const void* input_activations_void,
    const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, void* final_output,
    const WeightType* fc1_expert_weights, const WeightType* fc2_expert_weights, QuantParams quant_params,
    const ScaleBiasType* fc1_expert_biases, const ScaleBiasType* fc2_expert_biases,
    int start_expert, MOEParallelismConfig parallelism_config,
    cudaStream_t stream) {
  auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
  auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) && !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales;

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  const int64_t fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

  bool has_different_gemm_output_type = !std::is_same_v<T, UnfusedGemmOutputType>;
  const bool has_intermediate = has_different_gemm_output_type || is_gated_activation;
  auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  auto gemm2_input = use_prequant_scale_kernel ? smoothed_act_ : fc1_result_;

  {
    auto gemm1_input = use_prequant_scale_kernel ? smoothed_act_ : permuted_data_;

    gemm1_tma_ws_input.fusion = (is_gated_activation && use_fused_gated_activation)
                                    ? TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION
                                    : TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

    bool apply_bias = parallelism_config.tp_rank == 0;
    bool using_hopper_fused_finalize = !use_deterministic_hopper_reduce_ && gemm2_config_->sm_version == 90 &&
                                       !use_w4afp8 && !use_wfp4a16;
    if (using_hopper_fused_finalize) {
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
      gemm2_tma_ws_input.setFinalizeFusionParams(final_output, permuted_token_final_scales_,
                                                 expert_first_token_offset_, permuted_row_to_unpermuted_row_, apply_bias ? fc2_expert_biases : nullptr,
                                                 hidden_size, num_rows);
    }

    // fp8_mxfp4 memsets the scaling factors to 1.0f
    if (quant_params.fp8_mxfp4.fc1.weight_block_scale) {
      // We are in FP8 x MXFP4 mode
      ORT_ENFORCE(quant_params.fp8_mxfp4.fc2.weight_block_scale);
      ORT_ENFORCE(fc1_fp4_act_scale_ != nullptr);
      ORT_ENFORCE(fc1_fp4_act_scale_ == fc2_fp4_act_scale_,
                  "WFP4AFP8 expects the scaling factors to be aliased for gemm1 & gemm2");

      TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF weight_block_scale_value_int{};
#ifdef ENABLE_FP8
      __nv_fp8_e8m0 tmp;
      tmp.__x = __nv_cvt_float_to_e8m0(1.0f, __NV_SATFINITE, cudaRoundPosInf);
      std::memcpy(&weight_block_scale_value_int, &tmp, sizeof(tmp));
#endif

      auto act_sf_rows = std::min(expanded_num_rows, num_rows * num_experts_per_node);
      auto fc1_sf_offset = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size,
                                                 TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto fc2_sf_offset = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size,
                                                 TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto max_size = std::max(fc1_sf_offset, fc2_sf_offset) * sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF);
      CUDA_CALL_THROW(cudaMemsetAsync(fc1_fp4_act_scale_, weight_block_scale_value_int, max_size, stream));
    }

    ORT_ENFORCE(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecialized(expert_first_token_offset_, gemm1_tma_ws_input,
                                                  gemm2_tma_ws_input, num_rows, expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size,
                                                  num_experts_per_node, reinterpret_cast<const T*>(gemm1_input), reinterpret_cast<const T*>(gemm2_input),
                                                  fc1_expert_weights, fc2_expert_weights, quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2,
                                                  fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, fc1_expert_biases, fc2_expert_biases,
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), stream);
  }
}

// ==================== Helper for getting load balanced routing for profiling ==================================

__global__ void prepareFakeRouterBuffers(
    int* token_selected_experts, int64_t num_tokens, int64_t k, int64_t num_experts) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t sample = blockIdx.y;
  if (tid >= num_tokens) {
    return;
  }

  // Offset the buffers to the start of the sample
  token_selected_experts += sample * num_tokens * k;

  // This is not perf sensitive we just init the state here every time prepare is called
  // This means the first N tokens will always have the same distribution, regardless of num_tokens
  curandStatePhilox4_32_10_t state;
  curand_init(sample, tid, 0, &state);
  for (int k_idx = 0; k_idx < k; k_idx++) {
    while (true) {
      // curand_uniform includes 1 but not 0, so round up and subtract 1
      int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

      bool valid = true;
      for (int prev_k = 0; prev_k < k_idx; prev_k++) {
        int prev_expert = token_selected_experts[k * tid + prev_k];
        if (expert == prev_expert) {
          valid = false;
          break;
        }
      }

      if (valid) {
        token_selected_experts[k * tid + k_idx] = expert;
        break;
      }
    }
  }
}

__global__ void populateRandomBufferKernel(void* buffer_void, size_t size) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) {
    return;
  }

  curandStatePhilox4_32_10_t state;
  curand_init(size, tid, 0, &state);

  constexpr int elem_per_thread = 128 / sizeof(uint4);
  auto* buffer = reinterpret_cast<uint4*>(buffer_void);
#pragma unroll
  for (int i = 0; i < elem_per_thread; i++)
    buffer[tid * elem_per_thread + i] = curand4(&state);
}

void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream) {
  // Each thread initialises 128 bytes
  ORT_ENFORCE(size % 128 == 0, "Unexpected size alignment");
  auto threads = size / 128;
  populateRandomBufferKernel<<<ceilDiv(threads, 128), 128, 0, stream>>>(buffer_void, threads);
}

std::map<std::string, std::pair<size_t, size_t>> GemmProfilerBackend::getProfilerWorkspaces(
    int maxM, bool is_tma_ws_input) {
  size_t k = mK;
  size_t num_expanded_tokens = maxM * k;

  ORT_ENFORCE(mDType != nvinfer::DataType::kINT4);
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  bool is_4bit_act = mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64;
  bool is_4bit_weight = mWType == nvinfer::DataType::kINT4 || mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64;
  ORT_ENFORCE(!is_4bit_act || is_4bit_weight, "Cannot have 4-bit activation with non-4-bit weight");
  float dtype_bytes = is_4bit_act
                          ? 0.5f
                          : static_cast<float>(mWType == nvinfer::DataType::kINT4 ? getDTypeSize(mOType) : getDTypeSize(mDType));
  float weight_bytes = is_4bit_weight ? 0.5f : static_cast<float>(getDTypeSize(mWType));
  size_t output_bytes = getDTypeSize(mOType);
  size_t gemm_output_bytes = (mOType == nvinfer::DataType::kFP8)
                                 ? sizeof(TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
                                 : output_bytes;

  size_t hidden_size = mExpertHiddenSize;
  size_t inter_size = mExpertInterSize;  // Already divided by TP
  size_t num_experts_per_node = mNumExpertsPerNode;

  size_t fc1_out_size = inter_size;
  if (isGatedActivation(mActivationType)) {
    fc1_out_size = inter_size * 2;
  }

  // TODO Needs updated when gather/finalize fusion is integrated
  size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
  size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

  size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
  size_t output_size2 = hidden_size * output_bytes;

  size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
  size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

  // This may allocate a pointer when not required. That's fine it will be ignored at the cost of some memory
  size_t intermediate_size1 = fc1_out_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes
  size_t intermediate_size2 = hidden_size * num_expanded_tokens * gemm_output_bytes;   // Note gemm_output_bytes

  size_t intermediate_size = mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

  size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
  size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  if (false && !is_tma_ws_input)
    bias_1 = output_size1;
  size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
  size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

  size_t weights_size = mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

  // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
  bool is_int_w_quant = (mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize <= 0;
  bool is_int_groupwise_w_quant = (mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize > 0;
  bool is_fp8_act_quant = mDType == nvinfer::DataType::kFP8;
  bool is_fp8_w_quant = mWType == nvinfer::DataType::kFP8;
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  // bool is_fp4_act_quant = mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64;
  bool is_fp4_w_quant = mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64;
  bool is_wfp4a16_groupwise_quant = is_fp4_w_quant && !is_fp8_act_quant && mSM < 90;
  const int effective_group_size = is_wfp4a16_groupwise_quant && mGroupSize <= 0 ? 32 : mGroupSize;
  bool is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant;
  // bool is_wfp4afp8_quant = is_fp4_w_quant && is_fp8_act_quant;

  // Int sizes
  size_t quant_1_size = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  size_t quant_2_size = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;
  if (is_int_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes;
  } else if (is_int_groupwise_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / mGroupSize;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / mGroupSize;
  }

  // FP8 sizes
  quant_1_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1_size;
  quant_2_size = is_fp8_w_quant ? sizeof(float) : quant_2_size;
  size_t quant_3_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t quant_4_size = 0;  // Currently ignored by the GEMM
  if (is_int_groupwise_w_quant) {
    quant_3_size = quant_1_size;
    quant_4_size = quant_2_size;
  }

  if (is_wfp4a16_groupwise_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / effective_group_size;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / effective_group_size;
    quant_3_size = 0;
    quant_4_size = 0;
  }

  // FP4 sizes
  quant_1_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant ? sizeof(float) : quant_1_size;
  quant_2_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant ? getOffsetWeightSF(num_experts_per_node, inter_size, hidden_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                                               : quant_2_size;
  quant_3_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant ? num_experts_per_node * sizeof(float) : quant_3_size;
  quant_4_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant ? sizeof(float) : quant_4_size;
  size_t quant_5_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant
                            ? getOffsetWeightSF(num_experts_per_node, hidden_size, inter_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                            : 0;
  size_t quant_6_size = is_fp4_w_quant && !is_wfp4a16_groupwise_quant ? num_experts_per_node * sizeof(float) : 0;

  size_t tma_ws_input_workspace_size = 0;
  if (is_tma_ws_input) {
    tma_ws_input_workspace_size = TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, mScalingType) * (NUM_ROUTING_SAMPLES + 1);

    if (is_w4afp8_quant) {
      quant_3_size = 0;
      quant_4_size = 0;
    }
  }

  auto act_sf_rows = std::min(num_expanded_tokens, static_cast<size_t>(maxM * num_experts_per_node));
  // getOffsetActivationSF returns zero if scaling_type is NONE
  const size_t fc1_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  const size_t fc2_fp4_act_scale_size = getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, mScalingType) * sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  const size_t fp4_act_scale_flat_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  size_t w4a8_alpha_size = is_w4afp8_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
  size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

  // Routing info
  size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
  size_t map_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t unpermuted_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t permuted_size = num_expanded_tokens * sizeof(int);
  size_t token_topk_unpermuted_scales_size = num_expanded_tokens * sizeof(float);

  const int64_t num_tokens_per_block = computeNumTokensPerBlock(maxM, num_experts_per_node);
  const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(maxM, num_tokens_per_block);
  const size_t blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * sizeof(int);
  const size_t blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  const size_t blocked_row_to_unpermuted_row_size = num_experts_per_node * maxM * sizeof(int);

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                              \
  do {                                                    \
    auto aligned_size = alignSize(size, kCudaMemAlign);   \
    out_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                           \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(expert_first_token_offset);
  ADD_NAME(unpermuted_row_to_permuted_row, map_size);
  ADD_NAME(permuted_row_to_unpermuted_row, map_size);
  ADD_NAME(token_selected_experts, unpermuted_size);
  ADD_NAME(permuted_token_selected_experts, permuted_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(token_topk_unpermuted_scales);
  ADD(input);
  ADD(output);
  ADD(intermediate);
  ADD(weights);
  ADD(bias);
  ADD(quant_1);
  ADD(quant_2);
  ADD(quant_3);
  ADD(quant_4);
  ADD(quant_5);
  ADD(quant_6);
  ADD(tma_ws_input_workspace);
  ADD(w4a8_alpha);
  ADD(alpha_scale_ptr_array);
  ADD(fp4_act_scale_flat);
  ADD(gemm_workspace);

#undef ADD_NAME
#undef ADD

  return out_map;
}

void GemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR_BASE(type, name)                                                                                          \
  auto* name##_base = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                                  : nullptr)
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
  GET_WS_PTR_BASE(int*, unpermuted_row_to_permuted_row);
  GET_WS_PTR_BASE(int*, permuted_row_to_unpermuted_row);
  GET_WS_PTR_BASE(int*, token_selected_experts);
  GET_WS_PTR(int*, permuted_token_selected_experts);
  GET_WS_PTR(int*, blocked_expert_counts);
  GET_WS_PTR(int*, blocked_expert_counts_cumsum);
  GET_WS_PTR(int*, blocked_row_to_unpermuted_row);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

  {
    const int64_t num_expanded_tokens = num_tokens * mK;
    const int start_expert_id = mNumExpertsPerNode * mParallelismConfig.ep_rank;

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(
        token_selected_experts_base, num_tokens, mK, mNumExperts);
    sync_check_cuda_error(stream);

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
      int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
      int* unpermuted_row_to_permuted_row = unpermuted_row_to_permuted_row_base + i * num_expanded_tokens;
      int* permuted_row_to_unpermuted_row = permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;
      int* token_selected_experts = token_selected_experts_base + i * num_expanded_tokens;

      threeStepBuildExpertMapsSortFirstToken(token_selected_experts, permuted_token_selected_experts,
                                             permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, expert_first_token_offset,
                                             blocked_expert_counts, blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
                                             mNumExpertsPerNode, mK, start_expert_id, stream);
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepareQuantParams(int num_tokens, char* workspace_ptr_char, cudaStream_t) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)
  GET_WS_PTR(const void*, quant_1);
  GET_WS_PTR(const void*, quant_2);
  GET_WS_PTR(const void*, quant_3);
  GET_WS_PTR(const void*, quant_4);
  GET_WS_PTR(const void*, quant_5);
  GET_WS_PTR(const void*, quant_6);
  GET_WS_PTR(const float*, w4a8_alpha);
#undef GET_WS_PTR

  if ((mWType == nvinfer::DataType::kINT8 || mWType == nvinfer::DataType::kINT4) && mGroupSize < 0) {
    ORT_ENFORCE(quant_1 && quant_2);
    mQuantParams = QuantParams::Int(quant_1, quant_2);
  } else if (mWType == nvinfer::DataType::kINT4 || mWType == nvinfer::DataType::kINT8) {
    ORT_ENFORCE(quant_1 && quant_2);
    if (mDType == nvinfer::DataType::kFP8) {
      ORT_ENFORCE(w4a8_alpha);
      mQuantParams = QuantParams::GroupWise(
          mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4, w4a8_alpha, w4a8_alpha);
    } else {
      mQuantParams = QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4);
    }
  } else if (mWType == nvinfer::DataType::kFP8) {
    ORT_ENFORCE(quant_1 && quant_2 && quant_3);
    mQuantParams = QuantParams::FP8(static_cast<const float*>(quant_1), static_cast<const float*>(quant_2),
                                    static_cast<const float*>(quant_3), static_cast<const float*>(quant_4));
  } else if (mDType == nvinfer::DataType::kFP8 && (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64)) {
    ORT_ENFORCE(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP8MXFP4(static_cast<const float*>(quant_1),
                                         static_cast<const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF*>(quant_2),
                                         static_cast<const float*>(quant_3), static_cast<const float*>(quant_4),
                                         static_cast<const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF*>(quant_5),
                                         static_cast<const float*>(quant_6));
  } else if ((mDType == nvinfer::DataType::kFP4 || mDType == nvinfer::DataType::kINT64) && (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64)) {
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    ORT_ENFORCE(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP4(static_cast<const float*>(quant_1),
                                    static_cast<const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(quant_2),
                                    static_cast<const float*>(quant_3), static_cast<const float*>(quant_4),
                                    static_cast<const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(quant_5),
                                    static_cast<const float*>(quant_6));
  } else if ((mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64) && mSM < 90) {
    ORT_ENFORCE(quant_1 && quant_2);
    mQuantParams = QuantParams::GroupWise(mGroupSize > 0 ? mGroupSize : 32, quant_1, quant_2);
  } else if (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64) {
    // W4A16: FP4 weights with FP16/BF16 activations (no activation quantization)
    ORT_ENFORCE(quant_2 && quant_3 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP4(nullptr,
                                    static_cast<const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(quant_2),
                                    static_cast<const float*>(quant_3), nullptr,
                                    static_cast<const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(quant_5),
                                    static_cast<const float*>(quant_6));
  }
}

void GemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, const void* expert_weights, cudaStream_t stream) {
  if (mSM < 90) {
    return;
  }

  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR(int64_t*, expert_first_token_offset);
  int64_t* expert_first_token_offset_base = expert_first_token_offset;
  GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
  int* permuted_row_to_unpermuted_row_base = permuted_row_to_unpermuted_row;
  GET_WS_PTR(void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void*, weights);
  ORT_ENFORCE(mNeedWeights == (expert_weights == nullptr));
  const void* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void*, bias);
  GET_WS_PTR(float*, token_topk_unpermuted_scales);
  GET_WS_PTR(int8_t*, tma_ws_input_workspace);
  GET_WS_PTR(void*, gemm_workspace);
  GET_WS_PTR(float*, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);

#undef GET_WS_PTR

  size_t tma_ws_size = TmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode, mScalingType);

  TmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
  dummy_tma_ws_input.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                        workspaces.at("gemm_workspace").first, mScalingType);
  tma_ws_input_workspace += tma_ws_size;

  size_t num_expanded_tokens = num_tokens * mK;
  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    mTmaInputCache[i].configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                         workspaces.at("gemm_workspace").first, mScalingType);
    tma_ws_input_workspace += tma_ws_size;

    int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* permuted_row_to_unpermuted_row = permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;

    auto& gemm1_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_1 ? mTmaInputCache[i] : dummy_tma_ws_input;
    auto& gemm2_tma_ws_input = mGemmToProfile == GemmToProfile::GEMM_2 ? mTmaInputCache[i] : dummy_tma_ws_input;
    if (mSM >= 90) {
      /* GEMM1 */
      gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

      bool apply_bias = true;
      bool use_w4afp8 = (mDType == nvinfer::DataType::kFP8 && mWType == nvinfer::DataType::kINT4);
      bool use_wfp4a16 = (mDType != nvinfer::DataType::kFP4 && mDType != nvinfer::DataType::kINT64 && mDType != nvinfer::DataType::kFP8) && (mWType == nvinfer::DataType::kFP4 || mWType == nvinfer::DataType::kINT64);
      bool using_fused_finalize = !mInterface->use_deterministic_hopper_reduce_ && mSM == 90 && !use_w4afp8 && !use_wfp4a16;
      if (using_fused_finalize) {
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        gemm2_tma_ws_input.setFinalizeFusionParams(output, token_topk_unpermuted_scales,
                                                   expert_first_token_offset, permuted_row_to_unpermuted_row, apply_bias ? bias : nullptr,
                                                   mExpertHiddenSize, num_tokens);
      }

      auto fc1_output_size = isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;
      {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) = mInterface->computeStridesTmaWarpSpecializedDispatch(
            expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens, num_tokens * mK,
            fc1_output_size, mExpertHiddenSize, mExpertHiddenSize, mExpertInterSize, mNumExpertsPerNode, input,
            input, weights_sel, weights_sel, mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2,
            fp4_act_scale_flat, fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
            stream);
      }
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepare(
    int num_tokens, char* workspace_ptr_char, const void* expert_weights, cudaStream_t stream) {
  mAllTacticsSaved = mInterface->getTactics();
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, stream);
  prepareQuantParams(num_tokens, workspace_ptr_char, stream);
  prepareTmaWsInputs(num_tokens, workspace_ptr_char, expert_weights, stream);
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(), [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("MOE profiler workspace size: ", size));
  return size;
}

void GemmProfilerBackend::runProfiler(int original_num_tokens, const Config& tactic, char* workspace_ptr_char,
                                      const void* expert_weights, const cudaStream_t& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  // The workspace layout must match the one used to size the allocation (getWorkspaceSize) and to
  // populate routing/quant/TMA inputs (prepareRouting/prepareQuantParams/prepareTmaWsInputs), all of
  // which key the layout on `mSM >= 90`. Keying it here on the per-tactic `tactic.is_tma_warp_specialized`
  // instead diverges on SM90 whenever a non-TMA (Ampere-fallback) tactic is profiled — e.g. the INT4
  // mixed-input GEMM tiles. That divergence drops the tma_ws_input region and shifts every subsequent
  // sub-buffer offset, so runProfiler would read expert_first_token_offset (and the GEMM inputs) from the
  // wrong, random-filled locations, producing garbage per-expert token counts and an out-of-bounds read
  // in the grouped GEMM (surfacing as a sticky CUDA 700 at a later launch). Use the same key everywhere.
  auto workspaces = getProfilerWorkspaces(original_num_tokens, mSM >= 90);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                             \
  auto* name = (workspaces.at(#name).first                                                                \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset) \
                    : nullptr)
#define GET_WS_PTR(type, name)                                                                                        \
  auto* name = (workspaces.at(#name).first ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                                           : nullptr)

  GET_WS_PTR_OFFSET(const int64_t*, expert_first_token_offset, (mSampleIndex * (mNumExpertsPerNode + 1)));
  GET_WS_PTR_OFFSET(const int*, unpermuted_row_to_permuted_row, (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(const int*, permuted_row_to_unpermuted_row, (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(const int*, token_selected_experts, (mSampleIndex * expanded_num_tokens));

  GET_WS_PTR(const float*, token_topk_unpermuted_scales);
  const auto* token_topk_permuted_scales = token_topk_unpermuted_scales;

  GET_WS_PTR(const void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(const void*, weights);
  ORT_ENFORCE(mNeedWeights == (expert_weights == nullptr));
  const void* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(const void*, bias);

  GET_WS_PTR(const float**, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(void*, gemm_workspace);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  TmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
  if (tactic.is_tma_warp_specialized) {
    tma_ws_input_template = mTmaInputCache[mSampleIndex];
  }

  mInterface->is_profiler = true;
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    mInterface->gemm1(input,                                             //
                      output,                                            //
                      intermediate,                                      //
                      expert_first_token_offset,                         //
                      tma_ws_input_template,                             //
                      weights_sel,                                       //
                      bias,                                              //
                      expert_first_token_offset + num_experts_per_node,  //
                      mQuantParams.wo.fc1_weight_scales,                 //
                      mQuantParams.fp8.dequant_fc1,                      //
                      mQuantParams.fp8_mxfp4.fc2.act_global_scale ? mQuantParams.fp8_mxfp4.fc2.act_global_scale
                                                                  : mQuantParams.fp8.quant_fc2,  //
                      fp4_act_scale_flat,                                                        //
                      fp4_act_scale_flat,                                                        //
                      mQuantParams,                                                              //
                      original_num_tokens,                                                       //
                      expanded_num_tokens,                                                       //
                      mExpertHiddenSize,                                                         //
                      mExpertInterSize,                                                          //
                      num_experts_per_node,                                                      //
                      mActivationType,                                                           //
                      alpha_scale_ptr_array,                                                     //
                      true,                                                                      //
                      stream,                                                                    //
                      tactic,                                                                    //
                      {});                                                                       // activation params
  } else {
    ORT_ENFORCE(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(input,                                           //
                      intermediate,                                    //
                      output,                                          //
                      expert_first_token_offset,                       //
                      tma_ws_input_template,                           //
                      weights_sel,                                     //
                      bias,                                            //
                      mQuantParams.wo.fc2_weight_scales,               //
                      mQuantParams.fp8.dequant_fc2,                    //
                      fp4_act_scale_flat,                              //
                      mQuantParams,                                    //
                      token_topk_unpermuted_scales,                    //
                      token_topk_permuted_scales,                      //
                      unpermuted_row_to_permuted_row,                  //
                      permuted_row_to_unpermuted_row,                  //
                      token_selected_experts,                          //
                      expert_first_token_offset + mNumExpertsPerNode,  //
                      original_num_tokens,                             //
                      expanded_num_tokens,                             //
                      mExpertHiddenSize,                               //
                      mExpertInterSize,                                //
                      num_experts_per_node,                            //
                      mK,                                              //
                      alpha_scale_ptr_array,                           //
                      stream,                                          //
                      mParallelismConfig,                              //
                      tactic);                                         //
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
template class CutlassMoeFCRunner<half, __nv_fp4_e2m1>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1>;
#endif
// Native NVFP4 (W4A16-NVFP4) block-scaled prefill: FP4 e2m1 weights + FP4 e2m1 activations
// (activations dynamically quantized to NVFP4 inside expandInputRowsKernel). InputType/OutputType
// are the BF16/FP16 user tensor types. Native CUTLASS FP4xFP4 grouped GEMM requires SM120+.
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
#endif
#if defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
// W4A8 (WFP4AFP8): FP8 e4m3 activations + MXFP4 weights, BF16/FP16 input/output.
// InputType differs from T (the GEMM activation type) so the runner can accept BF16/FP16 user
// input and quantize it to FP8 inside expandInputRowsKernel. Native CUTLASS path requires SM100+.
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
#endif
#endif
#endif

#if defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
// W8A16-FP8: FP8 e4m3 weights with FP16/BF16 activations (native SM90)
template class CutlassMoeFCRunner<half, __nv_fp8_e4m3>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3>;
#endif
#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
