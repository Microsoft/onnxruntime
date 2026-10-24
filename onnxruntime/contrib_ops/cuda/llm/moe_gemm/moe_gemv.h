// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Batched GEMV fast path for symmetric int weight-only MoE at small expanded row
// counts (e.g. batch-1 decode with top_k experts). Each expanded row is a single
// token-expert pair; one thread-block handles one row and offsets the
// weight/scale/bias pointers by that row's expert. Reuses the device-side math
// (layout, dequantize, mma, epilogue) from the dense fpA_intB_gemv kernel.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

inline constexpr int64_t kMaxProfiledExpandedRows = 80;
inline constexpr int64_t kMaxProfiledExpandedRowsForSmallProblemDim = 4;
inline constexpr int64_t kMinProfiledProblemDim = 512;
// Lowered from 704 to 512 so block-wise decode shapes (e.g. Qwen top_k=8,
// inter_size=512) take the GEMV path. This also covers per-column INT4 shapes
// with inter_size in [512, 704); both bands are gated by ORT_DISABLE_MOE_GEMV.
inline constexpr int64_t kMinProfiledProblemDimForExpandedRowsAbove4 = 512;

// Returns true if the batched MoE GEMV fast path supports this problem shape.
// Requirements: FP16/BF16 activations, sm >= 80, small expanded_num_rows, supported
// INT weight type, supported group size, and n divisible by the kernel tile width.
bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size);

// Backward-compatible per-channel INT4 shape check.
bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k);

// Returns true if the FC2 GEMV with fused MoE finalize supports this problem shape. In addition to
// the plain GEMV requirements, the token->permuted-row mapping must be dense
// (expanded_num_rows == num_rows * experts_per_token) and n must tile evenly at the narrower
// column tile the fused kernel uses.
bool is_moe_gemv_fused_finalize_supported(int sm, int64_t num_rows, int64_t experts_per_token,
                                          int64_t expanded_num_rows, int64_t n, int64_t k,
                                          int weight_bits, int group_size);

// Launches symmetric INT MoE GEMV. group_size <= 0 means per-channel scales;
// group_size 32/64/128 means block-wise scales laid out as [num_experts, k_blocks, n].
// T is half or __nv_bfloat16. WeightType is cutlass::uint2b_t, cutlass::uint4b_t, or uint8_t.
template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric(
    const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int group_size, int sm, cudaStream_t stream);

// Launches the FC2 symmetric INT MoE GEMV with the MoE finalize reduction fused into the epilogue,
// writing the reduced [num_rows, n] output directly and making a separate finalizeMoeRouting launch
// unnecessary. Single-EP / single-TP only.
//   act:      [expanded_num_rows, k] permuted activations (FC1 output)
//   bias:     [num_experts, n] per-expert FC2 bias (T) or nullptr
//   out:      [num_rows, n] reduced MoE output
//   unpermuted_row_to_permuted_row: [experts_per_token, num_rows], indexed [k_idx * num_rows + token]
//   permuted_row_to_expert:         [expanded_num_rows] local expert id per permuted row (required)
//   final_scales:                   [num_rows, experts_per_token] routing weights, or nullptr for 1.0
template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_fused_finalize(
    const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
    const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_expert, const float* final_scales,
    int num_experts, int64_t num_rows, int64_t experts_per_token, int64_t n, int64_t k, int group_size, int sm,
    cudaStream_t stream);

// Launches symmetric INT MoE GEMV and fuses interleaved SwiGLU activation.
// weight/bias use raw FC1 output width n = 2 * inter_size. Scales are
// [num_experts, n] for group_size <= 0 and [num_experts, k_blocks, n] for
// block-wise group_size 32/64/128.
//
// `permuted_row_to_source_row` is optional. When it is nullptr, `act` is the already-expanded
// activation buffer of shape [expanded_num_rows, k] produced by expandInputRows. When it is
// non-null it must be `permuted_row_to_unpermuted_row` of shape [expanded_num_rows], and `act` is
// the *unexpanded* activation buffer of shape [num_rows, k]; the kernel then reads row
// `permuted_row_to_source_row[permuted_row] % num_rows`, which lets the caller skip the expansion
// kernel entirely.
template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_interleaved_swiglu(
    const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int group_size, int sm, cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int64_t num_rows,
    float* splitk_partials,
    cudaStream_t stream);

// Launches the int4 per-channel MoE GEMV.
//   act:      [expanded_num_rows, k]  permuted activations (row-major)
//   weight:   [num_experts, k, n] packed int4 in Sm80 ColumnMajorInterleave layout (uint8)
//   scales:   [num_experts, n] per-channel scales (T)
//   bias:     [num_experts, n] per-expert bias (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
//   expert_first_token_offset: [num_experts + 1] prefix offsets of permuted rows
//   permuted_row_to_expert: [expanded_num_rows] local expert id for each permuted row, or nullptr to scan offsets
// T is half or __nv_bfloat16.
template <typename T>
void launch_moe_gemv_int4_per_channel(
    const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int sm, cudaStream_t stream);

// Launches the int4 per-channel MoE GEMV and fuses interleaved SwiGLU activation.
//   weight/scales/bias use raw FC1 output width [num_experts, k, 2 * inter_size]
//   out is post-activation [expanded_num_rows, inter_size]
// Only interleaved SwiGLU layout (`swiglu_fusion == 1`) is supported.
template <typename T>
void launch_moe_gemv_int4_per_channel_interleaved_swiglu(
    const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int sm, cutlass_kernels::ActivationParams activation_params,
    cudaStream_t stream);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
