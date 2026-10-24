// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// MXFP4 (e2m1) batched GEMV fast path for symmetric weight-only MoE at small expanded
// row counts (e.g. batch-1 decode with top_k experts). Companion to the INT path declared
// in moe_gemv.h; both share the device-side machinery in moe_gemv_device.cuh.

#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

// Tiling/parallelization knob selected by the FP4 GEMV autotuner. Every config computes the same
// dot products with the same accumulation dtype, so the sweep only picks the fastest. CtaN is
// bit-exact (it only changes how many output columns a block owns). Threads is *not*: a block
// walks K in strides of StepK * Threads and the epilogue reduces across Threads/32 warps, so
// changing it changes the summation order and the low bits of the result can move.
enum class MoeGemvConfig {
  kDefault,
  kCtaN16,
  kThreads64,
};

// Cover Qwen-style top_k=8 MTP decode. An (N+1)-token verification for
// num_speculative_tokens=N expands to (N+1)*8 rows, up to 64 for N=7.
inline constexpr int64_t kMaxProfiledExpandedRowsFp4 = 64;
inline constexpr int64_t kMaxProfiledExpandedRowsNvfp4KPacked = 80;

// True when the opt-in interleaved MXFP4 GEMV path is enabled (env ORT_FP4_GEMV_INTERLEAVED=1).
// It combines three changes over the default path: (a) the INT4-style ColumnMajorInterleaved FP4
// weight layout (kInterleave=4, kStepK=32) for 4x fewer K-trips, (b) dtype-conditional accumulation
// (fp32 for bf16) to keep bf16 accuracy across the longer K-chains, and (c) a smaller CtaN to
// recover the occupancy the interleave + fp32-accum cost. Default OFF; when off the shipping
// single-pass ColumnMajor path is byte-for-byte unchanged. Both PrePack (weight layout) and the
// compute dispatch query this so the prepacked weights and the kernel always agree.
bool Fp4MoeGemvUseInterleaved();

// Checks the loaded kernel's compilation target, not just the current device's SM.
bool Fp4MoeGemvSupportsNativeScale();
// Identifies native-scale specializations for unfused, no-bias K-packed launches.
bool Fp4MoeGemvKernelUsesNativeScale(const void* kernel);

// Shape-derived default tiling for the non-interleaved ColumnMajor FP4 GEMV. Used whenever the
// runtime does not have a profiled result for the shape, which is the shipping default because
// ORT_FP4_GEMV_AUTOTUNE is off (it synchronizes the inference stream) and is skipped entirely
// during CUDA-graph capture. Only Threads is derived; CtaN stays at the default, so the result
// never changes which shapes is_moe_gemv_fp4_supported accepts. Note that Threads sets the K
// partition, so the choice moves the last bits of the output (see MoeGemvConfig above). Set
// ORT_FP4_GEMV_DEFAULT_TILING=0 to fall back to the fixed default tiling.
MoeGemvConfig Fp4MoeGemvDefaultConfig(int64_t expanded_num_rows, int64_t n, int64_t k,
                                      int multi_processor_count);

// FP4 GEMV shape support. MXFP4 uses the prepacked ColumnMajor or SM80 pair-interleaved layout;
// NVFP4 sets raw_n_packed and directly consumes the schema [E,K,N/2] layout. weights_row_major
// selects native K-packed NVFP4 with an 80-row bound instead of 64. Requires sm >= 80 and the
// profiled small-decode row/dim bounds. The opt-in interleaved layout is MXFP4-only.
// See launch_moe_gemv_fp4_symmetric.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size);
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config);
// `sm80_pair_interleaved` = the weights were pre-packed in the SM80 grouped-GEMM layout (the
// interleaved layout plus the [e0,e2,e4,e6,e1,e3,e5,e7] nibble pair-interleave) and the GEMV will
// un-permute it in-register. It implies the interleaved shape rules (MXFP4 group_size 32,
// n % 16 == 0, k % 64 == 0) regardless of ORT_FP4_GEMV_INTERLEAVED.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed = false,
                               bool weights_row_major = false);

// Layout-only (row-count independent) form of the rules above, for PrePack: true when a GEMV can
// decode an [n, k] MXFP4 problem straight out of the SM80 grouped-GEMM pair-interleaved buffer, so
// no separate ColToRow decode copy of the expert weights has to be allocated.
bool is_moe_gemv_fp4_sm80_layout_supported(int64_t n, int64_t k, int group_size);

// Table-free E2M1 decode in MoeGemvFp4DecodeKernel requires every block scale (e4m3 * global_scale,
// with E4M3 max 448) to stay below 4 so the rescaled block scale (block_scale * 2^126 in bf16,
// * 2^14 in fp16) stays finite. Projections whose per-expert global scales exceed this bound fall
// back to MoeGemvFp4RawKPackedKernel by default instead of taking the slower generic decode branch.
inline constexpr float kNvfp4E4M3Max = 448.0f;
inline constexpr float kNvfp4DecodeFastScaleMaxProduct = 3.98f;

__host__ __device__ inline bool IsNvfp4DecodeFastScale(float global_scale) {
#if defined(__CUDA_ARCH__)
  return isfinite(global_scale) && fabsf(global_scale) * kNvfp4E4M3Max < kNvfp4DecodeFastScaleMaxProduct;
#else
  return std::isfinite(global_scale) && std::fabs(global_scale) * kNvfp4E4M3Max < kNvfp4DecodeFastScaleMaxProduct;
#endif
}

// Returns true when the CUDA image selected on the current device (of architecture `sm`) for
// MoeGemvFp4DecodeKernel was compiled with virtual architecture >= SM90. Forward-compatible PTX
// builds below SM90 (e.g. CMAKE_CUDA_ARCHITECTURES=80) compile an empty decode kernel body and
// return false here so callers fall back to MoeGemvFp4RawKPackedKernel.
bool is_moe_gemv_fp4_decode_image_available(int sm, bool is_fp16);

// Launches the MXFP4 (e2m1) MoE GEMV in the non-interleaved ColumnMajor layout.
//   act:      [expanded_num_rows, k]  permuted activations (row-major), T = half/bf16
//   weight:   [num_experts, n, k/2]  e2m1 codes packed two per byte (even-K low nibble)
//             == LaunchQMoERepackFP4ColToRow output
//   scales:   [num_experts, k/group_size, n]  TypeA block scales already folded with the per-expert
//             global scale == LaunchQMoECombineFp4ScalesForGemv (MXFP4, group_size 32) or
//             LaunchQMoECombineNvfp4ScalesForGemv (NVFP4, group_size 16) output
//   raw_block_scales/raw_global_scales: NVFP4-only [E,n,k/16] E4M3 bytes and [E] fp32 globals
//   bias:     [num_experts, n] (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
// group_size is the FP4 block size (32 for MXFP4, 16 for NVFP4).
// When `sm80_pair_interleaved` is true, `weight` is instead the SM80 grouped-GEMM buffer produced
// by QMoE::PrePackRepackFP4Weights(gemv_interleaved=true, sm80_pair_interleave=true) and the
// kernel inverts the nibble pair-interleave while decoding, so prefill and decode share one copy.
// When `raw_n_packed` is true, `weight` is the NVFP4 schema initializer [E,K,N/2], with adjacent
// N values in each byte; no full-size GEMV weight repack is required.
template <typename T>
void launch_moe_gemv_fp4_symmetric(
    const T* act, const uint8_t* weight, const T* scales, const uint8_t* raw_block_scales,
    const float* raw_global_scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int group_size, int sm, MoeGemvConfig config, bool sm80_pair_interleaved,
    bool raw_n_packed, cudaStream_t stream, bool weights_row_major = false,
    bool decode_scale_eligible = false, bool* decode_kernel_launched = nullptr,
    bool programmatic_dependent_launch = false);

// `programmatic_dependent_launch` (raw row-major NVFP4 decode kernel only) launches the grid with the PDL attribute: it
// must directly follow the fused-SwiGLU launch below on the same stream and may only read that kernel's output after
// its own griddepcontrol.wait (see MoeGemvFp4DecodeKernel).

// Native K-packed NVFP4 FC2 with top-k reduction; false leaves unsupported shapes to the caller.
// The caller must use single-EP/single-TP routing. Each expert result is rounded to T before reduction.
template <typename T>
bool try_launch_moe_gemv_nvfp4_fused_finalize(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out, const int* unpermuted_row_to_permuted_row,
    const int* permuted_row_to_expert, const float* routing_scales, int num_experts,
    int num_rows, int experts_per_token, int n, int k, cudaStream_t stream);

// Launches the MXFP4 MoE GEMV and fuses interleaved SwiGLU activation.
//   weight/scales/bias use raw FC1 output width n = 2 * inter_size
//   out is post-activation [expanded_num_rows, inter_size]
template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    const T* act, const uint8_t* weight, const T* scales, const uint8_t* raw_block_scales,
    const float* raw_global_scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int group_size, int sm, cutlass_kernels::ActivationParams activation_params,
    MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream,
    bool weights_row_major = false, bool decode_scale_eligible = false, bool* decode_kernel_launched = nullptr,
    bool programmatic_dependent_launch = false);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
