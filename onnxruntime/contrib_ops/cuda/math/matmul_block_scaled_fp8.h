// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

// Weight-only block-scaled FP8 (E4M3) matrix multiplication.
//
// B has shape [N, K] with one FP32 scale per row and block_size consecutive K values.
// By default, weights are dequantized to the activation type (FP16/BF16). When a_scale
// is present, A is quantized to FP8 E4M3 and dequantized back before multiplication;
// without a_scale, A retains its full FP16/BF16 precision.
//
// ORT_FP8_MATMUL_DEEPGEMM enables native FP8 multiplication for supported SM90 shapes
// with a_scale present and block_size=128. Scales are applied to FP32 block partial
// sums, so intermediate rounding differs from the default path. The result is
// converted to the activation type before adding bias. Other cases keep the default path.
class MatMulBlockQuantizedFp8Weight final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  int64_t block_size_;
  std::vector<int64_t> split_sizes_;
  size_t max_dequant_scratch_bytes_;
  bool enable_deep_gemm_;
  bool shared_expert_;
  std::vector<int64_t> prefill_weight_offsets_;
};

struct Fp8GemvOutputSplit {
  static constexpr int kMaxOutputs = 8;
  void* outputs[kMaxOutputs]{};
  int widths[kMaxOutputs]{};
  int count = 0;
};

struct Fp8SharedExpertEpilogue {
  const float* row_gates = nullptr;
  const void* residual = nullptr;
};

Status LaunchMatMulFp8SharedExpert(void* y, const void* a, const void* weight,
                                   const float* scales, int m, int n, int k, int block_size,
                                   const Fp8SharedExpertEpilogue& epilogue,
                                   const cudaDeviceProp& device_prop, cudaStream_t stream);
Status LaunchFp8SharedExpertEpilogue(void* y, const Fp8SharedExpertEpilogue& epilogue,
                                     int m, int n, cudaStream_t stream);

bool MatMulBlockScaledFp8GemvSupportsSplitOutput(int m, int k, int block_size,
                                                 const cudaDeviceProp& device_prop);

#if defined(USE_DEEP_GEMM)
Status LaunchPrepareMatMulFp8DeepGemm(void* a_quant, float* a_scales, const void* a,
                                      const float* a_scale, int m, int k, int aligned_m,
                                      bool is_bf16, cudaStream_t stream);

Status LaunchMatMulFp8DeepGemm(void* y, const void* a_quant, const float* a_scales,
                               const void* b, const float* b_scales, const void* bias,
                               float* packed_b_scales, float* accum, int m, int n, int k,
                               int aligned_m, int output_stride, bool is_bf16,
                               int sm_count, cudaStream_t stream);
#endif

// Dequantizes FP8 (E4M3) weights with per-block FP32 scales into FP16/BF16. b_fp8 is [N, K]
// row-major FP8 E4M3, weight_scale is [N, ceil(K/block_size)] fp32. Output b_dequant is [N, K]
// in the activation type (is_bf16 selects BF16 vs FP16).
Status LaunchDequantizeBlockScaledFp8(void* b_dequant,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream,
                                      bool transpose = false);

// Adds a per-column bias of shape [N] to a [M, N] row-major output in place.
Status LaunchAddBiasBlockScaledFp8(void* y,
                                   const void* bias,
                                   int m,
                                   int n,
                                   bool is_bf16,
                                   cudaStream_t stream);

// Statically quantizes a [M, K] FP16/BF16 activation to FP8 E4M3 using a single per-tensor fp32
// scale and dequantizes it back to the activation type in place-compatible scratch a_out
// (a_out = fp8_e4m3(a_in / a_scale) * a_scale). Realizes W8A8 activation numerics. a_scale is a
// device fp32 scalar. Runs on any architecture with FP8 conversion intrinsics (CUDA >= 11.8).
Status LaunchQuantizeDequantizeActivationFp8(void* a_out,
                                             const void* a_in,
                                             const float* a_scale,
                                             int m,
                                             int k,
                                             bool is_bf16,
                                             cudaStream_t stream);

// Fused FP8 weight-only GEMV fast path for the decode phase (small M). Reads the FP8 weight
// directly (no [N, K] dequant buffer). a is [M, K] activation (FP16/BF16), b_fp8 is [N, K]
// FP8 E4M3, weight_scale is [N, ceil(K/block_size)] fp32, bias is an optional [N] vector (may be
// null). Output y is [M, N] in the activation type. Requires k % 16 == 0 and block_size % 16 == 0.
// A nonempty split descriptor routes tensor-core stores into consecutive column partitions;
// callers must check MatMulBlockScaledFp8GemvSupportsSplitOutput before using that mode.
// device_prop selects the tensor-core (mma.m16n8k16) variant, which needs SM80+.
// act_scale is an optional device fp32 scalar; when non-null the kernel applies the W8A8
// activation quantize/dequantize inline, bit-identically to LaunchQuantizeDequantizeActivationFp8,
// so no scratch buffer or extra launch is needed.
// Runs on any architecture with FP8 conversion intrinsics (CUDA >= 11.8).
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
                                      const Fp8GemvOutputSplit& split = {});

// Largest M selected for GEMV dispatch. The tensor-core default is conservative because the
// crossover with dequantize + cuBLAS depends on the matrix shape; ORT_FP8_GEMV_MAX_M can raise the
// limit through 64 for tuned workloads. The direct launcher accepts tensor-core cases through 64.
int MatMulBlockScaledFp8GemvMaxM(int k, int block_size, const cudaDeviceProp& device_prop);

}  // namespace onnxruntime::contrib::cuda
