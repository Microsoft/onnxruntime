// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

/**
 * @brief QMoE is the templated CPU implementation of the Quantized Mixture of Experts operator.
 *
 * This kernel supports both float and MLFloat16 data types for activations, scales, and outputs.
 * It parallelizes expert computation using the ONNX Runtime thread pool and minimizes memory
 * usage through on-the-fly block dequantization of weights.
 *
 * @tparam T The data type for the kernel (float or MLFloat16).
 */
template <typename T>
class QMoECPU final : public OpKernel, public MoEBaseCPU {
 public:
  explicit QMoECPU(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  struct ComputeInputs {
    const Tensor* input;
    const Tensor* router_probs;
    const Tensor* fc1_experts_weights;
    const Tensor* fc1_scales;
    const Tensor* fc1_experts_bias;
    const Tensor* fc2_experts_weights;
    const Tensor* fc2_scales;
    const Tensor* fc2_experts_bias;
    const Tensor* fc3_experts_weights;
    const Tensor* fc3_scales;
    const Tensor* fc3_experts_bias;
    const Tensor* fc1_zero_points;
    const Tensor* fc2_zero_points;
    const Tensor* fc3_zero_points;
    const Tensor* router_weights;
  };

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  Status ComputeCommon(OpKernelContext* context, const ComputeInputs& inputs, const MoEParameters& moe_params) const;

  void ApplyActivationVectorized(float* data, int64_t size) const;

  // Expert weights pre-packed for the MLAS QNBit GEMM kernels (the MatMulNBits kernels). Used for
  // block-wise 4/8-bit experts with optional constant zero points; each expert's [rows, cols]
  // matrix is packed independently and stored back to back.
  struct QNBitPackedExperts {
    IAllocatorUniquePtr<void> packed;  // num_experts * packed_size_per_expert bytes
    size_t packed_size_per_expert{0};
    IAllocatorUniquePtr<float> scales_fp32;  // fp32 copy of the [E, rows, cols/block_size] scales (T == MLFloat16 only)
    bool scales_packed{false};               // scales are folded into `packed`; pass no QuantBScale at compute time
    bool has_zero_point{false};              // packed with the constant zero points; pass the expert's slice at compute time
  };
  // Result of QNBitGemmEligible. `ineligible_reason` is set only when a block-wise node misses the
  // path for a reason worth surfacing (row-wise scales and a disabled path leave it null).
  struct QNBitEligibility {
    const Tensor* scales{nullptr};
    const Tensor* zero_points{nullptr};  // null when the node has no zero points input
    size_t packed_size_per_expert{0};
    const char* ineligible_reason{nullptr};
  };
  bool QNBitGemmEligible(int input_idx, int64_t num_experts, int64_t rows, int64_t cols,
                         QNBitEligibility& out) const;
  Status InitQNBitPacked(QNBitPackedExperts& packed, const QNBitEligibility& eligibility, int64_t num_experts,
                         int64_t rows, int64_t cols, AllocatorPtr alloc);
  Status PrePackQNBitExperts(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                             /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights);

  int64_t expert_weight_bits_;
  int64_t fc1_expert_weight_bits_;
  int64_t fc2_expert_weight_bits_;
  int64_t fc3_expert_weight_bits_;
  int64_t block_size_;
  bool weights_prepacked_{false};
  bool use_mlas_q4_gemm_{false};
  bool use_mlas_q4_gemm_overridden_{false};

  IAllocatorUniquePtr<void> packed_fc1_;
  IAllocatorUniquePtr<void> packed_fc2_;
  IAllocatorUniquePtr<void> packed_fc1_lut_cache_;
  IAllocatorUniquePtr<void> packed_fc2_lut_cache_;

  TensorShape fc1_shape_;
  TensorShape fc2_shape_;

  IAllocatorUniquePtr<void> packed_fc1_mlas_cache_;
  IAllocatorUniquePtr<void> packed_fc2_mlas_cache_;

  int64_t accuracy_level_{0};
  bool use_qnbit_gemm_{true};
  bool qnbit_fallback_logged_{false};
  MLAS_QNBIT_GEMM_COMPUTE_TYPE qnbit_compute_type_{SQNBIT_CompFp32};
  QNBitPackedExperts qnbit_fc1_;
  QNBitPackedExperts qnbit_fc2_;
};

}  // namespace contrib
}  // namespace onnxruntime
