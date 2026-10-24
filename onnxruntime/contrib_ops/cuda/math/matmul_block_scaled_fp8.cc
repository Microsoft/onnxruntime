// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"
#include "contrib_ops/cuda/bert/fused_tensor_ops.h"

#include <algorithm>
#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

namespace {
bool FusedFp8ActivationQdqDisabled() {
  static const bool disabled =
      ParseEnvironmentVariableWithDefault<bool>("ORT_DISABLE_FUSED_FP8_ACT_QDQ", false);
  return disabled;
}

size_t DequantScratchLimitBytes() {
  const int64_t limit = ParseEnvironmentVariableWithDefault<int64_t>("ORT_FP8_DEQUANT_SCRATCH_MIB", 256);
  ORT_ENFORCE(limit > 0, "ORT_FP8_DEQUANT_SCRATCH_MIB must be positive.");
  return SafeInt<size_t>(limit) * 1024 * 1024;
}
}  // namespace

#if !defined(DISABLE_FLOAT8_TYPES)
ONNX_OPERATOR_KERNEL_EX(
    MatMulBlockQuantizedFp8Weight,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<Float8E4M3FN>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float>())
        .TypeConstraint("TI", BuildKernelDefConstraints<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 5),
    MatMulBlockQuantizedFp8Weight);
#endif

MatMulBlockQuantizedFp8Weight::MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info)
    : CudaKernel(info),
      block_size_(info.GetAttrOrDefault<int64_t>("block_size", 128)),
      max_dequant_scratch_bytes_(DequantScratchLimitBytes()),
      enable_deep_gemm_(ParseEnvironmentVariableWithDefault<bool>("ORT_FP8_MATMUL_DEEPGEMM", false)),
      shared_expert_(info.GetAttrOrDefault<int64_t>("shared_expert", 0) == 1) {
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("shared_expert", 0) == 0 || shared_expert_,
              "shared_expert must be 0 or 1");
  ORT_ENFORCE(block_size_ > 0, "block_size must be positive.");
  static_cast<void>(info.GetAttrs<int64_t>("prefill_weight_offsets", prefill_weight_offsets_));
  ORT_ENFORCE(prefill_weight_offsets_.size() <= 3,
              "prefill_weight_offsets supports at most three original weight segments");
  if (info.GetAttrs<int64_t>("split_sizes", split_sizes_).IsOK()) {
    ORT_ENFORCE(!split_sizes_.empty() && split_sizes_.size() <= Fp8GemvOutputSplit::kMaxOutputs,
                "split_sizes requires one to eight widths.");
    int64_t total = 0;
    for (const auto width : split_sizes_) {
      ORT_ENFORCE(width > 0 && width <= INT64_MAX - total,
                  "split_sizes must contain positive widths with no sum overflow.");
      total += width;
    }
    ORT_ENFORCE(!shared_expert_ || (split_sizes_.empty() && prefill_weight_offsets_.empty()),
                "shared_expert does not support split_sizes or prefill_weight_offsets");
  }
}

template <typename T>
Status MatMulBlockQuantizedFp8Weight::ComputeImpl(OpKernelContext* context) const {
#if defined(DISABLE_FLOAT8_TYPES)
  ORT_UNUSED_PARAMETER(context);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires float8 support.");
#else
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  const Tensor* b_scale = context->Input<Tensor>(2);
  const Tensor* a_scale = context->Input<Tensor>(3);  // optional
  const Tensor* bias = context->Input<Tensor>(4);     // optional
  const Tensor* numerics = context->Input<Tensor>(5);
  ORT_RETURN_IF_NOT(numerics == nullptr ||
                        (numerics->IsDataType<int32_t>() && numerics->Shape().Size() == 1 &&
                         (numerics->Data<int32_t>()[0] == 0 || numerics->Data<int32_t>()[0] == 1)),
                    "prefill_numerics must be an int32 scalar containing 0 or 1");
  const bool dense_prefill = numerics != nullptr && numerics->Data<int32_t>()[0] == 1;
  ORT_RETURN_IF_NOT(!dense_prefill || a_scale == nullptr,
                    "dense prefill numerics requires weight-only FP8 projection");

  const auto& a_shape = a->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() >= 1, "A must have rank at least 1.");

  const int64_t a_rank = a_shape.NumDimensions();
  const int64_t k = a_shape[a_rank - 1];
  const auto& b_shape = b->Shape();
  ORT_ENFORCE(b_shape.NumDimensions() == 2 && b_shape[1] == k,
              "B must have shape [N, K] with K = ", k, ", got ", b_shape.ToString(), ".");
  const int64_t n = b_shape[0];
  const int64_t k_blocks = (k + block_size_ - 1) / block_size_;

  const auto& b_scale_shape = b_scale->Shape();
  ORT_ENFORCE(b_scale_shape.NumDimensions() == 2 && b_scale_shape[0] == n &&
                  b_scale_shape[1] == k_blocks,
              "b_scale must have shape [N, ceil(K/block_size)] = [", n, ", ", k_blocks, "], got ",
              b_scale_shape.ToString(), ".");
  if (a_scale != nullptr) {
    ORT_ENFORCE(a_scale->Shape().Size() == 1, "a_scale must be a scalar.");
  }
  if (bias != nullptr) {
    ORT_ENFORCE(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == n,
                "bias must have shape [N] = [", n, "], got ", bias->Shape().ToString(), ".");
  }

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_logical_shape({n, k});
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b_logical_shape, transa, transb));

  const int m_i = SafeInt<int>(helper.M());
  const int n_i = SafeInt<int>(helper.N());
  const int k_i = SafeInt<int>(helper.K());
  const Tensor* silu_gate = context->Input<Tensor>(9);
  const Tensor* gate_activation = context->Input<Tensor>(10);
  const Tensor* gate_weight = context->Input<Tensor>(11);
  const Tensor* residual = context->Input<Tensor>(12);
  if (shared_expert_) {
    ORT_RETURN_IF_NOT((std::is_same_v<T, BFloat16>) && a_scale == nullptr && bias == nullptr &&
                          silu_gate && gate_activation && gate_weight && residual && k_i > 0 && n_i > 0,
                      "shared_expert requires BF16, positive widths, all four fusion inputs, no bias or a_scale");
    ORT_RETURN_IF_NOT(silu_gate->Shape() == a_shape &&
                          gate_activation->Shape().NumDimensions() == a_shape.NumDimensions() &&
                          gate_activation->Shape()[a_rank - 1] > 0 &&
                          gate_weight->Shape().NumDimensions() == 2 &&
                          gate_weight->Shape()[0] == gate_activation->Shape()[a_rank - 1] &&
                          gate_weight->Shape()[1] == 1 && residual->Shape() == helper.OutputShape(),
                      "shared_expert input shapes are incompatible");
    for (int64_t axis = 0; axis + 1 < a_rank; ++axis) {
      ORT_RETURN_IF_NOT(gate_activation->Shape()[axis] == a_shape[axis],
                        "shared_expert leading dimensions must match");
    }
    for (int index = 6; index <= 8; ++index) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(index) == nullptr,
                        "shared_expert does not support original prefill weight inputs");
    }
  } else {
    ORT_RETURN_IF_NOT(!silu_gate && !gate_activation && !gate_weight && !residual,
                      "Shared expert inputs require shared_expert=1");
  }

  const bool use_gemv = !dense_prefill && m_i > 0 &&
                        m_i <= MatMulBlockScaledFp8GemvMaxM(k_i, SafeInt<int>(block_size_), GetDeviceProp()) &&
                        (k_i % 16 == 0) && (block_size_ % 16 == 0);
  const bool direct_split = !split_sizes_.empty() && use_gemv &&
                            MatMulBlockScaledFp8GemvSupportsSplitOutput(
                                m_i, k_i, SafeInt<int>(block_size_), GetDeviceProp());
  Fp8GemvOutputSplit split;
  void* y_data;
  IAllocatorUniquePtr<CudaT> y_scratch;
  const auto output_shape = helper.OutputShape();
  if (split_sizes_.empty()) {
    ORT_ENFORCE(context->OutputCount() == 1, "Without split_sizes exactly one output is required.");
    y_data = context->Output(0, output_shape)->MutableDataRaw();
  } else {
    ORT_ENFORCE(context->OutputCount() == static_cast<int>(split_sizes_.size()),
                "Output count must match split_sizes.");
    int64_t total = 0;
    for (const auto width : split_sizes_) {
      total += width;
    }
    ORT_ENFORCE(total == n, "split_sizes must sum to the weight's N dimension.");
    split.count = static_cast<int>(split_sizes_.size());
    for (int i = 0; i < split.count; ++i) {
      TensorShapeVector shape(output_shape.GetDims().begin(), output_shape.GetDims().end());
      shape.back() = split_sizes_[i];
      split.outputs[i] = context->Output(i, TensorShape(shape))->MutableDataRaw();
      split.widths[i] = SafeInt<int>(split_sizes_[i]);
    }
    if (direct_split) {
      y_data = split.outputs[0];
    } else {
      y_scratch = GetScratchBuffer<CudaT>(SafeInt<size_t>(output_shape.Size()), GetComputeStream(context));
      y_data = y_scratch.get();
    }
  }
  if (output_shape.Size() == 0) {
    return Status::OK();
  }
  IAllocatorUniquePtr<CudaT> shared_activation;
  IAllocatorUniquePtr<float> shared_gates;
  Fp8SharedExpertEpilogue shared_epilogue;
  bool shared_epilogue_done = false;
  if (shared_expert_) {
    shared_activation = GetScratchBuffer<CudaT>(SafeInt<size_t>(m_i) * k_i, GetComputeStream(context));
    shared_gates = GetScratchBuffer<float>(m_i, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchPrepareSharedExpert(
        Stream(context), reinterpret_cast<const uint16_t*>(a->DataRaw()),
        reinterpret_cast<const uint16_t*>(silu_gate->DataRaw()),
        reinterpret_cast<const uint16_t*>(gate_activation->DataRaw()),
        reinterpret_cast<const uint16_t*>(gate_weight->DataRaw()),
        reinterpret_cast<uint16_t*>(shared_activation.get()), shared_gates.get(), m_i, k_i,
        SafeInt<int>(gate_activation->Shape()[a_rank - 1])));
    shared_epilogue = {shared_gates.get(), residual->DataRaw()};
  }
  const auto finish_output = [&]() -> Status {
    if (shared_expert_ && !shared_epilogue_done) {
      ORT_RETURN_IF_ERROR(LaunchFp8SharedExpertEpilogue(y_data, shared_epilogue, m_i, n_i, Stream(context)));
    }
    if (split.count != 0 && !direct_split) {
      size_t offset = 0;
      for (int i = 0; i < split.count; ++i) {
        const size_t width_bytes = static_cast<size_t>(split.widths[i]) * sizeof(CudaT);
        CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(
            split.outputs[i], width_bytes, static_cast<const uint8_t*>(y_data) + offset,
            static_cast<size_t>(n_i) * sizeof(CudaT), width_bytes, m_i,
            cudaMemcpyDeviceToDevice, Stream(context)));
        offset += width_bytes;
      }
    }
    return Status::OK();
  };

  if (k_i == 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, SafeInt<size_t>(output_shape.Size()) * sizeof(CudaT), Stream(context)));
    if (bias != nullptr) {
      ORT_RETURN_IF_ERROR(LaunchAddBiasBlockScaledFp8(
          y_data,
          bias->DataRaw(),
          m_i,
          n_i,
          std::is_same<T, BFloat16>::value,
          Stream(context)));
    }
    return finish_output();
  }

  // Optional W8A8 activation path: statically quantize A to FP8 E4M3 and dequantize back so the
  // GEMM sees the same activation rounding as native W8A8 execution. When a_scale is absent the
  // activation is kept at full FP16/BF16 precision (weight-only W8A16).
  //
  // The decode GEMV absorbs this round trip in-register, so the standalone kernel (and its [M, K]
  // scratch buffer) is only materialized for the cuBLAS path below.
  const bool fuse_act_qdq = use_gemv && !FusedFp8ActivationQdqDisabled();

#if defined(USE_DEEP_GEMM)
  // Native FP8 changes intermediate rounding, so it is opt-in and requires the model's
  // activation scale. Keep the tuned small-M path and unsupported layouts on the fallback.
  if (!dense_prefill && enable_deep_gemm_ && !use_gemv && a_scale != nullptr && block_size_ == 128 &&
      GetDeviceProp().major == 9 && GetDeviceProp().minor == 0 &&
      k_i % 128 == 0 && n_i % 64 == 0 && n_i >= 2048 &&
      SafeInt<size_t>(n_i) * k_i >= 8 * 1024 * 1024 &&
      m_i <= 128) {
    const size_t rows_per_tile = max_dequant_scratch_bytes_ / (SafeInt<size_t>(m_i) * sizeof(float));
    const int tile_n = static_cast<int>(std::min<size_t>(n_i, rows_per_tile / 64 * 64));
    if (tile_n > 0) {
      LOGS_DEFAULT(VERBOSE) << "MatMulBlockQuantizedFp8Weight: using SM90 DeepGEMM";
      const int aligned_m = (m_i + 3) / 4 * 4;
      auto* stream = GetComputeStream(context);
      auto a_quant = GetScratchBuffer<uint8_t>(SafeInt<size_t>(m_i) * k_i, stream);
      auto a_scales = GetScratchBuffer<float>(SafeInt<size_t>(aligned_m) * (k_i / 128), stream);
      auto packed_b_scales = GetScratchBuffer<float>(SafeInt<size_t>(tile_n) * (k_i / 128), stream);
      auto accum = GetScratchBuffer<float>(SafeInt<size_t>(m_i) * tile_n, stream);
      ORT_RETURN_IF_ERROR(LaunchPrepareMatMulFp8DeepGemm(
          a_quant.get(), a_scales.get(), a->DataRaw(), a_scale->Data<float>(),
          m_i, k_i, aligned_m, std::is_same<T, BFloat16>::value, Stream(context)));
      for (int64_t offset = 0; offset < n_i; offset += tile_n) {
        const int rows = static_cast<int>(std::min<int64_t>(tile_n, n_i - offset));
        const size_t weight_offset = SafeInt<size_t>(offset) * k_i;
        const size_t scale_offset = SafeInt<size_t>(offset) * (k_i / 128);
        ORT_RETURN_IF_ERROR(LaunchMatMulFp8DeepGemm(
            static_cast<T*>(y_data) + offset, a_quant.get(), a_scales.get(),
            static_cast<const uint8_t*>(b->DataRaw()) + weight_offset,
            b_scale->Data<float>() + scale_offset,
            bias != nullptr ? bias->Data<T>() + offset : nullptr,
            packed_b_scales.get(), accum.get(), m_i, rows, k_i, aligned_m, n_i,
            std::is_same<T, BFloat16>::value, GetDeviceProp().multiProcessorCount, Stream(context)));
      }
      return finish_output();
    }
  }
#endif

  const void* a_ptr = shared_expert_ ? shared_activation.get() : a->DataRaw();
  IAllocatorUniquePtr<CudaT> a_dequant;
  if (a_scale != nullptr && !fuse_act_qdq) {
    a_dequant = GetScratchBuffer<CudaT>(SafeInt<size_t>(m_i) * SafeInt<size_t>(k_i),
                                        GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchQuantizeDequantizeActivationFp8(
        a_dequant.get(),
        a->DataRaw(),
        a_scale->Data<float>(),
        m_i,
        k_i,
        std::is_same<T, BFloat16>::value,
        Stream(context)));
    a_ptr = a_dequant.get();
  }

  // Decode fast path: for small M (autoregressive generation) this is a memory-bound GEMV.
  // A fused warp-per-column kernel reads the FP8 weight directly, avoiding both the [N, K]
  // dequant scratch buffer and the cuBLAS GEMM (which is underutilized at M == 1).
  if (use_gemv) {
    if (shared_expert_ && m_i <= 2 && GetDeviceProp().major == 9 &&
        MatMulBlockScaledFp8GemvSupportsSplitOutput(m_i, k_i, SafeInt<int>(block_size_), GetDeviceProp())) {
      ORT_RETURN_IF_ERROR(LaunchMatMulFp8SharedExpert(
          y_data, a_ptr, b->DataRaw(), b_scale->Data<float>(), m_i, n_i, k_i,
          SafeInt<int>(block_size_), shared_epilogue, GetDeviceProp(), Stream(context)));
      shared_epilogue_done = true;
      return finish_output();
    }
    ORT_RETURN_IF_ERROR(LaunchMatMulBlockScaledFp8Gemv(
        y_data,
        a_ptr,
        b->DataRaw(),
        b_scale->Data<float>(),
        bias != nullptr ? bias->DataRaw() : nullptr,
        (a_scale != nullptr && fuse_act_qdq) ? a_scale->Data<float>() : nullptr,
        m_i,
        n_i,
        k_i,
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        GetDeviceProp(),
        Stream(context), direct_split ? split : Fp8GemvOutputSplit{}));
    return finish_output();
  }

  // Dequantize the FP8 weight into a scratch buffer of the activation type, then GEMM. The scratch
  // is tiled over N because a full [N, K] buffer is 2.37 GiB for a 248320 x 5120 LM head.
  const int64_t rows_per_scratch =
      static_cast<int64_t>(max_dequant_scratch_bytes_ / (static_cast<size_t>(k) * sizeof(CudaT)));
  // Dense MatMul consumes a contiguous KN weight, not a transposed NK operand or N tiles.
  const int64_t tile_rows = dense_prefill ? n : std::clamp<int64_t>(rows_per_scratch, 1, n);

  IAllocatorUniquePtr<CudaT> b_dequant =
      GetScratchBuffer<CudaT>(SafeInt<size_t>(tile_rows) * SafeInt<size_t>(k), GetComputeStream(context));

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);
  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  const float* b_scale_data = b_scale->Data<float>();

  for (int64_t n_offset = 0; n_offset < n; n_offset += tile_rows) {
    const int rows = SafeInt<int>(std::min<int64_t>(tile_rows, n - n_offset));
    const size_t row_offset = static_cast<size_t>(n_offset);
    ORT_RETURN_IF_ERROR(LaunchDequantizeBlockScaledFp8(
        b_dequant.get(),
        b_data + row_offset * static_cast<size_t>(k),
        b_scale_data + row_offset * static_cast<size_t>(k_blocks),
        rows,
        SafeInt<int>(k),
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        Stream(context), dense_prefill));

    // The row-major [M, N] output is column-major [N, M] to cuBLAS, so an N tile is a row offset.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(context),
        dense_prefill ? CUBLAS_OP_N : CUBLAS_OP_T,
        CUBLAS_OP_N,  // transA
        rows,
        m_i,
        k_i,
        &alpha,
        b_dequant.get(),
        dense_prefill ? n_i : helper.Ldb(transb),
        reinterpret_cast<const CudaT*>(a_ptr),
        helper.Lda(transa),
        &zero,
        static_cast<CudaT*>(y_data) + n_offset,
        helper.Ldc(),
        GetDeviceProp(),
        UseTF32()));
  }

  if (dense_prefill) {
    for (size_t segment = 0; segment < prefill_weight_offsets_.size(); ++segment) {
      const auto* original = context->Input<Tensor>(6 + static_cast<int>(segment));
      ORT_RETURN_IF_NOT(original != nullptr && original->IsDataType<T>() &&
                            original->Shape().NumDimensions() == 2 &&
                            original->Shape()[0] == k,
                        "Original prefill weight must match activation type and shape [K, segmentN]");
      const int64_t width = original->Shape()[1];
      const int64_t offset = prefill_weight_offsets_[segment];
      ORT_RETURN_IF_NOT(offset >= 0 && width > 0 && offset <= n - width,
                        "Original prefill weight segment exceeds output width");
      auto original_output = GetScratchBuffer<CudaT>(
          SafeInt<size_t>(m_i) * SafeInt<size_t>(width), GetComputeStream(context));
      const int width_i = SafeInt<int>(width);
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          GetCublasHandle(context), CUBLAS_OP_N, CUBLAS_OP_N,
          width_i, m_i, k_i, &alpha,
          reinterpret_cast<const CudaT*>(original->Data<T>()), width_i,
          reinterpret_cast<const CudaT*>(a_ptr), helper.Lda(transa),
          &zero, original_output.get(), width_i, GetDeviceProp(), UseTF32()));
      CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(
          static_cast<CudaT*>(y_data) + offset, static_cast<size_t>(n) * sizeof(CudaT),
          original_output.get(), static_cast<size_t>(width) * sizeof(CudaT),
          static_cast<size_t>(width) * sizeof(CudaT), static_cast<size_t>(m_i),
          cudaMemcpyDeviceToDevice, Stream(context)));
    }
  }

  if (bias != nullptr) {
    ORT_RETURN_IF_ERROR(LaunchAddBiasBlockScaledFp8(
        y_data,
        bias->DataRaw(),
        m_i,
        n_i,
        std::is_same<T, BFloat16>::value,
        Stream(context)));
  }

  return finish_output();
#endif
}

Status MatMulBlockQuantizedFp8Weight::ComputeInternal(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  if (a->IsDataType<MLFloat16>()) {
    return ComputeImpl<MLFloat16>(context);
  }
  if (a->IsDataType<BFloat16>()) {
    return ComputeImpl<BFloat16>(context);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "MatMulBlockQuantizedFp8Weight only supports FP16 or BF16 activations.");
}

}  // namespace onnxruntime::contrib::cuda
