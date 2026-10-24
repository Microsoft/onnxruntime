// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"
#include "contrib_ops/cuda/bert/group_query_attention_eligibility.h"
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#endif
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "contrib_ops/cuda/bert/unfused_attention.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
// Map string attribute to quantization type enum
KVQuantizationType StringToKVQuantizationType(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  if (s == "NONE") {
    return KVQuantizationType::NONE;
  }
  if (s == "PER_TENSOR") {
    return KVQuantizationType::PER_TENSOR;
  }

  if (s == "PER_CHANNEL") {
    return KVQuantizationType::PER_CHANNEL;
  }
  return KVQuantizationType::NONE;
}
}  // namespace

#define REGISTER_KERNEL_TYPED(T, U)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      GroupQueryAttention,                                                            \
      kMSDomain,                                                                      \
      1,                                                                              \
      T##_##U,                                                                        \
      kCudaExecutionProvider,                                                         \
      (*KernelDefBuilder::Create())                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                      \
          .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<U>())                \
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()})              \
          .MayInplace(3, 1)                        /* past_key and present_key */     \
          .MayInplace(4, 2)                        /* past_value and present_value */ \
          .InputMemoryType(OrtMemTypeCPUInput, 6), /* total_sequence_length */        \
      GroupQueryAttention<T, U>);

REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, int8_t)
REGISTER_KERNEL_TYPED(BFloat16, int8_t)
#ifdef USE_FP8_KV_CACHE
REGISTER_KERNEL_TYPED(MLFloat16, Float8E4M3FN)
REGISTER_KERNEL_TYPED(BFloat16, Float8E4M3FN)
#endif
#ifdef USE_INT4_KV_CACHE
REGISTER_KERNEL_TYPED(MLFloat16, uint8_t)
REGISTER_KERNEL_TYPED(BFloat16, uint8_t)
#endif

constexpr const char* kDisableFlashDecode = "ORT_DISABLE_FLASH_DECODE";
constexpr int kHeadSinkInputIndex = 11;

// Group Query Attention (GQA) Operator
//
// This operator implements Group Query Attention, a variation of Multi-Head Attention (MHA)
// where the number of key/value heads is smaller than the number of query heads.
// It supports:
// - Rotary Positional Embeddings (RoPE)
// - KV Cache (past/present key/value)
// - Quantized KV Cache (Int8/Int4) via GroupQueryAttentionData
// - Flash Attention and Memory Efficient Attention backends
//
template <typename T, typename U>
GroupQueryAttention<T, U>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  const int64_t causal = info.GetAttrOrDefault<int64_t>("causal", 1);
  ORT_ENFORCE(causal == 0 || causal == 1, "causal must be 0 or 1.");
  is_unidirectional_ = causal == 1;
  const int64_t local_window_size_attr = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  // Validate before narrowing to int so an out-of-range attribute cannot wrap to a valid-looking
  // small window (e.g. 2^32 + 128) and silently run a different window than the model specifies.
  ORT_ENFORCE(local_window_size_attr == -1 || (local_window_size_attr > 0 && local_window_size_attr <= std::numeric_limits<int>::max()),
              "local_window_size must be -1 or greater than 0 (and not exceed INT_MAX).");
  local_window_size_ = static_cast<int>(local_window_size_attr);
  ORT_ENFORCE(is_unidirectional_ || local_window_size_ == -1,
              "GroupQueryAttention (CUDA): local_window_size must be -1 when causal is 0.");
  sliding_window_cache_ = info.GetAttrOrDefault<int64_t>("sliding_window_cache", 0) == 1;
  ORT_ENFORCE(!sliding_window_cache_ || local_window_size_ > 0,
              "GroupQueryAttention (CUDA): sliding_window_cache=1 requires local_window_size > 0.");
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;
  const float qk_norm_epsilon = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  ORT_ENFORCE(std::isfinite(qk_norm_epsilon) && qk_norm_epsilon > 0.0f,
              "GroupQueryAttention (CUDA): qk_norm_epsilon must be finite and positive.");
  q_norm_epsilon_ = info.GetAttrOrDefault<float>("q_norm_epsilon", qk_norm_epsilon);
  k_norm_epsilon_ = info.GetAttrOrDefault<float>("k_norm_epsilon", qk_norm_epsilon);
  ORT_ENFORCE(std::isfinite(q_norm_epsilon_) && q_norm_epsilon_ > 0.0f,
              "GroupQueryAttention (CUDA): q_norm_epsilon must be finite and positive.");
  ORT_ENFORCE(std::isfinite(k_norm_epsilon_) && k_norm_epsilon_ > 0.0f,
              "GroupQueryAttention (CUDA): k_norm_epsilon must be finite and positive.");

  k_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
  v_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));
  kv_cache_bit_width_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_cache_bit_width", 0));

  constexpr bool kIsFp16OrBf16 = std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>;
  // XQA defaults on for fp16/bf16; ORT_ENABLE_XQA=0 disables it explicitly.
  enable_xqa_ = kIsFp16OrBf16 && (ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_XQA", 1) != 0);

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  // Memory efficient attention supports float and float16. BFloat16 support added for SM80+.
  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  // cuDNN SDPA (cudnn_frontend) supports FP16 and BF16 and is auto-preferred on SM>=90.
  enable_cudnn_flash_attention_ = kIsFp16OrBf16 && kernel_options_->UseCudnnFlashAttention();
  auto_enable_cudnn_flash_attention_ = kIsFp16OrBf16 && kernel_options_->AllowCudnnFlashAttentionAuto();

  if (!disable_flash_attention_) {
    zeros_ = this->GetScratchBuffer<int>(kZerosCount, nullptr);
    CUDA_CALL_THROW(cudaMemset(zeros_.get(), 0, kZerosCount * sizeof(int)));
  }

  disable_flash_decode_ = ParseEnvironmentVariableWithDefault<bool>(kDisableFlashDecode, false);
}

template <typename T, typename U>
Status GroupQueryAttention<T, U>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                          bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  // Keep is_packed=false so the original fp16/bf16 head_sink remains available to the Flash/fallback
  // paths (which are used when XQA is disabled or ineligible). We only cache an extra FP32 copy for XQA.
  is_packed = false;

  if (input_idx != kHeadSinkInputIndex) {
    return Status::OK();
  }

  // XQA consumes the attention sink as FP32. When head_sink is a constant initializer, convert it once
  // here into a cached device buffer (xqa_head_sink_) to avoid a per-launch conversion. Dynamic /
  // non-initializer head_sink inputs are not prepacked and fall back to the per-launch scratch path.
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    const auto& shape = tensor.Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 1,
                      "head_sink must be a 1D tensor, got ", shape.NumDimensions(), " dimensions");
    ORT_RETURN_IF_NOT(shape[0] == num_heads_,
                      "head_sink dimension 0 must be equal to the num heads, got ", shape[0]);
    ORT_RETURN_IF_NOT(tensor.IsDataType<T>(), "head_sink type must match GroupQueryAttention input type");

    // Derive the element count from the tensor itself (one sink per head) rather than num_heads_.
    const int head_sink_count = static_cast<int>(shape.Size());
    const size_t head_sink_bytes = tensor.SizeInBytes();
    const void* head_sink_data = tensor.DataRaw();
    IAllocatorUniquePtr<void> head_sink_gpu;
    cudaStream_t stream = cudaStreamLegacy;

    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      head_sink_gpu = IAllocator::MakeUniquePtr<void>(alloc, head_sink_bytes, true);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(head_sink_gpu.get(), head_sink_data, head_sink_bytes,
                                           cudaMemcpyHostToDevice, stream));
      head_sink_data = head_sink_gpu.get();
    }

    xqa_head_sink_ = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(head_sink_count), true);
    using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
    ORT_RETURN_IF_ERROR(LaunchConvertHeadSinkToFloat<CudaT>(
        reinterpret_cast<const CudaT*>(head_sink_data), xqa_head_sink_.get(), head_sink_count, stream,
        GetDeviceProp().maxThreadsPerBlock));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    xqa_head_sink_count_ = head_sink_count;
  }

  return Status::OK();
}

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
template <typename T, typename U>
Status GroupQueryAttention<T, U>::DeclareWorkspaceRequirements(
    gsl::span<const WorkspaceInputShape> input_shapes,
    InlinedVector<WorkspaceRequirement>& requirements) const {
  requirements.clear();
  GQAWorkspaceEstimateConfig config;
  config.qkv_element_size = sizeof(T);
  config.cache_element_size = sizeof(U);
  config.num_heads = num_heads_;
  config.kv_num_heads = kv_num_heads_;
  config.causal = is_unidirectional_ ? 1 : 0;
  config.local_window_size = local_window_size_;
  config.sliding_window_cache = sliding_window_cache_;
  config.do_rotary = do_rotary_;
  config.smooth_softmax = use_smooth_softmax_;
  config.softcap = softcap_;
  config.k_quantization =
      k_quant_type_ == KVQuantizationType::PER_TENSOR
          ? GQAKvQuantizationType::PerTensor
          : (k_quant_type_ == KVQuantizationType::PER_CHANNEL
                 ? GQAKvQuantizationType::PerChannel
                 : GQAKvQuantizationType::None);
  config.v_quantization =
      v_quant_type_ == KVQuantizationType::PER_TENSOR
          ? GQAKvQuantizationType::PerTensor
          : (v_quant_type_ == KVQuantizationType::PER_CHANNEL
                 ? GQAKvQuantizationType::PerChannel
                 : GQAKvQuantizationType::None);
  config.kv_cache_bit_width = kv_cache_bit_width_;
  config.is_bf16 = std::is_same_v<T, BFloat16>;
  config.cache_is_fp8 = std::is_same_v<U, Float8E4M3FN>;
  config.enable_xqa = enable_xqa_;
  config.disable_flash_decode = disable_flash_decode_;
  config.head_sink_is_prepacked = xqa_head_sink_count_ == num_heads_;

  const auto estimate = EstimateGroupQueryAttentionWorkspace(
      config, input_shapes, GetDeviceProp(), *kernel_options_);
  if (estimate.has_value()) {
    SetGroupQueryAttentionWorkspaceRequirements(*estimate, requirements);
  }
  return Status::OK();
}
#endif

// ComputeInternal executes the GQA kernel.
//
// Inputs:
// 0. query             (Tensor) [batch, sequence_length, hidden_size]
// 1. key               (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 2. value             (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 3. past_key          (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 4. past_value        (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 5. seqlens_k         (Tensor) [batch] - Total sequence length minus 1 (for historical compatibility)
// 6. total_seqlen      (Tensor) - Max total sequence length
// 7. cos_cache         (Tensor) - Precomputed cosine table for RoPE
// 8. sin_cache         (Tensor) - Precomputed sine table for RoPE
// 9. position_ids      (Tensor) - Position indices for RoPE
// 10. attention_bias   (Tensor) [batch or 1, num_heads or 1, sequence_length, total_sequence_length] (Optional)
//                      Additive bias to QxK'; dispatches to the unfused fallback path.
// 11. head_sink        (Tensor) - Attention sink for GPT-OSS
template <typename T, typename U>
Status GroupQueryAttention<T, U>::ComputeInternal(OpKernelContext* context) const {
  auto ort_stream = GetOrtStream(context);

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);

  // The input seqlens_k is total sequence length - 1 for historical reasons.
  // Rename it to total_seq_lens_minus_one in cuda kernel to avoid confusion.
  const Tensor* total_seq_lens_minus_one = context->Input<Tensor>(5);

  // The max of total sequence lengths. The content of this tensor is a scalar stored in CPU memory.
  const Tensor* total_seqlen = context->Input<Tensor>(6);

  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);
  const Tensor* position_ids = context->Input<Tensor>(9);
  const Tensor* attention_bias = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);
  const Tensor* k_scale = context->Input<Tensor>(12);
  const Tensor* v_scale = context->Input<Tensor>(13);

  // q_norm_weight (input 14) / k_norm_weight (input 15) carry the per-head Q/K RMSNorm (QK-Norm)
  // prologue weights, each of shape (head_size,) and shared across heads. They are populated by the
  // GroupQueryAttentionPreNormFusion optimizer pass for Qwen3 / Gemma 2-3 / OLMo2 / SmolLM3 style
  // models. Both must be present together; shape validation and wiring happen after CheckInputs,
  // where head_size is known.
  const Tensor* q_norm_weight = (context->InputCount() > 14) ? context->Input<Tensor>(14) : nullptr;
  const Tensor* k_norm_weight = (context->InputCount() > 15) ? context->Input<Tensor>(15) : nullptr;
  if ((q_norm_weight != nullptr) != (k_norm_weight != nullptr)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "GroupQueryAttention (CUDA): q_norm_weight and k_norm_weight must be provided together.");
  }

  if (k_quant_type_ != KVQuantizationType::NONE) {
    if (k_scale == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "k_scale must be provided when k_quant_type is not NONE");
    }

    if (!k_scale->IsDataType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "k_scale must be float tensor");
    }
  }

  if (v_quant_type_ != KVQuantizationType::NONE) {
    if (v_scale == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "v_scale must be provided when v_quant_type is not NONE");
    }
    if (!v_scale->IsDataType<float>()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "v_scale must be float tensor");
    }
  }

  // attention_bias runs on the unfused fallback path (the fused kernels below have no bias input),
  // so the feature combinations that path excludes stay NOT_IMPLEMENTED with the bias present.
  const bool has_attention_bias = attention_bias != nullptr;
  if (has_attention_bias) {
    if (!attention_bias->IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias type must match GroupQueryAttention input type T.");
    }
    if (k_quant_type_ != KVQuantizationType::NONE || v_quant_type_ != KVQuantizationType::NONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "attention_bias with quantized KV cache is not implemented in GroupQueryAttention cuda kernel.");
    }
    if (use_smooth_softmax_ || head_sink != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "attention_bias with smooth softmax or head_sink is not implemented in GroupQueryAttention cuda kernel.");
    }
  }

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;

  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
  typedef typename onnxruntime::cuda::OrtToCudaType<U>::type CudaU;
  GroupQueryAttentionData<CudaT, CudaU> data;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                total_seq_lens_minus_one,
                                                                total_seqlen,
                                                                scale_,
                                                                softcap_,
                                                                kv_cache_bit_width_,
                                                                device_prop.maxThreadsPerBlock,
                                                                /*kv_cache_extra_bits=*/0,
                                                                sliding_window_cache_,
                                                                local_window_size_));

  // The CUDA kernel evicts the minimum number of positions on every step, so the cache it produces
  // is always full at min(T, C). That equals the resident range the operator specifies only when
  // the capacity has no slack above the window (C == W, i.e. G == 1). Reject a larger capacity
  // instead of silently returning a layout that neither matches the spec nor the CPU kernel.
  if (parameters.is_windowed_kv_cache && parameters.kv_cache_real_capacity != local_window_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GroupQueryAttention (CUDA): sliding_window_cache=1 requires the KV cache capacity (",
                           parameters.kv_cache_real_capacity, ") to equal local_window_size (", local_window_size_,
                           ").");
  }

#ifndef USE_INT4_KV_CACHE
  if (kv_cache_bit_width_ == 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv_cache_bit_width==4 is not enabled in this build.");
  }
#endif

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               parameters));

  if (has_attention_bias) {
    const auto& bias_dims = attention_bias->Shape().GetDims();
    parameters.broadcast_attn_bias_dim_0 = bias_dims[0] == 1;
    parameters.broadcast_attn_bias_dim_1 = bias_dims[1] == 1;
  }

  // Validate and enable the per-head Q/K RMSNorm (QK-Norm) prologue (inputs 14/15). Both weights
  // must be 1D tensors of shape (head_size) with element type T (shared across all heads).
  if (q_norm_weight != nullptr) {
    const auto& q_norm_shape = q_norm_weight->Shape();
    const auto& k_norm_shape = k_norm_weight->Shape();
    if (q_norm_shape.NumDimensions() != 1 || q_norm_shape[0] != parameters.head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "q_norm_weight must be a 1D tensor of shape (head_size=", parameters.head_size, ")");
    }
    if (k_norm_shape.NumDimensions() != 1 || k_norm_shape[0] != parameters.head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "k_norm_weight must be a 1D tensor of shape (head_size=", parameters.head_size, ")");
    }
    if (!q_norm_weight->IsDataType<T>() || !k_norm_weight->IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "q_norm_weight/k_norm_weight type must match GroupQueryAttention input type T.");
    }
    parameters.use_qk_norm = true;
  }
  parameters.q_norm_epsilon = q_norm_epsilon_;
  parameters.k_norm_epsilon = k_norm_epsilon_;

  parameters.local_window_size = local_window_size_;
  parameters.is_unidirectional = is_unidirectional_;
  parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr;
  parameters.zeros_count = kZerosCount;
  parameters.zero_ptr = zeros_.get();
  parameters.k_quant_type = k_quant_type_;
  parameters.v_quant_type = v_quant_type_;
  parameters.kv_cache_bit_width = kv_cache_bit_width_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  // The current GQA CUDA implementation will never be able to have a QK output.
  // GQA CUDA uses either flash attention or memory efficient attention. Neither kernel supports returning the QK output.
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckNoQKOutput(
      context->OutputCount(),
      static_cast<int>(Info().template GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Set up present KV output shapes
  // For 4-bit quantization, we pack two 4-bit values into one uint8 byte.
  // Therefore, the dense head size in the tensor shape is halved (rounded up).
  int dense_head_size = (parameters.kv_cache_bit_width == 4) ? (parameters.head_size + 1) / 2 : parameters.head_size;
  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, dense_head_size};

  TensorShape present_shape(present_dims);
  Tensor* present_key_output = context->Output(1, present_shape);    // present_key
  Tensor* present_value_output = context->Output(2, present_shape);  // present_value

  IAllocatorUniquePtr<void> k_buffer;
  IAllocatorUniquePtr<void> v_buffer;
  IAllocatorUniquePtr<void> rotary_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;
  IAllocatorUniquePtr<void> unpacked_qkv_buffer;
  IAllocatorUniquePtr<int> seq_lens_buffer;

  // Flash Attention buffers
  IAllocatorUniquePtr<void> softmax_lse_buffer;
  IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
  IAllocatorUniquePtr<void> out_accum_buffer;

  data.position_ids = (position_ids != nullptr) ? position_ids->Data<int64_t>() : nullptr;

  // Input pointers for both paths
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());

  // Handle Past/Present pointers
  data.k_scale = k_scale == nullptr ? nullptr : reinterpret_cast<const float*>(k_scale->DataRaw());
  data.v_scale = v_scale == nullptr ? nullptr : reinterpret_cast<const float*>(v_scale->DataRaw());

  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaU*>(past_key->Data<U>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaU*>(past_value->Data<U>());
  // Borrow only a non-quantized cache with no KV append and neither present output requested.
  // Aliasing past/present suppresses the copy; the zero-KV preparation path processes Q only.
  const bool read_only_cache = std::is_same<T, U>::value && parameters.kv_sequence_length == 0 &&
                               data.past_key != nullptr && data.past_value != nullptr &&
                               present_key_output == nullptr && present_value_output == nullptr;
  ORT_RETURN_IF(!read_only_cache && (present_key_output == nullptr || present_value_output == nullptr),
                "present_key and present_value are required unless borrowing a non-quantized KV cache");
  ORT_RETURN_IF(read_only_cache && parameters.total_sequence_length > parameters.seqlen_past_kv_cache,
                "total_sequence_length exceeds the borrowed KV cache capacity");
  // The kernel interface uses mutable present pointers, but this borrowed-cache path must not write them.
  data.present_key = read_only_cache ? const_cast<CudaU*>(data.past_key)
                                     : reinterpret_cast<CudaU*>(present_key_output->MutableData<U>());
  data.present_value = read_only_cache ? const_cast<CudaU*>(data.past_value)
                                       : reinterpret_cast<CudaU*>(present_value_output->MutableData<U>());
  // Compute past_present_share_buffer early since it's needed for flash attention path selection.
  bool past_key_shared = (data.past_key != nullptr && data.past_key == data.present_key);
  bool past_value_shared = (data.past_value != nullptr && data.past_value == data.present_value);
  parameters.past_present_share_buffer = past_key_shared && past_value_shared;

  // Eviction rewrites the cache in place, so past and present must be the same buffer.
  ORT_RETURN_IF(parameters.is_windowed_kv_cache && !parameters.past_present_share_buffer,
                "sliding_window_cache=1 requires past_key/present_key and past_value/present_value "
                "to share the same buffer.");

  const int original_present_kv_cache_capacity = parameters.seqlen_present_kv_cache;
  bool is_inputs_quantized = (k_quant_type_ != KVQuantizationType::NONE) || (v_quant_type_ != KVQuantizationType::NONE);
  constexpr bool is_int8 = std::is_same<U, int8_t>::value;
  constexpr bool is_fp8 = std::is_same<U, Float8E4M3FN>::value;

  // Keep the original cache capacity; backend and QKV sizing receive their
  // effective post-staging extents separately.
  GQAWorkspaceProblem workspace_problem;
  workspace_problem.qkv_element_size = sizeof(T);
  workspace_problem.cache_element_size = sizeof(U);
  workspace_problem.batch_size = parameters.batch_size;
  workspace_problem.sequence_length = parameters.sequence_length;
  workspace_problem.num_heads = parameters.num_heads;
  workspace_problem.kv_num_heads = parameters.kv_num_heads;
  workspace_problem.head_size = parameters.head_size;
  workspace_problem.present_kv_cache_capacity = original_present_kv_cache_capacity;
  // XQA derives eight-bit INT8/FP8 storage from U even when legacy models omit the bit-width attribute.
  // Normalize only the recipe input; preserve the original attribute for other runtime paths.
  workspace_problem.kv_cache_bit_width =
      parameters.kv_cache_bit_width == 0 && is_inputs_quantized && (is_int8 || is_fp8)
          ? 8
          : parameters.kv_cache_bit_width;
  workspace_problem.k_quantization =
      k_quant_type_ == KVQuantizationType::PER_TENSOR
          ? GQAKvQuantizationType::PerTensor
          : (k_quant_type_ == KVQuantizationType::PER_CHANNEL ? GQAKvQuantizationType::PerChannel
                                                              : GQAKvQuantizationType::None);
  workspace_problem.v_quantization =
      v_quant_type_ == KVQuantizationType::PER_TENSOR
          ? GQAKvQuantizationType::PerTensor
          : (v_quant_type_ == KVQuantizationType::PER_CHANNEL ? GQAKvQuantizationType::PerChannel
                                                              : GQAKvQuantizationType::None);
  workspace_problem.is_windowed_kv_cache = parameters.is_windowed_kv_cache;
  workspace_problem.is_first_prompt = parameters.is_first_prompt;
  workspace_problem.do_rotary = parameters.do_rotary;
  workspace_problem.is_packed_qkv = parameters.is_packed_qkv;
  workspace_problem.use_qk_norm = parameters.use_qk_norm;
  workspace_problem.requires_separate_past_buffer = past_key_shared != past_value_shared;
  const Tensor* shared_past = past_key_shared ? past_key : past_value;
  workspace_problem.past_kv_cache_capacity =
      workspace_problem.requires_separate_past_buffer ? shared_past->Shape()[2] : 0;

  // Preservation and staging size physical tensor rows. Compaction uses the
  // packed-row width from the cache attributes, matching its copy kernel.
  auto cache_row_problem = workspace_problem;
  if (workspace_problem.requires_separate_past_buffer) {
    cache_row_problem.head_size = shared_past->Shape()[3];
    cache_row_problem.kv_cache_bit_width = 0;
  } else if (parameters.is_windowed_kv_cache && parameters.sequence_length > 1) {
    cache_row_problem.head_size = dense_head_size;
    cache_row_problem.kv_cache_bit_width = 0;
  }
  size_t cache_row_bytes = 0;
  const auto cache_row_status = GetGQACacheRowBytes(cache_row_problem, cache_row_bytes);
  ORT_RETURN_IF_NOT(cache_row_status.IsOK(),
                    "GQA cache row sizing failed: ", cache_row_status.message);
  const auto cache_preparation = GetGQACachePreparationSizes(workspace_problem, cache_row_bytes);
  ORT_RETURN_IF_NOT(cache_preparation.status.IsOK(),
                    "GQA cache preparation sizing failed: ", cache_preparation.status.message);
  const auto& cache_sizes = cache_preparation.sizes;

  IAllocatorUniquePtr<CudaU> separate_past_buffer;
  if (past_key_shared != past_value_shared) {
    // Nonshared preprocessing overwrites present KV, so preserve the aliased past cache first.
    const size_t past_bytes = cache_sizes.separate_past_bytes;
    separate_past_buffer = GetScratchBuffer<CudaU>(past_bytes / sizeof(CudaU), GetComputeStream(context));
    if (past_bytes != 0) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(separate_past_buffer.get(), shared_past->DataRaw(), past_bytes,
                                           cudaMemcpyDeviceToDevice, Stream(context)));
    }
    if (past_key_shared) {
      data.past_key = separate_past_buffer.get();
    } else {
      data.past_value = separate_past_buffer.get();
    }
  }

  // The capacity C of a windowed cache is only guaranteed to cover the attention window, so a step
  // that appends S > 1 tokens can transiently need min(P, C) + S entries: the earliest queries of
  // the step still have to see keys that the last ones have already pushed out. Redirect such a
  // step to a staging cache of that length, seeded with the resident window, and copy the surviving
  // tail back afterwards.
  //
  // This is deliberately not gated on is_first_prompt. The op cannot tell a first prompt from a
  // later prefill chunk: total_sequence_length is supplied by the caller and some frameworks
  // (onnxruntime-genai) pass the allocated buffer length rather than the running total. Staging
  // every multi-token step is correct for both cases and leaves the single-token decode path, which
  // never needs more than C entries, on the cheap in-place eviction path.
  //
  // Doing this here, rather than inside a dedicated kernel path, keeps the staging independent of
  // which attention backend is selected and reuses the existing RoPE/QK-Norm/quantization handling.
  IAllocatorUniquePtr<void> staged_key_buffer;
  IAllocatorUniquePtr<void> staged_value_buffer;
  CudaU* windowed_present_key = nullptr;
  CudaU* windowed_present_value = nullptr;
  int windowed_cache_capacity = 0;
  int staged_cache_capacity = 0;
  if (parameters.is_windowed_kv_cache && parameters.sequence_length > 1) {
    windowed_present_key = data.present_key;
    windowed_present_value = data.present_value;
    windowed_cache_capacity = parameters.seqlen_present_kv_cache;
    staged_cache_capacity = static_cast<int>(cache_sizes.effective_kv_cache_capacity);
    const size_t staged_bytes = cache_sizes.staged_cache_bytes;
    staged_key_buffer = GetScratchBuffer<void>(staged_bytes, GetComputeStream(context));
    staged_value_buffer = GetScratchBuffer<void>(staged_bytes, GetComputeStream(context));

    ORT_RETURN_IF_ERROR(LaunchCopyKvCacheWindow(
        staged_key_buffer.get(), staged_value_buffer.get(),
        windowed_present_key, windowed_present_value,
        parameters.batch_size, parameters.kv_num_heads,
        /*src_capacity=*/windowed_cache_capacity, /*dst_capacity=*/staged_cache_capacity,
        /*rows=*/windowed_cache_capacity, /*src_offsets=*/nullptr,
        static_cast<int>(dense_head_size * sizeof(U)), Stream(context)));

    data.past_key = reinterpret_cast<const CudaU*>(staged_key_buffer.get());
    data.past_value = reinterpret_cast<const CudaU*>(staged_value_buffer.get());
    data.present_key = reinterpret_cast<CudaU*>(staged_key_buffer.get());
    data.present_value = reinterpret_cast<CudaU*>(staged_value_buffer.get());

    parameters.kv_cache_capacity = staged_cache_capacity;
    parameters.seqlen_past_kv_cache = staged_cache_capacity;
    parameters.seqlen_present_kv_cache = staged_cache_capacity;
  }

  const int effective_workspace_kv_length = static_cast<int>(GetGQAEffectiveWorkspaceKvLength(
      parameters.total_sequence_length,
      parameters.seqlen_present_kv_cache,
      parameters.is_windowed_kv_cache));

  // Allocate XQA scratch if needed (only for Flash Decoding path)
  IAllocatorUniquePtr<void> xqa_scratch_buffer;
  // Check conditions to enable XQA (Extreme Query Attention) kernel for optimized decoding.
  // XQA is a highly optimized kernel for generation phase (seq_len=1).
  // Constraints:
  // 1. Compute Capability SM 8.0+ (Ampere or newer).
  // 2. Not the first prompt (decoding phase).
  // 3. Sequence length is 1.
  // 4. Past and Present KV cache share the same buffer (required for XQA specific memory access).
  // 5. No Softcap (XQA doesn't support softcap).
  // 6. Standard Softmax, or smooth softmax represented by a head_sink tensor.
  // 7. Local window (sliding window) attention is supported on both the non-quantized and the
  //    quantized (INT8/FP8) XQA paths (local_window_size == -1 means global attention).
  // QK-Norm can use XQA for the non-quantized KV-cache path: ExtremeDecoding runs the same
  // UnpackRoPEAppend preprocess before XQA, so Q/K can be normalized before the XQA kernel consumes
  // Q and the appended cache. Keep quantized QK-Norm off the XQA route until scale correctness is
  // validated for normalized K before quantized-cache append.
  const bool xqa_qk_norm_ok = !parameters.use_qk_norm || !is_inputs_quantized;
  // Attention sinks (smooth softmax) work on the quantized INT8/FP8 XQA paths as well: the sink
  // term is folded into the softmax row sum by the same kernel code, and the KV dequant scale is
  // already applied to the QK scores (qkScale) before the row max/sum are computed, so sink and
  // score live in the same dequantized domain. This holds for both the single-block and the
  // multi-block (Flash Decoding) reductions, where the sink is added to the merged row sum.
  const bool use_xqa_attention_sinks = head_sink != nullptr;
  const bool is_xqa_smooth_softmax_supported = !parameters.use_smooth_softmax || use_xqa_attention_sinks;
  // XQA is enabled when enable_xqa_=true; ineligible shapes/group sizes fall back via data.use_xqa below.
  // The XQA kernel has no attention_bias input.
  GQAXqaSeqFreeInputs xqa_inputs;
  xqa_inputs.enable_xqa = enable_xqa_;
  xqa_inputs.is_unidirectional = parameters.is_unidirectional;
  xqa_inputs.has_attention_bias = has_attention_bias;
  xqa_inputs.device_major = device_prop.major;
  xqa_inputs.device_minor = device_prop.minor;
  xqa_inputs.softcap = parameters.softcap;
  xqa_inputs.qk_norm_ok = xqa_qk_norm_ok;
  xqa_inputs.smooth_softmax_supported = is_xqa_smooth_softmax_supported;
  xqa_inputs.is_inputs_quantized = is_inputs_quantized;
  xqa_inputs.head_size = parameters.head_size;
  xqa_inputs.num_heads = parameters.num_heads;
  xqa_inputs.kv_num_heads = parameters.kv_num_heads;
  xqa_inputs.k_quant_type = k_quant_type_;
  xqa_inputs.v_quant_type = v_quant_type_;

  // Phase and actual past/present buffer aliasing are per-Run facts, not partition-time inputs.
  const bool xqa_phase_ok = !parameters.is_first_prompt &&
                            parameters.sequence_length == 1 &&
                            parameters.kv_sequence_length > 0;  // Shared KV (kv_seq=0) has no new K/V to append
  if (xqa_phase_ok && parameters.past_present_share_buffer && IsGQAXqaEligibleSeqFree<U>(xqa_inputs)) {
    data.use_xqa = true;

    if (data.use_xqa) {
      // Consumer Blackwell (sm_120) and some other devices expose a smaller per-block opt-in
      // shared-memory limit than the SM the XQA kernel was compiled / JIT-compiled for (sm_80 and
      // sm_90 use a larger K/V-tile layout, ~140 KB for head_size 128/256). Launching XQA there
      // fails with cudaErrorInvalidValue because cudaFuncSetAttribute requests more shared memory
      // than the device allows. Query the actual kernel requirement (read from the device symbol,
      // so it is accurate even for a PTX kernel JIT-compiled for this SM) and fall back to cuDNN
      // SDPA / Flash when it does not fit. Cached per node because head_size and group size are
      // constant for a given GQA node (avoids a device->host copy on every decode step).
      int xqa_smem_ok = xqa_shared_memory_ok_.load(std::memory_order_relaxed);
      if (xqa_smem_ok < 0) {
        // GetXQARequiredSharedMemoryBytes issues a cudaMemcpyFromSymbol, which synchronizes and is
        // therefore illegal during CUDA graph capture (it would invalidate the capture). The result
        // is invariant for a node (fixed head_size / group size / device), and ORT performs at least
        // one non-captured warm-up run before capturing, so this normally resolves and caches during
        // warm-up. Guard defensively: only issue the synchronizing query when the compute stream is
        // not capturing. If we somehow reach here for the first time while capturing, conservatively
        // skip XQA for this run instead of breaking the capture, and leave the cache unresolved so a
        // later non-capturing run can determine it.
        if (!onnxruntime::llm::common::isCapturing(Stream(context))) {
          const size_t required_smem = onnxruntime::contrib::cuda::GetXQARequiredSharedMemoryBytes(
              device_prop, parameters.head_size, parameters.num_heads, parameters.kv_num_heads);
          xqa_smem_ok = (required_smem == 0 || required_smem <= device_prop.sharedMemPerBlockOptin) ? 1 : 0;
          xqa_shared_memory_ok_.store(xqa_smem_ok, std::memory_order_relaxed);
        } else {
          xqa_smem_ok = 0;  // capturing (or status query failed) and not yet resolved -> fall back
        }
      }
      data.use_xqa = (xqa_smem_ok != 0);
    }

    if (data.use_xqa) {
      // Backend-internal XQA scratch (internal_scratch_bytes reproduces GetXQAScratchSize) plus the
      // optional RoPE Q/K and dynamic head-sink regions come from the shared XQA workspace recipe.
      // The recipe lays out those trailing regions in the same single allocation and byte order as
      // the previous inline math (internal | rotary Q | rotary K | dynamic head sink).
      GQAXqaConfig xqa_config;
      xqa_config.device_major = device_prop.major;
      xqa_config.device_minor = device_prop.minor;
      xqa_config.multi_processor_count = device_prop.multiProcessorCount;
      xqa_config.kv_type = parameters.k_quant_type != KVQuantizationType::NONE
                               ? (is_fp8 ? GQAXqaKvType::Fp8 : GQAXqaKvType::Int8)
                               : GQAXqaKvType::None;
      const bool use_prepacked_xqa_head_sink =
          use_xqa_attention_sinks && xqa_head_sink_ != nullptr && xqa_head_sink_count_ == parameters.num_heads;
      const bool convert_xqa_head_sink = use_xqa_attention_sinks && !use_prepacked_xqa_head_sink;
      xqa_config.head_sink_storage =
          use_prepacked_xqa_head_sink
              ? GQAXqaHeadSinkStorage::PrepackedFp32
              : (convert_xqa_head_sink ? GQAXqaHeadSinkStorage::DynamicConversion
                                       : GQAXqaHeadSinkStorage::None);
      xqa_config.is_bf16 = std::is_same<T, BFloat16>::value;

      const auto xqa = GetGQAXqaWorkspaceRecipe(workspace_problem, xqa_config);
      ORT_RETURN_IF_NOT(xqa.status.IsOK(),
                        "GQA XQA workspace sizing failed: ", xqa.status.message);
      const GQAXqaWorkspaceRecipe& recipe = xqa.recipe;

      xqa_scratch_buffer = this->GetScratchBuffer<void>(recipe.total_backend_bytes, GetComputeStream(context));
      data.xqa_buffer = xqa_scratch_buffer.get();
      data.xqa_buffer_bytes = recipe.internal_scratch_bytes;

      char* xqa_base = reinterpret_cast<char*>(data.xqa_buffer);
      if (parameters.do_rotary) {
        data.qkv_buffer = reinterpret_cast<CudaT*>(xqa_base + recipe.rotary_q_offset_bytes);
      }
      if (use_prepacked_xqa_head_sink) {
        data.xqa_head_sink = xqa_head_sink_.get();
      } else if (convert_xqa_head_sink) {
        data.xqa_head_sink = reinterpret_cast<float*>(xqa_base + recipe.dynamic_head_sink_offset_bytes);
        data.xqa_head_sink_needs_conversion = true;
      }
    }
  }

  // === cuDNN SDPA eligibility (preferred on SM>=90, Hopper/Blackwell) ===
  // Constrained to non-quantized FP16/BF16 KV cache, no softcap, no smooth-softmax / head sink,
  // and no sliding window. Rotary and packed QKV are handled by PrepareQKV before the kernel runs;
  // cuDNN handles grouped-query attention natively. Keep this path bias-free because an arbitrary
  // attention bias cannot be composed with cuDNN's fused bottom-right causal mask.
  bool use_cudnn_sdpa = !data.use_xqa &&
                        IsGQACudnnSdpaCoreEligibleSeqFree<T, U>(
                            has_attention_bias,
                            is_inputs_quantized,
                            parameters.softcap,
                            parameters.use_smooth_softmax,
                            head_sink != nullptr,
                            parameters.local_window_size,
                            parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH,
                            enable_cudnn_flash_attention_ ||
                                (auto_enable_cudnn_flash_attention_ && device_prop.major >= 9)) &&
                        onnxruntime::cudnn_sdpa::is_stable() &&
                        onnxruntime::cudnn_sdpa::is_supported(device_prop,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads,
                                                              parameters.head_size,
                                                              parameters.head_size,
                                                              parameters.sequence_length,          // seq_len_q
                                                              parameters.seqlen_present_kv_cache,  // seq_len_kv (capacity)
                                                              parameters.is_unidirectional);
  data.use_cudnn_sdpa = use_cudnn_sdpa;

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !data.use_xqa &&
                             !data.use_cudnn_sdpa &&
                             IsGQAFlashEligibleSeqFree<T>(device_prop,
                                                          has_attention_bias,
                                                          disable_flash_attention_,
                                                          parameters.head_size,
                                                          parameters.num_heads,
                                                          parameters.kv_num_heads);

  data.use_flash_attention = use_flash_attention;
  // The fast-decode path lets the flash kernel perform RoPE and KV-append internally, bypassing
  // PrepareQKV (and therefore the fused QK-Norm prologue). Disable it when q/k norm weights are
  // present so the regular FlashAttention path (which normalizes via PrepareQKV) is used instead.
  // Reusing seqlens_k directly avoids a sequence-length kernel launch for single-token decode,
  // where the input T - 1 is also the cache append offset P. Multi-token decode requires
  // LaunchGetSequenceLengths to derive P = T - sequence_length.
  // It is also disabled for a windowed KV cache: the kernel derives both the absolute RoPE position
  // and the cache append offset from a single seqlens_k value, which those two no longer share.
  // Explicit RoPE positions may differ from cache offsets and require position-aware PrepareQKV.
  data.use_flash_attention_fast_decode = use_flash_attention && !disable_flash_decode_ && !parameters.is_first_prompt &&
                                         parameters.sequence_length == 1 && parameters.kv_sequence_length > 0 &&
                                         parameters.past_present_share_buffer && !is_inputs_quantized &&
                                         !parameters.use_qk_norm && !parameters.is_windowed_kv_cache &&
                                         (!parameters.do_rotary || data.position_ids == nullptr);

  if (use_flash_attention) {
    // Flash-specific buffer sizes (softmax LSE and the optional split accumulators) come from the
    // shared Flash workspace recipe. The recipe reproduces get_num_splits_and_buffer_sizes,
    // including the fast-decode correction that sizes the accumulators with num_heads rather than
    // the kv_num_heads used by the split heuristic, and the get_softmax_lse_* helpers.
    GQAFlashConfig flash_config;
    flash_config.total_sequence_length = effective_workspace_kv_length;
    flash_config.local_window_size = parameters.local_window_size;
    flash_config.multi_processor_count = device_prop.multiProcessorCount;
    flash_config.fast_decode = data.use_flash_attention_fast_decode;

    const auto flash_result = GetGQAFlashWorkspaceRecipe(workspace_problem, flash_config);
    ORT_RETURN_IF_NOT(flash_result.status.IsOK(),
                      "GQA flash attention workspace sizing failed: ", flash_result.status.message);
    const GQAFlashWorkspaceRecipe& recipe = flash_result.recipe;

    // Preserve the num_splits side effect: it feeds the kernel launch and debug info below.
    parameters.num_splits = static_cast<int>(recipe.runtime_num_splits);

    const size_t softmax_lse_bytes = recipe.softmax_lse_bytes;
    const size_t softmax_lse_accum_bytes = recipe.softmax_lse_accumulator_bytes;
    const size_t out_accum_bytes = recipe.output_accumulator_bytes;

    softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));
    softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, GetComputeStream(context));
    out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, GetComputeStream(context));

    auto cuda_stream = Stream(context);
    if (softmax_lse_accum_bytes > 0) {
      // Initialize to 0 is fine because Flash kernel will write -inf to it if needed.
      // However, the standard Flash kernel often doesn't zero it globally.
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0, softmax_lse_accum_bytes, cuda_stream));
    }
    if (out_accum_bytes > 0) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0, out_accum_bytes, cuda_stream));
    }

    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }
#endif

  auto cuda_stream = Stream(context);

  if (data.use_flash_attention_fast_decode) {
    // FlashAttention clamps this device input to the cache capacity before deriving offsets.
    data.past_seq_lens = const_cast<int*>(total_seq_lens_minus_one->Data<int>());
  } else {
    size_t seq_lens_vectors = 0;
    size_t seq_lens_bytes = 0;
    const auto seq_lens_status = GetGQASequenceLengthsSize(
        workspace_problem, data.use_flash_attention_fast_decode, seq_lens_vectors, seq_lens_bytes);
    ORT_RETURN_IF_NOT(seq_lens_status.IsOK(),
                      "GQA sequence-length sizing failed: ", seq_lens_status.message);
    seq_lens_buffer = GetScratchBuffer<int>(seq_lens_bytes / sizeof(int), GetComputeStream(context));
    data.past_seq_lens = seq_lens_buffer.get();
    data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
    data.padded_seq_lens = data.total_seq_lens + parameters.batch_size;
    if (parameters.is_windowed_kv_cache) {
      data.cache_past_seq_lens = data.padded_seq_lens + parameters.batch_size;
      data.cache_total_seq_lens = data.cache_past_seq_lens + parameters.batch_size;
      data.evict_counts = data.cache_total_seq_lens + parameters.batch_size;
    }
    ORT_RETURN_IF_ERROR(LaunchGetSequenceLengths(total_seq_lens_minus_one->Data<int>(),
                                                 data.past_seq_lens,
                                                 data.total_seq_lens,
                                                 data.padded_seq_lens,
                                                 data.cache_past_seq_lens,
                                                 data.cache_total_seq_lens,
                                                 data.evict_counts,
                                                 parameters.batch_size,
                                                 parameters.sequence_length,
                                                 parameters.is_first_prompt,
                                                 parameters.total_sequence_length,
                                                 parameters.kv_cache_capacity,
                                                 parameters.kv_cache_real_capacity,
                                                 cuda_stream,
                                                 device_prop.maxThreadsPerBlock));
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("total_seq_lens", data.total_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("past_seq_lens", data.past_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("padded_seq_lens", data.padded_seq_lens, parameters.batch_size, 1);
  }

  // Scratch used to left-shift the KV cache when the sliding window advances. Sized for the whole
  // cache because the number of evicted rows is only known on device (and must stay constant across
  // CUDA graph replays). K and V shift in the same kernel, so each needs its own half.
  IAllocatorUniquePtr<void> compaction_buffer;
  if (parameters.is_windowed_kv_cache && staged_cache_capacity == 0) {
    compaction_buffer = GetScratchBuffer<void>(cache_sizes.compaction_bytes, GetComputeStream(context));
    data.compaction_scratch = compaction_buffer.get();
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (!data.use_xqa && !data.use_cudnn_sdpa && !data.use_flash_attention) {
    // Fall back to memory efficient attention.
    int sm = (device_prop.major * 10) + device_prop.minor;
    // With attention_bias, MEA is skipped: the cutlass wrapper computes the bias row stride from
    // kv_sequence_length, which GQA sets to the KV-cache capacity (seqlen_present_kv_cache), not
    // the bias row length (total_sequence_length) — mismatched under past/present buffer sharing.
    // Bias-carrying nodes take the unfused fallback below instead.
    bool use_memory_efficient_attention =
        IsGQAMemoryEfficientEligible<T>(parameters, sm, disable_memory_efficient_attention_,
                                        is_inputs_quantized, has_attention_bias, head_sink != nullptr);
    data.use_memory_efficient_attention = use_memory_efficient_attention;

    // Head-expansion (K/V) and FP32 FMHA-accumulator scratch sizes come from the shared
    // MEA workspace recipe. It returns zero for the expansion buffers when num_heads ==
    // kv_num_heads and zero for the accumulator when head_size <= 128, matching the prior
    // inline gating (MemoryEfficientAttentionParams::need_workspace).
    size_t kv_buffer_bytes = 0;
    size_t fmha_buffer_bytes = 0;
    if (use_memory_efficient_attention) {
      const auto mea = GetGQAMemoryEfficientWorkspaceRecipe(
          workspace_problem, parameters.seqlen_present_kv_cache);
      ORT_RETURN_IF_NOT(mea.status.IsOK(),
                        "GQA memory-efficient attention workspace sizing failed: ", mea.status.message);
      kv_buffer_bytes = mea.recipe.expanded_key_bytes;
      fmha_buffer_bytes = mea.recipe.output_accumulator_bytes;
    }

    k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, GetComputeStream(context));
    v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, GetComputeStream(context));
    fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, GetComputeStream(context));

    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }
#endif

  GQAPreparationRoute preparation_route;
  preparation_route.preprocess_mode =
      data.use_xqa ? GQAPreprocessMode::Xqa
                   : (data.use_flash_attention
                          ? GQAPreprocessMode::Flash
                          : (data.use_memory_efficient_attention ? GQAPreprocessMode::MemoryEfficient
                                                                 : GQAPreprocessMode::Unfused));
  preparation_route.use_flash_attention_fast_decode = data.use_flash_attention_fast_decode;
  size_t qkv_preprocess_bytes = 0;
  const auto qkv_status = GetGQAQkvPreprocessBytes(
      workspace_problem, preparation_route, parameters.seqlen_present_kv_cache, qkv_preprocess_bytes);
  ORT_RETURN_IF_NOT(qkv_status.IsOK(),
                    "GQA QKV preparation sizing failed: ", qkv_status.message);

  if (qkv_preprocess_bytes > 0) {
    unpacked_qkv_buffer = GetScratchBuffer<void>(qkv_preprocess_bytes, GetComputeStream(context));
    data.qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }

  // ---------------------------------------------------------------------
  // GQA-capable unfused fallback (issue #28195).
  // Activates when Flash / MEA / XQA are all ineligible and KV is not quantized.
  // Supports any head_size (FP32 QK accumulation), GQA, sliding window, softcap.
  // See LaunchUnfusedAttention in contrib_ops/cuda/bert/unfused_attention.h.
  // ---------------------------------------------------------------------
  IAllocatorUniquePtr<void> unfused_scratch;
  if (!data.use_xqa && !data.use_cudnn_sdpa && !data.use_flash_attention && !data.use_memory_efficient_attention &&
      !is_inputs_quantized && !parameters.use_smooth_softmax && head_sink == nullptr &&
      parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH) {
    data.use_unfused = true;

    // CheckInputs validates equal Q/K/V head sizes; v_head_size == 0 means use head_size.
    // Enforce the UnfusedGqaAttention invariant before sizing its Q/Y buffers.
    ORT_RETURN_IF_NOT(parameters.v_head_size == 0 || parameters.v_head_size == parameters.head_size,
                      "UnfusedGqaAttention requires head_size == v_head_size");
    // The shared recipe retains the aligned Q/Y and FP32 QK/softmax layout, using the
    // same resident/staged KV extent passed to the unfused kernel.
    const auto unfused = GetGQAUnfusedWorkspaceRecipe(workspace_problem, effective_workspace_kv_length);
    ORT_RETURN_IF_NOT(unfused.status.IsOK(),
                      "GQA unfused attention workspace sizing failed: ", unfused.status.message);
    const GQAUnfusedWorkspaceRecipe& recipe = unfused.recipe;

    unfused_scratch = GetScratchBuffer<void>(recipe.total_backend_bytes, GetComputeStream(context));
    auto* base = reinterpret_cast<uint8_t*>(unfused_scratch.get());
    data.unfused_q_bnsh = reinterpret_cast<CudaT*>(base + recipe.q_bnsh_offset_bytes);
    data.unfused_y_bnsh = reinterpret_cast<CudaT*>(base + recipe.y_bnsh_offset_bytes);
    data.unfused_workspace = reinterpret_cast<void*>(base + recipe.qk_offset_bytes);
  }

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_xqa = data.use_xqa;
    debug_info.use_flash_attention = data.use_flash_attention;
    debug_info.use_efficient_attention = data.use_memory_efficient_attention;
    debug_info.use_cudnn_flash_attention = data.use_cudnn_sdpa;
    if (data.use_flash_attention) {
      debug_info.num_splits = parameters.num_splits;
    }
    debug_info.effective_kv_length_bound = effective_workspace_kv_length;

    debug_info.Print("GroupQueryAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  if (head_sink != nullptr) {
    data.head_sink = reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  }

  if (has_attention_bias) {
    data.attention_bias = reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  }

  if (parameters.use_qk_norm) {
    data.q_norm_weight = reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
    data.k_norm_weight = reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
  }

#if DUMP_TENSOR_LEVEL > 0
  DUMP_TENSOR_INIT();
  // Dump Scales
  if (data.k_scale) {
    if (parameters.k_quant_type == KVQuantizationType::PER_TENSOR) {
      DUMP_TENSOR("k_scale", data.k_scale, 1, 1);
    } else if (parameters.k_quant_type == KVQuantizationType::PER_CHANNEL) {
      DUMP_TENSOR("k_scale", data.k_scale, parameters.kv_num_heads, 1, parameters.head_size);
    }
  }
  if (data.v_scale) {
    if (parameters.v_quant_type == KVQuantizationType::PER_TENSOR) {
      DUMP_TENSOR("v_scale", data.v_scale, 1, 1);
    } else if (parameters.v_quant_type == KVQuantizationType::PER_CHANNEL) {
      DUMP_TENSOR("v_scale", data.v_scale, parameters.kv_num_heads, 1, parameters.head_size);
    }
  }
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  if (data.use_cudnn_sdpa) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&data.allocator));
    data.cudnn_handle = static_cast<void*>(GetCudnnHandle(context));
  }

  ORT_RETURN_IF_ERROR((QkvToContext<CudaT, CudaU>(
      device_prop, cublas, ort_stream.get(), parameters, data)));

  if (windowed_present_key != nullptr) {
    // Attention ran against the staging cache; keep only the entries the sliding window can still
    // reach. evict_counts[b] = max(0, min(P, C) + S - C) is exactly where that tail starts. The rows
    // are already in their final (possibly quantized) form.
    const int row_bytes = static_cast<int>(dense_head_size * sizeof(U));
    ORT_RETURN_IF_ERROR(LaunchCopyKvCacheWindow(
        windowed_present_key, windowed_present_value, data.present_key, data.present_value,
        parameters.batch_size, parameters.kv_num_heads,
        /*src_capacity=*/staged_cache_capacity, /*dst_capacity=*/windowed_cache_capacity,
        /*rows=*/windowed_cache_capacity, /*src_offsets=*/data.evict_counts,
        row_bytes, Stream(context)));
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
