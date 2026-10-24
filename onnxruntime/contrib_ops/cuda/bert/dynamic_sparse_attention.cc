// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/dynamic_sparse_attention.h"

#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_impl.h"
#include "core/common/safeint.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      DynamicSparseAttention,                                               \
      kMSDomain,                                                            \
      1,                                                                    \
      T,                                                                    \
      kCudaExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("TCache", BuildKernelDefConstraints<T, int8_t>()) \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())      \
          .MayInplace(3, 1)                                                 \
          .MayInplace(4, 2)                                                 \
          .InputMemoryType(OrtMemTypeCPUInput, 10),                         \
      DynamicSparseAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

namespace {

// Enables synchronous device metadata checks for diagnosing malformed standalone inputs.
constexpr const char* kStrictValidation = "ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION";

DynamicSparseAttentionMode ParseAttentionMode(const std::string& value) {
  ORT_ENFORCE(value == "selected_only" || value == "local_plus_selected",
              "DynamicSparseAttention: attention_mode must be selected_only or local_plus_selected.");
  return value == "selected_only" ? DynamicSparseAttentionMode::kSelectedOnly
                                  : DynamicSparseAttentionMode::kLocalPlusSelected;
}

DynamicSparseAttentionKvSource ParseKvSource(const std::string& value) {
  ORT_ENFORCE(value == "main" || value == "auxiliary",
              "DynamicSparseAttention: selected_kv_source must be main or auxiliary.");
  return value == "main" ? DynamicSparseAttentionKvSource::kMain
                         : DynamicSparseAttentionKvSource::kAuxiliary;
}

bool ParseBoolAttribute(const OpKernelInfo& info, const char* name, int64_t default_value) {
  const int64_t value = info.GetAttrOrDefault<int64_t>(name, default_value);
  ORT_ENFORCE(value == 0 || value == 1, "DynamicSparseAttention: ", name, " must be 0 or 1.");
  return value != 0;
}

}  // namespace

template <typename T>
DynamicSparseAttention<T>::DynamicSparseAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() &&
                  num_heads > 0 && num_heads <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: num_heads must be a positive int.");
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 && kv_num_heads <= std::numeric_limits<int>::max() &&
                  num_heads % kv_num_heads == 0,
              "DynamicSparseAttention: kv_num_heads must be positive and divide num_heads.");
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  const int64_t is_causal = info.GetAttrOrDefault<int64_t>("is_causal", 1);
  ORT_ENFORCE(is_causal == 1, "DynamicSparseAttention only supports is_causal=1.");

  const int64_t local_window_size = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  ORT_ENFORCE(local_window_size == -1 ||
                  (local_window_size > 0 && local_window_size <= std::numeric_limits<int>::max()),
              "DynamicSparseAttention: local_window_size must be -1 or a positive int.");
  local_window_size_ = static_cast<int>(local_window_size);

  const int64_t rotary_offset = info.GetAttrOrDefault<int64_t>("rotary_offset", 0);
  ORT_ENFORCE(rotary_offset >= 0 && rotary_offset <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: rotary_offset must be a nonnegative int.");
  rotary_offset_ = static_cast<int>(rotary_offset);

  scale_ = 0.0f;
  has_scale_ = info.GetAttr("scale", &scale_).IsOK();
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  do_rotary_ = ParseBoolAttribute(info, "do_rotary", 0);
  rotary_interleaved_ = ParseBoolAttribute(info, "rotary_interleaved", 0);
  use_smooth_softmax_ = ParseBoolAttribute(info, "smooth_softmax", 0);
  auxiliary_kv_shared_ = ParseBoolAttribute(info, "auxiliary_kv_shared", 0);
  strict_validation_ = ParseEnvironmentVariableWithDefault<bool>(kStrictValidation, false);
  attention_mode_ = ParseAttentionMode(
      info.GetAttrOrDefault<std::string>("attention_mode", "selected_only"));
  selected_kv_source_ = ParseKvSource(
      info.GetAttrOrDefault<std::string>("selected_kv_source", "main"));
}

template <typename T>
Status DynamicSparseAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* auxiliary_key = context->Input<Tensor>(5);
  const Tensor* auxiliary_value = context->Input<Tensor>(6);
  const Tensor* selected_indices = context->Input<Tensor>(7);
  const Tensor* selected_counts = context->Input<Tensor>(8);
  const Tensor* seqlens_k = context->Input<Tensor>(9);
  const Tensor* total_sequence_length = context->Input<Tensor>(10);
  const Tensor* cos_cache = context->Input<Tensor>(11);
  const Tensor* sin_cache = context->Input<Tensor>(12);
  const Tensor* position_ids = context->Input<Tensor>(13);
  const Tensor* q_norm_weight = context->Input<Tensor>(14);
  const Tensor* k_norm_weight = context->Input<Tensor>(15);
  const Tensor* head_sink = context->Input<Tensor>(16);
  const Tensor* committed_length =
      context->InputCount() > 17 ? context->Input<Tensor>(17) : nullptr;
  const Tensor* key_scale = context->InputCount() > 18 ? context->Input<Tensor>(18) : nullptr;
  const Tensor* value_scale = context->InputCount() > 19 ? context->Input<Tensor>(19) : nullptr;
  const bool quantized_cache = past_key != nullptr && past_key->IsDataType<int8_t>();
  ORT_RETURN_IF_NOT((key_scale != nullptr) == (value_scale != nullptr),
                    "DynamicSparseAttention: key_scale and value_scale must be provided together.");
  ORT_RETURN_IF_NOT(quantized_cache == (key_scale != nullptr),
                    "DynamicSparseAttention: INT8 caches require per-channel key_scale and value_scale.");
  ORT_RETURN_IF_NOT((past_key == nullptr || past_key->IsDataType<T>() || quantized_cache) &&
                        (past_value == nullptr || (quantized_cache ? past_value->IsDataType<int8_t>()
                                                                   : past_value->IsDataType<T>())),
                    "DynamicSparseAttention: cache types must match activations or both be INT8.");

  DynamicSparseAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(dynamic_sparse_attention_helper::CheckInputs(
      query, key, value, past_key, past_value, auxiliary_key, auxiliary_value,
      selected_indices, selected_counts, seqlens_k, total_sequence_length,
      cos_cache, sin_cache, position_ids, q_norm_weight, k_norm_weight, head_sink,
      num_heads_, kv_num_heads_, local_window_size_, rotary_offset_, do_rotary_,
      auxiliary_kv_shared_, attention_mode_, selected_kv_source_, scale_, has_scale_,
      qk_norm_epsilon_, parameters));
  parameters.rotary_interleaved = rotary_interleaved_;
  parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr;
  if (quantized_cache) {
    ORT_RETURN_IF_NOT(past_value != nullptr &&
                          key_scale->Shape() == TensorShape({parameters.kv_num_heads * parameters.head_size}) &&
                          value_scale->Shape() == key_scale->Shape(),
                      "DynamicSparseAttention: INT8 cache scales must have shape [kv_num_heads * head_size].");
  }

  ORT_RETURN_IF_NOT(parameters.head_size <= GetDeviceProp().maxThreadsPerBlock,
                    "DynamicSparseAttention: head_size exceeds the CUDA device thread-block limit.");
  ORT_RETURN_IF(committed_length != nullptr &&
                    (parameters.batch_size != 1 || committed_length->Shape().Size() != 1),
                "DynamicSparseAttention: committed_length requires batch_size=1 and one int32 element.");

  TensorShape output_shape({parameters.batch_size, parameters.sequence_length,
                            parameters.query_hidden_size});
  Tensor* output = context->Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "DynamicSparseAttention: output is required.");

  TensorShape present_shape({parameters.batch_size, parameters.kv_num_heads,
                             parameters.cache_capacity, parameters.head_size});
  Tensor* present_key_output = context->OutputCount() > 1 ? context->Output(1, present_shape) : nullptr;
  Tensor* present_value_output = context->OutputCount() > 2 ? context->Output(2, present_shape) : nullptr;
  Tensor* validation_status_output =
      context->OutputCount() > 3 ? context->Output(3, TensorShape({1})) : nullptr;
  const size_t query_elements = SafeInt<size_t>(parameters.batch_size) *
                                parameters.sequence_length * parameters.query_hidden_size;
  auto prepared_query = GetScratchBuffer<T>(query_elements, GetComputeStream(context));

  const size_t cache_elements = SafeInt<size_t>(parameters.batch_size) *
                                parameters.kv_num_heads * parameters.cache_capacity *
                                parameters.head_size;
  const size_t cache_bytes = SafeInt<size_t>(cache_elements) * (quantized_cache ? 1 : sizeof(T));
  IAllocatorUniquePtr<uint8_t> key_cache_scratch;
  IAllocatorUniquePtr<uint8_t> value_cache_scratch;
  void* key_cache = present_key_output == nullptr ? nullptr : present_key_output->MutableDataRaw();
  void* value_cache = present_value_output == nullptr ? nullptr : present_value_output->MutableDataRaw();
  if (key_cache == nullptr) {
    key_cache_scratch = GetScratchBuffer<uint8_t>(cache_bytes, GetComputeStream(context));
    key_cache = key_cache_scratch.get();
  }
  if (value_cache == nullptr) {
    value_cache_scratch = GetScratchBuffer<uint8_t>(cache_bytes, GetComputeStream(context));
    value_cache = value_cache_scratch.get();
  }

  using CudaT = typename OrtToCudaType<T>::type;
  DynamicSparseAttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.past_key = past_key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_key->DataRaw());
  data.past_value = past_value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_value->DataRaw());
  data.quantized_cache = quantized_cache;
  data.key_scale = key_scale == nullptr ? nullptr : key_scale->Data<float>();
  data.value_scale = value_scale == nullptr ? nullptr : value_scale->Data<float>();
  data.auxiliary_key =
      auxiliary_key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(auxiliary_key->Data<T>());
  data.auxiliary_value = auxiliary_value == nullptr
                             ? data.auxiliary_key
                             : reinterpret_cast<const CudaT*>(auxiliary_value->Data<T>());
  data.selected_indices = selected_indices->Data<int32_t>();
  data.selected_counts = selected_counts->Data<int32_t>();
  data.seqlens_k = seqlens_k->Data<int32_t>();
  data.cos_cache = cos_cache == nullptr ? nullptr : reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
  data.sin_cache = sin_cache == nullptr ? nullptr : reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  data.position_ids = position_ids == nullptr ? nullptr : position_ids->Data<int64_t>();
  data.q_norm_weight =
      q_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
  data.k_norm_weight =
      k_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
  data.head_sink = head_sink == nullptr ? nullptr : reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  data.prepared_query = reinterpret_cast<CudaT*>(prepared_query.get());
  data.present_key = reinterpret_cast<CudaT*>(key_cache);
  data.present_value = reinterpret_cast<CudaT*>(value_cache);
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  cudaStream_t stream = Stream(context);
  if (strict_validation_) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
    ORT_RETURN_IF_NOT(capture_status == cudaStreamCaptureStatusNone,
                      "DynamicSparseAttention strict validation is not supported during CUDA graph capture.");
  }

  IAllocatorUniquePtr<int32_t> validation_error_scratch;
  int32_t* validation_error = nullptr;
  if (validation_status_output != nullptr) {
    validation_error = validation_status_output->MutableData<int32_t>();
  } else if (strict_validation_) {
    validation_error_scratch = GetScratchBuffer<int32_t>(1, GetComputeStream(context));
    validation_error = validation_error_scratch.get();
  }
  if (validation_error != nullptr) {
    const size_t validation_workspace_elements =
        GetDynamicSparseAttentionValidationWorkspaceSize(
            parameters, GetDeviceProp().sharedMemPerBlock);
    auto validation_workspace =
        GetScratchBuffer<uint32_t>(validation_workspace_elements, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(ValidateDynamicSparseAttentionOnDevice(
        stream, data.selected_indices, data.selected_counts, data.seqlens_k,
        data.position_ids, parameters, validation_error, validation_workspace.get(),
        GetDeviceProp().sharedMemPerBlock, strict_validation_, data.key_scale, data.value_scale,
        committed_length == nullptr ? nullptr : committed_length->Data<int32_t>()));
  }

  const size_t attention_workspace_elements =
      GetDynamicSparseAttentionWorkspaceSize(
          parameters, sizeof(CudaT), GetDeviceProp().sharedMemPerBlock, quantized_cache);
  auto attention_workspace =
      GetScratchBuffer<float>(attention_workspace_elements, GetComputeStream(context));
  data.attention_workspace = attention_workspace.get();

  const bool initialize_key_cache = data.past_key != data.present_key;
  const bool initialize_value_cache = data.past_value != data.present_value;
  return LaunchDynamicSparseAttention<CudaT>(
      stream, parameters, data, initialize_key_cache, initialize_value_cache,
      GetDeviceProp().maxThreadsPerBlock, GetDeviceProp().sharedMemPerBlock);
}

template class DynamicSparseAttention<float>;
template class DynamicSparseAttention<MLFloat16>;
template class DynamicSparseAttention<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
