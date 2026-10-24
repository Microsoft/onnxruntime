// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;
namespace psai = onnxruntime::contrib::packed_sparse_attention_indexer;

#define REGISTER_KERNEL_TYPED(K, T, TState)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      PackedSparseAttentionIndexer,                                         \
      kMSDomain,                                                            \
      1,                                                                    \
      K,                                                                    \
      kCudaExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T_KEY", DataTypeImpl::GetTensorType<TState>())   \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())      \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())      \
          .InputMemoryType(OrtMemTypeCPUInput, psai::kStateUpdateActive)    \
          .InputMemoryType(OrtMemTypeCPUInput, psai::kIndexShareMode)       \
          .MayInplace(psai::kMergedIndices, psai::kSelectedIndices)         \
          .MayInplace(psai::kMergedCounts, psai::kSelectedCounts)           \
          .MayInplace(psai::kMergedStatus, psai::kIndexShareStatus)         \
          .MayInplace(psai::kPastKeyState, psai::kPresentKeyState)          \
          .MayInplace(psai::kPastStateLengths, psai::kPresentStateLengths), \
      PackedSparseAttentionIndexer<T, TState>);

REGISTER_KERNEL_TYPED(float_float, float, float)
REGISTER_KERNEL_TYPED(MLFloat16_MLFloat16, MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16_BFloat16, BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(BFloat16_float, BFloat16, float)

#undef REGISTER_KERNEL_TYPED

namespace {

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "PackedSparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "PackedSparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

// The optional per-request state update status. When requested, it doubles as the update kernels' rejection flags.
Tensor* GetStateStatusOutput(OpKernelContext* context, int64_t batch_size) {
  return context->OutputCount() > psai::kStateStatus ? context->Output(psai::kStateStatus, TensorShape({batch_size}))
                                                     : nullptr;
}

Status CheckIntDimension(const char* name, int64_t value, bool allow_zero = true) {
  ORT_RETURN_IF(value < (allow_zero ? 0 : 1) || value > std::numeric_limits<int>::max(),
                "PackedSparseAttentionIndexer: ", name, " must be in ", allow_zero ? "[0, INT_MAX]" : "(0, INT_MAX]",
                ", got ", value);
  return Status::OK();
}

// cos_cache / sin_cache may be shared across the batch ([max_position, rotary_width]) or
// request-specific ([batch_size, max_position, rotary_width]).
struct RotaryCacheShape {
  bool batched;
  int64_t max_rotary_length;
  int64_t rotary_width;
};

Status CheckRotaryCache(const Tensor* cos_cache, const Tensor* sin_cache, int64_t batch_size,
                        RotaryCacheShape& out) {
  ORT_RETURN_IF(cos_cache == nullptr, "PackedSparseAttentionIndexer: cos_cache is required");
  const auto& cos_shape = cos_cache->Shape();
  out.batched = cos_shape.NumDimensions() == 3;
  ORT_RETURN_IF_NOT(
      (out.batched && cos_shape[0] == batch_size && cos_shape[1] > 0) ||
          (cos_shape.NumDimensions() == 2 && cos_shape[0] > 0),
      "PackedSparseAttentionIndexer: cos_cache must have shape (max_position, rotary_width) or "
      "(batch_size, max_position, rotary_width), got ",
      cos_shape.ToString());
  out.max_rotary_length = out.batched ? cos_shape[1] : cos_shape[0];
  out.rotary_width = out.batched ? cos_shape[2] : cos_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("max_rotary_sequence_length", out.max_rotary_length, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("rotary_width", out.rotary_width, false));
  ORT_RETURN_IF_NOT(sin_cache != nullptr && sin_cache->Shape() == cos_shape,
                    "PackedSparseAttentionIndexer: sin_cache must have the same shape as cos_cache");
  return Status::OK();
}

}  // namespace

template <typename T, typename TState>
PackedSparseAttentionIndexer<T, TState>::PackedSparseAttentionIndexer(const OpKernelInfo& info) : CudaKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "PackedSparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(psai::TryParsePolicy(policy_mode, policy_), "PackedSparseAttentionIndexer: policy_mode must be '",
              psai::kPolicyModeQsa, "' or '", psai::kPolicyModeCsa, "', got '", policy_mode, "'");
  if constexpr (!std::is_same_v<T, TState>) {
    ORT_ENFORCE(policy_ == psai::Policy::kQsa,
                "PackedSparseAttentionIndexer: mixed input/state types are supported only for policy_mode 'qsa'");
  }

  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "PackedSparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0 &&
                  compress_ratio_ <= (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2,
              "PackedSparseAttentionIndexer: compress_ratio must be positive and produce a generic buffer capacity "
              "no greater than INT_MAX, got ",
              compress_ratio_);

  ORT_ENFORCE(info.GetAttr<int64_t>("state_capacity", &state_capacity_).IsOK(),
              "PackedSparseAttentionIndexer: state_capacity is required");
  ORT_ENFORCE(state_capacity_ > 0 && state_capacity_ <= std::numeric_limits<int>::max(),
              "PackedSparseAttentionIndexer: state_capacity must be in (0, INT_MAX], got ", state_capacity_);
  state_update_capacity_ = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity_ >= 0 && state_update_capacity_ <= 8,
              "PackedSparseAttentionIndexer: state_update_capacity must be in [0, 8], got ",
              state_update_capacity_);

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  float head_weight_scale = 0.0f;
  has_head_weight_scale_ = info.GetAttr<float>("head_weight_scale", &head_weight_scale).IsOK();

  if (policy_ == psai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget,
                "PackedSparseAttentionIndexer: token_budget is required when policy_mode is 'qsa'");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale_,
                "PackedSparseAttentionIndexer: index_topk and head_weight_scale must not be set when policy_mode "
                "is 'qsa'");
    ORT_ENFORCE(token_budget_ > 0 && token_budget_ % compress_ratio_ == 0 &&
                    token_budget_ <= std::numeric_limits<int>::max() - compress_ratio_ + 1,
                "PackedSparseAttentionIndexer: token_budget must be > 0, divisible by compress_ratio, and produce "
                "a selected capacity no greater than INT_MAX, got token_budget=",
                token_budget_, " compress_ratio=", compress_ratio_);
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(state_update_capacity_ == 0,
                "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
    ORT_ENFORCE(has_index_topk, "PackedSparseAttentionIndexer: index_topk is required when policy_mode is 'csa'");
    ORT_ENFORCE(!has_token_budget,
                "PackedSparseAttentionIndexer: token_budget must not be set when policy_mode is 'csa'");
    ORT_ENFORCE(index_topk_ > 0 && index_topk_ <= std::numeric_limits<int>::max(),
                "PackedSparseAttentionIndexer: index_topk must be in (0, INT_MAX], got ", index_topk_);
    token_budget_ = 0;
  }

  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "PackedSparseAttentionIndexer: epsilon must be >= 0, got ", epsilon_);
  has_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  head_weight_scale_ = head_weight_scale;
  output_capacity_ = info.GetAttrOrDefault<int64_t>("max_output_entries", 0);
  const std::string state_policy = info.GetAttrOrDefault<std::string>("indexshare_state_policy", "default");
  ORT_ENFORCE(state_policy == "default" || state_policy == "hold" || state_policy == "advance",
              "indexshare_state_policy must be default, hold, or advance");
  advance_reuse_state_ = state_policy == "advance" || (state_policy == "default" && policy_ == psai::Policy::kCsa);
  ORT_ENFORCE(!advance_reuse_state_ || policy_ == psai::Policy::kCsa, "State-only advance requires CSA");
  ORT_ENFORCE(output_capacity_ == 0 ||
                  (output_capacity_ >= psai::SelectedCapacity(policy_, token_budget_, index_topk_, compress_ratio_) &&
                   output_capacity_ <= std::numeric_limits<int>::max()),
              "PackedSparseAttentionIndexer: max_output_entries must cover the original selection capacity");
}

template <typename T, typename TState>
Status PackedSparseAttentionIndexer<T, TState>::ComputeInternal(OpKernelContext* context) const {
  bool reuse = false;
  const Tensor* mode = context->InputCount() > psai::kIndexShareMode
                           ? context->Input<Tensor>(psai::kIndexShareMode)
                           : nullptr;
  if (mode != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(mode, "indexshare_mode", {1}));
    ORT_RETURN_IF(output_capacity_ == 0, "IndexShare requires max_output_entries");
    const int32_t value = mode->Data<int32_t>()[0];
    ORT_RETURN_IF(value != 0 && value != 1, "indexshare_mode must be 0 (refresh) or 1 (reuse)");
    for (int index : {psai::kMergedIndices, psai::kMergedCounts, psai::kMergedStatus}) {
      ORT_RETURN_IF(index >= context->InputCount() || context->Input<Tensor>(index) == nullptr,
                    "All IndexShare merge inputs are required");
    }
    const Tensor* merged_indices = context->Input<Tensor>(psai::kMergedIndices);
    ORT_RETURN_IF(merged_indices->Shape().NumDimensions() != 2 || merged_indices->Shape()[1] != output_capacity_,
                  "Merged indices must match max_output_entries");
    const int64_t merged_rows = merged_indices->Shape()[0];
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kMergedCounts), "merged_counts", {merged_rows}));
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kMergedStatus), "merged_status", {merged_rows}));
    reuse = value == 1;
    if (!reuse) {
      ORT_RETURN_IF(merged_rows != 0, "Refresh requires zero merge rows");
      const Tensor* query = context->Input<Tensor>(psai::kQuery);
      ORT_RETURN_IF(query == nullptr || query->Shape().NumDimensions() != 2, "Refresh requires rank-2 queries");
      Tensor* status = context->Output(psai::kIndexShareStatus, TensorShape({query->Shape()[0]}));
      ORT_RETURN_IF(status == nullptr, "IndexShare status output is required");
      if (status->SizeInBytes() != 0) {
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(status->MutableDataRaw(), 0, status->SizeInBytes(), Stream(context)));
      }
    }
  } else {
    ORT_RETURN_IF(output_capacity_ != 0, "max_output_entries requires the IndexShare interface");
    for (int index : {psai::kMergedIndices, psai::kMergedCounts, psai::kMergedStatus}) {
      ORT_RETURN_IF(index < context->InputCount() && context->Input<Tensor>(index) != nullptr,
                    "Merge inputs require indexshare_mode");
    }
  }
  const bool is_qsa = policy_ == psai::Policy::kQsa;
  constexpr int kCsaOnlyInputs[] = {psai::kGate, psai::kPositionBias, psai::kHeadWeights};
  for (int index : kCsaOnlyInputs) {
    const bool provided = index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
    ORT_RETURN_IF(provided != !is_qsa, "PackedSparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode 'qsa'" : " is required for policy_mode 'csa'");
  }
  const bool position_ids_provided =
      psai::kPositionIds < context->InputCount() && context->Input<Tensor>(psai::kPositionIds) != nullptr;
  ORT_RETURN_IF(position_ids_provided == is_qsa,
                "PackedSparseAttentionIndexer: position_ids is required for "
                "policy_mode 'csa' and must be omitted for policy_mode 'qsa'");
  const bool gate_buffer_provided =
      psai::kPastGateBuffer < context->InputCount() && context->Input<Tensor>(psai::kPastGateBuffer) != nullptr;
  ORT_RETURN_IF(gate_buffer_provided != !is_qsa, "PackedSparseAttentionIndexer: input ", psai::kPastGateBuffer,
                gate_buffer_provided ? " must be omitted for policy_mode 'qsa'"
                                     : " is required for policy_mode 'csa'");
  if (!is_qsa) {
    for (int index : {psai::kStateUpdateCaptureCount, psai::kStateUpdateActive}) {
      ORT_RETURN_IF(index < context->InputCount() && context->Input<Tensor>(index) != nullptr,
                    "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
    }
    Tensor* state_update = context->Output(psai::kStateUpdate, TensorShape({0}));
    ORT_RETURN_IF(state_update != nullptr,
                  "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
  }

  if (reuse) return ComputeReuse(context);
  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

template <typename T, typename TState>
Status PackedSparseAttentionIndexer<T, TState>::ComputeQsa(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  using CudaStateT = typename OrtToCudaType<TState>::type;

  const Tensor* query = context->Input<Tensor>(psai::kQuery);
  const Tensor* key = context->Input<Tensor>(psai::kKey);
  const Tensor* query_norm_weight = context->Input<Tensor>(psai::kQueryNormWeight);
  const Tensor* key_norm_weight = context->Input<Tensor>(psai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(psai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(psai::kSinCache);
  const Tensor* cumulative_sequence_lengths = context->Input<Tensor>(psai::kCumulativeSequenceLengths);
  const Tensor* past_sequence_lengths = context->Input<Tensor>(psai::kPastSequenceLengths);
  const Tensor* position_ids = context->Input<Tensor>(psai::kPositionIds);
  const Tensor* past_key_state = context->Input<Tensor>(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context->Input<Tensor>(psai::kPastKvBuffer);
  const Tensor* past_state_lengths = context->Input<Tensor>(psai::kPastStateLengths);
  const Tensor* state_update_capture_count =
      context->InputCount() > psai::kStateUpdateCaptureCount
          ? context->Input<Tensor>(psai::kStateUpdateCaptureCount)
          : nullptr;
  const Tensor* state_update_active =
      context->InputCount() > psai::kStateUpdateActive ? context->Input<Tensor>(psai::kStateUpdateActive) : nullptr;

  ORT_RETURN_IF(state_update_capacity_ > 0 && state_update_capture_count == nullptr,
                "PackedSparseAttentionIndexer: state_update_capture_count is required when "
                "state_update_capacity is positive");

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2,
                    "PackedSparseAttentionIndexer: query must have shape (total_tokens, num_heads * head_size), "
                    "got ",
                    query_shape.ToString());
  const int64_t total_tokens = query_shape[0];
  ORT_RETURN_IF(query_norm_weight == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm_weight->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1 && query_norm_shape[0] > 0,
                    "PackedSparseAttentionIndexer: query_norm_weight must have shape (head_size), got ",
                    query_norm_shape.ToString());
  const int64_t head_size = query_norm_shape[0];
  const bool has_separate_key = key != nullptr;
  const int64_t query_width = query_shape[1] - (has_separate_key ? 0 : head_size);
  ORT_RETURN_IF(query_width <= 0 || query_width % head_size != 0,
                "PackedSparseAttentionIndexer: query width must contain a positive multiple of head_size",
                has_separate_key ? "" : " followed by one packed key head");
  const int64_t num_heads = query_width / head_size;
  ORT_RETURN_IF_ERROR(CheckIntDimension("total_tokens", total_tokens));
  ORT_RETURN_IF_ERROR(CheckIntDimension("num_heads", num_heads, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("head_size", head_size, false));

  ORT_RETURN_IF(cumulative_sequence_lengths == nullptr,
                "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cumulative_sequence_lengths->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: cumulative_sequence_lengths must have shape (batch_size + 1), "
                    "got ",
                    cu_shape.ToString());
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF_ERROR(CheckIntDimension("batch_size", batch_size));
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_sequence_lengths, "past_sequence_lengths", {batch_size}));
  if (position_ids != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));
  }
  if (has_separate_key) {
    ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, head_size}));
  }
  ORT_RETURN_IF_ERROR(CheckShape(query_norm_weight, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(
      rotary.rotary_width > 0 && rotary.rotary_width % 2 == 0 && rotary.rotary_width <= head_size,
      "PackedSparseAttentionIndexer: policy_mode 'qsa' requires an even rotary_width in (0, head_size], got ",
      rotary.rotary_width);

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: past_key_state must have shape "
                    "(batch_size, state_capacity, head_size), got ",
                    key_state_shape.ToString());
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("state_capacity", state_capacity, false));
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));
  if (state_update_capture_count != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(state_update_capture_count, "state_update_capture_count", {batch_size}));
  }
  if (state_update_active != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(state_update_active, "state_update_active", {1}));
  }

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kQsa, token_budget_, index_topk_, compress_ratio_);

  Tensor* present_gate_buffer = context->Output(psai::kPresentGateBuffer, TensorShape({0}));
  ORT_RETURN_IF(present_gate_buffer != nullptr,
                "PackedSparseAttentionIndexer: output ", psai::kPresentGateBuffer,
                " must be omitted for policy_mode 'qsa'");
  const int64_t output_capacity = output_capacity_ == 0 ? capacity : output_capacity_;
  Tensor* selected_indices = context->Output(psai::kSelectedIndices, TensorShape({total_tokens, output_capacity}));
  Tensor* selected_counts = context->Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context->Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context->Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, head_size}));
  Tensor* present_state_lengths =
      context->Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  Tensor* state_update = context->OutputCount() > psai::kStateUpdate
                             ? context->Output(
                                   psai::kStateUpdate, TensorShape({batch_size, state_update_capacity_, head_size}))
                             : nullptr;
  Tensor* mid_kv_buffer =
      context->OutputCount() > psai::kMidKvBuffer
          ? context->Output(psai::kMidKvBuffer, TensorShape({batch_size, buffer_capacity, head_size}))
          : nullptr;
  Tensor* mid_state_lengths =
      context->OutputCount() > psai::kMidStateLengths
          ? context->Output(psai::kMidStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}))
          : nullptr;
  Tensor* validation_status =
      context->OutputCount() > psai::kValidationStatus
          ? context->Output(psai::kValidationStatus, TensorShape({batch_size}))
          : nullptr;
  ORT_RETURN_IF(selected_indices == nullptr || selected_counts == nullptr || present_key_state == nullptr ||
                    present_kv_buffer == nullptr || present_state_lengths == nullptr,
                "PackedSparseAttentionIndexer: policy_mode 'qsa' requires selected_indices, selected_counts, "
                "present_key_state, present_kv_buffer and present_state_lengths outputs");
  ORT_RETURN_IF((mid_kv_buffer == nullptr) != (mid_state_lengths == nullptr),
                "PackedSparseAttentionIndexer: mid_kv_buffer and mid_state_lengths must be connected together");
  if (state_update != nullptr && state_update->SizeInBytes() > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(state_update->MutableDataRaw(), 0, state_update->SizeInBytes(),
                                         Stream(context)));
  }

  PackedSparseAttentionIndexerParams params;
  Tensor* mid_buffers[2]{};
  Tensor* mid_lengths[2]{};
  for (int prefix = 0; prefix < 2; ++prefix) {
    const int buffer_index = psai::kMidKvBuffer2 + prefix * 2;
    const int length_index = buffer_index + 1;
    if (context->OutputCount() > buffer_index)
      mid_buffers[prefix] = context->Output(buffer_index, TensorShape({batch_size, buffer_capacity, head_size}));
    if (context->OutputCount() > length_index)
      mid_lengths[prefix] = context->Output(length_index, TensorShape({batch_size, psai::kStateLengthColumns}));
    ORT_RETURN_IF((mid_buffers[prefix] == nullptr) != (mid_lengths[prefix] == nullptr),
                  "PackedSparseAttentionIndexer: every prefix buffer requires its matching lengths");
  }
  params.mid_kv_buffer2 = mid_buffers[0] ? mid_buffers[0]->MutableData<T>() : nullptr;
  params.mid_state_lengths2 = mid_lengths[0] ? mid_lengths[0]->MutableData<int32_t>() : nullptr;
  params.mid_kv_buffer3 = mid_buffers[1] ? mid_buffers[1]->MutableData<T>() : nullptr;
  params.mid_state_lengths3 = mid_lengths[1] ? mid_lengths[1]->MutableData<int32_t>() : nullptr;
  params.batch_size = static_cast<int>(batch_size);
  params.total_tokens = static_cast<int>(total_tokens);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.query_row_stride = static_cast<int>(query_shape[1]);
  params.key_row_stride = static_cast<int>(has_separate_key ? head_size : query_shape[1]);
  params.rotary_width = static_cast<int>(rotary.rotary_width);
  params.max_rotary_length = static_cast<int>(rotary.max_rotary_length);
  params.cos_cache_batched = rotary.batched;
  params.has_position_ids = position_ids != nullptr;
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.state_capacity = static_cast<int>(state_capacity);
  params.buffer_capacity = static_cast<int>(buffer_capacity);
  params.capacity = static_cast<int>(capacity);
  params.output_capacity = static_cast<int>(output_capacity);
  params.epsilon = epsilon_;
  params.scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.block_topk = static_cast<int>(token_budget_ / compress_ratio_);
  params.state_update_capacity = static_cast<int>(state_update_capacity_);
  params.mid_kv_buffer =
      mid_kv_buffer == nullptr ? nullptr : reinterpret_cast<CudaT*>(mid_kv_buffer->MutableData<T>());
  params.mid_state_lengths =
      mid_state_lengths == nullptr ? nullptr : mid_state_lengths->MutableData<int32_t>();
  params.validation_status =
      validation_status == nullptr ? nullptr : validation_status->MutableData<int32_t>();

  auto float_workspace = GetScratchBuffer<float>(GetQsaPackedWorkspaceFloatCount(params), GetComputeStream(context));
  Tensor* state_status = GetStateStatusOutput(context, batch_size);
  auto overflow_scratch = GetScratchBuffer<int32_t>(
      state_status == nullptr ? static_cast<size_t>(std::max<int64_t>(batch_size, 1)) : 0, GetComputeStream(context));
  int32_t* overflow_flags = state_status != nullptr ? state_status->MutableData<int32_t>() : overflow_scratch.get();

  const CudaT* query_data = reinterpret_cast<const CudaT*>(query->Data<T>());
  const CudaT* key_data = has_separate_key
                              ? reinterpret_cast<const CudaT*>(key->Data<T>())
                              : query_data + query_width;
  return LaunchQsaPackedSparseAttentionIndexer<CudaT, CudaStateT>(
      Stream(context), params,
      query_data,
      key_data,
      reinterpret_cast<const CudaT*>(query_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      cumulative_sequence_lengths->Data<int32_t>(),
      past_sequence_lengths->Data<int32_t>(),
      position_ids != nullptr ? position_ids->Data<int64_t>() : nullptr,
      reinterpret_cast<const CudaStateT*>(past_key_state->Data<TState>()),
      reinterpret_cast<const CudaT*>(past_kv_buffer->Data<T>()),
      past_state_lengths->Data<int32_t>(),
      state_update_capture_count != nullptr ? state_update_capture_count->Data<int32_t>() : nullptr,
      state_update_active == nullptr || state_update_active->Data<int32_t>()[0] != 0,
      selected_indices->MutableData<int32_t>(),
      selected_counts->MutableData<int32_t>(),
      reinterpret_cast<CudaStateT*>(present_key_state->MutableData<TState>()),
      reinterpret_cast<CudaT*>(present_kv_buffer->MutableData<T>()),
      present_state_lengths->MutableData<int32_t>(),
      state_update != nullptr ? reinterpret_cast<CudaT*>(state_update->MutableData<T>()) : nullptr,
      float_workspace.get(),
      overflow_flags);
}

template <typename T, typename TState>
Status PackedSparseAttentionIndexer<T, TState>::ComputeCsa(OpKernelContext* context, const bool state_only) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* query = context->Input<Tensor>(psai::kQuery);
  const Tensor* key = context->Input<Tensor>(psai::kKey);
  const Tensor* query_norm_weight = context->Input<Tensor>(psai::kQueryNormWeight);
  const Tensor* key_norm_weight = context->Input<Tensor>(psai::kKeyNormWeight);
  const Tensor* cos_cache = context->Input<Tensor>(psai::kCosCache);
  const Tensor* sin_cache = context->Input<Tensor>(psai::kSinCache);
  const Tensor* cumulative_sequence_lengths = context->Input<Tensor>(psai::kCumulativeSequenceLengths);
  const Tensor* past_sequence_lengths = context->Input<Tensor>(psai::kPastSequenceLengths);
  const Tensor* gate = context->Input<Tensor>(psai::kGate);
  const Tensor* position_bias = context->Input<Tensor>(psai::kPositionBias);
  const Tensor* head_weights = context->Input<Tensor>(psai::kHeadWeights);
  const Tensor* position_ids = context->Input<Tensor>(psai::kPositionIds);
  const Tensor* past_key_state = context->Input<Tensor>(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context->Input<Tensor>(psai::kPastKvBuffer);
  const Tensor* past_gate_buffer = context->Input<Tensor>(psai::kPastGateBuffer);
  const Tensor* past_state_lengths = context->Input<Tensor>(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2,
                    "PackedSparseAttentionIndexer: query must have shape (total_tokens, num_heads * head_size), "
                    "got ",
                    query_shape.ToString());
  ORT_RETURN_IF(state_only && (key == nullptr || key->Shape().NumDimensions() != 2),
                "CSA state-only advance requires rank-2 keys");
  const int64_t total_tokens = state_only ? key->Shape()[0] : query_shape[0];
  ORT_RETURN_IF(query_norm_weight == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm_weight->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1,
                    "PackedSparseAttentionIndexer: query_norm_weight must have shape (head_size), got ",
                    query_norm_shape.ToString());
  const int64_t head_size = query_norm_shape[0];
  ORT_RETURN_IF(head_size <= 0 || query_shape[1] % head_size != 0,
                "PackedSparseAttentionIndexer: query width must be divisible by head_size");
  const int64_t num_heads = query_shape[1] / head_size;
  ORT_RETURN_IF_ERROR(CheckIntDimension("total_tokens", total_tokens));
  ORT_RETURN_IF_ERROR(CheckIntDimension("num_heads", num_heads, false));
  ORT_RETURN_IF_ERROR(CheckIntDimension("head_size", head_size, false));
  ORT_RETURN_IF(head_size > std::numeric_limits<int>::max() / 2,
                "PackedSparseAttentionIndexer: 2 * head_size must be no greater than INT_MAX");
  const int64_t width = 2 * head_size;

  ORT_RETURN_IF(cumulative_sequence_lengths == nullptr,
                "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cumulative_sequence_lengths->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: cumulative_sequence_lengths must have shape (batch_size + 1), "
                    "got ",
                    cu_shape.ToString());
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF_ERROR(CheckIntDimension("batch_size", batch_size));
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_sequence_lengths, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(query_norm_weight, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm_weight, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(position_bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {state_only ? 0 : total_tokens, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && 2 * rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: policy_mode 'csa' requires 0 < 2 * rotary_width <= head_size, "
                    "got rotary_width=",
                    rotary.rotary_width, " head_size=", head_size);

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: past_key_state must have shape "
                    "(batch_size, state_capacity, head_size), got ",
                    key_state_shape.ToString());
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_ERROR(CheckIntDimension("state_capacity", state_capacity, false));
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_gate_buffer, "past_gate_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kCsa, token_budget_, index_topk_, compress_ratio_);

  const int64_t output_capacity = output_capacity_ == 0 ? capacity : output_capacity_;
  Tensor* selected_indices = context->Output(psai::kSelectedIndices, TensorShape({total_tokens, output_capacity}));
  Tensor* selected_counts = context->Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context->Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context->Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_gate_buffer =
      context->Output(psai::kPresentGateBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_state_lengths =
      context->Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  ORT_RETURN_IF(selected_indices == nullptr || selected_counts == nullptr || present_key_state == nullptr ||
                    present_kv_buffer == nullptr || present_gate_buffer == nullptr ||
                    present_state_lengths == nullptr,
                "PackedSparseAttentionIndexer: policy_mode 'csa' requires selected_indices, selected_counts, "
                "present_key_state, present_kv_buffer, present_gate_buffer and present_state_lengths outputs");

  PackedSparseAttentionIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.total_tokens = static_cast<int>(total_tokens);
  params.num_heads = static_cast<int>(num_heads);
  params.head_size = static_cast<int>(head_size);
  params.query_row_stride = static_cast<int>(query_shape[1]);
  params.rotary_width = static_cast<int>(rotary.rotary_width);
  params.max_rotary_length = static_cast<int>(rotary.max_rotary_length);
  params.cos_cache_batched = rotary.batched;
  params.has_position_ids = true;
  params.compress_ratio = static_cast<int>(compress_ratio_);
  params.state_capacity = static_cast<int>(state_capacity);
  params.buffer_capacity = static_cast<int>(buffer_capacity);
  params.capacity = static_cast<int>(capacity);
  params.output_capacity = static_cast<int>(output_capacity);
  params.epsilon = epsilon_;
  params.scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
  params.index_topk = static_cast<int>(index_topk_);
  params.head_weight_scale =
      has_head_weight_scale_ ? head_weight_scale_ : 1.0f / std::sqrt(static_cast<float>(num_heads));
  params.state_only = state_only;

  auto float_workspace = GetScratchBuffer<float>(GetCsaPackedWorkspaceFloatCount(params), GetComputeStream(context));
  Tensor* state_status = GetStateStatusOutput(context, batch_size);
  auto overflow_scratch = GetScratchBuffer<int32_t>(
      state_status == nullptr ? static_cast<size_t>(std::max<int64_t>(batch_size, 1)) : 0, GetComputeStream(context));
  int32_t* overflow_flags = state_status != nullptr ? state_status->MutableData<int32_t>() : overflow_scratch.get();

  Tensor* reuse_status = nullptr;
  if (state_only) {
    reuse_status = context->Output(psai::kIndexShareStatus, TensorShape({batch_size}));
    ORT_RETURN_IF(reuse_status == nullptr, "IndexShare status output is required");
    for (const auto& pair : {std::pair{psai::kMergedIndices, selected_indices},
                             std::pair{psai::kMergedCounts, selected_counts},
                             std::pair{psai::kMergedStatus, reuse_status}}) {
      const Tensor* input = context->Input<Tensor>(pair.first);
      if (input->SizeInBytes() != 0 && input->DataRaw() != pair.second->MutableDataRaw()) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(pair.second->MutableDataRaw(), input->DataRaw(), input->SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream(context)));
      }
    }
    if (batch_size > 0) {
      CUDA_RETURN_IF_ERROR(
          cudaMemsetAsync(overflow_flags, 0, static_cast<size_t>(batch_size) * sizeof(int32_t), Stream(context)));
    }
    ORT_RETURN_IF_ERROR(LaunchValidatePackedIndexerReuse(
        Stream(context), cumulative_sequence_lengths->Data<int32_t>(), past_sequence_lengths->Data<int32_t>(),
        past_state_lengths->Data<int32_t>(), selected_indices->MutableData<int32_t>(),
        selected_counts->MutableData<int32_t>(), reuse_status->MutableData<int32_t>(),
        static_cast<int>(batch_size), static_cast<int>(output_capacity), static_cast<int>(state_capacity),
        static_cast<int>(compress_ratio_), false, overflow_flags));
  }

  ORT_RETURN_IF_ERROR(LaunchCsaPackedSparseAttentionIndexer<CudaT>(
      Stream(context), params,
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_weight->Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->Data<T>()),
      reinterpret_cast<const CudaT*>(gate->Data<T>()),
      reinterpret_cast<const CudaT*>(position_bias->Data<T>()),
      reinterpret_cast<const CudaT*>(head_weights->Data<T>()),
      cumulative_sequence_lengths->Data<int32_t>(),
      past_sequence_lengths->Data<int32_t>(),
      position_ids->Data<int64_t>(),
      reinterpret_cast<const CudaT*>(past_key_state->Data<T>()),
      reinterpret_cast<const CudaT*>(past_kv_buffer->Data<T>()),
      reinterpret_cast<const CudaT*>(past_gate_buffer->Data<T>()),
      past_state_lengths->Data<int32_t>(),
      selected_indices->MutableData<int32_t>(),
      selected_counts->MutableData<int32_t>(),
      reinterpret_cast<CudaT*>(present_key_state->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_kv_buffer->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_gate_buffer->MutableData<T>()),
      present_state_lengths->MutableData<int32_t>(),
      float_workspace.get(),
      overflow_flags));
  if (!state_only) return Status::OK();
  return LaunchValidatePackedIndexerReuse(Stream(context), cumulative_sequence_lengths->Data<int32_t>(),
                                          past_sequence_lengths->Data<int32_t>(), past_state_lengths->Data<int32_t>(),
                                          selected_indices->MutableData<int32_t>(), selected_counts->MutableData<int32_t>(),
                                          reuse_status->MutableData<int32_t>(), static_cast<int>(batch_size),
                                          static_cast<int>(output_capacity), static_cast<int>(state_capacity),
                                          static_cast<int>(compress_ratio_), false, overflow_flags);
}

template <typename T, typename TState>
Status PackedSparseAttentionIndexer<T, TState>::ComputeReuse(OpKernelContext* context) const {
  ORT_RETURN_IF(context->OutputCount() > psai::kMidKvBuffer2,
                "PackedSparseAttentionIndexer: additional prefix outputs require refresh, not IndexShare reuse");
  for (int index : {psai::kMidKvBuffer, psai::kMidStateLengths, psai::kValidationStatus}) {
    Tensor* output = context->OutputCount() > index ? context->Output(index, TensorShape({0})) : nullptr;
    ORT_RETURN_IF(output != nullptr,
                  "PackedSparseAttentionIndexer: outputs 9-11 are not supported in IndexShare reuse mode");
  }
  const Tensor* query = context->Input<Tensor>(psai::kQuery);
  const Tensor* lengths = context->Input<Tensor>(psai::kPastSequenceLengths);
  const Tensor* indices = context->Input<Tensor>(psai::kMergedIndices);
  const Tensor* counts = context->Input<Tensor>(psai::kMergedCounts);
  const Tensor* status = context->Input<Tensor>(psai::kMergedStatus);
  ORT_RETURN_IF(query == nullptr || query->Shape().NumDimensions() != 2 || query->Shape()[0] != 0,
                "Reuse requires zero projected query rows");
  ORT_RETURN_IF(lengths == nullptr || lengths->Shape().NumDimensions() != 1, "Reuse requires request lengths");
  const int64_t batch = lengths->Shape()[0];
  ORT_RETURN_IF_ERROR(CheckIntDimension("batch_size", batch));
  ORT_RETURN_IF_ERROR(CheckShape(indices, "merged_indices", {batch, output_capacity_}));
  ORT_RETURN_IF_ERROR(CheckShape(counts, "merged_counts", {batch}));
  ORT_RETURN_IF_ERROR(CheckShape(status, "merged_status", {batch}));
  ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kCumulativeSequenceLengths),
                                 "cumulative_sequence_lengths", {batch + 1}));
  const bool is_qsa = policy_ == psai::Policy::kQsa;
  const bool advance = !is_qsa && advance_reuse_state_;
  const Tensor* norm = context->Input<Tensor>(psai::kQueryNormWeight);
  ORT_RETURN_IF(norm == nullptr || norm->Shape().NumDimensions() != 1 || norm->Shape()[0] <= 0,
                "Reuse requires query norm weights");
  const int64_t head_size = norm->Shape()[0];
  ORT_RETURN_IF_ERROR(CheckIntDimension("head_size", head_size, false));
  ORT_RETURN_IF(!is_qsa && head_size > std::numeric_limits<int>::max() / 2, "CSA reuse width exceeds INT_MAX");
  const Tensor* key = context->Input<Tensor>(psai::kKey);
  const int64_t query_width = query->Shape()[1] - (is_qsa && key == nullptr ? head_size : 0);
  ORT_RETURN_IF(query_width <= 0 || query_width % head_size != 0, "Reuse query width must contain complete heads");
  const int64_t num_heads = query_width / head_size;
  ORT_RETURN_IF_ERROR(CheckIntDimension("num_heads", num_heads, false));
  ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kKeyNormWeight), "key_norm_weight", {head_size}));
  const int64_t width = is_qsa ? head_size : 2 * head_size;
  ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPastKeyState), "past_key_state", {batch, state_capacity_, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPastKvBuffer), "past_kv_buffer",
                                 {batch, psai::GenericBufferCapacity(compress_ratio_), width}));
  ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPastStateLengths), "past_state_lengths", {batch, 2}));
  if (!is_qsa) {
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPastGateBuffer), "past_gate_buffer",
                                   {batch, psai::GenericBufferCapacity(compress_ratio_), width}));
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kGate), "gate", {advance ? batch : 0, width}));
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPositionBias), "position_bias", {compress_ratio_, width}));
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kHeadWeights), "head_weights", {0, num_heads}));
    ORT_RETURN_IF_ERROR(CheckShape(context->Input<Tensor>(psai::kPositionIds), "position_ids", {advance ? batch : 0}));
  } else {
    Tensor* present_gate_buffer = context->Output(psai::kPresentGateBuffer, TensorShape({0}));
    ORT_RETURN_IF(present_gate_buffer != nullptr, "QSA reuse must omit present_gate_buffer");
    const Tensor* capture_count = context->Input<Tensor>(psai::kStateUpdateCaptureCount);
    ORT_RETURN_IF(state_update_capacity_ > 0 && capture_count == nullptr, "Reuse requires state_update_capture_count");
    if (capture_count != nullptr) {
      ORT_RETURN_IF_ERROR(CheckShape(capture_count, "state_update_capture_count", {batch}));
    }
    const Tensor* active = context->Input<Tensor>(psai::kStateUpdateActive);
    if (active != nullptr) {
      ORT_RETURN_IF_ERROR(CheckShape(active, "state_update_active", {1}));
    }
  }
  if (key != nullptr || !is_qsa) {
    ORT_RETURN_IF_ERROR(CheckShape(key, "key", {advance ? batch : 0, width}));
  }
  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(context->Input<Tensor>(psai::kCosCache),
                                       context->Input<Tensor>(psai::kSinCache), batch, rotary));
  ORT_RETURN_IF(rotary.rotary_width <= 0 ||
                    (is_qsa ? rotary.rotary_width % 2 != 0 || rotary.rotary_width > head_size
                            : rotary.rotary_width > head_size / 2),
                "Reuse rotary cache width is incompatible with head_size");
  if (advance) return ComputeCsa(context, true);
  for (const auto& pair : {std::pair{psai::kMergedIndices, psai::kSelectedIndices},
                           std::pair{psai::kMergedCounts, psai::kSelectedCounts},
                           std::pair{psai::kMergedStatus, psai::kIndexShareStatus},
                           std::pair{psai::kPastKeyState, psai::kPresentKeyState},
                           std::pair{psai::kPastKvBuffer, psai::kPresentKvBuffer},
                           std::pair{psai::kPastStateLengths, psai::kPresentStateLengths},
                           std::pair{psai::kPastGateBuffer, psai::kPresentGateBuffer}}) {
    if (is_qsa && pair.first == psai::kPastGateBuffer) continue;
    const Tensor* input = context->Input<Tensor>(pair.first);
    ORT_RETURN_IF(input == nullptr, "Reuse state and selection inputs are required");
    Tensor* output = context->Output(pair.second, input->Shape());
    ORT_RETURN_IF(output == nullptr, "Reuse state and selection outputs are required");
    if (input->SizeInBytes() != 0 && output->MutableDataRaw() != input->DataRaw()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output->MutableDataRaw(), input->DataRaw(), input->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream(context)));
    }
  }
  if (is_qsa) {
    const Tensor* qsa_norm = context->Input<Tensor>(psai::kQueryNormWeight);
    ORT_RETURN_IF(qsa_norm == nullptr || qsa_norm->Shape().NumDimensions() != 1, "Reuse requires query norm weights");
    Tensor* snapshot = context->Output(psai::kStateUpdate, {batch, state_update_capacity_, qsa_norm->Shape()[0]});
    if (snapshot != nullptr && snapshot->SizeInBytes() != 0) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(snapshot->MutableDataRaw(), 0, snapshot->SizeInBytes(), Stream(context)));
    }
  }
  // The state is held (not updated) on this path, so there is nothing to reject.
  Tensor* state_status = GetStateStatusOutput(context, batch);
  if (state_status != nullptr && state_status->SizeInBytes() != 0) {
    CUDA_RETURN_IF_ERROR(
        cudaMemsetAsync(state_status->MutableDataRaw(), 0, state_status->SizeInBytes(), Stream(context)));
  }
  return LaunchValidatePackedIndexerReuse(Stream(context),
                                          context->Input<Tensor>(psai::kCumulativeSequenceLengths)->Data<int32_t>(),
                                          lengths->Data<int32_t>(),
                                          context->Input<Tensor>(psai::kPastStateLengths)->Data<int32_t>(),
                                          context->Output(psai::kSelectedIndices, indices->Shape())->MutableData<int32_t>(),
                                          context->Output(psai::kSelectedCounts, counts->Shape())->MutableData<int32_t>(),
                                          context->Output(psai::kIndexShareStatus, status->Shape())->MutableData<int32_t>(),
                                          static_cast<int>(batch), static_cast<int>(output_capacity_),
                                          static_cast<int>(state_capacity_), static_cast<int>(compress_ratio_), is_qsa);
}

template class PackedSparseAttentionIndexer<float, float>;
template class PackedSparseAttentionIndexer<MLFloat16, MLFloat16>;
template class PackedSparseAttentionIndexer<BFloat16, BFloat16>;
template class PackedSparseAttentionIndexer<BFloat16, float>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
