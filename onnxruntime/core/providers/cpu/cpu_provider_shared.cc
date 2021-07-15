// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CPU Provider functions used by shared providers

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cpu/math/cumsum.h"
#include "core/providers/cpu/math/einsum.h"
#include "core/providers/cpu/object_detection/non_max_suppression.h"
#include "core/providers/cpu/object_detection/roialign.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/providers/cpu/tensor/padbase.h"
#include "core/providers/cpu/tensor/scatter_nd.h"
#include "core/providers/cpu/tensor/split.h"
#include "core/providers/cpu/tensor/size.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/onehot.h"
#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cpu/tensor/gather_elements.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/cpu/bert/embed_layer_norm_helper.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"
#endif

#ifdef ENABLE_TRAINING
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"
#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "orttraining/training_ops/cpu/controlflow/record.h"
#include "orttraining/training_ops/cpu/controlflow/wait.h"
#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "orttraining/training_ops/cpu/tensor/split.h"
#endif

#include "cpu_provider_shared.h"

namespace onnxruntime {

struct ProviderHostCPUImpl : ProviderHostCPU {
  // From cpu/tensor/gatherbase.h (direct)
  Status GatherBase__PrepareForCompute(const GatherBase* p, OpKernelContext* context, GatherBase__Prepare& prepare) override { return p->GatherBase::PrepareForCompute(context, reinterpret_cast<GatherBase::Prepare&>(prepare)); }
  // From cpu/tensor/unsqueeze.h (direct)
  Status UnsqueezeBase__PrepareCompute(const UnsqueezeBase* p, OpKernelContext* ctx, UnsqueezeBase__Prepare& prepare) override { return p->UnsqueezeBase::PrepareCompute(ctx, reinterpret_cast<UnsqueezeBase::Prepare&>(prepare)); }

  // NonMaxSuppressionBase (direct)
  Status NonMaxSuppressionBase__PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) override { return NonMaxSuppressionBase::PrepareCompute(ctx, pc); }
  Status NonMaxSuppressionBase__GetThresholdsFromInputs(const PrepareContext& pc, int64_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold) override { return NonMaxSuppressionBase::GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold); }

#ifdef USE_CUDA
  // From cpu/tensor/size.h (direct)
  Status Size__Compute(const Size* p, OpKernelContext* context) override { return p->Size::Compute(context); }
  // From cpu/tensor/scatter_nd.h (direct)
  Status ScatterNDBase__ValidateShapes(const TensorShape& input_shape,
                                       const TensorShape& indice_shape,
                                       const TensorShape& update_shape) override { return ScatterNDBase::ValidateShapes(input_shape, indice_shape, update_shape); }
  // From cpu/tensor/padbase.h (direct)
  Status PadBase__HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, TensorShape& output_shape) override { return PadBase::HandleDimValueZero(mode, input_shape, output_shape); }
  // From cpu/tensor/split.h (direct)
  Status SplitBase__PrepareForCompute(const SplitBase* p, const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                      int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                      std::vector<int64_t>& split_sizes) override { return p->SplitBase::PrepareForCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }

  // From cpu/tensor/concatbase.h (direct)
  Status ConcatBase__PrepareForCompute(const ConcatBase* p, OpKernelContext* ctx, const std::vector<const Tensor*>& input_tensors, Prepare& prepare) override { return p->ConcatBase::PrepareForCompute(ctx, input_tensors, prepare); }

  // GatherElements (direct)
  Status GatherElements__ValidateInputShapes(const TensorShape& input_data_shape, const TensorShape& indices_shape, int64_t axis) override { return GatherElements::ValidateInputShapes(input_data_shape, indices_shape, axis); }

  // cumsum (direct)
  Status cumsum_op__GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) override { return cumsum_op::GetAxis(axis_tensor, input_rank, axis_out); }

  // TileOp (direct)
  bool TileOp__IsTileMemcpy(const TensorShape& input_shape, const int64_t* repeats, size_t rank, bool& is_batched_memcpy, size_t& num_of_elements_per_batch, size_t& num_of_copies_per_batch, size_t& num_of_batch_copies) override { return TileOp::IsTileMemcpy(input_shape, repeats, rank, is_batched_memcpy, num_of_elements_per_batch, num_of_copies_per_batch, num_of_batch_copies); }

  // ROI (direct)
  Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) override { return onnxruntime::CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr); }

  // From onehot.h (direct)
  Status ValidateInputs(const Tensor* depth, const Tensor* values) override { return onnxruntime::ValidateInputs(depth, values); }
  Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis, int64_t& prefix_dim_size, int64_t& suffix_dim_size, std::vector<int64_t>& output_shape) override { return onnxruntime::PrepareOutputShape(indices, depth_val, axis, prefix_dim_size, suffix_dim_size, output_shape); }

  // From cpu/tensor/slice.h (direct)
  Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      SliceOp__PrepareForComputeMetadata& compute_metadata) override { return SliceBase::PrepareForCompute(raw_starts, raw_ends, raw_axes, reinterpret_cast<SliceOp::PrepareForComputeMetadata&>(compute_metadata)); }

  Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      const std::vector<int64_t>& raw_steps,
                                      SliceOp__PrepareForComputeMetadata& compute_metadata) override { return SliceBase::PrepareForCompute(raw_starts, raw_ends, raw_axes, raw_steps, reinterpret_cast<SliceOp::PrepareForComputeMetadata&>(compute_metadata)); }

  Status SliceBase__FillVectorsFromInput(const Tensor& start_tensor,
                                         const Tensor& ends_tensor,
                                         const Tensor* axes_tensor,
                                         const Tensor* steps_tensor,
                                         std::vector<int64_t>& input_starts,
                                         std::vector<int64_t>& input_ends,
                                         std::vector<int64_t>& input_axes,
                                         std::vector<int64_t>& input_steps) override { return SliceBase::FillVectorsFromInput(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps); }

  // If (direct)
  void If__Init(If* p, const OpKernelInfo& info) override { p->If::Init(info); }
  Status If__Compute(const If* p, OpKernelContext* ctx) override { return p->If::Compute(ctx); }
  Status If__SetupSubgraphExecutionInfo(If* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->If::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  // Loop (direct)
  void Loop__Init(Loop* p, const OpKernelInfo& info) override { p->Loop::Init(info); }
  Status Loop__Compute(const Loop* p, OpKernelContext* ctx) override { return p->Loop::Compute(ctx); }
  Status Loop__SetupSubgraphExecutionInfo(Loop* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Loop::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  // Scan (direct)
  void Scan__Init(Scan<8>* p, const OpKernelInfo& info) override { p->Scan::Init(info); }
  void Scan__Init(Scan<9>* p, const OpKernelInfo& info) override { p->Scan::Init(info); }
  Status Scan__Compute(const Scan<8>* p, OpKernelContext* ctx) override { return p->Scan<8>::Compute(ctx); }
  Status Scan__Compute(const Scan<9>* p, OpKernelContext* ctx) override { return p->Scan<9>::Compute(ctx); }
  Status Scan__SetupSubgraphExecutionInfo(Scan<8>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Scan<8>::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }
  Status Scan__SetupSubgraphExecutionInfo(Scan<9>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Scan<9>::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  Status Einsum__Compute(const Einsum* p, OpKernelContext* context) override { return p->Einsum::Compute(context); }

  // EinsumComputePreprocessor (wrapped)
  void EinsumComputePreprocessor__operator_delete(EinsumComputePreprocessor* p) override { delete p; }
  std::unique_ptr<EinsumComputePreprocessor> EinsumComputePreprocessor__Create(EinsumEquationPreprocessor& equation_preprocessor,
                                                                               const std::vector<const Tensor*>& inputs,
                                                                               AllocatorPtr allocator,
                                                                               void* einsum_cuda_assets) override { return std::make_unique<EinsumComputePreprocessor>(equation_preprocessor, inputs, allocator, einsum_cuda_assets); }

  Status EinsumComputePreprocessor__Run(EinsumComputePreprocessor* p) override { return p->Run(); }
  void EinsumComputePreprocessor__SetDeviceHelpers(EinsumComputePreprocessor* p, const EinsumOp::DeviceHelpers::Diagonal& diagonal_func, const EinsumOp::DeviceHelpers::Transpose& transpose_func) override { return p->SetDeviceHelpers(diagonal_func, transpose_func); }

  // EinsumTypedComputeProcessor (wrapped)
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<float>* p) override { delete p; }
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<double>* p) override { delete p; }
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<MLFloat16>* p) override { delete p; }
  std::unique_ptr<EinsumTypedComputeProcessor<float>> EinsumTypedComputeProcessor_float__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<float>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  std::unique_ptr<EinsumTypedComputeProcessor<double>> EinsumTypedComputeProcessor_double__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<double>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  std::unique_ptr<EinsumTypedComputeProcessor<MLFloat16>> EinsumTypedComputeProcessor_MLFloat16__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<MLFloat16>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<float>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<float>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<float>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<double>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<double>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<double>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<MLFloat16>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<MLFloat16>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<MLFloat16>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<float>* p) override { return p->Run(); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<double>* p) override { return p->Run(); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<MLFloat16>* p) override { return p->Run(); }

#ifndef DISABLE_CONTRIB_OPS
  Status embed_layer_norm__CheckInputs(const OpKernelContext* context) override { return contrib::embed_layer_norm::CheckInputs(context); }
  Status bias_gelu_helper__CheckInputs(const OpKernelContext* context) override { return contrib::bias_gelu_helper::CheckInputs(context); }
  Status LongformerAttentionBase__CheckInputs(const contrib::LongformerAttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const TensorShape& mask_shape, const TensorShape& global_weights_shape, const TensorShape& global_bias_shape, const TensorShape& global_shape) override {
    return p->contrib::LongformerAttentionBase::CheckInputs(input_shape, weights_shape, bias_shape, mask_shape, global_weights_shape, global_bias_shape, global_shape);
  }
  Status AttentionBase__CheckInputs(const contrib::AttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const Tensor*& mask_index, const Tensor* past, const int max_threads_per_block) override { return p->contrib::AttentionBase::CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past, max_threads_per_block); }
  Tensor* AttentionBase__GetPresent(const contrib::AttentionBase* p, OpKernelContext* context, const Tensor* past, int batch_size, int head_size, int sequence_length, int& past_sequence_length) override { return p->contrib::AttentionBase::GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length); }
#endif

#ifdef ENABLE_TRAINING
  void ATenOpBase__Init(contrib::ATenOpBase* p, const OpKernelInfo& info, bool is_backward) override { return p->ATenOpBase::Init(info, is_backward); }
  Status ATenOpBase__Compute(const contrib::ATenOpBase* p, OpKernelContext* p_ctx) override { return p->ATenOpBase::Compute(p_ctx); }
  void contrib__record_event_in_tensor(const Tensor& event_id_tensor) override { return contrib::record_event_in_tensor(event_id_tensor); }
  void contrib__wait_event_in_tensor(const Tensor& event_id_tensor) override { return contrib::wait_event_in_tensor(event_id_tensor); }
  Status contrib__Group__Compute(const contrib::Group* p, OpKernelContext* context) override { return p->Group::Compute(context); }
  Status contrib__PassThrough__Compute(const contrib::PassThrough* p, OpKernelContext* context) override { return p->PassThrough::Compute(context); }
  void contrib__VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) override { contrib::VerifyLogitWeightAndLabelShape(logit_shape, label_shape, weight_shape); }
  void contrib__GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) override { contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C); }
  void contrib__GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, std::vector<int64_t>& new_shape, std::vector<size_t>& permutations) override { contrib::GetPermutationAndShape(ncd_to_ndc, tensor_shape, new_shape, permutations); }
  Status contrib__PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) override { return contrib::PrepareForTrainingCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }
  Status contrib__YieldOp__Compute(const contrib::YieldOp* p, OpKernelContext* context) override { return p->YieldOp::Compute(context); }
#endif
#endif
};

ProviderHostCPUImpl provider_host_cpu_;
ProviderHostCPU& GetProviderHostCPU() { return provider_host_cpu_; }

}  // namespace onnxruntime
