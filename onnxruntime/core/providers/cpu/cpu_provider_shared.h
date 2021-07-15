// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

namespace contrib {
class LongformerAttentionBase;
class AttentionBase;
}  // namespace contrib

class GatherBase__Prepare;
class SliceOp__PrepareForComputeMetadata;  // Directly maps to SliceOp::PrepareForComputeMetadata
class UnsqueezeBase__Prepare;              // Directly maps to UnsqueezeBase::Prepare

struct ProviderHostCPU {

  // From cpu/tensor/gatherbase.h
  virtual Status GatherBase__PrepareForCompute(const GatherBase* p, OpKernelContext* context, GatherBase__Prepare& prepare) = 0;
  // From cpu/tensor/unsqueeze.h
  virtual Status UnsqueezeBase__PrepareCompute(const UnsqueezeBase* p, OpKernelContext* ctx, UnsqueezeBase__Prepare& prepare) = 0;

  // NonMaxSuppresionBase
  virtual Status NonMaxSuppressionBase__PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) = 0;
  virtual Status NonMaxSuppressionBase__GetThresholdsFromInputs(const PrepareContext& pc, int64_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold) = 0;

#ifdef USE_CUDA

  // From cpu/tensor/size.h
  virtual Status Size__Compute(const Size* p, OpKernelContext* context) = 0;
  // From cpu/tensor/scatter_nd.h
  virtual Status ScatterNDBase__ValidateShapes(const TensorShape& input_shape,
                                               const TensorShape& indice_shape,
                                               const TensorShape& update_shape) = 0;
  // From cpu/tensor/padbase.h
  virtual Status PadBase__HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, TensorShape& output_shape) = 0;
  // From cpu/tensor/split.h
  virtual Status SplitBase__PrepareForCompute(const SplitBase* p, const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                              int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                              std::vector<int64_t>& split_sizes) = 0;
  // From cpu/tensor/concatbase.h
  virtual Status ConcatBase__PrepareForCompute(const ConcatBase* p, OpKernelContext* ctx, const std::vector<const Tensor*>& input_tensors, Prepare& prepare) = 0;

  // GatherElements
  virtual Status GatherElements__ValidateInputShapes(const TensorShape& input_data_shape, const TensorShape& indices_shape, int64_t axis) = 0;
  // cumsum.cc
  virtual Status cumsum_op__GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) = 0;

  // TileOp
  virtual bool TileOp__IsTileMemcpy(const TensorShape& input_shape, const int64_t* repeats, size_t rank, bool& is_batched_memcpy, size_t& num_of_elements_per_batch, size_t& num_of_copies_per_batch, size_t& num_of_batch_copies) = 0;

  // ROI
  virtual Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) = 0;

  // From onehot.h
  virtual Status ValidateInputs(const Tensor* depth, const Tensor* values) = 0;
  virtual Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis, int64_t& prefix_dim_size, int64_t& suffix_dim_size, std::vector<int64_t>& output_shape) = 0;

  // From cpu/tensor/slice.h
  virtual Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                              const std::vector<int64_t>& raw_ends,
                                              const std::vector<int64_t>& raw_axes,
                                              SliceOp__PrepareForComputeMetadata& compute_metadata) = 0;

  virtual Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                              const std::vector<int64_t>& raw_ends,
                                              const std::vector<int64_t>& raw_axes,
                                              const std::vector<int64_t>& raw_steps,
                                              SliceOp__PrepareForComputeMetadata& compute_metadata) = 0;
  virtual Status SliceBase__FillVectorsFromInput(const Tensor& start_tensor,
                                                 const Tensor& ends_tensor,
                                                 const Tensor* axes_tensor,
                                                 const Tensor* steps_tensor,
                                                 std::vector<int64_t>& input_starts,
                                                 std::vector<int64_t>& input_ends,
                                                 std::vector<int64_t>& input_axes,
                                                 std::vector<int64_t>& input_steps) = 0;

  virtual Status Einsum__Compute(const Einsum* p, OpKernelContext* context) = 0;

  // EinsumComputePreprocessor
  virtual void EinsumComputePreprocessor__operator_delete(EinsumComputePreprocessor* p) = 0;
  virtual std::unique_ptr<EinsumComputePreprocessor> EinsumComputePreprocessor__Create(EinsumEquationPreprocessor& equation_preprocessor,
                                                                                       const std::vector<const Tensor*>& inputs,
                                                                                       AllocatorPtr allocator,
                                                                                       void* einsum_cuda_assets) = 0;

  virtual Status EinsumComputePreprocessor__Run(EinsumComputePreprocessor* p) = 0;
  virtual void EinsumComputePreprocessor__SetDeviceHelpers(EinsumComputePreprocessor* p, const EinsumOp::DeviceHelpers::Diagonal& diagonal_func, const EinsumOp::DeviceHelpers::Transpose& transpose_func) = 0;

  // EinsumTypedComputeProcessor
  virtual void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<float>* p) = 0;
  virtual void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<double>* p) = 0;
  virtual void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<MLFloat16>* p) = 0;
  virtual std::unique_ptr<EinsumTypedComputeProcessor<float>> EinsumTypedComputeProcessor_float__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) = 0;
  virtual std::unique_ptr<EinsumTypedComputeProcessor<double>> EinsumTypedComputeProcessor_double__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) = 0;
  virtual std::unique_ptr<EinsumTypedComputeProcessor<MLFloat16>> EinsumTypedComputeProcessor_MLFloat16__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) = 0;
  virtual void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<float>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<float>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<float>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) = 0;
  virtual void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<double>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<double>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<double>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) = 0;
  virtual void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<MLFloat16>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<MLFloat16>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<MLFloat16>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) = 0;
  virtual Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<float>* p) = 0;
  virtual Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<double>* p) = 0;
  virtual Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<MLFloat16>* p) = 0;

  // If
  virtual void If__Init(If* p, const OpKernelInfo& info) = 0;
  virtual Status If__Compute(const If* p, OpKernelContext* ctx) = 0;
  virtual Status If__SetupSubgraphExecutionInfo(If* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;

  // Loop
  virtual void Loop__Init(Loop* p, const OpKernelInfo& info) = 0;
  virtual Status Loop__Compute(const Loop* p, OpKernelContext* ctx) = 0;
  virtual Status Loop__SetupSubgraphExecutionInfo(Loop* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;

  // Scan
  virtual void Scan__Init(Scan<8>* p, const OpKernelInfo& info) = 0;
  virtual void Scan__Init(Scan<9>* p, const OpKernelInfo& info) = 0;
  virtual Status Scan__Compute(const Scan<8>* p, OpKernelContext* ctx) = 0;
  virtual Status Scan__Compute(const Scan<9>* p, OpKernelContext* ctx) = 0;
  virtual Status Scan__SetupSubgraphExecutionInfo(Scan<8>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;
  virtual Status Scan__SetupSubgraphExecutionInfo(Scan<9>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;

#ifndef DISABLE_CONTRIB_OPS
  virtual Status embed_layer_norm__CheckInputs(const OpKernelContext* context) = 0;
  virtual Status bias_gelu_helper__CheckInputs(const OpKernelContext* context) = 0;
  virtual Status LongformerAttentionBase__CheckInputs(const contrib::LongformerAttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const TensorShape& mask_shape, const TensorShape& global_weights_shape, const TensorShape& global_bias_shape, const TensorShape& global_shape) = 0;
  virtual Status AttentionBase__CheckInputs(const contrib::AttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const Tensor*& mask_index, const Tensor* past, const int max_threads_per_block) = 0;
  virtual Tensor* AttentionBase__GetPresent(const contrib::AttentionBase* p, OpKernelContext* context, const Tensor* past, int batch_size, int head_size, int sequence_length, int& past_sequence_length) = 0;
#endif

#ifdef ENABLE_TRAINING
  virtual void ATenOpBase__Init(contrib::ATenOpBase* p, const OpKernelInfo& info, bool is_backward) = 0;
  virtual Status ATenOpBase__Compute(const contrib::ATenOpBase* p, OpKernelContext* p_ctx) = 0;
  virtual void contrib__record_event_in_tensor(const Tensor& event_id_tensor) = 0;
  virtual void contrib__wait_event_in_tensor(const Tensor& event_id_tensor) = 0;
  virtual Status contrib__Group__Compute(const contrib::Group* p, OpKernelContext* context) = 0;
  virtual Status contrib__PassThrough__Compute(const contrib::PassThrough* p, OpKernelContext* context) = 0;
  virtual void contrib__VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) = 0;
  virtual void contrib__GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) = 0;
  virtual void contrib__GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, std::vector<int64_t>& new_shape, std::vector<size_t>& permutations) = 0;
  virtual Status contrib__PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) = 0;
  virtual Status contrib__YieldOp__Compute(const contrib::YieldOp* p, OpKernelContext* context) = 0;
#endif
#endif
};

#ifdef SHARED_PROVIDER

extern ProviderHostCPU& g_host_cpu;

#ifdef USE_CUDA
namespace GatherElements {
inline Status ValidateInputShapes(const TensorShape& input_data_shape,
                                  const TensorShape& indices_shape,
                                  int64_t axis) { return g_host_cpu.GatherElements__ValidateInputShapes(input_data_shape, indices_shape, axis); }
}  // namespace GatherElements

namespace cumsum_op {
inline Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) { return g_host_cpu.cumsum_op__GetAxis(axis_tensor, input_rank, axis_out); }
}  // namespace cumsum_op

inline Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) { return g_host_cpu.CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr); }

// From onehot.h
inline Status ValidateInputs(const Tensor* depth, const Tensor* values) { return g_host_cpu.ValidateInputs(depth, values); }
inline Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                                 int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                                 std::vector<int64_t>& output_shape) { return g_host_cpu.PrepareOutputShape(indices, depth_val, axis, prefix_dim_size, suffix_dim_size, output_shape); }

struct EinsumComputePreprocessor {
  static void operator delete(void* p) { g_host_cpu.EinsumComputePreprocessor__operator_delete(reinterpret_cast<EinsumComputePreprocessor*>(p)); }
  static std::unique_ptr<EinsumComputePreprocessor> Create(EinsumEquationPreprocessor& equation_preprocessor,
                                                           const std::vector<const Tensor*>& inputs,
                                                           AllocatorPtr allocator,
                                                           void* einsum_cuda_assets) { return g_host_cpu.EinsumComputePreprocessor__Create(equation_preprocessor, inputs, allocator, einsum_cuda_assets); }

  Status Run() { return g_host_cpu.EinsumComputePreprocessor__Run(this); }

  void SetDeviceHelpers(const EinsumOp::DeviceHelpers::Diagonal& diagonal_func, const EinsumOp::DeviceHelpers::Transpose& transpose_func) { return g_host_cpu.EinsumComputePreprocessor__SetDeviceHelpers(this, diagonal_func, transpose_func); }
};

template <typename T>
struct EinsumTypedComputeProcessor {
  static void operator delete(void* p) { g_host_cpu.EinsumTypedComputeProcessor__operator_delete(reinterpret_cast<EinsumTypedComputeProcessor*>(p)); }
  static std::unique_ptr<EinsumTypedComputeProcessor> Create(OpKernelContext* context, AllocatorPtr allocator,
                                                             concurrency::ThreadPool* tp,
                                                             EinsumComputePreprocessor& einsum_compute_preprocessor,
                                                             void* einsum_cuda_assets);

  void SetDeviceHelpers(const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
                        const EinsumOp::DeviceHelpers::MatMul<T>& device_matmul_func,
                        const EinsumOp::DeviceHelpers::ReduceSum<T>& device_reduce_sum_func,
                        const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) {
    g_host_cpu.EinsumTypedComputeProcessor__SetDeviceHelpers(this, device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func);
  }

  Status Run() { return g_host_cpu.EinsumTypedComputeProcessor__Run(this); }
};

#ifdef ENABLE_TRAINING
namespace contrib {
inline void record_event_in_tensor(const Tensor& event_id_tensor) { return g_host_cpu.contrib__record_event_in_tensor(event_id_tensor); }
inline void wait_event_in_tensor(const Tensor& event_id_tensor) { return g_host_cpu.contrib__wait_event_in_tensor(event_id_tensor); }

inline void VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) { g_host_cpu.contrib__VerifyLogitWeightAndLabelShape(logit_shape, label_shape, weight_shape); }
inline void GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) { g_host_cpu.contrib__GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C); }
inline void GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, std::vector<int64_t>& new_shape, std::vector<size_t>& permutations) { g_host_cpu.contrib__GetPermutationAndShape(ncd_to_ndc, tensor_shape, new_shape, permutations); }
inline Status PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) { return g_host_cpu.contrib__PrepareForTrainingCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }
}  // namespace contrib
#endif  // ENABLE_TRAINING
#endif  // USE_CUDA
#endif

}  // namespace onnxruntime
