// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#ifdef ENABLE_TRAINING
#include "contrib_ops/cpu/aten_ops/aten_op.h"
#endif

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_VERSIONED_RANGE_TYPED(name, T, begin, end)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      begin, end,                                                                          \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

#define REGISTER_KERNEL_VERSIONED_SINCE_TYPED(name, T, version)                                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                  \
      name,                                                                                                                       \
      kOnnxDomain,                                                                                                                \
      version,                                                                                                                    \
      T,                                                                                                                          \
      kCudaExecutionProvider,                                                                                                     \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      name<T>);

#define REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(name, T, begin, end)                                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                                        \
      name,                                                                                                                       \
      kOnnxDomain,                                                                                                                \
      begin, end,                                                                                                                 \
      T,                                                                                                                          \
      kCudaExecutionProvider,                                                                                                     \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      name<T>);

#define REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(name, T, last, cur) \
  REGISTER_KERNEL_VERSIONED_RANGE_TYPED(name, T, 1, last)                   \
  REGISTER_KERNEL_VERSIONED_SINCE_TYPED(name, T, cur)

#define REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(name, T, last, cur, end, next) \
  REGISTER_KERNEL_VERSIONED_RANGE_TYPED(name, T, 1, last)                                  \
  REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(name, T, cur, end)                      \
  REGISTER_KERNEL_VERSIONED_SINCE_TYPED(name, T, next)

#define REGISTER_KERNEL_ARGMIN_OR_ARGMAX(name, T)        \
  REGISTER_KERNEL_VERSIONED_RANGE_TYPED(name, T, 1, 11)  \
  REGISTER_KERNEL_VERSIONED_RANGE_TYPED(name, T, 12, 12) \
  REGISTER_KERNEL_VERSIONED_SINCE_TYPED(name, T, 13)

// TODO ReduceKernel::ReduceKernelShared() is still used by some other training classes though it's not used here - this should be refactored.
template <bool allow_multi_axes>
template <typename T, typename OutT, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ReduceKernelShared(
    const T* X,
    const TensorShape& input_shape,
    OutT* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<OutT>::MappedType CudaOutT;
  cudnnDataType_t cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  const auto rank = input_shape.NumDimensions();

  auto cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
  // Block of fast matrix reduction.
  if (fast_reduction_) {
    int m{}, n{};
    const auto applicable_matrix_reduction = get_applicable_matrix_reduction(
        cudnn_reduce_op, input_shape.GetDims(), axes_, m, n);
    switch (applicable_matrix_reduction) {
      case ApplicableMatrixReduction::Rows: {
        return reduce_matrix_rows(
            cuda_stream,
            reinterpret_cast<const CudaT*>(X),
            reinterpret_cast<CudaOutT*>(Y),
            m, n, false);
      }
      case ApplicableMatrixReduction::Columns:
      // don't call reduce_matrix_columns() since it will reset initial output data
      default:
        break;
    }
  }

  int64_t input_count = input_shape.Size();
  IAllocatorUniquePtr<float> temp_X;
  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = GetScratchBuffer<float>(input_count, stream);
    cudnn_type_X = CUDNN_DATA_FLOAT;
    Impl_Cast<CudaT, float>(cuda_stream, reinterpret_cast<const CudaT*>(X), temp_X.get(), input_shape.Size());
  }

  // CUDNN requires at least 3D input, so pad 1s if needed
  auto input_dims_cudnn = input_shape.AsShapeVector();
  auto output_dims_cudnn = output_dims;
  if (rank < 3) {
    TensorShapeVector pads(3 - rank, 1);
    input_dims_cudnn.insert(input_dims_cudnn.end(), pads.begin(), pads.end());
    output_dims_cudnn.insert(output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  CudnnReduceDescriptor reduce_desc;
  if constexpr (std::is_same<T, MLFloat16>::value)
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, CudnnTensor::GetDataType<float>(), ReduceTensorIndices));
  else
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, ReduceTensorIndices));
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(cudnn_handle, reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes, stream);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_desc, input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes, stream);

  // need to allocate a separate buffer for ArgMin/ArgMax comparison output
  auto output_count = output_shape.Size();

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt_) {
      input_data_buffer = GetScratchBuffer<T>(input_count, stream);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(cuda_stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      reinterpret_cast<const CudaT*>(X), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp_) {
      // Reduce max -- Max/Min will output indices data
      CudnnReduceDescriptor reduce_max_desc;
      ORT_RETURN_IF_ERROR(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));
      size_t indices_bytes_max = 0;
      CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_max_desc, input_tensor, output_tensor, &indices_bytes_max));
      auto indices_cuda_max = GetScratchBuffer<uint32_t>(indices_bytes, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_max_desc, indices_cuda_max.get(), indices_bytes_max, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));

      // Exp(X-ReduceMax)
      const TensorShape rhs_shape(output_dims);
      auto exp_result_buffer = GetScratchBuffer<T>(input_count, stream);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = GetScratchBuffer<T>(output_count, stream);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, rhs_shape, input_shape));
      Impl_Sub<CudaT>(cuda_stream,
                      prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(X),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(Y),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // Fix NaN from exp(inf - inf): use original input to determine correct value.
      Impl_FixExpForReduceLogSumExp<CudaT>(cuda_stream, reinterpret_cast<const CudaT*>(X),
                                           reinterpret_cast<CudaT*>(exp_result), input_count);

      // ReduceSum
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, exp_result,
          &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));

      // Log(Sum)
      Impl_Log<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(cuda_stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(Y), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(Y), output_count);

      return Status::OK();
    }
    if (calculate_sqt_) {
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, input_data,
          &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
    } else {
      // cudnnReduceTensor has issues if input and output have the same size. This happens when the input is a
      // scalar or when a singleton axis (dim == 1) is reduced, making the reduction a no-op.
      if (input_count == output_count) {
        // For norm ops the result must still be non-negative (|x| for single-element no-op reductions).
        if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM1 || cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM2) {
          Impl_SaturatingAbs<CudaT>(cuda_stream, reinterpret_cast<const CudaT*>(X),
                                    reinterpret_cast<CudaT*>(Y), input_count);
        } else if (reinterpret_cast<const void*>(Y) != reinterpret_cast<const void*>(X)) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y, X, input_count * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream));
        }
      } else {
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(X),
            &zero, output_tensor, reinterpret_cast<CudaT*>(Y)));
      }
    }
  } else {  // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    if (temp_X) {
      auto temp_output = GetScratchBuffer<float>(output_count, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, temp_X.get(),
          &zero, output_tensor, temp_output.get()));
    } else {
      auto temp_output = GetScratchBuffer<CudaT>(output_count, stream);
      CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
          cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes, workspace_cuda.get(), workspace_bytes,
          &one, input_tensor, reinterpret_cast<const CudaT*>(X),
          &zero, output_tensor, temp_output.get()));
    }

    // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
    Impl_Cast<uint32_t, int64_t>(cuda_stream, reinterpret_cast<uint32_t*>(indices_cuda.get()), reinterpret_cast<int64_t*>(Y), output_count);
  }

  if (calculate_log_) {
    Impl_Log<CudaT>(cuda_stream, reinterpret_cast<CudaT*>(Y),
                    reinterpret_cast<CudaT*>(Y),
                    output_count);
  }

  return Status::OK();
}

template Status ReduceKernel<true>::ReduceKernelShared<double, double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const double* X,
    const TensorShape& input_shape,
    double* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<float, float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const float* X,
    const TensorShape& input_shape,
    float* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

template Status ReduceKernel<true>::ReduceKernelShared<MLFloat16, MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const MLFloat16* X,
    const TensorShape& input_shape,
    MLFloat16* Y,
    const TensorShape& output_shape,
    cudnnReduceTensorOp_t cudnn_reduce_op,
    cudnnHandle_t cudnn_handle,
    onnxruntime::Stream* stream,
    TensorShapeVector& output_dims) const;

// `input_shape_override` (if provided) is the input shape for compute purposes
Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        gsl::span<const int64_t> axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override) {
  ORT_ENFORCE(nullptr != X);

  const TensorShape& input_shape = input_shape_override ? *input_shape_override : X->Shape();
  const int64_t rank = gsl::narrow<int64_t>(input_shape.NumDimensions());
  prepare_reduce_metadata.input_count = input_shape.Size();

  const auto input_dims = input_shape.GetDims();
  std::vector<bool> reduced(rank, false);
  if (axes.size() > 0) {
    prepare_reduce_metadata.output_dims = input_shape.AsShapeVector();
    for (auto axis : axes) {
      axis = HandleNegativeAxis(axis, rank);
      ORT_ENFORCE(keepdims || input_dims[axis] != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      prepare_reduce_metadata.output_dims[axis] = 1;
      reduced[axis] = true;
    }
  } else {
    // no axes provided (i.e.) default axes  => reduce on all dims
    prepare_reduce_metadata.output_dims.reserve(input_dims.size());
    for (auto dim : input_dims) {
      ORT_ENFORCE(keepdims || dim != 0,
                  "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                  "Invalid output shape would be produced. input_shape:",
                  input_shape);
      prepare_reduce_metadata.output_dims.push_back(1);
    }
  }

  if (keepdims) {
    prepare_reduce_metadata.squeezed_output_dims = prepare_reduce_metadata.output_dims;
  } else if (axes.size() > 0) {
    // we are not going to keep the reduced dims, hence compute the final output dim accordingly
    prepare_reduce_metadata.squeezed_output_dims.reserve(rank);  // even though we won't use the full capacity, it is better to reserve for peak possible usage
    for (auto i = 0; i < rank; ++i) {
      if (!reduced[i])
        prepare_reduce_metadata.squeezed_output_dims.push_back(input_dims[i]);
    }
  } else {
    // 'axes' is empty and keepdims is false => we reduce on all axes AND drop all dims,
    // so the result is just a scalar, we keep 'squeezed_output_dims' empty (i.e.) no-op
  }

  // CUDNN requires at least 3D input, so pad 1s if needed
  prepare_reduce_metadata.input_dims_cudnn = input_shape.AsShapeVector();
  prepare_reduce_metadata.output_dims_cudnn = prepare_reduce_metadata.output_dims;
  if (rank < 3) {
    TensorShapeVector pads(3 - rank, 1);
    prepare_reduce_metadata.input_dims_cudnn.insert(prepare_reduce_metadata.input_dims_cudnn.end(), pads.begin(), pads.end());
    prepare_reduce_metadata.output_dims_cudnn.insert(prepare_reduce_metadata.output_dims_cudnn.end(), pads.begin(), pads.end());
  }

  prepare_reduce_metadata.output_count = TensorShape(prepare_reduce_metadata.output_dims).Size();

  return Status::OK();
}

// Unified scratch buffer allocation: when a CudaKernel pointer is available
// (plugin build path), use GetScratchBuffer which routes through the adapter
// allocator. Otherwise fall back to IAllocator::MakeUniquePtr (in-tree path).
template <typename T>
inline IAllocatorUniquePtr<T> AllocateScratchBuffer(
    const AllocatorPtr& gpu_allocator, const CudaKernel* kernel, size_t count, void* compute_stream) {
  if (count == 0) return nullptr;
  if (kernel) {
    return kernel->GetScratchBuffer<T>(count, compute_stream);
  }
  return IAllocator::MakeUniquePtr<T>(gpu_allocator, count, false, static_cast<onnxruntime::Stream*>(compute_stream));
}

// `input_shape_override` is the input shape for compute purposes (if provided)
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(const AllocatorPtr& gpu_allocator, const CudaKernel* kernel, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
                         gsl::span<const int64_t> axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         cudaStream_t cuda_stream, void* compute_stream, cudnnHandle_t cudnn_handle,
                         const TensorShape* input_shape_override) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const TensorShape& input_shape = input_shape_override ? *input_shape_override : input.Shape();
  cudaStream_t stream = cuda_stream;

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  auto& output_dims = prepare_reduce_metadata.output_dims;
  auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;
  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    if (output_count == 0) {
      return Status::OK();
    }
    // Empty-set reduction with non-empty output: fill with identity values per the ONNX spec.
    if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES) {
      // ArgMax/ArgMin: cannot select an index from an empty set (matches NumPy/PyTorch behavior).
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot perform ArgMax/ArgMin on an empty reduction axis. input_shape:", input_shape);
    }
    auto* output_data = reinterpret_cast<CudaT*>(output.MutableData<T>());
    if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MAX) {
      // Per ONNX spec: empty-set max yields -inf if supported, else lowest value.
      CudaT fill_val;
      if constexpr (std::numeric_limits<T>::has_infinity) {
        fill_val = ToCudaType<T>::FromFloat(-std::numeric_limits<float>::infinity());
      } else {
        // For integer types, CudaT == T.
        fill_val = static_cast<CudaT>(std::numeric_limits<T>::lowest());
      }
      Fill(stream, output_data, fill_val, output_count);
    } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MIN) {
      // Per ONNX spec: empty-set min yields +inf if supported, else max value.
      CudaT fill_val;
      if constexpr (std::numeric_limits<T>::has_infinity) {
        fill_val = ToCudaType<T>::FromFloat(std::numeric_limits<float>::infinity());
      } else {
        fill_val = static_cast<CudaT>(std::numeric_limits<T>::max());
      }
      Fill(stream, output_data, fill_val, output_count);
    } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MUL) {
      // Identity for product is 1
      Fill(stream, output_data, ToCudaType<T>::FromFloat(1.0f), output_count);
    } else {
      // Identity for sum, mean, norm1, norm2 is 0
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes(), stream));
    }
    return Status::OK();
  }

  // Block of fast matrix reduction.
  if (fast_reduction) {
    int m{}, n{};
    const auto applicable_matrix_reduction =
        get_applicable_matrix_reduction(cudnn_reduce_op, input_shape.GetDims(), axes, m, n);
    if (applicable_matrix_reduction != ApplicableMatrixReduction::None) {
      IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
      const CudaT* input_data = reinterpret_cast<const CudaT*>(input.Data<T>());
      if (calculate_sqt) {
        input_data_buffer = AllocateScratchBuffer<T>(gpu_allocator, kernel, input_count, compute_stream);
        input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
        fast_divmod tmp_div;
        Impl_Mul<CudaT>(stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                        reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                        reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr, tmp_div, tmp_div,
                        reinterpret_cast<CudaT*>(input_data_buffer.get()), input_count);
        input_data = reinterpret_cast<const CudaT*>(input_data_buffer.get());
      }

      switch (applicable_matrix_reduction) {
        case ApplicableMatrixReduction::Rows: {
          ORT_RETURN_IF_ERROR(reduce_matrix_rows(
              stream, input_data, reinterpret_cast<CudaT*>(output.MutableData<T>()), m, n));
        } break;
        case ApplicableMatrixReduction::Columns: {
          const auto buffer_size_bytes = compute_reduce_matrix_columns_buffer_size<CudaT>(m, n);
          auto buffer = buffer_size_bytes == 0
                            ? nullptr
                            : AllocateScratchBuffer<void>(gpu_allocator, kernel, buffer_size_bytes, compute_stream);
          ORT_RETURN_IF_ERROR(reduce_matrix_columns(stream, input_data,
                                                    reinterpret_cast<CudaT*>(output.MutableData<T>()), m, n,
                                                    buffer.get(), buffer_size_bytes));
        } break;
        default: {
          ORT_ENFORCE(false, "Invild matrix reduction type.");
        }
      }

      if (calculate_log) {
        Impl_Log<CudaT>(stream, reinterpret_cast<const CudaT*>(output.Data<T>()),
                        reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);
      } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_AVG) {
        float denominator_float = applicable_matrix_reduction == ApplicableMatrixReduction::Rows
                                      ? static_cast<float>(m)
                                      : static_cast<float>(n);
        CudaT denominator = ToCudaType<T>::FromFloat(denominator_float);
        UnaryDiv(stream, reinterpret_cast<const CudaT*>(output.Data<T>()),
                 reinterpret_cast<CudaT*>(output.MutableData<T>()), denominator, output_count);
      }

      return Status::OK();
    }
  }

  if constexpr (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES) {
    if (axes.size() == 1) {
      const int64_t rank = input_shape.NumDimensions();
      const int64_t axis = HandleNegativeAxis(axes[0], rank);
      if (axis == rank - 1) {
        const int64_t m = input_shape.SizeToDimension(axis);
        const int64_t n = input_shape[axis];
        if (n > 0 && m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
            (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MAX || cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MIN)) {
          const int rows = gsl::narrow_cast<int>(m);
          const int cols = gsl::narrow_cast<int>(n);
          // Wide rows are reduced by several cooperating blocks, which needs a small
          // buffer for the per block partial results.
          const auto buffer_size_bytes = compute_arg_min_max_last_axis_buffer_size<CudaT>(rows, cols);
          auto buffer = buffer_size_bytes == 0
                            ? nullptr
                            : AllocateScratchBuffer<void>(gpu_allocator, kernel, buffer_size_bytes, compute_stream);
          if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MAX) {
            return arg_min_max_last_axis<CudaT, true>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
                                                      output.MutableData<int64_t>(), rows, cols,
                                                      buffer.get(), buffer_size_bytes);
          }
          return arg_min_max_last_axis<CudaT, false>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
                                                     output.MutableData<int64_t>(), rows, cols,
                                                     buffer.get(), buffer_size_bytes);
        }
      }
    }
  }

  // Preserve the optimized matrix reductions above and the existing cuDNN path when available.
  // Without cuDNN, use a general CUDA kernel for plain ReduceSum axis layouts that cannot be
  // represented as a contiguous matrix reduction.
  if constexpr (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    if ((cudnn_handle == nullptr || input_shape.NumDimensions() > 8) &&
        cudnn_reduce_op == CUDNN_REDUCE_TENSOR_ADD &&
        !calculate_log && !calculate_sqt && !log_sum_exp) {
      return reduce_sum_nd(stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
                           reinterpret_cast<CudaT*>(output.MutableData<T>()), input_shape.GetDims(), axes);
    }
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes(), stream));

  IAllocatorUniquePtr<float> temp_X;
  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;

  // Reducesum with BFP16 is not supported by cudnn, so convert input to fp32 then call cudnn
  if ((ReduceTensorIndices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES && std::is_same<T, MLFloat16>::value) ||
      (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES && std::is_same<T, BFloat16>::value)) {
    // ArgMax/ArgMin with FP16 are not supported by cudnn, so convert input to fp32 then call cudnn
    temp_X = AllocateScratchBuffer<float>(gpu_allocator, kernel, input_count, compute_stream);
    Impl_Cast<CudaT, float>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()), temp_X.get(), input_shape.Size());
  } else {
    cudnn_type_X = CudnnTensor::GetDataType<CudaT>();
  }

  CudnnReduceDescriptor reduce_desc;
  if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, CudnnTensor::GetDataType<float>(), ReduceTensorIndices));
  } else {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, ReduceTensorIndices));
  }

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  ORT_ENFORCE(cudnn_handle != nullptr, "A cuDNN handle is required for CUDA reduction.");
  size_t workspace_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(cudnn_handle, reduce_desc,
                                                       input_tensor, output_tensor, &workspace_bytes));
  auto workspace_cuda = workspace_bytes == 0
                            ? nullptr
                            : AllocateScratchBuffer<void>(gpu_allocator, kernel, workspace_bytes, compute_stream);

  size_t indices_bytes = 0;
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_desc,
                                                     input_tensor, output_tensor, &indices_bytes));
  auto indices_cuda = indices_bytes == 0
                          ? nullptr
                          : AllocateScratchBuffer<void>(gpu_allocator, kernel, indices_bytes, compute_stream);

  if (ReduceTensorIndices == CUDNN_REDUCE_TENSOR_NO_INDICES) {
    IAllocatorUniquePtr<T> input_data_buffer(nullptr, [](T*) {});
    CudaT* input_data = nullptr;
    if (calculate_sqt) {
      input_data_buffer = AllocateScratchBuffer<T>(gpu_allocator, kernel, input_count, compute_stream);
      input_data = reinterpret_cast<CudaT*>(input_data_buffer.get());
      fast_divmod tmp_div;
      Impl_Mul<CudaT>(stream,
                      static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                      reinterpret_cast<const CudaT*>(input.Data<T>()), nullptr,
                      tmp_div, tmp_div,
                      input_data, input_count);
    } else if (log_sum_exp) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar
      if (input_count == output_count) {
        if (output.MutableData<T>() != input.Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableData<T>(), input.Data<T>(), input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
      } else {
        // Reduce max -- Max/Min will output indices data
        CudnnReduceDescriptor reduce_max_desc;
        cudnnDataType_t cudnn_reduce_max_type = cudnn_type_X;
        if ((std::is_same<T, MLFloat16>::value)) {
          cudnn_reduce_max_type = CUDNN_DATA_FLOAT;
        }
        ORT_RETURN_IF_ERROR(reduce_max_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_reduce_max_type, CUDNN_REDUCE_TENSOR_NO_INDICES));
        size_t indices_bytes_max = 0;
        CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(cudnn_handle, reduce_max_desc,
                                                           input_tensor, output_tensor, &indices_bytes_max));
        auto indices_cuda_max = indices_bytes_max == 0
                                    ? nullptr
                                    : AllocateScratchBuffer<void>(gpu_allocator, kernel, indices_bytes_max, compute_stream);
        auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_max_desc, indices_cuda_max.get(), indices_bytes_max,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
            &zero, output_tensor, p_output));
      }

      // Exp(X-ReduceMax)
      const TensorShape output_shape(output_dims);
      auto exp_result_buffer = AllocateScratchBuffer<T>(gpu_allocator, kernel, input_count, compute_stream);
      auto exp_result = exp_result_buffer.get();
      auto log_sum_result_buffer = output_count == 0
                                       ? nullptr
                                       : AllocateScratchBuffer<T>(gpu_allocator, kernel, output_count, compute_stream);
      auto log_sum_result = log_sum_result_buffer.get();
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(prepare.BinaryElementwiseBroadcastPrepareHelper(input_shape, output_shape, input_shape));
      Impl_Sub<CudaT>(stream,
                      prepare.output_rank_or_simple_broadcast,
                      &prepare.lhs_padded_strides,
                      reinterpret_cast<const CudaT*>(input.Data<T>()),
                      &prepare.rhs_padded_strides,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()),
                      &prepare.fdm_output_strides,
                      prepare.fdm_H, prepare.fdm_C,
                      reinterpret_cast<CudaT*>(exp_result), input_count);

      Impl_Exp<CudaT>(stream,
                      reinterpret_cast<CudaT*>(exp_result),
                      reinterpret_cast<CudaT*>(exp_result),
                      input_count);

      // Fix NaN from exp(inf - inf): use original input to determine correct value.
      Impl_FixExpForReduceLogSumExp<CudaT>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
                                           reinterpret_cast<CudaT*>(exp_result), input_count);

      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(log_sum_result), exp_result, input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        // ReduceSum
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, exp_result,
            &zero, output_tensor, reinterpret_cast<CudaT*>(log_sum_result)));
      }

      // Log(Sum)
      Impl_Log<CudaT>(stream, reinterpret_cast<CudaT*>(log_sum_result),
                      reinterpret_cast<CudaT*>(log_sum_result),
                      output_count);

      // Log + ReduceMax
      fast_divmod tmp_div;
      Impl_Add<CudaT>(stream, static_cast<int32_t>(SimpleBroadcast::NoBroadcast), nullptr,
                      reinterpret_cast<CudaT*>(log_sum_result), nullptr,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()), nullptr,
                      tmp_div, tmp_div,
                      reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);

      return Status::OK();
    }
    if (calculate_sqt) {
      // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
      // This happens when the input is Scalar. We do not need to add anything in this case.
      if (input_count == output_count) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(reinterpret_cast<CudaT*>(output.MutableData<T>()), input_data, input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      } else {
        auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, input_data,
            &zero, output_tensor, p_output));
      }
    } else {
      // cudnnReduceTensor has issues if input and output have the same size. This happens when the input is a
      // scalar or when a singleton axis (dim == 1) is reduced, making the reduction a no-op.
      if (input_count == output_count) {
        // For norm ops the result must still be non-negative (|x| for single-element no-op reductions).
        if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM1 || cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM2) {
          Impl_SaturatingAbs<CudaT>(stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
                                    reinterpret_cast<CudaT*>(output.MutableData<T>()), input_count);
        } else if (output.MutableData<T>() != input.Data<T>()) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableData<T>(), input.Data<T>(),
                                               input_count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
      } else {
        if (temp_X) {
          auto temp_output = output_count == 0
                                 ? nullptr
                                 : AllocateScratchBuffer<float>(gpu_allocator, kernel, output_count, compute_stream);
          CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
              cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
              workspace_cuda.get(), workspace_bytes,
              &one, input_tensor, temp_X.get(),
              &zero, output_tensor, temp_output.get()));

          Impl_Cast<float, CudaT>(stream, temp_output.get(), reinterpret_cast<CudaT*>(output.MutableData<T>()), output_count);
        } else {
          auto* p_output = reinterpret_cast<CudaT*>(output.template MutableData<T>());
          CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
              cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
              workspace_cuda.get(), workspace_bytes,
              &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
              &zero, output_tensor, p_output));
        }
      }
    }
  } else {
    // For ArgMax & ArgMin ops, use the indicies as the output with int64 type
    // cudnnReduceTensor has issue if input and output has same size, which will happen if the axis to be reduced has dim value of 1.
    // the output is zeros of the output size
    if (input_count == output_count) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableData<int64_t>(), static_cast<int64_t>(0), output_count * sizeof(int64_t), stream));
    } else {
      if (temp_X) {
        auto temp_output = output_count == 0
                               ? nullptr
                               : AllocateScratchBuffer<float>(gpu_allocator, kernel, output_count, compute_stream);
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, temp_X.get(),
            &zero, output_tensor, temp_output.get()));
      } else {
        auto temp_output = output_count == 0
                               ? nullptr
                               : AllocateScratchBuffer<CudaT>(gpu_allocator, kernel, output_count, compute_stream);
        CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
            cudnn_handle, reduce_desc, indices_cuda.get(), indices_bytes,
            workspace_cuda.get(), workspace_bytes,
            &one, input_tensor, reinterpret_cast<const CudaT*>(input.Data<T>()),
            &zero, output_tensor, temp_output.get()));
      }

      // CUDA reduction index is uint32_t for now, cast it to int64_t according to ONNX spec
      Impl_Cast<uint32_t, int64_t>(stream, reinterpret_cast<uint32_t*>(indices_cuda.get()),
                                   output.MutableData<int64_t>(), output_count);
    }
  }

  if (calculate_log) {
    Impl_Log<CudaT>(stream,
                    reinterpret_cast<CudaT*>(output.MutableData<T>()),
                    reinterpret_cast<CudaT*>(output.MutableData<T>()),
                    output_count);
  }

  return Status::OK();
}

template Status ReduceComputeCore<float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, const CudaKernel* kernel, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
    /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
    gsl::span<const int64_t> axes,
    bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
    cudaStream_t cuda_stream, void* compute_stream, cudnnHandle_t cudnn_handle,
    const TensorShape* input_shape_override);

template Status ReduceComputeCore<double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, const CudaKernel* kernel, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
    /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
    gsl::span<const int64_t> axes,
    bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
    cudaStream_t cuda_stream, void* compute_stream, cudnnHandle_t cudnn_handle,
    const TensorShape* input_shape_override);

template Status ReduceComputeCore<MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, const CudaKernel* kernel, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
    /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
    gsl::span<const int64_t> axes,
    bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
    cudaStream_t cuda_stream, void* compute_stream, cudnnHandle_t cudnn_handle,
    const TensorShape* input_shape_override);

template <bool allow_multi_axes>
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  TensorShapeVector axes;

  size_t num_inputs = ctx->InputCount();
  const Tensor* axes_tensor = num_inputs == 2 ? ctx->Input<Tensor>(1) : nullptr;  // optional input. may be nullptr.
  if (axes_tensor != nullptr) {
    // override the attribute value with the input value for reduction_axes
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(), X->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream(ctx)));
    return Status::OK();
  }

#ifdef ENABLE_TRAINING
  // Use ATen for ReduceSum if possible.
  const TensorShape& input_shape = X->Shape();
  if (contrib::IsATenOperatorExecutorInitialized() && cudnn_reduce_op == CUDNN_REDUCE_TENSOR_ADD && !calculate_log_ &&
      !calculate_sqt_ && !log_sum_exp_ && input_shape.Size() > 0) {
    if (axes.empty()) {
      axes.resize(input_shape.NumDimensions());
      std::iota(axes.begin(), axes.end(), 0);
    }
    ORT_RETURN_IF_ERROR(contrib::ExecuteReduceSumATen(ctx, axes, keepdims_));
    return Status::OK();
  }
#endif

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X, keepdims_, axes, prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();
  return ReduceComputeCore<T, ReduceTensorIndices>(AllocatorPtr{}, this, *X, prepare_reduce_metadata, *Y, cudnn_reduce_op, axes,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction,
                                                   Stream(ctx), GetComputeStream(ctx), TryGetCudnnHandle(ctx));
}

#define SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(T)                                                                           \
  template <>                                                                                                             \
  template <>                                                                                                             \
  Status ReduceKernel<true>::ComputeImpl<T, CUDNN_REDUCE_TENSOR_NO_INDICES>(                                              \
      OpKernelContext * ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {                                               \
    typedef typename ToCudaType<T>::MappedType CudaT;                                                                     \
    const Tensor* X = ctx->Input<Tensor>(0);                                                                              \
    TensorShapeVector axes;                                                                                               \
    size_t num_inputs = ctx->InputCount();                                                                                \
    const Tensor* axes_tensor = num_inputs == 2 ? ctx->Input<Tensor>(1) : nullptr;                                        \
    if (axes_tensor != nullptr) {                                                                                         \
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");                  \
      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);                                                          \
      const auto* data = axes_tensor->Data<int64_t>();                                                                    \
      axes.assign(data, data + nDims);                                                                                    \
    } else {                                                                                                              \
      axes.assign(axes_.begin(), axes_.end());                                                                            \
    }                                                                                                                     \
                                                                                                                          \
    if (axes.empty() && noop_with_empty_axes_) {                                                                          \
      auto* Y = ctx->Output(0, X->Shape());                                                                               \
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(), X->SizeInBytes(),                           \
                                           cudaMemcpyDeviceToDevice, Stream(ctx)));                                       \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    PrepareReduceMetadata prepare_reduce_metadata;                                                                        \
    ORT_RETURN_IF_ERROR(PrepareForReduce(X, keepdims_, axes, prepare_reduce_metadata));                                   \
                                                                                                                          \
    Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);                                             \
                                                                                                                          \
    int64_t input_count = prepare_reduce_metadata.input_count;                                                            \
    int64_t output_count = prepare_reduce_metadata.output_count;                                                          \
    auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;                                                    \
    auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;                                                  \
                                                                                                                          \
    if (input_count == 0) {                                                                                               \
      if (output_count == 0) {                                                                                            \
        return Status::OK();                                                                                              \
      }                                                                                                                   \
      /* Empty-set reduction with non-empty output: fill with identity values per ONNX spec. */                           \
      auto* output_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());                                                  \
      if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MAX) {                                                                   \
        CudaT fill_val = static_cast<CudaT>(std::numeric_limits<T>::lowest());                                            \
        Fill(Stream(ctx), output_data, fill_val, output_count);                                                           \
      } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MIN) {                                                            \
        CudaT fill_val = static_cast<CudaT>(std::numeric_limits<T>::max());                                               \
        Fill(Stream(ctx), output_data, fill_val, output_count);                                                           \
      } else if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_MUL) {                                                            \
        CudaT one_val = static_cast<CudaT>(1);                                                                            \
        Fill(Stream(ctx), output_data, one_val, output_count);                                                            \
      } else {                                                                                                            \
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(ctx)));                     \
      }                                                                                                                   \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    if (input_count == output_count) {                                                                                    \
      /* For no-op reductions where element count is unchanged, L1/L2 must still apply norm semantics. */                 \
      if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM1 || cudnn_reduce_op == CUDNN_REDUCE_TENSOR_NORM2) {                 \
        Impl_SaturatingAbs<CudaT>(Stream(ctx), reinterpret_cast<const CudaT*>(X->Data<T>()),                              \
                                  reinterpret_cast<CudaT*>(Y->MutableData<T>()), input_count);                            \
      } else if (Y->MutableData<T>() != X->Data<T>()) {                                                                   \
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(),                                           \
                                             input_count * sizeof(T), cudaMemcpyDeviceToDevice, Stream(ctx)));            \
      }                                                                                                                   \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
                                                                                                                          \
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {                                             \
      if (cudnn_reduce_op == CUDNN_REDUCE_TENSOR_ADD &&                                                                   \
          !calculate_log_ && !calculate_sqt_ && !log_sum_exp_) {                                                          \
        return reduce_sum_nd(Stream(ctx), reinterpret_cast<const CudaT*>(X->Data<T>()),                                   \
                             reinterpret_cast<CudaT*>(Y->MutableData<T>()), X->Shape().GetDims(), axes);                  \
      }                                                                                                                   \
    }                                                                                                                     \
                                                                                                                          \
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(ctx)));                         \
                                                                                                                          \
    size_t indices_bytes = 0;                                                                                             \
    size_t workspace_bytes = 0;                                                                                           \
    CudnnTensor input_tensor;                                                                                             \
    CudnnTensor output_tensor;                                                                                            \
    CudnnReduceDescriptor reduce_desc;                                                                                    \
                                                                                                                          \
    /* Use double for integer reductions to avoid integer overflow UB and improve precision.       */                     \
    /* double's 53-bit mantissa: int32 values are exactly representable, but their squares         */                     \
    /* (up to ~2^62) may exceed 53 bits and lose precision for large values. Still, double         */                     \
    /* provides vastly better precision than float (24-bit mantissa) for all integer types.        */                     \
    /* Note: cuDNN handles summation order internally, so Kahan compensation is not applicable.    */                     \
    /* For int64 values > 2^53, precision loss during accumulation is inherent to cuDNN's design.  */                     \
    /* The double accumulator cannot overflow: even INT64_MAX^2 * N needs > 10^270 elements.      */                      \
    cudnnDataType_t cudnn_type_X = CUDNN_DATA_DOUBLE;                                                                     \
    IAllocatorUniquePtr<double> temp_X = GetScratchBuffer<double>(input_count, GetComputeStream(ctx));                    \
    Impl_Cast<CudaT, double>(Stream(ctx), reinterpret_cast<const CudaT*>(X->Data<T>()), temp_X.get(), X->Shape().Size()); \
                                                                                                                          \
    ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));                  \
    ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));                                                \
    ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));                                              \
    CUDNN_RETURN_IF_ERROR(                                                                                                \
        cudnnGetReductionIndicesSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &indices_bytes));     \
    CUDNN_RETURN_IF_ERROR(                                                                                                \
        cudnnGetReductionWorkspaceSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &workspace_bytes)); \
    IAllocatorUniquePtr<void> indices_cuda = GetScratchBuffer<void>(indices_bytes, GetComputeStream(ctx));                \
    IAllocatorUniquePtr<void> workspace_cuda = GetScratchBuffer<void>(workspace_bytes, GetComputeStream(ctx));            \
                                                                                                                          \
    const auto one = Consts<double>::One;                                                                                 \
    const auto zero = Consts<double>::Zero;                                                                               \
    auto temp_Y = GetScratchBuffer<double>(output_count, GetComputeStream(ctx));                                          \
    CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(GetCudnnHandle(ctx), reduce_desc, indices_cuda.get(), indices_bytes,          \
                                            workspace_cuda.get(), workspace_bytes, &one, input_tensor, temp_X.get(),      \
                                            &zero, output_tensor, temp_Y.get()));                                         \
    /* Cast double result back to integer with explicit saturation (clamping to [T_MIN, T_MAX]).    */                    \
    /* A plain C++ cast from double to int is UB when out of range ([conv.fpint]/1). We use a     */                      \
    /* dedicated saturating kernel that clamps before casting, matching the CPU's explicit logic.  */                     \
    Impl_SaturatingCastFromDouble<CudaT>(Stream(ctx), temp_Y.get(),                                                       \
                                         reinterpret_cast<CudaT*>(Y->MutableData<T>()), output_count);                    \
                                                                                                                          \
    return Status::OK();                                                                                                  \
  }

// Integer reduction specializations.
// cuDNN does not natively support integer tensor reductions, so the strategy is:
//   1. Cast integer input → double (device kernel via Impl_Cast)
//   2. Perform cuDNN reduction in double precision (CUDNN_DATA_DOUBLE)
//   3. Cast double result → integer (device kernel via Impl_SaturatingCastFromDouble)
//
// Why double and not float?
//   - double avoids integer overflow UB and provides much better precision for all int types.
//   - int32 values are exactly representable in double; their squares (up to ~2^62) may lose
//     precision for |v| > 2^26, but this is at most 1 ULP error — far better than overflow.
//   - float (24-bit mantissa) loses precision for int32 values > 16M and any int64 values.
//   - For int64 values > 2^53, precision loss remains inherent (same limitation as CPU).
//   - Memory cost: 2× scratch buffer vs float — acceptable for integer reductions (rare).
//
// Saturation behavior:
//   - Impl_SaturatingCastFromDouble explicitly clamps to [T_MIN, T_MAX] before casting,
//     avoiding the undefined behavior of out-of-range double→int conversions in C++.
//   - This matches the CPU's explicit clamping logic in ReduceAggregatorL1/L2.
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int32_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int64_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(int8_t)
SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL(uint8_t)
namespace ReductionOps {

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
std::unique_ptr<Tensor> ReduceCompute(const AllocatorPtr& gpu_allocator, cudnnReduceTensorOp_t cudnn_reduce_op, AllocatorPtr allocator,
                                      const Tensor& input, gsl::span<const int64_t> axes,
                                      bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
                                      bool fast_reduction, Stream* stream, cudnnHandle_t cudnn_handle,
                                      const TensorShape* input_shape_override) {
  PrepareReduceMetadata prepare_reduce_metadata;
  auto status = PrepareForReduce(&input,
                                 keep_dims,
                                 axes,
                                 prepare_reduce_metadata,
                                 input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  auto output = Tensor::Create(input.DataType(), prepare_reduce_metadata.squeezed_output_dims, allocator);

  status = ReduceComputeCore<T, ReduceTensorIndices>(gpu_allocator, nullptr, input, prepare_reduce_metadata, *output, cudnn_reduce_op, axes,
                                                     calculate_log, calculate_sqt, log_sum_exp, fast_reduction,
                                                     stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr,
                                                     static_cast<void*>(stream), cudnn_handle,
                                                     input_shape_override);

  if (!status.IsOK()) {
    ORT_THROW(ONNXRUNTIME, FAIL, "Failed to perform reduce op: ", status.ErrorMessage());
  }

  return output;
}

// Explicit template instantiation (needed to be used in einsum_auxiliary_ops.cc)

template std::unique_ptr<Tensor> ReduceCompute<float, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, cudnnHandle_t cudnn_handle, const TensorShape* input_shape_override);

template std::unique_ptr<Tensor> ReduceCompute<double, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, cudnnHandle_t cudnn_handle, const TensorShape* input_shape_override);

template std::unique_ptr<Tensor> ReduceCompute<MLFloat16, CUDNN_REDUCE_TENSOR_NO_INDICES>(
    const AllocatorPtr& gpu_allocator, cudnnReduceTensorOp_t cudnn_reduce_op,
    AllocatorPtr allocator,
    const Tensor& input, gsl::span<const int64_t> axes,
    bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
    bool fast_reduction, Stream* stream, cudnnHandle_t cudnn_handle, const TensorShape* input_shape_override);

}  // namespace ReductionOps

REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMax, MLFloat16)
REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMax, float)
REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMax, double)

REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMin, MLFloat16)
REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMin, float)
REGISTER_KERNEL_ARGMIN_OR_ARGMAX(ArgMin, double)

REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, MLFloat16, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, float, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, double, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, int32_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, int64_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, int8_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMax, uint8_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, MLFloat16, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, float, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, double, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, int32_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, int64_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, int8_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMax, uint8_t, 18, 19)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, MLFloat16, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, float, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, double, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, int32_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, int64_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, int8_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMax, uint8_t, 20)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceMean, MLFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceMean, float, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceMean, double, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceMean, BFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceMean, int32_t, 17, 18)

REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, MLFloat16, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, float, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, double, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, int32_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, int64_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, int8_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_TYPED(ReduceMin, uint8_t, 1, 17)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, MLFloat16, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, float, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, double, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, int32_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, int64_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, int8_t, 18, 19)
REGISTER_KERNEL_VERSIONED_RANGE_AXES_INPUT_TYPED(ReduceMin, uint8_t, 18, 19)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, MLFloat16, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, float, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, double, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, int32_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, int64_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, int8_t, 20)
REGISTER_KERNEL_VERSIONED_SINCE_TYPED(ReduceMin, uint8_t, 20)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceProd, MLFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceProd, float, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceProd, double, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceProd, BFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceProd, int32_t, 17, 18)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, MLFloat16, 12, 13)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, float, 12, 13)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, double, 12, 13)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, int32_t, 12, 13)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, int64_t, 12, 13)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSum, BFloat16, 12, 13)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSum, MLFloat16, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSum, float, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSum, double, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSum, BFloat16, 17, 18, 27, 28)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSumSquare, MLFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSumSquare, float, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSumSquare, double, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceSumSquare, BFloat16, 17, 18)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSumExp, MLFloat16, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSumExp, float, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSumExp, double, 17, 18, 27, 28)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_TWO_VERSIONED(ReduceLogSumExp, BFloat16, 17, 18, 27, 28)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL1, MLFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL1, float, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL1, double, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL1, BFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL1, int32_t, 17, 18)

REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL2, MLFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL2, float, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL2, double, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL2, BFloat16, 17, 18)
REGISTER_KERNEL_TYPED_AXES_INPUT_WITH_VERSIONED(ReduceL2, int32_t, 17, 18)
}  // namespace cuda
}  // namespace onnxruntime
