// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_bias_gelu.h"

#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

namespace {

#pragma warning(disable : 4100)
Status CheckInputs(const Tensor* input,
                   const Tensor* skip,
                   const Tensor* gamma,
                   const Tensor* beta,
                   const Tensor* bias,
                   const Tensor* matmul_1_b,
                   const Tensor* bias_gelu_bias,
                   const Tensor* matmul_2_b,
                   const Tensor* output) {
  //
  // TODO(kreeger): write me
  //
  return Status::OK();
}

template <typename T>
void ComputeSkipLayerNorm(ptrdiff_t task_idx,
                          int64_t hidden_size,
                          float epsilon,
                          const T* input_data,
                          const T* skip_data,
                          const T* gamma_data,
                          const T* beta_data,
                          const T* bias_data,
                          T* output_data) {
  const T* cur_input = input_data + (task_idx * hidden_size);
  const T* cur_skip = skip_data + (task_idx * hidden_size);
  T* cur_output = output_data + (task_idx * hidden_size);

  T mean = 0;
  T mean_square = 0;

  for (int64_t i = 0; i < hidden_size; ++i) {
    T value = cur_input[i] + cur_skip[i];
    if (bias_data != nullptr) {
      value += bias_data[i];
    }

    cur_output[i] = value;
    mean += value;
    mean_square += value * value;
  }

  mean = mean / hidden_size;
  mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon);

  for (int64_t i = 0; i < hidden_size; ++i) {
    if (beta_data == nullptr) {
      cur_output[i] =
          (cur_output[i] - mean) / mean_square * gamma_data[i];
    } else {
      cur_output[i] =
          (cur_output[i] - mean) / mean_square * gamma_data[i] + beta_data[i];
    }
  }
}

template <typename T>
void ComputeMatMul() {
  //
  // TODO(kreeger): write me
  //
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormBiasGelu,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNormBiasGelu<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
EmbedLayerNormBiasGelu<T>::EmbedLayerNormBiasGelu(
    const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {}

template <typename T>
Status EmbedLayerNormBiasGelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* skip = context->Input<Tensor>(1);
  const Tensor* gamma = context->Input<Tensor>(2);
  const Tensor* beta = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* matmul_1_b = context->Input<Tensor>(5);
  const Tensor* bias_gelu_bias = context->Input<Tensor>(6);
  const Tensor* matmul_2_b = context->Input<Tensor>(7);

  Tensor* output = context->Output(0, input->Shape());

  ORT_RETURN_IF_ERROR(CheckInputs(input,
                                  skip,
                                  gamma,
                                  beta,
                                  bias,
                                  matmul_1_b,
                                  bias_gelu_bias,
                                  matmul_2_b,
                                  output));

  const auto& input_dims = input->Shape().GetDims();

  const int64_t batch_size = input_dims[0];
  const int64_t sequence_length = input_dims[1];
  const int64_t hidden_size = input_dims[2];

  const T* input_data = input->Data<T>();
  const T* skip_data = skip->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  // TODO(kreeger): add unit tests for this:
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* output_data = output->MutableData<T>();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  //
  // TODO(kreeger): Left off right here. I need a temp buffer to hold
  //                the placeholder of output?
  //

  //MatMulComputeHelper helper;
  //ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, trans_a, trans_b));

  // 

  //auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  //BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));


  int64_t task_count = batch_size * sequence_length;
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        // First, compute SkipLayerNorm:
        ComputeSkipLayerNorm(task_idx,
                             hidden_size,
                             epsilon(),
                             input_data,
                             skip_data,
                             gamma_data,
                             beta_data,
                             bias_data,
                             output_data);

        // Now perform MatMul
        //MLAS_SGEMM_DATA_PARAMS matmul_1_params;
        //matmul_1_params.A = cur_output;
        //matmul_1_params.lda = 0;  // first dim of cur_output?

        // Now perform BiasGelu

        // Now perform MatMul
      },
      0);

  return Status::OK();
}
#pragma warning(default : 4100)

}  // namespace contrib
}  // namespace onnxruntime
