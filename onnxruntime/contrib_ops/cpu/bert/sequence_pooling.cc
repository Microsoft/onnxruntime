// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_pooling.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SequencePooling,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SequencePooling<T>);

REGISTER_KERNEL_TYPED(float)

namespace {
template <typename T>
inline void MaxPoolingByRowImpl(T* start_dst, const T* start_src, const T* end_src) {
  while (start_src != end_src) {
    if (*start_src > *start_dst) {
      *start_dst = *start_src;
    }
    ++start_src;
    ++start_dst;
  }
}

template <typename T>
void MaxPoolingByRow(T* start_dst, const T* start_src, int64_t sentence_length, int hidden_size) {
  ORT_ENFORCE(sentence_length > 0);
  memcpy(start_dst, start_src, hidden_size * sizeof(T));
  for (int offset = 1; offset < sentence_length; ++offset) {
    start_src += hidden_size;
    MaxPoolingByRowImpl<T>(start_dst, start_src, start_src + hidden_size);
  }
}
} //namespace

template <typename T>
SequencePooling<T>::SequencePooling(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
}

template <typename T>
Status SequencePooling<T>::Compute(OpKernelContext* context) const {
  // get inputs tensors and data
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const T* input_data = input_tensor->template Data<T>();
  const Tensor* sentence_lengthes_tensor = context->Input<Tensor>(1);
  const int64_t* sentence_lengthes_data = sentence_lengthes_tensor->template Data<int64_t>();

  // shape info
  const auto& input_shape = input_tensor->Shape().GetDims();
  const int batch_size = static_cast<int>(input_shape[0]);
  const int sequence_length_for_split = static_cast<int>(input_shape[1]);
  const int hidden_size = static_cast<int>(input_shape[2]);
  const auto& sentence_lengthes_shape = sentence_lengthes_tensor->Shape().GetDims();
  const int num_sequences = static_cast<int>(sentence_lengthes_shape[1]);

  // check inputs
  std::vector<std::vector<int64_t>> sentence_lengthes_prefixsum;
  for (int batch = 0; batch < batch_size; ++batch) {
    std::vector<int64_t> sentence_length_prefixsum;
    sentence_length_prefixsum.resize(num_sequences);

    const std::ptrdiff_t offset(batch * num_sequences);
    std::partial_sum(sentence_lengthes_data + offset, sentence_lengthes_data + offset + num_sequences, sentence_length_prefixsum.begin());

    ORT_ENFORCE(sentence_length_prefixsum[num_sequences - 1] == sequence_length_for_split);

    sentence_lengthes_prefixsum.push_back(std::move(sentence_length_prefixsum));
  }

  // initialize outputs
  TensorShape output_shape({batch_size, num_sequences, hidden_size});
  Tensor* output_tensor(context->Output(0, output_shape));
  T* output_data = output_tensor->template MutableData<T>();

  Tensor* masks_tensor(context->Output(1, sentence_lengthes_tensor->Shape()));
  T* masks_data = masks_tensor->template MutableData<T>();

  // optional 1: row-based, parallel unfriendly(uneven distribution), but no need to transpose ahead for better locality
  auto* tp = context->GetOperatorThreadPool();
  const int loop_len = num_sequences;
  const double cost = static_cast<double>(sequence_length_for_split) * static_cast<double>(hidden_size) / static_cast<double>(num_sequences);
  for (int batch = 0; batch < batch_size; ++batch) {
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int64_t* sentence_length_data(sentence_lengthes_data + i);
        const int64_t sentence_length(*sentence_length_data);
        if (sentence_length == 0) {
          continue;
        }
        const std::ptrdiff_t past_sentence_length_sum = (i == 0) ? 0 : sentence_lengthes_prefixsum[batch][i - 1];
        const std::ptrdiff_t input_offset(batch * sequence_length_for_split * hidden_size + past_sentence_length_sum * hidden_size);
        const std::ptrdiff_t output_offset(batch * num_sequences * hidden_size + i * hidden_size);
        MaxPoolingByRow<T>(output_data + output_offset, input_data + input_offset, sentence_length, hidden_size);
      }
    });
  }

  for (int i = 0; i < batch_size * num_sequences; i++) {
    *masks_data++ = 1;
  }

  // optional 2: column-based, need to transpose first, easy to parallel(even distribution), especially in cuda
  // todo:

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
