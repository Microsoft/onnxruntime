// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "sequence_pooling_impl.h"
#include <cuda_fp16.h>
#include <stdio.h>

// An example
// In: input: [1, 4096, 768]
// In: sen_lens: [1, 47]     contains like [30, 40, 20, ....., 96] and sum up to 4096
// Out: output: [1, 256, 768]
//      where [0, 0:46, 768] is the max pooling result of input along axis=1 by sen_lens
//      and [0, 47:256, 768] part is all zeros


namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename InputIt, typename OutputIt>
__device__ inline void PrefixSumLinear(const InputIt* first, const InputIt* last, OutputIt* d_first) {
  if (first == last) {
    return;
  }
  OutputIt sum = *first;
  *d_first = sum;
  while (++first != last) {
    sum += *first;
    *(++d_first) = sum;
  }
}

template <typename T>
__global__ void SequencePoolingKernel(const T* input, const int64_t* sentence_lengthes, const int num_sequences, const int sequence_length_for_split, T* output) {

  const int hidden_size = gridDim.z;
  const int num_sequences_max = blockDim.x;
  const int batch_id = blockIdx.x;
  const int hidden_id = blockIdx.z;
  const int seq_id_per_batch = threadIdx.x;

  int sentence_lengthes_prefixsum[256]; // num_sequences <= 256

  const int offset = batch_id * num_sequences;
  const int num_sequences_limit = num_sequences < 256 ? num_sequences : 256;

  PrefixSumLinear(sentence_lengthes + offset, sentence_lengthes + offset + num_sequences_limit, sentence_lengthes_prefixsum);

  const int past_sequence_length = (seq_id_per_batch == 0) ? 0 : sentence_lengthes_prefixsum[seq_id_per_batch - 1];

  const int input_offset = batch_id * hidden_size * sequence_length_for_split + hidden_size * past_sequence_length + hidden_id;
  const int output_offset = batch_id * hidden_size * num_sequences_max + hidden_size * seq_id_per_batch + hidden_id;

  if (seq_id_per_batch >= num_sequences) {
    output[output_offset] = 0;
  } else {
    T local_max;
    const int sequence_length = sentence_lengthes_prefixsum[seq_id_per_batch] - past_sequence_length;

    for (int i = 0; i < sequence_length; ++i) {
      if (i == 0) {
        local_max = *(input + input_offset);
      } else {
        T value = *(input + input_offset + i * hidden_size);
        local_max = (float)value > (float)local_max ? value : local_max;
      }
    }

    output[output_offset] = local_max;
  }

}

template <typename T>
bool SequencePooling(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const T* input,
  const int64_t* sentence_lengthes,
  T* output) {
  const int num_sequences_max = 256;
  const dim3 grid(batch_size, 1, hidden_size);
  const dim3 block(num_sequences_max, 1, 1);

  SequencePoolingKernel<T><<<grid, block, 0, stream>>>(input, sentence_lengthes, num_sequences, sequence_length_for_split, output);
  cudaDeviceSynchronize();
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSequencePoolingKernel(
  cudaStream_t stream,
  void* output,
  const void* input,
  const void* sentence_lengthes,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const size_t element_size) {

  if (element_size == 2) {
    return SequencePooling<half>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      sequence_length_for_split,
      reinterpret_cast<const half*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<half*>(output)
    );
  } else {
    return SequencePooling<float>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      sequence_length_for_split,
      reinterpret_cast<const float*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<float*>(output)
    );
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
