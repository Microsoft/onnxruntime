// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchSequencePoolingKernel(
    cudaStream_t stream,
    void* output,
    //void* masks,
    const void* input,
    const void* sentence_lengthes,
    const int batch_size,
    const int hidden_size,
    const int num_sequences,
    const int sequence_length_for_split,
    const size_t element_size
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
