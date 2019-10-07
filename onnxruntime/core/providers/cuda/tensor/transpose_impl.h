// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void TransposeImpl(int32_t rank, int64_t N,
                   const TArray<int64_t>& input_strides, const T* input_data,
                   const TArray<fast_divmod>& output_strides, T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
