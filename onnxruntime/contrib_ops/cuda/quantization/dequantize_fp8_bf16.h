// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"

namespace onnxruntime::contrib::cuda {

void DequantizeFp8Bf16(cudaStream_t stream, const Float8E4M3FN* input, const float* scale,
                       const Float8E4M3FN* zero_point, BFloat16* output, size_t count,
                       size_t axis_stride, size_t scale_count);

}  // namespace onnxruntime::contrib::cuda
#endif
