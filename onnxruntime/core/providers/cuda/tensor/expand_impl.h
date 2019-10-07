// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/data_types.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

Status ExpandImpl(
    const size_t element_size,
    const int32_t rank,
    const size_t N,
    const size_t N_input,
    const void* input_data,
    void* output_data,
    const TArray<fast_divmod>* fdm_input_dims,
    const TArray<fast_divmod>* fdm_output_dims,
    const TArray<fast_divmod>* fdm_output_subdim_size);

}  // namespace cuda
}  // namespace onnxruntime
