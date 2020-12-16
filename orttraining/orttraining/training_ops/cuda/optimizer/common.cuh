// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace onnxruntime {
namespace cuda {

// ---------------------------------------------------------------------------
// _ComputeGradScale -- helper to calculate gradient scales based on global norms
// ---------------------------------------------------------------------------

template<typename TLossScale, typename TGradNorm, typename TFinalScale>
__device__ __forceinline__ TFinalScale _ComputeGradScale(
const TLossScale* loss_scale,
const TGradNorm* g_norm) {
TFinalScale scale = loss_scale != nullptr ? TFinalScale(*loss_scale) : TFinalScale(1.f);
scale *= TFinalScale(0.1f);
// TGradNorm 
if (g_norm != nullptr && TFinalScale(*g_norm) > scale) {
    const TFinalScale actual_g_norm = TFinalScale(*g_norm) / scale;
    scale *= (actual_g_norm / TFinalScale(0.1f));
    // scale = gnorm;
} else {
    scale *= TFinalScale(10.0f); 
}
return scale;
}
}  // namespace cuda
}  // namespace onnxruntime