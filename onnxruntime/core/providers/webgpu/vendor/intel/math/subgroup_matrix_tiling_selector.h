// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// MatMul selector's SLM budget, in f16 elements (16384 == 32 KB). Shared with the
// Conv kernel's own split-K safety clamp.
constexpr uint32_t kMaxScratchElems = 16384;

// Tile/split-K selector for the common subgroup-matrix kernel. Empty on non-Intel
// devices or ones lacking the 8x16x16 F16 config.
SubgroupMatrixTilingSelector CreateSubgroupMatrixTilingSelector(
    const ComputeContextBase& context);

// Conv counterpart: Conv scratch budget (also stages the im2col A tile) and a
// Conv-only heuristic, no pretuned table.
SubgroupMatrixTilingSelector CreateSubgroupMatrixConvTilingSelector(
    const ComputeContextBase& context);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
