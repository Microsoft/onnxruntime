// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/framework/cuda_provider_options.h"

namespace onnxruntime {
// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t cuda_mem_limit{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search{OrtCudnnConvAlgoSearch::EXHAUSTIVE};
  bool do_copy_in_default_stream{true};

  static CUDAExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const CUDAExecutionProviderInfo& info);
};
}  // namespace onnxruntime
