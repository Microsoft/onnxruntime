// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};

  CUDAExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
  }

  CUDAExecutionProviderExternalAllocatorInfo(void* a, void* f) {
    alloc = a;
    free = f;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }

  bool operator==(const CUDAExecutionProviderExternalAllocatorInfo& other) const {
    return alloc == other.alloc && free == other.free;
  }
};

struct CUDAExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};                         // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search{OrtCudnnConvAlgoSearch::EXHAUSTIVE};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  // The following OrtArenaCfg instance only characterizes the behavior of the default memory
  // arena allocator and not any other auxiliary allocator that may also be part of the CUDA EP.
  // For example, auxiliary allocators `CUDA_PINNED` and `CUDA_CPU` will not be configured using this
  // arena config.
  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};

  static CUDAExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const CUDAExecutionProviderInfo& info);
};

struct CUDAExecutionProviderInfoHash {
  size_t operator()(const CUDAExecutionProviderInfo& info) const {
    return static_cast<size_t>(info.device_id) ^
           info.gpu_mem_limit ^
           (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
           (static_cast<size_t>(info.cudnn_conv_algo_search) << 18) ^
           (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
           (static_cast<size_t>(info.has_user_compute_stream) << 22);
  }
};

struct CUDAExecutionProviderInfoEqual {
  bool operator()(const CUDAExecutionProviderInfo& lhs, const CUDAExecutionProviderInfo& rhs) const {
    return lhs.device_id == rhs.device_id &&
           lhs.gpu_mem_limit == rhs.gpu_mem_limit &&
           lhs.arena_extend_strategy == rhs.arena_extend_strategy &&
           lhs.cudnn_conv_algo_search == rhs.cudnn_conv_algo_search &&
           lhs.do_copy_in_default_stream == rhs.do_copy_in_default_stream &&
           lhs.has_user_compute_stream == rhs.has_user_compute_stream &&
           lhs.user_compute_stream == rhs.user_compute_stream &&
           // lhs.default_memory_arena_cfg == rhs.default_memory_arena_cfg &&
           lhs.external_allocator_info == rhs.external_allocator_info;
  }
};

}  // namespace onnxruntime
