// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <limits>
#include <string>

#include "core/common/hash_combine.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderExternalAllocatorInfo {
  void* alloc{nullptr};
  void* free{nullptr};
  void* empty_cache{nullptr};

  CUDAExecutionProviderExternalAllocatorInfo() {
    alloc = nullptr;
    free = nullptr;
    empty_cache = nullptr;
  }

  CUDAExecutionProviderExternalAllocatorInfo(void* a, void* f, void* e) {
    alloc = a;
    free = f;
    empty_cache = e;
  }

  bool UseExternalAllocator() const {
    return (alloc != nullptr) && (free != nullptr);
  }
};

namespace cuda {
struct TunableOpInfo {
  bool enable{false};
  bool tuning_enable{false};
  int max_tuning_duration_ms{};
};
}  // namespace cuda

struct CUDAExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t gpu_mem_limit{std::numeric_limits<size_t>::max()};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  size_t gpu_device_mem_limit{0};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};  // Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search{OrtCudnnConvAlgoSearchExhaustive};
  bool do_copy_in_default_stream{true};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  // The following OrtArenaCfg instance only characterizes the behavior of the default memory
  // arena allocator and not any other auxiliary allocator that may also be part of the CUDA EP.
  // For example, auxiliary allocators `CUDA_PINNED` and `CUDA_CPU` will not be configured using this
  // arena config.
  OrtArenaCfg* default_memory_arena_cfg{nullptr};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};

  // By default, try to use as much as possible memory for algo search.
  // If set to false, use fix workspace size (32M) for Conv algo search, the final algo might not be the best.
  bool cudnn_conv_use_max_workspace{true};

  bool enable_cuda_graph{false};

  // By default, for Conv1D, will pad [N,C,D] to [N,C,D,1], if turn on, will pad to [N,C,1,D].
  bool cudnn_conv1d_pad_to_nc1d{false};

  bool enable_cudnn{true};

  cuda::TunableOpInfo tunable_op{};

  bool prefer_nhwc{false};

  bool use_ep_level_unified_stream{false};

  // By default, enable TF32 to speed up float GEMM/MatMul or cuDNN convolution of float matrices.
  bool use_tf32{true};
  bool fuse_conv_bias{false};

  int sdpa_kernel{0};

  // 0 disables the custom external-data loader and retains the framework's existing path.
  // 1 uses the pinned-buffer loader with synchronous reads. 2..64 use that many parallel read tasks per block.
  size_t external_data_loader_reading_threads{4};

  static CUDAExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const CUDAExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtCUDAProviderOptionsV2& info);
};
}  // namespace onnxruntime

template <>
struct std::hash<::onnxruntime::CUDAExecutionProviderInfo> {
  size_t operator()(const ::onnxruntime::CUDAExecutionProviderInfo& info) const {
    size_t value{0xbc9f1d34};  // seed

    // Bits: device_id (16), arena_extend_strategy/cudnn_conv_algo_search (reserved 2), boolean options (1 each)
    // Do not exceed 32 bits here otherwise some bits will be lost in x86.
    size_t data = static_cast<size_t>(info.device_id) ^
                  (static_cast<size_t>(info.arena_extend_strategy) << 16) ^
                  (static_cast<size_t>(info.cudnn_conv_algo_search) << 18) ^
                  (static_cast<size_t>(info.do_copy_in_default_stream) << 20) ^
                  (static_cast<size_t>(info.has_user_compute_stream) << 21) ^
                  (static_cast<size_t>(info.cudnn_conv_use_max_workspace) << 22) ^
                  (static_cast<size_t>(info.enable_cuda_graph) << 23) ^
                  (static_cast<size_t>(info.tunable_op.enable) << 24) ^
                  (static_cast<size_t>(info.tunable_op.tuning_enable) << 25) ^
                  (static_cast<size_t>(info.cudnn_conv1d_pad_to_nc1d) << 26) ^
                  (static_cast<size_t>(info.prefer_nhwc) << 28) ^
                  (static_cast<size_t>(info.use_ep_level_unified_stream) << 29) ^
                  (static_cast<size_t>(info.use_tf32) << 30) ^
                  (static_cast<size_t>(info.fuse_conv_bias) << 31);
    onnxruntime::HashCombine(data, value);

    onnxruntime::HashCombine(info.gpu_mem_limit, value);
    onnxruntime::HashCombine(info.gpu_device_mem_limit, value);
    onnxruntime::HashCombine(info.tunable_op.max_tuning_duration_ms, value);
    onnxruntime::HashCombine(info.sdpa_kernel, value);
    onnxruntime::HashCombine(info.enable_cudnn, value);
    onnxruntime::HashCombine(info.external_data_loader_reading_threads, value);

    // Memory pointers
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.user_compute_stream), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.alloc), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.free), value);
    onnxruntime::HashCombine(reinterpret_cast<size_t>(info.external_allocator_info.empty_cache), value);

    // The default memory arena cfg is not used in hashing right now.
    return value;
  }
};
