// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"
#include <unordered_map>

namespace onnxruntime {

using AllocatorFactory = std::function<std::unique_ptr<IAllocator>(OrtDevice::DeviceId)>;

using AllocatorMap = std::unordered_map<int, AllocatorPtr>;
// TODO: update OrtMemoryInfo, use unordered_set instead
using MemoryInfoSet = std::set<OrtMemoryInfo>;

const int DEFAULT_CPU_ALLOCATOR_DEVICE_ID = 0;

struct AllocatorCreationInfo {
  AllocatorCreationInfo(AllocatorFactory device_alloc_factory,
                        OrtDevice::DeviceId device_id = 0,
                        bool use_arena = true,
                        OrtArenaCfg arena_cfg = {0, -1, -1, -1, -1})
      : device_alloc_factory(device_alloc_factory),
        device_id(device_id),
        use_arena(use_arena),
        arena_cfg(arena_cfg) {
  }

  AllocatorFactory device_alloc_factory;
  OrtDevice::DeviceId device_id;
  bool use_arena;
  OrtArenaCfg arena_cfg;
};

// Returns an allocator (an instance of IAllocator) based on the creation info provided.
// Returns nullptr if an invalid value of info.arena_cfg.arena_extend_strategy is supplied.
// Valid values can be found in onnxruntime_c_api.h.
AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info);

// TODO: Only used for TRT and CUDA EP currently, need to add more identifiers to use it across all EPs
class AllocatorManager {
  //
 public:
  AllocatorManager() = default;
  void InsertAllocator(AllocatorPtr allocator);
  void ReplaceAllocator(AllocatorPtr allocator);
  //Get an allocator with specified device id and MemType. Return nullptr if it doesn't exist
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const;

 private:
  AllocatorMap allocators_;
  // to ensure only allocators with unique OrtMemoryInfo are registered in the provider.
  MemoryInfoSet mem_info_set_;

  // convenience list of the allocators so GetAllocatorList doesn't have to build a new vector each time
  // contains the same instances as allocators_
  std::vector<AllocatorPtr> allocator_list_;
};

}  // namespace onnxruntime
