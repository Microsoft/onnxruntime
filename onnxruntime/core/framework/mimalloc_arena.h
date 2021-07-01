// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "onnxruntime_config.h"

namespace onnxruntime {

class MiMallocArena : public IAllocator {
 public:
  MiMallocArena(std::unique_ptr<IAllocator> resource_allocator, size_t total_memory);

  void* Alloc(size_t size) override;

  void Free(void* p) override;

  // mimalloc only maintains stats when compiled under debug, or when MI_STAT >= 2
  void GetStats(AllocatorStats* stats) override;

  size_t AllocatedSize(const void* ptr);

 private:
  AllocatorStats stats_;
};

}  // namespace onnxruntime

#endif
