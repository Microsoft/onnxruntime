// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <sstream>

namespace onnxruntime {

// Runtime statistics collected by an allocator.
struct AllocatorStats {
  int64_t num_allocs;             // Number of allocations.
  int64_t num_reserves;           // Number of reserves. (Number of calls to Reserve() in arena-based allocators)
  int64_t num_arena_extensions;   // Number of arena extensions (Relevant only for arena based allocators)
  int64_t num_arena_shrinkages;   // Number of arena shrinkages (Relevant only for arena based allocators)
  int64_t bytes_in_use;           // Number of bytes in use.
  int64_t total_allocated_bytes;  // The total number of allocated bytes by the allocator.
  int64_t max_bytes_in_use;       // The maximum bytes in use.
  int64_t max_alloc_size;         // The max single allocation seen.
                                  // The upper limit what the allocator can allocate, if such a limit
                                  // is known. Certain allocator may return 0 to indicate the limit is
                                  // unknown.
  int64_t bytes_limit;

  AllocatorStats() { Clear(); }

  void Clear() {
    this->num_allocs = 0;
    this->num_reserves = 0;
    this->num_arena_extensions = 0;
    this->num_arena_shrinkages = 0;
    this->bytes_in_use = 0;
    this->max_bytes_in_use = 0;
    this->max_alloc_size = 0;
    this->bytes_limit = 0;
    this->total_allocated_bytes = 0;
  }

  std::string DebugString() const {
    std::ostringstream ss;
    ss << "Limit:                    " << this->bytes_limit << "\n"
       << "InUse:                    " << this->bytes_in_use << "\n"
       << "TotalAllocated:           " << this->total_allocated_bytes << "\n"
       << "MaxInUse:                 " << this->max_bytes_in_use << "\n"
       << "NumAllocs:                " << this->num_allocs << "\n"
       << "NumReserves:              " << this->num_reserves << "\n"
       << "NumArenaExtensions:       " << this->num_arena_extensions << "\n"
       << "NumArenaShrinkages:       " << this->num_arena_shrinkages << "\n"
       << "MaxAllocSize:             " << this->max_alloc_size << "\n";
    return ss.str();
  }
};
}  // namespace onnxruntime