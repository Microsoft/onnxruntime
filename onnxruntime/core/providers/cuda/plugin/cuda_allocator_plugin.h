// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUDA device and pinned memory allocator implementations for the plugin EP.
// Provides CudaDeviceAllocator (cudaMalloc/cudaFree) and CudaPinnedAllocator
// (cudaHostAlloc/cudaFreeHost) conforming to the OrtAllocator interface.
// No arena or caching layer; every allocation goes directly to CUDA.

#pragma once

#include "cuda_plugin_utils.h"
#include "core/providers/cuda/cuda_allocator.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace onnxruntime {
namespace cuda_plugin {

/// Allocator type: device memory (GPU) or pinned (page-locked host) memory.
enum class CudaAllocatorKind {
  kDevice,  ///< GPU device memory via cudaMalloc
  kPinned,  ///< Page-locked host memory via cudaHostAlloc
};

/// Base class for CUDA allocators implementing the OrtAllocator C interface.
class CudaAllocatorBase : public OrtAllocator {
 public:
  explicit CudaAllocatorBase(CudaAllocatorKind kind, const OrtMemoryInfo* memory_info,
                             bool is_external_device_allocator = false)
      : OrtAllocator{},
        kind_(kind),
        memory_info_(memory_info),
        is_external_device_allocator_(is_external_device_allocator) {}

  CudaAllocatorKind GetKind() const { return kind_; }
  const OrtMemoryInfo* GetMemoryInfo() const { return memory_info_; }
  bool IsExternalDeviceAllocator() const { return is_external_device_allocator_; }

 private:
  CudaAllocatorKind kind_;
  const OrtMemoryInfo* memory_info_;
  bool is_external_device_allocator_;
};

// CudaAllocatorBase derives from OrtAllocator via single non-virtual inheritance.
// This guarantees OrtAllocator sits at offset 0 in the derived layout, so
// static_cast between OrtAllocator* and CudaAllocatorBase* is safe.
static_assert(!std::is_polymorphic_v<CudaAllocatorBase>,
              "CudaAllocatorBase must not be polymorphic (no virtual functions) "
              "to ensure OrtAllocator is at offset 0.");

/// Allocator statistics tracked by arena allocators.
struct AllocatorStats {
  int64_t num_allocs = 0;              // Number of allocations.
  int64_t num_reserves = 0;            // Number of reserves. (Number of calls to Reserve() in arena-based allocators)
  int64_t num_arena_extensions = 0;    // Number of arena extensions (Relevant only for arena based allocators)
  int64_t num_arena_shrinkages = 0;    // Number of arena shrinkages (Relevant only for arena based allocators)
  int64_t bytes_in_use = 0;            // Number of bytes in use (includes padding).
  int64_t bytes_requested_in_use = 0;  // Number of bytes actually requested by user code (excludes padding).
  int64_t total_allocated_bytes = 0;   // The total number of allocated bytes by the allocator.
  int64_t max_bytes_in_use = 0;        // The maximum bytes in use.
  int64_t max_alloc_size = 0;          // The max single allocation seen.
  int64_t bytes_limit = 0;             // The upper limit what the allocator can allocate (0 if unknown).

  void ToKeyValuePairs(const OrtApi& api, OrtKeyValuePairs* kvps) const {
    api.AddKeyValuePair(kvps, "Limit", std::to_string(bytes_limit).c_str());
    api.AddKeyValuePair(kvps, "InUse", std::to_string(bytes_in_use).c_str());
    api.AddKeyValuePair(kvps, "RequestedInUse", std::to_string(bytes_requested_in_use).c_str());
    api.AddKeyValuePair(kvps, "TotalAllocated", std::to_string(total_allocated_bytes).c_str());
    api.AddKeyValuePair(kvps, "MaxInUse", std::to_string(max_bytes_in_use).c_str());
    api.AddKeyValuePair(kvps, "NumAllocs", std::to_string(num_allocs).c_str());
    api.AddKeyValuePair(kvps, "NumReserves", std::to_string(num_reserves).c_str());
    api.AddKeyValuePair(kvps, "NumArenaExtensions", std::to_string(num_arena_extensions).c_str());
    api.AddKeyValuePair(kvps, "NumArenaShrinkages", std::to_string(num_arena_shrinkages).c_str());
    api.AddKeyValuePair(kvps, "MaxAllocSize", std::to_string(max_alloc_size).c_str());
  }

  std::string DebugString() const {
    std::ostringstream ss;
    ss << "Limit:                    " << bytes_limit << "\n"
       << "InUse:                    " << bytes_in_use << "\n"
       << "RequestedInUse:           " << bytes_requested_in_use << "\n"
       << "TotalAllocated:           " << total_allocated_bytes << "\n"
       << "MaxInUse:                 " << max_bytes_in_use << "\n"
       << "NumAllocs:                " << num_allocs << "\n"
       << "NumReserves:              " << num_reserves << "\n"
       << "NumArenaExtensions:       " << num_arena_extensions << "\n"
       << "NumArenaShrinkages:       " << num_arena_shrinkages << "\n"
       << "MaxAllocSize:             " << max_alloc_size << "\n";
    return ss.str();
  }
};

/// CUDA device memory allocator using cudaMalloc/cudaFree.
/// Lifetime is managed by the EP factory (ReleaseAllocatorImpl), not by a Release callback.
class CudaDeviceAllocator final : public CudaAllocatorBase {
 public:
  CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id, const OrtApi& api,
                      std::shared_ptr<SessionBudgetAccount> account = nullptr);
  ~CudaDeviceAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_ptr, OrtKeyValuePairs** stats) noexcept;

  int device_id_;
  const OrtApi& api_;
  CUDAAllocator allocator_;
};

/// CUDA device memory allocator using external user-provided function pointers.
/// Delegates alloc/free/empty_cache to the caller-supplied callbacks.
/// Lifetime is managed by the EP factory (ReleaseAllocatorImpl), not by a Release callback.
class CudaExternalDeviceAllocator final : public CudaAllocatorBase {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  CudaExternalDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id,
                              void* alloc_fn, void* free_fn, void* empty_cache_fn);
  ~CudaExternalDeviceAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;

  int device_id_;
  ExternalAlloc alloc_fn_;
  ExternalFree free_fn_;
  ExternalEmptyCache empty_cache_fn_;
  mutable std::mutex lock_;
  std::unordered_set<void*> reserved_;
};

/// CUDA pinned (host) memory allocator using cudaHostAlloc/cudaFreeHost.
/// Lifetime is managed by the EP factory (ReleaseAllocatorImpl), not by a Release callback.
class CudaPinnedAllocator final : public CudaAllocatorBase {
 public:
  CudaPinnedAllocator(const OrtMemoryInfo* memory_info);
  ~CudaPinnedAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
