// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator_plugin.h"

namespace onnxruntime {
namespace cuda_plugin {

namespace {

void RestoreDeviceIfKnown(bool restore_prev_device, int prev_device) noexcept {
  if (restore_prev_device) {
    static_cast<void>(cudaSetDevice(prev_device));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// CudaDeviceAllocator — uses cudaMalloc/cudaFree for GPU device memory.
//
// PERFORMANCE NOTE (Direct cudaMalloc Penalty):
// No arena or caching layer is provided within this plugin. Every allocation
// goes directly to CUDA (cudaMalloc). For models with dynamic shape resizing
// or many intermediate buffers, this can cause substantial overhead.
// Compared to the built-in CUDA Execution Provider, which has an integrated
// memory arena, this is a notable performance gap unless an external
// memory pool/arena is injected or configured by the application.
// ---------------------------------------------------------------------------

CudaDeviceAllocator::CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id, const OrtApi& api,
                                         std::shared_ptr<SessionBudgetAccount> account)
    : CudaAllocatorBase(CudaAllocatorKind::kDevice, memory_info),
      device_id_(device_id),
      api_(api),
      allocator_(static_cast<OrtDevice::DeviceId>(device_id), CUDA, std::move(account)) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = GetStatsImpl;
  AllocOnStream = nullptr;
}

/*static*/ void* ORT_API_CALL CudaDeviceAllocator::AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  auto* alloc = static_cast<CudaDeviceAllocator*>(this_ptr);
  void* p = nullptr;
  if (size == 0) return nullptr;
  // Save and restore CUDA device context to avoid corrupting the calling
  // thread's device state in multi-GPU scenarios.
  int prev_device = -1;
  const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
  if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
    return nullptr;
  }
  try {
    p = alloc->allocator_.Alloc(size);
  } catch (...) {
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
    return nullptr;
  }
  RestoreDeviceIfKnown(restore_prev_device, prev_device);
  return p;
}

/*static*/ void ORT_API_CALL CudaDeviceAllocator::FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
  auto* alloc = static_cast<CudaDeviceAllocator*>(this_ptr);
  if (p != nullptr) {
    int prev_device = -1;
    const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
    if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
      RestoreDeviceIfKnown(restore_prev_device, prev_device);
      return;
    }

    alloc->allocator_.Free(p);
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaDeviceAllocator::InfoImpl(const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaDeviceAllocator*>(this_ptr);
  return alloc->GetMemoryInfo();
}

/*static*/ void* ORT_API_CALL CudaDeviceAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  // Reserve currently delegates to Alloc (no separate reservation pool).
  return AllocImpl(this_ptr, size);
}

/*static*/ OrtStatus* ORT_API_CALL CudaDeviceAllocator::GetStatsImpl(
    const OrtAllocator* this_ptr, OrtKeyValuePairs** stats) noexcept {
  const auto* allocator = static_cast<const CudaDeviceAllocator*>(this_ptr);
  try {
    const auto phys_snapshot = allocator->allocator_.GetPhysicalStats();
    const auto sess_snapshot = allocator->allocator_.GetAccountStats();
    const size_t eff_limit = std::min(phys_snapshot.limit, sess_snapshot.limit);
    allocator->api_.CreateKeyValuePairs(stats);
    allocator->api_.AddKeyValuePair(*stats, "DeviceLimit", std::to_string(eff_limit).c_str());
    allocator->api_.AddKeyValuePair(*stats, "DeviceCommitted", std::to_string(phys_snapshot.committed).c_str());
    allocator->api_.AddKeyValuePair(*stats, "DevicePending", std::to_string(phys_snapshot.pending).c_str());
    allocator->api_.AddKeyValuePair(*stats, "DevicePeak", std::to_string(phys_snapshot.peak).c_str());
    allocator->api_.AddKeyValuePair(*stats, "DeviceFreeing", std::to_string(phys_snapshot.freeing).c_str());
    allocator->api_.AddKeyValuePair(*stats, "DeviceQuarantined", std::to_string(phys_snapshot.quarantined).c_str());
    allocator->api_.AddKeyValuePair(*stats, "SessionLimit", std::to_string(sess_snapshot.limit).c_str());
    allocator->api_.AddKeyValuePair(*stats, "SessionCommitted", std::to_string(sess_snapshot.committed).c_str());
    allocator->api_.AddKeyValuePair(*stats, "SessionPending", std::to_string(sess_snapshot.pending).c_str());
    allocator->api_.AddKeyValuePair(*stats, "SessionPeak", std::to_string(sess_snapshot.peak).c_str());
    allocator->api_.AddKeyValuePair(*stats, "PhysicalLimit", std::to_string(phys_snapshot.limit).c_str());
    allocator->api_.AddKeyValuePair(*stats, "PhysicalCommitted", std::to_string(phys_snapshot.committed).c_str());
    allocator->api_.AddKeyValuePair(*stats, "PhysicalPeak", std::to_string(phys_snapshot.peak).c_str());
    return nullptr;
  } catch (const std::exception& error) {
    return allocator->api_.CreateStatus(ORT_FAIL, error.what());
  } catch (...) {
    return allocator->api_.CreateStatus(ORT_FAIL, "Failed to read CUDA device memory budget");
  }
}

// ---------------------------------------------------------------------------
// CudaExternalDeviceAllocator — delegates to user-provided function pointers.
// ---------------------------------------------------------------------------

CudaExternalDeviceAllocator::CudaExternalDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id,
                                                         void* alloc_fn, void* free_fn, void* empty_cache_fn)
    : CudaAllocatorBase(CudaAllocatorKind::kDevice, memory_info, true),
      device_id_(device_id),
      alloc_fn_(reinterpret_cast<ExternalAlloc>(alloc_fn)),
      free_fn_(reinterpret_cast<ExternalFree>(free_fn)),
      empty_cache_fn_(reinterpret_cast<ExternalEmptyCache>(empty_cache_fn)) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
  AllocOnStream = nullptr;
}

/*static*/ void* ORT_API_CALL CudaExternalDeviceAllocator::AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  auto* alloc = static_cast<CudaExternalDeviceAllocator*>(this_ptr);
  if (size == 0) return nullptr;
  if (alloc->alloc_fn_ == nullptr) return nullptr;

  int prev_device = -1;
  const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
  if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
    return nullptr;
  }

  void* p = alloc->alloc_fn_(size);
  RestoreDeviceIfKnown(restore_prev_device, prev_device);
  return p;
}

/*static*/ void ORT_API_CALL CudaExternalDeviceAllocator::FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
  auto* alloc = static_cast<CudaExternalDeviceAllocator*>(this_ptr);
  if (p != nullptr && alloc->free_fn_ != nullptr) {
    int prev_device = -1;
    const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
    if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
      RestoreDeviceIfKnown(restore_prev_device, prev_device);
      return;
    }

    alloc->free_fn_(p);
    RestoreDeviceIfKnown(restore_prev_device, prev_device);

    // If this was a reserved allocation, invoke empty_cache to release cached memory
    // (matching bundled CUDA EP behavior).
    std::lock_guard<std::mutex> lock(alloc->lock_);
    auto it = alloc->reserved_.find(p);
    if (it != alloc->reserved_.end()) {
      alloc->reserved_.erase(it);
      if (alloc->empty_cache_fn_) {
        alloc->empty_cache_fn_();
      }
    }
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaExternalDeviceAllocator::InfoImpl(
    const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaExternalDeviceAllocator*>(this_ptr);
  return alloc->GetMemoryInfo();
}

/*static*/ void* ORT_API_CALL CudaExternalDeviceAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  void* p = AllocImpl(this_ptr, size);
  if (p != nullptr) {
    auto* alloc = static_cast<CudaExternalDeviceAllocator*>(this_ptr);
    std::lock_guard<std::mutex> lock(alloc->lock_);
    alloc->reserved_.insert(p);
  }
  return p;
}

// ---------------------------------------------------------------------------
// CudaPinnedAllocator — uses cudaHostAlloc/cudaFreeHost for page-locked
// host memory visible to the GPU.
// ---------------------------------------------------------------------------

CudaPinnedAllocator::CudaPinnedAllocator(const OrtMemoryInfo* memory_info)
    : CudaAllocatorBase(CudaAllocatorKind::kPinned, memory_info) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
  AllocOnStream = nullptr;
}

/*static*/ void* ORT_API_CALL CudaPinnedAllocator::AllocImpl(OrtAllocator* /*this_ptr*/, size_t size) noexcept {
  void* p = nullptr;
  if (size == 0) return nullptr;
  cudaError_t err = cudaHostAlloc(&p, size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return p;
}

/*static*/ void ORT_API_CALL CudaPinnedAllocator::FreeImpl(OrtAllocator* /*this_ptr*/, void* p) noexcept {
  if (p != nullptr) {
    cudaFreeHost(p);
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaPinnedAllocator::InfoImpl(const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaPinnedAllocator*>(this_ptr);
  return alloc->GetMemoryInfo();
}

/*static*/ void* ORT_API_CALL CudaPinnedAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  // Reserve currently delegates to Alloc (no separate reservation pool).
  return AllocImpl(this_ptr, size);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
