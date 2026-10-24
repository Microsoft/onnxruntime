// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator.h"
#include "cuda_common.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

std::shared_ptr<CudaDeviceMemoryBudget> GetCudaDeviceMemoryBudget(OrtDevice::DeviceId device_id, size_t limit) {
  return GetCudaDeviceMemoryBudgetByOrdinal(static_cast<int>(device_id), limit);
}

CUDAAllocator::CUDAAllocator(OrtDevice::DeviceId device_id, const char* name,
                             std::shared_ptr<SessionBudgetAccount> account)
    : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                         OrtDevice::VendorIds::NVIDIA, device_id),
                               OrtMemTypeDefault)),
      budget_(GetCudaDeviceMemoryBudget(device_id)),
      account_(std::move(account)) {
  if (budget_) {
    budget_->AddOwner();
    if (!account_) {
      const size_t pending_limit = budget_->ConsumePendingSessionLimit();
      account_ = std::make_shared<SessionBudgetAccount>(budget_, pending_limit);
    }
  }
}

CUDAAllocator::~CUDAAllocator() {
  if (budget_) budget_->RemoveOwner();
}

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    ORT_ENFORCE(current_device == Info().device.Id());
  } else if (throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().device.Id();
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  if (cuda_err != cudaSuccess && throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  if (size == 0) return nullptr;
  auto record_allocation = [this, size](void* allocation) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    const uint64_t gen = ++next_generation_;
    auto [it, inserted] = allocations_.emplace(
        allocation, AllocationRecord{size, gen, AllocationState::kCommitted});
    if (!inserted) {
      ORT_ENFORCE(it->second.state != AllocationState::kCommitted,
                  "CUDAAllocator encountered duplicate live address from driver");
      it->second = AllocationRecord{size, gen, AllocationState::kCommitted};
    }
  };
  if (account_) {
    auto reservation = account_->Reserve(size);
    void* allocation = nullptr;
    CUDA_CALL_THROW(cudaMalloc(&allocation, size));
    try {
      record_allocation(allocation);
    } catch (...) {
      if (cudaFree(allocation) != cudaSuccess) reservation.Commit();
      throw;
    }
    reservation.Commit();
    return allocation;
  }
  auto reservation = budget_->Reserve(size);
  void* allocation = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&allocation, size));
  try {
    record_allocation(allocation);
  } catch (...) {
    if (cudaFree(allocation) != cudaSuccess) reservation.Commit();
    throw;
  }
  reservation.Commit();
  return allocation;
}

void CUDAAllocator::Free(void* p) {
  if (p == nullptr) return;
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  size_t bytes = 0;
  uint64_t freeing_gen = 0;
  bool tracked = false;
  {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    const auto it = allocations_.find(p);
    if (it != allocations_.end()) {
      if (it->second.state != AllocationState::kCommitted) {
        return;  // Prevent concurrent double-free or re-free of quarantined pointer (F4).
      }
      it->second.state = AllocationState::kFreeing;
      bytes = it->second.bytes;
      freeing_gen = it->second.generation;
      tracked = true;
      if (account_) {
        account_->BeginFree(bytes);
      } else {
        budget_->BeginFree(bytes);
      }
    }
  }

  // Call driver free outside allocations_mutex_ (F4).
  const bool driver_ok = (cudaFree(p) == cudaSuccess);
  if (on_after_driver_free_for_test_) {
    on_after_driver_free_for_test_(p);
  }

  if (tracked) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    if (account_) {
      account_->FinishFree(bytes, driver_ok);
    } else {
      budget_->FinishFree(bytes, driver_ok);
    }
    const auto it = allocations_.find(p);
    if (it != allocations_.end() && it->second.generation == freeing_gen) {
      if (driver_ok) {
        allocations_.erase(it);
      } else {
        it->second.state = AllocationState::kQuarantined;
      }
    }
  }
}

void* CUDAExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropriate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void CUDAExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<std::mutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_) empty_cache_();
  }
}

void* CUDAExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<std::mutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}

}  // namespace onnxruntime
