// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "cuda_device_memory_budget.h"
#include <functional>
#include <memory>
#include <mutex>

namespace onnxruntime {

std::shared_ptr<CudaDeviceMemoryBudget> GetCudaDeviceMemoryBudget(OrtDevice::DeviceId device_id,
                                                                  size_t limit = 0);

class CUDAAllocator : public IAllocator {
 public:
  CUDAAllocator(OrtDevice::DeviceId device_id, const char* name,
                std::shared_ptr<SessionBudgetAccount> account = nullptr);
  ~CUDAAllocator() override;
  void* Alloc(size_t size) override;
  void Free(void* p) override;

  CudaDeviceMemoryBudget::Stats GetAccountStats() const {
    return account_ ? account_->GetAccountStats() : budget_->GetStats();
  }
  CudaDeviceMemoryBudget::Stats GetPhysicalStats() const {
    return budget_ ? budget_->GetStats() : CudaDeviceMemoryBudget::Stats{};
  }
  const std::shared_ptr<SessionBudgetAccount>& GetBudgetAccount() const noexcept { return account_; }
  void SetAfterDriverFreeHookForTest(std::function<void(void*)> hook) {
    on_after_driver_free_for_test_ = std::move(hook);
  }

 private:
  enum class AllocationState : uint8_t {
    kCommitted = 0,
    kFreeing = 1,
    kQuarantined = 2,
  };
  struct AllocationRecord {
    size_t bytes = 0;
    uint64_t generation = 0;
    AllocationState state = AllocationState::kCommitted;
  };
  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;
  std::shared_ptr<CudaDeviceMemoryBudget> budget_;
  std::shared_ptr<SessionBudgetAccount> account_;
  std::mutex allocations_mutex_;
  uint64_t next_generation_ = 0;
  InlinedHashMap<void*, AllocationRecord> allocations_;
  std::function<void(void*)> on_after_driver_free_for_test_;
};

class CUDAExternalAllocator : public CUDAAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);
  typedef void (*ExternalEmptyCache)();

 public:
  CUDAExternalAllocator(OrtDevice::DeviceId device_id, const char* name, void* alloc, void* free, void* empty_cache)
      : CUDAAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
    empty_cache_ = reinterpret_cast<ExternalEmptyCache>(empty_cache);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void* Reserve(size_t size) override;

 private:
  mutable std::mutex lock_;
  ExternalAlloc alloc_;
  ExternalFree free_;
  ExternalEmptyCache empty_cache_;
  InlinedHashSet<void*> reserved_;
};

// TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                                    device_id),
                          OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};
}  // namespace onnxruntime
