// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class ROCMAllocator : public IAllocator {
 public:
  ROCMAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
};

class ROCMExternalAllocator : public ROCMAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);

 public:
  ROCMExternalAllocator(OrtDevice::DeviceId device_id, const char* name, const void* alloc, const void* free)
      : ROCMAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(const_cast<void*>(alloc));
    free_ = reinterpret_cast<ExternalFree>(const_cast<void*>(free));
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  ExternalAlloc alloc_;
  ExternalFree free_;
};

//TODO: add a default constructor
class ROCMPinnedAllocator : public IAllocator {
 public:
  ROCMPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::GPU_PINNED, device_id),
                          device_id, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  FencePtr CreateFence(const SessionState* session_state) override;
};
}  // namespace onnxruntime
