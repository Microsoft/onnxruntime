// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

// Since all allocators are of type 'OrtAllocator' and there is a single
// OrtApis:ReleaseAllocator function, we need to have a common base type that lets us delete them.
struct OrtAllocatorImpl : public OrtAllocator {
  virtual ~OrtAllocatorImpl() = default;
};

// The following are "adapters" to allow using an IAllocator implementation wrapped as an OrtAllocator
// and vice versa to plug into any ORT internal code/ API implementation as necessary

struct OrtAllocatorImplWrappingIAllocator : public OrtAllocatorImpl {
  explicit OrtAllocatorImplWrappingIAllocator(onnxruntime::AllocatorPtr&& i_allocator);

  ~OrtAllocatorImplWrappingIAllocator() override = default;

  void* Alloc(size_t size);

  void Free(void* p);

  const OrtMemoryInfo* Info() const;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtAllocatorImplWrappingIAllocator);

 private:
  onnxruntime::AllocatorPtr i_allocator_;
};

class IAllocatorImplWrappingOrtAllocator : public IAllocator {
 public:
  explicit IAllocatorImplWrappingOrtAllocator(OrtAllocator* ort_allocator);

  ~IAllocatorImplWrappingOrtAllocator() override = default;

  void* Alloc(size_t size) override;

  void Free(void* p) override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(IAllocatorImplWrappingOrtAllocator);

 private:
  OrtAllocator* ort_allocator_ = nullptr;
};

}  // namespace onnxruntime
