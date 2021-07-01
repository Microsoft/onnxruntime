// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/utils.h"
#include "core/session/ort_apis.h"
#include <cstdlib>
#include <sstream>

namespace onnxruntime {

// private helper for calculation so SafeInt usage doesn't bleed into the public allocator.h header
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept {
  bool ok = true;

  ORT_TRY {
    SafeInt<size_t> alloc_size(size);
    if (alignment == 0) {
      *out = alloc_size * nmemb;
    } else {
      size_t alignment_mask = alignment - 1;
      *out = (alloc_size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
    }
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    // overflow in calculating the size thrown by SafeInt.
    ORT_HANDLE_EXCEPTION([&]() {
      LOGS_DEFAULT(ERROR) << ex.what();
      ok = false;
    });
  }
  return ok;
}

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
void* MiMallocAllocator::Alloc(size_t size) {
  return mi_malloc(size);
}

void MiMallocAllocator::Free(void* p) {
  mi_free(p);
}
#endif

void* CPUAllocator::Alloc(size_t size) {
  return utils::DefaultAlloc(size);
}

void CPUAllocator::Free(void* p) {
  utils::DefaultFree(p);
}
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info) { return (out << info.ToString()); }

ORT_API_STATUS_IMPL(OrtApis::CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                    enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out) {
  if (strcmp(name1, onnxruntime::CPU) == 0) {
    *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::CUDA, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)), id1,
        mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::CUDA_PINNED, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Specified device is not supported.");
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo* p) { delete p; }

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out) {
  *out = ptr->name;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out) {
  *out = ptr->id;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out) {
  *out = ptr->mem_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out) {
  *out = ptr->alloc_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2,
                    _Out_ int* out) {
  *out = (*info1 == *info2) ? 0 : -1;
  return nullptr;
}
