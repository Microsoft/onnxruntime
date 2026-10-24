// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>

#include "core/common/common.h"
#include "core/framework/allocator_stats.h"
#include "core/session/abi_key_value_pairs.h"
// some enums are defined in session/onnxruntime_c_api.h but used in ortdevice.h/ortmemory.h
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

// This configures the arena based allocator used by ORT
// See docs/C_API.md for details on what these mean and how to choose these values
struct OrtArenaCfg {
  OrtArenaCfg() : max_mem(0),
                  arena_extend_strategy(-1),
                  initial_chunk_size_bytes(-1),
                  max_dead_bytes_per_chunk(-1),
                  initial_growth_chunk_size_bytes(-1),
                  max_power_of_two_extend_bytes(-1) {}
  OrtArenaCfg(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
              int max_dead_bytes_per_chunk, int initial_growth_chunk_size_bytes,
              int64_t max_power_of_two_extend_bytes)
      : max_mem(max_mem),
        arena_extend_strategy(arena_extend_strategy),
        initial_chunk_size_bytes(initial_chunk_size_bytes),
        max_dead_bytes_per_chunk(max_dead_bytes_per_chunk),
        initial_growth_chunk_size_bytes(initial_growth_chunk_size_bytes),
        max_power_of_two_extend_bytes(max_power_of_two_extend_bytes) {}

  size_t max_mem;                         // use 0 to allow ORT to choose the default
  int arena_extend_strategy;              // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  int initial_chunk_size_bytes;           // use -1 to allow ORT to choose the default
  int max_dead_bytes_per_chunk;           // use -1 to allow ORT to choose the default
  int initial_growth_chunk_size_bytes;    // use -1 to allow ORT to choose the default
  int64_t max_power_of_two_extend_bytes;  // use -1 to allow ORT to choose the default
  // Use CudaMemPool based arena if available (starting with cuda 11.2)
  int use_cuda_mempool = -1;
  // Amount of reserved memory in bytes to hold onto before trying
  // to release memory back to the OS.
  uint64_t cuda_mempool_release_threshold = 0;
  // Bytes to keep on shrink for CudaMemPool, 0 is to attempt to release all, allocated space not affected.
  size_t cuda_mempool_bytes_to_keep_on_shrink = 0;

  bool IsValid() {
    return arena_extend_strategy >= -1 && arena_extend_strategy <= 1 &&
           initial_chunk_size_bytes >= -1 &&
           max_dead_bytes_per_chunk >= -1 &&
           initial_growth_chunk_size_bytes >= -1 &&
           max_power_of_two_extend_bytes >= -1;
  }

  // config key names that we parse in FromKeyValuePairs
  struct ConfigKeyNames {
    static constexpr const char* ArenaExtendStrategy = "arena.extend_strategy";
    static constexpr const char* InitialChunkSizeBytes = "arena.initial_chunk_size_bytes";
    static constexpr const char* MaxDeadBytesPerChunk = "arena.max_dead_bytes_per_chunk";
    static constexpr const char* InitialGrowthChunkSizeBytes = "arena.initial_growth_chunk_size_bytes";
    static constexpr const char* MaxPowerOfTwoExtendBytes = "arena.max_power_of_two_extend_bytes";
    static constexpr const char* MaxMem = "arena.max_mem";
    static constexpr const char* UseCudaMemPool = "arena.use_cuda_mempool";
    static constexpr const char* CudaMempoolReleaseThreshold = "arena.cuda_mempool_release_threshold";
    static constexpr const char* CudaMempoolBytesToKeepOnShrink = "arena.cuda_mempool_bytes_to_keep_on_shrink";
  };

  static onnxruntime::common::Status FromKeyValuePairs(const OrtKeyValuePairs& kvps, OrtArenaCfg& cfg);
};

namespace onnxruntime {
constexpr const char* CPU = "Cpu";
constexpr const char* CPU_ALIGNED_4K = "CpuAligned4K";
constexpr const char* CUDA = "Cuda";
constexpr const char* CUDA_PINNED = "CudaPinned";
constexpr const char* CANN = "Cann";
constexpr const char* CANN_PINNED = "CannPinned";
constexpr const char* DML = "DML";
constexpr const char* HIP = "Hip";
constexpr const char* HIP_PINNED = "HipPinned";
constexpr const char* OpenVINO_CPU = "OpenVINO_CPU";
constexpr const char* OpenVINO_GPU = "OpenVINO_GPU";
constexpr const char* OpenVINO_RT = "OpenVINO_RT";
constexpr const char* OpenVINO_RT_NPU = "OpenVINO_RT_NPU";
constexpr const char* QNN_HTP_SHARED = "QnnHtpShared";
constexpr const char* WEBGPU_BUFFER = "WebGPU_Buf";  // limited to 10 chars to ensure std::string SSO for web
constexpr const char* WEBNN_TENSOR = "WebNN_Ten";    // limited to 10 chars to ensure std::string SSO for web

constexpr size_t kAllocAlignment = 256;
constexpr const size_t kAlloc4KAlignment = 4096;

class IAllocator;
class Stream;
namespace synchronize {
class Notification;
}

using WaitNotificationFn = std::function<void(Stream*, synchronize::Notification&)>;
void* AllocateBufferWithOptions(IAllocator& allocator, size_t size, bool use_reserve, Stream* stream,
                                // wait fn is for backwards compat with provider bridge
                                WaitNotificationFn ignored = nullptr);

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

class IAllocator {
 public:
  IAllocator(const OrtMemoryInfo& info) : memory_info_(info) {}
  virtual ~IAllocator() = default;
  /**
   * Allocate memory of the specified size.
   * If size is 0, nullptr is returned.
   * If allocation fails, an exception is thrown.
   *
   * @remarks Use SafeInt when calculating the size of memory to allocate using Alloc.
   */
  virtual void* Alloc(size_t size) = 0;

  /** Return true if the allocator implements Stream handling in AllocOnStream.
   */
  virtual bool IsStreamAware() const { return false; }

  /** Allocate memory, handling usage across different Streams
   *
   * A device Stream may be available when executing a model on non-CPU devices. In this case operations are queued
   * asynchronously and the allocation/free call is made when the operation is queued rather than executed.
   * Due to this it is not safe to use the memory on another stream or with no stream unless synchronization has
   * occurred.
   *
   * ORT currently handles the synchronization when executing the model using streams.
   *
   * When two streams are synchronized the event used is identified by the producer stream's latest sync id.
   * This <producer stream:sync id> pair is copied into the sync information of the consumer stream.
   * Each new event creates a new sync id.
   *
   * It is safe to re-use an allocated piece of memory if:
   *   - the stream that currently owns the memory and the stream that wants to use the memory have been synchronized,
   *   - and the sync id from when the memory was assigned to the stream that currently owns it is less than the
   *     sync id from the last synchronization between the two streams.
   *     - e.g. stream0 is assigned the memory when its sync id is 1.
   *            stream0 (producer) and stream1 (consumer) are synchronized.
   *              stream0 sync id will be incremented to 2 when creating the event used in the synchronization.
   *              stream1 will copy this information into its sync info and now contains an entry for stream0
   *              with a sync id of 2.
   *            stream0 frees the memory
   *              the memory is marked as not in use, but still assigned to stream0
   *            stream1 is now able to use the memory as it is not in use, and the sync id from the allocation (1)
   *            is less than the sync id (2) that is has for stream0.
   *   or
   *   - the inference session that owned the Stream has completed inferencing
   *     - Stream::CleanUpOnRunEnd is called when this occurs
   *     - any memory assigned to the Stream is now able to be used by other streams when it is not longer in use.
   */
  virtual void* AllocOnStream(size_t size, Stream* /*stream*/) { return Alloc(size); }

  /**
   * Free memory at p.
   * If p is nullptr, do nothing.
   */
  virtual void Free(void* p) = 0;

  // Reserve() is an interface exposed for an implementation of IAllocator
  // to optionally implement some allocation logic that by-passes any arena-based
  // logic that may be housed in the Alloc() implementation.
  // There are SessionOptions config(s) that allow users to allocate some memory
  // by-passing arena-based logic.
  // By default, the base implementation  just calls Alloc().
  virtual void* Reserve(size_t size) { return Alloc(size); }

  const OrtMemoryInfo& Info() const { return memory_info_; };

  // Each implementation of IAllocator can override and provide their own implementation
  virtual void GetStats(AllocatorStats* stats) {
    *stats = {};
  }

  // Returns a pointer to this allocator as an IArena if it is one, nullptr otherwise.
  // Used by SafeArenaCast to avoid dependency on RTTI.
  virtual class IArena* AsArena() { return nullptr; }
  virtual const class IArena* AsArena() const { return nullptr; }

  virtual const OrtAllocator* GetWrappedOrtAllocator() const { return nullptr; }

  static bool CalcMemSizeForArray(size_t nmemb, size_t size, size_t* out) noexcept {
    return CalcMemSizeForArrayWithAlignment(nmemb, size, 0, out);
  }

  /**
   * Calculate the memory size for an array. The size is bounds checked using SafeInt.
   * \tparam alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   */
  [[nodiscard]] static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment,
                                                             size_t* out) noexcept;

  /**
   * https://cwe.mitre.org/data/definitions/190.html
   * \param alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   * \remarks This was the original API and was implemented in the header. Replaced with the above version
   *          implemented in the .cc file so that the SafeInt dependency is internal.
   */
  template <size_t alignment>
  [[nodiscard]] static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept;

  /**
   * allocate memory for an array which has nmemb items of data, each size bytes long
   */
  void* AllocArray(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArray(nmemb, size, &len)) {
      ORT_THROW("Invalid size requested for allocation: ", nmemb, " * ", size);
    }

    return Alloc(len);
  }

  /**
   * allocate memory for an array which has nmemb items of data, each size bytes long
   */
  template <size_t alignment>
  void* AllocArrayWithAlignment(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, &len)) {
      ORT_THROW("Invalid size requested for allocation: ", nmemb, " * ", size, " with alignment ", alignment);
    }

    return Alloc(len);
  }

  /**
     Create a std::unique_ptr that is allocated and freed by the provided IAllocator.
     @param allocator The allocator.
     @param count_or_bytes The exact bytes to allocate if T is void, otherwise the number of elements to allocate.
     @param use_reserve If true, call Reserve() instead of Alloc() to allocate memory.
     @param stream Which stream instance allocated chunk will be used with.
     @param wait_fn If the allocator want to dynamic reuse a chunk from another stream, use this wait_fn to sync on
                    the target stream to make the reuse safe.
     @returns std::unique_ptr with allocated memory and deleter. Throws if it cannot allocate memory.
  */
  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes,
                                              bool use_reserve = false,
                                              Stream* stream = nullptr) {
    ValidateAllocator(allocator);

    // for now limit to fundamental types. we could support others, but to do so either we or the caller
    // needs to call the dtor for the objects, for buffers allocated on device we don't have destructor
    // static_assert(std::is_fundamental<T>::value, "Fundamental type required as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if constexpr (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      constexpr auto size = sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);
      alloc_size = ValidatedCalcMemSizeForArray(count_or_bytes, size);
    }

    // allocate
    T* p = static_cast<T*>(AllocateBufferWithOptions(*allocator, alloc_size, use_reserve, stream, nullptr));
    ValidateAllocation(p, alloc_size);

    return IAllocatorUniquePtr<T>{p,
                                  [allocator = std::move(allocator)](T* p) {
                                    allocator->Free(p);
                                  }};
  }

  /**
     Create a std::unique_ptr that is allocated and freed by the provided OrtAllocator.
     @param ort_allocator The allocator.
     @param count_or_bytes The exact bytes to allocate if T is void, otherwise the number of elements to allocate.
     @returns std::unique_ptr with allocated memory and deleter. Throws if it cannot allocate memory.
  */
  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtrFromOrtAllocator(OrtAllocator* ort_allocator, size_t count_or_bytes,
                                                              bool use_reserve = false) {
    ValidateAllocator(ort_allocator);

    size_t alloc_size = count_or_bytes;
    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if constexpr (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      constexpr auto size = sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);
      alloc_size = ValidatedCalcMemSizeForArray(count_or_bytes, size);
    }

    T* p = nullptr;
    if (use_reserve) {
      p = static_cast<T*>(ort_allocator->Reserve(ort_allocator, alloc_size));
    } else {
      p = static_cast<T*>(ort_allocator->Alloc(ort_allocator, alloc_size));
    }
    ValidateAllocation(p, alloc_size);

    return IAllocatorUniquePtr<T>{p,
                                  [ort_allocator](T* p) {
                                    ort_allocator->Free(ort_allocator, p);
                                  }};
  }

 private:
  //
  // validation functions. split out from methods that are templatized on the data type to minimize binary size.
  //

  template <typename T>
  static void ValidateAllocator(const T& allocator) {
    ORT_ENFORCE(allocator != nullptr);
  }

  static size_t ValidatedCalcMemSizeForArray(size_t count, size_t size) {
    size_t alloc_size = 0;
    if (!CalcMemSizeForArray(count, size, &alloc_size)) {
      ORT_THROW("Invalid size requested for allocation: ", count, " * ", size);
    }

    return alloc_size;
  }

  static void ValidateAllocation(void* p, size_t size) {
    // allocator should throw directly but in case it didn't ensure we do here so that calling code doesn't
    // need to check for nullptr when an actual allocation was expected.
    ORT_ENFORCE(p != nullptr || size == 0, "Memory allocation failed. Size=", size);
  };

  OrtMemoryInfo memory_info_;
};

template <size_t alignment>
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept {
  return CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out);
}

using AllocatorPtr = std::shared_ptr<IAllocator>;
using AllocatorMap = std::map<OrtDevice, AllocatorPtr>;

class CPUAllocator : public IAllocator {
 public:
  explicit CPUAllocator(const OrtMemoryInfo& memory_info) : IAllocator(memory_info) {}

  // Creates a function local static and returns a shared pointer to it.
  // Re-use in all places where we need a standalone CPUAllocator instance
  static AllocatorPtr DefaultInstance();

  CPUAllocator() : IAllocator(OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

void* AllocatorDefaultAlloc(size_t size);
void AllocatorDefaultFree(void* p);
void* AllocatorDefaultAllocAligned(size_t size, size_t alignment);
void AllocatorDefaultFreeAligned(void* p, size_t alignment);

class IArena : public IAllocator {
 public:
  using IAllocator::IAllocator;
  virtual Status Shrink() = 0;
  // Only implemented when IsStreamAware() returns true
  virtual void ReleaseStreamBuffers(Stream* /*stream*/) {}
  IArena* AsArena() override { return this; }
  const IArena* AsArena() const override { return this; }
  static IArena* SafeArenaCast(IAllocator* allocator);
};

}  // namespace onnxruntime
