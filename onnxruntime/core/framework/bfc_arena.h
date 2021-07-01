/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <sstream>

#include "onnxruntime_config.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/common/safeint.h"

#include "core/platform/ort_mutex.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/allocator.h"

#if defined(PLATFORM_WINDOWS)
#include <intrin.h>
#endif
namespace onnxruntime {
#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_NULL_DEREFERENCE
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#endif

// A memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the memory, and that nearly
// all requests to allocate memory go through this interface.
class BFCArena : public IAllocator {
 public:
  static const ArenaExtendStrategy DEFAULT_ARENA_EXTEND_STRATEGY = ArenaExtendStrategy::kNextPowerOfTwo;
  static const int DEFAULT_INITIAL_CHUNK_SIZE_BYTES = 1 * 1024 * 1024;
  static const int DEFAULT_MAX_DEAD_BYTES_PER_CHUNK = 128 * 1024 * 1024;
  static const int DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES = 2 * 1024 * 1024;
  static const size_t DEFAULT_MAX_MEM = std::numeric_limits<size_t>::max();

  BFCArena(std::unique_ptr<IAllocator> resource_allocator,
           size_t total_memory,
           ArenaExtendStrategy arena_extend_strategy = DEFAULT_ARENA_EXTEND_STRATEGY,
           int initial_chunk_size_bytes = DEFAULT_INITIAL_CHUNK_SIZE_BYTES,
           int max_dead_bytes_per_chunk = DEFAULT_MAX_DEAD_BYTES_PER_CHUNK,
           int initial_growth_chunk_size_bytes = DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES);

  ~BFCArena() override;

  //If size is 0, then this function returns either NULL,
  //or a unique pointer value that can later be successfully
  //passed to free(). Whatever, do not dereference that pointer
  void* Alloc(size_t size) override;

  //If p is NULL, no operation is performed.
  void Free(void* p) override;

  // Frees all allocation regions in which no chunk is in use.
  // Does not free any reserved chunks.
  // Resets the size that the arena will grow by in the next allocation to
  // `initial_growth_chunk_size_bytes_` but ultimately all
  // future allocation sizes are determined by the arena growth strategy
  // and the allocation request.
  Status Shrink();

  void* Reserve(size_t size) override;

  FencePtr CreateFence(const SessionState* session_state) override {
    // arena always rely on its device allocator to create fence
    return device_allocator_->CreateFence(session_state);
  }

  void GetStats(AllocatorStats* stats) override;

  size_t RequestedSize(const void* ptr);

  size_t AllocatedSize(const void* ptr);

 private:
  void* AllocateRawInternal(size_t num_bytes, bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  using ChunkHandle = size_t;
  static const size_t kInvalidChunkHandle = static_cast<size_t>(-1);

  using BinNum = int;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  // Chunks point to memory.  Their prev/next pointers form a
  // doubly-linked list of addresses sorted by base address that
  // must be contiguous.  Chunks contain information about whether
  // they are in use or whether they are free, and contain a pointer
  // to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64_t allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    bool in_use() const { return allocation_id != -1; }

    std::string DebugString(BFCArena* a, bool recurse) {
      std::ostringstream ss;
      ss << "  Size: " << size << " | Requested Size: " << requested_size << " | in_use: " << in_use();
      if (recurse && prev != BFCArena::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        ss << ", prev: " << p->DebugString(a, false);
      }

      if (recurse && next != BFCArena::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        ss << ", next: " << n->DebugString(a, false);
      }
      return ss.str();
    }
  };

  // A Bin is a collection of similar-sized free chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      explicit ChunkComparator(BFCArena* allocator)
          : allocator_(allocator) {}

      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha,
                      const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCArena* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCArena* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static const size_t kMinAllocationBits = 8;
  static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // AllocationRegion maps pointers to ChunkHandles for a single
  // contiguous memory region.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size, int64_t id)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(
              static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)),
          id_(id) {
      ORT_ENFORCE(0 == memory_size % kMinAllocationSize);
      const size_t n_handles =
          (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_ = new ChunkHandle[n_handles];
      for (size_t i = 0; i < n_handles; i++) {
        handles_[i] = kInvalidChunkHandle;
      }
    }

    AllocationRegion() = default;

    ~AllocationRegion() { delete[] handles_; }

    AllocationRegion(AllocationRegion&& other) noexcept { Swap(other); }

    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    int64_t id() const { return id_; }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion& other) {
      std::swap(ptr_, other.ptr_);
      std::swap(memory_size_, other.memory_size_);
      std::swap(end_ptr_, other.end_ptr_);
      std::swap(id_, other.id_);
      std::swap(handles_, other.handles_);
    }

    int IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      ORT_ENFORCE(p_int >= base_int);
      ORT_ENFORCE(p_int < base_int + memory_size_);
      return static_cast<int>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;
    // A unique identifier for this allocation region
    // (May be used by the client to track which allocation region was allocated first, second, and so on)
    int64_t id_ = -1;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    ChunkHandle* handles_ = nullptr;

    ORT_DISALLOW_ASSIGNMENT(AllocationRegion);
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() = default;
    ~RegionManager() = default;

    void AddAllocationRegion(void* ptr, size_t memory_size, int64_t id) {
      // Insert sorted by end_ptr
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size, id));
    }

    void RemoveAllocationRegion(void* ptr) {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);

      ORT_ENFORCE(entry != regions_.end(), "Could not find Region for: ", ptr);

      regions_.erase(entry);
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }
    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RegionManager);

    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      LOGS_DEFAULT(FATAL) << "Could not find Region for " << p;
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  size_t RoundedBytes(size_t bytes);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.
  Status Extend(size_t rounded_bytes);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2);

  // Frees the memory represented by 'h', coalescing the chunk if
  // possible.
  void FreeAndMaybeCoalesce(ChunkHandle h);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h);

  void DumpMemoryLog(size_t num_bytes);

  ChunkHandle AllocateChunk();
  void DeallocateChunk(ChunkHandle h);

  Chunk* ChunkFromHandle(ChunkHandle h);

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> get_bin_debug_info();

  // Structures immutable after construction
  size_t memory_limit_ = 0;
  ArenaExtendStrategy arena_extend_strategy_ = ArenaExtendStrategy::kNextPowerOfTwo;

  int Log2FloorNonZeroSlow(uint64_t n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Returns floor(log2(n)).
  int Log2FloorNonZero(uint64_t n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS)
    unsigned long index;
#if defined(_WIN64)
    _BitScanReverse64(&index, n);
#else
    auto high = static_cast<unsigned long>(n >> 32);
    if (_BitScanReverse(&index, high) > 0) {
      index += 32;
    } else {
      auto low = static_cast<unsigned long>((n << 32) >> 32);
      _BitScanReverse(&index, low);
    }
#endif
    return index;
#else
    return Log2FloorNonZeroSlow(n);
#endif
  }

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }

  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }

  BinNum BinNumForSize(size_t bytes) {
    uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }

  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  // The size of the current region allocation.
  SafeInt<size_t> curr_region_allocation_bytes_;

  std::unique_ptr<IAllocator> device_allocator_;

  mutable OrtMutex lock_;

  RegionManager region_manager_;
  std::vector<Chunk> chunks_;
  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_;

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64_t next_allocation_id_;

  AllocatorStats stats_;

  std::unordered_map<void*, size_t> reserved_chunks_;

  const int initial_chunk_size_bytes_;
  const int max_dead_bytes_per_chunk_;
  const int initial_growth_chunk_size_bytes_;

  // This flag is only relevant if Shrink() is invoked.
  // This is a boolean flag that controls whether the first allocation region
  // is to be considered for shrinkage or not.
  bool consider_first_allocation_region_for_shrinkage_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BFCArena);
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime