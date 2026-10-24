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
// Adapted from onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.cc
// for the CUDA plugin EP arena allocator.

#include "cuda_arena.h"

#include <cassert>
#include <map>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/narrow.h"

namespace onnxruntime {
namespace cuda_plugin {

namespace {
std::string GetAllocatorName(const OrtApi& api, OrtAllocator& allocator) {
  const OrtMemoryInfo* mem_info = allocator.Info(&allocator);
  const char* allocator_name;
  auto* status = api.MemoryInfoGetName(mem_info, &allocator_name);  // never fails
  static_cast<void>(status);
  return allocator_name;
}
}  // namespace

ArenaImpl::ArenaImpl(AllocatorUniquePtr allocator, const ArenaConfig& config, const OrtApi& api,
                     const OrtLogger& logger)
    : device_allocator_{std::move(allocator)},
      allocator_name_{GetAllocatorName(api, *device_allocator_)},
      config_{config},
      next_allocation_id_(1),
      free_chunks_list_(kInvalidChunkHandle),
      api_{api},
      ep_api_{*api_.GetEpApi()},
      logger_{logger} {
  CUDA_ARENA_LOG(INFO, "Creating ArenaImpl for "
                           << allocator_name_
                           << " with following configs: initial_chunk_size_bytes: " << config_.initial_chunk_size_bytes
                           << " max_dead_bytes_per_chunk: " << config_.max_dead_bytes_per_chunk
                           << " initial_growth_chunk_size_bytes: " << config_.initial_growth_chunk_size_bytes
                           << " max_power_of_two_extend_bytes: " << config_.max_power_of_two_extend_bytes
                           << " memory limit: " << config_.max_mem
                           << " arena_extend_strategy: " << config_.arena_extend_strategy);

  curr_region_allocation_bytes_ = RoundedBytes(
      std::min(config_.max_mem, static_cast<size_t>(config_.initial_chunk_size_bytes)));

  stats_.bytes_limit = config.max_mem > static_cast<size_t>(std::numeric_limits<int64_t>::max())
                           ? std::numeric_limits<int64_t>::max()
                           : static_cast<int64_t>(config.max_mem);

  // Create bins of various sizes.
  CUDA_ARENA_LOG(VERBOSE, "Creating " << kNumBins << " bins of max chunk size "
                                      << BinNumToSize(0) << " to " << BinNumToSize(kNumBins - 1));

  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    new (BinFromIndex(b)) Bin(this, bin_size);
    CUDA_ARENA_ENFORCE((BinForSize(bin_size) == BinFromIndex(b) &&
                        BinForSize(bin_size + 255) == BinFromIndex(b) &&
                        BinForSize(bin_size * 2 - 1) == BinFromIndex(b)),
                       "Invalid bin size for bin " << b);

    if (b + 1 < kNumBins) {
      CUDA_ARENA_ENFORCE(BinForSize(bin_size * 2) != BinFromIndex(b), "Invalid bin size for " << b);
    }
  }
}

ArenaImpl::~ArenaImpl() {
  for (const auto& region : region_manager_.regions()) {
    device_allocator_->Free(device_allocator_.get(), region.ptr());
  }

  for (const auto& reserve_chunk : reserved_chunks_) {
    device_allocator_->Free(device_allocator_.get(), reserve_chunk.first);
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

ArenaImpl::Chunk* ArenaImpl::ChunkFromHandle(ChunkHandle h) {
  CUDA_ARENA_ENFORCE(h < chunks_.size(), "ChunkFromHandle");
  return &(chunks_[h]);
}

OrtStatus* ArenaImpl::Extend(size_t rounded_bytes) {
  size_t available_bytes = config_.max_mem - static_cast<size_t>(stats_.total_allocated_bytes);
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  if (rounded_bytes > available_bytes) {
    CUDA_ARENA_RETURN_ERROR(ORT_EP_FAIL, "Available memory of " << available_bytes
                                                                << " is smaller than requested bytes of "
                                                                << rounded_bytes);
  }

  auto safe_alloc = [this](size_t alloc_bytes) {
    void* new_mem = nullptr;
    ORT_TRY {
      new_mem = device_allocator_->Alloc(device_allocator_.get(), alloc_bytes);
    }
    ORT_CATCH(const std::bad_alloc&) {
    }
    return new_mem;
  };

  auto get_extend_bytes = [this, available_bytes](const size_t bytes, size_t& extend_bytes) -> OrtStatus* {
    extend_bytes = 0;
    if (config_.arena_extend_strategy == ArenaExtendStrategy::kNextPowerOfTwo) {
      bool increased_allocation = false;
      while (bytes > curr_region_allocation_bytes_) {
        if (curr_region_allocation_bytes_ > std::numeric_limits<size_t>::max() / 2) {
          // Cannot double without overflow — cap at max.
          curr_region_allocation_bytes_ = std::numeric_limits<size_t>::max();
          break;
        }
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
      }

      extend_bytes = std::min(static_cast<size_t>(curr_region_allocation_bytes_), available_bytes);

      if (!increased_allocation) {
        // Use overflow-safe comparison: double only when the current value
        // is less than half the cap, so the result cannot exceed the cap.
        const size_t max_extend = static_cast<size_t>(config_.max_power_of_two_extend_bytes);
        if (curr_region_allocation_bytes_ < max_extend / 2) {
          curr_region_allocation_bytes_ *= 2;
        } else {
          curr_region_allocation_bytes_ = max_extend;
        }
      }
    } else if (config_.arena_extend_strategy == ArenaExtendStrategy::kSameAsRequested) {
      extend_bytes = bytes;
    } else {
      CUDA_ARENA_RETURN_ERROR(ORT_INVALID_ARGUMENT,
                              "Invalid arena extend strategy." << config_.arena_extend_strategy);
    }

    return nullptr;
  };

  size_t bytes;
  {
    OrtStatus* status = get_extend_bytes(rounded_bytes, bytes);
    if (status != nullptr) return status;
  }

  void* mem_addr = safe_alloc(bytes);

  static constexpr float kBackpedalFactor = 0.9f;
  while (mem_addr == nullptr) {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26451)
#endif
    bytes = RoundedBytes(static_cast<size_t>(bytes * kBackpedalFactor));
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
    if (bytes < rounded_bytes || bytes < 8 * 1024)
      break;

    mem_addr = safe_alloc(bytes);
  }

  if (mem_addr == nullptr) {
    CUDA_ARENA_RETURN_ERROR(ORT_EP_FAIL,
                            "Failed to allocate memory for requested buffer of size " << rounded_bytes);
  }

  CUDA_ARENA_LOG(INFO, "Extended allocation by " << bytes << " bytes.");

  // Guard against leaking mem_addr if any operation below throws (e.g. vector reallocation
  // inside AddAllocationRegion). On success we set mem_addr to nullptr to dismiss the guard.
  struct AllocGuard {
    OrtAllocator* alloc;
    void*& addr;
    ~AllocGuard() {
      if (addr) alloc->Free(alloc, addr);
    }
  } alloc_guard{device_allocator_.get(), mem_addr};

  stats_.total_allocated_bytes += bytes;
  CUDA_ARENA_LOG(INFO, "Total allocated bytes: " << stats_.total_allocated_bytes);
  CUDA_ARENA_LOG(INFO, "Allocated memory at " << mem_addr << " to "
                                              << static_cast<void*>(static_cast<char*>(mem_addr) + bytes));

  region_manager_.AddAllocationRegion(mem_addr, bytes, stats_.num_arena_extensions);
  stats_.num_arena_extensions += 1;

  ChunkHandle h = AllocateChunk();
  Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->stream = nullptr;

  region_manager_.set_handle(c->ptr, h);

  InsertFreeChunkIntoBin(h);

  // All operations completed successfully — dismiss the guard.
  mem_addr = nullptr;

  return nullptr;
}

ArenaImpl::ChunkHandle ArenaImpl::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    *c = Chunk{};
    return h;
  }
  ChunkHandle h = chunks_.size();
  chunks_.resize(h + 1);
  return h;
}

void ArenaImpl::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);

  if (c->stream) {
    if (auto it = stream_to_chunks_.find(c->stream); it != stream_to_chunks_.end()) {
      size_t result = it->second.erase(h);
      static_cast<void>(result);

      if (it->second.empty()) {
        stream_to_chunks_.erase(it);
        impl_to_stream_.erase(ep_api_.SyncStream_GetImpl(c->stream));
      }
    }

    c->stream = nullptr;
    c->stream_sync_id = 0;
  }

  c->quarantined = false;

  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

size_t ArenaImpl::RoundedBytes(size_t bytes) {
  return (kMinAllocationSize * ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
}

void* ArenaImpl::Alloc(size_t size) {
  return AllocateRawInternal(size, nullptr, false);
}

void* ArenaImpl::AllocOnStream(size_t size, OrtSyncStream* stream) {
  return AllocateRawInternal(size, stream, false);
}

void* ArenaImpl::Reserve(size_t size) {
  if (size == 0)
    return nullptr;

  std::lock_guard<std::mutex> lock(lock_);

  if (abandoned_) {
    return nullptr;
  }

  // Check remaining budget before allocating.
  // Use narrow<> to catch truncation (int64_t -> size_t), then avoid overflow
  // by comparing size against the remaining budget rather than summing.
  size_t allocated = 0;
  ORT_TRY {
    allocated = onnxruntime::narrow<size_t>(stats_.total_allocated_bytes);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      CUDA_ARENA_LOG(ERROR, "Reserve: total_allocated_bytes (" << stats_.total_allocated_bytes
                                                               << ") cannot be converted to size_t: " << ex.what());
    });
    return nullptr;
  }
  if (allocated > config_.max_mem || size > config_.max_mem - allocated) {
    CUDA_ARENA_LOG(WARNING, "Reserve of " << size << " bytes would exceed arena max_mem ("
                                          << config_.max_mem << "). Returning nullptr.");
    return nullptr;
  }

  CUDA_ARENA_LOG(INFO, "Reserving memory in ArenaImpl for " << allocator_name_ << " size: " << size);

  void* ptr = device_allocator_->Alloc(device_allocator_.get(), size);
  if (ptr == nullptr) {
    return nullptr;
  }
  CUDA_ARENA_ENFORCE(reserved_chunks_.find(ptr) == reserved_chunks_.end(), __FUNCTION__);
  reserved_chunks_.insert(std::pair<void*, size_t>(ptr, size));
  stats_.bytes_in_use += size;
  stats_.bytes_requested_in_use += size;
  stats_.num_reserves += 1;
  stats_.num_allocs += 1;
  stats_.max_alloc_size = std::max<size_t>(static_cast<size_t>(stats_.max_alloc_size), size);
  stats_.max_bytes_in_use = std::max<int64_t>(static_cast<int64_t>(stats_.max_bytes_in_use), stats_.bytes_in_use);
  stats_.total_allocated_bytes += size;
  return ptr;
}

size_t ArenaImpl::RequestedSize(const void* ptr) {
  std::lock_guard<std::mutex> lock(lock_);
  ChunkHandle h = region_manager_.get_handle(ptr);
  CUDA_ARENA_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);
  Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t ArenaImpl::AllocatedSize(const void* ptr) {
  std::lock_guard<std::mutex> lock(lock_);
  ChunkHandle h = region_manager_.get_handle(ptr);
  CUDA_ARENA_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);
  Chunk* c = ChunkFromHandle(h);
  return c->size;
}

void* ArenaImpl::AllocateRawInternal(size_t num_bytes, OrtSyncStream* stream, bool dump_log_on_failure) {
  if (num_bytes == 0) {
    return nullptr;
  }

  size_t rounded_bytes = RoundedBytes(num_bytes);
  BinNum bin_num = BinNumForSize(rounded_bytes);

  std::lock_guard<std::mutex> lock(lock_);

  if (abandoned_) {
    return nullptr;
  }

  if (stream && stream_to_chunks_.find(stream) == stream_to_chunks_.end()) {
    const OrtSyncStreamImpl* stream_impl = ep_api_.SyncStream_GetImpl(stream);
    assert(stream_impl);
    bool impl_inserted = false;
    ORT_TRY {
      impl_inserted = impl_to_stream_.insert({stream_impl, stream}).second;
      stream_to_chunks_.insert({stream, std::set<size_t>{}});
    }
    ORT_CATCH(...) {
      if (impl_inserted) {
        impl_to_stream_.erase(stream_impl);
      }
      ORT_RETHROW;
    }
  }

  auto* chunk = FindChunkPtr(bin_num, rounded_bytes, num_bytes, stream);

  if (chunk != nullptr) {
    return chunk->ptr;
  }

  CUDA_ARENA_LOG(INFO, "Extending arena for " << allocator_name_
                                              << ". bin_num:" << bin_num
                                              << " (requested) num_bytes: " << num_bytes
                                              << " (actual) rounded_bytes:" << rounded_bytes);

  auto status = Extend(rounded_bytes);
  if (status == nullptr) {
    chunk = FindChunkPtr(bin_num, rounded_bytes, num_bytes, stream);
    if (chunk != nullptr) {
      return chunk->ptr;
    } else {
      status = api_.CreateStatus(ORT_EP_FAIL,
                                 ("Failed to find a free memory block despite calling Extend. rounded_bytes=" +
                                  std::to_string(rounded_bytes))
                                     .c_str());
    }
  }

  if (dump_log_on_failure) {
    CUDA_ARENA_LOG(ERROR, "BFC Arena ran out of memory trying to allocate " << num_bytes);
    DumpMemoryLog(rounded_bytes);
  }

  // Release the OrtStatus and return nullptr instead of throwing — allocate
  // calls must not propagate exceptions across the C API boundary.
  api_.ReleaseStatus(status);
  return nullptr;
}

OrtStatus* ArenaImpl::GetStats(OrtKeyValuePairs** stats) {
  std::lock_guard<std::mutex> lock(lock_);

  api_.CreateKeyValuePairs(stats);
  stats_.ToKeyValuePairs(api_, *stats);

  if (device_allocator_->GetStats != nullptr) {
    OrtKeyValuePairs* device_stats = nullptr;
    auto* status = device_allocator_->GetStats(device_allocator_.get(), &device_stats);
    if (status != nullptr) {
      api_.ReleaseKeyValuePairs(*stats);
      *stats = nullptr;
      return status;
    }
    for (const char* key : {"DeviceLimit", "DeviceCommitted", "DevicePending", "DevicePeak"}) {
      const char* value = api_.GetKeyValue(device_stats, key);
      if (value != nullptr) api_.AddKeyValuePair(*stats, key, value);
    }
    api_.ReleaseKeyValuePairs(device_stats);
  }

  return nullptr;
}

OrtStatus* ArenaImpl::Shrink() {
  std::lock_guard<std::mutex> lock(lock_);

  if (abandoned_) {
    return nullptr;
  }
  shrink_enabled_.store(true, std::memory_order_release);

  // Note: Reserved memory (via Reserve()) is allocated directly through the device
  // allocator and stored in reserved_chunks_, bypassing the region/chunk system.
  // Shrink() intentionally does NOT free reserved memory because it is used for
  // model initializers that must remain valid for the session lifetime.

  // Snapshot region pointers/sizes before mutation — we will modify the
  // region list while iterating.  Matches in-tree BFCArena::Shrink().
  const auto num_regions = region_manager_.regions().size();
  InlinedVector<void*> region_ptrs;
  InlinedVector<size_t> region_sizes;
  region_ptrs.reserve(num_regions);
  region_sizes.reserve(num_regions);

  for (const auto& region : region_manager_.regions()) {
    region_ptrs.push_back(region.ptr());
    region_sizes.push_back(region.memory_size());
  }

  // For each region, check if every chunk is free. If so, deallocate the region.
  size_t i = 0;
  for (void* region_ptr : region_ptrs) {
    bool deallocate_region = true;
    ChunkHandle region_begin_chunk = region_manager_.get_handle(region_ptr);
    ChunkHandle h = region_begin_chunk;
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use() || c->quarantined) {
        // at-least one used chunk found in the allocation region -
        // so we cannot deallocate it
        deallocate_region = false;
        break;
      }
      h = c->next;
    }

    if (deallocate_region) {
      auto shrink_size = region_sizes[i];
      stats_.num_arena_shrinkages += 1;
      stats_.total_allocated_bytes -= static_cast<int64_t>(shrink_size);

      CUDA_ARENA_LOG(VERBOSE, allocator_name_ << " ArenaImpl shrunk by "
                                              << shrink_size << " bytes. "
                                              << "Total allocated is now " << stats_.total_allocated_bytes);

      h = region_begin_chunk;
      ChunkHandle temp = region_begin_chunk;
      while (h != kInvalidChunkHandle) {
        const Chunk* c = ChunkFromHandle(h);
        temp = c->next;
        RemoveFreeChunkFromBin(h);
        DeleteChunk(h);
        h = temp;
      }

      device_allocator_->Free(device_allocator_.get(), region_ptr);
      region_manager_.RemoveAllocationRegion(region_ptr);
      stats_.num_arena_extensions--;
    }

    ++i;
  }

  // Reset growth so the arena can grow fresh if needed later.
  // Matches BFCArena which resets to initial_growth_chunk_size_bytes_.
  curr_region_allocation_bytes_ = RoundedBytes(
      static_cast<size_t>(config_.initial_growth_chunk_size_bytes));

  return nullptr;
}

ArenaImpl::Chunk* ArenaImpl::SplitFreeChunkFromBin(Bin::FreeChunkSet* free_chunks,
                                                   const Bin::FreeChunkSet::iterator& citer,
                                                   size_t rounded_bytes,
                                                   size_t num_bytes) {
  const ChunkHandle h = (*citer);
  Chunk* chunk = ChunkFromHandle(h);
  const BinNum original_bin_num = chunk->bin_num;
  auto free_chunk_node = free_chunks->extract(citer);
  chunk->bin_num = kInvalidBinNum;

  if (chunk->size >= rounded_bytes * 2 ||
      static_cast<int64_t>(chunk->size - rounded_bytes) >= config_.max_dead_bytes_per_chunk) {
    ORT_TRY {
      SplitChunk(h, rounded_bytes);
    }
    ORT_CATCH(...) {
      const auto insert_result = free_chunks->insert(std::move(free_chunk_node));
      CUDA_ARENA_ENFORCE(insert_result.inserted, "Could not restore chunk to bin");
      ChunkFromHandle(h)->bin_num = original_bin_num;
      ORT_RETHROW;
    }
    chunk = ChunkFromHandle(h);
  }

  chunk->requested_size = num_bytes;
  chunk->allocation_id = next_allocation_id_++;

  ++stats_.num_allocs;
  stats_.bytes_in_use += chunk->size;
  stats_.bytes_requested_in_use += num_bytes;
  stats_.max_bytes_in_use = std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
  stats_.max_alloc_size = std::max<int64_t>(stats_.max_alloc_size, static_cast<int64_t>(chunk->size));

  return chunk;
}

ArenaImpl::Chunk* ArenaImpl::FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                                          OrtSyncStream* stream) {
  for (; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end(); ++citer) {
      const ChunkHandle h = (*citer);
      Chunk* chunk = ChunkFromHandle(h);
      CUDA_ARENA_ENFORCE(!chunk->in_use(), __FUNCTION__);

      if (chunk->quarantined) {
        continue;
      }

      if (chunk->size >= rounded_bytes) {
        bool safe_to_use = chunk->stream == stream ||
                           !chunk->stream ||
                           (stream && chunk->stream &&
                            chunk->stream_sync_id < ep_api_.GetSyncIdForLastWaitOnSyncStream(chunk->stream, stream));

        if (safe_to_use) {
          bool inserted_destination_mapping = false;
          if (stream) {
            inserted_destination_mapping = stream_to_chunks_.at(stream).insert(h).second;
          }

          ORT_TRY {
            chunk = SplitFreeChunkFromBin(&b->free_chunks, citer, rounded_bytes, num_bytes);
          }
          ORT_CATCH(...) {
            if (inserted_destination_mapping) {
              stream_to_chunks_.at(stream).erase(h);
            }
            ORT_RETHROW;
          }

          if (stream) {
            if (chunk->stream && chunk->stream != stream) {
              OrtSyncStream* previous_stream = chunk->stream;
              auto previous_it = stream_to_chunks_.find(previous_stream);
              if (previous_it != stream_to_chunks_.end()) {
                previous_it->second.erase(h);
                if (previous_it->second.empty()) {
                  stream_to_chunks_.erase(previous_it);
                  impl_to_stream_.erase(ep_api_.SyncStream_GetImpl(previous_stream));
                }
              }
            }
            chunk->stream = stream;
            chunk->stream_sync_id = ep_api_.SyncStream_GetSyncId(stream);
          }

          return chunk;
        }
      }
    }
  }

  return nullptr;
}

void ArenaImpl::SplitChunk(ChunkHandle h, size_t num_bytes) {
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  CUDA_ARENA_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);

  Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->stream = c->stream;
  new_chunk->stream_sync_id = c->stream_sync_id;
  new_chunk->quarantined = c->quarantined;
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  new_chunk->size = c->size - num_bytes;
  new_chunk->allocation_id = -1;

  bool region_handle_set = false;
  ORT_TRY {
    // Track the remainder chunk's stream assignment so ResetChunksUsingStream
    // can clear it later. Without this, the free remainder retains a stale
    // stream pointer after the stream is released — risking use-after-free
    // in GetSyncIdForLastWaitOnSyncStream.
    if (new_chunk->stream) {
      stream_to_chunks_.at(new_chunk->stream).insert(h_new_chunk);
    }
    region_manager_.set_handle(new_chunk->ptr, h_new_chunk);
    region_handle_set = true;
    InsertFreeChunkIntoBin(h_new_chunk);
  }
  ORT_CATCH(...) {
    if (region_handle_set) {
      region_manager_.erase(new_chunk->ptr);
    }
    DeallocateChunk(h_new_chunk);
    ORT_RETHROW;
  }

  ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->size = num_bytes;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }
}

void ArenaImpl::Free(void* p) {
  if (p == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(lock_);
  if (abandoned_) {
    return;
  }
  auto it = reserved_chunks_.find(p);
  if (it != reserved_chunks_.end()) {
    device_allocator_->Free(device_allocator_.get(), it->first);
    stats_.bytes_in_use -= it->second;
    stats_.bytes_requested_in_use -= it->second;
    stats_.total_allocated_bytes -= it->second;
    reserved_chunks_.erase(it);
  } else {
    DeallocateRawInternal(p);
  }
}

void ArenaImpl::DeallocateRawInternal(void* ptr) {
  ChunkHandle h = region_manager_.get_handle(ptr);
  CUDA_ARENA_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);
  FreeAndMaybeCoalesce(h);
}

void ArenaImpl::Merge(ChunkHandle h1, ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  CUDA_ARENA_ENFORCE(!c1->in_use() && !c2->in_use() &&
                         c1->stream == c2->stream && c1->quarantined == c2->quarantined,
                     __FUNCTION__);

  ChunkHandle h3 = c2->next;
  c1->next = h3;
  CUDA_ARENA_ENFORCE(c2->prev == h1, __FUNCTION__);
  if (h3 != kInvalidChunkHandle) {
    Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  c1->size += c2->size;

  assert(c1->stream == c2->stream);
  c1->stream_sync_id = std::max(c1->stream_sync_id, c2->stream_sync_id);

  DeleteChunk(h2);
}

void ArenaImpl::DeleteChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void ArenaImpl::InsertFreeChunkIntoBin(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CUDA_ARENA_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  const auto [_, inserted] = new_bin->free_chunks.insert(h);
  CUDA_ARENA_ENFORCE(inserted, "Could not insert chunk into bin");
  c->bin_num = bin_num;
}

void ArenaImpl::RemoveFreeChunkFromBin(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CUDA_ARENA_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum), __FUNCTION__);
  CUDA_ARENA_ENFORCE(BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0, "Could not find chunk in bin");
  c->bin_num = kInvalidBinNum;
}

void ArenaImpl::FreeAndMaybeCoalesce(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CUDA_ARENA_ENFORCE(c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);

  c->allocation_id = -1;
  stats_.bytes_in_use -= c->size;
  stats_.bytes_requested_in_use -= c->requested_size;

  ChunkHandle chunk_to_reassign = Coalesce(h);
  InsertFreeChunkIntoBin(chunk_to_reassign);
}

ArenaImpl::ChunkHandle ArenaImpl::Coalesce(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CUDA_ARENA_ENFORCE(!c->in_use(), __FUNCTION__);

  ChunkHandle chunk_to_reassign = h;

  if (c->next != kInvalidChunkHandle) {
    Chunk* cnext = ChunkFromHandle(c->next);
    if (!cnext->in_use() && cnext->stream == c->stream && cnext->quarantined == c->quarantined) {
      chunk_to_reassign = h;
      RemoveFreeChunkFromBin(c->next);
      Merge(h, ChunkFromHandle(h)->next);
    }
  }

  c = ChunkFromHandle(h);
  if (c->prev != kInvalidChunkHandle) {
    Chunk* cprev = ChunkFromHandle(c->prev);
    if (!cprev->in_use() && cprev->stream == c->stream && cprev->quarantined == c->quarantined) {
      chunk_to_reassign = c->prev;
      RemoveFreeChunkFromBin(c->prev);
      Merge(ChunkFromHandle(h)->prev, h);
    }
  }

  return chunk_to_reassign;
}

std::array<ArenaImpl::BinDebugInfo, ArenaImpl::kNumBins> ArenaImpl::GetBinDebugInfo() {
  std::array<BinDebugInfo, kNumBins> bin_infos;

  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;

      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        CUDA_ARENA_ENFORCE(bin->free_chunks.count(h) == 1 && c->bin_num == bin_num, __FUNCTION__);
      }

      h = c->next;
    }
  }
  return bin_infos;
}

void ArenaImpl::DumpMemoryLog(size_t num_bytes) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = GetBinDebugInfo();
  CUDA_ARENA_LOG(INFO, "Allocator:" << allocator_name_);
  CUDA_ARENA_LOG(INFO, "Bin size: Chunks in_use/total (if not zero). Allocated bytes in_use/total. Requested bytes.");

  size_t waste = 0;
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    CUDA_ARENA_ENFORCE(b->free_chunks.size() == bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use,
                       __FUNCTION__);

    if (bin_info.total_chunks_in_bin > 0) {
      CUDA_ARENA_LOG(INFO, b->bin_size
                               << ": Chunks " << bin_info.total_chunks_in_use << "/" << bin_info.total_chunks_in_bin
                               << ". Bytes "
                               << bin_info.total_bytes_in_use << "/" << bin_info.total_bytes_in_bin << ". "
                               << "Requested " << bin_info.total_requested_bytes_in_use << ".");

      waste += bin_info.total_bytes_in_use - bin_info.total_requested_bytes_in_use;
    }
  }

  if (waste > 0) {
    CUDA_ARENA_LOG(INFO, "Diff between in-use and requested bytes is " << waste);
  }

  Bin* b = BinForSize(num_bytes);

  CUDA_ARENA_LOG(INFO, "Bin for " << num_bytes
                                  << " bytes has max bytes of " << b->bin_size
                                  << ", Chunk State: ");

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    CUDA_ARENA_LOG(INFO, "  " << c->DebugString(this, true));
  }

  CUDA_ARENA_LOG(INFO, "Overall chunks summary:");
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      CUDA_ARENA_LOG(INFO, (c->in_use() ? "  Chunk" : "  Free ")
                               << " at " << c->ptr << " of size " << c->size);
      h = c->next;
    }
  }

  CUDA_ARENA_LOG(INFO, "Summary of in-use chunks by size: ");
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    CUDA_ARENA_LOG(INFO, "  " << it.second << " chunks of size " << it.first
                              << ". Total " << it.first * it.second);
    total_bytes += (it.first * it.second);
  }

  CUDA_ARENA_LOG(INFO, "Sum Total of in-use chunks: " << total_bytes);
  CUDA_ARENA_LOG(INFO, "Stats: \n"
                           << stats_.DebugString());
}

OrtStatus* ArenaImpl::ResetChunksUsingStream(const OrtSyncStreamImpl* stream_impl) {
  std::lock_guard<std::mutex> lock(lock_);

  auto impl_it = impl_to_stream_.find(stream_impl);
  if (impl_it == impl_to_stream_.end()) {
    return nullptr;  // stream hasn't been used with this arena
  }

  const OrtSyncStream* stream = impl_it->second;

  auto it = stream_to_chunks_.find(stream);
  if (it != stream_to_chunks_.end()) {
    const auto& chunk_handles = it->second;
    for (size_t handle : chunk_handles) {
      Chunk* c = ChunkFromHandle(handle);
      assert(c->stream == stream);
      c->stream = nullptr;
    }

    stream_to_chunks_.erase(it);
    impl_to_stream_.erase(stream_impl);
  }

  // Coalesce free chunks after clearing stream assignments.
  // Coalesce returns the (possibly different) handle of the merged chunk,
  // so we must use that handle for the remainder of the iteration.
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (!c->in_use()) {
        RemoveFreeChunkFromBin(h);
        h = Coalesce(h);
        c = ChunkFromHandle(h);
        InsertFreeChunkIntoBin(h);
      }
      h = c->next;
    }
  }

  return nullptr;
}

OrtStatus* ArenaImpl::QuarantineChunksUsingStream(const OrtSyncStreamImpl* stream_impl, bool& quarantined) {
  std::lock_guard<std::mutex> lock(lock_);
  quarantined = false;

  auto impl_it = impl_to_stream_.find(stream_impl);
  if (impl_it == impl_to_stream_.end()) {
    return nullptr;
  }

  const OrtSyncStream* stream = impl_it->second;
  auto chunks_it = stream_to_chunks_.find(stream);
  if (chunks_it != stream_to_chunks_.end()) {
    quarantined = !chunks_it->second.empty();
    for (size_t handle : chunks_it->second) {
      Chunk* chunk = ChunkFromHandle(handle);
      assert(chunk->stream == stream);
      chunk->stream = nullptr;
      chunk->stream_sync_id = 0;
      chunk->quarantined = true;
    }
    stream_to_chunks_.erase(chunks_it);
  }

  impl_to_stream_.erase(impl_it);
  return nullptr;
}

void ArenaImpl::Abandon() {
  std::lock_guard<std::mutex> lock(lock_);
  abandoned_ = true;
}

// CudaArenaAllocator factory method
/*static*/
OrtStatus* CudaArenaAllocator::Create(CudaAllocatorKind kind,
                                      const OrtMemoryInfo* memory_info,
                                      AllocatorUniquePtr raw_allocator,
                                      const OrtKeyValuePairs* options,
                                      const OrtApi& api,
                                      const OrtLogger& logger,
                                      std::unique_ptr<CudaArenaAllocator>& out) {
  ArenaConfig config = options ? ArenaConfig::FromKeyValuePairs(api, *options) : ArenaConfig{};
  if (!config.IsValid()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid CUDA arena allocator configuration.");
  }
  auto impl = std::make_unique<ArenaImpl>(std::move(raw_allocator), config, api, logger);
  out = std::make_unique<CudaArenaAllocator>(kind, memory_info, std::move(impl));
  return nullptr;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
