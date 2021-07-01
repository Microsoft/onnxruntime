// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/bfc_arena.h"
#include <type_traits>

namespace onnxruntime {
BFCArena::BFCArena(std::unique_ptr<IAllocator> resource_allocator,
                   size_t total_memory,
                   ArenaExtendStrategy arena_extend_strategy,
                   int initial_chunk_size_bytes,
                   int max_dead_bytes_per_chunk,
                   int initial_growth_chunk_size_bytes)
    : IAllocator(OrtMemoryInfo(resource_allocator->Info().name,
                               OrtAllocatorType::OrtArenaAllocator,
                               resource_allocator->Info().device,
                               resource_allocator->Info().id,
                               resource_allocator->Info().mem_type)),
      device_allocator_(std::move(resource_allocator)),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1),
      initial_chunk_size_bytes_(initial_chunk_size_bytes),
      max_dead_bytes_per_chunk_(max_dead_bytes_per_chunk),
      initial_growth_chunk_size_bytes_(initial_growth_chunk_size_bytes) {
  LOGS_DEFAULT(INFO) << "Creating BFCArena for " << device_allocator_->Info().name
                     << " with following configs: initial_chunk_size_bytes: " << initial_chunk_size_bytes_
                     << " max_dead_bytes_per_chunk: " << max_dead_bytes_per_chunk_
                     << " initial_growth_chunk_size_bytes: " << initial_growth_chunk_size_bytes_
                     << " memory limit: " << total_memory
                     << " arena_extend_strategy: " << static_cast<int32_t>(arena_extend_strategy);

  // static_cast<std::underlying_type_t<ArenaExtendStrategy>>(arena_extend_strategy); doesn't work on this compiler

  curr_region_allocation_bytes_ = RoundedBytes(std::min(total_memory, static_cast<size_t>(initial_chunk_size_bytes_)));
  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  stats_.bytes_limit = static_cast<int64_t>(total_memory);

  arena_extend_strategy_ = arena_extend_strategy;

  // We never want to shrink the initial allocation if the arena extend strategy is kNextPowerOfTwo.
  // This could seem confusingly arbitrary but the rationale is as follows:
  // The user selected initial allocation chunk is only valid for the arena extend strategy kNextPowerOfTwo
  // and the user has likely chosen this initial value so that any ad-hoc arena extensions/shrinkages could potentially
  // be avoided. So we do not consider the initial allocation for shrinkage whatever its usage status.
  // On the other hand, if the arena extension strategy is kSameAsRequested, any initial chunk set by the user or otherwise,
  // is moot and the arena will only extend based on the request size. In these cases, we consider any allocation for shrinkage
  // if it is left unused (even if it is the first allocation).
  if (arena_extend_strategy_ == ArenaExtendStrategy::kSameAsRequested) {
    // Consider all allocation regions (including first allocation region) for shrinkage
    consider_first_allocation_region_for_shrinkage_ = true;
  } else {  // arena_extend_strategy_ == kNextPowerOfTwo
    // Do not consider the first allocation region for shrinkage
    consider_first_allocation_region_for_shrinkage_ = false;
  }
  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  LOGS_DEFAULT(VERBOSE) << "Creating " << kNumBins << " bins of max chunk size "
                        << BinNumToSize(0) << " to " << BinNumToSize(kNumBins - 1);
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    new (BinFromIndex(b)) Bin(this, bin_size);
    ORT_ENFORCE(BinForSize(bin_size) == BinFromIndex(b));
    ORT_ENFORCE(BinForSize(bin_size + 255) == BinFromIndex(b));
    ORT_ENFORCE(BinForSize(bin_size * 2 - 1) == BinFromIndex(b));
    if (b + 1 < kNumBins) {
      ORT_ENFORCE(BinForSize(bin_size * 2) != BinFromIndex(b));
    }
  }
}

BFCArena::~BFCArena() {
  for (const auto& region : region_manager_.regions()) {
    device_allocator_->Free(region.ptr());
  }

  for (const auto& reserve_chunk : reserved_chunks_) {
    device_allocator_->Free(reserve_chunk.first);
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

BFCArena::Chunk* BFCArena::ChunkFromHandle(ChunkHandle h) {
  ORT_ENFORCE(h < chunks_.size());
  return &(chunks_[h]);
}

Status BFCArena::Extend(size_t rounded_bytes) {
  size_t available_bytes = memory_limit_ - static_cast<size_t>(stats_.total_allocated_bytes);
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Available memory of ", available_bytes,
                           " is smaller than requested bytes of ", rounded_bytes);
  }

  auto safe_alloc = [this](size_t alloc_bytes) {
    void* new_mem = nullptr;
    ORT_TRY {
      new_mem = device_allocator_->Alloc(alloc_bytes);
    }
    ORT_CATCH(const std::bad_alloc&) {
      // attempted allocation can throw std::bad_alloc. we want to treat this the same as if it returned nullptr
      // so swallow the exception
    }
    ORT_CATCH(const OnnxRuntimeException& ort_exception) {
      // swallow if exception is our throw from a failed cudaMalloc call.
      // re-throw otherwise.
      ORT_HANDLE_EXCEPTION([&ort_exception]() {
        if (std::string(ort_exception.what()).find("cudaMalloc") == std::string::npos &&
            std::string(ort_exception.what()).find("hipMalloc") == std::string::npos) {
          ORT_RETHROW;
        }
      });
    }
    return new_mem;
  };

  auto get_extend_bytes = [this, available_bytes](const size_t bytes) -> size_t {
    size_t extend_bytes = 0;
    if (arena_extend_strategy_ == ArenaExtendStrategy::kNextPowerOfTwo) {
      // If curr_region_allocation_bytes_ is not enough to satisfy the
      // allocation, keep multiplying by a power of two until that is
      // sufficient.
      bool increased_allocation = false;
      while (bytes > curr_region_allocation_bytes_) {
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
      }

      extend_bytes = std::min(static_cast<size_t>(curr_region_allocation_bytes_), available_bytes);

      // we allocated the same number of bytes as the current region
      // the 2x is to double the minimum size of the next amount we'll allocate
      if (!increased_allocation) {
        curr_region_allocation_bytes_ *= 2;
      }
    } else if (arena_extend_strategy_ == ArenaExtendStrategy::kSameAsRequested) {
      // BFC Arena could cause internal and external fragmentation. But, running training with
      // big batch size will be very sensitive to fragmentation. So, to avoid fragmentation,
      // just extend arena with actual requested size.
      extend_bytes = bytes;
    } else {
      ORT_THROW("Incorrect arena extend strategy.", static_cast<int32_t>(arena_extend_strategy_));
    }

    return extend_bytes;
  };

  size_t bytes = get_extend_bytes(rounded_bytes);
  // Try allocating.
  void* mem_addr = safe_alloc(bytes);

  static constexpr float kBackpedalFactor = 0.9f;
  // Try allocating less memory.
  while (mem_addr == nullptr) {
    bytes = RoundedBytes(static_cast<size_t>(bytes * kBackpedalFactor));

    // give up if we can't satisfy the requested size, or we're attempting an allocation of less than 8K.
    //
    // the latter protects against an infinite loop that occurs when bytes is less than 2560. at that point the 10%
    // reduction to 2304 bytes is undone by rounding to a 256 boundary in RoundedBytes, leading to an infinite loop.
    // the 8K value is just to give up a little earlier vs. getting all the way down to 2560 bytes.
    // If we can't allocate 8K, we're pretty much dead.
    if (bytes < rounded_bytes || bytes < 8 * 1024)
      break;

    mem_addr = safe_alloc(bytes);
  }

  if (mem_addr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to allocate memory for requested buffer of size ", rounded_bytes);
  }

  LOGS_DEFAULT(INFO) << "Extended allocation by " << bytes << " bytes.";

  stats_.total_allocated_bytes += bytes;
  LOGS_DEFAULT(INFO) << "Total allocated bytes: "
                     << stats_.total_allocated_bytes;

  LOGS_DEFAULT(INFO) << "Allocated memory at " << mem_addr << " to "
                     << static_cast<void*>(static_cast<char*>(mem_addr) + bytes);
  region_manager_.AddAllocationRegion(mem_addr, bytes, stats_.num_arena_extensions);
  stats_.num_arena_extensions += 1;

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCArena::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;

  region_manager_.set_handle(c->ptr, h);

  // TODO(vrv): Try to merge this new region with an existing region,
  // if the address space is contiguous, to avoid fragmentation
  // across regions.

  // Insert the chunk into the right bin.
  InsertFreeChunkIntoBin(h);

  return Status::OK();
}

BFCArena::ChunkHandle BFCArena::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  }
  ChunkHandle h = chunks_.size();
  chunks_.resize(h + 1);
  return h;
}

void BFCArena::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

// static
size_t BFCArena::RoundedBytes(size_t bytes) {
  size_t rounded_bytes =
      (kMinAllocationSize *
       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
  ORT_ENFORCE(size_t{0} == rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

void* BFCArena::Alloc(size_t size) {
  return AllocateRawInternal(size, false);
}

void* BFCArena::Reserve(size_t size) {
  if (size == 0)
    return nullptr;

  std::lock_guard<OrtMutex> lock(lock_);

  LOGS_DEFAULT(INFO) << "Reserving memory in BFCArena for " << device_allocator_->Info().name << " size: " << size;

  void* ptr = device_allocator_->Alloc(size);
  ORT_ENFORCE(reserved_chunks_.find(ptr) == reserved_chunks_.end());
  reserved_chunks_.insert(std::pair<void*, size_t>(ptr, size));
  stats_.bytes_in_use += size;
  stats_.num_reserves += 1;
  stats_.num_allocs += 1;
  stats_.max_alloc_size = std::max<size_t>(static_cast<size_t>(stats_.max_alloc_size), size);
  stats_.max_bytes_in_use = std::max<int64_t>(static_cast<int64_t>(stats_.max_bytes_in_use), stats_.bytes_in_use);
  stats_.total_allocated_bytes += size;
  return ptr;
}

size_t BFCArena::RequestedSize(const void* ptr) {
  std::lock_guard<OrtMutex> lock(lock_);
  BFCArena::ChunkHandle h = region_manager_.get_handle(ptr);
  ORT_ENFORCE(h != kInvalidChunkHandle);
  BFCArena::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t BFCArena::AllocatedSize(const void* ptr) {
  std::lock_guard<OrtMutex> lock(lock_);
  BFCArena::ChunkHandle h = region_manager_.get_handle(ptr);
  ORT_ENFORCE(h != kInvalidChunkHandle);
  BFCArena::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

void* BFCArena::AllocateRawInternal(size_t num_bytes,
                                    bool dump_log_on_failure) {
  if (num_bytes == 0) {
    LOGS_DEFAULT(VERBOSE) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  std::lock_guard<OrtMutex> lock(lock_);
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
  if (ptr != nullptr) {
    return ptr;
  }

  LOGS_DEFAULT(INFO) << "Extending BFCArena for " << device_allocator_->Info().name
                     << ". bin_num:" << bin_num << " (requested) num_bytes: " << num_bytes << " (actual) rounded_bytes:" << rounded_bytes;

  // Try to extend
  auto status = Extend(rounded_bytes);
  if (status.IsOK()) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
    if (ptr != nullptr) {
      return ptr;
    } else {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to find a free memory block despite calling Extend. rounded_bytes=",
                               rounded_bytes);
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    LOGS_DEFAULT(ERROR) << "BFC Arena ran out of memory trying to allocate " << num_bytes
                        << ".  Current allocation summary follows.";
    DumpMemoryLog(rounded_bytes);
  }

  ORT_THROW(status.ErrorMessage());
}

void BFCArena::GetStats(AllocatorStats* stats) {
  std::lock_guard<OrtMutex> lock(lock_);
  *stats = stats_;
}

void* BFCArena::FindChunkPtr(BinNum bin_num, size_t rounded_bytes,
                             size_t num_bytes) {
  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const BFCArena::ChunkHandle h = (*citer);
      BFCArena::Chunk* chunk = ChunkFromHandle(h);
      ORT_ENFORCE(!chunk->in_use());
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do so.  In any case don't waste more than
        // max_dead_bytes_per_chunk bytes on padding this alloc.
        if (chunk->size >= rounded_bytes * 2 ||
            static_cast<int64_t>(chunk->size) - static_cast<int64_t>(rounded_bytes) >= max_dead_bytes_per_chunk_) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;
        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        stats_.max_bytes_in_use =
            std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
        stats_.max_alloc_size =
            std::max<int64_t>(stats_.max_alloc_size, static_cast<int64_t>(chunk->size));
        return chunk->ptr;
      }
    }
  }
  return nullptr;
}

void BFCArena::SplitChunk(BFCArena::ChunkHandle h, size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  ORT_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCArena::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCArena::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void BFCArena::Free(void* p) {
  if (p == nullptr) {
    return;
  }
  std::lock_guard<OrtMutex> lock(lock_);
  auto it = reserved_chunks_.find(p);
  if (it != reserved_chunks_.end()) {
    device_allocator_->Free(it->first);
    stats_.bytes_in_use -= it->second;
    stats_.total_allocated_bytes -= it->second;
    reserved_chunks_.erase(it);
  } else {
    DeallocateRawInternal(p);
  }
}

Status BFCArena::Shrink() {
  std::lock_guard<OrtMutex> lock(lock_);
  auto num_regions = region_manager_.regions().size();
  std::vector<void*> region_ptrs;
  std::vector<size_t> region_sizes;
  region_ptrs.reserve(num_regions);
  region_sizes.reserve(num_regions);

  for (const auto& region : region_manager_.regions()) {
    if (consider_first_allocation_region_for_shrinkage_ || region.id() != 0) {
      region_ptrs.push_back(region.ptr());
      region_sizes.push_back(region.memory_size());
    }
  }

  size_t i = 0;
  for (void* region_ptr : region_ptrs) {
    bool deallocate_region = true;
    ChunkHandle region_begin_chunk = region_manager_.get_handle(region_ptr);
    ChunkHandle h = region_begin_chunk;
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
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
      stats_.total_allocated_bytes -= shrink_size;

      LOGS_DEFAULT(VERBOSE) << device_allocator_->Info().name << " BFC Arena shrunk by "
                            << shrink_size << " bytes. "
                            << " The total allocated bytes is now " << stats_.total_allocated_bytes;

      h = region_begin_chunk;
      ChunkHandle temp = region_begin_chunk;
      while (h != kInvalidChunkHandle) {
        const Chunk* c = ChunkFromHandle(h);
        temp = c->next;
        RemoveFreeChunkFromBin(h);
        DeleteChunk(h);
        h = temp;
      }

      device_allocator_->Free(region_ptr);
      region_manager_.RemoveAllocationRegion(region_ptr);
    }

    ++i;
  }

  // Will affect how the arena grows if the arena extend strategy is kNextPowerOfTwo
  // In case the extend strategy is kSameAsRequested, the arena growth is exactly the size of the memory request itself
  curr_region_allocation_bytes_ = initial_growth_chunk_size_bytes_;

  return Status::OK();
}

void BFCArena::DeallocateRawInternal(void* ptr) {
  // Find the chunk from the ptr.
  BFCArena::ChunkHandle h = region_manager_.get_handle(ptr);
  ORT_ENFORCE(h != kInvalidChunkHandle);

  // Consider coalescing it.
  FreeAndMaybeCoalesce(h);
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void BFCArena::Merge(BFCArena::ChunkHandle h1,
                     BFCArena::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  ORT_ENFORCE(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCArena::ChunkHandle h3 = c2->next;
  c1->next = h3;
  ORT_ENFORCE(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCArena::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  DeleteChunk(h2);
}

void BFCArena::DeleteChunk(ChunkHandle h) {
  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  //  VLOG(4) << "Removing: " << c->ptr;
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void BFCArena::InsertFreeChunkIntoBin(BFCArena::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  ORT_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum));
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void BFCArena::RemoveFreeChunkIterFromBin(
    BFCArena::Bin::FreeChunkSet* free_chunks,
    const BFCArena::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  ORT_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void BFCArena::RemoveFreeChunkFromBin(BFCArena::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  ORT_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum));
  ORT_ENFORCE(BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0,
              "Could not find chunk in bin");
  c->bin_num = kInvalidBinNum;
}

void BFCArena::FreeAndMaybeCoalesce(BFCArena::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  ORT_ENFORCE(c->in_use() && (c->bin_num == kInvalidBinNum));

  // Mark the chunk as no longer in use
  c->allocation_id = -1;

  // Updates the stats.
  stats_.bytes_in_use -= c->size;

  // This chunk is no longer in-use, consider coalescing the chunk
  // with adjacent chunks.
  ChunkHandle chunk_to_reassign = h;

  // If the next chunk is free, coalesce the two
  if (c->next != kInvalidChunkHandle) {
    Chunk* cnext = ChunkFromHandle(c->next);
    if (!cnext->in_use()) {
      //      VLOG(8) << "Chunk at " << cnext->ptr << " merging with c " <<
      //      c->ptr;

      chunk_to_reassign = h;

      // Deletes c->next
      RemoveFreeChunkFromBin(c->next);
      Merge(h, ChunkFromHandle(h)->next);
    }
  }

  // If the previous chunk is free, coalesce the two
  c = ChunkFromHandle(h);
  if (c->prev != kInvalidChunkHandle) {
    Chunk* cprev = ChunkFromHandle(c->prev);
    if (!cprev->in_use()) {
      //      VLOG(8) << "Chunk at " << c->ptr << " merging into c->prev "
      //       << cprev->ptr;

      chunk_to_reassign = c->prev;

      // Deletes c
      RemoveFreeChunkFromBin(c->prev);
      Merge(ChunkFromHandle(h)->prev, h);
    }
  }

  InsertFreeChunkIntoBin(chunk_to_reassign);
}

std::array<BFCArena::BinDebugInfo, BFCArena::kNumBins>
BFCArena::get_bin_debug_info() {
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
        ORT_ENFORCE(bin->free_chunks.count(h) == 1);
        ORT_ENFORCE(c->bin_num == bin_num);
      }
      h = c->next;
    }
  }
  return bin_infos;
}

void BFCArena::DumpMemoryLog(size_t num_bytes) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  LOGS_DEFAULT(INFO) << "Allocator:" << device_allocator_->Info().name;
  LOGS_DEFAULT(INFO) << "Bin size: Chunks in_use/total (if not zero). Allocated bytes in_use/total. Requested bytes.";

  size_t waste = 0;
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    ORT_ENFORCE(b->free_chunks.size() ==
                bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

    if (bin_info.total_chunks_in_bin > 0) {
      LOGS_DEFAULT(INFO) << b->bin_size
                         << ": Chunks " << bin_info.total_chunks_in_use << "/" << bin_info.total_chunks_in_bin
                         << ". Bytes "
                         << bin_info.total_bytes_in_use << "/" << bin_info.total_bytes_in_bin << ". "
                         << "Requested " << bin_info.total_requested_bytes_in_use << ".";

      waste += bin_info.total_bytes_in_use - bin_info.total_requested_bytes_in_use;
    }
  }

  if (waste > 0) {
    LOGS_DEFAULT(INFO) << "Diff between in-use and requested bytes is " << waste;
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOGS_DEFAULT(INFO) << "Bin for " << num_bytes
                     << " bytes has max bytes of " << b->bin_size
                     << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOGS_DEFAULT(INFO) << "  " << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  LOGS_DEFAULT(INFO) << "Overall chunks summary:";
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      LOGS_DEFAULT(INFO) << (c->in_use() ? "  Chunk" : "  Free ") << " at " << c->ptr
                         << " of size " << c->size;
      h = c->next;
    }
  }

  LOGS_DEFAULT(INFO) << "Summary of in-use chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOGS_DEFAULT(INFO) << "  " << it.second << " chunks of size " << it.first
                       << ". Total " << it.first * it.second;
    total_bytes += (it.first * it.second);
  }

  LOGS_DEFAULT(INFO) << "Sum Total of in-use chunks: " << total_bytes;
  LOGS_DEFAULT(INFO) << "Stats: \n"
                     << stats_.DebugString();
}
}  // namespace onnxruntime