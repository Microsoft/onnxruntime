#pragma once

#include <functional>
#include <memory>
#include <new>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace onnxruntime::cuda {

class BoundedPinnedPool {
 public:
  using Allocate = std::function<void*(size_t)>;
  using Free = std::function<void(void*)>;

  BoundedPinnedPool(size_t limit, Allocate allocate, Free free)
      : limit_{limit}, allocate_{std::move(allocate)}, free_{std::move(free)} {
    if (!limit_ || !allocate_ || !free_) throw std::invalid_argument("Pinned pool configuration is invalid");
  }

  ~BoundedPinnedPool() {
    for (const auto& block : blocks_)
      if (block.available) free_(block.pointer);
  }

  void* Acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!bytes || bytes > limit_) throw std::invalid_argument("Pinned pool request exceeds its limit");
    Block* selected = nullptr;
    for (auto& block : blocks_) {
      if (block.available && block.bytes >= bytes && (!selected || block.bytes < selected->bytes)) selected = &block;
    }
    if (selected) {
      selected->available = false;
      return selected->pointer;
    }
    // Idle blocks that are too small must not hold capacity hostage; return them to the system before rejecting.
    if (bytes > limit_ - committed_) ReclaimAvailable(bytes - (limit_ - committed_));
    if (bytes > limit_ - committed_) throw std::runtime_error("Pinned pool capacity exhausted");
    auto* pointer = allocate_(bytes);
    if (!pointer) throw std::bad_alloc();
    try {
      blocks_.push_back({pointer, bytes, false});
    } catch (...) {
      free_(pointer);
      throw;
    }
    committed_ += bytes;
    return pointer;
  }

  void Release(void* pointer, bool safe) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
      if (block.pointer == pointer) {
        block.available = safe;
        return;
      }
    }
  }

  size_t Limit() const { return limit_; }
  size_t Committed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_;
  }

 private:
  struct Block {
    void* pointer;
    size_t bytes;
    bool available;
  };

  // Frees available blocks, largest first, until at least `needed` bytes are released. Caller holds mutex_.
  void ReclaimAvailable(size_t needed) {
    size_t reclaimed = 0;
    while (reclaimed < needed) {
      auto victim = blocks_.end();
      for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
        if (it->available && (victim == blocks_.end() || it->bytes > victim->bytes)) victim = it;
      }
      if (victim == blocks_.end()) return;
      free_(victim->pointer);
      reclaimed += victim->bytes;
      committed_ -= victim->bytes;
      blocks_.erase(victim);
    }
  }
  size_t limit_;
  size_t committed_ = 0;
  Allocate allocate_;
  Free free_;
  mutable std::mutex mutex_;
  std::vector<Block> blocks_;
};

}  // namespace onnxruntime::cuda
