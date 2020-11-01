// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/allocation_planner.h"

namespace onnxruntime {
struct MemoryBlock {
  size_t offset_{0};
  size_t size_{0};

  MemoryBlock() = default;
  MemoryBlock(size_t offset, size_t size) : offset_(offset), size_(size) {}
  bool operator<(const MemoryBlock& mb) const {
    return offset_ < mb.offset_;
  }
};

class MemoryPattern {
  friend class MemPatternPlanner;

 public:
  MemoryPattern() = default;

  MemoryPattern(MemoryPattern&& rhs) noexcept
      : patterns_{std::move(rhs.patterns_)},
        peak_size_{std::move(rhs.peak_size_)} {}

  MemoryPattern& operator=(MemoryPattern&& rhs) noexcept {
    patterns_ = std::move(rhs.patterns_);
    peak_size_ = std::move(rhs.peak_size_);
    return *this;
  }

  size_t PeakSize() const {
    return peak_size_;
  }

  const MemoryBlock* GetBlock(int ml_value_idx) const {
    auto it = patterns_.find(ml_value_idx);
    if (it == patterns_.end())
      return nullptr;

    return &it->second;
  }

  const std::unordered_map<int, MemoryBlock>& GetPatternsMap() const {
    return patterns_;
  }

 private:
  // allow move
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MemoryPattern);

  std::unordered_map<int, MemoryBlock> patterns_;
  size_t peak_size_{0};
};

struct MemoryPatternGroup {
  std::vector<OrtMemoryInfo> locations;
  std::vector<MemoryPattern> patterns;

  const MemoryPattern* GetPatterns(const OrtMemoryInfo& location) const {
    for (size_t i = 0; i < locations.size(); i++)
      if (locations[i] == location) {
        return &patterns[i];
      }
    return nullptr;
  }
};
}  // namespace onnxruntime
