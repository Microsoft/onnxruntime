//Part of the algo is derived from tensorflow.

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include "core/common/safeint.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/allocation_planner.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
// MemPatternPlanner is used to trace allocation/free steps
// in a single iteration, record the pattern and cached for
// future request if they have the same input shape.
// Thread-safe.
class MemPatternPlanner {
 public:
  // only the Training code currently uses the program counter based logic
  MemPatternPlanner(bool using_counters) : using_counters_{using_counters} {}

#ifdef ENABLE_TRAINING
  // TODO: OverlappingTimeSchedules should be private
  // Returns true if there is an intersection between two time schedules.
  // ProgramCounter values are validated when the execution plan is created
  bool OverlappingTimeSchedules(const AllocPlanPerValue::ProgramCounter& counter1,
                                const AllocPlanPerValue::ProgramCounter& counter2) const {
    const auto& starts_1 = counter1.Starts();
    const auto& ends_1 = counter1.Ends();
    const auto& starts_2 = counter2.Starts();
    const auto& ends_2 = counter2.Ends();

    size_t index_1 = 0;
    size_t index_2 = 0;
    size_t index_1_end = starts_1.size();
    size_t index_2_end = starts_2.size();

    while ((index_1 < index_1_end) && (index_2 < index_2_end)) {
      if (starts_1[index_1] <= starts_2[index_2]) {
        if (ends_1[index_1] >= starts_2[index_2]) {
          return true;
        }
        index_1 += 1;
      } else {
        if (ends_2[index_2] >= starts_1[index_1]) {
          return true;
        }
        index_2 += 1;
      }
    }

    return false;
  }

  void TraceAllocation(int ml_value_idx, const AllocPlanPerValue::ProgramCounter& counter, size_t size, bool insert_end = false) {
    ORT_ENFORCE(using_counters_);

    std::lock_guard<OrtMutex> lock(lock_);

    if (size == 0) {
      allocs_.emplace_back(ml_value_idx, MemoryBlock(0, 0));
      return;
    }

    // Insert the memory block in the end when laying out tensors that need to be contiguous in memory.
    if (insert_end) {
      allocs_.emplace_back(ml_value_idx, counter, MemoryBlock(buffer_size_, size));
      blocks_.emplace_back(static_cast<int>(allocs_.size()) - 1);
      buffer_size_ += size;
      return;
    }

    size_t current = 0;
    size_t waste_bytes = std::numeric_limits<size_t>::max();
    size_t best_offset = 0;
    bool best_offset_found = false;
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      // Memory block can be re-used as long as there is no overlap between their time schedules.
      if (allocs_[*it].reuse_ && !OverlappingTimeSchedules(counter, *allocs_[*it].counter_)) {
        continue;
      }

      if (allocs_[*it].block_.offset_ >= current) {
        auto gap = allocs_[*it].block_.offset_ - current;
        if (gap >= size && (gap - size) < waste_bytes) {
          waste_bytes = gap - size;
          best_offset = current;
          best_offset_found = true;
        }
      }

      current = std::max(current, allocs_[*it].block_.offset_ + allocs_[*it].block_.size_);
    }

    ORT_ENFORCE(current <= buffer_size_);

    if (current < buffer_size_) {
      auto gap = buffer_size_ - current;
      if ((gap >= size) && ((gap - size) < waste_bytes)) {
        best_offset = current;
        best_offset_found = true;
      }
    }

    if (!best_offset_found) {
      best_offset = current;
    }

    // we only need to bounds check the addition of size to best_offset as that is the only time we extend
    // the maximum size of the buffer.
    buffer_size_ = std::max(buffer_size_, SafeInt<size_t>(best_offset) + size);
    allocs_.emplace_back(ml_value_idx, counter, MemoryBlock(best_offset, size));
    std::list<int>::iterator best_fit_it = blocks_.end();
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ < best_offset)
        continue;

      if ((allocs_[*it].block_.offset_ > best_offset) || (allocs_[*it].block_.size_ >= size)) {
        best_fit_it = it;
        break;
      }
    }

    blocks_.insert(best_fit_it, (static_cast<int>(allocs_.size()) - 1));
  }
#endif

  void TraceAllocation(int ml_value_idx, size_t size) {
    ORT_ENFORCE(!using_counters_);

    std::lock_guard<OrtMutex> lock(lock_);

    if (size == 0) {
      allocs_.emplace_back(ml_value_idx, MemoryBlock(0, 0));
      return;
    }

    size_t current = 0;
    size_t waste_bytes = std::numeric_limits<size_t>::max();
    size_t best_offset = 0;
    bool best_offset_found = false;

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ >= current) {
        auto gap = allocs_[*it].block_.offset_ - current;
        if (gap >= size && (gap - size) < waste_bytes) {
          waste_bytes = gap - size;
          best_offset = current;
          best_offset_found = true;
        }
      }
      current = std::max(current, allocs_[*it].block_.offset_ + allocs_[*it].block_.size_);
    }

    ORT_ENFORCE(current <= buffer_size_);

    if (current < buffer_size_) {
      auto gap = buffer_size_ - current;
      if ((gap >= size) && ((gap - size) < waste_bytes)) {
        best_offset = current;
        best_offset_found = true;
      }
    }

    if (!best_offset_found) {
      best_offset = current;
    }

    // we only need to bounds check the addition of size to best_offset as that is the only time we extend
    // the maximum size of the buffer.
    buffer_size_ = std::max(buffer_size_, SafeInt<size_t>(best_offset) + size);
    allocs_.emplace_back(ml_value_idx, MemoryBlock(best_offset, size));
    std::list<int>::iterator best_fit_it = blocks_.end();
    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].block_.offset_ < best_offset)
        continue;

      if ((allocs_[*it].block_.offset_ > best_offset) || (allocs_[*it].block_.size_ >= size)) {
        best_fit_it = it;
        break;
      }
    }

    blocks_.insert(best_fit_it, (static_cast<int>(allocs_.size()) - 1));
  }

  void TraceFree(int ml_value_index) {
    std::lock_guard<OrtMutex> lock(lock_);

    for (auto it = blocks_.begin(); it != blocks_.end(); it++) {
      if (allocs_[*it].index_ == ml_value_index) {
        blocks_.erase(it);
        break;
      }
    }
  }

  MemoryPattern GenerateMemPattern() const {
    std::lock_guard<OrtMutex> lock(lock_);

#ifdef ENABLE_TRAINING
    if (using_counters_) {
      // Time schedules of overlapping memory blocks SHOULD NOT intersect.
      for (size_t index_1 = 0; index_1 < allocs_.size(); index_1 += 1) {
        if (!allocs_[index_1].reuse_)
          continue;

        for (size_t index_2 = index_1 + 1; index_2 < allocs_.size(); index_2 += 1) {
          if (!allocs_[index_2].reuse_)
            continue;

          size_t alloc_1_start = allocs_[index_1].block_.offset_;
          size_t alloc_1_end = alloc_1_start + allocs_[index_1].block_.size_ - 1;

          ORT_ENFORCE(alloc_1_start <= alloc_1_end);

          size_t alloc_2_start = allocs_[index_2].block_.offset_;
          size_t alloc_2_end = alloc_2_start + allocs_[index_2].block_.size_ - 1;

          ORT_ENFORCE(alloc_2_start <= alloc_2_end);

          if (((alloc_1_start >= alloc_2_start) && (alloc_1_start <= alloc_2_end)) ||
              ((alloc_2_start >= alloc_1_start) && (alloc_2_start <= alloc_1_end))) {
            ORT_ENFORCE(!OverlappingTimeSchedules(*allocs_[index_1].counter_, *allocs_[index_2].counter_));
          }
        }
      }
    }
#endif

    MemoryPattern pattern;
    pattern.peak_size_ = buffer_size_;
    for (auto& alloc : allocs_) {
      pattern.patterns_[alloc.index_] = alloc.block_;
    }

    return pattern;
  }

 private:
  struct OrtValueAllocationBlock {
    int index_{-1};
    MemoryBlock block_;
    const AllocPlanPerValue::ProgramCounter* counter_{nullptr};
    bool reuse_{false};
    OrtValueAllocationBlock() = default;
    OrtValueAllocationBlock(int index, const MemoryBlock& block) : index_(index), block_(block), reuse_{false} {}
    OrtValueAllocationBlock(int index, const AllocPlanPerValue::ProgramCounter& counter, const MemoryBlock& block)
        : index_(index), block_(block), counter_(&counter), reuse_{true} {
    }
  };

  std::vector<OrtValueAllocationBlock> allocs_;
  // blocks_ the list of currently allocated memory blocks, sorted in order of their offset
  std::list<int> blocks_;
  SafeInt<size_t> buffer_size_{0};
  bool using_counters_;
  mutable OrtMutex lock_;
};

}  // namespace onnxruntime
