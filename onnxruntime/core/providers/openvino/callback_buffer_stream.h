// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
#include <utility>

#include "onnxruntime_c_api.h"

namespace onnxruntime::openvino_ep {

struct OrtAllocatorDeleter {
  OrtAllocator* allocator{};
  void operator()(void* buffer) const noexcept {
    if (buffer != nullptr) {
      allocator->Free(allocator, buffer);
    }
  }
};

// Owns the callback allocation while allowing the deserializer to seek without a second copy.
class CallbackBufferIStream final : public std::istream {
 public:
  CallbackBufferIStream(std::unique_ptr<void, OrtAllocatorDeleter> buffer, size_t buffer_size)
      : std::istream(nullptr), buffer_{std::move(buffer)}, streambuf_{buffer_.get(), buffer_size} {
    rdbuf(&streambuf_);
  }

 private:
  class MemoryStreamBuf final : public std::streambuf {
   public:
    MemoryStreamBuf(void* buffer, size_t buffer_size)
        : begin_{static_cast<char*>(buffer)}, size_{buffer_size} {
      setg(begin_, begin_, begin_ + size_);
    }

   protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                     std::ios_base::openmode mode) override {
      if ((mode & std::ios_base::in) == 0) {
        return pos_type{off_type{-1}};
      }

      size_t base{};
      if (direction == std::ios_base::beg) {
        base = 0;
      } else if (direction == std::ios_base::cur) {
        base = static_cast<size_t>(gptr() - begin_);
      } else if (direction == std::ios_base::end) {
        base = size_;
      } else {
        return pos_type{off_type{-1}};
      }

      size_t position{};
      if (offset >= 0) {
        const auto delta = static_cast<uintmax_t>(offset);
        if (delta > size_ - base) {
          return pos_type{off_type{-1}};
        }
        position = base + static_cast<size_t>(delta);
      } else {
        const auto delta = static_cast<uintmax_t>(-(offset + 1)) + 1;
        if (delta > base) {
          return pos_type{off_type{-1}};
        }
        position = base - static_cast<size_t>(delta);
      }

      if (position > static_cast<uintmax_t>(std::numeric_limits<off_type>::max())) {
        return pos_type{off_type{-1}};
      }

      setg(begin_, begin_ + position, begin_ + size_);
      return pos_type{static_cast<off_type>(position)};
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
      return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
    }

   private:
    char* begin_;
    size_t size_;
  };

  std::unique_ptr<void, OrtAllocatorDeleter> buffer_;
  MemoryStreamBuf streambuf_;
};

}  // namespace onnxruntime::openvino_ep
