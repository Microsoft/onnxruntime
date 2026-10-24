// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>

#include "core/providers/openvino/callback_buffer_stream.h"
#include "gtest/gtest.h"

namespace onnxruntime::openvino_ep {
namespace {

struct CountingAllocator : OrtAllocator {
  CountingAllocator() : OrtAllocator{} {
    version = ORT_API_VERSION;
    Free = [](OrtAllocator* allocator, void* buffer) {
      auto& self = *static_cast<CountingAllocator*>(allocator);
      ++self.free_count;
      EXPECT_EQ(buffer, self.data.data());
    };
  }

  std::array<char, 4> data{'a', 'b', 'c', 'd'};
  int free_count = 0;
};

TEST(OpenVINOCallbackBufferStreamTest, ReadsSeeksAndReleasesAllocationOnce) {
  CountingAllocator allocator;
  {
    CallbackBufferIStream stream(
        {allocator.data.data(), OrtAllocatorDeleter{&allocator}}, allocator.data.size());
    EXPECT_EQ(stream.get(), 'a');
    stream.seekg(-1, std::ios::end);
    EXPECT_EQ(stream.get(), 'd');
    EXPECT_EQ(stream.tellg(), std::streampos{4});
    stream.seekg(0, std::ios::beg);
    std::array<char, 4> result{};
    stream.read(result.data(), result.size());
    EXPECT_EQ(result, allocator.data);
    EXPECT_EQ(allocator.free_count, 0);
  }
  EXPECT_EQ(allocator.free_count, 1);
}

TEST(OpenVINOCallbackBufferStreamTest, RejectsOutOfRangeAndExtremeOffsets) {
  CountingAllocator allocator;
  CallbackBufferIStream stream(
      {allocator.data.data(), OrtAllocatorDeleter{&allocator}}, allocator.data.size());
  for (auto direction : {std::ios::beg, std::ios::cur, std::ios::end}) {
    for (auto offset : {std::numeric_limits<std::streamoff>::min(),
                        std::numeric_limits<std::streamoff>::max()}) {
      stream.seekg(offset, direction);
      EXPECT_TRUE(stream.fail());
      stream.clear();
      EXPECT_EQ(stream.tellg(), std::streampos{0});
    }
  }
  stream.seekg(-1, std::ios::beg);
  EXPECT_TRUE(stream.fail());
  stream.clear();
  stream.seekg(1, std::ios::end);
  EXPECT_TRUE(stream.fail());
  stream.clear();
  stream.seekg(std::streampos{5});
  EXPECT_TRUE(stream.fail());
  stream.clear();
  EXPECT_EQ(stream.tellg(), std::streampos{0});
}

TEST(OpenVINOCallbackBufferStreamTest, ReleasesAllocationOnUnwind) {
  CountingAllocator allocator;
  try {
    CallbackBufferIStream stream(
        {allocator.data.data(), OrtAllocatorDeleter{&allocator}}, allocator.data.size());
    stream.exceptions(std::ios::failbit);
    stream.seekg(-1, std::ios::beg);
    FAIL() << "Expected invalid seek to throw";
  } catch (const std::ios_base::failure&) {
  }
  EXPECT_EQ(allocator.free_count, 1);
}

}  // namespace
}  // namespace onnxruntime::openvino_ep
