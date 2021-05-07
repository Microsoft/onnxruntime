// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace test {
TEST(AllocatorTest, CUDAAllocatorTest) {
  OrtDevice::DeviceId cuda_device_id = 0;

  // ensure CUDA device is avaliable.
  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));

  AllocatorCreationInfo default_memory_info(
      {[](OrtDevice::DeviceId id) { return std::make_unique<CUDAAllocator>(id, GPU); }, cuda_device_id});

  auto cuda_arena = CreateAllocator(default_memory_info);

  size_t size = 1024;

  EXPECT_STREQ(cuda_arena->Info().name, GPU);
  EXPECT_EQ(cuda_arena->Info().id, cuda_device_id);
  EXPECT_EQ(cuda_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cuda_arena->Info().alloc_type, OrtArenaAllocator);

  //test cuda allocation
  auto cuda_addr = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr);

  AllocatorCreationInfo pinned_memory_info(
      [](int) { return std::make_unique<CUDAPinnedAllocator>(static_cast<OrtDevice::DeviceId>(0), GPU_PINNED); });

  auto pinned_allocator = CreateAllocator(pinned_memory_info);

  EXPECT_STREQ(pinned_allocator->Info().name, GPU_PINNED);
  EXPECT_EQ(pinned_allocator->Info().id, 0);
  EXPECT_EQ(pinned_allocator->Info().mem_type, OrtMemTypeCPUOutput);
  EXPECT_EQ(pinned_allocator->Info().alloc_type, OrtArenaAllocator);

  //test pinned allocation
  auto pinned_addr = pinned_allocator->Alloc(size);
  EXPECT_TRUE(pinned_addr);

  const auto& cpu_arena = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  EXPECT_STREQ(cpu_arena->Info().name, CPU);
  EXPECT_EQ(cpu_arena->Info().id, 0);
  EXPECT_EQ(cpu_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cpu_arena->Info().alloc_type, OrtArenaAllocator);

  auto cpu_addr_a = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_a);
  auto cpu_addr_b = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_b);
  memset(cpu_addr_a, -1, 1024);

  //test host-device memory copy
  cudaMemcpy(cuda_addr, cpu_addr_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cpu_addr_b, cuda_addr, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(*((int*)cpu_addr_b), -1);

  cudaMemcpyAsync(pinned_addr, cuda_addr, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  EXPECT_EQ(*((int*)pinned_addr), -1);

  cpu_arena->Free(cpu_addr_a);
  cpu_arena->Free(cpu_addr_b);
  cuda_arena->Free(cuda_addr);
  pinned_allocator->Free(pinned_addr);
}

// test that we fallback to smaller allocations if the growth of the arena exceeds the available memory
TEST(AllocatorTest, CUDAAllocatorFallbackTest) {
  OrtDevice::DeviceId cuda_device_id = 0;

  size_t free = 0;
  size_t total = 0;

  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));

  // need extra test logic if this ever happens.
  EXPECT_NE(free, total) << "All memory is free. Test logic does not handle this.";

  AllocatorCreationInfo default_memory_info(
      {[](OrtDevice::DeviceId id) { return std::make_unique<CUDAAllocator>(id, GPU); },
       cuda_device_id});

  auto cuda_arena = CreateAllocator(default_memory_info);

  // initial allocation that sets the growth size for the next allocation
  size_t size = total / 2;
  void* cuda_addr_0 = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr_0);

  // this should trigger an allocation equal to the current total, which should fail initially and gradually fall back
  // to a smaller block.
  size_t next_size = 1024;

  void* cuda_addr_1 = cuda_arena->Alloc(next_size);
  EXPECT_TRUE(cuda_addr_1);
  cuda_arena->Free(cuda_addr_0);
  cuda_arena->Free(cuda_addr_1);
  cuda_arena = nullptr;

  auto last_error = cudaGetLastError();
  EXPECT_EQ(last_error, cudaSuccess) << "Last error should be cleared if handled gracefully";
}
}  // namespace test
}  // namespace onnxruntime
