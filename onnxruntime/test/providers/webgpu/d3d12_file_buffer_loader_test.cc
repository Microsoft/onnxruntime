// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include <dxgi1_6.h>
#include <gsl/gsl>
#include "gtest/gtest.h"
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"

namespace onnxruntime::test {
namespace {

using Microsoft::WRL::ComPtr;
using Loader = windows::d3d12::D3D12FileBufferLoader;

class D3D12FileBufferLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      GTEST_SKIP() << "DXGI adapter enumeration unavailable";
    }
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT index = 0; SUCCEEDED(factory->EnumAdapterByGpuPreference(
             index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
         ++index) {
      DXGI_ADAPTER_DESC1 description{};
      ASSERT_TRUE(SUCCEEDED(adapter->GetDesc1(&description)));
      if (!(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
          SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)))) {
        RecordProperty("adapter", ToUTF8String(description.Description));
        RecordProperty("vendor_id", static_cast<int>(description.VendorId));
        RecordProperty("device_id", static_cast<int>(description.DeviceId));
        break;
      }
      adapter.Reset();
    }
    if (!device_) {
      GTEST_SKIP() << "No hardware D3D12 adapter";
    }
    ASSERT_TRUE(SUCCEEDED(device_->CheckFeatureSupport(
        D3D12_FEATURE_ARCHITECTURE, &architecture_, sizeof(architecture_))));
    RecordProperty("uma", architecture_.UMA);
    RecordProperty("cache_coherent_uma", architecture_.CacheCoherentUMA);
  }

  void TearDown() override {
    for (const auto& path : files_) {
      EXPECT_TRUE(DeleteFileW(path.c_str()));
    }
  }

  std::wstring WriteFileData(size_t size, uint8_t seed = 0) {
    wchar_t directory[MAX_PATH]{};
    wchar_t filename[MAX_PATH]{};
    if (!GetTempPathW(MAX_PATH, directory) || !GetTempFileNameW(directory, L"ort", 0, filename)) {
      ADD_FAILURE() << "Failed to create temporary weights file";
      return {};
    }
    files_.emplace_back(filename);
    std::ofstream stream{std::filesystem::path{filename}, std::ios::binary | std::ios::trunc};
    InlinedVector<uint8_t> data(size);
    for (size_t index = 0; index < size; ++index) {
      data[index] = static_cast<uint8_t>(index * 37 + seed);
    }
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(size));
    EXPECT_TRUE(stream.good());
    return filename;
  }

  bool IsWriteBack(const Loader::BufferCollection& batch) {
    D3D12_HEAP_PROPERTIES properties{};
    EXPECT_TRUE(SUCCEEDED(batch.buffers.front()->GetHeapProperties(&properties, nullptr)));
    return properties.Type == D3D12_HEAP_TYPE_CUSTOM &&
           properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK &&
           properties.MemoryPoolPreference == D3D12_MEMORY_POOL_L0;
  }

  void CheckReadback(const Loader::BufferCollection& batch, const std::vector<Loader::BufferSource>& ranges) {
    ASSERT_EQ(batch.buffers.size(), ranges.size());
    uint64_t total = 0;
    for (const auto& resource : batch.buffers) {
      total += resource->GetDesc().Width;
    }
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    ASSERT_TRUE(SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12CommandAllocator> allocator;
    ASSERT_TRUE(SUCCEEDED(device_->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator))));
    ComPtr<ID3D12GraphicsCommandList> list;
    ASSERT_TRUE(SUCCEEDED(device_->CreateCommandList(0, queue_desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list))));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = total;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    ASSERT_TRUE(SUCCEEDED(device_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))));
    uint64_t offset = 0;
    for (const auto& resource : batch.buffers) {
      list->CopyBufferRegion(readback.Get(), offset, resource.Get(), 0, resource->GetDesc().Width);
      offset += resource->GetDesc().Width;
    }
    ASSERT_TRUE(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[]{list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    ASSERT_TRUE(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    wil::unique_handle event(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    ASSERT_TRUE(event);
    ASSERT_TRUE(SUCCEEDED(queue->Signal(fence.Get(), 1)));
    ASSERT_TRUE(SUCCEEDED(fence->SetEventOnCompletion(1, event.get())));
    ASSERT_EQ(WaitForSingleObject(event.get(), INFINITE), WAIT_OBJECT_0);
    const D3D12_RANGE read_range{0, static_cast<SIZE_T>(total)};
    void* mapped = nullptr;
    ASSERT_TRUE(SUCCEEDED(readback->Map(0, &read_range, &mapped)));
    auto unmap = gsl::finally([&] {
      const D3D12_RANGE written{0, 0};
      readback->Unmap(0, &written);
    });
    offset = 0;
    for (size_t index = 0; index < ranges.size(); ++index) {
      const auto& range = ranges[index];
      InlinedVector<uint8_t> expected(static_cast<size_t>(range.length));
      std::ifstream stream{std::filesystem::path{range.path}, std::ios::binary};
      stream.seekg(static_cast<std::streamoff>(range.offset));
      stream.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
      ASSERT_TRUE(stream.good());
      const auto* bytes = static_cast<const uint8_t*>(mapped) + offset;
      EXPECT_EQ(std::memcmp(bytes, expected.data(), expected.size()), 0) << "range " << index;
      const auto width = batch.buffers[index]->GetDesc().Width;
      for (uint64_t tail = range.length; tail < width; ++tail) {
        EXPECT_EQ(bytes[tail], expected[(tail - range.length) % range.length]);
      }
      offset += width;
    }
  }

  ComPtr<ID3D12Device> device_;
  D3D12_FEATURE_DATA_ARCHITECTURE architecture_{};
  InlinedVector<std::wstring> files_;
};

TEST_F(D3D12FileBufferLoaderTest, UmaReadsUnalignedOverlappingAndMultipleFileRanges) {
  const auto first = WriteFileData(256 * 1024 + 13);
  const auto second = WriteFileData(32 * 1024 + 7, 0x58);
  const std::vector<Loader::BufferSource> ranges{
      {second, 3, 17}, {first, 11, 131073}, {first, 5, 1}, {first, 7, 31}, {second, 7, 32768}};
  Loader::Config config;
  config.max_destination_heap_size = 64 * 1024;
  config.uma_read_size = 4093;
  std::unique_ptr<Loader> loader;
  auto status = Loader::Create(device_.Get(), loader, config);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  Loader::BufferCollection batch;
  status = loader->Load(ranges, batch);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!IsWriteBack(batch)) {
    GTEST_SKIP() << "Selected adapter does not support the UMA write-back path";
  }
  EXPECT_TRUE(architecture_.UMA);
  EXPECT_GT(batch.heaps.size(), 1u);
  CheckReadback(batch, ranges);
  // Reuse must not retain mappings or pending jobs from the preceding batch.
  status = loader->Load(ranges, batch);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  CheckReadback(batch, ranges);
}

TEST_F(D3D12FileBufferLoaderTest, DisabledUmaPreservesUploadPipeline) {
  const auto path = WriteFileData(192 * 1024 + 13);
  const std::vector<Loader::BufferSource> ranges{{path, 3, 17}, {path, 13, 192 * 1024}};
  Loader::Config config;
  config.enable_uma = false;
  config.upload_slot_count = 2;
  config.upload_slot_size = 64 * 1024;
  std::unique_ptr<Loader> loader;
  auto status = Loader::Create(device_.Get(), loader, config);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  Loader::BufferCollection batch;
  status = loader->Load(ranges, batch);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  D3D12_HEAP_PROPERTIES properties{};
  ASSERT_TRUE(SUCCEEDED(batch.buffers.front()->GetHeapProperties(&properties, nullptr)));
  EXPECT_EQ(properties.Type, D3D12_HEAP_TYPE_DEFAULT);
  CheckReadback(batch, ranges);
}

TEST_F(D3D12FileBufferLoaderTest, UmaCancellationDrainsReadsAndAllowsReuse) {
  const auto path = WriteFileData(4 * 1024 * 1024 + 1);
  const std::vector<Loader::BufferSource> ranges{{path, 0, 4 * 1024 * 1024 + 1}};
  Loader::Config config;
  config.uma_read_size = 4096;
  std::unique_ptr<Loader> loader;
  auto status = Loader::Create(device_.Get(), loader, config);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  Loader::BufferCollection batch;
  status = loader->Load(ranges, batch);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!IsWriteBack(batch)) {
    GTEST_SKIP() << "Selected adapter does not support the UMA write-back path";
  }
  struct CancellationState {
    std::atomic<unsigned> calls{0};
    std::atomic<unsigned> active_callbacks{0};
    std::atomic<bool> concurrent_callbacks{false};
  } state;
  const Loader::CancellationToken cancellation{
      [](void* opaque) noexcept {
        auto& state = *static_cast<CancellationState*>(opaque);
        if (state.active_callbacks.fetch_add(1) != 0) {
          state.concurrent_callbacks.store(true);
        }
        Sleep(1);
        const bool canceled = state.calls.fetch_add(1) >= 32;
        state.active_callbacks.fetch_sub(1);
        return canceled;
      },
      &state};
  for (int repeat = 0; repeat < 3; ++repeat) {
    state.calls.store(0);
    status = loader->Load(ranges, batch, cancellation);
    EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
    EXPECT_FALSE(state.concurrent_callbacks.load());
    EXPECT_EQ(state.active_callbacks.load(), 0u);
    EXPECT_TRUE(batch.buffers.empty());
    EXPECT_TRUE(batch.heaps.empty());
    status = loader->Load(ranges, batch);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    CheckReadback(batch, ranges);
  }
}

TEST_F(D3D12FileBufferLoaderTest, InvalidRangesAndPreCancellationLeaveNoBuffers) {
  const auto path = WriteFileData(31);
  std::unique_ptr<Loader> loader;
  auto status = Loader::Create(device_.Get(), loader);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  Loader::BufferCollection batch;
  EXPECT_FALSE(loader->Load({{path, 1, 31}}, batch).IsOK());
  EXPECT_TRUE(batch.buffers.empty());
  EXPECT_FALSE(loader->Load({{path, std::numeric_limits<uint64_t>::max(), 2}}, batch).IsOK());
  EXPECT_TRUE(batch.buffers.empty());
  EXPECT_FALSE(loader->Load({{path + L".missing", 0, 1}}, batch).IsOK());
  EXPECT_TRUE(batch.buffers.empty());
  const Loader::CancellationToken cancellation{[](void*) noexcept { return true; }, nullptr};
  EXPECT_EQ(loader->Load({{path, 0, 31}}, batch, cancellation).Code(), common::MODEL_LOAD_CANCELED);
  EXPECT_TRUE(batch.buffers.empty());
  status = loader->Load({{path, 0, 31}}, batch);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  CheckReadback(batch, {{path, 0, 31}});
}

TEST_F(D3D12FileBufferLoaderTest, InvalidUmaConfigurationIsRejected) {
  std::unique_ptr<Loader> loader;
  Loader::Config config;
  config.uma_reader_count = 0;
  EXPECT_FALSE(Loader::Create(device_.Get(), loader, config).IsOK());
  EXPECT_EQ(loader, nullptr);
  config.uma_reader_count = 17;
  EXPECT_FALSE(Loader::Create(device_.Get(), loader, config).IsOK());
  config.uma_reader_count = 8;
  config.uma_read_size = 0;
  EXPECT_FALSE(Loader::Create(device_.Get(), loader, config).IsOK());
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
