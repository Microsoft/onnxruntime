// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <d3d12.h>
#include <wil/Resource.h>
#include <wrl/client.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace windows {
namespace d3d12 {

class D3D12FileBufferLoader {
 public:
  struct Config {
    uint32_t upload_slot_count = 12;
    uint64_t upload_slot_size = 16ull * 1024ull * 1024ull;
    uint64_t max_destination_heap_size = 512ull * 1024ull * 1024ull;
    // Use CPU-visible final buffers when the device supports UMA write-back heaps.
    bool enable_uma = true;
    uint32_t uma_reader_count = 8;
    uint32_t uma_read_size = 16u * 1024u * 1024u;
  };

  struct BufferSource {
    std::wstring path;
    uint64_t offset = 0;
    uint64_t length = 0;
  };

  struct CancellationToken {
    // Queries may run on UMA reader threads, but are serialized within a load.
    using IsCancellationRequestedFn = bool (*)(void* state) noexcept;

    IsCancellationRequestedFn is_cancellation_requested = nullptr;
    void* state = nullptr;

    bool IsCancellationRequested() const noexcept {
      return is_cancellation_requested != nullptr &&
             is_cancellation_requested(state);
    }
  };

  struct BufferCollection {
    std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>> heaps;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers;

    void Clear() noexcept {
      buffers.clear();
      heaps.clear();
    }
  };

  static common::Status Create(
      ID3D12Device* device,
      std::unique_ptr<D3D12FileBufferLoader>& loader) noexcept;
  static common::Status Create(
      ID3D12Device* device,
      std::unique_ptr<D3D12FileBufferLoader>& loader,
      const Config& config) noexcept;

  ~D3D12FileBufferLoader();

  D3D12FileBufferLoader(const D3D12FileBufferLoader&) = delete;
  D3D12FileBufferLoader& operator=(const D3D12FileBufferLoader&) = delete;

  bool UsesUMA() const noexcept { return use_uma_; }

  // Each BufferSource describes a file range for one returned buffer.
  // The returned buffers correspond one-to-one with ranges and preserve input order.
  // Each buffer is in D3D12_RESOURCE_STATE_COMMON when this method succeeds.
  // Resource widths may include padding; ranges specify the loaded data lengths.
  common::Status Load(
      const std::vector<BufferSource>& ranges,
      BufferCollection& result) noexcept;
  common::Status Load(
      const std::vector<BufferSource>& ranges,
      BufferCollection& result,
      const CancellationToken& cancellation) noexcept;

 private:
  struct FileReadRegion;
  struct FileReadPlan;
  struct UploadStagingSlot;
  struct AllocationResult;

  D3D12FileBufferLoader(
      ID3D12Device* device,
      const Config& config) noexcept;

  common::Status Initialize();
  bool TryInitializeUma();
  common::Status LoadUma(
      const std::vector<BufferSource>& ranges,
      const std::vector<uint64_t>& sizes,
      BufferCollection& result,
      const CancellationToken& cancellation);
  common::Status LoadInternal(
      const std::vector<BufferSource>& ranges,
      BufferCollection& result,
      const CancellationToken& cancellation);
  common::Status AllocateDestinationGpuBuffers(
      const std::vector<uint64_t>& sizes,
      BufferCollection& batch,
      const std::function<common::Status(size_t, ID3D12Resource*)>& on_ready = {});
  common::Status CreateFileReadPlan(
      const std::vector<BufferSource>& ranges,
      const CancellationToken& cancellation,
      std::vector<FileReadPlan>& files);
  common::Status PrepareUploadStagingSlots(
      uint64_t alignment,
      const CancellationToken& cancellation);
  common::Status IssueFileRead(
      HANDLE file,
      UploadStagingSlot& staging_slot,
      uint64_t offset,
      DWORD size);
  common::Status ProcessFileReadCompletion(
      uint64_t file_size,
      UploadStagingSlot& staging_slot,
      DWORD& bytes_read);
  template <typename EnsureAllocationFn>
  common::Status ReadFileRegion(
      FileReadPlan& file,
      const FileReadRegion& region,
      const std::vector<BufferSource>& ranges,
      BufferCollection& batch,
      bool& allocation_ready,
      EnsureAllocationFn& ensure_allocation,
      const CancellationToken& cancellation,
      uint64_t& last_submitted_fence);
  common::Status SubmitStagingToDestinationCopies(
      UploadStagingSlot& staging_slot,
      uint64_t chunk_begin,
      uint64_t bytes,
      const std::vector<size_t>& buffer_source_indices,
      const std::vector<BufferSource>& ranges,
      BufferCollection& batch,
      uint64_t& last_submitted_fence);
  common::Status TransitionDestinationBuffersToCommonState(
      BufferCollection& batch,
      const CancellationToken& cancellation,
      uint64_t& last_submitted_fence);
  common::Status WaitForFence(
      uint64_t value,
      const CancellationToken& cancellation);
  common::Status SignalSubmittedWork(uint64_t value, BufferCollection& batch);
  void WaitForFenceUncancelled(uint64_t value) noexcept;
  void DrainActiveReads() noexcept;

  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  Config config_;
  bool use_uma_ = false;
  D3D12_HEAP_PROPERTIES destination_heap_properties_{};
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> copy_queue_;
  Microsoft::WRL::ComPtr<ID3D12Fence> copy_fence_;
  wil::unique_handle copy_fence_complete_event_;
  std::vector<UploadStagingSlot> upload_staging_slots_;
  uint64_t next_fence_value_ = 0;
  bool upload_error_gpu_completion_unknown_ = false;
  std::mutex load_mutex_;
};

}  // namespace d3d12
}  // namespace windows
}  // namespace onnxruntime
