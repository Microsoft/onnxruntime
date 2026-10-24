// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <system_error>
#include <utility>

#include <gsl/gsl>
#include <wil/Resource.h>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace windows {
namespace d3d12 {

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint64_t kBufferAlignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
// The cancellation token is callback-based and has no waitable handle. Start
// with a short wait to avoid adding fence-completion tail latency, then back
// off during longer transfers to avoid unnecessary CPU wakeups.
constexpr DWORD kInitialCancellationPollMilliseconds = 1;
constexpr DWORD kMaximumCancellationPollMilliseconds = 10;

common::Status HResultError(const char* operation, HRESULT hr) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, operation, " failed with HRESULT 0x",
      std::setfill('0'), std::uppercase, std::hex, std::setw(8),
      static_cast<uint32_t>(hr));
}

common::Status Win32Error(const char* operation, DWORD error) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, operation, " failed, error code ", error, " - ",
      std::system_category().message(error));
}

common::Status Win32FileError(const char* operation,
                              const std::wstring& path,
                              DWORD error) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, operation, " failed for \"", ToUTF8String(path),
      "\", error code ", error, " - ",
      std::system_category().message(error));
}

common::Status CancelledStatus() {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, MODEL_LOAD_CANCELED,
      "D3D12 file buffer loading was canceled.");
}

uint64_t AlignDown(uint64_t value, uint64_t alignment) noexcept {
  return value - value % alignment;
}

bool TryAlignUp(uint64_t value, uint64_t alignment, uint64_t& result) noexcept {
  if (alignment == 0 ||
      value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    return false;
  }
  result = AlignDown(value + alignment - 1, alignment);
  return true;
}

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept {
  D3D12_HEAP_PROPERTIES properties{};
  properties.Type = type;
  properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  properties.CreationNodeMask = 1;
  properties.VisibleNodeMask = 1;
  return properties;
}

D3D12_RESOURCE_DESC BufferDescription(
    uint64_t size, D3D12_RESOURCE_FLAGS flags) noexcept {
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  description.Width = size;
  description.Height = 1;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.Format = DXGI_FORMAT_UNKNOWN;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  description.Flags = flags;
  return description;
}

}  // namespace

struct D3D12FileBufferLoader::FileReadRegion {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct D3D12FileBufferLoader::FileReadPlan {
  std::wstring path;
  wil::unique_hfile file;
  wil::unique_hfile buffered_tail_file;
  uint64_t size = 0;
  uint64_t alignment = 0;
  std::vector<size_t> buffer_source_indices;
  std::vector<FileReadRegion> read_regions;
};

struct D3D12FileBufferLoader::UploadStagingSlot {
  ComPtr<ID3D12Resource> resource;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> command_list;
  wil::unique_handle file_read_complete_event;
  void* mapped = nullptr;
  OVERLAPPED overlapped{};
  HANDLE read_file = INVALID_HANDLE_VALUE;
  uint64_t file_offset = 0;
  DWORD requested_read_bytes = 0;
  uint64_t fence_value = 0;
  bool read_active = false;
};

struct D3D12FileBufferLoader::AllocationResult {
  common::Status status;
  BufferCollection batch;
};

D3D12FileBufferLoader::D3D12FileBufferLoader(
    ID3D12Device* device, const Config& config) noexcept
    : device_(device), config_(config), destination_heap_properties_(HeapProperties(D3D12_HEAP_TYPE_DEFAULT)) {
}

D3D12FileBufferLoader::~D3D12FileBufferLoader() {
  if (upload_error_gpu_completion_unknown_) {
    // Submitted work has no usable completion fence, so its resources must
    // remain alive for the lifetime of the process.
    device_.Detach();
    copy_queue_.Detach();
    copy_fence_.Detach();
    (void)copy_fence_complete_event_.release();
    for (auto& staging_slot : upload_staging_slots_) {
      staging_slot.resource.Detach();
      staging_slot.allocator.Detach();
      staging_slot.command_list.Detach();
      (void)staging_slot.file_read_complete_event.release();
    }
    return;
  }

  for (auto& staging_slot : upload_staging_slots_) {
    if (staging_slot.resource && staging_slot.mapped != nullptr) {
      staging_slot.resource->Unmap(0, nullptr);
      staging_slot.mapped = nullptr;
    }
  }
}

bool D3D12FileBufferLoader::TryInitializeUma() {
  D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
  if (!config_.enable_uma ||
      FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE,
                                          &architecture, sizeof(architecture))) ||
      !architecture.UMA) {
    return false;
  }

  auto properties = device_->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_READBACK);
  if (properties.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
      properties.MemoryPoolPreference != D3D12_MEMORY_POOL_L0) {
    return false;
  }

  // Some UMA drivers may not support CPU-visible UAV buffers. Keep the upload
  // pipeline available when creation or mapping of this resource type fails.
  const auto description = BufferDescription(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ComPtr<ID3D12Resource> probe;
  if (FAILED(device_->CreateCommittedResource(
          &properties, D3D12_HEAP_FLAG_NONE, &description,
          D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&probe)))) {
    return false;
  }
  void* mapped = nullptr;
  const D3D12_RANGE no_access{0, 0};
  if (FAILED(probe->Map(0, &no_access, &mapped))) {
    return false;
  }
  probe->Unmap(0, &no_access);
  destination_heap_properties_ = properties;
  return true;
}

common::Status D3D12FileBufferLoader::Initialize() {
  if (device_ == nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "D3D12FileBufferLoader requires a non-null D3D12 device.");
  }
  if (config_.upload_slot_count < 2 ||
      config_.upload_slot_count > MAXIMUM_WAIT_OBJECTS) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "upload_slot_count must be between 2 and ", MAXIMUM_WAIT_OBJECTS, ".");
  }
  if (config_.upload_slot_size == 0 ||
      config_.upload_slot_size > std::numeric_limits<DWORD>::max() ||
      config_.upload_slot_size % kBufferAlignment != 0) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "upload_slot_size must be a non-zero 64 KiB multiple no larger than DWORD_MAX.");
  }
  if (config_.max_destination_heap_size == 0 ||
      config_.max_destination_heap_size > 512ull * 1024ull * 1024ull ||
      config_.max_destination_heap_size % kBufferAlignment != 0) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "max_destination_heap_size must be a non-zero 64 KiB multiple no larger than 512 MiB.");
  }

  ORT_RETURN_IF(config_.uma_reader_count == 0 || config_.uma_reader_count > 16,
                "uma_reader_count must be between 1 and 16.");
  ORT_RETURN_IF(config_.uma_read_size == 0,
                "uma_read_size must be non-zero.");
  use_uma_ = TryInitializeUma();
  if (use_uma_) {
    return common::Status::OK();
  }

  D3D12_COMMAND_QUEUE_DESC queue_description{};
  queue_description.Type = D3D12_COMMAND_LIST_TYPE_COPY;
  HRESULT hr = device_->CreateCommandQueue(
      &queue_description, IID_PPV_ARGS(&copy_queue_));
  if (FAILED(hr)) {
    return HResultError("ID3D12Device::CreateCommandQueue", hr);
  }

  hr = device_->CreateFence(
      0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence_));
  if (FAILED(hr)) {
    return HResultError("ID3D12Device::CreateFence", hr);
  }

  copy_fence_complete_event_.reset(CreateEventExW(
      nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE));
  if (!copy_fence_complete_event_) {
    return Win32Error("CreateEventExW(copy fence)", GetLastError());
  }

  const auto upload_heap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
  const auto upload_description =
      BufferDescription(config_.upload_slot_size, D3D12_RESOURCE_FLAG_NONE);
  upload_staging_slots_.resize(config_.upload_slot_count);
  for (auto& staging_slot : upload_staging_slots_) {
    hr = device_->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &upload_description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&staging_slot.resource));
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12Device::CreateCommittedResource(upload)", hr);
    }

    hr = staging_slot.resource->Map(0, nullptr, &staging_slot.mapped);
    if (FAILED(hr)) {
      return HResultError("ID3D12Resource::Map(upload)", hr);
    }

    hr = device_->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&staging_slot.allocator));
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12Device::CreateCommandAllocator", hr);
    }

    hr = device_->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_COPY, staging_slot.allocator.Get(), nullptr,
        IID_PPV_ARGS(&staging_slot.command_list));
    if (FAILED(hr)) {
      return HResultError("ID3D12Device::CreateCommandList", hr);
    }
    hr = staging_slot.command_list->Close();
    if (FAILED(hr)) {
      return HResultError(
          "ID3D12GraphicsCommandList::Close(initial)", hr);
    }

    staging_slot.file_read_complete_event.reset(CreateEventExW(
        nullptr, nullptr, CREATE_EVENT_MANUAL_RESET,
        EVENT_MODIFY_STATE | SYNCHRONIZE));
    if (!staging_slot.file_read_complete_event) {
      return Win32Error("CreateEventExW(read)", GetLastError());
    }
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::LoadInternal(
    const std::vector<BufferSource>& ranges,
    BufferCollection& result,
    const CancellationToken& cancellation) {
  std::lock_guard<std::mutex> lock(load_mutex_);
  result.Clear();

  ORT_RETURN_IF(
      upload_error_gpu_completion_unknown_,
      "D3D12 file buffer loader cannot be reused when GPU completion is unknown after an upload error.");
  if (ranges.empty()) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "At least one file range is required.");
  }
  if (cancellation.IsCancellationRequested()) {
    return CancelledStatus();
  }

  std::vector<uint64_t> sizes;
  sizes.reserve(ranges.size());
  for (size_t index = 0; index < ranges.size(); ++index) {
    const auto& range = ranges[index];
    if (range.path.empty()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " has an empty path.");
    }
    if (range.length == 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " has zero length.");
    }
    if (range.offset >
        std::numeric_limits<uint64_t>::max() - range.length) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "File range ", index, " overflows uint64_t.");
    }
    sizes.push_back(range.length);
  }

  if (use_uma_) {
    return LoadUma(ranges, sizes, result, cancellation);
  }

  // Destination allocation can run while the CPU opens files and discovers
  // the sector alignment required for unbuffered reads.
  std::future<AllocationResult> allocation_future = std::async(
      std::launch::async, [this, sizes = std::move(sizes)]() noexcept {
        AllocationResult allocation;
        ORT_TRY {
          allocation.status =
              AllocateDestinationGpuBuffers(sizes, allocation.batch);
        }
        ORT_CATCH(const std::exception& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            allocation.status = ORT_MAKE_STATUS(
                ONNXRUNTIME, RUNTIME_EXCEPTION,
                "Destination allocation failed: ", ex.what());
          });
        }
        ORT_CATCH(...) {
          allocation.status = ORT_MAKE_STATUS(
              ONNXRUNTIME, RUNTIME_EXCEPTION,
              "Destination allocation failed with an unknown exception.");
        }
        return allocation;
      });

  // CreateFileReadPlan groups buffer sources by file and coalesces overlapping
  // or adjacent sector-aligned ranges into read regions to avoid redundant reads.
  std::vector<FileReadPlan> files;
  BufferCollection loaded_batch;
  uint64_t last_submitted_fence = 0;
  bool load_succeeded = false;
  // Resources referenced by in-flight I/O or GPU work must remain alive on
  // every error and cancellation path.
  auto cleanup = gsl::finally([&]() noexcept {
    if (!load_succeeded) {
      DrainActiveReads();
      WaitForFenceUncancelled(last_submitted_fence);
    }
    if (allocation_future.valid()) {
      allocation_future.wait();
    }
  });

  ORT_RETURN_IF_ERROR(CreateFileReadPlan(ranges, cancellation, files));

  uint64_t maximum_alignment = 1;
  for (const auto& file : files) {
    maximum_alignment = std::max(maximum_alignment, file.alignment);
  }
  ORT_RETURN_IF_ERROR(PrepareUploadStagingSlots(maximum_alignment, cancellation));

  bool allocation_ready = false;
  auto ensure_allocation = [&]() -> common::Status {
    if (allocation_ready) {
      return common::Status::OK();
    }
    AllocationResult allocation = allocation_future.get();
    if (!allocation.status.IsOK()) {
      return allocation.status;
    }
    loaded_batch = std::move(allocation.batch);
    allocation_ready = true;
    return common::Status::OK();
  };

  for (auto& file : files) {
    for (const auto& region : file.read_regions) {
      const auto status = ReadFileRegion(
          file, region, ranges, loaded_batch, allocation_ready,
          ensure_allocation, cancellation, last_submitted_fence);
      if (!status.IsOK()) {
        return status;
      }
    }
  }

  ORT_RETURN_IF_ERROR(ensure_allocation());
  ORT_RETURN_IF_ERROR(
      WaitForFence(last_submitted_fence, cancellation));
  ORT_RETURN_IF_ERROR(
      TransitionDestinationBuffersToCommonState(loaded_batch, cancellation, last_submitted_fence));

  result = std::move(loaded_batch);
  load_succeeded = true;
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::AllocateDestinationGpuBuffers(
    const std::vector<uint64_t>& sizes,
    BufferCollection& batch,
    const std::function<common::Status(size_t, ID3D12Resource*)>& on_ready) {
  struct Placement {
    bool committed = false;
    size_t heap_index = 0;
    uint64_t offset = 0;
    D3D12_RESOURCE_DESC description{};
  };

  std::vector<Placement> placements;
  placements.reserve(sizes.size());
  std::vector<uint64_t> heap_sizes;

  for (size_t index = 0; index < sizes.size(); ++index) {
    uint64_t resource_size = 0;
    if (!TryAlignUp(sizes[index], 16, resource_size)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Destination ", index, " size overflows while aligning for Dawn.");
    }
    const auto description = BufferDescription(
        resource_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto allocation_info =
        device_->GetResourceAllocationInfo(0, 1, &description);
    if (allocation_info.SizeInBytes ==
        std::numeric_limits<uint64_t>::max()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "GetResourceAllocationInfo failed for destination ", index, ".");
    }
    if (allocation_info.SizeInBytes > config_.max_destination_heap_size) {
      placements.push_back({true, 0, 0, description});
      continue;
    }

    if (heap_sizes.empty()) {
      heap_sizes.push_back(0);
    }
    size_t heap_index = heap_sizes.size() - 1;
    uint64_t offset = 0;
    if (!TryAlignUp(
            heap_sizes.back(), allocation_info.Alignment, offset)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Destination placement offset overflow.");
    }
    if (heap_sizes.back() != 0 &&
        offset >
            config_.max_destination_heap_size -
                allocation_info.SizeInBytes) {
      heap_sizes.push_back(0);
      heap_index = heap_sizes.size() - 1;
      offset = 0;
    }

    placements.push_back({false, heap_index, offset, description});
    heap_sizes[heap_index] =
        offset + allocation_info.SizeInBytes;
  }

  batch.heaps.resize(heap_sizes.size());
  auto allocate_heap = [&](size_t index) -> common::Status {
    uint64_t heap_size = 0;
    if (!TryAlignUp(
            heap_sizes[index], kBufferAlignment, heap_size) ||
        heap_size > config_.max_destination_heap_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Destination heap size is invalid.");
    }

    D3D12_HEAP_DESC heap_description{};
    heap_description.SizeInBytes = heap_size;
    heap_description.Properties =
        destination_heap_properties_;
    heap_description.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    const HRESULT hr = device_->CreateHeap(
        &heap_description, IID_PPV_ARGS(&batch.heaps[index]));
    if (FAILED(hr)) {
      return HResultError("ID3D12Device::CreateHeap", hr);
    }
    return common::Status::OK();
  };
  // UMA publishes each mapped buffer while later heaps are still being allocated.
  if (!on_ready) {
    for (size_t index = 0; index < heap_sizes.size(); ++index) {
      ORT_RETURN_IF_ERROR(allocate_heap(index));
    }
  }

  const auto initial_state = use_uma_ ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_COPY_DEST;
  batch.buffers.resize(sizes.size());
  for (size_t index = 0; index < placements.size(); ++index) {
    const auto& placement = placements[index];
    HRESULT hr = S_OK;
    if (placement.committed) {
      hr = device_->CreateCommittedResource(
          &destination_heap_properties_, D3D12_HEAP_FLAG_NONE, &placement.description,
          initial_state, nullptr,
          IID_PPV_ARGS(&batch.buffers[index]));
    } else {
      if (!batch.heaps[placement.heap_index]) {
        ORT_RETURN_IF_ERROR(allocate_heap(placement.heap_index));
      }
      hr = device_->CreatePlacedResource(
          batch.heaps[placement.heap_index].Get(),
          placement.offset, &placement.description,
          initial_state, nullptr,
          IID_PPV_ARGS(&batch.buffers[index]));
    }
    if (FAILED(hr)) {
      return HResultError(
          placement.committed
              ? "ID3D12Device::CreateCommittedResource(destination)"
              : "ID3D12Device::CreatePlacedResource",
          hr);
    }
    if (on_ready) {
      ORT_RETURN_IF_ERROR(on_ready(index, batch.buffers[index].Get()));
    }
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::LoadUma(
    const std::vector<BufferSource>& ranges,
    const std::vector<uint64_t>& sizes,
    BufferCollection& result,
    const CancellationToken& cancellation) {
  std::mutex cancellation_mutex;
  auto is_canceled = [&]() {
    std::lock_guard lock(cancellation_mutex);
    return cancellation.IsCancellationRequested();
  };
  struct SourceFile {
    wil::unique_hfile handle;
    uint64_t size = 0;
  };
  std::map<std::wstring, SourceFile> files;
  for (const auto& range : ranges) {
    if (is_canceled()) {
      return CancelledStatus();
    }
    auto [it, inserted] = files.try_emplace(range.path);
    auto& file = it->second;
    if (inserted) {
      file.handle.reset(CreateFileW(
          range.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr));
      if (!file.handle) {
        return Win32FileError("CreateFileW(UMA)", range.path, GetLastError());
      }
      LARGE_INTEGER size{};
      if (!GetFileSizeEx(file.handle.get(), &size)) {
        return Win32FileError("GetFileSizeEx(UMA)", range.path, GetLastError());
      }
      ORT_RETURN_IF(size.QuadPart < 0, "Source file has a negative size.");
      file.size = static_cast<uint64_t>(size.QuadPart);
    }
    ORT_RETURN_IF(range.offset > file.size || range.length > file.size - range.offset,
                  "UMA source range exceeds file bounds.");
  }

  struct ReadJob {
    HANDLE file;
    uint64_t offset;
    uint8_t* destination;
    DWORD size;
  };
  BufferCollection batch;
  InlinedVector<uint8_t*> mappings(ranges.size(), nullptr);
  auto unmap_all = [&]() noexcept {
    // The CPU writes must be visible before these resources are handed to Dawn,
    // including on UMA devices which do not report CacheCoherentUMA.
    MemoryBarrier();
    for (size_t index = 0; index < mappings.size(); ++index) {
      if (mappings[index]) {
        const D3D12_RANGE written{0, static_cast<SIZE_T>(batch.buffers[index]->GetDesc().Width)};
        batch.buffers[index]->Unmap(0, &written);
        mappings[index] = nullptr;
      }
    }
  };
  auto unmap_guard = gsl::finally(unmap_all);
  std::deque<ReadJob> jobs;
  std::mutex jobs_mutex;
  std::condition_variable jobs_ready;
  bool producer_done = false;
  std::atomic<bool> stop{false};
  InlinedVector<std::future<common::Status>> workers;
  workers.reserve(config_.uma_reader_count);
  auto request_stop = [&]() noexcept {
    {
      std::lock_guard lock(jobs_mutex);
      stop.store(true);
    }
    jobs_ready.notify_all();
  };
  // Declared after unmap_guard: join outstanding reads before unmapping or
  // releasing resources on allocation, thread-launch, I/O and cancellation errors.
  auto join_guard = gsl::finally([&]() noexcept {
    request_stop();
    for (auto& worker : workers) {
      if (worker.valid()) {
        worker.wait();
      }
    }
  });

  auto reader = [&]() -> common::Status {
    wil::unique_handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) {
      return Win32Error("CreateEventW(UMA read)", GetLastError());
    }
    for (;;) {
      ReadJob job{};
      {
        std::unique_lock lock(jobs_mutex);
        jobs_ready.wait(lock, [&] { return stop.load() || producer_done || !jobs.empty(); });
        if (stop.load() || jobs.empty()) {
          return common::Status::OK();
        }
        job = jobs.front();
        jobs.pop_front();
      }
      jobs_ready.notify_all();
      if (is_canceled()) {
        return CancelledStatus();
      }
      OVERLAPPED overlapped{};
      overlapped.Offset = static_cast<DWORD>(job.offset);
      overlapped.OffsetHigh = static_cast<DWORD>(job.offset >> 32);
      overlapped.hEvent = event.get();
      if (!ResetEvent(event.get())) {
        return Win32Error("ResetEvent(UMA read)", GetLastError());
      }
      DWORD count = 0;
      if (!ReadFile(job.file, job.destination, job.size, &count, &overlapped) &&
          GetLastError() != ERROR_IO_PENDING) {
        return Win32Error("ReadFile(UMA)", GetLastError());
      }
      for (;;) {
        if (GetOverlappedResultEx(job.file, &overlapped, &count,
                                  kMaximumCancellationPollMilliseconds, FALSE)) {
          break;
        }
        const auto error = GetLastError();
        if (error != WAIT_TIMEOUT) {
          return Win32Error("GetOverlappedResultEx(UMA)", error);
        }
        const bool canceled = is_canceled();
        if (stop.load() || canceled) {
          (void)CancelIoEx(job.file, &overlapped);
          (void)GetOverlappedResult(job.file, &overlapped, &count, TRUE);
          return canceled ? CancelledStatus() : common::Status::OK();
        }
      }
      ORT_RETURN_IF(count != job.size, "Short UMA file read.");
    }
  };
  for (uint32_t index = 0; index < config_.uma_reader_count; ++index) {
    // Own the reader closure until the task finishes, including during exception cleanup.
    workers.push_back(std::async(std::launch::async, [&, reader]() -> common::Status {
      bool succeeded = false;
      auto stop_on_failure = gsl::finally([&]() noexcept {
        if (!succeeded) {
          request_stop();
        }
      });
      common::Status status;
      ORT_TRY {
        status = reader();
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "UMA reader failed: ", ex.what());
        });
      }
      ORT_CATCH(...) {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "UMA reader failed with an unknown exception.");
      }
      succeeded = status.IsOK();
      return status;
    }));
  }

  auto publish = [&](size_t index, ID3D12Resource* resource) -> common::Status {
    if (is_canceled()) {
      return CancelledStatus();
    }
    // Stop allocation promptly; the failing worker's status is collected below.
    if (stop.load()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "UMA reader stopped.");
    }
    ORT_RETURN_IF(resource->GetDesc().Width > std::numeric_limits<SIZE_T>::max(),
                  "UMA resource is too large to map in this process.");
    const D3D12_RANGE no_reads{0, 0};
    const auto hr = resource->Map(0, &no_reads, reinterpret_cast<void**>(&mappings[index]));
    if (FAILED(hr)) {
      return HResultError("ID3D12Resource::Map(UMA)", hr);
    }
    const auto& range = ranges[index];
    const auto file = files.at(range.path).handle.get();
    for (uint64_t offset = 0; offset < range.length;) {
      if (is_canceled()) {
        return CancelledStatus();
      }
      const auto size = static_cast<DWORD>(std::min<uint64_t>(config_.uma_read_size, range.length - offset));
      {
        std::unique_lock lock(jobs_mutex);
        jobs_ready.wait(lock, [&] { return stop.load() || jobs.size() < 2 * config_.uma_reader_count; });
        if (stop.load()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "UMA reader stopped.");
        }
        jobs.push_back({file, range.offset + offset, mappings[index] + offset, size});
      }
      jobs_ready.notify_all();
      offset += size;
    }
    return common::Status::OK();
  };
  const auto allocation_status = AllocateDestinationGpuBuffers(sizes, batch, publish);
  if (!allocation_status.IsOK()) {
    request_stop();
  }
  {
    std::lock_guard lock(jobs_mutex);
    producer_done = true;
  }
  jobs_ready.notify_all();
  common::Status status;
  for (auto& worker : workers) {
    const auto worker_status = worker.get();
    if (status.IsOK() && !worker_status.IsOK()) {
      status = worker_status;
    }
  }
  ORT_RETURN_IF_ERROR(status);
  ORT_RETURN_IF_ERROR(allocation_status);
  if (is_canceled()) {
    return CancelledStatus();
  }
  // Preserve the upload path's padding convention. Workers have finished, so
  // the leading payload bytes are safe to read even when a tensor spans jobs.
  for (size_t index = 0; index < ranges.size(); ++index) {
    const auto length = ranges[index].length;
    const auto width = batch.buffers[index]->GetDesc().Width;
    for (uint64_t offset = length; offset < width; ++offset) {
      mappings[index][offset] = mappings[index][(offset - length) % length];
    }
  }
  unmap_all();
  result = std::move(batch);
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::CreateFileReadPlan(
    const std::vector<BufferSource>& ranges,
    const CancellationToken& cancellation,
    std::vector<FileReadPlan>& files) {
  // Group ranges by source file so each file is opened once. Later, aligned
  // overlapping ranges are merged to avoid redundant unbuffered reads.
  std::map<std::wstring, size_t> file_indices;
  for (size_t range_index = 0;
       range_index < ranges.size(); ++range_index) {
    const auto& range = ranges[range_index];
    const auto [file_index_it, inserted] =
        file_indices.try_emplace(range.path, files.size());
    if (inserted) {
      FileReadPlan file;
      file.path = range.path;
      files.push_back(std::move(file));
    }
    files[file_index_it->second].buffer_source_indices.push_back(range_index);
  }

  for (auto& file : files) {
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }

    file.file.reset(CreateFileW(
        file.path.c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr));
    if (!file.file) {
      return Win32FileError(
          "CreateFileW(unbuffered)", file.path, GetLastError());
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file.file.get(), &file_size)) {
      return Win32Error("GetFileSizeEx", GetLastError());
    }
    if (file_size.QuadPart < 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL, "Source file has a negative size.");
    }
    file.size = static_cast<uint64_t>(file_size.QuadPart);

    FILE_STORAGE_INFO storage_info{};
    if (!GetFileInformationByHandleEx(
            file.file.get(), FileStorageInfo,
            &storage_info, sizeof(storage_info))) {
      return Win32Error(
          "GetFileInformationByHandleEx(FileStorageInfo)",
          GetLastError());
    }
    file.alignment = std::max<uint64_t>(
        storage_info.LogicalBytesPerSector,
        storage_info.PhysicalBytesPerSectorForPerformance);
    if (file.alignment == 0 ||
        (file.alignment & (file.alignment - 1)) != 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Storage alignment is not a non-zero power of two.");
    }
    if (config_.upload_slot_size % file.alignment != 0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "upload_slot_size does not satisfy source file alignment.");
    }
    if (file.size % file.alignment != 0) {
      file.buffered_tail_file.reset(CreateFileW(
          file.path.c_str(), GENERIC_READ, FILE_SHARE_READ,
          nullptr, OPEN_EXISTING,
          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
              FILE_FLAG_SEQUENTIAL_SCAN,
          nullptr));
      if (!file.buffered_tail_file) {
        return Win32FileError(
            "CreateFileW(buffered tail)", file.path, GetLastError());
      }
    }

    std::sort(
        file.buffer_source_indices.begin(), file.buffer_source_indices.end(),
        [&](size_t left, size_t right) {
          return ranges[left].offset < ranges[right].offset;
        });

    for (size_t range_index : file.buffer_source_indices) {
      const auto& range = ranges[range_index];
      if (range.offset > file.size ||
          range.length > file.size - range.offset) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "File range ", range_index,
            " exceeds its source file.");
      }

      uint64_t aligned_end = 0;
      if (!TryAlignUp(
              range.offset + range.length,
              file.alignment, aligned_end)) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Aligned file range overflow.");
      }
      FileReadRegion region{
          AlignDown(range.offset, file.alignment),
          std::min(aligned_end, file.size)};
      if (!file.read_regions.empty() &&
          region.begin <= file.read_regions.back().end) {
        file.read_regions.back().end =
            std::max(file.read_regions.back().end, region.end);
      } else {
        file.read_regions.push_back(region);
      }
    }
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::PrepareUploadStagingSlots(
    uint64_t alignment,
    const CancellationToken& cancellation) {
  for (auto& staging_slot : upload_staging_slots_) {
    ORT_RETURN_IF_ERROR(
        WaitForFence(staging_slot.fence_value, cancellation));
    if (reinterpret_cast<uintptr_t>(staging_slot.mapped) %
            alignment !=
        0) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Mapped upload buffer does not satisfy source file alignment.");
    }
    staging_slot.overlapped = {};
    staging_slot.read_file = INVALID_HANDLE_VALUE;
    staging_slot.file_offset = 0;
    staging_slot.requested_read_bytes = 0;
    staging_slot.fence_value = 0;
    staging_slot.read_active = false;
  }
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::IssueFileRead(
    HANDLE file,
    UploadStagingSlot& staging_slot,
    uint64_t offset,
    DWORD size) {
  if (!ResetEvent(staging_slot.file_read_complete_event.get())) {
    return Win32Error("ResetEvent(read)", GetLastError());
  }
  staging_slot.overlapped = {};
  staging_slot.overlapped.Offset = static_cast<DWORD>(offset);
  staging_slot.overlapped.OffsetHigh =
      static_cast<DWORD>(offset >> 32);
  staging_slot.overlapped.hEvent =
      staging_slot.file_read_complete_event.get();
  staging_slot.file_offset = offset;
  staging_slot.requested_read_bytes = size;
  staging_slot.read_file = file;
  staging_slot.read_active = true;
  if (!ReadFile(
          file, staging_slot.mapped, size, nullptr,
          &staging_slot.overlapped)) {
    const DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      staging_slot.read_active = false;
      staging_slot.read_file = INVALID_HANDLE_VALUE;
      return Win32Error("ReadFile", error);
    }
  }
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::ProcessFileReadCompletion(
    uint64_t file_size,
    UploadStagingSlot& staging_slot,
    DWORD& bytes_read) {
  if (!GetOverlappedResult(
          staging_slot.read_file, &staging_slot.overlapped,
          &bytes_read, FALSE)) {
    return Win32Error(
        "GetOverlappedResult", GetLastError());
  }
  staging_slot.read_active = false;
  staging_slot.read_file = INVALID_HANDLE_VALUE;
  const uint64_t required_bytes =
      std::min<uint64_t>(
          staging_slot.requested_read_bytes,
          file_size - staging_slot.file_offset);
  if (bytes_read < required_bytes) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "Read ended before the required file range.");
  }
  return common::Status::OK();
}

template <typename EnsureAllocationFn>
common::Status D3D12FileBufferLoader::ReadFileRegion(
    FileReadPlan& file,
    const FileReadRegion& region,
    const std::vector<BufferSource>& ranges,
    BufferCollection& batch,
    bool& allocation_ready,
    EnsureAllocationFn& ensure_allocation,
    const CancellationToken& cancellation,
    uint64_t& last_submitted_fence) {
  uint64_t next_offset = region.begin;

  // Each upload staging slot moves through read-active, copy-in-flight, and
  // reusable states. Keep all eligible staging slots busy without reusing one
  // before its previous GPU copy reaches the fence.
  auto issue_available = [&]() -> common::Status {
    for (auto& staging_slot : upload_staging_slots_) {
      if (next_offset >= region.end) {
        break;
      }
      if (staging_slot.read_active) {
        continue;
      }
      if (staging_slot.fence_value != 0 &&
          copy_fence_->GetCompletedValue() <
              staging_slot.fence_value) {
        continue;
      }
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      const DWORD bytes = static_cast<DWORD>(
          std::min<uint64_t>(
              config_.upload_slot_size,
              region.end - next_offset));
      const bool use_buffered_tail =
          next_offset + bytes == file.size &&
          bytes % file.alignment != 0;
      ORT_RETURN_IF_ERROR(
          IssueFileRead(
              use_buffered_tail
                  ? file.buffered_tail_file.get()
                  : file.file.get(),
              staging_slot,
              next_offset, bytes));
      next_offset += bytes;
    }
    return common::Status::OK();
  };

  ORT_RETURN_IF_ERROR(issue_available());
  DWORD cancellation_poll_milliseconds =
      kInitialCancellationPollMilliseconds;
  while (true) {
    std::array<HANDLE, MAXIMUM_WAIT_OBJECTS> handles{};
    std::array<size_t, MAXIMUM_WAIT_OBJECTS> staging_slot_indices{};
    DWORD handle_count = 0;
    for (size_t index = 0;
         index < upload_staging_slots_.size(); ++index) {
      if (upload_staging_slots_[index].read_active) {
        handles[handle_count] =
            upload_staging_slots_[index].file_read_complete_event.get();
        staging_slot_indices[handle_count] = index;
        ++handle_count;
      }
    }

    if (handle_count == 0) {
      if (next_offset >= region.end) {
        break;
      }
      uint64_t next_fence =
          std::numeric_limits<uint64_t>::max();
      for (const auto& staging_slot : upload_staging_slots_) {
        if (!staging_slot.read_active &&
            staging_slot.fence_value != 0) {
          next_fence =
              std::min(next_fence, staging_slot.fence_value);
        }
      }
      if (next_fence ==
          std::numeric_limits<uint64_t>::max()) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "Event-driven read scheduler made no progress.");
      }
      ORT_RETURN_IF_ERROR(
          WaitForFence(next_fence, cancellation));
      ORT_RETURN_IF_ERROR(issue_available());
      continue;
    }

    const DWORD wait_result = WaitForMultipleObjects(
        handle_count, handles.data(), FALSE,
        cancellation_poll_milliseconds);
    if (wait_result == WAIT_TIMEOUT) {
      if (cancellation.IsCancellationRequested()) {
        return CancelledStatus();
      }
      cancellation_poll_milliseconds =
          std::min(
              cancellation_poll_milliseconds * 2,
              kMaximumCancellationPollMilliseconds);
      continue;
    }
    if (wait_result < WAIT_OBJECT_0 ||
        wait_result >= WAIT_OBJECT_0 + handle_count) {
      return Win32Error(
          "WaitForMultipleObjects(read)",
          GetLastError());
    }

    auto& staging_slot =
        upload_staging_slots_[staging_slot_indices[wait_result - WAIT_OBJECT_0]];
    DWORD bytes_read = 0;
    ORT_RETURN_IF_ERROR(
        ProcessFileReadCompletion(
            file.size, staging_slot, bytes_read));

    if (!allocation_ready) {
      ORT_RETURN_IF_ERROR(ensure_allocation());
    }
    ORT_RETURN_IF_ERROR(
        SubmitStagingToDestinationCopies(
            staging_slot, staging_slot.file_offset, bytes_read,
            file.buffer_source_indices, ranges, batch,
            last_submitted_fence));
    cancellation_poll_milliseconds =
        kInitialCancellationPollMilliseconds;
    ORT_RETURN_IF_ERROR(issue_available());
  }

  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::SubmitStagingToDestinationCopies(
    UploadStagingSlot& staging_slot,
    uint64_t chunk_begin,
    uint64_t bytes,
    const std::vector<size_t>& buffer_source_indices,
    const std::vector<BufferSource>& ranges,
    BufferCollection& batch,
    uint64_t& last_submitted_fence) {
  HRESULT hr = staging_slot.allocator->Reset();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandAllocator::Reset", hr);
  }
  hr = staging_slot.command_list->Reset(
      staging_slot.allocator.Get(), nullptr);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Reset", hr);
  }

  const uint64_t chunk_end = chunk_begin + bytes;
  for (size_t range_index : buffer_source_indices) {
    const auto& range = ranges[range_index];
    const uint64_t range_end =
        range.offset + range.length;
    const uint64_t copy_begin =
        std::max(chunk_begin, range.offset);
    const uint64_t copy_end =
        std::min(chunk_end, range_end);
    if (copy_begin < copy_end) {
      staging_slot.command_list->CopyBufferRegion(
          batch.buffers[range_index].Get(),
          copy_begin - range.offset,
          staging_slot.resource.Get(),
          copy_begin - chunk_begin,
          copy_end - copy_begin);
    }

    // Dawn buffer sizes are 16-byte aligned. Initialize the tail by
    // repeating source bytes so no padding remains uninitialized.
    const uint64_t aligned_size =
        (range.length + 15) & ~uint64_t{15};
    for (uint64_t padding_offset = range.length;
         padding_offset < aligned_size;) {
      const uint64_t source_relative_offset =
          (padding_offset - range.length) % range.length;
      const uint64_t source_offset =
          range.offset + source_relative_offset;
      if (source_offset >= chunk_begin &&
          source_offset < chunk_end) {
        const uint64_t copy_size = std::min(
            {aligned_size - padding_offset,
             range.length - source_relative_offset,
             chunk_end - source_offset});
        staging_slot.command_list->CopyBufferRegion(
            batch.buffers[range_index].Get(),
            padding_offset,
            staging_slot.resource.Get(),
            source_offset - chunk_begin,
            copy_size);
        padding_offset += copy_size;
      } else {
        ++padding_offset;
      }
    }
  }

  hr = staging_slot.command_list->Close();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Close", hr);
  }
  ID3D12CommandList* command_lists[] = {
      staging_slot.command_list.Get()};
  copy_queue_->ExecuteCommandLists(1, command_lists);

  const uint64_t fence_value = ++next_fence_value_;
  ORT_RETURN_IF_ERROR(SignalSubmittedWork(fence_value, batch));
  staging_slot.fence_value = fence_value;
  last_submitted_fence = fence_value;
  return common::Status::OK();
}

common::Status D3D12FileBufferLoader::TransitionDestinationBuffersToCommonState(
    BufferCollection& batch,
    const CancellationToken& cancellation,
    uint64_t& last_submitted_fence) {
  if (batch.buffers.empty()) {
    return common::Status::OK();
  }

  auto& staging_slot = upload_staging_slots_.front();
  HRESULT hr = staging_slot.allocator->Reset();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12CommandAllocator::Reset(transition)", hr);
  }
  hr = staging_slot.command_list->Reset(
      staging_slot.allocator.Get(), nullptr);
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Reset(transition)", hr);
  }

  std::vector<D3D12_RESOURCE_BARRIER> barriers;
  barriers.reserve(batch.buffers.size());
  for (const auto& buffer : batch.buffers) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type =
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource =
        buffer.Get();
    barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore =
        D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_COMMON;
    barriers.push_back(barrier);
  }
  staging_slot.command_list->ResourceBarrier(
      static_cast<UINT>(barriers.size()),
      barriers.data());

  hr = staging_slot.command_list->Close();
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12GraphicsCommandList::Close(transition)", hr);
  }
  ID3D12CommandList* command_lists[] = {
      staging_slot.command_list.Get()};
  copy_queue_->ExecuteCommandLists(1, command_lists);

  const uint64_t fence_value = ++next_fence_value_;
  ORT_RETURN_IF_ERROR(SignalSubmittedWork(fence_value, batch));
  last_submitted_fence = fence_value;
  return WaitForFence(
      last_submitted_fence, cancellation);
}

common::Status D3D12FileBufferLoader::SignalSubmittedWork(
    uint64_t value, BufferCollection& batch) {
  // ExecuteCommandLists has no status return. Once work has been submitted,
  // its resources must stay alive until either a fence is queued behind it or
  // the device is removed. If Signal fails while the device is still live,
  // completion is unknown. Detach destination resources now; the destructor
  // detaches the remaining GPU resources to keep them alive for process lifetime.
  const HRESULT signal_result =
      copy_queue_->Signal(copy_fence_.Get(), value);
  if (SUCCEEDED(signal_result)) {
    return common::Status::OK();
  }
  const HRESULT removal_reason =
      device_->GetDeviceRemovedReason();
  if (FAILED(removal_reason)) {
    return HResultError(
        "D3D12 device removed after command submission",
        removal_reason);
  }
  upload_error_gpu_completion_unknown_ = true;
  for (auto& buffer : batch.buffers) {
    buffer.Detach();
  }
  for (auto& heap : batch.heaps) {
    heap.Detach();
  }
  batch.Clear();
  return HResultError(
      "ID3D12CommandQueue::Signal after command submission",
      signal_result);
}

common::Status D3D12FileBufferLoader::WaitForFence(
    uint64_t value,
    const CancellationToken& cancellation) {
  if (value == 0) {
    return common::Status::OK();
  }
  uint64_t completed_value =
      copy_fence_->GetCompletedValue();
  if (completed_value ==
      std::numeric_limits<uint64_t>::max()) {
    return HResultError(
        "D3D12 device removed while waiting for copy fence",
        device_->GetDeviceRemovedReason());
  }
  if (completed_value >= value) {
    return common::Status::OK();
  }
  const HRESULT hr =
      copy_fence_->SetEventOnCompletion(
          value, copy_fence_complete_event_.get());
  if (FAILED(hr)) {
    return HResultError(
        "ID3D12Fence::SetEventOnCompletion", hr);
  }

  DWORD cancellation_poll_milliseconds =
      kInitialCancellationPollMilliseconds;
  while (true) {
    const DWORD wait_result = WaitForSingleObject(
        copy_fence_complete_event_.get(),
        cancellation_poll_milliseconds);
    if (wait_result != WAIT_OBJECT_0 &&
        wait_result != WAIT_TIMEOUT) {
      return Win32Error(
          "WaitForSingleObject(copy fence)",
          GetLastError());
    }
    completed_value = copy_fence_->GetCompletedValue();
    if (completed_value ==
        std::numeric_limits<uint64_t>::max()) {
      return HResultError(
          "D3D12 device removed while waiting for copy fence",
          device_->GetDeviceRemovedReason());
    }
    if (completed_value >= value) {
      return common::Status::OK();
    }
    if (cancellation.IsCancellationRequested()) {
      return CancelledStatus();
    }
    cancellation_poll_milliseconds =
        std::min(
            cancellation_poll_milliseconds * 2,
            kMaximumCancellationPollMilliseconds);
  }
}

void D3D12FileBufferLoader::WaitForFenceUncancelled(
    uint64_t value) noexcept {
  if (value == 0 ||
      copy_fence_->GetCompletedValue() >= value) {
    return;
  }
  if (SUCCEEDED(copy_fence_->SetEventOnCompletion(
          value, copy_fence_complete_event_.get())) &&
      WaitForSingleObject(
          copy_fence_complete_event_.get(), INFINITE) ==
          WAIT_OBJECT_0) {
    return;
  }

  while (true) {
    const uint64_t completed =
        copy_fence_->GetCompletedValue();
    if (completed >= value ||
        completed == std::numeric_limits<uint64_t>::max()) {
      return;
    }
    Sleep(1);
  }
}

void D3D12FileBufferLoader::DrainActiveReads() noexcept {
  for (auto& staging_slot : upload_staging_slots_) {
    if (!staging_slot.read_active) {
      continue;
    }
    (void)CancelIoEx(
        staging_slot.read_file, &staging_slot.overlapped);
  }
  for (auto& staging_slot : upload_staging_slots_) {
    if (!staging_slot.read_active) {
      continue;
    }
    DWORD bytes_read = 0;
    (void)GetOverlappedResult(
        staging_slot.read_file, &staging_slot.overlapped,
        &bytes_read, TRUE);
    staging_slot.read_active = false;
    staging_slot.read_file = INVALID_HANDLE_VALUE;
  }
}

common::Status D3D12FileBufferLoader::Create(
    ID3D12Device* device,
    std::unique_ptr<D3D12FileBufferLoader>& loader) noexcept {
  return Create(device, loader, Config{});
}

common::Status D3D12FileBufferLoader::Create(
    ID3D12Device* device,
    std::unique_ptr<D3D12FileBufferLoader>& loader,
    const Config& config) noexcept {
  loader.reset();
  common::Status status;
  ORT_TRY {
    auto instance = std::unique_ptr<D3D12FileBufferLoader>(
        new D3D12FileBufferLoader(device, config));
    ORT_RETURN_IF_ERROR(instance->Initialize());
    loader = std::move(instance);
    return common::Status::OK();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(
          ONNXRUNTIME, RUNTIME_EXCEPTION,
          "D3D12FileBufferLoader::Create failed: ",
          ex.what());
    });
  }
  ORT_CATCH(...) {
    status = ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Create failed with an unknown exception.");
  }
  return status;
}

common::Status D3D12FileBufferLoader::Load(
    const std::vector<BufferSource>& ranges,
    BufferCollection& result) noexcept {
  return Load(ranges, result, CancellationToken{});
}

common::Status D3D12FileBufferLoader::Load(
    const std::vector<BufferSource>& ranges,
    BufferCollection& result,
    const CancellationToken& cancellation) noexcept {
  common::Status status;
  ORT_TRY {
    return LoadInternal(ranges, result, cancellation);
  }
  ORT_CATCH(const std::exception& ex) {
    result.Clear();
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(
          ONNXRUNTIME, RUNTIME_EXCEPTION,
          "D3D12FileBufferLoader::Load failed: ",
          ex.what());
    });
  }
  ORT_CATCH(...) {
    result.Clear();
    status = ORT_MAKE_STATUS(
        ONNXRUNTIME, RUNTIME_EXCEPTION,
        "D3D12FileBufferLoader::Load failed with an unknown exception.");
  }
  return status;
}

}  // namespace d3d12
}  // namespace windows
}  // namespace onnxruntime
