// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/framework/tensor.h"
#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/d3d12_external_data_loader.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

using Microsoft::WRL::ComPtr;
using D3D12FileBufferLoader = windows::d3d12::D3D12FileBufferLoader;

namespace {

using Clock = std::chrono::steady_clock;

struct SharedBufferMemoryD3D12ResourceDescriptor : wgpu::ChainedStruct {
  SharedBufferMemoryD3D12ResourceDescriptor() {
    sType = static_cast<wgpu::SType>(
        WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor);
  }

  ComPtr<ID3D12Resource> resource;
};

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

struct TensorKey {
  std::filesystem::path path;
  std::string name;
  uint64_t offset;
  size_t length;

  bool operator<(const TensorKey& other) const {
    if (path < other.path) {
      return true;
    }
    if (other.path < path) {
      return false;
    }
    if (name != other.name) {
      return name < other.name;
    }
    if (offset != other.offset) {
      return offset < other.offset;
    }
    return length < other.length;
  }
};

struct AcceleratedTensorLoadEntry {
  TensorKey key;
  ComPtr<ID3D12Resource> resource;
  std::shared_ptr<D3D12FileBufferLoader::BufferCollection> heap_owner;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
  bool claimed = false;
};

}  // namespace

struct D3D12AcceleratedLoadBatch {
  struct FileInfo {
    std::filesystem::path canonical_path;
    size_t length;
  };

  std::vector<AcceleratedTensorLoadEntry> tensors;
  std::map<TensorKey, size_t> tensors_by_key;
  std::map<std::filesystem::path, FileInfo> files;
  size_t total_bytes = 0;
  size_t load_range_count = 0;
  bool load_candidates_committed = false;
};

namespace {

struct D3D12AcceleratedLoadMetrics {
  double load_ms = 0.0;
  size_t loaded_ranges = 0;
};

struct CancellationState {
  const std::function<bool()>* external = nullptr;
  const std::atomic<bool>* load_end_requested = nullptr;
};

bool InvokeCancellationNoThrow(
    const std::function<bool()>& is_canceled) noexcept {
  if (!is_canceled) {
    return false;
  }
  ORT_TRY {
    return is_canceled();
  }
  ORT_CATCH(...) {
    return true;
  }
}

bool IsCancellationRequested(void* opaque) noexcept {
  const auto& state = *static_cast<const CancellationState*>(opaque);
  if (state.load_end_requested != nullptr &&
      state.load_end_requested->load(std::memory_order_relaxed)) {
    return true;
  }
  if (state.external == nullptr || !*state.external) {
    return false;
  }
  return InvokeCancellationNoThrow(*state.external);
}

common::Status LoadBatchToD3D12(
    D3D12AcceleratedLoadBatch& batch,
    D3D12FileBufferLoader& loader,
    const std::function<bool()>& is_canceled,
    const std::atomic<bool>& load_end_requested,
    D3D12AcceleratedLoadMetrics& metrics) {
  if (load_end_requested.load(std::memory_order_relaxed) ||
      InvokeCancellationNoThrow(is_canceled)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "D3D12 accelerated initializer loading was canceled.");
  }

  std::vector<D3D12FileBufferLoader::BufferSource> ranges;
  std::vector<size_t> tensor_indices;
  ranges.reserve(batch.load_range_count);
  tensor_indices.reserve(batch.load_range_count);
  for (size_t index = 0; index < batch.tensors.size(); ++index) {
    const auto& tensor = batch.tensors[index];
    if (tensor.key.length == 0) {
      continue;
    }
    ranges.push_back(
        {tensor.key.path.native(),
         tensor.key.offset,
         static_cast<uint64_t>(tensor.key.length)});
    tensor_indices.push_back(index);
  }

  ORT_RETURN_IF_NOT(
      ranges.size() == batch.load_range_count,
      "D3D12 accelerated initializer range count is inconsistent.");
  if (ranges.empty()) {
    return common::Status::OK();
  }

  auto loaded = std::make_shared<D3D12FileBufferLoader::BufferCollection>();
  const CancellationState cancellation_state{
      &is_canceled, &load_end_requested};
  const D3D12FileBufferLoader::CancellationToken cancellation{
      IsCancellationRequested,
      const_cast<CancellationState*>(&cancellation_state)};

  const auto load_start = Clock::now();
  ORT_RETURN_IF_ERROR(loader.Load(ranges, *loaded, cancellation));
  const auto load_end = Clock::now();
  ORT_RETURN_IF_NOT(
      loaded->buffers.size() == tensor_indices.size(),
      "D3D12 file loader returned an unexpected buffer count.");

  for (size_t index = 0; index < tensor_indices.size(); ++index) {
    auto& tensor = batch.tensors[tensor_indices[index]];
    tensor.resource = loaded->buffers[index];
    tensor.heap_owner = loaded;
  }

  metrics.load_ms = Milliseconds(load_end - load_start);
  metrics.loaded_ranges = ranges.size();
  return common::Status::OK();
}

TensorKey MakeKey(const std::filesystem::path& path,
                  std::string_view name,
                  uint64_t offset,
                  size_t length) {
  return {path, std::string{name}, offset, length};
}

common::Status RegisterLoadCandidateForBatch(
    D3D12AcceleratedLoadBatch& batch,
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) {
  ORT_RETURN_IF(
      data_offset < 0,
      "D3D12 accelerated initializer \"", tensor_name,
      "\" has a negative file offset.");

  const size_t length = static_cast<size_t>(data_length);
  const uint64_t offset = static_cast<uint64_t>(data_offset);
  ORT_RETURN_IF(
      offset > std::numeric_limits<uint64_t>::max() -
                   static_cast<uint64_t>(length),
      "D3D12 accelerated initializer \"", tensor_name,
      "\" file range overflows.");

  const auto file_key = data_file_path.lexically_normal();
  auto file_iterator = batch.files.find(file_key);
  if (file_iterator == batch.files.end()) {
    size_t file_length = 0;
    ORT_RETURN_IF_ERROR(
        env.GetFileLength(data_file_path.c_str(), file_length));
    std::error_code path_error;
    const auto normalized_path =
        std::filesystem::weakly_canonical(
            data_file_path, path_error);
    ORT_RETURN_IF(
        path_error,
        "Failed to canonicalize D3D12 data file \"",
        data_file_path.string(), "\": ",
        path_error.message());
    file_iterator =
        batch.files
            .emplace(
                file_key,
                D3D12AcceleratedLoadBatch::FileInfo{
                    normalized_path, file_length})
            .first;
  }

  const size_t file_length = file_iterator->second.length;
  ORT_RETURN_IF(
      offset > static_cast<uint64_t>(file_length) ||
          static_cast<uint64_t>(length) >
              static_cast<uint64_t>(file_length) - offset,
      "D3D12 accelerated initializer \"", tensor_name,
      "\" range [", offset, ", ", offset + length,
      ") exceeds file size ", file_length, " for \"",
      data_file_path.string(), "\".");

  TensorKey key = MakeKey(
      file_iterator->second.canonical_path,
      tensor_name, offset, length);
  ORT_RETURN_IF(
      batch.tensors_by_key.find(key) !=
          batch.tensors_by_key.end(),
      "Duplicate D3D12 accelerated initializer \"",
      tensor_name, "\" from \"", key.path.string(),
      "\" range [", offset, ", ", offset + length, ").");
  ORT_RETURN_IF(
      batch.total_bytes >
          std::numeric_limits<size_t>::max() - length,
      "D3D12 accelerated initializer byte count overflow.");

  const size_t index = batch.tensors.size();
  batch.tensors.push_back({std::move(key)});
  batch.tensors_by_key.emplace(
      batch.tensors.back().key, index);
  batch.total_bytes += length;
  if (length != 0) {
    ++batch.load_range_count;
  }
  return common::Status::OK();
}

struct ImportedAllocation {
  ComPtr<ID3D12Resource> resource;
  std::shared_ptr<D3D12FileBufferLoader::BufferCollection> heap_owner;
  wgpu::SharedBufferMemory memory;
  wgpu::Buffer buffer;
  bool access_started = false;
};

void EndAccessNoThrow(
    ImportedAllocation& allocation) noexcept {
  if (!allocation.access_started ||
      !allocation.memory || !allocation.buffer) {
    return;
  }

  ORT_TRY {
    wgpu::SharedBufferMemoryEndAccessState end_state{};
    allocation.memory.EndAccess(
        allocation.buffer, &end_state);
    allocation.access_started = false;
  }
  ORT_CATCH(...) {
    // Cleanup paths, including allocator Free, must not propagate Dawn failures.
  }
}

}  // namespace

common::Status CheckD3D12AcceleratedExternalWeightsSupport(
    const WebGpuContext& context) {
  ORT_RETURN_IF_NOT(
      context.SelectedBackendType() ==
          wgpu::BackendType::D3D12,
      "D3D12 accelerated external weights require the Dawn D3D12 backend.");
  ORT_RETURN_IF_NOT(
      context.D3D12SharedResourceFeaturesAvailable(),
      "D3D12 accelerated external weights require Dawn "
      "SharedBufferMemoryD3D12Resource and SharedFenceDXGISharedHandle features.");

  ORT_RETURN_IF_NOT(
      context.WeightLoadingD3D12Device() != nullptr,
      "Failed to create a D3D12 device for accelerated file loading.");
  return common::Status::OK();
}

common::Status ResolveWeightLoadAccelerationMode(
    WeightLoadAccelerationMode mode,
    const common::Status& support_status,
    bool& enabled) {
  enabled = false;
  if (!IsWeightLoadAccelerationEnabled(mode)) {
    return common::Status::OK();
  }
  if (support_status.IsOK()) {
    enabled = true;
    return common::Status::OK();
  }
  if (IsWeightLoadAccelerationRequired(mode)) {
    return support_status;
  }
  return common::Status::OK();
}

// Shared by the loader and allocator so imported resources outlive initializer loading.
class D3D12ImportedBufferRegistry {
 public:
  ~D3D12ImportedBufferRegistry();

  std::mutex mutex;
  IAllocator* allocator = nullptr;
  std::unordered_map<
      WGPUBuffer,
      std::unique_ptr<ImportedAllocation>>
      imported_allocations;
};

D3D12ImportedBufferRegistry::
    ~D3D12ImportedBufferRegistry() {
  std::vector<std::unique_ptr<ImportedAllocation>>
      allocations;
  {
    std::lock_guard<std::mutex> lock{mutex};
    allocations.reserve(
        imported_allocations.size());
    for (auto& entry : imported_allocations) {
      allocations.push_back(std::move(entry.second));
    }
    imported_allocations.clear();
  }
  for (auto& allocation : allocations) {
    EndAccessNoThrow(*allocation);
  }
}

class D3D12AcceleratedWebGpuAllocator final
    : public IAllocator {
 public:
  D3D12AcceleratedWebGpuAllocator(
      WebGpuContext& context,
      int context_id,
      std::function<CommandRecordingState&()>
          recording_getter,
      std::shared_ptr<D3D12ImportedBufferRegistry>
          buffer_registry)
      : IAllocator(
            OrtMemoryInfo(
                WEBGPU_BUFFER,
                OrtAllocatorType::OrtReadOnlyAllocator,
                WebGpuDevice(context_id),
                OrtMemTypeDefault)),
        context_{context},
        recording_getter_{
            std::move(recording_getter)},
        buffer_registry_{std::move(buffer_registry)} {
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    const auto usage =
        wgpu::BufferUsage::Storage |
        wgpu::BufferUsage::CopySrc |
        wgpu::BufferUsage::CopyDst |
        wgpu::BufferUsage::Indirect;
    auto& recording = recording_getter_();
    return context_.InitializerBufferManager().Create(
        recording, size, usage);
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    auto& recording = recording_getter_();
    std::unique_ptr<ImportedAllocation> imported;
    {
      std::lock_guard<std::mutex> lock{
          buffer_registry_->mutex};
      const auto iterator =
          buffer_registry_->imported_allocations.find(
              static_cast<WGPUBuffer>(p));
      if (iterator !=
          buffer_registry_->imported_allocations.end()) {
        imported = std::move(iterator->second);
        buffer_registry_->imported_allocations.erase(
            iterator);
      }
    }

    if (imported) {
      if (recording.has_unsubmitted_work.load(
              std::memory_order_acquire)) {
        std::shared_ptr<ImportedAllocation> deferred{
            imported.release(),
            [](ImportedAllocation* allocation) {
              EndAccessNoThrow(*allocation);
              delete allocation;
            }};
        {
          std::lock_guard<std::mutex> lock{
              recording.pending_release_callbacks_mutex};
          if (recording.has_unsubmitted_work.load(
                  std::memory_order_acquire)) {
            recording.pending_release_callbacks.emplace_back(
                [deferred]() mutable {
                  deferred.reset();
                });
          }
        }
        deferred.reset();
      } else {
        EndAccessNoThrow(*imported);
      }
      return;
    }

    context_.InitializerBufferManager().Release(
        static_cast<WGPUBuffer>(p), &recording);
  }

 private:
  WebGpuContext& context_;
  std::function<CommandRecordingState&()>
      recording_getter_;
  std::shared_ptr<D3D12ImportedBufferRegistry>
      buffer_registry_;
};

AllocatorPtr CreateD3D12AcceleratedWebGpuAllocator(
    WebGpuContext& context,
    int context_id,
    std::function<CommandRecordingState&()>
        recording_getter,
    std::shared_ptr<D3D12ImportedBufferRegistry>&
        out_buffer_registry) {
  auto buffer_registry =
      std::make_shared<D3D12ImportedBufferRegistry>();
  auto allocator =
      std::make_shared<
          D3D12AcceleratedWebGpuAllocator>(
          context, context_id, std::move(recording_getter), buffer_registry);
  buffer_registry->allocator = allocator.get();
  out_buffer_registry = buffer_registry;
  return allocator;
}

void D3D12AcceleratedExternalDataLoader::ResolveSupport() const {
  std::call_once(device_support_resolution_once_, [this]() {
    common::Status support_status =
        common::Status::OK();
    if (IsWeightLoadAccelerationEnabled(acceleration_mode_)) {
      support_status =
          CheckD3D12AcceleratedExternalWeightsSupport(
              context_);
    }
    device_support_resolution_status_ =
        ResolveWeightLoadAccelerationMode(
            acceleration_mode_, support_status, resolved_acceleration_enabled_);
    if (device_support_resolution_status_.IsOK() && !resolved_acceleration_enabled_ &&
        acceleration_mode_ == WeightLoadAccelerationMode::Preferred) {
      LOGS_DEFAULT(WARNING)
          << "D3D12 accelerated external weights are unavailable; "
             "using the ordinary WebGPU initializer loading path. "
             "Reason: "
          << support_status.ErrorMessage();
    }
  });
}

common::Status D3D12AcceleratedExternalDataLoader::EnsureFileLoader() const {
  if (file_to_buffer_loader_) {
    return common::Status::OK();
  }
  D3D12FileBufferLoader::Config config;
  config.enable_uma = enable_uma_weight_loading_;
  return D3D12FileBufferLoader::Create(
      context_.WeightLoadingD3D12Device(), file_to_buffer_loader_, config);
}

D3D12AcceleratedExternalDataLoader::
    D3D12AcceleratedExternalDataLoader(
        WebGpuContext& context,
        int context_id,
        std::shared_ptr<
            D3D12ImportedBufferRegistry>
            buffer_registry,
        WeightLoadAccelerationMode mode,
        bool enable_uma_weight_loading)
    : context_{context},
      context_id_{context_id},
      buffer_registry_{std::move(buffer_registry)},
      acceleration_mode_{mode},
      enable_uma_weight_loading_{enable_uma_weight_loading} {
  ORT_ENFORCE(
      buffer_registry_ != nullptr,
      "D3D12 imported buffer registry is required.");
}

D3D12AcceleratedExternalDataLoader::
    ~D3D12AcceleratedExternalDataLoader() {
  EndLoad();
}

bool D3D12AcceleratedExternalDataLoader::CanLoad(
    const OrtMemoryInfo& target_memory_info) const {
  ResolveSupport();
  return target_memory_info.device == WebGpuDevice(context_id_) &&
         target_memory_info.name == WEBGPU_BUFFER &&
         resolved_acceleration_enabled_;
}

bool D3D12AcceleratedExternalDataLoader::UsesUMA() const noexcept {
  return file_to_buffer_loader_ != nullptr && file_to_buffer_loader_->UsesUMA();
}

bool D3D12AcceleratedExternalDataLoader::
    SupportsDataType(
        int32_t tensor_data_type) const {
  return tensor_data_type !=
         ONNX_NAMESPACE::TensorProto_DataType_BOOL;
}

bool D3D12AcceleratedExternalDataLoader::
    CreatesTensorForDevice(
        const OrtDevice& target_device) const {
  ResolveSupport();
  return target_device == WebGpuDevice(context_id_) &&
         (resolved_acceleration_enabled_ ||
          IsWeightLoadAccelerationRequired(acceleration_mode_));
}

common::Status
D3D12AcceleratedExternalDataLoader::BeginLoad() const {
  EndLoad();
  ResolveSupport();
  ORT_RETURN_IF_ERROR(device_support_resolution_status_);
  if (!resolved_acceleration_enabled_) {
    return common::Status::OK();
  }
  load_end_requested_.store(
      false, std::memory_order_relaxed);
  accelerated_load_batch_ =
      std::make_unique<D3D12AcceleratedLoadBatch>();
  return common::Status::OK();
}

common::Status
D3D12AcceleratedExternalDataLoader::RegisterLoadCandidate(
    const Env& env,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length) const {
  ORT_RETURN_IF_NOT(
      accelerated_load_batch_ != nullptr &&
          !accelerated_load_batch_->load_candidates_committed,
      "D3D12 accelerated initializer batch has not been started.");
  return RegisterLoadCandidateForBatch(
      *accelerated_load_batch_, env, data_file_path,
      tensor_name, data_offset, data_length);
}

common::Status
D3D12AcceleratedExternalDataLoader::CommitLoadCandidates(
    const std::function<bool()>& is_canceled) const {
  if (!resolved_acceleration_enabled_) {
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(
      accelerated_load_batch_ != nullptr &&
          !accelerated_load_batch_->load_candidates_committed,
      "D3D12 accelerated initializer batch has not been started.");
  auto& batch = *accelerated_load_batch_;
  const auto fail_or_fallback =
      [this, &is_canceled](
          const common::Status& status)
      -> common::Status {
    if (status.Code() ==
            common::MODEL_LOAD_CANCELED ||
        InvokeCancellationNoThrow(is_canceled)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, MODEL_LOAD_CANCELED,
          "D3D12 accelerated initializer loading was canceled.");
    }
    if (IsWeightLoadAccelerationRequired(
            acceleration_mode_)) {
      return status;
    }
    LOGS_DEFAULT(WARNING)
        << "D3D12 accelerated initializer loading failed; "
           "using the ordinary WebGPU initializer path: "
        << status.ErrorMessage();
    resolved_acceleration_enabled_ = false;
    EndLoad();
    return common::Status::OK();
  };

  if (batch.load_range_count == 0) {
    if (InvokeCancellationNoThrow(is_canceled)) {
      return fail_or_fallback(ORT_MAKE_STATUS(
          ONNXRUNTIME, MODEL_LOAD_CANCELED,
          "D3D12 accelerated initializer loading was canceled."));
    }
    batch.load_candidates_committed = true;
    return common::Status::OK();
  }

  D3D12AcceleratedLoadMetrics load_metrics;
  const auto loader_status = EnsureFileLoader();
  if (!loader_status.IsOK()) {
    return fail_or_fallback(loader_status);
  }
  const auto load_status = LoadBatchToD3D12(
      batch, *file_to_buffer_loader_, is_canceled,
      load_end_requested_, load_metrics);
  if (!load_status.IsOK()) {
    return fail_or_fallback(load_status);
  }

  if (InvokeCancellationNoThrow(is_canceled)) {
    return fail_or_fallback(ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "D3D12 accelerated initializer loading was canceled."));
  }

  const auto import_start = Clock::now();
  const auto import_status =
      [&]() -> common::Status {
    ORT_RETURN_IF_NOT(
        context_.D3D12SharedResourceFeaturesAvailable(),
        "D3D12 accelerated external weights require Dawn "
        "SharedBufferMemoryD3D12Resource and SharedFenceDXGISharedHandle features.");
    for (auto& tensor : batch.tensors) {
      if (InvokeCancellationNoThrow(is_canceled)) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, MODEL_LOAD_CANCELED,
            "D3D12 accelerated initializer loading was canceled.");
      }
      if (tensor.key.length == 0) {
        continue;
      }

      SharedBufferMemoryD3D12ResourceDescriptor
          resource_descriptor;
      resource_descriptor.resource = tensor.resource;
      wgpu::SharedBufferMemoryDescriptor
          memory_descriptor{};
      memory_descriptor.label =
          tensor.key.name.c_str();
      memory_descriptor.nextInChain =
          &resource_descriptor;
      tensor.memory =
          context_.Device()
              .ImportSharedBufferMemory(
                  &memory_descriptor);
      ORT_RETURN_IF_NOT(
          tensor.memory,
          "Failed to import D3D12 accelerated initializer \"",
          tensor.key.name, "\" into Dawn.");

      wgpu::SharedBufferMemoryProperties properties{};
      ORT_RETURN_IF_NOT(
          tensor.memory.GetProperties(&properties) ==
              wgpu::Status::Success,
          "Failed to query imported D3D12 accelerated initializer \"",
          tensor.key.name, "\".");
      ORT_RETURN_IF(
          properties.size <
              static_cast<uint64_t>(
                  tensor.key.length),
          "Imported D3D12 accelerated initializer \"",
          tensor.key.name,
          "\" is smaller than its tensor payload.");

      wgpu::BufferDescriptor buffer_descriptor{};
      buffer_descriptor.label =
          tensor.key.name.c_str();
      // Expose the initialized 16-byte-aligned resource so WebGPU's
      // 4-byte-normalized copies can access tensor tail padding safely. The
      // Tensor retains the logical payload length.
      buffer_descriptor.size = properties.size;
      buffer_descriptor.usage =
          wgpu::BufferUsage::Storage |
          wgpu::BufferUsage::CopySrc |
          wgpu::BufferUsage::CopyDst;
      tensor.buffer = tensor.memory.CreateBuffer(
          &buffer_descriptor);
      ORT_RETURN_IF_NOT(
          tensor.buffer,
          "Failed to create buffer for D3D12 accelerated initializer \"",
          tensor.key.name, "\".");

      wgpu::SharedBufferMemoryBeginAccessDescriptor
          access_descriptor{};
      access_descriptor.initialized = true;
      ORT_RETURN_IF_NOT(
          tensor.memory.BeginAccess(
              tensor.buffer,
              &access_descriptor) ==
              wgpu::Status::Success,
          "Failed to begin Dawn access for D3D12 accelerated initializer \"",
          tensor.key.name, "\".");
      tensor.access_started = true;
    }
    return common::Status::OK();
  }();
  if (!import_status.IsOK()) {
    return fail_or_fallback(import_status);
  }
  const auto import_end = Clock::now();
  batch.load_candidates_committed = true;

  LOGS_DEFAULT(VERBOSE)
      << "WebGPU D3D12 accelerated external initializer load: "
      << "file-to-D3D12=" << load_metrics.load_ms
      << " ms, Dawn import/access="
      << Milliseconds(import_end - import_start)
      << " ms, bytes=" << batch.total_bytes
      << ", tensors=" << batch.tensors.size()
      << ", loaded_ranges="
      << load_metrics.loaded_ranges
      << ", destination=" << (UsesUMA() ? "UMA" : "upload") << ".";
  return common::Status::OK();
}

void D3D12AcceleratedExternalDataLoader::EndLoad()
    const noexcept {
  load_end_requested_.store(
      true, std::memory_order_relaxed);

  ORT_TRY {
    if (accelerated_load_batch_) {
      for (auto& tensor :
           accelerated_load_batch_->tensors) {
        if (!tensor.claimed &&
            tensor.access_started) {
          ImportedAllocation allocation;
          allocation.resource =
              std::move(tensor.resource);
          allocation.heap_owner =
              std::move(tensor.heap_owner);
          allocation.memory =
              std::move(tensor.memory);
          allocation.buffer =
              std::move(tensor.buffer);
          allocation.access_started = true;
          EndAccessNoThrow(allocation);
          tensor.access_started = false;
        }
      }
      accelerated_load_batch_.reset();
    }
  }
  ORT_CATCH(...) {
    // Abort is best-effort and is required not to throw.
  }
}

common::Status
D3D12AcceleratedExternalDataLoader::LoadTensor(
    const Env&,
    const std::filesystem::path& data_file_path,
    std::string_view tensor_name,
    FileOffsetType data_offset,
    SafeInt<size_t> data_length,
    const std::shared_ptr<IAllocator>& allocator,
    Tensor& tensor) const {
  ORT_RETURN_IF_NOT(
      accelerated_load_batch_ != nullptr &&
          accelerated_load_batch_->load_candidates_committed,
      "D3D12 accelerated initializer batch has not been committed.");
  ORT_RETURN_IF(
      data_offset < 0,
      "D3D12 accelerated initializer has a negative file offset.");
  ORT_RETURN_IF_NOT(
      allocator != nullptr,
      "D3D12 accelerated initializer requires its device allocator.");
  ORT_RETURN_IF_NOT(
      allocator.get() ==
          buffer_registry_->allocator,
      "D3D12 accelerated initializer was passed a different allocator.");

  const size_t length =
      static_cast<size_t>(data_length);
  const auto file_iterator =
      accelerated_load_batch_->files.find(
          data_file_path.lexically_normal());
  ORT_RETURN_IF(
      file_iterator == accelerated_load_batch_->files.end(),
      "No prepared D3D12 data file matches \"",
      data_file_path.string(), "\".");
  const TensorKey key = MakeKey(
      file_iterator->second.canonical_path,
      tensor_name,
      static_cast<uint64_t>(data_offset),
      length);
  const auto iterator =
      accelerated_load_batch_->tensors_by_key.find(key);
  ORT_RETURN_IF(
      iterator ==
          accelerated_load_batch_->tensors_by_key.end(),
      "No prepared D3D12 accelerated initializer matches \"",
      tensor_name,
      "\" and the requested file range.");

  auto& prepared =
      accelerated_load_batch_->tensors[iterator->second];
  ORT_RETURN_IF(
      prepared.claimed,
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" has already been consumed.");
  ORT_RETURN_IF(
      length != tensor.SizeInBytes(),
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" length does not match the placeholder tensor.");
  if (length == 0) {
    prepared.claimed = true;
    tensor = Tensor{
        tensor.DataType(), tensor.Shape(),
        nullptr, allocator};
    return common::Status::OK();
  }
  ORT_RETURN_IF_NOT(
      prepared.buffer &&
          prepared.access_started,
      "D3D12 accelerated initializer \"",
      tensor_name,
      "\" does not have an imported buffer.");

  auto imported =
      std::make_unique<ImportedAllocation>();
  imported->resource =
      std::move(prepared.resource);
  imported->heap_owner =
      std::move(prepared.heap_owner);
  imported->memory =
      std::move(prepared.memory);
  imported->buffer =
      std::move(prepared.buffer);
  imported->access_started =
      prepared.access_started;
  WGPUBuffer buffer = imported->buffer.Get();

  {
    std::lock_guard<std::mutex> lock{
        buffer_registry_->mutex};
    ORT_RETURN_IF(
        buffer_registry_->imported_allocations
                .find(buffer) !=
            buffer_registry_->imported_allocations
                .end(),
        "D3D12 accelerated imported buffer was registered twice.");
    buffer_registry_->imported_allocations.emplace(
        buffer, std::move(imported));
  }

  prepared.access_started = false;
  prepared.claimed = true;
  tensor = Tensor{
      tensor.DataType(), tensor.Shape(),
      buffer, allocator};
  return common::Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
