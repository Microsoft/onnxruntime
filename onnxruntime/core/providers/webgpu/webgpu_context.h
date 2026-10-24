// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/common/common.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/webgpu_utils.h"

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
#include <d3d12.h>
#include <wrl/client.h>
#endif

namespace onnxruntime {
class Tensor;

namespace webgpu {

class WebGpuContext;
class WebGpuInstanceWrapper;
class ComputeContextBase;
class ComputeContext;
class ProgramBase;
// PendingKernelInfo stores profiling information for a kernel execution
struct PendingKernelInfo {
  PendingKernelInfo(std::string_view kernel_name,
                    std::string_view kernel_type,
                    std::string_view program_name,
                    std::string_view cache_key,
                    const std::vector<ProgramInput>& inputs,
                    const std::vector<ProgramOutput>& outputs)
      : name{absl::StrJoin({kernel_name, kernel_type, program_name}, "&")}, cache_key{cache_key} {
    // Store shape information instead of tensor pointers to avoid accessing released tensors
    input_shapes.reserve(inputs.size());
    for (const auto& input : inputs) {
      input_shapes.emplace_back(input.use_override_shape ? input.override_shape : input.tensor->Shape());
    }
    output_shapes.reserve(outputs.size());
    for (const auto& output : outputs) {
      output_shapes.emplace_back(output.use_override_shape ? output.override_shape : output.tensor->Shape());
    }
  }

  PendingKernelInfo(const PendingKernelInfo&) = default;
  PendingKernelInfo& operator=(const PendingKernelInfo&) = default;
  PendingKernelInfo(PendingKernelInfo&&) = default;
  PendingKernelInfo& operator=(PendingKernelInfo&&) = default;

  std::string name;
  std::string cache_key;
  std::vector<TensorShape> input_shapes;
  std::vector<TensorShape> output_shapes;
};

// State for one in-flight pipeline build. The compiled pipeline is written into
// `callback_context->pipeline` by the async callback, which shares ownership of the result state
// independently of this build's lifetime.
struct PendingPipelineBuild {
  std::string name;
  std::vector<int> shape_uniform_ranks;
  wgpu::BindGroupLayout bind_group_layout;
  std::shared_ptr<PipelineCallbackContext> callback_context;
  wgpu::Future future;
};

// Resources for one recorded dispatch. The pipeline fields represent these lifecycle states:
//
//                                 pending_build    compute_pipeline
// 1. Program cache hit                 empty              set
//    The pipeline was already available when the dispatch was recorded.
//
// 2. First cache miss for a key         set              empty
//    This command owns the asynchronous build for its program key.
//
// 3. Later cache miss for the same key  empty            empty
//    An earlier command in deferred_dispatches_ owns the build. This command keeps program_key so
//    pipeline resolution can obtain the completed pipeline from the program cache.
//
// 4. Deferred pipeline resolved         empty             set
//    WaitForDeferredPipelineBuilds() has waited for the owning build as needed, populated the
//    program cache, and assigned a ready pipeline to every command before encoding starts.
//
// 5. Captured command / graph replay    empty             set
//    The command was moved into captured graph storage after encoding and retains the ready
//    pipeline and bind group for later replays. Replay never depends on pending_build.
//
// A command with both fields empty is therefore valid only in state 3 before deferred pipeline
// resolution. DispatchCommand() requires compute_pipeline to be set.
struct DispatchCommandInfo {
  std::string program_key;
  std::optional<PendingPipelineBuild> pending_build;
  std::optional<wgpu::ComputePipeline> compute_pipeline;
  wgpu::BindGroup bind_group;
  std::array<uint32_t, 3> dispatch_group{1, 1, 1};
  // WGPUBuffer for indirect dispatch, nullptr if not using indirect dispatch
  WGPUBuffer indirect_buffer = nullptr;
  // Optional profiling data
  std::optional<PendingKernelInfo> pending_kernel_info;
};

struct CapturedBufferCopy {
  wgpu::Buffer source;
  wgpu::Buffer destination;
  uint64_t size;
};

struct CapturedBufferClear {
  wgpu::Buffer buffer;
  uint64_t size;
};

// Copies and clears retain their buffers and share dispatch ordering, but do not consume timestamp query slots.
struct CapturedCommand {
  std::variant<DispatchCommandInfo, CapturedBufferCopy, CapturedBufferClear> operation;
};

// A command recording timeline has one caller at a time. ORT serializes Run on a Session;
// plugin streamless operations use independent encoders. BufferManager synchronizes buffer
// releases from concurrent allocator callers separately from command recording.
struct CommandRecordingState {
  CommandRecordingState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CommandRecordingState);

  wgpu::CommandEncoder command_encoder;
  wgpu::ComputePassEncoder compute_pass_encoder;
  uint32_t num_pending_dispatches = 0;
  // Concurrent allocator frees inspect only this flag, never the encoders or deferred dispatches.
  // BufferManager clears it under its cache lock after submission or abandonment.
  std::atomic<bool> has_unsubmitted_work{false};
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  std::mutex pending_release_callbacks_mutex;
  std::vector<std::function<void()>> pending_release_callbacks;
#endif
  std::vector<DispatchCommandInfo> deferred_dispatches;
  std::vector<PendingKernelInfo> pending_kernels;
  GraphCaptureState graph_capture_state{GraphCaptureState::Default};
  std::vector<CapturedCommand>* external_captured_commands = nullptr;
};

struct WebGpuBufferCacheConfig {
  struct ConfigEntry {
    BufferCacheMode mode;
    std::string config_string;  // preserved for customized configuration, eg. bucket sizes
  };
  ConfigEntry storage{BufferCacheMode::Bucket, {}};
  ConfigEntry uniform{BufferCacheMode::Simple, {}};
  ConfigEntry query_resolve{BufferCacheMode::Disabled, {}};
  ConfigEntry default_entry{BufferCacheMode::Disabled, {}};
};

/// <summary>
/// Represents the configuration options for creating a WebGpuContext.
/// </summary>
struct WebGpuContextConfig {
  int context_id{0};
  WGPUInstance instance{nullptr};
  WGPUDevice device{nullptr};
  const void* dawn_proc_table{nullptr};
  ValidationMode validation_mode{
#ifndef NDEBUG
      webgpu::ValidationMode::Full  // for debug build, enable full validation by default
#else
      webgpu::ValidationMode::Basic  // for release build, enable basic validation by default
#endif  // !NDEBUG
  };
  bool validation_mode_explicitly_set{false};
  bool enable_robustness{
#ifndef NDEBUG
      true  // for debug builds, enable robust buffer access by default
#else
      false  // for release builds, disable robust buffer access for performance by default
#endif
  };
  bool enable_robustness_explicitly_set{false};
  bool disable_subgroup_matrix{false};
  bool disable_subgroup_matrix_explicitly_set{false};
  bool preserve_device{false};
  WeightLoadAccelerationMode weight_load_acceleration_mode{
      WeightLoadAccelerationMode::Off};
  // When true, skip Dawn adapter/device creation and all device-dependent initialization; the context
  // can only be used for graph transformation, not execution. Derived from kOrtSessionOptionCompileOnly.
  bool compile_only{false};
  uint32_t max_num_pending_dispatches{16};
  uint64_t max_storage_buffer_binding_size{0};
  // Internal test hook. Provider-option parsing never populates this field.
  uint64_t test_only_max_storage_buffer_binding_size{0};
  uint32_t max_storage_buffers_per_shader_stage{0};
  WebGpuBufferCacheConfig buffer_cache_config{};
  std::optional<uint32_t> adapter_index;
  int power_preference{static_cast<int>(WGPUPowerPreference_HighPerformance)};
  int backend_type{
#ifdef _WIN32
  // Setup Windows default backend type based on the build configuration
#if defined(DAWN_ENABLE_D3D12)
      static_cast<int>(WGPUBackendType_D3D12)
#elif defined(DAWN_ENABLE_VULKAN)
      static_cast<int>(WGPUBackendType_Vulkan)
#else
      0
#endif
#else
      0
#endif
  };
};

class WebGpuContextFactory {
 public:
  struct WebGpuContextInfo {
    std::unique_ptr<WebGpuContext> context;
    int ref_count;
  };

  /// <summary>
  /// Create a new WebGPU context for the specified context ID if not present, or return the existing one. (ref-count based)
  /// </summary>
  static WebGpuContext& CreateContext(const WebGpuContextConfig& config);

  /// <summary>
  /// Get the WebGPU context for the specified context ID. Throw if not present.
  /// </summary>
  static WebGpuContext& GetContext(int context_id);

  /// <summary>
  /// Retain an existing WebGPU context. (ref-count based)
  /// </summary>
  static void RetainContext(int context_id);

  /// <summary>
  /// Release the WebGPU context. (ref-count based)
  /// </summary>
  static void ReleaseContext(int context_id);

  static void Cleanup();

  /// <summary>
  /// Return the default context. Create if not present.
  /// </summary>
  static WebGpuContext& DefaultContext();

 private:
  WebGpuContextFactory() {}

  static std::mutex mutex_;
  static std::once_flag init_default_flag_;

  // Use pointers to heap-allocated objects so that their destructors do NOT run
  // during static destruction at process exit. This avoids crashes when dependent
  // DLLs (e.g. dxcompiler.dll) have already been unloaded by the OS.
  // Cleanup() explicitly deletes them during normal unload. In the shared-library
  // build this is reached via ReleaseEpFactory, and in the WebGPU static-lib build
  // it is reached from OrtEnv::~OrtEnv via CleanupWebGpuContexts().
  // On abnormal/process termination they simply leak, which is safe.
  static std::unordered_map<int32_t, WebGpuContextInfo>* contexts_;
  static WebGpuInstanceWrapper* default_instance_;
};

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext final {
 public:
  static constexpr uint64_t kMinConfigurableStorageBufferBindingSize = 256;
  // WebGPU's guaranteed maxStorageBufferBindingSize limit is 128 MiB.
  // See https://gpuweb.github.io/gpuweb/?utm_source=openai
  static constexpr uint64_t kWebGpuGuaranteedMaxStorageBufferBindingSize =
      128ULL * 1024 * 1024;

  ~WebGpuContext();

  Status Wait(wgpu::Future f);

  const wgpu::Instance& Instance() const { return instance_; }
  const wgpu::Device& Device() const { return device_; }

  const wgpu::AdapterInfo& AdapterInfo() const { return adapter_info_; }
  const wgpu::Limits& DeviceLimits() const { return device_limits_; }
  bool DeviceHasFeature(wgpu::FeatureName feature) const { return device_features_.contains(feature); }
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const { return subgroup_matrix_configs_; }

  const wgpu::CommandEncoder& GetCommandEncoder(CommandRecordingState& recording) {
    if (!recording.command_encoder) {
      recording.command_encoder = device_.CreateCommandEncoder();
      recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
    }
    return recording.command_encoder;
  }

  const wgpu::ComputePassEncoder& GetComputePassEncoder(CommandRecordingState& recording) {
    if (!recording.compute_pass_encoder) {
      auto& command_encoder = GetCommandEncoder(recording);

      wgpu::ComputePassDescriptor compute_pass_desc{};

      if (is_profiling_ && query_type_ == TimestampQueryType::AtPasses &&
          recording.graph_capture_state != GraphCaptureState::Capturing) {
        wgpu::PassTimestampWrites timestampWrites = {
            nullptr,
            query_set_,
            recording.num_pending_dispatches * 2,
            recording.num_pending_dispatches * 2 + 1};
        compute_pass_desc.timestampWrites = &timestampWrites;
      }

      recording.compute_pass_encoder = command_encoder.BeginComputePass(&compute_pass_desc);
    }
    return recording.compute_pass_encoder;
  }

  void EndComputePass(CommandRecordingState& recording) {
    if (recording.compute_pass_encoder) {
      recording.compute_pass_encoder.End();
      recording.compute_pass_encoder = nullptr;
    }
  }
  void CopyBuffer(WGPUBuffer source, WGPUBuffer destination, uint64_t size, CommandRecordingState& recording);
  void ClearBuffer(WGPUBuffer buffer, uint64_t size, CommandRecordingState& recording);
  void CaptureBegin(std::vector<webgpu::CapturedCommand>* captured_commands,
                    const webgpu::BufferManager& buffer_manager,
                    CommandRecordingState& recording);
  void CaptureEnd(CommandRecordingState& recording);
  void Replay(const std::vector<webgpu::CapturedCommand>& captured_commands,
              const webgpu::BufferManager& buffer_manager,
              CommandRecordingState& recording);
  void ReleaseGraphResources(std::vector<webgpu::CapturedCommand>& captured_commands);

  Status Flush(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);

  // Unlike Flush, waits for submitted queue work and checks scoped errors and device loss.
  // Serialize with other operations on this recording. Scopes cover flushing, not earlier encoding.
  Status FlushAndWaitChecked(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);

  // Scopes cover enqueue (which may run multiple programs), flushing and completion.
  // Drain and resolve all scopes even if enqueue fails or throws. Do not switch threads in enqueue.
  Status RunAndWaitChecked(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording,
                           const std::function<Status()>& enqueue);

  // Context-level managers are shared by sessions and synchronize their buffer caches internally.
  webgpu::BufferManager& BufferManager() const { return *buffer_mgr_; }
  webgpu::BufferManager& InitializerBufferManager() const { return *initializer_buffer_mgr_; }

  CommandRecordingState* ActiveSerializedRecording() const { return serialized_active_recording_; }
  void SetActiveSerializedRecording(CommandRecordingState* recording) { serialized_active_recording_ = recording; }

  inline webgpu::ValidationMode ValidationMode() const {
    return validation_mode_;
  }

  // False for a device-free ("virtual device") context, which has no Dawn device and can only run graph
  // transformation. Used to hand out a no-op allocator instead of a real GpuBufferAllocator.
  inline bool HasDevice() const { return device_ != nullptr; }
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  ID3D12Device* WeightLoadingD3D12Device() const;
  bool D3D12SharedResourceFeaturesAvailable() const {
    return d3d12_shared_resource_features_available_;
  }
  wgpu::BackendType SelectedBackendType() const {
    return selected_backend_type_;
  }
#endif

  //
  // Get Split-K configuration.
  //
  const SplitKConfig& GetSplitKConfig() const {
    return *split_k_config_;
  }

  // Set the CPU time base (ORT profiler's profiling_start_time) used to align GPU
  // timestamps with ORT CPU events. Pushed in by WebGpuProfiler::StartProfiling, which
  // is the only place that receives the framework's profiling start time for both
  // session-level and run-level profiling.
  void SetProfilingStartTime(TimePoint profiling_start_time) { profiling_start_time_ = profiling_start_time; }
  void StartProfiling();
  // Collect pending GPU profiling data into the given events vector.
  void CollectProfilingData(profiling::Events& events);
  // Collect pending GPU profiling data into the shared events_ vector (run-level).
  void CollectProfilingData();
  void EndProfiling(TimePoint, profiling::Events& events);

  //
  // Push error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  void PushErrorScope(wgpu::ErrorFilter filter = wgpu::ErrorFilter::Validation);

  //
  // Pop error scope.
  //
  // This is useful only when "skip_validation" is not set.
  //
  Status PopErrorScope();

  Status Run(ComputeContextBase& context, const ProgramBase& program);

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  std::unique_ptr<WebGpuPIXFrameGenerator> CreatePIXFrameGenerator() {
    return std::make_unique<WebGpuPIXFrameGenerator>(instance_,
                                                     Device());
  }
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

 private:
  enum class TimestampQueryType {
    None = 0,
    InsidePasses,
    AtPasses
  };

  static uint64_t ResolveMaxStorageBufferBindingSize(
      uint64_t max_storage_buffer_binding_size,
      uint64_t test_only_max_storage_buffer_binding_size) {
    ORT_ENFORCE(max_storage_buffer_binding_size == 0 ||
                    max_storage_buffer_binding_size >= kWebGpuGuaranteedMaxStorageBufferBindingSize,
                "max_storage_buffer_binding_size must be 0 or at least 128 MiB");
    ORT_ENFORCE(test_only_max_storage_buffer_binding_size == 0 ||
                    test_only_max_storage_buffer_binding_size >= kMinConfigurableStorageBufferBindingSize,
                "test_only_max_storage_buffer_binding_size must be 0 or at least 256 bytes");
    return test_only_max_storage_buffer_binding_size != 0
               ? test_only_max_storage_buffer_binding_size
               : max_storage_buffer_binding_size;
  }

  WebGpuContext(WGPUInstance instance,
                WGPUDevice device,
                webgpu::ValidationMode validation_mode,
                bool validation_mode_explicitly_set,
                bool preserve_device,
                uint64_t max_storage_buffer_binding_size,
                uint64_t test_only_max_storage_buffer_binding_size,
                uint32_t max_storage_buffers_per_shader_stage)
      : instance_{instance},
        device_{device},
        validation_mode_{validation_mode},
        validation_mode_explicitly_set_{validation_mode_explicitly_set},
        query_type_{TimestampQueryType::None},
        preserve_device_{preserve_device},
        max_storage_buffer_binding_size_{ResolveMaxStorageBufferBindingSize(
            max_storage_buffer_binding_size, test_only_max_storage_buffer_binding_size)},
        max_storage_buffers_per_shader_stage_{max_storage_buffers_per_shader_stage} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);

  void Initialize(const WebGpuContextConfig& config);

  wgpu::BindGroup CreateBindGroup(const std::vector<WGPUBuffer>& bind_buffers,
                                  const std::vector<uint32_t>& bind_buffers_segments,
                                  const wgpu::BindGroupLayout& bind_group_layout,
                                  std::string_view label) const;
  void DispatchCommand(const webgpu::DispatchCommandInfo& command, CommandRecordingState& recording);
  Status EncodeDeferredDispatches(CommandRecordingState& recording);

  std::vector<const char*> GetEnabledAdapterToggles() const;
  std::vector<const char*> GetEnabledDeviceToggles() const;
  std::vector<const char*> GetDisabledDeviceToggles() const;
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;
  wgpu::Limits GetRequiredLimits(const wgpu::Adapter& adapter) const;
  void WriteTimestamp(CommandRecordingState& recording, uint32_t query_index);

  struct PendingQueryInfo {
    PendingQueryInfo(std::vector<PendingKernelInfo>&& kernels, wgpu::Buffer query_buffer)
        : kernels{std::move(kernels)}, query_buffer{query_buffer} {}

    PendingQueryInfo(PendingQueryInfo&&) = default;
    PendingQueryInfo& operator=(PendingQueryInfo&&) = default;
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(PendingQueryInfo);

    std::vector<PendingKernelInfo> kernels;
    wgpu::Buffer query_buffer;
  };

  // Find the build owner for a cache key in the current deferred window.
  PendingPipelineBuild* FindPendingPipelineBuild(CommandRecordingState& recording, std::string_view key);
  Status WaitForDeferredPipelineBuilds(CommandRecordingState& recording);
  Status CheckDeviceStatus() const;

  friend class BufferManager;
  friend class ComputeContext;
  friend class WebGpuContextFactory;

  std::once_flag init_flag_;

  struct DeviceErrorState {
    std::mutex mutex;
    Status status;
  };
  std::shared_ptr<DeviceErrorState> device_error_state_{std::make_shared<DeviceErrorState>()};
  wgpu::Instance instance_;
  wgpu::Device device_;

  webgpu::ValidationMode validation_mode_;
  bool validation_mode_explicitly_set_;
  bool enable_robustness_ = false;
  bool disable_subgroup_matrix_ = false;
  std::optional<uint32_t> adapter_index_;
  int adapter_power_preference_ = static_cast<int>(WGPUPowerPreference_HighPerformance);
  int adapter_backend_type_ = 0;

  wgpu::Queue device_queue_;
  wgpu::AdapterInfo adapter_info_;
  wgpu::Limits device_limits_;
  std::unordered_set<wgpu::FeatureName> device_features_;
  wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs_;

  std::unique_ptr<webgpu::BufferManager> buffer_mgr_;
  std::unique_ptr<webgpu::BufferManager> initializer_buffer_mgr_;
  std::unique_ptr<ProgramManager> program_mgr_;

  // Old plugin hosts cannot reliably associate framework copies with a Session.
  // The active recording is borrowed from the running Session, never owned by the context.
  // All operations on this device must be serialized in compatibility mode, including across Sessions.
  CommandRecordingState* serialized_active_recording_ = nullptr;

  uint32_t max_num_pending_dispatches_ = 16;

  std::unique_ptr<SplitKConfig> split_k_config_;

  // profiling
  TimestampQueryType query_type_;
  wgpu::QuerySet query_set_;
  wgpu::Buffer query_resolve_buffer_;

  // info of queries pending appending to profiling events
  std::vector<PendingQueryInfo> pending_queries_;

  uint64_t gpu_timestamp_offset_ = 0;
  // ORT profiler's CPU time base, set via SetProfilingStartTime; GPU timestamps are
  // aligned to it. See SetProfilingStartTime.
  TimePoint profiling_start_time_;
  // CPU elapsed time (us) from profiling_start_time_ to the first GPU submit, applied
  // as an offset to GPU timestamps in CollectProfilingData. -1 until the first submit.
  int64_t profiling_first_submit_cpu_offset_us_ = -1;
  bool is_profiling_ = false;
  // Shared GPU profiling events for run-level profiling.
  profiling::Events events_;
  bool preserve_device_;
  WeightLoadAccelerationMode weight_load_acceleration_mode_{
      WeightLoadAccelerationMode::Off};
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  Microsoft::WRL::ComPtr<ID3D12Device> weight_loading_d3d12_device_;
  bool d3d12_shared_resource_features_available_ = false;
  wgpu::BackendType selected_backend_type_ = wgpu::BackendType::Undefined;
#endif
  uint64_t max_storage_buffer_binding_size_;
  uint32_t max_storage_buffers_per_shader_stage_;
  GraphCaptureState graph_capture_state_{GraphCaptureState::Default};

  // External vector to store captured commands, owned by EP
  std::vector<webgpu::CapturedCommand>* external_captured_commands_ = nullptr;
};

#if !defined(__wasm__)
namespace detail {

// Returns true when the largest host visible device local heap is at least as
// large as the largest device local heap. Without Resizable BAR a discrete GPU
// exposes only a small PCIe window of such memory where larger initializers
// fail to allocate so mapped initializer upload is not used there. It is kept
// separate from adapter feature discovery for heap layout testing.
bool CanMapDeviceLocalMemory(gsl::span<const wgpu::MemoryHeapInfo> heaps);

}  // namespace detail
#endif  // !defined(__wasm__)

}  // namespace webgpu
}  // namespace onnxruntime
