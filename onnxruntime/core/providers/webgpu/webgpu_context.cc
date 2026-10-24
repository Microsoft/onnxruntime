// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gsl/util>

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "core/providers/webgpu/d3d12_external_data_loader.h"
#if !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/D3D12Backend.h"
#endif
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#if !defined(__wasm__)
#if !defined(BUILD_DAWN_SHARED_LIBRARY)
#include "dawn/dawn_proc.h"
#endif
#if !defined(USE_EXTERNAL_DAWN)
#include "core/providers/webgpu/webgpu_context_dawn_platform.h"
#endif
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_instance_wrapper.h"
#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/string_macros.h"

namespace onnxruntime {
namespace webgpu {

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
namespace {

struct RequestAdapterOptionsLuid : wgpu::ChainedStruct {
  RequestAdapterOptionsLuid() {
    sType = static_cast<wgpu::SType>(WGPUSType_RequestAdapterOptionsLUID);
  }

  LUID adapter_luid{};
};

common::Status SelectWeightLoadingD3D12Device(
    wgpu::PowerPreference power_preference,
    LUID& selected_adapter_luid,
    Microsoft::WRL::ComPtr<ID3D12Device>& selected_device) {
  Microsoft::WRL::ComPtr<IDXGIFactory6> dxgi_factory;
  HRESULT result =
      CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory));
  ORT_RETURN_IF_ERROR(
      FAILED(result)
          ? ORT_MAKE_STATUS(
                ONNXRUNTIME, FAIL,
                "Accelerated weight loading failed to create the DXGI factory "
                "with HRESULT 0x",
                std::hex, static_cast<uint32_t>(result), ".")
          : common::Status::OK());

  DXGI_GPU_PREFERENCE gpu_preference =
      DXGI_GPU_PREFERENCE_UNSPECIFIED;
  if (power_preference == wgpu::PowerPreference::HighPerformance) {
    gpu_preference = DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
  } else if (power_preference == wgpu::PowerPreference::LowPower) {
    gpu_preference = DXGI_GPU_PREFERENCE_MINIMUM_POWER;
  }

  for (uint32_t adapter_index = 0;; ++adapter_index) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> dxgi_adapter;
    result = dxgi_factory->EnumAdapterByGpuPreference(
        adapter_index, gpu_preference,
        IID_PPV_ARGS(&dxgi_adapter));
    if (result == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    ORT_RETURN_IF(
        FAILED(result),
        "Accelerated weight loading failed to enumerate DXGI adapters with "
        "HRESULT 0x",
        std::hex, static_cast<uint32_t>(result), ".");

    DXGI_ADAPTER_DESC1 adapter_description{};
    result = dxgi_adapter->GetDesc1(&adapter_description);
    if (FAILED(result) ||
        (adapter_description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
      continue;
    }

    Microsoft::WRL::ComPtr<ID3D12Device> candidate_device;
    result = D3D12CreateDevice(
        dxgi_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&candidate_device));
    if (SUCCEEDED(result)) {
      selected_adapter_luid = adapter_description.AdapterLuid;
      selected_device = std::move(candidate_device);
      return common::Status::OK();
    }
  }
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL,
      "Accelerated weight loading did not find a hardware D3D12 adapter.");
}

void ConfigureWeightLoadingAdapterOptions(
    const WebGpuContextConfig& config,
    WeightLoadAccelerationMode weight_load_acceleration_mode,
    wgpu::RequestAdapterOptions& adapter_options,
    RequestAdapterOptionsLuid& luid_options,
    Microsoft::WRL::ComPtr<ID3D12Device>& weight_loading_d3d12_device) {
  // Retain the selected adapter's native device even when the first session
  // disables acceleration. Later sessions reuse this context and cannot
  // repeat adapter selection.
  const bool preselect_weight_loading_adapter =
      static_cast<wgpu::BackendType>(config.backend_type) ==
          wgpu::BackendType::D3D12 &&
      !config.adapter_index;
  if (!preselect_weight_loading_adapter) {
    return;
  }

  LUID selected_adapter_luid{};
  const auto selection_start = std::chrono::steady_clock::now();
  const auto selection_status = SelectWeightLoadingD3D12Device(
      adapter_options.powerPreference, selected_adapter_luid,
      weight_loading_d3d12_device);

  if (!selection_status.IsOK()) {
    if (IsWeightLoadAccelerationRequired(weight_load_acceleration_mode)) {
      ORT_THROW(selection_status.ErrorMessage());
    }
    LOGS_DEFAULT(WARNING)
        << selection_status.ErrorMessage()
        << " Continuing with ordinary Dawn adapter selection without "
           "accelerated weight loading.";
    weight_loading_d3d12_device.Reset();
  }
  if (weight_loading_d3d12_device) {
    luid_options.adapter_luid = selected_adapter_luid;
    luid_options.nextInChain = adapter_options.nextInChain;
    adapter_options.nextInChain = &luid_options;

    LOGS_DEFAULT(INFO)
        << "WebGPU D3D12 weight-loading GPU selection: "
        << std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - selection_start)
               .count()
        << " ms.";
  }
}

}  // namespace
#endif

WebGpuContext::~WebGpuContext() = default;

void WebGpuContext::Initialize(const WebGpuContextConfig& config) {
  std::call_once(init_flag_, [this, config]() {
    max_num_pending_dispatches_ = config.max_num_pending_dispatches;
    enable_robustness_ = config.enable_robustness;
    adapter_index_ = config.adapter_index;
    adapter_power_preference_ = config.power_preference;
    adapter_backend_type_ = config.backend_type;

    ORT_ENFORCE(!config.adapter_index || config.device == nullptr,
                "adapterIndex cannot be used with an externally supplied WebGPU device.");
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
    ORT_ENFORCE(!config.adapter_index,
                "adapterIndex requires a native Dawn build with adapter enumeration support.");
#endif
    weight_load_acceleration_mode_ = config.weight_load_acceleration_mode;

    // Three easily-conflated concepts, at three layers (a pipeline, not the same flag):
    //   * allow_virtual_devices (env)     -- selectability: surface a virtual GPU OrtEpDevice so WebGPU is
    //                                        pickable when OS enumeration finds no GPU (e.g. Win32k sandbox).
    //   * compile_only (session)          -- intent: transform only, never finalize/run.
    //   * device-free / HasDevice() (ctx) -- mechanism: no Dawn device, no-op allocator.
    // compile_only alone is valid (device-free even with a real GPU); a virtual device without compile_only is
    // rejected at factory CreateEp -- it would try to build a real Dawn device with no hardware.
    if (config.compile_only) {
      LOGS_DEFAULT(INFO) << "WebGPU EP context created device-free (compile-only session, no Dawn device).";
      return;
    }

    if (device_ == nullptr) {
      // Create wgpu::Adapter
      wgpu::RequestAdapterOptions req_adapter_options = {};
      req_adapter_options.backendType = static_cast<wgpu::BackendType>(config.backend_type);
      req_adapter_options.powerPreference = static_cast<wgpu::PowerPreference>(config.power_preference);

#if !defined(__wasm__)
      auto enabled_adapter_toggles = GetEnabledAdapterToggles();

      wgpu::DawnTogglesDescriptor adapter_toggles_desc = {};
      adapter_toggles_desc.enabledToggleCount = enabled_adapter_toggles.size();
      adapter_toggles_desc.enabledToggles = enabled_adapter_toggles.data();

      req_adapter_options.nextInChain = &adapter_toggles_desc;
#endif

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
      const auto* ordinary_adapter_options_chain =
          req_adapter_options.nextInChain;
      RequestAdapterOptionsLuid luid_options{};
      ConfigureWeightLoadingAdapterOptions(
          config, weight_load_acceleration_mode_, req_adapter_options,
          luid_options, weight_loading_d3d12_device_);
#endif

      wgpu::Adapter adapter;
      if (config.adapter_index) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
        const auto adapters = EnumerateBundledDawnAdapters(instance_.Get(), req_adapter_options);
        ORT_ENFORCE(*config.adapter_index < adapters.size(),
                    "WebGPU adapterIndex ", *config.adapter_index,
                    " is out of range; Dawn enumerated ", adapters.size(),
                    " adapter(s) for the requested backend and power-preference hint.");
        adapter = adapters[*config.adapter_index];
        LOGS_DEFAULT(INFO) << "WebGPU EP selected physical adapter index " << *config.adapter_index
                           << " of " << adapters.size()
                           << " adapter(s) for the requested backend and power-preference hint.";
#endif
      } else {
        // Capture adapter request result without throwing inside the Dawn callback.
        // Throwing C++ exceptions inside Dawn callbacks leaves Dawn's internal mutexes locked,
        // which causes a self-deadlock when the WGPUInstance is later released (e.g., during
        // OrtEnv teardown via EventManager::ShutDown()).
        struct RequestAdapterResult {
          wgpu::RequestAdapterStatus status = wgpu::RequestAdapterStatus::Error;
          wgpu::Adapter adapter;
          std::string message;
        };
        const auto request_adapter = [&]() {
          RequestAdapterResult result;
          ORT_ENFORCE(wgpu::WaitStatus::Success ==
                      instance_.WaitAny(instance_.RequestAdapter(
                                            &req_adapter_options,
                                            wgpu::CallbackMode::WaitAnyOnly,
                                            [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                                               wgpu::StringView message,
                                               RequestAdapterResult* result) noexcept {
                                              result->status = status;
                                              if (status == wgpu::RequestAdapterStatus::Success) {
                                                result->adapter = std::move(adapter);
                                              } else {
                                                result->message = std::string{message};
                                              }
                                            },
                                            &result),
                                        UINT64_MAX));
          return result;
        };
        RequestAdapterResult adapter_result = request_adapter();
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
        if (adapter_result.status != wgpu::RequestAdapterStatus::Success &&
            weight_loading_d3d12_device_ &&
            !IsWeightLoadAccelerationRequired(
                weight_load_acceleration_mode_)) {
          LOGS_DEFAULT(WARNING)
              << "Dawn rejected the adapter selected for accelerated weight "
                 "loading: "
              << adapter_result.message
              << " Retrying ordinary Dawn adapter selection without "
                 "accelerated weight loading.";
          weight_loading_d3d12_device_.Reset();
          req_adapter_options.nextInChain = ordinary_adapter_options_chain;
          adapter_result = request_adapter();
        }
#endif
        ORT_ENFORCE(adapter_result.status == wgpu::RequestAdapterStatus::Success,
                    "Failed to get a WebGPU adapter: ", adapter_result.message);
        adapter = std::move(adapter_result.adapter);
      }
      ORT_ENFORCE(adapter != nullptr, "Failed to get a WebGPU adapter.");

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
      d3d12_shared_resource_features_available_ =
          adapter.HasFeature(wgpu::FeatureName::SharedBufferMemoryD3D12Resource) &&
          adapter.HasFeature(wgpu::FeatureName::SharedFenceDXGISharedHandle);
#endif
      // Create wgpu::Device
      wgpu::DeviceDescriptor device_desc = {};

#if !defined(__wasm__)
      wgpu::DawnTogglesDescriptor device_toggles_desc = {};
      device_desc.nextInChain = &device_toggles_desc;

      auto enabled_device_toggles = GetEnabledDeviceToggles();
      device_toggles_desc.enabledToggleCount = enabled_device_toggles.size();
      device_toggles_desc.enabledToggles = enabled_device_toggles.data();

      auto disabled_device_toggles = GetDisabledDeviceToggles();
      device_toggles_desc.disabledToggleCount = disabled_device_toggles.size();
      device_toggles_desc.disabledToggles = disabled_device_toggles.data();
#endif

      std::vector<wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(adapter);
      if (!required_features.empty()) {
        device_desc.requiredFeatures = required_features.data();
        device_desc.requiredFeatureCount = required_features.size();
      }
      wgpu::Limits required_limits = GetRequiredLimits(adapter);
      device_desc.requiredLimits = &required_limits;

      device_desc.SetUncapturedErrorCallback(
          // Note: Don't throw from a Dawn callback.
          [](const wgpu::Device& /*device*/, wgpu::ErrorType type,
             wgpu::StringView message, DeviceErrorState* state) noexcept {
            {
              std::lock_guard<std::mutex> lock{state->mutex};
              if (state->status.IsOK()) {
                state->status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU device error(", int(type),
                                                "): ", std::string_view{message});
              }
            }
            if (logging::LoggingManager::HasDefaultLogger()) {
              LOGS_DEFAULT(ERROR) << "WebGPU device error(" << int(type) << "): " << std::string_view{message};
            }
          },
          device_error_state_.get());
      // Dawn stops uncaptured callbacks before invoking the lost callback, which retains their state.
      device_desc.SetDeviceLostCallback(
          wgpu::CallbackMode::AllowSpontaneous,
          // Note: Don't throw from a Dawn callback.
          [state = device_error_state_](const wgpu::Device& /*device*/, wgpu::DeviceLostReason reason,
                                        wgpu::StringView message) noexcept {
            {
              std::lock_guard<std::mutex> lock{state->mutex};
              if (state->status.IsOK()) {
                state->status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU device lost (", int(reason),
                                                "): ", std::string_view{message});
              }
            }
            if (logging::LoggingManager::HasDefaultLogger()) {
              LOGS_DEFAULT(INFO) << "WebGPU device lost (" << int(reason) << "): " << std::string_view{message};
            }
          });

      struct RequestDeviceResult {
        wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
        wgpu::Device device;
        std::string message;
      };
      RequestDeviceResult device_result;
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter.RequestDevice(
                                                                     &device_desc,
                                                                     wgpu::CallbackMode::WaitAnyOnly,
                                                                     // Note: Don't throw from a Dawn callback.
                                                                     [](wgpu::RequestDeviceStatus status,
                                                                        wgpu::Device device,
                                                                        wgpu::StringView message,
                                                                        RequestDeviceResult* result) noexcept {
                                                                       result->status = status;
                                                                       if (status == wgpu::RequestDeviceStatus::Success) {
                                                                         result->device = std::move(device);
                                                                       } else {
                                                                         result->message = std::string{message};
                                                                       }
                                                                     },
                                                                     &device_result),
                                                                 UINT64_MAX));
      ORT_ENFORCE(device_result.status == wgpu::RequestDeviceStatus::Success,
                  "Failed to get a WebGPU device: ", device_result.message);
      device_ = std::move(device_result.device);
      ORT_ENFORCE(device_ != nullptr, "Failed to get a WebGPU device.");
    }

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
    if (config.device != nullptr) {
      wgpu::SupportedFeatures supported_features;
      device_.GetFeatures(&supported_features);
      bool has_shared_buffer_memory = false;
      bool has_shared_fence = false;
      for (size_t i = 0; i < supported_features.featureCount; ++i) {
        has_shared_buffer_memory |=
            supported_features.features[i] ==
            wgpu::FeatureName::SharedBufferMemoryD3D12Resource;
        has_shared_fence |=
            supported_features.features[i] ==
            wgpu::FeatureName::SharedFenceDXGISharedHandle;
      }
      d3d12_shared_resource_features_available_ =
          has_shared_buffer_memory && has_shared_fence;
    }
#endif
    LOGS_DEFAULT(VERBOSE) << "WebGPU EP Context is created for: Instance=" << instance_.Get() << ", Device=" << device_.Get() << ".";

    // cache device queue
    device_queue_ = device_.GetQueue();
    // cache device limits
    ORT_ENFORCE(Device().GetLimits(&device_limits_) == wgpu::Status::Success);
    if (max_storage_buffer_binding_size_ != 0) {
      device_limits_.maxStorageBufferBindingSize =
          std::min(device_limits_.maxStorageBufferBindingSize, max_storage_buffer_binding_size_);
    }
    if (max_storage_buffers_per_shader_stage_ != 0) {
      ORT_ENFORCE(max_storage_buffers_per_shader_stage_ <= device_limits_.maxStorageBuffersPerShaderStage,
                  "maxStorageBuffersPerShaderStage exceeds the device limit");
      device_limits_.maxStorageBuffersPerShaderStage = max_storage_buffers_per_shader_stage_;
    }
    // Align maxStorageBufferBindingSize down to minStorageBufferOffsetAlignment so that
    // buffer segment offsets are always properly aligned for WebGPU bind group creation.
    if (device_limits_.minStorageBufferOffsetAlignment > 0) {
      device_limits_.maxStorageBufferBindingSize -=
          (device_limits_.maxStorageBufferBindingSize % device_limits_.minStorageBufferOffsetAlignment);
    }
    // cache device features
    wgpu::SupportedFeatures supported_features;
    Device().GetFeatures(&supported_features);
    for (size_t i = 0; i < supported_features.featureCount; i++) {
      device_features_.insert(supported_features.features[i]);
    }
    // cache adapter info
#if !defined(__wasm__)
    // Native sessions share device entry points outside their per-session command recordings.
    ORT_ENFORCE(DeviceHasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization),
                config.device != nullptr
                    ? "WebGPU: an externally supplied native device must enable ImplicitDeviceSynchronization "
                      "in DeviceDescriptor.requiredFeatures when it is created."
                    : "WebGPU: the internally created native device is missing the required "
                      "ImplicitDeviceSynchronization feature.");
#endif
    if (DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
      adapter_info_.nextInChain = &subgroup_matrix_configs_;
    }
    ORT_ENFORCE(Device().GetAdapterInfo(&adapter_info_) == wgpu::Status::Success);
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
    // Use the backend selected by Dawn rather than the requested backend. The
    // request can be Undefined when Dawn performs automatic adapter selection.
    selected_backend_type_ = adapter_info_.backendType;
#if !defined(USE_EXTERNAL_DAWN)
    if (selected_backend_type_ == wgpu::BackendType::D3D12 &&
        weight_loading_d3d12_device_ == nullptr) {
      weight_loading_d3d12_device_ =
          dawn::native::d3d12::GetD3D12Device(device_.Get());
    }
#endif
#endif

    // create buffer manager
    buffer_mgr_ = BufferManagerFactory::Create(*this,
                                               config.buffer_cache_config.storage.mode,
                                               config.buffer_cache_config.uniform.mode,
                                               config.buffer_cache_config.query_resolve.mode,
                                               config.buffer_cache_config.default_entry.mode);

    // create initializer buffer manager.
    initializer_buffer_mgr_ = BufferManagerFactory::Create(*this,
                                                           BufferCacheMode::LazyRelease,
                                                           BufferCacheMode::LazyRelease,
                                                           BufferCacheMode::Disabled,
                                                           BufferCacheMode::Disabled);

    // create program manager
    program_mgr_ = std::make_unique<ProgramManager>(*this);

    // create split-k config
    split_k_config_ = std::make_unique<SplitKConfig>(CreateSplitKConfig(adapter_info_));

    // set query type
#if !defined(__wasm__)
    if (DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses)) {
      query_type_ = TimestampQueryType::InsidePasses;
    } else
#endif
        if (DeviceHasFeature(wgpu::FeatureName::TimestampQuery)) {
      query_type_ = TimestampQueryType::AtPasses;
    } else {
      query_type_ = TimestampQueryType::None;
    }
  });

  if (max_num_pending_dispatches_ != config.max_num_pending_dispatches) {
    LOGS_DEFAULT(WARNING)
        << "WebGPU context is already initialized with "
        << "maxNumPendingDispatches="
        << max_num_pending_dispatches_
        << ". Requested value "
        << config.max_num_pending_dispatches
        << " will be ignored.";
  }

  if (config.adapter_index &&
      (config.adapter_index != adapter_index_ ||
       config.power_preference != adapter_power_preference_ ||
       config.backend_type != adapter_backend_type_)) {
    ORT_THROW("WebGPU context is already initialized with adapterIndex=",
              adapter_index_ ? std::to_string(*adapter_index_) : "automatic",
              ", powerPreference=", adapter_power_preference_,
              ", dawnBackendType=", adapter_backend_type_,
              ". Requested selector has adapterIndex=", *config.adapter_index,
              ", powerPreference=", config.power_preference,
              ", dawnBackendType=", config.backend_type,
              " and cannot be applied to the existing device. Use a separate process or an externally supplied custom context/device.");
  }

  if (config.enable_robustness_explicitly_set) {
    if (config.device != nullptr) {
      LOGS_DEFAULT(WARNING)
          << "WebGPU enableRobustness cannot affect an externally supplied WebGPU device. "
          << "The requested value will be ignored.";
    } else if (device_ != nullptr && enable_robustness_ != config.enable_robustness) {
      LOGS_DEFAULT(WARNING)
          << "WebGPU context is already initialized with enableRobustness=" << enable_robustness_
          << ". Requested value " << config.enable_robustness << " will be ignored.";
    }
  }
}

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
ID3D12Device* WebGpuContext::WeightLoadingD3D12Device() const {
  return weight_loading_d3d12_device_.Get();
}
#endif

Status WebGpuContext::Wait(wgpu::Future f) {
  auto status = instance_.WaitAny(f, UINT64_MAX);
  if (status == wgpu::WaitStatus::Success) {
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to wait for the operation:", uint32_t(status));
}

PendingPipelineBuild* WebGpuContext::FindPendingPipelineBuild(CommandRecordingState& recording,
                                                              std::string_view key) {
  for (auto& dispatch : recording.deferred_dispatches) {
    if (dispatch.program_key == key && dispatch.pending_build) {
      return &*dispatch.pending_build;
    }
  }

  return nullptr;
}

Status WebGpuContext::WaitForDeferredPipelineBuilds(CommandRecordingState& recording) {
  Status result = Status::OK();
  for (auto& dispatch : recording.deferred_dispatches) {
    if (dispatch.compute_pipeline) {
      continue;
    }

    const ProgramArtifact* artifact = program_mgr_->Get(dispatch.program_key);
    // Another thread may populate the cache after this dispatch starts its own build. Still wait
    // for our callback so its completion status is checked even when the cached pipeline is reused.
    if (artifact != nullptr && !dispatch.pending_build) {
      dispatch.compute_pipeline = artifact->compute_pipeline;
      continue;
    }

    if (!dispatch.pending_build) {
      result = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "No cached or pending pipeline for deferred dispatch: ", dispatch.program_key);
      // Drain the rest of the window before discarding all dispatches without encoding them.
      continue;
    }

    // A failed wait does not cancel a WaitAnyOnly callback. Its captured shared ownership keeps
    // the result state alive even if the failed dispatch is discarded.
    auto& build = *dispatch.pending_build;
    Status wait_status = Wait(build.future);
    if (!wait_status.IsOK()) {
      result = wait_status;
      continue;
    }
    if (build.callback_context && !build.callback_context->status.IsOK()) {
      result = build.callback_context->status;
      continue;
    }

    if (artifact == nullptr) {
      ProgramArtifact completed_artifact{std::move(build.name), std::move(build.callback_context->pipeline),
                                         std::move(build.bind_group_layout),
                                         std::move(build.shape_uniform_ranks)};
      artifact = program_mgr_->Set(dispatch.program_key, std::move(completed_artifact));
    }
    dispatch.compute_pipeline = artifact->compute_pipeline;
    dispatch.pending_build.reset();
  }

  return result;
}

Status WebGpuContext::EncodeDeferredDispatches(CommandRecordingState& recording) {
  if (recording.deferred_dispatches.empty()) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(static_cast<size_t>(recording.num_pending_dispatches) +
                            recording.deferred_dispatches.size() <=
                        max_num_pending_dispatches_,
                    "WebGpuContext::EncodeDeferredDispatches: encoded dispatch count (",
                    recording.num_pending_dispatches, ") plus deferred dispatch count (",
                    recording.deferred_dispatches.size(),
                    ") exceeds maxNumPendingDispatches (", max_num_pending_dispatches_, ").");

  auto reset_deferred_state = [&recording]() {
    recording.deferred_dispatches.clear();
  };

  // Resolve every pipeline before encoding so a failed build cannot leave a partially encoded run.
  Status result = WaitForDeferredPipelineBuilds(recording);
  if (!result.IsOK()) {
    reset_deferred_state();
    return result;
  }

  // Encode the recorded dispatches in order, using the same command objects for graph capture.
  for (auto& dispatch : recording.deferred_dispatches) {
    // Preserve profiling info in the captured command for future replays. Otherwise, replay it
    // into the current batch so pending_kernels_ stays in sync with num_pending_dispatches_.
    if (is_profiling_ && dispatch.pending_kernel_info.has_value()) {
      if (recording.graph_capture_state != GraphCaptureState::Capturing) {
        recording.pending_kernels.emplace_back(std::move(*dispatch.pending_kernel_info));
      }
    }
    DispatchCommand(dispatch, recording);
    if (recording.graph_capture_state == GraphCaptureState::Capturing) {
      ORT_ENFORCE(recording.external_captured_commands != nullptr);
      recording.external_captured_commands->push_back({std::move(dispatch)});
    }
  }

  reset_deferred_state();
  return result;
}

Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {
  const auto& inputs = program.Inputs();
  const auto& outputs = program.Outputs();

  if (outputs.empty()) {
    return Status::OK();
  }

  const webgpu::BufferManager& buffer_mgr = ComputeContextBase::BufferManagerAccessor::Get(context);
  CommandRecordingState& recording = ComputeContextBase::BufferManagerAccessor::GetRecording(context);

  // validate inputs and outputs are on WebGPU buffers
  if (ValidationMode() >= ValidationMode::Basic) {
    ORT_ENFORCE(std::all_of(inputs.begin(), inputs.end(), [](const ProgramInput& input) {
                  const auto* tensor = input.tensor;
                  return tensor != nullptr &&
                         tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                         tensor->Location().device.Type() == OrtDevice::GPU &&
                         !strcmp(tensor->Location().name.c_str(), WEBGPU_BUFFER);
                }),
                "All inputs must be tensors on WebGPU buffers.");

    if (program.IndirectDispatchTensor() != nullptr) {
      ORT_ENFORCE(!inputs.empty() && inputs.back().tensor == program.IndirectDispatchTensor(),
                  "The indirect dispatch tensor must be the last input. "
                  "Ensure no call to program.AddInput() occurs after program.SetIndirectDispatchTensor().");
    }

    ORT_ENFORCE(std::all_of(outputs.begin(), outputs.end(), [](const ProgramOutput& output) {
                  const auto* tensor = output.tensor;
                  return tensor != nullptr &&
                         tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                         tensor->Location().device.Type() == OrtDevice::GPU &&
                         !strcmp(tensor->Location().name.c_str(), WEBGPU_BUFFER);
                }),
                "All outputs must be tensors on WebGPU buffers.");
  }

  const ProgramMetadata& metadata = program.Metadata();

  // validate program metadata
  if (ValidationMode() >= ValidationMode::Basic) {
    const auto& [constants, overridable_constants, uniform_variables] = metadata;

    // check overridable constants
    ORT_RETURN_IF(program.OverridableConstants().size() != overridable_constants.size(),
                  "Size of overridable constants mismatch in program \"", program.Name(),
                  "\", Expected: ", overridable_constants.size(),
                  ", Actual: ", program.OverridableConstants().size());

    if (ValidationMode() >= ValidationMode::Full) {
      size_t num_overridable_constants = program.OverridableConstants().size();
      for (size_t i = 0; i < num_overridable_constants; ++i) {
        const auto& override_value = program.OverridableConstants()[i];
        const auto& definition = overridable_constants[i];
        ORT_RETURN_IF(override_value.has_value && override_value.type != definition.type,
                      "Overridable override_value[", i, "] (", definition.name, ") data type mismatch in program \"", program.Name(),
                      "\", Expected: ", definition.type,
                      ", Actual: ", override_value.type);
        ORT_RETURN_IF(!override_value.has_value && !definition.has_default_value,
                      "Overridable override_value[", i, "] (", definition.name, ") no override_value specified in program \"", program.Name(),
                      "\"");
      }
    }

    // check uniform variables
    ORT_RETURN_IF(program.UniformVariables().size() != uniform_variables.size(),
                  "Size of uniform_value variables mismatch in program \"", program.Name(),
                  "\", Expected: ", uniform_variables.size(),
                  ", Actual: ", program.UniformVariables().size());

    if (ValidationMode() >= ValidationMode::Full) {
      size_t num_uniform_variables = program.UniformVariables().size();
      for (size_t i = 0; i < num_uniform_variables; ++i) {
        const auto& uniform_value = program.UniformVariables()[i];
        const auto& definition = uniform_variables[i];
        ORT_RETURN_IF(uniform_value.length > 0 && uniform_value.data_type != definition.data_type,
                      "Uniform variable[", i, "] (", definition.name, ") data type mismatch in program \"", program.Name(),
                      "\", Expected: ", definition.data_type,
                      ", Actual: ", uniform_value.data_type);
      }
    }
  }

  // "Segments" is a feature that allows big buffer to be used in shader.
  //
  // For example, if `maxStorageBufferBindingSize` is 128MB, a 200MB sized input buffer can be split into two segments
  // (128MB + 72MB) to be bound to the shader. In this case, the input segment count is 2. There will be 2 input
  // bindings in the shader for this input buffer.
  //
  // See https://github.com/microsoft/onnxruntime/pull/25962 for more information.

  std::vector<uint32_t> inputs_segments;
  std::vector<uint32_t> outputs_segments;
  ORT_RETURN_IF_ERROR(program_mgr_->CalculateSegmentsForInputsAndOutputs(program, inputs_segments, outputs_segments));

  uint32_t x = program.DispatchGroupSizeX();
  uint32_t y = program.DispatchGroupSizeY();
  uint32_t z = program.DispatchGroupSizeZ();

  // Skip normalization for indirect dispatch since dimensions are determined by the indirect buffer
  if (program.IndirectDispatchTensor() == nullptr) {
    ORT_RETURN_IF_ERROR(program_mgr_->NormalizeDispatchGroupSize(x, y, z));
  } else {
    ORT_ENFORCE(x == 0 && y == 0 && z == 0,
                "Only one of SetIndirectDispatchTensor and SetDispatchGroupSize should be called for program", program.Name());
  }

  auto key = CalculateProgramCacheKey(program, inputs_segments, outputs_segments);

  LOGS(context.Logger(), INFO) << "Starting program \"" << key << "\" (" << x << ", " << y << ", " << z << ")";
  // The program cache prevents duplicate builds across encoded windows.
  // EncodeDeferredDispatches() inserts completed pipelines into this cache before clearing the window.
  const auto* program_artifact = program_mgr_->Get(key);

  // For cache misses, reuse a pending build already owned by this bounded dispatch window instead
  // of starting another build for the same key.
  std::optional<PendingPipelineBuild> pending_build;
  const std::vector<int>* deferred_ranks = nullptr;
  const wgpu::BindGroupLayout* bind_group_layout = nullptr;
  if (program_artifact == nullptr) {
    PendingPipelineBuild* in_flight_build = FindPendingPipelineBuild(recording, key);

    // Reuse an in-flight same-key build instead of compiling the shader again.
    if (in_flight_build == nullptr) {
      auto& build = pending_build.emplace();
      build.name = program.Name();
      build.callback_context = std::make_shared<PipelineCallbackContext>();
      ORT_RETURN_IF_ERROR(program_mgr_->Build(program, metadata, inputs_segments, outputs_segments,
                                              key, x, y, z,
                                              build.bind_group_layout,
                                              build.shape_uniform_ranks,
                                              build.future,
                                              build.callback_context));
      in_flight_build = &*pending_build;
    } else if (program_mgr_->IsShaderCacheValidationEnabled()) {
      ORT_RETURN_IF_ERROR(program_mgr_->ValidateShaderCache(program, inputs_segments, outputs_segments, key, x, y, z));
    }
    deferred_ranks = &in_flight_build->shape_uniform_ranks;
    bind_group_layout = &in_flight_build->bind_group_layout;
  } else {
    if (program_mgr_->IsShaderCacheValidationEnabled()) {
      ORT_RETURN_IF_ERROR(program_mgr_->ValidateShaderCache(program, inputs_segments, outputs_segments, key, x, y, z));
    }
    bind_group_layout = &program_artifact->bind_group_layout;
  }

  // prepare shape uniforms for shader variables (if any) and user defined uniforms
  // On a deferred cache miss, use the ranks produced while starting the pending build; otherwise
  // use the cached artifact's ranks.
  const std::vector<int>& shape_uniform_ranks = deferred_ranks ? *deferred_ranks
                                                               : program_artifact->shape_uniform_ranks;
  std::vector<ProgramUniformVariableValue> shape_uniforms;
  shape_uniforms.reserve(shape_uniform_ranks.size() * 2);
  if (ValidationMode() >= ValidationMode::Basic) {
    ORT_RETURN_IF_NOT(shape_uniform_ranks.size() == inputs.size() + outputs.size() + program.Indices().size(),
                      "Invalid program artifact: variable size (", shape_uniform_ranks.size(),
                      ") does not match current program (input: ", inputs.size(),
                      ", output: ", outputs.size(),
                      ", indices: ", program.Indices().size(), ")");
  }

  auto append_shape_uniforms = [&shape_uniforms, &shape_uniform_ranks](size_t i, const TensorShape& shape) {
    if (shape_uniform_ranks[i] > 0) {
      size_t expected_rank = static_cast<size_t>(shape_uniform_ranks[i]);
      ORT_RETURN_IF(expected_rank != shape.NumDimensions(),
                    "Invalid program artifact: variable[", i, "] rank mismatch. Expected: ", expected_rank,
                    ", Actual: ", shape.NumDimensions());

      std::vector<uint32_t> dims(expected_rank);
      std::vector<uint32_t> stride(expected_rank - 1);
      for (size_t j = 0; j < expected_rank; ++j) {
        dims[j] = onnxruntime::narrow<uint32_t>(shape[j]);
        if (j < expected_rank - 1) {
          stride[j] = onnxruntime::narrow<uint32_t>(shape.SizeFromDimension(j + 1));
        }
      }

      shape_uniforms.emplace_back(gsl::make_span(dims));
      if (expected_rank > 1) {
        shape_uniforms.emplace_back(gsl::make_span(stride));
      }
    }
    return Status::OK();
  };

  for (size_t i = 0; i < inputs.size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i,
                                              inputs[i].use_override_shape ? inputs[i].override_shape : inputs[i].tensor->Shape()));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size(),
                                              outputs[i].use_override_shape ? outputs[i].override_shape : outputs[i].tensor->Shape()));
  }
  for (size_t i = 0; i < program.Indices().size(); i++) {
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size() + outputs.size(), program.Indices()[i]));
  }

  std::vector<ProgramUniformVariableValue> buffer_view_offset_uniforms;
  buffer_view_offset_uniforms.reserve(inputs.size() + outputs.size());
  for (const auto& input : inputs) {
    if (input.is_buffer_view) {
      buffer_view_offset_uniforms.emplace_back(input.buffer_offset_in_elements);
    }
  }
  for (const auto& output : outputs) {
    if (output.is_buffer_view) {
      buffer_view_offset_uniforms.emplace_back(output.buffer_offset_in_elements);
    }
  }

  const size_t uniform_count =
      shape_uniforms.size() + buffer_view_offset_uniforms.size() + program.UniformVariables().size();
  size_t current_offset = 0;
  std::vector<std::tuple<const ProgramUniformVariableValue&, size_t>> uniform_and_offsets;
  uniform_and_offsets.reserve(uniform_count);
  for (size_t i = 0; i < uniform_count; i++) {
    const auto& uniform =
        i < shape_uniforms.size()
            ? shape_uniforms[i]
        : i < shape_uniforms.size() + buffer_view_offset_uniforms.size()
            ? buffer_view_offset_uniforms[i - shape_uniforms.size()]
            : program.UniformVariables()[i - shape_uniforms.size() - buffer_view_offset_uniforms.size()];
    size_t length = uniform.length;
    if (length == 0) {  // skip zero-length uniform
      continue;
    }

    // Calculate the size and alignment of the uniform variable.
    //
    // https://www.w3.org/TR/WGSL/#alignof
    //
    // For f16:
    // - length > 8      : array<vec4<u32>, N>   (align 16) (size 16 * N, N = ceil(length / 8))
    // - length == 7 or 8: vec4<u32>             (align 16) (size 16)
    // - length == 5 or 6: vec3<u32>             (align 16) (size 12)
    // - length == 3 or 4: vec2<u32>             (align 8)  (size 8)
    // - length == 1 or 2: u32                   (align 4)  (size 4)
    //
    // For other types (i32, u32, f32):
    // - length > 4      : array<vec4<T>, N>     (align 16) (size 16 * N, N = ceil(length / 4))
    // - length == 4     : vec4<T>               (align 16) (size 16)
    // - length == 3     : vec3<T>               (align 16) (size 12)
    // - length == 2     : vec2<T>               (align 8)  (size 8)
    // - length == 1     : T                     (align 4)  (size 4)
    //

    const bool is_f16 = uniform.data_type == ProgramUniformVariableDataType::Float16;

    size_t variable_alignment = 4;  // default alignment for scalar types
    size_t variable_size = 4;       // default size for scalar types

    if (is_f16) {
      if (length > 6) {
        variable_alignment = 16;
        variable_size = 16 * ((length + 7) / 8);
      } else if (length > 4) {
        variable_alignment = 16;
        variable_size = 12;
      } else if (length > 2) {
        variable_alignment = 8;
        variable_size = 8;
      }
    } else {
      if (length > 3) {
        variable_alignment = 16;
        variable_size = 16 * ((length + 3) / 4);
      } else if (length > 2) {
        variable_alignment = 16;
        variable_size = 12;
      } else if (length > 1) {
        variable_alignment = 8;
        variable_size = 8;
      }
    }
    current_offset = (current_offset + variable_alignment - 1) / variable_alignment * variable_alignment;
    uniform_and_offsets.emplace_back(uniform, current_offset);

    current_offset += variable_size;
  }

  // Meet alignment of struct here: https://www.w3.org/TR/WGSL/#alignment-and-size. For simplicity, set
  // max_alignment_of_field to 16 since the underlying buffer has been rounded up to 16.
  constexpr size_t max_alignment_of_field = 16;
  const size_t uniform_buffer_total_size = (current_offset + max_alignment_of_field - 1) / max_alignment_of_field * max_alignment_of_field;

  WGPUBuffer uniform_buffer = nullptr;
  if (uniform_buffer_total_size > 0) {
    std::vector<uint8_t> uniform_data_buffer(uniform_buffer_total_size);

    for (auto const& [uniform, offset] : uniform_and_offsets) {
      memcpy(uniform_data_buffer.data() + offset, uniform.data.data(), uniform.data.size());
    }

    uniform_buffer = buffer_mgr.Create(recording, uniform_buffer_total_size,
                                       wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);
    device_queue_.WriteBuffer(uniform_buffer, 0, uniform_data_buffer.data(), uniform_buffer_total_size);
  }

  const size_t total_buffer_count =
      std::count_if(inputs_segments.begin(), inputs_segments.end(), [](uint32_t segments) { return segments != 0; }) +
      std::count_if(outputs_segments.begin(), outputs_segments.end(), [](uint32_t segments) { return segments != 0; }) +
      (uniform_buffer ? 1 : 0);

  std::vector<WGPUBuffer> bind_buffers;
  std::vector<uint32_t> bind_buffers_segments;
  bind_buffers.reserve(total_buffer_count);
  bind_buffers_segments.reserve(total_buffer_count);
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs_segments[i] == 0) {
      continue;
    }
    bind_buffers.push_back(reinterpret_cast<WGPUBuffer>(const_cast<void*>(inputs[i].tensor->DataRaw())));
    bind_buffers_segments.push_back(inputs_segments[i]);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs_segments[i] == 0) {
      continue;
    }
    bind_buffers.push_back(reinterpret_cast<WGPUBuffer>(outputs[i].tensor->MutableDataRaw()));
    bind_buffers_segments.push_back(outputs_segments[i]);
  }
  if (uniform_buffer) {
    bind_buffers.push_back(uniform_buffer);
    bind_buffers_segments.push_back(1);  // uniform buffer defaults to 1 segment
  }

  // Record the ready bind group and return. The deferred drain only needs to wait for the pipeline
  // and encode the dispatch commands.
  webgpu::DispatchCommandInfo command;
  command.program_key = key;
  if (program_artifact != nullptr) {
    command.compute_pipeline = program_artifact->compute_pipeline;
  }
  command.bind_group = CreateBindGroup(bind_buffers, bind_buffers_segments,
                                       *bind_group_layout, program.Name());
  command.pending_build = std::move(pending_build);
  recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
  if (uniform_buffer) {
    // The bind group owns a reference now, so return the allocator's reference immediately.
    buffer_mgr.Release(uniform_buffer, &recording);
  }
  command.dispatch_group = {x, y, z};
  if (program.IndirectDispatchTensor() != nullptr) {
    command.indirect_buffer = reinterpret_cast<WGPUBuffer>(
        const_cast<void*>(program.IndirectDispatchTensor()->DataRaw()));
  }
  // Capture profiling info now (shapes must be read while tensors are alive); replayed in flush.
  if (is_profiling_) {
    command.pending_kernel_info.emplace(context.NodeName(), context.OpType(), program.Name(),
                                        key, inputs, outputs);
  }
  recording.deferred_dispatches.push_back(std::move(command));

  // Drain and submit a full window to bound both recorded and encoded dispatch state. Partial
  // windows are encoded and submitted by the caller at its execution boundary.
  if (static_cast<size_t>(recording.num_pending_dispatches) + recording.deferred_dispatches.size() >=
      max_num_pending_dispatches_) {
    ORT_RETURN_IF_ERROR(Flush(buffer_mgr, recording));
  }
  return Status::OK();
}

std::vector<const char*> WebGpuContext::GetEnabledAdapterToggles() const {
  // See the description of all the toggles in toggles.cpp
  // "use_dxc" for Shader Model 6+ features (e.g. float16)
  // "allow_unsafe_apis" for chromium experimental features
  constexpr const char* toggles[] = {
      "use_dxc",
      "allow_unsafe_apis",
      "decompose_uniform_buffers",
#if defined(DAWN_ENABLE_VULKAN)
      "use_vulkan_memory_model",
      "vulkan_enable_f16_on_nvidia",
#endif
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

std::vector<const char*> WebGpuContext::GetEnabledDeviceToggles() const {
  // Enable / disable other toggles that may affect the performance.
  // Other toggles that may be useful: "dump_shaders", "disable_symbol_renaming"
  constexpr const char* toggles[] = {
      "skip_validation",
      "d3d_disable_ieee_strictness",
  };
  std::vector<const char*> enabled_toggles;
#ifndef NDEBUG
  // validation_mode_explicitly_set_ only changes release behavior; mark it used in debug builds
  // to avoid -Wunused-private-field on toolchains that treat warnings as errors.
  ORT_UNUSED_PARAMETER(validation_mode_explicitly_set_);
  enabled_toggles = std::vector<const char*>(ValidationMode() >= ValidationMode::WGPUOnly
                                                 ? std::begin(toggles) + 1
                                                 : std::begin(toggles),
                                             std::end(toggles));
#else
  // In release/relwithdebinfo builds, default to skip_validation for performance,
  // but honor explicit validationMode overrides.
  if (!validation_mode_explicitly_set_) {
    enabled_toggles = std::vector<const char*>(std::begin(toggles), std::end(toggles));
  } else {
    enabled_toggles = std::vector<const char*>(ValidationMode() >= ValidationMode::WGPUOnly
                                                   ? std::begin(toggles) + 1
                                                   : std::begin(toggles),
                                               std::end(toggles));
  }
#endif

  if (!enable_robustness_) {
    enabled_toggles.push_back("disable_robustness");
  }
  enabled_toggles.push_back("lazy_clear_resource_on_first_use");
  return enabled_toggles;
}

std::vector<const char*> WebGpuContext::GetDisabledDeviceToggles() const {
  constexpr const char* toggles[] = {
      "timestamp_quantization",
  };
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));
}

#if !defined(__wasm__)
namespace detail {

bool CanMapDeviceLocalMemory(gsl::span<const wgpu::MemoryHeapInfo> heaps) {
  uint64_t largest_device_local = 0;
  uint64_t largest_mappable_device_local = 0;
  for (const auto& heap : heaps) {
    if (!(heap.properties & wgpu::HeapProperty::DeviceLocal)) {
      continue;
    }
    largest_device_local = std::max(largest_device_local, heap.size);
    if (heap.properties & wgpu::HeapProperty::HostVisible) {
      largest_mappable_device_local = std::max(largest_mappable_device_local, heap.size);
    }
  }
  return largest_device_local != 0 && largest_mappable_device_local >= largest_device_local;
}

}  // namespace detail

namespace {

bool AdapterCanMapDeviceLocalMemory(const wgpu::Adapter& adapter) {
  // Without heap information keep the existing mapped upload.
  if (!adapter.HasFeature(wgpu::FeatureName::AdapterPropertiesMemoryHeaps)) {
    return true;
  }
  wgpu::AdapterPropertiesMemoryHeaps heaps;
  wgpu::AdapterInfo info;
  info.nextInChain = &heaps;
  ORT_ENFORCE(adapter.GetInfo(&info) == wgpu::Status::Success);
  return detail::CanMapDeviceLocalMemory({heaps.heapInfo, heaps.heapCount});
}

}  // namespace
#endif  // !defined(__wasm__)

std::vector<wgpu::FeatureName> WebGpuContext::GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
#if !defined(__wasm__)
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
      wgpu::FeatureName::ImplicitDeviceSynchronization,
#endif
      wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16,
      wgpu::FeatureName::Subgroups,
      wgpu::FeatureName::SubgroupSizeControl,
#if !defined(__wasm__)
      wgpu::FeatureName::BufferMapExtendedUsages,
#endif
  };
  for (auto feature : features) {
#if !defined(__wasm__)
    if (feature == wgpu::FeatureName::BufferMapExtendedUsages && !AdapterCanMapDeviceLocalMemory(adapter)) {
      continue;
    }
#endif
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  constexpr wgpu::FeatureName d3d12_import_features[]{
      wgpu::FeatureName::SharedBufferMemoryD3D12Resource,
      wgpu::FeatureName::SharedFenceDXGISharedHandle,
  };
  // The default context can outlive the provider that first initialized it
  // (for example, through the shared Env allocator). Request supported import
  // features up front so a later provider can enable accelerated loading on
  // that same Dawn device.
  for (auto feature : d3d12_import_features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    } else {
      ORT_ENFORCE(!IsWeightLoadAccelerationRequired(
                      weight_load_acceleration_mode_),
                  "The selected Dawn D3D12 adapter does not support a feature required by "
                  "weightLoadAcceleration: ",
                  static_cast<uint32_t>(feature));
    }
  }
#endif
  return required_features;
}

wgpu::Limits WebGpuContext::GetRequiredLimits(const wgpu::Adapter& adapter) const {
  wgpu::Limits required_limits{};
  wgpu::Limits adapter_limits;
  ORT_ENFORCE(adapter.GetLimits(&adapter_limits) == wgpu::Status::Success);

  required_limits.maxBindGroups = adapter_limits.maxBindGroups;
  required_limits.maxComputeWorkgroupStorageSize = adapter_limits.maxComputeWorkgroupStorageSize;
  required_limits.maxComputeWorkgroupsPerDimension = adapter_limits.maxComputeWorkgroupsPerDimension;
  required_limits.maxStorageBuffersPerShaderStage = adapter_limits.maxStorageBuffersPerShaderStage;

  if (max_storage_buffer_binding_size_ < kWebGpuGuaranteedMaxStorageBufferBindingSize) {
    // Lower values are applied as a logical segmentation limit after device creation.
    required_limits.maxStorageBufferBindingSize = adapter_limits.maxStorageBufferBindingSize;
  } else {
    required_limits.maxStorageBufferBindingSize = max_storage_buffer_binding_size_;
  }

  required_limits.maxBufferSize = adapter_limits.maxBufferSize;
  required_limits.maxComputeInvocationsPerWorkgroup = adapter_limits.maxComputeInvocationsPerWorkgroup;
  required_limits.maxComputeWorkgroupSizeX = adapter_limits.maxComputeWorkgroupSizeX;
  required_limits.maxComputeWorkgroupSizeY = adapter_limits.maxComputeWorkgroupSizeY;
  required_limits.maxComputeWorkgroupSizeZ = adapter_limits.maxComputeWorkgroupSizeZ;

  return required_limits;
}

void WebGpuContext::WriteTimestamp(CommandRecordingState& recording, uint32_t query_index) {
  if (!is_profiling_ || recording.graph_capture_state == GraphCaptureState::Capturing ||
      query_type_ != TimestampQueryType::InsidePasses) {
    return;
  }

  const auto& compute_pass_encoder = GetComputePassEncoder(recording);
  compute_pass_encoder.WriteTimestamp(query_set_, query_index);
}

void WebGpuContext::StartProfiling() {
  if (query_type_ == TimestampQueryType::None) {
    return;
  }

  is_profiling_ = true;
  // profiling_start_time_ is supplied separately via SetProfilingStartTime, which is
  // driven by WebGpuProfiler::StartProfiling and carries the ORT profiler's CPU time
  // base for both session-level and run-level profiling.
  gpu_timestamp_offset_ = 0;
  profiling_first_submit_cpu_offset_us_ = -1;

  const uint32_t query_count = max_num_pending_dispatches_ * 2;

  if (!query_set_) {
    // Create query set
    wgpu::QuerySetDescriptor querySetDescriptor;
    querySetDescriptor.count = query_count;
    querySetDescriptor.type = wgpu::QueryType::Timestamp;
    query_set_ = device_.CreateQuerySet(&querySetDescriptor);
  }

  if (!query_resolve_buffer_) {
    // Create resolve buffer
    wgpu::BufferDescriptor bufferDescriptor;
    bufferDescriptor.size = query_count * sizeof(uint64_t);
    bufferDescriptor.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc |
                             wgpu::BufferUsage::CopyDst;
    query_resolve_buffer_ = device_.CreateBuffer(&bufferDescriptor);
  }
}

void WebGpuContext::CollectProfilingData(profiling::Events& events) {
  if (!pending_queries_.empty()) {
    // Shift GPU timestamps (which start from 0 at the first submit) onto the ORT
    // profiler's CPU timeline by adding the CPU elapsed time from profiling_start_time_
    // to that first submit. This keeps GPU events aligned with ORT CPU events.
    int64_t cpu_offset_us = profiling_first_submit_cpu_offset_us_ > 0
                                ? profiling_first_submit_cpu_offset_us_
                                : 0;

    for (const auto& pending_query : pending_queries_) {
      const auto& pending_kernels = pending_query.kernels;
      const auto& query_read_buffer = pending_query.query_buffer;

      struct MapAsyncResult {
        wgpu::MapAsyncStatus status{};
        std::string message{};
      } map_async_result;

      ORT_THROW_IF_ERROR(Wait(query_read_buffer.MapAsync(
          wgpu::MapMode::Read,
          0,
          static_cast<size_t>(query_read_buffer.GetSize()),
          wgpu::CallbackMode::WaitAnyOnly,
          // Note: Don't throw from a Dawn callback.
          [](wgpu::MapAsyncStatus status, wgpu::StringView message, MapAsyncResult* result) noexcept {
            result->status = status;
            if (auto message_sv = static_cast<std::string_view>(message);
                !message_sv.empty()) {
              result->message = std::string{message_sv};
            }
          },
          &map_async_result)));

      ORT_ENFORCE(map_async_result.status == wgpu::MapAsyncStatus::Success,
                  "Failed to download data from buffer. wgpu::MapAsyncStatus value: ",
                  static_cast<int>(map_async_result.status), ", message: ", map_async_result.message);

      auto mapped_data = static_cast<const uint64_t*>(query_read_buffer.GetConstMappedRange());

      for (size_t i = 0; i < pending_kernels.size(); i++) {
        const PendingKernelInfo& pending_kernel_info = pending_kernels[i];
        const auto& input_shapes = pending_kernel_info.input_shapes;
        const auto& output_shapes = pending_kernel_info.output_shapes;

        SS(shapes, 128);
        for (size_t s = 0; s < input_shapes.size(); s++) {
          shapes << "inputs[" << s << "] = " << input_shapes[s].ToString() << " ";
        }
        for (size_t s = 0; s < output_shapes.size(); s++) {
          shapes << "outputs[" << s << "] = " << output_shapes[s].ToString() << " ";
        }

        if (gpu_timestamp_offset_ == 0) {
          gpu_timestamp_offset_ = mapped_data[i * 2];
        }
        uint64_t start_time = mapped_data[i * 2] - gpu_timestamp_offset_;
        uint64_t end_time = mapped_data[i * 2 + 1] - gpu_timestamp_offset_;

        InlinedHashMap<std::string, std::string> event_args = {
            {"shapes", SS_GET(shapes)},
            {"cache_key", pending_kernel_info.cache_key},
        };

        profiling::EventRecord event(profiling::API_EVENT,
                                     -1,
                                     -1,
                                     pending_kernel_info.name,
                                     static_cast<int64_t>(std::round(start_time / 1000.0)) + cpu_offset_us,
                                     static_cast<int64_t>(std::round((end_time - start_time) / 1000.0)),
                                     event_args);
        events.emplace_back(std::move(event));
      }

      query_read_buffer.Unmap();
      query_read_buffer.Destroy();
    }

    pending_queries_.clear();
  }

  is_profiling_ = false;
}

void WebGpuContext::CollectProfilingData() {
  CollectProfilingData(events_);
}

void WebGpuContext::EndProfiling(TimePoint /* tp */, profiling::Events& events) {
  // This function is called when no active inference is ongoing.
  ORT_ENFORCE(!is_profiling_, "Profiling is ongoing in an inference run.");

  if (query_type_ != TimestampQueryType::None) {
    // No pending kernels or queries should be present at this point. They should have been collected in CollectProfilingData.
    ORT_ENFORCE(pending_queries_.empty(), "Pending queries are not empty.");

    events.insert(events.end(),
                  std::make_move_iterator(events_.begin()),
                  std::make_move_iterator(events_.end()));
    events_.clear();
  } else {
    LOGS_DEFAULT(WARNING) << "TimestampQuery is not supported in this device.";
  }
}

void WebGpuContext::PushErrorScope(wgpu::ErrorFilter filter) { device_.PushErrorScope(filter); }

Status WebGpuContext::PopErrorScope() {
  // A failed WaitAny does not cancel the callback. Let Dawn retain its result until it fires.
  auto status = std::make_shared<Status>();
  ORT_RETURN_IF_ERROR(Wait(device_.PopErrorScope(
      wgpu::CallbackMode::WaitAnyOnly,
      // Note: Don't throw from a Dawn callback.
      [status](wgpu::PopErrorScopeStatus pop_status, wgpu::ErrorType error_type, wgpu::StringView message) noexcept {
        if (pop_status != wgpu::PopErrorScopeStatus::Success) {
          *status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to pop WebGPU error scope. status=",
                                    static_cast<uint32_t>(pop_status));
        } else if (error_type != wgpu::ErrorType::NoError) {
          *status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU error scope failed. type=",
                                    static_cast<uint32_t>(error_type), ". ", std::string_view(message));
        }
      })));
  return *status;
}

Status WebGpuContext::CheckDeviceStatus() const {
  // Queue work-done and error-scope callbacks can both succeed on an already-lost device.
  // Poll the existing lost future without replacing externally supplied device callbacks.
  const auto wait_status = instance_.WaitAny(device_.GetLostFuture(), 0);
  ORT_RETURN_IF(wait_status == wgpu::WaitStatus::Success, "WebGPU device lost.");
  ORT_RETURN_IF_NOT(wait_status == wgpu::WaitStatus::TimedOut,
                    "Failed to check WebGPU device loss. status=", static_cast<uint32_t>(wait_status));
  std::lock_guard<std::mutex> lock{device_error_state_->mutex};
  return device_error_state_->status;
}

Status WebGpuContext::FlushAndWaitChecked(const webgpu::BufferManager& buffer_mgr,
                                          CommandRecordingState& recording) {
  return RunAndWaitChecked(buffer_mgr, recording, []() { return Status::OK(); });
}

Status WebGpuContext::RunAndWaitChecked(const webgpu::BufferManager& buffer_mgr,
                                        CommandRecordingState& recording,
                                        const std::function<Status()>& enqueue) {
  ORT_RETURN_IF_NOT(HasDevice(), "WebGPU checked completion requires a device.");

  PushErrorScope(wgpu::ErrorFilter::Internal);
  PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
  PushErrorScope(wgpu::ErrorFilter::Validation);

  Status result;
  const auto record_failure = [&result](const Status& status) {
    if (status.IsOK()) {
      return;
    }
    if (result.IsOK()) {
      result = status;
      return;
    }
    result = Status(result.Category(), result.Code(), result.ErrorMessage() + "; " + status.ErrorMessage());
  };

  size_t scopes_to_pop = 3;
  const auto pop_scopes = [&]() {
    while (scopes_to_pop > 0) {
      --scopes_to_pop;
      record_failure(PopErrorScope());
    }
  };
  const auto cleanup = gsl::finally(pop_scopes);

  ORT_TRY {
    record_failure(enqueue());
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      record_failure(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU checked enqueue failed: ", ex.what()));
    });
  }
  ORT_CATCH(...) {
    record_failure(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU checked enqueue failed: unknown exception."));
  }

  ORT_TRY {
    record_failure(Flush(buffer_mgr, recording));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      record_failure(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU checked flush failed: ", ex.what()));
    });
  }

  ORT_TRY {
    auto completion_status = std::make_shared<Status>();
    const auto future = device_queue_.OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [completion_status](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) noexcept {
          if (status != wgpu::QueueWorkDoneStatus::Success) {
            *completion_status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU queue completion failed. status=",
                                                 static_cast<uint32_t>(status), ". ", std::string_view{message});
          }
        });
    const auto wait_status = Wait(future);
    record_failure(wait_status);
    if (wait_status.IsOK()) {
      record_failure(*completion_status);
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      record_failure(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU checked wait failed: ", ex.what()));
    });
  }

  // Do not return early: every scope must be popped even if flushing or waiting failed.
  pop_scopes();
  record_failure(CheckDeviceStatus());
  return result;
}

Status WebGpuContext::Flush(const webgpu::BufferManager& buffer_mgr,
                            CommandRecordingState& recording) {
  // Graph runs and prepacking can release allocator/uniform buffers into the shared manager.
  // Retire this recording in both managers, keeping graph cache policy local to the active one.
  const auto refresh_pending_buffers = [&]() {
    buffer_mgr.RefreshPendingBuffers(recording, recording.graph_capture_state);
    if (&buffer_mgr != buffer_mgr_.get()) {
      buffer_mgr_->RefreshPendingBuffers(recording, GraphCaptureState::Default);
    }
  };

  Status status = EncodeDeferredDispatches(recording);
  if (!recording.command_encoder) {
    if (status.IsOK()) {
      refresh_pending_buffers();
    } else {
      buffer_mgr.DiscardPendingBuffers(recording);
      if (&buffer_mgr != buffer_mgr_.get()) {
        buffer_mgr_->DiscardPendingBuffers(recording);
      }
    }
    return status;
  }

  EndComputePass(recording);

  if (is_profiling_ && recording.num_pending_dispatches > 0 &&
      recording.graph_capture_state != GraphCaptureState::Capturing) {
    ORT_ENFORCE(recording.num_pending_dispatches == recording.pending_kernels.size(),
                "Number of pending dispatches (", recording.num_pending_dispatches,
                ") does not match pending kernels size (", recording.pending_kernels.size(), ")");

    // Capture the CPU elapsed time from the ORT profiler's start to this first submit.
    // Used in CollectProfilingData to offset GPU timestamps onto the ORT CPU timeline.
    if (profiling_first_submit_cpu_offset_us_ < 0) {
      profiling_first_submit_cpu_offset_us_ = TimeDiffMicroSeconds(profiling_start_time_);
    }

    uint32_t query_count = recording.num_pending_dispatches * 2;
    recording.command_encoder.ResolveQuerySet(
        query_set_,
        0,
        query_count,
        query_resolve_buffer_,
        0);

    wgpu::BufferDescriptor bufferDescriptor;
    bufferDescriptor.size = query_count * sizeof(uint64_t);
    bufferDescriptor.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer query_read_buffer = device_.CreateBuffer(&bufferDescriptor);

    recording.command_encoder.CopyBufferToBuffer(
        query_resolve_buffer_,
        0,
        query_read_buffer,
        0,
        query_count * sizeof(uint64_t));

    pending_queries_.emplace_back(std::move(recording.pending_kernels), query_read_buffer);
    recording.pending_kernels.clear();
  }
  auto command_buffer = recording.command_encoder.Finish();
  device_queue_.Submit(1, &command_buffer);
  refresh_pending_buffers();
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  std::vector<std::function<void()>> pending_release_callbacks;
  {
    std::lock_guard<std::mutex> callbacks_lock{
        recording.pending_release_callbacks_mutex};
    pending_release_callbacks.swap(recording.pending_release_callbacks);
  }
  for (auto& release : pending_release_callbacks) {
    release();
  }
#endif
  recording.command_encoder = nullptr;
  recording.num_pending_dispatches = 0;
  return status;
}

wgpu::BindGroup WebGpuContext::CreateBindGroup(const std::vector<WGPUBuffer>& bind_buffers,
                                               const std::vector<uint32_t>& bind_buffers_segments,
                                               const wgpu::BindGroupLayout& bind_group_layout,
                                               std::string_view label) const {
  uint32_t entry_index = 0;
  std::vector<WGPUBindGroupEntry> bind_group_entries;

  const uint64_t kMaxBufferSize = device_limits_.maxStorageBufferBindingSize;
  for (size_t buffer_idx = 0; buffer_idx < bind_buffers.size(); ++buffer_idx) {
    WGPUBuffer buffer = bind_buffers[buffer_idx];
    const uint32_t total_segments = bind_buffers_segments[buffer_idx];
    // `total_segments` we used is calculated by tensor size, not actual buffer size. Because for bucketed buffer,
    // the actual buffer size may be larger than the tensor size, an extreme case is that tensor size = 127MB, buffer size = 256MB,
    // maxStorageBufferBindingSize = 128MB, in this case we only need to bind 1 segment instead of 2 segments because
    // there is no data for the second segment.
    if (total_segments > 1) {
      uint64_t offset = 0;
      uint64_t buffer_size = wgpuBufferGetSize(buffer);
      for (uint32_t segment = 0; segment < total_segments; ++segment) {
        uint64_t segment_size = std::min(kMaxBufferSize, buffer_size - offset);
        bind_group_entries.push_back({nullptr, entry_index++, buffer, offset, segment_size, nullptr, nullptr});
        offset += segment_size;
      }
    } else {
      bind_group_entries.push_back({nullptr, entry_index++, buffer, 0, WGPU_WHOLE_SIZE, nullptr, nullptr});
    }
  }

  ORT_ENFORCE(entry_index <= device_limits_.maxBindingsPerBindGroup, "Number of bind group entries (", entry_index,
              ") exceeds device limit (", device_limits_.maxBindingsPerBindGroup, ").");

  WGPUBindGroupDescriptor bind_group_desc{};
  bind_group_desc.layout = bind_group_layout.Get();
  bind_group_desc.entryCount = bind_group_entries.size();
  bind_group_desc.entries = bind_group_entries.data();
  bind_group_desc.label = {label.data(), label.length()};

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(Device().Get(), &bind_group_desc);
  ORT_ENFORCE(bind_group != nullptr, "Failed to create bind group for program ", label, ".");
  return wgpu::BindGroup::Acquire(bind_group);
}

void WebGpuContext::DispatchCommand(const webgpu::DispatchCommandInfo& command,
                                    CommandRecordingState& recording) {
  ORT_ENFORCE(command.compute_pipeline.has_value());
  ORT_ENFORCE(command.bind_group != nullptr);
  const auto& compute_pass_encoder = GetComputePassEncoder(recording);
  WriteTimestamp(recording, recording.num_pending_dispatches * 2);
  compute_pass_encoder.SetPipeline(*command.compute_pipeline);
  compute_pass_encoder.SetBindGroup(0, command.bind_group);

  if (command.indirect_buffer != nullptr) {
    compute_pass_encoder.DispatchWorkgroupsIndirect(command.indirect_buffer, 0);
  } else {
    compute_pass_encoder.DispatchWorkgroups(command.dispatch_group[0],
                                            command.dispatch_group[1],
                                            command.dispatch_group[2]);
  }
  WriteTimestamp(recording, recording.num_pending_dispatches * 2 + 1);
  ++recording.num_pending_dispatches;
  if (recording.num_pending_dispatches >= max_num_pending_dispatches_ ||
      (is_profiling_ && query_type_ == TimestampQueryType::AtPasses)) {
    EndComputePass(recording);
  }
}

void WebGpuContext::CopyBuffer(WGPUBuffer source, WGPUBuffer destination, uint64_t size,
                               CommandRecordingState& recording) {
  // Materialize earlier deferred dispatches before inserting the copy into either command sequence.
  ORT_THROW_IF_ERROR(EncodeDeferredDispatches(recording));
  EndComputePass(recording);
  GetCommandEncoder(recording).CopyBufferToBuffer(source, 0, destination, 0, size);
  if (recording.graph_capture_state == GraphCaptureState::Capturing) {
    ORT_ENFORCE(recording.external_captured_commands != nullptr);
    recording.external_captured_commands->push_back(
        {CapturedBufferCopy{wgpu::Buffer{source}, wgpu::Buffer{destination}, size}});
  }
}

void WebGpuContext::ClearBuffer(WGPUBuffer buffer, uint64_t size, CommandRecordingState& recording) {
  // Materialize earlier deferred dispatches before inserting the clear into either command sequence.
  ORT_THROW_IF_ERROR(EncodeDeferredDispatches(recording));
  EndComputePass(recording);
  GetCommandEncoder(recording).ClearBuffer(buffer, 0, size);
  if (recording.graph_capture_state == GraphCaptureState::Capturing) {
    ORT_ENFORCE(recording.external_captured_commands != nullptr);
    recording.external_captured_commands->push_back({CapturedBufferClear{wgpu::Buffer{buffer}, size}});
  }
}

void WebGpuContext::CaptureBegin(std::vector<webgpu::CapturedCommand>* captured_commands,
                                 const webgpu::BufferManager& buffer_manager,
                                 CommandRecordingState& recording) {
  LOGS_DEFAULT(VERBOSE) << "CaptureBegin with external storage";
  // Flush any pending commands before we change the status
  ORT_THROW_IF_ERROR(Flush(buffer_manager, recording));
  recording.external_captured_commands = captured_commands;

  // Make sure the external vector is empty before we start capturing
  if (recording.external_captured_commands) {
    recording.external_captured_commands->clear();
  }

  recording.graph_capture_state = GraphCaptureState::Capturing;
}

void WebGpuContext::Replay(const std::vector<webgpu::CapturedCommand>& captured_commands,
                           const webgpu::BufferManager& buffer_manager,
                           CommandRecordingState& recording) {
  LOGS_DEFAULT(VERBOSE) << "Replay with external storage";
  recording.graph_capture_state = GraphCaptureState::Replaying;
  // Replay all captured commands from the provided vector
  const size_t command_count = captured_commands.size();
  for (size_t i = 0; i < command_count; ++i) {
    const auto& operation = captured_commands[i].operation;
    if (const auto* copy = std::get_if<CapturedBufferCopy>(&operation)) {
      CopyBuffer(copy->source.Get(), copy->destination.Get(), copy->size, recording);
      continue;
    }
    if (const auto* clear = std::get_if<CapturedBufferClear>(&operation)) {
      ClearBuffer(clear->buffer.Get(), clear->size, recording);
      continue;
    }
    const auto& command = std::get<DispatchCommandInfo>(operation);
    // Restore profiling info when profiling is enabled. All commands are expected
    // to have profiling data in this mode to keep pending_kernels_ consistent
    // with num_pending_dispatches_.
    if (is_profiling_) {
      ORT_ENFORCE(command.pending_kernel_info.has_value(),
                  "WebGpuContext::Replay: profiling is enabled but captured command at index ",
                  i,
                  " is missing pending_kernel_info.");
      recording.pending_kernels.emplace_back(*command.pending_kernel_info);
    }

    DispatchCommand(command, recording);
    if (recording.num_pending_dispatches >= max_num_pending_dispatches_) {
      ORT_THROW_IF_ERROR(Flush(buffer_manager, recording));
    }
  }

  // Flush any remaining commands
  ORT_THROW_IF_ERROR(Flush(buffer_manager, recording));

  recording.graph_capture_state = GraphCaptureState::Default;
}

void WebGpuContext::CaptureEnd(CommandRecordingState& recording) {
  LOGS_DEFAULT(VERBOSE) << "CaptureEnd";

  recording.graph_capture_state = GraphCaptureState::Default;
  recording.external_captured_commands = nullptr;
}

void WebGpuContext::ReleaseGraphResources(std::vector<webgpu::CapturedCommand>& captured_commands) {
  LOGS_DEFAULT(VERBOSE) << "ReleaseGraphResources: Releasing " << captured_commands.size() << " captured command resources";

  captured_commands.clear();
}

std::mutex WebGpuContextFactory::mutex_;
std::once_flag WebGpuContextFactory::init_default_flag_;

std::unordered_map<int32_t, WebGpuContextFactory::WebGpuContextInfo>* WebGpuContextFactory::contexts_ = nullptr;
WebGpuInstanceWrapper* WebGpuContextFactory::default_instance_ = nullptr;

WebGpuContext& WebGpuContextFactory::CreateContext(const WebGpuContextConfig& config) {
  const int context_id = config.context_id;
  WGPUInstance instance = config.instance;
  WGPUDevice device = config.device;

  std::call_once(init_default_flag_, [
#if !defined(__wasm__)
                                         dawn_proc_table = config.dawn_proc_table
#endif
  ]() {
  // Setup dawn proc table (only for non-WASM build)

#if !defined(__wasm__)
    const DawnProcTable* dawn_procs = reinterpret_cast<const DawnProcTable*>(dawn_proc_table);
#if defined(BUILD_DAWN_SHARED_LIBRARY)
    ORT_ENFORCE(dawn_procs == nullptr, "setting DawnProcTable is not allowed when dynamically linked to webgpu_dawn.");
#else
#if !defined(USE_EXTERNAL_DAWN)
    if (dawn_procs == nullptr) {
      dawn_procs = &GetBundledDawnProcs();
    }
#else
    ORT_ENFORCE(dawn_procs != nullptr, "DawnProcTable must be provided.");
#endif
    dawnProcSetProcs(dawn_procs);
#endif
#endif
  });

  std::lock_guard<std::mutex> lock(mutex_);

  if (default_instance_ == nullptr) {
    // Create wgpu::Instance
    wgpu::InstanceFeatureName required_instance_features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
    wgpu::InstanceDescriptor instance_desc{};
    instance_desc.requiredFeatures = required_instance_features;
    instance_desc.requiredFeatureCount = sizeof(required_instance_features) / sizeof(required_instance_features[0]);
    default_instance_ = new WebGpuInstanceWrapper(instance_desc);
  }

  if (context_id == 0) {
    ORT_ENFORCE(instance == nullptr && device == nullptr,
                "WebGPU EP default contexts must not have custom WebGPU instance or device.");

    instance = default_instance_->Get();
  } else {
    // for context ID > 0, user must provide custom WebGPU instance and device.
    ORT_ENFORCE(instance != nullptr && device != nullptr,
                "WebGPU EP custom context (contextId>0) must have custom WebGPU instance and device.");
  }

  // Lazy-allocate the contexts map on first use (heap-allocated to avoid static destruction crash).
  if (contexts_ == nullptr) {
    contexts_ = new std::unordered_map<int32_t, WebGpuContextInfo>();
  }

  auto it = contexts_->find(context_id);
  if (it == contexts_->end()) {
    GSL_SUPPRESS(r.11)
    auto context = std::unique_ptr<WebGpuContext>(new WebGpuContext(instance,
                                                                    device,
                                                                    config.validation_mode,
                                                                    config.validation_mode_explicitly_set,
                                                                    config.preserve_device,
                                                                    config.max_storage_buffer_binding_size,
                                                                    config.test_only_max_storage_buffer_binding_size,
                                                                    config.max_storage_buffers_per_shader_stage));
    it = contexts_->emplace(context_id, WebGpuContextFactory::WebGpuContextInfo{std::move(context), 0}).first;
  } else if (context_id != 0) {
    ORT_ENFORCE(it->second.context->instance_.Get() == instance &&
                    it->second.context->device_.Get() == device,
                "WebGPU EP context ID ", context_id, " is already created with different WebGPU instance or device.");
  }
  it->second.ref_count++;

  // perform initialization; on failure, undo the ref_count increment and remove the entry
  // if this was the first (and only) reference, so we don't leave a zombie context in the map
  // that would later deadlock during Cleanup().
  ORT_TRY {
    it->second.context->Initialize(config);
  }
  ORT_CATCH(...) {
    if (--it->second.ref_count == 0) {
      contexts_->erase(it);
    }
    ORT_RETHROW;
  }

  return *it->second.context;
}

WebGpuContext& WebGpuContextFactory::GetContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  ORT_ENFORCE(contexts_ != nullptr, "WebGPU contexts have not been initialized or have been cleaned up.");
  auto it = contexts_->find(context_id);
  ORT_ENFORCE(it != contexts_->end(), "WebGPU EP context ID ", context_id, " is not found.");

  return *it->second.context;
}

void WebGpuContextFactory::RetainContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  ORT_ENFORCE(contexts_ != nullptr, "WebGPU contexts have not been initialized or have been cleaned up.");
  auto it = contexts_->find(context_id);
  ORT_ENFORCE(it != contexts_->end(), "WebGPU EP context ID ", context_id, " is not found.");

  ++it->second.ref_count;
}

void WebGpuContextFactory::ReleaseContext(int context_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  ORT_ENFORCE(contexts_ != nullptr, "WebGPU contexts have not been initialized or have been cleaned up.");
  auto it = contexts_->find(context_id);
  ORT_ENFORCE(it != contexts_->end(), "WebGPU EP context ID ", context_id, " is not found.");

  if (--it->second.ref_count == 0 && !it->second.context->preserve_device_) {
    contexts_->erase(it);
  }
}

void WebGpuContextFactory::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (contexts_ != nullptr) {
    delete contexts_;
    contexts_ = nullptr;
  }

  if (default_instance_ != nullptr) {
    delete default_instance_;
    default_instance_ = nullptr;
  }
}

WebGpuContext& WebGpuContextFactory::DefaultContext() {
  WebGpuContextConfig config{};
  return WebGpuContextFactory::CreateContext(config);
}

void CleanupWebGpuContexts() {
  WebGpuContextFactory::Cleanup();
}

WGPUDevice GetDevice(int context_id) {
  return WebGpuContextFactory::GetContext(context_id).Device().Get();
}

}  // namespace webgpu
}  // namespace onnxruntime
