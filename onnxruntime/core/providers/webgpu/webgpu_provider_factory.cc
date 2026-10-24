// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <limits>

#include "core/framework/error_code_helper.h"
#include "core/framework/ortdevice.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"

#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/providers/webgpu/data_transfer.h"

#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/runtime_compatibility.h"
#include "core/providers/webgpu/ep/sync_stream.h"
#endif

using namespace onnxruntime::webgpu;
using namespace onnxruntime::webgpu::options;

namespace onnxruntime {
struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory(int context_id, WebGpuContext& context, WebGpuExecutionProviderConfig&& webgpu_ep_config)
      : context_id_{context_id}, context_{context}, config_{std::move(webgpu_ep_config)} {
  }

  ~WebGpuProviderFactory() override {
    WebGpuContextFactory::ReleaseContext(context_id_);
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_id_, context_, WebGpuExecutionProviderConfig{config_});
  }

 private:
  int context_id_;
  WebGpuContext& context_;
  WebGpuExecutionProviderConfig config_;
};

namespace {

constexpr const char* WeightLoadAccelerationModeName(
    WeightLoadAccelerationMode mode) {
  switch (mode) {
    case WeightLoadAccelerationMode::Off:
      return kWeightLoadAcceleration_Off;
    case WeightLoadAccelerationMode::Preferred:
      return kWeightLoadAcceleration_Preferred;
    case WeightLoadAccelerationMode::Required:
      return kWeightLoadAcceleration_Required;
  }
  return "unknown";
}

WeightLoadAccelerationMode ParseWeightLoadAccelerationMode(
    const ConfigOptions& config_options) {
  std::string value;
  if (!config_options.TryGetConfigEntry(kWeightLoadAcceleration, value) ||
      value == kWeightLoadAcceleration_Off) {
    return WeightLoadAccelerationMode::Off;
  }
  if (value == kWeightLoadAcceleration_Preferred) {
    return WeightLoadAccelerationMode::Preferred;
  }
  if (value == kWeightLoadAcceleration_Required) {
    return WeightLoadAccelerationMode::Required;
  }
  ORT_THROW(
      "Invalid weightLoadAcceleration value: ", value,
      ". Must be \"off\", \"preferred\", or \"required\".");
}

WeightLoadAccelerationMode ResolveWeightLoadAccelerationMode(
    const ConfigOptions& config_options) {
  const auto mode =
      ParseWeightLoadAccelerationMode(config_options);
  const bool use_environment_allocators =
      config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigUseEnvAllocators, "0") == "1";
  if (!use_environment_allocators ||
      mode == WeightLoadAccelerationMode::Off) {
    return mode;
  }
  ORT_ENFORCE(
      mode != WeightLoadAccelerationMode::Required,
      "weightLoadAcceleration=\"required\" is incompatible with "
      "session.use_env_allocators=1.");
  LOGS_DEFAULT(WARNING)
      << "weightLoadAcceleration=\"preferred\" is disabled because "
         "session.use_env_allocators=1.";
  return WeightLoadAccelerationMode::Off;
}

WebGpuExecutionProviderConfig ParseEpConfig(
    const ConfigOptions& config_options,
    WeightLoadAccelerationMode weight_load_acceleration_mode) {
  WebGpuExecutionProviderConfig webgpu_ep_config{};

  if (std::string preferred_layout_str;
      config_options.TryGetConfigEntry(kPreferredLayout, preferred_layout_str)) {
    if (preferred_layout_str == kPreferredLayout_NHWC) {
      webgpu_ep_config.data_layout = DataLayout::NHWC;
    } else if (preferred_layout_str == kPreferredLayout_NCHW) {
      webgpu_ep_config.data_layout = DataLayout::NCHW;
    } else {
      ORT_THROW("Invalid preferred layout: ", preferred_layout_str);
    }
  }

  if (std::string enable_graph_capture_str;
      config_options.TryGetConfigEntry(kEnableGraphCapture, enable_graph_capture_str)) {
    if (enable_graph_capture_str == kEnableGraphCapture_ON) {
      webgpu_ep_config.enable_graph_capture = true;
    } else if (enable_graph_capture_str == kEnableGraphCapture_OFF) {
      webgpu_ep_config.enable_graph_capture = false;
    } else {
      ORT_THROW("Invalid enable graph capture: ", enable_graph_capture_str);
    }
  }

  webgpu_ep_config.weight_load_acceleration_mode =
      weight_load_acceleration_mode;

  if (std::string pool_generations_str;
      config_options.TryGetConfigEntry(kSessionBufferPoolGenerations, pool_generations_str)) {
    size_t pool_generations = 0;
    const char* begin = pool_generations_str.data();
    const char* end = begin + pool_generations_str.size();
    auto result = std::from_chars(begin, end, pool_generations);
    if (result.ec == std::errc{} && result.ptr == end) {
      webgpu_ep_config.session_buffer_pool_generations = pool_generations;
    } else {
      ORT_THROW("Invalid sessionBufferPoolGenerations value: ", pool_generations_str, ". Must be a non-negative integer.");
    }
  }

  std::string enable_int64_str;
  if (config_options.TryGetConfigEntry(kEnableInt64, enable_int64_str)) {
    if (enable_int64_str == kEnableInt64_ON) {
      webgpu_ep_config.enable_int64 = true;
    } else if (enable_int64_str == kEnableInt64_OFF) {
      webgpu_ep_config.enable_int64 = false;
    } else {
      ORT_THROW("Invalid enableInt64 value: ", enable_int64_str);
    }
  }

  std::string multi_rotary_cache_concat_offset_str;
  if (config_options.TryGetConfigEntry(kMultiRotaryCacheConcatOffset, multi_rotary_cache_concat_offset_str)) {
    uint32_t offset_value = 0;
    auto result = std::from_chars(multi_rotary_cache_concat_offset_str.data(),
                                  multi_rotary_cache_concat_offset_str.data() + multi_rotary_cache_concat_offset_str.size(),
                                  offset_value);
    if (result.ec == std::errc{}) {
      webgpu_ep_config.multi_rotary_cache_concat_offset = offset_value;
    } else {
      ORT_THROW("Invalid multiRotaryCacheConcatOffset value: ", multi_rotary_cache_concat_offset_str, ". Must be a non-negative integer.");
    }
  }

  std::string kv_cache_quantization_bits_str;
  if (config_options.TryGetConfigEntry(kKvCacheQuantizationBits, kv_cache_quantization_bits_str)) {
    if (kv_cache_quantization_bits_str == kKvCacheQuantizationBits_OFF) {
      webgpu_ep_config.kv_cache_quantization_bits = 0;
    } else if (kv_cache_quantization_bits_str == kKvCacheQuantizationBits_4Bit) {
      webgpu_ep_config.kv_cache_quantization_bits = 4;
    } else if (kv_cache_quantization_bits_str == kKvCacheQuantizationBits_8Bit) {
      webgpu_ep_config.kv_cache_quantization_bits = 8;
    } else {
      ORT_THROW("Invalid kvCacheQuantizationBits value: ", kv_cache_quantization_bits_str,
                ". Must be \"0\", \"4\", or \"8\".");
    }
  }

  std::string enable_matmul_fp32_accumulation_str;
  if (config_options.TryGetConfigEntry(kEnableMatmulFp32Accumulation, enable_matmul_fp32_accumulation_str)) {
    if (enable_matmul_fp32_accumulation_str == kEnableMatmulFp32Accumulation_ON) {
      webgpu_ep_config.enable_matmul_fp32_accumulation = true;
    } else if (enable_matmul_fp32_accumulation_str == kEnableMatmulFp32Accumulation_OFF) {
      webgpu_ep_config.enable_matmul_fp32_accumulation = false;
    } else {
      ORT_THROW("Invalid enableMatmulFp32Accumulation value: ", enable_matmul_fp32_accumulation_str, ". Must be \"0\" or \"1\".");
    }
  }

  // parse force CPU node names
  // The force CPU node names are separated by EOL (\n or \r\n) in the config entry.
  // each line is a node name that will be forced to run on CPU.

  if (std::string force_cpu_node_names_str;
      config_options.TryGetConfigEntry(kForceCpuNodeNames, force_cpu_node_names_str)) {
    // split the string by EOL (\n or \r\n)
    std::istringstream ss(force_cpu_node_names_str);
    std::string line;
    while (std::getline(ss, line)) {
      // skip empty lines
      if (line.empty()) {
        continue;
      }

      webgpu_ep_config.force_cpu_node_names.push_back(line);
    }
  }

  // enable pix capture
  if (std::string enable_pix_capture_str;
      config_options.TryGetConfigEntry(kEnablePIXCapture, enable_pix_capture_str)) {
    if (enable_pix_capture_str == kEnablePIXCapture_ON) {
      webgpu_ep_config.enable_pix_capture = true;
    } else if (enable_pix_capture_str == kEnablePIXCapture_OFF) {
      webgpu_ep_config.enable_pix_capture = false;
    } else {
      ORT_THROW("Invalid enable pix capture: ", enable_pix_capture_str);
    }
  }

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP preferred layout: " << int(webgpu_ep_config.data_layout);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP graph capture enable: " << webgpu_ep_config.enable_graph_capture;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP weight load acceleration mode: "
                        << WeightLoadAccelerationModeName(webgpu_ep_config.weight_load_acceleration_mode);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP force CPU node count: " << webgpu_ep_config.force_cpu_node_names.size();
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP pix capture enable: " << webgpu_ep_config.enable_pix_capture;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP enable int64: " << webgpu_ep_config.enable_int64;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP multi rotary cache concat offset: " << webgpu_ep_config.multi_rotary_cache_concat_offset;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP session buffer pool generations: " << webgpu_ep_config.session_buffer_pool_generations;

  return webgpu_ep_config;
}

WebGpuContextConfig ParseWebGpuContextConfig(
    const ConfigOptions& config_options,
    WeightLoadAccelerationMode weight_load_acceleration_mode) {
  WebGpuContextConfig config{};

  config.weight_load_acceleration_mode =
      weight_load_acceleration_mode;

  if (std::string context_id_str;
      config_options.TryGetConfigEntry(kDeviceId, context_id_str)) {
    const auto result = std::from_chars(context_id_str.data(), context_id_str.data() + context_id_str.size(),
                                        config.context_id);
    ORT_ENFORCE(result.ec == std::errc{} && result.ptr == context_id_str.data() + context_id_str.size() &&
                    config.context_id >= 0 && config.context_id <= std::numeric_limits<OrtDevice::DeviceId>::max(),
                "Invalid deviceId value: ", context_id_str, ". Must be an integer in the range 0 to ",
                std::numeric_limits<OrtDevice::DeviceId>::max(), ".");
  }

  if (std::string adapter_index_str;
      config_options.TryGetConfigEntry(kAdapterIndex, adapter_index_str)) {
    uint32_t adapter_index = 0;
    const auto result = std::from_chars(adapter_index_str.data(),
                                        adapter_index_str.data() + adapter_index_str.size(),
                                        adapter_index);
    ORT_ENFORCE(result.ec == std::errc{} && result.ptr == adapter_index_str.data() + adapter_index_str.size(),
                "Invalid adapterIndex value: ", adapter_index_str,
                ". Must be a non-negative integer.");
    config.adapter_index = adapter_index;
  }

#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  ORT_ENFORCE(!config.adapter_index,
              "adapterIndex requires a native Dawn build with adapter enumeration support.");
#endif

  if (std::string webgpu_instance_str;
      config_options.TryGetConfigEntry(kWebGpuInstance, webgpu_instance_str)) {
    static_assert(sizeof(WGPUInstance) == sizeof(size_t), "WGPUInstance size mismatch");
    size_t webgpu_instance = 0;
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_instance_str.data(), webgpu_instance_str.data() + webgpu_instance_str.size(), webgpu_instance).ec);
    config.instance = reinterpret_cast<WGPUInstance>(webgpu_instance);
  }

  if (std::string webgpu_device_str;
      config_options.TryGetConfigEntry(kWebGpuDevice, webgpu_device_str)) {
    static_assert(sizeof(WGPUDevice) == sizeof(size_t), "WGPUDevice size mismatch");
    size_t webgpu_device = 0;
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_device_str.data(), webgpu_device_str.data() + webgpu_device_str.size(), webgpu_device).ec);
    config.device = reinterpret_cast<WGPUDevice>(webgpu_device);
  }

  if (std::string dawn_proc_table_str;
      config_options.TryGetConfigEntry(kDawnProcTable, dawn_proc_table_str)) {
    size_t dawn_proc_table = 0;
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(dawn_proc_table_str.data(), dawn_proc_table_str.data() + dawn_proc_table_str.size(), dawn_proc_table).ec);
    config.dawn_proc_table = reinterpret_cast<const void*>(dawn_proc_table);
  }

  if (std::string validation_mode_str;
      config_options.TryGetConfigEntry(kValidationMode, validation_mode_str)) {
    config.validation_mode_explicitly_set = true;
    if (validation_mode_str == kValidationMode_Disabled) {
      config.validation_mode = ValidationMode::Disabled;
    } else if (validation_mode_str == kValidationMode_wgpuOnly) {
      config.validation_mode = ValidationMode::WGPUOnly;
    } else if (validation_mode_str == kValidationMode_basic) {
      config.validation_mode = ValidationMode::Basic;
    } else if (validation_mode_str == kValidationMode_full) {
      config.validation_mode = ValidationMode::Full;
    } else {
      ORT_THROW("Invalid validation mode: ", validation_mode_str);
    }
  }

  if (std::string enable_robustness_str;
      config_options.TryGetConfigEntry(kEnableRobustness, enable_robustness_str)) {
    config.enable_robustness_explicitly_set = true;
    if (enable_robustness_str == kEnableRobustness_ON) {
      config.enable_robustness = true;
    } else if (enable_robustness_str == kEnableRobustness_OFF) {
      config.enable_robustness = false;
    } else {
      ORT_THROW("Invalid enableRobustness value: ", enable_robustness_str, ". Must be \"0\" or \"1\".");
    }
  }

  if (std::string preserve_device_str;
      config_options.TryGetConfigEntry(kPreserveDevice, preserve_device_str)) {
    if (preserve_device_str == kPreserveDevice_ON) {
      config.preserve_device = true;
    } else if (preserve_device_str == kPreserveDevice_OFF) {
      config.preserve_device = false;
    } else {
      ORT_THROW("Invalid preserve device: ", preserve_device_str);
    }
  }

  // Compile-only mode (skip Dawn adapter/device creation so graph transformation can run device-free)
  // is derived from the session config kOrtSessionOptionCompileOnly, which the Compile API sets
  // automatically. This is not a WebGPU-specific option.
  config.compile_only = config_options.GetConfigOrDefault(kOrtSessionOptionCompileOnly, "0") == "1";

  std::string max_storage_buffer_binding_size_str;
  if (config_options.TryGetConfigEntry(kMaxStorageBufferBindingSize, max_storage_buffer_binding_size_str)) {
    ORT_ENFORCE(
        std::errc{} == std::from_chars(
                           max_storage_buffer_binding_size_str.data(),
                           max_storage_buffer_binding_size_str.data() + max_storage_buffer_binding_size_str.size(),
                           config.max_storage_buffer_binding_size)
                           .ec,
        "Invalid maxStorageBufferBindingSize value: ", max_storage_buffer_binding_size_str);
  }

  std::string max_storage_buffers_per_shader_stage_str;
  if (config_options.TryGetConfigEntry(kMaxStorageBuffersPerShaderStage,
                                       max_storage_buffers_per_shader_stage_str)) {
    ORT_ENFORCE(
        std::errc{} == std::from_chars(
                           max_storage_buffers_per_shader_stage_str.data(),
                           max_storage_buffers_per_shader_stage_str.data() +
                               max_storage_buffers_per_shader_stage_str.size(),
                           config.max_storage_buffers_per_shader_stage)
                           .ec,
        "Invalid maxStorageBuffersPerShaderStage value: ", max_storage_buffers_per_shader_stage_str);
    ORT_ENFORCE(config.max_storage_buffers_per_shader_stage > 0,
                "maxStorageBuffersPerShaderStage must be greater than 0");
  }

  std::string max_num_pending_dispatches_str;
  if (config_options.TryGetConfigEntry(
          kMaxNumPendingDispatches,
          max_num_pending_dispatches_str)) {
    ORT_ENFORCE(
        std::errc{} ==
            std::from_chars(
                max_num_pending_dispatches_str.data(),
                max_num_pending_dispatches_str.data() +
                    max_num_pending_dispatches_str.size(),
                config.max_num_pending_dispatches)
                .ec,
        "Invalid maxNumPendingDispatches value: ",
        max_num_pending_dispatches_str);
    ORT_ENFORCE(
        config.max_num_pending_dispatches > 0,
        "maxNumPendingDispatches must be greater than 0");
    // Practical cap to avoid excessive query buffer sizing and
    // unpredictable memory/performance behavior.
    ORT_ENFORCE(
        config.max_num_pending_dispatches <= 4096,
        "maxNumPendingDispatches must be less than or equal to 4096");
  }

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Device ID: " << config.context_id;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP adapter index: "
                        << (config.adapter_index ? std::to_string(*config.adapter_index) : "automatic");
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUInstance: " << reinterpret_cast<size_t>(config.instance);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUDevice: " << reinterpret_cast<size_t>(config.device);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP DawnProcTable: " << reinterpret_cast<size_t>(config.dawn_proc_table);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP ValidationMode: " << config.validation_mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP enable robustness: " << config.enable_robustness;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP PreserveDevice: " << config.preserve_device;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP CompileOnly: " << config.compile_only;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP max storage buffer binding size: " << config.max_storage_buffer_binding_size;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP max storage buffers per shader stage: "
                        << config.max_storage_buffers_per_shader_stage;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP max pending dispatches: " << config.max_num_pending_dispatches;

  // buffer cache modes
  auto parse_buffer_cache_mode = [&config_options](const std::string& config_entry_str,
                                                   BufferCacheMode& value) -> void {
    std::string buffer_cache_mode_str;
    if (config_options.TryGetConfigEntry(config_entry_str, buffer_cache_mode_str)) {
      if (buffer_cache_mode_str == kBufferCacheMode_Disabled) {
        value = BufferCacheMode::Disabled;
      } else if (buffer_cache_mode_str == kBufferCacheMode_LazyRelease) {
        value = BufferCacheMode::LazyRelease;
      } else if (buffer_cache_mode_str == kBufferCacheMode_Simple) {
        value = BufferCacheMode::Simple;
      } else if (buffer_cache_mode_str == kBufferCacheMode_Bucket) {
        value = BufferCacheMode::Bucket;
      } else {
        ORT_THROW("Invalid buffer cache mode: ", buffer_cache_mode_str);
      }
    }
  };

  WebGpuBufferCacheConfig& buffer_cache_config = config.buffer_cache_config;
  parse_buffer_cache_mode(kStorageBufferCacheMode, buffer_cache_config.storage.mode);
  parse_buffer_cache_mode(kUniformBufferCacheMode, buffer_cache_config.uniform.mode);
  parse_buffer_cache_mode(kQueryResolveBufferCacheMode, buffer_cache_config.query_resolve.mode);
  parse_buffer_cache_mode(kDefaultBufferCacheMode, buffer_cache_config.default_entry.mode);

  // power preference
  if (std::string power_preference_str;
      config_options.TryGetConfigEntry(kPowerPreference, power_preference_str)) {
    if (power_preference_str == kPowerPreference_HighPerformance) {
      config.power_preference = static_cast<int>(WGPUPowerPreference_HighPerformance);
    } else if (power_preference_str == kPowerPreference_LowPower) {
      config.power_preference = static_cast<int>(WGPUPowerPreference_LowPower);
    } else {
      ORT_THROW("Invalid power preference: ", power_preference_str);
    }
  }

  // backend type
  if (std::string backend_type_str;
      config_options.TryGetConfigEntry(kDawnBackendType, backend_type_str)) {
    if (backend_type_str == kDawnBackendType_D3D12) {
      config.backend_type = static_cast<int>(WGPUBackendType_D3D12);
    } else if (backend_type_str == kDawnBackendType_Vulkan) {
      config.backend_type = static_cast<int>(WGPUBackendType_Vulkan);
    } else {
      ORT_THROW("Invalid Dawn backend type: ", backend_type_str);
    }
  }

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP storage buffer cache mode: " << config.buffer_cache_config.storage.mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP uniform buffer cache mode: " << config.buffer_cache_config.uniform.mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP query resolve buffer cache mode: " << config.buffer_cache_config.query_resolve.mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP default buffer cache mode: " << config.buffer_cache_config.default_entry.mode;

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP power preference: " << config.power_preference;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Dawn backend type: " << config.backend_type;

  return config;
}

}  // namespace

static std::shared_ptr<IExecutionProviderFactory> CreateWebGpuProviderFactory(
    const ConfigOptions& config_options, const WebGpuExecutionProviderTestOptions& test_options) {
  const auto weight_load_acceleration_mode =
      ResolveWeightLoadAccelerationMode(config_options);

  // prepare WebGpuExecutionProviderConfig
  WebGpuExecutionProviderConfig webgpu_ep_config =
      ParseEpConfig(config_options, weight_load_acceleration_mode);
  webgpu_ep_config.forced_matmul_algorithm = test_options.forced_matmul_algorithm;

  // prepare WebGpuContextConfig
  WebGpuContextConfig config =
      ParseWebGpuContextConfig(config_options, weight_load_acceleration_mode);
  config.test_only_max_storage_buffer_binding_size = test_options.max_storage_buffer_binding_size;

  // Load the Dawn library and create the WebGPU instance.
  auto& context = WebGpuContextFactory::CreateContext(config);

  // Create WebGPU EP factory.
  return std::make_shared<WebGpuProviderFactory>(config.context_id, context, std::move(webgpu_ep_config));
}

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(const ConfigOptions& config_options) {
  return CreateWebGpuProviderFactory(config_options, {});
}

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::CreateForTesting(
    const ConfigOptions& config_options, const WebGpuExecutionProviderTestOptions& test_options) {
  return CreateWebGpuProviderFactory(config_options, test_options);
}

// WebGPU DataTransfer implementation wrapper for the C API
struct WebGpuDataTransferImpl : OrtDataTransferImpl {
  explicit WebGpuDataTransferImpl(const OrtApi& ort_api_in)
      : ort_api{ort_api_in},
        ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;
    CanCopy = CanCopyImpl;          // OrtDataTransferImpl::CanCopy callback
    CopyTensors = CopyTensorsImpl;  // OrtDataTransferImpl::CopyTensors callback
    Release = ReleaseImpl;          // OrtDataTransferImpl::Release callback
  }

  static bool ORT_API_CALL CanCopyImpl(
      const OrtDataTransferImpl* this_ptr,
      const OrtMemoryDevice* src_memory_device,
      const OrtMemoryDevice* dst_memory_device) noexcept {
    const auto& impl = *static_cast<const WebGpuDataTransferImpl*>(this_ptr);
    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

    // Check if at least one device is GPU
    bool has_gpu = (src_type == OrtMemoryInfoDeviceType_GPU) || (dst_type == OrtMemoryInfoDeviceType_GPU);
    if (!has_gpu) {
      return false;
    }

    // WebGPU uses vendor ID 0 (VendorIds::NONE). Only handle GPU devices with vendor ID 0.
    // This prevents attempting to copy data for other EPs' fake GPU devices (e.g., example EP with vendor 0xBE57)
    if (src_type == OrtMemoryInfoDeviceType_GPU) {
      uint32_t src_vendor = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
      if (src_vendor != 0) {
        return false;  // Not a WebGPU device
      }
    }

    if (dst_type == OrtMemoryInfoDeviceType_GPU) {
      uint32_t dst_vendor = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);
      if (dst_vendor != 0) {
        return false;  // Not a WebGPU device
      }
    }

    // If both are GPU, they must have the same device ID
    if (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) {
      int src_device_id = impl.ep_api.MemoryDevice_GetDeviceId(src_memory_device);
      int dst_device_id = impl.ep_api.MemoryDevice_GetDeviceId(dst_memory_device);
      if (src_device_id != dst_device_id) {
        return false;  // Cannot copy between different devices
      }
    }

    // WebGPU supports GPU<->GPU, GPU<->CPU copies (where GPU has vendor ID 0)
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
  }

  // BufferManager::MemCpy reports misuse by throwing (ORT_ENFORCE), not by returning a Status.
  // This callback crosses the C ABI and is noexcept, so an escaping exception would terminate the
  // process. Convert throws to an OrtStatus.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(
      OrtDataTransferImpl* this_ptr,
      const OrtValue** src_tensors,
      OrtValue** dst_tensors,
      OrtSyncStream** streams,
      size_t num_tensors) noexcept {
    API_IMPL_BEGIN
    return CopyTensorsOrThrow(this_ptr, src_tensors, dst_tensors, streams, num_tensors);
    API_IMPL_END
  }

  static OrtStatus* CopyTensorsOrThrow(
      OrtDataTransferImpl* this_ptr,
      const OrtValue** src_tensors,
      OrtValue** dst_tensors,
      OrtSyncStream** streams,
      size_t num_tensors) {
    auto& impl = *static_cast<WebGpuDataTransferImpl*>(this_ptr);

    if (num_tensors == 0) {
      return nullptr;
    }

#if defined(ORT_USE_EP_API_ADAPTERS)
    // Batch unmapped inputs on the owning Session timeline, including on UMA devices.
    // Check the entire list before copying so mixed transfers retain their original ordering.
    if (num_tensors > 1 && streams != nullptr && streams[0] != nullptr) {
      InlinedVector<BufferManager::UploadInfo> uploads;
      uploads.reserve(num_tensors);
      int device_id = -1;
      for (size_t idx = 0; idx < num_tensors; ++idx) {
        if (streams[idx] != streams[0]) break;
        Ort::ConstValue src_value{src_tensors[idx]};
        Ort::UnownedValue dst_value{dst_tensors[idx]};
        auto src_info = src_value.GetTensorMemoryInfo();
        auto dst_info = dst_value.GetTensorMemoryInfo();
        if (src_info.GetDeviceType() != OrtMemoryInfoDeviceType_CPU ||
            dst_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU) break;
        const auto* src_device = impl.ep_api.MemoryInfo_GetMemoryDevice(src_info);
        const auto* dst_device = impl.ep_api.MemoryInfo_GetMemoryDevice(dst_info);
        if (!CanCopyImpl(this_ptr, src_device, dst_device)) break;
        const int dst_device_id = impl.ep_api.MemoryDevice_GetDeviceId(dst_device);
        if (idx == 0) device_id = dst_device_id;
        if (dst_device_id != device_id) break;
        const size_t size = src_value.GetTensorSizeInBytes();
        auto target = static_cast<WGPUBuffer>(dst_value.GetTensorMutableRawData());
        if (size != 0 && wgpuBufferGetMapState(target) != WGPUBufferMapState_Unmapped) break;
        uploads.push_back({src_value.GetTensorRawData(), target, size});
      }
      if (uploads.size() == num_tensors) {
        webgpu::ep::UploadTensorsOnWebGpuStream(streams[0], device_id, uploads);
        return nullptr;
      }
    }
#endif

    for (size_t idx = 0; idx < num_tensors; ++idx) {
      const OrtMemoryInfo* src_memory_info = nullptr;
      const OrtMemoryInfo* dst_memory_info = nullptr;
      ORT_THROW_IF_ERROR(ToStatusAndRelease(impl.ort_api.GetTensorMemoryInfo(src_tensors[idx], &src_memory_info)));
      ORT_THROW_IF_ERROR(ToStatusAndRelease(impl.ort_api.GetTensorMemoryInfo(dst_tensors[idx], &dst_memory_info)));
      const auto* src_device = impl.ep_api.MemoryInfo_GetMemoryDevice(src_memory_info);
      const auto* dst_device = impl.ep_api.MemoryInfo_GetMemoryDevice(dst_memory_info);
      if (!CanCopyImpl(this_ptr, src_device, dst_device)) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported memory devices for WebGPU copy.");
      }
#if defined(ORT_USE_EP_API_ADAPTERS)
      Ort::ConstValue src_value{src_tensors[idx]};
      const void* src_data = src_value.GetTensorRawData();
      size_t size = src_value.GetTensorSizeInBytes();
      bool src_is_gpu = src_value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU;

      Ort::UnownedValue dst_value{dst_tensors[idx]};
      void* dst_data = dst_value.GetTensorMutableRawData();
      bool dst_is_gpu = dst_value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
#else
      const Tensor& src_tensor = src_tensors[idx]->Get<Tensor>();
      const void* src_data = src_tensor.DataRaw();
      size_t size = src_tensor.SizeInBytes();
      bool src_is_gpu = src_tensor.Location().device.Type() == OrtDevice::GPU;

      Tensor& dst_tensor = *dst_tensors[idx]->GetMutable<Tensor>();
      void* dst_data = dst_tensor.MutableDataRaw();
      bool dst_is_gpu = dst_tensor.Location().device.Type() == OrtDevice::GPU;
#endif
      common::Status status;
#if defined(ORT_USE_EP_API_ADAPTERS)
      if (streams != nullptr && streams[idx] != nullptr) {
        status = webgpu::ep::CopyTensorOnWebGpuStream(streams[idx], src_data, src_is_gpu, dst_data, dst_is_gpu, size);
      } else
#else
      ORT_UNUSED_PARAMETER(streams);
#endif
      {
        const int context_id = impl.ep_api.MemoryDevice_GetDeviceId(src_is_gpu ? src_device : dst_device);
        WebGpuContextFactory::RetainContext(context_id);
        auto release_context = gsl::finally([context_id]() { WebGpuContextFactory::ReleaseContext(context_id); });
        auto& context = WebGpuContextFactory::GetContext(context_id);
#if defined(ORT_USE_EP_API_ADAPTERS)
        if (webgpu::ep::UseSerializedExecutionMode()) {
          // Old hosts can drop the stream on framework copies. Submit the active Session first.
          if (auto* active_recording = context.ActiveSerializedRecording()) {
            ORT_THROW_IF_ERROR(context.Flush(context.BufferManager(), *active_recording));
          }
        }
#endif
        CommandRecordingState recording;
        DataTransferImpl data_transfer{context.BufferManager(), recording};
        status = data_transfer.CopyTensor(src_data, src_is_gpu, dst_data, dst_is_gpu, size);
        if (status.IsOK() && src_is_gpu && dst_is_gpu) {
          ORT_THROW_IF_ERROR(context.Flush(context.BufferManager(), recording));
        }
      }
      if (!status.IsOK()) {
        return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, status.ErrorMessage().c_str());
      }
    }
    return nullptr;
  }

  static void ORT_API_CALL ReleaseImpl(
      OrtDataTransferImpl* this_ptr) noexcept {
    delete static_cast<WebGpuDataTransferImpl*>(this_ptr);
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

OrtDataTransferImpl* OrtWebGpuCreateDataTransfer() {
#if defined(ORT_USE_EP_API_ADAPTERS)
  return new WebGpuDataTransferImpl(onnxruntime::ep::Api().ort);
#else
  // Validate API version is supported
  const OrtApi* api = OrtApis::GetApi(ORT_API_VERSION);
  if (!api) {
    // API version not supported - return nullptr to indicate failure
    return nullptr;
  }
  return new WebGpuDataTransferImpl(*api);
#endif
}

}  // namespace onnxruntime
