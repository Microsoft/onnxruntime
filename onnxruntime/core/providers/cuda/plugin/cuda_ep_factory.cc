// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep_factory.h"
#include "cuda_device_mapping.h"
#include "cuda_ep.h"
#include "cuda_plugin_kernels.h"
#include "contrib_ops/cuda/quantization/matmul_nbits_tactic_cache.h"
#include "core/common/string_utils.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/session/onnxruntime_c_api.h"
#include <algorithm>
#include <cassert>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <cuda.h>
#endif

namespace onnxruntime {
namespace cuda_plugin {

CudaEpFactory::CudaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                             const OrtLogger& default_logger)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      default_logger_(default_logger) {
  ort_version_supported = ORT_API_VERSION;

  if (!::onnxruntime::ep::adapter::LoggingManager::HasDefaultLogger()) {
    ::onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(&default_logger);
  }

  // Assign callback function pointers
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

CudaEpFactory::~CudaEpFactory() {
  // Save any tactic caches of sessions still alive when the factory is released, before worker
  // threads are stopped or the plugin DLL is unloaded.
  onnxruntime::contrib::cuda::FlushMatMulNBitsTacticCaches();

  for (auto& [key, entry] : device_cache_) {
    static_cast<void>(key);
    for (auto& arena : entry.device_arenas) {
      if (arena.has_quarantine || arena.abandoned) {
        // Completion of quarantined work is unknown. Intentionally abandon the arena so its
        // device regions cannot be freed while the retained stream may still reference them.
        static_cast<void>(arena.allocator.release());
      }
    }
  }

  if (kernel_registry_ != nullptr) {
    ep_api_.ReleaseKernelRegistry(kernel_registry_);
  }

  for (const auto& entry : runtime_discovered_hardware_devices_) {
    ep_api_.ReleaseHardwareDevice(entry.second);
  }
}

OrtStatus* CudaEpFactory::GetKernelRegistryForEp(CudaEp& ep,
                                                 const OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  std::lock_guard<std::mutex> lock(registry_mutex_);

  if (kernel_registry_ == nullptr) {
    const char* ep_name = ep.GetEpName();
    // CreateCudaKernelRegistry dispatches between legacy/generated registrations
    // and adapter-mode registration path based on build configuration.
    RETURN_IF_ERROR(CreateCudaKernelRegistry(ep_api_, ep_name, nullptr, &kernel_registry_));
  }

  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

// ---------------------------------------------------------------------------
// OrtEpFactory callback implementations
// ---------------------------------------------------------------------------

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL CudaEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->vendor_id_;
}

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->ep_version_.c_str();
}

namespace {

// True when the options carry anything besides the EP instance identity that ORT adds to every allocator request.
bool HasCallerAllocatorOptions(const OrtApi& ort_api, const OrtKeyValuePairs* options) {
  if (options == nullptr) return false;
  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  size_t count = 0;
  ort_api.GetKeyValuePairs(options, &keys, &values, &count);
  for (size_t i = 0; i < count; ++i) {
    if (std::strcmp(keys[i], kEpInstanceIdOptionKey) != 0) return true;
  }
  return false;
}

std::string ToUpper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

std::string GetProviderOptionPrefix(std::string_view provider_name) {
  if (provider_name == kCudaExecutionProvider) {
    return "ep.cuda.";
  }

  return "ep." + onnxruntime::utils::GetLowercaseString(std::string{provider_name}) + ".";
}

void LogWarning(const OrtApi& ort_api, const OrtLogger& logger, const ORTCHAR_T* file, int line,
                const char* function, const char* msg) {
  OrtStatus* st = ort_api.Logger_LogMessage(&logger, ORT_LOGGING_LEVEL_WARNING, msg, file, line, function);
  if (st != nullptr) {
    ort_api.ReleaseStatus(st);
  }
}

bool IsCudaMempoolUnsupportedStatus(const OrtApi& ort_api, const OrtStatus* status) {
  if (status == nullptr) {
    return false;
  }

  const OrtErrorCode code = ort_api.GetErrorCode(status);
  if (code == ORT_NOT_IMPLEMENTED) {
    return true;
  }

  if (code != ORT_EP_FAIL) {
    return false;
  }

  const char* msg = ort_api.GetErrorMessage(status);
  return msg != nullptr &&
         (std::strstr(msg, "cudaErrorNotSupported") != nullptr ||
          std::strstr(msg, "operation not supported") != nullptr);
}

std::string GetCudaDeviceIdentity(int cuda_ordinal) {
#ifdef _WIN32
  CUdevice device;
  char luid[8]{};
  unsigned int node_mask = 0;
  if (cuDeviceGet(&device, cuda_ordinal) == CUDA_SUCCESS &&
      cuDeviceGetLuid(luid, &node_mask, device) == CUDA_SUCCESS) {
    uint64_t luid_value = 0;
    static_assert(sizeof(luid_value) == sizeof(luid));
    std::memcpy(&luid_value, luid, sizeof(luid_value));
    return std::to_string(luid_value);
  }
#else
  char pci_bus_id[32]{};
  if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_ordinal) == cudaSuccess) {
    return NormalizePciBusId(pci_bus_id);
  }
#endif

  return {};
}

std::string GetHardwareDeviceIdentity(const OrtApi& ort_api,
                                      const OrtHardwareDevice& device) {
  const OrtKeyValuePairs* metadata = ort_api.HardwareDevice_Metadata(&device);
  if (metadata == nullptr) {
    return {};
  }

#ifdef _WIN32
  const char* luid = ort_api.GetKeyValue(metadata, "LUID");
  return luid == nullptr ? std::string{} : std::string{luid};
#else
  const char* pci_bus_id = ort_api.GetKeyValue(metadata, "pci_bus_id");
  return pci_bus_id == nullptr ? std::string{} : NormalizePciBusId(pci_bus_id);
#endif
}

}  // namespace

CudaEpFactory::HardwareDeviceKey CudaEpFactory::MakeDeviceKey(const OrtApi& ort_api,
                                                              const OrtHardwareDevice& device,
                                                              int cuda_ordinal) {
  return {
      ort_api.HardwareDevice_Type(&device),
      ort_api.HardwareDevice_VendorId(&device),
      ort_api.HardwareDevice_DeviceId(&device),
      cuda_ordinal,
  };
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* hw_devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  // Clear stale ordinal mappings from any prior enumeration.
  {
    std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
    factory->ordinal_to_device_key_.clear();
  }

  auto release_ep_devices = [&](OrtStatus* status) -> OrtStatus* {
    for (size_t j = 0; j < num_ep_devices; ++j) {
      factory->ep_api_.ReleaseEpDevice(ep_devices[j]);
      ep_devices[j] = nullptr;
    }
    num_ep_devices = 0;
    return status;
  };

  // Query CUDA device count once upfront so we can validate assigned ordinals.
  int cuda_device_count = 0;
  cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_err != cudaSuccess) {
    cuda_device_count = 0;  // no CUDA devices available
  }

  InlinedVector<std::string> cuda_device_identities;
  InlinedVector<uint8_t> assigned_cuda_ordinals;
  cuda_device_identities.reserve(cuda_device_count);
  assigned_cuda_ordinals.reserve(cuda_device_count);
  for (int cuda_ordinal = 0; cuda_ordinal < cuda_device_count; ++cuda_ordinal) {
    cuda_device_identities.emplace_back(GetCudaDeviceIdentity(cuda_ordinal));
    assigned_cuda_ordinals.push_back(0);
  }

  auto add_ep_device = [&](const OrtHardwareDevice& device, int cuda_ordinal) -> OrtStatus* {
    const auto device_key = CudaEpFactory::MakeDeviceKey(factory->ort_api_, device, cuda_ordinal);
    DeviceCacheEntry* cache_entry = nullptr;
    {
      std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
      auto [it, inserted] = factory->device_cache_.try_emplace(device_key);
      if (inserted) {
        it->second.cuda_device_id = cuda_ordinal;
        it->second.device_memory_info = Ort::MemoryInfo{"Cuda",
                                                        OrtMemoryInfoDeviceType_GPU,
                                                        factory->vendor_id_,
                                                        static_cast<uint32_t>(cuda_ordinal),
                                                        OrtDeviceMemoryType_DEFAULT,
                                                        /*alignment is default*/ 0,
                                                        OrtAllocatorType::OrtDeviceAllocator};
        it->second.pinned_memory_info = Ort::MemoryInfo{"CudaPinned",
                                                        OrtAllocatorType::OrtDeviceAllocator,
                                                        cuda_ordinal,
                                                        OrtMemType::OrtMemTypeCPU};
      }

      cache_entry = &it->second;
      factory->ordinal_to_device_key_[cuda_ordinal] = device_key;
    }

    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
    factory->ort_api_.CreateKeyValuePairs(&ep_options);
    factory->ort_api_.AddKeyValuePair(ep_metadata, "cuda_device_id", std::to_string(cuda_ordinal).c_str());
    factory->ort_api_.AddKeyValuePair(ep_options, "device_id", std::to_string(cuda_ordinal).c_str());

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, cuda_ordinal) == cudaSuccess) {
      factory->ort_api_.AddKeyValuePair(ep_metadata, "cuda_device_name", prop.name);
      factory->ort_api_.AddKeyValuePair(
          ep_metadata, "cuda_compute_capability",
          (std::to_string(prop.major) + "." + std::to_string(prop.minor)).c_str());
    }

    OrtEpDevice* ep_device = nullptr;
    auto* status = factory->ep_api_.CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                   &ep_device);
    factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
    factory->ort_api_.ReleaseKeyValuePairs(ep_options);

    if (status != nullptr) {
      return status;
    }

    auto release_current_ep_device = [factory](OrtEpDevice* device_to_release) {
      factory->ep_api_.ReleaseEpDevice(device_to_release);
    };
    std::unique_ptr<OrtEpDevice, decltype(release_current_ep_device)> ep_device_guard(
        ep_device, release_current_ep_device);

    status = factory->ep_api_.EpDevice_AddAllocatorInfo(ep_device, cache_entry->device_memory_info);
    if (status != nullptr) {
      return status;
    }

    status = factory->ep_api_.EpDevice_AddAllocatorInfo(ep_device, cache_entry->pinned_memory_info);
    if (status != nullptr) {
      return status;
    }

    ep_devices[num_ep_devices++] = ep_device_guard.release();
    return nullptr;
  };

  InlinedVector<const OrtHardwareDevice*> hardware_devices_without_identity;
  hardware_devices_without_identity.reserve(num_devices);

  // Reserve all exact hardware identity matches before considering devices
  // without identity, so an unknown device cannot consume a later exact match.
  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *hw_devices[i];
    auto hw_type = factory->ort_api_.HardwareDevice_Type(&device);

    if (hw_type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // Filter by vendor ID to avoid claiming non-NVIDIA GPUs on mixed-vendor hosts.
      // vendor_id == 0 means the hardware enumeration did not provide a vendor ID,
      // in which case we fall through and let the CUDA runtime validate the device.
      uint32_t hw_vendor_id = factory->ort_api_.HardwareDevice_VendorId(&device);
      if (hw_vendor_id != 0 && hw_vendor_id != factory->vendor_id_) {
        continue;  // Skip non-NVIDIA GPUs
      }

      const std::string hardware_device_identity =
          GetHardwareDeviceIdentity(factory->ort_api_, device);
      if (hardware_device_identity.empty()) {
        hardware_devices_without_identity.push_back(&device);
        continue;
      }

      auto cuda_ordinal = FindCudaOrdinalForHardwareDeviceIdentity(
          hardware_device_identity, cuda_device_identities, assigned_cuda_ordinals);
      if (!cuda_ordinal.has_value()) {
        continue;
      }

      assigned_cuda_ordinals[*cuda_ordinal] = 1;
      if (auto* status = add_ep_device(device, *cuda_ordinal); status != nullptr) {
        return release_ep_devices(status);
      }
    }
  }

  // Preserve the previous positional behavior only when both sides lack a
  // platform identity. Never assign an unidentified hardware device to a CUDA
  // ordinal with a known identity.
  for (const OrtHardwareDevice* device : hardware_devices_without_identity) {
    if (num_ep_devices >= max_ep_devices) {
      break;
    }

    auto cuda_ordinal =
        FindCudaOrdinalWithoutIdentity(cuda_device_identities, assigned_cuda_ordinals);
    if (!cuda_ordinal.has_value()) {
      continue;
    }

    assigned_cuda_ordinals[*cuda_ordinal] = 1;
    if (auto* status = add_ep_device(*device, *cuda_ordinal); status != nullptr) {
      return release_ep_devices(status);
    }
  }

  // Platform discovery may not expose every CUDA-visible device. In particular, WSL
  // provides CUDA through /dev/dxg but sysfs reports Microsoft synthetic adapters,
  // which the NVIDIA factory must not claim. Create descriptors for any remaining
  // CUDA ordinals using the CUDA runtime as the authoritative device source.
  for (int cuda_ordinal = 0;
       cuda_ordinal < cuda_device_count && num_ep_devices < max_ep_devices;
       ++cuda_ordinal) {
    if (assigned_cuda_ordinals[cuda_ordinal] != 0) {
      continue;
    }

    OrtHardwareDevice* runtime_device = nullptr;
    {
      std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
      auto it = factory->runtime_discovered_hardware_devices_.find(cuda_ordinal);
      if (it != factory->runtime_discovered_hardware_devices_.end()) {
        runtime_device = it->second;
      }
    }

    if (runtime_device == nullptr) {
      OrtKeyValuePairs* hw_metadata = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&hw_metadata);
      factory->ort_api_.AddKeyValuePair(hw_metadata, "cuda_runtime_discovered", "1");

      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, cuda_ordinal) == cudaSuccess) {
        factory->ort_api_.AddKeyValuePair(hw_metadata, "Discrete",
                                          prop.integrated == 0 ? "1" : "0");
      }

      if (!cuda_device_identities[cuda_ordinal].empty()) {
#ifdef _WIN32
        factory->ort_api_.AddKeyValuePair(hw_metadata, "LUID",
                                          cuda_device_identities[cuda_ordinal].c_str());
#else
        factory->ort_api_.AddKeyValuePair(hw_metadata, "pci_bus_id",
                                          cuda_device_identities[cuda_ordinal].c_str());
#endif
      }

      auto* status = factory->ep_api_.CreateHardwareDevice(
          OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
          factory->vendor_id_,
          /*device_id*/ 0,
          factory->vendor_.c_str(),
          hw_metadata,
          &runtime_device);
      factory->ort_api_.ReleaseKeyValuePairs(hw_metadata);
      if (status != nullptr) {
        return release_ep_devices(status);
      }

      {
        std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
        auto [it, inserted] = factory->runtime_discovered_hardware_devices_.emplace(cuda_ordinal, runtime_device);
        if (!inserted) {
          factory->ep_api_.ReleaseHardwareDevice(runtime_device);
          runtime_device = it->second;
        }
      }
    }

    if (auto* status = add_ep_device(*runtime_device, cuda_ordinal); status != nullptr) {
      return release_ep_devices(status);
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA EP factory currently supports exactly one device per EP instance. "
        "Pass a single OrtHardwareDevice when creating the CUDA plugin EP.");
  }
  if (devices == nullptr || devices[0] == nullptr) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA EP factory requires a valid device.");
  }

  // Parse configuration from session options.
  // The read helpers intentionally swallow errors: if a config entry is
  // absent or malformed the default value in Config is kept.
  CudaEp::Config config{};

  {
    // Resolve the CUDA ordinal from ep_metadata (set during GetSupportedDevicesImpl).
    int cuda_ordinal = -1;
    if (!ep_metadata || !ep_metadata[0]) {
      return factory->ort_api_.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "CUDA EP factory requires ep_metadata with a 'cuda_device_id' entry. "
          "Ensure GetSupportedDevices has been called and its ep_metadata is forwarded.");
    }

    {
      const char* ordinal_str = factory->ort_api_.GetKeyValue(ep_metadata[0], "cuda_device_id");
      if (!ordinal_str) {
        return factory->ort_api_.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Missing 'cuda_device_id' in ep_metadata. "
            "Ensure GetSupportedDevices has been called and its ep_metadata is forwarded.");
      }
      char* end = nullptr;
      long parsed = std::strtol(ordinal_str, &end, 10);
      if (end == ordinal_str || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return factory->ort_api_.CreateStatus(
            ORT_INVALID_ARGUMENT,
            (std::string("Invalid cuda_device_id in ep_metadata: '") + ordinal_str + "'").c_str());
      }
      cuda_ordinal = static_cast<int>(parsed);
    }

    std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
    auto* entry = factory->FindDeviceCacheEntryByOrdinalLocked(cuda_ordinal);
    if (!entry) {
      return factory->ort_api_.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "CUDA EP factory could not resolve the requested device. "
          "Enumerate EP devices again and retry session creation.");
    }
    config.device_id = entry->cuda_device_id;
  }

  auto try_get_session_config = [&](std::string_view key) -> std::optional<std::string> {
    if (session_options == nullptr) {
      return std::nullopt;
    }

    size_t size = 0;
    OrtStatus* status = factory->ort_api_.GetSessionConfigEntry(session_options, key.data(), nullptr, &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return std::nullopt;
    }
    if (size == 0) {
      return std::nullopt;
    }
    std::vector<char> buf(size);
    status = factory->ort_api_.GetSessionConfigEntry(session_options, key.data(), buf.data(), &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return std::nullopt;
    }
    return std::string(buf.data());
  };

  auto log_invalid_session_config = [&](std::string_view key, std::string_view expected) {
    if (logger == nullptr) {
      return;
    }

    const std::string msg = std::string("Failed to parse session config for key '") +
                            std::string(key) + "'. Expected " + std::string(expected) +
                            ". Using default value.";

    OrtStatus* st = factory->ort_api_.Logger_LogMessage(
        logger, ORT_LOGGING_LEVEL_WARNING, msg.c_str(), ORT_FILE, __LINE__, "CudaEpFactory");
    if (st != nullptr) {
      factory->ort_api_.ReleaseStatus(st);
    }
  };

  auto read_session_config_bool = [&](std::initializer_list<std::string_view> keys, bool& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      const auto normalized = ToUpper(*raw_value);
      if (normalized == "1" || normalized == "TRUE") {
        value = true;
        return;
      }
      if (normalized == "0" || normalized == "FALSE") {
        value = false;
        return;
      }

      log_invalid_session_config(key, "a boolean");
      return;
    }
  };

  auto read_cudnn_conv_algo = [&](std::initializer_list<std::string_view> keys, int& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      ORT_TRY {
        value = std::stoi(*raw_value);
        return;
      }
      ORT_CATCH(const std::exception&) {
      }

      const auto normalized = ToUpper(*raw_value);
      if (normalized == "EXHAUSTIVE") {
        value = 0;
        return;
      }
      if (normalized == "HEURISTIC") {
        value = 1;
        return;
      }
      if (normalized == "DEFAULT") {
        value = 2;
        return;
      }

      log_invalid_session_config(key, "an integer or one of EXHAUSTIVE/HEURISTIC/DEFAULT");
      return;
    }
  };

  auto read_session_config_non_negative_int = [&](std::initializer_list<std::string_view> keys, int& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      ORT_TRY {
        int parsed = std::stoi(*raw_value);
        if (parsed < 0) {
          log_invalid_session_config(key, "a non-negative integer");
          return;
        }

        value = parsed;
        return;
      }
      ORT_CATCH(const std::exception&) {
      }

      log_invalid_session_config(key, "a non-negative integer");
      return;
    }
  };

  const std::string ep_options_prefix = GetProviderOptionPrefix(factory->GetEpName());
  const std::string prefer_nhwc_key = ep_options_prefix + "prefer_nhwc";
  const std::string prefer_nhwc_layout_key = ep_options_prefix + "prefer_nhwc_layout";
  const std::string use_tf32_key = ep_options_prefix + "use_tf32";
  const std::string cudnn_use_max_workspace_key = ep_options_prefix + "cudnn_conv_use_max_workspace";
  const std::string cudnn_conv1d_pad_key = ep_options_prefix + "cudnn_conv1d_pad_to_nc1d";
  const std::string cudnn_conv_algo_key = ep_options_prefix + "cudnn_conv_algo";
  const std::string cudnn_conv_algo_search_key = ep_options_prefix + "cudnn_conv_algo_search";
  const std::string enable_cudnn_key = ep_options_prefix + "enable_cudnn";
  const std::string fuse_conv_bias_key = ep_options_prefix + "fuse_conv_bias";
  const std::string sdpa_kernel_key = ep_options_prefix + "sdpa_kernel";
  const std::string enable_cuda_graph_key = ep_options_prefix + "enable_cuda_graph";
  const std::string min_runs_key = ep_options_prefix + "min_num_runs_before_cuda_graph_capture";
  const std::string has_user_compute_stream_key = ep_options_prefix + "has_user_compute_stream";
  const std::string user_compute_stream_key = ep_options_prefix + "user_compute_stream";
  const std::string do_copy_in_default_stream_key = ep_options_prefix + "do_copy_in_default_stream";
  const std::string use_ep_level_unified_stream_key = ep_options_prefix + "use_ep_level_unified_stream";
  const std::string gpu_external_alloc_key = ep_options_prefix + "gpu_external_alloc";
  const std::string gpu_external_free_key = ep_options_prefix + "gpu_external_free";
  const std::string gpu_external_empty_cache_key = ep_options_prefix + "gpu_external_empty_cache";
  const auto device_mem_limit = try_get_session_config(ep_options_prefix + "gpu_device_mem_limit");
  size_t device_limit_bytes = 0;
  if (device_mem_limit) {
    const auto parsed = std::from_chars(device_mem_limit->data(),
                                        device_mem_limit->data() + device_mem_limit->size(),
                                        device_limit_bytes);
    if (parsed.ec != std::errc{} || parsed.ptr != device_mem_limit->data() + device_mem_limit->size()) {
      return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "gpu_device_mem_limit must be a nonnegative byte count");
    }
  }

  // Prefer canonical EP-scoped keys, then fall back to historical flat session config names.
  read_session_config_bool(
      {prefer_nhwc_key, prefer_nhwc_layout_key, "ep.cuda.prefer_nhwc_layout", "prefer_nhwc", "prefer_nhwc_layout"},
      config.prefer_nhwc);
  read_session_config_bool({use_tf32_key, "ep.cuda.use_tf32", "use_tf32"}, config.use_tf32);
  read_session_config_bool(
      {cudnn_use_max_workspace_key, "ep.cuda.cudnn_conv_use_max_workspace", "cudnn_conv_use_max_workspace"},
      config.cudnn_conv_use_max_workspace);
  read_session_config_bool(
      {cudnn_conv1d_pad_key, "ep.cuda.cudnn_conv1d_pad_to_nc1d", "cudnn_conv1d_pad_to_nc1d"},
      config.cudnn_conv1d_pad_to_nc1d);
  read_cudnn_conv_algo(
      {cudnn_conv_algo_search_key, cudnn_conv_algo_key, "ep.cuda.cudnn_conv_algo_search", "ep.cuda.cudnn_conv_algo",
       "cudnn_conv_algo_search", "cudnn_conv_algo"},
      config.cudnn_conv_algo);
  read_session_config_bool(
      {enable_cudnn_key, "ep.cuda.enable_cudnn", "enable_cudnn"},
      config.enable_cudnn);
  read_session_config_bool(
      {fuse_conv_bias_key, "ep.cuda.fuse_conv_bias", "fuse_conv_bias"},
      config.fuse_conv_bias);
  read_session_config_non_negative_int(
      {sdpa_kernel_key, "ep.cuda.sdpa_kernel", "sdpa_kernel"},
      config.sdpa_kernel);
  read_session_config_bool(
      {enable_cuda_graph_key, "ep.cuda.enable_cuda_graph", "enable_cuda_graph"},
      config.enable_cuda_graph);
  read_session_config_non_negative_int(
      {min_runs_key, "ep.cuda.min_num_runs_before_cuda_graph_capture"},
      config.min_num_runs_before_cuda_graph_capture);

  const bool use_env_allocators =
      try_get_session_config("session.use_env_allocators").value_or("0") == "1";
  if (config.enable_cuda_graph && use_env_allocators) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA graph capture is incompatible with session.use_env_allocators=1 because captured graphs require a "
        "session-scoped device arena.");
  }

  // --- Stream and allocator options ---
  read_session_config_bool(
      {has_user_compute_stream_key, "ep.cuda.has_user_compute_stream", "has_user_compute_stream"},
      config.has_user_compute_stream);
  read_session_config_bool(
      {do_copy_in_default_stream_key, "ep.cuda.do_copy_in_default_stream", "do_copy_in_default_stream"},
      config.do_copy_in_default_stream);
  read_session_config_bool(
      {use_ep_level_unified_stream_key, "ep.cuda.use_ep_level_unified_stream", "use_ep_level_unified_stream"},
      config.use_ep_level_unified_stream);

  // Parse user_compute_stream as a pointer-sized integer (address of a cudaStream_t).
  // Uses base 0 so that "0x..." hex strings are auto-detected, and validates that
  // the entire string was consumed (matching the bundled EP's ParseStringWithClassicLocale behavior).
  auto read_session_config_pointer = [&](std::initializer_list<std::string_view> keys, void*& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      ORT_TRY {
        size_t pos = 0;
        unsigned long long address = std::stoull(*raw_value, &pos, 0);
        if (pos == raw_value->size()) {
          if (address > std::numeric_limits<std::uintptr_t>::max()) {
            log_invalid_session_config(key, "a pointer-sized integer (value exceeds address space)");
            return;
          }
          value = reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
          return;
        }
      }
      ORT_CATCH(const std::exception&) {
      }

      log_invalid_session_config(key, "a pointer-sized integer (decimal or 0x-prefixed hex address)");
      return;
    }
  };

  read_session_config_pointer(
      {user_compute_stream_key, "ep.cuda.user_compute_stream", "user_compute_stream"},
      config.user_compute_stream);

  // If user_compute_stream is provided, force has_user_compute_stream to true.
  if (config.user_compute_stream != nullptr) {
    config.has_user_compute_stream = true;
  }

  // Parse external allocator function pointers.
  read_session_config_pointer(
      {gpu_external_alloc_key, "ep.cuda.gpu_external_alloc", "gpu_external_alloc"},
      config.external_alloc);
  read_session_config_pointer(
      {gpu_external_free_key, "ep.cuda.gpu_external_free", "gpu_external_free"},
      config.external_free);
  read_session_config_pointer(
      {gpu_external_empty_cache_key, "ep.cuda.gpu_external_empty_cache", "gpu_external_empty_cache"},
      config.external_empty_cache);

  // Warn if only one of alloc/free is provided (both are required for external allocator).
  if ((config.external_alloc == nullptr) != (config.external_free == nullptr)) {
    LogWarning(factory->ort_api_, factory->default_logger_, ORT_FILE, __LINE__, "CudaEpFactory::CreateEpImpl",
               "Only one of gpu_external_alloc/gpu_external_free is set. "
               "Both must be provided for the external allocator to be used. Ignoring.");
    config.external_alloc = nullptr;
    config.external_free = nullptr;
    config.external_empty_cache = nullptr;
  }

  // Validate: user_compute_stream and external allocator cannot both be active.
  if (config.has_user_compute_stream && config.external_alloc != nullptr && config.external_free != nullptr) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA plugin EP does not support using both user_compute_stream and external allocator simultaneously.");
  }

  // user_compute_stream and enable_cuda_graph CAN be combined: when both are set, CUDA graph
  // capture/replay runs on the user-provided stream (the same stream kernels are issued to),
  // matching the bundled CUDA EP behavior. See CudaEp::GetPerThreadContext.

  // When user_compute_stream is set, force unified stream mode (matches bundled EP behavior).
  if (config.has_user_compute_stream) {
    config.use_ep_level_unified_stream = true;
  }

  // When external allocator is used, force unified stream mode (matches bundled EP behavior).
  if (config.external_alloc != nullptr && config.external_free != nullptr) {
    config.use_ep_level_unified_stream = true;
  }

  const OrtLogger& ep_logger = logger ? *logger : factory->default_logger_;
  const bool has_session_budget =
      (device_limit_bytes > 0 && device_limit_bytes != std::numeric_limits<size_t>::max());
  if (has_session_budget && (config.external_alloc != nullptr || config.external_free != nullptr)) {
    return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "gpu_device_mem_limit does not support external CUDA allocators");
  }
  const auto budget = GetCudaDeviceMemoryBudget(static_cast<OrtDevice::DeviceId>(config.device_id));
  config.device_mem_limit = device_limit_bytes;
  config.budget_account = std::make_shared<SessionBudgetAccount>(
      budget, has_session_budget ? device_limit_bytes : std::numeric_limits<size_t>::max());
  if (auto group_opt = try_get_session_config("session.moe.offload.budget_group");
      group_opt.has_value() && !group_opt->empty()) {
    config.budget_account->SetBudgetGroup(GetCudaBudgetGroupByName(
        config.device_id, *group_opt, has_session_budget ? device_limit_bytes : 0));
  }
  auto actual_ep = std::make_unique<CudaEp>(*factory, config, ep_logger);
  if (DeviceCacheEntry* entry = factory->FindDeviceCacheEntryByOrdinal(config.device_id)) {
    std::lock_guard<std::mutex> lock(entry->arena_mutex);
    entry->pending_session_accounts[static_cast<const OrtEp*>(actual_ep.get())] = config.budget_account;
  }
  *ep = actual_ep.release();

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
void ORT_API_CALL CudaEpFactory::ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept {
  static_cast<CudaEpFactory*>(this_ptr)->ForgetPendingSessionAccount(ep);
  delete static_cast<CudaEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options,
    OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  *allocator = nullptr;

  const char* name = "";
  OrtStatus* status = factory.ort_api_.MemoryInfoGetName(memory_info, &name);
  if (status != nullptr) {
    return status;
  }
  int req_device_id = 0;
  status = factory.ort_api_.MemoryInfoGetId(memory_info, &req_device_id);
  if (status != nullptr) {
    return status;
  }

  if (name != nullptr && strcmp(name, "Cuda") == 0) {
    // The returned pointer is safe to use after the cache mutex is released because
    // device_cache_ is std::unordered_map (node-based) and entries are never erased.
    DeviceCacheEntry* entry = factory.FindDeviceCacheEntryByOrdinal(req_device_id);
    if (!entry) {
      return factory.ort_api_.CreateStatus(
          ORT_INVALID_ARGUMENT,
          ("CUDA EP factory has no registered device for ordinal " +
           std::to_string(req_device_id))
              .c_str());
    }

    // Check if the caller requested CUDA native mempool instead of the BFC arena.
    bool use_mempool = false;
    bool has_explicit_limit_opt = false;
    size_t explicit_session_limit = std::numeric_limits<size_t>::max();
    const OrtEp* owning_ep = nullptr;
    const bool has_caller_options = HasCallerAllocatorOptions(factory.ort_api_, allocator_options);
    if (allocator_options) {
      if (const char* ep_id = factory.ort_api_.GetKeyValue(allocator_options, kEpInstanceIdOptionKey);
          ep_id != nullptr && ep_id[0] != '\0') {
        try {
          owning_ep = reinterpret_cast<const OrtEp*>(static_cast<uintptr_t>(std::stoull(std::string(ep_id))));
        } catch (...) {
          owning_ep = nullptr;
        }
      }
      const char* v = factory.ort_api_.GetKeyValue(
          allocator_options, CudaMempoolOrtAllocator::ConfigKeyNames::UseCudaMempool);
      use_mempool = (v != nullptr && std::string(v) == "1");
      if (const char* lim_v = factory.ort_api_.GetKeyValue(allocator_options, "gpu_device_mem_limit");
          lim_v != nullptr && lim_v[0] != '\0') {
        has_explicit_limit_opt = true;
        try {
          const uint64_t parsed = std::stoull(std::string(lim_v));
          explicit_session_limit = (parsed > 0) ? static_cast<size_t>(parsed) : std::numeric_limits<size_t>::max();
        } catch (...) {
          explicit_session_limit = std::numeric_limits<size_t>::max();
        }
      }
    }

    std::lock_guard<std::mutex> lock{entry->arena_mutex};

    // The account is matched by the identity of the owning OrtEp, never by limit or queue order: two sessions with
    // equal limits would otherwise be able to swap accounts and charge each other's budget group.
    std::shared_ptr<SessionBudgetAccount> session_account;
    if (owning_ep != nullptr) {
      if (auto it = entry->pending_session_accounts.find(owning_ep); it != entry->pending_session_accounts.end()) {
        session_account = std::move(it->second);
        entry->pending_session_accounts.erase(it);
      }
    }
    if (!session_account && has_explicit_limit_opt) {
      session_account = std::make_shared<SessionBudgetAccount>(
          GetCudaDeviceMemoryBudget(static_cast<OrtDevice::DeviceId>(req_device_id)), explicit_session_limit);
    }
    if (!session_account) {
      const auto budget = GetCudaDeviceMemoryBudget(static_cast<OrtDevice::DeviceId>(req_device_id));
      const size_t pending_limit = (!has_caller_options && budget)
                                       ? budget->ConsumePendingSessionLimit()
                                       : std::numeric_limits<size_t>::max();
      session_account = std::make_shared<SessionBudgetAccount>(budget, pending_limit);
    }

    if (use_mempool && session_account->GetAccountStats().limit != std::numeric_limits<size_t>::max()) {
      return factory.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "gpu_device_mem_limit does not support CUDA memory pools");
    }

    if (use_mempool) {
      if (!entry->mempool_allocator) {
        status = CudaMempoolOrtAllocator::Create(memory_info, allocator_options,
                                                 factory.ort_api_, factory.default_logger_,
                                                 entry->mempool_allocator);
        if (status != nullptr) {
          if (!IsCudaMempoolUnsupportedStatus(factory.ort_api_, status)) {
            return status;
          }

          LogWarning(factory.ort_api_, factory.default_logger_, ORT_FILE, __LINE__, __FUNCTION__,
                     "CUDA mempool requested but not supported on this device/driver. Falling back to default BFCArena with CUDA allocator.");
          factory.ort_api_.ReleaseStatus(status);
          status = nullptr;
          use_mempool = false;
        }
      }

      if (use_mempool) {
        ++entry->num_mempool_users;
        *allocator = entry->mempool_allocator.get();
        return nullptr;
      }
    }

    const SessionBudgetAccount* arena_owner = session_account.get();
    AllocatorUniquePtr raw_allocator(
        new CudaDeviceAllocator(memory_info, req_device_id, factory.ort_api_, std::move(session_account)),
        [](OrtAllocator* p) { delete static_cast<CudaDeviceAllocator*>(p); });
    std::unique_ptr<CudaArenaAllocator> arena;
    status = CudaArenaAllocator::Create(CudaAllocatorKind::kDevice, memory_info,
                                        std::move(raw_allocator), allocator_options,
                                        factory.ort_api_, factory.default_logger_, arena);
    if (status != nullptr) return status;
    entry->device_arenas.push_back(DeviceArena{std::move(arena), false, false, arena_owner, false});
    *allocator = entry->device_arenas.back().allocator.get();
    return nullptr;
  }

  if (name != nullptr && strcmp(name, "CudaPinned") == 0) {
    // Pinned memory is CPU-side; find the cache entry for the device it's associated with.
    // Pointer stability: same guarantee as the Cuda branch above.
    DeviceCacheEntry* entry = factory.FindDeviceCacheEntryByOrdinal(req_device_id);
    if (!entry) {
      // Fallback: if no device cache entry (shouldn't normally happen), create raw allocator.
      auto pinned_allocator = std::make_unique<CudaPinnedAllocator>(memory_info);
      *allocator = pinned_allocator.release();
      return nullptr;
    }

    std::lock_guard<std::mutex> lock{entry->arena_mutex};

    if (!entry->pinned_arena) {
      AllocatorUniquePtr raw_allocator(
          new CudaPinnedAllocator(memory_info),
          [](OrtAllocator* p) { delete static_cast<CudaPinnedAllocator*>(p); });
      status = CudaArenaAllocator::Create(CudaAllocatorKind::kPinned, memory_info,
                                          std::move(raw_allocator), allocator_options,
                                          factory.ort_api_, factory.default_logger_,
                                          entry->pinned_arena);
      if (status != nullptr) return status;
    } else if (HasCallerAllocatorOptions(factory.ort_api_, allocator_options)) {
      LogWarning(factory.ort_api_, factory.default_logger_, ORT_FILE, __LINE__, __FUNCTION__,
                 "CUDA pinned arena already exists; session arena options are ignored.");
    }
    ++entry->num_pinned_arena_users;
    *allocator = entry->pinned_arena.get();
    return nullptr;
  }

  return factory.ort_api_.CreateStatus(
      ORT_INVALID_ARGUMENT,
      "Unknown memory info provided to CUDA EP CreateAllocator.");

  EXCEPTION_TO_STATUS_END
}

/*static*/
void ORT_API_CALL CudaEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  if (!allocator) return;
  auto* factory = static_cast<CudaEpFactory*>(this_ptr);

  // Check if allocator is a shared arena or mempool (pointer identity match).
  // Lock ordering: device_cache_mutex_ must always be acquired BEFORE any entry.arena_mutex.
  {
    std::lock_guard<std::mutex> cache_lock(factory->device_cache_mutex_);
    for (auto& [key, entry] : factory->device_cache_) {
      std::lock_guard<std::mutex> lock{entry.arena_mutex};
      auto device_arena = std::find_if(entry.device_arenas.begin(), entry.device_arenas.end(),
                                       [allocator](const DeviceArena& arena) {
                                         return arena.allocator.get() == allocator;
                                       });
      if (device_arena != entry.device_arenas.end()) {
        if (device_arena->has_quarantine || device_arena->abandoned) {
          static_cast<void>(device_arena->allocator.release());
        }
        entry.device_arenas.erase(device_arena);
        return;
      }
      if (allocator == entry.pinned_arena.get()) {
        if (entry.num_pinned_arena_users <= 0) {
          LogWarning(factory->ort_api_, factory->default_logger_, ORT_FILE, __LINE__,
                     "CudaEpFactory::ReleaseAllocatorImpl",
                     "Refcount underflow in ReleaseAllocatorImpl (pinned_arena). Ignoring release.");
          return;
        }
        if (--entry.num_pinned_arena_users == 0) entry.pinned_arena.reset();
        return;
      }
      if (allocator == entry.mempool_allocator.get()) {
        if (entry.num_mempool_users <= 0) {
          LogWarning(factory->ort_api_, factory->default_logger_, ORT_FILE, __LINE__,
                     "CudaEpFactory::ReleaseAllocatorImpl",
                     "Refcount underflow in ReleaseAllocatorImpl (mempool). Ignoring release.");
          return;
        }
        if (--entry.num_mempool_users == 0) entry.mempool_allocator.reset();
        return;
      }
    }
  }

  // Fallback: raw allocator not managed by arena (e.g. read-only allocator).
  auto* typed_allocator = static_cast<CudaAllocatorBase*>(allocator);
  switch (typed_allocator->GetKind()) {
    case CudaAllocatorKind::kDevice:
      if (typed_allocator->IsExternalDeviceAllocator()) {
        delete static_cast<CudaExternalDeviceAllocator*>(allocator);
        return;
      }
      delete static_cast<CudaDeviceAllocator*>(allocator);
      return;
    case CudaAllocatorKind::kPinned:
      delete static_cast<CudaPinnedAllocator*>(allocator);
      return;
    default:
      LogWarning(factory->ort_api_, factory->default_logger_, ORT_FILE, __LINE__,
                 "CudaEpFactory::ReleaseAllocatorImpl",
                 "ReleaseAllocatorImpl received an unknown CudaAllocatorKind. Leaking the allocator instance.");
      assert(false && "Unknown CudaAllocatorKind");
      return;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  auto data_transfer_impl = std::make_unique<CudaDataTransfer>(factory.ort_api_, factory.ep_api_);
  *data_transfer = data_transfer_impl.release();
  return nullptr;
}

/*static*/
bool ORT_API_CALL CudaEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // CUDA EP is stream-aware
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  int req_device_id = factory->ep_api_.MemoryDevice_GetDeviceId(memory_device);
  // Factory-level streams are not tied to a specific EP instance's enable_cudnn policy. Default cuDNN
  // off here so this path never triggers a cuDNN load or handle creation; kernels that need cuDNN run
  // on EP-owned streams created with the EP's actual enable_cudnn setting, and otherwise fall back to
  // the per-thread default cuDNN handle.
  auto cuda_stream = std::make_unique<CudaSyncStream>(*factory, req_device_id, false, nullptr);

  // Initialize CUDA handles (stream, cuBLAS, cuDNN)
  RETURN_IF_ERROR(cuda_stream->InitHandles());

  *stream = cuda_stream.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

CudaEpFactory::DeviceCacheEntry* CudaEpFactory::FindDeviceCacheEntryByOrdinalLocked(int cuda_ordinal) {
  auto key_it = ordinal_to_device_key_.find(cuda_ordinal);
  if (key_it == ordinal_to_device_key_.end()) {
    return nullptr;
  }
  auto cache_it = device_cache_.find(key_it->second);
  if (cache_it == device_cache_.end()) {
    return nullptr;
  }
  return &cache_it->second;
}

// IMPORTANT: Entries are never erased from device_cache_ after insertion.
// This guarantees pointer stability for DeviceCacheEntry* returned by
// FindDeviceCacheEntryByOrdinal() after the lock is released.
CudaEpFactory::DeviceCacheEntry* CudaEpFactory::FindDeviceCacheEntryByOrdinal(int cuda_ordinal) {
  std::lock_guard<std::mutex> lock(device_cache_mutex_);
  return FindDeviceCacheEntryByOrdinalLocked(cuda_ordinal);
}

OrtStatus* CudaEpFactory::ResetDeviceArenaChunksUsingStream(int device_id,
                                                            const OrtSyncStreamImpl* stream_impl) {
  OrtStatus* status = nullptr;
  ORT_TRY {
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinal(device_id);
    if (!entry) return nullptr;
    std::lock_guard<std::mutex> lock{entry->arena_mutex};
    for (auto& arena : entry->device_arenas) {
      status = arena.allocator->ResetChunksUsingStream(stream_impl);
      if (status != nullptr) return status;
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
  }
  ORT_CATCH(...) {
    status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, "ResetDeviceArenaChunksUsingStream failed.");
  }
  return status;
}

OrtStatus* CudaEpFactory::QuarantineDeviceArenaChunksUsingStream(
    int device_id, const OrtSyncStreamImpl* stream_impl) {
  OrtStatus* status = nullptr;
  ORT_TRY {
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinal(device_id);
    if (!entry) return nullptr;
    std::lock_guard<std::mutex> lock{entry->arena_mutex};
    for (auto& arena : entry->device_arenas) {
      bool quarantined = false;
      status = arena.allocator->QuarantineChunksUsingStream(stream_impl, quarantined);
      if (status != nullptr) return status;
      arena.has_quarantine = arena.has_quarantine || quarantined;
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
  }
  ORT_CATCH(...) {
    status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, "QuarantineDeviceArenaChunksUsingStream failed.");
  }
  return status;
}

OrtStatus* CudaEpFactory::QuarantineAndAbandonDeviceArena(
    int device_id, const OrtSyncStreamImpl* stream_impl) noexcept {
  OrtStatus* status = nullptr;
  ORT_TRY {
    std::lock_guard<std::mutex> cache_lock(device_cache_mutex_);
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinalLocked(device_id);
    if (!entry) return nullptr;
    std::lock_guard<std::mutex> arena_lock{entry->arena_mutex};

    for (auto& arena : entry->device_arenas) {
      arena.allocator->Abandon();
      arena.abandoned = true;
    }

    for (auto& arena : entry->device_arenas) {
      bool quarantined = false;
      status = arena.allocator->QuarantineChunksUsingStream(stream_impl, quarantined);
      arena.has_quarantine = arena.has_quarantine || quarantined;
      if (status != nullptr) return status;
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
  }
  ORT_CATCH(...) {
    status = ort_api_.CreateStatus(ORT_RUNTIME_EXCEPTION, "QuarantineAndAbandonDeviceArena failed.");
  }
  return status;
}

void CudaEpFactory::AbandonDeviceArena(int device_id) noexcept {
  ORT_TRY {
    std::lock_guard<std::mutex> cache_lock(device_cache_mutex_);
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinalLocked(device_id);
    if (!entry) return;
    std::lock_guard<std::mutex> arena_lock{entry->arena_mutex};
    for (auto& arena : entry->device_arenas) {
      arena.allocator->Abandon();
      arena.abandoned = true;
    }
  }
  ORT_CATCH(...) {
    // Last-resort path from a noexcept release callback. There is no safe way to
    // free an arena whose stream completion and chunk detachment are both unknown.
  }
}

void CudaEpFactory::ShrinkEligibleDeviceArenas(int device_id, const SessionBudgetAccount* owner) noexcept {
  ORT_TRY {
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinal(device_id);
    if (!entry || owner == nullptr) return;
    bool shrunk = false;
    {
      std::lock_guard<std::mutex> arena_lock{entry->arena_mutex};
      for (auto& arena : entry->device_arenas) {
        if (arena.owner == owner && !arena.abandoned && !arena.has_quarantine && !arena.graph_capture_started &&
            arena.allocator && arena.allocator->IsShrinkEnabled()) {
          OrtStatus* st = arena.allocator->Shrink(arena.allocator.get());
          if (st != nullptr) {
            ort_api_.ReleaseStatus(st);
          }
          shrunk = true;
        }
      }
    }
    if (shrunk) {
      ::onnxruntime::cuda::NotifyQMoEDeviceArenasShrunk(device_id);
    }
  }
  ORT_CATCH(...) {
  }
}

void CudaEpFactory::MarkSessionGraphCapture(int device_id, const SessionBudgetAccount* owner) noexcept {
  ORT_TRY {
    DeviceCacheEntry* entry = FindDeviceCacheEntryByOrdinal(device_id);
    if (!entry || owner == nullptr) return;
    std::lock_guard<std::mutex> arena_lock{entry->arena_mutex};
    for (auto& arena : entry->device_arenas) {
      if (arena.owner == owner) arena.graph_capture_started = true;
    }
  }
  ORT_CATCH(...) {
  }
}

void CudaEpFactory::ForgetPendingSessionAccount(const OrtEp* ep) noexcept {
  ORT_TRY {
    std::lock_guard<std::mutex> lock(device_cache_mutex_);
    for (auto& [key, entry] : device_cache_) {
      static_cast<void>(key);
      std::lock_guard<std::mutex> arena_lock{entry.arena_mutex};
      entry.pending_session_accounts.erase(ep);
    }
  }
  ORT_CATCH(...) {
  }
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
