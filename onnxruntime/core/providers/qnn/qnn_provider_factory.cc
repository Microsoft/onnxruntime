// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <string>
#include <unordered_map>
#include <utility>
#include "core/providers/qnn/qnn_provider_factory_creator.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/session/abi_devices.h"

static const std::unordered_map<OrtHardwareDeviceType, std::string> kDefaultBackends = {
#if defined(_WIN32)
    {OrtHardwareDeviceType_NPU, "QnnHtp.dll"},
    {OrtHardwareDeviceType_GPU, "QnnGpu.dll"},
#else
    {OrtHardwareDeviceType_NPU, "libQnnHtp.so"},
    {OrtHardwareDeviceType_GPU, "libQnnGpu.so"},
#endif
};

namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map, const ConfigOptions* config_options)
      : provider_options_map_(provider_options_map), config_options_(config_options) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_, config_options_);
  }

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // We extract those EP options and pass them to QNN EP as separate "provider options".
    std::unordered_map<std::string, std::string> provider_options = provider_options_map_;
    std::string key_prefix = "ep.";
    key_prefix += qnn::utils::GetLowercaseString(kQnnExecutionProvider);
    key_prefix += ".";

    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    qnn::EpContextDataCallbacks ep_context_data_callbacks;
    session_options.GetEpContextDataCallbacks(&ep_context_data_callbacks.read_func,
                                              &ep_context_data_callbacks.read_state,
                                              &ep_context_data_callbacks.write_func,
                                              &ep_context_data_callbacks.write_state);
    auto qnn_ep = std::make_unique<QNNExecutionProvider>(provider_options, &config_options,
                                                         ep_context_data_callbacks);
    qnn_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return qnn_ep;
  }

 private:
  ProviderOptions provider_options_map_;
  const ConfigOptions* config_options_;
};

#if BUILD_QNN_EP_STATIC_LIB
std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                                                             const SessionOptions* session_options) {
  const ConfigOptions* config_options = nullptr;
  if (session_options != nullptr) {
    config_options = &session_options->config_options;
  }

  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map, config_options);
}
#else
/// @brief Gets the path of directory containing the dynamic library that contains the address.
/// @param address An address of a function or variable in the dynamic library.
/// @return The path of the directory containing the dynamic library, or an empty string if the path cannot be determined.
static onnxruntime::PathString GetDynamicLibraryLocationByAddress(const void* address) {
#ifdef _WIN32
  HMODULE moduleHandle;
  if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &moduleHandle)) {
    return {};
  }
  std::wstring buffer;
  for (std::uint32_t size{70}; size < 4096; size *= 2) {
    buffer.resize(size, L'\0');
    const std::uint32_t requiredSize = ::GetModuleFileNameW(moduleHandle, buffer.data(), size);
    if (requiredSize == 0) {
      break;
    }
    if (requiredSize == size) {
      continue;
    }
    buffer.resize(requiredSize);
    return {std::move(buffer)};
  }
#else
  std::ignore = address;
#endif
  return {};
}

struct QNN_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) override {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    return std::make_shared<onnxruntime::QNNProviderFactory>(*provider_options, config_options);
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (provider_options.find("backend_path") == provider_options.end() &&
        provider_options.find("backend_type") == provider_options.end()) {
      // If neither "backend_path" nor "backend_type" has been given in the provider options, then determine the backend based
      // on the provided devices. As QNN EP does not support partitioning across backends, if multiple devices are provided,
      // default to HTP (if present) or else to the GPU.
      const OrtHardwareDevice* device_to_use = nullptr;
      if (num_devices == 0) {
        return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "No devices were provided to QNN EP.");
      } else if (num_devices == 1) {
        device_to_use = devices[0];
      } else {
        const auto is_npu = [](const OrtHardwareDevice* device) { return device->type == OrtHardwareDeviceType_NPU; };
        const auto is_gpu = [](const OrtHardwareDevice* device) { return device->type == OrtHardwareDeviceType_GPU; };

        auto device_it = std::find_if(devices, devices + num_devices, is_npu);
        if (device_it != devices + num_devices) {
          LOGS_DEFAULT(WARNING) << "QNN EP only supports one device. Only the NPU device will be used.";
          device_to_use = *device_it;
        } else {
          device_it = std::find_if(devices, devices + num_devices, is_gpu);
          if (device_it != devices + num_devices) {
            LOGS_DEFAULT(WARNING)
                << "QNN EP only supports one device. An NPU device was not provided, so only the GPU device will be used.";
            device_to_use = *device_it;
          } else {
            return Status(common::ONNXRUNTIME, ORT_EP_FAIL,
                          "Multiple devices were provided to QNN EP, but neither an NPU nor a GPU was included.");
          }
        }
      }
      ORT_RETURN_IF(device_to_use == nullptr, "Failed to select device for QNN EP!");

      auto default_backends_it = kDefaultBackends.find(device_to_use->type);
      ORT_RETURN_IF(default_backends_it == kDefaultBackends.end(),
                    "Could not determine default backend path for device of type: ", device_to_use->type);

      // Identify the path of the current dynamic library, and expect that the backend library is in the same directory.
      onnxruntime::PathString current_path = GetDynamicLibraryLocationByAddress(
          reinterpret_cast<const void*>(&kDefaultBackends));

      std::filesystem::path parent_path;
      if (!current_path.empty()) {
        parent_path = std::filesystem::path{std::move(current_path)}.parent_path();
      }

      provider_options["backend_path"] = (parent_path / default_backends_it->second).string();
    }

    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);
    auto ep_factory = CreateExecutionProviderFactory(arg);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
  }

  void Initialize() override {}
  void Shutdown() override {}
} g_provider;
#endif  // BUILD_QNN_EP_STATIC_LIB

}  // namespace onnxruntime

#if !BUILD_QNN_EP_STATIC_LIB
extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

#include "core/framework/error_code_helper.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
struct QnnEpFactory : OrtEpFactory {
  QnnEpFactory(const OrtApi& ort_api_in,
               const OrtLogger& default_logger_in,
               const char* ep_name)
      : OrtEpFactory{},
        ort_api{ort_api_in},
        default_logger{default_logger_in},
        ep_name{ep_name} {
    ort_version_supported = ORT_API_VERSION;
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

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_vendor_id;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of QNN EP is not currently setup to partition a model among
  // multiple different QNN backends at once (e.g, npu, cpu, gpu), so currently this factory instance is set
  // to default to npu.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    num_ep_devices = 0;

    auto* factory = static_cast<QnnEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (kDefaultBackends.find(factory->ort_api.HardwareDevice_Type(&device)) != kDefaultBackends.end() &&
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtStatus* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, nullptr,
                                                                        &ep_devices[num_ep_devices++]);
        ORT_API_RETURN_IF_ERROR(status);
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "QNN EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* /*memory_info*/,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept {
    auto& factory = *static_cast<QnnEpFactory*>(this_ptr);
    *allocator = nullptr;

    // we don't add allocator info to the OrtEpDevice we return so this should never be called.
    return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "QNN EP factory does not support CreateAllocator.");
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* /*allocator*/) noexcept {
    // we don't support CreateAllocator so this should never be called.
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;  // return nullptr to indicate that this EP does not support data transfer.
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* /*memory_device*/,
                                                               const OrtKeyValuePairs* /*stream_options*/,
                                                               OrtSyncStreamImpl** ort_stream) noexcept {
    auto& factory = *static_cast<QnnEpFactory*>(this_ptr);
    *ort_stream = nullptr;

    // should never be called as IsStreamAware returns false
    return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "QNN EP factory does not support CreateSyncStreamForDevice.");
  }

  const OrtApi& ort_api;
  const OrtLogger& default_logger;
  const std::string ep_name;                 // EP name
  const std::string ep_vendor{"Microsoft"};  // EP vendor name
  uint32_t ep_vendor_id{0x1414};             // Microsoft vendor ID

  // Qualcomm vendor ID. Refer to the ACPI ID registry (search Qualcomm): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  auto factory = std::make_unique<QnnEpFactory>(*ort_api, *default_logger,
                                                onnxruntime::kQnnExecutionProvider);
  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<QnnEpFactory*>(factory);
  return nullptr;
}
}
#endif  // !BUILD_QNN_EP_STATIC_LIB
