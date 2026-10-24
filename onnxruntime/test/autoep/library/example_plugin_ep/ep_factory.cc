// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>
#include <limits>
#include <optional>
#include <string_view>

#include "ep.h"
#include "ep_allocator.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "ep_stream_support.h"
#include "ep_test_hooks.h"

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace {
int CompatibilityRank(OrtCompiledModelCompatibility c) {
  switch (c) {
    case OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL:
      return 3;
    case OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION:
      return 2;
    case OrtCompiledModelCompatibility_EP_NOT_APPLICABLE:
      return 1;
    case OrtCompiledModelCompatibility_EP_UNSUPPORTED:
    default:
      return 0;
  }
}

constexpr std::string_view kVersionKey = "version=";
constexpr std::string_view kOrtApiVersionKey = "ort_api_version=";
constexpr std::string_view kHardwareArchitectureKey = "hardware_architecture=";

std::optional<std::string> GetCompatibilityField(const std::string& info, std::string_view key) {
  for (size_t search_start = 0;;) {
    const size_t key_position = info.find(key.data(), search_start, key.size());
    if (key_position == std::string::npos) {
      return std::nullopt;
    }

    if (key_position == 0 || info[key_position - 1] == ';') {
      const size_t value_start = key_position + key.size();
      const size_t value_end = info.find(';', value_start);
      return value_end == std::string::npos
                 ? info.substr(value_start)
                 : info.substr(value_start, value_end - value_start);
    }

    search_start = key_position + 1;
  }
}
}  // namespace

ExampleEpFactory::ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger,
                                   bool create_duplicate_virtual_devices)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      create_duplicate_virtual_devices_{create_duplicate_virtual_devices},
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr},
      host_accessible_memory_info_{nullptr} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
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
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

  CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;

  InitGraphicsInterop = InitGraphicsInteropImpl;
  DeinitGraphicsInterop = DeinitGraphicsInteropImpl;

  GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
  GetCustomOpDomains = GetCustomOpDomainsImpl;
  ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
  SelectBestModelCandidate = SelectBestModelCandidateImpl;

  // setup the OrtMemoryInfo instances required by the EP.
  // We pretend the device the EP is running on is GPU.
  default_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU",
                                         OrtMemoryInfoDeviceType_GPU,
                                         /*vendor*/ 0xBE57, /* device_id */ 0,
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment*/ 0,
                                         // it is invalid to use OrtArenaAllocator as that is reserved for the internal
                                         // ORT Arena implementation
                                         OrtAllocatorType::OrtDeviceAllocator};

  // create data transfer for the device
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_impl_ = std::make_unique<ExampleDataTransfer>(apis, device);

  // create read-only allocator for use with initializers. same info as DEFAULT memory apart from the allocator type.
  readonly_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU readonly",
                                          OrtMemoryInfoDeviceType_GPU,
                                          /*vendor*/ 0xBE57, /* device_id */ 0,
                                          OrtDeviceMemoryType_DEFAULT,
                                          /*alignment*/ 0,
                                          OrtAllocatorType::OrtReadOnlyAllocator};

  // HOST_ACCESSIBLE memory example. use the non-CPU device type so it's clear which device the memory is also
  // accessible from. we infer from the type of HOST_ACCESSIBLE that it's CPU accessible.
  host_accessible_memory_info_ = Ort::MemoryInfo{"ExampleEP GPU pinned",
                                                 OrtMemoryInfoDeviceType_GPU,
                                                 /*vendor*/ 0xBE57, /* device_id */ 0,
                                                 OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                 /*alignment*/ 0,
                                                 OrtAllocatorType::OrtDeviceAllocator};
  // Custom Op Domains
  custom_op_domains_[0] = Ort::CustomOpDomain{"test"};
  custom_op_domains_[1] = Ort::CustomOpDomain{"test2"};

  std::vector<std::unique_ptr<ExampleEpCustomOp>> created_custom_op_list;
  created_custom_op_list.push_back(std::make_unique<ExampleEpCustomOp>(ep_name_.c_str(), this));
  created_custom_op_list.back().get()->SetName("Custom_Mul");
  custom_op_domains_[0].Add(created_custom_op_list.back().get());

  std::vector<std::unique_ptr<ExampleEpCustomOp>> created_custom_op_list_2;
  created_custom_op_list_2.push_back(std::make_unique<ExampleEpCustomOp>(ep_name_.c_str(), this));
  created_custom_op_list_2.back().get()->SetName("Custom_Mul2");
  custom_op_domains_[1].Add(created_custom_op_list_2.back().get());

  created_custom_op_lists_[0] = std::move(created_custom_op_list);
  created_custom_op_lists_[1] = std::move(created_custom_op_list_2);
}

ExampleEpFactory::~ExampleEpFactory() {
  for (auto* device : virtual_hardware_devices_) {
    ep_api.ReleaseHardwareDevice(device);
  }
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL ExampleEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                  const OrtHardwareDevice* const* devices,
                                                                  size_t num_devices,
                                                                  OrtEpDevice** ep_devices,
                                                                  size_t max_ep_devices,
                                                                  size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

  auto add_ep_device = [&](const OrtHardwareDevice& device) -> OrtStatus* {
    auto key_value_pairs_deleter = [factory](OrtKeyValuePairs* kvps) {
      if (kvps != nullptr) {
        factory->ort_api.ReleaseKeyValuePairs(kvps);
      }
    };

    OrtKeyValuePairs* raw_ep_metadata = nullptr;
    factory->ort_api.CreateKeyValuePairs(&raw_ep_metadata);
    std::unique_ptr<OrtKeyValuePairs, decltype(key_value_pairs_deleter)> ep_metadata(raw_ep_metadata,
                                                                                     key_value_pairs_deleter);

    OrtKeyValuePairs* raw_ep_options = nullptr;
    factory->ort_api.CreateKeyValuePairs(&raw_ep_options);
    std::unique_ptr<OrtKeyValuePairs, decltype(key_value_pairs_deleter)> ep_options(raw_ep_options,
                                                                                    key_value_pairs_deleter);

    // random example using made up values
    factory->ort_api.AddKeyValuePair(ep_metadata.get(), "supported_devices", "CrackGriffin 7+");
    // Example os_driver_version. A real EP would read the OS driver version from the device.
    // The format is a 4-part dot-separated version matching the DXCore DriverVersion property.
    factory->ort_api.AddKeyValuePair(ep_metadata.get(), kOrtEpDevice_EpMetadataKey_OSDriverVersion, "31.0.101.1000");
    // GroupQueryAttention Value cache layout preference. "BNSH" here because GetCapabilityImpl()
    // only claims Mul, Custom_Mul and EPContext nodes, so this EP cannot fuse the
    // Transpose -> GroupQueryAttention -> Transpose sequence that ORT inserts for "BNHS".
    // Reporting "BNHS" without implementing that fusion would steer applications into a layout
    // this EP cannot execute any faster, and the transposes would run for real.
    factory->ort_api.AddKeyValuePair(ep_metadata.get(), kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout, "BNSH");
    // Report weightless support for all initializers.
    factory->ort_api.AddKeyValuePair(ep_metadata.get(), kOrtEpDevice_EpMetadataKey_WeightlessSupport, "all");
    factory->ort_api.AddKeyValuePair(ep_options.get(), "run_really_fast", "true");

    // OrtEpDevice copies ep_metadata and ep_options.
    OrtEpDevice* ep_device = nullptr;
    auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata.get(), ep_options.get(),
                                                               &ep_device);
    if (status != nullptr) {
      return status;
    }

    // Register the allocator info required by the EP.
    // OrtReadOnlyAllocator + OrtDeviceMemoryType_DEFAULT allocator for use with initializers is optional.
    // OrtDeviceMemoryType_HOST_ACCESSIBLE is also optional and exposes CPU-accessible memory on the EP device.
    status = factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_);
    if (status != nullptr) {
      return status;
    }

    status = factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_);
    if (status != nullptr) {
      return status;
    }

    status = factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->host_accessible_memory_info_);
    if (status != nullptr) {
      return status;
    }

    ep_devices[num_ep_devices++] = ep_device;
    return nullptr;
  };

  if (factory->create_duplicate_virtual_devices_) {
    // Exercise the multi-device AutoEP case without relying on host hardware: both virtual devices belong to this
    // factory and therefore expose the same custom-op domains. Session creation must register each domain only once.
    for (size_t i = 0; i < factory->virtual_hardware_devices_.size() && num_ep_devices < max_ep_devices; ++i) {
      if (factory->virtual_hardware_devices_[i] == nullptr) {
        OrtKeyValuePairs* virtual_device_metadata = nullptr;
        factory->ort_api.CreateKeyValuePairs(&virtual_device_metadata);
        factory->ort_api.AddKeyValuePair(virtual_device_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");

        auto* status = factory->ep_api.CreateHardwareDevice(OrtHardwareDeviceType::OrtHardwareDeviceType_CPU,
                                                            factory->vendor_id_, static_cast<uint32_t>(i),
                                                            factory->vendor_.c_str(), virtual_device_metadata,
                                                            &factory->virtual_hardware_devices_[i]);
        factory->ort_api.ReleaseKeyValuePairs(virtual_device_metadata);
        if (status != nullptr) {
          return status;
        }
      }

      auto* status = add_ep_device(*factory->virtual_hardware_devices_[i]);
      if (status != nullptr) {
        return status;
      }
    }

    return nullptr;
  }

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      auto* status = add_ep_device(device);
      if (status != nullptr) {
        return status;
      }
    }

    // C++ API equivalent. Throws on error.
    //{
    //  Ort::ConstHardwareDevice device(devices[i]);
    //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
    //    Ort::KeyValuePairs ep_metadata;
    //    Ort::KeyValuePairs ep_options;
    //    ep_metadata.Add("supported_devices", "CrackGriffin 7+");
    //    ep_metadata.Add(kOrtEpDevice_EpMetadataKey_OSDriverVersion, "31.0.101.1000");
    //    ep_options.Add("run_really_fast", "true");
    //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
    //    ep_devices[num_ep_devices++] = ep_device.release();
    //  }
    //}
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* /*devices*/,
                                                       const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                       size_t num_devices,
                                                       const OrtSessionOptions* session_options,
                                                       const OrtLogger* logger,
                                                       OrtEp** ep) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for CPU and only expected to be selected for one CPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
    // the EP has been selected for.
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP only supports selection for one device.");
  }

  // Create the execution provider
  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  // Create EP configuration from session options, if needed.
  // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
  std::string ep_context_enable;
  std::string ep_context_embed_mode;
  std::string ep_context_output_model_path;
  std::string weightless_ep_context_nodes_enable;
  std::string advertise_ep_context_data_support;
  std::string test_ort_version;
  std::string use_default_cpu_allocator;
  std::string test_read_ep_context_during_compile;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextEnable, "0",
                                                 ep_context_enable));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextEmbedMode, "0",
                                                 ep_context_embed_mode));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpContextFilePath, "",
                                                 ep_context_output_model_path));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kOrtSessionOptionEpEnableWeightlessEpContextNodes,
                                                 "0", weightless_ep_context_nodes_enable));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kExampleEpTestEpContextDataSupport, "1",
                                                 advertise_ep_context_data_support));

  if (advertise_ep_context_data_support != "0" && advertise_ep_context_data_support != "1") {
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP context data support test option must be '0' or '1'.");
  }
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, kExampleEpTestOrtVersion, "current",
                                                 test_ort_version));
  if (test_ort_version != "current" && test_ort_version != "30") {
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP test ORT version must be 'current' or '30'.");
  }
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, "ep.example.use_default_cpu_allocator",
                                                 "0", use_default_cpu_allocator));
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options,
                                                 "ep.example.test_read_ep_context_during_compile", "0",
                                                 test_read_ep_context_during_compile));
  if (test_read_ep_context_during_compile != "0" && test_read_ep_context_during_compile != "1") {
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP test read EPContext during compile option must be '0' or '1'.");
  }

  ExampleEp::Config config = {};
  config.enable_ep_context = ep_context_enable == "1";
  config.embed_ep_context_in_model = ep_context_embed_mode == "1";
  config.ep_context_output_model_path = std::move(ep_context_output_model_path);
  config.enable_weightless_ep_context_nodes = weightless_ep_context_nodes_enable == "1";
  config.advertise_ep_context_data_support = advertise_ep_context_data_support == "1";
  config.use_default_cpu_allocator = use_default_cpu_allocator == "1";
  config.test_read_ep_context_during_compile = test_read_ep_context_during_compile == "1";

  // The EpContextConfig wrapper captures a stable snapshot of the EPContext callbacks from the session options and
  // owns the snapshot handle. EXCEPTION_TO_RETURNED_STATUS_END converts extraction failures into an OrtStatus.
  auto dummy_ep = std::make_unique<ExampleEp>(
      *factory, factory->ep_name_, config, *logger,
      Ort::EpContextConfig{Ort::ConstSessionOptions{session_options}});
  if (test_ort_version == "30") {
    // Keep the support callback populated to verify that ORT ignores it for older EP versions.
    dummy_ep->ort_version_supported = 30;
  }
  *ep = dummy_ep.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
  delete dummy_ep;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                              const OrtMemoryInfo* memory_info,
                                                              const OrtKeyValuePairs* allocator_options,
                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *allocator = nullptr;

  bool is_default_allocator = memory_info == factory.default_memory_info_;
  bool is_readonly_allocator = memory_info == factory.readonly_memory_info_;
  bool is_host_accessible_allocator = memory_info == factory.host_accessible_memory_info_;

  if (!is_default_allocator && !is_readonly_allocator && !is_host_accessible_allocator) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  // NOTE: The factory implementation is free to return a shared OrtAllocator* instance instead of creating a new
  //       allocator on each call. To do this have an allocator instance as an OrtEpFactory class member and make
  //       ReleaseAllocatorImpl a no-op.
  //
  // NOTE: EP should implement its own arena logic. ep_arena.cc/h is provided as a reference and we use it here for
  //       device memory. `allocator_options` can be used for arena configuration and there is a helper in ep_arena.h
  //       to convert from OrtKeyValuePairs to the same arena config settings that ORT uses.
  //       You are of course free to have completely different settings.

  // the read-only allocator is used for initializers. we don't need an arena for that.
  // host-accessible memory is also returned via a plain non-arena allocator.
  if (is_readonly_allocator || is_host_accessible_allocator) {
    auto simple_allocator = std::make_unique<CustomAllocator>(memory_info, factory);
    *allocator = simple_allocator.release();
    return nullptr;
  }

  // create/use the shared arena based allocator
  std::lock_guard<std::mutex> lock{factory.mutex_};

  if (!factory.arena_allocator_) {
    AllocatorUniquePtr ep_allocator = std::make_unique<CustomAllocator>(memory_info, factory);

    // initial shared allocator in environment does not have allocator options.
    // if the user calls CreateSharedAllocator they can provide options to configure the arena differently.
    factory.arena_allocator_using_default_settings_ = allocator_options == nullptr;
    RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(ep_allocator), allocator_options,
                                                            factory.ort_api,
                                                            factory.default_logger_, factory.arena_allocator_));

  } else {
    if (factory.arena_allocator_using_default_settings_ && allocator_options) {
      // potential change in arena settings. up to EP author to determine how to handle this.
      // we should not get here if replacing the shared allocator in the environment, as we free the existing one
      // before replacing it. i.e. ReleaseAllocatorImpl should have been called, and arena_allocator_ should be null.
    }
  }

  ++factory.num_arena_users_;
  *allocator = factory.arena_allocator_.get();

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  std::lock_guard<std::mutex> lock{factory.mutex_};

  if (allocator == factory.arena_allocator_.get()) {
    if (--factory.num_arena_users_ == 0) {
      factory.arena_allocator_ = nullptr;
    }
  } else {
    delete static_cast<CustomAllocator*>(allocator);
  }
}

OrtStatus* ExampleEpFactory::ResetArenaChunksUsingStream(const OrtSyncStreamImpl* stream_impl) {
  OrtStatus* status = nullptr;
  try {
    std::lock_guard<std::mutex> lock{mutex_};
    status = arena_allocator_ ? arena_allocator_->ResetChunksUsingStream(stream_impl) : nullptr;
  } catch (const std::exception& ex) {
    status = ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
  } catch (...) {
    status = ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "ResetArenaChunksUsingStream failed.");
  }
  return status;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();

  if (ShouldFailCreateDataTransfer()) {
    return factory.ort_api.CreateStatus(ORT_FAIL, "injected data transfer creation failure");
  }

  return nullptr;
}

/*static*/
bool ORT_API_CALL ExampleEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // the example EP implements stream synchronization.
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                        const OrtMemoryDevice* memory_device,
                                                                        const OrtKeyValuePairs* stream_options,
                                                                        OrtSyncStreamImpl** stream) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *stream = nullptr;

  // we only need stream synchronization on the device stream
  if (factory.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT) {
    auto sync_stream = std::make_unique<StreamImpl>(factory, /*OrtEp**/ nullptr, stream_options);
    *stream = sync_stream.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                                    _Out_ size_t* num_domains) noexcept {
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
  *num_domains = factory->custom_op_domains_.size();

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetCustomOpDomainsImpl(
    OrtEpFactory* this_ptr,
    _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
    _Out_ size_t num_domains) noexcept {
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

  // The `num_domains` should be 2 as ORT calls GetNumCustomOpDomainsImpl() to get the number prior to
  // call this function.
  gsl::span<OrtCustomOpDomain*> domains_span(domains, num_domains);
  domains_span[0] = factory->custom_op_domains_[0];
  domains_span[1] = factory->custom_op_domains_[1];

  return nullptr;
}

OrtStatusPtr ExampleEpCustomOp::CreateKernelV2(const OrtApi& /*api*/,
                                               const OrtKernelInfo* /*info*/,
                                               void** op_kernel) const {
  std::string node_input_0 = "X";
  std::string node_input_1 = "W";
  auto custom_kernel_op = std::make_unique<CustomMulKernel>(factory_->ort_api,
                                                            factory_->default_logger_,
                                                            float_initializers_,
                                                            node_input_0,
                                                            node_input_1);
  *op_kernel = custom_kernel_op.release();
  return nullptr;
}

OrtStatusPtr ExampleEpCustomOp::KernelComputeV2(void* op_kernel, OrtKernelContext* context) const {
  return static_cast<CustomMulKernel*>(op_kernel)->ComputeV2(context);
}

OrtStatus* ORT_API_CALL ExampleEpFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* hw,
    OrtDeviceEpIncompatibilityDetails* details) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  // Example: This EP only supports CPU devices. Report incompatibility for non-CPU devices.
  OrtHardwareDeviceType device_type = factory.ort_api.HardwareDevice_Type(hw);

  if (device_type != OrtHardwareDeviceType_CPU) {
    // Report that the device type is not supported
    uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return factory.ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        static_cast<int32_t>(device_type),  // Use device type as the error code for testing
        "ExampleEP only supports CPU devices");
  }

  // Device is compatible - details are already initialized with default values by ORT
  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::CreateExternalResourceImporterForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtEpDevice* /*ep_device*/,
    OrtExternalResourceImporterImpl** out_importer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (out_importer == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "out_importer cannot be nullptr");
  }

  // Create the external resource importer
  // NOTE: For production multi-GPU EPs, you should capture ep_device in the importer
  // to enable proper device validation and support multiple physical devices.
  // This example EP only supports a single device, so we don't store it.
  auto importer = std::make_unique<ExampleExternalResourceImporter>(factory);
  *out_importer = importer.release();

  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::InitGraphicsInteropImpl(OrtEpFactory* this_ptr,
                                                                  const OrtEpDevice* /*ep_device*/,
                                                                  const OrtGraphicsInteropConfig* config) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  // OrtGraphicsInteropConfig was added in ORT API version 25.
  if (config->version < 25) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "ExampleEP: OrtGraphicsInteropConfig version must be at least 25");
  }

  // The example EP has no real graphics device; it only validates the requested API and tracks state.
  if (config->graphics_api != ORT_GRAPHICS_API_D3D12 && config->graphics_api != ORT_GRAPHICS_API_VULKAN) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "ExampleEP: unsupported graphics API");
  }

  std::lock_guard<std::mutex> lock{factory.mutex_};
  if (factory.graphics_interop_initialized_) {
    return factory.ort_api.CreateStatus(ORT_FAIL, "ExampleEP: graphics interop is already initialized");
  }

  factory.graphics_interop_initialized_ = true;
  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::DeinitGraphicsInteropImpl(OrtEpFactory* this_ptr,
                                                                    const OrtEpDevice* /*ep_device*/) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  std::lock_guard<std::mutex> lock{factory.mutex_};
  if (!factory.graphics_interop_initialized_) {
    return factory.ort_api.CreateStatus(ORT_FAIL, "ExampleEP: graphics interop is not initialized");
  }

  factory.graphics_interop_initialized_ = false;
  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (model_compatibility == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "model_compatibility cannot be nullptr");
  }

  // Match CreateEpImpl's device-configuration requirements. A multi-device EP would instead apply the same
  // selection, fallback, or joint-participation rules that its CreateEp implementation uses.
  if (devices == nullptr || num_devices != 1 || devices[0] == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Example EP only supports selection for one device.");
  }

  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  const std::string info(compatibility_info);
  if (info.find(factory.ep_name_ + ";") != 0) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  // Expected format:
  // "<ep_name>;version=<semver>;ort_api_version=<ORT_API_VERSION>;hardware_architecture=<architecture>".
  const auto ep_version = GetCompatibilityField(info, kVersionKey);
  if (!ep_version.has_value()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  if (*ep_version != factory.ep_version_) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    return nullptr;
  }

  const auto ort_api_version = GetCompatibilityField(info, kOrtApiVersionKey);
  if (ort_api_version.has_value() && *ort_api_version != std::to_string(ORT_API_VERSION)) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    return nullptr;
  }

  const auto hardware_architecture = GetCompatibilityField(info, kHardwareArchitectureKey);
  if (hardware_architecture.has_value() && *hardware_architecture != "arch1") {
    *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    return nullptr;
  }

  *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  return nullptr;
}

OrtStatus* ORT_API_CALL ExampleEpFactory::SelectBestModelCandidateImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* device,
    const OrtKeyValuePairs* const* candidates,
    size_t num_candidates,
    const OrtSessionOptions* /*session_options*/,
    size_t* selected_index) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

  if (selected_index == nullptr) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "selected_index cannot be nullptr");
  }

  *selected_index = std::numeric_limits<size_t>::max();

  if (candidates == nullptr || num_candidates == 0) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "candidates cannot be nullptr or empty");
  }

  const OrtHardwareDevice* devices[] = {device};

  int best_rank = -1;
  size_t best_idx = std::numeric_limits<size_t>::max();

  for (size_t i = 0; i < num_candidates; ++i) {
    const char* compatibility_info =
        factory.ort_api.GetKeyValue(candidates[i], "ep_compatibility_info");

    if (compatibility_info == nullptr) {
      return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "candidate metadata is missing required key: ep_compatibility_info");
    }

    OrtCompiledModelCompatibility compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    OrtStatus* status = ValidateCompiledModelCompatibilityInfoImpl(
        this_ptr, devices, 1, compatibility_info, &compatibility);
    if (status != nullptr) {
      return status;
    }

    const int rank = CompatibilityRank(compatibility);
    if (rank > best_rank) {
      best_rank = rank;
      best_idx = i;
    }
  }

  if (best_rank <= CompatibilityRank(OrtCompiledModelCompatibility_EP_UNSUPPORTED)) {
    *selected_index = std::numeric_limits<size_t>::max();
  } else {
    *selected_index = best_idx;
  }

  return nullptr;
}
