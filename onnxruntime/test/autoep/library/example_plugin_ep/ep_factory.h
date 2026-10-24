// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <mutex>

#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "ep_external_resource_importer.h"
#include "../plugin_ep_utils.h"
#include "ep.h"
#include "ep_custom_op.h"

/// <summary>
/// Example EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class ExampleEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger,
                   bool create_duplicate_virtual_devices = false);
  ~ExampleEpFactory();

  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

  OrtStatus* ResetArenaChunksUsingStream(const OrtSyncStreamImpl* stream_impl);

  // Get the EP version string.
  const std::string& GetEpVersionString() const {
    return ep_version_;
  }

  // Get the vendor ID.
  uint32_t GetVendorIdValue() const {
    return vendor_id_;
  }

  const OrtMemoryInfo* GetDefaultMemoryInfo() const {
    return default_memory_info_;
  }

  const OrtLogger& default_logger_;  // default logger for the EP factory

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* ep_device,
      OrtExternalResourceImporterImpl** out_importer) noexcept;

  static OrtStatus* ORT_API_CALL InitGraphicsInteropImpl(OrtEpFactory* this_ptr,
                                                         const OrtEpDevice* ep_device,
                                                         const OrtGraphicsInteropConfig* config) noexcept;

  static OrtStatus* ORT_API_CALL DeinitGraphicsInteropImpl(OrtEpFactory* this_ptr,
                                                           const OrtEpDevice* ep_device) noexcept;

  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* hw,
      OrtDeviceEpIncompatibilityDetails* details) noexcept;

  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                           _Out_ size_t* num_domains) noexcept;

  static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                        _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
                                                        _Out_ size_t num_domains) noexcept;

  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      const char* compatibility_info,
      OrtCompiledModelCompatibility* model_compatibility) noexcept;

  static OrtStatus* ORT_API_CALL SelectBestModelCandidateImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* device,
      const OrtKeyValuePairs* const* candidates,
      size_t num_candidates,
      const OrtSessionOptions* session_options,
      size_t* selected_index) noexcept;

  const std::string ep_name_;              // EP name
  const std::string vendor_{"Contoso"};    // EP vendor name
  const uint32_t vendor_id_{0xB357};       // EP vendor ID
  const std::string ep_version_{"0.1.0"};  // EP version
  const bool create_duplicate_virtual_devices_{false};
  std::array<OrtHardwareDevice*, 2> virtual_hardware_devices_{};

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;  // used for initializers
  Ort::MemoryInfo host_accessible_memory_info_;

  bool arena_allocator_using_default_settings_{true};
  std::unique_ptr<ArenaAllocator> arena_allocator_;  // shared device allocator that uses an arena
  uint32_t num_arena_users_{0};
  std::mutex mutex_;  // mutex to protect arena_allocator_, num_arena_users_ and graphics_interop_initialized_

  // The example EP has a single device, so one flag tracks its graphics interop state.
  bool graphics_interop_initialized_{false};

  std::unique_ptr<ExampleDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory

  std::vector<Ort::CustomOpDomain> custom_op_domains_{2};
  std::vector<std::vector<std::unique_ptr<ExampleEpCustomOp>>> created_custom_op_lists_{2};
};
