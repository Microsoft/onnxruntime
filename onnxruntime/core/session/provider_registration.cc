// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <sstream>
#include <string>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/provider_options.h"
#include "core/graph/constants.h"
#include "core/platform/telemetry_strings.h"
#include "core/providers/provider_factory_creators.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"

#ifdef _WIN32
#include <winmeta.h>
#include "core/platform/tracing.h"
#endif

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

using namespace onnxruntime;

namespace onnxruntime {
// Constants for the maximum string lengths of provider options keys and values. The maximum lengths are related
// to the limits for session config options because we add provider options to the session configs map.
static constexpr size_t kMaxSessionConfigsEpPrefixLength = 64;  // prefix for new key would be "ep.<EP_NAME>."
static constexpr size_t kMaxProviderOptionKeyLength = ConfigOptions::kMaxKeyLength - kMaxSessionConfigsEpPrefixLength;
static constexpr size_t kMaxProviderOptionValueLength = ConfigOptions::kMaxValueLength;
}  // namespace onnxruntime

namespace {

OrtStatus* ParseProviderOptions(_In_reads_(num_keys) const char* const* provider_options_keys,
                                _In_reads_(num_keys) const char* const* provider_options_values,
                                _In_ size_t num_keys,
                                ProviderOptions& provider_options) {
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provider options key/value cannot be empty");
    }

    // Check that provider options keys and values are within the allowed maximum lengths.
    const size_t key_length = strlen(provider_options_keys[i]);
    if (key_length > kMaxProviderOptionKeyLength) {
      std::ostringstream error_builder;
      error_builder << "Provider option key length is " << key_length << " but the limit is "
                    << kMaxProviderOptionKeyLength << ". Provider option key: " << provider_options_keys[i];
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, error_builder.str().c_str());
    }

    const size_t value_length = strlen(provider_options_values[i]);
    if (value_length > kMaxProviderOptionValueLength) {
      std::ostringstream error_builder;
      error_builder << "Provider option value length is " << value_length << " but the limit is "
                    << kMaxProviderOptionValueLength << ". Provider option key: " << provider_options_keys[i];
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, error_builder.str().c_str());
    }

    provider_options[provider_options_keys[i]] = provider_options_values[i];
  }

  return nullptr;
}
}  // namespace
/**
 * Implementation of OrtApis functions for provider registration.
 *
 * EPs that use the provider bridge are handled in provider_bridge_ort.cc
 */

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* provider_name,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys) {
  API_IMPL_BEGIN
  enum class EpID {
    INVALID = 0,
    DML,
    QNN,
    OpenVINO,
    SNPE,  // TODO(adrianlizarraga): Remove SNPE entirely because it has been replaced by QNN EP.
    XNNPACK,
    WEBNN,
    WebGPU,
    AZURE,
    JS,
    VitisAI,
    CoreML,
    MIGraphX,
    CPU
  };

  struct EpToAppend {
    EpID id = EpID::INVALID;
    const char* short_name = nullptr;
    const char* canonical_name = nullptr;
  };

  static std::array<EpToAppend, 13> supported_eps = {
      EpToAppend{EpID::DML, "DML", kDmlExecutionProvider},
      EpToAppend{EpID::QNN, "QNN", kQnnExecutionProvider},
      EpToAppend{EpID::OpenVINO, "OpenVINO", kOpenVINOExecutionProvider},
      EpToAppend{EpID::SNPE, "SNPE", kSnpeExecutionProvider},
      EpToAppend{EpID::XNNPACK, "XNNPACK", kXnnpackExecutionProvider},
      EpToAppend{EpID::WEBNN, "WEBNN", kWebNNExecutionProvider},
      EpToAppend{EpID::WebGPU, "WebGPU", kWebGpuExecutionProvider},
      EpToAppend{EpID::AZURE, "AZURE", kAzureExecutionProvider},
      EpToAppend{EpID::JS, "JS", kJsExecutionProvider},
      EpToAppend{EpID::VitisAI, "VitisAI", kVitisAIExecutionProvider},
      EpToAppend{EpID::CoreML, "CoreML", kCoreMLExecutionProvider},
      EpToAppend{EpID::MIGraphX, "MIGraphX", kMIGraphXExecutionProvider},
      EpToAppend{EpID::CPU, "CPU", kCpuExecutionProvider}};

  ProviderOptions provider_options;
  OrtStatus* status = ParseProviderOptions(provider_options_keys,
                                           provider_options_values,
                                           num_keys,
                                           provider_options);
  if (status != nullptr) {
    return status;
  }

#if defined(_WIN32) && defined(ONNXRUNTIME_ENABLE_INSTRUMENT)
  size_t option_count = 0;
  for (const auto& config_pair : provider_options) {
    if (option_count++ == telemetry_detail::kMaxTelemetryCollectionEntries) break;
    telemetry_detail::TelemetryStrings strings;
    TraceLoggingWrite(
        telemetry_provider_handle,
        "ProviderOptionsAppendExecutionProvider",
        TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
        TraceLoggingLevel(WINEVENT_LEVEL_INFO),
        TraceLoggingString(strings.Utf8(provider_name), "ProviderName"),
        TraceLoggingString(strings.Utf8(config_pair.first), "Key"),
        TraceLoggingString(strings.Utf8(config_pair.second), "Value"));
  }
#endif

  auto create_not_supported_status = [&provider_name]() {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 (std::string(provider_name) + " execution provider is not supported in this build. ").c_str());
  };

  auto create_failed_to_load_provider_status = [&provider_name]() {
    return OrtApis::CreateStatus(ORT_FAIL,
                                 (std::string("Failed to load provider ") + provider_name).c_str());
  };

  auto create_unknown_provider_status = [&provider_name](gsl::span<const EpToAppend> supported_eps) -> OrtStatus* {
    std::ostringstream str_builder;
    str_builder << "Unknown provider name '" << provider_name << "'. "
                << "Currently supported values are ";
    const size_t num_eps = supported_eps.size();
    for (size_t i = 0; i < num_eps; ++i) {
      const EpToAppend& ep_info = supported_eps[i];

      str_builder << "'" << ep_info.short_name << "'/'" << ep_info.canonical_name << "'";
      if (num_eps >= 2 && i == num_eps - 2) {
        str_builder << ", and ";
      } else if (i == num_eps - 1) {
        str_builder << ".";
      } else {
        str_builder << ", ";
      }
    }

    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, str_builder.str().c_str());
  };

  // The provider-bridge NvTensorRtRtx EP is deprecated. Report the plugin EP flow with the deprecation error
  if (strcmp(provider_name, "NvTensorRtRtx") == 0 || strcmp(provider_name, kNvTensorRTRTXExecutionProvider) == 0) {
    return OrtApis::CreateStatus(
        ORT_INVALID_ARGUMENT,
        (std::string("The '") + provider_name +
         "' execution provider is no longer built into ONNX Runtime. Use the NVIDIA TensorRT RTX plugin EP "
         "instead: register its library with RegisterExecutionProviderLibrary, select an OrtEpDevice from "
         "GetEpDevices, and append it with SessionOptionsAppendExecutionProvider_V2.")
            .c_str());
  }

  auto ep_to_append_iter = std::find_if(supported_eps.begin(), supported_eps.end(),
                                        [&provider_name](const EpToAppend& elem) -> bool {
                                          return (strcmp(provider_name, elem.short_name) == 0) ||
                                                 (strcmp(provider_name, elem.canonical_name) == 0);
                                        });

  if (ep_to_append_iter == supported_eps.end()) {
    return create_unknown_provider_status(supported_eps);
  }

  const EpToAppend& ep_to_append = *ep_to_append_iter;
  ORT_ENFORCE(ep_to_append.id != EpID::INVALID);

  // Add provider options to the session config options.
  // Use a new key with the format: "ep.<lower_case_ep_name>.<PROVIDER_OPTION_KEY>"
  ORT_API_RETURN_IF_STATUS_NOT_OK(options->AddProviderOptionsToConfigOptions(provider_options,
                                                                             ep_to_append.canonical_name));

  switch (ep_to_append.id) {
    case EpID::CPU: {
      // CPU EP is always available by default. Accept the name as valid but do nothing,
      // since the CPU EP is implicitly registered in every session.
      break;
    }
    case EpID::DML: {
#if defined(USE_DML)
      options->provider_factories.push_back(
          DMLProviderFactoryCreator::CreateFromProviderOptions(options->value.config_options, provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::QNN: {
#if defined(USE_QNN) || defined(USE_QNN_PROVIDER_INTERFACE)
      if (auto ep_factory = QNNProviderFactoryCreator::Create(provider_options, &(options->value)); ep_factory) {
        options->provider_factories.push_back(std::move(ep_factory));
      } else {
        status = create_failed_to_load_provider_status();
      }
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::OpenVINO: {
#if defined(USE_OPENVINO) || defined(USE_OPENVINO_PROVIDER_INTERFACE)
      if (auto ep_factory = OpenVINOProviderFactoryCreator::Create(&provider_options, &(options->value)); ep_factory) {
        options->provider_factories.push_back(std::move(ep_factory));
      } else {
        status = create_failed_to_load_provider_status();
      }
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::SNPE: {
#if defined(USE_SNPE)
      options->provider_factories.push_back(SNPEProviderFactoryCreator::Create(provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::XNNPACK: {
#if defined(USE_XNNPACK)
      options->provider_factories.push_back(XnnpackProviderFactoryCreator::Create(provider_options, &(options->value)));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::WEBNN: {
#if defined(USE_WEBNN)
      std::string deviceType = options->value.config_options.GetConfigOrDefault("deviceType", "cpu");
      provider_options["deviceType"] = deviceType;
      options->provider_factories.push_back(WebNNProviderFactoryCreator::Create(provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::WebGPU: {
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
      options->provider_factories.push_back(WebGpuProviderFactoryCreator::Create(options->value.config_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::AZURE: {
#if defined(USE_AZURE)
      options->provider_factories.push_back(AzureProviderFactoryCreator::Create(provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::JS: {
#if defined(USE_JSEP)
      std::string preferred_layout;
      if (options->value.config_options.TryGetConfigEntry("preferredLayout", preferred_layout)) {
        provider_options["preferred_layout"] = preferred_layout;
      }
      options->provider_factories.push_back(JsProviderFactoryCreator::Create(provider_options, &(options->value)));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::MIGraphX: {
#if defined(USE_MIGRAPHX) || defined(USE_MIGRAPHX_PROVIDER_INTERFACE)
      options->provider_factories.push_back(MIGraphXProviderFactoryCreator::Create(provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::VitisAI: {
#if defined(USE_VITISAI) || defined(USE_VITISAI_PROVIDER_INTERFACE)
      status = OrtApis::SessionOptionsAppendExecutionProvider_VitisAI(options, provider_options_keys,
                                                                      provider_options_values, num_keys);
#else
      status = create_not_supported_status();
#endif
      break;
    }
    case EpID::CoreML: {
#if defined(USE_COREML)
      options->provider_factories.push_back(CoreMLProviderFactoryCreator::Create(provider_options));
#else
      status = create_not_supported_status();
#endif
      break;
    }
    default:
      ORT_UNUSED_PARAMETER(options);
      ORT_UNUSED_PARAMETER(create_failed_to_load_provider_status);
      status = create_unknown_provider_status(supported_eps);
  }

  return status;
  API_IMPL_END
}

static OrtStatus* CreateNotEnabledStatus(const std::string& ep) {
  return OrtApis::CreateStatus(ORT_FAIL, (ep + " execution provider is not enabled in this build. ").c_str());
}

/**
 * Stubs for the publicly exported static registration functions for EPs that are referenced in the C# bindings
 * and are not implemented in provider_bridge_ort.cc.
 *
 * NOTE: The nuget packages that the C# bindings will use are all for full builds, so we don't need to allow for
 * provider_bridge_ort.cc being excluded in a minimal build.
 *
 * These are required when building an iOS app using Xamarin as all external symbols must be defined at compile time.
 * In that case a static ORT library is used and the symbol needs to exist but doesn't need to be publicly exported.
 * TODO: Not sure if we need to purely limit to iOS builds, so limit to __APPLE__ for now
 */
#ifdef __APPLE__
#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_DML
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("DML");
}
#endif

#ifndef USE_NNAPI
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi,
                    _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(nnapi_flags);
  return CreateNotEnabledStatus("NNAPI");
}
#endif

#ifdef __cplusplus
}
#endif

#endif  // __APPLE__

/**
 * Stubs for EP functions from OrtApis that are implemented in provider_bridge_ort.cc in a full build.
 * That file is not included in a minimal build.
 */
#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptionsV2* cuda_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(cuda_options);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::CreateCUDAProviderOptions, _Outptr_ OrtCUDAProviderOptionsV2** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptions,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsAsString, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptionsWithValue,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _In_ void* value) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsByName,
                    _In_ const OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API(void, OrtApis::ReleaseCUDAProviderOptions, _Frees_ptr_opt_ OrtCUDAProviderOptionsV2* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("OpenVINO");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO_V2,
                    _In_ OrtSessionOptions* options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("OpenVINO");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptions,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsAsString,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptionsWithValue,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _In_ void* value) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsByName,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API(void, OrtApis::ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_MIGraphX,
                    _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(migraphx_options);
  return CreateNotEnabledStatus("MIGraphX");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CANN,
                    _In_ OrtSessionOptions* options, _In_ const OrtCANNProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::CreateCANNProviderOptions, _Outptr_ OrtCANNProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCANNProviderOptions,
                    _Inout_ OrtCANNProviderOptions* cann_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::GetCANNProviderOptionsAsString,
                    _In_ const OrtCANNProviderOptions* cann_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CANN execution provider is not enabled in this build.");
}

ORT_API(void, OrtApis::ReleaseCANNProviderOptions, _Frees_ptr_opt_ OrtCANNProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl,
                    _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(dnnl_options);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::CreateDnnlProviderOptions, _Outptr_ OrtDnnlProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateDnnlProviderOptions,
                    _Inout_ OrtDnnlProviderOptions* dnnl_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::GetDnnlProviderOptionsAsString,
                    _In_ const OrtDnnlProviderOptions* dnnl_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "Dnnl execution provider is not enabled in this build.");
}

ORT_API(void, OrtApis::ReleaseDnnlProviderOptions, _Frees_ptr_opt_ OrtDnnlProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_VitisAI,
                    _In_ OrtSessionOptions* options, _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("VitisAI");
}
#endif

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::CreateROCMProviderOptions, _Outptr_ OrtROCMProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateROCMProviderOptions,
                    _Inout_ OrtROCMProviderOptions* rocm_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::GetROCMProviderOptionsAsString,
                    _In_ const OrtROCMProviderOptions* rocm_options,
                    _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API(void, OrtApis::ReleaseROCMProviderOptions, _Frees_ptr_opt_ OrtROCMProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}
