// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/get_execution_providers.h"

#include "core/graph/constants.h"
#include "core/common/common.h"

#include <string_view>

namespace onnxruntime {

namespace {
struct ProviderInfo {
  std::string_view name;
  bool available;
};

// all providers ordered by default priority from highest to lowest
// kCpuExecutionProvider should always be last
constexpr ProviderInfo kProvidersInPriorityOrder[] =
    {
        {
            kTensorrtExecutionProvider,
#ifdef USE_TENSORRT
            true,
#else
            false,
#endif
        },
        {
            kCudaExecutionProvider,
#ifdef USE_CUDA
            true,
#else
            false,
#endif
        },
        {
            kMIGraphXExecutionProvider,
#ifdef USE_MIGRAPHX
            true,
#else
            false,
#endif
        },
        {
            kOpenVINOExecutionProvider,
#ifdef USE_OPENVINO
            true,
#else
            false,
#endif
        },
        {
            kDnnlExecutionProvider,
#ifdef USE_DNNL
            true,
#else
            false,
#endif
        },
        {
            kVitisAIExecutionProvider,
#ifdef USE_VITISAI
            true,
#else
            false,
#endif
        },
        {
            kQnnExecutionProvider,
#ifdef USE_QNN
            true,
#else
            false,
#endif
        },
        {
            kNnapiExecutionProvider,
#ifdef USE_NNAPI
            true,
#else
            false,
#endif
        },
        {
            kVSINPUExecutionProvider,
#ifdef USE_VSINPU
            true,
#else
            false,
#endif
        },
        {
            kJsExecutionProvider,
#ifdef USE_JSEP
            true,
#else
            false,
#endif
        },
        {
            kCoreMLExecutionProvider,
#ifdef USE_COREML
            true,
#else
            false,
#endif
        },
        {
            kAclExecutionProvider,
#ifdef USE_ACL
            true,
#else
            false,
#endif
        },
        {
            kDmlExecutionProvider,
#ifdef USE_DML
            true,
#else
            false,
#endif
        },
        {
            kRknpuExecutionProvider,
#ifdef USE_RKNPU
            true,
#else
            false,
#endif
        },
        {
            kWebNNExecutionProvider,
#ifdef USE_WEBNN
            true,
#else
            false,
#endif
        },
        {
            kWebGpuExecutionProvider,
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
            true,
#else
            false,
#endif
        },
        {
            kXnnpackExecutionProvider,
#ifdef USE_XNNPACK
            true,
#else
            false,
#endif
        },
        {
            kCannExecutionProvider,
#ifdef USE_CANN
            true,
#else
            false,
#endif
        },
        {
            kAzureExecutionProvider,
#ifdef USE_AZURE
            true,
#else
            false,
#endif
        },
        {kCpuExecutionProvider, true},  // kCpuExecutionProvider is always last
};

constexpr size_t kAllExecutionProvidersCount = sizeof(kProvidersInPriorityOrder) / sizeof(ProviderInfo);

}  // namespace

const std::vector<std::string>& GetAllExecutionProviderNames() {
  static const auto all_execution_providers = []() {
    std::vector<std::string> result{};
    result.reserve(kAllExecutionProvidersCount);
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      result.push_back(std::string(provider.name));
    }
    return result;
  }();

  return all_execution_providers;
}

const std::vector<std::string>& GetAvailableExecutionProviderNames() {
  static const auto available_execution_providers = []() {
    std::vector<std::string> result{};
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      if (provider.available) {
        result.push_back(std::string(provider.name));
      }
    }
    return result;
  }();

  return available_execution_providers;
}

}  // namespace onnxruntime
