// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "test/util/include/default_providers.h"

#include <memory>
#include <string>

#include "core/framework/session_options.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif
#if defined(USE_WEBGPU)
#include "core/graph/constants.h"
#include "core/session/abi_session_options_impl.h"
#endif
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/providers.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"

namespace onnxruntime {

namespace test {

namespace {

#if defined(USE_CUDA) && defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
void AddCudaPluginOption(ConfigOptions& config_options, const char* key, std::string value) {
  ORT_THROW_IF_ERROR(config_options.AddConfigEntry(key, value.c_str()));
}

std::unique_ptr<IExecutionProvider> CudaPluginExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options) {
  auto ep_name = dynamic_plugin_ep_infra::GetEpName();
  if (!ep_name.has_value()) {
    return nullptr;
  }

  ORT_ENFORCE(*ep_name == dynamic_plugin_ep_infra::kCudaExecutionProviderPluginName,
              "Dynamic plugin EP is not the CUDA EP. Expected \"", dynamic_plugin_ep_infra::kCudaExecutionProviderPluginName,
              "\", got \"", *ep_name, "\"");

  ConfigOptions config_options{};
  if (provider_options != nullptr) {
    AddCudaPluginOption(config_options, "do_copy_in_default_stream", std::to_string(provider_options->do_copy_in_default_stream));
    AddCudaPluginOption(config_options, "cudnn_conv_use_max_workspace", std::to_string(provider_options->cudnn_conv_use_max_workspace));
    AddCudaPluginOption(config_options, "cudnn_conv1d_pad_to_nc1d", std::to_string(provider_options->cudnn_conv1d_pad_to_nc1d));
    AddCudaPluginOption(config_options, "enable_cuda_graph", std::to_string(provider_options->enable_cuda_graph));
    AddCudaPluginOption(config_options, "prefer_nhwc", std::to_string(provider_options->prefer_nhwc));
    AddCudaPluginOption(config_options, "use_ep_level_unified_stream", std::to_string(provider_options->use_ep_level_unified_stream));
    AddCudaPluginOption(config_options, "use_tf32", std::to_string(provider_options->use_tf32));
    AddCudaPluginOption(config_options, "fuse_conv_bias", std::to_string(provider_options->fuse_conv_bias));
    AddCudaPluginOption(config_options, "sdpa_kernel", std::to_string(provider_options->sdpa_kernel));
  }

  return dynamic_plugin_ep_infra::MakeEp(nullptr, &config_options);
}
#endif

}  // namespace

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CPUProviderFactoryCreator::Create(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  OrtTensorRTProviderOptions params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0};
  if (auto factory = TensorrtProviderFactoryCreator::Create(&params))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptionsV2* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider() {
#ifdef USE_MIGRAPHX
  return MIGraphXProviderFactoryCreator::Create(ProviderOptions{})->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions* params) {
#ifdef USE_MIGRAPHX
  if (const auto factory = MIGraphXProviderFactoryCreator::Create(params); factory != nullptr)
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> OpenVINOExecutionProviderWithOptions(const ProviderOptions* params,
                                                                         const SessionOptions* session_options) {
#ifdef USE_OPENVINO
  return OpenVINOProviderFactoryCreator::Create(params, session_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
  ORT_UNUSED_PARAMETER(session_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  ProviderOptions provider_options_map;
  SessionOptions session_options;
  return OpenVINOProviderFactoryCreator::Create(&provider_options_map, &session_options)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  return CudaExecutionProviderWithOptions(&provider_options);
#else
  return nullptr;
#endif
}

#ifdef ENABLE_CUDA_NHWC_OPS
std::unique_ptr<IExecutionProvider> DefaultCudaNHWCExecutionProvider() {
#if defined(USE_CUDA)
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.prefer_nhwc = true;
  return CudaExecutionProviderWithOptions(&provider_options);
#else
  return nullptr;
#endif
}
#endif

std::unique_ptr<IExecutionProvider> CudaExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options) {
#ifdef USE_CUDA
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
  return CudaPluginExecutionProviderWithOptions(provider_options);
#else
  if (auto factory = CudaProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
  return nullptr;
#endif
#else
  ORT_UNUSED_PARAMETER(provider_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider() {
#ifdef USE_DNNL
  OrtDnnlProviderOptions dnnl_options;
  dnnl_options.use_arena = 1;
  dnnl_options.threadpool_args = nullptr;
  if (auto factory = DnnlProviderFactoryCreator::Create(&dnnl_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DnnlExecutionProviderWithOptions(const OrtDnnlProviderOptions* provider_options) {
#ifdef USE_DNNL
  if (auto factory = DnnlProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(provider_options);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
// The NNAPI EP uses a stub implementation on non-Android platforms so cannot be used to execute a model.
// Manually append an NNAPI EP instance to the session to unit test the GetCapability and Compile implementation.
#if defined(USE_NNAPI) && defined(__ANDROID__)
  return NnapiProviderFactoryCreator::Create(0, {})->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultVSINPUExecutionProvider() {
#if defined(USE_VSINPU)
  return VSINPUProviderFactoryCreator::Create()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider() {
#ifdef USE_RKNPU
  return RknpuProviderFactoryCreator::Create()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_fast_math) {
#ifdef USE_ACL
  return ACLProviderFactoryCreator::Create(enable_fast_math)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_fast_math);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider(bool use_mlprogram) {
  // To manually test CoreML model generation on a non-macOS platform, comment out the `&& defined(__APPLE__)` below.
  // The test will create a model but execution of it will obviously fail.
#if defined(USE_COREML) && defined(__APPLE__)
  // We want to run UT on CPU only to get output value without losing precision
  auto option = ProviderOptions();
  option[kCoremlProviderOption_MLComputeUnits] = "CPUOnly";

  if (use_mlprogram) {
    option[kCoremlProviderOption_ModelFormat] = "MLProgram";
  }

  return CoreMLProviderFactoryCreator::Create(option)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(use_mlprogram);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultSnpeExecutionProvider() {
#if defined(USE_SNPE)
  ProviderOptions provider_options_map;
  return SNPEProviderFactoryCreator::Create(provider_options_map)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultQnnExecutionProvider() {
#ifdef USE_QNN
  ProviderOptions provider_options_map;
  // Limit to CPU backend for now. TODO: Enable HTP emulator
  std::string backend_path = "./libQnnCpu.so";
#if defined(_WIN32) || defined(_WIN64)
  backend_path = "./QnnCpu.dll";
#endif
  provider_options_map["backend_path"] = backend_path;
  return QNNProviderFactoryCreator::Create(provider_options_map, nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> QnnExecutionProviderWithOptions(const ProviderOptions& options,
                                                                    const SessionOptions* session_options) {
#ifdef USE_QNN
  return QNNProviderFactoryCreator::Create(options, session_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(session_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultXnnpackExecutionProvider() {
#ifdef USE_XNNPACK
  return XnnpackProviderFactoryCreator::Create(ProviderOptions(), nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultWebGpuExecutionProvider(bool is_nhwc) {
#if defined(USE_WEBGPU)
  ConfigOptions config_options{};

  // Helper to strip the EP prefix from config entry keys when building as a plugin EP.
  // The full key is like "ep.webgpuexecutionprovider.storageBufferCacheMode", and the
  // config entry expects just "storageBufferCacheMode" in the EP API build.
  auto normalize_config_key = [](const char* key) -> std::string {
#if defined(ORT_USE_EP_API_ADAPTERS)
    std::string normalized_key = key;
    std::string prefix = OrtSessionOptions::GetProviderOptionPrefix(kWebGpuExecutionProvider);
    if (normalized_key.starts_with(prefix)) {
      normalized_key.erase(0, prefix.length());
    }
    return normalized_key;
#else
    return key;
#endif
  };

  // Disable storage buffer cache
  ORT_THROW_IF_ERROR(config_options.AddConfigEntry(normalize_config_key(webgpu::options::kStorageBufferCacheMode).c_str(),
                                                   webgpu::options::kBufferCacheMode_Disabled));
  // Enable int64 so the generic (EP-agnostic) int64 operator tests exercise the WebGPU int64
  // kernels instead of silently falling back to CPU. Kept on by default for the test harness so
  // operator tests don't have to build a bespoke WebGPU EP just to reach the int64 path.
  //
  // NOTE: this enables int64 for ALL WebGPU tests, not just the binary elementwise ops.
  // TODO: migrate the other int64-gated ops' WebGPU-only int64 tests to EP-agnostic form too.
  ORT_THROW_IF_ERROR(config_options.AddConfigEntry(normalize_config_key(webgpu::options::kEnableInt64).c_str(),
                                                   webgpu::options::kEnableInt64_ON));
  if (!is_nhwc) {
    // Enable NCHW support
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(normalize_config_key(webgpu::options::kPreferredLayout).c_str(),
                                                     webgpu::options::kPreferredLayout_NCHW));
  }

  return WebGpuExecutionProviderWithOptions(config_options);
#else
  ORT_UNUSED_PARAMETER(is_nhwc);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithOptions(const ConfigOptions& config_options) {
#if defined(USE_WEBGPU)
#if defined(ORT_USE_EP_API_ADAPTERS)
  ConfigOptions normalized_config_options{};
  const std::string prefix = OrtSessionOptions::GetProviderOptionPrefix(kWebGpuExecutionProvider);
  for (const auto& [key, value] : config_options.GetConfigOptionsMap()) {
    std::string normalized_key = key;
    if (normalized_key.starts_with(prefix)) {
      normalized_key.erase(0, prefix.length());
    }
    ORT_THROW_IF_ERROR(normalized_config_options.AddConfigEntry(normalized_key.c_str(), value.c_str()));
  }

  // Return nullptr (rather than throwing) when the dynamic plugin EP is uninitialized.
  // Tests interpret nullptr as "WebGPU EP unavailable" and skip themselves, which matches
  // the behavior of the non-plugin code path below when USE_WEBGPU is undefined.
  //
  // If the dynamic plugin EP is initialized to a different EP, fail fast. Many call sites pass
  // this helper directly into ConfigEp/RegisterExecutionProvider and do not null-check, so
  // silently returning nullptr here can lead to confusing downstream failures.
  auto ep_name = dynamic_plugin_ep_infra::GetEpName();
  if (!ep_name.has_value()) {
    return nullptr;
  }
  ORT_ENFORCE(*ep_name == kWebGpuExecutionProvider,
              "Dynamic plugin EP is not the WebGPU EP. Expected \"", kWebGpuExecutionProvider,
              "\", got \"", *ep_name, "\"");
  return dynamic_plugin_ep_infra::MakeEp(nullptr, &normalized_config_options);
#else
  return WebGpuProviderFactoryCreator::Create(config_options)->CreateProvider();
#endif
#else
  ORT_UNUSED_PARAMETER(config_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithTestOptions(
    const WebGpuExecutionProviderTestOptions& test_options) {
  return WebGpuExecutionProviderWithTestOptions(ConfigOptions{}, test_options);
}

std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithTestOptions(
    const ConfigOptions& config_options, const WebGpuExecutionProviderTestOptions& test_options) {
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  return WebGpuProviderFactoryCreator::CreateForTesting(config_options, test_options)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(config_options);
  ORT_UNUSED_PARAMETER(test_options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCannExecutionProvider() {
#ifdef USE_CANN
  OrtCANNProviderOptions provider_options{};
  if (auto factory = CannProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultDmlExecutionProvider() {
#ifdef USE_DML
  ConfigOptions config_options{};
  if (auto factory = DMLProviderFactoryCreator::CreateFromDeviceOptions(config_options, nullptr, false, false)) {
    return factory->CreateProvider();
  }
#endif
  return nullptr;
}

}  // namespace test
}  // namespace onnxruntime
