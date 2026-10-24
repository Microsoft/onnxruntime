// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.
#pragma once
#include "core/common/optional.h"
#include "core/providers/providers.h"
#include "core/providers/provider_factory_creators.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/framework/config_options.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

struct SessionOptions;

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML(uint32_t);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptionsV2* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(const OrtDnnlProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(const OrtMIGraphXProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(
    uint32_t flags, const optional<std::string>& partitioning_stop_ops_list);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VSINPU();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptionsV2* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cann(const OrtCANNProviderOptions* provider_options);

// EP for internal testing
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_InternalTesting(
    const std::unordered_set<std::string>& supported_ops);

namespace test {

// unique_ptr providers with default values for session registration
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();
#ifdef ENABLE_CUDA_NHWC_OPS
std::unique_ptr<IExecutionProvider> DefaultCudaNHWCExecutionProvider();
#endif
std::unique_ptr<IExecutionProvider> CudaExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options);
std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider();
std::unique_ptr<IExecutionProvider> DnnlExecutionProviderWithOptions(const OrtDnnlProviderOptions* provider_options);
// std::unique_ptr<IExecutionProvider> DefaultTvmExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();
std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions* params);
std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptionsV2* params);
std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider();
std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions* params);
std::unique_ptr<IExecutionProvider> OpenVINOExecutionProviderWithOptions(const ProviderOptions* params, const SessionOptions* session_options = nullptr);
std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultVSINPUExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_fast_math = false);
std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider(bool use_mlprogram = false);
std::unique_ptr<IExecutionProvider> DefaultSnpeExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultQnnExecutionProvider();
std::unique_ptr<IExecutionProvider> QnnExecutionProviderWithOptions(const ProviderOptions& options,
                                                                    const SessionOptions* session_options = nullptr);
std::unique_ptr<IExecutionProvider> DefaultXnnpackExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultWebGpuExecutionProvider(bool is_nhwc = true);
std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithOptions(const ConfigOptions& config_options);
std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithTestOptions(
    const WebGpuExecutionProviderTestOptions& test_options);
std::unique_ptr<IExecutionProvider> WebGpuExecutionProviderWithTestOptions(
    const ConfigOptions& config_options, const WebGpuExecutionProviderTestOptions& test_options);
std::unique_ptr<IExecutionProvider> DefaultCannExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultDmlExecutionProvider();

}  // namespace test
}  // namespace onnxruntime
