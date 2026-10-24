// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {

enum class WeightLoadAccelerationMode {
  Off,
  Preferred,
  Required,
};

constexpr bool IsWeightLoadAccelerationEnabled(WeightLoadAccelerationMode mode) {
  return mode != WeightLoadAccelerationMode::Off;
}

constexpr bool IsWeightLoadAccelerationRequired(WeightLoadAccelerationMode mode) {
  return mode == WeightLoadAccelerationMode::Required;
}

namespace options {

// The following are the options that can be set in the WebGPU provider options.

constexpr const char* kPreferredLayout = "ep.webgpuexecutionprovider.preferredLayout";
constexpr const char* kEnableGraphCapture = "ep.webgpuexecutionprovider.enableGraphCapture";
constexpr const char* kWeightLoadAcceleration = "ep.webgpuexecutionprovider.weightLoadAcceleration";
// Per-session native D3D12 weight-loading choice: "1" (default) permits UMA
// final-buffer reads; "0" keeps the accelerated upload staging/copy path.
// Has no effect when weightLoadAcceleration is off or on other backends.
constexpr const char* kEnableUMAWeightLoading = "ep.webgpuexecutionprovider.enableUMAWeightLoading";
// Number of generations of buffers to retain in the per-session pool for reuse
// across captured-graph lifetimes. 0 disables pooling. Default 1 caches one
// generator's worth of intermediate buffers.
constexpr const char* kSessionBufferPoolGenerations = "ep.webgpuexecutionprovider.sessionBufferPoolGenerations";
constexpr const char* kEnableInt64 = "ep.webgpuexecutionprovider.enableInt64";
constexpr const char* kMultiRotaryCacheConcatOffset = "ep.webgpuexecutionprovider.multiRotaryCacheConcatOffset";
constexpr const char* kKvCacheQuantizationBits = "ep.webgpuexecutionprovider.kvCacheQuantizationBits";
// Accumulate the dot products of MatMul, Gemm and MatMulNBits kernels in f32 instead of in the output element
// type. The input and weight tensors keep their own type, so global memory traffic is identical
// either way. Enabling it avoids saturating the f16 maximum (65504) when partial sums along K grow
// large, at the cost of registers and shared memory.
// It is not only the accumulator registers: where a fused kernel computes its epilogue on the
// accumulators, that epilogue runs in the same precision. This applies to the fused MLP decode fast
// path, which keeps the bias add, the SiLU and the gate/up product in f32 and rounds once at the
// final store instead of after every step; that is why its test tolerance against the unfused
// reference is looser with the option on than with it off. Fused MLP shapes that fall back to
// ApplyUnfusedMlp materialize the gate and up tensors in the output element type before the
// activation, so their epilogue keeps rounding at the output precision either way.
// Unquantized MatMul/Gemm covers both generic and Intel subgroup kernels. FP16-only
// subgroup-matrix implementations are bypassed when fp32 accumulation is required.
constexpr const char* kEnableMatmulFp32Accumulation = "ep.webgpuexecutionprovider.enableMatmulFp32Accumulation";

constexpr const char* kDawnProcTable = "ep.webgpuexecutionprovider.dawnProcTable";

constexpr const char* kDawnBackendType = "ep.webgpuexecutionprovider.dawnBackendType";
constexpr const char* kPowerPreference = "ep.webgpuexecutionprovider.powerPreference";

constexpr const char* kDeviceId = "ep.webgpuexecutionprovider.deviceId";
constexpr const char* kAdapterIndex = "ep.webgpuexecutionprovider.adapterIndex";
constexpr const char* kWebGpuInstance = "ep.webgpuexecutionprovider.webgpuInstance";
constexpr const char* kWebGpuDevice = "ep.webgpuexecutionprovider.webgpuDevice";

constexpr const char* kStorageBufferCacheMode = "ep.webgpuexecutionprovider.storageBufferCacheMode";
constexpr const char* kUniformBufferCacheMode = "ep.webgpuexecutionprovider.uniformBufferCacheMode";
constexpr const char* kQueryResolveBufferCacheMode = "ep.webgpuexecutionprovider.queryResolveBufferCacheMode";
constexpr const char* kDefaultBufferCacheMode = "ep.webgpuexecutionprovider.defaultBufferCacheMode";

constexpr const char* kValidationMode = "ep.webgpuexecutionprovider.validationMode";
constexpr const char* kEnableRobustness = "ep.webgpuexecutionprovider.enableRobustness";

constexpr const char* kForceCpuNodeNames = "ep.webgpuexecutionprovider.forceCpuNodeNames";
constexpr const char* kEnablePIXCapture = "ep.webgpuexecutionprovider.enablePIXCapture";

constexpr const char* kPreserveDevice = "ep.webgpuexecutionprovider.preserveDevice";

constexpr const char* kMaxStorageBufferBindingSize = "ep.webgpuexecutionprovider.maxStorageBufferBindingSize";
constexpr const char* kMaxStorageBuffersPerShaderStage =
    "ep.webgpuexecutionprovider.maxStorageBuffersPerShaderStage";
// Valid range: 1-4096. Larger values are rejected to avoid excessive
// query buffer sizing and unpredictable memory/performance behavior.
constexpr const char* kMaxNumPendingDispatches = "ep.webgpuexecutionprovider.maxNumPendingDispatches";

// The following are the possible values for the provider options.

constexpr const char* kDawnBackendType_D3D12 = "D3D12";
constexpr const char* kDawnBackendType_Vulkan = "Vulkan";

constexpr const char* kPowerPreference_HighPerformance = "high-performance";
constexpr const char* kPowerPreference_LowPower = "low-power";

constexpr const char* kPreferredLayout_NCHW = "NCHW";
constexpr const char* kPreferredLayout_NHWC = "NHWC";

constexpr const char* kEnableGraphCapture_ON = "1";
constexpr const char* kEnableGraphCapture_OFF = "0";

// Selects weight loading per session. ORT-owned D3D12 contexts retain the
// native device and supported import features even when initially set to off,
// so later sessions can enable acceleration without replacing the device.
constexpr const char* kWeightLoadAcceleration_Off = "off";
constexpr const char* kWeightLoadAcceleration_Preferred = "preferred";
constexpr const char* kWeightLoadAcceleration_Required = "required";

constexpr const char* kEnableInt64_ON = "1";
constexpr const char* kEnableInt64_OFF = "0";

constexpr const char* kEnablePIXCapture_ON = "1";
constexpr const char* kEnablePIXCapture_OFF = "0";

constexpr const char* kPreserveDevice_ON = "1";
constexpr const char* kPreserveDevice_OFF = "0";

// kKvCacheQuantizationBits value is the number of quantization bits as a string.
// "0" disables quantization, "4" selects TurboQuant centroid indices, and "8" selects
// symmetric block quantization with offset-binary storage.
constexpr const char* kKvCacheQuantizationBits_OFF = "0";
constexpr const char* kKvCacheQuantizationBits_4Bit = "4";
constexpr const char* kKvCacheQuantizationBits_8Bit = "8";

constexpr const char* kEnableMatmulFp32Accumulation_ON = "1";
constexpr const char* kEnableMatmulFp32Accumulation_OFF = "0";

constexpr const char* kBufferCacheMode_Disabled = "disabled";
constexpr const char* kBufferCacheMode_LazyRelease = "lazyRelease";
constexpr const char* kBufferCacheMode_Simple = "simple";
constexpr const char* kBufferCacheMode_Bucket = "bucket";

constexpr const char* kValidationMode_Disabled = "disabled";
constexpr const char* kValidationMode_wgpuOnly = "wgpuOnly";
constexpr const char* kValidationMode_basic = "basic";
constexpr const char* kValidationMode_full = "full";

constexpr const char* kEnableRobustness_ON = "1";
constexpr const char* kEnableRobustness_OFF = "0";

}  // namespace options
}  // namespace webgpu
}  // namespace onnxruntime
