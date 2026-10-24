// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/session_state.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
class ExecutionProviders;
struct FeedsFetchesInfo;
class FeedsFetchesManager;
struct MLValueCopyInfo;
class Graph;
class KernelDef;
class KernelRegistryManager;
class IExecutionProvider;
class Node;
#if !defined(ORT_MINIMAL_BUILD)
#endif
class Tensor;
struct KernelCreateInfo;
#ifdef ENABLE_TRAINING
struct PartialGraphExecutionState;
typedef InlinedHashMap<std::string, OrtValue> OrtValueCache;
typedef std::shared_ptr<OrtValueCache> OrtValueCachePtr;
#endif

namespace logging {
class Logger;
}

namespace utils {
/// <summary>
// Do the placement new for strings on pre-allocated buffer
// `elements` times.
/// </summary>
/// <param name="p_data"></param>
/// <param name="elements"></param>
void ConstructStrings(void* p_data, int64_t elements);

/// <summary>
/// Destroy std::string objects in the contiquous chunk of memory
/// by explicitly invoking ~string();
/// </summary>
/// <param name="p_data"></param>
/// <param name="elements"></param>
void DestroyStrings(void* p_data, int64_t elements);

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info);

// return true if the execution provider is CPU based (meaning no copies to device are required)
bool ProviderIsCpuBased(const IExecutionProvider& provider);

bool IsMemcpyNode(const Node& node);

// Returns true if src memory can satisfy tgt's requirements without a data copy.
// HOST_ACCESSIBLE -> DEFAULT is valid (device can access HOST_ACCESSIBLE memory directly).
// DEFAULT -> HOST_ACCESSIBLE is NOT valid (CPU cannot read device-only memory).
bool CanSourceSatisfyTarget(const OrtDevice& src, const OrtDevice& tgt);

common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue);

// Searches the allocation plan from the session_state to find the OrtMemoryInfo for the value 'name'.
const OrtDevice& FindDeviceForValue(const SessionState& session_state, std::string_view name);

// Initialize the feed and fetch copy info using session_state.
// Determines the device that each graph input that will be fed will be consumed on,
// and the device that each graph output that will be fetched will be created on.
common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager);

// Finalize the feed and fetch copy info using session_state and the device and location information from the feeds
// and fetches that will be used in graph execution.
void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtDevice> feed_locations,
                               gsl::span<const OrtDevice* const> fetch_alloc_info);

// Execute the main graph. The feed_fetches_manager will be finalized based on the provided feeds and fetches.
common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                            DeviceStreamCollectionHolder& device_stream_collection_holder,
#endif
                            bool only_execute_path_to_fetches = false,
                            Stream* parent_stream = nullptr,
                            profiling::Profiler* run_profiler = nullptr);

common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const RunOptions& run_options,
#ifdef ORT_ENABLE_STREAM
                            DeviceStreamCollectionHolder& device_stream_collection_holder,
#endif
                            const logging::Logger& logger,
                            profiling::Profiler* run_profiler = nullptr);

#ifdef ENABLE_TRAINING
common::Status ExecutePartialGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                   std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                   const logging::Logger& logger, PartialGraphExecutionState& state,
                                   const OrtValueCachePtr& cache,
                                   const bool& terminate_flag,
                                   int32_t partial_graph_index,
                                   Stream* parent_stream);
#endif

// Execute a subgraph. The feeds_fetches_manager should have been finalized prior to calling this function.
// See IControlFlowNode::SetupSubgraphExecutionInfo usage in the control flow kernels.
common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger,
                               Stream* parent_stream,
                               /*when this is enabled, we will sync the parent stream to make sure the subgraph fetches
                               is complete. this is mainly used when the parent kernel depends on the CPU value of the
                               subgraph fetches, i.e. the loop condition*/
                               bool sync_subgraph_fetches = false,
                               profiling::Profiler* run_profiler = nullptr);

bool IsInputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index);
bool IsOutputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index);

template <typename T>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<bool>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<std::string>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<float>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<double>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<MLFloat16>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<BFloat16>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int8_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint8_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int16_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint16_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int32_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint32_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int64_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint64_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E4M3FN>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E4M3FNUZ>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E5M2>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E5M2FNUZ>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E8M0>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0;
}

#endif

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float6E2M3>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT6E2M3;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float6E3M2>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT6E3M2;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Int4x2>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<UInt4x2>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Int2x4>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2;
}

template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<UInt2x4>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2;
}

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float4E2M1x2>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1;
}
#endif

int32_t ONNXTensorElementDataTypeToProtoTensorType(ONNXTensorElementDataType);

#ifdef ENABLE_TRAINING
common::Status VerifyInputTensorsAllocatedContiguously(OpKernelContext* context);
#endif

}  // namespace utils
}  // namespace onnxruntime
