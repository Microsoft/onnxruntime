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
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif
namespace ONNX_NAMESPACE {
class TensorShapeProto;
class TensorProto;
std::ostream& operator<<(std::ostream& out, const TensorShapeProto& shape_proto);
std::ostream& operator<<(std::ostream& out, const TensorProto& tensor_proto);
}  // namespace ONNX_NAMESPACE

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
class Tensor;
struct KernelCreateInfo;

namespace logging {
class Logger;
}

namespace utils {
void* DefaultAlloc(size_t size);
void DefaultFree(void* p);

/// <summary>
// Do the placement new for strings on pre-allocated buffer
// `elements` times.
/// </summary>
/// <param name="p_data"></param>
/// <param name="elements"></param>
void ConstructStrings(void* p_data, int64_t elements);

/// <summary>
/// The function uses an allocator (not null)
/// and releases the supplied buffer. If is_string is true
/// then the function assumes the buffer contains instances
/// of std::string and calls ~string() on each instance `elements`
/// times total.
/// </summary>
/// <param name="allocator"></param>
/// <param name="is_string"></param>
/// <param name="p_data"></param>
/// <param name="elements"></param>
void ReleaseTensorBuffer(const AllocatorPtr& allocator, bool is_string,
                         void* p_data, int64_t elements);

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info);

// EP used for internal testing. We define it here as it's used in ProviderIsCpuBased, but we don't want
// it to be in the public header include/onnxruntime/core/graph/constants.h as it's purely internal.
constexpr const char* kInternalTestingExecutionProvider = "InternalTestingExecutionProvider";

// return true if the execution provider is CPU based (meaning no copies to device are required)
bool ProviderIsCpuBased(const std::string& provider_type);

common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue);

// Searches the allocation plan from the session_state to find the OrtMemoryInfo for the value 'name'.
const OrtMemoryInfo& FindMemoryInfoForValue(const SessionState& session_state,
                                            const std::string& name);

// Initialize the feed and fetch copy info using session_state.
// Determines the device that each graph input that will be fed will be consumed on,
// and the device that each graph output that will be fetched will be created on.
common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager);

// Finalize the feed and fetch copy info using session_state and the device and location information from the feeds
// and fetches that will be used in graph execution.
void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtDevice>& feed_locations,
                               const std::vector<const OrtMemoryInfo*>& fetch_alloc_info);

// Execute the main graph. The feed_fetches_manager will be finalized based on the provided feeds and fetches.
common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger,
                            bool only_execute_path_to_fetches = false);

#ifdef ENABLE_TRAINING
common::Status ExecutePartialGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                   const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                   const logging::Logger& logger, PartialGraphExecutionState& state);
#endif

// Execute a subgraph. The feeds_fetches_manager should have been finalized prior to calling this function.
// See IControlFlowNode::SetupSubgraphExecutionInfo usage in the control flow kernels.
common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger);

bool IsInputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index);

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

int32_t ONNXTensorElementDataTypeToProtoTensorType(ONNXTensorElementDataType);

#ifdef ENABLE_TRAINING
common::Status VerifyInputTensorsAllocatedContiguously(OpKernelContext* context);
#endif

}  // namespace utils
}  // namespace onnxruntime
