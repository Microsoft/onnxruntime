// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

// This is the minimal Android API Level required by ORT NNAPI EP to run
// ORT running on any host system with Android API level less than this will fall back to CPU EP
#ifndef ORT_NNAPI_MIN_API_LEVEL
#define ORT_NNAPI_MIN_API_LEVEL ANEURALNETWORKS_FEATURE_LEVEL_1
#endif

// This is the maximum Android API level supported in the ort model conversion for NNAPI EP
// Note: This is only for running NNAPI for ort format model conversion on non-Android system since we cannot
//       get the actually Android system version.
//       If running on an actual Android system, this value will be ignored
#ifndef ORT_NNAPI_MAX_SUPPORTED_API_LEVEL
#define ORT_NNAPI_MAX_SUPPORTED_API_LEVEL 30
#endif

namespace onnxruntime {

using Shape = std::vector<uint32_t>;
using InitializerMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto&>;

class Node;
class NodeArg;
class GraphViewer;

namespace nnapi {

class IOpSupportChecker;
struct OpSupportCheckParams;

#define THROW_ON_ERROR(val)                  \
  {                                          \
    const auto ret = (val);                  \
    ORT_ENFORCE(                             \
        ret == ANEURALNETWORKS_NO_ERROR,     \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define THROW_ON_ERROR_WITH_NOTE(val, note)                \
  {                                                        \
    const auto ret = (val);                                \
    ORT_ENFORCE(                                           \
        ret == ANEURALNETWORKS_NO_ERROR,                   \
        "ResultCode: ", GetErrorCause(ret), ", ", (note)); \
  }

#define RETURN_STATUS_ON_ERROR(val)          \
  {                                          \
    const auto ret = (val);                  \
    ORT_RETURN_IF_NOT(                       \
        ret == ANEURALNETWORKS_NO_ERROR,     \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define RETURN_STATUS_ON_ERROR_WITH_NOTE(val, note)        \
  {                                                        \
    const auto ret = (val);                                \
    ORT_RETURN_IF_NOT(                                     \
        ret == ANEURALNETWORKS_NO_ERROR,                   \
        "ResultCode: ", GetErrorCause(ret), ", ", (note)); \
  }

std::string GetErrorCause(int error_code);

enum class QLinearOpType : uint8_t {
  Unknown,  // Unknown or not a linear quantized op
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  // Not yet supported
  // QLinearMul,
  // QLinearReduceMean,
};

enum class ConvType : uint8_t {
  Regular,
  Depthwise,
  Grouped,
};

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node);

// Return the type of the conv ops,
// This function assumes the input is a 2d conv node
ConvType GetConvType(const onnxruntime::Node& node, const InitializedTensorSet& initializers);

// This qlinear op is an operator takes 2 inputs and produces 1 output
// Such as QLinearConv, QLinearMatMul, QLinearAdd, ...
bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type);

// Check if a qlinear unary op has valid inputs, Qlinear[Sigmoid/AveragePool]
bool HasValidUnaryOpQuantizedInputs(const Node& node);
// Check if a qlinear binary op has valid inputs, Qlinear[Conv/MatMul/Add]
bool HasValidBinaryOpQuantizedInputs(const Node& node);
// Check if a qlinear op has valid scales for given indices
bool HasValidQuantizationScales(const InitializedTensorSet& initializers, const Node& node,
                                const std::vector<size_t>& indices, const OpSupportCheckParams& params);
// Check if a qlinear op has valid zero points for given indices
bool HasValidQuantizationZeroPoints(const InitializedTensorSet& initializers, const Node& node,
                                    const std::vector<size_t>& indices);

float GetQuantizationScale(const InitializedTensorSet& initializers, const Node& node, size_t idx);

common::Status GetQuantizationZeroPoint(const InitializedTensorSet& initializers,
                                        const Node& node, size_t idx, int32_t& zero_point) ORT_MUST_USE_RESULT;

// Get Shape/Type of a NodeArg
// TODO, move to shared_utils
bool GetShape(const NodeArg& node_arg, Shape& shape);
bool GetType(const NodeArg& node_arg, int32_t& type);

// Get the output shape of Flatten Op
void GetFlattenOutputShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2);

// If a node is supported by NNAPI
bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const OpSupportCheckParams& params);

// If a node is supported by NNAPI in a partition node group
// `node_outputs_in_group` is the set of the output names of the nodes added to this group so far
bool IsNodeSupportedInGroup(const Node& node, const GraphViewer& graph_viewer,
                            const OpSupportCheckParams& params,
                            const std::unordered_set<std::string>& node_outputs_in_group);

// If a graph input is supported by NNAPI
bool IsInputSupported(const NodeArg& input, const std::string& parent_name);

// If an NNAPI partition node group is valid
bool IsValidSupportedNodeGroup(const std::vector<const Node*>& supported_node_group);

// Get string representation of a Shape
std::string Shape2String(const std::vector<uint32_t>& shape);

// Check the given input is an initializer tensor
bool CheckIsInitializer(const InitializedTensorSet& initializers, const Node& node,
                        size_t index, const char* input_name) ORT_MUST_USE_RESULT;

}  // namespace nnapi
}  // namespace onnxruntime
