// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

// Node argument definition, for both input and output,
// including arg name, arg type (contains both type and shape).
//
// Design Question: in my opinion, shape should not be part of type.
// We may align the protobuf design with our operator registry interface,
// which has type specified for each operator, but no shape. Well, shape
// should be inferred with a separate shape inference function given
// input shapes, or input tensor data sometimes.
// With shape as part of type (current protobuf design),
// 1) we'll have to split the "TypeProto" into type and shape in this internal
// representation interface so that it could be easily used when doing type
// inference and matching with operator registry.
// 2) SetType should be always called before SetShape, otherwise, SetShape()
// will fail. Because shape is located in a TypeProto.
// Thoughts?
//

/**
@class NodeArg
Class representing a data type that is input or output for a Node, including the shape if it is a Tensor.
*/
class NodeArg {
 public:
  /**
  Construct a new NodeArg.
  @param name The name to use.
  @param p_arg_type Optional TypeProto specifying type and shape information.
  */
  NodeArg(const std::string& name,
          const ONNX_NAMESPACE::TypeProto* p_arg_type);

  NodeArg(NodeArg&& other) = default;

  /** Gets the name. */
  const std::string& Name() const noexcept;

  /** Gets the data type. */
  ONNX_NAMESPACE::DataType Type() const noexcept;

  /** Gets the TypeProto 
  @returns TypeProto if type is set. nullptr otherwise. */
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept;

  /** Gets the shape if NodeArg is for a Tensor.
  @returns TensorShapeProto if shape is set. nullptr if there's no shape specified. */
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const;

  /** Sets the shape. Type must have been previously set.
  @remarks Shape can only be set after setting type since shape information now is part of TypeProto. */
  void SetShape(const ONNX_NAMESPACE::TensorShapeProto& shape);

  /** Validate and merge type [and shape] info from input_type.
  @returns Success unless there is existing type or shape info that can't be cleanly updated. */
  common::Status UpdateTypeAndShape(const ONNX_NAMESPACE::TypeProto& input_type);

  /** Validate and merge type [and shape] info from node_arg.
  @returns Success unless there is existing type or shape info that can't be cleanly updated. */
  common::Status UpdateTypeAndShape(const NodeArg& node_arg);

  /** Gets this NodeArg as a ValueInfoProto. */
  const NodeArgInfo& ToProto() const noexcept { return node_arg_info_; }

  /** Gets a flag indicating whether this NodeArg exists or not.
  Optional inputs are allowed in ONNX and an empty #Name represents a non-existent input argument. */
  bool Exists() const noexcept;

 private:
  ONNXRUNTIME_DISALLOW_COPY_AND_ASSIGNMENT(NodeArg);
  friend class Graph;

  void SetType(ONNX_NAMESPACE::DataType p_type);
  void SetType(const ONNX_NAMESPACE::TypeProto& type_proto);

  NodeArg& operator=(NodeArg&& other) = delete;

  // Node arg PType.
  ONNX_NAMESPACE::DataType type_;

  // Node arg name, type and shape.
  NodeArgInfo node_arg_info_;

  // Flag indicates whether <*this> node arg exists or not.
  bool exists_;
};
}  // namespace onnxruntime
