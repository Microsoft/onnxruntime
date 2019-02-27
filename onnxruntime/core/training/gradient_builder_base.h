// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace training {

struct ArgDef {
  ArgDef(std::string name, const ONNX_NAMESPACE::TypeProto* type) : name(name), type_proto(type) {}
  std::string name;
  const ONNX_NAMESPACE::TypeProto* type_proto;
};

struct OpDef {
  OpDef(const std::string& op_type,
        const std::vector<ArgDef>& input_args,
        const std::vector<ArgDef>& output_args) : op_type(op_type),
                                                  input_args(input_args),
                                                  output_args(output_args){};
  OpDef(const std::string& op_type,
        const std::vector<ArgDef>& input_args,
        const std::vector<ArgDef>& output_args,
        const NodeAttributes& attr) : op_type(op_type),
                                      input_args(input_args),
                                      output_args(output_args),
                                      attr(attr){};

  OpDef(const std::string& op_type,
        const std::string& node_name,
        const std::vector<ArgDef>& input_args,
        const std::vector<ArgDef>& output_args) : op_type(op_type),
                                                  node_name(node_name),
                                                  input_args(input_args),
                                                  output_args(output_args){};

  OpDef(const std::string& op_type,
        const std::string& node_name,
        const std::vector<ArgDef>& input_args,
        const std::vector<ArgDef>& output_args,
        const NodeAttributes& attr) : op_type(op_type),
                                      node_name(node_name),
                                      input_args(input_args),
                                      output_args(output_args),
                                      attr(attr){};

  std::string op_type;
  std::string node_name;
  std::vector<ArgDef> input_args;
  std::vector<ArgDef> output_args;
  NodeAttributes attr;
};

class GradientBuilderBase {
 public:
  GradientBuilderBase(
      const Node* node,
      const std::unordered_set<std::string>& gradient_inputs,
      const std::unordered_set<std::string>& gradient_outputs)
      : node_(node), gradient_inputs_(gradient_inputs), gradient_outputs_(gradient_outputs) {
    unique_node_prefix_ = CreateUniqueNodePrefix();
  }

  // TODO: make this protected? Currently, compiler failure prevents it
  virtual std::vector<OpDef> GetGradientDefs() = 0;

 protected:
  ArgDef I(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return ArgDef(node_->InputDefs()[i]->Name(), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef GI(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return ArgDef(GradientName(node_->InputDefs()[i]->Name()), node_->InputDefs()[i]->TypeAsProto());
  }

  ArgDef GO(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->OutputDefs().size());
    return ArgDef(GradientName(node_->OutputDefs()[i]->Name()), node_->OutputDefs()[i]->TypeAsProto());
  }

  ArgDef IA(const std::string& argSuffix) {
    return ArgDef(Name(argSuffix), nullptr);
  }

  int GetSrcNodeOutputSize() {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->OutputDefs().size();
  }

  // returns true if the input at index i of the node_ requires gradient
  bool IsGradientRequiredForSrcNodeInput(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->InputDefs().size());
    return gradient_outputs_.find(node_->InputDefs()[i]->Name()) != gradient_outputs_.end();
  }

  // returns true if the output at index i of the node_ has a gradient
  bool IsGradientAvailableForSrcNodeOutput(const int i) {
    ORT_ENFORCE(i >= 0 && i < node_->OutputDefs().size());
    return gradient_inputs_.find(node_->OutputDefs()[i]->Name()) != gradient_inputs_.end();
  }

  std::string Name(const std::string& name) {
    return unique_node_prefix_ + name;
  }

  const NodeAttributes& SrcNodeAttributes() {
    return node_->GetAttributes();
  }

 private:
  friend class GradientGraphBuilder;

  // Utility functions for gradient name computation. We don't expose them
  // in order to discourage the use of such names explicitly.
  static std::string GradientName(const std::string& name) {
    return name + "_grad";
  }

  std::string CreateUniqueNodePrefix() {
    ORT_ENFORCE(node_ != nullptr);
    auto name = node_->Name();
    std::stringstream unique_prefix;
    if (!name.empty()) {
      unique_prefix << name << "_";
    } else {
      unique_prefix << node_->OpType() << "_";
    }
    unique_prefix << node_->Index() << "_";
    return unique_prefix.str();
  }

  const Node* node_;
  std::string unique_node_prefix_;

  // contains set of output arg names of node_ which is provided as gradient input to the bw node
  std::unordered_set<std::string> gradient_inputs_;

  // contains set of input arg names of node_ which requires gradient
  std::unordered_set<std::string> gradient_outputs_;
};

class EmptyGradientBuilder : GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  virtual std::vector<OpDef> GetGradientDefs() {
    return std::vector<OpDef>();
  }
};

}  // namespace training
}  // namespace onnxruntime
