// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "core/util/math.h"
#include "core/graph/graph.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/gradient_config.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/graph/aten_op_gradient.h"
#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {

using Dimension = onnx::TensorShapeProto_Dimension;

void ComputeBroadcastBackwardAxes(
    const std::vector<Dimension>& A_dims,
    const std::vector<Dimension>& B_dims,
    std::vector<int64_t>* A_axes,
    std::vector<int64_t>* B_axes,
    const std::string& node_name = "");

void ComputeBroadcastBackwardAxesDynamic(const ArgDef& a,
                                         const ArgDef& b,
                                         const ArgDef& a_shape,
                                         const ArgDef& b_shape,
                                         const ArgDef* a_axes,
                                         const ArgDef* b_axes,
                                         std::vector<NodeDef>& output);

Status GetShape(const ArgDef& arg_def, std::vector<Dimension>& shape);

typedef std::vector<NodeDef> GradientDef;

class GradientBuilderBase {
 public:
  GradientBuilderBase(const GradientGraphConfiguration& gradient_graph_config,
                      Graph* graph,
                      const Node* node,
                      const std::unordered_set<std::string>& gradient_inputs,
                      const std::unordered_set<std::string>& gradient_outputs,
                      const logging::Logger& logger)
      : gradient_graph_config_(gradient_graph_config),
        graph_(graph),
        node_(node),
        gradient_inputs_(gradient_inputs),
        gradient_outputs_(gradient_outputs),
        logger_(logger) {
    unique_node_prefix_ = CreateUniqueNodePrefix();
  }

  virtual ~GradientBuilderBase() {}

  GradientDef GetGradientDefs() const {
    GradientDef node_defs = GetGradientDefsImpl();
    for (size_t i = 0; i < node_defs.size(); ++i) {
      NodeDef& node_def = node_defs[i];
      if (node_def.name.empty()) {
        node_def.name = Name(node_def.op_type + "_" + std::to_string(i));
      }
    }
    return node_defs;
  }

  static std::string GradientName(const std::string& name) {
    return name + "_grad";
  }

  static std::string ExternalOutputName(const std::string& name) {
    return name + "_external";
  }

 protected:
  virtual GradientDef GetGradientDefsImpl() const = 0;

  const GradientGraphConfiguration& GetGradientGraphConfiguration() const {
    return gradient_graph_config_;
  }

  // i-th input of forward op
  ArgDef I(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());

    const std::string& name = node_->InputDefs()[i]->Name();
    const NodeArg* recomputed_nodearg = graph_->GetNodeArg(graph_utils::RecomputeName(name));
    if (recomputed_nodearg) {
      const Node* producer_node = graph_->GetProducerNode(name);
      LOGS(logger_, INFO) << "Recomputed node arg found for " << producer_node->Name();
      return ArgDef(recomputed_nodearg->Name(), recomputed_nodearg->TypeAsProto());
    }

    return ArgDef(node_->InputDefs()[i]->Name(), node_->InputDefs()[i]->TypeAsProto());
  }

  // i-th output of forward op
  ArgDef O(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());

    const std::string& name = node_->OutputDefs()[i]->Name();
    const NodeArg* recomputed_nodearg = graph_->GetNodeArg(graph_utils::RecomputeName(name));
    if (recomputed_nodearg) {
      const Node* producer_node = graph_->GetProducerNode(name);
      LOGS(logger_, INFO) << "Recomputed node arg found for " << producer_node->Name();
      return ArgDef(recomputed_nodearg->Name(), recomputed_nodearg->TypeAsProto());
    }

    return ArgDef(node_->OutputDefs()[i]->Name(), node_->OutputDefs()[i]->TypeAsProto());
  }

  // gradient of i-th input of forward op
  ArgDef GI(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());
    return ArgDef(GradientName(node_->InputDefs()[i]->Name()), node_->InputDefs()[i]->TypeAsProto());
  }

  // gradient of i-th output of forward op
  ArgDef GO(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());
    return ArgDef(GradientName(node_->OutputDefs()[i]->Name()), node_->OutputDefs()[i]->TypeAsProto());
  }

  // intermediate argument
  ArgDef IA(const std::string& argSuffix, const TypeProto* type_proto = nullptr) const {
    return ArgDef(Name(argSuffix), type_proto);
  }

  // type of i-th input of forward op
  const TypeProto* IType(const size_t i) const {
    ORT_ENFORCE(i < node_->InputDefs().size());
    return node_->InputDefs()[i]->TypeAsProto();
  }

  // type of i-th output of forward op
  const TypeProto* OType(const size_t i) const {
    ORT_ENFORCE(i < node_->OutputDefs().size());
    return node_->OutputDefs()[i]->TypeAsProto();
  }

  // Element type of i-th input of forward op.
  int IElemType(const size_t i) const {
    return IType(i)->tensor_type().elem_type();
  }

  // Element type of i-th output of forward op.
  int OElemType(const size_t i) const {
    return OType(i)->tensor_type().elem_type();
  }

  int GetSrcNodeInputSize() const {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->InputDefs().size();
  }

  int GetSrcNodeOutputSize() const {
    ORT_ENFORCE(node_ != nullptr);
    return (int)node_->OutputDefs().size();
  }

  // returns true if the input at index i of the node_ requires gradient
  bool IsGradientRequiredForSrcNodeInput(const size_t i) const {
    return i < node_->InputDefs().size() &&
           gradient_outputs_.find(node_->InputDefs()[i]->Name()) != gradient_outputs_.end();
  }

  // returns true if the output at index i of the node_ has a gradient
  bool IsGradientAvailableForSrcNodeOutput(const size_t i) const {
    return i < node_->OutputDefs().size() &&
           gradient_inputs_.find(node_->OutputDefs()[i]->Name()) != gradient_inputs_.end();
  }

  std::string Name(const std::string& name) const {
    return unique_node_prefix_ + name;
  }

  const NodeAttributes& SrcNodeAttributes() const {
    return node_->GetAttributes();
  }

  const std::string& SrcNodeOpType() const {
    return node_->OpType();
  }

  int SrcNodeOpsetVersion() const {
    return node_->Op()->since_version();
  }

  const std::string& SrcNodeDomain() const {
    return node_->Op()->domain();
  }

  int OnnxOpSetVersion() const {
    return graph_ != nullptr && graph_->DomainToVersionMap().find(kOnnxDomain) != graph_->DomainToVersionMap().end() ?
               graph_->DomainToVersionMap().at(kOnnxDomain) : -1;
  }

  template <typename T>
  static NodeDef ConstantVectorNode(const std::vector<T>& values, const std::string& arg_name) {
    auto t_proto = ONNX_NAMESPACE::ToTensor<T>(values);
    t_proto.add_dims(values.size());

    return NodeDef("Constant",
                   {},
                   {ArgDef(arg_name, nullptr)},
                   {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
  }

  template <typename T>
  static ONNX_NAMESPACE::TensorProto ScalarTensorProto(T value, std::vector<int64_t> shape) {
    ORT_ENFORCE(shape.size() == 0 || (shape.size() == 1 && shape[0] == 1));
    auto t_proto = ONNX_NAMESPACE::ToTensor<T>(value);
    for (auto dim : shape) {
      t_proto.add_dims(dim);
    }

    return t_proto;
  }

  template <typename T>
  static NodeDef ConstantScalarNode(T value, std::vector<int64_t> shape, const std::string& arg_name) {
    auto t_proto = ScalarTensorProto(value, shape);
    return NodeDef("Constant",
                   {},
                   {ArgDef(arg_name, nullptr)},
                   {ONNX_NAMESPACE::MakeAttribute("value", t_proto)});
  }

  // We only support FP32, FP16 and BF16 for these constant nodes for now.
  static NodeDef ConstantScalarNode(float value, const std::string& arg_name, int elem_type) {
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      return ConstantScalarNode(MLFloat16(math::floatToHalf(value)), {1}, arg_name);
    }

    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      return ConstantScalarNode(BFloat16(value), {1}, arg_name);
    }

    return ConstantScalarNode(value, {1}, arg_name);
  }

  static ONNX_NAMESPACE::TensorProto ScalarTensorProtoByElemType(float value, int elem_type) {
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      return ScalarTensorProto(MLFloat16(math::floatToHalf(value)), {1});
    }

    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      return ScalarTensorProto(BFloat16(value), {1});
    }

    return ScalarTensorProto(value, {1});
  }

  static NodeDef ZeroConstantNode(int elem_type) {
    return ConstantScalarNode(0.0f, "ZeroConstant_Type" + std::to_string(elem_type), elem_type);
  }

  static NodeDef HalfConstantNode(int elem_type) {
    return ConstantScalarNode(0.5f, "HalfConstant_Type" + std::to_string(elem_type), elem_type);
  }

  static NodeDef OneConstantNode(int elem_type) {
    return ConstantScalarNode(1.0f, "OneConstant_Type" + std::to_string(elem_type), elem_type);
  }

  void AddReduceSumNode(const ArgDef& input_arg_def,
                        const ArgDef& output_arg_def,
                        const std::vector<int64_t>& reduce_axes,
                        bool keep_dims,
                        std::vector<NodeDef>& output) const;

  void HandleBroadcasting(const ArgDef& input_grad,
                          const ArgDef& target,
                          const ArgDef& output_grad,
                          const std::vector<int64_t>& reduce_axes,
                          std::vector<NodeDef>& output) const;

  void HandleBroadcastingDynamic(const ArgDef& input_grad,
                                 const ArgDef& target,
                                 const ArgDef& target_shape,
                                 const ArgDef& output_grad,
                                 const ArgDef& reduce_axes,
                                 std::vector<NodeDef>& output) const;

  std::vector<NodeDef> GetBiasGeluGradNodes(
    bool use_approximation,
    const ArgDef& dY, const ArgDef& X, const ArgDef& B,  // inputs
    const ArgDef& dX, const ArgDef& dB,                  // outputs
    const ArgDef& b_axes, const ArgDef& b_shape, const ArgDef& x_shape,  //intermediate args
    const std::string& node_name) const;

  const std::string& NodeName() const { return node_->Name(); }

  AttributeProto AttributeDefinitionToAttributeProto(const std::string& name,
                                                     const AttributeDefinition& attr_def) const;

 private:
  friend class GradientGraphBuilder;

  std::string CreateUniqueNodePrefix() {
    ORT_ENFORCE(node_ != nullptr);
    auto name = node_->Name();
    std::stringstream unique_prefix;

    if (!name.empty()) {
      unique_prefix << name << "_Grad/";
    } else {
      unique_prefix << graph_->GenerateNodeName(node_->OpType()) << "_Grad/";
    }
    return unique_prefix.str();
  }

  const GradientGraphConfiguration& gradient_graph_config_;
  Graph* graph_;
  const Node* node_;
  std::string unique_node_prefix_;

  // contains set of output arg names of node_ which is provided as gradient input to the bw node
  std::unordered_set<std::string> gradient_inputs_;

  // contains set of input arg names of node_ which requires gradient
  std::unordered_set<std::string> gradient_outputs_;

  const logging::Logger& logger_;
};

class EmptyGradientBuilder : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  GradientDef GetGradientDefsImpl() const override {
    return GradientDef();
  }
};

class UnSupportedGradientBuilder : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;
  GradientDef GetGradientDefsImpl() const override {
    ORT_ENFORCE(false, "Gradient should not be requested for this operator");
  }
};

}  // namespace training
}  // namespace onnxruntime
