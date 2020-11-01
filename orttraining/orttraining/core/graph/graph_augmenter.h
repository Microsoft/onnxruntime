// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <string>
#include <vector>
#include "core/graph/basic_types.h"
#include "core/common/status.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TypeProto;

struct ArgDef {
  ArgDef() : name(""), type_proto(nullptr) {}
  ArgDef(std::string name, const TypeProto* type = nullptr) : name(name), type_proto(type) {}

  std::string name;
  const TypeProto* type_proto;

  bool operator==(const ArgDef& other) const {
    return name == other.name;
  }
};

struct OpDef {
  OpDef() {}
  OpDef(const std::string& type, const std::string& domain = kOnnxDomain, const int opset_version = 9)
      : type(type),
        domain(domain),
        opset_version(opset_version){};

  std::string type;
  std::string domain;
  int opset_version;
};

struct NodeDef {
  NodeDef(const OpDef& op_def,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const NodeAttributes& attributes = NodeAttributes(),
          const std::string& name = "") : op_type(op_def.type),
                                          domain(op_def.domain),
                                          input_args(input_args),
                                          output_args(output_args),
                                          attributes(attributes),
                                          name(name),
                                          priority_(0){};

  NodeDef(const std::string& op_type,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const NodeAttributes& attributes = NodeAttributes(),
          const std::string& name = "") : op_type(op_type),
                                          input_args(input_args),
                                          output_args(output_args),
                                          attributes(attributes),
                                          name(name),
                                          priority_(0){};

  NodeDef(const OpDef& op_def,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const std::vector<AttributeProto>& attribute_protos,
          const std::string& name = "") : op_type(op_def.type),
                                          domain(op_def.domain),
                                          input_args(input_args),
                                          output_args(output_args),
                                          name(name),
                                          priority_(0) {
    for (const AttributeProto& a : attribute_protos) {
      attributes.insert({a.name(), a});
    }
  }

  NodeDef(const std::string& op_type,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const std::vector<AttributeProto>& attribute_protos,
          const std::string& name = "") : op_type(op_type),
                                          input_args(input_args),
                                          output_args(output_args),
                                          name(name),
                                          priority_(0) {
    for (const AttributeProto& a : attribute_protos) {
      attributes.insert({a.name(), a});
    }
  }

  void SetPriority(int p) { priority_ = p; }

  std::string op_type;
  std::string domain = kOnnxDomain;
  std::vector<ArgDef> input_args;
  std::vector<ArgDef> output_args;
  NodeAttributes attributes;
  std::string name;
  int priority_;
};

/** GraphAugmenter is a stateless class to add new elements into a Graph.
    The elements to be added could be:
    1. Nodes
    2. Initializers
    3. Graph inputs
    4. Graph outputs
*/
class GraphAugmenter {
 public:
  class GraphDefs {
   public:
    void AddNodeDefs(const std::vector<NodeDef>& node_defs) {
      for (auto node_def : node_defs) {
        AddNodeDef(node_def);
      }
    }

    void AddNodeDef(const NodeDef& node_def) {
      // Copy constant node value to graph_initializers_
      if (node_def.op_type == kConstant) {
        TensorProto initializer = node_def.attributes.at("value").t();
        initializer.set_name(node_def.output_args[0].name);
        graph_initializers_.push_back(initializer);
      } else {
        node_defs_.push_back(node_def);
      }
    }

    const std::vector<NodeDef>& NodeDefs() const {
      return node_defs_;
    }

    std::vector<NodeDef>& NodeDefs() {
      return node_defs_;
    }

    void AddGraphInputs(const std::vector<std::string>& names) {
      graph_input_names_.insert(graph_input_names_.end(), names.begin(), names.end());
    }

    const std::vector<std::string>& GraphInputs() const {
      return graph_input_names_;
    }

    void AddGraphOutputs(const std::vector<std::string>& names) {
      graph_output_names_.insert(graph_output_names_.end(), names.begin(), names.end());
    }

    const std::vector<std::string>& GraphOutputs() const {
      return graph_output_names_;
    }

    void AddInitializers(const std::vector<TensorProto>& tensors) {
      graph_initializers_.insert(graph_initializers_.end(), tensors.begin(), tensors.end());
    }

    const std::vector<TensorProto>& Initializers() const {
      return graph_initializers_;
    }

    // When adding ArgDef, if new TypeProto is needed, call this func to get a new one
    // So that the life cycle is managed by GraphDefs.
    TypeProto* CreateTypeProto() {
      graph_type_protos_.push_back(onnxruntime::make_unique<TypeProto>());
      return graph_type_protos_.back().get();
    }

    TypeProto* CreateTypeProto(const std::vector<int64_t>& dims, ONNX_NAMESPACE::TensorProto_DataType data_type) {
      TypeProto* type_proto = CreateTypeProto();
      type_proto->mutable_tensor_type()->set_elem_type(data_type);
      for (int64_t dim : dims)
        type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
      return type_proto;
    }

    TypeProto* CopyTypeProto(const NodeArg* node_arg) {
      ORT_ENFORCE(node_arg != nullptr, "During CopyTypeProto, ", node_arg->Name(), "'s node_arg is null.");
      TypeProto* type_proto = CreateTypeProto();
      type_proto->CopyFrom(*(node_arg->TypeAsProto()));
      return type_proto;
    }

    TypeProto* CopyTypeProto(const ArgDef& argdef) {
      ORT_ENFORCE(argdef.type_proto, "During CopyTypeProto, ", argdef.name, "'s type_proto is null.");
      TypeProto* type_proto = CreateTypeProto();
      type_proto->CopyFrom(*argdef.type_proto);
      return type_proto;
    }

   private:
    std::vector<NodeDef> node_defs_;
    std::vector<std::string> graph_input_names_;
    std::vector<std::string> graph_output_names_;
    std::vector<TensorProto> graph_initializers_;

    // A pool of TypeProto, used when adding ArgDef if new TypeProto is needed.
    std::vector<std::unique_ptr<TypeProto>> graph_type_protos_;
  };

  // Augment the graph with new_graph_elements which defines new nodes, outputs, initializers.
   static common::Status AugmentGraph(Graph& graph,
                                     const GraphDefs& graph_element_defs,
                                     const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr);

  // Manually set the graph outputs
  static common::Status OverrideGraphOutputs(Graph& graph, const std::vector<std::string>& graph_outputs);
};
}  // namespace training
}  // namespace onnxruntime
