// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>
#include <queue>

#include "gsl/gsl"
#include "core/common/logging/logging.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/graph/graph_flatbuffers_utils.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/model.h"
#include "core/graph/model_load_utils.h"
#include "core/graph/op.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/function.h"
#include "core/graph/function_impl.h"
#include "core/graph/schema_registry.h"
#include "onnx/checker.h"
using namespace ONNX_NAMESPACE::checker;
#endif

using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ::onnxruntime::common;
using namespace onnxruntime::experimental;

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
#define NO_CHANGE_ON_SYNC_FLAG(...)                  \
  do {                                               \
    const bool sync_needed = GraphProtoSyncNeeded(); \
    { __VA_ARGS__; }                                 \
    GraphProtoSyncNeeded(sync_needed);               \
  } while (0)

static bool UsingLatestOnnxOpset(const DomainToVersionMap& opset_versions) {
  bool is_latest_opset = false;
  auto onnx_opset = opset_versions.find(kOnnxDomain);

  if (onnx_opset != opset_versions.cend()) {
    static int latest_onnx_version = model_load_utils::IsAllowReleasedONNXOpsetsOnlySet()
                                         ? ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().LastReleaseVersionMap().at(ONNX_NAMESPACE::ONNX_DOMAIN)
                                         : ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(ONNX_NAMESPACE::ONNX_DOMAIN).second;
    if (onnx_opset->second == latest_onnx_version) {
      is_latest_opset = true;
    }
  }

  return is_latest_opset;
}

static Status MergeShapeInfo(const std::string& output_name,
                             const TypeProto& source, TypeProto& target,
                             bool strict, const logging::Logger& logger) {

  if (!(utils::HasTensorType(source) && utils::HasTensorType(target)) &&
      !(utils::HasSparseTensorType(source) && utils::HasSparseTensorType(target))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Source and target must both be either tensors or sparse tensors");
  }

  auto status = Status::OK();

  ORT_TRY {
    if (utils::HasTensorType(source)) {
      ONNX_NAMESPACE::mergeInShapeInfo(source.tensor_type(), *target.mutable_tensor_type());
    } else {
      ONNX_NAMESPACE::mergeInShapeInfo(source.sparse_tensor_type(), *target.mutable_sparse_tensor_type());
    }
  }
  ORT_CATCH(const ONNX_NAMESPACE::InferenceError& ex) {
    // if this model was not created with the latest onnx version, allow the shape inferencing failure (strict == false).
    // we do this to have strict testing of the latest inferencing to detect bugs, but lenient shape inferencing for
    // older models in case later changes to the ONNX shape inferencing or ORT break them.
    if (!strict) {
      // mergeInShapeInfo does nothing unless source.shape() is not null, and there would be no conflict if
      // target.shape() was empty. 'assert' just in case that ever changes.
      assert(utils::HasShape(source) && utils::HasShape(target));
      LOGS(logger, WARNING) << "Error merging shape info for output. '" << output_name
                            << "' source:" << utils::GetShape(source) << " target:" << utils::GetShape(target)
                            << ". Falling back to lenient merge.";
      if (utils::HasTensorType(source)) {
        ONNX_NAMESPACE::UnionShapeInfo(utils::GetShape(source), *target.mutable_tensor_type());
      } else {
        ONNX_NAMESPACE::UnionShapeInfo(utils::GetShape(source), *target.mutable_sparse_tensor_type());
      }
    } else {
      ORT_UNUSED_PARAMETER(logger);
      ORT_UNUSED_PARAMETER(strict);
      ORT_UNUSED_PARAMETER(output_name);
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output:", output_name, " ", ex.what());
      });
    }
  }

  return status;
}

static bool GraphLoadedFromModelFile(const GraphProto* graph_proto) {
  return graph_proto && (graph_proto->node_size() != 0 ||
                         graph_proto->output_size() != 0);
}

// there are some known invalid usages of dim_param and dim_value. remove them from the TypeProto so that
// they don't affect shape inferencing or the allocation planner
static void RemoveInvalidValues(ONNX_NAMESPACE::TypeProto& type) {
  if (utils::HasTensorType(type) && utils::HasShape(type.tensor_type())) {
    auto* shape = type.mutable_tensor_type()->mutable_shape();
    for (int i = 0, end = shape->dim_size(); i < end; ++i) {
      auto& dim = *shape->mutable_dim(i);
      if (utils::HasDimParam(dim)) {
        if (dim.dim_param().empty()) {
          dim.clear_dim_param();
        }
      } else if (utils::HasDimValue(dim)) {
        if (dim.dim_value() < 0) {
          dim.clear_dim_value();
        }
      }
    }
  }
}

static TypeProto TypeProtoFromTensorProto(const TensorProto& tensor) {
  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(tensor.data_type());
  auto shape = t.mutable_tensor_type()->mutable_shape();
  for (auto dim : tensor.dims())
    shape->add_dim()->set_dim_value(dim);

  return t;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
NodeArg::NodeArg(const std::string& name, const TypeProto* p_node_arg_type) {
  node_arg_info_.set_name(name);
  // If the name is empty, it means the arg does not exist.
  exists_ = !(name.empty());
  if (nullptr != p_node_arg_type) {
    (*node_arg_info_.mutable_type()) = *p_node_arg_type;
#if !defined(ORT_MINIMAL_BUILD)
    // should not be possible to have invalid values in the ORT format model, so we don't need this
    // in a minimal build
    RemoveInvalidValues(*node_arg_info_.mutable_type());
#endif
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  } else {
    type_ = nullptr;
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

NodeArg::NodeArg(NodeArgInfo&& node_arg_info) {
  node_arg_info_ = std::move(node_arg_info);

  exists_ = !node_arg_info_.name().empty();
  if (node_arg_info_.has_type())
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  else {
    type_ = nullptr;
  }
}

const std::string& NodeArg::Name() const noexcept {
  return node_arg_info_.name();
}

DataType NodeArg::Type() const noexcept {
  return type_;
}

const TypeProto* NodeArg::TypeAsProto() const noexcept {
  if (utils::HasType(node_arg_info_))
    return &node_arg_info_.type();

  return nullptr;
}

const TensorShapeProto* NodeArg::Shape() const {
  const TypeProto* type = TypeAsProto();
  if (type == nullptr) return nullptr;
  const auto typeCase = type->value_case();
  switch (typeCase) {
    case TypeProto::kTensorType: {
      if (utils::HasShape(type->tensor_type())) {
        return &(type->tensor_type().shape());
      }
      return nullptr;
    }
    case TypeProto::kSparseTensorType: {
      if (utils::HasShape(type->sparse_tensor_type())) {
        return &(type->sparse_tensor_type().shape());
      }
      return nullptr;
    }
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return nullptr;
  }
}

bool NodeArg::HasTensorOrScalarShape() const {
  const TypeProto* type = TypeAsProto();
  if (!type) return false;
  const auto type_case = type->value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
    case TypeProto::kSparseTensorType:
      // Standard tensor has a valid shape field while
      // scalar's shape is empty. Thus, we don't need to
      // check shape here.
      return true;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return false;
  }
}

#if !defined(ORT_MINIMAL_BUILD)
void NodeArg::SetShape(const TensorShapeProto& shape) {
  const auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      *(node_arg_info_.mutable_type()->mutable_tensor_type()->mutable_shape()) = shape;
      break;
    case TypeProto::kSparseTensorType:
      *(node_arg_info_.mutable_type()->mutable_sparse_tensor_type()->mutable_shape()) = shape;
      break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

void NodeArg::ClearShape() {
  const auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      node_arg_info_.mutable_type()->mutable_tensor_type()->clear_shape();
      break;
    case TypeProto::kSparseTensorType:
      node_arg_info_.mutable_type()->mutable_sparse_tensor_type()->clear_shape();
      break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

common::Status NodeArg::UpdateTypeAndShape(const ONNX_NAMESPACE::TypeProto& input_type, bool strict,
                                           bool override_types, const logging::Logger& logger) {
  if (!utils::HasType(node_arg_info_)) {
    *node_arg_info_.mutable_type() = input_type;
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
    return Status::OK();
  }

  auto& current_type = *node_arg_info_.mutable_type();
  const auto current_type_case = current_type.value_case();
  const auto input_type_case = input_type.value_case();

  if (current_type_case != input_type_case)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type mismatch. Current=",
                           current_type_case, " Input=", input_type_case);

  switch (input_type_case) {
    case TypeProto::kTensorType: {
      const auto& input_tensor_type = input_type.tensor_type();
      const auto& input_tensor_elem_type = input_tensor_type.elem_type();
      const auto& current_tensor_elem_type = current_type.tensor_type().elem_type();

      if (input_tensor_elem_type != current_tensor_elem_type) {
        if (override_types) {
          DataType inferred_type = DataTypeUtils::ToType(input_type);
          // The "SetType" call will override the shape information to empty.
          // If the original tensor has shape information, need to set it back.
          if (Shape()) {
            auto old_shape = *Shape();
            SetType(inferred_type);
            SetShape(old_shape);
          } else {
            SetType(inferred_type);
          }
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Tensor element type mismatch. ",
                                 static_cast<TensorProto_DataType>(input_tensor_elem_type), " != ",
                                 static_cast<TensorProto_DataType>(current_tensor_elem_type));
        }
      }

      if (utils::HasShape(input_tensor_type)) {
        if (utils::HasShape(current_type)) {
          ORT_RETURN_IF_ERROR(MergeShapeInfo(Name(), input_type, current_type, strict, logger));
        } else {
          *current_type.mutable_tensor_type() = input_tensor_type;
        }
      }

      break;
    }
    case TypeProto::kSparseTensorType: {
      const auto& input_tensor_type = input_type.sparse_tensor_type();
      const auto input_tensor_elem_type = input_tensor_type.elem_type();
      const auto current_tensor_elem_type = current_type.sparse_tensor_type().elem_type();

      if (input_tensor_elem_type != current_tensor_elem_type) {
        if (override_types) {
          DataType inferred_type = DataTypeUtils::ToType(input_type);
          if (Shape()) {
            auto old_shape = *Shape();
            SetType(inferred_type);
            SetShape(old_shape);
          } else {
            SetType(inferred_type);
          }
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SparseTensor element type mismatch. ",
                                 static_cast<TensorProto_DataType>(input_tensor_elem_type), " != ",
                                 static_cast<TensorProto_DataType>(current_tensor_elem_type));
        }
      }

      if (utils::HasShape(input_tensor_type)) {
        if (utils::HasShape(current_type)) {
          ORT_RETURN_IF_ERROR(MergeShapeInfo(Name(), input_type, current_type, strict, logger));
        } else {
          *current_type.mutable_sparse_tensor_type() = input_tensor_type;
        }
      }
    } break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOptionalType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
      break;
  }

  return Status::OK();
}

common::Status NodeArg::UpdateTypeAndShape(const NodeArg& node_arg, bool strict, bool override_types,
                                           const logging::Logger& logger) {
  auto status = Status::OK();

  if (utils::HasType(node_arg.node_arg_info_))
    status = UpdateTypeAndShape(node_arg.node_arg_info_.type(), strict, override_types, logger);

  return status;
}

void NodeArg::SetType(DataType p_type) {
  if (nullptr == p_type) {
    return;
  }

  type_ = p_type;
  *(node_arg_info_.mutable_type()) = DataTypeUtils::ToTypeProto(p_type);
}

void NodeArg::SetType(const TypeProto& type_proto) {
  type_ = DataTypeUtils::ToType(type_proto);
  *(node_arg_info_.mutable_type()) = type_proto;
}

#endif  // !defined(ORT_MINIMAL_BUILD)

bool NodeArg::Exists() const noexcept {
  return exists_;
}

Node::EdgeEnd::EdgeEnd(const Node& node, int src_arg_index, int dst_arg_index) noexcept
    : node_(&node),
      src_arg_index_(src_arg_index),
      dst_arg_index_(dst_arg_index) {
}

Node::EdgeEnd::EdgeEnd(const Node& node) noexcept
    : EdgeEnd(node, INT_MAX, INT_MAX) {
}

Node::NodeConstIterator::NodeConstIterator(EdgeConstIterator p_iter) {
  m_iter = p_iter;
}

bool Node::NodeConstIterator::operator==(const NodeConstIterator& p_other) const {
  return m_iter == p_other.m_iter;
}

bool Node::NodeConstIterator::operator!=(const NodeConstIterator& p_other) const {
  return m_iter != p_other.m_iter;
}

void Node::NodeConstIterator::operator++() {
  ++m_iter;
}

void Node::NodeConstIterator::operator--() {
  --m_iter;
}

const Node& Node::NodeConstIterator::operator*() const {
  return (*m_iter).GetNode();
}

const Node* Node::NodeConstIterator::operator->() const {
  return &(operator*());
}

void Node::SetPriority(int priority) noexcept {
  priority_ = priority;
}

const Path& Node::ModelPath() const noexcept {
  return graph_->ModelPath();
}

#if !defined(ORT_MINIMAL_BUILD)

const Function* Node::GetFunctionBody(bool try_init_func_body) {
  if (nullptr != func_body_) {
    return func_body_;
  }

  // Initialize function body
  if (try_init_func_body) {
    graph_->InitFunctionBodyForNode(*this);
  }

  return func_body_;
}

void Node::SetFunctionBody(const Function& func) {
  func_body_ = &func;
  op_ = &func.OpSchema();
  since_version_ = op_->since_version();
}

void Node::ToProto(NodeProto& proto, bool update_subgraphs) const {
  proto.set_name(name_);
  proto.set_op_type(op_type_);

  if (!domain_.empty())
    proto.set_domain(domain_);

  if (!description_.empty())
    proto.set_doc_string(description_);

  // Set attributes.
  proto.clear_attribute();
  for (const auto& attribute : attributes_) {
    const gsl::not_null<AttributeProto*> attr{proto.add_attribute()};
    *attr = attribute.second;  // copy
    if (update_subgraphs && attr->has_g()) {
      attr->clear_g();
      *attr->mutable_g() = attr_to_subgraph_map_.find(attribute.first)->second->ToGraphProto();
    }
  }

  // Set inputs' definitions.
  proto.clear_input();
  for (auto& input_def : definitions_.input_defs) {
    proto.add_input(input_def->Name());
  }

  // Set outputs' definitions.
  proto.clear_output();
  for (auto& output_def : definitions_.output_defs) {
    proto.add_output(output_def->Name());
  }
}

Status Node::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                             flatbuffers::Offset<fbs::Node>& fbs_node) const {
  // if type is Primitive it's an ONNX function and currently we have kernel implementations for all those
  if (func_body_ != nullptr && node_type_ != Type::Primitive) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Serialization of fused function body is not currently supported, ",
                           "Node [", name_, "] op_type [", op_type_, "]");
  }

  auto GetNodeArgsOrtFormat = [&builder](const std::vector<NodeArg*>& src) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> node_args(src.size());
    std::transform(src.cbegin(), src.cend(), node_args.begin(),
                   [&builder](const NodeArg* nodearg) {
                     // NodeArg's name will be used by multiple places, create shared string
                     return builder.CreateSharedString(nodearg->Name());
                   });
    return builder.CreateVector(node_args);
  };

  auto name = builder.CreateString(name_);
  auto doc_string = builder.CreateString(description_);
  auto domain = builder.CreateSharedString(domain_);
  auto op_type = builder.CreateSharedString(op_type_);
  auto ep = builder.CreateSharedString(execution_provider_type_);
  auto inputs = GetNodeArgsOrtFormat(definitions_.input_defs);
  auto outputs = GetNodeArgsOrtFormat(definitions_.output_defs);
  auto input_arg_counts = builder.CreateVector(definitions_.input_arg_count);
  auto implicit_inputs = GetNodeArgsOrtFormat(definitions_.implicit_input_defs);

  // Node attributes
  std::vector<flatbuffers::Offset<fbs::Attribute>> attributes_vec;
  attributes_vec.reserve(attributes_.size());
  for (const auto& entry : attributes_) {
    const auto& attr_name = entry.first;
    const auto& attr_proto = entry.second;
    flatbuffers::Offset<fbs::Attribute> fbs_attr;
    Graph* subgraph = nullptr;
    if (attr_proto.has_g()) {
      const auto it = attr_to_subgraph_map_.find(attr_name);
      ORT_RETURN_IF_NOT(it != attr_to_subgraph_map_.cend(),
                        "Node [", name_, "] op_type [", op_type_, "] ", "does not have the graph for key ", attr_name);
      subgraph = it->second;
    }
    ORT_RETURN_IF_ERROR(
        experimental::utils::SaveAttributeOrtFormat(builder, attr_proto, fbs_attr, ModelPath(), subgraph));
    attributes_vec.push_back(fbs_attr);
  }
  auto attributes = builder.CreateVector(attributes_vec);

  fbs::NodeBuilder nb(builder);
  nb.add_name(name);
  nb.add_doc_string(doc_string);
  nb.add_domain(domain);
  nb.add_since_version(since_version_);
  nb.add_index(gsl::narrow<uint32_t>(index_));
  nb.add_op_type(op_type);
  nb.add_type(static_cast<fbs::NodeType>(node_type_));
  nb.add_execution_provider_type(ep);
  nb.add_inputs(inputs);
  nb.add_outputs(outputs);
  nb.add_attributes(attributes);
  nb.add_input_arg_counts(input_arg_counts);
  nb.add_implicit_inputs(implicit_inputs);
  fbs_node = nb.Finish();
  return Status::OK();
}

flatbuffers::Offset<fbs::NodeEdge> Node::SaveEdgesToOrtFormat(flatbuffers::FlatBufferBuilder& builder) const {
  auto get_edges = [](const EdgeSet& edge_set) {
    std::vector<fbs::EdgeEnd> edges;
    edges.reserve(edge_set.size());
    for (const auto& edge : edge_set)
      edges.push_back(fbs::EdgeEnd(gsl::narrow<uint32_t>(edge.GetNode().Index()),
                                   edge.GetSrcArgIndex(), edge.GetDstArgIndex()));

    return edges;
  };

  const auto input_edges = get_edges(relationships_.input_edges);
  const auto output_edges = get_edges(relationships_.output_edges);
  return fbs::CreateNodeEdgeDirect(builder, gsl::narrow<uint32_t>(index_), &input_edges, &output_edges);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if defined(ENABLE_ORT_FORMAT_LOAD)
Status Node::LoadFromOrtFormat(const onnxruntime::experimental::fbs::Node& fbs_node, Graph& graph,
                               const logging::Logger& logger, std::unique_ptr<Node>& node) {
  node.reset(new Node(fbs_node.index(), graph));
  return node->LoadFromOrtFormat(fbs_node, logger);
}

Status Node::LoadFromOrtFormat(const onnxruntime::experimental::fbs::Node& fbs_node, const logging::Logger& logger) {
  auto LoadNodeArgsFromOrtFormat =
      [&](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* fbs_node_arg_names,
          std::vector<NodeArg*>& node_args,
          bool check_parent_graph = false) -> Status {
    ORT_RETURN_IF(nullptr == fbs_node_arg_names, "fbs_node_arg_names cannot be null");
    node_args.reserve(fbs_node_arg_names->size());
    for (const auto* node_arg_name : *fbs_node_arg_names) {
      ORT_RETURN_IF(nullptr == node_arg_name, "node_arg_name cannot be null");
      auto* node_arg = check_parent_graph ? graph_->GetNodeArgIncludingParentGraphs(node_arg_name->str())
                                          : graph_->GetNodeArg(node_arg_name->str());
      ORT_RETURN_IF(nullptr == node_arg, "LoadNodeArgsFromOrtFormat: Node [", name_, "] op_type [", op_type_,
                    "], could not find NodeArg ", node_arg_name->str());
      node_args.push_back(node_arg);
    }
    return Status::OK();
  };

  // index_ was set in the ctor of this Node instance
  experimental::utils::LoadStringFromOrtFormat(name_, fbs_node.name());
  experimental::utils::LoadStringFromOrtFormat(description_, fbs_node.doc_string());
  experimental::utils::LoadStringFromOrtFormat(domain_, fbs_node.domain());
  since_version_ = fbs_node.since_version();
  experimental::utils::LoadStringFromOrtFormat(op_type_, fbs_node.op_type());
  node_type_ = static_cast<Node::Type>(fbs_node.type());
  experimental::utils::LoadStringFromOrtFormat(execution_provider_type_, fbs_node.execution_provider_type());
  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.inputs(), definitions_.input_defs));

  // attributes
  auto fbs_attributes = fbs_node.attributes();
  if (fbs_attributes) {
    for (const auto* fbs_attr : *fbs_attributes) {
      ORT_RETURN_IF(nullptr == fbs_attr, "fbs_attr cannot be null");
      AttributeProto attr_proto;
      std::unique_ptr<Graph> subgraph;
      ORT_RETURN_IF_ERROR(
          experimental::utils::LoadAttributeOrtFormat(*fbs_attr, attr_proto, subgraph, *graph_, *this, logger));

      // If we have a sub graph in this attributes, it will be loaded into subgraph ptr
      // while the attribute proto contains the sub graph will have the empty g() field
      if (attr_proto.type() == AttributeProto_AttributeType_GRAPH) {
        ORT_RETURN_IF_NOT(subgraph, "Serialization error. Graph attribute was serialized without Graph instance");
        attr_to_subgraph_map_.emplace(attr_proto.name(), gsl::not_null<Graph*>(subgraph.get()));
        subgraphs_.push_back(std::move(subgraph));
      }

      AddAttribute(attr_proto.name(), attr_proto);
    }
  }

  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.implicit_inputs(), definitions_.implicit_input_defs,
                                                /* check parent graphs */ true));

  {  // input_arg_counts
    auto fbs_input_arg_counts = fbs_node.input_arg_counts();
    ORT_RETURN_IF(nullptr == fbs_input_arg_counts, "Node::LoadFromOrtFormat, input_arg_counts is missing");
    auto& input_arg_count = definitions_.input_arg_count;
    input_arg_count.reserve(fbs_input_arg_counts->size());
    input_arg_count.insert(input_arg_count.begin(), fbs_input_arg_counts->cbegin(), fbs_input_arg_counts->cend());
  }

  ORT_RETURN_IF_ERROR(LoadNodeArgsFromOrtFormat(fbs_node.outputs(), definitions_.output_defs));

  return Status::OK();
}

Status Node::LoadEdgesFromOrtFormat(const onnxruntime::experimental::fbs::NodeEdge& fbs_node_edges,
                                    const Graph& graph) {
  ORT_RETURN_IF(fbs_node_edges.node_index() != index_,
                "input index: ", fbs_node_edges.node_index(), " is not the same as this node's index:", index_);

  auto add_edges = [&graph](const flatbuffers::Vector<const onnxruntime::experimental::fbs::EdgeEnd*>* fbs_edges,
                            EdgeSet& edge_set, const std::string dst_name) -> Status {
    if (fbs_edges) {
      for (const auto* fbs_edge : *fbs_edges) {
        ORT_RETURN_IF(nullptr == fbs_edge, "Node::LoadEdgesFromOrtFormat, edge is missing for ", dst_name);
        edge_set.emplace(*graph.GetNode(fbs_edge->node_index()), fbs_edge->src_arg_index(), fbs_edge->dst_arg_index());
      }
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(add_edges(fbs_node_edges.input_edges(), relationships_.input_edges, "input edges"));
  ORT_RETURN_IF_ERROR(add_edges(fbs_node_edges.output_edges(), relationships_.output_edges, "output edges"));

  return Status::OK();
}
#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void Node::Init(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const NodeAttributes* attributes,
                const std::string& domain) {
  name_ = name;
  op_type_ = op_type;
  description_ = description;
  definitions_.input_defs = input_args;
  definitions_.output_defs = output_args;
  domain_ = domain;
  priority_ = 0;
  if (kOnnxDomainAlias == domain_) {
    domain_ = kOnnxDomain;
  }

  // Set each arg count as 1 by default.
  // It could be adjusted when resolving the node with its operator
  // information.
  definitions_.input_arg_count.assign(input_args.size(), 1);

  if (attributes) {
    attributes_ = *attributes;

    for (auto& name_to_attr : attributes_) {
      if (utils::HasGraph(name_to_attr.second)) {
#if !defined(ORT_MINIMAL_BUILD)
        CreateSubgraph(name_to_attr.first);
#else
        ORT_THROW("Creating node with a subgraph via AddNode is not supported in this build.");
#endif
      }
    }
  }
}

Node::Definitions& Node::MutableDefinitions() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return definitions_;
}

Node::Relationships& Node::MutableRelationships() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return relationships_;
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
void Node::CreateSubgraph(const std::string& attr_name) {
  auto attr = attributes_.find(attr_name);

  if (attr != attributes_.cend() && utils::HasGraph(attr->second)) {
    GraphProto& mutable_graph = *attr->second.mutable_g();
    std::unique_ptr<Graph> subgraph{new Graph(*graph_, *this, mutable_graph)};
    attr_to_subgraph_map_.insert({std::string(attr_name), gsl::not_null<Graph*>{subgraph.get()}});
    subgraphs_.push_back(std::move(subgraph));
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

void Node::AddAttribute(const std::string& attr_name, const AttributeProto& value) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  attributes_[attr_name] = value;
}

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                           \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    a.set_##field(value);                                                    \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_ATTR_IMPL(type, enumType, field)                                 \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    *(a.mutable_##field()) = value;                                          \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_LIST_ATTR_IMPL(type, enumType, field)            \
  void Node::AddAttribute(const std::string& attr_name,      \
                          const std::vector<type>& values) { \
    graph_->SetGraphResolveNeeded();                         \
    graph_->SetGraphProtoSyncNeeded();                       \
    AttributeProto a;                                        \
    a.set_name(attr_name);                                   \
    a.set_type(enumType);                                    \
    for (const auto& val : values) {                         \
      *(a.mutable_##field()->Add()) = val;                   \
    }                                                        \
    attributes_[attr_name] = a;                              \
  };

void Node::AddAttribute(const std::string& attr_name, const GraphProto& value) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  AttributeProto a;
  a.set_name(attr_name);
  a.set_type(AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH);
  *a.mutable_g() = value;
  attributes_[attr_name] = a;

#if !defined(ORT_MINIMAL_BUILD)
  // subgraph is created via deserialization and not here in a minimal build
  CreateSubgraph(attr_name);
#endif
};

ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR, t)
ADD_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSOR, sparse_tensor)
ADD_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTO, tp)
ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)
ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS, graphs)
ADD_LIST_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSORS, sparse_tensors)
ADD_LIST_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTOS, type_protos)

#if !defined(ORT_MINIMAL_BUILD)
bool Node::ClearAttribute(const std::string& attr_name) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return attributes_.erase(attr_name) > 0;
}

Status Node::UpdateInputArgCount() {
  // The node refers to a primitive operator.
  // Infer and verify node input arg type information.
  int total_arg_count = std::accumulate(definitions_.input_arg_count.cbegin(),
                                        definitions_.input_arg_count.cend(), 0);

  if (total_arg_count < 0 || static_cast<size_t>(total_arg_count) != definitions_.input_defs.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "This is an invalid model. "
                           "The sum of input arg count is not equal to size of input defs in node (",
                           name_, ")");
  }

  // op_ is always valid when this is called
  const ONNX_NAMESPACE::OpSchema& op = *Op();

  // Verify size of node arg count is same as input number in
  // operator definition.
  if (op.inputs().size() != definitions_.input_arg_count.size()) {
    // Adjust input arg count array with op definition
    // The adjustment will work as below,
    // In total, there're <total_arg_count> inputs, which
    // will be split as <1, 1, 1, 1, ... 1, x> or
    // <1, 1, 1, 1, ...1, 0, 0, ...0>. The final input
    // arg count array's element number will be the same
    // as op definition, and the sum of all elements will
    // be equal to <total_arg_count>.
    auto& input_arg_count = definitions_.input_arg_count;
    input_arg_count.clear();
    size_t m = 0;
    auto arg_count_left = total_arg_count;

    if (!op.inputs().empty()) {
      for (; m < op.inputs().size() - 1; ++m) {
        if (arg_count_left > 0) {
          input_arg_count.push_back(1);
          arg_count_left--;
        } else {
          input_arg_count.push_back(0);
        }
      }
    }

    // Set the arg count for the last input formal parameter.
    // NOTE: in the case that there's no .input(...) defined
    // in op schema, all input args will be fed as one input
    // of the operator.
    input_arg_count.push_back(arg_count_left);

    graph_->SetGraphResolveNeeded();
    graph_->SetGraphProtoSyncNeeded();
  }

  return Status::OK();
}

Graph* Node::GetMutableGraphAttribute(const std::string& attr_name) {
  Graph* subgraph = nullptr;

  const auto& entry = attr_to_subgraph_map_.find(attr_name);
  if (entry != attr_to_subgraph_map_.cend()) {
    subgraph = entry->second;
  }

  return subgraph;
}

const Graph* Node::GetGraphAttribute(const std::string& attr_name) const {
  return const_cast<Node*>(this)->GetMutableGraphAttribute(attr_name);
}

void Node::ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements) {
  std::vector<std::vector<NodeArg*>*> all_defs = {&definitions_.input_defs, &definitions_.output_defs};

  for (auto pair : replacements)
    for (auto* defs : all_defs)
      for (auto& def : *defs)
        if (def == pair.first)
          def = pair.second;
}

#endif  // !defined(ORT_MINIMAL_BUILD)

std::vector<gsl::not_null<const Graph*>> Node::GetSubgraphs() const {
  std::vector<gsl::not_null<const Graph*>> subgraphs;
  subgraphs.reserve(attr_to_subgraph_map_.size());
  using value_type = std::unordered_map<std::string, gsl::not_null<Graph*>>::value_type;
  std::transform(attr_to_subgraph_map_.cbegin(), attr_to_subgraph_map_.cend(), std::back_inserter(subgraphs),
                 [](const value_type& entry) { return entry.second; });

  return subgraphs;
}

void Node::ForEachDef(std::function<void(const onnxruntime::NodeArg&, bool is_input)> func,
                      bool include_missing_optional_defs) const {
  for (const auto* arg : InputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, true);
  }

  for (const auto* arg : ImplicitInputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, true);
  }

  for (const auto* arg : OutputDefs()) {
    if (include_missing_optional_defs || arg->Exists())
      func(*arg, false);
  }
};

// Constructor: Given a <GraphProto> loaded from model file, construct
// a <Graph> object and Resolve() it.
//Status Graph::LoadGraph(const GraphProto& graph_proto,
//                        const std::unordered_map<std::string, int>& domain_to_version,
//                        Version ir_version,
//                        std::unique_ptr<Graph>& new_graph) {
//  // create instance. need to call private ctor so can't use make_unique
//  GSL_SUPPRESS(r .11)
//  new_graph.reset(new Graph(nullptr, &graph_proto, domain_to_version, ir_version));
//
//  // as we just loaded from file we want to fully initialize/Resolve, but not let that change
//  // the proto sync flag
//  ResolveOptions options;
//  options.no_proto_sync_required = true;
//  auto status = new_graph->Resolve(options);
//  return status;
//}
using google::protobuf::RepeatedPtrField;

#if !defined(ORT_MINIMAL_BUILD)

Graph::Graph(const Model& owning_model,
             GraphProto* graph_proto,
             const std::unordered_map<std::string, int>& domain_to_version,
             Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
             const logging::Logger& logger)
    : Graph(owning_model, graph_proto, domain_to_version, ir_version, schema_registry, nullptr, nullptr, logger) {}

Graph::Graph(const Model& owning_model,
             GraphProto* graph_proto, const std::unordered_map<std::string, int>& domain_to_version, Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry, Graph* parent_graph, const Node* parent_node,
             const logging::Logger& logger)
    : owning_model_(owning_model),
      graph_proto_(graph_proto),
      schema_registry_(schema_registry),
      graph_resolve_needed_(true),
      domain_to_version_(domain_to_version),
      ir_version_(ir_version),
      using_latest_onnx_opset_(UsingLatestOnnxOpset(domain_to_version)),
      parent_graph_(parent_graph),
      parent_node_(parent_node),
      logger_(logger),
      is_loaded_from_model_file_(GraphLoadedFromModelFile(graph_proto_)) {
  ORT_ENFORCE(graph_proto != nullptr, "graph_proto cannot be null");
  ArgNameToTypeMap name_to_type_map;
  const auto& model_path = ModelPath();

  // Process 'Constant' nodes
  // Put the 'TensorProto' stored in the 'Constant' nodes attribute into the graphs initializer list
  for (auto& node : graph_proto_->node()) {
    if (node.op_type() != kConstant) {
      continue;
    }

    const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
    auto status = utils::ConstantNodeProtoToTensorProto(node, model_path, *tensor);
    ORT_ENFORCE(status.IsOK(), status.ToString());
    if (node.attribute(0).type() == AttributeProto_AttributeType_SPARSE_TENSOR) {
      auto p = sparse_tensor_names_.emplace(tensor->name());
      ORT_ENFORCE(p.second, "Duplicate constant node sparse initializer name: '", tensor->name(), "' Model is invalid.");
    }
  }

  // Remove constant nodes as they're replaced with initializers above.
  const gsl::not_null<RepeatedPtrField<NodeProto>*> graph_mutable_nodes{graph_proto_->mutable_node()};
  graph_mutable_nodes->erase(
      std::remove_if(graph_mutable_nodes->begin(), graph_mutable_nodes->end(),
                     [](NodeProto& p) {
                       return (p.op_type() == kConstant);
                     }),
      graph_mutable_nodes->end());

  // For now we convert sparse_intializer to dense tensors
  // since there are currently no supported ops that consume sparse
  // initializers directly. We remove them from graph_proto. We will reconstitute them
  // when saving to ORT format to save space on disk.
  if (graph_proto_->sparse_initializer_size() > 0) {
    for (const auto& sparse_tensor : graph_proto_->sparse_initializer()) {
      ORT_ENFORCE(utils::HasName(sparse_tensor), "Sparse initializer must have a name. This model is invalid");
      const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
      auto status = utils::SparseTensorProtoToDenseTensorProto(sparse_tensor, model_path, *tensor);
      ORT_ENFORCE(status.IsOK(), status.ToString());
      auto p = sparse_tensor_names_.emplace(tensor->name());
      ORT_ENFORCE(p.second, "Duplicate sparse_tensor_initializer: '", tensor->name(), "' Model is invalid.");
    }

    // Remove sparse_initializers from protobuf to save memory as they are converted to dense now
    graph_proto_->mutable_sparse_initializer()->Clear();
    const int sparse_num_cleared = graph_proto_->sparse_initializer().ClearedCount();
    for (int i = 0; i < sparse_num_cleared; ++i) {
      delete graph_proto_->mutable_sparse_initializer()->ReleaseCleared();
    }
  }

  // Collect all node arg name, type, shape information in the graph.
  // type/shape information will be assigned to each node arg when going
  // thru all nodes later.

  // process graph inputs first as we want the type/shape from them to be preferred if a graph input
  // has a matching initializer
  for (auto& graph_input : graph_proto_->input()) {
    if (utils::HasName(graph_input)) {
      if (utils::HasType(graph_input)) {
        name_to_type_map[graph_input.name()] = graph_input.type();
        GetOrCreateNodeArg(graph_input.name(), &graph_input.type());
      } else {
        // subgraph inputs can have type inferred later. need to create a NodeArg in case this input is only used in
        // a nested subgraph (a NodeArg won't be added by AddNode for the nodes in this subgraph)
        if (IsSubgraph()) {
          GetOrCreateNodeArg(graph_input.name(), nullptr);
        }
      }
    }
  }

  // Copy initial tensors to a map.
  for (auto& tensor : graph_proto_->initializer()) {
    auto p = name_to_initial_tensor_.emplace(tensor.name(), &tensor);
    if (!p.second) {
      LOGS(logger_, WARNING) << "Duplicate initializer (dense, sparse or ConstantNode): '" << tensor.name()
                             << "' the model will use the latest encountered initializer"
                             << ". Please, fix your model.";
      p.first->second = &tensor;
    }

    NodeArg* matching_graph_input = GetNodeArg(tensor.name());
    TypeProto t{TypeProtoFromTensorProto(tensor)};

    if (!utils::HasElemType(t.tensor_type())) {
      ORT_THROW("This is an invalid model. Tensor does not have type information.");
    }

    if (ir_version_ < 4) {
      // initializers can have matching graph inputs but are treated as constant,
      // so we prefer the shape from the initializer
      name_to_type_map[tensor.name()] = t;
      if (matching_graph_input != nullptr) {
        ORT_THROW_IF_ERROR(matching_graph_input->UpdateTypeAndShape(t, true, false, logger));
      }
    } else {
      // v4 and later allows a constant initializer with no matching graph input. create a NodeArg for these.
      // otherwise we prefer the shape from the graph input so leave matching_graph_input as is.
      if (matching_graph_input == nullptr) {
        name_to_type_map[tensor.name()] = t;
        ORT_IGNORE_RETURN_VALUE(GetOrCreateNodeArg(tensor.name(), &t));
      } else {
        LOGS(logger_, WARNING) << "Initializer " << tensor.name()
                               << " appears in graph inputs and will not be treated as constant value/weight. "
                               << "This may prevent some of the graph optimizations, like const folding. "
                               << "Move it out of graph inputs if there is no need to override it, "
                               << "by either re-generating the model with latest exporter/converter "
                               << "or with the tool onnxruntime/tools/python/remove_initializer_from_input.py.";
      }
    }
  }

  for (auto& graph_output : graph_proto_->output()) {
    if (utils::HasName(graph_output) && utils::HasType(graph_output)) {
      auto& name = graph_output.name();
      name_to_type_map[name] = graph_output.type();
      // always create NodeArg for graph output, in case it's from initializer
      GetOrCreateNodeArg(name, &graph_output.type());
    }
  }

  for (auto& node_arg : graph_proto_->value_info()) {
    if (utils::HasName(node_arg) && utils::HasType(node_arg)) {
      name_to_type_map[node_arg.name()] = node_arg.type();
    }
  }

  for (const auto& node_proto : graph_proto_->node()) {
    AddNode(node_proto, name_to_type_map);
  }

  if (is_loaded_from_model_file_) {
    InitializeStateFromModelFileGraphProto();
  }
}

Graph::Graph(Graph& parent_graph, const Node& parent_node, ONNX_NAMESPACE::GraphProto& subgraph_proto)
    : Graph(parent_graph.owning_model_,
            &subgraph_proto,
            parent_graph.DomainToVersionMap(), parent_graph.IrVersion(), parent_graph.schema_registry_,
            &parent_graph,
            &parent_node, parent_graph.logger_) {
}

void Graph::InitializeStateFromModelFileGraphProto() {
  ORT_ENFORCE(
      graph_inputs_excluding_initializers_.empty() && graph_inputs_including_initializers_.empty() &&
          value_info_.empty() && graph_outputs_.empty(),
      "Graph state to be loaded into must be empty.");

  // Name to NodeArg mapping of all graph initializers.
  std::unordered_map<std::string, const NodeArg*> graph_initializers;

  // Name to NodeArg mapping of all graph inputs.
  std::unordered_map<std::string, const NodeArg*> graph_inputs;

  // Name to NodeArg mapping of all graph node outputs.
  std::unordered_map<std::string, const NodeArg*> nodes_outputs;

  for (auto& initializer : graph_proto_->initializer()) {
    auto& initializer_name = initializer.name();
    auto initializer_arg = GetNodeArg(initializer_name);
    graph_initializers.insert({initializer_name, initializer_arg});
  }

  // Set graph inputs.
  // <graph_inputs_including_initializers_> contains inputs exactly specified in proto.
  // <graph_inputs_excluding_initializers_> contains inputs without default value (specified as initializer).
  for (auto& graph_input : graph_proto_->input()) {
    auto& name = graph_input.name();
    const auto* node_arg = GetNodeArg(name);
    ORT_ENFORCE(node_arg, "Graph ctor should have created NodeArg for initializer. Missing:", name);
    graph_inputs.insert({name, node_arg});
    graph_inputs_including_initializers_.push_back(node_arg);
    if (graph_initializers.end() == graph_initializers.find(name)) {
      graph_inputs_excluding_initializers_.push_back(node_arg);
    }
  }

  for (const auto& node : Nodes()) {
    for (const auto* output_def : node.OutputDefs()) {
      nodes_outputs.insert({output_def->Name(), output_def});
    }
  }

  // Set graph outputs.
  // Graph outputs specified in the model must be nodes' outputs, initializer or graph inputs.
  for (auto& graph_output : graph_proto_->output()) {
    auto& graph_output_name = graph_output.name();
    auto iter = nodes_outputs.find(graph_output_name);
    if (nodes_outputs.end() == iter) {
      // Graph output is not found as any node's output.
      auto iter2 = graph_initializers.find(graph_output_name);
      if (graph_initializers.end() == iter2) {
        // Graph output is not found as any initializer.
        auto iter3 = graph_inputs.find(graph_output_name);
        if (graph_inputs.end() == iter3) {
          // Graph output is not found as any graph input.
          ORT_THROW(
              "This is an invalid model. Graph output (", graph_output_name,
              ") does not exist in the graph.");
        }
        graph_outputs_.push_back(iter3->second);
        continue;
      }
      graph_outputs_.push_back(iter2->second);
      continue;
    }
    graph_outputs_.push_back(iter->second);
  }

  // Set graph value_info_.
  for (const auto& graph_value_info : graph_proto_->value_info()) {
    const auto& name = graph_value_info.name();
    const auto* node_arg = GetNodeArg(name);
    if (node_arg != nullptr) {
      value_info_.push_back(node_arg);
    }
  }

  ComputeOverridableInitializers();
}

Status Graph::VerifyNoDuplicateName() {
  auto& inputs_and_initializers = resolve_context_.inputs_and_initializers;
  auto& output_args = resolve_context_.output_args;
  auto& node_name_to_index = resolve_context_.node_name_to_index;

  output_args.clear();
  node_name_to_index.clear();
  // inputs_and_initializers: this is passed in as a parameter, since functions don't have initializers
  // but graphs have them.

  for (auto& node : Nodes()) {
    // Verify node name should be unique.
    auto& node_name = node.Name();

    if (!node_name.empty() && node_name_to_index.end() != node_name_to_index.find(node_name)) {
      // The node has name and its name was used by another node.
      Status status(ONNXRUNTIME, FAIL,
                    "This is an invalid model. Error: two nodes with same node name (" + node_name + ").");
      return status;
    }

    node_name_to_index[node_name] = node.Index();

    // Verify node outputs' name should be unique.
    int output_index = -1;
    for (const auto* output_def : node.OutputDefs()) {
      ++output_index;
      if (output_def->Exists()) {
        auto& output_arg_name = output_def->Name();
        if (inputs_and_initializers.count(output_arg_name)) {
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
        auto result = output_args.insert({output_arg_name, {&node, output_index}});
        if (!result.second) {
          // Two outputs with same name, so that insertion fails.
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
      }
    }
  }
  return Status::OK();
}

// Recurse into any subgraphs to update the list of NodeArg values in outer scope.
// This information is needed to resolve any dependencies on outer scope values.
common::Status Graph::SetOuterScopeNodeArgs(const std::unordered_set<std::string>& outer_scope_node_args) {
  resolve_context_.outer_scope_node_args = outer_scope_node_args;

  if (!resolve_context_.nodes_with_subgraphs.empty()) {
    // Build the list of NodeArg's that are valid for a subgraph of this GraphBase instance:
    //   - outer scope for this graph
    //   - any inputs/initializers from this graph
    //   - any outputs from nodes in this graph
    //
    // We provide outputs from all nodes in this graph at this stage.
    // BuildConnections will link the node with the subgraph to any outer scope Node/NodeArgs it consumes.
    // PerformTopologicalSortAndCheckIsAcyclic will validate these links.
    std::unordered_set<std::string> node_args_in_scope_for_subgraph = outer_scope_node_args;

    node_args_in_scope_for_subgraph.insert(resolve_context_.inputs_and_initializers.cbegin(),
                                           resolve_context_.inputs_and_initializers.cend());

    std::transform(resolve_context_.output_args.cbegin(), resolve_context_.output_args.cend(),
                   std::inserter(node_args_in_scope_for_subgraph, node_args_in_scope_for_subgraph.end()),
                   [](const std::pair<std::string, std::pair<Node*, int>>& entry) { return entry.first; });

    for (auto* node : resolve_context_.nodes_with_subgraphs) {
      for (auto& subgraph : node->MutableSubgraphs()) {
        auto status = subgraph->SetOuterScopeNodeArgs(node_args_in_scope_for_subgraph);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void Graph::AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  if (nodes_.size() <= src_node_index || src_arg_slot < 0 || nodes_.size() <= dst_node_index || dst_arg_slot < 0 ||
      nullptr == nodes_[src_node_index] || nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ORT_THROW("Invalid node indexes specified when adding edge.");
  }

  NodeArg* src_arg = nullptr;
  NodeArg* dst_arg = nullptr;
  if (nodes_[src_node_index]->MutableDefinitions().output_defs.size() > static_cast<size_t>(src_arg_slot)) {
    src_arg = nodes_[src_node_index]->MutableDefinitions().output_defs[src_arg_slot];
  }

  if (nullptr == src_arg) {
    ORT_THROW("Invalid source node arg slot specified when adding edge.");
  }

  auto& dst_node_defs = nodes_[dst_node_index]->MutableDefinitions();
  NodeArg** dst_arg_pointer = nullptr;
  if (dst_node_defs.input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
    dst_arg_pointer = &dst_node_defs.input_defs[dst_arg_slot];
    dst_arg = *dst_arg_pointer;
  } else {
    auto num_of_explicit_inputs = dst_node_defs.input_defs.size();
    if (num_of_explicit_inputs + dst_node_defs.implicit_input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
      dst_arg_pointer = &dst_node_defs.implicit_input_defs[dst_arg_slot - num_of_explicit_inputs];
      dst_arg = *dst_arg_pointer;
    }
  }
  if (nullptr == dst_arg) {
    ORT_THROW("Invalid destination node arg slot specified when adding edge.");
  }

  if (src_arg != dst_arg) {
    if (src_arg->Type() != dst_arg->Type()) {
      // The output type of source node arg does not match the input type of destination node arg.
      ORT_THROW("Argument type mismatch when adding edge.");
    }
    *dst_arg_pointer = src_arg;
  }

  nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index], src_arg_slot, dst_arg_slot));
  nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index], src_arg_slot, dst_arg_slot));
}

void Graph::RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  if (nodes_.size() <= src_node_index || src_arg_slot < 0 || nodes_.size() <= dst_node_index || dst_arg_slot < 0 ||
      nullptr == nodes_[src_node_index] || nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ORT_THROW("Invalid node indexes specified when removing edge.");
  }

  const NodeArg* src_arg = nullptr;
  const NodeArg* dst_arg = nullptr;
  if (nodes_[src_node_index]->GetDefinitions().output_defs.size() > static_cast<size_t>(src_arg_slot)) {
    src_arg = nodes_[src_node_index]->GetDefinitions().output_defs[src_arg_slot];
  }

  if (nullptr == src_arg) {
    ORT_THROW("Invalid source node arg slot specified when removing edge.");
  }

  auto& dst_node_defs = nodes_[dst_node_index]->GetDefinitions();
  if (dst_node_defs.input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
    dst_arg = dst_node_defs.input_defs[dst_arg_slot];
  } else {
    auto num_of_explicit_inputs = dst_node_defs.input_defs.size();
    if (num_of_explicit_inputs + dst_node_defs.implicit_input_defs.size() > static_cast<size_t>(dst_arg_slot)) {
      dst_arg = dst_node_defs.implicit_input_defs[dst_arg_slot - num_of_explicit_inputs];
    }
  }
  if (nullptr == dst_arg) {
    ORT_THROW("Invalid destination node arg slot specified when removing edge.");
  }

  if (src_arg != dst_arg) {
    // The edge ends specified by source and destination arg slot are not referring to same node arg.
    // It means there was no edge between these two slots before.
    ORT_THROW("Argument mismatch when removing edge.");
  }

  nodes_[dst_node_index]->MutableRelationships().input_edges.erase(Node::EdgeEnd(*nodes_[src_node_index], src_arg_slot, dst_arg_slot));
  nodes_[src_node_index]->MutableRelationships().output_edges.erase(Node::EdgeEnd(*nodes_[dst_node_index], src_arg_slot, dst_arg_slot));
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
GSL_SUPPRESS(es .84)  // ignoring return value from unordered_map::insert causes noisy complaint
Status Graph::BuildConnections(std::unordered_set<std::string>& outer_scope_node_args_consumed) {
  const std::unordered_set<std::string>& outer_scope_node_args = resolve_context_.outer_scope_node_args;
  std::unordered_set<Node*> inner_nodes;

  std::unordered_set<std::string> node_args_consumed_by_subgraphs;

  // recurse into subgraphs first so we can update any nodes in this graph that are used by those subgraphs
  if (!resolve_context_.nodes_with_subgraphs.empty()) {
    for (auto* node : resolve_context_.nodes_with_subgraphs) {
      for (auto& subgraph : node->MutableSubgraphs()) {
        std::unordered_set<std::string> node_args_consumed;
        ORT_RETURN_IF_ERROR(subgraph->BuildConnections(node_args_consumed));

        for (auto& node_arg_name : node_args_consumed) {
          auto node_arg = GetNodeArg(node_arg_name);

          if (node_arg == nullptr) {
            // it's a node arg from outside this graph's scope, so add that to the list we return
            // so that we can add the dependency at the next level up. this happens if you have multiple
            // levels of subgraphs between the graph with the original NodeArg and the subgraph with implicit usage.
            ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(node_arg_name));

            if (!parent_graph_) {
              return ORT_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "This is an invalid model. At top level graph without matching NodeArg that subgraph consumes. Name=",
                  node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }

            node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);

            // make sure the node arg is found in the parent graph/s
            if (!node_arg) {
              return ORT_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "This is an invalid model. Failed to find NodeArg in all parent graphs. Name=", node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }
          } else {
            // this value may be produced by this graph, or it could still be coming from a parent graph if it
            // is also directly consumed at this level as we create a NodeArg for all Node inputs in this graph.
            // due to that we need to check the outputs from this level to determine if it is an outer scope value.
            // we don't have that info yet so store and check before returning from BuildConnections
            ORT_IGNORE_RETURN_VALUE(node_args_consumed_by_subgraphs.insert(node_arg_name));
          }

          // add it to the Node's list of implicit inputs
          auto& implicit_inputs = node->MutableDefinitions().implicit_input_defs;
          int input_slot_index = static_cast<int>(node->GetDefinitions().input_defs.size());
          auto iter = std::find(implicit_inputs.cbegin(), implicit_inputs.cend(), node_arg);
          if (implicit_inputs.cend() == iter) {
            implicit_inputs.push_back(node_arg);
            input_slot_index += static_cast<int>(implicit_inputs.size() - 1);
          } else {
            input_slot_index += static_cast<int>(iter - implicit_inputs.cbegin());
          }

          auto entry = resolve_context_.output_args.find(node_arg_name);
          if (entry != resolve_context_.output_args.end()) {
            // Create relationship between this node (node), and the node providing the output (output_node).
            Node& output_node = *entry->second.first;
            AddEdge(output_node.Index(), node->Index(), entry->second.second, input_slot_index);

            inner_nodes.insert(&output_node);

            // If this Graph was built manually, remove the implicit input from the graph outputs
            // if it is present there and not explicitly listed in the ordered graph outputs
            // (as that implies we should leave it as an output).
            // If the Graph was loaded from a GraphProto, honor the explicit graph outputs and leave as is.
            if (!is_loaded_from_model_file_) {
              graph_outputs_.erase(std::remove(graph_outputs_.begin(), graph_outputs_.end(), node_arg),
                                   graph_outputs_.end());
            }
          }
        }
      }
    }
  }

  // now build connections within this Graph instance
  node_arg_to_producer_node_.clear();
  node_arg_to_consumer_nodes_.clear();
  for (auto& node : Nodes()) {
    // Need mutable input defs to be able to set any outer scope NodeArg implicit inputs
    auto& input_args = node.MutableInputDefs();
    auto& output_args = node.MutableOutputDefs();

    if (!output_args.empty()) {
      for (const auto* output_arg : output_args) {
        if (output_arg->Exists()) {
          node_arg_to_producer_node_.insert({output_arg->Name(), node.Index()});
        }
      }
    }

    if (!input_args.empty()) {
      // This node needs inputs.

      int input_slot_index = -1;
      for (const auto* input_arg : input_args) {
        ++input_slot_index;
        if (!input_arg->Exists()) {
          // This input could be optional and it does not exist in this case.
          continue;
        }

        node_arg_to_consumer_nodes_[input_arg->Name()].insert(node.Index());
        const auto& input_arg_name = input_arg->Name();
        auto output_arg_iter = resolve_context_.output_args.find(input_arg_name);
        if (resolve_context_.output_args.end() != output_arg_iter) {
          // The input to this node is an output from a previous node in this graph.
          // Create relationship between this node (node), and the node providing the output (output_node).
          Node& output_node = *output_arg_iter->second.first;
          AddEdge(output_node.Index(), node.Index(), output_arg_iter->second.second, input_slot_index);

          inner_nodes.insert(&output_node);
        } else {
          // the value is either an input, an initializer, or coming from outer scope. we only need to take action
          // if coming from outer scope, so first check if this is a subgraph (otherwise there is no outer scope).
          if (parent_graph_ != nullptr) {
            // make sure it's not an input or initializer first as those override any outer scope values
            if (resolve_context_.inputs_and_initializers.find(input_arg_name) ==
                resolve_context_.inputs_and_initializers.cend()) {
              // If it is present in the outer scope it will be 'fed' by the execution frame
              // providing access to the OrtValue from the outer scope. Pass the name back up so nodes can
              // be linked correctly at that level.
              if (outer_scope_node_args.find(input_arg_name) != outer_scope_node_args.cend()) {
                ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(input_arg_name));
              }
            }
          } else {
            // Check all the inputs are found.
            //
            // Ignore a Fused node as it could have moved things like initializers to a different device
            // (they're internally available to the fused node but removed from the Graph instance).
            // Fusion happens after the model was loaded in full so we know the inputs were valid originally.
            bool check = node.NodeType() != Node::Type::Fused;
#if defined(ENABLE_TRAINING)
            // Only check initial model load for training as graph modifications there also render inputs 'invalid'.
            check = check && num_resolves_ == 0;
#endif
            if (check &&
                resolve_context_.inputs_and_initializers.find(input_arg_name) ==
                    resolve_context_.inputs_and_initializers.cend() &&
                // if we're manually creating a Graph for use as a subgraph the outer scope names are manually set
                outer_scope_node_arg_names_.find(input_arg_name) == outer_scope_node_arg_names_.cend()) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid model. Node input '", input_arg_name,
                                     "' is not a graph input, initializer, or output of a previous node.");
            }
          }
        }
      }
    } else if (node.OutputDefs().empty()) {
      // This is a useless node.
      // It has no input/output.
      RemoveNode(node.Index());
    }
  }

  // finally check any node args consumed by subgraphs to see if they're available locally.
  // if not we add them to the list of outer scope values consumed.
  for (const auto& name : node_args_consumed_by_subgraphs) {
    if (node_arg_to_producer_node_.count(name) == 0 &&
        resolve_context_.inputs_and_initializers.find(name) == resolve_context_.inputs_and_initializers.cend()) {
      ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(name));
    }
  }

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

NodeArg* Graph::GetNodeArgIncludingParentGraphs(const std::string& node_arg_name) {
  NodeArg* node_arg = GetNodeArg(node_arg_name);

  if (!node_arg && parent_graph_) {
    node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);
  }

  return node_arg;
}

void Graph::ReverseDFSFrom(const std::vector<NodeIndex>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  std::vector<const Node*> node_vec;
  node_vec.reserve(from.size());
  for (auto i : from) {
    node_vec.push_back(GetNode(i));
  }

  ReverseDFSFrom(node_vec, enter, leave, comp, {});
}

void Graph::ReverseDFSFrom(const std::vector<const Node*>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  ReverseDFSFrom(from, enter, leave, comp, {});
}

void Graph::ReverseDFSFrom(const std::vector<const Node*>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp,
                           const std::function<bool(const Node* from, const Node* to)>& stop) const {
  using WorkEntry = std::pair<const Node*, bool>;  // bool represents leave or not
  std::vector<WorkEntry> stack(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    stack[i] = WorkEntry(from[i], false);
  }

  std::vector<bool> visited(MaxNodeIndex(), false);
  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    if (last_entry.first == nullptr) {
      continue;
    }
    const Node& n = *last_entry.first;

    if (last_entry.second) {
      // leave node
      leave(&n);
      continue;
    }

    if (visited[n.Index()]) continue;

    visited[n.Index()] = true;

    if (enter) enter(&n);

    if (leave) stack.emplace_back(&n, true);

    if (comp) {
      std::vector<const Node*> sorted_nodes;
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        sorted_nodes.push_back(&(*iter));
      }
      std::sort(sorted_nodes.begin(), sorted_nodes.end(), comp);
      for (const auto* in : sorted_nodes) {
        const NodeIndex idx = in->Index();
        if (!visited[idx]) {
          stack.emplace_back(in, false);
        }
      }
    } else {
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        const NodeIndex idx = (*iter).Index();
        if (!visited[idx]) {
          stack.emplace_back(GetNode(idx), false);
        }
      }
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)
void Graph::KahnsTopologicalSort(const std::function<void(const Node*)>& enter,
                                 const std::function<bool(const Node*, const Node*)>& comp) const {
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::priority_queue<const Node*, std::vector<const Node*>, decltype(comp)> to_visit(comp);
  std::vector<NodeIndex> topo_order;

  for (auto& node : Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    in_degree.insert({node.Index(), input_edge_count});
    if (input_edge_count == 0) {
      to_visit.push(&node);
    }
  }

  while (!to_visit.empty()) {
    const Node* current = to_visit.top();
    to_visit.pop();

    if (!current) continue;

    if (enter) {
      enter(current);
    }

    for (auto node_it = current->OutputNodesBegin(); node_it != current->OutputNodesEnd(); ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        to_visit.push(&*node_it);
      }
    }
    topo_order.push_back(current->Index());
  }

  if (NumberOfNodes() != static_cast<int>(topo_order.size())) {
    ORT_THROW("Some nodes are not included in the topological sort, graph have a cycle.");
  }
}

GSL_SUPPRESS(es .84)  // noisy warning about ignoring return value from insert(...)
Status Graph::PerformTopologicalSortAndCheckIsAcyclic() {
  nodes_in_topological_order_.clear();
  std::unordered_set<NodeIndex> downstream_nodes;  // nodes downstream of the node we're currently checking
  std::unordered_set<NodeIndex> nodes_seen;        // nodes we have seen but may not have been added to nodes_added yet
  std::unordered_set<NodeIndex> nodes_added;       // nodes added to topo order
  std::stack<NodeIndex> stack;

  // push the root nodes into nodes_in_topological_order in the order they were defined in the model
  // to ensure that is consistent.
  auto& nodes_in_original_order = Nodes();
  std::for_each(nodes_in_original_order.cbegin(), nodes_in_original_order.cend(),
                [&](const Node& node) {
                  auto index = node.Index();

                  // find the top level nodes in the graph.
                  // need to also consider nodes that only have Constants as inputs as top level nodes,
                  // as the constant will get replaced by an initializer.
                  auto input_edges = node.GetRelationships().input_edges;
                  auto has_inputs = std::any_of(input_edges.cbegin(), input_edges.cend(),
                                                [](const Node::EdgeEnd& edge) {
                                                  return edge.GetNode().OpType() != kConstant;
                                                });

                  if (!has_inputs) {
                    // add to the topological list, and ensure we skip these nodes when walking the graph
                    nodes_in_topological_order_.push_back(index);
                    nodes_added.insert(index);
                    nodes_seen.insert(index);
                  }
                });

  // find all the leaf nodes (nodes with no output edges as there's no edge to a graph output)
  for (auto iter = Nodes().begin(); iter != Nodes().end(); ++iter) {
    if (iter->relationships_.output_edges.empty()) {
      stack.push(iter->Index());
    }
  }

  // work our way up from the leaf nodes
  while (!stack.empty()) {
    const NodeIndex current = stack.top();
    stack.pop();

    if (nodes_added.find(current) != nodes_added.end()) {
      continue;
    }

    if (nodes_seen.find(current) != nodes_seen.end()) {
      // we popped the stack and are back to a node that was seen previously,
      // so we know all the upstream nodes from it have been added.
      nodes_in_topological_order_.push_back(current);
      nodes_added.insert(current);
      downstream_nodes.erase(current);
      continue;
    }

    const Node* node = GetNode(current);
    if (!node) {
      continue;
    }

    // node hasn't been seen before, so mark it as seen and re-add it along with its inputs
    // also mark it as downstream of anything new that is added to the stack to detect acyclic graphs
    nodes_seen.insert(current);
    downstream_nodes.insert(current);

    stack.push(current);

    for (auto iter = node->InputNodesBegin(), end = node->InputNodesEnd(); iter != end; ++iter) {
      const NodeIndex idx = iter->Index();
      // the input to this node is also downstream of this node
      if (downstream_nodes.find(idx) != downstream_nodes.end()) {
        Status status(ONNXRUNTIME, FAIL, "This is an invalid model. Error: the graph is not acyclic.");
        return status;
      }

      // avoid re-processing nodes
      if (nodes_seen.find(idx) == nodes_seen.end()) {
        stack.push(idx);
      }
    }
  }

  if (num_of_nodes_ >= 0 && static_cast<size_t>(num_of_nodes_) == nodes_in_topological_order_.size()) {
    return Status::OK();
  }

  return Status(ONNXRUNTIME, FAIL, "This is an invalid model. Error: the graph is not acyclic.");
}

bool FullyDefinedType(const TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case TypeProto::kTensorType: {
      auto& tensor_type = type_proto.tensor_type();
      return utils::HasElemType(tensor_type);
    }
    case TypeProto::kSparseTensorType: {
      auto& tensor_type = type_proto.sparse_tensor_type();
      return utils::HasElemType(tensor_type);
    }
    case TypeProto::kSequenceType: {
      auto& seq_type = type_proto.sequence_type();
      return utils::HasElemType(seq_type) && FullyDefinedType(seq_type.elem_type());
    }
    case TypeProto::kMapType: {
      auto& map_type = type_proto.map_type();
      return utils::HasKeyType(map_type) &&
             utils::HasValueType(map_type) &&
             FullyDefinedType(map_type.value_type());
    }
    case TypeProto::kOpaqueType:
      return true;
    case TypeProto::VALUE_NOT_SET:
    default:
      return false;
  }
}

// function to handle type/shape inferencing of a subgraph.
// parameters are the Graph instance for the subgraph, the input types from the control flow node that contains
// the subgraph, and the vector to write the output from the inferencing.
using SubgraphInferencingFunc =
    std::function<Status(const Node&, Graph&, const std::vector<const TypeProto*>&, std::vector<const TypeProto*>&, const Graph::ResolveOptions&)>;

class GraphInferencerImpl : public ONNX_NAMESPACE::GraphInferencer {
 public:
  GraphInferencerImpl(const Node& node, Graph& graph, SubgraphInferencingFunc& inferencing_func, const Graph::ResolveOptions& options)
      : node_(node), graph_(graph), inferencing_func_(inferencing_func), options_(options) {
  }

  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  // We ignore input_data currently as the inferencing happens prior to receiving user input.
  std::vector<const TypeProto*> doInferencing(const std::vector<const TypeProto*>& input_types,
                                              const std::vector<const TensorProto*>& /*input_data*/) override {
    std::vector<const TypeProto*> output_types;

    auto status = inferencing_func_(node_, graph_, input_types, output_types, options_);

    if (status != Status::OK()) {
      fail_type_inference("Graph attribute inferencing failed: ", status.ErrorMessage());
    }

    return output_types;
  }

 private:
  const Node& node_;
  Graph& graph_;
  SubgraphInferencingFunc& inferencing_func_;
  const Graph::ResolveOptions& options_;
};

// An implementation of the InferenceContext interface required by operator-specific
// shape inference for onnxruntime graphs.
class InferenceContextImpl : public ONNX_NAMESPACE::InferenceContext {
  using AttributeGraphMap = std::unordered_map<std::string, Graph*>;

 public:
  InferenceContextImpl(Node& node,
                       SubgraphInferencingFunc subgraph_inferencing_func,
                       const Graph& graph,
                       const Graph::ResolveOptions& options) noexcept
      : node_(node),
        subgraph_inferencing_func_(subgraph_inferencing_func),
        graph_(graph),
        options_(options) {
    node_output_types_.resize(node.OutputDefs().size());
  }

  void RunInferencing() {
    auto schema = node_.Op();
    if (nullptr != schema) {
      schema->GetTypeAndShapeInferenceFunction()(*this);
    }
  }

  std::vector<TypeProto> InferredOutputTypes() const { return node_output_types_; }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto& attribute_value_map = node_.GetAttributes();
    auto iter = attribute_value_map.find(name);
    if (iter == attribute_value_map.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  size_t getNumInputs() const noexcept override {
    return node_.InputDefs().size();
  }

  const TypeProto* getInputType(size_t index) const override {
    const TypeProto* type = nullptr;
    auto p_node_arg = node_.InputDefs().at(index);
    if ((nullptr != p_node_arg) && p_node_arg->Exists()) {
      type = p_node_arg->TypeAsProto();
    }

    return type;
  }

  size_t getNumOutputs() const noexcept override {
    return node_output_types_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    return &node_output_types_[index];
  }

  const TensorProto* getInputData(size_t index) const override {
    auto def = node_.InputDefs()[index];
    if (!def)
      return nullptr;

    // only return data if it's for a constant initializer. checks for outer scope initializers
    // if this is a subgraph and the name isn't found locally.
    const TensorProto* initializer = graph_.GetConstantInitializer(def->Name(), true);
    return initializer;
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    GraphInferencer* graph_inferencer = nullptr;

    auto* subgraph = node_.GetMutableGraphAttribute(attribute_name);

    if (subgraph) {
      auto inferencer = std::make_unique<GraphInferencerImpl>(node_, *subgraph, subgraph_inferencing_func_, options_);
      graph_inferencer = inferencer.get();
      graph_inferencers_.push_back(std::move(inferencer));
    } else {
      fail_type_inference("No Graph instance was found for attribute ",
                          attribute_name, " in node ", node_.Name());
    }

    return graph_inferencer;
  }

  // XXX: When we changed and kept sparse constant initializers in sparse form,
  // we would adjust this method
  const SparseTensorProto* getInputSparseData(size_t) const override {
    return nullptr;
  }

 private:
  Node& node_;
  // node_output_types_ will be populated by the operator-specific shape inference.
  std::vector<TypeProto> node_output_types_;
  SubgraphInferencingFunc subgraph_inferencing_func_;
  std::vector<std::unique_ptr<GraphInferencerImpl>> graph_inferencers_;
  const Graph& graph_;
  const Graph::ResolveOptions& options_;
};

Status Graph::InferAndVerifySubgraphTypes(const Node& node, Graph& subgraph,
                                          const std::vector<const TypeProto*>& input_types,
                                          std::vector<const TypeProto*>& output_types,
                                          const Graph::ResolveOptions& options) {
  auto status = Status::OK();

  output_types.clear();

  // the spec says all inputs should be provided for the subgraph so default to that first
  auto* subgraph_inputs = &subgraph.GetInputsIncludingInitializers();
  auto num_subgraph_inputs = subgraph_inputs->size();

  if (num_subgraph_inputs != input_types.size()) {
    // we also allow for just the required inputs to be provided to be user friendly due to ONNX requiring
    // initializers to have matching inputs (making them optional inputs that most likely the user doesn't want to
    // override).
    auto& required_subgraph_inputs = subgraph.GetInputs();
    auto num_required_subgraph_inputs = required_subgraph_inputs.size();

    if (num_required_subgraph_inputs != input_types.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Size mismatch validating subgraph inputs. Got ", input_types.size(),
                             " inputs but subgraph has ", num_subgraph_inputs,
                             " inputs and requires ", num_required_subgraph_inputs,
                             " inputs. Either provide all subgraph inputs, or just the required inputs.");
    }

    subgraph_inputs = &required_subgraph_inputs;
    num_subgraph_inputs = num_required_subgraph_inputs;
  }

  // apply type/shape info to the subgraph's inputs
  for (size_t i = 0; i < num_subgraph_inputs; ++i) {
    const auto& input_type = *input_types[i];
    const auto& subgraph_input = *subgraph_inputs->at(i);

    NodeArg* mutable_nodearg = subgraph.GetNodeArg(subgraph_input.Name());
    status = mutable_nodearg->UpdateTypeAndShape(input_type, true, options.override_types, subgraph.logger_);
    if (!status.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }
  }

  // Apply any current input type/shape information to the Nodes in the subgraph that are implicitly
  // consuming NodeArg's from this scope or higher.
  // The NodeArg's that implicit_input_defs point to would have any type/shape inferencing applied to them
  // by now. As the subgraph is referring to the outer scope NodeArg, we simply replace any information in
  // the subgraph with the details from the outer scope NodeArg.
  auto implicit_input_defs = node.GetDefinitions().implicit_input_defs;
  for (const auto* implicit_node_arg : implicit_input_defs) {
    auto subgraph_nodearg = subgraph.GetNodeArg(implicit_node_arg->Name());

    // the implicit input defs may be for a nested subgraph, so it won't necessarily match here.
    // if that is the case, we will update the type/shape information when we descend into the
    // nested subgraph later.
    if (!subgraph_nodearg)
      continue;

    status = subgraph_nodearg->UpdateTypeAndShape(*implicit_node_arg, true, options.override_types, subgraph.logger_);
    if (!status.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }

    // all values above us should have a type by now due to ONNX requirements.
    if (subgraph_nodearg->Type() == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph input missing type.");
  }

  // now that we have handled the input types, do the type/shape inferencing for the subgraph
  // to flow the type/shape info through it
  status = subgraph.PerformTypeAndShapeInferencing(options);
  ORT_RETURN_IF_ERROR(status);

  auto& subgraph_outputs = subgraph.GetOutputs();
  for (const auto* output : subgraph_outputs) {
    output_types.push_back(output->TypeAsProto());
  }

  return Status::OK();
}

Status Graph::UpdateShapeInference(Node& node) {
  // We only use this during constant folding, and we don't constant fold control flow nodes.
  ORT_ENFORCE(node.GetAttributeNameToMutableSubgraphMap().empty(),
              "UpdateTypeShapeInference is not intended to be used with control flow nodes containing subgraphs");

  // Whilst the type inferencing will run again we don't allow type overrides due to using the default
  // ResolveOptions settings, so essentially this can only change the shape information.
  return InferAndVerifyTypeMatch(node, *node.Op(), {});
}

// Implementation of type-inference and type-checking for a single node
GSL_SUPPRESS(f .23)  // spurious warning about inferred_type never being checked for null
Status Graph::InferAndVerifyTypeMatch(Node& node, const OpSchema& op, const ResolveOptions& options) {
  auto& node_name = node.Name();

  // if we're building a graph we permit outer scope node args to have no type
  // as the 'real' Resolve at runtime will have type inferencing
  auto is_outer_scope_nodearg = [this](const std::string& name) {
    return outer_scope_node_arg_names_.find(name) != outer_scope_node_arg_names_.cend();
  };

  // <k> index used to navigate node->InputDefs().
  int k = 0;
  std::unordered_map<std::string, DataType> type_parameter_to_type_map;

  for (size_t i = 0; i < node.InputArgCount().size(); ++i) {
    // Number of inputs corresponding to the i-th argument.
    const int arg_count = node.InputArgCount()[i];
    // The i-th formal parameter definition.
    auto op_formal_parameter = op.inputs()[i];

    // Check all <arg_count> actual parameters (corresponding to the k-th input)
    // match the formal parameter definition (i-th argument).
    for (int j = 0; j < arg_count; ++j, ++k) {
      auto& input_def = node.MutableDefinitions().input_defs[k];
      if (!input_def->Exists())
        continue;

      if (input_def->Type() == nullptr) {
        // if we are building a subgraph that uses outer scope values,
        // allow an empty type as it will be copied from the outer scope graph at runtime
        if (is_outer_scope_nodearg(input_def->Name()))
          continue;

        // Logic error: This should not happen if we properly checked that every use has
        // a corresponding def, for which type-inference already produced a valid type
        Status status(ONNXRUNTIME, FAIL,
                      "This is an invalid model. "
                      "Node (" +
                          node_name + ") input arg (" +
                          input_def->Name() + ") does not have type information set by parent node.");
        return status;
      }

      // Verify that the actual parameter's type is one of permitted types of the formal parameter
      DataType input_type = input_def->Type();
      auto& permitted_types = op_formal_parameter.GetTypes();
      if (0 == permitted_types.count(input_type)) {
        std::string null_pointer("(null)");
        if (input_type == nullptr) input_type = &null_pointer;
        // Type error in input model/graph.

        Status status(ONNXRUNTIME, INVALID_GRAPH,
                      "This is an invalid model. "
                      "Type Error: Type '" +
                          *input_type + "' of input parameter (" + input_def->Name() +
                          ") of operator (" + op.Name() + ") in node (" + node_name + ") is invalid.");
        return status;
      }

      // When multiple parameters have the same type-variable, they are all required
      // to have the same type. E.g., when adding tensors A and B, it is an error if
      // input A is of type "tensor(int32)" and B is of type "tensor(float)".
      // For variadic arguments, this verification rule is normally applicable:
      // e.g., Concat/Max/Mean/Min/Sum all require all input tensors to be of same type.
      // However, some ops, like the control-flow constructs (Scan, If, Loop) have variadic
      // inputs and outputs of different types. The check is not applicable to such ops.
      if (op_formal_parameter.GetIsHomogeneous()) {
        auto param_to_type_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
        if (type_parameter_to_type_map.end() == param_to_type_iter) {
          // Bind the corresponding type-parameter's value to the actual type:
          type_parameter_to_type_map[op_formal_parameter.GetTypeStr()] = input_type;
        } else if (param_to_type_iter->second != input_type) {
          // Type error in input model/graph:
          // The type-parameter T is bound to different values for different inputs.
          Status status(ONNXRUNTIME, FAIL,
                        "Type Error: Type parameter (" + op_formal_parameter.GetTypeStr() +
                            ") of Optype (" + op.Name() + ") bound to different types (" + *(param_to_type_iter->second) +
                            " and " + *(input_def->Type()) +
                            " in node (" + node_name + ").");
          return status;
        }
      }
    }
  }

  // Apply ONNX's type/shape inference to this node.
  // This will call InferAndVerifySubgraphTypes if the ONNX level type/shape inferencing for the Node attempts
  // to do subgraph type/shape inferencing (Scan/If/Loop nodes).
  // InferAndVerifySubgraphTypes will call PerformTypeAndShapeInferencing for the subgraph, which will recursively
  // handle type/shape inferencing for it.
  // Once that completes, the outputs from the node containing the subgraph will be updated, and the final values
  // returned here.
  SubgraphInferencingFunc func(Graph::InferAndVerifySubgraphTypes);
  InferenceContextImpl context(node, func, *this, options);

  {
    auto status = Status::OK();
    ORT_TRY {
      context.RunInferencing();
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node (", node.Name(), ") Op (", node.OpType(), ") ", ex.what());
      });
    }
    ORT_RETURN_IF_ERROR(status);
  }

  const auto& onnx_inferred_types(context.InferredOutputTypes());

  // Infer and verify node output arg type information.
  int i = -1;
  for (auto& output_def : node.MutableDefinitions().output_defs) {
    ++i;
    if (!output_def->Exists()) continue;

    // if the number of actual parameters exceeds the number of formal parameters,
    // then the op has variadic outputs and the trailing extra actual parameters
    // correspond to the last formal parameter. (The ONNX schema verification check
    // would have checked that the corresponding formal parameter is variadic.)

    const int num_formal_params = gsl::narrow_cast<int>(op.outputs().size());
    auto operand_index = std::min(i, num_formal_params - 1);
    auto op_formal_parameter = op.outputs().at(operand_index);

    const TypeProto& onnx_inferred_type = onnx_inferred_types[i];
    DataType existing_type = output_def->Type();
    DataType inferred_type = nullptr;

    // Infer output arg type if it is constrained to be of the same type as some input:
    // For example, the output of "Abs" is of the same type as its input.
    bool homogeneous = op_formal_parameter.GetIsHomogeneous();
    auto input_types_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
    if (homogeneous && (type_parameter_to_type_map.end() != input_types_iter)) {
      inferred_type = input_types_iter->second;
    } else if (1 == op_formal_parameter.GetTypes().size()) {
      // Infer output arg type if operator definition specifies unique output type:
      inferred_type = *(op_formal_parameter.GetTypes().begin());
    } else if (FullyDefinedType(onnx_inferred_type)) {
      // Use output type inferred by ONNX inference
      inferred_type = DataTypeUtils::ToType(onnx_inferred_type);
    } else if (existing_type != nullptr) {
      inferred_type = existing_type;
    } else {
      // This should not happen: indicates incompleteness in ONNX inference.
      Status status(ONNXRUNTIME, FAIL,
                    "Node (" + node_name + ") output arg (" + output_def->Name() + ") type inference failed");
      return status;
    }

    if ((existing_type != inferred_type) && (existing_type != nullptr)) {
      // A type exists for this output but does not match the inferred type.

      if (options.override_types) {
        // Replace existing type by inferred type: for use after graph-transformations
        // that change types of variables such as mixed-precision transformation.
        // Note: This reuses the original shape, with inferred type. Transformations
        // that can affect the shape are not yet supported.

        // The "SetType" call will override the shape information to empty.
        // If the original tensor has shape information, need to set it back.
        if (output_def->Shape()) {
          auto old_shape = *output_def->Shape();
          output_def->SetType(inferred_type);
          output_def->SetShape(old_shape);
        } else {
          output_def->SetType(inferred_type);
        }
      } else
        return Status(ONNXRUNTIME, FAIL,
                      "Type Error: Type (" + *existing_type + ") of output arg (" +
                          output_def->Name() + ") of node (" + node_name +
                          ") does not match expected type (" + *inferred_type + ").");
    }

    if (existing_type == nullptr)
      output_def->SetType(inferred_type);

    // Update output-shape if it was inferred:
    // HasShape()/GetShape() work for tensor types
    // if the behavior changes the below may need adjustment
    if (utils::HasShape(onnx_inferred_type)) {
      if (output_def->Shape() == nullptr) {
        output_def->SetShape(utils::GetShape(onnx_inferred_type));
      } else {
        // we need to merge the shapes as a subgraph may have placeholder dimensions to represent the rank
        // that have no values.
        TypeProto merge_target;
        if (utils::HasTensorType(onnx_inferred_type)) {
          *merge_target.mutable_tensor_type()->mutable_shape() = *output_def->Shape();
        } else if (utils::HasSparseTensorType(onnx_inferred_type)) {
          *merge_target.mutable_sparse_tensor_type()->mutable_shape() = *output_def->Shape();
        }
        auto status = MergeShapeInfo(output_def->Name(), onnx_inferred_type, merge_target, using_latest_onnx_opset_, logger_);
        if (!status.IsOK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node_name, " ", status.ErrorMessage());
        }
        // we may have cleared the shape if there was a mismatch so handle that
        if (utils::HasShape(merge_target))
          output_def->SetShape(utils::GetShape(merge_target));
        else
          output_def->ClearShape();
      }
    }
  }

  return Status::OK();
}

// Apply type-inference and type-checking to all inputs and initializers:
common::Status Graph::TypeCheckInputsAndInitializers() {
  // Check that the type of every input is specified:
  for (auto* graph_input : GetInputs()) {
    if (nullptr == graph_input->Type()) {
      Status status(ONNXRUNTIME, FAIL,
                    "This is an invalid model. "
                    "Model input (" +
                        graph_input->Name() + ") does not have type information.");
      return status;
    }
  }

  // Infer/check type and shape for all initializers from their values
  for (auto& initializer_pair : name_to_initial_tensor_) {
    const std::string& name = initializer_pair.first;
    auto* node_arg = GetNodeArg(name);
    // If node_arg is null, we ignore this as a potentially unused initializer here
    if (nullptr != node_arg) {
      const TensorProto* tensor_proto = initializer_pair.second;
      TypeProto tensor_type;
      tensor_type.mutable_tensor_type()->set_elem_type(tensor_proto->data_type());
      auto initializer_type = DataTypeUtils::ToType(tensor_type);
      auto nodearg_type = node_arg->Type();
      if (nullptr == nodearg_type)
        node_arg->SetType(initializer_type);
      else if (initializer_type != nodearg_type) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Type Error: Data in initializer '", name, "' has element type ", *initializer_type,
                               " but usage of initializer in graph expects ", *nodearg_type);
      }

      // Set shape accordingly.
      TensorShapeProto inferred_shape;
      for (auto dim : tensor_proto->dims()) {
        inferred_shape.add_dim()->set_dim_value(dim);
      }

      const TensorShapeProto* p_existing_shape = node_arg->Shape();
      if (nullptr == p_existing_shape) {
        // use the inferred shape if this is a constant initializer (cannot be overridden).
        // if not it has a matching graph input, and we prefer the shape info (or lack of info) from the graph input
        if (GetConstantInitializer(name, false) != nullptr) {
          node_arg->SetShape(inferred_shape);
        }
      } else {
        if (p_existing_shape->dim_size() != tensor_proto->dims_size()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Type Error: Shape of initializer ", name, " does not match. ",
                                 *p_existing_shape, " != ", *tensor_proto);
        }

        for (int i = 0; i < p_existing_shape->dim_size(); ++i) {
          auto& d = p_existing_shape->dim(i);
          if (utils::HasDimValue(d) && (d.dim_value() != tensor_proto->dims(i))) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "Type Error: Shape of initializer ", name, " does not match. ",
                                   *p_existing_shape, " != ", *tensor_proto);
          }
        }
      }
    }
  }

  return Status::OK();
}

Status Graph::VerifyNodeAndOpMatch(const ResolveOptions& options) {
  CheckerContext ctx;
  ctx.set_ir_version(gsl::narrow_cast<int>(IrVersion()));
  ctx.set_opset_imports(DomainToVersionMap());
  ctx.set_schema_registry(schema_registry_.get());
  // Set the parent directory of model path to load external tensors if exist
  ctx.set_model_dir(ToMBString(ModelPath().ParentPath().ToPathString()));

  LexicalScopeContext lsc;
  lsc.output_names.insert(resolve_context_.inputs_and_initializers.cbegin(),
                          resolve_context_.inputs_and_initializers.cend());

  // technically we could add values from Node.GetDefinitions().implicit_input_defs on a per-node basis inside
  // the below loop so that we only check against the specific outer dependencies of the node.
  // doing that requires lots of copies of LexicalScopeContext.output_names to clear out the per-Node values
  // after each loop. instead add all the outer scope values upfront so we can just accumulate new inner scope values
  // during each loop iteration.
  lsc.output_names.insert(resolve_context_.outer_scope_node_args.cbegin(),
                          resolve_context_.outer_scope_node_args.cend());

  // we may have some locally defined outer scope args if we're in the middle of constructing a subgraph
  // and need to call Resolve
  lsc.output_names.insert(outer_scope_node_arg_names_.cbegin(), outer_scope_node_arg_names_.cend());

  for (auto node_index : nodes_in_topological_order_) {
    // Node verification.
    auto& node = *GetNode(node_index);

    NodeProto node_proto;
    node.ToProto(node_proto);
    auto& node_name = node.Name();
    auto& domain = node.Domain();

    if (!node.Op()) {
      {
        auto status = Status::OK();
        ORT_TRY {
          checker::check_node(node_proto, ctx, lsc);
        }
        ORT_CATCH(const std::exception& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "This is an invalid model. Error in Node:", node_name, " : ", ex.what());
          });
        }
        ORT_RETURN_IF_ERROR(status);
      }

      auto maxInclusiveVersion = DomainToVersionMap().find(domain)->second;
      node.op_ = schema_registry_->GetSchema(node.OpType(), maxInclusiveVersion, node.Domain());

      if (node.op_ && node.op_->Deprecated()) {
        node.op_ = nullptr;
      }

      InitFunctionBodyForNode(node);

      if (!node.op_) {
        return Status(ONNXRUNTIME, FAIL, "Fatal error: " + node.OpType() + " is not a registered function/op");
      }

      node.since_version_ = node.op_->since_version();
    }

    ORT_RETURN_IF_ERROR(node.UpdateInputArgCount());

    // currently an Op is required by ValidateVersion, so we use gsl::not_null to validate that.
    // This may change in the future to allow a null Op
    const gsl::not_null<const OpSchema*> p_op{node.Op()};

    // Attribute verification and fill node attribute with
    // default value defined in operator definition if needed.
    // Fill node attribute with default value specified in operator definition if any.
    const auto& node_attributes = node.GetAttributes();
    for (const auto& attr_def : p_op->attributes()) {
      auto node_attr_iter = node_attributes.find(attr_def.first);
      if (node_attributes.end() == node_attr_iter) {
        // The attribute was not specified in the node.
        if (!attr_def.second.required) {
          if (utils::HasName(attr_def.second.default_value)) {
            // Set default value to the node attributes.
            node.AddAttribute(attr_def.first, attr_def.second.default_value);
          }
          // TODO: Handle optional attribute but no default value specified in op definition.
        } else {
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. "
                        "Node (" +
                            node_name + ") attribute (" + attr_def.first +
                            ") is required but not specified.");
          return status;
        }
      }
    }

    NO_CHANGE_ON_SYNC_FLAG(ORT_RETURN_IF_ERROR(InferAndVerifyTypeMatch(node, *p_op, options)));

    // Accumulate output names of the iterated Node
    for (auto& output_name : node_proto.output()) {
      lsc.output_names.insert(output_name);
    }
  }

  return Status::OK();
}

void Graph::InitFunctionBodyForNode(Node& node) {
  if (node.op_ && (node.op_->HasFunction() || node.op_->HasContextDependentFunction())) {
    onnx::FunctionProto onnx_function_proto;
    if (node.op_->HasContextDependentFunction()) {
      NodeProto node_proto;
      node.ToProto(node_proto);
      std::vector<TypeProto> input_types;
      for (size_t i = 0, n = node.InputDefs().size(); i < n; i++) {
        auto p_node_arg = node.InputDefs().at(i);
        if ((nullptr != p_node_arg) && p_node_arg->Exists()) {
          auto& type = *(p_node_arg->TypeAsProto());
          input_types.emplace_back(type);
        } else
          input_types.emplace_back();
      }
      onnx::FunctionBodyBuildContextImpl function_body_ctx(node_proto, input_types);
      if (!node.op_->BuildContextDependentFunction(function_body_ctx, onnx_function_proto))
        return;
    } else {
      onnx_function_proto = *(node.op_->GetFunction());
    }

    // Check function's opset requirements are compatible with model's opset.
    auto& graphImports = DomainToVersionMap();
    for (const auto& fn_import : onnx_function_proto.opset_import()) {
      auto it = graphImports.find(fn_import.domain());
      if ((it != graphImports.end()) && (it->second != fn_import.version()))
        return;  // Incompatible. Do not use this function expansion.
    }

    auto func_ptr = std::make_unique<onnxruntime::FunctionImpl>(*this, node.Index(), onnx_function_proto,
                                                                logger_);

    function_container_.emplace_back(std::move(func_ptr));
    node.SetFunctionBody(*function_container_.back());
  }
}

Status Graph::VerifyInputAndInitializerNames() {
  std::unordered_set<std::string>& inputs_and_initializers = resolve_context_.inputs_and_initializers;

  for (auto* input : GetInputs()) {
    auto result = inputs_and_initializers.insert(input->Name());
    if (!result.second) {
      Status status(ONNXRUNTIME, FAIL,
                    "Error: Duplicate definition-site for (" + input->Name() + ").");
      return status;
    }
  }

  for (auto& initializer_pair : name_to_initial_tensor_) {
    GSL_SUPPRESS(es .84)
    inputs_and_initializers.insert(initializer_pair.first);
    // Initializers are expected to be included in inputs (according to ONNX spec).
    // onnxruntime relaxes this constraint. No duplicate-name check here.
  }

  return Status::OK();
}

Status Graph::InitInputsInitializersOutputs() {
  resolve_context_.Clear();

  // clear the previous relationships, as we re-create them when resolving.
  // same applies to the implicit input defs as they are built from any subgraphs within this graph.
  for (auto& node : Nodes()) {
    node.MutableRelationships().Clear();
    node.MutableDefinitions().implicit_input_defs.clear();
  }

  // add the subgraph pointers to the resolve context.
  for (auto& node : Nodes()) {
    auto& subgraphs = node.MutableSubgraphs();
    if (!subgraphs.empty()) {
      resolve_context_.nodes_with_subgraphs.insert(&node);
    }
  }

  ORT_RETURN_IF_ERROR(SetGraphInputsOutputs());
  ORT_RETURN_IF_ERROR(VerifyInputAndInitializerNames());
  ORT_RETURN_IF_ERROR(VerifyNoDuplicateName());

  return Status::OK();
}

Status Graph::PerformTypeAndShapeInferencing(const ResolveOptions& options) {
  ORT_RETURN_IF_ERROR(TypeCheckInputsAndInitializers());

  // type/shape inferencing on the nodes is done recursively as we need subgraph outputs
  // to be applied to Node outputs for the node containing the subgraph.
  // Call path is
  // VerifyNodeAndOpMatch
  //   Iterates Nodes
  //     Runs ONNX type/shape inferencing for each Node
  //      - If it hits a node with a subgraph, InferenceContext::getGraphAttributeInferencer is called
  //        by the ONNX level type/shape inferencing, which updates the subgraph inputs using GraphInferencerImpl
  //      - GraphInferencerImpl::doInferencing calls PerformTypeShapeInferencing to execute type/shape inferencing
  //        for all nodes in the subgraph. This leads to recursively handling all subgraphs contained in the node.
  //      - once we finish processing the subgraph/s we apply resultant type/shape information to the outputs
  //        of the node that contains the subgraph.
  ORT_RETURN_IF_ERROR(VerifyNodeAndOpMatch(options));

  return Status::OK();
}

void Graph::FindAllSubgraphs(std::vector<Graph*>& subgraphs) {
  for (auto& node : Nodes()) {
    for (auto& subgraph : node.MutableSubgraphs()) {
      subgraphs.push_back(subgraph.get());
      subgraph->FindAllSubgraphs(subgraphs);
    }
  }
}

Status Graph::ForThisAndAllSubgraphs(const std::vector<Graph*>& subgraphs, std::function<Status(Graph&)> func) {
  auto status = func(*this);
  ORT_RETURN_IF_ERROR(status);

  for (auto& subgraph : subgraphs) {
    status = func(*subgraph);
    ORT_RETURN_IF_ERROR(status);
  }

  return status;
}

Status Graph::Resolve(const ResolveOptions& options) {
  if (parent_graph_) {
    // Resolve must start at the top level graph in-order to handle outer scope
    // connections correctly, so recurse up to that level to start
    return parent_graph_->Resolve(options);
  }

  // find all subgraphs including nested ones.
  std::vector<Graph*> all_subgraphs;
  FindAllSubgraphs(all_subgraphs);

  bool subgraphs_need_resolve = std::any_of(all_subgraphs.cbegin(), all_subgraphs.cend(),
                                            [](const Graph* graph) {
                                              return graph->GraphResolveNeeded();
                                            });

  if (!GraphResolveNeeded() && !subgraphs_need_resolve) {
    return Status::OK();
  }

  // init all graph/subgraphs. non-recursive.
  auto init_func = [](Graph& graph) { return graph.InitInputsInitializersOutputs(); };
  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, init_func));

  // recursively set the outer scope node args.
  ORT_RETURN_IF_ERROR(SetOuterScopeNodeArgs(resolve_context_.outer_scope_node_args));

  std::unordered_set<std::string> outer_scope_node_args_consumed;

  // recursively build connections between nodes in this graph and all subgraphs
  ORT_RETURN_IF_ERROR(BuildConnections(outer_scope_node_args_consumed));
  ORT_ENFORCE(outer_scope_node_args_consumed.empty(),
              "Shouldn't be possible to have NodeArgs that haven't been handled already.");

  // topological sort of this and any subgraphs is non-recursive
  auto topo_sort_func = [](Graph& graph) { return graph.PerformTopologicalSortAndCheckIsAcyclic(); };
  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, topo_sort_func));

  // type/shape validation and inferencing on this and any subgraphs
  // recurses into subgraphs via the ONNX checker, which descends into the GraphProto in node attributes
  // which define a subgraph.
  ORT_RETURN_IF_ERROR(PerformTypeAndShapeInferencing(options));

  // perform the final steps for this graph and all subgraphs
  auto finalize_func = [&options](Graph& graph) {
            graph.CleanUnusedInitializers(options.initializer_names_to_preserve);
            graph.GraphResolveNeeded(false);

            // if we are resolving immediately after loading from a GraphProto, we don't need to
            // do a proto sync
            if (options.no_proto_sync_required) {
                graph.GraphProtoSyncNeeded(false);
            }

            return Status::OK(); };

  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, finalize_func));

  ++num_resolves_;

  return Status::OK();
}

void Graph::AddInitializedTensor(const TensorProto& tensor) {
  auto existing = name_to_initial_tensor_.find(tensor.name());
  if (existing != name_to_initial_tensor_.cend()) {
    ORT_ENFORCE(existing->second == &tensor,
                "AddInitializedTensor already has tensor with name ", tensor.name(), " but different TensorProto.");
    return;
  }

  const gsl::not_null<TensorProto*> tensor_added{graph_proto_->add_initializer()};
  *(tensor_added) = tensor;
  name_to_initial_tensor_[tensor.name()] = tensor_added;
  SetGraphResolveNeeded();
  if (!is_loaded_from_model_file_ && GetNodeArg(tensor.name()) == nullptr) {
    // make sure there is a NodeArg for the initializer as SetGraphInputsOutputs may add it to the graph inputs.
    // the shape will be set to the correct value in TypeCheckInputsAndInitializers as we don't yet know whether there
    // will be a matching graph input for this initializer (we prefer shape info from the graph input).
    TypeProto t;
    t.mutable_tensor_type()->set_elem_type(tensor.data_type());

    ORT_IGNORE_RETURN_VALUE(GetOrCreateNodeArg(tensor.name(), &t));
  }
}

void Graph::SetName(const std::string& name) {
  graph_proto_->set_name(name);
}

void Graph::SetDescription(const std::string& description) {
  graph_proto_->set_doc_string(description);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

const std::string& Graph::Name() const noexcept {
  return graph_proto_->name();
}

const std::string& Graph::Description() const noexcept {
  return graph_proto_->doc_string();
}

const Path& Graph::ModelPath() const {
  return owning_model_.ModelPath();
}

template <typename T, typename TIter>
static void RemoveRepeatedFieldEntry(T& repeated_field, const TIter& entry_to_remove) {
  auto num_entries = repeated_field.size();
  if (num_entries > 1) {
    // swap the entry being deleted with the last one, and delete it.
    // we do this so we don't have to move all the entries past the one being deleted down one.
    auto slot = entry_to_remove - repeated_field.begin();
    auto last_entry = repeated_field.end() - 1;
    repeated_field.SwapElements(slot, num_entries - 1);
    repeated_field.erase(last_entry);
  } else {
    repeated_field.erase(entry_to_remove);
  }
}

bool Graph::IsInitializedTensor(const std::string& name) const {
  return name_to_initial_tensor_.count(name) > 0;
}

bool Graph::IsSparseInitializer(const std::string& name) const {
  return sparse_tensor_names_.count(name) > 0;
}

void Graph::RemoveInitializedTensor(const std::string& tensor_name) {
  bool found = false;
  auto iter = name_to_initial_tensor_.find(tensor_name);
  found = iter != name_to_initial_tensor_.end();
  if (found) {
    name_to_initial_tensor_.erase(iter);
    sparse_tensor_names_.erase(tensor_name);
    SetGraphResolveNeeded();
  } else {
    ORT_ENFORCE(sparse_tensor_names_.count(tensor_name) == 0, "sparse_tensor_names_ not in sync with name_to_initial_tensor_");
  }

  auto& mutable_initializers = *(graph_proto_->mutable_initializer());
  auto proto_entry = std::find_if(mutable_initializers.begin(), mutable_initializers.end(),
                                  [&tensor_name](const TensorProto& entry) { return entry.name() == tensor_name; });

  if (proto_entry != mutable_initializers.end()) {
    RemoveRepeatedFieldEntry(mutable_initializers, proto_entry);
  } else {
    // these should always be in sync as the pointer in name_to_initial_tensor_ is to memory owned by graph_proto_
    ORT_ENFORCE(!found, "graph_proto_ is not in sync with name_to_initial_tensor_.");
  }
}

#if !defined(ORT_MINIMAL_BUILD)
Status Graph::ReplaceInitializedTensor(const ONNX_NAMESPACE::TensorProto& new_initializer) {
  // name_to_initial_tensor_ maps from name to const TensorProto*, so we first
  // look up the const pointer by name, then find and modify the mutable
  // pointed-to TensorProto in graph_proto_.
  const auto& initializer_name = new_initializer.name();
  const auto name_to_initializer_it = name_to_initial_tensor_.find(initializer_name);
  ORT_RETURN_IF_NOT(name_to_initializer_it != name_to_initial_tensor_.end(),
                    "Failed to find existing initializer with name ", initializer_name, ".");

  const auto& old_initializer = *(name_to_initializer_it->second);

  auto dims_eq = [&old_initializer, &new_initializer]() {
    if (old_initializer.dims_size() != new_initializer.dims_size()) return false;
    for (int i = 0; i < old_initializer.dims_size(); ++i) {
      if (old_initializer.dims(i) != new_initializer.dims(i)) return false;
    }
    return true;
  };

  ORT_RETURN_IF_NOT(dims_eq(), "Replacement tensor's dimensions do not match.");
  ORT_RETURN_IF_NOT(old_initializer.data_type() == new_initializer.data_type(),
                    "Replacement tensor's data type does not match.");

  auto& mutable_initializers = *(graph_proto_->mutable_initializer());
  // use cheaper pointer comparison to find old entry
  auto existing_entry = std::find(mutable_initializers.pointer_begin(), mutable_initializers.pointer_end(),
                                  &old_initializer);

  // these should always be in sync as the pointer in name_to_initial_tensor_ is to memory owned by graph_proto_
  ORT_ENFORCE(existing_entry != mutable_initializers.pointer_end(),
              "graph_proto_ is not in sync with name_to_initial_tensor_");

  **existing_entry = new_initializer;

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

bool Graph::GetInitializedTensor(const std::string& tensor_name, const TensorProto*& value) const {
  auto iter = name_to_initial_tensor_.find(tensor_name);
  if (name_to_initial_tensor_.end() == iter) {
    value = nullptr;
    return false;
  }
  value = iter->second;
  return true;
}

void Graph::CleanAllInitializedTensors() noexcept {
  name_to_initial_tensor_.clear();
  sparse_tensor_names_.clear();

  // Clearing RepeatedPtrFields does not free objects' memory. The memory is retained
  // and can be reused. Need to explicitly release the cleared objects and free the
  // memory.
  graph_proto_->mutable_initializer()->Clear();
  const int num_cleared = graph_proto_->initializer().ClearedCount();
  for (int i = 0; i < num_cleared; i++) {
    delete graph_proto_->mutable_initializer()->ReleaseCleared();
  }
}

const ONNX_NAMESPACE::TensorProto* Graph::GetConstantInitializer(const std::string& initializer_name,
                                                                 bool check_outer_scope) const {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  if (GetInitializedTensor(initializer_name, initializer)) {
    if (CanOverrideInitializer()) {
      const auto& graph_inputs = GetInputsIncludingInitializers();
      bool is_constant = std::none_of(graph_inputs.cbegin(), graph_inputs.cend(),
                                      [&initializer_name](const NodeArg* input) {
                                        return input->Name() == initializer_name;
                                      });

      if (!is_constant) {
        initializer = nullptr;
      }
    }
  } else if (check_outer_scope && IsSubgraph()) {
    // make sure there's not a local value with the same name. if there is it shadows any initializer in outer scope.
    if (IsOuterScopeValue(initializer_name)) {
      initializer = parent_graph_->GetConstantInitializer(initializer_name, check_outer_scope);
    }
  }

  return initializer;
}

#if !defined(ORT_MINIMAL_BUILD)
void Graph::AddValueInfo(const NodeArg* new_value_info) {
  for (const auto* info : value_info_) {
    ORT_ENFORCE(info->Name() != new_value_info->Name(), "Error: trying to add an existing value info.");
  }
  value_info_.push_back(new_value_info);
}

std::vector<NodeArg*> Graph::CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                            const ArgNameToTypeMap& name_to_type_map) {
  const auto name_to_type_map_end = name_to_type_map.end();
  std::vector<NodeArg*> results;
  results.reserve(names.size());

  for (auto& name : names) {
    const TypeProto* type = nullptr;

    auto name_to_type_iter = name_to_type_map.find(name);
    if (name_to_type_iter != name_to_type_map_end) {
      // This node input arg type/shape does exist in graph proto.
      // Assign type/shape information to node input arg.
      type = &(name_to_type_iter->second);
    }

    auto node_arg = &GetOrCreateNodeArg(name, type);
    results.push_back(node_arg);
  }

  return results;
}

Node& Graph::AddNode(const Node& other) {
  const auto& definitions = other.GetDefinitions();

  auto& new_node = AddNode(other.Name(), other.OpType(), other.Description(),
                           definitions.input_defs,
                           definitions.output_defs,
                           &other.GetAttributes(),
                           other.Domain());

  return new_node;
}

Node& Graph::AddNode(const NodeProto& node_proto,
                     const ArgNameToTypeMap& name_to_type_map) {
  auto input_defs = CreateNodeArgs(node_proto.input(), name_to_type_map);
  auto output_defs = CreateNodeArgs(node_proto.output(), name_to_type_map);

  const int num_attributes = node_proto.attribute_size();
  NodeAttributes attributes;
  attributes.reserve(num_attributes);

  for (int i = 0; i < num_attributes; ++i) {
    auto& attr = node_proto.attribute(i);
    attributes[attr.name()] = attr;
  }

  return AddNode(node_proto.name(),
                 node_proto.op_type(),
                 node_proto.doc_string(),
                 input_defs,
                 output_defs,
                 &attributes,
                 node_proto.domain());
}

std::string Graph::GenerateNodeArgName(const std::string& base_name) {
  std::string new_name = base_name;
  // Check if new_name has been used in as any of node_args_' names.
  // Check if new_name has been generated by this function.
  // If both are not, add new_name into name set and return the new_name
  // as the generated name. Otherwise, keep generating new names.
  while (node_args_.find(new_name) != node_args_.end() ||
         generated_node_arg_names_.find(new_name) != generated_node_arg_names_.end()) {
    std::ostringstream str;
    str << base_name << "_token_" << name_generator_++;
    new_name = str.str();
  }

  generated_node_arg_names_.insert(new_name);
  return new_name;
}

static flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
SaveInputsOutputsToOrtFormat(flatbuffers::FlatBufferBuilder& builder, const std::vector<const NodeArg*>& src) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> vec(src.size());
  std::transform(src.cbegin(), src.cend(), vec.begin(),
                 [&builder](const NodeArg* entry) {
                   return builder.CreateSharedString(entry->Name());
                 });
  return builder.CreateVector(vec);
}

common::Status Graph::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      flatbuffers::Offset<fbs::Graph>& fbs_graph) const {
  auto inputs = SaveInputsOutputsToOrtFormat(builder, graph_inputs_including_initializers_);
  auto outputs = SaveInputsOutputsToOrtFormat(builder, graph_outputs_);

  std::vector<flatbuffers::Offset<fbs::SparseTensor>> sparse_initializers_data;
  sparse_initializers_data.reserve(sparse_tensor_names_.size());
  const auto sparse_end = sparse_tensor_names_.end();

  std::vector<flatbuffers::Offset<fbs::Tensor>> initializers_data;
  assert(sparse_tensor_names_.size() <= name_to_initial_tensor_.size());
  initializers_data.reserve(name_to_initial_tensor_.size() - sparse_tensor_names_.size());
  const auto& model_path = ModelPath();

  for (const auto& pair : name_to_initial_tensor_) {
    if (sparse_tensor_names_.find(pair.first) == sparse_end) {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          experimental::utils::SaveInitializerOrtFormat(builder, *pair.second, model_path, fbs_tensor));
      initializers_data.push_back(fbs_tensor);
    } else {
      SparseTensorProto sparse_initializer;
      ORT_RETURN_IF_ERROR(utils::DenseTensorToSparseTensorProto(*pair.second, model_path, sparse_initializer));
      flatbuffers::Offset<fbs::SparseTensor> fbs_sparse_tensor;
      ORT_RETURN_IF_ERROR(
          experimental::utils::SaveSparseInitializerOrtFormat(builder, sparse_initializer, model_path, fbs_sparse_tensor));
      sparse_initializers_data.push_back(fbs_sparse_tensor);
    }
  }
  auto initializers = builder.CreateVector(initializers_data);
  auto sparse_initializers = builder.CreateVector(sparse_initializers_data);

  std::vector<flatbuffers::Offset<fbs::ValueInfo>> node_args_data;
  node_args_data.reserve(node_args_.size());
  for (const auto& pair : node_args_) {
    flatbuffers::Offset<fbs::ValueInfo> fbs_val_info;
    ORT_RETURN_IF_ERROR(
        experimental::utils::SaveValueInfoOrtFormat(builder, pair.second->ToProto(), fbs_val_info));
    node_args_data.push_back(fbs_val_info);
  }
  auto node_args = builder.CreateVector(node_args_data);

  std::vector<flatbuffers::Offset<fbs::Node>> nodes_vec;
  std::vector<flatbuffers::Offset<fbs::NodeEdge>> node_edges_vec;
  node_edges_vec.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    if (node != nullptr) {
      flatbuffers::Offset<fbs::Node> fbs_node;
      ORT_RETURN_IF_ERROR(node->SaveToOrtFormat(builder, fbs_node));
      nodes_vec.push_back(fbs_node);
      node_edges_vec.push_back(node->SaveEdgesToOrtFormat(builder));
    }
  }
  auto nodes = builder.CreateVector(nodes_vec);
  auto node_edges = builder.CreateVector(node_edges_vec);

  fbs::GraphBuilder gb(builder);
  gb.add_initializers(initializers);
  gb.add_node_args(node_args);
  gb.add_nodes(nodes);
  gb.add_max_node_index(gsl::narrow_cast<uint32_t>(nodes_.size()));
  gb.add_node_edges(node_edges);
  gb.add_inputs(inputs);
  gb.add_outputs(outputs);
  gb.add_sparse_initializers(sparse_initializers);
  fbs_graph = gb.Finish();
  return Status::OK();
}

std::string Graph::GenerateNodeName(const std::string& base_name) {
  // Define name-checking function for node name.
  // Return true if the input name hasn't been used. Otherwise, return false.
  auto name_is_ok = [&](const std::string name) {
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
      if (*it == nullptr) {
        continue;
      }
      if (it->get()->Name() != name) {
        continue;
      }
      // Find a matched name so we cannot reuse the input name.
      return false;
    }

    if (generated_node_names_.find(name) != generated_node_names_.end()) {
      // Find a matched name so we cannot reuse the input name.
      return false;
    }

    // The input name can be reused.
    return true;
  };

  // Start with the input name.
  std::string new_name = base_name;

  while (!name_is_ok(new_name)) {
    std::ostringstream str;
    str << base_name << "_token_" << name_generator_++;
    new_name = str.str();
  }

  // Make sure this new_name is not going to be reused.
  generated_node_names_.insert(new_name);

  return new_name;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Node& Graph::AddNode(const std::string& name,
                     const std::string& op_type,
                     const std::string& description,
                     const std::vector<NodeArg*>& input_args,
                     const std::vector<NodeArg*>& output_args,
                     const NodeAttributes* attributes,
                     const std::string& domain) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  inputs.resize(input_args.size());
  outputs.resize(output_args.size());
  int i = 0;
  for (auto input_arg : input_args) {
    inputs[i++] = &GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
  }
  i = 0;
  for (auto output_arg : output_args) {
    outputs[i++] = &GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
  }

  const gsl::not_null<Node*> node = AllocateNode();
  node->Init(name, op_type, description, inputs, outputs, attributes, domain);
  if (0 != op_type.compare(kNoOp)) {
    GraphProtoSyncNeeded(true);
  }

  return *node;
}

bool Graph::RemoveNode(NodeIndex p_index) {
  auto node = GetNode(p_index);
  if (nullptr == node) {
    return false;
  }

  // Node must be disconnected from any downstream nodes before removal
  ORT_ENFORCE(node->GetOutputEdgesCount() == 0, "Can't remove node ", node->Name(), " as it still has output edges.");

  // Remove all input edges.
  // Need to copy the edge info first so we can remove the real edges while iterating the copy of edge info.
  auto input_edges = node->GetRelationships().input_edges;

  for (auto& input_edge : input_edges) {
    RemoveEdge(input_edge.GetNode().Index(), p_index, input_edge.GetSrcArgIndex(), input_edge.GetDstArgIndex());
  }

  return ReleaseNode(p_index);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
bool Graph::AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index) {
  if (nodes_.size() <= src_node_index ||
      nodes_.size() <= dst_node_index ||
      nullptr == nodes_[src_node_index] ||
      nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    return false;
  }

  GSL_SUPPRESS(es .84) {  // ignoring return from insert()
    nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index]));
    nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index]));
    nodes_[dst_node_index]->MutableRelationships().control_inputs.insert(nodes_[src_node_index]->Name());
  }

  return true;
}

const ONNX_NAMESPACE::GraphProto& Graph::ToGraphProto() {
  if (!GraphProtoSyncNeeded()) {
    return *graph_proto_;
  }

  // Nodes.
  ToGraphProtoInternal(*graph_proto_);

  GraphProtoSyncNeeded(false);

  return *graph_proto_;
}

ONNX_NAMESPACE::GraphProto Graph::ToGraphProto() const {
  if (!GraphProtoSyncNeeded() && sparse_tensor_names_.empty()) {
    return *graph_proto_;
  }

  GraphProto result;
  ToGraphProtoInternal(result);
  // Path of the owning model
  // This is used for constructing full path for external data
  // if it exists
  const auto& model_path = ModelPath();

  // We want to make sure that sparse initializers do not appear
  // as dense duplicates within the initializers list.
  if (!sparse_tensor_names_.empty()) {
    const auto sparse_end = sparse_tensor_names_.end();
    auto* mutable_initializer = result.mutable_initializer();
    for (const auto& initializer : graph_proto_->initializer()) {
      if (sparse_end == sparse_tensor_names_.find(initializer.name())) {
        *mutable_initializer->Add() = initializer;
      } else {
        auto& sparse_initializer = *result.add_sparse_initializer();
        auto status = utils::DenseTensorToSparseTensorProto(initializer, model_path, sparse_initializer);
        ORT_ENFORCE(status.IsOK(), "Failed to convert dense initializer to sparse");
      }
    }
  } else {
    *result.mutable_initializer() = graph_proto_->initializer();
  }

  return result;
}

ONNX_NAMESPACE::GraphProto Graph::ToGraphProtoWithExternalInitializers(const std::string& external_file_name,
                                                                       size_t initializer_size_threshold) const {
  GraphProto result;
  ToGraphProtoInternal(result);
  const auto& model_path = ModelPath();

  std::ofstream external_stream(external_file_name, std::ofstream::out | std::ofstream::binary);
  ORT_ENFORCE(external_stream.is_open());
  int64_t external_offset = 0;

  // Add the initializers to the result graph.
  const auto sparse_end = sparse_tensor_names_.end();
  for (const auto& initializer : graph_proto_->initializer()) {
    if (sparse_end != sparse_tensor_names_.find(initializer.name())) {
      // Sparse tensors are added to the ONNX file.
      auto& sparse_initializer = *result.add_sparse_initializer();
      auto status = utils::DenseTensorToSparseTensorProto(initializer, model_path, sparse_initializer);
      ORT_ENFORCE(status.IsOK(), "Failed to convert dense initializer to sparse");
    } else {
      // Dense tensors larger than the threshold are added to the external file.
      TensorProto* output_proto = result.add_initializer();

      size_t tensor_bytes_size = 0;
      std::unique_ptr<uint8_t[]> raw_data;
      ORT_THROW_IF_ERROR(utils::UnpackInitializerData(initializer, Path(), raw_data, tensor_bytes_size));

      if (tensor_bytes_size < initializer_size_threshold) {
        *output_proto = initializer;
        continue;
      }

      for (size_t index = 0; index != tensor_bytes_size; ++index) {
        external_stream << raw_data[index];
      }

      output_proto->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
      ONNX_NAMESPACE::StringStringEntryProto* location = output_proto->add_external_data();
      location->set_key("location");
      location->set_value(external_file_name);
      ONNX_NAMESPACE::StringStringEntryProto* offset = output_proto->add_external_data();
      offset->set_key("offset");
      offset->set_value(std::to_string(external_offset));
      ONNX_NAMESPACE::StringStringEntryProto* length = output_proto->add_external_data();
      length->set_key("length");
      length->set_value(std::to_string(tensor_bytes_size));

      output_proto->set_name(initializer.name());
      output_proto->set_data_type(initializer.data_type());
      for (int i = 0; i != initializer.dims_size(); ++i) {
        output_proto->add_dims(initializer.dims(i));
      }
      output_proto->set_doc_string(initializer.doc_string());

      external_offset += tensor_bytes_size;
    }
  }

  return result;
}

void Graph::ToGraphProtoInternal(ONNX_NAMESPACE::GraphProto& graph_proto) const {
  graph_proto_->clear_node();
  graph_proto_->clear_input();
  graph_proto_->clear_output();
  graph_proto_->clear_value_info();
  graph_proto.set_name(Name());
  graph_proto.set_doc_string(Description());

  for (const auto* input_arg : GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : value_info_) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  // add the NodeArg info for outer scope NodeArgs so we capture the type information
  for (const auto& name : outer_scope_node_arg_names_) {
    auto* node_arg = GetNodeArg(name);
    ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
  }

  GraphViewer graph_viewer(*this);
  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    const gsl::not_null<NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }
}

void Graph::CleanUnusedInitializers(const std::unordered_set<std::string>* initializer_names_to_preserve) {
  std::unordered_set<std::string> used_args;

  // anything that provides a required graph input (GetInputs), an optional graph input (GetOverridableInitializers)
  // or a graph output (GetOutputs) cannot be removed
  const auto& inputs = GetInputs();
  const auto& overridable_initializers = GetOverridableInitializers();
  const auto& outputs = GetOutputs();

  std::for_each(inputs.cbegin(), inputs.cend(), [&used_args](const NodeArg* input) {
    ORT_IGNORE_RETURN_VALUE(used_args.insert(input->Name()));
  });

  std::for_each(overridable_initializers.cbegin(), overridable_initializers.cend(),
                [&used_args](const NodeArg* input) {
                  ORT_IGNORE_RETURN_VALUE(used_args.insert(input->Name()));
                });

  std::for_each(outputs.cbegin(), outputs.cend(), [&used_args](const NodeArg* output) {
    ORT_IGNORE_RETURN_VALUE(used_args.insert(output->Name()));
  });

  for (const auto& node : Nodes()) {
    for (const auto* def : node.InputDefs()) {
      ORT_IGNORE_RETURN_VALUE(used_args.insert(def->Name()));
    }

    for (const auto* def : node.ImplicitInputDefs()) {
      ORT_IGNORE_RETURN_VALUE(used_args.insert(def->Name()));
    }
  }

  std::vector<std::string> erase_list;
  auto end = used_args.end();
  for (const auto& pv : name_to_initial_tensor_) {
    const std::string& name = pv.first;
    if (used_args.find(name) == end &&
        (initializer_names_to_preserve == nullptr ||
         initializer_names_to_preserve->find(name) == initializer_names_to_preserve->cend())) {
      // on the first call to Graph::Resolve we are removing unnecessary initializers that should be removed
      // from the model.
      // on later calls we are removing initializers that optimizations have made redundant.
      if (num_resolves_ == 0) {
        LOGS(logger_, WARNING) << "Removing initializer '"
                               << name << "'. It is not used by any node and should be removed from the model.";
      } else {
        LOGS(logger_, INFO) << "Removing initializer '" << name << "'. It is no longer used by any node.";
      }

      erase_list.push_back(name);
    }
  }

  std::for_each(erase_list.cbegin(), erase_list.cend(),
                [this](const std::string& name) {
                  RemoveInitializedTensor(name);

                  // handle edge case where the unused initializer has a matching graph input.
                  // this can only happen when initializers cannot be overridden via an optional graph input.
                  // (otherwise this initializer wouldn't be allowed to be removed due to it backing an optional
                  // graph input).
                  if (CanOverrideInitializer() == false) {
                    auto& proto_inputs = *graph_proto_->mutable_input();
                    auto i = std::find_if(proto_inputs.begin(), proto_inputs.end(),
                                          [&name](const ONNX_NAMESPACE::ValueInfoProto& input) {
                                            return input.name() == name;
                                          });

                    if (i != proto_inputs.end()) {
                      RemoveRepeatedFieldEntry(proto_inputs, i);
                    }

                    auto& inputs_including_initializers = graph_inputs_including_initializers_;
                    auto j = std::find_if(inputs_including_initializers.begin(), inputs_including_initializers.end(),
                                          [&name](const NodeArg* input) { return input->Name() == name; });

                    if (j != inputs_including_initializers.end()) {
                      inputs_including_initializers.erase(j);
                    }
                  }
                });
}

#endif  // !defined(ORT_MINIMAL_BUILD)

void Graph::ComputeOverridableInitializers() {
  graph_overridable_initializers_.clear();
  if (CanOverrideInitializer()) {
    // graph_inputs_excluding_initializers_ and graph_inputs_including_initializers_
    // are inserted in the same order. So we walk and compute the difference.
    auto f_incl = graph_inputs_including_initializers_.cbegin();
    const auto l_incl = graph_inputs_including_initializers_.cend();
    auto f_excl = graph_inputs_excluding_initializers_.cbegin();
    const auto l_excl = graph_inputs_excluding_initializers_.cend();

    while (f_incl != l_incl) {
      // Equal means not an initializer
      if (f_excl != l_excl && *f_incl == *f_excl) {
        ++f_incl;
        ++f_excl;
        continue;
      }
      graph_overridable_initializers_.push_back(*f_incl);
      ++f_incl;
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)

GSL_SUPPRESS(es .84)  // warning about ignoring return value from insert(...)
Status Graph::SetGraphInputsOutputs() {
  // If loaded from a model file, we start from the specified inputs and
  // outputs set earlier by InitializeStateFromModelFileGraphProto().
  // Otherwise (!is_loaded_from_model_file_), we need to fix up the inputs and
  // may also need to infer inputs and outputs.
  // In either case, calls to SetInputs() or SetOutputs() may affect the actual
  // inputs and outputs.
  if (is_loaded_from_model_file_) return Status::OK();

  // Reset value_info.
  value_info_.clear();

  std::unordered_map<std::string, size_t> output_name_to_node_arg_index;
  std::vector<const NodeArg*> output_node_args_in_order;

  // if something is coming from outer scope, consider it already added
  std::unordered_set<std::string> added_input_names{outer_scope_node_arg_names_};
  graph_inputs_excluding_initializers_.clear();
  if (!graph_inputs_manually_set_) {
    graph_inputs_including_initializers_.clear();
  } else {
    // If we've set graph_inputs_including_initializers_ by calling SetInputs,
    // we copy its non-duplicate elements to graph_inputs_excluding_initializers_.
    // Later, we will erase initializers from graph_inputs_excluding_initializers_
    // if graph_inputs_manually_set_ is true.
    // In this way, we can ensure graph_inputs_excluding_initializers_ is the full
    // set of inputs less initializers, which could be a graph input used only
    // by a subgraph and thereby only an implicit input to a node, or a graph input
    // not used anywhere.
    // We also make sure graph_inputs_excluding_initializers_ list doesn't have any
    // duplicate names.
    std::unordered_set<std::string> existing_names;
    for (auto arg : graph_inputs_including_initializers_) {
      const std::string& name = arg->Name();
      if (existing_names.count(name) == 0) {
        graph_inputs_excluding_initializers_.push_back(arg);
        existing_names.insert(name);
      }
    }
  }

  if (!graph_outputs_manually_set_) {
    graph_outputs_.clear();
  }

  // Collect all nodes' outputs
  for (const auto& node : Nodes()) {
    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        output_node_args_in_order.push_back(output_def);
        output_name_to_node_arg_index.insert({output_def->Name(), output_node_args_in_order.size() - 1});
      }
    }
  }

  // Init graph output args with copy of all node output args.
  auto graph_output_args = output_name_to_node_arg_index;
  for (const auto& node : Nodes()) {
    // Go thru all node's inputs.
    for (const auto* input_arg : node.InputDefs()) {
      if (!input_arg->Exists()) {
        // It's an optional input and does not exist in this case.
        continue;
      }

      auto output_arg_iter = output_name_to_node_arg_index.find(input_arg->Name());
      if (output_name_to_node_arg_index.end() == output_arg_iter) {
        // This input arg is not the output of another node so must come from either a graph input or an initializer.
        const std::string& name = input_arg->Name();

        if (added_input_names.end() == added_input_names.find(name)) {
          // This graph input has not been added into <graph_inputs_>.
          bool is_initializer = name_to_initial_tensor_.find(name) != name_to_initial_tensor_.end();

          if (!graph_inputs_manually_set_) {
            // if IR version < 4 all initializers must have a matching graph input
            // (even though the graph input is not allowed to override the initializer).
            // if IR version >= 4 initializers are not required to have a matching graph input.
            // any graph inputs that are to override initializers must be specified by calling SetInputs.
            if (!is_initializer || ir_version_ < 4) {
              graph_inputs_including_initializers_.push_back(input_arg);
            }
            if (!is_initializer) {
              // If input_arg is not of an initializer, we add it into graph_inputs_excluding_initializers_.
              graph_inputs_excluding_initializers_.push_back(input_arg);
            }
          } else {
            // graph_inputs_including_initializers_ has been manually populated by SetInputs.
            // Validation: the <input_arg> must be in graph inputs or initializers when it's manually set.
            if (!is_initializer) {
              const auto& inputs = graph_inputs_including_initializers_;
              bool in_inputs = std::find(inputs.begin(), inputs.end(), input_arg) != inputs.end();
              if (!in_inputs) {
                return Status(ONNXRUNTIME, FAIL,
                              name + " must be either specified in graph inputs or graph initializers.");
              }
            } else {
              // If arg_input is of an initializer, we remove it from graph_inputs_excluding_initializers_
              // whose initial content has both initializers and non-initializers.
              auto input_pos = std::find(graph_inputs_excluding_initializers_.begin(),
                                         graph_inputs_excluding_initializers_.end(),
                                         input_arg);
              if (input_pos != graph_inputs_excluding_initializers_.end()) {
                graph_inputs_excluding_initializers_.erase(input_pos);
              }
            }
          }

          added_input_names.insert(name);
        }
      } else if (graph_output_args.erase(output_arg_iter->first) >= 1) {
        // Remove the output arg name from graph outputs since it's
        // the input of this node, which we call it intermediate result
        // and store it in <m_valueinfo>.
        if (std::find(value_info_.begin(), value_info_.end(), input_arg) == value_info_.end()) {
          value_info_.push_back(input_arg);
        }
      }
    }
  }

  if (!graph_outputs_manually_set_) {
    // Set graph outputs in order.
    std::vector<size_t> graph_output_args_index;
    graph_output_args_index.reserve(graph_output_args.size());
    for (const auto& output_arg : graph_output_args) {
      graph_output_args_index.push_back(output_arg.second);
    }

    std::sort(graph_output_args_index.begin(), graph_output_args_index.end());
    for (auto& output_arg_index : graph_output_args_index) {
      graph_outputs_.push_back(output_node_args_in_order[output_arg_index]);
    }
  }

  ComputeOverridableInitializers();

  return Status::OK();
}

IOnnxRuntimeOpSchemaCollectionPtr Graph::GetSchemaRegistry() const {
  return schema_registry_;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// calling private ctor
GSL_SUPPRESS(r .11)
gsl::not_null<Node*> Graph::AllocateNode() {
  ORT_ENFORCE(nodes_.size() < static_cast<unsigned int>(std::numeric_limits<int>::max()));
  std::unique_ptr<Node> new_node(new Node(nodes_.size(), *this));
  Node* node{new_node.get()};

  nodes_.push_back(std::move(new_node));
  ++num_of_nodes_;
  GraphResolveNeeded(true);

  return gsl::not_null<Node*>{node};
}

// TODO: Does this need (and maybe AllocateNode) to be threadsafe so nodes_ and num_of_nodes_ managed more carefully?
bool Graph::ReleaseNode(NodeIndex index) {
  if (index >= nodes_.size()) {
    return false;
  }

  // index is valid, but the entry may already be empty
  if (nodes_[index] != nullptr) {
    nodes_[index] = nullptr;
    --num_of_nodes_;
    GraphProtoSyncNeeded(true);
    GraphResolveNeeded(true);
  }

  return true;
}

Node& Graph::CreateFusedSubGraphNode(const IndexedSubGraph& sub_graph, const std::string& fused_node_name) {
  const auto* func_meta_def = sub_graph.GetMetaDef();
  ORT_ENFORCE(nullptr != func_meta_def);
  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;
  std::unordered_map<std::string, int> input_indexes;
  std::unordered_map<std::string, int> output_indexes;

  int cur_idx = 0;
  for (auto& arg_name : func_meta_def->inputs) {
    input_args.push_back(GetNodeArg(arg_name));
    input_indexes[arg_name] = cur_idx++;
  }

  cur_idx = 0;
  for (auto& arg_name : func_meta_def->outputs) {
    output_args.push_back(GetNodeArg(arg_name));
    output_indexes[arg_name] = cur_idx++;
  }

  auto& fused_node = AddNode(fused_node_name,
                             func_meta_def->name,
                             func_meta_def->doc_string,
                             input_args,
                             output_args,
                             &func_meta_def->attributes,
                             func_meta_def->domain);

  fused_node.SetNodeType(Node::Type::Fused);

  return fused_node;
}

Node& Graph::BeginFuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name) {
  Node& node = CreateFusedSubGraphNode(sub_graph, fused_node_name);

#if !defined(ORT_MINIMAL_BUILD)
  // if this is a full build create the lightweight Function implementation that provides the schema so that
  // kernel lookup works as per usual. in an extended minimal build we do the lookup via a hash so don't
  // need to create the schema.
  auto func = std::make_unique<ViewerFunctionImpl>(*this, sub_graph, logger_);
  function_container_.push_back(std::move(func));
  node.SetFunctionBody(*function_container_.back());
#endif

  return node;
}

void Graph::CancelFuseSubGraph(const Node& fused_node) {
  auto node_idx = fused_node.Index();
  if (!GetNode(node_idx))
    return;

  if (fused_node.NodeType() != Node::Type::Fused)
    return;

#if !defined(ORT_MINIMAL_BUILD)
  // Remove the function body from function container
  const auto* fused_node_func = fused_node.GetFunctionBody();
  auto it = std::find_if(
      function_container_.begin(), function_container_.end(),
      [fused_node_func](const std::unique_ptr<onnxruntime::Function>& func) {
        return func.get() == fused_node_func;
      });
  if (it != function_container_.end()) {
    function_container_.erase(it);
  }
#endif

  // Remove the fused_node
  RemoveNode(node_idx);
}

void Graph::FinalizeFuseSubGraph(const IndexedSubGraph& sub_graph, Node& fused_node) {
  const auto* func_meta_def = sub_graph.GetMetaDef();
  ORT_ENFORCE(nullptr != func_meta_def);

  std::unordered_map<std::string, int> input_indexes;
  std::unordered_map<std::string, int> output_indexes;

  int cur_idx = 0;
  for (auto& arg_name : func_meta_def->inputs) {
    input_indexes[arg_name] = cur_idx++;
  }

  cur_idx = 0;
  for (auto& arg_name : func_meta_def->outputs) {
    output_indexes[arg_name] = cur_idx++;
  }

  auto new_node_idx = fused_node.Index();

  // Remove nodes that were fused
  for (auto node_index : sub_graph.nodes) {
    auto node = GetNode(node_index);
    if (nullptr == node) {
      continue;
    }

    // move any applicable input edges to the new node. remove all others
    auto input_edges = node->GetRelationships().input_edges;  // copy so RemoveEdge doesn't invalidate iterator
    for (const auto& input_edge : input_edges) {
      const auto& producer = input_edge.GetNode();
      auto producer_idx = producer.Index();
      auto src_idx = input_edge.GetSrcArgIndex();
      auto dst_idx = input_edge.GetDstArgIndex();

      // if this input is an input of the fused node add an edge for that
      if (dst_idx < (int)node->InputDefs().size()) {
        auto it = input_indexes.find(node->InputDefs()[dst_idx]->Name());
        if (it != input_indexes.cend()) {
          AddEdge(producer_idx, new_node_idx, src_idx, it->second);
        }
      } else {
        int dst_implicit_input_idx = dst_idx - (int)node->InputDefs().size();
        ORT_ENFORCE(dst_implicit_input_idx < (int)node->ImplicitInputDefs().size());
        auto it = input_indexes.find(node->ImplicitInputDefs()[dst_implicit_input_idx]->Name());
        if (it != input_indexes.cend()) {
          AddEdge(producer_idx, new_node_idx, src_idx, it->second);
        }
      }
      RemoveEdge(producer_idx, node_index, src_idx, dst_idx);
    }

    // move any applicable output edges to the new node
    auto output_edges = node->GetRelationships().output_edges;  // copy so RemoveEdge doesn't invalidate iterator
    for (const auto& output_edge : output_edges) {
      const auto& consumer = output_edge.GetNode();
      auto consumer_idx = consumer.Index();
      auto src_idx = output_edge.GetSrcArgIndex();
      auto dst_idx = output_edge.GetDstArgIndex();

      // if this output is an output of the fused node add an edge for that
      auto it = output_indexes.find(node->OutputDefs()[src_idx]->Name());
      if (it != output_indexes.cend()) {
        AddEdge(new_node_idx, consumer_idx, it->second, dst_idx);
      }

      RemoveEdge(node_index, consumer_idx, src_idx, dst_idx);
    }

    RemoveNode(node_index);
  }
}

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
Node& Graph::FuseSubGraph(const IndexedSubGraph& sub_graph,
                          const std::string& fused_node_name) {
  Node& fused_node = CreateFusedSubGraphNode(sub_graph, fused_node_name);

  // create Function before we remove nodes
  function_container_.emplace_back(MakeFunction(*this, sub_graph, logger_));
  fused_node.SetFunctionBody(*function_container_.back());

  // remove nodes and update edges
  FinalizeFuseSubGraph(sub_graph, fused_node);

  return fused_node;
}

Status Graph::InlineFunction(Node& node) {
  // Remove the function node, add the nodes in function's subgraph into the
  // main graph.
  const Graph& subgraph = node.GetFunctionBody()->Body();
  auto output_edges = node.GetRelationships().output_edges;
  for (auto output_edge : output_edges) {
    RemoveEdge(node.Index(), output_edge.GetNode().Index(), output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());
  }
  std::unordered_map<std::string, NodeArg*> remap_input_output;
  if (node.MutableInputDefs().size() != subgraph.GetInputsIncludingInitializers().size())
    return Status(ONNXRUNTIME, FAIL, "Node " + node.Name() + "'s number of inputs is different from function body graph's number of input.");
  for (size_t i = 0; i < subgraph.GetInputsIncludingInitializers().size(); ++i) {
    auto* input = subgraph.GetInputsIncludingInitializers()[i];
    if (input->Name() != node.MutableInputDefs()[i]->Name())
      remap_input_output[input->Name()] = node.MutableInputDefs()[i];
  }

  ORT_ENFORCE(node.MutableOutputDefs().size() == subgraph.GetOutputs().size(),
              "Node ", node.Name(), "'s number of outputs is different from function body graph's number of outputs.");
  for (size_t i = 0; i < subgraph.GetOutputs().size(); ++i) {
    auto* output = subgraph.GetOutputs()[i];
    if (output->Name() != node.MutableOutputDefs()[i]->Name())
      remap_input_output[output->Name()] = node.MutableOutputDefs()[i];
  }

  RemoveNode(node.Index());
  const auto& model_path = ModelPath();
  for (const auto& subgraph_node : subgraph.Nodes()) {
    if (subgraph_node.OpType() == kConstant) {
      // Copy constant nodes _value to name_to_initial_tensor_
      ONNX_NAMESPACE::NodeProto subgraph_node_proto{};
      subgraph_node.ToProto(subgraph_node_proto);
      const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
      ORT_RETURN_IF_ERROR(utils::ConstantNodeProtoToTensorProto(subgraph_node_proto, model_path, *tensor));
      name_to_initial_tensor_[tensor->name()] = tensor;
    } else {
      std::vector<NodeArg*> inputs, outputs;
      for (auto* input : subgraph_node.InputDefs()) {
        auto it = remap_input_output.find(input->Name());
        if (it != remap_input_output.end())
          inputs.push_back(it->second);
        else
          inputs.push_back(const_cast<NodeArg*>(input));
      }
      for (auto* output : subgraph_node.OutputDefs()) {
        auto it = remap_input_output.find(output->Name());
        if (it != remap_input_output.end())
          outputs.push_back(it->second);
        else
          outputs.push_back(const_cast<NodeArg*>(output));
      }
      AddNode(subgraph_node.Name(), subgraph_node.OpType(), subgraph_node.Description(),
              inputs,
              outputs,
              &subgraph_node.GetAttributes(),
              subgraph_node.Domain());
    }
  }
  ORT_RETURN_IF_ERROR(this->Resolve());
  return Status::OK();
}

void Graph::SetInputs(const std::vector<const NodeArg*>& inputs) {
  if (is_loaded_from_model_file_) {
    // graph loaded from model file
    graph_inputs_including_initializers_ = inputs;
    graph_inputs_excluding_initializers_.clear();
    for (const auto* input : inputs) {
      ORT_ENFORCE(input->Exists(), "Input to set must exist.");
      if (name_to_initial_tensor_.find(input->Name()) == name_to_initial_tensor_.end()) {
        graph_inputs_excluding_initializers_.emplace_back(input);
      }
    }

    ComputeOverridableInitializers();
  } else {
    // creating graph from scratch
    // rely on SetGraphInputsOutputs() to fix up graph_inputs_excluding_initializers_
    graph_inputs_including_initializers_ = inputs;
  }

  graph_inputs_manually_set_ = true;
  GraphProtoSyncNeeded(true);
  GraphResolveNeeded(true);
}

void Graph::SetOutputs(const std::vector<const NodeArg*>& outputs) {
  graph_outputs_ = outputs;

  graph_outputs_manually_set_ = true;
  GraphProtoSyncNeeded(true);
  GraphResolveNeeded(true);
}

void Graph::SetNodeArgType(NodeArg& arg, const onnx::TypeProto& type_proto) {
  arg.SetType(type_proto);
  GraphResolveNeeded(true);
}

#endif  //  !defined(ORT_MINIMAL_BUILD)

Graph::~Graph() {
  // nothing to do, but we put it here so we don't need to fully define types in Graph that are held in unique_ptr
  // such as   std::unique_ptr<FunctionContainer> function_container_;
}

#if !defined(ORT_MINIMAL_BUILD)
std::ostream& operator<<(std::ostream& out, const Graph& graph) {
  out << "Inputs:\n";
  for (auto* x : graph.GetInputs()) {
    out << "   " << x->Name() << " : " << *x->Type() << "\n";
  }
  out << "Nodes:\n";
  for (auto& node : graph.Nodes()) {
    out << "   " << node.Name() << ": " << node.OpType() << " (";
    for (auto* x : node.InputDefs()) {
      if (x->Exists()) {
        out << x->Name() << ": " << *x->Type();
      }
      out << ", ";
    }
    out << ") -> ";
    for (auto* x : node.OutputDefs()) {
      if (x->Exists()) {
        out << x->Name() << ": " << *x->Type();
      }
      out << ", ";
    }
    out << "\n";
  }
  out << "Outputs:\n";
  for (auto* x : graph.GetOutputs()) {
    out << "   " << x->Name() << " : " << *x->Type() << "\n";
  }
  return out;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if defined(ENABLE_ORT_FORMAT_LOAD)
Status Graph::LoadFromOrtFormat(const onnxruntime::experimental::fbs::Graph& fbs_graph,
                                const Model& owning_model,
                                const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
                                IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
                                const logging::Logger& logger, std::unique_ptr<Graph>& graph) {
  // can't use make_unique as we're calling a private ctor
  graph.reset(new Graph(owning_model, domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
                        schema_registry,
#endif
                        nullptr, nullptr, logger));

  ORT_RETURN_IF_ERROR(graph->LoadFromOrtFormat(fbs_graph));

#if !defined(ORT_MINIMAL_BUILD)
  // in a full build we need to run Resolve to fully populate ResolveContext and Node::op_,
  // which will allow optimizers to run or non-ORT EPs to take nodes.
  // TODO: We could decide that an ORT model is load only even in a full build,
  // and in InferenceSession::Initialize skip partitioning and running optimizers.
  graph->SetGraphResolveNeeded();
  ORT_RETURN_IF_ERROR(graph->Resolve());
#endif

  return Status::OK();
}

Status Graph::LoadFromOrtFormat(const onnxruntime::experimental::fbs::Graph& fbs_graph,
                                Graph& parent_graph, const Node& parent_node,
                                const logging::Logger& logger, std::unique_ptr<Graph>& graph) {
  // can't use make_unique as we're calling a private ctor
  graph.reset(new Graph(parent_graph.owning_model_,
                        parent_graph.domain_to_version_,
#if !defined(ORT_MINIMAL_BUILD)
                        parent_graph.schema_registry_,
#endif
                        &parent_graph, &parent_node,
                        logger));

  return graph->LoadFromOrtFormat(fbs_graph);
}

Graph::Graph(const Model& owning_model,
             const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
             Graph* parent_graph, const Node* parent_node,
             const logging::Logger& logger)
    : owning_model_(owning_model),
      graph_proto_(&deserialized_proto_data_),
#if !defined(ORT_MINIMAL_BUILD)
      schema_registry_(schema_registry),
#endif
      domain_to_version_(domain_to_version),
      ir_version_(owning_model.IrVersion()),
      parent_graph_(parent_graph),
      parent_node_(parent_node),
      logger_(logger),
      is_loaded_from_model_file_(true) {  // true as the Graph isn't manually constructed from scratch
}

common::Status Graph::LoadFromOrtFormat(const onnxruntime::experimental::fbs::Graph& fbs_graph) {
  // We deserialize the graph from ORT format in the following order:
  // 1. Deserialize the initializers and sparse initializers. Convert sparse to dense.
  // 2. Deserialize the NodeArgs
  //        We need all NodeArg instances to exist when deserializing Nodes to setup the Node's
  //        inputs/outputs/implicit inputs which are collections of NodeArg*.
  // 3. Deserialize the Nodes
  // 4. Deserialize the NodeEdges
  //        We need all the Node instances to exist as the EdgeEnd has a Node* for the other end of the edge
  // 5. Deserialize the Inputs/Outputs/outer_scope_node_args

  // Initializers
  auto fbs_initializers = fbs_graph.initializers();
  auto fbs_sparse_initializers = fbs_graph.sparse_initializers();
  flatbuffers::uoffset_t map_size = (fbs_initializers != nullptr ? fbs_initializers->size() : 0U) +
                                    (fbs_sparse_initializers != nullptr ? fbs_sparse_initializers->size() : 0U);

  if (map_size > 0) {
    name_to_initial_tensor_.reserve(map_size);
  }

  if (fbs_initializers) {
    for (const auto* fbs_tensor : *fbs_initializers) {
      ORT_RETURN_IF(nullptr == fbs_tensor, "Initializer tensor is missing. Invalid ORT format model.");
      TensorProto* initializer = deserialized_proto_data_.add_initializer();
      ORT_RETURN_IF_ERROR(experimental::utils::LoadInitializerOrtFormat(*fbs_tensor, *initializer));
      auto p = name_to_initial_tensor_.emplace(initializer->name(), initializer);
      if (!p.second) {
        LOGS(logger_, WARNING) << "Duplicate initializer (dense or ConstantNode): '" << initializer->name()
                               << "' the model will use the latest encountered initializer"
                               << ". Please, fix your model.";
        p.first->second = initializer;
      }
    }
  }

  if (fbs_sparse_initializers) {
    sparse_tensor_names_.reserve(fbs_sparse_initializers->size());
    const auto& model_path = ModelPath();

    for (const auto* fbs_sparse_tensor : *fbs_sparse_initializers) {
      ORT_RETURN_IF(nullptr == fbs_sparse_tensor, "Sparse Initializer tensor is missing. Invalid ORT format model.");
      SparseTensorProto sparse_initializer;
      ORT_RETURN_IF_ERROR(experimental::utils::LoadSparseInitializerOrtFormat(*fbs_sparse_tensor, sparse_initializer));
      TensorProto& initializer = *deserialized_proto_data_.add_initializer();
      ORT_RETURN_IF_ERROR(utils::SparseTensorProtoToDenseTensorProto(sparse_initializer, model_path, initializer));
      auto p = name_to_initial_tensor_.emplace(initializer.name(), &initializer);
      if (!p.second) {
        LOGS(logger_, WARNING) << "Duplicate initializer (dense, sparse or ConstantNode): '" << initializer.name()
                               << "' the model will use the latest encountered initializer"
                               << ". Please, fix your model.";
        p.first->second = &initializer;
      }
      sparse_tensor_names_.emplace(initializer.name());
    }
  }

  // NodeArgs
  auto fbs_node_args = fbs_graph.node_args();
  if (fbs_node_args) {
    node_args_.reserve(fbs_node_args->size());
    for (const auto* fbs_value_info : *fbs_node_args) {
      ORT_RETURN_IF(nullptr == fbs_value_info, "NodeArg is missing. Invalid ORT format model.");
      NodeArgInfo node_arg_info;
      ORT_RETURN_IF_ERROR(experimental::utils::LoadValueInfoOrtFormat(*fbs_value_info, node_arg_info));
      // NodeArg ctor is private, cannot use make_unique
      node_args_[fbs_value_info->name()->str()] = std::unique_ptr<NodeArg>(new NodeArg(std::move(node_arg_info)));
    }
  }

  // Nodes
  //
  // Since we access a node using its index, we need to have nodes_ with size max_node_index to avoid
  // out of bounds access.
  nodes_.resize(fbs_graph.max_node_index());
  auto* fbs_nodes = fbs_graph.nodes();

  // It is possible to have no nodes in the model. Most likely scenario is the subgraph of an If Node
  // where the subgraph returns a Constant node. The Constant node will be lifted to an initializer by ORT
  // (prior to serializing to ORT format), leaving a valid Graph that contains no nodes.
  if (fbs_nodes != nullptr) {
    for (const auto* fbs_node : *fbs_nodes) {
      ORT_RETURN_IF(nullptr == fbs_node, "Node is missing. Invalid ORT format model.");
      std::unique_ptr<Node> node;
      ORT_RETURN_IF_ERROR(Node::LoadFromOrtFormat(*fbs_node, *this, logger_, node));
      ORT_RETURN_IF(node->Index() >= fbs_graph.max_node_index(), "Node index is out of range");
      nodes_[node->Index()] = std::move(node);
      ++num_of_nodes_;
    }
  }

  // NodeEdges
  auto* fbs_node_edges = fbs_graph.node_edges();
  if (fbs_node_edges != nullptr) {
    for (const auto* fbs_node_edge : *fbs_node_edges) {
      ORT_RETURN_IF(nullptr == fbs_node_edge, "NodeEdge is missing. Invalid ORT format model.");
      ORT_RETURN_IF(fbs_node_edge->node_index() >= fbs_graph.max_node_index(), "Node index is out of range");
      ORT_RETURN_IF_ERROR(nodes_[fbs_node_edge->node_index()]->LoadEdgesFromOrtFormat(*fbs_node_edge, *this));
    }
  }

  // Inputs/Outputs/outer_scope_node_args
  auto add_node_args = [&](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* fbs_node_args,
                           std::vector<const NodeArg*>& node_args) -> Status {
    if (fbs_node_args != nullptr) {
      node_args.reserve(fbs_node_args->size());
      for (const auto* fbs_node_arg_name : *fbs_node_args) {
        ORT_RETURN_IF(nullptr == fbs_node_arg_name, "NodeArg Name is missing. Invalid ORT format model.");
        gsl::not_null<NodeArg*> node_arg = GetNodeArg(fbs_node_arg_name->str());
        node_args.push_back(node_arg);
      }
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(add_node_args(fbs_graph.inputs(), graph_inputs_including_initializers_));
  for (const auto* input_arg : graph_inputs_including_initializers_) {
    if (name_to_initial_tensor_.count(input_arg->Name()) == 0) {
      graph_inputs_excluding_initializers_.push_back(input_arg);
    }
  }

  ComputeOverridableInitializers();

  ORT_RETURN_IF_ERROR(add_node_args(fbs_graph.outputs(), graph_outputs_));

  return Status::OK();
}

#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

}  // namespace onnxruntime
