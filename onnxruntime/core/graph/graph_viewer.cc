// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

#if !defined(ORT_MINIMAL_BUILD)
struct PriorityNodeCompare {
  inline bool IsHighPri(const Node* n) const {
    // local statics so we can compare std::strings in the checks
    static const std::string shape_op("Shape");
    static const std::string size_op("Size");

    const auto& op_type = n->OpType();
    return op_type == shape_op || op_type == size_op;
  }

  inline bool IsRecomputeNode(const Node* n) const {
    const auto& node_name = n->Name();
    return node_name.find("_recompute") != std::string::npos;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const Node* n1, const Node* n2) const {
    // nodes in global high priority list will be output first
    if (IsHighPri(n1) != IsHighPri(n2)) {
      return IsHighPri(n2);
    }

    // non-recompute nodes should be output first
    if (IsRecomputeNode(n1) != IsRecomputeNode(n2)) {
      return !IsRecomputeNode(n2);
    }

    // nodes with lower priority value will be output first
    if (n1->Priority() != n2->Priority()) {
      return n1->Priority() > n2->Priority();
    }

    // otherwise, nodes with lower index will be output first
    return n1->Index() > n2->Index();
  }
};
#endif

GraphViewer::GraphViewer(const Graph& graph)
    : GraphViewer(graph, nullptr) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph& filter_info)
    : GraphViewer(graph, &filter_info) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph* filter_info)
    : graph_{&graph},
      // we can setup the filter here if needed. filtered_node_indices_ will have been populated by the time it's used
      graph_nodes_{graph_->FilteredNodes(
          filter_info ? [this](NodeIndex idx) { return filtered_node_indices_.count(idx) == 0; }
                      : ConstGraphNodes::NodeFilterFunc(nullptr))},
      filter_info_{filter_info} {
  std::vector<const Node*> leaf_nodes;
  for (auto& node : graph_->Nodes()) {
    // This is a leaf node (without any output node)
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      leaf_nodes.push_back(&node);
    }
    // This is a root node (without any input node)
    if (node.InputEdgesBegin() == node.InputEdgesEnd()) {
      root_nodes_.push_back(node.Index());
    }
  }

  graph.ReverseDFSFrom(
      leaf_nodes,
      nullptr,
      [this](const Node* n) {
        nodes_in_topological_order_.push_back(n->Index());
      },
      NodeCompare());

#if !defined(ORT_MINIMAL_BUILD)
  graph.KahnsTopologicalSort(
      [this](const Node* n) {
        nodes_in_topological_order_with_priority_.push_back(n->Index());
      },
      PriorityNodeCompare());
#endif

  if (filter_info_) {
    // validate. if something is off here it's a bug in our code
    for (NodeIndex idx : filter_info->nodes) {
      ORT_ENFORCE(graph_->GetNode(idx) != nullptr, "IndexedSubGraph contains values not present in the Graph");
    }

    // create set of node indexes as we need quick lookups and don't care about the order
    filtered_node_indices_ = std::unordered_set<NodeIndex>(filter_info->nodes.cbegin(),
                                                           filter_info->nodes.cend());

    const auto& metadef = filter_info->GetMetaDef();

    filtered_node_inputs_.reserve(metadef->inputs.size());
    filtered_node_inputs_including_initializers_.reserve(metadef->inputs.size());

    for (const auto& input : metadef->inputs) {
      const auto* nodearg = graph.GetNodeArg(input);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Input not found:", input);
      filtered_node_inputs_including_initializers_.push_back(nodearg);
      if (!graph.IsInitializedTensor(input)) {
        filtered_node_inputs_.push_back(nodearg);
      }
    }

    for (const auto& output : metadef->outputs) {
      const auto* nodearg = graph.GetNodeArg(output);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Output not found:", output);
      filtered_node_outputs_.push_back(nodearg);
    }

    // filter nodes in topo order to just the nodes in filter_info_
    auto orig_order = std::move(nodes_in_topological_order_);
    nodes_in_topological_order_.reserve(filter_info->nodes.size());
    std::copy_if(orig_order.cbegin(), orig_order.cend(), std::back_inserter(nodes_in_topological_order_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });

    // Filter the initializers also
    // Get the names of all the inputs and implicit inputs of all the nodes in this subgraph
    for (const auto node_idx : filtered_node_indices_) {
      const auto* node = GetNode(node_idx);
      ORT_ENFORCE(node, "Mismatch between Graph and IndexedSubGraph. Node not found: ", node_idx);
      const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
      for (const auto* node_input : node->InputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }

      // The implicit inputs for subgraphs (if any)
      for (const auto* node_input : node->ImplicitInputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }
    }

#if !defined(ORT_MINIMAL_BUILD)
    auto orig_priority_order = std::move(nodes_in_topological_order_with_priority_);
    nodes_in_topological_order_with_priority_.reserve(filter_info->nodes.size());
    std::copy_if(orig_priority_order.cbegin(), orig_priority_order.cend(),
                 std::back_inserter(nodes_in_topological_order_with_priority_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });
#endif
  }
}

// Graph name.
const std::string& GraphViewer::Name() const noexcept {
  return (filter_info_ == nullptr) ? graph_->Name()
                                   : filter_info_->GetMetaDef()->name;
}

const std::string& GraphViewer::Description() const noexcept {
  // filter_info_ doesn't have description so return 'name' instead of nothing
  // and to disambiguate between the full graph's description
  return (filter_info_ == nullptr) ? graph_->Description()
                                   : filter_info_->GetMetaDef()->name;
}

bool GraphViewer::GetInitializedTensor(const std::string& tensor_name,
                                       const ONNX_NAMESPACE::TensorProto*& value) const {
  // if we are using filtered subgraph, the initializer has to be part of the subgraph
  if (filter_info_ != nullptr && filtered_initializers_.find(tensor_name) == filtered_initializers_.cend())
    return false;

  return graph_->GetInitializedTensor(tensor_name, value);
}

bool GraphViewer::CanOverrideInitializer() const noexcept {
  return graph_->CanOverrideInitializer();
}

// Graph inputs excluding initializers.
const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputs()
                                   : filtered_node_inputs_;
}
// Graph inputs including initializers. Contains no nullptr values.
// This will match the number and order of inputs from the GraphProto.
const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputsIncludingInitializers()
                                   : filtered_node_inputs_including_initializers_;
}

// Graph outputs. Should have no nullptr values.
const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetOutputs()
                                   : filtered_node_outputs_;
}

// Get graph value infos.
const std::vector<const NodeArg*>& GraphViewer::GetValueInfo() const noexcept {
  return graph_->GetValueInfo();
}

// Get const Node given specific node index. May return nullptr if node as been freed.
const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  if (filter_info_ && filtered_node_indices_.count(node_index) == 0) {
    return nullptr;
  }

  return graph_->GetNode(node_index);
}

const ConstGraphNodes& GraphViewer::Nodes() const noexcept {
  return graph_nodes_;
}

int GraphViewer::NumberOfNodes() const noexcept {
  return (filter_info_ == nullptr) ? graph_->NumberOfNodes()
                                   : gsl::narrow_cast<int>(filter_info_->nodes.size());
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return graph_->MaxNodeIndex();
}

const std::vector<NodeIndex>& GraphViewer::GetNodesInTopologicalOrder(ExecutionOrder order) const {
  switch (order) {
    case ExecutionOrder::DEFAULT:
      return nodes_in_topological_order_;
#if !defined(ORT_MINIMAL_BUILD)
    case ExecutionOrder::PRIORITY_BASED:
      return nodes_in_topological_order_with_priority_;
#endif
    default:
      ORT_THROW("Invalid ExecutionOrder");
  }
}

const std::vector<NodeIndex>& GraphViewer::GetRootNodes() const {
  // TODO: See if we need to calculate the root_nodes_ of the filtered graph.
  // GetRootNodes is only used by parallel executor currently, and isn't relevant to the usage of a filtered graph.
  ORT_ENFORCE(filter_info_ == nullptr, "Not supported with filtered graph.");

  return root_nodes_;
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return (filter_info_ == nullptr)
             ? graph_->GetAllInitializedTensors()
             : filtered_initializers_;
}

const NodeArg* GraphViewer::GetNodeArg(const std::string& name) const {
  return graph_->GetNodeArg(name);
}

bool GraphViewer::IsSubgraph() const {
  return graph_->IsSubgraph();
}

bool GraphViewer::IsConstantInitializer(const std::string& name, bool check_outer_scope) const {
  return graph_->GetConstantInitializer(name, check_outer_scope) != nullptr;
}

}  // namespace onnxruntime
