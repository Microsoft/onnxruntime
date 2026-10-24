// Copyright (C) 2019- Intel Corporation
// Licensed under the MIT License
#include <map>
#include <algorithm>
#include <unordered_set>
#include <type_traits>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/providers/openvino/ov_versions/utils.h"
#include "openvino/core/version.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

// Constructor
GetCapability::GetCapability(const EPCtxHandler& ep_ctx_handler,
                             const GraphViewer& graph_viewer_param,
                             const std::string device_type_param,
                             const bool enable_qdq_optimizer) : ep_ctx_handler_(ep_ctx_handler),
                                                                graph_viewer_(graph_viewer_param),
                                                                device_type_(std::move(device_type_param)) {
  bool npu_qdq_optimizer_enabled = false;
  if (device_type_.find("NPU") != std::string::npos) {
    device_type_ = "CPU";
    if (enable_qdq_optimizer) npu_qdq_optimizer_enabled = true;
  } else if (enable_qdq_optimizer && device_type_.find("GPU") != std::string::npos) {
    npu_qdq_optimizer_enabled = true;  // see data_ops.cc ~615 where we check for int16 types for gpu, this may change to a better approach later
  }

#if OPENVINO_VERSION_MAJOR == 2026 && OPENVINO_VERSION_MINOR == 2
  data_ops_ = std::make_unique<DataOps>(graph_viewer_, V_2026_2, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2026 && OPENVINO_VERSION_MINOR == 1
  data_ops_ = std::make_unique<DataOps>(graph_viewer_, V_2026_1, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2026 && OPENVINO_VERSION_MINOR == 0
  data_ops_ = std::make_unique<DataOps>(graph_viewer_, V_2026_0, device_type_, npu_qdq_optimizer_enabled);
#else
  data_ops_ = std::make_unique<DataOps>(graph_viewer_, V_2026_2, device_type_, npu_qdq_optimizer_enabled);
#endif
}

std::vector<std::unique_ptr<ComputeCapability>> GetCapability::Execute() {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Check if it is a subgraph
  if (graph_viewer_.IsSubgraph() && graph_viewer_.Name() == "tf2onnx") {
    return result;
  }

  auto Iterable2String = []<typename U, typename V>(U& strings, const V& node_args) {
    constexpr bool has_name = requires(V v) {
      (*v.begin())->Name();
    };
    for (const auto& arg : node_args) {
      if constexpr (has_name) {
        strings.push_back(arg->Name());
      } else {
        strings.push_back(arg);
      }
    }
  };

  // Check for EpContext nodes
  const auto& nodes = graph_viewer_.GetNodesInTopologicalOrder();
  if (nodes.size() > 1 && std::all_of(nodes.begin(), nodes.end(), [this](const auto index) {
        const auto* node = graph_viewer_.GetNode(index);
        return node != nullptr && ep_ctx_handler_.CheckForOVEPCtxNode(*node);
      })) {
    // Cached partitions must be imported individually, not translated as ONNX EPContext operations.
    for (const auto index : nodes) {
      const auto* node = graph_viewer_.GetNode(index);
      std::vector<std::string> inputs;
      std::vector<std::string> outputs;
      Iterable2String(inputs, node->InputDefs());
      Iterable2String(outputs, node->OutputDefs());
      AppendClusterToSubGraph({index}, inputs, outputs, result);
    }
    is_wholly_supported_graph_ = true;
    return result;
  }

  // Build set of all outputs actually produced by nodes in the graph
  std::unordered_set<std::string> valid_node_outputs;
  for (const auto& node_idx : nodes) {
    const Node* n = graph_viewer_.GetNode(node_idx);
    for (const auto* output_def : n->OutputDefs()) {
      if (output_def) {
        valid_node_outputs.insert(output_def->Name());
      }
    }
  }

  // Lambda to filter graph outputs, only including those with producer nodes
  auto FilterValidOutputs = [&valid_node_outputs, &graph_viewer = graph_viewer_](std::vector<std::string>& outputs) {
    for (const auto& output_node_arg : graph_viewer.GetOutputs()) {
      const auto& output_name = output_node_arg->Name();
      if (valid_node_outputs.count(output_name)) {
        outputs.push_back(output_name);
      }
#ifndef NDEBUG
      else {
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::cout << "Skipping orphaned graph output '" << output_name
                    << "' - no producer node found" << std::endl;
        }
      }
#endif
    }
  };

  // If all the nodes have been accounted for then no more processing is needed
  if (result.size() == nodes.size()) {
    is_wholly_supported_graph_ = true;
    return result;
  }

  // This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = data_ops_->GetUnsupportedNodeIndices(ng_required_initializers, has_external_weights_);
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "No of unsupported nodes " << unsupported_nodes.size() << std::endl;
    for (size_t i = 0; i < unsupported_nodes.size(); i++) {
      const Node* unode = graph_viewer_.GetNode(unsupported_nodes[i]);
      std::cout << "Unsupported node op " << unode->OpType() << std::endl;
    }
  }
#endif

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    auto input_nodes = graph_viewer_.GetInputs();
    // Input is not a tensor, OV only handle a tensor input
    for (auto& node : input_nodes) {
      auto shape = node->Shape();
      if (!shape) {
        return result;
      }
    }
    // Fill inputs with names
    Iterable2String(inputs, input_nodes);

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    const Node* node = graph_viewer_.GetNode(nodes[0]);

    // Handle cases where lone, reoccuring Ops in smaller models cannot be supported in OpenVINO
    // If only a node of the same lone,unsupported type is present, then do not proceed with the subgraph
    if (nodes.size() <= 3) {
      if (data_ops_->IsOpSupportedOnlyInModel(node->OpType())) {
        return result;
      }
    }

    // Nodes that work well in models but not as a single node
    if (nodes.size() == 1) {
      // If reshape is not an intermediate node, shape needs to be an initializer
      if (data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node)) {
        return result;
      }
    }

    // Use filtered outputs to avoid orphaned graph outputs
    FilterValidOutputs(outputs);

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer_.GetNodesInTopologicalOrder(), inputs, outputs, result);

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is fully supported by OpenVINO";
    // Enable CI Logs
    if (backend_utils::IsCILogEnabled()) {
      std::cout << "Model is fully supported on OpenVINO" << std::endl;
    }
    is_wholly_supported_graph_ = true;

  } else {                                     // unsupported_nodes_idx.empty()
#if defined(OPENVINO_DISABLE_GRAPH_PARTITION)  // disables graph partition at build time
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DISABLE_GRAPH_PARTITION option is set";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is not fully supported by OpenVINO, "
                       << "so making the full model fall back to default CPU Execution Provider";
    return result;
#endif

    std::vector<NodeIndex> modified_unsupported_nodes;
    for (const NodeIndex& node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
      if (find(unsupported_nodes.begin(), unsupported_nodes.end(), node_idx) != unsupported_nodes.end()) {
        modified_unsupported_nodes.push_back(node_idx);
      } else {
        const Node* node = graph_viewer_.GetNode(node_idx);
        const std::string& optype = node->OpType();
        if (data_ops_->InsertNode(optype)) {
          modified_unsupported_nodes.push_back(node_idx);
        }
      }
    }

    auto ng_clusters = GetPartitionedClusters(graph_viewer_.GetNodesInTopologicalOrder(), modified_unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_viewer_, ng_clusters);

    int no_of_clusters = 0;
    size_t cluster_index = 0;
    size_t total_clusters = connected_clusters.size();
    for (auto this_cluster : connected_clusters) {
      bool omit_subgraph = false;

      if (this_cluster.size() == 1) {
        // check next cluster
        auto index = this_cluster.at(0);
        size_t j = cluster_index;
        if (graph_viewer_.GetNode(index)->OpType() == "EPContext") {
          omit_subgraph = false;
        } else if (j < total_clusters - 1) {
          bool append_node = false;
          while (j < total_clusters - 1 && !append_node) {
            j = j + 1;
            append_node = AddTrivialClusterToNextClusterIfConnected(graph_viewer_, index, connected_clusters[j]);
          }
          if (append_node) {
            connected_clusters[j].emplace_back(index);
          }
          omit_subgraph = true;
        }
      }

      std::vector<std::string> cluster_graph_inputs, cluster_inputs, cluster_outputs;

      GetInputsOutputsOfCluster(graph_viewer_,
                                this_cluster,
                                ng_required_initializers,
                                cluster_graph_inputs,
                                cluster_inputs,
                                cluster_outputs);

      // Omitting zero dim subgraphs
      for (auto index : this_cluster) {
        const Node* node = graph_viewer_.GetNode(index);

        if (node->OpType() == "Conv" || node->OpType() == "Identity") {
          const auto& output_name = node->OutputDefs()[0]->Name();
          auto it = find(cluster_outputs.begin(), cluster_outputs.end(), output_name);
          if (it != cluster_outputs.end() && node->GetOutputEdgesCount() != 0) {
            omit_subgraph = true;
            break;
          }
        }

        std::map<std::string, int> slice_map;
        if (node->OpType() == "Slice") {
          auto input = node->InputDefs()[0];
          const auto& input_name = input->Name();
          auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), input_name);
          if (it != cluster_graph_inputs.end()) {
            if (slice_map.count(input_name) == 0) {
              slice_map[input_name] = 1;
            } else {
              omit_subgraph = true;
              break;
            }
          }
        }
      }

      /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
      if (!omit_subgraph) {
        if (!cluster_inputs.empty()) {
          AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
          no_of_clusters++;
        }
      }

      cluster_index = cluster_index + 1;
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }
  return result;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
