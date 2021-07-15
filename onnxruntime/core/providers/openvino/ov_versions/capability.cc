// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "../backend_utils.h"
#include "../backend_manager.h"
#include "capabilities.h"
#include "utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

//Constructor 
GetCapability::GetCapability(const GraphViewer& graph_viewer_param, std::string device_type_param,
                             const std::string version_param):
                graph_viewer_(graph_viewer_param), device_type_(device_type_param){
  if (version_param == "V_2021_2") {
    data_ops_ = new DataOps(graph_viewer_, V_2021_2, device_type_);
  } else if (version_param == "V_2021_1") {
    data_ops_ = new DataOps(graph_viewer_, V_2021_1, device_type_);
  } else if (version_param == "V_2021_3") {
    data_ops_ = new DataOps(graph_viewer_, V_2021_3, device_type_);
  } else if (version_param == "V_2021_4") {
    data_ops_ = new DataOps(graph_viewer_, V_2021_4, device_type_);
  } else {
    data_ops_ = new DataOps(graph_viewer_, V_2021_4, device_type_);
  }
}

std::vector<std::unique_ptr<ComputeCapability>> GetCapability::Execute() {

  std::vector<std::unique_ptr<ComputeCapability>> result;

  //Check if it is a subgraph
  if (graph_viewer_.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer_.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] Initializers with external data location are not currently supported";
      return result;
    }
  }

  // This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = data_ops_->GetUnsupportedNodeIndices(ng_required_initializers);
  #ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "No of unsupported nodes " << unsupported_nodes.size() << std::endl;
    for (size_t i = 0; i < unsupported_nodes.size(); i++) {
      const auto& node = graph_viewer_.GetNode(unsupported_nodes[i]);
      std::cout << "Unsupported node op " << node->OpType() << std::endl;
    }
  }
  #endif

  //If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    //Fill inputs with names
    std::for_each(graph_viewer_.GetInputs().begin(), graph_viewer_.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    const auto& nodes = graph_viewer_.GetNodesInTopologicalOrder();
    //Nodes that work well in models but not as a single node
    if (nodes.size() == 1) {
      const auto& node = graph_viewer_.GetNode(nodes[0]);
      if (data_ops_->IsOpSupportedOnlyInModel(node->OpType()))
        return result;
      //If reshape is not an intermediate node, shape needs to be an initializer
      if(data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node)) {
        return result;
      }
    }

    //Initializers need to be part of meta_def->inputs
    std::for_each(ng_required_initializers.begin(), ng_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    //Fill outputs with names
    std::for_each(graph_viewer_.GetOutputs().begin(), graph_viewer_.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer_.GetNodesInTopologicalOrder(), inputs, outputs, result);

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is fully supported by OpenVINO";
    openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph = true;

  } else {  // unsupported_nodes_idx.empty()
   
    std::vector<NodeIndex> modified_unsupported_nodes;
    for (const auto& node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
      if (find(unsupported_nodes.begin(), unsupported_nodes.end(), node_idx) != unsupported_nodes.end()) {
        modified_unsupported_nodes.push_back(node_idx);
      } else {
        auto node = graph_viewer_.GetNode(node_idx);
        const auto& optype = node->OpType();
        if (data_ops_->InsertNode(node, optype)) {
          modified_unsupported_nodes.push_back(node_idx);
        }
      }
    }

    auto ng_clusters = GetPartitionedClusters(graph_viewer_.GetNodesInTopologicalOrder(), modified_unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_viewer_, ng_clusters);

    //Myriad plugin can only load 10 subgraphs
    if (device_type_ == "MYRIAD" && connected_clusters.size() > 10) {
      std::sort(connected_clusters.begin(), connected_clusters.end(),
                [](const std::vector<NodeIndex>& v1, const std::vector<NodeIndex>& v2) -> bool {
                  return v1.size() > v2.size();
                });
    }
    int no_of_clusters = 0;

    for (auto this_cluster : connected_clusters) {
      if (device_type_ == "MYRIAD" && no_of_clusters == 10) {
        break;
      }

      //If subgraph has less then three, graph is considered trivial
      if (this_cluster.size() < 3) {
        continue;
      } else {
        //If subgraph only has Identity node, EyeLike or Dropout, OpenVINO EP doesn't support it.
        if (this_cluster.size() == 1) {
          const auto& node = graph_viewer_.GetNode(this_cluster[0]);
          if (IsOpSupportedOnlyInModel(node->OpType()))
            continue;
          //If reshape is not an intermediate node, shape needs to be an initializer
          if(data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node))
            continue;
        }
      }  

      std::vector<std::string> cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs;

      GetInputsOutputsOfCluster(graph_viewer_, this_cluster, ng_required_initializers, cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs);

      bool omit_subgraph = false;
      //Omitting zero dim subgraphs
      for (auto index : this_cluster) {
         const Node* node = graph_viewer_.GetNode(index);
        if (data_ops_->DoNotOmitSubGraph(node->OpType())) {
          for (const auto& input : node->InputDefs()) {
            auto input_name = input->Name();
            auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), input_name);
            if (it != cluster_graph_inputs.end()) {
              omit_subgraph = true;
              break;
            }
          }
        }
      
        if (node->OpType() == "Conv" || node->OpType() == "Identity") {
          auto output_name = node->OutputDefs()[0]->Name();
          auto it = find(cluster_outputs.begin(), cluster_outputs.end(), output_name);
          if (it != cluster_outputs.end() && node->GetOutputEdgesCount() != 0) {
            omit_subgraph = true;
            break;
          }
        }
        
        std::map<std::string, int> slice_map;
        if (node->OpType() == "Slice") {
          auto input = node->InputDefs()[0];
          auto input_name = input->Name();
          const bool is_data_int32 = input->Type()->find("int32") != std::string::npos;
          const bool is_data_int64 = input->Type()->find("int64") != std::string::npos;
          auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), input_name);
          if (it != cluster_graph_inputs.end()) {
            if (device_type_.find("MYRIAD") != std::string::npos && (is_data_int32 || is_data_int64)) {
              omit_subgraph = true;
              break;
            }
            if (slice_map.count(input_name) == 0) {
              slice_map[input_name] = 1;
            } else {
              omit_subgraph = true;
              break;
            }
          }
        }
      }
      if (omit_subgraph)
        continue;

      /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
      if (!cluster_inputs.empty()) {
        AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
        no_of_clusters++;
      }
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }

  return result;
} 

}
}