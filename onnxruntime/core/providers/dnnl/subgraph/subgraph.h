// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <map>
#include "core/providers/providers.h"

namespace onnxruntime {
namespace ort_dnnl {

// When training record extra node information to enable finding the
// forward node from the backward node.
#ifdef ENABLE_TRAINING
struct InputNode {
  onnxruntime::NodeIndex index;
  std::string op_type;
};
#endif  // ENABLE_TRAINING

struct DnnlNode {
  std::string name;
  int node_index = -1;
  int input_start_index = -1;  // start index in inputs()
  int num_inputs = 0;          // and how many inputs
  int output_index = -1;       // index in output()
  std::string weight_name;
  std::string output_name;
#ifdef ENABLE_TRAINING
  int num_outputs = 0;  // how many outputs
  std::vector<std::string> output_names;
  // num_outputs will equal the number of required outputs while is_ort_output_required
  // will have a true or false entry for each ORT output. The number of true entries
  // must equal the num_outputs.
  std::vector<bool> is_ort_output_required;
#endif  // ENABLE_TRAINING
  std::vector<size_t> parent_nodes;  // index to parents in vector mklnodes

#ifdef ENABLE_TRAINING
  onnxruntime::NodeIndex onnx_index;   // the index of the onnx runtime node
  std::vector<InputNode> input_nodes;  // index and node name of the onnx runtime input nodes to this node
#endif  // ENABLE_TRAINING

  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append(name);
    key.append("-");
    key.append(std::to_string(input_start_index));
    key.append("-");
    key.append(std::to_string(num_inputs));
    key.append("-");
    key.append(std::to_string(output_index));
    key.append("-");
    key.append(output_name);
    key.append("-");
    for (auto& out : parent_nodes)
      key.append(std::to_string(out) + ",");
    key.append(";");
    return key;
  }
};

struct Subgraph {
  struct SubgraphVariables {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> outputs_as_input_other_node;
    std::vector<onnxruntime::NodeIndex> subgraph_node_indexes;

    void Reset() {
      subgraph_node_indexes.clear();
      inputs.clear();
      outputs.clear();
      outputs_as_input_other_node.clear();
    }
  };

  Subgraph(const std::string& name) {
    graph_name = name;
  }

  std::string graph_name;
  std::string subgraph_id;
  std::vector<DnnlNode> dnnl_nodes;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
