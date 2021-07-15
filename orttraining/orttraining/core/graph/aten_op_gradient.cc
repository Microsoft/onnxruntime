// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/aten_op_gradient.h"

#include <nlohmann/json.hpp>

namespace onnxruntime {
namespace training {

using json = nlohmann::json;

void ParseATenOpGradientDefinition(const std::string& grad_def_json_str, std::vector<NodeDefinition>& grad_def) {
  json grad_def_json = json::parse(grad_def_json_str);
  ORT_ENFORCE(grad_def_json.is_array(), "Gradient definition must be a list of node definitions.");
  for (const auto& node_def_json : grad_def_json) {
    NodeDefinition node_def;
    node_def.op_type = node_def_json.at("op_type").get<std::string>();
    if (node_def_json.contains("domain")) {
      node_def.domain = node_def_json.at("domain").get<std::string>();
    }

    for (const auto& input : node_def_json.at("inputs")) {
      node_def.inputs.emplace_back(input.get<std::string>());
    }

    for (const auto& output : node_def_json.at("outputs")) {
      node_def.outputs.emplace_back(output.get<std::string>());
    }

    if (node_def_json.contains("attributes")) {
      for (auto& attribute : node_def_json.at("attributes").items()) {
        AttributeDefinition attr_def;
        const auto& attr_def_json = attribute.value();
        attr_def.value_json = attr_def_json.at("value").dump();
        attr_def.dtype = attr_def_json.at("dtype").get<std::string>();
        attr_def.is_tensor = false;
        if (attr_def_json.contains("is_tensor")) {
          attr_def.is_tensor = attr_def_json.at("is_tensor").get<bool>();
        }

        node_def.attributes.emplace(attribute.key(), attr_def);
      }
    }

    grad_def.emplace_back(node_def);
  }
}

}  // namespace training
}  // namespace onnxruntime
