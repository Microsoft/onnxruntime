// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace training {

typedef const char* (*GetGradientDefinitionFunc)(const char* op_name, const char* overload_name);

struct PairHash {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

// We try to parse more JSON contents to these structs as this is one-time effort. But for some attributes, the final
// value and dtype is determined at runtime, e.g., when the dtype is IElemType(i) or OElemType(i), so here we still keep
// the value in JSON format and dtype in string format. We will handle these at runtime for each node.
struct AttributeDefinition {
  std::string value_json;
  std::string dtype;
  bool is_tensor;
};

struct NodeDefinition {
  std::string op_type;
  std::string domain = kOnnxDomain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, AttributeDefinition> attributes;
};

void ParseATenOpGradientDefinition(const std::string& grad_def_json_str, std::vector<NodeDefinition>& grad_def);

class ATenOpGradientDefinitionGetter {
 public:
  static ATenOpGradientDefinitionGetter& Instance() { return InstanceImpl(); }

  static void Initialize(void* p_get_gradient_definition_func_raw) { InstanceImpl(p_get_gradient_definition_func_raw); }

  const std::vector<NodeDefinition>& operator()(const std::string& op_name, const std::string& overload_name) {
    auto key = std::make_pair(op_name, overload_name);
    if (definitions_.find(key) == definitions_.end()) {
      ORT_ENFORCE(p_get_gradient_definition_func_, "ATenOpGradientDefinitionGetter is not initialized.");
      std::vector<NodeDefinition> grad_def;
      const char* grad_def_json = p_get_gradient_definition_func_(op_name.c_str(), overload_name.c_str());
      if (grad_def_json) {
        ParseATenOpGradientDefinition(grad_def_json, grad_def);
      }
      definitions_.emplace(key, grad_def);
    }

    return definitions_.at(key);
  }

 private:
  static ATenOpGradientDefinitionGetter& InstanceImpl(void* p_get_gradient_definition_func_raw = nullptr) {
    static ATenOpGradientDefinitionGetter instance(p_get_gradient_definition_func_raw);
    return instance;
  }

  ATenOpGradientDefinitionGetter(void* p_get_gradient_definition_func_raw) {
    ORT_ENFORCE(p_get_gradient_definition_func_raw);
    p_get_gradient_definition_func_ = reinterpret_cast<GetGradientDefinitionFunc>(p_get_gradient_definition_func_raw);
  }

  GetGradientDefinitionFunc p_get_gradient_definition_func_;
  std::unordered_map<std::pair<std::string, std::string>, std::vector<NodeDefinition>, PairHash> definitions_;
};

}  // namespace training
}  // namespace onnxruntime
