// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/core/graph/aten_op_gradient.h"

namespace onnxruntime {
namespace test {

using namespace training;

namespace {
NodeDefinition CreateNodeDefinition(const std::string& op_type, const std::string& domain,
                                    const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
  NodeDefinition def;
  def.op_type = op_type;
  def.domain = domain;
  def.inputs = inputs;
  def.outputs = outputs;
  return def;
}

void AddAttribute(NodeDefinition& def, const std::string& name, const std::string& value_json, const std::string& dtype,
                  bool is_tensor) {
  AttributeDefinition attr_def;
  attr_def.value_json = value_json;
  attr_def.dtype = dtype;
  attr_def.is_tensor = is_tensor;
  def.attributes.emplace(name, attr_def);
}

void CompareAttributeDefinition(const AttributeDefinition& def, const AttributeDefinition& other) {
  EXPECT_TRUE(def.value_json == other.value_json && def.dtype == other.dtype && def.is_tensor == other.is_tensor);
}

void CompareGradientDefinition(const std::vector<NodeDefinition>& def, const std::vector<NodeDefinition>& other) {
  EXPECT_TRUE(def.size() == other.size());
  for (size_t i = 0; i < def.size(); i++) {
    EXPECT_TRUE(def[i].op_type == other[i].op_type && def[i].domain == other[i].domain &&
                def[i].inputs == other[i].inputs && def[i].outputs == other[i].outputs);
    EXPECT_TRUE(def[i].attributes.size() == other[i].attributes.size());
    for (const auto& attr : def[i].attributes) {
      EXPECT_TRUE(other[i].attributes.find(attr.first) != other[i].attributes.end());
      CompareAttributeDefinition(attr.second, other[i].attributes.at(attr.first));
    }
  }
}
}  // namespace

TEST(ATenOpGradientDefinitionTest, ValidATenOpGradientDefinition) {
  {
    std::string grad_json =
        R"***([{"op_type": "Constant", "inputs": [], "outputs": ["Const_0"], "attributes": {"value": {"value": 0, "dtype": "int", "is_tensor": true}}}, {"op_type": "CustomOp", "inputs": ["Const_0"], "outputs": ["GI(0)"], "attributes": {"ep": {"value": 2.0, "dtype": "float"}}}])***";
    std::vector<NodeDefinition> grad_def;
    ParseATenOpGradientDefinition(grad_json, grad_def);
    std::vector<NodeDefinition> expected;
    NodeDefinition node_def = CreateNodeDefinition("Constant", "", {}, {"Const_0"});
    AddAttribute(node_def, "value", "0", "int", true);
    expected.emplace_back(node_def);
    node_def = CreateNodeDefinition("CustomOp", "", {"Const_0"}, {"GI(0)"});
    AddAttribute(node_def, "ep", "2.0", "float", false);
    expected.emplace_back(node_def);
    CompareGradientDefinition(grad_def, expected);
  }

  {
    std::string grad_json =
        R"***([{"op_type": "CustomOp", "inputs": ["Input1", "Input2"], "outputs": ["Output1"], "attributes": {"axes": {"value": [0, 2, 4], "dtype": "int"}}}])***";
    std::vector<NodeDefinition> grad_def;
    ParseATenOpGradientDefinition(grad_json, grad_def);
    std::vector<NodeDefinition> expected;
    NodeDefinition node_def = CreateNodeDefinition("CustomOp", "", {"Input1", "Input2"}, {"Output1"});
    AddAttribute(node_def, "axes", "[0,2,4]", "int", false);
    expected.emplace_back(node_def);
    CompareGradientDefinition(grad_def, expected);
  }

  {
    std::string grad_json =
        R"***([{"op_type": "Shape", "inputs": ["I(0)"], "outputs": ["Shape_X"]}, {"op_type": "ATenOp", "domain": "com.microsoft", "inputs": ["GO(0)", "Shape_X", "I(1)", "I(2)", "I(3)"], "outputs": ["GI(0)"], "attributes": {"name": {"value": "aten::unfold_backward", "dtype": "string"}}}])***";
    std::vector<NodeDefinition> grad_def;
    ParseATenOpGradientDefinition(grad_json, grad_def);
    std::vector<NodeDefinition> expected;
    NodeDefinition node_def = CreateNodeDefinition("Shape", "", {"I(0)"}, {"Shape_X"});
    expected.emplace_back(node_def);
    node_def = CreateNodeDefinition("ATenOp", "com.microsoft", {"GO(0)", "Shape_X", "I(1)", "I(2)", "I(3)"}, {"GI(0)"});
    AddAttribute(node_def, "name", "\"aten::unfold_backward\"", "string", false);
    expected.emplace_back(node_def);
    CompareGradientDefinition(grad_def, expected);
  }
}

TEST(ATenOpGradientDefinitionTest, InvalidATenOpGradientDefinition) {
  bool is_valid = true;
  try {
    std::string grad_json = R"***({"op_type": "Shape", "inputs": ["I(0)"], "outputs": ["Shape_X"]})***";
    std::vector<NodeDefinition> grad_def;
    ParseATenOpGradientDefinition(grad_json, grad_def);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("Gradient definition must be a list of node definitions.");
    ASSERT_TRUE(ret != std::string::npos);
    is_valid = false;
  }

  ASSERT_FALSE(is_valid);
}

}  // namespace test
}  // namespace onnxruntime
