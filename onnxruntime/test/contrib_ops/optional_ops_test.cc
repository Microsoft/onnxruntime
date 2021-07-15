// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(OptionalOpTest, OptionalTensorCreateFromTensor) {
  OpTester test("Optional", 1, onnxruntime::kMSDomain);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddInput<float>("A", {2}, data);
  test.AddOptionalTypeTensorOutput<float>("Y", {2}, &data);

  test.Run();
}

TEST(OptionalOpTest, OptionalSeqTensorCreateFromSeqTensor) {
  OpTester test("Optional", 1, onnxruntime::kMSDomain);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  test.AddSeqInput("S", data);
  test.AddOptionalTypeSeqOutput<float>("Y", &data);

  test.Run();
}
TEST(OptionalOpTest, OptionalTensorCreateFromTypeProto) {
  OpTester test("Optional", 1, onnxruntime::kMSDomain);

  onnx::TypeProto tp;
  tp.mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

  test.AddAttribute<onnx::TypeProto>("type", tp);
  // expected value is nullptr because we expect a "None" output
  test.AddOptionalTypeTensorOutput<float>("Y", {2}, nullptr);

  test.Run();
}

TEST(OptionalOpTest, OptionalSeqTensorCreateFromTypeProto) {
  OpTester test("Optional", 1, onnxruntime::kMSDomain);

  onnx::TypeProto tensor_tp;
  tensor_tp.mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

  onnx::TypeProto tp;
  tp.mutable_sequence_type()->mutable_elem_type()->CopyFrom(tensor_tp);

  test.AddAttribute<onnx::TypeProto>("type", tp);
  // expected value is nullptr because we expect a "None" output
  test.AddOptionalTypeSeqOutput<float>("Y", nullptr);

  test.Run();
}

TEST(OptionalOpTest, OptionalTensorHasElement_True) {
  OpTester test("OptionalHasElement", 1, onnxruntime::kMSDomain);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddOptionalTypeTensorInput<float>("A", {2}, &data);
  test.AddOutput<bool>("Y", {}, {true});

  test.Run();
}

TEST(OptionalOpTest, OptionalSeqTensorHasElement_True) {
  OpTester test("OptionalHasElement", 1, onnxruntime::kMSDomain);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  test.AddOptionalTypeSeqInput<float>("A", &data);
  test.AddOutput<bool>("Y", {}, {true});

  test.Run();
}

TEST(OptionalOpTest, OptionalTensorHasElement_False) {
  OpTester test("OptionalHasElement", 1, onnxruntime::kMSDomain);

  // Input is an optional type and is None
  test.AddOptionalTypeTensorInput<float>("A", {2}, nullptr);
  test.AddOutput<bool>("Y", {}, {false});

  test.Run();
}

TEST(OptionalOpTest, OptionalSeqTensorHasElement_False) {
  OpTester test("OptionalHasElement", 1, onnxruntime::kMSDomain);

  // Input is an optional type and is None
  test.AddOptionalTypeSeqInput<float>("A", nullptr);
  test.AddOutput<bool>("Y", {}, {false});

  test.Run();
}

TEST(OptionalOpTest, OptionalTensorGetElement) {
  OpTester test("OptionalGetElement", 1, onnxruntime::kMSDomain);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddOptionalTypeTensorInput<float>("A", {2}, &data);
  test.AddOutput<float>("Y", {2}, {-1.0856307f, 0.99734545f});

  test.Run();
}

TEST(OptionalOpTest, OptionalSeqTensorGetElement) {
  OpTester test("OptionalGetElement", 1, onnxruntime::kMSDomain);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddOptionalTypeSeqInput<float>("A", &data);
  test.AddSeqOutput("S", data);

  test.Run();
}

class OptionalOpTester : public OpTester {
 public:
  OptionalOpTester()
      : OpTester("Optional", 1, kMSDomain) {}

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    /*
    * Both the Optional* ops produce intermediate outputs
    
                      Graph Input  
                          |
                          |
                      Optional
                          |
                          |
                      OptionalGetElement
                         |
                         |
                       SampleOp  (use something other than Identity because it could be optimized away during Identity elimination)
                                 Also preferably use something from kMsDomain as it seems like the setup to do cross-domain model creation
                                 is not complete in OpTester.
                                 SampleOp is just an Identity behavior-wise. 
                         |
                         | 
                      Graph output
      */

    onnx::TypeProto tensor_type_proto;
    tensor_type_proto.mutable_tensor_type()
        ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

    onnx::TypeProto optional_type_proto;
    optional_type_proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(tensor_type_proto);

    auto& optional_node_arg = graph.GetOrCreateNodeArg("optional_output", &optional_type_proto);
    ORT_IGNORE_RETURN_VALUE(graph.AddNode("optional_create", "Optional", "Create optional type",
                                          {graph_input_defs[0]}, {&optional_node_arg},
                                          nullptr, kMSDomain));

    auto& tensor_node_arg = graph.GetOrCreateNodeArg("tensor_output", &tensor_type_proto);
    ORT_IGNORE_RETURN_VALUE(graph.AddNode("optional_get_element", "OptionalGetElement",
                                          "Parse optional type",
                                          {&optional_node_arg}, {&tensor_node_arg},
                                          nullptr, kMSDomain));

    ORT_IGNORE_RETURN_VALUE(graph.AddNode("SampleOp", "SampleOp",
                                          "Identity",
                                          {&tensor_node_arg}, {graph_output_defs[0]},
                                          nullptr, kMSDomain));
  }
};

TEST(OptionalOpTest, OptionalOpsValidateOrtValueReUse) {
  // We create a simple model with 2 optional ops - Optional and OptionalGetElement
  // that don't produce graph outputs and we have logic in the allocation planner to
  // re-use input OrtValues as output OrtValues for such cases. We test that the model
  // executes fine with sch logic.
  // TODO: How to ensure that re-use took place ?

  OptionalOpTester test;

  std::initializer_list<float> data = {1.f, 2.f};

  test.AddInput<float>("A", {2}, data);
  test.AddOutput<float>("Y", {2}, {1.f, 2.f});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
