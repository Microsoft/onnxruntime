// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "test/providers/provider_test_utils.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "onnx/defs/function.h"
#include "core/graph/function_impl.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/inference_session.h"
#endif

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define OPERATOR_SCHEMA UNUSED ONNX_OPERATOR_SCHEMA

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

static bool RegisterCustomSchemas() {
  OPERATOR_SCHEMA(Variable_DFS)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(Add_DFS)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(NoOp_DFS)
      .SetDoc("Operator doing nothing.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  OPERATOR_SCHEMA(Variable_Fake)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(Add_Fake)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(NoOp_Fake)
      .SetDoc("Operator doing nothing.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  OPERATOR_SCHEMA(Identity_Fake)
      .SetDoc("Identity.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(Merge_Fake)
      .SetDoc("Merge.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  // we need more than 8 outputs to trigger the unordered_map that's used in Graph::SetGraphInputsOutputs to
  // re-allocate and re-order to prove the code works.
  OPERATOR_SCHEMA(Split_Fake)
      .SetDoc("Split.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)")
      .Output(1, "output_2", "docstr for output_2.", "tensor(int32)")
      .Output(2, "output_3", "docstr for output_3.", "tensor(int32)")
      .Output(3, "output_4", "docstr for output_4.", "tensor(int32)")
      .Output(4, "output_5", "docstr for output_5.", "tensor(int32)")
      .Output(5, "output_6", "docstr for output_6.", "tensor(int32)")
      .Output(6, "output_7", "docstr for output_7.", "tensor(int32)")
      .Output(7, "output_8", "docstr for output_8.", "tensor(int32)")
      .Output(8, "output_9", "docstr for output_9.", "tensor(int32)")
      .Output(9, "output_10", "docstr for output_10.", "tensor(int32)");

  OPERATOR_SCHEMA(Variable2_Fake)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(int32)", "tensor(float)"}, "input/output types");

  OPERATOR_SCHEMA(Max_Fake)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Input(1, "input_2", "docstr for input_2.", "T")
      .Input(2, "input_3", "docstr for input_3.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(int32)", "tensor(float)"}, "input/output types");

  OPERATOR_SCHEMA(ShapeInferenceThrowsOp)
      .SetDoc("Throw shape inference error.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)")
      .TypeAndShapeInferenceFunction([](InferenceContext&) {
        fail_shape_inference("try harder");
      });

  OPERATOR_SCHEMA(Fake_Sub)
      .SinceVersion(1)
      .SetDomain(kMSNchwcDomain)
      .Input(0, "A", "First operand.", "T")
      .Input(1, "B", "Second operand.", "T")
      .Output(0, "C", "Result, has same element type as two inputs", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });

  // Fake Function Op where the domain for the Op itself and the Ops in the function body is not same.
  // Function Op belongs to com.microsoft domain where as function body ops belong to onnx domain and nchwdomain.
  OPERATOR_SCHEMA(Fake_FunctionOp)
      .SetName("Fake_FunctionOp")
      .SetDomain("com.microsoft")
      .SinceVersion(1)
      .SetDoc("Fake function operator")
      .Input(0, "x", "Input tensor", "T1")
      .Output(0, "y", "Quantized output tensor", "T2")
      .Output(1, "y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.", "tensor(float)")
      .Output(2, "y_zero_point", "Output zero point. It's a scalar, which means a per-tensor/layer quantization.", "T2")
      .TypeConstraint("T1", {"tensor(float)"}, "Constrain 'x' to float tensor.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
      .FunctionBody([]() {
        auto nodes = ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes({// nodes: {outputs, op, inputs, attributes}
                                                                     ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Q_Min", 0.f),
                                                                     ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Q_Max", 255.f),
                                                                     {{"X_Min"}, "ReduceMin", {"x"}, {ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(0))}},
                                                                     {{"X_Max"}, "ReduceMax", {"x"}, {ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(0))}},
                                                                     {{"X_Range"}, "Fake_Sub", {"X_Max", "X_Min"}},
                                                                     {{"Scale"}, "Div", {"X_Range", "Q_Max"}},
                                                                     {{"Initial_ZeroPoint_FP"}, "Sub", {"Q_Min", "X_Min"}},
                                                                     {{"Clipped_ZeroPoint_FP"}, "Clip", {"Initial_ZeroPoint_FP", "Q_Min", "Q_Max"}},
                                                                     {{"Rounded_ZeroPoint_FP"}, "Round", {"Clipped_ZeroPoint_FP"}},
                                                                     {{"Zeropoint"}, "Cast", {"Initial_ZeroPoint_FP"}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t(2))}},
                                                                     {{"y_scale"}, "Identity", {"Scale"}},
                                                                     {{"y_zero_point"}, "Identity", {"Zeropoint"}},
                                                                     {{"y"}, "QuantizeLinear", {"x", "Scale", "Zeropoint"}}});
        for (auto& node : nodes) {
          if (node.op_type() == "Fake_Sub") {
            node.set_domain(kMSNchwcDomain);
          }
        }
        return nodes;
      }());

  return true;
}

static std::once_flag once;

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() {
    std::call_once(once, RegisterCustomSchemas);
    logger_ = DefaultLoggingManager().CreateLogger("GraphTest");
#if !defined(ORT_MINIMAL_BUILD)
    InferenceSession::ORTRegisterONNXOpsetSchema(0);
#endif
  }

  std::unique_ptr<logging::Logger> logger_;
};

static void SetTypeAndShape(TypeProto_Tensor* t, int type, const std::vector<int64_t>& shape) {
  t->set_elem_type(type);
  for (int64_t i : shape) {
    TensorShapeProto* s = t->mutable_shape();
    s->add_dim()->set_dim_value(i);
  }
}

static void ImportOpset(ModelProto& m, const char* domain, int64_t version) {
  OperatorSetIdProto* p = m.add_opset_import();
  p->set_domain(domain);
  p->set_version(version);
}

static void ConstructASimpleAddGraph(GraphProto& g, const char* domain) {
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_input() = "y";
  *node->add_output() = "sum";
  node->set_op_type("Add");
  if (domain != nullptr) {
    node->set_domain(domain);
  }
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* input2 = g.add_input();
  input2->set_name("y");
  SetTypeAndShape(input2->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
}

namespace sparse_details {
const std::vector<int64_t> shape = {3, 4, 5};
const std::vector<float> values = {13.f,
                                   17.f,
                                   19.f};

const std::vector<int64_t> indices = {9, 30, 50};  // Not to exceed 59
}  // namespace sparse_details

// To match a simple Add graph above
static void ConstructSparseTensor(const std::string& name,
                                  SparseTensorProto& sparse_proto) {
  const std::vector<int64_t>& shape = sparse_details::shape;
  const std::vector<float>& values = sparse_details::values;

  auto& m_values = *sparse_proto.mutable_values();
  m_values.set_name(name);
  m_values.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  *m_values.mutable_dims()->Add() = static_cast<int64_t>(values.size());
  std::string& raw_data = *m_values.mutable_raw_data();
  raw_data.resize(values.size() * sizeof(float));
  auto dest_span = gsl::make_span<float>(reinterpret_cast<float*>(&raw_data[0]), values.size());
  std::copy(values.cbegin(), values.cend(), dest_span.begin());

  const std::vector<int64_t>& indices = sparse_details::indices;  // Not to exceed 59
  auto& m_indicies = *sparse_proto.mutable_indices();
  m_indicies.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  *m_indicies.mutable_dims()->Add() = static_cast<int64_t>(indices.size());
  auto* m_indicies_data = m_indicies.mutable_int64_data();
  m_indicies_data->Resize(static_cast<int>(indices.size()), 0);
  std::copy(indices.cbegin(), indices.cend(), m_indicies_data->begin());

  auto& m_dims = *sparse_proto.mutable_dims();
  m_dims.Resize(static_cast<int>(shape.size()), 0);
  std::copy(shape.cbegin(), shape.cend(), m_dims.begin());
}

static void ValidateSparseTensorProto(const SparseTensorProto& proto) {
  // check values. We always generate float
  EXPECT_EQ(proto.values().data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  EXPECT_EQ(proto.values().raw_data().size() % sizeof(float), 0U);
  auto actual_values = gsl::make_span<const float>(reinterpret_cast<const float*>(proto.values().raw_data().data()),
                                                   proto.values().raw_data().size() / sizeof(float));
  // Can't use ContainerEq on float
  EXPECT_EQ(actual_values.size(), sparse_details::values.size());
  // std::equal() with a predicate is only in C++20
  auto actual_begin = actual_values.cbegin();
  const auto actual_end = actual_values.cend();
  auto expected_begin = sparse_details::values.cbegin();
  while (actual_begin != actual_end) {
    auto diff = *actual_begin - *expected_begin;
    EXPECT_TRUE(diff < std::numeric_limits<float>::epsilon()) << "Actual :" << *actual_begin << " does not match expected: " << *expected_begin;
    ++actual_begin;
    ++expected_begin;
  }
  // Check indices
  EXPECT_EQ(proto.indices().data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto expected_indices = gsl::make_span(sparse_details::indices);
  auto actual_indices = gsl::make_span<const int64_t>(proto.indices().int64_data().data(), proto.indices().int64_data_size());
  EXPECT_THAT(actual_indices, testing::ContainerEq(expected_indices));
  // check shape
  const auto& dims = proto.dims();
  auto actual_shape = gsl::make_span<const int64_t>(dims.data(), dims.size());
  auto expected_shape = gsl::make_span(sparse_details::shape);
  EXPECT_THAT(actual_shape, testing::ContainerEq(expected_shape));
}

TEST_F(GraphTest, SimpleAddWithoutDomain) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), nullptr);
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
}

TEST_F(GraphTest, SimpleAddDefaultDomain) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), "");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
}

TEST_F(GraphTest, SimpleAddFutureOpSet) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 9999);
  ConstructASimpleAddGraph(*m.mutable_graph(), "ai.onnx");
  std::shared_ptr<Model> model;
  Status st;
  ASSERT_FALSE((st = Model::Load(std::move(m), model, nullptr, *logger_)).IsOK());
}

TEST_F(GraphTest, SimpleAddONNXDomain) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), "ai.onnx");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
}

TEST_F(GraphTest, SimpleAddONNXDomain2) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "ai.onnx", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), "ai.onnx");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
}

TEST_F(GraphTest, SimpleAddWrongDomain) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), "AAAA");
  std::shared_ptr<Model> model;
  Status st;
  ASSERT_FALSE((st = Model::Load(std::move(m), model, nullptr, *logger_)).IsOK());
}

TEST_F(GraphTest, SimpleAddWrongDomain2) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "AAAA", 10);
  ConstructASimpleAddGraph(*m.mutable_graph(), "AAAA");
  std::shared_ptr<Model> model;
  Status st;
  ASSERT_FALSE((st = Model::Load(std::move(m), model, nullptr, *logger_)).IsOK());
}

TEST_F(GraphTest, SimpleUnique) {
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "", 11);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_output() = "sum";
  node->set_op_type("Unique");
  node->set_domain("");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
}

TEST_F(GraphTest, UnusedValueInfoSerializes) {
  ModelProto m;
  m.set_ir_version(4);
  ImportOpset(m, "", 11);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_output() = "sum";
  node->set_op_type("Unique");
  node->set_domain("");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  ValueInfoProto* unused = g.add_value_info();
  unused->set_name("unused");
  SetTypeAndShape(unused->mutable_type()->mutable_tensor_type(), 1, {123});
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
  model->MainGraph().SetGraphProtoSyncNeeded();
  EXPECT_TRUE(Model::Save(*model, "graph_with_unused_value_info.onnx").IsOK());
}

TEST_F(GraphTest, WrongOpset) {
  ModelProto m;
  m.set_ir_version(3);
  //No Op registered for Unique with domain_version of 1
  ImportOpset(m, "", 1);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_output() = "sum";
  node->set_op_type("Unique");
  node->set_domain("");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  std::shared_ptr<Model> model;
  Status st;
  ASSERT_FALSE((st = Model::Load(std::move(m), model, nullptr, *logger_)).IsOK());
}

TEST_F(GraphTest, ExtraInput) {
  ModelProto m;
  m.set_ir_version(3);
  //Node () has input size 2 not in range [min=1, max=1].
  ImportOpset(m, "", 11);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_input() = "y";
  *node->add_output() = "sum";
  node->set_op_type("Unique");
  node->set_domain("");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  std::shared_ptr<Model> model;
  Status st;
  ASSERT_FALSE((st = Model::Load(std::move(m), model, nullptr, *logger_)).IsOK());
}

TEST_F(GraphTest, LocalCustomRegistry) {
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {
      OpSchema().SetName("FakeUnique").Input(0, "X", "A N-D input tensor that is to be processed.", "T").Output(0, "Y", "desc", "T").TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.").SetDomain("FakeTestDomain")};
  ASSERT_TRUE(registry->RegisterOpSet(schema, "FakeTestDomain", 0, 1).IsOK());
  ModelProto m;
  m.set_ir_version(3);
  ImportOpset(m, "FakeTestDomain", 1);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_output() = "sum";
  node->set_op_type("FakeUnique");
  node->set_domain("FakeTestDomain");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  std::shared_ptr<Model> model;
  Status st;
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {registry};
  ASSERT_STATUS_OK(Model::Load(std::move(m), model, &regs, *logger_));
}

// Tests the case where function op and function body ops belong to different domains.
// Tests that such a model can be loaded successfully, function body initialization is
// successful and domain and verison mapping for each node is successful (by verifying
// op schema for each of the function body nodes can be found).
TEST_F(GraphTest, FunctionOpsetImportTest) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/function_opset_test.onnx"), model, {},
                               *logger_));
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  const auto& graph = model->MainGraph();
  for (const auto& node : graph.Nodes()) {
    const auto schema = schema_registry->GetSchema(node.OpType(), node.SinceVersion(), node.Domain());
    auto func_ptr = node.GetFunctionBody();
    if (func_ptr == nullptr) {
      // If Op Schema has function body then func_ptr cannot be nullptr
      // This is because we construct function body during graph resolve.
      // However in future if we move the function initialization in the graph partitioning
      // phase .i.e. Init function body only if none of EPs have a kernel matching the function op
      // then this check will not hold true and should be removed.
      ASSERT_TRUE(!schema->HasFunction() && !schema->HasContextDependentFunction());
      continue;
    }
    const auto& function_op_schema = func_ptr->OpSchema();
    ASSERT_TRUE(function_op_schema.domain() == node.Domain());

    const auto& domain_version_map = func_ptr->Body().DomainToVersionMap();
    // validate schema for each node in the function body can be found
    for (auto& n : func_ptr->Body().Nodes()) {
      auto it = domain_version_map.find(n.Domain());
      ASSERT_TRUE(it != domain_version_map.end());
      auto domain_version = it->second;
      const auto op_schema = schema_registry->GetSchema(n.OpType(), domain_version, n.Domain());
      ASSERT_TRUE(op_schema != nullptr);
    }
  }
}

TEST_F(GraphTest, LocalCustomRegistryWrongOpsetImportVersion) {
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {
      OpSchema().SetName("FakeUnique").Input(0, "X", "A N-D input tensor that is to be processed.", "T").Output(0, "Y", "desc", "T").TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.").SetDomain("FakeTestDomain")};
  ASSERT_TRUE(registry->RegisterOpSet(schema, "FakeTestDomain", 0, 1).IsOK());
  ModelProto m;
  m.set_ir_version(3);
  //Should be 1, but we put 11 herer so the model loading will fail
  ImportOpset(m, "FakeTestDomain", 11);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_output() = "sum";
  node->set_op_type("FakeUnique");
  node->set_domain("FakeTestDomain");
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {3, 4, 5});
  ValueInfoProto* output = g.add_output();
  output->set_name("sum");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {60});
  std::shared_ptr<Model> model;
  Status st;
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {registry};
  ASSERT_FALSE((st = Model::Load(std::move(m), model, &regs, *logger_)).IsOK());
}

TEST_F(GraphTest, ReverseDFS) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /* Case 1: A normal graph.
   *
   *                 SouceNode
   *                 /       \
   *  node_1 (Variable)      node_2 (Variable)    node_5 (Variable)
   *                 \       /                        |
   *                 node_3 (Add)                 node_6 (NoOp)
   *                     |                            |
   *                 node_4 (Add)  -------------------  <-- request stop
   *                     |
   *                  SinkNode
  */
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg);
  auto& node_1 = graph.AddNode("node_1", "Variable_DFS", "node 1", inputs, outputs);

  auto& input_arg2 = graph.GetOrCreateNodeArg("node_2_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg2);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "Variable_DFS", "node 2", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&output_arg2);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  auto& node_3 = graph.AddNode("node_3", "Add_DFS", "node 3", inputs, outputs);

  // side path
  inputs.clear();
  auto& input_arg5 = graph.GetOrCreateNodeArg("node_5_in_1", &tensor_int32);
  inputs.push_back(&input_arg5);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg5);
  graph.AddNode("node_5", "Variable_DFS", "node 5", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg5);
  auto& output_arg6 = graph.GetOrCreateNodeArg("node_6_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg6);
  graph.AddNode("node_6", "NoOp_DFS", "node 6", inputs, outputs);

  // merged
  inputs.clear();
  inputs.push_back(&output_arg3);
  inputs.push_back(&output_arg6);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  graph.AddNode("node_4", "Add_DFS", "node 4", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Remove/Add edge should not ask for resolving again.
  graph.RemoveEdge(node_1.Index(), node_3.Index(), 0, 0);
  graph.AddEdge(node_1.Index(), node_3.Index(), 0, 0);

  std::vector<const Node*> from;
  for (auto& node : graph.Nodes()) {
    if (node.OutputEdgesBegin() == node.OutputEdgesEnd()) {
      // This is a leaf node.
      from.push_back(&node);
    }
  }

  std::vector<std::string> enter_leave_sequence;

  struct NodeCompareName {
    bool operator()(const Node* n1, const Node* n2) const {
      return n1->Name() < n2->Name();
    }
  };

  graph.ReverseDFSFrom(
      from,
      [&enter_leave_sequence](const Node* n) {
        std::string s("enter:");
        s += n->Name();
        enter_leave_sequence.push_back(s);
      },
      [&enter_leave_sequence](const Node* n) {
        std::string s("leave:");
        s += n->Name();
        enter_leave_sequence.push_back(s);
      },
      NodeCompareName(),
      // don't traverse side path
      [](const Node* from, const Node* to) {
        return from->Name() == "node_4" && to->Name() == "node_6";
      });

  EXPECT_EQ(enter_leave_sequence.size(), 8u);
  EXPECT_EQ("enter:node_4", enter_leave_sequence.at(0));
  EXPECT_EQ("enter:node_3", enter_leave_sequence.at(1));
  EXPECT_EQ("enter:node_2", enter_leave_sequence.at(2));
  EXPECT_EQ("leave:node_2", enter_leave_sequence.at(3));
  EXPECT_EQ("enter:node_1", enter_leave_sequence.at(4));
  EXPECT_EQ("leave:node_1", enter_leave_sequence.at(5));
  EXPECT_EQ("leave:node_3", enter_leave_sequence.at(6));
  EXPECT_EQ("leave:node_4", enter_leave_sequence.at(7));
}

TEST_F(GraphTest, GraphConstruction_VerifyNoDuplicateName) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  EXPECT_EQ("graph_1", graph.Name());

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // INT32 vector.
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);

  // Case 1: Adding two nodes with same node name should fail.
  auto& node_with_dup_name = graph.AddNode("node_1", "Variable", "node 2", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ("This is an invalid model. Error: two nodes with same node name (node_1).", status.ErrorMessage());
  graph.RemoveNode(node_with_dup_name.Index());

  // Case 2: Adding two nodes with same output arg name should fail.
  graph.AddNode("node_2", "Variable", "node 2", inputs, outputs);
  status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool duplicate_error_found = status.ErrorMessage().find("Duplicate") != std::string::npos;
  EXPECT_TRUE(duplicate_error_found);
}

TEST_F(GraphTest, GraphConstruction_VerifyNodeAndOpMatch) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // INT32 vector.
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  // Case: Adding node referring to non-existing operator should fail.
  graph.AddNode("node_1", "OpNotExist", "node 1", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(0u, status.ErrorMessage().find_first_of("This is an invalid model. No Schema registered for OpNotExist"));
}

TEST_F(GraphTest, GraphConstruction_CheckIsAcyclic) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /* A normal graph.
   *                 SouceNode
   *                 /       \
   *    node_1 (Variable)  node_2 (Variable)
   *                 \       /
   *                 node_3 (Add)
   *                     |
   *                 node_4 (NoOp)
   *                     |
   *                  SinkNode
   */
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  std::unordered_map<std::string, std::pair<std::vector<NodeArg*>, std::vector<NodeArg*>>>
      expected_node_name_to_input_output_args;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg1);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg1);
  expected_node_name_to_input_output_args["node_1"] = {inputs, outputs};
  graph.AddNode("node_1", "Variable_Fake", "node 1", inputs, outputs);

  auto& input_arg2 = graph.GetOrCreateNodeArg("node_2_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg2);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  expected_node_name_to_input_output_args["node_2"] = {inputs, outputs};
  graph.AddNode("node_2", "Variable_Fake", "node 2", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg1);
  inputs.push_back(&output_arg2);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  expected_node_name_to_input_output_args["node_3"] = {inputs, outputs};
  graph.AddNode("node_3", "Add_Fake", "node 3", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg3);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  expected_node_name_to_input_output_args["node_4"] = {inputs, outputs};
  graph.AddNode("node_4", "NoOp_Fake", "node 4", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  EXPECT_TRUE(Model::Save(model, "graph_1.onnx").IsOK());
  std::shared_ptr<Model> model2;
  EXPECT_TRUE(Model::Load(ORT_TSTR("graph_1.onnx"), model2, nullptr, *logger_).IsOK());

  auto model_proto = model.ToProto();
  auto model_proto2 = model2->ToProto();
  bool equal_proto_1_and_2 = model_proto.SerializeAsString() == model_proto2.SerializeAsString();
  EXPECT_TRUE(equal_proto_1_and_2);

  // Load the model again to ensure that it's still the right thing.
  //EXPECT_EQ(Model::Load(model_proto2, &model2), Status::OK());
  model2.reset(new Model(model_proto2, nullptr, *logger_));
  Graph& graph2 = model2->MainGraph();
  for (auto& node : graph2.Nodes()) {
    auto node_name_to_input_output_iter = expected_node_name_to_input_output_args.find(node.Name());
    EXPECT_FALSE(node_name_to_input_output_iter == expected_node_name_to_input_output_args.end());

    EXPECT_EQ(node_name_to_input_output_iter->second.first.size(), node.InputDefs().size());
    for (size_t i = 0; i < node_name_to_input_output_iter->second.first.size(); ++i) {
      EXPECT_EQ(node_name_to_input_output_iter->second.first[i]->Name(), node.InputDefs()[i]->Name());
      EXPECT_EQ(node_name_to_input_output_iter->second.first[i]->Type(), node.InputDefs()[i]->Type());
    }

    EXPECT_EQ(node_name_to_input_output_iter->second.second.size(), node.OutputDefs().size());
    for (size_t i = 0; i < node_name_to_input_output_iter->second.second.size(); ++i) {
      EXPECT_EQ(node_name_to_input_output_iter->second.second[i]->Name(), node.OutputDefs()[i]->Name());
      EXPECT_EQ(node_name_to_input_output_iter->second.second[i]->Type(), node.OutputDefs()[i]->Type());
    }
  }
}

TEST_F(GraphTest, GraphConstruction_CheckInputNodeOrderMaintained) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  //    node_1 (Identity)  node_2 (Identity)
  //                |         |
  //    node_4 (Identity)  node_3 (Identity)   Cross inputs over so node_1 and node_2 would get swapped if we didn't
  //                 \       /                 maintain order.
  //                 node_5 (Merge)
  //                     |

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);

  auto& input_arg2 = graph.GetOrCreateNodeArg("node_2_in_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);

  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  inputs.push_back(&input_arg1);
  outputs.push_back(&output_arg1);
  graph.AddNode("node_1", "Identity_Fake", "node 1", inputs, outputs);

  inputs[0] = &input_arg2;
  outputs[0] = &output_arg2;
  graph.AddNode("node_2", "Identity_Fake", "node 2", inputs, outputs);

  inputs[0] = &output_arg2;
  outputs[0] = &output_arg3;
  graph.AddNode("node_3", "Identity_Fake", "node 3", inputs, outputs);

  inputs[0] = &output_arg1;
  outputs[0] = &output_arg4;
  graph.AddNode("node_4", "Identity_Fake", "node 4", inputs, outputs);

  inputs.resize(2);
  inputs[0] = &output_arg4;
  inputs[1] = &output_arg3;
  outputs[0] = &output_arg5;
  graph.AddNode("node_5", "Merge_Fake", "node 3", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);
  auto& topological_order = graph_viewer.GetNodesInTopologicalOrder();
  bool seen1 = false;
  bool seen2 = false;

  for (auto i : topological_order) {
    auto node = graph.GetNode(i);

    if (node->Name() == "node_1") {
      EXPECT_TRUE(!seen2) << "node_1 should remain before node_2 after the topological sort.";
      seen1 = true;
    } else if (node->Name() == "node_2") {
      EXPECT_TRUE(seen1) << "node_1 should be before node_2 after the topological sort.";
      seen2 = true;
    }
  }
}

TEST_F(GraphTest, GraphConstruction_PriorityBasedTopologicalSort_CompressDecompress) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                          |
                  node_0 (Identity)
                      /      \
        node_1 (Identity)   compress (pri = LOCAL_HIGH)
                    |         |
        node_4 (Identity)  decompress (pri = LOCAL_LOW)
                      \       /
                      node_5 (Merge)
                          |                   
  */

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg0 = graph.GetOrCreateNodeArg("node_0_in_1", &tensor_int32);
  auto& output_arg0 = graph.GetOrCreateNodeArg("node_0_out_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});

  auto& compress_node = graph.AddNode("compress", "Identity_Fake", "compress node", {&output_arg0}, {&output_arg2});
  compress_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));

  auto& decompress_node = graph.AddNode("decompress", "Identity_Fake", "decompress node", {&output_arg2}, {&output_arg3});
  decompress_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

  graph.AddNode("node_4", "Identity_Fake", "node 4", {&output_arg1}, {&output_arg4});
  graph.AddNode("node_5", "Merge_Fake", "node 3", {&output_arg4, &output_arg3}, {&output_arg5});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // PRIORITY_BASED order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
    const std::vector<std::string> expected_priority_based_order =
        {"node_0", "compress", "node_1", "node_4", "decompress", "node_5"};
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "Priority based execution order is wrong.";
    }
  }

  // TOPOLOGICAL order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);
    const std::vector<std::string> expected_topological_order = {
        "node_0", "node_1", "node_4", "compress", "decompress", "node_5"};
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_topological_order[i]) << "Priority based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_PriorityBasedTopologicalSort_CompressDecompress_Nested) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                                       |
                                node_0 (Identity)
                                /               \
                  node_1 (Identity)             compress_0 (n2, pri = LOCAL_HIGH)
                    /          \                    |
          node_4 (Identity)    compress_1 (n5)  decompress_0 (n3, pri = LOCAL_LOW)
                   |            |                   |
          node_7 (Identity)    decompress_1 (n6)    |
                    \           /                   |
                    node_8 (Merge)                  |
                           \                        /
                                 node_9 (Merge)                
                                      |                   
  */

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg0 = graph.GetOrCreateNodeArg("node_0_in_1", &tensor_int32);
  auto& output_arg0 = graph.GetOrCreateNodeArg("node_0_out_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);
  auto& output_arg6 = graph.GetOrCreateNodeArg("node_6_out_1", &tensor_int32);
  auto& output_arg7 = graph.GetOrCreateNodeArg("node_7_out_1", &tensor_int32);
  auto& output_arg8 = graph.GetOrCreateNodeArg("node_8_out_1", &tensor_int32);
  auto& output_arg9 = graph.GetOrCreateNodeArg("node_9_out_1", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});

  auto& compress_node0 = graph.AddNode("compress_0", "Identity_Fake", "compress node 0", {&output_arg0}, {&output_arg2});
  compress_node0.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));

  auto& decompress_node0 = graph.AddNode("decompress_0", "Identity_Fake", "decompress node 0", {&output_arg2}, {&output_arg3});
  decompress_node0.SetPriority(20);

  graph.AddNode("node_4", "Identity_Fake", "node 4", {&output_arg1}, {&output_arg4});

  auto& compress_node1 = graph.AddNode("compress_1", "Identity_Fake", "compress node 1", {&output_arg1}, {&output_arg5});
  compress_node1.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));

  auto& decompress_node1 = graph.AddNode("decompress_1", "Identity_Fake", "decompress node 1", {&output_arg5}, {&output_arg6});
  decompress_node1.SetPriority(10);  // lower number means high priority

  graph.AddNode("node_7", "Identity_Fake", "node 7", {&output_arg4}, {&output_arg7});
  graph.AddNode("node_8", "Merge_Fake", "node 8", {&output_arg7, &output_arg6}, {&output_arg8});
  graph.AddNode("node_9", "Merge_Fake", "node 9", {&output_arg8, &output_arg3}, {&output_arg9});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // PRIORITY_BASED order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
    const std::vector<std::string> expected_priority_based_order =
        {"node_0", "compress_0", "node_1", "compress_1", "node_4", "node_7", "decompress_1", "node_8", "decompress_0", "node_9"};

    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "Priority based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_PriorityBasedTopologicalSort_Recompute) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                         |
                  node_0 (Identity)
                     /       \
        node_1 (Identity)  recompute_node_1 (pri = LOCAL_LOW)
                    |         |
        node_4 (Identity)     |
                     \       /           
               node_1_grad (Merge)
                        |
  */

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg0 = graph.GetOrCreateNodeArg("node_0_in_1", &tensor_int32);
  auto& output_arg0 = graph.GetOrCreateNodeArg("node_0_out_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});

  auto& recompute_node = graph.AddNode("recompute_node_1", "Identity_Fake", "recompute node 1", {&output_arg0}, {&output_arg2});
  recompute_node.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

  graph.AddNode("node_4", "Identity_Fake", "node 4", {&output_arg1}, {&output_arg4});
  graph.AddNode("node_1_grad", "Merge_Fake", "node_1 gradient", {&output_arg4, &output_arg2}, {&output_arg5});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // PRIORITY_BASED order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
    const std::vector<std::string> expected_priority_based_order =
        {"node_0", "node_1", "node_4", "recompute_node_1", "node_1_grad"};
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "Priority based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_PriorityBasedTopologicalSort_MultiLayerRecompute) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                         |
                  node_0 (Identity)
                     /            \
            node_1 (Identity)       \
                    |        \        \
           node_2 (Identity)   \        \
                    |      \     \        \
        node_3 (Identity)   \      \        \
                    |    \   \       \        \
          loss (Identity) \   \        \        \
                    |     |    \         \        \
             1            |     |         \        \
               \         /      |          \        | 
                loss_grad  recom_node_3    |        |
                     \         /           |        |
                     node_3_grad      recom_node_2  |
                            \          /            |
                             node_2_grad       recom_node_1
                                   \           /
                                    node_1_grad
                                         |
  */

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // FW graph
  auto& input_arg0 = graph.GetOrCreateNodeArg("node_0_in", &tensor_int32);
  auto& output_arg0 = graph.GetOrCreateNodeArg("node_0_out", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out", &tensor_int32);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out", &tensor_int32);
  auto& output_loss = graph.GetOrCreateNodeArg("loss_out", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});
  graph.AddNode("node_2", "Identity_Fake", "node 2", {&output_arg1}, {&output_arg2});
  graph.AddNode("node_3", "Identity_Fake", "node 3", {&output_arg2}, {&output_arg3});
  graph.AddNode("loss", "Identity_Fake", "loss node", {&output_arg3}, {&output_loss});

  // Recompute graph
  auto& recomputed_arg3 = graph.GetOrCreateNodeArg("node_3_out_recomputed", &tensor_int32);
  auto& recomputed_arg2 = graph.GetOrCreateNodeArg("node_2_out_recomputed", &tensor_int32);
  auto& recomputed_arg1 = graph.GetOrCreateNodeArg("node_1_out_recomputed", &tensor_int32);

  auto& recompute_node3 = graph.AddNode("node_3_recompute", "Identity_Fake", "node 3 recompute", {&output_arg2}, {&recomputed_arg3});
  auto& recompute_node2 = graph.AddNode("node_2_recompute", "Identity_Fake", "node 2 recompute", {&output_arg1}, {&recomputed_arg2});
  auto& recompute_node1 = graph.AddNode("node_1_recompute", "Identity_Fake", "node 1 recompute", {&output_arg0}, {&recomputed_arg1});
  recompute_node3.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
  recompute_node2.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
  recompute_node1.SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));

  // BW Graph
  auto& gradient_start = graph.GetOrCreateNodeArg("gradient_start", &tensor_int32);
  auto& loss_grad_output = graph.GetOrCreateNodeArg("loss_grad_output", &tensor_int32);
  auto& node_3_grad_output = graph.GetOrCreateNodeArg("node_3_grad_output", &tensor_int32);
  auto& node_2_grad_output = graph.GetOrCreateNodeArg("node_2_grad_output", &tensor_int32);
  auto& node_1_grad_output = graph.GetOrCreateNodeArg("node_1_grad_output", &tensor_int32);

  graph.AddNode("loss_grad", "Merge_Fake", "loss gradient", {&gradient_start, &output_arg3}, {&loss_grad_output});
  graph.AddNode("node_3_grad", "Merge_Fake", "node 3 gradient", {&loss_grad_output, &recomputed_arg3}, {&node_3_grad_output});
  graph.AddNode("node_2_grad", "Merge_Fake", "node 2 gradient", {&node_3_grad_output, &recomputed_arg2}, {&node_2_grad_output});
  graph.AddNode("node_1_grad", "Merge_Fake", "node 1 gradient", {&node_2_grad_output, &recomputed_arg1}, {&node_1_grad_output});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // PRIORITY_BASED order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
    const std::vector<std::string> expected_priority_based_order = {
        "node_0",
        "node_1",
        "node_2",
        "node_3",
        "loss",
        "loss_grad",
        "node_3_recompute",
        "node_3_grad",
        "node_2_recompute",
        "node_2_grad",
        "node_1_recompute",
        "node_1_grad",
    };
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "Priority based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_CheckGraphInputOutputOrderMaintained) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  std::unordered_map<std::string, int> map;

  for (auto i = 0; i < 20; ++i) {
    map.insert({std::to_string(i), i});
  }

  /*               |         |
   *       b (Identity)  a (Identity)   values
   *                \   /
   *                  c (Merge)
   *                  |
   *                  d (Split)
   *                /   \
   *              1  ..  10
   */
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg_a = graph.GetOrCreateNodeArg("node_a_in_1", &tensor_int32);
  auto& output_arg_a = graph.GetOrCreateNodeArg("node_a_out_1", &tensor_int32);

  auto& input_arg_b = graph.GetOrCreateNodeArg("node_b_in_1", &tensor_int32);
  auto& output_arg_b = graph.GetOrCreateNodeArg("node_b_out_1", &tensor_int32);

  auto& output_arg_c = graph.GetOrCreateNodeArg("node_c_out_1", &tensor_int32);

  std::vector<NodeArg*> split_outputs;
  std::vector<const NodeArg*> graph_outputs;
  for (int i = 0; i < 10; ++i) {
    auto arg = &graph.GetOrCreateNodeArg("node_d_out_" + std::to_string(i + 1), &tensor_int32);
    split_outputs.push_back(arg);
    graph_outputs.push_back(arg);
  }
  std::reverse(graph_outputs.begin(), graph_outputs.end());
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  inputs.push_back(&input_arg_a);
  outputs.push_back(&output_arg_a);
  graph.AddNode("a", "Identity_Fake", "a", inputs, outputs);

  inputs.resize(2);
  inputs[0] = &output_arg_b;
  inputs[1] = &output_arg_a;
  outputs[0] = &output_arg_c;
  graph.AddNode("c", "Merge_Fake", "c", inputs, outputs);

  // deliberately add 'b' after 'c' to mix up the inputs as well
  inputs.resize(1);
  inputs[0] = &input_arg_b;
  outputs[0] = &output_arg_b;
  graph.AddNode("b", "Identity_Fake", "b", inputs, outputs);

  inputs[0] = &output_arg_c;
  graph.AddNode("d", "Split_Fake", "d", inputs, split_outputs);

  auto validate_inputs_outputs = [&graph_outputs](const Graph& graph) {
    auto inputs = graph.GetInputs();
    auto outputs = graph.GetOutputs();

    ASSERT_TRUE(inputs.size() == 2);

    EXPECT_TRUE(inputs[0]->Name() == "node_a_in_1");  // 'a' was added first
    EXPECT_TRUE(inputs[1]->Name() == "node_b_in_1");

    ASSERT_TRUE(outputs.size() == 10);
    for (int i = 0; i < 10; ++i) {
      EXPECT_TRUE(graph_outputs[i]->Name() == outputs[i]->Name());
    }
  };
  graph.SetInputs({&input_arg_a, &input_arg_b});
  graph.SetOutputs(graph_outputs);
  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  validate_inputs_outputs(graph);

  // serialize and reload so we check the loaded from proto path in SetGraphInputsOutputs
  auto proto = model.ToProto();
  std::string s1;
  //std::stringstream s1;
  model.ToProto().SerializeToString(&s1);

  ModelProto model_proto;
  //  const bool result = model_proto.ParseFromIstream(&s1);
  const bool result = model_proto.ParseFromString(s1);
  ASSERT_TRUE(result) << "Failed to load model from serialized protobuf";

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  auto x = onnxruntime::Model::Load(model_proto, p_tmp_model, nullptr, *logger_);

  auto& graph2 = p_tmp_model->MainGraph();
  status = graph2.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  validate_inputs_outputs(graph2);
}

// Validate that an unused initializer doesn't break graph loading/resolution
// and is removed as expected.
TEST_F(GraphTest, UnusedInitializerIsIgnored) {
  Model model("UnusedInitializerIsIgnored", false, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg_a = graph.GetOrCreateNodeArg("node_a_in_1", &tensor_int32);
  auto& output_arg_a = graph.GetOrCreateNodeArg("node_a_out_1", &tensor_int32);

  inputs.push_back(&input_arg_a);
  outputs.push_back(&output_arg_a);
  graph.AddNode("a", "Identity_Fake", "a", inputs, outputs);

  TensorProto initializer_tensor;
  initializer_tensor.set_name("unused");
  initializer_tensor.add_dims(1);
  initializer_tensor.add_float_data(1.f);
  initializer_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  graph.AddInitializedTensor(initializer_tensor);
  ASSERT_TRUE(graph.GetAllInitializedTensors().size() == 1);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_TRUE(graph.GetAllInitializedTensors().empty());

  // serialize and reload so we check the loaded from proto path in SetGraphInputsOutputs
  auto proto = model.ToProto();
  std::string s1;
  //std::stringstream s1;
  model.ToProto().SerializeToString(&s1);

  ModelProto model_proto;
  const bool result = model_proto.ParseFromString(s1);
  ASSERT_TRUE(result) << "Failed to load model from serialized protobuf";

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  auto x = onnxruntime::Model::Load(model_proto, p_tmp_model, nullptr, *logger_);

  auto& graph2 = p_tmp_model->MainGraph();
  status = graph2.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_TRUE(graph.GetAllInitializedTensors().empty());
}

TEST_F(GraphTest, UnusedSparseInitializerIsIgnored) {
  std::string s1;
  {
    Model model("UnusedSparseInitializerIsIgnored", false, *logger_);
    auto model_proto = model.ToProto();
    auto* m_graph = model_proto.mutable_graph();
    ConstructASimpleAddGraph(*m_graph, nullptr);
    auto* m_sparse_initializer = m_graph->add_sparse_initializer();
    ConstructSparseTensor("unused_sparse_initializer", *m_sparse_initializer);
    model_proto.SerializeToString(&s1);
  }

  ModelProto model_proto_1;
  const bool result = model_proto_1.ParseFromString(s1);
  ASSERT_TRUE(result) << "Failed to load model from serialized protobuf";
  ASSERT_EQ(model_proto_1.graph().initializer_size(), 0);
  ASSERT_EQ(model_proto_1.graph().sparse_initializer_size(), 1);

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  auto x = onnxruntime::Model::Load(model_proto_1, p_tmp_model, nullptr, *logger_);
  ASSERT_STATUS_OK(x);

  auto& graph2 = p_tmp_model->MainGraph();
  EXPECT_STATUS_OK(graph2.Resolve());
  // Because the sparse initializer was unused, it was also removed
  // from initializer as well as from sparse_initializer
  ASSERT_TRUE(graph2.GetAllInitializedTensors().empty());
  auto& graph_proto = graph2.ToGraphProto();
  ASSERT_TRUE(graph_proto.sparse_initializer().empty());
}

TEST_F(GraphTest, GraphConstruction_CheckIsNotAcyclic) {
  // A cyclic graph
  //                 SouceNode
  //                     |
  //             --> node_1 (Add)
  //            ^        |
  //            | <- node_2 (NoOp)

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();
  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  inputs.push_back(&input_arg1);
  inputs.push_back(&output_arg2);
  outputs.push_back(&output_arg1);
  graph.AddNode("node_1", "Add_Fake", "node 1", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg1);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "NoOp_Fake", "node 2", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ("This is an invalid model. Error: the graph is not acyclic.", status.ErrorMessage());
}

TEST_F(GraphTest, GraphConstruction_OnlyInitializer) {
  onnxruntime::Model model("graph", false, *logger_);
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.add_dims(1);
  weight.set_data_type(TensorProto_DataType_STRING);
  weight.add_string_data("test");
  weight.set_name("node_1_in_2");
  graph.AddInitializedTensor(weight);

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto& iii = graph.GetInputsIncludingInitializers();
  EXPECT_TRUE(iii.size() == 0);
}

TEST_F(GraphTest, GraphConstruction_TypeInference) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /* Case 1: A normal graph.
   *                         SourceNode
   *                   /         |         \
   *  node_1 (Variable)  node_2 (Variable)  node_3 (Variable)
   *                   \         |         / (it's all 3 nodes above outputs to the one input of node_4)
   *                        node_4 (Max)
   *                             |
   *                          SinkNode
  */
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Variable2_Fake", "node 1", inputs, outputs);

  inputs.clear();
  inputs.push_back(&input_arg);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "Variable2_Fake", "node 2", inputs, outputs);

  auto& input_arg3 = graph.GetOrCreateNodeArg("node_3_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg3);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  graph.AddNode("node_3", "Variable2_Fake", "node 3", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&output_arg2);
  inputs.push_back(&output_arg3);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  graph.AddNode("node_4", "Max_Fake", "node 4", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::unordered_set<std::string> expected_graph_inputs = {"node_1_in_1", "node_3_in_1"};
  EXPECT_EQ(2u, graph.GetInputs().size());
  for (auto& graph_input : graph.GetInputs()) {
    EXPECT_TRUE(expected_graph_inputs.find(graph_input->Name()) != expected_graph_inputs.end());
  }
  EXPECT_EQ(1u, graph.GetOutputs().size());
  EXPECT_EQ("node_4_out_1", graph.GetOutputs()[0]->Name());
  EXPECT_EQ(2u, graph.GetInputs().size());

  EXPECT_TRUE(Model::Save(model, "model_x.onnx").IsOK());
  std::shared_ptr<Model> loaded_model;
  EXPECT_TRUE(Model::Load(ORT_TSTR("model_x.onnx"), loaded_model, nullptr, *logger_).IsOK());
  EXPECT_EQ(2u, loaded_model->MainGraph().GetInputs().size());

  auto& graph_proto = graph.ToGraphProto();
  EXPECT_EQ(2, graph_proto.input_size());
  for (auto& graphProtoInput : graph_proto.input()) {
    EXPECT_TRUE(expected_graph_inputs.find(graphProtoInput.name()) != expected_graph_inputs.end());
  }
  EXPECT_EQ(1, graph_proto.output_size());
  EXPECT_EQ("node_4_out_1", graph_proto.output(0).name());
}

TEST_F(GraphTest, ShapeInferenceErrorHandling) {
  Model model("graph", false, *logger_);
  auto& graph = model.MainGraph();

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);

  graph.AddNode("node_1", "ShapeInferenceThrowsOp", "node 1", {&input_arg1}, {&output_arg1});

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Node (node_1) Op (ShapeInferenceThrowsOp) "
                                                        "[ShapeInferenceError] try harder"));
}

TEST_F(GraphTest, AddTensorAttribute) {
  OPERATOR_SCHEMA(__Constant)
      .SetDoc("Constant Op.")
      .Attr(kConstantValue, "constant value", AttrType::AttributeProto_AttributeType_TENSOR)
      .Output(0, "output_1", "docstr for output_1.", "tensor(int64)");
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  TensorShapeProto output_shape;
  output_shape.mutable_dim()->Add()->set_dim_value(1);
  output_shape.mutable_dim()->Add()->set_dim_value(3);
  *(output_type.mutable_tensor_type()->mutable_shape()) = output_shape;
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  auto& node_1 = graph.AddNode("node_1", "__Constant", "node 1.", inputs, outputs);
  TensorProto t;
  t.set_data_type(TensorProto_DataType_INT64);
  *(t.mutable_int64_data()->Add()) = 1;
  *(t.mutable_int64_data()->Add()) = 2;
  *(t.mutable_int64_data()->Add()) = 3;
  *(t.mutable_dims()->Add()) = 1;
  *(t.mutable_dims()->Add()) = 3;
  node_1.AddAttribute(kConstantValue, t);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void AddAttribute(onnxruntime::Node& p_node, const std::string& attr_name, int64_t attr_value) {
  AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(AttributeProto_AttributeType_INT);
  attr.set_i(attr_value);
  p_node.AddAttribute(attr_name, attr);
}

void AddAttribute(onnxruntime::Node& p_node, const std::string& attr_name, std::initializer_list<int64_t> attr_value) {
  AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(AttributeProto_AttributeType_INTS);
  for (auto v : attr_value) {
    attr.add_ints(v);
  }
  p_node.AddAttribute(attr_name, attr);
}

// Test that output type can be inferred for ops with a type-attribute
TEST_F(GraphTest, TypeAttribute) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", nullptr);
  outputs.push_back(&output_arg);
  auto& node_1 = graph.AddNode("node_1", "RandomNormal", "node 1.", inputs, outputs);
  AddAttribute(node_1, "dtype", TensorProto_DataType_FLOAT);
  AddAttribute(node_1, "shape", {2, 3});
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void CheckTensorEltType(const TypeProto* ptype, TensorProto_DataType elt_type) {
  EXPECT_NE(ptype, nullptr);
  EXPECT_TRUE(ptype->has_tensor_type());
  EXPECT_TRUE(ptype->tensor_type().has_elem_type());
  EXPECT_EQ(ptype->tensor_type().elem_type(), elt_type);
}

// Test that output type can be inferred for ops with variadic outputs
TEST_F(GraphTest, VariadicOutput) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();
  auto& X = graph.GetOrCreateNodeArg("X", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& Z = graph.GetOrCreateNodeArg("Z", nullptr);
  outputs.push_back(&Z);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  CheckTensorEltType(Y.TypeAsProto(), TensorProto_DataType_FLOAT);
  CheckTensorEltType(Z.TypeAsProto(), TensorProto_DataType_FLOAT);
}

// test that we prefer the graph input shape for a non-const initializer (initializer with matching graph input)
TEST_F(GraphTest, NonConstInitializer) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  TypeProto tensor_type_no_shape;
  tensor_type_no_shape.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  // tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& X = graph.GetOrCreateNodeArg("X", &tensor_type_no_shape);
  auto& Y = graph.GetOrCreateNodeArg("Y_Initializer", &tensor_type_no_shape);
  auto& Z = graph.GetOrCreateNodeArg("Z", nullptr);

  // 2 graph inputs, both without shapes
  graph.SetInputs({&X, &Y});

  // add initializer for the Y input with shape
  TensorProto t;
  t.set_data_type(TensorProto_DataType_FLOAT);
  t.add_float_data(0.1f);
  t.add_float_data(0.2f);
  t.add_dims(2);
  t.set_name("Y_Initializer");
  graph.AddInitializedTensor(t);

  graph.AddNode("node_1", "Add", "node 1.", {&X, &Y}, {&Z});

  auto resolve_and_validate = [](Graph& g) {
    auto status = g.Resolve();
    EXPECT_TRUE(status.IsOK()) << status;

    const auto* current_Y = g.GetNodeArg("Y_Initializer");
    const auto* current_Z = g.GetNodeArg("Z");

    // the graph input should still have no shape as we don't want to infer the shape from the initializer
    // as inputs have priority
    EXPECT_TRUE(current_Y != nullptr && current_Y->Shape() == nullptr);

    // and we should have type but no shape for Z after type/shape inferencing
    EXPECT_TRUE(current_Z != nullptr && current_Z->Type() == current_Y->Type());
    EXPECT_TRUE(current_Z->Shape() == nullptr);
  };

  resolve_and_validate(graph);

  // save and reload to validate same happens when graph is loaded from proto
  std::string s1;
  ModelProto model_proto;
  std::shared_ptr<onnxruntime::Model> p_model;
  ASSERT_TRUE(model.ToProto().SerializeToString(&s1));
  ASSERT_TRUE(model_proto.ParseFromString(s1));

  auto status = onnxruntime::Model::Load(model_proto, p_model, nullptr, *logger_);
  ASSERT_TRUE(status.IsOK()) << status;

  auto& graph2 = p_model->MainGraph();
  resolve_and_validate(graph2);
}

// Test that Graph::Resolve identifies name-duplication across initializer and node-output-arg
TEST_F(GraphTest, DuplicateName) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.set_data_type(TensorProto_DataType_FLOAT);
  weight.add_dims(1);
  weight.add_float_data(1.0f);
  weight.set_name("W");
  graph.AddInitializedTensor(weight);

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto& X = graph.GetOrCreateNodeArg("X", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& W = graph.GetOrCreateNodeArg("W", nullptr);
  outputs.push_back(&W);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool duplicate_error_found = status.ErrorMessage().find("Duplicate") != std::string::npos;
  EXPECT_TRUE(duplicate_error_found);
}

TEST_F(GraphTest, ReplaceInitializedTensor) {
  Model model{"GraphUpdateTest", false, *logger_};
  auto& graph = model.MainGraph();
  const std::string initializer_name = "initializer";

  ONNX_NAMESPACE::TensorProto original{};
  original.set_data_type(TensorProto_DataType_INT32);
  original.add_dims(2);
  original.add_int32_data(1);
  original.add_int32_data(2);
  original.set_name(initializer_name);

  graph.AddInitializedTensor(original);

  Status status;

  {
    ONNX_NAMESPACE::TensorProto bad_name = original;
    bad_name.set_name("invalid");

    status = graph.ReplaceInitializedTensor(bad_name);
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto bad_type = original;
    bad_type.set_data_type(TensorProto_DataType_FLOAT16);

    status = graph.ReplaceInitializedTensor(bad_type);
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto bad_dims = original;
    bad_dims.clear_dims();
    bad_dims.add_dims(2);
    bad_dims.add_dims(1);

    status = graph.ReplaceInitializedTensor(bad_dims);
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto valid_replacement = original;
    valid_replacement.clear_int32_data();
    valid_replacement.add_int32_data(3);
    valid_replacement.add_int32_data(4);

    status = graph.ReplaceInitializedTensor(valid_replacement);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    auto tensor_data_matches = [](const ONNX_NAMESPACE::TensorProto& a, const ONNX_NAMESPACE::TensorProto& b) {
      if (a.int32_data_size() != b.int32_data_size()) return false;
      for (int i = 0; i < a.int32_data_size(); ++i) {
        if (a.int32_data(i) != b.int32_data(i)) return false;
      }
      return true;
    };

    // check retrieved tensor
    const ONNX_NAMESPACE::TensorProto* result;
    ASSERT_TRUE(graph.GetInitializedTensor(initializer_name, result));
    ASSERT_TRUE(tensor_data_matches(*result, valid_replacement));

    // check GraphProto content
    const ONNX_NAMESPACE::GraphProto graph_proto = graph.ToGraphProto();
    ASSERT_EQ(graph_proto.initializer_size(), 1);
    ASSERT_TRUE(tensor_data_matches(graph_proto.initializer(0), valid_replacement));
  }
}

TEST_F(GraphTest, AddRemoveInitializerHandling) {
  Model m{"test_model", false, *logger_};
  Graph& graph = m.MainGraph();

  auto create_tensor_proto = [](const std::string& name, int32_t value) {
    ONNX_NAMESPACE::TensorProto init{};
    init.set_name(name);
    init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    init.add_dims(1);
    init.add_int32_data(value);

    return init;
  };

  auto init = create_tensor_proto("1", 1);
  auto init2 = create_tensor_proto("2", 2);

  // add both, remove the 1st (moves the second initializer into the first slot), and finally re-add the first
  graph.AddInitializedTensor(init);
  graph.AddInitializedTensor(init2);
  graph.RemoveInitializedTensor(init.name());
  graph.AddInitializedTensor(init);

  ASSERT_EQ(graph.GetAllInitializedTensors().size(), 2u);

  // check the values coming from name_to_initial_tensor_ are good;
  const TensorProto* i = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor(init.name(), i));
  ASSERT_TRUE(i->int32_data()[0] == 1);
  ASSERT_TRUE(graph.GetInitializedTensor(init2.name(), i));
  ASSERT_TRUE(i->int32_data()[0] == 2);

  // check the values in the GraphProto are also correct
  ONNX_NAMESPACE::GraphProto graph_proto_from_const_graph = static_cast<const Graph&>(graph).ToGraphProto();
  ONNX_NAMESPACE::GraphProto graph_proto_from_graph = graph.ToGraphProto();

  ASSERT_EQ(graph_proto_from_const_graph.initializer_size(), 2);
  ASSERT_EQ(graph_proto_from_graph.initializer_size(), 2);

  auto validate_proto = [&](const GraphProto& proto) {
    auto initializers = proto.initializer();
    // we expect '2' to be before '1' due to the remove moving the last initializer into the slot of the one being
    // removed in order to free memory and only move one entry
    EXPECT_EQ(initializers[0].name(), init2.name());
    EXPECT_EQ(initializers[0].int32_data()[0], 2);

    EXPECT_EQ(initializers[1].name(), init.name());
    EXPECT_EQ(initializers[1].int32_data()[0], 1);
  };

  validate_proto(graph_proto_from_const_graph);
  validate_proto(graph_proto_from_graph);

  // Call Graph::Resolve which should remove the initializers from the Graph instance and proto as they're unused.
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(graph.GetAllInitializedTensors().size(), 0u);

  ONNX_NAMESPACE::GraphProto graph_proto_from_resolved_graph = graph.ToGraphProto();
  auto num_initializers = graph_proto_from_resolved_graph.initializer_size();
  ASSERT_EQ(num_initializers, 0) << "Expected unused initializers to be removed from proto. "
                                 << num_initializers << " remain.";
}

TEST_F(GraphTest, SparseInitializerHandling) {
  const char* const input_initializer_name = "x";
  Model model("SparseInitializerHandling", false, *logger_);
  std::string s1;
  // Create model proto with sparse initializer
  {
    auto model_proto = model.ToProto();
    auto* m_graph = model_proto.mutable_graph();
    ConstructASimpleAddGraph(*m_graph, nullptr);
    auto* m_sparse_initializer = m_graph->add_sparse_initializer();
    ConstructSparseTensor(input_initializer_name, *m_sparse_initializer);
    model_proto.SerializeToString(&s1);
  }

  ModelProto model_proto_sparse;
  const bool result = model_proto_sparse.ParseFromString(s1);
  ASSERT_TRUE(result) << "Failed to load model from serialized protobuf";
  {
    auto& graph_proto = model_proto_sparse.graph();
    ASSERT_EQ(graph_proto.initializer_size(), 0);
    ASSERT_EQ(graph_proto.sparse_initializer_size(), 1);
    ValidateSparseTensorProto(graph_proto.sparse_initializer().at(0));
  }

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  auto x = onnxruntime::Model::Load(model_proto_sparse, p_tmp_model, nullptr, *logger_);

  auto& graph2 = p_tmp_model->MainGraph();
  EXPECT_STATUS_OK(graph2.Resolve());
  // Sparse initializer got converted to dense and appears on the list of initializers
  ASSERT_EQ(graph2.GetAllInitializedTensors().size(), 1U);
  ASSERT_EQ(graph2.GetAllInitializedTensors().cbegin()->first.compare(input_initializer_name), 0);

  auto& graph_proto = graph2.ToGraphProto();
  // Got propagated to initializers list
  ASSERT_EQ(graph_proto.initializer_size(), 1);
  ASSERT_EQ(graph_proto.initializer().at(0).name().compare(input_initializer_name), 0);
  // Got removed from sparse initializer list
  ASSERT_EQ(graph_proto.sparse_initializer_size(), 0);

  {
    // Check that Model::ToProto() does not return sparse among the normal initializers
    // but reconstitutes sparse initializer from dense. Thus, here we have dense initializer list empty
    // but it appears to be in the sparse.
    auto model_proto_get = p_tmp_model->ToProto();
    ASSERT_EQ(model_proto_get.graph().initializer_size(), 0);
    ASSERT_EQ(model_proto_get.graph().sparse_initializer_size(), 1);
    ValidateSparseTensorProto(model_proto_get.graph().sparse_initializer().at(0));
  }
}

TEST_F(GraphTest, SetInputsAndSetOutputs_NewInputAndOutput) {
  std::shared_ptr<Model> model;
  {
    ModelProto m;
    m.set_ir_version(4);
    ImportOpset(m, "", 10);
    ConstructASimpleAddGraph(*m.mutable_graph(), nullptr);
    ASSERT_STATUS_OK(Model::Load(std::move(m), model, nullptr, *logger_));
  }

  // starting from:
  //   x + y = sum
  // modify to:
  //   (x + y) + z = sum_with_z
  // set z as an additional input
  // set sum_with_z as an additional output

  Graph& graph = model->MainGraph();
  TypeProto type_proto{};
  SetTypeAndShape(type_proto.mutable_tensor_type(), 1, {3, 4, 5});
  auto* sum = graph.GetNodeArg("sum");
  auto* z = &graph.GetOrCreateNodeArg("z", &type_proto);
  auto* sum_with_z = &graph.GetOrCreateNodeArg("sum_with_z", &type_proto);

  graph.AddNode("add_z", "Add", "add z to sum", {sum, z}, {sum_with_z});

  auto inputs = graph.GetInputsIncludingInitializers();
  inputs.push_back(z);
  graph.SetInputs(inputs);

  auto outputs = graph.GetOutputs();
  outputs.push_back(sum_with_z);
  graph.SetOutputs(outputs);

  ASSERT_STATUS_OK(graph.Resolve());

  inputs = graph.GetInputsIncludingInitializers();
  ASSERT_TRUE(std::find(inputs.begin(), inputs.end(), z) != inputs.end()) << "expected new input z";

  outputs = graph.GetOutputs();
  ASSERT_TRUE(std::find(outputs.begin(), outputs.end(), sum_with_z) != outputs.end())
      << "expected new output sum_with_z";
}

TEST_F(GraphTest, LoadModelMissingInput) {
  ModelProto m;
  m.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  ImportOpset(m, "", 13);
  GraphProto& g = *m.mutable_graph();
  NodeProto* node = g.add_node();
  *node->add_input() = "x";
  *node->add_input() = "y";
  *node->add_output() = "z";
  node->set_op_type("Reshape");
  node->set_domain("");

  // add 'x' as a graph input but not 'y'
  ValueInfoProto* input1 = g.add_input();
  input1->set_name("x");
  SetTypeAndShape(input1->mutable_type()->mutable_tensor_type(), 1, {4});
  ValueInfoProto* output = g.add_output();
  output->set_name("z");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), 1, {2, 2});

  std::shared_ptr<Model> model;
  Status st = Model::Load(std::move(m), model, nullptr, *logger_);
  ASSERT_FALSE(st.IsOK());
  ASSERT_THAT(st.ErrorMessage(), testing::HasSubstr("Invalid model. Node input 'y' is not a graph input, "
                                                    "initializer, or output of a previous node."));
}

// if an initializer is backing an optional graph input, it can't be removed even if unused in the graph.
TEST_F(GraphTest, DontRemoveUnusedInitializerWithGraphInput) {
  const std::string unused_initializer_name("truncation:0");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/unused_initializer.onnx"), model, nullptr, *logger_));

  auto& graph = model->MainGraph();
  const auto& inputs_including_initializers = graph.GetInputsIncludingInitializers();
  auto j = std::find_if(inputs_including_initializers.cbegin(), inputs_including_initializers.cend(),
                        [&unused_initializer_name](const NodeArg* input) {
                          return input->Name() == unused_initializer_name;
                        });

  ASSERT_NE(j, inputs_including_initializers.cend()) << "Unused initializer was incorrectly removed.";
}
}  // namespace test
}  // namespace onnxruntime
