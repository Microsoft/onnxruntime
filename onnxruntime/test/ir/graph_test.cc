// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <chrono>
#include <fstream>
#include "core/common/inlined_containers.h"
#include "core/common/span_utils.h"
#include "core/flatbuffers/ort_format_version.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_flatbuffers_utils.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/model_helpers.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/op.h"
#include "core/graph/ort_format_load_options.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/temp_dir.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "onnx/defs/function.h"
#include "core/graph/function_impl.h"
#include "test/unittest_util/framework_test_utils.h"

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

  OPERATOR_SCHEMA(ShapeInferenceInputDataOutOfBoundsOp)
      .SetDoc("Access input data past the available inputs.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        ORT_ENFORCE(ctx.getInputData(ctx.getNumInputs()) == nullptr);
        propagateShapeAndTypeFromFirstInput(ctx);
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
      .FunctionBody((const std::vector<NodeProto>&)([]() {
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
        return nodes; }()),
                    (const std::vector<OperatorSetIdProto>&)([]() {
                      std::vector<OperatorSetIdProto> operator_sets(2);
                      auto& onnx_opset = operator_sets[0];
                      onnx_opset.set_domain("");
                      onnx_opset.set_version(11);

                      auto& test_opset = operator_sets[1];
                      test_opset.set_domain(kMSNchwcDomain);
                      test_opset.set_version(1);

                      return operator_sets;
                    }()));

  return true;
}

static std::once_flag once;

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() {
    std::call_once(once, RegisterCustomSchemas);
    logger_ = DefaultLoggingManager().CreateLogger("GraphTest");
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

#if !defined(DISABLE_SPARSE_TENSORS)
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
  auto actual_begin = actual_values.begin();
  const auto actual_end = actual_values.end();
  auto expected_begin = sparse_details::values.cbegin();
  while (actual_begin != actual_end) {
    auto diff = *actual_begin - *expected_begin;
    EXPECT_TRUE(diff < std::numeric_limits<float>::epsilon()) << "Actual :" << *actual_begin << " does not match expected: " << *expected_begin;
    ++actual_begin;
    ++expected_begin;
  }
  // Check indices
  const auto& indices = proto.indices();
  auto expected_indices = gsl::make_span(sparse_details::indices);
  SparseIndicesChecker(indices, expected_indices);
  // check shape
  const auto& dims = proto.dims();
  auto actual_shape = gsl::make_span<const int64_t>(dims.data(), dims.size());
  auto expected_shape = gsl::make_span(sparse_details::shape);
  EXPECT_THAT(actual_shape, testing::ContainerEq(expected_shape));
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

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

TEST_F(GraphTest, ModelEditorPreservesTensorElementTypes) {
  const std::pair<ONNXTensorElementDataType, int> element_types[] = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, TensorProto_DataType_FLOAT},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1, TensorProto_DataType_FLOAT4E2M1},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0, TensorProto_DataType_FLOAT8E8M0},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2, TensorProto_DataType_UINT2},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2, TensorProto_DataType_INT2},
  };

  for (const auto& [ort_type, proto_type] : element_types) {
    SCOPED_TRACE(proto_type);
    ModelEditorGraph editor_graph;
    const TensorShape shape{1};
    auto make_value_info = [&]() {
      auto value_info = std::make_unique<ModelEditorValueInfo>();
      value_info->name = "value";
      value_info->type_info = std::make_unique<OrtTypeInfo>(
          ONNX_TYPE_TENSOR, OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(ort_type, &shape, nullptr));
      return std::unique_ptr<ModelEditorValueInfo, OrtValueInfoDeleter>(value_info.release());
    };
    editor_graph.inputs.push_back(make_value_info());
    editor_graph.outputs.push_back(make_value_info());

    Model model("model_editor_element_types", false, *logger_);
    std::unique_ptr<Graph> graph;
    ASSERT_STATUS_OK(Graph::LoadFromModelEditorApiModel(editor_graph, model, model.MainGraph().DomainToVersionMap(),
                                                        nullptr, false, *logger_, graph));
    const auto& graph_proto = graph->ToGraphProto();
    ASSERT_EQ(graph_proto.input_size(), 1);
    ASSERT_EQ(graph_proto.output_size(), 1);
    EXPECT_EQ(graph_proto.input(0).type().tensor_type().elem_type(), proto_type);
    EXPECT_EQ(graph_proto.output(0).type().tensor_type().elem_type(), proto_type);
  }
}

TEST_F(GraphTest, ModelEditorFloat6ExternalInitializerSavesToOrtFormat) {
  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    constexpr int64_t count = 132;
    std::vector<uint8_t> bits(count);
    for (size_t i = 0; i < bits.size(); ++i) {
      bits[i] = static_cast<uint8_t>((i * 7 + 1) % 64);
    }

    auto editor_graph = std::make_unique<ModelEditorGraph>();
    auto value_info = std::make_unique<ModelEditorValueInfo>();
    value_info->name = "float6";
    const TensorShape shape{count};
    const auto ort_type = static_cast<ONNXTensorElementDataType>(data_type);
    value_info->type_info = std::make_unique<OrtTypeInfo>(
        ONNX_TYPE_TENSOR, OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(ort_type, &shape, nullptr));
    editor_graph->outputs.emplace_back(value_info.release());

    auto value = std::unique_ptr<OrtValue, OrtValueDeleter>(new OrtValue());
    const auto* type = DataTypeImpl::TensorTypeFromONNXEnum(data_type)->GetElementType();
    Tensor::InitOrtValue(type, shape, bits.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), *value);
    editor_graph->external_initializers.emplace("float6", std::move(value));

    OrtModel editor_model;
    editor_model.graph.reset(editor_graph.release());
    editor_model.domain_to_version[kOnnxDomain] = 21;
    std::unique_ptr<Model> model;
    ASSERT_STATUS_OK(Model::LoadFromModelEditorApiModel(editor_model, nullptr, ModelOptions{}, *logger_, model));
    ASSERT_STATUS_OK(model->MainGraph().Resolve());

    GraphProto exported_graph;
    GraphViewerToProto(GraphViewer(model->MainGraph()), exported_graph, true, false);
    ASSERT_EQ(exported_graph.initializer_size(), 1);
    const auto& exported_initializer = exported_graph.initializer(0);
    ASSERT_EQ(exported_initializer.raw_data().size(), count * 6 / 8);
    std::vector<uint8_t> exported_bits;
    ASSERT_STATUS_OK(utils::UnpackInitializerData(exported_initializer, exported_bits));
    EXPECT_EQ(exported_bits, bits);

    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<fbs::Model> offset;
    ASSERT_STATUS_OK(model->SaveToOrtFormat(builder, offset));
    builder.Finish(offset);

    const auto* saved_graph = flatbuffers::GetRoot<fbs::Model>(builder.GetBufferPointer())->graph();
    ASSERT_NE(saved_graph, nullptr);
    ASSERT_NE(saved_graph->initializers(), nullptr);
    ASSERT_EQ(saved_graph->initializers()->size(), 1U);
    const auto* saved_initializer = saved_graph->initializers()->Get(0);
    ASSERT_NE(saved_initializer->raw_data(), nullptr);
    EXPECT_EQ(saved_initializer->raw_data()->size(), count * 6 / 8);
    TensorProto loaded;
    OrtFormatLoadOptions options;
    ASSERT_STATUS_OK(fbs::utils::LoadInitializerOrtFormat(*saved_initializer, loaded, options));
    std::vector<uint8_t> decoded;
    ASSERT_STATUS_OK(utils::UnpackInitializerData(loaded, decoded));
    EXPECT_EQ(decoded, bits);
  }
}

TEST_F(GraphTest, GraphViewerExportsFileBackedFloat6Initializer) {
  TemporaryDirectory temp_dir{ORT_TSTR("float6_graph_export")};
  const auto model_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("model.onnx");
  const auto data_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("float6.bin");
  constexpr int64_t count = 132;
  std::string packed_data;
  std::vector<uint8_t> expected;
  for (int64_t i = 0; i < count / 4; ++i) {
    packed_data.append("\x01\x42\xc3", 3);
    expected.insert(expected.end(), {1, 8, 52, 48});
  }
  {
    std::ofstream output(data_path, std::ios::binary);
    output.write(packed_data.data(), packed_data.size());
  }

  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    ModelProto source;
    source.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    source.add_opset_import()->set_version(21);
    auto* graph = source.mutable_graph();
    graph->set_name("float6_export");
    auto* output = graph->add_output();
    output->set_name("float6");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);
    output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(count);
    auto* initializer = graph->add_initializer();
    initializer->set_name("float6");
    initializer->set_data_type(data_type);
    initializer->add_dims(count);
    ExternalDataInfo::SetExternalLocationToProto("float6.bin", 0, packed_data.size(), *initializer);

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(std::move(source), model_path.native(), model, nullptr, *logger_));
    GraphProto exported_graph;
    GraphViewerToProto(GraphViewer(model->MainGraph()), exported_graph, true, false);
    ASSERT_EQ(exported_graph.initializer_size(), 1);
    const auto& exported_initializer = exported_graph.initializer(0);
    EXPECT_TRUE(utils::HasExternalData(exported_initializer));
    TensorProto inlined;
    ASSERT_STATUS_OK(
        utils::TensorProtoWithExternalDataToTensorProto(exported_initializer, data_path.parent_path(), inlined));
    EXPECT_EQ(inlined.raw_data(), packed_data);
    std::vector<uint8_t> decoded;
    ASSERT_STATUS_OK(utils::UnpackInitializerData(inlined, decoded));
    EXPECT_EQ(decoded, expected);
  }
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
  EXPECT_TRUE(Model::Save(*model, ORT_TSTR("graph_with_unused_value_info.onnx")).IsOK());
}

TEST_F(GraphTest, WrongOpset) {
  ModelProto m;
  m.set_ir_version(3);
  // No Op registered for Unique with domain_version of 1
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
  // Node () has input size 2 not in range [min=1, max=1].
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
// successful and domain and version mapping for each node is successful (by verifying
// op schema for each of the function body nodes can be found).
TEST_F(GraphTest, FunctionOpsetImportTest) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/function_opset_test.onnx"), model, {},
                               *logger_));
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  const auto& graph = model->MainGraph();
  for (const auto& node : graph.Nodes()) {
    auto func_ptr = node.GetFunctionBody();
    if (func_ptr == nullptr) {
      // If Op Schema has function body then func_ptr cannot be nullptr
      // This is because we construct function body during graph resolve.
      // However in future if we move the function initialization in the graph partitioning
      // phase .i.e. Init function body only if none of EPs have a kernel matching the function op
      // then this check will not hold true and should be removed.

      // We delay the function instantiate until partition the graph
      // this check is no longer valid anymore.
      /*ASSERT_TRUE(!schema->HasFunction() && !schema->HasContextDependentFunction());*/
      continue;
    }

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
  // Should be 1, but we put 11 herer so the model loading will fail
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

  EXPECT_TRUE(Model::Save(model, ORT_TSTR("graph_1.onnx")).IsOK());
  std::shared_ptr<Model> model2;
  EXPECT_TRUE(Model::Load(ORT_TSTR("graph_1.onnx"), model2, nullptr, *logger_).IsOK());

  auto model_proto = model.ToProto();
  auto model_proto2 = model2->ToProto();
  bool equal_proto_1_and_2 = model_proto.SerializeAsString() == model_proto2.SerializeAsString();
  EXPECT_TRUE(equal_proto_1_and_2);

  // Load the model again to ensure that it's still the right thing.
  // EXPECT_EQ(Model::Load(model_proto2, &model2), Status::OK());
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

  inputs = {&output_arg4, &output_arg3};
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

  inputs = {&output_arg_b, &output_arg_a};
  outputs[0] = &output_arg_c;
  graph.AddNode("c", "Merge_Fake", "c", inputs, outputs);

  // deliberately add 'b' after 'c' to mix up the inputs as well
  inputs = {&input_arg_b};
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
  // std::stringstream s1;
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
// Validate unused NodeArgs are removed as expected
TEST_F(GraphTest, UnusedInitializerAndNodeArgsAreIgnored) {
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
  const std::string unused_initializer_name = "unused_initializer";
  initializer_tensor.set_name(unused_initializer_name);
  initializer_tensor.add_dims(1);
  initializer_tensor.add_float_data(1.f);
  initializer_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  graph.AddInitializedTensor(initializer_tensor);
  ASSERT_TRUE(graph.GetAllInitializedTensors().size() == 1);
  ASSERT_NE(nullptr, graph.GetNodeArg(unused_initializer_name));

  // Add unused NodeArgs
  const std::string unused_node_arg_name = graph.GenerateNodeArgName("unused_node_arg");
  ASSERT_EQ(nullptr, graph.GetNodeArg(unused_node_arg_name));
  graph.GetOrCreateNodeArg(unused_node_arg_name, nullptr);
  ASSERT_NE(nullptr, graph.GetNodeArg(unused_node_arg_name));

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_TRUE(graph.GetAllInitializedTensors().empty());
  ASSERT_EQ(nullptr, graph.GetNodeArg(unused_node_arg_name));

  // Verify NodeArg from the unused initializer is deleted as well
  // TODO, enable this when we can remove unused NodeArgs with type
  // See Graph::CleanUnusedInitializersAndNodeArgs
  // ASSERT_EQ(nullptr, graph.GetNodeArg(unused_initializer_name));

  // serialize and reload so we check the loaded from proto path in SetGraphInputsOutputs
  auto proto = model.ToProto();
  std::string s1;
  // std::stringstream s1;
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
  ASSERT_EQ(nullptr, graph.GetNodeArg(unused_node_arg_name));
  // TODO, enable this when we can remove unused NodeArgs with type
  // See Graph::CleanUnusedInitializersAndNodeArgs
  // ASSERT_EQ(nullptr, graph.GetNodeArg(unused_initializer_name));
}

#if !defined(DISABLE_SPARSE_TENSORS)
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

// Regression test for issue #28617: a SparseTensorProto loaded from a model protobuf must not
// be allowed to carry an ORT in-memory address marker on its values or indices sub-tensors.
// Those markers are an ORT-internal mechanism for trusted in-memory buffers (.ort flatbuffer
// load). Accepting them on a crafted .onnx protobuf would let the model make ORT dereference
// an attacker-supplied pointer during sparse-to-dense conversion.
static void RunRejectInMemoryMarkerOnSparseInitializerTest(bool marker_on_indices,
                                                           const onnxruntime::logging::Logger& logger) {
  Model model("RejectInMemoryMarkerOnSparseInitializer", false, logger);
  auto model_proto = model.ToProto();
  auto* m_graph = model_proto.mutable_graph();
  ConstructASimpleAddGraph(*m_graph, nullptr);

  auto* m_sparse_initializer = m_graph->add_sparse_initializer();
  ConstructSparseTensor("in_memory_marker_sparse", *m_sparse_initializer);

  // Overwrite either values or indices to declare external data pointing at an in-memory marker.
  // Allocate a real backing buffer so even an accidental dereference of "offset" stays in-process.
  static std::vector<uint8_t> backing(64, 0);
  auto* sub = marker_on_indices ? m_sparse_initializer->mutable_indices()
                                : m_sparse_initializer->mutable_values();
  sub->clear_raw_data();
  sub->clear_int64_data();
  sub->clear_float_data();
  sub->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = sub->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = sub->add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(reinterpret_cast<intptr_t>(backing.data())));
  auto* len = sub->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(backing.size()));

  std::string s1;
  model_proto.SerializeToString(&s1);

  ModelProto model_proto_1;
  ASSERT_TRUE(model_proto_1.ParseFromString(s1));

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  // The Graph ctor must reject the marker — Model::Load is expected to return a non-OK status
  // (Graph ctor's ORT_THROW is caught at the C++/Status boundary).
  ORT_TRY {
    auto status = onnxruntime::Model::Load(model_proto_1, p_tmp_model, nullptr, logger);
    EXPECT_FALSE(status.IsOK()) << "Loading a model with an in-memory marker on a sparse "
                                << (marker_on_indices ? "indices" : "values")
                                << " sub-tensor must fail.";
    if (!status.IsOK()) {
      EXPECT_THAT(status.ErrorMessage(),
                  ::testing::HasSubstr("in-memory address marker"));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      EXPECT_THAT(std::string(ex.what()),
                  ::testing::HasSubstr("in-memory address marker"));
    });
  }
}

TEST_F(GraphTest, RejectInMemoryMarkerOnSparseInitializerValues) {
  RunRejectInMemoryMarkerOnSparseInitializerTest(/*marker_on_indices=*/false, *logger_);
}

TEST_F(GraphTest, RejectInMemoryMarkerOnSparseInitializerIndices) {
  RunRejectInMemoryMarkerOnSparseInitializerTest(/*marker_on_indices=*/true, *logger_);
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Regression test: ORT in-memory address markers are an in-process sentinel only; they must never
// appear in a dense initializer deserialized from an .onnx protobuf. The Graph ctor must reject
// such a model.
TEST_F(GraphTest, RejectInMemoryMarkerOnDenseInitializer) {
  Model model("RejectInMemoryMarkerOnDenseInitializer", false, *logger_);
  auto model_proto = model.ToProto();
  auto* m_graph = model_proto.mutable_graph();
  ConstructASimpleAddGraph(*m_graph, nullptr);

  static std::vector<uint8_t> backing(64, 0);

  auto* init = m_graph->add_initializer();
  init->set_name("in_memory_marker_dense");
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  init->add_dims(static_cast<int64_t>(backing.size() / sizeof(float)));
  init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = init->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = init->add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(reinterpret_cast<intptr_t>(backing.data())));
  auto* len = init->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(backing.size()));

  std::string s1;
  model_proto.SerializeToString(&s1);

  ModelProto model_proto_1;
  ASSERT_TRUE(model_proto_1.ParseFromString(s1));

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  ORT_TRY {
    auto status = onnxruntime::Model::Load(model_proto_1, p_tmp_model, nullptr, *logger_);
    EXPECT_FALSE(status.IsOK()) << "Loading a model with an in-memory marker on a dense initializer must fail.";
    if (!status.IsOK()) {
      EXPECT_THAT(status.ErrorMessage(),
                  ::testing::HasSubstr("in-memory address marker"));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      EXPECT_THAT(std::string(ex.what()),
                  ::testing::HasSubstr("in-memory address marker"));
    });
  }
}

// Regression test: a Constant node with a dense tensor attribute carrying an ORT in-memory address
// marker must be rejected during model load. This is the attack vector described in the MSRC report
// where an attacker crafts a Constant node to make ORT dereference an attacker-supplied pointer.
TEST_F(GraphTest, RejectInMemoryMarkerOnConstantNodeTensorAttribute) {
  Model model("RejectInMemoryMarkerOnConstantNode", false, *logger_);
  auto model_proto = model.ToProto();
  auto* m_graph = model_proto.mutable_graph();

  // Build a minimal graph: Constant -> Identity -> output
  static std::vector<uint8_t> backing(16, 0);

  auto* const_node = m_graph->add_node();
  const_node->set_op_type("Constant");
  const_node->set_name("malicious_constant");
  const_node->add_output("c");

  auto* attr = const_node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* t = attr->mutable_t();
  t->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  t->add_dims(4);
  t->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = t->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = t->add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(reinterpret_cast<intptr_t>(backing.data())));
  auto* len = t->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(backing.size()));

  auto* identity_node = m_graph->add_node();
  identity_node->set_op_type("Identity");
  identity_node->set_name("identity");
  identity_node->add_input("c");
  identity_node->add_output("output");

  auto* output = m_graph->add_output();
  output->set_name("output");
  auto* type_proto = output->mutable_type()->mutable_tensor_type();
  type_proto->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  type_proto->mutable_shape()->add_dim()->set_dim_value(4);

  std::string serialized;
  model_proto.SerializeToString(&serialized);

  ModelProto model_proto_roundtrip;
  ASSERT_TRUE(model_proto_roundtrip.ParseFromString(serialized));

  std::shared_ptr<onnxruntime::Model> p_tmp_model;
  ORT_TRY {
    auto status = onnxruntime::Model::Load(model_proto_roundtrip, p_tmp_model, nullptr, *logger_);
    EXPECT_FALSE(status.IsOK()) << "Loading a model with an in-memory marker on a Constant node tensor attribute must fail.";
    if (!status.IsOK()) {
      EXPECT_THAT(status.ErrorMessage(),
                  ::testing::HasSubstr("in-memory address marker"));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      EXPECT_THAT(std::string(ex.what()),
                  ::testing::HasSubstr("in-memory address marker"));
    });
  }
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

  EXPECT_TRUE(Model::Save(model, ORT_TSTR("model_x.onnx")).IsOK());
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

  EXPECT_STATUS_NOT_OK_AND_HAS_SUBSTR(graph.Resolve(),
                                      "Node (node_1) Op (ShapeInferenceThrowsOp) [ShapeInferenceError] try harder");
}

TEST_F(GraphTest, ShapeInferenceInputDataOutOfBounds) {
  Model model("graph", false, *logger_);
  auto& graph = model.MainGraph();

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input = graph.GetOrCreateNodeArg("input", &tensor_int32);
  auto& output = graph.GetOrCreateNodeArg("output", nullptr);
  graph.AddNode("node", "ShapeInferenceInputDataOutOfBoundsOp", "", {&input}, {&output});

  ASSERT_STATUS_OK(graph.Resolve());
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
  node_1.AddAttribute(kConstantValue, std::move(t));
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void AddAttribute(onnxruntime::Node& p_node, const std::string& attr_name, int64_t attr_value) {
  p_node.AddAttribute(attr_name, attr_value);
}

void AddAttribute(onnxruntime::Node& p_node, const std::string& attr_name, gsl::span<const int64_t> attr_value) {
  p_node.AddAttribute(attr_name, attr_value);
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
  AddAttribute(node_1, "shape", AsSpan<int64_t>({2, 3}));
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

    status = graph.ReplaceInitializedTensor(bad_name, OrtValue());
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto bad_type = original;
    bad_type.set_data_type(TensorProto_DataType_FLOAT16);

    status = graph.ReplaceInitializedTensor(bad_type, OrtValue());
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto bad_dims = original;
    bad_dims.clear_dims();
    bad_dims.add_dims(2);
    bad_dims.add_dims(1);

    status = graph.ReplaceInitializedTensor(bad_dims, OrtValue());
    ASSERT_FALSE(status.IsOK());
  }

  {
    ONNX_NAMESPACE::TensorProto valid_replacement = original;
    valid_replacement.clear_int32_data();
    valid_replacement.add_int32_data(3);
    valid_replacement.add_int32_data(4);

    status = graph.ReplaceInitializedTensor(valid_replacement, OrtValue());
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    auto tensor_data_matches = [](const Graph& graph, const ONNX_NAMESPACE::TensorProto& a,
                                  const ONNX_NAMESPACE::TensorProto& b) -> bool {
      // For simplicity. We do not want to deal with external and raw data combinations.
      Tensor tensor_a;
      EXPECT_TRUE(utils::CreateTensorFromTensorProto(Env::Default(), graph.ModelPath(), a, tensor_a).IsOK());
      Tensor tensor_b;
      EXPECT_TRUE(utils::CreateTensorFromTensorProto(Env::Default(), graph.ModelPath(), b, tensor_b).IsOK());

      EXPECT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_a.GetElementType());

      if (tensor_a.GetElementType() != tensor_b.GetElementType()) {
        return false;
      }
      if (tensor_a.Shape() != tensor_b.Shape()) {
        return false;
      }
      const auto span_a = tensor_a.DataAsSpan<int32_t>();
      const auto span_b = tensor_b.DataAsSpan<int32_t>();
      return std::equal(span_a.begin(), span_a.end(), span_b.begin());
    };

    // check retrieved tensor
    const ONNX_NAMESPACE::TensorProto* result;
    ASSERT_TRUE(graph.GetInitializedTensor(initializer_name, result));
    ASSERT_TRUE(tensor_data_matches(graph, *result, valid_replacement));

    // check GraphProto content
    const ONNX_NAMESPACE::GraphProto graph_proto = graph.ToGraphProto();
    ASSERT_EQ(graph_proto.initializer_size(), 1);
    ASSERT_TRUE(tensor_data_matches(graph, graph_proto.initializer(0), valid_replacement));
  }
}

TEST_F(GraphTest, ReplaceInitializedTensorRejectsMissingOrtValueForExternalDataInMemory) {
  Model model{"GraphUpdateTest_ExternalData", false, *logger_};
  auto& graph = model.MainGraph();
  const std::string initializer_name = "initializer";
  constexpr int64_t kTensorSize = 64;

  ONNX_NAMESPACE::TensorProto original{};
  original.set_data_type(TensorProto_DataType_INT32);
  original.add_dims(kTensorSize);
  for (int32_t i = 0; i < kTensorSize; ++i) {
    original.add_int32_data(i);
  }
  original.set_name(initializer_name);
  graph.AddInitializedTensor(original);

  auto allocator = CPUAllocator::DefaultInstance();
  TensorShape shape({kTensorSize});
  OrtValue ort_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), shape, allocator, ort_value);
  auto* data = ort_value.GetMutable<Tensor>()->MutableData<int32_t>();
  for (int32_t i = 0; i < kTensorSize; ++i) {
    data[i] = i + 1;
  }

  auto replacement = utils::TensorToTensorProto(ort_value.Get<Tensor>(), initializer_name, true);
  Status status = graph.ReplaceInitializedTensor(replacement, OrtValue());
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("requires an allocated OrtValue"));
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)

namespace {
void SetTensorProtoExternalData(const std::string& key, const std::string& value,
                                ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto* external_data = tensor_proto.mutable_external_data();
  auto kvp_it = std::find_if(
      external_data->begin(), external_data->end(),
      [&key](const ONNX_NAMESPACE::StringStringEntryProto& kvp) { return kvp.key() == key; });
  auto* kvp = kvp_it != external_data->end() ? &(*kvp_it) : external_data->Add();
  kvp->set_key(key);
  kvp->set_value(value);
}
}  // namespace

TEST_F(GraphTest, InjectExternalInitializedTensors) {
  const std::string initializer_name = "test_external_initializer";

  std::vector<int32_t> tensor_data = []() {
    std::vector<int32_t> tensor_data(100);
    std::iota(tensor_data.begin(), tensor_data.end(), 0);
    return tensor_data;
  }();

  // Create OrtValue for replacement
  TensorShape data_shape{static_cast<int64_t>(tensor_data.size())};
  OrtValue ort_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), data_shape, tensor_data.data(),
                       OrtMemoryInfo(onnxruntime::CPU, OrtAllocatorType::OrtDeviceAllocator), ort_value);
  const InlinedHashMap<std::string, OrtValue> injection_initializers = {
      {initializer_name, ort_value}};

  // We do not need actual files there since we are not going to load it.
  const auto tensor_data_dir_path = ORT_TSTR(".");
  const auto tensor_data_dir_relative_path = ORT_TSTR("external_data.bin");

  const auto tensor_proto =
      [&]() {
        ONNX_NAMESPACE::TensorProto tensor_proto;
        tensor_proto.set_name(initializer_name);
        tensor_proto.add_dims(tensor_data.size());
        tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
        tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
        SetTensorProtoExternalData("location", ToUTF8String(tensor_data_dir_relative_path),
                                   tensor_proto);
        SetTensorProtoExternalData("offset", "0", tensor_proto);
        SetTensorProtoExternalData("length", std::to_string(tensor_data.size() * sizeof(int32_t)), tensor_proto);
        return tensor_proto;
      }();

  Model m{"test_model", false, *logger_};
  Graph& graph = m.MainGraph();
  graph.AddInitializedTensor(tensor_proto);

  ASSERT_EQ(graph.GetAllInitializedTensors().size(), 1U);

  const TensorProto* external_data = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor(initializer_name, external_data));
  if (external_data) {
    ASSERT_TRUE(utils::HasExternalData(*external_data));
  } else {
    // Fail the test because external_data shouldn't be nullptr.
    // This if-else is added for suppressing warning C6011: dereferencing NULL pointer.
    ASSERT_TRUE(false);
  }

  // Replace things.
  ASSERT_STATUS_OK(graph.InjectExternalInitializedTensors(injection_initializers));

  ASSERT_EQ(graph.GetAllInitializedTensors().size(), 1U);

  const TensorProto* with_data = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor(initializer_name, with_data));
  if (with_data) {
    // This proto still has external data, but now it points to the OrtValue.
    ASSERT_TRUE(utils::HasExternalData(*with_data));
    const auto& original_tensor = ort_value.Get<Tensor>();
    Tensor replaced_tensor;
    ASSERT_STATUS_OK(utils::CreateTensorFromTensorProto(Env::Default(), tensor_data_dir_path, *with_data,
                                                        replaced_tensor));
    ASSERT_EQ(original_tensor.GetElementType(), replaced_tensor.GetElementType());
    const auto original_span = original_tensor.DataAsSpan<int32_t>();
    const auto replaced_span = replaced_tensor.DataAsSpan<int32_t>();
    ASSERT_EQ(original_span, replaced_span);
  } else {
    // Fail the test because with_data shouldn't be nullptr.
    // This if-else is added for suppressing warning C6011: dereferencing NULL pointer.
    ASSERT_TRUE(false);
  }
}

TEST_F(GraphTest, InjectFloat6ExternalInitializedTensors) {
  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    for (const bool owned : {false, true}) {
      constexpr size_t count = 132;
      const std::string name = "float6";
      Model model{"float6_injection", false, *logger_};
      Graph& graph = model.MainGraph();
      TensorProto initializer;
      initializer.set_name(name);
      initializer.set_data_type(data_type);
      initializer.add_dims(count);
      ExternalDataInfo::SetExternalLocationToProto("float6.bin", 0, count * 6 / 8, initializer);
      graph.AddInitializedTensor(initializer);

      const auto* type = DataTypeImpl::TensorTypeFromONNXEnum(data_type)->GetElementType();
      TensorShape shape{count};
      std::vector<uint8_t> bits(count);
      for (size_t i = 0; i < count; ++i) bits[i] = static_cast<uint8_t>(i % 64);
      OrtValue value;
      if (owned) {
        Tensor::InitOrtValue(type, shape, CPUAllocator::DefaultInstance(), value);
        std::copy(bits.begin(), bits.end(),
                  static_cast<uint8_t*>(value.GetMutable<Tensor>()->MutableDataRaw()));
      } else {
        Tensor::InitOrtValue(type, shape, bits.data(),
                             OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), value);
      }

      ASSERT_STATUS_OK(graph.InjectExternalInitializedTensors({{name, value}}));
      const TensorProto* replaced = nullptr;
      ASSERT_TRUE(graph.GetInitializedTensor(name, replaced));
      ASSERT_NE(replaced, nullptr);
      EXPECT_FALSE(utils::HasExternalData(*replaced));
      ASSERT_EQ(replaced->raw_data().size(), count * 6 / 8);
      std::vector<uint8_t> decoded;
      ASSERT_STATUS_OK(utils::UnpackInitializerData(*replaced, decoded));
      EXPECT_EQ(decoded, bits);
    }
  }
}

TEST_F(GraphTest, InjectFloat6ExternalInitializersFromFilesInMemory) {
  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    for (const bool use_buffers_directly : {false, true}) {
      constexpr size_t count = 132;
      Model model{"float6_file_injection", false, *logger_};
      Graph& graph = model.MainGraph();
      TensorProto initializer;
      initializer.set_name("float6");
      initializer.set_data_type(data_type);
      initializer.add_dims(count);
      ExternalDataInfo::SetExternalLocationToProto("float6.bin", 1, count * 6 / 8, initializer);
      graph.AddInitializedTensor(initializer);

      std::string data(1, '\0');
      std::vector<uint8_t> expected;
      for (size_t i = 0; i < count / 4; ++i) {
        data.append("\x01\x42\xc3", 3);
        expected.insert(expected.end(), {1, 8, 52, 48});
      }
      InlinedHashMap<PathString, std::pair<char*, size_t>> files{
          {ORT_TSTR("float6.bin"), {data.data(), data.size()}}};
      files.at(ORT_TSTR("float6.bin")).second = data.size() - 1;
      EXPECT_FALSE(graph.InjectExternalInitializersFromFilesInMemory(files, use_buffers_directly).IsOK());
      files.at(ORT_TSTR("float6.bin")).second = data.size();
      ASSERT_STATUS_OK(graph.InjectExternalInitializersFromFilesInMemory(files, use_buffers_directly));
      const TensorProto* replaced = nullptr;
      ASSERT_TRUE(graph.GetInitializedTensor("float6", replaced));
      ASSERT_NE(replaced, nullptr);
      EXPECT_FALSE(utils::HasExternalData(*replaced));
      ASSERT_EQ(replaced->raw_data().size(), count * 6 / 8);
      EXPECT_EQ(replaced->raw_data(), data.substr(1));
      std::vector<uint8_t> decoded;
      ASSERT_STATUS_OK(utils::UnpackInitializerData(*replaced, decoded));
      EXPECT_EQ(decoded, expected);
    }
  }
}

TEST_F(GraphTest, InlineFloat6ExternalTensorAttribute) {
  TemporaryDirectory temp_dir{ORT_TSTR("float6_attribute_test")};
  const auto model_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("model.onnx");
  const auto data_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("attribute.bin");
  for (const auto type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    {
      std::ofstream output(data_path, std::ios::binary);
      output.write("\x01\x42\xc3", 3);
    }
    ModelProto source;
    source.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    source.mutable_graph()->set_name("float6_attribute");
    source.add_opset_import()->set_version(17);
    TensorProto tensor;
    tensor.set_data_type(type);
    tensor.add_dims(4);
    ExternalDataInfo::SetExternalLocationToProto("attribute.bin", 0, 3, tensor);
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(std::move(source), model_path.native(), model, nullptr, *logger_));
    auto& graph = model->MainGraph();
    NodeAttributes attributes;
    attributes.emplace("value", ONNX_NAMESPACE::MakeAttribute("value", tensor));
    graph.AddNode("attribute_node", "Custom", "", {}, {}, &attributes, "test");
    ASSERT_STATUS_OK(graph.ConvertInitializersIntoOrtValues());
    ASSERT_EQ(graph.NumberOfNodes(), 1);
    const auto& inlined = graph.Nodes().begin()->GetAttributes().at("value").t();
    EXPECT_FALSE(utils::HasExternalData(inlined));
    EXPECT_EQ(inlined.raw_data(), std::string("\x01\x42\xc3", 3));
  }
}
#endif

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
    // Due to changes in a way we generate ToGraphProto() const, we can not guarantee the order of initializers
    // in the generated GraphProto.
    auto initializers = proto.initializer();
    auto hit = std::find_if(initializers.begin(), initializers.end(),
                            [&init](const ONNX_NAMESPACE::TensorProto& t) { return t.name() == init.name(); });
    EXPECT_NE(hit, initializers.end())
        << "Initializer with name '" << init.name() << "' not found in the proto.";
    EXPECT_EQ(hit->int32_data()[0], 1);

    hit = std::find_if(initializers.begin(), initializers.end(),
                       [&init2](const ONNX_NAMESPACE::TensorProto& t) { return t.name() == init2.name(); });
    EXPECT_NE(hit, initializers.end())
        << "Initializer with name '" << init2.name() << "' not found in the proto.";

    EXPECT_EQ(hit->int32_data()[0], 2);
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

#if !defined(DISABLE_SPARSE_TENSORS)
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
#endif  //! defined(DISABLE_SPARSE_TENSORS)

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
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(Model::Load(std::move(m), model, nullptr, *logger_),
                                      "Invalid model. Node input 'y' is not a graph input, "
                                      "initializer, or output of a previous node.");
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

// The Model class internally:
// 1. Converts the ONNX Model to an ORT Model
// 2. Converts the ORT Model's Graph back to an ONNX Graph so that it
//    can run the ONNX Checker on it.
// Previously this was buggy for models containing subgraphs with IR Version < 4.
// We didn't always ensure that initializers are a subset of inputs, which is
// required for IR version < 4.
TEST_F(GraphTest, ConstantsBecomeInitializersAndInputs) {
  ModelProto m;
  m.set_ir_version(ONNX_NAMESPACE::IR_VERSION_2017_11_3);
  ImportOpset(m, "", 13);
  GraphProto* g = m.mutable_graph();
  g->set_name("test");

  // Construct "output = if x: 2.0 else 1.0"
  ValueInfoProto* x = g->add_input();
  x->set_name("x");
  SetTypeAndShape(x->mutable_type()->mutable_tensor_type(), TensorProto_DataType_BOOL, {1});

  NodeProto* if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->set_name("If");
  if_node->add_input("x");
  if_node->add_output("output");

  AttributeProto* if_attr = if_node->add_attribute();
  if_attr->set_name("then_branch");
  if_attr->set_type(AttributeProto_AttributeType_GRAPH);
  GraphProto* then_g = if_attr->mutable_g();
  then_g->set_name("then");
  ValueInfoProto* then_out = then_g->add_output();
  then_out->set_name("then_out");
  SetTypeAndShape(then_out->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {1});
  NodeProto* two_node = then_g->add_node();
  two_node->set_op_type("Constant");
  AttributeProto* two_attr = two_node->add_attribute();
  two_attr->set_name("value");
  two_attr->set_type(AttributeProto_AttributeType_TENSOR);
  two_attr->mutable_t()->add_float_data(2.0);
  two_attr->mutable_t()->set_data_type(TensorProto_DataType_FLOAT);
  two_attr->mutable_t()->add_dims(1);
  two_node->set_name("Constant_two");
  two_node->add_output("then_out");

  AttributeProto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto_AttributeType_GRAPH);
  GraphProto* else_g = else_attr->mutable_g();
  else_g->set_name("else");
  ValueInfoProto* else_out = else_g->add_output();
  else_out->set_name("else_out");
  SetTypeAndShape(else_out->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {1});
  NodeProto* one_node = else_g->add_node();
  one_node->set_op_type("Constant");
  AttributeProto* one_attr = one_node->add_attribute();
  one_attr->set_name("value");
  one_attr->set_type(AttributeProto_AttributeType_TENSOR);
  one_attr->mutable_t()->add_float_data(1.0);
  one_attr->mutable_t()->set_data_type(TensorProto_DataType_FLOAT);
  one_attr->mutable_t()->add_dims(1);
  one_node->set_name("Constant_one");
  one_node->add_output("else_out");

  ValueInfoProto* output = g->add_output();
  output->set_name("output");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {1});

  std::shared_ptr<Model> model;
  Status st = Model::Load(std::move(m), model, nullptr, *logger_);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
}

TEST_F(GraphTest, SubgraphOutputIsOuterScopeValue) {
  std::shared_ptr<Model> model;
  common::Status st = Model::Load(ORT_TSTR("./testdata/ort_github_issue_11536.onnx"), model, nullptr, *logger_);
  ASSERT_FALSE(st.IsOK());
  EXPECT_THAT(st.ErrorMessage(),
              ::testing::ContainsRegex("Subgraph output \\(.*\\) is an outer scope value being returned directly."));
}

static void CreateInitializerWithDataInMemory(const std::string& name, const AllocatorPtr& allocator, int64_t size,
                                              TensorProto& tensor_proto, OrtValue& ort_value) {
  TensorShape shape({size});
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape, allocator, ort_value);
  float v = 0;
  auto* data = ort_value.GetMutable<Tensor>()->MutableData<float>();
  for (int64_t i = 0; i < size; ++i) {
    *data++ = v++;
  }

  tensor_proto = utils::TensorToTensorProto(ort_value.Get<Tensor>(), name, true);
}

TEST(GraphGetOrtValueInitializerTest, ReturnsOrtValueForExistingInitializer) {
  Model model("TestModel", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {}, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  // Create a simple TensorProto initializer
  const std::string name = "init1";
  auto allocator = CPUAllocator::DefaultInstance();
  constexpr const int64_t kTensorSize = 256;
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory(name, allocator, kTensorSize, tensor_proto, ort_value);

  ASSERT_STATUS_OK(graph.AddInitializedOrtValue(tensor_proto, ort_value));

  // Test retrieval
  OrtValue retrieved;
  EXPECT_TRUE(graph.GetOrtValueInitializer(name, retrieved, false));
  const Tensor& t = retrieved.Get<Tensor>();
  EXPECT_EQ(t.Shape().Size(), kTensorSize);
}

TEST(GraphGetOrtValueInitializerTest, ReturnsFalseForNonExistentInitializer) {
  Model model("TestModel", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {}, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  OrtValue retrieved;
  EXPECT_FALSE(graph.GetOrtValueInitializer("does_not_exist", retrieved, false));
}

namespace {
// Casing only, do not add members
class NodeWrapper : public Node {
 public:
  Node::Definitions& MutableDefinitions() {
    return Node::MutableDefinitions();
  }
};
}  // namespace

TEST(GraphGetOrtValueInitializerTest, ReturnsOrtValueFromOuterScope) {
  // Create parent graph with initializer
  Model parent_model("ParentModel", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {}, {},
                     DefaultLoggingManager().DefaultLogger());
  Graph& parent_graph = parent_model.MainGraph();

  const std::string outer_init_name = "outer_init";
  auto allocator = CPUAllocator::DefaultInstance();
  constexpr const int64_t kTensorSize = 256;
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory(outer_init_name, allocator, kTensorSize, tensor_proto, ort_value);

  ASSERT_STATUS_OK(parent_graph.AddInitializedOrtValue(tensor_proto, ort_value));

  // Create a node in parent graph that will be the parent node for the subgraph
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto& input_arg = parent_graph.GetOrCreateNodeArg("node_input", &tensor_type);
  auto& output_arg = parent_graph.GetOrCreateNodeArg("node_output", &tensor_type);
  NodeArg* inputs[] = {&input_arg};
  NodeArg* outputs[] = {&output_arg};

  // Create parent node with a subgraph attribute
  auto& parent_node = parent_graph.AddNode("parent_node", "If", "parent node with subgraph", inputs, outputs);
  // Add the initializer name to the parent node's implicit input defs
  NodeArg* outer_init_nodearg = parent_graph.GetNodeArg(outer_init_name);
  ASSERT_NE(outer_init_nodearg, nullptr);
  {
    // Test hack to tweak an internal structure.
    auto& node_wrapper = static_cast<NodeWrapper&>(parent_node);
    node_wrapper.MutableDefinitions().implicit_input_defs.push_back(outer_init_nodearg);
  }

  // Create subgraph
  GraphProto subgraph_proto;
  subgraph_proto.set_name("Subgraph");
  Graph subgraph(parent_model, &subgraph_proto, parent_graph.DomainToVersionMap(), parent_model.IrVersion(),
                 nullptr, &parent_graph, &parent_node, DefaultLoggingManager().DefaultLogger(), false);

  // Test retrieval from outer scope
  OrtValue retrieved;
  EXPECT_TRUE(subgraph.GetOrtValueInitializer("outer_init", retrieved, true));
  const Tensor& t = retrieved.Get<Tensor>();
  EXPECT_EQ(t.Shape().Size(), kTensorSize);
}

TEST_F(GraphTest, AddInitializedOrtValueWithExternalData) {
  Model model("TestAddInitializedOrtValue", false, *logger_);
  Graph& graph = model.MainGraph();

  // Create a TensorProto with external data reference
  const std::string external_data_init = "external_data_init";
  auto allocator = CPUAllocator::DefaultInstance();
  constexpr const int64_t kTensorSize = 256;
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory(external_data_init, allocator, kTensorSize, tensor_proto, ort_value);

  // Test adding the initialized OrtValue with external data reference
  ASSERT_STATUS_OK(graph.AddInitializedOrtValue(tensor_proto, ort_value));

  // Verify the initializer was added correctly
  OrtValue retrieved_value;
  ASSERT_TRUE(graph.GetOrtValueInitializer(external_data_init, retrieved_value, false));

  // Verify the tensor data
  const Tensor& retrieved_tensor = retrieved_value.Get<Tensor>();
  ASSERT_EQ(retrieved_tensor.Shape().Size(), kTensorSize);
  ASSERT_EQ(retrieved_tensor.DataType(), DataTypeImpl::GetType<float>());

  // Verify the TensorProto was also added and has external data location
  const TensorProto* retrieved_proto = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor(external_data_init, retrieved_proto));
  ASSERT_NE(retrieved_proto, nullptr);
  ASSERT_EQ(retrieved_proto->name(), external_data_init);
  ASSERT_TRUE(utils::HasExternalDataInMemory(tensor_proto));
}

TEST_F(GraphTest, AddInitializedTensorRejectsExternalDataInMemoryWithoutOrtValue) {
  Model model("TestAddInitializedTensorRejectsExternalDataInMemory", false, *logger_);
  Graph& graph = model.MainGraph();

  auto allocator = CPUAllocator::DefaultInstance();
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory("external_data_init", allocator, 256, tensor_proto, ort_value);

  EXPECT_THROW(graph.AddInitializedTensor(tensor_proto), OnnxRuntimeException);
}

TEST_F(GraphTest, AddInitializedOrtValueRejectsMissingOrtValueForExternalDataInMemory) {
  Model model("TestAddInitializedOrtValueRejectsMissingOrtValue", false, *logger_);
  Graph& graph = model.MainGraph();

  auto allocator = CPUAllocator::DefaultInstance();
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory("external_data_init", allocator, 256, tensor_proto, ort_value);

  Status status = graph.AddInitializedOrtValue(tensor_proto, OrtValue());
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("requires an allocated ortvalue_initializer"));
}

TEST_F(GraphTest, AddInitializedOrtValueMismatch) {
  Model model("TestAddInitializedOrtValue_Mismatch", false, *logger_);
  Graph& graph = model.MainGraph();

  // Create a TensorProto with external data reference
  const std::string name = "init";
  constexpr const int64_t kTensorSize = 256;
  auto allocator = CPUAllocator::DefaultInstance();
  TensorProto tensor_proto;
  OrtValue ort_value;
  TensorShape shape({kTensorSize});
  CreateInitializerWithDataInMemory(name, allocator, kTensorSize, tensor_proto, ort_value);

  OrtValue ort_value_diff;
  // Now try to create a value that has a different data type
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), shape, allocator, ort_value_diff);
  Status status = graph.AddInitializedOrtValue(tensor_proto, ort_value_diff);
  ASSERT_FALSE(status.IsOK());

  // Create OrtValue with different shape [2]
  TensorShape diff_shape({2});
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), diff_shape, allocator, ort_value_diff);

  // Fails on shape mismatch
  status = graph.AddInitializedOrtValue(tensor_proto, ort_value_diff);
  ASSERT_FALSE(status.IsOK());
}

TEST_F(GraphTest, AddInitializedOrtValueDuplicate) {
  Model model("TestAddInitializedOrtValue_Duplicate", false, *logger_);
  Graph& graph = model.MainGraph();

  // Create a TensorProto with external data reference
  const std::string name = "init";
  constexpr const int64_t kTensorSize = 256;
  auto allocator = CPUAllocator::DefaultInstance();
  TensorProto tensor_proto;
  OrtValue ort_value;
  CreateInitializerWithDataInMemory(name, allocator, kTensorSize, tensor_proto, ort_value);

  // Add the first initializer successfully
  ASSERT_STATUS_OK(graph.AddInitializedOrtValue(tensor_proto, ort_value));

  // try again
  Status status = graph.AddInitializedOrtValue(tensor_proto, ort_value);
  ASSERT_FALSE(status.IsOK());
}

#ifdef ENABLE_TRAINING

TEST_F(GraphTest, GraphConstruction_MemoryEfficientTopologicalSort_Recompute) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                         |
                  node_0 (Identity)
                     /       \
        node_1 (Identity)     \
                    |         |
        node_4 (Identity)     |
                    |         |
                  YieldOp   recompute_node_1
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
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_yield_out_1", &tensor_int32);
  auto& output_arg6 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});
  graph.AddNode("recompute_node_1", "Identity_Fake", "recompute node 1", {&output_arg0}, {&output_arg2});

  graph.AddNode("node_4", "Identity_Fake", "node 4", {&output_arg1}, {&output_arg4});

  ONNX_NAMESPACE::AttributeProto full_shape_outputs;
  const std::string attribute_name = "full_shape_outputs";
  full_shape_outputs.set_name(attribute_name);
  full_shape_outputs.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  full_shape_outputs.add_ints(static_cast<int64_t>(0));
  NodeAttributes attributes({{attribute_name, full_shape_outputs}});

  graph.AddNode("node_yield", "YieldOp", "node yield", {&output_arg4}, {&output_arg5}, &attributes, kMSDomain);
  graph.AddNode("node_1_grad", "Merge_Fake", "node_1 gradient", {&output_arg5, &output_arg2}, {&output_arg6});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // MEMORY_EFFICIENT order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::MEMORY_EFFICIENT);
    const std::vector<std::string> expected_priority_based_order =
        {"node_0", "node_1", "node_4", "node_yield", "recompute_node_1", "node_1_grad"};
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "MEMORY_EFFICIENT based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_MemoryEfficientTopologicalSort_MultiLayerRecompute) {
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
              YieldOp     |     |         \        \
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
  auto& output_yield = graph.GetOrCreateNodeArg("yield_out", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});
  graph.AddNode("node_2", "Identity_Fake", "node 2", {&output_arg1}, {&output_arg2});
  graph.AddNode("node_3", "Identity_Fake", "node 3", {&output_arg2}, {&output_arg3});
  graph.AddNode("loss", "Identity_Fake", "loss node", {&output_arg3}, {&output_loss});
  ONNX_NAMESPACE::AttributeProto full_shape_outputs;
  const std::string attribute_name = "full_shape_outputs";
  full_shape_outputs.set_name(attribute_name);
  full_shape_outputs.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  full_shape_outputs.add_ints(static_cast<int64_t>(0));
  NodeAttributes attributes({{attribute_name, full_shape_outputs}});
  graph.AddNode("node_yield", "YieldOp", "node yield", {&output_loss}, {&output_yield}, &attributes, kMSDomain);

  // Recompute graph
  auto& recomputed_arg3 = graph.GetOrCreateNodeArg("node_3_out_recomputed", &tensor_int32);
  auto& recomputed_arg2 = graph.GetOrCreateNodeArg("node_2_out_recomputed", &tensor_int32);
  auto& recomputed_arg1 = graph.GetOrCreateNodeArg("node_1_out_recomputed", &tensor_int32);

  graph.AddNode("node_3_recompute", "Identity_Fake", "node 3 recompute", {&output_arg2}, {&recomputed_arg3});
  graph.AddNode("node_2_recompute", "Identity_Fake", "node 2 recompute", {&output_arg1}, {&recomputed_arg2});
  graph.AddNode("node_1_recompute", "Identity_Fake", "node 1 recompute", {&output_arg0}, {&recomputed_arg1});

  // BW Graph
  auto& loss_grad_output = graph.GetOrCreateNodeArg("loss_grad_output", &tensor_int32);
  auto& node_3_grad_output = graph.GetOrCreateNodeArg("node_3_grad_output", &tensor_int32);
  auto& node_2_grad_output = graph.GetOrCreateNodeArg("node_2_grad_output", &tensor_int32);
  auto& node_1_grad_output = graph.GetOrCreateNodeArg("node_1_grad_output", &tensor_int32);

  graph.AddNode("loss_grad", "Merge_Fake", "loss gradient", {&output_yield, &output_arg3}, {&loss_grad_output});
  graph.AddNode("node_3_grad", "Merge_Fake", "node 3 gradient", {&loss_grad_output, &recomputed_arg3}, {&node_3_grad_output});
  graph.AddNode("node_2_grad", "Merge_Fake", "node 2 gradient", {&node_3_grad_output, &recomputed_arg2}, {&node_2_grad_output});
  graph.AddNode("node_1_grad", "Merge_Fake", "node 1 gradient", {&node_2_grad_output, &recomputed_arg1}, {&node_1_grad_output});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // MEMORY_EFFICIENT order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::MEMORY_EFFICIENT);
    const std::vector<std::string> expected_priority_based_order = {
        "node_0",
        "node_1",
        "node_2",
        "node_3",
        "loss",
        "node_yield",
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
      EXPECT_TRUE(node->Name() == expected_priority_based_order[i]) << "MEMORY_EFFICIENT based execution order is wrong.";
    }
  }
}

TEST_F(GraphTest, GraphConstruction_MemoryEfficientTopologicalSort_SubgraphGeneratingNodeHavingNoConsumers) {
  Model model("graph_1", false, *logger_);
  auto& graph = model.MainGraph();

  /*
                         |
                  node_0 (Identity)
                     /       \    \
        node_1 (Identity)     \   Identity
                    |         |       \_____graph_output_0
        node_4 (Identity)     |
                    |         |
                  YieldOp   recompute_node_1
                     \       /      \
               node_1_grad (Merge)   Identity
                        |               |
                                   graph_output_1
  */

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg0 = graph.GetOrCreateNodeArg("node_0_in_1", &tensor_int32);
  auto& output_arg0 = graph.GetOrCreateNodeArg("node_0_out_1", &tensor_int32);
  auto& graph_output_0_identity = graph.GetOrCreateNodeArg("graphoutput_0_identity_out_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  auto& output_arg5 = graph.GetOrCreateNodeArg("node_yield_out_1", &tensor_int32);
  auto& output_arg6 = graph.GetOrCreateNodeArg("node_5_out_1", &tensor_int32);

  graph.AddNode("node_0", "Identity_Fake", "node 0", {&input_arg0}, {&output_arg0});
  graph.AddNode("node_1", "Identity_Fake", "node 1", {&output_arg0}, {&output_arg1});
  graph.AddNode("graph_output_0_identity", "Identity_Fake", "graph output 0 identity", {&output_arg0}, {&graph_output_0_identity});
  graph.AddNode("recompute_node_1", "Identity_Fake", "recompute node 1", {&output_arg0}, {&output_arg2});

  auto& graph_output1_identity = graph.GetOrCreateNodeArg("graphoutput_1_identity_out_1", &tensor_int32);
  graph.AddNode("graph_output_1_identity", "Identity_Fake", "graph output 1 identity", {&output_arg2}, {&graph_output1_identity});

  graph.AddNode("node_4", "Identity_Fake", "node 4", {&output_arg1}, {&output_arg4});

  ONNX_NAMESPACE::AttributeProto full_shape_outputs;
  const std::string attribute_name = "full_shape_outputs";
  full_shape_outputs.set_name(attribute_name);
  full_shape_outputs.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  full_shape_outputs.add_ints(static_cast<int64_t>(0));
  NodeAttributes attributes({{attribute_name, full_shape_outputs}});

  graph.AddNode("node_yield", "YieldOp", "node yield", {&output_arg4}, {&output_arg5}, &attributes, kMSDomain);
  graph.AddNode("node_1_grad", "Merge_Fake", "node_1 gradient", {&output_arg5, &output_arg2}, {&output_arg6});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  GraphViewer graph_viewer(graph);

  // MEMORY_EFFICIENT order
  {
    auto& order = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::MEMORY_EFFICIENT);
    const std::vector<std::string> expected_order =
        {
            "node_0",
            "node_1",
            "node_4",
            "node_yield",
            "recompute_node_1",
            "node_1_grad",
            "graph_output_0_identity",
            "graph_output_1_identity",
        };
    for (size_t i = 0; i < order.size(); ++i) {
      auto node = graph.GetNode(order[i]);
      EXPECT_TRUE(node->Name() == expected_order[i])
          << "MEMORY_EFFICIENT based execution order is wrong. expected node is " << expected_order[i]
          << " but got " << node->Name();
    }
  }
}

#endif

// Test for shape inference with in-memory external data (issue #26261)
// This tests the fix for a regression where Constant nodes with large tensors (>127 bytes)
// stored as in-memory external data would cause shape inference to fail
TEST_F(GraphTest, ShapeInferenceWithInMemoryExternalData) {
  // Create a model with a Constant node that produces a tensor larger than kSmallTensorExternalDataThreshold (127 bytes)
  // This will trigger the in-memory externalization path
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Create a Constant node with a tensor of 16 INT64 values (128 bytes, just over the 127 threshold)
  auto* constant_node = graph_proto->add_node();
  constant_node->set_op_type("Constant");
  constant_node->set_name("const_node");
  constant_node->add_output("const_output");

  // Add the value attribute with a tensor
  auto* attr = constant_node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* tensor = attr->mutable_t();
  tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tensor->add_dims(16);  // 16 elements * 8 bytes = 128 bytes
  // Each split will be size 1, totaling 16
  for (int64_t i = 0; i < 16; ++i) {
    tensor->add_int64_data(1);
  }

  // Create a Split node that uses the constant as input
  // Split requires constant input for the 'split' parameter, which triggers shape inference
  auto* split_node = graph_proto->add_node();
  split_node->set_op_type("Split");
  split_node->set_name("split_node");
  split_node->add_input("input_data");
  split_node->add_input("const_output");  // Use constant as split sizes
  for (int i = 0; i < 16; ++i) {
    split_node->add_output("split_output_" + std::to_string(i));
  }

  // Add axis attribute
  auto* axis_attr = split_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  axis_attr->set_i(0);

  // Add graph input
  auto* input = graph_proto->add_input();
  input->set_name("input_data");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(16);
  input_type->mutable_shape()->add_dim()->set_dim_value(10);

  // Add graph outputs
  for (int i = 0; i < 16; ++i) {
    auto* output = graph_proto->add_output();
    output->set_name("split_output_" + std::to_string(i));
  }

  // Load the model - this should succeed with the fix
  // Before the fix, this would fail with:
  // "Cannot parse data from external tensors. Please load external data into raw data for tensor"
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger_));

  // Verify the graph was properly constructed
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Verify the constant node was converted to an initializer
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor("const_output", initializer));
  ASSERT_NE(initializer, nullptr);

  // Verify the Split node can access the constant data during shape inference
  const Node* split_node_ptr = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.Name() == "split_node") {
      split_node_ptr = &node;
      break;
    }
  }
  ASSERT_NE(split_node_ptr, nullptr);

  // Verify outputs are properly shaped
  ASSERT_EQ(split_node_ptr->OutputDefs().size(), 16u);
}

TEST_F(GraphTest, RejectsUnregisteredInMemoryInitializer) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("source");

  auto* initializer = graph_proto->add_initializer();
  initializer->set_name("malformed_initializer");
  initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  initializer->add_dims(1);
  ExternalDataInfo::SetExternalLocationToProto(
      utils::kTensorProtoNativeEndianMemoryAddressTag, 1, sizeof(std::string), *initializer);

  auto* node = graph_proto->add_node();
  node->set_op_type("Identity");
  node->add_input(initializer->name());
  node->add_output("output");
  auto* output = graph_proto->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);

  std::shared_ptr<Model> source_model;
  ORT_TRY {
    const auto status = Model::Load(std::move(model_proto), source_model, nullptr, *logger_);
    EXPECT_FALSE(status.IsOK());
    if (!status.IsOK()) {
      EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("in-memory address marker"));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      EXPECT_THAT(std::string(ex.what()), ::testing::HasSubstr("in-memory address marker"));
    });
  }
}

// Test for shape inference with in-memory external data using InferenceSession
// This test more accurately reproduces the issue by going through the full session initialization
// which includes graph optimizations that trigger the in-memory externalization
TEST_F(GraphTest, ShapeInferenceWithInMemoryExternalDataViaSession) {
  // Create the same model as above
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Create a Constant node with a tensor of 16 INT64 values (128 bytes)
  auto* constant_node = graph_proto->add_node();
  constant_node->set_op_type("Constant");
  constant_node->set_name("const_node");
  constant_node->add_output("const_output");

  auto* attr = constant_node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* tensor = attr->mutable_t();
  tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tensor->add_dims(16);
  for (int64_t i = 0; i < 16; ++i) {
    tensor->add_int64_data(1);
  }

  // Create a Split node
  auto* split_node = graph_proto->add_node();
  split_node->set_op_type("Split");
  split_node->set_name("split_node");
  split_node->add_input("input_data");
  split_node->add_input("const_output");
  for (int i = 0; i < 16; ++i) {
    split_node->add_output("split_output_" + std::to_string(i));
  }

  auto* axis_attr = split_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  axis_attr->set_i(0);

  // Add graph input
  auto* input = graph_proto->add_input();
  input->set_name("input_data");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(16);
  input_type->mutable_shape()->add_dim()->set_dim_value(10);

  // Add graph outputs
  for (int i = 0; i < 16; ++i) {
    auto* output = graph_proto->add_output();
    output->set_name("split_output_" + std::to_string(i));
  }

  // Save to a temporary file
  const std::string model_path = "test_in_memory_external_data.onnx";
  {
    std::ofstream file(model_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    ASSERT_TRUE(model_proto.SerializeToOstream(&file));
  }

  // Test with ORT_DISABLE_ALL optimization which should trigger the bug without the fix
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Default;  // This triggers the issue
  so.session_logid = "GraphTest.ShapeInferenceWithInMemoryExternalDataViaSession";

  InferenceSession session_object{so, GetEnvironment()};

  // This should succeed with the fix, fail without it
  ASSERT_STATUS_OK(session_object.Load(model_path));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Clean up
  std::remove(model_path.c_str());
}

// Test that explicitly triggers the in-memory externalization and then shape inference
// This test directly reproduces the bug scenario
TEST_F(GraphTest, ShapeInferenceAfterInitializerExternalization) {
  // Create a model with a Split node that depends on a constant initializer
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Create initializer directly (not as Constant node) with 128 bytes
  auto* initializer = graph_proto->add_initializer();
  initializer->set_name("split_sizes");
  initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  initializer->add_dims(16);  // 16 * 8 = 128 bytes
  for (int64_t i = 0; i < 16; ++i) {
    initializer->add_int64_data(1);
  }

  // Create a Split node that uses this initializer
  auto* split_node = graph_proto->add_node();
  split_node->set_op_type("Split");
  split_node->set_name("split_node");
  split_node->add_input("input_data");
  split_node->add_input("split_sizes");  // Uses the large initializer
  for (int i = 0; i < 16; ++i) {
    split_node->add_output("split_output_" + std::to_string(i));
  }

  auto* axis_attr = split_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  axis_attr->set_i(0);

  // Add graph input
  auto* input = graph_proto->add_input();
  input->set_name("input_data");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(16);
  input_type->mutable_shape()->add_dim()->set_dim_value(10);

  // Add graph outputs
  for (int i = 0; i < 16; ++i) {
    auto* output = graph_proto->add_output();
    output->set_name("split_output_" + std::to_string(i));
  }

  // Load model
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger_));

  Graph& graph = model->MainGraph();
  // First resolve should succeed
  ASSERT_STATUS_OK(graph.Resolve());

  // Check if the initializer was actually externalized
  const ONNX_NAMESPACE::TensorProto* initializer_after = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor("split_sizes", initializer_after));
  ASSERT_NE(initializer_after, nullptr);
  // Debug: verify it was externalized
  ASSERT_FALSE(utils::HasExternalDataInMemory(*initializer_after))
      << "We no longer externalize data in the Graph constructor.";

  // Now externalize explicitly to trigger the bug scenario
  ASSERT_STATUS_OK(graph.ConvertInitializersIntoOrtValues());
  ASSERT_TRUE(graph.GetInitializedTensor("split_sizes", initializer_after));
  ASSERT_NE(initializer_after, nullptr);
  ASSERT_TRUE(utils::HasExternalDataInMemory(*initializer_after))
      << "The initializer should externalize now";

  // Mark the graph as needing resolve to force shape inference to run again
  graph.SetGraphResolveNeeded();

  // Resolve again - this should trigger shape inference with the externalized initializer
  // Without the fix, this will fail with "Cannot parse data from external tensors"
  // With the fix, getInputData() materializes the external data for shape inference
  Status second_resolve = graph.Resolve();
  ASSERT_TRUE(second_resolve.IsOK()) << "Second resolve failed: " << second_resolve.ErrorMessage();
}

// Targeted test for the TensorToTensorProto defense-in-depth: calling with a string tensor
// and use_tensor_buffer=true must produce a TensorProto with string_data (not external data).
TEST_F(GraphTest, TensorToTensorProtoStringTensorDefenseInDepth) {
  const int num_strings = 20;
  TensorShape shape({num_strings});
  Tensor string_tensor(DataTypeImpl::GetType<std::string>(), shape, CPUAllocator::DefaultInstance());
  auto* data = string_tensor.MutableData<std::string>();
  for (int i = 0; i < num_strings; ++i) {
    data[i] = "test_value_" + std::to_string(i);
  }

  // Verify the tensor is large enough to normally trigger the external data path.
  ASSERT_GT(string_tensor.SizeInBytes(), utils::kSmallTensorExternalDataThreshold);

  // Call with use_tensor_buffer=true — defense-in-depth should still produce string_data.
  auto tensor_proto = utils::TensorToTensorProto(string_tensor, "string_test", /*use_tensor_buffer=*/true);

  ASSERT_EQ(tensor_proto.string_data_size(), num_strings)
      << "TensorToTensorProto should populate string_data for string tensors even with use_tensor_buffer=true";
  ASSERT_FALSE(utils::HasExternalDataInMemory(tensor_proto))
      << "String tensor should not use external data in memory";

  for (int i = 0; i < num_strings; ++i) {
    EXPECT_EQ(tensor_proto.string_data(i), "test_value_" + std::to_string(i));
  }
}

// Regression test: ConvertInitializersIntoOrtValues must skip string tensors because their
// raw buffer contains std::string objects (with internal pointers), not serializable data.
// Without the fix, string initializer data was lost when the TensorProto was replaced with
// one using external data in memory, breaking ORT format model serialization.
TEST_F(GraphTest, ConvertInitializersIntoOrtValuesSkipsStringTensors) {
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_string_initializer_conversion");

  // Create a string initializer with enough elements to exceed kSmallTensorExternalDataThreshold (127 bytes).
  // sizeof(std::string) is typically 32 bytes (MSVC/libstdc++) or 24 bytes (libc++), so 20 elements
  // will exceed 127 bytes on all major platforms (20 * 24 = 480 > 127).
  const int num_strings = 20;
  auto* string_initializer = graph_proto->add_initializer();
  string_initializer->set_name("string_data");
  string_initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  string_initializer->add_dims(num_strings);
  for (int i = 0; i < num_strings; ++i) {
    string_initializer->add_string_data("value_" + std::to_string(i));
  }

  // Create a Gather node: Gather(string_data, indices) -> output
  auto* gather_node = graph_proto->add_node();
  gather_node->set_op_type("Gather");
  gather_node->add_input("string_data");
  gather_node->add_input("indices");
  gather_node->add_output("output");
  auto* axis_attr = gather_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  axis_attr->set_i(0);

  // Add graph input for indices
  auto* input = graph_proto->add_input();
  input->set_name("indices");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  input_type->mutable_shape()->add_dim()->set_dim_value(1);

  // Add graph output
  auto* output = graph_proto->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);

  // Load and resolve
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Verify string initializer has string_data before conversion
  const ONNX_NAMESPACE::TensorProto* init_before = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor("string_data", init_before));
  ASSERT_EQ(init_before->string_data_size(), num_strings);
  ASSERT_FALSE(utils::HasExternalDataInMemory(*init_before));

  // Convert initializers into OrtValues
  ASSERT_STATUS_OK(graph.ConvertInitializersIntoOrtValues());

  // After conversion, string initializer should still have string_data intact
  // (i.e., it should NOT have been replaced with external data in memory).
  const ONNX_NAMESPACE::TensorProto* init_after = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor("string_data", init_after));
  ASSERT_EQ(init_after->string_data_size(), num_strings)
      << "String initializer data was lost during ConvertInitializersIntoOrtValues";
  ASSERT_FALSE(utils::HasExternalDataInMemory(*init_after))
      << "String tensor should not use external data in memory";

  // Verify the string content is preserved
  for (int i = 0; i < num_strings; ++i) {
    EXPECT_EQ(init_after->string_data(i), "value_" + std::to_string(i));
  }

  // End-to-end: save to ORT format and reload, verifying string data survives the round-trip.
  flatbuffers::FlatBufferBuilder builder;
  {
    flatbuffers::Offset<fbs::Model> fbs_model_offset;
    ASSERT_STATUS_OK(model->SaveToOrtFormat(builder, fbs_model_offset));
    flatbuffers::Offset<fbs::InferenceSession> fbs_session_offset =
        fbs::CreateInferenceSessionDirect(builder,
                                          std::to_string(kOrtModelVersion).c_str(),
                                          fbs_model_offset,
                                          0);
    builder.Finish(fbs_session_offset);
  }

  // Load back from ORT format buffer
  {
    const auto* fbs_buffer = builder.GetBufferPointer();
    const auto* fbs_session = fbs::GetInferenceSession(fbs_buffer);
    ASSERT_NE(fbs_session, nullptr);
    ASSERT_NE(fbs_session->model(), nullptr);

    OrtFormatLoadOptions load_options;
    std::unique_ptr<Model> loaded_model;
    ASSERT_STATUS_OK(Model::LoadFromOrtFormat(*fbs_session->model(),
                                              nullptr,  // local_registries
                                              load_options,
                                              *logger_,
                                              loaded_model));

    // Verify the string initializer survived the ORT format round-trip
    const auto& loaded_graph = loaded_model->MainGraph();
    const ONNX_NAMESPACE::TensorProto* loaded_init = nullptr;
    ASSERT_TRUE(loaded_graph.GetInitializedTensor("string_data", loaded_init));
    ASSERT_EQ(loaded_init->string_data_size(), num_strings)
        << "String initializer data was lost during ORT format save/load round-trip";
    for (int i = 0; i < num_strings; ++i) {
      EXPECT_EQ(loaded_init->string_data(i), "value_" + std::to_string(i));
    }
  }
}

// Regression test for https://github.com/microsoft/onnxruntime/issues/28158
// Verifies that ToGraphProtoWithCustomInitializerHandling does not serialize
// _ORT_MEM_ADDR_ in-memory markers into the output model after
// ConvertInitializersIntoOrtValues has replaced large initializers.
TEST_F(GraphTest, CustomInitializerHandlingAfterConvertToOrtValues) {
  // Build a simple model with a large initializer (>127 bytes triggers conversion).
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(17);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Create a large initializer: 32 int64 values = 256 bytes (> 127 byte threshold).
  auto* initializer = graph_proto->add_initializer();
  initializer->set_name("large_init");
  initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  initializer->add_dims(32);
  for (int64_t i = 0; i < 32; ++i) {
    initializer->add_int64_data(i);
  }

  // Also add a small initializer (stays inline).
  auto* small_init = graph_proto->add_initializer();
  small_init->set_name("small_init");
  small_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  small_init->add_dims(1);
  small_init->add_int64_data(42);

  // Add node: output = input + large_init
  auto* add_node = graph_proto->add_node();
  add_node->set_op_type("Add");
  add_node->set_name("add_node");
  add_node->add_input("input");
  add_node->add_input("large_init");
  add_node->add_output("add_out");

  // Add node: output2 = add_out + small_init
  auto* add_node2 = graph_proto->add_node();
  add_node2->set_op_type("Add");
  add_node2->set_name("add_node2");
  add_node2->add_input("add_out");
  add_node2->add_input("small_init");
  add_node2->add_output("output");

  // Graph input
  auto* input = graph_proto->add_input();
  input->set_name("input");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  input_type->mutable_shape()->add_dim()->set_dim_value(32);

  // Graph output
  auto* output = graph_proto->add_output();
  output->set_name("output");

  // Load model and resolve
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Convert large initializers into OrtValues — this creates _ORT_MEM_ADDR_ markers.
  ASSERT_STATUS_OK(graph.ConvertInitializersIntoOrtValues());

  const ONNX_NAMESPACE::TensorProto* large_tp = nullptr;
  ASSERT_TRUE(graph.GetInitializedTensor("large_init", large_tp));
  ASSERT_TRUE(utils::HasExternalDataInMemory(*large_tp))
      << "large_init should have been externalized to in-memory OrtValue";

  // Serialize with a custom initializer handler that inlines everything.
  // Use a static function with ORT_API_CALL calling convention to match OrtGetInitializerLocationFunc.
  struct InlineAllHandler {
    static OrtStatus* ORT_API_CALL Func(void* /*state*/,
                                        const char* /*name*/,
                                        const OrtValue* /*value*/,
                                        const OrtExternalInitializerInfo* /*ext_info*/,
                                        OrtExternalInitializerInfo** new_ext_info) {
      *new_ext_info = nullptr;
      return nullptr;
    }
  };

  // Use Model::ToGraphProtoWithCustomInitializerHandling which is the real code path that triggers the bug.
  // It first copies model_proto_ (which contains the stale _ORT_MEM_ADDR_ initializers from
  // ConvertInitializersIntoOrtValues) into the output ModelProto, then calls the Graph-level function
  // on the pre-populated graph. Without the fix, the stale initializers remain alongside the newly
  // added ones, producing duplicates with _ORT_MEM_ADDR_ markers.
  ONNX_NAMESPACE::ModelProto output_model_proto;
  ASSERT_STATUS_OK(model->ToGraphProtoWithCustomInitializerHandling(InlineAllHandler::Func, nullptr,
                                                                    output_model_proto));

  const auto& output_graph = output_model_proto.graph();

  ASSERT_EQ(output_graph.input_size(), 1);
  ASSERT_EQ(output_graph.output_size(), 1);
  ASSERT_EQ(output_graph.node_size(), 2);

  // Verify: no initializer in the output should have _ORT_MEM_ADDR_ markers,
  // and there should be no duplicates.
  ASSERT_EQ(output_graph.initializer_size(), 2) << "Expected both initializers in output without duplication";

  size_t large_init_count = 0;
  size_t small_init_count = 0;
  for (const auto& init : output_graph.initializer()) {
    if (init.name() == "large_init") {
      ++large_init_count;
    } else if (init.name() == "small_init") {
      ++small_init_count;
    }
    EXPECT_FALSE(utils::HasExternalData(init))
        << "Initializer '" << init.name() << "' should be inline, not external (no _ORT_MEM_ADDR_)";
    EXPECT_TRUE(init.has_raw_data() || init.int64_data_size() > 0)
        << "Initializer '" << init.name() << "' should have data";
  }
  EXPECT_EQ(large_init_count, 1u) << "large_init should appear exactly once";
  EXPECT_EQ(small_init_count, 1u) << "small_init should appear exactly once";

  // Verify the large initializer data was correctly serialized.
  auto it = std::find_if(output_graph.initializer().begin(), output_graph.initializer().end(),
                         [](const TensorProto& tp) { return tp.name() == "large_init"; });
  ASSERT_NE(it, output_graph.initializer().end());
  // The data should be in raw_data (SetRawDataInTensorProto writes to raw_data).
  ASSERT_GT(it->raw_data().size(), 0u) << "large_init should have raw_data after inlining";
  ASSERT_EQ(it->raw_data().size(), 32 * sizeof(int64_t));
}

// Test that a malformed model with an initializer that has no corresponding NodeArg
// is properly rejected during graph loading.
TEST_F(GraphTest, MalformedModelInitializerWithoutNodeArg) {
  // Construct a model proto manually with ir_version < 4 where an initializer
  // has no matching graph input (and therefore no NodeArg is created for it).
  // The graph output references the orphaned initializer, but the output proto
  // entry lacks type info so no NodeArg is created via the output loop either.
  ModelProto model_proto;
  model_proto.set_ir_version(3);  // ir_version < 4 requires initializers to be graph inputs
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Add a valid graph input with type
  auto* input = graph_proto->add_input();
  input->set_name("X");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(2);

  // Add a node
  auto* node = graph_proto->add_node();
  node->add_input("X");
  node->add_output("Y");
  node->set_op_type("Identity");

  // Add a normal graph output with type
  auto* output = graph_proto->add_output();
  output->set_name("Y");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(2);

  // Add an orphaned initializer (no matching graph input for ir_version < 4).
  // This means no NodeArg is created for "orphan_init" in the constructor.
  auto* initializer = graph_proto->add_initializer();
  initializer->set_name("orphan_init");
  initializer->set_data_type(TensorProto_DataType_FLOAT);
  initializer->add_dims(2);
  initializer->add_float_data(1.0f);
  initializer->add_float_data(2.0f);

  // Add a second output referencing the orphaned initializer.
  // Intentionally omit type so that no NodeArg is created for it via the output loop.
  auto* output2 = graph_proto->add_output();
  output2->set_name("orphan_init");
  // No type set - this means GetOrCreateNodeArg won't be called for this output

  // Loading this model should fail with a proper error (not a crash)
  std::shared_ptr<Model> model;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      Model::Load(std::move(model_proto), model, nullptr, *logger_),
      "orphan_init");
}

// Test that a model with an If node containing empty subgraphs
// (present but empty GraphProto) is properly rejected.
TEST_F(GraphTest, MalformedModelEmptySubgraph) {
  // Construct a model with an If node whose then_branch and else_branch
  // subgraphs are present but completely empty (no inputs, no outputs, no nodes).
  ModelProto model_proto;
  model_proto.set_ir_version(9);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_graph");

  // Add a boolean condition input for the If node
  auto* cond_input = graph_proto->add_input();
  cond_input->set_name("cond");
  auto* cond_type = cond_input->mutable_type()->mutable_tensor_type();
  cond_type->set_elem_type(TensorProto_DataType_BOOL);
  cond_type->mutable_shape();  // scalar

  // Add an If node with empty then_branch and else_branch subgraphs
  auto* node = graph_proto->add_node();
  node->add_input("cond");
  node->add_output("Y");
  node->set_op_type("If");

  // Add then_branch attribute with an empty GraphProto
  auto* then_attr = node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto_AttributeType_GRAPH);
  then_attr->mutable_g();  // allocate empty GraphProto

  // Add else_branch attribute with an empty GraphProto
  auto* else_attr = node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto_AttributeType_GRAPH);
  else_attr->mutable_g();  // allocate empty GraphProto

  // Add graph output
  auto* output = graph_proto->add_output();
  output->set_name("Y");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(2);

  // Loading the model should fail gracefully — not crash.
  // The empty subgraphs have no outputs, which violates the If op spec.
  std::shared_ptr<Model> model;
  auto status = Model::Load(std::move(model_proto), model, nullptr, *logger_);
  EXPECT_FALSE(status.IsOK());
}

// Test for GitHub issue #24880: subgraph referencing an initializer from the outer graph
// should resolve successfully even without explicit value_info in the subgraph.
// Uses a custom op with a GRAPH attribute but NO type inference function, so subgraph
// type inferencing is never invoked. This directly exercises the verification-time
// propagation in VerifyNodeAndOpMatch.
TEST_F(GraphTest, OuterScopeInitializerTypeInfoPropagatedToSubgraph) {
  // Register a custom op with a GRAPH attribute but no type/shape inference function.
  // This simulates ops like BeamSearch whose schema inference does not invoke subgraph
  // inferencing, so InferAndVerifySubgraphTypes is never called during type inference.
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry =
      std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {
      OpSchema()
          .SetName("FakeSubgraphOp")
          .SetDomain("FakeTestDomain")
          .Input(0, "X", "Input tensor", "T")
          .Output(0, "Y", "Output tensor", "T")
          .Attr("body", "A subgraph", AttributeProto::GRAPH)
          .TypeConstraint("T", OpSchema::all_tensor_types(),
                          "Constrain input and output types to any tensor type.")};
  ASSERT_TRUE(registry->RegisterOpSet(schema, "FakeTestDomain", 0, 1).IsOK());

  // Build the model proto.
  // Main graph: float input "x" [2,3], float initializer "weight" [2,3],
  //             FakeSubgraphOp node with a subgraph that references "weight".
  // The subgraph uses "weight" via implicit capture (no value_info for it in the subgraph).
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  ImportOpset(model_proto, "", 16);
  ImportOpset(model_proto, "FakeTestDomain", 1);

  GraphProto& main_graph = *model_proto.mutable_graph();
  main_graph.set_name("main_graph");

  // Main graph input: float tensor "x"
  ValueInfoProto* x_input = main_graph.add_input();
  x_input->set_name("x");
  SetTypeAndShape(x_input->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Initializer "weight" in the main graph (float [2,3]).
  // Not added as a graph input — ORT relaxes the ONNX requirement that initializers
  // must also be listed as graph inputs.
  TensorProto* weight_init = main_graph.add_initializer();
  weight_init->set_name("weight");
  weight_init->set_data_type(TensorProto_DataType_FLOAT);
  weight_init->add_dims(2);
  weight_init->add_dims(3);
  for (int i = 0; i < 6; ++i) {
    weight_init->add_float_data(static_cast<float>(i));
  }

  // Build subgraph: Identity(weight) -> result
  // "weight" is from the outer scope — deliberately NO value_info for it in this subgraph.
  // It will be picked up as an implicit input by BuildConnections.
  GraphProto subgraph;
  subgraph.set_name("body_subgraph");

  // Subgraph output
  ValueInfoProto* sg_output = subgraph.add_output();
  sg_output->set_name("result");
  SetTypeAndShape(sg_output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Identity node: result = Identity(weight)
  NodeProto* identity_node = subgraph.add_node();
  identity_node->set_op_type("Identity");
  *identity_node->add_input() = "weight";  // outer scope initializer
  *identity_node->add_output() = "result";

  // FakeSubgraphOp node: takes "x" as input, has "body" subgraph attribute
  NodeProto* fake_node = main_graph.add_node();
  fake_node->set_op_type("FakeSubgraphOp");
  fake_node->set_domain("FakeTestDomain");
  fake_node->set_name("fake_subgraph_node");
  *fake_node->add_input() = "x";
  *fake_node->add_output() = "y";

  AttributeProto* body_attr = fake_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto_AttributeType_GRAPH);
  *body_attr->mutable_g() = subgraph;

  // Main graph output
  ValueInfoProto* main_output = main_graph.add_output();
  main_output->set_name("y");
  SetTypeAndShape(main_output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Load the model — this calls Graph::Resolve internally.
  // Without the fix, this fails with:
  //   "input arg (weight) does not have type information set by parent node"
  // because FakeSubgraphOp has no type inference that propagates types to the subgraph.
  // The fix propagates type info from implicit_input_defs during VerifyNodeAndOpMatch.
  std::shared_ptr<Model> model;
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {registry};
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, &regs, *logger_));
}

// A locally produced value captured by a nested graph must be serialized as value_info. Otherwise,
// reloading cannot recover its type when the producer's schema has no type-inference function.
TEST_F(GraphTest, LocalImplicitInputTypeInfoSurvivesSerialization) {
  auto registry = std::make_shared<onnxruntime::OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas = {
      OpSchema()
          .SetName("NoInferProducer")
          .SetDomain("FakeTestDomain")
          .Output(0, "Y", "Output whose type is supplied by the graph", "T")
          .TypeConstraint("T", OpSchema::all_tensor_types(), "Any tensor type")};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, "FakeTestDomain", 0, 1));

  IOnnxRuntimeOpSchemaRegistryList registries = {registry};
  Model model("local_capture", false, ModelMetaData(), PathString(), registries,
              {{kOnnxDomain, 13}, {"FakeTestDomain", 1}}, {}, *logger_);
  Graph& graph = model.MainGraph();

  TypeProto float_tensor;
  SetTypeAndShape(float_tensor.mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});
  TypeProto bool_scalar;
  SetTypeAndShape(bool_scalar.mutable_tensor_type(), TensorProto_DataType_BOOL, {});

  NodeArg& captured = graph.GetOrCreateNodeArg("captured", &float_tensor);
  graph.AddNode("producer", "NoInferProducer", "Producer without schema inference", {}, {&captured},
                nullptr, "FakeTestDomain");

  NodeArg& cond = graph.GetOrCreateNodeArg("cond", &bool_scalar);
  NodeArg& if_output = graph.GetOrCreateNodeArg("if_output", &float_tensor);

  auto make_branch = [](const std::string& graph_name, const std::string& output_name) {
    GraphProto branch;
    branch.set_name(graph_name);

    auto* output = branch.add_output();
    output->set_name(output_name);
    SetTypeAndShape(output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

    auto* identity = branch.add_node();
    identity->set_name(graph_name + "_identity");
    identity->set_op_type("Identity");
    identity->add_input("captured");
    identity->add_output(output_name);
    return branch;
  };

  NodeAttributes attributes;
  attributes.emplace("then_branch", utils::MakeAttribute("then_branch", make_branch("then_branch", "then_out")));
  attributes.emplace("else_branch", utils::MakeAttribute("else_branch", make_branch("else_branch", "else_out")));
  graph.AddNode("if_node", "If", "Captures a local producer output", {&cond}, {&if_output}, &attributes);
  graph.SetInputs({&cond});
  graph.SetOutputs({&if_output});

  ASSERT_STATUS_OK(graph.Resolve());

  ModelProto serialized = model.ToProto();
  const auto captured_value_info = std::find_if(
      serialized.graph().value_info().cbegin(), serialized.graph().value_info().cend(),
      [](const ValueInfoProto& value_info) { return value_info.name() == "captured"; });
  ASSERT_NE(captured_value_info, serialized.graph().value_info().cend());
  EXPECT_EQ(captured_value_info->type().tensor_type().elem_type(), TensorProto_DataType_FLOAT);

  std::shared_ptr<Model> reloaded_model;
  ASSERT_STATUS_OK(Model::Load(std::move(serialized), reloaded_model, &registries, *logger_));
}

// Negative companion to OuterScopeInitializerTypeInfoPropagatedToSubgraph.
// The subgraph declares a value_info for the outer-scope initializer "weight"
// with a conflicting element type (INT64 vs FLOAT).  Model::Load must fail and
// the error message must identify the type conflict.
TEST_F(GraphTest, OuterScopeInitializerConflictingTypeFails) {
  // Same custom-op registration as the positive test.
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry =
      std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {
      OpSchema()
          .SetName("FakeSubgraphOp2")
          .SetDomain("FakeTestDomain")
          .Input(0, "X", "Input tensor", "T")
          .Output(0, "Y", "Output tensor", "T")
          .Attr("body", "A subgraph", AttributeProto::GRAPH)
          .TypeConstraint("T", OpSchema::all_tensor_types(),
                          "Constrain input and output types to any tensor type.")};
  ASSERT_TRUE(registry->RegisterOpSet(schema, "FakeTestDomain", 0, 1).IsOK());

  // Build the model proto — identical outer graph to the positive test.
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  ImportOpset(model_proto, "", 16);
  ImportOpset(model_proto, "FakeTestDomain", 1);

  GraphProto& main_graph = *model_proto.mutable_graph();
  main_graph.set_name("main_graph");

  // Main graph input: float tensor "x" [2,3]
  ValueInfoProto* x_input = main_graph.add_input();
  x_input->set_name("x");
  SetTypeAndShape(x_input->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Initializer "weight" in the main graph: FLOAT [2,3].
  TensorProto* weight_init = main_graph.add_initializer();
  weight_init->set_name("weight");
  weight_init->set_data_type(TensorProto_DataType_FLOAT);
  weight_init->add_dims(2);
  weight_init->add_dims(3);
  for (int i = 0; i < 6; ++i) {
    weight_init->add_float_data(static_cast<float>(i));
  }

  // Build subgraph: Identity(weight) -> result.
  // Unlike the positive test, we add a value_info for "weight" that declares
  // it as INT64 [2,3] — conflicting with the outer-scope FLOAT type.
  GraphProto subgraph;
  subgraph.set_name("body_subgraph");

  // Conflicting value_info: "weight" as INT64 [2,3] instead of FLOAT [2,3].
  ValueInfoProto* weight_vi = subgraph.add_value_info();
  weight_vi->set_name("weight");
  SetTypeAndShape(weight_vi->mutable_type()->mutable_tensor_type(), TensorProto_DataType_INT64, {2, 3});

  // Subgraph output
  ValueInfoProto* sg_output = subgraph.add_output();
  sg_output->set_name("result");
  SetTypeAndShape(sg_output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Identity node: result = Identity(weight)
  NodeProto* identity_node = subgraph.add_node();
  identity_node->set_op_type("Identity");
  *identity_node->add_input() = "weight";
  *identity_node->add_output() = "result";

  // FakeSubgraphOp2 node
  NodeProto* fake_node = main_graph.add_node();
  fake_node->set_op_type("FakeSubgraphOp2");
  fake_node->set_domain("FakeTestDomain");
  fake_node->set_name("fake_subgraph_node");
  *fake_node->add_input() = "x";
  *fake_node->add_output() = "y";

  AttributeProto* body_attr = fake_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto_AttributeType_GRAPH);
  *body_attr->mutable_g() = subgraph;

  // Main graph output
  ValueInfoProto* main_output = main_graph.add_output();
  main_output->set_name("y");
  SetTypeAndShape(main_output->mutable_type()->mutable_tensor_type(), TensorProto_DataType_FLOAT, {2, 3});

  // Model::Load must fail because the subgraph declares "weight" as INT64
  // while the outer-scope initializer is FLOAT.  The strict UpdateTypeAndShape
  // call in VerifyNodeAndOpMatch returns "Tensor element type mismatch" which
  // is then wrapped with the subgraph context "[subgraph:body]".
  std::shared_ptr<Model> model;
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {registry};
  auto status = Model::Load(std::move(model_proto), model, &regs, *logger_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Tensor element type mismatch"));
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("[subgraph:body]"));
}

// Regression test for https://github.com/microsoft/onnxruntime/issues/29071
// InlineIfSubgraph must transfer OrtValues for initializers with in-memory external data
// from the subgraph to the parent graph. Previously, the OrtValue lookup always failed
// because name_to_initial_tensor_ was erased before GetOrtValueInitializer was called.
TEST_F(GraphTest, InlineIfSubgraphTransfersOrtValues) {
  // Build a model with an If node whose condition is a constant initializer (true).
  // The "then" subgraph contains a large initializer (>127 bytes, triggering
  // ConvertInitializersIntoOrtValues). When the If is constant-folded via
  // InlineIfSubgraph, the large initializer must be transferred with its OrtValue.
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_version(13);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("test_inline_if");

  // Constant condition = true
  auto* cond_init = graph_proto->add_initializer();
  cond_init->set_name("cond");
  cond_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  cond_init->add_int32_data(1);  // true

  // If node
  auto* if_node = graph_proto->add_node();
  if_node->set_op_type("If");
  if_node->set_name("if_node");
  if_node->add_input("cond");
  if_node->add_output("if_out");

  // Identity node after If to avoid naming conflicts during inlining
  auto* id_node = graph_proto->add_node();
  id_node->set_op_type("Identity");
  id_node->set_name("post_if_identity");
  id_node->add_input("if_out");
  id_node->add_output("output");

  // Helper to set type+shape on a ValueInfoProto's tensor_type
  auto SetTypeAndShape = [](auto* tensor_type, int32_t elem_type, std::initializer_list<int64_t> dims) {
    tensor_type->set_elem_type(elem_type);
    auto* shape = tensor_type->mutable_shape();
    for (auto d : dims) {
      shape->add_dim()->set_dim_value(d);
    }
  };

  // === then_branch subgraph ===
  auto* then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* then_graph = then_attr->mutable_g();
  then_graph->set_name("then_graph");

  // Large initializer in the then branch: 32 float values = 128 bytes (> 127 byte threshold)
  auto* large_init = then_graph->add_initializer();
  large_init->set_name("large_weight");
  large_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  large_init->add_dims(32);
  for (int i = 0; i < 32; ++i) {
    large_init->add_float_data(static_cast<float>(i));
  }

  // Identity node in the then branch: output = Identity(large_weight)
  auto* then_id_node = then_graph->add_node();
  then_id_node->set_op_type("Identity");
  then_id_node->add_input("large_weight");
  then_id_node->add_output("then_out");

  auto* then_output = then_graph->add_output();
  then_output->set_name("then_out");
  SetTypeAndShape(then_output->mutable_type()->mutable_tensor_type(),
                  TensorProto_DataType_FLOAT, {32});

  // === else_branch subgraph (required by schema, but won't be taken) ===
  auto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto* else_graph = else_attr->mutable_g();
  else_graph->set_name("else_graph");

  // Large initializer in the else branch (won't be used since condition is true)
  auto* else_init = else_graph->add_initializer();
  else_init->set_name("else_weight");
  else_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  else_init->add_dims(32);
  for (int i = 0; i < 32; ++i) {
    else_init->add_float_data(0.0f);
  }

  auto* else_id_node = else_graph->add_node();
  else_id_node->set_op_type("Identity");
  else_id_node->add_input("else_weight");
  else_id_node->add_output("else_out");

  auto* else_output = else_graph->add_output();
  else_output->set_name("else_out");
  SetTypeAndShape(else_output->mutable_type()->mutable_tensor_type(),
                  TensorProto_DataType_FLOAT, {32});

  // Graph output
  auto* output = graph_proto->add_output();
  output->set_name("output");
  SetTypeAndShape(output->mutable_type()->mutable_tensor_type(),
                  TensorProto_DataType_FLOAT, {32});

  // Load and resolve
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Convert initializers to OrtValues (large ones get in-memory external data markers).
  // This processes subgraphs too, so the then_branch's large_weight becomes external-in-memory.
  ASSERT_STATUS_OK(graph.ConvertInitializersIntoOrtValues());

  // Now inline the If subgraph (simulating what constant folding does).
  // Before the fix, this would leave the large initializer with HasExternalDataInMemory==true
  // but no cached OrtValue, causing a crash in SaveInitializedTensors.
  Node* if_node_ptr = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "If") {
      if_node_ptr = &node;
      break;
    }
  }
  ASSERT_NE(if_node_ptr, nullptr) << "If node should exist before inlining";

  ASSERT_STATUS_OK(graph.InlineIfSubgraph(/*condition_value=*/true, *if_node_ptr, *logger_));

  // Remove the If node (same as ConstantFoldIfNode does after inlining)
  graph_utils::RemoveNodeOutputEdges(graph, *if_node_ptr);
  graph.RemoveNode(if_node_ptr->Index());

  ASSERT_STATUS_OK(graph.Resolve());

  // Verify: the inlined initializer with in-memory external data must have a cached OrtValue
  for (const auto& [name, tp] : graph.GetAllInitializedTensors()) {
    if (utils::HasExternalDataInMemory(*tp)) {
      OrtValue ort_value;
      EXPECT_TRUE(graph.GetOrtValueInitializer(name, ort_value))
          << "Initializer '" << name << "' has external data in memory but no cached OrtValue";
      EXPECT_TRUE(ort_value.IsAllocated())
          << "OrtValue for '" << name << "' should be allocated";
    }
  }
}

// End-to-end regression test for https://github.com/microsoft/onnxruntime/issues/29071
// Loads the customer's repro model (which has an If node and triggers ConstantFolding)
// and verifies that session initialization succeeds.
// The model is generated by testdata/gh_issue_29071_if_constant_folding.py.
TEST_F(GraphTest, GH_Issue_29071_HasExternalDataInMemory) {
  SessionOptions so;
  so.session_logid = "GraphTest.GH_Issue_29071_HasExternalDataInMemory";

  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(ORT_TSTR("testdata/gh_issue_29071_if_constant_folding.onnx")));
  ASSERT_STATUS_OK(session_object.Initialize());
}

// Regression test for exponential subgraph type/shape inferencing.
// A model with deeply nested Loop nodes (each subgraph containing a single nested Loop) previously
// triggered O(2^depth) re-traversal during Graph::Resolve because both InferAndVerifyTypeMatch and
// the "verify subgraphs" loop in VerifyNodeAndOpMatch independently recursed into every subgraph.
// With 30 levels that made model loading take many minutes/hours. The fix memoizes subgraphs that
// already had type/shape inferencing performed, collapsing the work back to O(depth). This test
// simply verifies the model loads well within a generous time bound.
TEST_F(GraphTest, DeeplyNestedLoopSubgraphsResolveInReasonableTime) {
  const auto start = std::chrono::steady_clock::now();

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/30_nested_loops.onnx"), model, nullptr, *logger_));

  const auto elapsed = std::chrono::steady_clock::now() - start;
  const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

  // Without the memoization fix this takes many minutes (exponential in the nesting depth).
  // A generous 60s bound reliably distinguishes the fixed O(depth) behavior from the regression
  // without being flaky on slow/debug builds.
  EXPECT_LT(elapsed_seconds, 60) << "Loading the 30-level nested Loop model took " << elapsed_seconds
                                 << "s, which suggests the subgraph type/shape inferencing recursion "
                                    "regression has returned.";
}

static ModelProto CreateNestedSubgraphModel(size_t depth) {
  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(21);

  auto* graph = model_proto.mutable_graph();
  for (size_t i = 0; i < depth; ++i) {
    auto* node = graph->add_node();
    auto* attr = node->add_attribute();
    graph = attr->mutable_g();
  }

  return model_proto;
}

TEST_F(GraphTest, ExcessiveSubgraphDepthRejected) {
  auto model_proto = CreateNestedSubgraphModel(kMaxModelSubgraphDepth + 1);
  std::shared_ptr<Model> model;
  const auto status = Model::Load(std::move(model_proto), model, nullptr, *logger_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

TEST_F(GraphTest, ExcessiveSubgraphDepthRejectedFromLvalueProto) {
  const auto model_proto = CreateNestedSubgraphModel(kMaxModelSubgraphDepth + 1);
  std::shared_ptr<Model> model;
  const auto status = Model::Load(model_proto, model, nullptr, *logger_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::NOT_IMPLEMENTED);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("exceeds the maximum supported depth"));
}

}  // namespace test
}  // namespace onnxruntime
