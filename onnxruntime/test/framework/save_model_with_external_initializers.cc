// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/data_types.h"
#include "core/graph/model.h"
#include "core/graph/model_saving_options.h"
#include "core/framework/tensorprotoutils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime;

namespace onnxruntime {
namespace test {

Status LoadSaveAndCompareModel(const std::filesystem::path& input_onnx,
                               const std::filesystem::path& input_external_init_file,
                               const std::filesystem::path& output_onnx,
                               const std::filesystem::path& output_external_init_file,
                               const ModelSavingOptions& model_saving_options) {
  auto logger = DefaultLoggingManager().CreateLogger("LoadSaveAndCompareModel");
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(input_onnx, model, nullptr, *logger));
  std::filesystem::remove(output_onnx);
  std::filesystem::remove(output_external_init_file);
  ORT_RETURN_IF_ERROR(Model::SaveWithExternalInitializers(*model, output_onnx, output_external_init_file,
                                                          model_saving_options));

  std::shared_ptr<Model> model_from_external;
  ORT_RETURN_IF_ERROR(Model::Load(output_onnx.native(), model_from_external, nullptr, *logger));

  Graph& graph = model->MainGraph();
  // Perform shape inference on the graph, if this succeeds then it means that we could correctly read the
  // integer initializers used by reshape and transpose.
  ORT_RETURN_IF_ERROR(graph.Resolve());
  Graph& graph_from_external = model_from_external->MainGraph();

  InitializedTensorSet initializers = graph.GetAllInitializedTensors();
  InitializedTensorSet initializers_from_external = graph_from_external.GetAllInitializedTensors();

  ORT_RETURN_IF_NOT(initializers.size() == initializers_from_external.size(), "size mismatch");

  // Compare the initializers of the two versions.
  std::filesystem::path model_path{};
  std::filesystem::path external_data_path{};
  for (const auto& i : initializers) {
    const std::string kInitName = i.first;
    const ONNX_NAMESPACE::TensorProto* tensor_proto = i.second;
    const ONNX_NAMESPACE::TensorProto* from_external_tensor_proto = initializers_from_external[kInitName];

    std::vector<uint8_t> tensor_proto_data;
    model_path = input_onnx;
    external_data_path = (!input_external_init_file.empty()) ? (model_path.parent_path() / input_external_init_file) : std::filesystem::path();
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*tensor_proto, external_data_path, tensor_proto_data));
    size_t tensor_proto_size = tensor_proto_data.size();

    std::vector<uint8_t> from_external_tensor_proto_data;
    model_path = output_onnx;
    external_data_path = model_path.parent_path() / output_external_init_file;
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*from_external_tensor_proto, model_path, from_external_tensor_proto_data));
    size_t from_external_tensor_proto_size = from_external_tensor_proto_data.size();

    if (from_external_tensor_proto_size < model_saving_options.initializer_size_threshold) {
      // 'Small' tensors should be embedded in the onnx file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_DEFAULT, "location mismatch");
    } else {
      // 'Large' tensors should be added to the external binary file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL, "location mismatch");
    }

    ORT_RETURN_IF_NOT(tensor_proto_size == from_external_tensor_proto_size, "size mismatch");
    ORT_RETURN_IF_NOT(memcmp(tensor_proto_data.data(), from_external_tensor_proto_data.data(), tensor_proto_size) == 0, "data mismatch");

    if (model_saving_options.align_offset) {
      for (const StringStringEntryProto& entry : from_external_tensor_proto->external_data()) {
        if (entry.has_key() && entry.has_value() && entry.key() == "offset") {
          size_t tensor_offset;
          std::stringstream stream(entry.value());
          stream >> tensor_offset;
          ORT_RETURN_IF_NOT(tensor_offset % model_saving_options.on_disk_alignment == 0,
                            "tensor offset not align");
        }
      }
    }
  }
  // Cleanup.
  ORT_RETURN_IF_NOT(std::filesystem::remove(output_onnx), "delete file failed");
  ORT_RETURN_IF_NOT(std::filesystem::remove(external_data_path), "delete file failed");
  return Status::OK();
}

// Original model does not have external initializers
TEST(SaveWithExternalInitializers, Mnist) {
  ModelSavingOptions model_saving_options{100};
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/mnist.onnx"),
      ORT_TSTR(""), ORT_TSTR("testdata/mnist_with_external_initializers.onnx"),
      ORT_TSTR("mnist_external_initializers.bin"),
      model_saving_options));
}

// Original model has external initializers
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalData) {
  ModelSavingOptions model_saving_options{0};
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/model_with_orig_ext_data.onnx"),
      ORT_TSTR("model_with_orig_ext_data.onnx.data"),
      ORT_TSTR("testdata/model_with_new_external_initializers.onnx"),
      ORT_TSTR("model_with_new_external_initializers.bin"),
      model_saving_options));
}

// Original model has external initializers, align offset
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalDataAlignOffset) {
  ModelSavingOptions model_saving_options{0};
  model_saving_options.align_offset = true;
  model_saving_options.align_threshold = 0;
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/model_with_orig_ext_data.onnx"),
      ORT_TSTR("model_with_orig_ext_data.onnx.data"),
      ORT_TSTR("testdata/model_with_new_external_initializers.onnx"),
      ORT_TSTR("model_with_new_external_initializers.bin"), model_saving_options));
}

TEST(SaveWithExternalInitializers, NestedIfInitializersAreWrittenOnce) {
  const auto set_type_and_shape = [](ValueInfoProto& value_info, int32_t element_type,
                                     std::initializer_list<int64_t> dimensions) {
    auto* tensor_type = value_info.mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    for (int64_t dimension : dimensions) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
  };

  ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain(onnxruntime::kOnnxDomain);
  opset->set_version(13);

  GraphProto& graph = *model_proto.mutable_graph();
  graph.set_name("nested_external_initializers");

  auto* cond = graph.add_input();
  cond->set_name("cond");
  set_type_and_shape(*cond, TensorProto_DataType_BOOL, {});

  auto* output = graph.add_output();
  output->set_name("Y");
  set_type_and_shape(*output, TensorProto_DataType_FLOAT, {2});

  auto make_branch = [&set_type_and_shape](const std::string& graph_name, const std::string& initializer_name,
                                           const std::string& output_name) {
    GraphProto branch;
    branch.set_name(graph_name);

    auto* initializer = branch.add_initializer();
    initializer->set_name(initializer_name);
    initializer->set_data_type(TensorProto_DataType_FLOAT);
    initializer->add_dims(2);
    initializer->add_float_data(1.0f);
    initializer->add_float_data(2.0f);

    auto* branch_output = branch.add_output();
    branch_output->set_name(output_name);
    set_type_and_shape(*branch_output, TensorProto_DataType_FLOAT, {2});

    auto* identity = branch.add_node();
    identity->set_name(graph_name + "_identity");
    identity->set_op_type("Identity");
    identity->add_input(initializer_name);
    identity->add_output(output_name);
    return branch;
  };

  auto* if_node = graph.add_node();
  if_node->set_name("if_node");
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("Y");

  auto* then_attribute = if_node->add_attribute();
  then_attribute->set_name("then_branch");
  then_attribute->set_type(AttributeProto_AttributeType_GRAPH);
  *then_attribute->mutable_g() = make_branch("then_branch", "then_weight", "then_out");

  auto* else_attribute = if_node->add_attribute();
  else_attribute->set_name("else_branch");
  else_attribute->set_type(AttributeProto_AttributeType_GRAPH);
  *else_attribute->mutable_g() = make_branch("else_branch", "else_weight", "else_out");

  auto logger = DefaultLoggingManager().CreateLogger("NestedIfInitializersAreWrittenOnce");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, *logger));

  TemporaryDirectory temp_dir{ORT_TSTR("nested_external_initializer_test")};
  const std::filesystem::path model_path =
      std::filesystem::path(temp_dir.Path()) / ORT_TSTR("nested_initializers.onnx");
  const std::filesystem::path external_file_name = ORT_TSTR("nested_initializers.bin");
  ModelSavingOptions save_options{/*initializer_size_threshold=*/0};
  ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(*model, model_path, external_file_name, save_options));

  ModelProto saved_proto;
  ASSERT_STATUS_OK(Model::Load(model_path.native(), saved_proto));
  ASSERT_EQ(saved_proto.graph().node_size(), 1);

  const NodeProto& saved_if = saved_proto.graph().node(0);
  ASSERT_EQ(saved_if.attribute_size(), 2);
  for (const auto& attribute : saved_if.attribute()) {
    ASSERT_EQ(attribute.type(), AttributeProto_AttributeType_GRAPH);
    ASSERT_EQ(attribute.g().initializer_size(), 1) << attribute.name();
    const TensorProto& initializer = attribute.g().initializer(0);
    EXPECT_EQ(initializer.data_location(), TensorProto_DataLocation_EXTERNAL) << attribute.name();
    EXPECT_GT(initializer.external_data_size(), 0) << attribute.name();
  }

  std::shared_ptr<Model> reloaded_model;
  ASSERT_STATUS_OK(Model::Load(model_path.native(), reloaded_model, nullptr, *logger));
}

TEST(SaveWithExternalInitializers, Float6PackedData) {
  auto logger = DefaultLoggingManager().CreateLogger("Float6PackedData");
  TemporaryDirectory temp_dir{ORT_TSTR("float6_external_save_test")};
  for (const auto type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    const auto input_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("input.bin");
    const auto model_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("float6.onnx");
    const auto external_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("float6.bin");
    {
      std::ofstream input(input_path, std::ios::binary);
      input.write("\x01\x42\xc3", 3);
    }
    ModelProto source;
    source.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    source.mutable_graph()->set_name("float6_external_save");
    source.add_opset_import()->set_version(28);
    auto* initializer = source.mutable_graph()->add_initializer();
    initializer->set_name("float6");
    initializer->set_data_type(type);
    initializer->add_dims(4);
    ExternalDataInfo::SetExternalLocationToProto("input.bin", 0, 3, *initializer);
    auto* output = source.mutable_graph()->add_output();
    output->set_name("float6");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(type);
    output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(std::move(source), model_path.native(), model, nullptr, *logger,
                                 ModelOptions(false, false)));
    for (const bool embed : {false, true}) {
      ModelSavingOptions options{0};
      options.force_embed_external_ini = embed;
      ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(*model, model_path, ORT_TSTR("float6.bin"), options));
      ModelProto saved;
      ASSERT_STATUS_OK(Model::Load(model_path.native(), saved));
      ASSERT_EQ(saved.graph().initializer_size(), 1);
      const auto& result = saved.graph().initializer(0);
      std::string written;
      if (embed) {
        EXPECT_EQ(result.data_location(), TensorProto_DataLocation_DEFAULT);
        written = result.raw_data();
      } else {
        EXPECT_EQ(result.data_location(), TensorProto_DataLocation_EXTERNAL);
        std::ifstream stream(external_path, std::ios::binary);
        written.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
      }
      EXPECT_EQ(written, std::string("\x01\x42\xc3", 3));
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
