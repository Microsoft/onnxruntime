// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Coverage for com.microsoft.PackedSparseAttentionIndexer.
//
// The "ShapeInference" suite drives Graph::Resolve() directly, so it runs in every build: it pins
// the fixed selected_indices/selected_counts shapes, the fixed (never data-dependent) state output
// shapes, and the strict policy validation. Cases that are expected to fail shape inference call
// fail_shape_inference, which aborts in ORT_NO_EXCEPTIONS builds, so they are compiled out there.
//
// The numeric suite needs the CUDA or WebGPU execution provider (the operator has no CPU kernel)
// and is skipped when neither is available. Expectations come from a float reference in this file
// that mirrors the operator contract (mean-pool/RMSNorm/RoPE for qsa, softmax-gated pooling for
// csa, deterministic top-k selection); the inputs are rounded to the tested element type first so
// the reference sees exactly what the kernel reads.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

namespace psai = ::onnxruntime::contrib::packed_sparse_attention_indexer;

namespace {

constexpr int kOnnxOpsetVersion = 17;

// ---------------------------------------------------------------------------------------------
// Shape inference helpers
// ---------------------------------------------------------------------------------------------

Status BuildAndResolve(const std::function<void(ModelTestBuilder& builder)>& add_node,
                       std::unique_ptr<Model>& model) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = kOnnxOpsetVersion;
  domain_to_version[kMSDomain] = 1;

  model = std::unique_ptr<Model>(new Model("packed_sparse_attention_indexer", /*is_onnx_domain_only=*/false,
                                           ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                                           domain_to_version, {}, DefaultLoggingManager().DefaultLogger()));

  ModelTestBuilder builder(model->MainGraph());
  add_node(builder);
  builder.SetGraphOutputs();
  return model->MainGraph().Resolve();
}

void ExpectShape(const Graph& graph, const std::string& name, ONNX_NAMESPACE::TensorProto_DataType elem_type,
                 const std::vector<int64_t>& expected) {
  const NodeArg* arg = graph.GetNodeArg(name);
  ASSERT_NE(arg, nullptr);
  const ONNX_NAMESPACE::TypeProto* type = arg->TypeAsProto();
  ASSERT_NE(type, nullptr);
  ASSERT_TRUE(type->has_tensor_type());
  EXPECT_EQ(type->tensor_type().elem_type(), static_cast<int32_t>(elem_type));
  const ONNX_NAMESPACE::TensorShapeProto& shape = type->tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), static_cast<int>(expected.size()));
  for (int i = 0; i < shape.dim_size(); ++i) {
    ASSERT_TRUE(shape.dim(i).has_dim_value()) << "dimension " << i << " of " << name << " is not static";
    EXPECT_EQ(shape.dim(i).dim_value(), expected[static_cast<size_t>(i)]) << "dimension " << i << " of " << name;
  }
}

struct GraphOptions {
  int64_t batch_size = 2;
  int64_t total_tokens = 5;
  int64_t num_heads = 2;
  int64_t head_size = 8;
  int64_t rotary_width = 4;
  int64_t compress_ratio = 2;
  int64_t state_capacity = 6;
  int64_t input_state_capacity = -1;
  int64_t token_budget = 4;
  int64_t key_total_tokens = -1;
  int64_t kv_buffer_capacity = -1;
  int64_t gate_buffer_width = -1;
  bool packed_qk = false;
  bool add_index_topk = false;
  bool add_csa_inputs = false;
  bool add_position_ids = false;
  bool add_capture_count = false;
  bool add_state_update_active = false;
  int64_t state_update_capacity = 0;
  bool add_qsa_gate_output = false;
  bool omit_csa_gate_output = false;
  int output_count = psai::kFixedOutputCount;
  std::string policy_mode = psai::kPolicyModeQsa;
  bool indexshare = false;
  std::string indexshare_state_policy = "default";
  std::vector<int64_t> mode_shape{1};
  int64_t merged_rows = 0;
  int64_t merged_width = 8;
  int64_t merged_counts_rows = -1;
  int64_t merged_status_rows = -1;
  int64_t projection_tokens = -1;
};

int64_t BufferCapacity(int64_t compress_ratio) { return 2 * compress_ratio - 1; }

// Builds a fixed-input node; csa-only slots are left empty for policy_mode "qsa", as the schema
// requires. position_ids (slot 11) is optional for "qsa" and forced on for "csa".
void AddNode(ModelTestBuilder& builder, const GraphOptions& options) {
  const bool is_csa = options.policy_mode == psai::kPolicyModeCsa;
  const int64_t width = is_csa ? 2 * options.head_size : options.head_size;
  const int64_t buffer_capacity = BufferCapacity(options.compress_ratio);
  const int64_t projection_tokens =
      options.projection_tokens >= 0 ? options.projection_tokens : options.total_tokens;
  NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);

  std::vector<NodeArg*> inputs{
      builder.MakeInput<float>(
          std::vector<int64_t>{options.total_tokens,
                               (options.num_heads + (options.packed_qk ? 1 : 0)) * options.head_size}),
      options.packed_qk
          ? &empty
          : builder.MakeInput<float>(std::vector<int64_t>{
                options.key_total_tokens >= 0 ? options.key_total_tokens : options.total_tokens, width}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{options.head_size}),
      builder.MakeInput<float>(std::vector<int64_t>{64, options.rotary_width}),
      builder.MakeInput<float>(std::vector<int64_t>{64, options.rotary_width}),
      builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size + 1}),
      builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size}),
  };
  if (is_csa || options.add_csa_inputs) {
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{projection_tokens, width}));
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{options.compress_ratio, width}));
    inputs.push_back(builder.MakeInput<float>(std::vector<int64_t>{options.total_tokens, options.num_heads}));
  } else {
    inputs.push_back(&empty);
    inputs.push_back(&empty);
    inputs.push_back(&empty);
  }
  if (is_csa || options.add_position_ids) {
    inputs.push_back(builder.MakeInput<int64_t>(std::vector<int64_t>{projection_tokens}));
  } else {
    inputs.push_back(&empty);
  }
  const int64_t input_state_capacity =
      options.input_state_capacity >= 0 ? options.input_state_capacity : options.state_capacity;
  inputs.push_back(
      builder.MakeInput<float>(std::vector<int64_t>{options.batch_size, input_state_capacity, options.head_size}));
  inputs.push_back(
      builder.MakeInput<float>(std::vector<int64_t>{
          options.batch_size, options.kv_buffer_capacity >= 0 ? options.kv_buffer_capacity : buffer_capacity, width}));
  if (is_csa) {
    inputs.push_back(builder.MakeInput<float>(
        std::vector<int64_t>{options.batch_size, buffer_capacity,
                             options.gate_buffer_width >= 0 ? options.gate_buffer_width : width}));
  } else {
    inputs.push_back(&empty);
  }
  inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size, 2}));
  if (options.add_capture_count || options.add_state_update_active) {
    inputs.push_back(options.add_capture_count
                         ? builder.MakeInput<int32_t>(std::vector<int64_t>{options.batch_size})
                         : &empty);
  }
  if (options.add_state_update_active) {
    inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{1}));
  }
  if (options.indexshare) {
    inputs.resize(psai::kIndexShareMode, &empty);
    inputs.push_back(builder.MakeInput<int32_t>(options.mode_shape));
    inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{options.merged_rows, options.merged_width}));
    inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{
        options.merged_counts_rows >= 0 ? options.merged_counts_rows : options.merged_rows}));
    inputs.push_back(builder.MakeInput<int32_t>(std::vector<int64_t>{
        options.merged_status_rows >= 0 ? options.merged_status_rows : options.merged_rows}));
  }

  std::vector<NodeArg*> outputs;
  const int declared_outputs =
      options.indexshare ? std::max(options.output_count, psai::kIndexShareStatus + 1) : options.output_count;
  for (int i = 0; i < declared_outputs; ++i) {
    const bool omit_gate_output =
        i == psai::kPresentGateBuffer &&
        ((is_csa && options.omit_csa_gate_output) || (!is_csa && !options.add_qsa_gate_output));
    const bool omit_capture_output =
        i == psai::kStateUpdate && options.state_update_capacity == 0 &&
        options.output_count != psai::kStateUpdate + 1;
    const bool omit_indexshare_status = !options.indexshare && i == psai::kIndexShareStatus;
    outputs.push_back(omit_gate_output || omit_capture_output || omit_indexshare_status ? &empty
                                                                                        : builder.MakeOutput());
  }
  Node& node = builder.AddNode("PackedSparseAttentionIndexer", inputs, outputs, kMSDomain);
  node.AddAttribute("policy_mode", options.policy_mode);
  node.AddAttribute("compress_ratio", options.compress_ratio);
  node.AddAttribute("state_capacity", options.state_capacity);
  node.AddAttribute("indexshare_state_policy", options.indexshare_state_policy);
  if (options.indexshare) node.AddAttribute("max_output_entries", int64_t{8});
  if (options.state_update_capacity != 0) {
    node.AddAttribute("state_update_capacity", options.state_update_capacity);
  }
  if (is_csa || options.add_index_topk) {
    node.AddAttribute("index_topk", static_cast<int64_t>(3));
  } else {
    node.AddAttribute("token_budget", options.token_budget);
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerShapeInferenceTest, ValidatesIndexShareInterface) {
  for (int invalid = 0; invalid < 8; ++invalid) {
    GraphOptions options;
    options.indexshare = true;
    switch (invalid) {
      case 0:
        options.mode_shape = {2};
        break;
      case 1:
        options.mode_shape = {1, 1};
        break;
      case 2:
        options.merged_width = 7;
        break;
      case 3:
        options.merged_counts_rows = 1;
        break;
      case 4:
        options.merged_status_rows = 1;
        break;
      case 5:
        options.merged_rows = 3;
        break;
      case 6:
        options.indexshare_state_policy = "invalid";
        break;
      case 7:
        options.indexshare_state_policy = "advance";
        break;
    }
    std::unique_ptr<Model> model;
    SCOPED_TRACE(invalid);
    EXPECT_FALSE(BuildAndResolve([&](ModelTestBuilder& builder) { AddNode(builder, options); }, model).IsOK());
  }
  GraphOptions options;
  options.indexshare = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&](ModelTestBuilder& builder) { AddNode(builder, options); }, model));
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, IndexShareCsaValidatesProjectionRows) {
  for (bool mismatch : {false, true}) {
    GraphOptions options;
    options.indexshare = true;
    options.policy_mode = psai::kPolicyModeCsa;
    options.total_tokens = 0;
    options.key_total_tokens = 2;
    options.projection_tokens = mismatch ? 1 : 2;
    options.merged_rows = 2;
    std::unique_ptr<Model> model;
    const auto status = BuildAndResolve([&](ModelTestBuilder& builder) { AddNode(builder, options); }, model);
    EXPECT_EQ(status.IsOK(), !mismatch) << status.ErrorMessage();
  }
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, MergeValidatesPolicyInputsAndShapes) {
  for (bool append_indices : {false, true}) {
    for (int invalid = -1; invalid < 8; ++invalid) {
      std::unique_ptr<Model> model;
      const auto status = BuildAndResolve([&](ModelTestBuilder& builder) {
        NodeArg* empty = &builder.graph_.GetOrCreateNodeArg("", nullptr);
        std::vector<NodeArg*> inputs{
            builder.MakeInput<int32_t>(std::vector<int64_t>{2, 4}),
            builder.MakeInput<int32_t>(invalid == 0 ? std::vector<int64_t>{2, 1} : invalid == 1 ? std::vector<int64_t>{3}
                                                                                                : std::vector<int64_t>{2}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{3}), empty, empty, empty, empty};
        const int first = append_indices ? 5 : 3;
        inputs[first] = builder.MakeInput<int32_t>(invalid == 2 ? std::vector<int64_t>{3, 1, 1} : append_indices ? std::vector<int64_t>{3, 2}
                                                                                                                 : std::vector<int64_t>{3});
        inputs[first + 1] = builder.MakeInput<int32_t>(invalid == 3 ? std::vector<int64_t>{2} : invalid == 7 ? std::vector<int64_t>{3, 1}
                                                                                                             : std::vector<int64_t>{3});
        if (invalid == 4) inputs[first] = empty;
        if (invalid == 5) inputs[append_indices ? 3 : 5] = builder.MakeInput<int32_t>(std::vector<int64_t>{3});
        if (invalid == 6) inputs[first] = builder.MakeInput<int32_t>(
                              append_indices ? std::vector<int64_t>{2, 2} : std::vector<int64_t>{2});
        Node& node = builder.AddNode("PackedSparseAttentionIndexerMerge", inputs,
                                     {builder.MakeOutput(), builder.MakeOutput(), builder.MakeOutput()}, kMSDomain);
        node.AddAttribute("policy_mode", std::string(append_indices ? "append_indices" : "append_range"));
        node.AddAttribute("max_output_entries", int64_t{8});
      },
                                          model);
      SCOPED_TRACE(invalid);
      EXPECT_EQ(status.IsOK(), invalid == -1) << status.ErrorMessage();
    }
  }
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, QsaInfersFixedCapacityAndState) {
  GraphOptions options;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  const int64_t capacity = options.token_budget + options.compress_ratio - 1;
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens, capacity});
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedCounts]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKeyState]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_capacity, options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKvBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentStateLengths]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32, {options.batch_size, 2});
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, QsaAcceptsPackedQueryKey) {
  GraphOptions options;
  options.packed_qk = true;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Node& node = *model->MainGraph().Nodes().begin();
  EXPECT_FALSE(node.InputDefs()[psai::kKey]->Exists());
  ExpectShape(model->MainGraph(), node.OutputDefs()[psai::kPresentKeyState]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_capacity, options.head_size});
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, QsaInfersStateUpdate) {
  GraphOptions options;
  options.state_update_capacity = 4;
  options.add_capture_count = true;
  options.add_state_update_active = true;
  options.output_count = psai::kStateUpdate + 1;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[psai::kStateUpdate]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_update_capacity, options.head_size});
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, InfersStateStatus) {
  for (const char* policy_mode : {psai::kPolicyModeQsa, psai::kPolicyModeCsa}) {
    for (bool indexshare : {false, true}) {
      GraphOptions options;
      options.policy_mode = policy_mode;
      options.indexshare = indexshare;
      options.output_count = psai::kOutputCount;
      std::unique_ptr<Model> model;
      ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

      const Graph& graph = model->MainGraph();
      const Node& node = *graph.Nodes().begin();
      ExpectShape(graph, node.OutputDefs()[psai::kStateStatus]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
                  {options.batch_size});
    }
  }
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, CsaInfersFixedCapacityAndState) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  std::unique_ptr<Model> model;
  ASSERT_STATUS_OK(BuildAndResolve([&options](ModelTestBuilder& builder) { AddNode(builder, options); }, model));

  const Graph& graph = model->MainGraph();
  const Node& node = *graph.Nodes().begin();
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedIndices]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens, 3});
  ExpectShape(graph, node.OutputDefs()[psai::kSelectedCounts]->Name(), ONNX_NAMESPACE::TensorProto_DataType_INT32,
              {options.total_tokens});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKeyState]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, options.state_capacity, options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentKvBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), 2 * options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentGateBuffer]->Name(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              {options.batch_size, BufferCapacity(options.compress_ratio), 2 * options.head_size});
  ExpectShape(graph, node.OutputDefs()[psai::kPresentStateLengths]->Name(),
              ONNX_NAMESPACE::TensorProto_DataType_INT32, {options.batch_size, 2});
}

#ifdef USE_CUDA
TEST(PackedSparseAttentionIndexerCudaTest, DynamicSharedMemoryBoundaryValidation) {
  constexpr size_t supported_bytes = 64 * 1024;
  EXPECT_STATUS_OK(psai::ValidateDynamicSharedMemory(supported_bytes, supported_bytes, "BoundaryKernel"));
  const Status status = psai::ValidateDynamicSharedMemory(
      supported_bytes + 1, supported_bytes, "BoundaryKernel");
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("requires 65537 bytes"), std::string::npos);
  EXPECT_NE(status.ErrorMessage().find("supports at most 65536"), std::string::npos);
}
#endif

#ifndef ORT_NO_EXCEPTIONS

namespace {

void ExpectResolveFailure(const std::function<void(ModelTestBuilder& builder)>& add_node,
                          const std::string& expected_message) {
  std::unique_ptr<Model> model;
  const Status status = BuildAndResolve(add_node, model);
  ASSERT_FALSE(status.IsOK()) << "expected shape inference to reject the node";
  EXPECT_NE(status.ErrorMessage().find(expected_message), std::string::npos)
      << "actual message: " << status.ErrorMessage();
}

}  // namespace

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsStateCapacityShapeMismatch) {
  GraphOptions options;
  options.input_state_capacity = options.state_capacity + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_key_state dimension 1 must equal state_capacity");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsMismatchedKeyTokenCount) {
  GraphOptions options;
  options.key_total_tokens = options.total_tokens + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "key dimension 0 must equal query dimension 0");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsWrongKvBufferCapacity) {
  GraphOptions options;
  options.kv_buffer_capacity = BufferCapacity(options.compress_ratio) + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_kv_buffer dimension 1 must equal 2 * compress_ratio - 1");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsMismatchedCsaGateBufferWidth) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.gate_buffer_width = 2 * options.head_size + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "past_gate_buffer dimension 2 must equal 2 * head_size");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsUnknownPolicyMode) {
  GraphOptions options;
  options.policy_mode = "qsa_v2";
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "policy_mode must be 'qsa' or 'csa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaAttribute) {
  GraphOptions options;
  options.add_index_topk = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "index_topk and head_weight_scale must not be set");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsZeroNumHeads) {
  GraphOptions options;
  options.num_heads = 0;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "query width must contain at least one query head");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsGenericBufferCapacityOverflow) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.compress_ratio = static_cast<int64_t>(std::numeric_limits<int>::max()) / 2 + 2;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "generic buffer capacity no greater than INT_MAX");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaTokenBudgetNotDivisibleByCompressRatio) {
  GraphOptions options;
  options.token_budget = 5;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "requires token_budget > 0, divisible by compress_ratio");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithCsaInput) {
  GraphOptions options;
  options.add_csa_inputs = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "must be omitted when policy_mode is 'qsa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaWithPositionIds) {
  GraphOptions options;
  options.add_position_ids = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "must be omitted when policy_mode is 'qsa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsCsaMissingPositionIds) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.add_position_ids = false;
  // Force the csa node to omit position_ids by rebuilding without it.
  std::unique_ptr<Model> model;
  const Status status = BuildAndResolve(
      [&options](ModelTestBuilder& builder) {
        GraphOptions local = options;
        NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
        const int64_t width = 2 * local.head_size;
        const int64_t buffer_capacity = BufferCapacity(local.compress_ratio);
        std::vector<NodeArg*> inputs{
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, local.num_heads * local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{64, local.rotary_width}),
            builder.MakeInput<float>(std::vector<int64_t>{64, local.rotary_width}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size + 1}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.compress_ratio, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.total_tokens, local.num_heads}),
            &empty,  // position_ids omitted: invalid for csa
            builder.MakeInput<float>(
                std::vector<int64_t>{local.batch_size, local.state_capacity, local.head_size}),
            builder.MakeInput<float>(std::vector<int64_t>{local.batch_size, buffer_capacity, width}),
            builder.MakeInput<float>(std::vector<int64_t>{local.batch_size, buffer_capacity, width}),
            builder.MakeInput<int32_t>(std::vector<int64_t>{local.batch_size, 2}),
        };
        std::vector<NodeArg*> outputs;
        for (int i = 0; i < psai::kFixedOutputCount; ++i) {
          outputs.push_back(builder.MakeOutput());
        }
        Node& node = builder.AddNode("PackedSparseAttentionIndexer", inputs, outputs, kMSDomain);
        node.AddAttribute("policy_mode", local.policy_mode);
        node.AddAttribute("compress_ratio", local.compress_ratio);
        node.AddAttribute("state_capacity", local.state_capacity);
        node.AddAttribute("index_topk", static_cast<int64_t>(3));
      },
      model);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("position_ids"), std::string::npos) << status.ErrorMessage();
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsWrongOutputCount) {
  GraphOptions options;
  options.output_count = 4;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "output size 4 not in range [min=6, max=16]");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsStateUpdateCapacityWithoutCaptureCount) {
  GraphOptions options;
  options.state_update_capacity = 4;
  options.output_count = psai::kStateUpdate + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "state_update_capture_count is required");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsCsaStateUpdateCapture) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.state_update_capacity = 4;
  options.add_capture_count = true;
  options.output_count = psai::kStateUpdate + 1;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "only valid for policy_mode 'qsa'");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsQsaPresentGateBuffer) {
  GraphOptions options;
  options.add_qsa_gate_output = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "present_gate_buffer) must be omitted");
}

TEST(PackedSparseAttentionIndexerShapeInferenceTest, RejectsMissingCsaPresentGateBuffer) {
  GraphOptions options;
  options.policy_mode = psai::kPolicyModeCsa;
  options.omit_csa_gate_output = true;
  ExpectResolveFailure([&options](ModelTestBuilder& builder) { AddNode(builder, options); },
                       "present_gate_buffer) is required");
}

#endif  // ORT_NO_EXCEPTIONS

// ---------------------------------------------------------------------------------------------
// Numeric behaviour (CUDA / WebGPU only)
// ---------------------------------------------------------------------------------------------

namespace {

enum class ProviderKind {
  Cuda,
  WebGpu,
};

std::unique_ptr<IExecutionProvider> CreateProvider(ProviderKind provider_kind,
                                                   const ConfigOptions* webgpu_options = nullptr) {
  if (provider_kind == ProviderKind::Cuda) {
    return DefaultCudaExecutionProvider();
  }
#ifdef USE_WEBGPU
  return webgpu_options == nullptr ? DefaultWebGpuExecutionProvider()
                                   : WebGpuExecutionProviderWithOptions(*webgpu_options);
#else
  ORT_UNUSED_PARAMETER(webgpu_options);
  return nullptr;
#endif
}

void RunOnProvider(OpTester& test, std::unique_ptr<IExecutionProvider> provider,
                   OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
                   const std::string& expected_error = "") {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.Run(expected_result, expected_error, {}, nullptr, &providers);
}

std::vector<float> MakeWave(size_t count, float phase, float step) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::sin(phase + step * static_cast<float>(i));
  }
  return values;
}

template <typename T>
std::vector<T> ToElementType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// Rounds through the tested element type so the reference consumes exactly the kernel's inputs.
template <typename T>
std::vector<float> RoundTrip(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, float>) {
    return data;
  } else {
    std::vector<T> converted = ToElementType<T>(data);
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      result[i] = converted[i].ToFloat();
    }
    return result;
  }
}

template <typename T>
std::vector<float> CopyTensorToFloat(const Tensor& tensor) {
  const auto values = tensor.DataAsSpan<T>();
  std::vector<float> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if constexpr (std::is_same_v<T, float>) {
      result[i] = values[i];
    } else {
      result[i] = values[i].ToFloat();
    }
  }
  return result;
}

// Split-half rotary over the leading rotary_width channels.
std::vector<float> LeadingRope(const std::vector<float>& value, int rotary_width, const float* cos_row,
                               const float* sin_row) {
  const int head_size = static_cast<int>(value.size());
  const int half = rotary_width / 2;
  std::vector<float> result(value);
  for (int d = 0; d < rotary_width && d < head_size; ++d) {
    const float paired = (d < half) ? -value[static_cast<size_t>(d + half)] : value[static_cast<size_t>(d - half)];
    result[static_cast<size_t>(d)] = value[static_cast<size_t>(d)] * cos_row[d] + paired * sin_row[d];
  }
  return result;
}

// Interleaved rotary over the trailing 2 * rotary_width channels.
std::vector<float> TrailingRope(const std::vector<float>& value, int rotary_width, const float* cos_row,
                                const float* sin_row) {
  const int head_size = static_cast<int>(value.size());
  const int base = head_size - 2 * rotary_width;
  std::vector<float> result(value);
  for (int d = base; d < head_size; ++d) {
    const int offset = d - base;
    const float paired = ((offset & 1) == 0) ? -value[static_cast<size_t>(d + 1)] : value[static_cast<size_t>(d - 1)];
    result[static_cast<size_t>(d)] =
        value[static_cast<size_t>(d)] * cos_row[offset >> 1] + paired * sin_row[offset >> 1];
  }
  return result;
}

std::vector<float> RmsNormalize(const std::vector<float>& value, const std::vector<float>& weight, float epsilon) {
  float sum_squares = 0.0f;
  for (float element : value) {
    sum_squares += element * element;
  }
  const float inverse_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(value.size()) + epsilon);
  std::vector<float> result(value.size());
  for (size_t d = 0; d < value.size(); ++d) {
    result[d] = value[d] * inverse_rms * weight[d];
  }
  return result;
}

// Order used by both selection kernels: score descending, then entry index ascending.
std::vector<int> RankByScore(const std::vector<float>& scores, int count) {
  std::vector<int> order(static_cast<size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&scores](int left, int right) {
    if (scores[static_cast<size_t>(left)] != scores[static_cast<size_t>(right)]) {
      return scores[static_cast<size_t>(left)] > scores[static_cast<size_t>(right)];
    }
    return left < right;
  });
  return order;
}

int CausalThreshold(int position, int compress_ratio) {
  if (position < 0) return 0;
  return position / compress_ratio + (position % compress_ratio == compress_ratio - 1 ? 1 : 0);
}

struct QsaPackedProblem {
  int batch_size = 2;
  std::vector<int32_t> cumulative_sequence_lengths;  // [batch + 1]
  std::vector<int32_t> past_sequence_lengths;        // [batch]
  std::vector<int32_t> past_state_lengths;           // [batch, 2]
  int head_size = 4;
  int num_heads = 2;
  int rotary_width = 4;
  int compress_ratio = 2;
  int token_budget = 4;
  int state_capacity = 4;
  int max_position = 64;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;
  int state_update_capacity = 0;
  std::vector<int32_t> state_update_capture_count;
  std::optional<int32_t> state_update_active;
  bool report_state_status = false;
  bool report_mid_state = false;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> query_norm_weight;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;  // shared: [max_position, rotary_width]
  std::vector<float> sin_cache;
  std::vector<int64_t> position_ids;
  std::vector<float> past_key_state;  // [batch, state_capacity, head_size]
  std::vector<float> past_kv_buffer;  // [batch, buffer_capacity, head_size]

  int TotalTokens() const { return cumulative_sequence_lengths.back(); }
  int BufferCapacity() const { return 2 * compress_ratio - 1; }
  int Capacity() const { return token_budget + compress_ratio - 1; }
};

struct QsaPackedResult {
  std::vector<int32_t> selected_indices;
  std::vector<int32_t> selected_counts;
  std::vector<float> present_key_state;
  std::vector<float> present_kv_buffer;
  std::vector<int32_t> present_state_lengths;
  std::vector<float> state_update;
  std::vector<int32_t> state_status;
};

void QsaPackedReference(const QsaPackedProblem& p, QsaPackedResult& out) {
  const int total_tokens = p.TotalTokens();
  const int head_size = p.head_size;
  const int block_topk = p.token_budget / p.compress_ratio;
  const float scale = p.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));
  const int buffer_capacity = p.BufferCapacity();
  const int capacity = p.Capacity();

  out.present_key_state = p.past_key_state;
  out.present_kv_buffer = p.past_kv_buffer;
  out.present_state_lengths = p.past_state_lengths;
  out.state_update.assign(
      static_cast<size_t>(p.batch_size) * p.state_update_capacity * p.head_size, 0.0f);

  std::vector<int> key_len_after(static_cast<size_t>(p.batch_size));
  std::vector<bool> overflowed(static_cast<size_t>(p.batch_size), false);
  out.state_status.assign(static_cast<size_t>(p.batch_size), 0);
  for (int b = 0; b < p.batch_size; ++b) {
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    const int raw_key_len = p.past_state_lengths[static_cast<size_t>(b) * 2 + 0];
    const int raw_buf_len = p.past_state_lengths[static_cast<size_t>(b) * 2 + 1];
    const int past_sequence_length = p.past_sequence_lengths[static_cast<size_t>(b)];
    const bool invalid_metadata =
        p.cumulative_sequence_lengths[0] != 0 ||
        p.cumulative_sequence_lengths[static_cast<size_t>(p.batch_size)] != total_tokens ||
        req_start < 0 || req_end < req_start || req_end > total_tokens ||
        past_sequence_length < 0 || raw_key_len < 0 || raw_key_len > p.state_capacity ||
        raw_buf_len < 0 || raw_buf_len >= p.compress_ratio ||
        raw_key_len != past_sequence_length / p.compress_ratio ||
        raw_buf_len != past_sequence_length % p.compress_ratio;
    const int req_len = invalid_metadata ? 0 : req_end - req_start;
    const int old_key_len = std::clamp(raw_key_len, 0, p.state_capacity);
    const int old_buf_len = std::clamp(raw_buf_len, 0, p.compress_ratio - 1);
    const int pending = old_buf_len + req_len;
    const int full_new_block_count = pending / p.compress_ratio;
    overflowed[static_cast<size_t>(b)] =
        invalid_metadata || full_new_block_count > std::max(p.state_capacity - old_key_len, 0);
    out.state_status[static_cast<size_t>(b)] =
        invalid_metadata ? psai::kStateStatusInvalidMetadata
                         : (overflowed[static_cast<size_t>(b)] ? psai::kStateStatusCapacityOverflow : 0);
    if (overflowed[static_cast<size_t>(b)]) {
      out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len;
      out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = old_buf_len;
      key_len_after[static_cast<size_t>(b)] = old_key_len;
      continue;
    }
    const int new_block_count = full_new_block_count;
    const int new_buf_len = pending % p.compress_ratio;

    auto raw_value = [&](int virtual_pos, int d) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * head_size + d]
                 : p.key[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * head_size + d];
    };

    for (int k = 0; k < new_block_count; ++k) {
      std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
      for (int t = 0; t < p.compress_ratio; ++t) {
        for (int d = 0; d < head_size; ++d) {
          pooled[static_cast<size_t>(d)] += raw_value(k * p.compress_ratio + t, d);
        }
      }
      for (float& v : pooled) v /= static_cast<float>(p.compress_ratio);
      pooled = RmsNormalize(pooled, p.key_norm_weight, p.epsilon);
      const int entry = old_key_len + k;
      const int position = std::min(entry * p.compress_ratio, p.max_position - 1);
      pooled = LeadingRope(pooled, p.rotary_width, p.cos_cache.data() + position * p.rotary_width,
                           p.sin_cache.data() + position * p.rotary_width);
      for (int d = 0; d < head_size; ++d) {
        out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + entry) * head_size + d] =
            pooled[static_cast<size_t>(d)];
      }
    }
    for (int t = 0; t < new_buf_len; ++t) {
      for (int d = 0; d < head_size; ++d) {
        out.present_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * head_size + d] =
            raw_value(new_block_count * p.compress_ratio + t, d);
      }
    }
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len + new_block_count;
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = new_buf_len;
    key_len_after[static_cast<size_t>(b)] = old_key_len + new_block_count;

    const bool capture_active = p.state_update_capacity > 0 && !p.state_update_capture_count.empty() &&
                                (!p.state_update_active.has_value() || *p.state_update_active != 0);
    const int capture_count = capture_active
                                  ? std::clamp(p.state_update_capture_count[static_cast<size_t>(b)], 0,
                                               std::min(req_len, p.state_update_capacity))
                                  : 0;
    for (int t = 0; t < capture_count; ++t) {
      const bool completes_block = (old_buf_len + t + 1) % p.compress_ratio == 0;
      for (int d = 0; d < head_size; ++d) {
        const size_t capture_offset =
            (static_cast<size_t>(b) * p.state_update_capacity + t) * head_size + d;
        if (completes_block) {
          const int block = (old_buf_len + t + 1) / p.compress_ratio - 1;
          out.state_update[capture_offset] =
              out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + old_key_len + block) *
                                        head_size +
                                    d];
        } else {
          out.state_update[capture_offset] =
              p.key[(static_cast<size_t>(req_start) + t) * head_size + d];
        }
      }
    }
  }

  out.selected_indices.assign(static_cast<size_t>(total_tokens) * capacity, -1);
  out.selected_counts.assign(static_cast<size_t>(total_tokens), 0);
  for (int b = 0; b < p.batch_size; ++b) {
    if (overflowed[static_cast<size_t>(b)]) {
      continue;
    }
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    for (int token = req_start; token < req_end; ++token) {
      const int position = p.position_ids.empty()
                               ? p.past_sequence_lengths[static_cast<size_t>(b)] + (token - req_start)
                               : static_cast<int>(p.position_ids[static_cast<size_t>(token)]);
      const int causal_count = CausalThreshold(position, p.compress_ratio);
      const int visible_block_count = std::min(key_len_after[static_cast<size_t>(b)], causal_count);
      const int selected = std::min(block_topk, visible_block_count);

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(p.num_heads));
      for (int h = 0; h < p.num_heads; ++h) {
        const size_t base = (static_cast<size_t>(token) * p.num_heads + h) * head_size;
        std::vector<float> head(p.query.begin() + base, p.query.begin() + base + head_size);
        head = RmsNormalize(head, p.query_norm_weight, p.epsilon);
        const int clamped_position = std::min(std::max(position, 0), p.max_position - 1);
        rotated_query[static_cast<size_t>(h)] =
            LeadingRope(head, p.rotary_width, p.cos_cache.data() + clamped_position * p.rotary_width,
                        p.sin_cache.data() + clamped_position * p.rotary_width);
      }

      std::vector<float> scores(static_cast<size_t>(visible_block_count), 0.0f);
      for (int j = 0; j < visible_block_count; ++j) {
        float score = 0.0f;
        for (int h = 0; h < p.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] *
                   out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + j) * head_size + d];
          }
          score += std::max(dot, 0.0f);
        }
        scores[static_cast<size_t>(j)] = score * scale;
      }
      const std::vector<int> order = RankByScore(scores, visible_block_count);
      const int emitted_blocks = std::min(selected, visible_block_count);
      int32_t* out_row = out.selected_indices.data() + static_cast<size_t>(token) * capacity;
      for (int rank = 0; rank < emitted_blocks; ++rank) {
        for (int t = 0; t < p.compress_ratio; ++t) {
          out_row[rank * p.compress_ratio + t] = order[static_cast<size_t>(rank)] * p.compress_ratio + t;
        }
      }
      const int block_start = visible_block_count * p.compress_ratio;
      const int natural_tail = position >= block_start ? position - block_start + 1 : 0;
      const int remaining_capacity = capacity - emitted_blocks * p.compress_ratio;
      const int tail_count = std::clamp(natural_tail, 0, remaining_capacity);
      for (int t = 0; t < tail_count; ++t) {
        out_row[emitted_blocks * p.compress_ratio + t] = block_start + t;
      }
      out.selected_counts[static_cast<size_t>(token)] = emitted_blocks * p.compress_ratio + tail_count;
    }
  }
}

QsaPackedProblem MakeQsaPackedProblem(QsaPackedProblem problem = {}) {
  if (problem.cumulative_sequence_lengths.empty()) {
    // Default: 2 requests, 2 and 3 tokens respectively (unequal lengths).
    problem.cumulative_sequence_lengths = {0, 2, 5};
    problem.past_sequence_lengths = {3, 0};
  }
  const int total_tokens = problem.TotalTokens();
  const int buffer_capacity = problem.BufferCapacity();
  if (problem.past_state_lengths.empty()) {
    problem.past_state_lengths.assign(static_cast<size_t>(problem.batch_size) * 2, 0);
    for (int b = 0; b < problem.batch_size; ++b) {
      problem.past_state_lengths[static_cast<size_t>(b) * 2 + 0] = problem.past_sequence_lengths[static_cast<size_t>(b)] / problem.compress_ratio;
      problem.past_state_lengths[static_cast<size_t>(b) * 2 + 1] = problem.past_sequence_lengths[static_cast<size_t>(b)] % problem.compress_ratio;
    }
  }

  problem.query =
      MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads * problem.head_size, 0.35f, 0.41f);
  problem.key = MakeWave(static_cast<size_t>(total_tokens) * problem.head_size, 1.10f, 0.29f);
  problem.query_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 1.30f, 0.23f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.70f, 0.17f);
  problem.cos_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.20f, 0.13f);
  problem.sin_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.90f, 0.19f);
  problem.past_key_state =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.state_capacity * problem.head_size, 0.05f, 0.23f);
  problem.past_kv_buffer =
      MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * problem.head_size, 0.60f, 0.09f);
  return problem;
}

template <typename T, typename TState = T>
void RunQsaPackedTest(float tolerance, QsaPackedProblem problem = MakeQsaPackedProblem(),
                      ProviderKind provider_kind = ProviderKind::Cuda, QsaPackedResult* actual = nullptr,
                      bool packed_qk = false, const ConfigOptions* webgpu_options = nullptr, int padding = 0,
                      const std::string& expected_error = "") {
  auto provider = CreateProvider(provider_kind, webgpu_options);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.query_norm_weight = RoundTrip<T>(problem.query_norm_weight);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.past_key_state = RoundTrip<TState>(problem.past_key_state);
  problem.past_kv_buffer = RoundTrip<T>(problem.past_kv_buffer);

  QsaPackedResult expected;
  QsaPackedReference(problem, expected);
  const int64_t output_capacity = problem.Capacity() + padding;
  if (padding > 0) {
    std::vector<int32_t> padded(static_cast<size_t>(problem.TotalTokens() * output_capacity), -1);
    for (int token = 0; token < problem.TotalTokens(); ++token) {
      std::copy_n(expected.selected_indices.begin() + token * problem.Capacity(), problem.Capacity(),
                  padded.begin() + token * output_capacity);
    }
    expected.selected_indices = std::move(padded);
  }

  const int64_t total_tokens = problem.TotalTokens();
  const int64_t batch_size = problem.batch_size;
  const int64_t head_size = problem.head_size;
  const int64_t buffer_capacity = problem.BufferCapacity();

  OpTester test("PackedSparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(psai::kPolicyModeQsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("state_capacity", static_cast<int64_t>(problem.state_capacity));
  test.AddAttribute("token_budget", static_cast<int64_t>(problem.token_budget));
  if (problem.state_update_capacity > 0) {
    test.AddAttribute("state_update_capacity", static_cast<int64_t>(problem.state_update_capacity));
  }
  if (problem.scale.has_value()) {
    test.AddAttribute("scale", *problem.scale);
  }
  if (packed_qk) {
    const int64_t query_width = problem.num_heads * head_size;
    const int64_t packed_width = query_width + head_size;
    std::vector<float> query_key(static_cast<size_t>(total_tokens * packed_width));
    for (int64_t token = 0; token < total_tokens; ++token) {
      std::copy_n(problem.query.begin() + token * query_width, query_width,
                  query_key.begin() + token * packed_width);
      std::copy_n(problem.key.begin() + token * head_size, head_size,
                  query_key.begin() + token * packed_width + query_width);
    }
    test.AddInput<T>("query_key", {total_tokens, packed_width}, ToElementType<T>(query_key));
    test.AddOptionalInputEdge<T>();
  } else {
    test.AddInput<T>("query", {total_tokens, problem.num_heads * head_size}, ToElementType<T>(problem.query));
    test.AddInput<T>("key", {total_tokens, head_size}, ToElementType<T>(problem.key));
  }
  test.AddInput<T>("query_norm_weight", {head_size}, ToElementType<T>(problem.query_norm_weight));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  test.AddInput<T>("cos_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.sin_cache));
  test.AddInput<int32_t>("cumulative_sequence_lengths", {batch_size + 1}, problem.cumulative_sequence_lengths);
  test.AddInput<int32_t>("past_sequence_lengths", {batch_size}, problem.past_sequence_lengths);
  test.AddOptionalInputEdge<T>();  // gate
  test.AddOptionalInputEdge<T>();  // position_bias
  test.AddOptionalInputEdge<T>();  // head_weights
  if (problem.position_ids.empty()) {
    test.AddOptionalInputEdge<int64_t>();
  } else {
    test.AddInput<int64_t>("position_ids", {total_tokens}, problem.position_ids);
  }
  test.AddInput<TState>("past_key_state", {batch_size, problem.state_capacity, head_size},
                        ToElementType<TState>(problem.past_key_state));
  test.AddInput<T>("past_kv_buffer", {batch_size, buffer_capacity, head_size},
                   ToElementType<T>(problem.past_kv_buffer));
  test.AddOptionalInputEdge<T>();  // past_gate_buffer
  test.AddInput<int32_t>("past_state_lengths", {batch_size, 2}, problem.past_state_lengths);
  if (problem.state_update_capacity > 0) {
    test.AddInput<int32_t>("state_update_capture_count", {batch_size}, problem.state_update_capture_count);
    if (problem.state_update_active.has_value()) {
      test.AddInput<int32_t>("state_update_active", {1}, {*problem.state_update_active});
    }
  }

  if (padding > 0) {
    test.AddAttribute("max_output_entries", output_capacity);
    if (problem.state_update_capacity == 0) test.AddOptionalInputEdge<int32_t>();
    if (!problem.state_update_active.has_value()) test.AddOptionalInputEdge<int32_t>();
    test.AddInput<int32_t>("mode", {1}, {0});
    test.AddInput<int32_t>("merged", {0, output_capacity}, {});
    test.AddInput<int32_t>("merged_counts", {0}, {});
    test.AddInput<int32_t>("merged_status", {0}, {});
  }
  test.AddOutput<int32_t>("selected_indices", {total_tokens, output_capacity}, expected.selected_indices);
  test.AddOutput<int32_t>("selected_counts", {total_tokens}, expected.selected_counts);
  test.AddOutput<TState>("present_key_state", {batch_size, problem.state_capacity, head_size},
                         ToElementType<TState>(expected.present_key_state), false, 0.0f, tolerance);
  test.AddOutput<T>("present_kv_buffer", {batch_size, buffer_capacity, head_size},
                    ToElementType<T>(expected.present_kv_buffer), false, 0.0f, tolerance);
  test.AddOptionalOutputEdge<T>();  // present_gate_buffer
  test.AddOutput<int32_t>("present_state_lengths", {batch_size, 2}, expected.present_state_lengths);
  if (problem.state_update_capacity > 0) {
    test.AddOutput<T>("state_update", {batch_size, problem.state_update_capacity, head_size},
                      ToElementType<T>(expected.state_update), false, 0.0f, tolerance);
  }
  if (padding > 0) {
    if (problem.state_update_capacity == 0) test.AddOptionalOutputEdge<T>();
    test.AddOutput<int32_t>("status", {total_tokens}, std::vector<int32_t>(static_cast<size_t>(total_tokens), 0));
  }
  if (problem.report_state_status || problem.report_mid_state) {
    if (padding == 0) {
      if (problem.state_update_capacity == 0) test.AddOptionalOutputEdge<T>();
      test.AddOptionalOutputEdge<int32_t>();  // indexshare_status
    }
    if (problem.report_state_status) {
      test.AddOutput<int32_t>("state_status", {batch_size}, expected.state_status);
    } else {
      test.AddOptionalOutputEdge<int32_t>();
    }
  }
  if (problem.report_mid_state) {
    for (int batch = 0; batch < batch_size; ++batch) {
      ASSERT_EQ(problem.cumulative_sequence_lengths[batch + 1] - problem.cumulative_sequence_lengths[batch], 1);
    }
    test.AddOutput<T>("mid_kv_buffer", {batch_size, buffer_capacity, head_size},
                      ToElementType<T>(expected.present_kv_buffer), false, 0.0f, tolerance);
    test.AddOutput<int32_t>("mid_state_lengths", {batch_size, 2}, expected.present_state_lengths);
    test.AddOutput<int32_t>("validation_status", {batch_size},
                            std::vector<int32_t>(static_cast<size_t>(batch_size), 0));
  }
  RunOnProvider(test, std::move(provider),
                expected_error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
                expected_error);
  if (actual != nullptr) {
    const auto& fetches = test.GetFetches();
    actual->selected_indices.assign(fetches[0].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                    fetches[0].Get<Tensor>().DataAsSpan<int32_t>().end());
    actual->selected_counts.assign(fetches[1].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                   fetches[1].Get<Tensor>().DataAsSpan<int32_t>().end());
    actual->present_key_state = CopyTensorToFloat<TState>(fetches[2].Get<Tensor>());
    actual->present_kv_buffer = CopyTensorToFloat<T>(fetches[3].Get<Tensor>());
    actual->present_state_lengths.assign(fetches[4].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                         fetches[4].Get<Tensor>().DataAsSpan<int32_t>().end());
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerTest, QsaFloat) { RunQsaPackedTest<float>(1.0e-5f); }

TEST(PackedSparseAttentionIndexerTest, QsaFloat16) { RunQsaPackedTest<MLFloat16>(2.0e-3f); }

TEST(PackedSparseAttentionIndexerTest, QsaBFloat16) { RunQsaPackedTest<BFloat16>(2.0e-2f); }

TEST(PackedSparseAttentionIndexerTest, QsaMidStateOutputsPreserveStateStatusSlot) {
  for (bool report_state_status : {false, true}) {
    QsaPackedProblem problem;
    problem.cumulative_sequence_lengths = {0, 1, 2};
    problem.past_sequence_lengths = {0, 0};
    problem.past_state_lengths = {0, 0, 0, 0};
    problem.report_state_status = report_state_status;
    problem.report_mid_state = true;
    problem = MakeQsaPackedProblem(std::move(problem));
    RunQsaPackedTest<float>(1.0e-5f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16, float>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaBFloat16FloatState) {
  RunQsaPackedTest<BFloat16, float>(2.0e-2f);
}

TEST(PackedSparseAttentionIndexerTest, QsaPackedQueryKey) {
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::Cuda, nullptr, true);
}

TEST(PackedSparseAttentionIndexerTest, IndexShareRefreshPreservesBudgetAndPadsRows) {
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::Cuda, nullptr, true, nullptr, 6);
}

TEST(PackedSparseAttentionIndexerTest, IndexShareReusePackedQkAndStatus) {
  for (int invalid = 0; invalid < 9; ++invalid) {
    SCOPED_TRACE(invalid);
    std::vector<int32_t> cumulative{0, 1, 2};
    std::vector<int32_t> past_lengths{5, 5};
    std::vector<int32_t> state_lengths{2, 1, 2, 1};
    std::vector<int32_t> merged_counts{3, 3};
    if (invalid == 1) merged_counts[0] = -1;
    if (invalid == 2) merged_counts[0] = 6;
    if (invalid == 3) cumulative[2] = 5;
    if (invalid == 4) past_lengths[0] = -1;
    if (invalid == 5) state_lengths[0] = 5;
    if (invalid == 6) state_lengths[1] = 0;
    auto provider = DefaultCudaExecutionProvider();
    if (!provider) GTEST_SKIP() << "Requires CUDA";
    OpTester test("PackedSparseAttentionIndexer", 1, kMSDomain);
    test.AddAttribute("policy_mode", std::string("qsa"));
    test.AddAttribute("compress_ratio", int64_t{2});
    test.AddAttribute("state_capacity", int64_t{4});
    test.AddAttribute("token_budget", int64_t{2});
    test.AddAttribute("max_output_entries", int64_t{5});
    test.AddInput<float>("query_key", {0, 4}, {});
    test.AddOptionalInputEdge<float>();
    test.AddInput<float>("q_norm", {2}, {1, 1});
    test.AddInput<float>("k_norm", {2}, {1, 1});
    test.AddInput<float>("cos", {8, 2}, std::vector<float>(16, 1));
    test.AddInput<float>("sin", {8, 2}, std::vector<float>(16, 0));
    test.AddInput<int32_t>("cumulative", {3}, cumulative);
    test.AddInput<int32_t>("past_lengths", {2}, past_lengths);
    for (int index = 8; index <= 10; ++index) test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<int64_t>();
    test.AddInput<float>("past_keys", {2, 4, 2}, std::vector<float>(16, 1));
    test.AddInput<float>("past_buffer", {2, 3, 2}, std::vector<float>(12, 2));
    test.AddOptionalInputEdge<float>();
    test.AddInput<int32_t>("state_lengths", {2, 2}, state_lengths);
    test.AddOptionalInputEdge<int32_t>();
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<int32_t>("mode", {1}, {1});
    test.AddInput<int32_t>("merged_indices", {2, 5}, {0, 2, 5, -1, -1, 9, 8, 7, 6, 5});
    test.AddInput<int32_t>("merged_counts", {2}, merged_counts);
    test.AddInput<int32_t>("merged_status", {2}, {0, 2});
    test.AddOutput<int32_t>("selected", {2, 5}, invalid == 0 ? std::vector<int32_t>{0, 2, 5, -1, -1, -1, -1, -1, -1, -1} : std::vector<int32_t>(10, -1));
    test.AddOutput<int32_t>("selected_counts", {2}, {invalid == 0 ? 3 : 0, 0});
    test.AddOutput<float>("present_keys", {2, 4, 2}, std::vector<float>(16, 1));
    test.AddOutput<float>("present_buffer", {2, 3, 2}, std::vector<float>(12, 2));
    test.AddOptionalOutputEdge<float>();
    test.AddOutput<int32_t>("present_lengths", {2, 2}, state_lengths);
    test.AddOptionalOutputEdge<float>();
    test.AddOutput<int32_t>("status", {2}, {invalid == 0 ? 0 : 1, 2});
    test.AddOutput<int32_t>("state_status", {2}, {0, 0});
    if (invalid == 7) {
      test.AddOutput<float>("mid_buffer", {2, 3, 2}, std::vector<float>(12));
      test.AddOutput<int32_t>("mid_lengths", {2, 2}, std::vector<int32_t>(4));
    } else if (invalid == 8) {
      test.AddOptionalOutputEdge<float>();
      test.AddOptionalOutputEdge<int32_t>();
      test.AddOutput<int32_t>("validation", {2}, {0, 0});
    }
    RunOnProvider(test, std::move(provider),
                  invalid >= 7 ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
                  invalid >= 7 ? "outputs 9-11 are not supported in IndexShare reuse mode" : "");
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaPrefillThenDecodeIndependentState) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  QsaPackedProblem prefill;
  prefill.cumulative_sequence_lengths = {0, 3, 5};
  prefill.past_sequence_lengths = {0, 0};
  QsaPackedResult prefill_outputs;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(prefill)), ProviderKind::Cuda,
                          &prefill_outputs);

  QsaPackedProblem decode;
  decode.cumulative_sequence_lengths = {0, 1, 3};
  decode.past_sequence_lengths = {3, 2};
  decode = MakeQsaPackedProblem(std::move(decode));
  decode.past_key_state = std::move(prefill_outputs.present_key_state);
  decode.past_kv_buffer = std::move(prefill_outputs.present_kv_buffer);
  decode.past_state_lengths = std::move(prefill_outputs.present_state_lengths);
  RunQsaPackedTest<float>(1.0e-5f, std::move(decode));
}

// A zero-token request row (repeated cumulative offset) must not affect the other request and
// must not read out of bounds.
TEST(PackedSparseAttentionIndexerTest, QsaZeroTokenRequestRow) {
  QsaPackedProblem problem;
  problem.batch_size = 3;
  problem.cumulative_sequence_lengths = {0, 2, 2, 5};
  problem.past_sequence_lengths = {3, 7, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

// state_capacity smaller than what the incoming tokens would naturally produce: the update must
// reject the request's step without partially changing its state.
TEST(PackedSparseAttentionIndexerTest, QsaStateCapacityOverflowIsSafe) {
  QsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 6};
  problem.past_sequence_lengths = {4};
  problem.batch_size = 1;
  problem.state_capacity = 2;  // only room for 2 - 2(already used) = 0 new complete blocks
  problem.past_state_lengths = {2, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

TEST(PackedSparseAttentionIndexerTest, QsaStateUpdateCapturesRawAndBlockTransitions) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 3};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::Cuda);
}

TEST(PackedSparseAttentionIndexerTest, QsaBFloat16FloatStateCapture) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 3};
  RunQsaPackedTest<BFloat16, float>(2.0e-2f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::Cuda);
}

TEST(PackedSparseAttentionIndexerTest, QsaStateUpdateActiveZeroClearsAllSlots) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 5};
  problem.state_update_active = 0;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::Cuda);
}

TEST(PackedSparseAttentionIndexerTest, QsaStateUpdateActiveOneCapturesRawAndBlockTransitions) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 5};
  problem.state_update_active = 1;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::Cuda);
}

TEST(PackedSparseAttentionIndexerTest, QsaBoundedTopKMaximum) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {1040};
  problem.compress_ratio = 2;
  problem.token_budget = 1024;
  problem.state_capacity = 520;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalTopK) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {8204};
  problem.head_size = 128;
  problem.num_heads = 4;
  problem.rotary_width = 64;
  problem.compress_ratio = 4;
  problem.token_budget = 2048;
  problem.state_capacity = 2051;
  problem = MakeQsaPackedProblem(std::move(problem));
  std::fill(problem.query.begin(), problem.query.end(), 1.0f);
  std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
  std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
  std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
  for (int block = 0; block < problem.state_capacity; ++block) {
    const float value = static_cast<float>(block + 1) / problem.state_capacity;
    std::fill_n(problem.past_key_state.begin() + static_cast<size_t>(block) * problem.head_size,
                problem.head_size, value);
  }
  RunQsaPackedTest<float>(1.0e-5f, std::move(problem));
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalPackedVerification) {
  QsaPackedProblem problem;
  problem.batch_size = 3;
  problem.cumulative_sequence_lengths = {0, 0, 7, 8};
  problem.past_sequence_lengths = {0, 4101, 16388};
  problem.head_size = 128;
  problem.num_heads = 4;
  problem.rotary_width = 64;
  problem.compress_ratio = 4;
  problem.token_budget = 2048;
  problem.state_capacity = 65536;
  problem = MakeQsaPackedProblem(std::move(problem));
  std::fill(problem.query.begin(), problem.query.end(), 1.0f);
  std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
  std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
  std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
  for (int batch = 0; batch < problem.batch_size; ++batch) {
    for (int block = 0; block < problem.state_capacity; ++block) {
      const float value = static_cast<float>(block + 1) / problem.state_capacity;
      const size_t offset = (static_cast<size_t>(batch) * problem.state_capacity + block) * problem.head_size;
      std::fill_n(problem.past_key_state.begin() + offset, problem.head_size, value);
    }
  }
  RunQsaPackedTest<float>(1.0e-5f, std::move(problem), ProviderKind::Cuda);
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalPairedVerification) {
  for (int tokens : {5, 8}) {
    QsaPackedProblem problem;
    problem.batch_size = 1;
    problem.cumulative_sequence_lengths = {0, tokens};
    problem.past_sequence_lengths = {4098};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = 4099;
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 1.0f);
    std::fill(problem.key.begin(), problem.key.end(), 1.0f);
    std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
    std::fill(problem.key_norm_weight.begin(), problem.key_norm_weight.end(), 1.0f);
    std::fill(problem.past_kv_buffer.begin(), problem.past_kv_buffer.end(), 1.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    for (int block = 0; block < problem.state_capacity; ++block) {
      const float value = static_cast<float>(block + 1) / problem.state_capacity;
      std::fill_n(problem.past_key_state.begin() + static_cast<size_t>(block) * problem.head_size,
                  problem.head_size, value);
    }
    RunQsaPackedTest<float>(1.0e-5f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<MLFloat16>(2.0e-3f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaCompactChunkAndFallbackBoundaries) {
  for (int live_blocks : {511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 8193, 10240, 10241}) {
    SCOPED_TRACE(live_blocks);
    QsaPackedProblem problem;
    problem.batch_size = 2;
    problem.cumulative_sequence_lengths = {0, 2, 3};
    problem.past_sequence_lengths = {live_blocks * 4, 0};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = std::max(2051, live_blocks + 1);
    problem.max_position = live_blocks * 4 + 4;
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 0.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    RunQsaPackedTest<BFloat16>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaRaggedPrefill) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 65};
  problem.past_sequence_lengths = {1024};
  problem.head_size = 128;
  problem.num_heads = 4;
  problem.rotary_width = 64;
  problem.compress_ratio = 4;
  problem.token_budget = 128;
  problem.state_capacity = 512;
  problem.max_position = 2048;
  problem = MakeQsaPackedProblem(std::move(problem));
  std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
  std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
  std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
  RunQsaPackedTest<float>(1.0e-4f, std::move(problem));
}

TEST(PackedSparseAttentionIndexerTest, QsaMalformedStateLengthsAreClamped) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {3, -1};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalTileBoundariesPreserveStableTies) {
  for (int live_blocks : {0, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 2051}) {
    SCOPED_TRACE(live_blocks);
    QsaPackedProblem problem;
    problem.batch_size = 2;
    problem.cumulative_sequence_lengths = {0, 1, 2};
    problem.past_sequence_lengths = {live_blocks * 4, std::max(0, live_blocks - 1) * 4};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = 2051;
    problem.max_position = std::max(32, live_blocks * 4 + 2);
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 0.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    RunQsaPackedTest<MLFloat16>(2.0e-3f, problem);
    RunQsaPackedTest<BFloat16>(2.0e-2f, std::move(problem));
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalPackedRowsWithLargeCapacity) {
  for (int tokens : {1, 2, 3, 8, 64}) {
    SCOPED_TRACE(tokens);
    QsaPackedProblem problem;
    problem.batch_size = 1;
    problem.cumulative_sequence_lengths = {0, tokens};
    problem.past_sequence_lengths = {4097 * 4 + 3};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = 65536;
    problem.max_position = problem.past_sequence_lengths[0] + tokens + 1;
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 0.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    RunQsaPackedTest<MLFloat16>(2.0e-3f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalMergeLevelsWithPartialGroups) {
  for (int capacity : {513, 1025, 2051, 4097}) {
    SCOPED_TRACE(capacity);
    QsaPackedProblem problem;
    problem.batch_size = 2;
    problem.cumulative_sequence_lengths = {0, 2, 3};
    problem.past_sequence_lengths = {(capacity - 1) * 4, 33 * 4};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = capacity;
    problem.max_position = capacity * 4 + 4;
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 1.0f);
    std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    for (int batch = 0; batch < problem.batch_size; ++batch) {
      for (int block = 0; block < capacity; ++block) {
        const float value = static_cast<float>((block * 73) % capacity + 1) / capacity;
        std::fill_n(problem.past_key_state.begin() +
                        (static_cast<size_t>(batch) * capacity + block) * problem.head_size,
                    problem.head_size, value);
      }
    }
    RunQsaPackedTest<MLFloat16>(2.0e-3f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

TEST(PackedSparseAttentionIndexerTest, QsaHierarchicalScoringReusesKeysAcrossHeads) {
  for (int live_blocks : {4095, 4096, 4097, 8191}) {
    SCOPED_TRACE(live_blocks);
    QsaPackedProblem problem;
    problem.batch_size = 2;
    problem.cumulative_sequence_lengths = {0, 2, 3};
    problem.past_sequence_lengths = {live_blocks * 4, 33 * 4};
    problem.head_size = 128;
    problem.num_heads = 4;
    problem.rotary_width = 64;
    problem.compress_ratio = 4;
    problem.token_budget = 2048;
    problem.state_capacity = 8193;
    problem.max_position = live_blocks * 4 + 4;
    problem = MakeQsaPackedProblem(std::move(problem));
    std::fill(problem.query.begin(), problem.query.end(), 1.0f);
    std::fill(problem.query_norm_weight.begin(), problem.query_norm_weight.end(), 1.0f);
    std::fill(problem.cos_cache.begin(), problem.cos_cache.end(), 1.0f);
    std::fill(problem.sin_cache.begin(), problem.sin_cache.end(), 0.0f);
    for (int batch = 0; batch < problem.batch_size; ++batch) {
      for (int block = 0; block < problem.state_capacity; ++block) {
        const float value = static_cast<float>((block * 73) % problem.state_capacity + 1) /
                            problem.state_capacity;
        std::fill_n(problem.past_key_state.begin() +
                        (static_cast<size_t>(batch) * problem.state_capacity + block) * problem.head_size,
                    problem.head_size, value);
      }
    }
    RunQsaPackedTest<float>(1.0e-5f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<MLFloat16>(2.0e-3f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16>(2.0e-2f, problem, ProviderKind::Cuda);
    RunQsaPackedTest<BFloat16, float>(2.0e-2f, std::move(problem), ProviderKind::Cuda);
  }
}

// state_status is optional: requesting it must not change any other output, and every healthy
// request reports 0.
TEST(PackedSparseAttentionIndexerTest, QsaStateStatusOkForValidRequests) {
  QsaPackedProblem problem;
  problem.report_state_status = true;
  QsaPackedResult with_status;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(problem), ProviderKind::Cuda, &with_status);

  problem.report_state_status = false;
  QsaPackedResult without_status;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(problem), ProviderKind::Cuda, &without_status);
  EXPECT_EQ(with_status.selected_indices, without_status.selected_indices);
  EXPECT_EQ(with_status.selected_counts, without_status.selected_counts);
  EXPECT_EQ(with_status.present_key_state, without_status.present_key_state);
  EXPECT_EQ(with_status.present_kv_buffer, without_status.present_kv_buffer);
  EXPECT_EQ(with_status.present_state_lengths, without_status.present_state_lengths);
}

TEST(PackedSparseAttentionIndexerTest, QsaStateStatusReportsCapacityOverflow) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 6};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {2, 0};
  problem.report_state_status = true;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

// Cursor inconsistent with past_sequence_lengths: the request is rejected (status 1) and the healthy
// request next to it is unaffected (status 0).
TEST(PackedSparseAttentionIndexerTest, QsaStateStatusReportsInvalidCursor) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.cumulative_sequence_lengths = {0, 2, 4};
  problem.past_sequence_lengths = {5, 5};
  problem.past_state_lengths = {2, 1, 1, 1};  // request 1: complete-block count 1 != 5 / 2
  problem.report_state_status = true;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

TEST(PackedSparseAttentionIndexerTest, QsaStateStatusMixedRequests) {
  QsaPackedProblem problem;
  problem.batch_size = 3;
  problem.cumulative_sequence_lengths = {0, 2, 4, 6};
  problem.past_sequence_lengths = {5, 5, 3};
  problem.past_state_lengths = {2, 1, 1, 1, 1, 1};
  problem.state_capacity = 3;  // requests 0 and 2 fit their new block; request 1's cursor is inconsistent
  problem.report_state_status = true;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(problem));

  problem.past_sequence_lengths = {5, 5, 5};
  problem.past_state_lengths = {2, 1, 1, 1, 2, 1};
  problem.state_capacity = 2;  // requests 0 and 2 are full and would close a block; request 1 is inconsistent
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

TEST(PackedSparseAttentionIndexerTest, QsaStateStatusReportsMalformedStateLengths) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {3, -1};
  problem.report_state_status = true;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)));
}

namespace {

// Two query rows of one request (a two-token verify step) must produce exactly what two chained
// single-row calls produce: the same selections and the same final state, bit for bit.
void RunQsaTwoRowChainTest(int compress_ratio, int past_length) {
  QsaPackedProblem both;
  both.batch_size = 1;
  both.compress_ratio = compress_ratio;
  both.token_budget = 2 * compress_ratio;
  both.state_capacity = 16;
  both.cumulative_sequence_lengths = {0, 2};
  both.past_sequence_lengths = {past_length};
  both = MakeQsaPackedProblem(std::move(both));
  QsaPackedResult chained_result;
  RunQsaPackedTest<float>(1.0e-5f, both, ProviderKind::Cuda, &chained_result);

  const size_t query_row = static_cast<size_t>(both.num_heads) * both.head_size;
  const size_t key_row = static_cast<size_t>(both.head_size);
  QsaPackedProblem first = both;
  first.cumulative_sequence_lengths = {0, 1};
  first.query.assign(both.query.begin(), both.query.begin() + query_row);
  first.key.assign(both.key.begin(), both.key.begin() + key_row);
  QsaPackedResult first_result;
  RunQsaPackedTest<float>(1.0e-5f, first, ProviderKind::Cuda, &first_result);

  QsaPackedProblem second = both;
  second.cumulative_sequence_lengths = {0, 1};
  second.past_sequence_lengths = {past_length + 1};
  second.query.assign(both.query.begin() + query_row, both.query.end());
  second.key.assign(both.key.begin() + key_row, both.key.end());
  second.past_key_state = first_result.present_key_state;
  second.past_kv_buffer = first_result.present_kv_buffer;
  second.past_state_lengths = first_result.present_state_lengths;
  QsaPackedResult second_result;
  RunQsaPackedTest<float>(1.0e-5f, second, ProviderKind::Cuda, &second_result);

  std::vector<int32_t> expected_indices = first_result.selected_indices;
  expected_indices.insert(expected_indices.end(), second_result.selected_indices.begin(),
                          second_result.selected_indices.end());
  std::vector<int32_t> expected_counts = first_result.selected_counts;
  expected_counts.insert(expected_counts.end(), second_result.selected_counts.begin(),
                         second_result.selected_counts.end());
  EXPECT_EQ(chained_result.selected_indices, expected_indices);
  EXPECT_EQ(chained_result.selected_counts, expected_counts);
  EXPECT_EQ(chained_result.present_state_lengths, second_result.present_state_lengths);
  EXPECT_EQ(chained_result.present_key_state, second_result.present_key_state);

  // Slots at or beyond the pending length are stale and are not part of the state contract.
  const size_t pending = static_cast<size_t>(chained_result.present_state_lengths[1]);
  for (size_t t = 0; t < pending; ++t) {
    for (size_t d = 0; d < key_row; ++d) {
      const size_t at = t * key_row + d;
      EXPECT_EQ(chained_result.present_kv_buffer[at], second_result.present_kv_buffer[at]);
    }
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerTest, QsaTwoRowStepEqualsChainedSingleRowSteps) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  for (int past_length : {11, 12, 13}) {  // the first row closes a block, the second does, or neither does
    RunQsaTwoRowChainTest(2, past_length);
  }
  for (int past_length : {14, 15, 16}) {
    RunQsaTwoRowChainTest(4, past_length);
  }
}

#ifdef USE_WEBGPU
TEST(PackedSparseAttentionIndexerWebGpuTest, QsaFloat) {
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaFloat16) {
  RunQsaPackedTest<MLFloat16>(2.0e-3f, MakeQsaPackedProblem(), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaPackedQueryKey) {
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::WebGpu, nullptr, true);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaStateUpdateCapturesRawAndBlockTransitions) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 3};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaStateUpdateActiveZeroClearsAllSlots) {
  QsaPackedProblem problem;
  problem.batch_size = 2;
  problem.compress_ratio = 4;
  problem.cumulative_sequence_lengths = {0, 8, 13};
  problem.past_sequence_lengths = {0, 3};
  problem.state_update_capacity = 8;
  problem.state_update_capture_count = {8, 5};
  problem.state_update_active = 0;
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaPortableStorageBindingLimit) {
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kMaxStorageBuffersPerShaderStage, "8"));
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(), ProviderKind::WebGpu, nullptr, false, &config_options);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaAllEmptyBatchPreservesState) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 0};
  problem.past_sequence_lengths = {5};
  problem.past_state_lengths = {2, 1};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaStateCapacityOverflowIsRejected) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 6};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {2, 0};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, QsaMalformedStateLengthsAreClamped) {
  QsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {4};
  problem.state_capacity = 2;
  problem.past_state_lengths = {3, -1};
  RunQsaPackedTest<float>(1.0e-5f, MakeQsaPackedProblem(std::move(problem)), ProviderKind::WebGpu);
}
#endif

namespace {

struct CsaPackedProblem {
  int batch_size = 2;
  std::vector<int32_t> cumulative_sequence_lengths;
  std::vector<int32_t> past_sequence_lengths;  // unused by csa math, kept for symmetry with qsa
  std::vector<int32_t> past_state_lengths;     // [batch, 2]: compressed count, buffer length
  int head_size = 4;
  int num_heads = 2;
  int rotary_width = 2;
  int compress_ratio = 2;
  int index_topk = 3;
  int state_capacity = 4;
  int max_position = 64;
  float epsilon = 1.0e-6f;
  std::optional<float> scale;
  std::optional<float> head_weight_scale;
  bool report_state_status = false;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> query_norm_weight;
  std::vector<float> key_norm_weight;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<float> gate;
  std::vector<float> position_bias;
  std::vector<float> head_weights;
  std::vector<int64_t> position_ids;
  std::vector<float> past_key_state;
  std::vector<float> past_kv_buffer;
  std::vector<float> past_gate_buffer;

  int Width() const { return 2 * head_size; }
  int TotalTokens() const { return cumulative_sequence_lengths.back(); }
  int BufferCapacity() const { return 2 * compress_ratio - 1; }
};

struct CsaPackedResult {
  std::vector<int32_t> selected_indices;
  std::vector<int32_t> selected_counts;
  std::vector<float> present_key_state;
  std::vector<float> present_kv_buffer;
  std::vector<float> present_gate_buffer;
  std::vector<int32_t> present_state_lengths;
  std::vector<int32_t> state_status;
};

void CsaPackedReference(const CsaPackedProblem& p, CsaPackedResult& out) {
  const int total_tokens = p.TotalTokens();
  const int head_size = p.head_size;
  const int width = p.Width();
  const int buffer_capacity = p.BufferCapacity();
  const int capacity = p.index_topk;
  const float scale = p.scale.value_or(1.0f / std::sqrt(static_cast<float>(head_size)));
  const float head_weight_scale = p.head_weight_scale.value_or(1.0f / std::sqrt(static_cast<float>(p.num_heads)));

  out.present_key_state = p.past_key_state;
  out.present_kv_buffer = p.past_kv_buffer;
  out.present_gate_buffer = p.past_gate_buffer;
  out.present_state_lengths = p.past_state_lengths;
  std::vector<int> key_len_after(static_cast<size_t>(p.batch_size));
  std::vector<bool> overflowed(static_cast<size_t>(p.batch_size), false);
  out.state_status.assign(static_cast<size_t>(p.batch_size), 0);

  for (int b = 0; b < p.batch_size; ++b) {
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    const int raw_key_len = p.past_state_lengths[static_cast<size_t>(b) * 2 + 0];
    const int raw_buf_len = p.past_state_lengths[static_cast<size_t>(b) * 2 + 1];
    const bool invalid_metadata =
        p.cumulative_sequence_lengths[0] != 0 ||
        p.cumulative_sequence_lengths[static_cast<size_t>(p.batch_size)] != total_tokens ||
        req_start < 0 || req_end < req_start || req_end > total_tokens ||
        p.past_sequence_lengths[static_cast<size_t>(b)] < 0 ||
        raw_key_len < 0 || raw_key_len > p.state_capacity ||
        raw_buf_len < 0 || raw_buf_len > buffer_capacity;
    const int req_len = invalid_metadata ? 0 : req_end - req_start;
    const int old_key_len = std::clamp(raw_key_len, 0, p.state_capacity);
    const int old_buf_len = std::clamp(raw_buf_len, 0, buffer_capacity);
    const int overlap_length = old_buf_len >= p.compress_ratio ? p.compress_ratio : 0;
    const int leftover_length = old_buf_len - overlap_length;
    const int pending = leftover_length + req_len;
    const int full_new_window_count = pending / p.compress_ratio;
    overflowed[static_cast<size_t>(b)] =
        invalid_metadata || full_new_window_count > std::max(p.state_capacity - old_key_len, 0);
    out.state_status[static_cast<size_t>(b)] =
        invalid_metadata ? psai::kStateStatusInvalidMetadata
                         : (overflowed[static_cast<size_t>(b)] ? psai::kStateStatusCapacityOverflow : 0);
    if (overflowed[static_cast<size_t>(b)]) {
      out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len;
      out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = old_buf_len;
      key_len_after[static_cast<size_t>(b)] = old_key_len;
      continue;
    }
    const int new_window_count = full_new_window_count;
    int present_buffer_length = 0;
    int present_buffer_start = 0;
    if (new_window_count > 0) {
      present_buffer_length = p.compress_ratio + pending % p.compress_ratio;
      present_buffer_start = overlap_length + (new_window_count - 1) * p.compress_ratio;
    } else {
      present_buffer_length = old_buf_len + req_len;
      present_buffer_start = 0;
    }
    present_buffer_length = std::min(present_buffer_length, buffer_capacity);

    auto extended_key = [&](int virtual_pos, int channel) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * width + channel]
                 : p.key[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * width + channel];
    };
    auto extended_gate = [&](int virtual_pos, int channel) -> float {
      return virtual_pos < old_buf_len
                 ? p.past_gate_buffer[(static_cast<size_t>(b) * buffer_capacity + virtual_pos) * width + channel]
                 : p.gate[(static_cast<size_t>(req_start) + (virtual_pos - old_buf_len)) * width + channel];
    };

    for (int k = 0; k < new_window_count; ++k) {
      const bool has_previous = k >= 1 || overlap_length >= p.compress_ratio;
      const int previous_base = overlap_length + (k - 1) * p.compress_ratio;
      const int current_base = overlap_length + k * p.compress_ratio;
      std::vector<float> pooled(static_cast<size_t>(head_size), 0.0f);
      for (int d = 0; d < head_size; ++d) {
        float max_gate = -std::numeric_limits<float>::infinity();
        if (has_previous) {
          for (int slot = 0; slot < p.compress_ratio; ++slot) {
            max_gate = std::max(max_gate, extended_gate(previous_base + slot, d) + p.position_bias[static_cast<size_t>(slot) * width + d]);
          }
        }
        for (int slot = 0; slot < p.compress_ratio; ++slot) {
          max_gate = std::max(max_gate, extended_gate(current_base + slot, head_size + d) +
                                            p.position_bias[static_cast<size_t>(slot) * width + head_size + d]);
        }
        float denom = 0.0f, acc = 0.0f;
        if (has_previous) {
          for (int slot = 0; slot < p.compress_ratio; ++slot) {
            const float logit = extended_gate(previous_base + slot, d) + p.position_bias[static_cast<size_t>(slot) * width + d];
            const float w = std::exp(logit - max_gate);
            denom += w;
            acc += w * extended_key(previous_base + slot, d);
          }
        }
        for (int slot = 0; slot < p.compress_ratio; ++slot) {
          const float logit = extended_gate(current_base + slot, head_size + d) +
                              p.position_bias[static_cast<size_t>(slot) * width + head_size + d];
          const float w = std::exp(logit - max_gate);
          denom += w;
          acc += w * extended_key(current_base + slot, head_size + d);
        }
        pooled[static_cast<size_t>(d)] = denom > 0.0f ? acc / denom : 0.0f;
      }
      pooled = RmsNormalize(pooled, p.key_norm_weight, p.epsilon);
      const int entry = old_key_len + k;
      const int position = std::min(entry * p.compress_ratio, p.max_position - 1);
      pooled = TrailingRope(pooled, p.rotary_width, p.cos_cache.data() + position * p.rotary_width,
                            p.sin_cache.data() + position * p.rotary_width);
      for (int d = 0; d < head_size; ++d) {
        out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + entry) * head_size + d] =
            pooled[static_cast<size_t>(d)];
      }
    }
    for (int t = 0; t < present_buffer_length; ++t) {
      const int virtual_pos = present_buffer_start + t;
      for (int c = 0; c < width; ++c) {
        out.present_kv_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * width + c] = extended_key(virtual_pos, c);
        out.present_gate_buffer[(static_cast<size_t>(b) * buffer_capacity + t) * width + c] = extended_gate(virtual_pos, c);
      }
    }
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 0] = old_key_len + new_window_count;
    out.present_state_lengths[static_cast<size_t>(b) * 2 + 1] = present_buffer_length;
    key_len_after[static_cast<size_t>(b)] = old_key_len + new_window_count;
  }

  out.selected_indices.assign(static_cast<size_t>(total_tokens) * capacity, -1);
  out.selected_counts.assign(static_cast<size_t>(total_tokens), 0);
  for (int b = 0; b < p.batch_size; ++b) {
    if (overflowed[static_cast<size_t>(b)]) {
      continue;
    }
    const int req_start = p.cumulative_sequence_lengths[static_cast<size_t>(b)];
    const int req_end = p.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
    for (int token = req_start; token < req_end; ++token) {
      const int position = static_cast<int>(p.position_ids[static_cast<size_t>(token)]);
      const int threshold = std::min(CausalThreshold(position, p.compress_ratio), key_len_after[static_cast<size_t>(b)]);
      const int selected = std::min(p.index_topk, threshold);

      std::vector<std::vector<float>> rotated_query(static_cast<size_t>(p.num_heads));
      for (int h = 0; h < p.num_heads; ++h) {
        const size_t base = (static_cast<size_t>(token) * p.num_heads + h) * head_size;
        std::vector<float> head(p.query.begin() + base, p.query.begin() + base + head_size);
        head = RmsNormalize(head, p.query_norm_weight, p.epsilon);
        const int clamped_position = std::min(std::max(position, 0), p.max_position - 1);
        rotated_query[static_cast<size_t>(h)] =
            TrailingRope(head, p.rotary_width, p.cos_cache.data() + clamped_position * p.rotary_width,
                         p.sin_cache.data() + clamped_position * p.rotary_width);
      }

      std::vector<float> scores(static_cast<size_t>(threshold), 0.0f);
      for (int e = 0; e < threshold; ++e) {
        float score = 0.0f;
        for (int h = 0; h < p.num_heads; ++h) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; ++d) {
            dot += rotated_query[static_cast<size_t>(h)][static_cast<size_t>(d)] *
                   out.present_key_state[(static_cast<size_t>(b) * p.state_capacity + e) * head_size + d];
          }
          score += std::max(dot, 0.0f) * p.head_weights[static_cast<size_t>(token) * p.num_heads + h];
        }
        scores[static_cast<size_t>(e)] = score * scale * head_weight_scale;
      }
      const std::vector<int> order = RankByScore(scores, threshold);
      int32_t* out_row = out.selected_indices.data() + static_cast<size_t>(token) * capacity;
      for (int rank = 0; rank < selected; ++rank) {
        out_row[rank] = order[static_cast<size_t>(rank)];
      }
      out.selected_counts[static_cast<size_t>(token)] = selected;
    }
  }
}

CsaPackedProblem MakeCsaPackedProblem(CsaPackedProblem problem = {}) {
  if (problem.cumulative_sequence_lengths.empty()) {
    problem.cumulative_sequence_lengths = {0, 2, 5};
  }
  if (problem.past_state_lengths.empty()) {
    problem.past_state_lengths.assign(static_cast<size_t>(problem.batch_size) * 2, 0);
  }
  if (problem.past_sequence_lengths.empty()) {
    problem.past_sequence_lengths.assign(static_cast<size_t>(problem.batch_size), 0);
  }
  const int total_tokens = problem.TotalTokens();
  const int width = problem.Width();
  const int buffer_capacity = problem.BufferCapacity();

  problem.query = MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads * problem.head_size, 0.25f, 0.37f);
  problem.key = MakeWave(static_cast<size_t>(total_tokens) * width, 0.60f, 0.21f);
  problem.query_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 1.25f, 0.19f);
  problem.key_norm_weight = MakeWave(static_cast<size_t>(problem.head_size), 0.45f, 0.31f);
  problem.cos_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 0.15f, 0.27f);
  problem.sin_cache = MakeWave(static_cast<size_t>(problem.max_position) * problem.rotary_width, 1.05f, 0.33f);
  problem.gate = MakeWave(static_cast<size_t>(total_tokens) * width, 0.80f, 0.24f);
  problem.position_bias = MakeWave(static_cast<size_t>(problem.compress_ratio) * width, 0.33f, 0.11f);
  problem.head_weights = MakeWave(static_cast<size_t>(total_tokens) * problem.num_heads, 1.30f, 0.47f);
  problem.past_key_state =
      MakeWave(static_cast<size_t>(problem.batch_size) * problem.state_capacity * problem.head_size, 0.50f, 0.39f);
  problem.past_kv_buffer = MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * width, 0.95f, 0.18f);
  problem.past_gate_buffer = MakeWave(static_cast<size_t>(problem.batch_size) * buffer_capacity * width, 1.45f, 0.22f);

  if (problem.position_ids.empty()) {
    problem.position_ids.assign(static_cast<size_t>(total_tokens), 0);
    for (int b = 0; b < problem.batch_size; ++b) {
      const int req_start = problem.cumulative_sequence_lengths[static_cast<size_t>(b)];
      const int req_end = problem.cumulative_sequence_lengths[static_cast<size_t>(b) + 1];
      for (int token = req_start; token < req_end; ++token) {
        problem.position_ids[static_cast<size_t>(token)] = 2 + (token - req_start);
      }
    }
  }
  return problem;
}

template <typename T>
void RunCsaPackedTest(const CsaPackedProblem& base, float tolerance,
                      ProviderKind provider_kind = ProviderKind::Cuda, CsaPackedResult* actual = nullptr,
                      const ConfigOptions* webgpu_options = nullptr, bool reuse = false, bool reuse_failure = false,
                      int32_t merged_status = 0, int32_t merged_count = 1) {
  auto provider = CreateProvider(provider_kind, webgpu_options);
  if (provider == nullptr) {
    GTEST_SKIP() << (provider_kind == ProviderKind::Cuda ? "CUDA" : "WebGPU")
                 << " execution provider is not available";
  }

  CsaPackedProblem problem = base;
  problem.query = RoundTrip<T>(problem.query);
  problem.key = RoundTrip<T>(problem.key);
  problem.query_norm_weight = RoundTrip<T>(problem.query_norm_weight);
  problem.key_norm_weight = RoundTrip<T>(problem.key_norm_weight);
  problem.cos_cache = RoundTrip<T>(problem.cos_cache);
  problem.sin_cache = RoundTrip<T>(problem.sin_cache);
  problem.gate = RoundTrip<T>(problem.gate);
  problem.position_bias = RoundTrip<T>(problem.position_bias);
  problem.head_weights = RoundTrip<T>(problem.head_weights);
  problem.past_key_state = RoundTrip<T>(problem.past_key_state);
  problem.past_kv_buffer = RoundTrip<T>(problem.past_kv_buffer);
  problem.past_gate_buffer = RoundTrip<T>(problem.past_gate_buffer);

  CsaPackedResult expected;
  CsaPackedReference(problem, expected);

  const int64_t total_tokens = problem.TotalTokens();
  const int64_t batch_size = problem.batch_size;
  const int64_t head_size = problem.head_size;
  const int64_t width = problem.Width();
  const int64_t buffer_capacity = problem.BufferCapacity();
  const int64_t output_capacity = problem.index_topk + (reuse ? 6 : 0);
  std::vector<int32_t> merged;
  if (reuse) {
    ASSERT_EQ(total_tokens, batch_size);
    merged.assign(static_cast<size_t>(batch_size * output_capacity), -1);
    for (int64_t row = 0; row < batch_size; ++row) merged[static_cast<size_t>(row * output_capacity)] = 0;
    expected.selected_indices = merged;
    expected.selected_counts.assign(static_cast<size_t>(batch_size), 1);
    if (reuse_failure) {
      expected.selected_indices.assign(static_cast<size_t>(batch_size * output_capacity), -1);
      expected.selected_counts.assign(static_cast<size_t>(batch_size), 0);
      expected.present_key_state = problem.past_key_state;
      expected.present_kv_buffer = problem.past_kv_buffer;
      expected.present_gate_buffer = problem.past_gate_buffer;
      expected.present_state_lengths = problem.past_state_lengths;
    }
  }

  OpTester test("PackedSparseAttentionIndexer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("policy_mode", std::string(psai::kPolicyModeCsa));
  test.AddAttribute("compress_ratio", static_cast<int64_t>(problem.compress_ratio));
  test.AddAttribute("state_capacity", static_cast<int64_t>(problem.state_capacity));
  test.AddAttribute("index_topk", static_cast<int64_t>(problem.index_topk));
  if (problem.scale.has_value()) test.AddAttribute("scale", *problem.scale);
  if (problem.head_weight_scale.has_value()) test.AddAttribute("head_weight_scale", *problem.head_weight_scale);

  test.AddInput<T>("query", {reuse ? 0 : total_tokens, problem.num_heads * head_size},
                   reuse ? std::vector<T>{} : ToElementType<T>(problem.query));
  test.AddInput<T>("key", {total_tokens, width}, ToElementType<T>(problem.key));
  test.AddInput<T>("query_norm_weight", {head_size}, ToElementType<T>(problem.query_norm_weight));
  test.AddInput<T>("key_norm_weight", {head_size}, ToElementType<T>(problem.key_norm_weight));
  test.AddInput<T>("cos_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.cos_cache));
  test.AddInput<T>("sin_cache", {problem.max_position, problem.rotary_width}, ToElementType<T>(problem.sin_cache));
  test.AddInput<int32_t>("cumulative_sequence_lengths", {batch_size + 1}, problem.cumulative_sequence_lengths);
  test.AddInput<int32_t>("past_sequence_lengths", {batch_size}, problem.past_sequence_lengths);
  test.AddInput<T>("gate", {total_tokens, width}, ToElementType<T>(problem.gate));
  test.AddInput<T>("position_bias", {problem.compress_ratio, width}, ToElementType<T>(problem.position_bias));
  test.AddInput<T>("head_weights", {reuse ? 0 : total_tokens, problem.num_heads},
                   reuse ? std::vector<T>{} : ToElementType<T>(problem.head_weights));
  test.AddInput<int64_t>("position_ids", {total_tokens}, problem.position_ids);
  test.AddInput<T>("past_key_state", {batch_size, problem.state_capacity, head_size},
                   ToElementType<T>(problem.past_key_state));
  test.AddInput<T>("past_kv_buffer", {batch_size, buffer_capacity, width}, ToElementType<T>(problem.past_kv_buffer));
  test.AddInput<T>("past_gate_buffer", {batch_size, buffer_capacity, width},
                   ToElementType<T>(problem.past_gate_buffer));
  test.AddInput<int32_t>("past_state_lengths", {batch_size, 2}, problem.past_state_lengths);

  if (reuse) {
    test.AddAttribute("max_output_entries", output_capacity);
    test.AddOptionalInputEdge<int32_t>();
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<int32_t>("mode", {1}, {1});
    test.AddInput<int32_t>("merged", {batch_size, output_capacity}, merged);
    test.AddInput<int32_t>("merged_counts", {batch_size}, std::vector<int32_t>(static_cast<size_t>(batch_size), merged_count));
    test.AddInput<int32_t>("merged_status", {batch_size}, std::vector<int32_t>(static_cast<size_t>(batch_size), merged_status));
  }
  test.AddOutput<int32_t>("selected_indices", {total_tokens, output_capacity}, expected.selected_indices);
  test.AddOutput<int32_t>("selected_counts", {total_tokens}, expected.selected_counts);
  test.AddOutput<T>("present_key_state", {batch_size, problem.state_capacity, head_size},
                    ToElementType<T>(expected.present_key_state), false, 0.0f, tolerance);
  test.AddOutput<T>("present_kv_buffer", {batch_size, buffer_capacity, width},
                    ToElementType<T>(expected.present_kv_buffer), false, 0.0f, tolerance);
  test.AddOutput<T>("present_gate_buffer", {batch_size, buffer_capacity, width},
                    ToElementType<T>(expected.present_gate_buffer), false, 0.0f, tolerance);
  test.AddOutput<int32_t>("present_state_lengths", {batch_size, 2}, expected.present_state_lengths);
  if (reuse) {
    test.AddOptionalOutputEdge<T>();
    test.AddOutput<int32_t>("status", {batch_size}, std::vector<int32_t>(static_cast<size_t>(batch_size), merged_status != 0 ? merged_status : reuse_failure ? 1
                                                                                                                                                             : 0));
  }
  if (problem.report_state_status) {
    if (!reuse) {
      test.AddOptionalOutputEdge<T>();        // state_update
      test.AddOptionalOutputEdge<int32_t>();  // indexshare_status
    }
    test.AddOutput<int32_t>("state_status", {batch_size}, expected.state_status);
  }
  RunOnProvider(test, std::move(provider));
  if (actual != nullptr) {
    const auto& fetches = test.GetFetches();
    actual->present_key_state = CopyTensorToFloat<T>(fetches[2].Get<Tensor>());
    actual->present_kv_buffer = CopyTensorToFloat<T>(fetches[3].Get<Tensor>());
    actual->present_gate_buffer = CopyTensorToFloat<T>(fetches[4].Get<Tensor>());
    actual->present_state_lengths.assign(fetches[5].Get<Tensor>().DataAsSpan<int32_t>().begin(),
                                         fetches[5].Get<Tensor>().DataAsSpan<int32_t>().end());
  }
}

}  // namespace

TEST(PackedSparseAttentionIndexerTest, CsaFloat) { RunCsaPackedTest<float>(MakeCsaPackedProblem(), 1.0e-5f); }

TEST(PackedSparseAttentionIndexerTest, IndexShareCsaAdvanceWithoutScoring) {
  CsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 1, 2};
  problem.past_sequence_lengths = {3, 1};
  problem.past_state_lengths = {1, 3, 1, 1};
  problem = MakeCsaPackedProblem(std::move(problem));
  RunCsaPackedTest<float>(problem, 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true);
  RunCsaPackedTest<MLFloat16>(problem, 4.0e-3f, ProviderKind::Cuda, nullptr, nullptr, true);
}

TEST(PackedSparseAttentionIndexerTest, IndexShareCsaRejectsOffsetsAndStateOverflow) {
  CsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 0, 2};
  problem.past_sequence_lengths = {3, 3};
  problem.past_state_lengths = {1, 3, 1, 3};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(problem), 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true);
  problem.cumulative_sequence_lengths = {0, 1, 2};
  problem.state_capacity = 1;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(problem), 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true);
}

TEST(PackedSparseAttentionIndexerTest, IndexShareCsaRejectsInvalidSelectionsBeforeAdvancing) {
  CsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 1, 2};
  problem.past_sequence_lengths = {3, 1};
  problem.past_state_lengths = {1, 3, 1, 1};
  problem = MakeCsaPackedProblem(std::move(problem));
  RunCsaPackedTest<float>(problem, 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true, 2, 1);
  RunCsaPackedTest<float>(problem, 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true, 0, -1);
  RunCsaPackedTest<float>(problem, 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true, 0, INT32_MAX);
}

TEST(PackedSparseAttentionIndexerTest, CsaFloat16) { RunCsaPackedTest<MLFloat16>(MakeCsaPackedProblem(), 4.0e-3f); }

TEST(PackedSparseAttentionIndexerTest, CsaBFloat16) { RunCsaPackedTest<BFloat16>(MakeCsaPackedProblem(), 3.0e-2f); }

TEST(PackedSparseAttentionIndexerTest, CsaLargeRatioCompaction) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.compress_ratio = 64;
  problem.cumulative_sequence_lengths = {0, 64};
  problem.past_sequence_lengths = {65};
  problem.state_capacity = 4;
  problem.past_state_lengths = {0, 65};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, CsaPrefillThenDecodeIndependentState) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  CsaPackedProblem prefill;
  prefill.cumulative_sequence_lengths = {0, 3, 5};
  prefill.past_sequence_lengths = {0, 0};
  CsaPackedResult prefill_outputs;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(prefill)), 1.0e-5f, ProviderKind::Cuda,
                          &prefill_outputs);

  CsaPackedProblem decode;
  decode.cumulative_sequence_lengths = {0, 1, 3};
  decode.past_sequence_lengths = {3, 2};
  decode.position_ids = {3, 2, 3};
  decode = MakeCsaPackedProblem(std::move(decode));
  decode.past_key_state = std::move(prefill_outputs.present_key_state);
  decode.past_kv_buffer = std::move(prefill_outputs.present_kv_buffer);
  decode.past_gate_buffer = std::move(prefill_outputs.present_gate_buffer);
  decode.past_state_lengths = std::move(prefill_outputs.present_state_lengths);
  RunCsaPackedTest<float>(decode, 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, CsaStateCapacityOverflowIsRejected) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 4};
  problem.state_capacity = 1;
  problem.past_state_lengths = {1, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, CsaMalformedStateLengthsAreClamped) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {0};
  problem.state_capacity = 2;
  problem.past_state_lengths = {3, 8};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, CsaStateStatusOkAndReportsFailures) {
  CsaPackedProblem problem;
  problem.report_state_status = true;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(problem), 1.0e-5f);

  // Capacity overflow next to a healthy request.
  problem.batch_size = 2;
  problem.cumulative_sequence_lengths = {0, 4, 6};
  problem.state_capacity = 2;
  problem.past_state_lengths = {1, 0, 0, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(problem), 1.0e-5f);

  // Malformed state length.
  problem.cumulative_sequence_lengths = {0, 1, 2};
  problem.past_state_lengths = {3, 8, 0, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f);
}

TEST(PackedSparseAttentionIndexerTest, IndexShareCsaAdvanceReportsStateStatus) {
  CsaPackedProblem problem;
  problem.cumulative_sequence_lengths = {0, 1, 2};
  problem.past_sequence_lengths = {3, 3};
  problem.past_state_lengths = {1, 3, 1, 3};
  problem.state_capacity = 1;
  problem.report_state_status = true;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(problem), 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr, true, true);

  CsaPackedProblem empty_batch;
  empty_batch.batch_size = 0;
  empty_batch.cumulative_sequence_lengths = {0};
  empty_batch.report_state_status = true;
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(empty_batch)), 1.0e-5f, ProviderKind::Cuda, nullptr, nullptr,
                          true);
}

#ifdef USE_WEBGPU
TEST(PackedSparseAttentionIndexerWebGpuTest, CsaFloat) {
  RunCsaPackedTest<float>(MakeCsaPackedProblem(), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaFloat16) {
  RunCsaPackedTest<MLFloat16>(MakeCsaPackedProblem(), 4.0e-3f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaPortableStorageBindingLimit) {
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kMaxStorageBuffersPerShaderStage, "8"));
  RunCsaPackedTest<float>(MakeCsaPackedProblem(), 1.0e-5f, ProviderKind::WebGpu, nullptr, &config_options);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaAllEmptyBatchPreservesState) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 0};
  problem.past_sequence_lengths = {3};
  problem.past_state_lengths = {1, 1};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaStateCapacityOverflowIsRejected) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 4};
  problem.state_capacity = 1;
  problem.past_state_lengths = {1, 0};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f, ProviderKind::WebGpu);
}

TEST(PackedSparseAttentionIndexerWebGpuTest, CsaMalformedStateLengthsAreClamped) {
  CsaPackedProblem problem;
  problem.batch_size = 1;
  problem.cumulative_sequence_lengths = {0, 1};
  problem.past_sequence_lengths = {0};
  problem.state_capacity = 2;
  problem.past_state_lengths = {3, 8};
  RunCsaPackedTest<float>(MakeCsaPackedProblem(std::move(problem)), 1.0e-5f, ProviderKind::WebGpu);
}
#endif

}  // namespace test
}  // namespace onnxruntime
