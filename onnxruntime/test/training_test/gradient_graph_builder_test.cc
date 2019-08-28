// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/optimizer/gist_encode_decode.h"
#include "test/providers/gradient_op_test_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/training/runner/training_runner.h"

#ifdef USE_CUDA
#include "bert_toy_fetches.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;

namespace onnxruntime {
namespace test {

constexpr auto ORIGINAL_MODEL_PATH = "testdata/test_training_model.onnx";
constexpr auto BACKWARD_MODEL_PATH = "testdata/temp_backward_model.onnx";
constexpr auto GIST_MODEL_PATH = "testdata/temp_backward_model_with_gist.onnx";

constexpr auto TAB = "\t";

AllocatorPtr GetAllocator() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
}
template <typename T>
static void CreateMLValue(const AllocatorPtr& alloc,
                          const std::vector<int64_t>& dims,
                          const std::vector<T>& value,
                          MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  if (!value.empty()) {
    memcpy(buffer, &value[0], element_type->Size() * shape.Size());
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

static std::string BuildBackPropGraph(const TrainingRunner::Parameters& params) {
  const std::string forward_model_file = params.model_path;
  const std::string backward_model_file = params.model_with_training_graph_path;

  std::unique_ptr<Environment> env;
  EXPECT_TRUE(Environment::Create(env).IsOK());

  SessionOptions so{};
  TrainingSession training_session{so};

  std::cout << "Loading source model file = " << forward_model_file << std::endl;

  EXPECT_TRUE(training_session.Load(forward_model_file).IsOK());

  std::unordered_set<std::string> weights_to_train =
      training_session.GetTrainableModelInitializers(params.immutable_weights);
  for (const auto& not_to_train : params.weights_not_to_train) {
    weights_to_train.erase(not_to_train);
  }

  std::cout << "Model weights = [" << std::endl;
  for (auto& n : weights_to_train) {
    std::cout << TAB << n << std::endl;
  }
  std::cout << "]" << std::endl;

  auto model_outputs = training_session.GetModelOutputNames();
  std::cout << "Model output names = [" << std::endl;
  for (auto& n : model_outputs) {
    std::cout << TAB << n << std::endl;
  }
  std::cout << "]" << std::endl;

  EXPECT_TRUE(training_session.BuildLossFunction(params.loss_func_info).IsOK());
  EXPECT_TRUE(training_session.BuildGradientGraph(weights_to_train, params.loss_func_info.loss_name, true).IsOK());

  if (params.use_gist) {
    EXPECT_TRUE(training_session.AddGistEncoding().IsOK());

    if (!params.model_gist_encode.empty()) {
      EXPECT_TRUE(training_session.Save(params.model_gist_encode, TrainingSession::SaveOption::NO_RELOAD).IsOK());
    }
  }

  EXPECT_TRUE(training_session.Save(backward_model_file,
                                    TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS)
                  .IsOK());

  return backward_model_file;
}

/**
 * Run a training session for this model for 1 epoch, using batch size of 1 and synthetic input data.
 * @param so - SessionOptions for this run.
 * @param backprop_model_file - Mocel file to be run. This should already contain loss function and backward prop nodes.
 * @return TrainingSession for this run.
 */
static std::unique_ptr<TrainingSession> RunTrainingSessionWithChecks(
    const SessionOptions& so, const std::string& backprop_model_file) {
  std::unique_ptr<Environment> env;
  EXPECT_TRUE(Environment::Create(env).IsOK());

  const auto& log_manager = so.session_log_verbosity_level > 0 ? &DefaultLoggingManager() : nullptr;

  std::unique_ptr<TrainingSession> training_session = std::make_unique<TrainingSession>(so, log_manager);

  EXPECT_TRUE(training_session->Load(backprop_model_file).IsOK());

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  EXPECT_TRUE(res.first.IsOK());
  EXPECT_TRUE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << std::endl;

  EXPECT_TRUE(training_session->Initialize().IsOK());

  std::vector<MLValue> gradient_fetches;
  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;

  // Create dummy feeds
  std::vector<int64_t> image_dims = {1, 784};
  std::vector<int64_t> label_dims = {1, 10};
  std::vector<float> image_value(784, 1);
  std::vector<float> label_value(10, 1);

  MLValue imageMLValue;
  CreateMLValue(GetAllocator(), image_dims, image_value, &imageMLValue);
  MLValue labelMLValue;
  CreateMLValue(GetAllocator(), label_dims, label_value, &labelMLValue);

  auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X", "labels"}, {imageMLValue, labelMLValue});

  auto output_names_include_gradients = training_session->GetModelOutputNames();
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  auto start_time = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(training_session->Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches).IsOK());

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed = TimeDiffMicroSeconds(start_time, end_time);
  std::cout << "Training session run completed in " << elapsed << " microseconds." << std::endl;

  return training_session;
}

TEST(GradientGraphBuilderTest, BuildGradientGraphTest) {
  TrainingRunner::Parameters params;
  params.model_path = ORIGINAL_MODEL_PATH;
  params.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});
  params.training_optimizer_name = "SGDOptimizer";

  const std::string& backprop_model_file = BuildBackPropGraph(params);

  std::shared_ptr<Model> pModel;
  EXPECT_TRUE(Model::Load(backprop_model_file, pModel).IsOK());

  Graph& graph = pModel->MainGraph();
  EXPECT_FALSE(graph.GraphResolveNeeded());
  EXPECT_TRUE(graph.NumberOfNodes() > 0);
  EXPECT_TRUE(graph.MaxNodeIndex() > 0);

  std::cout << "Graph input names = [" << std::endl;
  for (const NodeArg* p_node_arg : graph.GetInputs()) {
    std::cout << TAB << p_node_arg->Name() << std::endl;
  }
  std::cout << "]" << std::endl;

  std::cout << "Graph output names = [" << std::endl;
  for (const NodeArg* p_node_arg : graph.GetOutputs()) {
    std::cout << TAB << p_node_arg->Name() << std::endl;
  }
  std::cout << "]" << std::endl;

  for (Node& node : graph.Nodes()) {
    const NodeIndex node_index = node.Index();
    const std::string& node_name = node.Name();
    const std::string& op_type = node.OpType();

    std::cout << "Operation node:"
              << " Index=" << node_index
              << (node.NodeType() == Node::Type::Fused ? "-(FUSED)" : "")
              << " OpType=" << op_type
              << " Name=" << node_name
              << std::endl;
  }
}

TEST(GradientGraphBuilderTest, TrainingSession_Basic) {
  TrainingRunner::Parameters params;
  params.model_path = ORIGINAL_MODEL_PATH;
  params.model_with_training_graph_path = BACKWARD_MODEL_PATH;

  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  const std::string& backprop_model_file = BuildBackPropGraph(params);

  SessionOptions so{};
  RunTrainingSessionWithChecks(so, backprop_model_file);
}

TEST(GradientGraphBuilderTest, TrainingSession_WithGist) {
  const std::string gist_model_file = GIST_MODEL_PATH;

  TrainingRunner::Parameters params;
  params.model_path = ORIGINAL_MODEL_PATH;
  params.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  params.model_gist_encode = gist_model_file;
  params.use_gist = true;

  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});
  params.training_optimizer_name = "SGDOptimizer";

  const std::string& backprop_model_file = BuildBackPropGraph(params);

  std::cout << "Loading gist model file = " << gist_model_file << std::endl;
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(onnxruntime::Model::Load(gist_model_file, p_model).IsOK());

  const Graph& graph = p_model->MainGraph();
  bool found_encoder = false;
  bool found_decoder = false;
  for (auto& node : graph.Nodes()) {
    const std::string& node_name = node.Name();
    std::cout << "Node name='" << node_name << "' op_type=" << node.OpType() << "\n";
    if (node_name.find(onnxruntime::GistEncodeDecode::GIST_ENCODER_NODE_NAME_BASE) != std::string::npos) {
      found_encoder = true;
      std::cout << "Found encoder node " << node_name << "\n";
    } else if (node_name.find(onnxruntime::GistEncodeDecode::GIST_DECODER_NODE_NAME_BASE) != std::string::npos) {
      found_decoder = true;
      std::cout << "Found decoder node " << node_name << "\n";
    }
  }
  ASSERT_TRUE(found_encoder);
  ASSERT_TRUE(found_decoder);

  SessionOptions so{};
  RunTrainingSessionWithChecks(so, backprop_model_file);
}

TEST(GradientGraphBuilderTest, TrainingSession_WithLogging) {
  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  log_manager.SetDefaultLoggerSeverity(Severity::kINFO);

  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kERROR, DataType::USER)) << "ERROR level logging enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kWARNING, DataType::USER)) << "WARNING level logging enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kINFO, DataType::USER)) << "INFO level logging enabled.";

  TrainingRunner::Parameters params;
  params.model_path = ORIGINAL_MODEL_PATH;
  params.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});
  params.training_optimizer_name = "SGDOptimizer";
  const std::string& backprop_model_file = BuildBackPropGraph(params);

  SessionOptions so;
  so.session_logid = "training_session_with_logging";
  so.session_log_verbosity_level = 1;  // 1 == detailed logging

  std::unique_ptr<TrainingSession> training_session = RunTrainingSessionWithChecks(so, backprop_model_file);

  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kERROR, DataType::USER)) << "ERROR level logging still enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kWARNING, DataType::USER)) << "WARNING level logging still enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kINFO, DataType::USER)) << "INFO level logging still enabled.";

  std::string profile_file = training_session->EndProfiling();

  log_manager.SetDefaultLoggerSeverity(Severity::kWARNING);

  EXPECT_EQ(profile_file, std::string()) << "There should be no profile output file.";
}

TEST(GradientGraphBuilderTest, TrainingSession_WithProfiler) {
  TrainingRunner::Parameters params;
  params.model_path = ORIGINAL_MODEL_PATH;
  params.model_with_training_graph_path = BACKWARD_MODEL_PATH;

  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});
  params.training_optimizer_name = "SGDOptimizer";

  const std::string& backprop_model_file = BuildBackPropGraph(params);

  SessionOptions so;
  so.enable_profiling = true;
  so.profile_file_prefix = ORT_TSTR("onnx_training_profiler_test");

  std::unique_ptr<TrainingSession> training_session = RunTrainingSessionWithChecks(so, backprop_model_file);

  std::string profile_file = training_session->EndProfiling();

  std::cout << "Profile output file = " << profile_file << std::endl;

  std::ifstream profile(profile_file);
  ASSERT_TRUE(profile);

  std::vector<std::string> core_trace_fields = {"pid", "dur", "ts", "ph", "X", "name", "args"};
  std::vector<std::string> fiddle_profile_data_fields = {"dur", "activation_size", "parameter_size", "output_size"};

  int count = 0;
  std::string line;
  while (std::getline(profile, line)) {
    if (count == 0) {
      ASSERT_TRUE(line.find('[') != std::string::npos)
          << "Missing opening array marker in first trace record: " << line;
      // Opening array marker found.
    } else if (line.find(']') != std::string::npos) {
      // Closing array marker found.
      break;
    } else if (count >= 1) {
      if (count == 1) {
        auto s = "model_loading_uri";
        ASSERT_TRUE(line.find(s) != std::string::npos)
            << "Missing field '" << s << "' in trace record: " << line;
      }

      // Check we have the core fields in each trace record.
      for (auto& s : core_trace_fields) {
        ASSERT_TRUE(line.find(s) != std::string::npos)
            << "Missing core trace field '" << s << "' in trace record: " << line;
      }

      // Check we have the data profile fields output for each kernel operation.
      if (line.find("_kernel_time") != std::string::npos) {
        for (auto& s : fiddle_profile_data_fields) {
          ASSERT_TRUE(line.find(s) != std::string::npos)
              << "Missing data profile field '" << s << "' in trace record: " << line;
        }
      }
    }

    count++;
  }
  ASSERT_TRUE(count > 1);
}

#ifdef USE_CUDA
static void RunBertTrainingWithChecks(
    const SessionOptions& so,
    const std::string& backprop_model_file) {
  std::unique_ptr<Environment> env;
  EXPECT_TRUE(Environment::Create(env).IsOK());

  const auto& log_manager = so.session_log_verbosity_level > 0 ? &DefaultLoggingManager() : nullptr;

  std::unique_ptr<TrainingSession> training_session = std::make_unique<TrainingSession>(so, log_manager);

  EXPECT_TRUE(training_session->Load(backprop_model_file).IsOK());

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  EXPECT_TRUE(res.first.IsOK());
  ASSERT_TRUE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << std::endl;

  CUDAExecutionProviderInfo xp_info;
  ASSERT_TRUE(training_session->RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info)).IsOK());

  ASSERT_TRUE(training_session->Initialize().IsOK());

  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;

  // Creating feeds
  int batch_size = 13;
  int max_seq_len_in_batch = 7;
  std::vector<std::string> feed_names = {
      "input_ids",
      "token_type_ids",
      "input_mask",
      "masked_lm_ids",
      "next_sentence_labels",
      "masked_lm_positions",
      "masked_lm_weights",
  };
  std::vector<TensorShape> tensor_shapes = {
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch}};

  std::vector<std::vector<int64_t>> tensor_values = {
      /*input_ids*/
      {49, 97, 53, 5, 33, 65, 62,
       51, 38, 61, 45, 74, 27, 64,
       17, 36, 17, 96, 12, 79, 32,
       68, 90, 77, 18, 39, 12, 93,
       9, 87, 42, 60, 71, 12, 45,
       55, 40, 78, 81, 26, 70, 61,
       56, 66, 33, 7, 70, 1, 11,
       92, 51, 90, 85, 80, 0, 78,
       63, 42, 31, 93, 41, 90, 8,
       24, 72, 28, 30, 18, 69, 57,
       11, 10, 40, 65, 62, 13, 38,
       70, 37, 90, 15, 70, 42, 69,
       26, 77, 70, 75, 36, 56, 11},
      /*token_type_ids*/
      {12, 13, 1, 8, 15, 12, 9,
       15, 11, 6, 4, 9, 4, 3,
       8, 4, 9, 3, 2, 10, 15,
       3, 11, 13, 10, 6, 15, 14,
       8, 1, 0, 2, 12, 0, 15,
       10, 7, 10, 2, 6, 7, 7,
       4, 14, 2, 2, 10, 15, 3,
       9, 9, 3, 10, 6, 9, 14,
       2, 12, 10, 7, 9, 5, 6,
       5, 1, 8, 15, 2, 2, 4,
       4, 1, 2, 12, 8, 7, 6,
       13, 8, 14, 15, 11, 2, 10,
       3, 15, 10, 6, 7, 0, 8},
      /*input_mask*/
      {1, 1, 0, 1, 1, 1, 1,
       1, 1, 0, 0, 1, 0, 0,
       1, 0, 1, 0, 0, 1, 1,
       0, 1, 1, 1, 0, 1, 1,
       1, 0, 0, 0, 1, 0, 1,
       1, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 1, 0,
       1, 1, 0, 1, 0, 1, 1,
       0, 1, 1, 0, 1, 0, 0,
       0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 0, 1,
       0, 1, 1, 0, 0, 0, 1},
      /*masked_lm_ids*/
      {1, 1, 0, 1, 2, 1, 1,
       1, 1, 1, 2, 0, 2, 0,
       1, 0, 0, 2, 1, 2, 2,
       2, 0, 1, 0, 2, 0, 2,
       1, 1, 2, 0, 1, 1, 1,
       2, 2, 0, 2, 1, 1, 2,
       1, 0, 2, 0, 0, 2, 1,
       2, 2, 2, 0, 2, 1, 1,
       0, 2, 1, 2, 0, 0, 2,
       0, 0, 0, 2, 1, 0, 0,
       1, 2, 1, 0, 1, 2, 1,
       2, 0, 2, 1, 2, 0, 2,
       2, 2, 1, 1, 0, 2, 1},
      /*next_sentence_labels*/
      {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0},
      /*masked_lm_positions*/
      {0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6}};
  std::vector<float> masked_lm_weights(13 * 7, 1.0f);

  std::vector<OrtValue> feeds(feed_names.size());
  for (size_t i = 0; i < 6; ++i) {
    CreateMLValue(GetAllocator(), tensor_shapes[i].GetDims(), tensor_values[i], &feeds[i]);
  }
  CreateMLValue(GetAllocator(), tensor_shapes[6].GetDims(), masked_lm_weights, &feeds[6]);

  auto output_names_include_gradients = training_session->GetModelOutputNames();
  std::vector<std::string> fetch_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  std::vector<OrtValue> fetches;

  EXPECT_TRUE(training_session->Run(run_options, feed_names, feeds, fetch_names, &fetches).IsOK());

  for (size_t i = 0; i < fetch_names.size(); ++i) {
    if (!fetches[i].IsAllocated() || !!fetches[i].IsTensor())
      continue;

    const Tensor& tensor = fetches[i].Get<Tensor>();
    if (DataTypeImpl::GetType<float>() != tensor.DataType()) {
      continue;
    }

    const std::string& name = fetch_names[i];
    if (BERT_TOY_FETCHES.find(name) == BERT_TOY_FETCHES.end()) {
      continue;
    }

    auto gradient_ref = BERT_TOY_FETCHES.at(name);
    EXPECT_TRUE(tensor.Shape().Size() == gradient_ref.size());

    float max_diff = 0;
    float max_percent_diff = 0;
    const float* data = tensor.template Data<float>();
    for (size_t idx = 0; idx < gradient_ref.size(); ++idx) {
      float diff = std::fabs(static_cast<float>(gradient_ref[idx]) - data[idx]);
      max_diff = std::fmax(max_diff, diff);
      max_percent_diff = std::fmax(max_percent_diff, diff / data[idx]);
    }
    EXPECT_TRUE(max_diff < 1e-5) << name << " is incorrect: max_diff is " << max_diff;
    if (max_diff > 1e-10) {
      EXPECT_TRUE(max_percent_diff < 0.01f) << name << " is incorrect: max_percent_diff is "
                                            << max_percent_diff;
    }
  }
}
#endif
TEST(GradientGraphBuilderTest, TrainingSession_BertToy) {
  TrainingRunner::Parameters params;
  params.model_path = "testdata/bert_toy_optimized.onnx";
  params.model_with_training_graph_path = "testdata/bert_toy_optimized_bw.onnx";
  params.loss_func_info = LossFunctionInfo(OpDef("BertLoss", kOnnxDomain),
                                           "total_loss",
                                           {/*prediction_masked_lm*/ "prediction_scores",
                                            /*prediction_next_sentence*/ "seq_relationship_score",
                                            /*masked_lm_positions*/ "masked_lm_positions",
                                            /*masked_lm_ids*/ "masked_lm_ids",
                                            /*masked_lm_weights*/ "masked_lm_weights",
                                            /*next_sentence_labels*/ "next_sentence_labels",
                                            /*mlm_loss*/ "mlm_loss",
                                            /*nsp_loss*/ "nsp_loss",
                                            /*batch_size*/ std::to_string(13),
                                            /*max_sequence_len*/ std::to_string(7),
                                            /*max_predictions_per_sequence*/ std::to_string(7),
                                            /*summary_loss*/ "summary"});
  params.weights_not_to_train = {
      "position_01",            // Slice's dat input
      "op_min_ends_expand_10",  //op_min_ends_expand_10
  };
  params.immutable_weights = {
      {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
      {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}}},
      {"Mul", {{1, 0.5f}, {1, -10000.0f}}},
      {"Sub", {{0, 1.0f}}}};

  params.training_optimizer_name = "AdamOptimizer";
  params.adam_opt_params.alpha = 0.9f;
  params.adam_opt_params.beta = 0.999f;
  params.adam_opt_params.lambda = 0;
  params.adam_opt_params.epsilon = 0.1f;

  BuildBackPropGraph(params);

#ifdef USE_CUDA
  SessionOptions so;
  RunBertTrainingWithChecks(so, params.model_with_training_graph_path);
#endif
}

}  // namespace test
}  // namespace onnxruntime
