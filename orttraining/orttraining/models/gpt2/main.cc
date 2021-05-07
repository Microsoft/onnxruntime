// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/util/math.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "core/framework/random_seed.h"
#include "core/framework/bfc_arena.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#endif
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/models/runner/constant.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"
#include "orttraining/models/runner/data_loader.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;

struct GPT2Parameters : public TrainingRunner::Parameters {
  int max_sequence_length = 1024;
};

struct OrtParameters {
  logging::Severity log_severity;
  int vlog_level;
};

Status ParseArguments(int argc, char* argv[], GPT2Parameters& params, OrtParameters& ort_params) {
  cxxopts::Options options("GPT-2 Training", "Main Program to train GPT-2");
  // clang-format off
  options
    .add_options()
      ("mode", "mode for running, can be one of [train|perf]", cxxopts::value<std::string>()->default_value("perf"))
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("data/1024/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("data/1024/books_wiki_en_corpus/test"))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))
      ("perf_output_dir", "The output directory where the trained perf metrics files will be written.",
        cxxopts::value<std::string>()->default_value(""))
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value(""))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("learning_rate", "The initial learning rate for the optimizer.", cxxopts::value<float>()->default_value("5e-5"))
      ("num_train_steps", "Total number of training steps to perform.", cxxopts::value<int>()->default_value("100"))
      ("warmup_ratio", "Fraction of training steps for learning rate warmup.", cxxopts::value<float>()->default_value("0"))
      ("warmup_mode", "Warmup mode, one of [None|Cosine|Constant|Linear|Poly], defaults None.",
       cxxopts::value<std::string>()->default_value("Linear"))
      ("display_loss_steps", "How often to dump loss into tensorboard", cxxopts::value<size_t>()->default_value("1"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a backward/update pass.",
        cxxopts::value<int>()->default_value("1"))
      ("seed", "Random seed.", cxxopts::value<int64_t>()->default_value("-1"))
      ("use_mixed_precision", "Whether to use a mix of fp32 and fp16 arithmetic on GPU.", cxxopts::value<bool>()->default_value("false"))
      ("use_bfloat16", "Whether to use BFloat16 arithmetic on GPU.", cxxopts::value<bool>()->default_value("false"))
      ("enable_adasum", "Whether to use Adasum for allreduction.", cxxopts::value<bool>()->default_value("false"))
      ("allreduce_in_fp16", "Whether to do AllReduce in fp16. If false, AllReduce will be done in fp32", cxxopts::value<bool>()->default_value("true"))
      ("loss_scale", "Loss scaling, positive power of 2 values can improve fp16 convergence. "
        "Set it 0 to uses dynamic scaling; Other none-zero value will used as static scale",
        cxxopts::value<float>()->default_value("0.0"))
      ("use_fp16_moments", "Whether to use fp16 version of moments.", cxxopts::value<bool>()->default_value("false"))
      ("use_fp16_initializer", "FP16 weights will be created. Otherwise, cast nodes will be inserted for converting weights from FP32 to FP16",
        cxxopts::value<bool>()->default_value("true"))
      ("use_gist", "Whether to use GIST encoding/decoding.")
      ("gist_op", "Opearator type(s) to which GIST is applied.", cxxopts::value<int>()->default_value("0"))
      ("gist_compr", "Compression type used for GIST", cxxopts::value<std::string>()->default_value("GistPack8"))
      ("max_seq_length",
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.", cxxopts::value<int>()->default_value("1024"))
      ("optimizer", "Adam or Lamb", cxxopts::value<std::string>()->default_value("Adam"))
      ("deepspeed_zero_stage", "Controls whether to partition state using the DeepSpeed ZeRO technique. "
       "Stages 0 (disabled) and 1 (optimizer state partitioning) are supported.",
       cxxopts::value<int>()->default_value("0"))
      ("alpha", "Adam/Lamb alpha parameter", cxxopts::value<float>()->default_value("0.9"))
      ("beta", "Adam/Lamb beta parameter", cxxopts::value<float>()->default_value("0.999"))
      ("lambda", "Adam/Lamb lambda parameter", cxxopts::value<float>()->default_value("0.01"))
      ("epsilon", "Adam/Lamb epsilon parameter", cxxopts::value<float>()->default_value("1e-8"))
      ("data_parallel_size", "Data parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("horizontal_parallel_size", "Horizontal model parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("enable_grad_norm_clip", "Specify whether to enable gradient clipping for optimizers.",
        cxxopts::value<bool>()->default_value("true"));
  options
    .add_options("ORT configuration")
      ("ort_log_severity", "ORT minimum logging severity (see onnxruntime::logging::Severity values)",
        cxxopts::value<int>()->default_value("2"/*logging::Severity::kWARNING*/))
      ("ort_vlog_level", "ORT maximum VLOG level (verbose debug logging)",
        cxxopts::value<int>()->default_value("-1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    std::string mode = flags["mode"].as<std::string>();
    if (mode == "perf" || mode == "train") {
      params.is_perf_test = mode == "perf";
    } else {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Incorrect command line for mode: it must be one of [perf|train]");
    }

    params.model_name = flags["model_name"].as<std::string>();
    float lr = flags["learning_rate"].as<float>();
    if (lr < 0.f || lr > 1.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "learning_rate is not in valid range [0.0, 1.0]");
    }
    params.lr_params.initial_lr = lr;

    float ratio = flags["warmup_ratio"].as<float>();
    if (ratio < 0.f || ratio > 1.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "warmup_ratio is not in valid range [0.0, 1.0]");
    }
    params.lr_params.warmup_ratio = ratio;

    params.enable_adasum = flags["enable_adasum"].as<bool>();

    params.num_train_steps = flags["num_train_steps"].as<int>();
    params.batch_size = flags["train_batch_size"].as<int>();
    params.gist_config.op_type = flags["gist_op"].as<int>();
    params.gist_config.compr_type = flags["gist_compr"].as<std::string>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.batch_size;
    }
    params.max_sequence_length = flags["max_seq_length"].as<int>();

    params.gradient_accumulation_steps = flags["gradient_accumulation_steps"].as<int>();
    if (params.gradient_accumulation_steps < 1) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid gradient_accumulation_steps parameter: should be >= 1");
    }

    params.display_loss_steps = flags["display_loss_steps"].as<size_t>();

    params.train_data_dir = ToPathString(flags["train_data_dir"].as<std::string>());
    params.test_data_dir = ToPathString(flags["test_data_dir"].as<std::string>());
    params.log_dir = ToPathString(flags["log_dir"].as<std::string>());
    params.output_dir = ToPathString(flags["output_dir"].as<std::string>());
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }
    params.perf_output_dir = ToPathString(flags["perf_output_dir"].as<std::string>());
    if (params.perf_output_dir.empty()) {
      printf("No perf output directory specified. Trained perf metrics will not be saved.\n");
    }

    params.use_mixed_precision = flags["use_mixed_precision"].as<bool>();
    params.use_bfloat16 = flags["use_bfloat16"].as<bool>();
    params.allreduce_in_mixed_precision_type = flags["allreduce_in_fp16"].as<bool>() && params.use_mixed_precision;
    if (params.use_mixed_precision) {
      printf("Mixed precision training is enabled.\n");
    }
    if (params.allreduce_in_mixed_precision_type) {
      printf("Performing AllReduce in mixed precision type \n");
    } else {
      printf("Performing AllReduce in fp32 \n");
    }

    const float loss_scale = flags["loss_scale"].as<float>();
    if (loss_scale < 0.0f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Loss scale should be >= 0.");
    }
    params.loss_scale = loss_scale;
    if (params.use_mixed_precision) {
      if (params.loss_scale == 0.0) {
        printf("Using Dynamic loss scale.\n");
      } else {
        printf("Mixed precision loss scale is: %f\n", params.loss_scale);
      }
    }

    params.use_mixed_precision_moments = flags["use_fp16_moments"].as<bool>();
    if (params.use_mixed_precision_moments) {
      printf("Using mixed precision version of moments.\n");
    }
    params.use_mixed_precision_initializer = flags["use_fp16_initializer"].as<bool>();
    if (params.use_mixed_precision && params.use_mixed_precision_initializer) {
      printf("Mixed precision initializer is enabled.\n");
    }

    std::string warmup_mode = flags["warmup_mode"].as<std::string>();
    if (warmup_mode == LRSchedule_NoWarmup ||
        warmup_mode == LRSchedule_Cosine ||
        warmup_mode == LRSchedule_Constant ||
        warmup_mode == LRSchedule_Linear ||
        warmup_mode == LRSchedule_Poly) {
      params.lr_params.warmup_mode = warmup_mode;
      printf("Using learning rate warmup mode: %s \n", warmup_mode.c_str());
    } else {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                    "Incorrect warmup_mode: it must be one of [None|Cosine|Constant|Linear|Poly]");
    }

    std::string optimizer_name = flags["optimizer"].as<std::string>();
    if (optimizer_name == "adam" || optimizer_name == "Adam") {
      params.training_optimizer_name = "AdamOptimizer";
    } else if (optimizer_name == "lamb" || optimizer_name == "Lamb") {
      params.training_optimizer_name = "LambOptimizer";
    } else {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Incorrect optimizer type: it must be one of [Adam|Lamb]");
    }

    params.deepspeed_zero = ZeROConfig(flags["deepspeed_zero_stage"].as<int>());
    params.enable_grad_norm_clip = flags["enable_grad_norm_clip"].as<bool>();
    params.use_gist = flags.count("use_gist") > 0;
    float alpha = flags["alpha"].as<float>();
    float beta = flags["beta"].as<float>();
    float lambda = flags["lambda"].as<float>();
    float epsilon = flags["epsilon"].as<float>();
    ORT_RETURN_IF_NOT(alpha >= 0.f && alpha <= 1.f, "alpha is not in valid range [0.0, 1.0]");
    ORT_RETURN_IF_NOT(beta >= 0.f && beta <= 1.f, "alpha is not in valid range [0.0, 1.0]");
    std::vector<std::string> no_decay{"bias", "gamma", "beta", "LayerNorm"};

    params.optimizer_attributes = [=](const std::string& weight) {
      // Set lambda attribute to zero if we don't want decay on this weight.
      bool zero_lambda = std::any_of(no_decay.begin(), no_decay.end(), [&](const std::string& name) {
        return weight.find(name) != std::string::npos;
      });

      return std::unordered_map<std::string, float>{
          {"alpha", alpha},
          {"beta", beta},
          {"lambda", zero_lambda ? 0.f : lambda},
          {"epsilon", epsilon},
      };
    };

    params.data_parallel_size = flags["data_parallel_size"].as<int>();
    params.horizontal_parallel_size = flags["horizontal_parallel_size"].as<int>();
    ORT_RETURN_IF_NOT(params.data_parallel_size > 0, "data_parallel_size must > 0");
    ORT_RETURN_IF_NOT(params.horizontal_parallel_size > 0, "horizontal_parallel_size must > 0");

    int64_t seed = flags["seed"].as<int64_t>();
    if (params.horizontal_parallel_size > 1 && seed <= 0) {
      seed = 8211; // Megatron needs a random seed.
    }
    if (seed > 0) {
      utils::SetRandomSeed(seed);
      std::cout << "Random seed is set to: " << seed << std::endl;
    }

    ort_params.log_severity = static_cast<logging::Severity>(flags["ort_log_severity"].as<int>());
    ORT_RETURN_IF_NOT(
        logging::Severity::kVERBOSE <= ort_params.log_severity &&
            ort_params.log_severity <= logging::Severity::kFATAL,
        "Log severity must be in the range [", static_cast<int>(logging::Severity::kVERBOSE),
        ", ", static_cast<int>(logging::Severity::kFATAL), "].");
    ort_params.vlog_level = flags["ort_vlog_level"].as<int>();
  } catch (const std::exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    std::cerr << msg << ": " << e.what() << "\n"
              << options.help() << "\n";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, msg);
  }
  return Status::OK();
}

// NOTE: these variables need to be alive when the error_function is called.
float mlm_loss = 0.0f;
std::vector<std::string> summary_loss;

float GetLossValue(const Tensor& loss_tensor) {
  float loss = 0;
  if (DataTypeImpl::GetType<float>() == loss_tensor.DataType()) {
    loss = *(loss_tensor.template Data<float>());
  } else if (DataTypeImpl::GetType<MLFloat16>() == loss_tensor.DataType()) {
    loss = math::halfToFloat(loss_tensor.template Data<MLFloat16>()->val);
  }
  return loss;
}

// mapping to define what to be stored in mapped_dimensions
// see GetTensorDimensionsFromInputs() in training_util.h and training_runner.cc for more details
const std::map<std::string, std::pair<std::string, size_t>> input_to_dimension_mapping = {
  {"input_ids", {"SeqLen", 0}},   // int64[batch,seqlen]    "seqlen" -> "SeqLen", 0
};

// generic properties for storing perf metrics
MapStringToString mapped_dimensions;

void setup_training_params(GPT2Parameters& params) {
  params.model_path = ToPathString(params.model_name) + ORT_TSTR(".onnx");
  params.model_with_loss_func_path = ToPathString(params.model_name) + ORT_TSTR("_with_cost.onnx");
  params.model_with_training_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw.onnx");
  params.model_actual_running_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw_running.onnx");
  params.model_with_gist_nodes_path = ToPathString(params.model_name) + ORT_TSTR("_with_gist.onnx");

  params.loss_func_info = LossFunctionInfo(OpDef("SparseSoftmaxCrossEntropy", kOnnxDomain),
                                           "mlm_loss",
                                           {/*prediction_name*/ "output",
                                            /*label_name*/ "labels"});

#if defined(USE_MPI)
  ORT_ENFORCE(params.horizontal_parallel_size <= MPIContext::GetInstance().GetWorldSize());
  ORT_ENFORCE(params.data_parallel_size <= MPIContext::GetInstance().GetWorldSize());
  if (MPIContext::GetInstance().GetWorldSize() % params.horizontal_parallel_size != 0) {
    LOGS_DEFAULT(ERROR) << "Cannot split horizontal parallel group because world_size is not divisible";
    return;
  }

  auto data_group_size = MPIContext::GetInstance().GetWorldSize() / params.horizontal_parallel_size;
  if (data_group_size != params.data_parallel_size) {
    LOGS_DEFAULT(WARNING) << "WARNING: data_parallel_size is not correct, tuned automatically to "
                          << data_group_size << std::endl;
    params.data_parallel_size = data_group_size;
  }

  params.enable_adasum = params.enable_adasum && (params.data_parallel_size > 1);
  if (params.enable_adasum)
    std::cout << "Use Adsum for allreduce." << std::endl;
#endif

  params.fetch_names = {"mlm_loss"};

  if (params.EnableTensorboard()) {
    params.fetch_names.push_back(params.summary_name);
    params.scalar_names = {"mlm_loss", params.lr_params.feed_name};
  }

  params.weights_not_to_train = {};
  params.immutable_weights = {
      {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
      {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}, {1, 1e-5f}}},
      {"Mul", {{1, 0.5f}, {1, 0.044715f}, {1, 0.7978845834732056f}, {1, -10000.f}, {1, 10000.f}}},
      {"Pow", {{1, 2.0f}}},
      {"Sub", {{0, 1.0f}}}};

  params.shuffle_data = false;

  // name_in_data_file -> name_in_model
  params.input_name_map = {
      {"input_ids", "input_ids"},
      {"position_ids", "position_ids"},
      {"attention_mask", "attention_mask"},
      {"labels", "labels"}};

  params.model_type = "gpt2";

#ifdef USE_CUDA
  {
    CUDAExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());

    params.providers.emplace(kCudaExecutionProvider, CreateExecutionProviderFactory_CUDA(info));
    params.input_allocator = std::make_shared<CUDAPinnedAllocator>(info.device_id, GPU_PINNED);
  }
#endif

  params.use_nccl = true;

  params.error_function = [params](const std::vector<std::string>& /*feed_names*/,
                                   const std::vector<OrtValue>& /*feeds*/,
                                   const std::vector<std::string>& /*fetch_names*/,
                                   const std::vector<OrtValue>& fetches,
                                   size_t /*step*/) {
    const Tensor& mlm_loss_t = fetches[0].Get<Tensor>();

    const auto curr_mlm_loss = GetLossValue(mlm_loss_t);

    mlm_loss += curr_mlm_loss;

    if (params.EnableTensorboard()) {
      const Tensor& summary_loss_t = fetches[1].Get<Tensor>();
      summary_loss.push_back(*(summary_loss_t.template Data<std::string>()));
    }
  };

  std::shared_ptr<EventWriter> tensorboard;
  if (params.EnableTensorboard())
    tensorboard = std::make_shared<EventWriter>(params.log_dir);

  params.post_evaluation_callback = [tensorboard](size_t num_samples, size_t step, const std::string tag) {
    if (tensorboard != nullptr) {
      for (const std::string& summary : summary_loss) {
        tensorboard->AddSummary(summary, step, tag);
      }
    }

    printf("Step: %zu, #examples: %d, mlm_loss: %0.04f\n\n",
           step,
           static_cast<int>(num_samples),
           mlm_loss);
    mlm_loss = 0.0f;
    summary_loss.clear();
  };
}

static Status RunPerformanceTest(const GPT2Parameters& params, const Environment& env) {
  // setup fake data
  const int batch_size = static_cast<int>(params.batch_size);
  std::vector<std::string> tensor_names = {"input_ids",
                                           "position_ids",
                                           "attention_mask",
                                           "labels"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, params.max_sequence_length},
                                            {batch_size, params.max_sequence_length},
                                            {batch_size, 1, params.max_sequence_length, params.max_sequence_length},
                                            {batch_size, params.max_sequence_length}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_INT64};
  const size_t num_of_perf_samples = params.num_train_steps * params.batch_size;
  auto random_perf_data = std::make_shared<RandomDataSet>(num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);
  auto random_perf_data_loader = std::make_unique<SingleDataLoader>(random_perf_data, tensor_names);

  TrainingRunner runner(params, env);
  ORT_RETURN_IF_ERROR(runner.Initialize());
  ORT_RETURN_IF_ERROR(runner.Run(random_perf_data_loader.get(), random_perf_data_loader.get()));

  return Status::OK();
}

static Status RunTraining(const GPT2Parameters& params, const Environment& env) {
  const size_t max_num_files_preload = 2;

  auto runner = std::make_unique<TrainingRunner>(params, env);
  ORT_RETURN_IF_ERROR(runner->Initialize());

  auto rank_in_data_parallel_group = MPIContext::GetInstance().GetWorldRank() / params.horizontal_parallel_size;
  auto training_data_loader = std::make_unique<DataLoader>(params.input_name_map,
                                                                   params.train_data_dir,
                                                                   max_num_files_preload,
                                                                   rank_in_data_parallel_group,
                                                                   params.data_parallel_size);

  std::unique_ptr<DataLoader> test_data_loader;
  // Evaluation is only done in device #0
  if (MPIContext::GetInstance().GetWorldRank() == 0) {
    test_data_loader = std::make_unique<DataLoader>(params.input_name_map,
                                                            params.test_data_dir,
                                                            max_num_files_preload);
  }

  if (!params.perf_output_dir.empty()) {
    // collecting GPT2 related params from training data
    auto training_data = training_data_loader->CurrentDataSet();
    ORT_RETURN_IF_ERROR(training_data->GetTensorDimensionsFromInputs(input_to_dimension_mapping, mapped_dimensions));
  }

  ORT_RETURN_IF_ERROR(runner->Run(training_data_loader.get(), test_data_loader.get(), mapped_dimensions));

  // only test and save trained model on device #0
  if (MPIContext::GetInstance().GetWorldRank() == 0) {
    test_data_loader = std::make_unique<DataLoader>(params.input_name_map,
                                                            params.test_data_dir,
                                                            max_num_files_preload);

    ORT_RETURN_IF_ERROR(runner->EndTraining(test_data_loader.get()));
  }

  return Status::OK();
}

int main(int argc, char* argv[]) {
  GPT2Parameters params;
  OrtParameters ort_params{logging::Severity::kWARNING, -1};
  RETURN_IF_FAIL(ParseArguments(argc, argv, params, ort_params));

  // setup logger
  std::string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  ort_params.log_severity,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id,
                                                  ort_params.vlog_level};
  setup_training_params(params);

  // setup onnxruntime env
  std::unique_ptr<Environment> env;
  RETURN_IF_FAIL(Environment::Create(nullptr, env));

  // start training session
  if (params.is_perf_test) {
    RETURN_IF_FAIL(RunPerformanceTest(params, *env));
  } else {
    RETURN_IF_FAIL(RunTraining(params, *env));
  }

#if defined(USE_MPI)
#ifdef _WIN32
  // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
  // shutdown_mpi() is not called within MPIContext destructor because of DllMain's restriction
  // call shutdown_mpi() here instead.
  MPIContext::shutdown_mpi();
#endif
#endif
  return 0;
}
