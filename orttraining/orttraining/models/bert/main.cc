// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/util/math.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/profiler.h"
#include "core/session/environment.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/random_seed.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#endif
#ifdef USE_ROCM
#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_provider_factory_creator.h"
#endif
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/models/runner/constant.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"
#include "orttraining/models/runner/data_loader.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

static SessionOptions session_options = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
    ExecutionOrder::PRIORITY_BASED,    //execution_order
    false,                             //enable_profiling
    ORT_TSTR(""),                      //optimized_model_filepath
    true,                              //enable_mem_pattern
    true,                              //enable_mem_reuse
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level1,          //graph_optimization_level
    {},                                //intra_op_param
    {},                                //inter_op_param
    {},                                //free_dimension_overrides
    true,                              //use_per_session_threads
    true,                              //thread_pool_allow_spinning
    false,                             //use_deterministic_compute
    {},                                //session_configurations
    {},                                // initializers_to_share_map
}; 

struct BertParameters : public TrainingRunner::Parameters {
  int max_sequence_length = 512;
  int max_predictions_per_sequence = 80;
  size_t batch_size_phase2;
  int gradient_accumulation_steps_phase2 = 1;
  float initial_lr_phase2;
  size_t num_train_steps_phase2;
  float warmup_ratio_phase2;
  float gpu_mem_limit_in_gb = -1;
  bool debug_break = false;
  PathString train_data_dir_phase2;
  PathString test_data_dir_phase2;

  PathString convergence_test_output_file;
};

struct OrtParameters {
  logging::Severity log_severity{logging::Severity::kWARNING};
  int vlog_level{-1};
  size_t max_num_profiling_events{0};  // 0 means use the default value
};

Status ParseArguments(int argc, char* argv[], BertParameters& params, OrtParameters& ort_params) {
  cxxopts::Options options("BERT Training", "Main Program to train BERT");
  // clang-format off
  options
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/test"))
      ("train_data_dir_phase2", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value(""))
      ("test_data_dir_phase2", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value(""))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))
      ("perf_output_dir", "The output directory where the trained perf metrics files will be written.",
        cxxopts::value<std::string>()->default_value(""))
      ("checkpoints_dir", "The output directory where the checkpoint files will be written.",
        cxxopts::value<std::string>()->default_value(""))
      ("checkpoint_to_load_path",
       "The path to the checkpoint to load. If not provided, the latest "
       "checkpoint in checkpoints_dir, if any, is used.",
        cxxopts::value<std::string>()->default_value(""))
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value(""))
      ("convergence_test_output_file", "The convergence test output file path.",
        cxxopts::value<std::string>()->default_value(""))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("train_batch_size_phase2", "Total batch size for training.", cxxopts::value<int>()->default_value("1"))
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("learning_rate", "The initial learning rate for the optimizer.", cxxopts::value<float>()->default_value("5e-5"))
      ("learning_rate_phase2", "The initial learning rate for the optimizer.", cxxopts::value<float>()->default_value("4e-3"))
      ("num_train_steps", "Total number of training steps to perform.", cxxopts::value<int>()->default_value("100000"))
      ("num_train_steps_phase2", "Total number of training steps to perform.", cxxopts::value<int>()->default_value("1563"))
      ("warmup_ratio", "Fraction of training steps for learning rate warmup.", cxxopts::value<float>()->default_value("0"))
      ("warmup_ratio_phase2", "Fraction of training steps for learning rate warmup.", cxxopts::value<float>()->default_value("0.128"))
      ("warmup_mode", "Warmup mode, one of [None|Cosine|Constant|Linear|Poly], defaults None.",
       cxxopts::value<std::string>()->default_value("None"))
      ("do_eval", "Whether to run eval on the dev set.", cxxopts::value<bool>()->default_value("false"))
      ("evaluation_period",
        "How many training steps to make before making an evaluation.",
        cxxopts::value<size_t>()->default_value("100"))
      ("display_loss_steps", "How often to dump loss into tensorboard", cxxopts::value<size_t>()->default_value("10"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a backward/update pass.",
        cxxopts::value<int>()->default_value("1"))
      ("checkpoint_period", "How many weight-update steps to run before saving a model checkpoint.", cxxopts::value<size_t>()->default_value("1000"))
      ("max_num_checkpoints", "Maximum number of checkpoint files to maintain.",
        cxxopts::value<size_t>()->default_value("10"))
      ("gradient_accumulation_steps_phase2", "The number of gradient accumulation steps before performing a backward/update pass in phase 2.",
        cxxopts::value<int>()->default_value("1"))
      ("iterations_per_loop", "How many steps to make in each estimator call.", cxxopts::value<int>()->default_value("1000"))
      ("max_eval_steps", "Maximum number of eval steps.", cxxopts::value<int>()->default_value("100"))
      ("seed", "Random seed.", cxxopts::value<int64_t>()->default_value("-1"))
      ("use_deterministic_compute", "Whether to enable deterministic compute.", cxxopts::value<bool>()->default_value("false"))
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
      ("use_nccl", "Whether to use NCCL for distributed training.", cxxopts::value<bool>()->default_value("false"))
      ("use_profiler", "Collect runtime profile data during this training run.", cxxopts::value<bool>()->default_value("false"))
      ("use_gist", "Whether to use GIST encoding/decoding.")
      ("gist_op", "Opearator type(s) to which GIST is applied.", cxxopts::value<int>()->default_value("0"))
      ("gist_compr", "Compression type used for GIST", cxxopts::value<std::string>()->default_value("GistPack8"))
      ("max_profile_records", "Maximum number of runtime profile data records to collect. 0 means use the default value.",
        cxxopts::value<size_t>()->default_value("0"))
      ("mode", "mode for running, can be one of [train|perf]", cxxopts::value<std::string>()->default_value("train"))
      ("histogram", "Tensor(s) to display a histogram on tensorboard (e.g. '417,3347,417_grad,3347_grad' for bert-large or '81,449,81_grad,449_grad' for bert-tiny)",
        cxxopts::value<std::vector<std::string>>()->default_value({}))
      ("norm", "Tensor(s) to display their L2-norm values on tensorboard (e.g. '417,3347,417_grad,3347_grad' for bert-large or '81,449,81_grad,449_grad' for bert-tiny)",
        cxxopts::value<std::vector<std::string>>()->default_value({}))
      ("dump_convergence_metrics", "specify if tensorboard should include convergence metrics such as gradient norm.",
        cxxopts::value<bool>()->default_value("false"))
      ("max_seq_length",
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.", cxxopts::value<int>()->default_value("512"))
      ("max_predictions_per_seq",
        "Maximum number of masked LM predictions per sequence. "
        "Must match data generation.", cxxopts::value<int>()->default_value("80"))
      ("optimizer", "Adam or Lamb", cxxopts::value<std::string>()->default_value("Adam"))
      ("deepspeed_zero_stage", "Controls whether to partition state using the DeepSpeed ZeRO technique. "
       "Stages 0 (disabled) and 1 (optimizer state partitioning) are supported.",
       cxxopts::value<int>()->default_value("0"))
      ("alpha", "Adam/Lamb alpha parameter", cxxopts::value<float>()->default_value("0.9"))
      ("beta", "Adam/Lamb beta parameter", cxxopts::value<float>()->default_value("0.999"))
      ("lambda", "Adam/Lamb lambda parameter", cxxopts::value<float>()->default_value("0.01"))
      ("epsilon", "Adam/Lamb epsilon parameter", cxxopts::value<float>()->default_value("1e-6"))
      ("do_bias_correction",
        "A flag controls if Adam/Lamb should do bias correction. "
        "Default is false, which means no bias correction. "
        "Use true to enable bias correction.",
        cxxopts::value<bool>()->default_value("false"))
      ("weight_decay_mode",
        "Chooses the weight decay mode for Adam optimizer "
        "Default is 0, which does weight decay before updating weight. "
        "Use 1 to do weight decay after updating weight.",
        cxxopts::value<int64_t>()->default_value("0"))
      ("ratio_min", "Lamb min ratio parameter", cxxopts::value<float>()->default_value("0.05"))
      ("ratio_max", "Lamb max ratio parameter", cxxopts::value<float>()->default_value("5.0"))
      ("gpu_mem_limit_in_gb", "Max cuda memory ort can use, in GB", cxxopts::value<float>()->default_value("-1.0"))
      ("data_parallel_size", "Data parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("horizontal_parallel_size", "Horizontal model parallel group size.", cxxopts::value<int>()->default_value("1"))
      ("pipeline_parallel_size", "Number of pipeline stages.", cxxopts::value<int>()->default_value("1"))
      ("pipeline_stage_paths", "Specify the forward ONNX files for pipeline evaluation.", cxxopts::value<std::vector<std::string>>()->default_value(""))
      ("cut_group_info", "Specify the cutting info for graph partition (pipeline only). An example of a cut_group_info of "
      "size two is: 1393:407-1463/1585/1707,2369:407-2439/2561/2683. Here, the cut info is split by ',', with the first "
      "cut_info equal to 1393:407-1463/1585/1707, and second cut_info equal to 2369:407-2439/2561/2683. Each CutEdge is "
      "seperated by ':'. If consumer nodes need to be specified, specify them after producer node with a '-' delimiter and "
      "separate each consumer node with a '/'. ", cxxopts::value<std::vector<std::string>>()->default_value(""))
      ("enable_grad_norm_clip", "Specify whether to enable gradient clipping for optimizers.",
        cxxopts::value<bool>()->default_value("true"))
      ("enable_gelu_approximation", "Specify whether to enable GELU approximation.",
        cxxopts::value<bool>()->default_value("true"))
      ("attn_dropout_recompute", "Enable checkpointing of attention dropout to save memory.",
        cxxopts::value<bool>()->default_value("false"))
      ("gelu_recompute", "Enable checkpointing of Gelu activation output to save memory.",
        cxxopts::value<bool>()->default_value("false"))
      ("transformer_layer_recompute", "Enable checkpointing of transformer layer output to save memory.",
        cxxopts::value<bool>()->default_value("false"))
      ("number_recompute_layers", "Number of layers to apply recompute.",
        cxxopts::value<int>()->default_value("0"))
      ("use_invertible_layernorm_grad", "Specify whether to use invertible laynorm(dropping the input activation)",
        cxxopts::value<bool>()->default_value("false"))
      ("debug_break", "Specify whether to break at app start, useful for multi-gpu debugging.",
        cxxopts::value<bool>()->default_value("false"));
  options
    .add_options("ORT configuration")
      ("ort_log_severity", "ORT minimum logging severity (see onnxruntime::logging::Severity values)",
        cxxopts::value<int>()->default_value("2"/*logging::Severity::kWARNING*/))
      ("ort_vlog_level", "ORT maximum VLOG level (verbose debug logging)",
        cxxopts::value<int>()->default_value("-1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_name = flags["model_name"].as<std::string>();
    params.debug_break = flags["debug_break"].as<bool>();
    float lr = flags["learning_rate"].as<float>();
    if (lr > 1.f || lr < 0.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "learning_rate is not in valid range [0.0, 1.0]");
    }
    params.lr_params.initial_lr = lr;

    float lr_phase2 = flags["learning_rate_phase2"].as<float>();
    if (lr_phase2 > 1.f || lr_phase2 < 0.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "learning_rate_phase2 is not in valid range [0.0, 1.0]");
    }
    params.initial_lr_phase2 = lr_phase2;

    float ratio = flags["warmup_ratio"].as<float>();
    if (ratio > 1.f || ratio < 0.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "warmup_ratio is not in valid range [0.0, 1.0]");
    }
    params.lr_params.warmup_ratio = ratio;

    params.gpu_mem_limit_in_gb = flags["gpu_mem_limit_in_gb"].as<float>();

    float ratio_phase2 = flags["warmup_ratio_phase2"].as<float>();
    if (ratio_phase2 > 1.f || ratio_phase2 < 0.f) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "warmup_ratio_phase2 is not in valid range [0.0, 1.0]");
    }
    params.warmup_ratio_phase2 = ratio_phase2;

    params.num_train_steps = flags["num_train_steps"].as<int>();
    params.num_train_steps_phase2 = flags["num_train_steps_phase2"].as<int>();

    params.batch_size = flags["train_batch_size"].as<int>();
    params.gist_config.op_type = flags["gist_op"].as<int>();
    params.gist_config.compr_type = flags["gist_compr"].as<std::string>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.batch_size;
    }

    params.batch_size_phase2 = flags["train_batch_size_phase2"].as<int>();

    params.max_sequence_length = flags["max_seq_length"].as<int>();
    params.max_predictions_per_sequence = flags["max_predictions_per_seq"].as<int>();

    params.gradient_accumulation_steps = flags["gradient_accumulation_steps"].as<int>();
    if (params.gradient_accumulation_steps < 1) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid gradient_accumulation_steps parameter: should be >= 1");
    }

    params.gradient_accumulation_steps_phase2 = flags["gradient_accumulation_steps_phase2"].as<int>();
    if (params.gradient_accumulation_steps_phase2 < 1) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid gradient_accumulation_steps_phase2 parameter: should be >= 1");
    }

    params.do_eval = flags["do_eval"].as<bool>();
    params.evaluation_period = flags["evaluation_period"].as<size_t>();
    params.display_loss_steps = flags["display_loss_steps"].as<size_t>();
    params.checkpoint_period = flags["checkpoint_period"].as<size_t>();
    params.max_num_checkpoints = flags["max_num_checkpoints"].as<size_t>();

    params.use_nccl = flags["use_nccl"].as<bool>();
    params.enable_adasum = flags["enable_adasum"].as<bool>();
    params.use_profiler = flags.count("use_profiler") > 0;
    ort_params.max_num_profiling_events = flags["max_profile_records"].as<size_t>();

    params.train_data_dir = ToPathString(flags["train_data_dir"].as<std::string>());
    params.test_data_dir = ToPathString(flags["test_data_dir"].as<std::string>());
    params.log_dir = ToPathString(flags["log_dir"].as<std::string>());
    params.train_data_dir_phase2 = ToPathString(flags["train_data_dir_phase2"].as<std::string>());
    params.test_data_dir_phase2 = ToPathString(flags["test_data_dir_phase2"].as<std::string>());
    params.convergence_test_output_file = ToPathString(flags["convergence_test_output_file"].as<std::string>());
    params.output_dir = ToPathString(flags["output_dir"].as<std::string>());
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }
    params.perf_output_dir = ToPathString(flags["perf_output_dir"].as<std::string>());
    if (params.perf_output_dir.empty()) {
      printf("No perf output directory specified. Trained perf metrics will not be saved.\n");
    }
    params.checkpoints_dir = ToPathString(flags["checkpoints_dir"].as<std::string>());
    if (params.checkpoints_dir.empty()) {
      printf("No checkpoints directory specified. Checkpoint files will not be saved.\n");
    }
    params.checkpoint_to_load_path = ToPathString(flags["checkpoint_to_load_path"].as<std::string>());

    params.histogram_names = flags["histogram"].as<std::vector<std::string>>();
    params.norm_names = flags["norm"].as<std::vector<std::string>>();
    params.dump_convergence_metrics = flags["dump_convergence_metrics"].as<bool>();

    std::string mode = flags["mode"].as<std::string>();
    if (mode == "perf" || mode == "train") {
      params.is_perf_test = mode == "perf";
    } else {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Incorrect command line for mode: it must be one of [perf|train]");
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
    {
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
    int64_t weight_decay_mode = flags["weight_decay_mode"].as<int64_t>();
    float ratio_min = flags["ratio_min"].as<float>();
    float ratio_max = flags["ratio_max"].as<float>();
    ORT_RETURN_IF_NOT(alpha >= 0.f && alpha <= 1.f, "alpha is not in valid range [0.0, 1.0]");
    ORT_RETURN_IF_NOT(beta >= 0.f && beta <= 1.f, "alpha is not in valid range [0.0, 1.0]");
    ORT_RETURN_IF_NOT(weight_decay_mode == 0 || weight_decay_mode == 1, "Only 0 and 1 are supported for weight decay mode.");
    ORT_RETURN_IF_NOT(epsilon >= 0.f, "epsilon should be non-negative.");
    ORT_RETURN_IF_NOT(epsilon >= 0.f, "epsilon should be non-negative.");
    ORT_RETURN_IF_NOT(ratio_min >= 0.f, "ratio_min should be non-negative.");
    ORT_RETURN_IF_NOT(ratio_max >= 0.f, "ratio_max should be non-negative.");
    ORT_RETURN_IF_NOT(ratio_max >= ratio_min, "ratio_max should be greater than or equal to ratio_min.");
    std::vector<std::string> no_decay{"bias", "gamma", "beta", "LayerNorm"};

    bool do_bias_correction = flags["do_bias_correction"].as<bool>();

    // Optimizer's float attributes.
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
          {"ratio_min", ratio_min},
          {"ratio_max", ratio_max}};
    };

    // Optimizer's int attributes.
    params.optimizer_int_attributes = [=](const std::string& /*weight*/) {
      return std::unordered_map<std::string, int64_t>{
          {"do_bias_correction", do_bias_correction ? static_cast<int64_t>(1) : static_cast<int64_t>(0)},
          {"weight_decay_mode", weight_decay_mode}};
    };

    params.data_parallel_size = flags["data_parallel_size"].as<int>();
    params.horizontal_parallel_size = flags["horizontal_parallel_size"].as<int>();
    ORT_RETURN_IF_NOT(params.data_parallel_size > 0, "data_parallel_size must > 0");
    ORT_RETURN_IF_NOT(params.horizontal_parallel_size > 0, "horizontal_parallel_size must > 0");

    // pipeline_parallel_size controls the number of pipeline's stages.
    // pipeline_parallel_size=1 means no model partition, which means all processes run
    // the same model. We only partition model when pipeline_parallel_size > 1.
    params.pipeline_parallel_size = flags["pipeline_parallel_size"].as<int>();
    ORT_RETURN_IF_NOT(params.pipeline_parallel_size > 0, "pipeline_parallel_size must > 0");

    // If user provides partitioned model files, the number of files should match the number of
    // processes. The i-th file should correspond to the i-th process' pipeline stage.
    // All files only store forward pass with a Recv and a Send.
    // Backward pass and optimizer nodes are implicitly generated by ORT.
    params.pipeline_stage_paths = flags["pipeline_stage_paths"].as<std::vector<std::string>>();

    // If user doesn't provide partitioned model files, a cut list should be provided for ORT to do partition
    // online. If the pipeline contains n stages, the cut list should be of length (n-1), in order to cut the
    // graph into n partitions.
    if (params.pipeline_parallel_size > 1 && params.pipeline_stage_paths.empty()) {
      auto cut_info_groups = flags["cut_group_info"].as<std::vector<std::string>>();

      ORT_RETURN_IF_NOT(static_cast<int>(cut_info_groups.size() + 1) == params.pipeline_parallel_size,
                        "cut_info length plus one must match pipeline parallel size");

      auto process_with_delimiter = [](std::string& input_str, const std::string& delimiter) {
        std::vector<std::string> result;
        size_t pos = 0;
        std::string token;
        while ((pos = input_str.find(delimiter)) != std::string::npos) {
          token = input_str.substr(0, pos);
          result.emplace_back(token);
          input_str.erase(0, pos + delimiter.length());
        }
        // push the last split of substring into result.
        result.emplace_back(input_str);
        return result;
      };

      auto process_cut_info = [&](std::string& cut_info_string) {
        TrainingSession::TrainingConfiguration::CutInfo cut_info;
        const std::string edge_delimiter = ":";
        const std::string consumer_delimiter = "/";
        const std::string producer_consumer_delimiter = "-";

        auto cut_edges = process_with_delimiter(cut_info_string, edge_delimiter);
        for (auto& cut_edge : cut_edges) {
          auto process_edge = process_with_delimiter(cut_edge, producer_consumer_delimiter);
          if (process_edge.size() == 1) {
            TrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0]};
            cut_info.emplace_back(edge);
          } else {
            ORT_ENFORCE(process_edge.size() == 2);
            auto consumer_list = process_with_delimiter(process_edge[1], consumer_delimiter);

            TrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0], consumer_list};
            cut_info.emplace_back(edge);
          }
        }
        return cut_info;
      };

      for (auto& cut_info : cut_info_groups) {
        TrainingSession::TrainingConfiguration::CutInfo cut = process_cut_info(cut_info);
        params.pipeline_partition_cut_list.emplace_back(cut);
      }
    }

    int64_t seed = flags["seed"].as<int64_t>();
    if (params.horizontal_parallel_size > 1 && seed <= 0) {
      seed = 8211;  // Megatron needs a random seed.
    }
    if (seed > 0) {
      utils::SetRandomSeed(seed);
      std::cout << "Random seed is set to: " << seed << std::endl;
    }

    session_options.use_deterministic_compute = flags["use_deterministic_compute"].as<bool>();

    params.enable_gelu_approximation = flags["enable_gelu_approximation"].as<bool>();
    params.attn_dropout_recompute = flags["attn_dropout_recompute"].as<bool>();
    params.gelu_recompute = flags["gelu_recompute"].as<bool>();
    params.transformer_layer_recompute = flags["transformer_layer_recompute"].as<bool>();
    params.number_recompute_layers = flags["number_recompute_layers"].as<int>();

    ort_params.log_severity = static_cast<logging::Severity>(flags["ort_log_severity"].as<int>());
    ORT_RETURN_IF_NOT(
        logging::Severity::kVERBOSE <= ort_params.log_severity &&
            ort_params.log_severity <= logging::Severity::kFATAL,
        "Log severity must be in the range [", static_cast<int>(logging::Severity::kVERBOSE),
        ", ", static_cast<int>(logging::Severity::kFATAL), "].");
    ort_params.vlog_level = flags["ort_vlog_level"].as<int>();

    params.use_invertible_layernorm_grad = flags["use_invertible_layernorm_grad"].as<bool>();
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, msg);
  }

  return Status::OK();
}

// specifies convergence test output file data
struct ConvergenceTestDataRecord {
  size_t step{};
  float total_loss{}, mlm_loss{}, nsp_loss{};

  static std::string GetCsvHeaderLine() { return "step,total_loss,mlm_loss,nsp_loss\n"; }

  std::string GetCsvLine() const {
    return onnxruntime::MakeString(step, ",", total_loss, ",", mlm_loss, ",", nsp_loss, "\n");
  }
};

// NOTE: these variables need to be alive when the error_function is called.
float total_loss = 0.0f;
float mlm_loss = 0.0f;
float nsp_loss = 0.0f;
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

// use this table mapping to define what to be stored in mapped_dimensions, and ultimately in json structure
// Be mindful on the position, if it's invalid or out of bound, the property population process will be
// either incorrect or aborted. Also make sure to substract the index position by 1 to get valid correspondent value
// namely, in the graph, sequence is at position 1, but in initial tensor shape vector loaded from training data is at position 0,
// batch is not part of the initial tensor shape vector till later
// see GetTensorDimensionsFromInputs() in training_util.h and training_runner.cc for more details
const std::map<std::string, std::pair<std::string, size_t>> input_to_dimension_mapping = {
    {"input1", {"SeqLen", 0}},                   // int64[batch,sequence]    "sequence" -> "SeqLen", 0
    {"masked_lm_ids", {"PredictionsPerSeq", 0}}  // int64[batch,dynamic_prediction_count]
};

// generic properties for storing perf metrics
MapStringToString mapped_dimensions;

void setup_training_params(BertParameters& params) {
  auto model_name_base = ToPathString(params.model_name);
  params.model_path = model_name_base + ORT_TSTR(".onnx");
  params.model_with_loss_func_path = model_name_base + ORT_TSTR("_with_cost.onnx");
  params.model_with_training_graph_path = model_name_base + ORT_TSTR("_bw.onnx");
  params.model_actual_running_graph_path = model_name_base + ORT_TSTR("_bw_running.onnx");
  params.model_with_gist_nodes_path = model_name_base + ORT_TSTR("_with_gist.onnx");

#if defined(USE_MPI)
  if (params.pipeline_parallel_size > 1) {
    auto pipeline_model_name_base = model_name_base + ToPathString(std::to_string(MPIContext::GetInstance().GetWorldRank()));
    params.pipeline_partitioned_model_path = pipeline_model_name_base + ORT_TSTR("_partitioned.onnx");
    params.model_with_loss_func_path = pipeline_model_name_base + ORT_TSTR("_with_cost.onnx");
    params.model_with_training_graph_path = pipeline_model_name_base + ORT_TSTR("_bw.onnx");
    params.model_actual_running_graph_path = pipeline_model_name_base + ORT_TSTR("_bw_running.onnx");
  }
  ORT_ENFORCE(params.horizontal_parallel_size <= MPIContext::GetInstance().GetWorldSize());
  ORT_ENFORCE(params.data_parallel_size <= MPIContext::GetInstance().GetWorldSize());
  if (MPIContext::GetInstance().GetWorldSize() % params.horizontal_parallel_size != 0) {
    LOGS_DEFAULT(ERROR) << "Cannot split horizontal parallel group because world_size is not divisible";
    return;
  }

  auto data_group_size = MPIContext::GetInstance().GetWorldSize() / (params.horizontal_parallel_size * params.pipeline_parallel_size);
  ORT_ENFORCE(data_group_size > 0, "Insufficient processes lead to zero-way data parallelism, which should be at least one-way.");
  if (data_group_size != params.data_parallel_size) {
    LOGS_DEFAULT(WARNING) << "WARNING: data_parallel_size is not correct, tuned automatically to "
                          << data_group_size << std::endl;
    params.data_parallel_size = data_group_size;
  }

  params.enable_adasum = params.enable_adasum && (params.data_parallel_size > 1);
  if (params.enable_adasum)
    std::cout << "Use Adasum for allreduce." << std::endl;
#endif

#ifdef USE_CUDA
  {
    CUDAExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());
    if (params.gpu_mem_limit_in_gb > 0) {
      info.gpu_mem_limit = gsl::narrow<size_t>(params.gpu_mem_limit_in_gb * 1024 * 1024 * 1024);
    }
    info.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;

    params.providers.emplace(kCudaExecutionProvider, CreateExecutionProviderFactory_CUDA(info));
    params.input_allocator = std::make_shared<CUDAPinnedAllocator>(info.device_id, GPU_PINNED);
  }
#endif

#ifdef USE_ROCM
  {
    ROCMExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());

    params.providers.emplace(kRocmExecutionProvider, CreateExecutionProviderFactory_ROCM(info));
    params.input_allocator = std::make_shared<ROCMPinnedAllocator>(info.device_id, GPU_PINNED);
  }
#endif

  params.loss_func_info = LossFunctionInfo(OpDef("BertLoss", kOnnxDomain),
                                           "total_loss",
                                           {/*prediction_masked_lm*/ "output1",
                                            /*prediction_next_sentence*/ "output2",
                                            /*masked_lm_positions*/ "masked_lm_positions",
                                            /*masked_lm_ids*/ "masked_lm_ids",
                                            /*next_sentence_labels*/ "next_sentence_labels",
                                            /*mlm_loss*/ "mlm_loss",
                                            /*nsp_loss*/ "nsp_loss"});

  params.fetch_names = {"total_loss", "mlm_loss", "nsp_loss"};

  if (params.EnableTensorboard()) {
    params.fetch_names.push_back(params.summary_name);
    params.scalar_names = {"total_loss", "mlm_loss", "nsp_loss", params.lr_params.feed_name};
  }

  params.immutable_weights = {
      {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
      {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}}},
      {"Mul", {{1, 0.5f}, {1, -10000.0f}}},
      {"Sub", {{0, 1.0f}}}};

  params.shuffle_data = false;

  // name_in_data_file -> name_in_model
  params.input_name_map = {
      {"input_ids", "input1"},
      {"segment_ids", "input2"},
      {"input_mask", "input3"},
      {"masked_lm_positions", "masked_lm_positions"},
      {"masked_lm_ids", "masked_lm_ids"},
      {"next_sentence_label", "next_sentence_labels"}};

  params.model_type = "bert";

  params.skip_evaluation = params.is_perf_test;

  params.error_function = [params](const std::vector<std::string>& /*feed_names*/,
                                   const std::vector<OrtValue>& /*feeds*/,
                                   const std::vector<std::string>& fetch_names,
                                   const std::vector<OrtValue>& fetches,
                                   size_t step) {
    const Tensor& total_loss_t = fetches[0].Get<Tensor>();
    const Tensor& mlm_loss_t = fetches[1].Get<Tensor>();
    const Tensor& nsp_loss_t = fetches[2].Get<Tensor>();

    const auto curr_total_loss = GetLossValue(total_loss_t);
    const auto curr_mlm_loss = GetLossValue(mlm_loss_t);
    const auto curr_nsp_loss = GetLossValue(nsp_loss_t);

    total_loss += curr_total_loss;
    mlm_loss += curr_mlm_loss;
    nsp_loss += curr_nsp_loss;

    if (params.EnableTensorboard()) {
      const Tensor& summary_loss_t = fetches[3].Get<Tensor>();
      summary_loss.push_back(*(summary_loss_t.template Data<std::string>()));
    }

    if (params.dump_fetches) {
      std::ostringstream filename;
      filename << "./fetch_dumps/rank_" << MPIContext::GetInstance().GetWorldRank() << "_step_" << step << ".txt";
      ofstream ofs(filename.str());
      for (size_t i = 0; i < fetch_names.size(); ++i) {
        TrainingUtil::PrintTensor(fetch_names[i], fetches[i].Get<Tensor>(), ofs);
      }
      ofs.close();
    }

    if (!params.convergence_test_output_file.empty()) {
      const ConvergenceTestDataRecord convergence_test_data{step, curr_total_loss, curr_mlm_loss, curr_nsp_loss};
      std::ofstream output_file{params.convergence_test_output_file, std::ios_base::app};
      output_file << convergence_test_data.GetCsvLine();
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

    printf("Step: %zu, #examples: %d, total_loss: %0.04f, mlm_loss: %0.04f, nsp_loss: %0.04f \n\n",
           step,
           static_cast<int>(num_samples),
           total_loss,
           mlm_loss,
           nsp_loss);
    total_loss = 0.0f;
    mlm_loss = 0.0f;
    nsp_loss = 0.0f;
    summary_loss.clear();
  };
}

static bool GetParametersForPhase(
    size_t phase,  // counting from 0
    const BertParameters& base_parameters, BertParameters& round_parameters) {
  // beyond phase 2
  if (phase >= 2) return false;

  // don't do phase 2
  if (phase == 1 && base_parameters.train_data_dir_phase2.empty()) return false;

  round_parameters = base_parameters;

  if (phase == 1) {  // phase 2
    round_parameters.train_data_dir = round_parameters.train_data_dir_phase2;
    round_parameters.test_data_dir = round_parameters.test_data_dir_phase2;

    round_parameters.lr_params.initial_lr = round_parameters.initial_lr_phase2;
    round_parameters.lr_params.warmup_ratio = round_parameters.warmup_ratio_phase2;
    round_parameters.num_train_steps = round_parameters.num_train_steps_phase2;
    round_parameters.batch_size = round_parameters.batch_size_phase2;
    round_parameters.gradient_accumulation_steps = round_parameters.gradient_accumulation_steps_phase2;
  }

  return true;
}

static Status RunPerformanceTest(const BertParameters& params, const Environment& env) {
  // setup fake data
  const int batch_size = static_cast<int>(params.batch_size);
  std::vector<std::string> tensor_names = {"input1", /*input_ids*/
                                           "input2", /*token_type_ids*/
                                           "input3", /*input_mask*/
                                           "masked_lm_positions",
                                           "masked_lm_ids",
                                           "next_sentence_labels"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, params.max_sequence_length},
                                            {batch_size, params.max_sequence_length},
                                            {batch_size, params.max_sequence_length},
                                            {batch_size, params.max_predictions_per_sequence},
                                            {batch_size, params.max_predictions_per_sequence},
                                            {batch_size}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64,
                                                          onnx::TensorProto_DataType_INT64};
  const size_t num_of_perf_samples = params.num_train_steps * params.batch_size;
  auto random_perf_data = std::make_shared<RandomDataSet>(num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);
  auto random_perf_data_loader = std::make_unique<SingleDataLoader>(random_perf_data, tensor_names);

  TrainingRunner runner{params, env, session_options};
  ORT_RETURN_IF_ERROR(runner.Initialize());
  ORT_RETURN_IF_ERROR(runner.Run(random_perf_data_loader.get(), random_perf_data_loader.get()));

  return Status::OK();
}

static Status RunTraining(const BertParameters& params, const Environment& env) {
  const size_t max_num_files_preload = 2;

  auto runner = std::make_unique<TrainingRunner>(params, env, session_options);
  ORT_RETURN_IF_ERROR(runner->Initialize());

  BertParameters params_for_phase;
  while (GetParametersForPhase(runner->GetRound(), params, params_for_phase)) {
    ORT_RETURN_IF_ERROR(runner->UpdateParams(params_for_phase));
    auto rank_in_data_parallel_group = (MPIContext::GetInstance().GetWorldRank() / params_for_phase.horizontal_parallel_size) % params_for_phase.data_parallel_size;
    auto training_data_loader = std::make_unique<DataLoader>(params_for_phase.input_name_map,
                                                                     params_for_phase.train_data_dir,
                                                                     max_num_files_preload,
                                                                     rank_in_data_parallel_group,
                                                                     params_for_phase.data_parallel_size);

    auto test_data_loader = std::unique_ptr<DataLoader>{};
    // Evaluation is only done in device #0
    if (MPIContext::GetInstance().GetWorldRank() == 0) {
      test_data_loader = std::make_unique<DataLoader>(params_for_phase.input_name_map,
                                                              params_for_phase.test_data_dir,
                                                              max_num_files_preload);
    }

    if (!params.perf_output_dir.empty()) {
      // collecting Bert related params from training data
      auto training_data = training_data_loader->CurrentDataSet();
      ORT_RETURN_IF_ERROR(training_data->GetTensorDimensionsFromInputs(input_to_dimension_mapping, mapped_dimensions));
    }

    ORT_RETURN_IF_ERROR(runner->Run(training_data_loader.get(), test_data_loader.get(), mapped_dimensions));

    ORT_RETURN_IF_ERROR(runner->ResetLossScaler());
  }

  if (MPIContext::GetInstance().GetWorldRank() == 0) {
    // Pass in empty dataloader to disable evaluation in EndTraining
    // to avoid a redundant synchronization caused by Tensorboard's SummaryMerge Op.
    ORT_RETURN_IF_ERROR(runner->EndTraining(nullptr));
  }

  return Status::OK();
}

int main(int argc, char* argv[]) {
  BertParameters params;
  OrtParameters ort_params{};
  RETURN_IF_FAIL(ParseArguments(argc, argv, params, ort_params));
  bool keep_looping = params.debug_break;
  while(keep_looping);

  // setup logger, be noted: LOGS_DEFAULT must be after logging manager initialization.
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  ort_params.log_severity,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id,
                                                  ort_params.vlog_level};
  setup_training_params(params);

  // setup profiling
  if (ort_params.max_num_profiling_events > 0) {
    profiling::Profiler::SetGlobalMaxNumEvents(ort_params.max_num_profiling_events);
  }

  // setup onnxruntime env
  unique_ptr<Environment> env;
  RETURN_IF_FAIL(Environment::Create(nullptr, env));

  // initialize test output file
  if (!params.convergence_test_output_file.empty()) {
    std::ofstream output_file(params.convergence_test_output_file);
    LOGS_DEFAULT_IF(!output_file, WARNING)
        << "Failed to open convergence test output file: " << ToMBString(params.convergence_test_output_file);
    output_file << ConvergenceTestDataRecord::GetCsvHeaderLine();
  }

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
