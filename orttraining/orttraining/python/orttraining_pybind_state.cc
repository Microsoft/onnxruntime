// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_state_common.h"

// pybind11/stl.h is needed to support std::unordered_set, etc.
#include <pybind11/stl.h>

#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/mpi_context.h"
#include "python/onnxruntime_pybind_mlvalue.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;
using namespace onnxruntime::training;

struct TrainingParameters {
  std::string loss_output_name;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_set<std::string> weights_not_to_train;
  onnxruntime::training::TrainingSession::ImmutableWeights immutable_weights;

  // optimizer
  std::string training_optimizer_name;
  std::string lr_params_feed_name = "Learning_Rate";
  std::unordered_map<std::string, std::unordered_map<std::string, float>> optimizer_attributes_map;
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> optimizer_int_attributes_map;
  onnxruntime::training::TrainingSession::OptimizerState optimizer_initial_state;
  bool use_fp16_moments = false;

  bool use_mixed_precision = false;
  bool allreduce_post_accumulation = false;
  float loss_scale = 0.0f;
  int world_rank = 0;
  int world_size = 1;
  int local_rank = 0;
  int local_size = 1;
  int gradient_accumulation_steps = 1;
  int data_parallel_size = 1;
  int horizontal_parallel_size = 1;
  int pipeline_parallel_size = 1;
  int deepspeed_zero_stage = 0;
  bool enable_grad_norm_clip = true;
  bool set_gradients_as_graph_outputs = false;
  bool use_invertible_layernorm_grad = false;
  std::string output_model_path;
  bool use_external_data_format = false;

  // recompute
  bool attn_dropout_recompute = false;
  bool gelu_recompute = false;
  bool transformer_layer_recompute = false;
  int number_recompute_layers = 0;
};

struct TrainingConfigurationResult {
  optional<std::string> loss_scale_input_name;
};

// TODO: this method does not handle parallel optimization.
TrainingConfigurationResult ConfigureSessionForTraining(
    training::TrainingSession* sess, TrainingParameters& parameters) {
  //TODO tix, refactor the mpi related code to populate all fields correctly by default.
  ORT_ENFORCE(parameters.horizontal_parallel_size <= parameters.world_size);
  ORT_ENFORCE(parameters.data_parallel_size <= parameters.world_size);

  if (parameters.world_size % parameters.data_parallel_size != 0) {
    throw std::runtime_error("Cannot split data parallel group because world_size is not divisible");
  }

  if (parameters.world_size % parameters.horizontal_parallel_size != 0) {
    throw std::runtime_error("Cannot split horizontal parallel group because world_size is not divisible");
  }

  auto data_group_size = parameters.world_size / (parameters.horizontal_parallel_size * parameters.pipeline_parallel_size);
  if (data_group_size != parameters.data_parallel_size) {
    LOGS(*(sess->GetLogger()), WARNING) << "data_parallel_size is not correct, tuned automatically to "
                                        << data_group_size;
    parameters.data_parallel_size = data_group_size;
  }

  training::TrainingSession::TrainingConfiguration config{};
  config.weight_names_to_train = parameters.weights_to_train;
  config.weight_names_to_not_train = parameters.weights_not_to_train;
  config.immutable_weights = parameters.immutable_weights;

  config.gradient_accumulation_steps = parameters.gradient_accumulation_steps;

  config.distributed_config.world_rank = parameters.world_rank;
  config.distributed_config.world_size = parameters.world_size;
  config.distributed_config.local_rank = parameters.local_rank;
  config.distributed_config.local_size = parameters.local_size;
  config.distributed_config.data_parallel_size = parameters.data_parallel_size;
  config.distributed_config.horizontal_parallel_size = parameters.horizontal_parallel_size;

  if (parameters.use_mixed_precision) {
    training::TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_mixed_precision_initializers = true;

    config.mixed_precision_config = mp;
  }

  config.loss_name = parameters.loss_output_name;

  if (!parameters.output_model_path.empty()) {
    config.model_with_loss_function_path = parameters.output_model_path + ORT_TSTR("_with_cost.onnx");
    config.model_with_training_graph_path = parameters.output_model_path + ORT_TSTR("_bw.onnx");
  }

  if (!parameters.training_optimizer_name.empty()) {
    training::TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
    opt.name = parameters.training_optimizer_name;
    opt.learning_rate_input_name = parameters.lr_params_feed_name;
    opt.weight_attributes_generator = [&parameters](const std::string& weight_name) {
      const auto it = parameters.optimizer_attributes_map.find(weight_name);
      ORT_ENFORCE(
          it != parameters.optimizer_attributes_map.end(),
          "Failed to find attribute map for weight ", weight_name);
      return it->second;
    };
    opt.weight_int_attributes_generator = [&parameters](const std::string& weight_name) {
      const auto it = parameters.optimizer_int_attributes_map.find(weight_name);
  
      ORT_ENFORCE(
          it != parameters.optimizer_int_attributes_map.end(),
          "Failed to find int attribute map for weight ", weight_name);
      return it->second;
    };
    opt.use_mixed_precision_moments = parameters.use_fp16_moments;
    opt.do_all_reduce_in_mixed_precision_type = false;

    // TODO: this mapping is temporary.
    // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
    // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
    // eventually we will have one all reduce kernel and let opt to have
    // an allreduce_post_accumulation option and remove the use_nccl option.
    opt.use_nccl = parameters.allreduce_post_accumulation;
    opt.deepspeed_zero = onnxruntime::training::ZeROConfig(parameters.deepspeed_zero_stage);
    // TODO: The norm clipping value is 1.0f which is the default used in most frameworks.
    // Need to have another option to support more values in the future.
    opt.enable_grad_norm_clip = parameters.enable_grad_norm_clip;

    config.optimizer_config = opt;
  }

  if (!parameters.optimizer_initial_state.empty()) {
    config.init_optimizer_states = parameters.optimizer_initial_state;
  }

  config.gradient_graph_config.use_invertible_layernorm_grad = parameters.use_invertible_layernorm_grad;
  config.gradient_graph_config.set_gradients_as_graph_outputs = parameters.set_gradients_as_graph_outputs;

  config.graph_transformer_config.attn_dropout_recompute = parameters.attn_dropout_recompute;
  config.graph_transformer_config.gelu_recompute = parameters.gelu_recompute;
  config.graph_transformer_config.transformer_layer_recompute = parameters.transformer_layer_recompute;
  config.graph_transformer_config.number_recompute_layers = parameters.number_recompute_layers;

  training::TrainingSession::TrainingConfigurationResult config_result{};

  OrtPybindThrowIfError(sess->ConfigureForTraining(config, config_result));

  TrainingConfigurationResult python_config_result{};
  if (config_result.mixed_precision_config_result.has_value()) {
    const auto& mp_config_result = config_result.mixed_precision_config_result.value();
    python_config_result.loss_scale_input_name = mp_config_result.loss_scale_input_name;
  }

  return python_config_result;
}

#if defined(USE_MPI)
void CopyMPIContextToTrainingParameters(TrainingParameters& parameters, const logging::Logger* logger) {
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetWorldRank(): " << MPIContext::GetInstance().GetWorldRank();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetLocalRank(): " << MPIContext::GetInstance().GetLocalRank();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetWorldSize(): " << MPIContext::GetInstance().GetWorldSize();
  LOGS(*logger, INFO) << "MPIContext::GetInstance().GetLocalSize(): " << MPIContext::GetInstance().GetLocalSize();

  parameters.local_rank = MPIContext::GetInstance().GetLocalRank();
  parameters.local_size = MPIContext::GetInstance().GetLocalSize();
  if (parameters.world_rank != MPIContext::GetInstance().GetWorldRank()) {
    if (parameters.world_rank != 0)
      LOGS(*logger, WARNING) << "TrainingParameters world_rank is not correct, tuned automatically to " << MPIContext::GetInstance().GetWorldRank();
    parameters.world_rank = MPIContext::GetInstance().GetWorldRank();
  }
  if (parameters.world_size != MPIContext::GetInstance().GetWorldSize()) {
    if (parameters.world_size != 1)
      LOGS(*logger, WARNING) << "TrainingParameters world_size is not correct, tuned automatically to " << MPIContext::GetInstance().GetWorldSize();
    parameters.world_size = MPIContext::GetInstance().GetWorldSize();
  }
}
#endif

void addObjectMethodsForTraining(py::module& m) {
  py::class_<TrainingParameters> parameters(m, "TrainingParameters", R"pbdoc(Configuration information for training.)pbdoc");
  parameters.def(py::init())
      .def_readwrite("loss_output_name", &TrainingParameters::loss_output_name)
      .def_readwrite("immutable_weights", &TrainingParameters::immutable_weights)
      .def_readwrite("weights_not_to_train", &TrainingParameters::weights_not_to_train)
      .def_readwrite("weights_to_train", &TrainingParameters::weights_to_train)
      .def_readwrite("training_optimizer_name", &TrainingParameters::training_optimizer_name)
      .def_readwrite("lr_params_feed_name", &TrainingParameters::lr_params_feed_name)
      .def_readwrite("optimizer_attributes_map", &TrainingParameters::optimizer_attributes_map)
      .def_readwrite("optimizer_int_attributes_map", &TrainingParameters::optimizer_int_attributes_map)
      .def_readwrite("use_fp16_moments", &TrainingParameters::use_fp16_moments)
      .def_readwrite("use_mixed_precision", &TrainingParameters::use_mixed_precision)
      .def_readwrite("allreduce_post_accumulation", &TrainingParameters::allreduce_post_accumulation)
      .def_readwrite("loss_scale", &TrainingParameters::loss_scale)
      .def_readwrite("world_rank", &TrainingParameters::world_rank)
      .def_readwrite("world_size", &TrainingParameters::world_size)
      .def_readwrite("gradient_accumulation_steps", &TrainingParameters::gradient_accumulation_steps)
      .def_readwrite("deepspeed_zero_stage", &TrainingParameters::deepspeed_zero_stage)
      .def_readwrite("enable_grad_norm_clip", &TrainingParameters::enable_grad_norm_clip)
      .def_readwrite("set_gradients_as_graph_outputs", &TrainingParameters::set_gradients_as_graph_outputs)
      .def_readwrite("use_invertible_layernorm_grad", &TrainingParameters::use_invertible_layernorm_grad)
      .def_readwrite("attn_dropout_recompute", &TrainingParameters::attn_dropout_recompute)
      .def_readwrite("gelu_recompute", &TrainingParameters::gelu_recompute)
      .def_readwrite("transformer_layer_recompute", &TrainingParameters::transformer_layer_recompute)
      .def_readwrite("number_recompute_layers", &TrainingParameters::number_recompute_layers)
      .def_readwrite("data_parallel_size", &TrainingParameters::data_parallel_size)
      .def_readwrite("horizontal_parallel_size", &TrainingParameters::horizontal_parallel_size)
      .def_readwrite("pipeline_parallel_size", &TrainingParameters::pipeline_parallel_size)
      .def_readwrite("output_model_path", &TrainingParameters::output_model_path)
      .def_readwrite("use_external_data_format", &TrainingParameters::use_external_data_format)
      .def("set_optimizer_initial_state",
           [](TrainingParameters& parameters, const std::unordered_map<std::string, std::unordered_map<std::string, py::object>>& py_state) -> void {
             onnxruntime::training::TrainingSession::OptimizerState optim_state;
             for (const auto& weight_it : py_state) {
               auto state = weight_it.second;
               NameMLValMap state_tensors;
               for (auto& initializer : state) {
                 OrtValue ml_value;

                 // InputDeflist is null because parameters havent been tied to session yet
                 // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
                 CreateGenericMLValue(nullptr, GetAllocator(), "", initializer.second, &ml_value, true);
                 ThrowIfPyErrOccured();
                 state_tensors.emplace(initializer.first, ml_value);
               }
               optim_state.emplace(weight_it.first, state_tensors);
             }
             parameters.optimizer_initial_state = optim_state;
           });

#if defined(USE_MPI)
  m.def("get_mpi_context_local_rank", []() -> int { return MPIContext::GetInstance().GetLocalRank(); });
  m.def("get_mpi_context_local_size", []() -> int { return MPIContext::GetInstance().GetLocalSize(); });
  m.def("get_mpi_context_world_rank", []() -> int { return MPIContext::GetInstance().GetWorldRank(); });
  m.def("get_mpi_context_world_size", []() -> int { return MPIContext::GetInstance().GetWorldSize(); });
#endif

  py::class_<TrainingConfigurationResult> config_result(m, "TrainingConfigurationResult", "pbdoc(Configuration result for training.)pbdoc");
  config_result.def(py::init())
      .def_property_readonly("loss_scale_input_name", [](const TrainingConfigurationResult& result) -> py::object {
        if (result.loss_scale_input_name.has_value()) {
          return py::str{result.loss_scale_input_name.value()};
        }
        return py::none();
      });

  // Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
  struct PyTrainingSession : public PyInferenceSession {
    PyTrainingSession(Environment& env, const PySessionOptions& so)
        : PyInferenceSession(onnxruntime::make_unique<TrainingSession>(so, env)) {
    }
  };

  py::class_<PyTrainingSession, PyInferenceSession> training_session(m, "TrainingSession");
  training_session
      .def(py::init([](const PySessionOptions& so) {
        Environment& env = GetEnv();
        return onnxruntime::make_unique<PyTrainingSession>(env, so);
      }))
      .def(py::init([]() {
        Environment& env = GetEnv();
        return onnxruntime::make_unique<PyTrainingSession>(env, GetDefaultCPUSessionOptions());
      }))
      .def("finalize", [](py::object) {
#if defined(USE_MPI)
#ifdef _WIN32
        // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
        // shutdown_mpi() is not called within MPIContext destructor because of DllMain's restriction
        // call shutdown_mpi() here instead.
        MPIContext::shutdown_mpi();
#endif
#endif
      })
      .def("load_model", [](PyTrainingSession* sess, const std::string& path, TrainingParameters& parameters) {
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(path));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<TrainingSession*>(sess->GetSessionHandle()), parameters);
        return config_result;
      })
      .def("init_train_session", [](PyTrainingSession* sess) {
        std::vector<std::string> provider_types = {};
        InitializeSession(sess->GetSessionHandle(), provider_types);
      })
      .def("read_bytes", [](PyTrainingSession* sess, const py::bytes& serialized_model, TrainingParameters& parameters) {
        std::istringstream buffer(serialized_model);
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(buffer));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<TrainingSession*>(sess->GetSessionHandle()), parameters);

        std::vector<std::string> provider_types = {};
        InitializeSession(sess->GetSessionHandle(), provider_types);

        return config_result;
      })
      .def("get_state", [](PyTrainingSession* sess) {
        NameMLValMap state_tensors;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetStateTensors(state_tensors));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        //convert to numpy array
        std::map<std::string, py::object> rmap;
        for (auto& kv : state_tensors) {
          if (kv.second.IsTensor()) {
            py::object obj;
            const Tensor& rtensor = kv.second.Get<Tensor>();
            GetPyObjFromTensor(rtensor, obj, &data_transfer_manager);
            rmap.insert({kv.first, obj});
          } else {
            throw std::runtime_error("Non tensor type in session state tensors is not expected.");
          }
        }
        return rmap;
      })
      .def("load_state", [](PyTrainingSession* sess, std::unordered_map<std::string, py::object>& state, bool strict) {
        NameMLValMap state_tensors;
        for (auto initializer : state) {
          OrtValue ml_value;
          auto px = sess->GetSessionHandle()->GetModelInputs();
          if (!px.first.IsOK() || !px.second) {
            throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
          }
          CreateGenericMLValue(px.second, GetAllocator(), initializer.first, initializer.second, &ml_value);
          ThrowIfPyErrOccured();
          state_tensors.insert(std::make_pair(initializer.first, ml_value));
        }
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->SetStateTensors(state_tensors, strict));
      })
      .def("is_output_fp32_node", [](PyTrainingSession* sess, const std::string& output_name) {
        return static_cast<TrainingSession*>(sess->GetSessionHandle())->IsGraphOutputFp32Node(output_name);
      });
}
}  // namespace python
}  // namespace onnxruntime
