// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_state_common.h"

// pybind11/stl.h is needed to support std::unordered_set, etc.
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "core/graph/model.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/agent/training_agent.h"
#include "orttraining/core/graph/gradient_config.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "python/onnxruntime_pybind_mlvalue.h"

PYBIND11_MAKE_OPAQUE(std::vector<OrtValue>);

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
  std::unordered_map<std::string, std::vector<int>> sliced_schema;
  std::unordered_map<std::string, int> sliced_axes;
  std::vector<std::string> sliced_tensor_names;
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
  int num_pipeline_micro_batches = 1;
  int deepspeed_zero_stage = 0;
  bool enable_grad_norm_clip = true;
  bool set_gradients_as_graph_outputs = false;
  bool use_invertible_layernorm_grad = false;

  std::string pipeline_cut_info_string = {};

  // recompute
  bool attn_dropout_recompute = false;
  bool gelu_recompute = false;
  bool transformer_layer_recompute = false;
  int number_recompute_layers = 0;
  bool enable_adasum = false;

  // transformation
  int propagate_cast_ops_level = 1;
  std::vector<std::string> propagate_cast_ops_allow;
  GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy propagate_cast_ops_strategy =
      GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::None;
  bool allow_layer_norm_mod_precision = false;

  // graph dumping
  std::string model_after_graph_transforms_path;
  std::string model_with_gradient_graph_path;
  std::string model_with_training_graph_path;
};

struct TrainingConfigurationResult {
  optional<std::string> loss_scale_input_name;
};

struct PyGradientGraphBuilder {
  GradientGraphBuilder& builder;
  Model& model;
  PyGradientGraphBuilder(GradientGraphBuilder& builder, Model& model)
      : builder(builder), model(model) {}
};

// TODO: this method does not handle parallel optimization.
TrainingConfigurationResult ConfigureSessionForTraining(
    training::PipelineTrainingSession* sess, TrainingParameters& parameters) {
  //TODO tix, refactor the mpi related code to populate all fields correctly by default.
  ORT_ENFORCE(parameters.data_parallel_size <= parameters.world_size, "data_parallel_size: ", parameters.data_parallel_size, ", world_size: ", parameters.world_size);
  ORT_ENFORCE(parameters.horizontal_parallel_size <= parameters.world_size, "horizontal_parallel_size: ", parameters.horizontal_parallel_size, ", world_size: ", parameters.world_size);
  ORT_ENFORCE(parameters.pipeline_parallel_size <= parameters.world_size, "pipeline_parallel_size: ", parameters.pipeline_parallel_size, ", world_size: ", parameters.world_size);

  // When DxHxP != the total number of ranks, we try adjusting D so that DxHxP == the total number of ranks.
  if (parameters.world_size != parameters.data_parallel_size * parameters.horizontal_parallel_size * parameters.pipeline_parallel_size) {
    ORT_ENFORCE(parameters.world_size % parameters.horizontal_parallel_size * parameters.pipeline_parallel_size == 0,
                "D, H, P sizes are incorrect. To enable automatic correction, total number of ranks must be a divisible by HxP.");

    const auto new_data_parallel_size = parameters.world_size / (parameters.horizontal_parallel_size * parameters.pipeline_parallel_size);
    parameters.data_parallel_size = new_data_parallel_size;

    const std::string msg = "Cannot distribute " + std::to_string(parameters.world_size) + " ranks for distributed computation with D=" + std::to_string(parameters.data_parallel_size) +
                            ", H=" + std::to_string(parameters.horizontal_parallel_size) + ", P=" + std::to_string(parameters.pipeline_parallel_size) + ", so D is automatically changed to " + std::to_string(new_data_parallel_size);
    LOGS(*(sess->GetLogger()), WARNING) << msg;
  }

  training::PipelineTrainingSession::TrainingConfiguration config{};
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
  config.distributed_config.pipeline_parallel_size = parameters.pipeline_parallel_size;
  config.distributed_config.num_pipeline_micro_batches = parameters.num_pipeline_micro_batches;
  config.distributed_config.sliced_schema = parameters.sliced_schema;
  config.distributed_config.sliced_axes = parameters.sliced_axes;
  config.distributed_config.sliced_tensor_names = parameters.sliced_tensor_names;

  if (parameters.use_mixed_precision) {
    training::PipelineTrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_mixed_precision_initializers = true;

    config.mixed_precision_config = mp;
  }

  if (config.distributed_config.pipeline_parallel_size > 1) {
    training::PipelineTrainingSession::TrainingConfiguration::PipelineConfiguration pipeline_config;

    // Currently don't support auto-partition. User needs to pass in cut information for pipeline
    pipeline_config.do_partition = true;
    assert(!parameters.pipeline_cut_info_string.empty());

    auto process_with_delimiter = [](std::string& input_str, const std::string& delimiter) {
      std::vector<std::string> result;
      size_t pos = 0;
      while ((pos = input_str.find(delimiter)) != std::string::npos) {
        std::string token = input_str.substr(0, pos);
        result.emplace_back(token);
        input_str.erase(0, pos + delimiter.length());
      }
      // push the last split of substring into result.
      result.emplace_back(input_str);
      return result;
    };

    auto process_cut_info = [&](std::string& cut_info_string) {
      std::vector<PipelineTrainingSession::TrainingConfiguration::CutInfo> cut_list;
      const std::string group_delimiter = ",";
      const std::string edge_delimiter = ":";
      const std::string consumer_delimiter = "/";
      const std::string producer_consumer_delimiter = "-";

      auto cut_info_groups = process_with_delimiter(cut_info_string, group_delimiter);
      for (auto& cut_info_group : cut_info_groups) {
        PipelineTrainingSession::TrainingConfiguration::CutInfo cut_info;
        auto cut_edges = process_with_delimiter(cut_info_group, edge_delimiter);
        for (auto& cut_edge : cut_edges) {
          auto process_edge = process_with_delimiter(cut_edge, producer_consumer_delimiter);
          if (process_edge.size() == 1) {
            PipelineTrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0]};
            cut_info.emplace_back(edge);
          } else {
            ORT_ENFORCE(process_edge.size() == 2);
            auto consumer_list = process_with_delimiter(process_edge[1], consumer_delimiter);

            PipelineTrainingSession::TrainingConfiguration::CutEdge edge{process_edge[0], consumer_list};
            cut_info.emplace_back(edge);
          }
        }
        cut_list.emplace_back(cut_info);
      }
      return cut_list;
    };

    pipeline_config.cut_list = process_cut_info(parameters.pipeline_cut_info_string);
    config.pipeline_config = pipeline_config;
  }
  config.loss_name = parameters.loss_output_name;

  if (!parameters.training_optimizer_name.empty()) {
    training::PipelineTrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
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
    opt.do_all_reduce_in_mixed_precision_type = true;
    // TODO: this mapping is temporary.
    // For now, nccl allreduce kernel only implements for allreduce_post_accumulation
    // hovorod allreduce kernel only implements for not allreduce_post_accumulation.
    // eventually we will have one all reduce kernel and let opt to have
    // an allreduce_post_accumulation option and remove the use_nccl option.
    opt.use_nccl = parameters.allreduce_post_accumulation;
    opt.deepspeed_zero = onnxruntime::training::ZeROConfig(parameters.deepspeed_zero_stage);
    opt.enable_grad_norm_clip = parameters.enable_grad_norm_clip;

    // TODO reduction types
    if (parameters.enable_adasum) {
#ifdef USE_CUDA
      opt.adasum_reduction_type = training::AdasumReductionType::GpuHierarchicalReduction;
#else
      opt.adasum_reduction_type = training::AdasumReductionType::CpuReduction;
#endif
    }

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
  config.graph_transformer_config.propagate_cast_ops_config.strategy = parameters.propagate_cast_ops_strategy;
  config.graph_transformer_config.propagate_cast_ops_config.level = parameters.propagate_cast_ops_level;
  config.graph_transformer_config.propagate_cast_ops_config.allow = parameters.propagate_cast_ops_allow;
  config.graph_transformer_config.allow_layer_norm_mod_precision = parameters.allow_layer_norm_mod_precision;

  if (!parameters.model_after_graph_transforms_path.empty()) {
    config.model_after_graph_transforms_path = ToPathString(parameters.model_after_graph_transforms_path);
  }
  if (!parameters.model_with_gradient_graph_path.empty()) {
    config.model_with_gradient_graph_path = ToPathString(parameters.model_with_gradient_graph_path);
  }
  if (!parameters.model_with_training_graph_path.empty()) {
    config.model_with_training_graph_path = ToPathString(parameters.model_with_training_graph_path);
  }

  training::PipelineTrainingSession::TrainingConfigurationResult config_result{};

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

std::unordered_map<std::string, std::unordered_map<std::string, py::object>> ConvertORTTensorMapToNumpy(std::unordered_map<std::string, NameMLValMap> c_tensor_state, const DataTransferManager& data_transfer_manager) {
  std::unordered_map<std::string, std::unordered_map<std::string, py::object>> py_tensor_state;
  for (const auto& layer1_item : c_tensor_state) {
    py_tensor_state[layer1_item.first] = {};
    for (const auto& layer2_item : layer1_item.second) {
      assert(layer2_item.second.IsTensor());
      py::object obj;
      const Tensor& rtensor = layer2_item.second.Get<Tensor>();
      GetPyObjFromTensor(rtensor, obj, &data_transfer_manager);
      py_tensor_state[layer1_item.first].insert({layer2_item.first, obj});
    }
  }
  return py_tensor_state;
}

void addObjectMethodsForTraining(py::module& m) {
  py::class_<std::vector<OrtValue>>(m, "OrtValueVector")
      .def(py::init<>())
      .def("push_back", [](std::vector<OrtValue>* v, const OrtValue& ortvalue) {
        v->push_back(ortvalue);
      })
      .def("push_back", [](std::vector<OrtValue>* v, py::object dlpack_tensor, const bool is_bool_tensor) {
        v->push_back(FromDlpack(dlpack_tensor.ptr(), is_bool_tensor));
      })
      .def("reserve", [](std::vector<OrtValue>* v, const size_t len) { v->reserve(len); })
      .def("shrink_to_fit", [](std::vector<OrtValue>* v) { v->shrink_to_fit(); })
      .def("__len__", [](const std::vector<OrtValue>& v) { return v.size(); })
      .def("__iter__", [](const std::vector<OrtValue>& v) {
        return py::make_iterator(v.cbegin(), v.cend());
      },
           py::keep_alive<0, 1>())
      .def("__getitem__", [](const std::vector<OrtValue>& v, const size_t idx) {
        return v.at(idx);
      })
      .def("dlpack_at", [](std::vector<OrtValue>* v, const size_t idx) {
        return py::reinterpret_steal<py::object>(ToDlpack(v->at(idx)));
      });

  py::class_<TrainingParameters> parameters(m, "TrainingParameters", R"pbdoc(Configuration information for training.)pbdoc");
  parameters.def(py::init())
      .def_readwrite("loss_output_name", &TrainingParameters::loss_output_name)
      .def_readwrite("immutable_weights", &TrainingParameters::immutable_weights)
      .def_readwrite("weights_not_to_train", &TrainingParameters::weights_not_to_train)
      .def_readwrite("weights_to_train", &TrainingParameters::weights_to_train)
      .def_readwrite("sliced_tensor_names", &TrainingParameters::sliced_tensor_names)
      .def_readwrite("training_optimizer_name", &TrainingParameters::training_optimizer_name)
      .def_readwrite("lr_params_feed_name", &TrainingParameters::lr_params_feed_name)
      .def_readwrite("optimizer_attributes_map", &TrainingParameters::optimizer_attributes_map)
      .def_readwrite("optimizer_int_attributes_map", &TrainingParameters::optimizer_int_attributes_map)
      .def_readwrite("sliced_schema", &TrainingParameters::sliced_schema)
      .def_readwrite("sliced_axes", &TrainingParameters::sliced_axes)
      .def_readwrite("use_fp16_moments", &TrainingParameters::use_fp16_moments)
      .def_readwrite("use_mixed_precision", &TrainingParameters::use_mixed_precision)
      .def_readwrite("allreduce_post_accumulation", &TrainingParameters::allreduce_post_accumulation)
      .def_readwrite("loss_scale", &TrainingParameters::loss_scale)
      .def_readwrite("world_rank", &TrainingParameters::world_rank)
      .def_readwrite("world_size", &TrainingParameters::world_size)
      .def_readwrite("data_parallel_size", &TrainingParameters::data_parallel_size)
      .def_readwrite("horizontal_parallel_size", &TrainingParameters::horizontal_parallel_size)
      .def_readwrite("pipeline_parallel_size", &TrainingParameters::pipeline_parallel_size)
      .def_readwrite("pipeline_cut_info_string", &TrainingParameters::pipeline_cut_info_string)
      .def_readwrite("num_pipeline_micro_batches", &TrainingParameters::num_pipeline_micro_batches)
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
           })
      .def_readwrite("model_after_graph_transforms_path", &TrainingParameters::model_after_graph_transforms_path)
      .def_readwrite("model_with_gradient_graph_path", &TrainingParameters::model_with_gradient_graph_path)
      .def_readwrite("model_with_training_graph_path", &TrainingParameters::model_with_training_graph_path)
      .def_readwrite("enable_adasum", &TrainingParameters::enable_adasum)
      .def_readwrite("propagate_cast_ops_level", &TrainingParameters::propagate_cast_ops_level)
      .def_readwrite("propagate_cast_ops_allow", &TrainingParameters::propagate_cast_ops_allow)
      .def_readwrite("allow_layer_norm_mod_precision", &TrainingParameters::allow_layer_norm_mod_precision);

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
        : PyInferenceSession(std::make_unique<PipelineTrainingSession>(so, env)) {
    }
  };

  py::class_<PyTrainingSession, PyInferenceSession> training_session(m, "TrainingSession");
  training_session
      .def(py::init([](const PySessionOptions& so) {
        Environment& env = GetEnv();
        return std::make_unique<PyTrainingSession>(env, so);
      }))
      .def(py::init([]() {
        Environment& env = GetEnv();
        return std::make_unique<PyTrainingSession>(env, GetDefaultCPUSessionOptions());
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
      .def("load_model", [](PyTrainingSession* sess, const std::string& path, TrainingParameters& parameters, const std::vector<std::string>& provider_types, const ProviderOptionsVector& provider_options) {
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(path));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle()), parameters);

        InitializeSession(sess->GetSessionHandle(), provider_types, provider_options);

        return config_result;
      })
      .def("read_bytes", [](PyTrainingSession* sess, const py::bytes& serialized_model, TrainingParameters& parameters, const std::vector<std::string>& provider_types, const ProviderOptionsVector& provider_options) {
        std::istringstream buffer(serialized_model);
        OrtPybindThrowIfError(sess->GetSessionHandle()->Load(buffer));

#if defined(USE_MPI)
        bool use_nccl = parameters.allreduce_post_accumulation;
        if (!use_nccl && parameters.world_size > 1)
          CopyMPIContextToTrainingParameters(parameters, sess->GetSessionHandle()->GetLogger());
#endif
        const auto config_result = ConfigureSessionForTraining(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle()), parameters);

        InitializeSession(sess->GetSessionHandle(), provider_types, provider_options);

        return config_result;
      })
      .def("get_state", [](PyTrainingSession* sess) {
        NameMLValMap state_tensors;
        ORT_THROW_IF_ERROR(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->GetStateTensors(state_tensors));
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
      .def("get_model_state", [](PyTrainingSession* sess, bool include_mixed_precision_weights) {
        std::unordered_map<std::string, NameMLValMap> model_state_tensors;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetModelState(model_state_tensors, include_mixed_precision_weights));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        return ConvertORTTensorMapToNumpy(model_state_tensors, data_transfer_manager);
      })
      .def("get_optimizer_state", [](PyTrainingSession* sess) {
        std::unordered_map<std::string, NameMLValMap> opt_state_tensors;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetOptimizerState(opt_state_tensors));
        auto& data_transfer_manager = sess->GetSessionHandle()->GetDataTransferManager();
        return ConvertORTTensorMapToNumpy(opt_state_tensors, data_transfer_manager);
      })
      .def("get_partition_info_map", [](PyTrainingSession* sess) {
        std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> part_info_map;
        ORT_THROW_IF_ERROR(static_cast<TrainingSession*>(sess->GetSessionHandle())->GetPartitionInfoMap(part_info_map));
        return part_info_map;
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
        ORT_THROW_IF_ERROR(static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->SetStateTensors(state_tensors, strict));
      })
      .def("is_output_fp32_node", [](PyTrainingSession* sess, const std::string& output_name) {
        return static_cast<PipelineTrainingSession*>(sess->GetSessionHandle())->IsGraphOutputFp32Node(output_name);
      });

  py::class_<PartialGraphExecutionState>(m, "PartialGraphExecutionState")
      .def(py::init([]() {
        return std::make_unique<PartialGraphExecutionState>();
      }));

  py::class_<TrainingAgent>(m, "TrainingAgent", R"pbdoc(This is the main class used to run a ORTModule model.)pbdoc")
      .def(py::init([](PyInferenceSession* session, const std::vector<std::string>& fw_feed_names,
                       const std::vector<OrtDevice>& fw_outputs_device_info,
                       const std::vector<std::string>& bw_fetches_names,
                       const std::vector<OrtDevice>& bw_outputs_device_info) {
        return std::make_unique<TrainingAgent>(*session->GetSessionHandle(), fw_feed_names, fw_outputs_device_info,
                                               bw_fetches_names, bw_outputs_device_info);
      }))
      .def("run_forward", [](TrainingAgent* agent, const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState* state) -> void {
        Status status = agent->RunForward(feeds, fetches, *state);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in forward pass execution: " + status.ErrorMessage());
        }
      })
      .def("run_backward", [](TrainingAgent* agent, const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState* state) -> void {
        Status status = agent->RunBackward(feeds, fetches, *state);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in backward pass execution: " + status.ErrorMessage());
        }
      });

  py::enum_<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(m, "PropagateCastOpsStrategy", py::module_local(), py::arithmetic{})
      .value("NONE", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::None)
      .value("INSERT_AND_REDUCE", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::InsertAndReduce)
      .value("FLOOD_FILL", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill)
      .value("REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS", GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::RemoveInputOutputUpDownCasts)
      .def("__or__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator|))
      .def("__and__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                        GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator&))
      .def("__eq__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator==))
      .def("__neq__", py::overload_cast<GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy,
                                        GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy>(&operator!=));

  py::class_<GraphTransformerConfiguration::PropagateCastOpsConfiguration>
      propagate_cast_ops_config(
          m, "PropagateCastOpsConfiguration",
          R"pbdoc(Propagate cast ops configuration.)pbdoc");
  propagate_cast_ops_config.def(py::init())
      .def_readwrite("strategy", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::strategy)
      .def_readwrite("level", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::level)
      .def_readwrite("allow", &GraphTransformerConfiguration::PropagateCastOpsConfiguration::allow);

  py::class_<GraphTransformerConfiguration> graph_transformer_config(
      m, "GraphTransformerConfiguration",
      R"pbdoc(Graph transformer configuration.)pbdoc");
  graph_transformer_config.def(py::init())
      .def_readwrite("propagate_cast_ops_config", &GraphTransformerConfiguration::propagate_cast_ops_config);

  py::class_<TrainingGraphTransformerConfiguration, GraphTransformerConfiguration> training_graph_transformer_config(
      m, "TrainingGraphTransformerConfiguration",
      R"pbdoc(Training Graph transformer configuration.)pbdoc");
  training_graph_transformer_config.def(py::init())
      .def_readwrite("enable_gelu_approximation", &TrainingGraphTransformerConfiguration::enable_gelu_approximation)
      .def_readwrite("attn_dropout_recompute", &TrainingGraphTransformerConfiguration::attn_dropout_recompute)
      .def_readwrite("gelu_recompute", &TrainingGraphTransformerConfiguration::gelu_recompute)
      .def_readwrite("transformer_layer_recompute", &TrainingGraphTransformerConfiguration::transformer_layer_recompute)
      .def_readwrite("number_recompute_layers", &TrainingGraphTransformerConfiguration::number_recompute_layers)
      .def_readwrite("allow_layer_norm_mod_precision", &TrainingGraphTransformerConfiguration::allow_layer_norm_mod_precision)
      .def_readwrite("propagate_cast_ops_config", &TrainingGraphTransformerConfiguration::GraphTransformerConfiguration::propagate_cast_ops_config);

  py::class_<OrtModuleGraphBuilderConfiguration> module_graph_builder_config(
      m, "OrtModuleGraphBuilderConfiguration",
      R"pbdoc(Configuration information for module graph builder.)pbdoc");

  py::enum_<Severity>(m, "Severity", py::arithmetic(), py::module_local())
      .value("VERBOSE", logging::Severity::kVERBOSE)
      .value("INFO", logging::Severity::kINFO)
      .value("WARNING", logging::Severity::kWARNING)
      .value("ERROR", logging::Severity::kERROR)
      .value("FATAL", logging::Severity::kFATAL);

  module_graph_builder_config.def(py::init())
      .def_readwrite("initializer_names", &OrtModuleGraphBuilderConfiguration::initializer_names)
      .def_readwrite("initializer_names_to_train", &OrtModuleGraphBuilderConfiguration::initializer_names_to_train)
      .def_readwrite("input_names_require_grad", &OrtModuleGraphBuilderConfiguration::input_names_require_grad)
      .def_readwrite("use_invertible_layernorm_grad",
                     &OrtModuleGraphBuilderConfiguration::use_invertible_layernorm_grad)
      .def_readwrite("build_gradient_graph", &OrtModuleGraphBuilderConfiguration::build_gradient_graph)
      .def_readwrite("graph_transformer_config", &OrtModuleGraphBuilderConfiguration::graph_transformer_config)
      .def_readwrite("enable_caching", &OrtModuleGraphBuilderConfiguration::enable_caching)
      .def_readwrite("loglevel", &OrtModuleGraphBuilderConfiguration::loglevel);

  py::class_<GraphInfo> graph_info(m, "GraphInfo",
                                   R"pbdoc(The information of split graphs for frontend.)pbdoc");
  graph_info.def(py::init())
      .def_readwrite("user_input_names", &GraphInfo::user_input_names)
      .def_readwrite("user_input_grad_names", &GraphInfo::user_input_grad_names)
      .def_readwrite("initializer_names", &GraphInfo::initializer_names)
      .def_readwrite("initializer_names_to_train", &GraphInfo::initializer_names_to_train)
      .def_readwrite("initializer_grad_names_to_train", &GraphInfo::initializer_grad_names_to_train)
      .def_readwrite("user_output_names", &GraphInfo::user_output_names)
      .def_readwrite("output_grad_indices_non_differentiable", &GraphInfo::output_grad_indices_non_differentiable)
      .def_readwrite("output_grad_indices_require_full_shape", &GraphInfo::output_grad_indices_require_full_shape)
      .def_readwrite("module_output_indices_requires_save_for_backward", &GraphInfo::module_output_indices_requires_save_for_backward)
      .def_readwrite("frontier_node_arg_map", &GraphInfo::frontier_node_arg_map)
      .def_readwrite("module_output_gradient_name", &GraphInfo::module_output_gradient_name);

  py::class_<OrtModuleGraphBuilder> ortmodule_graph_builder(m, "OrtModuleGraphBuilder");
  ortmodule_graph_builder.def(py::init([]() { return std::make_unique<OrtModuleGraphBuilder>(); }))
      .def("initialize",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder, const py::bytes& serialized_model,
              const OrtModuleGraphBuilderConfiguration& config) {
             std::istringstream buffer(serialized_model);
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Initialize(buffer, config));
           })
      .def("build",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Build());
           })
      .def("build",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder,
              const std::vector<std::vector<int64_t>>& input_shapes) {
             ORT_THROW_IF_ERROR(ortmodule_graph_builder->Build(&input_shapes));
           })
      .def("get_model",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
             return py::bytes(ortmodule_graph_builder->GetModel());
           })
      .def("get_inference_optimized_model",
           [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
             return py::bytes(ortmodule_graph_builder->GetInferenceOptimizedModel());
           })
      .def("get_graph_info", [](OrtModuleGraphBuilder* ortmodule_graph_builder) {
        return ortmodule_graph_builder->GetGraphInfo();
      });

  py::class_<PyGradientGraphBuilder> gradient_graph_builder(m, "GradientGraphBuilder", R"pbdoc(A utility for making a gradient graph that can be used to help train a model.)pbdoc");
  gradient_graph_builder.def(py::init([](
                                          const std::string& model_path,
                                          const std::unordered_set<std::string>& y_node_arg_names,
                                          const std::unordered_set<std::string>& x_node_arg_names,
                                          const std::string& loss_node_arg_name) {
                          auto file_path = ToPathString(model_path);
                          std::shared_ptr<Model> model;
                          auto logger = logging::LoggingManager::DefaultLogger();
                          ORT_THROW_IF_ERROR(Model::Load(file_path, model, nullptr, logger));
                          GradientGraphConfiguration gradient_graph_config{};
                          gradient_graph_config.set_gradients_as_graph_outputs = true;

                          GradientGraphBuilder builder{
                              &model->MainGraph(),
                              y_node_arg_names,
                              x_node_arg_names,
                              loss_node_arg_name,
                              gradient_graph_config,
                              logger};
                          return std::make_unique<PyGradientGraphBuilder>(builder, *model);
                        }))
      .def("build", [](PyGradientGraphBuilder* gradient_graph_builder) {
        ORT_THROW_IF_ERROR(gradient_graph_builder->builder.Build());
      })
      .def("save", [](PyGradientGraphBuilder* gradient_graph_builder, const std::string& path) {
        // TODO Maybe just call build here?
        ORT_THROW_IF_ERROR(Model::Save(gradient_graph_builder->model, path));
      });
}

}  // namespace python
}  // namespace onnxruntime
