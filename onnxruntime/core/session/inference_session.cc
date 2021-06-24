// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <memory>
#include <sstream>
#include <unordered_set>
#include <list>
#include <string>
#include <thread>

#include "core/common/denormal.h"
#include "core/common/logging/logging.h"
#include "core/common/parse_string.h"
#include "core/framework/arena.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/platform/Barrier.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#ifdef USE_DML  // TODO: This is necessary for the workaround in TransformGraph
#include "core/providers/dml/DmlExecutionProvider/src/GraphTransformer.h"
#endif
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/util/thread_utils.h"

// custom ops are not available in a minimal build unless ORT_MINIMAL_BUILD_CUSTOM_OPS is set
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/customregistry.h"
#include "core/session/custom_ops.h"
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::experimental;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace {
template <typename T>
const T* GetDateFormatString();

template <>
inline const char* GetDateFormatString<char>() {
  return "%Y-%m-%d_%H-%M-%S";
}
#ifdef _WIN32
template <>
inline const wchar_t* GetDateFormatString<wchar_t>() {
  return L"%Y-%m-%d_%H-%M-%S";
}
#endif
// TODO: use LoggingManager::GetTimestamp and date::operator<<
// (see ostream_sink.cc for an example)
// to simplify this and match the log file timestamp format.
template <typename T>
inline std::basic_string<T> GetCurrentTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm;  // NOLINT

#ifdef _WIN32
  ORT_ENFORCE(localtime_s(&local_tm, &in_time_t) == 0);
#else
  localtime_r(&in_time_t, &local_tm);
#endif

  T time_str[32];
  OrtStrftime<T>(time_str, sizeof(time_str), GetDateFormatString<T>(), &local_tm);
  return std::basic_string<T>(time_str);
}

}  // namespace

std::atomic<uint32_t> InferenceSession::global_session_id_{1};

// The current model versions for saving the ort format models
// This version is NOT onnxruntime version
// Only update this version when there is a file format change which will break the compatibilites
// Once this model version is updated, the kSupportedOrtModelVersions in IsOrtModelVersionSupported
// below will also need to be updated.
// See onnxruntime/core/flatbuffers/schema/README.md for more details on versioning.
// Version 1 - history begins
// Version 2 - add serialization/deserialization of sparse_initializer
// Version 3 - add `graph_doc_string` to Model
// Version 4 - update kernel def hashing to not depend on ordering of type constraint types (NOT BACKWARDS COMPATIBLE)
static constexpr const char* kOrtModelVersion = "4";

#if defined(ENABLE_ORT_FORMAT_LOAD)
// Check if the given ort model version is supported in this build
static bool IsOrtModelVersionSupported(const std::string& ort_model_version) {
  // The ort model versions we will support in this build
  // This may contain more versions than the kOrtModelVersion, based on the compatibilities
  static const std::unordered_set<std::string> kSupportedOrtModelVersions{
      std::string(kOrtModelVersion),
  };

  return kSupportedOrtModelVersions.find(ort_model_version) != kSupportedOrtModelVersions.cend();
}
#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

static Status FinalizeSessionOptions(const SessionOptions& user_provided_session_options,
                                     const ONNX_NAMESPACE::ModelProto& model_proto,
                                     bool is_model_proto_parsed,
                                     /*out*/ SessionOptions& finalized_session_options) {
#if !defined(ORT_MINIMAL_BUILD)
  const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();

  // By now the environment should have initialized. (It is enforced prior to this.)
  const Env& env_instance = Env::Default();

  bool session_options_from_model = false;

  // Get the value held by the environment variable - kOrtLoadConfigFromModelEnvVar
  const std::string load_config_from_model_env_var_value =
      env_instance.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar);

  // Ascertain if the model is to be read for the ORT config from the afore parsed env var
  if (!load_config_from_model_env_var_value.empty()) {
    // Check if the env var contains an unsupported value
    if (load_config_from_model_env_var_value.length() > 1 ||
        (load_config_from_model_env_var_value[0] != '0' && load_config_from_model_env_var_value[0] != '1')) {
      std::ostringstream oss;
      oss << "The only supported values for the environment variable "
          << inference_session_utils::kOrtLoadConfigFromModelEnvVar << " are '0' and '1'. "
          << "The environment variable contained the value: " << load_config_from_model_env_var_value;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, oss.str());
    }

    if (load_config_from_model_env_var_value[0] == '1') {
      LOGS(default_logger, INFO) << "Reading the provided model for the ORT config";
      session_options_from_model = true;
    }
  }

  // The model is to be read for an ORT config json that may hold some/all session options
  if (session_options_from_model) {
    SessionOptions constructed_session_options;

    // In theory we should not hit this condition unless this internal class' APIs are being called incorrectly.
    // This is a good sanity check to enforce that the model has been parsed prior to looking into it for ort config.
    ORT_ENFORCE(is_model_proto_parsed, "ModelProto needs to be parsed to check for ORT config within it");

    // Use default logger as the session_logger_ hasn't been initialized yet.
    inference_session_utils::JsonConfigParser config_parser(default_logger);

    auto status = config_parser.ParseOrtConfigJsonInModelProto(model_proto);
    if (!status.IsOK()) {
      return status;
    }

    status = config_parser.ParseSessionOptionsFromModelProto(constructed_session_options);
    if (!status.IsOK()) {
      return status;
    }

    // use the constructed session options
    finalized_session_options = constructed_session_options;
  } else {
    // use user provided session options instance
    finalized_session_options = user_provided_session_options;
  }
#else
  ORT_UNUSED_PARAMETER(model_proto);
  ORT_UNUSED_PARAMETER(is_model_proto_parsed);
  finalized_session_options = user_provided_session_options;
#endif  // !defined(ORT_MINIMAL_BUILD)

  return Status::OK();
}

void InferenceSession::ConstructorCommon(const SessionOptions& session_options,
                                         const Environment& session_env) {
  auto status = FinalizeSessionOptions(session_options, model_proto_, is_model_proto_parsed_, session_options_);
  // a monotonically increasing session id for use in telemetry
  session_id_ = global_session_id_.fetch_add(1);
  ORT_ENFORCE(status.IsOK(), "Could not finalize session options while constructing the inference session. Error Message: ",
              status.ErrorMessage());

  // The call to InitLogger depends on the final state of session_options_. Hence it should be invoked
  // after the invocation of FinalizeSessionOptions.
  InitLogger(logging_manager_);  // this sets session_logger_ so that it can be used for logging after this point.

#if !defined(ORT_MINIMAL_BUILD)
  // Update the number of steps for the graph transformer manager using the "finalized" session options
  ORT_ENFORCE(graph_transformation_mgr_.SetSteps(session_options_.max_num_graph_transformation_steps).IsOK());
  status = ORTRegisterONNXOpsetSchema(session_options_.session_onnx_opset_version);
  ORT_ENFORCE(status.IsOK(), "Fail to register ONNX Opset schema: ",
              status.ErrorMessage());
#endif
  bool set_denormal_as_zero =
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSetDenormalAsZero, "0") == "1";

  // The only first session option for flush-to-zero and denormal-as-zero is effective to main thread and OpenMP threads.
  {
    static std::once_flag once;

    std::call_once(once, [&] {
#ifdef _OPENMP
      InitializeWithDenormalAsZero(set_denormal_as_zero);
#endif
      SetDenormalAsZero(set_denormal_as_zero);

      LOGS(*session_logger_, INFO) << "Flush-to-zero and denormal-as-zero are " << ((set_denormal_as_zero) ? "on" : "off");
    });
  }

  use_per_session_threads_ = session_options.use_per_session_threads;

  if (use_per_session_threads_) {
    LOGS(*session_logger_, INFO) << "Creating and using per session threadpools since use_per_session_threads_ is true";
    {
      bool allow_intra_op_spinning =
          session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowIntraOpSpinning, "1") == "1";
      OrtThreadPoolParams to = session_options_.intra_op_param;
      std::basic_stringstream<ORTCHAR_T> ss;
      if (to.name) {
        ss << to.name << ORT_TSTR("-");
      }
      ss << ORT_TSTR("session-") << session_id_ << ORT_TSTR("-intra-op");
      thread_pool_name_ = ss.str();
      to.name = thread_pool_name_.c_str();
      to.set_denormal_as_zero = set_denormal_as_zero;
      // If the thread pool can use all the processors, then
      // we set affinity of each thread to each processor.
      to.auto_set_affinity = to.thread_pool_size == 0 &&
                             session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL &&
                             to.affinity_vec_len == 0;
      to.allow_spinning = allow_intra_op_spinning;
      thread_pool_ =
          concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
    }
    if (session_options_.execution_mode == ExecutionMode::ORT_PARALLEL) {
      bool allow_inter_op_spinning =
          session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowInterOpSpinning, "1") == "1";
      OrtThreadPoolParams to = session_options_.inter_op_param;
      // If the thread pool can use all the processors, then
      // we set thread affinity.
      to.auto_set_affinity =
          to.thread_pool_size == 0 && session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL;
      std::basic_stringstream<ORTCHAR_T> ss;
      if (to.name) {
        ss << to.name << ORT_TSTR("-");
      }
      ss << ORT_TSTR("session-") << session_id_ << ORT_TSTR("-inter-op");
      inter_thread_pool_name_ = ss.str();
      to.name = inter_thread_pool_name_.c_str();
      to.set_denormal_as_zero = set_denormal_as_zero;
      to.allow_spinning = allow_inter_op_spinning;
      inter_op_thread_pool_ =
          concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
      if (inter_op_thread_pool_ == nullptr) {
        LOGS(*session_logger_, INFO) << "Failed to create the inter-op thread pool for the parallel executor, setting ExecutionMode to SEQUENTIAL";
        session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
      }
    }
  } else {
    LOGS(*session_logger_, INFO) << "Using global/env threadpools since use_per_session_threads_ is false";
    intra_op_thread_pool_from_env_ = session_env.GetIntraOpThreadPool();
    inter_op_thread_pool_from_env_ = session_env.GetInterOpThreadPool();
    ORT_ENFORCE(session_env.EnvCreatedWithGlobalThreadPools(),
                "When the session is not configured to use per session"
                " threadpools, the env must be created with the the CreateEnvWithGlobalThreadPools API.");
  }

  session_profiler_.Initialize(session_logger_);
  if (session_options_.enable_profiling) {
    StartProfiling(session_options_.profile_file_prefix);
  }

  telemetry_ = {};
  allocator_manager_ = std::make_shared<onnxruntime::AllocatorManager>();
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env)
    :
#if !defined(ORT_MINIMAL_BUILD)
      graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      insert_cast_transformer_("CastFloat16Transformer"),
#endif
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  // Initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#if !defined(ORT_MINIMAL_BUILD)
InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const std::string& model_uri)
    : model_location_(ToWideString(model_uri)),
      graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      insert_cast_transformer_("CastFloat16Transformer"),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  auto status = Model::Load(model_location_, model_proto_);
  ORT_ENFORCE(status.IsOK(), "Given model could not be parsed while creating inference session. Error message: ",
              status.ErrorMessage());
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

common::Status InferenceSession::AddONNXOpDomain(int onnx_opset_version) {
  return InferenceSession::ORTRegisterONNXOpsetSchema(onnx_opset_version);
}

#ifdef _WIN32
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const Environment& session_env,
                                   const std::wstring& model_uri)
    : graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      insert_cast_transformer_("CastFloat16Transformer"),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  model_location_ = ToWideString(model_uri);
  auto status = Model::Load(model_location_, model_proto_);
  ORT_ENFORCE(status.IsOK(), "Given model could not be parsed while creating inference session. Error message: ",
              status.ErrorMessage());
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}
#endif

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   std::istream& model_istream)
    : graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      insert_cast_transformer_("CastFloat16Transformer"),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  Status st = Model::Load(model_istream, &model_proto_);
  ORT_ENFORCE(st.IsOK(), "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const void* model_data, int model_data_len)
    : graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      insert_cast_transformer_("CastFloat16Transformer"),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  const bool result = model_proto_.ParseFromArray(model_data, model_data_len);
  ORT_ENFORCE(result, "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

InferenceSession::~InferenceSession() {
  if (session_options_.enable_profiling) {
    ORT_TRY {
      EndProfiling();
    }
    ORT_CATCH(const std::exception& e) {
      // TODO: Currently we have no way to transport this error to the API user
      // Maybe this should be refactored, so that profiling must be explicitly
      // started and stopped via C-API functions.
      // And not like now a session option and therefore profiling must be started
      // and stopped implicitly.
      ORT_HANDLE_EXCEPTION([&]() {
        LOGS(*session_logger_, ERROR) << "Error during EndProfiling(): " << e.what();
      });
    }
    ORT_CATCH(...) {
      LOGS(*session_logger_, ERROR) << "Unknown error during EndProfiling()";
    }
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  if (session_activity_started_)
    TraceLoggingWriteStop(session_activity, "OrtInferenceSessionActivity");
#endif
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryInfo::GenerateMemoryProfile();
#endif
}

common::Status InferenceSession::RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
  if (p_exec_provider == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for exec provider");
  }

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_inited_) {
    // adding an EP is pointless as the graph as already been partitioned so no nodes will be assigned to
    // the new EP
    LOGS(*session_logger_, ERROR) << "Execution providers must be registered before the session is initialized. ";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Execution providers must be registered before the session is initialized.");
  }

  const std::string& provider_type = p_exec_provider->Type();

  p_exec_provider->RegisterAllocator(allocator_manager_);

  // Some session option values (default or user provided) may not work with some EPs.
  // Rather than put the onus on the user to know these, make the appropriate change while logging the change.
  if (provider_type == onnxruntime::kDmlExecutionProvider) {
    // DML's memory is not byte addressable and hence mem pattern doesn't work.
    if (session_options_.enable_mem_pattern) {
      LOGS(*session_logger_, WARNING)
          << "Having memory pattern enabled is not supported while using the DML Execution Provider. "
          << "So disabling it for this session since it uses the DML Execution Provider.";
      session_options_.enable_mem_pattern = false;
    }

    // Parallel execution mode does not support DML EP
    if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
      LOGS(*session_logger_, WARNING)
          << "Parallel execution mode does not support the DML Execution Provider. "
          << "So making the execution mode sequential for this session since it uses the DML Execution Provider.";

      session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    }
  }

  if (provider_type == onnxruntime::kCudaExecutionProvider) {
    // Parallel execution mode does not support the CUDA EP
    if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
      LOGS(*session_logger_, WARNING)
          << "Parallel execution mode does not support the CUDA Execution Provider. "
          << "So making the execution mode sequential for this session since it uses the CUDA Execution Provider.";
      session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    }

    auto trt_ep = execution_providers_.Get(kTensorrtExecutionProvider);
    if (trt_ep) {
      p_exec_provider->SetComputeStream(trt_ep->GetComputeStream());
    }
  }

  VLOGS(*session_logger_, 1) << "Adding execution provider of type: " << provider_type;
  auto p_data_xfr = p_exec_provider->GetDataTransfer();
  if (p_data_xfr) {
    auto st = data_transfer_mgr_.RegisterDataTransfer(std::move(p_data_xfr));
    if (!st.IsOK()) {
      return st;
    }
  }

  p_exec_provider->SetLogger(session_logger_);
  return execution_providers_.Add(provider_type, std::move(p_exec_provider));
}

// Custom Op support
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
common::Status InferenceSession::AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& op_domains) {
  std::shared_ptr<CustomRegistry> custom_registry;
  ORT_RETURN_IF_ERROR_SESSIONID_(CreateCustomRegistry(op_domains, custom_registry));
  ORT_RETURN_IF_ERROR_SESSIONID_(RegisterCustomRegistry(custom_registry));
  return Status::OK();
}

common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  if (custom_registry == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for custom registry");
  }

  custom_registries_.push_back(custom_registry);

  // Insert session-level customized kernel registry.
  kernel_registry_manager_.RegisterKernelRegistry(custom_registry->GetKernelRegistry());

#if !defined(ORT_MINIMAL_BUILD)
  custom_schema_registries_.push_back(custom_registry->GetOpschemaRegistry());
#endif
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#if !defined(ORT_MINIMAL_BUILD)
common::Status InferenceSession::RegisterGraphTransformer(
    std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer, TransformerLevel level) {
  if (p_graph_transformer == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for graph transformer");
  }

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_inited_) {
    // adding a transformer now is pointless as the graph as already been transformed
    LOGS(*session_logger_, ERROR) << "Graph transformers must be registered before the session is initialized.";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Graph transformers must be registered before the session is initialized.");
  }

  return graph_transformation_mgr_.Register(std::move(p_graph_transformer), level);
}

common::Status InferenceSession::FilterEnabledOptimizers(const std::unordered_set<std::string>& optimizers_to_disable) {
  optimizers_to_disable_ = optimizers_to_disable;
  return Status::OK();
}

common::Status InferenceSession::SaveToOrtFormat(const std::basic_string<ORTCHAR_T>& filepath) const {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ort format only supports little-edian machines");

  // Get the byte size of the ModelProto and round it to the next MB and use it as flatbuffers' init_size
  // TODO: Investigate whether we should set a max size, and clarify the cost of having a buffer smaller than
  // what the total flatbuffers serialized size will be.
  constexpr size_t m_bytes = 1024 * 1024;
  size_t fbs_buffer_size = std::max(m_bytes, model_->ToProto().ByteSizeLong());
  fbs_buffer_size = ((fbs_buffer_size + m_bytes - 1) / m_bytes) * m_bytes;
  flatbuffers::FlatBufferBuilder builder(fbs_buffer_size);

  auto ort_model_version = builder.CreateString(kOrtModelVersion);
  flatbuffers::Offset<fbs::Model> model;
  ORT_RETURN_IF_ERROR(
      model_->SaveToOrtFormat(builder, model));

  flatbuffers::Offset<fbs::SessionState> session_state;
  ORT_RETURN_IF_ERROR(
      session_state_->SaveToOrtFormat(builder, session_state));

  fbs::InferenceSessionBuilder sb(builder);
  sb.add_ort_version(ort_model_version);
  sb.add_model(model);
  sb.add_session_state(session_state);
  auto session = sb.Finish();
  builder.Finish(session, fbs::InferenceSessionIdentifier());

  // TODO: Do we need to catch any std::exceptions from creating/writing to disk and convert to Status codes?
  {
    std::ofstream file(filepath, std::ios::binary);

    uint8_t* buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    file.write(reinterpret_cast<const char*>(buf), size);
    file.close();
  }

  return Status::OK();
}

common::Status InferenceSession::Load(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                                      const std::string& event_name) {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }
  ORT_TRY {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (is_model_loaded_) {  // already loaded
      LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
      return common::Status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session already contains a loaded model.");
    }

    std::shared_ptr<onnxruntime::Model> p_tmp_model;
    status = loader(p_tmp_model);
    ORT_RETURN_IF_ERROR_SESSIONID_(status);

    model_ = p_tmp_model;

    status = DoPostLoadProcessing(*model_);
    ORT_RETURN_IF_ERROR_SESSIONID_(status);

    // all steps complete, mark the model as loaded.
    is_model_loaded_ = true;

    telemetry_.event_name_ = event_name;
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(common::ONNXRUNTIME, common::FAIL, "Exception during loading: " + std::string(ex.what()));
    });
  }
  ORT_CATCH(...) {
    LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
    status = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, event_name, tp);
  }

  return status;
}

template <typename T>
common::Status InferenceSession::Load(const std::basic_string<T>& model_uri) {
  model_location_ = ToWideString(model_uri);
  auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_location_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif
    return onnxruntime::Model::Load(model_location_, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                    *session_logger_);
  };

  common::Status st = Load(loader, "model_loading_uri");
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "Load model from " << ToMBString(model_uri) << " failed:" << st.ErrorMessage();
    return common::Status(st.Category(), st.Code(), oss.str());
  }
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::Load(const std::string& model_uri) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type && experimental::utils::IsOrtFormatModel(model_uri))) {
#if defined(ENABLE_ORT_FORMAT_LOAD)
    return LoadOrtModel(model_uri);
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ORT format model is not supported in this build.");
#endif
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  return Load<char>(model_uri);
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#ifdef _WIN32
common::Status InferenceSession::Load(const std::wstring& model_uri) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type && experimental::utils::IsOrtFormatModel(model_uri))) {
#if defined(ENABLE_ORT_FORMAT_LOAD)
    return LoadOrtModel(model_uri);
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ORT format model is not supported in this build.");
#endif
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  return Load<PATH_CHAR_TYPE>(model_uri);
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}
#endif

common::Status InferenceSession::Load(const void* model_data, int model_data_len) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type &&
       experimental::utils::IsOrtFormatModelBytes(model_data, model_data_len))) {
#if defined(ENABLE_ORT_FORMAT_LOAD)
    return LoadOrtModel(model_data, model_data_len);
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ORT format model is not supported in this build.");
#endif
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, model_data, model_data_len](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;

    const bool result = model_proto.ParseFromArray(model_data, model_data_len);
    if (!result) {
      return Status(common::ONNXRUNTIME, common::INVALID_PROTOBUF,
                    "Failed to load model because protobuf parsing failed.");
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif

    return onnxruntime::Model::Load(std::move(model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_);
  };

  return Load(loader, "model_loading_array");
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#if !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::Load(const ModelProto& model_proto) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif
    // This call will create a copy of model_proto and the constructed model instance will own the copy thereafter
    return onnxruntime::Model::Load(model_proto, PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_);
  };

  return Load(loader, "model_loading_proto");
}

common::Status InferenceSession::Load(std::unique_ptr<ModelProto> p_model_proto) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &p_model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(*p_model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif
    return onnxruntime::Model::Load(std::move(*p_model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_);
  };

  return Load(loader, "model_loading_proto");
}

common::Status InferenceSession::Load(std::istream& model_istream) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_istream](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;
    Status st = Model::Load(model_istream, &model_proto);
    if (!st.IsOK()) {
      return st;
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif
    return onnxruntime::Model::Load(std::move(model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_);
  };

  return Load(loader, "model_loading_istream");
}

common::Status InferenceSession::Load() {
  if (!is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has not been parsed yet. "
                           "This API should be called in conjunction with a ctor that takes a model abstraction.");
  }

  auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(this->model_proto_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      ORT_RETURN_IF_ERROR(AddCustomOpDomains({domain.get()}));
    }
#endif
    // Pass on ownership of the parsed ModelProto to the Model instance (its job here is done by this stage)
    return Model::Load(std::move(this->model_proto_), model_location_, model,
                       HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_);
  };

  return Load(loader, "model_loading_from_saved_proto");
}

common::Status InferenceSession::TransformGraph(onnxruntime::Graph& graph,
                                                const onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                                const ExecutionProviders& providers,
                                                KernelRegistryManager& kernel_registry_manager,
                                                const InsertCastTransformer& insert_cast_transformer,
                                                SessionState& session_state,
                                                bool saving_model_in_ort_format) {
  // The transformer order:
  // 1. built-in graph rewriter
  // 2. each execution provider's transformer
  // 3. do node placement according to kernel definition
  // 4. insert copy nodes
  // 5. insert cast nodes.

  // first apply global(execution provider independent),  level 1(default/system/basic) graph to graph optimizations
  ORT_RETURN_IF_ERROR_SESSIONID_(
      graph_transformer_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *session_logger_));

#ifdef USE_DML
  // TODO: this is a temporary workaround to apply the DML EP's custom graph transformer prior to partitioning. This
  // transformer applies DML-specific fusions that go beyond what ORT offers by default. Ideally the DML EP should
  // apply these transforms during partitioning, but the full mutable Graph object isn't exposed to
  // IExecutionProvider::GetCapability, which is necessary for the DML EP's transforms.
  //
  // To prevent this from interfering with other EPs, we only apply this transform if the DML EP is the only one that's
  // registered (aside from the CPU EP, which is always registered by default.)
  if (execution_providers_.Get(kDmlExecutionProvider) && execution_providers_.NumProviders() <= 2) {
    Dml::GraphTransformer dml_transformer(onnxruntime::kDmlExecutionProvider,
                                          execution_providers_.Get(kDmlExecutionProvider));

    bool modified = false;
    dml_transformer.Apply(graph, modified, *session_logger_);
  }
#endif

  // if saving model to ORT format we only assign nodes a custom EP can handle and don't compile them.
  // we do this to preserve the original nodes in the model but prevent optimizers from changing them.
  // at runtime, the ORT format model will re-do the partitioning/compilation of these nodes, which may change
  // to cover fewer nodes due to device capabilities.
  auto mode = saving_model_in_ort_format ? GraphPartitioner::Mode::kAssignOnly
                                         : GraphPartitioner::Mode::kNormal;

  // Do partitioning based on execution providers' capability.
  GraphPartitioner partitioner(kernel_registry_manager, providers);
  ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.Partition(graph, session_state.ExportDll(),
                                                       session_state.GetMutableFuncMgr(), mode));

  // apply transformers except default transformers
  // Default transformers are required for correctness and they are owned and run by inference session
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR_SESSIONID_(
        graph_transformer_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *session_logger_));
  }

  bool modified = false;
  // Insert cast node/s.
  ORT_RETURN_IF_ERROR_SESSIONID_(insert_cast_transformer.Apply(graph, modified, *session_logger_));

  // Now every node should be already assigned to an execution provider
  std::unordered_map<std::string, std::vector<std::string>> node_placements;
  bool is_verbose_mode = session_logger_->GetSeverity() == logging::Severity::kVERBOSE;
  for (auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();
    if (node_provider.empty()) {
      std::ostringstream oss;
      oss << "Could not find an implementation for the node ";
      if (!node.Name().empty())
        oss << node.Name() << ":";
      oss << node.OpType() << "(" << node.SinceVersion() << ")";

      return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, oss.str());
    } else {
      if (is_verbose_mode) {  // TODO: should we disable this if the number of nodes are above a certain threshold?
        std::string node_str = node.OpType();
        node_str += " (";
        node_str += node.Name();
        node_str += ")";
        node_placements[node_provider].push_back(node_str);
      }
    }
  }

  // print placement info
  if (is_verbose_mode) {
    LOGS(*session_logger_, VERBOSE) << "Node placements";
    if (node_placements.size() == 1) {
      LOGS(*session_logger_, VERBOSE) << "All nodes have been placed on [" << node_placements.begin()->first << "].";
    } else {
      for (const auto& pr : node_placements) {
        std::ostringstream all_nodes_str;
        std::copy(pr.second.begin(), pr.second.end(), std::ostream_iterator<std::string>(all_nodes_str, ", "));
        LOGS(*session_logger_, VERBOSE) << " Provider: [" << pr.first << "]"
                                        << ": [" << all_nodes_str.str() << "]";
      }
    }
  }

  std::vector<std::string> provider_types;
  for (auto& provider_ptr : providers) {
    provider_types.push_back(provider_ptr->Type());
  }

  // Insert copy node/s.
  MemcpyTransformer copy_transformer{provider_types, kernel_registry_manager};
  ORT_RETURN_IF_ERROR_SESSIONID_(copy_transformer.Apply(graph, modified, *session_logger_));

  return common::Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Status InferenceSession::PartitionOrtFormatModel(onnxruntime::Graph& graph,
                                                 const ExecutionProviders& providers,
                                                 KernelRegistryManager& kernel_registry_manager,
                                                 SessionState& session_state) const {
  std::unordered_map<std::string, uint64_t> compiled_kernel_hashes;

  GraphPartitioner partitioner(kernel_registry_manager, providers);
  ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.Partition(graph, session_state.ExportDll(),
                                                       session_state.GetMutableFuncMgr(),
                                                       GraphPartitioner::Mode::kOrtFormatLoad,
                                                       &compiled_kernel_hashes));

#if defined(ENABLE_ORT_FORMAT_LOAD)
  if (!compiled_kernel_hashes.empty()) {
    session_state.SetCompiledKernelHashes(std::move(compiled_kernel_hashes));
  }
#endif

  return Status::OK();
}
#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)
template <typename T>
static Status LoadOrtModelBytes(const std::basic_string<T>& model_uri,
                                std::basic_string<ORTCHAR_T>& model_location,
                                std::vector<uint8_t>& bytes) {
  size_t num_bytes = 0;
  model_location = ToWideString(model_uri);
  ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(model_location.c_str(), num_bytes));

  bytes.resize(num_bytes);

  std::ifstream bytes_stream(model_uri, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(bytes.data()), num_bytes);

  if (!bytes_stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Load model from ", ToMBString(model_uri), " failed. Only ",
                           bytes_stream.gcount(), "/", num_bytes, " bytes were able to be read.");
  }

  return Status::OK();
}

Status InferenceSession::LoadOrtModel(const std::string& model_uri) {
  return LoadOrtModel(
      [&]() {
        ORT_RETURN_IF_ERROR(LoadOrtModelBytes(model_uri, model_location_, ort_format_model_bytes_));
        return Status::OK();
      });
}

#ifdef WIN32
Status InferenceSession::LoadOrtModel(const std::wstring& model_uri) {
  return LoadOrtModel(
      [&]() {
        ORT_RETURN_IF_ERROR(LoadOrtModelBytes(model_uri, model_location_, ort_format_model_bytes_));
        return Status::OK();
      });
}
#endif

Status InferenceSession::LoadOrtModel(const void* model_data, int model_data_len) {
  return LoadOrtModel([&]() {
    // copy bytes as we need them to be available when InferenceSession::Initialize is called later.
    //
    // TODO: Provide Load API where we can take ownership of memory to avoid the copy,
    // and/or a combined Load+Initialize where we don't need this temporary copy.
    ort_format_model_bytes_.resize(model_data_len);
    std::copy_n(reinterpret_cast<const uint8_t*>(model_data), model_data_len, ort_format_model_bytes_.data());

    return Status::OK();
  });
}

Status InferenceSession::LoadOrtModel(std::function<Status()> load_ort_format_model_bytes) {
  static_assert(FLATBUFFERS_LITTLEENDIAN, "ORT format only supports little-endian machines");

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_model_loaded_) {  // already loaded
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session already contains a loaded model.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  if (is_inited_) {
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session has already been initialized.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  ORT_RETURN_IF_ERROR(load_ort_format_model_bytes());

  // Verify the ort_format_model_bytes_ is a valid InferenceSessionBuffer before we access the data
  flatbuffers::Verifier verifier(ort_format_model_bytes_.data(), ort_format_model_bytes_.size());
  ORT_RETURN_IF_NOT(fbs::VerifyInferenceSessionBuffer(verifier), "ORT model verification failed.");

  const auto* fbs_session = fbs::GetInferenceSession(ort_format_model_bytes_.data());
  ORT_RETURN_IF(nullptr == fbs_session, "InferenceSession is null. Invalid ORT format model.");

  // Check version mismatch, for now we will only proceed when runtime version matches the model's ort version
  const auto* fbs_ort_model_version = fbs_session->ort_version();
  ORT_RETURN_IF(fbs_ort_model_version == nullptr, "Serialized version info is null. Invalid ORT format model.");
  ORT_RETURN_IF_NOT(IsOrtModelVersionSupported(fbs_ort_model_version->str()),
                    "The ORT format model version [", fbs_ort_model_version->str(),
                    "] is not supported this build ", ORT_VERSION);

  const auto* fbs_model = fbs_session->model();
  ORT_RETURN_IF(nullptr == fbs_model, "Missing Model. Invalid ORT format model.");

  // need to go from unique_ptr to shared_ptr when moving into model_
  std::unique_ptr<Model> tmp_model;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model,
                                               HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                               *session_logger_, tmp_model));

#else
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model, *session_logger_, tmp_model));
#endif

  ORT_RETURN_IF_ERROR(SaveModelMetadata(*tmp_model));
  model_ = std::move(tmp_model);

  // Initialize takes the session_mutex_ as well so we need to have released it prior to calling this
  const auto* fbs_sess_state = fbs_session->session_state();
  ORT_RETURN_IF(nullptr == fbs_sess_state, "SessionState is null. Invalid ORT format model.");

  is_model_loaded_ = true;

  return Status::OK();
}
#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

bool InferenceSession::IsInitialized() const {
  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
  return is_inited_;
}

static bool ModelHasFP16InputsHelper(const onnx::TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case ::onnx::TypeProto::ValueCase::kTensorType: {
      if (type_proto.has_tensor_type()) {
        auto& tensor_type = type_proto.tensor_type();
        if (tensor_type.elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
          return true;
        }
      }
      break;
    }
    case ::onnx::TypeProto::ValueCase::kSequenceType: {
      if (type_proto.has_sequence_type()) {
        auto& sequence_type = type_proto.sequence_type();
        return ModelHasFP16InputsHelper(sequence_type.elem_type());
      }
      break;
    }
    case ::onnx::TypeProto::ValueCase::kMapType: {
      if (type_proto.has_map_type()) {
        auto& map_type = type_proto.map_type();
        return ModelHasFP16InputsHelper(map_type.value_type());
      }
      break;
    }
    default:
      break;
  }
  return false;
}

static bool ModelHasFP16Inputs(const Graph& graph) {
  for (auto& input : graph.GetInputs()) {
    if (input->Exists() && ModelHasFP16InputsHelper(*(input->TypeAsProto()))) {
      return true;
    }
  }
  return false;
}

common::Status InferenceSession::AddPrePackedWeightsContainer(PrepackedWeightsContainer* prepacked_weights_container) {
  if (prepacked_weights_container == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The provided PrePackedWeightsContainer instance to be added to the session is null");
  }

  if (prepacked_weights_container_ != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The session already has a PrePackedWeightsContainer instance");
  }

  prepacked_weights_container_ = prepacked_weights_container;

  return Status::OK();
}

common::Status InferenceSession::Initialize() {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }

  ORT_TRY {
    LOGS(*session_logger_, INFO) << "Initializing session.";
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogSessionCreationStart();

    bool have_cpu_ep = false;

    {
      std::lock_guard<onnxruntime::OrtMutex> initial_guard(session_mutex_);

      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded.");
      }

      if (is_inited_) {  // already initialized
        LOGS(*session_logger_, INFO) << "Session has already been initialized.";
        return common::Status::OK();
      }

      have_cpu_ep = execution_providers_.Get(onnxruntime::kCpuExecutionProvider) != nullptr;
    }

    // Verify that there are no external initializers in the graph if external data is disabled.
    onnxruntime::Graph& graph = model_->MainGraph();
#ifdef DISABLE_EXTERNAL_INITIALIZERS
    const InitializedTensorSet& initializers = graph.GetAllInitializedTensors();
    for (const auto& it: initializers) {
      if (utils::HasExternalData(*it.second)) {
        return common::Status(common::ONNXRUNTIME, common::FAIL,
                  "Initializer tensors with external data is not allowed.");
      }
    }
#endif

    // Register default CPUExecutionProvider if user didn't provide it through the Register() calls.
    // RegisterExecutionProvider locks the session_mutex_ so we can't be holding it when we call that
    if (!have_cpu_ep) {
      LOGS(*session_logger_, INFO) << "Adding default CPU execution provider.";
      CPUExecutionProviderInfo epi{session_options_.enable_cpu_mem_arena};
      auto p_cpu_exec_provider = std::make_unique<CPUExecutionProvider>(epi);
      ORT_RETURN_IF_ERROR_SESSIONID_(RegisterExecutionProvider(std::move(p_cpu_exec_provider)));
    }

    // re-acquire mutex
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

    // At this time we know all the providers that will be part of this session.
    // Read shared allocators from the environment and update them in the respective providers.
    //
    // The reason for updating the providers is so that when the session state is created the allocators
    // are setup appropriately keyed by OrtMemoryInfo with delegates going to the respective providers.
    // Secondly, the GetAllocator() method inside IExecutionProvider is still used in various places, hence
    // it doesn't make sense to just update the allocator map inside session state with these shared allocators; doing
    // so would cause inconsistency between the allocator map inside session sate and that inside the providers.
    // TODO: we could refactor the allocators to not require the call to GetAllocator but that change is much bigger
    // since we've to take into account the per-thread cuda allocators.
    // TODO (contd.) We could also possibly absorb the per-thread logic in a new allocator decorator that derives
    // from IAllocator to keep things clean.
    std::string use_env_allocators = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseEnvAllocators,
                                                                                        "0");
    if (use_env_allocators == "1") {
      LOGS(*session_logger_, INFO) << "This session will use the allocator registered with the environment.";
      UpdateProvidersWithSharedAllocators();
    }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    TraceLoggingWriteStart(session_activity, "OrtInferenceSessionActivity");
    session_activity_started_ = true;
#endif

    // now that we have all the execution providers, create the session state
    session_state_ = std::make_unique<SessionState>(
        model_->MainGraph(),
        execution_providers_,
        session_options_.enable_mem_pattern && session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL,
        GetIntraOpThreadPoolToUse(),
        GetInterOpThreadPoolToUse(),
        data_transfer_mgr_,
        *session_logger_,
        session_profiler_,
        session_options_.use_deterministic_compute,
        session_options_.enable_mem_reuse,
        prepacked_weights_container_);

    // Collect the kernel registries from execution provider instances;
    // There are 2 kinds of kernel registries with priority from high to low as below,
    // 1. Custom execution provider type specific kernel registries.
    // 2. common execution provider type specific kernel registries.
    // Kernel registries are shared across sessions.
    // The 1st ones should have already been registered via session-level API into KernelRegistryManager.
    //
    // Register 2nd registries into KernelRegistryManager.
    ORT_RETURN_IF_ERROR_SESSIONID_(kernel_registry_manager_.RegisterKernels(execution_providers_));

    bool loading_ort_format = !ort_format_model_bytes_.empty();
    bool saving_model = !session_options_.optimized_model_filepath.empty();
    bool saving_ort_format = false;
    if (saving_model) {
      std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSaveModelFormat, "");
      bool has_explicit_type = !model_type.empty();
      saving_ort_format = ((has_explicit_type && model_type == "ORT") ||
                           (!has_explicit_type &&
                            experimental::utils::IsOrtFormatModel(session_options_.optimized_model_filepath)));
    }

#if !defined(ORT_MINIMAL_BUILD)
    if (!loading_ort_format) {
      // add predefined transformers
      AddPredefinedTransformers(graph_transformation_mgr_, session_options_.graph_optimization_level);

      // apply any transformations to the main graph and any subgraphs
      ORT_RETURN_IF_ERROR_SESSIONID_(TransformGraph(graph, graph_transformation_mgr_,
                                                    execution_providers_, kernel_registry_manager_,
                                                    insert_cast_transformer_,
                                                    *session_state_,
                                                    saving_ort_format));

      // now that all the transforms are done, call Resolve on the main graph. this will recurse into the subgraphs.
      ORT_RETURN_IF_ERROR_SESSIONID_(graph.Resolve());

      // Update temporary copies of metadata, input- and output definitions to the same state as the resolved graph
      ORT_RETURN_IF_ERROR_SESSIONID_(SaveModelMetadata(*model_));
    } else
#endif  // !defined(ORT_MINIMAL_BUILD)
    {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      // nodes are already partitioned, but a custom EP may compile some at runtime.
      // run the partitioning to allow that to happen.
      //
      // We always have the CPU EP, so only need to run this if some other EP is enabled
      if (execution_providers_.NumProviders() > 1) {
        ORT_RETURN_IF_ERROR_SESSIONID_(PartitionOrtFormatModel(graph, execution_providers_, kernel_registry_manager_,
                                                               *session_state_));
      }
#endif
    }

    const experimental::fbs::SessionState* serialized_session_state =
        loading_ort_format
            ? fbs::GetInferenceSession(ort_format_model_bytes_.data())->session_state()
            : nullptr;

    ORT_RETURN_IF_ERROR_SESSIONID_(
        session_state_->FinalizeSessionState(model_location_, kernel_registry_manager_,
                                             session_options_,
                                             serialized_session_state,
                                             // need to keep the initializers if saving the optimized model
                                             !saving_model,
                                             saving_ort_format));

#if !defined(ORT_MINIMAL_BUILD)
    if (saving_model) {
      if (session_state_->GetFuncMgr().NumFuncs() > 0) {
        ORT_RETURN_IF_ERROR_SESSIONID_(
            ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                            "Unable to serialize model as it contains compiled nodes. "
                            "Please disable any execution providers which generate compiled nodes."));
      }

      // add a warning if the NchwcTransformer was enabled, as it contains the hardware specific logic
      if (session_options_.graph_optimization_level >= TransformerLevel::Level3 &&
          optimizers_to_disable_.find("NchwcTransformer") == optimizers_to_disable_.cend()) {
        LOGS(*session_logger_, WARNING)
            << "Serializing optimized model with Graph Optimization level greater than ORT_ENABLE_EXTENDED and the "
               "NchwcTransformer enabled. The generated model may contain hardware specific optimizations, and "
               "should only be used in the same environment the model was optimized in.";
      }

      if (saving_ort_format) {
        ORT_RETURN_IF_ERROR_SESSIONID_(SaveToOrtFormat(session_options_.optimized_model_filepath));
      } else {
        ORT_RETURN_IF_ERROR_SESSIONID_(Model::Save(*model_, session_options_.optimized_model_filepath));
      }
    }
#endif  // !defined(ORT_MINIMAL_BUILD)

    session_state_->ResolveMemoryPatternFlag();
    is_inited_ = true;

    // we don't directly use the ORT format bytes currently, so free those now
    std::vector<uint8_t>().swap(ort_format_model_bytes_);

    // and log telemetry
    bool model_has_fp16_inputs = ModelHasFP16Inputs(graph);
    env.GetTelemetryProvider().LogSessionCreation(
        session_id_, model_->IrVersion(), model_->ProducerName(), model_->ProducerVersion(), model_->Domain(),
        model_->MainGraph().DomainToVersionMap(), model_->MainGraph().Name(), model_->MetaData(),
        telemetry_.event_name_, execution_providers_.GetIds(), model_has_fp16_inputs);
    LOGS(*session_logger_, INFO) << "Session successfully initialized.";
  }
  ORT_CATCH(const NotImplementedException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    });
  }
  ORT_CATCH(...) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Encountered unknown exception in Initialize()");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "session_initialization", tp);
  }

  if (status.IsOK()) {
    for (auto& xp : execution_providers_) {
      auto end_status = xp->OnSessionInitializationEnd();
      if (status.IsOK()) {
        status = end_status;
      }
    }
  }

  return status;
}

// This method should be called from within Initialize() only and before the creation of the session state.
// This ensures all providers have been registered in the session and the session state is consistent with the providers.
void InferenceSession::UpdateProvidersWithSharedAllocators() {
  using namespace std;
  const auto& provider_ids = execution_providers_.GetIds();
  for (const auto& one_shared_alloc : environment_.GetRegisteredSharedAllocators()) {
    for (const auto& id : provider_ids) {
      auto* provider_ptr = execution_providers_.Get(id);
      provider_ptr->ReplaceAllocator(one_shared_alloc);
    }
  }
}

int InferenceSession::GetCurrentNumRuns() const {
  return current_num_runs_.load();
}

const std::vector<std::string>& InferenceSession::GetRegisteredProviderTypes() const {
  return execution_providers_.GetIds();
}

const ProviderOptionsMap& InferenceSession::GetAllProviderOptions() const {
  return execution_providers_.GetAllProviderOptions();
}

const SessionOptions& InferenceSession::GetSessionOptions() const {
  return session_options_;
}

const DataTransferManager& InferenceSession::GetDataTransferManager() const {
  return data_transfer_mgr_;
}

common::Status InferenceSession::CheckShapes(const std::string& input_name, const TensorShape& input_shape,
                                             const TensorShape& expected_shape) const {
  auto input_shape_sz = input_shape.NumDimensions();
  auto expected_shape_sz = expected_shape.NumDimensions();
  if (input_shape_sz != expected_shape_sz) {
    std::ostringstream ostr;
    ostr << "Invalid rank for input: " << input_name << " Got: " << input_shape_sz << " Expected: " << expected_shape_sz
         << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }

  std::vector<size_t> invalid_dim_indices;
  for (size_t i = 0; i < input_shape_sz; ++i) {
    if (expected_shape[i] < 0) {
      continue;  // this represents a symbolic shape dimension
    }
    if (input_shape[i] != expected_shape[i]) {
      invalid_dim_indices.push_back(i);
    }
  }

  if (!invalid_dim_indices.empty()) {
    std::ostringstream ostr;
    ostr << "Got invalid dimensions for input: " << input_name << " for the following indices\n";
    for (size_t i = 0, end = invalid_dim_indices.size(); i < end; ++i) {
      size_t idx = invalid_dim_indices[i];
      ostr << " index: " << idx << " Got: " << input_shape[idx] << " Expected: " << expected_shape[idx] << "\n";
    }
    ostr << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }
  return Status::OK();
}

static common::Status CheckTypes(MLDataType actual, MLDataType expected, const std::string& base_type) {
  if (actual == expected) {
    return Status::OK();
  }
  std::ostringstream ostr;
  ostr << "Unexpected input data type. Actual: (";
  ostr << base_type;
  ostr << "(";
  ostr << DataTypeImpl::ToString(actual);
  ostr << ")) , expected: (";
  ostr << base_type;
  ostr << "(";
  ostr << DataTypeImpl::ToString(expected);
  ostr << "))";

  return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
}

common::Status InferenceSession::ValidateInputs(const std::vector<std::string>& feed_names,
                                                const std::vector<OrtValue>& feeds) const {
  if (feed_names.size() != feeds.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Size mismatch: feed_names has ", feed_names.size(),
                           "elements, but feeds has ", feeds.size(), " elements.");
  }

  for (size_t i = 0; i < feeds.size(); ++i) {
    const auto& feed_name = feed_names[i];

    auto iter = input_def_map_.find(feed_name);
    if (input_def_map_.end() == iter) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Feed Input Name:", feed_name);
    }

    auto expected_type = iter->second.ml_data_type;
    auto& input_ml_value = feeds.at(i);
    if (input_ml_value.IsTensor()) {
      // check for type
      if (!expected_type->IsTensorType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type tensor.");
      }
      auto expected_element_type = expected_type->AsTensorType()->GetElementType();
      auto input_element_type = input_ml_value.Get<Tensor>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "tensor"));

      // check for shape
      const auto& expected_shape = iter->second.tensor_shape;
      if (expected_shape.NumDimensions() > 0) {
        const auto& input_shape = input_ml_value.Get<Tensor>().Shape();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(feed_name, input_shape, expected_shape));
      }
    } else if (input_ml_value.IsSparseTensor()) {
      if (!expected_type->IsSparseTensorType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type sparse tensor.");
      }
      auto expected_element_type = expected_type->AsSparseTensorType()->GetElementType();
      const SparseTensor& sparse_tensor = input_ml_value.Get<SparseTensor>();
      auto input_element_type = sparse_tensor.Values().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "sparse_tensor"));
      // Check shape
      const auto& expected_shape = iter->second.tensor_shape;
      if (expected_shape.NumDimensions() > 0) {
        const auto& input_shape = sparse_tensor.Shape();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(feed_name, input_shape, expected_shape));
      }
    } else if (input_ml_value.IsTensorSequence()) {
      if (!expected_type->IsTensorSequenceType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type tensor sequence.");
      }
      auto expected_element_type = expected_type->AsSequenceTensorBase()->GetElementType();
      auto input_element_type = input_ml_value.Get<TensorSeq>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "seq"));
    } else {
      auto input_type = input_ml_value.Type();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_type, expected_type, ""));
    }
  }

  return Status::OK();
}

common::Status InferenceSession::ValidateOutputs(const std::vector<std::string>& output_names,
                                                 const std::vector<OrtValue>* p_fetches) const {
  if (p_fetches == nullptr) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Output vector pointer is NULL");
  }

  if (output_names.empty()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "At least one output should be requested.");
  }

  if (!p_fetches->empty() && (output_names.size() != p_fetches->size())) {
    std::ostringstream ostr;
    ostr << "Output vector incorrectly sized: output_names.size(): " << output_names.size()
         << "p_fetches->size(): " << p_fetches->size();
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
  }

  for (const auto& name : output_names) {
    if (model_output_names_.find(name) == model_output_names_.end()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid Output Name:" + name);
    }
  }

  // TODO add more validation here like checking shape of the allocated buffers

  return common::Status::OK();
}

#ifdef ENABLE_TRAINING
Status InferenceSession::PartialRun(onnxruntime::RunOptions& run_options,
                                    const std::vector<OrtValue>& feeds,
                                    std::vector<OrtValue>& fetches,
                                    PartialGraphExecutionState& state,
                                    FeedsFetchesManager& feeds_fetches_manager) {
  Status retval = Status::OK();
  std::vector<IExecutionProvider*> exec_providers_to_stop;
  exec_providers_to_stop.reserve(execution_providers_.NumProviders());

  ORT_TRY {
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }

    if (!run_options.run_tag.empty()) {
      LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
    }

    // scope of owned_run_logger is just the call to Execute.
    // If Execute ever becomes async we need a different approach
    std::unique_ptr<logging::Logger> owned_run_logger;
    auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

    // info all execution providers InferenceSession:Run started
    // TODO: only call OnRunStart for all providers in-use
    for (auto& xp : execution_providers_) {
      // call OnRunStart and add to exec_providers_to_stop if successful
      auto start_func = [&xp, &exec_providers_to_stop]() {
        auto status = xp->OnRunStart();
        if (status.IsOK())
          exec_providers_to_stop.push_back(xp.get());

        return status;
      };

      ORT_CHECK_AND_SET_RETVAL(start_func());
    }

    ORT_ENFORCE(run_options.only_execute_path_to_fetches == false, "only_execute_path_to_fetches is not supported.");

    ORT_ENFORCE(session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL, "Only sequential mode is supported.");

    // execute the graph
    ORT_CHECK_AND_SET_RETVAL(utils::ExecutePartialGraph(*session_state_, feeds_fetches_manager, feeds, fetches,
                                                        run_logger, state));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
    });
  }
  ORT_CATCH(...) {
    retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
  }

  // info all execution providers InferenceSession:Run ended
  for (auto* xp : exec_providers_to_stop) {
    auto status = xp->OnRunEnd();
    ORT_CHECK_AND_SET_RETVAL(status);
  }

  return retval;
}
#endif

Status InferenceSession::Run(const RunOptions& run_options,
                             const std::vector<std::string>& feed_names, const std::vector<OrtValue>& feeds,
                             const std::vector<std::string>& output_names, std::vector<OrtValue>* p_fetches,
                             const std::vector<OrtDevice>* p_fetches_device_info) {
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingActivity<telemetry_provider_handle> ortrun_activity;
  ortrun_activity.SetRelatedActivity(session_activity);
  TraceLoggingWriteStart(ortrun_activity, "OrtRun");
#endif
  Status retval = Status::OK();
  const Env& env = Env::Default();

  std::vector<IExecutionProvider*> exec_providers_to_stop;
  exec_providers_to_stop.reserve(execution_providers_.NumProviders());

  std::vector<AllocatorPtr> arenas_to_shrink;

  ORT_TRY {
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }

    // log evaluation start to trace logging provider
    env.GetTelemetryProvider().LogEvaluationStart();

    ORT_RETURN_IF_ERROR_SESSIONID_(ValidateInputs(feed_names, feeds));
    ORT_RETURN_IF_ERROR_SESSIONID_(ValidateOutputs(output_names, p_fetches));

    // shrink certain default memory arenas if the user has requested for it
    const std::string& shrink_memory_arenas =
        run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "");

    if (!shrink_memory_arenas.empty()) {
      ORT_RETURN_IF_ERROR_SESSIONID_(ValidateAndParseShrinkArenaString(shrink_memory_arenas, arenas_to_shrink));
    }

    FeedsFetchesInfo info(feed_names, output_names, session_state_->GetOrtValueNameIdxMap());
    FeedsFetchesManager feeds_fetches_manager{std::move(info)};

    if (p_fetches_device_info) {
      // populate the target device info. ignored if pre-allocated fetches are provided
      const auto& fetch_device_info = *p_fetches_device_info;
      auto& fetch_info = feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo();

      for (size_t i = 0, end = output_names.size(); i < end; ++i) {
        fetch_info[i].target_device = fetch_device_info[i];
      }
    }

    if (!run_options.run_tag.empty()) {
      LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
    }

    ++current_num_runs_;

    // scope of owned_run_logger is just the call to Execute.
    // If Execute ever becomes async we need a different approach
    std::unique_ptr<logging::Logger> owned_run_logger;
    auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

    // info all execution providers InferenceSession:Run started
    // TODO: only call OnRunStart for all providers in-use
    for (auto& xp : execution_providers_) {
      // call OnRunStart and add to exec_providers_to_stop if successful
      auto start_func = [&xp, &exec_providers_to_stop]() {
        auto status = xp->OnRunStart();
        if (status.IsOK())
          exec_providers_to_stop.push_back(xp.get());

        return status;
      };

      ORT_CHECK_AND_SET_RETVAL(start_func());
    }

#if !defined(ORT_MINIMAL_BUILD)
    if (run_options.only_execute_path_to_fetches) {
      session_state_->UpdateToBeExecutedNodes(feeds_fetches_manager.GetFeedsFetchesInfo().fetches_mlvalue_idxs);
    }
#endif

    // execute the graph
    ORT_CHECK_AND_SET_RETVAL(utils::ExecuteGraph(*session_state_, feeds_fetches_manager, feeds, *p_fetches,
                                                 session_options_.execution_mode, run_options.terminate, run_logger,
                                                 run_options.only_execute_path_to_fetches));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
    });
  }
  ORT_CATCH(...) {
    retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
  }

  // info all execution providers InferenceSession:Run ended
  for (auto* xp : exec_providers_to_stop) {
    auto status = xp->OnRunEnd();
    ORT_CHECK_AND_SET_RETVAL(status);
  }

  if (!arenas_to_shrink.empty()) {
    ShrinkMemoryArenas(arenas_to_shrink);
  }

  --current_num_runs_;

  // keep track of telemetry
  ++telemetry_.total_runs_since_last_;
  telemetry_.total_run_duration_since_last_ += TimeDiffMicroSeconds(tp);

  // time to send telemetry?
  if (TimeDiffMicroSeconds(telemetry_.time_sent_last_) > telemetry_.kDurationBetweenSending) {
    // send the telemetry
    env.GetTelemetryProvider().LogRuntimePerf(session_id_, telemetry_.total_runs_since_last_,
                                              telemetry_.total_run_duration_since_last_);
    // reset counters
    telemetry_.time_sent_last_ = std::chrono::high_resolution_clock::now();
    telemetry_.total_runs_since_last_ = 0;
    telemetry_.total_run_duration_since_last_ = 0;
  }

  // log evaluation stop to trace logging provider
  env.GetTelemetryProvider().LogEvaluationStop();

  // send out profiling events (optional)
  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingWriteStop(ortrun_activity, "OrtRun");
#endif
  return retval;
}

common::Status InferenceSession::Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names,
                                     std::vector<OrtValue>* p_fetches) {
  return Run(RunOptions(), feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options, const NameMLValMap& feeds_map,
                                     const std::vector<std::string>& output_names, std::vector<OrtValue>* p_fetches) {
  std::vector<std::string> feed_names;
  std::vector<OrtValue> feeds;

  auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches, nullptr);
}

std::pair<common::Status, const ModelMetadata*> InferenceSession::GetModelMetadata() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &model_metadata_);
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetModelInputs() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  // return required inputs (excludes any inputs used for overriding initializers)
  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetInputs());
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetOverridableInitializers() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  // returns a list of initializers that can be overriden.
  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetOverridableInitializers());
}

std::pair<common::Status, const OutputDefList*> InferenceSession::GetModelOutputs() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &output_def_list_);
}

common::Status InferenceSession::NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }
  }

  // private constructor, can't use make_unique
  *io_binding = std::unique_ptr<IOBinding>(new IOBinding(*session_state_));
  return Status::OK();
}

common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  // TODO should Run() call io_binding.SynchronizeInputs() or should it let the callers do it?
  // io_binding.SynchronizeInputs();
  return Run(run_options, io_binding.GetInputNames(), io_binding.GetInputs(), io_binding.GetOutputNames(),
             &io_binding.GetOutputs(), &io_binding.GetOutputsDeviceInfo());
}

common::Status InferenceSession::Run(IOBinding& io_binding) {
  RunOptions run_options;
  return Run(run_options, io_binding);
}

template <typename T>
void InferenceSession::StartProfiling(const std::basic_string<T>& file_prefix) {
  std::basic_ostringstream<T> ss;
  ss << file_prefix << "_" << GetCurrentTimeString<T>() << ".json";
  session_profiler_.StartProfiling(ss.str());
}

void InferenceSession::StartProfiling(const std::string& file_prefix) {
  StartProfiling<char>(file_prefix);
}

#ifdef _WIN32
void InferenceSession::StartProfiling(const std::wstring& file_prefix) {
  StartProfiling<PATH_CHAR_TYPE>(file_prefix);
}
#endif

void InferenceSession::StartProfiling(const logging::Logger* logger_ptr) {
  session_profiler_.StartProfiling(logger_ptr);
}

std::string InferenceSession::EndProfiling() {
  if (is_model_loaded_) {
    if (session_profiler_.IsEnabled()) {
      return session_profiler_.EndProfiling();
    } else {
      LOGS(*session_logger_, VERBOSE) << "Profiler is disabled.";
      return std::string();
    }
  }
  LOGS(*session_logger_, ERROR) << "Could not write a profile because no model was loaded.";
  return std::string();
}

const profiling::Profiler& InferenceSession::GetProfiling() const {
  return session_profiler_;
}

AllocatorPtr InferenceSession::GetAllocator(const OrtMemoryInfo& mem_info) const {
  return session_state_->GetAllocator(mem_info);
}

common::Status InferenceSession::ValidateAndParseShrinkArenaString(const std::string& ort_device_list,
                                                                   /*out*/ std::vector<AllocatorPtr>& arenas_to_shrink) const {
  arenas_to_shrink.reserve(5);  // Allocate some memory for the container (we are unlikely to see more than 5 memory arena shrink requests)

  std::istringstream ss_1(ort_device_list);
  std::string device_id_pair;

  // Process all device-id pair(s)
  while (std::getline(ss_1, device_id_pair, ';')) {
    std::istringstream ss_2(device_id_pair);
    std::string device_id_component;

    // default values
    OrtDevice::DeviceType device_type = -1;
    OrtDevice::MemoryType memory_type = OrtDevice::MemType::DEFAULT;
    OrtDevice::DeviceId device_id = 0;

    int iter = 0;
    // Process this device-id pair
    while (std::getline(ss_2, device_id_component, ':')) {
      if (iter == 0) {  // this component corresponds to device
        if (device_id_component == "cpu") {
          device_type = OrtDevice::CPU;
        } else if (device_id_component == "gpu") {
          device_type = OrtDevice::GPU;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported device specified in the memory arena shrink list: ",
                                 device_id_component);
        }
      } else if (iter == 1) {  // This component corresponds to device id
        if (!TryParseStringWithClassicLocale<OrtDevice::DeviceId>(device_id_component, device_id)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported device id in the memory arena shrink list: ",
                                 device_id_component);
        }
      }

      ++iter;
    }

    // Shrink if it is an arena based allocator
    auto alloc = session_state_->GetAllocator(OrtDevice(device_type, memory_type, device_id));

    if (alloc == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Did not find an arena based allocator registered for device-id ",
                             " combination in the memory arena shrink list: ", device_id_pair);
    }

    if (alloc->Info().alloc_type != OrtAllocatorType::OrtArenaAllocator) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The registered allocator for device-id ",
                             " combination is not an arena based allocator: ", device_id_pair);
    }

    arenas_to_shrink.push_back(std::move(alloc));
  }

  return Status::OK();
}

void InferenceSession::ShrinkMemoryArenas(const std::vector<AllocatorPtr>& arenas_to_shrink) {
  for (auto& alloc : arenas_to_shrink) {
    auto status = static_cast<IArenaAllocator*>(alloc.get())->Shrink();

    if (!status.IsOK()) {
      LOGS(*session_logger_, WARNING) << "Unable to shrink arena: " << alloc->Info().ToString()
                                      << " error message: " << status.ErrorMessage();
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)
// assumes model has already been loaded before
common::Status InferenceSession::DoPostLoadProcessing(onnxruntime::Model& model) {
  // TODO add other post load processing here
  common::Status status = SaveModelMetadata(model);
  return status;
}
#endif

common::Status InferenceSession::SaveModelMetadata(const onnxruntime::Model& model) {
  VLOGS(*session_logger_, 1) << "Saving model metadata";
  const onnxruntime::Graph& graph = model.MainGraph();

  // save model metadata
  model_metadata_.producer_name = model.ProducerName();
  model_metadata_.description = model.DocString();
  model_metadata_.graph_description = model.GraphDocString();
  model_metadata_.domain = model.Domain();
  model_metadata_.version = model.ModelVersion();
  model_metadata_.custom_metadata_map = model.MetaData();
  model_metadata_.graph_name = graph.Name();

  required_inputs_.clear();
  for (auto input : graph.GetInputs()) {
    required_inputs_.insert(input->Name());
  }

  auto add_inputs = [this](const InputDefList& inputs) {
    input_def_map_.clear();
    input_def_map_.reserve(inputs.size());
    for (auto elem : inputs) {
      auto elem_type = utils::GetMLDataType(*elem);
      auto elem_shape_proto = elem->Shape();
      input_def_map_.insert(
          {elem->Name(),
           InputDefMetaData(
               elem, elem_type,
               elem_shape_proto ? utils::GetTensorShapeFromTensorShapeProto(*elem_shape_proto) : TensorShape())});
    }
  };

  if (graph.CanOverrideInitializer()) {
    // for IR 4 or higher it is optional to have a matching graph input for an initializer, and if one exists the
    // initializer is explicitly overridable.
    add_inputs(graph.GetInputsIncludingInitializers());
  } else {
    // for IR < 4 we don't allow overriding initializers so that they can be treated as constant. exclude them from
    // the list of valid inputs by just using the GetInputs() list.
    add_inputs(graph.GetInputs());
  }

  // save outputs
  const auto& outputs = graph.GetOutputs();
  output_def_list_ = outputs;  // A direct copy of outputs

  model_output_names_.clear();
  model_output_names_.reserve(outputs.size());
  for (const auto& elem : outputs) {
    model_output_names_.insert(elem->Name());
  }

  VLOGS(*session_logger_, 1) << "Done saving model metadata";
  return common::Status::OK();
}

// Create a Logger for a single execution if possible. Otherwise use the default logger.
// If a new logger is created, it will also be stored in new_run_logger,
// which must remain valid for the duration of the execution.
// If the default logger is used, new_run_logger will remain empty.
// The returned value should be used in the execution.
const logging::Logger& InferenceSession::CreateLoggerForRun(const RunOptions& run_options,
                                                            std::unique_ptr<logging::Logger>& new_run_logger) {
  const logging::Logger* run_logger;

  // create a per-run logger if we can
  if (logging_manager_ != nullptr) {
    std::string run_log_id{session_options_.session_logid};

    if (!session_options_.session_logid.empty() && !run_options.run_tag.empty()) {
      run_log_id += ":";
    }

    run_log_id += run_options.run_tag;

    logging::Severity severity = logging::Severity::kWARNING;
    if (run_options.run_log_severity_level == -1) {
      severity = session_logger_->GetSeverity();
    } else {
      ORT_ENFORCE(run_options.run_log_severity_level >= 0 &&
                      run_options.run_log_severity_level <= static_cast<int>(logging::Severity::kFATAL),
                  "Invalid run log severity level. Not a valid onnxruntime::logging::Severity value: ",
                  run_options.run_log_severity_level);
      severity = static_cast<logging::Severity>(run_options.run_log_severity_level);
    }

    new_run_logger = logging_manager_->CreateLogger(run_log_id, severity, false, run_options.run_log_verbosity_level);

    run_logger = new_run_logger.get();
    VLOGS(*run_logger, 1) << "Created logger for run with id of " << run_log_id;
  } else {
    // fallback to using default logger. this does NOT have any session or run specific id/tag in it
    run_logger = session_logger_;
    VLOGS(*run_logger, 1) << "Using default logger for run " << run_options.run_tag;
  }

  return *run_logger;
}

void InferenceSession::InitLogger(logging::LoggingManager* logging_manager) {
  // create logger for session, using provided logging manager if possible
  if (logging_manager != nullptr) {
    logging::Severity severity = logging::Severity::kWARNING;
    if (session_options_.session_log_severity_level == -1) {
      severity = logging::LoggingManager::DefaultLogger().GetSeverity();
    } else {
      ORT_ENFORCE(session_options_.session_log_severity_level >= 0 &&
                      session_options_.session_log_severity_level <= static_cast<int>(logging::Severity::kFATAL),
                  "Invalid session log severity level. Not a valid onnxruntime::logging::Severity value: ",
                  session_options_.session_log_severity_level);
      severity = static_cast<logging::Severity>(session_options_.session_log_severity_level);
    }

    owned_session_logger_ = logging_manager_->CreateLogger(session_options_.session_logid, severity, false,
                                                           session_options_.session_log_verbosity_level);
    session_logger_ = owned_session_logger_.get();
  } else {
    session_logger_ = &logging::LoggingManager::DefaultLogger();
  }
}

#if !defined(ORT_MINIMAL_BUILD)

// Registers all the predefined transformers with transformer manager
void InferenceSession::AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                                 TransformerLevel graph_optimization_level) {
  const auto& cpu_ep = *execution_providers_.Get(onnxruntime::kCpuExecutionProvider);
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (graph_optimization_level >= level) {
      // Generate and register transformers for level
      auto transformers_to_register = optimizer_utils::GenerateTransformers(level, session_options_, cpu_ep,
                                                                            optimizers_to_disable_);
      for (auto& entry : transformers_to_register) {
        transformer_manager.Register(std::move(entry), level);
      }
    }
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) {
  if (timeout_in_ms > 0) {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
  }
  p_executor_done->Wait();

  return Status::OK();
}

SessionIOBinding::SessionIOBinding(InferenceSession* session) : sess_(session) {
  ORT_ENFORCE(session->NewIOBinding(&binding_).IsOK());
}

const InferenceSession* SessionIOBinding::GetInferenceSession() const {
  return sess_;
}

InferenceSession* SessionIOBinding::GetInferenceSession() {
  return sess_;
}

const IOBinding* SessionIOBinding::Get() const {
  return binding_.get();
}

IOBinding* SessionIOBinding::Get() {
  return binding_.get();
}

}  // namespace onnxruntime
