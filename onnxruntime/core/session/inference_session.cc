// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <charconv>
#include <memory>
#include <sstream>
#include <list>
#include <string>
#include <unordered_map>
#include <thread>
#include <queue>
#include <iomanip>
#include <gsl/gsl>

#include "core/common/denormal.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/json_utils.h"
#endif
#include "core/common/parse_string.h"
#include "core/common/path_string.h"
#include "core/common/string_utils.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/ort_format_version.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/ep_context_utils.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_type_str_resolver_utils.h"
#include "core/framework/layering_annotations.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/framework/transform_layout_functions.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/model_saving_options.h"
#include "core/providers/providers.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_optimizer_registry.h"
#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
#include "core/optimizer/gqa_value_layout_transformer.h"
#endif
#include "core/optimizer/layout_transformation/layout_transformation.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/selectors_actions/selector_action_transformer_apply_contexts.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/platform/Barrier.h"
#include "core/platform/threadpool.h"
#include "core/platform/telemetry_strings.h"
#ifdef _WIN32
#include "core/platform/tracing.h"
#include <Windows.h>
#include "core/platform/windows/telemetry.h"
#include "core/platform/windows/logging/etw_sink.h"
#endif
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_devices.h"
#ifdef USE_DML  // TODO: This is necessary for the workaround in TransformGraph
#include "core/providers/dml/DmlExecutionProvider/src/DmlGraphFusionTransformer.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlRuntimeGraphFusionTransformer.h"
#include "core/providers/dml/DmlExecutionProvider/src/GraphTransformer.h"
#include "core/providers/dml/dml_session_options_config_keys.h"
#include "core/providers/dml/DmlExecutionProvider/src/ExecutionProvider.h"
#include "core/optimizer/stft_decomposition.h"
#endif
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/user_logging_sink.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/util/thread_utils.h"

#ifdef _WIN32
#include "core/platform/windows/logging/etw_sink.h"
#include "core/common/logging/sinks/composite_sink.h"
#endif

// custom ops are not available in a minimal build unless ORT_MINIMAL_BUILD_CUSTOM_OPS is set
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/customregistry.h"
#include "core/session/custom_ops.h"
#endif
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#include "core/framework/stream_execution_context.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_optimizer.h"
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace {

// Parse a spin duration config value (in microseconds) from a string.
// Returns kSpinDurationDefault (-1) if the config is not explicitly set.
// Returns the parsed value (>= -1) if valid. Logs a warning and returns
// kSpinDurationDefault on parse failure.
constexpr int kSpinDurationWarnThresholdUs = 10000;  // 10ms — warn above this
int ParseSpinDurationUs(std::string_view str, const char* config_key,
                        const logging::Logger& logger) {
  int spin_us = concurrency::kSpinDurationDefault;
  if (!TryParseStringWithClassicLocale(str, spin_us) || spin_us < -1) {
    LOGS(logger, WARNING) << "Invalid value for " << config_key
                          << ": \"" << str << "\", using default spin duration setting";
    return concurrency::kSpinDurationDefault;
  }
  if (spin_us > kSpinDurationWarnThresholdUs) {
    LOGS(logger, WARNING) << config_key << " is set to " << spin_us
                          << "us (>" << kSpinDurationWarnThresholdUs
                          << "us). Large spin durations increase CPU/power usage. "
                          << "Typical values are 500-2000us.";
  }
  return spin_us;
}

#if !defined(ORT_MINIMAL_BUILD)
// Returns the virtual model path derived from the
// kOrtSessionOptionsModelExternalInitializersFileFolderPath config option, or an empty
// PathString when the option is not set. When set, external initializers are resolved
// relative to this folder, overriding the model's own directory.
PathString GetExternalInitializersFolderModelPath(const ConfigOptions& config_options) {
  const std::string external_data_folder_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsModelExternalInitializersFileFolderPath, "");
  if (external_data_folder_path.empty()) {
    return PathString{};
  }
  return ToPathString(external_data_folder_path + "/virtual_model.onnx");
}
#endif  // !defined(ORT_MINIMAL_BUILD)

// Parse a spin backoff max config value (exponential-backoff cap). Defaults to
// 1 (no backoff, one SpinPause() per iteration). Values >= 2 enable backoff.
unsigned int ParseSpinBackoffMax(std::string_view str, const char* config_key,
                                 const logging::Logger& logger) {
  unsigned int backoff = 1U;
  if (!TryParseStringWithClassicLocale(str, backoff)) {
    LOGS(logger, WARNING) << "Invalid value for " << config_key
                          << ": \"" << str << "\", using default (no backoff)";
    return 1U;
  }
  if (backoff == 0U) {
    LOGS(logger, WARNING) << config_key << " is set to 0; treating as 1 (no backoff). "
                          << "Valid values are >= 1.";
    return 1U;
  }
  if (backoff > concurrency::kSpinBackoffMaxLimit) {
    LOGS(logger, WARNING) << config_key << " is set to " << backoff
                          << " (>" << concurrency::kSpinBackoffMaxLimit
                          << "); clamping to " << concurrency::kSpinBackoffMaxLimit
                          << ". Typical values are 4-8.";
    return concurrency::kSpinBackoffMaxLimit;
  }
  return backoff;
}

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
// TODO: use LoggingManager::GetTimestamp and operator<<
// (see ostream_sink.cc for an example)
// to simplify this and match the log file timestamp format.
template <typename T>
inline std::basic_string<T> GetCurrentTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm local_tm;  // NOLINT

#ifdef _WIN32
  ORT_ENFORCE(localtime_s(&local_tm, &in_time_t) == 0);
#else
  localtime_r(&in_time_t, &local_tm);
#endif

  T time_str[32];
  OrtStrftime<T>(time_str, sizeof(time_str), GetDateFormatString<T>(), &local_tm);

  std::basic_stringstream<T> ss;
  ss << time_str << T('_') << std::setfill(T('0')) << std::setw(3) << ms.count();
  return ss.str();
}

#if !defined(ORT_MINIMAL_BUILD)

static bool HasControlflowNodes(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.ContainsSubgraph()) {
      return true;
    }
  }

  return false;
}

static bool HasMemcpyNodes(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MemcpyFromHost" || node.OpType() == "MemcpyToHost") {
      return true;
    }
  }

  return false;
}

static bool AreAllComputeNodesAssignedToEpOrCpu(const Graph& graph, ProviderType provider) {
  bool has_node_on_provider = false;

  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    if (node_provider == provider) {
      has_node_on_provider = true;
    } else if (!node_provider.empty() &&
               node_provider != kCpuExecutionProvider) {
      return false;
    }
  }

  // Allow graph capture when there are no Memcpy nodes and either:
  //   (a) at least one node is assigned to the target EP, or
  //   (b) the graph has no nodes at all.
  // Case (b) covers models that are fully consumed by the EP (e.g. runtime graph
  // fusion) or trivially fold away to an empty graph (e.g. a lone Constant node,
  // as used for allocator-initialization sessions). Such graphs have nothing that
  // violates the "all compute on the EP or CPU" requirement, so they must not be
  // rejected merely because no node remains assigned to the EP.
  // CPU EPs are allowed to show up as long as there is no Memcpy involved, since
  // shape subgraphs are forced onto CPU and do not introduce Memcpy nodes.
  return (has_node_on_provider || graph.NumberOfNodes() == 0) && !HasMemcpyNodes(graph);
}

static bool AreAllNodesInMainGraphAssignedToOneEp(const Graph& graph, ProviderType provider) {
  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    if (node_provider.empty() || node_provider != provider) {
      return false;
    }
  }

  return true;
}

static bool HasShapeSubgraphNodes(const Graph& graph) {
  bool has_shape_nodes = false;
  bool has_cpu_ep_nodes = false;

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Shape") {
      has_shape_nodes = true;
      break;
    }
  }

  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    if (node_provider.empty() || node_provider == kCpuExecutionProvider) {
      has_cpu_ep_nodes = true;
      break;
    }
  }

  return has_shape_nodes && has_cpu_ep_nodes;
}

Status GetMinimalBuildOptimizationHandling(
    std::string_view config_value, bool saving_ort_format,
    InferenceSession::MinimalBuildOptimizationHandling& minimal_build_optimization_handling) {
  if (config_value == "save") {
    if (saving_ort_format) {
      minimal_build_optimization_handling =
          InferenceSession::MinimalBuildOptimizationHandling::SaveMinimalBuildRuntimeOptimizations;
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           kOrtSessionOptionsConfigMinimalBuildOptimizations,
                           " value of 'save' is only valid when saving an ORT format model.");
  }

  if (config_value == "apply") {
    minimal_build_optimization_handling =
        InferenceSession::MinimalBuildOptimizationHandling::OnlyApplyMinimalBuildOptimizations;
    return Status::OK();
  }

  if (config_value.empty()) {
    minimal_build_optimization_handling =
        InferenceSession::MinimalBuildOptimizationHandling::ApplyFullBuildOptimizations;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid value for ", kOrtSessionOptionsConfigMinimalBuildOptimizations, ": ", config_value);
};

#endif  // !defined(ORT_MINIMAL_BUILD)

// Maps an OrtHardwareDeviceType (from the V2 OrtEpDevice path) to a string
// matching the EpDeviceUsage telemetry schema.
const char* HardwareDeviceTypeToString(OrtHardwareDeviceType type) {
  switch (type) {
    case OrtHardwareDeviceType_CPU:
      return "CPU";
    case OrtHardwareDeviceType_GPU:
      return "GPU";
    case OrtHardwareDeviceType_NPU:
      return "NPU";
    default:
      return "UNKNOWN";
  }
}

// Maps a legacy OrtDevice::DeviceType (used by EPs that did not come through the V2
// OrtEpDevice path) to a string matching the EpDeviceUsage telemetry schema.
const char* OrtDeviceTypeToString(OrtDevice::DeviceType type) {
  switch (type) {
    case OrtDevice::CPU:
      return "CPU";
    case OrtDevice::GPU:
      return "GPU";
    case OrtDevice::NPU:
      return "NPU";
    case OrtDevice::FPGA:
      return "FPGA";
    default:
      return "UNKNOWN";
  }
}

}  // namespace

std::atomic<uint32_t> InferenceSession::global_session_id_{1};
std::map<uint32_t, InferenceSession*> InferenceSession::active_sessions_;
#ifdef _WIN32
std::mutex InferenceSession::active_sessions_mutex_;  // Protects access to active_sessions_
onnxruntime::WindowsTelemetry::EtwInternalCallback InferenceSession::callback_ML_ORT_provider_;
#endif

#if defined(ORT_USE_TELEMETRY)
InferenceSession::Telemetry::Telemetry()
    : time_sent_last_(std::chrono::high_resolution_clock::now()) {}
#else
InferenceSession::Telemetry::Telemetry() = default;
#endif

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

logging::Severity GetSeverity(const SessionOptions& session_options) {
  logging::Severity severity = logging::Severity::kWARNING;

  if (session_options.session_log_severity_level == -1) {
    severity = logging::LoggingManager::DefaultLogger().GetSeverity();
  } else {
    ORT_ENFORCE(session_options.session_log_severity_level >= 0 &&
                    session_options.session_log_severity_level <= static_cast<int>(logging::Severity::kFATAL),
                "Invalid session log severity level. Not a valid onnxruntime::logging::Severity value: ",
                session_options.session_log_severity_level);
    severity = static_cast<logging::Severity>(session_options.session_log_severity_level);
  }
  return severity;
}

void InferenceSession::SetLoggingManager(const SessionOptions& session_options,
                                         const Environment& session_env) {
  logging_manager_ = session_env.GetLoggingManager();
  std::unique_ptr<logging::ISink> sink;

  if (session_options.user_logging_function) {
    sink = std::make_unique<UserLoggingSink>(session_options.user_logging_function,
                                             session_options.user_logging_param);
    auto sessionSeverity = GetSeverity(session_options);
    auto etwOverrideSeverity = logging::OverrideLevelWithEtw(sessionSeverity);
#ifdef _WIN32
    sink = EnhanceSinkWithEtw(std::move(sink), sessionSeverity, etwOverrideSeverity);
#endif

    user_logging_manager_ = std::make_unique<logging::LoggingManager>(std::move(sink),
                                                                      std::min(sessionSeverity, etwOverrideSeverity),
                                                                      false,
                                                                      logging::LoggingManager::InstanceType::Temporal,
                                                                      &session_options.session_logid);
    logging_manager_ = user_logging_manager_.get();
  }
}

void InferenceSession::ConstructorCommon(const SessionOptions& session_options,
                                         const Environment& session_env) {
  auto status = FinalizeSessionOptions(session_options, model_proto_, is_model_proto_parsed_, session_options_);
  ORT_ENFORCE(status.IsOK(), "Could not finalize session options while constructing the inference session. Error Message: ",
              status.ErrorMessage());

  // a monotonically increasing session id for use in telemetry
  session_id_ = global_session_id_.fetch_add(1);

  SetLoggingManager(session_options, session_env);

  // The call to InitLogger depends on the final state of session_options_. Hence it should be invoked
  // after the invocation of FinalizeSessionOptions.
  InitLogger(logging_manager_);  // this sets session_logger_ so that it can be used for logging after this point.
  TraceSessionOptions(session_options, false, *session_logger_);

#if !defined(ORT_MINIMAL_BUILD)
  // Update the number of steps for the graph transformer manager using the "finalized" session options
  ORT_THROW_IF_ERROR(graph_transformer_mgr_.SetSteps(session_options_.max_num_graph_transformation_steps));
  graph_transformer_mgr_.SetLoadCancellationFn(this->check_load_cancellation_fn_);
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  {
    auto disabled_string = session_options_.config_options.GetConfigOrDefault(
        kOrtSessionOptionsDisableSpecifiedOptimizers, "");
    if (!disabled_string.empty()) {
      const auto disabled_list = utils::SplitString(disabled_string, ";");
      InlinedHashSet<std::string> disabled_rules_and_transformers;
      disabled_rules_and_transformers.reserve(disabled_list.size());
      disabled_rules_and_transformers.insert(disabled_list.cbegin(), disabled_list.cend());
      ORT_THROW_IF_ERROR(FilterEnabledOptimizers(std::move(disabled_rules_and_transformers)));
    }
  }
#endif

  bool set_denormal_as_zero =
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSetDenormalAsZero, "0") == "1";

  // The only first session option for flush-to-zero and denormal-as-zero is effective to main thread and OpenMP threads.
  {
    static std::once_flag once;

    std::call_once(once, [&] {
      SetDenormalAsZero(set_denormal_as_zero);

      LOGS(*session_logger_, INFO) << "Flush-to-zero and denormal-as-zero are " << ((set_denormal_as_zero) ? "on" : "off");
    });
  }

  use_per_session_threads_ = session_options.use_per_session_threads;
  force_spinning_stop_between_runs_ = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigForceSpinningStop, "0") == "1";

  if (use_per_session_threads_) {
    LOGS(*session_logger_, INFO) << "Creating and using per session threadpools since use_per_session_threads_ is true";

#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
    auto apply_work_callbacks = [&session_env](OrtThreadPoolParams& params) {
      const auto* env_cbs = session_env.GetPerSessionWorkCallbacks();
      if (env_cbs != nullptr) {
        params.work_callbacks = *env_cbs;
      }
    };
#endif

    {
      if (!external_intra_op_thread_pool_) {
        bool allow_intra_op_spinning =
#if !defined(ORT_CLIENT_PACKAGE_BUILD)
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowIntraOpSpinning, "1") == "1";
#else
            // default KOrtSessionOptionsConfigAllowIntraOpSpinning to "0" for ORT builds targeting client/on-device workloads,
            // to reduce CPU utilization and improve power efficiency.
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0") == "1";
#endif
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
        to.allow_spinning = allow_intra_op_spinning;
        if (allow_intra_op_spinning) {
          to.spin_duration_us = ParseSpinDurationUs(
              session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigIntraOpSpinDurationUs, "-1"),
              kOrtSessionOptionsConfigIntraOpSpinDurationUs, *session_logger_);
          to.spin_backoff_max = ParseSpinBackoffMax(
              session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigIntraOpSpinBackoffMax, "1"),
              kOrtSessionOptionsConfigIntraOpSpinBackoffMax, *session_logger_);
        } else {
          to.spin_duration_us = concurrency::kSpinDurationDefault;
          to.spin_backoff_max = 1U;
        }
        to.dynamic_block_base_ = std::stoi(session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDynamicBlockBase, "0"));
        LOGS(*session_logger_, INFO) << "Dynamic block base set to " << to.dynamic_block_base_;

        // Set custom threading functions
        to.custom_create_thread_fn = session_options_.custom_create_thread_fn;
        to.custom_thread_creation_options = session_options.custom_thread_creation_options;
        to.custom_join_thread_fn = session_options_.custom_join_thread_fn;
        if (session_options_.config_options.TryGetConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, to.affinity_str)) {
          ORT_ENFORCE(!to.affinity_str.empty(), "Affinity string must not be empty");
        }
        to.auto_set_affinity = to.thread_pool_size == 0 &&
                               session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL &&
                               to.affinity_str.empty();

        if (to.custom_create_thread_fn) {
          ORT_ENFORCE(to.custom_join_thread_fn, "custom join thread function not set for intra op thread pool");
        }

#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
        apply_work_callbacks(to);
#endif

        thread_pool_ =
            concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
      }
    }
    if (session_options_.execution_mode == ExecutionMode::ORT_PARALLEL) {
      if (!external_inter_op_thread_pool_) {
        bool allow_inter_op_spinning =
#if !defined(ORT_CLIENT_PACKAGE_BUILD)
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowInterOpSpinning, "1") == "1";
#else
            // default kOrtSessionOptionsConfigAllowInterOpSpinning to "0" for ORT builds targeting client/on-device workloads,
            // to reduce CPU utilization and improve power efficiency.
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowInterOpSpinning, "0") == "1";
#endif
        OrtThreadPoolParams to = session_options_.inter_op_param;
        to.auto_set_affinity = to.thread_pool_size == 0 && session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL;
        std::basic_stringstream<ORTCHAR_T> ss;
        if (to.name) {
          ss << to.name << ORT_TSTR("-");
        }
        ss << ORT_TSTR("session-") << session_id_ << ORT_TSTR("-inter-op");
        inter_thread_pool_name_ = ss.str();
        to.name = inter_thread_pool_name_.c_str();
        to.set_denormal_as_zero = set_denormal_as_zero;
        to.allow_spinning = allow_inter_op_spinning;
        if (allow_inter_op_spinning) {
          to.spin_duration_us = ParseSpinDurationUs(
              session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigInterOpSpinDurationUs, "-1"),
              kOrtSessionOptionsConfigInterOpSpinDurationUs, *session_logger_);
          to.spin_backoff_max = ParseSpinBackoffMax(
              session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigInterOpSpinBackoffMax, "1"),
              kOrtSessionOptionsConfigInterOpSpinBackoffMax, *session_logger_);
        } else {
          to.spin_duration_us = concurrency::kSpinDurationDefault;
          to.spin_backoff_max = 1U;
        }
        to.dynamic_block_base_ = std::stoi(session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDynamicBlockBase, "0"));

        // Set custom threading functions
        to.custom_create_thread_fn = session_options_.custom_create_thread_fn;
        to.custom_thread_creation_options = session_options.custom_thread_creation_options;
        to.custom_join_thread_fn = session_options_.custom_join_thread_fn;

        if (to.custom_create_thread_fn) {
          ORT_ENFORCE(to.custom_join_thread_fn, "custom join thread function not set for inter op thread pool");
        }

#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
        apply_work_callbacks(to);
#endif

        inter_op_thread_pool_ =
            concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
        if (inter_op_thread_pool_ == nullptr) {
          LOGS(*session_logger_, INFO) << "Failed to create the inter-op thread pool for the parallel executor, setting ExecutionMode to SEQUENTIAL";
          session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
        }
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

#ifdef _WIN32
  // Scope the lock so that it is not held across WindowsTelemetry::RegisterInternalCallback below.
  // Holding active_sessions_mutex_ while taking the telemetry callback registry lock would invert
  // the order used by the ETW dispatch path and can deadlock the process. See the destructor.
  {
    std::lock_guard<std::mutex> lock(active_sessions_mutex_);
    active_sessions_[session_id_] = this;
  }

  // Register callback for ETW capture state (rundown) for Microsoft.ML.ONNXRuntime provider
  callback_ML_ORT_provider_ = onnxruntime::WindowsTelemetry::EtwInternalCallback(
      [](LPCGUID SourceId,
         ULONG IsEnabled,
         UCHAR Level,
         ULONGLONG MatchAnyKeyword,
         ULONGLONG MatchAllKeyword,
         PEVENT_FILTER_DESCRIPTOR FilterData,
         PVOID CallbackContext) {
        (void)SourceId;
        (void)Level;
        (void)MatchAnyKeyword;
        (void)MatchAllKeyword;
        (void)FilterData;
        (void)CallbackContext;
        ORT_UNUSED_PARAMETER(SourceId);
        ORT_UNUSED_PARAMETER(Level);
        ORT_UNUSED_PARAMETER(MatchAnyKeyword);
        ORT_UNUSED_PARAMETER(MatchAllKeyword);
        ORT_UNUSED_PARAMETER(FilterData);
        ORT_UNUSED_PARAMETER(CallbackContext);

        // Check if this callback is for capturing state
        if ((IsEnabled == EVENT_CONTROL_CODE_CAPTURE_STATE) &&
            ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)) != 0)) {
          InferenceSession::LogAllSessions();
        }
      });
  WindowsTelemetry::RegisterInternalCallback(callback_ML_ORT_provider_);

  // Register callback for ETW start / stop so that LOGS tracing can be adjusted dynamically after session start
  auto& etwRegistrationManager = logging::EtwRegistrationManager::Instance();
  callback_ETWSink_provider_ = onnxruntime::logging::EtwRegistrationManager::EtwInternalCallback(
      [&etwRegistrationManager, this](LPCGUID SourceId,
                                      ULONG IsEnabled,
                                      UCHAR Level,
                                      ULONGLONG MatchAnyKeyword,
                                      ULONGLONG MatchAllKeyword,
                                      PEVENT_FILTER_DESCRIPTOR FilterData,
                                      PVOID CallbackContext) {
        ORT_UNUSED_PARAMETER(SourceId);
        ORT_UNUSED_PARAMETER(Level);
        ORT_UNUSED_PARAMETER(MatchAnyKeyword);
        ORT_UNUSED_PARAMETER(MatchAllKeyword);
        ORT_UNUSED_PARAMETER(FilterData);
        ORT_UNUSED_PARAMETER(CallbackContext);

        if (logging_manager_ != nullptr) {
          auto ortETWSeverity = etwRegistrationManager.MapLevelToSeverity();

          if ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Logs)) != 0 &&
              IsEnabled == EVENT_CONTROL_CODE_ENABLE_PROVIDER) {
            LOGS(*session_logger_, VERBOSE) << "Adding ETW Sink to logger with severity level: " << (ULONG)ortETWSeverity;
            logging_manager_->AddSinkOfType(
                onnxruntime::logging::SinkType::EtwSink,
                []() -> std::unique_ptr<onnxruntime::logging::ISink> { return std::make_unique<onnxruntime::logging::EtwSink>(); },
                ortETWSeverity);
            onnxruntime::logging::LoggingManager::GetDefaultInstance()->AddSinkOfType(
                onnxruntime::logging::SinkType::EtwSink,
                []() -> std::unique_ptr<onnxruntime::logging::ISink> { return std::make_unique<onnxruntime::logging::EtwSink>(); },
                ortETWSeverity);
            LOGS(*session_logger_, INFO) << "Done Adding ETW Sink to logger with severity level: " << (ULONG)ortETWSeverity;
          }
          if (IsEnabled == EVENT_CONTROL_CODE_DISABLE_PROVIDER) {
            LOGS(*session_logger_, INFO) << "Removing ETW Sink from logger";
            logging_manager_->RemoveSink(onnxruntime::logging::SinkType::EtwSink);
            LOGS(*session_logger_, VERBOSE) << "Done Removing ETW Sink from logger";
          }
        }
      });

  // Register callback for ETW capture state (rundown)
  etwRegistrationManager.RegisterInternalCallback(callback_ETWSink_provider_);

#endif
}

void InferenceSession::TraceSessionOptions(const SessionOptions& session_options, bool captureState, const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(captureState);  // Otherwise Linux build error

  LOGS(logger, INFO) << session_options;

#if defined(_WIN32) && defined(ONNXRUNTIME_ENABLE_INSTRUMENT)
  const std::string optimized_model_filepath =
      ToUTF8String(telemetry_detail::BoundedTelemetryWideString(session_options.optimized_model_filepath.native()));
  const std::string profile_file_prefix =
      ToUTF8String(telemetry_detail::BoundedTelemetryWideString(session_options.profile_file_prefix));
  telemetry_detail::TelemetryStrings strings;

  TraceLoggingWrite(telemetry_provider_handle,
                    "SessionOptions",
                    TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    TraceLoggingUInt8(static_cast<UINT8>(session_options.execution_mode), "execution_mode"),
                    TraceLoggingUInt8(static_cast<UINT8>(session_options.execution_order), "execution_order"),
                    TraceLoggingBoolean(session_options.enable_profiling, "enable_profiling"),
                    TraceLoggingString(optimized_model_filepath.c_str(), "optimized_model_filepath"),
                    TraceLoggingBoolean(session_options.enable_mem_pattern, "enable_mem_pattern"),
                    TraceLoggingBoolean(session_options.enable_mem_reuse, "enable_mem_reuse"),
                    TraceLoggingBoolean(session_options.enable_cpu_mem_arena, "enable_cpu_mem_arena"),
                    TraceLoggingString(profile_file_prefix.c_str(), "profile_file_prefix"),
                    TraceLoggingString(strings.Utf8(session_options.session_logid), "session_logid"),
                    TraceLoggingInt8(static_cast<INT8>(session_options.session_log_severity_level), "session_log_severity_level"),
                    TraceLoggingInt8(static_cast<INT8>(session_options.session_log_verbosity_level), "session_log_verbosity_level"),
                    TraceLoggingUInt32(session_options.max_num_graph_transformation_steps, "max_num_graph_transformation_steps"),
                    TraceLoggingUInt8(static_cast<UINT8>(session_options.graph_optimization_level), "graph_optimization_level"),
                    TraceLoggingBoolean(session_options.use_per_session_threads, "use_per_session_threads"),
                    TraceLoggingBoolean(session_options.thread_pool_allow_spinning, "thread_pool_allow_spinning"),
                    TraceLoggingBoolean(session_options.use_deterministic_compute, "use_deterministic_compute"),
                    TraceLoggingBoolean(captureState, "isCaptureState"));

  TraceLoggingWrite(
      telemetry_provider_handle,
      "SessionOptions_IntraOrtThreadPoolParams",
      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
      TraceLoggingInt32(session_options.intra_op_param.thread_pool_size, "thread_pool_size"),
      TraceLoggingBoolean(session_options.intra_op_param.auto_set_affinity, "auto_set_affinity"),
      TraceLoggingBoolean(session_options.intra_op_param.allow_spinning, "allow_spinning"),
      TraceLoggingInt32(session_options.intra_op_param.dynamic_block_base_, "dynamic_block_base_"),
      TraceLoggingUInt32(session_options.intra_op_param.stack_size, "stack_size"),
      TraceLoggingString(strings.Utf8(session_options.intra_op_param.affinity_str), "affinity_str"),
      TraceLoggingBoolean(session_options.intra_op_param.set_denormal_as_zero, "set_denormal_as_zero"),
      TraceLoggingBoolean(captureState, "isCaptureState"));

  size_t config_count = 0;
  for (const auto& config_pair : session_options.config_options.configurations) {
    if (config_count++ == telemetry_detail::kMaxTelemetryCollectionEntries) break;
    telemetry_detail::TelemetryStrings config_strings;
    TraceLoggingWrite(
        telemetry_provider_handle,
        "SessionOptions_ConfigEntry",
        TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
        TraceLoggingLevel(WINEVENT_LEVEL_INFO),
        TraceLoggingString(config_strings.Utf8(config_pair.first), "Key"),
        TraceLoggingString(config_strings.Utf8(config_pair.second), "Value"),
        TraceLoggingBoolean(captureState, "isCaptureState"));
  }
#endif
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env)
    :
#if !defined(ORT_MINIMAL_BUILD)
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
#endif
      environment_(session_env) {
  // Initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const Environment& session_env,
                                   onnxruntime::concurrency::ThreadPool* external_intra_op_thread_pool,
                                   onnxruntime::concurrency::ThreadPool* external_inter_op_thread_pool)
    :
#if !defined(ORT_MINIMAL_BUILD)
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
#endif
      external_intra_op_thread_pool_(external_intra_op_thread_pool),
      external_inter_op_thread_pool_(external_inter_op_thread_pool),
      environment_(session_env) {
  // Initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#if !defined(ORT_MINIMAL_BUILD)
InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const PathString& model_uri)
    : model_location_(model_uri),
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      environment_(session_env) {
  auto status = Model::Load(model_location_, model_proto_);
  ORT_ENFORCE(status.IsOK(), "Given model could not be parsed while creating inference session. Error message: ",
              status.ErrorMessage());
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#ifdef _WIN32
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const Environment& session_env,
                                   const std::string& model_uri)
    : InferenceSession(session_options, session_env, ToPathString(model_uri)) {
}
#endif

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   std::istream& model_istream)
    : graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      environment_(session_env) {
  Status st = Model::Load(model_istream, &model_proto_);
  ORT_ENFORCE(st.IsOK(), "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const void* model_data, int model_data_len)
    : graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      environment_(session_env) {
  const bool result = model_proto_.ParseFromArray(model_data, model_data_len);
  ORT_ENFORCE(result, "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

InferenceSession::~InferenceSession() {
  // Flush any remaining RuntimePerf counters
  ORT_TRY {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    if (telemetry_.total_runs_since_last_ > 0) {
      const auto& telemetry_provider = Env::Default().GetTelemetryProvider();
      telemetry_provider.LogRuntimePerf(session_id_,
                                        telemetry_.total_runs_since_last_,
                                        telemetry_.total_run_duration_since_last_,
                                        telemetry_.duration_per_batch_size_);
      // Also flush a final EpDeviceUsage per (EP, device) to capture any runs
      // that occurred between the last heartbeat and session destruction.
      for (const auto& ep_info : telemetry_.ep_device_info_) {
        telemetry_provider.LogEpDeviceUsage(
            session_id_, ep_info.ep_type, ep_info.hardware_device_type,
            ep_info.vendor_id, ep_info.device_id, ep_info.vendor, ep_info.ep_vendor,
            ep_info.ep_version, ep_info.assigned_node_count,
            telemetry_.total_runs_since_last_, telemetry_.total_run_duration_since_last_);
      }
    }
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      LOGS(*session_logger_, ERROR) << "Error during telemetry flush: " << e.what();
    });
  }
  ORT_CATCH(...) {
    LOGS(*session_logger_, ERROR) << "Unknown error during telemetry flush";
  }

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

  // Unregister the session and ETW callbacks
  //
#ifdef _WIN32
  // Remove the session first so that callbacks cannot observe it during teardown. Release
  // active_sessions_mutex_ before unregistering because the callback registries have their own
  // locks, and ETW dispatch takes those locks before LogAllSessions takes active_sessions_mutex_.
  {
    std::lock_guard<std::mutex> lock(active_sessions_mutex_);
    active_sessions_.erase(session_id_);
  }

  if (callback_ML_ORT_provider_ != nullptr) {
    WindowsTelemetry::UnregisterInternalCallback(callback_ML_ORT_provider_);
  }
  if (callback_ETWSink_provider_ != nullptr) {
    logging::EtwRegistrationManager::Instance().UnregisterInternalCallback(callback_ETWSink_provider_);
  }
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  if (session_activity_started_)
    TraceLoggingWriteStop(session_activity, "OrtInferenceSessionActivity");
#endif
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  GetMemoryProfiler().GenerateMemoryProfile();
#endif
}

common::Status InferenceSession::RegisterExecutionProvider(const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
  if (p_exec_provider == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for exec provider");
  }

  std::lock_guard<std::mutex> l(session_mutex_);

  if (is_inited_) {
    // adding an EP is pointless as the graph as already been partitioned so no nodes will be assigned to
    // the new EP
    LOGS(*session_logger_, ERROR) << "Execution providers must be registered before the session is initialized. ";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Execution providers must be registered before the session is initialized.");
  }

  const std::string& provider_type = p_exec_provider->Type();

  // Some session option values (default or user provided) may not work with some EPs.
  // Rather than put the onus on the user to know these, make the appropriate change while logging the change.
  if (provider_type == onnxruntime::kDmlExecutionProvider || provider_type == onnxruntime::kWebGpuExecutionProvider) {
    // DML and WebGPU memory is not byte addressable and hence mem pattern doesn't work.
    if (session_options_.enable_mem_pattern) {
      LOGS(*session_logger_, INFO)
          << "Having memory pattern enabled is not supported while using " << provider_type << ". "
          << "So disabling it for this session since it uses " << provider_type << ".";
      session_options_.enable_mem_pattern = false;
    }

    // Parallel execution mode does not support DML EP
    if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
      LOGS(*session_logger_, INFO)
          << "Parallel execution mode does not support the DML Execution Provider. "
          << "So making the execution mode sequential for this session since it uses the DML Execution Provider.";

      session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    }
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Register Custom Op if EP requests it
  std::vector<OrtCustomOpDomain*> custom_op_domains;
  std::vector<OrtCustomOpDomain*> candidate_custom_op_domains;
  p_exec_provider->GetCustomOpDomainList(candidate_custom_op_domains);

  auto registry_kernels = kernel_registry_manager_.GetKernelRegistriesByProviderType(p_exec_provider->Type());

  // Register the custom op domain only if it has not been registered before
  if (registry_kernels.empty()) {
    custom_op_domains = candidate_custom_op_domains;
  } else {
    for (auto candidate_custom_op_domain : candidate_custom_op_domains) {
      for (auto registry_kernel : registry_kernels) {
        const auto& kernel_map = registry_kernel->GetKernelCreateMap();
        bool need_register = true;
        // If the kernel registry is the ep's custom op registry, we only need to check the first kernel,
        // because all kernels in one kernel registry should have the same domain name.
        for (auto iter = kernel_map.begin(); iter != kernel_map.end(); iter++) {
          if (iter->second.kernel_def->Domain() == candidate_custom_op_domain->domain_) {
            need_register = false;
            break;
          }
        }
        if (need_register) {
          custom_op_domains.push_back(candidate_custom_op_domain);
        }
      }
    }
  }

  if (!custom_op_domains.empty()) {
    if (AddCustomOpDomains(custom_op_domains) != Status::OK()) {
      LOGS(*session_logger_, WARNING) << "Can't register custom op domains with ORT for " << provider_type;
    }
  }
#endif

  // if any EPs do not support concurrent calls to Run we add locking around graph execution
  if (p_exec_provider->ConcurrentRunSupported() == false) {
    is_concurrent_run_supported_ = false;
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
  session_profiler_.AddEpProfilers(p_exec_provider->GetProfiler());
  return execution_providers_.Add(provider_type, p_exec_provider);
}

// Custom Op support
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
common::Status InferenceSession::AddCustomOpDomains(gsl::span<OrtCustomOpDomain* const> op_domains) {
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

#define ORT_TELEMETRY_CAPTURE_STATUS_BEGIN(status) status = [&]() -> Status {
#define ORT_TELEMETRY_CAPTURE_STATUS_END() \
  return Status::OK();                     \
  }                                        \
  ()

#if !defined(ORT_MINIMAL_BUILD)
common::Status InferenceSession::RegisterGraphTransformer(
    std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer, TransformerLevel level) {
  if (p_graph_transformer == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for graph transformer");
  }

  std::lock_guard<std::mutex> l(session_mutex_);

  if (is_inited_) {
    // adding a transformer now is pointless as the graph as already been transformed
    LOGS(*session_logger_, ERROR) << "Graph transformers must be registered before the session is initialized.";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Graph transformers must be registered before the session is initialized.");
  }

  return graph_transformer_mgr_.Register(std::move(p_graph_transformer), level);
}

common::Status InferenceSession::SaveToOrtFormat(const std::filesystem::path& filepath) const {
  // Get the byte size of the ModelProto and round it to the next MB and use it as flatbuffers' init_size
  // TODO: Investigate whether we should set a max size, and clarify the cost of having a buffer smaller than
  // what the total flatbuffers serialized size will be.
  constexpr size_t m_bytes = 1024 * 1024;
  size_t fbs_buffer_size = std::max(m_bytes, model_->ToProto().ByteSizeLong());
  fbs_buffer_size = ((fbs_buffer_size + m_bytes - 1) / m_bytes) * m_bytes;
  flatbuffers::FlatBufferBuilder builder(fbs_buffer_size);

  auto ort_model_version = builder.CreateString(std::to_string(kOrtModelVersion));
  flatbuffers::Offset<fbs::Model> fbs_model;
  ORT_RETURN_IF_ERROR(
      model_->SaveToOrtFormat(builder, fbs_model));

  flatbuffers::Offset<fbs::KernelTypeStrResolver> fbs_kernel_type_str_resolver;
  KernelTypeStrResolver kernel_type_str_resolver{};
  ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterGraphNodeOpSchemas(model_->MainGraph()));
  ORT_RETURN_IF_ERROR(standalone::RegisterCustomOpNodeSchemas(kernel_type_str_resolver, model_->MainGraph()));

  for (const auto& op_schema : saved_runtime_optimization_produced_node_op_schemas_) {
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }

  ORT_RETURN_IF_ERROR(
      kernel_type_str_resolver.SaveToOrtFormat(builder, fbs_kernel_type_str_resolver));

  fbs::InferenceSessionBuilder sb(builder);
  sb.add_ort_version(ort_model_version);
  sb.add_model(fbs_model);
  sb.add_kernel_type_str_resolver(fbs_kernel_type_str_resolver);
  auto session = sb.Finish();
  builder.Finish(session, fbs::InferenceSessionIdentifier());

  {
    std::ofstream file(filepath, std::ios::binary);
    uint8_t* buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    file.write(reinterpret_cast<const char*>(buf), size);
    ORT_RETURN_IF_NOT(file, "Failed to save ORT format model to file: ", ToUTF8String(filepath.native()));
  }

  return Status::OK();
}

common::Status InferenceSession::LoadWithLoader(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                                                const std::string& event_name) {
  Status status = Status::OK();
  TimePoint tp{};
#if defined(ORT_USE_TELEMETRY)
  tp = std::chrono::high_resolution_clock::now();
#endif
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.Start();
  }
#if defined(_WIN32) || defined(ORT_USE_TELEMETRY)
  const Env& env = Env::Default();
#endif
  ORT_TRY {
#if defined(_WIN32) || defined(ORT_USE_TELEMETRY)
    env.GetTelemetryProvider().LogModelLoadStart(session_id_);
#endif

    ORT_TELEMETRY_CAPTURE_STATUS_BEGIN(status)
    std::lock_guard<std::mutex> l(session_mutex_);
    if (is_model_loaded_) {  // already loaded
      LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
      return common::Status(common::ONNXRUNTIME, common::MODEL_LOADED,
                            "This session already contains a loaded model.");
    }

    std::shared_ptr<onnxruntime::Model> p_tmp_model;
    ORT_RETURN_IF_ERROR_SESSIONID_(loader(p_tmp_model));

    model_ = p_tmp_model;

    ORT_RETURN_IF_ERROR_SESSIONID_(DoPostLoadProcessing(*model_));

    // all steps complete, mark the model as loaded.
    is_model_loaded_ = true;

    telemetry_.event_name_ = telemetry_detail::BoundedTelemetryString(event_name);
    ORT_TELEMETRY_CAPTURE_STATUS_END();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(common::ONNXRUNTIME, common::FAIL, "Exception during loading: " + std::string(ex.what()));
    });
  }
  ORT_CATCH(...) {
    LOGS(*session_logger_, ERROR) << "Unknown exception";
    status = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION,
                    "Encountered unknown exception in LoadWithLoader()");
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, event_name, tp);
  }

#if defined(ORT_USE_TELEMETRY)
  env.GetTelemetryProvider().LogModelLoadEnd(session_id_, status, TimeDiffMicroSeconds(tp));
#elif defined(_WIN32)
  env.GetTelemetryProvider().LogModelLoadEnd(session_id_, status, 0);
#endif

  return status;
}

common::Status InferenceSession::LoadOnnxModel(const PathString& model_uri) {
  model_location_ = model_uri;
  auto loader = [this, model_uri](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_location_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif

    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                                kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";
    ModelOptions model_opts{allow_released_opsets_only, strict_shape_type_inference, check_load_cancellation_fn_};

    // When set, the external initializers folder overrides the model's own directory as the
    // base for resolving external data. The model bytes are still read from model_uri.
    PathString external_data_model_path = GetExternalInitializersFolderModelPath(session_options_.config_options);
    if (!external_data_model_path.empty()) {
      model_location_ = external_data_model_path;
      return onnxruntime::Model::Load(model_uri, model_location_, model,
                                      HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                      *session_logger_, model_opts);
    }
    return onnxruntime::Model::Load(model_location_, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                    *session_logger_, model_opts);
  };

  common::Status st = LoadWithLoader(loader, "model_loading_uri");
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "Load model from " << ToUTF8String(model_uri) << " failed:" << st.ErrorMessage();
    return common::Status(st.Category(), st.Code(), oss.str());
  }
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status InferenceSession::FilterEnabledOptimizers(InlinedHashSet<std::string>&& optimizers_to_disable) {
  optimizers_to_disable_ = std::move(optimizers_to_disable);
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

common::Status InferenceSession::Load(const PathString& model_uri) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type && fbs::utils::IsOrtFormatModel(model_uri))) {
    return LoadOrtModel(model_uri);
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  return LoadOnnxModel(model_uri);
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#ifdef _WIN32
common::Status InferenceSession::Load(const std::string& model_uri) {
  return Load(ToPathString(model_uri));
}
#endif

common::Status InferenceSession::Load(const void* model_data, int model_data_len) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type &&
       fbs::utils::IsOrtFormatModelBytes(model_data, model_data_len))) {
    return LoadOrtModel(model_data, model_data_len);
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
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif

    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                                kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";

    PathString external_data_model_path = GetExternalInitializersFolderModelPath(session_options_.config_options);
    if (!external_data_model_path.empty()) {
      model_location_ = external_data_model_path;
    }

    return onnxruntime::Model::Load(std::move(model_proto), model_location_, model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                                    ModelOptions{allow_released_opsets_only, strict_shape_type_inference,
                                                 check_load_cancellation_fn_});
  };

  return LoadWithLoader(loader, "model_loading_array");
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
namespace {
// Validates the GroupQueryAttention Value layout session option and returns the requested layout.
//
// An unrecognized value is a caller mistake regardless of model format, so this has to run before any
// format-specific restriction; otherwise a typo like "NHWC" would be reported as an ORT format
// limitation instead of naming the bad value and the accepted ones.
//
// Shared by the ONNX and ORT format load paths when layout support is enabled.
Status GetGqaValueLayout(const ConfigOptions& config_options, std::string& layout, bool& explicitly_set) {
  explicitly_set = config_options.TryGetConfigEntry(kOrtSessionOptionsGqaValueLayout, layout);
  if (!explicitly_set) {
    layout = kGqaValueLayoutBNSH;
  }

  if (layout != kGqaValueLayoutBNSH && layout != kGqaValueLayoutBNHS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid value for session option '", kOrtSessionOptionsGqaValueLayout,
                           "': '", layout, "'. Expected '", kGqaValueLayoutBNSH, "' or '", kGqaValueLayoutBNHS,
                           "'.");
  }

  return Status::OK();
}
}  // namespace

#endif

#if !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::LoadOnnxModel(ModelProto model_proto) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                                kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";

    PathString external_data_model_path = GetExternalInitializersFolderModelPath(session_options_.config_options);
    if (!external_data_model_path.empty()) {
      model_location_ = external_data_model_path;
    }

    // This call will move model_proto to the constructed model instance
    return onnxruntime::Model::Load(std::move(model_proto), model_location_, model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                                    ModelOptions{allow_released_opsets_only, strict_shape_type_inference,
                                                 check_load_cancellation_fn_});
  };

  return LoadWithLoader(loader, "model_loading_proto");
}

common::Status InferenceSession::LoadOnnxModel(std::unique_ptr<ModelProto> p_model_proto) {
  return LoadOnnxModel(std::move(*p_model_proto));
}

common::Status InferenceSession::Load(std::istream& model_istream, bool allow_released_opsets_only) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_istream, &allow_released_opsets_only](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;
    Status st = Model::Load(model_istream, &model_proto);
    if (!st.IsOK()) {
      return st;
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    ModelOptions model_opts{allow_released_opsets_only, strict_shape_type_inference, check_load_cancellation_fn_};

    PathString external_data_model_path = GetExternalInitializersFolderModelPath(session_options_.config_options);
    if (!external_data_model_path.empty()) {
      model_location_ = external_data_model_path;
    }

    return onnxruntime::Model::Load(std::move(model_proto), model_location_, model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                    *session_logger_, model_opts);
  };

  return LoadWithLoader(loader, "model_loading_istream");
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
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                                kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";

    PathString external_data_model_path = GetExternalInitializersFolderModelPath(session_options_.config_options);
    if (!external_data_model_path.empty()) {
      model_location_ = external_data_model_path;
    }

    // Pass on ownership of the parsed ModelProto to the Model instance (its job here is done by this stage)
    return Model::Load(std::move(this->model_proto_), model_location_, model,
                       HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                       ModelOptions{allow_released_opsets_only, strict_shape_type_inference, check_load_cancellation_fn_});
  };

  return LoadWithLoader(loader, "model_loading_from_saved_proto");
}

common::Status InferenceSession::Load(const OrtModel& model_editor_api_model) {
  std::lock_guard<std::mutex> l(session_mutex_);

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

  const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                               kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
  const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                              kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";

  // need to go from unique_ptr to shared_ptr when moving into model_
  std::unique_ptr<Model> tmp_model;
  ORT_RETURN_IF_ERROR(Model::LoadFromModelEditorApiModel(model_editor_api_model,
                                                         HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                                         ModelOptions{allow_released_opsets_only,
                                                                      strict_shape_type_inference,
                                                                      check_load_cancellation_fn_},
                                                         *session_logger_, tmp_model));

  model_ = std::move(tmp_model);

  is_model_loaded_ = true;

  return Status::OK();
}

common::Status InferenceSession::ApplyUpdates(const OrtModel& model_editor_api_model) {
  std::lock_guard<std::mutex> l(session_mutex_);

  if (!is_model_loaded_) {
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session does not contain a loaded model.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  if (is_inited_) {
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session has already been initialized.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  return model_->MainGraph().UpdateUsingModelEditorApiModel(model_editor_api_model);
}

#if !defined(ORT_MINIMAL_BUILD)
static std::unique_ptr<OrtEpAssignedSubgraph> CreateEpAssignedSubgraph(const Graph& graph,
                                                                       const ComputeCapability& capability,
                                                                       const std::string& ep_name) {
  auto assigned_subgraph = std::make_unique<OrtEpAssignedSubgraph>();
  assigned_subgraph->ep_name = ep_name;

  gsl::span<NodeIndex> node_indices = capability.sub_graph->nodes;

  for (NodeIndex node_index : node_indices) {
    const Node* node = graph.GetNode(node_index);
    if (node != nullptr) {
      auto assigned_node = std::make_unique<OrtEpAssignedNode>();
      assigned_node->name = node->Name();
      assigned_node->domain = node->Domain();
      assigned_node->op_type = node->OpType();

      assigned_subgraph->nodes.push_back(assigned_node.get());
      assigned_subgraph->nodes_storage.push_back(std::move(assigned_node));
    }
  }

  return assigned_subgraph;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::TransformGraph(onnxruntime::Graph& graph, bool saving_model_in_ort_format) {
  // The transformer order:
  // 1. Ensure we inline as many functions as possible. We refer to it as Ahead Of Time (AOT) function inlining.
  // 2. ensure potential QDQ node units have unique DQ nodes (required transformer).
  //    - This is a required transformer as the ORT code has a hard requirement there are no overlapping QDQ node units.
  //    - We run it here in case optimizers are disabled.
  // 3. run level 1 optimizations. these only use ONNX operators.
  // 4. partition nodes based on EP capabilities. EPs may fuse nodes during this process.
  // 5. run level 2+ optimizations. level 2 and 3 optimizations use contrib ops.
  // 6. insert cast nodes (required transformer).
  // 7. run level 4 optimizations.
  // 8. Repeat steps 5 to 7 depending on the graph optimizations loop level.
  // 9. insert copy nodes (required transformer).

  OnPartitionAssignmentFunction on_partition_assignment_fn;
#if !defined(ORT_MINIMAL_BUILD)
  bool record_ep_graph_assignment =
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "0") == "1";
  if (record_ep_graph_assignment) {
    on_partition_assignment_fn = [this](const Graph& graph, const ComputeCapability& compute_capability,
                                        const std::string& ep_name) {
      std::unique_ptr<OrtEpAssignedSubgraph> assigned_subgraph = CreateEpAssignedSubgraph(graph,
                                                                                          compute_capability,
                                                                                          ep_name);

      this->ep_graph_assignment_info_.push_back(assigned_subgraph.get());
      this->ep_graph_assignment_info_storage_.push_back(std::move(assigned_subgraph));
    };
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  // Create GraphOptimizerRegistry instance for providing predefined graph optimizers and selection functions for EPs to lookup
  auto graph_optimizer_registry = std::make_unique<GraphOptimizerRegistry>(&session_options_,
                                                                           execution_providers_.Get(onnxruntime::kCpuExecutionProvider),
                                                                           session_logger_);
  GraphPartitioner partitioner(kernel_registry_manager_, execution_providers_, std::move(graph_optimizer_registry),
                               check_load_cancellation_fn_, on_partition_assignment_fn);

  // AOT capability discovery may copy initializer data, so reject arbitrary in-memory references first.
  ORT_RETURN_IF_ERROR_SESSIONID_(graph.ValidateInMemoryInitializers());

  // Run Ahead Of time function inlining
  if (const bool disable_aot_function_inlining =
          session_options_.config_options.GetConfigOrDefault(
              kOrtSessionOptionsDisableAheadOfTimeFunctionInlining, "0") == "1";
      !disable_aot_function_inlining) {
    uint32_t registered_ep_context_data_callbacks = session_options_.ep_context_data_read_func != nullptr
                                                        ? OrtEpContextDataCallbackSupportFlags_READ
                                                        : OrtEpContextDataCallbackSupportFlags_NONE;
    if (session_options_.ep_context_gen_options.TryGetEpContextDataWriteFunc() != nullptr) {
      registered_ep_context_data_callbacks |= OrtEpContextDataCallbackSupportFlags_WRITE;
    }
    ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.InlineFunctionsAOT(*model_,
                                                                  execution_providers_,
                                                                  kernel_registry_manager_,
                                                                  session_options_.config_options,
                                                                  *session_logger_,
                                                                  registered_ep_context_data_callbacks));
  }

  // We choose to convert initializers into OrtValues before partitioning here so plug-in EPs could
  // take advantage of the initializers being in OrtValue format and not to deal with protobuf.
  //
  // The initializers data is transferred to an OrtValue. The original TensorProto is replaced
  // with a TensorProto that has the same data type, shape and name. However, its external data
  // is used in a non-standard way. The location is set to a string constant utils::kTensorProtoNativeEndianMemoryAddressTag,
  // The file offset is set to the address of the OrtValue's data buffer,  and the length is set to the size of the
  // OrtValue's data buffer. Because this external location is non-standard, onnx code can not handle it. For this reason,
  // we do not convert them at the graph constructor because Node::ToProto() reconstructs Graph instances for subgraphs
  // and we do not want to have initializers converted at shape inference time, as Resolve() is called from EPs when
  // op_types are not assigned yet.
  //
  // If any transformations are applied later, they would not introduce any in-memory initializers,
  // type and shape inference would run only on any newly added nodes and any new initializers
  // will be converted at session finalization time.
  //
  // The conversion is performed using the following steps (within ConvertInitializersIntoOrtValues())
  //   constexpr const bool use_tensor_buffer_true = true;
  //   auto tensor_proto_to_add = utils::TensorToTensorProto(ort_value.Get<Tensor>(), tensor_proto.name(),
  //                                                        use_tensor_buffer_true);
  //   ORT_RETURN_IF_ERROR(graph.ReplaceInitializedTensor(tensor_proto_to_add, ort_value));
  ORT_RETURN_IF_ERROR_SESSIONID_(graph.ConvertInitializersIntoOrtValues());

  auto apply_transformer_once = [](const GraphTransformer& transformer, const logging::Logger& logger,
                                   Graph& graph, bool* is_graph_modified = nullptr) -> onnxruntime::common::Status {
    bool modified = false;
    auto status = transformer.Apply(graph, modified, logger);
    if (is_graph_modified) {
      *is_graph_modified = *is_graph_modified || modified;
    }
    return status;
  };

  auto apply_transformer_at_level = [](onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                       const TransformerLevel& level, const logging::Logger& logger, Graph& graph,
                                       bool* is_graph_modified = nullptr) -> onnxruntime::common::Status {
    graph_transformer_mgr.ClearGraphModified();
    auto status = graph_transformer_mgr.ApplyTransformers(graph, level, logger);
    if (is_graph_modified) {
      *is_graph_modified = *is_graph_modified || graph_transformer_mgr.IsGraphModified();
    }
    return status;
  };

  // ensure potential QDQ node units have unique DQ nodes
  if (const bool disable_quant_qdq =
          session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableQuantQDQ, "0") == "1";
      !disable_quant_qdq) {
    EnsureUniqueDQForNodeUnit ensure_unique_dq_for_node_unit{};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(ensure_unique_dq_for_node_unit, *session_logger_, graph));
  }

  // apply execution provider independent level 0 and 1 graph optimizations.
  ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Default, *session_logger_));
  ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Level1, *session_logger_));

#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
  // adapt GroupQueryAttention to a BNHS Value KV-cache if the application asked for one.
  // this is applied here rather than being registered as a level 1 optimizer for two reasons:
  //   - it changes the layout the session expects at its inputs and outputs, so it must run even
  //     when optimizations are disabled. AddPredefinedTransformers only registers level 1 and above
  //     when graph_optimization_level >= level.
  //   - it must run after the level 1 TransposeOptimizer, which moves, merges and cancels Transpose
  //     nodes, so that the Transpose -> GQA -> Transpose sequence reaches GetCapability intact for
  //     an EP that fuses it.
  // Builds without layout support reject an explicit option in Initialize().
  // An unrecognized value is the caller passing a bad argument, so GetGqaValueLayout() reports
  // INVALID_ARGUMENT. A recognized value that this particular model cannot satisfy is reported as
  // FAIL by the transformer, which keeps the two situations distinguishable to an application that
  // wants to fall back to BNSH.
  std::string gqa_value_layout;
  bool gqa_value_layout_explicitly_set = false;
  ORT_RETURN_IF_ERROR_SESSIONID_(
      GetGqaValueLayout(session_options_.config_options, gqa_value_layout, gqa_value_layout_explicitly_set));

  GqaValueLayoutBoundaries converted_gqa_value_boundaries;
  if (gqa_value_layout != kGqaValueLayoutBNSH) {
    GqaValueLayoutTransformer gqa_value_layout_transformer{&converted_gqa_value_boundaries};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(gqa_value_layout_transformer, *session_logger_, graph));

    // A GroupQueryAttention inside a subgraph cannot be converted: its Value cache boundary may be
    // carried in and out of the main graph, so the operator and the boundary live in different graphs
    // and there is nothing to rewire from here. A warning does not preserve the option contract --
    // the application would bind BNHS buffers to a boundary that is still BNSH, which passes input
    // validation whenever the trailing dimensions are dynamic or equal -- so this fails.
    //
    // Checked whatever else happened, not only when nothing converted: a model with a convertible
    // main-graph cache *and* a subgraph one would otherwise slip through on the strength of the part
    // that did convert.
    const GqaNodeCounts gqa_nodes = CountGqaNodes(graph);
    if (gqa_nodes.in_subgraphs != 0) {
      ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "'", kOrtSessionOptionsGqaValueLayout, "' was set to '", kGqaValueLayoutBNHS, "' but ",
          gqa_nodes.in_subgraphs,
          " GroupQueryAttention node(s) are inside a subgraph (a Loop body or BeamSearch "
          "decoder), which this option cannot reach. Their Value cache boundary would stay BNSH while the "
          "application supplied BNHS. Use '",
          kGqaValueLayoutBNSH,
          "', or a model whose GroupQueryAttention "
          "nodes are in the main graph."));
    }

    if (converted_gqa_value_boundaries.Empty()) {
      if (gqa_nodes.in_main_graph != 0) {
        // GQA is present and reachable, so the per-node warnings from the transformer already said
        // why each operand was left alone. Summarize rather than repeat.
        LOGS(*session_logger_, WARNING)
            << "'" << kOrtSessionOptionsGqaValueLayout << "' was set to '" << kGqaValueLayoutBNHS
            << "' but none of the " << gqa_nodes.in_main_graph
            << " GroupQueryAttention node(s) had a Value cache boundary in scope; see the warnings above. Value "
               "cache buffers bound to this session are still BNSH.";
      } else {
        // Harmless: nothing in this model uses a GQA Value cache, so there is nothing to bind.
        LOGS(*session_logger_, WARNING)
            << "'" << kOrtSessionOptionsGqaValueLayout << "' was set to '" << kGqaValueLayoutBNHS
            << "' but the model contains no GroupQueryAttention node, so the option has no effect.";
      }
    }
  } else if (gqa_value_layout_explicitly_set) {
    // An explicit BNSH request is a claim about the boundary, so it has to be enforced rather than
    // merely not acted on. A model saved from a BNHS session (via session.optimized_model_filepath)
    // still carries the Transposes and BNHS boundary shapes; honouring a BNSH request over it would
    // have the application bind BNSH buffers to a BNHS boundary, which is a shape error at best and a
    // silent misread when the dimensions are dynamic or happen to be square.
    //
    // Deliberately gated on the option being set rather than on its effective value. Defaulting to
    // BNSH and enforcing that would reject models whose Value cache already surfaces through boundary
    // Transposes -- which load and run correctly today -- and that is a compatibility break on the
    // default path, not an opt-in behaviour change. Such a model gets a warning below instead.
    //
    // Not applied on the ORT format path either: there the option is forced to BNSH and a converted
    // model is the documented way to use BNHS, so the same check would reject the supported workflow.
    const GqaValueLayoutBoundaries existing = FindConvertedGqaValueLayoutBoundaries(graph);
    if (!existing.Empty()) {
      ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "This model already carries the BNHS GroupQueryAttention Value layout: ",
          existing.past_value_inputs.size() + existing.present_value_outputs.size(),
          " boundary tensor(s) are declared BNHS. It cannot be loaded with '", kOrtSessionOptionsGqaValueLayout,
          "' set to '", kGqaValueLayoutBNSH,
          "', because the application would bind BNSH buffers to a BNHS "
          "boundary. Set '",
          kOrtSessionOptionsGqaValueLayout, "' to '", kGqaValueLayoutBNHS,
          "', or load a model whose Value cache boundary is BNSH."));
    }
  } else {
    // No layout requested, so ORT has no claim to enforce and the model keeps working exactly as it
    // did before this option existed. Still worth surfacing: the application has to bind BNHS buffers
    // to these boundaries, and saying so explicitly makes the contract checkable.
    const GqaValueLayoutBoundaries existing = FindConvertedGqaValueLayoutBoundaries(graph);
    if (!existing.Empty()) {
      LOGS(*session_logger_, WARNING)
          << "This model carries the BNHS GroupQueryAttention Value layout: "
          << (existing.past_value_inputs.size() + existing.present_value_outputs.size())
          << " boundary tensor(s) are declared BNHS, so the application must bind BNHS Value cache buffers. Set '"
          << kOrtSessionOptionsGqaValueLayout << "' to '" << kGqaValueLayoutBNHS
          << "' to state that explicitly and have ORT check it.";
    }
  }
#endif

  // if saving model to ORT format we only assign nodes a custom EP can handle and don't compile them.
  // we do this to preserve the original nodes in the model but prevent optimizers from changing them.
  // at runtime, the ORT format model will re-do the partitioning/compilation of these nodes, which may change
  // to cover fewer nodes due to device capabilities.
  auto mode = saving_model_in_ort_format ? GraphPartitioner::Mode::kAssignOnly
                                         : GraphPartitioner::Mode::kNormal;

  layout_transformation::TransformLayoutFunction transform_layout_fn = nullptr;

  // only provide NCWH to NHWC layout transformer if supported
  if (layout_transformation::IsSupportedOpset(graph)) {
    // we want to run L1 transformers after the layout transform primarily to constant fold any initializers
    // that get converted to an alternative layout.
    // create a lambda to combine the two operations in the layout transformation function
    transform_layout_fn = [this](Graph& graph_to_transform, bool& modified,
                                 const IExecutionProvider& execution_provider,
                                 const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
      AllocatorPtr cpu_allocator = CPUAllocator::DefaultInstance();
      ORT_RETURN_IF_ERROR_SESSIONID_(
          layout_transformation::TransformLayoutForEP(graph_to_transform, modified, execution_provider,
                                                      std::move(cpu_allocator), debug_graph_fn));

      // Previously we ran the L1 transformers to handle constant folding of any initializers that were transposed in
      // a QDQ format model. The transpose optimizer can now do the following, which takes care of most models without
      // needing this.
      //   - Look past DQ nodes to directly update initializers in-place.
      //   - Fix-up broken Transpose QDQ groups.
      //   - Constant fold inserted Squeeze and Transpose ops.
      //
      // if (modified) {
      //  ORT_RETURN_IF_ERROR_SESSIONID_(
      //      graph_transformer_mgr_.ApplyTransformers(graph_to_transform, TransformerLevel::Level1, *session_logger_));
      //
      // debug the graph after the L1 transformers have run against any layout transformation changes.
      // this is prior to GraphPartitioner::GetCapabilityForEP calling IExecutionProvider::GetCapability the second
      // time to validate the EP that requested the layout transformation can take all nodes using the new layout.
      // if that fails, this allows debugging the graph used in that GetCapability call.
      // if (debug_graph_fn) {
      //  debug_graph_fn(graph_to_transform);
      //}
      //}

      return Status::OK();
    };
  }

  // debug infrastructure for layout transformation. it's extremely difficult to trace the transpose optimizer changes
  // manually, so dumping out the model so it can be viewed in Netron makes it far easier
  layout_transformation::DebugGraphFn debug_graph_fn;
  if (transform_layout_fn) {
    bool enable_debug = session_options_.config_options.GetConfigOrDefault(kDebugLayoutTransformation, "0") == "1";

    if (enable_debug) {
      // init counter to 1 to match to documentation and have a more natural output filename of '..._step_1.onnx'
      // for the result of the first step in layout transformation
      debug_graph_fn = [counter = 1, this](const Graph& graph) mutable {
        if (graph.GraphProtoSyncNeeded()) {
          std::basic_ostringstream<ORTCHAR_T> modelpath;
          modelpath << ORT_TSTR("post_layout_transform_step_") << counter << ORT_TSTR(".onnx");
          ORT_THROW_IF_ERROR(Model::Save(*model_, modelpath.str()));
        }

        // counter is used to denote the step, so increment regardless of whether we wrote out the model in this step.
        ++counter;
      };
    }
  }

  LayeringIndex* layering_index = nullptr;
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  std::optional<LayeringIndex> layering_index_storage;
  const auto layering_config = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsLayerAssignmentSettings, "");
  const auto name_based_config = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsNameBasedLayerAssignment, "");
  if (!layering_config.empty() || !name_based_config.empty()) {
    ORT_RETURN_IF_ERROR_SESSIONID_(LayeringIndex::Create(graph, layering_config, name_based_config, {}, execution_providers_,
                                                         *session_logger_, layering_index_storage));
    if (layering_index_storage) {
      layering_index = &layering_index_storage.value();
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  const epctx::ModelGenOptions& ep_context_gen_options = session_options_.GetEpContextGenerationOptions();
  WorkspaceReservationMap workspace_reservations;

  // Do partitioning based on execution providers' capabilities.
  ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.Partition(graph, session_state_->GetMutableFuncMgr(), transform_layout_fn,
                                                       session_options_.config_options, *session_logger_, layering_index,
                                                       mode, ep_context_gen_options,
                                                       session_options_.ep_context_data_read_func != nullptr,
                                                       debug_graph_fn,
                                                       &workspace_reservations));
  graph.SetNodeReplacementCallback(
      [&workspace_reservations](const Graph& modified_graph,
                                gsl::span<const NodeIndex> source_node_indices,
                                NodeIndex destination_node_index) {
        auto graph_it = workspace_reservations.find(&modified_graph);
        if (graph_it != workspace_reservations.end()) {
          ConsolidateWorkspaceReservations(
              graph_it->second, source_node_indices, destination_node_index);
        }
      });
  graph.SetNodeRemovalCallback(
      [&workspace_reservations](const Graph& modified_graph,
                                gsl::span<const NodeIndex> node_indices) {
        auto graph_it = workspace_reservations.find(&modified_graph);
        if (graph_it == workspace_reservations.end()) {
          return;
        }

        for (const NodeIndex node_index : node_indices) {
          graph_it->second.erase(node_index);
        }

        if (graph_it->second.empty()) {
          workspace_reservations.erase(graph_it);
        }
      });
#ifdef ENABLE_TRAINING
  graph.SetNodeCloneCallback(
      [&workspace_reservations](const Graph& modified_graph,
                                NodeIndex source_node_index,
                                NodeIndex cloned_node_index) {
        auto graph_it = workspace_reservations.find(&modified_graph);
        if (graph_it == workspace_reservations.end()) {
          return;
        }

        const auto source_it = graph_it->second.find(source_node_index);
        if (source_it != graph_it->second.end()) {
          graph_it->second.insert_or_assign(cloned_node_index, source_it->second);
        }
      });
#endif
  auto clear_node_mutation_callbacks =
      gsl::finally([&graph]() {
        graph.SetNodeReplacementCallback({});
        graph.SetNodeRemovalCallback({});
#ifdef ENABLE_TRAINING
        graph.SetNodeCloneCallback({});
#endif
      });

#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
  // an EP that prefers BNHS is expected to fuse the Transpose nodes inserted above into its GQA
  // implementation. Report the ones that survived so the resulting cost is diagnosable.
  //
  // Skipped when saving an ORT format model: that runs the partitioner in kAssignOnly mode, which
  // deliberately leaves the original nodes in place instead of compiling or fusing them, so every
  // boundary would be reported as unfused even though the EP will fuse the pattern when the saved
  // model is loaded.
  if (!saving_model_in_ort_format && !converted_gqa_value_boundaries.Empty()) {
    ReportUnfusedGqaValueLayoutTransposes(graph, converted_gqa_value_boundaries, *session_logger_);
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (layering_index) {
    // Layering annotations maybe present even if index is not built although unlikely.
    ORT_RETURN_IF_ERROR_SESSIONID_(graph.RemoveAllLayeringAnnotations());
    // We are currently not using it beyond this point. Clear it to free up memory.
    layering_index = nullptr;
    layering_index_storage.reset();
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  // Decide whether we need to emit the plain optimized (non-EPContext) output model for a compile-only
  // session that compiled no nodes (the kGenerateModel case, e.g. a non-compiling EP such as WebGPU;
  // Partition() serialized only the EPContext form). If so, honor action_if_no_compiled_nodes: the compile
  // API uses kReturnError or kGenerateModel (kDontGenerateModel never reaches a compile-only session).
  //
  // The error is returned regardless of level. For kGenerateModel the serialization point is chosen by the
  // requested optimization level so the output matches what a plain (non-compile) session would produce:
  //   - level < Level2 (includes the Compile API's Default): serialize here, before the Level2+ loop and the
  //     InsertCast/Memcpy transformers below - a BASIC snapshot identical to the graph at the partition
  //     boundary (no Cast, no Memcpy nodes), unchanged from prior behavior.
  //   - level >= Level2 (explicitly requested): serialize after the loop and Memcpy insertion below, so the
  //     emitted model captures the Level2-Level4 fusions (and the device-placement copy nodes).
  //
  // In either case we intentionally do not return early for a compile-only session, so that the disable_cpu_ep_fallback
  // behavior is unchanged. There are two such non-early-return points:
  //   - Here in TransformGraph (the level < Level2 case, where the output model is already serialized): the
  //     InsertCast/Memcpy transformers below are inserted unconditionally, independent of the optimization level, so we
  //     must let them run to keep the transformed graph identical to prior behavior.
  //   - In Initialize() overall: the disable_cpu_ep_fallback check there runs on that fully transformed graph
  //     (including the Cast/Memcpy nodes). Returning early would change what that check sees and could let a
  //     compile-only session with CPU-assigned nodes slip through, which we intentionally want to fail.
  const bool emit_plain_compile_only_model =
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionCompileOnly, "0") == "1" &&
      ep_context_gen_options.enable && !partitioner.AnyEpContextNodesProduced();
  if (emit_plain_compile_only_model) {
    using ActionIfNoCompiledNodes = epctx::ModelGenOptions::ActionIfNoCompiledNodes;
    if (ep_context_gen_options.action_if_no_compiled_nodes == ActionIfNoCompiledNodes::kReturnError) {
      ORT_RETURN_IF_ERROR_SESSIONID_(
          ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "Unable to compile any nodes. Check that the session EPs support compilation "
                          "and can execute at least one subgraph in the model."));
    }
    if (session_options_.graph_optimization_level < TransformerLevel::Level2) {
      ORT_RETURN_IF_ERROR_SESSIONID_(
          epctx::BuildAndSaveOptimizedModel(*model_, ep_context_gen_options, *session_logger_));
    }
  }

  // Get graph optimizations loop level from session config, if not present, set to default value of 1 as per
  // the definition of kOrtSessionOptionsGraphOptimizationsLoopLevel.
  unsigned int graph_optimizations_loop_level = static_cast<unsigned int>(std::stoi(
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsGraphOptimizationsLoopLevel, "1")));

  // Running for an arbitrary number of time to mitigate if graph optimization is stuck in the loop.
  // Generating warning once we have gone through this loop for more than 3 times, it can be changed in the future
  for (int steps = 0; steps < 10; steps++) {
    // Warning that we are running this loop too many times
    if (steps >= 3) {
      LOGS(*session_logger_, WARNING) << "Running graph optimizations in loop " << (steps + 1) << " time/s"
                                      << " (Graph Optimizations Loop Level : " << graph_optimizations_loop_level << ")";
    } else {
      LOGS(*session_logger_, INFO) << "Running graph optimizations in loop " << (steps + 1) << " time/s"
                                   << " (Graph Optimizations Loop Level : " << graph_optimizations_loop_level << ")";
    }

    // Flag to check if applying optimizations should be repeated on basis of if the graph is changed.
    // If graph is not changed it will remain false and we will exit out of this loop.
    bool is_graph_modified = false;

    // Apply Level2 and higher transformers.
    // We do not run Level 1 again as those transformers assume partitioning will run later to do node assignment.
    // Re-Run the Level2+ optimizations. The idea behind re-running Level2 and Level3 graph transforms is that,
    // after the fusion, the nodes are can be in a format which might be supported by other graph transforms which
    // were skipped before. Hence, some of the transforms not applied before is now valid and can be applied to
    // create a more optimal graph for execution.
    ORT_RETURN_IF_ERROR_SESSIONID_(
        apply_transformer_at_level(graph_transformer_mgr_, TransformerLevel::Level2,
                                   *session_logger_, graph,
                                   ((graph_optimizations_loop_level > 1) ? &is_graph_modified : nullptr)));
    ORT_RETURN_IF_ERROR_SESSIONID_(
        apply_transformer_at_level(graph_transformer_mgr_, TransformerLevel::Level3,
                                   *session_logger_, graph,
                                   ((graph_optimizations_loop_level > 1) ? &is_graph_modified : nullptr)));

    // Insert cast node/s.
    {
      // The full list, custom registries included: the transformer decides whether a node has a usable CPU
      // kernel, and a kernel from a custom registry counts. Keeping only the CPU EP's own registry (the last
      // entry, per the design of GetKernelRegistriesByProviderType) would make a node covered by a custom
      // fp16 kernel look kernel-less, so it would be rewritten to fp32 and the custom kernel never used.
      InlinedVector<gsl::not_null<const KernelRegistry*>> kernel_regs =
          kernel_registry_manager_.GetKernelRegistriesByProviderType(kCpuExecutionProvider);

      InsertCastTransformer insert_cast_transformer{"CastFloat16Transformer", std::move(kernel_regs),
                                                    on_partition_assignment_fn};
      ORT_RETURN_IF_ERROR_SESSIONID_(
          apply_transformer_once(insert_cast_transformer, *session_logger_, graph,
                                 ((graph_optimizations_loop_level > 1) ? &is_graph_modified : nullptr)));
    }

    // Level 4 Transforms must be run after Insert Cast Node/s
    ORT_RETURN_IF_ERROR_SESSIONID_(
        apply_transformer_at_level(graph_transformer_mgr_, TransformerLevel::Level4,
                                   *session_logger_, graph,
                                   ((graph_optimizations_loop_level > 0) ? &is_graph_modified : nullptr)));

    // Break if no more optimizations are made
    if (!is_graph_modified) {
      break;
    }
  }

  // Insert copy node/s.
  {
    InlinedVector<gsl::not_null<const IExecutionProvider*>> providers;
    for (auto& provider_ptr : execution_providers_) {
      providers.push_back(provider_ptr.get());
    }

    MemcpyTransformer copy_transformer{std::move(providers), kernel_registry_manager_};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(copy_transformer, *session_logger_, graph));
  }

#ifdef ENABLE_TRAINING
  // Enable memory optimizations.
  // Only applicable for training scenarios.
  {
    const std::string memory_optimizer_config_file =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsMemoryOptimizerApplyConfig, "");
    const std::string probe_config =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsMemoryOptimizerProbeConfig, "0:0");

    MemoryOptimizer mem_transformer{memory_optimizer_config_file, probe_config};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(mem_transformer, *session_logger_, graph));
  }
#endif

  // Second serialization point for the plain optimized (non-EPContext) output model. Reached only for an
  // explicitly requested level >= Level2 (level < Level2 was already serialized before the loop above). This runs
  // after all graph transforms.
  // action_if_no_compiled_nodes was already handled before the loop (kReturnError returned there), so only
  // kGenerateModel reaches here.
  if (emit_plain_compile_only_model &&
      session_options_.graph_optimization_level >= TransformerLevel::Level2) {
    ORT_RETURN_IF_ERROR_SESSIONID_(
        epctx::BuildAndSaveOptimizedModel(*model_, ep_context_gen_options, *session_logger_));
  }

  session_state_->SetWorkspaceReservations(std::move(workspace_reservations));
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

static Status LoadOrtModelBytes(const PathString& model_uri,
                                gsl::span<const uint8_t>& bytes,
                                std::vector<uint8_t>& bytes_data_holder) {
  size_t num_bytes = 0;
  ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(model_uri.c_str(), num_bytes));

  bytes_data_holder.resize(num_bytes);

  std::ifstream bytes_stream(model_uri.c_str(), std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(bytes_data_holder.data()), num_bytes);

  if (!bytes_stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Load model from ", ToUTF8String(model_uri), " failed. Only ",
                           bytes_stream.gcount(), "/", num_bytes, " bytes were able to be read.");
  }

  bytes = gsl::span<const uint8_t>(bytes_data_holder.data(), num_bytes);

  return Status::OK();
}

static Status LoadOrtModelBytesMapped(const PathString& model_uri,
                                      gsl::span<const uint8_t>& bytes,
                                      Env::MappedMemoryPtr& mapped_memory) {
  size_t num_bytes = 0;
  ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(model_uri.c_str(), num_bytes));
  ORT_RETURN_IF(num_bytes == 0, "Cannot memory-map an empty file: ", ToUTF8String(model_uri));

  ORT_RETURN_IF_ERROR(Env::Default().MapFileIntoMemory(model_uri.c_str(), 0, num_bytes, mapped_memory));

  bytes = gsl::span<const uint8_t>(reinterpret_cast<const uint8_t*>(mapped_memory.get()), num_bytes);

  return Status::OK();
}

Status InferenceSession::LoadOrtModel(const PathString& model_uri) {
  return LoadOrtModelWithLoader(
      [&]() {
        model_location_ = model_uri;

        const auto& config_options = GetSessionOptions().config_options;
        const bool use_mmap =
            config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseMemoryMappedOrtModel, "0") == "1";

        if (use_mmap) {
          ORT_RETURN_IF_ERROR(
              LoadOrtModelBytesMapped(model_location_, ort_format_model_bytes_, ort_format_model_mapped_memory_));
          LOGS(*session_logger_, INFO) << "ORT model loaded via memory-mapped I/O.";
          return Status::OK();
        }

        ORT_RETURN_IF_ERROR(
            LoadOrtModelBytes(model_location_, ort_format_model_bytes_, ort_format_model_bytes_data_holder_));
        return Status::OK();
      });
}

Status InferenceSession::LoadOrtModel(const void* model_data, int model_data_len) {
  return LoadOrtModelWithLoader([&]() {
    const auto& config_options = GetSessionOptions().config_options;

    if (config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseMemoryMappedOrtModel, "0") == "1") {
      LOGS(*session_logger_, WARNING) << "session.use_memory_mapped_ort_model is ignored when loading from a buffer.";
    }

    const auto use_ort_model_bytes_directly =
        config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseORTModelBytesDirectly, "0") == "1";

    if (!use_ort_model_bytes_directly) {
      // copy bytes as we need them to be available when InferenceSession::Initialize is called later.
      ort_format_model_bytes_data_holder_.resize(model_data_len);
      std::copy_n(reinterpret_cast<const uint8_t*>(model_data), model_data_len,
                  ort_format_model_bytes_data_holder_.data());
      ort_format_model_bytes_ = gsl::span<const uint8_t>(ort_format_model_bytes_data_holder_.data(), model_data_len);
    } else {
      // Use the model_data directly to reduce memory consumption
      // This will require the model_data to be alive until the InferenceSession is initialized
      ort_format_model_bytes_ = gsl::span<const uint8_t>(reinterpret_cast<const uint8_t*>(model_data), model_data_len);
    }
    return Status::OK();
  });
}

Status InferenceSession::LoadOrtModelWithLoader(std::function<Status()> load_ort_format_model_bytes) {
#if defined(_WIN32) || defined(ORT_USE_TELEMETRY)
  const Env& env = Env::Default();
#endif
#if defined(ORT_USE_TELEMETRY)
  const TimePoint tp = std::chrono::high_resolution_clock::now();
#endif
#if defined(_WIN32) || defined(ORT_USE_TELEMETRY)
  env.GetTelemetryProvider().LogModelLoadStart(session_id_);
#endif

  Status status = Status::OK();
  ORT_TELEMETRY_CAPTURE_STATUS_BEGIN(status)
  std::lock_guard<std::mutex> l(session_mutex_);

  if (is_model_loaded_) {  // already loaded
    const Status load_status(
        common::ONNXRUNTIME, common::MODEL_LOADED, "This session already contains a loaded model.");
    LOGS(*session_logger_, ERROR) << load_status.ErrorMessage();
    return load_status;
  }

  if (is_inited_) {
    const Status initialized_status(
        common::ONNXRUNTIME, common::MODEL_LOADED, "This session has already been initialized.");
    LOGS(*session_logger_, ERROR) << initialized_status.ErrorMessage();
    return initialized_status;
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

  int model_version = -1;
  const std::string_view model_version_string = fbs_ort_model_version->string_view();
  int parsed_model_version = 0;
  const auto [model_version_end, model_version_error] =
      std::from_chars(model_version_string.data(),
                      model_version_string.data() + model_version_string.size(),
                      parsed_model_version);
  if (model_version_error == std::errc{} &&
      model_version_end == model_version_string.data() + model_version_string.size()) {
    model_version = parsed_model_version;
  }
  const bool is_supported = IsOrtModelVersionSupported(model_version);

  OrtFormatLoadOptions load_options{};
  const auto& config_options = session_options_.config_options;
  if (is_supported &&
      config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableSavedRuntimeOptimizations, "0") == "1") {
    load_options.ignore_saved_runtime_optimizations = false;
  }

#if defined(ORT_MINIMAL_BUILD)
  // Note about the ORT format version 5 breaking change.
  // TODO This change was introduced in 1.13. Remove this note a few releases later, e.g., 1.15.
  constexpr auto* kOrtFormatVersion5BreakingChangeNote =
      "This build doesn't support ORT format models older than version 5. "
      "See: https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ORT_Format_Update_in_1.13.md";

  ORT_RETURN_IF(!is_supported,
                "The ORT format model version [", fbs_ort_model_version->string_view(),
                "] is not supported in this build ", ORT_VERSION, ". ",
                kOrtFormatVersion5BreakingChangeNote);
#else   // ^^ defined(ORT_MINIMAL_BUILD) ^^ / vv !defined(ORT_MINIMAL_BUILD) vv
  const auto has_saved_runtime_optimizations = [](const fbs::InferenceSession& fbs_session) -> bool {
    if (const auto* fbs_model = fbs_session.model()) {
      if (const auto* fbs_graph = fbs_model->graph()) {
        if (const auto* fbs_runtime_opts = fbs_graph->runtime_optimizations()) {
          if (const auto* fbs_runtime_opt_records = fbs_runtime_opts->records()) {
            return fbs_runtime_opt_records->size() > 0;
          }
        }
      }
    }
    return false;
  };

  // models prior to v5 can be handled by inserting the kernel constraints in a full build
  const bool is_supported_with_update = model_version < 5;

  if (is_supported_with_update && has_saved_runtime_optimizations(*fbs_session)) {
    LOGS(*session_logger_, WARNING)
        << "The old ORT format model (version " << fbs_ort_model_version->string_view()
        << ") has saved runtime optimizations. They will be ignored.";
    load_options.ignore_saved_runtime_optimizations = true;
  }

  ORT_RETURN_IF_NOT(is_supported || is_supported_with_update,
                    "The ORT format model version [", fbs_ort_model_version->string_view(),
                    "] is not supported in this build ", ORT_VERSION, ".");
#endif  // !defined(ORT_MINIMAL_BUILD)

  const auto* fbs_model = fbs_session->model();
  ORT_RETURN_IF(nullptr == fbs_model, "Missing Model. Invalid ORT format model.");

  // if we're using the bytes directly because kOrtSessionOptionsConfigUseORTModelBytesDirectly was set and the user
  // provided an existing buffer of bytes when creating the InferenceSession, or because we memory-mapped the file,
  // ort_format_model_bytes_data_holder_ will be empty.
  // if that is the case we also allow creating initializers that directly use those bytes.
  using_ort_model_bytes_for_initializers_ =
      load_options.can_use_flatbuffer_for_initializers =
          ort_format_model_bytes_data_holder_.empty() &&
          config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseORTModelBytesForInitializers, "0") == "1";

  // need to go from unique_ptr to shared_ptr when moving into model_
  std::unique_ptr<Model> tmp_model;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model,
                                               HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                               load_options, *session_logger_, tmp_model));
#else
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model, load_options, *session_logger_, tmp_model));
#endif

  ORT_RETURN_IF_ERROR(SaveModelMetadata(*tmp_model));
  model_ = std::move(tmp_model);

  KernelTypeStrResolver kernel_type_str_resolver{};
  if (const auto* fbs_kernel_type_str_resolver = fbs_session->kernel_type_str_resolver();
      fbs_kernel_type_str_resolver != nullptr) {
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.LoadFromOrtFormat(*fbs_kernel_type_str_resolver));
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    // insert the kernel type constraints if we're updating an old model that had kernel hashes.
    if (is_supported_with_update) {
      ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterGraphNodeOpSchemas(model_->MainGraph()));
    }
#endif
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(
      kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(
          kernel_type_str_resolver));
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  kernel_registry_manager_.SetKernelTypeStrResolver(std::move(kernel_type_str_resolver));

  is_model_loaded_ = true;
  ORT_TELEMETRY_CAPTURE_STATUS_END();

#if defined(ORT_USE_TELEMETRY)
  env.GetTelemetryProvider().LogModelLoadEnd(session_id_, status, TimeDiffMicroSeconds(tp));
#elif defined(_WIN32)
  env.GetTelemetryProvider().LogModelLoadEnd(session_id_, status, 0);
#endif
  return status;
}

bool InferenceSession::IsInitialized() const {
  std::lock_guard<std::mutex> l(session_mutex_);
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

#if !defined(ORT_MINIMAL_BUILD)
[[maybe_unused]] static std::string ModelWeightDataType(const Graph& graph) {
  std::string data_type_list;
  auto weight_freq = graph.GetWeightDataTypeFrequency();

  for (size_t i = 0; i < weight_freq.size(); ++i) {
    if (weight_freq[i] > 0) {
      if (!data_type_list.empty()) {
        data_type_list += ", ";
      }
      data_type_list += TensorProto_DataType_Name(static_cast<int>(i));
      data_type_list += ": ";
      data_type_list += std::to_string(weight_freq[i]);
    }
  }

  return data_type_list;
}
#endif

#ifdef _WIN32
[[maybe_unused]] static std::size_t GetStringHash(const std::string& string, std::size_t prev_hash) {
  std::size_t hash = 0;
  std::hash<std::string> hasher;
  const uint64_t golden_ratio = 0x9e3779b9;

  /*
    Combine the current string's hash into the final hash using a mixing function.
    The mixing function ensures that the order of the string affects the final hash
    and reduces the likelihood of hash collisions.
    Here's the breakdown:
    - hasher(string): The hash of the current string being processed.
    - 0x9e3779b9: A constant derived from the golden ratio, often used in hash functions
                  to improve distribution and reduce collisions.
    - (prev_hash << 6) + (prev_hash >> 2): A bitwise operation that shifts the bits to
                                           introduce additional entropy.
  */

  hash = hasher(string) + golden_ratio + (prev_hash << 6) + (prev_hash >> 2);
  return hash;
}
#endif

#ifdef _WIN32
[[maybe_unused]] static std::string ComputeModelGraphHash(const Graph& graph) {
  // Skip hashing if the graph contains an EPContext node.
  const auto& nodes = graph.Nodes();
  for (const auto& node : nodes) {
    if (node.OpType() == "EPContext") {
      return "0";
    }
  }

  // Graph Hash
  std::size_t final_hash = 0;
  const std::size_t node_hash_count = TelemetrySampleCount;
  const std::size_t total_nodes = graph.NumberOfNodes();
  const std::size_t node_step = (total_nodes > node_hash_count) ? (total_nodes / node_hash_count) : 1;

  size_t index = 0;
  for (const auto& node : nodes) {
    if (index % node_step != 0) {
      ++index;
      continue;
    }

    // Combine the hash of each node component using GetStringHash
    final_hash = GetStringHash(node.Name(), final_hash);
    final_hash = GetStringHash(node.OpType(), final_hash);
    final_hash = GetStringHash(node.Domain(), final_hash);

    // Hash the input definitions
    for (const auto& input : node.InputDefs()) {
      if (input->Exists()) {
        final_hash = GetStringHash(input->Name(), final_hash);
      }
    }

    // Hash the output definitions
    for (const auto& output : node.OutputDefs()) {
      if (output->Exists()) {
        final_hash = GetStringHash(output->Name(), final_hash);
      }
    }

    ++index;
  }

  // Convert the final hash to a string
  std::ostringstream hash_stream;
  hash_stream << std::hex << final_hash;
  return hash_stream.str();
}
#endif

#ifdef _WIN32
[[maybe_unused]] static std::string ComputeModelWeightHash(const InitializedTensorSet& initializers) {
  std::size_t final_hash = 0;
  const std::size_t node_hash_count = TelemetrySampleCount;

  // Weight Hash
  const size_t total_initializers = initializers.size();
  const size_t initializer_step = (total_initializers > node_hash_count) ? (total_initializers / node_hash_count) : 1;

  size_t index = 0;
  for (const auto& [tensor_name, tensor] : initializers) {
    if (index % initializer_step != 0) {
      ++index;
      continue;
    }

    // Combine the hash of each tensor component using GetStringHash
    final_hash = GetStringHash(tensor_name, final_hash);

    if (tensor->has_data_type()) {
      final_hash = GetStringHash(std::to_string(tensor->data_type()), final_hash);
    }

    if (tensor->has_raw_data()) {
      final_hash = GetStringHash(tensor->raw_data(), final_hash);
    }

    ++index;
  }

  // Convert the final hash to a string
  std::ostringstream hash_stream;
  hash_stream << std::hex << final_hash;
  return hash_stream.str();
}
#endif

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

#if !defined(ORT_MINIMAL_BUILD)
Status onnxruntime::InferenceSession::CreateNodeStatsRecorder(const std::filesystem::path& node_stats_file) {
  if (node_stats_recorder_.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The session already has an instance of NodeStatsRecorder");
  }
  node_stats_recorder_.emplace(node_stats_file);
  return Status::OK();
}
#endif

namespace {
Status PartitionOrtFormatModel(onnxruntime::Graph& graph,
                               const ExecutionProviders& providers,
                               KernelRegistryManager& kernel_registry_manager,
                               SessionState& session_state,
                               const SessionOptions& sess_options,
                               const logging::Logger& logger) {
#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
  // The BNHS GroupQueryAttention Value layout is applied by TransformGraph, which the ORT format
  // load path does not run. Silently ignoring the option would leave the session expecting BNSH
  // while the application supplies BNHS: with dynamic or coincidentally square cache dimensions
  // that passes input validation and produces wrong results. Reject it instead.
  // An ORT format model that already had the transform applied at conversion time carries the BNHS
  // boundary shapes in the model itself and must be loaded without setting this option.
  //
  // Validate the value before applying the format restriction, so that a typo is reported as a bad
  // option value naming the accepted ones, rather than as an ORT format limitation.
  std::string gqa_value_layout;
  bool gqa_value_layout_explicitly_set = false;
  ORT_RETURN_IF_ERROR(GetGqaValueLayout(sess_options.config_options, gqa_value_layout,
                                        gqa_value_layout_explicitly_set));
  if (gqa_value_layout != kGqaValueLayoutBNSH) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Session option '", kOrtSessionOptionsGqaValueLayout,
                           "' is not supported for ORT format models. Apply the Value layout transform when "
                           "converting the model to ORT format and load it without setting this option, or load the "
                           "ONNX model instead.");
  }

  // Detected before partitioning, while the GQA nodes are still there to anchor on. A model converted
  // to ORT format after the transform was applied carries the Transposes and the BNHS boundary shapes.
  //
  // The diagnostic at the end of this function wants them whether or not the option was set, so that
  // a BNHS-converted model loaded without it is still reported.
  const auto converted_gqa_value_boundaries = FindConvertedGqaValueLayoutBoundaries(graph);
  const bool has_converted_gqa_value_boundaries = !converted_gqa_value_boundaries.Empty();

  // An explicit BNSH request is a claim about the boundary and has to hold here too, or an
  // application trusting the option would bind BNSH buffers against a BNHS boundary. An absent option
  // makes no claim: loading a converted model without setting anything is the documented way to use
  // BNHS with ORT format, so it stays allowed.
  if (gqa_value_layout_explicitly_set && has_converted_gqa_value_boundaries) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "This ORT format model already carries the BNHS GroupQueryAttention Value layout. "
        "It cannot be loaded with '",
        kOrtSessionOptionsGqaValueLayout, "' set to '", kGqaValueLayoutBNSH,
        "', because the application would bind BNSH buffers to a BNHS boundary. "
        "Leave the option unset and bind BNHS buffers, or load a model whose Value cache boundary is BNSH.");
  }
#endif

  layout_transformation::TransformLayoutFunction transform_layout_fn = nullptr;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // only provide NCWH to NHWC layout transformer if supported
  if (layout_transformation::IsSupportedOpset(graph)) {
    transform_layout_fn =
        [](Graph& graph_to_transform, bool& modified,
           const IExecutionProvider& execution_provider,
           const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
      AllocatorPtr cpu_allocator = CPUAllocator::DefaultInstance();
      return layout_transformation::TransformLayoutForEP(graph_to_transform, modified, execution_provider,
                                                         std::move(cpu_allocator), debug_graph_fn);
    };
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  // Create GraphOptimizerRegistry instance for providing predefined graph optimizers and selection functions for EPs to lookup
  auto graph_optimizer_registry = std::make_unique<GraphOptimizerRegistry>(&sess_options,
                                                                           providers.Get(onnxruntime::kCpuExecutionProvider),
                                                                           &logger);

  GraphPartitioner partitioner(kernel_registry_manager, providers, std::move(graph_optimizer_registry),
                               [&sess_options]() -> bool { return sess_options.IsLoadCancellationFlagSet(); });
  ORT_RETURN_IF_ERROR(partitioner.Partition(graph,
                                            session_state.GetMutableFuncMgr(),
                                            transform_layout_fn,
                                            sess_options.config_options,
                                            logger,
                                            nullptr /*layering_index*/,
                                            GraphPartitioner::Mode::kOrtFormatLoad,
                                            {},
                                            sess_options.ep_context_data_read_func != nullptr));

#if defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
  // kOrtFormatLoad does compile and fuse, unlike the kAssignOnly pass used when writing an ORT format
  // model, so a surviving Transpose here really will execute.
  if (!converted_gqa_value_boundaries.Empty()) {
    ReportUnfusedGqaValueLayoutTransposes(graph, converted_gqa_value_boundaries, logger);
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // a compiling EP (e.g. CoreML) may copy initializers to its own memory. run the cleanup of unused initializers
  // so that they can be freed.
  ORT_RETURN_IF_ERROR(graph.RemovedUnusedInitializersOrtFormat());
#endif
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Status ApplyOrtFormatModelRuntimeOptimizations(
    onnxruntime::Graph& graph, const logging::Logger& logger, const SessionOptions& session_options,
    const InlinedHashSet<std::string>& optimizers_to_disable, const IExecutionProvider& cpu_ep,
    concurrency::ThreadPool* intra_op_thread_pool) {
  bool modified = false;

  for (int level = static_cast<int>(TransformerLevel::Level2);
       level <= static_cast<int>(session_options.graph_optimization_level);
       ++level) {
    const auto transformers = optimizer_utils::GenerateTransformersForMinimalBuild(
        static_cast<TransformerLevel>(level), session_options, SatRuntimeOptimizationLoadContext{}, cpu_ep, logger,
        optimizers_to_disable, intra_op_thread_pool);

    for (const auto& transformer : transformers) {
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
    }
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

std::basic_string<ORTCHAR_T> GenerateProfileFilePath(std::basic_string<ORTCHAR_T> profile_file_prefix) {
  profile_file_prefix.append(ORT_TSTR("_"));
  profile_file_prefix.append(GetCurrentTimeString<ORTCHAR_T>());
  profile_file_prefix.append(ORT_TSTR(".json"));
  return profile_file_prefix;
}

}  // namespace

static void ResolveMemoryPatternFlags(SessionState& session_state) {
  session_state.ResolveMemoryPatternFlag();

  for (const auto& entry : session_state.GetSubgraphSessionStateMap()) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      ResolveMemoryPatternFlags(*name_to_subgraph_session_state.second);
    }
  }
}

// This function is called when the session is being initialized.
// For now, this function only checks for invalid combination of DML EP with other EPs.
// TODO: extend this function to check for other invalid combinations of EPs.
common::Status InferenceSession::HasInvalidCombinationOfExecutionProviders() const {
  // DML EP is not allowed with other GPU or NPU EPs.
  // historical reason for this is unknown. relaxing the limit that it must only be used with the CPU EP to support
  // scenarios where alternative EPs are CPU based (e.g. openvino).
  bool has_dml_ep = execution_providers_.Get(kDmlExecutionProvider) != nullptr;
  if (has_dml_ep) {
    for (const auto& ep : execution_providers_) {
      if (ep->Type() != kDmlExecutionProvider && ep->GetDevice().Type() != OrtDevice::CPU) {
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "DML EP can only be used with CPU EPs.");
      }
    }
  }
  return Status::OK();
}

common::Status InferenceSession::Initialize() {
  const auto start_timing = [this]() {
    TimePoint start_time{};
#if defined(ORT_USE_TELEMETRY)
    start_time = std::chrono::high_resolution_clock::now();
#endif
    if (session_profiler_.IsEnabled()) {
      start_time = session_profiler_.Start();
    }
#if defined(_WIN32) || defined(ORT_USE_TELEMETRY)
    Env::Default().GetTelemetryProvider().LogSessionCreationStart(session_id_);
#endif
    return start_time;
  };

  Status status = Status::OK();
  bool have_cpu_ep = false;
  if (session_options_.IsLoadCancellationFlagSet()) {
    status = ORT_MAKE_STATUS(
        ONNXRUNTIME, MODEL_LOAD_CANCELED,
        "Session initialization canceled due to user request.");
  } else {
    std::lock_guard<std::mutex> initial_guard(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      status = Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded.");
    } else if (is_inited_) {
      LOGS(*session_logger_, INFO) << "Session has already been initialized.";
      return common::Status::OK();
    } else {
#if !defined(ORT_ENABLE_GQA_VALUE_LAYOUT)
      for (const auto& [key, value] : session_options_.config_options.GetConfigOptionsMap()) {
        if (key == kOrtSessionOptionsGqaValueLayout) {
          status = Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "GQA layout disabled");
          break;
        }
      }
#endif
      if (status.IsOK()) {
        have_cpu_ep = execution_providers_.Get(onnxruntime::kCpuExecutionProvider) != nullptr;
      }
    }
  }

  const TimePoint tp = start_timing();
  if (!status.IsOK()) {
    return RecordSessionCreationEndTelemetry(tp, status);
  }

  ORT_TRY {
    ORT_TELEMETRY_CAPTURE_STATUS_BEGIN(status)
    LOGS(*session_logger_, INFO) << "Initializing session.";
#if defined(ORT_MINIMAL_BUILD)
    for (const auto& [key, value] : session_options_.config_options.GetConfigOptionsMap()) {
      const std::string_view option = key;
      if (((option == kOrtSessionOptionsConfigEnableMoeExpertCounting ||
            option == kOrtSessionOptionsConfigEnableMoeExpertStatistics) &&
           value != "0") ||
          option == kOrtSessionOptionsConfigMoeExpertCounterStateFile ||
          option == kOrtSessionOptionsConfigMoeExpertCounterAlpha ||
          option == kOrtSessionOptionsConfigMoeExpertCounterBeta) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT, key, " is not supported in a minimal build.");
      }
    }
#else
    const std::string& enable_moe_statistics =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "0");
    if (enable_moe_statistics != "0" && enable_moe_statistics != "1") {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT, kOrtSessionOptionsConfigEnableMoeExpertStatistics,
          " must be set to either \"0\" or \"1\". Received: \"", enable_moe_statistics, "\".");
    }
    const bool enable_moe_expert_statistics = enable_moe_statistics == "1";
    const auto enable_moe_counting =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableMoeExpertCounting, "0");
    ORT_RETURN_IF_NOT(enable_moe_counting == "0" || enable_moe_counting == "1",
                      kOrtSessionOptionsConfigEnableMoeExpertCounting, " must be \"0\" or \"1\".");
    const bool enable_moe_expert_counting = enable_moe_counting == "1";
    const auto moe_counter_state_file =
        session_options_.config_options.GetConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile);
    if (moe_counter_state_file) {
      ORT_RETURN_IF_NOT(enable_moe_expert_counting || enable_moe_expert_statistics,
                        kOrtSessionOptionsConfigMoeExpertCounterStateFile,
                        " requires expert counting or statistics logging to be enabled.");
      ORT_RETURN_IF(moe_counter_state_file->empty(),
                    kOrtSessionOptionsConfigMoeExpertCounterStateFile, " must not be empty.");
    }
    for (const char* key : {kOrtSessionOptionsConfigMoeExpertCounterAlpha,
                            kOrtSessionOptionsConfigMoeExpertCounterBeta}) {
      ORT_RETURN_IF(session_options_.config_options.GetConfigEntry(key).has_value() &&
                        !enable_moe_expert_counting && !enable_moe_expert_statistics,
                    key, " requires expert counting or statistics logging to be enabled.");
    }
    if (enable_moe_expert_counting) {
      for (const auto& execution_provider : execution_providers_) {
        ORT_RETURN_IF(execution_provider->Type() == kCudaExecutionProvider &&
                          execution_provider->GetOrtEp() != nullptr,
                      "MoE expert counting is not supported by the CUDA plugin execution provider.");
        ORT_RETURN_IF(execution_provider->IsGraphCaptureEnabled(),
                      "MoE expert counting is not supported when graph capture is enabled.");
      }
    }
    if (enable_moe_expert_statistics) {
      for (const auto& execution_provider : execution_providers_) {
        if (execution_provider->Type() == kCudaExecutionProvider &&
            execution_provider->GetOrtEp() != nullptr) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT, kOrtSessionOptionsConfigEnableMoeExpertStatistics,
              "=1 is not supported by the CUDA plugin execution provider.");
        }
        if (execution_provider->IsGraphCaptureEnabled()) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT, kOrtSessionOptionsConfigEnableMoeExpertStatistics,
              "=1 is not supported when graph capture is enabled by ", execution_provider->Type(), ".");
        }
      }
    }
#endif

    // Verify that there are no external initializers in the graph if external data is disabled.
    onnxruntime::Graph& graph = model_->MainGraph();

#ifdef DISABLE_EXTERNAL_INITIALIZERS
    const InitializedTensorSet& initializers = graph.GetAllInitializedTensors();
    for (const auto& it : initializers) {
      if (utils::HasExternalData(*it.second) && !utils::HasExternalDataInMemory(*it.second)) {
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
      execution_providers_.SetCpuProviderWasImplicitlyAdded(true);
    }

    // Check for the presence of an invalid combination of execution providers in the session
    // For e.g. we don't support DML EP and other GPU EPs to be present in the same session
    // This check is placed here because it serves as a common place for all language bindings.
    ORT_RETURN_IF_ERROR_SESSIONID_(HasInvalidCombinationOfExecutionProviders());

    // re-acquire mutex
    std::lock_guard<std::mutex> l(session_mutex_);

    auto clear_external_data_loaders = gsl::finally([this] { external_data_loader_mgr_.Clear(); });
    for (const auto& provider : execution_providers_) {
      if (auto loader = provider->GetExternalDataLoader()) {
        ORT_RETURN_IF_ERROR_SESSIONID_(external_data_loader_mgr_.RegisterExternalDataLoader(std::move(loader)));
      }
    }

#if !defined(DISABLE_EXTERNAL_INITIALIZERS) && !defined(ORT_MINIMAL_BUILD)
    if (!session_options_.external_initializers.empty()) {
      ORT_RETURN_IF_ERROR_SESSIONID_(graph.InjectExternalInitializedTensors(session_options_.external_initializers));
      InlinedHashMap<std::string, OrtValue>{}.swap(session_options_.external_initializers);
    }

    if (!session_options_.external_initializer_files_mmap.empty()) {
      const bool use_buffers_directly = session_options_.config_options.GetConfigOrDefault(
                                            kOrtSessionOptionsConfigUseExternalInitializerFileBuffersDirectly, "0") ==
                                        "1";
      ORT_RETURN_IF_ERROR_SESSIONID_(
          graph.InjectExternalInitializersFromFilesInMemory(session_options_.external_initializer_files_mmap,
                                                            use_buffers_directly));
      InlinedHashMap<std::basic_string<ORTCHAR_T>, std::pair<char*, size_t>>{}.swap(
          session_options_.external_initializer_files_mmap);
    }
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    TraceLoggingWriteStart(session_activity, "OrtInferenceSessionActivity");
    session_activity_started_ = true;
#endif
    // Generate and cache telemetry data for the model when caller framework is WinAI
    std::string model_weight_type, model_graph_hash, model_weight_hash;
#ifdef ORT_CALLER_FRAMEWORK
    if (std::string_view(ORT_CALLER_FRAMEWORK) == "WinAI") {
      InitializedTensorSet initializers = graph.GetAllInitializedTensors();
#if !defined(ORT_MINIMAL_BUILD)
      model_weight_type = ModelWeightDataType(graph);
      SetWeightDataType(model_weight_type);
#endif
#ifdef _WIN32
      // Check if model metadata contains a "model_hash" field
      const auto& metadata = model_->MetaData();
      auto model_hash_it = metadata.find("model_hash");

      if (model_hash_it != metadata.end()) {
        // Use the model_hash from metadata
        model_graph_hash = model_hash_it->second;
        model_weight_hash = model_hash_it->second;
      } else {
        // Compute hashes
        model_graph_hash = ComputeModelGraphHash(graph);
        model_weight_hash = (model_graph_hash == "0") ? "0" : ComputeModelWeightHash(initializers);
      }

      SetGraphHash(model_graph_hash);
      SetWeightHash(model_weight_hash);
#endif
    }
#endif

    // now that we have all the execution providers, create the session state
    session_state_ = std::make_unique<SessionState>(
        model_->MainGraph(),
        execution_providers_,
        GetIntraOpThreadPoolToUse(),
        GetInterOpThreadPoolToUse(),
        data_transfer_mgr_,
        external_data_loader_mgr_,
        *session_logger_,
        session_profiler_,
        session_options_,
        prepacked_weights_container_);

    bool use_env_allocators =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseEnvAllocators, "0") == "1";
    if (use_env_allocators) {
      LOGS(*session_logger_, INFO) << "This session will use the allocator registered with the environment.";
      session_state_->UpdateAllocatorsWithEnvAllocators(environment_.GetRegisteredSharedAllocators());
    }

    for (auto& ep : execution_providers_) {
      auto tuning_ctx = ep->GetTuningContext();
      if (nullptr != tuning_ctx) {
        tuning_ctx->RegisterAllocatorsView(&session_state_->GetAllocators());
      }
    }

#if !defined(ORT_MINIMAL_BUILD)
    const std::string node_stats_file = session_options_.config_options.GetConfigOrDefault(
        kOrtSessionOptionsCollectNodeMemoryStatsToFile, "");

    if (!node_stats_file.empty()) {
      ORT_RETURN_IF_ERROR_SESSIONID_(CreateNodeStatsRecorder(node_stats_file));
    }

    session_state_->SetNodeStatsRecorder(GetNodeStatsRecorder());
#endif

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    // Don't want to pollute SessionState constructor since memory profile is enabled optionally.
    session_state_->SetMemoryProfiler(&memory_profiler_);
#endif

    // Collect the kernel registries from execution provider instances;
    // There are 2 kinds of kernel registries with priority from high to low as below,
    // 1. Custom execution provider type specific kernel registries.
    // 2. common execution provider type specific kernel registries.
    // Kernel registries are shared across sessions.
    // The 1st ones should have already been registered via session-level API into KernelRegistryManager.
    //
    // Register 2nd registries into KernelRegistryManager.
    ORT_RETURN_IF_ERROR_SESSIONID_(kernel_registry_manager_.RegisterKernels(execution_providers_));

    const bool loading_ort_format = !ort_format_model_bytes_.empty();
    const bool saving_model = !session_options_.optimized_model_filepath.empty();
    const bool saving_ort_format = [&]() {
      if (saving_model) {
        const std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSaveModelFormat, "");
        const bool has_explicit_type = !model_type.empty();
        return ((has_explicit_type && model_type == "ORT") ||
                (!has_explicit_type &&
                 fbs::utils::IsOrtFormatModel(session_options_.optimized_model_filepath)));
      }
      return false;
    }();

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    ORT_RETURN_IF(
        loading_ort_format &&
            session_options_.config_options.GetConfigOrDefault(
                kOrtSessionOptionsStrictWorkspaceVerification, "0") == "1",
        "session.strict_workspace_verification is not supported when loading an ORT format model because "
        "partition-time workspace reservations are not serialized in the model. Load the ONNX model to use "
        "strict workspace verification.");
#endif

    if (!loading_ort_format) {
#if !defined(ORT_MINIMAL_BUILD)
      const auto minimal_build_opt_config_value = session_options_.config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigMinimalBuildOptimizations, "");
      MinimalBuildOptimizationHandling minimal_build_optimization_handling{};
      ORT_RETURN_IF_ERROR_SESSIONID_(GetMinimalBuildOptimizationHandling(minimal_build_opt_config_value,
                                                                         saving_ort_format,
                                                                         minimal_build_optimization_handling));

      auto record_runtime_optimization_produced_op_schema = [this](const ONNX_NAMESPACE::OpSchema& op_schema) {
        saved_runtime_optimization_produced_node_op_schemas_.insert(&op_schema);
        return Status::OK();
      };

      // Enable DQ->MatMulNBits fusion if the NvTensorRTRTX plugin EP is registered.
      if (execution_providers_.Get(onnxruntime::kNvTensorRTRTXExecutionProvider) != nullptr) {
        if (session_options_.config_options.GetConfigOrDefault(
                kOrtSessionOptionsEnableDQMatMulNBitsFusion, "") == "") {
          ORT_RETURN_IF_ERROR_SESSIONID_(
              session_options_.config_options.AddConfigEntry(
                  kOrtSessionOptionsEnableDQMatMulNBitsFusion, "1"));
        }
      }

      // add predefined transformers
      ORT_RETURN_IF_ERROR_SESSIONID_(AddPredefinedTransformers(graph_transformer_mgr_,
                                                               session_options_.graph_optimization_level,
                                                               minimal_build_optimization_handling,
                                                               record_runtime_optimization_produced_op_schema,
                                                               *session_logger_));

#ifdef USE_DML
      const IExecutionProvider* dmlExecutionProvider = execution_providers_.Get(kDmlExecutionProvider);

      if (dmlExecutionProvider) {
        // DML graph fusion is an important runtime optimization that cannot be done ahead of time; it must be disabled
        // when running in "offline mode" and saving an optimized model to disk. To support users that want to optimize
        // models offline, and then disable graph optimizations when running "online", this transformer ignores the ORT
        // graph optimization level and is generally always applied.
        bool dml_graph_fusion_enabled = session_options_.optimized_model_filepath.empty() &&
                                        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisableDmlGraphFusion, "0") == "0";
        std::string dml_graph_serialization_enabled_config_val = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableGraphSerialization, "0");
        std::transform(dml_graph_serialization_enabled_config_val.begin(),
                       dml_graph_serialization_enabled_config_val.end(),
                       dml_graph_serialization_enabled_config_val.begin(),
                       [](char ch) { return std::tolower(ch); });
        bool dml_graph_serialization_enabled = dml_graph_serialization_enabled_config_val == "true";

        if (static_cast<const Dml::ExecutionProvider*>(dmlExecutionProvider)->IsGraphCaptureEnabled()) {
          std::unique_ptr<onnxruntime::GraphTransformer> dmlRuntimeGraphFusionTransformer = std::make_unique<Dml::DmlRuntimeGraphFusionTransformer>("DmlRuntimeGraphFusionTransformer",
                                                                                                                                                    dmlExecutionProvider);
          if (dmlRuntimeGraphFusionTransformer == nullptr) {
            return Status(common::ONNXRUNTIME, common::FAIL, "DmlRuntimeGraphFusionTransformer is nullptr");
          }
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(dmlRuntimeGraphFusionTransformer), onnxruntime::TransformerLevel::Level3));
        } else if (dml_graph_fusion_enabled) {
          std::unique_ptr<onnxruntime::GraphTransformer> dmlGraphFusionTransformer = std::make_unique<Dml::DmlGraphFusionTransformer>("DmlGraphFusionTransformer",
                                                                                                                                      dmlExecutionProvider,
                                                                                                                                      dml_graph_serialization_enabled);
          if (dmlGraphFusionTransformer == nullptr) {
            return Status(common::ONNXRUNTIME, common::FAIL, "DmlGraphFusionTransformer is nullptr");
          }
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(dmlGraphFusionTransformer), onnxruntime::TransformerLevel::Level3));
        }

        // This transformer applies DML-specific fusions that go beyond what ORT offers by default
        bool dml_operator_fusion_enabled = session_options_.graph_optimization_level >= TransformerLevel::Level2;
        if (dml_operator_fusion_enabled) {
          std::unique_ptr<onnxruntime::GraphTransformer> dmlOperatorFusionTransformer = std::make_unique<Dml::GraphTransformer>("DmlOperatorFusionTransformer",
                                                                                                                                execution_providers_.Get(kDmlExecutionProvider));
          if (dmlOperatorFusionTransformer == nullptr) {
            return Status(common::ONNXRUNTIME, common::FAIL, "DmlOperatorFusionTransformer is nullptr");
          }
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(dmlOperatorFusionTransformer), onnxruntime::TransformerLevel::Level2));
        }

        const auto dml_ep_impl = static_cast<const Dml::ExecutionProvider*>(dmlExecutionProvider);
        auto is_mcdm_device = dml_ep_impl->GetImpl()->IsMcdmDevice();
        if (is_mcdm_device) {
          const InlinedHashSet<std::string_view> dml_ep = {onnxruntime::kDmlExecutionProvider};
          auto stft_decomposition_transformer = std::make_unique<STFTDecomposition>(dml_ep);
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(stft_decomposition_transformer), onnxruntime::TransformerLevel::Level1));
        }
      }
#endif

      // apply any transformations to the main graph and any subgraphs
      ORT_RETURN_IF_ERROR_SESSIONID_(TransformGraph(graph, saving_ort_format));

      // now that all the transforms are done, call Resolve on the main graph. this will recurse into the subgraphs.
      ORT_RETURN_IF_ERROR_SESSIONID_(graph.Resolve());
      if (session_options_.IsLoadCancellationFlagSet()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                               "Session initialization canceled due to user request.");
      }

      // Check if any EP is configured for graph capture (e.g., CUDA Graph, DML Graph).
      // If so, validate the graph and cache the EP for triggering ReplayGraph() in Run().
      for (const auto& ep : execution_providers_) {
        if (!ep->IsGraphCaptureEnabled()) {
          continue;
        }

        // Graph capture can't work with control flow nodes
        if (HasControlflowNodes(graph)) {
          LOGS(*session_logger_, ERROR) << "This session cannot use the graph capture feature as requested by the user "
                                        << "as the model has control flow nodes which can't be supported by "
                                        << ep->Type();
          ORT_RETURN_IF_ERROR_SESSIONID_(
              ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                              "This session cannot use the graph capture feature as requested by the user "
                              "as the model has control flow nodes which can't be supported by " +
                                  ep->Type()));
        }

        auto policy = ep->GetGraphCaptureNodeAssignmentPolicy();
        if (policy == OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES) {
          // Ensure that all nodes have been partitioned to the EP or CPU EP && there are no memcpy nodes.
          // The reasoning is that certain shape nodes will be forced onto CPU and as long as there are
          // no memcpy nodes this is confirmation that no compute nodes have been placed on the CPU EP.
          if (!AreAllComputeNodesAssignedToEpOrCpu(graph, ep->Type())) {
            LOGS(*session_logger_, ERROR) << "This session cannot use the graph capture feature as requested by the user "
                                          << " as all compute graph nodes have not been partitioned to the "
                                          << ep->Type();
            ORT_RETURN_IF_ERROR_SESSIONID_(
                ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                "This session cannot use the graph capture feature as requested by the user "
                                " as all compute graph nodes have not been partitioned to the " +
                                    ep->Type()));
          }

          // Log a warning for the user to know that there are shape subgraphs that will execute on CPU
          if (HasShapeSubgraphNodes(graph)) {
            LOGS(*session_logger_, WARNING) << "This model has shape massaging nodes that will execute on CPU. "
                                            << "Use the graph capture feature with caution. "
                                            << "As long as the intermediate shapes produced in the model "
                                            << "using the representative input used to capture the graph, "
                                            << "will match the shapes produced in the model for other inputs "
                                            << "of the same shape as the representative input (common case), "
                                            << "it is safe to use the graph capture feature.";
          }
        } else {
          // AllNodesOnEp: all nodes in the main graph must be assigned to this EP.
          if (!AreAllNodesInMainGraphAssignedToOneEp(graph, ep->Type())) {
            LOGS(*session_logger_, ERROR) << "This session cannot use the graph capture feature as requested by the user "
                                          << "as all the graph nodes have not been assigned to "
                                          << ep->Type();
            ORT_RETURN_IF_ERROR_SESSIONID_(
                ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                "This session cannot use the graph capture feature as requested by the user "
                                "as all the graph nodes have not been assigned to " +
                                    ep->Type()));
          }
        }

        LOGS(*session_logger_, INFO) << "This session will use the graph capture feature as requested by the user "
                                     << "with " << ep->Type() << ".";
        cached_execution_provider_for_graph_replay_.SetExecutionProvider(ep.get());
        break;  // Only one EP can use graph capture.
      }

      const bool disable_cpu_ep_fallback = session_options_.config_options.GetConfigOrDefault(
                                               kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

      // Handle the option to disable the fallback of graph nodes to the CPU EP.
      // If the user disabled fallback, but also explicitly added the CPU EP to the session, return an error status.
      // If the user disabled fallback and any graph node is assigned to the CPU EP, return an error status.
      if (disable_cpu_ep_fallback) {
        if (!execution_providers_.GetCpuProviderWasImplicitlyAdded()) {
          const char* err_msg =
              "Conflicting session configuration: explicitly added the CPU EP to the "
              "session, but also disabled fallback to the CPU EP via session configuration options.";

          LOGS(*session_logger_, ERROR) << err_msg;
          ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err_msg));
        } else if (inference_session_utils::AreAnyNodesAssignedToCpuEp(graph)) {
          const char* err_msg =
              "This session contains graph nodes that are assigned to the default CPU EP, "
              "but fallback to CPU EP has been explicitly disabled by the user.";
          LOGS(*session_logger_, ERROR) << err_msg;
          ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, err_msg));
        }
      }

      // Update temporary copies of metadata, input- and output definitions to the same state as the resolved graph
      ORT_RETURN_IF_ERROR_SESSIONID_(SaveModelMetadata(*model_));
#else   // !defined(ORT_MINIMAL_BUILD)
      ORT_RETURN_IF_ERROR_SESSIONID_(
          ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "Loading anything other than ORT format models is not enabled in this build."));
#endif  // !defined(ORT_MINIMAL_BUILD)
    } else {
      ORT_RETURN_IF_ERROR_SESSIONID_(PartitionOrtFormatModel(graph, execution_providers_, kernel_registry_manager_,
                                                             *session_state_, session_options_, *session_logger_));

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      const auto& cpu_ep = *execution_providers_.Get(onnxruntime::kCpuExecutionProvider);
      ORT_RETURN_IF_ERROR_SESSIONID_(
          ApplyOrtFormatModelRuntimeOptimizations(graph, *session_logger_, session_options_, optimizers_to_disable_,
                                                  cpu_ep, GetIntraOpThreadPoolToUse()));
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    }

    // Compile-only: a compile-only session never runs inference, so skip session-state finalization (kernel creation,
    // PrePack, initializer upload, memory planning) and early return here.
    //
    // Returning here also bypasses the model-save block below, because a compile-only session does not produce its
    // output via optimized_model_filepath. We still log the session-creation telemetry and record the
    // end-of-initialization bookkeeping that the normal path emits, so neither is lost by returning early.
    if (session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionCompileOnly, "0") == "1") {
      LOGS(*session_logger_, INFO)
          << "Compile-only session: skipping session-state finalization. The session is not runnable.";
#if defined(ORT_USE_TELEMETRY)
      LogSessionCreationTelemetry(graph, model_weight_type, model_graph_hash, model_weight_hash);
      return Status::OK();
#else
      return RecordSessionCreationEndTelemetry(tp, status);
#endif
    }

    ORT_RETURN_IF_ERROR_SESSIONID_(
        session_state_->FinalizeSessionState(model_location_, kernel_registry_manager_,
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
        const std::string optimized_model_external_initializers_file_name =
            session_options_.config_options.GetConfigOrDefault(
                kOrtSessionOptionsOptimizedModelExternalInitializersFileName, "");
        if (optimized_model_external_initializers_file_name.empty()) {
          ORT_RETURN_IF_ERROR_SESSIONID_(Model::Save(*model_, session_options_.optimized_model_filepath));
        } else {
          const size_t optimized_model_external_initializers_min_size_in_bytes =
              ParseStringWithClassicLocale<size_t>(session_options_.config_options.GetConfigOrDefault(
                  kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, "1024"));
          ModelSavingOptions model_saving_options{optimized_model_external_initializers_min_size_in_bytes};
          model_saving_options.align_offset = true;
          ORT_RETURN_IF_ERROR_SESSIONID_(Model::SaveWithExternalInitializers(*model_,
                                                                             session_options_.optimized_model_filepath,
                                                                             optimized_model_external_initializers_file_name,
                                                                             model_saving_options));
        }
      }
    }

    std::vector<TuningResults> tuning_results;
    bool found_tuning_results = false;
    ORT_RETURN_IF_ERROR_SESSIONID_(inference_session_utils::ParseTuningResultsFromModelMetadata(
        model_metadata_, tuning_results, found_tuning_results, *session_logger_));
    if (found_tuning_results) {
      ORT_RETURN_IF_ERROR_SESSIONID_(SetTuningResults(tuning_results, /*error_on_invalid*/ false, /*auto_enable*/ true));
    }
#endif  // !defined(ORT_MINIMAL_BUILD)

    // Resolve memory pattern flags of the main graph and subgraph session states
    ResolveMemoryPatternFlags(*session_state_);

    is_inited_ = true;

    if (!using_ort_model_bytes_for_initializers_) {
      ort_format_model_bytes_ = gsl::span<const uint8_t>();
      std::vector<uint8_t>().swap(ort_format_model_bytes_data_holder_);
      ort_format_model_mapped_memory_.reset();
    }

    // once the model is saved, we may remove unnecessary attributes for inference
    session_state_->PruneRemovableAttributes();

    // and log telemetry
#if defined(ORT_USE_TELEMETRY)
    LogSessionCreationTelemetry(graph, model_weight_type, model_graph_hash, model_weight_hash);
#endif

    LOGS(*session_logger_, INFO) << "Session successfully initialized.";
    ORT_TELEMETRY_CAPTURE_STATUS_END();
  }

  ORT_CATCH(const NotImplementedException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    });
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ex.Category(), ex.Code(), MakeString("Exception during initialization: ", ex.what()));
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

  return RecordSessionCreationEndTelemetry(tp, status);
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

SessionOptions& InferenceSession::GetMutableSessionOptions() {
  return session_options_;
}

const DataTransferManager& InferenceSession::GetDataTransferManager() const {
  return data_transfer_mgr_;
}

const ExternalDataLoaderManager& InferenceSession::GetExternalDataLoaderManager() const {
  return external_data_loader_mgr_;
}

common::Status InferenceSession::CheckShapes(const std::string& input_output_name, const TensorShape& input_output_shape,
                                             const TensorShape& expected_shape, const char* input_output_moniker) const {
  const auto shape_size = input_output_shape.NumDimensions();
  const auto expected_shape_size = expected_shape.NumDimensions();
  if (shape_size != expected_shape_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid rank for ", input_output_moniker, ": ",
                           input_output_name, " Got: ", shape_size, " Expected: ", expected_shape_size,
                           " Please fix either the inputs/outputs or the model.");
  }

  InlinedVector<size_t> invalid_dim_indices;
  for (size_t i = 0; i < shape_size; ++i) {
    if (expected_shape[i] < 0) {
      continue;  // this represents a symbolic shape dimension
    }
    if (input_output_shape[i] != expected_shape[i]) {
      invalid_dim_indices.push_back(i);
    }
  }

  if (!invalid_dim_indices.empty()) {
    std::ostringstream ostr;
    ostr << "Got invalid dimensions for " << input_output_moniker << ": " << input_output_name << " for the following indices\n";
    for (size_t i = 0, end = invalid_dim_indices.size(); i < end; ++i) {
      size_t idx = invalid_dim_indices[i];
      ostr << " index: " << idx << " Got: " << input_output_shape[idx] << " Expected: " << expected_shape[idx] << "\n";
    }
    ostr << " Please fix either the inputs/outputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }
  return Status::OK();
}

static common::Status CheckTypes(MLDataType actual, MLDataType expected, const std::string& base_type,
                                 const char* input_output_moniker) {
  if (actual == expected) {
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unexpected ", input_output_moniker, " data type. Actual: (",
                         base_type, "(",
                         DataTypeImpl::ToString(actual), ")) , expected: (", base_type, "(",
                         DataTypeImpl::ToString(expected), "))");
}

common::Status InferenceSession::ValidateInputsOutputs(gsl::span<const std::string> names,
                                                       gsl::span<const OrtValue> feeds_fetches,
                                                       const InputOutputDefMetaMap& input_output_meta_map,
                                                       ArgType arg_type) const {
  ORT_ENFORCE(arg_type == ArgType::kInput || arg_type == ArgType::kOutput, "Valid values kInput, kOutput");

  const bool is_inputs = arg_type == ArgType::kInput;

  const char* const input_output_moniker = is_inputs ? "input" : "output";
  const char* const feed_fetches_moniker = is_inputs ? "feed" : "fetch";

#if !defined(DISABLE_SPARSE_TENSORS)
  auto is_sparse_initializer = [this](const std::string& name) -> bool {
    int idx = -1;
    if (session_state_->GetOrtValueNameIdxMap().GetIdx(name, idx).IsOK()) {
      return session_state_->IsSparseInitializer(idx);
    }
    return false;
  };
#endif

  if (names.size() != feeds_fetches.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, feed_fetches_moniker, " names has ", names.size(),
                           " elements, but ", feed_fetches_moniker, " has ", feeds_fetches.size(), " elements.");
  }

  for (size_t i = 0; i < feeds_fetches.size(); ++i) {
    const auto& name = names[i];

    auto iter = input_output_meta_map.find(name);
    if (input_output_meta_map.end() == iter) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid ", input_output_moniker, " name: ", name);
    }

    const auto& input_output_ml_value = feeds_fetches[i];

    // For outputs the user may supply an unallocated placeholder.
    if (!is_inputs && !input_output_ml_value.IsAllocated()) {
      continue;
    }

    auto expected_type = iter->second.ml_data_type;

    if (input_output_ml_value.IsTensor()) {
      if (!expected_type->IsTensorType()
#if !defined(DISABLE_OPTIONAL_TYPE)
          && !utils::IsOptionalTensor(expected_type)
#endif
      ) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, input_output_moniker, " with name: '", name,
                               "' expected to be of type: ", static_cast<int>(expected_type->type_), " but received a tensor");
      }

      // check for type
#if !defined(DISABLE_OPTIONAL_TYPE)
      auto expected_element_type = expected_type->IsTensorType()
                                       ? expected_type
                                             ->AsTensorType()
                                             ->GetElementType()
                                       : utils::GetElementTypeFromOptionalTensor(expected_type);
#else
      auto expected_element_type = expected_type->AsTensorType()->GetElementType();
#endif

      const auto& input_output_tensor = input_output_ml_value.Get<Tensor>();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_output_tensor.DataType(),
                                                expected_element_type, "tensor", input_output_moniker));

      // check for shape
      const auto& opt_shape = iter->second.tensor_shape;
      if (opt_shape.has_value() && !opt_shape->GetDims().empty()) {
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(name, input_output_tensor.Shape(),
                                                   *opt_shape, input_output_moniker));
      }
    } else if (input_output_ml_value.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)

      const SparseTensor& sparse_tensor = input_output_ml_value.Get<SparseTensor>();
      if (expected_type->IsSparseTensorType()) {
        auto expected_element_type = expected_type->AsSparseTensorType()->GetElementType();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(sparse_tensor.DataType(), expected_element_type,
                                                  "sparse_tensor", input_output_moniker));
        // Check shape
        const auto& opt_shape = iter->second.tensor_shape;
        if (opt_shape.has_value() && !opt_shape->GetDims().empty()) {
          ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(name, sparse_tensor.DenseShape(),
                                                     *opt_shape, input_output_moniker));
        }
      } else if (is_sparse_initializer(name) &&
                 expected_type->IsTensorType()) {
        // If this metadata came from a sparse initializer converted to dense, then still validate it.
        auto expected_element_type = expected_type->AsTensorType()->GetElementType();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(sparse_tensor.DataType(), expected_element_type,
                                                  "sparse_tensor", input_output_moniker));
        // Check shape
        const auto& opt_shape = iter->second.tensor_shape;
        if (opt_shape.has_value() && !opt_shape->GetDims().empty()) {
          ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(name, sparse_tensor.DenseShape(),
                                                     *opt_shape, input_output_moniker));
        }
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, input_output_moniker, " with name: '", name,
                               "' expected to be of type: ", static_cast<int>(expected_type->type_), " but received a sparse tensor");
      }
#else
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, input_output_moniker, " with name ", name,
                             " is a sparse tensor, which is not supported in this build.");
#endif
    } else if (input_output_ml_value.IsTensorSequence()) {
      if (!expected_type->IsTensorSequenceType()
#if !defined(DISABLE_OPTIONAL_TYPE)
          && !utils::IsOptionalSeqTensor(expected_type)
#endif
      ) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, input_output_moniker, " with name: '", name,
                               "' expected to be of type: ", static_cast<int>(expected_type->type_), " but received a tensor sequence");
      }

#if !defined(DISABLE_OPTIONAL_TYPE)
      auto expected_element_type = expected_type->IsTensorSequenceType()
                                       ? expected_type
                                             ->AsSequenceTensorType()
                                             ->GetElementType()
                                       : utils::GetElementTypeFromOptionalSeqTensor(expected_type);
#else
      auto expected_element_type = expected_type->AsSequenceTensorType()->GetElementType();
#endif

      auto input_output_element_type = input_output_ml_value.Get<TensorSeq>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_output_element_type, expected_element_type, "seq", input_output_moniker));
    } else {
      auto input_output_type = input_output_ml_value.Type();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_output_type, expected_type, "", input_output_moniker));
    }
  }

  return Status::OK();
}

common::Status InferenceSession::ValidateInputs(gsl::span<const std::string> feed_names,
                                                gsl::span<const OrtValue> feeds) const {
  return ValidateInputsOutputs(feed_names, feeds, input_def_map_, ArgType::kInput);
}

common::Status InferenceSession::ValidateOutputs(gsl::span<const std::string> output_names,
                                                 const std::vector<OrtValue>* p_fetches) const {
  if (output_names.empty()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "At least one output should be requested.");
  }

  const auto fetches = (p_fetches == nullptr) ? EmptySpan<const OrtValue>() : gsl::make_span(*p_fetches);

  if (fetches.empty()) {
    for (const auto& name : output_names) {
      if (output_def_map_.count(name) == 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid output name:", name);
      }
    }
    return Status::OK();
  }

  return ValidateInputsOutputs(output_names, fetches, output_def_map_, ArgType::kOutput);
}

#ifdef ENABLE_TRAINING
Status InferenceSession::PartialRun(onnxruntime::RunOptions& run_options,
                                    std::vector<OrtValue>& feeds,
                                    std::vector<OrtValue>& fetches,
                                    PartialGraphExecutionState& state,
                                    FeedsFetchesManager& feeds_fetches_manager,
                                    const OrtValueCachePtr& cache,
                                    int32_t partial_graph_index) {
  Status retval = Status::OK();
  std::vector<IExecutionProvider*> exec_providers_to_stop;
  exec_providers_to_stop.reserve(execution_providers_.NumProviders());

  ORT_TRY {
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }

#if !defined(ORT_MINIMAL_BUILD)
    ORT_RETURN_IF(session_state_->GetMoeExpertState() != nullptr,
                  "MoE expert counting and statistics are not supported by the deprecated PartialRun path.");
#endif

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
      auto start_func = [&xp, &exec_providers_to_stop, run_options]() {
        auto status = xp->OnRunStart(run_options);
        if (status.IsOK())
          exec_providers_to_stop.push_back(xp.get());

        return status;
      };

      ORT_CHECK_AND_SET_RETVAL(start_func());
    }

    ORT_ENFORCE(run_options.only_execute_path_to_fetches == false, "only_execute_path_to_fetches is not supported.");

    ORT_ENFORCE(session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL, "Only sequential mode is supported.");

    // execute the graph
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    if (state.GetProgramCounterStart() == 0) {
      session_state_->IncrementGraphExecutionCounter();
    }
#endif
    ORT_CHECK_AND_SET_RETVAL(utils::ExecutePartialGraph(*session_state_, feeds_fetches_manager, feeds, fetches,
                                                        run_logger, state, cache, run_options.terminate,
                                                        partial_graph_index,
                                                        /*parent stream*/ nullptr));
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
    auto status = xp->OnRunEnd(/*sync_stream*/ false, run_options);
    ORT_CHECK_AND_SET_RETVAL(status);
  }

  return retval;
}
#endif

namespace {
// Concurrent runs counting and thread-pool spin control
struct ThreadPoolSpinningSwitch {
  concurrency::ThreadPool* intra_tp_{nullptr};
  concurrency::ThreadPool* inter_tp_{nullptr};
  std::atomic<int>& concurrent_num_runs_;
  // __Ctor Refcounting and spinning control
  ThreadPoolSpinningSwitch(concurrency::ThreadPool* intra_tp,
                           concurrency::ThreadPool* inter_tp,
                           std::atomic<int>& ref) noexcept
      : intra_tp_(intra_tp), inter_tp_(inter_tp), concurrent_num_runs_(ref) {
    if (concurrent_num_runs_.fetch_add(1, std::memory_order_relaxed) == 0) {
      if (intra_tp_) intra_tp_->EnableSpinning();
      if (inter_tp_) inter_tp_->EnableSpinning();
    }
  }
  ~ThreadPoolSpinningSwitch() {
    if (1 == concurrent_num_runs_.fetch_sub(1, std::memory_order_acq_rel)) {
      if (intra_tp_) intra_tp_->DisableSpinning();
      if (inter_tp_) inter_tp_->DisableSpinning();
    }
  }
};
}  // namespace

Status InferenceSession::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                             gsl::span<const char* const> values) {
  Status retval = Status::OK();

  if (!is_inited_) {
    LOGS(*session_logger_, ERROR) << "Session was not initialized";
    return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
  }

  // TODO: only call SetEpDynamicOptions for all providers in-use
  for (auto& xp : execution_providers_) {
    auto status = xp->SetEpDynamicOptions(keys, values);
    ORT_CHECK_AND_SET_RETVAL(status);
  }

  return retval;
}

Status InferenceSession::Run(const RunOptions& run_options,
                             gsl::span<const std::string> feed_names, gsl::span<const OrtValue> feeds,
                             gsl::span<const std::string> output_names, std::vector<OrtValue>* p_fetches,
                             const std::vector<OrtDevice>* p_fetches_device_info) {
  // Serialize graph-state checks and replay as well as normal execution. Keep the
  // lock here so RunImpl's internal capture retries do not acquire it recursively.
  std::optional<std::lock_guard<std::mutex>> sequential_run_lock;
  if (!is_concurrent_run_supported_) {
    sequential_run_lock.emplace(session_mutex_);
  }

  return RunImpl(run_options, feed_names, feeds, output_names, p_fetches, p_fetches_device_info,
                 /*graph_capture_depth=*/0);
}

Status InferenceSession::RunImpl(const RunOptions& run_options,
                                 gsl::span<const std::string> feed_names, gsl::span<const OrtValue> feeds,
                                 gsl::span<const std::string> output_names, std::vector<OrtValue>* p_fetches,
                                 const std::vector<OrtDevice>* p_fetches_device_info,
                                 int graph_capture_depth) {
  // Ignore run-level profiling request if session-level profiling is already enabled.
  std::optional<profiling::Profiler> run_profiler;
  if (run_options.enable_profiling && session_profiler_.IsEnabled()) {
    LOGS(*session_logger_, WARNING) << "RunOptions requests profiling but session-level profiling is already enabled. "
                                       "Ignoring run-level profiling request.";
  }
  if (run_options.enable_profiling && !session_profiler_.IsEnabled()) {
    run_profiler.emplace();
    run_profiler->Initialize(session_logger_);
    std::basic_string<ORTCHAR_T> profile_file = GenerateProfileFilePath(run_options.profile_file_prefix);
    for (auto& ep : execution_providers_) {
      run_profiler->AddEpProfilers(ep->GetProfiler());
    }
    run_profiler->StartProfiling(profile_file);
  }

#if !defined(ORT_MINIMAL_BUILD)
  const bool track_moe_experts =
      session_options_.config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigEnableMoeExpertCounting, "0") == "1" ||
      session_options_.config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigEnableMoeExpertStatistics, "0") == "1";
  const bool collect_moe_statistics =
      session_options_.config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigEnableMoeExpertStatistics, "0") == "1";
  KernelPilotMoeExpertState* moe_expert_state = nullptr;
  bool moe_run_active = false;
  auto end_moe_run = gsl::finally([&]() {
    if (moe_run_active) {
      ORT_IGNORE_RETURN_VALUE(moe_expert_state->EndRun());
    }
  });
#endif  // !defined(ORT_MINIMAL_BUILD)
  InlinedVector<AllocatorPtr> arenas_to_shrink;

  TimePoint tp = std::chrono::high_resolution_clock::now();
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.Start();
  } else if (run_profiler) {
    tp = run_profiler->Start();
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingActivity<telemetry_provider_handle> ortrun_activity;
  ortrun_activity.SetRelatedActivity(session_activity);
  TraceLoggingWriteStart(ortrun_activity, "OrtRun");
#endif
  Status retval = Status::OK();
  const Env& env = Env::Default();

  int graph_annotation_id = 0;
  const std::string& graph_annotation_str =
      run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigCudaGraphAnnotation, "");
  if (!graph_annotation_str.empty()) {
    if (!TryParseStringWithClassicLocale<int>(graph_annotation_str, graph_annotation_id)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to parse the cuda graph annotation id: ",
                             graph_annotation_str);
    }
  }

  // Increment/decrement concurrent_num_runs_ and control
  // session threads spinning as configured. Do nothing for graph replay except the counter.
  const bool control_spinning = use_per_session_threads_ &&
                                force_spinning_stop_between_runs_ &&
                                !cached_execution_provider_for_graph_replay_.IsGraphCaptured(graph_annotation_id);
  auto* intra_tp = (control_spinning) ? thread_pool_.get() : nullptr;
  auto* inter_tp = (control_spinning) ? inter_op_thread_pool_.get() : nullptr;
  ThreadPoolSpinningSwitch runs_refcounter_and_tp_spin_control(intra_tp, inter_tp, current_num_runs_);

  // Check if this Run() can skip normal execution and replay a previously captured graph.
  if (cached_execution_provider_for_graph_replay_.IsGraphCaptured(graph_annotation_id)) {
    LOGS(*session_logger_, INFO) << "Replaying the captured "
                                 << cached_execution_provider_for_graph_replay_.Type()
                                 << " graph for this model with tag: " << run_options.run_tag
                                 << " with graph annotation id: " << graph_annotation_id;
    // log evaluation start to trace logging provider
    env.GetTelemetryProvider().LogEvaluationStart(session_id_);
    bool sync_graph_replay = run_options.config_options.GetConfigOrDefault(
                                 kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "0";
    ORT_RETURN_IF_ERROR_SESSIONID_(cached_execution_provider_for_graph_replay_.ReplayGraph(graph_annotation_id,
                                                                                           sync_graph_replay));
  } else {
    InlinedVector<IExecutionProvider*> exec_providers_to_stop;
    exec_providers_to_stop.reserve(execution_providers_.NumProviders());

    ORT_TRY {
      if (!is_inited_) {
        LOGS(*session_logger_, ERROR) << "Session was not initialized";
        return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
      }

      // log evaluation start to trace logging provider
      env.GetTelemetryProvider().LogEvaluationStart(session_id_);

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

      // scope of owned_run_logger is just the call to Execute.
      // If Execute ever becomes async we need a different approach
      std::unique_ptr<logging::Logger> owned_run_logger;
      const auto& run_logger = CreateLoggerForRun(run_options, owned_run_logger);

#if !defined(ORT_MINIMAL_BUILD)
      if (track_moe_experts) {
        moe_expert_state = session_state_->GetMoeExpertState();
        ORT_RETURN_IF_NOT(moe_expert_state, "MoE expert state is unavailable.");
        ORT_RETURN_IF_ERROR_SESSIONID_(moe_expert_state->BeginRun(
            run_options.run_tag, collect_moe_statistics ? &run_logger : nullptr));
        moe_run_active = true;
      }
#endif  // !defined(ORT_MINIMAL_BUILD)

      // info all execution providers InferenceSession:Run started
      // TODO: only call OnRunStart for all providers in-use
      for (auto& xp : execution_providers_) {
        // call OnRunStart and add to exec_providers_to_stop if successful
        auto start_func = [&xp, &exec_providers_to_stop, &run_options]() {
          auto status = xp->OnRunStart(run_options);
          if (status.IsOK())
            exec_providers_to_stop.push_back(xp.get());

          return status;
        };

        ORT_CHECK_AND_SET_RETVAL(start_func());
      }

#ifdef ENABLE_TRAINING
      if (run_options.only_execute_path_to_fetches) {
        // TODO: this method is not thread safe, if multiple Run happened in parallel we might hit race condition issue.
        // currently it only used in training, there is no parallel run execution in training so it is ok.
        // but it is better we can fix it with a better solution.
        session_state_->UpdateToBeExecutedRange(feeds_fetches_manager.GetFeedsFetchesInfo().fetches_mlvalue_idxs);
      }
#endif

      // execute the graph
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
      session_state_->IncrementGraphExecutionCounter();
#endif

#ifdef ORT_ENABLE_STREAM
      DeviceStreamCollectionHolder device_stream_collection_holder(session_state_.get());
      if (run_options.sync_stream != nullptr) {
        if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
          // XXX: Not tested in Parallel execution mode and disabled at this time.
          LOGS(*session_logger_, WARNING) << "Setting sync stream is not supported in parallel execution mode.";
        } else {
          ORT_RETURN_IF_ERROR_SESSIONID_(
              device_stream_collection_holder.p_->SetStreamOverride(run_options.sync_stream));
        }
      }
#endif

      if (retval.IsOK()) {
        retval = utils::ExecuteGraph(*session_state_, feeds_fetches_manager, feeds, *p_fetches,
                                     session_options_.execution_mode,
                                     run_options,
#ifdef ORT_ENABLE_STREAM
                                     device_stream_collection_holder,
#endif
                                     run_logger,
                                     run_profiler ? &*run_profiler : nullptr);
      }

      // info all execution providers InferenceSession:Run ended
      for (auto* xp : exec_providers_to_stop) {
        bool synchronize_execution_providers = run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "0";
        auto status = xp->OnRunEnd(synchronize_execution_providers, run_options);
        ORT_CHECK_AND_SET_RETVAL(status);
      }

#if !defined(ORT_MINIMAL_BUILD)
      if (moe_run_active) {
        const Status moe_run_status = moe_expert_state->EndRun();
        moe_run_active = false;
        if (retval.IsOK() && !moe_run_status.IsOK()) {
          retval = moe_run_status;
        }
      }
#endif  // !defined(ORT_MINIMAL_BUILD)

      if (run_profiler) {
        run_profiler->EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
        run_profiler->EndProfiling();
      }

      // Move stream cleanup from ExecuteGraph to here for graph capture.
      // Cleanup may synchronize provider streams, which is not allowed while an
      // EP is still inside its capture window.
      // Graph capture, if any, ends when xp->OnRunEnd() returns above, so it is safe here.
#ifdef ORT_ENABLE_STREAM
      DeviceStreamCollection* device_stream_collection = device_stream_collection_holder.p_.get();
      if (device_stream_collection) {
        bool sync_execution_provider = run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "0";
        ORT_CHECK_AND_SET_RETVAL(device_stream_collection->CleanUp(sync_execution_provider));
      }
#endif
    }
    ORT_CATCH(const std::exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
      });
    }
    ORT_CATCH(...) {
      retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
    }

    if (!arenas_to_shrink.empty()) {
      ShrinkMemoryArenas(arenas_to_shrink);
    }
  }

  // keep track of telemetry
  int64_t batch_size = 1;
  for (const auto& feed : feeds) {
    if (!feed.IsTensor()) {
      continue;
    }

    const Tensor& tensor = feed.Get<Tensor>();
    const TensorShape& shape = tensor.Shape();
    if (shape.NumDimensions() > 0) {
      batch_size = shape[0];  // Extract batch size
    }
    // Exit the loop after finding the first tensor since subsequent feeds will have the same batch size.
    break;
  }

  // Only include successful inferences in batch since failed inferences can skew the metric
  if (retval.IsOK()) {
    // time to send telemetry?
    {
      // Adding lock_guard here to ensure that telemetry updates are thread-safe.
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.total_runs_since_last_;
      telemetry_.total_run_duration_since_last_ += TimeDiffMicroSeconds(tp);
      auto batch_entry = telemetry_.duration_per_batch_size_.find(batch_size);
      if (batch_entry != telemetry_.duration_per_batch_size_.end()) {
        batch_entry->second += TimeDiffMicroSeconds(tp);
      } else if (telemetry_.duration_per_batch_size_.size() < telemetry_detail::kMaxTelemetryCollectionEntries) {
        telemetry_.duration_per_batch_size_.emplace(batch_size, TimeDiffMicroSeconds(tp));
      }

      // Emit RuntimePerf on scheduled interval
      if ((TimeDiffMicroSeconds(telemetry_.time_sent_last_) > telemetry_.runtime_perf_interval_)) {
        env.GetTelemetryProvider().LogRuntimePerf(session_id_, telemetry_.total_runs_since_last_,
                                                  telemetry_.total_run_duration_since_last_,
                                                  telemetry_.duration_per_batch_size_);

        // Emit one EpDeviceUsage event per (EP, hardware device) tuple so
        // downstream consumers can attribute usage to a specific EP+device combo
        // without joining back to SessionCreation (which may fall outside the
        // telemetry pipeline's lookback window for long-lived sessions).
        for (const auto& ep_info : telemetry_.ep_device_info_) {
          env.GetTelemetryProvider().LogEpDeviceUsage(
              session_id_, ep_info.ep_type, ep_info.hardware_device_type,
              ep_info.vendor_id, ep_info.device_id, ep_info.vendor, ep_info.ep_vendor,
              ep_info.ep_version, ep_info.assigned_node_count,
              telemetry_.total_runs_since_last_, telemetry_.total_run_duration_since_last_);
        }

        // reset counters
        telemetry_.time_sent_last_ = std::chrono::high_resolution_clock::now();
        telemetry_.total_runs_since_last_ = 0;
        telemetry_.total_run_duration_since_last_ = 0;
        telemetry_.duration_per_batch_size_.clear();

        // Double the interval, capping at kRuntimePerfMaxInterval
        telemetry_.runtime_perf_interval_ = std::min(telemetry_.runtime_perf_interval_ * 2,
                                                     Telemetry::kRuntimePerfMaxInterval);
      }
    }
  } else {
    // Log runtime error with EP versions
    env.GetTelemetryProvider().LogRuntimeInferenceError(session_id_, retval, telemetry_.ep_versions_summary_,
                                                        telemetry_.ep_device_types_summary_);
  }

  // log evaluation stop to trace logging provider
  env.GetTelemetryProvider().LogEvaluationStop(session_id_);

  // send out profiling events (optional)
  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingWriteStop(ortrun_activity, "OrtRun");
#endif

#if !defined(ORT_MINIMAL_BUILD)
  if (IsNodeStatsCollectionEnabled() && retval.IsOK()) {
    // Dump node stats if the run was successful
    node_stats_recorder_->DumpStats(session_state_->GetGraphViewer().ModelPath());
    node_stats_recorder_->ResetPerRunNameDeduper();
  }
#endif

  reset_saturation_count();

  // Some EPs require multiple internal runs before replay is ready: warm-up
  // runs for allocations/setup followed by one run that captures the graph.
  // Retry within the same user-visible Session::Run() call until the EP reports
  // that capture has completed, subject to kMaxGraphCaptureRunAttempts.
  if (retval.IsOK() && cached_execution_provider_for_graph_replay_.IsGraphCaptureEnabled() &&
      cached_execution_provider_for_graph_replay_.AllowGraphCaptureOnRun(graph_annotation_id) &&
      !cached_execution_provider_for_graph_replay_.IsGraphCaptured(graph_annotation_id)) {
    if (graph_capture_depth + 1 >= kMaxGraphCaptureRunAttempts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Graph capture did not complete after ", kMaxGraphCaptureRunAttempts,
                             " internal runs for ", cached_execution_provider_for_graph_replay_.Type(),
                             ". The execution provider may not support ORT-managed graph capture/replay.");
    }

    LOGS(*session_logger_, INFO) << "Start another run for necessary memory allocation or graph capture.";
    // Disable run-level profiling for internal runs used for memory allocation or graph capture
    RunOptions recursive_run_options{run_options};
    recursive_run_options.enable_profiling = false;
    ORT_RETURN_IF_ERROR(RunImpl(recursive_run_options, feed_names, feeds, output_names, p_fetches,
                                p_fetches_device_info, graph_capture_depth + 1));
  }

  // Log runtime error telemetry if the return value is not OK
  ORT_RETURN_IF_ERROR_SESSIONID(retval, session_id_);
  return retval;
}

Status InferenceSession::Run(const RunOptions& run_options,
                             gsl::span<const char* const> feed_names,
                             gsl::span<const OrtValue* const> feeds,
                             gsl::span<const char* const> fetch_names,
                             gsl::span<OrtValue*> fetches) {
  size_t num_feeds = feed_names.size();
  size_t num_fetches = fetch_names.size();
  InlinedVector<std::string> feed_name_vec;
  feed_name_vec.reserve(num_feeds);
  InlinedVector<OrtValue> feed_vec;
  feed_vec.reserve(num_feeds);

  for (size_t i = 0; i != num_feeds; ++i) {
    if (feed_names[i] == nullptr || feed_names[i][0] == '\0') {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input name cannot be empty");
    }

    if (!feeds[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, MakeString("NULL input supplied for input ", feed_names[i]).c_str());
    }

    feed_name_vec.emplace_back(feed_names[i]);
    feed_vec.emplace_back(*feeds[i]);
  }

  // Create output feed
  InlinedVector<std::string> fetch_name_vec;
  fetch_name_vec.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetch_names[i] == nullptr || fetch_names[i][0] == '\0') {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "output name cannot be empty");
    }
    fetch_name_vec.emplace_back(fetch_names[i]);
  }

  std::vector<OrtValue> fetch_vec;
  fetch_vec.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] != nullptr) {
      fetch_vec.emplace_back(*fetches[i]);
    } else {
      fetch_vec.emplace_back();
    }
  }

  Status status;
  status = Run(run_options, feed_name_vec, feed_vec, fetch_name_vec, &fetch_vec, nullptr);

  if (!status.IsOK())
    return status;

  // We do it in two loops to make sure copy __ctors does not throw
  InlinedVector<std::unique_ptr<OrtValue>> fetch_unique_ptrs;
  fetch_unique_ptrs.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] == nullptr) {
      fetch_unique_ptrs.emplace_back(std::make_unique<OrtValue>(fetch_vec[i]));
    } else {
      fetch_unique_ptrs.emplace_back();
    }
  }

  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] == nullptr) {
      ORT_ENFORCE(fetch_unique_ptrs[i] != nullptr);
      fetches[i] = fetch_unique_ptrs[i].release();
    }
  }
  return Status::OK();
}

common::Status InferenceSession::RunAsync(const RunOptions* run_options,
                                          gsl::span<const char* const> feed_names,
                                          gsl::span<const OrtValue* const> feeds,
                                          gsl::span<const char* const> fetch_names,
                                          gsl::span<OrtValue*> fetches,
                                          RunAsyncCallbackFn callback,
                                          void* user_data) {
  size_t num_fetches = fetch_names.size();
  auto* tp = GetIntraOpThreadPoolToUse();
  if (!tp || concurrency::ThreadPool::DegreeOfParallelism(tp) < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "intra op thread pool must have at least one thread for RunAsync");
  }
  std::function<void()> run_fn = [run_options, feed_names, feeds, fetch_names, fetches, num_fetches,
                                  callback, user_data, this]() {
    Status status = Status::OK();
    ORT_TRY {
      if (run_options) {
        status = Run(*run_options, feed_names, feeds, fetch_names, fetches);
      } else {
        RunOptions default_run_options;
        status = Run(default_run_options, feed_names, feeds, fetch_names, fetches);
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    ORT_CATCH(...) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "unknown exception");
    }
    callback(user_data, fetches.data(), status.IsOK() ? num_fetches : 0, ToOrtStatus(status));
  };  // run_fn
  concurrency::ThreadPool::Schedule(tp, run_fn);
  return Status::OK();
}

common::Status InferenceSession::Run(const NameMLValMap& feeds, gsl::span<const std::string> output_names,
                                     std::vector<OrtValue>* p_fetches) {
  return Run(RunOptions(), feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options, const NameMLValMap& feeds_map,
                                     gsl::span<const std::string> output_names, std::vector<OrtValue>* p_fetches) {
  InlinedVector<std::string> feed_names;
  InlinedVector<OrtValue> feeds;

  const auto num_feeds = feeds_map.size();
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
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &model_metadata_);
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetModelInputs() const {
  {
    std::lock_guard<std::mutex> l(session_mutex_);
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
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  // returns a list of initializers that can be overridden.
  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetOverridableInitializers());
}

std::pair<common::Status, const OutputDefList*> InferenceSession::GetModelOutputs() const {
  {
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetOutputs());
}

common::Status InferenceSession::GetInputOutputMemoryInfo(SessionInputOutputType type,
                                                          InlinedVector<const OrtMemoryInfo*>& memory_info) const {
  memory_info.clear();

  if (!is_inited_) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Session has not been initialized.");
  }

  std::pair<common::Status, const OutputDefList*> result;
  switch (type) {
    case SessionInputOutputType::kInput:
      result = GetModelInputs();
      break;
    case SessionInputOutputType::kOutput:
      result = GetModelOutputs();
      break;
    case SessionInputOutputType::kOverridableInitializer:
      // add if/when needed
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "GetInputOutputMemoryInfo for kOverridableInitializer is not implemented.");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unexpected SessionInputOutputType of ", static_cast<uint8_t>(type));
  }

  ORT_RETURN_IF_ERROR(result.first);

  const auto& def_list = *result.second;
  memory_info.reserve(def_list.size());

  for (const auto* def : def_list) {
    InlinedVector<SessionState::NodeInfo> node_info_vec;
    Status status;
    if (type == SessionInputOutputType::kOutput) {
      status = session_state_->GetOutputNodeInfo(def->Name(), node_info_vec);
    } else {
      status = session_state_->GetInputNodeInfo(def->Name(), node_info_vec);
    }

    if (!status.IsOK()) {
      if (type == SessionInputOutputType::kInput) {
        return status;
      }

      // Check first if this output is produced by an input that directly
      // propagates to output with the same name.
      status = session_state_->GetInputNodeInfo(def->Name(), node_info_vec);
      if (status.IsOK()) {
        // all entries are for the same OrtDevice so use the first one.
        // we need to get an OrtMemoryInfo* that will remain valid, so we get the allocator for the OrtDevice
        // from the session state and use its OrtMemoryInfo.
        auto allocator = session_state_->GetAllocator(*node_info_vec.front().device);
        memory_info.push_back(&allocator->Info());
      } else {
        // Check if this output is produced by a constant initializer
        // Pick the MemoryInfo from the initializer's OrtValue
        const auto& ort_value_map = session_state_->GetOrtValueNameIdxMap();

        OrtValueIndex ort_value_index;
        status = ort_value_map.GetIdx(def->Name(), ort_value_index);
        if (!status.IsOK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to find node output or a constant initializer producing output: ",
                                 def->Name(), ".");
        }

        const auto& idx_to_ort_value = session_state_->GetInitializedTensors();
        auto it = idx_to_ort_value.find(ort_value_index);
        if (it == idx_to_ort_value.end()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to find node output or a constant initializer producing output: ",
                                 def->Name(), ".");
        }
        const auto& tensor = it->second.Get<Tensor>();
        auto allocator = session_state_->GetAllocator(tensor.Location());
        memory_info.push_back(&allocator->Info());
      }
    } else {
      // all entries are for the same OrtDevice so use the first one.
      // we need to get an OrtMemoryInfo* that will remain valid, so we get the allocator for the OrtDevice
      // from the session state and use its OrtMemoryInfo.
      auto allocator = session_state_->GetAllocator(*node_info_vec.front().device);
      memory_info.push_back(&allocator->Info());
    }
  }

  return Status::OK();
}

common::Status InferenceSession::GetEpDeviceForInputs(InlinedVector<const OrtEpDevice*>& ep_devices) const {
  ep_devices.clear();

#if defined(ORT_MINIMAL_BUILD)
  return common::Status(common::ONNXRUNTIME, common::FAIL,
                        "GetEpDeviceForInputs is not available in a minimal build.");
#else
  if (!is_inited_) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Session has not been initialized.");
  }

  std::pair<common::Status, const OutputDefList*> inputs = GetModelInputs();

  ORT_RETURN_IF_ERROR(inputs.first);

  const auto& def_list = *inputs.second;
  ep_devices.reserve(def_list.size());

  const auto& available_eps = environment_.GetOrtEpDevices();

  for (const auto* def : def_list) {
    InlinedVector<SessionState::NodeInfo> node_info_vec;
    ORT_RETURN_IF_ERROR(session_state_->GetInputNodeInfo(def->Name(), node_info_vec));
    assert(!node_info_vec.empty());
    // If we have an input that is not consumed by any node,
    // including nodes in subgraphs, then we return nullptr.
    const auto* p_node = node_info_vec.front().p_node;
    if (p_node != nullptr) {
      const auto ep_name = p_node->GetExecutionProviderType();
      auto it = std::find_if(available_eps.begin(), available_eps.end(), [&ep_name](const OrtEpDevice* entry) {
        return entry->ep_name == ep_name;
      });
      ep_devices.push_back(it != available_eps.end() ? *it : nullptr);
    } else {
      ep_devices.push_back(nullptr);
    }
  }

  return Status::OK();
#endif
}

common::Status InferenceSession::GetEpDeviceForOutputs(InlinedVector<const OrtEpDevice*>& ep_devices) const {
  ep_devices.clear();

#if defined(ORT_MINIMAL_BUILD)
  return common::Status(common::ONNXRUNTIME, common::FAIL,
                        "GetEpDeviceForOutputs is not available in a minimal build.");
#else
  if (!is_inited_) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Session has not been initialized.");
  }

  std::pair<common::Status, const OutputDefList*> outputs = GetModelOutputs();

  ORT_RETURN_IF_ERROR(outputs.first);

  const auto& def_list = *outputs.second;
  ep_devices.reserve(def_list.size());

  const auto& available_eps = environment_.GetOrtEpDevices();

  for (const auto* def : def_list) {
    InlinedVector<SessionState::NodeInfo> node_info_vec;
    ORT_RETURN_IF_ERROR(session_state_->GetOutputNodeInfo(def->Name(), node_info_vec));
    assert(!node_info_vec.empty());
    // If we have an output that is not produced by any node,
    // then we return nullptr.
    const auto* p_node = node_info_vec.front().p_node;
    if (p_node != nullptr) {
      const auto ep_name = p_node->GetExecutionProviderType();
      auto it = std::find_if(available_eps.begin(), available_eps.end(), [&ep_name](const OrtEpDevice* entry) {
        return entry->ep_name == ep_name;
      });
      ep_devices.push_back(it != available_eps.end() ? *it : nullptr);
    } else {
      ep_devices.push_back(nullptr);
    }
  }

  return Status::OK();
#endif
}

common::Status InferenceSession::NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
  {
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }
  }

  *io_binding = std::make_unique<IOBinding>(*session_state_);
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

common::Status InferenceSession::ReleaseCapturedGraph(int graph_annotation_id) {
  // Acquire session_mutex_ only when concurrent run is not supported, matching the
  // locking pattern in Run(). For concurrent EPs the mutex is not held by Run(),
  // so acquiring it here would not synchronize with in-flight runs; those EPs are
  // responsible for their own thread safety in ReleaseCapturedGraph.
  std::optional<std::lock_guard<std::mutex>> lock;
  if (!is_concurrent_run_supported_) {
    lock.emplace(session_mutex_);
  }
  return cached_execution_provider_for_graph_replay_.ReleaseCapturedGraph(graph_annotation_id);
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

#if !defined(ORT_MINIMAL_BUILD)
std::vector<TuningResults> InferenceSession::GetTuningResults() const {
  std::vector<TuningResults> ret;
  for (const auto& provider : execution_providers_) {
    const auto* tuning_ctx = provider->GetTuningContext();
    if (tuning_ctx != nullptr) {
      ret.emplace_back(tuning_ctx->GetTuningResults());
    }
  }
  return ret;
}

Status InferenceSession::SetTuningResults(
    const std::vector<TuningResults>& trs,
    bool error_on_invalid,
    bool auto_enable) {
  std::string msg;

  for (size_t i = 0; i < trs.size(); i++) {
    const auto& tr = trs[i];
    auto* provider = execution_providers_.Get(tr.ep);
    if (provider == nullptr) {
      msg = MakeString("Cannot find execution provider ", tr.ep);
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    auto* tuning_ctx = provider->GetTuningContext();
    if (tuning_ctx == nullptr) {
      msg = MakeString("Invalid TuningResults (index=", i, "). ", tr.ep, " does not support TunableOp.");
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    auto status = tuning_ctx->LoadTuningResults(tr);
    if (!status.IsOK()) {
      msg = MakeString("Failed to load TuningResults (index=", i, "). Reason: ", status.ErrorMessage());
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    if (auto_enable) {
      LOGS(*session_logger_, INFO) << "Correctly set TuningResults for " << tr.ep << ", enable TunableOp for using";
      tuning_ctx->EnableTunableOp();
    }
  }
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

AllocatorPtr InferenceSession::GetAllocator(const OrtMemoryInfo& mem_info) const {
  return session_state_->GetAllocator(mem_info);
}

common::Status InferenceSession::ValidateAndParseShrinkArenaString(const std::string& ort_device_list,
                                                                   /*out*/ InlinedVector<AllocatorPtr>& arenas_to_shrink) const {
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

    // Shrink if it is an arena allocator.
    // Both in-tree arenas (BFCArena) and plugin EP arenas (IArenaImplWrappingOrtAllocator)
    // inherit IArena, so AsArena() returns non-null for both.
    for (const auto& [device, allocator_ptr] : session_state_->GetAllocators()) {
      if (device.Type() == device_type && device.MemType() == memory_type && device.Id() == device_id) {
        if (allocator_ptr->AsArena() != nullptr) {
          arenas_to_shrink.push_back(allocator_ptr);
          break;
        }
      }
    }

    if (arenas_to_shrink.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Did not find an arena based allocator registered for device-id ",
                             "combination in the memory arena shrink list: ", device_id_pair);
    }
  }

  return Status::OK();
}

void InferenceSession::ShrinkMemoryArenas(gsl::span<const AllocatorPtr> arenas_to_shrink) {
  for (auto& alloc : arenas_to_shrink) {
    auto* arena = alloc->AsArena();
    if (!arena) {
      LOGS(*session_logger_, WARNING) << "Allocator is not an IArena, skipping Shrink: " << alloc->Info().ToString();
      continue;
    }

    auto status = arena->Shrink();

    if (!status.IsOK()) {
      LOGS(*session_logger_, WARNING) << "Unable to shrink arena: " << alloc->Info().ToString()
                                      << " error message: " << status.ErrorMessage();
    }
  }
}

void InferenceSession::PopulateEpDeviceInfo(const onnxruntime::Graph& graph) {
  telemetry_.ep_device_info_.clear();
  telemetry_.ep_device_types_summary_.clear();
  telemetry_.ep_device_vendor_ids_summary_.clear();
  telemetry_.ep_versions_summary_.clear();

  // First, count nodes assigned to each EP type after graph partitioning.
  // The graph node only carries the EP type string, so when a single EP targets
  // multiple devices (e.g. QNN targeting both NPU and GPU on the same SoC) we
  // cannot disambiguate node-to-device assignment here — every device row for the
  // same EP type will carry the same total count. Downstream aggregation should
  // use MAX (not SUM) across device rows to avoid double counting.
  std::unordered_map<std::string_view, int> nodes_per_ep;
  for (const auto& node : graph.Nodes()) {
    const auto ep = telemetry_detail::TelemetryStringView(std::string_view(node.GetExecutionProviderType()));
    if (!ep.empty()) {
      const auto entry = nodes_per_ep.find(ep);
      if (entry != nodes_per_ep.end()) {
        ++entry->second;
      } else if (nodes_per_ep.size() < telemetry_detail::kMaxTelemetryCollectionEntries) {
        nodes_per_ep.emplace(ep, 1);
      }
    }
  }

  // Then iterate registered EPs and emit one entry per (EP, OrtEpDevice) tuple,
  // falling back to OrtDevice for legacy EPs that were not created via the V2
  // OrtEpDevice path.
  for (const auto& provider : execution_providers_) {
    if (telemetry_.ep_device_info_.size() == telemetry_detail::kMaxTelemetryCollectionEntries) break;
    if (!provider) {
      continue;
    }

    const auto ep_type = telemetry_detail::TelemetryStringView(std::string_view(provider->Type()));
    const int assigned_nodes = nodes_per_ep.count(ep_type) ? nodes_per_ep[ep_type] : 0;
    const auto& ep_devices = provider->GetEpDevices();

    if (!ep_devices.empty()) {
      // V2 path: full hardware metadata available via OrtEpDevice / OrtHardwareDevice.
      for (const OrtEpDevice* ep_device : ep_devices) {
        if (telemetry_.ep_device_info_.size() == telemetry_detail::kMaxTelemetryCollectionEntries) break;
        if (!ep_device) {
          continue;
        }

        Telemetry::EpDeviceInfo entry;
        entry.ep_type = telemetry_detail::BoundedTelemetryString(ep_type);
        entry.ep_vendor = telemetry_detail::BoundedTelemetryString(ep_device->ep_vendor);
        auto it = ep_device->ep_metadata.Entries().find(kOrtEpDevice_EpMetadataKey_Version);
        if (it != ep_device->ep_metadata.Entries().end()) {
          entry.ep_version = telemetry_detail::BoundedTelemetryString(it->second);
        }
        if (ep_device->device != nullptr) {
          entry.hardware_device_type = HardwareDeviceTypeToString(ep_device->device->type);
          entry.vendor_id = ep_device->device->vendor_id;
          entry.device_id = ep_device->device->device_id;
          entry.vendor = telemetry_detail::BoundedTelemetryString(ep_device->device->vendor);
        } else {
          entry.hardware_device_type = "UNKNOWN";
        }
        entry.assigned_node_count = assigned_nodes;
        telemetry_.ep_device_info_.push_back(std::move(entry));
      }
    } else {
      // Legacy path: only OrtDevice is available. PCI vendor/device IDs and
      // vendor name are not populated for legacy EPs — downstream consumers
      // see vendor_id from OrtDevice (which matches PCI for known EPs) and
      // empty vendor / device_id of 0.
      const OrtDevice& device = provider->GetDevice();
      Telemetry::EpDeviceInfo entry;
      entry.ep_type = telemetry_detail::BoundedTelemetryString(ep_type);
      entry.hardware_device_type = OrtDeviceTypeToString(device.Type());
      entry.vendor_id = device.Vendor();
      entry.device_id = 0;
      entry.vendor = std::string();
      entry.ep_vendor = std::string();
      entry.assigned_node_count = assigned_nodes;
      telemetry_.ep_device_info_.push_back(std::move(entry));
    }
  }

  // Build the comma-separated summaries used to enrich SessionCreation. Order
  // matches execution_providers_.GetIds() iteration order so consumers can join
  // by position against the existing executionProviderIds field.
  bool first = true;
  for (const auto& entry : telemetry_.ep_device_info_) {
    // Format vendor IDs as hex for readability (PCI IDs are conventionally hex).
    std::ostringstream vendor_id;
    vendor_id << "0x" << std::hex << std::uppercase << std::setw(4)
              << std::setfill('0') << entry.vendor_id
              << std::dec << std::nouppercase << std::setfill(' ');
    if (!telemetry_detail::AppendTelemetryRow(
            std::array{&telemetry_.ep_device_types_summary_, &telemetry_.ep_device_vendor_ids_summary_,
                       &telemetry_.ep_versions_summary_},
            std::array<std::string_view, 3>{entry.hardware_device_type, vendor_id.str(),
                                            entry.ep_type + ":" + entry.ep_version},
            first)) {
      break;
    }
    first = false;
  }
}

void InferenceSession::LogSessionCreationTelemetry(const onnxruntime::Graph& graph,
                                                   const std::string& model_weight_type,
                                                   const std::string& model_graph_hash,
                                                   const std::string& model_weight_hash) {
  const Env& env = Env::Default();
  const auto& model_path = graph.ModelPath().native();
#ifdef _WIN32
  const auto path_view = telemetry_detail::TelemetryWideStringView(model_path, telemetry_detail::kMaxTelemetryPathBytes);
  const auto file_name = path_view.size() == model_path.size()
                             ? std::filesystem::path(path_view).filename().native()
                             : std::wstring{};
  std::string model_file_name = ToUTF8String(telemetry_detail::BoundedTelemetryWideString(file_name));
#else
  const auto path_view =
      telemetry_detail::TelemetryStringView(std::string_view(model_path), telemetry_detail::kMaxTelemetryPathBytes);
  const auto file_name = path_view.size() == model_path.size()
                             ? std::filesystem::path(path_view).filename().native()
                             : std::string{};
  std::string model_file_name = telemetry_detail::BoundedTelemetryString(file_name);
#endif
  bool model_has_fp16_inputs = ModelHasFP16Inputs(graph);

  // Populate per-(EP, hardware-device) telemetry data captured once at session
  // initialization. The graph partitioning is complete at this point, so we can
  // count how many nodes each EP is assigned. This populates telemetry_.ep_device_info_
  // and the comma-separated summary strings used to enrich SessionCreation.
  PopulateEpDeviceInfo(graph);

  env.GetTelemetryProvider().LogSessionCreation(
      session_id_, model_->IrVersion(), model_->ProducerName(), model_->ProducerVersion(), model_->Domain(),
      graph.DomainToVersionMap(), model_file_name, graph.Name(), model_weight_type, model_graph_hash, model_weight_hash,
      model_->MetaData(), telemetry_.event_name_, execution_providers_.GetIds(),
      telemetry_.ep_device_types_summary_, telemetry_.ep_device_vendor_ids_summary_,
      telemetry_.ep_versions_summary_,
      model_has_fp16_inputs, false);

  // Emit one initial EpDeviceUsage event per (EP, device) pair with run counts of 0.
  // Ensures we capture EP/device topology even for sessions that end before the
  // first RuntimePerf heartbeat (2s after the first Run()).
  for (const auto& ep_info : telemetry_.ep_device_info_) {
    env.GetTelemetryProvider().LogEpDeviceUsage(
        session_id_, ep_info.ep_type, ep_info.hardware_device_type,
        ep_info.vendor_id, ep_info.device_id, ep_info.vendor, ep_info.ep_vendor,
        ep_info.ep_version, ep_info.assigned_node_count, 0, 0);
  }
}

common::Status InferenceSession::RecordSessionCreationEndTelemetry(const TimePoint& tp, common::Status status) {
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

#if defined(ORT_USE_TELEMETRY)
  Env::Default().GetTelemetryProvider().LogSessionCreationEnd(
      session_id_, status, TimeDiffMicroSeconds(tp));
#elif defined(_WIN32)
  Env::Default().GetTelemetryProvider().LogSessionCreationEnd(session_id_, status, 0);
#endif
  return status;
}

#undef ORT_TELEMETRY_CAPTURE_STATUS_BEGIN
#undef ORT_TELEMETRY_CAPTURE_STATUS_END

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
  model_metadata_.producer_version = model.ProducerVersion();
  model_metadata_.description = model.DocString();
  model_metadata_.graph_description = model.GraphDocString();
  model_metadata_.domain = model.Domain();
  model_metadata_.version = model.ModelVersion();
  model_metadata_.custom_metadata_map = model.MetaData();
  model_metadata_.graph_name = graph.Name();

  auto add_inputs_outputs = [](const InputDefList& inputs_outputs, InputOutputDefMetaMap& map) {
    map.reserve(inputs_outputs.size());
    for (auto elem : inputs_outputs) {
      auto elem_type = utils::GetMLDataType(*elem);
      const auto* elem_shape_proto = elem->Shape();
      if (elem_shape_proto != nullptr) {
        map.emplace(elem->Name(), InputOutputDefMetaData(
                                      elem, elem_type,
                                      utils::GetTensorShapeFromTensorShapeProto(*elem_shape_proto)));
      } else {
        map.emplace(elem->Name(), InputOutputDefMetaData(elem, elem_type));
      }
    }
  };

  {
    InputOutputDefMetaMap input_defs;
    if (graph.CanOverrideInitializer()) {
      // for IR 4 or higher it is optional to have a matching graph input for an initializer, and if one exists the
      // initializer is explicitly overridable.
      add_inputs_outputs(graph.GetInputsIncludingInitializers(), input_defs);
    } else {
      // for IR < 4 we don't allow overriding initializers so that they can be treated as constant. exclude them from
      // the list of valid inputs by just using the GetInputs() list.
      add_inputs_outputs(graph.GetInputs(), input_defs);
    }
    input_def_map_.swap(input_defs);
  }

  const auto& outputs = graph.GetOutputs();
  {
    InputOutputDefMetaMap output_defs;
    add_inputs_outputs(outputs, output_defs);
    output_def_map_.swap(output_defs);
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
    logging::Severity severity = GetSeverity(session_options_);
    owned_session_logger_ = logging_manager_->CreateLogger(session_options_.session_logid, severity, false,
                                                           session_options_.session_log_verbosity_level);
    session_logger_ = owned_session_logger_.get();
  } else {
    session_logger_ = &logging::LoggingManager::DefaultLogger();
  }
}

#if !defined(ORT_MINIMAL_BUILD)

// Registers all the predefined transformers with transformer manager
common::Status InferenceSession::AddPredefinedTransformers(
    GraphTransformerManager& transformer_manager,
    TransformerLevel graph_optimization_level,
    MinimalBuildOptimizationHandling minimal_build_optimization_handling,
    RecordRuntimeOptimizationProducedNodeOpSchemaFn record_runtime_optimization_produced_op_schema_fn,
    const logging::Logger& logger) const {
  const auto& cpu_ep = *execution_providers_.Get(onnxruntime::kCpuExecutionProvider);
  for (int i = static_cast<int>(TransformerLevel::Default); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    std::function<onnxruntime::InlinedVector<std::unique_ptr<GraphTransformer>>()> transformers_to_register;

    // Enable free dimension override even when the graph optimization level is 0.
    // If the optimization level is above 0, the override will be applied during level 1 optimization.
    if (level == TransformerLevel::Default) {
      if (graph_optimization_level == TransformerLevel::Default) {
        transformers_to_register = [&]() {
          return optimizer_utils::GenerateTransformers(level, session_options_, cpu_ep, logger,
                                                       optimizers_to_disable_,
                                                       GetIntraOpThreadPoolToUse(),
                                                       &execution_providers_);
        };
      }
    } else {
      if (graph_optimization_level >= level) {
        // Generate and register transformers for level
        transformers_to_register = [&]() {
          const bool use_full_build_optimizations =
              level == TransformerLevel::Level1 ||
              minimal_build_optimization_handling == MinimalBuildOptimizationHandling::ApplyFullBuildOptimizations;

          if (use_full_build_optimizations) {
            return optimizer_utils::GenerateTransformers(level, session_options_, cpu_ep, logger,
                                                         optimizers_to_disable_,
                                                         GetIntraOpThreadPoolToUse(),
                                                         &execution_providers_);
          } else {
            const auto sat_context =
                minimal_build_optimization_handling ==
                        MinimalBuildOptimizationHandling::SaveMinimalBuildRuntimeOptimizations
                    ? SatApplyContextVariant{SatRuntimeOptimizationSaveContext{
                          record_runtime_optimization_produced_op_schema_fn}}
                    : SatApplyContextVariant{SatDirectApplicationContext{}};
            return optimizer_utils::GenerateTransformersForMinimalBuild(level, session_options_, sat_context, cpu_ep,
                                                                        logger,
                                                                        optimizers_to_disable_,
                                                                        GetIntraOpThreadPoolToUse());
          }
        };
      }
    }

    if (transformers_to_register) {  // Ensure the lambda is initialized before invoking it
      for (auto& entry : transformers_to_register()) {
        ORT_RETURN_IF_ERROR(transformer_manager.Register(std::move(entry), level));
      }
    }
  }
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) {
  if (timeout_in_ms > 0) {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
  }
  p_executor_done->Wait();

  return Status::OK();
}

const Model& InferenceSession::GetModel() const {
  return *model_;
}

const Environment& InferenceSession::GetEnvironment() const {
  return environment_;
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

#ifdef _WIN32
void InferenceSession::LogAllSessions() {
  const Env& env = Env::Default();

  std::lock_guard<std::mutex> lock(active_sessions_mutex_);
  for (const auto& session_pair : active_sessions_) {
    InferenceSession* session = session_pair.second;

    if (!session) {
      continue;
    }

    std::function<void()> log_snapshot;
    {
      std::unique_lock<std::mutex> session_lock(session->session_mutex_, std::try_to_lock);
      if (!session_lock.owns_lock()) {
        continue;
      }

      const auto session_options = session->session_options_;
      const auto* session_logger = session->session_logger_;
      auto model = session->model_;
      if (nullptr != model) {
        const onnxruntime::Graph& graph = model->MainGraph();
        auto provider_options = session->execution_providers_.GetProviderOptionsSnapshot();
        std::vector<std::string> provider_ids;
        provider_ids.reserve(provider_options.size());
        for (auto& [provider_id, options] : provider_options) {
          ORT_UNUSED_PARAMETER(options);
          provider_ids.push_back(std::move(provider_id));
        }

        log_snapshot = [&env,
                        session_options,
                        session_logger,
                        session_id = session->session_id_,
                        ir_version = model->IrVersion(),
                        producer_name = model->ProducerName(),
                        producer_version = model->ProducerVersion(),
                        domain = model->Domain(),
                        domain_to_version = graph.DomainToVersionMap(),
                        model_file_name = PathToUTF8String(graph.ModelPath().filename().native()),
                        graph_name = graph.Name(),
                        model_weight_type = session->GetWeightDataType(),
                        model_graph_hash = session->GetGraphHash(),
                        model_weight_hash = session->GetWeightHash(),
                        model_metadata = model->MetaData(),
                        event_name = session->telemetry_.event_name_,
                        provider_ids = std::move(provider_ids),
                        ep_device_types_summary = session->telemetry_.ep_device_types_summary_,
                        ep_device_vendor_ids_summary = session->telemetry_.ep_device_vendor_ids_summary_,
                        ep_versions_summary = session->telemetry_.ep_versions_summary_,
                        model_has_fp16_inputs = ModelHasFP16Inputs(graph)]() {
          env.GetTelemetryProvider().LogSessionCreation(
              session_id, ir_version, producer_name, producer_version, domain,
              domain_to_version, model_file_name, graph_name, model_weight_type, model_graph_hash, model_weight_hash,
              model_metadata, event_name, provider_ids,
              ep_device_types_summary, ep_device_vendor_ids_summary, ep_versions_summary,
              model_has_fp16_inputs, true);
          InferenceSession::TraceSessionOptions(session_options, true, *session_logger);
        };
      } else {
        log_snapshot = [session_options, session_logger]() {
          InferenceSession::TraceSessionOptions(session_options, true, *session_logger);
        };
      }
    }

    log_snapshot();
  }
}
#endif

}  // namespace onnxruntime
