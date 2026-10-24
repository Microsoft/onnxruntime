// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_execution_provider.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_set>

#include "core/common/string_utils.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_telemetry.h"
#include "core/providers/qnn/rpcmem_library.h"
#include "core/providers/qnn/shared_context.h"

namespace onnxruntime {

constexpr const char* QNN = "QNN";

static std::string MakeSharedLibraryPath(std::string_view name) {
#if defined(_WIN32)
  return MakeString(name, ".dll");
#else
  return MakeString("lib", name, ".so");
#endif
}

const std::string kDefaultCpuBackendPath = MakeSharedLibraryPath("QnnCpu");
const std::string kDefaultGpuBackendPath = MakeSharedLibraryPath("QnnGpu");
const std::string kDefaultHtpBackendPath = MakeSharedLibraryPath("QnnHtp");
const std::string kDefaultSaverBackendPath = MakeSharedLibraryPath("QnnSaver");
const std::string kDefaultIrBackendPath = MakeSharedLibraryPath("QnnIr");

static bool ParseBackendTypeName(std::string_view backend_type_name, std::string& backend_path) {
  constexpr std::string_view kCpuBackendTypeName{"cpu"};
  constexpr std::string_view kGpuBackendTypeName{"gpu"};
  constexpr std::string_view kHtpBackendTypeName{"htp"};
  constexpr std::string_view kSaverBackendTypeName{"saver"};
  constexpr std::string_view kIrBackendTypeName{"ir"};

  constexpr std::array kAllowedBackendTypeNames{
      kCpuBackendTypeName,
      kGpuBackendTypeName,
      kHtpBackendTypeName,
      kSaverBackendTypeName,
      kIrBackendTypeName,
  };

  std::optional<std::string> associated_backend_path{};
  if (backend_type_name == kCpuBackendTypeName) {
    associated_backend_path = kDefaultCpuBackendPath;
  } else if (backend_type_name == kGpuBackendTypeName) {
    associated_backend_path = kDefaultGpuBackendPath;
  } else if (backend_type_name == kHtpBackendTypeName) {
    associated_backend_path = kDefaultHtpBackendPath;
  } else if (backend_type_name == kSaverBackendTypeName) {
    associated_backend_path = kDefaultSaverBackendPath;
  } else if (backend_type_name == kIrBackendTypeName) {
    associated_backend_path = kDefaultIrBackendPath;
  }

  if (associated_backend_path.has_value()) {
    backend_path = std::move(*associated_backend_path);
    return true;
  }

  std::ostringstream warning{};
  warning << "Invalid backend type name: " << backend_type_name << ". Allowed backend type names: ";
  for (size_t i = 0; i < kAllowedBackendTypeNames.size(); ++i) {
    warning << kAllowedBackendTypeNames[i];
    if (i + 1 < kAllowedBackendTypeNames.size()) {
      warning << ", ";
    }
  }
  LOGS_DEFAULT(WARNING) << warning.str();
  return false;
}

static void ParseProfilingLevel(std::string profiling_level_string,
                                qnn::ProfilingLevel& profiling_level) {
  profiling_level_string = qnn::utils::GetLowercaseString(profiling_level_string);
  LOGS_DEFAULT(INFO) << "profiling_level: " << profiling_level_string;
  if (profiling_level_string == "off") {
    profiling_level = qnn::ProfilingLevel::OFF;
  } else if (profiling_level_string == "basic") {
    profiling_level = qnn::ProfilingLevel::BASIC;
  } else if (profiling_level_string == "detailed") {
    profiling_level = qnn::ProfilingLevel::DETAILED;
  } else if (profiling_level_string == "optrace") {
    profiling_level = qnn::ProfilingLevel::OPTRACE;
  } else {
    LOGS_DEFAULT(WARNING) << "Profiling level not valid.";
  }
}

static void ParseHtpPerformanceMode(std::string htp_performance_mode_string,
                                    qnn::HtpPerformanceMode& htp_performance_mode) {
  htp_performance_mode_string = qnn::utils::GetLowercaseString(htp_performance_mode_string);
  LOGS_DEFAULT(VERBOSE) << "Htp performance mode: " << htp_performance_mode_string;
  if (htp_performance_mode_string == "burst") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBurst;
  } else if (htp_performance_mode_string == "balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBalanced;
  } else if (htp_performance_mode_string == "default") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  } else if (htp_performance_mode_string == "high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPerformance;
  } else if (htp_performance_mode_string == "high_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPowerSaver;
  } else if (htp_performance_mode_string == "low_balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowBalanced;
  } else if (htp_performance_mode_string == "low_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowPowerSaver;
  } else if (htp_performance_mode_string == "power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpPowerSaver;
  } else if (htp_performance_mode_string == "extreme_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpExtremePowerSaver;
  } else if (htp_performance_mode_string == "sustained_high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpSustainedHighPerformance;
  } else {
    LOGS_DEFAULT(WARNING) << "Htp performance mode not valid.";
  }
}

static void ParseQnnContextPriority(std::string context_priority_string, qnn::ContextPriority& context_priority) {
  context_priority_string = qnn::utils::GetLowercaseString(context_priority_string);
  LOGS_DEFAULT(VERBOSE) << "QNN context priority: " << context_priority_string;
  if (context_priority_string == "low") {
    context_priority = qnn::ContextPriority::LOW;
  } else if (context_priority_string == "normal") {
    context_priority = qnn::ContextPriority::NORMAL;
  } else if (context_priority_string == "normal_high") {
    context_priority = qnn::ContextPriority::NORMAL_HIGH;
  } else if (context_priority_string == "high") {
    context_priority = qnn::ContextPriority::HIGH;
  } else {
    context_priority = qnn::ContextPriority::UNDEFINED;
    LOGS_DEFAULT(WARNING) << "QNN context priority: " << context_priority_string << " not valid, set to undefined.";
  }
}

void QNNExecutionProvider::ParseHtpGraphFinalizationOptimizationMode(const std::string& htp_graph_finalization_opt_mode_string) {
  LOGS_DEFAULT(VERBOSE) << "HTP graph finalization optimization mode: "
                        << htp_graph_finalization_opt_mode_string;

  if (htp_graph_finalization_opt_mode_string.empty() || htp_graph_finalization_opt_mode_string == "0") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  } else if (htp_graph_finalization_opt_mode_string == "1") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode1;
  } else if (htp_graph_finalization_opt_mode_string == "2") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode2;
  } else if (htp_graph_finalization_opt_mode_string == "3") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode3;
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid HTP graph finalization optimization mode: "
                          << htp_graph_finalization_opt_mode_string;
  }
}

static void ParseHtpArchitecture(const std::string& htp_arch_string, QnnHtpDevice_Arch_t& qnn_htp_arch) {
  if (htp_arch_string.empty() || htp_arch_string == "0") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  } else if (htp_arch_string == "68") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V68;
  } else if (htp_arch_string == "69") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V69;
  } else if (htp_arch_string == "73") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V73;
  } else if (htp_arch_string == "75") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V75;
  } else if (htp_arch_string == "81") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V81;
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid HTP architecture: " << htp_arch_string;
  }
}

static void ParseOpPackages(const std::string& op_packages_string, std::vector<onnxruntime::qnn::OpPackage>& op_packages) {
  for (const auto& op_package : utils::SplitString(op_packages_string, ",", true)) {
    auto splitStrings = utils::SplitString(op_package, ":", true);
    if (splitStrings.size() < 3 || splitStrings.size() > 4) {
      LOGS_DEFAULT(WARNING) << "Invalid op_package passed, expected <OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>], got " << op_package;
      LOGS_DEFAULT(WARNING) << "Skip registration.";
      continue;
    }

    std::string op_type = std::string(splitStrings[0]);
    std::string op_package_path = std::string(splitStrings[1]);
    std::string op_package_interface = std::string(splitStrings[2]);
    std::string op_package_target;

    if (op_type.empty()) {
      LOGS_DEFAULT(WARNING) << "Op type is empty. Skip registration";
      continue;
    }

    if (op_package_path.empty()) {
      LOGS_DEFAULT(WARNING) << "Op package path is empty. Skip registration";
      continue;
    }

    if (op_package_interface.empty()) {
      LOGS_DEFAULT(WARNING) << "Op package interface is empty. Skip registration";
      continue;
    }

    LOGS_DEFAULT(VERBOSE) << "Loading op package from path: " << op_package_path << " for op " << op_type;
    LOGS_DEFAULT(VERBOSE) << "Op package interface: " << op_package_interface;
    if (splitStrings.size() > 3 && splitStrings[3].size()) {
      op_package_target = std::string(splitStrings[3]);
      LOGS_DEFAULT(VERBOSE) << "Op package target: " << op_package_target;
    }
    op_packages.push_back({op_type, op_package_path, op_package_interface, op_package_target});
  }
}

static bool ParseBoolOption(const std::string& key, bool default_value,
                            const std::unordered_map<std::string, std::string>& options) {
  bool result = default_value;
  auto it = options.find(key);
  if (it != options.end()) {
    if ("1" == it->second) {
      result = true;
    } else if ("0" == it->second) {
      result = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid value for " << key << " (" << it->second << "). Only 0 or 1 allowed.";
    }
    LOGS_DEFAULT(VERBOSE) << "Using " << key << ": " << result;
  }
  return result;
}

qnn::ProfilingLevel QNNExecutionProvider::GetProfilingLevelFromETWLevel(unsigned char level) {
  if (level == 5) {
    LOGS_DEFAULT(INFO) << "Overriding profiling to basic based on ETW level: " << static_cast<int>(level);
    return qnn::ProfilingLevel::BASIC;
  } else if (level < 5) {
    LOGS_DEFAULT(INFO) << "QNN Profiler ETW level not supported below level 5. Level: "
                       << static_cast<int>(level);
    return qnn::ProfilingLevel::OFF;
  } else {
    LOGS_DEFAULT(INFO) << "Overriding profiling to detailed based on ETW level: " << static_cast<int>(level);
    return qnn::ProfilingLevel::DETAILED;
  }
}

static std::unique_ptr<qnn::QnnSerializerConfig> ParseSerializerBackendOptions(const ProviderOptions& provider_options_map) {
  // Enable use of QNN Saver if the user provides a path the QNN Saver backend library.
  static const std::string QNN_SAVER_PATH_KEY = "qnn_saver_path";
  auto qnn_saver_path_pos = provider_options_map.find(QNN_SAVER_PATH_KEY);
  if (qnn_saver_path_pos != provider_options_map.end()) {
    LOGS_DEFAULT(VERBOSE) << "User specified QNN Saver path: " << qnn_saver_path_pos->second;
    return qnn::QnnSerializerConfig::CreateSaver(qnn_saver_path_pos->second);
  }

  static const std::string DUMP_QNN_IR_DLC = "dump_qnn_ir_dlc";
  auto dump_qnn_ir_dlc = ParseBoolOption(DUMP_QNN_IR_DLC, false, provider_options_map);

  static const std::string DUMP_QNN_IR_DLC_DIR = "dump_qnn_ir_dlc_dir";
  std::string qnn_ir_dlc_dir;
  auto qnn_ir_dlc_dir_pos = provider_options_map.find(DUMP_QNN_IR_DLC_DIR);
  if (qnn_ir_dlc_dir_pos != provider_options_map.end()) {
    qnn_ir_dlc_dir = qnn_ir_dlc_dir_pos->second;
    if (dump_qnn_ir_dlc) {
      LOGS_DEFAULT(INFO) << "IR DLC directory: " << qnn_ir_dlc_dir;
    } else {
      LOGS_DEFAULT(WARNING) << "Provided a directory for dumping QNN graphs to DLC, "
                            << "but did not set dump_qnn_ir_dlc to 1.";
    }
  }

  static const std::string QNN_IR_BACKEND_PATH = "qnn_ir_backend_path";
  std::string qnn_ir_backend_path = kDefaultIrBackendPath;
  auto qnn_ir_backend_path_pos = provider_options_map.find(QNN_IR_BACKEND_PATH);
  if (qnn_ir_backend_path_pos != provider_options_map.end()) {
    qnn_ir_backend_path = qnn_ir_backend_path_pos->second;
    if (dump_qnn_ir_dlc) {
      LOGS_DEFAULT(INFO) << "IR backend path: " << qnn_ir_backend_path;
    } else {
      LOGS_DEFAULT(WARNING) << "Provided a path to the Ir backend for dumping QNN graphs to DLC, "
                            << "but did not set dump_qnn_ir_dlc to 1.";
    }
  }

  if (dump_qnn_ir_dlc) {
    return qnn::QnnSerializerConfig::CreateIr(std::move(qnn_ir_backend_path), std::move(qnn_ir_dlc_dir));
  }

  return nullptr;
}

QNNExecutionProvider::QNNExecutionProvider(const ProviderOptions& provider_options_map,
                                           const ConfigOptions* config_options,
                                           qnn::EpContextDataCallbacks ep_context_data_callbacks)
    : IExecutionProvider{onnxruntime::kQnnExecutionProvider},
      ep_context_data_callbacks_{ep_context_data_callbacks} {
  InitOrtCppApi();
  metadef_id_generator_ = Factory<ModelMetadefIdGenerator>::Create();

  if (config_options) {
    disable_cpu_ep_fallback_ = config_options->GetConfigOrDefault(
                                   kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

    context_cache_enabled_ = config_options->GetConfigOrDefault(
                                 kOrtSessionOptionEpContextEnable, "0") == "1";
    LOGS_DEFAULT(VERBOSE) << "Context cache enable: " << context_cache_enabled_;

    std::string embed_mode = config_options->GetConfigOrDefault(
        kOrtSessionOptionEpContextEmbedMode, "0");
    if ("1" == embed_mode) {
      qnn_context_embed_mode_ = true;
    } else if ("0" == embed_mode) {
      qnn_context_embed_mode_ = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid ep.context_embed_mode: " << embed_mode << " only 0 or 1 allowed. Set to 1.";
    }
    LOGS_DEFAULT(VERBOSE) << "User specified context cache embed mode: " << qnn_context_embed_mode_;

    context_cache_path_cfg_ = config_options->GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
    LOGS_DEFAULT(VERBOSE) << "User specified context cache path: " << context_cache_path_cfg_;

    // For the case that workaround QNN context PD memory limit, user need split the model into pieces and
    // generate the QNN context model separately.
    // It could happen that the generated EPContext node in separate graph has same node name.
    // User can set this context_node_name_prefix for each split pieces to avoid that happens.
    context_node_name_prefix_ = config_options->GetConfigOrDefault(kOrtSessionOptionEpContextNodeNamePrefix, "");
    LOGS_DEFAULT(VERBOSE) << "User specified QNN context node name prefix: " << context_node_name_prefix_;

    share_ep_contexts_ =
        config_options->GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
    LOGS_DEFAULT(VERBOSE) << "User specified option - share EP contexts across sessions: " << share_ep_contexts_;

    stop_share_ep_contexts_ =
        config_options->GetConfigOrDefault(kOrtSessionOptionStopShareEpContexts, "0") == "1";
    LOGS_DEFAULT(VERBOSE) << "User specified option - stop share EP contexts across sessions: " << stop_share_ep_contexts_;
  }

  std::string backend_path{};
  {
    std::optional<std::string> backend_path_from_options{};

    static const std::string BACKEND_TYPE = "backend_type";
    static const std::string BACKEND_PATH = "backend_path";

    auto backend_type_it = provider_options_map.find(BACKEND_TYPE);
    auto backend_path_it = provider_options_map.find(BACKEND_PATH);

    if (backend_type_it != provider_options_map.end() && backend_path_it != provider_options_map.end()) {
      ORT_THROW("Only one of '", BACKEND_TYPE, "' and '", BACKEND_PATH, "' should be set.");
    }

    if (backend_type_it != provider_options_map.end()) {
      if (std::string parsed_backend_path; ParseBackendTypeName(backend_type_it->second, parsed_backend_path)) {
        backend_path_from_options = parsed_backend_path;
      } else {
        LOGS_DEFAULT(ERROR) << "Failed to parse '" << BACKEND_TYPE << "' value.";
      }
    } else if (backend_path_it != provider_options_map.end()) {
      backend_path_from_options = backend_path_it->second;
    }

    if (backend_path_from_options.has_value()) {
      backend_path = std::move(*backend_path_from_options);
    } else {
      const auto& default_backend_path = kDefaultHtpBackendPath;
      backend_path = default_backend_path;
      LOGS_DEFAULT(WARNING) << "Unable to determine backend path from provider options. Using default.";
    }

    LOGS_DEFAULT(VERBOSE) << "Using backend path: " << backend_path;
  }

  std::unique_ptr<qnn::QnnSerializerConfig> qnn_serializer_config = ParseSerializerBackendOptions(provider_options_map);

  std::string profiling_file_path;
  static const std::string PROFILING_LEVEL = "profiling_level";
  qnn::ProfilingLevel profiling_level = qnn::ProfilingLevel::OFF;
  // separate out the profiling level for ETW in case it gets disabled later when we extract the events
  // set to invalid to indicate that ETW is no enabled when we setup QNN
  qnn::ProfilingLevel profiling_level_etw = qnn::ProfilingLevel::INVALID;

#ifdef _WIN32
  auto& provider = qnn::QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0) {
      if (level != 0) {
        profiling_level_etw = GetProfilingLevelFromETWLevel(level);
      }
    }
  }
#endif  // defined(_WIN32)

  // In case ETW gets disabled later
  auto profiling_level_pos = provider_options_map.find(PROFILING_LEVEL);
  if (profiling_level_pos != provider_options_map.end()) {
    ParseProfilingLevel(profiling_level_pos->second, profiling_level);
  }

  static const std::string PROFILING_FILE = "profiling_file_path";
  auto profiling_file_pos = provider_options_map.find(PROFILING_FILE);
  if (profiling_file_pos != provider_options_map.end()) {
    profiling_file_path = profiling_file_pos->second;
  }
  LOGS_DEFAULT(VERBOSE) << "Profiling file path: " << profiling_file_path;

  static const std::string RPC_CONTROL_LANTENCY = "rpc_control_latency";
  auto latency_pos = provider_options_map.find(RPC_CONTROL_LANTENCY);
  if (latency_pos != provider_options_map.end()) {
    default_rpc_control_latency_ = static_cast<uint32_t>(std::stoul(latency_pos->second));
    LOGS_DEFAULT(VERBOSE) << "rpc_control_latency: " << default_rpc_control_latency_;
  }

  // default_htp_performance_mode from QNN EP option.
  // set it once only for each thread as default so user don't need to set it for every session run
  static const std::string HTP_PERFORMANCE_MODE = "htp_performance_mode";
  auto htp_performance_mode_pos = provider_options_map.find(HTP_PERFORMANCE_MODE);
  if (htp_performance_mode_pos != provider_options_map.end()) {
    ParseHtpPerformanceMode(htp_performance_mode_pos->second, default_htp_performance_mode_);

    if (qnn::HtpPerformanceMode::kHtpBurst == default_htp_performance_mode_) {
      default_rpc_polling_time_ = 9999;
    }
  }

  htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  static const std::string HTP_GRAPH_FINALIZATION_OPT_MODE = "htp_graph_finalization_optimization_mode";
  auto htp_graph_finalization_opt_mode_pos = provider_options_map.find(HTP_GRAPH_FINALIZATION_OPT_MODE);
  if (htp_graph_finalization_opt_mode_pos != provider_options_map.end()) {
    ParseHtpGraphFinalizationOptimizationMode(htp_graph_finalization_opt_mode_pos->second);
  }

  static const std::string QNN_CONTEXT_PRIORITY = "qnn_context_priority";
  qnn::ContextPriority context_priority = qnn::ContextPriority::NORMAL;
  auto qnn_context_priority_pos = provider_options_map.find(QNN_CONTEXT_PRIORITY);
  if (qnn_context_priority_pos != provider_options_map.end()) {
    ParseQnnContextPriority(qnn_context_priority_pos->second, context_priority);
  }

  static const std::string QNN_VTCM_MB = "vtcm_mb";
  auto qnn_vtcm_mb_pos = provider_options_map.find(QNN_VTCM_MB);
  if (qnn_vtcm_mb_pos != provider_options_map.end()) {
    vtcm_size_in_mb_ = std::stoi(qnn_vtcm_mb_pos->second);
    LOGS_DEFAULT(VERBOSE) << "vtcm_mb: " << vtcm_size_in_mb_;
    if (vtcm_size_in_mb_ <= 0) {
      LOGS_DEFAULT(WARNING) << "Skip invalid vtcm_mb: " << vtcm_size_in_mb_;
    }
  }

  static const std::string HTP_VTCM_BACKUP_BUFFER_SHARING = "enable_vtcm_backup_buffer_sharing";
  auto htp_vtcm_backup_buffer_sharing_pos = provider_options_map.find(HTP_VTCM_BACKUP_BUFFER_SHARING);
  if (htp_vtcm_backup_buffer_sharing_pos != provider_options_map.end()) {
    if ("1" == htp_vtcm_backup_buffer_sharing_pos->second) {
      enable_vtcm_backup_buffer_sharing_ = true;
    } else if ("0" != htp_vtcm_backup_buffer_sharing_pos->second) {
      LOGS_DEFAULT(WARNING) << "Invalid value entered for " << HTP_VTCM_BACKUP_BUFFER_SHARING
                            << ": " << htp_vtcm_backup_buffer_sharing_pos->second
                            << ", only 1 or 0 are allowed. Setting to 0.";
    }

    LOGS_DEFAULT(VERBOSE) << "User specified enable_vtcm_backup_buffer_sharing: " << enable_vtcm_backup_buffer_sharing_;

#if QNN_API_VERSION_MAJOR < 2 || ((QNN_API_VERSION_MAJOR) == 2 && (QNN_API_VERSION_MINOR < 26))
    if (enable_vtcm_backup_buffer_sharing_) {
      LOGS_DEFAULT(WARNING) << "User specified enable_vtcm_backup_buffer_sharing but QNN API version is older than 2.26.";
    }
#endif
  }

  static const std::string DISABLE_FILE_MAPPED_WEIGHTS = "disable_file_mapped_weights";
  auto disable_file_mapped_weights_pos = provider_options_map.find(DISABLE_FILE_MAPPED_WEIGHTS);
  if (disable_file_mapped_weights_pos != provider_options_map.end()) {
    if ("1" == disable_file_mapped_weights_pos->second) {
      enable_file_mapped_weights_ = false;
    }
    LOGS_DEFAULT(VERBOSE) << "User specified disable_file_mapped_weights: " << enable_file_mapped_weights_;
  }

#ifndef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  enable_file_mapped_weights_ = false;
  LOGS_DEFAULT(WARNING) << "File mapped weights feature is only available on Windows arm64 devices for QNN API versions >= 2.32. "
                        << "Feature will be disabled by default";
#else
  if (qnn_context_embed_mode_ && enable_file_mapped_weights_) {
    enable_file_mapped_weights_ = false;
    LOGS_DEFAULT(WARNING) << "File mapped weights feature is incompatible with embedded EP contexts. Feature will be disabled by default.";
  }
#endif

  static const std::string QNN_DEVICE_ID = "device_id";
  auto dev_id_pos = provider_options_map.find(QNN_DEVICE_ID);
  if (dev_id_pos != provider_options_map.end()) {
    int value = std::stoi(dev_id_pos->second);
    if (value < 0) {
      LOGS_DEFAULT(WARNING) << "Invalid device ID '" << value
                            << "', only >= 0 allowed. Set to " << device_id_ << ".";
    } else {
      device_id_ = static_cast<uint32_t>(value);
    }
  }

  static const std::string QNN_HTP_ARCH = "htp_arch";
  QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  auto htp_arch_pos = provider_options_map.find(QNN_HTP_ARCH);
  if (htp_arch_pos != provider_options_map.end()) {
    ParseHtpArchitecture(htp_arch_pos->second, htp_arch);
  }

  static const std::string QNN_SOC_MODEL = "soc_model";
  uint32_t soc_model = QNN_SOC_MODEL_UNKNOWN;
  auto soc_model_pos = provider_options_map.find(QNN_SOC_MODEL);
  if (soc_model_pos != provider_options_map.end()) {
    int value = std::stoi(soc_model_pos->second);
    if (value < 0) {
      LOGS_DEFAULT(WARNING) << "Invalid SoC Model '" << value
                            << "', only >= 0 allowed. Set to " << soc_model << ".";
    } else {
      soc_model = static_cast<uint32_t>(value);
    }
  }

  static const std::string QNN_OP_PACKAGES = "op_packages";
  std::vector<onnxruntime::qnn::OpPackage> op_packages;
  auto op_packages_pos = provider_options_map.find(QNN_OP_PACKAGES);
  if (op_packages_pos != provider_options_map.end()) {
    ParseOpPackages(op_packages_pos->second, op_packages);
  }

  static const std::string QNN_HTP_FP16_MODE = "enable_htp_fp16_precision";
  auto htp_fp16_mode_pos = provider_options_map.find(QNN_HTP_FP16_MODE);
  if (htp_fp16_mode_pos != provider_options_map.end()) {
    if ("1" == htp_fp16_mode_pos->second) {
      enable_HTP_FP16_precision_ = true;
    } else if ("0" == htp_fp16_mode_pos->second) {
      enable_HTP_FP16_precision_ = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid enable_htp_fp16_precision: " << enable_HTP_FP16_precision_ << " only 0 or 1 allowed. Set to 0.";
    }
    LOGS_DEFAULT(VERBOSE) << "User specified enable_htp_fp16_precision: " << enable_HTP_FP16_precision_;
  }

  if (qnn_context_embed_mode_ && share_ep_contexts_) {
    LOGS_DEFAULT(ERROR) << "[EP context generation:] Weight sharing enabled conflict with EP context embed mode. Inference will not work as expected!";
  }

  if (qnn_context_embed_mode_ && enable_vtcm_backup_buffer_sharing_) {
    LOGS_DEFAULT(ERROR) << "[EP context generation:] VTCM backup buffer sharing enabled conflict with EP context embed mode. Inference will not work as expected!";
  }

  const bool has_ep_context_data_callback = ep_context_data_callbacks_.read_func != nullptr ||
                                            ep_context_data_callbacks_.write_func != nullptr;
  ORT_ENFORCE(!(has_ep_context_data_callback && share_ep_contexts_),
              "QNN shared EP contexts do not support EPContext data callbacks.");
  ORT_ENFORCE(!(has_ep_context_data_callback && enable_vtcm_backup_buffer_sharing_),
              "QNN VTCM backup buffer sharing does not support EPContext data callbacks.");

  // Add this option because this feature requires QnnSystem lib and it's no supported for Windows x86_64 platform
  enable_spill_fill_buffer_ = ParseBoolOption("enable_htp_spill_fill_buffer", false, provider_options_map);

  model_settings_.offload_graph_io_quantization = ParseBoolOption("offload_graph_io_quantization", true,
                                                                  provider_options_map);

  model_settings_.htp_bf16_enable = ParseBoolOption("htp_bf16_enable", false, provider_options_map);

  if (disable_cpu_ep_fallback_ && model_settings_.offload_graph_io_quantization) {
    LOGS_DEFAULT(INFO) << "Fallback to CPU EP is disabled, but user tried to configure QNN EP to offload graph I/O "
                       << "quantization/dequantization to another EP. These are conflicting options. Fallback to CPU "
                       << "EP will remain disabled and graph I/O quantization/dequantization will not be offloaded "
                       << "to another EP.";
    model_settings_.offload_graph_io_quantization = false;
  }

  static const std::string QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED = "enable_htp_shared_memory_allocator";
  enable_htp_shared_mem_allocator_ = ParseBoolOption(QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED, false, provider_options_map);
  if (enable_htp_shared_mem_allocator_) {
    // Initialize rpcmem_library_.
    // This library is only necessary for the inference (for the shared memory allocator), if we are in context
    // generation stage, there is no need to load it as no allocations will be made.
    if (!context_cache_enabled_) {
      rpcmem_library_ = std::make_shared<qnn::RpcMemLibrary>();
    }
    model_settings_.htp_shared_memory = enable_htp_shared_mem_allocator_;
  }

  if (enable_file_mapped_weights_ && !rpcmem_library_) {
    // Attempt to init rpcmem_library_ if needed. If this fails, then
    // disable file mapped weights and proceed with normal operation
    try {
      rpcmem_library_ = std::make_shared<qnn::RpcMemLibrary>();
    } catch (const std::exception& e) {
      LOGS_DEFAULT(WARNING) << "Unable to load RPCMem library: " << e.what()
                            << " - Disabling file mapped weights.";
      enable_file_mapped_weights_ = false;
    }
  }

  dump_json_qnn_graph_ = ParseBoolOption("dump_json_qnn_graph", false, provider_options_map);

  static const std::string QNN_GRAPH_DUMP_DIR = "json_qnn_graph_dir";
  auto json_graphs_dir_pos = provider_options_map.find(QNN_GRAPH_DUMP_DIR);

  if (json_graphs_dir_pos != provider_options_map.end()) {
    json_qnn_graph_dir_ = json_graphs_dir_pos->second;
    if (dump_json_qnn_graph_) {
      LOGS_DEFAULT(INFO) << "JSON graphs directory: " << json_qnn_graph_dir_;
    } else {
      LOGS_DEFAULT(WARNING) << "Provided a directory for dumping QNN JSON graphs, "
                            << "but did not enable dumping of QNN JSON graphs.";
    }
  }

  static const std::string QNN_HTP_EXTENDED_UDMA_MODE = "extended_udma";
  auto htp_extended_udma_pos = provider_options_map.find(QNN_HTP_EXTENDED_UDMA_MODE);
  if (htp_extended_udma_pos != provider_options_map.end()) {
    if ("1" == htp_extended_udma_pos->second) {
      enable_htp_extended_udma_mode_ = true;
    } else if ("0" == htp_extended_udma_pos->second) {
      enable_htp_extended_udma_mode_ = false;
    } else {
      LOGS_DEFAULT(WARNING) << "Invalid extended_udma mode: " << enable_htp_extended_udma_mode_ << " only 0 or 1 allowed. Set to 0.";
    }
    LOGS_DEFAULT(VERBOSE) << "User specified extended_udma mode: " << enable_htp_extended_udma_mode_;
  }

  // Option to skip QNN API interface version check to use other QNN library other than default.
  static const std::string SKIP_QNN_VERSION_CHECK = "skip_qnn_version_check";
  auto skip_qnn_version_check = ParseBoolOption(SKIP_QNN_VERSION_CHECK, false, provider_options_map);

  // For context binary generation with weight sharing enabled, use the QnnBackendManager from the shared context if it exits
  // So that all graphs from later sessions will be compiled into the same QNN context
  if (((context_cache_enabled_ && share_ep_contexts_) || enable_vtcm_backup_buffer_sharing_) && SharedContext::GetInstance().GetSharedQnnBackendManager()) {
    qnn_backend_manager_ = SharedContext::GetInstance().GetSharedQnnBackendManager();
    // Clear the QnnBackendManager from singleton to stop the resource share
    if (stop_share_ep_contexts_) {
      SharedContext::GetInstance().ResetSharedQnnBackendManager();
    }
  } else {
    qnn_backend_manager_ = qnn::QnnBackendManager::Create(
        qnn::QnnBackendManagerConfig{backend_path,
                                     profiling_level_etw,
                                     profiling_level,
                                     profiling_file_path,
                                     context_priority,
                                     std::move(qnn_serializer_config),
                                     device_id_,
                                     htp_arch,
                                     soc_model,
                                     op_packages,
                                     skip_qnn_version_check});
    if (enable_vtcm_backup_buffer_sharing_) {
      SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_);
    }
  }

#if defined(_WIN32)
  if (onnxruntime::logging::EtwRegistrationManager::SupportsETW()) {
    auto& etwRegistrationManager = logging::EtwRegistrationManager::Instance();
    // Register callback for ETW capture state (rundown)
    callback_ETWSink_provider_ = onnxruntime::logging::EtwRegistrationManager::EtwInternalCallback(
        [&etwRegistrationManager, this](
            LPCGUID SourceId,
            ULONG IsEnabled,
            UCHAR Level,
            ULONGLONG MatchAnyKeyword,
            ULONGLONG MatchAllKeyword,
            PEVENT_FILTER_DESCRIPTOR FilterData,
            PVOID CallbackContext) {
          ORT_UNUSED_PARAMETER(SourceId);
          ORT_UNUSED_PARAMETER(MatchAnyKeyword);
          ORT_UNUSED_PARAMETER(MatchAllKeyword);
          ORT_UNUSED_PARAMETER(FilterData);
          ORT_UNUSED_PARAMETER(CallbackContext);

          if (IsEnabled == EVENT_CONTROL_CODE_ENABLE_PROVIDER) {
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Logs)) != 0) {
              auto ortETWSeverity = etwRegistrationManager.MapLevelToSeverity();
              (void)qnn_backend_manager_->ResetQnnLogLevel(ortETWSeverity);
            }
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0) {
              if (Level != 0) {
                // Commenting out Dynamic QNN Profiling for now
                // There seems to be a crash in 3rd party QC QnnHtp.dll with this.
                // Repro Scenario - start ETW tracing prior to session creation.
                //    Then disable/enable ETW Tracing with the code below uncommented a few times
                // auto profiling_level_etw = GetProfilingLevelFromETWLevel(Level);
                // (void)qnn_backend_manager_->SetProfilingLevelETW(profiling_level_etw);
                //
                // NOTE(1/2/2025): It is possible that the above was not working in part because it is using the
                // *logging ETW* subsystem to modify profiling, which should use an entirely different
                // ETW provider (see QnnTelemetry). Should add callbacks for profiling to the QnnTelemetry ETW provider.
              }
            }
          }

          if (IsEnabled == EVENT_CONTROL_CODE_DISABLE_PROVIDER) {
            // (void)qnn_backend_manager_->SetProfilingLevelETW(qnn::ProfilingLevel::INVALID);
            (void)qnn_backend_manager_->ResetQnnLogLevel(std::nullopt);
          }
        });
    etwRegistrationManager.RegisterInternalCallback(callback_ETWSink_provider_);
  }
#endif
}

QNNExecutionProvider::~QNNExecutionProvider() {
  if (qnn_backend_manager_) {
    auto thread_id = std::this_thread::get_id();
    qnn_backend_manager_->RemovePerThreadHtpPowerConfigMapping(thread_id);

    std::lock_guard<std::mutex> lock(config_id_mutex_);
    if (htp_power_config_id_.has_value()) {
      ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->DestroyHTPPowerConfigID(*htp_power_config_id_));
    }
  }

  // Unregister the ETW callback
#if defined(_WIN32)
  if (callback_ETWSink_provider_ != nullptr) {
    logging::EtwRegistrationManager::Instance().UnregisterInternalCallback(callback_ETWSink_provider_);
  }
#endif
}

// Logs information about the supported/unsupported nodes.
static void LogNodeSupport(const logging::Logger& logger,
                           logging::Severity log_severity,
                           logging::DataType log_data_type,
                           const onnxruntime::CodeLocation& call_site,
                           const qnn::IQnnNodeGroup& qnn_node_group,
                           Status support_status) {
  if (!logger.OutputIsEnabled(log_severity, log_data_type)) {
    return;
  }

  size_t num_nodes = 0;
  std::ostringstream oss;
  for (const NodeUnit* node_unit : qnn_node_group.GetNodeUnits()) {
    for (const Node* node : node_unit->GetAllNodesInGroup()) {
      oss << "\tOperator type: " << node->OpType()
          << " Node name: " << node->Name()
          << " Node index: " << node->Index() << std::endl;
      num_nodes += 1;
    }
  }
  if (!support_status.IsOK()) {
    oss << "\tREASON : " << support_status.ErrorMessage() << std::endl;
  }

  auto log_capture = Factory<logging::Capture>::Create(logger, log_severity,
                                                       logging::Category::onnxruntime,
                                                       log_data_type, call_site);
  log_capture->Stream()
      << (support_status.IsOK() ? "Validation PASSED " : "Validation FAILED ") << "for " << num_nodes
      << " nodes in " << qnn_node_group.Type() << " (" << qnn_node_group.GetTargetNodeUnit()->OpType() << ") :"
      << std::endl
      << oss.str();
}

std::unordered_set<const Node*>
QNNExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer,
                                        const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                        const size_t node_unit_size,
                                        const logging::Logger& logger) const {
  std::unordered_set<const Node*> supported_nodes{};

  // Util function that initializes a table that maps a graph input or output name to its index.
  auto init_input_output_index_map = [](std::unordered_map<std::string, size_t>& index_map,
                                        const std::vector<const NodeArg*>& node_args) {
    const size_t num_args = node_args.size();
    for (size_t i = 0; i < num_args; i++) {
      index_map.emplace(node_args[i]->Name(), i);
    }
  };

  std::unordered_map<std::string, size_t> model_input_index_map;
  init_input_output_index_map(model_input_index_map, graph_viewer.GetInputs());  // GetInputs excludes initializers.

  std::unordered_map<std::string, size_t> model_output_index_map;
  init_input_output_index_map(model_output_index_map, graph_viewer.GetOutputs());

  auto qnn_model_wrapper = qnn::QnnModelWrapper(graph_viewer, logger,
                                                qnn_backend_manager_->GetQnnInterface(),
                                                qnn_backend_manager_->GetQnnBackendHandle(),
                                                model_input_index_map,
                                                model_output_index_map,
                                                qnn_backend_manager_->GetQnnBackendType(),
                                                model_settings_);

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_size);

  if (Status status = qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper,
                                            node_unit_map, node_unit_size, logger);
      !status.IsOK()) {
    LOGS(logger, ERROR) << status.ErrorMessage();
    return {};
  }

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    Status status = qnn_node_group->IsSupported(qnn_model_wrapper, logger);
    const bool supported = status.IsOK();

    constexpr auto log_severity = logging::Severity::kINFO;
    constexpr auto log_data_type = logging::DataType::SYSTEM;
    if (logger.OutputIsEnabled(log_severity, log_data_type)) {
      LogNodeSupport(logger, log_severity, log_data_type, ORT_WHERE, *qnn_node_group, status);
    }

    if (supported) {
      for (const NodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
        for (const Node* node : node_unit->GetAllNodesInGroup()) {
          supported_nodes.insert(node);
        }
      }
    }
  }

  return supported_nodes;
}

static bool EpSharedContextsHasAllGraphs(const onnxruntime::GraphViewer& graph_viewer,
                                         const logging::Logger& logger) {
  for (const auto& node : graph_viewer.Nodes()) {
    NodeAttrHelper node_helper(node);
    std::string cache_source = node_helper.Get(qnn::SOURCE, "");

    std::transform(cache_source.begin(),
                   cache_source.end(),
                   cache_source.begin(),
                   [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

    if (qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
      const std::string& graph_name = node.Name();
      bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
      if (!has_shared_qnn_model) {
        LOGS(logger, VERBOSE) << "Graph: " << graph_name << " from EpContext node not found from shared EP contexts.";
        return false;
      }
    }
  }

  return true;
}

static bool EpSharedContextsHasAllGraphs(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                                         const logging::Logger& logger) {
  for (auto fused_node_and_graph : fused_nodes_and_graphs) {
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
    const Node& ep_context_node = *graph_viewer.Nodes().begin();
    NodeAttrHelper node_helper(ep_context_node);
    std::string cache_source = node_helper.Get(qnn::SOURCE, "");

    const std::string& graph_name = ep_context_node.Name();
    bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
    if (!has_shared_qnn_model) {
      LOGS(logger, VERBOSE) << "Graph: " << graph_name << " from EpContext node not found from shared EP contexts.";
      return false;
    }
  }

  return true;
}

static bool IsMainQnnEpContextNode(const Node& node) {
  if (node.OpType() != qnn::EPCONTEXT_OP) {
    return false;
  }
  NodeAttrHelper node_helper(node);
  const bool is_main_context = node_helper.Get(qnn::MAIN_CONTEXT, static_cast<int64_t>(0));
  const auto cache_source = qnn::utils::GetLowercaseString(node_helper.Get(qnn::SOURCE, ""));
  return is_main_context && (cache_source == "qnnexecutionprovider" || cache_source == "qnn");
}

static void GetMainEPCtxNodes(const onnxruntime::GraphViewer& graph_viewer,
                              std::unordered_set<const Node*>& ep_context_nodes,
                              const logging::Logger& logger) {
  for (const auto& node : graph_viewer.Nodes()) {
    if (IsMainQnnEpContextNode(node)) {
      LOGS(logger, VERBOSE) << "EPContext Node found: [1] index: [" << node.Index()
                            << "] name: [" << node.Name();
      ep_context_nodes.insert(&node);
    }
  }
}

// For model with EPContext, filter in EPContext nodes only, and make sure each partition only has one single EPContext node
static void PartitionCtxModel(const onnxruntime::GraphViewer& graph_viewer,
                              const size_t num_nodes_in_graph,
                              std::vector<std::unique_ptr<ComputeCapability>>& result,
                              const std::function<std::string()>& gen_metadef_name,
                              const logging::Logger& logger) {
  std::unordered_set<const Node*> supported_nodes{};
  std::vector<std::vector<const Node*>> supported_groups{};

  for (const auto& node : graph_viewer.Nodes()) {
    NodeAttrHelper node_helper(node);
    std::string cache_source = node_helper.Get(qnn::SOURCE, "");

    std::transform(cache_source.begin(),
                   cache_source.end(),
                   cache_source.begin(),
                   [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

    if (qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
      LOGS(logger, VERBOSE) << "Node supported: [1] index: [" << node.Index()
                            << "] name: [" << node.Name()
                            << "] Operator type: [EPContext"
                            << "] index: [" << node.Index() << "]";
      supported_nodes.insert(&node);

      std::vector<const Node*> supported_group{&node};
      supported_groups.emplace_back(std::move(supported_group));
    }
  }

  result.reserve(supported_groups.size());

  std::transform(
      supported_groups.begin(), supported_groups.end(),
      std::back_inserter(result),
      [&](const auto& supported_partition) {
        return utils::MakeComputeCapability(graph_viewer, supported_partition, gen_metadef_name, QNN,
                                            /*drop_constant_initializers*/ false);  // TODO: could this be set to true?
      });

  const size_t num_of_partitions = result.size();
  const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
                                      ", number of nodes in the graph: ", num_nodes_in_graph,
                                      ", number of nodes supported by QNN: ", num_of_partitions);
  LOGS(logger, INFO) << summary_msg;

  return;
}

// Figure out the context cache Onnx file path to decide the folder location
static void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                        const onnxruntime::PathString& model_path_string,
                                        onnxruntime::PathString& context_model_path) {
  // always try the path set by user first, it's the only way to set it if load model from memory
  if (!user_context_cache_path.empty()) {
    context_model_path = ToPathString(user_context_cache_path);
  } else if (!model_path_string.empty()) {  // model loaded from file
    context_model_path = model_path_string;
  }
}

uint32_t QNNExecutionProvider::GetEpContextDataCallbackRequirements(const GraphViewer& graph_viewer) const {
  if (graph_viewer.IsSubgraph() || (!enable_vtcm_backup_buffer_sharing_ && !enable_file_mapped_weights_)) {
    return OrtEpContextDataCallbackSupportFlags_NONE;
  }

  for (const auto& node : graph_viewer.Nodes()) {
    if (IsMainQnnEpContextNode(node)) {
      NodeAttrHelper node_helper(node);
      if (!node_helper.Get(qnn::EMBED_MODE, true) && !node_helper.Get(qnn::EP_CACHE_CONTEXT, "").empty()) {
        return OrtEpContextDataCallbackSupportFlags_READ;
      }
    }
  }
  return OrtEpContextDataCallbackSupportFlags_NONE;
}

std::vector<std::unique_ptr<ComputeCapability>>
QNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/,
                                    const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                    IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }
  const size_t num_nodes_in_graph = static_cast<size_t>(graph_viewer.NumberOfNodes());

  const auto& logger = *GetLogger();

  // Check BF16 compatibility early
  if (model_settings_.htp_bf16_enable) {
    // Check SoC model
    uint32_t soc_model = qnn_backend_manager_->GetSocModel();
    if (soc_model == QNN_SOC_MODEL_UNKNOWN) {
      LOGS(logger, WARNING) << "BF16 mode is enabled but soc_model is not specified. "
                            << "Both parameters must be set together for BF16 support. "
                            << "QNN EP will not handle any nodes.";
      return result;  // Empty result means QNN EP won't handle any nodes
    } else if (soc_model < 88) {
      LOGS(logger, WARNING) << "BF16 mode is enabled but SoC model is " << soc_model
                            << " (expected 88 and above). QNN EP will not handle any nodes.";
      return result;  // Empty result means QNN EP won't handle any nodes
    }

    LOGS(logger, INFO) << "BF16 mode enabled with compatible hardware: SoC " << soc_model;
  }

  bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph_viewer);

  const auto gen_metadef_name = [&]() {
    uint64_t model_hash;
    int metadef_id = metadef_id_generator_->GenerateId(graph_viewer, model_hash);
    return MakeString(QNN, context_node_name_prefix_, "_", model_hash, "_", metadef_id);
  };

  // share ep contexts is enabled
  // check the ep_shared_contexts to see if it contains all the graphs in the context model
  // directly use the resource from ep_shared_contexts if it has all the graphs needed by the current session
  // no need to setup QNN backend
  if (is_qnn_ctx_model && share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
    if (EpSharedContextsHasAllGraphs(graph_viewer, logger)) {
      PartitionCtxModel(graph_viewer, num_nodes_in_graph, result, gen_metadef_name, logger);
      return result;
    }
  }

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
  const bool callback_backed_context = is_qnn_ctx_model && ep_context_data_callbacks_.read_func != nullptr;
  const bool effective_vtcm_backup_buffer_sharing = enable_vtcm_backup_buffer_sharing_ && !callback_backed_context;
  const bool effective_file_mapped_weights = enable_file_mapped_weights_ && !callback_backed_context;
  if (callback_backed_context && (enable_vtcm_backup_buffer_sharing_ || enable_file_mapped_weights_)) {
    LOGS(logger, VERBOSE) << "Disabling path-based QNN EPContext loading optimizations because a read callback is set.";
  }

  if (effective_vtcm_backup_buffer_sharing || effective_file_mapped_weights) {
    std::unordered_set<const Node*> ep_ctx_nodes;
    GetMainEPCtxNodes(graph_viewer, ep_ctx_nodes, logger);

    onnxruntime::PathString context_model_path;
    GetContextOnnxModelFilePath(context_cache_path_cfg_, graph_viewer.ModelPath().native(), context_model_path);

    std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

    for (auto& ep_ctx_node : ep_ctx_nodes) {
      NodeAttrHelper node_helper(*ep_ctx_node);
      std::string ep_cache_context_value = node_helper.Get(qnn::EP_CACHE_CONTEXT, "");
      if (ep_cache_context_value.empty()) {
        continue;
      }
      auto validate_status = utils::ValidateExternalDataPath(
          std::filesystem::path(context_model_path), std::filesystem::path(ep_cache_context_value));
      if (!validate_status.IsOK()) {
        LOGS(logger, ERROR) << "QNN EP context path validation failed: " << validate_status.ErrorMessage();
        return result;
      }
      std::string context_bin_filepath(parent_path.string());
      context_bin_filepath.append("/").append(ep_cache_context_value);
      if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
        context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
        // Push context bin filepath for lookup between sessions
        context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
      }
      context_bin_map.at(context_bin_filepath)->push_back(ep_ctx_node->Name());
    }
  }

  // It will load the QnnSystem lib if is_qnn_ctx_model=true, and
  // delay the Qnn context creation to Compile() using the cached context binary
  // or generate context cache enable, need to use use QnnSystem lib to parse the binary to get the max spill fill buffer size
  auto rt = qnn_backend_manager_->SetupBackend(logger, is_qnn_ctx_model,
                                               context_cache_enabled_ && enable_spill_fill_buffer_,
                                               share_ep_contexts_,
                                               effective_vtcm_backup_buffer_sharing,
                                               effective_file_mapped_weights,
                                               rpcmem_library_,
                                               context_bin_map,
                                               enable_htp_extended_udma_mode_);

  context_bin_map.clear();

  if (Status::OK() != rt) {
    LOGS(logger, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
    return result;
  }

  if (IsNpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
    // Set the power config id and the default power mode from provider option for main thread,
    // otherwise it will mess up the power mode if user just create session without run it.
    CreateHtpPowerConfigId();
  }

  // Report error if QNN CPU backend is loaded while CPU fallback is disabled
  if (disable_cpu_ep_fallback_ && qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::CPU) {
    LOGS(logger, ERROR) << "Qnn CPU backend is loaded while CPU fallback is disabled.";
    return result;
  }

  if ((context_cache_enabled_ || is_qnn_ctx_model) && !IsQpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
    LOGS(logger, ERROR) << "Qnn context cache only works for HTP/DSP/GPU backend.";
    return result;
  }

  // For model with EPContext, make sure each partition only has one single EPContext node
  if (is_qnn_ctx_model) {
    PartitionCtxModel(graph_viewer, num_nodes_in_graph, result, gen_metadef_name, logger);
    return result;
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = GetQDQNodeUnits(graph_viewer, logger);

  // remove is_qnn_ctx_model related code
  const auto supported_nodes = GetSupportedNodes(graph_viewer, node_unit_map,
                                                 node_unit_holder.size(), logger);

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
    std::stringstream ss;
    const size_t num_node_units = node_unit_holder.size();

    for (size_t i = 0; i < num_node_units; ++i) {
      const auto& node_unit = node_unit_holder[i];

      if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
        ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
        if (i == num_node_units - 1) {
          ss << ", ";
        }
      }
    }

    return ss.str();
  };

  if (supported_nodes.empty()) {
    LOGS(logger, INFO) << "Number of partitions supported by QNN EP: 0";
    return result;
  }

  size_t num_of_supported_nodes = 0;

  // Create partitions from supported nodes.
  std::vector<std::unique_ptr<ComputeCapability>> partitions = utils::CreateSupportedPartitions(
      graph_viewer, supported_nodes, {}, gen_metadef_name, QNN, kQnnExecutionProvider, &node_unit_map);

  // Filter out partitions that consist of a single QuantizeLinear or DequantizeLinear node.
  // We also count the number of supported nodes in all valid partitions.
  for (auto& partition : partitions) {
    bool is_valid_partition = true;
    size_t nodes_in_partition = 0;

    if (partition && ComputeCapability__SubGraph(*partition)) {
      const auto& subgraph = ComputeCapability__SubGraph(*partition);
      const auto& subgraph_nodes = IndexedSubGraph__Nodes(*subgraph);

      nodes_in_partition = subgraph_nodes.size();

      if (nodes_in_partition == 1 && !is_qnn_ctx_model) {
        const Node* node = graph_viewer.GetNode(subgraph_nodes[0]);

        if (!node) {
          LOGS(logger, ERROR) << "QNN EP: Invalid node in partition of one node.";
          is_valid_partition = false;
        } else if (node->OpType() == "QuantizeLinear" || node->OpType() == "DequantizeLinear") {
          LOGS(logger, WARNING) << "QNN EP does not support a single Quantize/Dequantize node in a partition.";
          is_valid_partition = false;
        }
      }
    } else {
      LOGS(logger, ERROR) << "QNN EP: Invalid partition.";
      is_valid_partition = false;
    }

    if (is_valid_partition) {
      result.push_back(std::move(partition));
      num_of_supported_nodes += nodes_in_partition;
    }
  }  // for

  const size_t num_of_partitions = result.size();
  const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
                                      ", number of nodes in the graph: ", num_nodes_in_graph,
                                      ", number of nodes supported by QNN: ", num_of_supported_nodes);
  LOGS(logger, INFO) << summary_msg;

  // Print list of unsupported nodes to the ERROR logger if the CPU EP
  // has been disabled for this inference session.
  if (!is_qnn_ctx_model && disable_cpu_ep_fallback_ && num_nodes_in_graph != num_of_supported_nodes) {
    LOGS(logger, ERROR) << "Unsupported nodes in QNN EP: " << get_unsupported_node_names();
  }

  return result;
}

DataLayout QNNExecutionProvider::GetPreferredLayout() const {
  return DataLayout::NHWC;
}

std::optional<bool> QNNExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                                       std::string_view node_op_type,
                                                                       DataLayout target_data_layout) const {
  if (target_data_layout != DataLayout::NHWC) {
    return std::nullopt;
  }

  if (node_domain == kOnnxDomain && node_op_type == "Upsample") {
    // Upsample is translated to QNN's Resize, which requires the NHWC layout for processing.
    return true;
  }

  return std::nullopt;
}

Status QNNExecutionProvider::CreateComputeFunc(std::vector<NodeComputeInfo>& node_compute_funcs,
                                               const logging::Logger& logger) {
  NodeComputeInfo compute_info;
  compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
    LOGS(logger, VERBOSE) << "compute_info.create_state_func context->node_name: " << context->node_name;
    *state = qnn_models_[context->node_name].get();
    return 0;
  };

  compute_info.release_state_func = [](FunctionState state) {
    // the 'state' is a qnn::QnnModel managed by unique_ptr
    ORT_UNUSED_PARAMETER(state);
  };

  compute_info.compute_func = [&logger](FunctionState state, const OrtApi*, OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(state);
    Status result = model->ExecuteGraph(ctx, logger);
    return result;
  };

  node_compute_funcs.push_back(compute_info);

  return Status::OK();
}

void QNNExecutionProvider::InitQnnHtpGraphConfigs(qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
  if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
    if (htp_graph_finalization_opt_mode_ != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
      htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
      htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
      htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(htp_graph_finalization_opt_mode_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
      graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config->customConfig = htp_graph_opt_config;
    }

    if (vtcm_size_in_mb_ > 0) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
      htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(vtcm_size_in_mb_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
      graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
    }

    if (enable_HTP_FP16_precision_) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
      htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

      gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
      graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_precision_config->customConfig = htp_graph_precision_config;
    }
  }
}

Status QNNExecutionProvider::CompileFromOrtGraph(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs,
                                                 const logging::Logger& logger) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(qnn_backend_manager_.get());

    std::vector<const QnnGraph_Config_t*> all_graph_configs;

    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> htp_graph_configs_builder(QNN_GRAPH_CONFIG_INIT,
                                                                                                    QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
    InitQnnHtpGraphConfigs(htp_graph_configs_builder);

    const QnnGraph_Config_t** htp_configs = htp_graph_configs_builder.GetQnnConfigs();
    if (htp_configs) {
      // Reserve enough for configs + nullptr
      all_graph_configs.reserve(htp_graph_configs_builder.GetSize() + 1);
      for (const QnnGraph_Config_t** config = htp_configs; *config; ++config) {
        all_graph_configs.push_back(*config);
      }
    }

    qnn::QnnSerializerConfig* qnn_serializer_config = qnn_backend_manager_->GetQnnSerializerConfig();
    if (qnn_serializer_config) {
      // We don't bother reserving here to keep the API simpler. Also note that if we're here,
      // we're likely debugging and not waiting for inference.
      qnn_serializer_config->SetGraphName(fused_node.Name());
      const QnnGraph_Config_t** serializer_configs = qnn_serializer_config->Configure();
      if (serializer_configs) {
        for (const QnnGraph_Config_t** config = serializer_configs; *config; ++config) {
          all_graph_configs.push_back(*config);
        }
      }
    }

    const QnnGraph_Config_t** all_graph_configs_ptr = nullptr;
    if (!all_graph_configs.empty()) {
      all_graph_configs.push_back(nullptr);
      all_graph_configs_ptr = all_graph_configs.data();
    }

    std::string json_graph_filepath;

    if (dump_json_qnn_graph_) {
      namespace fs = std::filesystem;
      fs::path path = fs::path(json_qnn_graph_dir_) / fs::path(fused_node.Name() + ".json");
      json_graph_filepath = path.string();
    }

    ORT_RETURN_IF_ERROR(qnn_model->ComposeGraph(graph_viewer, fused_node, model_settings_, logger,
                                                all_graph_configs_ptr, json_graph_filepath));
    ORT_RETURN_IF_ERROR(qnn_model->FinalizeGraphs(logger));
    ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput(logger));

    LOGS(logger, VERBOSE) << "fused node name: " << fused_node.Name();
    qnn_models_.emplace(fused_node.Name(), std::move(qnn_model));

    ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
  }
  return Status::OK();
}

Status QNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  const auto& logger = *GetLogger();
  bool is_qnn_ctx_model = qnn::IsFusedGraphHasCtxNode(fused_nodes_and_graphs);

  onnxruntime::PathString context_model_path;
  bool is_ctx_file_exist = false;
  if (is_qnn_ctx_model || context_cache_enabled_) {
    const onnxruntime::GraphViewer& graph_viewer_0(fused_nodes_and_graphs[0].filtered_graph);
    // Figure out the EP context model path from model path or session option
    GetContextOnnxModelFilePath(context_cache_path_cfg_,
                                graph_viewer_0.ModelPath().native(),
                                context_model_path);
  }

  if (is_qnn_ctx_model) {
    // Get QnnModel from EP shared contexts
    if (share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
      if (EpSharedContextsHasAllGraphs(fused_nodes_and_graphs, logger)) {
        for (auto fused_node_and_graph : fused_nodes_and_graphs) {
          const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
          const Node& ep_context_node = *graph_viewer.Nodes().begin();
          const Node& fused_node = fused_node_and_graph.fused_node;
          const std::string& graph_meta_id = fused_node.Name();
          std::string key = ep_context_node.Name();
          auto qnn_model_shared = SharedContext::GetInstance().GetSharedQnnModel(key);
          ORT_RETURN_IF(nullptr == qnn_model_shared, "Graph: " + key + " not found from shared EP contexts.");
          ORT_RETURN_IF_ERROR(qnn_model_shared->SetGraphInputOutputInfo(graph_viewer, fused_node, logger));
          ORT_RETURN_IF_ERROR(qnn_model_shared->SetupQnnInputOutput(logger));
          qnn_models_.emplace(graph_meta_id, std::move(qnn_model_shared));
          ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
        }
        return Status::OK();
      }
    }

    // Table<EPContext node name, QnnModel>, the node name is the graph_meta_id (old) created from user model which used to generate the EP context model
    // for this session (created from an EP context model), the graph_meta_id is new
    qnn::QnnModelLookupTable qnn_models;

    std::vector<int> main_context_pos_list;
    ORT_RETURN_IF_ERROR(qnn::GetMainContextNode(fused_nodes_and_graphs, main_context_pos_list));
    uint32_t total_context_size = SafeInt<uint32_t>(main_context_pos_list.size());

    int64_t max_spill_fill_size = 0;

    // Adjust the main_context_pos_list, move the one with max spill fill buffer to the beginning
    // HTP spill fill buffer only works for multiple QNN contexts generated after QNN v2.28
    if (total_context_size > 1) {
      ORT_RETURN_IF_ERROR(qnn::TryGetMaxSpillFillSize(fused_nodes_and_graphs, total_context_size,
                                                      max_spill_fill_size, main_context_pos_list));
    }

    for (auto main_context_pos : main_context_pos_list) {
      const onnxruntime::GraphViewer& main_ctx_graph_viewer(fused_nodes_and_graphs[main_context_pos].filtered_graph);
      // Create QNN context from the cached binary, deserialize the QNN graph from the binary
      ORT_RETURN_IF_ERROR(qnn::LoadQnnCtxFromOnnxGraph(main_ctx_graph_viewer,
                                                       context_model_path,
                                                       qnn_backend_manager_.get(),
                                                       qnn_models,
                                                       logger,
                                                       max_spill_fill_size,
                                                       ep_context_data_callbacks_));
    }

    for (auto fused_node_and_graph : fused_nodes_and_graphs) {
      const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
      const Node& ep_context_node = *graph_viewer.Nodes().begin();
      const Node& fused_node = fused_node_and_graph.fused_node;
      const std::string& graph_meta_id = fused_node.Name();
      std::string key = ep_context_node.Name();
      ORT_RETURN_IF(qnn_models.find(key) == qnn_models.end(), key + " key name not exist in table qnn_models.");
      auto qnn_model = std::move(qnn_models[key]);
      ORT_RETURN_IF_ERROR(qnn_model->SetGraphInputOutputInfo(graph_viewer, fused_node, logger));
      ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput(logger));

      // fused node name is QNNExecutionProvider_QNN_[hash_id]_[id]
      // the name here must be same with context->node_name in compute_info
      qnn_models_.emplace(graph_meta_id, std::move(qnn_model));
      qnn_models.erase(key);

      ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
    }

    if (share_ep_contexts_ && qnn_models.size() > 0) {
      std::vector<std::unique_ptr<qnn::QnnModel>> shared_qnn_models;
      for (auto& [key, value] : qnn_models) {
        shared_qnn_models.push_back(std::move(qnn_models[key]));
      }
      std::string duplicate_graph_names;
      bool has_duplicate_graph = SharedContext::GetInstance().SetSharedQnnModel(std::move(shared_qnn_models),
                                                                                duplicate_graph_names);
      ORT_RETURN_IF(has_duplicate_graph, "Duplicate graph names detect across sessions: " + duplicate_graph_names);
    }

    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(CompileFromOrtGraph(fused_nodes_and_graphs, node_compute_funcs, logger));
  // Generate QNN context model if it's QDQ model + context_cache_enabled=true + not exist already
  if (!is_qnn_ctx_model && context_cache_enabled_ && !is_ctx_file_exist) {
    // All partitioned graph share single QNN context, included in the same context binary
    uint64_t buffer_size(0);
    auto context_buffer = qnn_backend_manager_->GetContextBinaryBuffer(buffer_size);
    // Get max spill fill buffer size
    uint64_t max_spill_fill_buffer_size = 0;
    if (enable_spill_fill_buffer_) {
      ORT_RETURN_IF_ERROR(qnn_backend_manager_->GetMaxSpillFillBufferSize(context_buffer.get(),
                                                                          buffer_size,
                                                                          max_spill_fill_buffer_size));
    }
    qnn_ep_context_model_ = Factory<Model>::Create(std::string{"qnn_ep_context_model"}, false, logger);
    ORT_RETURN_IF_ERROR(qnn::CreateEPContextNodes(qnn_ep_context_model_.get(),
                                                  context_buffer.get(),
                                                  buffer_size,
                                                  qnn_backend_manager_->GetSdkVersion(),
                                                  fused_nodes_and_graphs,
                                                  qnn_models_,
                                                  context_model_path,
                                                  qnn_context_embed_mode_,

                                                  max_spill_fill_buffer_size,
                                                  logger,
                                                  share_ep_contexts_,
                                                  stop_share_ep_contexts_,
                                                  ep_context_data_callbacks_));

    if (share_ep_contexts_ && !stop_share_ep_contexts_ &&
        nullptr == SharedContext::GetInstance().GetSharedQnnBackendManager()) {
      ORT_RETURN_IF_NOT(SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_),
                        "Failed to set shared QnnBackendManager.");
    }
  }
  return Status::OK();
}

const InlinedVector<const Node*> QNNExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_nodes;
  if (qnn_ep_context_model_) {
    const auto& graph = qnn_ep_context_model_->MainGraph();
    for (gsl::not_null<const Node*> node : Graph__Nodes(graph)) {
      ep_context_nodes.push_back(graph.GetNode(node->Index()));
    }
  }

  return ep_context_nodes;
}

static bool TryGetConfigEntry(const ConfigOptions& config_options, const std::string& key, std::string& value) {
  std::optional<std::string> new_value = config_options.GetConfigEntry(key);
  if (!new_value.has_value()) {
    return false;
  }

  value = *new_value;
  return true;
}

bool QNNExecutionProvider::GetPerThreadHtpPowerConfigs(qnn::PerThreadHtpPowerConfigs_t& per_thread_htp_power_configs,
                                                       const ConfigOptions& config_options) {
  qnn::HtpPerformanceMode pre_run_htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  qnn::HtpPerformanceMode post_run_htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;

  bool configs_set = false;
  std::string htp_perf_mode = "";
  if (TryGetConfigEntry(config_options, kOrtRunOptionsConfigQnnPerfMode, htp_perf_mode)) {
    ParseHtpPerformanceMode(htp_perf_mode, pre_run_htp_performance_mode);
  }

  if (TryGetConfigEntry(config_options, kOrtRunOptionsConfigQnnPerfModePostRun, htp_perf_mode)) {
    ParseHtpPerformanceMode(htp_perf_mode, post_run_htp_performance_mode);
  }

  std::string rpc_latency = "";
  uint32_t rpc_control_latency = 0;
  if (TryGetConfigEntry(config_options, kOrtRunOptionsConfigQnnRpcControlLatency, rpc_latency)) {
    rpc_control_latency = static_cast<uint32_t>(std::stoul(rpc_latency));
    per_thread_htp_power_configs.rpc_control_latency = rpc_control_latency;
    configs_set = true;

    LOGS_DEFAULT(VERBOSE) << "rpc_control_latency: " << rpc_control_latency;
  }

  uint32_t rpc_polling_time = 0;
  if (qnn::HtpPerformanceMode::kHtpBurst == pre_run_htp_performance_mode) {
    rpc_polling_time = 9999;
  }

  if (qnn::HtpPerformanceMode::kHtpDefault != pre_run_htp_performance_mode) {
    per_thread_htp_power_configs.pre_run_perf_mode = pre_run_htp_performance_mode;
    // rpc polling time will only be updated with perf mode changes
    per_thread_htp_power_configs.rpc_polling_time = rpc_polling_time;
    configs_set = true;
  }

  if (qnn::HtpPerformanceMode::kHtpDefault != post_run_htp_performance_mode) {
    per_thread_htp_power_configs.post_run_perf_mode = post_run_htp_performance_mode;
    configs_set = true;
  }

  return configs_set;
}

Status QNNExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  auto backend_type = qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return Status::OK();
  }

  const ConfigOptions& config_options = RunOptions__GetConfigOptions(run_options);

  uint32_t htp_power_config_id = 0;
  if (GetHtpPowerConfigId(htp_power_config_id)) {
    auto thread_id = std::this_thread::get_id();
    qnn::PerThreadHtpPowerConfigs_t per_thread_htp_power_configs;
    if (GetPerThreadHtpPowerConfigs(per_thread_htp_power_configs, config_options)) {
      per_thread_htp_power_configs.power_config_id = htp_power_config_id;
      ORT_RETURN_IF_ERROR(qnn_backend_manager_->AddPerThreadHtpPowerConfigMapping(thread_id,
                                                                                  per_thread_htp_power_configs));
    }
  }

  std::string lora_config = "";
  if (TryGetConfigEntry(config_options, kOrtRunOptionsConfigQnnLoraConfig, lora_config)) {
    LOGS_DEFAULT(VERBOSE) << "lora_config: " << lora_config;
    ORT_RETURN_IF_ERROR(qnn_backend_manager_->ParseLoraConfig(lora_config));
  }

  return Status::OK();
}

Status QNNExecutionProvider::OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& run_options) {
  ORT_UNUSED_PARAMETER(run_options);

  auto backend_type = qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return Status::OK();
  }

  uint32_t htp_power_config_id;
  if (GetHtpPowerConfigId(htp_power_config_id)) {
    auto thread_id = std::this_thread::get_id();
    qnn_backend_manager_->RemovePerThreadHtpPowerConfigMapping(thread_id);
  }

  return Status::OK();
}

std::vector<AllocatorPtr> QNNExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators{};

  if (IsHtpSharedMemoryAllocatorAvailable()) {
    LOGS_DEFAULT(INFO) << "Creating HtpSharedMemoryAllocator.";

    AllocatorFactory rpcmem_allocator_factory = [this](OrtDevice::DeviceId) {
      return std::make_unique<qnn::HtpSharedMemoryAllocator>(rpcmem_library_);
    };

    AllocatorCreationInfo rpcmem_allocator_creation_info{rpcmem_allocator_factory,
                                                         /* device_id */ 0,
                                                         /* use_arena */ false};

    allocators.emplace_back(CreateAllocator(rpcmem_allocator_creation_info));
  }

  return allocators;
}

OrtDevice QNNExecutionProvider::GetOrtDeviceByMemType(OrtMemType /* em_type */) const {
  // We are disabling the HTP shared memory allocator for intermediate values
  // until we learn how to deal with memhandle costs.
  // if (IsHtpSharedMemoryAllocatorAvailable()) {
  //  return qnn::HtpSharedMemoryAllocator::AssociatedMemoryInfo().device;
  //}
  // Default CPU allocator
  return default_device_;
}

Status QNNExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                 gsl::span<const char* const> values) {
  if (keys.size() != values.size()) {
    LOGS_DEFAULT(ERROR) << "SetEpDynamicOptions: number of keys (" << keys.size()
                        << ") does not equal number of values (" << values.size() << ").";
  }
  auto key_it = keys.begin();
  auto value_it = values.begin();

  while (key_it != keys.end() && value_it != values.end()) {
    std::string key(*key_it);
    std::string value(*value_it);

    if (key == kOrtEpDynamicOptionsWorkloadType) {
      if (value == "Default") {
        ORT_RETURN_IF_ERROR(qnn_backend_manager_->ResetContextPriority());
      } else if (value == "Efficient") {
        ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetContextPriority(qnn::ContextPriority::LOW));
      } else {
        LOGS_DEFAULT(ERROR) << "Invalid EP Workload Type: " << value;
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid EP Workload Type.");
      }
    } else if (key == kOrtEpDynamicOptionsQnnHtpPerformanceMode) {
      auto backend_type = qnn_backend_manager_->GetQnnBackendType();
      if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
        return Status::OK();
      }
      qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
      ParseHtpPerformanceMode(value, htp_performance_mode);

      uint32_t rpc_polling_time = 0;
      if (htp_performance_mode == qnn::HtpPerformanceMode::kHtpBurst) {
        rpc_polling_time = 9999;
      }

      uint32_t htp_power_config_id = 0;
      if (GetHtpPowerConfigId(htp_power_config_id)) {
        ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetHtpPowerConfigs(htp_power_config_id,
                                                                     htp_performance_mode,
                                                                     rpc_polling_time,
                                                                     default_rpc_control_latency_));
      }
    } else {
      LOGS_DEFAULT(ERROR) << "EP Dynamic Option \"" << key << "\" is not currently supported.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported EP Dynamic Option");
    }

    key_it++;
    value_it++;
  }

  return Status::OK();
}

bool QNNExecutionProvider::GetHtpPowerConfigId(uint32_t& htp_power_config_id) {
  std::lock_guard<std::mutex> lock(config_id_mutex_);
  if (!htp_power_config_id_.has_value()) {
    return false;
  }

  htp_power_config_id = *htp_power_config_id_;
  return true;
}

void QNNExecutionProvider::CreateHtpPowerConfigId() const {
  std::lock_guard<std::mutex> lock(config_id_mutex_);
  if (htp_power_config_id_.has_value()) {
    return;
  }

  constexpr uint32_t core_id = 0;
  uint32_t htp_power_config_id;

  Status rt = qnn_backend_manager_->CreateHtpPowerCfgId(device_id_, core_id, htp_power_config_id);

  if (rt.IsOK()) {
    htp_power_config_id_ = htp_power_config_id;

    rt = qnn_backend_manager_->SetHtpPowerConfigs(htp_power_config_id,
                                                  default_htp_performance_mode_,
                                                  default_rpc_polling_time_,
                                                  default_rpc_control_latency_);

    if (!rt.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Unable to set HTP power configurations.";
    }
  } else {
    LOGS_DEFAULT(ERROR) << "Failed to create HTP power config id.";
  }
}

}  // namespace onnxruntime
