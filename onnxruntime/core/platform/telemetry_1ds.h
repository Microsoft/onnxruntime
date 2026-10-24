// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/telemetry.h"
#include "core/platform/telemetry_environment.h"
#include "core/platform/telemetry_no_throw.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declarations of 1DS SDK types
namespace Microsoft::Applications::Events {
class ILogger;
class ILogManager;
class ILogConfiguration;
class EventProperties;
}  // namespace Microsoft::Applications::Events

namespace onnxruntime {

/**
 * @brief Cross-platform telemetry implementation using 1DS SDK (cpp_client_telemetry).
 *
 * This class provides telemetry logging capabilities for all platforms
 * using the cpp_client_telemetry library (1DS SDK). It implements the same interface
 * as the original WindowsTelemetry to provide consistent telemetry across all platforms.
 *
 * Configuration:
 * - Supported native build-driver targets enable telemetry unless --no_telemetry is passed.
 */
class OneDsTelemetry : public Telemetry {
 public:
  OneDsTelemetry();
  // The local diagnostic provider must outlive this instance.
  explicit OneDsTelemetry(const Telemetry& local_telemetry);
  ~OneDsTelemetry() override;

  void EnableTelemetryEvents() const override;
  void DisableTelemetryEvents() const override;
  void SetLanguageProjection(uint32_t projection) const override;

  bool IsEnabled() const override;
  unsigned char Level() const override;
  uint64_t Keyword() const override;

  void LogProcessInfo() const override;
  void LogSessionCreationStart(uint32_t session_id) const override;
  void LogEvaluationStop(uint32_t session_id) const override;
  void LogEvaluationStart(uint32_t session_id) const override;

  void LogSessionCreation(uint32_t session_id, int64_t ir_version,
                          const std::string& model_producer_name,
                          const std::string& model_producer_version,
                          const std::string& model_domain,
                          const std::unordered_map<std::string, int>& domain_to_version_map,
                          const std::string& model_file_name,
                          const std::string& model_graph_name,
                          const std::string& model_weight_type,
                          const std::string& model_graph_hash,
                          const std::string& model_weight_hash,
                          const std::unordered_map<std::string, std::string>& model_metadata,
                          const std::string& loadedFrom,
                          const std::vector<std::string>& execution_provider_ids,
                          const std::string& hardware_device_types,
                          const std::string& hardware_vendor_ids,
                          const std::string& ep_versions,
                          bool use_fp16, bool captureState) const override;

  void LogCompileModelStart(uint32_t session_id,
                            const std::string& input_source,
                            const std::string& output_target,
                            uint32_t flags,
                            int graph_optimization_level,
                            bool embed_ep_context,
                            bool has_external_initializers_file,
                            const std::vector<std::string>& execution_provider_ids) const override;

  void LogCompileModelComplete(uint32_t session_id,
                               bool success,
                               uint32_t error_code,
                               uint32_t error_category,
                               const std::string& error_message) const override;

  void LogRuntimeError(uint32_t session_id, const common::Status& status,
                       const char* file, const char* function, uint32_t line) const override;

  void LogRuntimeInferenceError(uint32_t session_id, const common::Status& status,
                                const std::string& ep_versions,
                                const std::string& ep_device_types) const override;

  void LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last,
                      int64_t total_run_duration_since_last,
                      const std::unordered_map<int64_t, long long>& duration_per_batch_size) const override;

  void LogExecutionProviderEvent(LUID* adapterLuid) const override;
  void LogDriverInfoEvent(const std::string_view device_class,
                          const std::wstring_view& driver_names,
                          const std::wstring_view& driver_versions) const override;

  void LogAutoEpSelection(uint32_t session_id, const std::string& selection_policy,
                          const std::vector<std::string>& requested_execution_provider_ids,
                          const std::vector<std::string>& available_execution_provider_ids) const override;

  void LogProviderOptions(const std::string& provider_id,
                          const std::string& provider_options_string,
                          bool capture_state) const override;

  void LogModelLoadStart(uint32_t session_id) const override;
  void LogModelLoadEnd(uint32_t session_id, const common::Status& status,
                       int64_t duration_us) const override;

  void LogSessionCreationEnd(uint32_t session_id, const common::Status& status,
                             int64_t duration_us) const override;

  void LogEpDeviceUsage(uint32_t session_id,
                        const std::string& ep_type,
                        const std::string& hardware_device_type,
                        uint32_t hardware_vendor_id,
                        uint32_t hardware_device_id,
                        const std::string& hardware_vendor,
                        const std::string& ep_vendor,
                        const std::string& ep_version,
                        int assigned_node_count,
                        uint32_t total_runs_since_last,
                        int64_t total_run_duration_since_last) const override;

  void LogRegisterEpLibraryStart(const std::string& registration_name) const override;
  void LogRegisterEpLibraryEnd(const std::string& registration_name,
                               const common::Status& status,
                               int64_t duration_us) const override;
  void LogRegisterEpLibraryWithLibPath(const std::string& registration_name,
                                       const std::string& lib_path) const override;

 private:
  static void ReportFailure(const char* operation_name, const char* message);

  template <typename Operation>
  static void RunTelemetryOperation(const char* operation_name, Operation&& operation) noexcept {
    telemetry_internal::RunTelemetryOperationNoThrow(
        std::forward<Operation>(operation),
        [operation_name](const char* message) { ReportFailure(operation_name, message); });
  }

  const Telemetry* local_telemetry_ = nullptr;

  // Initialize telemetry SDK logger
  void Initialize();

  // Shutdown telemetry SDK logger
  void Shutdown();

  // Helper to get platform name
  std::string GetPlatformInfo() const;

  // Process/system info helpers for LogProcessInfo
  std::string GetOsDescription() const;
  std::string GetCpuModel() const;
  std::string GetDeviceClass() const;
  static telemetry_detail::HostEnvironmentInfo GetHostEnvironmentInfo();
  static std::string GetProcessName();
  static std::string GetArchitecture();
  static int64_t GetTotalMemoryMB();

  // Safe async event logging.
  void LogEventAsync(::Microsoft::Applications::Events::EventProperties&& props) const;

  // All shared telemetry state below is static: OneDsTelemetry is a process-wide singleton whose
  // lifetime is gated by global_register_count_ (the first instance initializes the SDK, the last
  // tears it down), matching WindowsTelemetry. Keeping the SDK handles and state static ensures a
  // single owner regardless of how many OneDsTelemetry objects exist.

  // Mutex for thread-safe init/shutdown of the shared SDK state.
  static std::shared_mutex mutex_;

  // Telemetry SDK instances.
  // log_manager_ is owned by LogManagerProvider; logger_ is owned by log_manager_.
  static ::Microsoft::Applications::Events::ILogManager* log_manager_;
  static std::atomic<::Microsoft::Applications::Events::ILogger*> logger_;

  // SDK configuration — must outlive log_manager_ (LogManagerImpl holds a reference).
  static std::unique_ptr<::Microsoft::Applications::Events::ILogConfiguration> config_;

  // State tracking
  static std::atomic<bool> enabled_;
  static std::atomic<bool> telemetry_disabled_;
  static std::atomic<bool> network_context_suppressed_;
  static std::atomic<uint32_t> projection_;

  // Process info tracking
  static std::atomic<bool> process_info_logged_;

  // Global registration count for singleton behavior
  static std::atomic<uint32_t> global_register_count_;
  static std::mutex global_mutex_;
};

}  // namespace onnxruntime
