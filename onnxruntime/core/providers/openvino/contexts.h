// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <filesystem>
#include <memory>
#include "core/common/common.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/shared_library/provider_api.h"
#include "ov_bin_manager.h"
#include "ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {

namespace fs = std::filesystem;

using config_t = std::map<std::string, ov::AnyMap>;
using reshape_t = std::map<std::string, ov::PartialShape>;
using layout_t = std::map<std::string, ov::Layout>;

struct ProviderInfo {
  std::string device_type{""};             // [device_type]: Overrides the accelerator hardware type and
                                           // precision with these values at runtime.
  std::string precision{""};               // [precision]: Sets the inference precision for execution.
                                           // Supported precision for devices are
                                           // CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                           // Not setting precision will execute with optimized precision for
                                           // best inference latency. set Precision=ACCURACY for executing
                                           // models with input precision for best accuracy.
  uint32_t num_of_threads{0};              // [num_of_threads]: Overrides the accelerator default value of
                                           // number of threads with this value at runtime.
  config_t load_config{};                  // JSON config map to load custom OV parameters.
  fs::path cache_dir{""};                  // [cache_dir]: specify the path to
                                           // dump and load the blobs for the model caching/kernel caching
                                           // (GPU) feature. If blob files are already present,
                                           // it will be directly loaded.
  reshape_t reshape{};                     // Used for reshaping the ov input tensor shape at runtime.
  layout_t layout{};                       // Used for specifying the ov input/output tensor layout at runtime.
  std::string model_priority{"DEFAULT"};   // High-level OpenVINO model priority hint
                                           // Defines what model should be provided with more performant
                                           // bounded resource first
  uint32_t num_streams{1};                 // [num_streams]: Option that specifies the number of parallel
                                           // inference requests to be processed on a given `device_type`.
                                           // Overrides the accelerator default value of number of streams
                                           // with this value at runtime.
  void* context{nullptr};                  // OpenCL context
  bool enable_opencl_throttling{false};    // [enable_opencl_throttling]: Enables OpenCL queue throttling for
                                           // GPU device (Reduces CPU Utilization when using GPU)
  bool disable_dynamic_shapes{false};      // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to
                                           // static shape at runtime and execute.
  bool enable_qdq_optimizer{false};        // Enables QDQ pruning for efficient inference latency with NPU
  bool enable_causallm{false};             // Enables Causal LM Compilation for ORT GenAI OVEP Pass
  bool so_context_enable{false};           // ORT session option
  bool so_disable_cpu_ep_fallback{false};  // ORT session option
  bool so_context_embed_mode{false};       // ORT session option
  bool so_share_ep_contexts{false};        // ORT session option
  bool so_stop_share_ep_contexts{false};   // ORT session option
  fs::path so_context_file_path{};         // ORT session option
  OrtReadNamedBufferFunc ep_context_data_read_func{nullptr};
  void* ep_context_data_read_state{nullptr};
  OrtWriteNamedBufferFunc ep_context_data_write_func{nullptr};
  void* ep_context_data_write_state{nullptr};
  const ConfigOptions* config_options{NULL};
  const std::unordered_set<std::string> valid_provider_keys = {"device_type", "device_id", "device_luid", "cache_dir", "precision",
                                                               "load_config", "context", "num_of_threads", "model_priority", "num_streams", "enable_opencl_throttling", "enable_qdq_optimizer",
                                                               "enable_causallm", "disable_dynamic_shapes", "reshape_input", "layout"};
};

struct RuntimeConfig {
  std::unordered_map<std::string, std::string> options;
  std::optional<std::string> Get(const std::string& key) const {
    auto it = options.find(key);
    return it != options.end() ? std::optional{it->second} : std::nullopt;
  }
};

// Holds context applicable to the entire EP instance.
struct SessionContext : ProviderInfo {
  SessionContext(const ProviderInfo& info) : ProviderInfo{info} {
    InitRuntimeConfig();
  }

  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::filesystem::path onnx_model_path_name;
  uint32_t onnx_opset_version{0};
  mutable bool is_wholly_supported_graph = false;  // Value is set to mutable to modify from capability
  mutable bool has_external_weights = false;       // Value is set to mutable to modify from capability
  const std::vector<uint32_t> OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR};
  const std::string openvino_sdk_version = std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR);

  RuntimeConfig runtime_config;

  const std::filesystem::path& GetModelPath() const {
    return onnx_model_path_name.empty() ? so_context_file_path : onnx_model_path_name;
  }

  const std::filesystem::path& GetOutputModelPath() const {
    return so_context_file_path.empty() ? onnx_model_path_name : so_context_file_path;
  }

  std::filesystem::path GetOutputBinPath() const {
    const auto& bin_file_name = GetOutputModelPath();
    if (bin_file_name.empty()) {
      return {};
    }
    return BinManager::GetBinPathForModel(bin_file_name);
  }

 private:
  void InitRuntimeConfig() {
    if (config_options) {
      runtime_config.options = config_options->GetConfigOptionsMap();
    }
  }
};

// Holds context specific to subgraph.
struct SubGraphContext {
  using string_index_map_t = std::unordered_map<std::string, uint32_t>;
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_npu_config = false;
  bool is_constant = false;
  void* context = 0;
  std::string subgraph_name;
  string_index_map_t input_names;
  string_index_map_t output_names;
  std::string model_precision;
  bool is_ep_ctx_graph = false;
  bool is_ep_ctx_ovir_encapsulated = false;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
