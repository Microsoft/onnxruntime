// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"
#include "core/providers/openvino/openvino_parser_utils.h"
#include "ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {
void ParseConfigOptions(ProviderInfo& pi) {
  if (pi.config_options == nullptr)
    return;

  pi.so_disable_cpu_ep_fallback = pi.config_options->GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
  pi.so_context_enable = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  pi.so_context_embed_mode = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
  pi.so_share_ep_contexts = pi.config_options->GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
  pi.so_context_file_path = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  pi.so_stop_share_ep_contexts = pi.config_options->GetConfigOrDefault(kOrtSessionOptionStopShareEpContexts, "0") == "1";

  if (pi.so_share_ep_contexts) {
    // Set default NPU compilation params only if user hasn't provided them
    auto& npu_config = pi.load_config["NPU"];
    if (npu_config.find("NPU_COMPILATION_MODE_PARAMS") == npu_config.end()) {
      npu_config["NPU_COMPILATION_MODE_PARAMS"] = "enable-wd-blockarg-input=true compute-layers-with-higher-precision=Sqrt,Power,ReduceSum";
    }
  }
}

void* ParseUint64(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    uint64_t number = std::strtoull(provider_options.at(option_name).data(), nullptr, 16);
    return reinterpret_cast<void*>(number);
  } else {
    return nullptr;
  }
}

bool ParseBooleanOption(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    const auto& value = provider_options.at(option_name);
    if (value == "true" || value == "True") {
      return true;
    } else if (value == "false" || value == "False") {
      return false;
    } else {
      ORT_THROW("[ERROR] [OpenVINO-EP] ", option_name, " should be a boolean.\n");
    }
  }
  return false;
}

std::string ParseDeviceType(std::shared_ptr<OVCore> ov_core, const ProviderOptions& provider_options) {
  std::set<std::string> supported_device_types = {"CPU", "GPU", "NPU"};
  std::set<std::string> supported_device_modes = {"AUTO", "HETERO", "MULTI"};
  std::vector<std::string> devices_to_check;
  std::string selected_device;
  std::vector<std::string> luid_list;
  std::string device_mode = "";
  std::map<std::string, std::string> ov_luid_map;

  if (provider_options.contains("device_type")) {
    selected_device = provider_options.at("device_type");
    std::erase(selected_device, ' ');
    if (selected_device == "AUTO") return selected_device;

    if (auto delimit = selected_device.find(":"); delimit != std::string::npos) {
      device_mode = selected_device.substr(0, delimit);
      if (supported_device_modes.contains(device_mode)) {
        const auto& devices = selected_device.substr(delimit + 1);
        devices_to_check = split(devices, ',');
        ORT_ENFORCE(devices_to_check.size() > 0, "Mode AUTO/HETERO/MULTI should have devices listed based on priority");
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Invalid device_type is selected. Supported modes are AUTO/HETERO/MULTI");
      }
    } else {
      devices_to_check.push_back(selected_device);
    }
  } else {
    // Take default behavior from project configuration
#if defined OPENVINO_CONFIG_CPU
    selected_device = "CPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_GPU
    selected_device = "GPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_NPU
    selected_device = "NPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
    selected_device = DEVICE_NAME;

    // Add sub-devices to check-list
    int delimit = selected_device.find(":");
    const auto& devices = selected_device.substr(delimit + 1);
    devices_to_check = split(devices, ',');
#endif
  }

  // Get the LUID passed from the provider option in a comma separated string list
  // Compare each of the LUID's against the LUID obtained using ov property and map with the right device
  if (provider_options.contains("device_luid")) {
    std::string luid_str = provider_options.at("device_luid");
    std::erase(luid_str, ' ');
    luid_list = split(luid_str, ',');
  }

  for (auto device : devices_to_check) {
    bool device_found = false;
    // Check deprecated device format (CPU_FP32, GPU.0_FP16, etc.) and remove the suffix in place
    // Suffix will be parsed in ParsePrecision
    if (auto delimit = device.find("_"); delimit != std::string::npos) {
      device = device.substr(0, delimit);
    }
    // Just the device name without .0, .1, etc. suffix
    auto device_prefix = device;
    // Check if device index is appended (.0, .1, etc.), if so, remove it
    if (auto delimit = device_prefix.find("."); delimit != std::string::npos)
      device_prefix = device_prefix.substr(0, delimit);
    if (supported_device_types.contains(device_prefix)) {
      try {
        std::vector<std::string> available_devices = ov_core->GetAvailableDevices(device_prefix);
        // Here we need to find the full device name (with .idx, but without _precision)
        if (std::find(std::begin(available_devices), std::end(available_devices), device) != std::end(available_devices))
          device_found = true;
        if (!device_found) {
          ORT_THROW("[ERROR] [OpenVINO] Device ", device, " is not available");
        }
        if (device_prefix != "CPU" && luid_list.size() > 0) {
          for (const auto& dev : available_devices) {
            ov::device::LUID ov_luid = OVCore::Get()->core.get_property(dev, ov::device::luid);
            std::stringstream ov_luid_str;
            ov_luid_str << ov_luid;
            ov_luid_map.emplace(ov_luid_str.str(), dev);
          }
        }
      } catch (const char* msg) {
        ORT_THROW(msg);
      }
    }
  }
  if (luid_list.size() > 0) {
    std::string ov_luid_devices;
    for (const auto& luid_str : luid_list) {
      if (ov_luid_map.contains(luid_str)) {
        std::string ov_dev = ov_luid_map.at(luid_str);
        std::string ov_dev_strip = split(ov_dev, '.')[0];
        if (std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev) != std::end(devices_to_check) ||
            std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev_strip) != std::end(devices_to_check)) {
          if (!ov_luid_devices.empty()) ov_luid_devices = ov_luid_devices + ",";
          ov_luid_devices = ov_luid_devices + ov_dev;
        } else {
          ORT_THROW(" LUID : ", ov_dev, " does not match with device_type : ", selected_device);
        }
      } else {
        ORT_THROW(provider_options.at("device_luid"), " does not exist for the selected device_type : ", selected_device);
      }
    }
    if (!device_mode.empty()) {
      selected_device = device_mode + ":" + ov_luid_devices;
      for (const auto& dev_str : devices_to_check) {
        const std::string default_dev = split(dev_str, '.')[0];

        if (ov_luid_devices.find(default_dev) == std::string::npos)
          selected_device = selected_device + "," + dev_str;
      }
    } else {
      selected_device = std::move(ov_luid_devices);
    }
  }

  LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
  return selected_device;
}

void ParseProviderOptions([[maybe_unused]] ProviderInfo& result, [[maybe_unused]] const ProviderOptions& config_options) {}

static void ParseInnerMap(const nlohmann::json& json_map, ov::AnyMap& inner_map, size_t level = 0) {
  const size_t max_levels = 8;
  if (level >= max_levels) {
    ORT_THROW("ParseInnerMap: load_config can have only up to " + std::to_string(max_levels) +
              " levels of nested maps. Current level = " + std::to_string(level));
  }

  if (!json_map.is_object()) {
    ORT_THROW("ParseInnerMap: Expected an object as input");
  }

  for (auto& [inner_key, inner_value] : json_map.items()) {
    if (inner_value.is_string()) {
      inner_map[inner_key] = ov::Any(inner_value.get<std::string>());
    } else if (inner_value.is_number_integer()) {
      inner_map[inner_key] = ov::Any(inner_value.get<int64_t>());
    } else if (inner_value.is_number_float()) {
      inner_map[inner_key] = ov::Any(inner_value.get<double>());
    } else if (inner_value.is_boolean()) {
      inner_map[inner_key] = ov::Any(inner_value.get<bool>());
    } else if (inner_value.is_object()) {
      auto inner_inner_map = ov::AnyMap();
      ParseInnerMap(inner_value, inner_inner_map, level + 1);
      inner_map[inner_key] = std::move(inner_inner_map);
    } else {
      ORT_THROW("load_config: unsupported JSON value type=" + std::string(inner_value.type_name()) + ", for key=" + inner_key);
    }
  }
}

// Initializes a ProviderInfo struct from a ProviderOptions map and a ConfigOptions map.
static void ParseProviderInfo(const ProviderOptions& provider_options,
                              const ConfigOptions* config_options,
                              /*output*/ ProviderInfo& pi) {
  pi.config_options = config_options;

  // Lambda function to check for invalid keys and throw an error
  auto validateKeys = [&]() {
    for (const auto& pair : provider_options) {
      if (pi.valid_provider_keys.find(pair.first) == pi.valid_provider_keys.end()) {
        ORT_THROW("Invalid provider_option key: " + pair.first);
      }
    }
  };
  validateKeys();

  std::string bool_flag = "";

  // Minor optimization: we'll hold an OVCore reference to ensure we don't create a new core between ParseDeviceType and
  // (potential) SharedContext creation.
  auto ov_core = OVCore::Get();
  pi.device_type = ParseDeviceType(std::move(ov_core), provider_options);

  if (provider_options.contains("device_id")) {
    std::string dev_id = provider_options.at("device_id").data();
    LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                          << "Upgrade to set deice_type and precision session options.\n";
    if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
      pi.device_type = std::move(dev_id);
    } else {
      ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
    }
  }
  if (provider_options.contains("cache_dir")) {
    pi.cache_dir = provider_options.at("cache_dir");
  }

  pi.precision = OpenVINOParserUtils::ParsePrecision(provider_options, pi.device_type, "precision");

  if (provider_options.contains("reshape_input")) {
    pi.reshape = OpenVINOParserUtils::ParseInputShape(provider_options.at("reshape_input"));
  }

  if (provider_options.contains("layout")) {
    pi.layout = OpenVINOParserUtils::ParseLayout(provider_options.at("layout"));
  }

  if (provider_options.contains("load_config")) {
    auto parse_config = [&](const std::string& config_str) -> std::map<std::string, ov::AnyMap> {
      // If the config string is empty, return an empty map and skip processing
      if (config_str.empty()) {
        LOGS_DEFAULT(WARNING) << "Empty OV Config Map passed. Skipping load_config option parsing.\n";
        return {};
      }

      std::stringstream input_str_stream(config_str);
      std::map<std::string, ov::AnyMap> target_map;

      try {
        nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

        if (!json_config.is_object()) {
          ORT_THROW("Invalid JSON structure: Expected an object at the root.");
        }

        for (auto& [key, value] : json_config.items()) {
          ov::AnyMap inner_map;
          std::set<std::string> valid_ov_devices = {"CPU", "GPU", "NPU", "AUTO", "HETERO", "MULTI"};
          // Ensure the key is one of "CPU", "GPU", or "NPU"
          if (valid_ov_devices.find(key) == valid_ov_devices.end()) {
            LOGS_DEFAULT(WARNING) << "Unsupported device key: " << key << ". Skipping entry.\n";
            continue;
          }

          // Ensure that the value for each device is an object (PROPERTY -> VALUE)
          if (!value.is_object()) {
            ORT_THROW("Invalid JSON structure: Expected an object for device properties.");
          }

          ParseInnerMap(value, inner_map);
          target_map[key] = std::move(inner_map);
        }
      } catch (const nlohmann::json::parse_error& e) {
        // Handle syntax errors in JSON
        ORT_THROW("JSON parsing error: " + std::string(e.what()));
      } catch (const nlohmann::json::type_error& e) {
        // Handle invalid type accesses
        ORT_THROW("JSON type error: " + std::string(e.what()));
      } catch (const std::exception& e) {
        ORT_THROW("Error parsing load_config Map: " + std::string(e.what()));
      }
      return target_map;
    };

    pi.load_config = parse_config(provider_options.at("load_config"));
  }

  pi.context = ParseUint64(provider_options, "context");
#if defined(IO_BUFFER_ENABLED)
  // a valid context must be provided to enable IO Buffer optimizations
  if (pi.context == nullptr) {
#undef IO_BUFFER_ENABLED
#define IO_BUFFER_ENABLED = 0
    LOGS_DEFAULT(WARNING) << "Context is not set. Disabling IO Buffer optimization";
  }
#endif

  if (provider_options.contains("num_of_threads")) {
    if (!std::all_of(provider_options.at("num_of_threads").begin(),
                     provider_options.at("num_of_threads").end(), ::isdigit)) {
      ORT_THROW("[ERROR] [OpenVINO-EP] Number of threads should be a number. \n");
    }
    pi.num_of_threads = std::stoi(provider_options.at("num_of_threads"));
    if (pi.num_of_threads <= 0) {
      pi.num_of_threads = 1;
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                            << "Executing with num_threads=1";
    }
  }

  if (provider_options.contains("model_priority")) {
    pi.model_priority = provider_options.at("model_priority").data();
    std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
    if (std::find(supported_priorities.begin(), supported_priorities.end(),
                  pi.model_priority) == supported_priorities.end()) {
      pi.model_priority = "DEFAULT";
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                            << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                            << "Executing with model_priorty=DEFAULT";
    }
  }

  if (provider_options.contains("num_streams")) {
    pi.num_streams = std::stoi(provider_options.at("num_streams"));
    if (pi.num_streams <= 0) {
      pi.num_streams = 1;
      LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                            << "Executing with num_streams=1";
    }
  }
  try {
    pi.enable_opencl_throttling = ParseBooleanOption(provider_options, "enable_opencl_throttling");

    pi.enable_qdq_optimizer = ParseBooleanOption(provider_options, "enable_qdq_optimizer");

    pi.enable_causallm = ParseBooleanOption(provider_options, "enable_causallm");

    pi.disable_dynamic_shapes = ParseBooleanOption(provider_options, "disable_dynamic_shapes");
  } catch (std::string msg) {
    ORT_THROW(msg);
  }

  // Should likely account for meta devices as well, but for now keep the current behavior.
  // Keep the user setting for CPU/GPU. For NPU, disable dynamic shapes by default and enable them when CausalLM is enabled.
  const bool is_npu_device = pi.device_type.find("NPU") != std::string::npos;
  if (is_npu_device) {
    pi.disable_dynamic_shapes = true;
    if (pi.enable_causallm) {
      LOGS_DEFAULT(WARNING) << "Enabling dynamic shapes for NPU because CausalLM is enabled.";
      pi.disable_dynamic_shapes = false;
    }
  }
}

struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(ProviderInfo provider_info, std::shared_ptr<OVCore> ov_core)
      : provider_info_(std::move(provider_info)), ov_core_(ov_core) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ParseConfigOptions(provider_info_);
    return std::make_unique<OpenVINOExecutionProvider>(provider_info_);
  }

  // Called by InferenceSession when registering EPs. Allows creation of an EP instance that is initialized with
  // session-level configurations.
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // Extract those EP options into a new "provider_options" map.
    std::string lowercase_ep_name = kOpenVINOExecutionProvider;
    std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    std::string key_prefix = "ep.";
    key_prefix += lowercase_ep_name;
    key_prefix += ".";

    std::unordered_map<std::string, std::string> provider_options;
    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    if (provider_options.find("device_type") == provider_options.end()) {
      // Preserve the device selected during factory creation unless the session-level
      // OpenVINO provider options explicitly override it.
      provider_options["device_type"] = provider_info_.device_type;
    }

    ProviderInfo provider_info = provider_info_;
    ParseProviderInfo(provider_options, &config_options, provider_info);
    ParseConfigOptions(provider_info);
    session_options.GetEpContextDataCallbacks(&provider_info.ep_context_data_read_func,
                                              &provider_info.ep_context_data_read_state,
                                              &provider_info.ep_context_data_write_func,
                                              &provider_info.ep_context_data_write_state);

    auto ov_ep = std::make_unique<OpenVINOExecutionProvider>(provider_info);
    ov_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return ov_ep;
  }

  // This is called during session creation when AppendExecutionProvider_V2 is used.
  // This one is called because ParseProviderInfo / ParseConfigOptions, etc. are already
  // performed in CreateIExecutionProvider, and so provider_info_ has already been populated.
  std::unique_ptr<IExecutionProvider> CreateProvider_V2(const OrtSessionOptions& session_options,
                                                        const OrtLogger& session_logger) {
    ProviderInfo provider_info = provider_info_;
    session_options.GetEpContextDataCallbacks(&provider_info.ep_context_data_read_func,
                                              &provider_info.ep_context_data_read_state,
                                              &provider_info.ep_context_data_write_func,
                                              &provider_info.ep_context_data_write_state);
    auto ov_ep = std::make_unique<OpenVINOExecutionProvider>(provider_info);
    ov_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return ov_ep;
  }

 private:
  ProviderInfo provider_info_;
  std::shared_ptr<OVCore> ov_core_;
};

struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    return OVCore::Get()->GetAvailableDevices();
  }
};

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &info_; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    if (void_params == nullptr) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<void*, 2> pointers_array = *reinterpret_cast<const std::array<void*, 2>*>(void_params);
    const ProviderOptions provider_options = *reinterpret_cast<ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<ConfigOptions*>(pointers_array[1]);

    ProviderInfo pi;
    ParseProviderInfo(provider_options, config_options, pi);

    return std::make_shared<OpenVINOProviderFactory>(pi, OVCore::Get());
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* ep_metadata,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    // Check if no devices are provided
    if (num_devices == 0) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "No devices provided to CreateIExecutionProvider");
    }

    // For provider options that we don't support directly but are still supported through load_config,
    // give some specific guidance & example about how to make use of the option through load_config.
    const std::vector<std::pair<std::string, std::string>> block_and_advise_entries = {
        {"cache_dir", "\"CACHE_DIR\": \"<filesystem_path>\""},
        {"precision", "\"INFERENCE_PRECISION_HINT\": \"F32\""},
        {"num_of_threads", "\"INFERENCE_NUM_THREADS\": \"1\""},
        {"num_streams", "\"NUM_STREAMS\": \"1\""},
        {"model_priority", "\"MODEL_PRIORITY\": \"LOW\""},
        {"enable_opencl_throttling", "\"GPU\": {\"PLUGIN_THROTTLE\": \"1\"}"},
        {"enable_qdq_optimizer", "\"NPU\": {\"NPU_QDQ_OPTIMIZATION\": \"YES\"}"}};

    for (auto& block_and_advise_entry : block_and_advise_entries) {
      if (provider_options.find(block_and_advise_entry.first) != provider_options.end()) {
        std::string message = "OpenVINO EP: Option '" + block_and_advise_entry.first +
                              "' cannot be set when using AppendExecutionProvider_V2. " +
                              "It can instead be enabled by a load_config key / value pair. For example: " +
                              block_and_advise_entry.second;
        return Status(common::ONNXRUNTIME, ORT_INVALID_ARGUMENT, message);
      }
    }

    // For the rest of the disallowed provider options, give a generic error message.
    const std::vector<std::string> blocked_provider_keys = {
        "device_type", "device_id", "device_luid", "context", "disable_dynamic_shapes"};

    for (const auto& key : blocked_provider_keys) {
      if (provider_options.find(key) != provider_options.end()) {
        return Status(common::ONNXRUNTIME, ORT_INVALID_ARGUMENT,
                      "OpenVINO EP: Option '" + key + "' cannot be set when using AppendExecutionProvider_V2.");
      }
    }

    const char* ov_device_key = "ov_device";
    const char* ov_meta_device_key = "ov_meta_device";

    // Create a unique list of ov_devices that were passed in.
    std::unordered_set<std::string_view> unique_ov_devices;
    std::vector<std::string_view> ordered_unique_ov_devices;
    for (size_t i = 0; i < num_devices; ++i) {
      const auto& device_meta_data = ep_metadata[i];
      auto ov_device_it = device_meta_data->Entries().find(ov_device_key);
      if (ov_device_it == device_meta_data->Entries().end()) {
        return Status(common::ONNXRUNTIME, ORT_INVALID_ARGUMENT, "OpenVINO EP device metadata not found.");
      }
      auto& ov_device = ov_device_it->second;

      // Add to ordered_unique only if not already present
      if (unique_ov_devices.insert(ov_device).second) {
        ordered_unique_ov_devices.push_back(ov_device);
      }
    }

    std::string ov_meta_device_type = "NONE";
    {
      auto ov_meta_device_it = ep_metadata[0]->Entries().find(ov_meta_device_key);
      if (ov_meta_device_it != ep_metadata[0]->Entries().end()) {
        ov_meta_device_type = ov_meta_device_it->second;
      }
    }

    bool is_meta_device_factory = (ov_meta_device_type != "NONE");

    if (ordered_unique_ov_devices.size() > 1 && !is_meta_device_factory) {
      LOGS_DEFAULT(WARNING) << "[OpenVINO EP] Multiple devices were specified that are not OpenVINO meta devices. Using first ov_device only: " << ordered_unique_ov_devices.at(0);
      ordered_unique_ov_devices.resize(1);  // Use only the first device if not a meta device factory
    }

    std::string ov_device_string;
    if (is_meta_device_factory) {
      // Build up a meta device string based on the devices that are passed in. E.g. AUTO:NPU,GPU.0,CPU
      ov_device_string = std::move(ov_meta_device_type);
      ov_device_string += ":";
    }

    bool prepend_comma = false;
    for (const auto& ov_device : ordered_unique_ov_devices) {
      if (prepend_comma) {
        ov_device_string += ",";
      }
      ov_device_string += ov_device;
      prepend_comma = true;
    }

    provider_options["device_type"] = std::move(ov_device_string);

    // Parse provider info with the device type
    ProviderInfo pi;
    const auto& config_options = session_options.GetConfigOptions();
    try {
      ParseProviderInfo(provider_options, &config_options, pi);
      ParseConfigOptions(pi);
    } catch (std::exception& e) {
      return Status(common::ONNXRUNTIME, ORT_INVALID_ARGUMENT, e.what());
    }

    // Create and return the execution provider
    auto factory = std::make_unique<OpenVINOProviderFactory>(pi, OVCore::Get());
    ep = factory->CreateProvider_V2(session_options, logger);
    return Status::OK();
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

 private:
  ProviderInfo_OpenVINO_Impl info_;
};  // OpenVINO_Provider

}  // namespace openvino_ep
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  static onnxruntime::openvino_ep::OpenVINO_Provider g_provider;
  return &g_provider;
}
}
