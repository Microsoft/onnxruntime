// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/perftest/utils.h"
#include "test/perftest/strings_helper.h"
#include <core/platform/path_lib.h>

#include <cstdint>

#include <filesystem>

namespace onnxruntime {
namespace perftest {
namespace utils {

void ListEpDevices(const Ort::Env& env) {
  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

  for (size_t i = 0; i < ep_devices.size(); ++i) {
    auto device = ep_devices[i];
    std::string device_info_msg = "===== device id " + std::to_string(i) + " ======\n";
    device_info_msg += "name: " + std::string(device.EpName()) + "\n";
    device_info_msg += "vendor: " + std::string(device.EpVendor()) + "\n";

    auto metadata = device.EpMetadata();
    std::unordered_map<std::string, std::string> metadata_entries = metadata.GetKeyValuePairs();
    if (!metadata_entries.empty()) {
      device_info_msg += "metadata:\n";
    }

    for (auto& entry : metadata_entries) {
      device_info_msg += "  " + entry.first + ": " + entry.second + "\n";
    }
    device_info_msg += "\n";
    fprintf(stdout, "%s", device_info_msg.c_str());
  }
}

void RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  if (!test_config.plugin_ep_names_and_libs.empty()) {
    std::unordered_map<std::string, std::string> ep_names_to_libs;
    ParseSessionConfigs(ToUTF8String(test_config.plugin_ep_names_and_libs), ep_names_to_libs);
    if (ep_names_to_libs.size() > 0) {
      for (auto& pair : ep_names_to_libs) {
        const std::filesystem::path library_path = pair.second;
        const std::string registration_name = pair.first;
        Ort::Status status(Ort::GetApi().RegisterExecutionProviderLibrary(env, registration_name.c_str(), ToPathString(library_path.string()).c_str()));
        if (status.IsOK()) {
          test_config.registered_plugin_eps.push_back(registration_name);
        } else {
          fprintf(stderr, "Can't register %s plugin library: %s\n", registration_name.c_str(), status.GetErrorMessage().c_str());
        }
      }
    }
  }
}

void UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  for (auto& registration_name : test_config.registered_plugin_eps) {
    Ort::Status status(Ort::GetApi().UnregisterExecutionProviderLibrary(env, registration_name.c_str()));
    if (!status.IsOK()) {
      fprintf(stderr, "%s", status.GetErrorMessage().c_str());
    }
  }
}

std::vector<std::string> ConvertArgvToUtf8Strings(int argc, ORTCHAR_T* argv[]) {
  std::vector<std::string> utf8_args;
  utf8_args.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string utf8_string = ToUTF8String(argv[i]);

    // Abseil flags doens't natively alias "-h" to "--help".
    // We make "-h" alias to "--help" here.
    if (utf8_string == "-h" || utf8_string == "--h") {
      utf8_args.push_back("--help");
    } else {
      utf8_args.push_back(utf8_string);
    }
  }
  return utf8_args;
}

std::vector<char*> CStringsFromStrings(std::vector<std::string>& utf8_args) {
  std::vector<char*> utf8_argv;
  utf8_argv.reserve(utf8_args.size());
  for (auto& str : utf8_args) {
    utf8_argv.push_back(&str[0]);
  }
  return utf8_argv;
}

std::vector<Ort::ConstEpDevice> AppendPluginExecutionProviders(Ort::Env& env,
                                                               Ort::SessionOptions& session_options,
                                                               const PerformanceTestConfig& test_config) {
  if (test_config.registered_plugin_eps.empty()) {
    return {};
  }

  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
  // EP -> associated EP devices (All OrtEpDevice instances must be from the same execution provider)
  std::unordered_map<std::string, std::vector<Ort::ConstEpDevice>> added_ep_devices;
  std::unordered_set<int> added_ep_device_index_set;

  auto& ep_list = test_config.machine_config.plugin_provider_type_list;
  std::unordered_set<std::string> ep_set(ep_list.begin(), ep_list.end());

  // Select EP devices by provided device index
  if (!test_config.selected_ep_device_indices.empty()) {
    std::vector<int> device_list;
    device_list.reserve(test_config.selected_ep_device_indices.size());
    ParseEpDeviceIndexList(test_config.selected_ep_device_indices, device_list);
    for (auto index : device_list) {
      if (static_cast<size_t>(index) > (ep_devices.size() - 1)) {
        fprintf(stderr, "%s", "The device index provided is not correct. Will skip this device id.");
        continue;
      }

      Ort::ConstEpDevice& device = ep_devices[index];
      if (ep_set.find(std::string(device.EpName())) != ep_set.end()) {
        if (added_ep_device_index_set.find(index) == added_ep_device_index_set.end()) {
          added_ep_devices[device.EpName()].push_back(device);
          added_ep_device_index_set.insert(index);
          fprintf(stdout, "[Plugin EP] EP Device [Index: %d, Name: %s, Type: %d] has been added to session.\n", static_cast<int>(index), device.EpName(), device.Device().Type());
        }
      } else {
        std::string err_msg = "[Plugin EP] [WARNING] : The EP device index and its corresponding OrtEpDevice is not created from " +
                              test_config.machine_config.provider_type_name + ". Will skip adding this device.\n";
        fprintf(stderr, "%s", err_msg.c_str());
      }
    }
  } else if (!test_config.filter_ep_device_kv_pairs.empty()) {
    // Find and select the OrtEpDevice associated with the EP in "--filter_ep_devices".
    for (const auto& kv : test_config.filter_ep_device_kv_pairs) {
      for (size_t index = 0; index < ep_devices.size(); ++index) {
        auto device = ep_devices[index];
        if (ep_set.find(std::string(device.EpName())) == ep_set.end())
          continue;

        // Skip if deviceid was already added
        if (added_ep_devices.find(device.EpName()) != added_ep_devices.end() &&
            std::find(added_ep_devices[device.EpName()].begin(), added_ep_devices[device.EpName()].end(), device) != added_ep_devices[device.EpName()].end())
          continue;

        // Check both EP metadata and device metadata for a match
        auto ep_metadata_kv_pairs = device.EpMetadata().GetKeyValuePairs();
        auto device_metadata_kv_pairs = device.Device().Metadata().GetKeyValuePairs();
        auto ep_metadata_itr = ep_metadata_kv_pairs.find(kv.first);
        auto device_metadata_itr = device_metadata_kv_pairs.find(kv.first);

        if ((ep_metadata_itr != ep_metadata_kv_pairs.end() && kv.second == ep_metadata_itr->second) ||
            (device_metadata_itr != device_metadata_kv_pairs.end() && kv.second == device_metadata_itr->second)) {
          added_ep_devices[device.EpName()].push_back(device);
          fprintf(stdout, "[Plugin EP] EP Device [Index: %d, Name: %s, Type: %d] has been added to session.\n", static_cast<int>(index), device.EpName(), device.Device().Type());
          break;
        }
      }
    }
  } else {
    // Find and select the OrtEpDevice associated with the EP in "--plugin_eps".
    for (size_t index = 0; index < ep_devices.size(); ++index) {
      Ort::ConstEpDevice& device = ep_devices[index];
      if (ep_set.find(std::string(device.EpName())) != ep_set.end()) {
        added_ep_devices[device.EpName()].push_back(device);
        fprintf(stdout, "[Plugin EP] EP Device [Index: %d, Name: %s] has been added to session.\n", static_cast<int>(index), device.EpName());
      }
    }
  }

  if (added_ep_devices.empty()) {
    ORT_THROW("[ERROR] [Plugin EP]: No matching EP devices found.");
  }

  std::string ep_option_string = ToUTF8String(test_config.run_config.ep_runtime_config_string);

  // EP's associated provider option lists
  std::vector<std::unordered_map<std::string, std::string>> ep_options_list;
  ParseEpOptions(ep_option_string, ep_options_list);

  // If user only provide the EPs' provider option lists for the first several EPs,
  // add empty provider option lists for the rest EPs.
  if (ep_options_list.size() < ep_list.size()) {
    for (size_t i = ep_options_list.size(); i < ep_list.size(); ++i) {
      ep_options_list.emplace_back();  // Adds a new empty map
    }
  } else if (ep_options_list.size() > ep_list.size()) {
    ORT_THROW("[ERROR] [Plugin EP]: Too many EP provider option lists provided.");
  }

  // EP -> associated provider options
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> ep_options_map;
  for (size_t i = 0; i < ep_list.size(); ++i) {
    ep_options_map.emplace(ep_list[i], ep_options_list[i]);
  }

  std::vector<Ort::ConstEpDevice> selected_ep_devices;
  for (auto& ep_and_devices : added_ep_devices) {
    auto& ep = ep_and_devices.first;
    auto& devices = ep_and_devices.second;
    session_options.AppendExecutionProvider_V2(env, devices, ep_options_map[ep]);
    selected_ep_devices.insert(selected_ep_devices.end(), devices.begin(), devices.end());
  }

  return selected_ep_devices;
}

std::optional<PluginEpAllocatorSelection> GetPluginEpAllocator(Ort::Env& env,
                                                               const std::vector<Ort::ConstEpDevice>& ep_devices) {
  // More than one device appended means no single unambiguous allocator to pick.
  if (ep_devices.size() != 1) {
    fprintf(stdout, "[Plugin EP] %d EP devices appended. Using the CPU allocator.\n",
            static_cast<int>(ep_devices.size()));
    return std::nullopt;
  }

  const Ort::ConstEpDevice& ep_device = ep_devices[0];

  Ort::ConstMemoryInfo default_mem_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  if (default_mem_info) {
    Ort::UnownedAllocator allocator = env.GetSharedAllocator(default_mem_info);
    if (allocator) {
      fprintf(stdout, "[Plugin EP] Using the default allocator.\n");
      return PluginEpAllocatorSelection{allocator, /*is_host_accessible*/ false};
    }
  }

  Ort::ConstMemoryInfo host_accessible_mem_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (host_accessible_mem_info) {
    Ort::UnownedAllocator allocator = env.GetSharedAllocator(host_accessible_mem_info);
    if (allocator) {
      fprintf(stdout, "[Plugin EP] Using the host accessible allocator.\n");
      return PluginEpAllocatorSelection{allocator, /*is_host_accessible*/ true};
    }
  }

  return std::nullopt;
}

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
