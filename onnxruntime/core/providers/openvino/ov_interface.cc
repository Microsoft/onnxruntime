// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#include <format>

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
#include "core/providers/openvino/ov_stateful_patch_utils.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/exceptions.h"

namespace onnxruntime {
namespace openvino_ep {

template <bool typed, typename Func, typename... Args>
inline auto OvExceptionBoundary(Func&& func, std::format_string<Args...>&& fmt, Args&&... args) {
  try {
    return func();
  } catch (const ov::Exception& e) {
    if constexpr (typed) {
      throw ovep_exception(e, ovep_exception::type::import_model);
    } else {
      ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)) + ": " + std::string(e.what()));
    }
  } catch (const std::exception& e) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)) + ": [std::exception] " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)));
  }
}

#ifndef NDEBUG
void printDebugInfo(const ov::CompiledModel& obj) {
  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    // output of the actual settings that the device selected
    auto supported_properties = obj.get_property(ov::supported_properties);
    std::cout << "Model:" << std::endl;
    for (const auto& cfg : supported_properties) {
      if (cfg == ov::supported_properties)
        continue;
      auto prop = obj.get_property(cfg);
      if (cfg == ov::device::properties) {
        auto devices_properties = prop.as<ov::AnyMap>();
        for (auto& item : devices_properties) {
          std::cout << "  " << item.first << ": " << std::endl;
          for (auto& item2 : item.second.as<ov::AnyMap>()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                item2.first == "SUPPORTED_METRICS")
              continue;
            OPENVINO_SUPPRESS_DEPRECATED_END
            std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
          }
        }
      } else {
        std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
      }
    }
  }
}
#endif

// Function to check if a given OV property is enabled
std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type) {
  try {
    // Get the property value
    auto supported_properties = OVCore::Get()->core.get_property(device_type, ov::supported_properties);
    return std::find(supported_properties.begin(), supported_properties.end(), property) != supported_properties.end();
  } catch (const std::exception&) {
    return std::nullopt;  // Property not found or invalid
  }
}

std::shared_ptr<OVNetwork> OVCore::ReadModel(std::string&& model, const std::string& model_path) {
  return OvExceptionBoundary<false>([&]() {
    std::istringstream modelStringStream(std::move(model));
    std::istream& modelStream = modelStringStream;
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream, model_path};

    FE = manager.load_by_model(params);
    if (FE) {
      inputModel = FE->load(params);
      return FE->convert(inputModel);
    } else {
      ORT_THROW(log_tag + "Unknown exception while Reading network");
    }
  },
                                    "Exception while Reading network");
}

OVExeNetwork OVCore::StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                          std::string& hw_target,
                                          const ov::AnyMap& device_config) {
  ov::CompiledModel compiled_model;
  ov::AnyMap config = device_config;

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateless OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  bool model_status = IsStateful(model);
  LOGS_DEFAULT(INFO) << log_tag << "Model IsStateful() Status:\t" << (model_status ? "True" : "False");
  // Flag to add Gather+ScatterElementsUpdate subgraph to reorder KV cache for LLM speculative decoding
  bool should_add_kvcache_reorder = false;
  if (!model_status) {
    LOGS_DEFAULT(INFO) << log_tag << "Converting from Stateless OV Model to Stateful OV Model" << std::endl;
    // TO-DO: extend to NPU device when OpenVINO NPU has related optimization
    should_add_kvcache_reorder = hw_target.find("GPU") != std::string::npos;
    PatchStatefulDecoder(model, should_add_kvcache_reorder);
  }

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateful OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  auto kv_pos = GetKVAxesPos(model);

  if (hw_target.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    auto parse_genai_config = [&](const std::string& key, unsigned int default_value) {
      return (config.count(key) && !config.at(key).empty() && config.at(key).as<std::string>() != "0") ? config.at(key).as<unsigned int>() : default_value;
    };

    kv_desc.max_prompt_len = parse_genai_config("MAX_PROMPT_LEN", CausalLMConfig().max_prompt_len);
    kv_desc.min_response_len = parse_genai_config("MIN_RESPONSE_LEN", CausalLMConfig().min_response_len);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    if (kv_desc.max_prompt_len == 0 || kv_desc.min_response_len == 0) {
      ORT_THROW(log_tag + "MAX_PROMPT_LEN and MIN_RESPONSE_LEN cannot be 0 or empty");
    }

    if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
      std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;
      std::cout << "kv_desc.max_prompt_len:\t" << kv_desc.max_prompt_len << std::endl;
      std::cout << "kv_desc.min_response_len:\t" << kv_desc.min_response_len << std::endl;
    }

    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // Actually either way that happens within NPUW::LLMCompiledModel creation for NPU device,
    // while this is here mostly to align this behavior for other devices viz. (CPU, GPU).
    ApplySliceBeforeMatmulTransformation(model);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Compiling OV Model using Stateful Transformation flow";
  compiled_model = OVCore::Get()->core.compile_model(model, hw_target, config);
  OVExeNetwork exe(compiled_model, hw_target, true, should_add_kvcache_reorder);
  return exe;
}

OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  bool enable_causallm,
                                  const std::string& name) {
  return OvExceptionBoundary<false>([&]() {
    OVExeNetwork exe;
    if (enable_causallm) {
      auto mutable_model = ie_cnn_network->clone();
      exe = OVCore::Get()->StatefulCompileModel(mutable_model, hw_target, device_config);
    } else {
      auto obj = core.compile_model(ie_cnn_network, hw_target, device_config);
      exe = OVExeNetwork(obj, hw_target);
    }

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif

    return exe;
  },
                                    "Exception while Loading Network for graph {}", name);
}

OVExeNetwork OVCore::CompileModel(const std::string& onnx_model,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  return OvExceptionBoundary<false>([&]() {
    ov::CompiledModel obj;

    obj = core.compile_model(onnx_model, ov::Tensor(), hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  },
                                    "Exception while Loading Network for graph {}", name);
}

OVExeNetwork OVCore::ImportModel(ModelBlobWrapper& model_blob,
                                 std::string hw_target,
                                 const ov::AnyMap& device_config,
                                 std::string name) {
  return OvExceptionBoundary<true>([&]() {
    ov::CompiledModel obj;
    // Import from tensor enables file mapping and should be used as primary option. Import from stream is only used for OV versions prior to OV25.3.
    // TODO: Remove import from stream when OV26.0 support is added and OV 2025.2 support is removed.
#if (OPENVINO_VERSION_MAJOR > 2025 || (OPENVINO_VERSION_MAJOR == 2025 && OPENVINO_VERSION_MINOR >= 3))
    if (model_blob.tensor_) {
      obj = core.import_model(model_blob.tensor_, hw_target, device_config);
    } else {
      obj = core.import_model(*model_blob.stream_, hw_target, device_config);
    }
#else
      obj = core.import_model(*model_blob.stream_, hw_target, device_config);
#endif
    OVExeNetwork exe(obj, hw_target);

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif
    return exe;
  },
                                   "Exception while Loading Network for graph {}", name);
}

OVExeNetwork OVCore::ImportEPCtxOVIREncapsulation(std::istream& model_stream,
                                                  std::string& hw_target,
                                                  const ov::AnyMap& device_config,
                                                  bool enable_causallm,
                                                  std::filesystem::path xml_file_path,
                                                  const SessionContext& session_context) {
  return OvExceptionBoundary<false>([&]() {
    OVExeNetwork exe;

    bool isXML = backend_utils::IsModelStreamXML(model_stream);

    // Helper function to check if file exists and is readable
    const auto check_file_access = [](const std::filesystem::path& path) {
      try {
        if (!std::filesystem::exists(path) || std::filesystem::is_empty(path)) {
          ORT_THROW(log_tag + "Required file missing or empty: " + path.string());
        }
        std::ifstream file(path);
        if (!file) {
          ORT_THROW(log_tag + "Required file not readable: " + path.string());
        }
      } catch (const std::exception& e) {
        ORT_THROW(log_tag + "Exception while checking file access for: " + path.string() + " - " + e.what());
      }
    };

    if (isXML) {
      // If the model is XML, we need to load it with the XML content in read_model()
      // where weights from bin file is directly consumed
      check_file_access(xml_file_path);

      LOGS_DEFAULT(INFO) << log_tag << "Reading OVIR from XML file path: " << xml_file_path.string();

      // Load the model explicitly with XML contents
      std::shared_ptr<ov::Model> model = core.read_model(xml_file_path.string());

      if (!session_context.reshape.empty()) {
        LOGS_DEFAULT(INFO) << log_tag << "Reshaping OV-IR model to specified shape";
        model->reshape(session_context.reshape);
      }

      if (enable_causallm) {
        exe = OVCore::Get()->StatefulCompileModel(model, hw_target, device_config);
      } else {
        auto obj = core.compile_model(model, hw_target, device_config);
        exe = OVExeNetwork(obj, hw_target);
      }
    }

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif
    return exe;
  },
                                    "Exception while Loading Network from OVIR model file: {}", xml_file_path.string());
}

void OVCore::SetCache(const std::string& cache_dir_path) {
  core.set_property(ov::cache_dir(cache_dir_path));
}

std::vector<std::string> OVCore::GetAvailableDevices() const {
  std::vector<std::string> available_devices = core.get_available_devices();
  return available_devices;
}

std::vector<std::string> OVCore::GetAvailableDevices(const std::string& device_type) const {
  std::vector<std::string> available_devices;
  std::vector<std::string> devicesIDs;
  // Uses logic from OpenVINO to only return available devices of the specified type (e.g. CPU, NPU or GPU)
  try {
    devicesIDs = core.get_property(device_type, ov::available_devices);
  } catch (const ov::Exception&) {
    // plugin is not created by e.g. invalid env
    // Empty device list will be returned
  } catch (const std::exception& ex) {
    ORT_THROW(log_tag + "An exception occurred while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (...) {
    ORT_THROW(log_tag + "Unknown exception occurred while trying to create the ",
              device_type,
              " device");
  }

  if (devicesIDs.size() > 1 ||
      (devicesIDs.size() == 1 && devicesIDs[0] == "0")) {
    for (const auto& deviceID : devicesIDs) {
      available_devices.push_back(device_type + '.' + deviceID);
    }
  }
  if (!devicesIDs.empty()) {
    available_devices.push_back(device_type);
  }

  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  core.set_property(device_type, {ov::num_streams(num_streams)});
}

std::shared_ptr<OVInferRequest> OVExeNetwork::CreateInferRequest() {
  return OvExceptionBoundary<false>([&]() {
    auto infReq = compiled_model_obj.create_infer_request();
    std::shared_ptr<OVInferRequest> ovInfReq;
    if (is_stateful_causallm) {
      ovInfReq = std::make_shared<StatefulOVInferRequest>(std::move(infReq), target_device, is_kvcache_reorder_added);
    } else {
      ovInfReq = std::make_shared<OVInferRequest>(std::move(infReq));
    }
    return ovInfReq;
  },

                                    "Exception while creating InferRequest object");
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  return OvExceptionBoundary<false>([&]() {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  },
                                    " Cannot access IE Blob for input: {}", input_name);
}

std::string OVInferRequest::GetInputTensorName(uint32_t index) {
  return OvExceptionBoundary<false>([&]() {
    const auto& model = ovInfReq.get_compiled_model();
    return *model.input(index).get_names().begin();
  },
                                    " Cannot access IE Blob for input number: {}", index);
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  OvExceptionBoundary<false>([&]() {
    ovInfReq.set_tensor(name, *(blob.get()));
  },
                             " Cannot set Remote Blob for output: {}", name);
}

uint32_t OVInferRequest::GetNumInputs() {
  return static_cast<uint32_t>(ovInfReq.get_compiled_model().inputs().size());
}

void OVInferRequest::Infer() {
  OvExceptionBoundary<false>([&]() {
    ovInfReq.infer();
  },
                             "In Error Couldn't start Inference");
}

StatefulOVInferRequest::StatefulOVInferRequest(ov::InferRequest infer_request, std::string device, bool kvcache_reorder_added)
    : OVInferRequest(std::move(infer_request)), target_device(device), is_kvcache_reorder_added(kvcache_reorder_added) {
  bool gpu_or_npu = ((device.find("NPU") != std::string::npos) || (device.find("GPU") != std::string::npos));

  _npu_logits_slice_required = IsNPULogitsSliceRequired();

  // check if there is input_ids tensors and if the tensor type is int64,
  // because logic prefill_use_full_chat_history is only for specific inputs and data type
  auto input_ids_opt = FindTensor("input_ids");
  if (gpu_or_npu && input_ids_opt.has_value() && input_ids_opt->get_element_type() == ov::element::i64) {
    prefill_use_full_chat_history = true;
  }
}

static inline bool IsNPUWSliceOutEnabled(const ov::CompiledModel& compiled_model) {
  auto slice_out_val = compiled_model.get_property("NPUW_SLICE_OUT");
  if (!slice_out_val.empty()) {
    if (slice_out_val.is<std::string>()) {
      return (slice_out_val.as<std::string>() == "YES");
    } else if (slice_out_val.is<bool>()) {
      return slice_out_val.as<bool>();
    }
  }

  return false;
}

bool StatefulOVInferRequest::IsNPULogitsSliceRequired() {
  if (target_device.find("NPU") != std::string::npos) {
    const auto& model = ovInfReq.get_compiled_model();
    // If NPUW_SLICE_OUT is enabled, it means that it's not required to slice within OVEP.
    // Otherwise, if NPUW_SLICE_OUT is NOT enabled, then we need to perform some explicit logit
    // slicing in OVEP.
    return !IsNPUWSliceOutEnabled(model);
  }

  return false;
}

void StatefulOVInferRequest::FillTensor(const std::string& tensor_name, const ov::element::Type& type,
                                        const std::vector<size_t>& shape, int32_t fill_value) {
  ov::Tensor tensor = ov::Tensor(type, shape);
  std::fill_n(tensor.data<int32_t>(), tensor.get_size(), fill_value);
  ovInfReq.set_tensor(tensor_name, tensor);
}

void StatefulOVInferRequest::CacheTensor(const std::string& tensor_name, std::vector<int64_t>& cache) {
  auto tensor = ovInfReq.get_tensor(tensor_name);
  auto* pData = tensor.data<int64_t>();
  for (size_t i = 0; i < tensor.get_size(); i++) {
    cache.emplace_back(pData[i]);
  }
}

void StatefulOVInferRequest::SetTensorFromCache(const std::string& tensor_name,
                                                const std::vector<int64_t>& cache_data) {
  auto tensor = ovInfReq.get_tensor(tensor_name);
  auto new_shape = tensor.get_shape();
  new_shape[1] = cache_data.size();

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  auto* pNewData = new_tensor.data<int64_t>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(int64_t));

  ovInfReq.set_tensor(tensor_name, new_tensor);
}

std::optional<ov::Tensor> StatefulOVInferRequest::FindTensor(const std::string& tensor_name) {
  // Check if tensor exists by examining input names in the compiled model
  const auto& model = ovInfReq.get_compiled_model();
  bool tensor_exists = false;

  for (const auto& input : model.inputs()) {
    const auto& names = input.get_names();
    if (names.find(tensor_name) != names.end()) {
      tensor_exists = true;
      break;
    }
  }

  if (tensor_exists) {
    return ovInfReq.get_tensor(tensor_name);
  }

  return std::nullopt;
}

void StatefulOVInferRequest::PreProcessInferRequest() {
  if (is_kvcache_reorder_added) {
    ov::Shape dst_idx_shape = ovInfReq.get_tensor("dst_idx").get_shape();
    if (dst_idx_shape.size() < 4) {
      ORT_THROW(log_tag + "dst_idx tensor shape must have at least 4 dimensions, but got " +
                std::to_string(dst_idx_shape.size()) + " dimensions");
    }
    const auto kv_num_heads = dst_idx_shape[1];
    const auto kv_head_size = dst_idx_shape[3];
    if (kv_src_indices.size() > 0) {
      ov::Tensor src_idx_tensor = ov::Tensor(ov::element::i32, {kv_src_indices.size()});
      const auto src_idx_ptr = src_idx_tensor.data<int32_t>();
      for (size_t i = 0; i < kv_src_indices.size(); ++i) {
        src_idx_ptr[i] = static_cast<int32_t>(kv_src_indices[i]);
      }
      ovInfReq.set_tensor("src_idx", src_idx_tensor);

      ov::Tensor dst_idx_tensor = ov::Tensor(ov::element::i32, {1, kv_num_heads, kv_dst_indices.size(), kv_head_size});
      const auto dst_idx_ptr = dst_idx_tensor.data<int32_t>();
      for (size_t i = 0; i < kv_num_heads; ++i) {
        for (size_t j = 0; j < kv_dst_indices.size(); ++j) {
          std::fill_n(dst_idx_ptr + (i * kv_dst_indices.size() + j) * kv_head_size, kv_head_size, kv_dst_indices[j]);
        }
      }
      ovInfReq.set_tensor("dst_idx", dst_idx_tensor);
    } else {
      FillTensor("src_idx", ov::element::i32, {0}, 0);
      FillTensor("dst_idx", ov::element::i32, {1, kv_num_heads, 0, kv_head_size}, 0);
    }
  }

  if (FindTensor("beam_idx")) {
    // Workaround: Setting the value here as it cannot be set at the ORT GenAI layer currently.
    // TODO(ankit): Address this issue and implement the fix at the appropriate layer.
    FillTensor("beam_idx", ov::element::i32, {1}, 0);
  }

  if (is_kvcache_reorder_added) {
    ov::Shape dst_idx_shape = ovInfReq.get_tensor("dst_idx").get_shape();
    const auto kv_num_heads = dst_idx_shape[1];
    const auto kv_head_size = dst_idx_shape[3];
    if (kv_src_indices.size() > 0) {
      ov::Tensor src_idx_tensor = ov::Tensor(ov::element::i32, {kv_src_indices.size()});
      const auto src_idx_ptr = src_idx_tensor.data<int32_t>();
      for (size_t i = 0; i < kv_src_indices.size(); ++i) {
        src_idx_ptr[i] = static_cast<int32_t>(kv_src_indices[i]);
      }
      ovInfReq.set_tensor("src_idx", src_idx_tensor);

      ov::Tensor dst_idx_tensor = ov::Tensor(ov::element::i32, {1, kv_num_heads, kv_dst_indices.size(), kv_head_size});
      const auto dst_idx_ptr = dst_idx_tensor.data<int32_t>();
      for (size_t i = 0; i < kv_num_heads; ++i) {
        for (size_t j = 0; j < kv_dst_indices.size(); ++j) {
          std::fill_n(dst_idx_ptr + (i * kv_dst_indices.size() + j) * kv_head_size, kv_head_size, kv_dst_indices[j]);
        }
      }
      ovInfReq.set_tensor("dst_idx", dst_idx_tensor);
    } else {
      FillTensor("src_idx", ov::element::i32, {0}, 0);
      FillTensor("dst_idx", ov::element::i32, {1, kv_num_heads, 0, kv_head_size}, 0);
    }
  }

  // If 'prefill use full chat history' mode is enabled, we need to cache input_ids and position_ids.
  if (prefill_use_full_chat_history) {
    auto input_ids_tensor = ovInfReq.get_tensor("input_ids");
    CacheTensor("input_ids", cached_input_ids);

    // "position_ids" (GQA with Rotary Embeddings doesnt have position_ids) - check if exists
    auto position_ids_opt = FindTensor("position_ids");
    bool has_position_ids = position_ids_opt.has_value();

    if (has_position_ids) {
      CacheTensor("position_ids", cached_position_ids);
    }

    // If we're about to run the prefill model
    if (input_ids_tensor.get_size() > 1) {
      // Check if the size of the current "input_ids" tensor does not match the size of the cached "input_ids".
      // This indicates that we are running a subsequent prompt (not the initial prefill).
      if (input_ids_tensor.get_shape()[1] != cached_input_ids.size()) {
        // Clear the internal KVCache state. For NPU device, this operation is a no-op.
        ovInfReq.reset_state();

        // Set tensors using cached values
        SetTensorFromCache("input_ids", cached_input_ids);

        // Only set position_ids if it exists and we have cached values
        if (has_position_ids && !cached_position_ids.empty()) {
          SetTensorFromCache("position_ids", cached_position_ids);
        }
      }
    }
  }
}

void StatefulOVInferRequest::Infer() {
  PreProcessInferRequest();
  OVInferRequest::Infer();
  PostProcessInferRequest();
}

void StatefulOVInferRequest::PostProcessInferRequest() {
  CleanReorderKVCacheStatus();
}

void StatefulOVInferRequest::CleanReorderKVCacheStatus() {
  if (is_kvcache_reorder_added) {
    kv_src_indices.clear();
    kv_dst_indices.clear();
  }
}

void StatefulOVInferRequest::SetReorderKVCacheStatus(const std::vector<int32_t>& src_indices, const std::vector<int32_t>& dst_indices) {
  // Validate input parameters
  if (src_indices.size() != dst_indices.size()) {
    ORT_THROW(log_tag +
              "SetReorderKVCacheStatus: src_indices and dst_indices must have the same size. "
              "Got src_indices.size()=" +
              std::to_string(src_indices.size()) +
              ", dst_indices.size()=" + std::to_string(dst_indices.size()));
  }

  LOGS_DEFAULT(INFO) << log_tag << "SetReorderKVCacheStatus: Reordering OpenVINO-internal KVCache state with "
                     << src_indices.size() << " index pairs";

  kv_src_indices = src_indices;
  kv_dst_indices = dst_indices;
}

void StatefulOVInferRequest::RewindKVCache(size_t index) {
  LOGS_DEFAULT(INFO) << log_tag << "RewindKVCache: Rewinding OpenVINO-internal KVCache state to index=" << index;

  if (is_kvcache_reorder_added) {
    if (index > 0) {
      // TODO: Trigger a KVCache eviction inference pass here to physically compact the KV cache
      // by running infer_request with the src_idx/dst_idx reorder tensors set.
      ORT_THROW(log_tag + "KVCache eviction via reorder inference is not yet implemented which is required to rewind KVCache with index > 0 when KVCache reorder optimization is enabled. Requested index: " + std::to_string(index));
    }
    CleanReorderKVCacheStatus();
  }

  if (prefill_use_full_chat_history) {
    // Clear the internal KVCache state. For NPU device, this operation is a no-op.
    ovInfReq.reset_state();

    // Resize the cached "input_ids" and "position_ids" to the specified index.
    if (cached_input_ids.size() > index) {
      cached_input_ids.resize(index);
    }

    if (cached_position_ids.size() > index) {
      cached_position_ids.resize(index);
    }
  } else {
    if (index == 0) {
      // In this case, since we're resetting the entire KVCache, simply reset the state.
      ovInfReq.reset_state();
    } else {
      // Retrieve KVCache states and trim them to the specified index.
      // The following logic is adapted from:
      // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
      auto states = ovInfReq.query_state();
      for (auto& state : states) {
        ov::Tensor old_tensor = state.get_state();
        // Tensor shape: [batch_size, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();

        if (shape[2] > index) {
          // Update the sequence length dimension to the specified index.
          shape[2] = index;

          ov::Coordinate new_shape_begin{0, 0, 0, 0};
          ov::Coordinate new_shape_end{shape};

          // Create a trimmed tensor with the updated shape.
          auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

          // Copy the trimmed tensor into a new tensor and update the state.
          ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
          trimmed_tensor.copy_to(new_tensor);

          state.set_state(new_tensor);
        }
      }
    }
  }
}

OVTensorPtr StatefulOVInferRequest::GetTensor(const std::string& input_name) {
  auto tobj = OVInferRequest::GetTensor(input_name);

  if (_npu_logits_slice_required) {
    if (input_name == "logits") {
      if (tobj->get_shape().size() != 3) {
        ORT_THROW(log_tag + std::format("Expected logits to have shape of rank 3, but it has shape of rank {}",
                                        tobj->get_shape().size()));
      }

      // When _npu_logits_slice_required is true, it means that prefill may produce logits of shape:
      // [<batch_size>, sequence_length, <vocab_size>]
      // (Where 'sequence_length' is number of input tokens to prefill)
      // But, ORT GenAI is expecting to receive logits of shape:
      // [<batch_size>, 1, <vocab_size>]
      // In this case, detect when shape[1] is not 1. When it is, create a slice of shape [<batch_size>, 1, <vocab_size>]
      if (tobj->get_shape()[1] > 1) {
        return OvExceptionBoundary<false>([&]() {
          const ov::Coordinate begin = {0, tobj->get_shape()[1] - 1, 0};
          const ov::Coordinate end = {tobj->get_shape()[0], tobj->get_shape()[1], tobj->get_shape()[2]};
          auto sliced_tensor = ov::Tensor(*tobj, begin, end);
          if (sliced_tensor.is_continuous()) {
            OVTensorPtr blob = std::make_shared<OVTensor>(sliced_tensor);
            return blob;
          } else {
            auto continuous_sliced_tensor = ov::Tensor(sliced_tensor.get_element_type(), sliced_tensor.get_shape());
            sliced_tensor.copy_to(continuous_sliced_tensor);
            OVTensorPtr blob = std::make_shared<OVTensor>(continuous_sliced_tensor);
            return blob;
          }
        },
                                          "Could not create sliced logits tensor");
      }
    }
  }

  return tobj;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
