// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <unordered_set>

#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <utility>
#include <iostream>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ov_stateful_patch_utils.h"

namespace onnxruntime {

namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
                           SessionContext& session_context,
                           const SubGraphContext& subgraph_context,
                           SharedContext& shared_context,
                           ptr_stream_t& model_stream)
    : session_context_{session_context}, subgraph_context_{subgraph_context}, shared_context_{shared_context} {
  std::string& hw_target = session_context_.device_type;
  bool enable_causallm = session_context_.enable_causallm;

  if (ValidateSubgraph(const_outputs_map_))
    return;

  ov::AnyMap device_config;
  SetOVDeviceConfiguration(device_config);
  if (subgraph_context_.is_ep_ctx_graph) {
    try {
      if (subgraph_context_.is_ep_ctx_ovir_encapsulated) {
        ORT_ENFORCE(!model_stream->model_path_.empty(),
                    log_tag + "OpenVINO IR path is unavailable for the EPContext model.");
        if (session_context_.onnx_model_path_name.empty()) {
          ORT_ENFORCE(model_stream->model_path_.is_absolute(),
                      log_tag +
                          "Context file path must be absolute when using CreateSessionFromArray(). "
                          "Please set a valid ep.context_file_path in session options.");
        }
        // If the EPContext node with OVIR Encapsulation, then create
        // an executable network from EP_CACHE_CONTEXT using read_model() & compile_model()
        exe_network_ = OVCore::Get()->ImportEPCtxOVIREncapsulation(*model_stream->stream_,
                                                                   hw_target,
                                                                   device_config,
                                                                   enable_causallm,
                                                                   model_stream->model_path_,
                                                                   session_context_);
      } else {
        // If the blob is held in an EPContext node, then skip FE+Compile
        // and directly move on to creating a backend with the executable blob
        exe_network_ = OVCore::Get()->ImportModel(*model_stream,
                                                  hw_target,
                                                  device_config,
                                                  subgraph_context_.subgraph_name);
      }
      model_stream.reset();
    } catch (const char* msg) {
      ORT_THROW(msg);
    }  // Delete stream after it is no longer needed
  } else {
    std::shared_ptr<const onnxruntime::openvino_ep::OVNetwork> ov_model;
    std::string model = model_proto->SerializeAsString();
    if (!subgraph_context.has_dynamic_input_shape) {
      model_proto.reset();
    }
    bool eligible_for_cpu_fallback = session_context_.device_type.find("NPU") != std::string::npos &&
                                     !session_context_.so_disable_cpu_ep_fallback &&
                                     !subgraph_context_.is_ep_ctx_graph;
#if defined(OPENVINO_DISABLE_NPU_FALLBACK)
    eligible_for_cpu_fallback = false;
#endif
    auto auto_unified_compile = (hw_target.find("AUTO") == std::string::npos);

    // Unified compile is efficient with cahce_dir cached model loading that bypass Read Model
    // Does not support model with exteral weights, dynamic input shape, Epctx onnx cached model,
    // reshape, enable_causallm, and for NPU CPU fallback

    auto is_unified_compile = (!session_context_.has_external_weights &&
                               !subgraph_context_.has_dynamic_input_shape &&
                               !session_context_.so_context_enable &&
                               session_context_.reshape.empty() &&
                               session_context_.layout.empty() &&
                               !enable_causallm &&
                               !eligible_for_cpu_fallback &&
                               auto_unified_compile);
    try {
      if (is_unified_compile) {
        exe_network_ = OVCore::Get()->CompileModel(model,
                                                   hw_target,
                                                   device_config,
                                                   subgraph_context_.subgraph_name);
      } else {  // For all other types use ov::ov_core read_model() to generate OV IR
                // followed by ov::ov_core compile_model()
        ov_model = CreateOVModel(std::move(model), session_context_, const_outputs_map_);
        exe_network_ = OVCore::Get()->CompileModel(
            ov_model, hw_target, device_config, enable_causallm, subgraph_context_.subgraph_name);
      }
      LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    } catch (const OnnxRuntimeException& ex) {
      std::string exception_str = ex.what();

      if (eligible_for_cpu_fallback) {
        LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                              << "Falling back to OV CPU for execution";
        session_context_.device_type = "CPU";
        session_context_.precision = "FP32";
        device_config.clear();
        SetOVDeviceConfiguration(device_config);
        try {
          exe_network_ = OVCore::Get()->CompileModel(
              ov_model, hw_target, device_config, enable_causallm, subgraph_context_.subgraph_name);
        } catch (std::string const& msg) {
          ORT_THROW(msg);
        }
      } else {
        ORT_THROW(ex.what());
      }
    }
  }
  int num_infer_req = (session_context_.num_of_threads > 0) ? session_context_.num_of_threads : 1;
  std::function<void(OVInferRequestPtr)> initializer = [](OVInferRequestPtr) {};
  if (session_context_.so_share_ep_contexts) {
    auto model_dir = session_context_.GetModelPath().parent_path();
    initializer = [this, model_dir = std::move(model_dir)](OVInferRequestPtr ir_ptr) {
      shared_context_.SetSharedWeightsOnInferRequest(ir_ptr->GetInfReq(), model_dir);
    };
  }

  infer_req_pool_ = std::make_unique<InferRequestPool>(exe_network_, num_infer_req, std::move(initializer));
  bindings_ = std::make_unique<OnnxToOvNetworkBindings>(exe_network_, subgraph_context_, session_context_);

  std::string perf_count = openvino_ep::backend_utils::GetPerfCountDumpPath();
  if (!perf_count.empty()) {
    PerfDirCreate(perf_count);
  } else {
    dir_.clear();
  }
}

bool BasicBackend::ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map) {
  if (const_outputs_map.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;
  if (subgraph_context_.is_constant) {
    LOGS_DEFAULT(INFO) << log_tag << "The subgraph is a const. Directly moving to Infer stage.";
    return true;
  }
  return false;
}

void BasicBackend::PopulateConfigValue(ov::AnyMap& device_config) {
  device_config = {};
  // Set inference precision based on device precision for OV backend
  if (session_context_.precision.find("FP16") != std::string::npos &&
      session_context_.device_type == "GPU") {
    device_config.emplace(ov::hint::inference_precision("f16"));
  }
  if (session_context_.precision.find("FP32") != std::string::npos) {
    device_config.emplace(ov::hint::inference_precision("f32"));
  }
  if (session_context_.precision.find("ACCURACY") != std::string::npos) {
    if (session_context_.OpenVINO_Version.at(0) >= 2024) {
      device_config.emplace(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    } else {
      if (!subgraph_context_.model_precision.empty())
        device_config.emplace(ov::hint::inference_precision(subgraph_context_.model_precision));
    }
  }

  // Set a priority level for the current workload for preemption;  default priority is "DEFAULT"
  // CPU Plugin doesn't support workload priority
  if (session_context_.device_type.find("CPU") == std::string::npos)
    device_config.emplace(ov::hint::model_priority(session_context_.model_priority));

#if (((OPENVINO_VERSION_MAJOR == 2024) && (OPENVINO_VERSION_MINOR > 3)) || (OPENVINO_VERSION_MAJOR > 2024))
  if (session_context_.device_type.find("NPU") != std::string::npos) {
    if (session_context_.so_context_enable) {
      OVCore::Get()->core.set_property("NPU", ov::intel_npu::bypass_umd_caching(true));
    }
  }
#endif

  if (!session_context_.load_config.empty()) {
    const std::map<std::string, ov::AnyMap>& target_config = session_context_.load_config;
    // Only skip compilation params for pre-compiled blob import, not OVIR (which compiles OVIR)
    const bool skip_compilation_params = subgraph_context_.is_ep_ctx_graph && !subgraph_context_.is_ep_ctx_ovir_encapsulated;

    // Extract device names from device string and apply their configs
    // Examples: "GPU" -> ["GPU"], "AUTO:GPU.0,CPU" -> ["AUTO", "GPU", "CPU"]
    auto apply_device_config = [&](std::string_view device) {
      if (device.empty()) return;

      // Remove device index: "GPU.0" -> "GPU"
      auto base_device = device.substr(0, device.find('.'));

      if (auto config_it = target_config.find(std::string(base_device)); config_it != target_config.end()) {
        for (const auto& [key, value] : config_it->second) {
          // Skip compilation-only parameters when importing pre-compiled models
          if (skip_compilation_params && key == "NPU_COMPILATION_MODE_PARAMS") continue;
          device_config[key] = value;
        }
      }
    };

    // Parse device string by splitting on ':' and ',' delimiters
    const auto& device_str = session_context_.device_type;
    for (size_t start = 0, pos = 0; pos <= device_str.size(); ++pos) {
      if (pos == device_str.size() || device_str[pos] == ':' || device_str[pos] == ',') {
        if (pos > start) {
          apply_device_config(std::string_view(device_str).substr(start, pos - start));
        }
        start = pos + 1;
      }
    }
  }
}

void BasicBackend::EnableCaching(ov::AnyMap& device_config) {
  // cache_dir argument has no effect when working with an embed-mode EPContext Graph
  if (subgraph_context_.is_ep_ctx_graph) return;

  if (!session_context_.cache_dir.empty() && !session_context_.so_context_enable) {
    LOGS_DEFAULT(INFO) << log_tag << "Enables Caching";
    device_config.emplace(ov::cache_dir(session_context_.cache_dir.string()));
  }
}

void BasicBackend::EnableGPUThrottling(ov::AnyMap& device_config) {
  if (session_context_.enable_opencl_throttling == true &&
      session_context_.device_type.find("GPU") != std::string::npos) {
    LOGS_DEFAULT(INFO) << log_tag << "Enabled OpenCL queue throttling for GPU device";
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("PLUGIN_THROTTLE", "1");
    device_config.emplace(ov::device::properties("GPU_CONFIG_KEY", device_property));
  }
}

void BasicBackend::EnableStreams(ov::AnyMap& device_config) {
  // Return silently for NPU as it's currently treated as a read-only flag by the NPU plugin
  // and throws an exception for the same
  if (session_context_.device_type.find("NPU") != std::string::npos)
    return;

  // Streams can be set only if the device is not one of AUTO, MULTI, or HETERO
  // Throw an exception if the user tries to set num_streams for these devices
  if ((session_context_.device_type.find("MULTI") != std::string::npos) ||
      (session_context_.device_type.find("HETERO") != std::string::npos) ||
      (session_context_.device_type.find("AUTO") != std::string::npos)) {
    if (session_context_.num_streams != 1) {
      ORT_THROW(log_tag + "Cannot set NUM_STREAMS to " +
                std::to_string(session_context_.num_streams) + " for device " + session_context_.device_type);
    }
    // Do nothing
  } else {
    device_config.emplace(ov::num_streams(session_context_.num_streams));
  }
}

void BasicBackend::SetNumThreads(ov::AnyMap& device_config) {
  // inference_num_threads is applicable only for the CPU device
  if (session_context_.device_type.find("CPU") != std::string::npos)
    device_config.emplace(ov::inference_num_threads(session_context_.num_of_threads));
}

void BasicBackend::SetOVDeviceConfiguration(ov::AnyMap& device_config) {
  PopulateConfigValue(device_config);

  // Enable caching
  EnableCaching(device_config);

  // Setting OpenCL queue throttling for GPU
  EnableGPUThrottling(device_config);

  // Enable streams; default=1 unless overridden by user configuration
  EnableStreams(device_config);

  // Set the inference_num_threads property of the CPU
  SetNumThreads(device_config);

  auto npuw_status =
      std::any_of(device_config.begin(), device_config.end(), [&](const std::pair<std::string, ov::Any>& pair) {
        return (pair.first.find("NPU_USE_NPUW") != std::string::npos) && (pair.second.is<std::string>()) &&
               (pair.second.as<std::string>() == "YES");
      });

  if (npuw_status) {
    LOGS_DEFAULT(INFO) << log_tag << "NPUW Enabled during compilation";
  }
}

void BasicBackend::RewindKVCache(size_t index) {
  infer_req_pool_->forEachIdleRequest([&](OVInferRequestPtr& infer_request) {
    infer_request->RewindKVCache(index);
  });
}

void BasicBackend::SetReorderKVCacheStatus(const std::vector<int32_t>& src_indices, const std::vector<int32_t>& dst_indices) {
  infer_req_pool_->forEachIdleRequest([&](OVInferRequestPtr& infer_request) {
    infer_request->SetReorderKVCacheStatus(src_indices, dst_indices);
  });
}

void BasicBackend::Infer(OrtKernelContext* ctx) {
  Ort::KernelContext context(ctx);

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
    for (const auto& item : const_outputs_map_) {
      std::string out_name = item.first;
      std::shared_ptr<ov::Node> node = item.second;
      Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                        std::move(out_name),
                                                        subgraph_context_.output_names,
                                                        node);
      FillOutputsWithConstantData(std::move(node), output_tensor);
    }

    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";

    if (IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }
    return;
  }

  // guarded_request will be released back to the pool when it goes out of scope
  auto guarded_request = infer_req_pool_->getRequest();
  auto& infer_request = guarded_request.infer_request_;

  if (bindings_->has_dynamic_io_) {
    // Dynamic shape inference

    // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
    // tensors instead of binding them to the infer request directly.

    // Bind inputs
    for (const auto& input_info : bindings_->network_inputs_) {
      // Set the input shape based on the input tensor from ort
      auto tensor = context.GetInput(input_info.onnx_index);
      auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
      auto input_shape = ParameterShape(ort_shape);

      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_shape,
                               const_cast<void*>(tensor.GetTensorRawData()));
    }

    // Run Inference
    infer_request->Infer();

    // Copy outputs
    for (const auto& output_info : bindings_->network_outputs_) {
      auto ov_tensor = infer_request->GetTensor(output_info.name);
      auto output_shape = ParameterShape::ToOrtShape(ov_tensor->get_shape());
      auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);

      ORT_ENFORCE(ov_tensor->get_byte_size() == ort_tensor.GetTensorSizeInBytes(),
                  log_tag + "Output tensor size mismatch for " + output_info.name);

      std::memcpy(ort_tensor.GetTensorMutableRawData(),
                  ov_tensor->data(),
                  ov_tensor->get_byte_size());
    }
  } else {
    // Static shape inference

    // Bind inputs
    for (const auto& input_info : bindings_->network_inputs_) {
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_info.shape,
                               const_cast<void*>(context.GetInput(input_info.onnx_index).GetTensorRawData()));
    }

    // Bind outputs
    for (const auto& output_info : bindings_->network_outputs_) {
      infer_request->SetTensor(output_info.name,
                               output_info.type,
                               output_info.shape,
                               context.GetOutput(output_info.onnx_index, output_info.shape).GetTensorMutableRawData());
    }

    // Run Inference
    infer_request->Infer();
  }

  // Fill constant outputs if needed
  for (const auto& [name, node] : const_outputs_map_) {
    Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                      name,
                                                      subgraph_context_.output_names,
                                                      node);
    FillOutputsWithConstantData(node, output_tensor);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
  if (IsCILogEnabled()) {
    std::cout << "Inference successful" << std::endl;
  }

  if (!dir_.empty()) {
    PerfDump(infer_request);
  }
}

void BasicBackend::PerfDirCreate(const std::string& perf_count) {
  bool is_profiling_enabled = false;
  try {
    is_profiling_enabled = GetOVCompiledModel().get_property(ov::enable_profiling);
  } catch (const ov::Exception& e) {
    LOGS_DEFAULT(INFO) << log_tag << e.what();
  }

  if (!is_profiling_enabled) {
    LOGS_DEFAULT(WARNING) << log_tag
                          << "Performance counting was requested, but OpenVINO compiled model profiling "
                          << "(ov::enable_profiling) is disabled. No perf CSV will be produced.";
    dir_.clear();
    return;
  }

  try {
    std::filesystem::path perf_path(perf_count);
    if (std::filesystem::exists(perf_path)) {
      if (!std::filesystem::is_directory(perf_path)) {
        LOGS_DEFAULT(INFO) << log_tag << "Perf count path '" << perf_path.string() << "' exists but is not a directory.";
        dir_.clear();
        return;
      }
    } else {
      std::filesystem::create_directories(perf_path);
    }
    dir_ = std::move(perf_path);
  } catch (const std::filesystem::filesystem_error& e) {
    LOGS_DEFAULT(INFO) << log_tag << "Filesystem error for perf count path '" << perf_count << "': " << e.what();
    dir_.clear();
  }
}

void BasicBackend::PerfDump(OVInferRequestPtr infer_request) {
  // Print performance counts before releasing the infer_request for thread safety
  static std::mutex _mutex;
  const std::string prefix = "OpenVINOExecutionProvider_OpenVINO-EP";
  std::string subgraph_suffix;
  if (subgraph_context_.subgraph_name.size() >= prefix.size() &&
      subgraph_context_.subgraph_name.compare(0, prefix.size(), prefix) == 0) {
    subgraph_suffix = subgraph_context_.subgraph_name.substr(prefix.size());
  } else {
    // Fallback: use the full subgraph name if it does not start with the expected prefix
    subgraph_suffix = subgraph_context_.subgraph_name;
  }

  std::string filestring = subgraph_suffix + "_perf_count.csv";
  std::filesystem::path filename(session_context_.GetModelPath().stem().string() + filestring);
  filename = dir_ / filename;
  std::unique_lock<std::mutex> lock(_mutex);
  std::ofstream out(filename);
  {
    if (out.is_open()) {
      printPerformanceCounts(infer_request, out);
      out.close();
    } else {
      LOGS_DEFAULT(INFO) << log_tag << filename << ": File error";
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
