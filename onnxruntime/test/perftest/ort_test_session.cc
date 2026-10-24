// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "ort_test_session.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <set>
#include <list>
#include <type_traits>
#include <core/framework/allocator.h>
#include <core/platform/threadpool_config.h>
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/dnnl/dnnl_provider_options.h"
#include <assert.h>
#include "providers.h"
#include "TestCase.h"
#include "strings_helper.h"
#include "utils.h"

#ifdef USE_OPENVINO
#include "nlohmann/json.hpp"
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#include "core/providers/dml/dml_session_options_config_keys.h"
#endif

#ifdef _WIN32
#define strdup _strdup
#endif
extern const OrtApi* g_ort;

namespace onnxruntime {
namespace perftest {

RunTiming OnnxRuntimeTestSession::Run() {
  // Select input set: round-robin for multi-shape mode, random otherwise.
  size_t id;
  if (use_round_robin_ && test_inputs_.size() > 1) {
    id = round_robin_counter_.fetch_add(1, std::memory_order_relaxed) % test_inputs_.size();
  } else {
    // Random selection (not thread-safe).
    const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(test_inputs_.size() - 1));
    id = static_cast<size_t>(dist_(rand_engine_, p));
  }

  auto& input = test_inputs_.at(id);
  auto start = std::chrono::high_resolution_clock::now();
  Ort::RunOptions run_options;
  for (const auto& kv : run_config_entries_) {
    run_options.AddConfigEntry(kv.first.c_str(), kv.second.c_str());
  }

  RunTiming timing;
  if (CUDA == device_memory_name_ || use_plugin_ep_io_binding_) {
    Ort::IoBinding io_binding(session_);
    auto mem_info = allocator_.GetInfo();

    for (size_t i = 0; i < input_names_.size(); ++i) {
      io_binding.BindInput(input_names_[i], input[i]);
    }
    // String outputs require host-accessible memory, so let the session allocate them on the CPU.
    const auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < output_names_.size(); ++i) {
      io_binding.BindOutput(output_names_[i].c_str(), is_output_string_[i] ? cpu_mem_info : mem_info);
    }

    // Plugin EPs may synchronize at run end (OnRunEnd / stream flush) without implementing OrtEp::Sync, in which case
    // IoBinding::SynchronizeOutputs() does not wait. Only defer synchronization for the built-in CUDA/TensorRT path.
    if (!use_plugin_ep_io_binding_) {
      // Use async execution and rely on IO binding's SynchronizeOutputs to synchronize.
      run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    }

    // do not time IO binding creation
    start = std::chrono::high_resolution_clock::now();
    session_.Run(run_options, io_binding);
    timing.submit_timing = std::chrono::high_resolution_clock::now() - start;
    io_binding.SynchronizeOutputs();
    timing.total_timing = std::chrono::high_resolution_clock::now() - start;
  } else {
    // For outputs with data-dependent shapes (e.g. NonZero), the shape changes
    // between runs. ORT caches the allocated tensor and fails shape verification
    // on the next run. Reset outputs so ORT always allocates fresh buffers.
    // Only do this for models with dynamic output shapes to avoid the allocation
    // overhead on fixed-shape models.
    if (has_dynamic_output_shapes_) {
      for (size_t i = 0; i < outputs_.size(); ++i) {
        if (is_output_dynamic_[i]) {
          outputs_[i] = Ort::Value(nullptr);
        }
      }
    }
    session_.Run(run_options, input_names_.data(), input.data(), input_names_.size(),
                 output_names_raw_ptr.data(), outputs_.data(), output_names_raw_ptr.size());
    timing.submit_timing = std::chrono::high_resolution_clock::now() - start;
    timing.total_timing = timing.submit_timing;
  }
  timing.test_input_index = id;
  return timing;
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd,
                                               const PerformanceTestConfig& performance_test_config,
                                               const TestModelInfo& m)
    : rand_engine_(rd()),
      input_names_(m.GetInputCount()),
      input_names_str_(m.GetInputCount()),
      input_length_(m.GetInputCount()),
      run_config_entries_(performance_test_config.run_config.run_config_entries),
      env_(env) {
  Ort::SessionOptions session_options;

  // Add EP devices if any (created by plugin EP)
  if (!performance_test_config.registered_plugin_eps.empty()) {
    std::vector<Ort::ConstEpDevice> selected_ep_devices =
        perftest::utils::AppendPluginExecutionProviders(env, session_options, performance_test_config);

    // Pick an allocator from the plugin EP devices unless the user explicitly requested to force the CPU allocator.
    if (performance_test_config.plugin_ep_force_cpu_allocator) {
      fprintf(stdout, "[Plugin EP] Forcing CPU allocator (--plugin_ep_force_cpu_allocator was specified).\n");
    } else if (auto plugin_ep_allocator = perftest::utils::GetPluginEpAllocator(env, selected_ep_devices)) {
      allocator_ = plugin_ep_allocator->allocator;
      plugin_ep_allocator_selection_ = std::move(plugin_ep_allocator);
    }

    // IO binding binds outputs to allocator_'s memory info, which comes from the selected EP device (including its
    // device id), so it is only used when that allocator is device memory.
    if (performance_test_config.run_config.enable_cuda_io_binding) {
      use_plugin_ep_io_binding_ = IsAllocatorDeviceOnly();
      if (!use_plugin_ep_io_binding_) {
        if (performance_test_config.plugin_ep_force_cpu_allocator) {
          fprintf(stdout,
                  "[Plugin EP] -g is ignored because --plugin_ep_force_cpu_allocator was specified. "
                  "IO binding is disabled.\n");
        } else {
          fprintf(stdout,
                  "[Plugin EP] -g requires a single selected EP device with a device allocator. "
                  "IO binding is disabled.\n");
        }
      }
    }
  }

  provider_name_ = performance_test_config.machine_config.provider_type_name;
  std::unordered_map<std::string, std::string> provider_options;

  if (provider_name_ == onnxruntime::kDnnlExecutionProvider) {
#ifdef USE_DNNL
    // Generate provider options
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = 1;
    dnnl_options.threadpool_args = nullptr;

#if !defined(DNNL_ORT_THREAD)
#if defined(_MSC_VER)
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif  // defined(_MSC_VER)
    int num_threads = 0;
    ParseSessionConfigs(ov_string, provider_options, {"num_of_threads"});
    for (const auto& provider_option : provider_options) {
      if (provider_option.first == "num_of_threads") {
        std::stringstream sstream(provider_option.second);
        sstream >> num_threads;
        if (num_threads < 0) {
          ORT_THROW(
              "[ERROR] [OneDNN] Invalid entry for the key 'num_of_threads',"
              " set number of threads or use '0' for default\n");
          // If the user doesnt define num_threads, auto detect threads later
        }
      }
    }
    dnnl_options.threadpool_args = static_cast<void*>(&num_threads);
#endif  // !defined(DNNL_ORT_THREAD)
    dnnl_options.use_arena = performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0;

    session_options.AppendExecutionProvider_Dnnl(dnnl_options);
#else
    ORT_THROW("DNNL is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA

    Ort::CUDAProviderOptions cuda_options;
    const char* config_val = nullptr;
    switch (performance_test_config.run_config.cudnn_conv_algo) {
      case 0:
        config_val = "EXHAUSTIVE";
        break;
      case 1:
        config_val = "HEURISTIC";
        break;
      default:
        config_val = "DEFAULT";
        break;
    }
    provider_options.emplace("cudnn_conv_algo_search", config_val);
    provider_options.emplace("do_copy_in_default_stream",
                             (!performance_test_config.run_config.do_cuda_copy_in_separate_stream ? "1" : "0"));

    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);

    ParseSessionConfigs(ov_string, provider_options);
    if (performance_test_config.run_config.enable_cuda_io_binding) {
      device_memory_name_ = CUDA;
      if (cudaStreamCreate(&stream_) != cudaError_t::cudaSuccess) {
        ORT_THROW("Unable to create CUDA stream for IOBinding");
      }
      auto stream_str = std::to_string(reinterpret_cast<uintptr_t>(stream_));
      provider_options["user_compute_stream"] = stream_str;
    }
    cuda_options.Update(provider_options);

    if (performance_test_config.run_config.cuda_mempool_arena_config) {
      // Enable and configure cuda_mempool arena
      const size_t release_threshold =
          static_cast<size_t>(std::atoll(performance_test_config.run_config.cuda_mempool_arena_config->release_threshold.c_str()));
      const size_t bytes_to_keep_on_shrink =
          static_cast<size_t>(std::atoll(performance_test_config.run_config.cuda_mempool_arena_config->bytes_to_keep.c_str()));
      // Create a map of properties for the arena configuration
      std::unordered_map<std::string, size_t> arena_config_map = {
          {"use_cuda_mempool", 1U},
          {"cuda_mempool_bytes_to_keep_on_shrink", bytes_to_keep_on_shrink},
          {"cuda_mempool_release_threshold", release_threshold},
      };
      // Must be kept alive while session is alive
      Ort::ArenaCfg cuda_arena_cfg(arena_config_map);
      cuda_mempool_arena_cfg_ = std::move(cuda_arena_cfg);
      (*cuda_options).default_memory_arena_cfg = cuda_mempool_arena_cfg_;
    }

    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    Ort::TensorRTProviderOptions tensorrt_options;
    // used to keep all option keys and value strings alive
    std::list<std::string> buffer;
    OrtCUDAProviderOptions cuda_options;
    if (performance_test_config.run_config.enable_cuda_io_binding) {
      device_memory_name_ = CUDA;
      if (cudaStreamCreate(&stream_) != cudaError_t::cudaSuccess) {
        ORT_THROW("Unable to create CUDA stream for IOBinding");
      }
      auto stream_str = std::to_string(reinterpret_cast<uintptr_t>(stream_));
      cuda_options.has_user_compute_stream = 1;
      cuda_options.user_compute_stream = reinterpret_cast<void*>(stream_);
      provider_options["user_compute_stream"] = stream_str;
    }
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(ov_string, provider_options);
    tensorrt_options.Update(provider_options);
    session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

    cuda_options.device_id = static_cast<OrtTensorRTProviderOptionsV2*>(tensorrt_options)->device_id;
    cuda_options.cudnn_conv_algo_search = static_cast<OrtCudnnConvAlgoSearch>(performance_test_config.run_config.cudnn_conv_algo);
    cuda_options.do_copy_in_default_stream = !performance_test_config.run_config.do_cuda_copy_in_separate_stream;
    // TODO: Support arena configuration for users of perf test
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kQnnExecutionProvider) {
#ifdef USE_QNN
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(option_string, provider_options,
                        {"backend_type", "backend_path", "profiling_file_path", "profiling_level",
                         "rpc_control_latency", "vtcm_mb", "soc_model", "device_id", "htp_performance_mode", "op_packages",
                         "qnn_saver_path", "htp_graph_finalization_optimization_mode", "qnn_context_priority",
                         "htp_arch", "enable_htp_fp16_precision", "offload_graph_io_quantization",
                         "enable_htp_spill_fill_buffer", "enable_htp_shared_memory_allocator", "dump_json_qnn_graph",
                         "json_qnn_graph_dir", "disable_file_mapped_weights", "htp_bf16_enable", "enable_vtcm_backup_buffer_sharing", "extended_udma"});

    for (const auto& provider_option : provider_options) {
      const std::string& key = provider_option.first;
      const std::string& value = provider_option.second;
      if (key == "backend_path" || key == "profiling_file_path" || key == "json_qnn_graph_dir") {
        if (value.empty()) {
          ORT_THROW("Please provide the valid file path.");
        }
      } else if (key == "profiling_level") {
        std::set<std::string> supported_profiling_level = {"off", "basic", "detailed", "optrace"};
        if (supported_profiling_level.find(value) == supported_profiling_level.end()) {
          std::ostringstream str_stream;
          std::copy(supported_profiling_level.begin(), supported_profiling_level.end(),
                    std::ostream_iterator<std::string>(str_stream, ","));
          std::string str = str_stream.str();
          ORT_THROW("Supported profiling_level: " + str);
        }
      } else if (key == "backend_type" || key == "rpc_control_latency" || key == "vtcm_mb" || key == "soc_model" ||
                 key == "device_id") {
        // no validation
      } else if (key == "htp_performance_mode") {
        std::set<std::string> supported_htp_perf_mode = {"burst", "balanced", "default", "high_performance",
                                                         "high_power_saver", "low_balanced", "extreme_power_saver", "low_power_saver",
                                                         "power_saver", "sustained_high_performance"};
        if (supported_htp_perf_mode.find(value) == supported_htp_perf_mode.end()) {
          std::ostringstream str_stream;
          std::copy(supported_htp_perf_mode.begin(), supported_htp_perf_mode.end(),
                    std::ostream_iterator<std::string>(str_stream, ","));
          std::string str = str_stream.str();
          ORT_THROW("Supported htp_performance_mode: " + str);
        }
      } else if (key == "op_packages") {
        if (value.empty()) {
          ORT_THROW("Please provide the valid op_packages.");
        }
      } else if (key == "qnn_saver_path") {
        // no validation
      } else if (key == "htp_graph_finalization_optimization_mode") {
        std::set<std::string> supported_htp_graph_final_opt_modes = {"0", "1", "2", "3"};
        if (supported_htp_graph_final_opt_modes.find(value) == supported_htp_graph_final_opt_modes.end()) {
          std::ostringstream str_stream;
          std::copy(supported_htp_graph_final_opt_modes.begin(), supported_htp_graph_final_opt_modes.end(),
                    std::ostream_iterator<std::string>(str_stream, ","));
          std::string str = str_stream.str();
          ORT_THROW("Wrong value for htp_graph_finalization_optimization_mode. select from: " + str);
        }
      } else if (key == "qnn_context_priority") {
        std::set<std::string> supported_qnn_context_priority = {"low", "normal", "normal_high", "high"};
        if (supported_qnn_context_priority.find(value) == supported_qnn_context_priority.end()) {
          ORT_THROW("Supported qnn_context_priority: low, normal, normal_high, high");
        }
      } else if (key == "htp_arch") {
        std::set<std::string> supported_htp_archs = {"0", "68", "69", "73", "75", "81"};
        if (supported_htp_archs.find(value) == supported_htp_archs.end()) {
          std::ostringstream str_stream;
          std::copy(supported_htp_archs.begin(), supported_htp_archs.end(),
                    std::ostream_iterator<std::string>(str_stream, ","));
          std::string str = str_stream.str();
          ORT_THROW("Wrong value for htp_arch. select from: " + str);
        }
      } else if (key == "enable_htp_fp16_precision" ||
                 key == "offload_graph_io_quantization" ||
                 key == "enable_htp_spill_fill_buffer" ||
                 key == "enable_htp_shared_memory_allocator" ||
                 key == "dump_json_qnn_graph" ||
                 key == "extended_udma" ||
                 key == "disable_file_mapped_weights" ||
                 key == "enable_vtcm_backup_buffer_sharing") {
        std::set<std::string> supported_options = {"0", "1"};
        if (supported_options.find(value) == supported_options.end()) {
          std::ostringstream str_stream;
          std::copy(supported_options.begin(), supported_options.end(),
                    std::ostream_iterator<std::string>(str_stream, ","));
          std::string str = str_stream.str();
          ORT_THROW("Wrong value for ", key, ". select from: ", str);
        }

        if (key == "enable_htp_shared_memory_allocator" && value == "1") {
          // if this option is set, also use the enabled allocator
          device_memory_name_ = "QnnHtpShared";
        }
      }
    }
    session_options.AppendExecutionProvider("QNN", provider_options);
#else
    ORT_THROW("QNN is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kSnpeExecutionProvider) {
#ifdef USE_SNPE
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(option_string, provider_options, {"runtime", "priority", "buffer_type", "enable_init_cache"});
    for (const auto& provider_option : provider_options) {
      if (key == "runtime") {
        std::set<std::string> supported_runtime = {"CPU", "GPU_FP32", "GPU", "GPU_FLOAT16", "DSP", "AIP_FIXED_TF"};
        if (supported_runtime.find(value) == supported_runtime.end()) {
          ORT_THROW(R"(Wrong configuration value for the key 'runtime'.
select from 'CPU', 'GPU_FP32', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n)");
        }
      } else if (key == "priority") {
        // no validation
      } else if (key == "buffer_type") {
        std::set<std::string> supported_buffer_type = {"TF8", "TF16", "UINT8", "FLOAT", "ITENSOR"};
        if (supported_buffer_type.find(value) == supported_buffer_type.end()) {
          ORT_THROW(R"(Wrong configuration value for the key 'buffer_type'.
select from 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. \n)");
        }
      } else if (key == "enable_init_cache") {
        if (value != "1") {
          ORT_THROW("Set to 1 to enable_init_cache.");
        }
      }
    }

    session_options.AppendExecutionProvider("SNPE", provider_options);
#else
    ORT_THROW("SNPE is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    uint32_t nnapi_flags = 0;
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::istringstream ss(ov_string);
    std::string key;
    while (ss >> key) {
      if (key == "NNAPI_FLAG_USE_FP16") {
        nnapi_flags |= NNAPI_FLAG_USE_FP16;
      } else if (key == "NNAPI_FLAG_USE_NCHW") {
        nnapi_flags |= NNAPI_FLAG_USE_NCHW;
      } else if (key == "NNAPI_FLAG_CPU_DISABLED") {
        nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
      } else if (key == "NNAPI_FLAG_CPU_ONLY") {
        nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
      } else if (key.empty()) {
      } else {
        ORT_THROW(
            "[ERROR] [NNAPI] wrong key type entered. Choose from the following runtime key options "
            "that are available for NNAPI. "
            "['NNAPI_FLAG_USE_FP16', 'NNAPI_FLAG_USE_NCHW', 'NNAPI_FLAG_CPU_DISABLED', 'NNAPI_FLAG_CPU_ONLY'] \n");
      }
    }
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kVSINPUExecutionProvider) {
#ifdef USE_VSINPU
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_VSINPU(session_options));
#else
    ORT_THROW("VSINPU is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kCoreMLExecutionProvider) {
#ifdef __APPLE__
#ifdef USE_COREML
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
    static const std::unordered_set<std::string> available_keys = {kCoremlProviderOption_MLComputeUnits,
                                                                   kCoremlProviderOption_ModelFormat,
                                                                   kCoremlProviderOption_RequireStaticInputShapes,
                                                                   kCoremlProviderOption_EnableOnSubgraphs,
                                                                   kCoremlProviderOption_SpecializationStrategy,
                                                                   kCoremlProviderOption_ProfileComputePlan,
                                                                   kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU,
                                                                   kCoremlProviderOption_ModelCacheDirectory};
    ParseSessionConfigs(ov_string, provider_options, available_keys);

    std::unordered_map<std::string, std::string> available_options = {
        {"CPUAndNeuralEngine", "1"},
        {"CPUAndGPU", "1"},
        {"CPUOnly", "1"},
        {"ALL", "1"},
    };
    for (const auto& provider_option : provider_options) {
      if (provider_option.first == kCoremlProviderOption_MLComputeUnits &&
          available_options.find(provider_option.second) != available_options.end()) {
      } else if (provider_option.first == kCoremlProviderOption_ModelFormat &&
                 (provider_option.second == "MLProgram" || provider_option.second == "NeuralNetwork")) {
      } else if (provider_option.first == kCoremlProviderOption_RequireStaticInputShapes &&
                 (provider_option.second == "1" || provider_option.second == "0")) {
      } else if (provider_option.first == kCoremlProviderOption_EnableOnSubgraphs &&
                 (provider_option.second == "0" || provider_option.second == "1")) {
      } else if (provider_option.first == kCoremlProviderOption_SpecializationStrategy &&
                 (provider_option.second == "Default" || provider_option.second == "FastPrediction")) {
      } else if (provider_option.first == kCoremlProviderOption_ProfileComputePlan &&
                 (provider_option.second == "0" || provider_option.second == "1")) {
      } else if (provider_option.first == kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU &&
                 (provider_option.second == "0" || provider_option.second == "1")) {
      } else if (provider_option.first == kCoremlProviderOption_ModelCacheDirectory) {
      } else {
        ORT_THROW("Invalid value for option ", provider_option.first, ": ", provider_option.second);
      }
    }
    // COREML_FLAG_CREATE_MLPROGRAM
    session_options.AppendExecutionProvider("CoreML", provider_options);
#else
    ORT_THROW("CoreML is not supported in this build\n");
#endif
#else
    ORT_THROW("COREML is not supported on this platform.\n");
#endif
  } else if (provider_name_ == onnxruntime::kDmlExecutionProvider) {
#ifdef USE_DML
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(ov_string, provider_options,
                        {"device_filter", "performance_preference", "disable_metacommands",
                         "enable_graph_capture", "enable_graph_serialization"});
    for (const auto& provider_option : provider_options) {
      const std::string& key = provider_option.first;
      const std::string& value = provider_option.second;
      if (key == "device_filter") {
        std::set<std::string> ov_supported_device_types = {"gpu", "npu"};
        if (ov_supported_device_types.find(value) != ov_supported_device_types.end()) {
        } else {
          ORT_THROW(
              "[ERROR] [DML] You have selected a wrong configuration value for the key 'device_filter'. "
              "Select from 'gpu', or 'npu' \n");
        }
      } else if (key == "performance_preference") {
        std::set<std::string> ov_supported_values = {"default", "high_performance", "minimum_power"};
        if (ov_supported_values.find(value) != ov_supported_values.end()) {
        } else {
          ORT_THROW(
              "[ERROR] [DML] You have selected a wrong configuration value for the key 'performance_preference'. "
              "Select from 'default', 'high_performance' or 'minimum_power' \n");
        }
      } else if (key == "disable_metacommands") {
        std::set<std::string> ov_supported_values = {"true", "True", "false", "False"};
        if (ov_supported_values.find(value) != ov_supported_values.end()) {
        } else {
          ORT_THROW(
              "[ERROR] [DML] You have selected a wrong value for the key 'disable_metacommands'. "
              "Select from 'true' or 'false' \n");
        }
      } else if (key == "enable_graph_capture") {
        std::set<std::string> ov_supported_values = {"true", "True", "false", "False"};
        if (ov_supported_values.find(value) != ov_supported_values.end()) {
        } else {
          ORT_THROW(
              "[ERROR] [DML] You have selected a wrong value for the key 'enable_graph_capture'. "
              "Select from 'true' or 'false' \n");
        }
      } else if (key == "enable_graph_serialization") {
        std::set<std::string> ov_supported_values = {"true", "True", "false", "False"};
        if (ov_supported_values.find(value) != ov_supported_values.end()) {
          session_options.AddConfigEntry(kOrtSessionOptionsConfigEnableGraphSerialization, value.data());
        } else {
          ORT_THROW(
              "[ERROR] [DML] You have selected a wrong value for the key 'enable_graph_serialization'. "
              "Select from 'true' or 'false' \n");
        }
      }
    }
    if (provider_options.find("performance_preference") == provider_options.end()) {
      provider_options["performance_preference"] = "high_performance";
    }
    if (provider_options.find("device_filter") == provider_options.end()) {
      provider_options["device_filter"] = "gpu";
    }
    if (provider_options.find("disable_metacommands") == provider_options.end()) {
      provider_options["disable_metacommands"] = "false";
    }
    if (provider_options.find("enable_graph_capture") == provider_options.end()) {
      provider_options["enable_graph_capture"] = "false";
    }
    session_options.AppendExecutionProvider("DML", provider_options);
#else
    ORT_THROW("DML is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kAclExecutionProvider) {
#ifdef USE_ACL
#if defined(_MSC_VER)
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif  // defined(_MSC_VER)
    bool enable_fast_math = false;
    ParseSessionConfigs(ov_string, provider_options, {"enable_fast_math"});
    for (const auto& provider_option : provider_options) {
      const std::string& key = provider_option.first;
      const std::string& value = provider_option.second;
      if (key == "enable_fast_math") {
        std::set<std::string> ov_supported_values = {"true", "True", "false", "False"};
        if (ov_supported_values.find(value) != ov_supported_values.end()) {
          enable_fast_math = (value == "true") || (value == "True");
        } else {
          ORT_THROW(
              "[ERROR] [ACL] You have selcted an invalid value for the key 'enable_fast_math'. "
              "Select from 'true' or 'false' \n");
        }
      }
    }
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_ACL(session_options, enable_fast_math));
#else
    ORT_THROW("Acl is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(session_options, 0));
#else
    ORT_THROW("MIGraphX is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kXnnpackExecutionProvider) {
#ifdef USE_XNNPACK
    session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    session_options.AppendExecutionProvider(
        "XNNPACK", {{"intra_op_num_threads", std::to_string(performance_test_config.run_config.intra_op_num_threads)}});
#else
    ORT_THROW("Xnnpack is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kWebGpuExecutionProvider) {
#ifdef USE_WEBGPU
    // Use the short key form here: -i "enableGraphCapture|1". AppendExecutionProvider prefixes
    // provider-option keys with "ep.webgpuexecutionprovider." itself, so a fully qualified key is
    // double-prefixed and silently ignored. Session config entries (-C) do not reach the EP
    // factory, which reads these at append time.
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(option_string, provider_options);
    session_options.AppendExecutionProvider("WebGPU", provider_options);
#else
    ORT_THROW("WebGPU is not supported in this build\n");
#endif
  } else if (provider_name_ == onnxruntime::kVitisAIExecutionProvider) {
#ifdef USE_VITISAI
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    ParseSessionConfigs(option_string, provider_options);

    session_options.AppendExecutionProvider_VitisAI(provider_options);
#else
    ORT_THROW("VitisAI is not supported in this build\n");
#endif
  } else if (!provider_name_.empty() &&
             provider_name_ != onnxruntime::kCpuExecutionProvider &&
             provider_name_ != onnxruntime::kOpenVINOExecutionProvider) {
    ORT_THROW("This backend is not included in perf test runner.\n");
  }

  if (performance_test_config.run_config.enable_cpu_mem_arena)
    session_options.EnableCpuMemArena();
  else
    session_options.DisableCpuMemArena();
  if (performance_test_config.run_config.enable_memory_pattern &&
      performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL)
    session_options.EnableMemPattern();
  else
    session_options.DisableMemPattern();
  session_options.SetExecutionMode(performance_test_config.run_config.execution_mode);

  // Set any extra session configuration entries provided by the user via command-line arguments.
  //
  // Some session config entries can also be set via dedicated command-line options.
  // If the user uses multiple command-line options to set the same session config entry,
  // we'll print a warning. Note that the dedicated command-line options will take precedence.
  const auto& user_session_configs = performance_test_config.run_config.session_config_entries;
  for (auto& it : user_session_configs) {
    session_options.AddConfigEntry(it.first.c_str(), it.second.c_str());
  }

  auto warn_dup_config_entry = [&user_session_configs](const char* key) -> void {
    if (user_session_configs.find(key) != user_session_configs.end()) {
      fprintf(stderr, "[WARNING]: Trying to set session config entry '%s' via multiple command-line options\n", key);
    }
  };

  if (performance_test_config.run_config.intra_op_num_threads > 0) {
    fprintf(stdout, "Setting intra_op_num_threads to %d\n", performance_test_config.run_config.intra_op_num_threads);
    session_options.SetIntraOpNumThreads(performance_test_config.run_config.intra_op_num_threads);
  }

  if (!performance_test_config.run_config.intra_op_thread_affinities.empty()) {
    warn_dup_config_entry(kOrtSessionOptionsConfigIntraOpThreadAffinities);
    fprintf(stdout, "Setting intra op thread affinity as %s\n", performance_test_config.run_config.intra_op_thread_affinities.c_str());
    session_options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, performance_test_config.run_config.intra_op_thread_affinities.c_str());
  }

  if (performance_test_config.run_config.disable_spinning) {
    warn_dup_config_entry(kOrtSessionOptionsConfigAllowIntraOpSpinning);
    fprintf(stdout, "Disabling intra-op thread spinning entirely\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
  }

  if (performance_test_config.run_config.spin_duration_us >= 0) {
    if (performance_test_config.run_config.disable_spinning) {
      fprintf(stdout, "Ignoring intra-op spin duration because spinning is disabled\n");
    } else {
      warn_dup_config_entry(kOrtSessionOptionsConfigIntraOpSpinDurationUs);
      auto val = std::to_string(performance_test_config.run_config.spin_duration_us);
      fprintf(stdout, "Setting intra-op spin duration to %s us\n", val.c_str());
      session_options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpSpinDurationUs, val.c_str());
    }
  }

  if (performance_test_config.run_config.spin_backoff_max_set &&
      performance_test_config.run_config.spin_backoff_max >= 1) {
    if (performance_test_config.run_config.disable_spinning) {
      fprintf(stdout, "Ignoring intra-op spin backoff max because spinning is disabled\n");
    } else {
      warn_dup_config_entry(kOrtSessionOptionsConfigIntraOpSpinBackoffMax);
      const auto requested_spin_backoff_max = performance_test_config.run_config.spin_backoff_max;
      const auto effective_spin_backoff_max =
          std::min(static_cast<unsigned int>(requested_spin_backoff_max), concurrency::kSpinBackoffMaxLimit);
      if (effective_spin_backoff_max != static_cast<unsigned int>(requested_spin_backoff_max)) {
        fprintf(stdout,
                "Requested intra-op spin backoff max %d exceeds the runtime limit %u; clamping to %u\n",
                requested_spin_backoff_max,
                concurrency::kSpinBackoffMaxLimit,
                effective_spin_backoff_max);
      }
      auto val = std::to_string(effective_spin_backoff_max);
      fprintf(stdout, "Setting intra-op spin backoff max to %s\n", val.c_str());
      session_options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpSpinBackoffMax, val.c_str());
    }
  } else if (performance_test_config.run_config.spin_backoff_max_set) {
    fprintf(stderr,
            "Warning: --spin_backoff_max must be >= 1; got %d. Ignoring (using default).\n",
            performance_test_config.run_config.spin_backoff_max);
  }

  if (performance_test_config.run_config.disable_spinning_between_run) {
    warn_dup_config_entry(kOrtSessionOptionsConfigForceSpinningStop);
    fprintf(stdout, "Disabling intra-op thread spinning between runs\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop, "1");
  }

  if (!performance_test_config.run_config.register_custom_op_path.empty()) {
    session_options.RegisterCustomOpsLibrary(performance_test_config.run_config.register_custom_op_path.c_str());
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL && performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    session_options.SetInterOpNumThreads(performance_test_config.run_config.inter_op_num_threads);
  }

  // Set optimization level.
  session_options.SetGraphOptimizationLevel(performance_test_config.run_config.optimization_level);
  if (!performance_test_config.run_config.profile_file.empty()) {
    session_options.EnableProfiling(performance_test_config.run_config.profile_file.c_str());
  }
  if (!performance_test_config.run_config.optimized_model_path.empty()) {
    session_options.SetOptimizedModelFilePath(performance_test_config.run_config.optimized_model_path.c_str());
    if (!performance_test_config.run_config.optimized_model_data_path.empty()) {
      session_options.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersFileName,
                                     performance_test_config.run_config.optimized_model_data_path.c_str());
      if (!performance_test_config.run_config.optimized_model_weight_min_size.empty()) {
        session_options.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes,
                                       performance_test_config.run_config.optimized_model_weight_min_size.c_str());
      }
      if (performance_test_config.run_config.optimized_save_optimized_prepacks) {
        session_options.AddConfigEntry(kOrtSessionOptionsSavePrePackedConstantInitializers, "1");
      }
    }
  }
  if (performance_test_config.run_config.set_denormal_as_zero) {
    warn_dup_config_entry(kOrtSessionOptionsConfigSetDenormalAsZero);
    session_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");
  }
  if (!performance_test_config.run_config.free_dim_name_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_name_overrides) {
      if (g_ort->AddFreeDimensionOverrideByName(session_options, ToUTF8String(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverrideByName failed for named dimension: %s\n", ToUTF8String(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with name, %s, to %d\n", ToUTF8String(dim_override.first).c_str(), (int)dim_override.second);
      }
    }
  }
  if (!performance_test_config.run_config.free_dim_denotation_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_denotation_overrides) {
      if (g_ort->AddFreeDimensionOverride(session_options, ToUTF8String(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverride failed for dimension denotation: %s\n", ToUTF8String(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with denotation, %s, to %d\n", ToUTF8String(dim_override.first).c_str(), (int)dim_override.second);
      }
    }
  }
  if (provider_name_ == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::unordered_map<std::string, std::string> ov_options;
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW("[ERROR] [OpenVINO] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      auto key = token.substr(0, pos);
      auto value = token.substr(pos + 1);

      if (key == "device_type") {
        std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                           "GPU.0", "GPU.1", "NPU"};
        std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                         "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                         "GPU.0_FP16", "GPU.1_FP16"};
        size_t num_gpus = 10;
        for (size_t i = 0; i <= num_gpus; i++) {
          ov_supported_device_types.emplace("GPU." + std::to_string(i));
        }
        if (ov_supported_device_types.find(value) != ov_supported_device_types.end()) {
          ov_options[key] = value;
        } else if (deprecated_device_types.find(value) != deprecated_device_types.end()) {
          ov_options[key] = value;
        } else if (value.find("HETERO") == 0) {
          ov_options[key] = value;
        } else if (value.find("MULTI") == 0) {
          ov_options[key] = value;
        } else if (value.find("AUTO") == 0) {
          ov_options[key] = value;
        } else {
          ORT_THROW(
              "[ERROR] [OpenVINO] You have selcted wrong configuration value for the key 'device_type'. "
              "Select from 'CPU', 'GPU', 'GPU.0', 'GPU.1', 'NPU' or from"
              " HETERO/MULTI/AUTO options available. \n");
        }
      } else if (key == "device_id") {
        if (value == "CPU" || value == "GPU" || value == "NPU") {
          ov_options[key] = value;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
        }
      } else if (key == "precision") {
        auto device_type = ov_options["device_type"];
        if (device_type.find("GPU") != std::string::npos) {
          if (value == "") {
            ov_options[key] = "FP16";
            continue;
          } else if (value == "ACCURACY" || value == "FP16" || value == "FP32") {
            ov_options[key] = value;
            continue;
          } else {
            ORT_THROW(
                "[ERROR] [OpenVINO] Unsupported inference precision is selected. "
                "GPU only supported FP32 / FP16. \n");
          }
        } else if (device_type.find("NPU") != std::string::npos) {
          if (value == "" || value == "ACCURACY" || value == "FP16") {
            ov_options[key] = "FP16";
            continue;
          } else {
            ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. NPU only supported FP16. \n");
          }
        } else if (device_type.find("CPU") != std::string::npos) {
          if (value == "" || value == "ACCURACY" || value == "FP32") {
            ov_options[key] = "FP32";
            continue;
          } else {
            ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. CPU only supports FP32 . \n");
          }
        }
      } else if (key == "enable_opencl_throttling") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'enable_opencl_throttling' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "enable_qdq_optimizer") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'enable_qdq_optimizer' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "enable_causallm") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW(
              "[ERROR] [OpenVINO] The value for the key 'enable_causallm' should be a boolean i.e. true or false."
              " Default value is false. This provider option must be used with CausalLM Models viz. LLMs & SLMs only.\n");
        }
      } else if (key == "disable_dynamic_shapes") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW(
              "[ERROR] [OpenVINO] The value for the key 'enable_dynamic_shapes' "
              "should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "num_of_threads") {
        if (std::stoi(value) <= 0) {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'num_of_threads' should be greater than 0\n");
        } else {
          ov_options[key] = value;
        }
      } else if (key == "load_config") {
        auto load_json = [&](std::string filename) -> std::string {
          std::ifstream input_filestream(filename);
          if (!input_filestream.is_open()) {
            ORT_THROW("Passed an invalid JSON config file path \"" + filename + "\".");
          }
          nlohmann::json json_config;
          try {
            input_filestream >> json_config;
          } catch (const OnnxRuntimeException& ex) {
            ORT_THROW("Exception parsing config file \"" + filename + "\".\n" + ex.what());
          } catch (const std::exception& ex) {
            throw std::runtime_error("Standard exception for config file \"" + filename + "\".\n" + ex.what());
          } catch (...) {
            throw std::runtime_error("Unknown exception for config file \"" + filename + "\".\n");
          }
          if (json_config.empty()) {
            ORT_THROW("Empty JSON content passed \"" + filename + "\".");
          }
          return json_config.dump();
        };
        ov_options[key] = load_json(value);
      } else if (key == "model_priority") {
        ov_options[key] = value;
      } else if (key == "cache_dir") {
        ov_options[key] = value;
      } else if (key == "context") {
        ov_options[key] = value;
      } else if (key == "num_streams") {
        if (std::stoi(value) <= 0 && std::stoi(value) > 8) {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'num_streams' should be in the range of 1-8 \n");
        } else {
          ov_options[key] = value;
        }
      } else if (key == "device_memory_name") {
        device_memory_name_ = std::move(value);
      } else if (key == "device_luid") {
        ov_options[key] = value;
      } else if (key == "reshape_input") {
        ov_options[key] = value;
      } else if (key == "layout") {
        ov_options[key] = value;
      } else {
        ORT_THROW(
            "[ERROR] [OpenVINO] wrong key type entered. Choose from the following runtime key options that are available for OpenVINO."
            " ['device_type', 'device_id', 'num_of_threads', 'load_config', 'cache_dir', 'num_streams', "
            "'enable_opencl_throttling', 'disable_dynamic_shapes', 'enable_qdq_optimizer',"
            " 'enable_causallm', 'reshape_input', 'layout', 'model_priority'] \n");
      }
    }
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  }

  if (performance_test_config.run_config.use_extensions) {
    session_options.EnableOrtCustomOps();
  }

  if (!performance_test_config.model_info.load_via_path) {
    session_ = Ort::Session(env, performance_test_config.model_info.model_file_path.c_str(), session_options);
  } else {
    std::ifstream file(performance_test_config.model_info.model_file_path.c_str(),
                       std::ios::binary | std::ios::in | std::ios::ate);
    if (file.is_open()) {
      const std::streampos fsize = file.tellg();
      file.seekg(0, std::ios_base::beg);
      std::vector<char> model_bytes(narrow<size_t>(fsize));
      file.read(model_bytes.data(), narrow<std::streamsize>(fsize));
      session_ = Ort::Session(env, model_bytes.data(), model_bytes.size(), session_options);
    } else {
      ORT_THROW("Model file could not be opened.\n");
    }
  }
  size_t output_count = session_.GetOutputCount();
  output_names_.resize(output_count);
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i != output_count; ++i) {
    auto output_name = session_.GetOutputNameAllocated(i, a);
    assert(output_name != nullptr);
    output_names_[i] = output_name.get();
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }

  const size_t input_count = static_cast<size_t>(m.GetInputCount());
  for (size_t i = 0; i != input_count; ++i) {
    input_names_str_[i] = m.GetInputName(i);
    input_names_[i] = input_names_str_[i].c_str();
  }

  if (!device_memory_name_.empty()) {
    Ort::MemoryInfo memory_info(nullptr);  // Default initialize, will be overwritten
    if (device_memory_name_ == CUDA) {
      memory_info = Ort::MemoryInfo(device_memory_name_.data(), OrtArenaAllocator, 0, OrtMemTypeDefault);
    } else {
      memory_info = Ort::MemoryInfo(device_memory_name_.data(), OrtArenaAllocator, 0, OrtMemTypeCPUOutput);
    }
    custom_allocator_ = Ort::Allocator(session_, memory_info);
    // Switch to custom allocator
    allocator_ = Ort::UnownedAllocator(custom_allocator_);
  }
  for (size_t i = 0; i < output_names_raw_ptr.size(); i++) {
    Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    auto is_dynamic = std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end();
    is_output_dynamic_.push_back(is_dynamic);
    if (is_dynamic) {
      has_dynamic_output_shapes_ = true;
    }
    // String tensors require host-accessible memory; skip pre-allocation for them when allocator_
    // is device-only and let the session allocate them itself.
    bool is_string_output = tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    is_output_string_.push_back(is_string_output);
    // With plugin EP IO binding, Run() binds fresh outputs and never uses outputs_.
    if (is_dynamic || use_plugin_ep_io_binding_ ||
        (device_memory_name_.empty() && !plugin_ep_allocator_selection_.has_value()) ||
        (is_string_output && IsAllocatorDeviceOnly())) {
      outputs_.emplace_back(Ort::Value(nullptr));
    } else {
      auto new_value = Ort::Value::CreateTensor(allocator_, output_shape.data(), output_shape.size(), tensor_info.GetElementType());
      outputs_.emplace_back(std::move(new_value));
    }
  }
}

void OnnxRuntimeTestSession::PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) {
  StoreTestData(test_data_id, input_id, StageInputForPluginEpAllocator(std::move(value)));
}

void OnnxRuntimeTestSession::StoreTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) {
  if (test_inputs_.size() < test_data_id + 1) {
    test_inputs_.resize(test_data_id + 1);
  }
  if (test_inputs_[test_data_id].size() == 0) {
    for (int i = 0; i < input_length_; i++)
      test_inputs_[test_data_id].emplace_back(nullptr);
  }
  test_inputs_[test_data_id][input_id] = std::move(value);
}

bool OnnxRuntimeTestSession::IsAllocatorDeviceOnly() const {
  if (plugin_ep_allocator_selection_.has_value()) {
    return !plugin_ep_allocator_selection_->is_host_accessible;
  }
#if defined(USE_CUDA) || defined(USE_TENSORRT)
  if (device_memory_name_ == CUDA) {
    return true;
  }
#endif
  return false;
}

Ort::Value OnnxRuntimeTestSession::StageInputForPluginEpAllocator(Ort::Value&& value) {
  if (!plugin_ep_allocator_selection_.has_value() || !value.IsTensor()) {
    return std::move(value);
  }

  Ort::TensorTypeAndShapeInfo tensor_info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType element_type = tensor_info.GetElementType();
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    // Strings require host-accessible memory and always live on the CPU regardless of EP.
    return std::move(value);
  }

  std::vector<int64_t> shape = tensor_info.GetShape();
  Ort::Value device_value = Ort::Value::CreateTensor(allocator_, shape.data(), shape.size(), element_type);
  Ort::Status copy_status = env_.CopyTensor(value, device_value, nullptr);
  if (!copy_status.IsOK()) {
    ORT_THROW("Failed to copy loaded input tensor to plugin EP allocator memory: ", copy_status.GetErrorMessage());
  }
  return device_value;
}

template <typename T>
static void FillTensorDataTyped(Ort::Value& tensor, size_t count, int32_t seed = -1, T value = T{}) {
  T* data = tensor.GetTensorMutableData<T>();

  bool random_init = false;

  if (seed >= 0) {
    random_init = true;

    std::default_random_engine engine;
    engine.seed(seed);
    if constexpr (std::is_same<T, float>::value) {
      T max_value = 5.0f;
      const std::uniform_real_distribution<float>::param_type p(0, static_cast<float>(max_value));
      std::uniform_real_distribution<T> dist;
      for (size_t i = 0; i < count; ++i) {
        data[i] = dist(engine, p);
      }
    } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
      T max_value = std::numeric_limits<T>::max();
      const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(max_value));
      std::uniform_int_distribution<int> dist;
      for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(dist(engine, p));
      }
    } else {
      random_init = false;
      fprintf(stdout, " this type of data won't be random initialized\n");
    }
  }
  if (!random_init) {
    std::fill_n(data, count, value);
  }
}

// seed=-1 means we keep the initialized it with a constant value "T{}"
// in some case, we want to check the results for multi-runs, with the given we can recap the input data
// another reason is that, the input would be always 255/-127 for uint8_t or int8_t types of input.
// which will produce all zero outputs.
static void InitializeTensorWithSeed(int32_t seed, Ort::Value& tensor) {
  const auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
  const auto count = type_and_shape.GetElementCount();
  const auto element_type = type_and_shape.GetElementType();

#define CASE_FOR_TYPE(T)                         \
  case Ort::TypeToTensorType<T>::type: {         \
    FillTensorDataTyped<T>(tensor, count, seed); \
  } break

  switch (element_type) {
    CASE_FOR_TYPE(Ort::Float16_t);
    CASE_FOR_TYPE(Ort::BFloat16_t);
    CASE_FOR_TYPE(float);
    CASE_FOR_TYPE(double);
    CASE_FOR_TYPE(int8_t);
    CASE_FOR_TYPE(int16_t);
    CASE_FOR_TYPE(int32_t);
    CASE_FOR_TYPE(int64_t);
    CASE_FOR_TYPE(uint8_t);
    CASE_FOR_TYPE(uint16_t);
    CASE_FOR_TYPE(uint32_t);
    CASE_FOR_TYPE(uint64_t);
    CASE_FOR_TYPE(bool);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_FOR_TYPE(Ort::Float8E4M3FN_t);
    CASE_FOR_TYPE(Ort::Float8E4M3FNUZ_t);
    CASE_FOR_TYPE(Ort::Float8E5M2_t);
    CASE_FOR_TYPE(Ort::Float8E5M2FNUZ_t);
#endif
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      // string tensors are already initialized to contain empty strings
      // see onnxruntime::Tensor::Init()
      break;
    default:
      ORT_THROW("Unsupported tensor data type: ", element_type);
  }

#undef CASE_FOR_TYPE
}

void OnnxRuntimeTestSession::CreateAndStoreGeneratedInput(size_t test_data_id, size_t input_idx,
                                                          const std::vector<int64_t>& dims,
                                                          ONNXTensorElementDataType element_type, int32_t seed) {
#if defined(USE_CUDA) || defined(USE_TENSORRT)
  if (device_memory_name_ == CUDA) {
    Ort::AllocatorWithDefaultOptions default_allocator;
    Ort::Value default_tensor = Ort::Value::CreateTensor(default_allocator, dims.data(),
                                                         dims.size(), element_type);
    InitializeTensorWithSeed(seed, default_tensor);

    const void* default_ptr = default_tensor.GetTensorRawData();
    size_t total_bytes = default_tensor.GetTensorSizeInBytes();

    Ort::Value cuda_tensor = Ort::Value::CreateTensor(allocator_, dims.data(),
                                                      dims.size(), element_type);
    void* cuda_ptr = cuda_tensor.GetTensorMutableData<void>();

    cudaError_t cuda_err = cudaMemcpy(cuda_ptr, default_ptr, total_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
      ORT_THROW("Failed to copy tensor data from CPU to CUDA device. CUDA Error: ", cudaGetErrorString(cuda_err));
    }
    StoreTestData(test_data_id, input_idx, std::move(cuda_tensor));
    return;
  }
#endif

  if (IsAllocatorDeviceOnly()) {
    // allocator_ is device-only memory; fill a CPU tensor and stage it via StageInputForPluginEpAllocator.
    Ort::AllocatorWithDefaultOptions default_allocator;
    Ort::Value cpu_tensor = Ort::Value::CreateTensor(default_allocator, dims.data(),
                                                     dims.size(), element_type);
    InitializeTensorWithSeed(seed, cpu_tensor);
    StoreTestData(test_data_id, input_idx, StageInputForPluginEpAllocator(std::move(cpu_tensor)));
    return;
  }

  Ort::Value input_tensor = Ort::Value::CreateTensor(allocator_, dims.data(),
                                                     dims.size(), element_type);
  InitializeTensorWithSeed(seed, input_tensor);
  StoreTestData(test_data_id, input_idx, std::move(input_tensor));
}

bool OnnxRuntimeTestSession::PopulateGeneratedInputTestData(int32_t seed) {
  for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      continue;
    }

    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dim = tensor_info.GetShape();

    // free dimensions are treated as 1 if not overridden
    auto transform_fcn = [](int64_t input) { return (input == -1) ? -input : input; };
    std::transform(input_node_dim.begin(), input_node_dim.end(), input_node_dim.begin(), transform_fcn);

    CreateAndStoreGeneratedInput(0, i, input_node_dim, tensor_info.GetElementType(), seed);
  }
  return true;
}

bool OnnxRuntimeTestSession::PopulateGeneratedMultiShapeInputTestData(
    int32_t seed,
    const std::map<std::string, std::vector<std::vector<int64_t>>>& data_shape_groups) {
  // Validate that all input names in data_shape_groups exist in the model
  for (const auto& [name, groups] : data_shape_groups) {
    bool found = false;
    for (int i = 0; i < input_length_; i++) {
      if (input_names_str_[i] == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "Error: --data_shape specifies unknown input '" << name << "'." << std::endl;
      return false;
    }
  }

  const size_t num_groups = data_shape_groups.begin()->second.size();

  for (size_t g = 0; g < num_groups; g++) {
    for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
      Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        continue;
      }

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_node_dim;

      // Use user-specified shape if available, otherwise fall back to model metadata
      auto it = data_shape_groups.find(input_names_str_[i]);
      if (it != data_shape_groups.end()) {
        input_node_dim = it->second[g];
        const auto model_shape = tensor_info.GetShape();
        if (!model_shape.empty() && input_node_dim.size() != model_shape.size()) {
          std::cerr << "Error: --data_shape rank mismatch for input '" << input_names_str_[i]
                    << "': expected " << model_shape.size() << " dims but got " << input_node_dim.size() << "." << std::endl;
          return false;
        }
      } else {
        input_node_dim = tensor_info.GetShape();
        bool has_dynamic_dim = std::any_of(input_node_dim.begin(), input_node_dim.end(),
                                           [](int64_t d) { return d == -1; });
        auto transform_fcn = [](int64_t input) { return (input == -1) ? -input : input; };
        std::transform(input_node_dim.begin(), input_node_dim.end(), input_node_dim.begin(), transform_fcn);
        if (g == 0 && has_dynamic_dim) {
          std::cerr << "Warning: input '" << input_names_str_[i]
                    << "' not specified in --data_shape; using inferred shape [";
          for (size_t d = 0; d < input_node_dim.size(); d++) {
            if (d > 0) std::cerr << ",";
            std::cerr << input_node_dim[d];
          }
          std::cerr << "] (dynamic dims defaulted to 1)." << std::endl;
        }
      }

      CreateAndStoreGeneratedInput(g, i, input_node_dim, tensor_info.GetElementType(), seed);
    }
  }
  use_round_robin_ = true;
  return true;
}

std::vector<int64_t> OnnxRuntimeTestSession::GetLoadedInputShape(size_t test_data_id, size_t input_id) const {
  const auto& v = test_inputs_.at(test_data_id).at(input_id);
  if (!v.IsTensor()) {
    ORT_THROW("--data_shape only supports tensor inputs; input_id=", input_id, " in test_data_id=", test_data_id,
              " is not a tensor.");
  }
  return v.GetTensorTypeAndShapeInfo().GetShape();
}

void OnnxRuntimeTestSession::SelectTestDataSets(const std::vector<size_t>& selected_ids) {
  std::vector<std::vector<Ort::Value>> filtered;
  filtered.reserve(selected_ids.size());
  for (size_t id : selected_ids) {
    filtered.push_back(std::move(test_inputs_.at(id)));
  }
  test_inputs_ = std::move(filtered);
}

OnnxRuntimeTestSession::~OnnxRuntimeTestSession() {
#ifdef USE_CUDA
  if (device_memory_name_ == CUDA && stream_ != nullptr) {
    // Need to synchronize here before the custom allocator is destroyedif (cudaStreamSynchronize(stream_);
    if (cudaStreamSynchronize(stream_) != cudaError_t::cudaSuccess) {
      std::cerr << "Unable to sync CUDA stream";
    }
    if (cudaStreamDestroy(stream_) != cudaError_t::cudaSuccess) {
      std::cerr << "Unable to destroy CUDA stream";
    }
  }
#endif
}

}  // namespace perftest
}  // namespace onnxruntime
