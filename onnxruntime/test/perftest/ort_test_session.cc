#include "ort_test_session.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_session_options_config_keys.h"
#include <assert.h>
#include "providers.h"
#include "TestCase.h"

#ifdef _WIN32
#define strdup _strdup
#endif
extern const OrtApi* g_ort;

namespace onnxruntime {
namespace perftest {

std::chrono::duration<double> OnnxRuntimeTestSession::Run() {
  //Randomly pick one OrtValueArray from test_inputs_. (NOT ThreadSafe)
  const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(test_inputs_.size() - 1));
  const size_t id = static_cast<size_t>(dist_(rand_engine_, p));
  auto& input = test_inputs_.at(id);
  auto start = std::chrono::high_resolution_clock::now();
  auto output_values = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input.data(), input_names_.size(),
                                    output_names_raw_ptr.data(), output_names_raw_ptr.size());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - start;
  return duration_seconds;
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd,
                                               const PerformanceTestConfig& performance_test_config,
                                               const TestModelInfo& m)
    : rand_engine_(rd()), input_names_(m.GetInputCount()), input_length_(m.GetInputCount()) {
  Ort::SessionOptions session_options;
  const std::string& provider_name = performance_test_config.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kDnnlExecutionProvider) {
#ifdef USE_DNNL
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options,
                                                      performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("DNNL is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options{
        0,
        static_cast<OrtCudnnConvAlgoSearch>(performance_test_config.run_config.cudnn_conv_algo),
        std::numeric_limits<size_t>::max(),
        0,
        !performance_test_config.run_config.do_cuda_copy_in_separate_stream,
        0,
        nullptr};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options, /*allow_unaligned_buffers*/ 1, ""));
#else
    ORT_THROW("Nuphar is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    bool has_trt_options = false;
    size_t trt_max_workspace_size = 1 << 30;
    bool trt_fp16_enable = false;
    bool trt_int8_enable = false;
    std::string trt_int8_calibration_table_name = "";
    bool trt_int8_use_native_calibration_table = false;
    int trt_max_partition_iterations = 1000;
    int trt_min_subgraph_size = 1;
    bool trt_dump_subgraphs = false;
    bool trt_engine_cache_enable = false;
    std::string trt_cache_path = "";

    #ifdef _MSC_VER
    std::string ov_string = ToMBString(performance_test_config.run_config.ep_runtime_config_string);
    #else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
    #endif
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if(token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW("[ERROR] [TensorRT] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      auto key = token.substr(0,pos);
      auto value = token.substr(pos+1);
      if (key == "has_trt_options") {
        if(value == "true" || value == "True"){
          has_trt_options = true;
        } else if (value == "false" || value == "False") {
          has_trt_options = false;
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'has_trt_options' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "trt_max_workspace_size") {
        if(!value.empty()) {
          trt_max_workspace_size = std::stoull(value);
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_max_workspace_size' should be a number.\n");
        }
      } else if (key == "trt_fp16_enable") {
        if(value == "true" || value == "True"){
          trt_fp16_enable = true;
        } else if (value == "false" || value == "False") {
          trt_fp16_enable = false;
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_fp16_enable' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "trt_int8_enable") {
        if(value == "true" || value == "True"){
          trt_int8_enable = true;
        } else if (value == "false" || value == "False") {
          trt_int8_enable = false;
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_enable' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "trt_int8_calibration_table_name") {
        if(!value.empty()) {
          trt_int8_calibration_table_name = value;
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_calibration_table_name' should be a non-emtpy string.\n");
        }
      } else if (key == "trt_int8_use_native_calibration_table") {
        if(value == "true" || value == "True"){
          trt_int8_use_native_calibration_table = true;
        } else if (value == "false" || value == "False") {
          trt_int8_use_native_calibration_table = false;
        } else {
          ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_use_native_calibration_table' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else {
          ORT_THROW("[ERROR] [TensorRT] wrong key type entered. Choose from the following runtime key options that are available for TensorRT. ['use_trt_options', 'trt_fp16_enable', 'trt_int8_enable', 'trt_int8_calibration_table_name', 'trt_int8_use_native_calibration_table'] \n");
      }
    }
    OrtTensorRTProviderOptions tensorrt_options;
    tensorrt_options.device_id = 0;
    tensorrt_options.has_user_compute_stream = 0;
    tensorrt_options.user_compute_stream = nullptr;
    tensorrt_options.has_trt_options = has_trt_options;
    tensorrt_options.trt_max_workspace_size = trt_max_workspace_size;
    tensorrt_options.trt_fp16_enable = trt_fp16_enable;
    tensorrt_options.trt_int8_enable = trt_int8_enable;
    tensorrt_options.trt_int8_calibration_table_name = trt_int8_calibration_table_name.c_str();
    tensorrt_options.trt_int8_use_native_calibration_table = trt_int8_use_native_calibration_table;
    tensorrt_options.trt_max_partition_iterations = trt_max_partition_iterations;
    tensorrt_options.trt_min_subgraph_size = trt_min_subgraph_size;
    tensorrt_options.trt_dump_subgraphs = trt_dump_subgraphs;
    tensorrt_options.trt_engine_cache_enable = trt_engine_cache_enable;
    tensorrt_options.trt_cache_path = trt_cache_path.c_str();
    session_options.AppendExecutionProvider_TensorRT(tensorrt_options);

    OrtCUDAProviderOptions cuda_options{
        0,
        static_cast<OrtCudnnConvAlgoSearch>(performance_test_config.run_config.cudnn_conv_algo),
        std::numeric_limits<size_t>::max(),
        0,
        !performance_test_config.run_config.do_cuda_copy_in_separate_stream,
        0,
        nullptr};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    std::string device_type = ""; // [device_type]: Overrides the accelerator hardware type and precision with these values at runtime.
    bool enable_vpu_fast_compile = false; // [enable_vpu_fast_compile]: Fast-compile may be optionally enabled to speeds up the model's compilation to VPU device specific format.
    std::string device_id = ""; // [device_id]: Selects a particular hardware device for inference.
    size_t num_of_threads = 8; // [num_of_threads]: Overrides the accelerator default value of number of threads with this value at runtime.
    bool use_compiled_network = false; // [use_compiled_network]: Can be enabled to directly import pre-compiled blobs if exists.
    std::string blob_dump_path = ""; // [blob_dump_path]: Explicitly specify the path where you would like to dump and load the blobs for the use_compiled_network(save/load blob) feature. This overrides the default path.

    #ifdef _MSC_VER
    std::string ov_string = ToMBString(performance_test_config.run_config.ep_runtime_config_string);
    #else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
    #endif
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if(token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW("[ERROR] [OpenVINO] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      auto key = token.substr(0,pos);
      auto value = token.substr(pos+1);

      if (key == "device_type") {
        std::set<std::string> ov_supported_device_types = {"CPU_FP32", "GPU_FP32", "GPU_FP16", "VAD-M_FP16", "MYRIAD_FP16", "VAD-F_FP32"};
        if (ov_supported_device_types.find(value) != ov_supported_device_types.end()) {
          device_type = value;
        }
        else {
          ORT_THROW("[ERROR] [OpenVINO] You have selcted wrong configuration value for the key 'device_type'. select from 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'VAD-M_FP16', 'MYRIAD_FP16', 'VAD-F_FP32' or from Hetero/Multi options available. \n");
        }
      } else if (key == "device_id") {
        device_id = value;
      } else if (key == "enable_vpu_fast_compile") {
        if(value == "true" || value == "True"){
          enable_vpu_fast_compile = true;
        } else if (value == "false" || value == "False") {
          enable_vpu_fast_compile = false;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'enable_vpu_fast_compile' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "use_compiled_network") {
        if(value == "true" || value == "True"){
          use_compiled_network = true;
        } else if (value == "false" || value == "False") {
          use_compiled_network = false;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'use_compiled_network' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "num_of_threads") {
        std::stringstream sstream(value);
        sstream >> num_of_threads;
        if ((int)num_of_threads <=0) {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'num_of_threads' should be greater than 0\n");
        }
      } else if (key == "blob_dump_path") {
        blob_dump_path = value;
      } else {
          ORT_THROW("[ERROR] [OpenVINO] wrong key type entered. Choose from the following runtime key options that are available for OpenVINO. ['device_type', 'device_id', 'enable_vpu_fast_compile', 'num_of_threads', 'use_compiled_network', 'blob_dump_path'] \n");
      }
    }
    OrtOpenVINOProviderOptions options;
    options.device_type = device_type.c_str(); //To set the device_type
    options.device_id = device_id.c_str(); // To set the device_id
    options.enable_vpu_fast_compile = enable_vpu_fast_compile; // To enable_vpu_fast_compile, default is false
    options.num_of_threads = num_of_threads; // To set number of free InferRequests, default is 8
    options.use_compiled_network = use_compiled_network; // To use_compiled_network, default is false
    options.blob_dump_path = blob_dump_path.c_str(); // sets the blob_dump_path, default is ""
    session_options.AppendExecutionProvider_OpenVINO(options);
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kCoreMLExecutionProvider) {
#ifdef USE_COREML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
#else
    ORT_THROW("COREML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kDmlExecutionProvider) {
#ifdef USE_DML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
#else
    ORT_THROW("DirectML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kAclExecutionProvider) {
#ifdef USE_ACL
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_ACL(session_options,
                                                     performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("Acl is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kArmNNExecutionProvider) {
#ifdef USE_ARMNN
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(session_options,
                                                                     performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("ArmNN is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kRocmExecutionProvider) {
#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options{
        0,
        0,
        std::numeric_limits<size_t>::max(),
        0};
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#else
    ORT_THROW("ROCM is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(session_options, 0));
#else
    ORT_THROW("MIGraphX is not supported in this build\n");
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
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

  if (performance_test_config.run_config.intra_op_num_threads > 0) {
    fprintf(stdout, "Setting intra_op_num_threads to %d\n", performance_test_config.run_config.intra_op_num_threads);
    session_options.SetIntraOpNumThreads(performance_test_config.run_config.intra_op_num_threads);
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL && performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    session_options.SetInterOpNumThreads(performance_test_config.run_config.inter_op_num_threads);
  }

  // Set optimization level.
  session_options.SetGraphOptimizationLevel(performance_test_config.run_config.optimization_level);
  if (!performance_test_config.run_config.profile_file.empty())
    session_options.EnableProfiling(performance_test_config.run_config.profile_file.c_str());
  if (!performance_test_config.run_config.optimized_model_path.empty())
    session_options.SetOptimizedModelFilePath(performance_test_config.run_config.optimized_model_path.c_str());
  if (performance_test_config.run_config.set_denormal_as_zero)
    session_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");
  if (!performance_test_config.run_config.free_dim_name_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_name_overrides) {
      if (g_ort->AddFreeDimensionOverrideByName(session_options, ToMBString(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverrideByName failed for named dimension: %s\n", ToMBString(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with name, %s, to %d\n", ToMBString(dim_override.first).c_str(), (int) dim_override.second);
      }
    }
  }
  if (!performance_test_config.run_config.free_dim_denotation_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_denotation_overrides) {
      if (g_ort->AddFreeDimensionOverride(session_options, ToMBString(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverride failed for dimension denotation: %s\n", ToMBString(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with denotation, %s, to %d\n", ToMBString(dim_override.first).c_str(), (int) dim_override.second);
      }
    }
  }

  session_ = Ort::Session(env, performance_test_config.model_info.model_file_path.c_str(), session_options);

  size_t output_count = session_.GetOutputCount();
  output_names_.resize(output_count);
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name = session_.GetOutputName(i, a);
    assert(output_name != nullptr);
    output_names_[i] = output_name;
    a.Free(output_name);
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }

  size_t input_count = static_cast<size_t>(m.GetInputCount());
  for (size_t i = 0; i != input_count; ++i) {
    input_names_[i] = strdup(m.GetInputName(i).c_str());
  }
}

bool OnnxRuntimeTestSession::PopulateGeneratedInputTestData() {
  // iterate over all input nodes
  for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_node_dim = tensor_info.GetShape();

      // free dimensions are treated as 1 if not overriden
      for (int64_t& dim : input_node_dim) {
        if (dim == -1) {
          dim = 1;
        }
      }
      // default allocator doesn't have to be freed by user
      auto allocator = static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions());
      Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, (const int64_t*)input_node_dim.data(),
                                                         input_node_dim.size(), tensor_info.GetElementType());
      PreLoadTestData(0, i, std::move(input_tensor));
    }
  }
  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
