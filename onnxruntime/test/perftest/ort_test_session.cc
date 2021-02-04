#include "ort_test_session.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_session_options_config_keys.h"
#include <assert.h>
#include "providers.h"
#include "TestCase.h"

#ifdef _WIN32
#define strdup _strdup
#endif

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
    const auto& api = Ort::GetApi();

    OrtCUDAProviderOptions* cuda_options;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
    std::unique_ptr<OrtCUDAProviderOptions, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);

    std::vector<const char*> cuda_options_keys{"device_id", "arena_extend_strategy", "cuda_mem_limit",
                                               "cudnn_conv_algo_search", "do_copy_in_default_stream"};
    std::vector<const char*> cuda_options_values;
    cuda_options_values.reserve(5);

    // device id
    cuda_options_values.push_back("0");

    // arena extend strategy
    cuda_options_values.push_back("kNextPowerOfTwo");

    // cuda mem limit
    auto size_t_numeric_limits_max_string = std::to_string(std::numeric_limits<size_t>::max());
    cuda_options_values.push_back(size_t_numeric_limits_max_string.c_str());

    // cudnn conv algo search
    switch (performance_test_config.run_config.cudnn_conv_algo) {
      case 0:
        cuda_options_values.push_back("EXHAUSTIVE");
        break;
      case 1:
        cuda_options_values.push_back("HEURISTIC");
        break;
      case 2:
        cuda_options_values.push_back("DEFAULT");
        break;
      default:
        ORT_THROW("Unsupported value for cudnn_conv_algo. only 0, 1, 2 are supported. Got: ", performance_test_config.run_config.cudnn_conv_algo);
    }

    // do copy in default stream
    if (!performance_test_config.run_config.do_cuda_copy_in_separate_stream)
      cuda_options_values.push_back("1");
    else
      cuda_options_values.push_back("0");

    session_options.AppendExecutionProvider_CUDA(*cuda_options);
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
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options, ""));
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
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

      // free dimensions are treated as 1
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
