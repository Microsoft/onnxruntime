// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"
#include <cstring>
#include <cassert>
#include "core/session/inference_session.h"
#include "abi_session_options_impl.h"

OrtSessionOptions::~OrtSessionOptions() = default;

OrtSessionOptions& OrtSessionOptions::operator=(const OrtSessionOptions&) {
  ORT_THROW("not implemented");
}
OrtSessionOptions::OrtSessionOptions(const OrtSessionOptions& other)
    : value(other.value), provider_factories(other.provider_factories) {
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionOptions, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtSessionOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseSessionOptions, _Frees_ptr_opt_ OrtSessionOptions* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  *out = new OrtSessionOptions(*input);
  return nullptr;
  API_IMPL_END
}

// Set execution_mode.
ORT_API_STATUS_IMPL(OrtApis::SetSessionExecutionMode, _In_ OrtSessionOptions* options,
                    ExecutionMode execution_mode) {
  switch (execution_mode) {
    case ORT_SEQUENTIAL:
    case ORT_PARALLEL:
      options->value.execution_mode = execution_mode;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "execution_mode is not valid");
  }

  return nullptr;
}

// set filepath to save optimized onnx model.
ORT_API_STATUS_IMPL(OrtApis::SetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath) {
  options->value.optimized_model_filepath = optimized_model_filepath;
  return nullptr;
}

// enable profiling for this session.
ORT_API_STATUS_IMPL(OrtApis::EnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix) {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtApis::DisableProfiling, _In_ OrtSessionOptions* options) {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
  return nullptr;
}

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ORT_API_STATUS_IMPL(OrtApis::EnableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = true;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtApis::DisableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = false;
  return nullptr;
}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API_STATUS_IMPL(OrtApis::EnableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::DisableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = false;
  return nullptr;
}

///< logger id to use for session output
ORT_API_STATUS_IMPL(OrtApis::SetSessionLogId, _In_ OrtSessionOptions* options, const char* logid) {
  options->value.session_logid = logid;
  return nullptr;
}

///< applies to session load, initialization, etc
ORT_API_STATUS_IMPL(OrtApis::SetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level) {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetSessionLogSeverityLevel, _In_ OrtSessionOptions* options, int session_log_severity_level) {
  options->value.session_log_severity_level = session_log_severity_level;
  return nullptr;
}

// Set Graph optimization level.
ORT_API_STATUS_IMPL(OrtApis::SetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options,
                    GraphOptimizationLevel graph_optimization_level) {
  if (graph_optimization_level < 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  }

  switch (graph_optimization_level) {
    case ORT_DISABLE_ALL:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Default;
      break;
    case ORT_ENABLE_BASIC:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level1;
      break;
    case ORT_ENABLE_EXTENDED:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level2;
      break;
    case ORT_ENABLE_ALL:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::MaxLevel;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  }

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads) {
#ifdef _OPENMP
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(intra_op_num_threads);
  // Can't use the default logger here since it's possible that the default logger has not been created
  // at this point. The default logger gets created when the env is created and these APIs don't require
  // the env to be created first.
  std::cout << "WARNING: Since openmp is enabled in this build, this API cannot be used to configure"
               " intra op num threads. Please use the openmp environment variables to control"
               " the number of threads.\n";
#else
  options->value.intra_op_param.thread_pool_size = intra_op_num_threads;
#endif
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads) {
  options->value.inter_op_param.thread_pool_size = inter_op_num_threads;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options,
                    _In_ const char* dim_denotation, _In_ int64_t dim_value) {
  options->value.free_dimension_overrides.push_back(
      onnxruntime::FreeDimensionOverride{dim_denotation, onnxruntime::FreeDimensionOverrideType::Denotation, dim_value});
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddFreeDimensionOverrideByName, _Inout_ OrtSessionOptions* options,
                    _In_ const char* dim_name, _In_ int64_t dim_value) {
  options->value.free_dimension_overrides.push_back(
      onnxruntime::FreeDimensionOverride{dim_name, onnxruntime::FreeDimensionOverrideType::Name, dim_value});
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::DisablePerSessionThreads, _In_ OrtSessionOptions* options) {
  options->value.use_per_session_threads = false;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddSessionConfigEntry, _Inout_ OrtSessionOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value) {
  return onnxruntime::ToOrtStatus(options->value.config_options.AddConfigEntry(config_key, config_value));
}

ORT_API_STATUS_IMPL(OrtApis::AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name,
                    _In_ const OrtValue* val) {
  auto st = options->value.AddInitializer(name, val);
  if (!st.IsOK()) {
    return onnxruntime::ToOrtStatus(st);
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetSessionOnnxOpsetVersion, _In_ OrtSessionOptions* options, int session_onnx_opset_version) {
  options->value.session_onnx_opset_version = session_onnx_opset_version;
  return nullptr;
}