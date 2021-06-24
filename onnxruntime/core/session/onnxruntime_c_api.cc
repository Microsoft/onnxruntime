// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/framework/allocator.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/environment.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/ort_mutex.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif

using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

// Return the OrtStatus if it indicates an error
#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status)                      \
      return _status;                 \
  } while (0)

// Convert internal onnxruntime::Status to OrtStatus and return if there's an error
#define ORT_API_RETURN_IF_STATUS_NOT_OK(expr) \
  do {                                        \
    auto _status = (expr);                    \
    if (!_status.IsOK())                      \
      return ToOrtStatus(_status);            \
  } while (0)

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel logging_level, _In_ const char* logid,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnv, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status, tp_options);
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                    OrtLoggingLevel logging_level, _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  *out = OrtEnv::GetInstance(lm_info, status, tp_options);
  return ToOrtStatus(status);
  API_IMPL_END
}

// enable platform telemetry
ORT_API_STATUS_IMPL(OrtApis::EnableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().EnableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::DisableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().DisableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len,
                                _Inout_ OrtAllocator* allocator, std::unique_ptr<Tensor>* out) {
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    shapes[i] = shape[i];
  }
  std::shared_ptr<IAllocator> alloc_ptr = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  *out = std::make_unique<Tensor>(ml_type, onnxruntime::TensorShape(shapes), alloc_ptr);
  return nullptr;
}

ORT_STATUS_PTR CreateTensorImplForSeq(MLDataType elem_type, const int64_t* shape, size_t shape_len, Tensor& out) {
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    shapes[i] = shape[i];
  }
  OrtAllocator* allocator;
  // TODO(pranav): what allocator should be used to create the tensor here?
  // for the sake of simplicity of the API using the default one here
  auto st = OrtApis::GetAllocatorWithDefaultOptions(&allocator);
  if (st) {
    return st;
  }
  std::shared_ptr<IAllocator> alloc_ptr = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  out = Tensor(elem_type, onnxruntime::TensorShape(shapes), alloc_ptr);
  return nullptr;
}

/**
 *
 * this function will create a copy of the allocator info
 */
ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                                void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    if (shape[i] < 0)
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
    elem_count *= static_cast<size_t>(shape[i]);
    shapes[i] = shape[i];
  }

  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArray(ml_type->Size(), elem_count, &size_to_allocate)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "size overflow");
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = std::make_unique<Tensor>(ml_type, onnxruntime::TensorShape(shapes), p_data, *info);
  return nullptr;
}

namespace c_api_internal {

template <class T>
inline ORT_STATUS_PTR CallCreateTensorImpl(const int64_t* shape, size_t shape_len, const OrtMemoryInfo* info,
                                           void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  auto ml_value = DataTypeImpl::GetType<T>();
  return CreateTensorImpl(ml_value, shape, shape_len, info, p_data, p_data_len, out);
}

template <class T>
inline ORT_STATUS_PTR CallCreateTensorImpl(const int64_t* shape, size_t shape_len, _Inout_ OrtAllocator* allocator,
                                           std::unique_ptr<Tensor>* out) {
  auto ml_type = DataTypeImpl::GetType<T>();
  return CreateTensorImpl(ml_type, shape, shape_len, allocator, out);
}

}  // namespace c_api_internal

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<float>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<std::string>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<bool>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<MLFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<BFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<double>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  auto value = std::make_unique<OrtValue>();
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value->Init(tensor.release(),
              ml_tensor,
              ml_tensor->GetDeleteFunc());
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<float>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int8_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int16_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint32_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<int64_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<uint64_t>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<std::string>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<bool>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<MLFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<BFloat16>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ORT_API_RETURN_IF_ERROR(c_api_internal::CallCreateTensorImpl<double>(shape, shape_len, allocator, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  auto value = std::make_unique<OrtValue>();
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value->Init(tensor.release(),
              ml_tensor,
              ml_tensor->GetDeleteFunc());
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out) {
  API_IMPL_BEGIN
  auto custom_op_domain = std::make_unique<OrtCustomOpDomain>();
  custom_op_domain->domain_ = domain;
  *out = custom_op_domain.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op) {
  API_IMPL_BEGIN
  custom_op_domain->custom_ops_.emplace_back(op);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddCustomOpDomain, _Inout_ OrtSessionOptions* options,
                    _In_ OrtCustomOpDomain* custom_op_domain) {
  API_IMPL_BEGIN
  options->custom_op_domains_.emplace_back(custom_op_domain);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle) {
  API_IMPL_BEGIN

  Env::Default().LoadDynamicLibrary(library_path, false, library_handle);
  if (!*library_handle)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

  OrtStatus*(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

  Env::Default().GetSymbolFromLibrary(*library_handle, "RegisterCustomOps", (void**)&RegisterCustomOps);
  if (!RegisterCustomOps)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

  return RegisterCustomOps(options, OrtGetApiBase());
  API_IMPL_END
}

namespace {
// provider either model_path, or modal_data + model_data_length.
static ORT_STATUS_PTR CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                                _In_ const OrtEnv* env,
                                                _In_opt_z_ const ORTCHAR_T* model_path,
                                                _In_opt_ const void* model_data,
                                                size_t model_data_length,

                                                std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  // quick check here to decide load path. InferenceSession will provide error message for invalid values.
  // TODO: Could move to a helper
  const Env& os_env = Env::Default();  // OS environment (!= ORT environment)
  bool load_config_from_model =
      os_env.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar) == "1";

  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    if (model_path != nullptr) {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());
  }
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Add custom domains
  if (options && !options->custom_op_domains_.empty()) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddCustomOpDomains(options->custom_op_domains_));
  }
#endif

  // Finish load
  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load());
#endif
  } else {
    if (model_path != nullptr) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_path));
    } else {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_data, static_cast<int>(model_data_length)));
    }
  }

  return nullptr;
}

static ORT_STATUS_PTR InitializeSession(_In_ const OrtSessionOptions* options,
                                        _In_ std::unique_ptr<::onnxruntime::InferenceSession>& sess,
                                        _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr) {
  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      provider_list.push_back(std::move(provider));
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Initialize());

  return nullptr;
}

}  // namespace

ORT_API_STATUS_IMPL(OrtApis::CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data,
                    size_t model_data_length, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data, model_data_length, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  const int queue_id = 0;

  std::vector<std::string> feed_names(input_len);
  std::vector<OrtValue> feeds(input_len);

  for (size_t i = 0; i != input_len; ++i) {
    if (input_names[i] == nullptr || input_names[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input name cannot be empty");
    }

    feed_names[i] = input_names[i];
    auto& ort_value = feeds[i] = *reinterpret_cast<const ::OrtValue*>(input[i]);

    if (ort_value.Fence()) ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
  }

  // Create output feed
  std::vector<std::string> output_names(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output_names1[i] == nullptr || output_names1[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "output name cannot be empty");
    }
    output_names[i] = output_names1[i];
  }

  std::vector<OrtValue> fetches(output_names_len);
  for (size_t i = 0; i != output_names_len; ++i) {
    if (output[i] != nullptr) {
      ::OrtValue& value = *(output[i]);
      if (value.Fence())
        value.Fence()->BeforeUsingAsOutput(onnxruntime::kCpuExecutionProvider, queue_id);
      fetches[i] = value;
    }
  }
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions op;
    status = session->Run(op, feed_names, feeds, output_names, &fetches, nullptr);
  } else {
    status = session->Run(*run_options, feed_names, feeds, output_names, &fetches, nullptr);
  }

  if (!status.IsOK())
    return ToOrtStatus(status);
  for (size_t i = 0; i != output_names_len; ++i) {
    ::OrtValue& value = fetches[i];
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (output[i] == nullptr) {
      output[i] = new OrtValue(value);
    }
  }
  return nullptr;
  API_IMPL_END
}

struct OrtIoBinding {
  std::unique_ptr<::onnxruntime::IOBinding> binding_;
  explicit OrtIoBinding(std::unique_ptr<::onnxruntime::IOBinding>&& binding) : binding_(std::move(binding)) {}
  OrtIoBinding(const OrtIoBinding&) = delete;
  OrtIoBinding& operator=(const OrtIoBinding&) = delete;
};

ORT_API_STATUS_IMPL(OrtApis::RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options,
                    _In_ const OrtIoBinding* binding_ptr) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  auto status = session->Run(*run_options, *binding_ptr->binding_);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::unique_ptr<::onnxruntime::IOBinding> binding;
  auto status = session->NewIOBinding(&binding);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  *out = new OrtIoBinding(std::move(binding));
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseIoBinding, _Frees_ptr_opt_ OrtIoBinding* binding_ptr) {
  delete binding_ptr;
}

ORT_API_STATUS_IMPL(OrtApis::BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindInput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* mem_info_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, mem_info_ptr->device);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Out_ char** buffer, _Outptr_result_maybenull_ size_t** lengths, _Out_ size_t* count) {
  API_IMPL_BEGIN
  const auto& output_names = binding_ptr->binding_->GetOutputNames();
  if (output_names.empty()) {
    *buffer = nullptr;
    *lengths = nullptr;
    *count = 0U;
    return nullptr;
  }

  IAllocatorUniquePtr<size_t> lengths_alloc(reinterpret_cast<size_t*>(allocator->Alloc(allocator, output_names.size() * sizeof(size_t))),
                                            [allocator](size_t* p) { if(p) allocator->Free(allocator, p); });

  if (!lengths_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "lengths allocation failed");
  }

  size_t total_len = 0;
  auto* len_ptr = lengths_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    total_len += sz;
    *len_ptr++ = sz;
  }

  IAllocatorUniquePtr<char> buffer_alloc(reinterpret_cast<char*>(allocator->Alloc(allocator, total_len * sizeof(char))),
                                         [allocator](char* p) { if(p) allocator->Free(allocator, p); });

  if (!buffer_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "string buffer allocation failed");
  }

  char* buf_ptr = buffer_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    memcpy(buf_ptr, n.data(), sz);
    buf_ptr += sz;
  }

  *buffer = buffer_alloc.release();
  *lengths = lengths_alloc.release();
  *count = output_names.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Outptr_result_maybenull_ OrtValue*** output, _Out_ size_t* output_count) {
  API_IMPL_BEGIN
  const auto& outputs = binding_ptr->binding_->GetOutputs();
  if (outputs.empty()) {
    *output = nullptr;
    *output_count = 0U;
    return nullptr;
  }

  // Used to destroy and de-allocate on exception
  size_t created = 0;
  IAllocatorUniquePtr<OrtValue*> ortvalues_alloc(reinterpret_cast<OrtValue**>(allocator->Alloc(allocator, outputs.size() * sizeof(OrtValue*))),
                                                 [&created, allocator](OrtValue** buffer) {
                                                   if (buffer) {
                                                     while (created > 0) {
                                                       auto p = buffer + --created;
                                                       delete (*p);
                                                     }
                                                     allocator->Free(allocator, buffer);
                                                   }
                                                 });

  if (!ortvalues_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "Output buffer allocation failed");
  }

  OrtValue** out_ptr = ortvalues_alloc.get();
  for (const auto& out_value : outputs) {
    *out_ptr = new OrtValue(out_value);
    ++out_ptr;
    ++created;
  }

  assert(created == outputs.size());

  *output = ortvalues_alloc.release();
  *output_count = created;
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ClearBoundInputs, _Inout_ OrtIoBinding* binding_ptr) {
  binding_ptr->binding_->ClearInputs();
}

ORT_API(void, OrtApis::ClearBoundOutputs, _Inout_ OrtIoBinding* binding_ptr) {
  binding_ptr->binding_->ClearOutputs();
}

ORT_API_STATUS_IMPL(OrtApis::IsTensor, _In_ const OrtValue* value, _Out_ int* out) {
  auto v = reinterpret_cast<const ::OrtValue*>(value);
  *out = v->IsTensor() ? 1 : 0;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  //TODO: test if it's a string tensor
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len != len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array doesn't equal tensor size");
  }
  for (size_t i = 0; i != len; ++i) {
    //allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (index >= len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "element index is out of bounds");
  }

  dst[index] = s;

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* out) {
  TENSOR_READ_API_BEGIN
  const auto* src = tensor.Data<std::string>();
  int64_t len = tensor.Shape().Size();
  if (len >= 0) {
    size_t ret = 0;
    for (int64_t i = 0; i != len; ++i) {
      ret += src[i].size();
    }
    *out = ret;
  } else
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out) {
  TENSOR_READ_API_BEGIN
  const auto* src = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (index < len) {
    *out = src[index].size();
  } else
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                    size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (offsets_len != len) {
    return OrtApis::CreateStatus(ORT_FAIL, "offsets buffer is not equal to tensor size");
  }
  {
    size_t ret = 0;
    for (size_t i = 0; i != len; ++i) {
      ret += input[i].size();
    }
    if (s_len < ret) {
      return OrtApis::CreateStatus(ORT_FAIL, "output buffer is too small");
    }
  }
  size_t f = 0;
  char* p = static_cast<char*>(s);
  for (size_t i = 0; i != len; ++i, ++offsets) {
    memcpy(p, input[i].data(), input[i].size());
    p += input[i].size();
    *offsets = f;
    f += input[i].size();
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());

  if (index >= len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "element index is out of bounds");
  }

  size_t ret = input[index].size();
  if (s_len < ret) {
    return OrtApis::CreateStatus(ORT_FAIL, "buffer size is too small for string");
  }

  memcpy(s, input[index].data(), input[index].size());

  return nullptr;
  API_IMPL_END
}

#define ORT_C_API_RETURN_IF_ERROR(expr)                 \
  do {                                                  \
    auto _status = (expr);                              \
    if ((!_status.IsOK())) return ToOrtStatus(_status); \
  } while (0)

#define DEFINE_RELEASE_ORT_OBJECT_FUNCTION(INPUT_TYPE, REAL_TYPE)                       \
  ORT_API(void, OrtApis::Release##INPUT_TYPE, _Frees_ptr_opt_ Ort##INPUT_TYPE* value) { \
    delete reinterpret_cast<REAL_TYPE*>(value);                                         \
  }

using DefListResult = std::pair<Status, const InputDefList*>;
using GetDefListFn = DefListResult (*)(const ::onnxruntime::InferenceSession*);
const auto get_inputs_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetModelInputs(); };
const auto get_outputs_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetModelOutputs(); };
const auto get_overridable_initializers_fn = [](const ::onnxruntime::InferenceSession* session) -> DefListResult { return session->GetOverridableInitializers(); };

static ORT_STATUS_PTR GetNodeDefListCountHelper(const OrtSession* sess, GetDefListFn get_fn, size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = p.second->size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_inputs_fn, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_outputs_fn, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out) {
  return GetNodeDefListCountHelper(sess, get_overridable_initializers_fn, out);
}

static ORT_STATUS_PTR GetNodeDefTypeInfoHelper(const OrtSession* sess, GetDefListFn get_fn, size_t index,
                                               _Outptr_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second->size() <= index)
    return OrtApis::CreateStatus(ORT_FAIL, "out of index");
  const ONNX_NAMESPACE::TypeProto* type_proto = (*p.second)[index]->TypeAsProto();
  return OrtTypeInfo::FromTypeProto(type_proto, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_inputs_fn, index, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_outputs_fn, index, out);
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ struct OrtTypeInfo** out) {
  return GetNodeDefTypeInfoHelper(sess, get_overridable_initializers_fn, index, out);
}

static char* StrDup(const std::string& str, _Inout_ OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static ORT_STATUS_PTR GetNodeDefNameImpl(_In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                                         GetDefListFn get_fn, _Outptr_ char** output) {
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  std::pair<Status, const InputDefList*> p = get_fn(session);
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  if (p.second == nullptr)
    return OrtApis::CreateStatus(ORT_FAIL, "internal error");
  const InputDefList& defs = *p.second;
  if (index >= defs.size())
    return OrtApis::CreateStatus(ORT_FAIL, "index out of range");
  *output = StrDup(defs[index]->Name(), allocator);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  auto profile_file_name = session->EndProfiling();
  *out = StrDup(profile_file_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetModelMetadata, _In_ const OrtSession* sess,
                    _Outptr_ OrtModelMetadata** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto p = session->GetModelMetadata();
  if (!p.first.IsOK())
    return ToOrtStatus(p.first);
  *out = reinterpret_cast<OrtModelMetadata*>(new ModelMetadata(*p.second));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetProducerName,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto producer_name = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->producer_name;
  *value = StrDup(producer_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetGraphName,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto graph_name = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->graph_name;
  *value = StrDup(graph_name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetDomain,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto domain = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->domain;
  *value = StrDup(domain, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetDescription,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto description = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->description;
  *value = StrDup(description, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetGraphDescription,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value) {
  API_IMPL_BEGIN
  auto description = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->graph_description;
  *value = StrDup(description, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value) {
  API_IMPL_BEGIN
  auto custom_metadata_map =
      reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->custom_metadata_map;

  std::string temp(key);

  auto iter = custom_metadata_map.find(temp);

  if (iter == custom_metadata_map.end()) {
    *value = nullptr;
  } else {
    *value = StrDup(iter->second, allocator);
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetCustomMetadataMapKeys,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys) {
  API_IMPL_BEGIN
  const auto& custom_metadata_map =
      reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->custom_metadata_map;

  auto count = custom_metadata_map.size();
  if (count == 0) {
    *keys = nullptr;
  } else {
    // To guard against overflow in the next step where we compute bytes to allocate
    SafeInt<size_t> alloc_count(count);

    // alloc_count * sizeof(...) will throw if there was an overflow which will be caught in API_IMPL_END
    // and be returned to the user as a status
    char** p = reinterpret_cast<char**>(allocator->Alloc(allocator, alloc_count * sizeof(char*)));
    assert(p != nullptr);
    auto map_iter = custom_metadata_map.cbegin();
    int64_t i = 0;
    while (map_iter != custom_metadata_map.cend()) {
      p[i++] = StrDup(map_iter->first, allocator);
      ++map_iter;
    }
    *keys = p;
  }

  *num_keys = static_cast<int64_t>(count);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ModelMetadataGetVersion,
                    _In_ const OrtModelMetadata* model_metadata,
                    _Out_ int64_t* value) {
  API_IMPL_BEGIN
  *value = reinterpret_cast<const ::onnxruntime::ModelMetadata*>(model_metadata)->version;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetInputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_inputs_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOutputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_outputs_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  return GetNodeDefNameImpl(sess, index, allocator, get_overridable_initializers_fn, output);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out) {
  API_IMPL_BEGIN
  *out = ptr->Alloc(ptr, size);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorFree, _Inout_ OrtAllocator* ptr, void* p) {
  API_IMPL_BEGIN
  ptr->Free(ptr, p);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out) {
  API_IMPL_BEGIN
  *out = ptr->Info(ptr);
  return nullptr;
  API_IMPL_END
}

template <typename T>
ORT_STATUS_PTR OrtGetNumSequenceElements(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<T>();
  *out = data.size();
  return nullptr;
}

template <>
ORT_STATUS_PTR OrtGetNumSequenceElements<TensorSeq>(const OrtValue* p_ml_value, size_t* out) {
  auto& data = p_ml_value->Get<TensorSeq>();
  *out = data.Size();
  return nullptr;
}

#if !defined(DISABLE_ML_OPS)
static const int NUM_MAP_INDICES = 2;
#endif

static ORT_STATUS_PTR OrtGetValueCountImpl(const OrtValue* value, size_t* out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    *out = NUM_MAP_INDICES;
    return nullptr;
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    auto v = reinterpret_cast<const OrtValue*>(value);
    auto type = v->Type();
    // Note: keep these in sync with the registered types in data_types.h
    if (type->IsTensorSequenceType()) {
      return OrtGetNumSequenceElements<TensorSeq>(v, out);
    } else {
#if !defined(DISABLE_ML_OPS)
      utils::ContainerChecker c_checker(type);
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        return OrtGetNumSequenceElements<VectorMapStringToFloat>(v, out);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        return OrtGetNumSequenceElements<VectorMapInt64ToFloat>(v, out);
      } else {
        return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
      }
#else
      return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
    }
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out) {
  API_IMPL_BEGIN
  return OrtGetValueCountImpl(value, out);
  API_IMPL_END
}

#if !defined(DISABLE_ML_OPS)
///////////////////
// OrtGetValueImplSeqOfMap
template <typename T>
static ORT_STATUS_PTR OrtGetValueImplSeqOfMap(const OrtValue* p_ml_value, int index, _Outptr_ OrtValue** out) {
  using TKey = typename T::value_type::key_type;
  using TVal = typename T::value_type::mapped_type;
  using MapType = std::map<TKey, TVal>;
  auto& data_vec = p_ml_value->Get<T>();
  auto& data_elem = data_vec.at(index);
  auto copy_data_elem = std::make_unique<MapType>(data_elem);
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<MapType>();
  value->Init(copy_data_elem.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}
#endif

ORT_STATUS_PTR PopulateTensorWithData(_Inout_ OrtValue* oval, _In_ const void* data_elem, size_t num_elems,
                                      size_t elem_size) {
  void* raw_data = nullptr;
  auto st = OrtApis::GetTensorMutableData(oval, &raw_data);
  if (st) {
    return st;
  }
  memcpy(raw_data, data_elem, elem_size * num_elems);
  return nullptr;
}

ORT_STATUS_PTR PopulateTensorWithData(_Inout_ OrtValue* oval, _In_reads_(num_elems) const std::string* data_elem,
                                      size_t num_elems, size_t /* elem_size */) {
  auto v = reinterpret_cast<OrtValue*>(oval);
  auto tensor = v->GetMutable<Tensor>();
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (num_elems < len) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i < len; ++i) {
    dst[i] = data_elem[i];
  }
  return nullptr;
}

namespace c_api_internal {
template <class TensorElemType>
struct CallGetValueImpl {
  ORT_STATUS_PTR operator()(_Inout_ OrtAllocator* allocator, const onnxruntime::Tensor& tensor,
                            _Outptr_ OrtValue** out) const {
    const auto& shape = tensor.Shape();
    const auto* tensor_data = tensor.Data<TensorElemType>();
    OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, shape.GetDims().data(), shape.NumDimensions(),
                                                    onnxruntime::utils::GetONNXTensorElementDataType<TensorElemType>(), out);
    //TODO: check overflow before doing static_cast
    return st ? st : PopulateTensorWithData(*out, tensor_data, static_cast<size_t>(shape.Size()), sizeof(TensorElemType));
  }
};

// Return status instead of throwing if unsupported type specified
struct UnsupportedReturnFailStatus {
  void operator()(int32_t dt_type, OrtStatusPtr& status) const {
    std::string msg("Unsupported tensor element type in the input: ");
    msg.append(std::to_string(dt_type));
    status = OrtApis::CreateStatus(ORT_FAIL, msg.c_str());
  }
};
}  // namespace c_api_internal
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6101)
#endif
ORT_STATUS_PTR OrtGetValueImplSeqOfTensors(_In_ const OrtValue* p_ml_value, int index, _In_opt_ OrtAllocator* allocator,
                                           _Outptr_ OrtValue** out) {
  auto& data = p_ml_value->Get<TensorSeq>();
  auto& one_tensor = data.Get(index);

  using namespace c_api_internal;
  utils::MLTypeCallDispatcher<float, double, MLFloat16, BFloat16, bool, std::string,
                              int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
      t_disp(one_tensor.GetElementType());
  return t_disp.template InvokeRetWithUnsupportedPolicy<OrtStatusPtr, CallGetValueImpl, UnsupportedReturnFailStatus>(
      allocator, one_tensor, out);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static ORT_STATUS_PTR OrtGetValueImplSeq(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                         _Outptr_ OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  if (type->IsTensorSequenceType()) {
    return OrtGetValueImplSeqOfTensors(p_ml_value, index, allocator, out);
  } else {
#if !defined(DISABLE_ML_OPS)
    utils::ContainerChecker c_checker(type);
    if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
      return OrtGetValueImplSeqOfMap<VectorMapStringToFloat>(p_ml_value, index, out);
    } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
      return OrtGetValueImplSeqOfMap<VectorMapInt64ToFloat>(p_ml_value, index, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported sequence types.");
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
}

#if !defined(DISABLE_ML_OPS)
template <typename T>
static ORT_STATUS_PTR OrtGetValueImplMapHelper(_In_ const OrtValue* p_ml_value, int index,
                                               _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out) {
  using namespace onnxruntime::utils;
  using TKey = typename T::key_type;
  using TVal = typename T::mapped_type;
  auto& data = p_ml_value->Get<T>();
  int64_t num_kv_pairs = data.size();
#if defined(_WIN32) && !defined(_M_AMD64)
  ORT_ENFORCE(static_cast<uint64_t>(num_kv_pairs) < std::numeric_limits<size_t>::max());
#endif
  switch (index) {
    case 0: {  // user is requesting keys
      std::vector<TKey> vec;
      vec.reserve(static_cast<size_t>(num_kv_pairs));
      for (const auto& kv : data) {
        vec.push_back(kv.first);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TKey>(), out);
      return st ? st : PopulateTensorWithData(*out, vec.data(), static_cast<size_t>(num_kv_pairs), sizeof(TKey));
    }
    case 1: {  // user is requesting values
      std::vector<TVal> vec;
      vec.reserve(static_cast<size_t>(num_kv_pairs));
      for (const auto& kv : data) {
        vec.push_back(kv.second);
      }
      std::vector<int64_t> dims{num_kv_pairs};
      OrtStatus* st = OrtApis::CreateTensorAsOrtValue(allocator, dims.data(), dims.size(),
                                                      GetONNXTensorElementDataType<TVal>(), out);
      return st ? st : PopulateTensorWithData(*out, vec.data(), static_cast<size_t>(num_kv_pairs), sizeof(TVal));
    }
    default:
      return OrtApis::CreateStatus(ORT_FAIL, "Invalid index requested for map type.");
  }
}

static ORT_STATUS_PTR OrtGetValueImplMap(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                         _Outptr_ OrtValue** out) {
  auto p_ml_value = reinterpret_cast<const OrtValue*>(value);
  auto type = p_ml_value->Type();
  // Note: keep these in sync with the registered types in data_types.h
  utils::ContainerChecker c_checker(type);
  if (c_checker.IsMap()) {
    if (c_checker.IsMapOf<std::string, std::string>()) {
      return OrtGetValueImplMapHelper<MapStringToString>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, int64_t>()) {
      return OrtGetValueImplMapHelper<MapStringToInt64>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, float>()) {
      return OrtGetValueImplMapHelper<MapStringToFloat>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<std::string, double>()) {
      return OrtGetValueImplMapHelper<MapStringToDouble>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, std::string>()) {
      return OrtGetValueImplMapHelper<MapInt64ToString>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, int64_t>()) {
      return OrtGetValueImplMapHelper<MapInt64ToInt64>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, float>()) {
      return OrtGetValueImplMapHelper<MapInt64ToFloat>(p_ml_value, index, allocator, out);
    } else if (c_checker.IsMapOf<int64_t, double>()) {
      return OrtGetValueImplMapHelper<MapInt64ToDouble>(p_ml_value, index, allocator, out);
    }
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
}
#endif

static ORT_STATUS_PTR OrtGetValueImpl(_In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                                      _Outptr_ OrtValue** out) {
  ONNXType value_type;
  if (auto status = OrtApis::GetValueType(value, &value_type))
    return status;
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    return OrtGetValueImplMap(value, index, allocator, out);
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtGetValueImplSeq(value, index, allocator, out);
  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  return OrtGetValueImpl(value, index, allocator, out);
  API_IMPL_END
}

///////////////////
// OrtCreateValue

#if !defined(DISABLE_ML_OPS)
template <typename T>
static OrtStatus* OrtCreateValueImplSeqHelperMap(const OrtValue* const* in, size_t num_values,
                                                 _Outptr_ OrtValue** out) {
  using SeqType = std::vector<T>;
  auto seq_ptr = std::make_unique<SeqType>();
  seq_ptr->reserve(num_values);
  for (size_t idx = 0; idx < num_values; ++idx) {
    auto& m = reinterpret_cast<const OrtValue*>(in[idx])->Get<T>();
    seq_ptr->push_back(m);
  }
  // create OrtValue with this vector
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<SeqType>();
  value->Init(seq_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}
#endif

template <typename TensorElemType>
static OrtStatus* OrtCreateValueImplSeqHelperTensor(const Tensor& tensor,
                                                    Tensor& out) {
  auto data = tensor.Data<TensorElemType>();
  if (!data) {
    return OrtApis::CreateStatus(ORT_FAIL, "Encountered nullptr.");
  }

  auto elem_type = DataTypeImpl::GetType<TensorElemType>();
  OrtStatus* st = CreateTensorImplForSeq(elem_type, tensor.Shape().GetDims().data(), tensor.Shape().NumDimensions(), out);
  if (st) {
    return st;
  }

  //TODO: check the cast below
  size_t num_elems = static_cast<size_t>(tensor.Shape().Size());
  auto* out_data = out.MutableData<TensorElemType>();
  for (size_t i = 0; i < num_elems; ++i) {
    *out_data++ = *data++;
  }
  return nullptr;
}

namespace c_api_internal {

template <class T>
struct CallCreateValueImpl {
  OrtStatus* operator()(const onnxruntime::Tensor& one_tensor, onnxruntime::Tensor& out) const {
    return OrtCreateValueImplSeqHelperTensor<T>(one_tensor, out);
  }
};

}  // namespace c_api_internal

static ORT_STATUS_PTR OrtCreateValueImplSeqHelper(const OrtValue* const* in, size_t num_values,
                                                  _Outptr_ OrtValue** out) {
  using namespace c_api_internal;
  std::vector<Tensor> tensors;
  tensors.resize(num_values);
  auto dtype = static_cast<const OrtValue*>(in[0])->Get<Tensor>().DataType();

  for (size_t idx = 0; idx < num_values; ++idx) {
    ORT_ENFORCE(in[idx]->IsTensor(), "Expecting all elements to be tensors. Got: ", DataTypeImpl::ToString(in[idx]->Type()));
    auto& one_tensor = static_cast<const OrtValue*>(in[idx])->Get<Tensor>();
    auto tensor_elem_type = one_tensor.DataType();

    // sequences must have tensors of the same data type
    if (idx > 0 && (tensor_elem_type != dtype)) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "Sequences must have tensors of the same data type. There was at least one tensor in the input that was different.");
    }

    OrtStatus* st{};
    utils::MLTypeCallDispatcher<bool, float, double, std::string,
                                MLFloat16, BFloat16, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
        t_disp(one_tensor.GetElementType());

    st = t_disp.InvokeRetWithUnsupportedPolicy<OrtStatus*, CallCreateValueImpl, UnsupportedReturnFailStatus>(
        one_tensor, tensors[idx]);

    if (st) {
      return st;
    }
  }
  // create OrtValue with this vector
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<TensorSeq>();
  auto seq_ptr = std::make_unique<TensorSeq>(dtype);
  seq_ptr->SetElements(std::move(tensors));
  value->Init(seq_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

static ORT_STATUS_PTR OrtCreateValueImplSeq(_In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                                            _Outptr_ OrtValue** out) {
  // We only support limited sequence types. For the sake of simplicity the type of the first
  // OrtValue* in OrtValue** will determine the type of the vector used to create the output OrtValue
  // this type should be either a tensor of limited types or map of limited types
  const OrtValue* ovfirst = in[0];
  ONNXType first_value_type;
  if (auto status = OrtApis::GetValueType(ovfirst, &first_value_type))
    return status;
  // in onnxruntime type registrations we can support only a fixed vector types
  // this check ensures that the input conforms to that
  if (!(first_value_type == ONNX_TYPE_TENSOR || first_value_type == ONNX_TYPE_MAP)) {
    return OrtApis::CreateStatus(ORT_FAIL, "Each element of the sequence should be either tensor or map.");
  }
  // check if all OrtValues in the input array are of the same type
  // this is because even though the ONNX spec and this API spec supports heterogenous sequences,
  // only a fixed types are registered in onnxruntime
  for (size_t i = 0; i < num_values; ++i) {
    const OrtValue* ov = in[i];
    ONNXType ov_type;
    if (auto status = OrtApis::GetValueType(ov, &ov_type))
      return status;
    if (ov_type != first_value_type) {
      return OrtApis::CreateStatus(ORT_FAIL,
                                   "At least one element in the sequence is of a type different from others.");
    }
  }

  // finally create the output vector/MLValue
  auto first_mlvalue = reinterpret_cast<const OrtValue*>(ovfirst);
  if (first_value_type == ONNX_TYPE_TENSOR) {
    return OrtCreateValueImplSeqHelper(in, num_values, out);
  } else if (first_value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    auto map_type = first_mlvalue->Type();
    utils::ContainerChecker c_checker(map_type);
    if (c_checker.IsMapOf<std::string, float>()) {
      return OrtCreateValueImplSeqHelperMap<MapStringToFloat>(in, num_values, out);
    }
    if (c_checker.IsMapOf<int64_t, float>()) {
      return OrtCreateValueImplSeqHelperMap<MapInt64ToFloat>(in, num_values, out);
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "Input is not of one of the supported map types.");
    }
#else
    ORT_UNUSED_PARAMETER(first_mlvalue);
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif

  } else {
    return OrtApis::CreateStatus(ORT_FAIL, "Unsupported input type");
  }
}

#if !defined(DISABLE_ML_OPS)
template <typename KeyType, typename ValueType>
static OrtStatus* OrtCreateMapMLValue(const Tensor& key_tensor, const Tensor& value_tensor, _Outptr_ OrtValue** out) {
  using MapType = std::map<KeyType, ValueType>;
  auto map_ptr = std::make_unique<MapType>();
  // iterate through the key and value tensors and populate map
  auto key_data = key_tensor.Data<KeyType>();
  auto value_data = value_tensor.Data<ValueType>();
  auto len = key_tensor.Shape().Size();
  ORT_ENFORCE(len >= 0 && static_cast<uint64_t>(len) < std::numeric_limits<size_t>::max());
  size_t num_kv_pairs = static_cast<size_t>(key_tensor.Shape().Size());
  for (size_t n = 0; n < num_kv_pairs; ++n, ++key_data, ++value_data) {
    map_ptr->insert({*key_data, *value_data});
  }
  // create ort_value with this map
  auto value = std::make_unique<OrtValue>();
  auto ml_type = DataTypeImpl::GetType<MapType>();
  value->Init(map_ptr.release(),
              ml_type,
              ml_type->GetDeleteFunc());
  *out = value.release();
  return nullptr;
}

template <typename KeyType>
static ORT_STATUS_PTR OrtCreateValueImplMapHelper(const Tensor& key_tensor, const Tensor& value_tensor,
                                                  _Outptr_ OrtValue** out) {
  auto value_type = value_tensor.DataType()->AsPrimitiveDataType();
  ORT_ENFORCE(value_type != nullptr, "Tensor must always contain primitive types. Found: ",
              DataTypeImpl::ToString(value_tensor.DataType()));

  switch (value_type->GetDataType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return OrtCreateMapMLValue<KeyType, std::string>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return OrtCreateMapMLValue<KeyType, int64_t>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return OrtCreateMapMLValue<KeyType, float>(key_tensor, value_tensor, out);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return OrtCreateMapMLValue<KeyType, double>(key_tensor, value_tensor, out);
      break;
    default:
      break;
  }

  std::string msg("Value type is not supported yet: ");
  msg += DataTypeImpl::ToString(value_tensor.DataType());
  return OrtApis::CreateStatus(ORT_FAIL, msg.c_str());
}

static ORT_STATUS_PTR OrtCreateValueImplMap(const OrtValue* const* in, size_t num_values, _Outptr_ OrtValue** out) {
  if (num_values != NUM_MAP_INDICES) {
    return OrtApis::CreateStatus(ORT_FAIL, "For map type num_values MUST be 2");
  }

  const OrtValue* ort_keys = in[0];
  auto p_key_ml_value = reinterpret_cast<const OrtValue*>(ort_keys);
  auto& key_tensor = p_key_ml_value->Get<Tensor>();

  const OrtValue* ort_values = in[1];
  auto p_value_ml_value = reinterpret_cast<const OrtValue*>(ort_values);
  auto& value_tensor = p_value_ml_value->Get<Tensor>();

  // as per data_types.h, we only support maps of primitive data types.
  if (key_tensor.Shape().NumDimensions() > 1 || value_tensor.Shape().NumDimensions() > 1) {
    return OrtApis::CreateStatus(ORT_FAIL, "Either the key tensor or the value tensor has NumDimensions > 1");
  }

  // since maps are represented by key and value tensors, their sizes have to be the same.
  if (key_tensor.Shape().Size() != value_tensor.Shape().Size()) {
    return OrtApis::CreateStatus(ORT_FAIL, "Key and value tensors have unequal number of elements.");
  }

  if (key_tensor.IsDataTypeString()) {
    return OrtCreateValueImplMapHelper<std::string>(key_tensor, value_tensor, out);
  }
  if (key_tensor.IsDataType<int64_t>()) {
    return OrtCreateValueImplMapHelper<int64_t>(key_tensor, value_tensor, out);
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Key type is not supported yet.");
}
#endif

static ORT_STATUS_PTR OrtCreateValueImpl(_In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                                         enum ONNXType value_type, _Outptr_ OrtValue** out) {
  if (num_values <= 0) {
    return OrtApis::CreateStatus(ORT_FAIL, "Number of values should be at least 1.");
  }
  if (value_type == ONNX_TYPE_MAP) {
#if !defined(DISABLE_ML_OPS)
    return OrtCreateValueImplMap(in, num_values, out);
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Map type is not supported in this build.");
#endif
  }
  if (value_type == ONNX_TYPE_SEQUENCE) {
    return OrtCreateValueImplSeq(in, num_values, out);
  }
  return OrtApis::CreateStatus(ORT_FAIL, "Input is not of type sequence or map.");
}

ORT_API_STATUS_IMPL(OrtApis::CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                    enum ONNXType value_type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  return OrtCreateValueImpl(in, num_values, value_type, out);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                    _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  std::string dtype("opaque(");
  dtype.append(domain_name).append(",").append(type_name).append(")");
  MLDataType ml_type = DataTypeImpl::GetDataType(dtype);
  ORT_ENFORCE(ml_type != nullptr,
              "Specified domain and type names combination does not refer to a registered opaque type");
  const auto* non_tensor_base = ml_type->AsNonTensorTypeBase();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  std::unique_ptr<OrtValue> ort_val(new OrtValue);
  non_tensor_base->FromDataContainer(data_container, data_container_size, *ort_val);
  *out = ort_val.release();
  API_IMPL_END
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size) {
  API_IMPL_BEGIN
  std::string dtype("opaque(");
  dtype.append(domain_name).append(",").append(type_name).append(")");
  MLDataType ml_type = DataTypeImpl::GetDataType(dtype);
  ORT_ENFORCE(ml_type != nullptr,
              "Specified domain and type names combination does not refer to a registered opaque type");
  const auto* non_tensor_base = ml_type->AsNonTensorTypeBase();
  ORT_ENFORCE(non_tensor_base != nullptr, "Opaque type is not a non_tensor type!!!");
  non_tensor_base->ToDataContainer(*in, data_container_size, data_container);
  API_IMPL_END
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetAvailableProviders, _Outptr_ char*** out_ptr,
                    _In_ int* providers_length) {
  API_IMPL_BEGIN
  //TODO: there is no need to manually malloc/free these memory, it is insecure
  //and inefficient. Instead, the implementation could scan the array twice,
  //and use a single string object to hold all the names.
  const size_t MAX_LEN = 30;
  const auto& available_providers = GetAvailableExecutionProviderNames();
  const int available_count = gsl::narrow<int>(available_providers.size());
  char** const out = new char*[available_count];
  if (out) {
    for (int i = 0; i < available_count; i++) {
      out[i] = new char[MAX_LEN + 1];
#ifdef _MSC_VER
      strncpy_s(out[i], MAX_LEN, available_providers[i].c_str(), MAX_LEN);
      out[i][MAX_LEN] = '\0';
#elif defined(__APPLE__)
      strlcpy(out[i], available_providers[i].c_str(), MAX_LEN);
#else
      strncpy(out[i], available_providers[i].c_str(), MAX_LEN);
      out[i][MAX_LEN] = '\0';
#endif
    }
  }
  *providers_length = available_count;
  *out_ptr = out;
  API_IMPL_END
  return nullptr;
}

//TODO: we don't really need the second parameter
ORT_API_STATUS_IMPL(OrtApis::ReleaseAvailableProviders, _In_ char** ptr,
                    _In_ int providers_length) {
  API_IMPL_BEGIN
  if (ptr) {
    for (int i = 0; i < providers_length; i++) {
      delete[] ptr[i];
    }
    delete[] ptr;
  }
  API_IMPL_END
  return NULL;
}

ORT_API_STATUS_IMPL(OrtApis::TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count,
                    _Outptr_ void** out) {
  TENSOR_READWRITE_API_BEGIN

  if (tensor->IsDataTypeString()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "this API does not support strings");
  }

  const auto& tensor_shape = tensor->Shape();
  const auto num_dimensions = tensor_shape.NumDimensions();
  if (location_values_count != num_dimensions) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "location dimensions do not match shape size");
  }

  for (size_t i = 0; i < location_values_count; i++) {
    if (location_values[i] >= tensor_shape[i] || location_values[i] < 0) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "invalid location range");
    }
  }

  // compute strides
  // TensorPitches p;
  std::vector<int64_t> strides(num_dimensions);
  {
    int64_t stride = 1;
    for (size_t dim = num_dimensions; dim > 0; --dim) {
      strides[dim - 1] = stride;
      stride *= tensor_shape[dim - 1];
    }
  }

  // For Scalers the offset would always be zero
  int64_t offset = 0;
  for (size_t i = 0; i < num_dimensions; i++) {
    offset += location_values[i] * strides[i];
  }

  auto data = reinterpret_cast<char*>(tensor->MutableDataRaw()) + tensor->DataType()->Size() * offset;
  *out = data;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().SetLanguageProjection(static_cast<uint32_t>(projection));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionGetProfilingStartTimeNs, _In_ const OrtSession* sess, _Out_ uint64_t* out) {
  API_IMPL_BEGIN
  const auto* session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto profiling_start_time = session->GetProfiling().GetStartTimeNs();
  *out = static_cast<uint64_t>(profiling_start_time);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionAddONNXOpDomain, int session_onnx_opset_version) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  Status status = InferenceSession::ORTRegisterONNXOpsetSchema(session_onnx_opset_version);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
#else
  // Does not support selective ONNX opset loading for minimal build
  (void) session_onnx_opset_version; 
#endif
  return nullptr;
  API_IMPL_END
}

// End support for non-tensor types

#ifndef USE_ROCM
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(rocm_options);
  return CreateStatus(ORT_FAIL, "ROCM execution provider is not enabled.");
}
#endif

#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateStatus(ORT_FAIL, "OpenVINO execution provider is not enabled.");
}
#endif

#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled.");
}
#endif

ORT_API_STATUS_IMPL(OrtApis::CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
                    int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out) {
  API_IMPL_BEGIN
  *out = new OrtArenaCfg();
  (*out)->max_mem = max_mem;
  (*out)->arena_extend_strategy = arena_extend_strategy;
  (*out)->initial_chunk_size_bytes = initial_chunk_size_bytes;
  (*out)->max_dead_bytes_per_chunk = max_dead_bytes_per_chunk;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys, _In_reads_(num_keys) const size_t* arena_config_values,
                    _In_ size_t num_keys, _Outptr_ OrtArenaCfg** out) {
  API_IMPL_BEGIN
  auto cfg = std::make_unique<OrtArenaCfg>();

  for (size_t i = 0; i < num_keys; ++i) {
    if (strcmp(arena_config_keys[i], "max_mem") == 0) {
      cfg->max_mem = arena_config_values[i];
    } else if (strcmp(arena_config_keys[i], "arena_extend_strategy") == 0) {
      cfg->arena_extend_strategy = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "initial_chunk_size_bytes") == 0) {
      cfg->initial_chunk_size_bytes = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "max_dead_bytes_per_chunk") == 0) {
      cfg->max_dead_bytes_per_chunk = static_cast<int>(arena_config_values[i]);
    } else if (strcmp(arena_config_keys[i], "initial_growth_chunk_size_bytes") == 0) {
      cfg->initial_growth_chunk_size_bytes = static_cast<int>(arena_config_values[i]);
    } else {
      std::ostringstream oss;
      oss << "Invalid key found: " << arena_config_keys[i];

      return CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
    }
  }

  *out = cfg.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseArenaCfg, _Frees_ptr_opt_ OrtArenaCfg* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out) {
  API_IMPL_BEGIN
  std::unique_ptr<PrepackedWeightsContainer> container(new PrepackedWeightsContainer());
  *out = reinterpret_cast<OrtPrepackedWeightsContainer*>(container.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleasePrepackedWeightsContainer, _Frees_ptr_opt_ OrtPrepackedWeightsContainer* ptr) {
  delete reinterpret_cast<PrepackedWeightsContainer*>(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess, prepacked_weights_container));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data,
                                                      model_data_length, sess));
    ORT_API_RETURN_IF_ERROR(InitializeSession(options, sess, prepacked_weights_container));

    *out = reinterpret_cast<OrtSession*>(sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled.");
}
#endif

static constexpr OrtApiBase ort_api_base = {
    &OrtApis::GetApi,
    &OrtApis::GetVersionString,
};

/* Rules on how to add a new Ort API version

In general, NEVER remove or rearrange the members in this structure unless a new version is being created. The
goal is for newer shared libraries of the Onnx Runtime to work with binaries targeting the previous versions.
In order to do that we need to ensure older binaries get the older interfaces they are expecting.

If the next version of the OrtApi only adds members, new members can be added at the end of the OrtApi structure
without breaking anything. In this case, rename the ort_api_# structure in a way that shows the range of versions
it supports, for example 'ort_api_1_to_2', and then GetApi can return the same structure for a range of versions.

If methods need to be removed or rearranged, then make a copy of the OrtApi structure and name it 'OrtApi#to#'.
The latest Api should always be named just OrtApi. Then make a copy of the latest ort_api_* structure below and
name it ort_api_# to match the latest version number supported, you'll need to be sure the structure types match
the API they're for (the compiler should complain if this isn't correct).

If there is no desire to have the headers still expose the older APIs (clutter, documentation, etc) then the
definition should be moved to a file included by this file so that it's still defined here for binary compatibility
but isn't visible in public headers.

So for example, if we wanted to just add some new members to the ort_api_1_to_2, we'd take the following steps:

	In include\onnxruntime\core\session\onnxruntime_c_api.h we'd just add the members to the end of the structure

	In this file, we'd correspondingly add the member values to the end of the ort_api_1_to_2 structure, and also rename
	it to ort_api_1_to_3.

	Then in GetApi we'd make it return ort_api_1_to_3 for versions 1 through 3.

Second example, if we wanted to add and remove some members, we'd do this:

	In include\onnxruntime\core\session\onnxruntime_c_api.h we'd make a copy of the OrtApi structure and name the
	old one OrtApi1to2. In the new OrtApi we'd add or remove any members that we desire.

	In this file, we'd create a new copy of ort_api_1_to_2 called ort_api_3 and make the corresponding changes that were
	made to the new OrtApi.

	In GetApi we now make it return ort_api_3 for version 3.
*/

static constexpr OrtApi ort_api_1_to_8 = {
    // NOTE: The ordering of these fields MUST not change after that version has shipped since existing binaries depend on this ordering.

    // Shipped as version 1 - DO NOT MODIFY (see above text for more information)
    &OrtApis::CreateStatus,
    &OrtApis::GetErrorCode,
    &OrtApis::GetErrorMessage,

    &OrtApis::CreateEnv,
    &OrtApis::CreateEnvWithCustomLogger,
    &OrtApis::EnableTelemetryEvents,
    &OrtApis::DisableTelemetryEvents,

    &OrtApis::CreateSession,
    &OrtApis::CreateSessionFromArray,
    &OrtApis::Run,

    &OrtApis::CreateSessionOptions,
    &OrtApis::SetOptimizedModelFilePath,
    &OrtApis::CloneSessionOptions,
    &OrtApis::SetSessionExecutionMode,
    &OrtApis::EnableProfiling,
    &OrtApis::DisableProfiling,
    &OrtApis::EnableMemPattern,
    &OrtApis::DisableMemPattern,
    &OrtApis::EnableCpuMemArena,
    &OrtApis::DisableCpuMemArena,
    &OrtApis::SetSessionLogId,
    &OrtApis::SetSessionLogVerbosityLevel,
    &OrtApis::SetSessionLogSeverityLevel,
    &OrtApis::SetSessionGraphOptimizationLevel,
    &OrtApis::SetIntraOpNumThreads,
    &OrtApis::SetInterOpNumThreads,

    &OrtApis::CreateCustomOpDomain,
    &OrtApis::CustomOpDomain_Add,
    &OrtApis::AddCustomOpDomain,
    &OrtApis::RegisterCustomOpsLibrary,

    &OrtApis::SessionGetInputCount,
    &OrtApis::SessionGetOutputCount,
    &OrtApis::SessionGetOverridableInitializerCount,
    &OrtApis::SessionGetInputTypeInfo,
    &OrtApis::SessionGetOutputTypeInfo,
    &OrtApis::SessionGetOverridableInitializerTypeInfo,
    &OrtApis::SessionGetInputName,
    &OrtApis::SessionGetOutputName,
    &OrtApis::SessionGetOverridableInitializerName,

    &OrtApis::CreateRunOptions,
    &OrtApis::RunOptionsSetRunLogVerbosityLevel,
    &OrtApis::RunOptionsSetRunLogSeverityLevel,
    &OrtApis::RunOptionsSetRunTag,
    &OrtApis::RunOptionsGetRunLogVerbosityLevel,
    &OrtApis::RunOptionsGetRunLogSeverityLevel,
    &OrtApis::RunOptionsGetRunTag,
    &OrtApis::RunOptionsSetTerminate,
    &OrtApis::RunOptionsUnsetTerminate,

    &OrtApis::CreateTensorAsOrtValue,
    &OrtApis::CreateTensorWithDataAsOrtValue,
    &OrtApis::IsTensor,
    &OrtApis::GetTensorMutableData,
    &OrtApis::FillStringTensor,

    &OrtApis::GetStringTensorDataLength,
    &OrtApis::GetStringTensorContent,

    &OrtApis::CastTypeInfoToTensorInfo,
    &OrtApis::GetOnnxTypeFromTypeInfo,
    &OrtApis::CreateTensorTypeAndShapeInfo,
    &OrtApis::SetTensorElementType,

    &OrtApis::SetDimensions,
    &OrtApis::GetTensorElementType,
    &OrtApis::GetDimensionsCount,
    &OrtApis::GetDimensions,
    &OrtApis::GetSymbolicDimensions,
    &OrtApis::GetTensorShapeElementCount,
    &OrtApis::GetTensorTypeAndShape,
    &OrtApis::GetTypeInfo,
    &OrtApis::GetValueType,
    &OrtApis::CreateMemoryInfo,
    &OrtApis::CreateCpuMemoryInfo,
    &OrtApis::CompareMemoryInfo,
    &OrtApis::MemoryInfoGetName,
    &OrtApis::MemoryInfoGetId,
    &OrtApis::MemoryInfoGetMemType,
    &OrtApis::MemoryInfoGetType,
    &OrtApis::AllocatorAlloc,
    &OrtApis::AllocatorFree,
    &OrtApis::AllocatorGetInfo,
    &OrtApis::GetAllocatorWithDefaultOptions,
    &OrtApis::AddFreeDimensionOverride,
    &OrtApis::GetValue,
    &OrtApis::GetValueCount,
    &OrtApis::CreateValue,
    &OrtApis::CreateOpaqueValue,
    &OrtApis::GetOpaqueValue,

    &OrtApis::KernelInfoGetAttribute_float,
    &OrtApis::KernelInfoGetAttribute_int64,
    &OrtApis::KernelInfoGetAttribute_string,
    &OrtApis::KernelContext_GetInputCount,
    &OrtApis::KernelContext_GetOutputCount,
    &OrtApis::KernelContext_GetInput,
    &OrtApis::KernelContext_GetOutput,

    &OrtApis::ReleaseEnv,
    &OrtApis::ReleaseStatus,
    &OrtApis::ReleaseMemoryInfo,
    &OrtApis::ReleaseSession,
    &OrtApis::ReleaseValue,
    &OrtApis::ReleaseRunOptions,
    &OrtApis::ReleaseTypeInfo,
    &OrtApis::ReleaseTensorTypeAndShapeInfo,
    &OrtApis::ReleaseSessionOptions,
    &OrtApis::ReleaseCustomOpDomain,
    // End of Version 1 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetDenotationFromTypeInfo,
    &OrtApis::CastTypeInfoToMapTypeInfo,
    &OrtApis::CastTypeInfoToSequenceTypeInfo,
    &OrtApis::GetMapKeyType,
    &OrtApis::GetMapValueType,
    &OrtApis::GetSequenceElementType,
    &OrtApis::ReleaseMapTypeInfo,
    &OrtApis::ReleaseSequenceTypeInfo,
    &OrtApis::SessionEndProfiling,
    &OrtApis::SessionGetModelMetadata,
    &OrtApis::ModelMetadataGetProducerName,
    &OrtApis::ModelMetadataGetGraphName,
    &OrtApis::ModelMetadataGetDomain,
    &OrtApis::ModelMetadataGetDescription,
    &OrtApis::ModelMetadataLookupCustomMetadataMap,
    &OrtApis::ModelMetadataGetVersion,
    &OrtApis::ReleaseModelMetadata,
    // End of Version 2 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::CreateEnvWithGlobalThreadPools,
    &OrtApis::DisablePerSessionThreads,
    &OrtApis::CreateThreadingOptions,
    &OrtApis::ReleaseThreadingOptions,
    &OrtApis::ModelMetadataGetCustomMetadataMapKeys,
    &OrtApis::AddFreeDimensionOverrideByName,
    // End of Version 3 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetAvailableProviders,
    &OrtApis::ReleaseAvailableProviders,
    // End of Version 4 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::GetStringTensorElementLength,
    &OrtApis::GetStringTensorElement,
    &OrtApis::FillStringTensorElement,
    &OrtApis::AddSessionConfigEntry,

    // IoBinding and above are propagated in the same order to C# API
    // Do not move
    &OrtApis::CreateAllocator,
    &OrtApis::ReleaseAllocator,
    &OrtApis::RunWithBinding,
    &OrtApis::CreateIoBinding,
    &OrtApis::ReleaseIoBinding,
    &OrtApis::BindInput,
    &OrtApis::BindOutput,
    &OrtApis::BindOutputToDevice,
    &OrtApis::GetBoundOutputNames,
    &OrtApis::GetBoundOutputValues,
    &OrtApis::ClearBoundInputs,
    &OrtApis::ClearBoundOutputs,
    &OrtApis::TensorAt,
    &OrtApis::CreateAndRegisterAllocator,
    &OrtApis::SetLanguageProjection,
    &OrtApis::SessionGetProfilingStartTimeNs,
    &OrtApis::SetGlobalIntraOpNumThreads,
    &OrtApis::SetGlobalInterOpNumThreads,
    &OrtApis::SetGlobalSpinControl,
    // End of Version 5 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::AddInitializer,
    &OrtApis::CreateEnvWithCustomLoggerAndGlobalThreadPools,
    &OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
    &OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
    &OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
    &OrtApis::SetGlobalDenormalAsZero,
    &OrtApis::CreateArenaCfg,
    &OrtApis::ReleaseArenaCfg,
    // End of Version 6 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::ModelMetadataGetGraphDescription,
    &OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
    &OrtApis::SetCurrentGpuDeviceId,
    &OrtApis::GetCurrentGpuDeviceId,
    // End of Version 7 - DO NOT MODIFY ABOVE (see above text for more information)

    &OrtApis::KernelInfoGetAttributeArray_float,
    &OrtApis::KernelInfoGetAttributeArray_int64,
    &OrtApis::CreateArenaCfgV2,
    &OrtApis::AddRunConfigEntry,
    &OrtApis::CreatePrepackedWeightsContainer,
    &OrtApis::ReleasePrepackedWeightsContainer,
    &OrtApis::CreateSessionWithPrepackedWeightsContainer,
    &OrtApis::CreateSessionFromArrayWithPrepackedWeightsContainer,
    // End of Version 8 - DO NOT MODIFY ABOVE (see above text for more information)

    // Version 9 - In development, feel free to add/remove/rearrange here
    &OrtApis::SetSessionOnnxOpsetVersion,
    &OrtApis::SessionAddONNXOpDomain,
};

// Assert to do a limited check to ensure Version 1 of OrtApi never changes (will detect an addition or deletion but not if they cancel out each other)
// If this assert hits, read the above 'Rules on how to add a new Ort API version'
static_assert(offsetof(OrtApi, ReleaseCustomOpDomain) / sizeof(void*) == 101, "Size of version 1 API cannot change");

ORT_API(const OrtApi*, OrtApis::GetApi, uint32_t version) {
  if (version >= 1 && version <= ORT_API_VERSION)
    return &ort_api_1_to_8;

  fprintf(stderr, "The given version [%u] is not supported, only version 1 to %u is supported in this build.\n",
          version, ORT_API_VERSION);

  return nullptr;  // Unsupported version
}

ORT_API(const char*, OrtApis::GetVersionString) {
  return ORT_VERSION;
}

const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
  return &ort_api_base;
}

ORT_API(void, OrtApis::ReleaseEnv, OrtEnv* value) {
  OrtEnv::Release(value);
}

DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Value, OrtValue)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(RunOptions, OrtRunOptions)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(Session, ::onnxruntime::InferenceSession)
DEFINE_RELEASE_ORT_OBJECT_FUNCTION(ModelMetadata, ::onnxruntime::ModelMetadata)
