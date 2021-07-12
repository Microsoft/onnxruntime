// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"

#ifdef ENABLE_TRAINING
#include "core/dlpack/dlpack_converter.h"
#endif

#include <pybind11/pybind11.h>

// execution provider factory creator headers
struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

#define BACKEND_DEVICE BACKEND_PROC BACKEND_DNNL BACKEND_OPENVINO BACKEND_NUPHAR BACKEND_OPENBLAS BACKEND_MIGRAPHX BACKEND_ACL BACKEND_ARMNN BACKEND_DML
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"

#if defined(USE_CUDA) || defined(USE_ROCM)
#define BACKEND_PROC "GPU"
#else
#define BACKEND_PROC "CPU"
#endif

#if _OPENMP
#define BACKEND_OPENMP "-OPENMP"
#else
#define BACKEND_OPENMP ""
#endif

#if USE_DNNL
#define BACKEND_DNNL "-DNNL"
#else
#define BACKEND_DNNL ""
#endif

#if USE_MIGRAPHX
#define BACKEND_MIGRAPHX "-MIGRAPHX"
#else
#define BACKEND_MIGRAPHX ""
#endif

#ifdef USE_OPENVINO
#if OPENVINO_CONFIG_CPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_CPU_FP32"

#elif OPENVINO_CONFIG_GPU_FP32
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP32"

#elif OPENVINO_CONFIG_GPU_FP16
#define BACKEND_OPENVINO "-OPENVINO_GPU_FP16"

#elif OPENVINO_CONFIG_MYRIAD
#define BACKEND_OPENVINO "-OPENVINO_MYRIAD"

#elif OPENVINO_CONFIG_VAD_M
#define BACKEND_OPENVINO "-OPENVINO_VAD_M"

#elif OPENVINO_CONFIG_VAD_F
#define BACKEND_OPENVINO "-OPENVINO_VAD_F"

#elif OPENVINO_CONFIG_MULTI
#define BACKEND_OPENVINO "-OPENVINO_MULTI"

#elif OPENVINO_CONFIG_HETERO
#define BACKEND_OPENVINO "-OPENVINO_HETERO"
#endif
#else
#define BACKEND_OPENVINO ""
#endif

#ifdef USE_NUPHAR
#define BACKEND_NUPHAR "-NUPHAR"
#else
#define BACKEND_NUPHAR ""
#endif

#if USE_VITISAI
#define BACKEND_VITISAI "-VITISAI"
#include "core/providers/vitisai/vitisai_execution_provider.h"
#else
#define BACKEND_VITISAI ""
#endif

#if USE_OPENBLAS
#define BACKEND_OPENBLAS "-OPENBLAS"
#else
#define BACKEND_OPENBLAS ""
#endif

#if USE_ACL
#define BACKEND_ACL "-ACL"
#else
#define BACKEND_ACL ""
#endif

#if USE_ARMNN
#define BACKEND_ARMNN "-ARMNN"
#else
#define BACKEND_ARMNN ""
#endif

#if USE_DML
#define BACKEND_DML "-DML"
#else
#define BACKEND_DML ""
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "core/providers/migraphx/migraphx_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
// TODO remove deprecated global config
namespace onnxruntime {
ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO();
namespace python {
extern std::string openvino_device_type;
}
}  // namespace onnxruntime
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
// TODO remove deprecated global config
namespace onnxruntime {
namespace python {
extern std::string nuphar_settings;
}
}  // namespace onnxruntime
#endif
#ifdef USE_VITISAI
#include "core/providers/vitisai/vitisai_provider_factory.h"
#endif
#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif
#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
#ifdef USE_CUDA
namespace onnxruntime {
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
ProviderInfo_CUDA& GetProviderInfo_CUDA();
namespace python {
// TODO remove deprecated global config
extern OrtCudnnConvAlgoSearch cudnn_conv_algo_search;
// TODO remove deprecated global config
extern bool do_copy_in_default_stream;
extern onnxruntime::CUDAExecutionProviderExternalAllocatorInfo external_allocator_info;
}  // namespace python
}  // namespace onnxruntime
#endif

#ifdef USE_ROCM
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_provider_factory_creator.h"
namespace onnxruntime {
namespace python {
extern onnxruntime::ROCMExecutionProviderExternalAllocatorInfo external_allocator_info;
}
}  // namespace onnxruntime
#endif

// TODO remove deprecated global config
namespace onnxruntime {
namespace python {
extern onnxruntime::ArenaExtendStrategy arena_extend_strategy;
}
}  // namespace onnxruntime
#endif

#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/shared_library/provider_host_api.h"

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class SparseTensor;
#endif
namespace python {

// TODO remove deprecated global config
extern OrtDevice::DeviceId cuda_device_id;
// TODO remove deprecated global config
extern size_t gpu_mem_limit;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
struct CustomOpLibrary {
  CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so);

  ~CustomOpLibrary();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpLibrary);

 private:
  void UnloadLibrary();

  std::string library_path_;
  void* library_handle_ = nullptr;
};
#endif

// Thin wrapper over internal C++ SessionOptions to accommodate custom op library management for the Python user
struct PySessionOptions : public SessionOptions {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // `PySessionOptions` has a vector of shared_ptrs to CustomOpLibrary, because so that it can be re-used for all
  // `PyInferenceSession`s using the same `PySessionOptions` and that each `PyInferenceSession` need not construct
  // duplicate CustomOpLibrary instances.
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;

  // Hold raw `OrtCustomOpDomain` pointers - it is upto the shared library to release the OrtCustomOpDomains
  // that was created when the library is unloaded
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
#endif
};

// Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
struct PyInferenceSession {
  PyInferenceSession(Environment& env, const PySessionOptions& so) {
    sess_ = std::make_unique<InferenceSession>(so, env);
  }

#if !defined(ORT_MINIMAL_BUILD)
  PyInferenceSession(Environment& env, const PySessionOptions& so, const std::string& arg, bool is_arg_file_name) {
    if (is_arg_file_name) {
      // Given arg is the file path. Invoke the corresponding ctor().
      sess_ = std::make_unique<InferenceSession>(so, env, arg);
    } else {
      // Given arg is the model content as bytes. Invoke the corresponding ctor().
      std::istringstream buffer(arg);
      sess_ = std::make_unique<InferenceSession>(so, env, buffer);
    }
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  void AddCustomOpLibraries(const std::vector<std::shared_ptr<CustomOpLibrary>>& custom_op_libraries) {
    if (!custom_op_libraries.empty()) {
      custom_op_libraries_.reserve(custom_op_libraries.size());
      for (size_t i = 0; i < custom_op_libraries.size(); ++i) {
        custom_op_libraries_.push_back(custom_op_libraries[i]);
      }
    }
  }
#endif

  InferenceSession* GetSessionHandle() const { return sess_.get(); }

  virtual ~PyInferenceSession() {}

 protected:
  PyInferenceSession(std::unique_ptr<InferenceSession> sess) {
    sess_ = std::move(sess);
  }

 private:
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Hold CustomOpLibrary resources so as to tie it to the life cycle of the InferenceSession needing it.
  // NOTE: Define this above `sess_` so that this is destructed AFTER the InferenceSession instance -
  // this is so that the custom ops held by the InferenceSession gets destroyed prior to the library getting unloaded
  // (if ref count of the shared_ptr reaches 0)
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;
#endif

  std::unique_ptr<InferenceSession> sess_;
};

inline const PySessionOptions& GetDefaultCPUSessionOptions() {
  static PySessionOptions so;
  return so;
}

inline AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<TAllocator>();
  return alloc;
}

// This class exposes SparseTensor to Python
// The class serves two major purposes
// - to be able to map numpy arrays memory and use it on input, this serves as a reference holder
//   so incoming arrays do not disappear
// - to be able to expose SparseTensor returned from run method
class PySparseTensor {
 public:
  /// <summary>
  /// Use this constructor when you created a SparseTensor instance which is backed
  /// by python array storage and it important that they stay alive while this object is
  /// alive
  /// </summary>
  /// <param name="instance">a fully constructed and populated instance of SparseTensor</param>
  /// <param name="storage">a collection reference guards</param>
  PySparseTensor(std::unique_ptr<SparseTensor>&& instance,
                 std::vector<pybind11::object>&& storage)
      : backing_storage_(std::move(storage)), ort_value_() {
    Init(std::move(instance));
  }

  /// <summary>
  /// Same as above but no backing storage as SparseTensor owns the memory
  /// </summary>
  /// <param name="instance"></param>
  explicit PySparseTensor(std::unique_ptr<SparseTensor>&& instance)
      : backing_storage_(), ort_value_() {
    Init(std::move(instance));
  }

  explicit PySparseTensor(const OrtValue& ort_value)
      : backing_storage_(), ort_value_(ort_value) {}

  PySparseTensor(const PySparseTensor&) = delete;
  PySparseTensor& operator=(const PySparseTensor&) = delete;

  PySparseTensor(PySparseTensor&& o) noexcept {
    *this = std::move(o);
  }

  PySparseTensor& operator=(PySparseTensor&& o) noexcept {
    ort_value_ = std::move(o.ort_value_);
    backing_storage_ = std::move(o.backing_storage_);
    return *this;
  }

  ~PySparseTensor();

  const SparseTensor& Instance() const {
    return ort_value_.Get<SparseTensor>();
  }

  SparseTensor* Instance() {
    return ort_value_.GetMutable<SparseTensor>();
  }

  std::unique_ptr<OrtValue> AsOrtValue() const {
    return std::make_unique<OrtValue>(ort_value_);
  }

 private:
  void Init(std::unique_ptr<SparseTensor>&& instance);

  // These will hold references to underpinning python array objects
  // when they serve as a backing storage for a feeding SparseTensor
  std::vector<pybind11::object> backing_storage_;
  OrtValue ort_value_;
};

class SessionObjectInitializer {
 public:
  typedef const PySessionOptions& Arg1;
  // typedef logging::LoggingManager* Arg2;
  static const std::string default_logger_id;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  // operator Arg2() {
  //   static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
  //                                                 Severity::kWARNING, false, LoggingManager::InstanceType::Default,
  //                                                 &default_logger_id};
  //   return &default_logging_manager;
  // }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

Environment& GetEnv();

// Initialize an InferenceSession.
// Any provider_options should have entries in matching order to provider_types.
void InitializeSession(InferenceSession* sess,
                       const std::vector<std::string>& provider_types = {},
                       const ProviderOptionsVector& provider_options = {},
                       const std::unordered_set<std::string>& disabled_optimizer_names = {});

// Checks if PyErrOccured, fetches status and throws.
void ThrowIfPyErrOccured();

void addOrtValueMethods(pybind11::module& m);

void addIoBindingMethods(pybind11::module& m);

void addSparseTensorMethods(pybind11::module& m);

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
void addGlobalSchemaFunctions(pybind11::module& m);

void addOpKernelSubmodule(pybind11::module& m);

void addOpSchemaSubmodule(pybind11::module& m);
#endif

const char* GetDeviceName(const OrtDevice& device);

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

bool CheckIfTensor(const std::vector<const NodeArg*>& def_list,
                   const std::string& name,
                   /*out*/ ONNX_NAMESPACE::TypeProto& type_proto);

#ifdef ENABLE_TRAINING

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject* ToDlpack(OrtValue ort_value);

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the conversion.
OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor);

#endif

}  // namespace python

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VITISAI(const char* backend_type, int device_id,
                                                                                  const char* export_runtime_module,
                                                                                  const char* load_runtime_module);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(uint32_t flags);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();

}  // namespace onnxruntime
