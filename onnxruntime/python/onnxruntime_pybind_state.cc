// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <functional>
#include <mutex>
#include <thread>
#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_model_compiler.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/common/narrow.h"
#include "core/common/optional.h"
#include "core/common/path_string.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/data_transfer_utils.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/framework/provider_options_utils.h"
#include "core/framework/random_seed.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/env.h"
#include "core/providers/get_execution_providers.h"
#include "core/providers/providers.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/ort_env.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/provider_bridge_ort.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "core/session/lora_adapters.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/abi_graph_types.h"
#include "core/session/abi_devices.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"
#endif

#ifdef ENABLE_ATEN
#include "contrib_ops/cpu/aten_ops/aten_op_executor.h"
#endif

#ifdef USE_CUDA
#include <cuda.h>   // for CUDA_VERSION
#include <cudnn.h>  // for CUDNN_MAJOR
#endif

#if defined(USE_COREML)
#include "core/providers/coreml/coreml_provider_factory.h"
#endif

#include <pybind11/functional.h>

// Explicitly provide a definition for the static const var 'GPU' in the OrtDevice struct,
// GCC 4.x doesn't seem to define this and it breaks the pipelines based on CentOS as it uses
// GCC 4.x.
// (This static var is referenced in GetCudaToHostMemCpyFunction())
const OrtDevice::DeviceType OrtDevice::GPU;

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503)
#endif  // _MSC_VER

#include <iterator>
#include <algorithm>
#include <limits>
#include <string_view>
#include <utility>

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;
using namespace onnxruntime::logging;

namespace {

constexpr std::string_view kEpCudaProviderOptionPrefix{"ep.cuda."};

std::unique_ptr<OrtValue> CreateSessionOrtValue(PyInferenceSession* session,
                                                const std::vector<int64_t>& shape,
                                                MLDataType element_type,
                                                const OrtDevice& device) {
  // Accessing session state enforces that initialization has completed.
  (void)session->GetSessionHandle()->GetSessionState();

  const char* allocator_name = GetDeviceAllocatorName(device);
  OrtMemoryInfo memory_info{allocator_name, OrtDeviceAllocator, device};
  auto allocator = session->GetSessionHandle()->GetAllocator(memory_info);
  if (!allocator) {
    throw std::runtime_error("No session allocator found for " + device.ToString());
  }

  auto ort_value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(element_type, gsl::make_span(shape), std::move(allocator), *ort_value);
  return ort_value;
}

struct AdaptedProviderOptions {
  std::vector<std::string> keys;
  std::vector<std::string> values;
  std::vector<const char*> key_ptrs;
  std::vector<const char*> value_ptrs;
};

AdaptedProviderOptions AdaptProviderOptionsForRegisteredPluginEp(const std::string& ep_name,
                                                                 const ProviderOptions& provider_options) {
  AdaptedProviderOptions adapted_options;
  adapted_options.keys.reserve(provider_options.size());
  adapted_options.values.reserve(provider_options.size());

  for (const auto& [key, value] : provider_options) {
    std::string_view adapted_key{key};

    if (adapted_key.rfind(kEpCudaProviderOptionPrefix, 0) == 0) {
      adapted_key.remove_prefix(kEpCudaProviderOptionPrefix.size());
    }

    if (ep_name == kCudaExecutionProvider) {
      if (adapted_key == "device_id") {
        continue;
      }

      if (adapted_key == "prefer_nhwc_layout") {
        adapted_key = "prefer_nhwc";
      }
    }

    adapted_options.keys.emplace_back(adapted_key);
    adapted_options.values.push_back(value);
  }

  adapted_options.key_ptrs.reserve(adapted_options.keys.size());
  adapted_options.value_ptrs.reserve(adapted_options.values.size());
  for (size_t i = 0; i < adapted_options.keys.size(); ++i) {
    adapted_options.key_ptrs.push_back(adapted_options.keys[i].c_str());
    adapted_options.value_ptrs.push_back(adapted_options.values[i].c_str());
  }

  return adapted_options;
}

}  // namespace

using PyCallback = std::function<void(std::vector<py::object>, py::object user_data, std::string)>;

struct PendingPythonReleases {
  std::mutex mutex;
  std::vector<PyObject*> objects;
  std::vector<PyObject*> cleanup_thread_objects;
  bool callback_scheduled = false;
  bool cleanup_thread_running = false;
};

PendingPythonReleases& GetPendingPythonReleases() {
  static auto* pending_releases = new PendingPythonReleases();
  return *pending_releases;
}

int DrainPendingPythonReleases(void*) {
  std::vector<PyObject*> objects;
  auto& pending_releases = GetPendingPythonReleases();
  {
    std::lock_guard<std::mutex> lock{pending_releases.mutex};
    objects.swap(pending_releases.objects);
    pending_releases.callback_scheduled = false;
  }

  for (PyObject* object : objects) {
    Py_DECREF(object);
  }

  return 0;
}

void DrainCleanupThreadPythonReleases() {
  for (;;) {
    std::vector<PyObject*> objects;
    auto& pending_releases = GetPendingPythonReleases();
    {
      std::lock_guard<std::mutex> lock{pending_releases.mutex};
      if (pending_releases.cleanup_thread_objects.empty()) {
        pending_releases.cleanup_thread_running = false;
        return;
      }
      objects.swap(pending_releases.cleanup_thread_objects);
    }

    {
      py::gil_scoped_acquire acquire;
      for (PyObject* object : objects) {
        Py_DECREF(object);
      }
    }
  }
}

void ReleasePythonObjectOnCleanupThread(PyObject* object) noexcept {
  auto& pending_releases = GetPendingPythonReleases();
  std::lock_guard<std::mutex> lock{pending_releases.mutex};
  ORT_TRY {
    pending_releases.cleanup_thread_objects.push_back(object);
  }
  ORT_CATCH(...) {
    // Retaining the reference is safer than releasing it on the ORT worker.
    return;
  }

  if (!pending_releases.cleanup_thread_running) {
    pending_releases.cleanup_thread_running = true;
    std::thread* cleanup_thread = nullptr;
    ORT_TRY {
      cleanup_thread = new std::thread(DrainCleanupThreadPythonReleases);
    }
    ORT_CATCH(...) {
      pending_releases.cleanup_thread_running = false;
      return;
    }

    ORT_TRY {
      cleanup_thread->detach();
      delete cleanup_thread;
    }
    ORT_CATCH(...) {
      // The valid thread remains joinable and will drain the queue. Leak its
      // handle rather than destroying it and terminating the process.
    }
  }
}

void DeferPythonRelease(PyObject* object) noexcept {
  if (object == nullptr) {
    return;
  }

  bool release_on_cleanup_thread = false;
  ORT_TRY {
    auto& pending_releases = GetPendingPythonReleases();
    std::lock_guard<std::mutex> lock{pending_releases.mutex};
    pending_releases.objects.push_back(object);
    if (!pending_releases.callback_scheduled &&
        Py_AddPendingCall(DrainPendingPythonReleases, nullptr) != 0) {
      pending_releases.objects.pop_back();
      release_on_cleanup_thread = true;
    } else {
      pending_releases.callback_scheduled = true;
    }
  }
  ORT_CATCH(...) {
    release_on_cleanup_thread = true;
  }

  if (release_on_cleanup_thread) {
    ReleasePythonObjectOnCleanupThread(object);
  }
}

struct AsyncResource {
  std::vector<py::object> feed_objects;
  py::object session;
  py::object run_options;

  std::vector<OrtValue> feeds;
  std::vector<const OrtValue*> feeds_raw;

  std::vector<std::string> feed_names;
  std::vector<const char*> feed_names_raw;

  std::vector<OrtValue*> fetches_raw;  // will be released during destruction

  std::vector<std::string> fetch_names;
  std::vector<const char*> fetch_names_raw;

  RunOptions default_run_option;
  PyCallback callback;
  py::object user_data;

  void ReserveFeeds(size_t sz) {
    feed_objects.reserve(sz);
    feeds.reserve(sz);
    feeds_raw.reserve(sz);
    feed_names.reserve(sz);
    feed_names_raw.reserve(sz);
  }

  void ReserveFetches(size_t sz) {
    fetches_raw.reserve(sz);
    fetch_names.reserve(sz);
    fetch_names_raw.reserve(sz);
  }

  ~AsyncResource() {
    std::for_each(fetches_raw.begin(), fetches_raw.end(), [](const OrtValue* fetch) {
      if (fetch) {
        std::unique_ptr<const OrtValue> fetch_recycler(fetch);
      }
    });
    fetches_raw.clear();
  }
};

void AsyncCallback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr ort_status) {
  ORT_ENFORCE(user_data, "user data must not be NULL for callback in python");

  auto invoke_callback = [&]() {
    std::unique_ptr<AsyncResource> async_resource{reinterpret_cast<AsyncResource*>(user_data)};
    PyObject* session = async_resource->session.release().ptr();

    try {
      Ort::Status status(ort_status);
      if (!status.IsOK()) {
        async_resource->callback({}, async_resource->user_data, status.GetErrorMessage());
      } else {
        std::vector<py::object> rfetch;
        rfetch.reserve(num_outputs);
        size_t pos = 0;
        for (size_t ith = 0; ith < num_outputs; ++ith) {
          const auto& fet = *outputs[ith];
          if (fet.IsAllocated()) {
            if (fet.IsTensor()) {
              rfetch.push_back(AddTensorAsPyObj(fet, nullptr, nullptr));
            } else if (fet.IsSparseTensor()) {
              rfetch.push_back(GetPyObjectFromSparseTensor(pos, fet, nullptr));
            } else {
              rfetch.push_back(AddNonTensorAsPyObj(fet, nullptr, nullptr));
            }
          } else {
            rfetch.push_back(py::none());
          }
          ++pos;
        }
        async_resource->callback(rfetch, async_resource->user_data, "");
      }
    } catch (py::error_already_set& ex) {
      ex.discard_as_unraisable("onnxruntime.InferenceSession.run_async callback");
    } catch (const std::exception& ex) {
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      PyErr_WriteUnraisable(Py_None);
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError, "Unknown exception in run_async callback");
      PyErr_WriteUnraisable(Py_None);
    }

    DeferPythonRelease(session);
  };

  if (PyGILState_Check()) {
    invoke_callback();
  } else {
    // acquire GIL to safely:
    // 1) invoke python callback
    // 2) create, manipulate, and destroy python objects
    py::gil_scoped_acquire acquire;
    invoke_callback();
  }
}

void AppendLoraParametersAsInputs(const RunOptions& run_options,
                                  size_t total_entries,
                                  NameMLValMap& feeds) {
  for (const auto* adapter : run_options.active_adapters) {
    total_entries += adapter->GetParamNum();
  }
  feeds.reserve(total_entries + feeds.size());

  // Append necessary inputs for active adapters
  for (const auto* adapter : run_options.active_adapters) {
    auto [begin, end] = adapter->GetParamIterators();
    for (; begin != end; ++begin) {
      const auto& [name, param] = *begin;
      feeds.insert(std::make_pair(name, param.GetMapped()));
    }
  }
}

template <typename T>
static py::object AddNonTensor(const OrtValue& val,
                               const DataTransferManager* /*data_transfer_manager*/,
                               const std::unordered_map<OrtDevice, MemCpyFunc>* /*mem_cpy_to_host_functions*/) {
  return py::cast(val.Get<T>());
}

// This function is used to return strings from a string tensor to python
// as a numpy array of strings
// Strings are always on CPU and must always be copied to python memory
py::array StringTensorToNumpyArray(const Tensor& tensor) {
  // Create the result and allocate memory with the right size
  py::array result(py::dtype(NPY_OBJECT), tensor.Shape().GetDims());
  const auto span = tensor.DataAsSpan<std::string>();
  auto* mutable_data = reinterpret_cast<py::object*>(result.mutable_data());
  for (size_t i = 0, lim = span.size(); i < lim; ++i) {
    mutable_data[i] = py::cast(span[i]);
  }
  return result;
}

pybind11::array PrimitiveTensorToNumpyOverOrtValue(const OrtValue& ort_value) {
  const Tensor& tensor = ort_value.Get<Tensor>();
  // The capsule destructor must be stateless
  // We create a copy of OrtValue on the heap.
  auto memory_release = [](void* data) {
    auto* ort_value = reinterpret_cast<OrtValue*>(data);
    delete ort_value;
  };

  const int numpy_type = OnnxRuntimeTensorToNumpyType(tensor.DataType());
  auto ort_value_ptr = std::make_unique<OrtValue>(ort_value);
  pybind11::capsule caps(ort_value_ptr.get(), memory_release);
  ort_value_ptr.release();

  // Not using array_t<T> because it may not handle MLFloat16 properly
  pybind11::array result(py::dtype(numpy_type), tensor.Shape().GetDims(),
                         tensor.DataRaw(),
                         caps);
  return result;
}

pybind11::array PrimitiveTensorToNumpyFromDevice(const OrtValue& ort_value, const DataTransferAlternative& dtm) {
  const Tensor& tensor = ort_value.Get<Tensor>();
  const int numpy_type = OnnxRuntimeTensorToNumpyType(tensor.DataType());
  pybind11::array result(py::dtype(numpy_type), tensor.Shape().GetDims());
  void* data = result.mutable_data();

  if (std::holds_alternative<const DataTransferManager*>(dtm)) {
    const DataTransferManager* data_transfer = std::get<const DataTransferManager*>(dtm);
    static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
    const auto span = gsl::make_span<char>(reinterpret_cast<char*>(data), tensor.SizeInBytes());
    ORT_THROW_IF_ERROR(CopyTensorDataToByteSpan(*data_transfer, tensor, cpu_alloc_info, span));
  } else {
    std::get<MemCpyFunc>(dtm)(data, tensor.DataRaw(), tensor.SizeInBytes());
  }
  return result;
}

// In all cases, we may not have access to a DataTransferManager, hence the user may specify functions that
// pretty much does what a DataTransferManager does - copy data from device(s) to the host
py::object GetPyObjFromTensor(const OrtValue& ort_value,
                              const DataTransferManager* data_transfer_manager,
                              const std::unordered_map<OrtDevice, MemCpyFunc>* mem_cpy_to_host_functions,
                              bool zero_copy_non_owning) {
  ORT_ENFORCE(ort_value.IsTensor(), "This function only supports tensors");

  const auto& tensor = ort_value.Get<Tensor>();
  const auto& device = tensor.Location().device;

  if (tensor.IsDataTypeString()) {
    ORT_ENFORCE(device.Type() == OrtDevice::CPU, "Strings can only be on CPU");
    // Create a numpy array of strings (python objects) by copy/converting them
    py::array result = StringTensorToNumpyArray(tensor);
    return py::cast<py::object>(result);
  }

  // Create a numpy array on top of the OrtValue memory, no copy,
  // but only when the tensor owns the buffer. When the tensor wraps external
  // memory (e.g. a numpy input array passed through as output), the buffer
  // lifetime is not tied to the OrtValue and zero-copy would create a
  // dangling pointer. See https://github.com/microsoft/onnxruntime/issues/21922
  if (device.UsesCpuMemory()) {
    if (tensor.OwnsBuffer() || zero_copy_non_owning) {
      py::array result = PrimitiveTensorToNumpyOverOrtValue(ort_value);
      return py::cast<py::object>(result);
    }
    // Tensor does not own the buffer — must copy to avoid dangling pointers
    // when the underlying memory (e.g. a numpy input array) is freed.
    // See https://github.com/microsoft/onnxruntime/issues/21922
    MemCpyFunc cpu_copy = CpuToCpuMemCpy;
    py::array result = PrimitiveTensorToNumpyFromDevice(ort_value, cpu_copy);
    return py::cast<py::object>(result);
  }

  py::array result;
  if (data_transfer_manager != nullptr) {
    result = PrimitiveTensorToNumpyFromDevice(ort_value, data_transfer_manager);
  } else {
    bool copied = false;
    if (mem_cpy_to_host_functions) {
      auto it = std::find_if(mem_cpy_to_host_functions->begin(), mem_cpy_to_host_functions->end(),
                             [&device](const auto& entry) {
                               const auto& copy_device = entry.first;
                               // We're ignoring OrtDevice.Id() currently for historical reasons.
                               // The key to mem_cpy_to_host_functions was previously the device type (CPU/GPU/NPU).
                               // This changed to be OrtDevice to get the vendor id.
                               // Assumably it would be better to also match on device id, but that was not possible
                               // previously and to preserve existing behavior we keep the old logic and expect the
                               // copy function to handle the device id correctly.
                               return device.Type() == copy_device.Type() &&
                                      device.MemType() == copy_device.MemType() &&
                                      device.Vendor() == copy_device.Vendor();
                             });

      if (it != mem_cpy_to_host_functions->end()) {
        result = PrimitiveTensorToNumpyFromDevice(ort_value, it->second);
        copied = true;
      }
    }

    if (!copied) {
      // see if we have a shared data transfer function from a plugin EP
      auto device_to_cpu_copy_func = CreateDataTransferMemCpy(device, OrtDevice{});
      if (device_to_cpu_copy_func) {
        result = PrimitiveTensorToNumpyFromDevice(ort_value, device_to_cpu_copy_func);
      } else {
        throw std::runtime_error(
            "GetPyObjFromTensor: Either data transfer manager or a "
            "function to copy data to the host is needed to convert non-CPU tensor to numpy array");
      }
    }
  }

  return py::cast<py::object>(result);
}

const char* GetDeviceName(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      switch (device.Vendor()) {
        case OrtDevice::VendorIds::NVIDIA:
          return CUDA;
        case OrtDevice::VendorIds::AMD:
          return HIP;
        case OrtDevice::VendorIds::MICROSOFT:
          return DML;
      }

      return CUDA;  // default to CUDA for backwards compatibility

    case OrtDevice::DML:
      return DML;
    case OrtDevice::FPGA:
      return "FPGA";
    case OrtDevice::NPU:
#ifdef USE_CANN
      if (device.Vendor() == OrtDevice::VendorIds::HUAWEI) {
        return CANN;
      }
      return "NPU";
#else
      return "NPU";
#endif
    default:
      ORT_THROW("Unknown device type: ", device.Type());
  }
}

const char* GetDeviceAllocatorName(const OrtDevice& device) {
  if (device.Type() == OrtDevice::GPU && device.Vendor() == OrtDevice::VendorIds::NONE) {
    return WEBGPU_BUFFER;
  }
  return GetDeviceName(device);
}

py::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager) {
#if !defined(DISABLE_SPARSE_TENSORS)
  if (!ort_value.IsSparseTensor()) {
    ORT_THROW("Must be a sparse tensor");
  }
  auto& logger = logging::LoggingManager::DefaultLogger();
  const SparseTensor& src_sparse_tensor = ort_value.Get<SparseTensor>();
  std::unique_ptr<PySparseTensor> py_sparse_tensor;
  auto device_type = src_sparse_tensor.Location().device.Type();
  if (device_type != OrtDevice::CPU) {
    if (!data_transfer_manager) {
      LOGS(logger, WARNING) << "Returned OrtValue with sparse tensor at position: " << pos << " is on GPU but no data_transfer_manager provided."
                            << " Returned it will have its data on GPU, you can copy it using numpy_array_to_cpu()";
      py_sparse_tensor = std::make_unique<PySparseTensor>(ort_value);
    } else {
      auto dst_sparse_tensor = std::make_unique<SparseTensor>(src_sparse_tensor.DataType(), src_sparse_tensor.DenseShape(), GetAllocator());
      auto status = src_sparse_tensor.Copy(*data_transfer_manager, *dst_sparse_tensor);
      OrtPybindThrowIfError(status);
      py_sparse_tensor = std::make_unique<PySparseTensor>(std::move(dst_sparse_tensor));
    }
  } else {
    py_sparse_tensor = std::make_unique<PySparseTensor>(ort_value);
  }

  py::object result = py::cast(py_sparse_tensor.get(), py::return_value_policy::take_ownership);
  py_sparse_tensor.release();
  return result;
#else
  ORT_UNUSED_PARAMETER(pos);
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_transfer_manager);
  ORT_THROW("SparseTensor support is disabled in this build.");
#endif  // !defined(DISABLE_SPARSE_TENSORS)
}

template <>
py::object AddNonTensor<TensorSeq>(const OrtValue& val,
                                   const DataTransferManager* data_transfer_manager,
                                   const std::unordered_map<OrtDevice, MemCpyFunc>* mem_cpy_to_host_functions) {
  const auto& seq_tensors = val.Get<TensorSeq>();
  py::list py_list;
  for (const auto& ort_value : seq_tensors) {
    py::object obj = GetPyObjFromTensor(ort_value, data_transfer_manager, mem_cpy_to_host_functions);
    py_list.append(std::move(obj));
  }
  // XToolChain kills the build
  // local variable 'py_list' will be copied despite being returned by name [-Werror,-Wreturn-std-move]
  // call 'std::move' explicitly to avoid copying
  // We choose to cast it to object explicitly
  return py::cast<py::object>(py_list);
}

py::object AddNonTensorAsPyObj(const OrtValue& val,
                               const DataTransferManager* data_transfer_manager,
                               const std::unordered_map<OrtDevice, MemCpyFunc>* mem_cpy_to_host_functions) {
  // Should be in sync with core/framework/datatypes.h
  auto val_type = val.Type();
  if (val_type->IsTensorSequenceType()) {
    return AddNonTensor<TensorSeq>(val, data_transfer_manager, mem_cpy_to_host_functions);
  } else {
#if !defined(DISABLE_ML_OPS)
    utils::ContainerChecker c_checker(val_type);
    if (c_checker.IsMap()) {
      if (c_checker.IsMapOf<std::string, std::string>()) {
        return AddNonTensor<MapStringToString>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, int64_t>()) {
        return AddNonTensor<MapStringToInt64>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, float>()) {
        return AddNonTensor<MapStringToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<std::string, double>()) {
        return AddNonTensor<MapStringToDouble>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, std::string>()) {
        return AddNonTensor<MapInt64ToString>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, int64_t>()) {
        return AddNonTensor<MapInt64ToInt64>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, float>()) {
        return AddNonTensor<MapInt64ToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsMapOf<int64_t, double>()) {
        return AddNonTensor<MapInt64ToDouble>(val, data_transfer_manager, mem_cpy_to_host_functions);
      }

    } else {
      if (c_checker.IsSequenceOf<std::map<std::string, float>>()) {
        return AddNonTensor<VectorMapStringToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      } else if (c_checker.IsSequenceOf<std::map<int64_t, float>>()) {
        return AddNonTensor<VectorMapInt64ToFloat>(val, data_transfer_manager, mem_cpy_to_host_functions);
      }
    }
#endif
  }
  ORT_THROW("Non-tensor type is not supported in this build: ", val_type);
}

py::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                            const std::unordered_map<OrtDevice, MemCpyFunc>* mem_cpy_to_host_functions) {
  return GetPyObjFromTensor(val, data_transfer_manager, mem_cpy_to_host_functions);
}

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
const CUDAExecutionProviderInfo GetCudaExecutionProviderInfo(ProviderInfo_CUDA* cuda_provider_info,
                                                             const ProviderOptionsMap& provider_options_map) {
  ORT_ENFORCE(cuda_provider_info);
  const auto it = provider_options_map.find(kCudaExecutionProvider);
  CUDAExecutionProviderInfo info;
  if (it != provider_options_map.end())
    cuda_provider_info->CUDAExecutionProviderInfo__FromProviderOptions(it->second, info);
  else {
    info.device_id = cuda_device_id;
    info.gpu_mem_limit = gpu_mem_limit;
    info.arena_extend_strategy = arena_extend_strategy;
    info.cudnn_conv_algo_search = cudnn_conv_algo_search;
    info.do_copy_in_default_stream = do_copy_in_default_stream;
    info.external_allocator_info = external_allocator_info;
    info.tunable_op = tunable_op;
  }
  return info;
}
#endif

#ifdef USE_CANN
const CANNExecutionProviderInfo GetCannExecutionProviderInfo(ProviderInfo_CANN* cann_provider_info,
                                                             const ProviderOptionsMap& provider_options_map) {
  ORT_ENFORCE(cann_provider_info);
  const auto it = provider_options_map.find(kCannExecutionProvider);
  CANNExecutionProviderInfo info;
  if (it != provider_options_map.end())
    cann_provider_info->CANNExecutionProviderInfo__FromProviderOptions(it->second, info);
  return info;
}
#endif

#if defined(USE_TENSORRT) || defined(USE_TENSORRT_PROVIDER_INTERFACE)
void RegisterTensorRTPluginsAsCustomOps(PySessionOptions& so, const ProviderOptions& options) {
  if (auto* tensorrt_provider_info = TryGetProviderInfo_TensorRT()) {
    auto is_already_in_domains = [&](std::string& domain_name, std::vector<OrtCustomOpDomain*>& domains) {
      for (auto ptr : domains) {
        if (domain_name == ptr->domain_) {
          return true;
        }
      }
      return false;
    };

    std::string trt_extra_plugin_lib_paths = "";
    const auto it = options.find("trt_extra_plugin_lib_paths");
    if (it != options.end()) {
      trt_extra_plugin_lib_paths = it->second;
    }
    std::vector<OrtCustomOpDomain*> custom_op_domains;
    tensorrt_provider_info->GetTensorRTCustomOpDomainList(custom_op_domains, trt_extra_plugin_lib_paths);
    for (auto ptr : custom_op_domains) {
      if (!is_already_in_domains(ptr->domain_, so.custom_op_domains_)) {
        so.custom_op_domains_.push_back(ptr);
      } else {
        LOGS_DEFAULT(WARNING) << "The custom op domain name " << ptr->domain_ << " is already in session option.";
      }
    }
  } else {
    ORT_THROW("Please install TensorRT libraries as mentioned in the GPU requirements page, make sure they're in the PATH or LD_LIBRARY_PATH, and that your GPU is supported.");
  }
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
// Find a registered plugin EP device matching the given EP name and optional device_id from provider options.
// Returns nullptr if no matching device is found.
static const OrtEpDevice* FindRegisteredPluginEpDevice(
    const std::string& ep_name,
    const ProviderOptions* provider_options) {
  const auto& ep_devices = GetEnv().GetOrtEpDevices();
  if (ep_devices.empty()) {
    return nullptr;
  }

  bool has_requested_device_id = false;
  int requested_device_id = 0;
  if (provider_options != nullptr) {
    if (const auto device_id_it = provider_options->find("device_id"); device_id_it != provider_options->end()) {
      try {
        requested_device_id = std::stoi(device_id_it->second);
        has_requested_device_id = requested_device_id >= 0;
      } catch (const std::exception&) {
        LOGS_DEFAULT(WARNING) << "Invalid device_id value '" << device_id_it->second
                              << "' in provider options for EP '" << ep_name << "'; ignoring.";
      }
    }
  }

  for (const OrtEpDevice* ep_device : ep_devices) {
    if (!ep_device || ep_device->ep_name != ep_name) {
      continue;
    }

    if (has_requested_device_id) {
      Ort::ConstEpDevice current_device(ep_device);
      std::optional<int> current_device_id{};
      if (const char* device_id = current_device.EpOptions().GetValue("device_id"); device_id != nullptr) {
        try {
          current_device_id = std::stoi(device_id);
        } catch (const std::exception&) {
        }
      }

      if (!current_device_id.has_value()) {
        if (const char* device_id = current_device.EpMetadata().GetValue("cuda_device_id"); device_id != nullptr) {
          try {
            current_device_id = std::stoi(device_id);
          } catch (const std::exception&) {
          }
        }
      }

      if (!current_device_id.has_value()) {
        current_device_id = static_cast<int>(current_device.Device().DeviceId());
      }

      if (*current_device_id != requested_device_id) {
        continue;
      }
    }

    return ep_device;
  }

  return nullptr;
}
#endif

/**
 * Creates an IExecutionProviderFactory instance of the specified type.
 * @param session_options The session options.
 * @param type The execution provider type (e.g., CUDAExecutionProvider).
 * @param provider_options_map A map of provider options.
 *
 * @return A shared_ptr with the factory instance, or null if unable to create it.
 */
static std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactoryInstance(
    const SessionOptions& session_options,
    const std::string& type,
    const ProviderOptionsMap& provider_options_map) {
#if !defined(ORT_MINIMAL_BUILD)
  auto get_registered_plugin_ep_devices = [&]() -> InlinedVector<const OrtEpDevice*> {
    InlinedVector<const OrtEpDevice*> selected_devices;

    const ProviderOptions* provider_options = nullptr;
    if (const auto provider_it = provider_options_map.find(type); provider_it != provider_options_map.end()) {
      provider_options = &provider_it->second;
    }

    const OrtEpDevice* selected_device = FindRegisteredPluginEpDevice(type, provider_options);
    if (selected_device == nullptr) {
      if (provider_options != nullptr) {
        if (const auto device_id_it = provider_options->find("device_id"); device_id_it != provider_options->end()) {
          LOGS_DEFAULT(WARNING) << "No registered plugin EP device found for '" << type
                                << "' with device_id=" << device_id_it->second;
        }
      }
      return selected_devices;
    }

    selected_devices.push_back(selected_device);
    return selected_devices;
  };

  auto try_create_registered_plugin_factory = [&]() -> std::shared_ptr<IExecutionProviderFactory> {
    auto selected_devices = get_registered_plugin_ep_devices();
    if (selected_devices.empty()) {
      return nullptr;
    }

    std::unique_ptr<IExecutionProviderFactory> ep_factory;
    const auto status = onnxruntime::CreateIExecutionProviderFactoryForEpDevices(GetEnv(), selected_devices, ep_factory);
    if (!status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to create dynamic EP factory for '" << type
                            << "' from registered EP devices: " << status;
      return nullptr;
    }

    return std::shared_ptr<IExecutionProviderFactory>(std::move(ep_factory));
  };

  if (type == kCudaExecutionProvider) {
    if (auto ep_factory = try_create_registered_plugin_factory(); ep_factory) {
      return ep_factory;
    }
  }
#endif

  if (type == kCpuExecutionProvider) {
    return onnxruntime::CPUProviderFactoryCreator::Create(
        session_options.enable_cpu_mem_arena);
  } else if (type == kTensorrtExecutionProvider) {
#if defined(USE_TENSORRT) || defined(USE_TENSORRT_PROVIDER_INTERFACE)
    // If the environment variable 'ORT_TENSORRT_UNAVAILABLE' exists, then we do not load TensorRT. This is set by _ld_preload for the manylinux case
    // as in that case, trying to load the library itself will result in a crash due to the way that auditwheel strips dependencies.
    if (Env::Default().GetEnvironmentVar("ORT_TENSORRT_UNAVAILABLE").empty()) {
      // provider_options_map is just a reference to the ProviderOptionsMap instance, so it can be released anytime from application.
      // So we need these std::string variables defined here as they will be kept alive for the lifetime of TRT EP and we can still access them from OrtTensorRTProviderOptionsV2 instance.
      // (The reason is string copy is involved, for example params.trt_engine_cache_path = cache_path.c_str() and those std::string variable is referenced by OrtTensorRTProviderOptionsV2 instance
      // and TRT EP instance, so it won't be released.)
      std::string calibration_table, cache_path, cache_prefix, timing_cache_path, lib_path, trt_tactic_sources,
          trt_extra_plugin_lib_paths, min_profile, max_profile, opt_profile, ep_context_file_path,
          onnx_model_folder_path, trt_op_types_to_exclude, preview_features;
      auto it = provider_options_map.find(type);
      if (it != provider_options_map.end()) {
        OrtTensorRTProviderOptionsV2 params;
        for (auto option : it->second) {
          if (option.first == "device_id") {
            if (!option.second.empty()) {
              params.device_id = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'device_id' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "user_compute_stream") {
            if (!option.second.empty()) {
              auto stream = std::stoull(option.second, nullptr, 0);
              params.user_compute_stream = reinterpret_cast<void*>(stream);
              params.has_user_compute_stream = true;
            } else {
              params.has_user_compute_stream = false;
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'user_compute_stream' should be a string to define the compute stream for the inference to run on.\n");
            }
          } else if (option.first == "trt_max_partition_iterations") {
            if (!option.second.empty()) {
              params.trt_max_partition_iterations = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_max_partition_iterations' should be a positive integer number i.e. '1000'.\n");
            }
          } else if (option.first == "trt_min_subgraph_size") {
            if (!option.second.empty()) {
              params.trt_min_subgraph_size = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_min_subgraph_size' should be a positive integer number i.e. '1'.\n");
            }
          } else if (option.first == "trt_max_workspace_size") {
            if (!option.second.empty()) {
              params.trt_max_workspace_size = std::stoull(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_max_workspace_size' should be a number in byte i.e. '1073741824'.\n");
            }
          } else if (option.first == "trt_fp16_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_fp16_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_fp16_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_fp16_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_bf16_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_bf16_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_bf16_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_bf16_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_int8_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_int8_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_int8_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_int8_calibration_table_name") {
            if (!option.second.empty()) {
              calibration_table = option.second;
              params.trt_int8_calibration_table_name = calibration_table.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_calibration_table_name' should be a file name i.e. 'cal_table'.\n");
            }
          } else if (option.first == "trt_int8_use_native_calibration_table") {
            if (option.second == "True" || option.second == "true") {
              params.trt_int8_use_native_calibration_table = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_int8_use_native_calibration_table = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_int8_use_native_calibration_table' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_dla_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_dla_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_dla_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dla_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_dla_core") {
            if (!option.second.empty()) {
              params.trt_dla_core = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dla_core' should be a positive integer number i.e. '0'.\n");
            }
          } else if (option.first == "trt_dump_subgraphs") {
            if (option.second == "True" || option.second == "true") {
              params.trt_dump_subgraphs = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_dump_subgraphs = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dump_subgraphs' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_cache_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_engine_cache_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_engine_cache_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_cache_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_cache_path") {
            if (!option.second.empty()) {
              cache_path = option.second;
              params.trt_engine_cache_path = cache_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_cache_path' should be a path string i.e. 'engine_cache'.\n");
            }
          } else if (option.first == "trt_engine_cache_prefix") {
            if (!option.second.empty()) {
              cache_prefix = option.second;
              params.trt_engine_cache_prefix = cache_prefix.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_cache_prefix' should be a string to customize engine cache prefix i.e. 'FRCNN' or 'yolov4'.\n");
            }
          } else if (option.first == "trt_weight_stripped_engine_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_weight_stripped_engine_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_weight_stripped_engine_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_weight_stripped_engine_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_onnx_model_folder_path") {
            if (!option.second.empty()) {
              onnx_model_folder_path = option.second;
              params.trt_onnx_model_folder_path = onnx_model_folder_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_onnx_model_folder_path' should be a path string i.e. 'engine_cache'.\n");
            }
          } else if (option.first == "trt_engine_decryption_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_engine_decryption_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_engine_decryption_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_decryption_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_engine_decryption_lib_path") {
            if (!option.second.empty()) {
              lib_path = option.second;
              params.trt_engine_decryption_lib_path = lib_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_decryption_lib_path' should be a path string i.e. 'decryption_lib'.\n");
            }
          } else if (option.first == "trt_force_sequential_engine_build") {
            if (option.second == "True" || option.second == "true") {
              params.trt_force_sequential_engine_build = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_force_sequential_engine_build = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_force_sequential_engine_build' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_context_memory_sharing_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_context_memory_sharing_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_context_memory_sharing_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_context_memory_sharing_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_layer_norm_fp32_fallback") {
            if (option.second == "True" || option.second == "true") {
              params.trt_layer_norm_fp32_fallback = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_layer_norm_fp32_fallback = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_layer_norm_fp32_fallback' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_timing_cache_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_timing_cache_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_timing_cache_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_timing_cache_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_timing_cache_path") {
            if (!option.second.empty()) {
              timing_cache_path = option.second;
              params.trt_timing_cache_path = timing_cache_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_timing_cache_path' should be a path string i.e. 'cache_folder/'.\n");
            }
          } else if (option.first == "trt_force_timing_cache") {
            if (option.second == "True" || option.second == "true") {
              params.trt_force_timing_cache = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_force_timing_cache = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_force_timing_cache' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_detailed_build_log") {
            if (option.second == "True" || option.second == "true") {
              params.trt_detailed_build_log = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_detailed_build_log = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_detailed_build_log' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_build_heuristics_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_build_heuristics_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_build_heuristics_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_build_heuristics_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_sparsity_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_sparsity_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_sparsity_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_sparsity_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_builder_optimization_level") {
            if (!option.second.empty()) {
              params.trt_builder_optimization_level = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_builder_optimization_level' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "trt_auxiliary_streams") {
            if (!option.second.empty()) {
              params.trt_auxiliary_streams = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_auxiliary_streams' should be a number i.e. '0'.\n");
            }
          } else if (option.first == "trt_tactic_sources") {
            if (!option.second.empty()) {
              trt_tactic_sources = option.second;
              params.trt_tactic_sources = trt_tactic_sources.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_tactic_sources' should be a string. e.g. \"-CUDNN,+CUBLAS\" available keys: \"CUBLAS\"|\"CUBLAS_LT\"|\"CUDNN\"|\"EDGE_MASK_CONVOLUTIONS\".\n");
            }
          } else if (option.first == "trt_extra_plugin_lib_paths") {
            if (!option.second.empty()) {
              trt_extra_plugin_lib_paths = option.second;
              params.trt_extra_plugin_lib_paths = trt_extra_plugin_lib_paths.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_extra_plugin_lib_paths' should be a path string.\n");
            }
          } else if (option.first == "trt_profile_min_shapes") {
            if (!option.second.empty()) {
              min_profile = option.second;
              params.trt_profile_min_shapes = min_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_min_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_profile_max_shapes") {
            if (!option.second.empty()) {
              max_profile = option.second;
              params.trt_profile_max_shapes = max_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_max_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_profile_opt_shapes") {
            if (!option.second.empty()) {
              opt_profile = option.second;
              params.trt_profile_opt_shapes = opt_profile.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_profile_opt_shapes' should be a string of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'.\n");
            }
          } else if (option.first == "trt_cuda_graph_enable") {
            if (option.second == "True" || option.second == "true") {
              params.trt_cuda_graph_enable = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_cuda_graph_enable = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_cuda_graph_enable' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_dump_ep_context_model") {
            if (option.second == "True" || option.second == "true") {
              params.trt_dump_ep_context_model = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_dump_ep_context_model = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_dump_ep_context_model' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_ep_context_file_path") {
            if (!option.second.empty()) {
              ep_context_file_path = option.second;
              params.trt_ep_context_file_path = ep_context_file_path.c_str();
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_ep_context_file_path' should be a string.\n");
            }
          } else if (option.first == "trt_ep_context_embed_mode") {
            if (!option.second.empty()) {
              params.trt_ep_context_embed_mode = std::stoi(option.second);
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_ep_context_embed_mode' should be a positive integer number i.e. '1'.\n");
            }
          } else if (option.first == "trt_engine_hw_compatible") {
            if (option.second == "True" || option.second == "true") {
              params.trt_engine_hw_compatible = true;
            } else if (option.second == "False" || option.second == "false") {
              params.trt_engine_hw_compatible = false;
            } else {
              ORT_THROW("[ERROR] [TensorRT] The value for the key 'trt_engine_hw_compatible' should be 'True' or 'False'. Default value is 'False'.\n");
            }
          } else if (option.first == "trt_op_types_to_exclude") {
            trt_op_types_to_exclude = option.second;
            params.trt_op_types_to_exclude = trt_op_types_to_exclude.c_str();
          } else if (option.first == "trt_preview_features") {
            if (!option.second.empty()) {
              preview_features = option.second;
              params.trt_preview_features = preview_features.c_str();
            }
          } else {
            ORT_THROW("Invalid TensorRT EP option: ", option.first);
          }
        }
        if (std::shared_ptr<IExecutionProviderFactory> tensorrt_provider_factory = onnxruntime::TensorrtProviderFactoryCreator::Create(&params)) {
          return tensorrt_provider_factory;
        }
      } else {
        if (std::shared_ptr<IExecutionProviderFactory> tensorrt_provider_factory = onnxruntime::TensorrtProviderFactoryCreator::Create(cuda_device_id)) {
          return tensorrt_provider_factory;
        }
      }
    }
    LOGS_DEFAULT(WARNING) << "Failed to create "
                          << type
                          << ". Please reference "
                          << "https://onnxruntime.ai/docs/execution-providers/"
                          << "TensorRT-ExecutionProvider.html#requirements to ensure all dependencies are met.";
#endif

  } else if (type == kMIGraphXExecutionProvider) {
#if defined(USE_MIGRAPHX) || defined(USE_MIGRAPHX_PROVIDER_INTERFACE)
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      if (std::shared_ptr<IExecutionProviderFactory> migraphx_provider_factory =
              onnxruntime::MIGraphXProviderFactoryCreator::Create(it->second)) {
        return migraphx_provider_factory;
      }
    } else {
      if (std::shared_ptr<IExecutionProviderFactory> migraphx_provider_factory =
              onnxruntime::MIGraphXProviderFactoryCreator::Create(cuda_device_id)) {
        return migraphx_provider_factory;
      }
    }
#endif
  } else if (type == kCudaExecutionProvider) {
#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
    // If the environment variable 'CUDA_UNAVAILABLE' exists, then we do not load cuda.
    // This is set by _ld_preload for the manylinux case as in that case,
    // trying to load the library itself will result in a crash due to the way that auditwheel strips dependencies.
    if (Env::Default().GetEnvironmentVar("ORT_CUDA_UNAVAILABLE").empty()) {
      if (auto* cuda_provider_info = TryGetProviderInfo_CUDA()) {
        const CUDAExecutionProviderInfo info = GetCudaExecutionProviderInfo(cuda_provider_info,
                                                                            provider_options_map);

        // This variable is never initialized because the APIs by which it should be initialized are deprecated,
        // however they still exist are are in-use. Nevertheless, it is used to return CUDAAllocator,
        // hence we must try to initialize it here if we can since FromProviderOptions might contain
        // external CUDA allocator.
        external_allocator_info = info.external_allocator_info;
        return cuda_provider_info->CreateExecutionProviderFactory(info);
      }
    }
#if defined(USE_CUDA)
    LOGS_DEFAULT(WARNING) << "Failed to create "
                          << type
                          << ". Require cuDNN " << CUDNN_MAJOR << ".* and "
                          << "CUDA " << (CUDA_VERSION / 1000) << ".*"
#if defined(_MSC_VER)
                          << ", and the latest MSVC runtime"
#endif
                          << ". Please install all dependencies as mentioned in the GPU requirements page"
                             " (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), "
                             "make sure they're in the PATH, and that your GPU is supported.";
#elif defined(USE_CUDA_PROVIDER_INTERFACE)
    // Can't include "cuda.h", so don't print version info.
    LOGS_DEFAULT(WARNING) << "Failed to create "
                          << type
                          << ". Please install all dependencies as mentioned in the GPU requirements page"
                             " (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), "
                             "make sure they're in the PATH, and that your GPU is supported.";
#endif  // defined(USE_CUDA)
#endif  // defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
  } else if (type == kDnnlExecutionProvider) {
#ifdef USE_DNNL
    // Generate dnnl_options
    OrtDnnlProviderOptions dnnl_options;
// For Eigen and OpenMP
#if defined(DNNL_OPENMP)
    int num_threads = 0;
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "num_of_threads") {
          num_threads = std::stoi(option.second);
          if (num_threads < 0) {
            ORT_THROW(
                "[ERROR] [OneDNN] Invalid entry for the key 'num_of_threads',"
                " set number of threads or use '0' for default\n");
            // If the user doesnt define num_threads, auto detect threads later
          }
        } else {
          ORT_THROW("Invalid OneDNN EP option: ", option.first);
        }
      }
    }
    dnnl_options.threadpool_args = static_cast<void*>(&num_threads);
#endif  // !defined(DNNL_ORT_THREAD)
    dnnl_options.use_arena = session_options.enable_cpu_mem_arena;

    return onnxruntime::DnnlProviderFactoryCreator::Create(&dnnl_options);
#endif
  } else if (type == kOpenVINOExecutionProvider) {
#if defined(USE_OPENVINO) || defined(USE_OPENVINO_PROVIDER_INTERFACE)
    ProviderOptions OV_provider_options_map;
    const std::unordered_set<std::string> valid_provider_keys = {"device_type", "device_id", "device_luid", "cache_dir", "precision",
                                                                 "load_config", "context", "num_of_threads", "model_priority", "num_streams", "enable_opencl_throttling", "enable_qdq_optimizer",
                                                                 "enable_causallm", "disable_dynamic_shapes", "reshape_input", "layout"};
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (valid_provider_keys.count(option.first)) {
          OV_provider_options_map[option.first] = option.second;
          continue;
        } else {
          ORT_THROW("Invalid OpenVINO EP option: ", option.first);
        }
      }
    }
    if (std::shared_ptr<IExecutionProviderFactory> openvino_provider_factory = onnxruntime::OpenVINOProviderFactoryCreator::Create(
            &OV_provider_options_map, &session_options)) {
      // Reset global variables config to avoid it being accidentally passed on to the next session
      openvino_device_type.clear();
      return openvino_provider_factory;
    } else {
      if (!Env::Default().GetEnvironmentVar("INTEL_OPENVINO_DIR").empty()) {
        ORT_THROW("INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded. Please install a supported version of OpenVINO as mentioned in the requirements page (https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements), ensure dependency libraries are in the PATH and your hardware is supported.");
      } else {
        LOGS_DEFAULT(WARNING) << "Failed to create " << type << ". Please refer https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements to ensure all dependencies are met.";
      }
    }
#endif
  } else if (type == kVitisAIExecutionProvider) {
#if defined(USE_VITISAI) || defined(USE_VITISAI_PROVIDER_INTERFACE)
    ProviderOptions info{};
    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      info = it->second;
    }
    info["session_options"] = std::to_string((uintptr_t)(void*)&session_options);
    if (auto vitisai_factory = onnxruntime::VitisAIProviderFactoryCreator::Create(info); vitisai_factory) {
      return vitisai_factory;
    }
    LOGS_DEFAULT(WARNING) << "Failed to create "
                          << type
                          << ". Please reference "
                          << "https://onnxruntime.ai/docs/execution-providers/"
                          << "Vitis-AI-ExecutionProvider.html#requirements to ensure all dependencies are met.";
#endif
  } else if (type == kAclExecutionProvider) {
#ifdef USE_ACL
    bool enable_fast_math = false;
    auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      for (auto option : it->second) {
        if (option.first == "enable_fast_math") {
          std::set<std::string> supported_values = {"true", "True", "false", "False"};
          if (supported_values.find(option.second) != supported_values.end()) {
            enable_fast_math = (option.second == "true") || (option.second == "True");
          } else {
            ORT_THROW(
                "Invalid value for enable_fast_math. "
                "Select from 'true' or 'false'\n");
          }
        } else {
          ORT_THROW("Unrecognized option: ", option.first);
        }
      }
    }
    return onnxruntime::ACLProviderFactoryCreator::Create(enable_fast_math);
#endif
  } else if (type == kDmlExecutionProvider) {
#ifdef USE_DML
    auto cit = provider_options_map.find(type);
    return onnxruntime::DMLProviderFactoryCreator::CreateFromProviderOptions(
        session_options.config_options, cit == provider_options_map.end() ? ProviderOptions{} : cit->second, true);
#endif
  } else if (type == kNnapiExecutionProvider) {
#if defined(USE_NNAPI)
#if !defined(__ANDROID__)
    LOGS_DEFAULT(WARNING) << "NNAPI execution provider can only be used to generate ORT format model in this build.";
#endif
    const auto partitioning_stop_ops_list = session_options.config_options.GetConfigEntry(
        kOrtSessionOptionsConfigNnapiEpPartitioningStopOps);
    return onnxruntime::NnapiProviderFactoryCreator::Create(0, partitioning_stop_ops_list);
#endif
  } else if (type == kVSINPUExecutionProvider) {
#ifdef USE_VSINPU
    return onnxruntime::VSINPUProviderFactoryCreator::Create();
#endif
  } else if (type == kRknpuExecutionProvider) {
#ifdef USE_RKNPU
    return onnxruntime::RknpuProviderFactoryCreator::Create();
#endif
  } else if (type == kCoreMLExecutionProvider) {
#if defined(USE_COREML)
#if !defined(__APPLE__)
    LOGS_DEFAULT(WARNING) << "CoreML execution provider can only be used to generate ORT format model in this build.";
#endif
    uint32_t coreml_flags = 0;

    const auto it = provider_options_map.find(type);
    if (it != provider_options_map.end()) {
      const ProviderOptions& options = it->second;
      auto flags = options.find("flags");
      if (flags != options.end()) {
        const auto& flags_str = flags->second;

        if (flags_str.find("COREML_FLAG_USE_CPU_ONLY") != std::string::npos) {
          coreml_flags |= COREMLFlags::COREML_FLAG_USE_CPU_ONLY;
        } else if (flags_str.find("COREML_FLAG_USE_CPU_AND_GPU") != std::string::npos) {
          coreml_flags |= COREMLFlags::COREML_FLAG_USE_CPU_AND_GPU;
        }

        if (flags_str.find("COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES") != std::string::npos) {
          coreml_flags |= COREMLFlags::COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES;
        }

        if (flags_str.find("COREML_FLAG_CREATE_MLPROGRAM") != std::string::npos) {
          coreml_flags |= COREMLFlags::COREML_FLAG_CREATE_MLPROGRAM;
        }
      } else {
        // read from provider_options
        return onnxruntime::CoreMLProviderFactoryCreator::Create(options);
      }
    }

    return onnxruntime::CoreMLProviderFactoryCreator::Create(coreml_flags);
#endif
  } else if (type == kXnnpackExecutionProvider) {
#if defined(USE_XNNPACK)
    auto cit = provider_options_map.find(type);
    return onnxruntime::XnnpackProviderFactoryCreator::Create(
        cit == provider_options_map.end() ? ProviderOptions{} : cit->second, &session_options);
#endif
  } else if (type == kWebGpuExecutionProvider) {
#if !defined(ORT_MINIMAL_BUILD)
    if (auto ep_factory = try_create_registered_plugin_factory(); ep_factory) {
      return ep_factory;
    }
#endif
#if defined(USE_WEBGPU)
    return onnxruntime::WebGpuProviderFactoryCreator::Create(session_options.config_options);
#endif
  } else if (type == kCannExecutionProvider) {
#ifdef USE_CANN
    if (auto* cann_provider_info = TryGetProviderInfo_CANN()) {
      const CANNExecutionProviderInfo info = GetCannExecutionProviderInfo(cann_provider_info,
                                                                          provider_options_map);
      return cann_provider_info->CreateExecutionProviderFactory(info);
    } else {
      ORT_THROW("create CANN ExecutionProvider fail");
    }
#endif
  } else if (type == kAzureExecutionProvider) {
#ifdef USE_AZURE
    return onnxruntime::AzureProviderFactoryCreator::Create({});
#endif
  } else if (type == kQnnExecutionProvider) {
#if defined(USE_QNN) || defined(USE_QNN_PROVIDER_INTERFACE)
    auto cit = provider_options_map.find(type);
    auto qnn_factory = onnxruntime::QNNProviderFactoryCreator::Create(
        cit == provider_options_map.end() ? ProviderOptions{} : cit->second, &session_options);
    if (qnn_factory) {
      return qnn_factory;
    }
    LOGS_DEFAULT(WARNING) << "Failed to create "
                          << type
                          << ". Please reference "
                          << "https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html"
                          << " to ensure all dependencies are met.";
#endif
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    // Try EPs dynamically registered via register_execution_provider_library().
    if (auto ep_factory = try_create_registered_plugin_factory(); ep_factory) {
      return ep_factory;
    }
#endif

    // unknown provider
    throw std::runtime_error("Unknown Provider Type: " + type);
  }
  return nullptr;
}

/**
 * Create an IExecutionProvider instance of the specified type. Note: this is called by orttraining code.
 * @param session_options The session options.
 * @param type The execution provider type (e.g., CUDAExecutionProvider).
 * @param provider_options_map A map of provider options.
 *
 * @return A unique_ptr with the execution provider instance, or null if unable to create it.
 */
std::unique_ptr<IExecutionProvider> CreateExecutionProviderInstance(const SessionOptions& session_options,
                                                                    const std::string& type,
                                                                    const ProviderOptionsMap& provider_options_map) {
  auto ep_factory = CreateExecutionProviderFactoryInstance(session_options, type, provider_options_map);
  if (ep_factory) {
    const auto& default_logger = GetEnv().GetLoggingManager()->DefaultLogger();
    OrtSessionOptions ort_session_options;
    ort_session_options.value = session_options;

#if !defined(ORT_MINIMAL_BUILD)
    auto add_registered_plugin_ep_options_to_session = [&]() -> Status {
      const ProviderOptions* provider_options = nullptr;
      if (const auto provider_it = provider_options_map.find(type); provider_it != provider_options_map.end()) {
        provider_options = &provider_it->second;
      }

      if (provider_options == nullptr || provider_options->empty()) {
        return Status::OK();
      }

      const OrtEpDevice* selected_device = FindRegisteredPluginEpDevice(type, provider_options);
      if (selected_device == nullptr) {
        return Status::OK();
      }

      InlinedVector<const OrtEpDevice*> selected_devices;
      selected_devices.push_back(selected_device);

      auto adapted_options = AdaptProviderOptionsForRegisteredPluginEp(type, *provider_options);
      return AddEpOptionsToSessionOptions(selected_devices,
                                          adapted_options.key_ptrs,
                                          adapted_options.value_ptrs,
                                          ort_session_options.value);
    };

    auto status = add_registered_plugin_ep_options_to_session();
    if (!status.IsOK()) {
      ORT_THROW("Error applying registered plugin EP options: ", status);
    }
#endif

    return ep_factory->CreateProvider(ort_session_options, *default_logger.ToExternal());
  }
  return nullptr;
}

/**
 * Adds an explicit execution provider factory to the session options.
 *
 * @param py_sess_options The session options.
 * @param provider_type The type of the provider to add.
 * @param provider_options The options for the execution provider as a map of string key/value pairs.
 *
 * @return A Status indicating an error or success.
 */
static Status AddExplicitEpFactory(PySessionOptions& py_sess_options, const std::string& provider_type,
                                   const ProviderOptions& provider_options) {
  const ProviderOptionsMap provider_options_map = {{provider_type, provider_options}};
  auto ep_factory = CreateExecutionProviderFactoryInstance(py_sess_options.value, provider_type, provider_options_map);
  if (!ep_factory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to add provider of type '",
                           provider_type, "' to SessionOptions. Provider configuration is not supported.");
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (!provider_options.empty()) {
    const OrtEpDevice* selected_device = FindRegisteredPluginEpDevice(provider_type, &provider_options);
    if (selected_device != nullptr) {
      InlinedVector<const OrtEpDevice*> selected_devices;
      selected_devices.push_back(selected_device);

      auto adapted_options = AdaptProviderOptionsForRegisteredPluginEp(provider_type, provider_options);
      ORT_RETURN_IF_ERROR(AddEpOptionsToSessionOptions(selected_devices,
                                                       adapted_options.key_ptrs,
                                                       adapted_options.value_ptrs,
                                                       py_sess_options.value));
    }
  }
#endif

  py_sess_options.provider_factories.push_back(std::move(ep_factory));
  return Status::OK();
}

/**
 * Generate a map for mapping execution provider to excution provider options.
 *
 * @param providers vector of excution providers. [ep1, ep2, ...]
 * @param provider_options_vector vector of excution provider options. [option1, option2 ...]
 * @param provider_options_map an unordered map for mapping excution provider to excution provider options.
 *        {'ep1' -> option1, 'ep2' -> option2 ...}
 *
 */
static void GenerateProviderOptionsMap(const std::vector<std::string>& providers,
                                       const ProviderOptionsVector& provider_options_vector,
                                       ProviderOptionsMap& provider_options_map) {
  if (provider_options_vector.empty() || providers.empty()) {
    return;
  }

  std::size_t j = 0;  // index for provider_options_vector

  for (const std::string& type : providers) {
    if (j < provider_options_vector.size() && !provider_options_vector[j].empty()) {
      provider_options_map[type] = provider_options_vector[j];
    }

    j += 1;
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
static void RegisterCustomOpDomains(PyInferenceSession* sess, const OrtSessionOptions& so) {
  if (!so.custom_op_domains_.empty()) {
    // Register all custom op domains that will be needed for the session
    std::vector<OrtCustomOpDomain*> custom_op_domains;
    custom_op_domains.reserve(so.custom_op_domains_.size());
    for (size_t i = 0; i < so.custom_op_domains_.size(); ++i) {
      custom_op_domains.emplace_back(so.custom_op_domains_[i]);
    }
    OrtPybindThrowIfError(sess->GetSessionHandle()->AddCustomOpDomains(custom_op_domains));
  }
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
/**
 * Add the execution provider that is responsible for the selected OrtEpDevice instances to the session options.
 *
 * @param py_sess_options The session options.
 * @param provider_type The type of the provider to add.
 * @param provider_options The options for the execution provider as a map of string key/value pairs.
 *
 * @return A Status indicating an error or success.
 */
static Status AddEpFactoryFromEpDevices(PySessionOptions& py_sess_options,
                                        const std::vector<const OrtEpDevice*>& ep_devices,
                                        const ProviderOptions& provider_options) {
  onnxruntime::Environment& env = GetEnv();
  const auto ep_name = ep_devices.empty() ? std::string{} : ep_devices[0]->ep_name;
  auto adapted_options = AdaptProviderOptionsForRegisteredPluginEp(ep_name, provider_options);

  std::unique_ptr<IExecutionProviderFactory> provider_factory = nullptr;
  ORT_RETURN_IF_ERROR(CreateIExecutionProviderFactoryForEpDevices(env,
                                                                  ep_devices,
                                                                  /*output*/ provider_factory));

  ORT_RETURN_IF_ERROR(AddEpOptionsToSessionOptions(ep_devices,
                                                   adapted_options.key_ptrs,
                                                   adapted_options.value_ptrs,
                                                   py_sess_options.value));

  ORT_RETURN_IF_ERROR(AddEpCustomDomainsToSessionOptions(ep_devices,
                                                         py_sess_options));

  py_sess_options.provider_factories.push_back(std::move(provider_factory));
  return Status::OK();
}

/**
 * Initializes the inference session using EPs specified in the session options.
 *
 * @param py_sess The inference session.
 * @param disabled_optimizer_names Set of optimizers to disable.
 * @return A Status indicating error or success.
 */
static Status InitializeSessionEpsFromSessionOptions(PyInferenceSession& py_sess,
                                                     const std::unordered_set<std::string>& disabled_optimizer_names) {
  ORT_RETURN_IF(py_sess.GetSessionHandle() == nullptr, "Invalid Python InferenceSession handle");
  InferenceSession& sess = *py_sess.GetSessionHandle();

  const logging::Logger* sess_logger = sess.GetLogger();
  ORT_RETURN_IF(sess_logger == nullptr, "Invalid InferenceSession logger handle");

  const OrtSessionOptions& ort_session_options = py_sess.GetOrtSessionOptions();

  // if there are no providers registered, and there's an ep selection policy set, do auto ep selection
  if (ort_session_options.provider_factories.empty() && ort_session_options.value.ep_selection_policy.enable) {
    ProviderPolicyContext context;
    ORT_RETURN_IF_ERROR(context.SelectEpsForSession(GetEnv(), ort_session_options, sess));
  } else {
    for (const auto& provider_factory : ort_session_options.provider_factories) {
      std::unique_ptr<IExecutionProvider> ep = provider_factory->CreateProvider(ort_session_options,
                                                                                *(sess_logger->ToExternal()));
      if (ep) {
        ORT_RETURN_IF_ERROR(sess.RegisterExecutionProvider(std::move(ep)));
      }
    }
  }

  if (!disabled_optimizer_names.empty()) {
    ORT_RETURN_IF_ERROR(sess.FilterEnabledOptimizers({disabled_optimizer_names.cbegin(),
                                                      disabled_optimizer_names.cend()}));
  }

  ORT_RETURN_IF_ERROR(sess.Initialize());
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

void InitializeSession(InferenceSession* sess,
                       ExecutionProviderRegistrationFn ep_registration_fn,
                       const std::vector<std::string>& provider_types,
                       const ProviderOptionsVector& provider_options,
                       const std::unordered_set<std::string>& disabled_optimizer_names) {
  ProviderOptionsMap provider_options_map;
  GenerateProviderOptionsMap(provider_types, provider_options, provider_options_map);

  ep_registration_fn(sess, provider_types, provider_options_map);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (!disabled_optimizer_names.empty()) {
    OrtPybindThrowIfError(sess->FilterEnabledOptimizers({disabled_optimizer_names.cbegin(), disabled_optimizer_names.cend()}));
  }
#else
  ORT_UNUSED_PARAMETER(disabled_optimizer_names);
#endif

  OrtPybindThrowIfError(sess->Initialize());
}

bool CheckIfTensor(const std::vector<const NodeArg*>& def_list,
                   const std::string& name,
                   /*out*/ onnx::TypeProto& type_proto) {
  auto ret_it = std::find_if(std::begin(def_list), std::end(def_list),
                             [&name](const NodeArg* node_arg) { return name == node_arg->Name(); });
  if (ret_it == std::end(def_list)) {
    throw std::runtime_error("Failed to find NodeArg with name: " + name + " in the def list");
  }

  const auto* temp = (*ret_it)->TypeAsProto();
  if (!temp) {
    throw std::runtime_error("Corresponding type_proto is null");
  } else {
    type_proto = *temp;
  }

  return type_proto.has_tensor_type();
}

#if defined(USE_OPENVINO) || defined(USE_OPENVINO_PROVIDER_INTERFACE) || \
    defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
static void LogDeprecationWarning(
    const std::string& deprecated, const optional<std::string>& alternative = nullopt) {
  LOGS_DEFAULT(WARNING) << "This is DEPRECATED and will be removed in the future: " << deprecated;
  LOGS_DEFAULT_IF(alternative.has_value(), WARNING) << "As an alternative, use: " << *alternative;
}
#endif

void addGlobalMethods(py::module& m) {
  m.def("set_global_thread_pool_sizes", [](int intra_op_num_threads, int inter_op_num_threads) {
          static std::mutex global_thread_pool_mutex;
          OrtThreadingOptions to;
          to.intra_op_thread_pool_params.thread_pool_size = intra_op_num_threads;
          to.inter_op_thread_pool_params.thread_pool_size = inter_op_num_threads;
          std::lock_guard<std::mutex> lock(global_thread_pool_mutex);
          SetGlobalThreadingOptions(std::move(to)); },
        py::arg("intra_op_num_threads") = 0,  // Default value for intra_op_num_threads
        py::arg("inter_op_num_threads") = 0,  // Default value for inter_op_num_threads
        "Set the number of threads used by the global thread pools for intra and inter op parallelism.");
  m.def("get_default_session_options", &GetDefaultCPUSessionOptions, "Return a default session_options instance.");
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
  m.def(
      "get_device", []() -> std::string { return BACKEND_DEVICE; },
      "Return the device used to compute the prediction (CPU, MKL, ...)");
  m.def(
      "set_seed", [](const int64_t seed) { utils::SetRandomSeed(seed); },
      "Sets the seed used for random number generation in Onnxruntime.");
  m.def(
      "set_default_logger_severity", [](int severity) {
        ORT_ENFORCE(severity >= 0 && severity <= 4,
                    "Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
        logging::LoggingManager* default_logging_manager = GetEnv().GetLoggingManager();
        default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity));
      },
      "Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal");
  m.def(
      "set_default_logger_verbosity", [](int vlog_level) {
        logging::LoggingManager* default_logging_manager = GetEnv().GetLoggingManager();
        default_logging_manager->SetDefaultLoggerVerbosity(vlog_level);
      },
      "Sets the default logging verbosity level. To activate the verbose log, "
      "you need to set the default logging severity to 0:Verbose level.");
  m.def(
      "get_all_providers", []() -> const std::vector<std::string>& { return GetAllExecutionProviderNames(); },
      "Return list of Execution Providers that this version of Onnxruntime can support. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");
  m.def(
      "enable_telemetry_events", []() -> void { Env::Default().GetTelemetryProvider().EnableTelemetryEvents(); },
      "Enables platform-specific telemetry collection where applicable.");
  m.def(
      "disable_telemetry_events", []() -> void { Env::Default().GetTelemetryProvider().DisableTelemetryEvents(); },
      "Disables platform-specific telemetry collection.");
  m.def(
      "create_and_register_allocator", [](const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg = nullptr) -> void {
        auto st = GetEnv().CreateAndRegisterAllocator(mem_info, arena_cfg);
        if (!st.IsOK()) {
          throw std::runtime_error("Error when creating and registering allocator: " + st.ErrorMessage());
        }
      });
  m.def(
      "create_and_register_allocator_v2", [](const std::string& provider_type, const OrtMemoryInfo& mem_info, const ProviderOptions& options, const OrtArenaCfg* arena_cfg = nullptr) -> void {
        auto st = GetEnv().CreateAndRegisterAllocatorV2(provider_type, mem_info, options, arena_cfg);
        if (!st.IsOK()) {
          throw std::runtime_error("Error when creating and registering allocator in create_and_register_allocator_v2: " + st.ErrorMessage());
        }
      });
  m.def(
      "register_execution_provider_library",
      [](const std::string& registration_name, const PathString& library_path) -> void {
#if !defined(ORT_MINIMAL_BUILD)
        OrtPybindThrowIfError(GetEnv().RegisterExecutionProviderLibrary(registration_name, library_path.c_str()));
#else
        ORT_UNUSED_PARAMETER(registration_name);
        ORT_UNUSED_PARAMETER(library_path);
        ORT_THROW("Execution provider libraries are not supported in this build.");
#endif
      },
      R"pbdoc(Register an execution provider library with ONNX Runtime.)pbdoc");
  m.def(
      "unregister_execution_provider_library",
      [](const std::string& registration_name) -> void {
#if !defined(ORT_MINIMAL_BUILD)
        OrtPybindThrowIfError(GetEnv().UnregisterExecutionProviderLibrary(registration_name));
#else
        ORT_UNUSED_PARAMETER(registration_name);
        ORT_THROW("Execution provider libraries are not supported in this build.");
#endif
      },
      R"pbdoc(Unregister an execution provider library from ONNX Runtime.)pbdoc");
  m.def(
      "get_ep_devices",
      []() -> const std::vector<const OrtEpDevice*>& {
#if !defined(ORT_MINIMAL_BUILD)
        return GetEnv().GetOrtEpDevices();
#else
        ORT_THROW("OrtEpDevices are not supported in this build");
#endif
      },
      R"pbdoc(Get the list of available OrtEpDevice instances.)pbdoc",
      py::return_value_policy::reference);

  m.def(
      "get_model_compatibility_for_ep_devices",
      [](const std::vector<const OrtEpDevice*>& ep_devices,
         const std::string& compatibility_info) -> OrtCompiledModelCompatibility {
        OrtCompiledModelCompatibility status = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
        Ort::ThrowOnError(Ort::GetApi().GetModelCompatibilityForEpDevices(
            ep_devices.data(), ep_devices.size(), compatibility_info.c_str(), &status));
        return status;
      },
      R"pbdoc("Validate a compiled model's compatibility information for one or more EP devices.)pbdoc");

  m.def(
      "get_compatibility_info_from_model",
      [](const std::basic_string<ORTCHAR_T>& model_path, const std::string& ep_type) -> py::object {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr compat_info = Ort::GetCompatibilityInfoFromModelAllocated(
            model_path.c_str(), ep_type.c_str(), allocator);
        if (compat_info.get() == nullptr) {
          return py::none();
        }
        return py::str(compat_info.get());
      },
      R"pbdoc(Extract EP compatibility info from a precompiled model file.

Parses the model file to extract the compatibility info string for a specific execution provider
from the model's metadata properties. Returns None if no compatibility info exists for the EP.

Args:
    model_path: Path to the ONNX model file.
    ep_type: The execution provider type string (e.g. "CPUExecutionProvider").

Returns:
    The compatibility info string, or None if not found.
)pbdoc");

  m.def(
      "get_compatibility_info_from_model_bytes",
      [](const py::buffer& model_data, const std::string& ep_type) -> py::object {
        py::buffer_info info = model_data.request();
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr compat_info = Ort::GetCompatibilityInfoFromModelBytesAllocated(
            info.ptr, static_cast<size_t>(info.size * info.itemsize), ep_type.c_str(), allocator);
        if (compat_info.get() == nullptr) {
          return py::none();
        }
        return py::str(compat_info.get());
      },
      R"pbdoc(Extract EP compatibility info from precompiled model bytes in memory.

Same as get_compatibility_info_from_model but reads from a buffer instead of a file.
Accepts bytes, bytearray, memoryview, or any object supporting the buffer protocol.

Args:
    model_data: The model data as a buffer (bytes, bytearray, memoryview, etc.).
    ep_type: The execution provider type string (e.g. "CPUExecutionProvider").

Returns:
    The compatibility info string, or None if not found.
)pbdoc");

  m.def(
      "get_hardware_devices",
      []() -> const std::vector<const OrtHardwareDevice*>& {
        return GetEnv().GetSortedOrtHardwareDevices();
      },
      R"pbdoc(Get the list of available hardware devices.)pbdoc",
      py::return_value_policy::reference);

  m.def(
      "get_hardware_device_ep_incompatibility_details",
      [](const std::string& ep_name,
         const OrtHardwareDevice* hw) -> py::dict {
        std::unique_ptr<OrtDeviceEpIncompatibilityDetails> details;
        OrtPybindThrowIfError(GetEnv().GetHardwareDeviceEpIncompatibilityDetails(ep_name, hw, details));

        py::dict result;
        result["reasons_bitmask"] = details->reasons_bitmask;
        if (!details->notes.empty()) {
          result["notes"] = py::str(details->notes);
        } else {
          result["notes"] = py::none();
        }
        result["error_code"] = details->error_code;
        return result;
      },
      R"pbdoc(Check for known incompatibility issues between hardware device and a specific execution provider.

Returns a dictionary with:
  - reasons_bitmask: Bitmask of OrtDeviceEpIncompatibilityReason values (0 = no known incompatibility)
  - notes: Optional human-readable notes about the incompatibility
  - error_code: EP-specific error code (0 = no error code set)
)pbdoc");

  m.def(
      "copy_tensors",
      [](const std::vector<const OrtValue*>& src, const std::vector<OrtValue*>& dest, py::object& py_arg) {
        const OrtEnv* ort_env = GetOrtEnv();
        OrtSyncStream* stream = nullptr;
        if (!py_arg.is_none()) {
          stream = py_arg.cast<OrtSyncStream*>();
        }
        Ort::ThrowOnError(Ort::GetApi().CopyTensors(ort_env, src.data(), dest.data(), stream, src.size()));
      },
      R"pbdoc("Copy tensors from sources to destinations using specified stream handle (or None))pbdoc");

#if defined(USE_OPENVINO) || defined(USE_OPENVINO_PROVIDER_INTERFACE)
  m.def(
      "get_available_openvino_device_ids", []() -> std::vector<std::string> {
        if (auto* info = TryGetProviderInfo_OpenVINO()) {
          return info->GetAvailableDevices();
        }
        return {};
      },
      "Lists all OpenVINO device ids available.");
  /*
   * The following APIs to set config options are deprecated. Use Session.set_providers() instead.
   */
  // TODO remove deprecated global config
  m.def(
      "set_openvino_device", [](const std::string& device_type) {
        LogDeprecationWarning("set_openvino_device", "OpenVINO execution provider option \"device_type\"");
        openvino_device_type = device_type;
      },
      "Set the preferred OpenVINO device type to be used. If left unset, "
      "the device type selected during build time will be used.");
  // TODO remove deprecated global config
  m.def(
      "get_openvino_device", []() -> std::string {
        LogDeprecationWarning("get_openvino_device");
        return openvino_device_type;
      },
      "Gets the dynamically selected OpenVINO device type for inference.");
#endif

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
  /*
   * The following set_* methods are deprecated.
   *
   * To achieve same result, please use the following python api:
   * InferenceSession.set_providers(list_of_providers, list_of_provider_option_dicts)
   *
   */
  // TODO remove deprecated global config
  m.def("set_cuda_device_id", [](const int id) {
    LogDeprecationWarning("set_cuda_device_id", "CUDA execution provider option \"device_id\"");
    cuda_device_id = static_cast<OrtDevice::DeviceId>(id);
  });
  // TODO remove deprecated global config
  m.def("set_cudnn_conv_algo_search", [](const OrtCudnnConvAlgoSearch algo) {
    LogDeprecationWarning("set_cudnn_conv_algo_search", "CUDA execution provider option \"cudnn_conv_algo_search\"");
    cudnn_conv_algo_search = algo;
  });
  // TODO remove deprecated global config
  m.def("set_do_copy_in_default_stream", [](const bool use_single_stream) {
    LogDeprecationWarning(
        "set_do_copy_in_default_stream", "CUDA execution provider option \"do_copy_in_default_stream\"");
    do_copy_in_default_stream = use_single_stream;
  });
  // TODO remove deprecated global config
  m.def("set_gpu_mem_limit", [](const int64_t limit) {
    LogDeprecationWarning(
        "set_gpu_mem_limit",
        "CUDA execution provider option \"gpu_mem_limit\"");
    gpu_mem_limit = gsl::narrow<size_t>(limit);
  });
  // TODO remove deprecated global config
  m.def("set_arena_extend_strategy", [](const onnxruntime::ArenaExtendStrategy strategy) {
    LogDeprecationWarning("set_arena_extend_strategy", "CUDA execution provider option \"arena_extend_strategy\"");
    arena_extend_strategy = strategy;
  });
#endif

#if defined(USE_TENSORRT) || defined(USE_TENSORRT_PROVIDER_INTERFACE)
  m.def(
      "register_tensorrt_plugins_as_custom_ops", [](PySessionOptions& so, const ProviderOptions& options) { RegisterTensorRTPluginsAsCustomOps(so, options); },
      "Register TensorRT plugins as custom ops.");
#endif

#ifdef ENABLE_ATEN
  m.def("register_aten_op_executor",
        [](const std::string& is_tensor_argument_address_str, const std::string& aten_op_executor_address_str) -> void {
          size_t is_tensor_argument_address_int, aten_op_executor_address_int;
          ORT_THROW_IF_ERROR(
              ParseStringWithClassicLocale(is_tensor_argument_address_str, is_tensor_argument_address_int));
          ORT_THROW_IF_ERROR(ParseStringWithClassicLocale(aten_op_executor_address_str, aten_op_executor_address_int));
          void* p_is_tensor_argument = reinterpret_cast<void*>(is_tensor_argument_address_int);
          void* p_aten_op_executor = reinterpret_cast<void*>(aten_op_executor_address_int);
          contrib::aten_ops::ATenOperatorExecutor::Instance().Initialize(p_is_tensor_argument, p_aten_op_executor);
        });
#endif
}

#if !defined(ORT_MINIMAL_BUILD)
/**
 * Calls the user's Python EP selection function and converts the results to a format that can be used
 * by ORT to select OrtEpDevice instances. The user's function is set by calling
 * SessionOptions.set_provider_selection_policy_delegate() on the Python side. The result of this wrapper
 * function is used in core/session/provider_policy_context.cc.
 *
 * @param ep_devices OrtEpDevices to select from.
 * @param num_devices Number of OrtEpDevices to select from.
 * @param model_metadata Model's metadata.
 * @param runtime_metadata Runtime metadata.
 * @param selected Pre-allocated OrtEpDevice buffer to update with selected devices.
 * @param max_selected Maximum number of entries in the pre-allocated 'selected' buffer.
 * @param state Opaque state that holds a pointer to the user's Python function.
 *
 * @return nullptr OrtStatus* to indicate success.
 */
static OrtStatus* ORT_API_CALL PyEpSelectionPolicyWrapper(_In_ const OrtEpDevice** ep_devices,
                                                          _In_ size_t num_devices,
                                                          _In_ const OrtKeyValuePairs* model_metadata,
                                                          _In_opt_ const OrtKeyValuePairs* runtime_metadata,
                                                          _Inout_ const OrtEpDevice** selected,
                                                          _In_ size_t max_selected,
                                                          _Out_ size_t* num_selected,
                                                          _In_ void* state) {
  *num_selected = 0;
  OrtStatus* status = nullptr;

  // Call the Python delegate function and convert any exceptions to a status.
  ORT_TRY {
    auto* registration = static_cast<PyEpSelectionRegistration*>(state);
    std::vector<const OrtEpDevice*> py_ep_devices(ep_devices, ep_devices + num_devices);
    std::map<std::string, std::string> py_model_metadata =
        model_metadata ? model_metadata->Entries() : std::map<std::string, std::string>{};
    std::map<std::string, std::string> py_runtime_metadata =
        runtime_metadata ? runtime_metadata->Entries() : std::map<std::string, std::string>{};

    py::gil_scoped_acquire acquire;
    auto py_selected = registration->delegate(py_ep_devices, py_model_metadata, py_runtime_metadata, max_selected);
    if (py_selected.size() > max_selected) {
      return ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "selected too many EP devices (", py_selected.size(), "). ",
                                         "The limit is ", max_selected, " EP devices."));
    }

    *num_selected = py_selected.size();
    for (size_t i = 0; i < py_selected.size(); ++i) {
      selected[i] = py_selected[i];
    }
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what()));
    });
  }

  return status;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

void addObjectMethods(py::module& m, ExecutionProviderRegistrationFn ep_registration_fn) {
  py::enum_<GraphOptimizationLevel>(m, "GraphOptimizationLevel")
      .value("ORT_DISABLE_ALL", GraphOptimizationLevel::ORT_DISABLE_ALL)
      .value("ORT_ENABLE_BASIC", GraphOptimizationLevel::ORT_ENABLE_BASIC)
      .value("ORT_ENABLE_EXTENDED", GraphOptimizationLevel::ORT_ENABLE_EXTENDED)
      .value("ORT_ENABLE_LAYOUT", GraphOptimizationLevel::ORT_ENABLE_LAYOUT)
      .value("ORT_ENABLE_ALL", GraphOptimizationLevel::ORT_ENABLE_ALL);

  py::enum_<ExecutionMode>(m, "ExecutionMode")
      .value("ORT_SEQUENTIAL", ExecutionMode::ORT_SEQUENTIAL)
      .value("ORT_PARALLEL", ExecutionMode::ORT_PARALLEL);

  py::enum_<ExecutionOrder>(m, "ExecutionOrder")
      .value("DEFAULT", ExecutionOrder::DEFAULT)
      .value("PRIORITY_BASED", ExecutionOrder::PRIORITY_BASED)
      .value("MEMORY_EFFICIENT", ExecutionOrder::MEMORY_EFFICIENT);

  py::enum_<OrtCompiledModelCompatibility>(m, "OrtCompiledModelCompatibility")
      .value("EP_NOT_APPLICABLE", OrtCompiledModelCompatibility_EP_NOT_APPLICABLE)
      .value("EP_SUPPORTED_OPTIMAL", OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL)
      .value("EP_SUPPORTED_PREFER_RECOMPILATION", OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION)
      .value("EP_UNSUPPORTED", OrtCompiledModelCompatibility_EP_UNSUPPORTED);

  py::enum_<OrtDeviceEpIncompatibilityReason>(m, "OrtDeviceEpIncompatibilityReason", py::arithmetic())
      .value("NONE", OrtDeviceEpIncompatibility_NONE)
      .value("DRIVER_INCOMPATIBLE", OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE)
      .value("DEVICE_INCOMPATIBLE", OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE)
      .value("MISSING_DEPENDENCY", OrtDeviceEpIncompatibility_MISSING_DEPENDENCY)
      .value("UNKNOWN", OrtDeviceEpIncompatibility_UNKNOWN);

  py::enum_<OrtAllocatorType>(m, "OrtAllocatorType")
      .value("INVALID", OrtInvalidAllocator)
      .value("ORT_DEVICE_ALLOCATOR", OrtDeviceAllocator)
      .value("ORT_ARENA_ALLOCATOR", OrtArenaAllocator);

  py::enum_<OrtMemType>(m, "OrtMemType")
      .value("CPU_INPUT", OrtMemTypeCPUInput)
      .value("CPU_OUTPUT", OrtMemTypeCPUOutput)
      .value("CPU", OrtMemTypeCPU)
      .value("DEFAULT", OrtMemTypeDefault);

  py::enum_<OrtMemoryInfoDeviceType>(m, "OrtMemoryInfoDeviceType")
      .value("CPU", OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU)
      .value("GPU", OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU)
      .value("NPU", OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_NPU)
      .value("FPGA", OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_FPGA);

  py::enum_<OrtDeviceMemoryType>(m, "OrtDeviceMemoryType")
      .value("DEFAULT", OrtDeviceMemoryType_DEFAULT)
      .value("HOST_ACCESSIBLE", OrtDeviceMemoryType_HOST_ACCESSIBLE);

  py::class_<OrtDevice> device(m, "OrtDevice", R"pbdoc(ONNXRuntime device information.)pbdoc");
  device.def(py::init<OrtDevice::DeviceType, OrtDevice::MemoryType, OrtDevice::VendorId, OrtDevice::DeviceId>())
      .def(py::init([](OrtDevice::DeviceType type,
                       OrtDevice::MemoryType mem_type,
                       OrtDevice::DeviceId device_id) {
             // backwards compatibility. generally there's only one GPU EP in the python package, with the exception
             // of a build with CUDA and DML.
             OrtDevice::VendorId vendor = OrtDevice::VendorIds::NONE;
             if (type == OrtDevice::DML) {
               type = OrtDevice::GPU;
               vendor = OrtDevice::VendorIds::MICROSOFT;
             } else if (type == OrtDevice::GPU) {
#if USE_CUDA || USE_CUDA_PROVIDER_INTERFACE
               vendor = OrtDevice::VendorIds::NVIDIA;
#elif USE_MIGRAPHX
               vendor = OrtDevice::VendorIds::AMD;
#endif
             } else if (type == OrtDevice::NPU) {
#if USE_CANN
               vendor = OrtDevice::VendorIds::HUAWEI;
#endif
             }
             return OrtDevice(type, mem_type, vendor, device_id);
           }),
           R"pbdoc(Constructor with vendor_id defaulted to 0 for backward compatibility.)pbdoc")
      .def("device_id", &OrtDevice::Id, R"pbdoc(Device Id.)pbdoc")
      .def("device_type", &OrtDevice::Type, R"pbdoc(Device Type.)pbdoc")
      .def("vendor_id", &OrtDevice::Vendor, R"pbdoc(Vendor Id.)pbdoc")
      .def("mem_type", &OrtDevice::MemType, R"pbdoc(Device Memory Type.)pbdoc")
      // generic device types that are typically used with a vendor id.
      .def_static("cpu", []() { return OrtDevice::CPU; })
      .def_static("gpu", []() { return OrtDevice::GPU; })
      .def_static("npu", []() { return OrtDevice::NPU; })
      // EP specific device types for backward compatibility.
      .def_static("cuda", []() { return OrtDevice::GPU; })
      .def_static("cann", []() { return OrtDevice::NPU; })
      .def_static("dml", []() { return OrtDevice::DML; })
      .def_static("fpga", []() { return OrtDevice::FPGA; })
      .def_static("webgpu", []() { return OrtDevice::GPU; })
      .def_static("default_memory", []() { return OrtDevice::MemType::DEFAULT; });

  py::enum_<OrtExecutionProviderDevicePolicy>(m, "OrtExecutionProviderDevicePolicy")
      .value("DEFAULT", OrtExecutionProviderDevicePolicy_DEFAULT)
      .value("PREFER_CPU", OrtExecutionProviderDevicePolicy_PREFER_CPU)
      .value("PREFER_NPU", OrtExecutionProviderDevicePolicy_PREFER_NPU)
      .value("PREFER_GPU", OrtExecutionProviderDevicePolicy_PREFER_GPU)
      .value("MAX_PERFORMANCE", OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE)
      .value("MAX_EFFICIENCY", OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY)
      .value("MIN_OVERALL_POWER", OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER);

  py::enum_<OrtHardwareDeviceType>(m, "OrtHardwareDeviceType")
      .value("CPU", OrtHardwareDeviceType_CPU)
      .value("GPU", OrtHardwareDeviceType_GPU)
      .value("NPU", OrtHardwareDeviceType_NPU);

  py::class_<OrtHardwareDevice> py_hw_device(m, "OrtHardwareDevice", R"pbdoc(ONNX Runtime hardware device information.)pbdoc");
  py_hw_device.def_property_readonly(
                  "type",
                  [](OrtHardwareDevice* hw_device) -> OrtHardwareDeviceType { return hw_device->type; },
                  R"pbdoc(Hardware device's type.)pbdoc")
      .def_property_readonly(
          "vendor_id",
          [](OrtHardwareDevice* hw_device) -> uint32_t { return hw_device->vendor_id; },
          R"pbdoc(Hardware device's vendor identifier.)pbdoc")
      .def_property_readonly(
          "vendor",
          [](OrtHardwareDevice* hw_device) -> std::string { return hw_device->vendor; },
          R"pbdoc(Hardware device's vendor name.)pbdoc")
      .def_property_readonly(
          "device_id",
          [](OrtHardwareDevice* hw_device) -> uint32_t { return hw_device->device_id; },
          R"pbdoc(Hardware device's unique identifier.)pbdoc")
      .def_property_readonly(
          "metadata",
          [](OrtHardwareDevice* hw_device) -> std::map<std::string, std::string> {
            return hw_device->metadata.Entries();
          },
          R"pbdoc(Hardware device's metadata as string key/value pairs.)pbdoc");

  py::class_<OrtSyncStream> py_sync_stream(m, "OrtSyncStream",
                                           R"pbdoc(Represents a synchronization stream for model inference.)pbdoc");
  py_sync_stream.def("get_handle", [](OrtSyncStream* stream) -> uintptr_t {
      Ort::UnownedSyncStream ort_stream(stream);
      return reinterpret_cast<uintptr_t>(ort_stream.GetHandle()); }, R"pbdoc(SyncStream handle that can be converted to a string and added to SessionOptions)pbdoc");

  py::class_<OrtEpDevice> py_ep_device(m, "OrtEpDevice",
                                       R"pbdoc(Represents a hardware device that an execution provider supports
for model inference.)pbdoc");
  py_ep_device.def_property_readonly(
                  "ep_name",
                  [](const OrtEpDevice* ep_device) -> std::string { return ep_device->ep_name; },
                  R"pbdoc(The execution provider's name.)pbdoc")
      .def_property_readonly(
          "ep_vendor",
          [](const OrtEpDevice* ep_device) -> std::string { return ep_device->ep_vendor; },
          R"pbdoc(The execution provider's vendor name.)pbdoc")
      .def_property_readonly(
          "ep_metadata",
          [](const OrtEpDevice* ep_device) -> std::map<std::string, std::string> {
            return ep_device->ep_metadata.Entries();
          },
          R"pbdoc(The execution provider's additional metadata for the OrtHardwareDevice.)pbdoc")
      .def_property_readonly(
          "ep_options",
          [](const OrtEpDevice* ep_device) -> std::map<std::string, std::string> {
            return ep_device->ep_options.Entries();
          },
          R"pbdoc(The execution provider's options used to configure the provider to use the OrtHardwareDevice.)pbdoc")
      .def_property_readonly(
          "device",
          [](const OrtEpDevice* ep_device) -> const OrtHardwareDevice& {
            return *ep_device->device;
          },
          R"pbdoc(The OrtHardwareDevice instance for the OrtEpDevice.)pbdoc",
          py::return_value_policy::reference_internal)
      .def(
          "memory_info",
          [](const OrtEpDevice* ep_device, OrtDeviceMemoryType memory_type) -> const OrtMemoryInfo* {
            Ort::ConstEpDevice ep_dev(ep_device);
            return static_cast<const OrtMemoryInfo*>(ep_dev.GetMemoryInfo(memory_type));
          },
          R"pbdoc(The OrtMemoryInfo instance for the OrtEpDevice specific to the device memory type.)pbdoc",
          py::return_value_policy::reference_internal)
      .def(
          "create_sync_stream",
          [](const OrtEpDevice* ep_device) -> std::unique_ptr<OrtSyncStream> {
            Ort::ConstEpDevice ep_dev(ep_device);
            Ort::SyncStream stream = ep_dev.CreateSyncStream();
            return std::unique_ptr<OrtSyncStream>(stream.release());
          },
          R"pbdoc(The OrtSyncStream instance for the OrtEpDevice.)pbdoc");

  py::class_<OrtEpAssignedNode> py_ep_node(m, "OrtEpAssignedNode",
                                           R"pbdoc(Contains information about a node assigned to an execution
provider)pbdoc");
  py_ep_node
      .def_property_readonly(
          "name",
          [](const OrtEpAssignedNode* ep_node) -> std::string {
            return ep_node->name;
          },
          R"pbdoc(The node's name)pbdoc")
      .def_property_readonly(
          "domain",
          [](const OrtEpAssignedNode* ep_node) -> std::string {
            return ep_node->domain;
          },
          R"pbdoc(The node's domain)pbdoc")
      .def_property_readonly(
          "op_type",
          [](const OrtEpAssignedNode* ep_node) -> std::string {
            return ep_node->op_type;
          },
          R"pbdoc(The node's operator type)pbdoc");

  py::class_<OrtEpAssignedSubgraph> py_ep_subgraph(m, "OrtEpAssignedSubgraph",
                                                   R"pbdoc(Contains information about a subgraph assigned to an
execution provider)pbdoc");
  py_ep_subgraph
      .def_property_readonly(
          "ep_name",
          [](const OrtEpAssignedSubgraph* ep_subgraph) -> std::string {
            return ep_subgraph->ep_name;
          },
          R"pbdoc(The name of the execution provider to which this subgraph is assigned.)pbdoc")
      .def(
          "get_nodes",
          [](const OrtEpAssignedSubgraph* ep_subgraph) -> const std::vector<const OrtEpAssignedNode*>& {
            return ep_subgraph->nodes;
          },
          py::return_value_policy::reference_internal,
          R"pbdoc(List of nodes in the subgraph.)pbdoc");

  py::class_<OrtArenaCfg> ort_arena_cfg_binding(m, "OrtArenaCfg");
  // Note: Doesn't expose initial_growth_chunk_sizes_bytes/max_power_of_two_extend_bytes option.
  // This constructor kept for backwards compatibility, key-value pair constructor overload exposes all options
  // There is a global var: arena_extend_strategy, which means we can't use that var name here
  // See docs/C_API.md for details on what the following parameters mean and how to choose these values
  ort_arena_cfg_binding.def(py::init([](size_t max_mem, int arena_extend_strategy_local,
                                        int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) {
                         auto ort_arena_cfg = std::make_unique<OrtArenaCfg>();
                         ort_arena_cfg->max_mem = max_mem;
                         ort_arena_cfg->arena_extend_strategy = arena_extend_strategy_local;
                         ort_arena_cfg->initial_chunk_size_bytes = initial_chunk_size_bytes;
                         ort_arena_cfg->max_dead_bytes_per_chunk = max_dead_bytes_per_chunk;
                         return ort_arena_cfg;
                       }))
      .def(py::init([](const py::dict& feeds) {
        auto ort_arena_cfg = std::make_unique<OrtArenaCfg>();
        for (const auto kvp : feeds) {
          std::string key = kvp.first.cast<std::string>();
          if (key == "max_mem") {
            ort_arena_cfg->max_mem = kvp.second.cast<size_t>();
          } else if (key == "arena_extend_strategy") {
            ort_arena_cfg->arena_extend_strategy = kvp.second.cast<int>();
          } else if (key == "initial_chunk_size_bytes") {
            ort_arena_cfg->initial_chunk_size_bytes = kvp.second.cast<int>();
          } else if (key == "max_dead_bytes_per_chunk") {
            ort_arena_cfg->max_dead_bytes_per_chunk = kvp.second.cast<int>();
          } else if (key == "initial_growth_chunk_size_bytes") {
            ort_arena_cfg->initial_growth_chunk_size_bytes = kvp.second.cast<int>();
          } else if (key == "max_power_of_two_extend_bytes") {
            ort_arena_cfg->max_power_of_two_extend_bytes = kvp.second.cast<int>();
          } else {
            ORT_THROW("Invalid OrtArenaCfg option: ", key);
          }
        }
        return ort_arena_cfg;
      }))
      .def_readwrite("max_mem", &OrtArenaCfg::max_mem)
      .def_readwrite("arena_extend_strategy", &OrtArenaCfg::arena_extend_strategy)
      .def_readwrite("initial_chunk_size_bytes", &OrtArenaCfg::initial_chunk_size_bytes)
      .def_readwrite("max_dead_bytes_per_chunk", &OrtArenaCfg::max_dead_bytes_per_chunk)
      .def_readwrite("initial_growth_chunk_size_bytes", &OrtArenaCfg::initial_growth_chunk_size_bytes)
      .def_readwrite("max_power_of_two_extend_bytes", &OrtArenaCfg::max_power_of_two_extend_bytes);

  py::class_<OrtMemoryInfo> ort_memory_info_binding(m, "OrtMemoryInfo");
  ort_memory_info_binding.def(
                             py::init([](const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
                               Ort::MemoryInfo result(name, type, id, mem_type);
                               return std::unique_ptr<OrtMemoryInfo>(result.release());
                             }))
      .def_static(
          "create_v2",
          [](const char* name, OrtMemoryInfoDeviceType device_type, uint32_t vendor_id,
             int32_t device_id, OrtDeviceMemoryType device_mem_type, size_t alignment, OrtAllocatorType type) {
            Ort::MemoryInfo result(name, device_type, vendor_id, device_id, device_mem_type, alignment, type);
            return std::unique_ptr<OrtMemoryInfo>(result.release());
          },
          R"pbdoc(Create an OrtMemoryInfo instance using CreateMemoryInfo_V2())pbdoc")
      .def_property_readonly("name", [](const OrtMemoryInfo* mem_info) -> std::string { return mem_info->name; }, R"pbdoc(Arbitrary name supplied by the user)pbdoc")
      .def_property_readonly("device_id", [](const OrtMemoryInfo* mem_info) -> int { return mem_info->device.Id(); }, R"pbdoc(Device Id.)pbdoc")
      .def_property_readonly("mem_type", [](const OrtMemoryInfo* mem_info) -> OrtMemType { return mem_info->mem_type; }, R"pbdoc(OrtMemoryInfo memory type.)pbdoc")
      .def_property_readonly("allocator_type", [](const OrtMemoryInfo* mem_info) -> OrtAllocatorType { return mem_info->alloc_type; }, R"pbdoc(Allocator type)pbdoc")
      .def_property_readonly("device_mem_type", [](const OrtMemoryInfo* mem_info) -> OrtDeviceMemoryType {
              auto mem_type = mem_info->device.MemType();
              return (mem_type == OrtDevice::MemType::DEFAULT) ?
                  OrtDeviceMemoryType_DEFAULT: OrtDeviceMemoryType_HOST_ACCESSIBLE ; }, R"pbdoc(Device memory type (Device or Host accessible).)pbdoc")
      .def_property_readonly("device_vendor_id", [](const OrtMemoryInfo* mem_info) -> uint32_t { return mem_info->device.Vendor(); });

  py::class_<PySessionOptions>
      sess(m, "SessionOptions", R"pbdoc(Configuration information for a session.)pbdoc");
#if !defined(ORT_MINIMAL_BUILD)
  py::class_<PyEpContextData, std::shared_ptr<PyEpContextData>>(
      m, "OrtEpContextData",
      R"pbdoc(Non-owning view of named EPContext data, valid only during its write callback.)pbdoc")
      .def_property_readonly("size", &PyEpContextData::Size)
      .def("read", &PyEpContextData::Read,
           py::arg("offset") = 0, py::arg("length") = std::nullopt,
           R"pbdoc(Copy a bounded range into Python bytes. Use chunks for large payloads.)pbdoc");

  py::class_<PyEpContextDataBuffer, std::shared_ptr<PyEpContextDataBuffer>>(
      m, "OrtEpContextDataBuffer",
      R"pbdoc(Allocator-backed output valid only during its EPContext data read callback.)pbdoc")
      .def_property_readonly("size", &PyEpContextDataBuffer::Size)
      .def_property_readonly("max_size", &PyEpContextDataBuffer::MaxSize)
      .def_property_readonly("is_allocated", &PyEpContextDataBuffer::IsAllocated)
      .def("allocate", &PyEpContextDataBuffer::Allocate, py::arg("size"),
           R"pbdoc(Allocate the final ORT-owned output buffer.)pbdoc")
      .def("write", [](PyEpContextDataBuffer& output, const py::buffer& data, size_t offset) { output.Write(offset, data); }, py::arg("data"), py::arg("offset") = 0, R"pbdoc(Copy a contiguous bytes-like chunk into the allocated output.)pbdoc");
#endif  // !defined(ORT_MINIMAL_BUILD)

  sess
      .def(py::init())
      .def(
          // Equivalent to the C API's SessionOptionsAppendExecutionProvider.
          "add_provider",
          [](PySessionOptions* sess_options,
             const std::string& provider_name,
             const ProviderOptions& provider_options = {}) {
            OrtPybindThrowIfError(AddExplicitEpFactory(*sess_options, provider_name, provider_options));
          },
          R"pbdoc(Adds an explicit execution provider.)pbdoc")
      .def(
          // Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2.
          "add_provider_for_devices",
          [](PySessionOptions* sess_options,
             const std::vector<const OrtEpDevice*>& ep_devices,
             const ProviderOptions& provider_options = {}) {
#if !defined(ORT_MINIMAL_BUILD)
            OrtPybindThrowIfError(AddEpFactoryFromEpDevices(*sess_options,
                                                            ep_devices,
                                                            provider_options));
#else
            ORT_UNUSED_PARAMETER(sess_options);
            ORT_UNUSED_PARAMETER(ep_devices);
            ORT_UNUSED_PARAMETER(provider_options);
            ORT_THROW("OrtEpDevices are not supported in this build");
#endif
          },
          R"pbdoc(Adds the execution provider that is responsible for the selected OrtEpDevice instances. All OrtEpDevice instances
must refer to the same execution provider.)pbdoc")
      .def(
          // Equivalent to the C API's SessionOptionsSetEpSelectionPolicy.
          "set_provider_selection_policy",
          [](PySessionOptions* py_sess_options,
             OrtExecutionProviderDevicePolicy policy) {
#if !defined(ORT_MINIMAL_BUILD)
            std::shared_ptr<PyEpSelectionRegistration> old_registration;
            {
              std::lock_guard<std::mutex> lock{py_sess_options->py_callback_mutex};
              old_registration = std::move(py_sess_options->py_ep_selection_registration);

              py_sess_options->value.ep_selection_policy.enable = true;
              py_sess_options->value.ep_selection_policy.policy = policy;
              py_sess_options->value.ep_selection_policy.delegate = nullptr;
              py_sess_options->value.ep_selection_policy.state = nullptr;
            }
#else
            ORT_UNUSED_PARAMETER(py_sess_options);
            ORT_UNUSED_PARAMETER(policy);
            ORT_THROW("EP selection policies are not supported in this build");
#endif
          },
          R"pbdoc(Sets the execution provider selection policy for the session. Allows users to specify a
selection policy for automatic execution provider (EP) selection.)pbdoc")
      .def(
          // Equivalent to the C API's SessionOptionsSetEpSelectionPolicyDelegate.
          "set_provider_selection_policy_delegate",
          [](PySessionOptions* py_sess_options,
             PyEpSelectionDelegate delegate_fn) {
#if !defined(ORT_MINIMAL_BUILD)
            auto registration = std::make_shared<PyEpSelectionRegistration>(
                PyEpSelectionRegistration{std::move(delegate_fn)});
            std::shared_ptr<PyEpSelectionRegistration> old_registration;
            {
              std::lock_guard<std::mutex> lock{py_sess_options->py_callback_mutex};
              old_registration = std::move(py_sess_options->py_ep_selection_registration);

              py_sess_options->value.ep_selection_policy.enable = true;
              py_sess_options->value.ep_selection_policy.policy = OrtExecutionProviderDevicePolicy_DEFAULT;
              py_sess_options->value.ep_selection_policy.delegate = PyEpSelectionPolicyWrapper;
              py_sess_options->value.ep_selection_policy.state = registration.get();
              py_sess_options->py_ep_selection_registration = std::move(registration);
            }
#else
            ORT_UNUSED_PARAMETER(py_sess_options);
            ORT_UNUSED_PARAMETER(delegate_fn);
            ORT_THROW("EP selection policies are not supported in this build");
#endif
          },
          R"pbdoc(Sets the execution provider selection policy delegate for the session. Allows users to specify a
custom selection policy function for automatic execution provider (EP) selection. The delegate must return a list of
selected OrtEpDevice instances. The signature of the delegate is
def custom_delegate(ep_devices: Sequence[OrtEpDevice], model_metadata: dict[str, str], runtime_metadata: dict[str, str],
max_selections: int) -> Sequence[OrtEpDevice])pbdoc")
      .def(
          "has_providers",
          [](PySessionOptions* sess_options) -> bool {
            return !sess_options->provider_factories.empty() || sess_options->value.ep_selection_policy.enable;
          },
          R"pbdoc(Returns true if the SessionOptions has been configured with providers, OrtEpDevices, or
policies that will run the model.)pbdoc")
      .def_property(
          "enable_cpu_mem_arena",
          [](const PySessionOptions* options) -> bool { return options->value.enable_cpu_mem_arena; },
          [](PySessionOptions* options, bool enable_cpu_mem_arena) -> void {
            options->value.enable_cpu_mem_arena = enable_cpu_mem_arena;
          },
          R"pbdoc(Enables the memory arena on CPU. Arena may pre-allocate memory for future usage.
Set this option to false if you don't want it. Default is True.)pbdoc")
      .def_property(
          "enable_profiling",
          [](const PySessionOptions* options) -> bool { return options->value.enable_profiling; },
          [](PySessionOptions* options, bool enable_profiling) -> void {
            options->value.enable_profiling = enable_profiling;
          },
          R"pbdoc(Enable profiling for this session. Default is false.)pbdoc")
      .def_property(
          "profile_file_prefix",
          [](const PySessionOptions* options) -> std::basic_string<ORTCHAR_T> {
            return options->value.profile_file_prefix;
          },
          [](PySessionOptions* options, std::basic_string<ORTCHAR_T> profile_file_prefix) -> void {
            options->value.profile_file_prefix = std::move(profile_file_prefix);
          },
          R"pbdoc(The prefix of the profile file. The current time will be appended to the file name.)pbdoc")
      .def_property(
          "optimized_model_filepath",
          [](const PySessionOptions* options) -> std::basic_string<ORTCHAR_T> {
            return options->value.optimized_model_filepath;
          },
          [](PySessionOptions* options, std::basic_string<ORTCHAR_T> optimized_model_filepath) -> void {
            options->value.optimized_model_filepath = std::move(optimized_model_filepath);
          },
          R"pbdoc(
File path to serialize optimized model to.
Optimized model is not serialized unless optimized_model_filepath is set.
Serialized model format will default to ONNX unless:
- add_session_config_entry is used to set 'session.save_model_format' to 'ORT', or
- there is no 'session.save_model_format' config entry and optimized_model_filepath ends in '.ort' (case insensitive)

)pbdoc")
      .def_property(
          "enable_cpu_mem_arena",
          [](const PySessionOptions* options) -> bool { return options->value.enable_cpu_mem_arena; },
          [](PySessionOptions* options, bool enable_cpu_mem_arena) -> void {
            options->value.enable_cpu_mem_arena = enable_cpu_mem_arena;
          },
          R"pbdoc(Enable memory arena on CPU. Default is true.)pbdoc")
      .def_property(
          "enable_mem_pattern",
          [](const PySessionOptions* options) -> bool { return options->value.enable_mem_pattern; },
          [](PySessionOptions* options, bool enable_mem_pattern) -> void {
            options->value.enable_mem_pattern = enable_mem_pattern;
          },
          R"pbdoc(Enable the memory pattern optimization. Default is true.)pbdoc")
      .def_property(
          "enable_mem_reuse",
          [](const PySessionOptions* options) -> bool { return options->value.enable_mem_reuse; },
          [](PySessionOptions* options, bool enable_mem_reuse) -> void {
            options->value.enable_mem_reuse = enable_mem_reuse;
          },
          R"pbdoc(Enable the memory reuse optimization. Default is true.)pbdoc")
      .def_property(
          "logid",
          [](const PySessionOptions* options) -> std::string {
            return options->value.session_logid;
          },
          [](PySessionOptions* options, std::string logid) -> void {
            options->value.session_logid = std::move(logid);
          },
          R"pbdoc(Logger id to use for session output.)pbdoc")
      .def_property(
          "log_severity_level",
          [](const PySessionOptions* options) -> int { return options->value.session_log_severity_level; },
          [](PySessionOptions* options, int log_severity_level) -> void {
            options->value.session_log_severity_level = log_severity_level;
          },
          R"pbdoc(Log severity level. Applies to session load, initialization, etc.
0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_property(
          "log_verbosity_level",
          [](const PySessionOptions* options) -> int { return options->value.session_log_verbosity_level; },
          [](PySessionOptions* options, int log_verbosity_level) -> void {
            options->value.session_log_verbosity_level = log_verbosity_level;
          },
          R"pbdoc(VLOG level if DEBUG build and session_log_severity_level is 0.
Applies to session load, initialization, etc. Default is 0.)pbdoc")
      .def_property(
          "use_per_session_threads",
          [](const PySessionOptions* options) -> bool { return options->value.use_per_session_threads; },
          [](PySessionOptions* options, bool use_per_session_threads) -> void {
            options->value.use_per_session_threads = use_per_session_threads;
          },
          R"pbdoc(Whether to use per-session thread pool. Default is True.)pbdoc")
      .def_property(
          "intra_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->value.intra_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->value.intra_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose.)pbdoc")
      .def_property(
          "inter_op_num_threads",
          [](const PySessionOptions* options) -> int { return options->value.inter_op_param.thread_pool_size; },
          [](PySessionOptions* options, int value) -> void { options->value.inter_op_param.thread_pool_size = value; },
          R"pbdoc(Sets the number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.)pbdoc")
      .def_property(
          "execution_mode",
          [](const PySessionOptions* options) -> ExecutionMode { return options->value.execution_mode; },
          [](PySessionOptions* options, ExecutionMode execution_mode) -> void {
            options->value.execution_mode = execution_mode;
          },
          R"pbdoc(Sets the execution mode. Default is sequential.)pbdoc")
      .def(
          "set_load_cancellation_flag",
          [](PySessionOptions* options, bool value) -> void {
            options->value.SetLoadCancellationFlag(value);
          },
          R"pbdoc(Request inference session load cancellation)pbdoc")
      .def_property(
          "execution_order",
          [](const PySessionOptions* options) -> ExecutionOrder { return options->value.execution_order; },
          [](PySessionOptions* options, ExecutionOrder execution_order) -> void {
            options->value.execution_order = execution_order;
          },
          R"pbdoc(Sets the execution order. Default is basic topological order.)pbdoc")
      .def_property(
          "graph_optimization_level",
          [](const PySessionOptions* options) -> GraphOptimizationLevel {
            GraphOptimizationLevel retval = ORT_ENABLE_ALL;
            switch (options->value.graph_optimization_level) {
              case onnxruntime::TransformerLevel::Default:
                retval = ORT_DISABLE_ALL;
                break;
              case onnxruntime::TransformerLevel::Level1:
                retval = ORT_ENABLE_BASIC;
                break;
              case onnxruntime::TransformerLevel::Level2:
                retval = ORT_ENABLE_EXTENDED;
                break;
              case onnxruntime::TransformerLevel::Level3:
                retval = ORT_ENABLE_LAYOUT;
                break;
              case onnxruntime::TransformerLevel::MaxLevel:
                retval = ORT_ENABLE_ALL;
                break;
              default:
                retval = ORT_ENABLE_ALL;
                LOGS_DEFAULT(WARNING) << "Got invalid graph optimization level; defaulting to ORT_ENABLE_ALL";
                break;
            }
            return retval;
          },

          [](PySessionOptions* options, GraphOptimizationLevel level) -> void {
            switch (level) {
              case ORT_DISABLE_ALL:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Default;
                break;
              case ORT_ENABLE_BASIC:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level1;
                break;
              case ORT_ENABLE_EXTENDED:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level2;
                break;
              case ORT_ENABLE_LAYOUT:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level3;
                break;
              case ORT_ENABLE_ALL:
                options->value.graph_optimization_level = onnxruntime::TransformerLevel::MaxLevel;
                break;
            }
          },
          R"pbdoc(Graph optimization level for this session.)pbdoc")
      .def_property(
          "use_deterministic_compute",
          [](const PySessionOptions* options) -> bool { return options->value.use_deterministic_compute; },
          [](PySessionOptions* options, bool use_deterministic_compute) -> void {
            options->value.use_deterministic_compute = use_deterministic_compute;
          },
          R"pbdoc(Whether to use deterministic compute. Default is false.)pbdoc")
      .def(
          "add_free_dimension_override_by_denotation",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->value.free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Denotation,
                                dim_value}); },
          R"pbdoc(Specify the dimension size for each denotation associated with an input's free dimension.)pbdoc")
      .def(
          "add_free_dimension_override_by_name",
          [](PySessionOptions* options, const char* dim_name, int64_t dim_value)
              -> void { options->value.free_dimension_overrides.push_back(
                            onnxruntime::FreeDimensionOverride{
                                dim_name,
                                onnxruntime::FreeDimensionOverrideType::Name,
                                dim_value}); },
          R"pbdoc(Specify values of named dimensions within model inputs.)pbdoc")
      .def(
          "add_session_config_entry",
          [](PySessionOptions* options, const char* config_key, const char* config_value) -> void {
            // config_key and config_value will be copied
            const Status status = options->value.config_options.AddConfigEntry(config_key, config_value);
            if (!status.IsOK())
              throw std::runtime_error(status.ErrorMessage());
          },
          R"pbdoc(Set a single session configuration entry as a pair of strings.)pbdoc")
#if !defined(ORT_MINIMAL_BUILD)
      .def(
          "set_ep_context_data_read_func",
          [](PySessionOptions* options, PyEpContextDataReadFunc read_func, size_t max_data_size) {
            ORT_ENFORCE(static_cast<bool>(read_func), "EPContext data read callback must not be None.");
            ORT_ENFORCE(max_data_size != 0 && max_data_size != std::numeric_limits<size_t>::max(),
                        "EPContext data max_data_size must be finite and greater than zero.");

            auto registration = std::make_shared<PyEpContextDataReadRegistration>(
                PyEpContextDataReadRegistration{std::move(read_func), max_data_size});
            std::shared_ptr<PyEpContextDataReadRegistration> old_registration;
            {
              std::lock_guard<std::mutex> lock{options->py_callback_mutex};
              old_registration = std::move(options->py_ep_context_data_read_registration);
              options->value.ep_context_data_read_func = PyEpContextDataReadFuncWrapper;
              options->value.ep_context_data_read_state = registration.get();
              options->py_ep_context_data_read_registration = std::move(registration);
            }
          },
          py::arg("read_func"), py::arg("max_data_size"),
          R"pbdoc(Register a bounded callback that supplies external EPContext data during session loading.

The callback receives the data name and an OrtEpContextDataBuffer. It must call allocate(size) exactly once,
where size is at most max_data_size, then populate that buffer with one or more write(data, offset) calls.
The output object and its allocator are valid only until the callback returns.
The Python callback registration retains and enforces max_data_size; the native API stores only the callback and state.

ORT may invoke the callback concurrently. Synchronize any shared Python state.)pbdoc")
      .def(
          "clear_ep_context_data_read_func",
          [](PySessionOptions* options) {
            std::shared_ptr<PyEpContextDataReadRegistration> old_registration;
            {
              std::lock_guard<std::mutex> lock{options->py_callback_mutex};
              options->value.ep_context_data_read_func = nullptr;
              options->value.ep_context_data_read_state = nullptr;
              old_registration = std::move(options->py_ep_context_data_read_registration);
            }
          },
          R"pbdoc(Clear the external EPContext data read callback.)pbdoc")
#endif  // !defined(ORT_MINIMAL_BUILD)
      .def(
          "get_session_config_entry",
          [](const PySessionOptions* options, const char* config_key) -> std::string {
            const std::string key(config_key);
            std::string value;
            if (!options->value.config_options.TryGetConfigEntry(key, value))
              throw std::runtime_error("SessionOptions does not have configuration with key: " + key);

            return value;
          },
          R"pbdoc(Get a single session configuration value using the given configuration key.)pbdoc")
      .def(
          "register_custom_ops_library",
          [](PySessionOptions* options, const char* library_name) -> void {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
            OrtPybindThrowIfError(options->RegisterCustomOpsLibrary(ToPathString(library_name)));
#else
            ORT_UNUSED_PARAMETER(options);
            ORT_UNUSED_PARAMETER(library_name);
            ORT_THROW("Custom Ops are not supported in this build.");
#endif
          },
          R"pbdoc(Specify the path to the shared library containing the custom op kernels required to run a model.)pbdoc")
      .def(
          "add_initializer", [](PySessionOptions* options, const char* name, py::object& ml_value_pyobject) -> void {
            ORT_ENFORCE(strcmp(Py_TYPE(ml_value_pyobject.ptr())->tp_name, PYTHON_ORTVALUE_OBJECT_NAME) == 0, "The provided Python object must be an OrtValue");
            // The user needs to ensure that the python OrtValue being provided as an overriding initializer
            // is not destructed as long as any session that uses the provided OrtValue initializer is still in scope
            // This is no different than the native APIs
            const OrtValue* ml_value = ml_value_pyobject.attr(PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR).cast<OrtValue*>();
            ORT_THROW_IF_ERROR(options->value.AddInitializer(name, ml_value));
          })
      .def("add_external_initializers", [](PySessionOptions* options, py::list& names, const py::list& ort_values) -> void {
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
        const auto init_num = ort_values.size();
        ORT_ENFORCE(init_num == names.size(), "Expecting names and ort_values lists to have equal length");
        InlinedVector<std::string> names_ptrs;
        InlinedVector<OrtValue> values_ptrs;
        names_ptrs.reserve(init_num);
        values_ptrs.reserve(init_num);
        for (size_t i = 0; i < init_num; ++i) {
          names_ptrs.emplace_back(py::str(names[i]));
          values_ptrs.emplace_back(*ort_values[i].attr(PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR).cast<const OrtValue*>());
        }
        ORT_THROW_IF_ERROR(options->value.AddExternalInitializers(names_ptrs, values_ptrs));
#else
        ORT_UNUSED_PARAMETER(options);
        ORT_UNUSED_PARAMETER(names);
        ORT_UNUSED_PARAMETER(ort_values);
        ORT_THROW("External initializers are not supported in this build.");
#endif
      })
      .def("add_external_initializers_from_files_in_memory", [](PySessionOptions* options, std::vector<std::string> names, std::vector<py::buffer> buffers, std::vector<size_t> lengths) -> void {
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
        const auto num = names.size();
        ORT_ENFORCE(num == buffers.size() && num == lengths.size(),
                    "add_external_initializers_from_files_in_memory: expecting 'names', 'buffers' and 'lengths' to have equal length");

        InlinedVector<PathString> file_names;
        InlinedVector<std::pair<char*, const size_t>> files_buffers;
        file_names.reserve(num);
        files_buffers.reserve(num);

        for (size_t i = 0; i < num; ++i) {
          // Convert name and buffer using pybind-provided conversions
          file_names.emplace_back(ToPathString(names[i]));

          // buffers[i] is a py::buffer; request() retrieves pointer without copying
          py::buffer_info info = buffers[i].request();
          char* data_ptr = static_cast<char*>(info.ptr);

          files_buffers.emplace_back(std::make_pair(data_ptr, lengths[i]));
        }

        ORT_THROW_IF_ERROR(options->value.AddExternalInitializersFromFilesInMemory(file_names, files_buffers));
#else
        ORT_UNUSED_PARAMETER(options);
        ORT_UNUSED_PARAMETER(names);
        ORT_UNUSED_PARAMETER(buffers);
        ORT_UNUSED_PARAMETER(lengths);
        ORT_THROW("External initializers are not supported in this build.");
#endif
      },
           R"pbdoc(
Provide external initializer file contents from memory.

Args:
  names: sequence[str] of external file names (as referenced by the model's external_data locations).
  buffers: sequence[bytes-like] objects exposing the buffer protocol (e.g., bytes, bytearray, memoryview, numpy uint8 array) containing the corresponding file contents.
  lengths: sequence[int] sizes in bytes for each buffer.

Notes:
  - Keep the provided buffers alive until after session creation completes. ONNX Runtime copies needed data during session creation.
  - The bytestream must match the external file layout expected by the model (raw tensor bytes at the specified offsets).
)pbdoc");

  py::class_<RunOptions>(m, "RunOptions", R"pbdoc(Configuration information for a single Run.)pbdoc")
      .def(py::init())
      .def_readwrite("log_severity_level", &RunOptions::run_log_severity_level,
                     R"pbdoc(Log severity level for a particular Run() invocation. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.)pbdoc")
      .def_readwrite("log_verbosity_level", &RunOptions::run_log_verbosity_level,
                     R"pbdoc(VLOG level if DEBUG build and run_log_severity_level is 0.
Applies to a particular Run() invocation. Default is 0.)pbdoc")
      .def_readwrite("logid", &RunOptions::run_tag,
                     "To identify logs generated by a particular Run() invocation.")
      .def_readwrite("terminate", &RunOptions::terminate,
                     R"pbdoc(Set to True to terminate any currently executing calls that are using this
RunOptions instance. The individual calls will exit gracefully and return an error status.)pbdoc")
#ifdef ENABLE_TRAINING
      .def_readwrite("training_mode", &RunOptions::training_mode,
                     R"pbdoc(Choose to run in training or inferencing mode)pbdoc")
#endif
      .def_readwrite("only_execute_path_to_fetches", &RunOptions::only_execute_path_to_fetches,
                     R"pbdoc(Only execute the nodes needed by fetch list)pbdoc")
      .def(
          "add_run_config_entry",
          [](RunOptions* options, const char* config_key, const char* config_value) -> void {
            // config_key and config_value will be copied
            const Status status = options->config_options.AddConfigEntry(config_key, config_value);
            if (!status.IsOK())
              throw std::runtime_error(status.ErrorMessage());
          },
          R"pbdoc(Set a single run configuration entry as a pair of strings.)pbdoc")
      .def(
          "get_run_config_entry",
          [](const RunOptions* options, const char* config_key) -> std::string {
            const std::string key(config_key);
            std::string value;
            if (!options->config_options.TryGetConfigEntry(key, value))
              throw std::runtime_error("RunOptions does not have configuration with key: " + key);

            return value;
          },
          R"pbdoc(Get a single run configuration value using the given configuration key.)pbdoc")
      .def(
          "add_active_adapter", [](RunOptions* options, lora::LoraAdapter* adapter) {
            options->active_adapters.push_back(adapter);
          },
          R"pbdoc(Adds specified adapter as an active adapter)pbdoc");

  py::class_<ModelMetadata>(m, "ModelMetadata", R"pbdoc(Pre-defined and custom metadata about the model.
It is usually used to identify the model used to run the prediction and
facilitate the comparison.)pbdoc")
      .def_readwrite("producer_name", &ModelMetadata::producer_name, "producer name")
      .def_readwrite("graph_name", &ModelMetadata::graph_name, "graph name")
      .def_readwrite("domain", &ModelMetadata::domain, "ONNX domain")
      .def_readwrite("description", &ModelMetadata::description, "description of the model")
      .def_readwrite("graph_description", &ModelMetadata::graph_description, "description of the graph hosted in the model")
      .def_readwrite("version", &ModelMetadata::version, "version of the model")
      .def_readwrite("custom_metadata_map", &ModelMetadata::custom_metadata_map, "additional metadata");

  py::class_<onnxruntime::NodeArg>(m, "NodeArg", R"pbdoc(Node argument definition, for both input and output,
including arg name, arg type (contains both type and shape).)pbdoc")
      .def_property_readonly("name", &onnxruntime::NodeArg::Name, "node name")
      .def_property_readonly(
          "type", [](const onnxruntime::NodeArg& na) -> std::string {
            return *(na.Type());
          },
          "node type")
      .def("__str__", [](const onnxruntime::NodeArg& na) -> std::string {
            std::ostringstream res;
            res << "NodeArg(name='" << na.Name() << "', type='" << *(na.Type()) << "', shape=";
            auto shape = na.Shape();
            std::vector<py::object> arr;
            if (shape == nullptr || shape->dim_size() == 0) {
              res << "[]";
            } else {
              res << "[";
              for (int i = 0; i < shape->dim_size(); ++i) {
                if (utils::HasDimValue(shape->dim(i))) {
                  res << shape->dim(i).dim_value();
                } else if (utils::HasDimParam(shape->dim(i))) {
                  res << "'" << shape->dim(i).dim_param() << "'";
                } else {
                  res << "None";
                }

                if (i < shape->dim_size() - 1) {
                  res << ", ";
                }
              }
              res << "]";
            }
            res << ")";

            return std::string(res.str()); }, "converts the node into a readable string")
      .def_property_readonly("shape", [](const onnxruntime::NodeArg& na) -> std::vector<py::object> {
            auto shape = na.Shape();
            std::vector<py::object> arr;
            if (shape == nullptr || shape->dim_size() == 0) {
              return arr;
            }

            arr.resize(shape->dim_size());
            for (int i = 0; i < shape->dim_size(); ++i) {
              if (utils::HasDimValue(shape->dim(i))) {
                arr[i] = py::cast(shape->dim(i).dim_value());
              } else if (utils::HasDimParam(shape->dim(i))) {
                arr[i] = py::cast(shape->dim(i).dim_param());
              } else {
                arr[i] = py::none();
              }
            }
            return arr; }, "node shape (assuming the node holds a tensor)");

  py::class_<SessionObjectInitializer> sessionObjectInitializer(m, "SessionObjectInitializer");
  py::class_<PyInferenceSession>(m, "InferenceSession", R"pbdoc(This is the main class used to run a model.)pbdoc")
      // In Python3, a Python bytes object will be passed to C++ functions that accept std::string or char*
      // without any conversion. So this init method can be used for model file path (string) and model content (bytes)
      .def(py::init([](const PySessionOptions& so, const std::string arg, bool is_arg_file_name,
                       bool load_config_from_model = false) {
        std::unique_ptr<PyInferenceSession> sess;
        auto so_snapshot = CreatePySessionOptionsSnapshot(so);

        if (CheckIfUsingGlobalThreadPool() && so_snapshot.options.value.use_per_session_threads) {
          ORT_THROW("use_per_session_threads must be false when using a global thread pool");
        }

        if (CheckIfUsingGlobalThreadPool() && (so_snapshot.options.value.intra_op_param.thread_pool_size != 0 || so_snapshot.options.value.inter_op_param.thread_pool_size != 0)) {
          LOGS_DEFAULT(WARNING) << "session options intra_op_param.thread_pool_size and inter_op_param.thread_pool_size are ignored when using a global thread pool";
        }

        // separate creation of the session from model loading unless we have to read the config from the model.
        // in a minimal build we only support load via Load(...) and not at session creation time
        if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
          {
            py::gil_scoped_release release;
            sess = std::make_unique<PyInferenceSession>(*GetOrtEnv(), so_snapshot.options,
                                                        so_snapshot.py_ep_selection_registration,
                                                        so_snapshot.py_ep_context_data_read_registration,
                                                        arg, is_arg_file_name);
          }

          RegisterCustomOpDomains(sess.get(), so_snapshot.options);

          {
            py::gil_scoped_release release;
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load());
          }
#else
          ORT_THROW("Loading configuration from an ONNX model is not supported in this build.");
#endif
        } else {
          {
            py::gil_scoped_release release;
            sess = std::make_unique<PyInferenceSession>(*GetOrtEnv(), so_snapshot.options,
                                                        so_snapshot.py_ep_selection_registration,
                                                        so_snapshot.py_ep_context_data_read_registration);
          }
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
          RegisterCustomOpDomains(sess.get(), so_snapshot.options);
#endif

          if (is_arg_file_name) {
            py::gil_scoped_release release;
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg));
          } else {
            py::gil_scoped_release release;
            OrtPybindThrowIfError(sess->GetSessionHandle()->Load(arg.data(), narrow<int>(arg.size())));
          }
        }

        return sess;
      }))
      .def(
          "initialize_session",
          [ep_registration_fn](PyInferenceSession* sess,
                               const std::vector<std::string>& provider_types = {},
                               const ProviderOptionsVector& provider_options = {},
                               const std::unordered_set<std::string>& disabled_optimizer_names = {}) {
            OrtPybindThrowIfError(sess->BeginInitialization());
            auto finish_initialization = gsl::finally([sess]() { sess->EndInitialization(); });
            py::gil_scoped_release release;
            // If the user did not explicitly specify providers when creating InferenceSession and the SessionOptions
            // has provider information (i.e., either explicit EPs or an EP selection policy), then use the information
            // in the session options to initialize the session.
            if (provider_types.empty() && sess->HasProvidersInSessionOptions()) {
              OrtPybindThrowIfError(InitializeSessionEpsFromSessionOptions(*sess, disabled_optimizer_names));
            } else {
              InitializeSession(sess->GetSessionHandle(),
                                ep_registration_fn,
                                provider_types,
                                provider_options,
                                disabled_optimizer_names);
            }
          },
          R"pbdoc(Load a model saved in ONNX or ORT format.)pbdoc")
      .def("run",
           [](PyInferenceSession* sess, const std::vector<std::string>& output_names,
              const std::map<std::string, const py::object>& pyfeeds, RunOptions* run_options = nullptr)
               -> py::list {
             NameMLValMap feeds;
             if (run_options != nullptr && !run_options->active_adapters.empty()) {
               AppendLoraParametersAsInputs(*run_options, pyfeeds.size(), feeds);
             } else {
               feeds.reserve(pyfeeds.size());
             }

             for (const auto& feed : pyfeeds) {
               // No need to process 'None's sent in by the user
               // to feed Optional inputs in the graph.
               // We just won't include anything in the feed and ORT
               // will handle such implicit 'None's internally.
               if (!feed.second.is(py::none())) {
                 OrtValue ml_value;
                 auto px = sess->GetSessionHandle()->GetModelInputs();
                 if (!px.first.IsOK() || !px.second) {
                   throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
                 }
                 CreateGenericMLValue(px.second, GetAllocator(), feed.first, feed.second, &ml_value);
                 ThrowIfPyErrOccured();
                 feeds.insert(std::make_pair(feed.first, std::move(ml_value)));
               }
             }

             std::vector<OrtValue> fetches;
             fetches.reserve(output_names.size());
             common::Status status;

             {
               // release GIL to allow multiple python threads to invoke Run() in parallel.
               py::gil_scoped_release release;
               if (run_options != nullptr) {
                 OrtPybindThrowIfError(sess->GetSessionHandle()->Run(*run_options, feeds, output_names, &fetches));
               } else {
                 OrtPybindThrowIfError(sess->GetSessionHandle()->Run(feeds, output_names, &fetches));
               }
             }

             py::list result;
             size_t pos = 0;
             for (const auto& fet : fetches) {
               if (fet.IsAllocated()) {
                 if (fet.IsTensor()) {
                   result.append(AddTensorAsPyObj(fet, nullptr, nullptr));
                 } else if (fet.IsSparseTensor()) {
                   result.append(GetPyObjectFromSparseTensor(pos, fet, nullptr));
                 } else {
                   result.append(AddNonTensorAsPyObj(fet, nullptr, nullptr));
                 }
               } else {  // Send back None because the corresponding OrtValue was empty
                 result.append(py::none());
               }
               ++pos;
             }
             return result;
           })
      .def("run_async",
           [](py::object session,
              const std::vector<std::string>& output_names,
              const std::map<std::string, py::object>& pyfeeds,
              PyCallback callback, py::object user_data = {},
              py::object run_options = py::none())
               -> void {
             auto* sess = session.cast<PyInferenceSession*>();
             auto* run_options_ptr = run_options.is_none()
                                         ? nullptr
                                         : run_options.cast<RunOptions*>();
             if (run_options_ptr != nullptr && !run_options_ptr->active_adapters.empty()) {
               LOGS(*sess->GetSessionHandle()->GetLogger(), WARNING)
                   << "run_async has active adapters specified, but won't have an effect";
             }

             std::unique_ptr<AsyncResource> async_resource = std::make_unique<AsyncResource>();
             async_resource->session = std::move(session);
             async_resource->run_options = std::move(run_options);
             async_resource->callback = callback;
             async_resource->user_data = user_data;
             // prepare feeds
             async_resource->ReserveFeeds(pyfeeds.size());
             for (const auto& feed : pyfeeds) {
               if (!feed.second.is(py::none())) {
                 async_resource->feed_objects.push_back(feed.second);
                 OrtValue ml_value;
                 auto px = sess->GetSessionHandle()->GetModelInputs();
                 if (!px.first.IsOK() || !px.second) {
                   throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
                 }
                 CreateGenericMLValue(px.second, GetAllocator(), feed.first,
                                      async_resource->feed_objects.back(), &ml_value);
                 ThrowIfPyErrOccured();
                 async_resource->feeds.push_back(ml_value);
                 async_resource->feeds_raw.push_back(&async_resource->feeds.back());
                 async_resource->feed_names.push_back(feed.first);
                 async_resource->feed_names_raw.push_back(async_resource->feed_names.back().c_str());
               }
             }
             // prepare fetches
             async_resource->ReserveFetches(output_names.size());
             for (const auto& output_name : output_names) {
               async_resource->fetch_names.push_back(output_name);
               async_resource->fetch_names_raw.push_back(async_resource->fetch_names.back().c_str());
               async_resource->fetches_raw.push_back({});
             }
             const RunOptions* run_async_option = run_options_ptr ? run_options_ptr : &async_resource->default_run_option;
             common::Status status = sess->GetSessionHandle()->RunAsync(run_async_option,
                                                                        gsl::span(async_resource->feed_names_raw.data(), async_resource->feed_names_raw.size()),
                                                                        gsl::span(async_resource->feeds_raw.data(), async_resource->feeds_raw.size()),
                                                                        gsl::span(async_resource->fetch_names_raw.data(), async_resource->fetch_names_raw.size()),
                                                                        gsl::span(async_resource->fetches_raw.data(), async_resource->fetches_raw.size()),
                                                                        AsyncCallback,
                                                                        async_resource.get());
             if (status.IsOK()) {
               async_resource.release();
             }
             OrtPybindThrowIfError(status);
           })
      /// This method accepts a dictionary of feeds (name -> OrtValue) and the list of output_names
      /// and returns a list of python objects representing OrtValues. Each name may represent either
      /// a Tensor, SparseTensor or a TensorSequence.
      .def("run_with_ort_values", [](PyInferenceSession* sess, const py::dict& feeds, const std::vector<std::string>& output_names, RunOptions* run_options = nullptr) -> std::vector<OrtValue> {
        NameMLValMap ort_feeds;
        if (run_options != nullptr && !run_options->active_adapters.empty()) {
          AppendLoraParametersAsInputs(*run_options, feeds.size(), ort_feeds);
        } else {
          ort_feeds.reserve(feeds.size());
        }

        // item is always a copy since dict returns a value and not a ref
        // and Apple XToolChain barks
        for (const auto& item : feeds) {
          auto name = item.first.cast<std::string>();
          const OrtValue* ort_value = item.second.cast<const OrtValue*>();
          ort_feeds.emplace(name, *ort_value);
        }

        std::vector<OrtValue> fetches;
        fetches.reserve(output_names.size());
        {
          // release GIL to allow multiple python threads to invoke Run() in parallel.
          py::gil_scoped_release release;
          if (run_options != nullptr) {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Run(*run_options, ort_feeds, output_names, &fetches));
          } else {
            OrtPybindThrowIfError(sess->GetSessionHandle()->Run(ort_feeds, output_names, &fetches));
          }
        }
        return fetches;
      })
      .def("run_with_ortvaluevector", [](PyInferenceSession* sess, RunOptions run_options, const std::vector<std::string>& feed_names, const std::vector<OrtValue>& feeds, const std::vector<std::string>& fetch_names, std::vector<OrtValue>& fetches, const std::vector<OrtDevice>& fetch_devices) -> void {
        if (!run_options.active_adapters.empty()) {
          LOGS(*sess->GetSessionHandle()->GetLogger(), WARNING)
              << "run_with_ortvaluevector has active adapters specified, but won't have an effect";
        }

        // release GIL to allow multiple python threads to invoke Run() in parallel.
        py::gil_scoped_release release;
        OrtPybindThrowIfError(sess->GetSessionHandle()->Run(run_options, feed_names, feeds, fetch_names, &fetches, &fetch_devices));
      })
      .def("end_profiling", [](const PyInferenceSession* sess) -> std::string {
        return sess->GetSessionHandle()->EndProfiling();
      })
      .def_property_readonly("get_profiling_start_time_ns", [](const PyInferenceSession* sess) -> uint64_t {
        return sess->GetSessionHandle()->GetProfiling().GetStartTimeNs();
      })
      .def("get_providers", [](const PyInferenceSession* sess) -> const std::vector<std::string>& { return sess->GetSessionHandle()->GetRegisteredProviderTypes(); }, py::return_value_policy::reference_internal)
      .def("get_provider_options", [](const PyInferenceSession* sess) -> const ProviderOptionsMap& { return sess->GetSessionHandle()->GetAllProviderOptions(); }, py::return_value_policy::reference_internal)
      .def("is_webgpu_graph_capture_enabled", [](const PyInferenceSession* sess) {
        const auto* webgpu_ep =
            sess->GetSessionHandle()->GetExecutionProviders().Get(kWebGpuExecutionProvider);
        return webgpu_ep != nullptr && webgpu_ep->IsGraphCaptureEnabled(); })
      .def("webgpu_context_id", [](const PyInferenceSession* sess) {
        // WebGpuExecutionProvider::GetDeviceId() returns its WebGPU context id. Context 0 is the
        // default context, which is the only one the environment-registered shared data transfer
        // can serve; a caller-supplied instance/device gets a context id > 0.
        // Returns -1 when the session has no WebGPU EP.
        const auto* webgpu_ep =
            sess->GetSessionHandle()->GetExecutionProviders().Get(kWebGpuExecutionProvider);
        return webgpu_ep != nullptr ? webgpu_ep->GetDeviceId() : -1; })
      .def("create_ortvalue_from_shape_and_type", [](PyInferenceSession* sess, const std::vector<int64_t>& shape, py::object& numpy_element_type, const OrtDevice& device) {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(numpy_element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }

        int type_num = dtype->type_num;
        Py_DECREF(dtype);
        if (!IsNumericNumpyType(type_num)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported for numeric tensor types");
        }

        py::gil_scoped_release release;
        return CreateSessionOrtValue(sess, shape, NumpyTypeToOnnxRuntimeTensorType(type_num), device); }, py::keep_alive<0, 1>())
      .def("create_ortvalue_from_shape_and_onnx_type", [](PyInferenceSession* sess, const std::vector<int64_t>& shape, int32_t onnx_element_type, const OrtDevice& device) {
        if (onnx_element_type == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
          throw std::runtime_error("Creation of OrtValues is currently only supported for numeric tensor types");
        }

        py::gil_scoped_release release;
        return CreateSessionOrtValue(sess, shape, OnnxTypeToOnnxRuntimeTensorType(onnx_element_type), device); }, py::keep_alive<0, 1>())
      .def("release_captured_graph", [](PyInferenceSession* sess, int graph_annotation_id) {
        py::gil_scoped_release release;
        OrtPybindThrowIfError(sess->GetSessionHandle()->ReleaseCapturedGraph(graph_annotation_id)); })
      .def("get_provider_graph_assignment_info", [](const PyInferenceSession* sess) -> const std::vector<const OrtEpAssignedSubgraph*>& {
#if !defined(ORT_MINIMAL_BUILD)
        const auto* inference_session = sess->GetSessionHandle();
        const auto& sess_options = inference_session->GetSessionOptions();
        bool is_enabled =
            sess_options.config_options.GetConfigOrDefault(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "0") == "1";

        if (!is_enabled) {
          OrtPybindThrowIfError(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, !is_enabled, "Session configuration entry '",
                                                kOrtSessionOptionsRecordEpGraphAssignmentInfo,
                                                "' must be set to \"1\" to retrieve EP graph assignment information."));
        }
        return inference_session->GetEpGraphAssignmentInfo();
#else
        ORT_UNUSED_PARAMETER(sess);
        ORT_THROW("EP graph assignment information is not supported in this build");
#endif
      },
           py::return_value_policy::reference_internal, R"pbdoc(Returns information on the subgraph/nodes assigned to execution providers in the session.)pbdoc")
      .def_property_readonly("session_options", [](const PyInferenceSession* sess) -> PySessionOptions* {
            auto session_options = std::make_unique<PySessionOptions>();
            session_options->value = sess->GetSessionHandle()->GetSessionOptions();
            session_options->py_ep_selection_registration = sess->GetEpSelectionRegistration();
        session_options->py_ep_context_data_read_registration = sess->GetEpContextDataReadRegistration();
            if (session_options->py_ep_selection_registration) {
              session_options->value.ep_selection_policy.state =
                  session_options->py_ep_selection_registration.get();
            }
            if (session_options->py_ep_context_data_read_registration) {
              session_options->value.ep_context_data_read_state =
                  session_options->py_ep_context_data_read_registration.get();
            }
            return session_options.release(); }, py::return_value_policy::take_ownership)
      .def_property_readonly("inputs_meta", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetModelInputs();
            OrtPybindThrowIfError(res.first);
            return *(res.second); }, py::return_value_policy::reference_internal)
      .def_property_readonly("outputs_meta", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetModelOutputs();
            OrtPybindThrowIfError(res.first);
            return *(res.second); }, py::return_value_policy::reference_internal)
      .def_property_readonly("overridable_initializers", [](const PyInferenceSession* sess) -> const std::vector<const onnxruntime::NodeArg*>& {
            auto res = sess->GetSessionHandle()->GetOverridableInitializers();
            OrtPybindThrowIfError(res.first);
            return *(res.second); }, py::return_value_policy::reference_internal)
      .def_property_readonly("model_meta", [](const PyInferenceSession* sess) -> const onnxruntime::ModelMetadata& {
            auto res = sess->GetSessionHandle()->GetModelMetadata();
            OrtPybindThrowIfError(res.first);
            return *(res.second); }, py::return_value_policy::reference_internal)
      .def_property_readonly("input_meminfos", [](const PyInferenceSession* sess) -> py::list {
          Ort::ConstSession session(reinterpret_cast<const OrtSession*>(sess->GetSessionHandle()));
          auto inputs_mem_info = session.GetMemoryInfoForInputs();
          py::list result;
          for (const auto& info : inputs_mem_info) {
            const auto* p_info = static_cast<const OrtMemoryInfo*>(info);
            result.append(py::cast(p_info, py::return_value_policy::reference));
          }
          return result; })
      .def_property_readonly("output_meminfos", [](const PyInferenceSession* sess) -> py::list {
          Ort::ConstSession session(reinterpret_cast<const OrtSession*>(sess->GetSessionHandle()));
          auto outputs_mem_info = session.GetMemoryInfoForOutputs();
          py::list result;
          for (const auto& info : outputs_mem_info) {
            const auto* p_info = static_cast<const OrtMemoryInfo*>(info);
            result.append(py::cast(p_info, py::return_value_policy::reference));
          }
          return result; })
      .def_property_readonly("input_epdevices", [](const PyInferenceSession* sess) -> py::list {
         Ort::ConstSession session(reinterpret_cast<const OrtSession*>(sess->GetSessionHandle()));
         auto ep_devices = session.GetEpDeviceForInputs();
         py::list result;
         for (const auto& device : ep_devices) {
           const auto* p_device = static_cast<const OrtEpDevice*>(device);
           result.append(py::cast(p_device, py::return_value_policy::reference));
         }
         return result; })
      .def("run_with_iobinding", [](PyInferenceSession* sess, SessionIOBinding& io_binding, RunOptions* run_options = nullptr) -> void {

        Status status;

        if (run_options != nullptr && !run_options->active_adapters.empty()) {
          LOGS(*sess->GetSessionHandle()->GetLogger(), WARNING)
              << "run_with_iobinding has active adapters specified, but won't have an effect";
        }

        // release GIL to allow multiple python threads to invoke Run() in parallel.
        py::gil_scoped_release release;
        if (!run_options)
          status = sess->GetSessionHandle()->Run(*io_binding.Get());
        else
          status = sess->GetSessionHandle()->Run(*run_options, *io_binding.Get());
        if (!status.IsOK())
          throw std::runtime_error("Error in execution: " + status.ErrorMessage()); })
      .def("get_tuning_results", [](PyInferenceSession* sess) -> py::list {
#if !defined(ORT_MINIMAL_BUILD)
        auto results = sess->GetSessionHandle()->GetTuningResults();
        py::list ret;
        for (const auto& trs : results) {
          py::dict py_trs;
          py_trs["ep"] = trs.ep;
          py_trs["results"] = trs.results;
          py_trs["validators"] = trs.validators;
          ret.append(std::move(py_trs));
        }

        return ret;
#else
        ORT_UNUSED_PARAMETER(sess);
        ORT_THROW("TunableOp and get_tuning_results are not supported in this build.");
#endif
      })
      .def("set_ep_dynamic_options", [](PyInferenceSession* sess, const py::dict& options) {
            InlinedVector<const char*> keys;
            InlinedVector<const char*> values;
            InlinedVector<std::string> key_strings;
            InlinedVector<std::string> value_strings;

            // Reserve space to avoid reallocations
            key_strings.reserve(options.size());
            value_strings.reserve(options.size());
            keys.reserve(options.size());
            values.reserve(options.size());

            // Convert Python dict to C-style arrays
            for (const auto& item : options) {
              key_strings.emplace_back(py::str(item.first));
              value_strings.emplace_back(py::str(item.second));
              keys.push_back(key_strings.back().c_str());
              values.push_back(value_strings.back().c_str());
            }

            if (keys.empty()) {
              ORT_THROW("No options were provided");
            }

            ORT_THROW_IF_ERROR(sess->GetSessionHandle()->SetEpDynamicOptions(keys, values)); },
           R"pbdoc(Set dynamic options for execution providers.

          Args:
              options (dict): Dictionary of key-value pairs where both keys and values are strings.
                            These options will be passed to the execution providers to modify
                            their runtime behavior.

          Example:
              session.set_ep_dynamic_options({
                  "option1": "value1",
                  "option2": "value2"
              })

          Raises:
              RuntimeError: If no options are provided or if setting the options fails.
          )pbdoc")
      .def("set_tuning_results", [](PyInferenceSession* sess, py::list results, bool error_on_invalid) -> void {
#if !defined(ORT_MINIMAL_BUILD)
        std::vector<TuningResults> tuning_results;
        for (auto handle : results) {
          auto py_trs = handle.cast<py::dict>();
          TuningResults trs;
          trs.ep = py_trs["ep"].cast<py::str>();

          for (const auto [py_op_sig, py_kernel_map] : py_trs["results"].cast<py::dict>()) {
            KernelMap kernel_map;
            for (const auto [py_params_sig, py_kernel_id] : py_kernel_map.cast<py::dict>()) {
              kernel_map[py_params_sig.cast<py::str>()] = py_kernel_id.cast<py::int_>();
            }
            trs.results[py_op_sig.cast<py::str>()] = kernel_map;
          }

          for (const auto [k, v] : py_trs["validators"].cast<py::dict>()) {
            trs.validators[k.cast<py::str>()] = v.cast<py::str>();
          }

          tuning_results.emplace_back(std::move(trs));
        }

        Status status = sess->GetSessionHandle()->SetTuningResults(tuning_results, error_on_invalid);
        if (!status.IsOK()) {
          throw std::runtime_error("Error in execution: " + status.ErrorMessage());
        }
#else
        ORT_UNUSED_PARAMETER(sess);
        ORT_UNUSED_PARAMETER(results);
        ORT_UNUSED_PARAMETER(error_on_invalid);
        ORT_THROW("TunableOp and set_tuning_results are not supported in this build.");
#endif
      });

  py::enum_<onnxruntime::ArenaExtendStrategy>(m, "ArenaExtendStrategy", py::arithmetic())
      .value("kNextPowerOfTwo", onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo)
      .value("kSameAsRequested", onnxruntime::ArenaExtendStrategy::kSameAsRequested)
      .export_values();

  // Must use a std::shared_ptr to hold OrtExternalInitializerInfo because the same instances is passed
  // between C++ and Python (and Python cannot transfer ownership to C++).
  py::class_<OrtExternalInitializerInfo, std::shared_ptr<OrtExternalInitializerInfo>> ort_external_initializer_info_binding(
      m, "OrtExternalInitializerInfo",
      R"pbdoc(Location information for initializer data stored in an external file)pbdoc");
  ort_external_initializer_info_binding
      .def(py::init([](const std::basic_string<ORTCHAR_T>& filepath, int64_t file_offset, size_t byte_size) {
#if !defined(ORT_MINIMAL_BUILD)
        return std::make_shared<OrtExternalInitializerInfo>(filepath, file_offset, byte_size);
#else
        ORT_UNUSED_PARAMETER(filepath);
        ORT_UNUSED_PARAMETER(file_offset);
        ORT_UNUSED_PARAMETER(byte_size);
        ORT_THROW("OrtExternalInitializerInfo creation is not supported in this build");
#endif
      }))
      .def_property_readonly(
          "filepath",
          [](OrtExternalInitializerInfo* info) -> std::basic_string<ORTCHAR_T> { return info->GetRelPath(); },
          R"pbdoc(The relative path to the file in which initializer data is stored.)pbdoc")
      .def_property_readonly(
          "file_offset",
          [](OrtExternalInitializerInfo* info) -> int64_t { return info->GetOffset(); },
          R"pbdoc(The file byte offset where the initializer data is stored.)pbdoc")
      .def_property_readonly(
          "byte_size",
          [](OrtExternalInitializerInfo* info) -> size_t { return info->GetLength(); },
          R"pbdoc(The byte size of the initializer data in the file.)pbdoc");

  py::enum_<OrtCompileApiFlags>(m, "OrtCompileApiFlags", py::arithmetic())
      .value("NONE", OrtCompileApiFlags_NONE)
      .value("ERROR_IF_NO_NODES_COMPILED", OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED)
      .value("ERROR_IF_OUTPUT_FILE_EXISTS", OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS);

  py::class_<PyModelCompiler>(m, "ModelCompiler",
                              R"pbdoc(This is the class used to compile an ONNX model.)pbdoc")
      .def(py::init([](const PySessionOptions& sess_options,
                       std::string path_or_bytes,
                       bool is_path,
                       bool embed_compiled_data_into_model = false,
                       std::string external_initializers_file_path = {},
                       size_t external_initializers_size_threshold = 1024,
                       uint32_t flags = OrtCompileApiFlags_NONE,
                       GraphOptimizationLevel graph_optimization_level = GraphOptimizationLevel::ORT_DISABLE_ALL,
                       const PyGetInitializerLocationFunc& py_get_initializer_location_func = nullptr) {
#if !defined(ORT_MINIMAL_BUILD)
        std::unique_ptr<PyModelCompiler> result;
        OrtPybindThrowIfError(PyModelCompiler::Create(result, GetEnv(), sess_options,
                                                      std::move(path_or_bytes), is_path,
                                                      embed_compiled_data_into_model,
                                                      external_initializers_file_path,
                                                      external_initializers_size_threshold,
                                                      flags, graph_optimization_level,
                                                      py_get_initializer_location_func));
        return result;
#else
        ORT_UNUSED_PARAMETER(sess_options);
        ORT_UNUSED_PARAMETER(path_or_bytes);
        ORT_UNUSED_PARAMETER(is_path);
        ORT_UNUSED_PARAMETER(embed_compiled_data_into_model);
        ORT_UNUSED_PARAMETER(external_initializers_file_path);
        ORT_UNUSED_PARAMETER(external_initializers_size_threshold);
        ORT_UNUSED_PARAMETER(flags);
        ORT_UNUSED_PARAMETER(graph_optimization_level);
        ORT_UNUSED_PARAMETER(py_get_initializer_location_func);
        ORT_THROW("Compile API is not supported in this build.");
#endif
      }))
      .def(
          "compile_to_file",
          [](PyModelCompiler* model_compiler, std::string output_model_path = {}) {
#if !defined(ORT_MINIMAL_BUILD)
            OrtPybindThrowIfError(model_compiler->CompileToFile(output_model_path));
#else
            ORT_UNUSED_PARAMETER(model_compiler);
            ORT_UNUSED_PARAMETER(output_model_path);
            ORT_THROW("Compile API is not supported in this build.");
#endif
          },
          R"pbdoc(Compile an ONNX model into a file.)pbdoc")
      .def(
          "compile_to_bytes",
          [](PyModelCompiler* model_compiler) -> py::bytes {
#if !defined(ORT_MINIMAL_BUILD)
            std::string output_bytes;
            OrtPybindThrowIfError(model_compiler->CompileToBytes(output_bytes));
            return py::bytes(output_bytes);
#else
            ORT_UNUSED_PARAMETER(model_compiler);
            ORT_THROW("Compile API is not supported in this build.");
#endif
          },
          R"pbdoc(Compile an ONNX model into a buffer.)pbdoc")
      .def(
          "compile_to_stream",
          [](PyModelCompiler* model_compiler, PyOutStreamWriteFunc& py_stream_write_func) {
#if !defined(ORT_MINIMAL_BUILD)
            OrtPybindThrowIfError(model_compiler->CompileToOutStream(py_stream_write_func));
#else
            ORT_UNUSED_PARAMETER(model_compiler);
            ORT_UNUSED_PARAMETER(py_stream_write_func);
            ORT_THROW("Compile API is not supported in this build.");
#endif
          },
          R"pbdoc(Compile an ONNX model into an output stream using the provided write functor.)pbdoc")
#if !defined(ORT_MINIMAL_BUILD)
      .def(
          "set_ep_context_data_write_func",
          [](PyModelCompiler* model_compiler, PyEpContextDataWriteFunc write_func) {
            model_compiler->SetEpContextDataWriteFunc(std::move(write_func));
          },
          py::arg("write_func"),
          R"pbdoc(Register a callback that receives external EPContext data during compilation.

        ORT may invoke the callback concurrently. Synchronize any shared Python state.)pbdoc")
      .def(
          "clear_ep_context_data_write_func",
          [](PyModelCompiler* model_compiler) {
            model_compiler->ClearEpContextDataWriteFunc();
          },
          R"pbdoc(Clear the external EPContext data write callback.)pbdoc")
#endif  // !defined(ORT_MINIMAL_BUILD)
      ;
}

bool InitArray() {
  import_array1(false);
  return true;
}

}  // namespace python
}  // namespace onnxruntime
