// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocatormgr.h"
#include "core/session/environment.h"
#include "core/framework/ml_value.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace python {

extern const char* PYTHON_ORTVALUE_OBJECT_NAME;
extern const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR;

bool IsNumericNumpyType(int npy_type);

bool IsNumericNumpyArray(const pybind11::object& py_object);

bool IsNumpyArray(pybind11::object& obj);

int GetNumpyArrayType(const pybind11::object& obj);

bool IsNumericDType(const pybind11::dtype& dtype);

TensorShape GetShape(const pybind11::array& arr);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type);

MLDataType NumpyToOnnxRuntimeTensorType(int numpy_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);

void CpuToCpuMemCpy(void*, const void*, size_t);

void CopyDataToTensor(const pybind11::array& py_array, int npy_type, Tensor& tensor, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

pybind11::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                                  const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

pybind11::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager);

pybind11::object AddNonTensorAsPyObj(const OrtValue& val,
                                     const DataTransferManager* data_transfer_manager,
                                     const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

OrtMemoryInfo GetMemoryInfoPerDeviceType(const OrtDevice& ort_device);

#ifdef USE_CUDA

void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes);

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCudaToHostMemCpyFunction();

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

std::unique_ptr<IDataTransfer> GetGPUDataTransfer();

#endif

#ifdef USE_ROCM

bool IsRocmDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetRocmAllocator(OrtDevice::DeviceId id);

void CpuToRocmMemCpy(void* dst, const void* src, size_t num_bytes);

void RocmToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetRocmToHostMemCpyFunction();

#endif

void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc,
                          const std::string& name_input, const pybind11::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array = false, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

void GetPyObjFromTensor(const Tensor& rtensor, pybind11::object& obj,
                        const DataTransferManager* data_transfer_manager = nullptr,
                        const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions = nullptr);

template <class T>
struct DecRefFn {
  void operator()(T* pyobject) const {
    Py_XDECREF(pyobject);
  }
};

template <class T>
using UniqueDecRefPtr = std::unique_ptr<T, DecRefFn<T>>;

}  // namespace python
}  // namespace onnxruntime
