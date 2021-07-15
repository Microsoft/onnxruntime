// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;

void addOrtValueMethods(pybind11::module& m) {
  py::class_<OrtValue> ortvalue_binding(m, "OrtValue");
  ortvalue_binding
      // Factory method to create an OrtValue (Tensor) from the given Numpy object
      // The Tensor allocates and manages its own memory (on the specified device) and copies data from the Numpy data buffer
      .def_static("ortvalue_from_numpy", [](py::object& array_on_cpu, const OrtDevice& device) {
        if (!IsNumericNumpyArray(array_on_cpu)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto ml_value = std::make_unique<OrtValue>();

        // The tensor's memory is allocated on the CPU
        if (strcmp(GetDeviceName(device), CPU) == 0) {
          // InputDeflist is null because OrtValue creation is not tied to a specific model
          // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)

          CreateGenericMLValue(nullptr, GetAllocator(), "", array_on_cpu, ml_value.get(), true);
        } else if (strcmp(GetDeviceName(device), CUDA) == 0) {
      // The tensor's memory is allocated on CUDA

#ifdef USE_CUDA
          if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          // InputDeflist is null because OrtValue creation is not tied to a specific model
          // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
          // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors in CUDA
          CreateGenericMLValue(nullptr, GetCudaAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToCudaMemCpy);
#elif USE_ROCM
          if (!IsRocmDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          // InputDeflist is null because OrtValue creation is not tied to a specific model
          // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
          // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors in CUDA
          CreateGenericMLValue(nullptr, GetRocmAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToRocmMemCpy);

#else
        throw std::runtime_error(
            "Can't allocate memory on the CUDA device using this package of OnnxRuntime. "
            "Please use the CUDA package of OnnxRuntime to use this feature.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
        }

        return ml_value;
      })

      // Factory method to create an OrtValue (Tensor) from the given shape and element type with memory on the specified device
      // The memory is left uninitialized
      .def_static("ortvalue_from_shape_and_type", [](const std::vector<int64_t>& shape, py::object& element_type, const OrtDevice& device) {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }

        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        if (!IsNumericNumpyType(type_num)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto ml_value = std::make_unique<OrtValue>();

        std::unique_ptr<Tensor> tensor;
        // The tensor's memory is allocated on the CPU
        if (strcmp(GetDeviceName(device), CPU) == 0) {
          tensor = std::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, GetAllocator());
        } else if (strcmp(GetDeviceName(device), CUDA) == 0) {
      // The tensor's memory is allocated on CUDA
#ifdef USE_CUDA
          if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          tensor = std::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, GetCudaAllocator(device.Id()));
#else
      throw std::runtime_error(
          "Can't allocate memory on the CUDA device using this package of OnnxRuntime. "
          "Please use the CUDA package of OnnxRuntime to use this feature.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
        }

        auto ml_tensor = DataTypeImpl::GetType<Tensor>();
        ml_value->Init(tensor.release(),
                       ml_tensor,
                       ml_tensor->GetDeleteFunc());

        return ml_value;
      })
      // Get a pointer to Tensor data
      .def("data_ptr", [](OrtValue* ml_value) -> int64_t {
        // TODO: Assumes that the OrtValue is a Tensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are currently supported");

        auto* tensor = ml_value->GetMutable<Tensor>();

        if (tensor->Shape().Size() == 0) {
          return 0;
        }

        // Should cover x86 and x64 platforms
        return reinterpret_cast<int64_t>(tensor->MutableDataRaw());
      })
      .def("device_name", [](const OrtValue* ort_value) -> std::string {
        if (ort_value->IsTensor()) {
          return std::string(GetDeviceName(ort_value->Get<Tensor>().Location().device));
        } else {
          ORT_THROW("Only OrtValues that are Tensors are currently supported");
        }
      })
      .def("shape", [](const OrtValue* ort_value) -> py::list {
        // OrtValue can only be a Tensor/SparseTensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ort_value->IsTensor() || ort_value->IsSparseTensor(),
                    "Only OrtValues that are Tensors/SpareTensors are currently supported");

        py::list shape_arr;
        const auto& dims = (ort_value->IsTensor())
                               ? ort_value->Get<Tensor>().Shape().GetDims()
                               : ort_value->Get<SparseTensor>().Shape().GetDims();

        for (auto dim : dims) {
          // For sequence tensors - we would append a list of dims to the outermost list
          // For now only tensors are supported in OrtValue
          shape_arr.append(dim);
        }

        return shape_arr;
      })
      .def("data_type", [](const OrtValue* ort_value) -> std::string {
        const ONNX_NAMESPACE::TypeProto* type_proto;
        // Handle gutless types first to get the actual type
        if (ort_value->IsTensor()) {
          auto elem_type = ort_value->Get<Tensor>().GetElementType();
          type_proto = DataTypeImpl::TensorTypeFromONNXEnum(elem_type)->GetTypeProto();
        } else if (ort_value->IsSparseTensor()) {
          auto elem_type = ort_value->Get<SparseTensor>().Values().GetElementType();
          type_proto = DataTypeImpl::SparseTensorTypeFromONNXEnum(elem_type)->GetTypeProto();
        } else if (ort_value->IsTensorSequence()) {
          auto elem_type = ort_value->Get<TensorSeq>().DataType()->AsPrimitiveDataType()->GetDataType();
          type_proto = DataTypeImpl::SequenceTensorTypeFromONNXEnum(elem_type)->GetTypeProto();
        } else {
          // Plane sequences and maps probably have their specific type
          type_proto = ort_value->Type()->GetTypeProto();
        }

        ORT_ENFORCE(type_proto != nullptr, "Unknown type of OrtValue: ", ort_value->Type());

        return *ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto);
      })
      .def("has_element", [](const OrtValue* ort_value) -> bool {
        return ort_value->HasElement();
      })
      .def("is_tensor", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsTensor();
      })
      .def("is_sparse_tensor", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsSparseTensor();
      })
      .def("is_tensor_sequence", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsTensorSequence();
      })
      // Converts Tensor into a numpy array
      .def("numpy", [](const OrtValue* ml_value) -> py::object {
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are convertible to Numpy objects");

        py::object obj;

#ifdef USE_CUDA
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetCudaToHostMemCpyFunction());
#elif USE_ROCM
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetRocmToHostMemCpyFunction());
#else
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, nullptr);
#endif
        return obj;
      })
#ifdef ENABLE_TRAINING
      .def("to_dlpack", [](OrtValue* ort_value) -> py::object {
        return py::reinterpret_steal<py::object>(ToDlpack(*ort_value));
      })
      .def_static("from_dlpack", [](py::object data, bool is_bool_tensor = false) {
        return FromDlpack(data.ptr(), is_bool_tensor);
      })
#endif
      ;
}

}  // namespace python
}  // namespace onnxruntime
