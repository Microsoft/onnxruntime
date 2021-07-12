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
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/session/IOBinding.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;

void addIoBindingMethods(pybind11::module& m) {
  py::class_<SessionIOBinding> session_io_binding(m, "SessionIOBinding");
  session_io_binding
      .def(py::init([](PyInferenceSession* sess) {
        auto sess_io_binding = std::make_unique<SessionIOBinding>(sess->GetSessionHandle());
        return sess_io_binding;
      }))
      // May create Tensor/Sequence based OrtValues. Use bind_ortvalue_input for universal binding.
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, py::object& arr_on_cpu) -> void {
        InferenceSession* sess = io_binding->GetInferenceSession();
        auto px = sess->GetModelInputs();
        if (!px.first.IsOK() || !px.second) {
          throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
        }

        // For now, limit binding support to only non-string Tensors
        // TODO: Support non-tensors
        const auto& def_list = *px.second;
        onnx::TypeProto type_proto;
        if (!CheckIfTensor(def_list, name, type_proto)) {
          throw std::runtime_error("Only binding Tensors is currently supported");
        }

        ORT_ENFORCE(utils::HasTensorType(type_proto) && utils::HasElemType(type_proto.tensor_type()));
        if (type_proto.tensor_type().elem_type() == onnx::TensorProto::STRING) {
          throw std::runtime_error("Only binding non-string Tensors is currently supported");
        }

        OrtValue ml_value;
        // Set the parameter `accept_only_numpy_array` to `true` (we only support binding Tensors)
        CreateGenericMLValue(px.second, GetAllocator(), name, arr_on_cpu, &ml_value, true);

        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when bind input: " + status.ErrorMessage());
        }
      })
      // This binds input as a Tensor that wraps memory pointer along with the OrtMemoryInfo
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, py::object& element_type, std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        ORT_ENFORCE(data_ptr != 0, "Pointer to data memory is not valid");

        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device, device.Id());
        std::unique_ptr<Tensor> p_tensor =
            std::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, reinterpret_cast<void*>(data_ptr), info);

        OrtValue ml_value;
        ml_value.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding input: " + status.ErrorMessage());
        }
      })
      // This binds input as an OrtValue which may contain various types and point to the user pre-allocated
      // buffers
      .def("bind_ortvalue_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtValue& ml_value) -> void {
        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding input: " + status.ErrorMessage());
        }
      })
      // This binds output to a pre-allocated memory as a Tensor
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, py::object& element_type, std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        ORT_ENFORCE(data_ptr != 0, "Pointer to data memory is not valid");

        InferenceSession* sess = io_binding->GetInferenceSession();
        auto px = sess->GetModelOutputs();
        if (!px.first.IsOK() || !px.second) {
          throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
        }

        // For now, limit binding support to only non-string Tensors
        const auto& def_list = *px.second;
        onnx::TypeProto type_proto;
        if (!CheckIfTensor(def_list, name, type_proto)) {
          throw std::runtime_error("Only binding Tensors is currently supported");
        }

        ORT_ENFORCE(utils::HasTensorType(type_proto) && utils::HasElemType(type_proto.tensor_type()));
        if (type_proto.tensor_type().elem_type() == onnx::TensorProto::STRING) {
          throw std::runtime_error("Only binding non-string Tensors is currently supported");
        }

        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device, device.Id());

        std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(NumpyTypeToOnnxRuntimeType(type_num), shape, reinterpret_cast<void*>(data_ptr), info);

        OrtValue ml_value;
        ml_value.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

        auto status = io_binding->Get()->BindOutput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
        }
      })
      // This binds output to a device. Meaning that the output OrtValue must be allocated on a specific device.
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device) -> void {
        auto status = io_binding->Get()->BindOutput(name, device);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
        }
      })
      // Binds output to a pre-constructed OrtValue which may contain various elements (e.g. Tensor/SparseTensor/TensorSequece)
      .def("bind_ortvalue_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtValue& ml_value) -> void {
        auto status = io_binding->Get()->BindOutput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
        }
      })
      .def("clear_binding_inputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearInputs();
      })
      .def("clear_binding_outputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearOutputs();
      })
      .def(
          "get_outputs", [](const SessionIOBinding* io_binding) -> const std::vector<OrtValue>& {
            return io_binding->Get()->GetOutputs();
          },
          py::return_value_policy::reference_internal)
      .def("copy_outputs_to_cpu", [](const SessionIOBinding* io_binding) -> std::vector<py::object> {
        const std::vector<OrtValue>& outputs = io_binding->Get()->GetOutputs();
        std::vector<py::object> rfetch;
        rfetch.reserve(outputs.size());
        size_t pos = 0;
        const auto& dtm = io_binding->GetInferenceSession()->GetDataTransferManager();
        for (const auto& ort_value : outputs) {
          if (ort_value.IsTensor()) {
            rfetch.push_back(AddTensorAsPyObj(ort_value, &dtm, nullptr));
          } else if (ort_value.IsSparseTensor()) {
            rfetch.push_back(GetPyObjectFromSparseTensor(pos, ort_value, &dtm));
          } else {
            rfetch.push_back(AddNonTensorAsPyObj(ort_value, &dtm, nullptr));
          }
          ++pos;
        }
        return rfetch;
      });
}

}  // namespace python
}  // namespace onnxruntime