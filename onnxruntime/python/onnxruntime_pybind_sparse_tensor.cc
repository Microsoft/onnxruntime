// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"
#include "pybind11/numpy.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"

#include "core/framework/data_types_internal.h"
#include "core/providers/get_execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_bridge_ort.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime::logging;

namespace {
// Create a pybind11:dtype numpy instance using ONNX Tensor Element Type
template <typename T>
struct MakeDType {
  py::dtype operator()() const {
    return py::dtype::of<T>();
  }
};
}  // namespace

class PySparseCooView : public SparseTensor::CooView {
  py::object parent_;

 public:
  PySparseCooView(const SparseTensor::CooView& view, const py::object& parent) noexcept
      : SparseTensor::CooView(view), parent_(parent) {}
};

class PySparseCsrView : public SparseTensor::CsrView {
  py::object parent_;

 public:
  PySparseCsrView(const SparseTensor::CsrView& view, const py::object& parent) noexcept
      : SparseTensor::CsrView(view), parent_(parent) {}
};

class PySparseBlockSparseView : public SparseTensor::BlockSparseView {
  py::object parent_;
 public:
  PySparseBlockSparseView(const SparseTensor::BlockSparseView& view, const py::object& parent) noexcept
      : SparseTensor::BlockSparseView(view), parent_(parent) {}
};

void addSparseTensorMethods(pybind11::module& m) {
  py::enum_<OrtSparseFormat>(m, "OrtSparseFormat")
      .value("ORT_SPARSE_UNDEFINED", OrtSparseFormat::ORT_SPARSE_UNDEFINED)
      .value("ORT_SPARSE_COO", OrtSparseFormat::ORT_SPARSE_COO)
      .value("ORT_SPARSE_CSRC", OrtSparseFormat::ORT_SPARSE_CSRC)
      .value("ORT_SPARSE_BLOCK_SPARSE", OrtSparseFormat::ORT_SPARSE_BLOCK_SPARSE);

  py::class_<PySparseCooView>(m, "SparseCooView")
      // Returns a numpy array of COO indices backed by Sparse Tensor memory
      // be aware that indices may reside on GPU if Sparse Tensor is on GPU
      .def("indices", [](const PySparseCooView* view) -> py::array {
        const auto& indices = view->Index();
        // Since rep is a registered pybind object, it will create an extra reference
        // as base object for numpy array to make sure rep python object does not go away
        // while numpy array view is in use.
        // See https://github.com/pybind/pybind11/issues/2271
        py::array result(indices.Shape().GetDims(), indices.Data<int64_t>(), py::cast(*view));
        assert(!result.owndata());
        // Set a read-only flag
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(result.ptr()), NPY_ARRAY_WRITEABLE);
        return result;
      });

  py::class_<PySparseCsrView>(m, "SparseCsrView")
      .def("inner", [](const PySparseCsrView* view) -> py::array {
        const auto& indices = view->Inner();
        // Since rep is a registered pybind object, it will create an extra reference
        // as base object for numpy array to make sure rep python object does not go away
        // while numpy array view is in use.
        // See https://github.com/pybind/pybind11/issues/2271
        py::array result(indices.Shape().GetDims(), indices.Data<int64_t>(), py::cast(*view));
        assert(!result.owndata());
        // Set a read-only flag
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(result.ptr()), NPY_ARRAY_WRITEABLE);
        return result;
      })
      .def("outer", [](const PySparseCsrView* view) -> py::array {
        const auto& indices = view->Outer();
        // Since rep is a registered pybind object, it will create an extra reference
        // as base object for numpy array to make sure rep python object does not go away
        // while numpy array view is in use.
        // See https://github.com/pybind/pybind11/issues/2271
        py::array result(indices.Shape().GetDims(), indices.Data<int64_t>(), py::cast(*view));
        assert(!result.owndata());
        // Set a read-only flag
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(result.ptr()), NPY_ARRAY_WRITEABLE);
        return result;
      });

  py::class_<PySparseBlockSparseView>(m, "SparseBlockSparseView")
      .def("index", [](const PySparseBlockSparseView* view) -> py::array {
        const auto& indices = view->Index();
        // Since rep is a registered pybind object, it will create an extra reference
        // as base object for numpy array to make sure rep python object does not go away
        // while numpy array view is in use.
        // See https://github.com/pybind/pybind11/issues/2271
        py::array result(indices.Shape().GetDims(), indices.Data<int64_t>(), py::cast(*view));
        assert(!result.owndata());
        // Set a read-only flag
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(result.ptr()), NPY_ARRAY_WRITEABLE);
        return result;
      });

  py::class_<PySparseTensor>(m, "SparseTensor")
      // Factor method to create a COO Sparse Tensor from numpy arrays acting as backing storage.
      // Numeric arrays memory is used as is with reference count increment. All other supported
      // types are copied and supported only on CPU.
      // Use numpy.ascontiguousarray() to obtain contiguous array of values and indices if necessary
      // py_dense_shape - numpy dense shape of the sparse tensor
      // ort_device - desribes the allocation. Only primitive types allocations can be mapped to
      // py_values - contiguous and homogeneous numpy array of values
      // py_indices - contiguous numpy array of int64_t indices
      // ort_device - where the value and indices buffers are allocated. For non-primitive types,
      //              only cpu device is supported. There is not a way to verify that ort_device
      //              accurately describes the memory that is backing values and indices.
      .def_static("sparse_coo_from_numpy", [](const std::vector<int64_t>& py_dense_shape, const py::array& py_values, const py::array_t<int64_t>& py_indices, const OrtDevice& ort_device) -> std::unique_ptr<PySparseTensor> {
        if (1 != py_values.ndim()) {
          ORT_THROW("Expecting values 1-D numpy values array for COO format. Got dims: ", py_values.ndim());
        }

        TensorShape dense_shape(py_dense_shape);
        auto values_type = GetNumpyArrayType(py_values);
        auto ml_type = NumpyToOnnxRuntimeTensorType(values_type);

        std::unique_ptr<PySparseTensor> result;
        if (IsNumericNumpyType(values_type)) {
          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_values.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of values");
          }

          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_indices.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of indices");
          }

          // create references to make sure storage does not disappear
          std::vector<py::object> reference_holders = {py_values, py_indices};
          OrtMemoryInfo mem_info = GetMemoryInfoPerDeviceType(ort_device);
          TensorShape values_shape{py_values.size()};
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, values_shape,
                                                              const_cast<void*>(py_values.data()), mem_info);
          auto index_span = gsl::make_span(py_indices.data(), py_indices.size());
          ORT_THROW_IF_ERROR(sparse_tensor->UseCooIndex(index_span));
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor), std::move(reference_holders));
        } else if (values_type == NPY_UNICODE || values_type == NPY_STRING) {
          if (ort_device.Type() != OrtDevice::CPU) {
            throw std::runtime_error("Only CPU based devices are supported for non-numeric datatypes");
          }
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, GetAllocator());
          auto mutator = sparse_tensor->MakeCooData(py_values.size(), py_indices.size());
          CopyDataToTensor(py_values, values_type, mutator.Values());
          CopyDataToTensor(py_indices, GetNumpyArrayType(py_indices), mutator.Index());
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor));
        } else {
          ORT_THROW("Unsupported values data type: ", values_type);
        }
        return result;
      })
      // Factor method to create a CSR Sparse Tensor from numpy arrays acting as backing storage.
      // Numeric arrays memory is used as is with reference count increment. All other supported
      // types are copied and supported only on CPU.
      // Use numpy.ascontiguousarray() to obtain contiguous array of values and indices if necessary
      // py_dense_shape - numpy dense shape of the sparse tensor
      // py_values - contiguous and homogeneous numpy array of values
      // py_inner_indices - contiguous numpy array of int64_t indices
      // py_outer_indices - contiguous numpy array of int64_t indices
      // ort_device - where the value and indices buffers are allocated. For non-primitive types,
      //              only cpu device is supported. There is not a way to verify that ort_device
      //              accurately describes the memory that is backing values and indices.
      .def_static("sparse_csr_from_numpy", [](const std::vector<int64_t>& py_dense_shape, const py::array& py_values, const py::array_t<int64_t>& py_inner_indices, const py::array_t<int64_t>& py_outer_indices, const OrtDevice& ort_device) -> std::unique_ptr<PySparseTensor> {
        if (1 != py_values.ndim() || 1 != py_inner_indices.ndim() || 1 != py_outer_indices.ndim()) {
          ORT_THROW("Expecting all data to be 1-D numpy arrays for CSR format.");
        }

        TensorShape dense_shape(py_dense_shape);
        auto values_type = GetNumpyArrayType(py_values);
        auto ml_type = NumpyToOnnxRuntimeTensorType(values_type);

        std::unique_ptr<PySparseTensor> result;
        if (IsNumericNumpyType(values_type)) {
          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_values.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of values");
          }

          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_inner_indices.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of indices");
          }

          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_outer_indices.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of indices");
          }

          // go ahead and create references to make sure storage does not disappear
          std::vector<py::object> reference_holders = {py_values, py_inner_indices, py_outer_indices};
          OrtMemoryInfo mem_info = GetMemoryInfoPerDeviceType(ort_device);
          TensorShape values_shape{py_values.size()};
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, values_shape,
                                                              const_cast<void*>(py_values.data()), mem_info);
          auto inner_span = gsl::make_span<int64_t>(const_cast<int64_t*>(py_inner_indices.data()), py_inner_indices.size());
          auto outer_span = gsl::make_span<int64_t>(const_cast<int64_t*>(py_outer_indices.data()), py_outer_indices.size());
          ORT_THROW_IF_ERROR(sparse_tensor->UseCsrIndices(inner_span, outer_span));
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor), std::move(reference_holders));
        } else if (values_type == NPY_UNICODE || values_type == NPY_STRING) {
          if (ort_device.Type() != OrtDevice::CPU) {
            throw std::runtime_error("Only CPU based devices are supported for non-numeric datatypes");
          }
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, GetAllocator());
          auto mutator = sparse_tensor->MakeCsrData(py_values.size(), py_inner_indices.size(), py_outer_indices.size());
          CopyDataToTensor(py_values, values_type, mutator.Values());
          CopyDataToTensor(py_inner_indices, GetNumpyArrayType(py_inner_indices), mutator.Inner());
          CopyDataToTensor(py_outer_indices, GetNumpyArrayType(py_outer_indices), mutator.Outer());
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor));
        } else {
          ORT_THROW("Unsupported values data type: ", values_type);
        }

        return result;
      })
      // Factor method to create a BlockSparse Tensor from numpy arrays acting as backing storage.
      // Numeric arrays memory is used as is with reference count increment. All other supported
      // types are copied and supported only on CPU.
      // Use numpy.ascontiguousarray() to obtain contiguous array of values and indices if necessary
      // py_dense_shape - numpy dense shape of the sparse tensor
      // ort_device - desribes the allocation. Only primitive types allocations can be mapped to
      // py_values - contiguous and homogeneous numpy array of values
      // py_indices - contiguous numpy array of int32_t indices
      // ort_device - where the value and indices buffers are allocated. For non-primitive types,
      //              only cpu device is supported. There is not a way to verify that ort_device
      //              accurately describes the memory that is backing values and indices.
      .def_static("blocksparse_from_numpy", [](const std::vector<int64_t>& py_dense_shape,
                                               const py::array& py_values,
                                               const py::array_t<int32_t>& py_indices,
                                               const OrtDevice& ort_device) -> std::unique_ptr<PySparseTensor> {
        TensorShape dense_shape(py_dense_shape);
        TensorShape values_shape = GetShape(py_values);
        TensorShape index_shape = GetShape(py_indices);
        auto values_type = GetNumpyArrayType(py_values);
        auto ml_type = NumpyToOnnxRuntimeTensorType(values_type);

        std::unique_ptr<PySparseTensor> result;
        if (IsNumericNumpyType(values_type)) {
          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_values.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of values");
          }

          if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(py_indices.ptr()))) {
            throw std::runtime_error("Require contiguous numpy array of indices");
          }

          // create references to make sure storage does not disappear
          std::vector<py::object> reference_holders = {py_values, py_indices};
          OrtMemoryInfo mem_info = GetMemoryInfoPerDeviceType(ort_device);
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, values_shape,
                                                              const_cast<void*>(py_values.data()), mem_info);
          ORT_THROW_IF_ERROR(sparse_tensor->UseBlockSparseIndices(index_shape, py_indices.data()));
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor), std::move(reference_holders));
        } else if (values_type == NPY_UNICODE || values_type == NPY_STRING) {
          if (ort_device.Type() != OrtDevice::CPU) {
            throw std::runtime_error("Only CPU based devices are supported for non-numeric datatypes");
          }
          auto sparse_tensor = std::make_unique<SparseTensor>(ml_type, dense_shape, GetAllocator());
          auto mutator = sparse_tensor->MakeBlockSparseData(values_shape, index_shape);
          CopyDataToTensor(py_values, values_type, mutator.Values());
          CopyDataToTensor(py_indices, GetNumpyArrayType(py_indices), mutator.Index());
          result = std::make_unique<PySparseTensor>(std::move(sparse_tensor));
        } else {
          ORT_THROW("Unsupported values data type: ", values_type);
        }
        return result;
      })
      // Returns a numpy array that is backed by SparseTensor values memory
      // be aware that it may be on GPU
      .def("values", [](const PySparseTensor* py_tensor) -> py::array {
        const SparseTensor& sparse_tensor = py_tensor->Instance();
        if (sparse_tensor.Format() == SparseFormat::kUndefined) {
          ORT_THROW("This sparse tensor instance does not contain data");
        }
        if (sparse_tensor.IsDataTypeString()) {
          // Strings can not be on GPU but we can not expose them as mapped memory
          // So we need to create a copy.
          py::list str_list;
          auto str_span = sparse_tensor.Values().DataAsSpan<std::string>();
          for (const auto& s : str_span) {
            // valid UTF-8 to python UNICODE is handled automatically
            str_list.append(py::cast(s));
          }
          return py::cast<py::array>(str_list);
        } else {
          utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
              t_disp(sparse_tensor.GetElementType());
          auto dtype = t_disp.InvokeRet<py::dtype, MakeDType>();
          const auto& values = sparse_tensor.Values();
          // See https://github.com/pybind/pybind11/issues/2271
          py::array result(dtype, values.Shape().GetDims(), values.DataRaw(), py::cast(*py_tensor));
          assert(!result.owndata());
          // Set a read-only flag
          PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(result.ptr()), NPY_ARRAY_WRITEABLE);
          return result;
        }
      })
      // Returns a Coo view of data
      .def(
          "get_coo_data", [](const PySparseTensor* py_tensor) -> std::unique_ptr<PySparseCooView> {
            const SparseTensor& sparse_tensor = py_tensor->Instance();
            if (sparse_tensor.Format() != SparseFormat::kCoo) {
              ORT_THROW("This sparse tensor does not contain COO format");
            }
            return std::make_unique<PySparseCooView>(sparse_tensor.AsCoo(), py::cast(*py_tensor));
          })
      // Returns a CSR view of data
      .def(
          "get_csrc_data", [](const PySparseTensor* py_tensor) -> std::unique_ptr<PySparseCsrView> {
            const SparseTensor& sparse_tensor = py_tensor->Instance();
            if (sparse_tensor.Format() != SparseFormat::kCsrc) {
              ORT_THROW("This sparse tensor does not contain CSR(C) format");
            }
            return std::make_unique<PySparseCsrView>(sparse_tensor.AsCsr(), py::cast(*py_tensor));
          })
      // Returns a blocksparse view of data
      .def(
          "get_blocksparse_data", [](const PySparseTensor* py_tensor) -> std::unique_ptr<PySparseBlockSparseView> {
            const SparseTensor& sparse_tensor = py_tensor->Instance();
            if (sparse_tensor.Format() != SparseFormat::kBlockSparse) {
              ORT_THROW("This sparse tensor does not contain BlockSparse format");
            }
            return std::make_unique<PySparseBlockSparseView>(sparse_tensor.AsBlockSparse(), py::cast(*py_tensor));
          })

  /// This will copy SparseTensor into a new instance on a specified CUDA device or throw:
  /// - if this sparse tensor contains strings
  /// - if this sparse tensor is already on GPU
  /// - if CUDA is not present in this build
  /// - if the specified device is not valid
#ifdef USE_CUDA
      .def("to_cuda", [](const PySparseTensor* py_tensor, const OrtDevice& ort_device) -> std::unique_ptr<PySparseTensor> {
        const SparseTensor& sparse_tensor = py_tensor->Instance();
        if (sparse_tensor.IsDataTypeString()) {
          ORT_THROW("Can not copy string tensor to GPU devices.");
        }
        if (sparse_tensor.Location().device.Type() == OrtDevice::GPU) {
          ORT_THROW("This sparse_tensor is already allocated on cuda. Cross device copy not supported.");
        }
        if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), ort_device.Id())) {
          ORT_THROW("The provided device id doesn't match any available GPUs on the machine: ", ort_device.Id());
        }
        auto cuda_allocator = GetCudaAllocator(ort_device.Id());
        auto gpu_transfer = GetGPUDataTransfer();
        auto dest_tensor = std::make_unique<SparseTensor>(sparse_tensor.DataType(), sparse_tensor.Shape(), std::move(cuda_allocator));
        ORT_THROW_IF_ERROR(sparse_tensor.Copy(*gpu_transfer, *dest_tensor, 0));
        auto result = std::make_unique<PySparseTensor>(std::move(dest_tensor));
        return result;
#else
      .def("to_cuda", [](const PySparseTensor*, const OrtDevice&) {
        ORT_THROW("Cuda is not available in this build");
#endif  // USE_CUDA
      })
      .def("shape", [](const PySparseTensor* py_tensor) -> py::list {
        const SparseTensor& st = py_tensor->Instance();
        const auto& dims = st.Shape().GetDims();
        // We create a copy of dimensions, it is small
        py::list py_dims;
        for (auto d : dims) {
          py_dims.append(d);
        }
        return py_dims;
      })
      .def("device_name", [](const PySparseTensor* py_tensor) -> std::string {
        return std::string(GetDeviceName(py_tensor->Instance().Location().device));
      })
      .def("data_type", [](const PySparseTensor* py_tensor) -> std::string {
        const SparseTensor& tensor = py_tensor->Instance();
        const auto elem_type = tensor.GetElementType();
        const auto* type_proto = DataTypeImpl::SparseTensorTypeFromONNXEnum(elem_type)->GetTypeProto();
        if (type_proto == nullptr) {
          ORT_THROW("Unknown type of SparseTensor: ", tensor.DataType());
        }
        return *ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto);
      })
      // pybind apparently has a bug with returning enums from def_property_readonly or methods
      // returning a method object instead of the enumeration value
      // so we are using def_property and throw on a potential modificaiton
      .def_property(
          "format", [](const PySparseTensor* py_tensor) -> OrtSparseFormat {
        const SparseTensor& tensor = py_tensor->Instance();
        auto retval = OrtSparseFormat::ORT_SPARSE_UNDEFINED;
        switch (tensor.Format()) {
          case SparseFormat::kUndefined:
            break;
          case SparseFormat::kCoo:
            retval = OrtSparseFormat::ORT_SPARSE_COO;
            break;
          case SparseFormat::kCsrc:
            retval = OrtSparseFormat::ORT_SPARSE_CSRC;
            break;
          case SparseFormat::kBlockSparse:
            retval = OrtSparseFormat::ORT_SPARSE_BLOCK_SPARSE;
            break;
          default:
            throw std::runtime_error("Can't switch on FormatFlags()");
        }
        return retval; }, [](PySparseTensor*, OrtSparseFormat) -> void { throw std::runtime_error("This is a readonly property"); });
}

}  // namespace python
}  // namespace onnxruntime