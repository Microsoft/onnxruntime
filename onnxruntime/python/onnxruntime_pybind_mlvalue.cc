// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/graph/graph.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/data_types.h"
#include "core/framework/onnxruntime_typeinfo.h"

#include "core/framework/data_transfer_utils.h"
#include "core/framework/data_types_internal.h"
#include "core/providers/get_execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime::logging;

const char* PYTHON_ORTVALUE_OBJECT_NAME = "OrtValue";
const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR = "_ortvalue";

static bool PyObjectCheck_NumpyArray(PyObject* o) {
  return PyObject_HasAttrString(o, "__array_finalize__");
}

bool IsNumericNumpyType(int npy_type) {
  return npy_type < NPY_OBJECT || npy_type == NPY_HALF;
}

bool IsNumericNumpyArray(py::object& py_object) {
  if (PyObjectCheck_NumpyArray(py_object.ptr())) {
    int npy_type = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(py_object.ptr()));
    return IsNumericNumpyType(npy_type);
  }

  return false;
}

static TensorShape GetArrayShape(PyArrayObject* pyObject) {
  int ndim = PyArray_NDIM(pyObject);
  const npy_intp* npy_dims = PyArray_DIMS(pyObject);
  std::vector<int64_t> dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    dims[i] = npy_dims[i];
  }
  TensorShape shape(std::move(dims));
  return shape;
}

void CpuToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  memcpy(dst, src, num_bytes);
}

#ifdef USE_CUDA
void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_CUDA().cudaMemcpy_HostToDevice(dst, src, num_bytes);
}

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_CUDA().cudaMemcpy_DeviceToHost(dst, src, num_bytes);
}

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCudaToHostMemCpyFunction() {
  static std::unordered_map<OrtDevice::DeviceType, MemCpyFunc> map{
      {OrtDevice::GPU, CudaToCpuMemCpy}};

  return &map;
}

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id) {
  int num_devices = GetProviderInfo_CUDA().cudaGetDeviceCount();

  if (0 == num_devices) {
    LOGS(logger, WARNING) << "your system does not have a CUDA capable device.";
    return false;
  }

  if (id < 0 || id >= num_devices) {
    LOGS(logger, WARNING) << "cuda_device=" << id << " is invalid, must choose device ID between 0 and " << num_devices - 1;
    return false;
  }

  return true;
}

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id) {
  // Current approach is not thread-safe, but there are some bigger infra pieces to put together in order to make
  // multi-threaded CUDA allocation work we need to maintain a per-thread CUDA allocator

  static auto* id_to_allocator_map = new std::unordered_map<OrtDevice::DeviceId, AllocatorPtr>();

  if (id_to_allocator_map->find(id) == id_to_allocator_map->end()) {
    // TODO: Expose knobs so that users can set fields associated with OrtArenaCfg so that we can pass it to the following method
    id_to_allocator_map->insert({id, GetProviderInfo_CUDA().CreateCudaAllocator(id, gpu_mem_limit, arena_extend_strategy, external_allocator_info, nullptr)});
  }

  return (*id_to_allocator_map)[id];
}

#endif

#ifdef USE_ROCM

bool IsRocmDeviceIdValid(const onnxruntime::logging::Logger& logger, int id) {
  int num_devices = 0;
  HIP_CALL_THROW(hipGetDeviceCount(&num_devices));

  if (0 == num_devices) {
    LOGS(logger, WARNING) << "your system does not have a ROCM capable device.";
    return false;
  }

  if (id < 0 || id >= num_devices) {
    LOGS(logger, WARNING) << "rocm_device=" << id << " is invalid, must choose device ID between 0 and " << num_devices - 1;
    return false;
  }

  return true;
}

AllocatorPtr GetRocmAllocator(OrtDevice::DeviceId id) {
  // Current approach is not thread-safe, but there are some bigger infra pieces to put together in order to make
  // multi-threaded ROCM allocation work we need to maintain a per-thread ROCM allocator
  static std::unordered_map<OrtDevice::DeviceId, AllocatorPtr> id_to_allocator_map;

  if (id_to_allocator_map.find(id) == id_to_allocator_map.end()) {
    id_to_allocator_map.insert({id, ROCMExecutionProvider::CreateRocmAllocator(id, gpu_mem_limit, arena_extend_strategy, external_allocator_info)});
  }

  return id_to_allocator_map[id];
}

void CpuToRocmMemCpy(void* dst, const void* src, size_t num_bytes) {
  HIP_CALL_THROW(hipMemcpy(dst, src, num_bytes, hipMemcpyHostToDevice));
}

void RocmToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  HIP_CALL_THROW(hipMemcpy(dst, src, num_bytes, hipMemcpyDeviceToHost));
}

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetRocmToHostMemCpyFunction() {
  static std::unordered_map<OrtDevice::DeviceType, MemCpyFunc> map{
      {OrtDevice::GPU, RocmToCpuMemCpy}};

  return &map;
}

#endif

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type) {
  static std::map<MLDataType, int> type_map{
      {DataTypeImpl::GetType<bool>(), NPY_BOOL},
      {DataTypeImpl::GetType<float>(), NPY_FLOAT},
      {DataTypeImpl::GetType<MLFloat16>(), NPY_FLOAT16},
      {DataTypeImpl::GetType<double>(), NPY_DOUBLE},
      {DataTypeImpl::GetType<int8_t>(), NPY_INT8},
      {DataTypeImpl::GetType<uint8_t>(), NPY_UINT8},
      {DataTypeImpl::GetType<int16_t>(), NPY_INT16},
      {DataTypeImpl::GetType<uint16_t>(), NPY_UINT16},
      {DataTypeImpl::GetType<int32_t>(), NPY_INT},
      {DataTypeImpl::GetType<uint32_t>(), NPY_UINT},
      {DataTypeImpl::GetType<int64_t>(), NPY_LONGLONG},
      {DataTypeImpl::GetType<uint64_t>(), NPY_ULONGLONG},
      {DataTypeImpl::GetType<std::string>(), NPY_OBJECT},
  };

  const auto it = type_map.find(tensor_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for Tensor Type.");
  } else {
    return it->second;
  }
}

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type) {
  static std::map<int, MLDataType> type_map{
      {NPY_BOOL, DataTypeImpl::GetType<bool>()},
      {NPY_FLOAT, DataTypeImpl::GetType<float>()},
      {NPY_FLOAT16, DataTypeImpl::GetType<MLFloat16>()},
      {NPY_DOUBLE, DataTypeImpl::GetType<double>()},
      {NPY_INT8, DataTypeImpl::GetType<int8_t>()},
      {NPY_UINT8, DataTypeImpl::GetType<uint8_t>()},
      {NPY_INT16, DataTypeImpl::GetType<int16_t>()},
      {NPY_UINT16, DataTypeImpl::GetType<uint16_t>()},
      {NPY_INT, DataTypeImpl::GetType<int32_t>()},
      {NPY_UINT, DataTypeImpl::GetType<uint32_t>()},
      {NPY_LONGLONG, DataTypeImpl::GetType<int64_t>()},
      {NPY_ULONGLONG, DataTypeImpl::GetType<uint64_t>()},
      {NPY_OBJECT, DataTypeImpl::GetType<std::string>()},
  };
  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for Tensor Type.");
  } else {
    return it->second;
  }
}

const DataTypeImpl* NumpyToOnnxRuntimeTensorType(int numpy_type) {
  static std::map<int, MLDataType> type_map{
      {NPY_BOOL, DataTypeImpl::GetType<bool>()},
      {NPY_FLOAT, DataTypeImpl::GetType<float>()},
      // Special, not a C type expands to enum value of 16
      {NPY_FLOAT16, DataTypeImpl::GetType<MLFloat16>()},
      {NPY_DOUBLE, DataTypeImpl::GetType<double>()},
      // We don't want to use size specific types such
      // as NPY_INT32 bc they are not enums but hash defines
      // which may map into other enums and may conflict with other entries here
      // also NPY docs define these sizes as platform specific, thus we
      // choose to do some rudimentary checks for proper mapping on C++ size
      {NPY_BYTE, DataTypeImpl::GetType<int8_t>()},
      {NPY_UBYTE, DataTypeImpl::GetType<uint8_t>()},
      {NPY_SHORT, sizeof(short) == sizeof(int16_t) ? DataTypeImpl::GetType<int16_t>()
                                                   : DataTypeImpl::GetType<int32_t>()},
      {NPY_USHORT, sizeof(unsigned short) == sizeof(uint16_t) ? DataTypeImpl::GetType<uint16_t>()
                                                              : DataTypeImpl::GetType<uint32_t>()},
      {NPY_INT,
       sizeof(int) == sizeof(int32_t) ? DataTypeImpl::GetType<int32_t>()
                                      : DataTypeImpl::GetType<int64_t>()},
      {NPY_UINT, sizeof(int) == sizeof(int32_t) ? DataTypeImpl::GetType<uint32_t>()
                                                : DataTypeImpl::GetType<uint64_t>()},

      {NPY_LONG,
       sizeof(long) == sizeof(int32_t) ? DataTypeImpl::GetType<int32_t>()
                                       : DataTypeImpl::GetType<int64_t>()},
      {NPY_ULONG,
       sizeof(unsigned long) == sizeof(uint32_t) ? DataTypeImpl::GetType<uint32_t>()
                                                 : DataTypeImpl::GetType<uint64_t>()},
      {NPY_LONGLONG, DataTypeImpl::GetType<int64_t>()},
      {NPY_ULONGLONG, DataTypeImpl::GetType<uint64_t>()},
      {NPY_UNICODE, DataTypeImpl::GetType<std::string>()},
      {NPY_STRING, DataTypeImpl::GetType<std::string>()},
      {NPY_OBJECT, DataTypeImpl::GetType<std::string>()},
      {NPY_VOID, DataTypeImpl::GetType<std::string>()}};

  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error("Numpy_type " + std::to_string(numpy_type) +
                             " can't be converted to MLDataType.");
  } else {
    return it->second;
  }
}

// This is a one time use, ad-hoc allocator that allows Tensors to take ownership of
// python array objects and use the underlying memory directly and
// properly deallocated them when they are done.
//
// This addresses the case when our interfaces receive python lists on the input.
// We have to convert them into new Numpy arrays which 1) needs to be properly deallocated
// because they are not owned by the calling python code (we create it inside pybind code)
// 2) we still want to avoid yet another data copy and use it directly if possible
// 3) string data types still need to be copied
//
// This is a stateful allocator. It will always return the same pre-allocated
// buffer pointer and will own references to underlying objects.
class OrtPybindSingleUseAllocator : public IAllocator {
 public:
  // This constructor is used when we create numpy array from python list
  OrtPybindSingleUseAllocator(PyArrayObject* pyObject, const std::string& value_name, const OrtMemoryInfo& mem_info)
      : IAllocator(mem_info),
        pyObject_(pyObject, DecRefFn<PyArrayObject>()),
        pyObjectContiguous_(PyArray_GETCONTIGUOUS(pyObject), DecRefFn<PyArrayObject>()) {
    ORT_ENFORCE(pyObjectContiguous_ != nullptr, "The object must be a contiguous array for input :", value_name);
  }

  // Constructor to use when a contiguous array had to be copied. Instead of creating yet another copy
  // we are still able to use it directly for primitive types
  OrtPybindSingleUseAllocator(UniqueDecRefPtr<PyArrayObject>&& pyContiguous, const std::string& value_name,
                              const OrtMemoryInfo& mem_info)
      : IAllocator(mem_info),
        pyObject_(nullptr, DecRefFn<PyArrayObject>()),
        pyObjectContiguous_(std::move(pyContiguous)) {
    ORT_ENFORCE(pyObjectContiguous_ != nullptr, "Expecting a valid contiguous array:", value_name);
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtPybindSingleUseAllocator);

  // Always return pre-allocated buffer
  // which actually contains the array data
  void* Alloc(size_t) override {
    return static_cast<void*>(PyArray_DATA(pyObjectContiguous_.get()));
  }

  void Free(void*) override {
    // Free when requested, do not wait for
    // destruction of the allocator which may
    // be non-deterministic. However, we do not anticipate
    // true shared ownership of the allocator object except
    // at the creation stack.
    pyObjectContiguous_.reset();
    pyObject_.reset();
  }

  PyArrayObject* GetContiguous() const {
    return pyObjectContiguous_.get();
  }

 private:
  UniqueDecRefPtr<PyArrayObject> pyObject_;
  UniqueDecRefPtr<PyArrayObject> pyObjectContiguous_;
};

using OrtPybindSingleUseAllocatorPtr = std::shared_ptr<OrtPybindSingleUseAllocator>;

// Expects p_tensor properly created
// Does not manage darray life-cycle

static void CopyDataToTensor(PyArrayObject* darray, int npy_type, std::unique_ptr<Tensor>& p_tensor,
                             MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy) {
  const auto total_items = p_tensor->Shape().Size();
  if (npy_type == NPY_UNICODE) {
    // Copy string data which needs to be done after Tensor is allocated.
    // Strings are Python strings or numpy.unicode string.
    std::string* dst = p_tensor->MutableData<std::string>();
    const auto item_size = PyArray_ITEMSIZE(darray);
    const auto num_chars = item_size / PyUnicode_4BYTE_KIND;
    const char* src = reinterpret_cast<const char*>(PyArray_DATA(darray));
    for (int i = 0; i < total_items; i++, src += item_size) {
      // Python unicode strings are assumed to be USC-4. Strings are stored as UTF-8.
      PyObject* pStr = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, src, num_chars);
      UniqueDecRefPtr<PyObject> strGuard(pStr, DecRefFn<PyObject>());
      const char* str = PyUnicode_AsUTF8(pStr);
      if (str == NULL) {
        dst[i].clear();
      } else {
        // Size is equal to the longest string size, numpy stores
        // strings in a single array.
        dst[i] = str;
      }
    }
  } else if (npy_type == NPY_STRING || npy_type == NPY_VOID) {
    // Copy string data which needs to be done after Tensor is allocated.
    // Strings are given as bytes (encoded strings).
    // NPY_VOID does not trim final 0.
    // NPY_STRING assumes bytes string ends with a final 0.
    std::string* dst = p_tensor->MutableData<std::string>();
    const auto item_size = PyArray_ITEMSIZE(darray);
    const char* src = reinterpret_cast<const char*>(PyArray_DATA(darray));
    for (int i = 0; i < total_items; i++, src += item_size) {
      if (npy_type == NPY_STRING) {
        dst[i] = src;
      } else {
        dst[i].assign(src, item_size);
      }
    }
  } else if (npy_type == NPY_OBJECT) {
    // Converts object into string.
    std::string* dst = p_tensor->MutableData<std::string>();
    const auto item_size = PyArray_ITEMSIZE(darray);
    const char* src = reinterpret_cast<const char*>(PyArray_DATA(darray));
    for (int i = 0; i < total_items; ++i, src += item_size) {
      // Python unicode strings are assumed to be USC-4. Strings are stored as UTF-8.
      PyObject* item = PyArray_GETITEM(darray, src);
      PyObject* pStr = PyObject_Str(item);
      UniqueDecRefPtr<PyObject> strGuard(pStr, DecRefFn<PyObject>());
      dst[i] = py::reinterpret_borrow<py::str>(pStr);
    }
  } else {
    void* buffer = p_tensor->MutableDataRaw();
    size_t len;
    if (!IAllocator::CalcMemSizeForArray(p_tensor->DataType()->Size(), p_tensor->Shape().Size(), &len)) {
      throw std::runtime_error("length overflow");
    }
    mem_cpy_to_device(buffer, PyArray_DATA(darray), len);
  }
}

// Setting `use_numpy_data_memory` to `true` will ensure that the underlying numpy array buffer is directly used
// as the backing data buffer for the ORT Tensor where applicable (for numeric tensors)
// The numpy object owns the memory and needs to be alive until the corresponding OrtValue is in scope
static std::unique_ptr<Tensor> CreateTensor(const AllocatorPtr& alloc, const std::string& name_input,
                                            PyArrayObject* pyObject, bool use_numpy_data_memory = true,
                                            MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy) {
  PyArrayObject* darray = PyArray_GETCONTIGUOUS(pyObject);
  ORT_ENFORCE(darray != nullptr, "The object must be a contiguous array for input '", name_input, "'.");

  UniqueDecRefPtr<PyArrayObject> darray_guard(darray, DecRefFn<PyArrayObject>());
  std::unique_ptr<Tensor> p_tensor;

  const int npy_type = PyArray_TYPE(darray);
  TensorShape shape = GetArrayShape(darray);
  auto element_type = NumpyToOnnxRuntimeTensorType(npy_type);
  if (IsNumericNumpyType(npy_type) && use_numpy_data_memory) {
    if (pyObject == darray) {
      // Use the memory of numpy array directly. The ownership belongs to the calling
      // python code. In this case, the incoming pyObject must itself be contiguous (pyObject == darray).
      // darray reference will be decremented but the original array is still alive
      p_tensor = std::make_unique<Tensor>(element_type, shape, PyArray_DATA(darray), alloc->Info());
    } else {
      // This is the case when a contiguous array is a copy. We still can use it directly with OrtPybindSingleUseAllocator
      // which takes ownership of the array.
      auto pybind_alloc = std::make_shared<OrtPybindSingleUseAllocator>(std::move(darray_guard), name_input, alloc->Info());
      p_tensor = std::make_unique<Tensor>(element_type, shape, std::move(pybind_alloc));
    }
  } else {
    p_tensor = std::make_unique<Tensor>(element_type, shape, alloc);
    CopyDataToTensor(darray, npy_type, p_tensor, mem_cpy_to_device);
  }

  return p_tensor;
}

static bool CheckIfInputIsSequenceType(const std::string& name_input,
                                       const InputDefList* input_def_list,
                                       /*out*/ onnx::TypeProto& type_proto) {
  // get sequence type from the model
  const auto& def_list = *input_def_list;
  auto ret_it = std::find_if(std::begin(def_list), std::end(def_list),
                             [&name_input](const NodeArg* node_arg) { return name_input == node_arg->Name(); });
  if (ret_it == std::end(def_list)) {
    throw std::runtime_error("Failed to find input with name: " + name_input + " in the model input def list");
  }
  const auto* temp = (*ret_it)->TypeAsProto();
  if (!temp) {
    throw std::runtime_error("Corresponding type_proto is null");
  } else {
    type_proto = *temp;
  }

  return type_proto.has_sequence_type();
}

static void CreateSequenceOfTensors(AllocatorPtr alloc, const std::string& name_input,
                                    const InputDefList* input_def_list, PyObject* pylist_obj, OrtValue* p_mlvalue) {
  onnx::TypeProto type_proto;
  if (!CheckIfInputIsSequenceType(name_input, input_def_list, type_proto)) {
    throw std::runtime_error("Input is not of sequence type");
  }

  // populate the seq
  std::vector<Tensor> tensors;
  auto list_size = PyList_Size(pylist_obj);
  if (list_size > 0) {
    tensors.resize(list_size);
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      auto* py_obj = PyList_GetItem(pylist_obj, i);
      if (!PyObjectCheck_NumpyArray(py_obj)) {
        throw std::runtime_error("CreateSequenceOfTensors: Input is not a tensor");
      }
      auto p_tensor = CreateTensor(alloc, name_input, reinterpret_cast<PyArrayObject*>(py_obj));
      tensors[i] = std::move(*p_tensor);
    }
  }

  // set the seq type
  MLDataType seq_dtype = OrtTypeInfo::ElementTypeFromProto(
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto.sequence_type().elem_type().tensor_type().elem_type()));
  auto p_seq_tensors = std::make_unique<TensorSeq>(seq_dtype);
  p_seq_tensors->SetElements(std::move(tensors));
  auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
  p_mlvalue->Init(p_seq_tensors.release(),
                  ml_tensor_sequence,
                  ml_tensor_sequence->GetDeleteFunc());
}

// Setting `use_numpy_data_memory` to `true` will ensure that the underlying numpy array buffer is directly used
// as the backing data buffer for the ORT Tensor where applicable (for numeric tensors)
// The numpy object owns the memory and needs to be alive until the corresponding OrtValue is in scope
static void CreateTensorMLValue(const AllocatorPtr& alloc, const std::string& name_input, PyArrayObject* pyObject,
                                OrtValue* p_mlvalue, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy) {
  auto p_tensor = CreateTensor(alloc, name_input, pyObject, use_numpy_data_memory, mem_cpy_to_device);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  p_mlvalue->Init(p_tensor.release(),
                  ml_tensor,
                  ml_tensor->GetDeleteFunc());
}

// This function will create a Tensor that owns the python array memory. This is done to properly
// release python arrays allocated within the pybind code.
static void CreateTensorMLValueOwned(const OrtPybindSingleUseAllocatorPtr& pybind_alloc, const AllocatorPtr& alloc, OrtValue* p_mlvalue) {
  auto npy_type = PyArray_TYPE(pybind_alloc->GetContiguous());
  TensorShape shape = GetArrayShape(pybind_alloc->GetContiguous());
  auto element_type = NumpyToOnnxRuntimeTensorType(npy_type);

  std::unique_ptr<Tensor> p_tensor;

  if (npy_type != NPY_UNICODE && npy_type != NPY_STRING &&
      npy_type != NPY_VOID && npy_type != NPY_OBJECT) {
    // We are able to reuse the memory of the contiguous python buffer and avoid
    // extra copy using OrtPybindAllocator which will take care of the memory
    p_tensor = std::make_unique<Tensor>(element_type, shape, pybind_alloc);
  } else {
    // We still need to copy elements properly from the contiguous buffer
    p_tensor = std::make_unique<Tensor>(element_type, shape, alloc);
    CopyDataToTensor(pybind_alloc->GetContiguous(), npy_type, p_tensor);
  }

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  p_mlvalue->Init(p_tensor.release(),
                  ml_tensor,
                  ml_tensor->GetDeleteFunc());
}

std::string _get_type_name(int64_t&) {
  return std::string("int64_t");
}

std::string _get_type_name(float&) {
  return std::string("float");
}

std::string _get_type_name(std::string&) {
  return std::string("string");
}

#if !defined(DISABLE_ML_OPS)
template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
static void CreateMapMLValue_LoopIntoMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                         PyObject* item, std::map<KeyType, ValueType>& current,
                                         KeyGetterType keyGetter, ValueGetterType valueGetter) {
  KeyType ckey;
  ValueType cvalue;
  do {
    if (!keyGetter(key, ckey)) {
      PyObject* pType = PyObject_Type(key);
      auto pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);
      Py_XDECREF(item);
      throw std::runtime_error(std::string("Unexpected key type  ") + sType +
                               std::string(", it cannot be linked to C type ") +
                               _get_type_name(ckey) + std::string(" for input '") +
                               name_input + std::string("'."));
    }

    if (!valueGetter(value, cvalue)) {
      PyObject* pType = PyObject_Type(value);
      auto pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);
      Py_XDECREF(item);
      throw std::runtime_error(std::string("Unexpected value type  ") + sType +
                               std::string(", it cannot be linked to C type ") +
                               _get_type_name(ckey) + std::string(" for input '") +
                               name_input + std::string("'."));
    }
    current[ckey] = cvalue;
  } while (PyDict_Next(item, &pos, &key, &value));
}

template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
static void CreateMapMLValue_Map(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                 PyObject* item, AllocatorPtr /*alloc*/, OrtValue* p_mlvalue, KeyGetterType keyGetter,
                                 ValueGetterType valueGetter) {
  std::unique_ptr<std::map<KeyType, ValueType>> dst;
  dst = std::make_unique<std::map<KeyType, ValueType>>();
  CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, *dst, keyGetter, valueGetter);
  p_mlvalue->Init(dst.release(), DataTypeImpl::GetType<std::map<KeyType, ValueType>>(),
                  DataTypeImpl::GetType<std::map<KeyType, ValueType>>()->GetDeleteFunc());
}

template <typename KeyType, typename ValueType, typename KeyGetterType, typename ValueGetterType>
void CreateMapMLValue_VectorMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                PyObject* iterator, PyObject* item, AllocatorPtr /*alloc*/, OrtValue* p_mlvalue,
                                KeyGetterType keyGetter, ValueGetterType valueGetter) {
  std::unique_ptr<std::vector<std::map<KeyType, ValueType>>> dstVector;
  dstVector = std::make_unique<std::vector<std::map<KeyType, ValueType>>>();
  int index = 0;
  do {
    dstVector->push_back(std::map<KeyType, ValueType>());
    CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, (*dstVector)[index], keyGetter, valueGetter);
    Py_DECREF(item);
    ++index;
    item = iterator == NULL ? NULL : PyIter_Next(iterator);
  } while (item != NULL);
  p_mlvalue->Init(dstVector.release(), DataTypeImpl::GetType<std::vector<std::map<KeyType, ValueType>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<KeyType, ValueType>>>()->GetDeleteFunc());
}

static void CreateMapMLValue_AgnosticMap(Py_ssize_t& pos, PyObject*& key, const std::string& name_input, PyObject*& value,
                                         PyObject* iterator, PyObject* item, AllocatorPtr alloc, OrtValue* p_mlvalue) {
  // If iterator is NULL, it returns a single Map,
  // if is not NULL, it returns a VectorMap.
  auto int64Getter = [](PyObject* obj, int64_t& value) -> bool {
    value = PyLong_AsLong(obj);
    return !PyErr_Occurred();
  };

  auto floatGetter = [](PyObject* obj, float& value) -> bool {
    if (PyFloat_Check(obj)) {
      value = (float)PyFloat_AS_DOUBLE(obj);
      return true;
    } else if (PyNumber_Check(obj)) {
      value = (float)PyFloat_AsDouble(obj);
      return true;
    } else {
      return false;
    }
  };

  auto stringGetter = [](PyObject* obj, std::string& value) -> bool {
    PyObject* pStr = PyObject_Str(obj);
    if (pStr == NULL) {
      return false;
    }
    value = py::reinterpret_borrow<py::str>(pStr);
    Py_DECREF(pStr);
    return true;
  };

  if (iterator == NULL) {
    if (PyLong_Check(key)) {
      // Regular Python.
      CreateMapMLValue_Map<int64_t, float>(pos, key, name_input, value, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyNumber_Check(key)) {
      // For numpy type.
      CreateMapMLValue_Map<int64_t, float>(pos, key, name_input, value, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyUnicode_Check(key)) {
      CreateMapMLValue_Map<std::string, float>(pos, key, name_input, value, item, alloc, p_mlvalue, stringGetter, floatGetter);
    } else {
      PyObject* pType = PyObject_Type(key);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Key type must be int or string (not ") + sType +
                               std::string(") for input '") + name_input + std::string("'."));
    }
  } else {
    if (PyLong_Check(key)) {
      CreateMapMLValue_VectorMap<int64_t, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyNumber_Check(key)) {
      // For numpy type.
      CreateMapMLValue_VectorMap<int64_t, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, int64Getter, floatGetter);
    } else if (PyUnicode_Check(key)) {
      CreateMapMLValue_VectorMap<std::string, float>(pos, key, name_input, value, iterator, item, alloc, p_mlvalue, stringGetter, floatGetter);
    } else {
      PyObject* pType = PyObject_Type(value);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Key type must be int or string (not ") + sType +
                               std::string(") for input '") + name_input + std::string("'."));
    }
  }
}

static void CreateMapMLValue_AgnosticVectorMap(PyObject* iterator, PyObject* item, AllocatorPtr alloc,
                                               const std::string& name_input, OrtValue* p_mlvalue) {
  // CreateMapMLValue is called by CreateGenericTerableMLValue
  // or CreateGenericMLValue which ensures
  // item is a dictionary, no need to check type again.
  // This functions starts to iterate on the first
  // element of the dictionary and calls CreateMapMLValue_AgnosticMap
  // which determines the container type. This type
  // is based on the first pair of the dictionary
  // and all the function assumes the key and value type remain the same
  // for all pairs in the dictionary.

  // If iterator is NULL, it returns a single Map,
  // if is not NULL, it returns a VectorMap.

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  if (PyDict_Next(item, &pos, &key, &value)) {
    CreateMapMLValue_AgnosticMap(pos, key, name_input, value, iterator, item, alloc, p_mlvalue);
  } else {
    throw std::runtime_error("Size of dictionary is empty, unable to run the prediction.");
  }
}
#endif

static void CreateGenericIterableMLValue(PyObject* iterator, AllocatorPtr alloc, const std::string& name_input,
                                         OrtValue* p_mlvalue) {
  PyObject* item;
  OrtValue ml_value;
  item = PyIter_Next(iterator);
  if (item == NULL) {
    throw std::runtime_error("Input '" + name_input + "' must not be empty.");
  }
  if (PyObjectCheck_NumpyArray(item)) {
    PyObject* pType = PyObject_Type(item);
    PyObject* pStr = PyObject_Str(pType);
    py::str spyType = py::reinterpret_borrow<py::str>(pStr);
    std::string sType = spyType;
    Py_XDECREF(pType);
    Py_XDECREF(pStr);
    throw std::runtime_error("Iterable of " + sType + " should be given as array for input '" +
                             name_input + std::string("'."));
  } else {
    // We expect a dictionary.
    if (!PyDict_Check(item)) {
      throw std::runtime_error("Input must be a list of dictionaries or a single numpy array for input '" +
                               name_input + std::string("'."));
    }
#if !defined(DISABLE_ML_OPS)
    CreateMapMLValue_AgnosticVectorMap(iterator, item, alloc, name_input, p_mlvalue);
#else
    ORT_UNUSED_PARAMETER(alloc);
    ORT_UNUSED_PARAMETER(p_mlvalue);
    throw std::runtime_error("Map type is not supported in this build.");
#endif
  }
}

// Setting `use_numpy_data_memory` to `true` will ensure that the underlying numpy array buffer is directly used
// as the backing data buffer for the ORT Tensor where applicable (for numeric tensors)
// The numpy object owns the memory and needs to be alive until the corresponding OrtValue is in scope
void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc, const std::string& name_input,
                          py::object& value, OrtValue* p_mlvalue, bool accept_only_numpy_array,
                          bool use_numpy_data_memory, MemCpyFunc mem_cpy_to_device) {
  onnx::TypeProto type_proto;
  if (PyObjectCheck_NumpyArray(value.ptr())) {
    // The most frequent case: input comes as an array.
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(value.ptr());
    CreateTensorMLValue(alloc, name_input, arr, p_mlvalue, use_numpy_data_memory, mem_cpy_to_device);
  } else if (!accept_only_numpy_array &&
             PyList_Check(value.ptr()) &&
             !CheckIfInputIsSequenceType(name_input, input_def_list, type_proto)) {
    // This is not a sequence tensor. This is just a regular tensor fed through as a list.
    ORT_ENFORCE(type_proto.tensor_type().has_elem_type(), "The graph is missing type information needed to construct the ORT tensor");

    MLDataType dtype = OrtTypeInfo::ElementTypeFromProto(
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto.tensor_type().elem_type()));

    int numpy_dtype = OnnxRuntimeTensorToNumpyType(dtype);

    // This creates a new object with its own reference count
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FromAny(value.ptr(), PyArray_DescrFromType(numpy_dtype), 0, 0, 0, nullptr));

    if (!arr) {
      throw std::runtime_error("Could not create tensor from given input list");
    }

    // The allocator will own the array memory and will decrement the reference on Free()
    // or when destroyed
    auto pybind_alloc = std::make_shared<OrtPybindSingleUseAllocator>(arr, name_input, alloc->Info());
    CreateTensorMLValueOwned(pybind_alloc, alloc, p_mlvalue);
  } else if (!accept_only_numpy_array && PyList_Check(value.ptr())) {
    auto* seq_tensors = reinterpret_cast<PyObject*>(value.ptr());
    CreateSequenceOfTensors(alloc, name_input, input_def_list, seq_tensors, p_mlvalue);
  } else if (!accept_only_numpy_array && PyDict_Check(value.ptr())) {
#if !defined(DISABLE_ML_OPS)
    CreateMapMLValue_AgnosticVectorMap((PyObject*)NULL, value.ptr(), alloc, name_input, p_mlvalue);
#else
    ORT_UNUSED_PARAMETER(p_mlvalue);
    throw std::runtime_error("Map type is not supported in this build.");
#endif

  } else if (!accept_only_numpy_array && strcmp(Py_TYPE(value.ptr())->tp_name, PYTHON_ORTVALUE_OBJECT_NAME) == 0) {
    // This is an OrtValue coming in directly from Python, so assign the underlying native OrtValue handle
    // to the OrtValue object that we are going to use for Run().
    // This should just increase the ref counts of the underlying shared_ptrs in the native OrtValue
    // and the ref count will be decreased when the OrtValue used for Run() is destroyed upon exit.
    *p_mlvalue = *value.attr(PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR).cast<OrtValue*>();
  } else if (!accept_only_numpy_array) {
    auto iterator = PyObject_GetIter(value.ptr());
    if (iterator == NULL) {
      // The pype cannot be handled.
      PyObject* pType = PyObject_Type(value.ptr());
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Unable to handle object of type ") + sType);
    }
    // We assume the object is iterable.
    // iterator should not be NULL due to previous test.
    try {
      CreateGenericIterableMLValue(iterator, alloc, name_input, p_mlvalue);
    } catch (const std::runtime_error&) {
      Py_DECREF(iterator);
      throw;
    }
    Py_DECREF(iterator);
  } else {
    throw std::runtime_error("Unable to create OrtValue from the given python object");
  }
}

}  // namespace python
}  // namespace onnxruntime
