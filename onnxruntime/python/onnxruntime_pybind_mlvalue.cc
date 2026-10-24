// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"
#include "pybind11/numpy.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"

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

#ifdef USE_DML
using Microsoft::WRL::ComPtr;

#include <wil/wrl.h>
#include "core/providers/dml/DmlExecutionProvider/src/External/D3DX12/d3dx12.h"
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#include "core/providers/dml/DmlExecutionProvider/src/DescriptorPool.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlCommittedResourceAllocator.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/DmlExecutionProvider/src/BucketizedBufferAllocator.h"
#include "core/providers/dml/DmlExecutionProvider/src/PooledUploadHeap.h"
#include "core/providers/dml/DmlExecutionProvider/src/ReadbackHeap.h"
#include "core/providers/dml/DmlExecutionProvider/src/AllocationInfo.h"
#endif
namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime::logging;

const char* PYTHON_ORTVALUE_OBJECT_NAME = "OrtValue";
const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR = "_ortvalue";

static bool PyObjectCheck_NumpyArray(PyObject* o) {
  return (PyObject_HasAttrString(o, "__array_finalize__") != 0);
}

bool IsNumpyArray(py::object& obj) {
  return PyObjectCheck_NumpyArray(obj.ptr());
}

int GetNumpyArrayType(const py::object& obj) {
  return PyArray_TYPE(reinterpret_cast<PyArrayObject*>(obj.ptr()));
}

bool IsNumericNumpyArray(const py::object& py_object) {
  if (PyObjectCheck_NumpyArray(py_object.ptr())) {
    int npy_type = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(py_object.ptr()));
    return IsNumericNumpyType(npy_type);
  }

  return false;
}

namespace {
template <typename... T>
std::vector<py::dtype> MakeTypes() {
  std::vector<py::dtype> result = {py::dtype::of<T>()...};
  return result;
}
}  // namespace

bool IsNumericDType(const py::dtype& dtype) {
  static const std::vector<py::dtype> numeric =
      MakeTypes<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>();
  return std::any_of(numeric.cbegin(), numeric.cend(), [&dtype](const py::dtype& dt) {
    return dtype.is(dt);
  });
}

static TensorShape GetArrayShape(PyArrayObject* pyObject) {
  const int ndim = PyArray_NDIM(pyObject);
  const npy_intp* npy_dims = PyArray_DIMS(pyObject);
  auto span = gsl::make_span(npy_dims, ndim);
  TensorShapeVector shape_vec(span.begin(), span.end());
  TensorShape shape(shape_vec);
  return shape;
}

TensorShape GetShape(const py::array& arr) {
  auto span = gsl::make_span(arr.shape(), arr.ndim());
  TensorShapeVector shape_vec(span.begin(), span.end());
  TensorShape shape(shape_vec);
  return shape;
}

AllocatorPtr GetSharedAllocator(const OrtDevice& device) {
  auto& env = GetOrtEnv()->GetEnvironment();

  OrtMemoryInfo mem_info("ignored", OrtDeviceAllocator, device);
  return env.GetRegisteredSharedAllocator(mem_info);
}

MemCpyFunc CreateDataTransferMemCpy([[maybe_unused]] const OrtDevice& src_device,
                                    [[maybe_unused]] const OrtDevice& dst_device) {
#if defined(ORT_MINIMAL_BUILD)
  // plugin EPs are not supported in a minimal build so there won't be any data transfers registered
  return nullptr;
#else

  auto& env = GetOrtEnv()->GetEnvironment();
  const DataTransferManager& data_transfer_manager = env.GetDataTransferManager();
  const IDataTransfer* data_transfer = data_transfer_manager.GetDataTransfer(src_device, dst_device);
  if (!data_transfer) {
    return nullptr;
  }

  const auto copy_func = [src_device, dst_device, data_transfer](void* dst, const void* src, size_t bytes) {
    OrtMemoryInfo src_memory_info("ignored", OrtDeviceAllocator, src_device);
    OrtMemoryInfo dst_memory_info("ignored", OrtDeviceAllocator, dst_device);

    // real shape doesn't matter as the Tensor instances here are temporary in order to be able to call CopyTensor.
    // we set the shape to `bytes` and the data type to uint8_t to copy the correct number of bytes.
    TensorShape shape = {narrow<int64_t>(bytes)};
    Tensor src_tensor{DataTypeImpl::GetType<uint8_t>(), shape, const_cast<void*>(src), src_memory_info};
    Tensor dst_tensor{DataTypeImpl::GetType<uint8_t>(), shape, dst, dst_memory_info};

    ORT_THROW_IF_ERROR(data_transfer->CopyTensor(src_tensor, dst_tensor));
  };

  return copy_func;
#endif
}

void CpuToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  memcpy(dst, src, num_bytes);
}

OrtMemoryInfo GetMemoryInfoPerDeviceType(const OrtDevice& ort_device) {
  OrtMemoryInfo mem_info;
  if (ort_device.Type() == OrtDevice::CPU) {
    mem_info = GetAllocator()->Info();
  }
#if USE_CUDA
  else if (ort_device.Type() == OrtDevice::GPU) {
    if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), ort_device.Id())) {
      ORT_THROW("The provided device id doesn't match any available GPUs on the machine: ", ort_device.Id());
    }
    mem_info = GetCudaAllocator(ort_device.Id())->Info();
  }
#endif
#if USE_MIGRAPHX
  else if (ort_device.Type() == OrtDevice::GPU) {
    mem_info = GetMIGraphXAllocator(ort_device.Id())->Info();
  }
#endif
  else {
    ORT_THROW("Unsupported OrtDevice type: ", ort_device.Type());
  }
  return mem_info;
}

int32_t GetTensorProtoType(const OrtValue& ort_value) {
  if (ort_value.IsTensor()) {
    return ort_value.Get<Tensor>().GetElementType();
#if !defined(DISABLE_SPARSE_TENSORS)
  } else if (ort_value.IsSparseTensor()) {
    return ort_value.Get<SparseTensor>().GetElementType();
#endif
  } else if (ort_value.IsTensorSequence()) {
    return ort_value.Get<TensorSeq>().DataType()->AsPrimitiveDataType()->GetDataType();
  } else {
    throw std::runtime_error("Tensor proto_type is unavailable for this value.");
  }
}

#ifdef USE_CUDA

void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes) {
  if (TryGetProviderInfo_CUDA() != nullptr) {
    GetProviderInfo_CUDA().cudaMemcpy_HostToDevice(dst, src, num_bytes);
    return;
  }

  ORT_THROW("CUDA provider interface is not available for host-to-device copy.");
}

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  if (TryGetProviderInfo_CUDA() != nullptr) {
    GetProviderInfo_CUDA().cudaMemcpy_DeviceToHost(dst, src, num_bytes);
    return;
  }

  ORT_THROW("CUDA provider interface is not available for device-to-host copy.");
}

const std::unordered_map<OrtDevice, MemCpyFunc>* GetCudaToHostMemCpyFunction(const OrtDevice& device) {
  static std::unordered_map<OrtDevice, MemCpyFunc> map{
      {OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, device.Id()}, CudaToCpuMemCpy},
  };

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

  // We are leaking this map so we do not accidentally destroy CUDA Allocator instance
  // after we unloaded CUDA provider library. Appeasing static analysis warning and using make_unique.
  static auto* id_to_allocator_map = std::make_unique<std::unordered_map<OrtDevice::DeviceId, AllocatorPtr>>().release();

  auto hit = id_to_allocator_map->find(id);
  if (hit == id_to_allocator_map->end()) {
    // TODO: Expose knobs so that users can set fields associated with OrtArenaCfg so that we can pass it to the following method
    auto cuda_allocator = GetProviderInfo_CUDA().CreateCudaAllocator(id, gpu_mem_limit, arena_extend_strategy, external_allocator_info, nullptr);
    hit = id_to_allocator_map->emplace(id, std::move(cuda_allocator)).first;
  }

  return hit->second;
}

std::unique_ptr<IDataTransfer> GetGPUDataTransfer() {
  // Using default stream
  return GetProviderInfo_CUDA().CreateGPUDataTransfer();
}

#endif

#ifdef USE_MIGRAPHX

void CpuToMIGraphXMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_MIGraphX().MIGraphXMemcpy_HostToDevice(dst, src, num_bytes);
}

void MIGraphXToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_MIGraphX().MIGraphXMemcpy_DeviceToHost(dst, src, num_bytes);
}

const std::unordered_map<OrtDevice, MemCpyFunc>* GetMIGraphXToHostMemCpyFunction(const OrtDevice& device) {
  static std::unordered_map<OrtDevice, MemCpyFunc> map{
      {OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD, device.Id()}, MIGraphXToCpuMemCpy},
  };

  return &map;
}

AllocatorPtr GetMIGraphXAllocator(OrtDevice::DeviceId id) {
  // Current approach is not thread-safe, but there are some bigger infra pieces to put together in order to make
  // multi-threaded MIGraphX allocation work we need to maintain a per-thread MIGraphX allocator

  static auto* id_to_allocator_map = new std::unordered_map<OrtDevice::DeviceId, AllocatorPtr>();

  if (id_to_allocator_map->find(id) == id_to_allocator_map->end()) {
    // TODO: Expose knobs so that users can set fields associated with OrtArenaCfg so that we can pass it to the following method
    id_to_allocator_map->insert(
        {id, GetProviderInfo_MIGraphX().CreateMIGraphXAllocator(
                 id, gpu_mem_limit, arena_extend_strategy,
                 migraphx::external::alloc_fn, migraphx::external::free_fn, migraphx::external::empty_cache_fn,
                 nullptr)});
  }

  return (*id_to_allocator_map)[id];
}

#endif

#ifdef USE_DML

constexpr GUID dml_readback_heap_guid = {0x00d32df8, 0xea2d, 0x40bf, {0xa4, 0x47, 0x9c, 0xb4, 0xbc, 0xf1, 0x1d, 0x5e}};
constexpr GUID dml_upload_heap_guid = {0x125235f9, 0xef41, 0x4043, {0xa4, 0x9d, 0xdd, 0xc9, 0x61, 0xe7, 0xdb, 0xee}};

AllocatorPtr GetDmlAllocator(OrtDevice::DeviceId id) {
  // Current approach is not thread-safe, but there are some bigger infra pieces to put together in order to make
  // multi-threaded DML allocation work, including maintaining a per-thread DML allocator.

  // We are leaking this map so we do not accidentally destroy the DML Allocator instance
  // after we unloaded DML provider library. Appeasing static analysis warning and using make_unique.
  static auto* id_to_allocator_map = std::make_unique<std::unordered_map<OrtDevice::DeviceId, AllocatorPtr>>().release();

  auto hit = id_to_allocator_map->find(id);
  if (hit == id_to_allocator_map->end()) {
    constexpr uint32_t device_id = 0;
    auto d3d12_device = onnxruntime::DMLProviderFactoryCreator::CreateD3D12Device(device_id, false);

    ComPtr<Dml::ExecutionContext> context;
    uint32_t execution_context_ptr_size = gsl::narrow_cast<uint32_t>(sizeof(context.GetAddressOf()));

    // First, check if an I/O binding API that was used before this session or another session has already created a queue
    if (FAILED(d3d12_device->GetPrivateData(dml_execution_context_guid, &execution_context_ptr_size, context.GetAddressOf()))) {
      D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
      cmd_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

      ComPtr<ID3D12CommandQueue> cmd_queue;
      ORT_THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

      auto dml_device = onnxruntime::DMLProviderFactoryCreator::CreateDMLDevice(d3d12_device.Get());
      ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_device_guid, dml_device.Get()));

      context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device.Get(), cmd_queue.Get(), true, true);
      ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_execution_context_guid, context.Get()));
    }

    // We leak the readback and upload heap to keep them alive, just like the map
    auto readback_heap = std::make_unique<Dml::ReadbackHeap>(d3d12_device.Get(), context.Get()).release();
    auto upload_heap = std::make_unique<Dml::PooledUploadHeap>(d3d12_device.Get(), context.Get()).release();

    auto dml_allocator = std::make_shared<Dml::BucketizedBufferAllocator>(
        d3d12_device.Get(),
        context.Get(),
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::make_unique<Dml::DmlCommittedResourceAllocator>(d3d12_device.Get()));
    dml_allocator->SetDefaultRoundingMode(AllocatorRoundingMode::Enabled);
    context->SetAllocator(dml_allocator);

    ORT_THROW_IF_FAILED(d3d12_device->SetPrivateData(dml_readback_heap_guid, sizeof(readback_heap), &readback_heap));
    ORT_THROW_IF_FAILED(d3d12_device->SetPrivateData(dml_upload_heap_guid, sizeof(upload_heap), &upload_heap));

    hit = id_to_allocator_map->emplace(id, std::move(dml_allocator)).first;
  }

  return hit->second;
}

void CpuToDmlMemCpy(void* dst, const void* src, size_t num_bytes) {
  const auto* allocInfo = static_cast<const Dml::AllocationInfo*>(dst);
  ID3D12Resource* dst_data = allocInfo->GetResource();

  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(dst_data->GetDevice(IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  Dml::PooledUploadHeap* upload_heap = nullptr;
  uint32_t upload_heap_size = gsl::narrow_cast<uint32_t>(sizeof(upload_heap));
  ORT_THROW_IF_FAILED(d3d12_device->GetPrivateData(dml_upload_heap_guid, &upload_heap_size, &upload_heap));

  upload_heap->BeginUploadToGpu(
      dst_data, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, gsl::make_span(static_cast<const std::byte*>(src), num_bytes));
}

void DmlToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  const auto* allocInfo = static_cast<const Dml::AllocationInfo*>(src);
  ID3D12Resource* src_data = allocInfo->GetResource();

  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(src_data->GetDevice(IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  Dml::ReadbackHeap* readback_heap = nullptr;
  uint32_t readback_heap_size = gsl::narrow_cast<uint32_t>(sizeof(readback_heap));
  ORT_THROW_IF_FAILED(d3d12_device->GetPrivateData(dml_readback_heap_guid, &readback_heap_size, &readback_heap));

  // ReadbackFromGpu already syncs with the CPU and waits for the copy to be completed, so we dont need to sync after
  // this call
  readback_heap->ReadbackFromGpu(
      gsl::make_span(static_cast<std::byte*>(dst), num_bytes),
      src_data,
      0,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

const std::unordered_map<OrtDevice, MemCpyFunc>* GetDmlToHostMemCpyFunction(const OrtDevice& device) {
  static std::unordered_map<OrtDevice, MemCpyFunc> map{
      {OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::MICROSOFT, device.Id()}, DmlToCpuMemCpy},
  };

  return &map;
}

#endif

#ifdef USE_CANN
void CpuToCannMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_CANN().cannMemcpy_HostToDevice(dst, src, num_bytes);
}

void CannToCpuMemCpy(void* dst, const void* src, size_t num_bytes) {
  GetProviderInfo_CANN().cannMemcpy_DeviceToHost(dst, src, num_bytes);
}

const std::unordered_map<OrtDevice, MemCpyFunc>* GetCannToHostMemCpyFunction() {
  static std::unordered_map<OrtDevice, MemCpyFunc> map{
      {OrtDevice{OrtDevice::NPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::HUAWEI, 0}, CannToCpuMemCpy},
  };

  return &map;
}

bool IsCannDeviceIdValid(const onnxruntime::logging::Logger& logger, int id) {
  int num_devices = GetProviderInfo_CANN().cannGetDeviceCount();

  if (0 == num_devices) {
    LOGS(logger, WARNING) << "your system does not have a CANN capable device.";
    return false;
  }

  if (id < 0 || id >= num_devices) {
    LOGS(logger, WARNING) << "cann_device=" << id << " is invalid, must choose device ID between 0 and "
                          << num_devices - 1;
    return false;
  }

  return true;
}

AllocatorPtr GetCannAllocator(OrtDevice::DeviceId id) {
  size_t npu_mem_limit = std::numeric_limits<size_t>::max();
  onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;

  static auto* id_to_allocator_map =
      std::make_unique<std::unordered_map<OrtDevice::DeviceId, AllocatorPtr>>().release();
  auto hit = id_to_allocator_map->find(id);
  if (hit == id_to_allocator_map->end()) {
    auto cann_allocator = GetProviderInfo_CANN().CreateCannAllocator(id, npu_mem_limit, arena_extend_strategy, nullptr);
    hit = id_to_allocator_map->emplace(id, std::move(cann_allocator)).first;
  }

  return hit->second;
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
#if !defined(DISABLE_FLOAT4_TYPES)
      {DataTypeImpl::GetType<Float4E2M1x2>(), NPY_UINT8},
#endif
      {DataTypeImpl::GetType<Float6E2M3>(), NPY_UINT8},
      {DataTypeImpl::GetType<Float6E3M2>(), NPY_UINT8},
#if !defined(DISABLE_FLOAT8_TYPES)
      {DataTypeImpl::GetType<Float8E4M3FN>(), NPY_UINT8},
#endif
  };

  const auto it = type_map.find(tensor_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for Tensor Type. " + std::string(DataTypeImpl::ToString(tensor_type)));
  } else {
    return it->second;
  }
}

MLDataType NumpyTypeToOnnxRuntimeTensorType(int numpy_type) {
  static std::map<int, MLDataType> type_map{
      {NPY_BOOL, DataTypeImpl::GetType<bool>()},
      {NPY_FLOAT, DataTypeImpl::GetType<float>()},
      // Special, not a C type expands to enum value of 16
      {NPY_FLOAT16, DataTypeImpl::GetType<MLFloat16>()},
      {NPY_DOUBLE, DataTypeImpl::GetType<double>()},
      // We dont want to use size specific types such
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

MLDataType OnnxTypeToOnnxRuntimeTensorType(int onnx_element_type) {
  return DataTypeImpl::TensorTypeFromONNXEnum(onnx_element_type)->GetElementType();
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

static void CopyDataToTensor(PyArrayObject* darray, int npy_type, Tensor& tensor,
                             const MemCpyFunc& mem_cpy_to_device = CpuToCpuMemCpy) {
  const auto total_items = tensor.Shape().Size();
  if (npy_type == NPY_UNICODE) {
    // Copy string data which needs to be done after Tensor is allocated.
    // Strings are Python strings or numpy.unicode string.
    std::string* dst = tensor.MutableData<std::string>();
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
    std::string* dst = tensor.MutableData<std::string>();
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
    std::string* dst = tensor.MutableData<std::string>();
    const auto item_size = PyArray_ITEMSIZE(darray);
    const char* src = reinterpret_cast<const char*>(PyArray_DATA(darray));
    for (int i = 0; i < total_items; ++i, src += item_size) {
      // Python unicode strings are assumed to be USC-4. Strings are stored as UTF-8.
      PyObject* item = PyArray_GETITEM(darray, src);
      UniqueDecRefPtr<PyObject> itemGuard(item, DecRefFn<PyObject>());
      PyObject* pStr = PyObject_Str(item);
      UniqueDecRefPtr<PyObject> strGuard(pStr, DecRefFn<PyObject>());
      dst[i] = py::reinterpret_borrow<py::str>(pStr);
    }
  } else {
    void* buffer = tensor.MutableDataRaw();
    size_t len = 0;
    Status status = Tensor::CalculateTensorStorageSize(tensor.DataType(), tensor.Shape(), /*alignment*/ 0, len);
    if (!status.IsOK()) {
      throw std::runtime_error(status.ErrorMessage());
    }
    mem_cpy_to_device(buffer, PyArray_DATA(darray), len);
  }
}

inline void CopyDataToTensor(PyArrayObject* darray, int npy_type, std::unique_ptr<Tensor>& p_tensor,
                             const MemCpyFunc& mem_cpy_to_device = CpuToCpuMemCpy) {
  CopyDataToTensor(darray, npy_type, *p_tensor, mem_cpy_to_device);
}

void CopyDataToTensor(const py::array& py_array, int npy_type, Tensor& tensor, const MemCpyFunc& mem_cpy_to_device) {
  CopyDataToTensor(reinterpret_cast<PyArrayObject*>(py_array.ptr()), npy_type, tensor, mem_cpy_to_device);
}

// Setting `use_numpy_data_memory` to `true` will ensure that the underlying numpy array buffer is directly used
// as the backing data buffer for the ORT Tensor where applicable (for numeric tensors)
// The numpy object owns the memory and needs to be alive until the corresponding OrtValue is in scope
static std::unique_ptr<Tensor> CreateTensor(const AllocatorPtr& alloc, const std::string& name_input,
                                            PyArrayObject* pyObject, bool use_numpy_data_memory = true,
                                            const MemCpyFunc& mem_cpy_to_device = CpuToCpuMemCpy) {
  PyArrayObject* darray = PyArray_GETCONTIGUOUS(pyObject);
  ORT_ENFORCE(darray != nullptr, "The object must be a contiguous array for input '", name_input, "'.");

  UniqueDecRefPtr<PyArrayObject> darray_guard(darray, DecRefFn<PyArrayObject>());
  std::unique_ptr<Tensor> p_tensor;

  const int npy_type = PyArray_TYPE(darray);
  TensorShape shape = GetArrayShape(darray);
  auto element_type = NumpyTypeToOnnxRuntimeTensorType(npy_type);
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
    if (temp->has_optional_type()) {
      const ::onnx::TypeProto_Optional& optional_type_proto = temp->optional_type();
      type_proto = optional_type_proto.elem_type();
    } else {
      type_proto = *temp;
    }
  }

  return type_proto.has_sequence_type();
}

static void CreateSequenceOfTensors(AllocatorPtr alloc, const std::string& name_input,
                                    const InputDefList* input_def_list, PyObject* pylist_obj, OrtValue* p_mlvalue) {
  onnx::TypeProto type_proto;
  if (!CheckIfInputIsSequenceType(name_input, input_def_list, type_proto)) {
    throw std::runtime_error("Input is not of sequence type");
  }

  // set the seq type
  MLDataType seq_dtype = OrtTypeInfo::ElementTypeFromProto(
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto.sequence_type().elem_type().tensor_type().elem_type()));
  auto p_seq_tensors = std::make_unique<TensorSeq>(seq_dtype);

  // populate the seq
  auto list_size = PyList_Size(pylist_obj);
  if (list_size > 0) {
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      auto* py_obj = PyList_GetItem(pylist_obj, i);
      if (!PyObjectCheck_NumpyArray(py_obj)) {
        throw std::runtime_error("CreateSequenceOfTensors: Input is not a tensor");
      }
      auto p_tensor = CreateTensor(alloc, name_input, reinterpret_cast<PyArrayObject*>(py_obj));
      p_seq_tensors->Add(std::move(*p_tensor));
    }
  }

  auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
  p_mlvalue->Init(p_seq_tensors.release(),
                  ml_tensor_sequence,
                  ml_tensor_sequence->GetDeleteFunc());
}

// Setting `use_numpy_data_memory` to `true` will ensure that the underlying numpy array buffer is directly used
// as the backing data buffer for the ORT Tensor where applicable (for numeric tensors)
// The numpy object owns the memory and needs to be alive until the corresponding OrtValue is in scope
static void CreateTensorMLValue(const AllocatorPtr& alloc, const std::string& name_input, PyArrayObject* pyObject,
                                OrtValue* p_mlvalue, bool use_numpy_data_memory = true,
                                const MemCpyFunc& mem_cpy_to_device = CpuToCpuMemCpy) {
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
  auto element_type = NumpyTypeToOnnxRuntimeTensorType(npy_type);

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
                                         PyObject* item, bool owns_item_ref, std::map<KeyType, ValueType>& current,
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
      if (owns_item_ref) {
        Py_XDECREF(item);
      }
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
      if (owns_item_ref) {
        Py_XDECREF(item);
      }
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
  CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, false, *dst, keyGetter, valueGetter);
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
    CreateMapMLValue_LoopIntoMap(pos, key, name_input, value, item, true, (*dstVector)[index], keyGetter, valueGetter);
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
void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc,
                          const std::string& name_input, const py::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array, bool use_numpy_data_memory,
                          const MemCpyFunc& mem_cpy_to_device) {
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

void UpdateOrtValueInplace(OrtValue& dst, const OrtValue& src) {
  if (!dst.IsTensor()) {
    throw std::runtime_error("Inplace update of OrtValues is only supported for Tensors");
  }
  if (!src.IsTensor()) {
    throw std::runtime_error("The source OrtValue must contain a Tensor");
  }

  const auto& dst_tensor = dst.Get<Tensor>();
  const auto& src_tensor = src.Get<Tensor>();

  if (dst_tensor.DataType() != src_tensor.DataType()) {
    throw std::runtime_error("The source and destination OrtValues must have the same data type");
  }

  if (dst_tensor.Shape().Size() != src_tensor.Shape().Size()) {
    throw std::runtime_error("The source and destination OrtValues must have the same size");
  }

  if (dst_tensor.IsDataTypeString()) {
    throw std::runtime_error("Inplace update of string tensors is not supported");
  }

  size_t bytes = 0;
  auto status = Tensor::CalculateTensorStorageSize(dst_tensor.DataType(), dst_tensor.Shape(), 0, bytes);
  if (!status.IsOK()) {
    throw std::runtime_error(status.ErrorMessage());
  }

  const auto src_device = src_tensor.Location().device;
  const auto dst_device = dst_tensor.Location().device;

  void* dst_ptr = dst.GetMutable<Tensor>()->MutableDataRaw();
  const void* src_ptr = src_tensor.DataRaw();

  if (src_device.UsesCpuMemory() && dst_device.UsesCpuMemory()) {
    memcpy(dst_ptr, src_ptr, bytes);
  } else {
    auto copy_fn = CreateDataTransferMemCpy(src_device, dst_device);
    if (!copy_fn) {
      // Fall back to built-in EP copy functions.
      // Gate each path on (Type, VendorId) so that builds with multiple GPU EPs
      // (e.g. CUDA + DML) route through the correct backend.
#ifdef USE_CUDA
      const auto is_cuda_device = [](const OrtDevice& device) {
        return device.Type() == OrtDevice::GPU && device.Vendor() == OrtDevice::VendorIds::NVIDIA;
      };

      if (is_cuda_device(src_device) && is_cuda_device(dst_device)) {
        auto data_transfer = GetGPUDataTransfer();
        ORT_THROW_IF_ERROR(data_transfer->CopyTensor(src_tensor, *dst.GetMutable<Tensor>()));
        return;
      }
      if (src_device.UsesCpuMemory() && is_cuda_device(dst_device)) {
        CpuToCudaMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
      if (is_cuda_device(src_device) && dst_device.UsesCpuMemory()) {
        CudaToCpuMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
#endif
#if USE_MIGRAPHX
      const auto is_migraphx_device = [](const OrtDevice& device) {
        return device.Type() == OrtDevice::GPU && device.Vendor() == OrtDevice::VendorIds::AMD;
      };

      if (src_device.UsesCpuMemory() && is_migraphx_device(dst_device)) {
        CpuToMIGraphXMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
      if (is_migraphx_device(src_device) && dst_device.UsesCpuMemory()) {
        MIGraphXToCpuMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
#endif
#if USE_DML
      const auto is_dml_device = [](const OrtDevice& device) {
        return (device.Type() == OrtDevice::GPU && device.Vendor() == OrtDevice::VendorIds::MICROSOFT) ||
               device.Type() == OrtDevice::DML;
      };

      if (src_device.UsesCpuMemory() && is_dml_device(dst_device)) {
        CpuToDmlMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
      if (is_dml_device(src_device) && dst_device.UsesCpuMemory()) {
        DmlToCpuMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
#endif
#ifdef USE_CANN
      const auto is_cann_device = [](const OrtDevice& device) {
        return device.Type() == OrtDevice::NPU && device.Vendor() == OrtDevice::VendorIds::HUAWEI;
      };

      if (src_device.UsesCpuMemory() && is_cann_device(dst_device)) {
        CpuToCannMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
      if (is_cann_device(src_device) && dst_device.UsesCpuMemory()) {
        CannToCpuMemCpy(dst_ptr, src_ptr, bytes);
        return;
      }
#endif
      throw std::runtime_error("Unable to copy data between the source and destination devices");
    }
    copy_fn(dst_ptr, src_ptr, bytes);
  }
}

}  // namespace python
}  // namespace onnxruntime
