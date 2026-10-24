// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_data_transfer_plugin.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda_plugin {

CudaDataTransfer::CudaDataTransfer(const OrtApi& ort_api, const OrtEpApi& ep_api)
    : OrtDataTransferImpl{},
      ort_api_(ort_api),
      ep_api_(ep_api) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
}

/*static*/ void ORT_API_CALL CudaDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  delete static_cast<CudaDataTransfer*>(this_ptr);
}

/*static*/ bool ORT_API_CALL CudaDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_device,
    const OrtMemoryDevice* dst_device) noexcept {
  auto* dt = static_cast<const CudaDataTransfer*>(this_ptr);
  const OrtEpApi& ep_api = dt->ep_api_;
  auto src_type = ep_api.MemoryDevice_GetDeviceType(src_device);
  auto dst_type = ep_api.MemoryDevice_GetDeviceType(dst_device);

  bool src_is_cpu = (src_type == OrtMemoryInfoDeviceType_CPU);
  bool dst_is_cpu = (dst_type == OrtMemoryInfoDeviceType_CPU);
  bool src_is_gpu = (src_type == OrtMemoryInfoDeviceType_GPU);
  bool dst_is_gpu = (dst_type == OrtMemoryInfoDeviceType_GPU);

  if ((src_is_gpu && ep_api.MemoryDevice_GetVendorId(src_device) != OrtDevice::VendorIds::NVIDIA) ||
      (dst_is_gpu && ep_api.MemoryDevice_GetVendorId(dst_device) != OrtDevice::VendorIds::NVIDIA)) {
    return false;
  }

  // Support CPU→GPU, GPU→CPU, GPU→GPU
  return (src_is_cpu && dst_is_gpu) ||
         (src_is_gpu && dst_is_cpu) ||
         (src_is_gpu && dst_is_gpu);
}

/*static*/ OrtStatus* ORT_API_CALL CudaDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr,
    const OrtValue** src_tensors,
    OrtValue** dst_tensors,
    OrtSyncStream** streams,
    size_t count) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* dt = static_cast<CudaDataTransfer*>(this_ptr);
  bool need_stream_sync = false;

  for (size_t i = 0; i < count; ++i) {
    Ort::ConstValue src{src_tensors[i]};
    Ort::UnownedValue dst{dst_tensors[i]};

    size_t bytes = 0;
    auto* status = dt->ort_api_.GetTensorSizeInBytes(src_tensors[i], &bytes);
    if (status != nullptr) {
      return status;
    }
    if (bytes == 0) continue;

    const void* src_data = src.GetTensorRawData();
    void* dst_data = dst.GetTensorMutableRawData();

    // Determine copy direction
    const OrtMemoryInfo* src_mem_info = src.GetTensorMemoryInfo();
    const OrtMemoryInfo* dst_mem_info = dst.GetTensorMemoryInfo();
    const OrtMemoryDevice* src_dev = dt->ep_api_.MemoryInfo_GetMemoryDevice(src_mem_info);
    const OrtMemoryDevice* dst_dev = dt->ep_api_.MemoryInfo_GetMemoryDevice(dst_mem_info);
    auto src_dev_type = dt->ep_api_.MemoryDevice_GetDeviceType(src_dev);
    auto dst_dev_type = dt->ep_api_.MemoryDevice_GetDeviceType(dst_dev);
    auto src_mem_type = dt->ep_api_.MemoryDevice_GetMemoryType(src_dev);

    if (src_data == nullptr || dst_data == nullptr) {
      int device_id = 0;
      const auto* gpu_info = dst_dev_type == OrtMemoryInfoDeviceType_GPU ? dst_mem_info : src_mem_info;
      auto* id_status = dt->ort_api_.MemoryInfoGetId(gpu_info, &device_id);
      if (id_status != nullptr) return id_status;
      const auto budget = GetCudaDeviceMemoryBudget(static_cast<OrtDevice::DeviceId>(device_id))->GetStats();
      return dt->ort_api_.CreateStatus(
          ORT_FAIL, ("CUDA tensor allocation failed before copy: requested " + std::to_string(bytes) +
                     " bytes; device memory budget limit " + std::to_string(budget.limit) +
                     ", committed " + std::to_string(budget.committed) + " bytes")
                        .c_str());
    }

    cudaMemcpyKind copy_kind;
    if (src_dev_type == OrtMemoryInfoDeviceType_CPU && dst_dev_type == OrtMemoryInfoDeviceType_GPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (src_dev_type == OrtMemoryInfoDeviceType_GPU && dst_dev_type == OrtMemoryInfoDeviceType_CPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else if (src_dev_type == OrtMemoryInfoDeviceType_GPU && dst_dev_type == OrtMemoryInfoDeviceType_GPU) {
      copy_kind = cudaMemcpyDeviceToDevice;
    } else {
      return dt->ort_api_.CreateStatus(ORT_EP_FAIL, "Unsupported copy direction");
    }

    // Use async copy if stream is provided
    if (streams != nullptr && streams[i] != nullptr) {
      cudaStream_t cuda_stream = static_cast<cudaStream_t>(
          Ort::GetApi().SyncStream_GetHandle(streams[i]));
      PL_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, copy_kind, cuda_stream));
    } else {
      if (copy_kind == cudaMemcpyDeviceToDevice && dst_data == src_data) {
        continue;
      }

      PL_CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, copy_kind));

      if (copy_kind == cudaMemcpyDeviceToDevice) {
        // Match the built-in CUDA EP: cudaMemcpy D2D launches on the default
        // stream but does not guarantee host-side completion on return.
        need_stream_sync = true;
      } else if (copy_kind == cudaMemcpyHostToDevice && src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
        // Pageable host memory may still be in flight after cudaMemcpy returns.
        need_stream_sync = true;
      }
    }
  }

  if (need_stream_sync) {
    PL_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
