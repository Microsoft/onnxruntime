// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "migraphx_call.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {
  // create streams, default is nullptr
  streams_[kHipStreamDefault] = nullptr;
  HIP_CALL_THROW(hipStreamCreateWithFlags(&streams_[kHipStreamCopyIn], hipStreamNonBlocking));
  HIP_CALL_THROW(hipStreamCreateWithFlags(&streams_[kHipStreamCopyOut], hipStreamNonBlocking));
}

GPUDataTransfer::~GPUDataTransfer() {
  HIP_CALL_THROW(hipStreamDestroy(streams_[kHipStreamCopyIn]));
  HIP_CALL_THROW(hipStreamDestroy(streams_[kHipStreamCopyOut]));
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::HIP_PINNED
         || dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::HIP_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, streams_[exec_queue_id]));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToDevice, streams_[kHipStreamDefault]));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, streams_[exec_queue_id]));
    } else {
      // copying from GPU to CPU memory, this is blocking
      HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
