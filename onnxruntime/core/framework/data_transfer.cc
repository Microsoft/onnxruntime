// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_transfer.h"
#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#endif

#include "core/framework/ortdevice.h"

namespace onnxruntime {

common::Status IDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensor(src, dst, 0);
}

common::Status IDataTransfer::CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const {
  for (const auto& pair : src_dst_pairs) {
    ORT_RETURN_IF_ERROR(CopyTensor(pair.src, pair.dst, pair.exec_queue_id));
  }

  return Status::OK();
}

common::Status IDataTransfer::CopyTensors(const std::vector<SparseSrcDstPair>& src_dst_pairs) const {
  for (const auto& pair : src_dst_pairs) {
    ORT_RETURN_IF_ERROR(pair.src.get().Copy(*this, pair.dst, pair.exec_queue_id));
  }
  return Status::OK();
}

bool CPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::CPU;
}

common::Status CPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  if (src_data == dst_data) {
    // no need copying as both pointers are referring to same piece of memory.
    return Status::OK();
  }
  // Copying only happens between two same size tensors.
  ORT_ENFORCE(src.SizeInBytes() == dst.SizeInBytes());
  memcpy(dst_data, src_data, src.SizeInBytes());
  return Status::OK();
}

};  // namespace onnxruntime
