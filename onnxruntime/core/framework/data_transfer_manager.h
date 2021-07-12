// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/common.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

// Data transfer manager, which has all functions registered to copy tensors with different location.
// It's not thread-safe.
class DataTransferManager {
 public:
  DataTransferManager() = default;
  //static DataTransferManager& Instance();

  common::Status RegisterDataTransfer(std::unique_ptr<IDataTransfer> data_transfer);

  const IDataTransfer* GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const;
  common::Status CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const;
  common::Status CopyTensor(const SparseTensor& src, SparseTensor& dst) const;
  common::Status CopyTensor(const SparseTensor& src, SparseTensor& dst, int exec_queue_id) const;
  common::Status CopyTensors(const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DataTransferManager);

  // It's assumed that data transfers in this array have no overlap in terms of copying functionality.
  std::vector<std::unique_ptr<IDataTransfer>> datatransfers_;
};
}  // namespace onnxruntime
