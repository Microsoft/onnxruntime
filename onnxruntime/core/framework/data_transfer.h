// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>

struct OrtDevice;

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
class SparseTensor;
#endif
namespace common {
class Status;
}

// Data transfer interface.
class IDataTransfer {
 public:
  virtual ~IDataTransfer() = default;

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const = 0;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const = 0;

  struct SrcDstPair {
    std::reference_wrapper<const Tensor> src;
    std::reference_wrapper<Tensor> dst;
    int exec_queue_id;
  };

  // batched copy. default implementation copies each entry sequentially, and returns on first failure.
  virtual common::Status CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const;

  struct SparseSrcDstPair {
    std::reference_wrapper<const SparseTensor> src;
    std::reference_wrapper<SparseTensor> dst;
    int exec_queue_id;
  };

  virtual common::Status CopyTensors(const std::vector<SparseSrcDstPair>& src_dst_pairs) const;
};

class CPUDataTransfer : public IDataTransfer {
 public:
  CPUDataTransfer() = default;
  // Dampen MSVC warning about not fully overriding CopyTensor
  using IDataTransfer::CopyTensor;
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};
}  // namespace onnxruntime
