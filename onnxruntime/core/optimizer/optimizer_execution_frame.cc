// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/callback.h"
#include "core/framework/TensorSeq.h"
#include "core/optimizer/optimizer_execution_frame.h"

namespace onnxruntime {

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const InitializedTensorSet& initialized_tensor_set,
                                    const Path& model_path,
                                    const IExecutionProvider& execution_provider,
                                    const std::function<bool(const std::string&)>& is_sparse_initializer_func)
    : execution_provider_(execution_provider),
      is_sparse_initializer_func_(is_sparse_initializer_func) {
  allocator_ptr_ = execution_provider_.GetAllocator(device_id_, mem_type_);
  ORT_ENFORCE(allocator_ptr_, "Failed to get allocator for optimizer");

  data_transfer_mgr_.RegisterDataTransfer(std::make_unique<CPUDataTransfer>());

  // Create MLValues related maps
  auto initialize_maps = [this, &initialized_tensor_set, &model_path](const NodeArg& arg, size_t /*index*/) -> Status {
    int idx = ort_value_name_idx_map_.Add(arg.Name());
    ort_value_idx_nodearg_map_[idx] = &arg;

    // Only create OrtValue instances for initializers used by an array of nodes.
    InitializedTensorSet::const_iterator it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      const auto& tensor_proto = *(it->second);
      size_t cpu_tensor_length;
      ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor_proto, &cpu_tensor_length));
      OrtValue ort_value;
      std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
      std::unique_ptr<Tensor> p_tensor;
      ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(Env::Default(),
                                                      model_path.IsEmpty() ? nullptr : model_path.ToPathString().c_str(),
                                                      tensor_proto,
                                                      MemBuffer(data.get(), cpu_tensor_length, allocator_ptr_->Info()),
                                                      ort_value));

      initializers_[idx] = ort_value;
      buffer_for_initialized_tensors_[idx] = std::move(data);
    }

    return Status::OK();
  };

  // TODO: node->ImplicitInputDefs() need to be added here for control flow nodes.
  for (auto* node : nodes) {
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps));
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps));
  }

  node_index_info_ = std::make_unique<NodeIndexInfo>(nodes, ort_value_name_idx_map_);
}

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const std::unordered_map<std::string, OrtValue>& initialized_tensor_set,
                                    const Path& model_path,
                                    const IExecutionProvider& execution_provider,
                                    const std::function<bool(const std::string&)>& is_sparse_initializer_func)
    : execution_provider_(execution_provider),
      is_sparse_initializer_func_(is_sparse_initializer_func) {
  allocator_ptr_ = execution_provider_.GetAllocator(device_id_, mem_type_);
  ORT_ENFORCE(allocator_ptr_, "Failed to get allocator for optimizer");

  data_transfer_mgr_.RegisterDataTransfer(std::make_unique<CPUDataTransfer>());

  // Create MLValues related maps
  auto initialize_maps = [this, &initialized_tensor_set, &model_path](const NodeArg& arg, size_t /*index*/) -> Status {
    (void)model_path;
    int idx = ort_value_name_idx_map_.Add(arg.Name());
    ort_value_idx_nodearg_map_[idx] = &arg;

    // Only create OrtValue instances for initializers used by an array of nodes.
    std::unordered_map<std::string, OrtValue>::const_iterator it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      initializers_[idx] = it->second;
    }
    return Status::OK();
  };

  // TODO: node->ImplicitInputDefs() need to be added here for control flow nodes.
  for (auto* node : nodes) {
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps));
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps));
  }

  node_index_info_ = std::make_unique<NodeIndexInfo>(nodes, ort_value_name_idx_map_);
}

std::unique_ptr<const OpKernel> OptimizerExecutionFrame::Info::CreateKernel(const Node* node) const {
  std::unique_ptr<OpKernel> op_kernel;
  std::shared_ptr<KernelRegistry> kernel_registry = execution_provider_.GetKernelRegistry();
  auto status = kernel_registry->TryCreateKernel(*node, execution_provider_, initializers_,
                                                 ort_value_name_idx_map_, FuncManager(), data_transfer_mgr_,
                                                 op_kernel);

  // Kernel found in the CPU kernel registry
  if (status.IsOK())
    return std::unique_ptr<const OpKernel>(std::move(op_kernel));

  // No kernel found in the CPU kernel registry
  return nullptr;
}

// For optimizer, probably no need to pass feed_mlvalue_idxs, feeds to initialize IExecutionFrame.
// If needed, the parameters of OptimizerExecutionFrame ctor can be changed later.
OptimizerExecutionFrame::OptimizerExecutionFrame(const Info& info,
                                                 const std::vector<int>& fetch_mlvalue_idxs,
                                                 const std::vector<OrtValue>& fetches)
    : IExecutionFrame(info.GetMLValueNameIdxMap(), info.GetNodeIndexInfo(), fetch_mlvalue_idxs),
      info_(info) {
  Init(std::vector<int>(), std::vector<OrtValue>(), info.GetInitializers(), info.GetSparseInitializerLookupFunc(), fetches);
}

AllocatorPtr OptimizerExecutionFrame::GetAllocatorImpl(const OrtMemoryInfo& info) const {
  return info_.GetAllocator(info);
}

Status OptimizerExecutionFrame::CopyTensor(const Tensor& src, Tensor& dest) const {
  return info_.GetDataTransferManager().CopyTensor(src, dest);
}

const DataTransferManager& OptimizerExecutionFrame::GetDataTransferManager() const {
  return info_.GetDataTransferManager();
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status OptimizerExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) {
  const DataTypeImpl* ml_type = utils::GetMLDataType(*(info_.GetMLValueIdxNodeArgMap().at(ort_value_idx)));
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, ort_value index=" + std::to_string(ort_value_idx));
  if (ml_type->IsSparseTensorType()) {
    auto element_type = ml_type->AsSparseTensorType()->GetElementType();
    auto container_type = DataTypeImpl::GetType<SparseTensor>();
    auto sparse = std::make_unique<SparseTensor>(element_type, *shape, info_.GetAllocator());
    ort_value.Init(sparse.release(), container_type, container_type->GetDeleteFunc());
    return Status::OK();
  }

  if (ml_type->IsTensorSequenceType()) {
    auto element_type = ml_type->AsSequenceTensorBase()->GetElementType();
    auto p_sequence = std::make_unique<TensorSeq>(element_type);
    auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
    ort_value.Init(p_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());
    return Status::OK();
  }

  if (!ml_type->IsTensorType()) {
    assert(ml_type->AsNonTensorTypeBase() != nullptr);
    const NonTensorTypeBase* non_tensor_type = static_cast<const NonTensorTypeBase*>(ml_type);
    auto creator = non_tensor_type->GetCreateFunc();
    ort_value.Init(creator(), non_tensor_type, non_tensor_type->GetDeleteFunc());
    return Status::OK();
  }

  // tensors
  auto element_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
  AllocatorPtr allocator_ptr = info_.GetAllocator();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              *shape,
                                                              allocator_ptr);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  return Status::OK();
}

}  // namespace onnxruntime
