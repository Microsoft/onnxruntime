// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_op_manager.h"

namespace onnxruntime {
DnnlOpManager::DnnlOpManager() {
  dnnl_ops_map_.emplace(std::make_pair("AveragePool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("BatchNormalization", std::unique_ptr<DnnlNodeCapability>(new DnnlBatchNormalizationNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Conv", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("GlobalAveragePool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("GlobalMaxPool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("LRN", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MatMul", std::unique_ptr<DnnlNodeCapability>(new DnnlMatMulNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MaxPool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("ReduceMean", std::unique_ptr<DnnlNodeCapability>(new DnnlReduceMeanNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Relu", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Sum", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
#if defined(ENABLE_TRAINING)
  dnnl_ops_map_.emplace(std::make_pair("ConvGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("ReluGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MaxPoolGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
#endif  // ENABLE_TRAINING
}

bool DnnlOpManager::IsNodeSupported(const Node* node) const {
  auto it = dnnl_ops_map_.find(node->OpType());
  if (it == dnnl_ops_map_.end()) {
    return false;
  }
  return it->second->Supported(node);
}

bool DnnlOpManager::IsOpTypeAvalible(const std::string& opType) const {
  auto op_it = dnnl_ops_map_.find(opType);
  return (op_it != dnnl_ops_map_.end());
}
}  // namespace onnxruntime
