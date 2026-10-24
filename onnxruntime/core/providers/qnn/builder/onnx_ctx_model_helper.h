// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

namespace qnn {

class QnnModel;
class QnnBackendManager;
using QnnModelLookupTable = std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>;

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string MAIN_CONTEXT = "main_context";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string EP_SDK_VER = "ep_sdk_version";
static const std::string PARTITION_NAME = "partition_name";
static const std::string SOURCE = "source";
static const std::string MAX_SIZE = "max_size";

bool GraphHasEpContextNode(const onnxruntime::GraphViewer& graph_viewer);

bool IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs);

Status GetMainContextNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                          std::vector<int>& main_context_pos);

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph);

Status GetEpContextFromMainNode(const onnxruntime::Node& main_context_node,
                                const onnxruntime::PathString& ctx_onnx_model_path,
                                QnnBackendManager* qnn_backend_manager,
                                QnnModelLookupTable& qnn_models,
                                int64_t max_spill_fill_size,
                                const EpContextDataCallbacks& callbacks);

Status TryGetMaxSpillFillSize(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                              uint32_t total_context_size,
                              int64_t& max_spill_fill_size,
                              std::vector<int>& main_context_pos_list);

Status LoadQnnCtxFromOnnxGraph(const onnxruntime::GraphViewer& graph_viewer,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               QnnModelLookupTable& qnn_models,
                               const logging::Logger& logger,
                               int64_t max_spill_fill_size,
                               const EpContextDataCallbacks& callbacks);

Status CreateEPContextNodes(Model* model,
                            unsigned char* buffer,
                            uint64_t buffer_size,
                            const std::string& sdk_build_version,
                            const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                            const std::unordered_map<std::string, std::unique_ptr<QnnModel>>& qnn_models,
                            const onnxruntime::PathString& context_model_path,
                            bool qnn_context_embed_mode,
                            uint64_t max_spill_fill_buffer_size,
                            const logging::Logger& logger,
                            bool share_ep_contexts,
                            bool stop_share_ep_contexts,
                            const EpContextDataCallbacks& callbacks);
}  // namespace qnn
}  // namespace onnxruntime
