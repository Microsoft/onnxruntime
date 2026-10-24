// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"

#include <iostream>
#include <fstream>
#include <filesystem>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/shared_context.h"

namespace onnxruntime {
namespace qnn {
namespace {

Status ConvertAndReleaseCallbackStatus(OrtStatus* status) {
  Ort::Status callback_status{status};
  if (callback_status.IsOK()) {
    return Status::OK();
  }

  return Status(common::StatusCategory::ONNXRUNTIME,
                static_cast<int>(callback_status.GetErrorCode()),
                callback_status.GetErrorMessage());
}

}  // namespace

bool GraphHasEpContextNode(const onnxruntime::GraphViewer& graph_viewer) {
  // It's an Onnx model with Qnn context cache binary if it has a node with EPContext type
  // and the source is QNN or QNNExecutionProvider.
  for (const auto& node : graph_viewer.Nodes()) {
    if (EPCONTEXT_OP == node.OpType()) {
      NodeAttrHelper node_helper(node);
      std::string cache_source = qnn::utils::GetLowercaseString(node_helper.Get(SOURCE, ""));

      if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
        return true;
      }
    }
  }
  return false;
}

bool IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const onnxruntime::GraphViewer& graph_viewer(fused_node_graph.filtered_graph);
    bool has_qnn_ep_context_node = GraphHasEpContextNode(graph_viewer);
    if (has_qnn_ep_context_node) {
      return true;
    }
  }
  return false;
}

Status GetMainContextNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                          std::vector<int>& main_context_pos) {
  for (size_t i = 0; i < fused_nodes_and_graphs.size(); ++i) {
    // Only EPContext nodes are filtered in
    // There is only one EPContext node in one filtered graph -- this is guaranteed by GetCapability
    const onnxruntime::GraphViewer& graph_viewer(fused_nodes_and_graphs[i].filtered_graph);
    ORT_RETURN_IF(graph_viewer.NumberOfNodes() != 1, "One filtered graph should has only one EPContext node!");
    const Node& ep_context_node = *graph_viewer.Nodes().begin();
    ORT_RETURN_IF_NOT(EPCONTEXT_OP == ep_context_node.OpType(), "Should only filter in the EPContext node.");
    NodeAttrHelper node_helper(ep_context_node);
    int64_t is_main_context = node_helper.Get(MAIN_CONTEXT, static_cast<int64_t>(0));
    if (1 == is_main_context) {
      main_context_pos.push_back(static_cast<int>(i));
    }
  }

  ORT_RETURN_IF(main_context_pos.size() < 1, "Failed to find the EPContext node with main_context=1");
  return Status::OK();
}

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph) {
  for (size_t i = 0; i < names.size(); ++i) {
    std::string name = names[i];
    ORT_RETURN_IF(tensor_info_table.find(name) == tensor_info_table.end(), "Tensor name: ", name, " not found in tensor_info_table");
    const OnnxTensorInfo& tensor_info = tensor_info_table.at(name);
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> tensor_type = Factory<ONNX_NAMESPACE::TypeProto>::Create();
    tensor_type->mutable_tensor_type()->set_elem_type(tensor_info.data_type_);
    tensor_type->mutable_tensor_type()->mutable_shape();
    for (size_t j = 0; j < tensor_info.shape_.size(); ++j) {
      tensor_type->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(tensor_info.shape_[j]);
    }
    auto& input_arg = graph.GetOrCreateNodeArg(name, tensor_type.get());
    node_args.push_back(&input_arg);
  }
  return Status::OK();
}

Status GetEpContextFromMainNode(const onnxruntime::Node& main_context_node,
                                const onnxruntime::PathString& ctx_onnx_model_path,
                                QnnBackendManager* qnn_backend_manager,
                                QnnModelLookupTable& qnn_models,
                                int64_t max_spill_fill_size,
                                const EpContextDataCallbacks& callbacks) {
  ORT_RETURN_IF_NOT(EPCONTEXT_OP == main_context_node.OpType(), "Should only filter in the EPContext node.");
  NodeAttrHelper node_helper(main_context_node);
  bool is_embed_mode = node_helper.Get(EMBED_MODE, true);
  if (is_embed_mode) {
    static const std::string empty_context_binary;
    const std::string& context_binary = node_helper.Get(EP_CACHE_CONTEXT, empty_context_binary);
    return qnn_backend_manager->LoadCachedQnnContextFromBuffer(const_cast<char*>(context_binary.c_str()),
                                                               static_cast<uint64_t>(context_binary.length()),
                                                               "",
                                                               main_context_node.Name(),
                                                               qnn_models,
                                                               max_spill_fill_size);
  }

  std::filesystem::path folder_path = std::filesystem::path(ctx_onnx_model_path).parent_path();
  std::string external_qnn_ctx_binary_file_name = node_helper.Get(EP_CACHE_CONTEXT, "");
  ORT_RETURN_IF(external_qnn_ctx_binary_file_name.empty(), "The file path in ep_cache_context should not be empty.");

  if (callbacks.read_func != nullptr) {
    Ort::AllocatorWithDefaultOptions allocator;
    void* context_buffer = nullptr;
    size_t context_buffer_size = 0;
    OrtStatus* callback_status = callbacks.read_func(callbacks.read_state,
                                                     external_qnn_ctx_binary_file_name.c_str(),
                                                     allocator,
                                                     &context_buffer,
                                                     &context_buffer_size);
    std::unique_ptr<void, Ort::detail::AllocatedFree> context_buffer_guard{
        context_buffer, Ort::detail::AllocatedFree{allocator}};
    ORT_RETURN_IF_ERROR(ConvertAndReleaseCallbackStatus(callback_status));
    if (context_buffer_size == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "QNN EPContext read callback returned an empty payload.");
    }
    ORT_RETURN_IF(context_buffer == nullptr && context_buffer_size != 0,
                  "EPContext read callback returned a null buffer for non-empty QNN context data.");
    return qnn_backend_manager->LoadCachedQnnContextFromBuffer(static_cast<char*>(context_buffer),
                                                               static_cast<uint64_t>(context_buffer_size),
                                                               "",
                                                               main_context_node.Name(),
                                                               qnn_models,
                                                               max_spill_fill_size);
  }

  // Validate that the cache path does not escape the model directory.
  // Rejects absolute paths, ".." traversal, and symlink-based escapes.
  constexpr const char* path_resolution_guidance =
      ". If session.model_external_initializers_file_folder_path is set, set ep.context_file_path to the "
      "EPContext model path so relative ep_cache_context paths are resolved from the EPContext model directory.";
  auto validate_status = ::onnxruntime::utils::ValidateExternalDataPath(
      std::filesystem::path(ctx_onnx_model_path), std::filesystem::path(external_qnn_ctx_binary_file_name));
  if (!validate_status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, validate_status.ErrorMessage(), path_resolution_guidance);
  }

  std::filesystem::path context_binary_path = folder_path / external_qnn_ctx_binary_file_name;
  if (!std::filesystem::is_regular_file(context_binary_path)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_GRAPH, "The external EP context file '", context_binary_path.string(),
        "' does not exist or is not accessible", path_resolution_guidance);
  }

  std::string context_binary_path_str = context_binary_path.string();
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  if (qnn_backend_manager->FileMappingIsEnabled()) {
    return qnn_backend_manager->LoadCachedQnnContextFromBuffer(nullptr,
                                                               0,
                                                               context_binary_path_str,
                                                               main_context_node.Name(),
                                                               qnn_models,
                                                               max_spill_fill_size);
  }
#endif

  size_t buffer_size{0};
  std::ifstream cache_file(context_binary_path.string().c_str(), std::ifstream::binary);
  ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to open cache file.");

  cache_file.seekg(0, cache_file.end);
  buffer_size = static_cast<size_t>(cache_file.tellg());
  ORT_RETURN_IF(0 == buffer_size, "Empty cache file encountered.");

  cache_file.seekg(0, cache_file.beg);
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);
  ORT_RETURN_IF(nullptr == buffer, "Failed to allocate memory for cache file.");
  // Load file into buffer
  const auto& read_result = cache_file.read(buffer.get(), buffer_size);
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context file.");
  cache_file.close();
  return qnn_backend_manager->LoadCachedQnnContextFromBuffer(buffer.get(),
                                                             static_cast<uint64_t>(buffer_size),
                                                             context_binary_path_str,
                                                             main_context_node.Name(),
                                                             qnn_models,
                                                             max_spill_fill_size);
}

Status TryGetMaxSpillFillSize(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                              uint32_t total_context_size,
                              int64_t& max_spill_fill_size,
                              std::vector<int>& main_context_pos_list) {
  max_spill_fill_size = 0;
  int max_size_index = 0;
  for (uint32_t i = 0; i < total_context_size; ++i) {
    auto index = main_context_pos_list[i];
    const onnxruntime::GraphViewer& main_ctx_graph_viewer(fused_nodes_and_graphs[index].filtered_graph);
    ORT_RETURN_IF(main_ctx_graph_viewer.NumberOfNodes() != 1, "One filtered graph should has only one EPContext node!");
    const Node& ep_context_node = *main_ctx_graph_viewer.Nodes().begin();
    NodeAttrHelper node_helper(ep_context_node);
    int64_t max_size = node_helper.Get(MAX_SIZE, static_cast<int64_t>(0));
    if (max_size > max_spill_fill_size) {
      max_spill_fill_size = max_size;
      max_size_index = i;
    }
  }
  if (0 != max_size_index) {
    int tmp_index = main_context_pos_list[0];
    main_context_pos_list[0] = main_context_pos_list[max_size_index];
    main_context_pos_list[max_size_index] = tmp_index;
  }

  return Status::OK();
}

Status LoadQnnCtxFromOnnxGraph(const onnxruntime::GraphViewer& graph_viewer,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               QnnModelLookupTable& qnn_models,
                               const logging::Logger& logger,
                               int64_t max_spill_fill_size,
                               const EpContextDataCallbacks& callbacks) {
  ORT_RETURN_IF(graph_viewer.NumberOfNodes() != 1, "One filtered graph should has only one EPContext node!");
  Status status = GetEpContextFromMainNode(*graph_viewer.Nodes().begin(), ctx_onnx_model_path, qnn_backend_manager,
                                           qnn_models, max_spill_fill_size, callbacks);

  // This is the protocol with customer that status with INVALID_GRAPH will be generated if failed to load context model
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Failed to load from EpContext model. " << status.ErrorMessage();
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to load from EpContext model. ", status.ErrorMessage());
  }

  return Status::OK();
}

Status CreateEPContextNodes(Model* model,
                            unsigned char* buffer,
                            uint64_t buffer_size,
                            const std::string& sdk_build_version,
                            const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                            const QnnModelLookupTable& qnn_models,
                            const onnxruntime::PathString& context_model_path,
                            bool qnn_context_embed_mode,
                            uint64_t max_spill_fill_buffer_size,
                            const logging::Logger& logger,
                            bool share_ep_contexts,
                            bool stop_share_ep_contexts,
                            const EpContextDataCallbacks& callbacks) {
  auto& graph = model->MainGraph();

  using namespace ONNX_NAMESPACE;
  int index = 0;
  // Still need more work to support multiple partition, it's out of EP's scope.
  // Already have code to make sure it's single partition before this method get invoked.
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_graph.fused_node;
    auto qnn_model_kv = qnn_models.find(fused_node.Name());
    ORT_RETURN_IF(qnn_model_kv == qnn_models.end(), fused_node.Name(), " not exist in QnnModel table.");

    auto qnn_model = qnn_model_kv->second.get();
    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;
    ORT_RETURN_IF_ERROR(CreateNodeArgs(qnn_model->GetInputNames(), qnn_model->GetInputsInfo(), inputs, graph));
    ORT_RETURN_IF_ERROR(CreateNodeArgs(qnn_model->GetOutputNames(), qnn_model->GetOutputsInfo(), outputs, graph));

    const std::string& graph_name = fused_node.Name();
    auto& ep_node = graph.AddNode(graph_name,
                                  EPCONTEXT_OP,
                                  "Onnx Qnn context binary cache for graph partition: " + graph_name,
                                  inputs,
                                  outputs,
                                  nullptr,
                                  kMSDomain);

    // Only dump the context buffer once since all QNN graphs are in one single context
    if (0 == index) {
      if (qnn_context_embed_mode) {
        std::string cache_payload(buffer, buffer + buffer_size);
        ep_node.AddAttribute(EP_CACHE_CONTEXT, cache_payload);
      } else {
        onnxruntime::PathString context_bin_path;
        std::string context_cache_name;
        auto pos = context_model_path.find_last_of(ORT_TSTR("."));
        if (pos != std::string::npos) {
          context_bin_path = context_model_path.substr(0, pos);
        } else {
          context_bin_path = context_model_path;
        }

        if (context_bin_path.empty()) {
          // Context bin path is empty, so just use the graph name (e.g., "QNNExecutionProvider_QNN_13728744673520368385_2_0").
          // This happens if both the input model and output model are stored in buffers (i.e., there are no paths).
          context_bin_path = ToPathString(graph_name);
        }

        context_bin_path = context_bin_path + ToPathString("_qnn.bin");
        context_cache_name = PathToUTF8String(std::filesystem::path(context_bin_path).filename().native());

        // If generate ctx.onnx with share_ep_context enabled, all ctx.onnx should point to the same ctx.bin
        if (share_ep_contexts) {
          auto shared_ctx_bin_name = SharedContext::GetInstance().GetSharedCtxBinFileName();
          if (shared_ctx_bin_name.empty()) {
            SharedContext::GetInstance().SetSharedCtxBinFileName(context_cache_name);
          } else {
            context_cache_name = shared_ctx_bin_name;
            auto model_folder_path = std::filesystem::path(context_bin_path).parent_path().string();
            context_bin_path = ToPathString(model_folder_path + "/" + context_cache_name);
          }
        }

        // Write the ctx.bin file for the case: 1. no share_ep_context enabled, write for every session
        // 2. share_ep_context enabled, only write for the last session which has stop_share_ep_contexts enabled
        if (!share_ep_contexts || (share_ep_contexts && stop_share_ep_contexts)) {
          if (callbacks.write_func != nullptr) {
            ORT_RETURN_IF_ERROR(ConvertAndReleaseCallbackStatus(callbacks.write_func(callbacks.write_state,
                                                                                     context_cache_name.c_str(),
                                                                                     buffer,
                                                                                     SafeInt<size_t>(buffer_size))));
          } else {
            std::ofstream of_stream(context_bin_path.c_str(), std::ofstream::binary);
            if (!of_stream) {
              LOGS(logger, ERROR) << "Failed to open create context file.";
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to open context cache file.");
            }
            of_stream.write(reinterpret_cast<char*>(buffer), buffer_size);
          }
        }

        ep_node.AddAttribute(EP_CACHE_CONTEXT, context_cache_name);
        if (share_ep_contexts && stop_share_ep_contexts) {
          SharedContext::GetInstance().ResetSharedCtxBinFileName();
        }

        ep_node.AddAttribute(MAX_SIZE, static_cast<int64_t>(max_spill_fill_buffer_size));
      }
    } else {
      ep_node.AddAttribute(MAIN_CONTEXT, static_cast<int64_t>(0));
    }
    int64_t embed_mode = qnn_context_embed_mode ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    ep_node.AddAttribute(EMBED_MODE, embed_mode);
    ep_node.AddAttribute(EP_SDK_VER, sdk_build_version);
    ep_node.AddAttribute(PARTITION_NAME, graph_name);
    ep_node.AddAttribute(SOURCE, kQnnExecutionProvider);
    ++index;
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
