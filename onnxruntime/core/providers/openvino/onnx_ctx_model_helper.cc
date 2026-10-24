// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/callback_buffer_stream.h"

namespace {

constexpr const char* kPathResolutionGuidance =
    ". If session.model_external_initializers_file_folder_path is set, set ep.context_file_path to the "
    "EPContext model path so relative ep_cache_context paths are resolved from the EPContext model directory.";

onnxruntime::Status ValidateEpContextDataPath(const std::filesystem::path& model_path,
                                              const std::filesystem::path& data_path) {
  const auto status = onnxruntime::utils::ValidateExternalDataPath(model_path, data_path);
  if (!status.IsOK()) {
    return onnxruntime::Status(status.Category(), status.Code(), status.ErrorMessage() + kPathResolutionGuidance);
  }
  return onnxruntime::Status::OK();
}

using onnxruntime::openvino_ep::CallbackBufferIStream;
using onnxruntime::openvino_ep::OrtAllocatorDeleter;

onnxruntime::Status ReadEpContextData(OrtReadNamedBufferFunc read_func, void* read_state,
                                      const std::string& name,
                                      std::unique_ptr<CallbackBufferIStream>& stream) {
  OrtAllocator* allocator = nullptr;
  ORT_RETURN_IF_ERROR(onnxruntime::openvino_ep::ConvertAndReleaseCallbackStatus(
      Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator)));
  void* buffer = nullptr;
  size_t buffer_size = 0;
  OrtStatus* callback_status = read_func(read_state, name.c_str(), allocator, &buffer, &buffer_size);
  std::unique_ptr<void, OrtAllocatorDeleter> buffer_guard{buffer, OrtAllocatorDeleter{allocator}};
  ORT_RETURN_IF_ERROR(onnxruntime::openvino_ep::ConvertAndReleaseCallbackStatus(callback_status));
  if (buffer_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "OpenVINO EPContext read callback returned an empty payload.");
  }
  if (buffer_size > static_cast<uintmax_t>(std::numeric_limits<std::streamoff>::max())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "OpenVINO EPContext read callback payload is too large to stream.");
  }
  ORT_RETURN_IF(buffer == nullptr,
                "EPContext read callback returned a null buffer for non-empty OpenVINO context data.");
  stream = std::make_unique<CallbackBufferIStream>(std::move(buffer_guard), buffer_size);
  return onnxruntime::Status::OK();
}

}  // namespace

namespace onnxruntime {
namespace openvino_ep {

Status ConvertAndReleaseCallbackStatus(OrtStatus* status) {
  Ort::Status callback_status{status};
  if (callback_status.IsOK()) {
    return Status::OK();
  }

  return Status(common::StatusCategory::ONNXRUNTIME,
                static_cast<int>(callback_status.GetErrorCode()),
                callback_status.GetErrorMessage());
}

EPCtxHandler::EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger, std::shared_ptr<SharedContextManager> shared_context_manager)
    : openvino_sdk_version_(std::move(ov_sdk_version)), logger_(logger), shared_context_manager_(std::move(shared_context_manager)) {
  ORT_ENFORCE(shared_context_manager_ != nullptr, "SharedContextManager pointer is null in EPCtxHandler constructor.");

  epctx_model_ = Model::Create("ovep_context_model", false, logger_);
}

Status EPCtxHandler::AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                                           const std::string& graph_name,
                                           const bool embed_mode,
                                           std::string&& model_blob_str) const {
  auto& graph = epctx_model_->MainGraph();

  // Get graph inputs and outputs
  const auto& viewer_inputs = graph_viewer.GetInputs();
  const auto& viewer_outputs = graph_viewer.GetOutputs();
  std::vector<onnxruntime::NodeArg*> inputs(viewer_inputs.size()), outputs(viewer_outputs.size());
  auto transform_f = [&](const onnxruntime::NodeArg* iter) { return &graph.GetOrCreateNodeArg(iter->Name(), iter->TypeAsProto()); };
  auto fill_vectors = [transform_f](auto& src, auto& dst) {
    std::transform(src.begin(), src.end(), dst.begin(), transform_f);
  };
  fill_vectors(viewer_inputs, inputs);
  fill_vectors(viewer_outputs, outputs);

  // Create EP context node attributes
  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  node_attributes->reserve(6);
  {
    // Create EP context node attributes

    // embed mode
    auto embed_mode_attr = ONNX_NAMESPACE::AttributeProto::Create();
    embed_mode_attr->set_name(EMBED_MODE);
    embed_mode_attr->set_type(onnx::AttributeProto_AttributeType_INT);
    embed_mode_attr->set_i(embed_mode);
    node_attributes->emplace(EMBED_MODE, std::move(*embed_mode_attr));

    // main context
    auto main_graph_attr = ONNX_NAMESPACE::AttributeProto::Create();
    main_graph_attr->set_name(MAIN_CONTEXT);
    main_graph_attr->set_type(onnx::AttributeProto_AttributeType_INT);
    main_graph_attr->set_i(model_blob_str.empty() ? 0 : 1);
    node_attributes->emplace(MAIN_CONTEXT, std::move(*main_graph_attr));

    // ep context
    auto ep_cache_context_attr = ONNX_NAMESPACE::AttributeProto::Create();
    ep_cache_context_attr->set_name(EP_CACHE_CONTEXT);
    ep_cache_context_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    ep_cache_context_attr->set_s(std::move(model_blob_str));
    node_attributes->emplace(EP_CACHE_CONTEXT, std::move(*ep_cache_context_attr));

    // sdk version
    auto sdk_version_attr = ONNX_NAMESPACE::AttributeProto::Create();
    sdk_version_attr->set_name(EP_SDK_VER);
    sdk_version_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    sdk_version_attr->set_s(openvino_sdk_version_);
    node_attributes->emplace(EP_SDK_VER, std::move(*sdk_version_attr));

    // source
    auto source_attr = ONNX_NAMESPACE::AttributeProto::Create();
    source_attr->set_name(SOURCE);
    source_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    source_attr->set_s(kOpenVINOExecutionProvider);
    node_attributes->emplace(SOURCE, std::move(*source_attr));

    // partition name
    auto partition_name_attr = ONNX_NAMESPACE::AttributeProto::Create();
    partition_name_attr->set_name(PARTITION_NAME);
    partition_name_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    partition_name_attr->set_s(graph_name);
    node_attributes->emplace(PARTITION_NAME, std::move(*partition_name_attr));
  }

  // Create EP context node
  graph.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, std::move(*node_attributes), kMSDomain);

  ORT_ENFORCE(graph.Resolve().IsOK());

  return Status::OK();
}

std::unique_ptr<ModelBlobWrapper> EPCtxHandler::GetModelBlobStream(
    const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer,
    const std::string& device_type, SharedContext& shared_context, bool callback_backed) const {
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  auto node = graph_viewer.GetNode(first_index);
  ORT_ENFORCE(node != nullptr);
  auto& attrs = node->GetAttributes();

  ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) == 1);
  const auto& ep_cache_context = attrs.at(EP_CACHE_CONTEXT).s();

  ORT_ENFORCE(attrs.count(EMBED_MODE) == 1);
  bool embed_mode = static_cast<bool>(attrs.at(EMBED_MODE).i());

  if (!embed_mode && callback_backed) {
    ORT_ENFORCE(attrs.count(PARTITION_NAME) == 1, "Expected partition name for native ep context node");
    const auto& partition_name = attrs.at(PARTITION_NAME).s();
    if (device_type.find("NPU") == std::string::npos) {
      ORT_ENFORCE((attrs.count(EP_SDK_VER) == 1) && (attrs.at(EP_SDK_VER).s() == openvino_sdk_version_),
                  "EPCtx blob was exported / is compatible with OpenVINO SDK version " + attrs.at(EP_SDK_VER).s() +
                      ", but OpenVINO SDK version currently in use is " + openvino_sdk_version_);
    }
    return std::make_unique<ModelBlobWrapper>(shared_context.GetNativeBlobAsStream(partition_name),
                                              shared_context.GetNativeBlob(partition_name));
  }

  std::unique_ptr<std::istream> result;
  std::filesystem::path blob_filepath{};
  if (embed_mode) {
    result.reset((std::istream*)new std::istringstream(ep_cache_context));
  } else {
    blob_filepath = so_context_file_path;
    if (blob_filepath.empty() && !graph_viewer.ModelPath().empty()) {
      blob_filepath = graph_viewer.ModelPath();
    }
    ORT_THROW_IF_ERROR(ValidateEpContextDataPath(blob_filepath, std::filesystem::path(ep_cache_context)));
    blob_filepath = blob_filepath.parent_path() / ep_cache_context;
    ORT_ENFORCE(
        std::filesystem::exists(blob_filepath),
        "External EP context file not found: ", blob_filepath.string(), kPathResolutionGuidance);
    result.reset((std::istream*)new std::ifstream(blob_filepath, std::ios_base::binary | std::ios_base::in));
  }

  bool isXML = backend_utils::IsModelStreamXML(*result);
  if (!isXML) {
    ORT_ENFORCE(attrs.count(PARTITION_NAME) == 1, "Expected partition name for native ep context node");
    const auto& partition_name = attrs.at(PARTITION_NAME).s();

    // If the model stream is not an XML (i.e. precompiled blob), the OpenVINO SDK version that it was
    // exported with must match the version that is currently running.
    // Skip SDK version check for NPU devices as they may use different SDK versions.
    if (device_type.find("NPU") == std::string::npos) {
      ORT_ENFORCE((attrs.count(EP_SDK_VER) == 1) && (attrs.at(EP_SDK_VER).s() == openvino_sdk_version_),
                  "EPCtx blob was exported / is compatible with OpenVINO SDK version " + attrs.at(EP_SDK_VER).s() +
                      ", but OpenVINO SDK version currently in use is " + openvino_sdk_version_);
    }
    result.reset();  // Release the stream as we will get the native blob from SharedContext
    std::shared_ptr<SharedContext> filesystem_context;
    SharedContext* native_blob_context = &shared_context;
    if (!embed_mode && !callback_backed) {
      filesystem_context = shared_context_manager_->GetOrCreateSharedContext(blob_filepath);
      native_blob_context = filesystem_context.get();
    }
    return std::make_unique<ModelBlobWrapper>(native_blob_context->GetNativeBlobAsStream(partition_name),
                                              native_blob_context->GetNativeBlob(partition_name));
  }

  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";
  return std::make_unique<ModelBlobWrapper>(std::move(result), ov::Tensor(), std::move(blob_filepath));
}

bool EPCtxHandler::CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const {
  if (graph_viewer.NumberOfNodes() == 1) {
    auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
    if (auto node = graph_viewer.GetNode(first_index); (node != nullptr) && CheckForOVEPCtxNode(*node)) {
      return true;
    }
  }
  return false;
}

bool EPCtxHandler::CheckForOVEPCtxNode(const Node& node) const {
  // Check for correct Op Type, EP SOURCE, and SDK version
  if (node.OpType() == EPCONTEXT_OP) {
    auto& attrs = node.GetAttributes();
    bool result = (attrs.count(SOURCE) == 1) && (attrs.at(SOURCE).s() == kOpenVINOExecutionProvider);
    result &= attrs.count(EMBED_MODE) == 1;
    result &= attrs.count(EP_CACHE_CONTEXT) == 1;
    return result;
  }
  return false;
}

InlinedVector<const Node*> EPCtxHandler::GetEPCtxNodes() const {
  const auto& epctx_nodes{epctx_model_->MainGraph().Nodes()};
  return InlinedVector<const Node*>(epctx_nodes.begin(), epctx_nodes.end());
}

// Check if graph's only node is EPContext & EP_CACHE_CONTEXT attribute has target extension.
// @param graph_viewer: The graph to inspect.
// @param target_attr_extn: The string to search for in the EP_CACHE_CONTEXT attribute.
// @return true if the node exists, is of the correct type, and the attribute contains the extension; false otherwise.
bool EPCtxHandler::CheckEPCacheContextAttribute(const GraphViewer& graph_viewer, const std::string& target_attr_extn) const {
  // Only check if the graph has exactly one node
  if (graph_viewer.NumberOfNodes() != 1) {
    return false;
  }
  // Get the first node in topological order
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  const Node* node = graph_viewer.GetNode(first_index);
  if (!node) {
    return false;
  }
  // Check OpType and required attributes
  if (node->OpType() != EPCONTEXT_OP) {
    return false;
  }
  const auto& attrs = node->GetAttributes();
  auto it = attrs.find(EP_CACHE_CONTEXT);
  if (it != attrs.end()) {
    auto context_filename = std::filesystem::path{it->second().s()}.filename().string();
    auto target_filename = std::filesystem::path{target_attr_extn}.filename().string();
    std::transform(context_filename.begin(), context_filename.end(), context_filename.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(target_filename.begin(), target_filename.end(), target_filename.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return context_filename == target_filename;
  }
  return false;
}

std::shared_ptr<SharedContext> EPCtxHandler::Initialize(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes, const SessionContext& session_context) {
  bool has_embed_nodes = false;
  bool has_non_embed_nodes = false;
  bool has_main_context = false;
  std::unordered_set<std::string> loaded_callback_names;

  std::shared_ptr<SharedContext> shared_context{};
  for (const auto& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;

    // Only process graphs that contain ep context nodes.
    if (!CheckForOVEPCtxNodeInGraph(graph_viewer)) {
      continue;
    }

    auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
    const Node* node = graph_viewer.GetNode(first_index);
    ORT_ENFORCE(node != nullptr, "Node pointer is null despite CheckForOVEPCtxNodeInGraph returning true");

    auto& attrs = node->GetAttributes();
    ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) == 1, "EP_CACHE_CONTEXT attribute missing");

    bool embed_mode = false;
    if (attrs.count(EMBED_MODE) == 1) {
      embed_mode = static_cast<bool>(attrs.at(EMBED_MODE).i());
    }

    bool main_context = true;
    if (attrs.count(MAIN_CONTEXT) == 1) {
      main_context = static_cast<bool>(attrs.at(MAIN_CONTEXT).i());
    }

    has_main_context |= main_context;
    has_embed_nodes |= embed_mode;
    has_non_embed_nodes |= !embed_mode;

    const std::string& ep_cache_context = attrs.at(EP_CACHE_CONTEXT).s();
    if (embed_mode) {
      std::filesystem::path dummy_path{};
      shared_context = shared_context_manager_->GetOrCreateSharedContext(dummy_path);
      if (main_context) {
        ORT_ENFORCE(!ep_cache_context.empty(), "Embedded EP context is indicated but EP_CACHE_CONTEXT attribute is empty.");
        std::istringstream ss(ep_cache_context);
        shared_context->Deserialize(ss);
      }
    } else {
      const std::filesystem::path cache_context_path{ep_cache_context};
      // Compare the extension case-insensitively so that ".XML", ".Xml", etc. are also treated as XML.
      std::string extension = cache_context_path.extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      const bool is_xml = (extension == ".xml");
      const std::filesystem::path& validation_base_path = is_xml
                                                              ? session_context.GetModelPath()
                                                              : session_context.GetOutputModelPath();
      if (!is_xml) {
        if (session_context.ep_context_data_read_func != nullptr) {
          if (!shared_context) {
            shared_context = std::make_shared<SharedContext>(std::filesystem::path{});
          }
          if (loaded_callback_names.insert(ep_cache_context).second) {
            std::unique_ptr<CallbackBufferIStream> stream;
            ORT_THROW_IF_ERROR(ReadEpContextData(session_context.ep_context_data_read_func,
                                                 session_context.ep_context_data_read_state,
                                                 ep_cache_context, stream));
            shared_context->Deserialize(*stream);
            ORT_ENFORCE(shared_context->GetMetadataCopy().empty(),
                        "OpenVINO callback-backed EPContext does not support external shared weights.");
          }
        } else {
          ORT_THROW_IF_ERROR(ValidateEpContextDataPath(validation_base_path, cache_context_path));
          const std::filesystem::path ep_context_path = validation_base_path.parent_path() / cache_context_path;
          shared_context = shared_context_manager_->GetOrCreateSharedContext(ep_context_path);
          shared_context->Deserialize();
        }
      } else {
        ORT_ENFORCE(session_context.ep_context_data_read_func == nullptr,
                    "OpenVINO OVIR EPContext does not support EPContext data read callbacks.");
        ORT_THROW_IF_ERROR(ValidateEpContextDataPath(validation_base_path, cache_context_path));
      }
    }
  }

  ORT_ENFORCE(!(has_embed_nodes && has_non_embed_nodes),
              "Mixed embed and non-embed EP context nodes are not supported in a single model.");
  ORT_ENFORCE(!(has_embed_nodes && !has_main_context),
              "Expected at least one main context node when embedded EP context nodes are present.");

  // No ep context nodes found - create a shared context that can hold native blobs or shared weights.
  if (!shared_context) {
    if (session_context.so_context_enable && session_context.so_share_ep_contexts) {
      // We're creating a shared ep context model get or create the active context.
      shared_context = shared_context_manager_->GetOrCreateActiveSharedContext(session_context.GetOutputBinPath());
    } else {
      shared_context = shared_context_manager_->GetOrCreateSharedContext(session_context.GetOutputBinPath());
    }
  }

  return shared_context;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
