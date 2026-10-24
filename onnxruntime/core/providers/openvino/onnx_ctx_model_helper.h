// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/execution_provider.h"
#include "ov_shared_context.h"
#include "contexts.h"

namespace onnxruntime {
namespace openvino_ep {

Status ConvertAndReleaseCallbackStatus(OrtStatus* status);

struct ModelBlobWrapper {
  ModelBlobWrapper(std::unique_ptr<std::istream> stream, const ov::Tensor& tensor,
                   std::filesystem::path model_path = {})
      : stream_(std::move(stream)), tensor_(tensor), model_path_(std::move(model_path)) {}
  std::unique_ptr<std::istream> stream_;
  ov::Tensor tensor_;  // May be empty if model blob is provided as stream only.
  std::filesystem::path model_path_;
};

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const char EPCONTEXT_OP[] = "EPContext";
static const char EMBED_MODE[] = "embed_mode";
static const char MAIN_CONTEXT[] = "main_context";
static const char PARTITION_NAME[] = "partition_name";
static const char EP_CACHE_CONTEXT[] = "ep_cache_context";
static const char EP_SDK_VER[] = "ep_sdk_version";
static const char SOURCE[] = "source";

class EPCtxHandler {
 public:
  EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger, std::shared_ptr<SharedContextManager> shared_context_manager);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  bool CheckForOVEPCtxNodeInGraph(const GraphViewer& subgraph_view) const;
  bool CheckForOVEPCtxNode(const Node& node) const;
  Status AddOVEPCtxNodeToGraph(const GraphViewer& subgraph_view,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<ModelBlobWrapper> GetModelBlobStream(const std::filesystem::path& so_context_file_path,
                                                       const GraphViewer& subgraph_view,
                                                       const std::string& device_type,
                                                       SharedContext& shared_context,
                                                       bool callback_backed) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;
  bool CheckEPCacheContextAttribute(const GraphViewer& subgraph_view, const std::string& target_attr_extn) const;
  std::shared_ptr<SharedContext> Initialize(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes, const SessionContext& session_context);

 private:
  const std::string openvino_sdk_version_;
  std::unique_ptr<Model> epctx_model_;
  const logging::Logger& logger_;
  std::shared_ptr<SharedContextManager> shared_context_manager_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
