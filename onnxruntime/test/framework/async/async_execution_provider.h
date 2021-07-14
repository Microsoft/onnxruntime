// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: use provider_api.h
#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

#include "async_execution_stream.h"
#include "async_execution_event.h"

namespace onnxruntime {

constexpr const char* kAsyncExecutionProvider = "AsyncExecutionProvider";

class AsyncExecutionProvider : public IExecutionProvider {
 public:
  explicit AsyncExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  Status
  Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
          std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  AsyncExecutionStream& GetAsyncStream(int64_t i) const {
    ORT_ENFORCE(gsl::narrow<size_t>(i) < streams_.size());
    return *streams_[i];
  }

  AsyncExecutionEvent& GetAsyncEvent(int64_t i) const {
    ORT_ENFORCE(gsl::narrow<size_t>(i) < events_.size());
    return *events_[i];
  }

  std::vector<std::unique_ptr<AsyncExecutionStream>> streams_;
  std::vector<std::unique_ptr<AsyncExecutionEvent>> events_;

  enum {
    NumStreams = 3,
    NumEvents = 10,
  };

  enum : int64_t {
    // note that custom MemType > 0
    MemType_Stream2 = 1,
    MemType_MaxPlusOne
  };
  AllocatorPtr custom_allocators_[MemType_MaxPlusOne - 1];

 public:
  static const std::string FusedNodeDomain;
  static const std::string NodeAttr_Stream;
  static const std::string NodeAttr_WaitEvents;
  static const std::string NodeAttr_RecordEvent;
  static const std::string NodeAttr_PriorSyncStream;
  static const std::string NodeAttr_PosteriorSyncStream;

  enum : int64_t {
    EmptyStream = -1,
    EmptyEvent = -1,
  };
};

}  // namespace onnxruntime