// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/inference_session.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "async_execution_provider.h"
#include "async_kernel.h"

namespace onnxruntime {

const std::string AsyncExecutionProvider::FusedNodeDomain("AsyncEP");

const std::string AsyncExecutionProvider::NodeAttr_Stream("stream");
const std::string AsyncExecutionProvider::NodeAttr_WaitEvents("wait_events");
const std::string AsyncExecutionProvider::NodeAttr_RecordEvent("record_event");
const std::string AsyncExecutionProvider::NodeAttr_PriorSyncStream("prior_sync_stream");
const std::string AsyncExecutionProvider::NodeAttr_PosteriorSyncStream("posterior_sync_stream");

static void SetIntAttr(IndexedSubGraph::MetaDef* meta_def,
                       const std::string& attr_name,
                       int64_t value) {
  ONNX_NAMESPACE::AttributeProto ap;
  ap.set_name(attr_name);
  ap.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  ap.set_i(value);
  meta_def->attributes[attr_name] = ap;
}

static void SetIntsAttr(IndexedSubGraph::MetaDef* meta_def,
                        const std::string& attr_name,
                        const std::vector<int64_t>& values) {
  ONNX_NAMESPACE::AttributeProto ap;
  ap.set_name(attr_name);
  ap.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  for (auto v : values) {
    ap.add_ints(v);
  }
  meta_def->attributes[attr_name] = ap;
}

static void SetAsyncInfo(IndexedSubGraph::MetaDef* meta_def,
                         int64_t stream_id,
                         const std::vector<int64_t>& wait_events = {},
                         int64_t record_event = AsyncExecutionProvider::EmptyEvent,
                         int64_t prior_sync_stream_id = AsyncExecutionProvider::EmptyStream,
                         int64_t posterior_sync_stream_id = AsyncExecutionProvider::EmptyStream) {
  SetIntAttr(meta_def, AsyncExecutionProvider::NodeAttr_Stream, stream_id);

  if (!wait_events.empty()) {
    SetIntsAttr(meta_def, AsyncExecutionProvider::NodeAttr_WaitEvents, wait_events);
  }

  if (record_event != AsyncExecutionProvider::EmptyEvent) {
    SetIntAttr(meta_def, AsyncExecutionProvider::NodeAttr_RecordEvent, record_event);
  }

  if (prior_sync_stream_id != AsyncExecutionProvider::EmptyStream) {
    SetIntAttr(meta_def, AsyncExecutionProvider::NodeAttr_PriorSyncStream, prior_sync_stream_id);
  }

  if (posterior_sync_stream_id != AsyncExecutionProvider::EmptyStream) {
    SetIntAttr(meta_def, AsyncExecutionProvider::NodeAttr_PosteriorSyncStream, posterior_sync_stream_id);
  }
}

// mimic cudaStreamWaitEvent
static void AsyncStreamWaitEvent(AsyncExecutionStream& stream, AsyncExecutionEvent& event) {
  // wait event in stream
  stream.Launch([&event]() {
    event.Wait();
  });
}

// mimic cudaEventRecord
static void AsyncEventRecord(AsyncExecutionEvent& event, AsyncExecutionStream& stream) {
  // reset event in dispatcher thread
  event.Reset();
  // signal event in stream
  stream.Launch([&event]() {
    event.Signal();
  });
}

AsyncExecutionProvider::AsyncExecutionProvider() : IExecutionProvider{kAsyncExecutionProvider} {
  // the default allocator, note we could use arena allocator if needed
  AllocatorCreationInfo device_info{
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo("Async", OrtAllocatorType::OrtDeviceAllocator));
      }};
  InsertAllocator(device_info.device_alloc_factory(/*device_id*/ 0));

  // some more allocators, for async streams, note we could use arena allocator if needed
  // using a separate allocator would simplify synchronization,
  // at the cost of bigger memory footprint since the memory between allocators are not shared
  // we may replace per-stream allocator with BFC arena with stream tracking,
  // similar to RAPIDS Memory Management
  // note that custom MemType is not supported by IExecutionProvider::GetAllocator
  // so EP needs to override the GetAllocator, and manage the allocator outside of registry
  for (int64_t i = 1; i < MemType_MaxPlusOne; ++i) {
    custom_allocators_[i - 1] = onnxruntime::make_unique<CPUAllocator>(
        OrtMemoryInfo("AsyncCustom", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(),
                      /*device_id*/ 0, /*mem_type*/ (OrtMemType)i));
  }

  for (uint32_t i = 0; i < NumStreams; ++i) {
    streams_.push_back(onnxruntime::make_unique<AsyncExecutionStream>(std::to_string(i)));
  }
  for (uint32_t i = 0; i < NumEvents; ++i) {
    events_.push_back(onnxruntime::make_unique<AsyncExecutionEvent>());
  }
}

AllocatorPtr AsyncExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type <= 0) {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
  ORT_ENFORCE(gsl::narrow<int64_t>(mem_type) < MemType_MaxPlusOne);
  return custom_allocators_[mem_type - 1];  // custom MemType is 1-based, while the array is 0-based
}

std::vector<std::unique_ptr<ComputeCapability>>
AsyncExecutionProvider::GetCapability(const GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node : graph.Nodes()) {
    // each node is a subgraph
    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node.Index());
    auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = node.Name();
    meta_def->domain = "Async";
    node.ForEachDef([&meta_def](const NodeArg& def, bool is_input) {
      if (is_input) {
        meta_def->inputs.push_back(def.Name());
      }
    });
    node.ForEachDef([&meta_def](const NodeArg& def, bool is_input) {
      if (!is_input) {
        meta_def->outputs.push_back(def.Name());
      }
    });
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

    // use fused node's attributes to carry async execution info
    // hard code stream assignment and event to wait/signal for testing purpose only
    // actual logic should be based on graph partition and data dependency
    // note: a fused function always record events after compute, and wait events before compute
    const auto& op_type = node.OpType();

    if (op_type == "Add") {
      // runs in stream 0
      // record event 0 after kernel compute done in stream 0
      SetAsyncInfo(meta_def.get(), /*stream*/ 0, /*wait_events*/ {}, /*record_event*/ 0);
      // specify output to be MemType_Stream2
      SetIntsAttr(meta_def.get(), KernelDefBuilder::NodeAttr_OutputMemType, {MemType_Stream2});
    } else if (op_type == "Mul") {
      // runs in stream 1
      // record event 1 after kernel compute done in stream 1
      SetAsyncInfo(meta_def.get(), /*stream*/ 1, /*wait_events*/ {}, /*record_event*/ 1);
      // specify output to be MemType_Stream2
      SetIntsAttr(meta_def.get(), KernelDefBuilder::NodeAttr_OutputMemType, {MemType_Stream2});
    } else if (op_type == "Sub") {
      // runs in stream 2
      // wait for event 0 and 1
      // do not record event, but synchronize dispatcher thread to stream 2 after done
      SetAsyncInfo(meta_def.get(), /*stream*/ 2, /*wait_events*/ {0, 1}, /*record_event*/ EmptyEvent, /*prior_sync*/ EmptyStream, /*post_sync*/ 2);
    } else {
      ORT_THROW("Unexpected OpType: ", op_type);
    }

    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

Status
AsyncExecutionProvider::Compile(const std::vector<Node*>& fused_nodes,
                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* node : fused_nodes) {
    NodeComputeInfo info;

    // Create state function
    info.create_state_func =
        [&, node](ComputeContext* /*ctx*/, FunctionState* state) {
          std::unique_ptr<AsyncKernel> s =
              onnxruntime::make_unique<AsyncKernel>(*node);

          *state = s.release();
          return 0;
        };

    // Release state function
    info.release_state_func =
        [](FunctionState state) {
          if (state)
            delete static_cast<AsyncKernel*>(state);
        };

    // Compute function
    // note Launch() run shape inferenece/allocate output in dispatcher thread, and queues compute kernel to stream
    info.compute_func =
        [this](FunctionState state, const OrtCustomOpApi*, OrtKernelContext* op_kernel_context) {
          AsyncKernel* s = reinterpret_cast<AsyncKernel*>(state);
          const auto& cfg = s->GetAsyncExecConfig();

          // synchronize stream prior to launch
          if (cfg.prior_sync_stream_id != EmptyStream) {
            GetAsyncStream(cfg.prior_sync_stream_id).Synchronize();
          }

          auto& stream = GetAsyncStream(cfg.stream_id);

          // wait events on stream
          for (auto e : cfg.wait_events) {
            AsyncStreamWaitEvent(stream, GetAsyncEvent(e));
          }

          // launch kernel on stream
          auto ret = s->Launch(reinterpret_cast<OpKernelContext*>(op_kernel_context), stream);

          // record event on stream
          if (cfg.record_event != EmptyEvent) {
            AsyncEventRecord(GetAsyncEvent(cfg.record_event), stream);
          }

          // synchronize stream post launch
          if (cfg.posterior_sync_stream_id != EmptyStream) {
            GetAsyncStream(cfg.posterior_sync_stream_id).Synchronize();
          }

          return ret;
        };

    node_compute_funcs.push_back(info);
  }
  return Status::OK();
}

};  // namespace onnxruntime
