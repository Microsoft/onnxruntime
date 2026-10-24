// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/framework/utils.h"

#include <iomanip>

#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_frame.h"
#include "core/framework/stream_execution_context.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/run_options.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif

#ifdef ENABLE_ATEN
#include "contrib_ops/cpu/aten_ops/aten_op_executor.h"
#endif

namespace onnxruntime {
namespace utils {
void ConstructStrings(void* p_data, int64_t elements) {
  auto* ptr = static_cast<std::string*>(p_data);
  for (int64_t i = 0; i < elements; ++i) {
    new (ptr + i) std::string();
  }
}

void DestroyStrings(void* p_data, int64_t elements) {
  using string = std::string;
  auto* ptr = static_cast<std::string*>(p_data);
  for (int64_t i = 0; i < elements; i++)
    ptr[i].~string();
}

bool ProviderIsCpuBased(const IExecutionProvider& provider) {
  return provider.GetDevice().Type() == OrtDevice::CPU;
}

// Returns true if src memory can satisfy tgt's requirements without a data copy.
//
// HOST_ACCESSIBLE → DEFAULT is valid: the device can access HOST_ACCESSIBLE memory directly.
// DEFAULT → HOST_ACCESSIBLE is NOT valid: HOST_ACCESSIBLE implies CPU consumers, and DEFAULT
// memory is device-only — the CPU cannot read it.
//
// For the mixed case, src alignment must meet tgt's minimum requirement.
// Alignment 0 on tgt means "no alignment requirement". Alignment 0 on src means "unknown" and
// does not satisfy a non-zero tgt alignment requirement.
bool CanSourceSatisfyTarget(const OrtDevice& src, const OrtDevice& tgt) {
  const bool src_is_cpu_mem = src.UsesCpuMemory();
  const bool tgt_is_cpu_mem = tgt.UsesCpuMemory();

  // Identical devices are always compatible.
  if (src == tgt) {
    return true;
  }

  // Alignment 0 means "unspecified" — treat tgt as compatible with any alignment requirement.
  const bool is_alignment_satisfied = tgt.GetAlignment() == 0 ||
                                      src.GetAlignment() >= tgt.GetAlignment();

  const bool is_same_physical_device = src.Type() == tgt.Type() &&
                                       src.Vendor() == tgt.Vendor() &&
                                       src.Id() == tgt.Id();

  // Both are CPU-accessible (CPU type or HOST_ACCESSIBLE memory).
  if (src_is_cpu_mem && tgt_is_cpu_mem) {
    // CPU target can read from any CPU or HOST_ACCESSIBLE source, regardless of the source device
    if (tgt.Type() == OrtDevice::CPU) {
      return is_alignment_satisfied;
    }
    // Both are HOST_ACCESSIBLE on some device: require the same physical device.
    return is_same_physical_device && is_alignment_satisfied;
  }

  // HOST_ACCESSIBLE source can serve a DEFAULT target on the same physical device —
  // the device can DMA from HOST_ACCESSIBLE memory directly.
  // The reverse (DEFAULT → HOST_ACCESSIBLE) is unsafe: HOST_ACCESSIBLE implies CPU consumers,
  // and DEFAULT memory is device-only so the CPU cannot read it.
  if (src_is_cpu_mem && !tgt_is_cpu_mem) {
    return is_same_physical_device && is_alignment_satisfied;
  }

  return false;
}

bool IsMemcpyNode(const Node& node) {
  return node.Domain() == kOnnxDomain &&
         (node.OpType() == "MemcpyFromHost" || node.OpType() == "MemcpyToHost");
}

static common::Status AllocateHelper(const AllocatorPtr& allocator,
                                     Stream* target_stream,
                                     const OrtValue& source_mlvalue,
                                     OrtValue& target_mlvalue) {
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator.");
  }

  if (source_mlvalue.IsTensor()) {
    const Tensor& source_tensor = source_mlvalue.Get<Tensor>();
    void* p_data = nullptr;
    if (target_stream && allocator->IsStreamAware()) {
      size_t len = Tensor::CalculateTensorStorageSize(source_tensor.DataType(), source_tensor.Shape());
      p_data = allocator->AllocOnStream(len, target_stream);
      if (p_data == nullptr && len > 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Allocation failed.");
      }
    }

    if (p_data) {
      Tensor::InitOrtValue(source_tensor.DataType(),
                           source_tensor.Shape(),
                           p_data,
                           allocator, target_mlvalue);

    } else {
      Tensor::InitOrtValue(source_tensor.DataType(),
                           source_tensor.Shape(),
                           allocator, target_mlvalue);
    }
  } else if (source_mlvalue.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    const SparseTensor& source_tensor = source_mlvalue.Get<SparseTensor>();
    SparseTensor::InitOrtValue(source_tensor.DataType(), source_tensor.DenseShape(), allocator, target_mlvalue);
#endif
  } else if (source_mlvalue.IsTensorSequence()) {
    const TensorSeq& source_tensor_seq = source_mlvalue.Get<TensorSeq>();
    TensorSeq::InitOrtValue(source_tensor_seq, allocator, target_mlvalue);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported OrtValue type.");
  }
  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input && info.kci && info.kci->kernel_def->IsInputOnCpu(info.index);

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

// Populate device_fetches for the output-copy path.
// When the user pre-allocates a fetch buffer, reuse it directly as the EP's output buffer if
// the user's buffer (tgt) can satisfy the EP's output device (src) requirements — i.e.,
// CanSourceSatisfyTarget(tgt, src). This avoids a post-execution copy.
// Otherwise inserts an empty placeholder for the EP to allocate into.
static void PopulateDeviceFetches(gsl::span<const MLValueCopyInfo> fetch_copy_info,
                                  const std::vector<OrtValue>& fetches,
                                  std::vector<OrtValue>& device_fetches) {
  // fetches is empty on the subgraph path, where control-flow nodes such as Loop pass an empty fetches vector
  // and let ExecuteThePlan allocate the outputs; in that case device_fetches stays empty. A partially sized
  // fetches vector indicates a bug.
  ORT_ENFORCE(fetches.empty() || fetch_copy_info.size() == fetches.size());
  device_fetches.clear();
  device_fetches.reserve(fetches.size());
  for (size_t i = 0; i < fetches.size(); ++i) {
    const auto& src = fetch_copy_info[i].source_device;
    const auto& tgt = fetch_copy_info[i].target_device;

    // The swapped order is intentional. We're checking if a user's fetch buffer (tgt)
    // can be reused for EP's output (src) buffer — i.e. CanSourceSatisfyTarget(tgt, src).
    // Example: A user provided CPU buffer cannot satisfy a non-CPU host accessible EP output device so a copy should be
    // inserted.
    if (CanSourceSatisfyTarget(tgt, src) && fetches[i].IsAllocated()) {
      device_fetches.push_back(fetches[i]);
    } else {
      device_fetches.push_back({});
    }
  }
}

// Copy MLValue. Uses DataTransferManager for device copy if necessary. If copy_tensor_pairs/copy_sparse_pairs is provided,
// src/dst pairs that need a device copy are added to copy_pairs so copying can be batches by the DataTransferManager
// implementation for performance reasons.
static Status BatchOrCopyMLValue(const SessionState& session_state,
                                 const MLValueCopyInfo& copy_info,
                                 const OrtValue& source_mlvalue,
                                 OrtValue& target_mlvalue,
                                 Stream* stream,
#if !defined(DISABLE_SPARSE_TENSORS)
                                 std::vector<IDataTransfer::SrcDstPair>* copy_tensor_pairs = nullptr,
                                 std::vector<IDataTransfer::SparseSrcDstPair>* copy_sparse_pairs = nullptr)
#else
                                 std::vector<IDataTransfer::SrcDstPair>* copy_tensor_pairs = nullptr)
#endif
{
  // No data transfer needed if devices are identical, or the source can satisfy the target
  // (HOST_ACCESSIBLE source serving a DEFAULT target on the same physical device).
  if (CanSourceSatisfyTarget(copy_info.source_device, copy_info.target_device)) {
    target_mlvalue = source_mlvalue;
    return Status::OK();
  }

  auto allocator = session_state.GetAllocator(copy_info.target_device);
  if (!target_mlvalue.IsAllocated()) {
    ORT_ENFORCE(allocator != nullptr, "Failed to find allocator for device ", copy_info.target_device.ToString());
    ORT_RETURN_IF_ERROR(utils::AllocateHelper(allocator, stream, source_mlvalue, target_mlvalue));
  }

  if (source_mlvalue.IsTensor()) {
    const auto& source_tensor = source_mlvalue.Get<Tensor>();
    Tensor* p_output_tensor = target_mlvalue.GetMutable<Tensor>();

    if (copy_tensor_pairs != nullptr) {
      copy_tensor_pairs->push_back({source_tensor, *p_output_tensor, stream});
    } else {
      ORT_RETURN_IF_ERROR(stream ? session_state.GetDataTransferMgr().CopyTensorAsync(source_tensor, *p_output_tensor, *stream) : session_state.GetDataTransferMgr().CopyTensor(source_tensor, *p_output_tensor));
    }
  } else if (source_mlvalue.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    const auto& source_tensor = source_mlvalue.Get<SparseTensor>();
    SparseTensor* p_output_tensor = target_mlvalue.GetMutable<SparseTensor>();
    if (copy_sparse_pairs != nullptr) {
      copy_sparse_pairs->push_back({source_tensor, *p_output_tensor, 0});
    } else {
      ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensor(source_tensor, *p_output_tensor));
    }
#endif
  } else if (source_mlvalue.IsTensorSequence()) {
    const TensorSeq& source_tensor_seq = source_mlvalue.Get<TensorSeq>();
    TensorSeq& target_tensor_seq = const_cast<TensorSeq&>(target_mlvalue.Get<TensorSeq>());
    size_t size = 0;
    while ((size = target_tensor_seq.Size()) < source_tensor_seq.Size()) {
      if (0 == size) {
        target_tensor_seq.SetType(source_tensor_seq.DataType());
      }
      const Tensor& source_tensor = source_tensor_seq.Get(size);
      std::unique_ptr<Tensor> target_tensor = std::make_unique<Tensor>(source_tensor.DataType(), source_tensor.Shape(), allocator);
      target_tensor_seq.Add(std::move(*target_tensor));
    }
    const auto& data_transfer_mgr = session_state.GetDataTransferMgr();
    auto source_iter = source_tensor_seq.begin();
    auto target_iter = target_tensor_seq.begin();

    while (source_iter != source_tensor_seq.end() &&
           target_iter != target_tensor_seq.end()) {
      if (copy_tensor_pairs != nullptr) {
        copy_tensor_pairs->push_back({source_iter->Get<Tensor>(), *target_iter->GetMutable<Tensor>(), stream});
      } else {
        if (stream)
          ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensorAsync(source_iter->Get<Tensor>(), *target_iter->GetMutable<Tensor>(), *stream));
        else
          ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensor(source_iter->Get<Tensor>(), *target_iter->GetMutable<Tensor>()));
      }
      ++source_iter;
      ++target_iter;
    }  // while
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported OrtValue type to copy between device.");
  }

  return Status::OK();
}  // namespace utils

static bool HaveCpuExecutionProvidersOnly(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (!ProviderIsCpuBased(*execution_provider)) {
      return false;
    }
  }

  return true;
}

static const OrtDevice& FindDeviceForValue(const OrtValueNameIdxMap& map,
                                           const SequentialExecutionPlan& plan,
                                           std::string_view name) {
  int idx = -1;
  auto status = map.GetIdx(name, idx);
  ORT_THROW_IF_ERROR(status);

  const auto& location = plan.GetLocation(idx);
  return location;
}

const OrtDevice& FindDeviceForValue(const SessionState& session_state, std::string_view name) {
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  return FindDeviceForValue(session_state.GetOrtValueNameIdxMap(), *exec_plan_ptr, name);
}

// get the target device info for the node consuming each input provided in the feeds.
// source_device info is not known until runtime
static common::Status CalculateStaticCopyInfoForFeed(const SessionState& session_state,
                                                     const std::string& input_name,
                                                     MLValueCopyInfo& copy_info) {
  InlinedVector<SessionState::NodeInfo> node_info_vec;
#ifdef ENABLE_TRAINING
  if (session_state.GetInputNodeInfo(input_name, node_info_vec) == Status::OK()) {
#else
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));
#endif
    const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine

    if (node_info.p_node == nullptr) {
      // ignore dummy entry for an input that we didn't find a use of in the graph.
      return Status::OK();
    }

    copy_info.target_device = *node_info.device;
    copy_info.unique_stream_index_consumes_it = node_info.stream_index;
    ORT_RETURN_IF(node_info.stream_index < 0, "node_info.stream_index < 0");
    for (size_t i = 1; i < node_info_vec.size(); i++) {
      ORT_RETURN_IF(node_info_vec[i].stream_index < 0, "node_info_vec[i].stream_index < 0");
      if (node_info_vec[i].stream_index != node_info.stream_index) {
        copy_info.unique_stream_index_consumes_it = -1;
        break;
      }
    }

#ifdef ENABLE_TRAINING
  } else {
    // This input might be for an intermediate tensor for partial graph execution.
    const auto* exec_plan = session_state.GetExecutionPlan();
    const auto& name_to_id = session_state.GetOrtValueNameIdxMap();
    int index;
    ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_name, index));
    const auto& device = exec_plan->GetLocation(index);
    copy_info.target_device = device;
  }
#endif

  return Status::OK();
}

static common::Status CalculateStaticCopyInfoForFeeds(const SessionState& session_state,
                                                      gsl::span<const std::string> feed_names,
                                                      std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = feed_names.size(); idx < end; ++idx) {
    ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, feed_names[idx], copy_info[idx]));
  }

  return Status::OK();
}

// get the source device info for the node producing each output that we will return in the fetches.
// target device info is not known until runtime.
static common::Status CalculateStaticCopyInfoForFetches(const SessionState& session_state,
                                                        gsl::span<const std::string> fetch_names,
                                                        std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = fetch_names.size(); idx < end; ++idx) {
    const std::string& output_name = fetch_names[idx];

    const auto& info = FindDeviceForValue(session_state, output_name);
    copy_info[idx].source_device = info;
  }

  return Status::OK();
}

common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager) {
  // if we only have CPU based EPs we can skip all the copy logic
  auto cpu_only = HaveCpuExecutionProvidersOnly(session_state.GetExecutionProviders());

  if (cpu_only) {
    feeds_fetches_manager.SetDeviceCopyChecks(DeviceCopyCheck::NoCopy, DeviceCopyCheck::NoCopy);
  } else {
    // setup all the static info about where the graph inputs and outputs are located
    auto info = feeds_fetches_manager.GetFeedsFetchesInfo();
    auto& feed_copy_info = feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo();
    auto& fetch_copy_info = feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo();
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFeeds(session_state, info.feed_names, feed_copy_info));
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFetches(session_state, info.output_names, fetch_copy_info));
  }

  return Status::OK();
}

// update the allocation_provider in the copy info based on the actual feeds
static bool FinalizeCopyInfoForFeeds(gsl::span<const OrtDevice> feed_locations,
                                     std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(feed_locations.size() == copy_info.size());
  bool copy_needed = false;

  for (size_t i = 0, end = feed_locations.size(); i < end; ++i) {
    copy_info[i].source_device = feed_locations[i];

    if (!CanSourceSatisfyTarget(copy_info[i].source_device, copy_info[i].target_device)) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

static bool FinalizeCopyInfoForFetches(gsl::span<const OrtDevice* const>& fetch_alloc_info,
                                       std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(fetch_alloc_info.size() == copy_info.size());
  bool copy_needed = false;

  auto num_outputs = fetch_alloc_info.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtDevice* alloc_info = fetch_alloc_info[i];

    if (alloc_info != nullptr) {
      copy_info[i].target_device = *alloc_info;
    }

    if (!CanSourceSatisfyTarget(copy_info[i].source_device, copy_info[i].target_device)) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

// Finalize the copy info using the OrtDevice and OrtMemoryInfo for the feeds and fetches
// This can be used by control flow nodes prior to the execution of the overall graph.
void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtDevice> feed_locations,
                               gsl::span<const OrtDevice* const> fetch_alloc_info) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return;

  bool need_copy = FinalizeCopyInfoForFeeds(feed_locations, feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo());
  DeviceCopyCheck input_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  need_copy = FinalizeCopyInfoForFetches(fetch_alloc_info, feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo());
  DeviceCopyCheck output_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  feeds_fetches_manager.SetDeviceCopyChecks(input_copy, output_copy);
}

// Finalize the copy info using the OrtValue instances for the feeds and fetches
static void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                                      gsl::span<const OrtValue> feeds,
                                      std::vector<OrtValue>& fetches) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return;

  auto num_inputs = feeds.size();
  auto num_outputs = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();

  std::vector<OrtDevice> feed_locations(num_inputs);
  std::vector<const OrtDevice*> fetch_alloc_info(num_outputs, nullptr);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& feed = feeds[i];
    if (feed.IsTensor()) {
      feed_locations[i] = feed.Get<Tensor>().Location().device;
    } else if (feed.IsTensorSequence()) {
      const auto& tensor_seq = feed.Get<TensorSeq>();
      if (tensor_seq.Size() != std::size_t{0}) {
        feed_locations[i] = tensor_seq.Get(0).Location().device;
      }
    } else if (feed.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
      feed_locations[i] = feed.Get<SparseTensor>().Location().device;
#endif
    }
  }

  // create default instances if needed
  fetches.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& fetch = fetches[i];
    if (fetch.IsAllocated()) {
      if (fetch.IsTensor()) {
        fetch_alloc_info[i] = &fetch.Get<Tensor>().Location().device;
      } else if (fetch.IsTensorSequence()) {
        const auto& tensor_seq = fetch.Get<TensorSeq>();
        if (tensor_seq.Size() != std::size_t{0}) {
          fetch_alloc_info[i] = &tensor_seq.Get(0).Location().device;
        }
      } else if (fetch.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
        fetch_alloc_info[i] = &fetch.Get<SparseTensor>().Location().device;
#endif
      }
    }
  }

  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feed_locations, fetch_alloc_info);
}

static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              gsl::span<const OrtValue> orig_feeds,
                                              std::vector<OrtValue>& new_feeds,
#ifdef ORT_ENABLE_STREAM
                                              DeviceStreamCollection* device_stream_collection,
#endif
                                              gsl::span<const MLValueCopyInfo> copy_info) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);
  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
#if !defined(DISABLE_SPARSE_TENSORS)
  std::vector<IDataTransfer::SparseSrcDstPair> batched_sparse_data_transfers;
#endif

  std::unordered_set<Stream*> stream_to_flush;
  for (size_t idx = 0; idx < num_feeds; ++idx) {
    Stream* copy_this_feed = nullptr;
#ifdef ORT_ENABLE_STREAM
    if (device_stream_collection) {
      if (copy_info[idx].unique_stream_index_consumes_it < 0) {
        for (size_t i = 0; i < device_stream_collection->NumStreams(); i++) {
          Stream* stream = device_stream_collection->GetStream(i);
          if (stream && stream->GetDevice().Type() == copy_info[idx].target_device.Type()) {
            copy_this_feed = stream;
            stream_to_flush.insert(stream);
            break;
          }
        }
      } else {
        copy_this_feed = device_stream_collection->GetStream(copy_info[idx].unique_stream_index_consumes_it);
      }
    }
#endif
#if !defined(DISABLE_SPARSE_TENSORS)
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], orig_feeds[idx], new_feeds[idx],
                                           copy_this_feed,
                                           &batched_data_transfers, &batched_sparse_data_transfers));
#else
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], orig_feeds[idx], new_feeds[idx],
                                           copy_this_feed,
                                           &batched_data_transfers));
#endif
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (!batched_sparse_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensors(batched_sparse_data_transfers));
  }
#endif

  // flush the stream to make sure the inputs are ready before launch the inference.
  // TODO: this sync is because the graph inputs can be consumed by multiple stream,
  // but we can only place the MemCpyAsync on one of the stream. Ideally we should make
  // other stream wait on the event of the memory copy stream, instead of host sync stream.
  for (const auto& stream : stream_to_flush) stream->Flush();
  return Status::OK();
}

#ifdef ORT_ENABLE_STREAM
static void UpdateWithParentStream(DeviceStreamCollection& device_stream_collection,
                                   Stream* parent_stream) {
  if (parent_stream) {
    // TODO: in theory, we should make current subgraph's stream depends on parent stream.
    // but in current code structure, it causing issues with the resource sharing and stream
    // lifetime. it also may cause additional cost of stream sync for single stream case.
    // In first phase, let's just put all the subgraph execution on the parent stream.
    for (size_t i = 0; i < device_stream_collection.NumStreams(); ++i) {
      auto* stream = device_stream_collection.GetStream(i);
      if (stream) {
        // if current logic stream is not on the same EP instance as parent stream
        // and the EP instance does have async streams (not EP like CPU)
        // throw error as we don't have the code to setup the dependency at this moment.
        if (stream->GetDevice() != parent_stream->GetDevice()) {
          ORT_THROW("Subgraph has nodes running on device: ", stream->GetDevice().Type(),
                    " while parent graph node running on device: ", parent_stream->GetDevice().Type(),
                    ", this is not supported yet.");
        }
        device_stream_collection.SetDeviceStream(i, parent_stream);
      }
    }
  }
}
#endif

// public method to do a single copy. used by external partners
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  if (!orig_mlvalue.IsTensor() && !orig_mlvalue.IsSparseTensor()) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  MLValueCopyInfo copy_info;
  // Sets copy_info.target_device.
  ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, input_name, copy_info));
#if !defined(DISABLE_SPARSE_TENSORS)
  copy_info.source_device = (orig_mlvalue.IsTensor())
                                ? orig_mlvalue.Get<Tensor>().Location().device
                                : orig_mlvalue.Get<SparseTensor>().Location().device;
#else
  copy_info.source_device = orig_mlvalue.Get<Tensor>().Location().device;
#endif

  Stream* device_stream = nullptr;
#ifdef ORT_ENABLE_STREAM
  DeviceStreamCollectionHolder device_stream_collection_holder(&session_state);
  if (device_stream_collection_holder.p_ != nullptr) {
    DeviceStreamCollection* device_stream_collection = device_stream_collection_holder.p_.get();
    size_t num_streams = device_stream_collection->NumStreams();
    for (size_t i = 0; i < num_streams; i++) {
      Stream* stream = device_stream_collection->GetStream(i);
      if (stream && !stream->GetDevice().UsesCpuMemory()) {
        device_stream = stream;
        break;
      }
    }
  }
#endif

  // copy_info.target_device is not set leaving to be equal to CPU.
  return BatchOrCopyMLValue(session_state, copy_info, orig_mlvalue, new_mlvalue, device_stream);
}

static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               gsl::span<const OrtValue> fetches,
                                               std::vector<OrtValue>& user_fetches,
                                               gsl::span<const MLValueCopyInfo> copy_info,
                                               gsl::span<Stream* const> fetch_streams) {
  auto num_outputs = fetches.size();
  user_fetches.resize(num_outputs);

  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
#if !defined(DISABLE_SPARSE_TENSORS)
  std::vector<IDataTransfer::SparseSrcDstPair> batched_sparse_data_transfers;
#endif

  for (size_t idx = 0; idx < num_outputs; ++idx) {
#if !defined(DISABLE_SPARSE_TENSORS)
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], fetches[idx], user_fetches[idx], fetch_streams[idx],
                                           &batched_data_transfers, &batched_sparse_data_transfers));
#else
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], fetches[idx], user_fetches[idx], fetch_streams[idx],
                                           &batched_data_transfers));
#endif
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (!batched_sparse_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensors(batched_sparse_data_transfers));
  }
#endif

  return Status::OK();
}

static common::Status
ExecuteGraphImpl(const SessionState& session_state,
                 const FeedsFetchesManager& feeds_fetches_manager,
                 gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                 const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                 ExecutionMode execution_mode, const bool& terminate_flag,
                 const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                 DeviceStreamCollection* device_stream_collection,
#endif
                 const bool only_execute_path_to_fetches = false,
                 Stream* parent_stream = nullptr,
                 profiling::Profiler* run_profiler = nullptr) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();
#ifdef ORT_ENABLE_STREAM
  auto* execution_plan = session_state.GetExecutionPlan();
  if (device_stream_collection)
    UpdateWithParentStream(*device_stream_collection, parent_stream);
#else
  ORT_UNUSED_PARAMETER(parent_stream);
#endif

  bool is_subgraph = session_state.GetGraphViewer().ParentNode() != nullptr;
  // in following two cases, we execute the workload in main thread:
  // 1. execution mode is sequential.
  // 2. execute a subgraph. Because in current implementation, the execute of subgraph is launched through parent kernel.
  //    the parent kernel will occupy a thread in thread pool. if we use multiple threads to execute subgraph, it may cause
  //    deadlock when we reach the limitation of thread pool.
  bool single_thread_mode = execution_mode == ExecutionMode::ORT_SEQUENTIAL || is_subgraph;

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    auto status = (ExecuteThePlan(session_state,
                                  feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                  feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators,
                                  logger,
#ifdef ORT_ENABLE_STREAM
                                  device_stream_collection,
#endif
                                  terminate_flag,
                                  only_execute_path_to_fetches,
                                  // single thread mode
                                  single_thread_mode,
                                  run_profiler));
    ORT_RETURN_IF_ERROR(status);
  } else {
    auto feeds_to_use = feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      auto status = CopyInputsAcrossDevices(session_state, feeds, device_feeds,
#ifdef ORT_ENABLE_STREAM
                                            device_stream_collection,
#endif
                                            feed_copy_info);
      ORT_RETURN_IF_ERROR(status);
      feeds_to_use = device_feeds;
    }

    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();
    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      PopulateDeviceFetches(fetch_copy_info, fetches, device_fetches);
      p_fetches = &device_fetches;
    }

    // no device copies are needed so simple execute
    auto status = (ExecuteThePlan(session_state,
                                  feeds_fetches_info.feeds_mlvalue_idxs, feeds_to_use,
                                  feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                  logger,
#ifdef ORT_ENABLE_STREAM
                                  device_stream_collection,
#endif
                                  terminate_flag,
                                  only_execute_path_to_fetches,
                                  single_thread_mode,
                                  run_profiler));
    ORT_RETURN_IF_ERROR(status);
    InlinedVector<Stream*> fetches_streams;
    fetches_streams.reserve(feeds_fetches_info.fetches_mlvalue_idxs.size());
#ifdef ORT_ENABLE_STREAM
    auto& value_to_stream_map = execution_plan->value_to_stream_map;
    for (auto fetch_idx : feeds_fetches_info.fetches_mlvalue_idxs) {
      auto it = value_to_stream_map.find(fetch_idx);
      if (it != value_to_stream_map.end()) {
        fetches_streams.push_back(device_stream_collection ? device_stream_collection->GetStream(it->second) : nullptr);
      } else {
        // for subgraph, it is possible the graph is empty,
        // the fetches are come from parent graph.
        fetches_streams.push_back(parent_stream);
      }
    }
#else
    for (size_t i = 0; i < feeds_fetches_info.fetches_mlvalue_idxs.size(); ++i) {
      fetches_streams.push_back(nullptr);
    }
#endif

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info, fetches_streams));
    }
  }
  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag,
                            const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                            DeviceStreamCollectionHolder& device_stream_collection_holder,
#endif
                            bool only_execute_path_to_fetches,
                            Stream* parent_stream,
                            profiling::Profiler* run_profiler) {
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(session_state, feeds_fetches_manager));

  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);
#ifdef ORT_ENABLE_STREAM
  DeviceStreamCollection* device_stream_collection = device_stream_collection_holder.p_.get();
  auto retval = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, {},
                                 execution_mode, terminate_flag, logger,
                                 device_stream_collection,
                                 only_execute_path_to_fetches,
                                 parent_stream,
                                 run_profiler);
  return retval;
#else
  return ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, {},
                          execution_mode, terminate_flag, logger,
                          only_execute_path_to_fetches,
                          parent_stream,
                          run_profiler);
#endif
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const RunOptions& run_options,
#ifdef ORT_ENABLE_STREAM
                            DeviceStreamCollectionHolder& device_stream_collection_holder,
#endif
                            const logging::Logger& logger,
                            profiling::Profiler* run_profiler) {
  return ExecuteGraph(session_state, feeds_fetches_manager, feeds, fetches,
                      execution_mode, run_options.terminate, logger,
#ifdef ORT_ENABLE_STREAM
                      device_stream_collection_holder,
#endif
                      run_options.only_execute_path_to_fetches,
                      nullptr /* parent_stream */,
                      run_profiler);
}

#ifdef ENABLE_TRAINING
common::Status ExecutePartialGraphImpl(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                       std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                       const logging::Logger& logger, PartialGraphExecutionState& state,
                                       const OrtValueCachePtr& cache, const bool& terminate_flag,
                                       DeviceStreamCollection* device_stream_collection,
                                       int32_t partial_graph_index,
                                       Stream* parent_stream) {
  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();
  // always use single_stream mode for training, to have a stable execution order
  bool single_thread_mode = true;

  auto* execution_plan = session_state.GetExecutionPlan();
  if (device_stream_collection)
    UpdateWithParentStream(*device_stream_collection, parent_stream);

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(PartialExecuteThePlan(session_state,
                                              feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                              feeds_fetches_info.fetches_mlvalue_idxs, fetches, {},
                                              logger,
                                              device_stream_collection,
                                              terminate_flag,
                                              // single thread mode
                                              single_thread_mode,
                                              state,
                                              cache,
                                              partial_graph_index));
  } else {
    auto p_feeds = feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state, feeds, device_feeds, device_stream_collection, feed_copy_info));
      p_feeds = device_feeds;
    }

    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      PopulateDeviceFetches(fetch_copy_info, fetches, device_fetches);
      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(PartialExecuteThePlan(session_state,
                                              feeds_fetches_info.feeds_mlvalue_idxs, p_feeds,
                                              feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, {},
                                              logger,
                                              device_stream_collection,
                                              terminate_flag,
                                              // single thread mode
                                              single_thread_mode,
                                              state,
                                              cache,
                                              partial_graph_index));

    InlinedVector<Stream*> fetches_streams;
    fetches_streams.reserve(feeds_fetches_info.fetches_mlvalue_idxs.size());
    auto& value_to_stream_map = execution_plan->value_to_stream_map;
    for (auto fetch_idx : feeds_fetches_info.fetches_mlvalue_idxs) {
      auto it = value_to_stream_map.find(fetch_idx);
      if (it != value_to_stream_map.end()) {
        fetches_streams.push_back(device_stream_collection ? device_stream_collection->GetStream(it->second) : nullptr);
      } else {
        // for subgraph, it is possible the graph is empty,
        // the fetches are come from parent graph.
        fetches_streams.push_back(parent_stream);
      }
    }

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info, fetches_streams));
    }
    // training don't want to flush the stream
  }

  return Status::OK();
}

common::Status ExecutePartialGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                   std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                   const logging::Logger& logger, PartialGraphExecutionState& state,
                                   const OrtValueCachePtr& cache, const bool& terminate_flag,
                                   int32_t partial_graph_index,
                                   Stream* parent_stream) {
  DeviceStreamCollection* device_stream_collection = state.GetDeviceStreamCollection(session_state);
  auto retval = ExecutePartialGraphImpl(session_state, feeds_fetches_manager, feeds, fetches,
                                        logger, state, cache, terminate_flag, device_stream_collection,
                                        partial_graph_index, parent_stream);
  if (device_stream_collection)
    ORT_CHECK_AND_SET_RETVAL(device_stream_collection->CleanUp(false));
  return retval;
}
#endif

common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger,
                               Stream* parent_stream,
                               bool sync_subgraph_fetches,
                               profiling::Profiler* run_profiler) {
#ifdef ORT_ENABLE_STREAM
  DeviceStreamCollectionHolder device_stream_collection_holder(&session_state);
  DeviceStreamCollection* device_stream_collection = device_stream_collection_holder.p_.get();

  auto retval = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, fetch_allocators,
                                 execution_mode, terminate_flag, logger, device_stream_collection, false, parent_stream,
                                 run_profiler);
  if (device_stream_collection)
    ORT_CHECK_AND_SET_RETVAL(device_stream_collection->CleanUp(false));
#else
  auto retval = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, fetch_allocators,
                                 execution_mode, terminate_flag, logger, false, parent_stream,
                                 run_profiler);
#endif
  if (retval.IsOK() && sync_subgraph_fetches && parent_stream) {
    parent_stream->Flush();
  }
  return retval;
}

int32_t ONNXTensorElementDataTypeToProtoTensorType(ONNXTensorElementDataType onnx_enum) {
  switch (onnx_enum) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return onnx::TensorProto_DataType::TensorProto_DataType_STRING;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return onnx::TensorProto_DataType::TensorProto_DataType_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT6E2M3:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT6E2M3;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT6E3M2:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT6E3M2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
      assert(false);
      return onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED;
  }
}

#ifdef ENABLE_TRAINING
// Needed only when NCCL kernels are enabled.
common::Status VerifyInputTensorsAllocatedContiguously(OpKernelContext* context) {
  const Tensor* prev_input = context->Input<Tensor>(0);
  for (int i = 1; i < context->InputCount(); i++) {
    const Tensor* curr_input = context->Input<Tensor>(i);

    ORT_ENFORCE(prev_input->Shape().Size() >= 0);

    const void* curr_address = curr_input->DataRaw();
    const void* prev_address = prev_input->DataRaw();
    const void* prev_end_address = reinterpret_cast<const char*>(prev_address) + prev_input->SizeInBytes();

    void* aligned_address = const_cast<void*>(prev_end_address);
    size_t dummy_space = kAllocAlignment * 2;
    std::align(kAllocAlignment, 1, aligned_address, dummy_space);

    if (!(curr_address == prev_end_address || curr_address == aligned_address)) {
      const std::string node = context->GetNodeName().empty() ? context->GetOpType() : context->GetNodeName();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Contiguous memory checking failed on node ", node, ": ",
                             "input #", i - 1, " address is ", prev_address, " and #bytes = ", prev_input->SizeInBytes(),
                             ", input #", i, " address is ", curr_address);
    }

    prev_input = curr_input;
  }
  return Status::OK();
}
#endif

bool IsInputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index) {
  if (p_kci && p_kci->kernel_def->IsInputOnCpu(index)) {
    return true;
  }

#ifdef ENABLE_ATEN
  // For ATen node, we assume that all tensor inputs are on device, all non-tensor inputs are on CPU,
  // except those specified in attribute cpu_input_args;
  if (node.GetExecutionProviderType() == kCudaExecutionProvider && node.OpType() == "ATen" &&
      node.Domain() == kPytorchAtenDomain) {
    const auto& attrs = node.GetAttributes();
    if (auto entry = attrs.find("cpu_input_args"); entry != attrs.end()) {
      const auto& attr = entry->second;
      if (utils::HasInts(attr) && std::any_of(attr.ints().cbegin(), attr.ints().cend(),
                                              [index](int64_t arg) { return static_cast<int64_t>(index) == arg; })) {
        return true;
      }
    }

    ORT_ENFORCE(utils::HasString(attrs.at("operator")));
    std::string op_name = attrs.at("operator").s();
    std::string overload_name = "";
    if (attrs.find("overload_name") != attrs.end() && utils::HasString(attrs.at("overload_name"))) {
      overload_name = attrs.at("overload_name").s();
    }

    return !contrib::aten_ops::ATenOperatorExecutor::Instance().IsTensorArgument(op_name, overload_name, index, true);
  }
#else
  ORT_UNUSED_PARAMETER(node);
#endif

  return false;
}

bool IsOutputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index) {
  if (p_kci && p_kci->kernel_def->IsOutputOnCpu(index)) {
    return true;
  }

#ifdef ENABLE_ATEN
  // For ATen node, we assume that all tensor outputs are on device, all non-tensor outputs are on CPU,
  // except those specified in attribute cpu_output_args;
  if (node.GetExecutionProviderType() == kCudaExecutionProvider && node.OpType() == "ATen" &&
      node.Domain() == kPytorchAtenDomain) {
    const auto& attrs = node.GetAttributes();
    if (auto entry = attrs.find("cpu_output_args"); entry != attrs.end()) {
      const auto& attr = entry->second;
      if (utils::HasInts(attr) && std::any_of(attr.ints().cbegin(), attr.ints().cend(),
                                              [index](int64_t arg) { return static_cast<int64_t>(index) == arg; })) {
        return true;
      }
    }

    ORT_ENFORCE(utils::HasString(attrs.at("operator")));
    std::string op_name = attrs.at("operator").s();
    std::string overload_name = "";
    if (attrs.find("overload_name") != attrs.end() && utils::HasString(attrs.at("overload_name"))) {
      overload_name = attrs.at("overload_name").s();
    }

    return !contrib::aten_ops::ATenOperatorExecutor::Instance().IsTensorArgument(op_name, overload_name, index, false);
  }
#else
  ORT_UNUSED_PARAMETER(node);
#endif

  return false;
}

}  // namespace utils
}  // namespace onnxruntime
