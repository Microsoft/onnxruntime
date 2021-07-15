// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/framework/utils.h"

#include <iomanip>

#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/TensorSeq.h"
#ifdef ENABLE_TRAINING
#include "core/framework/orttraining_partial_executor.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_executor.h"
#endif

namespace ONNX_NAMESPACE {
std::ostream& operator<<(std::ostream& out, const TensorShapeProto& shape_proto) {
  std::string result;
  result.reserve(128);

  result.append("{");
  bool first = true;
  for (auto& dim : shape_proto.dim()) {
    if (!first) {
      result.append(",");
    }

    if (onnxruntime::utils::HasDimValue(dim))
      result.append(std::to_string(dim.dim_value()));
    else if (onnxruntime::utils::HasDimParam(dim))
      result.append(dim.dim_param());

    first = false;
  }
  result.append("}");

  return (out << result);
}

std::ostream& operator<<(std::ostream& out, const TensorProto& tensor_proto) {
  std::string result;
  result.reserve(128);

  result.append("{");
  bool first = true;
  for (auto& dim : tensor_proto.dims()) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return (out << result);
}
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace utils {
void* DefaultAlloc(size_t size) {
  if (size <= 0) return nullptr;
  void* p;
  size_t alignment = MlasGetPreferredBufferAlignment();
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void DefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

bool ProviderIsCpuBased(const std::string& provider_type) {
  return provider_type == onnxruntime::kCpuExecutionProvider ||
         provider_type == onnxruntime::kDnnlExecutionProvider ||
         provider_type == onnxruntime::kNupharExecutionProvider ||
         provider_type == onnxruntime::kVitisAIExecutionProvider ||
         provider_type == onnxruntime::kOpenVINOExecutionProvider ||
         provider_type == onnxruntime::kNnapiExecutionProvider ||
         provider_type == onnxruntime::kAclExecutionProvider ||
         provider_type == onnxruntime::kArmNNExecutionProvider ||
         provider_type == onnxruntime::kRknpuExecutionProvider ||
         provider_type == onnxruntime::kCoreMLExecutionProvider ||
         provider_type == onnxruntime::utils::kInternalTestingExecutionProvider;
}

static common::Status AllocateHelper(const AllocatorPtr& allocator,
                                     const OrtValue& source_mlvalue,
                                     OrtValue& target_mlvalue) {
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator.");
  }
  if (source_mlvalue.IsTensor()) {

    const Tensor& source_tensor = source_mlvalue.Get<Tensor>();
    std::unique_ptr<Tensor> target_tensor = std::make_unique<Tensor>(source_tensor.DataType(),
                                                                     source_tensor.Shape(),
                                                                     allocator);
    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    target_mlvalue.Init(target_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  } else if (source_mlvalue.IsTensorSequence()) {
    const TensorSeq& source_tensor_seq = source_mlvalue.Get<TensorSeq>();
    auto target_tensor_seq = std::make_unique<TensorSeq>(source_tensor_seq.DataType());
    std::vector<Tensor> tensors;
    for (auto iter = source_tensor_seq.begin(); iter != source_tensor_seq.end(); ++iter) {
      tensors.emplace_back(iter->DataType(), onnxruntime::TensorShape(iter->Shape()), allocator);
    }
    target_tensor_seq->SetElements(std::move(tensors)); 
    auto ml_tensor_seq = DataTypeImpl::GetType<TensorSeq>();
    target_mlvalue.Init(target_tensor_seq.release(), ml_tensor_seq, ml_tensor_seq->GetDeleteFunc());
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

// Copy MLValue. Uses DataTransferManager for device copy if necessary. If copy_pairs is provided,
// src/dst pairs that need a device copy are added to copy_pairs so copying can be batches by the DataTransferManager
// implementation for performance reasons.
static Status BatchOrCopyMLValue(const SessionState& session_state,
                                 const MLValueCopyInfo& copy_info,
                                 const OrtValue& source_mlvalue,
                                 OrtValue& target_mlvalue,
                                 std::vector<IDataTransfer::SrcDstPair>* copy_pairs = nullptr) {
  // same device so direct copy
  if (copy_info.source_device == copy_info.target_device) {
    target_mlvalue = source_mlvalue;
    return Status::OK();
  }

  auto allocator = session_state.GetAllocator(copy_info.target_device);
  if (!target_mlvalue.IsAllocated()) {
    ORT_ENFORCE(allocator != nullptr, "Failed to find allocator for device ", copy_info.target_device.ToString());
    ORT_RETURN_IF_ERROR(utils::AllocateHelper(allocator, source_mlvalue, target_mlvalue));
  }

  if (source_mlvalue.IsTensor()) {
    const Tensor& source_tensor = source_mlvalue.Get<Tensor>();
    Tensor& target_tensor = *target_mlvalue.GetMutable<Tensor>();
    if (copy_pairs != nullptr) {
      copy_pairs->push_back({source_tensor, target_tensor, 0});
    } else {
      ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensor(source_tensor, target_tensor));
    }
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
    auto source_iter = source_tensor_seq.begin();
    auto target_iter = target_tensor_seq.begin();
    while (source_iter != source_tensor_seq.end() &&
           target_iter != target_tensor_seq.end()) {
      if (copy_pairs != nullptr) {
        copy_pairs->push_back({*source_iter, const_cast<Tensor&>(*target_iter), 0});
      } else {
        ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensor(*source_iter, const_cast<Tensor&>(*target_iter)));
      }
      ++source_iter;
      ++target_iter;
    }//while
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported OrtValue type to copy between device.");
  }
  return Status::OK();
}

static bool HaveCpuExecutionProvidersOnly(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (!ProviderIsCpuBased(execution_provider->Type())) {
      return false;
    }
  }

  return true;
}

static const OrtMemoryInfo& FindMemoryInfoForValue(const OrtValueNameIdxMap& map,
                                                   const SequentialExecutionPlan& plan,
                                                   const std::string& name) {
  int idx = -1;
  auto status = map.GetIdx(name, idx);
  ORT_THROW_IF_ERROR(status);

  const auto& location = plan.GetLocation(idx);
  return location;
}

const OrtMemoryInfo& FindMemoryInfoForValue(const SessionState& session_state,
                                            const std::string& name) {
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  return FindMemoryInfoForValue(session_state.GetOrtValueNameIdxMap(), *exec_plan_ptr, name);
}

// get the target device info for the node consuming each input provided in the feeds.
// source_device info is not known until runtime
static common::Status CalculateStaticCopyInfoForFeed(const SessionState& session_state,
                                                     const std::string& input_name,
                                                     MLValueCopyInfo& copy_info) {
#ifdef ENABLE_TRAINING
  std::vector<SessionState::NodeInfo> node_info_vec;
  if (session_state.GetInputNodeInfo(input_name, node_info_vec) == Status::OK()) {
    const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine

    if (node_info.p_node == nullptr) {
      // ignore dummy entry for an input that we didn't find a use of in the graph.
      return Status::OK();
    }

    copy_info.target_device = *node_info.device;

  } else {
    // This input might be for an intermediate tensor for partial graph execution.
    const auto* exec_plan = session_state.GetExecutionPlan();
    const auto& name_to_id = session_state.GetOrtValueNameIdxMap();
    int index;
    ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_name, index));
    const auto& device = exec_plan->GetLocation(index).device;
    copy_info.target_device = device;
  }

  return Status::OK();
#else
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));
  const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine

  if (node_info.p_node == nullptr) {
    // ignore dummy entry for an input that we didn't find a use of in the graph.
    return Status::OK();
  }

  copy_info.target_device = *node_info.device;

  return Status::OK();
#endif
}

static common::Status CalculateStaticCopyInfoForFeeds(const SessionState& session_state,
                                                      const std::vector<std::string>& feed_names,
                                                      std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = feed_names.size(); idx < end; ++idx) {
    ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, feed_names[idx], copy_info[idx]));
  }

  return Status::OK();
}

// get the source device info for the node producing each output that we will return in the fetches.
// target device info is not known until runtime.
static common::Status CalculateStaticCopyInfoForFetches(const SessionState& session_state,
                                                        const std::vector<std::string>& fetch_names,
                                                        std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = fetch_names.size(); idx < end; ++idx) {
    const std::string& output_name = fetch_names[idx];

    const auto& info = FindMemoryInfoForValue(session_state, output_name);
    copy_info[idx].source_device = info.device;

    // If for some reason using just the device from the allocation plan isn't enough, the following
    // would use the NodeInfo from the node producing the output
    //
    //std::vector<SessionState::NodeInfo> node_info_vec;
    //auto status = session_state.GetOutputNodeInfo(output_name, node_info_vec);
    //if (status.IsOK()) {
    //  const auto& node_info = node_info_vec.front();  // only one entry as only one node can produce a given output
    //  copy_info[idx].source_device = *node_info.device;
    //} else {
    //  // edge case where an initializer directly provides output so no NodeInfo involved
    //  const auto& info = FindMemoryInfoForValue(session_state, output_name);
    //  copy_info[idx].source_device = info.device;
    //}
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
static bool FinalizeCopyInfoForFeeds(const std::vector<OrtDevice>& feed_locations,
                                     std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(feed_locations.size() == copy_info.size());
  bool copy_needed = false;

  for (size_t i = 0, end = feed_locations.size(); i < end; ++i) {
    copy_info[i].source_device = feed_locations[i];

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

static bool FinalizeCopyInfoForFetches(const std::vector<const OrtMemoryInfo*>& fetch_alloc_info,
                                       std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(fetch_alloc_info.size() == copy_info.size());
  bool copy_needed = false;

  auto num_outputs = fetch_alloc_info.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtMemoryInfo* alloc_info = fetch_alloc_info[i];

    if (alloc_info != nullptr) {
      copy_info[i].target_device = alloc_info->device;
    }

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

// Finalize the copy info using the OrtDevice and OrtMemoryInfo for the feeds and fetches
// This can be used by control flow nodes prior to the execution of the overall graph.
void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtDevice>& feed_locations,
                               const std::vector<const OrtMemoryInfo*>& fetch_alloc_info) {
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
                                      const std::vector<OrtValue>& feeds,
                                      std::vector<OrtValue>& fetches) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return;

  auto num_inputs = feeds.size();
  auto num_outputs = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();

  std::vector<OrtDevice> feed_locations(num_inputs);
  std::vector<const OrtMemoryInfo*> fetch_alloc_info(num_outputs, nullptr);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& feed = feeds[i];
    if (feed.IsTensor()) {
      feed_locations[i] = feed.Get<Tensor>().Location().device;
    }
  }

  // create default instances if needed
  fetches.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& fetch = fetches[i];
    if (fetch.IsAllocated() && fetch.IsTensor()) {
      fetch_alloc_info[i] = &fetch.Get<Tensor>().Location();
    }
  }

  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feed_locations, fetch_alloc_info);
}

static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              const std::vector<OrtValue>& orig_feeds,
                                              std::vector<OrtValue>& new_feeds,
                                              const std::vector<MLValueCopyInfo>& copy_info) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);
  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
  batched_data_transfers.reserve(num_feeds);

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], orig_feeds[idx], new_feeds[idx],
                                           &batched_data_transfers));
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

  return Status::OK();
}

// public method to do a single copy. used by external partners
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  if (!orig_mlvalue.IsTensor()) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  MLValueCopyInfo copy_info;
  ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, input_name, copy_info));
  copy_info.source_device = orig_mlvalue.Get<Tensor>().Location().device;

  return BatchOrCopyMLValue(session_state, copy_info, orig_mlvalue, new_mlvalue);
}

static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               const std::vector<OrtValue>& fetches,
                                               std::vector<OrtValue>& user_fetches,
                                               const std::vector<MLValueCopyInfo>& copy_info) {
  auto num_outputs = fetches.size();
  user_fetches.resize(num_outputs);

  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
  batched_data_transfers.reserve(num_outputs);

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], fetches[idx], user_fetches[idx],
                                           &batched_data_transfers));
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

  return Status::OK();
}

static common::Status ExecuteGraphImpl(const SessionState& session_state,
                                       const FeedsFetchesManager& feeds_fetches_manager,
                                       const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                       const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                       ExecutionMode execution_mode, const bool& terminate_flag,
                                       const logging::Logger& logger, const bool only_execute_path_to_fetches = false) {
  std::unique_ptr<IExecutor> p_exec;
  if (execution_mode == ExecutionMode::ORT_SEQUENTIAL) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag, only_execute_path_to_fetches));
  } else if (execution_mode == ExecutionMode::ORT_PARALLEL) {
    auto* p_inter_op_thread_pool = session_state.GetInterOpThreadPool();
    if (!p_inter_op_thread_pool) {
      LOGS(logger, WARNING) << "Only one thread was configured for parallel execution. Hence will use sequential execution.";
      p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag, only_execute_path_to_fetches));
    } else {
      p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
    }
  }

  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators,
                                        logger));
  } else {
    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state, feeds, device_feeds, feed_copy_info));
      p_feeds = &device_feeds;
    }

    auto num_outputs = fetches.size();
    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      // need intermediate fetches. use pre-allocated fetches where possible.
      device_fetches.reserve(num_outputs);

      for (size_t i = 0; i < num_outputs; ++i) {
        if (fetch_copy_info[i].source_device == fetch_copy_info[i].target_device && fetches[i].IsAllocated()) {
          device_fetches.push_back(fetches[i]);
        } else {
          // use temporary value
          device_fetches.push_back({});
        }
      }

      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag,
                            const logging::Logger& logger, bool only_execute_path_to_fetches) {
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(session_state, feeds_fetches_manager));

  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);

  auto status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, {},
                                 execution_mode, terminate_flag, logger, only_execute_path_to_fetches);

  return status;
}

#ifdef ENABLE_TRAINING
common::Status ExecutePartialGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                   const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                   const logging::Logger& logger, PartialGraphExecutionState& state) {

  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);
  PartialExecutor executor{state};
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(executor.Execute(session_state,
                                         feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                         feeds_fetches_info.fetches_mlvalue_idxs, fetches, {},
                                         logger));
  } else {
    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state, feeds, device_feeds, feed_copy_info));
      p_feeds = &device_feeds;
    }

    auto num_outputs = fetches.size();
    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      // need intermediate fetches. use pre-allocated fetches where possible.
      device_fetches.reserve(num_outputs);

      for (size_t i = 0; i < num_outputs; ++i) {
        if (fetch_copy_info[i].source_device == fetch_copy_info[i].target_device && fetches[i].IsAllocated()) {
          device_fetches.push_back(fetches[i]);
        } else {
          // use temporary value
          device_fetches.push_back({});
        }
      }

      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(executor.Execute(session_state,
                                         feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                         feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, {},
                                         logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info));
    }
  }

  return Status::OK();
}
#endif

common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger) {
  auto status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, fetch_allocators,
                                 execution_mode, terminate_flag, logger);
  return status;
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

#ifdef ENABLE_TRAINING
  if (node.GetExecutionProviderType() == kCudaExecutionProvider && node.OpType() == "ATenOp") {
    const auto& attrs = node.GetAttributes();
    ORT_ENFORCE(utils::HasString(attrs.at("name")));
    std::string op_name = attrs.at("name").s();
    std::string overload_name = "";
    if (attrs.find("overload_name") != attrs.end() && utils::HasString(attrs.at("overload_name"))) {
      overload_name = attrs.at("overload_name").s();
    }

    return !contrib::aten_ops::ATenOperatorExecutor::Instance().IsTensorArgument(op_name, overload_name, index);
  }
#else
  ORT_UNUSED_PARAMETER(node);
#endif

  return false;
}

}  // namespace utils
}  // namespace onnxruntime
