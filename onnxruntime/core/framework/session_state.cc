// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <atomic>
#include <mutex>
#include <sstream>

#if !defined(ORT_MINIMAL_BUILD)
#include <filesystem>
#include <fstream>
#include "core/common/parse_string.h"
#endif

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/allocator.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/session_state_utils.h"
#include "core/framework/utils.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/max_shape_inference.h"
#include "core/framework/moe_offload_plan.h"
#include "core/framework/node_shape_resolver.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
#ifdef ORT_ENABLE_STREAM
static inline std::string GetWaitKey(const OrtDevice::DeviceType notification_device_type,
                                     const OrtDevice::DeviceType executor_device_type) {
  return std::to_string(notification_device_type) + ":" + std::to_string(executor_device_type);
}

static const std::shared_ptr<const int>& GetDeviceStreamPoolThreadToken() {
  // Use a thread-lifetime token instead of std::thread::id so a bucket from a
  // dead thread cannot be accidentally reused if the runtime later reuses the
  // same thread id value for a different thread.
  thread_local const auto thread_token = std::make_shared<const int>(0);
  return thread_token;
}

class StreamCommandHandleRegistryImpl : public IStreamCommandHandleRegistry {
 public:
  // Wait is a little special as we need to consider the source stream the notification generated,
  // and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  WaitNotificationFn GetWaitHandle(const OrtDevice& notification_owner_device,
                                   const OrtDevice& executor_device) const override {
    auto notification_owner_device_type = notification_owner_device.UsesCpuMemory() ? OrtDevice::CPU
                                                                                    : notification_owner_device.Type();
    auto executor_device_type = executor_device.UsesCpuMemory() ? OrtDevice::CPU : executor_device.Type();

    auto it = notification_wait_map_.find(GetWaitKey(notification_owner_device_type, executor_device_type));
    return it == notification_wait_map_.end() ? nullptr : it->second;
  }

  CreateStreamFn GetCreateStreamFn(const OrtDevice::DeviceType device_type) const override {
    auto it = create_stream_map_.find(device_type);
    return it == create_stream_map_.end() ? nullptr : it->second;
  }

  void RegisterWaitFn(const OrtDevice::DeviceType notification_device_type,
                      const OrtDevice::DeviceType device_type,
                      WaitNotificationFn fn) override {
    notification_wait_map_.insert({GetWaitKey(notification_device_type, device_type), fn});
  }

  void RegisterCreateStreamFn(const OrtDevice::DeviceType device_type, CreateStreamFn f) override {
    create_stream_map_.insert({device_type, f});
  }

  void RegisterSetDeviceFn(const OrtDevice::DeviceType device_type, SetDeviceFn f) override {
    set_device_map_.insert({device_type, f});
  }

  std::optional<SetDeviceFn> GetSetDeviceFn(const OrtDevice::DeviceType device_type) const override {
    auto it = set_device_map_.find(device_type);
    if (it != set_device_map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  StreamCommandHandleRegistryImpl() = default;

 private:
  InlinedHashMap<std::string, WaitNotificationFn> notification_wait_map_;
  InlinedHashMap<OrtDevice::DeviceType, CreateStreamFn> create_stream_map_;
  InlinedHashMap<OrtDevice::DeviceType, SetDeviceFn> set_device_map_;
};
#endif

SessionState::SessionState(Graph& graph,
                           const ExecutionProviders& execution_providers,
                           concurrency::ThreadPool* thread_pool,
                           concurrency::ThreadPool* inter_op_thread_pool,
                           const DataTransferManager& data_transfer_mgr,
                           const ExternalDataLoaderManager& external_data_loader_mgr,
                           const logging::Logger& logger,
                           profiling::Profiler& profiler,
                           const SessionOptions& sess_options,
                           PrepackedWeightsContainer* prepacked_weights_container,
                           AllocatorMap* parent_allocators,
                           AllocatorMap* parent_initializer_allocators)
    : graph_(graph),
      execution_providers_(execution_providers),
      logger_(logger),
      profiler_(profiler),
      thread_pool_(thread_pool),
      inter_op_thread_pool_(inter_op_thread_pool),
      data_transfer_mgr_(data_transfer_mgr),
      external_data_loader_mgr_(external_data_loader_mgr),
      sess_options_(sess_options),
      prepacked_weights_container_(prepacked_weights_container)
#ifdef ORT_ENABLE_STREAM
      ,
      stream_handles_registry_(std::make_unique<StreamCommandHandleRegistryImpl>())
#endif
{
  enable_mem_pattern_ = sess_options_.enable_mem_pattern &&
                        sess_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL;
  if (parent_allocators) {
    allocators_ = parent_allocators;
    initializer_allocators_ = parent_initializer_allocators;
  } else {
    allocators_unique_ptr_ = std::make_unique<AllocatorMap>();
    allocators_ = allocators_unique_ptr_.get();

    initializer_allocators_unique_ptr_ = std::make_unique<AllocatorMap>();
    initializer_allocators_ = initializer_allocators_unique_ptr_.get();

    // The allocator registration rule:
    // Each location (OrtDevice) will only have 1 allocator used for whole session.
    // The EP which is registered first will have higher priority.
    // Allocators with a OrtAllocatorType of OrtReadOnlyAllocator go in the initializer allocators
    for (auto& ep : execution_providers_) {
      auto allocators = ep->CreatePreferredAllocators();
      for (auto& alloc : allocators) {
        if (alloc->Info().alloc_type == OrtReadOnlyAllocator) {
          initializer_allocators_->insert({alloc->Info().device, alloc});  // DON'T overwrite existing key
        } else {
          allocators_->insert({alloc->Info().device, alloc});  // DON'T overwrite existing key
        }
      }
    }
  }
}

AllocatorPtr SessionState::GetAllocator(const OrtMemoryInfo& location) const noexcept {
  return GetAllocator(location.device);
}

AllocatorPtr SessionState::GetAllocator(const OrtDevice& device) const noexcept {
  auto it = allocators_->find(device);
  if (it != allocators_->end()) {
    return it->second;
  }

  return nullptr;
}

AllocatorPtr SessionState::GetInitializerAllocator(const OrtDevice& device) const noexcept {
  auto it = initializer_allocators_->find(device);
  if (it != initializer_allocators_->end()) {
    return it->second;
  }

  return GetAllocator(device);
}

void SessionState::UpdateAllocatorsWithEnvAllocators(const std::vector<AllocatorPtr>& env_allocators) {
  for (const auto& env_alloc : env_allocators) {
    if (env_alloc->Info().alloc_type == OrtReadOnlyAllocator) {
      (*initializer_allocators_)[env_alloc->Info().device] = env_alloc;
    } else {
      (*allocators_)[env_alloc->Info().device] = env_alloc;
    }
  }
}

void SessionState::CreateGraphInfo(bool save_prepacked_on) {
  graph_.ConstructPrepackedSharedContainerAndSetMode(save_prepacked_on);

  graph_viewer_.emplace(graph_);
  // use graph_viewer_ to initialize ort_value_name_idx_map_
  LOGS(logger_, VERBOSE) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_viewer_->GetInputsIncludingInitializers()) {
    idx = ort_value_name_idx_map_.Add(input_def->Name());
    VLOGS(logger_, 1) << "Added graph_viewer_ input with name: " << input_def->Name()
                      << " to OrtValueIndex with index: " << idx;
  }

  for (auto& node : graph_viewer_->Nodes()) {
    // build the OrtValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added implicit input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        ort_value_name_idx_map_.Add(output_def->Name());
        VLOGS(logger_, 1) << "Added output argument with name: " << output_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }
  }

  // allocate OrtValue for graph outputs when coming from initializers
  for (const auto& output : graph_viewer_->GetOutputs()) {
    if (output->Exists()) {
      idx = ort_value_name_idx_map_.Add(output->Name());
      VLOGS(logger_, 1) << "Added graph output with name: " << output->Name()
                        << " to OrtValueIndex with index: " << idx;
    }
  }

  LOGS(logger_, VERBOSE) << "Done saving OrtValue mappings.";
}

Status SessionState::PopulateKernelCreateInfo(const KernelRegistryManager& kernel_registry_manager,
                                              bool saving_ort_format) {
  for (auto& node : graph_.Nodes()) {
    const KernelCreateInfo* kci = nullptr;
    auto status = kernel_registry_manager.SearchKernelRegistry(node, logger_, &kci);

    // There are two cases where we allow fallback to CPU EP kernels:
    //
    // 1. if we didn't find the kernel and are saving to ORT format an EP that compiles nodes is enabled.
    // in that case we assigned the node to that EP but do not compile it into a fused node.
    // this keeps the original node and prevents level 2 and level 3 optimizers from modifying it.
    // we now revert to the CPU EP kernel as a fallback.
    // at runtime when the model is loaded in a minimal build, the compiling EP will replace this node if possible.
    // if that's not possible for some reason we can fallback to the CPU EP implementation.
    //
    // 2. If the node is a memcpy node.
    // EPs may provide their own memcpy kernels. The CPU EP provides a generic version to fall back to if the EP does
    // not provide one.
    const bool allow_cpu_ep_kernel_fallback = saving_ort_format || utils::IsMemcpyNode(node);

    if (!status.IsOK() && allow_cpu_ep_kernel_fallback) {
      node.SetExecutionProviderType(kCpuExecutionProvider);
      status = kernel_registry_manager.SearchKernelRegistry(node, logger_, &kci);
    }

    ORT_RETURN_IF_ERROR(status);

    ORT_IGNORE_RETURN_VALUE(
        kernel_create_info_map_.insert({node.Index(), gsl::not_null<const KernelCreateInfo*>(kci)}));
  }

  for (const auto& entry : subgraph_session_states_) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      ORT_RETURN_IF_ERROR(subgraph_session_state.PopulateKernelCreateInfo(kernel_registry_manager,
                                                                          saving_ort_format));
    }
  }

  return Status::OK();
}

const KernelCreateInfo& SessionState::GetNodeKernelCreateInfo(NodeIndex node_index) const {
  auto entry = kernel_create_info_map_.find(node_index);
  // invalid node index or FinalizeSessionState should have been called. Either way it's an internal logic error
  ORT_ENFORCE(entry != kernel_create_info_map_.cend());

  return *entry->second;
}

Status SessionState::CreateKernels(const KernelRegistryManager& kernel_registry_manager) {
  const auto& nodes = graph_viewer_->Nodes();
  if (!nodes.empty()) {
    size_t max_nodeid = 0;
    for (const auto& node : nodes) {
      max_nodeid = std::max(max_nodeid, node.Index());
    }
    session_kernels_.clear();
    session_kernels_.resize(max_nodeid + 1);
    for (const auto& node : nodes) {
      // construct and save the kernels
      const KernelCreateInfo& kci = GetNodeKernelCreateInfo(node.Index());

      // the execution provider was required to be valid to find the KernelCreateInfo so we don't need to check it here
      onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();
      const IExecutionProvider& exec_provider = *execution_providers_.Get(exec_provider_name);

      // assumes vector is already resize()'ed to the number of nodes in the graph
      ORT_RETURN_IF_ERROR(kernel_registry_manager.CreateKernel(node, exec_provider, *this, kci, session_kernels_[node.Index()]));
    }
  }
  node_index_info_.emplace(*graph_viewer_, ort_value_name_idx_map_);
  return Status::OK();
}

void SessionState::PruneRemovableAttributes() {
  InlinedVector<std::string> removable_attributes;
  for (size_t i = 0; i < session_kernels_.size(); ++i) {
    if (session_kernels_[i].get() == nullptr)
      continue;
    auto status = session_kernels_[i].get()->GetRemovableAttributes(removable_attributes);
    if (!status.IsOK()) {
      const Node& node_const = session_kernels_[i].get()->Node();
      LOGS(logger_, WARNING) << "failed at retrieving the removable attributes"
                             << "for node '" << node_const.Name() << "' ('" << node_const.OpType() << "').";
      continue;
    }
    if (removable_attributes.empty())
      continue;
    auto index = session_kernels_[i].get()->Node().Index();
    Node* node = graph_.GetNode(index);
    int n_removed = node->PruneRemovableAttributes(removable_attributes);
    if (n_removed == 0)
      continue;
    LOGS(logger_, INFO) << "removed " << n_removed << " removable attributes "
                        << "for node '" << node->Name() << "' ('" << node->OpType() << "'), "
                        << "among attributes: " << [removable_attributes]() -> std::string {
      std::ostringstream os;
      for (auto it = removable_attributes.cbegin(); it != removable_attributes.cend(); ++it) {
        if (it != removable_attributes.cbegin())
          os << ", ";
        os << *it;
      }
      return os.str();
    }() << ".";
  }
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const {
  if (!p_seq_exec_plan_.has_value()) {
    return nullptr;
  }
  return &p_seq_exec_plan_.value();
}

const std::vector<AllocPlanPerValue>& SessionState::GetPerValueAllocPlan() const {
  return p_seq_exec_plan_->allocation_plan;
}

Status SessionState::AddInitializedTensor(int ort_value_index, const OrtValue& ort_value,
                                          bool constant, bool sparse) {
  auto p = initialized_tensors_.insert({ort_value_index, ort_value});
  if (!p.second)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "duplicated ort_value index:", ort_value_index,
                           ". Do you have duplicated calls to SessionState::AddInitializedTensor function?");

  if (constant) {
    constant_initialized_tensors_.insert({ort_value_index, ort_value});
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (sparse) {
    sparse_initialized_tensors_.insert(ort_value_index);
  }
#else
  ORT_UNUSED_PARAMETER(sparse);
#endif

  return Status::OK();
}

const std::unordered_map<int, OrtValue>& SessionState::GetInitializedTensors() const { return initialized_tensors_; }

const std::unordered_map<int, OrtValue>& SessionState::GetConstantInitializedTensors() const {
  return constant_initialized_tensors_;
}

const std::unordered_map<int, OrtValue>& SessionState::GetConstantInitializedTensorsForKernelCreation() const {
  return outer_scope_augmented_map_built_ ? outer_scope_augmented_constant_tensors_
                                          : constant_initialized_tensors_;
}

const PrepackedWeightsForGraph& onnxruntime::SessionState::GetPrepackedIniitializersForGraph() const {
  return graph_.GetPrepacked();
}

#if !defined(DISABLE_SPARSE_TENSORS)
bool SessionState::IsSparseInitializer(int ort_value_index) const {
  return sparse_initialized_tensors_.count(ort_value_index) > 0;
}
#endif

#ifdef ENABLE_TRAINING
Status SessionState::GetInitializedTensors(
    const std::unordered_set<std::string>& interested_weights,
    bool allow_missing_weights, NameMLValMap& retrieved_weights) const {
  NameMLValMap result;
  result.reserve(interested_weights.size());
  for (const auto& weight_name : interested_weights) {
    int idx;
    const auto status = GetOrtValueNameIdxMap().GetIdx(weight_name, idx);
    if (!status.IsOK()) {
      ORT_RETURN_IF_NOT(
          allow_missing_weights,
          "Failed to get OrtValue index from name: ", status.ErrorMessage());
      continue;
    }
    if (initialized_tensors_.find(idx) != initialized_tensors_.end()) {
      result.emplace(weight_name, initialized_tensors_.at(idx));
    } else {
      ORT_RETURN_IF_NOT(
          allow_missing_weights,
          "Failed to get initializer with name: ", weight_name, " and index:", idx);
      continue;
    }
  }
  retrieved_weights = std::move(result);
  return Status::OK();
}

NameMLValMap SessionState::GetInitializedTensors(const std::unordered_set<std::string>& interested_weights) const {
  NameMLValMap result;
  const auto status = GetInitializedTensors(interested_weights, true, result);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return result;
}
#endif

void SessionState::CleanInitializedTensorsFromGraph() {
  graph_.CleanAllInitializedTensors();
}

static Status KernelUseSharedPrePackedBuffers(OpKernel& kernel, int input_idx,
                                              const PrePackedWeights& prepacked_weights,
                                              const std::string& node_name) {
  const size_t num_buffers = prepacked_weights.buffers_.size();
  assert(prepacked_weights.buffer_sizes_.size() == num_buffers);

  std::vector<BufferUniquePtr> shared_prepacked_buffers;
  std::vector<size_t> shared_prepacked_buffer_sizes;
  shared_prepacked_buffers.reserve(num_buffers);
  shared_prepacked_buffer_sizes.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; i++) {
    // BufferDeleter is nullptr because the kernel should not delete the shared buffer - it can only use it
    shared_prepacked_buffers.emplace_back(prepacked_weights.buffers_[i].get(), BufferDeleter(nullptr));
    shared_prepacked_buffer_sizes.push_back(prepacked_weights.buffer_sizes_[i]);
  }

  bool used_shared_buffers = false;
  ORT_RETURN_IF_ERROR(kernel.UseSharedPrePackedBuffers(shared_prepacked_buffers, shared_prepacked_buffer_sizes,
                                                       input_idx, used_shared_buffers));

  // BUG CHECK: Ensure that the kernel used the provided shared buffers
  // Mostly a debug check to ensure that the kernel has an overridden implementation of the
  // base UseSharedPrePackedBuffers() which is basically a no-op.
  if (!used_shared_buffers)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The kernel corresponding to the node ", node_name,
                           " doesn't have an implementation that can consume provided pre-packed weights");

  return Status::OK();
}

static std::string GenerateKeyForPrepackedWeightsMap(const std::string& op_type,
                                                     const PrePackedWeights& pre_packed_weights) {
  std::ostringstream ss_1;
  ss_1 << op_type;
  ss_1 << "+";
  ss_1 << std::to_string(pre_packed_weights.GetHash());

  return ss_1.str();
}

Status SessionState::PrepackConstantInitializedTensors(
    InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map) {
  // Guards the bookkeeping that mutates state shared across nodes (the constant-initializer/
  // use-count maps, the per-graph PrepackedWeightsForGraph container, and the prepack counter)
  // when the per-node work below runs concurrently across the intra-op thread pool. It is only
  // acquired on the (default) path that doesn't use cross-session pre-packed-weight sharing --
  // that path is already fully serialized by the mutex_ lock further down. kernel->PrePack()
  // itself -- the expensive part: reading a (possibly mmap'd, multi-GB) constant tensor and
  // repacking it -- runs unguarded and can proceed concurrently across nodes, since it only
  // touches that node's own kernel instance and a node-local scratch PrePackedWeights object.
  std::mutex commit_mutex;

  // On the parallel path, erasing a fully-consumed constant initializer from
  // SessionState::constant_initialized_tensors_ while another node's PrePack() concurrently reads that same
  // map (e.g. via OpKernelInfo::TryGetConstantInput(), which is unguarded) is undefined behavior even though
  // the two operations target different keys. Defer the actual erase() calls until every parallel task has
  // joined below, where they can run single-threaded.
  InlinedVector<std::pair<SessionState*, int>> pending_erasures;

  auto process_node = [this, &constant_initializers_use_count, &initializers_to_share_map, &commit_mutex,
                       &pending_erasures](
                          const Node& node, bool should_cache_prepacked_weights_for_shared_initializers,
                          bool guard_bookkeeping, bool outer_prepack_parallelism_enabled) -> Status {
    auto kernel = GetMutableKernel(node.Index());
    if (kernel != nullptr) {
      kernel->SetOuterPrePackParallelism(outer_prepack_parallelism_enabled);
    }
    auto reset_outer_prepack_parallelism = gsl::finally([kernel]() {
      if (kernel != nullptr) {
        kernel->SetOuterPrePackParallelism(false);
      }
    });
    int input_idx = 0;
    for (auto& input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        const std::string& input_name = input_def->Name();
        SessionState* st = this;
        auto* prepacked_for_graph = &graph_.GetPrepacked();
        // subgraph can use the value from outer scope,
        // so it needs to check if current node uses constant initialized tensor from current and outer graphs
        do {
          int ort_value_idx;
          if (st->GetOrtValueNameIdxMap().GetIdx(input_name, ort_value_idx).IsOK()) {
            std::unordered_map<int, OrtValue>& constant_initialized_tensors = st->constant_initialized_tensors_;

            OrtValue const_initialized_value;
            bool has_const_initializer = false;
            {
              std::unique_lock<std::mutex> initializer_lock(commit_mutex, std::defer_lock);
              if (guard_bookkeeping) {
                initializer_lock.lock();
              }

              const auto initializer_it = constant_initialized_tensors.find(ort_value_idx);
              if (initializer_it != constant_initialized_tensors.end()) {
                // Keep the tensor alive while PrePack runs. Another node may consume the final use
                // and erase the map entry as soon as the bookkeeping lock is released.
                const_initialized_value = initializer_it->second;
                has_const_initializer = true;
              }
            }

            if (has_const_initializer) {
              bool is_packed = false;
              const Tensor& const_initialized_tensor = const_initialized_value.Get<Tensor>();

              auto iter = initializers_to_share_map.find(input_name);
              bool is_shared_initializer = (iter != initializers_to_share_map.end());

              // CPU EP only. An initializer joins the shared pre-packed container either when it was
              // registered via OrtApi::AddInitializer (is_shared_initializer) or when a graph transformer
              // tagged this synthesized initializer with a sharing identity. Only the tag's *presence*
              // matters here: it is the enrollment signal. The container key below is the packed-bytes
              // hash, never the tag value (see the rationale at the key computation).
              const bool enroll_tagged_initializer =
                  (st->graph_.GetSharedPrepackInitializerId(input_name) != nullptr);

              // Deferred lock covering the bookkeeping below. Only actually acquired when
              // guard_bookkeeping is set (parallel path, no cross-session sharing); left unowned
              // (no-op) on the sequential/cross-session-sharing paths.
              std::unique_lock<std::mutex> bookkeeping_lock(commit_mutex, std::defer_lock);

              if ((is_shared_initializer || enroll_tagged_initializer) &&
                  should_cache_prepacked_weights_for_shared_initializers &&
                  node.GetExecutionProviderType() == kCpuExecutionProvider) {
                // caching of pre-packed weights' turned ON

                AllocatorPtr allocator_for_caching = prepacked_weights_container_->GetOrCreateAllocator(CPU);
                ORT_ENFORCE(allocator_for_caching.get() != nullptr);

                PrePackedWeights weights_to_be_filled_in;
                // The reason we invoke PrePack() before looking into the container for any pre-packed weight
                // cached by another instance of the same op_type (for the same constant initializer) is because
                // to truly know if we can use a cached pre-packed weight, we would have to compare the cached
                // pre-packed  weight with the pre-packed weight generated by this instance of the same op_type
                // because other static properties of the node like node attributes could play a role in the
                // pre-packed weights' contents.
                ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx, allocator_for_caching,
                                                    is_packed,
                                                    &weights_to_be_filled_in));

                if (is_packed) {
                  // BUG CHECK: Ensure that a kernel either filled in the pre-packed weights
                  // to be cached, or explicitly marked the packed weights as kernel-owned.
                  ORT_RETURN_IF_NOT(!weights_to_be_filled_in.buffers_.empty() ||
                                        weights_to_be_filled_in.has_kernel_owned_packed_weights_,
                                    "The kernel corresponding to the node ", node.Name(),
                                    " doesn't have an implementation that can cache computed pre-packed weights");
                }

                if (is_packed && !weights_to_be_filled_in.has_kernel_owned_packed_weights_) {
                  const auto& op_type = node.OpType();

                  // Sanity check
                  // TODO: Check if some version of the ONNX IR allows op_type to be empty
                  ORT_ENFORCE(!op_type.empty(), "The op type of a node cannot be empty");

                  // Key by the packed-bytes hash (op_type + a hash of the packed buffer), exactly as the
                  // AddInitializer path does, so only byte-identical packed buffers are ever shared. The
                  // tag is solely the enrollment signal that opted this fusion-generated initializer into
                  // the container; it must NOT be used as the key, because it is derived from the
                  // *unpacked* initializer content and so cannot distinguish packings that differ by node
                  // options/attributes that change the packed layout (e.g. mlas.use_lut_gemm or a CPU
                  // backend-selector difference). Two sessions that share a container but differ in such an
                  // option compute the same tag yet produce different packed bytes; keying by the packed
                  // bytes gives them distinct keys and prevents reusing an incompatible buffer
                  // (wrong results/crash).
                  const std::string prepacked_weights_container_key =
                      GenerateKeyForPrepackedWeightsMap(op_type, weights_to_be_filled_in);

                  bool container_contains_packed_weight = prepacked_weights_container_->HasWeight(
                      prepacked_weights_container_key);

                  if (container_contains_packed_weight) {
                    LOGS(logger_, INFO) << "Using cached version of pre-packed weight for constant initializer: "
                                        << input_name
                                        << " used in the node: " << node.Name() << " which is of op type: "
                                        << node.OpType();

                    const auto& prepacked_shared = prepacked_weights_container_->GetWeight(
                        prepacked_weights_container_key);
                    ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                        prepacked_shared,
                                                                        node.Name()));

                    ++used_shared_pre_packed_weights_counter_;

                    // Write references to what is stored in the shared container
                    // and release memory mapped entries this container may have loaded from disk
                    prepacked_for_graph->DiscardAndReplaceWithReferenceIfSaving(
                        input_name, prepacked_weights_container_key, prepacked_shared);

                  } else {
                    // container doesn't contain the pre-packed weight - so write into it for sharing across
                    // kernel instances

                    // Check if we loaded it from disk, then put it into the shared container so
                    // everybody can share the same memory mapped entry
                    // the shared container takes ownership of the memory mapped entries

                    // The next line replaces the existing entry with references to it
                    // and returns the container that holds the memory mapped entries
                    // so we can transfer it to shared container.
                    // if there is not an entry, we replace it with references to weights_to_be_filled_in
                    // in saving mode and return std::nullopt
                    auto prepacked_from_disk = prepacked_for_graph->ReplaceWithReferenceIfSaving(
                        input_name,
                        prepacked_weights_container_key,
                        weights_to_be_filled_in);

                    if (prepacked_from_disk.has_value()) {
                      weights_to_be_filled_in = std::move(*prepacked_from_disk);
                    }

                    if (!prepacked_weights_container_->WriteWeight(prepacked_weights_container_key,
                                                                   std::move(weights_to_be_filled_in))) {
                      return ORT_MAKE_STATUS(
                          ONNXRUNTIME, FAIL,
                          "Unable to write the provided PrePackedWeights instance into the container");
                    }

                    const auto& shared_prepacked = prepacked_weights_container_->GetWeight(
                        prepacked_weights_container_key);
                    ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                        shared_prepacked,
                                                                        node.Name()));
                  }
                }

              } else {
                // cross session caching of pre-packed weights' turned OFF
                // we use serialization container to share weights loaded from disk
                // within this session. Or if the weight is not present on disk,
                // we store the newly minted pre-packed data.

                AllocatorPtr session_initializer_alloc = GetInitializerAllocator(
                    kernel->Info().GetDevice(OrtMemType::OrtMemTypeDefault));
                // A plugin EP registered as a separate library may not have an initializer
                // allocator registered under the kernel's device key, so the lookup above can
                // return null. Fall back to the kernel's own default-memory allocator (resolved
                // through the EP), which is always valid. This keeps PrePack implementations from
                // each having to special-case a null allocator at the library boundary.
                if (!session_initializer_alloc) {
                  session_initializer_alloc = kernel->Info().GetAllocator(OrtMemType::OrtMemTypeDefault);
                }
                PrePackedWeights weights_to_be_filled_in;
                // The reason we invoke PrePack() before looking into the container for any pre-packed weight
                // cached by another instance of the same op_type (for the same constant initializer) is because
                // to truly know if we can use a cached pre-packed weight, we would have to compare the cached
                // pre-packed weight with the pre-packed weight generated by this instance of the same op_type because
                // other static properties of the node like node attributes could play a role in the pre-packed
                // weights' contents.
                //
                // This is the expensive step (reads/repacks the full constant tensor, which may be
                // mmap'd external data): intentionally left unguarded so it can run concurrently with
                // other nodes' PrePack() calls on the parallel path.
                ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx, session_initializer_alloc,
                                                    is_packed,
                                                    &weights_to_be_filled_in));

                if (guard_bookkeeping) {
                  bookkeeping_lock.lock();
                }

                // Some kernels (non-CPU related kernels) do not share their pre-packed results
                // even though they set is_packed = true so we leave it up to them.
                // We can change their behavior if we wish do so in a separate PR
                if (is_packed && !weights_to_be_filled_in.buffers_.empty()) {
                  const auto& op_type = node.OpType();
                  const std::string prepacked_weights_container_key = GenerateKeyForPrepackedWeightsMap(
                      op_type,
                      weights_to_be_filled_in);

                  // See if we can use pre-packed data from disk
                  const auto* weights_to_use = prepacked_for_graph->GetPrepackedWeights(
                      prepacked_weights_container_key);

                  if (weights_to_use == nullptr) {
                    // In this case pre-packed container owns the data
                    prepacked_for_graph->WritePackedMaybeForSave(input_name, prepacked_weights_container_key,
                                                                 std::move(weights_to_be_filled_in));
                    weights_to_use = prepacked_for_graph->GetPrepackedWeights(prepacked_weights_container_key);
                    assert(weights_to_use != nullptr);
                  }

                  ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                      *weights_to_use,
                                                                      node.Name()));
                }
              }

              if (is_packed) {
                if (guard_bookkeeping && !bookkeeping_lock.owns_lock()) {
                  bookkeeping_lock.lock();
                }

                ++number_of_prepacks_counter_;

                if (constant_initializers_use_count.count(input_name) && --constant_initializers_use_count[input_name] == 0) {
                  // release the constant initialized tensor
                  if (guard_bookkeeping) {
                    // Defer the erase() calls to after the parallel tasks have joined; see the comment on
                    // pending_erasures above.
                    pending_erasures.emplace_back(st, ort_value_idx);
                  } else {
                    st->initialized_tensors_.erase(ort_value_idx);
                    constant_initialized_tensors.erase(ort_value_idx);
                  }
                }
              }
            }
            // stop searching in 2 cases:
            // 1. value is not from OuterScope
            // 2. value is from OuterScope and the current OuterScope has the value
            if (st != this || !st->graph_.IsOuterScopeValue(input_name)) {
              break;
            }
          }
          st = st->Parent();
          prepacked_for_graph = &st->graph_.GetPrepacked();
        } while (st);
      }
      input_idx++;
    }

    return Status::OK();
  };

  // Initializers may already have been removed from Graph after being materialized as OrtValues, so
  // Graph::GetConstantInitializer() cannot reliably find them here. Use the materialized constants map, which
  // includes outer-scope constants re-indexed for subgraphs.
  auto has_constant_initializer_input = [this](const Node& node) {
    const auto& constant_initialized_tensors = GetConstantInitializedTensorsForKernelCreation();
    for (const auto* input_def : node.InputDefs()) {
      if (!input_def->Exists()) {
        continue;
      }

      int ort_value_idx;
      if (GetOrtValueNameIdxMap().GetIdx(input_def->Name(), ort_value_idx).IsOK() &&
          constant_initialized_tensors.count(ort_value_idx) != 0) {
        return true;
      }
    }

    return false;
  };

  bool should_cache_prepacked_weights_for_shared_initializers = (prepacked_weights_container_ != nullptr);

  auto is_parallel_prepack_candidate = [&has_constant_initializer_input](const Node& node) {
    return node.GetExecutionProviderType() == kCpuExecutionProvider && has_constant_initializer_input(node);
  };

  std::vector<const Node*> parallel_prepack_nodes;
  auto* intra_op_thread_pool = GetThreadPool();
  if (!should_cache_prepacked_weights_for_shared_initializers &&
      concurrency::ThreadPool::DegreeOfParallelism(intra_op_thread_pool) > 1 &&
      sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsEnableParallelPrepack, "0") == "1") {
    parallel_prepack_nodes.reserve(GetGraphViewer().NumberOfNodes());
    for (auto& node : GetGraphViewer().Nodes()) {
      if (is_parallel_prepack_candidate(node)) {
        parallel_prepack_nodes.push_back(&node);
      }
    }
  }

  const bool enable_parallel_prepack = parallel_prepack_nodes.size() > 1;

  if (should_cache_prepacked_weights_for_shared_initializers) {
    // serialize calls to the method that looks up the container, calls UseCachedPrePackedWeight/PrePack
    // and writes pre-packed weights to the container
    std::lock_guard<std::mutex> l(prepacked_weights_container_->mutex_);
    for (auto& node : GetGraphViewer().Nodes()) {
      if (sess_options_.IsLoadCancellationFlagSet()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                               "Weight pre-packing was canceled due to user request.");
      }
      ORT_RETURN_IF_ERROR(process_node(node, /*should_cache*/ true, /*guard_bookkeeping*/ false,
                                       /*outer_prepack_parallelism_enabled*/ false));
    }
    return Status::OK();
  }

  if (!enable_parallel_prepack) {
    for (auto& node : GetGraphViewer().Nodes()) {
      if (sess_options_.IsLoadCancellationFlagSet()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                               "Weight pre-packing was canceled due to user request.");
      }
      ORT_RETURN_IF_ERROR(process_node(node, /*should_cache*/ false, /*guard_bookkeeping*/ false,
                                       /*outer_prepack_parallelism_enabled*/ false));
    }
    return Status::OK();
  }

  // Parallel path: fan out eligible CPU nodes across the intra-op thread pool. Nodes assigned to
  // every other EP remain on the sequential path.
  LOGS(logger_, INFO) << "Pre-packing constant initializers using the intra-op thread pool.";
  for (auto& node : GetGraphViewer().Nodes()) {
    if (is_parallel_prepack_candidate(node)) {
      continue;
    }

    if (sess_options_.IsLoadCancellationFlagSet()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "Weight pre-packing was canceled due to user request.");
    }
    ORT_RETURN_IF_ERROR(process_node(node, /*should_cache*/ false, /*guard_bookkeeping*/ false,
                                     /*outer_prepack_parallelism_enabled*/ false));
  }

  std::vector<Status> statuses(parallel_prepack_nodes.size());
  std::atomic<bool> cancellation_requested{false};
  std::atomic<bool> failure_requested{false};
  concurrency::ThreadPool::TrySimpleParallelFor(
      intra_op_thread_pool, static_cast<std::ptrdiff_t>(parallel_prepack_nodes.size()),
      [&](std::ptrdiff_t i) {
        if (failure_requested) {
          return;
        }
        if (sess_options_.IsLoadCancellationFlagSet()) {
          cancellation_requested = true;
          return;
        }

        Status& status = statuses[static_cast<size_t>(i)];
        const Node* node = parallel_prepack_nodes[static_cast<size_t>(i)];
        ORT_TRY {
          status = process_node(*node, /*should_cache*/ false, /*guard_bookkeeping*/ true,
                                /*outer_prepack_parallelism_enabled*/ true);
        }
        ORT_CATCH(const OnnxRuntimeException& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            status = Status(ex.Category(), ex.Code(),
                            MakeString("Exception while pre-packing node '", node->Name(), "': ", ex.what()));
          });
        }
        ORT_CATCH(const std::exception& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            status = ORT_MAKE_STATUS(
                ONNXRUNTIME, RUNTIME_EXCEPTION, "Exception while pre-packing node '", node->Name(), "': ", ex.what());
          });
        }
        ORT_CATCH(...) {
          status = ORT_MAKE_STATUS(
              ONNXRUNTIME, RUNTIME_EXCEPTION, "Unknown exception while pre-packing node '", node->Name(), "'.");
        }
        if (!status.IsOK()) {
          failure_requested = true;
        }
      });

  // All parallel tasks have joined; it is now safe to perform the deferred erasures single-threaded.
  for (const auto& [st, ort_value_idx] : pending_erasures) {
    st->initialized_tensors_.erase(ort_value_idx);
    st->constant_initialized_tensors_.erase(ort_value_idx);
  }

  // A worker's initial cancellation check can pass before the flag is set, letting it spend a long time
  // inside PrePack() while cancellation_requested stays false. Recheck the shared flag directly now that
  // every task has joined, instead of relying solely on what individual workers observed.
  if (cancellation_requested || sess_options_.IsLoadCancellationFlagSet()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                           "Weight pre-packing was canceled due to user request.");
  }
  for (auto& st : statuses) {
    ORT_RETURN_IF_ERROR(st);
  }
  return Status::OK();
}

static int64_t
CalculateMemoryPatternsKey(const gsl::span<const OrtValue>& tensor_inputs) {
  int64_t key = 0;
  for (const auto& input : tensor_inputs) {
    for (auto dim : input.Get<Tensor>().Shape().GetDims()) key ^= dim;
  }
  return key;
}

#ifdef ENABLE_TRAINING
namespace {
Status ResolveDimParams(const GraphViewer& graph,
                        const InlinedHashMap<std::string, TensorShape>& feeds,
                        InlinedHashMap<std::string, int64_t>& out) {
  out.reserve(graph.GetInputs().size());
  for (const auto* input : graph.GetInputs()) {
    auto* shape = input->Shape();
    auto it = feeds.find(input->Name());
    if (it == feeds.end()) {
      return Status(ONNXRUNTIME, FAIL,
                    "Graph input " + input->Name() +
                        " is not found in the feed list, unable to resolve the value for dynamic shape.");
    }
    if (it->second.NumDimensions() == 0 && !shape) {
      // This is a scalar, which has nothing to do with symbolic shapes.
      continue;
    }
    if (!shape || shape->dim_size() != static_cast<int>(it->second.NumDimensions())) {
      return Status(ONNXRUNTIME, FAIL, "Graph input " + input->Name() +
                                           "'s shape is not present or its shape doesn't match feed's shape."
                                           "Unable to resolve the value for dynamic shape");
    }
    for (int k = 0, end = shape->dim_size(); k < end; ++k) {
      if (shape->dim()[k].has_dim_param()) {
        out.insert({shape->dim()[k].dim_param(), it->second.GetDims()[k]});
      }
    }
  }
  return Status::OK();
}

Status TryResolveShape(
    const NodeArg* arg,
    const InlinedHashMap<std::string, int64_t>& symbolic_dimensions,
    size_t& is_resolved,  // indicate whether resolve successfully or not.
    TensorShapeVector& resolved_shape) {
  if (!arg->Shape()) {
    is_resolved = 0;
    return Status::OK();
  }

  TensorShapeVector shape;

  SafeInt<size_t> safe_size = 1;
  for (auto& dim : arg->Shape()->dim()) {
    if (dim.has_dim_param()) {
      auto it = symbolic_dimensions.find(dim.dim_param());
      if (it == symbolic_dimensions.end()) {
        return Status(ONNXRUNTIME, FAIL, "Unknown symbolic dimension, " + dim.dim_param() + ", found in memory pattern compute.");
      }
      safe_size *= it->second;
      shape.push_back(it->second);
    } else if (dim.has_dim_value() && dim.dim_value() > 0) {
      safe_size *= dim.dim_value();
      shape.push_back(dim.dim_value());
    } else {
      // tensor shape is unknown.
      safe_size = 0;
    }
  }

  is_resolved = safe_size;
  // Only assign shape if all symbolic dimensions are resolved.
  if (is_resolved != 0) {
    resolved_shape = std::move(shape);
  }

  return Status::OK();
}

void TryCalculateSizeFromResolvedShape(int ml_value_idx, const InlinedHashMap<int, TensorShape>& resolved_shapes, size_t& size) {
  size = 0;
  auto shape = resolved_shapes.find(ml_value_idx);
  if (shape != resolved_shapes.end()) {
    size = 1;
    for (auto dim : shape->second.GetDims())
      size *= dim;
  }
}

}  // namespace

// If this function fails NO memory planning will take place, hence lets ONLY FAIL and stop training where warranted, example SIZE overflow.
Status SessionState::GeneratePatternGroupCache(gsl::span<const OrtValue> tensor_inputs,
                                               gsl::span<const int> feed_mlvalue_idxs,
                                               MemoryPatternGroup& output,
                                               InlinedHashMap<int, TensorShape>& resolved_shapes) const {
  InlinedHashMap<std::string, TensorShape> feeds;
  feeds.reserve(feed_mlvalue_idxs.size());
  for (size_t i = 0, end = feed_mlvalue_idxs.size(); i < end; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(feed_mlvalue_idxs[i], name));
    feeds.emplace(std::move(name), tensor_inputs[i].Get<Tensor>().Shape());
  }
  InlinedHashMap<std::string, int64_t> map;
  ORT_RETURN_IF_ERROR(ResolveDimParams(*graph_viewer_, feeds, map));
  auto* exe_plan = GetExecutionPlan();
  ORT_ENFORCE(exe_plan);
  OrtValuePatternPlanner mem_planner(*exe_plan, /*using counters*/ true);

  // Try to resolve shapes for activations.
  auto& node_index_info = GetNodeIndexInfo();
  auto& execution_order = exe_plan->node_execution_order_in_training;
  for (auto& node_idx : execution_order) {
    int node_index = node_index_info.GetNodeOffset(node_idx);
    auto* node = graph_viewer_->GetNode(node_idx);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) +
                       static_cast<int>(node->ImplicitInputDefs().size());

    for (int i = 0, end = static_cast<int>(node->OutputDefs().size()); i < end; ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry)
        continue;

      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;

      auto* arg = node->OutputDefs()[i];
      size_t is_resolved = 0;
      TensorShapeVector resolved_shape;

      // Tensors whose shape cannot be resolved statically will be allocated at runtime.
      if (TryResolveShape(arg, map, is_resolved, resolved_shape).IsOK()) {
        // Store all valid resolved shapes. They will be queried in, for example,
        // Recv operator to bypass the dependency of output shapes on inputs.
        if (is_resolved != 0) {
          resolved_shapes[ml_value_idx] = gsl::make_span(resolved_shape);
        }
      } else {
        LOGS(logger_, INFO) << "[Static memory planning] Could not resolve shape for tensor with ML index "
                            << ml_value_idx << ", will allocate dynamically.";
      }
    }
  }

  // Allocate activations that want to be laid out contiguously in memory.
  for (auto ml_value_idx : exe_plan->activation_allocation_order) {
    ORT_ENFORCE(ml_value_idx >= 0);

    const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
    if (!ml_type->IsTensorType())
      continue;

    const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
    if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
        ml_data_type != DataTypeImpl::GetType<std::string>()) {
      size_t size = 0;
      TryCalculateSizeFromResolvedShape(ml_value_idx, resolved_shapes, size);
      if (size == 0) {
        std::string node_name;
        ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(ml_value_idx, node_name));
        return Status(ONNXRUNTIME, FAIL, "Unknown shape found in memory pattern compute, node name is : " + node_name);
      }

      if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(size, ml_data_type->Size(), &size)) {
        return Status(ONNXRUNTIME, FAIL, "Size overflow");
      }

      ORT_ENFORCE(exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate);

      const auto& counter = exe_plan->allocation_plan[ml_value_idx].program_counter;
      ORT_RETURN_IF_ERROR(mem_planner.TraceAllocation(ml_value_idx, counter, size));
    }
  }
  // TODO: add check for single stream
  // Allocate all other activations.
  for (auto& step_index : execution_order) {
    int node_index = node_index_info.GetNodeOffset(step_index);
    auto* node = graph_viewer_->GetNode(step_index);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) +
                       static_cast<int>(node->ImplicitInputDefs().size());
    // allocate output
    for (int i = 0, end = static_cast<int>(node->OutputDefs().size()); i < end; ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry ||
          (std::find(exe_plan->activation_allocation_order.begin(),
                     exe_plan->activation_allocation_order.end(), ml_value_idx) !=
           exe_plan->activation_allocation_order.end()))
        continue;
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;

      if (exe_plan->allocation_plan[ml_value_idx].location.MemType() != OrtDevice::MemType::DEFAULT)
        continue;

      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      size_t size = 0;
      TryCalculateSizeFromResolvedShape(ml_value_idx, resolved_shapes, size);

      // Plan memory if conditions are met.
      if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
          ml_data_type != DataTypeImpl::GetType<std::string>() && size != 0) {
        size_t aligned_size = 0;
        if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(size, ml_data_type->Size(), &aligned_size)) {
          return Status(ONNXRUNTIME, FAIL, "Size overflow");
        }

        ORT_ENFORCE(exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate);

        const auto& counter = exe_plan->allocation_plan[ml_value_idx].program_counter;
        ORT_RETURN_IF_ERROR(mem_planner.TraceAllocation(ml_value_idx, counter, aligned_size));
      }
    }

    // release nodes
    auto& release_actions = exe_plan->node_release_list[step_index];
    for (auto it = release_actions.begin(); it != release_actions.end(); ++it) {
      auto& action = exe_plan->release_actions[*it];
      // if the value consumed by multiple stream, we can't pre-release it statically.
      if (action.ref_count != 1)
        continue;

      auto ml_value_idx = action.value_index;
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      if (ml_data_type != DataTypeImpl::GetType<std::string>()) {
        ORT_RETURN_IF_ERROR(mem_planner.TraceFree(static_cast<int>(ml_value_idx)));
      }
    }
  }

  if (!mem_planner.GeneratePatterns(output).IsOK()) {
    return Status(ONNXRUNTIME, FAIL, "Generate Memory Pattern failed");
  }
  return Status::OK();
}

#endif

// MemoryPatternGroup pointer is cached. It only inserted upon creation
// and is not updated if already present.
const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(
    gsl::span<const OrtValue> tensor_inputs,
    gsl::span<const int> feed_mlvalue_idxs,
    const InlinedHashMap<int, TensorShape>*& out_inferred_shapes) const {
  out_inferred_shapes = nullptr;
  int64_t key = CalculateMemoryPatternsKey(tensor_inputs);
  std::lock_guard<std::mutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
#ifdef ENABLE_TRAINING
    MemoryPatternGroup mem_patterns;
    InlinedHashMap<int, TensorShape> inferred_shapes;
    if (GeneratePatternGroupCache(tensor_inputs, feed_mlvalue_idxs, mem_patterns, inferred_shapes).IsOK()) {
      auto patt_insert = mem_patterns_.insert_or_assign(key, std::move(mem_patterns));
      auto ptr = &patt_insert.first->second;
      auto shape_insert = shape_patterns_.insert_or_assign(key, std::move(inferred_shapes));
      out_inferred_shapes = &shape_insert.first->second;
      return ptr;
    }
#else
    ORT_UNUSED_PARAMETER(feed_mlvalue_idxs);
#endif
    return nullptr;
  }

  auto patt_hit = shape_patterns_.find(key);
  if (patt_hit != shape_patterns_.cend()) {
    out_inferred_shapes = &patt_hit->second;
  }
  return &it->second;
}

void SessionState::ResolveMemoryPatternFlag() {
  if (enable_mem_pattern_) {
    for (auto* input : graph_viewer_->GetInputs()) {
      if (!input->HasTensorOrScalarShape()) {
        enable_mem_pattern_ = false;
        break;
      }
    }

    // if there are nodes belong to the same device be partitioned to multiple streams
    // disable the memory pattern because the execution order is not fixed.
    // TODO: we can improve memory pattern to support multiple streams
    bool multi_stream = false;
    auto cmp = [](const OrtDevice& op1, const OrtDevice& op2) {
      if (op1.Type() != op2.Type()) return op1.Type() < op2.Type();
      if (op1.MemType() != op2.MemType()) return op1.MemType() < op2.MemType();
      return op1.Id() < op2.Id();
    };
    std::set<OrtDevice, decltype(cmp)> device_set(cmp);
    auto& streams = GetExecutionPlan()->execution_plan;
    for (auto& logic_stream : streams) {
      if (device_set.find(logic_stream->device_) != device_set.end()) {
        multi_stream = true;
        break;
      }
      device_set.insert(logic_stream->device_);
    }

    if (multi_stream)
      enable_mem_pattern_ = false;

    // For subgraphs, the implicit inputs need to meet the same crieria
    // as the explicit inputs for memory pattern to be enabled
    if (graph_viewer_->IsSubgraph()) {
      const auto* parent_node = graph_viewer_->ParentNode();

      for (auto* implicit_input : parent_node->ImplicitInputDefs()) {
        if (!implicit_input->HasTensorOrScalarShape()) {
          enable_mem_pattern_ = false;
          break;
        }
      }
    }
  }
}

Status SessionState::UpdateMemoryPatternGroupCache(gsl::span<const OrtValue> tensor_inputs,
                                                   MemoryPatternGroup mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(tensor_inputs);

  std::lock_guard<std::mutex> lock(mem_patterns_lock_);
  // Do not update if present, as the pointer to the existing one is cached
  mem_patterns_.emplace(key, std::move(mem_patterns));
  return Status::OK();
}

bool SessionState::GetEnableMemoryPattern() const { return enable_mem_pattern_; }

bool SessionState::GetEnableMemoryReuse() const { return sess_options_.enable_mem_reuse; }

common::Status SessionState::AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info) {
  // Graph partitioning should ensure an input is only consumed from one device. Copy nodes should have been inserted
  // to handle a scenario where an input is required on different devices by different nodes. Validate that.
  auto& entries = input_names_to_nodeinfo_mapping_[input_name];

  if (entries.empty()) {
    entries.push_back(node_info);
  } else {
    const auto& existing_entry = entries.front();

    // if index == max it's an entry for an implicit input to a subgraph or unused graph input.
    // we want to prefer the entry for explicit usage in this graph, as the implicit usage in a
    // subgraph will be handled by the subgraph's SessionState.
    if (node_info.index == std::numeric_limits<size_t>::max()) {
      // ignore and preserve existing value
    } else if (existing_entry.index == std::numeric_limits<size_t>::max()) {
      // replace existing entry that is for an implicit input with new entry for explicit usage in this graph
      entries[0] = node_info;
    } else {
      // if the devices match we can add the new entry for completeness (it will be ignored in
      // utils::CopyOneInputAcrossDevices though).
      // if they don't, we are broken.
      const auto& current_device = entries[0].device;
      const auto& new_device = node_info.device;

      if (current_device == new_device) {
        entries.push_back(node_info);
      } else {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, NOT_IMPLEMENTED,
            "Using an input in multiple nodes on different devices is not supported currently. Input:", input_name,
            " is used by node ", existing_entry.p_node->Name(), " (", current_device->ToString(), ") and node ",
            node_info.p_node->Name(), " (", new_device->ToString(), ").");
      }
    }
  }

  return Status::OK();
}

common::Status SessionState::GetInputNodeInfo(const std::string& input_name,
                                              InlinedVector<NodeInfo>& node_info_vec) const {
  auto entry = input_names_to_nodeinfo_mapping_.find(input_name);
  if (entry == input_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find input name in the mapping: " + input_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetInputNodeInfoMap() const {
  return input_names_to_nodeinfo_mapping_;
}

void SessionState::AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info) {
  auto& output_names_to_nodeinfo = output_names_to_nodeinfo_mapping_[output_name];
  ORT_ENFORCE(output_names_to_nodeinfo.empty(), "Only one node should produce an output. Existing entry for ",
              output_name);

  output_names_to_nodeinfo.push_back(node_info);
}

common::Status SessionState::GetOutputNodeInfo(const std::string& output_name,
                                               InlinedVector<NodeInfo>& node_info_vec) const {
  auto entry = output_names_to_nodeinfo_mapping_.find(output_name);
  if (entry == output_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find output name in the mapping: " + output_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetOutputNodeInfoMap() const {
  return output_names_to_nodeinfo_mapping_;
}

void SessionState::AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                                           std::unique_ptr<SessionState> session_state) {
  auto entry = subgraph_session_states_.find(index);

  // make sure this is new. internal logic error if it is not so using ORT_ENFORCE.
  if (entry != subgraph_session_states_.cend()) {
    const auto& existing_entries = entry->second;
    ORT_ENFORCE(existing_entries.find(attribute_name) == existing_entries.cend(), "Entry exists in node ", index,
                " for attribute ", attribute_name);
  }

  session_state->parent_ = this;

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  GenerateGraphId();
#endif

  subgraph_session_states_[index].insert(std::make_pair(attribute_name, std::move(session_state)));
}

SessionState* SessionState::GetMutableSubgraphSessionState(onnxruntime::NodeIndex index,
                                                           const std::string& attribute_name) {
  SessionState* session_state = nullptr;

  auto node_entry = subgraph_session_states_.find(index);
  if (node_entry != subgraph_session_states_.cend()) {
    const auto& attribute_state_map = node_entry->second;

    const auto& subgraph_entry = attribute_state_map.find(attribute_name);
    if (subgraph_entry != attribute_state_map.cend()) {
      session_state = subgraph_entry->second.get();
    }
  }

  return session_state;
}

const SessionState* SessionState::GetSubgraphSessionState(onnxruntime::NodeIndex index,
                                                          const std::string& attribute_name) const {
  return const_cast<SessionState*>(this)->GetMutableSubgraphSessionState(index, attribute_name);
}

const NodeIndexInfo& SessionState::GetNodeIndexInfo() const {
  ORT_ENFORCE(node_index_info_.has_value(), "SetGraphAndCreateKernels must be called prior to GetExecutionInfo.");
  return *node_index_info_;
}

#ifdef ENABLE_TRAINING
void SessionState::UpdateToBeExecutedRange(gsl::span<int const> fetch_mlvalue_idxs) {
  InlinedVector<int> sorted_idxs;
  sorted_idxs.reserve(fetch_mlvalue_idxs.size());
  sorted_idxs.assign(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end());
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  if (to_be_executed_nodes_.find(sorted_idxs) != to_be_executed_nodes_.end())
    return;

  // Get the nodes generating the fetches.
  InlinedVector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  InlinedHashSet<NodeIndex> reachable_nodes;
  reachable_nodes.reserve(graph_.NumberOfNodes());

  for (auto idx : fetch_mlvalue_idxs) {
    std::string node_arg_name;
    const auto status = this->GetOrtValueNameIdxMap().GetName(idx, node_arg_name);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    auto ending_node = graph_.GetProducerNode(node_arg_name);
    nodes.push_back(ending_node);
  }

  // Reversely traverse to get reachable nodes.
  graph_.ReverseDFSFrom(
      nodes, {}, [&reachable_nodes](const Node* n) { reachable_nodes.insert(n->Index()); });

  // global start, end doesn't matters
  to_be_executed_nodes_.emplace(std::move(sorted_idxs), std::move(reachable_nodes));
}

const InlinedHashSet<NodeIndex>* SessionState::GetToBeExecutedRange(
    gsl::span<int const> fetch_mlvalue_idxs) const {
  InlinedVector<int> sorted_idxs;
  sorted_idxs.reserve(fetch_mlvalue_idxs.size());
  sorted_idxs.assign(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end());
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  auto it = to_be_executed_nodes_.find(sorted_idxs);
  return (it != to_be_executed_nodes_.end()) ? &it->second : nullptr;
}
#endif

Status SessionState::CreateSubgraphSessionState() {
  for (auto& node : graph_.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      const auto& ep_type = node.GetExecutionProviderType();
      const IExecutionProvider* ep = execution_providers_.Get(ep_type);
      const bool is_plugin_ep = ep != nullptr && ep->GetOrtEp() != nullptr;

      if (!ep_type.empty() &&
          ep_type != kCpuExecutionProvider && ep_type != kCudaExecutionProvider &&
          ep_type != kDmlExecutionProvider &&
          ep_type != kJsExecutionProvider && ep_type != kWebGpuExecutionProvider &&
          !is_plugin_ep) {
        // SessionState is only used when ORT is executing the subgraph. If a non-ORT EP has taken the control flow
        // node containing the subgraph it will create whatever state it needs internally.
        continue;
      }

      auto& attr_name = entry.first;
      Graph* subgraph = entry.second;
      ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

      auto subgraph_session_state =
          std::make_unique<SessionState>(*subgraph, execution_providers_,
                                         thread_pool_, inter_op_thread_pool_, data_transfer_mgr_,
                                         external_data_loader_mgr_, logger_, profiler_, sess_options_,
                                         prepacked_weights_container_, allocators_, initializer_allocators_);

      // Pass fused function manager to subgraph
      subgraph_session_state->fused_funcs_mgr_.SetFusedFuncs(fused_funcs_mgr_);

      // recurse
      ORT_RETURN_IF_ERROR(subgraph_session_state->CreateSubgraphSessionState());

      // add the subgraph SessionState instance to the parent graph SessionState so it can be retrieved
      // by Compute() via OpKernelContextInternal.
      AddSubgraphSessionState(node.Index(), attr_name, std::move(subgraph_session_state));
    }
  }

  return Status::OK();
}

// Calculate the use count of a constant initialized tensor, including the use in subgraph.
// Note: This function doesn't handle the case below:
// The main graph has a constant initializer called X, and the subgraph also has a constant initializer called X,
// which overrides the X from main graph.
// For case like this, the current implementation will calculate the use count as 2, but they could contain completely
// different values so each should have a use count of 1.
// This is a very rare case. If it happens and X is prepacked, the consequence is that X won't be released and memory
// usage of X won't be saved. This will be fine.
static void ComputeConstantInitializerUseCount(const Graph& graph, InlinedHashMap<std::string, size_t>& constant_initializers_use_count) {
  for (const auto& node : graph.Nodes()) {
    for (const auto* arg : node.InputDefs()) {
      if (arg->Exists() && graph.GetConstantInitializer(arg->Name(), true /*check_outer_scope*/)) {
        constant_initializers_use_count[arg->Name()]++;
      }
    }

    if (node.ContainsSubgraph()) {
      for (const gsl::not_null<const Graph*>& subgraph : node.GetSubgraphs()) {
        ComputeConstantInitializerUseCount(*subgraph, constant_initializers_use_count);
      }
    }
  }
  // Initializers can be used as graph outputs
  for (const auto* arg : graph.GetOutputs()) {
    if (arg->Exists() && graph.GetConstantInitializer(arg->Name(), true /*check_outer_scope*/)) {
      constant_initializers_use_count[arg->Name()]++;
    }
  }
}

using NodePlacementMap = std::unordered_map<std::string, std::vector<std::string>>;
using NodePlacementSet = std::unordered_set<std::string>;

static Status VerifyEachNodeIsAssignedToAnEpImpl(const Graph& graph, bool is_verbose,
                                                 NodePlacementMap& node_placements,
                                                 NodePlacementSet& node_placement_provider_set,
                                                 const ExecutionProviders& providers) {
  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();
    if (node_provider.empty()) {
      // Provide a more descriptive error for EPContext nodes that were not assigned to an EP.
      if (node.OpType() == "EPContext") {
        // Get information about who generated the EPContext node from the 'source' attribute.
        // Commonly, 'source' will be the name of the EP that generated the node, but that is not required.
        // An EP may choose to use a different source identifier.
        std::string source = "(unknown)";
        const auto& attrs = node.GetAttributes();
        auto it = attrs.find("source");

        if (it != attrs.end() && it->second.has_s()) {
          source = it->second.s();
        }

        const auto& ep_ids = providers.GetIds();
        std::ostringstream session_ep_names;

        for (size_t i = 0; i < ep_ids.size(); ++i) {
          if (i > 0) {
            session_ep_names << ", ";
          }
          session_ep_names << ep_ids[i];
        }

        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "EPContext node generated by '", source, "' is not ",
                               "compatible with any execution provider added to the session. ",
                               "EPContext node name: '", node.Name(), "'. Available session execution providers: [",
                               session_ep_names.str(), "].");
      }

      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Could not find an implementation for ",
                             node.OpType(), "(", node.SinceVersion(), ") node with name '", node.Name(), "'");
    }

    node_placement_provider_set.insert(node_provider);

#if !defined(ORT_MINIMAL_BUILD)
    if (is_verbose) {  // TODO: should we disable this if the number of nodes is above a certain threshold?
      const std::string node_str = node.OpType() + " (" + node.Name() + ")";
      node_placements[node_provider].push_back(node_str);
    }
#endif  // !defined(ORT_MINIMAL_BUILD)

    // recurse into subgraphs
    if (node.ContainsSubgraph()) {
      const auto subgraphs = node.GetSubgraphs();
      for (const auto& subgraph : subgraphs) {
        ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEpImpl(*subgraph, is_verbose, node_placements,
                                                               node_placement_provider_set, providers));
      }
    }
  }

  return Status::OK();
}

static Status VerifyEachNodeIsAssignedToAnEp(const Graph& graph, const logging::Logger& logger,
                                             const ExecutionProviders& providers) {
  NodePlacementMap node_placements{};
  NodePlacementSet node_placement_provider_set{};
#if !defined(ORT_MINIMAL_BUILD)
  const bool is_verbose_mode = logger.GetSeverity() == logging::Severity::kVERBOSE;
#else
  ORT_UNUSED_PARAMETER(logger);
  const bool is_verbose_mode = false;
#endif  // !defined(ORT_MINIMAL_BUILD)

  ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEpImpl(graph, is_verbose_mode, node_placements,
                                                         node_placement_provider_set, providers));

#if !defined(ORT_MINIMAL_BUILD)
  // print placement info
  if (is_verbose_mode) {
    LOGS(logger, VERBOSE) << "Node placements";
    if (node_placements.size() == 1) {
      const auto& [provider, node_strs] = *node_placements.begin();
      LOGS(logger, VERBOSE) << " All nodes placed on [" << provider << "]. Number of nodes: " << node_strs.size();
    } else {
      for (const auto& [provider, node_strs] : node_placements) {
        LOGS(logger, VERBOSE) << " Node(s) placed on [" << provider << "]. Number of nodes: " << node_strs.size();
        for (const auto& node_str : node_strs) {
          LOGS(logger, VERBOSE) << "  " << node_str;
        }
      }
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  // Silent fallback from GPU/NPU to CPU nodes can cause performance issues due to memory copies and frequent stalls.
  // If the user explicitly included the CPU provider anyway, then remain silent, but if it was implicitly added,
  // and unexpected fallback happened to a non-preferred provider, warn the user.
  size_t explicit_provider_count = providers.NumProviders() - (providers.GetCpuProviderWasImplicitlyAdded() ? 1 : 0);
  if (node_placement_provider_set.size() > explicit_provider_count && explicit_provider_count > 0) {
    LOGS(logger, WARNING) << "Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.";
    if (!is_verbose_mode) {
      LOGS(logger, WARNING) << "Rerunning with verbose output on a non-minimal build will show node assignments.";
    }
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
Status SessionState::InitializeMoeExpertState(std::shared_ptr<KernelPilotMoeExpertState> state,
                                              std::string graph_scope) {
  moe_expert_state_ = std::move(state);
  for (const auto& node : graph_.Nodes()) {
    if (node.Domain() != kMSDomain || (node.OpType() != "MoE" && node.OpType() != "QMoE")) {
      continue;
    }
    const auto& ep = node.GetExecutionProviderType();
    ORT_RETURN_IF_NOT(ep == kCpuExecutionProvider || ep == kCudaExecutionProvider,
                      "MoE expert counting is not supported by ", ep);
    const auto* shape = node.InputDefs().at(1)->Shape();
    ORT_RETURN_IF_NOT(shape && shape->dim_size() == 2 && shape->dim(1).has_dim_value() &&
                          shape->dim(1).dim_value() > 0,
                      "MoE expert counting requires a static positive expert dimension for ", node.Name());
    ORT_RETURN_IF_ERROR(moe_expert_state_->RegisterNode(
        GetKernel(node.Index()), graph_scope, node.Index(), node.OpType(), static_cast<size_t>(shape->dim(1).dim_value())));
  }
  for (auto& [node_index, subgraphs] : subgraph_session_states_) {
    for (auto& [attribute, subgraph] : subgraphs) {
      const std::string scope = MakeString(graph_scope, "/", node_index, "/",
                                           attribute.size(), ":", attribute);
      ORT_RETURN_IF_ERROR(subgraph->InitializeMoeExpertState(moe_expert_state_, scope));
    }
  }
  return Status::OK();
}
#endif

Status SessionState::FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                          const KernelRegistryManager& kernel_registry_manager,
                                          bool remove_initializers,
                                          bool saving_ort_format) {
  // recursively create the subgraph session state instances and populate the kernel create info in them.
  // it's simpler to handle the kernel create info recursively when deserializing,
  // so also do it recursively when calling PopulateKernelCreateInfo for consistency.
  ORT_RETURN_IF_ERROR(CreateSubgraphSessionState());

  ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEp(graph_, logger_, execution_providers_));
  ORT_RETURN_IF_ERROR(PopulateKernelCreateInfo(kernel_registry_manager, saving_ort_format));

  InlinedHashMap<std::string, size_t> constant_initializers_use_count;
  ComputeConstantInitializerUseCount(graph_, constant_initializers_use_count);

  MaxShapeInferenceResult max_shape_inference_result;
  const std::string max_shape_config = sess_options_.config_options.GetConfigOrDefault(
      kOrtSessionOptionsMaxShapeOverride, "");
  if (!max_shape_config.empty()) {
#if defined(ORT_MINIMAL_BUILD)
    LOGS(logger_, WARNING)
        << "session.max_shape_override is not supported in a minimal build and will be ignored.";
#else
    // Partitioning results cannot be reused here because the final graph contains the
    // completed set of EP fusions and transformations.
    MaxShapeOverrideMap input_overrides;
    ORT_RETURN_IF_ERROR(ParseMaxShapeOverride(max_shape_config, input_overrides));
    ORT_RETURN_IF_ERROR(InferMaxShapes(graph_, input_overrides, max_shape_inference_result));
#endif
  }

  Status status = FinalizeSessionStateImpl(graph_location, kernel_registry_manager, nullptr, sess_options_,
                                           remove_initializers,
                                           GetSaveModeForPrepacks(!remove_initializers, saving_ort_format),
                                           constant_initializers_use_count, max_shape_inference_result);

#if !defined(ORT_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(status);
  const bool enable_moe_expert_counting =
      sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableMoeExpertCounting, "0") == "1";
  const bool enable_moe_expert_statistics =
      sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "0") == "1";
  if (enable_moe_expert_counting || enable_moe_expert_statistics) {
    double alpha = 0.0;
    double beta = 0.0;
    auto state = std::make_shared<KernelPilotMoeExpertState>();
    const auto alpha_value =
        sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigMoeExpertCounterAlpha, "0.9");
    const auto beta_value =
        sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigMoeExpertCounterBeta, "0.1");
    ORT_RETURN_IF_NOT(TryParseStringWithClassicLocale(alpha_value, alpha),
                      "Invalid ", kOrtSessionOptionsConfigMoeExpertCounterAlpha, " value: ", alpha_value);
    ORT_RETURN_IF_NOT(TryParseStringWithClassicLocale(beta_value, beta),
                      "Invalid ", kOrtSessionOptionsConfigMoeExpertCounterBeta, " value: ", beta_value);
    ORT_RETURN_IF_ERROR(state->SetCounterParameters(alpha, beta));
    ORT_RETURN_IF_ERROR(InitializeMoeExpertState(std::move(state), "main"));
    const auto state_file =
        sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigMoeExpertCounterStateFile, "");
    if (!state_file.empty()) {
      std::ifstream input{std::filesystem::path(ToPathString(state_file))};
      ORT_RETURN_IF_NOT(input.is_open(), "Unable to open MoE expert counter state file: ", state_file);
      ORT_RETURN_IF_ERROR(moe_expert_state_->Load(input));
    }
    ORT_RETURN_IF_ERROR(moe_expert_state_->FinalizeInitialization());
  }
#endif
  return status;
}

namespace {

bool IsMoeOffloadExpertInputIndex(std::string_view op_type, size_t input_idx) noexcept {
  if (op_type == "MoE") {
    return input_idx >= 2 && input_idx <= 7;
  }
  if (op_type == "QMoE") {
    return (input_idx >= 2 && input_idx <= 13) || input_idx == 15 || input_idx == 16;
  }
  return false;
}

bool ValidateNodeForMoeOffload(const Node& node,
                               const InitializedTensorSet& initialized_tensors,
                               const InlinedHashSet<std::string_view>& graph_output_names,
                               std::string& out_reason) {
  if (node.OpType() != "QMoE") {
    out_reason = "operator '" + node.OpType() + "' does not support the QMoE offload backend";
    return false;
  }
  const auto& attrs = node.GetAttributes();
  auto get_str_attr = [&](const char* key, std::string_view def_val = {}) -> std::string {
    auto it = attrs.find(key);
    if (it != attrs.end() && it->second.has_s()) return it->second.s();
    return std::string(def_val);
  };
  auto get_int_attr = [&](const char* key, int64_t def_val) -> int64_t {
    auto it = attrs.find(key);
    if (it != attrs.end() && it->second.has_i()) return it->second.i();
    return def_val;
  };

  const std::string quant_type = moe_offload::MoeOffloadSessionPolicy::ToLowerAscii(get_str_attr("quant_type", "int"));
  if (quant_type != "nvfp4" && quant_type != "fp4" && quant_type != "mxfp4") {
    out_reason = "unsupported quant_type='" + quant_type + "' (expected 'nvfp4' or 'fp4')";
    return false;
  }
  const int64_t weight_bits = get_int_attr("expert_weight_bits", 8);
  if (weight_bits != 4) {
    out_reason = "unsupported expert_weight_bits=" + std::to_string(weight_bits) + " (expected 4)";
    return false;
  }
  const int64_t prepacked = get_int_attr("weights_prepacked", 0);
  if (prepacked != 1) {
    out_reason = "requires weights_prepacked=1 (offline K-packed row-major FP4)";
    return false;
  }
  const int64_t expected_blk = (quant_type == "nvfp4") ? 16 : 32;
  const int64_t blk = get_int_attr("block_size", -1);
  if (blk != -1 && blk != expected_blk) {
    out_reason = "invalid block_size=" + std::to_string(blk) + " for quant_type='" + quant_type + "'";
    return false;
  }
  const std::string act_type = moe_offload::MoeOffloadSessionPolicy::ToLowerAscii(get_str_attr("activation_type", ""));
  if (act_type != "swiglu") {
    out_reason = "unsupported activation_type='" + act_type + "' (expected 'swiglu')";
    return false;
  }
  const int64_t swiglu_fusion = get_int_attr("swiglu_fusion", 0);
  if (swiglu_fusion != 0 && swiglu_fusion != 1) {
    out_reason = "unsupported swiglu_fusion=" + std::to_string(swiglu_fusion) + " (expected interleaved SwiGLU)";
    return false;
  }
  const int64_t sparse_mixer = get_int_attr("use_sparse_mixer", 0);
  if (sparse_mixer != 0) {
    out_reason = "use_sparse_mixer=1 is not supported by the QMoE offload backend";
    return false;
  }
  const int64_t top_k = get_int_attr("k", 0);
  if (top_k <= 0) {
    out_reason = "attribute k must be positive";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  auto input_exists = [&](size_t idx) -> bool {
    return idx < input_defs.size() && input_defs[idx] != nullptr && input_defs[idx]->Exists() &&
           !input_defs[idx]->Name().empty();
  };
  for (size_t bad_slot : {size_t{8}, size_t{9}, size_t{10}, size_t{11}, size_t{12}, size_t{13},
                          size_t{14}, size_t{17}, size_t{18}, size_t{19}, size_t{20}}) {
    if (input_exists(bad_slot)) {
      out_reason = "unsupported optional input slot " + std::to_string(bad_slot) + " is present";
      return false;
    }
  }
  for (size_t req_slot : {size_t{2}, size_t{3}, size_t{5}, size_t{6}}) {
    if (!input_exists(req_slot)) {
      out_reason = "missing required expert weight/scale input at slot " + std::to_string(req_slot);
      return false;
    }
    const std::string& name = input_defs[req_slot]->Name();
    auto it = initialized_tensors.find(name);
    if (it == initialized_tensors.end() || it->second == nullptr) {
      out_reason = "expert input slot " + std::to_string(req_slot) + " ('" + name + "') must be a constant initializer";
      return false;
    }
    if (it->second->dims_size() != 3) {
      out_reason = "expert input slot " + std::to_string(req_slot) + " ('" + name + "') must be 3D";
      return false;
    }
    if (graph_output_names.find(name) != graph_output_names.end()) {
      out_reason = "expert initializer '" + name + "' is also a graph output";
      return false;
    }
  }
  if (quant_type == "nvfp4") {
    for (size_t g_slot : {size_t{15}, size_t{16}}) {
      if (!input_exists(g_slot)) {
        out_reason = "NVFP4 requires constant global scale at slot " + std::to_string(g_slot);
        return false;
      }
      const std::string& name = input_defs[g_slot]->Name();
      if (initialized_tensors.find(name) == initialized_tensors.end()) {
        out_reason = "NVFP4 global scale slot " + std::to_string(g_slot) + " ('" + name + "') must be a constant initializer";
        return false;
      }
      if (graph_output_names.find(name) != graph_output_names.end()) {
        out_reason = "expert initializer '" + name + "' is also a graph output";
        return false;
      }
    }
  }
  for (size_t opt_slot : {size_t{4}, size_t{7}, size_t{15}, size_t{16}}) {
    if (input_exists(opt_slot)) {
      const std::string& name = input_defs[opt_slot]->Name();
      if (initialized_tensors.find(name) == initialized_tensors.end()) {
        out_reason = "optional expert slot " + std::to_string(opt_slot) + " ('" + name + "') must be a constant initializer";
        return false;
      }
      if (graph_output_names.find(name) != graph_output_names.end()) {
        out_reason = "expert initializer '" + name + "' is also a graph output";
        return false;
      }
    }
  }
  return true;
}

Status PlaceMoeOffloadInitializersOnCpu(const GraphViewer& graph_viewer,
                                        const moe_offload::MoeOffloadSessionPolicy& policy,
                                        const OrtValueNameIdxMap& ort_value_name_idx_map,
                                        const OrtMemoryInfo& cpu_mem_info,
                                        SequentialExecutionPlan& exec_plan,
                                        const logging::Logger& logger) {
  const auto& initialized_tensors = graph_viewer.GetAllInitializedTensors();
  InlinedHashSet<std::string_view> graph_output_names;
  for (const auto* out_arg : graph_viewer.GetOutputs()) {
    if (out_arg != nullptr && out_arg->Exists() && !out_arg->Name().empty()) {
      graph_output_names.insert(out_arg->Name());
    }
  }

  InlinedHashSet<const Node*> eligible_moe_nodes;
  InlinedHashSet<std::string_view> non_offload_consumed_initializers;

  for (const auto& node : graph_viewer.Nodes()) {
    if (policy.mode == moe_offload::OffloadMode::kRequired && node.ContainsSubgraph()) {
      for (const auto& [attr_name, subgraph_ptr] : node.GetAttributeNameToSubgraphMap()) {
        ORT_UNUSED_PARAMETER(attr_name);
        if (subgraph_ptr == nullptr) continue;
        for (const auto& sub_node : subgraph_ptr->Nodes()) {
          if (sub_node.Domain() == kMSDomain && (sub_node.OpType() == "QMoE" || sub_node.OpType() == "MoE")) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "MoE offload mode='required' does not support MoE/QMoE nodes inside subgraph node '",
                                   node.Name(), "'.");
          }
        }
      }
    }

    const bool is_cuda_moe_node = (node.Domain() == kMSDomain) &&
                                  (node.OpType() == "QMoE" || node.OpType() == "MoE") &&
                                  (node.GetExecutionProviderType() == kCudaExecutionProvider);
    bool node_eligible = false;
    if (is_cuda_moe_node) {
      std::string reason;
      node_eligible = ValidateNodeForMoeOffload(node, initialized_tensors, graph_output_names, reason);
      if (!node_eligible && policy.mode == moe_offload::OffloadMode::kRequired) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MoE offload mode='required' rejected node '", node.Name(),
                               "' (", node.OpType(), "): ", reason, ".");
      }
      if (node_eligible) {
        eligible_moe_nodes.insert(&node);
      }
    }

    const auto& input_defs = node.InputDefs();
    for (size_t i = 0; i < input_defs.size(); ++i) {
      const NodeArg* arg = input_defs[i];
      if (arg == nullptr || !arg->Exists() || arg->Name().empty()) {
        continue;
      }
      if (initialized_tensors.find(arg->Name()) == initialized_tensors.end()) {
        continue;
      }
      const bool is_eligible_expert_slot = node_eligible && IsMoeOffloadExpertInputIndex(node.OpType(), i);
      if (!is_eligible_expert_slot) {
        non_offload_consumed_initializers.insert(arg->Name());
      }
    }
  }

  // A node that shares an expert initializer with a non-offload consumer cannot relocate it. Rejecting the node makes
  // all of its other expert initializers non-offload consumed too, which can in turn reject other nodes that share
  // them, so iterate until no more nodes are rejected.
  for (;;) {
    InlinedVector<const Node*> rejected_nodes;
    for (const Node* node : eligible_moe_nodes) {
      const auto& input_defs = node->InputDefs();
      for (size_t i = 0; i < input_defs.size(); ++i) {
        if (!IsMoeOffloadExpertInputIndex(node->OpType(), i)) continue;
        const NodeArg* arg = input_defs[i];
        if (arg == nullptr || !arg->Exists() || arg->Name().empty()) continue;
        if (non_offload_consumed_initializers.find(arg->Name()) == non_offload_consumed_initializers.end()) continue;
        if (policy.mode == moe_offload::OffloadMode::kRequired) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "MoE offload mode='required' rejected node '", node->Name(),
                                 "': expert initializer '", arg->Name(),
                                 "' is shared with a non-offload consumer.");
        }
        rejected_nodes.push_back(node);
        break;
      }
    }
    if (rejected_nodes.empty()) break;
    for (const Node* node : rejected_nodes) {
      eligible_moe_nodes.erase(node);
      const auto& input_defs = node->InputDefs();
      for (size_t i = 0; i < input_defs.size(); ++i) {
        const NodeArg* arg = input_defs[i];
        if (arg == nullptr || !arg->Exists() || arg->Name().empty()) continue;
        if (initialized_tensors.find(arg->Name()) != initialized_tensors.end()) {
          non_offload_consumed_initializers.insert(arg->Name());
        }
      }
    }
  }

  InlinedHashSet<std::string_view> initializers_to_place_on_cpu;
  for (const Node* node : eligible_moe_nodes) {
    const auto& input_defs = node->InputDefs();
    for (size_t i = 0; i < input_defs.size(); ++i) {
      if (!IsMoeOffloadExpertInputIndex(node->OpType(), i)) continue;
      const NodeArg* arg = input_defs[i];
      if (arg == nullptr || !arg->Exists() || arg->Name().empty()) continue;
      if (initialized_tensors.find(arg->Name()) != initialized_tensors.end()) {
        initializers_to_place_on_cpu.insert(arg->Name());
      }
    }
  }

  size_t placed_on_cpu_count = 0;
  for (std::string_view name : initializers_to_place_on_cpu) {
    int ort_value_idx = -1;
    ORT_RETURN_IF_ERROR(ort_value_name_idx_map.GetIdx(std::string(name), ort_value_idx));
    exec_plan.SetLocation(static_cast<size_t>(ort_value_idx), cpu_mem_info.device);
    ++placed_on_cpu_count;
  }

  if (placed_on_cpu_count > 0) {
    LOGS(logger, INFO) << "PlaceMoeOffloadInitializersOnCpu placed " << placed_on_cpu_count
                       << " MoE/QMoE expert initializer tensor(s) on CPU prior to SaveInitializedTensors.";
  }
  return Status::OK();
}

}  // namespace

bool SessionState::GetSaveModeForPrepacks(bool saving_model, bool saving_ort_format) {
  bool save_prepacked_constant_initializers =
      sess_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsSavePrePackedConstantInitializers,
                                                      "0") == "1";

  if (save_prepacked_constant_initializers && !saving_model) {
    save_prepacked_constant_initializers = false;
    LOGS(logger_, WARNING)
        << "SavePrePackedConstantInitializers is set to true but the model is not being saved. Ignoring the flag.";
  }

  if (save_prepacked_constant_initializers && saving_ort_format) {
    save_prepacked_constant_initializers = false;
    LOGS(logger_, WARNING)
        << "Serializing optimized model in ORT format with external pre-packed constant initializers is not supported."
        << " Ignoring the flag.";
  }

  return save_prepacked_constant_initializers;
}

static Status Index(const OrtValueNameIdxMap& ort_value_name_idx_map,
                    const OrtValueName& name,
                    /*out*/ OrtValueIndex& value) {
  return ort_value_name_idx_map.GetIdx(name, value);
}

static bool IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs(const Node& node) {
  const auto& op_type = node.OpType();
  int since_version = node.SinceVersion();

  // TODO: Re-visit this method if more subgraph ops get accepted into ONNX

  // At the time of writing, there are only 3 ops in ONNX that have subgraphs
  // 1) If
  // 2) Loop
  // 3) Scan

  // `If` - The op doesn't have explicit subgraph inputs (return false)
  // `Loop`- In all opset versions of Loop (at the time of writing) the node inputs
  // have a one-to-one mapping between them and the explicit subgraph inputs
  // of the subgraph held in the Loop (return true)
  // `Scan` - Except opset 8 version of Scan (at the time of writing), all other
  // versions have the same one-to-one mapping as Loop (return true for opset > 8)

  return (op_type == "Loop" || (op_type == "Scan" && since_version >= 9));
}

// The following method accumulates the locations of all inputs (implicit and explicit)
// to a control flow node at the current graph level. This information will be used in
// the allocation planner while determining the location of such inputs in the subgraph.
// This method will not be called for the main graph (there is no concept of "outer scope" for the main graph).
static Status OuterScopeNodeArgLocationAccumulator(const SequentialExecutionPlan& plan,
                                                   const OrtValueNameIdxMap& ort_value_name_to_idx_map,
                                                   const Node& parent_node,
                                                   const GraphViewer& subgraph,
                                                   /*out*/ InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_arg_to_location_map) {
  // Process implicit inputs to the node
  outer_scope_arg_to_location_map.reserve(parent_node.ImplicitInputDefs().size() + parent_node.InputDefs().size());
  auto process_implicit_input = [&plan, &ort_value_name_to_idx_map,
                                 &outer_scope_arg_to_location_map](const NodeArg& input, size_t /*arg_idx*/) {
    const auto& name = input.Name();
    OrtValueIndex index = -1;
    ORT_RETURN_IF_ERROR(Index(ort_value_name_to_idx_map, name, index));
    outer_scope_arg_to_location_map.insert({name, plan.GetLocation(index)});
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(parent_node.ImplicitInputDefs(), process_implicit_input));

  // Process explicit inputs to the node
  // (they are passed through as explicit subgraph inputs and hence requires a re-mapping of names
  // to their corresponding names in the inner nested subgraph(s) held by the node)
  //
  // Only nodes whose inputs map one-to-one onto the explicit subgraph inputs (Loop, Scan>=9) reach the
  // re-mapping below; for other control flow nodes (e.g. If, or Scan opset 8) there is no such positional
  // mapping and nothing is accumulated here.
  if (IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs(parent_node)) {
    // The parent node's explicit inputs map positionally onto the subgraph's declared inputs.
    //
    // GetInputs() (inputs that are not backed by an initializer) is the list every downstream Loop/Scan path
    // validates and indexes against - see Loop::Info, scan::detail::Info and scan_8/scan_9's
    // ValidateSubgraphInput - so it is also the list used for the re-mapping here. Those downstream checks are
    // ORT_ENFORCE based, which abort() in ORT_NO_EXCEPTIONS builds, so the mismatch has to be rejected with a
    // clean status here before we get that far.
    //
    // A malformed/hostile model can declare a subgraph input that is also an initializer of the subgraph. Such
    // an input is dropped from GetInputs() while remaining in GetInputsIncludingInitializers(), making
    // GetInputs() shorter than the parent's input list. Without this check, indexing the shorter vector by the
    // parent's input index below is an out-of-bounds read.
    const auto num_parent_inputs = parent_node.InputDefs().size();
    const auto& subgraph_inputs = subgraph.GetInputs();
    if (subgraph_inputs.size() != num_parent_inputs) {
      const char* initializer_hint =
          subgraph.GetInputsIncludingInitializers().size() != subgraph_inputs.size()
              ? " The subgraph declares a graph input that is also one of its initializers, which is not supported"
                " for the body of a Loop or Scan node."
              : "";
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                             "Subgraph input count does not match the number of inputs provided by the parent node '",
                             parent_node.Name(), "' (OpType: ", parent_node.OpType(), "). Parent provides ",
                             num_parent_inputs, " inputs but the subgraph has ", subgraph_inputs.size(),
                             " inputs that are not also initializers.", initializer_hint);
    }

    auto process_input = [&plan, &ort_value_name_to_idx_map, &outer_scope_arg_to_location_map,
                          &subgraph_inputs](const NodeArg& input, size_t arg_idx) {
      const auto& name = input.Name();
      OrtValueIndex index = -1;
      ORT_RETURN_IF_ERROR(Index(ort_value_name_to_idx_map, name, index));

      // Store the location of the outer scope value in the map using the subgraph input as the key
      // as that will be the referenced name in the subgraph (i.e.) re-mapping of names is required
      outer_scope_arg_to_location_map.insert({subgraph_inputs[arg_idx]->Name(), plan.GetLocation(index)});

      return Status::OK();
    };

    return Node::ForEachWithIndex(parent_node.InputDefs(), process_input);
  }

  return Status::OK();
}

// We accumulate all nested subgraph(s) kernel create info maps relative to the current depth
// (i.e.) if we were on the first nested subgraph, we accumulate information from ALL the
// nested subgraphs within it.
// This information is necessary to plan the right location for initializers
// in a given level because they could be used in one of the nested subgraphs relative to the
// current level (not just within the same level or even one level deep).
// Since we need to package up information from multiple levels of nested subgraphs, the key we use
// is "{key_for_node_containing_subgraph} + current_depth + node_index_containing_the_subgraph + attribute_name".
// {key_for_node_containing_subgraph} is empty for the main graph.

// For example, if we want to store information corresponding to a nested subgraph wrt to the main graph and
// the node index  of the node in the main graph was 2 and the attribute containing the specific
// subgraph was "then_branch", the key would be depth + node_index + attribute = 0 + 2 + then_branch
// = "02then_branch".

// If that subgraph contained another subgraph at node index 1, then the key would be,
// {02then_branch} + 1 + 1 + "then_branch" = "02then_branch11then_branch".

static void AccumulateAllNestedSubgraphsInfo(
    const SessionState& session_state,
    const std::string& subgraph_kernel_create_info_map_key_base,
    size_t graph_depth,
    /*out*/ SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps) {
  for (const auto& entry : session_state.GetSubgraphSessionStateMap()) {
    auto node_index = entry.first;

    for (const auto& name_to_subgraph_session_state : entry.second) {
      const auto& subgraph_attr_name = name_to_subgraph_session_state.first;

      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;

      const auto& local_subgraph_kernel_create_info_map_key =
          NestedSubgraphInfoDetails::ComposeNestedSubgraphInfoKeyHelper(subgraph_kernel_create_info_map_key_base,
                                                                        graph_depth, node_index, subgraph_attr_name);

      // The end user is never likely to see an error with the following line.
      // Points to an internal processing error if we hit this.
      ORT_ENFORCE(subgraphs_kernel_create_info_maps.find(local_subgraph_kernel_create_info_map_key) ==
                  subgraphs_kernel_create_info_maps.end());

      subgraphs_kernel_create_info_maps.insert({local_subgraph_kernel_create_info_map_key,
                                                subgraph_session_state.GetKernelCreateInfoMap()});

      // Recurse into the subgraph session state
      AccumulateAllNestedSubgraphsInfo(subgraph_session_state,
                                       local_subgraph_kernel_create_info_map_key,
                                       graph_depth + 1, subgraphs_kernel_create_info_maps);
    }
  }
}

Status SessionState::FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                              const KernelRegistryManager& kernel_registry_manager,
                                              _In_opt_ const Node* parent_node,
                                              const SessionOptions& session_options,
                                              bool remove_initializers,
                                              bool save_prepacked_initializers,
                                              InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
                                              const MaxShapeInferenceResult& max_shape_inference_result,
                                              const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_node_arg_to_location_map,
                                              bool graph_info_already_created) {
  if (!graph_info_already_created) {
    CreateGraphInfo(save_prepacked_initializers);
  }

#if defined(ORT_EXTENDED_MINIMAL_BUILD)
  // Remove any unused initializers.
  // Not needed in a full build because unused initializers should have been removed earlier by Graph::Resolve().
  // Not needed in a basic minimal build because only runtime optimizations are expected to possibly result in unused
  //   initializers and they are only enabled in an extended minimal build.
  {
    InlinedVector<std::string> unused_initializer_names;
    for (const auto& [name, tensor_proto] : graph_.GetAllInitializedTensors()) {
      ORT_UNUSED_PARAMETER(tensor_proto);
      int idx;
      if (!ort_value_name_idx_map_.GetIdx(name, idx).IsOK()) {
        unused_initializer_names.push_back(name);
      }
    }

    for (const auto& name : unused_initializer_names) {
      graph_.RemoveInitializedTensor(name);
    }
  }
#endif  // defined(ORT_EXTENDED_MINIMAL_BUILD)

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  InlinedVector<const NodeArg*> valid_outer_scope_node_args;
  if (parent_node) {
    auto outer_scope_node_args = parent_node->ImplicitInputDefs();
    valid_outer_scope_node_args.reserve(outer_scope_node_args.size());

    std::for_each(outer_scope_node_args.cbegin(), outer_scope_node_args.cend(),
                  [this, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                    int idx;
                    if (ort_value_name_idx_map_.GetIdx(node_arg->Name(), idx).IsOK()) {
                      valid_outer_scope_node_args.push_back(node_arg);
                    };
                  });
  }

  // TODO: we avoid instantiate it in subgraph session state

  // register stream handles from EP instances
#ifdef ORT_ENABLE_STREAM
  auto& eps = GetExecutionProviders();
  for (auto& ep : eps) {
    ep->RegisterStreamHandlers(GetStreamHandleRegistryInstance(), *allocators_);
  }
#endif

  SubgraphsKernelCreateInfoMaps subgraphs_kernel_create_info_maps;
  AccumulateAllNestedSubgraphsInfo(*this, "", 0, subgraphs_kernel_create_info_maps);

  SequentialPlannerContext context(session_options.execution_mode,
                                   session_options.execution_order,
                                   session_options.enable_mem_reuse);

#ifdef _WIN32

  PathString partition_config_file =
      ToWideString(session_options.config_options.GetConfigOrDefault(
          kNodePartitionConfigFile, ""));

#else

  PathString partition_config_file =
      session_options.config_options.GetConfigOrDefault(
          kNodePartitionConfigFile, "");

#endif

  auto status = SequentialPlanner::CreatePlan(parent_node, *graph_viewer_, valid_outer_scope_node_args,
                                              execution_providers_, kernel_create_info_map_,
                                              subgraphs_kernel_create_info_maps,
                                              outer_scope_node_arg_to_location_map,
                                              ort_value_name_idx_map_, context,
#ifdef ORT_ENABLE_STREAM
                                              GetStreamHandleRegistryInstance(),
#endif
                                              partition_config_file,
                                              Logger(),
                                              p_seq_exec_plan_);
  ORT_RETURN_IF_ERROR(status);

  if (session_options.IsLoadCancellationFlagSet()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                           "SessionState finalize is canceled due to user request");
  }

  moe_offload::MoeOffloadSessionPolicy moe_offload_policy;
  ORT_RETURN_IF_ERROR(moe_offload::MoeOffloadSessionPolicy::ParseFromConfigOptions(
      session_options.config_options, std::filesystem::path(graph_location), moe_offload_policy));
  if (moe_offload_policy.mode != moe_offload::OffloadMode::kDisabled && !moe_offload::kExpertRelocationBackendAvailable) {
    // Relocating expert weights to CPU is only valid when the consuming kernel understands offloaded weights;
    // otherwise it would hand host pointers to GPU execution.
    if (moe_offload_policy.mode == moe_offload::OffloadMode::kRequired) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "MoE offload mode='required' is not available: no offload-capable MoE/QMoE kernel "
                             "backend is present in this build.");
    }
    LOGS(logger_, INFO) << "MoE offload mode='auto' ignored: no offload-capable MoE/QMoE kernel backend is present.";
  } else if (moe_offload_policy.mode != moe_offload::OffloadMode::kDisabled) {
    auto cpu_alloc = GetAllocator(OrtDevice());
    ORT_ENFORCE(cpu_alloc != nullptr, "Default CPU allocator is required for MoE offload initializer placement.");
    ORT_RETURN_IF_ERROR(PlaceMoeOffloadInitializersOnCpu(
        *graph_viewer_, moe_offload_policy, ort_value_name_idx_map_, cpu_alloc->Info(), *p_seq_exec_plan_, logger_));
  }

  // Record the allocation plan

  // Uncomment the below to dump the allocation plan to std::cout
  // std::cout << std::make_pair(&*p_seq_exec_plan_, this);

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  GetMemoryProfiler()->Init(GetExecutionPlan(), GetOrtValueNameIdxMap());
#endif

  // Note: For Training Prepacking should be always disabled.
  // For inference it is enabled by default, but users can choose to disable it via session options.
  const bool disable_prepacking =
      session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisablePrepacking, "0") == "1";
  // Memory pattern tracer allocates all initializers on a single contiguous
  // buffer. This has the effect of reducing memory fragmentation.
  // Further more, in training scenarios NCCL kernels require initializers to be allocated
  // contiguously.
  //
  // In inferencing scenarios, however, we often want to pre-process and then
  // release some initializers. See OpKernel::PrePack(). Letting all initializers
  // sharing a single buffer makes it hard to release individual ones, leading
  // to memory waste.
  //
  // TODO!! disabling memory pattern tracer increases fragmentation, leading to
  //  out of memory error in some training tests. Need to create kernel first,
  //  and let the kernel tells us whether the initializer needs to be traced.
  //
  std::unique_ptr<ITensorAllocator> tensor_allocator = nullptr;
  if (disable_prepacking) {
    tensor_allocator = ITensorAllocator::Create(enable_mem_pattern_, *p_seq_exec_plan_, *this, weights_buffers_);
  } else {
    tensor_allocator = ITensorAllocator::Create(false, *p_seq_exec_plan_, *this, weights_buffers_);
  }

  const auto& initializer_allocation_order = p_seq_exec_plan_->initializer_allocation_order;

  // move initializers from TensorProto instances in Graph to OrtValue instances in SessionState
  session_state_utils::MemoryProfileFunction memory_profile_func = nullptr;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  memory_profile_func = [this](ITensorAllocator& planner) {
    GetMemoryProfiler()->GetMemoryInfo().RecordPatternInfo(
        planner.GetMemPatterns(), MemoryInfo::MapType::Initializer);
    GetMemoryProfiler()->CreateEvents(
        "initializer_" + std::to_string(GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
        GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::Initializer, "", 0);
  };

#endif

#ifdef ORT_ENABLE_STREAM
  // set the has_device_stream_enabled_ep_ flag
  has_device_stream_enabled_ep_ = false;
  if (p_seq_exec_plan_.has_value()) {
    auto& execution_plan = (*p_seq_exec_plan_).execution_plan;
    for (size_t i = 0; i < execution_plan.size(); ++i) {
      auto& logic_stream = execution_plan[i];
      if (logic_stream->steps_.size() > 0) {
        auto create_stream_fn = GetStreamHandleRegistryInstance().GetCreateStreamFn(logic_stream->device_.Type());
        if (create_stream_fn) {
          has_device_stream_enabled_ep_ = true;
        }
      }
    }
  }
#endif

  ORT_RETURN_IF_ERROR(session_state_utils::SaveInitializedTensors(
      Env::Default(), graph_location, *graph_viewer_,
      GetAllocator(OrtDevice()),
      ort_value_name_idx_map_, initializer_allocation_order, *tensor_allocator,
      [this, remove_initializers](const std::string& name, int idx, const OrtValue& value,
                                  bool constant, bool sparse) -> Status {
        ORT_RETURN_IF_ERROR(AddInitializedTensor(idx, value, constant, sparse));
        if (remove_initializers) {
          graph_.RemoveInitializedTensor(name);
        }
        return Status::OK();
      },
      logger_, data_transfer_mgr_, external_data_loader_mgr_, *p_seq_exec_plan_, session_options,
      memory_profile_func, graph_.GetPrepacked()));

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Record Weight allocation info on device
  GetMemoryProfiler()->GetMemoryInfo().RecordInitializerAllocInfo(GetInitializedTensors());
#endif

  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  if (remove_initializers) {
    CleanInitializedTensorsFromGraph();
  }

  // For subgraph session states: build a merged constant-tensor map (outer_scope_augmented_constant_tensors_)
  // that extends this subgraph's own constant_initialized_tensors_ with any outer-scope constant initializers
  // from parent graphs, re-indexed into this subgraph's ort_value_name_idx_map_. This allows
  // TryGetConstantInput (called from kernel constructors and PrePack) to resolve constants that live in
  // parent-graph scope, such as the scales/zero_points paired with a MatMulNBits B tensor when all three
  // initializers belong to the outer graph but the MatMulNBits node is inside a subgraph (e.g. If branch).
  // The augmented map is only used for kernel creation; PrepackConstantInitializedTensors keeps using the
  // unaugmented constant_initialized_tensors_ to avoid double-prepacking outer-scope tensors.
  // After the subgraph finalization loop below, the parent-scope OrtValue copies in this map are erased
  // (via outer_scope_parent_only_indices_) to allow the parent's constant_initialized_tensors_ to release
  // memory once all prepack use counts reach zero.
  if (parent_node != nullptr && parent_ != nullptr) {
    outer_scope_augmented_constant_tensors_ = constant_initialized_tensors_;

    for (const NodeArg* outer_arg : parent_node->ImplicitInputDefs()) {
      const std::string& name = outer_arg->Name();

      // Get the index for this name in the current (subgraph) scope.
      int current_idx = -1;
      if (!ort_value_name_idx_map_.GetIdx(name, current_idx).IsOK()) {
        continue;
      }

      // Skip if this name is already covered by the subgraph's own constants.
      if (constant_initialized_tensors_.count(current_idx) > 0) {
        continue;
      }

      // Walk the parent session state chain (handles deeply nested subgraphs) until we either
      // find the constant or exhaust all ancestors.
      const SessionState* p = parent_;
      while (p != nullptr) {
        int parent_idx = -1;
        if (!p->ort_value_name_idx_map_.GetIdx(name, parent_idx).IsOK()) {
          p = p->parent_;
          continue;
        }

        // Use the parent's already-augmented map so that constants from grandparent scopes
        // are also visible (the parent's FinalizeSessionStateImpl has already run).
        const auto& parent_const = p->GetConstantInitializedTensorsForKernelCreation();
        auto it = parent_const.find(parent_idx);
        if (it != parent_const.end()) {
          outer_scope_augmented_constant_tensors_.emplace(current_idx, it->second);
          outer_scope_parent_only_indices_.insert(current_idx);
          break;
        }

        p = p->parent_;
      }
    }

    outer_scope_augmented_map_built_ = true;
  }

  ORT_RETURN_IF_ERROR(CreateKernels(kernel_registry_manager));

  if (!disable_prepacking) {
    ORT_RETURN_IF_ERROR(PrepackConstantInitializedTensors(constant_initializers_use_count,
                                                          session_options.initializers_to_share_map));
  }

#if !defined(ORT_MINIMAL_BUILD)
  // Level-2 workspace declaration: after kernels are created and PrePack'd, call
  // DeclareWorkspaceRequirements() on each kernel with positional input presence/shape metadata.
  // Static graph shapes remain usable when no max-shape inference result is available.
  // This collects workspace slot requirements for future offset planning.
  // Requirements are reported per graph but are not yet persisted in a workspace plan.
  {
    SafeInt<size_t> aggregate_declared_workspace_bytes = 0;
    size_t nodes_with_workspace = 0;
    size_t compared_nodes = 0;
    size_t level2_larger = 0;
    size_t reservation_larger = 0;
    size_t equal = 0;
    size_t missing_reservation = 0;
    size_t missing_declaration = 0;
    size_t orphaned_reservations = 0;
    size_t orphaned_workspace_reservations = 0;
    SafeInt<size_t> level2_excess_bytes = 0;
    SafeInt<size_t> reservation_excess_bytes = 0;
    SafeInt<size_t> orphaned_reservation_bytes = 0;
    bool strict_verification_failed = false;
    const bool strict_workspace_verification =
        session_options.config_options.GetConfigOrDefault(
            kOrtSessionOptionsStrictWorkspaceVerification, "0") == "1";

    const SessionState* reservation_owner = this;
    while (reservation_owner->parent_ != nullptr) {
      reservation_owner = reservation_owner->parent_;
    }
    const auto graph_reservations_it = reservation_owner->workspace_reservations_.find(&graph_);
    const NodeWorkspaceReservationMap* graph_reservations =
        graph_reservations_it == reservation_owner->workspace_reservations_.end()
            ? nullptr
            : &graph_reservations_it->second;

    for (const auto& node : graph_viewer_->Nodes()) {
      const WorkspaceEstimateSelection* reservation = nullptr;
      if (graph_reservations != nullptr) {
        const auto reservation_it = graph_reservations->find(node.Index());
        if (reservation_it != graph_reservations->end()) {
          reservation = &reservation_it->second;
        }
      }

      auto* kernel = GetMutableKernel(node.Index());
      if (kernel == nullptr) {
        if (reservation != nullptr) {
          ++missing_declaration;
        }
        continue;
      }

      auto resolved = ResolveNodeInputShapes(node, &graph_, max_shape_inference_result);

      InlinedVector<WorkspaceRequirement> requirements;
      ORT_RETURN_IF_ERROR(kernel->DeclareWorkspaceRequirements(
          gsl::make_span(resolved), requirements));

      if (requirements.empty()) {
        if (reservation != nullptr) {
          ++missing_declaration;
        }
        continue;
      }

      SafeInt<size_t> node_workspace = 0;
      for (const auto& req : requirements) {
        node_workspace += req.size_bytes;
        LOGS(logger_, VERBOSE) << "Level-2 workspace: node '" << node.Name()
                               << "' slot_id=" << req.slot_id
                               << " size=" << req.size_bytes << " bytes"
                               << " alignment=" << req.alignment_bytes << " bytes";
      }
      const size_t declared_bytes = static_cast<size_t>(node_workspace);
      aggregate_declared_workspace_bytes += node_workspace;
      ++nodes_with_workspace;
      LOGS(logger_, VERBOSE) << "Level-2 workspace: node '" << node.Name()
                             << "' (" << node.OpType() << "): " << requirements.size()
                             << " slot(s), " << declared_bytes << " bytes total";

      if (reservation == nullptr) {
        ++missing_reservation;
        continue;
      }

      ++compared_nodes;
      if (declared_bytes > reservation->bytes) {
        ++level2_larger;
        level2_excess_bytes += declared_bytes - reservation->bytes;
        strict_verification_failed = strict_verification_failed || strict_workspace_verification;
        LOGS(logger_, WARNING) << "Level-2 workspace declaration for node '" << node.Name()
                               << "' (" << node.OpType() << ") is " << declared_bytes
                               << " bytes, exceeding its partitioning reservation of "
                               << reservation->bytes << " bytes";
      } else if (declared_bytes < reservation->bytes) {
        ++reservation_larger;
        reservation_excess_bytes += reservation->bytes - declared_bytes;
      } else {
        ++equal;
      }
    }

    if (graph_reservations != nullptr) {
      for (const auto& [node_index, reservation] : *graph_reservations) {
        if (graph_.GetNode(node_index) == nullptr) {
          ++orphaned_reservations;
          orphaned_reservation_bytes += reservation.bytes;
          if (reservation.bytes > 0) {
            ++orphaned_workspace_reservations;
          }
        }
      }
    }

    const bool strict_reservation_ownership_failed =
        strict_workspace_verification &&
        orphaned_workspace_reservations > 0;
    if (strict_reservation_ownership_failed) {
      LOGS(logger_, WARNING)
          << "Level-2 workspace verification found " << orphaned_workspace_reservations
          << " nonzero reservation(s) for removed nodes";
    }

    if (nodes_with_workspace > 0 || missing_declaration > 0 || orphaned_reservations > 0) {
      LOGS(logger_, INFO) << "Level-2 workspace declaration summary for graph '"
                          << graph_viewer_->Name() << "': "
                          << nodes_with_workspace << " kernel(s) declared workspace, "
                          << "aggregate declared workspace bytes: "
                          << static_cast<size_t>(aggregate_declared_workspace_bytes)
                          << ", compared=" << compared_nodes
                          << ", Level-2 larger=" << level2_larger
                          << ", reservation larger=" << reservation_larger
                          << ", equal=" << equal
                          << ", missing reservation=" << missing_reservation
                          << ", missing declaration=" << missing_declaration
                          << ", orphaned reservation=" << orphaned_reservations
                          << ", orphaned nonzero reservation=" << orphaned_workspace_reservations
                          << ", Level-2 excess=" << static_cast<size_t>(level2_excess_bytes) << " bytes"
                          << ", reservation excess=" << static_cast<size_t>(reservation_excess_bytes) << " bytes"
                          << ", orphaned reservation bytes="
                          << static_cast<size_t>(orphaned_reservation_bytes) << " bytes";
    }
    ORT_RETURN_IF(
        strict_verification_failed || strict_reservation_ownership_failed,
        "Level-2 workspace verification failed: one or more declarations exceed "
        "the workspace reserved during graph partitioning, or a post-partition graph transformation "
        "orphaned a nonzero workspace reservation.");
  }
#endif

  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInputOutputNamesToNodeMapping(*graph_viewer_, *this, valid_outer_scope_node_args));

  // Need to recurse into subgraph session state instances to finalize them and add the execution info

  // Currently all subgraphs need to be executed using the sequential EP due to potential deadlock with the current
  // parallel executor implementation
  SessionOptions subgraph_session_options(session_options);
  subgraph_session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;

  for (const auto& node_to_subgraph_ss : subgraph_session_states_) {
    Node& node = *graph_.GetNode(node_to_subgraph_ss.first);

    for (const auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& attr_name = attr_subgraph_pair.first;
      auto entry = node_to_subgraph_ss.second.find(attr_name);
      // CreateSubgraphSessionState should ensure all these entries are created
      ORT_ENFORCE(entry != node_to_subgraph_ss.second.cend(),
                  "Missing session state for subgraph. Node:'", node.Name(),
                  "' OpType:", node.OpType(), " Index:", node.Index(), " Attribute:", attr_name);

      SessionState& subgraph_session_state = *entry->second;

      // recurse

      // We need to create graph info for the subgraphs because information accumulated there
      // is used in OuterScopeNodeArgLocationAccumulator()
      subgraph_session_state.CreateGraphInfo(save_prepacked_initializers);

      InlinedHashMap<OrtValueName, OrtDevice> subgraph_outer_scope_node_arg_to_location_map;
      ORT_RETURN_IF_ERROR(OuterScopeNodeArgLocationAccumulator(*p_seq_exec_plan_, GetOrtValueNameIdxMap(),
                                                               node,
                                                               subgraph_session_state.GetGraphViewer(),
                                                               subgraph_outer_scope_node_arg_to_location_map));

      ORT_RETURN_IF_ERROR(subgraph_session_state.FinalizeSessionStateImpl(
          graph_location, kernel_registry_manager, &node, subgraph_session_options, remove_initializers,
          save_prepacked_initializers,
          constant_initializers_use_count, max_shape_inference_result,
          subgraph_outer_scope_node_arg_to_location_map, true));

      // setup all the info for handling the feeds and fetches used in subgraph execution
      auto* p_op_kernel = GetMutableKernel(node.Index());
      ORT_ENFORCE(p_op_kernel);

      // A node carries a subgraph whenever it has a GRAPH-typed attribute, which Node::Init
      // materializes with no schema gate. The downcast below is only valid for control flow
      // kernels, so reject any other kernel that reached here with a subgraph instead of
      // reading an out-of-bounds vtable slot.
      ORT_RETURN_IF_NOT(p_op_kernel->IsControlFlowKernel(), "Node '", node.Name(),
                        "' (OpType: ", node.OpType(),
                        ") has a subgraph but is not a control flow node.");

      // Downcast is safe: only control flow nodes reach here (guarded above), and
      // node.GetAttributeNameToMutableSubgraphMap() is non-empty.
      auto& control_flow_kernel = static_cast<controlflow::IControlFlowKernel&>(*p_op_kernel);
      ORT_RETURN_IF_ERROR(control_flow_kernel.SetupSubgraphExecutionInfo(*this, attr_name, subgraph_session_state));
    }

    // TODO: Once the subgraph session states have been finalized, can we go back and plan the location of implicit
    // inputs that are fed through as graph inputs in the graph level holding the subgraphs ? Ideally the planned
    // locations for these would be the locations they are explicitly consumed on in nested subgraphs.
  }

  // Release the extra OrtValue copies from parent scope that were added to the augmented constant
  // map for this subgraph session state. These copies were needed to let TryGetConstantInput resolve
  // parent-scope constants during CreateKernels and PrepackConstantInitializedTensors above, and to
  // let any nested (grandchild) subgraph session states read them via GetConstantInitializedTensorsForKernelCreation().
  // Now that all subgraphs are fully finalized, the copies are no longer needed. Releasing them lets
  // the parent's constant_initialized_tensors_ free the underlying tensor buffers once all prepack
  // use counts reach zero, avoiding a memory regression for large initializers such as quantized B
  // matrices and scales in MatMulNBits subgraph patterns.
  for (int idx : outer_scope_parent_only_indices_) {
    outer_scope_augmented_constant_tensors_.erase(idx);
  }
  outer_scope_parent_only_indices_.clear();

  return Status::OK();
}

#ifdef ORT_ENABLE_STREAM
static void BindToDeviceStream(const SequentialExecutionPlan& execution_plan,
                               DeviceStreamCollection& device_stream_map,
                               IStreamCommandHandleRegistry& stream_handle_registry) {
  for (size_t i = 0; i < execution_plan.execution_plan.size(); ++i) {
    auto& logic_stream = execution_plan.execution_plan[i];
    if (logic_stream->steps_.size() > 0) {
      auto create_stream_fn = stream_handle_registry.GetCreateStreamFn(logic_stream->device_.Type());
      if (create_stream_fn) {
        auto device_stream = create_stream_fn(logic_stream->device_);
        device_stream_map.AddDeviceStream(i, std::move(device_stream));
      } else {
        device_stream_map.SetDeviceStream(i, nullptr);
      }
    } else {
      device_stream_map.SetDeviceStream(i, nullptr);
    }
  }
}

std::unique_ptr<DeviceStreamCollection> SessionState::AcquireDeviceStreamCollection() const {
  if (has_device_stream_enabled_ep_) {
    const auto& thread_token = GetDeviceStreamPoolThreadToken();
    const void* thread_key = thread_token.get();

    std::lock_guard<std::mutex> lock(device_stream_pool_mutex_);
    PruneExpiredDeviceStreamPoolsLocked();

    auto it = device_stream_pools_.find(thread_key);
    if (it != device_stream_pools_.end() && !it->second.device_streams.empty()) {
      auto device_stream = std::move(it->second.device_streams.back());
      it->second.device_streams.pop_back();
      if (it->second.device_streams.empty()) {
        device_stream_pools_.erase(it);
      }
      return device_stream;
    }

    auto device_stream = std::make_unique<DeviceStreamCollection>(this->GetExecutionPlan()->execution_plan.size(), *allocators_, graph_viewer_->ParentNode() == nullptr);
    BindToDeviceStream(*this->GetExecutionPlan(), *device_stream, *stream_handles_registry_);
    return device_stream;
  } else {
    // no reusing of device stream is needed, just return nullptr, the caller will handle it
    return nullptr;
  }
}

void SessionState::RecycleDeviceStreamCollection(std::unique_ptr<DeviceStreamCollection> device_stream_collection) const {
  // if no need to reuse the device stream, don't perform the recycle
  if (has_device_stream_enabled_ep_) {
    const auto& thread_token = GetDeviceStreamPoolThreadToken();
    const void* thread_key = thread_token.get();

    std::lock_guard<std::mutex> lock(device_stream_pool_mutex_);
    PruneExpiredDeviceStreamPoolsLocked();
    auto& bucket = device_stream_pools_[thread_key];
    bucket.thread_token = thread_token;
    bucket.device_streams.push_back(std::move(device_stream_collection));
  } else {
    device_stream_collection.reset(nullptr);
  }
}

void SessionState::PruneExpiredDeviceStreamPoolsLocked() const {
  for (auto it = device_stream_pools_.begin(); it != device_stream_pools_.end();) {
    if (it->second.thread_token.expired()) {
      auto expired_it = it++;
      device_stream_pools_.erase(expired_it);
    } else {
      ++it;
    }
  }
}
#endif

}  // namespace onnxruntime
