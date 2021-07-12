//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/callback.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/framework_common.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/path_lib.h"
#include "core/platform/threadpool.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {

namespace experimental {
namespace fbs {
struct SessionState;
}  // namespace fbs
}  // namespace experimental

class ExecutionProviders;
class KernelDef;
class OpKernel;
class NodeIndexInfo;
struct SequentialExecutionPlan;
struct MemoryPatternGroup;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
class MemoryInfo;
#endif

/**
 * SessionState should be modified by the inference session class only.
 * It is supposed to be passed by const-ref only to all the executors.
 * This class owns all the initializers.
 * Brief usage:
 *   SessionState s(...);
 *   <process subgraphs to populate subgraph SessionState instances>
 *   <run transformers or any other graph editing steps>
 *   for(...) // copy initializers from GraphProto format in Graph to OrtValue format in SessionState
        s.AddInitializedTensor(...);
 *   s.CleanInitializedTensorsFromGraph(); // remove GraphProto instances from Graph if not needed
 *
 *   s.CreateGraphInfo();
 *   s.CreateKernels(...);
 * Then you can use:
 *   s.GetKernel(...);
 */
class SessionState {
 public:
  SessionState(Graph& graph,
               const ExecutionProviders& execution_providers,
               bool enable_mem_pattern,
               concurrency::ThreadPool* thread_pool,
               concurrency::ThreadPool* inter_op_thread_pool,
               const DataTransferManager& data_transfer_mgr,
               const logging::Logger& logger,
               profiling::Profiler& profiler,
               bool use_deterministic_compute = false,
               bool enable_mem_reuse = true,
               PrepackedWeightsContainer* prepacked_weights_container = nullptr)
      : graph_(graph),
        execution_providers_(execution_providers),
        logger_(logger),
        profiler_(profiler),
        enable_mem_pattern_(enable_mem_pattern),
        thread_pool_(thread_pool),
        inter_op_thread_pool_(inter_op_thread_pool),
        data_transfer_mgr_(data_transfer_mgr),
        use_deterministic_compute_(use_deterministic_compute),
        enable_mem_reuse_(enable_mem_reuse),
        prepacked_weights_container_(prepacked_weights_container) {
    SetupAllocators();
  }

  ~SessionState() {
    for (auto* p : session_kernels_) {
      delete p;
    }
    for (auto& kvp : deleter_for_initialized_tensors_) {
      kvp.second.f(kvp.second.param);
    }
  }

  // Graph viewer. CreateGraphInfo must have been called previously.
  const GraphViewer& GetGraphViewer() const noexcept { return *graph_viewer_.get(); };

  // kernels
  // Get kernel for specified node.
  // It should called right before graph execution only.
  const OpKernel* GetKernel(size_t node_id) const {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id] : nullptr;
  }

  OpKernel* GetMutableKernel(size_t node_id) {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id] : nullptr;
  }

  const ExecutionProviders& GetExecutionProviders() const noexcept { return execution_providers_; }

  /**
    Get the allocator for the given OrtMemoryInfo location
    */
  AllocatorPtr GetAllocator(const OrtMemoryInfo& location) const noexcept;

  /** Get the allocator for a given OrtDevice. The first allocator that matches will be returned. */
  AllocatorPtr GetAllocator(OrtDevice device) const noexcept;

  const OrtValueNameIdxMap& GetOrtValueNameIdxMap() const noexcept { return ort_value_name_idx_map_; }

  /**
     * Adds an initialized tensor (weight) so that it can be used by the
     * execution frame to setup the appropriate OrtValue vectors.
     * This function will take a shallow copy of d if d is not NULL.
     * If 'constant' is true the tensor value cannot be overridden by an input at runtime.
     * If 'sparse' is true the tensor value represents a densified weight that was initially stored in the model
     * as sparse tensor.
     */
  Status AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d, bool constant, bool sparse);

  /**
     * Gets the map of ort_value_index to initialized tensors (weights) so that it can be used by the
     * execution frame to setup the appropriate OrtValue vectors.
     * The lifetime of returned OrtValues are limited by this SessionState object.
     */
  const std::unordered_map<int, OrtValue>& GetInitializedTensors() const;

  /**
     * Gets the map of ort_value_index to initialized tensors (e.g. weights) that are constant
     * and cannot be overridden at runtime.
     * The lifetime of returned OrtValues are limited by this SessionState object.
     */
  const std::unordered_map<int, OrtValue>& GetConstantInitializedTensors() const;

  bool IsSparseInitializer(int ort_value_index) const;

#ifdef ENABLE_TRAINING
  /**
    Get some initialized tensors (weights).
    @param interested_weights The names of the weights to retrieve.
    @param allow_missing_weights Whether to allow names in interested_weights
           with no corresponding weight.
    @param[out] retrieved_weights The retrieved weights.
    @return The status of the operation.
    */
  Status GetInitializedTensors(
      const std::unordered_set<std::string>& interested_weights,
      bool allow_missing_weights, NameMLValMap& retrieved_weights) const;

  /**
    Get some initialized tensors (weights).
    Any names in interested_weights with no corresponding weight are ignored.
    */
  NameMLValMap GetInitializedTensors(const std::unordered_set<std::string>& interested_weights) const;
#endif

  // execution plan. nullptr until FinalizeSessionState is called
  const SequentialExecutionPlan* GetExecutionPlan() const;
  /**
  Get the logger for this session.
  Falls back to returning Logging::LoggingManager::DefaultLogger if SetLogger has not been called.
  */
  const logging::Logger& Logger() const noexcept { return logger_; }

  /**
  Get the profiler for this session. It needs to be enabled via the InferenceSession to perform
  profiling actions.
  */
  profiling::Profiler& Profiler() const noexcept { return profiler_; }

  /**
  Get cached memory pattern based on input shapes
  */
  const MemoryPatternGroup* GetMemoryPatternGroup(
      const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
      const std::vector<int>& feed_mlvalue_idxs,
      std::unordered_map<int, TensorShape>& inferred_shapes) const;

  /**
  Set generated memory pattern with a given input shapes.
  Const as it's an internal cache update only.
  */
  Status UpdateMemoryPatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
                                       std::unique_ptr<MemoryPatternGroup> mem_patterns) const;

  bool GetUseDeterministicCompute() const { return use_deterministic_compute_; }

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

  /**
  Get enable memory re-use flag.
  */

  bool GetEnableMemoryReuse() const;

  /**
  Update enable_mem_pattern_ flag according to the presence of graph inputs' shape
  If any one of the graph input is shapeless, enable_mem_pattern_ will be set to false
  */
  void ResolveMemoryPatternFlag();

  struct NodeInfo {
    /**
     *
     * \param index0
     * \param p_node0 Nullable
     * \param kci0 Nullable
     */
    NodeInfo(size_t index0, const onnxruntime::Node* p_node0, const KernelCreateInfo* kci0, const OrtDevice& device0)
        : index(index0), p_node(p_node0), kci(kci0), device(&device0) {}

    size_t index;
    // Nullable
    const onnxruntime::Node* p_node = nullptr;
    // Nullable
    const KernelCreateInfo* kci = nullptr;
    const OrtDevice* device = nullptr;
  };

  using NameNodeInfoMapType = std::unordered_map<std::string, std::vector<NodeInfo>>;

  common::Status AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info);
  common::Status GetInputNodeInfo(const std::string& input_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetInputNodeInfoMap() const;

  void AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info);
  common::Status GetOutputNodeInfo(const std::string& output_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetOutputNodeInfoMap() const;

  // Get the KernelCreateInfo entry for a node. SessionState must be finalized before calling.
  const KernelCreateInfo& GetNodeKernelCreateInfo(NodeIndex node_index) const;

  /// Return SessionState for the given Node index and attribute name if found.
  const SessionState* GetSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name) const;

  concurrency::ThreadPool* GetThreadPool() const noexcept { return thread_pool_; }
  concurrency::ThreadPool* GetInterOpThreadPool() const noexcept { return inter_op_thread_pool_; }

  bool ExportDll() const noexcept { return export_fused_dll_; }
  void SetExportDllFlag(bool flag) noexcept { export_fused_dll_ = flag; }

  const FuncManager& GetFuncMgr() const noexcept { return fused_funcs_mgr_; }
  FuncManager& GetMutableFuncMgr() noexcept { return fused_funcs_mgr_; }

  const DataTransferManager& GetDataTransferMgr() const noexcept { return data_transfer_mgr_; }

  std::vector<BufferUniquePtr>& GetMutableWeightsBuffers() noexcept { return weights_buffers_; }

  const NodeIndexInfo& GetNodeIndexInfo() const;

#if !defined(ORT_MINIMAL_BUILD)
  void UpdateToBeExecutedNodes(const std::vector<int>& fetch_mlvalue_idxs);
  const std::unordered_set<NodeIndex>* GetToBeExecutedNodes(const std::vector<int>& fetch_mlvalue_idxs) const;
  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<onnxruntime::experimental::fbs::SessionState>& fbs_session_state) const;
#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)
  void SetCompiledKernelHashes(std::unordered_map<std::string, uint64_t>&& compiled_kernel_hashes) {
    compiled_kernel_hashes_ = std::move(compiled_kernel_hashes);
  }

  Status LoadFromOrtFormat(const onnxruntime::experimental::fbs::SessionState& fbs_session_state,
                           const KernelRegistryManager& kernel_registry_manager);
#endif

  Status FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                              KernelRegistryManager& kernel_registry_manager,
                              const SessionOptions& session_options = {},
                              const onnxruntime::experimental::fbs::SessionState* serialized_session_state = nullptr,
                              bool remove_initializers = true,
                              bool saving_ort_format = false);

  SessionState* Parent() {
    return parent_;
  }

  size_t GetNumberOfPrepacksCounter() const {
    return number_of_prepacks_counter_;
  }

  size_t GetUsedSharedPrePackedWeightCounter() const {
    return used_shared_pre_packed_weights_counter_;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionState);

  void SetupAllocators();

  // Populate OrtValueNameIdxMap and create the graph viewer.
  void CreateGraphInfo();

  // create kernels using info in kernel_create_info_map_
  Status CreateKernels(const KernelRegistryManager& custom_registry_manager);

  // remove TensorProto versions of initializers from Graph instance
  // (replaced byOrtValue instances in initialized_tensors_)
  void CleanInitializedTensorsFromGraph();

  /**
  * Prepack the constant initialized tensors for better performance.
  * The original constant initialized tensors will be removed to save memory.
  */
  Status PrepackConstantInitializedTensors(std::unordered_map<std::string, size_t>& constant_initializers_use_count,
                                           const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map);

  SessionState* GetMutableSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name);

  Status CreateSubgraphSessionState();

  void AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                               std::unique_ptr<SessionState> session_state);

#if !defined(ORT_MINIMAL_BUILD)
  Status PopulateKernelCreateInfo(KernelRegistryManager& kernel_registry_manager, bool saving_ort_format);
#endif

  Status FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                  KernelRegistryManager& kernel_registry_manager,
                                  _In_opt_ const Node* parent_node,
                                  const SessionOptions& session_options,
                                  bool remove_initializers,
                                  std::unordered_map<std::string, size_t>& constant_initializers_use_count);

#ifdef ENABLE_TRAINING
  Status GeneratePatternGroupCache(
      const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
      const std::vector<int>& feed_mlvalue_idxs,
      MemoryPatternGroup* output,
      std::unordered_map<int, TensorShape>& inferred_shapes) const;
#endif

  // the SessionState for the main Graph contains the compiled kernel hashes for the entire model
  const std::unordered_map<std::string, uint64_t>& GetCompiledKernelHashes() const {
    return parent_ ? parent_->GetCompiledKernelHashes() : compiled_kernel_hashes_;
  }

  // KernelCreateInfo for each node so we do kernel lookup once
  std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>> kernel_create_info_map_;

  // If we compile kernels in a minimal build we need a way to find the kernel using the hash.
  // We populate this map when doing the kernel compilation in GraphPartitioner, and use it in LoadFromOrtFormat.
  std::unordered_map<std::string, uint64_t> compiled_kernel_hashes_;

  // cache of the constructed kernels to avoid spending construction time per executor
  std::vector<OpKernel*> session_kernels_;
  Graph& graph_;
  std::unique_ptr<GraphViewer> graph_viewer_;  // GraphViewer for const access to Graph

  const ExecutionProviders& execution_providers_;

  // currently the allocator type is an implementation detail and we don't make any  behavioral choices based on it,
  // so exclude it from the key comparison for allocator_idx_map_.
  // we also don't expect to have two allocators with the same name, one using an arena and one not.
  struct OrtMemoryInfoLessThanIgnoreAllocType {
    bool operator()(const OrtMemoryInfo& lhs, const OrtMemoryInfo& rhs) const {
      //if (lhs.alloc_type != rhs.alloc_type)
      //  return lhs.alloc_type < rhs.alloc_type;
      if (lhs.mem_type != rhs.mem_type)
        return lhs.mem_type < rhs.mem_type;

      if (lhs.id != rhs.id)
        return lhs.id < rhs.id;

      return strcmp(lhs.name, rhs.name) < 0;
    }
  };

  // using std::map as OrtMemoryInfo would need a custom hash function to be used with std::unordered_map,
  // and as this isn't considered performance critical currently it's not worth the maintenance overhead of adding one.
  // We do get an allocator from ExecutionFrame so this is looked up frequently, however there most likely aren't many
  // entries in the map
  //
  // NOTE: We store a delegate to get the allocator to support scenarios such as the CUDA EP where a thread_local
  // allocator is returned.
  //
  // TODO: The CUDA EP may not need to use the per-thread allocator for allocations that would use this map
  // (e.g. primarily from ExecutionFrame and utils::Copy{Inputs|Outputs}AcrossDevices). It does need it
  // for internal allocations by CUDAExecutionProvider::GetScratchBuffer, but could access the per-thread allocator
  // directly instead of going through CUDAExecutionProvider::GetAllocator.
  // If that can be validated we could simply store the AllocatorPtr here and get rid of the delegate.
  std::map<OrtMemoryInfo, std::function<AllocatorPtr(int id, OrtMemType mem_type)>,
           OrtMemoryInfoLessThanIgnoreAllocType>
      allocators_;

  OrtValueNameIdxMap ort_value_name_idx_map_;

  // initialized tensors
  std::unordered_map<int, OrtValue> initialized_tensors_;  // key is ort_value_index
  // subset of initialized_tensors_ that are constant and cannot be overridden at runtime
  std::unordered_map<int, OrtValue> constant_initialized_tensors_;

  // This is an auxiliary lookup to check if the OrtValue was actually a sparse tensor
  // this is needed because we currently convert all sparse initializer into dense Tensors
  // if and when we actually place SparseTensor instances (we should) into OrtValues, we
  // will not need this structure.
  std::unordered_set<int> sparse_initialized_tensors_;

  // This data structure is for uninitializing string tensors and
  // munmap memory region and close file descriptor
  std::unordered_map<int, OrtCallback> deleter_for_initialized_tensors_;
  std::vector<BufferUniquePtr> weights_buffers_;
  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan_ = nullptr;

  const logging::Logger& logger_;
  profiling::Profiler& profiler_;

  // switch for enable memory pattern optimization or not.
  bool enable_mem_pattern_;

  // lock for the mem_patterns_
  mutable OrtMutex mem_patterns_lock_;

  // cache for the generated mem_patterns. key is calculated based on input shapes.
  mutable std::map<int64_t, std::unique_ptr<MemoryPatternGroup>> mem_patterns_;
  mutable std::map<int64_t, std::unordered_map<int, TensorShape>> shape_patterns_;

  NameNodeInfoMapType input_names_to_nodeinfo_mapping_;
  NameNodeInfoMapType output_names_to_nodeinfo_mapping_;

  // subgraph SessionState. entry for node containing subgraph, with value containing attribute:SessionState pair
  // as a node may contain multiple subgraphs (e.g. 'If' has one for both the 'then' and 'else' branches).
  using SubgraphSessionStateMap =
      std::unordered_map<onnxruntime::NodeIndex, std::unordered_map<std::string, std::unique_ptr<SessionState>>>;
  SubgraphSessionStateMap subgraph_session_states_;

  // either threadpool could be nullptr
  concurrency::ThreadPool* const thread_pool_{};
  concurrency::ThreadPool* const inter_op_thread_pool_{};

  bool export_fused_dll_ = false;
  FuncManager fused_funcs_mgr_;
  const DataTransferManager& data_transfer_mgr_;

  bool use_deterministic_compute_;
  bool enable_mem_reuse_;
  std::unique_ptr<NodeIndexInfo> node_index_info_;
  std::multimap<int, std::unique_ptr<FeedsFetchesManager>> cached_feeds_fetches_managers_;

  // Container to store pre-packed weights to share between sessions.
  // The life-cycle of the cache itself is maintained by the user and the user will ensure
  // the cache is valid until any session reliant on it is still in scope.
  // prepacked_weights_container_ can be nullptr if no caching is required for prepacked weights
  PrepackedWeightsContainer* const prepacked_weights_container_{};

#if !defined(ORT_MINIMAL_BUILD)
  std::map<std::vector<int>, std::unordered_set<NodeIndex>> to_be_executed_nodes_;
#endif

  SessionState* parent_ = nullptr;
  //Assign each graph in each session an unique id.
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  int graph_id_ = 0;
  int next_graph_id_ = 1;

  void GenerateGraphId() {
    SessionState* p = this;
    while (p->parent_ != nullptr) p = p->parent_;
    graph_id_ = p->next_graph_id_++;
  }
#endif

  // Counter for number of times pre-packing of weights was performed across kernels
  // part the model
  size_t number_of_prepacks_counter_ = 0;

  // Counter for number of times a shared version of the pre-packed weight corresponding to
  // a constant initialized weight was used by the session state
  size_t used_shared_pre_packed_weights_counter_ = 0;
};

}  // namespace onnxruntime
