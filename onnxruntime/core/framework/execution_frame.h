// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <vector>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/iexecutor.h"
#include "core/framework/ml_value.h"
#include "core/framework/node_index_info.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

class DataTransferManager;
class SessionState;
class OrtValueNameIdxMap;
class OrtValuePatternPlanner;
struct MemoryPatternGroup;
class NodeIndexInfo;

class IExecutionFrame {
 protected:
  // Derived class must call Init in its ctor. We need to use some of the virtual methods in Init and those aren't
  // initialized until the derived class is constructed.
  IExecutionFrame(const OrtValueNameIdxMap& ort_value_idx_map,
                  const NodeIndexInfo& node_index_info,
                  const std::vector<int>& fetch_mlvalue_idxs);

  void Init(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
            const std::unordered_map<int, OrtValue>& initializers,
            const std::function<bool(const std::string& name)>& is_initializer_sparse_func,
            const std::vector<OrtValue>& fetches);

 public:
  virtual ~IExecutionFrame();

  // Get the index for the first entry of the given node.
  int GetNodeOffset(NodeIndex index) const {
    return node_index_info_.GetNodeOffset(index);
  }

  // Return nullptr if index map to an value that is an unused optional input/output
  const OrtValue* GetNodeInputOrOutputMLValue(int index) const;
  OrtValue* GetMutableNodeInputOrOutputMLValue(int index);

#ifdef ENABLE_TRAINING
  // Override the index-th output with ort_value
  Status SetOutputMLValue(int index, const OrtValue& ort_value);
  void UpdateFeeds(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds);
  void UpdateFetches(const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                     const std::unordered_map<int, OrtValue>& initializers);
  Status GetOutputs(const std::vector<int>& fetch_mlvalue_idxs, std::vector<OrtValue>& fetches);
#endif

  // TO DO: make it thread safe
  // This method is not thread safe!
  // Return S_OK and nullptr if index map to an value that is an unused optional input/output
  // Shape is required for tensors but not traditional ML values.
  Status GetOrCreateNodeOutputMLValue(const int index, int output_arg_index, const TensorShape* shape,
                                      OrtValue*& p_ort_value, const Node& node);

  // This function try retrieve the inferred shapes for the given NodeArg index.
  // If the retrieval is successful, this function returns true and false otherwise.
  virtual bool TryGetInferredShape(int index, TensorShape& shape) const;

  /**
   * write the output values to the 'fetches' vector
   * Don't access the values after SessionState is destroyed 
   */
  Status GetOutputs(std::vector<OrtValue>& fetches);

  AllocatorPtr GetAllocator(const OrtMemoryInfo& info) const;

  Status ReleaseMLValue(int ort_value_idx);

 protected:
  // get the ort_value_idx from NodeIndexInfo
  int GetNodeIdxToMLValueIdx(int index) const;

  OrtValue& GetMutableMLValue(int ort_value_index) { return const_cast<OrtValue&>(GetMLValue(ort_value_index)); }

  virtual Status ReleaseMLValueImpl(int ort_value_idx);

  // returns true if the ort_value_idx is an output from the graph
  bool IsOutput(int ort_value_idx) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IExecutionFrame);

  const OrtValue& GetMLValue(int ort_value_index) const {
    ORT_ENFORCE(ort_value_index >= 0 && static_cast<size_t>(ort_value_index) < all_values_size_);
    return all_values_[ort_value_index];
  }

  // optional function that can check if the requested output_shape matched what was specified/inferred
  // for the node.
  virtual void VerifyOutputSizes(int /*output_index*/, const Node& /*node*/, const TensorShape& /*output_shape*/) {}

  virtual AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo& info) const = 0;

  virtual Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) = 0;

  virtual Status CopyTensor(const Tensor& src, Tensor& dest) const = 0;

  virtual const DataTransferManager& GetDataTransferManager() const = 0;

  const NodeIndexInfo& node_index_info_;

  // All the intermediate values for the entire graph.
  // Input and Output values are passed in by executors
  std::vector<OrtValue> all_values_;

  // perf optimization to avoid calling all_values_.size() repeatedly as the size is fixed once constructed
  const size_t all_values_size_;

  std::vector<int> fetch_mlvalue_idxs_;

  const OrtValueNameIdxMap& ort_value_idx_map_;
};

class ExecutionFrame final : public IExecutionFrame {
 public:
  ExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                 const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                 // optional custom allocators. key is index in fetches
                 const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                 const SessionState& session_state);

  ~ExecutionFrame() override;

  // TODO: These two AllocateMLValue... methods are in the API purely for unit test usage.
  // Fix the unit tests so they set an execution plan that results in these methods being called by
  // GetOrCreateNodeOutputMLValue instead
  Status AllocateMLValueTensorSelfOwnBuffer(OrtValue& ort_value, int ort_value_index, MLDataType element_type,
                                            const OrtMemoryInfo& location, const TensorShape& shape,
                                            bool create_fence = false);

  Status AllocateMLValueTensorPreAllocateBuffer(OrtValue& ort_value, int ort_value_index_reuse, MLDataType element_type,
                                                const OrtMemoryInfo& location, const TensorShape& shape,
                                                bool create_fence = false);

  // thread-safe
  Status GeneratePatterns(MemoryPatternGroup* out) const;

  bool HasMemoryPatternPlanner() const {
    return planner_ != nullptr;
  }

  // This function try retrieve the inferred shapes for the given NodeArg index.
  // If the retrival is sucessful, this function returns true and false otherwise.
  bool TryGetInferredShape(int index, TensorShape& shape) const override;

  // Return the size of virtual memory allocated in runtime.
  // The memory is usually used for activations in forward and backward passes.
  const std::unordered_map<std::string, size_t>& GetDynamicMemorySizeInfo() {
    // This function is not thread-safe. Please make sure dynamic_activation_memory_sizes_in_byte_
    // is not being changed when calling this function.
    // If one day, race condition happens, please uncomment the following line:
    //   std::unique_lock<std::mutex> lock(mtx_);
    return dynamic_activation_memory_sizes_in_byte_;
  }

  // Return the size of virtual memory allocated before computation.
  // The memory is usually used for activations in forward and backward passes.
  const std::unordered_map<std::string, size_t>& GetStaticMemorySizeInfo() {
    // This function is not thread-safe. Please make sure static_activation_memory_sizes_in_byte_
    // is not being changed when calling this function.
    // If one day, race condition happens, please uncomment the following line:
    //   std::unique_lock<std::mutex> lock(mtx_);
    return static_activation_memory_sizes_in_byte_;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecutionFrame);

  AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo& info) const override;
  Status ReleaseMLValueImpl(int ort_value_idx) override;
  Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) override;
  void VerifyOutputSizes(int output_index, const Node& node, const TensorShape& output_shape) override;
  Status CopyTensor(const Tensor& src, Tensor& dest) const override;
  const DataTransferManager& GetDataTransferManager() const override;

  common::Status AllocateAsPerAllocationPlan(OrtValue& ort_value, int ort_value_index, const TensorShape* shape);

  Status AllocateMLValueTensorSelfOwnBufferHelper(OrtValue& ort_value, int ort_value_index, MLDataType element_type,
                                                  const OrtMemoryInfo& location, const TensorShape& shape,
                                                  bool create_fence);

  Status AllocateTensorWithPreAllocateBufferHelper(OrtValue& ort_value, void* pBuffer, MLDataType element_type,
                                                   const OrtMemoryInfo& location, const TensorShape& shape);

  void TraceAllocate(int ort_value_idx, size_t size);
  void TraceFree(int ort_value_idx);

  const AllocPlanPerValue& GetAllocationPlan(int ort_value_idx);

  const SessionState& session_state_;

  // map of index to custom allocator
  std::unordered_map<int, IExecutor::CustomAllocator> custom_allocators_;

  // If we already have cached memory pattern on these input shapes
  // Use this mem pattern that create a big chunk for all the internal
  // kernel's input/output tensors.
  const MemoryPatternGroup* mem_patterns_;

  // If no cached memory pattern, and we enable the memory pattern optimization
  // use this planner_ to trace the memory allocation in current executor.
  std::unique_ptr<OrtValuePatternPlanner> planner_;

  // Big chunks on different locations that will be used by mem_pattern.
  std::map<OrtMemoryInfo, BufferUniquePtr> buffers_;

  // Given the input shapes of the executed graph, ExecutionFrame tries inferring
  // all symbolic shapes. inferred_shapes_[i] is the shape of OrtValue indexed
  // by i, if the key i exists.
  // inferred_shapes_ is generated together with mem_patterns_.
  std::unordered_map<int, TensorShape> inferred_shapes_;

  // Size of virtual memory allocated before any kernel execution.
  // This field is not physical memory size.
  // static_activation_memory_sizes_in_byte_[location] is the static memory consumption on "location".
  std::unordered_map<std::string, size_t> static_activation_memory_sizes_in_byte_;

  // Size of virtual memory allocated during kernel execution (i.e., inside a kernel,
  // we may allocate some memory for its outputs, if not planned.).
  // This field is not physical memory size.
  // dynamic_activation_memory_sizes_in_byte_[location] is the dynamic memory consumption on "location".
  std::unordered_map<std::string, size_t> dynamic_activation_memory_sizes_in_byte_;

  // Mutex which should be acquired when executing non-thread-safe member functions.
  // A current example is the tracker of dynamic memory allocation.
  mutable std::mutex mtx_;
};
}  // namespace onnxruntime
