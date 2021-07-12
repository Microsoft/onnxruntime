// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"

#include <sstream>

#include "core/framework/mem_pattern_planner.h"
#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/utils.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

using namespace onnxruntime::common;

namespace onnxruntime {

IExecutionFrame::IExecutionFrame(const OrtValueNameIdxMap& ort_value_idx_map,
                                 const NodeIndexInfo& node_index_info,
                                 const std::vector<int>& fetch_mlvalue_idxs)
    : node_index_info_(node_index_info),
      all_values_size_(static_cast<size_t>(ort_value_idx_map.MaxIdx()) + 1),
      fetch_mlvalue_idxs_(fetch_mlvalue_idxs),
      ort_value_idx_map_(ort_value_idx_map) {
  ORT_ENFORCE(node_index_info_.GetMaxMLValueIdx() == ort_value_idx_map.MaxIdx(),
              "node_index_info and ort_value_idx_map are out of sync and cannot be used");
}

IExecutionFrame::~IExecutionFrame() = default;

#ifdef ENABLE_TRAINING
Status IExecutionFrame::SetOutputMLValue(int index, const OrtValue& ort_value) {
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(ort_value_idx) >= all_values_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", ort_value_idx);
  }

  ORT_ENFORCE(!all_values_[ort_value_idx].IsAllocated());
  all_values_[ort_value_idx] = ort_value;
  return Status::OK();
}

void IExecutionFrame::UpdateFeeds(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds) {
  ORT_ENFORCE(feed_mlvalue_idxs.size() == feeds.size());

  for (size_t idx = 0, end = feed_mlvalue_idxs.size(); idx < end; ++idx) {
    int ort_value_idx = feed_mlvalue_idxs[idx];
    // we are sharing the underlying tensor/object for OrtValue

    ORT_ENFORCE(!all_values_[ort_value_idx].IsAllocated());

    all_values_[ort_value_idx] = feeds[idx];
  }
}

void IExecutionFrame::UpdateFetches(const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches, const std::unordered_map<int, OrtValue>& initializers) {
  ORT_ENFORCE(fetch_mlvalue_idxs.size() == fetches.size());

  if (!fetches.empty()) {
    fetch_mlvalue_idxs_ = fetch_mlvalue_idxs;

    auto num_fetches = fetch_mlvalue_idxs_.size();

    for (size_t idx = 0; idx < num_fetches; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs_[idx];

      ORT_ENFORCE(!all_values_[ort_value_idx].IsAllocated());

      all_values_[ort_value_idx] = fetches[idx];

      // Copy the initializer if it is a fetch entry.
      auto entry = initializers.find(ort_value_idx);
      if (entry != initializers.end()) {
        const Tensor& src = entry->second.Get<Tensor>();
        OrtValue& dest = all_values_[ort_value_idx];

        if (!dest.IsAllocated()) {
          AllocatorPtr allocator = GetAllocator(src.Location());
          auto p_tensor = std::make_unique<Tensor>(src.DataType(), src.Shape(), allocator);
          auto ml_tensor = DataTypeImpl::GetType<Tensor>();
          dest.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        }

        ORT_THROW_IF_ERROR(CopyTensor(src, *dest.GetMutable<Tensor>()));
      }
    }
  }
}

Status IExecutionFrame::GetOutputs(const std::vector<int>& fetch_mlvalue_idxs, std::vector<OrtValue>& fetches) {
  auto num_fetches = fetch_mlvalue_idxs.size();

  if (fetches.empty()) {
    fetches.resize(num_fetches);
  } else {
    // if there's a mismatch things are out so sync so fail
    if (fetches.size() != num_fetches) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Fetches vector passed to GetOutputs contains ", fetches.size(),
                             " entries which doesn't match the number of fetches the frame was initialized with of ",
                             num_fetches);
    }
  }

  for (size_t idx = 0; idx < num_fetches; ++idx) {
    fetches[idx] = GetMLValue(fetch_mlvalue_idxs[idx]);
  }

  return Status::OK();
}

#endif

// Return nullptr if index map to an value that is an unused optional input/output
const OrtValue* IExecutionFrame::GetNodeInputOrOutputMLValue(int index) const {
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);
  return ort_value_idx != NodeIndexInfo::kInvalidEntry ? &(all_values_[ort_value_idx]) : nullptr;
}

OrtValue* IExecutionFrame::GetMutableNodeInputOrOutputMLValue(int index) {
  return const_cast<OrtValue*>(GetNodeInputOrOutputMLValue(index));
}

// TO DO: make it thread safe
// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output

Status IExecutionFrame::GetOrCreateNodeOutputMLValue(const int output_index, int output_arg_index,
                                                     const TensorShape* shape, OrtValue*& p_ort_value,
                                                     const Node& node) {
  auto status = Status::OK();
  int ort_value_idx = GetNodeIdxToMLValueIdx(output_arg_index);

  // return nullptr if it is optional
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry) {
    p_ort_value = nullptr;
  } else {
    p_ort_value = &all_values_[ort_value_idx];

    if (p_ort_value->IsAllocated()) {
      // already allocated. verify shape matches if tensor.
      if (p_ort_value->IsTensor()) {
        const Tensor& tensor = p_ort_value->Get<Tensor>();
        ORT_ENFORCE(shape && tensor.Shape() == *shape,
                    "OrtValue shape verification failed. Current shape:", tensor.Shape(),
                    " Requested shape:", shape ? shape->ToString() : "null");
      } else if (p_ort_value->IsSparseTensor()) {
        const SparseTensor& sp_tensor = p_ort_value->Get<SparseTensor>();
        ORT_ENFORCE(shape && sp_tensor.Shape() == *shape,
                    "OrtValue shape verification failed. Current shape:", sp_tensor.Shape(),
                    " Requested shape:", shape ? shape->ToString() : "null");
      }
    } else {
      // shape is nullptr for traditional ML output values
      if (shape != nullptr && IsOutput(ort_value_idx)) {
        VerifyOutputSizes(output_index, node, *shape);
      }
      status = CreateNodeOutputMLValueImpl(*p_ort_value, ort_value_idx, shape);
    }
  }

  return status;
}

bool IExecutionFrame::TryGetInferredShape(int /*index*/, TensorShape& /*shape*/) const {
  // By default, there is not information about inferred shape, so this default
  // implementation always returns false. The derived class of IExecutionFrame
  // can override this function to provide, for example, activations' shape information.
  return false;
}

AllocatorPtr IExecutionFrame::GetAllocator(const OrtMemoryInfo& info) const {
  return GetAllocatorImpl(info);
}

Status IExecutionFrame::ReleaseMLValue(int ort_value_idx) { return ReleaseMLValueImpl(ort_value_idx); }

Status IExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(ort_value_idx) >= all_values_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", ort_value_idx);
  }

  // If fence is available, check whether async read has completed or not.
  Fence_t fence = GetMLValue(ort_value_idx).Fence();
  if (fence && !fence->CanRelease()) {
    // Async data reading is not done yet, defer mem release until Session.run() end.
    return Status::OK();
  }

  all_values_[ort_value_idx] = OrtValue();
  return Status::OK();
}

int IExecutionFrame::GetNodeIdxToMLValueIdx(int index) const {
  // the validity of index is checked by GetMLValueIndex
  int ort_value_idx = node_index_info_.GetMLValueIndex(index);
  return ort_value_idx;
}

void IExecutionFrame::Init(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                           const std::unordered_map<int, OrtValue>& initializers,
                           const std::function<bool(const std::string& name)>& is_initializer_sparse_func,
                           const std::vector<OrtValue>& fetches) {
  ORT_ENFORCE(feeds.size() == feed_mlvalue_idxs.size());
  ORT_ENFORCE(fetches.empty() || fetches.size() == fetch_mlvalue_idxs_.size());

  // Need this for sparse conversions in host memory
  OrtMemoryInfo cpu_mem_info("Cpu", OrtDeviceAllocator);
  AllocatorPtr cpu_allocator = GetAllocator(cpu_mem_info);

  // 1. resize the all_value_ vector
  all_values_.resize(all_values_size_);

  // 2. Handle non-empty output vector
  if (!fetches.empty()) {
    auto num_fetches = fetch_mlvalue_idxs_.size();

    for (size_t idx = 0; idx < num_fetches; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs_[idx];
      all_values_[ort_value_idx] = fetches[idx];
    }
  }

  // 3. handle the weights.
  // We do this after the fetches to handle an edge case where an initializer is an output.
  // e.g. A Constant node gets lifted to an initializer so there's no Node producing the value as an output during
  // Graph execution (i.e. Graph execution won't write the value to all_values_).
  // A non-empty fetches vector will overwrite the actual weight in all_values_[ort_value_idx] if we did this earlier.
  // This makes the ONNX Constant test (onnx\backend\test\data\node\test_constant) happy as that
  // involves a graph with a single Constant node.
  for (const auto& entry : initializers) {
    int ort_value_index = entry.first;

    // if the initializer is an output we need to allocate or use a provided fetch buffer and copy the data
    // so it can be returned to the caller.
    //
    // The alternative to handling this as a special case would be to disallow an initializer providing a graph output.
    // There's nothing in the ONNX spec that says a graph output must come from a node output though.
    // If we took that approach we'd need to:
    //   - reject a model with an initializer or Constant node (as we convert those to initializers in Graph::Graph)
    //     that produces a graph output even though it conforms to the ONNX spec
    //   - update optimizers to not convert something to an initializer that is a graph output
    //     (e.g. constant folding)
    if (IsOutput(ort_value_index)) {
      std::string name;
      ORT_THROW_IF_ERROR(ort_value_idx_map_.GetName(ort_value_index, name));
      const bool is_sparse_initializer = is_initializer_sparse_func(name);
      const Tensor& src = entry.second.Get<Tensor>();  // all initializers in ONNX are tensors
      OrtValue& dest = all_values_[ort_value_index];

      if (is_sparse_initializer) {
        if (!dest.IsAllocated()) {
          auto p_tensor = std::make_unique<SparseTensor>();
          auto ml_tensor = DataTypeImpl::GetType<SparseTensor>();
          dest.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        }

        // Outputting Coo format bc initializers are Constant nodes are not stored in COO format.
        AllocatorPtr allocator = GetAllocator(src.Location());
        constexpr bool has_linear_coo_index = true;
        ORT_THROW_IF_ERROR(sparse_utils::DenseTensorToSparseCoo(GetDataTransferManager(), src,
                                                                cpu_allocator, allocator, has_linear_coo_index,
                                                                *dest.GetMutable<SparseTensor>()));
      } else {
        if (!dest.IsAllocated()) {
          // NOTE: This doesn't need to support ExecutionFrame custom allocators as they only come into play
          // for a subgraph with an output of unknown shape that needs to be accumulated by the control flow node.
          // If the initializer is providing the output, the shape is known.
          AllocatorPtr allocator = GetAllocator(src.Location());

          auto p_tensor = std::make_unique<Tensor>(src.DataType(), src.Shape(), allocator);
          auto ml_tensor = DataTypeImpl::GetType<Tensor>();
          dest.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        }
        ORT_THROW_IF_ERROR(CopyTensor(src, *dest.GetMutable<Tensor>()));
      }
    } else {
      all_values_[ort_value_index] = entry.second;
    }
  }

  // 4. handle feed in values. these can override initializer values so must be last
  for (size_t idx = 0, end = feed_mlvalue_idxs.size(); idx < end; ++idx) {
    int ort_value_idx = feed_mlvalue_idxs[idx];
    // we are sharing the underline tensor/object for MLValue
    all_values_[ort_value_idx] = feeds[idx];
  }
}

Status IExecutionFrame::GetOutputs(std::vector<OrtValue>& fetches) {
  auto num_fetches = fetch_mlvalue_idxs_.size();

  if (fetches.empty()) {
    fetches.resize(num_fetches);
  } else {
    // if there's a mismatch things are out so sync so fail
    if (fetches.size() != num_fetches) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Fetches vector passed to GetOutputs contains ", fetches.size(),
                             " entries which doesn't match the number of fetches the frame was initialized with of ",
                             num_fetches);
    }
  }

  for (size_t idx = 0; idx < num_fetches; ++idx) {
    fetches[idx] = GetMLValue(fetch_mlvalue_idxs_[idx]);
  }

  return Status::OK();
}

bool IExecutionFrame::IsOutput(int ort_value_idx) const {
  return std::find(fetch_mlvalue_idxs_.begin(), fetch_mlvalue_idxs_.end(), ort_value_idx) != fetch_mlvalue_idxs_.end();
}

ExecutionFrame::ExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                               const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               const SessionState& session_state)
    : IExecutionFrame(session_state.GetOrtValueNameIdxMap(), session_state.GetNodeIndexInfo(), fetch_mlvalue_idxs),
      session_state_(session_state),
      mem_patterns_(nullptr),
      planner_(nullptr) {
  Init(
      feed_mlvalue_idxs, feeds, session_state.GetInitializedTensors(),
      [&session_state](const std::string& name) -> bool {
        int idx = -1;
        if (session_state.GetOrtValueNameIdxMap().GetIdx(name, idx).IsOK()) {
          return session_state.IsSparseInitializer(idx);
        }
        return false;
      },
      fetches);
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryInfo::IncreaseIteration();
#endif

  // map the custom allocators to ort_value_idx entries
  if (!fetch_allocators.empty()) {
    for (size_t idx = 0, end = fetch_mlvalue_idxs.size(); idx < end; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs[idx];

      auto custom_alloc_entry = fetch_allocators.find(idx);
      if (custom_alloc_entry != fetch_allocators.cend()) {
        custom_allocators_[ort_value_idx] = custom_alloc_entry->second;
      }
    }
  }

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimization.
  if (session_state.GetEnableMemoryPattern() && session_state.GetExecutionPlan()) {
    std::vector<std::reference_wrapper<const TensorShape>> input_shapes;
    bool all_tensors = true;
    // Reserve mem to avoid re-allocation.
    input_shapes.reserve(feeds.size());
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.Get<Tensor>();
      input_shapes.push_back(std::cref(tensor.Shape()));
    }

    //if there are some traditional ml value type in inputs disable the memory pattern optimization.
    if (all_tensors) {
      mem_patterns_ = session_state.GetMemoryPatternGroup(input_shapes, feed_mlvalue_idxs, inferred_shapes_);
      // if no existing patterns, generate one in this executionframe
      if (!mem_patterns_) {
        planner_ = std::make_unique<OrtValuePatternPlanner>(*session_state.GetExecutionPlan());
      } else {
        // pre-allocate the big chunk requested in memory pattern.
        // all the internal kernel's input/output tensors will be allocated on these buffer.
        for (size_t i = 0; i < mem_patterns_->locations.size(); i++) {
          const auto& location = mem_patterns_->locations[i];
          ORT_ENFORCE(buffers_.find(location) == buffers_.end());
          if (mem_patterns_->patterns[i].PeakSize() > 0) {
            AllocatorPtr alloc = GetAllocator(location);
            void* buffer = nullptr;
            // it's possible we can't allocate the large block. if we have memory patterns we know we have successfully
            // executed once before, so if there's an arena involved it probably has smaller blocks available.
            // due to that we can still run and use those blocks (inside the arena logic) instead of one large one.
            // it's less efficient (the arena will add some overhead to coalesce individual allocations
            // back into blocks on 'free'), but better than failing completely.
            ORT_TRY {
              auto peak_size = mem_patterns_->patterns[i].PeakSize();
              // Planning of one memory type should only happen once.
              ORT_ENFORCE(
                  static_activation_memory_sizes_in_byte_.find(location.name) ==
                      static_activation_memory_sizes_in_byte_.end(),
                  "Memory type ",
                  location.name,
                  " should only appear once.");
              // static_activation_memory_in_bytes_ is max virtual memory size the planner computes.
              // Memory dynamically allocated when executing kernels is not recorded using this field.
              static_activation_memory_sizes_in_byte_[location.name] = peak_size;
              buffer = alloc->Alloc(peak_size);
              // handle allocator that doesn't throw
              if (buffer == nullptr) {
                // INFO level as this may fire on every run and there may not be much a user can do
                LOGS(session_state_.Logger(), INFO) << "Allocation of memory pattern buffer for "
                                                    << location.ToString() << " returned nullptr";
              }
            }
            ORT_CATCH(const OnnxRuntimeException& ex) {
              ORT_HANDLE_EXCEPTION([&]() {
                LOGS(session_state_.Logger(), INFO) << "Allocation of memory pattern buffer for "
                                                    << location.ToString() << " failed. Error:" << ex.what();
              });
            }

            if (buffer != nullptr) {
              buffers_[location] = BufferUniquePtr(buffer, alloc);
            }
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
            //Record activation memory pattern
            MemoryInfo::ClearMemoryInfoPerExecution();
            if (mem_patterns_ && buffer != nullptr) {
              MemoryInfo::RecordPatternInfo(*mem_patterns_, MemoryInfo::MapType::StaticActivation);
              MemoryInfo::MemoryInfoProfile::CreateEvents("static activations_" + std::to_string(MemoryInfo::GetIteration()),
                                                          MemoryInfo::MemoryInfoProfile::GetAndIncreasePid(), MemoryInfo::MapType::StaticActivation, "", 0);
            }
#endif
            // log size of activation. Keep it commented out for now to avoid log flooding.
            // VLOGS(session_state_.Logger(), 1) << "**** Allocated memory for activations, size: " <<mem_patterns_->patterns[i].PeakSize();
          }
        }
      }
    }
  }
}

ExecutionFrame::~ExecutionFrame() = default;

Status ExecutionFrame::CopyTensor(const Tensor& src, Tensor& dest) const {
  return session_state_.GetDataTransferMgr().CopyTensor(src, dest);
}

const DataTransferManager& ExecutionFrame::GetDataTransferManager() const {
  return session_state_.GetDataTransferMgr();
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(OrtValue& ort_value, int ort_value_index,
                                                          MLDataType element_type, const OrtMemoryInfo& location,
                                                          const TensorShape& shape, bool create_fence) {
  return AllocateMLValueTensorSelfOwnBufferHelper(ort_value, ort_value_index, element_type, location, shape,
                                                  create_fence);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(OrtValue& ort_value, int ort_value_index,
                                                                MLDataType element_type,
                                                                const OrtMemoryInfo& location,
                                                                const TensorShape& shape, bool create_fence) {
  if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
  }

  size_t size;
  int64_t len = shape.Size();
  if (len < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  if (static_cast<uint64_t>(len) > std::numeric_limits<size_t>::max()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape is too large");
  }
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(static_cast<size_t>(len), element_type->Size(), &size)) {
    return Status(ONNXRUNTIME, FAIL, "size overflow");
  }

  // Lazily get the allocator only if needed.
  AllocatorPtr alloc = nullptr;

  // create fence if needed
  if (create_fence) {
    ORT_ENFORCE(ort_value.Fence() == nullptr);
    alloc = GetAllocator(location);
    FencePtr f = alloc->CreateFence(&session_state_);
    // it is OK to have fence been nullptr if the execution provider has no async execution,
    // and allocator::CreateFence returns nullptr
    ort_value.SetFence(f);
  }

  // if we have pre-calculated memory pattern, and the ort_value is not output mlvalue
  // try to allocated on pre-allocated big chunk.
  const auto& per_alloc_plan = GetAllocationPlan(ort_value_index);

  if (mem_patterns_ && per_alloc_plan.alloc_kind != AllocKind::kAllocateOutput &&
      per_alloc_plan.alloc_kind != AllocKind::kAllocatedExternally) {
    auto pattern = mem_patterns_->GetPatterns(location);
    if (pattern) {
      auto block = pattern->GetBlock(ort_value_index);
      // if block not found, fall back to default behavior
      if (block) {
        auto it = buffers_.find(location);
        if (it != buffers_.end()) {
          // if the block is not correct, log message then fall back to default behavior
          if (block->size_ == size) {
            void* buffer = it->second.get();
            auto status = AllocateTensorWithPreAllocateBufferHelper(
                ort_value, static_cast<void*>(static_cast<char*>(buffer) + block->offset_), element_type, location,
                shape);
            return status;
          } else {
            // the block size may vary especially if the model has NonZero ops, or different sequence lengths are
            // fed in, so use VERBOSE as the log level as it's expected.
            // TODO: Should we re-use the block if the size is large enough? Would probably need to allow it
            // to be freed if the size difference was too large so our memory usage doesn't stick at a high water mark
            LOGS(session_state_.Logger(), VERBOSE) << "For ort_value with index: " << ort_value_index
                                                   << ", block in memory pattern size is: " << block->size_
                                                   << " but the actually size is: " << size
                                                   << ", fall back to default allocation behavior";
          }
        }
        // else { we couldn't allocate the large block for the buffer so we didn't insert an entry }
      }
    }
  }

  //no memory pattern, or the pattern is not correct.
  if (!alloc) alloc = GetAllocator(location);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type, shape, alloc);

  {
    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  }

  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't support it in memory pattern optimization.
  if (!utils::IsDataTypeString(element_type)) {
    TraceAllocate(ort_value_index, size);
  }

  {
    // This code block is not thread-safe.
    // Dynamic activation size would be accessed by multiple threads
    // if parallel executor is used.
    std::unique_lock<std::mutex> lock(mtx_);
    dynamic_activation_memory_sizes_in_byte_[location.name] += size;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    MemoryInfo::SetDynamicAllocation(ort_value_index);
#endif
  }

  return Status::OK();
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(OrtValue& ort_value, int ort_value_index_reuse,
                                                              MLDataType element_type, const OrtMemoryInfo& location,
                                                              const TensorShape& shape, bool create_fence) {
  OrtValue& ort_value_reuse = GetMutableMLValue(ort_value_index_reuse);

  auto* reuse_tensor = ort_value_reuse.GetMutable<Tensor>();
  auto buffer_num_elements = reuse_tensor->Shape().Size();
  auto required_num_elements = shape.Size();

  // check number of elements matches. shape may not be an exact match (e.g. Reshape op)
  if (buffer_num_elements != required_num_elements) {
    // could be an allocation planner bug (less likely) or the model incorrectly uses something like 'None'
    // as a dim_param, or -1 in dim_value in multiple places making the planner think those shapes are equal.
    auto message = onnxruntime::MakeString(
        "Shape mismatch attempting to re-use buffer. ",
        reuse_tensor->Shape(), " != ", shape,
        ". Validate usage of dim_value (values should be > 0) and "
        "dim_param (all values with the same string should equate to the same size) in shapes in the model.");

    // be generous and use the buffer if it's large enough. log a warning though as it indicates a bad model
    if (buffer_num_elements >= required_num_elements) {
      // View Operator is reusing the buffer bigger than the required size.
      // Disabling warning message for now. The op is in the process of being deprecated.
#ifndef ENABLE_TRAINING
      LOGS(session_state_.Logger(), WARNING) << message;
#endif
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, message);
    }
  }

  void* reuse_buffer = reuse_tensor->MutableDataRaw();

  // create fence on reused ort_value if needed
  // TODO: differentiate reuse and alias, by add AllocKind::kAlias?
  if (create_fence && ort_value_reuse.Fence() == nullptr) {
    FencePtr f = GetAllocator(location)->CreateFence(&session_state_);
    ort_value_reuse.SetFence(f);
  }

  // reused OrtValue share the same fence
  ort_value.ShareFenceWith(ort_value_reuse);
  return AllocateTensorWithPreAllocateBufferHelper(ort_value, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(OrtValue& ort_value, void* pBuffer,
                                                                 MLDataType element_type,
                                                                 const OrtMemoryInfo& location,
                                                                 const TensorShape& shape) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, pBuffer, location);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  return Status::OK();
}

static Status AllocateTraditionalMLValue(OrtValue& ort_value, const NonTensorTypeBase& type) {
  auto creator = type.GetCreateFunc();
  ort_value.Init(creator(), &type, type.GetDeleteFunc());
  return Status::OK();
}

static Status AllocateTensorSequence(OrtValue& ort_value) {
  auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
  auto p_tensor_sequence = std::make_unique<TensorSeq>();
  ort_value.Init(p_tensor_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());

  return Status::OK();
}

static Status AllocateSparseTensor(OrtValue& mlvalue, const DataTypeImpl& ml_type, AllocatorPtr allocator,
                                   const TensorShape& shape, bool create_fence,
                                   const SessionState& session_state) {
  auto element_type = ml_type.AsSparseTensorType()->GetElementType();
  auto sparse = std::make_unique<SparseTensor>(element_type, shape, allocator);
  auto deleter = DataTypeImpl::GetType<SparseTensor>()->GetDeleteFunc();
  mlvalue.Init(sparse.release(), DataTypeImpl::GetType<SparseTensor>(), deleter);

  // create fence if needed
  if (create_fence) {
    ORT_ENFORCE(mlvalue.Fence() == nullptr);
    FencePtr f = allocator->CreateFence(&session_state);
    mlvalue.SetFence(f);
  }

  return Status::OK();
}

// This method is not thread safe!
Status ExecutionFrame::AllocateAsPerAllocationPlan(OrtValue& ort_value, int ort_value_index, const TensorShape* shape) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(ort_value_index >= 0 && static_cast<size_t>(ort_value_index) < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[ort_value_index];

  const auto& alloc_info = per_alloc_plan.location;
  const auto* ml_type = per_alloc_plan.value_type;
  if (ml_type == nullptr) {
    return Status(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Tried to allocate without valid type information, ort_value index=" + std::to_string(ort_value_index));
  }

  // if there is a custom allocator for this ort_value_index, call it to do the allocation
  auto custom_alloc_entry = custom_allocators_.find(ort_value_index);
  if (custom_alloc_entry != custom_allocators_.cend()) {
    ORT_ENFORCE(shape, "We don't expect custom allocators for non-tensor types, so a shape is mandatory here.");
    bool allocated = false;
    // see if custom allocator can handle allocation
    auto status = (custom_alloc_entry->second)(*shape, alloc_info, ort_value, allocated);
    if (allocated || !status.IsOK())
      return status;
  }

  if (ml_type->IsTensorType()) {
    ORT_ENFORCE(shape, "Allocation of tensor types requires a shape.");

    // tensors
    const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

    AllocKind alloc_kind = per_alloc_plan.alloc_kind;
    switch (alloc_kind) {
      // Right now for kAllocate and kAllocateOutput we are using same approach.
      // In the future we may want to have different way to handle it.
      case AllocKind::kAllocateOutput:
      case AllocKind::kAllocate: {
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(ort_value, ort_value_index, ml_data_type, alloc_info,
                                                               *shape, per_alloc_plan.create_fence_if_async));
        break;
      }
      case AllocKind::kReuse: {
        int reuse_mlvalue_index = per_alloc_plan.reused_buffer;

        // In case OrtRunOptions.only_execute_path_to_fetches == true, it is possible that 'reuse_value'
        // is not allocated (its upstream op is not executed due to the option).
        // In this case we need to allocate 'reuse_value' and then let 'ort_value' to reuse it.
        OrtValue& reuse_value = GetMutableMLValue(reuse_mlvalue_index);
        if (!reuse_value.IsAllocated()) {
          ORT_RETURN_IF_ERROR(AllocateAsPerAllocationPlan(reuse_value, reuse_mlvalue_index, shape));
        }
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(
            ort_value, reuse_mlvalue_index, ml_data_type, alloc_info, *shape, per_alloc_plan.create_fence_if_async));
        break;
      }
      case AllocKind::kShare: {
        int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
        // copy at the OrtValue level so the shared_ptr for the data is shared between the two OrtValue instances
        ort_value = GetMutableMLValue(reuse_mlvalue_index);
        break;
      }
      default: {
        std::ostringstream ostr;
        ostr << "Invalid allocation kind: " << static_cast<std::underlying_type<AllocKind>::type>(alloc_kind);
        return Status(ONNXRUNTIME, FAIL, ostr.str());
      }
    }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    MemoryInfo::RecordActivationAllocInfo(ort_value_index, ort_value);
#endif

    return Status::OK();
  } else if (ml_type->IsSparseTensorType()) {
    return AllocateSparseTensor(ort_value, *ml_type, GetAllocator(alloc_info),
                                *shape, per_alloc_plan.create_fence_if_async, session_state_);
  } else if (ml_type->IsTensorSequenceType()) {
    return AllocateTensorSequence(ort_value);
  } else {
    return AllocateTraditionalMLValue(ort_value, *static_cast<const NonTensorTypeBase*>(ml_type));
  }
}

AllocatorPtr ExecutionFrame::GetAllocatorImpl(const OrtMemoryInfo& info) const {
  return session_state_.GetAllocator(info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status ExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) {
  return AllocateAsPerAllocationPlan(ort_value, ort_value_idx, shape);
}

void ExecutionFrame::VerifyOutputSizes(int output_index, const Node& node, const TensorShape& output_shape) {
  const NodeArg* output_def = node.OutputDefs()[output_index];
  const auto* expected_shape = output_def->Shape();
  if (expected_shape == nullptr) {
    // model didn't specify shape and shape inferencing wasn't able to calculate it so nothing to compare against
    return;
  }

  const size_t expected_rank = expected_shape->dim_size();
  bool compatible = expected_rank == output_shape.NumDimensions();
  if (compatible) {
    for (size_t i = 0; i < expected_rank; ++i) {
      const auto& expected_dim = expected_shape->dim().Get(static_cast<int>(i));
      if (expected_dim.has_dim_value() && expected_dim.dim_value() != output_shape[i]) {
        compatible = false;
        break;
      }
    }
  }

  if (!compatible) {
    LOGS(session_state_.Logger(), WARNING) << "Expected shape from model of " << *expected_shape
                                           << " does not match actual shape of " << output_shape
                                           << " for output " << output_def->Name();
  }
}

Status ExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  ORT_RETURN_IF_ERROR(IExecutionFrame::ReleaseMLValueImpl(ort_value_idx));
  TraceFree(ort_value_idx);
  return Status::OK();
}

const AllocPlanPerValue& ExecutionFrame::GetAllocationPlan(int ort_value_idx) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
  return alloc_plan[ort_value_idx];
}

void ExecutionFrame::TraceAllocate(int ort_value_idx, size_t size) {
  if (planner_) {
    // don't trace the output tensors or external outputs.
    auto& allocation_plan = GetAllocationPlan(ort_value_idx);
    if (allocation_plan.alloc_kind == AllocKind::kAllocateOutput ||
        allocation_plan.alloc_kind == AllocKind::kAllocatedExternally) {
      return;
    }
    auto status = planner_->TraceAllocation(ort_value_idx, size);
    if (!status.IsOK())
      LOGS(session_state_.Logger(), WARNING) << "TraceAllocation for ort_value_idx=" << ort_value_idx
                                             << " size=" << size << " failed: " << status.ErrorMessage();
  }
}

void ExecutionFrame::TraceFree(int ort_value_idx) {
  // don't trace free on output tensors.
  if (planner_ && !IsOutput(ort_value_idx)) {
    const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
    const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
    ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
    const auto& per_alloc_plan = alloc_plan[ort_value_idx];

    // only trace tensors
    auto ml_type = per_alloc_plan.value_type;
    if (ml_type->IsTensorType()) {
      // tensors
      auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      // don't trace string tensors
      if (!utils::IsDataTypeString(ml_data_type)) {
        auto status = planner_->TraceFree(ort_value_idx);
        if (!status.IsOK()) {
          LOGS(session_state_.Logger(), WARNING)
              << "TraceFree for ort_value_idx=" << ort_value_idx << " failed: " << status.ErrorMessage();
        }
      }
    }
  }
}

// generate memory pattern based on the tracing of memory allocation/free in current execution
// return error if the planner is not setup.
Status ExecutionFrame::GeneratePatterns(MemoryPatternGroup* out) const {
  if (!planner_) {
    return Status(ONNXRUNTIME, FAIL, "Memory pattern planner is not enabled on this execution framework.");
  }

  return planner_->GeneratePatterns(out);
}

bool ExecutionFrame::TryGetInferredShape(int index, TensorShape& shape) const {
  // NodeArg index to OrtValue index.
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);

  // Check if index is valid.
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry) {
    return false;
  }

  // Search for inferred shape.
  // If inferred shape is found, it's assigned to "shape" so that caller can use it.
  auto it = inferred_shapes_.find(ort_value_idx);
  if (it != inferred_shapes_.end()) {
    shape = it->second;
    return true;
  }

  // Tell the caller if the search is successful or not.
  return false;
}

}  // namespace onnxruntime
