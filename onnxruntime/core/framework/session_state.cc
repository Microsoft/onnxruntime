// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <sstream>

#include "core/platform/ort_mutex.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/allocator.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/session_state_utils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

using namespace ::onnxruntime::common;
using namespace ::onnxruntime::experimental;

namespace onnxruntime {

void SessionState::SetupAllocators() {
  for (const auto& provider : execution_providers_) {
    for (const auto& allocator : provider->GetAllocators()) {
      const OrtMemoryInfo& memory_info = allocator->Info();
      if (allocators_.find(memory_info) != allocators_.end()) {
        // EPs are ordered by priority so ignore the duplicate allocator for this memory location.
        LOGS(logger_, INFO) << "Allocator already registered for " << allocator->Info()
                            << ". Ignoring allocator from " << provider->Type();
      } else {
        // slightly weird indirection to go back to the provider to get the allocator each time it's needed
        // in order to support scenarios such as the CUDA EP's per-thread allocator.
        allocators_[memory_info] = [&provider](int id, OrtMemType mem_type) {
          return provider->GetAllocator(id, mem_type);
        };
      }
    }
  }
}

AllocatorPtr SessionState::GetAllocator(const OrtMemoryInfo& location) const noexcept {
  AllocatorPtr result;
  auto entry = allocators_.find(location);
  if (entry != allocators_.cend()) {
    result = entry->second(location.id, location.mem_type);
  }

  return result;
}

AllocatorPtr SessionState::GetAllocator(OrtDevice device) const noexcept {
  for (const auto& iter : allocators_) {
    if (iter.first.device == device) {
      return iter.second(device.Id(), iter.first.mem_type);
    }
  }
  return nullptr;
}

void SessionState::CreateGraphInfo() {
  graph_viewer_ = std::make_unique<onnxruntime::GraphViewer>(graph_);
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

#if !defined(ORT_MINIMAL_BUILD)
Status SessionState::PopulateKernelCreateInfo(KernelRegistryManager& kernel_registry_manager,
                                              bool saving_ort_format) {
  for (auto& node : graph_.Nodes()) {
    const KernelCreateInfo* kci = nullptr;

    auto status = kernel_registry_manager.SearchKernelRegistry(node, &kci);
    if (!status.IsOK() && saving_ort_format) {
      // if we didn't find the kernel and are saving to ORT format an EP that compiles nodes is enabled.
      // in that case we assigned the node to that EP but do not compile it into a fused node.
      // this keeps the original node and prevents level 2 and level 3 optimizers from modifying it.
      // we now revert to the CPU EP to include the hash for the kernel as a fallback. at runtime when the model
      // is loaded in a minimal build, the compiling EP will replace this node if possible. if that's not possible for
      // some reason we can fallback to the CPU EP implementation via this hash.
      node.SetExecutionProviderType(kCpuExecutionProvider);
      status = kernel_registry_manager.SearchKernelRegistry(node, &kci);
    }

    ORT_RETURN_IF_ERROR(status);

    ORT_IGNORE_RETURN_VALUE(
        kernel_create_info_map_.insert({node.Index(), gsl::not_null<const KernelCreateInfo*>(kci)}));
  }

  for (const auto& entry : subgraph_session_states_) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      ORT_RETURN_IF_ERROR(subgraph_session_state.PopulateKernelCreateInfo(kernel_registry_manager, saving_ort_format));
    }
  }

  return Status::OK();
}
#endif

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
    session_kernels_.resize(max_nodeid + 1, nullptr);
    for (const auto& node : nodes) {
      // construct and save the kernels
      const KernelCreateInfo& kci = GetNodeKernelCreateInfo(node.Index());

      // the execution provider was required to be valid to find the KernelCreateInfo so we don't need to check it here
      onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();
      const IExecutionProvider& exec_provider = *execution_providers_.Get(exec_provider_name);

      auto op_kernel = kernel_registry_manager.CreateKernel(node, exec_provider, *this, kci);

      // assumes vector is already resize()'ed to the number of nodes in the graph
      session_kernels_[node.Index()] = op_kernel.release();
    }
  }
  node_index_info_ = std::make_unique<NodeIndexInfo>(*graph_viewer_, ort_value_name_idx_map_);
  return Status::OK();
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const { return p_seq_exec_plan_.get(); }

Status SessionState::AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d,
                                          bool constant, bool sparse) {
  auto p = initialized_tensors_.insert({ort_value_index, ort_value});
  if (!p.second)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "duplicated ort_value index:", ort_value_index,
                           ". Do you have duplicated calls to SessionState::AddInitializedTensor function?");

  if (d != nullptr && d->f != nullptr) {
    deleter_for_initialized_tensors_[ort_value_index] = *d;
  }

  if (constant) {
    constant_initialized_tensors_.insert({ort_value_index, ort_value});
  }

  if (sparse) {
    sparse_initialized_tensors_.insert(ort_value_index);
  }

  return Status::OK();
}

const std::unordered_map<int, OrtValue>& SessionState::GetInitializedTensors() const { return initialized_tensors_; }

const std::unordered_map<int, OrtValue>& SessionState::GetConstantInitializedTensors() const {
  return constant_initialized_tensors_;
}

bool SessionState::IsSparseInitializer(int ort_value_index) const {
  return sparse_initialized_tensors_.count(ort_value_index) > 0;
}

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
  std::vector<BufferUniquePtr> shared_prepacked_buffers;
  shared_prepacked_buffers.reserve(4);  // Unlikely to see more than 4 prepacked buffers per initializer

  for (const auto& prepacked_buffer : prepacked_weights.buffers_) {
    // BufferDeleter is nullptr because the kernel should not delete the shared buffer - it can only use it
    shared_prepacked_buffers.emplace_back(prepacked_buffer.get(), BufferDeleter(nullptr));
  }

  bool used_shared_buffers = false;
  ORT_RETURN_IF_ERROR(kernel.UseSharedPrePackedBuffers(shared_prepacked_buffers, input_idx, used_shared_buffers));

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

Status SessionState::PrepackConstantInitializedTensors(std::unordered_map<std::string, size_t>& constant_initializers_use_count,
                                                       const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map) {
  auto prepacked_constant_weights = [this, &constant_initializers_use_count, &initializers_to_share_map](
                                        bool should_cache_prepacked_weights_for_shared_initializers) -> Status {
    for (auto& node : GetGraphViewer().Nodes()) {
      auto kernel = GetMutableKernel(node.Index());
      int input_idx = 0;
      for (auto& input_def : node.InputDefs()) {
        if (input_def->Exists()) {
          const std::string& input_name = input_def->Name();
          SessionState* st = this;
          // subgraph can use the value from outer scope,
          // so it needs to check if current node uses constant initialized tensor from current and outer graphs
          do {
            int ort_value_idx;
            if (st->GetOrtValueNameIdxMap().GetIdx(input_name, ort_value_idx).IsOK()) {
              std::unordered_map<int, OrtValue>& constant_initialized_tensors = st->constant_initialized_tensors_;

              if (constant_initialized_tensors.count(ort_value_idx)) {
                bool is_packed = false;
                const Tensor& const_initialized_tensor = constant_initialized_tensors[ort_value_idx].Get<Tensor>();

                auto iter = initializers_to_share_map.find(input_name);
                bool is_shared_initializer = (iter != initializers_to_share_map.end());

                // Caching pre-packed weights is limited to shared initializers associated with the CPU EP for now
                if (is_shared_initializer && should_cache_prepacked_weights_for_shared_initializers &&
                    node.GetExecutionProviderType() == kCpuExecutionProvider) {  // caching of pre-packed weights' turned ON

                  AllocatorPtr allocator_for_caching = prepacked_weights_container_->GetOrCreateAllocator(CPU);
                  ORT_ENFORCE(allocator_for_caching.get() != nullptr);

                  PrePackedWeights weights_to_be_filled_in;
                  // The reason we invoke PrePack() before looking into the container for any pre-packed weight
                  // cached by another instance of the same op_type (for the same constant initializer) is because
                  // to truly know if we can use a cached pre-packed weight, we would have to compare the cached pre-packed
                  // weight with the pre-packed weight generated by this instance of the same op_type because other static
                  // properties of the node like node attributes could play a role in the pre-packed weights' contents.
                  ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx, allocator_for_caching,
                                                      is_packed,
                                                      &weights_to_be_filled_in));

                  if (is_packed) {
                    // BUG CHECK: Ensure that the kernel has filled in the pre-packed weight to be cached if the weight was pre-packed
                    ORT_ENFORCE(weights_to_be_filled_in.buffers_.size() > 0, "The kernel corresponding to the node ", node.Name(),
                                " doesn't have an implementation that can cache computed pre-packed weights");

                    const auto& op_type = node.OpType();

                    // Sanity check
                    // TODO: Check if some version of the ONNX IR allows op_type to be empty
                    ORT_ENFORCE(!op_type.empty(), "The op type of a node cannot be empty");

                    // The key for the pre-packed weights container lookup is the op_type + hash of the prepacked-weight
                    // that we just got by invoking PrePack() on this kernel.

                    const std::string& prepacked_weights_container_key = GenerateKeyForPrepackedWeightsMap(op_type,
                                                                                                           weights_to_be_filled_in);

                    bool container_contains_packed_weight = prepacked_weights_container_->HasWeight(prepacked_weights_container_key);

                    if (container_contains_packed_weight) {
                      LOGS(logger_, INFO) << "Using cached version of pre-packed weight for constant initializer: " << input_name
                                          << " used in the node: " << node.Name() << " which is of op type: " << node.OpType();

                      ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                          prepacked_weights_container_->GetWeight(prepacked_weights_container_key),
                                                                          node.Name()));

                      ++used_shared_pre_packed_weights_counter_;
                    } else {  // container doesn't contain the pre-packed weight - so write into it for sharing across kernel instances

                      if (!prepacked_weights_container_->WriteWeight(prepacked_weights_container_key, std::move(weights_to_be_filled_in))) {
                        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to write the provided PrePackedWeights instance into the container");
                      }

                      ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                          prepacked_weights_container_->GetWeight(prepacked_weights_container_key),
                                                                          node.Name()));
                    }
                  }

                } else {  // caching of pre-packed weights' turned OFF
                  AllocatorPtr session_cpu_alloc = kernel->Info().GetAllocator(0, OrtMemType::OrtMemTypeDefault);
                  ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx,
                                                      session_cpu_alloc,  // use allocator tied to this session
                                                      is_packed,
                                                      nullptr  // no caching required
                                                      ));
                }
                if (is_packed) {
                  ++number_of_prepacks_counter_;

                  if (constant_initializers_use_count.count(input_name) && --constant_initializers_use_count[input_name] == 0) {
                    // release the constant initialized tensor
                    st->initialized_tensors_.erase(ort_value_idx);
                    constant_initialized_tensors.erase(ort_value_idx);
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
          } while (st);
        }
        input_idx++;
      }
    }

    return Status::OK();
  };

  bool should_cache_prepacked_weights_for_shared_initializers = (prepacked_weights_container_ != nullptr);

  if (should_cache_prepacked_weights_for_shared_initializers) {
    // serialize calls to the method that looks up the container, calls UseCachedPrePackedWeight/PrePack
    // and writes pre-packed weights to the container
    std::lock_guard<onnxruntime::OrtMutex> l(prepacked_weights_container_->mutex_);
    return prepacked_constant_weights(true);
  } else {
    return prepacked_constant_weights(false);
  }
}

static int64_t CalculateMemoryPatternsKey(const std::vector<std::reference_wrapper<const TensorShape>>& shapes) {
  int64_t key = 0;
  for (auto shape : shapes) {
    for (auto dim : shape.get().GetDims()) key ^= dim;
  }
  return key;
}

#ifdef ENABLE_TRAINING
namespace {
Status ResolveDimParams(const GraphViewer& graph,
                        const std::map<std::string, TensorShape>& feeds,
                        std::unordered_map<std::string, int64_t>& out) {
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
    const std::unordered_map<std::string, int64_t>& symbolic_dimensions,
    size_t& is_resolved,  // indicate whether resolve successfully or not.
    std::vector<int64_t>& resolved_shape) {
  if (!arg->Shape()) {
    is_resolved = 0;
    return Status::OK();
  }

  std::vector<int64_t> shape;

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

void TryCalculateSizeFromResolvedShape(int ml_value_idx, std::unordered_map<int, TensorShape>& resolved_shapes, size_t& size) {
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
Status SessionState::GeneratePatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
                                               const std::vector<int>& feed_mlvalue_idxs,
                                               MemoryPatternGroup* output,
                                               std::unordered_map<int, TensorShape>& resolved_shapes) const {
  std::map<std::string, TensorShape> feeds;
  for (size_t i = 0, end = feed_mlvalue_idxs.size(); i < end; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(feed_mlvalue_idxs[i], name));
    feeds.insert({name, input_shape[i]});
  }
  std::unordered_map<std::string, int64_t> map;
  ORT_RETURN_IF_ERROR(ResolveDimParams(*graph_viewer_, feeds, map));
  auto* exe_plan = GetExecutionPlan();
  ORT_ENFORCE(exe_plan);
  OrtValuePatternPlanner mem_planner(*exe_plan, /*using counters*/ true);

  // Try to resolve shapes for activations.
  auto& node_index_info = GetNodeIndexInfo();
  for (auto& node_plan : exe_plan->execution_plan) {
    int node_index = node_index_info.GetNodeOffset(node_plan.node_index);
    auto* node = graph_viewer_->GetNode(node_plan.node_index);
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
      std::vector<int64_t> resolved_shape;

      // Tensors whose shape cannot be resolved statically will be allocated at runtime.
      if (TryResolveShape(arg, map, is_resolved, resolved_shape).IsOK()) {
        // Store all valid resolved shapes. They will be queried in, for example,
        // Recv operator to bypass the dependency of output shapes on inputs.
        if (is_resolved != 0) {
          resolved_shapes[ml_value_idx] = resolved_shape;
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
      mem_planner.TraceAllocation(ml_value_idx, counter, size);
    }
  }

  // Allocate all other activations.
  for (auto& node_plan : exe_plan->execution_plan) {
    int node_index = node_index_info.GetNodeOffset(node_plan.node_index);
    auto* node = graph_viewer_->GetNode(node_plan.node_index);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) +
                       static_cast<int>(node->ImplicitInputDefs().size());
    //allocate output
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
        mem_planner.TraceAllocation(ml_value_idx, counter, aligned_size);
      }
    }

    //release nodes
    for (int index = node_plan.free_from_index; index <= node_plan.free_to_index; ++index) {
      auto ml_value_idx = exe_plan->to_be_freed[index];
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      if (ml_data_type != DataTypeImpl::GetType<std::string>()) {
        mem_planner.TraceFree(ml_value_idx);
      }
    }
  }

  if (!mem_planner.GeneratePatterns(output).IsOK()) {
    return Status(ONNXRUNTIME, FAIL, "Generate Memory Pattern failed");
  }
  return Status::OK();
}
#endif

const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
                                                              const std::vector<int>& feed_mlvalue_idxs,
                                                              std::unordered_map<int, TensorShape>& inferred_shapes) const {
  int64_t key = CalculateMemoryPatternsKey(input_shapes);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
#ifdef ENABLE_TRAINING
    auto mem_patterns = std::make_unique<MemoryPatternGroup>();
    if (GeneratePatternGroupCache(input_shapes, feed_mlvalue_idxs, mem_patterns.get(), inferred_shapes).IsOK()) {
      key = CalculateMemoryPatternsKey(input_shapes);
      auto ptr = mem_patterns.get();
      mem_patterns_[key] = std::move(mem_patterns);
      shape_patterns_[key] = inferred_shapes;
      return ptr;
    }
    return nullptr;
#else
    ORT_UNUSED_PARAMETER(feed_mlvalue_idxs);
    return nullptr;
#endif
  }

  inferred_shapes = shape_patterns_[key];
  return it->second.get();
}

void SessionState::ResolveMemoryPatternFlag() {
  if (enable_mem_pattern_) {
    for (auto* input : graph_viewer_->GetInputs()) {
      if (!input->HasTensorOrScalarShape()) {
        enable_mem_pattern_ = false;
        break;
      }
    }
  }
}

Status SessionState::UpdateMemoryPatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
                                                   std::unique_ptr<MemoryPatternGroup> mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(input_shapes);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
    mem_patterns_[key] = std::move(mem_patterns);
  }

  return Status::OK();
}

bool SessionState::GetEnableMemoryPattern() const { return enable_mem_pattern_; }

bool SessionState::GetEnableMemoryReuse() const { return enable_mem_reuse_; }

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
                                              std::vector<NodeInfo>& node_info_vec) const {
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
                                               std::vector<NodeInfo>& node_info_vec) const {
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
  ORT_ENFORCE(node_index_info_, "SetGraphAndCreateKernels must be called prior to GetExecutionInfo.");
  return *node_index_info_;
}

static std::string GetSubGraphId(const NodeIndex node_idx, const std::string& attr_name) {
  return std::to_string(node_idx) + "_" + attr_name;
}

#if !defined(ORT_MINIMAL_BUILD)
void SessionState::UpdateToBeExecutedNodes(const std::vector<int>& fetch_mlvalue_idxs) {
  std::vector<int> sorted_idxs = fetch_mlvalue_idxs;
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  if (to_be_executed_nodes_.find(sorted_idxs) != to_be_executed_nodes_.end())
    return;

  // Get the nodes generating the fetches.
  std::vector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  std::unordered_set<NodeIndex> reachable_nodes;

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
  to_be_executed_nodes_.insert(std::make_pair(sorted_idxs, reachable_nodes));
}

const std::unordered_set<NodeIndex>* SessionState::GetToBeExecutedNodes(
    const std::vector<int>& fetch_mlvalue_idxs) const {
  std::vector<int> sorted_idxs = fetch_mlvalue_idxs;
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  auto it = to_be_executed_nodes_.find(sorted_idxs);
  return (it != to_be_executed_nodes_.end()) ? &it->second : nullptr;
}

static Status GetSubGraphSessionStatesOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const std::unordered_map<NodeIndex, std::unordered_map<std::string, std::unique_ptr<SessionState>>>& subgraph_session_states,
    std::vector<flatbuffers::Offset<fbs::SubGraphSessionState>>& fbs_subgraph_session_states) {
  fbs_subgraph_session_states.clear();
  for (const auto& pair : subgraph_session_states) {
    const auto node_idx = pair.first;
    const auto& session_states = pair.second;
    for (const auto& name_to_subgraph_session_state : session_states) {
      const std::string& attr_name = name_to_subgraph_session_state.first;
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      auto graph_id = builder.CreateString(GetSubGraphId(node_idx, attr_name));
      flatbuffers::Offset<fbs::SessionState> session_state;
      ORT_RETURN_IF_ERROR(
          subgraph_session_state.SaveToOrtFormat(builder, session_state));

      fbs_subgraph_session_states.push_back(
          fbs::CreateSubGraphSessionState(builder, graph_id, session_state));
    }
  }
  return Status::OK();
}

Status SessionState::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                     flatbuffers::Offset<fbs::SessionState>& fbs_session_state) const {
  size_t size = kernel_create_info_map_.size();
  std::vector<uint32_t> node_indices;
  std::vector<uint64_t> kernel_def_hashes;
  node_indices.reserve(size);
  kernel_def_hashes.reserve(size);
  for (const auto& kvp : kernel_create_info_map_) {
    node_indices.push_back(gsl::narrow<uint32_t>(kvp.first));
    kernel_def_hashes.push_back(kvp.second->kernel_def->GetHash());
  }

  auto kernels = fbs::CreateKernelCreateInfosDirect(builder, &node_indices, &kernel_def_hashes);

  // Subgraph session states
  std::vector<flatbuffers::Offset<fbs::SubGraphSessionState>> sub_graph_session_states;
  ORT_RETURN_IF_ERROR(
      GetSubGraphSessionStatesOrtFormat(builder, subgraph_session_states_, sub_graph_session_states));

  fbs_session_state = fbs::CreateSessionStateDirect(builder, kernels, &sub_graph_session_states);
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Status SessionState::CreateSubgraphSessionState() {
  for (auto& node : graph_.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      const auto& ep = node.GetExecutionProviderType();
      if (ep != kCpuExecutionProvider && ep != kCudaExecutionProvider) {
        // SessionState is only used when ORT is executing the subgraph. If a non-ORT EP has taken the control flow
        // node containing the subgraph it will create whatever state it needs internally.
        continue;
      }

      auto& attr_name = entry.first;
      Graph* subgraph = entry.second;
      ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

      auto subgraph_session_state =
          std::make_unique<SessionState>(*subgraph, execution_providers_, enable_mem_pattern_,
                                         thread_pool_, inter_op_thread_pool_, data_transfer_mgr_,
                                         logger_, profiler_);

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

#if defined(ENABLE_ORT_FORMAT_LOAD)
Status SessionState::LoadFromOrtFormat(const fbs::SessionState& fbs_session_state,
                                       const KernelRegistryManager& kernel_registry_manager) {
  const auto* fbs_kcis = fbs_session_state.kernels();
  ORT_RETURN_IF(nullptr == fbs_kcis, "Kernel create info is null. Invalid ORT format model.");
  auto* node_indices = fbs_kcis->node_indices();
  auto* kernel_def_hashes = fbs_kcis->kernel_def_hashes();
  ORT_RETURN_IF(nullptr == node_indices, "Kernel create info node indices are null. Invalid ORT format model.");
  ORT_RETURN_IF(nullptr == kernel_def_hashes, "Kernel create info hashes are null. Invalid ORT format model.");
  ORT_RETURN_IF_NOT(node_indices->size() == kernel_def_hashes->size(),
                    "Size mismatch for kernel create info node indexes and hashes. Invalid ORT format model.",
                    node_indices->size(), " != ", kernel_def_hashes->size());

  auto add_kernel_by_hash =
      [&kernel_registry_manager, this](const Node& node, uint64_t hash) {
        const KernelCreateInfo* kci = nullptr;
        ORT_RETURN_IF_ERROR(kernel_registry_manager.SearchKernelRegistry(node, hash, &kci));
        kernel_create_info_map_.emplace(node.Index(), gsl::not_null<const KernelCreateInfo*>(kci));
        return Status::OK();
      };

  // kernel hashes for model are in top level SessionState
  const auto& compiled_kernel_hashes = GetCompiledKernelHashes();

  // process the nodes that existed when the model was created
  for (flatbuffers::uoffset_t i = 0; i < node_indices->size(); i++) {
    auto node_idx = node_indices->Get(i);
    auto kernel_hash = kernel_def_hashes->Get(i);

    const Node* node = graph_.GetNode(node_idx);
    if (node == nullptr) {
      // this is OK if we have compiled kernels and the original node was replaced. if not the model is invalid.
      ORT_RETURN_IF(compiled_kernel_hashes.empty(),
                    "Can't find node with index ", node_idx, ". Invalid ORT format model.");
      continue;
    }

    ORT_RETURN_IF_ERROR(add_kernel_by_hash(*node, kernel_hash));
  }

  // lookup the hashes for any nodes we compiled. the nodes indexes for compiled nodes are not in node_indices
  // as they were created at runtime.
  if (!compiled_kernel_hashes.empty()) {
    for (const auto& node : graph_.Nodes()) {
      if (kernel_create_info_map_.count(node.Index()) == 0) {
        auto hash_info = compiled_kernel_hashes.find(node.OpType());
        ORT_RETURN_IF(hash_info == compiled_kernel_hashes.cend(),
                      "Unable to find compiled kernel hash for node '", node.Name(), "'.")

        ORT_RETURN_IF_ERROR(add_kernel_by_hash(node, hash_info->second));
      }
    }
  }

  if (!subgraph_session_states_.empty()) {
    auto* fbs_sub_graph_session_states = fbs_session_state.sub_graph_session_states();
    ORT_RETURN_IF(nullptr == fbs_sub_graph_session_states,
                  "SessionState for subgraphs is null. Invalid ORT format model.");

    for (const auto& pair : subgraph_session_states_) {
      const auto node_idx = pair.first;
      const auto& session_states = pair.second;
      for (const auto& name_to_subgraph_session_state : session_states) {
        const std::string& attr_name = name_to_subgraph_session_state.first;
        SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;

        // Use the graphid as the key to search the for the fbs::SubGraphSessionState
        std::string key = GetSubGraphId(node_idx, attr_name);
        auto* fbs_sub_graph_ss = fbs_sub_graph_session_states->LookupByKey(key.c_str());
        ORT_RETURN_IF(nullptr == fbs_sub_graph_ss,
                      "Subgraph SessionState entry for ", key, " is missing. Invalid ORT format model.");

        auto* fbs_sub_session_state = fbs_sub_graph_ss->session_state();
        ORT_RETURN_IF(nullptr == fbs_sub_session_state,
                      "Subgraph SessionState for ", key, " is null. Invalid ORT format model.");
        subgraph_session_state.LoadFromOrtFormat(*fbs_sub_session_state, kernel_registry_manager);
      }
    }
  }

  return Status::OK();
}
#endif

// Calculate the use count of a constant initialized tensor, including the use in subgraph.
// Note: This function doesn't handle the case below:
// The main graph has a constant initializer called X, and the subgraph also has a constant initializer called X, which overrides the X from main graph.
// For case like this, the current implementation will calculate the use count as 2, but they could contain completely different values so each should have a use count of 1.
// This is a very rare case. If it happens and X is prepacked, the consequence is that X won't be released and memory usage of X won't be saved. This will be fine.
static void ComputeConstantInitializerUseCount(const Graph& graph, std::unordered_map<std::string, size_t>& constant_initializers_use_count) {
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

Status SessionState::FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                          KernelRegistryManager& kernel_registry_manager,
                                          const SessionOptions& session_options,
                                          const onnxruntime::experimental::fbs::SessionState* serialized_session_state,
                                          bool remove_initializers,
                                          bool saving_ort_format) {
  // recursively create the subgraph session state instances and populate the kernel create info in them.
  // it's simpler to handle the kernel create info recursively when deserializing,
  // so also do it recursively when calling PopulateKernelCreateInfo for consistency.
  ORT_RETURN_IF_ERROR(CreateSubgraphSessionState());

  if (serialized_session_state) {
#if defined(ENABLE_ORT_FORMAT_LOAD)
    ORT_RETURN_IF_ERROR(LoadFromOrtFormat(*serialized_session_state, kernel_registry_manager));
#else
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "ORT format model is not supported in this build.");
#endif

  } else {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_RETURN_IF_ERROR(PopulateKernelCreateInfo(kernel_registry_manager, saving_ort_format));
#else
    ORT_UNUSED_PARAMETER(graph_location);
    ORT_UNUSED_PARAMETER(kernel_registry_manager);
    ORT_UNUSED_PARAMETER(session_options);
    ORT_UNUSED_PARAMETER(remove_initializers);
    ORT_UNUSED_PARAMETER(saving_ort_format);
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Serialized session state must be provided from an ORT format model in this build.");
#endif
  }

  std::unordered_map<std::string, size_t> constant_initializers_use_count;
  ComputeConstantInitializerUseCount(graph_, constant_initializers_use_count);
  return FinalizeSessionStateImpl(graph_location, kernel_registry_manager, nullptr, session_options,
                                  remove_initializers, constant_initializers_use_count);
}

Status SessionState::FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                              KernelRegistryManager& kernel_registry_manager,
                                              _In_opt_ const Node* parent_node,
                                              const SessionOptions& session_options,
                                              bool remove_initializers,
                                              std::unordered_map<std::string, size_t>& constant_initializers_use_count) {
  CreateGraphInfo();

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  std::vector<const NodeArg*> valid_outer_scope_node_args;
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

  SequentialPlannerContext context(session_options.execution_mode, session_options.execution_order, session_options.enable_mem_reuse);
  ORT_RETURN_IF_ERROR(SequentialPlanner::CreatePlan(parent_node, *graph_viewer_, valid_outer_scope_node_args,
                                                    execution_providers_, kernel_create_info_map_,
                                                    ort_value_name_idx_map_, context, p_seq_exec_plan_));
  //Record the allocation plan

  // Uncomment the below to dump the allocation plan to std::cout
  // LOGS(logger_, VERBOSE) << std::make_pair(p_seq_exec_plan_.get(), this);
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryInfo::GenerateTensorMap(GetExecutionPlan(), GetOrtValueNameIdxMap());
#endif

  // Memory pattern tracer allocates all initializers on a single continous
  // buffer. This has the effect of reducing memory fragementation.
  // Further more, NCCL kernels require initializers to be allocated
  // continously.
  //
  // In inferencing scenarios, however, we often want to pre-process and then
  // release some initializers. See OpKernel::PrePack(). Letting all initializers
  // sharing a single buffer makes it hard to release individual ones, leading
  // to memory waste.
  //
  // TODO!! disabling memory pattern tracer increases fragementation, leading to
  //  out of memory error in some training tests. Need to create kernel first,
  //  and let the kernel tells us whether the initalizer needs to be traced.
  //
#if defined(ENABLE_TRAINING)
  std::unique_ptr<ITensorAllocator> tensor_allocator(
      ITensorAllocator::Create(enable_mem_pattern_, *p_seq_exec_plan_, *this, weights_buffers_));
#else
  std::unique_ptr<ITensorAllocator> tensor_allocator(
      ITensorAllocator::Create(false, *p_seq_exec_plan_, *this, weights_buffers_));
#endif

  const auto& initializer_allocation_order = p_seq_exec_plan_->initializer_allocation_order;

  // move initializers from TensorProto instances in Graph to OrtValue instances in SessionState
  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInitializedTensors(
          Env::Default(), graph_location, *graph_viewer_,
          execution_providers_.GetDefaultCpuAllocator(),
          ort_value_name_idx_map_, initializer_allocation_order, *tensor_allocator,
          [this](int idx, const OrtValue& value, const OrtCallback& d, bool constant, bool sparse) -> Status {
            return AddInitializedTensor(idx, value, &d, constant, sparse);
          },
          logger_, data_transfer_mgr_, *p_seq_exec_plan_.get(), session_options));
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  //Record Weight allocation info on device
  MemoryInfo::RecordInitializerAllocInfo(GetInitializedTensors());
#endif

  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  if (remove_initializers) {
    CleanInitializedTensorsFromGraph();
  }

  ORT_RETURN_IF_ERROR(CreateKernels(kernel_registry_manager));

#ifndef ENABLE_TRAINING
  const auto disable_prepacking =
      session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisablePrepacking, "0");

  if (disable_prepacking != "1") {
    ORT_RETURN_IF_ERROR(PrepackConstantInitializedTensors(constant_initializers_use_count,
                                                          session_options.initializers_to_share_map));
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
      ORT_RETURN_IF_ERROR(subgraph_session_state.FinalizeSessionStateImpl(
          graph_location, kernel_registry_manager, &node, subgraph_session_options, remove_initializers, constant_initializers_use_count));

      // setup all the info for handling the feeds and fetches used in subgraph execution
      auto* p_op_kernel = GetMutableKernel(node.Index());
      ORT_ENFORCE(p_op_kernel);

      // Downcast is safe, since only control flow nodes have subgraphs
      // (node.GetAttributeNameToMutableSubgraphMap() is non-empty)
      auto& control_flow_kernel = static_cast<controlflow::IControlFlowKernel&>(*p_op_kernel);
      ORT_RETURN_IF_ERROR(control_flow_kernel.SetupSubgraphExecutionInfo(*this, attr_name, subgraph_session_state));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
