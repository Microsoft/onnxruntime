// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

namespace {
//It assumes max(OrtMemType) <= 1, min(OrtMemType) = -2
inline int MakeKey(int id, OrtMemType mem_type) {
  return id << 2 | (mem_type + 2);
}
}  // namespace

AllocatorPtr IExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
#if !defined(ORT_MINIMAL_BUILD)
  for (auto& node : graph.Nodes()) {
    for (auto registry : kernel_registries) {
      if (KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
        break;
      }
    }
  }

  return result;
#else
  // We have saved hashes to lookup static kernels in an ORT format model so the default behavior is to return an
  // empty vector to leave that in place. An EP that compiles nodes can override this in a minimal build.
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(kernel_registries);
  return result;
#endif
}

// Returns true if an allocator was found and replaced
// Optionally user can provide an OrtMemoryInfo to swap in place
// of an OrtMemoryInfo whose corresponding allocator is being replaced.
static bool FindAndReplaceAllocator(const OrtMemoryInfo& mem_info,
                                    MemoryInfoSet& mem_info_set,
                                    AllocatorMap& allocators,
                                    AllocatorPtr replacing_allocator,
                                    const OrtMemoryInfo* mem_info_to_be_swapped_in = nullptr) {
  auto ite = mem_info_set.find(mem_info);
  if (ite != mem_info_set.end()) {
    const int key = MakeKey(mem_info.id, mem_info.mem_type);
    allocators[key] = replacing_allocator;

    if (mem_info_to_be_swapped_in != nullptr) {
      mem_info_set.erase(ite);
      mem_info_set.insert(*mem_info_to_be_swapped_in);
    }
    return true;
  }

  return false;
}

// Update allocator in the provider if already present; ignore if not.
void IExecutionProvider::ReplaceAllocator(AllocatorPtr allocator) {
  const auto& info = allocator->Info();

  if (FindAndReplaceAllocator(info, mem_info_set_, allocators_, allocator)) {
    // We found an allocator corresponding to the provided
    // allocator's OrtmemoryInfo, we hence return
    return;
  }

  else {
    // If we can't find an allocator registered with the exact OrtMemoryInfo
    // as that of the replacing allocator, we do a "loosened" check
    // (i.e.) check if there is an allocator registered with OrtAllocatorType
    // as OrtArenaAllocator because for external user provided allocator
    // we only accept OrtAllocatorType as OrtDeviceAllocator.
    // If we do find such a registered allocator, we can safely go ahead
    // and replace that with the provided allocator. This may seem like
    // we are replacing an arena allocator with a non-arena allocator
    // but in reality any user provided allocator may still be an arena
    // allocator. We don't allow users to use OrtAllocatorType as
    // OrtArenaAllocator for their allocators because we reserve its usage
    // for our internal BFCArena.
    // TODO: Should we remove the OrtAllocatorType field from OrtmemoryArena to
    // avoid such problems and also remove the unintuitive phenomenon of binding
    // the allocator type info to OrtmemoryInfo (which loosely is just device info) ?
    const auto& original_info = allocator->Info();

    // If the alloc_type was OrtArenaAllocator already, then it is a no-op
    if (original_info.alloc_type == OrtAllocatorType::OrtArenaAllocator) {
      return;
    }

    auto check_info = original_info;

    // Mutate the alloc_type
    check_info.alloc_type = OrtAllocatorType::OrtArenaAllocator;

    FindAndReplaceAllocator(check_info, mem_info_set_,
                            allocators_, allocator,
                            &allocator->Info());
  }
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    ORT_THROW("duplicated allocator");
  }
  const int key = MakeKey(info.id, info.mem_type);
  allocators_.insert({key, allocator});
  mem_info_set_.insert(ite, info);
  allocator_list_.push_back(allocator);
}

void IExecutionProvider::TryInsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    LOGS_DEFAULT(WARNING) << "duplicated allocator: " << info.ToString();
    return;
  }
  InsertAllocator(allocator);
}

void IExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager>) {
  return;
}

#if !defined(ORT_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with fused Node is not implemented by " + type_);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::string& /*dll_path*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with fused Node and dll path is not implemented by " + type_);
}
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with FusedNodeAndGraph is not implemented by " + type_);
}
#endif

int IExecutionProvider::ModelMetadefIdGenerator::GenerateId(const onnxruntime::GraphViewer& graph_viewer,
                                                            uint64_t& model_hash) {
  model_hash = 0;

  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  uint32_t instance_hash[4] = {0, 0, 0, 0};

  const Graph& main_graph = *cur_graph;

  // hash the bytes in the Graph instance. we can't just use the address as a new Graph instance may use
  // the same memory (unit tests prove this can occur). the raw bytes of the Graph instance should be a unique
  // fingerprint for the instance that can use used as the key to the hash of the model path/contents.
  MurmurHash3::x86_128(&main_graph, gsl::narrow_cast<int32_t>(sizeof(Graph)), instance_hash[0], &instance_hash);
  uint64_t graph_instance_hash = instance_hash[0] | (uint64_t(instance_hash[1]) << 32);

  // if we've already hashed this main graph instance use the cached value
  auto entry = main_graph_hash_.find(graph_instance_hash);
  if (entry != main_graph_hash_.cend()) {
    model_hash = entry->second;
  } else {
    uint32_t hash[4] = {0, 0, 0, 0};

    // prefer path the model was loaded from
    // this may not be available if the model was loaded from a stream or in-memory bytes
    const auto& model_path_str = main_graph.ModelPath().ToPathString();
    if (!model_path_str.empty()) {
      MurmurHash3::x86_128(model_path_str.data(), gsl::narrow_cast<int32_t>(model_path_str.size()), hash[0], &hash);
    } else {
      auto hash_str = [&hash](const std::string& str) {
        MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
      };

      // fingerprint the main graph by hashing graph inputs and the ordered outputs from each node
      for (const auto* node_arg : main_graph.GetInputsIncludingInitializers()) {
        hash_str(node_arg->Name());
      }

      // note: process nodes in order defined in model to be deterministic
      for (const auto& node : main_graph.Nodes()) {
        for (const auto* node_arg : node.OutputDefs()) {
          if (node_arg->Exists()) {
            hash_str(node_arg->Name());
          }
        }
      }
    }

    model_hash = hash[0] | (uint64_t(hash[1]) << 32);

    main_graph_hash_[graph_instance_hash] = model_hash;
  }

  // return the current unique id, and increment to update
  return model_metadef_id_[model_hash]++;
}

int IExecutionProvider::GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) const {
  ORT_ENFORCE(metadef_id_generator_,
              "IExecutionProvider constructor must be called with true for use_metadef_id_creator");

  // if the EP is shared across multiple sessions there's a very small potential for concurrency issues.
  // use a lock when generating an id to be paranoid
  static OrtMutex mutex;
  std::lock_guard<OrtMutex> lock(mutex);
  return metadef_id_generator_->GenerateId(graph_viewer, model_hash);
}

}  // namespace onnxruntime
