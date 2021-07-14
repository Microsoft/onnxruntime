// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/func_kernel.h"

// uncomment this line to count non-CUDA ops in ONNX domain
//#define COUNT_NON_CUDA_OPS

#ifdef COUNT_NON_CUDA_OPS
class NonCudaOps {
 public:
  ~NonCudaOps() {
    printf("Non-CUDA ops:\n");
    for (auto i : map_) {
      printf("%s: %d\n", i.first.c_str(), i.second);
    }
  }

  void AddOp(const std::string& name) {
    if (map_.count(name))
      map_.at(name)++;
    else
      map_.insert({name, 1});
  }

 private:
  std::map<std::string, int> map_;
};

NonCudaOps non_cuda;
#endif

using namespace ::onnxruntime::common;
namespace onnxruntime {

// minimal KernelDef based on MetaDef instead of a Function based node
static void BuildFusedKernelDef(KernelDefBuilder& builder, const IndexedSubGraph::MetaDef& metadef,
                                const std::string& provider_type) {
  builder.SetName(metadef.name)
      .SetDomain(metadef.domain)
      .SinceVersion(metadef.since_version)
      .Provider(provider_type);
}

#if !defined(ORT_MINIMAL_BUILD)
static void BuildFusedKernelDef(KernelDefBuilder& builder, const onnxruntime::Node& node) {
  auto schema = node.Op();
  builder.SetName(schema->Name())
      .SetDomain(schema->domain())
      .SinceVersion(schema->SinceVersion())
      .Provider(node.GetExecutionProviderType());

  // process input/output MemType for fused node
  ProtoHelperNodeContext proto_ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&proto_ctx);
  std::vector<int64_t> input_mem_types;
  if (info.GetAttrs<int64_t>(KernelDefBuilder::NodeAttr_InputMemType, input_mem_types).IsOK()) {
    for (int i = 0; i < gsl::narrow<int>(input_mem_types.size()); ++i) {
      builder.InputMemoryType((OrtMemType)input_mem_types[i], i);
    }
  }
  std::vector<int64_t> output_mem_types;
  if (info.GetAttrs<int64_t>(KernelDefBuilder::NodeAttr_OutputMemType, output_mem_types).IsOK()) {
    for (int i = 0; i < gsl::narrow<int>(output_mem_types.size()); ++i) {
      builder.OutputMemoryType((OrtMemType)output_mem_types[i], i);
    }
  }
}

/**
 * Check if a node can be placed on a specific provider.
 * Do nothing if the node is already assigned
 * \param graph
 * \param capability
 * \param kernel_registry_mgr
 * \param provider_type name of the provider to test
 * \param count A counter for generating fused node names. Unique across the entire model.
 * \return Fused node. Return nullptr if there is no fuse
 */
static Node* PlaceNode(Graph& graph, const IndexedSubGraph& capability,
                       const KernelRegistryManager& kernel_registry_mgr, const std::string& provider_type,
                       IExecutionProvider::FusionStyle fusion_style,
                       GraphPartitioner::Mode mode,
                       int& fused_node_unique_id) {
  Node* result = nullptr;

  if (nullptr == capability.GetMetaDef()) {
    // The <provider> can run a single node in the <graph> if not using meta-defs.
    // A fused kernel is not supported in this case.
    ORT_ENFORCE(1 == capability.nodes.size());

    auto* node = graph.GetNode(capability.nodes[0]);
    if (nullptr != node && node->GetExecutionProviderType().empty()) {
      // The node was not fused or assigned. Assign it to this <provider>.
      node->SetExecutionProviderType(provider_type);
    }
  } else {
    // The <provider> can run a fused <sub_graph> in the <graph>.

    // Check whether any node in the <sub_graph> was already assigned. If so it cannot be stolen as assignment is done
    // in order of EP priority
    bool sub_graph_available_for_assignment = true;
    for (auto node_index : capability.nodes) {
      const auto* node = graph.GetNode(node_index);
      if (nullptr == node || !node->GetExecutionProviderType().empty()) {
        // if mode is kAssignOnly we want all nodes that can _potentially_ be taken by compiling EPs to be assigned,
        // so that we aggregate the nodes covered and ensure the original nodes remain in the ORT format model by
        // preventing level 2 and 3 optimizers from changing them. optimizers check the EP the node is assigned to
        // and only make changes if the EP is on the optimizer's list of supported EPs. an EP that compiles nodes
        // should never be on those lists.
        //
        // when the ORT format model is loaded we will process it normally with EP priority being applied for
        // whichever EPs are enabled at the time.
        //
        // e.g. an Android NNAPI EP may take different/overlapping nodes to a iOS CoreML EP.
        // We want the ORT format model to be able to be run as efficiently as possible on either platform,
        // so we want all the nodes that either may take to be preserved. If we did not do this we would
        // need to create one ORT format model for Android and one for iOS.
        if (mode != GraphPartitioner::Mode::kAssignOnly) {
          // The node was fused or assigned, so that the whole sub-graph will not be assigned to this <provider>
          // The assumption is that this <provider> can only run the sub-graph as a whole unit.
          sub_graph_available_for_assignment = false;
          break;
        }
      }
    }

    if (sub_graph_available_for_assignment) {
      if (mode == GraphPartitioner::Mode::kNormal) {
        std::ostringstream oss;
        oss << provider_type << "_" << capability.GetMetaDef()->name << "_" << fused_node_unique_id++;
        std::string node_name = oss.str();

        Node* fused_node = nullptr;
        if (fusion_style == IExecutionProvider::FusionStyle::Function) {
          fused_node = &graph.FuseSubGraph(capability, node_name);
        } else {
          // create a fused node without copying everything to a Function body. The IndexedSubGraph will be passed
          // through to Compile via a filtered GraphViewer.
          fused_node = &graph.BeginFuseSubGraph(capability, node_name);
        }

        fused_node->SetExecutionProviderType(provider_type);

        // searching in kernel registries, if no kernel registered for the fused_node, use compile approach
        if (!KernelRegistryManager::HasImplementationOf(kernel_registry_mgr, *fused_node, provider_type)) {
          result = fused_node;
        }
      } else {
        // assign the nodes in the indexed subgraph to the current EP so that level 2+ optimizers will not change them.
        // This is used when exporting an ORT format model to maintain the original nodes and re-do the fusion
        // at runtime. The original nodes provide a fallback if fewer nodes can be fused at runtime due to device
        // capabilities.
        for (auto node_index : capability.nodes) {
          auto* node = graph.GetNode(node_index);
          if (node != nullptr) {
            node->SetExecutionProviderType(provider_type);
          }
        }
      }
    }
  }

  return result;
}

// for the current EP, recursively iterate through the Graph and any nested subgraphs (recursion is bottom-up).
// assign any nodes to the EP that are currently unassigned, and that the EP can handle.
static Status PartitionOnnxFormatModelImpl(Graph& graph, bool export_dll, FuncManager& func_mgr,
                                           KernelRegistryManager& kernel_registry_mgr,
                                           KernelRegistry& fused_kernel_registry,
                                           IExecutionProvider& current_ep,
                                           GraphPartitioner::Mode mode,
                                           int& fused_node_unique_id) {
  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  // recurse into nested graphs first to partition bottom up.
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      // we pass through the export_dll value and FuncManager from the top level graph
      ORT_RETURN_IF_ERROR(PartitionOnnxFormatModelImpl(*subgraph, export_dll, func_mgr, kernel_registry_mgr,
                                                       fused_kernel_registry, current_ep, mode, fused_node_unique_id));
    }
  }

  // If an execution provider return the capability that he could run a sub-graph,
  // onnxruntime will fuse the sub-graph into a function node. if the execution provider
  // says he need to compile the graph at runtime (by need_compile flag),
  // onnxruntime will invoke the "Compile" method to get compiled binary.
  // There are two mode of compile, one is return the entry point to the compiled binary
  // directly, another is export the compiled binary to shared library for future reuse.

  // TODO: when the graph contain a function node, and user pass in the dll which could
  // run the function by SessionOption, we should create a function kernel for it and
  // delegate the compute to the functions inside the dlls.
  const std::string& type = current_ep.Type();
  auto fusion_style = current_ep.GetFusionStyle();
  std::vector<Node*> nodes_to_compile;

  GraphViewer graph_viewer(graph);
  std::vector<std::unique_ptr<ComputeCapability>> capabilities =
      current_ep.GetCapability(graph_viewer, kernel_registry_mgr.GetKernelRegistriesByProviderType(type));

  // filter out the ComputeCapability instances that do not need compiling so we have a std::vector that's 1:1 with
  // nodes_to_compile.
  std::vector<std::unique_ptr<ComputeCapability>> capabilities_to_compile;
  capabilities_to_compile.reserve(std::count_if(capabilities.cbegin(), capabilities.cend(),
                                                [](const std::unique_ptr<ComputeCapability>& entry) {
                                                  return entry != nullptr &&
                                                         entry->sub_graph != nullptr &&
                                                         entry->sub_graph->GetMetaDef() != nullptr;
                                                }));

  for (auto& capability : capabilities) {
    if (!capability || !capability->sub_graph) {  // in theory an EP could return an empty value...
      continue;
    }

    Node* n = PlaceNode(graph, *capability->sub_graph, kernel_registry_mgr, type, fusion_style, mode, fused_node_unique_id);
    if (n != nullptr) {
      nodes_to_compile.push_back(n);
      capabilities_to_compile.push_back(std::move(capability));
    }
  }

  // NOTE: if mode_ is kAssignOnly, nodes_to_compile will be empty at this point due to logic in PlaceNode
  if (!nodes_to_compile.empty()) {
    std::vector<NodeComputeInfo> node_compute_funcs;

    if (export_dll) {
      ORT_ENFORCE(fusion_style == IExecutionProvider::FusionStyle::Function,
                  "Must use Function based fusion when exporting compiled nodes to dll.");
    }

    if (fusion_style == IExecutionProvider::FusionStyle::Function) {
      // Create a Function based node where the fused nodes have a new Graph instance.

      if (export_dll) {
        std::string dll_path;
        ORT_RETURN_IF_ERROR(current_ep.Compile(nodes_to_compile, dll_path));

        for (auto* node : nodes_to_compile) {
          ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), dll_path));
        }
      } else {
        ORT_RETURN_IF_ERROR(current_ep.Compile(nodes_to_compile, node_compute_funcs));

        if (node_compute_funcs.size() != nodes_to_compile.size()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
        }

        for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
          ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(nodes_to_compile[j]->Name(), std::move(node_compute_funcs[j])));
        }
      }

      for (auto* node : nodes_to_compile) {
        // add the KernelDef instances for the compiled nodes
        KernelDefBuilder builder;
        BuildFusedKernelDef(builder, *node);
        ORT_RETURN_IF_ERROR(fused_kernel_registry.Register(builder,
                                                           static_cast<KernelCreatePtrFn>(
                                                               [](const OpKernelInfo& info) -> OpKernel* {
                                                                 return new FunctionKernel(info);
                                                               })));
      }

    } else {
      // temporary storage for the GraphViewer for each IndexedSubGraph
      std::vector<std::unique_ptr<GraphViewer>> viewers;
      viewers.reserve(nodes_to_compile.size());
      std::vector<IExecutionProvider::FusedNodeAndGraph> nodes_and_viewers;

      for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
        auto* node = nodes_to_compile[j];
        const auto& cur_capability = *capabilities_to_compile[j];
        viewers.push_back(std::make_unique<GraphViewer>(graph, *cur_capability.sub_graph));
        nodes_and_viewers.push_back(IExecutionProvider::FusedNodeAndGraph{*node, *viewers.back()});
      }

      ORT_RETURN_IF_ERROR(current_ep.Compile(nodes_and_viewers, node_compute_funcs));

      if (node_compute_funcs.size() != nodes_to_compile.size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
      }

      for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
        auto* node = nodes_to_compile[j];

        ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), std::move(node_compute_funcs[j])));

        const auto& cur_capability = capabilities_to_compile[j];
        const IndexedSubGraph& indexed_sub_graph = *cur_capability->sub_graph;
        const IndexedSubGraph::MetaDef& metadef = *indexed_sub_graph.GetMetaDef();

        // create the func kernel for the name in the MetaDef. this is also the node name and that name that will
        // used as the key in the FuncManager entry. We need the registry to own the KernelCreateInfo that is
        // used by SessionState
        KernelDefBuilder builder;
        BuildFusedKernelDef(builder, metadef, type);
        ORT_RETURN_IF_ERROR(fused_kernel_registry.Register(builder,
                                                           static_cast<KernelCreatePtrFn>(
                                                               [](const OpKernelInfo& info) -> OpKernel* {
                                                                 return new FunctionKernel(info);
                                                               })));

        // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
        graph.FinalizeFuseSubGraph(indexed_sub_graph, *node);
      }
    }
  }

  // if this is the main graph call Resolve to put the Graph back into a guaranteed good state
  // TODO: Graph::FuseSubGraph and Graph::FinalizeFuseSubGraph should now create valid edges so this call to
  // Graph::Resolve should not be required. Need to test to validate that, especially if node being fused
  // was a control flow node with its own subgraph as more than just the edges may need updating.
  if (!graph.IsSubgraph()) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }

  // For some cases, like fp16 on cpu, right now we don't have any kernel support that.
  // But we will insert cast op to run the model, so skip the error checking here.
  // If after graph transform phase, the node still not assigned, we will report error
  // during kernel creation phase.
#ifdef COUNT_NON_CUDA_OPS
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != kCudaExecutionProvider &&
        node.Domain() != kMLDomain &&
        node.Domain() != kMSDomain)
      non_cuda.AddOp(node.OpType());
  }
#endif

  return Status::OK();
}

// expand any nodes that have an ONNX function definition but no matching ORT kernel
static Status InlineNodes(Graph& graph, bool& modified_graph) {
  // recurse into nested graphs first so we process from bottom up
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      ORT_RETURN_IF_ERROR(InlineNodes(*subgraph, modified_graph));
    }
  }

  // See if the node with no provider can be inlined. If one such nodes can be
  // successfully inlined, we re-run the partitioner on the modified graph.
  // NOTE: Inlining the function will change the nodes in the Graph instance, so we can't do that while iterating
  // using graph.Nodes().
  std::vector<Node*> nodes_to_inline;
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType().empty() && node.GetFunctionBody() != nullptr) {
      nodes_to_inline.push_back(&node);
    }
  }

  for (auto* node : nodes_to_inline) {
    ORT_RETURN_IF_ERROR(graph.InlineFunction(*node));
    modified_graph = true;
  }

  return Status::OK();
}

Status GraphPartitioner::PartitionOnnxFormatModel(Graph& graph, bool export_dll, FuncManager& func_mgr,
                                                  KernelRegistry& fused_kernel_registry, Mode mode,
                                                  int& fused_node_unique_id) const {
  bool modified_graph = false;

  do {
    // process full graph with each EP
    for (const auto& ep : providers_) {
      ORT_RETURN_IF_ERROR(PartitionOnnxFormatModelImpl(graph, export_dll, func_mgr, kernel_registry_mgr_,
                                                       fused_kernel_registry, *ep, mode, fused_node_unique_id));
    }

    // expand any nodes that have an ONNX function definition but no matching ORT kernel.
    modified_graph = false;
    ORT_RETURN_IF_ERROR(InlineNodes(graph, modified_graph));

    // Resolve and rerun graph partitioning and inlining if there was a change
    if (modified_graph) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }
  } while (modified_graph);

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

static Status PartitionOrtFormatModelImpl(Graph& graph, FuncManager& func_mgr,
                                          KernelRegistryManager& kernel_registry_mgr,
                                          KernelRegistry& fused_kernel_registry,
                                          IExecutionProvider& current_ep,
                                          std::unordered_map<std::string, uint64_t>& compiled_kernel_hashes,
                                          int& fused_node_unique_id) {
  // recurse into nested graphs first to partition bottom up.
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      ORT_RETURN_IF_ERROR(PartitionOrtFormatModelImpl(*subgraph, func_mgr, kernel_registry_mgr, fused_kernel_registry,
                                                      current_ep, compiled_kernel_hashes, fused_node_unique_id));
    }
  }

  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  const std::string& type = current_ep.Type();
  GraphViewer graph_viewer(graph);
  std::vector<IExecutionProvider::FusedNodeAndGraph> nodes_and_viewers;

  std::vector<std::unique_ptr<ComputeCapability>> capabilities =
      current_ep.GetCapability(graph_viewer, kernel_registry_mgr.GetKernelRegistriesByProviderType(type));

  if (capabilities.empty()) {
    return Status::OK();
  }

  // storage for the GraphViewer for each IndexedSubGraph
  std::vector<std::unique_ptr<GraphViewer>> viewers;
  viewers.reserve(capabilities.size());

  for (auto& capability : capabilities) {
    const IndexedSubGraph& indexed_sub_graph = *capability->sub_graph;
    const IndexedSubGraph::MetaDef* metadef = indexed_sub_graph.GetMetaDef();
    if (!metadef) {
      // Static kernel - use the kernel hash that was saved in the ORT format model
      continue;
    }

    std::ostringstream oss;
    oss << type << "_" << metadef->name << "_" << fused_node_unique_id++;
    std::string node_name = oss.str();

    Node& fused_node = graph.BeginFuseSubGraph(indexed_sub_graph, node_name);
    fused_node.SetExecutionProviderType(type);

    // create filtered graph viewer for this set of nodes
    //
    // TODO: Could avoid the topological sort in the GraphViewer ctor by constructing from an existing
    // GraphViewer instance instead of the Graph (copying the topological order instead of recalculating).
    viewers.push_back(std::make_unique<GraphViewer>(graph, indexed_sub_graph));
    nodes_and_viewers.push_back(IExecutionProvider::FusedNodeAndGraph{fused_node, *viewers.back()});
  }

  // We will compile the fused nodes one by one, and fuse the subgraph if successful.
  // If a compilation fails we undo the fusion and leave the original nodes available for other EPs to take
  for (size_t j = 0, end = nodes_and_viewers.size(); j < end; ++j) {
    Node& node = nodes_and_viewers[j].fused_node;
    std::vector<NodeComputeInfo> single_node_compute_func;
    auto status = current_ep.Compile({nodes_and_viewers[j]}, single_node_compute_func);
    if (!status.IsOK()) {
      // There is compile error with the nodes_and_viewer[j], remove the fused_node and function from the graph
      LOGS_DEFAULT(ERROR) << "EP: " << current_ep.Type() << " has Compile error: " << status.ErrorMessage();
      graph.CancelFuseSubGraph(node);
    } else {
      ORT_RETURN_IF(single_node_compute_func.empty(), "single_node_compute_func should have 1 elements");
      ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node.Name(), std::move(single_node_compute_func[0])));

      const auto& cur_capability = capabilities[j];
      const IndexedSubGraph& indexed_sub_graph = *cur_capability->sub_graph;
      const IndexedSubGraph::MetaDef& metadef = *indexed_sub_graph.GetMetaDef();

      KernelDefBuilder builder;
      BuildFusedKernelDef(builder, metadef, type);
      auto kernel_def = builder.Build();

      // save hash so SessionState can find the kernel. each kernel name should be unique
      if (compiled_kernel_hashes.insert({metadef.name, kernel_def->GetHash()}).second == false) {
        ORT_THROW("Existing entry in compiled kernel hashes for ", metadef.name,
                  ". Execution Provider must generate unique names across the entire model.");
      }

      ORT_RETURN_IF_ERROR(fused_kernel_registry.Register(
          KernelCreateInfo(std::move(kernel_def), static_cast<KernelCreatePtrFn>(
                                                      [](const OpKernelInfo& info) -> OpKernel* {
                                                        return new FunctionKernel(info);
                                                      }))));

      // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
      graph.FinalizeFuseSubGraph(indexed_sub_graph, node);
    }
  }

  return Status::OK();
}

// Simplified partitioning where custom EPs may produce compiled nodes.
// EPs with static kernels do not need to be processed as their kernels are matched via hash information serialized
// as part of the ORT format model.
Status GraphPartitioner::PartitionOrtFormatModel(
    Graph& graph, FuncManager& func_mgr,
    KernelRegistry& fused_kernel_registry,
    std::unordered_map<std::string, uint64_t>& compiled_kernel_hashes,
    int& fused_node_unique_id) const {
  // process full graph with each EP
  for (const auto& ep : providers_) {
    if (ep->Type() == kCpuExecutionProvider) {
      // hash for kernel is stored in session state for EPs that have pre-registered kernels
      // (vs. runtime fused kernels) so nothing to do here.
      continue;
    }

    ORT_RETURN_IF_ERROR(PartitionOrtFormatModelImpl(graph, func_mgr, kernel_registry_mgr_, fused_kernel_registry,
                                                    *ep, compiled_kernel_hashes, fused_node_unique_id));
  }

  return Status::OK();
}

Status GraphPartitioner::Partition(Graph& graph, bool export_dll, FuncManager& func_mgr, Mode mode,
                                   std::unordered_map<std::string, uint64_t>* compiled_kernel_hashes) const {
  // It is a greedy partitioning algorithm per provider preferences user provided when calling ONNX RUNTIME right now.
  // 1. Execution providers' capabilities are checked one by one.
  // 2. All sub-graphs that an execution provider returns will be assigned to it if it's not assigned yet.
  //    NOTE: A 'sub-graph' is a subset of nodes within the current Graph instance.
  //          The control flow nodes have nested Graph instance/s which are also called subgraphs,
  //          but are completely separate Graph instances and not a subset of nodes within a single Graph instance.
  // 3. CPU execution provider is expected to be able to run any node and is the last one in execution provider
  //    preference.
  if (providers_.Empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }

  // fused_kernel_registry is preparing the kernels created on the fly for fused sub graph.
  // It is only visible for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();

  // we make sure each fused node name is unique across the entire model for clarity
  int fused_node_unique_id = 0;

  if (mode == Mode::kNormal || mode == Mode::kAssignOnly) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_RETURN_IF_ERROR(PartitionOnnxFormatModel(graph, export_dll, func_mgr, *fused_kernel_registry, mode,
                                                 fused_node_unique_id));
#else
    ORT_UNUSED_PARAMETER(export_dll);
    ORT_THROW("Not supported in this build.");
#endif
  } else {
    ORT_ENFORCE(compiled_kernel_hashes != nullptr, "Compiled kernel hashes must be provided");

    ORT_RETURN_IF_ERROR(PartitionOrtFormatModel(graph, func_mgr, *fused_kernel_registry, *compiled_kernel_hashes,
                                                fused_node_unique_id));
  }

  if (!fused_kernel_registry->IsEmpty()) {
    kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry);
  }

  return Status::OK();
}
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
