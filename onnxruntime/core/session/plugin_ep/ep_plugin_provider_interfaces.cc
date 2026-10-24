// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"

#include <algorithm>
#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/plugin_data_transfer.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/framework/resource_accountant.h"
#include "core/common/inlined_containers.h"
#include <limits>

#include "core/graph/constants.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/model_editor_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/allocator_adapters.h"
#include "core/session/plugin_ep/ep_allocator_utils.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/session/plugin_ep/ep_event_profiling.h"
#include "core/session/ort_apis.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {

//
// PluginExecutionProviderFactory
//

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory},
      devices_{ep_devices.begin(), ep_devices.end()} {
  hardware_devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : ep_devices) {
    hardware_devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices,
                                                               gsl::span<const OrtHardwareDevice* const> hw_devices,
                                                               gsl::span<const OrtKeyValuePairs* const> ep_metadata)
    : ep_factory_{ep_factory},
      devices_{ep_devices.begin(), ep_devices.end()},
      hardware_devices_{hw_devices.begin(), hw_devices.end()},
      ep_metadata_{ep_metadata.begin(), ep_metadata.end()} {
}

std::unique_ptr<IExecutionProvider>
PluginExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                               const OrtLogger& session_logger) {
  std::unique_ptr<PluginExecutionProvider> plugin_ep;
  Status status = CreatePluginExecutionProvider(session_options, session_logger, plugin_ep);

  if (!status.IsOK()) {
    ORT_THROW("Error creating execution provider: ", status.ToString());
  }

  return plugin_ep;
}

// Do some basic checks on `ort_ep` to detect obvious issues early on.
static Status SanityCheckOrtEp(const OrtEp& ort_ep) {
  // Plugin EPs were first introduced in ORT 1.22, so we expect at least this API version.
  constexpr auto kMinAllowedOrtVersionSupported = 22;

  ORT_RETURN_IF_NOT(ort_ep.ort_version_supported >= kMinAllowedOrtVersionSupported,
                    "OrtEp has invalid ort_version_supported=", ort_ep.ort_version_supported,
                    " (expected at least ", kMinAllowedOrtVersionSupported, ").");

  ORT_RETURN_IF_NOT(ort_ep.GetName != nullptr, "OrtEp has null GetName function pointer.");

  ORT_RETURN_IF_NOT(ort_ep.GetName(&ort_ep) != nullptr, "OrtEp's GetName function returned null.");

  return Status::OK();
}

Status PluginExecutionProviderFactory::CreatePluginExecutionProvider(
    const OrtSessionOptions& session_options,
    const OrtLogger& logger,
    /*out*/ std::unique_ptr<PluginExecutionProvider>& plugin_ep) {
  plugin_ep = nullptr;
  OrtEp* ort_ep_raw = nullptr;

  ORT_RETURN_IF_ERROR(ToStatusAndRelease(ep_factory_.CreateEp(&ep_factory_, hardware_devices_.data(),
                                                              ep_metadata_.data(), hardware_devices_.size(),
                                                              &session_options, &logger, &ort_ep_raw)));
  ORT_RETURN_IF(ort_ep_raw == nullptr, "OrtEpFactory::CreateEp() for '", ep_factory_.GetName(&ep_factory_),
                "' returned a NULL OrtEp instance");

  auto ort_ep = UniqueOrtEp(ort_ep_raw, OrtEpDeleter(ep_factory_));

  ORT_RETURN_IF_ERROR(SanityCheckOrtEp(*ort_ep));

  std::shared_ptr<KernelRegistry> kernel_registry;
  ORT_RETURN_IF_ERROR(GetPluginEpKernelRegistry(*ort_ep, kernel_registry));

  plugin_ep = std::make_unique<PluginExecutionProvider>(std::move(ort_ep),
                                                        session_options, ep_factory_, devices_,
                                                        kernel_registry,
                                                        *logger.ToInternal());
  return Status::OK();
}

/// <summary>
/// Functor used to generate a Metadef name for a subgraph supported by a plugin EP.
/// The generated name is a concatenation of a prefix (i.e., the EP name) with
/// the model's hash and a unique ID.
/// </summary>
struct PluginEpMetaDefNameFunctor {
  explicit PluginEpMetaDefNameFunctor(const ModelMetadefIdGenerator& generator,
                                      const GraphViewer& graph_viewer,
                                      const std::string& prefix)
      : generator_(generator), graph_viewer_(graph_viewer), prefix_(prefix) {}

  std::string operator()() {
    uint64_t model_hash = 0;
    int id = generator_.GenerateId(graph_viewer_, model_hash);
    return MakeString(prefix_, "_", model_hash, "_", id);
  }

  const ModelMetadefIdGenerator& generator_;
  const GraphViewer& graph_viewer_;
  const std::string& prefix_;
};

//
// PluginExecutionProvider
//

static OrtDevice GetOrtDeviceForPluginEp(const OrtEp& ep, gsl::span<const OrtEpDevice* const> ep_devices) {
  // Resolve the EP's default device. If the EP implements GetDefaultMemoryDevice, use its
  // answer directly. Otherwise fall back to the first OrtEpDevice's default memory info.

  ORT_ENFORCE(!ep_devices.empty());  // Should not be possible to create an EP without OrtEpDevices.

  if (ep.ort_version_supported >= 27 && ep.GetDefaultMemoryDevice != nullptr) {
    const OrtMemoryDevice* memory_device = nullptr;
    Ort::ThrowOnError(ep.GetDefaultMemoryDevice(&ep, &memory_device));
    if (memory_device != nullptr) {
      return *static_cast<const OrtDevice*>(memory_device);
    }
  }

  // If there's no explicit default memory device, choose the first default memory info.
  const OrtMemoryInfo* device_memory_info = ep_devices[0]->device_memory_info;

  return device_memory_info != nullptr ? device_memory_info->device : OrtDevice();
}

static const Node* FindFirstNodeAssignedToOtherEP(const std::string& ep_type,
                                                  gsl::span<const EpNode* const> ep_nodes) {
  auto node_iter = std::find_if(ep_nodes.begin(), ep_nodes.end(),
                                [&ep_type](const EpNode* node) -> bool {
                                  const auto& node_ep_type = node->GetInternalNode().GetExecutionProviderType();
                                  return !node_ep_type.empty() && node_ep_type != ep_type;
                                });

  return node_iter != ep_nodes.end() ? &(*node_iter)->GetInternalNode() : nullptr;
}

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep, const OrtSessionOptions& session_options,
                                                 OrtEpFactory& ep_factory,
                                                 gsl::span<const OrtEpDevice* const> ep_devices,
                                                 std::shared_ptr<KernelRegistry> kernel_registry,
                                                 const logging::Logger& logger)
    : IExecutionProvider(ep->GetName(ep.get()), GetOrtDeviceForPluginEp(*ep, ep_devices),
                         std::vector<const OrtEpDevice*>(ep_devices.begin(), ep_devices.end()), logger),
      ort_ep_(std::move(ep)),
      ep_factory_(ep_factory),
      ep_devices_(ep_devices.begin(), ep_devices.end()),
      kernel_registry_(std::move(kernel_registry)) {
  generate_ep_ctx_model_ = session_options.value.GetEpContextGenerationOptions().enable;

  // Record if the app requested weightless mode. Validation is deferred to Compile().
  weightless_requested_ =
      session_options.value.config_options.GetConfigOrDefault(kOrtSessionOptionEpEnableWeightless, "0") != "0";

  // Extract EP-scoped session config entries.
  // Arena options go to session_arena_options_; the rest go to provider_options_.
  {
    const std::string ep_prefix = OrtSessionOptions::GetProviderOptionPrefix(ort_ep_->GetName(ort_ep_.get()));
    const std::string arena_prefix = ep_prefix + "arena.";
    const bool extract_arena = ep_factory_.CreateAllocator && !ort_ep_->CreateAllocator;

    for (const auto& [key, value] : session_options.value.config_options.GetConfigOptionsMap()) {
      if (key.compare(0, ep_prefix.size(), ep_prefix) != 0) {
        continue;
      }

      const std::string bare_key = key.substr(ep_prefix.size());
      if (key.compare(0, arena_prefix.size(), arena_prefix) == 0) {
        if (extract_arena) {
          if (!session_arena_options_) {
            session_arena_options_.emplace();
          }
          session_arena_options_->Add(bare_key.c_str(), value.c_str());
        }
        continue;
      }

      // Also map standard CUDA EP arena option names (e.g. "arena_extend_strategy") into session_arena_options_.
      if (extract_arena) {
        const char* mapped_arena_key = nullptr;
        if (bare_key == "arena_extend_strategy") {
          mapped_arena_key = "arena.extend_strategy";
        } else if (bare_key == "initial_chunk_size_bytes") {
          mapped_arena_key = "arena.initial_chunk_size_bytes";
        } else if (bare_key == "max_dead_bytes_per_chunk") {
          mapped_arena_key = "arena.max_dead_bytes_per_chunk";
        } else if (bare_key == "initial_growth_chunk_size_bytes") {
          mapped_arena_key = "arena.initial_growth_chunk_size_bytes";
        } else if (bare_key == "max_power_of_two_extend_bytes") {
          mapped_arena_key = "arena.max_power_of_two_extend_bytes";
        } else if (bare_key == "max_mem") {
          mapped_arena_key = "arena.max_mem";
        } else if (bare_key == "gpu_device_mem_limit") {
          mapped_arena_key = "gpu_device_mem_limit";
        }
        if (mapped_arena_key != nullptr) {
          if (!session_arena_options_) {
            session_arena_options_.emplace();
          }
          session_arena_options_->Add(mapped_arena_key, value.c_str());
        }
      }

      // Store the bare option name (strip the EP-specific prefix) for GetProviderOptions().
      provider_options_[bare_key] = value;
    }
  }

  for (const auto* ep_device : ep_devices_) {
    if (ep_device->device_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->device_memory_info);
    }

    if (ep_device->host_accessible_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->host_accessible_memory_info);
    }

    if (ep_device->read_only_device_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->read_only_device_memory_info);
    }
  }
}

PluginExecutionProvider::~PluginExecutionProvider() {
  if (ort_ep_ && !api_node_compute_infos_.empty() && ort_ep_->ReleaseNodeComputeInfos != nullptr) {
    ort_ep_->ReleaseNodeComputeInfos(ort_ep_.get(), api_node_compute_infos_.data(),
                                     api_node_compute_infos_.size());
  }
}

const logging::Logger& PluginExecutionProvider::GetEpLoggerOrDefault() const {
  return GetLogger() != nullptr ? *GetLogger() : logging::LoggingManager::DefaultLogger();
}

std::shared_ptr<KernelRegistry> PluginExecutionProvider::GetKernelRegistry() const {
  return kernel_registry_;
}

std::vector<std::unique_ptr<ComputeCapability>>
PluginExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       IResourceAccountant* resource_accountant) const {
  ORT_UNUSED_PARAMETER(graph_optimizer_registry);  // TODO: Add support

  const logging::Logger& logger = GetEpLoggerOrDefault();

  // Early exit if a previous GetCapability pass already signaled stop (e.g., budget exhausted).
  // The framework calls GetCapability multiple times (e.g., after layout transformation),
  // and each EP is responsible for checking the stop flag.
  if (resource_accountant != nullptr && resource_accountant->IsStopIssued()) {
    LOGS(logger, WARNING) << Type() << " returning due to stop already set";
    return {};
  }

  std::unique_ptr<EpGraph> ep_graph = nullptr;
  if (Status status = EpGraph::Create(graph_viewer, ep_graph, true); !status.IsOK()) {
    LOGS(logger, ERROR) << "Failed to create OrtGraph for " << Type() << ": " << status.ToString();
    return {};
  }

  OrtEpGraphSupportInfo api_graph_support_info(*ep_graph, kernel_lookup);
  api_graph_support_info.resource_accountant = resource_accountant;
  Status status = ToStatusAndRelease(ort_ep_->GetCapability(ort_ep_.get(), ep_graph->ToExternal(), &api_graph_support_info));

  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "OrtEp::GetCapability() for " << Type() << " failed with error: " << status.ToString();
    return {};
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result.reserve(api_graph_support_info.node_groupings.size());
  if (api_graph_support_info.node_groupings.empty()) {
    return {};
  }

  // Host-side resource budget enforcement.
  // The host computes costs and enforces the budget uniformly for all node grouping kinds.
  // Plugin EPs only propose supported nodes; the host decides which to accept.

  // If an accountant exists but has no explicit threshold (i.e., the session option didn't specify a memory limit),
  // ask the EP for the available device resource (e.g., free GPU memory) and use that as the threshold.
  // This mirrors the in-tree CUDA EP behavior that calls cudaMemGetInfo as a fallback.
  if (resource_accountant != nullptr && !resource_accountant->GetThreshold().has_value() &&
      ort_ep_->ort_version_supported >= 26 && ort_ep_->GetAvailableResource != nullptr) {
    OrtResourceCount available{};
    if (auto* ort_status = ort_ep_->GetAvailableResource(ort_ep_.get(), &available); ort_status == nullptr) {
      if (available.kind == OrtResourceCountKind_TotalBytes) {
        const auto bytes = available.value.total_bytes;
        // Clamp to size_t max on 32-bit builds instead of throwing via narrow<>.
        const size_t clamped = (bytes > std::numeric_limits<size_t>::max())
                                   ? std::numeric_limits<size_t>::max()
                                   : static_cast<size_t>(bytes);
        resource_accountant->SetThreshold(ResourceCount{clamped});
        LOGS(logger, VERBOSE) << Type() << " set resource threshold from device: "
                              << clamped << " bytes"
                              << (clamped != bytes ? " (clamped from " + std::to_string(bytes) + " bytes)" : "");
      } else if (available.kind != OrtResourceCountKind_None) {
        LOGS(logger, WARNING) << Type() << " GetAvailableResource returned unsupported kind: "
                              << static_cast<int>(available.kind);
      }
    } else {
      // Log warning and continue without a device-derived threshold
      auto status_releaser = Status(ToStatusAndRelease(ort_status));
      LOGS(logger, WARNING) << Type() << " GetAvailableResource failed: " << status_releaser.ToString();
    }
  }

  const bool has_budget = resource_accountant != nullptr && resource_accountant->GetThreshold().has_value();
  ResourceCount consumed = resource_accountant != nullptr
                               ? resource_accountant->GetConsumedAmount()
                               : ResourceCount{};
  ResourceCount budget = has_budget ? *resource_accountant->GetThreshold() : ResourceCount{};

  // Create ComputeCapability instances from OrtEpGraphSupportInfo::NodeGrouping instances.
  for (const OrtEpGraphSupportInfo::NodeGrouping& node_grouping : api_graph_support_info.node_groupings) {
    // Skip this node grouping if any node has already been assigned to another EP.
    if (const Node* node_for_other_ep = FindFirstNodeAssignedToOtherEP(Type(), node_grouping.nodes);
        node_for_other_ep != nullptr) {
      LOGS(logger, WARNING) << "OrtEp::GetCapability() specified nodes that cannot be assigned to " << Type() << ". "
                            << "Found one or more nodes that were already assigned to a different EP named '"
                            << node_for_other_ep->GetExecutionProviderType() << "'. Ex: "
                            << node_for_other_ep->OpType() << " node with name '"
                            << node_for_other_ep->Name() << "'.";
      continue;
    }

    // Plugin accounting cannot receive Level-1 estimates yet. Reject the whole
    // group rather than use generic or profiled costs for an unproven GQA bound.
    if (resource_accountant != nullptr && Type() == kCudaExecutionProvider) {
      const auto unproven_gqa = std::find_if(
          node_grouping.nodes.begin(), node_grouping.nodes.end(), [](const EpNode* ep_node) {
            const Node& node = ep_node->GetInternalNode();
            return node.GetExecutionProviderType().empty() &&
                   node.OpType() == "GroupQueryAttention" && node.Domain() == kMSDomain;
          });
      if (unproven_gqa != node_grouping.nodes.end()) {
        LOGS(logger, WARNING)
            << Type() << " declining capacity-aware assignment for GroupQueryAttention node "
            << (*unproven_gqa)->GetInternalNode().Name()
            << " because the plugin accounting path has no proven route-aware workspace estimate.";
        continue;
      }
    }

    if (node_grouping.kind == OrtEpGraphSupportInfo::NodeGroupingKind::kSingleAssignedNode) {
      if (node_grouping.nodes.size() != 1) {
        LOGS(logger, ERROR) << "OrtEp::GetCapability() for " << Type() << " did not specify exactly one valid node "
                            << "when calling EpGraphSupportInfo_AddSingleNode().";
        return {};
      }

      const Node& internal_node = node_grouping.nodes[0]->GetInternalNode();
      const NodeIndex node_index = internal_node.Index();

      // Node already assigned from a previous pass (e.g., before layout transformation
      // or after function inlining). Its cost was already committed — skip re-computation to avoid
      // double-charging the output-size component.
      // FindFirstNodeAssignedToOtherEP already filtered out nodes assigned to a different EP,
      // so a non-empty EP type here means it was assigned to this EP.
      const bool previously_assigned = !internal_node.GetExecutionProviderType().empty();

      auto indexed_sub_graph = std::make_unique<IndexedSubGraph>();
      indexed_sub_graph->nodes.push_back(node_index);

      // Host-side budget enforcement for single nodes.
      if (resource_accountant != nullptr && !previously_assigned) {
        ResourceCount cost = resource_accountant->ComputeResourceCount(internal_node);
        ResourceCount would_be_consumed = AddResourceCounts(consumed, cost);

        LOGS(logger, VERBOSE) << Type() << " node: " << internal_node.Name()
                              << " (" << internal_node.OpType() << ")"
                              << " cost: " << FormatResourceCount(cost)
                              << " would_be_consumed: " << FormatResourceCount(would_be_consumed)
                              << " budget: " << FormatResourceCount(budget);

        if (has_budget && ResourceCountExceeds(would_be_consumed, budget)) {
          LOGS(logger, WARNING) << Type() << " halting assignment due to budget at node: "
                                << internal_node.Name();
          resource_accountant->SetStopAssignment();
          break;  // stop processing further groupings
        }

        consumed = would_be_consumed;
        indexed_sub_graph->SetAccountant(resource_accountant);
        indexed_sub_graph->AppendNodeCost(cost);
      }

      result.push_back(std::make_unique<ComputeCapability>(std::move(indexed_sub_graph)));
    } else if (node_grouping.kind == OrtEpGraphSupportInfo::NodeGroupingKind::kFusedNode) {
      if (node_grouping.nodes.empty()) {
        LOGS(logger, ERROR) << "OrtEp::GetCapability() for " << Type() << " set an empty array of nodes "
                            << "when specifying supported nodes.";
        return {};
      }

      std::unordered_set<const Node*> node_set;
      node_set.reserve(node_grouping.nodes.size());

      for (const EpNode* ep_node : node_grouping.nodes) {
        node_set.insert(&ep_node->GetInternalNode());
      }

      // We now require the OrtEp to only provide individual groups of supported nodes that each maps to exactly
      // one ComputeCapability. Calling utils::CreateSupportedPartitions() may create multiple ComputeCapability
      // instances, and if so, log an error and return.
      //
      // TODO(adrianlizarraga): Do not use the heavy-weight CreateSupportedPartitions just to check if the user
      // provided a single partition. Use utils::MakeCapability() and create a new helper to check that there are no
      // unsupported nodes in any path between supported nodes.
      auto metadef_gen_functor = PluginEpMetaDefNameFunctor(metadef_id_generator_, graph_viewer, this->Type());
      std::vector<std::unique_ptr<ComputeCapability>> capabilities = utils::CreateSupportedPartitions(
          graph_viewer, node_set, /*stop_ops*/ {}, std::move(metadef_gen_functor),
          this->Type(), this->Type(), /*node_unit_map*/ nullptr,
          node_grouping.fusion_options.drop_constant_initializers);

      if (capabilities.size() != 1) {
        LOGS(logger, ERROR) << "OrtEp::GetCapability() for " << Type() << " set nodes that cannot be fused together. "
                            << "Please ensure that the nodes provided to EpGraphSupportInfo_AddNodesToFuse() do not "
                            << "have an unsupported node in any path between two of the supported nodes.";
        return {};
      }

      // Log an error if the nodes in node_set do not match the nodes in capabilities[0].
      // TODO(adrianlizarraga): This check can be removed when we stop using utils::CreateSupportedPartitions() above.
      std::vector<NodeIndex>& capability_node_indices = capabilities[0]->sub_graph->nodes;
      InlinedHashSet<NodeIndex> capability_node_indices_set(capability_node_indices.begin(),
                                                            capability_node_indices.end());

      if (node_set.size() != capability_node_indices_set.size()) {
        LOGS(logger, ERROR) << "OrtEp::GetCapability() for " << Type()
                            << " set nodes that cannot all be fused together.";
        return {};
      }

      // Host-side budget enforcement for fused capabilities.
      // Compute per-component-node costs from the accountant and check total against budget.
      // Skip cost computation for nodes already assigned to this EP from a previous pass
      // to avoid double-charging the output-size component.
      if (resource_accountant != nullptr) {
        auto* fused_sub_graph = capabilities[0]->sub_graph.get();
        fused_sub_graph->SetAccountant(resource_accountant);

        ResourceCount group_cost{};
        InlinedVector<ResourceCount> node_costs;
        node_costs.reserve(fused_sub_graph->nodes.size());

        for (NodeIndex idx : fused_sub_graph->nodes) {
          const Node* node = graph_viewer.GetNode(idx);
          const bool node_already_assigned =
              node != nullptr && !node->GetExecutionProviderType().empty();
          ResourceCount cost = (node != nullptr && !node_already_assigned)
                                   ? resource_accountant->ComputeResourceCount(*node)
                                   : ResourceCount{};
          group_cost = AddResourceCounts(group_cost, cost);
          node_costs.push_back(cost);
        }

        ResourceCount would_be_consumed = AddResourceCounts(consumed, group_cost);

        if (has_budget) {
          LOGS(logger, VERBOSE) << Type() << " fused group cost: " << FormatResourceCount(group_cost)
                                << " would_be_consumed: " << FormatResourceCount(would_be_consumed)
                                << " budget: " << FormatResourceCount(budget);

          if (ResourceCountExceeds(would_be_consumed, budget)) {
            LOGS(logger, WARNING) << Type() << " halting assignment: fused group exceeds budget.";
            resource_accountant->SetStopAssignment();
            break;  // stop processing further groupings
          }
        }

        consumed = would_be_consumed;

        for (const auto& cost : node_costs) {
          fused_sub_graph->AppendNodeCost(cost);
        }
      }

      result.push_back(std::move(capabilities[0]));
    } else {
      LOGS(logger, ERROR) << "PluginExecutionProvider::GetCapability() has invalid NodeGroupingKind: "
                          << static_cast<int>(node_grouping.kind);
      return {};
    }
  }

  return result;
}

// Out-of-line destructor: EpNode and EpValueInfo must be complete types
// when unique_ptr members are destroyed (required by libc++).
PluginExecutionProvider::FusedNodeState::~FusedNodeState() = default;

Status PluginExecutionProvider::FusedNodeState::AddFusedNode(const Node& fused_node, /*out*/ EpNode*& added_ep_node) {
  std::unique_ptr<EpNode> unique_ep_fused_node = nullptr;
  ORT_RETURN_IF_ERROR(EpNode::Create(fused_node, /*parent graph*/ nullptr, this->value_infos, unique_ep_fused_node));
  this->nodes.push_back(std::move(unique_ep_fused_node));
  added_ep_node = this->nodes.back().get();
  return Status::OK();
}

/// <summary>
/// Converts the EPContext nodes provided by the plugin EP (OrtNode instances) to onnxruntime::Node instances.
/// Note that the EP plugin uses the model editor API to create the OrtNode instances.
/// </summary>
/// <param name="ep_name">Name of the plugin EP.</param>
/// <param name="fused_nodes">fused nodes provided by ORT.</param>
/// <param name="plugin_ep_context_nodes">EPContext nodes provided by the plugin EP.</param>
/// <param name="result_nodes">Output parameter set to the resulting array of EPContext nodes.</param>
/// <param name="result_node_args">Output parameter that stores the NodeArgs used by the EPContext nodes.</param>
/// <returns>A status indicating success or an error.</returns>
static Status ConvertEpContextNodes(const std::string& ep_name, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes, const std::vector<OrtNode*> plugin_ep_context_nodes,
                                    /*out*/ std::vector<std::unique_ptr<Node>>& result_nodes,
                                    /*out*/ std::vector<std::unique_ptr<NodeArg>>& result_node_args) {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (plugin_ep_context_nodes.empty()) {
    return Status::OK();  // No EPContext nodes.
  }

  std::vector<std::unique_ptr<Node>> ep_context_nodes_holder;
  std::vector<std::unique_ptr<NodeArg>> ep_context_node_args_holder;

  ep_context_nodes_holder.reserve(plugin_ep_context_nodes.size());
  int index = -1;
  for (const OrtNode* ort_node : plugin_ep_context_nodes) {
    ++index;
    auto& fused_node_filtered_graph = fused_nodes[index].filtered_graph;
    ORT_RETURN_IF_NOT(ort_node != nullptr, ep_name, ": OrtEp::Compile() returned a NULL EPContext node.");

    const ModelEditorNode* editor_node = ModelEditorNode::ToInternal(ort_node);
    ORT_RETURN_IF_NOT(editor_node != nullptr, ep_name, ": OrtEp::Compile() returned OrtNode objects ",
                      "that were not created with OrtModelEditorApi.");

    // Create NodeArg for each input/output.
    std::vector<NodeArg*> input_node_args;
    std::vector<NodeArg*> output_node_args;

    input_node_args.reserve(editor_node->input_names.size());
    output_node_args.reserve(editor_node->output_names.size());

    for (const std::string& input_name : editor_node->input_names) {
      auto node_arg_on_fused_graph = fused_node_filtered_graph.get().GetNodeArg(input_name);
      const ONNX_NAMESPACE::TypeProto* p_arg_type = node_arg_on_fused_graph ? node_arg_on_fused_graph->TypeAsProto() : nullptr;
      auto node_arg = std::make_unique<NodeArg>(input_name, p_arg_type);  // Graph.Resolve() cannot set type because EP Context OP does not have proper shape inference function available.
      input_node_args.push_back(node_arg.get());
      ep_context_node_args_holder.push_back(std::move(node_arg));
    }

    for (const std::string& output_name : editor_node->output_names) {
      auto node_arg_on_fused_graph = fused_node_filtered_graph.get().GetNodeArg(output_name);
      const ONNX_NAMESPACE::TypeProto* p_arg_type = node_arg_on_fused_graph ? node_arg_on_fused_graph->TypeAsProto() : nullptr;
      auto node_arg = std::make_unique<NodeArg>(output_name, p_arg_type);  // Graph.Resolve() cannot set type because EP Context OP does not have proper shape inference function available.
      output_node_args.push_back(node_arg.get());
      ep_context_node_args_holder.push_back(std::move(node_arg));
    }

    // Create a name -> attribute map.
    NodeAttributes attributes;
    attributes.reserve(editor_node->attributes.size());

    for (const ONNX_NAMESPACE::AttributeProto& attr : editor_node->attributes) {
      attributes.emplace(attr.name(), attr);
    }

    // Create Node
    auto internal_node = std::make_unique<Node>(editor_node->node_name,
                                                editor_node->operator_name,
                                                "EPContext node for " + ep_name,
                                                input_node_args,
                                                output_node_args,
                                                &attributes,
                                                editor_node->domain_name);

    ep_context_nodes_holder.push_back(std::move(internal_node));
  }

  result_nodes = std::move(ep_context_nodes_holder);
  result_node_args = std::move(ep_context_node_args_holder);

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(ep_name);
  ORT_UNUSED_PARAMETER(plugin_ep_context_nodes);
  ORT_UNUSED_PARAMETER(result_nodes);
  ORT_UNUSED_PARAMETER(result_node_args);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Creating EPContext models is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
}

Status PluginExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                        std::vector<NodeComputeInfo>& node_compute_infos) {
  ORT_RETURN_IF(ort_ep_->Compile == nullptr, "OrtEp for ", Type(), " did not provide a valid Compile() function");
  ORT_RETURN_IF(ort_ep_->ReleaseNodeComputeInfos == nullptr, "OrtEp for ", Type(),
                " did not provide a valid ReleaseNodeComputeInfos() function");

  // Validate EP weightless support if the app requested it.
  if (weightless_requested_) {
    if (ort_ep_->ort_version_supported >= 29) {
      if (ort_ep_->GetWeightlessSupport == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Weightless mode requested (ep.enable_weightless=1) but EP '", Type(),
                               "' does not implement GetWeightlessSupport.");
      }

      OrtWeightlessSupport support = OrtWeightlessSupport_NONE;
      auto* ort_status = ort_ep_->GetWeightlessSupport(ort_ep_.get(), &support);
      if (ort_status != nullptr) {
        return ToStatusAndRelease(ort_status);
      }

      if (support == OrtWeightlessSupport_NONE) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "Weightless mode requested (ep.enable_weightless=1) but EP '", Type(),
                               "' does not support weightless mode on this device.");
      }
    } else {
      LOGS(GetEpLoggerOrDefault(), INFO) << "Weightless mode requested (ep.enable_weightless=1) but EP '"
                                         << Type() << "' was compiled with API version "
                                         << ort_ep_->ort_version_supported
                                         << " which predates GetWeightlessSupport (version 29). "
                                         << "ORT cannot verify EP weightless support. "
                                         << "The EP may still handle weightless via its own provider options.";
    }
  }

  const logging::Logger& logger = GetEpLoggerOrDefault();
  const size_t num_graphs = fused_nodes_and_graphs.size();
  std::vector<std::unique_ptr<EpGraph>> api_graphs_holder;
  std::vector<const OrtGraph*> api_graphs;
  std::vector<OrtNodeComputeInfo*> api_node_compute_infos(num_graphs, nullptr);
  std::vector<const OrtNode*> api_fused_nodes;

  // Push a new FusedNodeState to store the EpNode instances that we'll create to wrap the original fused nodes.
  // Fused nodes must be valid throughout model inference because they may be cached in NodeComputeInfo instances.
  fused_node_states_.push_back(FusedNodeState());
  FusedNodeState& fused_node_state = fused_node_states_.back();

  fused_node_state.nodes.reserve(num_graphs);
  api_graphs_holder.reserve(num_graphs);
  api_graphs.reserve(num_graphs);
  api_fused_nodes.reserve(num_graphs);
  api_node_compute_infos_.reserve(api_node_compute_infos_.size() + num_graphs);

  // Wrap GraphViewers into OrtGraphs and fused Nodes into OrtNodes.
  for (const FusedNodeAndGraph& node_and_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = node_and_graph.filtered_graph;
    const Node& fused_node = node_and_graph.fused_node;

    std::unique_ptr<EpGraph> ep_graph = nullptr;
    ORT_RETURN_IF_ERROR(EpGraph::Create(graph_viewer, ep_graph));
    api_graphs.push_back(ep_graph->ToExternal());
    api_graphs_holder.push_back(std::move(ep_graph));

    EpNode* ep_fused_node = nullptr;
    ORT_RETURN_IF_ERROR(fused_node_state.AddFusedNode(fused_node, ep_fused_node));
    api_fused_nodes.push_back(ep_fused_node->ToExternal());
  }

  // Provide an output buffer for the plugin EP to store EPContext nodes if it needs to (i.e., enabled in session options).
  std::vector<std::unique_ptr<OrtNode, decltype(&OrtApis::ReleaseNode)>> plugin_ep_context_nodes_holder;
  std::vector<OrtNode*> plugin_ep_context_nodes;
  plugin_ep_context_nodes_holder.reserve(num_graphs);
  plugin_ep_context_nodes.resize(num_graphs, nullptr);

  Status compile_status = ToStatusAndRelease(ort_ep_->Compile(ort_ep_.get(), api_graphs.data(), api_fused_nodes.data(),
                                                              num_graphs, api_node_compute_infos.data(),
                                                              plugin_ep_context_nodes.data()));

  // Store any EPContext nodes provided by the plugin EP in std::unique_ptr so that they are always properly released.
  for (OrtNode* ort_node : plugin_ep_context_nodes) {
    auto unique_ort_node = std::unique_ptr<OrtNode, decltype(&OrtApis::ReleaseNode)>(ort_node, OrtApis::ReleaseNode);
    plugin_ep_context_nodes_holder.push_back(std::move(unique_ort_node));
  }

  // Save OrtNodeComputeInfo created by OrtEp instance. They're freed when this IExecutionProvider
  // is destroyed.
  for (size_t i = 0; i < num_graphs; i++) {
    if (api_node_compute_infos[i] != nullptr) {
      api_node_compute_infos_.push_back(api_node_compute_infos[i]);
    }
  }

  ORT_RETURN_IF_ERROR(compile_status);

  // Initialize node_compute_infos as wrappers to api_node_compute_infos.
  for (size_t i = 0; i < num_graphs; i++) {
    OrtNodeComputeInfo* api_node_compute_info = api_node_compute_infos[i];
    ORT_RETURN_IF(api_node_compute_info == nullptr, "OrtEp::Compile() did not set a valid OrtNodeComputeInfo ",
                  "instance for graph at index ", i);

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [api_node_compute_info, &logger](ComputeContext* context,
                                                                      FunctionState* compute_state) -> int {
      Status status = ToStatusAndRelease(
          api_node_compute_info->CreateState(api_node_compute_info,
                                             reinterpret_cast<OrtNodeComputeContext*>(context),
                                             compute_state));
      const bool success = status.IsOK();
      if (!success) {
        LOGS(logger, ERROR) << "OrtNodeComputeInfo::CreateState() failed with error: "
                            << status.ErrorMessage();
      }

      return success ? 0 : 1;
    };

    compute_info.release_state_func = [api_node_compute_info](FunctionState compute_state) -> void {
      api_node_compute_info->ReleaseState(api_node_compute_info, compute_state);
    };

    compute_info.compute_func = [api_node_compute_info](FunctionState compute_state,
                                                        const OrtApi* /*c_api*/,
                                                        OrtKernelContext* kernel_context) -> Status {
      ORT_RETURN_IF_ERROR(ToStatusAndRelease((api_node_compute_info->Compute(api_node_compute_info, compute_state,
                                                                             kernel_context))));
      return Status::OK();
    };

    node_compute_infos.push_back(std::move(compute_info));
  }

  // Convert the EPContext nodes provided by the plugin EP into onnxruntime::Node instances.
  // We store the converted Node and NodeArg instances as members to ensure they can be returned to the ORT graph
  // partitioner via a call to IExecutionProvider::GetEpContextNodes().
  if (generate_ep_ctx_model_) {
    ORT_RETURN_IF_ERROR(ConvertEpContextNodes(Type(), fused_nodes_and_graphs, plugin_ep_context_nodes,
                                              /*out*/ ep_context_nodes_, /*out*/ ep_context_node_args_));
  }

  return Status::OK();
}

const InlinedVector<const Node*> PluginExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> result;

  for (const std::unique_ptr<Node>& node : ep_context_nodes_) {
    result.push_back(node.get());
  }

  return result;
}

Status PluginExecutionProvider::GetEpContextDataCallbackSupport(uint32_t& supported_flags) const {
  supported_flags = OrtEpContextDataCallbackSupportFlags_NONE;
  if (ort_ep_->ort_version_supported < 31 || ort_ep_->GetEpContextDataCallbackSupport == nullptr) {
    return Status::OK();
  }

  OrtStatus* ort_status = ort_ep_->GetEpContextDataCallbackSupport(ort_ep_.get(), &supported_flags);
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(ort_status));

  constexpr uint32_t known_flags = OrtEpContextDataCallbackSupportFlags_READ | OrtEpContextDataCallbackSupportFlags_WRITE;
  ORT_RETURN_IF(supported_flags & ~known_flags, "OrtEp for ", Type(),
                " returned unknown EPContext data support flags: ", supported_flags, ".");
  return Status::OK();
}

namespace {

struct DataLayoutMapping {
  DataLayout data_layout;
  OrtEpDataLayout api_data_layout;
};

// Maps enum values between `onnxruntime::DataLayout` and `OrtEpDataLayout`.
constexpr std::array kDataLayoutMappings{
    DataLayoutMapping{DataLayout::NCHW, OrtEpDataLayout::OrtEpDataLayout_NCHW},
    DataLayoutMapping{DataLayout::NHWC, OrtEpDataLayout::OrtEpDataLayout_NHWC},
};

}  // namespace

DataLayout PluginExecutionProvider::GetPreferredLayout() const {
  if (ort_ep_->GetPreferredDataLayout == nullptr) {
    return Base::GetPreferredLayout();
  }

  OrtEpDataLayout api_data_layout{};

  ORT_THROW_IF_ERROR(ToStatusAndRelease(ort_ep_->GetPreferredDataLayout(ort_ep_.get(), &api_data_layout)));

  const auto data_layout_mapping = std::find_if(kDataLayoutMappings.begin(), kDataLayoutMappings.end(),
                                                [api_data_layout](const DataLayoutMapping& mapping) {
                                                  return mapping.api_data_layout == api_data_layout;
                                                });

  ORT_ENFORCE(data_layout_mapping != kDataLayoutMappings.end(),
              "OrtEp::GetPreferredDataLayout() returned an invalid data layout: ", static_cast<int>(api_data_layout));

  return data_layout_mapping->data_layout;
}

std::optional<bool> PluginExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                                          std::string_view node_op_type,
                                                                          DataLayout target_data_layout) const {
  if (ort_ep_->ShouldConvertDataLayoutForOp == nullptr) {
    return Base::ShouldConvertDataLayoutForOp(node_domain, node_op_type, target_data_layout);
  }

  const auto data_layout_mapping = std::find_if(kDataLayoutMappings.begin(), kDataLayoutMappings.end(),
                                                [target_data_layout](const DataLayoutMapping& mapping) {
                                                  return mapping.data_layout == target_data_layout;
                                                });

  ORT_ENFORCE(data_layout_mapping != kDataLayoutMappings.end(),
              "Unable to map target_data_layout (", static_cast<int>(target_data_layout), ") to OrtEpDataLayout.");

  // Ensure domain and op type strings are null-terminated.
  const std::string node_domain_str{node_domain}, node_op_type_str{node_op_type};
  int should_convert = -1;

  ORT_THROW_IF_ERROR(ToStatusAndRelease(
      ort_ep_->ShouldConvertDataLayoutForOp(ort_ep_.get(),
                                            node_domain_str.c_str(), node_op_type_str.c_str(),
                                            data_layout_mapping->api_data_layout,
                                            &should_convert)));

  if (should_convert > 0) {
    return true;
  } else if (should_convert == 0) {
    return false;
  } else {
    return std::nullopt;
  }
}

bool PluginExecutionProvider::ConcurrentRunSupported() const {
  if (ort_ep_->ort_version_supported < 24 || ort_ep_->IsConcurrentRunSupported == nullptr) {
    return true;
  }

  bool is_supported = false;
  ORT_THROW_IF_ERROR(ToStatusAndRelease(ort_ep_->IsConcurrentRunSupported(ort_ep_.get(), &is_supported)));

  return is_supported;
}

Status PluginExecutionProvider::OnRunStart(const RunOptions& run_options) {
  if (ort_ep_->OnRunStart == nullptr) {
    return Base::OnRunStart(run_options);
  }

  return ToStatusAndRelease(ort_ep_->OnRunStart(ort_ep_.get(), &run_options));
}

Status PluginExecutionProvider::OnRunEnd(bool sync_stream, const RunOptions& run_options) {
  if (ort_ep_->OnRunEnd == nullptr) {
    return Base::OnRunEnd(sync_stream, run_options);
  }

  return ToStatusAndRelease(ort_ep_->OnRunEnd(ort_ep_.get(), &run_options, sync_stream));
}

Status PluginExecutionProvider::OnSessionInitializationEnd() {
  if (ort_ep_->ort_version_supported < 27 || ort_ep_->OnSessionInitializationEnd == nullptr) {
    return Base::OnSessionInitializationEnd();
  }
  return ToStatusAndRelease(ort_ep_->OnSessionInitializationEnd(ort_ep_.get()));
}

Status PluginExecutionProvider::Sync() const {
  if (ort_ep_->ort_version_supported < 25 || ort_ep_->Sync == nullptr) {
    return Base::Sync();
  }
  return ToStatusAndRelease(ort_ep_->Sync(ort_ep_.get()));
}

Status PluginExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                    gsl::span<const char* const> values) {
  if (ort_ep_->SetDynamicOptions == nullptr) {
    return Base::SetEpDynamicOptions(keys, values);
  }

  ORT_RETURN_IF_NOT(keys.size() == values.size(),
                    "The number of keys (", keys.size(), ") and number of values (", values.size(),
                    ") must be the same.");

  return ToStatusAndRelease(ort_ep_->SetDynamicOptions(ort_ep_.get(), keys.data(), values.data(), keys.size()));
}
std::unique_ptr<onnxruntime::IDataTransfer> PluginExecutionProvider::GetDataTransfer() const {
  OrtDataTransferImpl* data_transfer_impl = nullptr;

  if (ep_factory_.CreateDataTransfer != nullptr) {
    OrtStatus* status = ep_factory_.CreateDataTransfer(&ep_factory_, &data_transfer_impl);
    plugin_ep::OrtDataTransferImplUniquePtr owned_impl{data_transfer_impl};
    if (status != nullptr) {
      ORT_THROW("Error creating data transfer: ", ToStatusAndRelease(status).ToString());
    }

    if (owned_impl != nullptr) {
      return std::make_unique<plugin_ep::DataTransfer>(std::move(owned_impl));
    }
  }

  return {};
}

std::vector<AllocatorPtr> PluginExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;
  allocators.reserve(allocator_mem_infos_.size());

  // Identify the OrtEp the allocators are created for so a factory shared by several sessions can bind each allocator
  // to the right session state instead of guessing from device-wide queues.
  OrtKeyValuePairs allocator_options_storage;
  if (session_arena_options_) {
    allocator_options_storage = *session_arena_options_;
  }
  allocator_options_storage.Add("ep_instance_id",
                                std::to_string(reinterpret_cast<uintptr_t>(ort_ep_.get())).c_str());
  const OrtKeyValuePairs* allocator_options = &allocator_options_storage;

  auto* ep_factory = &ep_factory_;

  for (const auto* memory_info : allocator_mem_infos_) {
    AllocatorPtr alloc_ptr;
    ORT_THROW_IF_ERROR(ep_allocator_utils::CreateAndWrapEpAllocator(ort_ep_.get(), *ep_factory, *memory_info,
                                                                    allocator_options, alloc_ptr));
    if (alloc_ptr != nullptr) {
      allocators.push_back(std::move(alloc_ptr));
    }
  }

  return allocators;
}

void PluginExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& registry,
                                                     AllocatorMap& /*allocators*/) const {
  if (ep_factory_.IsStreamAware == nullptr || !ep_factory_.IsStreamAware(&ep_factory_)) {
    return;
  }

  for (const auto* mem_info : allocator_mem_infos_) {
    if (mem_info->device.UsesCpuMemory()) {
      // CPU memory does not need a stream
      continue;
    }

    if (!ort_ep_->CreateSyncStreamForDevice && !ep_factory_.CreateSyncStreamForDevice) {
      ORT_THROW("The OrtEpFactory is stream aware, but did not provide CreateSyncStreamForDevice.");
    }

    auto device_type = mem_info->device.Type();

    registry.RegisterCreateStreamFn(
        device_type,
        [this](const OrtDevice& device) {
          OrtSyncStreamImpl* stream = nullptr;
          const OrtMemoryDevice* memory_device = static_cast<const OrtMemoryDevice*>(&device);

          // prefer OrtEp function if available, otherwise fall back to using the OrtEpFactory implementation.
          OrtStatus* status = ort_ep_->CreateSyncStreamForDevice
                                  ? ort_ep_->CreateSyncStreamForDevice(ort_ep_.get(), memory_device, &stream)
                                  : ep_factory_.CreateSyncStreamForDevice(&ep_factory_, memory_device,
                                                                          /*stream_options*/ nullptr, &stream);

          ORT_ENFORCE(status == nullptr && stream != nullptr,
                      "Error creating sync stream for device: ", ToStatusAndRelease(status).ToString());
          return std::make_unique<plugin_ep::Stream>(device, *stream, *GetLogger()->ToExternal());
        });

    registry.RegisterWaitFn(device_type, device_type, plugin_ep::Notification::WaitNotificationOnDevice);
    registry.RegisterWaitFn(device_type, OrtDevice::CPU, plugin_ep::Notification::WaitNotificationOnHost);
  }
}

std::string PluginExecutionProvider::GetCompiledModelCompatibilityInfo(const onnxruntime::GraphViewer& graph_viewer) const {
  if (ort_ep_->GetCompiledModelCompatibilityInfo == nullptr) {
    // Plugin EP did not provide an implementation of this function, so we call a default implementation.
    return Base::GetCompiledModelCompatibilityInfo(graph_viewer);
  }
  std::unique_ptr<EpGraph> ep_graph = nullptr;
  auto ort_status = EpGraph::Create(graph_viewer, ep_graph);
  if (!ort_status.IsOK()) {
    LOGS(GetEpLoggerOrDefault(), ERROR) << "Failed to create EpGraph: " << ort_status.ToString();
    return {};
  }
  // Call EP plugin's OrtEp::GenerateCompiledModelCompatibilityInfo() function.
  std::string compatibility_info_string;
  compatibility_info_string = ort_ep_->GetCompiledModelCompatibilityInfo(ort_ep_.get(), ep_graph.get());
  return compatibility_info_string;
}

Status PluginExecutionProvider::ValidateCompiledModelCompatibilityInfo(const std::string& compatibility_info,
                                                                       OrtCompiledModelCompatibility& model_compatibility) const {
  if (ep_factory_.ValidateCompiledModelCompatibilityInfo == nullptr) {
    // Plugin EP did not provide an implementation of this function, so we call a default implementation.
    return Base::ValidateCompiledModelCompatibilityInfo(compatibility_info, model_compatibility);
  }
  // Delegate to the EP factory's validation method, passing hardware devices derived from our ep_devices_
  std::vector<const OrtHardwareDevice*> hardware_devices;
  hardware_devices.reserve(ep_devices_.size());
  for (const auto* ep_device : ep_devices_) {
    hardware_devices.push_back(ep_device->device);
  }
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(ep_factory_.ValidateCompiledModelCompatibilityInfo(&ep_factory_,
                                                                                            hardware_devices.data(),
                                                                                            hardware_devices.size(),
                                                                                            compatibility_info.c_str(),
                                                                                            &model_compatibility)));
  return Status::OK();
}

const OrtEp* PluginExecutionProvider::GetOrtEp() const {
  return ort_ep_.get();
}

std::unique_ptr<profiling::EpProfiler> PluginExecutionProvider::GetProfiler() {
  if (ort_ep_->ort_version_supported < 25 || ort_ep_->CreateProfiler == nullptr) {
    return {};
  }

  const logging::Logger& logger = GetEpLoggerOrDefault();
  OrtEpProfilerImpl* profiler_impl = nullptr;
  Status status = ToStatusAndRelease(ort_ep_->CreateProfiler(ort_ep_.get(), &profiler_impl));

  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "OrtEp::CreateProfiler for " << Type() << " returned an error status: "
                        << status.ErrorMessage();
    return {};
  }

  if (profiler_impl == nullptr) {
    return {};  // plugin EP doesn't have a profiler
  }

  std::unique_ptr<PluginEpProfiler> ep_profiler;
  status = PluginEpProfiler::Create(*profiler_impl, logger, Type(), /*out*/ ep_profiler);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << status.ErrorMessage();
    return {};
  }

  return ep_profiler;
}

ProviderOptions PluginExecutionProvider::GetProviderOptions() const {
  return provider_options_;
}

bool PluginExecutionProvider::IsGraphCaptureEnabled() const {
  if (ort_ep_->ort_version_supported < 26 || ort_ep_->IsGraphCaptureEnabled == nullptr) {
    return false;
  }

  if (!ort_ep_->IsGraphCaptureEnabled(ort_ep_.get())) {
    return false;
  }

  // Validate that the EP also implements IsGraphCaptured and ReplayGraph. Without these,
  // ORT-managed graph capture/replay cannot function correctly.
  if (ort_ep_->IsGraphCaptured == nullptr || ort_ep_->ReplayGraph == nullptr) {
    std::string missing;
    if (ort_ep_->IsGraphCaptured == nullptr) missing += "OrtEp::IsGraphCaptured ";
    if (ort_ep_->ReplayGraph == nullptr) missing += "OrtEp::ReplayGraph";
    LOGS(GetEpLoggerOrDefault(), WARNING)
        << Type() << " returned true from OrtEp::IsGraphCaptureEnabled but did not implement "
        << missing << ". ORT will not use this EP for graph capture/replay.";
    return false;
  }

  return true;
}

bool PluginExecutionProvider::IsGraphCaptured(int graph_annotation_id) const {
  if (ort_ep_->ort_version_supported < 26 || ort_ep_->IsGraphCaptured == nullptr) {
    return false;
  }
  return ort_ep_->IsGraphCaptured(ort_ep_.get(), graph_annotation_id);
}

Status PluginExecutionProvider::ReplayGraph(int graph_annotation_id, bool sync) {
  if (ort_ep_->ort_version_supported < 26 || ort_ep_->ReplayGraph == nullptr) {
    return Base::ReplayGraph(graph_annotation_id, sync);
  }
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(ort_ep_->ReplayGraph(ort_ep_.get(), graph_annotation_id)));
  if (sync) {
    ORT_RETURN_IF_ERROR(Sync());
  }
  return Status::OK();
}

Status PluginExecutionProvider::ReleaseCapturedGraph(int graph_annotation_id) {
  // For plugin EPs that don't implement ReleaseCapturedGraph (version < 27 or null function pointer),
  // fall back to the base class no-op implementation. This is intentional: the request is silently
  // ignored since the plugin EP doesn't support explicit graph resource release.
  if (ort_ep_->ort_version_supported < 27 || ort_ep_->ReleaseCapturedGraph == nullptr) {
    return Base::ReleaseCapturedGraph(graph_annotation_id);
  }
  return ToStatusAndRelease(ort_ep_->ReleaseCapturedGraph(ort_ep_.get(), graph_annotation_id));
}

OrtGraphCaptureNodeAssignmentPolicy PluginExecutionProvider::GetGraphCaptureNodeAssignmentPolicy() const {
  if (ort_ep_->ort_version_supported < 26 || ort_ep_->GetGraphCaptureNodeAssignmentPolicy == nullptr) {
    return OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP;
  }

  return ort_ep_->GetGraphCaptureNodeAssignmentPolicy(ort_ep_.get());
}

}  // namespace onnxruntime
