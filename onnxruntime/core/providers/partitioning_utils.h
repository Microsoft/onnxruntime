// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/graph/basic_types.h"

namespace onnxruntime {
struct ComputeCapability;
class GraphViewer;
class NodeArg;
class Node;

namespace utils {

/**
Called to check whether a node is supported.

@param node The node to check.

@return Whether the node is supported.
*/
using IsNodeSupportedFn = std::function<bool(const Node& node)>;

/**
Called to indicate a completed partition node group.
The partition is kept or discarded based on the return value.

@param group The partition node group.

@return Whether to keep the partition.
*/
using OnGroupClosedFn = std::function<bool(const std::vector<const Node*>& group)>;

/**
Called to create a metadef name.
Most likely should call IExecutionProvider::GenerateMetaDefId.
See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc for example usage.

@return The metadef name.
*/
using GenerateMetadefNameFn = std::function<std::string()>;

/**
Create the supported partitions for the execution provider.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param is_node_supported_fn Callback to check whether a node is supported.
@param on_group_closed_fn Callback to indicate a completed partition node group.
@param generate_metadef_name_fn Callback to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param debug_output Print diagnostic output about the partitions and reasons for partition breaks.
                    No-op in a release build.

@returns ComputeCapability instances for all partitions assigned to the execution provider.
*/
std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const IsNodeSupportedFn& is_node_supported_fn,
                          const OnGroupClosedFn& on_group_closed_fn,
                          const GenerateMetadefNameFn& generate_metadef_name_fn,
                          const std::string& execution_provider_name,
                          bool debug_output = false);

/** 
Create the supported partitions for the execution provider.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param supported_nodes Set of nodes that the execution provider wants to handle.
@param stop_ops Set of operator names at which we stop considering nodes for assignment to this execution provider.
@param generate_metadef_name Functor to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param debug_output Print diagnostic output about the partitions and reasons for partition breaks.
                    No-op in a release build.

@returns ComputeCapability instances for all partitions assigned to the execution provider.
*/
std::vector<std::unique_ptr<ComputeCapability>> CreateSupportedPartitions(
    const GraphViewer& graph_viewer,
    const std::unordered_set<const Node*>& supported_nodes,
    const std::unordered_set<std::string>& stop_ops,
    const GenerateMetadefNameFn& generate_metadef_name,
    const std::string& execution_provider_name,
    bool debug_output = false);

/**
Create a ComputeCapability instance from the group of nodes.
Will automatically determine the inputs and outputs required.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param group Group of nodes to include in the ComputeCapability instance.
@param generate_metadef_name Functor to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.

@returns New ComputeCapability instance.

@remarks Prefer using CreateSupportedPartitions where possible, but if you need custom handling this provides a
         convenient way to correctly create the ComputeCapability instance.
*/
std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const GenerateMetadefNameFn& generate_metadef_name,
                                                         const std::string& execution_provider_name);

/**
Create the set of nodes to exclude based on a set of stop ops.
Stop op nodes and nodes downstream from them will be excluded.

@param graph_viewer GraphViewer with the nodes to consider.
@param stop_ops The set of stop ops.

@return The set of excluded nodes.
*/
std::unordered_set<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                      const std::unordered_set<std::string>& stop_ops);
}  // namespace utils
}  // namespace onnxruntime
