// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <map>

#include "core/common/const_pointer_container.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_allocator.h"
#include "core/framework/session_options.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/platform/path_lib.h"

namespace onnxruntime {
class Env;
class KernelRegistryManager;
class Node;
class SessionState;
class GraphViewer;
class OrtValueNameIdxMap;
class DataTransferManager;
class NodeArg;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
class MemoryInfo;
#endif

namespace logging {
class Logger;
}

namespace session_state_utils {
common::Status SaveInitializedTensors(
    const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
    const GraphViewer& graph, const AllocatorPtr& default_cpu_memory_info,
    const OrtValueNameIdxMap& ort_value_name_idx_map, const std::vector<OrtValueIndex>& initializer_allocation_order,
    ITensorAllocator& planner,
    const std::function<Status(int idx, const OrtValue& value, const OrtCallback& d, bool constant, bool sparse)>& save_tensor_func,
    const logging::Logger& logger,
    const DataTransferManager& data_transfer_mgr,
    const ExecutionPlanBase& exec_plan,
    const SessionOptions& session_options);
common::Status SaveInputOutputNamesToNodeMapping(const GraphViewer& graph,
                                                 SessionState& session_state,
                                                 const std::vector<const NodeArg*>& implicit_inputs);
}  // namespace session_state_utils
}  // namespace onnxruntime
