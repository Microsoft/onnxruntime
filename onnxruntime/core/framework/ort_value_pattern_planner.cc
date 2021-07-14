// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/execution_plan_base.h"

namespace onnxruntime {
OrtValuePatternPlanner::OrtValuePatternPlanner(const ExecutionPlanBase& execution_plan, bool trace_using_counters)
    : execution_planner_(execution_plan) {
  for (auto& location : execution_plan.GetAllLocations()) {
    planner_map_.emplace(location, std::make_unique<MemPatternPlanner>(trace_using_counters));
  }
}

#ifdef ENABLE_TRAINING
common::Status OrtValuePatternPlanner::TraceAllocation(int ort_value_idx,
                                                       const AllocPlanPerValue::ProgramCounter& counter,
                                                       size_t size,
                                                       bool insert_end) {
  // TODO(codemzs): refactor code.
  const auto& location = execution_planner_.GetLocation(ort_value_idx);
  auto it = planner_map_.find(location);
  if (it == planner_map_.end()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  it->second->TraceAllocation(ort_value_idx, counter, size, insert_end);
  return common::Status::OK();
}
#endif

common::Status OrtValuePatternPlanner::TraceAllocation(int ort_value_idx, size_t size) {
  const auto& location = execution_planner_.GetLocation(ort_value_idx);
  auto it = planner_map_.find(location);
  if (it == planner_map_.end()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  it->second->TraceAllocation(ort_value_idx, size);
  return common::Status::OK();
}

common::Status OrtValuePatternPlanner::TraceFree(int ort_value_index) {
  const auto& location = execution_planner_.GetLocation(ort_value_index);
  auto it = planner_map_.find(location);
  if (it == planner_map_.end()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  it->second->TraceFree(ort_value_index);
  return common::Status::OK();
}

common::Status OrtValuePatternPlanner::GeneratePatterns(MemoryPatternGroup* out) {
  if (!out) return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);

  for (auto& it : planner_map_) {
    out->locations.push_back(it.first);
    out->patterns.push_back(it.second->GenerateMemPattern());
  }

  return common::Status::OK();
}

}  // namespace onnxruntime
