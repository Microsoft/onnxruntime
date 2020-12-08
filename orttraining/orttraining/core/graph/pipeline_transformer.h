// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);

Status TransformGraphForPipeline(
    Graph& graph,
    const std::unordered_set<std::string>& initializer_names_to_preserve,
    pipeline::PipelineTensorNames& pipeline_tensor_names);

Status ApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage,
    std::vector<std::string>& non_differentiable_edge_names);

Status RemoveNonDifferentiableEdgeInPartition(
  Graph& graph,
  const std::vector<std::string>& non_differentiable_edge_names);

}  // namespace training
}  // namespace onnxruntime
