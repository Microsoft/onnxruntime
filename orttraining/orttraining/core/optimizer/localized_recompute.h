// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GeluRecompute

Recompute Gelu/BiasGelu/FastGelu

*/
class GeluRecompute : public GraphTransformer {
 public:
  GeluRecompute() noexcept : GraphTransformer("GeluRecompute") {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  bool SatisfyCondition(const Node& node) const;
};

/**
@Class AttentionDropoutRecompute

Recompute Dropout in the attention layer

*/
class AttentionDropoutRecompute : public GraphTransformer {
 public:
  AttentionDropoutRecompute() noexcept : GraphTransformer("AttentionDropoutRecompute") {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  bool SatisfyCondition(const Node& node) const;
};

/**
@Class ExpandRecompute

Recompute the Expand nodes

*/
class ExpandRecompute : public GraphTransformer {
 public:
  ExpandRecompute() noexcept : GraphTransformer("ExpandRecompute") {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  bool SatisfyCondition(const Node& node) const;
};

}  // namespace onnxruntime
