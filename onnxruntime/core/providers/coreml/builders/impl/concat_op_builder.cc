// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ConcatOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  layer->mutable_concat()->set_sequenceconcat(false);

  for (const auto* input : node.InputDefs()) {
    LOGS(logger, VERBOSE) << "input name " << input->Name();
    *layer->mutable_input()->Add() = input->Name();
  }

  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related
bool ConcatOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /* input_params */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 2) {
    LOGS(logger, VERBOSE) << "Concat only support 2+ inputs, actual number of inputs: " << input_defs.size();
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  auto rank = input_shape.size();
  if (rank != 4) {
    // For some reason, the concat in CoreML running on 3d tensor will concat on wrong axis
    // Instead of concat on axis 0, it will concat on axis 1
    // Disable Concat support for 3d tensor for now
    // TODO, add ExpandDims and Squeeze, 3d -ExpandDims-> 4d -> Concat -Squeeze-> 3d
    LOGS(logger, VERBOSE) << "Concat only support 4d shape for now, input is "
                          << rank << "d shape";
    return false;
  }

  NodeAttrHelper helper(node);
  auto axis = static_cast<size_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));
  if (rank != axis + 3) {
    LOGS(logger, VERBOSE) << "Concat only support axis to be -3, actual axis: " << axis
                          << ", actual rank: " << rank;
    return false;
  }

  return true;
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
