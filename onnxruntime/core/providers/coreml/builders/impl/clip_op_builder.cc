// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ClipOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Both min and max values will be injected into the layer, no need to add to the model
  if (node.SinceVersion() >= 11) {
    if (node.InputDefs().size() > 1)
      model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());

    if (node.InputDefs().size() > 2)
      model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());
  }
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& node_name = node.Name();
  const auto& input_name = node.InputDefs()[0]->Name();
  const auto& output_name = node.OutputDefs()[0]->Name();
  float min, max;
  ORT_RETURN_IF_NOT(GetClipMinMax(model_builder.GetInitializerTensors(), node, min, max, logger), "GetClipMinMax failed");

  bool has_min = min != std::numeric_limits<float>::lowest();
  bool has_max = max != std::numeric_limits<float>::max();

  if (!has_min && !has_max) {
    // Clip without min/max is an identity node
    // In CoreML we don't have identity, use ActivationLinear instead
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);
    layer->mutable_activation()->mutable_linear()->set_alpha(1.0f);
    *layer->mutable_input()->Add() = input_name;
    *layer->mutable_output()->Add() = output_name;

    model_builder.AddLayer(std::move(layer));
  } else {
    // The implementation of clip(min, max) is done by
    // 1. Clipping at min -> max(input, min) is handled by
    //    min_output = threshold(input, min)
    // 2. Clipping at max -> min(min_output, max) is handled by
    //    output = -1 * (threshold(-min_output, -max))

    // Now we have at least one or min or max is not default value
    // Clipping at max will need take the output of clipping at min, or the node input, if min value is default
    // If max value is default, the output of clipping at min will be the output of the node
    std::string min_output_name = output_name;
    if (has_max) {
      min_output_name = has_min
                            ? model_builder.GetUniqueName(node_name + "min_output")
                            : input_name;
    }

    // Handle clipping at min first
    if (has_min) {
      const auto clip_min_layer_name = model_builder.GetUniqueName(MakeString(node_name, "_Clip_min"));
      std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> min_layer = CreateNNLayer(clip_min_layer_name);
      if (min == 0.0f) {  // If min is 0. then this min will be handled by relu
        min_layer->mutable_activation()->mutable_relu();
      } else {  // otherwise, min will be handled by unary->threshold
        min_layer->mutable_unary()->set_alpha(min);
        min_layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::THRESHOLD);
      }

      *min_layer->mutable_input()->Add() = input_name;
      *min_layer->mutable_output()->Add() = min_output_name;
      model_builder.AddLayer(std::move(min_layer));
    }

    // Clipping at max is handled by -1 * (threshold (-min_output, -max))
    if (has_max) {
      const auto threshold_output_name = model_builder.GetUniqueName(MakeString(node_name, "threshold_output"));
      {  // Add threshold layer, which is actually max( -1 * min_output, -max)
        const auto clip_max_threshold_layer_name =
            model_builder.GetUniqueName(MakeString(node_name, "_Clip_max_threshold"));
        auto threshold_layer = CreateNNLayer(clip_max_threshold_layer_name);
        threshold_layer->mutable_unary()->set_alpha(-max);
        threshold_layer->mutable_unary()->set_scale(-1.0f);
        threshold_layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::THRESHOLD);
        *threshold_layer->mutable_input()->Add() = min_output_name;
        *threshold_layer->mutable_output()->Add() = threshold_output_name;
        model_builder.AddLayer(std::move(threshold_layer));
      }
      {  // Add linear activation layer -1 * threshold_output
        const auto clip_max_linear_layer_name =
            model_builder.GetUniqueName(MakeString(node_name, "_Clip_max_linear"));
        auto linear_layer = CreateNNLayer(clip_max_linear_layer_name);
        linear_layer->mutable_activation()->mutable_linear()->set_alpha(-1.0f);
        *linear_layer->mutable_input()->Add() = threshold_output_name;
        *linear_layer->mutable_output()->Add() = output_name;
        model_builder.AddLayer(std::move(linear_layer));
      }
    }
  }

  return Status::OK();
}

// Operator support related

bool ClipOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  float min, max;
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  return GetClipMinMax(initializers, node, min, max, logger);
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
