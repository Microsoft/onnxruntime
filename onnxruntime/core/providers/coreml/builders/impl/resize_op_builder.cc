// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ResizeOpBuilder : public BaseOpBuilder {
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

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }
};

// Helper functions
bool GetResizeScales(const InitializedTensorSet& initializers, const Node& node, std::vector<float>& scales) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3)
    return false;

  const auto& scales_tensor = *initializers.at(input_defs[2]->Name());
  if (scales_tensor.dims_size() != 1 || scales_tensor.dims()[0] != 4)
    return false;

  const float* scales_data = GetTensorFloatData(scales_tensor);
  scales = std::vector<float>{scales_data, scales_data + 4};
  return true;
}

bool GetResizeOutputSizes(const InitializedTensorSet& initializers, const Node& node, std::vector<int64_t>& sizes) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 4)
    return false;

  const auto& sizes_tensor = *initializers.at(input_defs[3]->Name());
  if (sizes_tensor.dims_size() != 1 || sizes_tensor.dims()[0] != 4)
    return false;

  const int64_t* sizes_data = GetTensorInt64Data(sizes_tensor);
  sizes = std::vector<int64_t>{sizes_data, sizes_data + 4};
  return true;
}

// Add operator related

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // We don't really use ROI here, so add it to skipped list if it's an initializer tensor
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // ROI
  model_builder.AddInputToSkip(node.InputDefs()[1]->Name());        // ROI

  // We will still add scales to the skipped list even sizes are present
  // since there is no use of it, we will not process it later
  model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // scales
  model_builder.AddInputToSkip(node.InputDefs()[2]->Name());        // scales

  if (node.InputDefs().size() > 3) {
    model_builder.AddInitializerToSkip(node.InputDefs()[3]->Name());  // sizes
    model_builder.AddInputToSkip(node.InputDefs()[3]->Name());        // sizes
  }
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  auto* coreml_upsample = layer->mutable_upsample();
  NodeAttrHelper helper(node);
  const auto mode = helper.Get("mode", "nearest");
  if (mode == "linear") {
    coreml_upsample->set_mode(COREML_SPEC::UpsampleLayerParams_InterpolationMode_BILINEAR);
  } else {  // we already checked the mode must be NN or Bilinear in IsOpSupportedImpl
    coreml_upsample->set_mode(COREML_SPEC::UpsampleLayerParams_InterpolationMode_NN);
  }

  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());

  if (input_defs.size() == 3) {  // use scales
    std::vector<float> scales;
    ORT_RETURN_IF_NOT(GetResizeScales(initializers, node, scales), "Error getting resize scales");
    coreml_upsample->add_scalingfactor(static_cast<int64_t>(scales[2]));
    coreml_upsample->add_scalingfactor(static_cast<int64_t>(scales[3]));
  } else {  // we already checked number of inputs in IsOpSupportedImpl
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Error getting input shape");
    std::vector<int64_t> output_sizes;
    ORT_RETURN_IF_NOT(GetResizeOutputSizes(initializers, node, output_sizes), "Error getting resize output_sizes");
    coreml_upsample->add_scalingfactor(static_cast<int64_t>(output_sizes[2] / input_shape[2]));
    coreml_upsample->add_scalingfactor(static_cast<int64_t>(output_sizes[3] / input_shape[3]));
  }

  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related

bool ResizeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  {  // check attributes
    NodeAttrHelper helper(node);
    const auto mode = helper.Get("mode", "nearest");
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS(logger, VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }

    const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool using_asymmetric = coord_trans_mode == "asymmetric";
    if (is_linear_resize) {
      // TODO, add support of align_corners and half_pixel
      if (!using_asymmetric) {
        LOGS(logger, VERBOSE) << "Resize bilinear, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }
    } else {
      // nearest neighbor resizing
      // For resize using nearest neighbor, we only support coord_trans_mode == "asymmetric" && nearest_mode == "floor"
      if (!using_asymmetric) {
        LOGS(logger, VERBOSE) << "Resize nearest neighbor, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }

      const auto nearest_mode = helper.Get("nearest_mode", "round_prefer_floor");
      if (nearest_mode != "floor") {
        LOGS(logger, VERBOSE) << "Resize nearest neighbor, unsupported nearest_mode, " << nearest_mode;
        return false;
      }
    }
  }

  {  // scales and sizes (if present) must be initializers
    if (input_defs.size() < 3) {
      LOGS(logger, VERBOSE) << "Input scales or sizes of Resize must be known";
      return false;
    }

    // scales
    if (input_defs.size() == 3 && !Contains(initializers, input_defs[2]->Name())) {
      LOGS(logger, VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    if (input_defs.size() > 3 && !Contains(initializers, input_defs[3]->Name())) {
      LOGS(logger, VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }

    // We want to check if the scales or sizes are not trying to resize on N/C channels here
    if (input_defs.size() == 3) {  // we are using scales
      std::vector<float> scales;
      if (!GetResizeScales(initializers, node, scales))
        return false;

      float scale_n = scales[0];
      float scale_c = scales[1];
      if (scale_n != 1.0f || scale_c != 1.0f) {
        LOGS(logger, VERBOSE) << "Scales of N/C channel should be 1"
                              << "Resize of N/C channels are not supported"
                              << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
        return false;
      }

      // For now we only support upscale, so the scale_h and scale_w should be an integer >= 1
      // TODO support ResizeBilinear
      float scale_h = scales[2];
      float scale_w = scales[3];

      // Onnx spec requires scale to be a positive float, so we are not checking that here
      if (roundf(scale_h) != scale_h) {
        LOGS(logger, VERBOSE) << "Resize: scale_h: " << scale_h << " is not a whole number";
        return false;
      }

      if (roundf(scale_w) != scale_w) {
        LOGS(logger, VERBOSE) << "Resize: scale_w: " << scale_w << " is not a whole number";
        return false;
      }
    } else {
      // we are using sizes
      std::vector<int64_t> output_sizes;
      if (!GetResizeOutputSizes(initializers, node, output_sizes))
        return false;

      auto output_size_n = output_sizes[0];
      auto output_size_c = output_sizes[1];
      if (output_size_n != input_shape[0] || output_size_c != input_shape[1]) {
        LOGS(logger, VERBOSE) << "Output sizes of N/C chanel should match the input sizes, "
                              << "Resize of N/C channels are not supported"
                              << ", input_size_n, " << input_shape[0] << ", output_size_n, " << output_size_n
                              << ". input_size_c, " << input_shape[1] << ", output_size_c, " << output_size_c;
        return false;
      }

      // For now we only support upscale, so the output_size_h and output_size_w should be an integer >= 1
      // TODO support ResizeBilinear
      auto output_size_h = output_sizes[2];
      auto output_size_w = output_sizes[3];
      auto input_size_h = input_shape[2];
      auto input_size_w = input_shape[3];

      // Onnx spec requires output sizes to be a positive integer, so we are not checking that here
      if (output_size_h % input_size_h != 0) {
        LOGS(logger, VERBOSE) << "Resize: output_size_h: " << output_size_h
                              << " is not a mutliple of input_size_h: " << input_size_h;
        return false;
      }

      if (output_size_w % input_size_w != 0) {
        LOGS(logger, VERBOSE) << "Resize: output_size_w: " << output_size_w
                              << " is not a mutliple of input_size_w: " << input_size_w;
        return false;
      }
    }
  }

  return true;
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
