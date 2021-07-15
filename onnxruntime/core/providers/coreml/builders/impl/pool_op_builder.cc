// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace coreml {

class PoolOpBuilder : public BaseOpBuilder {
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

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  auto* coreml_pool = layer->mutable_pooling();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  bool is_global_pooling = false;
  bool is_average_pool = false;
  if (op_type == "GlobalAveragePool") {
    is_global_pooling = true;
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_AVERAGE);
  } else if (op_type == "GlobalMaxPool") {
    is_global_pooling = true;
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_MAX);
  } else if (op_type == "AveragePool") {
    is_average_pool = true;
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_AVERAGE);
  } else if (op_type == "MaxPool") {
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_MAX);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "PoolOpBuilder, unknown op: ", op_type);
  }

  if (is_global_pooling) {
    coreml_pool->set_globalpooling(true);
    coreml_pool->mutable_valid();
  } else {  // AveragePool or MaxPool
    NodeAttrHelper helper(node);
    const auto kernel_shape = helper.Get("kernel_shape", std::vector<int64_t>{0, 0});
    const auto strides = helper.Get("strides", std::vector<int64_t>{1, 1});
    const auto onnx_pads = helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0});

    coreml_pool->add_kernelsize(kernel_shape[0]);
    coreml_pool->add_kernelsize(kernel_shape[1]);
    coreml_pool->add_stride(strides[0]);
    coreml_pool->add_stride(strides[1]);
    coreml_pool->set_avgpoolexcludepadding(helper.Get("count_include_pad", 0) == 0);
    coreml_pool->set_globalpooling(false);

    // Add Padding
    // Usually using autopadding is more efficient than using explicit padding
    // Try to see if we can map explicit padding to auto padding
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
    AutoPadType auto_pad_type;
    ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, kernel_shape[0], kernel_shape[1],
                                      onnx_pads, strides, {1, 1} /* dilations */,
                                      StringToAutoPadType(helper.Get("auto_pad", "NOTSET")),
                                      auto_pad_type));

    if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
      auto* padding_type = coreml_pool->mutable_same();
      if (AutoPadType::SAME_LOWER == auto_pad_type) {  // default is SAME_UPPER
        padding_type->set_asymmetrymode(COREML_SPEC::SamePadding_SamePaddingMode_TOP_LEFT_HEAVY);
      }
    } else {
      auto* padding_type = coreml_pool->mutable_valid();
      if (AutoPadType::NOTSET == auto_pad_type && onnx_pads != std::vector<int64_t>{0, 0, 0, 0}) {
        // NOTSET is adding the explicit padding to the ValidPadding.paddingAmounts
        auto* height_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        height_border->set_startedgesize(onnx_pads[0]);
        height_border->set_endedgesize(onnx_pads[2]);
        auto* width_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        width_border->set_startedgesize(onnx_pads[1]);
        width_border->set_endedgesize(onnx_pads[3]);
      }
    }
  }

  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related
bool PoolOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /* input_params */,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << input_defs[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  if (op_type == "AveragePool" || op_type == "MaxPool") {
    NodeAttrHelper helper(node);
    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS(logger, VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS(logger, VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    // TODO, add support of the ceil_mode by adjusting the padding
    // See https://stackoverflow.com/questions/59906456/in-pytorchs-maxpool2d-is-padding-added-depending-on-ceil-mode
    // and https://github.com/apple/coremltools/blob/1931758aae383c83daddfc56f11a24a9d2bf4b87/coremltools/converters/mil/frontend/torch/ops.py#L621-L644
    if (helper.Get("ceil_mode", 0) == 1) {
      LOGS(logger, VERBOSE) << "ceil_mode == 1 is not supported for pooling";
      return false;
    }

    if (helper.Get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      LOGS(logger, VERBOSE) << "Dilations of pooling is not supported";
      return false;
    }

    if (node.OutputDefs().size() != 1) {
      LOGS(logger, VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  }

  return true;
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
      };

  op_registrations.builders.push_back(std::make_unique<PoolOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
