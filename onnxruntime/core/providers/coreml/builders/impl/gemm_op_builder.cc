// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace coreml {

class GemmOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& /* node */, const OpBuilderInputParams& /* input_params */,
                         const logging::Logger& /* logger */) const override;
};

// Add operator related

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& op = node.OpType();
  const auto& input_defs(node.InputDefs());
  // We have already embedded the weights (matrix B and C(if any)) into the coreml layer
  // No need to copy them later to reduce memory consumption
  model_builder.AddInitializerToSkip(input_defs[1]->Name());
  if (op == "Gemm" && input_defs.size() > 2) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());
  }
}

// This is an internal function, requires input tensor to be 2d float tensor
// TODO, add support of other data types
static std::vector<float> GetTensorFloatDataTransposed(const ONNX_NAMESPACE::TensorProto& tensor) {
  const float* src_data = GetTensorFloatData(tensor);
  const auto& tensor_shape = tensor.dims();
  auto x_t = SafeInt<size_t>(tensor_shape[0]);
  auto y_t = SafeInt<size_t>(tensor_shape[1]);
  std::vector<float> transposed_data(x_t * y_t);
  for (size_t x = 0; x < x_t; x++) {
    for (size_t y = 0; y < y_t; y++) {
      transposed_data[y * x_t + x] = src_data[x * y_t + y];
    }
  }

  return transposed_data;
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& b_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
  const auto& b_shape = b_tensor.dims();

  auto* coreml_inner_product = layer->mutable_innerproduct();

  // The coreml innerproduct weight (matrix B) is stored transposed
  // - for MatMul and Gemm (transB = 0), the coreml weight is B'
  // - for Gemm (transB = 1), the coreml weight is B
  if (op_type == "MatMul") {
    coreml_inner_product->set_inputchannels(b_shape[0]);
    coreml_inner_product->set_outputchannels(b_shape[1]);
    // Add weight (b of MatMul)
    const auto b_transposed = GetTensorFloatDataTransposed(b_tensor);
    CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_transposed.data(), b_transposed.size());
  } else {  // Gemm
    NodeAttrHelper helper(node);
    const auto transB = helper.Get("transB", 0);
    if (transB == 0) {
      coreml_inner_product->set_inputchannels(b_shape[0]);
      coreml_inner_product->set_outputchannels(b_shape[1]);
      const auto b_transposed = GetTensorFloatDataTransposed(b_tensor);
      CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_transposed.data(), b_transposed.size());
    } else {
      coreml_inner_product->set_inputchannels(b_shape[1]);
      coreml_inner_product->set_outputchannels(b_shape[0]);
      // Add weight (b of MatMul)
      CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_tensor);
    }

    // Add bias if present
    if (input_defs.size() > 2) {
      coreml_inner_product->set_hasbias(true);
      const auto& bias_tensor = *model_builder.GetInitializerTensors().at(input_defs[2]->Name());
      CreateCoreMLWeight(*coreml_inner_product->mutable_bias(), bias_tensor);
    }
  }

  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related

bool GemmOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (!Contains(initializers, input_defs[b_idx]->Name())) {
    LOGS(logger, VERBOSE) << "B of Gemm/Matmul must be an initializer tensor";
    return false;
  }

  std::vector<int64_t> a_shape;
  {
    if (!GetShape(*input_defs[a_idx], a_shape, logger))
      return false;

    if (a_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "A must be 2D";
      return false;
    }

    if (Product(a_shape) == 0) {
      LOGS(logger, VERBOSE) << "A must be non-empty";
      return false;
    }
  }

  std::vector<int64_t> b_shape;
  {
    if (!GetShape(*input_defs[b_idx], b_shape, logger))
      return false;

    if (b_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "B must be 2D";
      return false;
    }

    if (Product(b_shape) == 0) {
      LOGS(logger, VERBOSE) << "B must be non-empty";
      return false;
    }
  }

  if (op_type == "Gemm") {
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS(logger, VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported."
                            << " transA " << transA
                            << " alpha " << alpha
                            << " beta " << beta;
      return false;
    }

    // C of Gemm
    // For now we only support {n} or {1,n} tensor
    if (input_defs.size() == 3) {
      if (!Contains(initializers, input_defs[c_idx]->Name())) {
        LOGS(logger, VERBOSE) << "C of Gemm must be an initializer tensor";
        return false;
      }

      std::vector<int64_t> c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape, logger))
        return false;

      size_t c_dim = c_shape.size();

      if (c_dim == 0) {
        LOGS(logger, VERBOSE) << "C of Gemm cannot be a scalar";
        return false;
      }

      if (c_dim != 1) {
        // If C is a (2+)d tensor, it must have the format {1, 1, ..., 1, n}
        // where every except the last dimension should be 1
        for (size_t i = 0; i < c_dim - 1; ++i) {
          if (c_shape[i] != 1) {
            LOGS(logger, VERBOSE) << "C of Gemm must be a vector or a tensor with only last dimension != 1";
            return false;
          }
        }
      }

      auto c_size = c_shape[c_dim - 1];
      if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS(logger, VERBOSE) << "C of Gemm must be a vector of b_shape["
                              << (transB == 0 ? "1" : "0") << "]"
                              << " b_shape: [" << b_shape[0] << ", " << b_shape[1] << "]"
                              << " c_size: " << c_size;

        return false;
      }
    }
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Gemm",
          "MatMul",
      };

  op_registrations.builders.push_back(std::make_unique<GemmOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}
}  // namespace coreml
}  // namespace onnxruntime
