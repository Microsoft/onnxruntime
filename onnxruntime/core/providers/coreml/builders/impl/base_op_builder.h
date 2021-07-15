// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/coreml/builders/op_builder.h"

namespace onnxruntime {
namespace coreml {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  // Add operator related
 public:
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */, const Node& /* node */) const override {}
  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                           const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                       const logging::Logger& logger) const ORT_MUST_USE_RESULT = 0;

  static std::unique_ptr<COREML_SPEC::NeuralNetworkLayer>
  CreateNNLayer(ModelBuilder& model_builder, const Node& node);

  static std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> CreateNNLayer(const std::string& layer_name);

  // Operator support related
 public:
  bool IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                     const logging::Logger& logger) const override;

 protected:
  virtual bool IsOpSupportedImpl(const Node& /* node */, const OpBuilderInputParams& /* input_params */,
                                 const logging::Logger& /* logger */) const {
    return true;
  }

  virtual bool HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const;

  virtual int GetMinSupportedOpSet(const Node& /* node */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /* node */) const { return 14; }

 private:
  bool HasSupportedOpSet(const Node& node, const logging::Logger& logger) const;
  bool HasSupportedInputs(const Node& node, const logging::Logger& logger) const;
};

}  // namespace coreml
}  // namespace onnxruntime