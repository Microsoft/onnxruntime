// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include "onnxruntime_c_api.h"
#include "ep.h"

// Plugin EPs can provide two types of custom ops:
//
// 1. A full OrtCustomOp with a concrete kernel implementation
//    - This Example EP demonstrates this approach.
//    - In GetCapability(), it calls EpGraphSupportInfo_AddSingleNode() to inform ORT
//      that the custom node should NOT be fused or compiled. Instead, ORT should invoke
//      the custom node's Compute() function at runtime.
//
// 2. A "placeholder" OrtCustomOp with an empty kernel implementation
//    - A compile-based Plugin EP can supply an OrtCustomOp whose CustomKernel::Compute()
//      does nothing. The purpose is to satisfy model validation during model loading by
//      registering the custom op as a valid operator in the session.
//    - In GetCapability(), the EP should call EpGraphSupportInfo_AddNodesToFuse() to
//      notify ORT that this custom node should be fused and compiled by the EP.
//    - In Compile(), the EP executes its compiled bits to perform inference for
//      the fused custom node.
//
// Note: Approach #2 is suitable for plugin EPs that support custom operator libraries.

struct CustomMulKernel : MulKernel {
  CustomMulKernel(const OrtApi& ort_api,
                  const OrtLogger& logger,
                  const std::unordered_map<std::string, FloatInitializer>& float_initializers,
                  std::string input0_name,
                  std::string input1_name) : MulKernel(ort_api, logger, float_initializers,
                                                       input0_name, input1_name) {
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* kernel_ctx) {
    return MulKernel::Compute(kernel_ctx);
  }
};

struct ExampleEpCustomOp : Ort::CustomOpBase<ExampleEpCustomOp, CustomMulKernel, /*WithStatus*/ true> {
  explicit ExampleEpCustomOp(const char* provider, ExampleEpFactory* factory) : provider_(provider),
                                                                                factory_(factory) {
  }

  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** op_kernel) const;

  OrtStatusPtr KernelComputeV2(void* op_kernel, OrtKernelContext* context) const;

  const char* GetName() const { return name_; };

  void SetName(const char* name) { name_ = name; };

  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return num_inputs_; };

  void SetInputTypeCount(size_t num) { num_inputs_ = num; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  size_t GetOutputTypeCount() const { return num_outputs_; };

  void SetOutputTypeCount(size_t num) { num_outputs_ = num; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  bool GetVariadicInputHomogeneity() const {
    return false;  // heterogenous
  }

  bool GetVariadicOutputHomogeneity() const {
    return false;  // heterogeneous
  }

 private:
  const char* provider_ = nullptr;
  const char* name_ = nullptr;
  size_t num_inputs_ = 1;   // set to 1 to match with default min_arity for variadic input
  size_t num_outputs_ = 1;  // set to 1 to match with default min_arity for variadic output
  ExampleEpFactory* factory_ = nullptr;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
};
