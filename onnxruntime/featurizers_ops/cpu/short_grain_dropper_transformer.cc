// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/ShortGrainDropperFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace featurizers {

void ShortGrainDropperTransformerImpl(OpKernelContext* ctx) {
  // Create the transformer
  Microsoft::Featurizer::Featurizers::ShortGrainDropperTransformer transformer(
      [ctx]() {
        const auto* state_tensor(ctx->Input<Tensor>(0));
        const uint8_t* const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return Microsoft::Featurizer::Featurizers::ShortGrainDropperTransformer(archive);
      }());

  // Get the input
  const auto* input_tensor = ctx->Input<Tensor>(1);
  const std::string* input_data = input_tensor->template Data<std::string>();

  // Prepare the output
  const int64_t input_rows_num = input_tensor->Shape()[0];
  const int64_t strings_num = input_tensor->Shape()[1];
  TensorShape rows_shape({input_rows_num});
  Tensor* output_tensor(ctx->Output(0, rows_shape));
  bool* output_data(output_tensor->MutableData<bool>());

  // Transform
  for (int64_t rows_idx = 0; rows_idx < input_rows_num; ++rows_idx) {
    std::vector<std::string> input_data_vec;
    input_data_vec.reserve(strings_num);
    for (int64_t string_idx = 0; string_idx < strings_num; ++string_idx)
      input_data_vec.emplace_back(std::move(input_data[string_idx + rows_idx * strings_num]));
    output_data[rows_idx] = transformer.execute(input_data_vec);
  }
};

class ShortGrainDropperTransformer final : public OpKernel {
 public:
  explicit ShortGrainDropperTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override {
    ShortGrainDropperTransformerImpl(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    ShortGrainDropperTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>()),
    ShortGrainDropperTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
