// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/dropout.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

#ifdef ENABLE_TRAINING
ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    Dropout);
#else
ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);
#endif

Status Dropout::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& shape = X.Shape();
  Tensor& Y = *context->Output(0, shape);

  if (!is_train_) {
    auto X_type = X.DataType();
    const void* source = X.DataRaw(X_type);
    void* target = Y.MutableDataRaw(X_type);
    if (target != source) {
      //If source and target pointers are not equal, we need to copy the data.
      memcpy(target, source, shape.Size() * X_type->Size());
    }
  } else {
    float scale = 1.0f / keep_prob_;
    Tensor& mask = *context->Output(1, shape);
    bool* mask_data = mask.template MutableData<bool>();

    // TODO: Compute is a const function, generator cannot be a private meber
    float seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
    std::bernoulli_distribution distribution(keep_prob_);
    std::generate_n(mask_data, shape.Size(), [&] { return distribution(generator); });

    EigenMap<float>(Y) = scale * EigenMap<float>(X).cwiseProduct(EigenMap<bool>(mask).cast<float>());
  }

  return Status::OK();
}

}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_KERNEL(
    DropoutGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    DropoutGrad);

Status DropoutGrad::Compute(OpKernelContext* context) const {
  const Tensor& dY = *context->Input<Tensor>(0);
  const TensorShape& shape = dY.Shape();
  Tensor& dX = *context->Output(0, shape);

  if (!is_train_) {
    auto data_type = dY.DataType();
    const void* source = dY.DataRaw(data_type);
    void* target = dX.MutableDataRaw(data_type);
    if (target != source) {
      //If source and target pointers are not equal, we need to copy the data.
      memcpy(target, source, shape.Size() * data_type->Size());
    }
  } else {
    const Tensor& mask = *context->Input<Tensor>(1);
    float scale = 1.0f / keep_prob_;
    EigenMap<float>(dX) = scale * EigenMap<float>(dY).cwiseProduct(EigenMap<bool>(mask).cast<float>());
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
