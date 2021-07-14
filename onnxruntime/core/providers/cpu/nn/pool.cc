// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/pool.h"

#include "core/framework/data_types_internal.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/nn/pool_functors.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 8, Input, 0,
    float,
    double);
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 12, Input, 0,
    double,
    float,
    int8_t,
    uint8_t);
}  // namespace op_kernel_type_control

using MaxPool8DataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 8, Input, 0);
using EnabledMaxPool8DataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 8, Input, 0);
using MaxPool12DataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 12, Input, 0);
using EnabledMaxPool12DataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, MaxPool, 12, Input, 0);

using AllEnabledMaxPoolDataTypes =
    utils::TypeSetUnion<
        EnabledMaxPool8DataTypes,
        EnabledMaxPool12DataTypes>;

template <typename T>
inline static void RunLoop(concurrency::ThreadPool* tp, std::ptrdiff_t total_channels, T&& task) {
  concurrency::ThreadPool::TryParallelFor(tp, total_channels, task.Cost(), task);
}

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  if (pool_attrs_.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, output_dims);

  const auto* X_data = X->template Data<T>();
  auto* Y_data = Y->template MutableData<T>();

  // The main loop
  const int64_t channels = x_shape[1];
  const int64_t height = x_shape[2];
  const int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  const int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  const int64_t pooled_height = output_dims[2];
  const int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  const int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;
  const int64_t total_channels = x_shape[0] * channels;
  const int64_t x_step = height * width * depth;
  const int64_t y_step = pooled_height * pooled_width * pooled_depth;

  switch (kernel_shape.size()) {
    case 1: {
      RunLoop<Pool1DTask<T, PoolType>>(tp, total_channels,
                                       {X_data, Y_data, x_step, y_step, pooled_height, stride_h(), height, kernel_shape,
                                        pads, pool_context_, pool_attrs_});

      break;
    }

    case 2: {
      RunLoop<Pool2DTask<T, PoolType>>(tp, total_channels,
                                       {X_data, Y_data, x_step, y_step, pooled_height, pooled_width, stride_h(),
                                        stride_w(), height, width, kernel_shape, pads, pool_context_, pool_attrs_});

      break;
    }
    case 3: {
      RunLoop<Pool3DTask<T, PoolType>>(
          tp, total_channels,
          {X_data, Y_data, x_step, y_step, pooled_height, pooled_width, pooled_depth, stride_h(), stride_w(),
           stride_d(), height, width, depth, kernel_shape, pads, pool_context_, pool_attrs_});

      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

Status PoolBase::Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!pool_attrs_.global_pooling) {
    ORT_RETURN_IF_NOT(pooling_dims == pool_attrs_.kernel_shape.size(),
                      "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape output_shape(output_dims);
  Tensor* Y = context->Output(0, output_shape);

  // edge case: one or more dims with value of 0
  if (output_shape.Size() == 0)
    return Status::OK();

  // Get access to the internal threadpool
  // Temporarily derive concurrency parameters without access to session state
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  MlasPool(kind, pooling_dims, X->Shape().GetDims().data(),
           pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
           pool_attrs_.global_pooling ? nullptr : pads.data(),
           pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(), output_dims.data(),
           X->template Data<float>(), Y->template MutableData<float>(), thread_pool);

  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1 /*VERSION*/>>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context, MlasMaximumPooling);
}

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context,
                           pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad : MlasAveragePoolingExcludePad);
}

Status MaxPoolV8::Compute(OpKernelContext* context) const {
  utils::MLTypeCallDispatcherFromTypeList<AllEnabledMaxPoolDataTypes>
      t_disp(context->Input<Tensor>(0)->GetElementType());
  return t_disp.InvokeRet<Status, ComputeHelper>(this, context);
}

template <typename T>
Status MaxPoolV8::ComputeImpl(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  // Use MLAS pooling if the index output tensor is not used
  // and also if dilation is not required

  bool need_dilation = false;
  for (auto n : pool_attrs_.dilations) {
    need_dilation |= n > 1;
  }

  // MLAS implementation currently supports only floats
  if (std::is_same<T, float>::value) {
    if (OpKernel::Node().OutputDefs().size() == 1 && pool_attrs_.storage_order == 0 && !need_dilation) {
      return PoolBase::Compute(context, MlasMaximumPooling);
    }
  }

  const auto* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, output_dims);
  Tensor* I = context->Output(1, output_dims);

  const auto* X_data = X->template Data<T>();
  auto* Y_data = Y->template MutableData<T>();
  int64_t* I_data = I != nullptr ? I->template MutableData<int64_t>() : nullptr;

  // The main loop
  int64_t channels = x_shape[1];
  int64_t height = x_shape[2];
  int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  int64_t pooled_height = output_dims[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;
  const int64_t total_channels = x_shape[0] * channels;

  switch (kernel_shape.size()) {
    case 1: {
      int64_t x_step = height;
      int64_t y_step = pooled_height;
      const int64_t dilation_h = pool_attrs_.dilations[0];

      RunLoop<MaxPool1DTask<T>>(tp, total_channels,
                                {X_data, Y_data, I_data, x_step, y_step, dilation_h, pooled_height, stride_h(),
                                 height, kernel_shape, pads});
      break;
    }

    case 2: {
      int64_t x_step = height * width;
      int64_t y_step = pooled_height * pooled_width;
      const int64_t dilation_h = pool_attrs_.dilations[0];
      const int64_t dilation_w = pool_attrs_.dilations[1];
      RunLoop<MaxPool2DTask<T>>(
          tp, total_channels,
          {X_data, Y_data, I_data, x_step, y_step, dilation_h, dilation_w, pooled_height, pooled_width, stride_h(),
           stride_w(), height, width, kernel_shape, pads, pool_attrs_.storage_order});
      break;
    }
    case 3: {
      int64_t x_step = height * width * depth;
      int64_t y_step = pooled_height * pooled_width * pooled_depth;
      const int64_t dilation_h = pool_attrs_.dilations[0];
      const int64_t dilation_w = pool_attrs_.dilations[1];
      const int64_t dilation_d = pool_attrs_.dilations[2];
      RunLoop<MaxPool3DTask<T>>(tp, total_channels,
                                {X_data, Y_data, I_data, x_step, y_step,
                                 dilation_h, dilation_w, dilation_d, pooled_height, pooled_width,
                                 pooled_depth, stride_h(), stride_w(), stride_d(), height,
                                 width, depth, kernel_shape, pads, pool_attrs_.storage_order});
      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

template <typename T>
Status MaxPoolV8::ComputeImplOptimized(OpKernelContext* context) const {

  const auto* X = context->Input<Tensor>(0);
  const auto* X_data = X->template Data<T>();
  const TensorShape& x_shape = X->Shape();
  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  const auto& dilations = pool_attrs_.dilations;
  const auto& pool_size = pool_attrs_.kernel_shape;

  if (dilations[0] != stride_h() ||
      pool_size.size() > 1 && dilations[1] != stride_w() || 
      pool_size.size() > 2 && dilations[2] != stride_d() ||
      context->Output(1, output_dims) ||
      pool_size.size() == 1 && pool_size[0] < 128 || 
      pool_size.size() == 2 && pool_size[0] * pool_size[1] < 64*64 || 
      pool_size.size() == 3 && pool_size[0] * pool_size[1] * pool_size[2] < 32*32*32) {

    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Arguments are not optimizable.");
  }

  Tensor* Y = context->Output(0, output_dims);
  auto* Y_data = Y->template MutableData<T>();
  auto tp = context->GetOperatorThreadPool();
  auto channels = x_shape[0] * x_shape[1];

  switch (pool_size.size()) {
    case 1: {
      RunLoop<MaxPool1DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], output_dims[2],
                                    pads[0], pads[1], stride_h(), pool_size[0]});
      break;
    }
    case 2: {
      RunLoop<MaxPool2DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], x_shape[3],
                                    output_dims[2], output_dims[3],
                                    pads[0], pads[2], pads[1], pads[3],
                                    stride_h(), stride_w(),
                                    pool_size[0], pool_size[1]});
      break;
    }
    case 3: {
      RunLoop<MaxPool3DTaskOpt<T>>(tp,
                                   channels,
                                   {X_data, Y_data,
                                    x_shape[2], x_shape[3], x_shape[4],
                                    output_dims[2], output_dims[3], output_dims[4],
                                    pads[0], pads[3], pads[1], pads[4], pads[2], pads[5],
                                    stride_h(), stride_w(), stride_d(),
                                    pool_size[0], pool_size[1], pool_size[2]});
      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  return Status::OK();
}
 
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(AveragePool, 7, 9,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(AveragePool, 10, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(AveragePool, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(MaxPool, 1, 7,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, MaxPool<1 /*VERSION*/>>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(MaxPool, 8, 11,
                                   KernelDefBuilder()
                                       .TypeConstraint(
                                           "T",
                                           BuildKernelDefConstraintsFromTypeList<MaxPool8DataTypes>(),
                                           BuildKernelDefConstraintsFromTypeList<EnabledMaxPool8DataTypes>())
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                   MaxPoolV8);

ONNX_CPU_OPERATOR_KERNEL(MaxPool, 12,
                         KernelDefBuilder()
                             .TypeConstraint(
                                 "T",
                                 BuildKernelDefConstraintsFromTypeList<MaxPool12DataTypes>(),
                                 BuildKernelDefConstraintsFromTypeList<EnabledMaxPool12DataTypes>())
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                         MaxPoolV8);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(LpPool, 2, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(LpPool, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalLpPool, 2, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalAveragePool, 1,
                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(GlobalMaxPool, 1, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Pool<float, MaxPool<1 /*VERSION*/>>);

}  // namespace onnxruntime
