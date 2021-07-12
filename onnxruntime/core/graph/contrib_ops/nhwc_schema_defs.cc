// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/quantization_defs.h"

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
}  // namespace ONNX_NAMESPACE

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
class NhwcInferenceContext : public InferenceContext {
 public:
  NhwcInferenceContext(InferenceContext& ctx) : ctx_(ctx) {
  }

  void TransposeInputShape() {
    const auto* nhwc_type = ctx_.getInputType(0);
    if (nhwc_type != nullptr && nhwc_type->tensor_type().has_shape()) {
      const auto& nhwc_shape = nhwc_type->tensor_type().shape();
      const int rank = nhwc_shape.dim_size();
      if (rank < 2) {
        fail_shape_inference("Input tensor must have at least 2 dimensions");
      }
      // Convert input shape from {N, H, W, C} to {N, C, H, w}.
      auto* nchw_shape = input_type_.mutable_tensor_type()->mutable_shape();
      *nchw_shape->add_dim() = nhwc_shape.dim(0);
      *nchw_shape->add_dim() = nhwc_shape.dim(rank - 1);
      for (int i = 1; i < rank - 1; i++) {
        *nchw_shape->add_dim() = nhwc_shape.dim(i);
      }
    }
  }

  void TransposeOutputShape() {
    if (output_type_.tensor_type().has_shape()) {
      const auto& nchw_shape = output_type_.tensor_type().shape();
      const int rank = nchw_shape.dim_size();
      if (rank < 2) {
        fail_shape_inference("Output tensor must have at least 2 dimensions");
      }
      // Convert output shape from {N, C, H, W} to {N, H, w, C}.
      auto* nhwc_shape = ctx_.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      *nhwc_shape->add_dim() = nchw_shape.dim(0);
      for (int i = 2; i < rank; i++) {
        *nhwc_shape->add_dim() = nchw_shape.dim(i);
      }
      *nhwc_shape->add_dim() = nchw_shape.dim(1);
    }
  }

 protected:
  const AttributeProto* getAttribute(const std::string& name) const override {
    return ctx_.getAttribute(name);
  }

  size_t getNumInputs() const noexcept override {
    return ctx_.getNumInputs();
  }

  const TypeProto* getInputType(size_t index) const override {
    return (index == 0) ? &input_type_ : ctx_.getInputType(index);
  }

  const TensorProto* getInputData(size_t index) const override {
    ORT_UNUSED_PARAMETER(index);
    return nullptr;
  }

  size_t getNumOutputs() const noexcept override {
    return ctx_.getNumOutputs();
  }

  TypeProto* getOutputType(size_t index) override {
    return (index == 0) ? &output_type_ : ctx_.getOutputType(index);
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    ORT_UNUSED_PARAMETER(attribute_name);
    return nullptr;
  }

  const SparseTensorProto* getInputSparseData(size_t) const override {
    return nullptr;
  }


 private:
  InferenceContext& ctx_;
  TypeProto input_type_;
  TypeProto output_type_;
};

void convPoolShapeInferenceNhwc(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // Reuse the NCHW implementation by transposing the input/output tensor using
  // a local inference context.
  NhwcInferenceContext nhwc_ctx(ctx);
  nhwc_ctx.TransposeInputShape();
  convPoolShapeInference(nhwc_ctx, use_dilation, require_kernel_shape, input1Idx, input2Idx);
  nhwc_ctx.TransposeOutputShape();
}

void RegisterNhwcSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearConv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "x", "", "T1")
      .Input(1, "x_scale", "", "tensor(float)")
      .Input(2, "x_zero_point", "", "T1")
      .Input(3, "w", "", "T2")
      .Input(4, "w_scale", "", "tensor(float)")
      .Input(5, "w_zero_point", "", "T2")
      .Input(6, "y_scale", "", "tensor(float)")
      .Input(7, "y_zero_point", "", "T3")
      .Input(8, "B", "", "T4", OpSchema::Optional)
      .Output(0, "y", "", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "")
      .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "")
      .TypeConstraint("T4", {"tensor(int32)"}, "")
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("channels_last", "", AttributeProto::INT, static_cast<int64_t>(0))
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        auto x_type = ctx.getInputType(0);
        auto w_type = ctx.getInputType(3);
        if (nullptr == x_type || nullptr == w_type ||
            x_type->value_case() != TypeProto::kTensorType ||
            w_type->value_case() != TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        auto x_zero_point_type = ctx.getInputType(2);
        if (nullptr == x_zero_point_type ||
            x_zero_point_type->tensor_type().elem_type() !=
                x_type->tensor_type().elem_type()) {
          fail_type_inference(
              "input and zero_point pair is expected to have be same type.");
        }

        auto w_zero_point_type = ctx.getInputType(5);
        if (nullptr == w_zero_point_type ||
            w_zero_point_type->tensor_type().elem_type() !=
                w_type->tensor_type().elem_type()) {
          fail_type_inference(
              "weight and zero_point pair is expected to have same type.");
        }

        propagateElemTypeFromInputToOutput(ctx, 7, 0);

        if (getAttribute(ctx, "channels_last", 0) == 0) {
          convPoolShapeInference(ctx, true, false, 0, 3);
        } else {
          convPoolShapeInferenceNhwc(ctx, true, false, 0, 3);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NhwcMaxPool)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "x", "", "T")
      .Output(0, "y", "", "T")
      .TypeConstraint("T", {"tensor(int8)", "tensor(uint8)"}, "")
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0))
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        convPoolShapeInferenceNhwc(ctx, true, true, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearGlobalAveragePool)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
QLinearGlobalAveragePool consumes an input tensor X and applies Average pooling across
the values in the same channel. This is equivalent to AveragePool with kernel size
equal to the spatial dimension of input tensor. Input is of type uint8_t or int8_t.
)DOC")
      .Attr("channels_last", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(
          0,
          "X",
          "Input data tensor from the previous operator; According to channels_last, "
          "dimensions for image case are (N x C x H x W), or (N x H x W x C) "
          "where N is the batch size, C is the number of "
          "channels, and H and W are the height and the width "
          "of the data. For non image case, the dimensions are "
          "in the form of (N x C x D1 x D2 ... Dn), or (N x D1 X D2 ... Dn x C) "
          "where N is the batch size.",
          "T")
      .Input(
          1,
          "x_scale",
          "Scale of quantized input 'X'. It must be a scalar.",
          "tensor(float)")
      .Input(
          2,
          "x_zero_point",
          "Zero point tensor for input 'X'. It must be a scalar.",
          "T")
      .Input(
          3,
          "y_scale",
          "Scale of quantized output 'Y'. It must be a scalar.",
          "tensor(float)")
      .Input(
          4,
          "y_zero_point",
          "Zero point tensor for output 'Y'. It must be a scalar.",
          "T")
      .Output(
          0,
          "Y",
          "Output data tensor from pooling across the input "
          "tensor. The output tensor has the same rank as the input. "
          "with the N and C value keep it value, while the other"
          "dimensions are all 1.",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to singed/unsigned int8 tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        int64_t channel_last = getAttribute(ctx, "channels_last", 0);

        // needs at least one input with shape.
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        auto input_shape = ctx.getInputType(0)->tensor_type().shape();
        if (input_shape.dim_size() < 2) {
          return;
        }

        // (N, C, 1, 1, ..., 1) or (N, 1, 1, ..., 1, C)
        auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        output_shape->CopyFrom(input_shape);
        int image_dim_index = (channel_last ? 1 : 2);
        for (auto n_hw_dims = input_shape.dim_size() - 2; n_hw_dims > 0; --n_hw_dims) {
          output_shape->mutable_dim(image_dim_index)->clear_dim_param();
          output_shape->mutable_dim(image_dim_index)->set_dim_value(1);
          ++image_dim_index;
        }
      });

  const char* QLinearAveragePoolDoc_ver1 = R"DOC(
 QLinearAveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```

The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).

Input and output scales and zero points are used to convert the output to a new quantization range.
Output = Dequantize(Input) -> AveragePool on fp32 data -> Quantize(output)
)DOC";

  static const char* contrib_ops_pads_doc =
      "Padding for the beginning and ending along each spatial axis, it can take any value greater "
      "than or equal to 0. The value represent the number of pixels added to the beginning "
      "and end part of the corresponding axis. `pads` format should be as follow "
      "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
      "added at the beginning of axis `i` and xi_end, the number of pixels added at "
      "the end of axis `i`. This attribute cannot be used simultaneously with "
      "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
  static const char* contrib_ops_auto_pad_doc =
      "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
      "default value is NOTSET, which means explicit padding is used. "
      "SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input."
      "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
      "beginning for SAME_LOWER. VALID mean no padding.";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearAveragePool)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearAveragePoolDoc_ver1)
      .Attr(
          "count_include_pad",
          "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "kernel_shape",
          "The size of the kernel along each axis.",
          AttributeProto::INTS)
      .Attr(
          "strides",
          "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
          AttributeProto::INTS,
          OPTIONAL_VALUE)
      .Attr(
          "auto_pad",
          contrib_ops_auto_pad_doc,
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr("pads", contrib_ops_pads_doc, AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr(
          "ceil_mode",
          "Whether to use ceil or floor (default) to compute the output shape.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr("channels_last", "Works on NHWC layout or not? Default not.", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(
          0,
          "X",
          "Input data tensor from the previous operator; "
          "dimensions for image case are (N x C x H x W), "
          "where N is the batch size, C is the number of "
          "channels, and H and W are the height and the "
          "width of the data. For non image case, the "
          "dimensions are in the form of "
          "(N x C x D1 x D2 ... Dn), where N is the batch "
          "size. Optionally, if dimension denotation is "
          "in effect, the operation expects the input "
          "data tensor to arrive with the dimension denotation "
          "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
          "T")
      .Input(
          1,
          "x_scale",
          "Input scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          2,
          "x_zero_point",
          "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(
          3,
          "y_scale",
          "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          4,
          "y_zero_point",
          "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(
          0,
          "Y",
          "Output data tensor from average or max pooling across "
          "the input tensor. Dimensions will vary based "
          "on various kernel, stride, and pad sizes. Floor value of "
          "the dimension is used",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

        auto data_type = ctx.getInputType(0);
        if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        // validate scale and zero points
        ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

        if (getAttribute(ctx, "channels_last", 0) == 0) {
          ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 5);
        } else {
          convPoolShapeInferenceNhwc(ctx, false, true, 0, 5);
        }
      });
}

}  // namespace contrib
}  // namespace onnxruntime
