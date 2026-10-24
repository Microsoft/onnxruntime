// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/shape_inference_functions.h"
#include "core/graph/onnx_protobuf.h"

// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif

namespace ONNX_NAMESPACE {
void RNNShapeInference(InferenceContext& ctx);
void convTransposeShapeInference(InferenceContext& ctx);
void convPoolShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, bool use_dilation, bool require_kernel_shape,
                            int input1Idx, int input2Idx);
namespace defs::math::utils {
void MatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx);
}

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;
#ifndef NDEBUG
using ONNX_NAMESPACE::DbgOperatorSetTracker;
#endif

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index,
                                       ::google::protobuf::int32 expectedType,
                                       QuantParamTensorType expectedScalar, int expectedTensorSize) {
  if (ctx.getNumInputs() > static_cast<size_t>(index)) {
    auto data_type = ctx.getInputType(index);
    if (nullptr == data_type) {
      fail_type_inference("Input data type does not match the expected data type");
    }
    if (data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        data_type->tensor_type().elem_type() != expectedType) {
      fail_type_inference("Input data type does not match the expected data type. Current data type is ",
                          data_type->tensor_type().elem_type());
    }
  }

  if (hasInputShape(ctx, index)) {
    ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(index)->tensor_type().shape();
    if (expectedScalar == QuantParamTensorType::Scalar) {
      if (shape.dim_size() != 0) {
        fail_type_inference("Scale and Zero-point must be a scalar");
      }
    } else {
      if (expectedScalar == QuantParamTensorType::Both && shape.dim_size() == 0) {
        return;
      }
      if (shape.dim_size() != 1) {
        fail_type_inference("Scale and Zero-point must be of rank 1");
      }

      if (shape.dim((int)0).has_dim_value() && shape.dim((int)0).dim_value() != expectedTensorSize) {
        fail_type_inference(
            "Scale and Zero-point must be of rank 1 and the number of elements should be equal to the number of rows "
            "of the corresponding input.");
      }
    }
  }
}

std::function<void(OpSchema&)> QLinearMathDocGenerator(const char* name, const char* additionalDocumentation) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} on 8 bit data types (with Numpy-style broadcasting support).

{additionalDocumentation}
)DOC";
    ONNX_NAMESPACE::ReplaceAll(doc, "{name}", name);
    ONNX_NAMESPACE::ReplaceAll(doc, "{additionalDocumentation}", additionalDocumentation);
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(1, "A_scale", "Input A's scale. It's a scalar, which means a per-tensor/layer quantization.",
                 "tensor(float)");
    schema.Input(2, "A_zero_point",
                 "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
                 "per-tensor/layer quantization.",
                 "T", OpSchema::Optional);
    schema.Input(3, "B", "Second operand.", "T");
    schema.Input(4, "B_scale", "Input B's scale. It's a scalar, which means a per-tensor/layer quantization.",
                 "tensor(float)");
    schema.Input(5, "B_zero_point",
                 "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
                 "per-tensor/layer quantization.",
                 "T", OpSchema::Optional);
    schema.Input(6, "C_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
                 "tensor(float)");
    schema.Input(7, "C_zero_point",
                 "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
                 "per-tensor/layer quantization.",
                 "T", OpSchema::Optional);
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                          "Constrain input and output types to 8 bit signed and unsigned tensors.");
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      auto a_type = ctx.getInputType(0);
      auto b_type = ctx.getInputType(3);

      if (nullptr == a_type || nullptr == b_type || a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
          b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
      }

      // validate scale and zero points
      ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
      ValidateTypeAndShapeForScaleAndZP(ctx, 2, a_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
      ValidateTypeAndShapeForScaleAndZP(ctx, 4, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
      ValidateTypeAndShapeForScaleAndZP(ctx, 5, b_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
      ValidateTypeAndShapeForScaleAndZP(ctx, 6, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
      ValidateTypeAndShapeForScaleAndZP(ctx, 7, a_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);

      if (hasInputShape(ctx, 0) && hasInputShape(ctx, 3))
        bidirectionalBroadcastShapeInference(ctx.getInputType(0)->tensor_type().shape(),
                                             ctx.getInputType(3)->tensor_type().shape(),
                                             *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

static const char* QuantizeLinear_ver1_doc = R"DOC(
The linear quantization operator. It consumes a full precision data, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, [-128, 127] if it's int8,
[0, 65,535] if it's uint16, and [-32,768, 32,767] if it's int16. For (x / y_scale), it's rounding to nearest ties to even.
Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QuantizeLinear, 1,
    OpSchema()
        .Attr("axis",
              "The axis along which same quantization parameters are applied. It's optional."
              "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be "
              "scalars."
              "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D "
              "tensors.",
              AttributeProto::INT, false)
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(1, "y_scale",
               "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer "
               "quantization, or a 1-D tensor for per-axis quantization.",
               "T1")
        .Input(2, "y_zero_point",
               "Zero point for doing quantization to get 'y'. Shape must match y_scale. Default is "
               "uint8 with zero point of 0 if it's not specified.",
               "T2", OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .TypeConstraint("T1", {"tensor(float16)", "tensor(float)"}, "Constrain 'x', 'y_scale' to float tensors.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)", "tensor(int4)",
                               "tensor(uint4)"},
                        "Constrain 'y_zero_point' and 'y' to 8-bit and 16-bit integer tensors.")
        .SetDoc(QuantizeLinear_ver1_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getNumInputs() == 3 && ctx.getInputType(2) != nullptr) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::UINT8);
          }

          if (!hasInputShape(ctx, 0)) return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver1_doc = R"DOC(
The linear dequantization operator. It consumes quantized data, a scale, and an optional zero point.
The dequantization formula is y = (x - x_zero_point) * x_scale. Scale and zero point are scalar
or per-axis vectors. FP8 E4M3FN inputs can be dequantized directly to BF16 with FP32 scales.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(DequantizeLinear, 1,
                            OpSchema()
                                .Attr("output_dtype",
                                      "Optional output element type. When omitted, the scale "
                                      "element type determines the output type.",
                                      AttributeProto::INT, false)
                                .Attr("axis",
                                      "The axis along which same quantization parameters are applied. It's optional."
                                      "If it's not specified, it means per-tensor quantization and input 'x_scale' and "
                                      "'x_zero_point' must be scalars."
                                      "If it's specified, it means per 'axis' quantization and input 'x_scale' and "
                                      "'x_zero_point' must be 1-D tensors.",
                                      AttributeProto::INT, false)
                                .Input(0, "x", "N-D quantized Input tensor to be de-quantized.", "T1")
                                .Input(1, "x_scale",
                                       "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer "
                                       "dequantization, or a 1-D tensor for per-axis dequantization.",
                                       "TS")
                                .Input(2, "x_zero_point",
                                       "Zero point for input 'x'. Shape must match x_scale. It's optional. "
                                       "Zero point is 0 when it's not specified.",
                                       "T1", OpSchema::Optional)
                                .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.",
                                        "T2")
                                .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)", "tensor(int32)", "tensor(int4)", "tensor(uint4)", "tensor(float8e4m3fn)"},
                                                "Constrain 'x' and 'x_zero_point' to 8-bit integer tensors, "
                                                "16-bit integer tensors, 32-bit signed integer tensors, or "
                                                "FP8 E4M3FN tensors.")
                                .TypeConstraint("T2", {"tensor(float16)", "tensor(float)", "tensor(bfloat16)"},
                                                "Constrain the output element type.")
                                .TypeConstraint("TS", {"tensor(float16)", "tensor(float)", "tensor(bfloat16)"},
                                                "Constrain the scale element type.")
                                .SetDoc(DequantizeLinear_ver1_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  const int64_t output_dtype =
                                      getAttribute(ctx, "output_dtype", static_cast<int64_t>(0));
                                  if (output_dtype != 0) {
                                    if (output_dtype != ONNX_NAMESPACE::TensorProto::FLOAT &&
                                        output_dtype != ONNX_NAMESPACE::TensorProto::FLOAT16 &&
                                        output_dtype != ONNX_NAMESPACE::TensorProto::BFLOAT16) {
                                      fail_shape_inference(
                                          "DequantizeLinear: output_dtype must be float, float16, or bfloat16");
                                    }
                                    updateOutputElemType(ctx, 0, static_cast<int32_t>(output_dtype));
                                  } else if (ctx.getInputType(1) != nullptr) {
                                    propagateElemTypeFromInputToOutput(ctx, 1, 0);
                                  }

                                  if (!hasInputShape(ctx, 0)) return;

                                  auto& input_shape = getInputShape(ctx, 0);
                                  updateOutputShape(ctx, 0, input_shape);
                                }));

static const char* QuantizeBFP_ver1_doc = R"DOC(
The BFP quantization operator. It consumes a full precision tensor and computes an BFP tensor.
More documentation on the BFP format can be found in this paper: https://www.microsoft.com/en-us/research/publication/pushing-the-limits-of-narrow-precision-inferencing-at-cloud-scale-with-microsoft-floating-point/)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QuantizeBFP, 1,
    OpSchema()
        .Attr("bfp_type", "The type of BFP - must match with the BFPType enum", AttributeProto::INT)
        .Attr("block_dim",
              "Each bounding box spans this dimension."
              "Typically, the block dimension corresponds to the reduction dimension of the matrix multipication that "
              "consumes the output of this operator."
              "For example, for a 2D matrix multiplication A@W, QuantizeBFP(A) would use block_dim 1 and "
              "QuantizeBFP(W) would use block_dim 0."
              "The default is the last dimension.",
              AttributeProto::INT, static_cast<int64_t>(-1))
        .Input(0, "x", "N-D full precision input tensor to be quantized.", "T1")
        .Output(0, "y", "1-D, contiguous BFP data", "T2")
        .Output(1, "shape", "Shape of x", "T3")
        .Output(2, "strides", "Strides of x", "T3")
        .TypeConstraint("T1", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain the input to float and bfloat.")
        .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain y to uint8.")
        .TypeConstraint("T3", {"tensor(int64)"}, "Constrain shape and strides to uint64.")
        .SetDoc(QuantizeBFP_ver1_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // Shape of raw, quantized tensor is specific to the hardware; for example, different hardware pad the data in
          // different ways. So do not set the shape.
          ONNX_NAMESPACE::setTensorElementType(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8,
                                               ONNX_NAMESPACE::TypeProto::kTensorType, *ctx.getOutputType(0));
          ONNX_NAMESPACE::setTensorElementType(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                                               ONNX_NAMESPACE::TypeProto::kTensorType, *ctx.getOutputType(1));
          ONNX_NAMESPACE::setTensorElementType(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                                               ONNX_NAMESPACE::TypeProto::kTensorType, *ctx.getOutputType(2));

          if (!hasInputShape(ctx, 0)) return;

          auto& input_shape = getInputShape(ctx, 0);
          auto num_dims = input_shape.dim_size();
          ONNX_NAMESPACE::TensorShapeProto::Dimension num_dims_proto;
          num_dims_proto.set_dim_value(num_dims);
          updateOutputShape(ctx, 1, {num_dims_proto});
          updateOutputShape(ctx, 2, {num_dims_proto});
        }));

static const char* DequantizeBFP_ver1_doc = R"DOC(
The BFP dequantization operator.
It consumes the raw BFP data and some metadata such as the shape and strides of the original tensor and computes the dequantized tensor.
More documentation on the BFP format can be found in this paper: https://www.microsoft.com/en-us/research/publication/pushing-the-limits-of-narrow-precision-inferencing-at-cloud-scale-with-microsoft-floating-point/)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DequantizeBFP, 1,
    OpSchema()
        .Attr("bfp_type", "The type of BFP - must match with the BFPType enum", AttributeProto::INT)
        .Attr("block_dim",
              "Each bounding box spans this dimension."
              "Typically, the block dimension corresponds to the reduction dimension of the matrix multipication that "
              "consumes the output of this operator."
              "For example, for a 2D matrix multiplication A@W, QuantizeBFP(A) would use block_dim 1 and "
              "QuantizeBFP(W) would use block_dim 0."
              "The default is the last dimension.",
              AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("dtype", "The datatype to dequantize to.", AttributeProto::INT,
              static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT))  // default
        .Input(0, "x", "1-D, contiguous, raw, BFP data to be de-quantized.", "T1")
        .Input(1, "shape", "shape of the original tensor.", "T2")
        .Input(2, "strides", "strides of the original tensor.", "T2")
        .Output(0, "y", "de-quantized tensor.", "T3")
        .TypeConstraint("T1", {"tensor(uint8)"}, "Constrain the input to uint8.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain shape and strides to uint64.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain y to float and bfloat16.")
        .SetDoc(DequantizeBFP_ver1_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 1u) {
              fail_shape_inference("Shape of quantized tensor must be 1D.")
            }
          }

          auto y_type = ctx.getOutputType(0);
          auto dtype_proto = ctx.getAttribute("dtype");
          y_type->mutable_tensor_type()->set_elem_type(static_cast<int>(dtype_proto->i()));
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(ReduceSumInteger, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.)DOC")
                                .Input(0, "data", "An input tensor.", "T1")
                                .Output(0, "reduced", "Reduced output tensor.", "T2")
                                .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"},
                                                "Constrain input type to 8-bit integer tensor.")
                                .TypeConstraint("T2", {"tensor(int32)", "tensor(uint32)"},
                                                "Constrain output data type to 32-bit integer tensor."
                                                "T2 must be tensor(uint32) when T1 is tensor(uint8),"
                                                "or must be tensor(int32) when T1 is tensor(int8).")
                                .Attr("axes",
                                      "A list of integers, along which to reduce. The default is to reduce over all "
                                      "the dimensions of the input tensor.",
                                      AttributeProto::INTS)
                                .Attr("keepdims",
                                      "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                                      AttributeProto::INT));

ONNX_MS_OPERATOR_SET_SCHEMA(
    MulInteger, 1,
    OpSchema()
        .SetDoc(R"DOC(Performs element-wise binary quantized multiplication (with Numpy-style broadcasting support).
"This operator supports **multidirectional (i.e., Numpy-style) broadcasting**"
The output of this op is the int32 accumulated result of the mul operation

```
C (int32) = (A - A_zero_point) * (B - B_zero_point)
```

)DOC")
        .Input(0, "A", "First operand.", "T")
        .Input(1, "A_zero_point",
               "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Input(2, "B", "Second operand.", "T")
        .Input(3, "B_zero_point",
               "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Output(0, "C", "Constrain output to 32 bit tensor", "T1")
        .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input types to 8 bit signed and unsigned tensors.")
        .TypeConstraint("T1", {"tensor(int32)"}, "Constrain output types to 32 bit tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto c_type = ctx.getOutputType(0);
          c_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(3);
          if (nullptr == a_type || nullptr == b_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          ValidateTypeAndShapeForScaleAndZP(ctx, 1, a_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 3, b_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);

          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
            bidirectionalBroadcastShapeInference(ctx.getInputType(0)->tensor_type().shape(),
                                                 ctx.getInputType(2)->tensor_type().shape(),
                                                 *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
          }
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    DynamicQuantizeMatMul, 1,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T1")
        .Input(1, "B", "N-dimensional matrix B", "T2")
        .Input(2, "b_scale",
               "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "T1")
        .Input(3, "b_zero_point",
               "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D "
               "tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "T2", OpSchema::Optional)
        .Input(4, "bias", "1D input tensor, whose dimension is same as B's last dimension", "T1", OpSchema::Optional)
        .Output(0, "Y", "Matrix multiply results from A * B", "T1")
        .TypeConstraint("T1", {"tensor(float)"}, "Constrain input A, b_scale and output Y data type as float tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input B data type to 8-bit integer tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          ONNX_NAMESPACE::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    MatMulIntegerToFloat, 1,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T1")
        .Input(1, "B", "N-dimensional matrix B", "T2")
        .Input(2, "a_scale",
               "Scale of quantized input 'A'. It could be a scalar or a 1-D tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'A'.",
               "T3")
        .Input(3, "b_scale",
               "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "T3")
        .Input(4, "a_zero_point",
               "Zero point tensor for input 'A'. It's optional and default value is 0.  It could be a scalar or a 1-D "
               "tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'A'.",
               "T1", OpSchema::Optional)
        .Input(5, "b_zero_point",
               "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D "
               "tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "T2", OpSchema::Optional)
        .Input(6, "bias", "1D input tensor, whose dimension is same as B's last dimension", "T3", OpSchema::Optional)
        .Output(0, "Y", "Matrix multiply results from A * B", "T3")
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input A data type to 8-bit integer tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input B data type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(float16)"},
                        "Constrain input a_scale, b_scale and output Y data type as float tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
          ONNX_NAMESPACE::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearAdd, 1,
    OpSchema().FillUsing(QLinearMathDocGenerator(
        "addition", "C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point")));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearMul, 1,
    OpSchema().FillUsing(QLinearMathDocGenerator(
        "multiplication",
        "C = ((A - A_zero_point) * (B - B_zero_point)) * (A_scale * B_scale)/C_scale + C_zero_point")));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearReduceMean, 1,
    OpSchema()
        .SetDoc(R"DOC(
Computes the mean of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.
Input and Output scales and zero points are used to requantize the output in a new range.
This helps to improve accuracy as after ReduceMean operation the range of the output is expected to decrease.

```
"Output = Dequantize(Input) -> ReduceMean on fp32 data -> Quantize(output)",

```
)DOC")
        .Input(0, "data", "An input tensor.", "T")
        .Input(1, "data_scale", "Input scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(2, "data_zero_point",
               "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Input(3, "reduced_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(4, "reduced_zero_point",
               "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Output(0, "reduced", "Reduced output tensor.", "T")
        .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input types to 8 bit signed and unsigned tensors.")
        .Attr("axes",
              "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the "
              "input tensor.",
              AttributeProto::INTS)
        .Attr("keepdims", "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
              AttributeProto::INT)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto data_type = ctx.getInputType(0);
          if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          // validate scale and zero points
          ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);

          int64_t keep_dims = 1;
          auto attr_proto = ctx.getAttribute("keepdims");
          if (attr_proto) {
            keep_dims = attr_proto->i();
          }

          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          int64_t input_ndim = input_shape.dim_size();
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getAttribute("axes");
          if (axes_proto) axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

          for (size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
              fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
            }
            if (axes[i] < 0) axes[i] += input_ndim;
          }
          // do we need to handle negative axis?
          for (int i = 0; i < input_ndim; ++i) {
            // axes empty means reduce all dim
            if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
              auto dim = output_shape->add_dim();
              dim->CopyFrom(input_shape.dim(i));
            } else {
              if (keep_dims == 1) {
                auto dim = output_shape->add_dim();
                dim->set_dim_value(1);
              }
            }
          }
        }));

const char* QLinearLeakyReluDoc_ver1 = R"DOC(
QLinearLeakyRelu takes quantized input data (Tensor), an argument alpha, and quantize parameter for output,
and produces one output data (Tensor<T>) where the function `f(x) = quantize(alpha * dequantize(x)) for dequantize(x) < 0`,
`f(x) = quantize(dequantize(x)) for dequantize(x) >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearLeakyRelu, 1,
    OpSchema()
        .SetDoc(QLinearLeakyReluDoc_ver1)
        .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
        .Input(0, "X", "Input tensor", "T")
        .Input(1, "X_scale", "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(2, "X_zero_point",
               "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Input(3, "Y_scale", "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(4, "Y_zero_point",
               "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input and output types to 8 bit tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

const char* QLinearSigmoidDoc_ver1 = R"DOC(
QLinearSigmoid takes quantized input data (Tensor), and quantize parameter for output, and produces one output data
(Tensor<T>) where the function `f(x) = quantize(Sigmoid(dequantize(x)))`, is applied to the data tensor elementwise.
Wwhere the function `Sigmoid(x) = 1 / (1 + exp(-x))` )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearSigmoid, 1,
    OpSchema()
        .SetDoc(QLinearSigmoidDoc_ver1)
        .Input(0, "X", "Input tensor", "T")
        .Input(1, "X_scale", "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(2, "X_zero_point",
               "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Input(3, "Y_scale", "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
               "tensor(float)")
        .Input(4, "Y_zero_point",
               "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a "
               "per-tensor/layer quantization.",
               "T", OpSchema::Optional)
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input and output types to 8 bit tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearSoftmax, 1,
    OpSchema()
        .SetDoc(R"DOC(
QLinearSoftmax computes the normalized exponential values for the given input:
Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
The input does not need to explicitly be a 2D vector. The "axis" attribute
indicates the dimension along which QLinearSoftmax will be performed for onnx v.13+.
or the dimension coerced to NxD Matrix for onnx v.12-.
The output tensor has the same shape.
)DOC")
        .Attr("axis",
              "apply softmax to elements for dimensions axis,"
              "or all dims along with axis according to op-version",
              AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("opset", "opset version of corresponding SoftMax.", AttributeProto::INT)
        .Input(0, "X", "The input tensor", "T")
        .Input(1, "X_scale", "Scale of quantized input 'X'. It must be a scalar.", "tensor(float)")
        .Input(2, "x_zero_point",
               "Zero point tensor for input 'X'."
               "It must be a scalar.",
               "T", OpSchema::Optional)
        .Input(3, "y_scale", "Scale of quantized output 'Y'. It must be a scalar.", "tensor(float)")
        .Input(4, "y_zero_point",
               "Zero point tensor for output 'Y'. "
               "It must be a scalar.",
               "T")
        .Output(0, "Y",
                "Output data tensor from pooling across the input "
                "tensor. The output tensor has the same rank as the input. ",
                "T")
        .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input and output types to signed/unsigned int8 tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference starts
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          // Validate the value of 'axis'
          const ONNX_NAMESPACE::TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          int r = input_shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
          if (axis < -r || axis >= r) {
            fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
          }

          // Shape inference
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLSTM, 1,
    OpSchema()
        .Attr("direction",
              "Specify if the RNN is forward, reverse, or bidirectional. "
              "Must be one of forward (default), reverse, or bidirectional.",
              AttributeProto::STRING, std::string("forward"))
        .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("activation_alpha",
              "Optional scaling values used by some activation functions. The values "
              "are consumed in the order of activation functions, for example (f, g, h) "
              "in LSTM. Default values are the same as of corresponding ONNX operators."
              "For example with LeakyRelu, the default alpha is 0.01.",
              AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("activation_beta",
              "Optional scaling values used by some activation functions. The values "
              "are consumed in the order of activation functions, for example (f, g, h) "
              "in LSTM. Default values are the same as of corresponding ONNX operators.",
              AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("clip",
              "Cell clip threshold. Clipping bounds the elements of a tensor "
              "in the range of [-threshold, +threshold] and is applied to the input "
              "of activations. No clip if not specified.",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("activations",
              "A list of 3 (or 6 if bidirectional) activation functions "
              "for input, output, forget, cell, and hidden. The activation functions must "
              "be one of the activation functions specified above. Optional: See the equations "
              "for default if not specified.",
              AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr("input_forget", "Couple the input and forget gates if 1.", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "X",
               "The input sequences packed (and potentially padded) into one 3-D "
               "tensor with the shape of `[seq_length, batch_size, input_size]`.",
               "T")
        .Input(1, "W",
               "The weight tensor for the gates. Concatenation of `W[iofc]` and "
               "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
               "`[num_directions, input_size, 4*hidden_size]`.",
               "T2")
        .Input(2, "R",
               "The recurrence weight tensor. Concatenation of `R[iofc]` and "
               "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
               "`[num_directions, hidden_size, 4*hidden_size]`.",
               "T2")
        .Input(3, "B",
               "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
               "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
               "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
               "specified - assumed to be 0.",
               "T", OpSchema::Optional)
        .Input(4, "sequence_lens",
               "Optional tensor specifying lengths of the sequences in a batch. "
               "If not specified - assumed all sequences in the batch to have "
               "length `seq_length`. It has shape `[batch_size]`.",
               "T1", OpSchema::Optional)
        .Input(5, "initial_h",
               "Optional initial value of the hidden. If not specified - assumed "
               "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
               "T", OpSchema::Optional)
        .Input(6, "initial_c",
               "Optional initial value of the cell. If not specified - assumed "
               "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
               "T", OpSchema::Optional)
        .Input(7, "P",
               "The weight tensor for peepholes. Concatenation of `P[iof]` and "
               "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
               "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
               "assumed to be 0.",
               "T", OpSchema::Optional)
        .Input(8, "W_scale",
               "W's scale. Its size is [num_directions] for per-tensor/layer quantization, "
               "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
               "T")
        .Input(9, "W_zero_point",
               "W's zero point. Its size is [num_directions] for per-tensor/layer quantization, "
               "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
               "T2")
        .Input(10, "R_scale",
               "R's scale. Its size is [num_directions] for per-tensor/layer quantization, "
               "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
               "T")
        .Input(11, "R_zero_point",
               "R's zero point. Its size is [num_directions] for per-tensor/layer quantization, "
               "or [num_directions, 4*hidden_size] for per-channel quantization on the axis input_size.",
               "T2")
        .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional, true, 1, OpSchema::Differentiable)
        .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.",
                "T", OpSchema::Optional, true, 1, OpSchema::Differentiable)
        .Output(2, "Y_c",
                "The last output value of the cell. It has shape "
                "`[num_directions, batch_size, hidden_size]`.",
                "T", OpSchema::Optional, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.")
        .TypeConstraint("T2", {"tensor(uint8)", "tensor(int8)"}, "Constrain weights types to 8 bit tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::RNNShapeInference));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearConcat, 1,
    OpSchema()
        .Attr("axis", "Which axis to concat on", AttributeProto::INT)
        .SetDoc("Concatenate a list of tensors into a single tensor."
                "All input tensors must have the same shape, except "
                "for the dimension size of the axis to concatenate on.")
        .Input(0, "Y_scale", "Y's scale.", "TF")
        .Input(1, "Y_zero_point", "Y's zero point.", "T8")
        .Input(2, "inputs", "List of tensors/scale/zero_point for concatenation", "TV", OpSchema::Variadic, false)
        .Output(0, "Y", "Concatenated tensor", "T8")
        .TypeConstraint("T8", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input and output types to 8 bit signed and unsigned tensors.")
        .TypeConstraint("TF", {"tensor(float)"}, "Constrain scale types to any float tensor type.")
        .TypeConstraint("TV", {"tensor(uint8)", "tensor(int8)", "tensor(float)"},
                        "Sequence of (Tensor, Scale, ZeroPoint) tuples. The type is sequence of (T8, TF, T8).")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto numInputs = ctx.getNumInputs();
          if (numInputs < 5 || (numInputs - 2) % 3 != 0 || !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }
          auto rank = ctx.getInputType(2)->tensor_type().shape().dim_size();

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axisAttr->i());
          if (rank <= axis || axis < -rank) {
            fail_shape_inference("axis must be in [-rank, rank)");
          }
          if (axis < 0) {
            axis += rank;
          }

          bool all_lengths_known = true;
          int total_length = 0;

          auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int64_t i = 0; i < rank; ++i) {
            output_shape->add_dim();
          }

          for (size_t i = 2; i < numInputs; i += 3) {
            const auto& shape = ctx.getInputType(i)->tensor_type().shape();
            if (shape.dim_size() != rank) {
              fail_shape_inference("All inputs to Concat must have same rank");
            }
            for (int j = 0; j < rank; j++) {
              if (j == axis) {
                if (shape.dim(j).has_dim_value()) {
                  total_length += static_cast<int>(shape.dim(j).dim_value());
                } else {
                  all_lengths_known = false;
                }
              } else {
                auto& output_dim = *output_shape->mutable_dim(j);
                const auto& input_dim = shape.dim(j);
                mergeInDimensionInfo(input_dim, output_dim, j);
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        }));

  ONNX_MS_OPERATOR_SET_SCHEMA(QLinearWhere, 1, OpSchema()
    .SetDoc("Return elements, either from X or Y, depending on condition.")
      .Input(0, "condition", " When True (nonzero), yield x, otherwise yield y", "B")
      .Input(1, "X", "Y's zero point.", "T")
      .Input(2, "x_scale", "X's scale.", "TF")
      .Input(3, "x_zero_point", "X's zero point.", "T")
      .Input(4, "Y", "Y's zero point.", "T")
      .Input(5, "y_scale", "Y's scale.", "TF")
      .Input(6, "y_zero_point", "Y's zero point.", "T")
      .Input(7, "z_scale", "Z's scale.", "TF")
      .Input(8, "z_zero_point", "Z's zero point.", "T")
      .Output(0, "Z", "Tensor of shape equal to the broadcasted shape of condition, X, and Y", "T")
      .TypeConstraint(
        "B",
        {"tensor(bool)"},
        "Constrain input and output types to 8 bit signed and unsigned tensors.")
      .TypeConstraint(
        "TF",
        {"tensor(float)"},
        "Constrain scale types to any float tensor type.")
      .TypeConstraint(
        "T",
        {"tensor(uint8)", "tensor(int8)"},
        "Constrain input and output types to 8 bit signed and unsigned tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 9)) {
    std::vector<const onnx::TensorShapeProto*> shapes;
    shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(4)->tensor_type().shape());
    multidirectionalBroadcastShapeInference(
        shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QGemm, 1,
    OpSchema()
        .SetDoc("Quantized Gemm")
        .Input(0, "A",
               "Input tensor A. "
               "The shape of A should be (M, K) if transA is 0, "
               "or (K, M) if transA is non-zero.",
               "TA")
        .Input(1, "a_scale",
               "Scale of quantized input 'A'. "
               "It is a scalar,which means a per-tensor quantization.",
               "T")
        .Input(2, "a_zero_point", "Zero point tensor for input 'A'. It is a scalar.", "TA")
        .Input(3, "B",
               "Input tensor B. "
               "The shape of B should be (K, N) if transB is 0, "
               "or (N, K) if transB is non-zero.",
               "TB")
        .Input(4, "b_scale",
               "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "T")
        .Input(5, "b_zero_point",
               "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D "
               "tensor, "
               "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
               "of elements should be equal to the number of columns of input 'B'.",
               "TB")
        .Input(6, "C",
               "Optional input tensor C. "
               "If not specified, the computation is done as if C is a scalar 0. "
               "The shape of C should be unidirectional broadcastable to (M, N). "
               "Its type is int32_t and must be quantized with zero_point = 0 and "
               "scale = alpha / beta * a_scale * b_scale.",
               "TC", OpSchema::Optional)
        .Input(7, "y_scale",
               "Scale of output 'Y'. It is a scalar, which means a per-tensor quantization. "
               "It is optional. The output is full precision(float32) if it is not provided. "
               "Or the output is quantized.",
               "T", OpSchema::Optional)
        .Input(8, "y_zero_point",
               "Zero point tensor for output 'Y'. It is a scalar, which means a per-tensor quantization. "
               "It is optional. The output is full precision(float32) if it is not provided. "
               "Or the output is quantized.",
               "TYZ", OpSchema::Optional)
        .Output(0, "Y", "Output tensor of shape (M, N).", "TY")
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("alpha", "Scalar multiplier for the product of input tensors A * B.", AttributeProto::FLOAT, 1.0f)
        .TypeConstraint("T", {"tensor(float)"}, "Constrain scale types to float tensors.")
        .TypeConstraint("TA", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input A and its zero point types to 8 bit tensors.")
        .TypeConstraint("TB", {"tensor(uint8)", "tensor(int8)"},
                        "Constrain input B and its zero point types to 8 bit tensors.")
        .TypeConstraint("TC", {"tensor(int32)"}, "Constrain input C to 32 bit integer tensors.")
        .TypeConstraint("TYZ", {"tensor(uint8)", "tensor(int8)"}, "Constrain output zero point types to 8 bit tensors.")
        .TypeConstraint("TY", {"tensor(float)", "tensor(uint8)", "tensor(int8)"},
                        "Constrain output type to float32 or 8 bit tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumInputs() == 9 && nullptr != ctx.getInputType(8)) {
            propagateElemTypeFromInputToOutput(ctx, 8, 0);
          } else {
            updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::FLOAT);
          }

          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 3)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 3);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
          }
        }));
ONNX_MS_OPERATOR_SET_SCHEMA(
    QAttention, 1,
    OpSchema()
        .SetDoc("Quantization of Multi-Head Self Attention.")
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("unidirectional", "Whether every token can only attend to previous tokens. Default value is 0.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("do_rotary", "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its shape is "
              "(2, batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, input_hidden_size)", "T1")
        .Input(1, "weight",
               "2D input tensor with shape (input_hidden_size, 3 * hidden_size), hidden_size = num_heads * head_size",
               "T2")
        .Input(2, "bias", "1D input tensor with shape (3 * hidden_size)", "T3")
        .Input(3, "input_scale",
               "scale of quantized input tensor. It's a scalar, which means a per-tensor/layer quantization.", "T3")
        .Input(4, "weight_scale",
               "scale of weight scale. It's a scalar or a 1D tensor, which means a per-tensor/per-column quantization."
               "Its size should be 3 * hidden_size if it is per-column quantization",
               "T3")
        .Input(5, "mask_index", "Attention mask index with shape (batch_size)", "T4", OpSchema::Optional)
        .Input(6, "input_zero_point",
               "zero point of quantized input tensor.It's a scalar, which means a per-tensor/layer quantization.", "T1",
               OpSchema::Optional)
        .Input(7, "weight_zero_point",
               "zero point of quantized weight tensor. It's a scalar or a 1D tensor, which means a "
               "per-tensor/per-column quantization."
               "Its size should be 3 * hidden_size if it is per-column quantization",
               "T2", OpSchema::Optional)
        .Input(8, "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).",
               "T3", OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T3")
        .Output(1, "present",
                "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + "
                "sequence_length, head_size)",
                "T3", OpSchema::Optional)
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("T4", {"tensor(int32)"}, "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          constexpr int past_input_index = 8;

          AttentionTypeAndShapeInference(ctx, past_input_index);
        }));

constexpr const char* QEmbedLayerNormalization_ver1_doc = R"DOC(
QEmbedLayerNormalization is the quantized fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors. The input_ids
and segment_ids remain int32. All embeddings, gamma, and beta tensors are converted to int8/uint8. The last input mask is optional.
segment_ids, segment_embedding, segment_embedding_scale, and segment_embedding_zero_point must either all be provided or all
be omitted.
If mask is provided, mask index (that is position of first 0 in mask, or number of words will be calculated.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QEmbedLayerNormalization, 1,
    OpSchema()
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetDoc(QEmbedLayerNormalization_ver1_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT,
              kDefaultEmbedLayerNormEpsilon)
        .Input(0, "input_ids", "2D words IDs with shape (batch_size, sequence_length)", "T1")
        .Input(1, "segment_ids", "2D segment IDs with shape (batch_size, sequence_length). Part of the all-or-none segment input group.", "T1", OpSchema::Optional)
        .Input(2, "word_embedding_quant", "2D with shape (,hidden_size)", "T2")
        .Input(3, "position_embedding_quant", "2D with shape (, hidden_size)", "T2")
        .Input(4, "segment_embedding", "2D with shape (, hidden_size). Part of the all-or-none segment input group.", "T2", OpSchema::Optional)
        .Input(5, "gamma_quant", "1D gamma tensor for layer normalization with shape (hidden_size)", "T2")
        .Input(6, "beta_quant", "1D beta tensor for layer normalization  with shape (hidden_size)", "T2")
        .Input(7, "mask", "Mask", "T1", OpSchema::Optional)
        .Input(8, "word_embedding_scale", "Scale for word embeddings", "T")
        .Input(9, "position_embedding_scale", "Scale for position embeddings", "T")
        .Input(10, "segment_embedding_scale", "Scale for segment embeddings. Part of the all-or-none segment input group.", "T", OpSchema::Optional)
        .Input(11, "gamma_scale", "Scale for 1D gamma tensor", "T")
        .Input(12, "beta_scale", "Scale for 1D beta tensor", "T")
        .Input(13, "word_embedding_zero_point", "Zero point for word embeddings", "T2")
        .Input(14, "position_embedding_zero_point", "Zero point for position embeddings", "T2")
        .Input(15, "segment_embedding_zero_point", "Zero Point for segment embeddings. Part of the all-or-none segment input group.", "T2", OpSchema::Optional)
        .Input(16, "gamma_zero_point", "Zero Point for 1D gamma tensor", "T2")
        .Input(17, "beta_zero_point", "Zero Point for 1D beta tensor", "T2")
        .Output(0, "layernorm_out", "LayerNorm Output", "T")
        .Output(1, "mask_index_out", "Mask Index Output", "T1")
        .TypeConstraint("T1", {"tensor(int32)"}, "Constrain mask index to integer types")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float32 tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          const bool has_segment_ids = ctx.getInputType(1) != nullptr;
          if (has_segment_ids != (ctx.getInputType(4) != nullptr) ||
              has_segment_ids != (ctx.getInputType(10) != nullptr) ||
              has_segment_ids != (ctx.getInputType(15) != nullptr)) {
            fail_type_inference(
                "segment_ids, segment_embedding, segment_embedding_scale, and segment_embedding_zero_point "
                "must either all be provided or all be omitted");
          }

          EmbedLayerNormalizationShapeInference(ctx);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QuantizeWithOrder, 1,
    OpSchema()
        .SetDoc(R"DOC(Quantize input matrix to specific layout used in cublaslt.)DOC")
        .Attr("order_input",
              "cublasLt order of input matrix. ORDER_COL = 0, ORDER_ROW = 1, ORDER_COL32 = 2, ORDER_COL4_4R2_8C = 3, "
              "ORDER_COL32_2R_4R4 = 4. "
              "Please refer https://docs.nvidia.com/cuda/cublas/index.html#cublasLtOrder_t for their meaning.",
              AttributeProto::INT)
        .Attr("order_output", "cublasLt order of output matrix.", AttributeProto::INT)
        .Input(0, "input",
               "TODO: input tensor of (ROWS, COLS). if less than 2d, will broadcast to (1, X). If 3d, it is treated as "
               "(B, ROWS, COS)",
               "F")
        .Input(1, "scale_input", "scale of the input", "S")
        .Output(0, "output", "output tensor", "Q")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("F", {"tensor(float16)", "tensor(float)"}, "Constrain to float types")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain Scale to float32 types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto::INT8, 0);
          if (!hasInputShape(ctx, 0)) return;
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    DequantizeWithOrder, 1,
    OpSchema()
        .SetDoc(
            R"DOC(Dequantize input matrix to specific layout used in cublaslt. attr to specify output type, float16 or float32)DOC")
        .Attr("order_input",
              "cublasLt order of input matrix. See the schema of QuantizeWithOrder for order definition.",
              AttributeProto::INT)
        .Attr("order_output", "cublasLt order of output matrix", AttributeProto::INT)
        .Attr("to",
              "The output data type, only support TensorProto_DataType_FLOAT (1) and TensorProto_DataType_FLOAT16 (10)",
              AttributeProto::INT)
        .Input(0, "input",
               "TODO: input tensor of (ROWS, COLS). if less than 2d, will broadcast to (1, X). If 3d, it is treated as "
               "(B, ROWS, COS)",
               "Q")
        .Input(1, "scale_input", "scale of the input", "S")
        .Output(0, "output", "output tensor", "F")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("F", {"tensor(float16)", "tensor(float)"}, "Constrain to float types")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain Scale to float32 types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (!hasInputShape(ctx, 0)) return;
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

constexpr const char* QOrderedMatMul_ver1_doc = R"DOC(
Quantize (Int8) MatMul with order. Implement Y = alpha * A * B + bias + beta * C. Matrix A, B, C, Y are all int8 matrix.
Two type of order combination supported:
  *) When order_B is ORDER_COL, order_A must be ORDER_ROW.
         bias is vector of {#cols of Y} of float32, C should be batch 1/batch_A. B could be of batch 1 or batch_A.
         Note B is reorder to ORDER_COL, or Transposed. Not Transposed first and then Reordered here.
  *) When order_B is specify ORDER_COL4_4R2_8C or ORDER_COL32_2R_4R4, orderA must be ORDER_COL32.
         MatMul will be implemented using alpha(A * B) + beta * C => Y.
         bias is not supported here. B in fact is transposed first then reordered into ORDER_COL4_4R2_8C or ORDER_COL32_2R_4R4 here.
order_Y and order_C will be same as order_A.
Support per column quantized weight, ie, scale_B is 1-D vector of size [#cols of matrix B].
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QOrderedMatMul, 1,
    OpSchema()
        .SetDoc(QOrderedMatMul_ver1_doc)
        .Attr("order_A", "cublasLt order of matrix A. See the schema of QuantizeWithOrder for order definition.",
              AttributeProto::INT)
        .Attr("order_B", "cublasLt order of matrix B", AttributeProto::INT)
        .Attr("order_Y", "cublasLt order of matrix Y and optional matrix C", AttributeProto::INT)
        .Input(0, "A", "3-dimensional matrix A", "Q")
        .Input(1, "scale_A", "scale of the input A.", "S")
        .Input(2, "B", "2-dimensional matrix B. Transposed if order_B is ORDER_COL.", "Q")
        .Input(3, "scale_B", "scale of the input B. Scalar or 1-D float32.", "S")
        .Input(4, "scale_Y", "scale of the output Y.", "S")
        .Input(5, "bias", "1d bias, not scaled with scale_Y.", "S", OpSchema::Optional)
        .Input(6, "C", "3d or 2d matrix C. if 2d expand to 3d first. Shape[0] should be 1 or same as A.shape[0] ", "Q",
               OpSchema::Optional)
        .Input(7, "scale_C", "scale of the input A.", "S", OpSchema::Optional)
        .Output(0, "Y", "Matrix multiply results from A * B", "Q")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain bias and scales to float32")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          ONNX_NAMESPACE::defs::math::utils::MatMulShapeInference(ctx, 0, 2);
        }));

static const char* Attention_QOrdered_doc = R"DOC(
Quantized version of simplified Multi-Head Self Attention(using int8 with specific matrix Layout).
Multi-Head Self Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT).
The mask_index input is optional. Besides raw attention mask with shape (batch_size, past_sequence_length + sequence_length)
or (batch_size, sequence_length, past_sequence_length + sequence_length) with value 0 for masked and 1 otherwise,
we also support other two formats: When input has right-side padding, mask_index is one dimension with shape (batch_size),
where value of each element is the end position, or valid length of actual sequence excluding padding. When input has
left-side padding, mask_index has shape (2 * batch_size), where the values are the exclusive end positions followed by
the inclusive start positions. When unidirectional is 1, and each token only attend to previous tokens. For GPT-2, both past
and present state are optional. Present state could appear in output even when past state is not in input.
Current version does not support past/present, attention_bias and qkv_hidden_sizes.
TODO: Support them if needed in the future.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    QOrderedAttention, 1,
    OpSchema()
        .SetDoc(Attention_QOrdered_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("unidirectional", "Whether every token can only attend to previous tokens. Default value is 0.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("qkv_hidden_sizes", "Hidden layer sizes of Q, K, V paths in Attention", AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("order_input",
              "cublasLt order of input matrix. See the schema of QuantizeWithOrder for order definition.",
              AttributeProto::INT)
        .Attr("order_weight", "cublasLt order of weight matrix", AttributeProto::INT)
        .Attr("order_output", "cublasLt order of global bias", AttributeProto::INT)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, input_hidden_size)", "Q")
        .Input(1, "scale_input", "scale of the input, scalar value (per tensor) currently.", "S")
        .Input(2, "scale_Q_gemm", "scale of the gemm - scalar (per-tensor quantization)", "S")
        .Input(3, "scale_K_gemm", "scale of the gemm - scalar (per-tensor quantization)", "S")
        .Input(4, "scale_V_gemm", "scale of the gemm - scalar (per-tensor quantization)", "S")
        .Input(5, "Q_weight",
               "2D input tensor with shape (input_hidden_size, hidden_size), where hidden_size = num_heads * head_size",
               "Q")
        .Input(6, "K_weight",
               "2D input tensor with shape (input_hidden_size, hidden_size), where hidden_size = num_heads * head_size",
               "Q")
        .Input(7, "V_weight",
               "2D input tensor with shape (input_hidden_size, hidden_size), where hidden_size = num_heads * head_size",
               "Q")
        .Input(8, "scale_Q_weight",
               "scale of the weight (scalar for per-tensor quantization or 1-D of dims [hidden_size] for per-channel "
               "quantization)",
               "S")
        .Input(9, "scale_K_weight",
               "scale of the weight (scalar for per-tensor quantization or 1-D of dims [hidden_size] for per-channel "
               "quantization)",
               "S")
        .Input(10, "scale_V_weight",
               "scale of the weight (scalar for per-tensor quantization or 1-D of dims [hidden_size] for per-channel "
               "quantization)",
               "S")
        .Input(11, "Q_bias", "1D input tensor with shape (hidden_size)", "S")
        .Input(12, "K_bias", "1D input tensor with shape (hidden_size)", "S")
        .Input(13, "V_bias", "1D input tensor with shape (hidden_size)", "S")
        .Input(14, "scale_QKT_gemm", "scale of the gemm - scalar (per-tensor quantization)", "S", OpSchema::Optional)
        .Input(15, "scale_QKT_softmax", "scale of the softmax result - scalar (per-tensor quantization)", "S",
               OpSchema::Optional)
        .Input(16, "scale_values_gemm",
               "scale of the gemm - scalar (per-tensor quantization). Also this is the output scale for the operator.",
               "S")
        .Input(17, "mask_index",
               "Attention mask with shape (batch_size, 1, max_sequence_length, max_sequence_length), (batch_size, "
               "past_sequence_length + sequence_length)"
               "or (batch_size, sequence_length, past_sequence_length + sequence_length), or index with shape "
               "(batch_size) or (2 * batch_size).",
               "G", OpSchema::Optional)
        .Input(18, "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).",
               "Q", OpSchema::Optional)
        .Input(19, "attention_bias",
               "additional add to QxK' with shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length).", "S",
               OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "Q")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain scales to float32 tensors.")
        .TypeConstraint("G", {"tensor(int32)"}, "Constrain to integer types")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(QOrderedLayerNormalization, 1,
                            OpSchema()
                                .SetDoc("QOrderedLayerNormalization")
                                .Attr("axis",
                                      "The first normalization dimension: normalization "
                                      "will be performed along dimensions axis "
                                      ": rank(inputs).",
                                      AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("epsilon", "The epsilon value to use to avoid division by zero.",
                                      AttributeProto::FLOAT, 1e-5f)
                                .Attr("order_X",
                                      "cublasLt order of input X. Default is ROW MAJOR. See the schema of "
                                      "QuantizeWithOrder for order definition.",
                                      AttributeProto::INT, static_cast<int64_t>(1))
                                .Attr("order_Y",
                                      "cublasLt order of matrix Y, must be same as order_X. Default is ROW MAJOR.",
                                      AttributeProto::INT, static_cast<int64_t>(1))
                                .AllowUncheckedAttributes()
                                .Input(0, "X", "Input data tensor from the previous layer.", "Q")
                                .Input(1, "scale_X", "scale of the quantized X", "S")
                                .Input(2, "scale", "Scale tensor, i.e., gamma vector.", "F")
                                .Input(3, "B", "Bias tensor.", "F", OpSchema::Optional)
                                .Input(4, "scale_Y", "scale of the quantized X", "S")
                                .Output(0, "Y", "Output data tensor.", "Q")
                                .TypeConstraint("F", {"tensor(float16)", "tensor(float)"},
                                                "Constrain input gamma and bias could be float16/float tensors. "
                                                "float may get better precision, float16 runs faster.")
                                .TypeConstraint("S", {"tensor(float)"}, "quantization scale must be float tensors.")
                                .TypeConstraint("Q", {"tensor(int8)"}, "quantization tensor must be int8 tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateShapeAndTypeFromFirstInput(ctx);
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QOrderedGelu, 1,
    OpSchema()
        .SetDoc(R"DOC(Ordered Quantize Gelu.)DOC")
        .Attr("order_X",
              "cublasLt order of input X. Optional. See the schema of QuantizeWithOrder for order definition.",
              AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("order_Y", "cublasLt order of matrix Y, must be same as order_X if specified together. Optional.",
              AttributeProto::INT, OPTIONAL_VALUE)
        .Input(0, "X", "N-dimensional input A", "Q")
        .Input(1, "scale_X", "scale of the input A", "S")
        .Input(2, "scale_Y", "scale of the output Y", "S")
        .Output(0, "Y", "Output of the Gelu", "Q")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain scales to float32")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(
    QOrderedLongformerAttention, 1,
    OpSchema()
        .SetDoc(R"DOC(Quantized version of Longformer Self Attention (using int8 with specific matrix Layout).)DOC")
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("window", "One sided attention windows length W, or half of total window length", AttributeProto::INT)
        .Attr("order_input",
              "cublasLt order of input matrix. See the schema of QuantizeWithOrder for order definition.",
              AttributeProto::INT)
        .Attr("order_weight", "cublasLt order of weight matrix", AttributeProto::INT)
        .Attr("order_global_weight", "cublasLt order of weight matrix", AttributeProto::INT)
        .Attr("order_output", "cublasLt order of global bias", AttributeProto::INT)
        .Input(0, "input",
               "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * "
               "head_size",
               "Q")
        .Input(1, "scale_input", "scale of the input", "S")
        .Input(2, "weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "Q")
        .Input(3, "scale_weight", "scale of the weight", "S")
        .Input(4, "bias", "1D input tensor with shape (3 * hidden_size), fp32 only currently.", "S")
        .Input(5, "scale_bias", "reserved. (not used as add bias need float value in cublasLt for normal order.)", "S")
        .Input(6, "scale_qkv_gemm", "scale of the output for fused kqv gemm", "S")
        .Input(7, "mask", "Attention mask with shape (batch_size, sequence_length)", "F")
        .Input(8, "global_weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "Q")
        .Input(9, "scale_global_weight", "scale of the global_weight", "S")
        .Input(10, "global_bias", "1D input tensor with shape (3 * hidden_size)", "S")
        .Input(11, "scale_global_gemm", "scale of the global_qkv_gemm", "S")
        .Input(12, "global", "Global attention flags with shape (batch_size, sequence_length)", "G")
        .Input(13, "scale_output", "scale of the output", "S")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "Q")
        .TypeConstraint("Q", {"tensor(int8)"}, "Constrain input and output types to int8 tensors.")
        .TypeConstraint("S", {"tensor(float)"}, "Constrain scales to float32 tensors.")
        .TypeConstraint("G", {"tensor(int32)"}, "Constrain to integer types")
        .TypeConstraint("F", {"tensor(float16)"}, "Be compatible with float version.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

/*
static void convTransposeShapeInference(InferenceContext& ctx,
                                        std::array<int, 3>& input_remap) {
  // Sometimes, the onnx shapeInfer function does supply the QlinearVersion of
  // relavant Op. But there is not too much difference except input_idx.
  // For example, ConvTranspose has 3 inputs (input,weight,bias),
  // while QlinearConvTranspose has 7 inputs (input,scale_input,Zp,weight,
  // scale_weight,Zp,output_scale,Zp,bias).
  //That's why we need to remap the input index.
  class InputIdxReMappingInferContext : public InferenceContext {
   private:
    InferenceContext& ctx_pa_;
    std::array<int, 3>& input_map_;

   public:
    InputIdxReMappingInferContext(InferenceContext& ctx, std::array<int, 3>& input_map) : ctx_pa_(ctx),
                                                                                       input_map_(input_map) {}
    const onnx::TypeProto* getInputType(size_t index) const override {
      return ctx_pa_.getInputType(input_map_[index]);
    }
    const AttributeProto* getAttribute(const std::string& name) const override {
      return ctx_pa_.getAttribute(name);
    }
    size_t getNumInputs() const override {
      return ctx_pa_.getNumInputs();
    }
    const onnx::TensorProto* getInputData(size_t index) const override {
      return ctx_pa_.getInputData(input_map_[index]);
    }
    size_t getNumOutputs() const override {
      return ctx_pa_.getNumOutputs();
    }
    onnx::TypeProto* getOutputType(size_t index) override {
      return ctx_pa_.getOutputType(input_map_[index]);
    }
    onnx::GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
      return ctx_pa_.getGraphAttributeInferencer(attribute_name);
    }
    const onnx::SparseTensorProto* getInputSparseData(size_t index) const override {
      return ctx_pa_.getInputSparseData(input_map_[index]);
    }
    // Gets the shape inputs computed by partial data propagation.
    const onnx::TensorShapeProto* getSymbolicInput(size_t index) const override {
      return ctx_pa_.getSymbolicInput(input_map_[index]);
    }
  };

  InputReMappingInferContext remapped_ctx(ctx, input_remap);
  ONNX_NAMESPACE::convTransposeShapeInference(remapped_ctx);
}
*/

ONNX_MS_OPERATOR_SET_SCHEMA(
    QLinearConvTranspose,
    1,
    OpSchema()
        .SetDoc(R"DOC( Similar to ConvTranspose in onnx, but with quantization.
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC")
        .Input(
            0,
            "x",
            "Input data tensor from previous layer; has size (N x C x H x W)"
            ", where N is the batch size, C is the number of channels, and"
            " H and W are the height and width. Note that this is for the 2D image. "
            "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
            "T1")
        .Input(
            1,
            "x_scale",
            "Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "T1")
        .Input(
            3,
            "w",
            "The weight tensor that will be used in the "
            "convolutions; has size (C x M/group x kH x kW), where C "
            "is the number of channels, and kH and kW are the "
            "height and width of the kernel, and M is the number "
            "of feature maps. ",
            "T2")
        .Input(
            4,
            "w_scale",
            "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "tensor(float)")
        .Input(
            5,
            "w_zero_point",
            "Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "T2")
        .Input(
            6,
            "y_scale",
            "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            7,
            "y_zero_point",
            "Zero point tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "T3")
        .Input(
            8,
            "B",
            "Optional 1D bias to be added to the convolution, has size of M. "
            "Bias must be quantized using scale = x_scale * w_scale and zero_point = 0",
            "T4",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.",
            "T3")
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain filter type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output type to 8-bit integer tensor.")
        .TypeConstraint("T4", {"tensor(int32)"}, "Constrain bias type to 32-bit integer tensor.")
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input W.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "output_shape",
            "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
            "pads values are ignored. See doc for details for equations to generate pads",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "output_padding",
            "Additional elements added to the side with higher coordinate indices in the output. "
            "Each padding value in \"output_padding\" must be less than the corresponding stride/dilation dimension. "
            "By default, this attribute is a zero vector. "
            "Note that this attribute doesn't directly affect the computed output values. "
            "It only controls the selection of the computed values, "
            "so changing this attribute only adds or removes output elements. "
            "If \"output_shape\" is explicitly provided, "
            "\"output_padding\" does not contribute additional size to \"output_shape\" but "
            "participates in the computation of the needed padding amount. "
            "This is also called adjs or adjustment in some frameworks.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("auto_pad",
              "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
              "default value is NOTSET",
              AttributeProto::STRING, std::string("NOTSET"))
        .Attr("pads", "Padding for the beginning and ending along each spatial axis", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(3);
          if (nullptr == x_type || nullptr == w_type || x_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              w_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          auto x_zero_point_type = ctx.getInputType(2);
          if (nullptr == x_zero_point_type ||
              x_zero_point_type->tensor_type().elem_type() != x_type->tensor_type().elem_type()) {
            fail_type_inference("input and zero_point pair is expected to have be same type.");
          }

          auto w_zero_point_type = ctx.getInputType(5);
          if (nullptr == w_zero_point_type ||
              w_zero_point_type->tensor_type().elem_type() != w_type->tensor_type().elem_type()) {
            fail_type_inference("weight and zero_point pair is expected to have same type.");
          }

          if (nullptr == x_type || nullptr == w_type || x_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              w_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          // validate scale and zero points
          // scale and zero points could be scalar or 1-D tensor which depends on quanization per-channel or per-tensor
          ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 2, x_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 4, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Both);
          ValidateTypeAndShapeForScaleAndZP(ctx, 5, w_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 6, ONNX_NAMESPACE::TensorProto::FLOAT, QuantParamTensorType::Scalar);
          ValidateTypeAndShapeForScaleAndZP(ctx, 7, x_type->tensor_type().elem_type(), QuantParamTensorType::Scalar);

          propagateElemTypeFromInputToOutput(ctx, 7, 0);

          // TODO: uncomment this after we really need to infer the output shape.
          // Since we haven't implemented QLinearConvTranspose in CPU EP yet,
          // we just leverage ConvTranspose's shape inference.
          // input, weight, bias
          // std::array<int, 3> input_remap = {0, 3, 8};
          // convTransposeShapeInference(ctx, input_remap);
        }));

}  // namespace contrib
}  // namespace onnxruntime
