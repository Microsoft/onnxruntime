// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/contrib_ops/contrib_defs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include "core/graph/onnx_protobuf.h"

#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/op.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"
#include "onnx/defs/function.h"
// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif
namespace ONNX_NAMESPACE {

inline int64_t HandleNegativeAxis(int64_t axis, int64_t rank) {
  if (rank < 0 || axis >= rank || axis < -rank) {
    fail_shape_inference("axis ", axis,
                         " is not in valid range [-", rank, ",", rank - 1, "]");
  }

  // Handle negative axis
  return axis < 0 ? axis + rank : axis;
}

void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);

namespace defs::math::utils {
void MatMulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);
}

void convTransposeWithDynamicPadsShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);
  if (group <= 0) {
    fail_shape_inference("group attribute must be positive. Got: ", group);
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  // ConvTranspose requires X=(N x C x D1...Dn) and W=(C x M/group x k1...kn), both rank >= 3.
  // The upstream ONNX ConvTranspose shape inference only checks rank >= 2, which allows rank-2
  // inputs to pass shape inference but crash at kernel execution time. We tighten the check here
  // to fail early at model load with a clear error. Fixing ONNX upstream is tracked separately.
  if (input_shape.dim_size() < 3) {
    fail_shape_inference("Input tensor must have at least 3 dimensions. Got: ", input_shape.dim_size());
  }

  auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
  if (weight_shape.dim_size() < 3) {
    fail_shape_inference("Weight tensor must have at least 3 dimensions. Got: ", weight_shape.dim_size());
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - size_t{2});

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
    // A longer kernel_shape (W rank > X rank) overruns `dilations` in the loop right below;
    // a shorter one (W rank < X rank) leaves `effective_kernel_shape` too short for the
    // output-shape loop further below.
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;

  // Infer output shape if 'pads' tensor is available
  if (ctx.getNumInputs() <= 2) {
    return;
  }

  const auto* pads_initializer = ctx.getInputData(2);
  if (nullptr == pads_initializer) {
    return;
  }

  if (pads_initializer->dims_size() != 1 ||
      pads_initializer->data_type() != TensorProto::INT64)
    fail_shape_inference(
        "'pads' input must be a 1D (shape: [2 * n_input_dims]) tensor of type int64");

  pads = ParseData<int64_t>(pads_initializer);

  if (pads.size() != static_cast<size_t>(2 * n_input_dims))
    fail_shape_inference("Pads has incorrect number of values");

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) {  // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto final_output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      weight_shape.dim(1) *
      group;  // channels should be the second dim of second input multiply
  // group.

  int size_of_output;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO: throw exception?
          return;  // output shape value cannot be smaller than the input shape
                   // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim =
            strides[i] * (input_shape.dim(i + 2).dim_value() - 1) +
            output_padding[i] + effective_kernel_shape[i] - pads[i] -
            pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {
using namespace ONNX_NAMESPACE;
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void FusedMatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto transAAttr = ctx.getAttribute("transA");
  bool transa = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
  auto transBAttr = ctx.getAttribute("transB");
  bool transb = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
  auto trans_batch_a_attr = ctx.getAttribute("transBatchA");
  bool trans_batch_a = trans_batch_a_attr ? static_cast<int>(trans_batch_a_attr->i()) != 0 : false;
  auto trans_batch_b_attr = ctx.getAttribute("transBatchB");
  bool trans_batch_b = trans_batch_b_attr ? static_cast<int>(trans_batch_b_attr->i()) != 0 : false;
  int input1Idx = 0;
  int input2Idx = 1;
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0_raw = getInputShape(ctx, input1Idx);
  const auto shape1_raw = getInputShape(ctx, input2Idx);

  if (shape0_raw.dim_size() == 0 || shape1_raw.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  // numpy transpose on a vector does not change anything.
  if (shape0_raw.dim_size() == 1) {
    transa = false;
  }
  if (shape1_raw.dim_size() == 1) {
    transb = false;
  }

  ONNX_NAMESPACE::TensorShapeProto shape0, shape1;
  auto rank0 = shape0_raw.dim_size();
  if (rank0 == 1) {
    // for vector input, transa does not make impact on the dim.
    shape0 = shape0_raw;
  } else {
    int start = trans_batch_a ? 1 : 0;
    int end = trans_batch_a ? rank0 - 1 : rank0 - 2;
    for (int i = start; i < end; ++i) {
      *shape0.add_dim() = shape0_raw.dim(i);
    }
    *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 1 : (trans_batch_a ? 0 : rank0 - 2));
    *shape0.add_dim() = shape0_raw.dim(transa ? (trans_batch_a ? 0 : rank0 - 2) : rank0 - 1);
  }

  auto rank1 = shape1_raw.dim_size();
  if (rank1 == 1) {
    // for vector input, transb does not make impact on the dim.
    shape1 = shape1_raw;
  } else {
    int start = trans_batch_b ? 1 : 0;
    int end = trans_batch_b ? rank1 - 1 : rank1 - 2;
    for (int i = start; i < end; ++i) {
      *shape1.add_dim() = shape1_raw.dim(i);
    }
    *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 1 : (trans_batch_b ? 0 : rank1 - 2));
    *shape1.add_dim() = shape1_raw.dim(transb ? (trans_batch_b ? 0 : rank1 - 2) : rank1 - 1);
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }
  updateOutputShape(ctx, 0, resultShape);
}

// input1Idx - sparse matrix
// input2Idx - dense matrix.
// Output is dense
void sparseCompatibleMatmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = getInputShape(ctx, input1Idx);
  const auto shape1 = getInputShape(ctx, input2Idx);

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  // if the input 2 type was not previously propagate to output
  // we want to make sure that it is the tensor type of input 2
  auto default_tensor_type = ctx.getInputType(input2Idx)->value_case();
  updateOutputShape(ctx, 0, resultShape, default_tensor_type);
}

bool ParseScalar(const TensorProto* initializer, int& value) {
  std::vector<int32_t> parsed_data;
  if (initializer->data_type() == TensorProto::INT32) {
    const auto& data = ParseData<int32_t>(initializer);
    parsed_data.insert(parsed_data.end(), data.begin(), data.end());

    if (parsed_data.size() == 1) {
      value = parsed_data[0];
      return true;
    }
  }

  return false;
}

void BeamSearchShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (ctx.getNumOutputs() > 1) {
    // Here we assume that the third output exist only if second output exists.
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 5, 1);
    if (ctx.getNumOutputs() > 2) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 5, 2);
    }
  }

  // Shape inference
  // input 0 (input_ids) shape: (batch_size, sequence_length)
  // output 0 (sequences) shape: (batch_size, num_return_sequences, max_length)
  // output 1 (sequences_scores) shape: (batch_size, num_return_sequences)
  // output 2 (scores) shape: (max_length - sequence_length, batch_size, num_beams, vocab_size)
  // output 3 (cross_attention): shape: (batch_size, num_return_sequences, Layers, Heads, max_length, Frames)
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  auto& input_ids_shape = getInputShape(ctx, 0);
  auto& input_ids_dims = input_ids_shape.dim();
  auto model_type_attr = ctx.getAttribute("model_type");
  int64_t model_type = model_type_attr ? static_cast<int64_t>(model_type_attr->i()) : -1;
  if (model_type == onnxruntime::contrib::transformers::IGenerationParameters::kModelTypeWhisper) {
    if (input_ids_dims.size() != 3) {
      fail_shape_inference("Inputs 0 shall be 3 dimensions in whisper graph");
    }
    if (!(input_ids_dims[0].has_dim_value() && input_ids_dims[1].has_dim_value() && input_ids_dims[2].has_dim_value())) {
      return;
    }
  } else if (input_ids_dims.size() != 2) {
    fail_shape_inference("Inputs 0 shall be 2 dimensions", model_type);
  }
  if (!(input_ids_dims[0].has_dim_value() && input_ids_dims[1].has_dim_value())) {
    return;
  }

  int64_t batch_size = input_ids_dims[0].dim_value();
  int64_t sequence_length = input_ids_dims[1].dim_value();

  const auto max_length = ctx.getInputData(1);
  const auto num_beams = ctx.getInputData(3);
  const auto num_return_sequences = ctx.getInputData(4);
  if (num_beams == nullptr || max_length == nullptr || num_return_sequences == nullptr) {  // not initializer
    return;
  }

  int max_length_value = 0;
  if (!ParseScalar(max_length, max_length_value) || max_length_value <= 0) {
    fail_shape_inference("Failed to parse max_length or it is not positive integer scalar");
  }

  int num_beams_value = 0;
  if (!ParseScalar(num_beams, num_beams_value) || num_beams_value <= 0) {
    fail_shape_inference("Failed to parse num_beams or it is not positive integer scalar");
  }

  int num_return_sequences_value = 0;
  if (!ParseScalar(num_return_sequences, num_return_sequences_value) || num_return_sequences_value <= 0) {
    fail_shape_inference("Failed to parse num_return_sequences or it is not positive integer scalar");
  }

  ONNX_NAMESPACE::TensorShapeProto sequences_shape;
  sequences_shape.add_dim()->set_dim_value(batch_size);
  sequences_shape.add_dim()->set_dim_value(num_return_sequences_value);
  sequences_shape.add_dim()->set_dim_value(max_length_value);
  updateOutputShape(ctx, 0, sequences_shape);

  if (ctx.getNumOutputs() > 1) {
    ONNX_NAMESPACE::TensorShapeProto sequences_scores_shape;
    sequences_shape.add_dim()->set_dim_value(batch_size);
    sequences_shape.add_dim()->set_dim_value(num_return_sequences_value);
    updateOutputShape(ctx, 1, sequences_scores_shape);

    if (ctx.getNumOutputs() > 2) {
      auto vocab_size_attr = ctx.getAttribute("vocab_size");
      int64_t vocab_size = vocab_size_attr ? static_cast<int64_t>(vocab_size_attr->i()) : -1;

      ONNX_NAMESPACE::TensorShapeProto scores_shape;
      scores_shape.add_dim()->set_dim_value(max_length_value - sequence_length);
      scores_shape.add_dim()->set_dim_value(batch_size);
      scores_shape.add_dim()->set_dim_value(num_beams_value);
      if (vocab_size != -1) {
        // vocab_size is provided by the user - use that
        scores_shape.add_dim()->set_dim_value(vocab_size);
      } else {
        // vocab_size is unknown
        scores_shape.add_dim();
      }
      updateOutputShape(ctx, 2, scores_shape);
    }
  }
}

void GreedySearchShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  // input 0 (input_ids) shape: (batch_size, sequence_length)
  // output 0 (sequences) shape: (batch_size, max_length)
  // output 1 (scores) shape: (max_length - sequence_length, batch_size, num_beams, vocab_size)
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  auto& input_ids_shape = getInputShape(ctx, 0);
  auto& input_ids_dims = input_ids_shape.dim();
  if (input_ids_dims.size() != 2) {
    fail_shape_inference("Inputs 0 shall be 2 dimensions");
  }
  if (!(input_ids_dims[0].has_dim_value() && input_ids_dims[1].has_dim_value())) {
    return;
  }

  int64_t batch_size = input_ids_dims[0].dim_value();

  const auto max_length = ctx.getInputData(1);
  if (max_length == nullptr) {  // not initializer
    return;
  }

  int max_length_value = 0;
  if (!ParseScalar(max_length, max_length_value) || max_length_value <= 0) {
    fail_shape_inference("Failed to parse max_length or it is not positive integer scalar");
  }

  ONNX_NAMESPACE::TensorShapeProto sequences_shape;
  sequences_shape.add_dim()->set_dim_value(batch_size);
  sequences_shape.add_dim()->set_dim_value(max_length_value);
  updateOutputShape(ctx, 0, sequences_shape);

  if (ctx.getNumOutputs() > 1) {
    ONNX_NAMESPACE::TensorShapeProto logits_to_debug_shape;
    logits_to_debug_shape.add_dim()->set_dim_value(batch_size);
    logits_to_debug_shape.add_dim();
    updateOutputShape(ctx, 1, logits_to_debug_shape);
  }
}

constexpr const char* Gelu_ver1_doc =
    R"DOC(Gaussian Error Linear Unit.
A high-performing neural network activation function.The GELU nonlinearity is
the expected transformation of a stochastic regularizer which randomly applies
the identity or zero map to a neuron's input. The GELU nonlinearity weights
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Gelu, 1,
                            OpSchema()
                                .SetDoc(Gelu_ver1_doc)
                                .Input(0, "X", "The input data as Tensor.", "T")
                                .Output(0, "Y", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
                                .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
                                .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
                                  // gelu(x) = x * Phi(x) = x * 1/2(1+erf(x/sqrt(2)))
                                  auto* tp = ctx.getInputType(0);
                                  if ((tp == nullptr) || (!tp->has_tensor_type()))
                                    return false;
                                  auto elem_type = (TensorProto_DataType)(tp->tensor_type().elem_type());

                                  FunctionBuilder builder(functionProto);
                                  builder
                                      .AddOpset("", 13)
                                      .Const("Half", ToTensor(0.5, elem_type))
                                      .Const("One", ToTensor(1.0, elem_type))
                                      .Const("C", ToTensor(std::sqrt(0.5), elem_type))
                                      .Add(R"(
                CX = Mul (C, X)
                ERFCX = Erf (CX)
                ERFCXPlus1 = Add (ERFCX, One)
                PhiX = Mul (ERFCXPlus1, Half)
                Y = Mul (X, PhiX)
            )");

                                  schema.BuildFunction(functionProto);
                                  return true;
                                }));

constexpr const char* BiasGelu_ver1_doc =
    R"DOC(Bias Gelu.
It's an extension of Gelu. It takes the sum of input A and bias input B as the input of Gelu activation. )DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(BiasGelu, 1,
                            OpSchema()
                                .SetDomain(kMSDomain)
                                .SinceVersion(1)
                                .SetDoc(BiasGelu_ver1_doc)
                                .Input(0, "A", "The normal input data.", "T")
                                .Input(1, "B", "The bias input data that is a 1D tensor.", "T")
                                .Output(0, "C", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* QuickGelu_ver1_doc = R"DOC(Compute x * Sigmoid(alpha * x).)DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(
    QuickGelu, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(QuickGelu_ver1_doc)
        .Attr("alpha", "Alpha value.", AttributeProto::FLOAT, 1.702f)
        .Input(0, "X", "The input data as Tensor.", "T")
        .Output(0, "Y", "The output.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx, const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          auto* tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type())) return false;
          auto elem_type = (TensorProto_DataType)(tp->tensor_type().elem_type());
          auto* alpha_attr = ctx.getAttribute("alpha");
          float alpha = (alpha_attr != nullptr) ? alpha_attr->f() : 1.702f;
          FunctionBuilder builder(functionProto);
          builder.AddOpset("", 13).Const("Alpha", ToTensor(alpha, elem_type)).Add(R"(
                CX = Mul (Alpha, X)
                SIGMOIDCX = Sigmoid (CX)
                Y = Mul (X, SIGMOIDCX)
            )");
          schema.BuildFunction(functionProto);
          return true;
        }));

// Used to be ONNX 1.7 Inverse(12)
// Comment out docs not to increase the binary size
//
//  constexpr const char* Inverse_ver1_doc = R"DOC(
// Calculates inverse of a square matrix or batches of square matrices.
// Inverse takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
// and the inner-most 2 dimensions form square matrices. These matrices must be invertible (full-rank).
// The behavior where one of the matrices is not invertible is undefined. The implementation can choose
// to throw an error or output (garbage) results as is. The output is a tensor of shape `[*, M, M]`,
// containing the individual inverses of all input submatrices.
//)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Inverse, 1,
                            OpSchema()
                                .Input(0, "X", "Input tensor. Every matrix in the batch must be invertible.", "T")
                                .Output(0, "Y", "Output tensor of the same type and shape as the input tensor.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (hasInputShape(ctx, 0)) {
                                    const TensorShapeProto& input_shape =
                                        ctx.getInputType(0)->tensor_type().shape();
                                    const int rank = static_cast<int>(input_shape.dim_size());

                                    if (rank < 2) {
                                      fail_shape_inference("Input rank must be >= 2.")
                                    }

                                    const auto mat_w = input_shape.dim(rank - 1);
                                    const auto mat_h = input_shape.dim(rank - 2);
                                    if (mat_w.has_dim_value() && mat_h.has_dim_value() &&
                                        (mat_w.dim_value() != mat_h.dim_value())) {
                                      fail_shape_inference(
                                          "The inner-most 2 dimensions must have the same size (mat_w:",
                                          mat_w.dim_value(),
                                          " != mat_h:",
                                          mat_h.dim_value(),
                                          ").");
                                    }

                                    // Shape inference
                                    propagateShapeFromInputToOutput(ctx, 0, 0);
                                  }
                                }));

constexpr const char* TorchEmbedding_ver1_doc = R"DOC(
      Based on Torch operator Embedding, creates a lookup table of embedding vectors of fixed size,
       for a dictionary of fixed size.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(TorchEmbedding, 1,
                            OpSchema()
                                .SetDoc(TorchEmbedding_ver1_doc)
                                .Input(
                                    0,
                                    "weight",
                                    "The embedding matrix of size N x M. 'N' is equal to the maximum possible index + 1, and 'M' is "
                                    "equal to the embedding size",
                                    "T")
                                .Input(
                                    1,
                                    "indices",
                                    "Long tensor containing the indices to extract from embedding matrix.",
                                    "tensor(int64)")
                                .Input(
                                    2,
                                    "padding_idx",
                                    "A 0-D scalar tensor. If specified, the entries at `padding_idx` do not contribute to the gradient; "
                                    "therefore, the embedding vector at `padding_idx` is not updated during training, "
                                    "i.e. it remains as a fixed pad.",
                                    "tensor(int64)",
                                    OpSchema::Optional)
                                .Input(
                                    3,
                                    "scale_grad_by_freq",
                                    "A 0-D bool tensor. If given, this will scale gradients by the inverse of frequency of "
                                    "the indices (words) in the mini-batch. Default  is ``False``",
                                    "tensor(bool)",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "Output tensor of the same type as the input tensor. Shape of the output is * x M, where '*' is the shape of "
                                    "input indices, and 'M' is the embedding size.",
                                    "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(bfloat16)",
                                     "tensor(uint8)",
                                     "tensor(uint16)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int8)",
                                     "tensor(int16)",
                                     "tensor(int32)",
                                     "tensor(int64)"},
                                    "Constrain input and output types to all numeric tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  TensorShapeProto outputs_shape;
                                  Dim input_dim_i;

                                  if (hasInputShape(ctx, 1)) {
                                    auto& input_shape = getInputShape(ctx, 1);
                                    for (int32_t i = 0; i < input_shape.dim_size(); i++) {
                                      input_dim_i = input_shape.dim(i);
                                      *outputs_shape.add_dim() = input_dim_i;
                                    }
                                  }

                                  Dim embedding_dim;
                                  unifyInputDim(ctx, 0, 1, embedding_dim);
                                  *outputs_shape.add_dim() = embedding_dim;
                                  updateOutputShape(ctx, 0, outputs_shape);
                                }));

constexpr const char* Trilu_ver1_doc = R"DOC(
      Returns the upper or lower triangular part of a 2-D matrix, or batches of 2-D matrices. If the attribute "upper" is set to true,
      the upper triangular matrix is retained. Lower triangular matrix is retained otherwise. Default value for upper is true.
      Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
      of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
      All other elements in the matrix are set to zero.
      If k = 0, the triangular part on and above/below the main diagonal is retained.
      If upper is set to true, a positive k retains the upper triangular matrix excluding k diagonals above
      the main diagonal. A negative k value includes as many diagonals below the main diagonal.
      If upper is set to false, a positive k retains the lower triangular matrix including k diagonals above
      the main diagonal. A negative k value excludes as many diagonals below the main diagonal.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Trilu, 1,
                            OpSchema()
                                .SetDoc(Trilu_ver1_doc)
                                .Attr("upper",
                                      "Boolean. Indicates whether upper or lower part of matrix is retained. Default is true.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(1))
                                .Input(
                                    0,
                                    "X",
                                    "Input tensor of rank 2 or higher.",
                                    "T")
                                .Input(
                                    1,
                                    "k",
                                    "A 0-D tensor containing a single value corresponding to the number diagonals above or the main diagonal to exclude or include."
                                    "Default value is 0 if it's not specified.",
                                    "tensor(int64)",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "Output tensor of the same type and shape as the input tensor.",
                                    "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(bfloat16)",
                                     "tensor(uint8)",
                                     "tensor(uint16)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int8)",
                                     "tensor(int16)",
                                     "tensor(int32)",
                                     "tensor(int64)",
                                     "tensor(bool)"},
                                    "Constrain input and output types to all numeric tensors and bool tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  if (hasInputShape(ctx, 0)) {
                                    const TensorShapeProto& input_shape =
                                        ctx.getInputType(0)->tensor_type().shape();
                                    const int rank = static_cast<int>(input_shape.dim_size());
                                    if (rank < 2) {
                                      fail_shape_inference("Input rank must be >= 2.")
                                    }
                                    propagateShapeFromInputToOutput(ctx, 0, 0);
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(BiasSoftmax, 1,
                            OpSchema()
                                .SetDoc(
                                    "Y = softmax(scores + bias)) with simple broadcast on bias. "
                                    "Intended to specialize softmax(scores + additive_mask) commonly found in transformer models.")
                                .Attr("axis", "apply softmax to elements for dimensions axis or higher", AttributeProto::INT, static_cast<int64_t>(1))
                                .Attr("is_inner_broadcast",
                                      "true if broadcast bias across input for dimensions broadcast_axis to axis-1, "
                                      "otherwise broadcast bias across input for dimensions 0 to broadcast_axis - 1",
                                      AttributeProto::INT)
                                .Input(0, "data", "The input data as Tensor.", "T")
                                .Input(1, "bias", "The bias (or mask) as Tensor.", "T")
                                .Output(0, "output", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(BiasDropout, 1,
                            OpSchema()
                                .SetDoc(
                                    "output, dropout_mask = Dropout(data + bias, ratio) + residual, "
                                    "Intended to specialize the dropout pattern commonly found in transformer models.")
                                .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL_VALUE)
                                .AllowUncheckedAttributes()
                                .Input(0, "data", "The input data as Tensor.", "T")
                                .Input(1, "bias", "The bias input, a vector with the same shape as last dim of data OR same shape with data", "T")
                                .Input(2, "residual", "The residual input, must have the same shape as data", "T", OpSchema::Optional)
                                .Input(3, "ratio",
                                       "The ratio of random dropout, with value in [0, 1). If this input was not set, "
                                       "or if it was set to 0, the output would be a simple copy of the input. "
                                       "If it's non-zero, output will be a random dropout of the scaled input, which is typically "
                                       "the case during training. It is an optional value, if not specified it will default to 0.5.",
                                       "T1",
                                       OpSchema::Optional)
                                .Input(4, "training_mode",
                                       "If set to true then it indicates dropout is being used for "
                                       "training. It is an optional value hence unless specified explicitly, it is false. "
                                       "If it is false, ratio is ignored and the operation mimics inference mode where nothing "
                                       "will be dropped from the input data and if mask is requested as output it will contain "
                                       "all ones.",
                                       "T2",
                                       OpSchema::Optional)
                                .Output(0, "output", "The output.", "T")
                                .Output(1, "mask", "The output mask of dropout.", "T2", OpSchema::Optional)
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input 'ratio' types to float tensors.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(bool)"},
                                    "Constrain output 'mask' types to boolean tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateShapeAndTypeFromFirstInput(ctx);
                                  if (ctx.getNumOutputs() == 2) {
                                    updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
                                    if (hasNInputShapes(ctx, 1)) {
                                      propagateShapeFromInputToOutput(ctx, 0, 1);
                                    }
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    BitmaskBiasDropout, 1,
    OpSchema()
        .SetDoc("output, dropout_bitmask = Dropout(data + bias, ratio) + residual, "
                "Intended to specialize the dropout pattern commonly found in transformer models.")
        .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.",
              AttributeProto::INT, OPTIONAL_VALUE)
        .AllowUncheckedAttributes()
        .Input(0, "data", "The input data as Tensor.", "T")
        .Input(1, "bias", "The bias input, a vector with the same shape as last dim of data OR same shape with data",
               "T")
        .Input(2, "residual", "The residual input, must have the same shape as data", "T", OpSchema::Optional)
        .Input(3, "ratio",
               "The ratio of random dropout, with value in [0, 1). If this input was not set, "
               "or if it was set to 0, the output would be a simple copy of the input. "
               "If it's non-zero, output will be a random dropout of the scaled input, which is typically "
               "the case during training. It is an optional value, if not specified it will default to 0.5.",
               "T1", OpSchema::Optional)
        .Input(4, "training_mode",
               "If set to true then it indicates dropout is being used for "
               "training. It is an optional value hence unless specified explicitly, it is false. "
               "If it is false, ratio is ignored and the operation mimics inference mode where nothing "
               "will be dropped from the input data and if mask is requested as output it will contain "
               "all ones.",
               "T2", OpSchema::Optional)
        .Output(0, "output", "The output.", "T")
        .Output(1, "mask", "The output mask of dropout.", "T3", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("T1", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                        "Constrain input 'ratio' types to float tensors.")
        .TypeConstraint("T2", {"tensor(bool)"}, "Constrain input 'training_mode' types to boolean tensors.")
        .TypeConstraint("T3", {"tensor(uint32)"}, "Constrain output 'mask' types to uint32 tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          if (ctx.getNumOutputs() == 2) {
            updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::UINT32);
          }
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(IsAllFinite, 1,
                            OpSchema()
                                .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
                                .SetDoc("IsAllFinite")
                                .SetDomain(kMSDomain)
                                .SinceVersion(1)
                                .Attr("isinf_only",
                                      "If true, check only for Inf, -Inf.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(0))
                                .Attr("isnan_only",
                                      "If true, check only for NaN.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(0))
                                .TypeConstraint(
                                    "V",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(bool)"},
                                    "Constrain the output to a boolean tensor.")
                                .Input(0, "input", "Input tensors to check.", "V",
                                       OpSchema::Variadic)
                                .Output(
                                    0,
                                    "output",
                                    "The output scalar. Its value is true if all input "
                                    "tensors are finite. Otherwise, the output value would "
                                    "be false.",
                                    "T")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  bool isinf_only = static_cast<bool>(getAttribute(ctx, "isinf_only", int64_t(0)));
                                  bool isnan_only = static_cast<bool>(getAttribute(ctx, "isnan_only", int64_t(0)));
                                  if (isinf_only && isnan_only) {
                                    fail_shape_inference("Both attributes isinf_only and isnan_only cannot be set. Unset both to check for both conditions.");
                                  }
                                  updateOutputShape(ctx, 0, {});
                                  updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
                                }));

constexpr const char* GridSample_ver1_doc = R"DOC(
      Given an `input` and a flow-field `grid`, computes the `output` using `input` values and pixel locations from `grid`.
      Currently, only spatial (4-D) inputs are supported. For `input` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
      the `output` will have shape (N, C, H_out, W_out).
      For each output location `output[n, :, h, w]`, the size-2 vector `grid[n, h, w]` specifies `input` pixel locations `x` and `y`,
      which are used to interpolate the output value `output[n, :, h, w]`.
      The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
      See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(GridSample, 1,
                            OpSchema()
                                .SetDoc(GridSample_ver1_doc)
                                .Attr(
                                    "mode",
                                    "Three interpolation modes: bilinear (default), nearest and bicubic.",
                                    AttributeProto::STRING,
                                    std::string("bilinear"))
                                .Attr(
                                    "padding_mode",
                                    "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. "
                                    "zeros: use 0 for out-of-bound grid locations, "
                                    "border: use border values for out-of-bound grid locations, "
                                    "reflection: use values at locations reflected by the border for out-of-bound grid locations.",
                                    AttributeProto::STRING,
                                    std::string("zeros"))
                                .Attr(
                                    "align_corners",
                                    "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. "
                                    "If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic.",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Input(
                                    0,
                                    "X",
                                    "4-D tensor of shape (N, C, H, W), "
                                    "where N is the batch size, C is the numbers of channels, "
                                    "H and W are the height and width of the input data.",
                                    "T1")
                                .Input(
                                    1,
                                    "Grid",
                                    "Input offset, 4-D tensor of shape (N, H_out, W_out, 2), "
                                    "where H_out and W_out are the height and width of grid and output, "
                                    "Grid specifies the sampling pixel locations normalized by the input spatial dimensions. "
                                    "Therefore, it should have most values in the range of [-1, 1]. "
                                    "If grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode.",
                                    "T1")
                                .Output(
                                    0,
                                    "Y",
                                    "4-D tensor of shape (N, C, H_out, W_out).",
                                    "T2")
                                .TypeConstraint(
                                    "T1",
                                    OpSchema::all_tensor_types(),
                                    "Constrain input types to all tensor types.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  size_t input_param = 0, grid_param = 1;

                                  checkInputRank(ctx, input_param, 4);
                                  checkInputRank(ctx, grid_param, 4);

                                  // Output dimensions, initialized to an unknown-dimension-value
                                  Dim N, C, H_out, W_out;

                                  // Get value of N from dim 0 of input_param, if available
                                  unifyInputDim(ctx, input_param, 0, N);
                                  // Get value of C from dim 1 of input_param, if available
                                  unifyInputDim(ctx, input_param, 1, C);

                                  // Get value of H_out from dim 1 of grid_param, if available
                                  unifyInputDim(ctx, grid_param, 1, H_out);
                                  // Get value of W_out from dim 2 of grid_param, if available
                                  unifyInputDim(ctx, grid_param, 2, W_out);

                                  // set output shape:
                                  updateOutputShape(ctx, 0, {N, C, H_out, W_out});
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    UnfoldTensor, 1,
    OpSchema()
        .SetDoc("Returns a tensor which contains all slices of size `size` from input tensor in the dimension `dim`. "
                "Step between two slices is given by `step`. "
                "If `sizedim` is the size of dimension `dim` for input tensor, the size of dimension `dim` in "
                "the returned tensor will be `(sizedim - size) / step + 1`. "
                "An additional dimension of size `size` is appended in the returned tensor.")
        .Attr("dim", "specify the dimension to unfold", AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("size", "specify the size", AttributeProto::INT)
        .Attr("step", "specify the step.", AttributeProto::INT, static_cast<int64_t>(1))
        .Input(0, "input", "input tensor", "T")
        .Output(0, "output", "Output tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Allow inputs and outputs to be any kind of tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          if (!hasInputShape(ctx, 0)) return;
          auto& input_shape = getInputShape(ctx, 0);
          const int rank = input_shape.dim_size();
          int64_t dim = getAttribute(ctx, "dim", -1);
          dim = HandleNegativeAxis(dim, rank);
          if (!input_shape.dim(static_cast<int>(dim)).has_dim_value()) {
            return;
          }
          int64_t dim_size = input_shape.dim(static_cast<int>(dim)).dim_value();

          const int64_t step = getAttribute(ctx, "step", -1);
          if (step <= 0) {
            fail_shape_inference("size attribute in UnfoldTensor must greater than 0.")
          }
          int64_t size = -1;
          auto size_proto = ctx.getAttribute("size");
          if (!(size_proto)) {
            fail_shape_inference("size attribute in UnfoldTensor not specified!")
          }
          size = size_proto->i();
          if (size > dim_size || size <= 0) {
            fail_shape_inference("size attribute in UnfoldTensor not positive and less than the dim size!")
          }

          ONNX_NAMESPACE::TensorShapeProto output_shape;
          for (int d = 0; d < rank; d++) {
            if (d == dim) {
              output_shape.add_dim()->set_dim_value((dim_size - size) / step + 1);
            } else {
              *output_shape.add_dim() = input_shape.dim(d);
            }
          }
          output_shape.add_dim()->set_dim_value(size);
          updateOutputShape(ctx, 0, output_shape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    DynamicTimeWarping, 1,
    OpSchema()
        .SetDoc("Input is cost matrix where each value in input[r][c] is the cost for pass the point (r, c). From current point"
                "(r, c),  points (r+1, c), (r+1, c+1) or (r, c+1) could be arrived in next move. Given such cost matrix, return "
                "dynamic time warping of shape [2, x], where the path made by all points (output[0][t], output[1][t])"
                "have the lowest cost among all paths from (0, 0) to (M-1, N-1).")
        .Input(0, "input", "Input cost tensor, it must be 2D tensor of shape M x N, or 1 x M x N", "F")
        .Output(0, "output", "Output tensor. shape is [2, x], where max(M, N) <= x < M + N", "I")
        .TypeConstraint("F", {"tensor(float)"}, "Constrain to float tensors.")
        .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT32);
          ONNX_NAMESPACE::TensorShapeProto resultShape;
          resultShape.add_dim()->set_dim_value(2);
          resultShape.add_dim();
          updateOutputShape(ctx, 0, resultShape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(BeamSearch, 1,
                            OpSchema()
                                .SetDoc("Beam Search for text generation. Supports GPT-2 decoder.")
                                .Attr("eos_token_id", "The id of the end-of-sequence token", AttributeProto::INT)
                                .Attr("pad_token_id", "The id of the padding token", AttributeProto::INT)
                                .Attr("decoder_start_token_id", "The id of the token that indicates decoding starts.", AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("no_repeat_ngram_size", "no repeat ngrams size", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("early_stopping", "early stop or not", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("model_type", "model type: 0 for GPT-2; 1 for encoder decoder like T5", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("encoder", "The subgraph for initialization of encoder and decoder. It will be called once before decoder subgraph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("init_decoder",
                                      "The subgraph for the first decoding run. It will be called once before `decoder` subgraph. "
                                      "This is relevant only for the GPT2 model. If this attribute is missing, the `decoder` subgraph will be used for all decoding runs",
                                      AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("decoder", "Decoder subgraph to execute in a loop.", AttributeProto::GRAPH)
                                .Attr("vocab_size",
                                      "Size of the vocabulary. "
                                      "If not provided, it will be inferred from the decoder subgraph's output shape",
                                      AttributeProto::INT, static_cast<int64_t>(-1))
                                .Input(0, "input_ids", "The sequence used as a prompt for the generation in the encoder subgraph. Shape is (batch_size, sequence_length)", "F")
                                .Input(1, "max_length", "The maximum length of the sequence to be generated. Shape is (1)", "I")
                                .Input(2, "min_length", "The minimum length below which the score of eos_token_id is set to -Inf. Shape is (1)", "I", OpSchema::Optional)
                                .Input(3, "num_beams", "Number of beams for beam search. 1 means no beam search. Shape is (1)", "I")
                                .Input(4, "num_return_sequences", "The number of returned sequences in the batch. Shape is (1)", "I")
                                .Input(5, "length_penalty",
                                       "Exponential penalty to the length. Default value 1.0 means no penalty."
                                       "Value > 1.0 encourages longer sequences, while values < 1.0 produces shorter sequences."
                                       "Shape is (1,)",
                                       "T", OpSchema::Optional)
                                .Input(6, "repetition_penalty", "The parameter for repetition penalty. Default value 1.0 means no penalty. Accepts value > 0.0. Shape is (1)", "T", OpSchema::Optional)
                                .Input(7, "vocab_mask", "Mask of vocabulary. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (vocab_size)", "M", OpSchema::Optional)
                                .Input(8, "prefix_vocab_mask", "Mask of vocabulary for first step. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (batch_size, vocab_size)", "M", OpSchema::Optional)
                                .Input(9, "attention_mask", "Custom attention mask. Shape is (batch_size, sequence_length)", "I", OpSchema::Optional)
                                .Input(10, "decoder_input_ids", "The forced input id sequence for the decoder subgraph. Shape is (batch_size, initial_sequence_length)", "I", OpSchema::Optional)
                                .Input(11, "logits_processor", "Specific logits processor for different types of beamsearch models. Default value 0 means no specific logit processor. Accepts value >= 0. Shape is (1)", "I", OpSchema::Optional)
                                .Output(0, "sequences", "Word IDs of generated sequences. Shape is (batch_size, num_return_sequences, max_sequence_length)", "I")
                                .Output(1, "sequences_scores", "Final beam score of the generated sequences. Shape is (batch_size, num_return_sequences)", "T", OpSchema::Optional)
                                .Output(2, "scores",
                                        "Processed beam scores for each vocabulary token at each generation step."
                                        "Beam scores consisting of log softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam."
                                        "Shape is (max_length - sequence_length, batch_size, num_beams, vocab_size)",
                                        "T", OpSchema::Optional)
                                .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain to float tensors.")
                                .TypeConstraint("F", {"tensor(float)", "tensor(int32)", "tensor(float16)"}, "Constrain input type to float or int tensors.")
                                .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types")
                                .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to integer types")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  BeamSearchShapeInference(ctx);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(WhisperBeamSearch, 1,
                            OpSchema()
                                .SetDoc("Beam Search for whisper model, especially with cross_qk features etc.")
                                .Attr("eos_token_id", "The id of the end-of-sequence token", AttributeProto::INT)
                                .Attr("pad_token_id", "The id of the padding token", AttributeProto::INT)
                                .Attr("decoder_start_token_id", "The id of the token that indicates decoding starts (i.e. the start of transcription token id)", AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("translate_token_id", "The id of the translate task", AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("transcribe_token_id", "The id of the transcribe task", AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("start_of_lm_token_id", "The id of the token that indicates LM starts", AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("no_speech_token_id",
                                      "The token in whisper model that marks all sequence empty. With this model, whisper could output no_speech_prob after. Default -1.",
                                      AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("no_timestamps_token_id", "The id of the token that indicates no timestamps", AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("beginning_timestamp_token_id", "The id of the first timestamp", AttributeProto::INT, OPTIONAL_VALUE)
                                .Attr("no_repeat_ngram_size", "no repeat ngrams size", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("early_stopping", "early stop or not", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("model_type", "Must be 2 for whisper", AttributeProto::INT, static_cast<int64_t>(2))
                                .Attr("encoder", "The subgraph for initialization of encoder and decoder. It will be called once before decoder subgraph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("init_decoder",
                                      "The subgraph for the first decoding run. It will be called once before `decoder` subgraph. "
                                      "This is relevant only for the GPT2 model. If this attribute is missing, the `decoder` subgraph will be used for all decoding runs",
                                      AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("decoder", "Decoder subgraph to execute in a loop.", AttributeProto::GRAPH)
                                .Attr("vocab_size",
                                      "Size of the vocabulary. "
                                      "If not provided, it will be inferred from the decoder subgraph's output shape",
                                      AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("decoder_output_cross_qk", "If nozero, decoder subgraph contains output Q*K from cross attentions. Default 0.", AttributeProto::INT, OPTIONAL_VALUE)
                                .Input(0, "input_ids", "The sequence used as a prompt for the generation in the encoder subgraph. Shape is (batch_size, sequence_length)", "F")
                                .Input(1, "max_length", "The maximum length of the sequence to be generated. Shape is (1)", "I")
                                .Input(2, "min_length", "The minimum length below which the score of eos_token_id is set to -Inf. Shape is (1)", "I", OpSchema::Optional)
                                .Input(3, "num_beams", "Number of beams for beam search. 1 means no beam search. Shape is (1)", "I")
                                .Input(4, "num_return_sequences", "The number of returned sequences in the batch. Shape is (1)", "I")
                                .Input(5, "length_penalty",
                                       "Exponential penalty to the length. Default value 1.0 means no penalty. "
                                       "Value > 1.0 encourages longer sequences, while values < 1.0 produces shorter sequences. "
                                       "Shape is (1,)",
                                       "T", OpSchema::Optional)
                                .Input(6, "repetition_penalty", "The parameter for repetition penalty. Default value 1.0 means no penalty. Accepts value > 0.0. Shape is (1)", "T", OpSchema::Optional)
                                .Input(7, "vocab_mask", "Mask of vocabulary. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (vocab_size)", "M", OpSchema::Optional)
                                .Input(8, "prefix_vocab_mask", "Mask of vocabulary for first step. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (batch_size, vocab_size)", "M", OpSchema::Optional)
                                .Input(9, "attention_mask", "Custom attention mask. Shape is (batch_size, sequence_length)", "I", OpSchema::Optional)
                                .Input(10, "decoder_input_ids", "The forced input id sequence for the decoder subgraph. Shape is (batch_size, initial_sequence_length)", "I", OpSchema::Optional)
                                .Input(11, "logits_processor", "Specific logits processor for different types of beamsearch models. Default value 0 means no specific logit processor. Accepts value >= 0. Shape is (1)", "I", OpSchema::Optional)
                                .Input(12, "cross_qk_layer_head",
                                       "Only keep this list of (layer, head) of QK in the final cross_qk output when use_cross_qk is set. Default collect all "
                                       "its shape is (number of (layer, head) to keep, 2), i.e., [[layer_id1, head_id1], [layer_id2, head_id2]......]",
                                       "I", OpSchema::Optional)
                                .Input(13, "extra_decoding_ids",
                                       "Part of the decoder_input_ids that we need cross qk for it. it is of shape  (batch_size, extra_decoding_ids_len)."
                                       "In such case, we should remove this from the tail of the decoder_input_ids, and put it here. ids < 0 in it (for multiple batch) "
                                       "are treated as stop of the extra_decoding_ids for corresponding batch.",
                                       "I", OpSchema::Optional)
                                .Input(14, "temperature", "Temperature value to apply to logits processing during this execution's decoding. Shape is (1)", "T", OpSchema::Optional)
                                .Output(0, "sequences", "Word IDs of generated sequences. Shape is (batch_size, num_return_sequences, max_sequence_length)", "I")
                                .Output(1, "sequences_scores", "Final beam score of the generated sequences. Shape is (batch_size, num_return_sequences)", "T", OpSchema::Optional)
                                .Output(2, "scores",
                                        "Processed beam scores for each vocabulary token at each generation step. "
                                        "Beam scores consisting of log softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam. "
                                        "Shape is (max_length - sequence_length, batch_size, num_beams, vocab_size)",
                                        "T", OpSchema::Optional)
                                .Output(3, "cross_qk",
                                        "Output the accumulated stacked Q*K in cross attentions. Let H = number of Head of cross attention, "
                                        "F = the frames or kv-seq-len of the cross attention input, T = real decoded token length, L = number of layers, "
                                        "B = batch size, R = num_return_sequences. It then should return tensor of shape [B, R, L*H, T, F]. "
                                        "If cross_qk_layer_head is given, shape is [B, R, cross_qk_layer_head.shape[0], T, F]",
                                        "V", OpSchema::Optional)
                                .Output(4, "non_speech_probs",
                                        "For whisper model, output the probabilities from logits after encoder and context decoding for the no_speech_token_id. "
                                        "The shape of non_speech_probs is [B]",
                                        "T", OpSchema::Optional)
                                .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain to float tensors.")
                                .TypeConstraint("F", {"tensor(float)", "tensor(int32)", "tensor(float16)"}, "Constrain input type to float or int tensors.")
                                .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types")
                                .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to integer types")
                                .TypeConstraint("V", {"tensor(float)"}, "Constrain cross_qk to float32 tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  BeamSearchShapeInference(ctx);
                                  if (ctx.getNumOutputs() > 3) {
                                    ONNX_NAMESPACE::updateOutputElemType(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT);
                                  }
                                  if (!hasInputShape(ctx, 0)) {
                                    return;
                                  }
                                  auto& input_ids_shape = getInputShape(ctx, 0);
                                  auto& input_ids_dims = input_ids_shape.dim();
                                  int64_t batch_size = input_ids_dims[0].dim_value();
                                  int64_t sequence_length = input_ids_dims[1].dim_value();

                                  const auto max_length = ctx.getInputData(1);
                                  const auto num_return_sequences = ctx.getInputData(4);
                                  if (max_length == nullptr || num_return_sequences == nullptr) {  // not initializer
                                    return;
                                  }
                                  int max_length_value = 0;
                                  if (!ParseScalar(max_length, max_length_value) || max_length_value <= 0) {
                                    fail_shape_inference("Failed to parse max_length or it is not positive integer scalar");
                                  }

                                  int num_return_sequences_value = 0;
                                  if (!ParseScalar(num_return_sequences, num_return_sequences_value) || num_return_sequences_value <= 0) {
                                    fail_shape_inference("Failed to parse num_return_sequences or it is not positive integer scalar");
                                  }

                                  if (ctx.getNumOutputs() > 3) {
                                    ONNX_NAMESPACE::TensorShapeProto cross_attn_shape;
                                    cross_attn_shape.add_dim()->set_dim_value(batch_size);
                                    cross_attn_shape.add_dim()->set_dim_value(num_return_sequences_value);
                                    cross_attn_shape.add_dim();  // num of layer is unknown, no need to calc it from subgraph here
                                    cross_attn_shape.add_dim();  // num of head is unknown, no need to calc it from subgraph here
                                    cross_attn_shape.add_dim()->set_dim_value(max_length_value);
                                    cross_attn_shape.add_dim()->set_dim_value(sequence_length);
                                    updateOutputShape(ctx, 3, cross_attn_shape);
                                  }
                                  if (ctx.getNumOutputs() > 4) {
                                    ONNX_NAMESPACE::TensorShapeProto non_speech_probs_shape;
                                    non_speech_probs_shape.add_dim()->set_dim_value(batch_size);
                                    updateOutputShape(ctx, 4, non_speech_probs_shape);
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(GreedySearch, 1,
                            OpSchema()
                                .SetDoc("Greedy Search for text generation.")
                                .Attr("eos_token_id", "The id of the end-of-sequence token", AttributeProto::INT)
                                .Attr("pad_token_id", "The id of the padding token", AttributeProto::INT)
                                .Attr("decoder_start_token_id", "The id of the token that indicates decoding starts.", AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("no_repeat_ngram_size", "no repeat ngrams size", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("model_type", "model type: 0 for decoder only like GPT-2; 1 for encoder decoder like Bart", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("encoder", "The subgraph for initialization of encoder and decoder. It will be called once before `decoder` subgraph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("init_decoder",
                                      "The subgraph for the first decoding run. It will be called once before `decoder` subgraph. "
                                      "This is relevant only for the GPT2 model. If this attribute is missing, the `decoder` subgraph will be used for all decoding runs",
                                      AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("decoder", "Decoder subgraph to execute in a loop.", AttributeProto::GRAPH)
                                .Attr("vocab_size",
                                      "Size of the vocabulary. "
                                      "If not provided, it will be inferred from the decoder subgraph's output shape",
                                      AttributeProto::INT, static_cast<int64_t>(-1))
                                .Input(0, "input_ids", "The sequence used as a prompt for the generation. Shape is (batch_size, sequence_length)", "I")
                                .Input(1, "max_length", "The maximum length of the sequence to be generated. Shape is (1)", "I")
                                .Input(2, "min_length", "The minimum length below which the score of eos_token_id is set to -Inf. Shape is (1)", "I", OpSchema::Optional)
                                .Input(3, "repetition_penalty", "The parameter for repetition penalty. Default value 1.0 means no penalty. Accepts value > 0.0. Shape is (1)", "T", OpSchema::Optional)
                                .Input(4, "vocab_mask", "Mask of vocabulary. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (vocab_size)", "I", OpSchema::Optional)
                                .Input(5, "prefix_vocab_mask", "Mask of vocabulary for first step. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (batch_size, vocab_size)", "I", OpSchema::Optional)
                                .Input(6, "attention_mask", "Custom attention mask. Shape is (batch_size, sequence_length)", "I", OpSchema::Optional)
                                .Output(0, "sequences", "Word IDs of generated sequences. Shape is (batch_size, max_sequence_length)", "I")
                                // TODO(wy): support scores if needed.
                                .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
                                .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  GreedySearchShapeInference(ctx);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(Sampling, 1,
                            OpSchema()
                                .SetDoc("Greedy Sampling for text generation.")
                                .Attr("eos_token_id", "The id of the end-of-sequence token", AttributeProto::INT)
                                .Attr("pad_token_id", "The id of the padding token", AttributeProto::INT)
                                .Attr("decoder_start_token_id", "The id of the token that indicates decoding starts.", AttributeProto::INT, static_cast<int64_t>(-1))
                                .Attr("no_repeat_ngram_size", "no repeat ngrams size", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("temperature", "The value used to module the next token probabilities.", AttributeProto::FLOAT, 1.0f)
                                .Attr("top_p",
                                      "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.",
                                      AttributeProto::FLOAT, 0.0f)
                                .Attr("filter_value", "All filtered values will be set to this float value.", AttributeProto::FLOAT, -1e20f)
                                .Attr("min_tokens_to_keep", "Minimumber of tokens we keep per batch example in the output.", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("presence_penalty", "Presence penalty for custom sampling", AttributeProto::FLOAT, 0.0f)
                                .Attr("custom", "If 1 custom sampling logic", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("model_type", "Model type: 0 for decoder only like GPT-2; 1 for encoder decoder like Bart", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("encoder", "The subgraph for initialization of encoder and decoder. It will be called once before decoder subgraph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("init_decoder",
                                      "The subgraph for the first decoding run. It will be called once before `decoder` subgraph. "
                                      "This is relevant only for the GPT2 model. If this attribute is missing, the `decoder` subgraph will be used for all decoding runs",
                                      AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("decoder", "Decoder subgraph to execute in a loop.", AttributeProto::GRAPH)
                                .Attr("vocab_size",
                                      "Size of the vocabulary. "
                                      "If not provided, it will be inferred from the decoder subgraph's output shape",
                                      AttributeProto::INT, static_cast<int64_t>(-1))
                                .Input(0, "input_ids", "The sequence used as a prompt for the generation. Shape is (batch_size, sequence_length)", "I")
                                .Input(1, "max_length", "The maximum length of the sequence to be generated. Shape is (1)", "I")
                                .Input(2, "min_length", "The minimum length below which the score of eos_token_id is set to -Inf. Shape is (1)", "I", OpSchema::Optional)
                                .Input(3, "repetition_penalty", "The parameter for repetition penalty. Default value 1.0 means no penalty. Accepts value > 0.0. Shape is (1)", "T", OpSchema::Optional)
                                .Input(4, "vocab_mask", "Mask of vocabulary. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (vocab_size)", "I", OpSchema::Optional)
                                .Input(5, "prefix_vocab_mask", "Mask of vocabulary for first step. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (batch_size, vocab_size)", "I", OpSchema::Optional)
                                .Input(6, "attention_mask", "Custom attention mask. Shape is (batch_size, sequence_length)", "I", OpSchema::Optional)
                                .Input(7, "presence_mask", "Presence penalty mask. Shape is (batch_size, vocab_size)", "I", OpSchema::Optional)
                                .Input(8, "seed", "Seed for random number generator. Shape is (1)", "I", OpSchema::Optional)
                                .Output(0, "sequences", "Word IDs of generated sequences. Shape is (batch_size, max_sequence_length)", "I")
                                .Output(1, "filtered_logits", "Filtered logits as input to the mutinomial function for debug purpose. Shape is (batch_size, vocab_size)", "T", OpSchema::Optional)
                                .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
                                .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  GreedySearchShapeInference(ctx);
                                }));

constexpr const char* MoE_ver1_doc = R"DOC(
      Mixture of experts. Examples: Switch transformer(https://arxiv.org/pdf/2101.03961.pdf) use top 1,
      GLaM(https://arxiv.org/abs/2112.06905) activates top 2 FFN, Vision MOE(https://arxiv.org/pdf/2106.05974.pdf)
      usually uses top 32 experts and Mixtral(https://huggingface.co/blog/mixtral).
      A 2D input is the packed token-major form used by continuous-batching engines: tokens from
      different requests are concatenated along dimension 0 without padding. MoE is token-local,
      so request boundaries do not affect the result and no cumulative sequence-length input is
      required. A 3D input is the dense convenience form and is processed as batch_size *
      sequence_length independent token rows. router_probs must contain one corresponding row per
      token in either form.

      The SwiGLU (Swish-Gated Linear Unit) activation function is like:
         g = xW + b
         l = xV + c
         G = clamp(g, max=limit)
         L = clamp(l, min=-limit, max=limit)
         swiglu = G * sigmoid(alpha * G) * (L + beta)
      where x is the input, W and V are weight matrices, b and c are bias vectors, and alpha, beta and limit are constant float parameters.
      When swiglu_fusion=0, two GEMMs are not fused, and they are FC1 and FC3 in the inputs.
      When swiglu_fusion=1, two GEMMs are fused so that g and l are computed in a single GEMM (FC1), and g and l are interleaved on each row of size 2 * inter_size.
      When swiglu_fusion=2, two GEMMs are fused, and g and l are concatenated on each row.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    MoE, 1,
    OpSchema()
        .SetDoc(MoE_ver1_doc)
        .Attr("activation_type", "Activation function to use. Choose from relu, gelu, silu, swiglu and identity. CUDA also supports relu2 (squared ReLU). Default is relu", AttributeProto::STRING, std::string("relu"))
        .Attr("swiglu_fusion", "0: not fused, 1: fused and interleaved. 2: fused and not interleaved.", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("swiglu_limit", "The limit used to clamp in SwiGLU. No clamp when limit is not provided.", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("activation_alpha", "Alpha parameter used in activation function.", AttributeProto::FLOAT, 1.0f)
        .Attr("activation_beta", "Beta parameter used in activation function.", AttributeProto::FLOAT, 0.0f)
        .Attr("k", "Number of top experts to select from expert pool", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("normalize_routing_weights", "Whether to normalize routing weights", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("use_sparse_mixer", "Whether to use sparse mixer", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "input",
               "2D packed token tensor with shape (total_tokens, hidden_size), where tokens "
               "from ragged sequences may be concatenated without padding, or 3D input tensor with shape "
               "(batch_size, sequence_length, hidden_size)",
               "T")
        .Input(1, "router_probs",
               "2D input tensor with shape (total_tokens, num_experts), where total_tokens "
               "must match the flattened token count of input",
               "T")
        .Input(2, "fc1_experts_weights", "3D input tensor with shape (num_experts, fusion_size * inter_size, hidden_size), where fusion_size is 2 for fused swiglu, and 1 otherwise", "T")
        .Input(3, "fc1_experts_bias", "2D optional input tensor with shape (num_experts, fusion_size * inter_size)", "T", OpSchema::Optional)
        .Input(4, "fc2_experts_weights", "3D input tensor with shape (num_experts, hidden_size, inter_size)", "T")
        .Input(5, "fc2_experts_bias", "2D optional input tensor with shape (num_experts, hidden_size)", "T", OpSchema::Optional)
        .Input(6, "fc3_experts_weights", "3D optional input tensor with shape (num_experts, inter_size, hidden_size)", "T", OpSchema::Optional)
        .Input(7, "fc3_experts_bias", "2D optional input tensor with shape (num_experts, inter_size)", "T", OpSchema::Optional)
        .Output(0, "output",
                "Same shape as input: packed (total_tokens, hidden_size) or padded "
                "(batch_size, sequence_length, hidden_size)",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* qMoE_ver1_doc = R"DOC(
      Quantized mixture of experts (MoE).
      A 2D input is the packed token-major form used by continuous-batching engines: tokens from
      different requests are concatenated along dimension 0 without padding. QMoE is token-local,
      so request boundaries do not affect the result and no cumulative sequence-length input is
      required. A 3D input is the dense convenience form and is processed as batch_size *
      sequence_length independent token rows. router_probs and optional router_weights must contain
      one corresponding row per token in either form.

      The quantized weights are stored in column major order per expert.
      The quantization block size can be specified. If not provided, column wise quantization is used.

      For integer quantization, the formula of linear dequantization using scale and (optionally) zero-point is:
        dequantized_weight = (quantized_weight - zero_point) * scale
      When zero_point is not provided, the default value is 2^(bits-1): 2 for 2 bits, 8 for 4 bits, 128 for 8 bits.

      For integer quantization, if block_size is provided, both hidden_size and inter_size must be divisible by the block size, and
      the dequantization is performed per block of size block_size along the K (input feature) dimension.

      For quant_type='fp8', weights instead use row-major float8e4m3fn tensors shaped [num_experts, N, K].
      A positive block_size selects square block scaling with float32, float16, or bfloat16 fc*_scales tensors shaped
      [num_experts, ceil(N / block_size), ceil(K / block_size)]:
        dequantized_weight[e, n, k] = float(weight[e, n, k]) * scale[e, n / block_size, k / block_size]
      Partial edge blocks are allowed. No zero points or activation scales are accepted.
      Without a positive block_size, FP8 uses the legacy per-expert fc*_global_scale inputs instead.
      Block-scaled FP8 does not use global scales. Activations retain the input type (weight-only quantization).
      On CUDA SM80-or-newer GPUs, SiLU/SwiGLU block-scaled FP8 uses fused weight-only GEMV/GEMM by default.
      These kernels decode and scale FP8 tiles on chip without materializing full FP16/BF16 weight buffers.
      Set ORT_ENABLE_FP8_FUSED=0 before session creation to select the compact dense-dequant fallback.
      Other activation types and unsupported routing sizes retain the fallback.
      The WebGPU block-FP8 kernel supports block_size=128 with float32 scales and rejects projections
      that exceed 32-bit shader addressing or the device's per-dimension dispatch limit.

      Packed byte dimensions are computed as logical_element_count * effective_expert_weight_bits / 8.
      Weight rows must be byte-aligned. Zero-point rows are padded to a whole byte when necessary.

      fc1_expert_weight_bits, fc2_expert_weight_bits, and fc3_expert_weight_bits optionally override
      expert_weight_bits for the corresponding projection. An omitted override inherits expert_weight_bits.
      When SwiGLU is fused, FC3 is stored in FC1 and fc3_expert_weight_bits must be omitted or equal to
      fc1_expert_weight_bits after inheritance.

      The SwiGLU (Swish-Gated Linear Unit) activation function is like:
         g = xW + b
         l = xV + c
         G = clamp(g, max=limit)
         L = clamp(l, min=-limit, max=limit)
         swiglu = G * sigmoid(alpha * G) * (L + beta)
      where x is the input, W and V are weight matrices, b and c are bias vectors, and alpha, beta and limit are constant float parameters.
      When swiglu_fusion=0, two GEMMs are not fused, and they are FC1 and FC3 in the inputs.
      When swiglu_fusion=1, two GEMMs are fused so that g and l are computed in a single GEMM (FC1), and g and l are interleaved on each row of size 2 * inter_size.
      When swiglu_fusion=2, two GEMMs are fused, and g and l are concatenated on each row.
      )DOC";

static void QMoETypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  using namespace ONNX_NAMESPACE;
  const int64_t root_bits = getAttribute(ctx, "expert_weight_bits", 4);
  const int64_t fc1_bits = getAttribute(ctx, "fc1_expert_weight_bits", root_bits);
  const int64_t fc2_bits = getAttribute(ctx, "fc2_expert_weight_bits", root_bits);
  const int64_t fc3_bits = getAttribute(ctx, "fc3_expert_weight_bits", root_bits);
  propagateShapeAndTypeFromFirstInput(ctx);
  if (fc1_bits != 3 && fc2_bits != 3 && fc3_bits != 3) {
    return;
  }
  for (int64_t bits : {root_bits, fc1_bits, fc2_bits, fc3_bits}) {
    if (bits != 2 && bits != 3 && bits != 4 && bits != 8) {
      fail_shape_inference("QMoE INT3 supports only expert weight widths 2, 3, 4, and 8.");
    }
  }

  const auto* quant_type = ctx.getAttribute("quant_type");
  if ((quant_type && quant_type->s() != "int") || getAttribute(ctx, "weights_prepacked", -1) != 0) {
    fail_shape_inference("QMoE INT3 requires integer quantization and explicit weights_prepacked=0 (raw weights).");
  }
  const auto* activation_type_attr = ctx.getAttribute("activation_type");
  if (!activation_type_attr || activation_type_attr->s() != "swiglu" ||
      getAttribute(ctx, "swiglu_fusion", 0) != 1 || fc3_bits != fc1_bits) {
    fail_shape_inference("QMoE INT3 requires activation_type='swiglu', swiglu_fusion=1 and equal effective FC1/FC3 widths after inheritance.");
  }
  const int64_t block_size = getAttribute(ctx, "block_size", 0);
  if (block_size != 32 && block_size != 64 && block_size != 128) {
    fail_shape_inference("QMoE INT3 requires block_size 32, 64, or 128.");
  }
  auto input_type = [&](size_t index) -> const TypeProto* {
    return index < ctx.getNumInputs() ? ctx.getInputType(index) : nullptr;
  };
  for (size_t index : {8, 9, 10, 13, 15, 16, 17, 18, 19, 20}) {
    if (input_type(index)) {
      fail_shape_inference("QMoE INT3 fused raw profile does not accept input ", index, ".");
    }
  }
  const auto* activation = input_type(0);
  const int activation_type = activation && activation->has_tensor_type() ? activation->tensor_type().elem_type() : 0;
  if (activation_type && activation_type != TensorProto::FLOAT16 && activation_type != TensorProto::BFLOAT16) {
    fail_type_inference("QMoE INT3 requires FP16 or BF16 activations.");
  }
  auto check_type = [&](size_t index, int expected, bool required) {
    const auto* type = input_type(index);
    if (!type) {
      if (required) {
        fail_type_inference("QMoE INT3 requires input ", index, ".");
      }
      return;
    }
    if (!type->has_tensor_type() ||
        (expected && type->tensor_type().elem_type() && type->tensor_type().elem_type() != expected)) {
      fail_type_inference("QMoE INT3 input ", index, " has an incompatible tensor element type.");
    }
  };
  for (size_t index : {2, 5, 11, 12}) {
    check_type(index, TensorProto::UINT8, index == 2 || index == 5);
  }
  for (size_t index : {1, 3, 4, 6, 7, 14}) {
    check_type(index, activation_type, index == 1 || index == 3 || index == 6);
  }
  auto shape = [&](size_t index, int rank) -> const TensorShapeProto* {
    if (!input_type(index) || !hasInputShape(ctx, index)) {
      return nullptr;
    }
    const auto& result = getInputShape(ctx, index);
    if (result.dim_size() != rank) {
      fail_shape_inference("QMoE INT3 input ", index, " must have rank ", rank, ".");
    }
    for (const auto& dimension : result.dim()) {
      if (dimension.has_dim_value() && dimension.dim_value() < 0) {
        fail_shape_inference("QMoE INT3 dimensions must be nonnegative.");
      }
    }
    return &result;
  };
  const TensorShapeProto* activation_shape = nullptr;
  if (hasInputShape(ctx, 0)) {
    const auto& candidate = getInputShape(ctx, 0);
    if (candidate.dim_size() != 2 && candidate.dim_size() != 3) {
      fail_shape_inference("QMoE INT3 input must have rank 2 or 3.");
    }
    activation_shape = shape(0, candidate.dim_size());
  }
  const auto* fc1 = shape(2, 3);
  const auto* fc2 = shape(5, 3);
  using Dimension = TensorShapeProto::Dimension;
  const Dimension* hidden = activation_shape ? &activation_shape->dim(activation_shape->dim_size() - 1) : nullptr;
  Dimension experts;
  if (fc1) experts = fc1->dim(0);
  Dimension intermediate;
  if (fc1 && fc1->dim(1).has_dim_value()) {
    if (fc1->dim(1).dim_value() % 2 != 0) {
      fail_shape_inference("QMoE INT3 fused FC1 row count must be even.");
    }
    intermediate.set_dim_value(fc1->dim(1).dim_value() / 2);
  }
  for (const auto* dimension : {hidden, static_cast<const Dimension*>(&intermediate),
                                static_cast<const Dimension*>(&experts)}) {
    if (dimension && dimension->has_dim_value() && dimension->dim_value() <= 0) {
      fail_shape_inference("QMoE INT3 hidden, intermediate, and expert dimensions must be positive.");
    }
  }
  for (const auto* dimension : {hidden, static_cast<const Dimension*>(&intermediate)}) {
    if (dimension && dimension->has_dim_value() && dimension->dim_value() % block_size != 0) {
      fail_shape_inference("QMoE INT3 hidden and intermediate sizes must be divisible by block_size.");
    }
  }
  auto match = [](const Dimension& actual, const Dimension* expected, const char* name) {
    if (expected && expected->has_dim_value() && actual.has_dim_value() &&
        actual.dim_value() != expected->dim_value()) {
      fail_shape_inference("QMoE INT3 incompatible ", name, " dimension.");
    }
  };
  auto packed_bytes = [](int64_t count, int64_t bits) {
    return (count / 8) * bits + ((count % 8) * bits + 7) / 8;
  };
  auto match_experts = [&](const Dimension& actual) {
    match(actual, &experts, "expert count");
    if (actual.has_dim_value()) {
      if (actual.dim_value() <= 0) {
        fail_shape_inference("QMoE INT3 expert count must be positive.");
      }
      experts = actual;
    }
  };
  auto projection = [&](const TensorShapeProto* weights, size_t scale_index, size_t zp_index,
                        size_t bias_index, const Dimension* rows, const Dimension* reduction, int64_t bits) {
    const auto* scales = shape(scale_index, 3);
    const auto* zero_points = shape(zp_index, 3);
    const auto* bias = shape(bias_index, 2);
    Dimension concrete_rows;
    if (rows) concrete_rows = *rows;
    for (const auto* tensor : {weights, scales, zero_points, bias}) {
      if (tensor) {
        match_experts(tensor->dim(0));
        match(tensor->dim(1), &concrete_rows, "projection metadata rows");
        if (tensor->dim(1).has_dim_value()) concrete_rows = tensor->dim(1);
      }
    }
    if (reduction && reduction->has_dim_value()) {
      Dimension expected;
      expected.set_dim_value(packed_bytes(reduction->dim_value(), bits));
      if (weights) match(weights->dim(2), &expected, "packed weight bytes");
      const int64_t blocks = reduction->dim_value() / block_size;
      expected.set_dim_value(blocks);
      if (scales) match(scales->dim(2), &expected, "scale blocks");
      expected.set_dim_value(packed_bytes(blocks, bits));
      if (zero_points) match(zero_points->dim(2), &expected, "packed zero-point bytes");
    }
  };
  projection(fc1, 3, 11, 4, fc1 ? &fc1->dim(1) : nullptr, hidden, fc1_bits);
  projection(fc2, 6, 12, 7, hidden, &intermediate, fc2_bits);
  const auto* router = shape(1, 2);
  const auto* router_weights = shape(14, 2);
  Dimension tokens;
  if (activation_shape) {
    if (activation_shape->dim_size() == 2) {
      tokens = activation_shape->dim(0);
    } else if (activation_shape->dim(0).has_dim_value() && activation_shape->dim(1).has_dim_value()) {
      const int64_t batch = activation_shape->dim(0).dim_value();
      const int64_t sequence = activation_shape->dim(1).dim_value();
      if (sequence != 0 && batch > std::numeric_limits<int64_t>::max() / sequence) {
        fail_shape_inference("QMoE INT3 flattened token count overflows int64.");
      }
      tokens.set_dim_value(batch * sequence);
    }
  }
  for (const auto* tensor : {router, router_weights}) {
    if (tensor) {
      match_experts(tensor->dim(1));
      match(tensor->dim(0), &tokens, "token count");
      if (router) match(tensor->dim(0), &router->dim(0), "router token count");
    }
  }
}

constexpr const char* qMoE_int3_doc = R"DOC(
  The raw INT3 profile extends version 1; execution support is provider-specific.
  For INT3, the requirements below supersede the legacy format descriptions above.
  Effective widths are 2, 3, 4, or 8. Models without INT3 retain the legacy format contracts.
  With INT3, use quant_type='int' (the default), and explicitly set activation_type='swiglu', weights_prepacked=0, swiglu_fusion=1,
  and block_size=32, 64, or 128. Effective FC3 must equal FC1 after independent inheritance.
  H comes from input and I from half the FC1 row count; both must be divisible by block_size.
  Weights are uint8 [E,N,ceil(K*bits/8)], row-local LSB-first bitstreams, without per-block padding.
  Scales are required [E,N,K/block_size], matching FP16/BF16 activations.
  INT3 codes dequantize as (u-4)*scale; scales must be finite and nonnegative.
  Optional INT3 zero points are uint8 [E,N,ceil(3*(K/block_size)/8)]. Each unpacked 3-bit
  zero-point code must equal 4 (binary 100), the unsigned offset representing logical zero;
  this does not mean each packed byte is 4. Omission implies the same zero point 4.
  Unused high padding bits must be zero. INT2/4/8 projections retain their zero-point semantics.
  FC1 rows are interleaved gate/up; FC2 is [E,H,ceil(I*fc2_bits/8)].
  Separate FC3, global scales, and activation scales are not accepted by the INT3 profile.
)DOC";

static OpSchema QMoESchema() {
  constexpr const char* quant_type_doc =
      "Quantization type: 'int' for integer quantization (default), 'fp4' for MXFP4 quantization, "
      "'nvfp4' for NVFP4 quantization, 'fp8' for FP8 e4m3 weight-only quantization, "
      "or 'wfp4afp8' for MXFP4 weight with FP8 activation. "
      "When quant_type is 'fp4' or 'nvfp4', weights are stored in E2M1 FP4 format (2 values per byte), "
      "fc*_scales inputs contain the FP4 block scales, and fc*_global_scale inputs must be provided. "
      "'fp4' uses Float8E8M0 block scales with block_size 32; 'nvfp4' uses Float8E4M3FN block scales "
      "with block_size 16. 'fp8' uses float8e4m3fn weights and, when block_size > 0, "
      "float32, float16, or bfloat16 square-block scales in fc*_scales instead of fc*_global_scale. "
      "The INT3 profile requires integer quantization.";
  auto schema = OpSchema()
                    .SetDoc(std::string(qMoE_ver1_doc) + qMoE_int3_doc)
                    .Attr("activation_type",
                          "Activation function to use. Choose from relu, gelu, silu, swiglu and identity. CUDA also supports relu2 (squared ReLU). Default is relu",
                          AttributeProto::STRING,
                          std::string("relu"))
                    .Attr("k",
                          "Number of top experts to select from expert pool",
                          AttributeProto::INT,
                          static_cast<int64_t>(1))
                    .Attr("normalize_routing_weights",
                          "Whether to normalize routing weights",
                          AttributeProto::INT,
                          static_cast<int64_t>(0))
                    .Attr("use_sparse_mixer",
                          "Whether to use sparse mixer",
                          AttributeProto::INT,
                          static_cast<int64_t>(0))
                    .Attr("expert_weight_bits",
                          "Number of bits used in quantized weights: 2, 3, 4, or 8. Default is 4 bits.",
                          AttributeProto::INT,
                          static_cast<int64_t>(4))
                    .Attr("fc1_expert_weight_bits",
                          "Optional FC1 override for expert_weight_bits. Inherits expert_weight_bits when omitted.",
                          AttributeProto::INT,
                          OPTIONAL_VALUE)
                    .Attr("fc2_expert_weight_bits",
                          "Optional FC2 override for expert_weight_bits. Inherits expert_weight_bits when omitted.",
                          AttributeProto::INT,
                          OPTIONAL_VALUE)
                    .Attr("fc3_expert_weight_bits",
                          "Optional FC3 override for expert_weight_bits. Inherits expert_weight_bits when omitted. "
                          "For fused SwiGLU, the effective FC3 width must equal the effective FC1 width.",
                          AttributeProto::INT,
                          OPTIONAL_VALUE)
                    .Attr("swiglu_fusion",
                          "0: not fused, 1: fused and interleaved. 2: fused and not interleaved.",
                          AttributeProto::INT,
                          static_cast<int64_t>(0))
                    .Attr("swiglu_limit",
                          "The limit used to clamp inputs in SwiGLU. It is infinite when limit is not provided.",
                          AttributeProto::FLOAT,
                          OPTIONAL_VALUE)
                    .Attr("activation_alpha",
                          "Alpha parameter used in activation function.",
                          AttributeProto::FLOAT, 1.0f)
                    .Attr("activation_beta",
                          "Beta parameter used in activation function.",
                          AttributeProto::FLOAT, 0.0f)
                    .Attr("accuracy_level",
                          "Minimum accuracy level of the expert GEMMs on CPU, with the MatMulNBits meaning. For block-wise 4-bit "
                          "experts, 0 (default) or 1 keeps fp32 activations and 4 allows int8 activations (int8 dot-product kernels). "
                          "Block-wise 8-bit experts have no fp32 kernel and use int8 activations at every level. "
                          "Other values are treated as 0.",
                          AttributeProto::INT, static_cast<int64_t>(0))
                    .Attr("block_size",
                          "For integer quantization, size of each quantization block along the K (input feature) dimension. "
                          "Must be power of two and ≥ 16 (e.g., 16, 32, 64, 128). "
                          "For integer and FP4 quantization, both hidden_size and inter_size must be divisible by "
                          "the block size. FP8 with block_size=128 supports partial 128x128 tiles. "
                          "The FP4 modes always use blocking: MXFP4 ('fp4'/'wfp4afp8') is normalized to block_size 32 "
                          "and NVFP4 ('nvfp4') to block_size 16, even when block_size is omitted. "
                          "For FP8 ('fp8'), a positive value instead specifies square blocks along both N and K, "
                          "with floating-point scales shaped [E, ceil(N/block_size), ceil(K/block_size)]; partial blocks are allowed. "
                          "Without a positive value, FP8 uses per-expert global scales. "
                          "For integer quantization ('int'), omitting block_size means there is no blocking "
                          "and a whole column shares one scaling factor. ",
                          AttributeProto::INT,
                          OPTIONAL_VALUE)
                    .Attr("weights_prepacked",
                          "Expert-weight layout selector with quantization-specific meanings. For quant_type='int', "
                          "-1 and 1 select execution-provider-specific prepacked int4/int8 fc1/fc2 layouts. "
                          "0 selects raw [E,N,K/pack] tensors as produced by quantize_matmul_{4,8}bits. "
                          "Uniform INT2 on CUDA also accepts explicit 1 for SM80 CUTLASS W2_A16 bytes, "
                          "retaining logical [E,N,K/4] dimensions and ordinary block scales. It requires "
                          "supported packed GEMV/grouped-GEMM execution; raw fallback is unavailable. "
                          "Mixed-width CUDA INT execution requires 0; INT2 does not accept -1. "
                          "For quant_type='nvfp4' on CUDA, -1 (default) and 0 select legacy N-packed [E,K,N/2] bytes; "
                          "1 selects K-packed row-major [E,N,K/2] bytes while retaining logical [E,K,N/2] dimensions. "
                          "NVFP4 block scales and global scales retain their original layouts and values. "
                          "Ignored for other quantization types. Supported values are -1, 0, and 1; defaults to -1.",
                          AttributeProto::INT,
                          static_cast<int64_t>(-1))
                    .Input(0,
                           "input",
                           "2D packed token tensor with shape (total_tokens, hidden_size), where tokens from ragged sequences "
                           "may be concatenated without padding, or "
                           "3D tensor with shape (batch_size, sequence_length, hidden_size)",
                           "T")
                    .Input(1,
                           "router_probs",
                           "2D tensor with shape (total_tokens, num_experts), where total_tokens must match the flattened "
                           "token count of input",
                           "T")
                    .Input(2,
                           "fc1_experts_weights",
                           "3D tensor with shape (num_experts, fusion_size * inter_size, "
                           "hidden_size * effective_fc1_bits / 8). The last dimension must be byte-aligned. "
                           "The fusion_size is 2 for fused swiglu, or 1 otherwise. effective_fc1_bits is "
                           "fc1_expert_weight_bits when provided, otherwise expert_weight_bits.",
                           "T1")
                    .Input(3,
                           "fc1_scales",
                           "Optional weight scales. For quant_type='int', this is a 2D tensor with shape "
                           "(num_experts, fusion_size * inter_size), or a 3D tensor with shape "
                           "(num_experts, fusion_size * inter_size, hidden_size / block_size) when block_size is provided. "
                           "For quant_type='fp4' or 'wfp4afp8', this is a float8e8m0 MXFP block-scale tensor with shape "
                           "(num_experts, fusion_size * inter_size, hidden_size / 32). "
                           "For quant_type='nvfp4', this is a float8e4m3fn NVFP4 block-scale tensor with shape "
                           "(num_experts, fusion_size * inter_size, hidden_size / 16). "
                           "For quant_type='fp8' and block_size > 0, required float32, float16, or bfloat16 scales with shape "
                           "(num_experts, ceil(fusion_size * inter_size / block_size), ceil(hidden_size / block_size)).",
                           "T2",
                           OpSchema::Optional)
                    .Input(4,
                           "fc1_experts_bias",
                           "2D optional tensor with shape (num_experts, fusion_size * inter_size)", "T", OpSchema::Optional)
                    .Input(5,
                           "fc2_experts_weights",
                           "3D tensor with shape (num_experts, hidden_size, inter_size * effective_fc2_bits / 8). "
                           "The last dimension must be byte-aligned. effective_fc2_bits is fc2_expert_weight_bits "
                           "when provided, otherwise expert_weight_bits.",
                           "T1")
                    .Input(6,
                           "fc2_scales",
                           "Optional weight scales. For quant_type='int', this is a 2D tensor with shape "
                           "(num_experts, hidden_size), or a 3D tensor with shape "
                           "(num_experts, hidden_size, inter_size / block_size) when block_size is provided. "
                           "For quant_type='fp4' or 'wfp4afp8', this is a float8e8m0 MXFP block-scale tensor with shape "
                           "(num_experts, hidden_size, inter_size / 32). "
                           "For quant_type='nvfp4', this is a float8e4m3fn NVFP4 block-scale tensor with shape "
                           "(num_experts, hidden_size, inter_size / 16). "
                           "For quant_type='fp8' and block_size > 0, required float32, float16, or bfloat16 scales with shape "
                           "(num_experts, ceil(hidden_size / block_size), ceil(inter_size / block_size)).",
                           "T2",
                           OpSchema::Optional)
                    .Input(7,
                           "fc2_experts_bias",
                           "2D optional tensor with shape (num_experts, hidden_size)",
                           "T",
                           OpSchema::Optional)
                    .Input(8,
                           "fc3_experts_weights",
                           "3D optional tensor with shape (num_experts, inter_size, hidden_size * effective_fc3_bits / 8). "
                           "The last dimension must be byte-aligned. effective_fc3_bits is fc3_expert_weight_bits "
                           "when provided, otherwise expert_weight_bits.",
                           "T1",
                           OpSchema::Optional)
                    .Input(9,
                           "fc3_scales",
                           "Optional weight scales. For quant_type='int', this is a 2D tensor with shape "
                           "(num_experts, inter_size), or a 3D tensor with shape "
                           "(num_experts, inter_size, hidden_size / block_size) when block_size is provided. "
                           "For quant_type='fp4' or 'wfp4afp8', this is a float8e8m0 MXFP block-scale tensor with shape "
                           "(num_experts, inter_size, hidden_size / 32). "
                           "For quant_type='fp8' and block_size > 0, required when FC3 is present, with floating-point scales shaped "
                           "(num_experts, ceil(inter_size / block_size), ceil(hidden_size / block_size)).",
                           "T2",
                           OpSchema::Optional)
                    .Input(10,
                           "fc3_experts_bias",
                           "2D optional tensor with shape (num_experts, inter_size)",
                           "T",
                           OpSchema::Optional)
                    .Input(11,
                           "fc1_zero_points",
                           "2D tensor with shape (num_experts, ceil(fusion_size * inter_size * effective_fc1_bits / 8)), or "
                           "3D tensor with shape (num_experts, fusion_size * inter_size, "
                           "ceil((hidden_size / block_size) * effective_fc1_bits / 8)) when block_size is provided.",
                           "T1",
                           OpSchema::Optional)
                    .Input(12,
                           "fc2_zero_points",
                           "2D tensor with shape (num_experts, ceil(hidden_size * effective_fc2_bits / 8)), or "
                           "3D tensor with shape (num_experts, hidden_size, "
                           "ceil((inter_size / block_size) * effective_fc2_bits / 8)) when block_size is provided.",
                           "T1",
                           OpSchema::Optional)
                    .Input(13,
                           "fc3_zero_points",
                           "2D optional tensor with shape (num_experts, ceil(inter_size * effective_fc3_bits / 8)), or "
                           "3D optional tensor with shape (num_experts, inter_size, "
                           "ceil((hidden_size / block_size) * effective_fc3_bits / 8)) when block_size is provided.",
                           "T1",
                           OpSchema::Optional)
                    .Input(14,
                           "router_weights",
                           "2D optional tensor with shape (num_tokens, num_experts). "
                           "When provided, router_probs is used only for Top-K expert selection, and router_weights is used "
                           "for aggregating expert outputs (the values at the selected expert indices are gathered and used as "
                           "mixing weights). This enables DeepSeek-style noaux_tc routing where different tensors are used for "
                           "selection and aggregation. When not provided, router_probs is used for both selection and aggregation "
                           "(backward compatible).",
                           "T",
                           OpSchema::Optional)
                    .Input(15,
                           "fc1_global_scale",
                           "1D optional tensor with shape (num_experts,). "
                           "Per-expert global weight scale for FC1. Required when quant_type is 'fp4', 'nvfp4', or 'wfp4afp8', "
                           "or 'fp8' with block_size <= 0. Not used for block-scaled FP8.",
                           "T4",
                           OpSchema::Optional)
                    .Input(16,
                           "fc2_global_scale",
                           "1D optional tensor with shape (num_experts,). "
                           "Per-expert global weight scale for FC2. Required when quant_type is 'fp4', 'nvfp4', or 'wfp4afp8', "
                           "or 'fp8' with block_size <= 0. Not used for block-scaled FP8.",
                           "T4",
                           OpSchema::Optional)
                    .Input(17,
                           "fc1_act_scale",
                           "1D optional tensor with shape (1,) or (num_experts,). Activation scale for FC1 FP8 activation modes.",
                           "T4",
                           OpSchema::Optional)
                    .Input(18,
                           "fc2_act_scale",
                           "1D optional tensor with shape (1,) or (num_experts,). Activation scale for FC2 FP8 activation modes.",
                           "T4",
                           OpSchema::Optional)
                    .Input(19,
                           "fc1_act_block_scale",
                           "3D optional float8e8m0 MXFP activation block-scale tensor for FC1 FP8 activation modes.",
                           "T2",
                           OpSchema::Optional)
                    .Input(20,
                           "fc2_act_block_scale",
                           "3D optional float8e8m0 MXFP activation block-scale tensor for FC2 FP8 activation modes.",
                           "T2",
                           OpSchema::Optional)
                    .Output(0,
                            "output",
                            "output tensor with same shape of input",
                            "T")
                    .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
                    .TypeConstraint("T1", {"tensor(uint8)", "tensor(float8e4m3fn)"},
                                    "Constrain quantized weight types. Integer and FP4 weights use uint8. FP8 weights use float8e4m3fn.")
                    .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(float8e8m0)", "tensor(float8e4m3fn)"},
                                    "Constrain scale types. Float tensors are used for integer quantization and FP8 square-block scales. "
                                    "Float8e8m0 tensors are used for MXFP4 block scales; float8e4m3fn tensors are used for NVFP4 block scales.")
                    .TypeConstraint("T4", {"tensor(float)"}, "Constrain FP4 global scale type to float32 tensors.")
                    .TypeAndShapeInferenceFunction(QMoETypeAndShapeInference);
  schema.Attr("quant_type", quant_type_doc, AttributeProto::STRING, std::string("int"));
  return schema;
}

ONNX_MS_OPERATOR_SET_SCHEMA(QMoE, 1, QMoESchema());

ONNX_MS_OPERATOR_SET_SCHEMA(SampleOp, 1,
                            OpSchema()
                                .Input(0, "X", "input", "T")
                                .Output(0, "Y", "output", "T")
                                .TypeConstraint(
                                    "T",
                                    ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
                                    "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
                                .SetDoc(R"DOC(
Sample echo operator.)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(MaxpoolWithMask, 1,
                            OpSchema()
                                .SetDoc(R"DOC(For internal use.)DOC")
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr("pads",
                                      "",
                                      AttributeProto::INTS, OPTIONAL_VALUE)
                                .Attr(
                                    "storage_order",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(1, "M", "mask", "tensor(int32)")
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float)"}, "Constrain input0 and output types to float tensors")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(Rfft, 1,
                            OpSchema()
                                .SetDoc(R"DOC(This function computes the n-point one dimensional Fourier transform for a real-valued input where n is an even number.)DOC")
                                .Input(0, "X", "input tensor of size n in the signal dim", "T")
                                .Attr("signal_ndim", "number of dimensions comprising the signal, collected in reverse order (e.g. 1 = last dimension is the signal)", AttributeProto::INT, static_cast<int64_t>(1))
                                .Attr("normalized", "must be 0, normalization currently not supported", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("onesided", "must be 1, only one sided FFTs supported", AttributeProto::INT, static_cast<int64_t>(1))
                                .Output(0, "Y", "output tensor of size (n//2 + 1) in the signal dim and 2 in the last dimension for the real and complex parts", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(Irfft, 1,
                            OpSchema()
                                .SetDoc(R"DOC(This function computes the inverse of the one-dimensional n-point RFFT computed in 'com.microsoft.rfft'.)DOC")
                                .Input(0, "X", "input tensor with size (n//2 + 1) in the signal dim and 2 in the last dimension for the real and complex parts", "T")
                                .Attr("signal_ndim", "number of dimensions comprising the signal", AttributeProto::INT)
                                .Attr("normalized", "must be 0, normalization currently not supported", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("onesided", "must be 1, only one sided FFTs supported", AttributeProto::INT, static_cast<int64_t>(1))
                                .Output(0, "Y", "output tensor with size n in the signal dim", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ComplexMul, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "A", "input_0", "T")
                                .Input(1, "B", "input_1", "T")
                                .Output(0, "C", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ComplexMulConj, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "A", "input_0", "T")
                                .Input(1, "B", "input_1", "T")
                                .Output(0, "C", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ConvTransposeWithDynamicPads, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr("output_padding",
                                      "",
                                      AttributeProto::INTS,
                                      OPTIONAL_VALUE)
                                .Attr(
                                    "dilations",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "strides",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "group",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(
                                    1,
                                    "W",
                                    "",
                                    "T")
                                // Pads is required by the kernels, but v1 published it as optional.
                                // Keep the schema compatible and reject a missing tensor at runtime.
                                .Input(2, "Pads", "", "tensor(int64)", OpSchema::Optional)
                                .Input(3, "B", "", "T", OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::convTransposeWithDynamicPadsShapeInference));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedConv, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.)DOC")
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "dilations",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "strides",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "pads",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "group",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Attr(
                                    "activation",
                                    "",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_params",
                                    "",
                                    AttributeProto::FLOATS,
                                    OPTIONAL_VALUE)
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(
                                    1,
                                    "W",
                                    "",
                                    "T")
                                .Input(
                                    2,
                                    "B",
                                    "",
                                    "T",
                                    OpSchema::Optional)
                                .Input(
                                    3,
                                    "Z",
                                    "",
                                    "T",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedGemm, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.)DOC")
                                .Input(
                                    0,
                                    "A",
                                    "Input tensor A. "
                                    "The shape of A should be (M, K) if transA is 0, "
                                    "or (K, M) if transA is non-zero.",
                                    "T")
                                .Input(
                                    1,
                                    "B",
                                    "Input tensor B. "
                                    "The shape of B should be (K, N) if transB is 0, "
                                    "or (N, K) if transB is non-zero.",
                                    "T")
                                .Input(
                                    2,
                                    "C",
                                    "Input tensor C. "
                                    "The shape of C should be unidirectional broadcastable to (M, N).",
                                    "T",
                                    OpSchema::Optional)
                                .Output(0, "Y", "Output tensor of shape (M, N).", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int32)",
                                     "tensor(int64)"},
                                    "Constrain input and output types to float/int tensors.")
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of input tensors A * B.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "beta",
                                    "Scalar multiplier for input tensor C.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "activation",
                                    "",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_alpha",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_beta",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_gamma",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  if (hasNInputShapes(ctx, 2)) {
                                    auto transAAttr = ctx.getAttribute("transA");
                                    bool transA =
                                        transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
                                    auto transBAttr = ctx.getAttribute("transB");
                                    bool transB =
                                        transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
                                    auto& first_input_shape = getInputShape(ctx, 0);
                                    auto& second_input_shape = getInputShape(ctx, 1);
                                    if (first_input_shape.dim_size() != 2)
                                      fail_shape_inference("First input does not have rank 2");
                                    if (second_input_shape.dim_size() != 2)
                                      fail_shape_inference("Second input does not have rank 2");
                                    updateOutputShape(
                                        ctx,
                                        0,
                                        {first_input_shape.dim(transA ? 1 : 0),
                                         second_input_shape.dim(transB ? 0 : 1)});
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(ExpandDims, 1,
                            OpSchema()
                                .Input(0, "X", "input", "T")
                                .Input(1, "axis", "Specified axis to insert a dimension", "tensor(int32)")
                                .Output(0, "Y", "output", "T")
                                .TypeConstraint(
                                    "T",
                                    ONNX_NAMESPACE::OpSchema::all_tensor_types(),
                                    "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  auto& input_shape = getInputShape(ctx, 0);
                                  const int rank = input_shape.dim_size();
                                  const ONNX_NAMESPACE::TensorProto* axis_initializer = ctx.getInputData(1);
                                  if (!axis_initializer)
                                    return;
                                  // Read the scalar axis robustly. ParseScalar handles both raw_data and
                                  // int32_data encodings and validates the element count, avoiding an
                                  // out-of-bounds read when the value is stored as raw_data. A present but
                                  // malformed initializer (wrong element type or not a single scalar) is a
                                  // model error, so fail shape inference rather than silently skipping it.
                                  int axis = 0;
                                  if (!ParseScalar(axis_initializer, axis)) {
                                    fail_shape_inference("Input axis must be a single int32 scalar initializer");
                                  }
                                  if (axis > rank || axis < -rank - 1) {
                                    fail_shape_inference("Input axis is invalid: ", axis);
                                  }
                                  // The output has rank + 1 dimensions, so a negative axis is normalized
                                  // against the output rank: pos = axis + (rank + 1).
                                  int pos = axis >= 0 ? axis : rank + axis + 1;
                                  ONNX_NAMESPACE::TensorShapeProto output_shape;
                                  for (int i = 0; i < pos; ++i) {
                                    output_shape.add_dim();
                                    *(output_shape.mutable_dim(i)) = input_shape.dim(i);
                                  }
                                  output_shape.add_dim();
                                  output_shape.mutable_dim(pos)->set_dim_value(1);
                                  for (int i = pos + 1; i < rank + 1; ++i) {
                                    output_shape.add_dim();
                                    *(output_shape.mutable_dim(i)) = input_shape.dim(i - 1);
                                  }
                                  updateOutputShape(ctx, 0, output_shape);
                                })
                                .SetDoc(R"DOC(ExpandDims echo operator.)DOC"));

constexpr const char* Tokenizer_ver1_doc = R"DOC(
  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by matching and removing separators.
  "separators" is a list of strings which are RE2 regular expressions. "tokenexp" is a single RE2 regular expression.
  Let's assume "separators" is [" "] and consider an example.
  If input is
  ["Hello World", "I love computer science !"] whose shape is [2],
  then the output would be
 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]
 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.
 If "separators" contains a single empty string, the Tokenizer will enter into character tokenization mode. This means all strings
 will be broken apart into individual characters.
 Similarly, if "tokenexp" is set to "." (match any single character), character tokenization mode is used.
 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.
  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
If tokenizer removes the entire content of [C]-input, it will produce [[]].
I.e. the output shape should be [C][0] or [N][C][0] if input shape was [N][C].
If the tokenizer receives empty input of [0] then the output is [0] if empty input
of [N, 0] then [N, 0].
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Tokenizer, 1,
                            OpSchema()
                                .Input(0, "X", "Strings to tokenize", "T")
                                .Output(0, "Y", "Tokenized strings", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(string)"},
                                    "Input/Output is a string tensor")
                                .Attr(
                                    "mark",
                                    "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03).",
                                    AttributeProto::INT)
                                .Attr(
                                    "pad_value",
                                    "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers.",
                                    AttributeProto::STRING)
                                .Attr(
                                    "tokenexp",
                                    "An optional string. Token's regular expression in RE2 format"
                                    " (https://github.com/google/re2/wiki/Syntax)."
                                    " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only one of"
                                    " 'tokenexp' and 'separators' should be set."
                                    " If tokenexp is \".\", the tokenizer enters character tokenization mode.",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "separators",
                                    "an optional list of strings attribute that contains a list of separators - RE2 regular expressions to match separators."
                                    " Two consecutive segments in X connected by a separator would be divided into two tokens."
                                    " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
                                    " the corresponding output would be [\"Hello\", \"World!\"]."
                                    " To achieve character-level tokenization,"
                                    " one should set the 'separators' to [\"\"], which contains an empty string.",
                                    AttributeProto::STRINGS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "mincharnum",
                                    "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored",
                                    AttributeProto::INT)
                                .SetDoc(Tokenizer_ver1_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  ONNX_NAMESPACE::TensorShapeProto output_shape;
                                  auto& input_shape = getInputShape(ctx, 0);
                                  auto& dims = input_shape.dim();
                                  if (dims.size() < 1 || dims.size() > 2) {
                                    fail_shape_inference("Input dimensions are either [C] or [N][C] allowed");
                                  }

                                  int64_t size = 1;
                                  for (auto& dim : dims) {
                                    if (utils::HasDimValue(dim)) {
                                      size *= dim.dim_value();
                                    }
                                  }

                                  if (size > 0) {
                                    for (auto& dim : dims) {
                                      *output_shape.add_dim() = dim;
                                    }
                                    // Add the last unknown dimension
                                    // only if the input is not empty
                                    output_shape.add_dim();
                                  } else if (size == 0) {
                                    if (dims.size() == 2) {
                                      *output_shape.add_dim() = dims[0];
                                    }
                                    output_shape.add_dim()->set_dim_value(0);
                                  }
                                  updateOutputShape(ctx, 0, output_shape);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(MatMulInteger16, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.)DOC")
                                .Input(0, "A", "N-dimensional matrix A", "T1")
                                .Input(1, "B", "N-dimensional matrix B", "T2")
                                .Output(0, "Y", "Matrix multiply results from A * B", "T3")
                                .TypeConstraint("T1", {"tensor(int16)", "tensor(uint16)"}, "Constrain input A data types as 16-bit integer tensor")
                                .TypeConstraint("T2", {"tensor(int16)", "tensor(uint16)"}, "Constrain input B data types as 16-bit integer tensor")
                                .TypeConstraint("T3",
                                                {"tensor(int32)", "tensor(uint32)"},
                                                "Constrain output Y data types as 32-bit integer tensor."
                                                "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint16),"
                                                "or must be tensor(int32) when either T1 or T2 is tensor(int16).")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  auto a_type = ctx.getInputType(0);
                                  auto b_type = ctx.getInputType(1);
                                  auto y_type = ctx.getOutputType(0);
                                  if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
                                      a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
                                      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
                                    fail_type_inference(
                                        "inputs are expected to have tensor type and output type should not be null.");
                                  }

                                  // Right now we only support int32
                                  y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

                                  ONNX_NAMESPACE::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
                                }));

/**
 * @brief Shape inference for MatMul with right hand side matrix quantized into int4
 * @param ctx
 * @param input_a_idx         points to the left hand size matrix input
 * @param input_b_idx         points to the quantized right hand side matrix
 * @param input_bshape_idx    points to the shape tensor of the right hand side matrix
 */
static void matmulQ4ShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input_a_idx, int input_b_idx, int input_bshape_idx, MLAS_BLK_QUANT_TYPE blk_quant_type) {
  if (!hasInputShape(ctx, input_a_idx) || !hasInputShape(ctx, input_b_idx) ||
      !hasInputShape(ctx, input_bshape_idx)) {
    return;
  }

  const auto& a_shape = ctx.getInputType(input_a_idx)->tensor_type().shape();
  if (a_shape.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  const auto& blob_shape = ctx.getInputType(input_b_idx)->tensor_type().shape();
  if (blob_shape.dim_size() != 1 ||
      (blob_shape.dim(0).has_dim_value() && blob_shape.dim(0).dim_value() < 0)) {
    fail_shape_inference("B input for MatMulFpQ4 must be a 1-D tensor.");
  }

  const auto& shape_shape = ctx.getInputType(input_bshape_idx)->tensor_type().shape();
  if (shape_shape.dim_size() != 1 ||
      (shape_shape.dim(0).has_dim_value() && shape_shape.dim(0).dim_value() != 2)) {
    fail_shape_inference("B_shape input for MatMulFpQ4 must be a 1-D int64 tensor of length 2.");
  }

  const TensorProto* b_shape_tensor = ctx.getInputData(input_bshape_idx);
  if (nullptr == b_shape_tensor) {
    // Can't find shape info, quiting
    return;
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  std::vector<int64_t> shape_r_data = ParseData<int64_t>(b_shape_tensor);
  if (shape_r_data.size() != 2) {
    fail_shape_inference("B_shape initializer for MatMulFpQ4 must contain exactly 2 int64 values.");
  }
  if (shape_r_data[0] < 0 || shape_r_data[1] < 0) {
    fail_shape_inference("B_shape initializer for MatMulFpQ4 must contain non-negative dimensions.");
  }
  for (int d = 0; d < 2; d++) {
    shapeR.add_dim()->set_dim_value(shape_r_data[d]);
  }

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (a_shape.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = a_shape.dim(0);
    } else {
      *shapeL.mutable_dim() = a_shape.dim();
    }
  }

  size_t expectedPackSize = MlasQ4GemmPackBSize(
      blk_quant_type,
      static_cast<size_t>(shapeR.dim(shapeR.dim_size() - 1).dim_value()),
      static_cast<size_t>(shapeR.dim(shapeR.dim_size() - 2).dim_value()));
  if (expectedPackSize == 0) {
    fail_shape_inference("4b quantization not yet supported on this hardware platform!");
  }
  if (blob_shape.dim(0).has_dim_value() &&
      static_cast<size_t>(blob_shape.dim(0).dim_value()) != expectedPackSize) {
    fail_shape_inference("Input q4 tensors of wrong size!");
  }

  // Check for compatible matrix multiply dimensions
  {
    const auto& dimL = shapeL.dim(shapeL.dim_size() - 1);
    const auto& dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (a_shape.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

#ifndef ORT_MINIMAL_BUILD
ONNX_MS_OPERATOR_SET_SCHEMA(MatMulFpQ4, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
Matrix product with right hand matrix being pre-packed and quantized int4 data blob.
During quantization, the matrix is divided into blocks, where each block is a
contiguous subset inside each column. Each block is quantized into a
sequence of 4b integers with a scaling factor and an optional offset.
Currently 3 quantization types are supported:
(0): block size 32, no offset, (1): block size 32, with offset, (2): block size 64,
no offset
)DOC")
                                .Attr("blk_quant_type", "Quantization type", AttributeProto::INT, static_cast<int64_t>(1))
                                .Input(0, "A", "N-dimensional matrix A", "T1")
                                .Input(1, "B", "1-dimensional data blob", "T2")
                                .Input(2, "B_shape", "Shape information of B", "T3")
                                .Output(0, "Y", "Matrix multiply results from A * B", "T1")
                                .TypeConstraint("T1", {"tensor(float)"}, "Constrain input matrix data types as single precision float tensor")
                                .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain input B data types as data blob")
                                .TypeConstraint("T3", {"tensor(int64)"}, "Constrain shape of B must be int64 tensor.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  auto a_type = ctx.getInputType(0);
                                  auto b_type = ctx.getInputType(1);
                                  auto b_shape_type = ctx.getInputType(2);
                                  auto y_type = ctx.getOutputType(0);
                                  if (nullptr == a_type || nullptr == b_type || nullptr == b_shape_type || nullptr == y_type ||
                                      a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
                                      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
                                      b_shape_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
                                    fail_type_inference(
                                        "inputs are expected to have tensor type and output type should not be null.");
                                  }

                                  y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
                                  auto blk_quant_v = getAttribute(ctx, "blk_quant_type", 1);
                                  MLAS_BLK_QUANT_TYPE blk_quant_type = blk_quant_v == 0 ? BlkQ4Sym : BlkQ4Zp8;

                                  matmulQ4ShapeInference(ctx, 0, 1, 2, blk_quant_type);
                                }));
#endif

constexpr const char* TransposeMatMul_doc = R"DOC(
Duplicate of FusedMatMul. Going forward FusedMatMul should be used. This OP will be supported for backward compatibility.
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

constexpr const char* FusedMatMul_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

constexpr const char* FusedMatMulActivation_doc = R"DOC(
Executes the same operation as FusedMatMul, but also has an activation function fused to its output.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(TransposeMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "N-dimensional matrix A", "T")
                                .Input(1, "B", "N-dimensional matrix B", "T")
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of the input tensors.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .SetDoc(TransposeMatMul_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  FusedMatMulShapeInference(ctx);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "N-dimensional matrix A", "T")
                                .Input(1, "B", "N-dimensional matrix B", "T")
                                .Attr("alpha", "Scalar multiplier for the product of the input tensors.", AttributeProto::FLOAT, 1.0f)
                                .Attr("transA", "Whether A should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transB", "Whether B should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchA",
                                      "Whether A should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchB",
                                      "Whether B should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                                "Constrain input and output types to float tensors.")
                                .SetDoc(FusedMatMul_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) { FusedMatMulShapeInference(ctx); }));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedMatMulActivation, 1,
                            OpSchema()
                                .Input(0, "A", "N-dimensional matrix A", "T")
                                .Input(1, "B", "N-dimensional matrix B", "T")
                                .Attr("alpha", "Scalar multiplier for the product of the input tensors.", AttributeProto::FLOAT, 1.0f)
                                .Attr("transA", "Whether A should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transB", "Whether B should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchA",
                                      "Whether A should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchB",
                                      "Whether B should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr(
                                    "activation",
                                    "",
                                    AttributeProto::STRING)
                                .Attr(
                                    "activation_alpha",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_beta",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_gamma",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_axis",
                                    "",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Output(0, "Y", "Matrix multiply results", "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                                "Constrain input and output types to float tensors.")
                                .SetDoc(FusedMatMulActivation_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) { FusedMatMulShapeInference(ctx); }));

ONNX_MS_OPERATOR_SET_SCHEMA(SparseToDenseMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "2-dimensional sparse matrix A. Either COO or CSR format", "T")
                                .Input(1, "B", "N-dimensional dense matrix B", "T1")
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of the input tensors.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T1")
                                .TypeConstraint(
                                    "T",
                                    {"sparse_tensor(float)", "sparse_tensor(double)", "sparse_tensor(int64)", "sparse_tensor(int32)",
                                     "sparse_tensor(uint64)", "sparse_tensor(uint32)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)",
                                     "tensor(uint64)", "tensor(uint32)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // 1- dense tensor to output
                                  propagateElemTypeFromInputToOutput(ctx, 1, 0);
                                  // TODO: replace with ONNX one when that one is fixed
                                  sparseCompatibleMatmulShapeInference(ctx, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(MurmurHash3, 1,
                            OpSchema()
                                .SetDoc(R"DOC(The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.)DOC")
                                .Input(0, "X", "An input tensor to hash.", "T1")
                                .Output(0, "Y", "32-bit hash value.", "T2")
                                .TypeConstraint("T1", {"tensor(uint32)", "tensor(int32)", "tensor(uint64)", "tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)"}, "Constrain input type to unsigned or signed 32-bit integer tensor, or string tensor. It should be utf-8 encoded if using unicode.")
                                .TypeConstraint("T2", {"tensor(uint32)", "tensor(int32)"}, "Constrain output type to unsigned and signed 32-bit integer tensor.")
                                .Attr(
                                    "seed",
                                    "Seed for the hashing algorithm, unsigned 32-bit integer, default to 0.",
                                    AttributeProto::INT,
                                    (int64_t)0LL)
                                .Attr(
                                    "positive",
                                    "If value is 1, output type is uint32_t, else int32_t. Default value is 1.",
                                    AttributeProto::INT,
                                    (int64_t)1LL)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // type inference
                                  auto positive_attr = ctx.getAttribute("positive");
                                  bool is_positive =
                                      positive_attr ? (static_cast<int>(positive_attr->i()) == 1 ? true : false) : true /* default value if attribute not present */;
                                  auto output_data_type = ctx.getOutputType(0)->mutable_tensor_type();
                                  if (is_positive) {
                                    output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32);
                                  } else {
                                    output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
                                  }

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  auto& input_shape = getInputShape(ctx, 0);
                                  updateOutputShape(ctx, 0, input_shape);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(GatherND, 1,
                            OpSchema()
                                .Input(0, "data", "Tensor of rank r >= 1.", "T")
                                .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
                                .Output(0, "output", "Tensor of rank q-1+r-indices[-1].", "T")
                                .TypeConstraint(
                                    "T",
                                    OpSchema::all_tensor_types(),
                                    "Constrain input and output types to any tensor type.")
                                .TypeConstraint(
                                    "Tind",
                                    {"tensor(int32)", "tensor(int64)"},
                                    "Constrain indice type to int32 or int64")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  if (!hasNInputShapes(ctx, 2)) {
                                    return;
                                  }
                                  auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
                                  auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
                                  auto data_rank = data_shape.dim_size();
                                  auto indices_rank = indices_shape.dim_size();
                                  if (data_rank < 1 || indices_rank < 1) {
                                    fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
                                  }
                                  auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value();
                                  if (last_indice_dimension > data_rank) {
                                    fail_shape_inference("last dimension of indices must not be larger and rank of data tensor");
                                  }
                                  for (int i = 0; i < indices_rank - 1; ++i) {
                                    *ctx.getOutputType(0)
                                         ->mutable_tensor_type()
                                         ->mutable_shape()
                                         ->add_dim() = indices_shape.dim(i);
                                  }
                                  for (int i = static_cast<int>(last_indice_dimension); i < data_rank; ++i) {
                                    *ctx.getOutputType(0)
                                         ->mutable_tensor_type()
                                         ->mutable_shape()
                                         ->add_dim() = data_shape.dim(i);
                                  }
                                })
                                .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]]
)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(WordConvEmbedding, 1,
                            OpSchema()
                                .Attr(
                                    "embedding_size",
                                    "Integer representing the embedding vector size for each word."
                                    "If not provide, use the filter size of conv weight",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "conv_window_size",
                                    "This operator applies convolution to word from left to right with window equal to conv_window_size and stride to 1."
                                    "Take word 'example' for example, with conv_window_size equal to 2, conv is applied to [ex],[xa], [am], [mp]..."
                                    "If not provide, use the first dimension of conv kernel shape.",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "char_embedding_size",
                                    "Integer representing the embedding vector size for each char."
                                    "If not provide, use the char embedding size of embedding vector.",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Input(0, "Sequence", "Specify batchs of sequence words to embedding", "T")
                                .Input(1, "W", "Specify weights of conv", "T1")
                                .Input(2, "B", "Specify bias of conv", "T1")
                                .Input(3, "C", "Specify embedding vector of char", "T1")
                                .Output(0, "Y", "output", "T1")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(int32)"},
                                    "Constrain to tensor(int32).")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float)"},
                                    "Constrain to tensor(float).")
                                .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(Pad, 1,
                            OpSchema()
                                .Attr(
                                    "mode",
                                    "Three modes: `constant`(default) - pads with a given constant value, "
                                    "`reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis, "
                                    "`edge` - pads with the edge values of array",
                                    AttributeProto::STRING,
                                    std::string("constant"))
                                .Input(0, "data", "Input tensor.", "T")
                                .Input(
                                    1,
                                    "pads",
                                    "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
                                    "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
                                    "`pads` should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]. "
                                    "`pads` format (1D example) should be as follow [x1_begin, x2_begin,...,x1_end, x2_end,...], "
                                    "where xi_begin is the number of pixels added at the beginning of axis `i` and "
                                    "xi_end, the number of pixels added at the end of axis `i`.",
                                    "tensor(int64)")
                                .Input(
                                    2,
                                    "value",
                                    "(Optional) A scalar or rank 1 tensor containing a single value to be filled if the mode chosen is `constant` (by default it is 0.0).",
                                    "T",
                                    OpSchema::Optional)
                                .Output(0, "output", "Tensor after padding.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  // Shape inference needs the input data shape
                                  if (!hasNInputShapes(ctx, 1)) {
                                    return;
                                  }
                                  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
                                  const auto input_rank = input_shape.dim_size();

                                  // Infer output shape if 'pads' tensor is available
                                  const auto* pads_initializer = ctx.getInputData(1);
                                  if (nullptr != pads_initializer) {
                                    const auto& pads_shape = ctx.getInputType(1)->tensor_type().shape();
                                    if ((pads_initializer->dims_size() != 1 &&
                                         pads_initializer->dims_size() != 2) ||
                                        (pads_initializer->dims_size() == 2 &&
                                         pads_shape.dim(static_cast<int>(0)).dim_value() != 1) ||
                                        pads_initializer->data_type() != ONNX_NAMESPACE::TensorProto::INT64)
                                      fail_shape_inference(
                                          "'pads' input must be a 1D (shape: [input_rank]) "
                                          "or 2D tensor (shape: [1, input_rank]) of type int64");

                                    // make a copy of the returned const vector - may have to resize
                                    // this in next step
                                    std::vector<int64_t> pads_data;
                                    if (utils::HasRawData(*pads_initializer))
                                      return;
                                    else
                                      pads_data.insert(
                                          pads_data.end(),
                                          pads_initializer->int64_data().begin(),
                                          pads_initializer->int64_data().end());

                                    // fill with zeros if needed to reach appropriate size
                                    if (pads_data.size() != 2 * static_cast<size_t>(input_rank))
                                      pads_data.resize(size_t{2} * input_rank, 0);

                                    const auto& output_shape =
                                        ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                                    for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
                                      const auto& input_dim = input_shape.dim(static_cast<int>(i));
                                      auto* output_dim = output_shape->add_dim();
                                      if (utils::HasDimValue(input_dim)) {
                                        output_dim->set_dim_value(
                                            input_dim.dim_value() + pads_data[i] + pads_data[i + input_rank]);
                                      } else if (pads_data[i] + pads_data[i + input_rank] == 0) {
                                        *output_dim = input_dim;
                                      }
                                    }
                                  } else {
                                    // Infer output shapes' rank in any case
                                    auto* output_shape_0 = getOutputShape(ctx, 0);
                                    for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
                                      output_shape_0->add_dim();
                                    }
                                  }
                                  return;
                                })
                                .SetDoc(R"DOC(
            Given `data` tensor, pads, mode, and value.
            Example:
            Insert 0 pads to the beginning of the second dimension.
            data = [
                    [1.0, 1.2],
                    [2.3, 3.4],
                    [4.5, 5.7],
                    ]
            pads = [0, 2, 0, 0]
            output = [
                    [
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 2.3, 3.4],
                    [0.0, 0.0, 4.5, 5.7],
                    ],
                    ]
            )DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(Unique, 1,
                            OpSchema()
                                .Input(0, "x", "A 1-D input tensor that is to be processed.", "T")
                                .Output(0, "y",
                                        "A 1-D tensor of the same type as 'x' "
                                        "containing all the unique values in 'x' sorted "
                                        "in the same order that they occur in the input 'x'",
                                        "T")
                                .Output(1, "idx",
                                        "A 1-D INT64 tensor of the same size as 'x' "
                                        "containing the indices for each value in 'x' "
                                        "in the output 'uniques'",
                                        "tensor(int64)")
                                .Output(2, "counts",
                                        "A 1-D INT64 tensor containing the "
                                        "the count of each element "
                                        "of 'uniques' in the input 'x'",
                                        "tensor(int64)")
                                .TypeConstraint("T", OpSchema::all_tensor_types(), "Input can be of any tensor type.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::INT64);
                                  ONNX_NAMESPACE::updateOutputElemType(ctx, 2, ONNX_NAMESPACE::TensorProto::INT64);

                                  // Shape inference

                                  // shape of output 'uniques' and 'counts'
                                  // depends on actual input data, but the rank is always 1
                                  ctx.getOutputType(0)
                                      ->mutable_tensor_type()
                                      ->mutable_shape()
                                      ->add_dim();

                                  ctx.getOutputType(2)
                                      ->mutable_tensor_type()
                                      ->mutable_shape()
                                      ->add_dim();

                                  // if the input shape doesn't exist, further shape inference is not possible
                                  if (!hasNInputShapes(ctx, 1)) {
                                    return;
                                  }

                                  // 'idx' output has same shape as input
                                  ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 1);

                                  return;
                                })
                                .SetDoc(R"DOC(
              Finds all the unique values (deduped list) present in the given input tensor.
              This operator returns 3 outputs.
              The first output tensor 'uniques' contains all of the unique elements of the input,
              sorted in the same order that they occur in the input.
              The second output tensor 'idx' is the same size as the input and it contains the index
              of each value of the input in 'uniques'.
              The third output tensor 'counts' contains the count of each element of 'uniques' in the input.
              Example:
                input_x = [2, 1, 1, 3, 4, 3]
                output_uniques = [2, 1, 3, 4]
                output_idx = [0, 1, 1, 2, 3, 2]
                output_counts = [1, 2, 2, 1]
              )DOC"));

// see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
ONNX_MS_OPERATOR_SET_SCHEMA(CDist, 1,
                            OpSchema()
                                .Attr("metric",
                                      "The distance metric to use. If a string, the distance function can be \"braycurtis\", \"canberra\", "
                                      "\"chebyshev\", \"cityblock\", \"correlation\", \"cosine\", \"dice\", \"euclidean\", \"hamming\", \"jaccard\", "
                                      "\"jensenshannon\", \"kulsinski\", \"mahalanobis\", \"matching\", \"minkowski\", \"rogerstanimoto\", \"russellrao\", "
                                      "\"seuclidean\", \"sokalmichener\", \"sokalsneath\", \"sqeuclidean\", \"wminkowski\", \"yule\".",
                                      AttributeProto::STRING, std::string("sqeuclidean"))
                                .Input(0, "A", "2D matrix with shape (M,N)", "T")
                                .Input(1, "B", "2D matrix with shape (K,N)", "T")
                                .Output(0, "C",
                                        "A 2D Matrix that represents the distance between each pair of the two collections of inputs.",
                                        "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Constrains input to only numeric types."));

ONNX_MS_OPERATOR_SET_SCHEMA(CropAndResize, 1,
                            OpSchema()
                                .Attr(
                                    "mode",
                                    "The pooling method. Two modes are supported: 'bilinear' and 'nearest'. "
                                    "Default is 'bilinear'.",
                                    AttributeProto::STRING,
                                    std::string("bilinear"))
                                .Attr(
                                    "extrapolation_value",
                                    "Value used for extrapolation, when applicable. "
                                    "Default is 0.0f. ",
                                    AttributeProto::FLOAT,
                                    0.f)
                                .Input(
                                    0,
                                    "X",
                                    "Input data tensor from the previous operator; "
                                    "4-D feature map of shape (N, C, H, W), "
                                    "where N is the batch size, C is the number of channels, "
                                    "and H and W are the height and the width of the data.",
                                    "T1")
                                .Input(
                                    1,
                                    "rois",
                                    "RoIs (Regions of Interest) to pool over; rois is "
                                    "2-D input of shape (num_rois, 4) given as "
                                    "[[y1, x1, y2, x2], ...]. "
                                    "The RoIs' coordinates are normalized in the coordinate system of the input image. "
                                    "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.",
                                    "T1")
                                .Input(
                                    2,
                                    "batch_indices",
                                    "1-D tensor of shape (num_rois,) with each element denoting "
                                    "the index of the corresponding image in the batch.",
                                    "T2")
                                .Input(
                                    3,
                                    "crop_size",
                                    "1-D tensor of 2 elements: [crop_height, crop_width]. "
                                    "All cropped image patches are resized to this size. Both crop_height and crop_width need to be positive.",
                                    "T2")
                                .Output(
                                    0,
                                    "Y",
                                    "RoI pooled output, 4-D tensor of shape "
                                    "(num_rois, C, crop_height, crop_width). The r-th batch element Y[r-1] "
                                    "is a pooled feature map corresponding to the r-th RoI X[r-1].",
                                    "T1")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain types to float tensors.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(int32)"},
                                    "Constrain types to int tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  if (!hasNInputShapes(ctx, 4)) {
                                    return;
                                  }
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  auto& input_shape = getInputShape(ctx, 0);
                                  auto& rois_shape = getInputShape(ctx, 1);
                                  auto& batch_index_shape = getInputShape(ctx, 2);
                                  auto& crop_size_shape = getInputShape(ctx, 3);

                                  if (input_shape.dim_size() != 4) {
                                    fail_shape_inference("first input tensor has wrong dimension");
                                  }
                                  if (rois_shape.dim_size() != 2) {
                                    fail_shape_inference("rois input tensor has wrong dimension");
                                  }
                                  if (batch_index_shape.dim_size() != 1) {
                                    fail_shape_inference("batch_indices shape input tensor has wrong dimension");
                                  }
                                  if (crop_size_shape.dim_size() != 1) {
                                    fail_shape_inference("crop_size shape input tensor has wrong dimension");
                                  }
                                  if (crop_size_shape.dim(0).has_dim_value() &&
                                      crop_size_shape.dim(0).dim_value() != 2) {
                                    fail_shape_inference("crop_size input tensor must have exactly 2 elements; got ",
                                                         crop_size_shape.dim(0).dim_value());
                                  }
                                })
                                .SetDoc(R"DOC(
        Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling
        (possibly with aspect ratio change) to a common output size specified by crop_height and crop_width.
        Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.
        The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to
        a fixed size = [crop_height, crop_width]. The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth].
        The resizing is corner aligned.)DOC"));

#if !defined(DISABLE_FLOAT8_TYPES)
#define GEMM_FLOAT8_TYPES \
  {"tensor(float8e4m3fn)", "tensor(float8e5m2)", "tensor(float16)", "tensor(bfloat16)", "tensor(float)"}
#else
#define GEMM_FLOAT8_TYPES \
  {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"}
#endif

ONNX_MS_OPERATOR_SET_SCHEMA(GemmFloat8, 1,
                            OpSchema()
                                .SetDoc(R"DOC(Generic Gemm for float and float 8.)DOC")
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed. Float 8 only supported transA=0.",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed. Float 8 only supported transB=1.",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of input tensors A * B.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "beta",
                                    "Scalar multiplier for the product of input bias C.",
                                    AttributeProto::FLOAT,
                                    0.0f)
                                .Attr(
                                    "dtype",
                                    "Output Type. Same definition as attribute 'to' for operator Cast.",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Attr(
                                    "activation",
                                    "Activation function, RELU or GELU or NONE (default).",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Input(
                                    0,
                                    "A",
                                    "Input tensor A. "
                                    "The shape of A should be (M, K) if transA is 0, "
                                    "or (K, M) if transA is non-zero.",
                                    "TA")
                                .Input(
                                    1,
                                    "B",
                                    "Input tensor B. "
                                    "The shape of B should be (K, N) if transB is 0, "
                                    "or (N, K) if transB is non-zero.",
                                    "TB")
                                .Input(
                                    2,
                                    "C",
                                    "Input tensor C.",
                                    "TC",
                                    OpSchema::Optional)
                                .Input(
                                    3,
                                    "scaleA",
                                    "Scale of tensor A if A is float 8 tensor",
                                    "TS",
                                    OpSchema::Optional)
                                .Input(
                                    4,
                                    "scaleB",
                                    "Scale of tensor B if B is float 8 tensor",
                                    "TS",
                                    OpSchema::Optional)
                                .Input(
                                    5,
                                    "scaleY",
                                    "Scale of the output tensor if A or B is float 8.",
                                    "TS",
                                    OpSchema::Optional)
                                .Output(0, "Y", "Output tensor of shape (M, N).", "TR")
                                .TypeConstraint(
                                    "TA",
                                    GEMM_FLOAT8_TYPES,
                                    "Constrain type to input A.")
                                .TypeConstraint(
                                    "TB",
                                    GEMM_FLOAT8_TYPES,
                                    "Constrain type to input B.")
                                .TypeConstraint(
                                    "TC",
                                    {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"},
                                    "Constrain type to input C.")
                                .TypeConstraint(
                                    "TR",
                                    GEMM_FLOAT8_TYPES,
                                    "Constrain type to result type.")
                                .TypeConstraint("TS", {"tensor(float)"},
                                                "Constrain type for all input scales (scaleA, scaleB, scaleY).")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
                                  if (!hasNInputShapes(ctx, 2)) {
                                    return;
                                  }
                                  auto transAAttr = ctx.getAttribute("transA");
                                  bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
                                  auto transBAttr = ctx.getAttribute("transB");
                                  bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
                                  auto& first_input_shape = getInputShape(ctx, 0);
                                  auto& second_input_shape = getInputShape(ctx, 1);
                                  if (first_input_shape.dim_size() != 2) {
                                    fail_shape_inference("First input does not have rank 2");
                                  }
                                  if (second_input_shape.dim_size() != 2) {
                                    fail_shape_inference("Second input does not have rank 2");
                                  }
                                  updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    MatMulBlockQuantizedFp4Weight, 1,
    OpSchema()
        .SetDoc(R"DOC(Weight-only NVFP4 (E2M1) matrix multiplication.

The weight tensor B is stored as packed NVFP4: two E2M1 values per byte (low nibble first).
The dequantized weight value is `e2m1(B) * weight_scale_2 * e4m3(weight_scale[n, k / block_size])`,
where `weight_scale` holds one E4M3 scale per `block_size` (default 16) consecutive K values and
`weight_scale_2` is a single global fp32 scale. The weight is dequantized to the activation type
(FP16/BF16) and multiplied with the FP16/BF16 activation. This path is architecture independent and
runs on Hopper (SM90) as well as Blackwell.

The output columns `N` and the contraction dimension `K` are derived from the weight shape:
`N = B.shape[0]` and `K = 2 * B.shape[1]`. `K` must therefore be even.)DOC")
        .Attr("block_size", "Number of consecutive K values that share one E4M3 weight scale. Default 16.",
              AttributeProto::INT, static_cast<int64_t>(16))
        .Input(0, "A", "Row-major FP16/BF16 activation of shape [..., K].", "T")
        .Input(1, "B",
               "Packed NVFP4 weight of shape [N, K/2] stored as uint8 (two E2M1 values per byte, low nibble first).",
               "T1")
        .Input(2, "weight_scale",
               "Per-block E4M3 weight scales of shape [N, ceil(K / block_size)] stored as raw uint8 bytes.", "T2")
        .Input(3, "weight_scale_2", "Global fp32 weight scale (scalar).", "T3")
        .Input(4, "input_scale",
               "Optional global fp32 activation scale (scalar). Accepted for parity with quantized checkpoints; "
               "it is a no-op on the weight-only FP16/BF16 path and is reserved for the native NVFP4 path on Blackwell.",
               "T3", OpSchema::Optional)
        .Input(5, "bias", "Optional bias of shape [N].", "T", OpSchema::Optional)
        .Output(0, "Y", "Output of shape [..., N] in the activation type.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)"},
                        "Constrain activation, bias and output to FP16 or BF16.")
        .TypeConstraint("T1", {"tensor(uint8)"}, "Constrain packed NVFP4 weight to uint8.")
        .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain E4M3 weight scales to uint8.")
        .TypeConstraint("T3", {"tensor(float)"}, "Constrain scalar scales to FP32.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const auto& a_shape = getInputShape(ctx, 0);
          const auto& b_shape = getInputShape(ctx, 1);
          if (a_shape.dim_size() < 1 || b_shape.dim_size() != 2) {
            fail_shape_inference("A must have rank at least 1 and B must have rank 2.");
          }
          // B is packed two E2M1 values per byte, so the logical K is twice B's last dimension.
          const auto& a_k = a_shape.dim(a_shape.dim_size() - 1);
          if (a_k.has_dim_value() && b_shape.dim(1).has_dim_value() &&
              a_k.dim_value() != 2 * b_shape.dim(1).dim_value()) {
            fail_shape_inference("A and B have incompatible K dimensions.");
          }
          ONNX_NAMESPACE::TensorShapeProto output_shape;
          for (int i = 0; i < a_shape.dim_size() - 1; ++i) {
            *output_shape.add_dim() = a_shape.dim(i);
          }
          *output_shape.add_dim() = b_shape.dim(0);
          updateOutputShape(ctx, 0, output_shape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    MatMulBlockQuantizedFp8Weight, 1,
    OpSchema()
        .SetDoc(R"DOC(Block-scaled FP8 (E4M3) matrix multiplication with optional FP8 activation quantization.

The weight tensor B has shape [N, K] with one FP32 scale per `block_size` consecutive K values
(`b_scale` of shape [N, ceil(K / block_size)]). The scaled weight value is
`B_scaled[n, k] = fp8_e4m3(B[n, k]) * b_scale[n, k / block_size]`.

When the optional scalar `a_scale` is provided, the activation values used in the multiplication
are `A_scaled = fp8_e4m3(A / a_scale) * a_scale` (W8A8). Otherwise, A retains its FP16/BF16
precision (weight-only W8A16).

The operator multiplies the activation by the transpose of B_scaled and adds the optional bias.
The output has shape [..., N] and the same element type as A. When split_sizes is provided,
return up to eight consecutive last-dimension partitions instead, equivalent to Split(Y, split_sizes).
The partition widths must be positive and sum to N.)DOC")
        .Attr("block_size", "Number of consecutive K values that share one weight scale. Default 128.",
              AttributeProto::INT, static_cast<int64_t>(128))
        .Attr("split_sizes", "Optional last-dimension output widths (one to eight positive sizes).",
              AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("shared_expert",
              "When one, BF16 A is the up branch and shared_silu_gate is the equal-shaped gate branch. "
              "Multiply A by BF16(shared_silu_gate * BF16(sigmoid(shared_silu_gate))) before projection. "
              "Round the projection to BF16, multiply by the BF16 sigmoid of the BF16 dot product "
              "shared_gate_activation @ shared_gate_weight, round to BF16, then add shared_residual. "
              "Sigmoid uses BF16 intermediates. Requires all shared inputs and excludes bias, a_scale, "
              "split_sizes and original prefill weights. Default zero preserves ordinary projection.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("prefill_weight_offsets",
              "Output-column offsets for optional original-type [K, segmentN] prefill weights. "
              "Used only when prefill_numerics is one; decode retains FP8 weights.",
              AttributeProto::INTS, std::vector<int64_t>{})
        .Input(0, "A", "Row-major FP16/BF16 activation of shape [..., K].", "T")
        .Input(1, "B", "Row-major FP8 E4M3 weight of shape [N, K].", "T1")
        .Input(2, "b_scale", "Per-block FP32 weight scales of shape [N, ceil(K / block_size)].", "T2")
        .Input(3, "a_scale",
               "Optional global fp32 activation scale (scalar). When present, A is statically "
               "quantized to FP8 E4M3 with this scale (W8A8 numerics); when absent, A retains "
               "its FP16/BF16 precision.",
               "T2", OpSchema::Optional)
        .Input(4, "bias", "Optional bias of shape [N].", "T", OpSchema::Optional)
        .Input(5, "prefill_numerics",
               "Optional CUDA CPU-memory int32 control of shape [1]. Zero or omission preserves existing "
               "dispatch; one selects dense BF16 dequantization/GEMM rounding during eager prefill. "
               "Decode CUDA graph capture must bind an immutable zero.",
               "TI", OpSchema::Optional)
        .Input(6, "prefill_weight", "Optional original-type [K, segmentN] eager-prefill weight.", "T", OpSchema::Optional)
        .Input(7, "prefill_weight2", "Optional second original-type eager-prefill weight segment.", "T", OpSchema::Optional)
        .Input(8, "prefill_weight3", "Optional third original-type eager-prefill weight segment.", "T", OpSchema::Optional)
        .Input(9, "shared_silu_gate", "BF16 gate branch matching A, for shared_expert=1.", "T", OpSchema::Optional)
        .Input(10, "shared_gate_activation", "BF16 projected-gate activation with matching leading axes.", "T", OpSchema::Optional)
        .Input(11, "shared_gate_weight", "BF16 [gate_input_width, 1] projected-gate weight.", "T", OpSchema::Optional)
        .Input(12, "shared_residual", "BF16 residual matching the unsplit output.", "T", OpSchema::Optional)
        .Output(0, "Y", "Output(s) of shape [..., N] or [..., split_sizes[i]] in the activation type.",
                "T", OpSchema::Variadic)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)"},
                        "Constrain activation, bias and output to FP16 or BF16.")
        .TypeConstraint("T1", {"tensor(float8e4m3fn)"}, "Constrain weight to FP8 E4M3.")
        .TypeConstraint("T2", {"tensor(float)"}, "Constrain scales to FP32.")
        .TypeConstraint("TI", {"tensor(int32)"}, "CPU prefill numerical-mode control.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          const auto* split = ctx.getAttribute("split_sizes");
          const auto shared = getAttribute(ctx, "shared_expert", 0);
          if (shared != 0 && shared != 1) {
            fail_shape_inference("shared_expert must be 0 or 1.");
          }
          if (shared) {
            if (split != nullptr || (ctx.getNumInputs() > 3 && ctx.getInputType(3)) ||
                (ctx.getNumInputs() > 4 && ctx.getInputType(4)) ||
                (ctx.getAttribute("prefill_weight_offsets") &&
                 ctx.getAttribute("prefill_weight_offsets")->ints_size() != 0)) {
              fail_shape_inference("shared_expert excludes split_sizes, bias, a_scale and prefill weights.");
            }
            if (ctx.getInputType(0) &&
                ctx.getInputType(0)->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto::BFLOAT16) {
              fail_type_inference("shared_expert requires BF16.");
            }
            for (size_t index = 9; index <= 12; ++index) {
              if (ctx.getNumInputs() <= index || ctx.getInputType(index) == nullptr) {
                fail_shape_inference("shared_expert requires all shared inputs.");
              }
              if (ctx.getInputType(index)->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto::BFLOAT16) {
                fail_type_inference("shared_expert requires BF16 shared inputs.");
              }
            }
            for (size_t index = 6; index <= 8 && index < ctx.getNumInputs(); ++index) {
              if (ctx.getInputType(index)) {
                fail_shape_inference("shared_expert excludes original prefill weights.");
              }
            }
          } else {
            for (size_t index = 9; index < ctx.getNumInputs(); ++index) {
              if (ctx.getInputType(index)) {
                fail_shape_inference("Shared expert inputs require shared_expert=1.");
              }
            }
          }
          if ((split == nullptr && ctx.getNumOutputs() != 1) ||
              (split != nullptr && (split->ints_size() < 1 || split->ints_size() > 8 ||
                                    static_cast<size_t>(split->ints_size()) != ctx.getNumOutputs()))) {
            fail_shape_inference("Output count must match split_sizes, or be one when split_sizes is absent.");
          }
          int64_t split_total = 0;
          if (split != nullptr) {
            for (const auto width : split->ints()) {
              if (width <= 0 || width > INT64_MAX - split_total) {
                fail_shape_inference("split_sizes must contain positive widths with no sum overflow.");
              }
              split_total += width;
            }
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const auto& a_shape = getInputShape(ctx, 0);
          const auto& b_shape = getInputShape(ctx, 1);
          if (a_shape.dim_size() < 1 || b_shape.dim_size() != 2) {
            fail_shape_inference("A must have rank at least 1 and B must have rank 2.");
          }
          if (a_shape.dim(a_shape.dim_size() - 1).has_dim_value() && b_shape.dim(1).has_dim_value() &&
              a_shape.dim(a_shape.dim_size() - 1).dim_value() != b_shape.dim(1).dim_value()) {
            fail_shape_inference("A and B have incompatible K dimensions.");
          }
          ONNX_NAMESPACE::TensorShapeProto output_shape;
          for (int i = 0; i < a_shape.dim_size() - 1; ++i) {
            *output_shape.add_dim() = a_shape.dim(i);
          }
          *output_shape.add_dim() = b_shape.dim(0);
          if (shared) {
            const auto check_shape = [&](size_t input, const ONNX_NAMESPACE::TensorShapeProto& expected) {
              if (!hasInputShape(ctx, input)) return;
              const auto& actual = getInputShape(ctx, input);
              if (actual.dim_size() != expected.dim_size()) {
                fail_shape_inference("shared_expert input ranks are incompatible.");
              }
              for (int axis = 0; axis < actual.dim_size(); ++axis) {
                if (actual.dim(axis).has_dim_value() && expected.dim(axis).has_dim_value() &&
                    actual.dim(axis).dim_value() != expected.dim(axis).dim_value()) {
                  fail_shape_inference("shared_expert input shapes are incompatible.");
                }
              }
            };
            check_shape(9, a_shape);
            check_shape(12, output_shape);
            if (hasInputShape(ctx, 10)) {
              auto gate_shape = a_shape;
              const auto& activation = getInputShape(ctx, 10);
              if (activation.dim_size() != a_shape.dim_size()) {
                fail_shape_inference("shared_expert gate activation rank must match A.");
              }
              *gate_shape.mutable_dim(gate_shape.dim_size() - 1) = activation.dim(activation.dim_size() - 1);
              check_shape(10, gate_shape);
              ONNX_NAMESPACE::TensorShapeProto weight_shape;
              *weight_shape.add_dim() = activation.dim(activation.dim_size() - 1);
              weight_shape.add_dim()->set_dim_value(1);
              check_shape(11, weight_shape);
            }
          }
          if (split != nullptr && b_shape.dim(0).has_dim_value() &&
              split_total != b_shape.dim(0).dim_value()) {
            fail_shape_inference("split_sizes must sum to the weight's N dimension.");
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            if (split != nullptr) {
              output_shape.mutable_dim(output_shape.dim_size() - 1)->set_dim_value(split->ints(static_cast<int>(i)));
            }
            updateOutputShape(ctx, i, output_shape);
          }
        }));

static void MatmulWithQuantWeightShapeInference(ONNX_NAMESPACE::InferenceContext& ctx,
                                                int64_t K,
                                                int64_t N,
                                                bool transB) {
  int input_a_idx = 0;
  if (!hasInputShape(ctx, input_a_idx)) {
    return;
  }

  const auto& a_shape = ctx.getInputType(input_a_idx)->tensor_type().shape();
  if (a_shape.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  // TODO: check B shape

  const auto& dim_last = a_shape.dim(a_shape.dim_size() - 1);
  ONNX_NAMESPACE::TensorShapeProto resultShape;
  if (dim_last.has_dim_value() && dim_last.dim_value() != (transB ? K : N)) {
    fail_shape_inference("Incompatible dimensions for matrix multiplication");
  }

  for (int i = 0; i < a_shape.dim_size() - 1; ++i) {
    *resultShape.add_dim() = a_shape.dim(i);
  }
  resultShape.add_dim()->set_dim_value(transB ? N : K);

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

void RegisterContribSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalization)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalization")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT, 1e-5f)
      .Attr("stash_type",
            "type used for stash mean/inv_std_var",
            AttributeProto::INT, static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "Scale", "Scale tensor.", "V")
      .Input(2, "B", "Bias tensor.", "V", OpSchema::Optional)
      .Output(0, "Y", "Output data tensor.", "V")
      .Output(1, "Mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
      .Output(2, "InvStdDev", "Saved inverse standard deviation used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input X type to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)", "tensor(double)"},
          "Type of Mean and InvStdDev tensors.")
      .TypeConstraint(
          "V",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain output Y, scale and bias type to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        auto type = ctx.getAttribute("stash_type")->i();
        if (ctx.getNumOutputs() > 1) {
          auto output_type = ctx.getOutputType(1);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (ctx.getNumOutputs() > 2) {
          auto output_type = ctx.getOutputType(2);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
          axis = axis_proto->i();
        }
        axis = HandleNegativeAxis(axis, input_ndim);

        if (ctx.getNumOutputs() > 1) {
          auto saved_mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          saved_mean_shape->CopyFrom(input_shape);
          for (int64_t d = axis; d < input_ndim; ++d)
            saved_mean_shape->mutable_dim(static_cast<int>(d))->set_dim_value(1);
        }

        if (ctx.getNumOutputs() > 2) {
          auto saved_inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
          saved_inv_std_dev_shape->CopyFrom(input_shape);
          for (int64_t d = axis; d < input_ndim; ++d)
            saved_inv_std_dev_shape->mutable_dim(static_cast<int>(d))->set_dim_value(1);
        }
      })
      .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
            // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)

            auto* tp = ctx.getInputType(1);
            if ((tp == nullptr) || (!tp->has_tensor_type()))
              return false;
            int64_t V = tp->tensor_type().elem_type();

            auto type_attr = ctx.getAttribute("stash_type");
            int64_t U = (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
              return false;  // Error

            auto* axis_attr = ctx.getAttribute("axis");
            int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
            auto* epsilon_attr = ctx.getAttribute("epsilon");
            float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

            auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
              auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
              tp.add_dims(1);
              return tp;
            };

            // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
            // This complicates the function definition, requiring reshaping inputs/outputs.
            // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
            // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
            // Normalization is applied to the second dimension.
            // Output Y has same shape as X
            // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
            FunctionBuilder builder(functionProto);
            builder
                .AddOpset("", 13)
                .Const("Epsilon", ToTensor(epsilon, (TensorProto_DataType)U))
                .Add("XShape = Shape (X)")                                                    // shape of input tensor: 1D tensor
                .Add("Rank = Size (XShape)")                                                  // rank of input tensor: scalar
                .Add("Zero1D = Constant()", "value", mktensor(0))                             // [0] : 1D tensor
                .Add("Axis1D = Constant()", "value", mktensor(axis))                          // [axis] : 1D tensor
                .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)")                          // [d[0], ..., d[axis-1]]
                .Add(axis > 0                                                                 // number of axes that are reduced =
                         ? "NumReducedAxes = Sub (Rank, Axis1D)"                              // [rank - axis]: 1D tensor
                         : "NumReducedAxes = Neg (Axis1D)")                                   // [-axis] : 1D tensor
                .Add("SuffixShape = ConstantOfShape (NumReducedAxes)", "value", mktensor(1))  // [1, ..., 1] for reduced axes
                .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)")           // [d[0], ..., d[axis-1], 1, ..., 1]
                .Add("X2D = Flatten (X)", "axis", axis)
                .Add("XU = Cast (X2D)", "to", U)
                .Add("Mean2D = ReduceMean <axes = [1]> (XU)")
                .Add("Square = Mul (XU, XU)")
                .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)")
                .Add("SquareOfMean = Mul (Mean2D, Mean2D)")
                .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
                .Add("VarPlusEpsilon = Add (Var, Epsilon)")
                .Add("StdDev = Sqrt (VarPlusEpsilon)")
                .Add("Deviation = Sub (XU, Mean2D)")
                .Add("Normalized = Div (Deviation, StdDev)")
                .Add("NormalizedV = Cast (Normalized)", "to", V)
                .Add("Scale2D = Flatten <axis = 0> (Scale)")
                .Add("Scaled = Mul (NormalizedV, Scale2D)");
            if (ctx.hasInput(2)) {
              builder.Add("B2D = Flatten <axis=0> (B)");
              builder.Add("Biased = Add (Scaled, B2D)");
            } else {
              builder.Add("Biased = Identity (Scaled)");
            }
            builder.Add("Y = Reshape (Biased, XShape)");
            builder.Add("InvStdDev2D = Reciprocal (StdDev)");
            if (ctx.hasOutput(1))
              builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
            if (ctx.hasOutput(2))
              builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

            schema.BuildFunction(functionProto);
            return true;
          });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SimplifiedLayerNormalization)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SimplifiedLayerNormalization")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT, 1e-5f)
      .Attr("stash_type",
            "type used for stash mean/inv_std_var",
            AttributeProto::INT, static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "scale", "Scale tensor.", "V")
      .Output(0, "Y", "Output data tensor.", "V")
      .Output(1, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input X type to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)", "tensor(double)"},
          "Constrain mean and inv_std_var to be float tensors.")
      .TypeConstraint(
          "V",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain output Y and scale type to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        auto type = ctx.getAttribute("stash_type")->i();
        if (ctx.getNumOutputs() > 1) {
          auto output_type = ctx.getOutputType(1);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
          axis = axis_proto->i();
        }
        axis = HandleNegativeAxis(axis, input_ndim);

        if (ctx.getNumOutputs() > 1) {
          auto saved_inv_std_var_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          saved_inv_std_var_shape->CopyFrom(input_shape);
          saved_inv_std_var_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }
      });

  // ORT will not regsiter TRT plugins as contrib ops, instead it will use custom ops handled by TRT EP.
  // In order not to break the old models using those TRT plugins which were registered with ONNX domain and maintain backward compatible,
  // we still keep EfficientNMS_TRT, MultilevelCropAndResize_TRT, PyramidROIAlign_TRT and DisentangledAttention_TRT as legacy code.
  // We don't need to add new schema definition when a new TRT plugin is introduced, TRT EP will register it as custom op for us.
  // Moving forward, please create TRT plugin node with "trt.plugins" domain.

  static const char* EfficientNMS_TRT_ver1_doc =
      R"DOC(Efficient NMS TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(EfficientNMS_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(EfficientNMS_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "scores", "The scores input tensor.", "T")
      .Input(2, "anchors", "The anchors input tensor.", "T", OpSchema::Optional)
      .Output(0, "num_detections", "The num_detections output tensor.", "tensor(int32)")
      .Output(1, "detection_boxes", "The detection_boxes output tensor.", "T")
      .Output(2, "detection_scores", "The detection_scores output tensor.", "T")
      .Output(3, "detection_classes", "The detection_classes output tensor.", "tensor(int32)")
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
      .Attr("background_class", "Background class ID.", AttributeProto::INT)
      .Attr("box_coding", "Encoding type for the boxes or anchors inputs.", AttributeProto::INT)
      .Attr("iou_threshold", "Box IOU threshold value.", AttributeProto::FLOAT)
      .Attr("max_output_boxes", "Max detections to output.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .Attr("score_activation", "Activation function to apply to the scores input.", AttributeProto::INT)
      .Attr("score_threshold", "Score threshold value.", AttributeProto::FLOAT)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        using namespace ONNX_NAMESPACE;
        ONNX_NAMESPACE::updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT32);
        propagateElemTypeFromInputToOutput(ctx, 0, 1);
        propagateElemTypeFromInputToOutput(ctx, 0, 2);
        ONNX_NAMESPACE::updateOutputElemType(ctx, 3, ONNX_NAMESPACE::TensorProto::INT32);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t max_output_boxes = 1;
        auto max_output_boxes_proto = ctx.getAttribute("max_output_boxes");
        if (max_output_boxes_proto) {
          max_output_boxes = max_output_boxes_proto->i();
        }
        if (max_output_boxes < 1) {
          fail_shape_inference("Attribute 'max_output_boxes' must be >= 1.")
        }

        Dim batch_size;
        unifyInputDim(ctx, 0, 0, batch_size);

        ONNX_NAMESPACE::TensorShapeProto num_detections_shape;
        *num_detections_shape.add_dim() = batch_size;
        num_detections_shape.add_dim()->set_dim_value(1);
        updateOutputShape(ctx, 0, num_detections_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_boxes_shape;
        *detection_boxes_shape.add_dim() = batch_size;
        detection_boxes_shape.add_dim()->set_dim_value(max_output_boxes);
        detection_boxes_shape.add_dim()->set_dim_value(4);
        updateOutputShape(ctx, 1, detection_boxes_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_scores_shape;
        *detection_scores_shape.add_dim() = batch_size;
        detection_scores_shape.add_dim()->set_dim_value(max_output_boxes);
        updateOutputShape(ctx, 2, detection_scores_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_classes_shape;
        *detection_classes_shape.add_dim() = batch_size;
        detection_classes_shape.add_dim()->set_dim_value(max_output_boxes);
        updateOutputShape(ctx, 3, detection_classes_shape);
      });

  static const char* MultilevelCropAndResize_TRT_ver1_doc =
      R"DOC(Multilevel Crop and Resize TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MultilevelCropAndResize_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(MultilevelCropAndResize_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "feature_map_0", "The first feature map input tensor.", "T")
      .Input(2, "feature_map_1", "The second feature map input tensor.", "T")
      .Input(3, "feature_map_2", "The third feature map input tensor.", "T")
      .Input(4, "feature_map_3", "The fourth feature map input tensor.", "T")
      .Output(0, "patches", "The cropped patches output tensor.", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
      .Attr("image_size", "Image size.", AttributeProto::INTS)
      .Attr("pooled_size", "Pooled size.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t pooled_size = 1;
        auto pooled_size_proto = ctx.getAttribute("pooled_size");
        if (pooled_size_proto) {
          pooled_size = pooled_size_proto->i();
        }
        if (pooled_size < 1) {
          fail_shape_inference("Attribute 'pooled_size' must be >= 1.")
        }

        Dim batch_size, number_boxes, channels;
        unifyInputDim(ctx, 0, 0, batch_size);
        unifyInputDim(ctx, 0, 1, number_boxes);
        unifyInputDim(ctx, 1, 1, channels);

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = batch_size;
        *output_shape.add_dim() = number_boxes;
        *output_shape.add_dim() = channels;
        output_shape.add_dim()->set_dim_value(pooled_size);
        output_shape.add_dim()->set_dim_value(pooled_size);
        updateOutputShape(ctx, 0, output_shape);
      });

  static const char* PyramidROIAlign_TRT_ver1_doc =
      R"DOC(Pyramid ROI Align TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(PyramidROIAlign_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(PyramidROIAlign_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "feature_map_0", "The first feature map input tensor.", "T")
      .Input(2, "feature_map_1", "The second feature map input tensor.", "T")
      .Input(3, "feature_map_2", "The third feature map input tensor.", "T")
      .Input(4, "feature_map_3", "The fourth feature map input tensor.", "T")
      .Output(0, "patches", "The cropped patches output tensor.", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
      .Attr("pooled_size", "Pooled size.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t pooled_size = 1;
        auto pooled_size_proto = ctx.getAttribute("pooled_size");
        if (pooled_size_proto) {
          pooled_size = pooled_size_proto->i();
        }
        if (pooled_size < 1) {
          fail_shape_inference("Attribute 'pooled_size' must be >= 1.")
        }

        Dim batch_size, number_boxes, channels;
        unifyInputDim(ctx, 0, 0, batch_size);
        unifyInputDim(ctx, 0, 1, number_boxes);
        unifyInputDim(ctx, 1, 1, channels);

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = batch_size;
        *output_shape.add_dim() = number_boxes;
        *output_shape.add_dim() = channels;
        output_shape.add_dim()->set_dim_value(pooled_size);
        output_shape.add_dim()->set_dim_value(pooled_size);
        updateOutputShape(ctx, 0, output_shape);
      });

  static const char* DisentangledAttention_TRT_ver1_doc =
      R"DOC(Disentangled Attention TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(DisentangledAttention_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(DisentangledAttention_TRT_ver1_doc)
      .Input(0, "c2c_attention", "content-to-content attention tensor, QcKc^T.", "T")
      .Input(1, "c2p_attention", "content-to-position attention tensor, QcKr^T.", "T")
      .Input(2, "p2c_attention", "position-to-content attention tensor, KcQr^T.", "T")
      .Output(0, "disentangled_attention", "The disentangled attention output tensor.", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
      .Attr("span", "Maximum relative distance, k.", AttributeProto::INT)
      .Attr("factor", "Scaling factor applied to attention values, 1/sqrt(3d). d is hidden size per head = H/N. H is hidden size, N is number of heads.", AttributeProto::FLOAT)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        using namespace ONNX_NAMESPACE;
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }

        auto& input0_shape = getInputShape(ctx, 0);
        auto& input0_dims = input0_shape.dim();
        if (input0_dims.size() != 3) {
          fail_shape_inference("Input 0 shall be 3 dimensions");
        }

        // output dims is same as input[0] dims, i.e., regular c2c attention dims
        // ONNX_NAMESPACE::TensorShapeProto disentangled_attention_shape;
        // for (auto& dim : input0_dims) {
        //   *disentangled_attention_shape.add_dim() = dim;
        // }
        // updateOutputShape(ctx, 0, disentangled_attention_shape);
        propagateShapeFromInputToOutput(ctx, 0, 0);
      });

  // Please note that we don't need to add new schema definition when a new TRT plugin is introduced, TRT EP will register it as custom op for us.

  ONNX_CONTRIB_OPERATOR_SCHEMA(Snpe)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Onnx node for SNPE.")
      .Attr("DLC", "payload of the SNPE DLC file.", AttributeProto::STRING)
      .Attr(
          "snpe_version",
          "(Optional) SNPE version used to convert the model.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr("target_device", "(Optional) Target device like CPU, DSP, etc.", AttributeProto::STRING, OPTIONAL_VALUE)
      .Attr("notes", "(Optional) Some notes for the model", AttributeProto::STRING, OPTIONAL_VALUE)
      .AllowUncheckedAttributes()
      .Input(
          0,
          "inputs",
          "List of tensors for SNPE DLC input",
          "T",
          OpSchema::Variadic,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(
          0,
          "outputs",
          "One or more outputs, list of tensors for DLC output",
          "T",
          OpSchema::Variadic,
          true,
          1,
          OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(uint16)", "tensor(float)"},
          "Constrain input and output types to uint8, uint16, float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(EPContext)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Onnx node container for EP context.")
      .Attr(
          "main_context",
          "Usually each single EPContext associate with a graph partition."
          "But for some case like QNN, it has single EPContext contains all partitions."
          "In that case, the node with ep_cache_context should set main_context=1. Other nodes set main_context=0 and skip ep_cache_context."
          "The path is relative to this Onnx file. Default is 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "ep_cache_context",
          "payload of the execution provider context if embed_mode=1, or path to the context file if embed_mode=0.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr(
          "embed_mode",
          "1: indicate ep_cache_context is the context content. 0: indicate ep_cache_context is the file path to the context content."
          "The path is relative to this Onnx file. Default is 1.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "ep_sdk_version",
          "(Optional) SDK version used to convert the model.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr(
          "onnx_model_filename",
          "(Optional) Filename of the original ONNX model.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr(
          "hardware_architecture",
          "(Optional) Hardware architecture.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr(
          "partition_name",
          "(Optional) partitioned graph name.",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr(
          "source",
          "(Optional) the source used to generate the engine/context cache file. Ort EP or native SDK tool chain",
          AttributeProto::STRING,
          OPTIONAL_VALUE)
      .Attr("notes", "(Optional) Some notes for the model", AttributeProto::STRING, OPTIONAL_VALUE)
      .Attr(
          "max_size",
          "max size in the context. Usage depend on the EP.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .AllowUncheckedAttributes()
      .Input(
          0,
          "inputs",
          "List of tensors for inputs",
          "T",
          OpSchema::Variadic,
          false,
          0,
          OpSchema::NonDifferentiable)
      .Output(
          0,
          "outputs",
          "One or more outputs, list of tensors for outputs",
          "T",
          OpSchema::Variadic,
          false,
          1,
          OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T",
          {"tensor(bool)",
           "tensor(int8)",
           "tensor(int16)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(uint8)",
           "tensor(uint16)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(float16)",
           "tensor(float)",
           "tensor(double)",
           "tensor(bfloat16)"},
          "Constrain input and output types.");

  static const char* BitmaskDropout_ver1_doc = R"DOC(
BitmaskDropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar).
It produces two tensor outputs: output (floating-point tensor) and mask (optional `Tensor<uint32>`). If `training_mode` is true then the output Y will be a random dropout.
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode, the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```

This op functions in much the same was as Dropout-11 and Dropout-13 do, except that the mask is output as a bit-packed uint32 tensor, instead of a boolean tensor.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(BitmaskDropout)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(BitmaskDropout_ver1_doc)
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::INT, OPTIONAL_VALUE)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of the scaled input, which is typically "
             "the case during training. It is an optional value, if not specified it will default to 0.5.",
             "T1", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Input(
          2, "training_mode",
          "If set to true then it indicates dropout is being used for training. It is an optional value hence unless "
          "specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode "
          "where "
          "nothing will be dropped from the input data and if mask is requested as output it will contain all ones.",
          "T2", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Output(0, "output", "The output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(1, "mask", "The bit-packed output mask.", "T3", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("T1", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint("T2", {"tensor(bool)"}, "Constrain 'training_mode' to boolean tensor.")
      .TypeConstraint("T3", {"tensor(uint32)"}, "Constrain output 'mask' types to bit-packed uint32 tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        if (ctx.getNumOutputs() == 2) {
          updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::UINT32);
        }
      });

  static const char* MatMulNBits_ver1_doc = R"DOC(
MatMulNBits performs a matrix multiplication where the right-hand-side matrix (weights) is quantized to N bits.

It is a fusion of two operations:
1. Linear dequantization of the quantized weights using scale and (optionally) zero-point with formula:
   dequantized_weight = (quantized_weight - zero_point) * scale
2. Matrix multiplication between the input matrix A and the dequantized weight matrix.

The weight matrix is a 2D constant matrix with the input feature count and output feature count specified by attributes 'K' and 'N'.
It is quantized block-wise along the K dimension with a block size specified by the 'block_size' attribute.
The block size must be a power of 2 and not smaller than 16 (e.g., 16, 32, 64, 128). Each block has its own scale and zero-point.
The quantization is performed using a bit-width specified by the 'bits' attribute, which can take values from 2 to 8.

The quantized weights are stored in a bit-packed format along the K dimension, with each block being represented by a blob of uint8.
For example, for 4 bits, the first 4 bits are stored in the lower 4 bits of a byte, and the second 4 bits are stored in the higher 4 bits of a byte.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulNBits)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(MatMulNBits_ver1_doc)
      .Attr("K", "Input feature dimension of the weight matrix.", AttributeProto::INT)
      .Attr("N", "Output feature dimension of the weight matrix.", AttributeProto::INT)
      .Attr("bits", "Bit-width used to quantize the weights (supported values: 2, 4, 8)", AttributeProto::INT, static_cast<int64_t>(4))
      .Attr("block_size",
            "Size of each quantization block along the K (input feature) dimension. "
            "Must be a power of two and ≥ 16 (e.g., 16, 32, 64, 128).",
            AttributeProto::INT)
      .Attr("accuracy_level",
            "The minimum accuracy level of input A, can be: 0(unset), 1(fp32), 2(fp16), 3(bf16), or 4(int8) "
            "(default unset). It is used to control how input A is quantized or downcast internally while "
            "doing computation, for example: 0 means input A will not be quantized or downcast while doing "
            "computation. 4 means input A can be quantized with the same block_size to int8 internally from "
            "type T1.",
            AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("weight_prepacked",
            "If set, input B is already prepacked into an EP-specific layout and the EP skips runtime "
            "weight prepacking. 0 (default): not prepacked. 1: prepacked in the CUDA SM80 fpA_intB layout. "
            "2: prepacked in the CUDA SM90 (Hopper) fpA_intB layout, consumed by the native SM90 kernel "
            "(requires a compute capability 9.0 device and block_size in {64, 128}).",
            AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "A", "The input tensor, not quantized.", "T1")
      .Input(1, "B",
             "Packed uint8 tensor of shape (N, k_blocks, blob_size), "
             "where k_blocks = ceil(K / block_size) and blob_size = (block_size * bits / 8). "
             "The quantized weights are stored in a bit-packed format along the K dimension, packed within each block_size.",
             "T2")
      .Input(2, "scales", "Per-block scaling factors for dequantization with shape (N, k_blocks) and same data type as input A.", "T1")
      .Input(3, "zero_points",
             "Per-block zero point for dequantization. It can be either packed or unpacked: "
             "Packed (uint8) format has shape (N, ceil(k_blocks * bits / 8)), and it uses same bit-packing method as Input B. "
             "Unpacked (same type as A) format has shape (N, k_blocks). "
             "If not provided, a default zero point is used: 2^(bits - 1) (e.g., 8 for 4-bit quantization, 128 for 8-bit). ",
             "T3", OpSchema::Optional)
      .Input(4, "g_idx", "group_idx. This input is deprecated", "T4", OpSchema::Optional)
      .Input(5, "bias", "Bias to add to result. It should have shape [N].", "T1", OpSchema::Optional)
      .Output(0, "Y", "tensor. The output tensor has the same rank as the input. ", "T1")
      .TypeConstraint("T1", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain quantized weight types to uint8.")
      .TypeConstraint("T3", {"tensor(uint8)", "tensor(float16)", "tensor(float)", "tensor(bfloat16)"},
                      "Constrain quantized zero point types to uint8 or float tensors.")
      .TypeConstraint("T4", {"tensor(int32)"}, "the index tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        // Shape inference
        int64_t in_features = getAttribute(ctx, "K", -1);
        int64_t out_features = getAttribute(ctx, "N", -1);
        MatmulWithQuantWeightShapeInference(ctx, in_features, out_features, true);

        // validate bias shape
        if (ctx.hasInput(5)) {
          if (!hasInputShape(ctx, 5)) {
            fail_shape_inference("bias shape must be known");
          }

          const auto& bias_shape = getInputShape(ctx, 5);
          if (bias_shape.dim_size() != 1 ||
              !bias_shape.dim(0).has_dim_value() ||
              bias_shape.dim(0).dim_value() != out_features) {
            fail_shape_inference("bias shape must be [N] where N = ", out_features);
          }
        }
      });

  static const char* MatMulNBitsMlp_ver1_doc = R"DOC(
MatMulNBitsMlp fuses two MatMulNBits projections that share the same input and computes

    gate = MatMulNBits(A, gate_weight) + gate_bias
    up = MatMulNBits(A, up_weight) + up_bias
    Y = activation(gate) * up

It can also optionally fuse SimplifiedLayerNormalization or SkipSimplifiedLayerNormalization before the
two projections:

  A_norm = SimplifiedLayerNormalization(A, norm_scale, epsilon)
    gate = MatMulNBits(A_norm, gate_weight) + gate_bias
    up = MatMulNBits(A_norm, up_weight) + up_bias
    Y = activation(gate) * up

  A_norm = SkipSimplifiedLayerNormalization(A, skip, norm_scale, epsilon)
    gate = MatMulNBits(A_norm, gate_weight) + gate_bias
    up = MatMulNBits(A_norm, up_weight) + up_bias
    Y = activation(gate) * up

This operator is intended for decoder MLP patterns such as Qwen-style gate and up projections, but it remains
semantically valid for both prefill and decode because the output shape is the standard MatMul result shape
derived from the runtime shape of A and the shared attributes K and N.

The operator contract includes a string attribute describing the fused gate activation.

When fused from SkipSimplifiedLayerNormalization, the optional residual-sum output may also be materialized:

  A_norm, input_skip_bias_sum = SkipSimplifiedLayerNormalization(A, skip, norm_scale, epsilon)
  gate = MatMulNBits(A_norm, gate_weight) + gate_bias
  up = MatMulNBits(A_norm, up_weight) + up_bias
  Y = activation(gate) * up
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulNBitsMlp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(MatMulNBitsMlp_ver1_doc)
      .Attr("K", "Input feature dimension shared by both quantized weight matrices.", AttributeProto::INT)
      .Attr("N", "Output feature dimension shared by both quantized weight matrices.", AttributeProto::INT)
      .Attr("bits", "Bit-width used to quantize both weight matrices. Currently only bits=4 is supported by the WebGPU kernel.", AttributeProto::INT, static_cast<int64_t>(4))
      .Attr("block_size",
            "Size of each quantization block along the K dimension. Currently only block_size=32 is supported by the WebGPU kernel.",
            AttributeProto::INT)
      .Attr("accuracy_level",
            "The minimum accuracy level of input A. It follows the same semantics as MatMulNBits.",
            AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("activation",
            "Activation applied to the gate projection.",
            AttributeProto::STRING)
      .Attr("epsilon",
            "Epsilon used by the optional fused (Skip)SimplifiedLayerNormalization. Defaults to 1e-5.",
            AttributeProto::FLOAT, 1e-5f)
      .Input(0, "A", "The shared input tensor.", "T1")
      .Input(1, "skip", "Optional skip input used by SkipSimplifiedLayerNormalization.", "T1", OpSchema::Optional)
      .Input(2, "norm_scale", "Optional RMSNorm scale with shape [K] used by SimplifiedLayerNormalization or SkipSimplifiedLayerNormalization.", "T1", OpSchema::Optional)
      .Input(3, "gate_B", "Packed uint8 tensor for the gate projection weights.", "T2")
      .Input(4, "gate_scales", "Per-block scaling factors for the gate projection.", "T1")
      .Input(5, "gate_bias", "Optional bias for the gate projection with shape [N].", "T1", OpSchema::Optional)
      .Input(6, "up_B", "Packed uint8 tensor for the up projection weights.", "T2")
      .Input(7, "up_scales", "Per-block scaling factors for the up projection.", "T1")
      .Input(8, "up_bias", "Optional bias for the up projection with shape [N].", "T1", OpSchema::Optional)
      .Output(0, "Y", "The fused gated MLP output tensor.", "T1")
      .Output(1, "input_skip_bias_sum", "Optional residual-sum output for SkipSimplifiedLayerNormalization.", "T1", OpSchema::Optional)
      .TypeConstraint("T1", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain quantized weight types to uint8.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getNumOutputs() > 1) {
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
        }

        const int64_t in_features = getAttribute(ctx, "K", -1);
        const int64_t out_features = getAttribute(ctx, "N", -1);
        MatmulWithQuantWeightShapeInference(ctx, in_features, out_features, true);

        if (ctx.hasInput(1) && !ctx.hasInput(2)) {
          fail_shape_inference("norm_scale input must be present when skip input is provided");
        }

        if (ctx.hasOutput(1)) {
          if (!ctx.hasInput(1)) {
            fail_shape_inference("skip input must be present when input_skip_bias_sum output is requested");
          }

          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto* skip_sum_shape = getOutputShape(ctx, 1);
          *skip_sum_shape = getInputShape(ctx, 0);
        }

        if (ctx.hasInput(2)) {
          if (!hasInputShape(ctx, 2)) {
            fail_shape_inference("norm_scale shape must be known");
          }

          const auto& norm_scale_shape = getInputShape(ctx, 2);
          if (norm_scale_shape.dim_size() != 1 ||
              !norm_scale_shape.dim(0).has_dim_value() ||
              norm_scale_shape.dim(0).dim_value() != in_features) {
            fail_shape_inference("norm_scale shape must be [K] where K = ", in_features);
          }
        }

        for (size_t bias_input_index : {5U, 8U}) {
          if (!ctx.hasInput(static_cast<int>(bias_input_index))) {
            continue;
          }

          if (!hasInputShape(ctx, static_cast<int>(bias_input_index))) {
            fail_shape_inference("bias shape must be known");
          }

          const auto& bias_shape = getInputShape(ctx, static_cast<int>(bias_input_index));
          if (bias_shape.dim_size() != 1 ||
              !bias_shape.dim(0).has_dim_value() ||
              bias_shape.dim(0).dim_value() != out_features) {
            fail_shape_inference("bias shape must be [N] where N = ", out_features);
          }
        }
      });

  static const char* MatMulNBitsQkv_ver1_doc = R"DOC(
MatMulNBitsQkv fuses either SimplifiedLayerNormalization (RMSNorm)
or SkipSimplifiedLayerNormalization with three MatMulNBits projections that share the
same normalized activation.

  A_norm = SimplifiedLayerNormalization(A, norm_scale, epsilon)
  Q = MatMulNBits(A_norm, q_weight) + q_bias
  K = MatMulNBits(A_norm, k_weight) + k_bias
  V = MatMulNBits(A_norm, v_weight) + v_bias

If skip is provided, the operator computes the SkipSimplifiedLayerNormalization variant
and may also return the input+skip residual sum as output 3.

This operator is intended as a decode-oriented QKV fusion primitive.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulNBitsQkv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(MatMulNBitsQkv_ver1_doc)
      .Attr("K", "Input feature dimension shared by the normalized input and all projection weights.", AttributeProto::INT)
      .Attr("Nq", "Output feature dimension of the Q projection.", AttributeProto::INT)
      .Attr("Nkv", "Output feature dimension shared by the K and V projections.", AttributeProto::INT)
      .Attr("bits", "Bit-width used to quantize all weight matrices. Currently only bits=4 is supported by the WebGPU kernel.", AttributeProto::INT, static_cast<int64_t>(4))
      .Attr("block_size",
            "Size of each quantization block along the K dimension. Currently only block_size=32 is supported by the WebGPU kernel.",
            AttributeProto::INT)
      .Attr("accuracy_level",
            "The minimum accuracy level of input A. It follows the same semantics as MatMulNBits.",
            AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("epsilon", "Epsilon used by the simplified layer norm reduction.", AttributeProto::FLOAT, 1e-6f)
      .Input(0, "A", "The shared input tensor.", "T1")
      .Input(1, "skip", "Optional residual input for SkipSimplifiedLayerNormalization.", "T1", OpSchema::Optional)
      .Input(2, "norm_scale", "Scale input for the simplified layer norm with shape [K].", "T1")
      .Input(3, "q_B", "Packed uint8 tensor for the Q projection weights.", "T2")
      .Input(4, "q_scales", "Per-block scaling factors for the Q projection.", "T1")
      .Input(5, "q_bias", "Optional bias for the Q projection with shape [Nq].", "T1", OpSchema::Optional)
      .Input(6, "k_B", "Packed uint8 tensor for the K projection weights.", "T2")
      .Input(7, "k_scales", "Per-block scaling factors for the K projection.", "T1")
      .Input(8, "k_bias", "Optional bias for the K projection with shape [Nkv].", "T1", OpSchema::Optional)
      .Input(9, "v_B", "Packed uint8 tensor for the V projection weights.", "T2")
      .Input(10, "v_scales", "Per-block scaling factors for the V projection.", "T1")
      .Input(11, "v_bias", "Optional bias for the V projection with shape [Nkv].", "T1", OpSchema::Optional)
      .Output(0, "Q", "The Q projection output tensor.", "T1")
      .Output(1, "K", "The K projection output tensor.", "T1")
      .Output(2, "V", "The V projection output tensor.", "T1")
      .Output(3, "input_skip_bias_sum", "Optional residual-sum output for SkipSimplifiedLayerNormalization.", "T1", OpSchema::Optional)
      .TypeConstraint("T1", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain quantized weight types to uint8.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        for (size_t output_index = 0; output_index < ctx.getNumOutputs(); ++output_index) {
          propagateElemTypeFromInputToOutput(ctx, 0, output_index);
        }

        if (!hasInputShape(ctx, 0)) {
          return;
        }

        const auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() == 0) {
          fail_shape_inference("A must have rank >= 1");
        }

        const int64_t q_out_features = getAttribute(ctx, "Nq", -1);
        const int64_t kv_out_features = getAttribute(ctx, "Nkv", -1);

        auto set_output_shape = [&](int output_index, int64_t out_features) {
          auto* output_shape = getOutputShape(ctx, output_index);
          *output_shape = input_shape;
          output_shape->mutable_dim(output_shape->dim_size() - 1)->set_dim_value(out_features);
        };

        set_output_shape(0, q_out_features);
        set_output_shape(1, kv_out_features);
        set_output_shape(2, kv_out_features);
        if (ctx.getNumOutputs() > 3) {
          auto* output_shape = getOutputShape(ctx, 3);
          *output_shape = input_shape;
        }

        if (ctx.hasInput(5)) {
          if (!hasInputShape(ctx, 5)) {
            fail_shape_inference("q_bias shape must be known");
          }

          const auto& q_bias_shape = getInputShape(ctx, 5);
          if (q_bias_shape.dim_size() != 1 ||
              !q_bias_shape.dim(0).has_dim_value() ||
              q_bias_shape.dim(0).dim_value() != q_out_features) {
            fail_shape_inference("q_bias shape must be [Nq] where Nq = ", q_out_features);
          }
        }

        for (int bias_input_index : {8, 11}) {
          if (!ctx.hasInput(bias_input_index)) {
            continue;
          }

          if (!hasInputShape(ctx, bias_input_index)) {
            fail_shape_inference("bias shape must be known");
          }

          const auto& bias_shape = getInputShape(ctx, bias_input_index);
          if (bias_shape.dim_size() != 1 ||
              !bias_shape.dim(0).has_dim_value() ||
              bias_shape.dim(0).dim_value() != kv_out_features) {
            fail_shape_inference("bias shape must be [Nkv] where Nkv = ", kv_out_features);
          }
        }
      });

  static const char* MatMulBnb4_ver1_doc = R"DOC(
MatMulBnb4 is a MatMul with weight quantized with 4 bits using either FP4 or NF4 data type (https://arxiv.org/pdf/2305.14314.pdf). It does Matrix Multiplication like MatMul (https://github.com/onnx/onnx/blob/main/docs/Operators.md#matmul) with differences:
  1. Input B is a 2D constant Matrix. Its input feature count and output feature count are specified by attribute 'K' and 'N'.
  2. Input B is quantized with 4 bits with quantization data type specified by attribute 'quant_type'. It is transposed, flattened and quantized blockwisely with block size specified by attribute 'block_size'.
     And block_size is not an arbitrary number and must be a power of 2 and not smaller than 16, like 16, 32, 64, 128,..
  3. Input B's quantization constants or scales are specified by input 'absmax'.

  Input B is stored as uint8_t with shape: [(N * K + 1) / 2].
  Input absmax is stored in same type as original type of B(float32, float16) with shape like: [(N * K + block_size - 1) / block_size].


  1. (Default value) transB=True (Majorly used for forward pass)
    Shape of A: [D0, D1, ..., Dn, K]
    Shape of Dequanted B: [N, K], this is aligned with how PyTorch defined the linear weight, .e.g [out_features, in_features].

    The computation math:
      dequant_B = dequant(B, absmax, quant_type, block_size)
      transposed_dequant_B = dequant_B^T
      output = A @ transposed_dequant_B

    Shape of output: [D0, D1, ..., Dn, N]

  2. transB=False (Majorly used for backward pass)
    Shape of A: [D0, D1, ..., Dn, N]
    Shape of Dequanted B: [N, K], this is aligned with how PyTorch defined the linear weight, .e.g [out_features, in_features].

    The computation math:
      dequant_B = dequant(B, absmax, quant_type, block_size)
      output = A @ dequant_B

    Shape of output: [D0, D1, ..., Dn, K]

)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulBnb4)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(MatMulBnb4_ver1_doc)
      .Attr("K", "size of each input feature", AttributeProto::INT)
      .Attr("N", "size of each output feature", AttributeProto::INT)
      .Attr("block_size", "number of groupsize used for weight quantization. It needs to be a power of 2 and not smaller than 16.", AttributeProto::INT)
      .Attr("quant_type", "quantization data type. 0 for FP4, 1 for NF4.", AttributeProto::INT)
      .Attr("training_mode",
            "Indicate if the ops run in training_mode, by default, False.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Attr("transB", "Whether B should be transposed on the last two dimensions before doing multiplication. Default to be 1.",
            AttributeProto::INT, static_cast<int64_t>(1))
      .Input(0, "A", "The input tensor, not quantized", "T1")
      .Input(1, "B", "1-dimensional quantized data for weight", "T2")
      .Input(2, "absmax", "quantization constants", "T1")
      .Output(0, "Y", "tensor. The output tensor has the same rank as the input. ", "T1")
      .TypeConstraint("T1", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float/half_float/brain_float tensors.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain quantized weight types to uint8.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        // Shape inference
        int64_t in_features = getAttribute(ctx, "K", -1);
        int64_t out_features = getAttribute(ctx, "N", -1);
        bool transB = getAttribute(ctx, "transB", 1) != 0;
        MatmulWithQuantWeightShapeInference(ctx, in_features, out_features, transB);
      });

  static const char* GatherBlockQuantized_ver1_doc = R"DOC(
GatherBlockQuantized is a Gather with data quantized. It is similar to Gather (https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather) with differences:
  1. Input `data` is a constant. It is quantized block-wise along attribute `quantize_axis` with block size specified by attribute `block_size`.
     `block_size` must be a power of 2 and not smaller than 16, like 16, 32, 64, 128, ...
     For an FP8 or FP4 `data` type (see point 6 below), `block_size` may also be 0, meaning the entire `quantize_axis`
     dimension forms a single block (i.e. one scale per row).
  2. Input `data`'s scale and zero point are specified by input `scales` and `zero_points`. `scales` and `zero_points` are also constants.
     If `zero_points` is not provided, the default value is 0 for int4/uint4, or 2^(bits-1) for uint8.
     `zero_points` must not be provided when `data` is an FP8 or FP4 type: FP8/FP4 quantization is symmetric.
  3. During the op execution, `data` and `indices` are first used to generate the quantized output. Then, `scales` and `zero_points` are used
     to dequantize the output.
  4. The `output` and `scales` have the same type. The `data` and `zero_points` have the same type.
  5. For uint8 data, the `gather_axis` must be 0. The supported `bits` values for uint8 data are 2, 4, and 8;
     for `bits` < 8 the values are packed along the last dimension (low-order bits first), and `gather_axis`
     and `quantize_axis` must differ.
  6. `data` may also be an FP8 type (float8e4m3fn, float8e4m3fnuz, float8e5m2 or float8e5m2fnuz) or an FP4 type
     (float4e2m1), rather than an integer block-quantized type. In that case `bits` is ignored, there is
     no `zero_points` input, and dequantization is simply `output[...] = float(data[...]) * scales[block_index(...)]`.
     On any axis other than `quantize_axis`, the corresponding `scales` dimension must either equal `data`'s
     dimension, or be 1, in which case the scale is broadcast along that axis (e.g. a single scale shared by
     every row, as with a per-tensor scale applied to an entire embedding table).
)DOC";

  std::vector<std::string> gather_block_quantized_T1_types = {"tensor(int4)", "tensor(uint4)", "tensor(uint8)"};
#if !defined(DISABLE_FLOAT8_TYPES)
  gather_block_quantized_T1_types.insert(
      gather_block_quantized_T1_types.end(),
      {"tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"});
#endif  // !defined(DISABLE_FLOAT8_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
  gather_block_quantized_T1_types.push_back("tensor(float4e2m1)");
#endif  // !defined(DISABLE_FLOAT4_TYPES)

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherBlockQuantized)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(GatherBlockQuantized_ver1_doc)
      .Attr("gather_axis",
            "(Optional) Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("quantize_axis",
            "(Optional) Which axis to block-wise quantize. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("block_size",
            "(Optional) block size used for weight quantization. It needs to be a power of 2 and not smaller than 16, "
            "or 0. A value of 0 is only valid for an FP8 or FP4 `data` type and means the entire `quantize_axis` "
            "dimension forms a single block.",
            AttributeProto::INT,
            static_cast<int64_t>(128))
      .Attr("bits",
            "Number of bits used for weight quantization. Must be 2, 4 or 8. Ignored when `data` is an FP8 or "
            "FP4 type.",
            AttributeProto::INT,
            static_cast<int64_t>(4))
      .Input(0, "data", "Tensor of rank r >= 1. Block-wise quantized.", "T1")
      .Input(1,
             "indices",
             "Tensor of int32/int64 indices, of any rank q. Values in [-s, s-1] select elements along an axis of "
             "size s. Unlike ONNX Gather, an out-of-range index produces zeros for the corresponding output slice.",
             "Tind")
      .Input(2, "scales",
             "quantization scale. Same rank as data. On axes other than quantize_axis, a dimension of 1 broadcasts "
             "the scale along that axis (e.g. a single per-tensor scale for the whole table); only applicable when "
             "`data` is an FP8 or FP4 type.",
             "T2")
      .Input(3, "zero_points",
             "quantization zero points. Must not be provided when `data` is an FP8 or FP4 type.",
             "T1", OpSchema::Optional)
      .Output(0, "output", "Dequantized output tensor of rank q + (r - 1).", "T2")
      .TypeConstraint("T1", gather_block_quantized_T1_types, "Constrain quantized types.")
      .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain dequantized types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 2, 0);

        // The first 3 inputs must have shape.
        if (!hasNInputShapes(ctx, 3)) {
          return;
        }
        const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
        const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
        const TensorShapeProto& scales_shape = ctx.getInputType(2)->tensor_type().shape();

        int r = data_shape.dim_size();
        if (r <= 1) {
          fail_shape_inference("data tensor must have rank > 1");
        }

        int gather_axis = static_cast<int>(getAttribute(ctx, "gather_axis", 0));
        int quantize_axis = static_cast<int>(getAttribute(ctx, "quantize_axis", 1));
        int bits = static_cast<int>(getAttribute(ctx, "bits", 4));
        auto block_size = getAttribute(ctx, "block_size", 128);

        if (gather_axis < -r || gather_axis >= r) {
          fail_shape_inference("gather_axis must be in [-r, r-1]");
        }
        if (quantize_axis < -r || quantize_axis >= r) {
          fail_shape_inference("quantize_axis must be in [-r, r-1]");
        }

        const auto data_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
        const bool is_fp_quantized = data_elem_type == onnx::TensorProto_DataType_FLOAT8E4M3FN ||
                                     data_elem_type == onnx::TensorProto_DataType_FLOAT8E4M3FNUZ ||
                                     data_elem_type == onnx::TensorProto_DataType_FLOAT8E5M2 ||
                                     data_elem_type == onnx::TensorProto_DataType_FLOAT8E5M2FNUZ ||
                                     data_elem_type == onnx::TensorProto_DataType_FLOAT4E2M1;

        if (data_elem_type == onnx::TensorProto_DataType_UINT8) {
          if (bits != 2 && bits != 4 && bits != 8) {
            fail_shape_inference("bits must be 2, 4, or 8 for uint8 data");
          }
        } else if (!is_fp_quantized && bits != 4) {
          fail_shape_inference("bits must be 4 for int4/uint4 data");
        }

        const bool block_size_valid = block_size == 0
                                          ? is_fp_quantized
                                          : (block_size >= 16 && (block_size & (block_size - 1)) == 0);
        if (!block_size_valid) {
          fail_shape_inference("block_size must be a power of 2 and not smaller than 16, or 0 for FP8/FP4 data");
        }

        gather_axis = (gather_axis + r) % r;
        quantize_axis = (quantize_axis + r) % r;

        if (data_elem_type == onnx::TensorProto_DataType_UINT8) {
          if (gather_axis != 0) {
            fail_shape_inference("gather_axis must be 0, for uint8 data");
          }
          // CPU implementation requires quantize_axis to be the last dimension right now.
          // we are relaxing it in the spec and shape inference since other EP might not have such restriction.
        }

        if (is_fp_quantized && ctx.hasInput(3)) {
          fail_shape_inference("zero_points must not be provided when data is an FP8 or FP4 type");
        }

        if (scales_shape.dim_size() != r) {
          fail_shape_inference("scales must have the same rank as data");
        }

        uint32_t components = (data_elem_type == onnx::TensorProto_DataType_UINT8) ? (8 / bits) : 1;
        if (components > 1 && gather_axis == quantize_axis) {
          fail_shape_inference("gather_axis and quantize_axis must not be the same for packed uint8 data");
        }

        for (int i = 0; i < r; ++i) {
          if (data_shape.dim(i).has_dim_value() && scales_shape.dim(i).has_dim_value()) {
            if (i == quantize_axis) {
              int64_t effective_block_size =
                  block_size == 0 ? std::max<int64_t>(data_shape.dim(i).dim_value(), 1) : block_size;
              if ((data_shape.dim(i).dim_value() * components + effective_block_size - 1) / effective_block_size !=
                  scales_shape.dim(i).dim_value()) {
                fail_shape_inference("data shape and scales shape do not match");
              }
            } else if (data_shape.dim(i).dim_value() != scales_shape.dim(i).dim_value() &&
                       !(is_fp_quantized && scales_shape.dim(i).dim_value() == 1)) {
              fail_shape_inference("data shape and scales shape do not match");
            }
          }
        }

        // validate zero point shape
        if (ctx.hasInput(3)) {
          if (!hasInputShape(ctx, 3)) {
            fail_shape_inference("zero_points shape must be known");
          }

          const auto& zp_shape = getInputShape(ctx, 3);
          if (zp_shape.dim_size() != r) {
            fail_shape_inference("zero points must have the same rank as data");
          }

          for (int i = 0; i < r; ++i) {
            if (!zp_shape.dim(i).has_dim_value() ||
                zp_shape.dim(i).dim_value() != scales_shape.dim(i).dim_value()) {
              if (data_elem_type == onnx::TensorProto_DataType_UINT8 &&
                  components > 1 &&
                  i == quantize_axis &&
                  zp_shape.dim(i).dim_value() == (scales_shape.dim(i).dim_value() + components - 1) / components) {
                continue;
              }
              fail_shape_inference("zero points shape and scales shape do not match");
            }
          }
        }

        int q = indices_shape.dim_size();
        int out_rank = q + r - 1;
        auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        output_shape->clear_dim();
        for (int i = 0; i < gather_axis; ++i) {
          *output_shape->add_dim() = data_shape.dim(i);
        }
        for (int i = 0; i < q; ++i) {
          *output_shape->add_dim() = indices_shape.dim(i);
        }
        for (int i = gather_axis + 1; i < r; ++i) {
          *output_shape->add_dim() = data_shape.dim(i);
        }

        // Find the correct dimension to expand and multiply it by components
        if (components > 1) {
          int quantize_output_dim_idx = (quantize_axis < gather_axis) ? quantize_axis : quantize_axis + q - 1;
          if (quantize_output_dim_idx >= 0 && quantize_output_dim_idx < out_rank) {
            auto* dim_to_update = output_shape->mutable_dim(quantize_output_dim_idx);
            if (dim_to_update->has_dim_value()) {
              dim_to_update->set_dim_value(dim_to_update->dim_value() * components);
            }
          }
        }
      });

#ifdef ENABLE_ATEN
  ONNX_CONTRIB_OPERATOR_SCHEMA(ATen)
      .SetDomain(kPytorchAtenDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("ATen")
      .Input(0, "inputs", "ATen Op inputs.", "T", OpSchema::Variadic,
             /*is_homogeneous*/ false,
             /*min_arity*/ 1)
      .Output(0, "outputs", "ATen Op outputs.", "T", OpSchema::Variadic,
              /*is_homogeneous*/ false,
              /*min_arity*/ 1)
      .Attr("operator", "Name of ATen operator.", AttributeProto::STRING)
      .Attr("overload_name", "Overload name of ATen operator.", AttributeProto::STRING, false)
      .Attr("cpu_input_args", "CPU input argument indices.", AttributeProto::INTS, false)
      .Attr("cpu_output_args", "CPU output argument indices.", AttributeProto::INTS, false)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(),
                      "Allow inputs and outputs to be any kind of tensor.");
#endif

#ifdef ENABLE_TRAINING_OPS
  // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
  // 2). this is needed by inference for other purpose.

  static const char* ShrunkenGather_ver1_doc = R"DOC(
    This op is a specialised case of Gather-13, adding additional constraint including: indices being 1D,
and indices count < input element count on the specified axis.

Having this op allows runtime to do operator re-ordering to reduce compute FLOPs.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ShrunkenGather)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(ShrunkenGather_ver1_doc)
      .AllowUncheckedAttributes()
      .Attr(
          "axis",
          "Which axis to gather on. Negative value means "
          "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "indices",
          "Tensor of int64 indices, with rank = 1. All index values are expected to be within bounds [-s, s-1] "
          "along axis of size s. It is an error if any of the index values are out of bounds."
          "The number of elements in indices must be less than the number of elements in the input tensor,"
          "which is the reason why this op is called ShrunkenGather.",
          "Tind",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Tensor of rank r.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types_ir4(),
          "Constrain input and output types to any tensor type.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        if (!hasNInputShapes(ctx, 2)) {
          return;
        }

        const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
        const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
        int r = data_shape.dim_size();
        if (r < 1) {
          fail_shape_inference("data tensor must have rank >= 1");
        }
        int q = indices_shape.dim_size();
        int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
        if (axis < -r || axis >= r) {
          fail_shape_inference("axis must be in [-r, r-1]");
        }
        if (axis < 0) {
          axis += r;
        }

        int out_rank = q + r - 1;
        auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        int i = 0;
        for (; i < axis; ++i) {
          *final_output_shape->add_dim() = data_shape.dim(i);
        }

        for (; i < axis + q; ++i) {
          *final_output_shape->add_dim() = indices_shape.dim(i - axis);
        }

        for (; i < out_rank; ++i) {
          *final_output_shape->add_dim() = data_shape.dim(i - q + 1);
        }
      });

#endif

#ifndef _OPSCHEMA_LIB_
  // Register the NCHWc schemas if supported by the platform.
  if (MlasNchwcGetBlockSize() > 1) {
    RegisterNchwcSchemas();
  }
#endif

#ifdef ORT_USE_NCCL
  RegisterCollectiveOps();
#endif
}

}  // namespace contrib
}  // namespace onnxruntime
