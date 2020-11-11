// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/providers/common.h"
#include "orttraining/core/graph/training_op_defs.h"
#include "onnx/defs/function.h"
#include <math.h>

namespace onnxruntime {
namespace training {

using namespace ONNX_NAMESPACE;

void AddRepeatedInputs(
    OpSchema& op_schema,
    const int start,
    const int count,
    const std::vector<std::string>& names,
    const std::vector<std::string>& descriptions,
    const std::vector<std::string>& type_strs,
    const OpSchema::FormalParameterOption param_option) {
  ORT_ENFORCE(names.size() == descriptions.size(),
              "Names and descriptions must be equal-length.");
  ORT_ENFORCE(names.size() == type_strs.size(),
              "Names and type_strs must be equal-length.");
  ORT_ENFORCE(param_option != OpSchema::Variadic,
              "param_option cannot be variadic.");
  ORT_ENFORCE(count > 0, "Count must be positive.");

  for (int i = 0; i < count; ++i) {
    const int input_index_start = start + i * static_cast<int>(names.size());
    // Repeat one group of names once.
    for (size_t j = 0; j < names.size(); ++j) {
      const int input_index = input_index_start + static_cast<int>(j);
      std::string modified_input_name = "__group_" + std::to_string(i) + "__" + names[j];
      ORT_ENFORCE(input_index >= static_cast<int>(op_schema.inputs().size()),
                  "Invalid redefinition of input ", input_index, " for OpSchema ", op_schema.Name());
      op_schema.Input(input_index, modified_input_name, descriptions[j], type_strs[j], param_option, false);
    }
  }
}

void AddRepeatedOutputs(
    OpSchema& op_schema,
    const int start,
    const int count,
    const std::vector<std::string>& names,
    const std::vector<std::string>& descriptions,
    const std::vector<std::string>& type_strs,
    const OpSchema::FormalParameterOption param_option) {
  ORT_ENFORCE(names.size() == descriptions.size(),
              "Names and descriptions must be equal-length.");
  ORT_ENFORCE(names.size() == type_strs.size(),
              "Names and type_strs must be equal-length.");
  ORT_ENFORCE(param_option != OpSchema::Variadic,
              "param_option cannot be variadic.");
  ORT_ENFORCE(count > 0, "Count must be positive.");

  for (int i = 0; i < count; ++i) {
    const int output_index_start = start + i * static_cast<int>(names.size());
    // Repeat one group of names once.
    for (int j = 0; j < static_cast<int>(names.size()); ++j) {
      const int output_index = output_index_start + j;
      std::string modified_output_name = "__group_" + std::to_string(i) + "__" + names[j];
      ORT_ENFORCE(output_index >= static_cast<int>(op_schema.outputs().size()),
                  "Invalid redefinition of output ", output_index, " for OpSchema ", op_schema.Name());
      op_schema.Output(output_index, modified_output_name, descriptions[j], type_strs[j], param_option, false);
    }
  }
}

static void checkSendInputTensorElemTypes(
    InferenceContext& ctx,
    const std::string& attributeName,
    const size_t inputSize) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) {  // attribute not present
    fail_type_inference("Value of attribute ", attributeName, " not specified");
  }

  size_t tensor_num = static_cast<size_t>(attr_proto->ints_size());

  if (tensor_num != inputSize) {
    fail_type_inference("Attribute ", attributeName, " has a wrong size");
  }

  const int64_t* elem_types = attr_proto->ints().data();

  for (size_t i = 0; i < tensor_num; ++i) {
    auto elem_type = static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(elem_types[i]);
    if (!TensorProto_DataType_IsValid(elem_type)) {
      fail_type_inference("Attribute ", attributeName, " does not specify a valid type.");
    }

    auto input_type = ctx.getInputType(i + 2);
    if (input_type->tensor_type().has_elem_type()) {
      auto input_elem_type = static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(input_type->tensor_type().elem_type());
      if (input_elem_type != elem_type) {
        fail_type_inference("Attribute ", attributeName, " does not match an input's element type.");
      }
    } else {
      fail_type_inference("Attribute ", attributeName, " does not match an input type.");
    }
  }
}

static void propagateRecvOutputTensorElemTypes(
    InferenceContext& ctx,
    const std::string& attributeName,
    const size_t outputSize) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) {  // attribute not present
    fail_type_inference("Value of attribute ", attributeName, " not specified");
  }

  size_t tensor_num = static_cast<size_t>(attr_proto->ints_size());

  if (tensor_num != outputSize) {
    fail_type_inference("Attribute ", attributeName, " has a wrong size");
  }

  const int64_t* elem_types = attr_proto->ints().data();

  for (size_t i = 0; i < tensor_num; ++i) {
    auto elem_type = static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(elem_types[i]);
    if (!TensorProto_DataType_IsValid(elem_type)) {
      fail_type_inference("Attribute ", attributeName, " does not specify a valid type.");
    }
    updateOutputElemType(ctx, i + 1, elem_type);
  }
}

// TODO: This is copied from onnx schemas. When the change is in and we update this can be removed.
// For Brevity documentation was not copied
OpSchema& RegisterLambOpSchema(OpSchema&& op_schema) {
  op_schema
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "alpha",
          "Coefficient of previous gradient in running average.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.9f))
      .Attr(
          "beta",
          "Coefficient of previous squared gradient in running average."
          "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
          "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.999f))
      .Attr(
          "lambda",
          "Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to 0, "
          "which means no regularization.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.0f))
      .Attr(
          "ratio_min",
          "Lower bound on confidence ratio.",
          AttributeProto::FLOAT,
          -std::numeric_limits<float>::infinity())
      .Attr(
          "ratio_max",
          "Upper bound on confidence ratio.",
          AttributeProto::FLOAT,
          std::numeric_limits<float>::infinity())
      .Attr(
          "epsilon",
          "Small scalar to avoid dividing by zero.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 1e-6f))
      .Attr(
          "do_bias_correction",
          "Compute unbiased 1st and 2nd momentums.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float scalars.")
      .TypeConstraint(
          "T2",
          {"tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T3",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T4",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_MIXED_PRECISION_FP",
          {"tensor(float16)", "tensor(bfloat16)"},
          "Constrain input types to float16 or bfloat16 tensors.")
      .TypeConstraint(
          "T_GRAD_NORM",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain update count to 64-bit integer")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Handle update count, the first output.
        const size_t step_input_index = 4;
        const size_t step_output_index = 0;
        auto input_type = ctx.getInputType(step_input_index);
        if (input_type != nullptr) {
          propagateElemTypeFromInputToOutput(ctx, step_input_index, step_output_index);
          if (hasInputShape(ctx, step_input_index)) {
            propagateShapeFromInputToOutput(ctx, step_input_index, step_output_index);
          }
        }

        // Handle other tensors including new weight, new gradient (update direction),
        // new momentums.
        for (size_t i = 0; i < ctx.getNumInputs() - 5; ++i) {
          const size_t input_index = 5 + i;   // The first 5 inputs don't affect output shape.
          const size_t output_index = 1 + i;  // The first output has been processed above.
          input_type = ctx.getInputType(input_index);
          if (input_type != nullptr) {
            propagateElemTypeFromInputToOutput(ctx, input_index, output_index);
            if (hasInputShape(ctx, input_index)) {
              propagateShapeFromInputToOutput(ctx, input_index, output_index);
            }
          }
        }
      });

  op_schema
      .Input(
          0,
          "update_signal",
          "This signal indicates if weight tensors should be updated.",
          "T_BOOL",
          OpSchema::Optional)
      .Input(
          1,
          "loss_scale",
          "Loss scale for mixed precision training.",
          "T2",
          OpSchema::Optional)
      .Input(
          2,
          "gradient_norm",
          "Norm of global gradient.",
          "T_GRAD_NORM",
          OpSchema::Optional)
      .Input(
          3,
          "R",
          "The initial learning rate.",
          "T1",
          OpSchema::Optional)
      .Input(
          4,
          "step",
          "One-based index of the current training iteration.",
          "TInt64",
          OpSchema::Optional);

  AddRepeatedInputs(
      op_schema,
      5,
      1024,
      {"weights",
       "gradients",
       "moment1",
       "moment2",
       "mixed_precision_weights"},
      {"weights to optimize.",
       "gradients computed in this iteration.",
       "exponentially averaged historical gradients.",
       "exponentially averaged historical squared gradients.",
       "FP16 or BF16 weights to optimize."},
      {"T2",
       "T3",
       "T4",
       "T4",
       "T_MIXED_PRECISION_FP"},
      OpSchema::Optional);

  op_schema
      .Output(
          0,
          "new_step",
          "One-based index of the next training iteration.",
          "TInt64",
          OpSchema::Optional);

  AddRepeatedOutputs(
      op_schema,
      1,
      1024,
      {"new_weights",
       "new_gradients",
       "new_moment_1",
       "new_moment_2",
       "new_mixed_precision_weights"},
      {"New weights",
       "New gradients",
       "New averaged gradients",
       "New averaged squared gradients",
       "New FP16 or BF16 weights"},
      {"T2",
       "T3",
       "T4",
       "T4",
       "T_MIXED_PRECISION_FP"},
      OpSchema::Optional);

  return op_schema;
}

bool BuildContextDependentFunctionBodyMSD(const FunctionBodyBuildContext& ctx, 
                          const OpSchema& schema, FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back(FunctionBodyHelper::Const<float>("Q_Pow", 2));
  // TBD: Investigate how to specify op domain/version
  body.push_back({{"X_Sub"}, "Sub", {"scores", "labels"}}); 
  if (ctx.hasInput(2)) { // Input 2 is "weights"
    body.push_back({{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}});
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"output"}, "Mul", {"weights", "X_Pow"}});
    } else {
      body.push_back({{"X_Mul"}, "Mul", {"weights", "X_Pow"}});
      if (ctx.getAttribute("reduction")->s() == "sum") {
        body.push_back({{"output"}, "ReduceSum", {"X_Mul"}, {MakeAttribute("keepdims", (int64_t)0)}});
      } else {
        body.push_back({{"output"}, "ReduceMean", {"X_Mul"}, {MakeAttribute("keepdims", (int64_t)0)}});
      }
    }
  } else {
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"output"}, "Pow", {"X_Sub", "Q_Pow"}});
    } else {
      body.push_back({{"X_Pow"}, "Pow", {"X_Sub", "Q_Pow"}});
      if (ctx.getAttribute("reduction")->s() == "sum") {
        body.push_back({{"output"}, "ReduceSum", {"X_Pow"}, {MakeAttribute("keepdims", (int64_t)0)}});
      } else {
        body.push_back({{"output"}, "ReduceMean", {"X_Pow"}, {MakeAttribute("keepdims", (int64_t)0)}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

void RegisterTrainingOpSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(ReluGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Attr(
          "axis",
          "Describes the axis of the inputs when coerced "
          "to 2D; defaults to one because the 0th axis most likely describes "
          "the batch_size",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(LogSoftmaxGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Attr(
          "axis",
          "Describes the axis of the inputs when coerced "
          "to 2D; defaults to one because the 0th axis most likely describes "
          "the batch_size",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(AveragePoolGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Attr(
          "kernel_shape",
          "The size of the kernel along each axis.",
          AttributeProto::INTS)
      .Attr(
          "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr(
          "auto_pad",
          "auto_pad doc",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr("pads", "pads_doc", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr(
          "count_include_pad",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxPoolGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output, Y", "T")
      .Input(1, "Indices", "Indices tensor from max pooling across the input tensor.", "I")
      .Output(0, "dX", "Gradient of input, X", "T")
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain index tensor to int64");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Input(2, "W", "Weight tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T", OpSchema::Optional)
      .Output(1, "dW", "Gradient of W", "T")
      .Output(2, "dB", "Gradient of B", "T", OpSchema::Optional)
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "shape", "Shape of the Gather input X.", "I")
      .Input(1, "indices", "Tensor of int32/int64 indices, of any rank q.", "Tind")
      .Input(2, "dY", "Gradient of output", "T")
      .Output(0, "dX", "Gradient of input", "T")
      .Attr(
          "axis",
          "Which axis to gather on. Negative value means "
          "counting dimensions from the back. Accepted range in [-r, r-1]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain input shape to integer tensors.")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherElementsGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("GatherElementsGrad")
      .Attr(
          "axis",
          "Which axis to scatter on. Negative value means "
          "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(
          0,
          "dY",
          "Tensor of rank r >=1 (same rank and shape as indices)",
          "T")
      .Input(1, "shape", "Shape of the GatherElements input data.", "I")
      .Input(
          2,
          "indices",
          "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
          "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
          "Tind")
      .Output(0, "dX", "Tensor of rank r >= 1 (same rank as input).", "T")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain input shape to integer tensors.")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Input and output types can be of any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(DivGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output", "T")
      .Input(1, "A", "dividend", "T")
      .Input(2, "B", "divisor", "T")
      .Output(0, "dA", "Gradient of dividend", "T", OpSchema::Optional)
      .Output(1, "dB", "Gradient of divisor", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to numeric tensors.")
      .FunctionBody([]() {
        auto nodes = ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}

           // Get input shapes and dynamic reduction axes.
           {{"shape_A"}, "Shape", {"A"}},
           {{"shape_B"}, "Shape", {"B"}},
           {{"axes_A", "axes_B"}, "BroadcastGradientArgs", {"shape_A", "shape_B"}},

           // dA = reshape(reduce_sum(dY / B, axes_A), shape_A)
           {{"dY_over_B"}, "Div", {"dY", "B"}},
           {{"reduce_dA"}, "ReduceSumTraining", {"dY_over_B", "axes_A"},
            {ONNX_NAMESPACE::MakeAttribute("noop_with_empty_axes", int64_t(1))}},
           {{"dA"}, "Reshape", {"reduce_dA", "shape_A"}},

           // dB = reshape(reduce_sum(dY * -A / (B * B)), axes_B), shape_B)
           {{"B_squared"}, "Mul", {"B", "B"}},
           {{"minus_A"}, "Neg", {"A"}},
           {{"minus_A_over_B_squared"}, "Div", {"minus_A", "B_squared"}},
           {{"pre_reduce_dB"}, "Mul", {"dY", "minus_A_over_B_squared"}},
           {{"reduce_dB"}, "ReduceSumTraining", {"pre_reduce_dB", "axes_B"},
            {ONNX_NAMESPACE::MakeAttribute("noop_with_empty_axes", int64_t(1))}},
           {{"dB"}, "Reshape", {"reduce_dB", "shape_B"}}});

           for (size_t contrib_node_index : {2, 4, 10}) {
             nodes[contrib_node_index].set_domain(kMSDomain);
           }
           return nodes;
      }())
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
          propagateElemTypeFromTensorInputToOutput(ctx, 0, i);
          propagateShapeFromInputToOutput(ctx, i + 1, i);
        }
      });

  //TODO: Move this to the right location. Its only here for quick experimentation.
  //TODO: Use the mutli weight / grad version.
  ONNX_CONTRIB_OPERATOR_SCHEMA(SGDOptimizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "ETA", "Learning Rate", "L")
      .Input(1, "W", "Original weight(s)", "T")
      .Input(2, "G", "Gradient of Weight(s)", "T")
      .Output(0, "NW", "Updated weight(s)", "T", OpSchema::Optional)
      .Output(1, "NG", "Updated gradients(s)", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "L",
          {"float"},
          "Constrain learning rate to float");

  // TODO: This is copied from onnx schemas. When the change is in and we update this can be removed.
  // For Brevity documentation was not copied
  ONNX_CONTRIB_OPERATOR_SCHEMA(AdamOptimizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "R", "The initial learning rate.", "T1")
      .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
      .Input(
          2,
          "weights",
          "weights to optimize.",
          "T3")
      .Input(
          3,
          "gradients",
          "gradients computed in this iteration.",
          "T_GRAD")
      .Input(
          4,
          "moment_1",
          "exponentially averaged historical gradients.",
          "T4")
      .Input(
          5,
          "moment_2",
          "exponentially averaged historical squared gradients.",
          "T4")
      .Input(
          6,
          "mixed_precision_weights",
          "FP16 or BFloat16 weights to optimize.",
          "T_MIXED_PRECISION_FP",
          OpSchema::Optional)
      .Input(
          7,
          "loss_scale",
          "loss scale for mixed precision training",
          "T3",
          OpSchema::Optional)
      .Input(
          8,
          "global_gradient_norm",
          "Global gradient norm.",
          "T_GRAD_NORM",
          OpSchema::Optional)
      .Input(
          9,
          "update_signal",
          "This signal indicates if weight tensors should be updated.",
          "T_BOOL",
          OpSchema::Optional)
      .Output(
          0,
          "new_T",
          "New update count.",
          "T2")
      .Output(
          1,
          "new_moment_1",
          "New averaged gradients.",
          "T4")
      .Output(
          2,
          "new_moment_2",
          "New averaged squared gradients.",
          "T4")
      .Output(
          3,
          "new_weights",
          "New weights.",
          "T3",
          OpSchema::Optional)
      .Output(
          4,
          "new_gradients",
          "New gradients.",
          "T_GRAD",
          OpSchema::Optional)
      .Output(
          5,
          "new_mixed_precision_weights",
          "New FP16 or BFloat16 weights",
          "T_MIXED_PRECISION_FP",
          OpSchema::Optional)
      .Attr(
          "alpha",
          "Coefficient of previous gradient in running average.",
          AttributeProto::FLOAT,
          0.9f)
      .Attr(
          "beta",
          "Coefficient of previous squared gradient in running average."
          "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
          "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
          AttributeProto::FLOAT,
          0.999f)
      .Attr(
          "lambda",
          "Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to 0, "
          "which means no regularization.",
          AttributeProto::FLOAT,
          0.0f)
      .Attr(
          "epsilon",
          "Small scalar to avoid dividing by zero.",
          AttributeProto::FLOAT,
          1e-8f)
      .Attr(
          "do_bias_correction",
          "Compute unbiased 1st and 2nd momentums.",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "weight_decay_mode",
          "Modes for applying weight decay, "
          "0 means applying decay before weight update, "
          "1 means applying decay after weight update.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain learning rate to float")
      .TypeConstraint(
          "T2",
          {"int64"},
          "Constrain step count to 64-bit integer")
      .TypeConstraint(
          "T3",
          {"tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T4",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_GRAD",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_MIXED_PRECISION_FP",
          {"tensor(float16)", "tensor(bfloat16)"},
          "Constrain input types to float16 or bfloat16 tensors.")
      .TypeConstraint(
          "T_GRAD_NORM",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(LambOptimizer, RegisterLambOpSchema);

  ONNX_CONTRIB_OPERATOR_SCHEMA(InPlaceAccumulator)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("in-place accumulator for tensors")
      .Input(0, "old_sum", "historical result of accumulator", "T")
      .Input(1, "value", "the value that will be added to the accumulator", "T_GRAD")
      .Input(2, "update_signal", "This signal indicates if tensor should be updated", "T_BOOL", OpSchema::Optional)
      .Output(0, "new_sum", "updated result of accumulator", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T_GRAD",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ZeroGradient)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("reset the accumulator for gradient")
      .Input(0, "old_gradient", "historical result of accumulated gradient", "T1")
      .Input(1, "reset_signal", "if this input is available, it is ready to reset the accumulator", "T2")
      .Output(0, "zero_gradient", "reset the gradient", "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output gradient types to float tensors.")
      .TypeConstraint(
          "T2",
          OpSchema::all_tensor_types(),
          "reset_signal can be of any tensor type.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  // TODO: Depreacate this schema when training support is udpated to opset-12
  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherND)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .Attr(
          "batch_dims",
          "The number of batch dims. The gather of indexing starts from dimension of data[batch_dims:]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "data", "Tensor of rank r >= 1.", "T")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Output(0, "output", "Tensor of rank q-1+r-indices[-1].", "T")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types_with_bfloat(),
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
        auto batch_dims = ctx.getAttribute("batch_dims");
        int64_t batch_dims_data = batch_dims ? static_cast<int>(batch_dims->i()) : 0;
        if (data_rank < 1 || indices_rank < 1) {
          fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
        }
        auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value() + batch_dims_data;
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
)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherNDGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "batch_dims",
          "The number of batch dims. The gather of indexing starts from dimension of data[batch_dims+1:]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "shape", "The shape of source data input of GatherND.", "T1")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Input(2, "update", "The gradient of the output.", "T")
      .Output(0, "output", "Tensor graident of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int64)"},
          "Constrain indice type to int32 or int64")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain shape type to int64");

  // TODO: push this to ONNX
  static const char* reduction_doc =
      "Type of reduction to apply to loss: none, sum, mean(default). "
      "'none': the output is the loss for each sample in the batch."
      "'sum': the output will be summed. "
      "'mean': the sum of the output will be divided by the batch_size.";

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropy)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, N-D input of shape (-1, num_classes).", "T")
      .Input(1, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "Y", "loss.", "T")
      .Output(1, "log_prob", "logsoftmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropy)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(MeanSquaredDistance)
      .SetDomain(kOnnxDomain) 
      .SinceVersion(12) 
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .AllowUncheckedAttributes()
      .Input(0, "scores", "The predicted outputs.", "T")
      .Input(
            1,
            "labels",
            "The ground truth output tensor, same dimensions as 'scores'.",
            "T")
      .Input(
            2,
            "weights",
            "Weights acts as a coefficient for the loss, it should be "
            "broadcastable to shape of 'scores'.",
            "T",
            OpSchema::Optional)
      .Output(0, "output", 
            "Weighted loss float Tensor. If reduction is none, this has the "
            "shape of [batch_size]; otherwise, it is scalar.",
            "T")
      .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")  
      .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyMSD)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
            std::string reduction = getAttribute(ctx, "reduction", "mean");
            if (reduction.compare("none") == 0) {
              if (hasInputShape(ctx, 0)) {
                  propagateShapeFromInputToOutput(ctx, 0, 0);
              }
            } else {
                updateOutputShape(ctx, 0, {});
            }});

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), N-D input of shape (-1, num_classes).", "T")
      .Input(2, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodAllReduce)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduce_op", "Reduce operation supported by Horovod. Valid values are: AVERAGE(0), SUM(1) or ADASUM(2)", AttributeProto::INT, int64_t(1))
      .Input(0, "input", "tensor to be reduced", "T")
      .Output(0, "output", "reduced tensor", "T")
      .Output(1, "ready", "true when reduced tensor is ready", "B")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain to bool tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodBarrier)
      .SetDomain(kOnnxDomain)
      .SetDoc("Waits for one or more async Horovod operators to complete")
      .SinceVersion(9)
      .Input(0, "input", "input tensor", "T")
      .Input(1, "input_ready", "one or more bool tensors to wait on", "B", OpSchema::Variadic)
      .Output(0, "output", "output tensor", "T")
      .Output(1, "output_ready", "output tensor is ready", "B")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllReduce)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllGather)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be sent", "T", OpSchema::Variadic)
      .Output(0, "output", "gathered tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclReduceScatter)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced and scattered", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, (N+1)-D input of shape (-1, num_classes).", "T")
      .Input(1, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(2, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "Y", "loss.", "T")
      .Output(1, "log_prob", "logsoftmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropy)DOC")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        std::string reduction = getAttribute(ctx, "reduction", "mean");
        if (reduction.compare("none") == 0) {
          if (hasInputShape(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 1, 0);
          }
        } else {
          updateOutputShape(ctx, 0, TensorShapeProto());
        }

        if (ctx.getNumOutputs() == 2) {
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), (N+1)-D input of shape (batch_size).", "T")
      .Input(2, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(3, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyLossGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Attr(
          "ignore_index",
          "Specifies a target value that is ignored and does not contribute to the input gradient.",
          AttributeProto::INT,
          false)
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), (N+1)-D input of shape (batch_size).", "T")
      .Input(2, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(3, "weight", "weight for each sample. The shape is 1-D tensor.", "T", OpSchema::Optional)
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SoftmaxCrossEntropyLossGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(BiasDropout)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("BiasDropout")
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL_VALUE)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "bias", "The bias input, a vector with the same shape as last dim of data", "T")
      .Input(2, "residual", "The residual input, must have the same shape as data", "T", OpSchema::Optional)
      .Input(3, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
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
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceSumTraining)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("ReduceSumTraining")
      .Attr("keepdims",
            "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Attr("noop_with_empty_axes",
            "Perform reduction or not when axes is empty, default false mean perform reduction."
            "when axes is empty and this attribute is set to true, input tensor will not be reduced,"
            "thus output tensor would be equivalent to input tensor.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .AllowUncheckedAttributes()
      .Input(0, "data", "An input tensor.", "T")
      .Input(1, "axes",
             "A list of integers, along which to reduce. The default is to reduce over "
             "all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data).",
             "tensor(int64)")
      .Output(0, "reduced", "Reduced output tensor.", "T")
      .TypeConstraint(
          "T",
          OpSchema::numeric_types_for_math_reduction(),
          "Constrain input and output types to high-precision numeric tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        // skip if axes is not an initializer
        auto axes_proto = ctx.getInputData(1);
        if (axes_proto == nullptr) {
          return;
        }

        int64_t keep_dims = 1;
        auto attr_proto = ctx.getAttribute("keepdims");
        if (attr_proto) {
          keep_dims = attr_proto->i();
        }
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        auto output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        std::vector<int64_t> axes_values = ParseData<int64_t>(axes_proto);
        std::vector<int64_t> axes;
        axes.reserve(axes_values.size());
        for (int64_t axis : axes_values) {
          axes.push_back(HandleNegativeAxis(axis, input_ndim));
        }

        for (int i = 0; i < input_ndim; ++i) {
          // axes empty means reduce all dim
          if (!axes.empty() &&
              std::find(axes.begin(), axes.end(), i) == axes.end()) {
            auto dim = output_shape->add_dim();
            dim->CopyFrom(input_shape.dim(i));
          } else {
            if (keep_dims == 1) {
              auto dim = output_shape->add_dim();
              dim->set_dim_value(1);
            }
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SplitTraining)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SplitTraining")
      .Attr("axis",
            "Which axis to split on. "
            "A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] "
            "where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .AllowUncheckedAttributes()
      .Input(0, "input", "The tensor to split", "T")
      .Input(1, "split", "length of each output", "tensor(int64)")
      .Output(0,
              "outputs",
              "One or more outputs forming list of tensors after splitting",
              "T",
              OpSchema::Variadic)
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain input and output types to all tensor types.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
          propagateElemTypeFromInputToOutput(ctx, 0, i);
        }
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        // skip if split is not an initializer
        auto split_proto = ctx.getInputData(1);
        if (split_proto == nullptr) {
          return;
        }
        std::vector<int64_t> split = ParseData<int64_t>(split_proto);

        if (!ctx.getInputType(0)->tensor_type().has_shape()) {
          return;
        }
        const auto& shape = ctx.getInputType(0)->tensor_type().shape();
        int rank = shape.dim_size();
        int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
        if (axis < -rank || axis >= rank) {
          fail_type_inference(
              "Invalid value of attribute 'axis'. Rank=",
              rank,
              " Value=",
              axis);
        }
        if (axis < 0) {
          axis += rank;
        }
        const auto& splitDim = shape.dim(axis);
        if (!splitDim.has_dim_value()) {
          return;
        }
        int splitDimValue = static_cast<int>(splitDim.dim_value());
        if (split.empty()) {
          int chunkSize =
              splitDimValue / static_cast<int>(ctx.getNumOutputs());
          int leftOver = splitDimValue -
                         (chunkSize * static_cast<int>(ctx.getNumOutputs()));
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
            split.push_back(i < leftOver ? chunkSize + 1 : chunkSize);
          }
        }
        for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
          *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() =
              shape;
          ctx.getOutputType(i)
              ->mutable_tensor_type()
              ->mutable_shape()
              ->mutable_dim(axis)
              ->set_dim_value(split[i]);
        }

      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConcatTraining)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Concatenate a list of tensors into a single tensor")
      .Attr("axis", "Which axis to concat on", AttributeProto::INT)
      .Input(0,
             "inputs",
             "List of tensors for concatenation",
             "T",
             OpSchema::Variadic)
      .Output(0, "concat_result", "Concatenated tensor", "T")
      .Output(1, "per_input_length",
              "Vector of length of each concatenated "
              "input along the 'axis' dimension",
              "Tint")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain output types to any tensor type.")
      .TypeConstraint(
          "Tint",
          {"tensor(int64)"},
          "Constrain output len types to integer type.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        auto numInputs = ctx.getNumInputs();
        if (numInputs < 1 ||
            !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
          return;
        }

        auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

        auto axisAttr = ctx.getAttribute("axis");
        if (!axisAttr) {
          fail_shape_inference("Required attribute axis is missing");
        }
        int64_t axis = static_cast<int64_t>(axisAttr->i());
        axis = HandleNegativeAxis(axis, rank);

        bool all_lengths_known = true;
        int total_length = 0;

        auto* output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        for (int64_t i = 0; i < rank; ++i) {
          output_shape->add_dim();
        }

        ONNX_NAMESPACE::TensorShapeProto per_input_len_shape;
        per_input_len_shape.add_dim()->set_dim_value(numInputs);
        updateOutputShape(ctx, 1, per_input_len_shape);

        for (size_t i = 0; i < numInputs; i++) {
          const auto& shape = ctx.getInputType(i)->tensor_type().shape();
          if (shape.dim_size() != rank)
            fail_shape_inference("All inputs to Concat must have same rank");
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
          output_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(total_length);
        }

      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropout)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TrainableDropout")
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL_VALUE)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "output", "The output.", "T")
      .Output(1, "mask", "The output mask.", "T2", OpSchema::Optional)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropoutGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("TrainableDropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask tensor of the dropout. ", "T2")
      .Input(2, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "dx", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DropoutGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("DropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask tensor of the dropout. ", "T2")
      .Input(2, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Input(3, "training_mode",
             "If set to true then it indicates dropout is being used for training. It is an optional value hence unless "
             "specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where "
             "nothing will be dropped from the input data and if mask is requested as output it will contain all ones.",
             "T2",
             OpSchema::Optional)
      .Output(0, "dx", "Gradient of the input.", "T")
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
          "Constrain 'mask' and 'training_mode' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(BroadcastGradientArgs)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(
          "Returns the reduction axes for computing gradients of s0 op s1 with broadcast."
          "The ouput axes are deterministic from last to first. "
          "Output is an empty vector when no reduction is necessary for the corresponding input.")
      .Input(0, "a_shape", "The 1st input shape as Tensor.", "T")
      .Input(1, "b_shape", "The 2nd input shape as Tensor.", "T")
      .Output(0, "a_axes", "The reduction axes for 1st input, last to first.", "T", OpSchema::Optional)
      .Output(1, "b_axes", "The reduction axes for 2nd input, last to first.", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(int64)"},
          "Constrain input and output types to 64-bit integer.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // NOTE: Both outputs are optional.
        for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
          updateOutputElemType(ctx, i, ONNX_NAMESPACE::TensorProto::INT64);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeEncoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "uncompressed output", "T")
      .Output(1, "Y1", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeDecoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X1", "dummy input for late decoding", "T")
      .Input(1, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SinGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Sin")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "Sin output's grad", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Sin input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"X_1"}, "Cos", {"X"}},
           {{"dX"}, "Mul", {"X_1", "dY"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReshapeGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Reshape")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Reshape output's grad", "T")
      .Output(0, "dX", "REshape input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"x_shape"}, "Shape", {"X"}},
           {{"dX"}, "Reshape", {"dY", "x_shape"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryScalar)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryScalar")
      .Attr("tags", "The tags corresponding to each input scalar.", AttributeProto::STRINGS)
      .Input(0, "input", "The scalar tensor to summarize as simple values.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(bfloat16)"},
          "Constrain input type to float and bool tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryHistogram)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryHistogram")
      .Attr("tag", "The tag corresponding to the histogram data.", AttributeProto::STRING)
      .Input(0, "input", "The scalar tensor to produce a histogram over.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input type to float tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryMerge)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryMerge")
      .Input(0, "input", "One or more serialized Tensorboard Summary tensors to merge into a single Summary.", "S", OpSchema::Variadic)
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryText)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryText")
      .Attr("tag", "The tag corresponding to the text data.", AttributeProto::STRING)
      .Input(0, "input", "The string tensor to render in the Tensorboard Text dashboard.", "S")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GeluGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("GeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalizationGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalizationGrad")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .AllowUncheckedAttributes()
      .Input(0, "Y_grad", "The gradient tensor from output.", "T")
      .Input(1, "X", "Input data tensor from the forward path", "T")
      .Input(2, "scale", "Scale tensor.", "T")
      .Input(3, "mean", "mean of X.", "U")
      .Input(4, "inv_std_var", "inverse std variance of X.", "U")
      .Output(0, "X_grad", "Gradient of the input.", "T")
      .Output(1, "scale_grad", "Gradient of the scale.", "T")
      .Output(2, "bias_grad", "Gradient of the bias.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)", "tensor(bfloat16)"},
          "Constrain mean and inv_std_var to float tensors.");
  
    ONNX_CONTRIB_OPERATOR_SCHEMA(SimplifiedLayerNormalizationGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SimplifiedLayerNormalizationGrad")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .AllowUncheckedAttributes()
      .Input(0, "Y_grad", "The gradient tensor from output.", "T")
      .Input(1, "X", "Input data tensor from the forward path", "T")
      .Input(2, "scale", "Scale tensor.", "T")
      .Input(3, "inv_std_var", "inverse std variance of X.", "U")
      .Output(0, "X_grad", "Gradient of the input.", "T")
      .Output(1, "scale_grad", "Gradient of the scale.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain mean and inv_std_var to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(InvertibleLayerNormalizationGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalizationGrad")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .AllowUncheckedAttributes()
      .Input(0, "Y_grad", "The gradient tensor from output.", "T")
      .Input(1, "Y", "Output data tensor from the forward path", "T")
      .Input(2, "scale", "Scale tensor.", "T")
      .Input(3, "bias", "Bias tensor.", "T")
      .Input(4, "inv_std_var", "inverse std variance of X.", "U")
      .Output(0, "X_grad", "Gradient of the input.", "T")
      .Output(1, "scale_grad", "Gradient of the scale.", "T")
      .Output(2, "bias_grad", "Gradient of the bias.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain mean and inv_std_var to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(BatchNormalizationGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("BatchNormalization")
      .Attr("epsilon",
            "epsilon value",
            AttributeProto::FLOAT)
      .Input(0, "dY", "Gradient output from previous node", "T")
      .Input(1, "X", "Input", "T")
      .Input(2, "scale", "Scale tensor", "T")
      .Input(3, "mean", "Mean of X", "T")
      .Input(4, "variance", "Variance of X", "T")
      .Output(0, "X_grad", "Gradient of the input", "T")
      .Output(1, "scale_grad", "Gradient of the scale", "T")
      .Output(2, "bias_grad", "Gradient of the bias", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Group)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("if all the inputs are available, the output will be true")
      .Input(0, "input_tensors", "list of dependency tensors", "T", OpSchema::Variadic, false)
      .Output(0, "done", "all the dependency tensors are ready", "B")
      .TypeConstraint("T", OpSchema::all_tensor_types_with_bfloat(), "All Tensor types")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsFinite)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("IsFinite")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Constrain the output to a boolean tensor.")
      .Input(
          0,
          "X",
          "The input tensor.",
          "T")
      .Output(
          0,
          "Y",
          "The output tensor. Its shape is the same as the input.",
          "T1");

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsAllFinite)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("IsAllFinite")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
          "T");

  static const char* All_doc = R"DOC(
Return true if all elements are true and false otherwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(All)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(bool)"},
          "Constrain input and output types to boolean tensors.")
      .SetDoc(All_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MixedPrecisionScale)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("MixedPrecisionScale")
      .Input(0, "S", "scale", "ScaleT")
      .Input(1, "X", "inputs", "SrcT", OpSchema::Variadic)
      .Output(0, "Y", "output", "DstT", OpSchema::Variadic)
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .Attr("fuse_outputs",
            "If true, fuse all outputs into one continous buffer.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .TypeConstraint(
          "SrcT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "ScaleT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain scale types to float tensors.")
      .TypeConstraint(
          "DstT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        bool fuse_outputs = static_cast<bool>(getAttribute(ctx, "fuse_outputs", int64_t(0)));
        if (fuse_outputs) {
          int64_t total_num_elements = 0;
          for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
            if (!hasInputShape(ctx, i))
              return;
            auto& input_shape = getInputShape(ctx, i);
            int rank = static_cast<int>(input_shape.dim_size());
            int64_t num_elements = multiplyDims(input_shape, 0, rank).dim_value();
            total_num_elements += num_elements;
          }

          ONNX_NAMESPACE::TensorShapeProto output_shape;
          output_shape.add_dim()->set_dim_value(total_num_elements);
          updateOutputShape(ctx, 0, output_shape);
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
        } else {
          for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
            propagateElemTypeFromAttributeToOutput(ctx, "to", i - 1);
            propagateShapeFromInputToOutput(ctx, i, i - 1);
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Scale)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Scale")
      .Input(0, "input", "Input tensor.", "T")
      .Input(1, "scale", "Scale scalar tensor.", "ScaleT")
      .Output(0, "output", "The scaled output tensor.", "T")
      .Attr("scale_down",
            "If true, the output tensor is input tensor devided by scale, "
            "otherwise, it's input tensor multiplied by scale. "
            "The default value is false.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "ScaleT",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
          "Constrain scale types to float and int64 tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(View)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("View. The output tensors are views of the input, according to the shapes provided.")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "Input tensor.", "T")
      .Input(1, "shapes", "Shapes of each view output. The shapes must adds up to the input buffer size.",
             "tensor(int64)",
             OpSchema::Variadic)
      .Output(0, "outputs", "Output tensors viewed according the shapes input. It has a one to one mapping to the shapes input",
              "T",
              OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceAllL2)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Multi-tensor version of ReduceL2.")
      .Input(0, "X", "inputs", "TIn", OpSchema::Variadic)
      .Output(0, "Y", "output", "TOut")
      .TypeConstraint(
          "TIn",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "TOut",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain scale types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Send)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Send data tensor to the specified destination.")
      .Input(0, "InputSignal", "Input control signal. It must be a scalar.", "TBool")
      .Input(1, "Remote", "Remote dst rank. It must be a scalar.", "TInt64")
      .Input(2, "Data", "Tensors to send.", "V", OpSchema::Variadic, false)
      .Output(0, "OutputSignal", "Output control signal. It must be a scalar.", "TBool")
      .Attr("tag", "The tag of the message carrying Data.",
            AttributeProto::INT)
      .Attr("element_types", "Element types of the sent tensors.",
            AttributeProto::INTS)
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeConstraint(
          "TBool",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() < 3) {
          fail_shape_inference("Send must have at least three inputs.");
        } else {
          auto& signal_input_shape = getInputShape(ctx, 0);
          if (static_cast<int>(signal_input_shape.dim_size()) != 0) {
            fail_shape_inference("InputSignal of Send must be a scalar.");
          }
          auto& remote_input_shape = getInputShape(ctx, 1);
          if (static_cast<int>(remote_input_shape.dim_size()) != 0) {
            fail_shape_inference("Remote of Send must be a scalar.");
          }

          checkSendInputTensorElemTypes(ctx, "element_types", ctx.getNumInputs() - 2);
        }

        if (ctx.getNumOutputs() != 1) {
          fail_shape_inference("Send must have one output.");
        }

        auto output_element_type = ctx.getOutputType(0)->mutable_tensor_type();
        output_element_type->set_elem_type(TensorProto::BOOL);
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        updateOutputShape(ctx, 0, {});
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Recv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Receive a tensor from the the specified source.")
      .Input(0, "InputSignal", "Input control signal. It must be a scalar.", "TBool")
      .Input(1, "Remote", "Remote src rank. It must be a scalar.", "TInt64")
      .Output(0, "OutputSignal", "Output control signal. It must be a scalar.", "TBool")
      .Output(1, "Data", "The Received tensors.", "V", OpSchema::Variadic, false)
      .Attr("tag", "The tag of the message carrying Data.",
            AttributeProto::INT)
      .Attr("element_types", "Element types of the received tensors.",
            AttributeProto::INTS)
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeConstraint(
          "TBool",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() != 2) {
          fail_shape_inference("Recv must have two inputs.");
        } else {
          auto& signal_input_shape = getInputShape(ctx, 0);
          if (static_cast<int>(signal_input_shape.dim_size()) != 0) {
            fail_shape_inference("InputSignal of Recv must be a scalar.");
          }
          auto& remote_input_shape = getInputShape(ctx, 1);
          if (static_cast<int>(remote_input_shape.dim_size()) != 0) {
            fail_shape_inference("Remote of Recv must be a scalar.");
          }
        }

        if (ctx.getNumOutputs() < 2) {
          fail_shape_inference("Recv must have at least two outputs.");
        }

        updateOutputShape(ctx, 0, {});
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
        propagateRecvOutputTensorElemTypes(ctx, "element_types", ctx.getNumOutputs() - 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MegatronF)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "The input data as Tensor.", "T")
      .Output(0, "output", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MegatronG)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "The input data as Tensor.", "T")
      .Output(0, "output", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SliceGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output", "T")
      .Input(1, "shape", "Shape of the Slice input X.", "I")
      .Input(2, "starts", "Tensor of starting indices of corresponding axis in axes", "Tind")
      .Input(3, "ends", "Tensor of starting indices of corresponding axis in 'axes'", "Tind")
      .Input(4, "axes", "Tensor of axes that `starts` and `ends` apply to", "Tind", OpSchema::Optional)
      .Input(5, "steps", "Tensor of slice step of corresponding axis in `axes`", "Tind", OpSchema::Optional)
      .Output(0, "dX", "Gradient of input", "T")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain input shape to integer tensors.")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types_with_bfloat(),
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(FastGeluGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("FastGeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(BiasGeluGrad_dX)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Computes dX for BiasGeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Input(2, "B", "The bias tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(BiasFastGeluGrad_dX)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Computes dX for FastGeluGrad with bias")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Input(2, "B", "The bias tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(RecordEvent)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Record an event.")
      .Input(
          0,
          "EventIdentifier",
          "Event identifier to record.",
          "TInt64")
      .Input(
          1,
          "InputData",
          "Input data.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Output(
          0,
          "OutputData",
          "Output data.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 0)
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Allow inputs and outputs to be any kind of tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() < ctx.getNumOutputs() + 1)
          fail_shape_inference("RecordEvent must have at least (num_outputs + 1) inputs.");

        // note: if num_input > num_output + 1,
        // the additional inputs (idx >= num_ouput + 1) are regarded as dependencies
        // which are only used for maintain topological order
        for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
          propagateElemTypeFromInputToOutput(ctx, i + 1, i);
          auto typeProto = ctx.getInputType(i + 1);
          if (!hasShape(*typeProto)) {
            continue;
          }
          propagateShapeFromInputToOutput(ctx, i + 1, i);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(WaitEvent)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Wait for an event to be recorded.")
      .Input(
          0,
          "EventIdentifier",
          "Event identifier to record.",
          "TInt64")
      .Input(
          1,
          "InputData",
          "Input data.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Output(
          0,
          "OutputData",
          "Output data.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Allow inputs and outputs to be any kind of tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() < ctx.getNumOutputs() + 1)
          fail_shape_inference("WaitEvent must have at least (num_outputs + 1) inputs.");
        if (ctx.getNumOutputs() < 1)
          fail_shape_inference("WaitEvent must have at least 1 output.");

        // note: if num_input > num_output + 1,
        // the additional inputs (idx >= num_ouput + 1) are regarded as dependencies
        // which are only used for maintain topological order
        for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
          propagateElemTypeFromInputToOutput(ctx, i + 1, i);
          auto typeProto = ctx.getInputType(i + 1);
          if (!hasShape(*typeProto)) {
            continue;
          }
          propagateShapeFromInputToOutput(ctx, i + 1, i);
        }
      });
}
}  // namespace training
}  // namespace onnxruntime
