// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/training_op_defs.h"

#include <math.h>
#include <sstream>
#include "core/graph/op.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/providers/common.h"
#include "onnx/defs/function.h"
#include "orttraining/core/framework/distributed_run_context.h"

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

TensorProto ToDimensionOneFloatTensor(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneTensor(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

bool BuildContextDependentFunctionBodyNllLossInternal(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back(
      {{"const_zero"},
       "Constant",
       {},
       {MakeAttribute("value", ToDimensionOneTensor(0))}});

  body.push_back(
      {{"const_one"},
       "Constant",
       {},
       {MakeAttribute("value", ToDimensionOneTensor(1))}});

  body.push_back(
      {{"expanded_target"},
       "Unsqueeze",
       {"target"},
       {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  if (!ctx.hasInput(3)) {
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "expanded_target"},
         {MakeAttribute("axis", (int64_t)1)}});

    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element"}});

    body.push_back(
        {{"loss_N1dd"},
         "Slice",
         {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      if (ctx.getAttribute("reduction")->s() == "none") {
        body.push_back(
            {{"loss"},
             "Squeeze",
             {"loss_N1dd"},
             {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      } else {
        body.push_back(
            {{"loss_Ndd"},
             "Squeeze",
             {"loss_N1dd"},
             {MakeAttribute("axes", std::vector<int64_t>({1}))}});
        if (ctx.getAttribute("reduction")->s() == "mean") {
          body.push_back(
              {{"loss"},
               "ReduceMean",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        } else {
          body.push_back(
              {{"loss"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    } else {
      body.push_back({{"weight_gather"}, "Gather", {"weight", "target"}});
      body.push_back(
          {{"loss_unweighted"},
           "Squeeze",
           {"loss_N1dd"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      if (ctx.getAttribute("reduction")->s() == "none") {
        body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
      } else {
        body.push_back(
            {{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
        if (ctx.getAttribute("reduction")->s() == "mean") {
          body.push_back(
              {{"loss_sum"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back(
              {{"weight_gather_sum"},
               "ReduceSum",
               {"weight_gather"},
               {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
        } else {
          body.push_back(
              {{"loss"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    }
  } else {
    body.push_back(
        {{"const_zero_target_typed"},
         "Sub",
         {"expanded_target", "expanded_target"}});
    body.push_back(
        {{"expanded_target_int64"},
         "Cast",
         {"expanded_target"},
         {MakeAttribute(
             "to",
             (int64_t)TensorProto_DataType::TensorProto_DataType_INT64)}});

    body.push_back(
        {{"mask"}, "Equal", {"expanded_target_int64", "ignore_index"}});
    body.push_back(
        {{"transform_targets"},
         "Where",
         {"mask", "const_zero_target_typed", "expanded_target"}});
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "transform_targets"},
         {MakeAttribute("axis", (int64_t)1)}});
    body.push_back(
        {{"const_zero_float"},
         "Constant",
         {},
         {MakeAttribute("value", ToDimensionOneFloatTensor(0.0f))}});
    if (!float_input) {
      body.push_back(
          {{"const_zero_casted"},
           "Cast",
           {"const_zero_float"},
           {MakeAttribute("to", static_cast<int64_t>(input_type))}});
    }
    body.push_back(
        {{"input_gather_element_transform"},
         "Where",
         {"mask", float_input ? "const_zero_float" : "const_zero_casted", "input_gather_element"}});
    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element_transform"}});
    body.push_back(
        {{"loss_N1dd"},
         "Slice",
         {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      body.push_back(
          {{"squeeze_mask"},
           "Squeeze",
           {"mask"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});

      body.push_back(
          {{"const_one_float"},
           "Constant",
           {},
           {MakeAttribute("value", ToDimensionOneFloatTensor(1.0f))}});
      if (!float_input) {
        body.push_back(
            {{"const_one_casted"},
             "Cast",
             {"const_one_float"},
             {MakeAttribute("to", static_cast<int64_t>(input_type))}});
      }
      body.push_back(
          {{"weight_gather"},
           "Where",
           {"squeeze_mask", float_input ? "const_zero_float" : "const_zero_casted",
            float_input ? "const_one_float" : "const_one_casted"}});

    } else {
      body.push_back(
          {{"weight_gather_temp"}, "Gather", {"weight", "transform_targets"}});

      body.push_back(
          {{"weight_gather_temp_1"},
           "Where",
           {"mask", float_input ? "const_zero_float" : "const_zero_casted", "weight_gather_temp"}});

      body.push_back(
          {{"weight_gather"},
           "Squeeze",
           {"weight_gather_temp_1"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    }

    body.push_back(
        {{"loss_unweighted"},
         "Squeeze",
         {"loss_N1dd"},
         {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
    } else {
      body.push_back(
          {{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
      if (ctx.getAttribute("reduction")->s() == "mean") {
        body.push_back(
            {{"loss_sum"},
             "ReduceSum",
             {"loss_Ndd"},
             {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back(
            {{"weight_gather_sum"},
             "ReduceSum",
             {"weight_gather"},
             {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
      } else {
        body.push_back(
            {{"loss"},
             "ReduceSum",
             {"loss_Ndd"},
             {MakeAttribute("keepdims", (int64_t)0)}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
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
          "max_norm_clip",
          "clip threshold of gradients.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 1.f))
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
      .Input(1, "Y", "Input tensor", "T")
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
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
            // SoftmaxGrad computes dX = Y * ( dY - dot(Y, dY))
            // ONNX does not have a dot product, which can be simulated as a pointwise-multiplication ("Mul"),
            // followed by a "ReduceSum". Unfortunately, the treatment of "axis" is different in "SoftmaxGrad"
            // and "ReduceSum". If axis=k for SoftmaxGrad, we need to specify [k, ..., n-1] as the axes of
            // reduction for "ReduceSum", after accounting for negative-axis specification.
            // An alternative solution would be to Flatten inputs to 2D and then reshape output back to original shape.
            // Hopefully, many of these ops can be optimized away in the common-case of statically-known shapes.

            auto* axis_attr = ctx.getAttribute("axis");
            int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : 1;
            auto zero1d = ToTensor(std::vector<int64_t>({0}));
            zero1d.add_dims(1);

            // nodes: {outputs, op, inputs, attributes}

            // First, convert axis specification k to reduction axes [k, k+1, ..., n-1]
            std::vector<FunctionBodyHelper::NodeDef> body{
                FunctionBodyHelper::Const<int64_t>("one", 1),
                FunctionBodyHelper::Const<int64_t>("k", axis),
                {{"axis_zero"}, "Constant", {}, {{"value", zero1d}}},
                {{"shape"}, "Shape", {"dY"}},
                {{"n_as_vector"}, "Shape", {"shape"}},
                {{"n"}, "Squeeze", {"n_as_vector", "axis_zero"}},
            };

            // For negative axis, add n to axis-value k; then use Range(...).
            if (axis >= 0) {
              body.push_back({{"reduction_axes"}, "Range", {"k", "n", "one"}});
            } else {
              body.push_back({{"n_plus_k"}, "Add", {"n", "k"}});
              body.push_back({{"reduction_axes"}, "Range", {"n_plus_k", "n", "one"}});
            }

            // compute dX = Y * ( dY - dot(Y, dY)) = Y * ( dY - ReduceSum(Y * dY))
            body.push_back({{"a"}, "Mul", {"Y", "dY"}});
            body.push_back({{"b"}, "ReduceSum", {"a", "reduction_axes"}});
            body.push_back({{"c"}, "Sub", {"dY", "b"}});
            body.push_back({{"dX"}, "Mul", {"Y", "c"}});

            OperatorSetIdProto onnx_opset_13;
            onnx_opset_13.set_domain("");
            onnx_opset_13.set_version(13);

            return ONNX_NAMESPACE::BuildFunctionProto(functionProto, schema, body, {onnx_opset_13});
          });

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
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Input(2, "W", "Weight tensor", "T")
      .Output(0, "dX", "Gradient of X", "T", OpSchema::Optional)
      .Output(1, "dW", "Gradient of W", "T", OpSchema::Optional)
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
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
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
             {{"reduce_dA"}, "ReduceSumTraining", {"dY_over_B", "axes_A"}, {ONNX_NAMESPACE::MakeAttribute("noop_with_empty_axes", int64_t(1))}},
             {{"dA"}, "Reshape", {"reduce_dA", "shape_A"}},

             // dB = reshape(reduce_sum(dY * -A / (B * B)), axes_B), shape_B)
             {{"B_squared"}, "Mul", {"B", "B"}},
             {{"minus_A"}, "Neg", {"A"}},
             {{"minus_A_over_B_squared"}, "Div", {"minus_A", "B_squared"}},
             {{"pre_reduce_dB"}, "Mul", {"dY", "minus_A_over_B_squared"}},
             {{"reduce_dB"}, "ReduceSumTraining", {"pre_reduce_dB", "axes_B"}, {ONNX_NAMESPACE::MakeAttribute("noop_with_empty_axes", int64_t(1))}},
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
          "max_norm_clip",
          "clip threshold of gradients.",
          AttributeProto::FLOAT,
          1.0f)
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
          OpSchema::all_tensor_types_with_bfloat(),
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
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        std::string reduction = getAttribute(ctx, "reduction", "mean");
        if (reduction.compare("none") == 0) {
          if (hasInputShape(ctx, 1)) {
            // If no reduction is performed the shape of the loss looks
            // like the shape of the labels, without the onehot dimension.

            TensorShapeProto loss_shape;
            const TensorShapeProto& label_shape = ctx.getInputType(1)->tensor_type().shape();

            for (int i = 0; i != label_shape.dim_size() - 1; ++i) {
              *loss_shape.add_dim() = label_shape.dim(i);
            }

            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
                loss_shape;
          }

        } else {
          updateOutputShape(ctx, 0, {});
        }

        if (ctx.getNumOutputs() == 2) {
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      })
      .SetDoc(R"DOC(SoftmaxCrossEntropy)DOC");

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
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        propagateShapeFromInputToOutput(ctx, 1, 0);
      })
      .SetDoc(R"DOC(SoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllReduce)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type",
            "0 - global parallel group, 1 - data parallel group, "
            "2 - node local data parallel group, 3 - cross node data parallel group, "
            "4 - horozontal parallel, 5 - model parallel.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        assert(getAttribute(ctx, "group_type", 0) < static_cast<int64_t>(WorkerGroupType::WorkerGroupTypeCount));
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllGather)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type",
            "0 - global parallel group, 1 - data parallel group, "
            "2 - node local data parallel group, 3 - cross node data parallel group, "
            "4 - horozontal parallel, 5 - model parallel.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be sent", "T", OpSchema::Variadic)
      .Output(0, "output", "gathered tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        assert(getAttribute(ctx, "group_type", 0) < static_cast<int64_t>(WorkerGroupType::WorkerGroupTypeCount));
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclReduceScatter)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type",
            "0 - global parallel group, 1 - data parallel group, "
            "2 - node local data parallel group, 3 - cross node data parallel group, "
            "4 - horozontal parallel, 5 - model parallel.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced and scattered", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
#ifdef _DEBUG
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        assert(getAttribute(ctx, "group_type", 0) < static_cast<int64_t>(WorkerGroupType::WorkerGroupTypeCount));
      })
#endif
      ;

  ONNX_CONTRIB_OPERATOR_SCHEMA(AdasumAllReduce)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduce_algo", "Algorithms for Adasum. Valid values are: CpuReduction(1) or GpuHierarchicalReduction(2)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() != ctx.getNumOutputs())
          fail_shape_inference("AdasumAllReduce's input count must be equal to output count.");

        for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
          propagateElemTypeFromInputToOutput(ctx, i, i);
          auto typeProto = ctx.getInputType(i);
          if (!hasShape(*typeProto)) {
            continue;
          }
          propagateShapeFromInputToOutput(ctx, i, i);
        }
      });

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
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        propagateShapeFromInputToOutput(ctx, 1, 0);
      })
      .SetDoc(R"DOC(SoftmaxCrossEntropyLossGrad)DOC");

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
              "Tint",
              OpSchema::Optional)
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

        if (ctx.getNumOutputs() > 1) {
          ONNX_NAMESPACE::TensorShapeProto per_input_len_shape;
          per_input_len_shape.add_dim()->set_dim_value(numInputs);
          updateOutputShape(ctx, 1, per_input_len_shape);
        }

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

  ONNX_CONTRIB_OPERATOR_SCHEMA(DropoutGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("DropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask output of the dropout. ", "T2")
      .Input(2, "ratio",
             "Same value as the ratio input supplied to the dropout op with value in [0, 1). "
             "If this input is not specified, a default value of 0.5 is used.",
             "T1",
             OpSchema::Optional)
      .Input(3, "training_mode",
             "Same value as the training_mode input supplied to the dropout op. "
             "If this input is not specified, a default value of false is used.",
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
      })
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
            /* DropoutGrad (dy, mask, optional ratio, optional training_mode) => dX 
                 dX = Where (mask, dY / (1-ratio), 0)
              where ratio = 0.5 if not specified.

              TODO: Note that the above doesn't handle the case where training_mode=false and a non-zero
              value is specified for ratio. In general, it is unclear why we need the training_mode as an
              input here, since the Gradient will be used only for training.
            */
            OperatorSetIdProto onnx_opset_13;
            onnx_opset_13.set_domain("");
            onnx_opset_13.set_version(13);

            auto* tp = ctx.getInputType(0);
            if ((tp == nullptr) || (!tp->has_tensor_type()))
              return false;
            auto elem_type = (ONNX_NAMESPACE::TensorProto_DataType)tp->tensor_type().elem_type();
            if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16)
              return false;  // ONNX op Where doesn't support bfloat16 yet.

            if (ctx.hasInput(2)) {
              // ratio specified.
              std::vector<FunctionBodyHelper::NodeDef> body{
                  ONNX_NAMESPACE::Const("C0", 0.0f, elem_type),
                  ONNX_NAMESPACE::Const("C1", 1.0f, elem_type),
                  {{"ratio_elem_type"}, "Cast", {"ratio"}, {MakeAttribute("to", int64_t(elem_type))}},
                  {{"scale"}, "Sub", {"C1", "ratio_elem_type"}},
                  {{"scaled_dy"}, "Div", {"dy", "scale"}},
                  {{"dx"}, "Where", {"mask", "scaled_dy", "C0"}}};

              return ONNX_NAMESPACE::BuildFunctionProto(functionProto, schema, body, {onnx_opset_13});
            } else {
              // ratio not specified. Use a value of 0.5
              std::vector<FunctionBodyHelper::NodeDef> body{
                  ONNX_NAMESPACE::Const("C0", 0.0f, elem_type),
                  ONNX_NAMESPACE::Const("C1", 1.0f, elem_type),
                  ONNX_NAMESPACE::Const("ratio_elem_type", 0.5f, elem_type),
                  {{"scale"}, "Sub", {"C1", "ratio_elem_type"}},
                  {{"scaled_dy"}, "Div", {"dy", "scale"}},
                  {{"dx"}, "Where", {"mask", "scaled_dy", "C0"}}};

              return ONNX_NAMESPACE::BuildFunctionProto(functionProto, schema, body, {onnx_opset_13});
            }
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
      .Output(0, "Y", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeDecoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack1Encoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "1 bit compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(bool)", "tensor(float)"},
          "boolean or float uncompressed tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits represent 8 1-bit compressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::UINT8);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack1Decoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "1 bit compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits represent 8 1-bit compressed tensors.")
      .TypeConstraint(
          "T",
          {"tensor(bool)", "tensor(float)"},
          "boolean or float uncompressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack8Encoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits compressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::UINT8);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack8Decoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits compressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack16Encoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float16)"},
          "16 bits compressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::FLOAT16);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPack16Decoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "compressed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float16)"},
          "16 bits compressed tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPackMsfp15Encoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits represent 7 bit of sign and mantissa of compressed tensor, and remaining 1 bit (across TILE_SIZE worth of 8 bit elements) is used to store the shared exponent.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::UINT8);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistPackMsfp15Decoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(uint8)"},
          "8 bits represent 7 bit of sign and mantissa of compressed tensor, and remaining 1 bit (across TILE_SIZE worth of 8 bit elements) is used to store the shared exponent.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
      });

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
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
            /* Default GeluGrad computation: 
              dX = dY * [0.5f * [erf(sqrt(1/2)*X) + 1.0] + alpha*X*exp(-0.5f * X * X)]
            which expands to the following ONNX graph:
            */
            auto* tp = ctx.getInputType(0);
            if ((tp == nullptr) || (!tp->has_tensor_type()))
              return false;
            auto elem_type = (ONNX_NAMESPACE::TensorProto_DataType)tp->tensor_type().elem_type();
            double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
            std::vector<FunctionBodyHelper::NodeDef> body{
                ONNX_NAMESPACE::Const("C_Half", 0.5f, elem_type),
                ONNX_NAMESPACE::Const("C_One", 1.0f, elem_type),
                ONNX_NAMESPACE::Const("C_SqrtHalf", float(M_SQRT1_2), elem_type),
                ONNX_NAMESPACE::Const("C_MinusHalf", -0.5f, elem_type),
                ONNX_NAMESPACE::Const("C_alpha", kAlpha, elem_type),
                {{"ErfArg"}, "Mul", {"X", "C_SqrtHalf"}},
                {{"ErfTerm"}, "Erf", {"ErfArg"}},
                {{"PartialSum"}, "Add", {"ErfTerm", "C_One"}},
                {{"HalfPartialSum"}, "Mul", {"C_Half", "PartialSum"}},
                {{"AlphaX"}, "Mul", {"X", "C_alpha"}},
                {{"MinusHalfX"}, "Mul", {"C_MinusHalf", "X"}},
                {{"ExpArg"}, "Mul", {"MinusHalfX", "X"}},
                {{"ExpTerm"}, "Exp", {"ExpArg"}},
                {{"Term3"}, "Mul", {"AlphaX", "ExpTerm"}},
                {{"FullSum"}, "Add", {"HalfPartialSum", "Term3"}},
                {{"dX"}, "Mul", {"dY", "FullSum"}}};

            OperatorSetIdProto onnx_opset_13;
            onnx_opset_13.set_domain("");
            onnx_opset_13.set_version(13);

            return ONNX_NAMESPACE::BuildFunctionProto(functionProto, schema, body, {onnx_opset_13});
          });

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
          "Constrain mean and inv_std_var to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        propagateShapeFromInputToOutput(ctx, 1, 0);
        propagateElemTypeFromInputToOutput(ctx, 2, 1);
        propagateShapeFromInputToOutput(ctx, 2, 1);
        // The bias tensor has the same shape of the scale tensor.
        propagateElemTypeFromInputToOutput(ctx, 2, 2);
        propagateShapeFromInputToOutput(ctx, 2, 2);
      });

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

  ONNX_CONTRIB_OPERATOR_SCHEMA(PassThrough)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Barrier op with value pass through, outputs = inputs")
      .Input(0, "inputs", "input tensors", "T", OpSchema::Variadic, false)
      .Output(0, "outputs", "output tensors", "T", OpSchema::Variadic, false)
      .TypeConstraint("T", OpSchema::all_tensor_types_with_bfloat(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
          propagateElemTypeFromInputToOutput(ctx, i, i);
          if (hasInputShape(ctx, i)) {
            propagateShapeFromInputToOutput(ctx, i, i);
          }
        }
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
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "ScaleT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain scale types to float tensors.")
      .TypeConstraint(
          "DstT",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(BatchNormInternal)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Variant of BatchNormalization with additional output for saved_mean/inv_std_dev.")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("epsilon", "epsilon value", AttributeProto::FLOAT, 1e-5f)
      .Attr("momentum", "momentum value", AttributeProto::FLOAT, 0.9f)
      .Attr("training_mode", "true if training", AttributeProto::INT, static_cast<int64_t>(1))
      .Input(0, "X", "Input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(1, "scale", "Scale tensor of shape (C).", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(2, "B", "Bias tensor of shape (C).", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(3, "input_mean", "running mean tensor of shape (C).", "U", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(4, "input_var", "running variance tensor of shape (C).", "U", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(0, "Y", "The output tensor of the same shape as X", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(1, "running_mean", "The running mean after BN.", "U", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Output(2, "running_var", "Running var after BN", "U", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Output(3, "saved_mean", "Mean of the batch", "U", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Output(4, "saved_inv_std", "Inverse standard deviation for the batch", "U", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain mean and variance types to float tensors. It allows all float type for U.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        propagateShapeFromInputToOutput(ctx, 0, 0);

        Dim num_channels;

        unifyInputDim(ctx, 0, 1, num_channels);
        unifyInputDim(ctx, 1, 0, num_channels);
        unifyInputDim(ctx, 2, 0, num_channels);
        unifyInputDim(ctx, 3, 0, num_channels);
        unifyInputDim(ctx, 4, 0, num_channels);

        if (ctx.getAttribute("training_mode") &&
            static_cast<int>(ctx.getAttribute("training_mode")->i()) != 0) {
          if (ctx.getNumOutputs() != 5)
            fail_shape_inference(
                "This number of op outputs should be 5 when Training_mode = True, but it is not.");
        } else {
          if (ctx.getNumOutputs() != 1)
            fail_shape_inference(
                "This number of op outputs should be 1 when Training_mode = False, but it is not.");
        }

        if (ctx.getNumOutputs() > 1) {
          ONNX_NAMESPACE::TensorShapeProto outputs_shape;
          *outputs_shape.add_dim() = num_channels;  // channel

          propagateElemTypeFromInputToOutput(ctx, 3, 1);
          updateOutputShape(ctx, 1, outputs_shape);
          propagateElemTypeFromInputToOutput(ctx, 4, 2);
          updateOutputShape(ctx, 2, outputs_shape);
          propagateElemTypeFromInputToOutput(ctx, 3, 3);
          updateOutputShape(ctx, 3, outputs_shape);
          propagateElemTypeFromInputToOutput(ctx, 4, 4);
          updateOutputShape(ctx, 4, outputs_shape);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceAllL2)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Multi-tensor version of ReduceL2.")
      .Input(0, "X", "inputs", "TIn", OpSchema::Variadic)
      .Output(0, "Y", "output", "TOut")
      .TypeConstraint(
          "TIn",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "TOut",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain scale types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputShape(ctx, 0, {});
      });

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
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(YieldOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Yield Op.")
      .Input(0, "module_outputs", "Module outputs to be returned to pytorch.", "T", OpSchema::Variadic,
             /*is_homogeneous*/ false,
             /*min_arity*/ 1)
      /*
      For a situation where there are no trainable parameters in a model, the YieldOp minimum
      number of arguments expected for module_output_grad should be 0.
      */
      .Output(0, "module_outputs_grad", "Gradient of module outputs returned from pytorch.", "T", OpSchema::Variadic,
              /*is_homogeneous*/ false,
              /*min_arity*/ 0)
      .Attr("non_differentiable_outputs", "The indices of the module outputs that doesn't have a gradient.", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("full_shape_outputs", "The indices of the module outputs that must have full shape.", AttributeProto::INTS)
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Allow inputs and outputs to be any kind of tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto non_differentiable_outputs = ctx.getAttribute("non_differentiable_outputs");
        std::unordered_set<size_t> non_differentiable_outputs_indices{};
        if (nullptr != non_differentiable_outputs) {
          for (int i = 0, n = non_differentiable_outputs->ints_size(); i < n; ++i) {
            non_differentiable_outputs_indices.insert(static_cast<size_t>(non_differentiable_outputs->ints(i)));
          }
        }
        ORT_ENFORCE(ctx.getNumInputs() == ctx.getNumOutputs() + non_differentiable_outputs_indices.size());

        auto full_shape_outputs = ctx.getAttribute("full_shape_outputs");
        std::unordered_set<size_t> full_shape_outputs_indices{};
        if (nullptr == full_shape_outputs) {  // attribute not present
          fail_type_inference("Value of attribute 'full_shape_outputs' not specified");
        } else {
          for (int i = 0, n = full_shape_outputs->ints_size(); i < n; ++i) {
            full_shape_outputs_indices.insert(static_cast<size_t>(full_shape_outputs->ints(i)));
          }
        }

        for (size_t i = 0, j = 0; i < ctx.getNumInputs(); ++i) {
          // skip module outputs that are non differentiable
          if (non_differentiable_outputs_indices.count(i) > 0) {
            continue;
          }

          propagateElemTypeFromInputToOutput(ctx, i, j);
          if (full_shape_outputs_indices.count(i) > 0) {
            auto typeProto = ctx.getInputType(i);
            if (hasShape(*typeProto)) {
              propagateShapeFromInputToOutput(ctx, i, j);
            }
          }
          j++;
        }
      });

#ifdef ENABLE_TRAINING
  ONNX_CONTRIB_OPERATOR_SCHEMA(ATenOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("ATenOp")
      .Input(0, "inputs", "ATenOp inputs.", "T", OpSchema::Variadic,
             /*is_homogeneous*/ false,
             /*min_arity*/ 1)
      .Output(0, "outputs", "ATenOp outputs.", "T", OpSchema::Variadic,
              /*is_homogeneous*/ false,
              /*min_arity*/ 1)
      .Attr("name", "Name of ATen operator.", AttributeProto::STRING)
      .Attr("overload_name", "Overload name of ATen operator.", AttributeProto::STRING, false)
      .Attr("output_types", "Output types of ATenOp's gradient op.", AttributeProto::INTS, false)
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Allow inputs and outputs to be any kind of tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Load and set output types if there is attribute 'output_type'.
        // This is usually set in gradient nodes to help resolving graph.
        const auto output_types_proto = ctx.getAttribute("output_types");
        if (output_types_proto) {
          ORT_ENFORCE(static_cast<size_t>(output_types_proto->ints_size()) == ctx.getNumOutputs(),
                      "ATenOpGrad's output list and \"output_types\" attribute should have same length.");
          // Set inferred output types.
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            updateOutputElemType(ctx, i, static_cast<int>(output_types_proto->ints(static_cast<int>(i))));
          }
        }
      });
#endif

  ONNX_CONTRIB_OPERATOR_SCHEMA(PythonOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Wrapper of Pytorch's autograd.Function implementation.")
      .Input(
          0,
          "inputs",
          "Module outputs to be returned to pytorch.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Output(
          0,
          "context",
          "Address of context created in this operator. It can be used in backward.",
          "TInt64")
      .Output(
          1,
          "outputs",
          "Outputs returned from pytorch.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Attr(
          "name",
          "Name of custom class.",
          AttributeProto::STRING)
      .Attr(
          "input_convention",
          "input_convention[i]==c means a non-tensor argument. input_convention[i]==d means a tensor.",
          AttributeProto::STRING)
      .Attr(
          "input_requires_grads",
          "Flags to indicate whether the torch.autograd.apply's inputs require gradients (including flags for both tensor"
          " and non-tensor inputs)",
          AttributeProto::INTS)
      // Input Pytorch tensors.
      .Attr(
          "input_tensor_types",
          "Input types of autograd.Function.apply.",
          AttributeProto::INTS)
      .Attr(
          "input_tensor_ranks",
          "Input tensors' ranks of autograd.Function.apply.",
          AttributeProto::INTS)
      // Input int scalars.
      .Attr(
          "input_int_scalars",
          "Python int arguments.",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_int_scalar_positions",
          "",
          AttributeProto::INTS,
          false)
      // Input float scalars.
      .Attr(
          "input_float_scalars",
          "Python float arguments.",
          AttributeProto::FLOATS,
          false)
      .Attr(
          "input_float_scalar_positions",
          "",
          AttributeProto::INTS,
          false)
      // Input int tuple.
      .Attr(
          "input_int_tuples",
          "Python int-tuple arguments.",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_int_tuple_positions",
          "",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_int_tuple_begins",
          "",
          AttributeProto::INTS,
          false)
      // Input float tuple.
      .Attr(
          "input_float_tuples",
          "",
          AttributeProto::FLOATS,
          false)
      .Attr(
          "input_float_tuple_positions",
          "",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_float_tuple_begins",
          "",
          AttributeProto::INTS,
          false)
      // Output tensors.
      .Attr(
          "input_pointer_scalars",
          "",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_pointer_scalar_positions",
          "",
          AttributeProto::INTS,
          false)
      .Attr(
          "output_tensor_requires_grads",
          "Flags to indicate which output has gradient",
          AttributeProto::INTS)
      .Attr(
          "output_tensor_types",
          "Output types of autograd.Function.apply.",
          AttributeProto::INTS)
      .Attr(
          "output_tensor_ranks",
          "Output tensors' ranks of autograd.Function.apply.",
          AttributeProto::INTS)
      // Other attributes.
      .Attr(
          "inplace",
          "Indicate if the output should reuse input memory.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "training_mode",
          "Indicate if the model is exported in training_mode, by default, False.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Allow inputs and outputs to be any kind of tensor.")
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Load expected input types.
        const auto input_tensor_types_proto = ctx.getAttribute("input_tensor_types");
        // This is a required field.
        ORT_ENFORCE(input_tensor_types_proto, "PythonOp's must have \"input_tensor_types\" attribute.");
        // Check if the inferred input types match those described in the
        // "input_tensor_types" attributes.
        const int64_t input_tensor_types_count = input_tensor_types_proto->ints_size();
        ORT_ENFORCE(static_cast<size_t>(input_tensor_types_count) == ctx.getNumInputs(),
                    "PythonOp's input list and \"input_tensor_types\" attribute should have the same length.");
        for (auto i = 0; i < input_tensor_types_count; ++i) {
          const auto inferred_input_type = ctx.getInputType(i);
          ORT_ENFORCE(inferred_input_type, "PythonOp's ", i, "th input type is missing.");
          ORT_ENFORCE(inferred_input_type->value_case() == TypeProto::kTensorType,
                      "PythonOp's ", i, "th input type must be a tensor.");
          ORT_ENFORCE(inferred_input_type->tensor_type().elem_type() == input_tensor_types_proto->ints().at(i),
                      "PythonOp's ", i, "th input type must be ", input_tensor_types_proto->ints().at(i));
        }

        // The first output is a pointer which points to
        // a Python object created by torch.autograd.Function.apply.
        // For details, see how we interpret it (the 1st input of PythonOpGrad)
        // in PythonOpGrad's implementation.
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT64);
        updateOutputShape(ctx, 0, {});
        // Load expected output types.
        const auto output_tensor_types_proto = ctx.getAttribute("output_tensor_types");
        ORT_ENFORCE(static_cast<size_t>(output_tensor_types_proto->ints_size()) == ctx.getNumOutputs() - 1,
                    "PythonOp's output list has one more element than \"output_tensor_types\" attribute.");
        // This is a required field.
        ORT_ENFORCE(output_tensor_types_proto, "PythonOp's must have \"output_tensor_types\" attribute.");

        size_t rank_count = 0;
        // Set inferred output types.
        for (auto i = 1; i < static_cast<int64_t>(ctx.getNumOutputs()); ++i) {
          updateOutputElemType(ctx, i, static_cast<int32_t>(output_tensor_types_proto->ints().at(i - 1)));

          // Create symbolic shape.
          const auto output_tensor_ranks = ctx.getAttribute("output_tensor_ranks");
          ONNX_NAMESPACE::TensorShapeProto rank_only_shape;
          for (int64_t j = 0; j < output_tensor_ranks->ints().at(i - 1); ++j) {
            std::stringstream ss;
            ss << "PythonOp_unknown_rank_" << rank_count++;
            rank_only_shape.add_dim()->set_dim_param(ss.str());
          }

          // Assign symbolic shape.
          updateOutputShape(ctx, i, rank_only_shape);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(PythonOpGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Wrapper of Pytorch's autograd.Function's backward implementaiton.")
      .Input(
          0,
          "context",
          "Address of context created in this operator. It should be generated by the corresponding forward.",
          "TInt64")
      .Input(
          1,
          "inputs",
          "There are 2*N inputs: "
          "  N gradient inputs (as inputs of autograd.Function.backward) + "
          "  N forward run activations of autograd.Function.apply."
          "The N forward run inputs are used as control dependency between PythonOpGrad and PythonOp",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Output(
          0,
          "outputs",
          "Outputs returned from pytorch.",
          "T",
          OpSchema::Variadic,
          /*is_homogeneous*/ false,
          /*min_arity*/ 1)
      .Attr(
          "name",
          "Name of custom class.",
          AttributeProto::STRING)
      .Attr(
          "inplace",
          "Indicate if the output should reuse input memory. Todo(pengwa): do we really need it?",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "input_tensor_types",
          "Input types of autograd.Function.backward (including only tensor inputs)."
          "This attribute is mostly used for input checks for better robustnes.",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_tensor_ranks",
          "Input ranks of autograd.Function.backward (including only tensor inputs)."
          "This attribute is mostly used for input checks for better robustness.",
          AttributeProto::INTS,
          false)
      .Attr(
          "input_tensor_requires_grads",
          "Flags to indicate which inputs have gradients (including only tensor inputs)."
          "This attribute is mostly used for input checks for better robustness.",
          AttributeProto::INTS)
      .Attr(
          "output_tensor_types",
          "Output types of autograd.Function.backward outputs (including only tensor outputs).",
          AttributeProto::INTS,
          false)
      .Attr(
          "output_tensor_ranks",
          "Output ranks of autograd.Function.backward outputs (including only tensor outputs).",
          AttributeProto::INTS,
          false)
      .Attr(
          "output_tensor_requires_grads",
          "Flags to indicate which outputs have gradients (including only tensor outputs).",
          AttributeProto::INTS)
      .Attr(
          "output_convention",
          "A string inidicating autograd.Function.backward outputs's type."
          "value 'c' - non-tensor output; value 'd' - tensor output.",
          AttributeProto::STRING)
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Allow inputs and outputs to be any kind of tensor.")
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain input type to 64-bit integer.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Load expected input types.
        const auto input_tensor_types_proto = ctx.getAttribute("input_tensor_types");
        // This is a required field.
        ORT_ENFORCE(input_tensor_types_proto, "PythonOpGrad's must have \"input_tensor_types\" attribute.");
        // Check if the inferred input types match those described in the
        // "input_tensor_types" attributes.
        const auto input_tensor_requires_grads = ctx.getAttribute("input_tensor_requires_grads");
        // Expected input schema: [ctx, grad_input_1, ..., grad_input_N, unused_1, ..., unused_M]
        // "unused" inputs are just control inputs and they are not used actual computation.
        // Other variables are used to invoke autograd.Function.backward(ctx, grad_input1, ..., grad_input_N).
        // The "input_count" here means 1 + N.
        const auto input_count = input_tensor_requires_grads->ints().size();
        ORT_ENFORCE(input_tensor_types_proto->ints_size() == input_count - 1,
                    "PythonOp's input list should have one more element than \"input_tensor_types\" attribute.");
        // The first input is a pointer which points to
        // a Python object created by torch.autograd.Function.apply.
        // For details, see how we interpret it in PythonOpGrad implementation.
        for (auto i = 1; i < input_count; ++i) {
          const auto inferred_input_type = ctx.getInputType(i);
          ORT_ENFORCE(inferred_input_type, "PythonOpGrad's ", i, "th input type is missing.");
          ORT_ENFORCE(inferred_input_type->value_case() == TypeProto::kTensorType,
                      "PythonOpGrad's ", i, "th input type must be a tensor.");
          ORT_ENFORCE(inferred_input_type->tensor_type().elem_type() == input_tensor_types_proto->ints().at(i - 1),
                      "PythonOpGrad's ", i, "th input type must be ", input_tensor_types_proto->ints().at(i - 1));
        }

        // Load expected output types.
        const auto output_tensor_types_proto = ctx.getAttribute("output_tensor_types");
        ORT_ENFORCE(static_cast<size_t>(output_tensor_types_proto->ints_size()) == ctx.getNumOutputs(),
                    "PythonOpGrad's output list and \"output_tensor_types\" attribute should have the same length.");
        // This is a required field.
        ORT_ENFORCE(output_tensor_types_proto, "PythonOpGrad's must have \"output_tensor_types\" attribute.");
        // Set inferred output types.
        size_t rank_count = 0;
        for (auto i = 0; i < static_cast<int64_t>(ctx.getNumOutputs()); ++i) {
          updateOutputElemType(ctx, i, static_cast<int32_t>(output_tensor_types_proto->ints().at(i)));
          const auto output_tensor_ranks = ctx.getAttribute("output_tensor_ranks");
          ONNX_NAMESPACE::TensorShapeProto rank_only_shape;
          for (int64_t j = 0; j < output_tensor_ranks->ints().at(i); ++j) {
            std::stringstream ss;
            ss << "PythonOpGrad_unknown_rank_" << rank_count++;
            rank_only_shape.add_dim()->set_dim_param(ss.str());
          }

          // Assign symbolic shape.
          updateOutputShape(ctx, i, rank_only_shape);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyLossInternal)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction", reduction_doc, AttributeProto::STRING, std::string("mean"))
      .Input(0, "scores",
             "The predicted outputs with shape [batch_size, class_size], or "
             "[batch_size, class_size, D1, D2 , ..., Dk], where K is the number of dimensions.",
             "T")
      .Input(1, "labels",
             "The ground truth output tensor, with shape [batch_size], or "
             "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions. "
             "Labels element value shall be in range of [0, C). "
             "If ignore_index is specified, it may have a value outside [0, C) and the label values should either be "
             "in the range [0, C) or have the value ignore_index.",
             "Tind")
      .Input(2, "weights",
             "A manual rescaling weight given to each class. If given, it has to "
             "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
             "it is treated as if having all ones.",
             "T", OpSchema::Optional)
      .Input(3, "ignore_index",
             "Scalar tensor to specify a target value that is ignored and does not contribute to the input gradient.",
             "I", OpSchema::Optional)
      .Output(0, "output",
              "Weighted loss float Tensor. If reduction is 'none', this has the "
              "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
              "K-dimensional loss. Otherwise, it is a scalar.",
              "T")
      .Output(1, "log_prob", "Log probability tensor. If the output of softmax is prob, its value is log(prob).", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain target to integer types")
      .TypeConstraint("I", {"tensor(int64)"}, "Constrain ignore_index tensor to int64")
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
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      })
      .SetDoc(R"DOC(SoftmaxCrossEntropyLossInternal)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyLossInternalGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction", reduction_doc, AttributeProto::STRING, std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), (N+1)-D input of shape (batch_size).", "T")
      .Input(2, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(3, "weight", "weight for each sample. The shape is 1-D tensor.", "T", OpSchema::Optional)
      .Input(4, "ignore_index",
             "Scalar tensor to specify a target value that is ignored and does not contribute to the input gradient.",
             "I", OpSchema::Optional)
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types")
      .TypeConstraint("I", {"tensor(int64)"}, "Constrain ignore_index tensor to int64")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        propagateShapeFromInputToOutput(ctx, 1, 0);
      })
      .SetDoc(R"DOC(SoftmaxCrossEntropyLossInternalGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(NegativeLogLikelihoodLossInternal)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction", reduction_doc, AttributeProto::STRING, std::string("mean"))
      .Input(0, "input", "Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).", "T")
      .Input(1, "target",
             "Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value shall be in range of [0, C). "
             "If ignore_index is specified, it may have a value outside [0, C) and the target values should either be "
             "in the range [0, C) or have the value ignore_index.",
             "Tind")
      .Input(2, "weight",
             "Optional rescaling weight tensor. "
             "If given, it has to be a tensor of size C. Otherwise, it is treated as if having all ones.",
             "T", OpSchema::Optional)
      .Input(3, "ignore_index",
             "Scalar tensor to specify a target value that is ignored and does not contribute to the input gradient.",
             "I", OpSchema::Optional)
      .Output(0, "loss", "The negative log likelihood loss", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                      "Constrain input and output types to float tensors.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain target to integer types")
      .TypeConstraint("I", {"tensor(int64)"}, "Constrain ignore_index tensor to int64")
      .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyNllLossInternal)
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateElemTypeFromInputToOutput(ctx, 0, 0); })
      .SetDoc(R"DOC(NegativeLogLikelihoodLossInternal)DOC");
}

}  // namespace training

void RegisterOrtOpSchemas() {
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSExperimentalDomain, 1, 1);
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSFeaturizersDomain, 1, 1);

  onnxruntime::contrib::RegisterContribSchemas();
  onnxruntime::training::RegisterTrainingOpSchemas();
}

}  // namespace onnxruntime
