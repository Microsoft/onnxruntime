// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include "gradient_builder_base.h"

namespace onnxruntime {
namespace training {

using json = nlohmann::json;

std::string ToString(const std::vector<Dimension>& dims) {
  std::stringstream output;
  output << "[";
  if (!dims.empty()) {
    for (auto& dim : dims) {
      if (dim.has_dim_value()) {
        output << dim.dim_value() << ",";
      }
      if (dim.has_dim_param()) {
        output << dim.dim_param() << ",";
      }
    }
    output.seekp(-1, output.cur);
  }
  output << "]";

  return output.str();
}

void ComputeBroadcastBackwardAxes(
    const std::vector<Dimension>& A_dims,
    const std::vector<Dimension>& B_dims,
    std::vector<int64_t>* A_axes,
    std::vector<int64_t>* B_axes,
    const std::string& node_name) {
  if (A_axes) A_axes->clear();
  if (B_axes) B_axes->clear();

  int ndim = int(std::max(A_dims.size(), B_dims.size()));
  int i = int(A_dims.size() - 1);
  int j = int(B_dims.size() - 1);
  int k = ndim - 1;

  for (; i >= 0 && j >= 0; --k) {
    if (A_dims[i].has_dim_value() && B_dims[j].has_dim_value()) {
      auto A_dim = A_dims[i].dim_value(),
           B_dim = B_dims[j].dim_value();

      if (A_dim != B_dim) {
        if (A_axes && A_dim == 1) {
          A_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
        if (B_axes && B_dim == 1) {
          B_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
      }
    } else if (A_dims[i].has_dim_param() && B_dims[j].has_dim_param()) {
      auto A_dim = A_dims[i].dim_param(),
           B_dim = B_dims[j].dim_param();
      if (A_dim != B_dim) {
        LOGS_DEFAULT(INFO) << "Gradient building for node " << node_name << ": symbolic dimension expects to match. " <<
                  "A_dims:" << ToString(A_dims) << ", B_dims:" << ToString(B_dims) <<   
                  " This is a relaxing case, and the kernel might run into problem later if A_dims and B_dims turns out not broadcastable.";
      }
    } else if (A_dims[i].has_dim_param() && B_dims[j].has_dim_value()) {
      auto A_dim = A_dims[i].dim_param();
      auto B_dim = B_dims[j].dim_value();

      if (B_dim != 1) {
        LOGS_DEFAULT(INFO) << "Gradient building for node " << node_name << ": symbolic broadcasting expects the B_dimension to be 1. " <<
                  "A_dims:" << ToString(A_dims) << ", B_dims:" << ToString(B_dims) <<   
                  " This is a relaxing case, and the kernel might run into problem later if A_dims and B_dims turns out not broadcastable.";   
      } else {
        if (B_axes) {
          B_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
      }
    } else if (A_dims[i].has_dim_value() && B_dims[j].has_dim_param()) {
      auto A_dim = A_dims[i].dim_value();
      auto B_dim = B_dims[j].dim_param();

      if (A_dim != 1) {
        LOGS_DEFAULT(INFO) << "Gradient building for node " << node_name << ": symbolic broadcasting expects the A_dimension to be 1. " <<
                  "A_dims:" << ToString(A_dims) << ", B_dims:" << ToString(B_dims) <<   
                  " This is a relaxing case, and the kernel might run into problem later if A_dims and B_dims turns out not broadcastable.";
      } else {
        if (A_axes) {
          A_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
      }
    }

    --i;
    --j;
  }

  if (i < 0) {
    if (A_axes) {
      for (; k >= 0; --k) {
        A_axes->push_back(gsl::narrow_cast<int64_t>(k));
      }
    }
  } else {
    if (B_axes) {
      for (; k >= 0; --k) {
        B_axes->push_back(gsl::narrow_cast<int64_t>(k));
      }
    }
  }
}

Status GetShape(const ArgDef& arg_def, std::vector<Dimension>& shape) {
  shape.clear();
  ORT_RETURN_IF_NOT(arg_def.type_proto && arg_def.type_proto->has_tensor_type() && arg_def.type_proto->tensor_type().has_shape(),
                    "During GetShape, ", arg_def.name, "'s shape is null.");
  const auto& dims = arg_def.type_proto->tensor_type().shape().dim();
  for (auto dim = dims.begin(); dim < dims.end(); dim++) {
    ORT_RETURN_IF_NOT(dim->dim_value() > 0 || dim->has_dim_param(),
                      "During GetShape, ", arg_def.name, "'s dim value is invalid ", dim->dim_value());
    shape.push_back(*dim);
  }
  return Status::OK();
}

void ComputeBroadcastBackwardAxesDynamic(const ArgDef& a,
                                         const ArgDef& b,
                                         const ArgDef& a_shape,
                                         const ArgDef& b_shape,
                                         const ArgDef* a_axes,
                                         const ArgDef* b_axes,
                                         std::vector<NodeDef>& output) {
  // Populate the node names explicitly in case a and b are the same tensor and
  // resulting in duplicated node name for Shape node. For example, y = x^2 is sometimes represented as Mul(x,x)
  output.push_back(
      NodeDef("Shape", {a}, {a_shape}, NodeAttributes(), a_shape.name + "_lhs"));

  output.push_back(
      NodeDef("Shape", {b}, {b_shape}, NodeAttributes(), b_shape.name + "_rhs"));


  ArgDef a_op = ArgDef(""), b_op = ArgDef("");
  if (a_axes)
    a_op = *a_axes;
  if (b_axes)
    b_op = *b_axes;
  output.push_back(
      NodeDef(OpDef{"BroadcastGradientArgs", kMSDomain, 1},
              {a_shape, b_shape},
              {a_op, b_op}));
}

void GradientBuilderBase::AddReduceSumNode(const ArgDef& input_arg_def,
                                           const ArgDef& output_arg_def,
                                           const std::vector<int64_t>& reduce_axes,
                                           bool keep_dims,
                                           std::vector<NodeDef>& output) const {
  // Use the graph ONNX OpSet version instead of the node since version for non-onnx-domain Ops.
  int opset_version = SrcNodeDomain() == kOnnxDomain ? SrcNodeOpsetVersion() : OnnxOpSetVersion();
  if (opset_version < 13) {
    output.emplace_back(
        NodeDef("ReduceSum",
                {input_arg_def},
                {output_arg_def},
                {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", static_cast<int64_t>(keep_dims))},
                {"axes", ONNX_NAMESPACE::MakeAttribute("axes", reduce_axes)}}));
    return;
  }

  ArgDef reduce_axes_arg_def = IA("ReduceAxes_for_" + output_arg_def.name);
  output.emplace_back(ConstantVectorNode(reduce_axes, reduce_axes_arg_def.name));
  output.emplace_back(
      NodeDef(OpDef{"ReduceSum", kOnnxDomain, opset_version},
              {input_arg_def, reduce_axes_arg_def},
              {output_arg_def},
              {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", static_cast<int64_t>(keep_dims))}}));
}

void GradientBuilderBase::HandleBroadcasting(const ArgDef& input_grad,
                                             const ArgDef& target,
                                             const ArgDef& output_grad,
                                             const std::vector<int64_t>& reduce_axes,
                                             std::vector<NodeDef>& output) const {
  std::unordered_set<size_t> reduce_axes_set(reduce_axes.begin(), reduce_axes.end());
  std::vector<Dimension> reduced_shape, input_grad_shape, target_shape;
  ORT_ENFORCE(GetShape(input_grad, input_grad_shape).IsOK());
  ORT_ENFORCE(GetShape(target, target_shape).IsOK());

  bool keep_dims = (input_grad_shape.size() == target_shape.size());

  Dimension dim_one;
  dim_one.set_dim_value(int64_t(1));

  for (size_t i = 0; i < input_grad_shape.size(); ++i) {
    if (reduce_axes_set.count(i) > 0) {
      if (keep_dims) {
        reduced_shape.push_back(dim_one);
      }
    } else {
      reduced_shape.push_back(input_grad_shape[i]);
    }
  }

  bool skip_reshape = true;
  if (target_shape.size() != reduced_shape.size()) {
    skip_reshape = false;
  } else {
    for (size_t i = 0; i < target_shape.size(); ++i) {
      if (target_shape[i].dim_value() != reduced_shape[i].dim_value() ||
          target_shape[i].dim_param() != reduced_shape[i].dim_param()) {
        skip_reshape = false;
        break;
      }
    }
  }

  if (skip_reshape) {
    AddReduceSumNode(input_grad, output_grad, reduce_axes, keep_dims, output);
  } else {
    ArgDef reduce_grad_arg = IA("ReduceSum_" + input_grad.name + "_for_" + target.name);
    AddReduceSumNode(input_grad, reduce_grad_arg, reduce_axes, true, output);
    ArgDef target_shape_arg = IA(target.name + "_shape");
    output.push_back(
        NodeDef("Shape",
                {target},
                {target_shape_arg}));

    output.push_back(
        NodeDef("Reshape",
                {reduce_grad_arg, target_shape_arg},
                {output_grad}));
  }
}

void GradientBuilderBase::HandleBroadcastingDynamic(const ArgDef& input_grad,
                                                    const ArgDef& target,
                                                    const ArgDef& target_shape,
                                                    const ArgDef& output_grad,
                                                    const ArgDef& reduce_axes,
                                                    std::vector<NodeDef>& output) const {
  ArgDef reduce_grad_arg = IA("ReduceSum_" + input_grad.name + "_for_" + target.name);
  int opset_version = SrcNodeDomain() == kOnnxDomain ? SrcNodeOpsetVersion() : OnnxOpSetVersion();
  output.push_back(
      NodeDef(opset_version >= 13 ? OpDef{"ReduceSum", kOnnxDomain, opset_version} : OpDef{"ReduceSumTraining", kMSDomain, 1},
              {input_grad,
               reduce_axes},
              {reduce_grad_arg},
              {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(1))},
               {"noop_with_empty_axes", ONNX_NAMESPACE::MakeAttribute("noop_with_empty_axes", int64_t(1))}}));

  output.push_back(
      NodeDef("Reshape",
              {reduce_grad_arg, target_shape},
              {output_grad}));
}

std::vector<NodeDef> GradientBuilderBase::GetBiasGeluGradNodes(
    bool use_approximation,
    const ArgDef& dY, const ArgDef& X, const ArgDef& B,  // inputs
    const ArgDef& dX, const ArgDef& dB,                  // outputs
    const ArgDef& b_axes, const ArgDef& b_shape, const ArgDef& x_shape,  //intermediate args
    const std::string& node_name) const {
  std::vector<Dimension> B_shape, X_shape;
  std::vector<NodeDef> result;
  if (GetShape(B, B_shape).IsOK() && GetShape(X, X_shape).IsOK()) {
    ORT_ENFORCE(B_shape.size() == 1, "B must have exactly one dimension.");

    const std::vector<int64_t> B_axes = [&B_shape, &X_shape, &node_name]() {
      std::vector<int64_t> result{};
      ComputeBroadcastBackwardAxes(B_shape, X_shape, &result, nullptr, node_name);
      return result;
    }();

    result.push_back(
        NodeDef(OpDef{use_approximation ? "BiasFastGeluGrad_dX" : "BiasGeluGrad_dX", kMSDomain, 1},
                {dY, X, B},
                {dX}));
    AddReduceSumNode(dX, dB, B_axes, false, result);
  } else {
    ComputeBroadcastBackwardAxesDynamic(B, X, b_shape, x_shape, &b_axes, nullptr, result);
    result.push_back(
        NodeDef(OpDef{use_approximation ? "BiasFastGeluGrad_dX" : "BiasGeluGrad_dX", kMSDomain, 1},
                {dY, X, B},
                {dX}));
    int opset_version = SrcNodeDomain() == kOnnxDomain ? SrcNodeOpsetVersion() : OnnxOpSetVersion();
    result.push_back(
        NodeDef(opset_version >= 13 ? OpDef{"ReduceSum", kOnnxDomain, opset_version} : OpDef{"ReduceSumTraining", kMSDomain, 1},
                {dX, b_axes},
                {dB},
                {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t{0})}}));
  }

  return result;
}

// We will not implement all kinds of possibilities here. Instead, we only support those cases that are possible for
// gradient builder. For example, for attribute value that is tensor type (such as 'value' attribute for Constant),
// the only case we need is scalar tensor. For list type, from the ONNX Op spec, except little uncommon Ops, all list
// type attributes are of INT type. We will add support to more cases if needed.
AttributeProto GradientBuilderBase::AttributeDefinitionToAttributeProto(const std::string& name,
                                                                        const AttributeDefinition& attr_def) const {
  AttributeProto attr_proto;
  const auto& value = json::parse(attr_def.value_json);
  int elem_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (attr_def.dtype.find("IElemType(") == 0) {
    int index = std::stoi(attr_def.dtype.substr(10, attr_def.dtype.length() - 11));
    elem_type = IElemType(static_cast<size_t>(index));
  } else if (attr_def.dtype.find("OElemType(") == 0) {
    int index = std::stoi(attr_def.dtype.substr(10, attr_def.dtype.length() - 11));
    elem_type = OElemType(static_cast<size_t>(index));
  } else if (attr_def.dtype == "int") {
    elem_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  } else if (attr_def.dtype == "float") {
    elem_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  } else if (attr_def.dtype == "bool") {
    elem_type = ONNX_NAMESPACE::TensorProto_DataType_BOOL;
  } else if (attr_def.dtype == "string") {
    elem_type = ONNX_NAMESPACE::TensorProto_DataType_STRING;
  } else {
    ORT_ENFORCE(false, "Type ", attr_def.dtype, " is not supported.");
  }

  if (value.is_array()) {
    ORT_ENFORCE(!attr_def.is_tensor, "1D tensor attribute is not supported for now.");
    // Support only INTs for now.
    switch (elem_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        std::vector<int64_t> int_array;
        for (const auto& elem : value) {
          ORT_ENFORCE(elem.is_number_integer());
          int_array.emplace_back(static_cast<int64_t>(elem.get<int>()));
        }
        attr_proto = ONNX_NAMESPACE::MakeAttribute(name, int_array);
      } break;
      default:
        ORT_ENFORCE(false, "List attribute of type ", std::to_string(elem_type), " is not supported for now.");
    }
  } else if (attr_def.is_tensor) {
    TensorProto tensor_proto;
    if (value.is_number_integer()) {
      ORT_ENFORCE(elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64);
      tensor_proto = ScalarTensorProto(static_cast<int64_t>(value.get<int>()), {1});
    } else if (value.is_number_float()) {
      ORT_ENFORCE(elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                  elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
                  elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
      float float_value = value.get<float>();
      if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        tensor_proto = ScalarTensorProto(MLFloat16(math::floatToHalf(float_value)), {1});
      } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
        tensor_proto = ScalarTensorProto(BFloat16(float_value), {1});
      } else {
        tensor_proto = ScalarTensorProto(float_value, {1});
      }
    } else if (value.is_boolean()) {
      ORT_ENFORCE(elem_type == ONNX_NAMESPACE::TensorProto_DataType_BOOL);
      tensor_proto = ScalarTensorProto(value.get<bool>(), {1});
    } else {
      ORT_ENFORCE(false, "Tensor attribute of type ", std::to_string(elem_type), " is not supported for now.");
    }

    attr_proto = ONNX_NAMESPACE::MakeAttribute(name, tensor_proto);
  } else {
    switch (elem_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        ORT_ENFORCE(value.is_number_integer());
        attr_proto = ONNX_NAMESPACE::MakeAttribute(name, static_cast<int64_t>(value.get<int>()));
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        ORT_ENFORCE(value.is_number_float());
        attr_proto = ONNX_NAMESPACE::MakeAttribute(name, value.get<float>());
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
        ORT_ENFORCE(value.is_string());
        attr_proto = ONNX_NAMESPACE::MakeAttribute(name, value.get<std::string>());
      } break;
      default:
        ORT_ENFORCE(false, "Attribute of type ", std::to_string(elem_type), " is not supported for now.");
    }
  }

  return attr_proto;
}

}  // namespace training
}  // namespace onnxruntime
