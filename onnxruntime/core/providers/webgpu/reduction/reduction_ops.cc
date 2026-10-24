// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/reduction/reduction_ops.h"
#include <memory>
#include <sstream>
#include "core/framework/data_transfer_manager.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"

namespace onnxruntime {
namespace webgpu {

#define REGISTER_REDUCE_VERSIONED_KERNEL(ReduceOp, begin, end)                         \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                   \
      ReduceOp,                                                                        \
      kOnnxDomain,                                                                     \
      begin, end,                                                                      \
      kWebGpuExecutionProvider,                                                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      ReduceOp);

#define REGISTER_REDUCE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT(ReduceOp, begin, end)                                             \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                          \
      ReduceOp,                                                                                                               \
      kOnnxDomain,                                                                                                            \
      begin, end,                                                                                                             \
      kWebGpuExecutionProvider,                                                                                               \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      ReduceOp);

#define REGISTER_REDUCE_KERNEL(ReduceOp, version)                                                                             \
  ONNX_OPERATOR_KERNEL_EX(                                                                                                    \
      ReduceOp,                                                                                                               \
      kOnnxDomain,                                                                                                            \
      version,                                                                                                                \
      kWebGpuExecutionProvider,                                                                                               \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      ReduceOp);

#define REGISTER_REDUCE_FLOAT_KERNEL(ReduceOp, version)                                                                      \
  ONNX_OPERATOR_KERNEL_EX(                                                                                                   \
      ReduceOp,                                                                                                              \
      kOnnxDomain,                                                                                                           \
      version,                                                                                                               \
      kWebGpuExecutionProvider,                                                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      ReduceOp);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMean, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMean, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMean, 13, 17);
REGISTER_REDUCE_KERNEL(ReduceMean, 18);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMax, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMax, 11, 11);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMax, 12, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMax, 13, 17);
REGISTER_REDUCE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT(ReduceMax, 18, 19);
REGISTER_REDUCE_KERNEL(ReduceMax, 20);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMin, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMin, 11, 11);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMin, 12, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceMin, 13, 17);
REGISTER_REDUCE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT(ReduceMin, 18, 19);
REGISTER_REDUCE_KERNEL(ReduceMin, 20);

// ReduceSum: versions 1-12 use axes as attribute; version 13+ uses axes as a CPU input tensor.
// Factory functions allow conditional int64 support on the T type constraint.
// NOTE: int64 reduction in the WebGPU shader uses i32 (low 32 bits only); values outside
// the int32 range will produce incorrect results — same limitation as Range.
KernelCreateInfo CreateReduceSumVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<ReduceSum>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("ReduceSum")
              .SetDomain(kOnnxDomain)
              .SinceVersion(start_version, end_version)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .Build(),
          kernel_create_fn};
}

KernelCreateInfo CreateReduceSumKernelInfo(int since_version, bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<ReduceSum>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("ReduceSum")
              .SetDomain(kOnnxDomain)
              .SinceVersion(since_version)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .InputMemoryType(OrtMemTypeCPUInput, 1)
              .Build(),
          kernel_create_fn};
}

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceProd, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceProd, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceProd, 13, 17);
REGISTER_REDUCE_KERNEL(ReduceProd, 18);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL1, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL1, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL1, 13, 17);
REGISTER_REDUCE_KERNEL(ReduceL1, 18);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL2, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL2, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceL2, 13, 17);
REGISTER_REDUCE_KERNEL(ReduceL2, 18);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSum, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSum, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSum, 13, 17);
REGISTER_REDUCE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT(ReduceLogSum, 18, 27);
REGISTER_REDUCE_FLOAT_KERNEL(ReduceLogSum, 28);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceSumSquare, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceSumSquare, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceSumSquare, 13, 17);
REGISTER_REDUCE_KERNEL(ReduceSumSquare, 18);

REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSumExp, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSumExp, 11, 12);
REGISTER_REDUCE_VERSIONED_KERNEL(ReduceLogSumExp, 13, 17);
REGISTER_REDUCE_VERSIONED_KERNEL_WITH_AXIS_IN_INPUT(ReduceLogSumExp, 18, 27);
REGISTER_REDUCE_FLOAT_KERNEL(ReduceLogSumExp, 28);

REGISTER_REDUCE_VERSIONED_KERNEL(ArgMax, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ArgMax, 11, 12);
REGISTER_REDUCE_KERNEL(ArgMax, 13);

REGISTER_REDUCE_VERSIONED_KERNEL(ArgMin, 1, 10);
REGISTER_REDUCE_VERSIONED_KERNEL(ArgMin, 11, 12);
REGISTER_REDUCE_KERNEL(ArgMin, 13);

std::unordered_map<std::string, ReduceOpType> reduce_op_types = {
    {"ReduceMax", ReduceOpType::Max},
    {"ReduceMin", ReduceOpType::Min},
    {"ReduceMean", ReduceOpType::Mean},
    {"ReduceSum", ReduceOpType::Sum},
    {"ReduceProd", ReduceOpType::Prod},
    {"ReduceSumSquare", ReduceOpType::SumSquare},
    {"ReduceLogSumExp", ReduceOpType::LogSumExp},
    {"ReduceL1", ReduceOpType::L1},
    {"ReduceL2", ReduceOpType::L2},
    {"ReduceLogSum", ReduceOpType::LogSum},
    {"ArgMax", ReduceOpType::ArgMax},
    {"ArgMin", ReduceOpType::ArgMin},
    {"ArgMax_select_last_index", ReduceOpType::ArgMax_select_last_index},
    {"ArgMin_select_last_index", ReduceOpType::ArgMin_select_last_index},
};

std::unordered_map<ReduceOpType, std::string> reduce_op_code_map = {
    {ReduceOpType::Max, "select(bestValue, candidate, candidate > bestValue)"},
    {ReduceOpType::Min, "select(bestValue, candidate, candidate < bestValue)"},
    {ReduceOpType::Mean, "bestValue + candidate"},
    {ReduceOpType::Sum, "bestValue + candidate"},
    {ReduceOpType::Prod, "bestValue * candidate"},
    {ReduceOpType::SumSquare, "bestValue + candidate * candidate"},
    {ReduceOpType::LogSumExp, "bestValue + output_value_t(exp(f32(candidate)))"},
    {ReduceOpType::L1, "bestValue + abs(candidate)"},
    {ReduceOpType::L2, "bestValue + candidate * candidate"},
    {ReduceOpType::LogSum, "bestValue + candidate"},
};

std::unordered_map<ReduceOpType, std::string> reduce_op_shared_code_map = {
    {ReduceOpType::Max, "select(bestValue, candidate, candidate > bestValue)"},
    {ReduceOpType::Min, "select(bestValue, candidate, candidate < bestValue)"},
    {ReduceOpType::Mean, "bestValue + candidate"},
    {ReduceOpType::Sum, "bestValue + candidate"},
    {ReduceOpType::Prod, "bestValue * candidate"},
    {ReduceOpType::SumSquare, "bestValue + candidate"},
    {ReduceOpType::LogSumExp, "bestValue + candidate"},
    {ReduceOpType::L1, "bestValue + candidate"},
    {ReduceOpType::L2, "bestValue + candidate"},
    {ReduceOpType::LogSum, "bestValue + candidate"},
};

std::unordered_map<ReduceOpType, std::string> reduce_op_init_values_map = {
    {ReduceOpType::Max, "_A[offset]"},
    {ReduceOpType::Min, "_A[offset]"},
    {ReduceOpType::Mean, "0"},
    {ReduceOpType::Sum, "0"},
    {ReduceOpType::Prod, "1"},
    {ReduceOpType::SumSquare, "0"},
    {ReduceOpType::LogSumExp, "0"},
    {ReduceOpType::L1, "0"},
    {ReduceOpType::L2, "0"},
    {ReduceOpType::LogSum, "0"},
};

std::unordered_map<ReduceOpType, std::string> reduce_op_output_values_map = {
    {ReduceOpType::Max, "bestValue"},
    {ReduceOpType::Min, "bestValue"},
    {ReduceOpType::Mean, "bestValue"},
    {ReduceOpType::Sum, "bestValue"},
    {ReduceOpType::Prod, "bestValue"},
    {ReduceOpType::SumSquare, "bestValue"},
    {ReduceOpType::LogSumExp, "log(f32(bestValue))"},
    {ReduceOpType::L1, "bestValue"},
    {ReduceOpType::L2, "sqrt(f32(bestValue))"},
    {ReduceOpType::LogSum, "log(f32(bestValue))"},
};

std::unordered_map<ReduceOpType, ReduceOpSpecificCode> reduce_op_naive_code_map = {
    {ReduceOpType::Max, {"var max_element = first_element;", "max_element = max(max_element, current_element);", "let output_value = output_value_t(max_element);"}},
    {ReduceOpType::Min, {"var min_element = first_element;", "min_element = min(min_element, current_element);", "let output_value = output_value_t(min_element);"}},
    {ReduceOpType::Mean, {"var sum = f32(0);", "sum += f32(current_element);", "let output_value = output_value_t(sum / f32(uniforms.reduce_size));"}},
    {ReduceOpType::Sum, {"var sum = f32(0);", "sum += f32(current_element);", "let output_value = output_value_t(sum);"}},
    {ReduceOpType::Prod, {"var prod = f32(1);", "prod *= f32(current_element);", "let output_value = output_value_t(prod);"}},
    {ReduceOpType::SumSquare, {"var sum_square = f32(0);", "sum_square += f32(current_element * current_element);", "let output_value = output_value_t(sum_square);"}},
    {ReduceOpType::LogSumExp, {"var log_sum_exp = f32(0);", "log_sum_exp += exp(f32(current_element));", "let output_value = output_value_t(log(log_sum_exp));"}},
    {ReduceOpType::L1, {"var l1 = f32(0);", "l1 += abs(f32(current_element));", "let output_value = output_value_t(l1);"}},
    {ReduceOpType::L2, {"var l2 = f32(0);", "l2 += f32(current_element * current_element);", "let output_value = output_value_t(sqrt(l2));"}},
    {ReduceOpType::LogSum, {"var sum = f32(0);", "sum += f32(current_element);", "let output_value = output_value_t(log(sum));"}},
    {ReduceOpType::ArgMax, {"var best_element = first_element; var best_index = u32(0);", "if (current_element > best_element) { best_element = current_element; best_index = last_index; };", "let output_value = output_value_t(best_index);"}},
    {ReduceOpType::ArgMin, {"var best_element = first_element;; var best_index = u32(0);", "if (current_element < best_element) { best_element = current_element; best_index = last_index; };", "let output_value = output_value_t(best_index);"}},
    {ReduceOpType::ArgMax_select_last_index, {"var best_element = first_element; var best_index = u32(0);", "if (current_element >= best_element) { best_element = current_element; best_index = last_index; };", "let output_value = output_value_t(best_index);"}},
    {ReduceOpType::ArgMin_select_last_index, {"var best_element = first_element;; var best_index = u32(0);", "if (current_element <= best_element) { best_element = current_element; best_index = last_index; };", "let output_value = output_value_t(best_index);"}},
};

ReduceOpType StringToReduceOp(std::string name) {
  ORT_ENFORCE(reduce_op_types.contains(name), "Unsupported reduction op type: ", name);
  return reduce_op_types[name];
}

Status ReduceNaiveProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& code = reduce_op_naive_code_map.at(reduce_op_type_);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  if (is_input_empty_) {
    shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                              << code.loop_header_
                              << code.loop_footer_
                              << output.SetByOffset("global_idx", "output_value");
    return Status::OK();
  }
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  bool reduce_on_all_axes = no_op_with_empty_axes_ == false && axes_.empty();
  std::string loop_header = code.loop_header_.find("first_element") == std::string::npos ? code.loop_header_ : "let first_element = " + input.GetByIndices("input_indices") + ";\n" + code.loop_header_ + "\n";
  std::string loop_body = "let current_element: input_value_t = " + input.GetByIndices("input_indices") + ";\n" + code.loop_body_;
  std::string loop_footer = code.loop_footer_;
  const auto input_rank = input.Rank();
  for (int i = 0, l = 0; i < input_rank; ++i) {
    if (reduce_on_all_axes || std::find(axes_.begin(), axes_.end(), i) != axes_.end()) {
      if (keepdims_) {
        l++;
      }
      std::stringstream ss;
      std::string index = "i" + std::to_string(i);
      ss << "for (var " << index << " : u32 = 0; " << index << " < " << input.IndicesGet("uniforms.input_shape", i) << "; " << index << "++) {\n";
      if (loop_body.find("last_index") != std::string::npos) {
        ss << "let last_index = " + index + ";\n";
      }
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      ss << loop_body << "\n";
      ss << "}\n";
      loop_body = ss.str();
    } else {
      std::stringstream ss;
      std::string index = "i" + std::to_string(i);
      ss << "let " << index << " = " << output.IndicesGet("output_indices", l) << ";\n";
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      ss << loop_header << "\n";
      loop_header = ss.str();
      l++;
    }
  }
  std::stringstream input_indices_init_value;
  for (int i = 0; i < input_rank - 1; ++i) {
    input_indices_init_value << "0, ";
  }
  input_indices_init_value << "0";
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices: output_indices_t = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t = input_indices_t(" << input_indices_init_value.str() << ");\n"
                            << loop_header << loop_body << loop_footer;
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "output_value");
  return Status::OK();
}

Status ReduceSharedProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("_A", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation()
      << "var<workgroup> aBestValues : array<output_value_t, " << workgroup_size_ << ">;\n\n"
      << "fn DIV_CEIL(a : u32, b : u32) -> u32 {\n"
      << "  return ((a - 1u) / b + 1u);\n"
      << "}\n";
  shader.MainFunctionBody() << "let outputIndex = global_idx / " << workgroup_size_ << ";\n"
                            << "let offset = outputIndex * uniforms.reduceSize;\n"
                            << "var bestValue = output_value_t(" << reduce_op_init_values_map[reduce_op_type_] << ");\n"
                            << "let length = uniforms.reduceSize;\n"
                            << "for (var k = local_idx; k < length; k += " << workgroup_size_ << ") {\n"
                            << "  let candidate = output_value_t(" << input.GetByOffset("offset + k") << ");\n"
                            << "  bestValue = " << reduce_op_code_map[reduce_op_type_] << ";\n"
                            << "}\n"
                            << "aBestValues[local_idx] = bestValue;\n"
                            << "workgroupBarrier();\n"
                            << "var reduceSize = min(length, " << workgroup_size_ << ");\n"
                            << "for (var currentSize = reduceSize / 2; reduceSize > 1; currentSize = reduceSize / 2) {\n"
                            << "  let interval = DIV_CEIL(reduceSize, 2u);\n"
                            << "  if (local_idx < currentSize) {\n"
                            << "    let candidate = aBestValues[local_idx + interval];\n"
                            << "    bestValue = " << reduce_op_shared_code_map[reduce_op_type_] << ";\n"
                            << "    aBestValues[local_idx] = bestValue;\n"
                            << "  }\n"
                            << "  reduceSize = interval;\n"
                            << "  workgroupBarrier();\n"
                            << "}\n"
                            << "if (local_idx == 0) {\n"
                            << "  let outputValue = output_value_t(" << (reduce_op_type_ == ReduceOpType::Mean ? "(bestValue / output_element_t(uniforms.reduceSize))" : reduce_op_output_values_map[reduce_op_type_]) << ");\n"
                            << "  " << output.SetByOffset("outputIndex", "outputValue") << ";\n"
                            << "}\n";
  return Status::OK();
}

template <bool allow_multi_axes>
Status ReduceKernel<allow_multi_axes>::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  ORT_RETURN_IF_ERROR(CheckInput(input_tensor));
  InlinedVector<uint32_t> input_axes;
  bool add_suffix = name_ == "ArgMax" || name_ == "ArgMin";
  ReduceOpType reduce_op_type = StringToReduceOp(name_ + std::string((select_last_index_ != 0 && add_suffix) ? "_select_last_index" : ""));
  auto rank = input_tensor->Shape().NumDimensions();
  auto transform_axis = [rank](int64_t axis) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
      ORT_THROW("Axes values must be in the range [-rank, rank-1]. Got: ", axis);
    }
    return static_cast<uint32_t>(axis);
  };
  // Check if axes input is provided and copy the axes values to input_axes
  if (context.InputCount() > 1) {
    ORT_ENFORCE(axes_.empty(), "Axes attribute may not be specified when axes input is also provided.");
    const Tensor* axes_tensor = context.Input<Tensor>(1);
    if (nullptr != axes_tensor) {
      auto size = static_cast<size_t>(axes_tensor->Shape()[0]);
      const auto* data = axes_tensor->Data<int64_t>();
      input_axes.reserve(size);
      std::transform(data, data + size, std::back_inserter(input_axes), transform_axis);
    }
  } else {
    input_axes.reserve(axes_.size());
    std::transform(axes_.begin(), axes_.end(), std::back_inserter(input_axes), transform_axis);
  }
  if (input_axes.empty()) {
    if (noop_with_empty_axes_ || rank == 0) {
      // If axes is empty and noop_with_empty_axes_ is true, it is a no-op according to the spec
      // If input tensor is a scalar and it's not a ReduceLogSum or ReduceSumSquare, return the input tensor as is.
      if (rank == 0 && (name_ == "ReduceLogSum" || name_ == "ReduceSumSquare" || name_ == "ReduceL1" || name_ == "ReduceL2")) {
        // For ReduceLogSum with scalar input, output = log(input)
        // For ReduceSumSquare with scalar input, output = input * input
        auto output = context.Output(0, input_tensor->Shape());
        // We need to run the operation even for scalar inputs for these ops
        constexpr uint32_t output_size = 1;
        constexpr uint32_t reduce_size = 1;
        ReduceNaiveProgram program(name_, reduce_op_type, keepdims_, noop_with_empty_axes_, input_axes, false);
        program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank})
            .SetDispatchGroupSize(1)
            .AddUniformVariables({{output_size}, {static_cast<uint32_t>(noop_with_empty_axes_ ? 1 : 0)}, {reduce_size}});
        return context.RunProgram(program);
      } else {
        // For other ops, or when axes is empty with noop_with_empty_axes_ true, just copy the input
        auto output = context.Output(0, input_tensor->Shape());
        if (output->DataRaw() != input_tensor->DataRaw()) {
          ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input_tensor, *output));
        }
        return Status::OK();
      }
    } else {
      // If axes is empty and noop_with_empty_axes_ is false, it is a reduction over all axes
      input_axes.resize(rank);
      std::iota(input_axes.begin(), input_axes.end(), 0);
    }
  }
  // reduce_axes element is either 1 or 0 depending on whether the axis is reduced or not
  std::vector<uint32_t> reduce_axes;
  reduce_axes.resize(rank, 0);
  for (auto axis : input_axes) {
    reduce_axes[axis] = 1;
  }
  size_t output_size = 1;
  size_t reduce_size = 1;
  // Compute output shape
  TensorShapeVector output_shape_vector;
  bool is_input_empty = false;
  for (size_t i = 0; i < input_tensor->Shape().NumDimensions(); ++i) {
    is_input_empty |= input_tensor->Shape()[i] == 0;
    if (reduce_axes[i] == 1) {
      if (keepdims_) {
        output_shape_vector.push_back(1);
      }
      reduce_size *= input_tensor->Shape()[i];
    } else {
      output_shape_vector.push_back(input_tensor->Shape()[i]);
      output_size *= input_tensor->Shape()[i];
    }
  }
  TensorShape output_shape(output_shape_vector);
  if (output_size == 0) {
    ORT_IGNORE_RETURN_VALUE(context.Output(0, output_shape));
    return Status::OK();
  }

  // Prefer the naive Reduce path when the shared path would pay extra overhead (transposing non-innermost reduce axes).
  constexpr size_t kReduceNaiveMaxReduceSize = 128;
  constexpr size_t kReduceNaiveMinOutputSize = 20000;
  bool are_axes_innermost = true;
  size_t axes_rank = input_axes.size();
  for (size_t i = 0; i < input_axes.size() && are_axes_innermost; ++i) {
    if (input_axes[axes_rank - 1 - i] != rank - 1 - i) {
      are_axes_innermost = false;
      break;
    }
  }
  bool use_naive_reduction = name_ == "ArgMin" || name_ == "ArgMax" || (reduce_size < 32 && output_size > 1024) ||
                             (!are_axes_innermost && reduce_size <= kReduceNaiveMaxReduceSize &&
                              output_size > kReduceNaiveMinOutputSize) ||
                             is_input_empty || input_tensor->Shape().NumDimensions() == 0;

  if (use_naive_reduction) {
    ReduceNaiveProgram program(name_, reduce_op_type, keepdims_, noop_with_empty_axes_, input_axes, is_input_empty);
    if (!is_input_empty) {
      program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank});
    }

    // TODO: the ReduceKernel class is designed to use `keepdims_`, `noop_with_empty_axes_` and input axes as uniform variables,
    //       but the current implementation does not work without them in cache key.
    //       This is a temporary workaround to make it work. We should fix this in the future.
    program.CacheHint(keepdims_,
                      noop_with_empty_axes_,
                      select_last_index_,
                      absl::StrJoin(input_axes, ","))
        .AddOutput({context.Output(0, output_shape), ProgramTensorMetadataDependency::TypeAndRank})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                              {static_cast<uint32_t>(noop_with_empty_axes_ ? 1 : 0)},
                              {static_cast<uint32_t>(reduce_size)}});

    return context.RunProgram(program);
  } else {
    Tensor input_transpose;
    if (!are_axes_innermost) {
      InlinedVector<size_t> perm;
      for (size_t i = 0; i < rank; ++i) {
        if (reduce_axes[i] == 0) {
          perm.push_back(static_cast<size_t>(i));
        }
      }
      for (size_t i = 0; i < rank; ++i) {
        if (reduce_axes[i] == 1) {
          perm.push_back(static_cast<size_t>(i));
        }
      }
      // If the axes are not innermost, we need to reorder the input tensor to make them innermost
      TensorShapeVector input_shape_vector = input_tensor->Shape().AsShapeVector();
      TensorShapeVector input_transpose_shape_vector(input_shape_vector.size());
      for (size_t i = 0; i < input_shape_vector.size(); ++i) {
        input_transpose_shape_vector[i] = input_shape_vector[perm[i]];
      }
      TensorShape input_transpose_shape(input_transpose_shape_vector);
      input_transpose = context.CreateGPUTensor(input_tensor->DataType(), input_transpose_shape);
      TransposeProgram transpose_program(perm, false);
      transpose_program.CacheHint(absl::StrJoin(perm, "-"))
          .AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
          .AddOutput({&input_transpose, ProgramTensorMetadataDependency::TypeAndRank})
          .SetDispatchGroupSize((input_tensor->Shape().Size() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
          .AddUniformVariable({static_cast<uint32_t>(input_transpose_shape.Size())});
      ORT_RETURN_IF_ERROR(context.RunProgram(transpose_program));
      input_tensor = &input_transpose;
    }
    auto workgroup_size = output_size == 1 ? static_cast<uint32_t>(256) : static_cast<uint32_t>(WORKGROUP_SIZE);
    ReduceSharedProgram program(name_, reduce_op_type, workgroup_size);
    program.CacheHint(keepdims_,
                      noop_with_empty_axes_,
                      select_last_index_,
                      workgroup_size,
                      absl::StrJoin(input_axes, ","))
        .AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({context.Output(0, output_shape), ProgramTensorMetadataDependency::TypeAndRank})
        .SetDispatchGroupSize(static_cast<uint32_t>(output_size))
        .SetWorkgroupSize(workgroup_size)
        .AddUniformVariable({static_cast<uint32_t>(reduce_size)});
    return context.RunProgram(program);
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
