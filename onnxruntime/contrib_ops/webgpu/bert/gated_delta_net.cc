// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

namespace {

constexpr uint32_t kValueChannelsPerWorkgroup = 4;
constexpr uint32_t kParallelPrefillChunkSize = 32;
constexpr uint32_t kDeltaPrefillChunkSize = 16;

GatedDeltaNetUpdateRule ParseUpdateRule(const std::string& rule) {
  if (rule == "linear") return GatedDeltaNetUpdateRule::Linear;
  if (rule == "gated") return GatedDeltaNetUpdateRule::Gated;
  if (rule == "delta") return GatedDeltaNetUpdateRule::Delta;
  if (rule == "gated_delta") return GatedDeltaNetUpdateRule::GatedDelta;
  return GatedDeltaNetUpdateRule::Invalid;
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    GatedDeltaNet,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 10)
        .MayInplace(6, 1),
    GatedDeltaNet);

GatedDeltaNet::GatedDeltaNet(const OpKernelInfo& info) : WebGpuKernel(info) {
  update_rule_ = ParseUpdateRule(info.GetAttrOrDefault<std::string>("update_rule", "gated_delta"));
  ORT_ENFORCE(update_rule_ != GatedDeltaNetUpdateRule::Invalid,
              "update_rule must be one of: linear, gated, delta, gated_delta");
  const auto gate_activation = info.GetAttrOrDefault<std::string>("gate_activation", "none");
  ORT_ENFORCE(gate_activation == "none" || gate_activation == "qwen",
              "gate_activation must be one of: none, qwen");
  qwen_gate_ = gate_activation == "qwen";
  const auto beta_activation = info.GetAttrOrDefault<std::string>("beta_activation", "none");
  ORT_ENFORCE(beta_activation == "none" || beta_activation == "sigmoid",
              "beta_activation must be one of: none, sigmoid");
  sigmoid_beta_ = beta_activation == "sigmoid";
  qk_l2_norm_ = info.GetAttrOrDefault<int64_t>("qk_l2_norm", 0) != 0;
  chunkwise_prefill_ = info.GetAttrOrDefault<int64_t>("chunk_size", 64) == kDeltaPrefillChunkSize;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  const auto state_update_capacity = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity >= 0 && state_update_capacity <= 8,
              "state_update_capacity must be in [0, 8], got ", state_update_capacity);
  state_update_capacity_ = static_cast<int>(state_update_capacity);
}

Status GatedDeltaNetProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseElementTypeAlias);
  const auto& key = shader.AddInput("key", ShaderUsage::UseElementTypeAlias);
  const auto& value = shader.AddInput("value", ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* cu_seqlens = &query;
  if (has_cu_seqlens_) cu_seqlens = &shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const ShaderVariableHelper* decay = &query;
  if ((update_rule_ == GatedDeltaNetUpdateRule::Gated || update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) &&
      !use_packed_params_) {
    decay = &shader.AddInput("decay", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* beta = &query;
  if ((update_rule_ == GatedDeltaNetUpdateRule::Delta || update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) &&
      !use_packed_params_) {
    beta = &shader.AddInput("beta", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* initial_state = &query;
  if (has_initial_state_ && !initial_state_in_final_state_) {
    initial_state = &shader.AddInput("initial_state", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* a_log = &query;
  const ShaderVariableHelper* dt_bias = &query;
  if (qwen_gate_ && !use_packed_params_) {
    a_log = &shader.AddInput("a_log", ShaderUsage::UseUniform);
    dt_bias = &shader.AddInput("dt_bias", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* parameters = &query;
  if (use_packed_params_) parameters = &shader.AddInput("parameters", ShaderUsage::UseUniform);
  const auto& output =
      shader.AddOutput("output", ShaderUsage::UseElementTypeAlias | ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper* final_state = &output;
  if (output_final_state_) final_state = &shader.AddOutput("final_state", ShaderUsage::UseUniform);
  const ShaderVariableHelper* capture_count = &query;
  if (capture_state_updates_) capture_count = &shader.AddInput("capture_count", ShaderUsage::UseUniform);
  const ShaderVariableHelper* state_update = &output;
  if (capture_state_updates_) state_update = &shader.AddOutput("state_update", ShaderUsage::UseUniform);

  int update_rule = 0;
  if (update_rule_ == GatedDeltaNetUpdateRule::Gated) update_rule = 1;
  if (update_rule_ == GatedDeltaNetUpdateRule::Delta) update_rule = 2;
  if (update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) update_rule = 3;
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(capture_state_updates, capture_state_updates_),
                             WGSL_TEMPLATE_PARAMETER(has_cu_seqlens, has_cu_seqlens_),
                             WGSL_TEMPLATE_PARAMETER(has_initial_state, has_initial_state_),
                             WGSL_TEMPLATE_PARAMETER(initial_state_in_final_state, initial_state_in_final_state_),
                             WGSL_TEMPLATE_PARAMETER(output_final_state, output_final_state_),
                             WGSL_TEMPLATE_PARAMETER(qk_l2_norm, qk_l2_norm_),
                             WGSL_TEMPLATE_PARAMETER(qwen_gate, qwen_gate_),
                             WGSL_TEMPLATE_PARAMETER(sigmoid_beta, sigmoid_beta_),
                             WGSL_TEMPLATE_PARAMETER(update_rule, update_rule),
                             WGSL_TEMPLATE_PARAMETER(use_packed_params, use_packed_params_),
                             WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, kValueChannelsPerWorkgroup),
                             WGSL_TEMPLATE_PARAMETER(vectorized_value_io, vectorized_value_io_),
                             WGSL_TEMPLATE_VARIABLE(a_log, *a_log),
                             WGSL_TEMPLATE_VARIABLE(beta, *beta),
                             WGSL_TEMPLATE_VARIABLE(capture_count, *capture_count),
                             WGSL_TEMPLATE_VARIABLE(cu_seqlens, *cu_seqlens),
                             WGSL_TEMPLATE_VARIABLE(decay, *decay),
                             WGSL_TEMPLATE_VARIABLE(dt_bias, *dt_bias),
                             WGSL_TEMPLATE_VARIABLE(final_state, *final_state),
                             WGSL_TEMPLATE_VARIABLE(initial_state, *initial_state),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(parameters, *parameters),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(state_update, *state_update),
                             WGSL_TEMPLATE_VARIABLE(value, value));
}

Status GatedDeltaNetNormalizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseElementTypeAlias);
  const auto& key = shader.AddInput("key", ShaderUsage::UseElementTypeAlias);
  const auto& normalized = shader.AddOutput("normalized", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_normalize.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(normalized, normalized),
                             WGSL_TEMPLATE_VARIABLE(query, query));
}

Status GatedDeltaNetChunkPrepareProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& normalized = shader.AddInput("normalized", ShaderUsage::UseUniform);
  const auto& value = shader.AddInput("value", ShaderUsage::UseElementTypeAlias);
  const auto& parameters = shader.AddInput("parameters", ShaderUsage::UseUniform);
  const ShaderVariableHelper* cu_seqlens = &normalized;
  if (has_cu_seqlens_) cu_seqlens = &shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const auto& prepared = shader.AddOutput("prepared", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_chunk_prepare.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(chunk_size, kDeltaPrefillChunkSize),
                             WGSL_TEMPLATE_PARAMETER(has_cu_seqlens, has_cu_seqlens_),
                             WGSL_TEMPLATE_VARIABLE(cu_seqlens, *cu_seqlens),
                             WGSL_TEMPLATE_VARIABLE(normalized, normalized),
                             WGSL_TEMPLATE_VARIABLE(parameters, parameters),
                             WGSL_TEMPLATE_VARIABLE(prepared, prepared),
                             WGSL_TEMPLATE_VARIABLE(value, value));
}

Status GatedDeltaNetChunkScanProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& prepared = shader.AddInput("prepared", ShaderUsage::UseUniform);
  const ShaderVariableHelper* cu_seqlens = &prepared;
  if (has_cu_seqlens_) cu_seqlens = &shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const ShaderVariableHelper* initial_state = &prepared;
  if (has_initial_state_ && !state_alias_) initial_state = &shader.AddInput("initial_state", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* final_state = &output;
  if (output_final_state_) final_state = &shader.AddOutput("final_state", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_chunk_scan.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(chunk_size, kDeltaPrefillChunkSize),
                             WGSL_TEMPLATE_PARAMETER(has_cu_seqlens, has_cu_seqlens_),
                             WGSL_TEMPLATE_PARAMETER(has_initial_state, has_initial_state_),
                             WGSL_TEMPLATE_PARAMETER(head_size, head_size_),
                             WGSL_TEMPLATE_PARAMETER(output_final_state, output_final_state_),
                             WGSL_TEMPLATE_PARAMETER(state_alias, state_alias_),
                             WGSL_TEMPLATE_VARIABLE(cu_seqlens, *cu_seqlens),
                             WGSL_TEMPLATE_VARIABLE(final_state, *final_state),
                             WGSL_TEMPLATE_VARIABLE(initial_state, *initial_state),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(prepared, prepared));
}

Status GatedDeltaNetClearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.element_count")
                            << "  " << output.SetByOffset("global_idx", "output_element_t(0.0)") << "\n";
  return Status::OK();
}

Status GatedDeltaNetPrefillPrepareProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseElementTypeAlias);
  const auto& value = shader.AddInput("value", ShaderUsage::UseElementTypeAlias);
  const auto& chunk_contribution = shader.AddOutput("chunk_contribution", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_prefill_prepare.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, kValueChannelsPerWorkgroup),
                             WGSL_TEMPLATE_VARIABLE(chunk_contribution, chunk_contribution),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(value, value));
}

Status GatedDeltaNetPrefillScanProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& chunk_contribution = shader.AddInput("chunk_contribution", ShaderUsage::UseUniform);
  const ShaderVariableHelper* initial_state = &chunk_contribution;
  if (has_initial_state_) initial_state = &shader.AddInput("initial_state", ShaderUsage::UseUniform);
  const ShaderVariableHelper* carry_state = &chunk_contribution;
  if (has_carry_state_) carry_state = &shader.AddInput("carry_state", ShaderUsage::UseUniform);
  const auto& chunk_state = shader.AddOutput("chunk_state", ShaderUsage::UseUniform);
  const auto& next_carry_state = shader.AddOutput("next_carry_state", ShaderUsage::UseUniform);
  const ShaderVariableHelper* final_state = &chunk_state;
  if (output_final_state_) final_state = &shader.AddOutput("final_state", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_prefill_scan.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_carry_state, has_carry_state_),
                             WGSL_TEMPLATE_PARAMETER(has_initial_state, has_initial_state_),
                             WGSL_TEMPLATE_PARAMETER(output_final_state, output_final_state_),
                             WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, kValueChannelsPerWorkgroup),
                             WGSL_TEMPLATE_VARIABLE(carry_state, *carry_state),
                             WGSL_TEMPLATE_VARIABLE(chunk_contribution, chunk_contribution),
                             WGSL_TEMPLATE_VARIABLE(chunk_state, chunk_state),
                             WGSL_TEMPLATE_VARIABLE(final_state, *final_state),
                             WGSL_TEMPLATE_VARIABLE(initial_state, *initial_state),
                             WGSL_TEMPLATE_VARIABLE(next_carry_state, next_carry_state));
}

Status GatedDeltaNetPrefillOutputProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseElementTypeAlias);
  const auto& key = shader.AddInput("key", ShaderUsage::UseElementTypeAlias);
  const auto& value = shader.AddInput("value", ShaderUsage::UseElementTypeAlias);
  const auto& chunk_state = shader.AddInput("chunk_state", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_prefill_output.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, kValueChannelsPerWorkgroup),
                             WGSL_TEMPLATE_VARIABLE(chunk_state, chunk_state),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(value, value));
}

Status GatedDeltaNetParamsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& parameters = shader.AddOutput("parameters", ShaderUsage::UseUniform);
  const ShaderVariableHelper* decay = &parameters;
  if (has_decay_ && !has_gate_projections_) decay = &shader.AddInput("decay", ShaderUsage::UseUniform);
  const ShaderVariableHelper* beta = &parameters;
  if (has_beta_ && !has_gate_projections_) beta = &shader.AddInput("beta", ShaderUsage::UseUniform);
  const ShaderVariableHelper* gate_projections = &parameters;
  if (has_gate_projections_) gate_projections = &shader.AddInput("gate_projections", ShaderUsage::UseUniform);
  const ShaderVariableHelper* a_log = &parameters;
  const ShaderVariableHelper* dt_bias = &parameters;
  if (qwen_gate_) {
    a_log = &shader.AddInput("a_log", ShaderUsage::UseUniform);
    dt_bias = &shader.AddInput("dt_bias", ShaderUsage::UseUniform);
  }
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_params.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_beta, has_beta_),
                             WGSL_TEMPLATE_PARAMETER(has_decay, has_decay_),
                             WGSL_TEMPLATE_PARAMETER(has_gate_projections, has_gate_projections_),
                             WGSL_TEMPLATE_PARAMETER(qwen_gate, qwen_gate_),
                             WGSL_TEMPLATE_PARAMETER(sigmoid_beta, sigmoid_beta_),
                             WGSL_TEMPLATE_VARIABLE(a_log, *a_log),
                             WGSL_TEMPLATE_VARIABLE(beta, *beta),
                             WGSL_TEMPLATE_VARIABLE(decay, *decay),
                             WGSL_TEMPLATE_VARIABLE(dt_bias, *dt_bias),
                             WGSL_TEMPLATE_VARIABLE(gate_projections, *gate_projections),
                             WGSL_TEMPLATE_VARIABLE(parameters, parameters));
}

Status GatedDeltaNetCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src");
  const auto& dst = shader.AddOutput("dst");
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.element_count")
                            << "  " << dst.SetByOffset("global_idx", src.GetByOffset("global_idx")) << "\n";
  return Status::OK();
}

Status GatedDeltaNetUnpackQkvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& packed_qkv = shader.AddInput("packed_qkv");
  const auto& unpacked_qkv = shader.AddOutput("unpacked_qkv");
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes(
             "uniforms.total_tokens * (2u * uniforms.query_size + uniforms.value_size)")
      << "  let query_elements = uniforms.total_tokens * uniforms.query_size;\n"
      << "  let value_base = 2u * query_elements;\n"
      << "  var token: u32;\n"
      << "  var offset: u32;\n"
      << "  if (global_idx < query_elements) {\n"
      << "    token = global_idx / uniforms.query_size;\n"
      << "    offset = global_idx % uniforms.query_size;\n"
      << "  } else if (global_idx < value_base) {\n"
      << "    let key_idx = global_idx - query_elements;\n"
      << "    token = key_idx / uniforms.query_size;\n"
      << "    offset = uniforms.query_size + key_idx % uniforms.query_size;\n"
      << "  } else {\n"
      << "    let value_idx = global_idx - value_base;\n"
      << "    token = value_idx / uniforms.value_size;\n"
      << "    offset = 2u * uniforms.query_size + value_idx % uniforms.value_size;\n"
      << "  }\n"
      << "  let packed_offset = token * (2u * uniforms.query_size + uniforms.value_size) + offset;\n"
      << "  " << unpacked_qkv.SetByOffset("global_idx", packed_qkv.GetByOffset("packed_offset")) << "\n";
  return Status::OK();
}

Status GatedDeltaNet::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* query = context.Input(0);
  const auto* key = context.Input(1);
  const auto* value = context.Input(2);
  const auto* cu_seqlens = context.Input(3);
  const auto* decay = context.Input(4);
  const auto* beta = context.Input(5);
  const auto* initial_state = context.Input(6);
  const auto* a_log = context.Input(7);
  const auto* dt_bias = context.Input(8);
  const auto* capture_count = context.Input(9);
  const auto* state_update_active = context.Input(10);
  const auto* gate_projections = context.Input(11);

  const bool needs_decay = update_rule_ == GatedDeltaNetUpdateRule::Gated ||
                           update_rule_ == GatedDeltaNetUpdateRule::GatedDelta;
  const bool needs_beta = update_rule_ == GatedDeltaNetUpdateRule::Delta ||
                          update_rule_ == GatedDeltaNetUpdateRule::GatedDelta;
  ORT_RETURN_IF_NOT(query != nullptr, "query is required");
  const bool input_is_packed_qkv = key == nullptr && value == nullptr;
  ORT_RETURN_IF_NOT(input_is_packed_qkv || (key != nullptr && value != nullptr),
                    "key and value must be both present or both absent");
  if (initial_state != nullptr) {
    ORT_RETURN_IF_NOT(initial_state->Shape().NumDimensions() == 4,
                      "initial_state must be rank 4 [batch, num_heads_v, head_size_v, head_size_qk]");
  }
  if (gate_projections != nullptr) {
    ORT_RETURN_IF_NOT(update_rule_ == GatedDeltaNetUpdateRule::GatedDelta && qwen_gate_ && sigmoid_beta_,
                      "gate_projections requires gated_delta, qwen and sigmoid");
    ORT_RETURN_IF_NOT(decay == nullptr && beta == nullptr,
                      "gate_projections requires omitted decay and beta");
  } else {
    ORT_RETURN_IF_NOT(needs_decay == (decay != nullptr), "decay input presence must match update_rule");
    ORT_RETURN_IF_NOT(needs_beta == (beta != nullptr), "beta input presence must match update_rule");
  }
  ORT_RETURN_IF_NOT((state_update_capacity_ > 0) == (capture_count != nullptr),
                    "capture_count must be present exactly when state_update_capacity is positive");
  if (state_update_active != nullptr) {
    ORT_RETURN_IF_NOT(state_update_active->Shape() == TensorShape({1}),
                      "state_update_active must have shape [1]");
  }

  if (capture_count != nullptr) {
    ORT_RETURN_IF_NOT(capture_count->Shape().NumDimensions() == 1,
                      "capture_count must be [batch]");
  }

  const auto& q_shape = query->Shape();
  const size_t rank = q_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_is_packed_qkv ? (rank == 2 || rank == 3) : (rank == 3 || rank == 4),
                    input_is_packed_qkv
                        ? "packed QKV must be rank 2 [total_tokens, packed_size] or rank 3 "
                          "[batch, sequence, packed_size]"
                        : "query, key and value must be rank 3 or 4");
  const size_t token_dims = rank - (input_is_packed_qkv ? 1 : 2);
  const int64_t total_tokens = q_shape.SizeToDimension(token_dims);
  constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxUint32 = std::numeric_limits<uint32_t>::max();
  ORT_RETURN_IF_NOT(total_tokens > 0, "total_tokens must be positive");

  int64_t hq = 0;
  int64_t hk = 0;
  int64_t hv = 0;
  int64_t dk = 0;
  int64_t dv = 0;
  if (input_is_packed_qkv) {
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "initial_state is required to derive packed QKV dimensions");
    const auto& state_shape = initial_state->Shape();
    ORT_RETURN_IF_NOT(state_shape[1] > 0 && state_shape[1] <= kMaxInt32 &&
                          state_shape[2] > 0 && state_shape[2] <= kMaxInt32 &&
                          state_shape[3] > 0 && state_shape[3] <= kMaxInt32,
                      "packed QKV head counts and head sizes must be positive and fit in int32");
    hv = state_shape[1];
    dv = state_shape[2];
    dk = state_shape[3];
    const int64_t packed_size = q_shape[token_dims];
    const int64_t value_size = hv * dv;
    ORT_RETURN_IF_NOT(hv > 0 && dv > 0 && dk > 0 && packed_size > value_size &&
                          (packed_size - value_size) % (2 * dk) == 0,
                      "packed QKV last dimension must be 2 * num_heads_q * head_size_qk + "
                      "num_heads_v * head_size_v");
    hq = (packed_size - value_size) / (2 * dk);
    hk = hq;
  } else {
    const auto& k_shape = key->Shape();
    const auto& v_shape = value->Shape();
    ORT_RETURN_IF_NOT(k_shape.NumDimensions() == rank && v_shape.NumDimensions() == rank,
                      "query, key and value must have the same rank");
    ORT_RETURN_IF_NOT(k_shape.SizeToDimension(token_dims) == total_tokens &&
                          v_shape.SizeToDimension(token_dims) == total_tokens,
                      "query, key and value must agree on a positive total_tokens");
    hq = q_shape[token_dims];
    hk = k_shape[token_dims];
    hv = v_shape[token_dims];
    dk = q_shape[token_dims + 1];
    dv = v_shape[token_dims + 1];
    ORT_RETURN_IF_NOT(k_shape[token_dims + 1] == dk,
                      "key head_size must equal query head_size");
  }
  ORT_RETURN_IF_NOT(hq > 0 && hq == hk && hv > 0 && hv % hq == 0 && dk > 0 && dv > 0,
                    "query/key heads must match and value heads must be a positive multiple of query heads");
  ORT_RETURN_IF_NOT(total_tokens <= kMaxInt32 && hq <= kMaxInt32 && hv <= kMaxInt32 &&
                        dk <= kMaxInt32 && dv <= kMaxInt32,
                    "GatedDeltaNet dimensions must fit in int32");
  ORT_RETURN_IF_NOT(total_tokens <= kMaxUint32 / hq / dk &&
                        total_tokens <= kMaxUint32 / hv / dv,
                    "GatedDeltaNet input sizes must fit in uint32");
  ORT_RETURN_IF_NOT(dk <= 256, "WebGPU GatedDeltaNet requires head_size_qk <= 256, got ", dk);

  int64_t batch = 1;
  if (cu_seqlens != nullptr) {
    ORT_RETURN_IF_NOT(token_dims == 1 && cu_seqlens->Shape().NumDimensions() == 1 &&
                          cu_seqlens->Shape()[0] >= 2,
                      "cu_seqlens requires one leading token dimension and must have at least two elements");
    batch = cu_seqlens->Shape()[0] - 1;
  } else if (token_dims == 2) {
    batch = q_shape[0];
  } else {
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "rank-3 uniform packing requires initial_state to determine batch size");
    batch = initial_state->Shape()[0];
    ORT_RETURN_IF_NOT(batch > 0, "batch size must be positive");
    ORT_RETURN_IF_NOT(total_tokens % batch == 0, "total_tokens must be divisible by batch");
  }
  ORT_RETURN_IF_NOT(batch > 0, "batch size must be positive");
  ORT_RETURN_IF_NOT(batch <= kMaxInt32 && batch <= kMaxUint32 / hv / dv &&
                        batch <= kMaxUint32 / hv / dv / dk,
                    "GatedDeltaNet dispatch size must fit in uint32");
  const TensorShape state_shape{batch, hv, dv, dk};
  if (initial_state != nullptr) {
    ORT_RETURN_IF_NOT(initial_state->Shape() == state_shape,
                      "initial_state must be [batch, num_heads_v, head_size_v, head_size_qk] (V-major)");
  }

  TensorShapeVector query_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  query_dims.insert(query_dims.end(), {hq, dk});
  TensorShapeVector value_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  value_dims.insert(value_dims.end(), {hv, dv});
  const TensorShape query_view_shape(query_dims);
  const TensorShape key_view_shape(query_dims);
  const TensorShape value_view_shape(value_dims);
  if (decay != nullptr) {
    ORT_RETURN_IF_NOT(decay->Shape().NumDimensions() == token_dims + 1 &&
                          decay->Shape().SizeToDimension(token_dims) == total_tokens &&
                          decay->Shape()[token_dims] == hv,
                      "WebGPU GatedDeltaNet supports scalar decay with shape [...tokens, num_heads_v]");
  }
  if (beta != nullptr) {
    ORT_RETURN_IF_NOT(beta->Shape().NumDimensions() == token_dims + 1 &&
                          beta->Shape().SizeToDimension(token_dims) == total_tokens &&
                          beta->Shape()[token_dims] == hv,
                      "beta must have shape [...tokens, num_heads_v]");
  }
  if (gate_projections != nullptr) {
    ORT_RETURN_IF_NOT(gate_projections->Shape().NumDimensions() == token_dims + 1 &&
                          gate_projections->Shape().SizeToDimension(token_dims) == total_tokens &&
                          gate_projections->Shape()[token_dims] == 2 * hv &&
                          gate_projections->GetElementType() == query->GetElementType(),
                      "gate_projections must match query type and have shape [...tokens, 2 * num_heads_v]");
    ORT_RETURN_IF_NOT(static_cast<uint64_t>(total_tokens) * hv * 2 <= kMaxUint32,
                      "gate_projections size must fit in uint32");
  }
  if (qwen_gate_) {
    ORT_RETURN_IF_NOT(a_log != nullptr && dt_bias != nullptr &&
                          a_log->Shape().NumDimensions() == 1 && a_log->Shape()[0] == hv &&
                          dt_bias->Shape().NumDimensions() == 1 && dt_bias->Shape()[0] == hv,
                      "gate_activation=qwen requires a_log and dt_bias with shape [num_heads_v]");
  } else {
    ORT_RETURN_IF_NOT(a_log == nullptr && dt_bias == nullptr, "a_log and dt_bias require gate_activation=qwen");
  }

  TensorShapeVector output_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  output_dims.push_back(std::max(hq, hv));
  output_dims.push_back(dv);
  auto* output = context.Output(0, TensorShape(output_dims));
  auto* final_state = context.Output(1, state_shape);
  const uint64_t state_update_width_64 = static_cast<uint64_t>(state_update_capacity_) *
                                         (static_cast<uint64_t>(hv) +
                                          static_cast<uint64_t>(hq) * dk +
                                          static_cast<uint64_t>(hv) * dv);
  ORT_RETURN_IF_NOT(state_update_width_64 <= kMaxUint32 &&
                        static_cast<uint64_t>(batch) * state_update_width_64 <= kMaxUint32,
                    "GatedDeltaNet state_update is too large for WebGPU");
  const int64_t state_update_width = static_cast<int64_t>(state_update_width_64);
  auto* state_update = context.Output(2, TensorShape{batch, state_update_width});
  ORT_RETURN_IF_NOT(output != nullptr, "output is required");

  ORT_RETURN_IF(capture_count != nullptr && capture_count->Shape()[0] != batch,
                "capture_count must be [batch]");
  const bool capture_state_updates = state_update != nullptr && state_update_capacity_ > 0 &&
                                     (state_update_active == nullptr || state_update_active->Data<int32_t>()[0] != 0);
  if (state_update != nullptr && state_update->Shape().Size() > 0 && !capture_state_updates) {
    GatedDeltaNetClearProgram clear_program;
    clear_program.AddOutput({state_update, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(state_update->Shape().Size()) + WORKGROUP_SIZE - 1) /
                              WORKGROUP_SIZE)
        .SetWorkgroupSize(WORKGROUP_SIZE)
        .AddUniformVariable({onnxruntime::narrow<uint32_t>(state_update->Shape().Size())});
    ORT_RETURN_IF_ERROR(context.RunProgram(clear_program));
  }

  // A WebGPU storage buffer cannot be bound for both read-only and read-write access in one pass.
  const bool state_alias =
      initial_state != nullptr && final_state != nullptr && initial_state->DataRaw() == final_state->DataRaw();
  const auto binding_count = [&context](const Tensor* tensor) {
    if (tensor == nullptr) return 0u;
    const uint64_t max_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
    return onnxruntime::narrow<uint32_t>((tensor->SizeInBytes() + max_binding_size - 1) / max_binding_size);
  };
  const uint32_t source_qkv_binding_count = input_is_packed_qkv
                                                ? binding_count(query)
                                                : binding_count(query) + binding_count(key) + binding_count(value);
  uint32_t direct_binding_count = source_qkv_binding_count +
                                  binding_count(cu_seqlens) + binding_count(decay) + binding_count(beta) + binding_count(gate_projections) +
                                  binding_count(initial_state) + binding_count(a_log) + binding_count(dt_bias) +
                                  binding_count(capture_state_updates ? capture_count : nullptr) +
                                  binding_count(output) + binding_count(final_state) +
                                  binding_count(capture_state_updates ? state_update : nullptr);
  if (state_alias) direct_binding_count -= binding_count(initial_state);
  const uint32_t max_storage_buffers = context.DeviceLimits().maxStorageBuffersPerShaderStage;
  const uint32_t qkv_binding_count = source_qkv_binding_count;
  const uint64_t qkv_element_count =
      static_cast<uint64_t>(query_view_shape.Size()) + key_view_shape.Size() + value_view_shape.Size();
  const uint64_t qkv_size_in_bytes = input_is_packed_qkv
                                         ? query->SizeInBytes()
                                         : query->SizeInBytes() + key->SizeInBytes() + value->SizeInBytes();
  const uint32_t packed_qkv_binding_count =
      onnxruntime::narrow<uint32_t>((qkv_size_in_bytes + context.DeviceLimits().maxStorageBufferBindingSize - 1) /
                                    context.DeviceLimits().maxStorageBufferBindingSize);
  const uint32_t largest_qkv_binding_count = input_is_packed_qkv
                                                 ? binding_count(query)
                                                 : std::max({binding_count(query), binding_count(key), binding_count(value)});
  const bool needs_dynamic_params = qwen_gate_ || (needs_decay && needs_beta);
  uint32_t dynamic_param_binding_count = 0;
  if (needs_decay) dynamic_param_binding_count += binding_count(decay);
  if (needs_beta) dynamic_param_binding_count += binding_count(beta);
  dynamic_param_binding_count += binding_count(gate_projections);
  if (qwen_gate_) dynamic_param_binding_count += binding_count(a_log) + binding_count(dt_bias);
  const uint64_t packed_params_size_in_bytes =
      static_cast<uint64_t>(total_tokens) * hv * 2 * sizeof(float);
  const uint32_t packed_params_binding_count =
      onnxruntime::narrow<uint32_t>((packed_params_size_in_bytes + context.DeviceLimits().maxStorageBufferBindingSize - 1) /
                                    context.DeviceLimits().maxStorageBufferBindingSize);
  const bool can_pack_params =
      needs_dynamic_params &&
      dynamic_param_binding_count + packed_params_binding_count <= max_storage_buffers;
  ORT_RETURN_IF_NOT(gate_projections == nullptr || can_pack_params,
                    "gate_projections exceeds WebGPU parameter binding limits");
  const bool can_copy_qkv =
      qkv_element_count <= kMaxUint32 &&
      largest_qkv_binding_count + packed_qkv_binding_count <= max_storage_buffers;
  ORT_RETURN_IF_NOT(!input_is_packed_qkv || can_copy_qkv,
                    "packed QKV input exceeds WebGPU buffer binding limits");
  const bool use_packed_qkv = input_is_packed_qkv ||
                              (direct_binding_count > max_storage_buffers &&
                               !(gate_projections != nullptr &&
                                 direct_binding_count - dynamic_param_binding_count + packed_params_binding_count <= max_storage_buffers) &&
                               can_copy_qkv &&
                               (direct_binding_count - qkv_binding_count + packed_qkv_binding_count <=
                                    max_storage_buffers ||
                                (can_pack_params &&
                                 direct_binding_count - qkv_binding_count + packed_qkv_binding_count -
                                         dynamic_param_binding_count + packed_params_binding_count <=
                                     max_storage_buffers)));
  std::optional<Tensor> packed_qkv;
  if (use_packed_qkv) {
    packed_qkv.emplace(
        context.CreateGPUTensor(query->DataType(), TensorShape{onnxruntime::narrow<int64_t>(qkv_element_count)}));
    if (input_is_packed_qkv) {
      GatedDeltaNetUnpackQkvProgram unpack_program;
      unpack_program
          .AddInput({query, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*packed_qkv, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(qkv_element_count) + WORKGROUP_SIZE - 1) /
                                WORKGROUP_SIZE)
          .SetWorkgroupSize(WORKGROUP_SIZE)
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                                {onnxruntime::narrow<uint32_t>(hq * dk)},
                                {onnxruntime::narrow<uint32_t>(hv * dv)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(unpack_program));
    } else {
      uint32_t packed_offset = 0;
      const auto copy_to_packed_qkv = [&](const Tensor* source) -> Status {
        GatedDeltaNetCopyProgram copy_program;
        copy_program
            .AddInput({source, ProgramTensorMetadataDependency::Type})
            .AddOutput(ProgramOutput::BufferView(&*packed_qkv,
                                                 ProgramTensorMetadataDependency::Type,
                                                 source->Shape(),
                                                 packed_offset))
            .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(source->Shape().Size()) + WORKGROUP_SIZE - 1) /
                                  WORKGROUP_SIZE)
            .SetWorkgroupSize(WORKGROUP_SIZE)
            .AddUniformVariable({onnxruntime::narrow<uint32_t>(source->Shape().Size())});
        ORT_RETURN_IF_ERROR(context.RunProgram(copy_program));
        packed_offset += onnxruntime::narrow<uint32_t>(source->Shape().Size());
        return Status::OK();
      };
      ORT_RETURN_IF_ERROR(copy_to_packed_qkv(query));
      ORT_RETURN_IF_ERROR(copy_to_packed_qkv(key));
      ORT_RETURN_IF_ERROR(copy_to_packed_qkv(value));
    }
  }
  const uint32_t binding_count_after_qkv =
      direct_binding_count - (use_packed_qkv ? qkv_binding_count - packed_qkv_binding_count : 0);
  const bool use_packed_params =
      needs_dynamic_params &&
      can_pack_params &&
      (gate_projections != nullptr ||
       (binding_count_after_qkv > max_storage_buffers &&
        binding_count_after_qkv - dynamic_param_binding_count + packed_params_binding_count <= max_storage_buffers));
  std::optional<Tensor> packed_params;
  if (use_packed_params) {
    packed_params.emplace(
        context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{total_tokens, hv, 2}));
    GatedDeltaNetParamsProgram params_program{needs_decay, needs_beta, qwen_gate_, sigmoid_beta_, gate_projections != nullptr};
    if (decay != nullptr) params_program.AddInput({decay, ProgramTensorMetadataDependency::None});
    if (beta != nullptr) params_program.AddInput({beta, ProgramTensorMetadataDependency::None});
    if (gate_projections != nullptr) params_program.AddInput({gate_projections, ProgramTensorMetadataDependency::Type});
    if (qwen_gate_) params_program.AddInputs({{a_log, ProgramTensorMetadataDependency::None},
                                              {dt_bias, ProgramTensorMetadataDependency::None}});
    params_program.AddOutput({&*packed_params, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(
            (static_cast<uint64_t>(total_tokens) * static_cast<uint64_t>(hv) + 63u) / 64u))
        .SetWorkgroupSize(64)
        .CacheHint(needs_decay, needs_beta, qwen_gate_, sigmoid_beta_, gate_projections != nullptr)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                              {onnxruntime::narrow<uint32_t>(hv)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(params_program));
  }

  const float scale = scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(dk));
  const uint64_t delta_chunks = GatedDeltaNetChunkCapacity(total_tokens, batch, kDeltaPrefillChunkSize,
                                                           cu_seqlens != nullptr);
  const uint64_t delta_chunk_elements = kDeltaPrefillChunkSize * (3ull * dk + dv) +
                                        kDeltaPrefillChunkSize * kDeltaPrefillChunkSize + 1;
  constexpr uint64_t delta_workspace_cap = 64ull << 20;
  const uint64_t delta_chunk_count = std::min<uint64_t>(hv * delta_chunks,
                                                        delta_workspace_cap / sizeof(float) / delta_chunk_elements + 1);
  const uint64_t delta_prepared_elements = delta_chunk_count * delta_chunk_elements;
  const uint64_t normalized_elements = 2ull * total_tokens * hq * dk;
  const uint64_t delta_workspace_bytes = (delta_prepared_elements + normalized_elements) * sizeof(float) +
                                         packed_params_size_in_bytes;
  const uint64_t delta_workgroup_bytes =
      (2ull * kDeltaPrefillChunkSize * 128 +
       2ull * kDeltaPrefillChunkSize * kDeltaPrefillChunkSize +
       2ull * kDeltaPrefillChunkSize) *
      sizeof(float);
  const auto fits_single_binding = [&context](uint64_t bytes) {
    return bytes <= context.DeviceLimits().maxStorageBufferBindingSize && bytes <= context.DeviceLimits().maxBufferSize;
  };
  const bool use_chunkwise_prefill = chunkwise_prefill_ &&
                                     context.AdapterInfo().vendor == std::string_view{"apple"} &&
                                     update_rule_ == GatedDeltaNetUpdateRule::GatedDelta &&
                                     qwen_gate_ && sigmoid_beta_ && qk_l2_norm_ && !capture_state_updates &&
                                     total_tokens / batch >= 32 && dk > 0 && dv > 0 && dk <= 128 && dv <= 128 &&
                                     delta_workspace_bytes <= delta_workspace_cap && max_storage_buffers >= 5 &&
                                     delta_workgroup_bytes <= context.DeviceLimits().maxComputeWorkgroupStorageSize &&
                                     fits_single_binding(delta_prepared_elements * sizeof(float)) &&
                                     fits_single_binding(normalized_elements * sizeof(float)) &&
                                     fits_single_binding(packed_params_size_in_bytes) &&
                                     binding_count(query) == 1 && binding_count(key) <= 1 && binding_count(value) <= 1 &&
                                     (!use_packed_qkv || binding_count(&*packed_qkv) == 1) &&
                                     binding_count(initial_state) <= 1 && binding_count(final_state) <= 1 &&
                                     binding_count(output) == 1 && binding_count(cu_seqlens) <= 1;
  if (use_chunkwise_prefill) {
    Tensor normalized = context.CreateGPUTensor(DataTypeImpl::GetType<float>(),
                                                TensorShape{onnxruntime::narrow<int64_t>(normalized_elements)});
    Tensor prepared = context.CreateGPUTensor(DataTypeImpl::GetType<float>(),
                                              TensorShape{onnxruntime::narrow<int64_t>(delta_prepared_elements)});
    if (!use_packed_params) {
      packed_params.emplace(context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{total_tokens, hv, 2}));
      GatedDeltaNetParamsProgram params_program{true, true, true, true};
      params_program.AddInputs({{decay, ProgramTensorMetadataDependency::None},
                                {beta, ProgramTensorMetadataDependency::None},
                                {a_log, ProgramTensorMetadataDependency::None},
                                {dt_bias, ProgramTensorMetadataDependency::None}})
          .AddOutput({&*packed_params, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64)
          .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>((static_cast<uint64_t>(total_tokens) * hv + 63) / 64))
          .CacheHint(true, true, true, true)
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)}, {onnxruntime::narrow<uint32_t>(hv)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(params_program));
    }
    GatedDeltaNetNormalizeProgram normalize_program;
    if (use_packed_qkv) {
      normalize_program.AddInputs({ProgramInput::BufferView(&*packed_qkv, ProgramTensorMetadataDependency::Type,
                                                            query_view_shape, 0),
                                   ProgramInput::BufferView(&*packed_qkv, ProgramTensorMetadataDependency::Type,
                                                            key_view_shape,
                                                            onnxruntime::narrow<uint32_t>(query_view_shape.Size()))});
    } else {
      normalize_program.AddInputs({{query, ProgramTensorMetadataDependency::Type},
                                   {key, ProgramTensorMetadataDependency::Type}});
    }
    normalize_program.AddOutput({&normalized, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(static_cast<uint64_t>(total_tokens) * hq))
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                              {onnxruntime::narrow<uint32_t>(hq)},
                              {onnxruntime::narrow<uint32_t>(dk)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(normalize_program));
    GatedDeltaNetChunkPrepareProgram prepare_program{cu_seqlens != nullptr};
    prepare_program.AddInput({&normalized, ProgramTensorMetadataDependency::None});
    if (use_packed_qkv) {
      prepare_program.AddInput(ProgramInput::BufferView(&*packed_qkv, ProgramTensorMetadataDependency::Type,
                                                        value_view_shape,
                                                        onnxruntime::narrow<uint32_t>(query_view_shape.Size() +
                                                                                      key_view_shape.Size())));
    } else {
      prepare_program.AddInput({value, ProgramTensorMetadataDependency::Type});
    }
    prepare_program.AddInput({&*packed_params, ProgramTensorMetadataDependency::None});
    if (cu_seqlens != nullptr) prepare_program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
    prepare_program.AddOutput({&prepared, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(hv * delta_chunks))
        .CacheHint(cu_seqlens != nullptr)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)}, {onnxruntime::narrow<uint32_t>(batch)}, {onnxruntime::narrow<uint32_t>(hq)}, {onnxruntime::narrow<uint32_t>(hv)}, {onnxruntime::narrow<uint32_t>(dk)}, {onnxruntime::narrow<uint32_t>(dv)}, {onnxruntime::narrow<uint32_t>(delta_chunks)}, {onnxruntime::narrow<uint32_t>(delta_chunk_elements)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(prepare_program));
    GatedDeltaNetChunkScanProgram scan_program{onnxruntime::narrow<uint32_t>(dk), cu_seqlens != nullptr,
                                               initial_state != nullptr, state_alias, final_state != nullptr};
    scan_program.AddInput({&prepared, ProgramTensorMetadataDependency::None});
    if (cu_seqlens != nullptr) scan_program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
    if (initial_state != nullptr && !state_alias) {
      scan_program.AddInput({initial_state, ProgramTensorMetadataDependency::None});
    }
    scan_program.AddOutput({output, ProgramTensorMetadataDependency::Type});
    if (final_state != nullptr) scan_program.AddOutput({final_state, ProgramTensorMetadataDependency::None});
    scan_program.SetWorkgroupSize(128)
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(static_cast<uint64_t>(batch) * hv))
        .CacheHint(dk, cu_seqlens != nullptr, initial_state != nullptr, state_alias, final_state != nullptr)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)}, {onnxruntime::narrow<uint32_t>(batch)}, {onnxruntime::narrow<uint32_t>(hv)}, {onnxruntime::narrow<uint32_t>(dk)}, {onnxruntime::narrow<uint32_t>(dv)}, {onnxruntime::narrow<uint32_t>(delta_chunks)}, {onnxruntime::narrow<uint32_t>(delta_chunk_elements)}, {scale}});
    return context.RunProgram(scan_program);
  }
  uint32_t workgroup_size = 1;
  while (workgroup_size < dk) workgroup_size <<= 1;
  const uint32_t value_tiles =
      (onnxruntime::narrow<uint32_t>(dv) + kValueChannelsPerWorkgroup - 1) / kValueChannelsPerWorkgroup;
  const auto add_qkv_inputs = [&](auto& program) {
    if (use_packed_qkv) {
      const uint32_t query_offset = 0;
      const uint32_t key_offset = onnxruntime::narrow<uint32_t>(query_view_shape.Size());
      const uint32_t value_offset = key_offset + onnxruntime::narrow<uint32_t>(key_view_shape.Size());
      program.AddInputs({ProgramInput::BufferView(&*packed_qkv,
                                                  ProgramTensorMetadataDependency::Type,
                                                  query_view_shape,
                                                  query_offset),
                         ProgramInput::BufferView(&*packed_qkv,
                                                  ProgramTensorMetadataDependency::Type,
                                                  key_view_shape,
                                                  key_offset),
                         ProgramInput::BufferView(&*packed_qkv,
                                                  ProgramTensorMetadataDependency::Type,
                                                  value_view_shape,
                                                  value_offset)});
      return;
    }
    program.AddInputs({{query, ProgramTensorMetadataDependency::Type},
                       {key, ProgramTensorMetadataDependency::Type},
                       {value, ProgramTensorMetadataDependency::Type}});
  };
  const uint32_t sequence_length = onnxruntime::narrow<uint32_t>(total_tokens / batch);
  const auto add_key_value_inputs = [&](auto& program) {
    if (use_packed_qkv) {
      const uint32_t key_offset = onnxruntime::narrow<uint32_t>(query_view_shape.Size());
      const uint32_t value_offset = key_offset + onnxruntime::narrow<uint32_t>(key_view_shape.Size());
      program.AddInputs({ProgramInput::BufferView(&*packed_qkv,
                                                  ProgramTensorMetadataDependency::Type,
                                                  key_view_shape,
                                                  key_offset),
                         ProgramInput::BufferView(&*packed_qkv,
                                                  ProgramTensorMetadataDependency::Type,
                                                  value_view_shape,
                                                  value_offset)});
      return;
    }
    program.AddInputs({{key, ProgramTensorMetadataDependency::Type},
                       {value, ProgramTensorMetadataDependency::Type}});
  };

  // Linear state transitions are additive and can be split safely. All other
  // update rules retain the recurrent path, including state-dependent delta rules.
  const bool prefill_rule_supported = update_rule_ == GatedDeltaNetUpdateRule::Linear;
  const uint32_t total_chunks =
      (sequence_length + kParallelPrefillChunkSize - 1) / kParallelPrefillChunkSize;
  const uint64_t state_elements =
      static_cast<uint64_t>(batch) * static_cast<uint64_t>(hv) * static_cast<uint64_t>(dv) * dk;
  constexpr uint64_t kMaxParallelPrefillWorkspaceBytes = 64ull << 20;
  const uint64_t workspace_cap_bytes =
      std::min<uint64_t>(kMaxParallelPrefillWorkspaceBytes, context.DeviceLimits().maxBufferSize / 8);
  const auto prefill_plan =
      SelectGatedDeltaNetParallelPrefillPlan(state_elements, total_chunks, workspace_cap_bytes);
  const auto binding_count_for_bytes = [&context](uint64_t bytes) {
    const uint64_t max_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
    return (bytes + max_binding_size - 1) / max_binding_size;
  };
  const uint64_t state_bytes = state_elements * sizeof(float);
  const uint64_t chunk_state_bytes = prefill_plan.has_value()
                                         ? static_cast<uint64_t>(prefill_plan->chunks_per_pass) * state_bytes
                                         : 0;
  const uint64_t qkv_binding_count_for_prefill =
      use_packed_qkv ? binding_count(&*packed_qkv)
                     : binding_count(query) + binding_count(key) + binding_count(value);
  const uint64_t key_value_binding_count =
      use_packed_qkv ? binding_count(&*packed_qkv) : binding_count(key) + binding_count(value);
  const uint64_t chunk_state_binding_count = binding_count_for_bytes(chunk_state_bytes);
  const uint64_t carry_binding_count = binding_count_for_bytes(state_bytes);
  const uint64_t prepare_binding_count = key_value_binding_count + chunk_state_binding_count;
  const uint64_t scan_first_binding_count = chunk_state_binding_count + binding_count(initial_state) +
                                            chunk_state_binding_count +
                                            carry_binding_count + binding_count(final_state);
  const uint64_t scan_later_binding_count = chunk_state_binding_count + carry_binding_count +
                                            chunk_state_binding_count +
                                            carry_binding_count + binding_count(final_state);
  const uint64_t output_binding_count =
      qkv_binding_count_for_prefill + chunk_state_binding_count + binding_count(output);
  const uint64_t prefill_dispatch_group_count =
      static_cast<uint64_t>(batch) * hv * value_tiles *
      (prefill_plan.has_value() ? prefill_plan->chunks_per_pass : 0);
  const bool use_parallel_prefill =
      prefill_rule_supported &&
      cu_seqlens == nullptr &&
      !qwen_gate_ &&
      !qk_l2_norm_ &&
      !state_alias &&
      !capture_state_updates &&
      !use_packed_params &&
      prefill_plan.has_value() &&
      prefill_dispatch_group_count <= kMaxUint32 &&
      prepare_binding_count <= max_storage_buffers &&
      scan_first_binding_count <= max_storage_buffers &&
      scan_later_binding_count <= max_storage_buffers &&
      output_binding_count <= max_storage_buffers;
  if (use_parallel_prefill) {
    const uint32_t chunks_per_pass = prefill_plan->chunks_per_pass;
    const TensorShape chunk_state_shape{chunks_per_pass, batch, hv, dv, dk};
    Tensor chunk_contribution =
        context.CreateGPUTensor(DataTypeImpl::GetType<float>(), chunk_state_shape);
    Tensor chunk_state =
        context.CreateGPUTensor(DataTypeImpl::GetType<float>(), chunk_state_shape);
    Tensor carry_state_a =
        context.CreateGPUTensor(DataTypeImpl::GetType<float>(), state_shape);
    Tensor carry_state_b =
        context.CreateGPUTensor(DataTypeImpl::GetType<float>(), state_shape);

    for (uint32_t chunk_base = 0; chunk_base < total_chunks; chunk_base += chunks_per_pass) {
      const uint32_t chunks_in_pass = std::min(chunks_per_pass, total_chunks - chunk_base);
      const uint64_t dispatch_group_count =
          static_cast<uint64_t>(batch) * hv * value_tiles * chunks_in_pass;

      GatedDeltaNetPrefillPrepareProgram prepare_program;
      add_key_value_inputs(prepare_program);
      prepare_program
          .AddOutput({&chunk_contribution, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(dispatch_group_count))
          .SetWorkgroupSize(workgroup_size)
          .CacheHint(use_packed_qkv, workgroup_size, kValueChannelsPerWorkgroup)
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                                {onnxruntime::narrow<uint32_t>(batch)},
                                {onnxruntime::narrow<uint32_t>(hq)},
                                {onnxruntime::narrow<uint32_t>(hv)},
                                {onnxruntime::narrow<uint32_t>(dk)},
                                {onnxruntime::narrow<uint32_t>(dv)},
                                {kParallelPrefillChunkSize},
                                {chunk_base},
                                {chunks_in_pass}});
      ORT_RETURN_IF_ERROR(context.RunProgram(prepare_program));

      const bool is_first_pass = chunk_base == 0;
      const bool is_last_pass = chunk_base + chunks_in_pass == total_chunks;
      const bool write_carry_a = ((chunk_base / chunks_per_pass) & 1u) == 0;
      Tensor* next_carry_state = write_carry_a ? &carry_state_a : &carry_state_b;
      const Tensor* carry_state = write_carry_a ? &carry_state_b : &carry_state_a;
      GatedDeltaNetPrefillScanProgram scan_program{
          is_first_pass && initial_state != nullptr, !is_first_pass, final_state != nullptr};
      scan_program
          .AddInput({&chunk_contribution, ProgramTensorMetadataDependency::None});
      if (is_first_pass && initial_state != nullptr) {
        scan_program.AddInput({initial_state, ProgramTensorMetadataDependency::None});
      }
      if (!is_first_pass) {
        scan_program.AddInput({carry_state, ProgramTensorMetadataDependency::None});
      }
      scan_program
          .AddOutputs({{&chunk_state, ProgramTensorMetadataDependency::None},
                       {next_carry_state, ProgramTensorMetadataDependency::None}});
      if (final_state != nullptr) {
        scan_program.AddOutput({final_state, ProgramTensorMetadataDependency::None});
      }
      scan_program
          .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(static_cast<uint64_t>(batch) * hv * value_tiles))
          .SetWorkgroupSize(workgroup_size)
          .CacheHint(is_first_pass && initial_state != nullptr, !is_first_pass, final_state != nullptr,
                     workgroup_size, kValueChannelsPerWorkgroup)
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(batch)},
                                {onnxruntime::narrow<uint32_t>(hv)},
                                {onnxruntime::narrow<uint32_t>(dk)},
                                {onnxruntime::narrow<uint32_t>(dv)},
                                {chunks_in_pass},
                                {is_last_pass ? 1u : 0u}});
      ORT_RETURN_IF_ERROR(context.RunProgram(scan_program));

      GatedDeltaNetPrefillOutputProgram output_program;
      add_qkv_inputs(output_program);
      output_program
          .AddInput({&chunk_state, ProgramTensorMetadataDependency::None})
          .AddOutput({output, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(dispatch_group_count))
          .SetWorkgroupSize(workgroup_size)
          .CacheHint(use_packed_qkv, workgroup_size, kValueChannelsPerWorkgroup)
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                                {onnxruntime::narrow<uint32_t>(batch)},
                                {onnxruntime::narrow<uint32_t>(hq)},
                                {onnxruntime::narrow<uint32_t>(hv)},
                                {onnxruntime::narrow<uint32_t>(dk)},
                                {onnxruntime::narrow<uint32_t>(dv)},
                                {kParallelPrefillChunkSize},
                                {chunk_base},
                                {chunks_in_pass},
                                {scale}});
      ORT_RETURN_IF_ERROR(context.RunProgram(output_program));
    }
    return Status::OK();
  }

  const bool vectorized_value_io = !use_packed_qkv && dv % kValueChannelsPerWorkgroup == 0;
  const int value_io_components = vectorized_value_io ? onnxruntime::narrow<int>(kValueChannelsPerWorkgroup) : 1;
  GatedDeltaNetProgram program{update_rule_, cu_seqlens != nullptr, initial_state != nullptr, state_alias,
                               final_state != nullptr, qwen_gate_, sigmoid_beta_, qk_l2_norm_, use_packed_params,
                               capture_state_updates, vectorized_value_io};
  if (use_packed_qkv) {
    add_qkv_inputs(program);
  } else {
    program.AddInputs({{query, ProgramTensorMetadataDependency::Type},
                       {key, ProgramTensorMetadataDependency::Type},
                       {value, ProgramTensorMetadataDependency::Type, value_io_components}});
  }
  if (cu_seqlens != nullptr) program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
  if (decay != nullptr && !use_packed_params) program.AddInput({decay, ProgramTensorMetadataDependency::None});
  if (beta != nullptr && !use_packed_params) program.AddInput({beta, ProgramTensorMetadataDependency::None});
  if (initial_state != nullptr && !state_alias) program.AddInput({initial_state, ProgramTensorMetadataDependency::None});
  if (qwen_gate_ && !use_packed_params) program.AddInputs({{a_log, ProgramTensorMetadataDependency::None},
                                                           {dt_bias, ProgramTensorMetadataDependency::None}});
  if (use_packed_params) program.AddInput({&*packed_params, ProgramTensorMetadataDependency::None});
  if (capture_state_updates) program.AddInput({capture_count, ProgramTensorMetadataDependency::None});
  program.AddOutput({output, ProgramTensorMetadataDependency::Type, value_io_components});
  if (final_state != nullptr) {
    program.AddOutput({final_state, ProgramTensorMetadataDependency::None});
  }
  if (capture_state_updates) program.AddOutput({state_update, ProgramTensorMetadataDependency::None});
  program
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(batch * hv) * value_tiles)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(static_cast<int>(update_rule_), cu_seqlens != nullptr, initial_state != nullptr, state_alias,
                 final_state != nullptr, qwen_gate_, sigmoid_beta_, qk_l2_norm_, use_packed_qkv, use_packed_params,
                 capture_state_updates, vectorized_value_io, workgroup_size, kValueChannelsPerWorkgroup)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                            {onnxruntime::narrow<uint32_t>(batch)},
                            {onnxruntime::narrow<uint32_t>(hq)},
                            {onnxruntime::narrow<uint32_t>(hv)},
                            {onnxruntime::narrow<uint32_t>(dk)},
                            {onnxruntime::narrow<uint32_t>(dv)},
                            {onnxruntime::narrow<uint32_t>(state_update_capacity_)},
                            {scale}});
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
