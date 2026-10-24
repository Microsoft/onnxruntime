// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/dynamic_sparse_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

constexpr uint32_t kMaxBatchSize = 64;
constexpr uint32_t kMaxSequenceLength = 256;
constexpr uint32_t kMaxNumHeads = 128;
constexpr uint32_t kMaxHeadSize = 256;
constexpr uint32_t kMaxCacheLength = 65536;
constexpr uint32_t kMaxSelected = 4096;
constexpr uint32_t kMaxLocalWindow = 4096;
constexpr uint32_t kAttentionWorkgroupSize = 64;
static_assert(kMaxHeadSize % kAttentionWorkgroupSize == 0);
constexpr uint32_t kAttentionValuesPerInvocation = kMaxHeadSize / kAttentionWorkgroupSize;

DynamicSparseAttentionMode ParseAttentionMode(const std::string& value) {
  ORT_ENFORCE(value == "selected_only" || value == "local_plus_selected",
              "DynamicSparseAttention: attention_mode must be selected_only or local_plus_selected.");
  return value == "selected_only" ? DynamicSparseAttentionMode::kSelectedOnly
                                  : DynamicSparseAttentionMode::kLocalPlusSelected;
}

DynamicSparseAttentionKvSource ParseKvSource(const std::string& value) {
  ORT_ENFORCE(value == "main" || value == "auxiliary",
              "DynamicSparseAttention: selected_kv_source must be main or auxiliary.");
  return value == "main" ? DynamicSparseAttentionKvSource::kMain
                         : DynamicSparseAttentionKvSource::kAuxiliary;
}

bool ParseBoolAttribute(const OpKernelInfo& info, const char* name, int64_t default_value) {
  const int64_t value = info.GetAttrOrDefault<int64_t>(name, default_value);
  ORT_ENFORCE(value == 0 || value == 1, "DynamicSparseAttention: ", name, " must be 0 or 1.");
  return value != 0;
}

Status NotImplementedBound(const char* name, int value, uint32_t limit) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "DynamicSparseAttention (WebGPU): ",
                         name, "=", value, " exceeds the supported bound ", limit, ".");
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    DynamicSparseAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TCache", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())
        .MayInplace(3, 1)
        .MayInplace(4, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 10),
    DynamicSparseAttention);

Status DynamicSparseAttentionPrepareQueryProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& seqlens_k = shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  const ShaderVariableHelper* q_norm_weight = nullptr;
  if (use_qk_norm_) {
    q_norm_weight = &shader.AddInput("q_norm_weight", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* cos_cache = nullptr;
  const ShaderVariableHelper* sin_cache = nullptr;
  if (do_rotary_) {
    cos_cache = &shader.AddInput("cos_cache", ShaderUsage::UseUniform);
    sin_cache = &shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* position_ids = nullptr;
  if (has_position_ids_) {
    position_ids = &shader.AddInput("position_ids", ShaderUsage::UseUniform);
  }

  const auto& prepared_query =
      shader.AddOutput("prepared_query", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  if (use_qk_norm_) {
    shader.AdditionalImplementation()
        << "var<workgroup> q_sumsq_partials: array<f32, " << kAttentionWorkgroupSize << ">;\n"
        << "var<workgroup> q_inv_rms: f32;\n";
  }

  auto& body = shader.MainFunctionBody();
  body << "  if (workgroup_idx >= uniforms.num_workgroups) { return; }\n"
       << "  let h = workgroup_idx % uniforms.num_heads;\n"
       << "  let row = workgroup_idx / uniforms.num_heads;\n"
       << "  let s = row % uniforms.sequence_length;\n"
       << "  let b = row / uniforms.sequence_length;\n";
  if (packed_qkv_) {
    body << "  let q_base = row * uniforms.packed_stride + h * uniforms.head_size;\n";
  } else {
    body << "  let q_base = row * uniforms.query_hidden_size + h * uniforms.head_size;\n";
  }
  body << "  let output_base = (row * uniforms.num_heads + h) * uniforms.head_size;\n";
  if (use_qk_norm_) {
    body << "  var q_sumsq = 0.0;\n"
         << "  for (var c = local_idx; c < uniforms.head_size; c += "
         << kAttentionWorkgroupSize << "u) {\n"
         << "    let qv = f32(" << query.GetByOffset("q_base + c") << ");\n"
         << "    q_sumsq += qv * qv;\n"
         << "  }\n"
         << "  q_sumsq_partials[local_idx] = q_sumsq;\n"
         << "  workgroupBarrier();\n"
         << "  for (var stride = " << (kAttentionWorkgroupSize / 2)
         << "u; stride > 0u; stride >>= 1u) {\n"
         << "    if (local_idx < stride) {\n"
         << "      q_sumsq_partials[local_idx] += q_sumsq_partials[local_idx + stride];\n"
         << "    }\n"
         << "    workgroupBarrier();\n"
         << "  }\n"
         << "  if (local_idx == 0u) {\n"
         << "    q_inv_rms = inverseSqrt(q_sumsq_partials[0] / f32(uniforms.head_size)"
            " + uniforms.qk_norm_epsilon);\n"
         << "  }\n"
         << "  workgroupBarrier();\n";
  } else {
    body << "  let q_inv_rms = 1.0;\n";
  }
  body << "  for (var d = local_idx; d < uniforms.head_size; d += "
       << kAttentionWorkgroupSize << "u) {\n"
       << "    var q_value = f32(" << query.GetByOffset("q_base + d") << ") * q_inv_rms;\n";
  if (use_qk_norm_) {
    body << "    q_value *= f32(" << q_norm_weight->GetByOffset("d") << ");\n";
  }
  if (do_rotary_) {
    body << "    if (d >= uniforms.rotary_offset && d < uniforms.rotary_offset + uniforms.rotary_dim) {\n"
         << "      let rotary_d = d - uniforms.rotary_offset;\n"
         << "      let half_dim = uniforms.rotary_dim / 2u;\n";
    if (rotary_interleaved_) {
      body << "      let pair_d = uniforms.rotary_offset + (rotary_d ^ 1u);\n"
           << "      let cache_d = rotary_d / 2u;\n"
           << "      let first = (rotary_d & 1u) == 0u;\n";
    } else {
      body << "      let first = rotary_d < half_dim;\n"
           << "      let pair_d = uniforms.rotary_offset"
              " + select(rotary_d - half_dim, rotary_d + half_dim, first);\n"
           << "      let cache_d = rotary_d % half_dim;\n";
    }
    body << "      var q_pair = f32(" << query.GetByOffset("q_base + pair_d") << ") * q_inv_rms;\n";
    if (use_qk_norm_) {
      body << "      q_pair *= f32(" << q_norm_weight->GetByOffset("pair_d") << ");\n";
    }
    if (has_position_ids_) {
      body << "      let position_words = " << position_ids->GetByOffset("row", true) << ";\n"
           << "      let position_valid = position_words.y == 0u"
              " && position_words.x < uniforms.rotary_max_position;\n"
           << "      let position = i32(position_words.x);\n";
    } else {
      body << "      let position = " << seqlens_k.GetByOffset("b")
           << " + 1i - i32(uniforms.sequence_length) + i32(s);\n"
           << "      let position_valid = position >= 0i"
              " && position < i32(uniforms.rotary_max_position);\n";
    }
    body << "      if (position_valid) {\n"
         << "        let cosine = f32("
         << cos_cache->GetByOffset("u32(position) * half_dim + cache_d") << ");\n"
         << "        let sine = f32("
         << sin_cache->GetByOffset("u32(position) * half_dim + cache_d") << ");\n"
         << "        q_value = q_value * cosine + select(q_pair * sine, -q_pair * sine, first);\n"
         << "      }\n"
         << "    }\n"
         << "    " << prepared_query.SetByOffset("output_base + d", "prepared_query_element_t(q_value)") << "\n";
  }
  if (!do_rotary_) {
    body << "    " << prepared_query.SetByOffset("output_base + d", "prepared_query_element_t(q_value)") << "\n";
  }
  body << "  }\n";
  return Status::OK();
}

Status DynamicSparseAttentionInitializeCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* past_key = nullptr;
  const ShaderVariableHelper* past_value = nullptr;
  if (initialize_key_ && has_past_key_) {
    past_key = &shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (initialize_value_ && has_past_value_) {
    past_value = &shader.AddInput("past_value", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* present_key = nullptr;
  const ShaderVariableHelper* present_value = nullptr;
  if (initialize_key_) {
    present_key = &shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  if (initialize_value_) {
    present_value = &shader.AddOutput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }

  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.dispatch_size");
  if (initialize_key_) {
    const std::string value = past_key == nullptr ? "0.0" : "f32(" + past_key->GetByOffset("global_idx") + ")";
    body << "  " << present_key->SetByOffset("global_idx", "present_key_element_t(" + value + ")") << "\n";
  }
  if (initialize_value_) {
    const std::string value = past_value == nullptr ? "0.0" : "f32(" + past_value->GetByOffset("global_idx") + ")";
    body << "  " << present_value->SetByOffset("global_idx", "present_value_element_t(" + value + ")") << "\n";
  }
  return Status::OK();
}

Status DynamicSparseAttentionAppendKvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* query = nullptr;
  const ShaderVariableHelper* key = nullptr;
  const ShaderVariableHelper* value = nullptr;
  if (packed_qkv_) {
    query = &shader.AddInput("query", ShaderUsage::UseUniform);
  } else {
    key = &shader.AddInput("key", ShaderUsage::UseUniform);
    value = &shader.AddInput("value", ShaderUsage::UseUniform);
  }
  const auto& seqlens_k = shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  const ShaderVariableHelper* k_norm_weight = nullptr;
  if (use_qk_norm_) {
    k_norm_weight = &shader.AddInput("k_norm_weight", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* cos_cache = nullptr;
  const ShaderVariableHelper* sin_cache = nullptr;
  if (do_rotary_) {
    cos_cache = &shader.AddInput("cos_cache", ShaderUsage::UseUniform);
    sin_cache = &shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* position_ids = nullptr;
  if (has_position_ids_) {
    position_ids = &shader.AddInput("position_ids", ShaderUsage::UseUniform);
  }
  const auto& present_key =
      shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_value =
      shader.AddOutput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  if (use_qk_norm_) {
    shader.AdditionalImplementation()
        << "var<workgroup> k_sumsq_partials: array<f32, " << kAttentionWorkgroupSize << ">;\n"
        << "var<workgroup> k_inv_rms: f32;\n";
  }
  shader.AdditionalImplementation() << "var<workgroup> destination_shared: i32;\n";

  auto& body = shader.MainFunctionBody();
  body << "  if (workgroup_idx >= uniforms.num_workgroups) { return; }\n"
       << "  let kv_head = workgroup_idx % uniforms.kv_num_heads;\n"
       << "  let row = workgroup_idx / uniforms.kv_num_heads;\n"
       << "  let s = row % uniforms.sequence_length;\n"
       << "  let b = row / uniforms.sequence_length;\n"
       << "  if (local_idx == 0u) {\n"
       << "    destination_shared = " << seqlens_k.GetByOffset("b")
       << " + 1i - i32(uniforms.sequence_length) + i32(s);\n"
       << "  }\n"
       << "  let destination = workgroupUniformLoad(&destination_shared);\n"
       << "  if (destination < 0i || destination >= i32(uniforms.cache_capacity)) { return; }\n";
  if (packed_qkv_) {
    body << "  let key_base = row * uniforms.packed_stride + uniforms.query_hidden_size"
         << " + kv_head * uniforms.head_size;\n"
         << "  let value_base = row * uniforms.packed_stride + uniforms.query_hidden_size"
         << " + uniforms.kv_hidden_size + kv_head * uniforms.head_size;\n";
  } else {
    body << "  let key_base = row * uniforms.kv_hidden_size + kv_head * uniforms.head_size;\n"
         << "  let value_base = key_base;\n";
  }
  if (use_qk_norm_) {
    body << "  var k_sumsq = 0.0;\n"
         << "  for (var c = local_idx; c < uniforms.head_size; c += "
         << kAttentionWorkgroupSize << "u) {\n"
         << "    let kv = f32("
         << (packed_qkv_ ? query->GetByOffset("key_base + c") : key->GetByOffset("key_base + c"))
         << ");\n"
         << "    k_sumsq += kv * kv;\n"
         << "  }\n"
         << "  k_sumsq_partials[local_idx] = k_sumsq;\n"
         << "  workgroupBarrier();\n"
         << "  for (var stride = " << (kAttentionWorkgroupSize / 2)
         << "u; stride > 0u; stride >>= 1u) {\n"
         << "    if (local_idx < stride) {\n"
         << "      k_sumsq_partials[local_idx] += k_sumsq_partials[local_idx + stride];\n"
         << "    }\n"
         << "    workgroupBarrier();\n"
         << "  }\n"
         << "  if (local_idx == 0u) {\n"
         << "    k_inv_rms = inverseSqrt(k_sumsq_partials[0] / f32(uniforms.head_size)"
            " + uniforms.qk_norm_epsilon);\n"
         << "  }\n"
         << "  workgroupBarrier();\n";
  } else {
    body << "  let k_inv_rms = 1.0;\n";
  }
  body << "  for (var d = local_idx; d < uniforms.head_size; d += "
       << kAttentionWorkgroupSize << "u) {\n"
       << "    var key_value = f32("
       << (packed_qkv_ ? query->GetByOffset("key_base + d") : key->GetByOffset("key_base + d"))
       << ") * k_inv_rms;\n";
  if (use_qk_norm_) {
    body << "    key_value *= f32(" << k_norm_weight->GetByOffset("d") << ");\n";
  }
  if (do_rotary_) {
    body << "    if (d >= uniforms.rotary_offset && d < uniforms.rotary_offset + uniforms.rotary_dim) {\n"
         << "      let rotary_d = d - uniforms.rotary_offset;\n"
         << "      let half_dim = uniforms.rotary_dim / 2u;\n";
    if (rotary_interleaved_) {
      body << "      let pair_d = uniforms.rotary_offset + (rotary_d ^ 1u);\n"
           << "      let cache_d = rotary_d / 2u;\n"
           << "      let first = (rotary_d & 1u) == 0u;\n";
    } else {
      body << "      let first = rotary_d < half_dim;\n"
           << "      let pair_d = uniforms.rotary_offset"
              " + select(rotary_d - half_dim, rotary_d + half_dim, first);\n"
           << "      let cache_d = rotary_d % half_dim;\n";
    }
    body << "      var key_pair = f32("
         << (packed_qkv_ ? query->GetByOffset("key_base + pair_d") : key->GetByOffset("key_base + pair_d"))
         << ") * k_inv_rms;\n";
    if (use_qk_norm_) {
      body << "      key_pair *= f32(" << k_norm_weight->GetByOffset("pair_d") << ");\n";
    }
    if (has_position_ids_) {
      body << "      let position_words = " << position_ids->GetByOffset("row", true) << ";\n"
           << "      let rotary_position_valid = position_words.y == 0u"
              " && position_words.x < uniforms.rotary_max_position;\n"
           << "      let rotary_position = i32(position_words.x);\n";
    } else {
      body << "      let rotary_position = destination;\n"
           << "      let rotary_position_valid = rotary_position >= 0i"
              " && rotary_position < i32(uniforms.rotary_max_position);\n";
    }
    body << "      if (rotary_position_valid) {\n"
         << "        let cosine = f32("
         << cos_cache->GetByOffset("u32(rotary_position) * half_dim + cache_d") << ");\n"
         << "        let sine = f32("
         << sin_cache->GetByOffset("u32(rotary_position) * half_dim + cache_d") << ");\n"
         << "        key_value = key_value * cosine + select(key_pair * sine, -key_pair * sine, first);\n"
         << "      }\n"
         << "    }\n";
  }
  const std::string value_expression =
      packed_qkv_ ? query->GetByOffset("value_base + d") : value->GetByOffset("value_base + d");
  body << "    let cache_offset = ((b * uniforms.kv_num_heads + kv_head) * uniforms.cache_capacity"
       << " + u32(destination)) * uniforms.head_size + d;\n"
       << "    " << present_key.SetByOffset("cache_offset", "present_key_element_t(key_value)") << "\n"
       << "    " << present_value.SetByOffset("cache_offset", value_expression) << "\n"
       << "  }\n";
  return Status::OK();
}

Status DynamicSparseAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& main_key = shader.AddInput("main_key", ShaderUsage::UseUniform);
  const auto& main_value = shader.AddInput("main_value", ShaderUsage::UseUniform);
  const ShaderVariableHelper* auxiliary_key = nullptr;
  const ShaderVariableHelper* auxiliary_value = nullptr;
  if (selected_from_auxiliary_ && has_selection_) {
    auxiliary_key = &shader.AddInput("auxiliary_key", ShaderUsage::UseUniform);
    if (has_auxiliary_value_) {
      auxiliary_value = &shader.AddInput("auxiliary_value", ShaderUsage::UseUniform);
    }
  }
  const ShaderVariableHelper* selected_indices = nullptr;
  const ShaderVariableHelper* selected_counts = nullptr;
  if (has_selection_) {
    selected_indices = &shader.AddInput("selected_indices", ShaderUsage::UseUniform);
    selected_counts = &shader.AddInput("selected_counts", ShaderUsage::UseUniform);
  }
  const auto& seqlens_k = shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  const ShaderVariableHelper* head_sink = nullptr;
  if (has_head_sink_) {
    head_sink = &shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "var<workgroup> dot_partials: array<f32, " << kAttentionWorkgroupSize << ">;\n"
      << "var<workgroup> total_length_shared: i32;\n";
  if (has_selection_) {
    shader.AdditionalImplementation() << "var<workgroup> selected_count_shared: u32;\n";
  }
  auto& body = shader.MainFunctionBody();
  body << "  if (workgroup_idx >= uniforms.num_workgroups) { return; }\n"
       << "  let head = workgroup_idx % uniforms.num_heads;\n"
       << "  let row = workgroup_idx / uniforms.num_heads;\n"
       << "  let s = row % uniforms.sequence_length;\n"
       << "  let b = row / uniforms.sequence_length;\n"
       << "  let kv_head = head / (uniforms.num_heads / uniforms.kv_num_heads);\n"
       << "  let q_base = (row * uniforms.num_heads + head) * uniforms.head_size;\n"
       << "  if (local_idx == 0u) {\n"
       << "    total_length_shared = " << seqlens_k.GetByOffset("b") << " + 1i;\n"
       << "  }\n"
       << "  let total_length = workgroupUniformLoad(&total_length_shared);\n"
       << "  let query_position = total_length - i32(uniforms.sequence_length) + i32(s);\n"
       << "  var max_logit = -3.402823466e+38;\n"
       << "  var denominator = 0.0;\n"
       << "  var accumulator: array<f32, " << kAttentionValuesPerInvocation << ">;\n";

  auto emit_candidate = [&](const ShaderVariableHelper& candidate_key,
                            const ShaderVariableHelper& candidate_value,
                            const std::string& candidate_valid,
                            const std::string& candidate_base) {
    body << "      {\n"
         << "        let candidate_valid = " << candidate_valid << ";\n"
         << "        let candidate_base = " << candidate_base << ";\n"
         << "        var dot_product = 0.0;\n"
         << "        for (var c = local_idx; c < uniforms.head_size; c += "
         << kAttentionWorkgroupSize << "u) {\n"
         << "          dot_product += f32(" << query.GetByOffset("q_base + c") << ") * f32("
         << candidate_key.GetByOffset("candidate_base + c") << ");\n"
         << "        }\n"
         << "        dot_partials[local_idx] = dot_product;\n"
         << "        workgroupBarrier();\n"
         << "        for (var stride = " << (kAttentionWorkgroupSize / 2)
         << "u; stride > 0u; stride >>= 1u) {\n"
         << "          if (local_idx < stride) {\n"
         << "            dot_partials[local_idx] += dot_partials[local_idx + stride];\n"
         << "          }\n"
         << "          workgroupBarrier();\n"
         << "        }\n"
         << "        let logit = dot_partials[0] * uniforms.scale;\n"
         << "        if (candidate_valid) {\n"
         << "          let new_max = max(max_logit, logit);\n"
         << "          let old_scale = select(exp(max_logit - new_max), 0.0, denominator == 0.0);\n"
         << "          let weight = exp(logit - new_max);\n"
         << "          for (var slot = 0u; slot < " << kAttentionValuesPerInvocation << "u; slot++) {\n"
         << "            let d = local_idx + slot * " << kAttentionWorkgroupSize << "u;\n"
         << "            if (d < uniforms.head_size) {\n"
         << "              accumulator[slot] = accumulator[slot] * old_scale + weight * f32("
         << candidate_value.GetByOffset("candidate_base + d") << ");\n"
         << "            }\n"
         << "          }\n"
         << "          denominator = denominator * old_scale + weight;\n"
         << "          max_logit = new_max;\n"
         << "        }\n"
         << "        workgroupBarrier();\n"
         << "      }\n";
  };

  if (local_plus_selected_) {
    body << "  let local_start = max(0i, query_position - i32(uniforms.local_window_size) + 1i);\n"
         << "  let local_end = min(query_position, min(total_length - 1i, i32(uniforms.cache_capacity) - 1i));\n"
         << "  for (var index = local_start; index <= local_end; index++) {\n"
         << "    let safe_index = u32(index);\n";
    emit_candidate(main_key, main_value,
                   "index >= 0i && index < i32(uniforms.cache_capacity)",
                   "((b * uniforms.kv_num_heads + kv_head) * uniforms.cache_capacity + safe_index)"
                   " * uniforms.head_size");
    body << "  }\n";
  }

  if (has_selection_) {
    body << "  if (local_idx == 0u) {\n"
         << "    let selected_count_i32 = " << selected_counts->GetByOffset("row") << ";\n"
         << "    selected_count_shared = u32(clamp(selected_count_i32, 0i, i32(uniforms.max_selected)));\n"
         << "  }\n"
         << "  let selected_count = workgroupUniformLoad(&selected_count_shared);\n"
         << "  for (var i = 0u; i < selected_count; i++) {\n"
         << "    let selected_index = "
         << selected_indices->GetByOffset("row * uniforms.max_selected + i") << ";\n";
    if (selected_from_auxiliary_) {
      body << "    let safe_index = u32(clamp(selected_index, 0i,"
              " i32(uniforms.auxiliary_sequence_length) - 1i));\n";
      emit_candidate(*auxiliary_key, has_auxiliary_value_ ? *auxiliary_value : *auxiliary_key,
                     "selected_index >= 0i && selected_index < i32(uniforms.auxiliary_sequence_length)",
                     "((b * uniforms.kv_num_heads + kv_head) * uniforms.auxiliary_sequence_length"
                     " + safe_index) * uniforms.head_size");
    } else {
      body << "    let safe_index = u32(clamp(selected_index, 0i, i32(uniforms.cache_capacity) - 1i));\n";
      emit_candidate(main_key, main_value,
                     "selected_index >= 0i && selected_index < total_length"
                     " && selected_index <= query_position"
                     " && selected_index < i32(uniforms.cache_capacity)",
                     "((b * uniforms.kv_num_heads + kv_head) * uniforms.cache_capacity"
                     " + safe_index) * uniforms.head_size");
    }
    body << "  }\n";
  }

  if (has_head_sink_ || use_smooth_softmax_) {
    body << "  {\n";
    if (has_head_sink_) {
      body << "    let sink_logit = f32(" << head_sink->GetByOffset("head") << ");\n";
    } else {
      body << "    let sink_logit = 0.0;\n";
    }
    body << "    let new_max = max(max_logit, sink_logit);\n"
         << "    let old_scale = select(exp(max_logit - new_max), 0.0, denominator == 0.0);\n"
         << "    denominator = denominator * old_scale + exp(sink_logit - new_max);\n"
         << "    for (var slot = 0u; slot < " << kAttentionValuesPerInvocation << "u; slot++) {\n"
         << "      accumulator[slot] *= old_scale;\n"
         << "    }\n"
         << "    max_logit = new_max;\n"
         << "  }\n";
  }
  body << "  for (var slot = 0u; slot < " << kAttentionValuesPerInvocation << "u; slot++) {\n"
       << "    let d = local_idx + slot * " << kAttentionWorkgroupSize << "u;\n"
       << "    if (d < uniforms.head_size) {\n"
       << "      let result = select(0.0, accumulator[slot] / denominator, denominator > 0.0);\n"
       << "      " << output.SetByOffset("q_base + d", "output_element_t(result)") << "\n"
       << "    }\n"
       << "  }\n";
  return Status::OK();
}

DynamicSparseAttention::DynamicSparseAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
  const int64_t num_heads = info.GetAttrOrDefault<int64_t>("num_heads", 0);
  const int64_t kv_num_heads = info.GetAttrOrDefault<int64_t>("kv_num_heads", 0);
  ORT_ENFORCE(num_heads > 0 && num_heads <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: num_heads must be a positive int.");
  ORT_ENFORCE(kv_num_heads > 0 && kv_num_heads <= std::numeric_limits<int>::max() &&
                  num_heads % kv_num_heads == 0,
              "DynamicSparseAttention: kv_num_heads must be positive and divide num_heads.");
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("is_causal", 1) == 1,
              "DynamicSparseAttention only supports is_causal=1.");

  const int64_t local_window_size = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  ORT_ENFORCE(local_window_size == -1 ||
                  (local_window_size > 0 && local_window_size <= std::numeric_limits<int>::max()),
              "DynamicSparseAttention: local_window_size must be -1 or a positive int.");
  local_window_size_ = static_cast<int>(local_window_size);
  const int64_t rotary_offset = info.GetAttrOrDefault<int64_t>("rotary_offset", 0);
  ORT_ENFORCE(rotary_offset >= 0 && rotary_offset <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: rotary_offset must be a nonnegative int.");
  rotary_offset_ = static_cast<int>(rotary_offset);
  scale_ = 0.0f;
  has_scale_ = info.GetAttr("scale", &scale_).IsOK();
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  do_rotary_ = ParseBoolAttribute(info, "do_rotary", 0);
  rotary_interleaved_ = ParseBoolAttribute(info, "rotary_interleaved", 0);
  use_smooth_softmax_ = ParseBoolAttribute(info, "smooth_softmax", 0);
  auxiliary_kv_shared_ = ParseBoolAttribute(info, "auxiliary_kv_shared", 0);
  attention_mode_ =
      ParseAttentionMode(info.GetAttrOrDefault<std::string>("attention_mode", "selected_only"));
  selected_kv_source_ =
      ParseKvSource(info.GetAttrOrDefault<std::string>("selected_kv_source", "main"));
}

Status DynamicSparseAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  ORT_RETURN_IF(context.InputCount() > 17 && context.Input<Tensor>(17) != nullptr,
                "DynamicSparseAttention: committed_length is only supported by CUDA.");
  Tensor* validation_status = context.OutputCount() > 3 ? context.Output(3, TensorShape({0})) : nullptr;
  ORT_RETURN_IF(validation_status != nullptr,
                "DynamicSparseAttention: validation_status is only supported by CUDA.");
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* past_key = context.Input<Tensor>(3);
  const Tensor* past_value = context.Input<Tensor>(4);
  ORT_RETURN_IF_NOT((past_key == nullptr || past_key->GetElementType() == query->GetElementType()) &&
                        (past_value == nullptr || past_value->GetElementType() == query->GetElementType()),
                    "DynamicSparseAttention: WebGPU cache types must match the query.");
  ORT_RETURN_IF_NOT((context.InputCount() <= 18 || context.Input<Tensor>(18) == nullptr) &&
                        (context.InputCount() <= 19 || context.Input<Tensor>(19) == nullptr),
                    "DynamicSparseAttention: INT8 cache scales are only supported by CUDA.");
  const Tensor* auxiliary_key = context.Input<Tensor>(5);
  const Tensor* auxiliary_value = context.Input<Tensor>(6);
  const Tensor* selected_indices = context.Input<Tensor>(7);
  const Tensor* selected_counts = context.Input<Tensor>(8);
  const Tensor* seqlens_k = context.Input<Tensor>(9);
  const Tensor* total_sequence_length = context.Input<Tensor>(10);
  const Tensor* cos_cache = context.Input<Tensor>(11);
  const Tensor* sin_cache = context.Input<Tensor>(12);
  const Tensor* position_ids = context.Input<Tensor>(13);
  const Tensor* q_norm_weight = context.Input<Tensor>(14);
  const Tensor* k_norm_weight = context.Input<Tensor>(15);
  const Tensor* head_sink = context.Input<Tensor>(16);

  DynamicSparseAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(dynamic_sparse_attention_helper::CheckInputs(
      query, key, value, past_key, past_value, auxiliary_key, auxiliary_value,
      selected_indices, selected_counts, seqlens_k, total_sequence_length,
      cos_cache, sin_cache, position_ids, q_norm_weight, k_norm_weight, head_sink,
      num_heads_, kv_num_heads_, local_window_size_, rotary_offset_, do_rotary_,
      auxiliary_kv_shared_, attention_mode_, selected_kv_source_, scale_, has_scale_,
      qk_norm_epsilon_, parameters));
  parameters.rotary_interleaved = rotary_interleaved_;
  parameters.use_smooth_softmax = use_smooth_softmax_ && head_sink == nullptr;

  if (parameters.batch_size > static_cast<int>(kMaxBatchSize)) {
    return NotImplementedBound("batch_size", parameters.batch_size, kMaxBatchSize);
  }
  if (parameters.sequence_length > static_cast<int>(kMaxSequenceLength)) {
    return NotImplementedBound("sequence_length", parameters.sequence_length, kMaxSequenceLength);
  }
  if (parameters.num_heads > static_cast<int>(kMaxNumHeads)) {
    return NotImplementedBound("num_heads", parameters.num_heads, kMaxNumHeads);
  }
  if (parameters.head_size > static_cast<int>(kMaxHeadSize)) {
    return NotImplementedBound("head_size", parameters.head_size, kMaxHeadSize);
  }
  if (parameters.cache_capacity > static_cast<int>(kMaxCacheLength)) {
    return NotImplementedBound("cache_capacity", parameters.cache_capacity, kMaxCacheLength);
  }
  if (parameters.auxiliary_sequence_length > static_cast<int>(kMaxCacheLength)) {
    return NotImplementedBound("auxiliary_sequence_length", parameters.auxiliary_sequence_length, kMaxCacheLength);
  }
  if (parameters.max_selected > static_cast<int>(kMaxSelected)) {
    return NotImplementedBound("max_selected", parameters.max_selected, kMaxSelected);
  }
  if (parameters.local_window_size > static_cast<int>(kMaxLocalWindow)) {
    return NotImplementedBound("local_window_size", parameters.local_window_size, kMaxLocalWindow);
  }

  const bool selected_from_auxiliary =
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary;
  const bool local_plus_selected =
      parameters.attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected;
  const bool has_selection =
      parameters.max_selected > 0 && (!selected_from_auxiliary || parameters.auxiliary_sequence_length > 0);
  const bool has_auxiliary_value = auxiliary_value != nullptr;
  const bool use_smooth_softmax = use_smooth_softmax_ && head_sink == nullptr;
  const uint32_t required_storage_buffers =
      5u +                         // prepared Q, main K/V, seqlens, and output
      (has_selection ? 2u : 0u) +  // selected indices and counts
      (selected_from_auxiliary && has_selection
           ? 1u + static_cast<uint32_t>(has_auxiliary_value)
           : 0u) +  // auxiliary K and optional distinct V
      static_cast<uint32_t>(head_sink != nullptr);
  if (required_storage_buffers > context.DeviceLimits().maxStorageBuffersPerShaderStage) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "DynamicSparseAttention (WebGPU): attention configuration requires ",
        required_storage_buffers, " storage buffers, but the device supports ",
        context.DeviceLimits().maxStorageBuffersPerShaderStage,
        " per shader stage. Use shared auxiliary K/V, omit head_sink, or use an adapter with a higher limit.");
  }

  const TensorShape output_shape(
      {parameters.batch_size, parameters.sequence_length, parameters.query_hidden_size});
  const TensorShape cache_shape(
      {parameters.batch_size, parameters.kv_num_heads, parameters.cache_capacity, parameters.head_size});
  const uint64_t query_elements_64 = static_cast<uint64_t>(output_shape.Size());
  const uint64_t cache_elements_64 = static_cast<uint64_t>(cache_shape.Size());
  const uint64_t max_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
  const uint64_t element_size = query->DataType()->Size();
  if (query_elements_64 > std::numeric_limits<uint32_t>::max() ||
      query_elements_64 * element_size > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): output tensor exceeds WebGPU storage bounds.");
  }
  if (static_cast<uint64_t>(query->Shape().Size()) * element_size > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): query tensor exceeds WebGPU storage bounds.");
  }
  if (cache_elements_64 > std::numeric_limits<uint32_t>::max() ||
      cache_elements_64 * element_size > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): main cache tensor exceeds WebGPU storage bounds.");
  }
  if (static_cast<uint64_t>(selected_indices->Shape().Size()) * sizeof(int32_t) > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): selection metadata exceeds WebGPU storage bounds.");
  }
  if (auxiliary_key != nullptr &&
      static_cast<uint64_t>(auxiliary_key->Shape().Size()) * element_size > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): auxiliary cache exceeds WebGPU storage bounds.");
  }
  if (position_ids != nullptr &&
      static_cast<uint64_t>(position_ids->Shape().Size()) * sizeof(int64_t) > max_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DynamicSparseAttention (WebGPU): position_ids exceeds WebGPU storage bounds.");
  }

  Tensor* output = context.Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "DynamicSparseAttention: output is required.");
  Tensor* present_key_output = context.OutputCount() > 1 ? context.Output(1, cache_shape) : nullptr;
  Tensor* present_value_output = context.OutputCount() > 2 ? context.Output(2, cache_shape) : nullptr;
  Tensor internal_present_key;
  Tensor internal_present_value;
  if (present_key_output == nullptr) {
    internal_present_key = context.CreateGPUTensor(query->DataType(), cache_shape);
    present_key_output = &internal_present_key;
  }
  if (present_value_output == nullptr) {
    internal_present_value = context.CreateGPUTensor(query->DataType(), cache_shape);
    present_value_output = &internal_present_value;
  }
  Tensor prepared_query = context.CreateGPUTensor(query->DataType(), output_shape);

  const bool past_key_aliases_present =
      past_key != nullptr && past_key->DataRaw() == present_key_output->DataRaw();
  const bool past_value_aliases_present =
      past_value != nullptr && past_value->DataRaw() == present_value_output->DataRaw();
  const uint32_t cache_elements = narrow<uint32_t>(cache_elements_64);
  const uint32_t query_workgroups = narrow<uint32_t>(
      static_cast<uint64_t>(parameters.batch_size) * parameters.sequence_length * parameters.num_heads);
  const bool has_position_ids = parameters.do_rotary && position_ids != nullptr;

  DynamicSparseAttentionPrepareQueryProgram prepare_query_program(
      parameters.is_packed_qkv, parameters.use_qk_norm, parameters.do_rotary,
      parameters.rotary_interleaved, has_position_ids);
  prepare_query_program
      .CacheHint(parameters.is_packed_qkv, parameters.use_qk_norm, parameters.do_rotary,
                 parameters.rotary_interleaved, has_position_ids)
      .AddInput({query, ProgramTensorMetadataDependency::TypeAndRank});
  prepare_query_program.AddInput({seqlens_k, ProgramTensorMetadataDependency::TypeAndRank});
  if (parameters.use_qk_norm) {
    prepare_query_program.AddInput({q_norm_weight, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (parameters.do_rotary) {
    prepare_query_program.AddInputs({
        {cos_cache, ProgramTensorMetadataDependency::TypeAndRank},
        {sin_cache, ProgramTensorMetadataDependency::TypeAndRank},
    });
  }
  if (has_position_ids) {
    prepare_query_program.AddInput({position_ids, ProgramTensorMetadataDependency::TypeAndRank});
  }
  prepare_query_program
      .AddOutput({&prepared_query, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({
          {narrow<uint32_t>(parameters.sequence_length)},
          {narrow<uint32_t>(parameters.num_heads)},
          {narrow<uint32_t>(parameters.head_size)},
          {narrow<uint32_t>(parameters.query_hidden_size)},
          {narrow<uint32_t>(parameters.query_hidden_size + 2 * parameters.kv_hidden_size)},
          {narrow<uint32_t>(parameters.rotary_dim)},
          {narrow<uint32_t>(parameters.rotary_offset)},
          {narrow<uint32_t>(parameters.rotary_max_position)},
          {parameters.qk_norm_epsilon},
          {query_workgroups},
      })
      .SetDispatchGroupSize(query_workgroups)
      .SetWorkgroupSize(kAttentionWorkgroupSize);
  ORT_RETURN_IF_ERROR(context.RunProgram(prepare_query_program));

  const bool initialize_key = !past_key_aliases_present;
  const bool initialize_value = !past_value_aliases_present;
  if (initialize_key || initialize_value) {
    DynamicSparseAttentionInitializeCacheProgram initialize_cache_program(
        initialize_key, initialize_value, past_key != nullptr, past_value != nullptr);
    initialize_cache_program.CacheHint(
        initialize_key, initialize_value, past_key != nullptr, past_value != nullptr);
    if (initialize_key && past_key != nullptr) {
      initialize_cache_program.AddInput({past_key, ProgramTensorMetadataDependency::TypeAndRank});
    }
    if (initialize_value && past_value != nullptr) {
      initialize_cache_program.AddInput({past_value, ProgramTensorMetadataDependency::TypeAndRank});
    }
    if (initialize_key) {
      initialize_cache_program.AddOutput(
          {present_key_output, ProgramTensorMetadataDependency::TypeAndRank});
    }
    if (initialize_value) {
      initialize_cache_program.AddOutput(
          {present_value_output, ProgramTensorMetadataDependency::TypeAndRank});
    }
    initialize_cache_program.AddUniformVariable({cache_elements})
        .SetDispatchGroupSize((cache_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    ORT_RETURN_IF_ERROR(context.RunProgram(initialize_cache_program));
  }

  const uint64_t append_workgroups_64 =
      static_cast<uint64_t>(parameters.batch_size) * parameters.sequence_length * parameters.kv_num_heads;
  ORT_RETURN_IF_NOT(append_workgroups_64 <= std::numeric_limits<uint32_t>::max(),
                    "DynamicSparseAttention (WebGPU): KV append dispatch exceeds WebGPU bounds.");
  const uint32_t append_workgroups = narrow<uint32_t>(append_workgroups_64);
  DynamicSparseAttentionAppendKvProgram append_kv_program(
      parameters.is_packed_qkv, parameters.use_qk_norm, parameters.do_rotary,
      parameters.rotary_interleaved, has_position_ids);
  append_kv_program.CacheHint(
      parameters.is_packed_qkv, parameters.use_qk_norm, parameters.do_rotary,
      parameters.rotary_interleaved, has_position_ids);
  if (parameters.is_packed_qkv) {
    append_kv_program.AddInput({query, ProgramTensorMetadataDependency::TypeAndRank});
  } else {
    append_kv_program.AddInputs({
        {key, ProgramTensorMetadataDependency::TypeAndRank},
        {value, ProgramTensorMetadataDependency::TypeAndRank},
    });
  }
  append_kv_program.AddInput({seqlens_k, ProgramTensorMetadataDependency::TypeAndRank});
  if (parameters.use_qk_norm) {
    append_kv_program.AddInput({k_norm_weight, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (parameters.do_rotary) {
    append_kv_program.AddInputs({
        {cos_cache, ProgramTensorMetadataDependency::TypeAndRank},
        {sin_cache, ProgramTensorMetadataDependency::TypeAndRank},
    });
  }
  if (has_position_ids) {
    append_kv_program.AddInput({position_ids, ProgramTensorMetadataDependency::TypeAndRank});
  }
  append_kv_program
      .AddOutputs({
          {present_key_output, ProgramTensorMetadataDependency::TypeAndRank},
          {present_value_output, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {narrow<uint32_t>(parameters.sequence_length)},
          {narrow<uint32_t>(parameters.kv_num_heads)},
          {narrow<uint32_t>(parameters.head_size)},
          {narrow<uint32_t>(parameters.query_hidden_size)},
          {narrow<uint32_t>(parameters.kv_hidden_size)},
          {narrow<uint32_t>(parameters.cache_capacity)},
          {narrow<uint32_t>(parameters.query_hidden_size + 2 * parameters.kv_hidden_size)},
          {narrow<uint32_t>(parameters.rotary_dim)},
          {narrow<uint32_t>(parameters.rotary_offset)},
          {narrow<uint32_t>(parameters.rotary_max_position)},
          {parameters.qk_norm_epsilon},
          {append_workgroups},
      })
      .SetDispatchGroupSize(append_workgroups)
      .SetWorkgroupSize(kAttentionWorkgroupSize);
  ORT_RETURN_IF_ERROR(context.RunProgram(append_kv_program));

  DynamicSparseAttentionProgram attention_program(
      has_selection, local_plus_selected, selected_from_auxiliary, has_auxiliary_value,
      head_sink != nullptr, use_smooth_softmax);
  attention_program.CacheHint(has_selection, local_plus_selected, selected_from_auxiliary, has_auxiliary_value,
                              head_sink != nullptr, use_smooth_softmax)
      .AddInputs({
          {&prepared_query, ProgramTensorMetadataDependency::TypeAndRank},
          {present_key_output, ProgramTensorMetadataDependency::TypeAndRank},
          {present_value_output, ProgramTensorMetadataDependency::TypeAndRank},
      });
  if (selected_from_auxiliary && has_selection) {
    attention_program.AddInput({auxiliary_key, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_auxiliary_value) {
      attention_program.AddInput({auxiliary_value, ProgramTensorMetadataDependency::TypeAndRank});
    }
  }
  if (has_selection) {
    attention_program.AddInputs({
        {selected_indices, ProgramTensorMetadataDependency::TypeAndRank},
        {selected_counts, ProgramTensorMetadataDependency::TypeAndRank},
    });
  }
  attention_program.AddInput({seqlens_k, ProgramTensorMetadataDependency::TypeAndRank});
  if (head_sink != nullptr) {
    attention_program.AddInput({head_sink, ProgramTensorMetadataDependency::TypeAndRank});
  }
  attention_program
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({
          {narrow<uint32_t>(parameters.sequence_length)},
          {narrow<uint32_t>(parameters.num_heads)},
          {narrow<uint32_t>(parameters.kv_num_heads)},
          {narrow<uint32_t>(parameters.head_size)},
          {narrow<uint32_t>(parameters.cache_capacity)},
          {narrow<uint32_t>(parameters.auxiliary_sequence_length)},
          {narrow<uint32_t>(parameters.max_selected)},
          {narrow<uint32_t>(std::max(parameters.local_window_size, 0))},
          {parameters.scale},
          {query_workgroups},
      })
      .SetDispatchGroupSize(query_workgroups)
      .SetWorkgroupSize(kAttentionWorkgroupSize);
  return context.RunProgram(attention_program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
