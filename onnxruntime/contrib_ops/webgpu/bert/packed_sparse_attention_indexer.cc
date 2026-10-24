// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// WebGPU implementation of com.microsoft.PackedSparseAttentionIndexer. Like its dense
// SparseAttentionIndexer counterpart, every program below is single-invocation-per-row
// (`if (... || local_idx != 0u) { return; }`) and recomputes reductions on demand rather than
// staging per-row intermediates in workgroup memory; this keeps every kernel correct without
// requiring WGSL arrays sized by a runtime (uniform) head_size, which WGSL does not support for
// function-local variables. See docs/contrib_ops/webgpu/packed_sparse_attention_indexer.md and
// packed_sparse_attention_indexer_impl.cu (the CUDA implementation) for the full operator
// contract and the device-side metadata-safety argument, which apply unchanged here: every
// per-request quantity is read directly from device buffers inside the shader (never on the
// host), and values are clamped into the fixed-capacity range before use so malformed metadata
// can never cause an out-of-bounds access.

#include "contrib_ops/webgpu/bert/packed_sparse_attention_indexer.h"

#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace psai = onnxruntime::contrib::packed_sparse_attention_indexer;

ONNX_OPERATOR_KERNEL_EX(
    PackedSparseAttentionIndexer,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T_KEY", WebGpuSupportedFloatTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())
        .MayInplace(psai::kPastKeyState, psai::kPresentKeyState)
        .MayInplace(psai::kPastKvBuffer, psai::kPresentKvBuffer)
        .MayInplace(psai::kPastGateBuffer, psai::kPresentGateBuffer)
        .MayInplace(psai::kPastStateLengths, psai::kPresentStateLengths),
    PackedSparseAttentionIndexer);

namespace {

constexpr uint32_t kWorkgroupSize = 64;

uint32_t StorageBindingSegments(const Tensor* tensor, uint64_t max_binding_size) {
  if (tensor == nullptr) {
    return 0;
  }
  const uint64_t bytes = tensor->SizeInBytes();
  if (max_binding_size == 0 || bytes <= max_binding_size) {
    return 1;
  }
  return static_cast<uint32_t>((bytes + max_binding_size - 1) / max_binding_size);
}

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "PackedSparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "PackedSparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

Status CheckStorageBindingLimit(const onnxruntime::webgpu::ComputeContext& context,
                                const char* program_name,
                                std::initializer_list<const Tensor*> tensors) {
  const auto& limits = context.DeviceLimits();
  uint32_t binding_count = 0;
  for (const Tensor* tensor : tensors) {
    ORT_RETURN_IF(tensor == nullptr, "PackedSparseAttentionIndexer: missing tensor for ", program_name);
    binding_count += StorageBindingSegments(tensor, limits.maxStorageBufferBindingSize);
  }
  if (binding_count > limits.maxStorageBuffersPerShaderStage) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PackedSparseAttentionIndexer (WebGPU): ", program_name, " requires ", binding_count,
                           " storage-buffer bindings after segmentation, but this adapter supports only ",
                           limits.maxStorageBuffersPerShaderStage, " per shader stage.");
  }
  return Status::OK();
}

uint32_t ToUint32(int64_t value) { return onnxruntime::narrow<uint32_t>(value); }

struct RotaryCacheShape {
  bool batched;
  int64_t max_rotary_length;
  int64_t rotary_width;
};

Status CheckRotaryCache(const Tensor* cos_cache, const Tensor* sin_cache, int64_t batch_size,
                        RotaryCacheShape& out) {
  ORT_RETURN_IF(cos_cache == nullptr, "PackedSparseAttentionIndexer: cos_cache is required");
  const auto& cos_shape = cos_cache->Shape();
  out.batched = cos_shape.NumDimensions() == 3;
  ORT_RETURN_IF_NOT(
      (out.batched && cos_shape[0] == batch_size && cos_shape[1] > 0) ||
          (cos_shape.NumDimensions() == 2 && cos_shape[0] > 0),
      "PackedSparseAttentionIndexer: cos_cache must have shape (max_position, rotary_width) or "
      "(batch_size, max_position, rotary_width), got ",
      cos_shape.ToString());
  out.max_rotary_length = out.batched ? cos_shape[1] : cos_shape[0];
  out.rotary_width = out.batched ? cos_shape[2] : cos_shape[1];
  ORT_RETURN_IF_NOT(out.max_rotary_length > 0 && out.rotary_width > 0,
                    "PackedSparseAttentionIndexer: invalid cos_cache shape ", cos_shape.ToString());
  ORT_RETURN_IF_NOT(sin_cache != nullptr && sin_cache->Shape() == cos_shape,
                    "PackedSparseAttentionIndexer: sin_cache must have the same shape as cos_cache");
  return Status::OK();
}

}  // namespace

Status PackedSparseAttentionIndexerCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseUniform);
  const auto& dst = shader.AddOutput("dst", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  "
      << dst.SetByOffset("uniforms.dst_offset + global_idx",
                         "dst_element_t(" + src.GetByOffset("global_idx") + ")")
      << "\n";
  return Status::OK();
}

static Status PackTensors(onnxruntime::webgpu::ComputeContext& context,
                          std::initializer_list<const Tensor*> sources,
                          Tensor& packed) {
  uint32_t dst_offset = 0;
  for (const Tensor* source : sources) {
    const uint32_t total = ToUint32(source->Shape().Size());
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({source, ProgramTensorMetadataDependency::Type})
          .AddOutput({&packed, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize((total + kWorkgroupSize - 1) / kWorkgroupSize)
          .AddUniformVariables({{total}, {dst_offset}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
    dst_offset += total;
  }
  return Status::OK();
}

static Status PackTwoTensors(onnxruntime::webgpu::ComputeContext& context,
                             const Tensor& first,
                             const Tensor& second,
                             Tensor& packed) {
  return PackTensors(context, {&first, &second}, packed);
}

Status PackedSparseAttentionIndexerQsaUpdateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& rotary_cache = shader.AddInput("rotary_cache", ShaderUsage::UseUniform);
  const auto& sequence_metadata = shader.AddInput("sequence_metadata", ShaderUsage::UseUniform);
  const ShaderVariableHelper* capture_count = nullptr;
  const ShaderVariableHelper* state_update_active = nullptr;
  if (capture_state_update_) {
    capture_count = &shader.AddInput("state_update_capture_count", ShaderUsage::UseUniform);
    if (has_state_update_active_) {
      state_update_active = &shader.AddInput("state_update_active", ShaderUsage::UseUniform);
    }
  }
  const auto& present_key_state =
      shader.AddOutput("present_key_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_kv_buffer =
      shader.AddOutput("present_kv_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_state_lengths = shader.AddOutput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& overflow_flags = shader.AddOutput("overflow_flags", ShaderUsage::UseUniform);
  const ShaderVariableHelper* state_update = nullptr;
  if (capture_state_update_) {
    state_update = &shader.AddOutput(
        "state_update", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const ShaderVariableHelper& kv_buffer_history = present_kv_buffer;
  const ShaderVariableHelper& state_lengths_history = present_state_lengths;

  shader.AdditionalImplementation()
      << "fn extended_key(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * uniforms.head_size + d;\n"
      << "    return f32(" << kv_buffer_history.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * uniforms.key_row_stride + "
         "uniforms.key_offset + d;\n"
      << "  return f32(" << key.GetByOffset("idx2") << ");\n"
      << "}\n"
      << "fn pooled(b: u32, k: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  var sum = 0.0;\n"
      << "  for (var t = 0u; t < uniforms.compress_ratio; t++) {\n"
      << "    sum += extended_key(b, k * i32(uniforms.compress_ratio) + i32(t), old_buf_len, req_start, d);\n"
      << "  }\n"
      << "  return sum / f32(uniforms.compress_ratio);\n"
      << "}\n"
      << "fn normalized(b: u32, k: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var c = 0u; c < uniforms.head_size; c++) {\n"
      << "    let value = pooled(b, k, old_buf_len, req_start, c);\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  return pooled(b, k, old_buf_len, req_start, d) * "
         "inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon) * f32("
      << norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn rotated(b: u32, k: i32, old_buf_len: i32, req_start: i32, old_key_len: i32, d: u32) -> f32 {\n"
      << "  let value = normalized(b, k, old_buf_len, req_start, d);\n"
      << "  if (d >= uniforms.rotary_width) { return value; }\n"
      << "  let half = uniforms.rotary_width / 2u;\n"
      << "  let pair_d = select(d - half, d + half, d < half);\n"
      << "  let sign = select(1.0, -1.0, d < half);\n"
      << "  let paired = sign * normalized(b, k, old_buf_len, req_start, pair_d);\n"
      << "  let entry = old_key_len + k;\n"
      << "  let position = clamp_position(entry * i32(uniforms.compress_ratio));\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width + d;\n";
  } else {
    shader.AdditionalImplementation() << "  let cache = position * uniforms.rotary_width + d;\n";
  }
  shader.AdditionalImplementation()
      << "  let cache_size = select(1u, uniforms.batch_size, " << (cos_cache_batched_ ? "true" : "false")
      << ") * uniforms.max_rotary_length * uniforms.rotary_width;\n"
      << "  return value * f32(" << rotary_cache.GetByOffset("cache") << ") + paired * f32("
      << rotary_cache.GetByOffset("cache_size + cache") << ");\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size || local_idx != 0u) { return; }\n"
      << "  let req_start = " << sequence_metadata.GetByOffset("b") << ";\n"
      << "  let req_end = " << sequence_metadata.GetByOffset("b + 1u") << ";\n"
      << "  let raw_key_len = " << state_lengths_history.GetByOffset("b * 2u") << ";\n"
      << "  let raw_buf_len = " << state_lengths_history.GetByOffset("b * 2u + 1u") << ";\n"
      << "  let old_key_len = clamp(raw_key_len, 0, i32(uniforms.state_capacity));\n"
      << "  let old_buf_len = clamp(raw_buf_len, 0, i32(uniforms.compress_ratio) - 1);\n"
      << "  let past_len = " << sequence_metadata.GetByOffset("uniforms.batch_size + 1u + b") << ";\n"
      << "  let invalid = " << sequence_metadata.GetByOffset("0") << " != 0 || "
      << sequence_metadata.GetByOffset("uniforms.batch_size") << " != i32(uniforms.total_tokens) || "
      << "req_start < 0 || req_end < req_start || req_end > i32(uniforms.total_tokens) || past_len < 0 || "
      << "raw_key_len < 0 || raw_key_len > i32(uniforms.state_capacity) || raw_buf_len < 0 || "
      << "raw_buf_len >= i32(uniforms.compress_ratio) || "
      << "raw_key_len != past_len / i32(uniforms.compress_ratio) || "
      << "raw_buf_len != past_len % i32(uniforms.compress_ratio);\n"
      << "  if (invalid) {\n"
      << "    " << present_state_lengths.SetByOffset("b * 2u", "old_key_len") << "\n"
      << "    " << present_state_lengths.SetByOffset("b * 2u + 1u", "old_buf_len") << "\n"
      << "    " << overflow_flags.SetByOffset("b", "1") << "\n"
      << "    return;\n"
      << "  }\n"
      << "  let req_len = req_end - req_start;\n"
      << "  let pending = old_buf_len + req_len;\n"
      << "  let full_new_block_count = pending / i32(uniforms.compress_ratio);\n"
      << "  let capacity_left = max(i32(uniforms.state_capacity) - old_key_len, 0);\n"
      << "  let overflowed = full_new_block_count > capacity_left;\n"
      << "  " << overflow_flags.SetByOffset("b", "select(0, 1, overflowed)") << "\n"
      << "  if (overflowed) { return; }\n"
      << "  let new_block_count = full_new_block_count;\n"
      << "  let new_buf_len = pending % i32(uniforms.compress_ratio);\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u", "old_key_len + new_block_count") << "\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u + 1u", "new_buf_len") << "\n";
  if (capture_state_update_) {
    shader.MainFunctionBody()
        << "  let capture_is_active = "
        << (has_state_update_active_ ? state_update_active->GetByOffset("0") + " != 0" : "true") << ";\n"
        << "  let capture_tokens = select(0, min(max(" << capture_count->GetByOffset("b")
        << ", 0), min(req_len, i32(uniforms.state_update_capacity))), capture_is_active);\n";
  } else {
    shader.MainFunctionBody() << "  let capture_tokens = 0;\n";
  }
  shader.MainFunctionBody()
      << "  for (var k = 0; k < new_block_count; k++) {\n"
      << "    let entry = u32(old_key_len + k);\n"
      << "    let completion_token = (k + 1) * i32(uniforms.compress_ratio) - old_buf_len - 1;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      let value = rotated(b, k, old_buf_len, req_start, old_key_len, d);\n"
      << "      "
      << present_key_state.SetByOffset("(b * uniforms.state_capacity + entry) * uniforms.head_size + d",
                                       "present_key_state_element_t(value)")
      << "\n";
  if (capture_state_update_) {
    shader.MainFunctionBody()
        << "      if (completion_token < capture_tokens) {\n"
        << "        "
        << state_update->SetByOffset(
               "(b * uniforms.state_update_capacity + u32(completion_token)) * uniforms.head_size + d",
               "state_update_element_t(value)")
        << "\n"
        << "      }\n";
  }
  shader.MainFunctionBody()
      << "    }\n"
      << "  }\n"
      << "  for (var t = 0; t < capture_tokens; t++) {\n"
      << "    if ((old_buf_len + t + 1) % i32(uniforms.compress_ratio) != 0) {\n"
      << "      for (var d = 0u; d < uniforms.head_size; d++) {\n";
  if (capture_state_update_) {
    shader.MainFunctionBody()
        << "        let value = "
        << key.GetByOffset("u32(req_start + t) * uniforms.key_row_stride + uniforms.key_offset + d") << ";\n"
        << "        "
        << state_update->SetByOffset(
               "(b * uniforms.state_update_capacity + u32(t)) * uniforms.head_size + d",
               "state_update_element_t(value)")
        << "\n";
  }
  shader.MainFunctionBody()
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "  for (var t = 0; t < new_buf_len; t++) {\n"
      << "    let virtual_pos = new_block_count * i32(uniforms.compress_ratio) + t;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      let value = extended_key(b, virtual_pos, old_buf_len, req_start, d);\n"
      << "      "
      << present_kv_buffer.SetByOffset("(b * uniforms.buffer_capacity + u32(t)) * uniforms.head_size + d",
                                       "present_kv_buffer_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerQsaCaptureProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cumulative_sequence_lengths", ShaderUsage::UseUniform);
  const auto& past_seqlens = shader.AddInput("past_sequence_lengths", ShaderUsage::UseUniform);
  const auto& capture_count = shader.AddInput("state_update_capture_count", ShaderUsage::UseUniform);
  const auto& present_key_state =
      shader.AddInput("present_key_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& overflow_flags = shader.AddInput("overflow_flags", ShaderUsage::UseUniform);
  const ShaderVariableHelper* state_update_active = nullptr;
  if (has_state_update_active_) {
    state_update_active = &shader.AddInput("state_update_active", ShaderUsage::UseUniform);
  }
  const auto& state_update =
      shader.AddOutput("state_update", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size || local_idx != 0u || " << overflow_flags.GetByOffset("b")
      << " != 0) { return; }\n";
  if (has_state_update_active_) {
    shader.MainFunctionBody()
        << "  if (" << state_update_active->GetByOffset("0") << " == 0) { return; }\n";
  }
  shader.MainFunctionBody()
      << "  let req_start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let req_end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  let req_len = max(req_end - req_start, 0);\n"
      << "  let past_len = max(" << past_seqlens.GetByOffset("b") << ", 0);\n"
      << "  let old_key_len = past_len / i32(uniforms.compress_ratio);\n"
      << "  let old_buf_len = past_len % i32(uniforms.compress_ratio);\n"
      << "  let capture_tokens = min(max(" << capture_count.GetByOffset("b")
      << ", 0), min(req_len, i32(uniforms.state_update_capacity)));\n"
      << "  for (var t = 0; t < capture_tokens; t++) {\n"
      << "    let completes_block = (old_buf_len + t + 1) % i32(uniforms.compress_ratio) == 0;\n"
      << "    let completed_entry = old_key_len + (old_buf_len + t + 1) / i32(uniforms.compress_ratio) - 1;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      var value = f32("
      << key.GetByOffset("u32(req_start + t) * uniforms.key_row_stride + uniforms.key_offset + d") << ");\n"
      << "      if (completes_block && completed_entry >= 0 && completed_entry < i32(uniforms.state_capacity)) {\n"
      << "        value = f32("
      << present_key_state.GetByOffset(
             "(b * uniforms.state_capacity + u32(completed_entry)) * uniforms.head_size + d")
      << ");\n"
      << "      }\n"
      << "      "
      << state_update.SetByOffset(
             "(b * uniforms.state_update_capacity + u32(t)) * uniforms.head_size + d",
             "state_update_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerQsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query_metadata = shader.AddInput("query_metadata", ShaderUsage::UseUniform);
  const auto& present_key_state = shader.AddInput("present_key_state", ShaderUsage::UseUniform);
  const auto& rotary_cache = shader.AddInput("rotary_cache", ShaderUsage::UseUniform);
  const auto& sequence_metadata = shader.AddInput("sequence_metadata", ShaderUsage::UseUniform);
  const auto& present_state_lengths = shader.AddInput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& overflow_flags = shader.AddInput("overflow_flags", ShaderUsage::UseUniform);
  const auto& selected_indices = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  const auto& selected_counts = shader.AddOutput("selected_counts", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn batch_of_token(token: u32) -> u32 {\n"
      << "  var b = 0u;\n"
      << "  for (var i = 0u; i < uniforms.batch_size; i++) {\n"
      << "    if (u32(" << sequence_metadata.GetByOffset("i") << ") <= token) { b = i; } else { break; }\n"
      << "  }\n"
      << "  return b;\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn causal_count(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  let p = u32(position);\n"
      << "  let cr = uniforms.compress_ratio;\n"
      << "  return p / cr + select(0u, 1u, p % cr == cr - 1u);\n"
      << "}\n"
      << "fn abs_position(token: u32, b: u32) -> i32 {\n"
      << "  let req_start = " << sequence_metadata.GetByOffset("b") << ";\n"
      << "  return " << sequence_metadata.GetByOffset("uniforms.batch_size + 1u + b")
      << " + (i32(token) - req_start);\n"
      << "}\n"
      << "fn normalized_query_value(token: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = token * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = f32(" << query_metadata.GetByOffset("base + k") << ");\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  return f32(" << query_metadata.GetByOffset("base + d")
      << ") * inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon) * f32("
      << query_metadata.GetByOffset("uniforms.query_norm_offset + d")
      << ");\n"
      << "}\n"
      << "fn query_value(token: u32, head: u32, d: u32, position: i32, b: u32) -> f32 {\n"
      << "  let base = token * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "  var value = normalized_query_value(token, head, d);\n"
      << "  if (d >= uniforms.rotary_width) { return value; }\n"
      << "  let half = uniforms.rotary_width / 2u;\n"
      << "  let pair_d = select(d - half, d + half, d < half);\n"
      << "  let sign = select(1.0, -1.0, d < half);\n"
      << "  let paired = sign * normalized_query_value(token, head, pair_d);\n"
      << "  let position_clamped = clamp_position(position);\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position_clamped) * uniforms.rotary_width + d;\n";
  } else {
    shader.AdditionalImplementation() << "  let cache = position_clamped * uniforms.rotary_width + d;\n";
  }
  shader.AdditionalImplementation()
      << "  let cache_size = select(1u, uniforms.batch_size, " << (cos_cache_batched_ ? "true" : "false")
      << ") * uniforms.max_rotary_length * uniforms.rotary_width;\n"
      << "  value = value * f32(" << rotary_cache.GetByOffset("cache") << ") + paired * f32("
      << rotary_cache.GetByOffset("cache_size + cache") << ");\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(token: u32, b: u32, block_index: u32, position: i32) -> f32 {\n"
      << "  let key_base = (b * uniforms.state_capacity + block_index) * uniforms.head_size;\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(token, head, d, position, b) * f32("
      << present_key_state.GetByOffset("key_base + d") << ");\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let token = workgroup_idx;\n"
      << "  if (token >= uniforms.total_tokens || local_idx != 0u) { return; }\n"
      << "  let output_base = token * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected_indices.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let b = batch_of_token(token);\n"
      << "  if (" << overflow_flags.GetByOffset("b") << " != 0) {\n"
      << "    " << selected_counts.SetByOffset("token", "0") << "\n"
      << "    return;\n"
      << "  }\n"
      << "  let key_len_after = u32(" << present_state_lengths.GetByOffset("b * 2u") << ");\n"
      << "  let position = abs_position(token, b);\n"
      << "  let causal = causal_count(position);\n"
      << "  let visible_block_count = min(key_len_after, causal);\n"
      << "  let selected = min(uniforms.block_topk, visible_block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  var emitted_blocks = 0u;\n"
      << "  for (var rank = 0u; rank < selected; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < visible_block_count; candidate++) {\n"
      << "      let score = block_score(token, b, candidate, position);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < "
         "best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    for (var t = 0u; t < uniforms.compress_ratio; t++) {\n"
      << "      "
      << selected_indices.SetByOffset("output_base + rank * uniforms.compress_ratio + t",
                                      "best_index * i32(uniforms.compress_ratio) + i32(t)")
      << "\n"
      << "    }\n"
      << "    emitted_blocks = rank + 1u;\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n"
      << "  let block_start = i32(visible_block_count * uniforms.compress_ratio);\n"
      << "  let natural_tail = select(0, position - block_start + 1, position >= block_start);\n"
      << "  let remaining_capacity = i32(uniforms.capacity) - i32(emitted_blocks * uniforms.compress_ratio);\n"
      << "  var tail_count = min(natural_tail, remaining_capacity);\n"
      << "  tail_count = max(tail_count, 0);\n"
      << "  for (var t = 0; t < tail_count; t++) {\n"
      << "    "
      << selected_indices.SetByOffset("output_base + emitted_blocks * uniforms.compress_ratio + u32(t)",
                                      "block_start + t")
      << "\n"
      << "  }\n"
      << "  "
      << selected_counts.SetByOffset("token", "i32(emitted_blocks * uniforms.compress_ratio) + tail_count")
      << "\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerCsaUpdateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key_gate = shader.AddInput("key_gate", ShaderUsage::UseUniform);
  const auto& norm_bias = shader.AddInput("norm_bias", ShaderUsage::UseUniform);
  const auto& rotary_cache = shader.AddInput("rotary_cache", ShaderUsage::UseUniform);
  const auto& sequence_metadata = shader.AddInput("sequence_metadata", ShaderUsage::UseUniform);
  const auto& present_key_state =
      shader.AddOutput("present_key_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_kv_buffer =
      shader.AddOutput("present_kv_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_gate_buffer =
      shader.AddOutput("present_gate_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_state_lengths = shader.AddOutput("present_state_lengths", ShaderUsage::UseUniform);
  const ShaderVariableHelper& kv_buffer_history = present_kv_buffer;
  const ShaderVariableHelper& gate_buffer_history = present_gate_buffer;

  shader.AdditionalImplementation()
      << "fn extended_key(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, channel: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * width + channel;\n"
      << "    return f32(" << kv_buffer_history.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * width + channel;\n"
      << "  return f32(" << key_gate.GetByOffset("idx2") << ");\n"
      << "}\n"
      << "fn extended_gate(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, channel: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * width + channel;\n"
      << "    return f32(" << gate_buffer_history.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * width + channel;\n"
      << "  let gate_base = uniforms.total_tokens * 2u * uniforms.head_size;\n"
      << "  return f32(" << key_gate.GetByOffset("gate_base + idx2") << ");\n"
      << "}\n"
      << "fn pooled(b: u32, k: i32, old_buf_len: i32, req_start: i32, overlap_length: i32, d: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  let has_previous = k >= 1 || overlap_length >= i32(uniforms.compress_ratio);\n"
      << "  let previous_base = overlap_length + (k - 1) * i32(uniforms.compress_ratio);\n"
      << "  let current_base = overlap_length + k * i32(uniforms.compress_ratio);\n"
      << "  var max_gate = -3.4028234663852886e+38;\n"
      << "  if (has_previous) {\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let v = extended_gate(b, previous_base + i32(slot), old_buf_len, req_start, d) + f32("
      << norm_bias.GetByOffset("uniforms.head_size + slot * width + d") << ");\n"
      << "      max_gate = max(max_gate, v);\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let v = extended_gate(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d) + f32("
      << norm_bias.GetByOffset("uniforms.head_size + slot * width + uniforms.head_size + d") << ");\n"
      << "    max_gate = max(max_gate, v);\n"
      << "  }\n"
      << "  var denominator = 0.0;\n"
      << "  var accumulator = 0.0;\n"
      << "  if (has_previous) {\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let logit = extended_gate(b, previous_base + i32(slot), old_buf_len, req_start, d) + f32("
      << norm_bias.GetByOffset("uniforms.head_size + slot * width + d") << ");\n"
      << "      let weight = exp(logit - max_gate);\n"
      << "      denominator += weight;\n"
      << "      accumulator += weight * extended_key(b, previous_base + i32(slot), old_buf_len, req_start, d);\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let logit = extended_gate(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d) + f32("
      << norm_bias.GetByOffset("uniforms.head_size + slot * width + uniforms.head_size + d") << ");\n"
      << "    let weight = exp(logit - max_gate);\n"
      << "    denominator += weight;\n"
      << "    accumulator += weight * extended_key(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d);\n"
      << "  }\n"
      << "  return select(0.0, accumulator / denominator, denominator > 0.0);\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size || local_idx != 0u) { return; }\n"
      << "  let req_start = " << sequence_metadata.GetByOffset("b") << ";\n"
      << "  let req_end = " << sequence_metadata.GetByOffset("b + 1u") << ";\n"
      << "  let state_lengths_offset = 2u * uniforms.batch_size + 1u + b * 2u;\n"
      << "  let raw_key_len = " << sequence_metadata.GetByOffset("state_lengths_offset") << ";\n"
      << "  let raw_buf_len = " << sequence_metadata.GetByOffset("state_lengths_offset + 1u") << ";\n"
      << "  let old_key_len = clamp(raw_key_len, 0, i32(uniforms.state_capacity));\n"
      << "  let old_buf_len = clamp(raw_buf_len, 0, i32(uniforms.buffer_capacity));\n"
      << "  let invalid = " << sequence_metadata.GetByOffset("0") << " != 0 || "
      << sequence_metadata.GetByOffset("uniforms.batch_size") << " != i32(uniforms.total_tokens) || "
      << "req_start < 0 || req_end < req_start || req_end > i32(uniforms.total_tokens) || "
      << sequence_metadata.GetByOffset("uniforms.batch_size + 1u + b") << " < 0 || raw_key_len < 0 || "
      << "raw_key_len > i32(uniforms.state_capacity) || raw_buf_len < 0 || "
      << "raw_buf_len > i32(uniforms.buffer_capacity);\n"
      << "  if (invalid) {\n"
      << "    " << present_state_lengths.SetByOffset("b * 2u", "old_key_len") << "\n"
      << "    " << present_state_lengths.SetByOffset("b * 2u + 1u", "old_buf_len") << "\n"
      << "    return;\n"
      << "  }\n"
      << "  let req_len = req_end - req_start;\n"
      << "  let overlap_length = select(0, i32(uniforms.compress_ratio), old_buf_len >= "
         "i32(uniforms.compress_ratio));\n"
      << "  let leftover_length = old_buf_len - overlap_length;\n"
      << "  let pending = leftover_length + req_len;\n"
      << "  let full_new_window_count = pending / i32(uniforms.compress_ratio);\n"
      << "  let capacity_left = max(i32(uniforms.state_capacity) - old_key_len, 0);\n"
      << "  let overflowed = full_new_window_count > capacity_left;\n"
      << "  if (overflowed) { return; }\n"
      << "  let new_window_count = full_new_window_count;\n"
      << "  var present_buffer_length = 0;\n"
      << "  var present_buffer_start = 0;\n"
      << "  if (new_window_count > 0) {\n"
      << "    present_buffer_length = i32(uniforms.compress_ratio) + pending % i32(uniforms.compress_ratio);\n"
      << "    present_buffer_start = overlap_length + (new_window_count - 1) * i32(uniforms.compress_ratio);\n"
      << "  } else {\n"
      << "    present_buffer_length = old_buf_len + req_len;\n"
      << "    present_buffer_start = 0;\n"
      << "  }\n"
      << "  present_buffer_length = min(present_buffer_length, i32(uniforms.buffer_capacity));\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u", "old_key_len + new_window_count") << "\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u + 1u", "present_buffer_length") << "\n"
      << "  for (var k = 0; k < new_window_count; k++) {\n"
      << "    var square_sum = 0.0;\n"
      << "    for (var c = 0u; c < uniforms.head_size; c++) {\n"
      << "      let v = pooled(b, k, old_buf_len, req_start, overlap_length, c);\n"
      << "      square_sum += v * v;\n"
      << "    }\n"
      << "    let inverse_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "    let entry = u32(old_key_len + k);\n"
      << "    let position = clamp_position(i32(entry) * i32(uniforms.compress_ratio));\n";
  if (cos_cache_batched_) {
    shader.MainFunctionBody()
        << "    let cache_base = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width;\n";
  } else {
    shader.MainFunctionBody() << "    let cache_base = position * uniforms.rotary_width;\n";
  }
  shader.MainFunctionBody()
      << "    let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      var value = pooled(b, k, old_buf_len, req_start, overlap_length, d) * inverse_rms * f32("
      << norm_bias.GetByOffset("d") << ");\n"
      << "      if (d >= rotary_base) {\n"
      << "        let offset = d - rotary_base;\n"
      << "        let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "        let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "        let paired = sign * pooled(b, k, old_buf_len, req_start, overlap_length, pair_d) * "
         "inverse_rms * f32("
      << norm_bias.GetByOffset("pair_d") << ");\n"
      << "        let cache_size = select(1u, uniforms.batch_size, "
      << (cos_cache_batched_ ? "true" : "false")
      << ") * uniforms.max_rotary_length * uniforms.rotary_width;\n"
      << "        value = value * f32(" << rotary_cache.GetByOffset("cache_base + offset / 2u")
      << ") + paired * f32("
      << rotary_cache.GetByOffset("cache_size + cache_base + offset / 2u") << ");\n"
      << "      }\n"
      << "      "
      << present_key_state.SetByOffset("(b * uniforms.state_capacity + entry) * uniforms.head_size + d",
                                       "present_key_state_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n"
      << "  for (var t = 0; t < present_buffer_length; t++) {\n"
      << "    let virtual_pos = present_buffer_start + t;\n"
      << "    for (var c = 0u; c < 2u * uniforms.head_size; c++) {\n"
      << "      let key_value = extended_key(b, virtual_pos, old_buf_len, req_start, c);\n"
      << "      let gate_value = extended_gate(b, virtual_pos, old_buf_len, req_start, c);\n"
      << "      "
      << present_kv_buffer.SetByOffset("(b * uniforms.buffer_capacity + u32(t)) * (2u * uniforms.head_size) + c",
                                       "present_kv_buffer_element_t(key_value)")
      << "\n"
      << "      "
      << present_gate_buffer.SetByOffset(
             "(b * uniforms.buffer_capacity + u32(t)) * (2u * uniforms.head_size) + c",
             "present_gate_buffer_element_t(gate_value)")
      << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerCsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query_metadata = shader.AddInput("query_metadata", ShaderUsage::UseUniform);
  const auto& present_key_state = shader.AddInput("present_key_state", ShaderUsage::UseUniform);
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& rotary_cache = shader.AddInput("rotary_cache", ShaderUsage::UseUniform);
  const auto& sequence_metadata = shader.AddInput("sequence_metadata", ShaderUsage::UseUniform);
  const auto& present_state_lengths = shader.AddInput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& selected_indices = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  const auto& selected_counts = shader.AddOutput("selected_counts", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn batch_of_token(token: u32) -> u32 {\n"
      << "  var b = 0u;\n"
      << "  for (var i = 0u; i < uniforms.batch_size; i++) {\n"
      << "    if (u32(" << sequence_metadata.GetByOffset("i") << ") <= token) { b = i; } else { break; }\n"
      << "  }\n"
      << "  return b;\n"
      << "}\n"
      << "fn clamped_position(raw: vec2<u32>) -> u32 {\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u) { return 0xffffffffu; }\n"
      << "  return raw.x;\n"
      << "}\n"
      << "fn causal_count(raw: vec2<u32>) -> u32 {\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u) { return 0xffffffffu; }\n"
      << "  let p = raw.x;\n"
      << "  let cr = uniforms.compress_ratio;\n"
      << "  return p / cr + select(0u, 1u, p % cr == cr - 1u);\n"
      << "}\n"
      << "fn normalized_query_value(token: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = (token * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = f32(" << query_metadata.GetByOffset("base + k") << ");\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  let query_elements = uniforms.total_tokens * uniforms.num_heads * uniforms.head_size;\n"
      << "  return f32(" << query_metadata.GetByOffset("base + d")
      << ") * inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon) * f32("
      << query_metadata.GetByOffset("query_elements + d") << ");\n"
      << "}\n"
      << "fn query_value(token: u32, head: u32, d: u32, raw: vec2<u32>, b: u32) -> f32 {\n"
      << "  let base = (token * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = normalized_query_value(token, head, d);\n"
      << "  let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "  if (d < rotary_base) { return value; }\n"
      << "  let offset = d - rotary_base;\n"
      << "  let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "  let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "  let paired = sign * normalized_query_value(token, head, pair_d);\n"
      << "  let position = min(clamped_position(raw), uniforms.max_rotary_length - 1u);\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width + offset / 2u;\n";
  } else {
    shader.AdditionalImplementation()
        << "  let cache = position * uniforms.rotary_width + offset / 2u;\n";
  }
  shader.AdditionalImplementation()
      << "  let cache_size = select(1u, uniforms.batch_size, " << (cos_cache_batched_ ? "true" : "false")
      << ") * uniforms.max_rotary_length * uniforms.rotary_width;\n"
      << "  value = value * f32(" << rotary_cache.GetByOffset("cache") << ") + paired * f32("
      << rotary_cache.GetByOffset("cache_size + cache") << ");\n"
      << "  return value;\n"
      << "}\n"
      << "fn entry_score(token: u32, b: u32, entry: u32, raw: vec2<u32>) -> f32 {\n"
      << "  let key_base = (b * uniforms.state_capacity + entry) * uniforms.head_size;\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(token, head, d, raw, b) * f32("
      << present_key_state.GetByOffset("key_base + d") << ");\n"
      << "    }\n"
      << "    score += max(dot, 0.0) * f32("
      << query_metadata.GetByOffset(
             "uniforms.total_tokens * uniforms.num_heads * uniforms.head_size + uniforms.head_size + "
             "token * uniforms.num_heads + head")
      << ");\n"
      << "  }\n"
      << "  return score * uniforms.scale * uniforms.head_weight_scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let token = workgroup_idx;\n"
      << "  if (token >= uniforms.total_tokens || local_idx != 0u) { return; }\n"
      << "  let output_base = token * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected_indices.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let b = batch_of_token(token);\n"
      << "  let req_start = " << sequence_metadata.GetByOffset("b") << ";\n"
      << "  let req_end = " << sequence_metadata.GetByOffset("b + 1u") << ";\n"
      << "  let state_lengths_offset = 2u * uniforms.batch_size + 1u + b * 2u;\n"
      << "  let raw_key_len = " << sequence_metadata.GetByOffset("state_lengths_offset") << ";\n"
      << "  let raw_buf_len = " << sequence_metadata.GetByOffset("state_lengths_offset + 1u") << ";\n"
      << "  let old_key_len = clamp(raw_key_len, 0, i32(uniforms.state_capacity));\n"
      << "  let old_buf_len = clamp(raw_buf_len, 0, i32(uniforms.buffer_capacity));\n"
      << "  let invalid = " << sequence_metadata.GetByOffset("0") << " != 0 || "
      << sequence_metadata.GetByOffset("uniforms.batch_size") << " != i32(uniforms.total_tokens) || "
      << "req_start < 0 || req_end < req_start || req_end > i32(uniforms.total_tokens) || "
      << sequence_metadata.GetByOffset("uniforms.batch_size + 1u + b") << " < 0 || raw_key_len < 0 || "
      << "raw_key_len > i32(uniforms.state_capacity) || raw_buf_len < 0 || "
      << "raw_buf_len > i32(uniforms.buffer_capacity);\n"
      << "  let overlap_length = select(0, i32(uniforms.compress_ratio), old_buf_len >= "
         "i32(uniforms.compress_ratio));\n"
      << "  let pending = old_buf_len - overlap_length + req_end - req_start;\n"
      << "  let full_new_window_count = pending / i32(uniforms.compress_ratio);\n"
      << "  let overflowed = full_new_window_count > max(i32(uniforms.state_capacity) - old_key_len, 0);\n"
      << "  if (invalid || overflowed) {\n"
      << "    " << selected_counts.SetByOffset("token", "0") << "\n"
      << "    return;\n"
      << "  }\n"
      << "  let key_len_after = u32(" << present_state_lengths.GetByOffset("b * 2u") << ");\n"
      << "  let raw = " << position_ids.GetByOffset("token", true) << ";\n"
      << "  let threshold = min(causal_count(raw), key_len_after);\n"
      << "  let selected = min(uniforms.index_topk, threshold);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  var emitted = 0u;\n"
      << "  for (var rank = 0u; rank < selected; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < threshold; candidate++) {\n"
      << "      let score = entry_score(token, b, candidate, raw);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < "
         "best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    " << selected_indices.SetByOffset("output_base + rank", "best_index") << "\n"
      << "    emitted = rank + 1u;\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n"
      << "  " << selected_counts.SetByOffset("token", "i32(emitted)") << "\n";
  return Status::OK();
}

PackedSparseAttentionIndexer::PackedSparseAttentionIndexer(const OpKernelInfo& info) : WebGpuKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "PackedSparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(psai::TryParsePolicy(policy_mode, policy_), "PackedSparseAttentionIndexer: policy_mode must be '",
              psai::kPolicyModeQsa, "' or '", psai::kPolicyModeCsa, "', got '", policy_mode, "'");
  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "PackedSparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0 &&
                  compress_ratio_ <= (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2,
              "PackedSparseAttentionIndexer: compress_ratio must be positive and produce a generic buffer capacity "
              "no greater than INT_MAX");
  ORT_ENFORCE(info.GetAttr<int64_t>("state_capacity", &state_capacity_).IsOK() && state_capacity_ > 0,
              "PackedSparseAttentionIndexer: state_capacity must be a positive integer");
  state_update_capacity_ = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity_ >= 0 && state_update_capacity_ <= 8,
              "PackedSparseAttentionIndexer: state_update_capacity must be in [0, 8]");

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  has_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  has_head_weight_scale_ = info.GetAttr<float>("head_weight_scale", &head_weight_scale_).IsOK();
  if (policy_ == psai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget && token_budget_ > 0 && token_budget_ % compress_ratio_ == 0,
                "PackedSparseAttentionIndexer: token_budget must be > 0 and divisible by compress_ratio for qsa");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale_,
                "PackedSparseAttentionIndexer: csa attributes must be omitted for qsa");
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(state_update_capacity_ == 0,
                "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
    ORT_ENFORCE(has_index_topk && index_topk_ > 0, "PackedSparseAttentionIndexer: index_topk must be > 0 for csa");
    ORT_ENFORCE(!has_token_budget, "PackedSparseAttentionIndexer: token_budget must be omitted for csa");
    token_budget_ = 0;
  }
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "PackedSparseAttentionIndexer: epsilon must be >= 0");
}

Status PackedSparseAttentionIndexer::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  ORT_RETURN_IF(context.InputCount() > psai::kIndexShareMode && context.Input(psai::kIndexShareMode),
                "PackedSparseAttentionIndexer IndexShare requires the CUDA provider");
  for (int index : {psai::kMidKvBuffer, psai::kMidStateLengths, psai::kValidationStatus}) {
    Tensor* output = context.OutputCount() > index ? context.Output(index, TensorShape({0})) : nullptr;
    ORT_RETURN_IF(output != nullptr,
                  "PackedSparseAttentionIndexer: mid-state and validation outputs require the CUDA provider");
  }
  const Tensor* query = context.Input(psai::kQuery);
  const Tensor* key_state = context.Input(psai::kPastKeyState);
  ORT_RETURN_IF(query != nullptr && key_state != nullptr && query->DataType() != key_state->DataType(),
                "PackedSparseAttentionIndexer: mixed input/state types require the CUDA provider");
  ORT_RETURN_IF(
      context.OutputCount() > psai::kStateStatus && context.Output(psai::kStateStatus, TensorShape({0})) != nullptr,
      "PackedSparseAttentionIndexer state_status requires the CUDA provider");
  const bool is_qsa = policy_ == psai::Policy::kQsa;
  constexpr int kCsaOnlyInputs[] = {psai::kGate, psai::kPositionBias, psai::kHeadWeights};
  for (int index : kCsaOnlyInputs) {
    const bool provided = index < context.InputCount() && context.Input(index) != nullptr;
    ORT_RETURN_IF(provided != !is_qsa, "PackedSparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode 'qsa'" : " is required for policy_mode 'csa'");
  }
  const bool position_ids_provided =
      psai::kPositionIds < context.InputCount() && context.Input(psai::kPositionIds) != nullptr;
  ORT_RETURN_IF(position_ids_provided == is_qsa,
                "PackedSparseAttentionIndexer: position_ids is required for "
                "policy_mode 'csa' and must be omitted for policy_mode 'qsa'");
  const bool gate_buffer_provided =
      psai::kPastGateBuffer < context.InputCount() && context.Input(psai::kPastGateBuffer) != nullptr;
  ORT_RETURN_IF(gate_buffer_provided != !is_qsa, "PackedSparseAttentionIndexer: past_gate_buffer ",
                gate_buffer_provided ? "must be omitted for policy_mode 'qsa'"
                                     : "is required for policy_mode 'csa'");
  if (!is_qsa) {
    for (int index : {psai::kStateUpdateCaptureCount, psai::kStateUpdateActive}) {
      ORT_RETURN_IF(index < context.InputCount() && context.Input(index) != nullptr,
                    "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
    }
    Tensor* state_update =
        context.OutputCount() > psai::kStateUpdate ? context.Output(psai::kStateUpdate, TensorShape({0})) : nullptr;
    ORT_RETURN_IF(state_update != nullptr,
                  "PackedSparseAttentionIndexer: state update capture is only valid for policy_mode 'qsa'");
  }
  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

Status PackedSparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(psai::kQuery);
  const Tensor* key = context.Input(psai::kKey);
  const Tensor* query_norm = context.Input(psai::kQueryNormWeight);
  const Tensor* norm = context.Input(psai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(psai::kCosCache);
  const Tensor* sin_cache = context.Input(psai::kSinCache);
  const Tensor* cu_seqlens = context.Input(psai::kCumulativeSequenceLengths);
  const Tensor* past_seqlens = context.Input(psai::kPastSequenceLengths);
  const Tensor* past_key_state = context.Input(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context.Input(psai::kPastKvBuffer);
  const Tensor* past_state_lengths = context.Input(psai::kPastStateLengths);
  const Tensor* state_update_capture_count =
      context.InputCount() > psai::kStateUpdateCaptureCount ? context.Input(psai::kStateUpdateCaptureCount) : nullptr;
  const Tensor* state_update_active =
      context.InputCount() > psai::kStateUpdateActive ? context.Input(psai::kStateUpdateActive) : nullptr;

  ORT_RETURN_IF(state_update_capacity_ > 0 && state_update_capture_count == nullptr,
                "PackedSparseAttentionIndexer: state_update_capture_count is required when "
                "state_update_capacity is positive");

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2, "PackedSparseAttentionIndexer: query must have rank 2");
  const int64_t total_tokens = query_shape[0];
  ORT_RETURN_IF(query_norm == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1 && query_norm_shape[0] > 0 &&
                        (query_shape[1] - (key != nullptr ? 0 : query_norm_shape[0])) > 0 &&
                        (query_shape[1] - (key != nullptr ? 0 : query_norm_shape[0])) % query_norm_shape[0] == 0,
                    "PackedSparseAttentionIndexer: invalid flattened query dimensions");
  const int64_t head_size = query_norm_shape[0];
  const bool has_separate_key = key != nullptr;
  const int64_t query_width = query_shape[1] - (has_separate_key ? 0 : head_size);
  const int64_t num_heads = query_width / head_size;
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "PackedSparseAttentionIndexer: invalid query dimensions");

  ORT_RETURN_IF(cu_seqlens == nullptr, "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: invalid cumulative_sequence_lengths shape");
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_seqlens, "past_sequence_lengths", {batch_size}));
  if (has_separate_key) {
    ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, head_size}));
  }
  ORT_RETURN_IF_ERROR(CheckShape(query_norm, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && rotary.rotary_width % 2 == 0 && rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: invalid qsa rotary_width");

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: invalid past_key_state shape");
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));
  if (state_update_capture_count != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(state_update_capture_count, "state_update_capture_count", {batch_size}));
  }
  if (state_update_active != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(state_update_active, "state_update_active", {1}));
  }

  Tensor rotary_cache =
      context.CreateGPUTensor(cos_cache->DataType(), TensorShape({2 * cos_cache->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *cos_cache, *sin_cache, rotary_cache));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kQsa, token_budget_, index_topk_, compress_ratio_);
  Tensor* present_gate_buffer = context.Output(psai::kPresentGateBuffer, TensorShape({0}));
  ORT_RETURN_IF(present_gate_buffer != nullptr,
                "PackedSparseAttentionIndexer: output ", psai::kPresentGateBuffer,
                " must be omitted for policy_mode 'qsa'");
  Tensor* selected_indices = context.Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context.Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context.Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context.Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, head_size}));
  Tensor* present_state_lengths =
      context.Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  Tensor* state_update = context.OutputCount() > psai::kStateUpdate
                             ? context.Output(psai::kStateUpdate,
                                              TensorShape({batch_size, state_update_capacity_, head_size}))
                             : nullptr;
  if (state_update != nullptr) {
    context.FillZero(*state_update);
  }
  Tensor overflow_flags = context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), TensorShape({batch_size}));

  if (present_key_state->DataRaw() != past_key_state->DataRaw()) {
    const int64_t total = present_key_state->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_key_state, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_key_state, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}, {0u}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }
  if (present_kv_buffer->DataRaw() != past_kv_buffer->DataRaw()) {
    const int64_t total = present_kv_buffer->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_kv_buffer, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_kv_buffer, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}, {0u}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }
  if (present_state_lengths->DataRaw() != past_state_lengths->DataRaw()) {
    const int64_t total = present_state_lengths->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_state_lengths, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}, {0u}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }

  if (total_tokens == 0) {
    return Status::OK();
  }

  Tensor sequence_metadata = context.CreateGPUTensor(
      cu_seqlens->DataType(), TensorShape({cu_seqlens->Shape().Size() + past_seqlens->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *cu_seqlens, *past_seqlens, sequence_metadata));
  Tensor query_metadata = context.CreateGPUTensor(
      query->DataType(), TensorShape({query->Shape().Size() + query_norm->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *query, *query_norm, query_metadata));

  if (batch_size > 0) {
    const Tensor* key_source = has_separate_key ? key : query;
    ORT_RETURN_IF_ERROR(CheckStorageBindingLimit(
        context, "QSA update",
        {key_source, norm, &rotary_cache, &sequence_metadata, present_key_state, present_kv_buffer,
         present_state_lengths, &overflow_flags}));
    PackedSparseAttentionIndexerQsaUpdateProgram update{rotary.batched, false, false};
    update.CacheHint(rotary.batched, false, false)
        .SetWorkgroupSize(kWorkgroupSize)
        .AddInputs({{key_source, ProgramTensorMetadataDependency::Type},
                    {norm, ProgramTensorMetadataDependency::Type},
                    {&rotary_cache, ProgramTensorMetadataDependency::Type},
                    {&sequence_metadata, ProgramTensorMetadataDependency::Type}});
    update.AddOutputs({{present_key_state, ProgramTensorMetadataDependency::Type},
                       {present_kv_buffer, ProgramTensorMetadataDependency::Type}})
        .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
        .AddOutput({&overflow_flags, ProgramTensorMetadataDependency::Type});
    update
        .SetDispatchGroupSize(ToUint32(batch_size))
        .AddUniformVariables({{ToUint32(batch_size)},
                              {ToUint32(total_tokens)},
                              {ToUint32(has_separate_key ? head_size : query_shape[1])},
                              {ToUint32(has_separate_key ? 0 : query_width)},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(state_capacity)},
                              {ToUint32(buffer_capacity)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary.rotary_width)},
                              {ToUint32(rotary.max_rotary_length)},
                              {ToUint32(state_update_capacity_)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(update));

    if (state_update != nullptr && state_update_capacity_ > 0) {
      PackedSparseAttentionIndexerQsaCaptureProgram capture{state_update_active != nullptr};
      capture.CacheHint(state_update_active != nullptr)
          .SetWorkgroupSize(kWorkgroupSize)
          .AddInputs({{key_source, ProgramTensorMetadataDependency::Type},
                      {cu_seqlens, ProgramTensorMetadataDependency::Type},
                      {past_seqlens, ProgramTensorMetadataDependency::Type},
                      {state_update_capture_count, ProgramTensorMetadataDependency::Type},
                      {present_key_state, ProgramTensorMetadataDependency::Type},
                      {&overflow_flags, ProgramTensorMetadataDependency::Type}});
      if (state_update_active != nullptr) {
        capture.AddInput({state_update_active, ProgramTensorMetadataDependency::Type});
      }
      capture.AddOutput({state_update, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32(batch_size))
          .AddUniformVariables({{ToUint32(batch_size)},
                                {ToUint32(total_tokens)},
                                {ToUint32(has_separate_key ? head_size : query_shape[1])},
                                {ToUint32(has_separate_key ? 0 : query_width)},
                                {ToUint32(compress_ratio_)},
                                {ToUint32(state_capacity)},
                                {ToUint32(state_update_capacity_)},
                                {ToUint32(head_size)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(capture));
    }
  }

  ORT_RETURN_IF_ERROR(CheckStorageBindingLimit(
      context, "QSA selection",
      {&query_metadata, present_key_state, &rotary_cache, &sequence_metadata, present_state_lengths, &overflow_flags,
       selected_indices, selected_counts}));
  PackedSparseAttentionIndexerQsaSelectProgram select{rotary.batched};
  select.CacheHint(rotary.batched)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddInputs({{&query_metadata, ProgramTensorMetadataDependency::Type},
                  {present_key_state, ProgramTensorMetadataDependency::Type},
                  {&rotary_cache, ProgramTensorMetadataDependency::Type},
                  {&sequence_metadata, ProgramTensorMetadataDependency::Type},
                  {present_state_lengths, ProgramTensorMetadataDependency::Type},
                  {&overflow_flags, ProgramTensorMetadataDependency::Type}})
      .AddOutputs({{selected_indices, ProgramTensorMetadataDependency::Type},
                   {selected_counts, ProgramTensorMetadataDependency::Type}})
      .SetDispatchGroupSize(ToUint32(total_tokens))
      .AddUniformVariables({{ToUint32(total_tokens)},
                            {ToUint32(batch_size)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(query_shape[1])},
                            {ToUint32(query_shape.Size())},
                            {ToUint32(rotary.rotary_width)},
                            {ToUint32(rotary.max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(state_capacity)},
                            {ToUint32(capacity)},
                            {ToUint32(token_budget_ / compress_ratio_)},
                            {epsilon_},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
  return context.RunProgram(select);
}

Status PackedSparseAttentionIndexer::ComputeCsa(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(psai::kQuery);
  const Tensor* key = context.Input(psai::kKey);
  const Tensor* query_norm = context.Input(psai::kQueryNormWeight);
  const Tensor* norm = context.Input(psai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(psai::kCosCache);
  const Tensor* sin_cache = context.Input(psai::kSinCache);
  const Tensor* cu_seqlens = context.Input(psai::kCumulativeSequenceLengths);
  const Tensor* past_seqlens = context.Input(psai::kPastSequenceLengths);
  const Tensor* gate = context.Input(psai::kGate);
  const Tensor* position_bias = context.Input(psai::kPositionBias);
  const Tensor* head_weights = context.Input(psai::kHeadWeights);
  const Tensor* position_ids = context.Input(psai::kPositionIds);
  const Tensor* past_key_state = context.Input(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context.Input(psai::kPastKvBuffer);
  const Tensor* past_gate_buffer = context.Input(psai::kPastGateBuffer);
  const Tensor* past_state_lengths = context.Input(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2, "PackedSparseAttentionIndexer: query must have rank 2");
  const int64_t total_tokens = query_shape[0];
  ORT_RETURN_IF(query_norm == nullptr, "PackedSparseAttentionIndexer: query_norm_weight is required");
  const auto& query_norm_shape = query_norm->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1 && query_norm_shape[0] > 0 &&
                        query_shape[1] % query_norm_shape[0] == 0,
                    "PackedSparseAttentionIndexer: invalid flattened query dimensions");
  const int64_t head_size = query_norm_shape[0];
  const int64_t num_heads = query_shape[1] / head_size;
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "PackedSparseAttentionIndexer: invalid query dimensions");
  const int64_t width = 2 * head_size;

  ORT_RETURN_IF(cu_seqlens == nullptr, "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: invalid cumulative_sequence_lengths shape");
  const int64_t batch_size = cu_shape[0] - 1;
  ORT_RETURN_IF(batch_size == 0 && total_tokens != 0,
                "PackedSparseAttentionIndexer: total_tokens must be 0 when batch_size is 0");

  ORT_RETURN_IF_ERROR(CheckShape(past_seqlens, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(query_norm, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(position_bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {total_tokens, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && 2 * rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: invalid csa rotary_width");

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: invalid past_key_state shape");
  const int64_t state_capacity = key_state_shape[1];
  ORT_RETURN_IF_NOT(state_capacity == state_capacity_,
                    "PackedSparseAttentionIndexer: past_key_state capacity must match the state_capacity attribute");

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_gate_buffer, "past_gate_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kCsa, token_budget_, index_topk_, compress_ratio_);
  Tensor* selected_indices = context.Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context.Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context.Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context.Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_gate_buffer =
      context.Output(psai::kPresentGateBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_state_lengths =
      context.Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));
  auto copy_if_needed = [&](const Tensor* src, Tensor* dst) -> Status {
    if (src->DataRaw() == dst->DataRaw()) {
      return Status::OK();
    }
    const int64_t total = dst->Shape().Size();
    if (total == 0) {
      return Status::OK();
    }
    PackedSparseAttentionIndexerCopyProgram copy;
    copy.SetWorkgroupSize(kWorkgroupSize)
        .AddInput({src, ProgramTensorMetadataDependency::Type})
        .AddOutput({dst, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
        .AddUniformVariables({{ToUint32(total)}, {0u}});
    return context.RunProgram(copy);
  };
  ORT_RETURN_IF_ERROR(copy_if_needed(past_key_state, present_key_state));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_kv_buffer, present_kv_buffer));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_gate_buffer, present_gate_buffer));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_state_lengths, present_state_lengths));

  if (total_tokens == 0) {
    return Status::OK();
  }

  Tensor key_gate =
      context.CreateGPUTensor(key->DataType(), TensorShape({key->Shape().Size() + gate->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *key, *gate, key_gate));
  Tensor rotary_cache =
      context.CreateGPUTensor(cos_cache->DataType(), TensorShape({2 * cos_cache->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *cos_cache, *sin_cache, rotary_cache));
  Tensor sequence_metadata = context.CreateGPUTensor(
      cu_seqlens->DataType(),
      TensorShape({cu_seqlens->Shape().Size() + past_seqlens->Shape().Size() + past_state_lengths->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTensors(context, {cu_seqlens, past_seqlens, past_state_lengths}, sequence_metadata));
  Tensor norm_bias = context.CreateGPUTensor(
      norm->DataType(), TensorShape({norm->Shape().Size() + position_bias->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTwoTensors(context, *norm, *position_bias, norm_bias));
  Tensor query_metadata = context.CreateGPUTensor(
      query->DataType(),
      TensorShape({query->Shape().Size() + query_norm->Shape().Size() + head_weights->Shape().Size()}));
  ORT_RETURN_IF_ERROR(PackTensors(context, {query, query_norm, head_weights}, query_metadata));

  if (batch_size > 0) {
    ORT_RETURN_IF_ERROR(CheckStorageBindingLimit(
        context, "CSA update",
        {&key_gate, &norm_bias, &rotary_cache, &sequence_metadata, present_key_state, present_kv_buffer,
         present_gate_buffer, present_state_lengths}));
    PackedSparseAttentionIndexerCsaUpdateProgram update{rotary.batched};
    update.CacheHint(rotary.batched)
        .SetWorkgroupSize(kWorkgroupSize)
        .AddInputs({{&key_gate, ProgramTensorMetadataDependency::Type},
                    {&norm_bias, ProgramTensorMetadataDependency::Type},
                    {&rotary_cache, ProgramTensorMetadataDependency::Type},
                    {&sequence_metadata, ProgramTensorMetadataDependency::Type}});
    update.AddOutputs({{present_key_state, ProgramTensorMetadataDependency::Type},
                       {present_kv_buffer, ProgramTensorMetadataDependency::Type},
                       {present_gate_buffer, ProgramTensorMetadataDependency::Type}})
        .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(ToUint32(batch_size))
        .AddUniformVariables({{ToUint32(batch_size)},
                              {ToUint32(total_tokens)},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(state_capacity)},
                              {ToUint32(buffer_capacity)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary.rotary_width)},
                              {ToUint32(rotary.max_rotary_length)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(update));
  }

  ORT_RETURN_IF_ERROR(CheckStorageBindingLimit(
      context, "CSA selection",
      {&query_metadata, present_key_state, position_ids, &rotary_cache, &sequence_metadata, present_state_lengths,
       selected_indices, selected_counts}));
  PackedSparseAttentionIndexerCsaSelectProgram select{rotary.batched};
  select.CacheHint(rotary.batched)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddInputs({{&query_metadata, ProgramTensorMetadataDependency::Type},
                  {present_key_state, ProgramTensorMetadataDependency::Type},
                  {position_ids, ProgramTensorMetadataDependency::Type},
                  {&rotary_cache, ProgramTensorMetadataDependency::Type},
                  {&sequence_metadata, ProgramTensorMetadataDependency::Type},
                  {present_state_lengths, ProgramTensorMetadataDependency::Type}})
      .AddOutputs({{selected_indices, ProgramTensorMetadataDependency::Type},
                   {selected_counts, ProgramTensorMetadataDependency::Type}})
      .SetDispatchGroupSize(ToUint32(total_tokens))
      .AddUniformVariables({{ToUint32(total_tokens)},
                            {ToUint32(batch_size)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary.rotary_width)},
                            {ToUint32(rotary.max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(state_capacity)},
                            {ToUint32(buffer_capacity)},
                            {ToUint32(capacity)},
                            {ToUint32(index_topk_)},
                            {epsilon_},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))},
                            {has_head_weight_scale_ ? head_weight_scale_
                                                    : 1.0f / std::sqrt(static_cast<float>(num_heads))}});
  return context.RunProgram(select);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
