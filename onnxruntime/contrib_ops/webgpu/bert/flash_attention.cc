// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/kv_cache_block_quant_int8.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/bert/hadamard_transform.h"
#include "contrib_ops/webgpu/bert/kv_cache_quantization.h"
#include "contrib_ops/webgpu/bert/turbo_quant_hadamard.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// WGSL helper function for normalizing on-device indirect dispatch dims.
// Shared by CopyKVCacheProgram and SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram.
// Mirrors ProgramManager::NormalizeDispatchGroupSize three tiers:
//   1) direct (x, y, z) write when every dim is within the spec limit (65535);
//   2) 2D sqrt collapse when the product fits a square layout;
//   3) 3D cbrt collapse otherwise.
// Consumers are unaffected by the chosen layout: ShaderHelper flattens
// workgroup_id (x, y, z) into a single linear workgroup_idx.
// Caller contract: must register a storage output named exactly
// `indirect_buffer` of array<u32> with at least 3 elements.
constexpr const char kPopulateIndirectDispatchBufferFn[] = R"(
fn populate_indirect_dispatch_buffer(x: u32, y: u32, z: u32) {
  let limit = 65535u;  // WebGPU spec maxComputeWorkgroupsPerDimension
  if (x <= limit && y <= limit && z <= limit) {
    indirect_buffer[0] = x;
    indirect_buffer[1] = y;
    indirect_buffer[2] = z;
    return;
  }
  let size = f32(x) * f32(y) * f32(z);
  let dispatch_avg_2d = u32(ceil(sqrt(size)));
  if (dispatch_avg_2d <= limit) {
    indirect_buffer[0] = dispatch_avg_2d;
    indirect_buffer[1] = dispatch_avg_2d;
    indirect_buffer[2] = 1u;
    return;
  }
  let dispatch_avg_3d = u32(ceil(pow(size, 1.0 / 3.0)));
  indirect_buffer[0] = dispatch_avg_3d;
  indirect_buffer[1] = dispatch_avg_3d;
  indirect_buffer[2] = dispatch_avg_3d;
}
)";

constexpr int SelectDensePrefillMaxKStep(bool use_shm_path, bool is_fp16, int head_size) {
  if (!use_shm_path) {
    return 16;
  }

  // Preserve the existing tile selection, which targets the guaranteed WebGPU
  // workgroup-storage budget even when the device exposes a higher limit.
  const int element_size = is_fp16 ? 2 : 4;
  constexpr int kMinWorkgroupStorageBudgetBytes = 16384;
  const int max_k_from_shm = kMinWorkgroupStorageBudgetBytes / (2 * element_size * head_size);
  return max_k_from_shm >= 32 ? 32 : 16;
}

constexpr size_t DensePrefillWorkgroupStorageBytes(bool use_shm_path,
                                                   bool is_fp16,
                                                   int head_size,
                                                   uint32_t kv_cache_quantization_bits,
                                                   bool is_qualcomm,
                                                   uint32_t workgroup_size) {
  const size_t element_size = is_fp16 ? 2 : 4;
  const size_t max_k_step = SelectDensePrefillMaxKStep(use_shm_path, is_fp16, head_size);
  const size_t head_size_bytes = static_cast<size_t>(head_size) * element_size;
  const size_t kv_tiles = 2 * head_size_bytes * max_k_step;
  const size_t q4_lut = kv_cache_quantization_bits == 4 ? 16 * sizeof(float) : 0;
  const size_t qualcomm_output_tile = is_qualcomm ? head_size_bytes * workgroup_size / 2 : 0;
  return kv_tiles + q4_lut + qualcomm_output_tile;
}

constexpr bool DensePrefillFitsWorkgroupStorage(bool use_shm_path,
                                                bool is_fp16,
                                                int head_size,
                                                uint32_t kv_cache_quantization_bits,
                                                bool is_qualcomm,
                                                uint32_t workgroup_size,
                                                uint64_t max_workgroup_storage_size) {
  return DensePrefillWorkgroupStorageBytes(use_shm_path, is_fp16, head_size,
                                           kv_cache_quantization_bits, is_qualcomm,
                                           workgroup_size) <=
         max_workgroup_storage_size;
}

static_assert(!DensePrefillFitsWorkgroupStorage(true, false, 256, 8, false, 64, 16384));
static_assert(DensePrefillFitsWorkgroupStorage(true, false, 256, 8, false, 64, 32768));
static_assert(!DensePrefillFitsWorkgroupStorage(true, false, 128, 4, false, 64, 16384));
static_assert(DensePrefillFitsWorkgroupStorage(true, true, 128, 0, false, 64, 16384));
static_assert(DensePrefillFitsWorkgroupStorage(false, true, 128, 0, false, 64, 16384));
static_assert(!DensePrefillFitsWorkgroupStorage(true, true, 128, 8, true, 64, 16384));

constexpr size_t Q8QuantizationWorkgroupStorageBytes(int head_size) {
  return 2 * static_cast<size_t>(head_size) * sizeof(uint32_t) + 64 * sizeof(float);
}

static_assert(Q8QuantizationWorkgroupStorageBytes(4096) == 33024);

constexpr size_t DecodeWorkgroupStorageBytes(uint32_t m_tile,
                                             uint32_t tile_size,
                                             uint32_t head_size_vec,
                                             size_t element_size,
                                             uint32_t kv_cache_quantization_bits,
                                             bool use_paged_kv_cache) {
  const uint32_t tile_size_k_vec = m_tile == 1u ? 32u : 8u;
  const uint32_t workgroup_size = m_tile == 1u ? 128u : 64u;
  const size_t value_size = 4 * element_size;
  const bool quantized = kv_cache_quantization_bits != 0;

  const size_t q_tile = m_tile * (quantized ? head_size_vec : tile_size_k_vec) * value_size;
  const size_t kv_scales = quantized ? 2 * tile_size * sizeof(float) : 0;
  const size_t inner_qk = m_tile * tile_size * tile_size_k_vec * sizeof(float);
  const size_t tile_qk = m_tile * tile_size * sizeof(float);
  const size_t tile_output = m_tile * head_size_vec * value_size;
  const size_t qkv_values = m_tile * workgroup_size * value_size;
  const size_t tile_stats = 2 * m_tile * sizeof(float);
  const size_t q4_lut = kv_cache_quantization_bits == 4 ? 16 * sizeof(float) : 0;
  const size_t paged_row_offsets = use_paged_kv_cache && !quantized ? tile_size * sizeof(uint32_t) : 0;

  return q_tile + kv_scales + inner_qk + tile_qk + tile_output + qkv_values + tile_stats + q4_lut +
         paged_row_offsets;
}

constexpr uint32_t SelectDecodeMTile(uint32_t desired_m_tile,
                                     uint32_t tile_size,
                                     uint32_t head_size_vec,
                                     size_t element_size,
                                     uint32_t kv_cache_quantization_bits,
                                     bool use_paged_kv_cache,
                                     uint64_t max_workgroup_storage_size) {
  uint32_t m_tile = desired_m_tile;
  while (m_tile > 1u &&
         DecodeWorkgroupStorageBytes(m_tile, tile_size, head_size_vec, element_size,
                                     kv_cache_quantization_bits, use_paged_kv_cache) >
             max_workgroup_storage_size) {
    m_tile /= 2u;
  }
  return m_tile;
}

static_assert(SelectDecodeMTile(4, 64, 96 / 4, sizeof(float), 8, false, 16384) == 2);
static_assert(SelectDecodeMTile(4, 64, 128 / 4, sizeof(float), 8, false, 16384) == 2);
static_assert(SelectDecodeMTile(4, 64, 128 / 4, sizeof(MLFloat16), 0, false, 16384) == 4);
static_assert(SelectDecodeMTile(4, 64, 128 / 4, sizeof(float), 0, true, 16384) == 4);

FlashAttentionProgram::FlashAttentionProgram(const std::string& kernel_name,
                                             bool has_attention_bias,
                                             bool is_qualcomm,
                                             bool is_fp16,
                                             int qkv_head_size,
                                             int qkv_num_heads,
                                             bool is_unidirectional,
                                             bool is_nvidia,
                                             bool is_apple,
                                             bool has_subgroups,
                                             bool q_BNSH,
                                             bool use_seqlen_k,
                                             bool has_head_sink,
                                             bool has_local_window,
                                             uint32_t kv_cache_quantization_bits,
                                             int compressed_head_size_u32,
                                             bool use_seqlens_q)
    : Program{kernel_name},
      has_attention_bias_(has_attention_bias),
      is_qualcomm_(is_qualcomm),
      qkv_head_size_(qkv_head_size),
      qkv_num_heads_(qkv_num_heads),
      is_unidirectional_(is_unidirectional),
      is_nvidia_(is_nvidia),
      use_shm_path_(is_apple || is_nvidia || !has_subgroups),
      q_BNSH_(q_BNSH),
      use_seqlen_k_(use_seqlen_k),
      has_head_sink_(has_head_sink),
      has_local_window_(has_local_window),
      max_k_step_(SelectDensePrefillMaxKStep(use_shm_path_, is_fp16, qkv_head_size)),
      kv_cache_quantization_(kv_cache_quantization_bits != 0),
      kv_cache_quantization_bits_(kv_cache_quantization_bits),
      compressed_head_size_u32_(compressed_head_size_u32),
      use_seqlens_q_(use_seqlens_q) {
}

Status SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);
  if (prepare_indirect_dispatch_) {
    sh.AddInput("total_sequence_length_input", ShaderUsage::None);
  }

  const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
  const auto& present_key = sh.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = sh.AddOutput("present_value", ShaderUsage::UseUniform);

  if (prepare_indirect_dispatch_) {
    sh.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv_with_rotary_embedding_and_copykv.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                             WGSL_TEMPLATE_PARAMETER(multi_rotary_cache_concat_offset, multi_rotary_cache_concat_offset_),
                             WGSL_TEMPLATE_PARAMETER(prepare_indirect_dispatch, prepare_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_multi_rotary_cache_concat, multi_rotary_cache_concat_offset_ > 0),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(seqlens, seqlens),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

Status CopyKVCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(total_sequence_length)
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);
  const auto& copy_kv_shape = shader.AddIndices("copy_kv_shape");
  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }
  // If prepare_indirect_dispatch is enabled, add total_sequence_length_input
  // and indirect_buffer output. total_sequence_length_input is the global max
  // total sequence length across the batch (from GQA input #6); using it for
  // dispatch sizing covers right-padded batches where batch 0 is not the max.
  if (prepare_indirect_dispatch_) {
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.copy_size")
                            << "  let output_indices = " << copy_kv_shape.OffsetToIndices("global_idx") << ";\n"
                            << "  let head_size_id = output_indices[3];\n"
                               "  let sequence_id = output_indices[2];\n"
                               "  let num_head_id = output_indices[1];\n"
                               "  let batch = output_indices[0];\n";
  if (use_seqlen_k_) {
    shader.MainFunctionBody() << "  let raw_total_seq_length = u32(max(seqlen_k[batch], 0)) + 1u;\n"
                              << "  let total_seq_length = min(raw_total_seq_length, uniforms.present_sequence_length);\n";
  } else {
    shader.MainFunctionBody() << "  let total_seq_length = uniforms.total_sequence_length;\n";
  }
  // Right-padded batches with prompt shorter than kv_sequence_length would underflow u32; clamp to 0.
  shader.MainFunctionBody() << "  let past_sequence_length = select(total_seq_length - uniforms.kv_sequence_length, 0u, total_seq_length <= uniforms.kv_sequence_length);\n";

  // Add indirect dispatch logic for thread 0
  if (prepare_indirect_dispatch_) {
    shader.AdditionalImplementation() << kPopulateIndirectDispatchBufferFn;
    shader.MainFunctionBody() << "  if (global_idx == 0u) {\n"
                              << "    let raw_global_total_seq_length = total_sequence_length_input[0];\n"
                              << "    let global_total_seq_length = min(u32(max(raw_global_total_seq_length, 0)), uniforms.present_sequence_length);\n"
                              << "    let num_total_seq_length_tile = (global_total_seq_length + uniforms.tile_size - 1u) / uniforms.tile_size;\n"
                              << "    populate_indirect_dispatch_buffer(num_total_seq_length_tile, uniforms.num_heads * uniforms.num_q_tiles, uniforms.batch_size);\n"
                              << "  }\n\n";
  }

  shader.MainFunctionBody() << "  if (sequence_id >= total_seq_length) {\n"
                            << "    return;\n"
                            << "  }\n";
  if (past_present_share_buffer_) {
    shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, past_sequence_length + sequence_id, head_size_id)") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n";
  }

  if (has_past_) {
    const auto& past_key = shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
    shader.AddInput("past_value", ShaderUsage::UseUniform);
    shader.MainFunctionBody() << "if (sequence_id < past_sequence_length) {\n"
                              << "  let pastOffset = " << past_key.IndicesToOffset("past_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "past_key[pastOffset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "past_value[pastOffset]") << ";\n"
                              << "} else {\n"
                              << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id - past_sequence_length, head_size_id)" : "key_indices_t(batch, sequence_id - past_sequence_length, num_head_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n"
                              << "}";
  } else {
    shader.MainFunctionBody() << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
  }
  return Status::OK();
}

Status PrepareIndirectDispatchProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  shader.AddOutput("indirect_buffer", ShaderUsage::None);
  shader.AdditionalImplementation() << kPopulateIndirectDispatchBufferFn;
  shader.MainFunctionBody()
      << "  let raw_global_total_seq_length = total_sequence_length_input[0];\n"
      << "  let global_total_seq_length = min(u32(max(raw_global_total_seq_length, 0)), uniforms.present_sequence_length);\n"
      << "  let num_total_seq_length_tile = (global_total_seq_length + uniforms.tile_size - 1u) / uniforms.tile_size;\n"
      << "  populate_indirect_dispatch_buffer(num_total_seq_length_tile, uniforms.num_heads * uniforms.num_q_tiles, uniforms.batch_size);\n";
  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                   uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer, uint32_t num_q_tiles,
                   const Tensor* total_seqlen) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  // If indirect_buffer is provided, also prepare indirect dispatch buffer for flash attention.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  // has_past means non-static kv cache with valid past data
  bool has_past = !parameters.past_present_share_buffer_ && past_key != nullptr && past_value != nullptr && past_key->SizeInBytes() > 0;
  // parameters.total_sequence_length_ is past_sequence_length + kv_sequence_length.
  // parameters.kv_num_heads_ may be smaller than parameters.num_heads_ when parameters.is_gqa_ is true.
  int num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  // Only copy the new kv data for static kv cache
  int copy_sequence_length = parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  TensorShape copy_kv_shape{parameters.batch_size_, num_heads, copy_sequence_length, parameters.head_size_ / components};
  int64_t copy_size = copy_kv_shape.Size();

  // Determine if we need to prepare indirect dispatch
  bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  bool use_seqlen_k = (seqlen_k != nullptr);
  bool kv_BNSH = parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH || parameters.qkv_format_ == Q_K_V_BNSH;
  CopyKVCacheProgram program{"CopyKVCache", has_past, kv_BNSH, parameters.past_present_share_buffer_,
                             prepare_indirect_dispatch, use_seqlen_k};
  if (kv_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_ENFORCE(parameters.qkv_format_ == Q_K_V_BSNH, "qkv format ", parameters.qkv_format_, " is not supported yet in CopyKVCache.");
    // Reshape (batch_size, kv_sequence_length, kv_hidden_size) to (batch_size, kv_sequence_length, num_head, head_size)
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, num_heads, parameters.head_size_ / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }

  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (prepare_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }

  if (has_past) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  program.AddIndices(std::move(copy_kv_shape));
  program.SetDispatchGroupSize(static_cast<uint32_t>((copy_size + 63) / 64))
      .SetWorkgroupSize(64)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_, prepare_indirect_dispatch, use_seqlen_k)
      .AddUniformVariables({{static_cast<uint32_t>(copy_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(present_key->Shape()[2])},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {num_q_tiles}});

  return context.RunProgram(program);
}

Status FlashAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(new_sequence_length)(total_sequence_length)
  //
  //  Expectation is that present_key, and present_value contain past key and values since
  //  we are out of storage buffers a shader can have and both past/present cant be passed.
  // The hidden size of each q head should be a multiple of 4 because shader uses vectorized loads.
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (use_seqlens_q_) {
    shader.AddInput("seqlens_q", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(bit_width, kv_cache_quantization_bits_),
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(has_local_window, has_local_window_),
                             WGSL_TEMPLATE_PARAMETER(is_qualcomm, is_qualcomm_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(kv_cache_quantization, kv_cache_quantization_),
                             WGSL_TEMPLATE_PARAMETER(max_k_step_param, max_k_step_),
                             WGSL_TEMPLATE_PARAMETER(prefer_subgroupshuffle, !is_nvidia_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, qkv_head_size_),
                             WGSL_TEMPLATE_PARAMETER(qkv_num_heads, qkv_num_heads_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlens_q, use_seqlens_q_),
                             WGSL_TEMPLATE_PARAMETER(use_shm_path, use_shm_path_));
}

Status FlashAttentionPagedPrefillProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // q / key_cache / value_cache / output are addressed via getByOffset /
  // setByOffset so tensors larger than maxStorageBufferBindingSize (128 MiB
  // on most adapters) transparently work when the framework splits them
  // across bindings. block_table uses .getByIndices (2-D lookup).
  const auto& q = shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& key_cache = shader.AddInput("key_cache", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& value_cache = shader.AddInput("value_cache", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& block_table = shader.AddInput("block_table", ShaderUsage::UseUniform);
  shader.AddInput("seqlens_k", ShaderUsage::None);
  shader.AddInput("seqlens_q", ShaderUsage::None);
  if (q_varlen_) {
    // Optional per-batch running Q-token offsets (size batch_size + 1). Used
    // by the shader to compute q_row = cumulative_seqlens_q[batch] + q_idx
    // when Q arrives already-packed (no BSNH padding).
    shader.AddInput("cumulative_seqlens_q", ShaderUsage::None);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_paged_prefill.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(max_k_step_param, max_k_step_),
                             WGSL_TEMPLATE_PARAMETER(q_varlen, q_varlen_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, qkv_head_size_),
                             WGSL_TEMPLATE_PARAMETER(qkv_num_heads, qkv_num_heads_),
                             WGSL_TEMPLATE_VARIABLE(block_table, block_table),
                             WGSL_TEMPLATE_VARIABLE(key_cache, key_cache),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(q, q),
                             WGSL_TEMPLATE_VARIABLE(value_cache, value_cache));
}

Status ApplyFlashAttentionPagedPrefill(onnxruntime::webgpu::ComputeContext& context,
                                       const Tensor* q,
                                       const Tensor* key_cache,
                                       const Tensor* value_cache,
                                       const Tensor* block_table,
                                       Tensor* output,
                                       const Tensor* seqlen_k,
                                       const Tensor* seqlens_q,
                                       const WebgpuAttentionParameters& parameters,
                                       uint32_t block_size,
                                       uint32_t max_num_blocks_per_seq,
                                       const Tensor* cumulative_seqlens_q) {
  ORT_RETURN_IF_NOT(q != nullptr && key_cache != nullptr && value_cache != nullptr && block_table != nullptr,
                    "Paged prefill requires Q, K/V cache, and block_table.");
  ORT_RETURN_IF_NOT(seqlen_k != nullptr && seqlens_q != nullptr,
                    "Paged prefill requires per-batch seqlen_k and seqlens_q.");
  ORT_RETURN_IF(parameters.qkv_format_ != Q_K_V_BSNH,
                "Paged prefill supports BSNH Q layout only.");

  const bool is_apple = context.AdapterInfo().vendor == std::string_view{"apple"};
  const bool is_fp16 = q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  const bool q_varlen = cumulative_seqlens_q != nullptr;
  FlashAttentionPagedPrefillProgram program{is_fp16,
                                            parameters.head_size_,
                                            parameters.num_heads_,
                                            parameters.is_unidirectional_,
                                            q_varlen};

  // Dispatch shape mirrors the dense FA shm_path: one workgroup per (batch, head,
  // Q-tile) triple, where a Q-tile is workgroup_size_x contiguous Q rows.
  // On Apple GPUs the dense FA uses a 128-lane workgroup to reduce barrier
  // overhead; other vendors use 64.
  const uint32_t prefill_tile_size = is_apple ? 128u : 64u;
  const uint32_t num_seq_tile =
      (static_cast<uint32_t>(parameters.sequence_length_) + prefill_tile_size - 1u) / prefill_tile_size;
  const float alpha =
      parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_)) : parameters.scale_;

  constexpr int components = 4;
  program.AddInputs({{q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {key_cache, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {value_cache, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {block_table, ProgramTensorMetadataDependency::TypeAndRank},
                     {seqlen_k, ProgramTensorMetadataDependency::None},
                     {seqlens_q, ProgramTensorMetadataDependency::None}});
  if (q_varlen) {
    program.AddInputs({{cumulative_seqlens_q, ProgramTensorMetadataDependency::None}});
  }
  program
      .AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .SetDispatchGroupSize(static_cast<uint32_t>(parameters.batch_size_) *
                            static_cast<uint32_t>(parameters.num_heads_) *
                            num_seq_tile)
      .SetWorkgroupSize(prefill_tile_size)
      .CacheHint(parameters.head_size_, parameters.num_heads_, parameters.is_unidirectional_,
                 parameters.kv_num_heads_, block_size, max_num_blocks_per_seq,
                 program.max_k_step(), prefill_tile_size, is_fp16, q_varlen)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {alpha},
                            {num_seq_tile},
                            {block_size},
                            {static_cast<uint32_t>(parameters.kv_num_heads_)}});
  return context.RunProgram(program);
}

Status FlashAttentionPrefillProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (use_seqlens_q_) {
    shader.AddInput("seqlens_q", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_prefill.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(has_local_window, has_local_window_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(kv_step_param, kv_step_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(qk_f32, true),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, qkv_head_size_),
                             WGSL_TEMPLATE_PARAMETER(qkv_num_heads, qkv_num_heads_),
                             WGSL_TEMPLATE_PARAMETER(shuffle_chunk_param, shuffle_chunk_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlens_q, use_seqlens_q_));
}

bool CanApplyFlashAttentionPrefill(bool is_intel,
                                   bool kv_cache_quantization_enabled) {
  return is_intel && !kv_cache_quantization_enabled;
}

Status ApplyFlashAttentionPrefill(onnxruntime::webgpu::ComputeContext& context,
                                  const Tensor* Q,
                                  const Tensor* present_key,
                                  const Tensor* present_value,
                                  const Tensor* attention_bias,
                                  const Tensor* seqlen_k,
                                  Tensor* attn_output,
                                  const WebgpuAttentionParameters& parameters,
                                  bool has_attention_bias,
                                  bool use_seqlen_k,
                                  bool has_local_window,
                                  int local_window_size,
                                  uint32_t present_sequence_length,
                                  uint32_t subgroup_min_size,
                                  float alpha,
                                  uint32_t attn_bias_dim0,
                                  uint32_t attn_bias_dim1,
                                  uint32_t attn_bias_dim3,
                                  bool use_seqlens_q,
                                  const Tensor* seqlens_q,
                                  bool q_BNSH,
                                  bool has_head_sink,
                                  const Tensor* head_sink) {
  const auto& adapter_info = context.AdapterInfo();
  const bool is_intel_xe2 = adapter_info.vendor == std::string_view{"intel"} &&
                            (adapter_info.architecture == std::string_view{"xe-2lpg"} ||
                             adapter_info.architecture == std::string_view{"xe-2hpg"});
  const bool is_intel_xe3 = adapter_info.vendor == std::string_view{"intel"} &&
                            (adapter_info.architecture == std::string_view{"xe-3lpg"} ||
                             adapter_info.architecture == std::string_view{"xe-3lpg-xs"});
  // Default: 64 threads, 16 KV rows per step.
  uint32_t wg_size = 64u;
  uint32_t kv_step = 16u;
  // Xe2: query length <= 128 uses 128 threads and 64 KV rows per step;
  //      longer queries use 256 threads and 32 KV rows per step.
  // Xe3: 128 threads, 64 KV rows per step for all query lengths.
  if (is_intel_xe3 || (is_intel_xe2 && parameters.sequence_length_ <= 128)) {
    wg_size = 128u;
    kv_step = 64u;
  } else if (is_intel_xe2) {
    wg_size = 256u;
    kv_step = 32u;
  }
  // The K tile (full head) and V slice (up to 16 vec4) are staged in workgroup memory for kv_step rows.
  // If that exceeds the device limit, fall back to the default config (64 threads, 16 KV rows per step).
  const uint64_t head_size_vec = static_cast<uint64_t>(parameters.head_size_) / 4;
  const uint64_t vec4_bytes = (Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ? 2u : 4u) * 4u;
  if ((head_size_vec + std::min<uint64_t>(head_size_vec, 16)) * kv_step * vec4_bytes >
      context.DeviceLimits().maxComputeWorkgroupStorageSize) {
    wg_size = 64u;
    kv_step = 16u;
  }
  // Values below 8 select direct shared-memory reads, including without subgroups.
  const uint32_t shuffle_chunk = std::min(kv_step, subgroup_min_size);
  const uint32_t num_seq_tile = CeilDiv(static_cast<uint32_t>(parameters.sequence_length_), wg_size);
  const uint32_t dispatch_size =
      parameters.batch_size_ * parameters.num_heads_ * num_seq_tile;

  FlashAttentionPrefillProgram program{"FlashAttentionPrefill",
                                       has_attention_bias,
                                       parameters.is_unidirectional_,
                                       parameters.head_size_,
                                       parameters.num_heads_,
                                       use_seqlen_k,
                                       kv_step,
                                       shuffle_chunk,
                                       has_local_window,
                                       use_seqlens_q,
                                       q_BNSH,
                                       has_head_sink};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  if (has_attention_bias) {
    program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  if (use_seqlen_k) {
    program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::None}});
  }
  if (use_seqlens_q) {
    program.AddInputs({{seqlens_q, ProgramTensorMetadataDependency::None}});
  }
  if (has_head_sink) {
    program.AddInputs({{head_sink, ProgramTensorMetadataDependency::Type}});
  }
  program.AddOutputs({{attn_output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  program.SetDispatchGroupSize(dispatch_size)
      .SetWorkgroupSize(wg_size)
      .CacheHint(has_attention_bias, parameters.head_size_, parameters.num_heads_, parameters.is_unidirectional_,
                 use_seqlen_k, wg_size, kv_step, shuffle_chunk,
                 has_local_window, use_seqlens_q, q_BNSH, has_head_sink)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(present_sequence_length)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {alpha},
                            {num_seq_tile},
                            {attn_bias_dim0},
                            {attn_bias_dim1},
                            {attn_bias_dim3},
                            {static_cast<uint32_t>(has_local_window ? local_window_size : 0)}});
  return context.RunProgram(program);
}

Status FlashAttentionDecodeQKVProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& q = shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_value = shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (use_seqlens_q_) {
    shader.AddInput("seqlens_q", ShaderUsage::None);
  }
  if (use_indirect_dispatch_) {
    // Global max total sequence length across batches (from GQA input #6).
    // Used in indirect-dispatch mode for the workgroup_idx slicing so that
    // batch 0's per-batch length cannot undersize the dispatch grid.
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  const auto& out_split_vx = shader.AddOutput("out_split_vx", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Wider K tiling (32 vec4) with a 128-thread workgroup is used for decode (m_tile == 1) to
  // mirror MatMulNBits and improve GPU time. For prefill (m_tile > 1) the shared-memory
  // arrays that scale with tile_size_k_vec and m_tile would exceed the 32 KB workgroup
  // storage limit on some adapters, so keep the original 8 vec4 / 64-thread shape there.
  const uint32_t tile_size_k_vec = (m_tile_ == 1u) ? 32u : 8u;
  const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;
  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_qkv.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(bit_width, kv_cache_quantization_bits_),
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(kv_cache_quantization, kv_cache_quantization_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_PARAMETER(use_indirect_dispatch, use_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlens_q, use_seqlens_q_),
                             WGSL_TEMPLATE_PARAMETER(v_head_size_vec, head_size_vec_),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(out_split_vx, out_split_vx),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(q, q));
}

Status FlashAttentionPagedDecodeQKVProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& q = shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_value = shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& block_table = shader.AddInput("block_table", ShaderUsage::UseUniform);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (use_seqlens_q_) {
    shader.AddInput("seqlens_q", ShaderUsage::None);
  }
  if (use_indirect_dispatch_) {
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  const auto& out_split_vx = shader.AddOutput("out_split_vx", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const uint32_t tile_size_k_vec = (m_tile_ == 1u) ? 32u : 8u;
  const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;
  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_paged_decode_qkv.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(bit_width, kv_cache_quantization_bits_),
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(kv_cache_quantization, kv_cache_quantization_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_PARAMETER(use_indirect_dispatch, use_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlens_q, use_seqlens_q_),
                             WGSL_TEMPLATE_PARAMETER(v_head_size_vec, head_size_vec_),
                             WGSL_TEMPLATE_VARIABLE(block_table, block_table),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(out_split_vx, out_split_vx),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(q, q));
}

Status ComputeFlashAttentionDecodeQKV(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                                      const Tensor* attention_bias, Tensor* out_split_vx, Tensor* present_key, Tensor* present_value,
                                      Tensor* metadata, const Tensor* seqlen_k,
                                      const WebgpuAttentionParameters& parameters, const Tensor* indirect_buffer, uint32_t num_total_seq_length_tile, uint32_t num_present_sequence_length_tile, uint32_t tile_size, bool use_indirect_dispatch, uint32_t present_sequence_length, uint32_t m_tile, bool use_seqlen_k, const Tensor* total_seqlen,
                                      uint32_t kv_cache_quantization_bits,
                                      int compressed_head_size_u32,
                                      bool use_seqlens_q, const Tensor* seqlens_q,
                                      int local_window_size) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  const int components = 4;
  // Quantized cache tensor views use packed scalar u32 elements.
  const bool kv_cache_quantization = kv_cache_quantization_bits != 0;
  const int kv_cache_components = kv_cache_quantization ? 1 : components;
  const int head_size_vec = parameters.v_head_size_ / components;

  bool q_BNSH = parameters.qkv_format_ == Q_K_V_BNSH;
  bool is_unidirectional = parameters.is_unidirectional_;
  FlashAttentionDecodeQKVProgram program{
      "FlashAttentionDecodeQKV", has_attention_bias, tile_size, head_size_vec,
      use_indirect_dispatch, q_BNSH, is_unidirectional, m_tile, use_seqlen_k,
      kv_cache_quantization_bits, compressed_head_size_u32, use_seqlens_q};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (use_seqlens_q) {
    program.AddInput({seqlens_q, ProgramTensorMetadataDependency::None});
  }
  if (use_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = parameters.head_size_ / components;

  uint32_t attn_bias_dim0 = 1;
  uint32_t attn_bias_dim1 = 1;
  uint32_t attn_bias_dim3 = 0;
  if (has_attention_bias) {
    const auto& bias_shape = attention_bias->Shape();
    attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
    attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
    attn_bias_dim3 = static_cast<uint32_t>(bias_shape[3]);
  }

  if (use_indirect_dispatch) {
    program.SetIndirectDispatchTensor(indirect_buffer);
  } else {
    program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_total_seq_length_tile);
  }
  // Workgroup size mirrors the tile_size_k_vec choice inside the program's shader (see
  // FlashAttentionDecodeQKVProgram::GenerateShaderCode): 128 threads with 32 vec4 K tiles
  // for decode, 64 threads with 8 vec4 K tiles for prefill.
  const uint32_t workgroup_size = (m_tile == 1u) ? 128u : 64u;
  program.SetWorkgroupSize(workgroup_size)
      .CacheHint(tile_size, head_size_vec, has_attention_bias, use_indirect_dispatch, q_BNSH,
                 is_unidirectional, m_tile, use_seqlen_k, kv_cache_quantization_bits,
                 compressed_head_size_u32, use_seqlens_q, local_window_size)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<float>(alpha)},
                            present_sequence_length,
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {num_present_sequence_length_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {attn_bias_dim0},
                            {attn_bias_dim1},
                            {attn_bias_dim3},
                            {static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(std::max(local_window_size, 0))}});

  return context.RunProgram(program);
}

Status ComputeFlashAttentionPagedDecodeQKV(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                                           const Tensor* attention_bias, Tensor* out_split_vx, Tensor* present_key, Tensor* present_value,
                                           Tensor* metadata, const Tensor* seqlen_k, const Tensor* block_table,
                                           const WebgpuAttentionParameters& parameters, const Tensor* indirect_buffer, uint32_t num_total_seq_length_tile, uint32_t num_present_sequence_length_tile, uint32_t tile_size, bool use_indirect_dispatch, uint32_t present_sequence_length, uint32_t m_tile, bool use_seqlen_k, const Tensor* total_seqlen,
                                           uint32_t kv_cache_quantization_bits,
                                           int compressed_head_size_u32,
                                           bool use_seqlens_q, const Tensor* seqlens_q,
                                           uint32_t block_size, uint32_t max_num_blocks_per_seq) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  const int components = 4;
  const bool kv_cache_quantization = kv_cache_quantization_bits != 0;
  const int kv_cache_components = kv_cache_quantization ? 1 : components;
  const int head_size_vec = parameters.v_head_size_ / components;

  bool q_BNSH = parameters.qkv_format_ == Q_K_V_BNSH;
  bool is_unidirectional = parameters.is_unidirectional_;
  FlashAttentionPagedDecodeQKVProgram program{
      "FlashAttentionPagedDecodeQKV", has_attention_bias, tile_size, head_size_vec,
      use_indirect_dispatch, q_BNSH, is_unidirectional, m_tile, use_seqlen_k,
      kv_cache_quantization_bits, compressed_head_size_u32, use_seqlens_q};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components},
                     {block_table, ProgramTensorMetadataDependency::TypeAndRank}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (use_seqlens_q) {
    program.AddInput({seqlens_q, ProgramTensorMetadataDependency::None});
  }
  if (use_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = parameters.head_size_ / components;

  uint32_t attn_bias_dim0 = 1;
  uint32_t attn_bias_dim1 = 1;
  uint32_t attn_bias_dim3 = 0;
  if (has_attention_bias) {
    const auto& bias_shape = attention_bias->Shape();
    attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
    attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
    attn_bias_dim3 = static_cast<uint32_t>(bias_shape[3]);
  }

  if (use_indirect_dispatch) {
    program.SetIndirectDispatchTensor(indirect_buffer);
  } else {
    program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_total_seq_length_tile);
  }
  const uint32_t workgroup_size = (m_tile == 1u) ? 128u : 64u;
  program.SetWorkgroupSize(workgroup_size)
      .CacheHint(tile_size, head_size_vec, has_attention_bias, use_indirect_dispatch, q_BNSH,
                 is_unidirectional, m_tile, use_seqlen_k, kv_cache_quantization_bits,
                 compressed_head_size_u32, use_seqlens_q, block_size,
                 max_num_blocks_per_seq, parameters.kv_num_heads_)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<float>(alpha)},
                            present_sequence_length,
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {num_present_sequence_length_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {attn_bias_dim0},
                            {attn_bias_dim1},
                            {attn_bias_dim3},
                            {static_cast<uint32_t>(parameters.sequence_length_)},
                            {block_size},
                            {max_num_blocks_per_seq},
                            {static_cast<uint32_t>(parameters.kv_num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeVxReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddInput("metadata", ShaderUsage::UseUniform);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_vx_reduce.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(seq_tile_size, seq_tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status FlashAttentionPagedDecodeVxReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddInput("metadata", ShaderUsage::UseUniform);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_paged_decode_vx_reduce.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(seq_tile_size, seq_tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status ComputeFlashAttentionDecodeVxReduce(onnxruntime::webgpu::ComputeContext& context,
                                           const Tensor* out_split_vx,
                                           const Tensor* metadata,
                                           Tensor* output,
                                           const Tensor* seqlen_k,
                                           const WebgpuAttentionParameters& parameters,
                                           uint32_t num_total_seq_length_tile,
                                           uint32_t num_present_sequence_length_tile,
                                           uint32_t present_sequence_length,
                                           uint32_t seq_tile_size,
                                           const Tensor* head_sink,
                                           uint32_t m_tile,
                                           bool use_seqlen_k) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_head_size = tile_size * components;
  bool has_head_sink = head_sink != nullptr;
  FlashAttentionDecodeVxReduceProgram program{"FlashAttentionDecodeVxReduce", tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k};
  program.AddInputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {metadata, ProgramTensorMetadataDependency::TypeAndRank, 2}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (has_head_sink) {
    program.AddInput({head_sink, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  const uint32_t num_head_size_tile = static_cast<uint32_t>((parameters.v_head_size_ + tile_head_size - 1) / tile_head_size);
  const uint32_t batch_heads = static_cast<uint32_t>(parameters.batch_size_ * parameters.num_heads_);
  program.SetDispatchGroupSize(batch_heads * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_head_size_tile)
      .CacheHint(tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k)
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            num_total_seq_length_tile,
                            num_present_sequence_length_tile,
                            present_sequence_length,
                            {num_head_size_tile},
                            {batch_heads},
                            {static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status ComputeFlashAttentionPagedDecodeVxReduce(onnxruntime::webgpu::ComputeContext& context,
                                                const Tensor* out_split_vx,
                                                const Tensor* metadata,
                                                Tensor* output,
                                                const Tensor* seqlen_k,
                                                const WebgpuAttentionParameters& parameters,
                                                uint32_t num_total_seq_length_tile,
                                                uint32_t num_present_sequence_length_tile,
                                                uint32_t seq_tile_size,
                                                const Tensor* head_sink,
                                                uint32_t m_tile,
                                                bool use_seqlen_k) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_head_size = tile_size * components;
  bool has_head_sink = head_sink != nullptr;
  FlashAttentionPagedDecodeVxReduceProgram program{"FlashAttentionPagedDecodeVxReduce", tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k};
  program.AddInputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {metadata, ProgramTensorMetadataDependency::TypeAndRank, 2}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (has_head_sink) {
    program.AddInput({head_sink, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  const uint32_t num_head_size_tile = static_cast<uint32_t>((parameters.v_head_size_ + tile_head_size - 1) / tile_head_size);
  const uint32_t batch_heads = static_cast<uint32_t>(parameters.batch_size_ * parameters.num_heads_);
  program.SetDispatchGroupSize(batch_heads * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_head_size_tile)
      .CacheHint(tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k)
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            num_total_seq_length_tile,
                            num_present_sequence_length_tile,
                            {num_head_size_tile},
                            {batch_heads},
                            {static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

namespace {

bool CanApplyDensePrefill(const Tensor* Q, const WebgpuAttentionParameters& parameters,
                          onnxruntime::webgpu::ComputeContext& context,
                          const Tensor* block_table, uint32_t kv_cache_quantization_bits) {
  if (block_table != nullptr || parameters.sequence_length_ < 32) {
    return false;
  }
  const auto vendor = context.AdapterInfo().vendor;
  const bool is_apple = vendor == std::string_view{"apple"};
  return DensePrefillFitsWorkgroupStorage(
      is_apple || vendor == std::string_view{"nvidia"} || !context.HasFeature(wgpu::FeatureName::Subgroups),
      Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
      parameters.head_size_, kv_cache_quantization_bits, vendor == std::string_view{"qualcomm"},
      is_apple ? 128u : 64u, context.DeviceLimits().maxComputeWorkgroupStorageSize);
}

bool CanApplyFusedPagedPrefill(const Tensor* Q, const WebgpuAttentionParameters& parameters,
                               onnxruntime::webgpu::ComputeContext& context,
                               const Tensor* block_table, uint32_t block_size,
                               uint32_t kv_cache_quantization_bits, const Tensor* attention_bias,
                               const Tensor* head_sink, const Tensor* seqlen_k, const Tensor* seqlens_q) {
  return block_table != nullptr && kv_cache_quantization_bits == 0 &&
         attention_bias == nullptr && head_sink == nullptr && parameters.qkv_format_ == Q_K_V_BSNH &&
         seqlen_k != nullptr && seqlens_q != nullptr &&
         CanApplyFlashAttentionPagedPrefill(context,
                                            Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                            parameters.sequence_length_, parameters.head_size_, static_cast<int>(block_size));
}

bool CanApplySplitReduce(const Tensor* Q, const WebgpuAttentionParameters& parameters,
                         onnxruntime::webgpu::ComputeContext& context,
                         const Tensor* block_table, uint32_t kv_cache_quantization_bits) {
  // Short queries favor split-reduce; dense prefill also falls back when storage does not fit.
  return parameters.sequence_length_ < 32 ||
         (block_table == nullptr &&
          !CanApplyDensePrefill(Q, parameters, context, block_table, kv_cache_quantization_bits));
}

bool CanApplyPagedDecode(const Tensor* block_table, uint32_t kv_cache_quantization_bits) {
  return block_table != nullptr && kv_cache_quantization_bits == 0;
}

Status ApplyDensePrefill(const Tensor* Q, Tensor* present_key, Tensor* present_value,
                         const Tensor* attention_bias, Tensor* attn_output,
                         const WebgpuAttentionParameters& parameters,
                         onnxruntime::webgpu::ComputeContext& context,
                         const Tensor* seqlen_k, const Tensor* seqlens_q, const Tensor* head_sink,
                         uint32_t present_sequence_length, uint32_t kv_cache_quantization_bits,
                         int compressed_head_size_u32, int local_window_size) {
  constexpr uint32_t tile_size = 64;
  const bool use_seqlen_k = seqlen_k != nullptr;
  const bool use_seqlens_q = seqlens_q != nullptr;
  const bool kv_cache_quantization_enabled = kv_cache_quantization_bits != 0;
  const bool has_local_window = local_window_size > 0;
  const bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};
  const bool is_apple = context.AdapterInfo().vendor == std::string_view{"apple"};
  const bool is_qualcomm = context.AdapterInfo().vendor == std::string_view{"qualcomm"};
  const bool has_subgroups = context.HasFeature(wgpu::FeatureName::Subgroups);
  // Prefill path: FlashAttentionProgram (single kernel with subgroup shuffles)
  bool has_attention_bias = attention_bias != nullptr;
  bool is_fp16 = Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  bool q_BNSH = parameters.qkv_format_ == Q_K_V_BNSH;
  bool has_head_sink = head_sink != nullptr;
  FlashAttentionProgram program{"FlashAttention",
                                has_attention_bias,
                                is_qualcomm,
                                is_fp16,
                                parameters.head_size_,
                                parameters.num_heads_,
                                parameters.is_unidirectional_,
                                is_nvidia,
                                is_apple,
                                has_subgroups,
                                q_BNSH,
                                use_seqlen_k,
                                has_head_sink,
                                has_local_window,
                                kv_cache_quantization_bits,
                                compressed_head_size_u32,
                                use_seqlens_q};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank,
                      kv_cache_quantization_enabled ? 1 : 4},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank,
                      kv_cache_quantization_enabled ? 1 : 4}});
  if (has_attention_bias) {
    program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  if (use_seqlen_k) {
    program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::None}});
  }
  if (use_seqlens_q) {
    program.AddInputs({{seqlens_q, ProgramTensorMetadataDependency::None}});
  }
  if (has_head_sink) {
    program.AddInputs({{head_sink, ProgramTensorMetadataDependency::Type}});
  }
  program.AddOutputs({{attn_output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  // On Apple GPUs, use a larger workgroup size to reduce barrier overhead.
  const uint32_t prefill_tile_size = is_apple ? 128 : tile_size;
  const uint32_t num_seq_tile = (parameters.sequence_length_ + prefill_tile_size - 1) / prefill_tile_size;

  uint32_t attn_bias_dim0 = 1;
  uint32_t attn_bias_dim1 = 1;
  uint32_t attn_bias_dim3 = 0;
  if (has_attention_bias) {
    const auto& bias_shape = attention_bias->Shape();
    attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
    attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
    attn_bias_dim3 = static_cast<uint32_t>(bias_shape[3]);
  }

  const bool is_intel = context.AdapterInfo().vendor == std::string_view{"intel"};
  const uint32_t subgroup_min_size = has_subgroups ? context.AdapterInfo().subgroupMinSize : 0u;
  // TODO(daijh): Merge back into `FlashAttentionProgram`.
  if (CanApplyFlashAttentionPrefill(is_intel, kv_cache_quantization_enabled)) {
    return ApplyFlashAttentionPrefill(
        context, Q, present_key, present_value, attention_bias, seqlen_k, attn_output,
        parameters, has_attention_bias, use_seqlen_k, has_local_window, local_window_size,
        present_sequence_length, subgroup_min_size, alpha, attn_bias_dim0, attn_bias_dim1, attn_bias_dim3,
        use_seqlens_q, seqlens_q, q_BNSH, has_head_sink, head_sink);
  }

  program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * num_seq_tile)
      .SetWorkgroupSize(prefill_tile_size)
      .CacheHint(has_attention_bias, parameters.head_size_, parameters.num_heads_,
                 parameters.is_unidirectional_, is_qualcomm, is_nvidia, is_apple,
                 has_subgroups, q_BNSH, use_seqlen_k, has_head_sink, has_local_window,
                 kv_cache_quantization_bits,
                 compressed_head_size_u32, program.max_k_step(), use_seqlens_q)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(present_sequence_length)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {alpha},
                            {num_seq_tile},
                            {attn_bias_dim0},
                            {attn_bias_dim1},
                            {attn_bias_dim3},
                            {static_cast<uint32_t>(has_local_window ? local_window_size : 0)}});

  return context.RunProgram(program);
}

Status ApplySplitReduce(const Tensor* Q, Tensor* present_key, Tensor* present_value,
                        const Tensor* attention_bias, Tensor* attn_output,
                        const WebgpuAttentionParameters& parameters,
                        onnxruntime::webgpu::ComputeContext& context,
                        const Tensor* seqlen_k, const Tensor* seqlens_q, const Tensor* head_sink,
                        uint32_t present_sequence_length, uint32_t kv_cache_quantization_bits,
                        int compressed_head_size_u32, int local_window_size, const Tensor* block_table,
                        uint32_t block_size, uint32_t max_num_blocks_per_seq,
                        Tensor* indirect_buffer_ptr, uint32_t m_tile,
                        const Tensor* total_seqlen) {
  constexpr uint32_t tile_size = 64;
  const bool use_seqlen_k = seqlen_k != nullptr;
  const bool use_seqlens_q = seqlens_q != nullptr;
  const bool use_paged_kv_cache = block_table != nullptr;
  const bool use_indirect_dispatch = indirect_buffer_ptr != nullptr;
  // Split-reduce path (fused QKV + VxReduce). Handles quantized and unquantized caches.
  const uint32_t num_total_seq_length_tile = (parameters.total_sequence_length_ + tile_size - 1) / tile_size;
  const uint32_t num_present_sequence_length_tile = (present_sequence_length + tile_size - 1) / tile_size;

  const TensorShapeVector metadata_dims({parameters.batch_size_, parameters.num_heads_,
                                         parameters.sequence_length_, num_present_sequence_length_tile, 2});
  const TensorShape metadata_shape(metadata_dims);
  Tensor metadata = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), metadata_shape);

  const TensorShapeVector out_split_vx_dims({parameters.batch_size_, parameters.num_heads_,
                                             parameters.sequence_length_, num_present_sequence_length_tile, parameters.head_size_});
  const TensorShape out_split_vx_shape(out_split_vx_dims);
  Tensor out_split_vx = context.CreateGPUTensor(Q->DataType(), out_split_vx_shape);

  if (CanApplyPagedDecode(block_table, kv_cache_quantization_bits)) {
    ORT_RETURN_IF_ERROR(ComputeFlashAttentionPagedDecodeQKV(context, Q, attention_bias, &out_split_vx, present_key, present_value,
                                                            &metadata, seqlen_k, block_table,
                                                            parameters, indirect_buffer_ptr, num_total_seq_length_tile,
                                                            num_present_sequence_length_tile, tile_size, use_indirect_dispatch,
                                                            present_sequence_length, m_tile, use_seqlen_k, total_seqlen,
                                                            kv_cache_quantization_bits,
                                                            compressed_head_size_u32,
                                                            use_seqlens_q, seqlens_q,
                                                            block_size, max_num_blocks_per_seq));

    ORT_RETURN_IF_ERROR(ComputeFlashAttentionPagedDecodeVxReduce(context, &out_split_vx, &metadata, attn_output, seqlen_k, parameters,
                                                                 num_total_seq_length_tile,
                                                                 num_present_sequence_length_tile, tile_size,
                                                                 head_sink, m_tile, use_seqlen_k));
  } else {
    // Quantized paged caches cannot be interpreted by the dense decode shader.
    if (use_paged_kv_cache) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "FlashAttention (WebGPU): paged KV cache present but the paged "
                             "decode path was rejected (KV cache quantization enabled). Extend "
                             "FlashAttentionPagedDecodeQKV to support the requested feature, "
                             "or gate the feature off at the PagedAttention layer before "
                             "dispatching FA.");
    }
    ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeQKV(context, Q, attention_bias, &out_split_vx, present_key, present_value,
                                                       &metadata, seqlen_k,
                                                       parameters, indirect_buffer_ptr, num_total_seq_length_tile,
                                                       num_present_sequence_length_tile, tile_size, use_indirect_dispatch,
                                                       present_sequence_length, m_tile, use_seqlen_k, total_seqlen,
                                                       kv_cache_quantization_bits,
                                                       compressed_head_size_u32,
                                                       use_seqlens_q, seqlens_q, local_window_size));

    ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeVxReduce(context, &out_split_vx, &metadata, attn_output, seqlen_k, parameters,
                                                            num_total_seq_length_tile,
                                                            num_present_sequence_length_tile, present_sequence_length, tile_size,
                                                            head_sink, m_tile, use_seqlen_k));
  }
  return Status::OK();
}

}  // namespace

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k,
                           const Tensor* cos_cache, const Tensor* sin_cache, const Tensor* head_sink,
                           const Tensor* total_seqlen, const Tensor* seqlens_q,
                           const Tensor* block_table, uint32_t block_size, uint32_t max_num_blocks_per_seq,
                           const Tensor* cumulative_seqlens_q, int local_window_size) {
  constexpr uint32_t tile_size = 64;
  const bool use_paged_kv_cache = block_table != nullptr;

  const uint32_t kv_cache_quantization_bits = context.KvCacheQuantizationBits();
  const bool kv_cache_quantization_enabled = kv_cache_quantization_bits != 0;
  const bool use_q4_turbo_quant = kv_cache_quantization_bits == 4;
  const bool use_q8_block_quant = kv_cache_quantization_bits == 8;
  if (use_q4_turbo_quant &&
      (parameters.head_size_ < 8 || (parameters.head_size_ & (parameters.head_size_ - 1)) != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Q4 TurboQuant KV cache requires head_size >= 8 and a power of 2. Got head_size=",
                           parameters.head_size_);
  }
  if (use_q8_block_quant && (parameters.head_size_ < 4 || parameters.head_size_ % 4 != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Q8 block-quantized KV cache requires head_size to be divisible by 4. Got head_size=",
                           parameters.head_size_);
  }
  if (use_q8_block_quant &&
      Q8QuantizationWorkgroupStorageBytes(parameters.head_size_) >
          context.DeviceLimits().maxComputeWorkgroupStorageSize) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Q8 block-quantized KV cache requires more workgroup storage than the device supports. Required=",
        Q8QuantizationWorkgroupStorageBytes(parameters.head_size_),
        " bytes, supported=", context.DeviceLimits().maxComputeWorkgroupStorageSize, " bytes.");
  }

  // Compressed head dimension, expressed in two units:
  //   compressed_head_size_u32 — u32 words per head (1 scale + packed quantized values),
  //                              passed to the shaders as the packed KV dimension.
  //   present_last_dim         — the same span counted in Q elements (fp16/fp32), used to size an
  //                              internally-allocated present buffer so its u32 view lines up
  //                              (compressed_head_size_u32 * 4 bytes == present_last_dim * sizeof(Q elem)).
  const int compressed_head_size_u32 =
      kv_cache_quantization_enabled
          ? KvCacheQuantizedHeadSizeU32(parameters.head_size_, kv_cache_quantization_bits)
          : 0;
  const int64_t present_last_dim =
      kv_cache_quantization_enabled
          ? KvCacheQuantizedHeadSize(parameters.head_size_, kv_cache_quantization_bits,
                                     Q->DataType()->Size())
          : parameters.head_size_;
  // Create present_key and present_value tensors if they are nullptr.
  // Skip allocation for kv_empty — present will be aliased to past below.
  Tensor internal_present_key;
  Tensor internal_present_value;
  const int present_kv_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  const bool kv_empty = (parameters.kv_sequence_length_ == 0);
  if (!kv_empty && present_key == nullptr) {
    TensorShapeVector present_kv_shape({parameters.batch_size_, present_kv_heads,
                                        parameters.total_sequence_length_, present_last_dim});
    internal_present_key = context.CreateGPUTensor(Q->DataType(), TensorShape(present_kv_shape));
    present_key = &internal_present_key;
  }
  if (!kv_empty && present_value == nullptr) {
    TensorShapeVector present_kv_shape({parameters.batch_size_, present_kv_heads,
                                        parameters.total_sequence_length_, present_last_dim});
    internal_present_value = context.CreateGPUTensor(Q->DataType(), TensorShape(present_kv_shape));
    present_value = &internal_present_value;
  }

  // Read seqlens_k per batch_idx in the shader whenever seqlens_k is supplied.
  // This covers both graph-capture (total_sequence_length_ is 0 on the host) and
  // right-padded batches (batch_size > 1 with distinct per-batch totals), and lets
  // batch=1 share the same path. When seqlens_k is null, kernels fall back to
  // uniforms.total_sequence_length.
  const bool use_seqlen_k = seqlen_k != nullptr;

  // Declare query_output at function scope to ensure it persists throughout the function
  Tensor query_output;
  // Declare rotated_q at function scope so the pointer remains valid
  Tensor rotated_q;

  // Compute m_tile early so it can be passed to CopyKVCache for indirect dispatch.
  uint32_t m_tile = parameters.sequence_length_ >= 4 ? 4u : (parameters.sequence_length_ >= 2 ? 2u : 1u);
  const uint32_t head_size_vec = static_cast<uint32_t>(parameters.v_head_size_ / 4);
  m_tile = SelectDecodeMTile(
      m_tile, tile_size, head_size_vec, Q->DataType()->Size(), kv_cache_quantization_bits,
      use_paged_kv_cache, context.DeviceLimits().maxComputeWorkgroupStorageSize);
  ORT_RETURN_IF_NOT(
      DecodeWorkgroupStorageBytes(m_tile, tile_size, head_size_vec, Q->DataType()->Size(),
                                  kv_cache_quantization_bits, use_paged_kv_cache) <=
          context.DeviceLimits().maxComputeWorkgroupStorageSize,
      "FlashAttention requires more workgroup storage than the device supports.");
  const uint32_t num_q_tiles = (static_cast<uint32_t>(parameters.sequence_length_) + m_tile - 1u) / m_tile;

  // Create indirect dispatch buffer if using indirect dispatch
  Tensor* indirect_buffer_ptr = nullptr;
  Tensor indirect_buffer;

  // Prepare indirect dispatch buffer for split-reduce path with static KV cache.
  // When graph capture is enabled, total_sequence_length_ may be 0 (GPU-based
  // seqlen_k), so the indirect buffer computes dispatch sizes on GPU.
  // Static KV cache (past_present_share_buffer_) is guaranteed by GQA's
  // ORT_ENFORCE when graph capture is enabled.
  const bool use_indirect_dispatch = seqlen_k != nullptr &&
                                     total_seqlen != nullptr &&
                                     context.IsGraphCaptureEnabled();
  if (use_indirect_dispatch) {
    const TensorShape indirect_buffer_shape{3};  // 3 uint32 values for dispatch dimensions
    indirect_buffer = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), indirect_buffer_shape);
    indirect_buffer_ptr = &indirect_buffer;
  }

  const bool do_rotary = (cos_cache != nullptr && sin_cache != nullptr);

  // kv_empty (kv_sequence_length_ == 0) occurs in KV-shared / cross-layer KV reuse layers: the
  // layer computes its own Q but borrows another layer's already-populated KV cache instead of
  // producing new K/V. There is nothing to copy, so CopyKVCache is skipped and attention reads
  // the past buffers directly. Because no new KV is written, present buffers are intentionally
  // not allocated above and some call sites pass nullptr present outputs — so we alias past as
  // present here.
  if (kv_empty) {
    // do_rotary must be false here: GQA passes cos_cache=nullptr, sin_cache=nullptr for kv_empty
    // layers (rotary is applied to Q separately in GQA before calling ApplyFlashAttention).
    ORT_ENFORCE(!do_rotary, "kv_empty (kv_sequence_length==0) is incompatible with fused rotary+copyKV.");
    ORT_ENFORCE(past_key != nullptr && past_value != nullptr,
                "kv_empty path requires past KV context (KV-shared layers reuse another layer's cache).");
    // When past_present_share_buffer_ is true (MayInplace optimization), present already
    // shares the past buffer. No aliasing needed — the data is already in place.
    if (!parameters.past_present_share_buffer_) {
      // Alias past as present — flash attention only reads present_key/present_value,
      // and CopyKVCache is skipped when kv_empty, so no writes occur through these pointers.
      present_key = const_cast<Tensor*>(past_key);
      present_value = const_cast<Tensor*>(past_value);
    }

    // CopyKVCache normally prepares the indirect dispatch buffer. For kv_empty layers
    // CopyKVCache is skipped, so we prepare it here. Only needed under graph capture
    // because that is when total_seqlen is GPU-resident and CPU-side dispatch sizing
    // is unavailable.
    if (use_indirect_dispatch) {
      const uint32_t present_sequence_capacity =
          use_paged_kv_cache ? block_size * max_num_blocks_per_seq
                             : static_cast<uint32_t>(present_key->Shape()[2]);
      PrepareIndirectDispatchProgram program;
      program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
      program.AddOutput({indirect_buffer_ptr, ProgramTensorMetadataDependency::None});
      program.SetDispatchGroupSize(1)
          .SetWorkgroupSize(1)
          .AddUniformVariables({{tile_size},
                                {static_cast<uint32_t>(parameters.num_heads_)},
                                {num_q_tiles},
                                {static_cast<uint32_t>(parameters.batch_size_)},
                                {present_sequence_capacity}});
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    }
  }

  // Quantized KV caches use u32 views over buffers whose external element type matches Q.
  Tensor present_key_u32, present_value_u32;
  Tensor past_key_u32, past_value_u32;
  Tensor* quantized_present_key = present_key;
  Tensor* quantized_present_value = present_value;
  const Tensor* quantized_past_key = past_key;
  const Tensor* quantized_past_value = past_value;
  if (kv_cache_quantization_enabled) {
    const int64_t bytes_per_elem = static_cast<int64_t>(present_key->DataType()->Size());
    const int64_t expected_last_dim_bytes = static_cast<int64_t>(compressed_head_size_u32) * 4;
    ORT_RETURN_IF_ERROR(
        (present_key->Shape().NumDimensions() == 4 && present_value->Shape().NumDimensions() == 4)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "KV cache quantization expects present_key/present_value to be 4-D tensors."));
    ORT_RETURN_IF_ERROR(
        (present_key->Shape()[3] * bytes_per_elem == expected_last_dim_bytes)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Quantized KV cache shape mismatch for present_key. Expected last_dim_bytes==",
                              expected_last_dim_bytes, ", got shape=", present_key->Shape().ToString()));
    ORT_RETURN_IF_ERROR(
        (present_value->Shape()[3] * bytes_per_elem == expected_last_dim_bytes)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Quantized KV cache shape mismatch for present_value. Expected last_dim_bytes==",
                              expected_last_dim_bytes, ", got shape=", present_value->Shape().ToString()));

    TensorShapeVector u32_present_shape({present_key->Shape()[0], present_key->Shape()[1],
                                         present_key->Shape()[2],
                                         static_cast<int64_t>(compressed_head_size_u32)});
    present_key_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_present_shape),
                             present_key->MutableDataRaw(), present_key->Location());
    present_value_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_present_shape),
                               present_value->MutableDataRaw(), present_value->Location());
    quantized_present_key = &present_key_u32;
    quantized_present_value = &present_value_u32;

    if (past_key != nullptr && past_key->SizeInBytes() > 0) {
      TensorShapeVector u32_past_shape({past_key->Shape()[0], past_key->Shape()[1],
                                        past_key->Shape()[2],
                                        static_cast<int64_t>(compressed_head_size_u32)});
      // past_key_u32 / past_value_u32 are read-only aliases over the past KV cache buffers.
      // The Tensor ctor takes a non-const data pointer, so const_cast is required here, but the
      // flash attention kernels only read through the quantized aliases — never write.
      past_key_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_past_shape),
                            const_cast<void*>(past_key->DataRaw()), past_key->Location());
      past_value_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_past_shape),
                              const_cast<void*>(past_value->DataRaw()), past_value->Location());
      quantized_past_key = &past_key_u32;
      quantized_past_value = &past_value_u32;
    }
  }

  // K/V copy is skipped for kv_empty (see the aliasing block above for why).
  if (!kv_empty) {
    if (do_rotary) {
      ORT_ENFORCE(parameters.is_packed_qkv_, "Fused SplitPackedQKVWithRotaryEmbeddingAndCopyKV requires packed QKV input.");
      ORT_ENFORCE(parameters.past_present_share_buffer_, "Fused SplitPackedQKVWithRotaryEmbeddingAndCopyKV requires static KV cache.");

      // Q points to the packed QKV tensor in this case, create query output tensor
      query_output = context.CreateGPUTensor(Q->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));

      if (use_q4_turbo_quant) {
        ORT_RETURN_IF_ERROR(TurboQuantApplyRotaryAndCopyToQuantizedKVCache(context, parameters,
                                                                           Q, seqlen_k,
                                                                           cos_cache, sin_cache,
                                                                           &query_output,
                                                                           quantized_present_key,
                                                                           quantized_present_value,
                                                                           indirect_buffer_ptr, tile_size, num_q_tiles,
                                                                           total_seqlen));
      } else if (use_q8_block_quant) {
        ORT_RETURN_IF_ERROR(BlockQuantInt8ApplyRotaryAndCopyToKvCache(
            context, parameters, Q, seqlen_k, cos_cache, sin_cache, &query_output,
            quantized_present_key, quantized_present_value, indirect_buffer_ptr,
            tile_size, num_q_tiles, total_seqlen));
      } else {
        ORT_RETURN_IF_ERROR(RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV(context, parameters,
                                                                          Q, seqlen_k,
                                                                          cos_cache, sin_cache,
                                                                          &query_output, present_key, present_value,
                                                                          indirect_buffer_ptr, tile_size, num_q_tiles,
                                                                          total_seqlen));
      }
      Q = &query_output;
    } else if (kv_cache_quantization_enabled) {
      ORT_ENFORCE(K != nullptr && V != nullptr,
                  "KV cache quantization requires non-null K/V inputs when kv_sequence_length > 0.");
      if (use_q4_turbo_quant) {
        ORT_RETURN_IF_ERROR(TurboQuantCopyToQuantizedKVCache(
            context, parameters, K, quantized_past_key, quantized_present_key,
            V, quantized_past_value, quantized_present_value, tile_size,
            use_seqlen_k ? seqlen_k : nullptr, indirect_buffer_ptr, num_q_tiles,
            total_seqlen));
      } else {
        ORT_RETURN_IF_ERROR(BlockQuantInt8CopyToKvCache(
            context, parameters, K, quantized_past_key, quantized_present_key,
            V, quantized_past_value, quantized_present_value, tile_size,
            use_seqlen_k ? seqlen_k : nullptr, indirect_buffer_ptr, num_q_tiles,
            total_seqlen));
      }
    } else {
      ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, tile_size, use_seqlen_k ? seqlen_k : nullptr, indirect_buffer_ptr, num_q_tiles, total_seqlen));
    }
  }

  // Dense KV cache uses BNSH and takes sequence length from shape[2]. Paged KV cache
  // uses (num_blocks, block_size, kv_num_heads, head_size), so use the validated
  // per-call max KV length carried in parameters.total_sequence_length_.
  const uint32_t present_sequence_length = use_paged_kv_cache
                                               ? static_cast<uint32_t>(parameters.total_sequence_length_)
                                               : static_cast<uint32_t>(present_key->Shape()[2]);

  // Q4 stores Hadamard-rotated K/V, so rotate Q into the same basis. Q8 is vanilla INT8.
  if (use_q4_turbo_quant) {
    rotated_q = context.CreateGPUTensor(Q->DataType(), Q->Shape());
    ORT_RETURN_IF_ERROR(ApplyHadamardTransform(context, Q, &rotated_q, parameters.head_size_));
    Q = &rotated_q;
  }

  // Q4 attention produces values in the Hadamard basis and needs an inverse transform.
  Tensor attn_output_temp;
  Tensor* attn_output = output;
  if (use_q4_turbo_quant) {
    attn_output_temp = context.CreateGPUTensor(output->DataType(), output->Shape());
    attn_output = &attn_output_temp;
  }

  Tensor* attention_key = kv_cache_quantization_enabled ? quantized_present_key : present_key;
  Tensor* attention_value = kv_cache_quantization_enabled ? quantized_present_value : present_value;

  if (CanApplyFusedPagedPrefill(Q, parameters, context, block_table, block_size,
                                kv_cache_quantization_bits, attention_bias, head_sink, seqlen_k, seqlens_q)) {
    ORT_RETURN_IF_ERROR(ApplyFlashAttentionPagedPrefill(context, Q, attention_key, attention_value,
                                                        block_table, attn_output, seqlen_k, seqlens_q,
                                                        parameters, block_size, max_num_blocks_per_seq,
                                                        cumulative_seqlens_q));
  } else if (CanApplyDensePrefill(Q, parameters, context, block_table, kv_cache_quantization_bits)) {
    ORT_RETURN_IF_ERROR(ApplyDensePrefill(Q, attention_key, attention_value, attention_bias,
                                          attn_output, parameters, context, seqlen_k, seqlens_q,
                                          head_sink, present_sequence_length, kv_cache_quantization_bits,
                                          compressed_head_size_u32, local_window_size));
  } else if (CanApplySplitReduce(Q, parameters, context, block_table, kv_cache_quantization_bits)) {
    ORT_RETURN_IF_ERROR(ApplySplitReduce(Q, attention_key, attention_value, attention_bias,
                                         attn_output, parameters, context, seqlen_k, seqlens_q,
                                         head_sink, present_sequence_length, kv_cache_quantization_bits,
                                         compressed_head_size_u32, local_window_size, block_table,
                                         block_size, max_num_blocks_per_seq, indirect_buffer_ptr,
                                         m_tile, total_seqlen));
  } else {
    // A paged cache must never fall through to a shader that assumes dense BNSH storage.
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FlashAttention (WebGPU): paged KV cache present but the fused "
                           "paged-prefill path was rejected by the dtype / shape / attention_bias / "
                           "head_sink / KV cache quantization / qkv_format / sequence-length gates.");
  }

  // Apply the Q4 inverse Hadamard transform: attn_output_temp -> output.
  if (use_q4_turbo_quant) {
    ORT_RETURN_IF_ERROR(ApplyHadamardTransform(context, attn_output, output, parameters.head_size_));
  }

  return Status::OK();
}

bool CanApplyFlashAttention(const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return !parameters.is_packed_qkv_ &&
         parameters.head_size_ == parameters.v_head_size_ &&
         ((context.AdapterInfo().vendor == std::string_view{"qualcomm"} && parameters.head_size_ % 8 == 0) || parameters.head_size_ % 4 == 0);
}

bool CanApplyFlashAttentionPagedPrefill(onnxruntime::webgpu::ComputeContext& context,
                                        bool is_fp16,
                                        int max_seqlen_q,
                                        int head_size,
                                        int block_size) {
  // The paged prefill shader uses only shared-memory algorithms (no subgroup
  // intrinsics), so it runs correctly on every WebGPU adapter. No adapter
  // gate here — the only gates are dtype, shape, shm-budget, and
  // block/tile alignment.
  (void)context;
  if (!is_fp16) {
    return false;
  }
  // Below the split-reduce threshold ApplyFlashAttention routes to
  // FlashAttentionPagedDecode* instead of the fused prefill shader.
  if (max_seqlen_q < 32) {
    return false;
  }
  // Shared-memory budget: FlashAttentionPagedPrefillProgram requires a
  // K/V tile of at least max_k_step (min = 16) worth of shm. Mirrors the
  // constructor arithmetic (see flash_attention.h). Falling below 16 would
  // cause the workgroup to over-declare shm and either fail to compile or
  // OOB at runtime.
  const int element_size = is_fp16 ? 2 : 4;
  constexpr int kMinWorkgroupStorageBudgetBytes = 16384;
  const int max_k_from_shm =
      kMinWorkgroupStorageBudgetBytes / (2 * element_size * head_size);
  if (max_k_from_shm < 16) {
    return false;
  }
  // Block-size vs. tile-size alignment: the shader looks up block_table once
  // per K/V tile and reads linearly (see loadk/loadv), which only works when a
  // tile fits entirely inside a single paged block. Mirrors the same
  // max_k_step selection that FlashAttentionPagedPrefillProgram uses (shm
  // path already ruled out above → this is exactly the shm branch).
  const int max_k_step = max_k_from_shm >= 32 ? 32 : 16;
  if (block_size < max_k_step) {
    return false;
  }
  return true;
}

Status RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV(onnxruntime::webgpu::ComputeContext& context,
                                                     const WebgpuAttentionParameters& params,
                                                     const Tensor* packedQKV,
                                                     const Tensor* seqlen_k,
                                                     const Tensor* cos_cache,
                                                     const Tensor* sin_cache,
                                                     Tensor* query,
                                                     Tensor* present_key,
                                                     Tensor* present_value,
                                                     Tensor* indirect_buffer,
                                                     uint32_t tile_size, uint32_t num_q_tiles,
                                                     const Tensor* total_seqlen) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  int components = 1;
  // Currently we only support vectorization when RoPE is not interleaved
  if (!params.rotary_interleaved_) {
    if ((params.head_size_ % 4 == 0) && (half_rotary_embedding_dim % 4 == 0)) {
      components = 4;
    } else if ((params.head_size_ % 2 == 0) && (half_rotary_embedding_dim % 2 == 0)) {
      components = 2;
    }
  }
  // Adjust dimensions for vectorization
  const auto half_rotary_embedding_dim_vec = half_rotary_embedding_dim / components;
  const auto head_size_vec = head_size / components;

  // Dispatch: batch_size * sequence_length * num_heads * (half_rotary_dim + need_copy_dim)
  // work_per_head = half_rotary_dim + (head_size - 2 * half_rotary_dim)
  //               = head_size - half_rotary_dim
  const auto work_per_head = head_size_vec - half_rotary_embedding_dim_vec;
  auto dispatch_size = static_cast<uint32_t>(params.batch_size_ * params.sequence_length_ * params.num_heads_ * work_per_head);

  // Extract present_sequence_length from present_key tensor shape
  const uint32_t present_sequence_length = gsl::narrow_cast<uint32_t>(present_key->Shape()[2]);

  const bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  const uint32_t multi_rotary_cache_concat_offset = context.MultiRotaryCacheConcatOffset();

  SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram program(params.rotary_interleaved_, prepare_indirect_dispatch, multi_rotary_cache_concat_offset);
  program
      .CacheHint(params.rotary_interleaved_, prepare_indirect_dispatch, multi_rotary_cache_concat_offset)
      .AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddInputs({
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::TypeAndRank, components},
          {sin_cache, ProgramTensorMetadataDependency::Rank, components},
      });
  if (prepare_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  program.AddOutputs({{query, ProgramTensorMetadataDependency::None, components},
                      {present_key, ProgramTensorMetadataDependency::None, components},
                      {present_value, ProgramTensorMetadataDependency::None, components}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  program.AddUniformVariables({
      {static_cast<uint32_t>(params.sequence_length_)},
      {static_cast<uint32_t>(params.hidden_size_ / components)},
      {static_cast<uint32_t>(params.kv_hidden_size_ / components)},
      {static_cast<uint32_t>(params.num_heads_)},
      {static_cast<uint32_t>(params.kv_num_heads_)},
      {static_cast<uint32_t>(head_size_vec)},
      {static_cast<uint32_t>(half_rotary_embedding_dim_vec)},
      {present_sequence_length},
      {tile_size},
      {static_cast<uint32_t>(dispatch_size)},
      {static_cast<uint32_t>(params.batch_size_)},
      {num_q_tiles},
      {static_cast<uint32_t>(params.total_sequence_length_)},
  });

  program.SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
