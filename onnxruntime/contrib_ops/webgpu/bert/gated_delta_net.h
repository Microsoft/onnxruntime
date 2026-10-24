// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

enum class GatedDeltaNetUpdateRule {
  Linear,
  Gated,
  Delta,
  GatedDelta,
  Invalid,
};

struct GatedDeltaNetParallelPrefillPlan {
  uint32_t chunks_per_pass;
  uint64_t workspace_bytes;
};

inline uint64_t GatedDeltaNetChunkCapacity(uint64_t total_tokens, uint64_t batch,
                                           uint32_t chunk_size, bool ragged) {
  if (ragged) return total_tokens / chunk_size + batch;
  return batch * ((total_tokens / batch + chunk_size - 1) / chunk_size);
}

// The prepare and output passes each keep one state-shaped tile per live chunk. Two
// additional state tiles ping-pong the recurrent carry between passes.
inline std::optional<GatedDeltaNetParallelPrefillPlan> SelectGatedDeltaNetParallelPrefillPlan(
    uint64_t state_elements, uint32_t total_chunks, uint64_t workspace_cap_bytes = 64ull << 20) {
  if (total_chunks < 2 || state_elements == 0 ||
      state_elements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return std::nullopt;
  }

  const uint64_t state_bytes = state_elements * sizeof(float);
  if (state_bytes > workspace_cap_bytes / 2) {
    return std::nullopt;
  }

  const uint64_t fixed_bytes = 2 * state_bytes;
  const uint64_t bytes_per_chunk = 2 * state_bytes;
  if (bytes_per_chunk == 0 || fixed_bytes >= workspace_cap_bytes) {
    return std::nullopt;
  }

  const uint64_t max_chunks = (workspace_cap_bytes - fixed_bytes) / bytes_per_chunk;
  if (max_chunks == 0) {
    return std::nullopt;
  }

  const uint32_t chunks_per_pass =
      static_cast<uint32_t>(std::min<uint64_t>(total_chunks, max_chunks));
  const uint64_t workspace_bytes = fixed_bytes + chunks_per_pass * bytes_per_chunk;
  return GatedDeltaNetParallelPrefillPlan{chunks_per_pass, workspace_bytes};
}

class GatedDeltaNetProgram final : public Program<GatedDeltaNetProgram> {
 public:
  GatedDeltaNetProgram(GatedDeltaNetUpdateRule update_rule, bool has_cu_seqlens, bool has_initial_state,
                       bool initial_state_in_final_state, bool output_final_state, bool qwen_gate,
                       bool sigmoid_beta, bool qk_l2_norm, bool use_packed_params, bool capture_state_updates,
                       bool vectorized_value_io)
      : Program{"GatedDeltaNet"},
        update_rule_(update_rule),
        has_cu_seqlens_(has_cu_seqlens),
        has_initial_state_(has_initial_state),
        initial_state_in_final_state_(initial_state_in_final_state),
        output_final_state_(output_final_state),
        qwen_gate_(qwen_gate),
        sigmoid_beta_(sigmoid_beta),
        qk_l2_norm_(qk_l2_norm),
        use_packed_params_(use_packed_params),
        capture_state_updates_(capture_state_updates),
        vectorized_value_io_(vectorized_value_io) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"state_update_capacity", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  GatedDeltaNetUpdateRule update_rule_;
  bool has_cu_seqlens_;
  bool has_initial_state_;
  bool initial_state_in_final_state_;
  bool output_final_state_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool qk_l2_norm_;
  bool use_packed_params_;
  bool capture_state_updates_;
  bool vectorized_value_io_;
};

class GatedDeltaNetClearProgram final : public Program<GatedDeltaNetClearProgram> {
 public:
  GatedDeltaNetClearProgram() : Program{"GatedDeltaNetClear"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"element_count", ProgramUniformVariableDataType::Uint32});
};

class GatedDeltaNetPrefillPrepareProgram final : public Program<GatedDeltaNetPrefillPrepareProgram> {
 public:
  GatedDeltaNetPrefillPrepareProgram() : Program{"GatedDeltaNetPrefillPrepare"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"chunk_base", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32});
};

class GatedDeltaNetPrefillScanProgram final : public Program<GatedDeltaNetPrefillScanProgram> {
 public:
  GatedDeltaNetPrefillScanProgram(bool has_initial_state, bool has_carry_state, bool output_final_state)
      : Program{"GatedDeltaNetPrefillScan"},
        has_initial_state_(has_initial_state),
        has_carry_state_(has_carry_state),
        output_final_state_(output_final_state) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32},
      {"is_last_pass", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_initial_state_;
  bool has_carry_state_;
  bool output_final_state_;
};

class GatedDeltaNetPrefillOutputProgram final : public Program<GatedDeltaNetPrefillOutputProgram> {
 public:
  GatedDeltaNetPrefillOutputProgram() : Program{"GatedDeltaNetPrefillOutput"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"chunk_size", ProgramUniformVariableDataType::Uint32},
      {"chunk_base", ProgramUniformVariableDataType::Uint32},
      {"chunks_in_pass", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});
};

class GatedDeltaNetParamsProgram final : public Program<GatedDeltaNetParamsProgram> {
 public:
  GatedDeltaNetParamsProgram(bool has_decay, bool has_beta, bool qwen_gate, bool sigmoid_beta,
                             bool has_gate_projections = false)
      : Program{"GatedDeltaNetParams"},
        has_decay_(has_decay),
        has_beta_(has_beta),
        qwen_gate_(qwen_gate),
        sigmoid_beta_(sigmoid_beta),
        has_gate_projections_(has_gate_projections) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_decay_;
  bool has_beta_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool has_gate_projections_;
};

class GatedDeltaNetCopyProgram final : public Program<GatedDeltaNetCopyProgram> {
 public:
  GatedDeltaNetCopyProgram() : Program{"GatedDeltaNetCopy"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"element_count", ProgramUniformVariableDataType::Uint32});
};

class GatedDeltaNetUnpackQkvProgram final : public Program<GatedDeltaNetUnpackQkvProgram> {
 public:
  GatedDeltaNetUnpackQkvProgram() : Program{"GatedDeltaNetUnpackQkv"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"query_size", ProgramUniformVariableDataType::Uint32},
      {"value_size", ProgramUniformVariableDataType::Uint32});
};

class GatedDeltaNetNormalizeProgram final : public Program<GatedDeltaNetNormalizeProgram> {
 public:
  GatedDeltaNetNormalizeProgram() : Program{"GatedDeltaNetNormalize"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32});
};

class GatedDeltaNetChunkPrepareProgram final : public Program<GatedDeltaNetChunkPrepareProgram> {
 public:
  GatedDeltaNetChunkPrepareProgram(bool has_cu_seqlens) : Program{"GatedDeltaNetChunkPrepare"},
                                                          has_cu_seqlens_(has_cu_seqlens) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_q", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"total_chunks", ProgramUniformVariableDataType::Uint32},
      {"chunk_elements", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_cu_seqlens_;
};

class GatedDeltaNetChunkScanProgram final : public Program<GatedDeltaNetChunkScanProgram> {
 public:
  GatedDeltaNetChunkScanProgram(uint32_t head_size, bool has_cu_seqlens, bool has_initial_state,
                                bool state_alias, bool output_final_state)
      : Program{"GatedDeltaNetChunkScan"}, head_size_(head_size), has_cu_seqlens_(has_cu_seqlens), has_initial_state_(has_initial_state), state_alias_(state_alias), output_final_state_(output_final_state) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_tokens", ProgramUniformVariableDataType::Uint32},
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_qk", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"total_chunks", ProgramUniformVariableDataType::Uint32},
      {"chunk_elements", ProgramUniformVariableDataType::Uint32},
      {"scale", ProgramUniformVariableDataType::Float32});

 private:
  uint32_t head_size_;
  bool has_cu_seqlens_;
  bool has_initial_state_;
  bool state_alias_;
  bool output_final_state_;
};

class GatedDeltaNet final : public WebGpuKernel {
 public:
  explicit GatedDeltaNet(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  GatedDeltaNetUpdateRule update_rule_;
  float scale_;
  int state_update_capacity_;
  bool qwen_gate_;
  bool sigmoid_beta_;
  bool qk_l2_norm_;
  bool chunkwise_prefill_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
