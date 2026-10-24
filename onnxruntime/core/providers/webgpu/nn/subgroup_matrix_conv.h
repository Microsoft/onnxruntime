// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Implicit-GEMM channels-last Conv via subgroupMatrixMultiplyAccumulate.
// sg_mat_count_m/n: subgroup matrices per output tile along M/N. split_k:
// subgroups cooperatively reducing K.
class SubgroupMatrixConvProgram final : public Program<SubgroupMatrixConvProgram> {
 public:
  SubgroupMatrixConvProgram(bool has_bias, SubgroupMatrixConfig config,
                            uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k,
                            uint32_t vec_size, const Activation& activation)
      : Program{"SubgroupMatrixConv"},
        has_bias_(has_bias),
        config_(config),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k),
        vec_size_(vec_size),
        activation_(activation) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"num_n_tile", ProgramUniformVariableDataType::Uint32},
                                          {"batch", ProgramUniformVariableDataType::Uint32},
                                          {"input_h", ProgramUniformVariableDataType::Uint32},
                                          {"input_w", ProgramUniformVariableDataType::Uint32},
                                          {"cin", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_h", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_w", ProgramUniformVariableDataType::Uint32},
                                          {"output_w", ProgramUniformVariableDataType::Uint32},
                                          {"dilations", ProgramUniformVariableDataType::Uint32},
                                          {"pads", ProgramUniformVariableDataType::Uint32},
                                          {"strides", ProgramUniformVariableDataType::Uint32},
                                          // Activation uniforms must remain last because definitions and values
                                          // are matched by index (see AppendActivationUniformsData).
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  const bool has_bias_;
  const SubgroupMatrixConfig config_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
  const uint32_t vec_size_;
  const Activation activation_;
};

// Eligibility: fp16, channels-last, group == 1, a 4D non-1x1 kernel, a device
// reporting the 8x16x16 F16 subgroup-matrix config, and K = kernel_h * kernel_w *
// Cin a multiple of 16, N = Cout >= 16. Callers fall back to the default conv
// paths when this returns false.
bool CanApplySubgroupMatrixConv(ComputeContextBase& context,
                                bool is_channels_last,
                                const TensorShape& kernel_shape,
                                uint32_t group,
                                MLDataType data_type);

// Nullptr on devices with no vendor tiling policy; callers fall back to normal Conv.
template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context);

}  // namespace webgpu
}  // namespace onnxruntime
