// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/subgroup_matrix_conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Workgroup size is kSubgroupMatrixSubgroupSize * split_k.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

// Max sg_mat_c<mi>_<ni> the WGSL template unrolls (mi < 8, ni < 4); a larger
// tiling would be accepted by the selector but silently mis-computed.
constexpr uint32_t kMaxSgMatCountM = 8;
constexpr uint32_t kMaxSgMatCountN = 4;

// Device's config for the fixed 8x16x16 F16 config this kernel implements, or
// nullopt if unsupported. Single source of truth for eligibility: both
// CanApplySubgroupMatrixConv and CreateSubgroupMatrixConvImpl call this.
std::optional<SubgroupMatrixConfig> GetSupportedConfig(const ComputeContextBase& context) {
  constexpr auto kF16 = wgpu::SubgroupMatrixComponentType::F16;
  const std::optional<SubgroupMatrixConfig> config =
      SelectSubgroupMatrixConfig(context, {{kF16, kF16, 8, 16, 16, 32, false}});
  if (!config || !config->Is(8, 16, 16)) {
    return std::nullopt;
  }
  // Adapters here may expose a 16-32 subgroup size range, but the kernel hard-codes
  // 32 (sg_index = local_idx / 32); pin it rather than risk a wasteful driver choice.
  if (!context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
    return std::nullopt;
  }
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return std::nullopt;
  }
  return config;
}

Status GenerateShaderCode8x16x16(ShaderHelper& shader, const ShaderVariableHelper& input_src,
                                 uint32_t sg_mat_count_m, uint32_t sg_mat_count_n,
                                 uint32_t split_k, uint32_t vec_size) {
  return WGSL_TEMPLATE_APPLY(shader, "nn/subgroup_matrix_conv_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, 16),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, 8),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, 16),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k),
                             WGSL_TEMPLATE_PARAMETER(vec_size, vec_size),
                             WGSL_TEMPLATE_VARIABLE(input_src, input_src));
}

// Computes a channels-last 2D convolution as an implicit GEMM via
// subgroupMatrixMultiplyAccumulate (only 8x16x16 implemented so far). Runs its own
// Conv shape inference, since Compute runs before auto_pad/output are resolved.
template <bool is_channels_last, bool is_fused>
class SubgroupMatrixConvImpl final : public Conv<is_channels_last, is_fused>::ConvOptImpl {
  using ConvT = Conv<is_channels_last, is_fused>;
  using Base = typename ConvT::ConvOptImpl;

 public:
  SubgroupMatrixConvImpl(const ConvT& parent, SubgroupMatrixConfig config,
                         SubgroupMatrixTilingSelector tiling_selector)
      : Base(parent), config_(config), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const ConvT& parent = this->parent_;
    const ConvAttributes& conv_attrs = parent.ConvAttrs();
    const Activation& activation = parent.ConvActivation();

    // Fast decline for any Conv this path doesn't accept: PrePackInternal only skips
    // prepacking for problems CanApplySubgroupMatrixConv accepts, so a non-null
    // prepacked_kernel_ here (grouped/1x1/etc. convs prepacked for another path)
    // means this one would decline below anyway -- this just avoids the shape
    // inference to get there. The kernel needs OIHW regardless.
    if (parent.PrepackedKernel() != nullptr) {
      return Status::OK();
    }
    const auto* input = context.Input<Tensor>(0);
    const auto* weight = context.Input<Tensor>(1);
    if (input == nullptr || weight == nullptr) {
      return Status::OK();
    }
    const TensorShape& weight_shape = weight->Shape();  // OIHW: [Cout, Cin, kh, kw]
    if (!CanApplySubgroupMatrixConv(context, is_channels_last, weight_shape,
                                    narrow<uint32_t>(conv_attrs.group), weight->DataType())) {
      return Status::OK();
    }

    // Only a rank-4 NHWC activation is served (reshaped Conv1D's 1xW kernel is
    // already rejected by CanApplySubgroupMatrixConv).
    const TensorShape& input_shape = input->Shape();  // NHWC: [batch, input_h, input_w, Cin]
    if (input_shape.NumDimensions() != 4) {
      return Status::OK();
    }
    ConvAttributes::ConvPadVector local_pads(conv_attrs.pads.begin(), conv_attrs.pads.end());
    TensorShapeVector local_dilations(conv_attrs.dilations.begin(), conv_attrs.dilations.end());
    TensorShapeVector local_strides(conv_attrs.strides.begin(), conv_attrs.strides.end());
    TensorShapeVector kernel_spacial_shape_vector;
    ORT_RETURN_IF_ERROR(conv_attrs.ComputeKernelShape(weight_shape, kernel_spacial_shape_vector, false));
    if (local_pads.empty()) {
      local_pads.resize(kernel_spacial_shape_vector.size() * 2, 0);
    }
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_spacial_shape_vector.size(), 1);
    }
    if (local_strides.empty()) {
      local_strides.resize(kernel_spacial_shape_vector.size(), 1);
    }
    TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
    TensorShapeVector output_shape_vector = {input_shape[0]};
    TensorShape src_spacial_shape = TensorShape(
        TensorShapeVector(std::next(input_shape_vector.begin()), std::prev(input_shape_vector.end())));
    ORT_RETURN_IF_ERROR(conv_attrs.InferPadsAndOutputShape(src_spacial_shape, kernel_spacial_shape_vector,
                                                           local_strides, local_dilations, local_pads,
                                                           output_shape_vector));
    output_shape_vector.push_back(weight_shape[0]);  // Cout
    const TensorShape output_shape(output_shape_vector);  // NHWC: [batch, out_h, out_w, Cout]

    // static_cast, not narrow: an input the normal Conv path accepts must not abort here.
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
    std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
    std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
    std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);
    if (dilations.size() < 2 || pads.size() < 2 || strides.size() < 2) {
      return Status::OK();
    }

    const uint32_t batch = narrow<uint32_t>(input_shape[0]);
    const uint32_t input_h = narrow<uint32_t>(input_shape[1]);
    const uint32_t input_w = narrow<uint32_t>(input_shape[2]);
    const uint32_t cin = narrow<uint32_t>(weight_shape[1]);
    const uint32_t kernel_h = narrow<uint32_t>(weight_shape[2]);
    const uint32_t kernel_w = narrow<uint32_t>(weight_shape[3]);
    const uint32_t output_h = narrow<uint32_t>(output_shape[1]);
    const uint32_t output_w = narrow<uint32_t>(output_shape[2]);

    const uint32_t M = output_h * output_w;                // output pixels per batch slice
    const uint32_t K = kernel_h * kernel_w * cin;          // contraction dim
    const uint32_t N = narrow<uint32_t>(weight_shape[0]);  // Cout

    // Empty batch/M/N (ONNX permits zero-length dims): let the generic Conv paths
    // handle it instead of dispatching a zero-workgroup kernel.
    if (batch == 0 || M == 0 || N == 0) {
      return Status::OK();
    }

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return Status::OK();
    }

    // ClampTileNToProblem already guarantees tile_n <= N; mirrored here as the last
    // possible decline point before output allocation/weight transpose commit below.
    if (N < tiling->tile_n) {
      return Status::OK();
    }

    Tensor* output = context.Output(0, output_shape);

    // Transpose weight OIHW -> OHWI = [Cout, kh, kw, Cin] (B column-major, B[k,n] at
    // n*K+k). Constant weights are cached; non-constant ones are redone every Run.
    //
    // TODO: prepack the constant case too (PrePackIm2ColMatMulWeight already produces
    // this layout) -- blocked on PrePackInternal committing from the weight alone
    // while Compute can still decline at runtime onto the im2col path.
    Tensor run_local_ohwi_weight;
    const Tensor* ohwi_weight = nullptr;
    if (parent.IsWeightConstant()) {
      ORT_RETURN_IF_ERROR(EnsureTransposedKernel(context, weight, weight_shape));
      ohwi_weight = cached_ohwi_weight_.get();
    } else {
      ORT_RETURN_IF_ERROR(
          TransposeKernel(context, weight, weight_shape, &run_local_ohwi_weight, {0, 2, 3, 1}));
      ohwi_weight = &run_local_ohwi_weight;
    }

    const auto& config = config_;
    const uint32_t tile_m = tiling->tile_m;
    const uint32_t tile_n = tiling->tile_n;
    ORT_ENFORCE(tile_m % config.M == 0 && tile_n % config.N == 0,
                "Tiling must be a multiple of the subgroup-matrix shape: ",
                tile_m, "x", tile_n, " vs ", config.M, "x", config.N);
    const uint32_t sg_mat_count_m = tile_m / config.M;
    const uint32_t sg_mat_count_n = tile_n / config.N;
    ORT_ENFORCE(sg_mat_count_m <= kMaxSgMatCountM && sg_mat_count_n <= kMaxSgMatCountN,
                "Tiling exceeds the subgroup-matrix counts the kernel unrolls: ",
                tile_m, "x", tile_n, " spans ", sg_mat_count_m, "x", sg_mat_count_n,
                " subgroup matrices, at most ", kMaxSgMatCountM, "x", kMaxSgMatCountN);

    const uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    const uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    // Vectorize the im2col gather along cin (contiguous in NHWC).
    const uint32_t vec_size = cin % 4 == 0 ? 4 : (cin % 2 == 0 ? 2 : 1);

    const Tensor* bias = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;
    const bool has_bias = bias != nullptr;
    const uint32_t split_k = tiling->split_k;
    SubgroupMatrixConvProgram program{has_bias, config, sg_mat_count_m, sg_mat_count_n, split_k, vec_size,
                                      activation};
    program.SetWorkgroupSize(kSubgroupMatrixSubgroupSize * split_k);
    program.SetSubgroupSize(kSubgroupMatrixSubgroupSize);
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(has_bias, static_cast<uint32_t>(config.componentType),
                      static_cast<uint32_t>(config.resultComponentType),
                      config.M, config.N, config.K, config.subgroupSize,
                      sg_mat_count_m, sg_mat_count_n, split_k, vec_size,
                      activation.CacheKey())
        .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(vec_size)},
                    {ohwi_weight, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output_shape, 1})
        .AddUniformVariables({{M},
                              {N},
                              {K},
                              {dispatch_x},
                              {batch},
                              {input_h},
                              {input_w},
                              {cin},
                              {kernel_h},
                              {kernel_w},
                              {output_w},
                              {dilations},
                              {pads},
                              {strides}});
    // Activation uniforms must remain last because definitions and values are matched by index.
    AppendActivationUniformsData(activation, program);
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    handled = true;
    return Status::OK();
  }

 private:
  // Builds the OIHW -> OHWI transpose on first use and caches it; call_once guards
  // against concurrent Compute calls on the same (shared) impl.
  Status EnsureTransposedKernel(ComputeContext& context, const Tensor* weight,
                                const TensorShape& weight_shape) const {
    std::call_once(transpose_once_, [&]() {
      auto transposed = std::make_unique<Tensor>();
      if (TransposeKernel(context, weight, weight_shape, transposed.get(), {0, 2, 3, 1}).IsOK()) {
        cached_ohwi_weight_ = std::move(transposed);
      }
    });
    return cached_ohwi_weight_ ? Status::OK()
                               : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                                 "Failed to transpose the subgroup-matrix Conv weight to OHWI.");
  }

  const SubgroupMatrixConfig config_;
  SubgroupMatrixTilingSelector tiling_selector_;

  // OIHW -> OHWI transposed constant weight, built once by EnsureTransposedKernel.
  mutable std::once_flag transpose_once_;
  mutable std::unique_ptr<Tensor> cached_ohwi_weight_;
};

}  // namespace

Status SubgroupMatrixConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_src = shader.AddInput("input_src", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput(
      "output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // Write-out epilogue the template calls per output element: bias add, then the
  // fused activation.
  auto& additional_implementation = shader.AdditionalImplementation();
  additional_implementation
      << GetActivationDeclaration(activation_, "output_value_t", "output_element_t")
      << "fn write_output(output_offset: u32, bias_offset: u32, value_in: output_value_t) {\n"
         "  var value = value_in;\n";
  if (has_bias_) {
    additional_implementation << "  value += output_value_t(bias[bias_offset]);\n";
  }
  additional_implementation
      << "  " << GetActivationSnippet(activation_, "output_value_t", "output_element_t") << "\n"
      << "  " << output.SetByOffset("output_offset", "value") << "\n"
      << "}\n";

  const auto& config = config_;
  if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, input_src, sg_mat_count_m_, sg_mat_count_n_, split_k_, vec_size_);
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

bool CanApplySubgroupMatrixConv(ComputeContextBase& context,
                                bool is_channels_last,
                                const TensorShape& kernel_shape,
                                uint32_t group,
                                MLDataType data_type) {
  if (data_type != DataTypeImpl::GetType<MLFloat16>()) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: data_type is not F16.";
    return false;
  }
  if (!is_channels_last || group != 1) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: requires channels_last && group==1, but got"
                          << " is_channels_last=" << is_channels_last
                          << " group=" << group << ".";
    return false;
  }
  if (kernel_shape.NumDimensions() != 4) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: kernel is not 4D (rank="
                          << kernel_shape.NumDimensions() << ", shape=" << kernel_shape.ToString() << ").";
    return false;
  }

  const uint32_t channel_output = narrow<uint32_t>(kernel_shape[0]);  // Cout (GEMM N)
  const uint32_t channel_input = narrow<uint32_t>(kernel_shape[1]);   // Cin
  const uint32_t kernel_height = narrow<uint32_t>(kernel_shape[2]);
  const uint32_t kernel_width = narrow<uint32_t>(kernel_shape[3]);

  // Mirrors the im2col-matmul path: 1xW/Hx1/1x1 kernels (incl. reshaped Conv1D) are
  // handled better by the plain MatMul/naive conv paths.
  if (kernel_height == 1 || kernel_width == 1) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: 1xW/Hx1/1x1 kernel not handled (kernel_shape[OIHW]="
                          << kernel_shape.ToString() << ", kh=" << kernel_height << " kw=" << kernel_width << ").";
    return false;
  }

  const uint32_t K = kernel_height * kernel_width * channel_input;
  const uint32_t N = channel_output;
  // K: the kernel walks it in sg_mat_k blocks with no tail handling, so it must be a
  // multiple of 16. N: B is loaded unchecked, shifting a trailing partial tile back
  // to N - tile_n, which needs one full 16-wide block to exist; must agree with the
  // N guard in CreateSubgroupMatrixConvTilingSelector.
  if (K == 0 || K % 16 != 0 || N < 16) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: K/N constraints not met (kernel_shape[OIHW]="
                          << kernel_shape.ToString() << ", Cout(N)=" << N << " Cin=" << channel_input
                          << " kh=" << kernel_height << " kw=" << kernel_width << " K=kh*kw*Cin=" << K
                          << "). Need K%16==0 (K%16=" << (K % 16) << ") and N>=16.";
    return false;
  }

  // subgroupMatrixLoad addresses the weight as a single binding; a weight past
  // maxStorageBufferBindingSize gets split by ProgramManager, which this offset
  // arithmetic doesn't know about.
  const uint64_t max_weight_elems = context.DeviceLimits().maxStorageBufferBindingSize / sizeof(MLFloat16);
  if (static_cast<uint64_t>(kernel_shape.Size()) > max_weight_elems) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: weight exceeds maxStorageBufferBindingSize"
                          << " (kernel_shape[OIHW]=" << kernel_shape.ToString() << ", elems="
                          << kernel_shape.Size() << ", limit_elems=" << max_weight_elems
                          << ", maxStorageBufferBindingSize=" << context.DeviceLimits().maxStorageBufferBindingSize
                          << " bytes).";
    return false;
  }

  const std::optional<SubgroupMatrixConfig> config = GetSupportedConfig(context);
  if (!config) {
    LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv rejected: device does not report the required 8x16x16 F16"
                          << " subgroup-matrix config, lacks subgroup-size control, or vendor != intel."
                          << " vendor=\"" << std::string_view{context.AdapterInfo().vendor}
                          << "\" architecture=\"" << std::string_view{context.AdapterInfo().architecture}
                          << "\" subgroup_size_control="
                          << context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)
                          << " subgroup_size_range=[" << context.AdapterInfo().subgroupMinSize << ", "
                          << context.AdapterInfo().subgroupMaxSize << "].";
    return false;
  }

  LOGS_DEFAULT(VERBOSE) << "SubgroupMatrixConv ACCEPTED: kernel_shape[OIHW]=" << kernel_shape.ToString()
                        << " Cout(N)=" << N << " Cin=" << channel_input << " kh=" << kernel_height
                        << " kw=" << kernel_width << " K=" << K << ".";
  return true;
}

template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context) {
  if constexpr (!is_channels_last) {
    ORT_UNUSED_PARAMETER(parent);
    ORT_UNUSED_PARAMETER(context);
    return nullptr;
  } else {
    const std::optional<SubgroupMatrixConfig> config = GetSupportedConfig(context);
    if (!config) {
      return nullptr;
    }
    SubgroupMatrixTilingSelector tiling_selector = intel::CreateSubgroupMatrixConvTilingSelector(context);
    if (!tiling_selector) {
      return nullptr;
    }
    return std::make_unique<SubgroupMatrixConvImpl<is_channels_last, is_fused>>(parent, *config,
                                                                               std::move(tiling_selector));
  }
}

// Explicit instantiation for every Conv instantiation (see conv.cc).
#define WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(CHANNELS_LAST, FUSED)               \
  template std::unique_ptr<Conv<CHANNELS_LAST, FUSED>::ConvOptImpl>                            \
  CreateSubgroupMatrixConvImpl<CHANNELS_LAST, FUSED>(const Conv<CHANNELS_LAST, FUSED>& parent, \
                                                     const ComputeContextBase& context);

WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, true)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, true)

#undef WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL

}  // namespace webgpu
}  // namespace onnxruntime
