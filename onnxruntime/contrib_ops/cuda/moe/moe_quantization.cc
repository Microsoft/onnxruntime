// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "contrib_ops/cuda/moe/moe_quantization.h"
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
#include "contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"
#include "core/framework/kernel_pilot.h"
#endif
#include <charconv>
#include <type_traits>
#include "core/common/float8.h"
#include "cutlass/numeric_types.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_int2_prefill.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
#include "contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.h"
#endif

#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

#include <algorithm>
#include <optional>
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace {
constexpr size_t kMinH200GlobalMemoryBytes = size_t{120} << 30;

void LogQMoESwigluFusionRemapOnce() {
  static std::once_flag log_warning;
  std::call_once(log_warning, []() {
    LOGS_DEFAULT(WARNING) << "QMoE swiglu_fusion is 0; assuming interleaved SwiGLU layout "
                             "for backward compatibility.";
  });
}

void LogQMoEFp4DualLayoutOnce() {
  static std::once_flag log_info;
  std::call_once(log_info, []() {
    LOGS_DEFAULT(INFO) << "QMoE MXFP4: this expert shape is outside the interleaved GEMV rules, so "
                          "the decode GEMV needs its own copy of the e2m1 weights alongside the "
                          "SM80 grouped-GEMM layout. This doubles the persistent MoE weight "
                          "footprint; shapes with k % 64 == 0 and n % 16 == 0 avoid it.";
  });
}

// Per-shape autotune of the fused FP4 GEMV CtaN/Threads tiling. It is opt-in because profiling
// synchronizes the inference stream; set ORT_FP4_GEMV_AUTOTUNE=1 to enable it.
// ORT_FP4_GEMV_AUTOTUNE_LOG=1 logs the chosen configs per shape.
bool Fp4GemvAutotuneEnabled() {
  return onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_AUTOTUNE", 0) == 1;
}

bool Fp4GemvAutotuneLogEnabled() {
  return onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_AUTOTUNE_LOG", 0) == 1;
}

bool Fp4GemvSkipExpandDisabled() {
  return onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_FP4_GEMV_SKIP_EXPAND", 0) == 1;
}

void PrintQMoEKernelDebugInfo(const char* route, int64_t num_rows,
                              int64_t row_tile_size, int64_t final_row_tile_size,
                              int64_t tactic_m, int64_t final_tactic_m,
                              size_t runner_workspace_bytes, size_t total_scratch_bytes) {
  std::cout << "\x1B[36m"
            << "Operator=QMoE"
            << " Route=" << route
            << " Rows=" << num_rows
            << " RowTileSize=" << row_tile_size
            << " RunnerWorkspaceBytes=" << runner_workspace_bytes
            << " TotalScratchBytes=" << total_scratch_bytes
            << " TacticMBucket="
            << onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::bucketM(tactic_m);
  if (final_row_tile_size != row_tile_size) {
    std::cout << " FinalRowTileSize=" << final_row_tile_size
              << " FinalTacticMBucket="
              << onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::bucketM(final_tactic_m);
  }
  std::cout << "\x1B[0m" << std::endl;
}

const char* QMoEGemvConfigName(onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig config) {
  using onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig;
  if (config == MoeGemvConfig::kCtaN16) {
    return "ctan16";
  }
  if (config == MoeGemvConfig::kThreads64) {
    return "threads64";
  }
  return "default";
}

bool TryGetStaticDim(const ONNX_NAMESPACE::TensorShapeProto* shape, int dim_index, int64_t& value) {
  if (shape == nullptr || dim_index >= shape->dim_size() || !shape->dim(dim_index).has_dim_value()) {
    return false;
  }
  value = shape->dim(dim_index).dim_value();
  return value > 0;
}

// Profile-free tactic choice for the dense A16 grouped GEMMs (NVFP4/FP4 dequant route) when ORT_FORCE_DETERMINISTIC_MOE
// pins the tactic. The first entry of the tactic list is a decode tile (CTA 128x16x64) that runs prefill-sized problems at
// ~12% of the tensor-core peak. Picks below were taken from the profiler's choices on H200 for the Qwen3.8 shapes
// (hidden 2560, inter 640, 512 experts, top-10) at 128..8192 tokens per call; rows_per_expert = expanded rows / experts.
// ORT_QMOE_PREFILL_TACTIC=0 restores the first-tactic behaviour.
std::optional<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig> PickPrefillTactic(
    const std::vector<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig>& tactics, bool is_fc2, double rows_per_expert) {
  using onnxruntime::llm::cutlass_extensions::ClusterShape;
  using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;
  using onnxruntime::llm::cutlass_extensions::CutlassTileConfig;
  using onnxruntime::llm::cutlass_extensions::CutlassTileConfigSM90;
  static const bool enabled = onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_TACTIC", 1) != 0;
  if (!enabled || tactics.empty() || rows_per_expert < 4.0) {
    return std::nullopt;
  }
  auto find_tma = [&](CutlassTileConfigSM90 tile, ClusterShape cluster) -> std::optional<CutlassGemmConfig> {
    for (const auto& t : tactics) {
      if (t.is_tma_warp_specialized && t.sm_version == 90 && t.tile_config_sm90 == tile && t.cluster_shape == cluster) return t;
    }
    return std::nullopt;
  };
  auto find_ampere = [&](CutlassTileConfig tile) -> std::optional<CutlassGemmConfig> {
    // QMoE always dispatches with use_fused_moe=false (ENABLE_FUSED_GATED_ACTIVATION is false),
    // which uses DefaultGemmGrouped in moe_gemm_kernels_{fp16,bf16}.cu; under ORT_QUICK_BUILD
    // only stages==4 is instantiated for CtaShape{32,64}x128x64.
    std::optional<CutlassGemmConfig> best;
    for (const auto& t : tactics) {
#if defined(ORT_QUICK_BUILD)
      if (t.stages != 4) continue;
#endif
      if (!t.is_tma_warp_specialized && t.sm_version < 90 && t.tile_config_sm80 == tile &&
          (!best.has_value() || t.stages >= best->stages)) {
        best = t;
      }
    }
    return best;
  };
  std::optional<CutlassGemmConfig> pick;
  if (rows_per_expert >= 100.0) {
    pick = is_fc2 ? find_tma(CutlassTileConfigSM90::CtaShape128x64x128B, ClusterShape::ClusterShape_2x1x1)
                  : find_tma(CutlassTileConfigSM90::CtaShape128x256x128B, ClusterShape::ClusterShape_2x1x1);
  } else if (rows_per_expert >= 24.0) {
    pick = is_fc2 ? find_tma(CutlassTileConfigSM90::CtaShape128x64x128B, ClusterShape::ClusterShape_2x1x1)
                  : find_ampere(CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64);
  } else {
    pick = find_ampere(CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64);
  }
  return pick;
}

bool StaticFp4CutlassShapeSupported(const OpKernelInfo& op_kernel_info, bool is_swiglu) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  if (op_kernel_info.GetInputCount() <= 5) {
    return false;
  }

  const auto fc1_shape = op_kernel_info.GetKernelInfo().GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
  const auto fc2_shape = op_kernel_info.GetKernelInfo().GetInputTypeInfo(5).GetTensorTypeAndShapeInfo().GetShape();
  if (fc1_shape.size() != 3 || fc2_shape.size() != 3) {
    return false;
  }

  const int64_t fc1_k = fc1_shape[1];
  const int64_t fc1_packed_n = fc1_shape[2];
  const int64_t fc2_k = fc2_shape[1];
  const int64_t fc2_packed_n = fc2_shape[2];
  if (fc1_k <= 0 || fc1_packed_n <= 0 || fc2_k <= 0 || fc2_packed_n <= 0) {
    return false;
  }
#else
  const auto& input_defs = op_kernel_info.node().InputDefs();
  if (input_defs.size() <= 5 || input_defs[2] == nullptr || input_defs[5] == nullptr) {
    return false;
  }

  const auto* fc1_shape = input_defs[2]->Shape();
  const auto* fc2_shape = input_defs[5]->Shape();
  if (fc1_shape == nullptr || fc2_shape == nullptr || fc1_shape->dim_size() != 3 || fc2_shape->dim_size() != 3) {
    return false;
  }

  int64_t fc1_k = 0;
  int64_t fc1_packed_n = 0;
  int64_t fc2_k = 0;
  int64_t fc2_packed_n = 0;
  if (!TryGetStaticDim(fc1_shape, 1, fc1_k) || !TryGetStaticDim(fc1_shape, 2, fc1_packed_n) ||
      !TryGetStaticDim(fc2_shape, 1, fc2_k) || !TryGetStaticDim(fc2_shape, 2, fc2_packed_n)) {
    return false;
  }
#endif

  const int64_t hidden_size = fc1_k;
  const int64_t hidden_size_from_fc2 = fc2_packed_n * 2;
  const int64_t fc1_n = fc1_packed_n * 2;
  if (hidden_size != hidden_size_from_fc2 || (is_swiglu && fc1_n % 2 != 0)) {
    return false;
  }

  const int64_t inter_size = fc2_k;
  const int64_t inter_size_from_fc1 = is_swiglu ? fc1_n / 2 : fc1_n;
  if (inter_size != inter_size_from_fc1) {
    return false;
  }

  constexpr int64_t kMxfp4CutlassAlignment = 64;
  return hidden_size % kMxfp4CutlassAlignment == 0 && inter_size % kMxfp4CutlassAlignment == 0;
}

// Returns the per-rank expert count DeepGEMM would run, or 0 if the static shapes rule it out.
int StaticFp4DeepGemmNumExperts(const OpKernelInfo& op_kernel_info) {
#if !defined(HAS_SM90_OR_LATER) || !defined(USE_DEEP_GEMM)
  ORT_UNUSED_PARAMETER(op_kernel_info);
  return 0;
#else
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  if (op_kernel_info.GetInputCount() <= 5) {
    return 0;
  }

  const auto fc1_shape = op_kernel_info.GetKernelInfo().GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
  const auto fc2_shape = op_kernel_info.GetKernelInfo().GetInputTypeInfo(5).GetTensorTypeAndShapeInfo().GetShape();
  if (fc1_shape.size() != 3 || fc2_shape.size() != 3) {
    return 0;
  }

  const int64_t fc1_e = fc1_shape[0];
  const int64_t fc1_k = fc1_shape[1];
  const int64_t fc1_packed_n = fc1_shape[2];
  const int64_t fc2_e = fc2_shape[0];
  const int64_t fc2_k = fc2_shape[1];
  const int64_t fc2_packed_n = fc2_shape[2];
#else
  const auto& input_defs = op_kernel_info.node().InputDefs();
  if (input_defs.size() <= 5 || input_defs[2] == nullptr || input_defs[5] == nullptr) {
    return 0;
  }

  const auto* fc1_shape = input_defs[2]->Shape();
  const auto* fc2_shape = input_defs[5]->Shape();
  int64_t fc1_e = 0;
  int64_t fc1_k = 0;
  int64_t fc1_packed_n = 0;
  int64_t fc2_e = 0;
  int64_t fc2_k = 0;
  int64_t fc2_packed_n = 0;
  if (fc1_shape == nullptr || fc2_shape == nullptr || fc1_shape->dim_size() != 3 || fc2_shape->dim_size() != 3 ||
      !TryGetStaticDim(fc1_shape, 0, fc1_e) || !TryGetStaticDim(fc1_shape, 1, fc1_k) ||
      !TryGetStaticDim(fc1_shape, 2, fc1_packed_n) || !TryGetStaticDim(fc2_shape, 0, fc2_e) ||
      !TryGetStaticDim(fc2_shape, 1, fc2_k) || !TryGetStaticDim(fc2_shape, 2, fc2_packed_n)) {
    return 0;
  }
#endif

  const bool supported =
      fc1_e == onnxruntime::llm::kernels::deep_gemm_sm90::kNumExpertsWorld8 && fc2_e == fc1_e &&
      fc1_k == onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize &&
      fc1_packed_n * 2 == onnxruntime::llm::kernels::deep_gemm_sm90::kFc1OutputSize &&
      fc2_k == onnxruntime::llm::kernels::deep_gemm_sm90::kInterSize &&
      fc2_packed_n * 2 == onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize;
  return supported ? static_cast<int>(fc1_e) : 0;
#endif
}
}  // namespace

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      QMoE,                                                                    \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(0, 0)                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(),       \
                                 DataTypeImpl::GetTensorType<Float8E4M3FN>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),         \
                                 DataTypeImpl::GetTensorType<MLFloat16>(),     \
                                 DataTypeImpl::GetTensorType<BFloat16>(),      \
                                 DataTypeImpl::GetTensorType<Float8E8M0>(),    \
                                 DataTypeImpl::GetTensorType<Float8E4M3FN>()}) \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<float>()),         \
      QMoE);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  enable_kernel_debug_info_ =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>(kEnableQMoEKernelDebugInfo, 0) != 0;
  const auto configured_row_tile_size =
      op_kernel_info.GetConfigOptions().GetConfigEntry(kQMoERowTileSizeConfig);
  const char* row_tile_size_source = kQMoERowTileSizeConfig;
  if (configured_row_tile_size.has_value()) {
    const auto& value = *configured_row_tile_size;
    const auto parse_result = std::from_chars(value.data(), value.data() + value.size(), row_tile_size_);
    ORT_ENFORCE(parse_result.ec == std::errc{} && parse_result.ptr == value.data() + value.size(),
                kQMoERowTileSizeConfig, " must be an integer, got '", value, "'.");
  } else {
    row_tile_size_source = kQMoERowTileSize;
    row_tile_size_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
        kQMoERowTileSize, qmoe::kDisabledRowTileSize);
  }
  ORT_ENFORCE(row_tile_size_ == qmoe::kDisabledRowTileSize || qmoe::IsValidRowTileSize(row_tile_size_),
              row_tile_size_source, " must be 0 (disabled) or an integer in [1, ",
              std::numeric_limits<int>::max(), "], got ", row_tile_size_, ".");
  const auto configured_int_dequant_max_scratch_bytes =
      op_kernel_info.GetConfigOptions().GetConfigEntry(kQMoEIntDequantMaxScratchBytesConfig);
  const char* int_dequant_max_scratch_bytes_source = kQMoEIntDequantMaxScratchBytesConfig;
  if (configured_int_dequant_max_scratch_bytes.has_value()) {
    const auto& value = *configured_int_dequant_max_scratch_bytes;
    const auto parse_result = std::from_chars(
        value.data(), value.data() + value.size(), int_dequant_max_scratch_bytes_);
    ORT_ENFORCE(parse_result.ec == std::errc{} && parse_result.ptr == value.data() + value.size(),
                kQMoEIntDequantMaxScratchBytesConfig, " must be an integer, got '", value, "'.");
  } else {
    int_dequant_max_scratch_bytes_source = kQMoEIntDequantMaxScratchBytes;
    int_dequant_max_scratch_bytes_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
        kQMoEIntDequantMaxScratchBytes, int_dequant_max_scratch_bytes_);
  }
  ORT_ENFORCE(int_dequant_max_scratch_bytes_ > 0,
              int_dequant_max_scratch_bytes_source, " must be greater than 0, got ",
              int_dequant_max_scratch_bytes_, ".");
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4 || expert_weight_bits_ == 2,
              "expert_weight_bits must be 2, 4, or 8, but got ", expert_weight_bits_);
  fc1_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc1_expert_weight_bits", expert_weight_bits_);
  fc2_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc2_expert_weight_bits", expert_weight_bits_);
  fc3_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc3_expert_weight_bits", expert_weight_bits_);
  ORT_ENFORCE((fc1_expert_weight_bits_ == 2 || fc1_expert_weight_bits_ == 4 || fc1_expert_weight_bits_ == 8) &&
                  (fc2_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 4 || fc2_expert_weight_bits_ == 8) &&
                  (fc3_expert_weight_bits_ == 2 || fc3_expert_weight_bits_ == 4 || fc3_expert_weight_bits_ == 8),
              "FC-specific expert weight bits must be 2, 4, or 8.");
  ORT_ENFORCE(swiglu_fusion_ == 0 || fc3_expert_weight_bits_ == fc1_expert_weight_bits_,
              "Fused SwiGLU requires FC1 and FC3 expert weight bits to match.");
  const bool is_mixed_width = fc1_expert_weight_bits_ != expert_weight_bits_ ||
                              fc2_expert_weight_bits_ != expert_weight_bits_ ||
                              fc3_expert_weight_bits_ != expert_weight_bits_;

  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);
  this->quant_type_ = op_kernel_info.GetAttrOrDefault<std::string>("quant_type", "int");
  ORT_ENFORCE(quant_type_ == "int" || quant_type_ == "fp4" || quant_type_ == "nvfp4" ||
                  quant_type_ == "fp8" || quant_type_ == "wfp4afp8",
              "quant_type must be 'int', 'fp4', 'nvfp4', 'fp8', or 'wfp4afp8', but got '", quant_type_, "'");
  const bool use_int_dequant_fallback =
      quant_type_ == "int" && (is_mixed_width || expert_weight_bits_ == 2);
  enable_int2_gemv_ = quant_type_ == "int" &&
                      (fc1_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 2) &&
                      fc1_expert_weight_bits_ <= 4 && fc2_expert_weight_bits_ <= 4 &&
                      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_GEMV", 0) != 1;
  if (quant_type_ == "nvfp4") {
    constexpr int64_t kNvfp4BlockSize = 16;
    ORT_ENFORCE(block_size_ == -1 || block_size_ == kNvfp4BlockSize,
                "QMoE quant_type='nvfp4' requires block_size=", kNvfp4BlockSize,
                ", but got ", block_size_);
    block_size_ = kNvfp4BlockSize;
  } else if (quant_type_ == "fp4" || quant_type_ == "wfp4afp8") {
    constexpr int64_t kMxfp4BlockSize = 32;
    ORT_ENFORCE(block_size_ == -1 || block_size_ == kMxfp4BlockSize,
                "QMoE quant_type='", quant_type_, "' requires block_size=", kMxfp4BlockSize,
                ", but got ", block_size_);
    block_size_ = kMxfp4BlockSize;
  }
  // ``weights_prepacked`` is an optional tri-state attribute (default -1) that
  // declares the layout of the int4/int8 fc1/fc2 weight initializers. The
  // concrete prepacked layouts selected by -1 and 1 are determined by the
  // execution provider. The CUDA EP maps the tri-state as:
  //   -1 (default): already prepacked in the EP's default int weight layout.
  //    1: already prepacked in an alternate EP-selected int weight layout.
  //    0: raw [E, N, K/pack] initializers; the PrePack hook lays them out.
  //
  // Important: the CUDA QMoE int4/int8 MoE GEMM always dispatches to the
  // Ampere (SM80) grouped-GEMM kernel -- even on SM90 -- because mixed
  // int-weight + fp16/bf16 activation is not a valid Hopper TMA warp-specialized
  // specialisation (see isValidHopperMOESpecialisation). The kernel therefore
  // consumes the SM80/Ampere CUTLASS fpA_intB layout on every GPU. As a result
  // the EP default (-1) is the SM80 layout regardless of the runtime device SM,
  // and SM80-format weights are valid on SM90 (they run via the SM80 kernel).
  // For CUDA today, -1 and 1 are equivalent (both SM80 layout), and 1 is
  // reserved for a possible future Hopper-specific layout.
  // PrePack (weights_prepacked=0) packs for the SM80 layout accordingly.
  const int64_t weights_prepacked_mode =
      op_kernel_info.GetAttrOrDefault<int64_t>("weights_prepacked", static_cast<int64_t>(-1));
  ORT_ENFORCE(weights_prepacked_mode == -1 || weights_prepacked_mode == 0 || weights_prepacked_mode == 1,
              "weights_prepacked must be -1 (default), 0, or 1, but got ", weights_prepacked_mode);
  weights_prepacked_ = (weights_prepacked_mode != 0);
  if (use_int_dequant_fallback) {
    ORT_ENFORCE(!weights_prepacked_,
                "INT2 or mixed-width CUDA QMoE requires raw weights with weights_prepacked=0.");
    ORT_ENFORCE(block_size_ >= 16 && block_size_ <= 256 &&
                    (block_size_ & (block_size_ - 1)) == 0,
                "INT2 or mixed-width CUDA QMoE requires block_size to be a power of two in [16, 256], got ",
                block_size_, ".");
  }
#if !defined(ENABLE_FP4) || !defined(USE_FP4_QMOE)
  ORT_ENFORCE(quant_type_ != "fp4", "QMoE quant_type='fp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
  ORT_ENFORCE(quant_type_ != "nvfp4", "QMoE quant_type='nvfp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
  ORT_ENFORCE(quant_type_ != "wfp4afp8",
              "QMoE quant_type='wfp4afp8' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
#endif
#if !defined(ENABLE_FP8) || !defined(USE_FP8_QMOE)
  ORT_ENFORCE(quant_type_ != "fp8", "QMoE quant_type='fp8' requires USE_FP8_QMOE with CUDA 11.8 or newer.");
  ORT_ENFORCE(quant_type_ != "wfp4afp8", "QMoE quant_type='wfp4afp8' requires USE_FP8_QMOE with CUDA 11.8 or newer.");
#endif
  enable_fp8_fused_ = quant_type_ == "fp8" && sm_ >= 80 &&
                      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP8_FUSED", 1) != 0;

  using namespace onnxruntime::llm::kernels::cutlass_kernels;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  auto input_type = op_kernel_info.GetKernelInfo().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  bool is_fp16 = input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
#else
  int32_t input_type = op_kernel_info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  bool is_fp16 = input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
#endif
  is_fp16_ = is_fp16;
  enable_int2_prefill_ = quant_type_ == "int" && sm_ >= 80 &&
                         (block_size_ == 32 || block_size_ == 64 || block_size_ == 128) &&
                         (fc1_expert_weight_bits_ == 2 || fc1_expert_weight_bits_ == 4) &&
                         (fc2_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 4) &&
                         (fc1_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 2) &&
                         onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_QMOE_INT2_PREFILL", 1) != 0;

  fp4_deep_gemm_num_experts_ = StaticFp4DeepGemmNumExperts(op_kernel_info);
  enable_fp4_deep_gemm_ =
      quant_type_ == "fp4" && !is_fp16_ && sm_ == 90 && GetDeviceProp().multiProcessorCount == 132 &&
      GetDeviceProp().totalGlobalMem >= kMinH200GlobalMemoryBytes && k_ == 6 &&
      activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
      fp4_deep_gemm_num_experts_ > 0 &&
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_FP4_DEEPGEMM", 0) == 1;

  if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8") {
    if (quant_type_ == "fp4") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "FP4 quantization requires expert_weight_bits=4");
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      // The native WFP4A16 grouped GEMM is an sm_90a-only WGMMA kernel (opt-in below); SM120 cannot run it.
      use_fp4_dequant_fallback_ = true;
      const bool requested_fp4_cutlass_gemm =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_CUTLASS_GEMM", 0) == 1;
      const bool allow_unsafe_fp4_cutlass_gemm =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_CUTLASS_UNSAFE", 0) == 1;
      const bool fp4_cutlass_shape_supported = StaticFp4CutlassShapeSupported(
          op_kernel_info, activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu);
      enable_fp4_cutlass_gemm_ = is_fp16_ && sm_ == 90 && requested_fp4_cutlass_gemm &&
                                 allow_unsafe_fp4_cutlass_gemm && fp4_cutlass_shape_supported;
      if (requested_fp4_cutlass_gemm && is_fp16_ && sm_ == 90 && !allow_unsafe_fp4_cutlass_gemm) {
        LOGS_DEFAULT(WARNING) << "QMoE native FP4 CUTLASS GEMM was requested, but it is disabled by default "
                                 "while its SM90 MXFP4 conversion/layout path is still being validated; "
                                 "falling back to the dequant/GEMV path. Set ORT_ENABLE_FP4_CUTLASS_UNSAFE=1 "
                                 "only for debugging the experimental native path.";
      } else if (requested_fp4_cutlass_gemm && is_fp16_ && sm_ == 90 && !fp4_cutlass_shape_supported) {
        LOGS_DEFAULT(WARNING) << "QMoE native FP4 CUTLASS GEMM was requested, but the static FP4 weight "
                                 "shape does not meet the current WFP4A16 alignment requirements; falling "
                                 "back to the dequant/GEMV path.";
      }
      if (enable_fp4_cutlass_gemm_) {
        use_fp4_dequant_fallback_ = false;
      }
      // Fused MXFP4 GEMV (W4A16) decode path for the fallback regime. This is the
      // default: on real decode shapes it is ~18x faster than re-dequantizing all experts to
      // dense BF16/FP16 every token, and it is validated bit-exact against the fallback. Set
      // ORT_ENABLE_FP4_GEMV=0 to force the dequant fallback (e.g. for debugging). Prefill and
      // any unsupported shape still fall through to the dequant path at dispatch time.
      if (use_fp4_dequant_fallback_) {
        enable_fp4_gemv_ =
            onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_GEMV", 1) != 0;
      } else if (enable_fp4_cutlass_gemm_) {
        // Native CUTLASS WFP4A16 scales well with M but is underfilled for decode (small M):
        // route prefill (M >= ORT_FP4_PREFILL_MIN_TOKENS) through native, and small-decode
        // shapes through the fused GEMV. Both weight/scale layouts are pre-packed alongside
        // each other so a single QMoE node can serve both regimes.
        enable_fp4_gemv_ = true;
        fp4_prefill_min_tokens_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_PREFILL_MIN_TOKENS", 64);
        // Above this average per-expert token count, prefill routes to the dense A16 fallback
        // runner instead of the native FP4 grouped GEMM (measured crossover ~128 tokens/expert
        // on H200). 0 disables the upper bound. See ComputeInternal dispatch.
        fp4_native_max_tokens_per_expert_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT", 128);
      }
      // SM80 FP4 grouped GEMM (port of the INT4 fused-dequant Ampere path to MXFP4).
      // Used in the dequant-fallback regime on every SM >= 80, including SM120/SM121, which have no
      // WFP4A16 tensor-core instruction and run the Ampere mma.sync kernel natively.
      // This SM80 grouped GEMM is several times faster at the gpt-oss-20b prefill regime, so it is enabled by DEFAULT for FP16/BF16;
      // set ORT_FP4_SM80_GEMM=0 to fall back to the dequant path.
      // If the user EXPLICITLY
      // requested the native CUTLASS GEMM (ORT_ENABLE_FP4_CUTLASS_GEMM=1) we honor that intent
      // and do not take the SM80 path, including the native-requested-but-shape-unsupported
      // fallback (which then uses the dequant path). The decision is pushed into the runner via
      // setUseSm80Fp4() below.
      // When enabled we force the GEMV prepack (which also produces the SM80 CUTLASS-interleaved
      // e2m1 weights + activation-dtype group scales) and later override the runner to the FP4 runner so
      // prefill can dispatch to the SM80 DqMma grouped GEMM (see moeUseSm80Fp4 in the kernels).
      enable_fp4_sm80_gemm_ =
          use_fp4_dequant_fallback_ && sm_ >= 80 && !requested_fp4_cutlass_gemm &&
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_SM80_GEMM", 1) == 1;
      if (enable_fp4_sm80_gemm_) {
        enable_fp4_gemv_ = true;
        // In this regime every dispatch is served by a pre-packed buffer -- prefill by the SM80
        // interleaved weights, decode by the fused GEMV -- and the dequant fallback (the only
        // consumer of the raw [E, K, N/2] e2m1 initializer) is unreachable. Let PrePack release the
        // initializer instead of pinning ~9 GiB of dead device memory for a 20B-class MXFP4 MoE.
        release_fp4_raw_weights_ = true;
      }
      // Capture the fused MXFP4 GEMV autotune knobs once here (at op-construction time) instead of
      // re-reading the environment on every inference call, so a session's autotune behavior cannot
      // change underneath it if the process mutates the environment after construction (mirrors the
      // ORT_FP4_SM80_GEMM constructor-plumbed decision below).
      enable_fp4_gemv_autotune_ = Fp4GemvAutotuneEnabled();
      enable_fp4_gemv_autotune_log_ = Fp4GemvAutotuneLogEnabled();
      fp4_gemv_skip_expand_ = !Fp4GemvSkipExpandDisabled();
#else
      use_fp4_dequant_fallback_ = true;
#endif
    } else if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "NVFP4 quantization requires expert_weight_bits=4");
      // Native block-scaled CUTLASS FP4xFP4 grouped GEMM for NVFP4 is supported on SM120 and SM121. There the
      // BF16/FP16 activation is quantized to NVFP4 (block size 16, E4M3 block scales) inside the
      // runner's expandInputRowsKernel and the E2M1 weights + E4M3 block scales feed the native
      // block-scaled tensor op. On older GPUs (and for shapes the native tile alignment does not
      // support) NVFP4 falls back to dequantizing E2M1 weights to FP16/BF16 and running the dense
      // A16 MoE runner.
      use_fp4_dequant_fallback_ = true;
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      const bool nvfp4_cutlass_shape_supported = StaticFp4CutlassShapeSupported(
          op_kernel_info, activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu);
      enable_nvfp4_cutlass_gemm_ =
          (sm_ == 120 || sm_ == 121) && nvfp4_cutlass_shape_supported &&
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_NVFP4_CUTLASS_GEMM", 1) == 1;
      if (enable_nvfp4_cutlass_gemm_) {
        use_fp4_dequant_fallback_ = false;
      }
      // Fused FP4 GEMV (W4A16) decode fast path, shared with MXFP4 but with block size 16 and
      // Float8E4M3FN block scales. Default-on (opt-out via ORT_ENABLE_FP4_GEMV=0): small-decode
      // shapes route through the fused GEMV instead of re-dequantizing every expert to dense
      // BF16/FP16 each token, and any unsupported shape (prefill / large batch) falls through to
      // the dequant fallback.
      enable_fp4_gemv_ =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_ENABLE_FP4_GEMV", 1) != 0;
      if (enable_nvfp4_cutlass_gemm_) {
        // Native FP4xFP4 scales well with M but is underfilled for decode (small M): route prefill
        // (M >= ORT_FP4_PREFILL_MIN_TOKENS) through native, and small-decode shapes through the
        // fused GEMV. Both weight/scale layouts are pre-packed alongside each other.
        enable_fp4_gemv_ = true;
        fp4_prefill_min_tokens_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_PREFILL_MIN_TOKENS", 64);
        // Above this average per-expert token count, prefill routes to the dense A16 fallback
        // runner instead of the native FP4xFP4 grouped GEMM. 0 disables the upper bound.
        fp4_native_max_tokens_per_expert_ = onnxruntime::ParseEnvironmentVariableWithDefault<int64_t>(
            "ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT", 0);
      }
      nvfp4_gemv_raw_layout_ =
          onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_NVFP4_GEMV_RAW_LAYOUT", 0) == 1;
      enable_fp4_gemv_autotune_ = !nvfp4_gemv_raw_layout_ && Fp4GemvAutotuneEnabled();
      enable_fp4_gemv_autotune_log_ = Fp4GemvAutotuneLogEnabled();
      fp4_gemv_skip_expand_ = !Fp4GemvSkipExpandDisabled();
#endif
    } else if (quant_type_ == "wfp4afp8") {
      ORT_ENFORCE(expert_weight_bits_ == 4, "WFP4AFP8 (W4A8) quantization requires expert_weight_bits=4");
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE) && defined(ENABLE_FP8)
      // The native FP8 x MXFP4 path uses CUTLASS block-scaled tensor ops which require SM100+ (Blackwell).
      // The activation BF16/FP16 -> FP8 quantization is performed inside the runner's
      // expandInputRowsKernel using the MXFP8 branch: the runner is constructed with T=__nv_fp8_e4m3,
      // InputType=half/bf16, and the QuantParams sets mxfp8_mxfp4.fc{1,2}.weight_block_scale to the MXFP4
      // weight block scales. Activation block scales are written to fc1_fp4_act_scale_ at runtime.
      // On older GPUs we fall back to dequantizing MXFP4 weights to BF16/FP16 and using the A16 runner.
      use_wfp4afp8_dequant_fallback_ = sm_ < 100;
#else
      use_wfp4afp8_dequant_fallback_ = true;
#endif
    } else {
      ORT_ENFORCE(expert_weight_bits_ == 8, "FP8 quantization requires expert_weight_bits=8");
      ORT_ENFORCE(!is_mixed_width, "FP8 quantization requires all FC expert weight bits to be 8.");
      ORT_ENFORCE(block_size_ <= std::numeric_limits<int>::max(), "FP8 block_size exceeds the CUDA integer range.");
      // Native W8A16-FP8 accepts only per-expert scales, not square block scales.
      use_fp8_dequant_fallback_ = sm_ < 90 || block_size_ > 0;
    }
    if (quant_type_ == "fp4" && (!use_fp4_dequant_fallback_ || enable_fp4_sm80_gemm_)) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_, enable_fp4_sm80_gemm_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_, enable_fp4_sm80_gemm_);
      }
      // Dense A16 fallback runner for the large-per-expert-M prefill regime, where dequantizing
      // the MXFP4 weights to FP16/BF16 and running the dense grouped GEMM beats the native FP4
      // grouped GEMM. Constructed alongside the FP4 runner so a single QMoE node can route per
      // call (see fp4_native_max_tokens_per_expert_). Not used by the SM80 grouped-GEMM path
      // (enable_fp4_sm80_gemm_), which routes prefill through the FP4 runner directly.
      if (is_fp16) {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
      ORT_ENFORCE(!m_moe_runner->getTactics().empty(),
                  "QMoE MXFP4: no grouped GEMM kernel for SM", sm_,
                  " is available in this build. Unset ORT_ENABLE_FP4_CUTLASS_GEMM to use the default path.");
#endif
    } else if (quant_type_ == "nvfp4" && !use_fp4_dequant_fallback_) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
      // Native NVFP4 (Blackwell SM120+): FP4 e2m1 activations + FP4 e2m1 weights, BF16/FP16
      // input/output. Template parameters: <T=fp4, WeightType=fp4, OutputType=BF16/FP16,
      // InputType=BF16/FP16>. The runner accepts BF16/FP16 input from the caller and quantizes it
      // to NVFP4 (block size 16, per-block E4M3 scales, global scale 1.0) inside
      // expandInputRowsKernel (NVFP4 branch, triggered by quant_params.fp4.weight_block_scale being
      // non-null). CUTLASS routes this through the SM120 block-scaled tensor op path.
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
      // Dense A16 fallback runner for the large-per-expert-M prefill regime (dequantize E2M1
      // weights to FP16/BF16 and run the dense grouped GEMM). Constructed alongside the FP4xFP4
      // runner so a single QMoE node can route per call (see fp4_native_max_tokens_per_expert_).
      if (is_fp16) {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_fp4_dense_fallback_runner_ = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE) && defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
      // Native W4A8: FP8 e4m3 activations + MXFP4 weights, BF16/FP16 input/output.
      // Template parameters: <T=fp8, WeightType=fp4, OutputType=BF16/FP16, InputType=BF16/FP16>.
      // CUTLASS routes this through the SM100+ block-scaled tensor op path. The runner accepts
      // BF16/FP16 input from the caller and quantizes it to FP8 inside expandInputRowsKernel
      // (MXFP8 branch, triggered by mxfp8_mxfp4.fc{1,2}.weight_block_scale being non-null).
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    } else if (quant_type_ == "fp8" && !use_fp8_dequant_fallback_) {
#if defined(ENABLE_FP8) && defined(USE_FP8_QMOE)
      // Native W8A16-FP8: activations are half/bf16, weights are __nv_fp8_e4m3
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp8_e4m3, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    } else {
      // FP4/NVFP4/WFP4AFP8 dequant fallback or FP8 dequant fallback: use A16 runner
      if (is_fp16) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // BFloat16
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
    }
  } else {
    // Integer quantization (INT4/INT8)
    if (is_fp16) {
      if (use_int_dequant_fallback) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, half, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else if (expert_weight_bits_ == 4) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // expert_weight_bits_ == 8
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
    }
#if defined(ENABLE_BF16)
    else {  // BFloat16
      if (use_int_dequant_fallback) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#if !defined(ORT_QUICK_BUILD)
      else if (expert_weight_bits_ == 4) {
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      } else {  // expert_weight_bits_ == 8
        m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t, __nv_bfloat16>>(
            sm_, activation_type_, normalize_routing_weights_, use_sparse_mixer_);
      }
#endif
    }
#endif
  }  // end integer quantization

  nvfp4_weights_row_major_ = quant_type_ == "nvfp4" && weights_prepacked_mode == 1;
  {
    int64_t attr_layer = -1;
    if (op_kernel_info.GetAttr<int64_t>("layer_index", &attr_layer).IsOK() && attr_layer >= 0) {
      offload_layer_index_ = static_cast<int32_t>(attr_layer);
    } else {
      int32_t parsed_layer = -1;
      auto get_input_name = [&](size_t slot) -> std::string {
#ifdef ORT_USE_EP_API_ADAPTERS
        if (slot < op_kernel_info.GetKernelInfo().GetInputCount()) {
          return op_kernel_info.GetKernelInfo().GetInputName(slot);
        }
#else
        const auto& in_defs = op_kernel_info.node().InputDefs();
        if (slot < in_defs.size() && in_defs[slot] != nullptr && in_defs[slot]->Exists()) {
          return in_defs[slot]->Name();
        }
#endif
        return {};
      };
      if (onnxruntime::moe_offload::MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(
              op_kernel_info.node().Name(), parsed_layer)) {
        offload_layer_index_ = parsed_layer;
      } else {
        const std::string in2_name = get_input_name(2);
        if (!in2_name.empty() &&
            onnxruntime::moe_offload::MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(
                in2_name, parsed_layer)) {
          offload_layer_index_ = parsed_layer;
        }
      }
    }
  }
  if (nvfp4_weights_row_major_) {
    nvfp4_gemv_raw_layout_ = true;
    enable_fp4_gemv_autotune_ = false;
    const Tensor* fc1_w = nullptr;
    const Tensor* fc1_s = nullptr;
    const Tensor* fc2_w = nullptr;
    const Tensor* fc2_s = nullptr;
    const Tensor* fc1_g = nullptr;
    const Tensor* fc2_g = nullptr;
    if (offload_layer_index_ >= 0 &&
        op_kernel_info.TryGetConstantInput(2, &fc1_w) && fc1_w &&
        op_kernel_info.TryGetConstantInput(3, &fc1_s) && fc1_s &&
        op_kernel_info.TryGetConstantInput(5, &fc2_w) && fc2_w &&
        op_kernel_info.TryGetConstantInput(6, &fc2_s) && fc2_s &&
        op_kernel_info.TryGetConstantInput(15, &fc1_g) && fc1_g &&
        op_kernel_info.TryGetConstantInput(16, &fc2_g) && fc2_g &&
        fc1_w->Location().device.Type() == OrtDevice::CPU &&
        fc1_w->Shape().NumDimensions() == 3 && fc2_w->Shape().NumDimensions() == 3) {
      int dev_id = 0;
      static_cast<void>(cudaGetDevice(&dev_id));
      const int num_e = static_cast<int>(fc1_w->Shape()[0]);
      const bool legacy_w = (fc1_w->Shape()[1] == fc2_w->Shape()[2] * 2 &&
                             fc1_w->Shape()[2] == fc2_w->Shape()[1]);
      const int hid = (fc2_s->Shape().NumDimensions() == 3)
                          ? static_cast<int>(fc2_s->Shape()[1])
                          : static_cast<int>(legacy_w ? fc1_w->Shape()[1] : fc2_w->Shape()[1]);
      const int inter = (fc1_s->Shape().NumDimensions() == 3)
                            ? static_cast<int>(fc1_s->Shape()[1] / 2)
                            : static_cast<int>(legacy_w ? fc2_w->Shape()[1] : (fc1_w->Shape()[1] / 2));
      offload_registered_dev_id_ = dev_id;
      offload_registered_fc1_w_ = static_cast<const uint8_t*>(fc1_w->DataRaw());
      RegisterQMoEDenseLayerHostWeights(
          dev_id, offload_layer_index_, num_e, hid, inter,
          static_cast<const uint8_t*>(fc1_w->DataRaw()),
          static_cast<const uint8_t*>(fc1_s->DataRaw()),
          static_cast<const float*>(fc1_g->DataRaw()),
          static_cast<const uint8_t*>(fc2_w->DataRaw()),
          static_cast<const uint8_t*>(fc2_s->DataRaw()),
          static_cast<const float*>(fc2_g->DataRaw()));
    }
  }

  // Parse v4 MoE offload policy and register with QMoESessionController when enabled (F03, F04, F10, F13, F17).
  {
    auto controller = GetQMoESessionController();
    std::filesystem::path model_path = GetSessionModelPath();
    if (model_path.empty() && controller) {
      model_path = controller->GetModelPath();
    }
    if (controller && !model_path.empty() && controller->GetModelPath().empty()) {
      controller->SetModelPath(model_path);
    }
    ORT_THROW_IF_ERROR(onnxruntime::moe_offload::MoeOffloadSessionPolicy::ParseFromConfigOptions(
        op_kernel_info.GetConfigOptions(), model_path, offload_policy_));
    if (offload_policy_.mode != onnxruntime::moe_offload::OffloadMode::kDisabled && controller) {
      auto get_input_name = [&](size_t slot) -> std::string {
#ifdef ORT_USE_EP_API_ADAPTERS
        if (slot < op_kernel_info.GetKernelInfo().GetInputCount()) {
          return op_kernel_info.GetKernelInfo().GetInputName(slot);
        }
#else
        const auto& in_defs = op_kernel_info.node().InputDefs();
        if (slot < in_defs.size() && in_defs[slot] != nullptr && in_defs[slot]->Exists()) {
          return in_defs[slot]->Name();
        }
#endif
        return {};
      };
      auto input_slot_exists = [&](size_t slot) -> bool {
        return !get_input_name(slot).empty();
      };
      bool has_unsupported_slot = false;
      int bad_slot_idx = -1;
      for (int bad_slot : {8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20}) {
        if (input_slot_exists(static_cast<size_t>(bad_slot))) {
          has_unsupported_slot = true;
          bad_slot_idx = bad_slot;
          break;
        }
      }
      const Tensor* fc1_w = nullptr;
      const Tensor* fc1_s = nullptr;
      const Tensor* fc1_b = nullptr;
      const Tensor* fc2_w = nullptr;
      const Tensor* fc2_s = nullptr;
      const Tensor* fc2_b = nullptr;
      const Tensor* fc1_g = nullptr;
      const Tensor* fc2_g = nullptr;
      const bool has_const_weights =
          op_kernel_info.TryGetConstantInput(2, &fc1_w) && fc1_w &&
          op_kernel_info.TryGetConstantInput(3, &fc1_s) && fc1_s &&
          op_kernel_info.TryGetConstantInput(5, &fc2_w) && fc2_w &&
          op_kernel_info.TryGetConstantInput(6, &fc2_s) && fc2_s &&
          fc1_w->Shape().NumDimensions() == 3 && fc2_w->Shape().NumDimensions() == 3 &&
          fc1_w->Location().device.Type() == OrtDevice::CPU &&
          fc1_s->Location().device.Type() == OrtDevice::CPU &&
          fc2_w->Location().device.Type() == OrtDevice::CPU &&
          fc2_s->Location().device.Type() == OrtDevice::CPU;
      if (input_slot_exists(4)) static_cast<void>(op_kernel_info.TryGetConstantInput(4, &fc1_b));
      if (input_slot_exists(7)) static_cast<void>(op_kernel_info.TryGetConstantInput(7, &fc2_b));
      if (input_slot_exists(15)) static_cast<void>(op_kernel_info.TryGetConstantInput(15, &fc1_g));
      if (input_slot_exists(16)) static_cast<void>(op_kernel_info.TryGetConstantInput(16, &fc2_g));

      const bool is_interleaved_swiglu =
          (activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu) &&
          (swiglu_fusion_ == 0 || swiglu_fusion_ == 1);
      const bool nvfp4_globals_ok = (quant_type_ != "nvfp4") || (fc1_g != nullptr && fc2_g != nullptr);
      // The offload backend reads the raw K-packed row-major bytes. Only NVFP4 with weights_prepacked=1 defines that
      // layout; MXFP4 and other weights_prepacked modes are N-packed in the standard path, so admitting them would
      // change how the same initializer bytes are interpreted.
      const bool node_eligible =
          nvfp4_weights_row_major_ &&
          expert_weight_bits_ == 4 &&
          is_interleaved_swiglu &&
          !use_sparse_mixer_ &&
          !has_unsupported_slot &&
          has_const_weights &&
          nvfp4_globals_ok;

      if (!node_eligible) {
        ORT_ENFORCE(offload_policy_.mode != onnxruntime::moe_offload::OffloadMode::kRequired,
                    "QMoE offload mode='required' rejected node '", op_kernel_info.node().Name(),
                    "': quant_type='", quant_type_, "', nvfp4_row_major_weights=", nvfp4_weights_row_major_,
                    " (requires quant_type='nvfp4' and weights_prepacked=1)",
                    ", swiglu_fusion=", swiglu_fusion_, ", use_sparse_mixer=", use_sparse_mixer_,
                    ", unsupported_slot=", bad_slot_idx, ", has_const_cpu_weights=", has_const_weights);
      } else {
        const int32_t num_e = static_cast<int32_t>(fc1_w->Shape()[0]);
        const int32_t hid = static_cast<int32_t>(fc1_w->Shape()[1]);
        const int32_t inter = static_cast<int32_t>(fc2_w->Shape()[1]);
        const size_t blk = 16u;

        offload_source_desc_.op_type = "QMoE";
        offload_source_desc_.quant_type = quant_type_;
        offload_source_desc_.stable_node_id = op_kernel_info.node().Name();
        offload_source_desc_.num_experts = num_e;
        offload_source_desc_.top_k = static_cast<int32_t>(k_);
        offload_source_desc_.hidden_size = hid;
        offload_source_desc_.inter_size = inter;
        offload_source_desc_.block_size = static_cast<int32_t>(blk);
        offload_source_desc_.weight_bits = 4;
        offload_source_desc_.weights_prepacked = weights_prepacked_;
        offload_source_desc_.placement.device_id = controller->DeviceId();
        offload_source_desc_.shard_spec.global_num_experts = num_e;
        offload_source_desc_.shard_spec.local_num_experts = num_e;
        offload_source_desc_.shard_spec.global_hidden_size = hid;
        offload_source_desc_.shard_spec.global_inter_size = inter;
        offload_source_desc_.shard_spec.local_hidden_size = hid;
        offload_source_desc_.shard_spec.local_inter_size = inter;
        offload_source_desc_.shard_spec.quant_block_size = static_cast<int32_t>(blk);

        auto record_slice = [&](onnxruntime::moe_offload::ExpertBankKind kind, int slot_idx, const Tensor* t) {
          if (!t || num_e <= 0) return;
          onnxruntime::moe_offload::MoeExpertBankSlice slice;
          slice.kind = kind;
          slice.input_index = slot_idx;
          slice.initializer_name = get_input_name(static_cast<size_t>(slot_idx));
          slice.elem_type = static_cast<ONNXTensorElementDataType>(t->GetElementType());
          slice.full_shape = t->Shape();
          slice.total_bytes = t->SizeInBytes();
          slice.bytes_per_expert = t->SizeInBytes() / static_cast<size_t>(num_e);
          slice.host_data = (t->Location().device.Type() == OrtDevice::CPU) ? t->DataRaw() : nullptr;
          slice.is_file_mapped = onnxruntime::moe_offload::IsReloadableFileMapping(slice.host_data, slice.total_bytes);
          offload_source_desc_.banks[static_cast<size_t>(kind)] = std::move(slice);
        };
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Weights, 2, fc1_w);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Weights, 5, fc2_w);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Scales, 3, fc1_s);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Scales, 6, fc2_s);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1GlobalScale, 15, fc1_g);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2GlobalScale, 16, fc2_g);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Bias, 4, fc1_b);
        record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Bias, 7, fc2_b);

        const size_t raw_slot_bytes =
            static_cast<size_t>(2) * inter * (hid / 2) +
            static_cast<size_t>(hid) * (inter / 2) +
            static_cast<size_t>(2) * inter * (hid / blk) +
            static_cast<size_t>(hid) * (inter / blk) +
            2 * sizeof(float);
        const size_t fc1_bias_slot_bytes =
            (fc1_b != nullptr && num_e > 0) ? (fc1_b->SizeInBytes() / static_cast<size_t>(num_e)) : 0;
        const size_t fc2_bias_fixed_bytes = (fc2_b != nullptr) ? fc2_b->SizeInBytes() : 0;

        onnxruntime::contrib::qmoe_shared::Nvfp4ExpertBlockLayout host_layout{};
        const bool layout_ok = (quant_type_ == "nvfp4")
                                   ? onnxruntime::contrib::qmoe_shared::MakeNvfp4ExpertBlockLayout(hid, inter, &host_layout)
                                   : onnxruntime::contrib::qmoe_shared::MakeMxfp4ExpertBlockLayout(hid, inter, &host_layout);
        const size_t cpu_slot_stride = layout_ok ? host_layout.stride : raw_slot_bytes;

        offload_capabilities_.supports_full_gpu_resident = true;
        offload_capabilities_.supports_hybrid_doorbell_decode = true;
        offload_capabilities_.supports_gpu_streaming_prefill = true;
        offload_capabilities_.supports_file_backed_bounded_host_cache = true;
        offload_capabilities_.supports_cuda_graph_capture = true;
        offload_capabilities_.decode_kernel_skips_negative_slots = true;
        offload_capabilities_.source_bytes_per_expert = raw_slot_bytes;
        offload_capabilities_.gpu_slot_bytes = raw_slot_bytes + fc1_bias_slot_bytes;
        // QMoE keeps its own full FC1/FC2 bias banks on the device for prefill dequantization, in addition to the
        // per-slot FC1 bias and full FC2 bias buffer owned by the offload backend; charge every persistent copy.
        const size_t fc1_bias_full_bytes = (fc1_b != nullptr) ? fc1_b->SizeInBytes() : 0;
        offload_capabilities_.gpu_fixed_table_bytes =
            static_cast<size_t>(2) * num_e * sizeof(int32_t) + 2 * fc2_bias_fixed_bytes + fc1_bias_full_bytes;
        offload_capabilities_.cpu_slot_bytes = cpu_slot_stride;
        offload_capabilities_.min_host_cache_slots = 1;

        controller->RegisterMoeOffloadNode(this, offload_policy_, offload_layer_index_);
        offload_source_desc_.layer_index = offload_layer_index_;
        offload_node_registered_ = true;
      }
    }
  }

  ORT_ENFORCE(m_moe_runner != nullptr,
              "QMoE: failed to construct MoE runner for quant_type='", quant_type_,
              "', expert_weight_bits=", expert_weight_bits_,
              ", input_type=", (is_fp16 ? "float16" : "bfloat16"),
              ". Build configuration may be missing the corresponding kernel.");
  if (enable_fp4_deep_gemm_) {
    m_moe_runner->setUseFp4DeepGemm(true);
    if (m_fp4_dense_fallback_runner_ != nullptr) {
      m_fp4_dense_fallback_runner_->setUseFp4DeepGemm(true);
    }
  }
}

QMoE::~QMoE() {
  if (offload_registered_dev_id_ >= 0 && offload_layer_index_ >= 0) {
    UnregisterQMoEDenseLayerHostWeights(
        offload_registered_dev_id_, offload_layer_index_, offload_registered_fc1_w_);
  }
  if (auto controller = GetQMoESessionController()) {
    controller->UnregisterMoeOffloadNode(this);
  }
}

Status QMoE::InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                    const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) {
  offload_policy_ = policy;
  offload_plan_ = plan;
  offload_layer_index_ = plan.layer_index;

  AllocatorPtr alloc = Info().GetAllocator(OrtMemTypeDefault);
  ORT_RETURN_IF_NOT(alloc != nullptr, "QMoE InitializeAdmittedPlan requires a valid CUDA device allocator.");
  cudaStream_t stream = nullptr;
  auto copy_optional_cpu_bias_to_gpu = [&](int slot, IAllocatorUniquePtr<void>& dst) {
    const Tensor* t = nullptr;
    if (Info().TryGetConstantInput(slot, &t) && t != nullptr && t->SizeInBytes() > 0 &&
        t->Location().device.Type() == OrtDevice::CPU) {
      bool dummy_packed = false;
      PrePackCopyToGpu(*t, stream, alloc, dst, dummy_packed);
    }
  };
  copy_optional_cpu_bias_to_gpu(4, offload_gpu_fc1_bias_);
  copy_optional_cpu_bias_to_gpu(7, offload_gpu_fc2_bias_);

  auto check_cpu_global_scale_eligible = [&](int slot) -> bool {
    if (quant_type_ != "nvfp4") return false;
    const Tensor* t = nullptr;
    if (!Info().TryGetConstantInput(slot, &t) || t == nullptr || t->Location().device.Type() != OrtDevice::CPU) {
      return false;
    }
    const int64_t count = t->Shape().Size();
    if (count <= 0) return false;
    const float* scales = t->Data<float>();
    for (int64_t i = 0; i < count; ++i) {
      if (!onnxruntime::llm::kernels::moe_gemv::IsNvfp4DecodeFastScale(scales[i])) return false;
    }
    return true;
  };
  nvfp4_fc1_decode_scale_eligible_ = check_cpu_global_scale_eligible(15);
  nvfp4_fc2_decode_scale_eligible_ = check_cpu_global_scale_eligible(16);

  if (plan.resident_slots >= plan.num_experts && nvfp4_weights_row_major_) {
    // Full GPU residency admitted for row-major NVFP4: upload CPU tensors to GPU buffers now.
    auto copy_input_to_gpu = [&](int slot, IAllocatorUniquePtr<void>& dst) {
      const Tensor* t = nullptr;
      if (Info().TryGetConstantInput(slot, &t) && t != nullptr && t->SizeInBytes() > 0) {
        bool dummy_packed = false;
        PrePackCopyToGpu(*t, stream, alloc, dst, dummy_packed);
      }
    };
    copy_input_to_gpu(2, packed_fp4_fc1_weights_);
    copy_input_to_gpu(3, packed_fp4_fc1_block_scales_);
    copy_input_to_gpu(5, packed_fp4_fc2_weights_);
    copy_input_to_gpu(6, packed_fp4_fc2_block_scales_);
    copy_input_to_gpu(15, packed_fc1_global_scale_);
    copy_input_to_gpu(16, packed_fc2_global_scale_);
    return Status::OK();
  }

  if (nvfp4_weights_row_major_) {
    auto copy_scale_to_gpu = [&](int slot, IAllocatorUniquePtr<void>& dst) {
      const Tensor* t = nullptr;
      if (Info().TryGetConstantInput(slot, &t) && t != nullptr && t->SizeInBytes() > 0) {
        bool dummy_packed = false;
        PrePackCopyToGpu(*t, stream, alloc, dst, dummy_packed);
      }
    };
    copy_scale_to_gpu(15, packed_fc1_global_scale_);
    copy_scale_to_gpu(16, packed_fc2_global_scale_);
  }

  offload_backend_ = std::make_unique<QMoEOffload>(Info(), policy, offload_source_desc_, plan);
  return Status::OK();
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
  if (offload_backend_ != nullptr) {
    const Tensor* in_tensor = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(in_tensor != nullptr, "QMoE requires input tensor 0.");
    const auto& in_shape = in_tensor->Shape();
    const int64_t rows = (in_shape.NumDimensions() == 2)
                             ? in_shape[0]
                             : static_cast<int64_t>(SafeInt<int64_t>(in_shape[0]) * in_shape[1]);
    if (!nvfp4_weights_row_major_ || rows <= offload_policy_.async_max_tokens) {
      return offload_backend_->ComputeInternal(context);
    }
    offload_backend_->SyncPrefillSlotTables(Stream(context));
  }
  const bool is_fp4 = (quant_type_ == "fp4");
  const bool is_nvfp4 = (quant_type_ == "nvfp4");
  // NVFP4 shares the E2M1 4-bit weight format, [E,K,N/2] weight layout, and the per-expert
  // global-scale (15/16) + block-scale (3/6) inputs with MXFP4 "fp4"; it differs only in the
  // block-scale decode (Float8E4M3FN vs Float8E8M0) and block size (16 vs 32). Treat both as an
  // fp4-family type wherever weight-scale handling is shared.
  const bool is_fp4_family = is_fp4 || is_nvfp4;
  const bool is_fp8 = (quant_type_ == "fp8");
  const bool is_block_fp8 = is_fp8 && block_size_ > 0;
  const bool is_wfp4afp8 = (quant_type_ == "wfp4afp8");
  const bool is_int = (quant_type_ == "int");
  const bool is_mixed_width = fc1_expert_weight_bits_ != expert_weight_bits_ ||
                              fc2_expert_weight_bits_ != expert_weight_bits_ ||
                              fc3_expert_weight_bits_ != expert_weight_bits_;
  const bool use_int_dequant_fallback =
      is_int && (is_mixed_width || expert_weight_bits_ == 2);
  // Modes that consume FP4 weight block scales (inputs 3/6) and per-expert global weight scales.
  const bool uses_fp4_weight_scales = is_fp4_family || is_wfp4afp8;
  // Modes that consume per-expert FP-format global weight scales (inputs 15/16).
  const bool uses_global_weight_scales = is_fp4_family || (is_fp8 && !is_block_fp8) || is_wfp4afp8;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  // When PrePack consumed the int4/int8 expert-weight initializers
  // (``weights_prepacked == false`` opt-in path), the original tensors
  // were freed; ``context->Input<Tensor>(2)/(5)`` would return nothing.
  // Mirror how ``MatMulNBits`` reads its prepacked B input.
  // Gate on *both* prepacked buffers being present. If only fc1 were prepacked
  // (e.g. a partial prepack from an earlier failure or a future refactor), this
  // path must not null out fc2_experts_weights and feed a null fc2 weight/shape
  // to the runner.
  const bool int_weights_consumed_by_prepack =
      is_int && !use_int_dequant_fallback && !weights_prepacked_ &&
      packed_fc1_weights_ != nullptr && packed_fc2_weights_ != nullptr;
  // Same idea for MXFP4 in the SM80 grouped-GEMM regime: PrePack laid the e2m1 weights out in the
  // CUTLASS interleaved layout (and, unless low-memory mode is on, an extra ColToRow copy for the
  // decode GEMV), nothing reads the raw [E, K, N/2] initializer, so it was released
  // (``is_packed = true``). Gate on the packed buffers actually existing so a session with
  // ``session.disable_prepacking`` (where PrePack never ran and the initializer is still live)
  // keeps reading the raw tensors.
  const bool fp4_weights_consumed_by_prepack =
      is_fp4 && release_fp4_raw_weights_ &&
      gemv_fp4_fc1_weights_ != nullptr && gemv_fp4_fc2_weights_ != nullptr &&
      fc1_weights_shape_.NumDimensions() == 3 && fc2_weights_shape_.NumDimensions() == 3;
  const bool weights_consumed_by_prepack =
      int_weights_consumed_by_prepack || fp4_weights_consumed_by_prepack;
  // When ``weights_prepacked == 0`` the raw ``[E, N, K/pack]`` int weights must be
  // converted to the CUTLASS fpA_intB layout by PrePack before the runner can consume
  // them. If PrePack never ran (e.g. ``session.disable_prepacking`` is set), the prepack
  // buffers stay null and falling through to the raw initializer pointers would feed
  // non-CUTLASS bytes to the runner, producing silently wrong output. Fail loudly instead.
  if (is_int && !use_int_dequant_fallback && !weights_prepacked_ &&
      (packed_fc1_weights_ == nullptr || packed_fc2_weights_ == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE weights_prepacked=0 requires PrePack to run, but the int weight "
                           "buffers were not produced (is session.disable_prepacking set?). Provide "
                           "CUTLASS-prepacked weights with weights_prepacked=1, or enable prepacking.");
  }
  const Tensor* fc1_experts_weights = weights_consumed_by_prepack ? nullptr : context->Input<Tensor>(2);
  const Tensor* fc1_scales = (is_int && !packed_fc1_scales_) ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = weights_consumed_by_prepack ? nullptr : context->Input<Tensor>(5);
  const Tensor* fc2_scales = (is_int && !packed_fc2_scales_) ? context->Input<Tensor>(6) : nullptr;
  if (is_int) {
    for (const auto& [input_idx, scales] : std::array{
             std::pair{3, fc1_scales},
             std::pair{6, fc2_scales}}) {
      ORT_RETURN_IF(scales && !(is_fp16_ ? scales->IsDataType<MLFloat16>() : scales->IsDataType<BFloat16>()),
                    "CUDA QMoE integer fc", input_idx == 3 ? 1 : 2,
                    "_scales must match the activation type (FP16/BF16).");
    }
  }
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights = context->Input<Tensor>(8);
  const bool split_fp8_fc1 = is_block_fp8 && fc3_experts_weights != nullptr;
  ORT_ENFORCE(fc3_experts_weights == nullptr || split_fp8_fc1,
              "QMoE in CUDA execution provider does not support separate fc3_experts_weights. "
              "Gate and up projection weights must be pre-concatenated into fc1, except for block-scaled FP8.");
  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  if (split_fp8_fc1) {
    ORT_RETURN_IF_NOT(activation_type_ == ActivationType::Silu && swiglu_fusion_ == 0,
                      "Separate FC3 block-scaled FP8 requires activation_type='silu' and swiglu_fusion=0.");
    ORT_RETURN_IF(fc1_experts_bias_optional || context->Input<Tensor>(10),
                  "Separate FC3 block-scaled FP8 does not support FC1/FC3 biases.");
  }
  const auto kernel_activation_type = split_fp8_fc1 ? ActivationType::Swiglu : activation_type_;

  // Backward compatibility: the published gpt-oss-20b model (and any model exported by ORT < 1.27)
  // hard-coded the interleaved SwiGLU fusion layout and did not emit a swiglu_fusion attribute, so it
  // falls back to the default of 0 ("not fused"). QMoE never has a separate FC3 (enforced above), so a
  // SwiGLU activation with swiglu_fusion == 0 means the gate and value projections are actually pre-fused
  // into FC1 (interleaved layout). Treat this as swiglu_fusion == 1 so those legacy models keep working.
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
      swiglu_fusion == 0 && !split_fp8_fc1) {
    swiglu_fusion = 1;
    LogQMoESwigluFusionRemapOnce();
  }
  if (split_fp8_fc1) {
    swiglu_fusion = 1;
  } else if (is_block_fp8 && swiglu_fusion == 2) {
    // Normalize block-fused weights to the interleaved layout during dequantization.
    swiglu_fusion = 1;
  }

  const Tensor* fc1_zeros = packed_fc1_bias_ ? nullptr : context->Input<Tensor>(11);
  const Tensor* fc2_zeros = packed_fc2_bias_ ? nullptr : context->Input<Tensor>(12);

  auto check_weight_type = [](const Tensor* tensor, const char* name, bool expect_fp8) -> Status {
    ORT_RETURN_IF_NOT(tensor != nullptr, "Input '", name, "' is required.");
    if (expect_fp8) {
      ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E4M3FN>(), name, " must be a float8e4m3fn tensor when quant_type='fp8'.");
    } else {
      ORT_RETURN_IF_NOT(tensor->IsDataType<uint8_t>(), name, " must be a uint8 tensor when quant_type is 'int' or 'fp4'.");
    }
    return Status::OK();
  };

  // When PrePack consumed the int/MXFP4 weight initializers, the dtype check
  // is no longer applicable (we know they were uint8 — that's what
  // PrePackIntExpertWeights / PrePackRepackFP4Weights validated and consumed).
  if (!weights_consumed_by_prepack) {
    ORT_RETURN_IF_ERROR(check_weight_type(fc1_experts_weights, "fc1_experts_weights", is_fp8));
    ORT_RETURN_IF_ERROR(check_weight_type(fc2_experts_weights, "fc2_experts_weights", is_fp8));
    if (split_fp8_fc1) {
      ORT_RETURN_IF_ERROR(check_weight_type(fc3_experts_weights, "fc3_experts_weights", true));
    }
  }

  // Unified FP4 inputs: block scales in fc*_scales (3/6), global scales in 15/16.
  // FP4 scales are copied to GPU during PrePack but intentionally remain live inputs. The raw
  // tensors provide the dtype and shape metadata required to validate packed buffers before any
  // dequant or GEMV kernel indexes them.
  const Tensor* fp4_fc1_block_scales = uses_fp4_weight_scales ? context->Input<Tensor>(3) : nullptr;
  const Tensor* fp4_fc2_block_scales = uses_fp4_weight_scales ? context->Input<Tensor>(6) : nullptr;
  const Tensor* fc1_global_scale = uses_global_weight_scales ? context->Input<Tensor>(15) : nullptr;
  const Tensor* fc2_global_scale = uses_global_weight_scales ? context->Input<Tensor>(16) : nullptr;
  if (is_fp8) {
    ORT_RETURN_IF(context->Input<Tensor>(11) || context->Input<Tensor>(12) || context->Input<Tensor>(13),
                  "QMoE FP8 does not support zero_points.");
  }
  if (is_block_fp8) {
    ORT_RETURN_IF(context->Input<Tensor>(15) || context->Input<Tensor>(16),
                  "Block-scaled FP8 uses fc*_scales, not per-expert global scales.");
    ORT_RETURN_IF(!split_fp8_fc1 && (context->Input<Tensor>(9) || context->Input<Tensor>(10)),
                  "Block-scaled FP8 FC3 scales/bias require fc3_experts_weights.");
    ORT_RETURN_IF(swiglu_fusion_ == 2 && fc1_experts_bias_optional,
                  "Block-fused SwiGLU FP8 does not support FC1 bias; use interleaved fusion.");
    ORT_RETURN_IF(activation_type_ != ActivationType::Swiglu && !split_fp8_fc1 && swiglu_fusion_ != 0,
                  "Block-scaled FP8 swiglu_fusion requires activation_type='swiglu'.");
  }

  // W4A8 (WFP4AFP8) optional Variant A activation scales (per-tensor or per-expert FP8 global act scale).
  const Tensor* fc1_act_scale = (is_wfp4afp8 && !packed_fc1_act_scale_) ? context->Input<Tensor>(17) : nullptr;
  const Tensor* fc2_act_scale = (is_wfp4afp8 && !packed_fc2_act_scale_) ? context->Input<Tensor>(18) : nullptr;

  const bool has_any_zero_point = (fc1_zeros != nullptr || fc2_zeros != nullptr ||
                                   packed_fc1_bias_ != nullptr || packed_fc2_bias_ != nullptr);

  // Row-wise quantization path does not support asymmetric zero-points in QMoE.
  // QuantParams::Int only carries scales (no zero/bias tensor).
  if (block_size_ <= 0 && has_any_zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE row-wise quantization (block_size <= 0) does not support zero_points. "
                           "Remove fc*_zero_points or use block-wise quantization.");
  }
  if (block_size_ > 0 && block_size_ < 32 && has_any_zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE asymmetric zero_points are currently supported only when block_size >= 32. "
                           "Use block_size >= 32 or remove fc*_zero_points.");
  }

  bool is_fused_swiglu = activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu;
  MoEParameters moe_params;
  // Prefer the cached shapes when PrePack consumed the source initializer.
  const TensorShape& fc1_shape = weights_consumed_by_prepack ? fc1_weights_shape_ : fc1_experts_weights->Shape();
  const TensorShape& fc2_shape = weights_consumed_by_prepack ? fc2_weights_shape_ : fc2_experts_weights->Shape();
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs, &fc1_shape,
      fc1_experts_bias_optional, fc1_scales, fc1_zeros,
      &fc2_shape, fc2_experts_bias_optional, fc2_scales, fc2_zeros,
      split_fp8_fc1 ? &fc3_experts_weights->Shape() : nullptr, nullptr, nullptr, nullptr,
      moe_helper::MoEWeightBits{fc1_expert_weight_bits_,
                                fc2_expert_weight_bits_,
                                fc3_expert_weight_bits_},
      is_fused_swiglu, block_size_));
  if (is_block_fp8) {
    const int64_t fc1_n = is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
    ORT_RETURN_IF_NOT(fc1_shape == TensorShape({moe_params.num_experts, fc1_n, moe_params.hidden_size}) &&
                          fc2_shape == TensorShape({moe_params.num_experts, moe_params.hidden_size, moe_params.inter_size}),
                      "Block-scaled FP8 requires row-major expert weights [E,N,K].");
    ORT_RETURN_IF_NOT(moe_params.hidden_size % 8 == 0 && moe_params.inter_size % 8 == 0,
                      "Block-scaled FP8 CUDA requires hidden_size and inter_size to be multiples of 8.");
    ORT_RETURN_IF_ERROR(moe_helper::CheckFp8BlockScale<Tensor>(
        context->Input<Tensor>(3), moe_params.num_experts, fc1_n, moe_params.hidden_size, block_size_, "fc1_scales"));
    ORT_RETURN_IF_ERROR(moe_helper::CheckFp8BlockScale<Tensor>(
        context->Input<Tensor>(6), moe_params.num_experts, moe_params.hidden_size, moe_params.inter_size,
        block_size_, "fc2_scales"));
    if (split_fp8_fc1) {
      ORT_RETURN_IF_NOT(fc3_experts_weights->Shape() ==
                            TensorShape({moe_params.num_experts, moe_params.inter_size, moe_params.hidden_size}),
                        "Block-scaled FP8 requires row-major fc3_experts_weights [E,N,K].");
      ORT_RETURN_IF_ERROR(moe_helper::CheckFp8BlockScale<Tensor>(
          context->Input<Tensor>(9), moe_params.num_experts, moe_params.inter_size, moe_params.hidden_size,
          block_size_, "fc3_scales"));
    }
  }
  if (is_mixed_width) {
    ORT_RETURN_IF_NOT(is_int, "Mixed-width QMoE execution is currently supported only for quant_type='int'.");
  }
  size_t int_dequant_fc1_bytes = 0;
  size_t int_dequant_fc2_bytes = 0;
  if (use_int_dequant_fallback) {
    ORT_RETURN_IF_NOT(moe_params.hidden_size % block_size_ == 0,
                      "INT2 or mixed-width CUDA QMoE requires hidden_size to be divisible by block_size, got hidden_size=",
                      moe_params.hidden_size, " and block_size=", block_size_, ".");
    ORT_RETURN_IF_NOT(moe_params.inter_size % block_size_ == 0,
                      "INT2 or mixed-width CUDA QMoE requires inter_size to be divisible by block_size, got inter_size=",
                      moe_params.inter_size, " and block_size=", block_size_, ".");

    const int64_t fc1_n = is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size;
    const int64_t fc1_k = moe_params.hidden_size;
    const int64_t fc2_n = moe_params.hidden_size;
    const int64_t fc2_k = moe_params.inter_size;
    auto validate_fallback_input = [&](const TensorShape& weight_shape, const Tensor* scales,
                                       const char* weight_name, const char* scales_name,
                                       int64_t bits, int64_t n, int64_t k) -> Status {
      const int64_t packed_k = k / (8 / bits);
      ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3 &&
                            weight_shape[0] == moe_params.num_experts &&
                            weight_shape[1] == n && weight_shape[2] == packed_k,
                        "INT2 or mixed-width CUDA QMoE dense fallback requires canonical ", weight_name,
                        " shape [E,N,K/pack]=[", moe_params.num_experts, ",", n, ",", packed_k,
                        "], got ", weight_shape, ". Legacy transposed weight layouts are not supported.");
      ORT_RETURN_IF_NOT(scales != nullptr, scales_name, " is required.");
      ORT_RETURN_IF_NOT(is_fp16_ ? scales->IsDataType<MLFloat16>() : scales->IsDataType<BFloat16>(),
                        "INT2 or mixed-width CUDA QMoE requires ", scales_name,
                        " dtype to match the input dtype (", is_fp16_ ? "float16" : "bfloat16", ").");
      const auto& scale_shape = scales->Shape();
      const int64_t blocks_per_row = k / block_size_;
      ORT_RETURN_IF_NOT(scale_shape.NumDimensions() == 3 &&
                            scale_shape[0] == moe_params.num_experts &&
                            scale_shape[1] == n && scale_shape[2] == blocks_per_row,
                        "INT2 or mixed-width CUDA QMoE dense fallback requires block-wise ", scales_name,
                        " shape [E,N,K/block_size]=[", moe_params.num_experts, ",", n, ",", blocks_per_row,
                        "], got ", scale_shape, ".");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(validate_fallback_input(fc1_shape, fc1_scales, "fc1_experts_weights", "fc1_scales",
                                                fc1_expert_weight_bits_, fc1_n, fc1_k));
    ORT_RETURN_IF_ERROR(validate_fallback_input(fc2_shape, fc2_scales, "fc2_experts_weights", "fc2_scales",
                                                fc2_expert_weight_bits_, fc2_n, fc2_k));

    const size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    int_dequant_fc1_bytes = SafeInt<size_t>(moe_params.num_experts) * fc1_n * fc1_k * element_size;
    int_dequant_fc2_bytes = SafeInt<size_t>(moe_params.num_experts) * fc2_n * fc2_k * element_size;
  }
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "QMoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);

  // The INT4/INT8 weight-only path stores B in the column-interleaved layout (ColumnMajorTileInterleave),
  // whose CUTLASS pitchlinear iterators require the GEMM reduction dim K to be a whole multiple of the
  // interleave tile (kInterleaveKTile == 64 for fp16/bf16 activations). For the two MoE GEMMs the
  // reduction dims are fc1.K == hidden_size and fc2.K == inter_size. A partial final K tile is read past
  // the valid range and silently yields garbage/NaN (the single-matrix fpA_intB GEMM throws on this; the
  // grouped MoE GEMM and the decode GEMV have no such guard), so reject it up front with a clear error.
  if (quant_type_ == "int" && !use_int_dequant_fallback) {
    constexpr int64_t kInterleaveKTile = 64;
    ORT_RETURN_IF_NOT(moe_params.hidden_size % kInterleaveKTile == 0,
                      "QMoE int weight-only quantization requires hidden_size to be a multiple of ",
                      kInterleaveKTile, " (the interleaved-weight K tile), got hidden_size=", moe_params.hidden_size, ".");
    ORT_RETURN_IF_NOT(moe_params.inter_size % kInterleaveKTile == 0,
                      "QMoE int weight-only quantization requires inter_size to be a multiple of ",
                      kInterleaveKTile, " (the interleaved-weight K tile), got inter_size=", moe_params.inter_size, ".");
  }

  if (uses_fp4_weight_scales) {
    // MXFP4 ("fp4"/"wfp4afp8") uses block size 32 with Float8E8M0 block scales; NVFP4 ("nvfp4")
    // uses block size 16 with Float8E4M3FN block scales. Both are consumed as raw uint8 bytes.
    const int64_t fp4_block_size = is_nvfp4 ? 16 : 32;
    const char* fp4_mode_label = is_nvfp4 ? "quant_type='nvfp4'" : "quant_type='fp4'/'wfp4afp8'";
    const char* fp4_scale_label = is_nvfp4 ? "NVFP4 block scales" : "MXFP4 block scales";
    ORT_RETURN_IF_NOT(moe_params.hidden_size % fp4_block_size == 0,
                      "QMoE ", fp4_mode_label, " requires hidden_size to be a multiple of ",
                      fp4_block_size, " for ", fp4_scale_label, ", got hidden_size=", moe_params.hidden_size, ".");
    ORT_RETURN_IF_NOT(moe_params.inter_size % fp4_block_size == 0,
                      "QMoE ", fp4_mode_label, " requires inter_size to be a multiple of ",
                      fp4_block_size, " for ", fp4_scale_label, ", got inter_size=", moe_params.inter_size, ".");
    const int64_t fc1_out_size = is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size;
    auto check_fp4_block_scale = [is_nvfp4](const Tensor* tensor, const char* name, int64_t num_experts,
                                            int64_t n, int64_t k) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='fp4'/'nvfp4'/'wfp4afp8' requires ", name, ".");
      if (is_nvfp4) {
        ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E4M3FN>(), name,
                          " must be a float8e4m3fn NVFP4 block-scale tensor, got element type ",
                          tensor->GetElementType(), ".");
      } else {
        ORT_RETURN_IF_NOT(tensor->IsDataType<Float8E8M0>(), name,
                          " must be a float8e8m0 MXFP block-scale tensor, got element type ",
                          tensor->GetElementType(), ".");
      }
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 3 && dims[0] == num_experts && dims[1] == n && dims[2] == k,
                        name, " must have shape (", num_experts, ", ", n, ", ", k, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    auto check_global_scale = [](const Tensor* tensor, const char* name, int64_t num_experts, const char* quant_type) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='", quant_type, "' requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && dims[0] == num_experts,
                        name, " must have shape (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };

    if (fp4_fc1_block_scales) {
      ORT_RETURN_IF_ERROR(check_fp4_block_scale(fp4_fc1_block_scales, "fc1_scales", moe_params.num_experts,
                                                fc1_out_size, moe_params.hidden_size / fp4_block_size));
    }
    if (fp4_fc2_block_scales) {
      ORT_RETURN_IF_ERROR(check_fp4_block_scale(fp4_fc2_block_scales, "fc2_scales", moe_params.num_experts,
                                                moe_params.hidden_size, moe_params.inter_size / fp4_block_size));
    }
    if (fc1_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc1_global_scale, "fc1_global_scale", moe_params.num_experts, quant_type_.c_str()));
    }
    if (fc2_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc2_global_scale, "fc2_global_scale", moe_params.num_experts, quant_type_.c_str()));
    }
  }

  if (is_wfp4afp8) {
    auto check_act_scale = [](const Tensor* tensor, const char* name, int64_t num_experts) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='wfp4afp8' Variant A requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && (dims[0] == 1 || dims[0] == num_experts),
                        name, " must have shape (1,) or (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    // fc*_act_scale are optional; when absent the runner uses the MXFP8 block-scaled Variant B.
    if (fc1_act_scale) {
      ORT_RETURN_IF_ERROR(check_act_scale(fc1_act_scale, "fc1_act_scale", moe_params.num_experts));
    }
    if (fc2_act_scale) {
      ORT_RETURN_IF_ERROR(check_act_scale(fc2_act_scale, "fc2_act_scale", moe_params.num_experts));
    }
  }

  if (is_fp8) {
    auto check_global_scale = [](const Tensor* tensor, const char* name, int64_t num_experts) -> Status {
      ORT_RETURN_IF_NOT(tensor != nullptr, "QMoE quant_type='fp8' requires ", name, ".");
      ORT_RETURN_IF_NOT(tensor->IsDataType<float>(), name, " must be a float tensor.");
      const auto& dims = tensor->Shape().GetDims();
      ORT_RETURN_IF_NOT(dims.size() == 1 && dims[0] == num_experts,
                        name, " must have shape (", num_experts, "), got ", tensor->Shape().ToString(), ".");
      return Status::OK();
    };
    if (fc1_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc1_global_scale, "fc1_global_scale", moe_params.num_experts));
    }
    if (fc2_global_scale) {
      ORT_RETURN_IF_ERROR(check_global_scale(fc2_global_scale, "fc2_global_scale", moe_params.num_experts));
    }
  }

  // Validate minimum dimensions for CUTLASS kernels.
  // SM >= 90 TMA WarpSpecialized: smallest tile is 128x16x128B (N=16 for FP16). K < tile_K handled by TMA.
  // SM < 90 Ampere GemmGrouped: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
  // On SM90 with mixed-type (INT4/INT8), the Ampere fallback is used — same predication applies.
  // Alignment of dimensions to 128 bits is enforced separately in moe_kernels.cu.
  {
    constexpr int min_dim = 16;
    ORT_RETURN_IF(moe_params.hidden_size < min_dim,
                  "QMoE CUDA kernel requires hidden_size >= ", min_dim,
                  " for SM", sm_, ", got ", moe_params.hidden_size);
    ORT_RETURN_IF(moe_params.inter_size < min_dim,
                  "QMoE CUDA kernel requires inter_size >= ", min_dim,
                  " for SM", sm_, ", got ", moe_params.inter_size);
  }

  bool use_awq = (fc1_zeros != nullptr) || (packed_fc1_bias_ != nullptr);
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  // Per-call native-vs-dense routing for the FP4 prefill regime. The native FP4 grouped GEMM wins
  // when each expert processes few tokens, but the dense A16 grouped GEMM (MXFP4 weights
  // dequantized to FP16/BF16) is faster once the average per-expert token count is large
  // (compute-bound; measured crossover ~128 tokens/expert on H200). When native FP4 is enabled and
  // the average exceeds fp4_native_max_tokens_per_expert_, route this call through the dense
  // fallback runner instead. avg_tokens_per_expert = num_rows * top_k / num_experts.
  const bool fp4_native_available = is_fp4_family && !use_fp4_dequant_fallback_ && m_fp4_dense_fallback_runner_ != nullptr;
  const int64_t avg_tokens_per_expert =
      moe_params.num_experts > 0
          ? (static_cast<int64_t>(moe_params.num_rows) * static_cast<int64_t>(k_)) /
                static_cast<int64_t>(moe_params.num_experts)
          : static_cast<int64_t>(moe_params.num_rows) * static_cast<int64_t>(k_);
  // Offload-managed NVFP4 keeps raw scales and initializers on the host or in raw GPU copies, which the native
  // block-scaled kernels (swizzled scales, device weight pointers) cannot consume.
  const bool route_native_fp4 =
      fp4_native_available && !offload_node_registered_ &&
      (!nvfp4_weights_row_major_ || (packed_fp4_fc1_block_scales_ != nullptr &&
                                     packed_fp4_fc2_block_scales_ != nullptr)) &&
      (fp4_native_max_tokens_per_expert_ <= 0 || avg_tokens_per_expert <= fp4_native_max_tokens_per_expert_) &&
      // Native FP4xFP4 is underfilled for decode (small M) and quantizes activations to NVFP4;
      // keep NVFP4 decode shapes (num_rows < prefill threshold) off the native runner so they route
      // through the fused GEMV or the dense A16 fallback instead. (MXFP4 keeps its existing routing.)
      !(is_nvfp4 && fp4_prefill_min_tokens_ > 0 &&
        static_cast<int64_t>(moe_params.num_rows) < fp4_prefill_min_tokens_);
#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
  const bool use_fp4_deep_gemm =
      enable_fp4_deep_gemm_ && moe_params.num_rows > 0 &&
      moe_params.num_rows <= onnxruntime::llm::kernels::deep_gemm_sm90::kMaxTokensPerExpert &&
      moe_params.num_experts == fp4_deep_gemm_num_experts_ &&
      moe_params.hidden_size == onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize &&
      moe_params.inter_size == onnxruntime::llm::kernels::deep_gemm_sm90::kInterSize &&
      is_fused_swiglu && swiglu_fusion == 1 && !use_awq && fc1_experts_bias_optional == nullptr &&
      fc2_experts_bias_optional == nullptr && fp4_deep_gemm_fc1_weights_ != nullptr &&
      fp4_deep_gemm_fc2_weights_ != nullptr && fp4_deep_gemm_fc1_weight_scales_ != nullptr &&
      fp4_deep_gemm_fc2_weight_scales_ != nullptr;
#else
  constexpr bool use_fp4_deep_gemm = false;
#endif
  // SM80 FP4 grouped-GEMM prefill path. Active only when ORT_FP4_SM80_GEMM=1 and the
  // GEMV prepack produced the SM80 CUTLASS-interleaved e2m1 weights + activation-dtype group scales. Decode
  // shapes are still served by the fused GEMV (which returns early below); everything that
  // falls through to the runner here (prefill / GEMV-unsupported shapes) runs on the FP4
  // runner via the Ampere/SM80 DqMma grouped GEMM with QuantParams::GroupWise(32, ...).
  const bool fp4_sm80_prefill =
      !use_fp4_deep_gemm && is_fp4 && enable_fp4_sm80_gemm_ &&
      gemv_fp4_fc1_weights_ != nullptr && gemv_fp4_fc2_weights_ != nullptr &&
      gemv_fp4_fc1_scales_ != nullptr && gemv_fp4_fc2_scales_ != nullptr;
  // Releasing the raw e2m1 initializers (fp4_weights_consumed_by_prepack) is only sound because
  // every dispatch is served by a pre-packed buffer. If the SM80 prefill buffers were not fully
  // built (e.g. the group scales never materialized because a global-scale input was missing),
  // the dequant fallback below would be selected and would dereference the released initializer.
  // Fail loudly rather than crash.
  if (fp4_weights_consumed_by_prepack && !fp4_sm80_prefill && !use_fp4_deep_gemm) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "QMoE MXFP4: PrePack released the raw expert-weight initializers but the "
                           "SM80 grouped-GEMM buffers are incomplete (missing pre-packed weights or "
                           "group scales). Set ORT_FP4_SM80_GEMM=0 to keep the dequant fallback.");
  }
  // The dense fallback runner is selected only when native FP4 is available but this call exceeds
  // the per-expert threshold. In every other configuration (native chosen, or native unavailable so
  // m_moe_runner is itself the dense A16 runner) we use m_moe_runner.
  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface* active_runner =
      use_fp4_deep_gemm && m_fp4_dense_fallback_runner_ != nullptr
          ? m_fp4_dense_fallback_runner_.get()
          : ((fp4_native_available && !route_native_fp4) ? m_fp4_dense_fallback_runner_.get() : m_moe_runner.get());

  // Decide the standalone FP4 GEMV route before sizing grouped-runner scratch. GEMV owns a
  // different routing/workspace pipeline and remains deliberately untiled. Every other route
  // reaches runMoe, whose input, router logits, top-k metadata, and output are all row-major and
  // row-independent, so those pointers can be advanced by a row offset without changing expert
  // routing or output placement.
  const bool fp4_decode_regime =
      use_fp4_dequant_fallback_ ||
      ((enable_fp4_cutlass_gemm_ || enable_nvfp4_cutlass_gemm_) && fp4_prefill_min_tokens_ > 0 &&
       moe_params.num_rows < fp4_prefill_min_tokens_);
  const bool fc1_gemv_sm80_layout = gemv_fp4_fc1_reads_sm80_layout_;
  const bool fc2_gemv_sm80_layout = gemv_fp4_fc2_reads_sm80_layout_;
  const bool use_raw_nvfp4_gemv = is_nvfp4 && nvfp4_gemv_raw_layout_;
  const bool fp4_gemv_buffers_ready =
      (use_raw_nvfp4_gemv
           ? (fc1_experts_weights != nullptr && fc2_experts_weights != nullptr)
       : enable_fp4_sm80_gemm_
           ? ((fc1_gemv_sm80_layout ? gemv_fp4_fc1_weights_ != nullptr : gemv_fp4_fc1_weights_decode_ != nullptr) &&
              (fc2_gemv_sm80_layout ? gemv_fp4_fc2_weights_ != nullptr : gemv_fp4_fc2_weights_decode_ != nullptr))
           : (gemv_fp4_fc1_weights_ != nullptr && gemv_fp4_fc2_weights_ != nullptr)) &&
      (use_raw_nvfp4_gemv
           ? (nvfp4_weights_row_major_
                  ? (context->Input<Tensor>(3) != nullptr && context->Input<Tensor>(6) != nullptr &&
                     context->Input<Tensor>(15) != nullptr && context->Input<Tensor>(16) != nullptr)
                  : (packed_fp4_fc1_block_scales_ != nullptr && packed_fp4_fc2_block_scales_ != nullptr &&
                     packed_fc1_global_scale_ != nullptr && packed_fc2_global_scale_ != nullptr))
           : (gemv_fp4_fc1_scales_ != nullptr && gemv_fp4_fc2_scales_ != nullptr));
  bool use_fp4_gemv = false;
  const bool fp4_gemv_prologue_supported =
      (k_ == 1 || k_ == 2 || k_ == 4 || k_ == 6 || k_ == 8 || k_ == 10) &&
      moe_params.num_experts <= 1022;
  if (offload_backend_ == nullptr && is_fp4_family && fp4_decode_regime && enable_fp4_gemv_ && is_fused_swiglu &&
      fp4_gemv_buffers_ready && fp4_gemv_prologue_supported) {
    namespace gemv = onnxruntime::llm::kernels::moe_gemv;
    const int64_t expanded = moe_params.num_rows * static_cast<int64_t>(k_);
    const int64_t fc1_n = moe_params.inter_size * 2;
    const int gemv_group_size = is_nvfp4 ? 16 : 32;
    use_fp4_gemv =
        moe_params.num_rows > 0 && moe_params.num_rows <= 256 && expanded > 0 &&
        gemv::is_moe_gemv_fp4_supported(sm_, expanded, fc1_n, moe_params.hidden_size, gemv_group_size,
                                        gemv::MoeGemvConfig::kDefault, fc1_gemv_sm80_layout, use_raw_nvfp4_gemv,
                                        nvfp4_weights_row_major_) &&
        gemv::is_moe_gemv_fp4_supported(sm_, expanded, moe_params.hidden_size, moe_params.inter_size,
                                        gemv_group_size, gemv::MoeGemvConfig::kDefault,
                                        fc2_gemv_sm80_layout, use_raw_nvfp4_gemv, nvfp4_weights_row_major_);
  }

  bool use_packed_int_gemv = false;
  const int64_t packed_int_expanded = moe_params.num_rows * static_cast<int64_t>(k_);
  const bool packed_int_fc1_shape_supported = onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_supported(
      sm_, packed_int_expanded, moe_params.inter_size * 2, moe_params.hidden_size,
      static_cast<int>(fc1_expert_weight_bits_), static_cast<int>(block_size_));
  const bool packed_int_fc2_shape_supported = onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_supported(
      sm_, packed_int_expanded, moe_params.hidden_size, moe_params.inter_size,
      static_cast<int>(fc2_expert_weight_bits_), static_cast<int>(block_size_));
  const bool packed_int_routing_supported =
      moe_params.num_rows <= 256 && moe_params.num_experts <= 256 &&
      (k_ == 1 || k_ == 2 || k_ == 4 || k_ == 6 || k_ == 8);
  if (enable_int2_gemv_ && !has_any_zero_point && is_fused_swiglu && swiglu_fusion == 1 &&
      (block_size_ == 64 || block_size_ == 128) && packed_fc1_weights_ != nullptr &&
      packed_fc2_weights_ != nullptr) {
    use_packed_int_gemv = packed_int_fc1_shape_supported && packed_int_fc2_shape_supported &&
                          packed_int_routing_supported;
  }
  if (enable_kernel_debug_info_ && enable_int2_gemv_ && !use_packed_int_gemv) {
    LOGS_DEFAULT(WARNING) << "QMoE packed INT GEMV gate: fp16=" << is_fp16_
                          << " zero_points=" << has_any_zero_point
                          << " fused_swiglu=" << is_fused_swiglu
                          << " swiglu_fusion=" << swiglu_fusion
                          << " block_size=" << block_size_
                          << " fc1_bits=" << fc1_expert_weight_bits_
                          << " fc2_bits=" << fc2_expert_weight_bits_
                          << " fc1_packed=" << (packed_fc1_weights_ != nullptr)
                          << " fc2_packed=" << (packed_fc2_weights_ != nullptr)
                          << " fc1_shape=" << packed_int_fc1_shape_supported
                          << " fc2_shape=" << packed_int_fc2_shape_supported
                          << " routing=" << packed_int_routing_supported;
  }
  const bool use_packed_int_prefill =
      enable_int2_prefill_ && !use_packed_int_gemv && !has_any_zero_point && is_fused_swiglu &&
      swiglu_fusion == 1 && packed_fc1_weights_ != nullptr && packed_fc2_weights_ != nullptr &&
      moe_params.num_rows > 0 && packed_int_expanded <= std::numeric_limits<int>::max() &&
      moe_params.hidden_size > 0 && moe_params.hidden_size <= std::numeric_limits<int>::max() &&
      moe_params.hidden_size % 64 == 0 && moe_params.inter_size > 0 &&
      moe_params.inter_size <= std::numeric_limits<int>::max() / 2 && moe_params.inter_size % 64 == 0 &&
      moe_params.num_experts > 0 && moe_params.num_experts <= 256;
  int64_t effective_row_tile_size = row_tile_size_;
  size_t packed_prefill_workspace_bytes = 0;
  if (use_packed_int_prefill) {
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    constexpr size_t kMaxPackedPrefillScratchBytes = 256 * 1024 * 1024;
    ck::Int2MoePrefillParams sizing_params;
    sizing_params.hidden_size = static_cast<int>(moe_params.hidden_size);
    sizing_params.inter_size = static_cast<int>(moe_params.inter_size);
    sizing_params.num_experts = static_cast<int>(moe_params.num_experts);
    sizing_params.top_k = static_cast<int>(k_);
    size_t scale_scratch_bytes = 0;
    if (!gemv_int_fc1_scales_ && fc1_scales) {
      scale_scratch_bytes = SafeInt<size_t>(scale_scratch_bytes) + fc1_scales->SizeInBytes();
    }
    if (!gemv_int_fc2_scales_ && fc2_scales) {
      scale_scratch_bytes = SafeInt<size_t>(scale_scratch_bytes) + fc2_scales->SizeInBytes();
    }
    effective_row_tile_size = row_tile_size_ == qmoe::kDisabledRowTileSize
                                  ? moe_params.num_rows
                                  : std::min(moe_params.num_rows, row_tile_size_);
    for (;;) {
      const auto plan = qmoe::MakeRowTilePlan(moe_params.num_rows, effective_row_tile_size, true);
      sizing_params.num_rows = plan.rows_per_tile;
      packed_prefill_workspace_bytes = ck::GetInt2MoePrefillWorkspaceSize(sizing_params);
      sizing_params.num_rows = plan.RowsInTile(plan.TileCount() - 1);
      packed_prefill_workspace_bytes = std::max(
          packed_prefill_workspace_bytes, ck::GetInt2MoePrefillWorkspaceSize(sizing_params));
      const size_t packed_scratch_bytes = SafeInt<size_t>(packed_prefill_workspace_bytes) +
                                          qmoe::MakeScratchLayout(0, plan, k_).total_bytes + scale_scratch_bytes;
      if (packed_scratch_bytes <= kMaxPackedPrefillScratchBytes) {
        break;
      }
      ORT_RETURN_IF_NOT(effective_row_tile_size > 1,
                        "QMoE packed INT prefill requires ", packed_scratch_bytes,
                        " bytes of scratch for one row, exceeding the packed scratch limit of ",
                        kMaxPackedPrefillScratchBytes, " bytes.");
      effective_row_tile_size = std::max<int64_t>(1, effective_row_tile_size / 2);
    }
  }
  const bool use_packed_int = use_packed_int_gemv || use_packed_int_prefill;
  const bool use_fp8_fused = is_block_fp8 && enable_fp8_fused_ &&
                             (kernel_activation_type == ActivationType::Silu ||
                              kernel_activation_type == ActivationType::Swiglu) &&
                             moe_params.num_rows <= 65535 / k_;
  if (use_int_dequant_fallback && !use_packed_int) {
    const size_t total_dequant_bytes = SafeInt<size_t>(int_dequant_fc1_bytes) + int_dequant_fc2_bytes;
    ORT_RETURN_IF_NOT(total_dequant_bytes <= static_cast<size_t>(int_dequant_max_scratch_bytes_),
                      "INT2 or mixed-width CUDA QMoE dense fallback requires ", total_dequant_bytes,
                      " bytes of dequantized weight scratch, exceeding the configured limit of ",
                      int_dequant_max_scratch_bytes_, " bytes. Increase ",
                      kQMoEIntDequantMaxScratchBytesConfig, " only for bounded correctness workloads.");
  }

  const qmoe::RowTilePlan row_tile_plan =
      qmoe::MakeRowTilePlan(
          moe_params.num_rows, effective_row_tile_size,
          effective_row_tile_size != qmoe::kDisabledRowTileSize && !use_fp4_gemv && !use_packed_int_gemv);
  const bool use_nvfp4_compaction = is_nvfp4 && !route_native_fp4 && !use_fp4_gemv;
  const bool use_expert_compaction = is_block_fp8 || use_nvfp4_compaction;
#if defined(_WIN32)
  static const int max_compact_experts_env =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_COMPACT_EXPERTS", 64);
#else
  static const int max_compact_experts_env =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_COMPACT_EXPERTS", 0);
#endif
  const int default_compact_experts =
      (offload_backend_ != nullptr && max_compact_experts_env <= 0)
          ? static_cast<int>(offload_policy_.prefill_wave_experts)
          : max_compact_experts_env;
  const int dynamic_compact_experts =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS", default_compact_experts);
  static const bool dequant_transient_env =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_DEQUANT_TRANSIENT", 0) != 0;
  const int full_active_experts =
      (use_nvfp4_compaction && (dynamic_compact_experts > 0 || offload_backend_ != nullptr))
          ? narrow<int>(moe_params.num_experts)
          : (use_expert_compaction ? qmoe::MaxActiveExperts(row_tile_plan, narrow<int>(moe_params.num_experts), narrow<int>(k_))
                                   : narrow<int>(moe_params.num_experts));
  int eff_compact_experts =
      (offload_backend_ != nullptr && dynamic_compact_experts >= full_active_experts && full_active_experts > 1)
          ? std::max(1, full_active_experts / 2)
          : dynamic_compact_experts;
  if (offload_backend_ != nullptr && use_nvfp4_compaction &&
      offload_policy_.device_working_bytes_explicit && offload_policy_.device_working_bytes > 0) {
    const size_t elem_bytes = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    const size_t per_expert_dequant_bytes =
        static_cast<size_t>(3) * static_cast<size_t>(moe_params.hidden_size) *
        static_cast<size_t>(moe_params.inter_size) * elem_bytes;
    if (per_expert_dequant_bytes > 0) {
      const int max_wave_by_working_budget = static_cast<int>(std::clamp<size_t>(
          (offload_policy_.device_working_bytes * 3 / 4) / per_expert_dequant_bytes,
          size_t{1}, static_cast<size_t>(full_active_experts)));
      if (eff_compact_experts <= 0 || eff_compact_experts > max_wave_by_working_budget) {
        eff_compact_experts = max_wave_by_working_budget;
      }
    }
  }
  const int runner_num_experts =
      (use_nvfp4_compaction && eff_compact_experts > 0)
          ? std::min(full_active_experts, eff_compact_experts)
          : full_active_experts;
  const int num_compact_passes =
      (use_nvfp4_compaction && runner_num_experts < full_active_experts)
          ? (full_active_experts + runner_num_experts - 1) / runner_num_experts
          : 1;

  // Profile and capture the best tactics under the profiler mutex, then release the mutex so
  // that scratch allocation, weight dequantization, scale prepping, softmax, and other
  // CPU-bound work can proceed concurrently across QMoE inferences. The mutex is reacquired
  // around setTactic + runMoe because they mutate shared `m_moe_runner` state.
  using MoeConfig = onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::Config;
  struct RunnerTileConfig {
    int64_t num_rows = 0;
    std::optional<MoeConfig> config1;
    std::optional<MoeConfig> config2;
    size_t workspace_size = 0;
  };
  std::array<int64_t, 2> distinct_tile_rows{row_tile_plan.rows_per_tile, 0};
  size_t distinct_tile_row_count = 1;
  const int64_t final_tile_rows = row_tile_plan.RowsInTile(row_tile_plan.TileCount() - 1);
  if (final_tile_rows != row_tile_plan.rows_per_tile) {
    distinct_tile_rows[distinct_tile_row_count++] = final_tile_rows;
  }
  std::array<RunnerTileConfig, 2> runner_tile_configs{};
  size_t runner_tile_config_count = 0;
  size_t workspace_size = 0;
  // The packed INT GEMV path allocates its own scratch below and returns before reaching the
  // dense grouped-GEMM tile loop, so profiling/sizing the dense runner here would be pure
  // overhead (mutex, two dense-tactic profiling launches, and an unused large workspace
  // allocation). Skip it entirely for that path; workspace_size stays 0.
  if (!use_packed_int && !use_fp8_fused) {
    std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);

    // Profiling launches grouped-GEMM kernels, records/synchronizes CUDA events, and
    // allocates/frees scratch from the temp allocator on the compute stream. All of these are
    // illegal while that stream is being captured into a CUDA graph; performing them corrupts the
    // capture and later surfaces as an illegal memory access (CUDA 700) reported at a downstream
    // MoE kernel launch (e.g. moe_kernels.cu cudaFuncSetAttribute). During capture we therefore
    // skip profiling and reuse a config cached from an earlier non-capturing run, falling back to
    // the default tactic when nothing is cached.
    cudaStream_t compute_stream = Stream(context);
    // compute_stream == nullptr is the legacy default stream (stream 0), which can itself be
    // under CUDA-graph capture; cudaStreamIsCapturing handles it, so query it directly rather
    // than assuming a null stream is never capturing.
    const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(compute_stream);

    // Use profiler with proper weight type for quantized weights. A partial final tile can land
    // in a different M bucket, so profile and retain a config for each distinct actual tile size.
    if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
      auto tactics = active_runner->getTactics();
      std::optional<MoeConfig> deterministic_config;
      if (!tactics.empty()) {
        deterministic_config = tactics[0];
      }
      for (size_t i = 0; i < distinct_tile_row_count; ++i) {
        const int64_t tile_rows = distinct_tile_rows[i];
        std::optional<MoeConfig> config1 = deterministic_config;
        std::optional<MoeConfig> config2 = deterministic_config;
        if (is_nvfp4 && !use_fp4_gemv && !route_native_fp4) {
          const double rows_per_expert = static_cast<double>(tile_rows) * static_cast<double>(k_) / runner_num_experts;
          if (auto fc1 = PickPrefillTactic(tactics, false, rows_per_expert)) config1 = fc1;
          if (auto fc2 = PickPrefillTactic(tactics, true, rows_per_expert)) config2 = fc2;
        }
        RunnerTileConfig tile_config{tile_rows, config1, config2, 0};
        active_runner->setTactic(tile_config.config1, tile_config.config2);
        tile_config.workspace_size = active_runner->getWorkspaceSize(
            tile_rows, moe_params.hidden_size, moe_params.inter_size, runner_num_experts, k_,
            kernel_activation_type, parallelism_config, use_awq, swiglu_fusion);
        workspace_size = std::max(workspace_size, tile_config.workspace_size);
        runner_tile_configs[runner_tile_config_count++] = std::move(tile_config);
      }
    } else {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      mGemmProfiler.setAllocator(std::move(allocator));
      mGemmProfiler.setProfilerParams(runner_num_experts, static_cast<int>(k_),
                                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                      fp4_sm80_prefill ? int64_t{32} : static_cast<int64_t>(block_size_), kernel_activation_type,
                                      false, true, parallelism_config, sm_);

      onnxruntime::llm::nvinfer::DataType dtype = is_fp16_ ? onnxruntime::llm::nvinfer::DataType::kHALF : onnxruntime::llm::nvinfer::DataType::kBF16;
      if (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_) {
        dtype = onnxruntime::llm::nvinfer::DataType::kFP8;
      }
      // Weight type: FP4 for MXFP4, INT4 for 4-bit integer, INT8 for 8-bit integer. When the
      // native FP4 path routes to the dense fallback for this call, profile the dense (A16) tactic.
      onnxruntime::llm::nvinfer::DataType wtype;
      if (is_nvfp4) {
        // Native NVFP4 routes prefill through the FP4xFP4 block-scaled grouped GEMM (profile the
        // kFP4 tactic); decode and the large-per-expert-M fallback run the dense (A16) tactic.
        wtype = route_native_fp4 ? onnxruntime::llm::nvinfer::DataType::kFP4 : dtype;
      } else if (is_fp4) {
        // fp4_sm80_prefill runs the e2m1 weights through the SM80 fused-dequant grouped GEMM,
        // whose scratch + groupwise scale layout match INT4-groupwise (4-bit weight + activation-dtype
        // scales). Profile against kINT4 so the workspace is sized for the groupwise path
        // rather than the FP4 TMA block-scale layout; the launched kernel is still the e2m1
        // kernel (templated on the runner's WeightType), and the profiler's per-tactic
        // try/catch discards tile shapes that fail can_implement.
        wtype = fp4_sm80_prefill ? onnxruntime::llm::nvinfer::DataType::kINT4
                                 : (route_native_fp4 ? onnxruntime::llm::nvinfer::DataType::kFP4 : dtype);
      } else if (is_wfp4afp8) {
        // Native W4A8 path uses FP8 activation + FP4 weights through the block-scaled dispatch.
        // Profile against the FP4 weight tactic; fall back to dense dtype when the dequant path is selected.
        wtype = use_wfp4afp8_dequant_fallback_ ? dtype : onnxruntime::llm::nvinfer::DataType::kFP4;
      } else if (is_fp8) {
        wtype = use_fp8_dequant_fallback_ ? dtype : onnxruntime::llm::nvinfer::DataType::kFP8;
      } else if (use_int_dequant_fallback) {
        wtype = dtype;
      } else {
        wtype = (expert_weight_bits_ == 4) ? onnxruntime::llm::nvinfer::DataType::kINT4
                                           : onnxruntime::llm::nvinfer::DataType::kINT8;
      }

      using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId;
      using onnxruntime::llm::kernels::weight_only::GemmDims;

      // For gated activations (SwiGLU), fc1_out_size is doubled
      int64_t fc1_out_size = static_cast<int64_t>(moe_params.inter_size);
      if (is_fused_swiglu || split_fp8_fc1) {
        fc1_out_size = static_cast<int64_t>(moe_params.inter_size) * 2;
      }

      // GEMM 1: N=fc1_out_size (doubled for gated), K=hidden_size
      MoeGemmId id1(static_cast<int>(fc1_out_size), static_cast<int>(moe_params.hidden_size), dtype, wtype, MoeGemmId::GemmType::Gemm1);
      MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, wtype, MoeGemmId::GemmType::Gemm2);

      for (size_t i = 0; i < distinct_tile_row_count; ++i) {
        const int64_t tile_rows = distinct_tile_rows[i];
        RunnerTileConfig tile_config;
        tile_config.num_rows = tile_rows;
        if (!stream_is_capturing) {
          // profileTactics caches per (GemmId, M bucket). Use the tile's actual M so both the
          // configured tile and a non-divisible final tile select tactics from their own bucket.
          GemmDims dims1(tile_rows, tile_rows, fc1_out_size,
                         static_cast<int64_t>(moe_params.hidden_size));
          mGemmProfiler.profileTactics(active_runner, dims1, id1, compute_stream);
          GemmDims dims2(tile_rows, tile_rows, static_cast<int64_t>(moe_params.hidden_size),
                         static_cast<int64_t>(moe_params.inter_size));
          mGemmProfiler.profileTactics(active_runner, dims2, id2, compute_stream);
        }
        tile_config.config1 = mGemmProfiler.getBestConfig(static_cast<int>(tile_rows), id1);
        tile_config.config2 = mGemmProfiler.getBestConfig(static_cast<int>(tile_rows), id2);

        // Capture-safe fallback: if profiling was skipped (graph capture) and no tuned config was
        // cached from a prior non-capturing run, use the runner's default tactic instead of leaving
        // the config unset.
        if (!tile_config.config1 || !tile_config.config2) {
          auto tactics = active_runner->getTactics();
          if (!tactics.empty()) {
            if (!tile_config.config1) {
              tile_config.config1 = tactics[0];
            }
            if (!tile_config.config2) {
              tile_config.config2 = tactics[0];
            }
          }
        }

        active_runner->setTactic(tile_config.config1, tile_config.config2);
        tile_config.workspace_size = active_runner->getWorkspaceSize(
            tile_rows, moe_params.hidden_size, moe_params.inter_size, runner_num_experts, k_,
            kernel_activation_type, parallelism_config, use_awq, swiglu_fusion);
        workspace_size = std::max(workspace_size, tile_config.workspace_size);
        runner_tile_configs[runner_tile_config_count++] = std::move(tile_config);
      }
    }
  }
  // Lock released — concurrent QMoE inferences can now run prep work in parallel.

  // One allocation holds the maximum runner workspace reported for any actual tile plus tile-local
  // routing metadata. Every tile completes its launches on the same stream before the next tile
  // reuses these addresses.
  const qmoe::ScratchLayout scratch_layout =
      qmoe::MakeScratchLayout(workspace_size, row_tile_plan, k_);
  const size_t total_scratch_bytes = scratch_layout.total_bytes;

  auto work_space = (dequant_transient_env && !use_fp4_gemv && !use_packed_int)
                        ? GetTransientScratchBuffer<void>(total_scratch_bytes)
                        : GetScratchBuffer<void>(total_scratch_bytes, GetComputeStream(context));
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + workspace_size);
  int* expert_indices =
      reinterpret_cast<int*>(workspace_ptr + workspace_size + scratch_layout.scales_bytes);
  int* unpermuted_row_to_permuted_row =
      reinterpret_cast<int*>(workspace_ptr + workspace_size + scratch_layout.scales_bytes +
                             scratch_layout.indices_bytes);

  cudaStream_t stream = Stream(context);

  bool is_fp16 = input->IsDataType<MLFloat16>();
  bool is_bf16 = input->IsDataType<BFloat16>();
  auto route_tile = [&](int64_t row_offset, int64_t tile_rows) {
    onnxruntime::llm::kernels::cutlass_kernels::FusedRoutingParams fused_routing;
    const size_t router_element_offset =
        SafeInt<size_t>(row_offset) * SafeInt<size_t>(moe_params.num_experts);
    const size_t router_byte_offset =
        SafeInt<size_t>(router_element_offset) * router_probs->DataType()->Size();
    const void* tile_router_probs =
        reinterpret_cast<const char*>(router_probs->DataRaw()) + router_byte_offset;

    // Fused routing remains limited to an untiled one-row request; a one-row final tile
    // continues using the same routing implementation as the preceding tiles.
    if (!row_tile_plan.IsTiled() && !is_fp4_family && !use_packed_int && !is_block_fp8 &&
        onnxruntime::llm::kernels::cutlass_kernels::isFusedMoeRoutingSupported(
            tile_rows, static_cast<int>(moe_params.num_experts),
            static_cast<int>(moe_params.num_experts) / parallelism_config.ep_size,
            static_cast<int>(k_), parallelism_config.ep_size)) {
      fused_routing.router_logits = tile_router_probs;
      fused_routing.token_selected_experts = expert_indices;
      fused_routing.token_final_scales = expert_scales;
      fused_routing.normalize_routing_weights = normalize_routing_weights_;
    } else if (is_fp16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const half*>(tile_router_probs),
          expert_scales,
          expert_indices,
          static_cast<int>(tile_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    } else if (is_bf16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const __nv_bfloat16*>(tile_router_probs),
          expert_scales,
          expert_indices,
          static_cast<int>(tile_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    } else {
      LaunchSoftmaxTopK(
          reinterpret_cast<const float*>(tile_router_probs),
          expert_scales,
          expert_indices,
          static_cast<int>(tile_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    }
    return fused_routing;
  };

  if (use_fp8_fused) {
    Tensor* output = context->Output(0, input->Shape());
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    const int num_experts = narrow<int>(moe_params.num_experts);
    const int hidden = narrow<int>(moe_params.hidden_size);
    const int inter = narrow<int>(moe_params.inter_size);
    const int fusion = split_fp8_fc1 ? 1 : (is_fused_swiglu ? static_cast<int>(std::max<int64_t>(1, swiglu_fusion_)) : 0);
    const int fc1_n = inter * (fusion ? 2 : 1);
    const int max_routes = narrow<int>(row_tile_plan.rows_per_tile * k_);
    const size_t element_size = input->DataType()->Size();
    const size_t projection_element_size = fusion ? element_size : sizeof(float);
    auto row_map = GetScratchBuffer<int>(max_routes, GetComputeStream(context));
    auto sorted_experts = GetScratchBuffer<int>(max_routes, GetComputeStream(context));
    auto expert_offsets = GetScratchBuffer<int64_t>(num_experts + 1, GetComputeStream(context));
    auto tile_offsets = GetScratchBuffer<int>(num_experts + 1, GetComputeStream(context));
    const size_t projection_bytes = SafeInt<size_t>(max_routes) * fc1_n * projection_element_size;
    auto projection = GetScratchBuffer<void>(projection_bytes, GetComputeStream(context));
    auto activation = GetScratchBuffer<void>(SafeInt<size_t>(max_routes) * inter * element_size, GetComputeStream(context));
    auto result = GetScratchBuffer<void>(SafeInt<size_t>(max_routes) * hidden * element_size, GetComputeStream(context));
    auto scale_type = [](const Tensor* tensor) {
      return tensor->IsDataType<float>() ? 0 : (tensor->IsDataType<MLFloat16>() ? 1 : 2);
    };
    const auto* fc1_scale_tensor = context->Input<Tensor>(3);
    const auto* fc2_scale_tensor = context->Input<Tensor>(6);
    const auto* fc3_scale_tensor = split_fp8_fc1 ? context->Input<Tensor>(9) : nullptr;
    for (int64_t tile = 0; tile < row_tile_plan.TileCount(); ++tile) {
      const int rows = narrow<int>(row_tile_plan.RowsInTile(tile));
      const int routes = rows * narrow<int>(k_);
      const int64_t row_offset = row_tile_plan.RowOffset(tile);
      route_tile(row_offset, rows);
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
      if (routing_snapshot_) {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(expert_indices, routes, stream));
      }
#endif
      IAllocatorUniquePtr<int> counts;
      IAllocatorUniquePtr<int> cumulative_counts;
      IAllocatorUniquePtr<int> blocked_rows;
      if (!ck::fusedBuildExpertMapsSortFirstToken(
              expert_indices, row_map.get(), unpermuted_row_to_permuted_row, sorted_experts.get(), expert_offsets.get(),
              rows, num_experts, narrow<int>(k_), 0, num_experts, stream)) {
        const int64_t tokens_per_block = ck::computeNumTokensPerBlock(rows, num_experts);
        const int64_t blocks = onnxruntime::llm::common::ceilDiv(rows, tokens_per_block);
        counts = GetScratchBuffer<int>(SafeInt<size_t>(num_experts) * blocks, GetComputeStream(context));
        cumulative_counts = GetScratchBuffer<int>(SafeInt<size_t>(num_experts) * blocks, GetComputeStream(context));
        blocked_rows = GetScratchBuffer<int>(SafeInt<size_t>(num_experts) * rows, GetComputeStream(context));
        ck::threeStepBuildExpertMapsSortFirstToken(
            expert_indices, sorted_experts.get(), row_map.get(), unpermuted_row_to_permuted_row, expert_offsets.get(),
            counts.get(), cumulative_counts.get(), blocked_rows.get(), rows, num_experts, k_, 0, stream);
      }
      const bool gemm = rows > 8;
      if (gemm) {
        LaunchQMoEFp8ExpertTiles(expert_offsets.get(), tile_offsets.get(), num_experts, stream);
      }
      auto execute = [&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        QMoEFp8ProjectionParams params;
        params.weights = static_cast<const uint8_t*>(fc1_experts_weights->DataRaw());
        params.scales = fc1_scale_tensor->DataRaw();
        params.scale_type = scale_type(fc1_scale_tensor);
        params.up_weights = split_fp8_fc1 ? static_cast<const uint8_t*>(fc3_experts_weights->DataRaw()) : nullptr;
        params.up_scales = fc3_scale_tensor ? fc3_scale_tensor->DataRaw() : nullptr;
        params.up_scale_type = fc3_scale_tensor ? scale_type(fc3_scale_tensor) : 0;
        params.row_to_unpermuted = row_map.get();
        params.experts = sorted_experts.get();
        params.expert_offsets = expert_offsets.get();
        params.tile_offsets = gemm ? tile_offsets.get() : nullptr;
        params.num_experts = num_experts;
        params.num_rows = rows;
        params.expanded_rows = routes;
        params.n = fc1_n;
        params.k = hidden;
        params.block_size = narrow<int>(block_size_);
        params.fusion = fusion;
        const auto* tile_input = static_cast<const T*>(input->DataRaw()) + row_offset * hidden;
        const auto* fc1_bias = fc1_experts_bias_optional ? static_cast<const T*>(fc1_experts_bias_optional->DataRaw()) : nullptr;
        auto project_and_activate = [&](auto* projected) {
          LaunchQMoEFp8Projection(params, tile_input, projected, stream);
          LaunchQMoEFp8Activation(projected, fc1_bias, static_cast<T*>(activation.get()),
                                  sorted_experts.get(), routes, inter, fusion,
                                  split_fp8_fc1 || !fusion ? 1.0f : activation_alpha_,
                                  split_fp8_fc1 ? 0.0f : activation_beta_,
                                  split_fp8_fc1 ? std::numeric_limits<float>::infinity() : swiglu_limit_, stream);
        };
        if (fusion) {
          project_and_activate(static_cast<T*>(projection.get()));
        } else {
          // Match the dense SiLU epilogue: narrow only after bias and activation.
          project_and_activate(static_cast<float*>(projection.get()));
        }
        params.weights = static_cast<const uint8_t*>(fc2_experts_weights->DataRaw());
        params.scales = fc2_scale_tensor->DataRaw();
        params.scale_type = scale_type(fc2_scale_tensor);
        params.up_weights = nullptr;
        params.up_scales = nullptr;
        params.row_to_unpermuted = nullptr;
        params.n = hidden;
        params.k = inter;
        params.fusion = 0;
        LaunchQMoEFp8Projection(params, static_cast<const T*>(activation.get()), static_cast<T*>(result.get()), stream);
        const auto* fc2_bias = fc2_experts_bias_optional ? static_cast<const T*>(fc2_experts_bias_optional->DataRaw()) : nullptr;
        ck::finalizeMoeRoutingKernelLauncher<T, T, T>(
            static_cast<const T*>(result.get()), static_cast<T*>(output->MutableDataRaw()) + row_offset * hidden,
            fc2_bias, expert_scales, unpermuted_row_to_permuted_row, row_map.get(), expert_indices, expert_offsets.get(),
            rows, hidden, k_, num_experts, parallelism_config, false, stream);
      };
      if (is_fp16_) {
        execute(static_cast<half*>(nullptr));
      } else {
        execute(static_cast<__nv_bfloat16*>(nullptr));
      }
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
      if (routing_snapshot_) {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
      }
#endif
    }
    if (enable_kernel_debug_info_) {
      const size_t activation_bytes = SafeInt<size_t>(projection_bytes) +
                                      SafeInt<size_t>(max_routes) * (inter + hidden) * element_size;
      PrintQMoEKernelDebugInfo(row_tile_plan.rows_per_tile > 8 ? "fp8_fused_gemm" : "fp8_fused_gemv",
                               moe_params.num_rows, row_tile_plan.rows_per_tile, final_tile_rows,
                               max_routes, final_tile_rows * k_, 0, total_scratch_bytes + activation_bytes);
      std::cout << "QMoE FP8 ExpertCapacity=" << num_experts << " DequantWeightBytes=0" << std::endl;
    }
    return Status::OK();
  }

  // Holders for packed tensors (if packing is needed for SwiGLU)
  IAllocatorUniquePtr<void> packed_fc1_scales_holder;
  IAllocatorUniquePtr<void> packed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc1_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc2_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc2_zp_holder;

  // Determine effective pointers for scales and zero points
  const void* p_fc1_scales = nullptr;
  const void* p_fc1_zp = nullptr;
  const void* p_fc2_scales = nullptr;
  const void* p_fc2_zp = nullptr;

  // Use pre-packed buffers if available, otherwise use input tensors (and potentially compute bias on the fly)
  IAllocatorUniquePtr<void> transient_fc1_bias;
  IAllocatorUniquePtr<void> transient_fc2_bias;

  auto prepare_scale_zp = [&](const Tensor* scales, const Tensor* zeros,
                              const IAllocatorUniquePtr<void>& packed_scale, const IAllocatorUniquePtr<void>& packed_bias,
                              IAllocatorUniquePtr<void>& transposed_scale_holder,
                              IAllocatorUniquePtr<void>& transposed_zp_holder,
                              IAllocatorUniquePtr<void>& transient_bias,
                              const void*& eff_scale, const void*& eff_zp,
                              int64_t weight_bits) {
    if (packed_scale) {
      eff_scale = packed_scale.get();
    } else if (scales) {
      eff_scale = scales->DataRaw();

      // For block-wise quantization, Cutlass expects scales laid out as [Experts, Blocks, N].
      // Input tensors are provided as [Experts, N, Blocks], so transpose when PrePack is not used.
      auto scale_shape = scales->Shape();
      if (block_size_ > 0 && scale_shape.NumDimensions() == 3) {
        size_t rows = scale_shape[1];   // N
        size_t cols = scale_shape[2];   // Blocks
        size_t batch = scale_shape[0];  // Experts
        size_t bytes = scales->SizeInBytes();

        transposed_scale_holder = GetScratchBuffer<void>(bytes, GetComputeStream(context));
        eff_scale = transposed_scale_holder.get();

        if (scales->IsDataType<MLFloat16>()) {
          LaunchQMoETranspose2D(static_cast<const half*>(scales->DataRaw()), static_cast<half*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        } else if (scales->IsDataType<BFloat16>()) {
          LaunchQMoETranspose2D(static_cast<const __nv_bfloat16*>(scales->DataRaw()), static_cast<__nv_bfloat16*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        } else {
          ORT_ENFORCE(scales->IsDataType<float>(),
                      "QMoE scale transpose supports only float16, bfloat16, and float tensors.");
          LaunchQMoETranspose2D(static_cast<const float*>(scales->DataRaw()), static_cast<float*>(transposed_scale_holder.get()), batch, rows, cols, stream);
        }
      }
    }

    if (packed_bias) {
      eff_zp = packed_bias.get();
    } else if (zeros) {
      if (weight_bits == 4 || (weight_bits == 8 && block_size_ > 0)) {
        // Compute bias on the fly: bias = -zp * scale
        // We need 'eff_scale' to be available.
        if (eff_scale && block_size_ > 0) {
          size_t num_elements = zeros->Shape().Size();
          // Determine type size based on scale type
          bool is_fp16 = scales->IsDataType<MLFloat16>();
          bool is_bf16 = scales->IsDataType<BFloat16>();
          size_t bytes = num_elements * (is_fp16 ? 2 : 4);

          transient_bias = GetScratchBuffer<void>(bytes, GetComputeStream(context));
          eff_zp = transient_bias.get();

          const uint8_t* p_zp = static_cast<const uint8_t*>(zeros->DataRaw());

          // Determine whether zeros are stored packed (two uint4 ZP per byte) or unpacked.
          // For block-wise 4-bit quantization, scales have shape [E, N, K_blocks] and zeros
          // have shape either [E, N, K_blocks] (unpacked) or [E, N, ceil(K_blocks/2)] (packed).
          // Compare the last dim of zeros vs scales explicitly instead of relying on a fragile
          // numeric heuristic on Shape().Size() ratios, which can mis-classify pathological
          // shapes (e.g., K_blocks=1 where ceil(1/2)=1 makes packed indistinguishable from
          // unpacked by element count alone).
          bool zp_is_packed_4bit = false;
          if (weight_bits == 4) {
            const auto& zeros_shape = zeros->Shape();
            const auto& scales_shape = scales->Shape();
            ORT_ENFORCE(zeros_shape.NumDimensions() == 3 && scales_shape.NumDimensions() == 3,
                        "Block-wise 4-bit zeros and scales must be 3D, got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            ORT_ENFORCE(zeros_shape[0] == scales_shape[0] && zeros_shape[1] == scales_shape[1],
                        "Block-wise 4-bit zeros and scales must agree on the first two dims, got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            const int64_t scales_k = scales_shape[2];
            const int64_t zeros_k = zeros_shape[2];
            const int64_t expected_packed_k = (scales_k + 1) / 2;
            if (zeros_k == scales_k) {
              zp_is_packed_4bit = false;
            } else if (zeros_k == expected_packed_k) {
              zp_is_packed_4bit = true;
            } else {
              ORT_THROW("Block-wise 4-bit zeros last dim must be ", scales_k,
                        " (unpacked) or ", expected_packed_k, " (packed). Got zeros=",
                        zeros_shape.ToString(), ", scales=", scales_shape.ToString());
            }
          }

          // Transpose ZP if needed (for 3D ZP)
          auto shape = zeros->Shape();
          IAllocatorUniquePtr<void> temp_zp_transposed;
          if (shape.NumDimensions() == 3 && shape[2] > 1) {
            size_t rows = shape[1];   // N
            size_t cols = shape[2];   // Blocks
            size_t batch = shape[0];  // Experts
            size_t zp_bytes = zeros->SizeInBytes();
            temp_zp_transposed = GetScratchBuffer<void>(zp_bytes, GetComputeStream(context));
            LaunchQMoETranspose2D(p_zp, static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
            p_zp = static_cast<const uint8_t*>(temp_zp_transposed.get());
          }

          if (is_fp16) {
            if (weight_bits == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          } else if (is_bf16) {
            if (weight_bits == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const __nv_bfloat16*>(eff_scale),
                  static_cast<__nv_bfloat16*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          } else {
            if (weight_bits == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else if (zp_is_packed_4bit) {
              size_t scale_el = scales->Shape().Size();
              int N_stride = static_cast<int>(zeros->Shape()[1]);
              LaunchQMoEPrePackPacked4BitZPKernel(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(scale_el),
                  N_stride,
                  stream);
            } else {
              LaunchQMoEPrePackZP(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  stream);
            }
          }
        }
      } else {
        // For 8-bit, ZP is used as is (or transposed).
        // Since we are not packing, we use the raw pointer unless transpose is needed.
        // Transpose on the fly is tricky without allocation. BUT, ComputeInternal is usually called
        // with pre-packed weights/scales if coming from unit tests or offline tools.
        // If not pre-packed (e.g. dynamic graph), we might need to transpose if 3D.
        // For now, assuming standard path or 1D ZP for 2D weights.
        // If 3D, we must transpose.
        auto shape = zeros->Shape();
        if (shape.NumDimensions() == 3 && shape[2] > 1) {
          // Need temporary buffer for transpose
          size_t bytes = zeros->SizeInBytes();
          transposed_zp_holder = GetScratchBuffer<void>(bytes, GetComputeStream(context));
          eff_zp = transposed_zp_holder.get();

          size_t rows = shape[1];   // N
          size_t cols = shape[2];   // Blocks
          size_t batch = shape[0];  // Experts
          LaunchQMoETranspose2D(static_cast<const uint8_t*>(zeros->DataRaw()), static_cast<uint8_t*>(transposed_zp_holder.get()), batch, rows, cols, stream);
        } else {
          eff_zp = zeros->DataRaw();
        }
      }
    }
  };

  if (!use_int_dequant_fallback || use_packed_int) {
    const auto& fc1_prepared_scales = use_packed_int ? gemv_int_fc1_scales_ : packed_fc1_scales_;
    const auto& fc2_prepared_scales = use_packed_int ? gemv_int_fc2_scales_ : packed_fc2_scales_;
    prepare_scale_zp(fc1_scales, fc1_zeros, fc1_prepared_scales, packed_fc1_bias_,
                     transposed_fc1_scales_holder, transposed_fc1_zp_holder, transient_fc1_bias, p_fc1_scales, p_fc1_zp,
                     fc1_expert_weight_bits_);
    prepare_scale_zp(fc2_scales, fc2_zeros, fc2_prepared_scales, packed_fc2_bias_,
                     transposed_fc2_scales_holder, transposed_fc2_zp_holder, transient_fc2_bias, p_fc2_scales, p_fc2_zp,
                     fc2_expert_weight_bits_);
  }

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params;
  if (is_fp4 && fp4_sm80_prefill) {
    // SM80 FP4 grouped-GEMM prefill: the e2m1 weights are in the SM80 CUTLASS interleaved
    // layout and the scales are the activation-dtype group scales (group=32, [E, K/32, N]) the GEMV path
    // already built (gemv_fp4_fc*_scales_ = e8m0_block_scale * global_scale). Feed them through
    // the generic fine-grained groupwise path — identical plumbing to INT4 block-wise QMoE.
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        /*group_size=*/32,
        gemv_fp4_fc1_scales_.get(),
        gemv_fp4_fc2_scales_.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr);
  } else if (is_fp4) {
    // FP4 quantization: use QuantParams::FP4 with block scales and global scales
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE quant_type='fp4' requires fc1_scales, fc2_scales, fc1_global_scale, and fc2_global_scale.");
    // Only the native FP4 runner consumes block scales via quant_params; the dense fallback runner
    // receives weights already dequantized to FP16/BF16 and leaves quant_params empty.
    if (route_native_fp4) {
      using MXFPXElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF;
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::MXFP8MXFP4(
          static_cast<const MXFPXElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          static_cast<const MXFPXElementSF*>(p_fc2_block_scales),
          static_cast<const float*>(p_fc2_global_scale));
    }
  } else if (is_nvfp4) {
    // NVFP4: use QuantParams::FP4 with per-16 E4M3 weight block scales and per-expert global
    // scales. packed_fp4_*_block_scales_ holds the TMA-swizzled block scales (native path); the raw
    // e4m3 copy for the GEMV/dequant paths lives in gemv_fp4_*_block_raw_.
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE quant_type='nvfp4' requires fc1_scales, fc2_scales, fc1_global_scale, and fc2_global_scale.");
    // Only the native FP4xFP4 runner consumes block scales via quant_params; the dense fallback
    // runner receives weights already dequantized to FP16/BF16 and leaves quant_params empty.
    if (route_native_fp4) {
      using NVFP4ElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF;
      // act_global_scale = nullptr: activations are dynamically quantized to NVFP4 with per-block
      // E4M3 scales and a global scale of 1.0 inside the runner's expandInputRowsKernel.
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::FP4(
          /*fc1_act_global_scale=*/nullptr,
          static_cast<const NVFP4ElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          /*fc2_act_global_scale=*/nullptr,
          static_cast<const NVFP4ElementSF*>(p_fc2_block_scales),
          static_cast<const float*>(p_fc2_global_scale));
    }
  } else if (is_wfp4afp8) {
    // W4A8 (WFP4AFP8): MXFP4 weights + FP8 e4m3 activations.
    //   - Weight block scales (uint8 MXFPX) are read from fc*_scales (inputs 3/6)
    //   - Per-expert weight global scales come from inputs 15/16
    //   - Optional per-expert/per-tensor FP8 activation global scales come from inputs 18/19
    const void* p_fc1_block_scales = packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr);
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales = packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE quant_type='wfp4afp8' requires fc1_scales, fc2_scales, fc1_global_scale, and fc2_global_scale.");
    if (!use_wfp4afp8_dequant_fallback_) {
      // Native W4A8 path (SM100+): use QuantParams::MXFP8MXFP4 (Variant B). The activation
      // is quantized BF16/FP16 -> MXFP8 (FP8 + per-block ue8m0 scales) inside the runner's
      // expandInputRowsKernel; the activation block scales are written to fc1_fp4_act_scale_
      // at runtime. The mxfp8_mxfp4 weight_block_scale field holds the MXFP4 weight block
      // scales (same uint8 ue8m0 element type as MXFP8 activation block scales) and is
      // checked by the expansion kernel as a marker to take the MXFP8 quantization path.
      //
      // Variant A (global-scaled FP8 activation) would consume the per-expert/per-tensor
      // scale from inputs 18/19 via QuantParams::FP8MXFP4. That path requires the user to
      // feed FP8 input directly, which the QMoE op does not support (its input is BF16/FP16),
      // so we use Variant B instead. The act_scale inputs are still validated and pre-packed
      // for forward compatibility.
      using MXFPXElementSF = onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF;
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::MXFP8MXFP4(
          static_cast<const MXFPXElementSF*>(p_fc1_block_scales),
          static_cast<const float*>(p_fc1_global_scale),
          static_cast<const MXFPXElementSF*>(p_fc2_block_scales),
          static_cast<const float*>(p_fc2_global_scale));
    }
  } else if (is_fp8 && !use_fp8_dequant_fallback_) {
    // Native W8A16-FP8: per-expert global scale applied via alpha_scale_ptr_array in the epilogue.
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_global_scale && p_fc2_global_scale,
                      "QMoE native W8A16-FP8 requires fc1_global_scale and fc2_global_scale.");
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::FP8(
        static_cast<const float*>(p_fc1_global_scale),  // dequant_fc1 = per-expert weight global scale
        nullptr,                                        // quant_fc2 (not used for W8A16)
        static_cast<const float*>(p_fc2_global_scale),  // dequant_fc2 = per-expert weight global scale
        nullptr,                                        // quant_final
        nullptr,                                        // dequant_input
        false);                                         // fc2_use_per_expert_act_scale
  } else if (use_int_dequant_fallback || (is_fp8 && use_fp8_dequant_fallback_)) {
    // Dequantized weights are consumed by the dense A16 runner without quantization parameters.
  } else if (block_size_ > 0) {
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        block_size_,
        p_fc1_scales,
        p_fc2_scales,
        nullptr,
        nullptr,
        p_fc1_zp,
        p_fc2_zp);
  } else {
    // Per-column quantization
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::Int(
        p_fc1_scales,
        p_fc2_scales);
  }

  Tensor* output = context->Output(0, input->Shape());

  // ---------------------------------------------------------------------------
  // Fused MXFP4 GEMV (W4A16) decode fast path. Default-on (opt-out via ORT_ENABLE_FP4_GEMV=0)
  // on the SM<120 fallback regime. Instead of dequantizing every active expert's MXFP4 weights
  // to dense BF16/FP16 HBM, route small-decode shapes through a standalone fused pipeline:
  //   build expert maps -> expand permuted activations -> fc1 SwiGLU GEMV -> fc2 GEMV ->
  //   finalize routing. The pre-packed [E,n,k/2] weights and [E,k/32,n] scales are produced
  //   by PrePack/TryBuildGemvFp4Scales. Unsupported shapes (prefill / large batch / missing
  //   pre-pack buffers) fall through to the dequant fallback below.
  // MXFP4 uses a fixed group size of 32 (intrinsic to the e2m1 + e8m0 block format); the
  // block_size_ attribute is unset (-1) for fp4, so it is not part of the gate.
  // When native CUTLASS WFP4A16 is enabled, GEMV serves only the decode regime
  // (num_rows < fp4_prefill_min_tokens_); prefill (M >= threshold) falls through to native.
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (routing_snapshot_) {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "MoE expert tracking is enabled but its collector is unavailable.");
    ORT_RETURN_IF_ERROR(routing_snapshot_->BeginInvocation(pilot->Moe(), static_cast<size_t>(moe_params.num_experts)));
  }
#endif

  if (use_packed_int_prefill) {
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    ck::Int2MoePrefillParams params;
    params.is_bf16 = !is_fp16_;
    params.fc1_weights = static_cast<const uint8_t*>(packed_fc1_weights_.get());
    params.fc2_weights = static_cast<const uint8_t*>(packed_fc2_weights_.get());
    params.fc1_scales = p_fc1_scales;
    params.fc2_scales = p_fc2_scales;
    params.fc1_bias = fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr;
    params.fc2_bias = fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr;
    params.selected_experts = expert_indices;
    params.routing_weights = expert_scales;
    params.unpermuted_to_permuted = unpermuted_row_to_permuted_row;
    params.hidden_size = static_cast<int>(moe_params.hidden_size);
    params.inter_size = static_cast<int>(moe_params.inter_size);
    params.block_size = static_cast<int>(block_size_);
    params.fc1_weight_bits = static_cast<int>(fc1_expert_weight_bits_);
    params.fc2_weight_bits = static_cast<int>(fc2_expert_weight_bits_);
    params.num_experts = static_cast<int>(moe_params.num_experts);
    params.top_k = static_cast<int>(k_);
    params.sm = sm_;
    params.multiprocessor_count = GetDeviceProp().multiProcessorCount;
    params.alpha = activation_alpha_;
    params.beta = activation_beta_;
    params.limit = swiglu_limit_;
    params.stream = stream;
    const size_t prefill_bytes = packed_prefill_workspace_bytes;
    auto prefill_workspace = GetScratchBuffer<void>(prefill_bytes, GetComputeStream(context));
    for (int64_t tile_index = 0; tile_index < row_tile_plan.TileCount(); ++tile_index) {
      const int64_t row_offset = row_tile_plan.RowOffset(tile_index);
      const int64_t tile_rows = row_tile_plan.RowsInTile(tile_index);
      const auto fused_routing = route_tile(row_offset, tile_rows);
      ORT_ENFORCE(fused_routing.router_logits == nullptr,
                  "QMoE packed INT prefill requires materialized routing outputs.");
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
      if (routing_snapshot_) {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(
            expert_indices, SafeInt<size_t>(tile_rows) * SafeInt<size_t>(k_), stream));
      }
#endif
      const size_t element_offset = SafeInt<size_t>(row_offset) * SafeInt<size_t>(moe_params.hidden_size);
      const size_t input_byte_offset = SafeInt<size_t>(element_offset) * input->DataType()->Size();
      const size_t output_byte_offset = SafeInt<size_t>(element_offset) * output->DataType()->Size();
      params.input = reinterpret_cast<const char*>(input->DataRaw()) + input_byte_offset;
      params.output = reinterpret_cast<char*>(output->MutableDataRaw()) + output_byte_offset;
      params.num_rows = tile_rows;
      ck::RunInt2MoePrefill(params, prefill_workspace.get());
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
      if (routing_snapshot_) {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
      }
#endif
    }
    if (enable_kernel_debug_info_) {
      PrintQMoEKernelDebugInfo("packed_int_prefill", moe_params.num_rows, row_tile_plan.rows_per_tile,
                               final_tile_rows, row_tile_plan.rows_per_tile * k_, final_tile_rows * k_,
                               prefill_bytes, SafeInt<size_t>(total_scratch_bytes) + prefill_bytes);
    }
    return Status::OK();
  }

  if (use_packed_int_gemv) {
    namespace gemv = onnxruntime::llm::kernels::moe_gemv;
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    const int num_experts = static_cast<int>(moe_params.num_experts);
    const int64_t num_rows = moe_params.num_rows;
    const int64_t expanded = num_rows * static_cast<int64_t>(k_);
    const int64_t hidden = moe_params.hidden_size;
    const int64_t inter = moe_params.inter_size;

    auto p_r2u_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
    auto p_exp_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
    auto p_efto_buf = GetScratchBuffer<int64_t>(num_experts + 1, GetComputeStream(context));
    const size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    auto p_fc1_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * inter * element_size,
                                            GetComputeStream(context));
    auto p_fc2_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * hidden * element_size,
                                            GetComputeStream(context));
    int* p_r2u = p_r2u_buf.get();
    int* p_exp = p_exp_buf.get();
    int64_t* p_efto = p_efto_buf.get();

    const auto fused_routing = route_tile(0, num_rows);
    ORT_ENFORCE(fused_routing.router_logits == nullptr,
                "QMoE packed INT GEMV requires materialized routing outputs.");
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    if (routing_snapshot_) {
      ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(
          expert_indices, SafeInt<size_t>(num_rows) * SafeInt<size_t>(k_), stream));
    }
#endif
    const bool expert_maps_built = ck::fusedBuildExpertMapsSortFirstToken(
        expert_indices, p_r2u, unpermuted_row_to_permuted_row, p_exp, p_efto,
        num_rows, num_experts, static_cast<int>(k_), 0, num_experts, stream);
    ORT_ENFORCE(expert_maps_built,
                "QMoE packed INT GEMV failed to build fused expert maps for a supported routing configuration.");

    ck::ActivationParams act_params(activation_type_);
    act_params.alpha = activation_alpha_;
    act_params.beta = activation_beta_;
    act_params.swiglu_fusion = swiglu_fusion;
    act_params.limit = swiglu_limit_;
    auto run_packed_int_gemv = [&](auto* type_ptr) {
      using T = std::remove_pointer_t<decltype(type_ptr)>;
      const auto* fc1_bias = fc1_experts_bias_optional
                                 ? reinterpret_cast<const T*>(fc1_experts_bias_optional->DataRaw())
                                 : nullptr;
      const auto* fc2_bias = fc2_experts_bias_optional
                                 ? reinterpret_cast<const T*>(fc2_experts_bias_optional->DataRaw())
                                 : nullptr;
      auto* fc1_output = static_cast<T*>(p_fc1_buf.get());
      auto* fc2_output = static_cast<T*>(p_fc2_buf.get());

      auto launch_fc1 = [&](auto* weight_type) {
        using WeightType = std::remove_pointer_t<decltype(weight_type)>;
        gemv::launch_moe_gemv_int_symmetric_interleaved_swiglu<T, WeightType>(
            reinterpret_cast<const T*>(input->DataRaw()),
            static_cast<const WeightType*>(packed_fc1_weights_.get()),
            static_cast<const T*>(p_fc1_scales), fc1_bias, fc1_output,
            p_efto, p_exp, num_experts, expanded, inter, hidden, static_cast<int>(block_size_), sm_,
            act_params, p_r2u, num_rows, nullptr, stream);
      };
      auto launch_fc2 = [&](auto* weight_type) {
        using WeightType = std::remove_pointer_t<decltype(weight_type)>;
        gemv::launch_moe_gemv_int_symmetric<T, WeightType>(
            fc1_output, static_cast<const WeightType*>(packed_fc2_weights_.get()),
            static_cast<const T*>(p_fc2_scales), fc2_bias, fc2_output,
            p_efto, p_exp, num_experts, expanded, hidden, inter, static_cast<int>(block_size_), sm_, stream);
      };
      if (fc1_expert_weight_bits_ == 2) {
        launch_fc1(static_cast<cutlass::uint2b_t*>(nullptr));
      } else {
        launch_fc1(static_cast<cutlass::uint4b_t*>(nullptr));
      }
      if (fc2_expert_weight_bits_ == 2) {
        launch_fc2(static_cast<cutlass::uint2b_t*>(nullptr));
      } else {
        launch_fc2(static_cast<cutlass::uint4b_t*>(nullptr));
      }
      ck::finalizeMoeRoutingKernelLauncher<T, T, T>(
          fc2_output, reinterpret_cast<T*>(output->MutableDataRaw()),
          nullptr, expert_scales, unpermuted_row_to_permuted_row, p_r2u, expert_indices,
          p_efto, num_rows, hidden, k_, num_experts, parallelism_config, false, stream);
    };
    if (is_fp16_) {
      run_packed_int_gemv(static_cast<half*>(nullptr));
    } else {
      run_packed_int_gemv(static_cast<__nv_bfloat16*>(nullptr));
    }

    if (enable_kernel_debug_info_) {
      PrintQMoEKernelDebugInfo("packed_int_gemv", num_rows, num_rows, num_rows, expanded, expanded,
                               workspace_size, total_scratch_bytes);
    }
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    if (routing_snapshot_) {
      ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
    }
#endif
    return Status::OK();
  }

  // The FP4 DeepGEMM path, when eligible, outranks the GEMV for the same shapes.
  if (!use_fp4_deep_gemm && use_fp4_gemv) {
    namespace gemv = onnxruntime::llm::kernels::moe_gemv;
    namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
    const int num_experts = static_cast<int>(moe_params.num_experts);
    const int64_t num_rows = moe_params.num_rows;
    const int64_t expanded = num_rows * static_cast<int64_t>(k_);
    const int64_t hidden = moe_params.hidden_size;
    const int64_t inter = moe_params.inter_size;
    const int64_t fc1_n = inter * 2;
    // MXFP4 uses block size 32 (e2m1 + e8m0); NVFP4 uses block size 16 (e2m1 + e4m3).
    const int gemv_group_size = is_nvfp4 ? 16 : 32;
    // When the SM80 grouped-GEMM prefill repurposes gemv_fp4_fc*_weights_ for its pair-
    // interleaved layout, the decode GEMV either un-permutes that same buffer in-register
    // (fc*_gemv_sm80_layout, the single-copy default) or reads the dedicated GEMV-native copy
    // in gemv_fp4_fc*_weights_decode_.
    const uint8_t* gemv_fc1_weight = use_raw_nvfp4_gemv
                                         ? (packed_fp4_fc1_weights_
                                                ? static_cast<const uint8_t*>(packed_fp4_fc1_weights_.get())
                                                : static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()))
                                         : static_cast<const uint8_t*>(
                                               ((enable_fp4_sm80_gemm_ && !fc1_gemv_sm80_layout)
                                                    ? gemv_fp4_fc1_weights_decode_
                                                    : gemv_fp4_fc1_weights_)
                                                   .get());
    const uint8_t* gemv_fc2_weight = use_raw_nvfp4_gemv
                                         ? (packed_fp4_fc2_weights_
                                                ? static_cast<const uint8_t*>(packed_fp4_fc2_weights_.get())
                                                : static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()))
                                         : static_cast<const uint8_t*>(
                                               ((enable_fp4_sm80_gemm_ && !fc2_gemv_sm80_layout)
                                                    ? gemv_fp4_fc2_weights_decode_
                                                    : gemv_fp4_fc2_weights_)
                                                   .get());
    // With native FP4 prepacking, packed_fp4_*_block_scales_ holds swizzled scales, so raw GEMV reads the original
    // tensor when it is GPU-resident and uses the uploaded raw copy only for CPU-backed offload initializers.
    auto raw_row_major_ptr = [&](int input_idx, const IAllocatorUniquePtr<void>& uploaded) -> const void* {
      const Tensor* t = context->Input<Tensor>(input_idx);
      if (t != nullptr && t->Location().device.Type() != OrtDevice::CPU) return t->DataRaw();
      return uploaded ? uploaded.get() : (t != nullptr ? t->DataRaw() : nullptr);
    };
    const uint8_t* gemv_fc1_raw_block_scales =
        nvfp4_weights_row_major_ ? static_cast<const uint8_t*>(raw_row_major_ptr(3, packed_fp4_fc1_block_scales_))
        : is_nvfp4               ? static_cast<const uint8_t*>((gemv_fp4_fc1_block_raw_ ? gemv_fp4_fc1_block_raw_
                                                                                        : packed_fp4_fc1_block_scales_)
                                                                   .get())
                                 : nullptr;
    const uint8_t* gemv_fc2_raw_block_scales =
        nvfp4_weights_row_major_ ? static_cast<const uint8_t*>(raw_row_major_ptr(6, packed_fp4_fc2_block_scales_))
        : is_nvfp4               ? static_cast<const uint8_t*>((gemv_fp4_fc2_block_raw_ ? gemv_fp4_fc2_block_raw_
                                                                                        : packed_fp4_fc2_block_scales_)
                                                                   .get())
                                 : nullptr;
    const float* gemv_fc1_raw_global_scales =
        nvfp4_weights_row_major_ ? static_cast<const float*>(raw_row_major_ptr(15, packed_fc1_global_scale_))
                                 : (is_nvfp4 ? static_cast<const float*>(packed_fc1_global_scale_.get()) : nullptr);
    const float* gemv_fc2_raw_global_scales =
        nvfp4_weights_row_major_ ? static_cast<const float*>(raw_row_major_ptr(16, packed_fc2_global_scale_))
                                 : (is_nvfp4 ? static_cast<const float*>(packed_fc2_global_scale_.get()) : nullptr);
    auto p_r2u_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
    auto p_exp_buf = GetScratchBuffer<int>(expanded, GetComputeStream(context));
    auto p_efto_buf = GetScratchBuffer<int64_t>(num_experts + 1, GetComputeStream(context));
    const size_t elt = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    auto p_act_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * hidden * elt, GetComputeStream(context));
    auto p_fc1_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * inter * elt, GetComputeStream(context));
    auto p_fc2_buf = GetScratchBuffer<void>(SafeInt<size_t>(expanded) * hidden * elt, GetComputeStream(context));
    int* p_r2u = p_r2u_buf.get();
    int* p_exp = p_exp_buf.get();
    int64_t* p_efto = p_efto_buf.get();

    ck::ActivationParams act_params(activation_type_);
    act_params.alpha = activation_alpha_;
    act_params.beta = activation_beta_;
    act_params.swiglu_fusion = swiglu_fusion;
    act_params.limit = swiglu_limit_;

    // Decode routing (softmax + top-k + expert maps) in one single-CTA launch for the raw row-major NVFP4 GEMV route;
    // bit-identical to LaunchSoftmaxTopK + fusedBuildExpertMapsSortFirstToken (env ORT_FP4_GEMV_FUSED_ROUTE=0 disables).
    static const bool fp4_gemv_pdl =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_PDL", 1) != 0;
    static const bool fp4_gemv_fused_route =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_FUSED_ROUTE", 1) != 0;
    static const bool fp4_gemv_fc1_pdl =
        onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_FC1_PDL", 0) != 0;
    bool fused_route_launched = false;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    const bool routing_snapshot_active = routing_snapshot_.has_value();
#else
    constexpr bool routing_snapshot_active = false;
#endif
    if (fp4_gemv_fused_route && use_raw_nvfp4_gemv && nvfp4_weights_row_major_ && sm_ >= 90 &&
        !routing_snapshot_active && (is_fp16 || is_bf16)) {
      if (is_fp16) {
        fused_route_launched = LaunchFusedRouteMaps(
            static_cast<const half*>(router_probs->DataRaw()), expert_scales, expert_indices, p_r2u,
            unpermuted_row_to_permuted_row, p_exp, p_efto, static_cast<int>(num_rows), num_experts,
            static_cast<int>(k_), normalize_routing_weights_, fp4_gemv_pdl, stream);
      } else {
        fused_route_launched = LaunchFusedRouteMaps(
            static_cast<const __nv_bfloat16*>(router_probs->DataRaw()), expert_scales, expert_indices, p_r2u,
            unpermuted_row_to_permuted_row, p_exp, p_efto, static_cast<int>(num_rows), num_experts,
            static_cast<int>(k_), normalize_routing_weights_, fp4_gemv_pdl, stream);
      }
    }
    IAllocatorUniquePtr<int> blocked_expert_counts;
    IAllocatorUniquePtr<int> blocked_expert_counts_cumsum;
    IAllocatorUniquePtr<int> blocked_row_to_unpermuted_row;
    if (!fused_route_launched) {
      const auto fused_routing = route_tile(0, num_rows);
      ORT_ENFORCE(fused_routing.router_logits == nullptr,
                  "QMoE FP4 GEMV does not support fused routing.");
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
      if (routing_snapshot_) {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(
            expert_indices, SafeInt<size_t>(moe_params.num_rows) * SafeInt<size_t>(k_), stream));
      }
#endif
      if (!ck::fusedBuildExpertMapsSortFirstToken(
              expert_indices, p_r2u, unpermuted_row_to_permuted_row, p_exp, p_efto,
              num_rows, num_experts, static_cast<int>(k_), 0, num_experts, stream)) {
        const int64_t num_tokens_per_block = ck::computeNumTokensPerBlock(num_rows, num_experts);
        const int64_t num_blocks_per_seq = onnxruntime::llm::common::ceilDiv(num_rows, num_tokens_per_block);
        const size_t num_counts = SafeInt<size_t>(num_experts) * num_blocks_per_seq;
        blocked_expert_counts = GetScratchBuffer<int>(num_counts, GetComputeStream(context));
        blocked_expert_counts_cumsum = GetScratchBuffer<int>(num_counts, GetComputeStream(context));
        blocked_row_to_unpermuted_row = GetScratchBuffer<int>(
            SafeInt<size_t>(num_experts) * num_rows, GetComputeStream(context));
        ck::threeStepBuildExpertMapsSortFirstToken(
            expert_indices, p_exp, p_r2u, unpermuted_row_to_permuted_row, p_efto,
            blocked_expert_counts.get(), blocked_expert_counts_cumsum.get(),
            blocked_row_to_unpermuted_row.get(), num_rows, num_experts, k_, 0, stream);
      }
    }

    const void* fc1_bias = offload_gpu_fc1_bias_
                               ? offload_gpu_fc1_bias_.get()
                               : (fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr);
    const void* fc2_bias = offload_gpu_fc2_bias_
                               ? offload_gpu_fc2_bias_.get()
                               : (fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr);

    using MoeGemvConfig = gemv::MoeGemvConfig;

    // Choose the fc1 (SwiGLU) and fc2 GEMV tiling configs. Every config computes the same
    // dot products with the same accumulation dtype, so the only goal is picking the
    // fastest; Threads does set the K partition, so the low bits of the output can move
    // between configs (see MoeGemvConfig). Reuse a cached per-shape result when available;
    // otherwise start from the shape-derived analytic default and, when autotune is on,
    // profile on a non-captured (warmup) call and freeze the choice for CUDA-graph replay.
    // During capture (or when autotune is off, which is the shipping default) the analytic
    // default is what actually runs.
    const int multi_processor_count = GetDeviceProp().multiProcessorCount;
    MoeGemvConfig fc1_config =
        gemv::Fp4MoeGemvDefaultConfig(expanded, fc1_n, hidden, multi_processor_count);
    MoeGemvConfig fc2_config =
        gemv::Fp4MoeGemvDefaultConfig(expanded, hidden, inter, multi_processor_count);

    const int64_t row_bucket =
        onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler::bucketM(expanded);
    const Fp4GemvTuneKey tune_key{is_fp16_, row_bucket, hidden, inter, sm_};
    bool have_tune = false;
    {
      std::lock_guard<std::mutex> lock(fp4_gemv_tune_cache_mutex_);
      const auto cached_tune = fp4_gemv_tune_cache_.find(tune_key);
      have_tune = cached_tune != fp4_gemv_tune_cache_.end();
      if (have_tune) {
        fc1_config = cached_tune->second.fc1_config;
        fc2_config = cached_tune->second.fc2_config;
      }
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CALL_THROW(cudaStreamIsCapturing(stream, &capture_status));
    const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
    const bool do_tune = enable_fp4_gemv_autotune_ && !have_tune && !is_capturing;

    bool fc1_used_decode = false;
    bool fc2_used_decode = false;
    // FC1 -> FC2 -> finalize overlap through programmatic dependent launch (row-major NVFP4 decode kernels only):
    // FC2 prefetches its weights and finalize its routing metadata while the previous kernel finishes.
    const bool fc2_finalize_pdl =
        fp4_gemv_pdl && use_raw_nvfp4_gemv && nvfp4_weights_row_major_ && sm_ >= 90;
    auto run_fused = [&](auto* t_ptr) {
      using T = std::remove_pointer_t<decltype(t_ptr)>;
      const bool skip_expand = fp4_gemv_skip_expand_;
      if (!skip_expand) {
        ck::expandInputRowsKernelLauncher<T, T>(
            static_cast<const T*>(input->DataRaw()), static_cast<T*>(p_act_buf.get()),
            nullptr, nullptr, p_r2u, num_rows, hidden, static_cast<int>(k_), num_experts,
            quant_params, false, p_efto, nullptr, nullptr, nullptr, stream);
      }

      auto launch_fc1 = [&](MoeGemvConfig cfg, bool pdl = false) {
        gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>(
            skip_expand ? static_cast<const T*>(input->DataRaw()) : static_cast<const T*>(p_act_buf.get()),
            gemv_fc1_weight,
            static_cast<const T*>(gemv_fp4_fc1_scales_.get()), gemv_fc1_raw_block_scales,
            gemv_fc1_raw_global_scales,
            static_cast<const T*>(fc1_bias), static_cast<T*>(p_fc1_buf.get()),
            p_efto, p_exp, num_experts, expanded, inter, hidden, gemv_group_size, sm_, act_params, cfg,
            fc1_gemv_sm80_layout, use_raw_nvfp4_gemv, skip_expand ? p_r2u : nullptr, num_rows, stream,
            nvfp4_weights_row_major_, nvfp4_fc1_decode_scale_eligible_, &fc1_used_decode, pdl);
      };
      auto launch_fc2 = [&](MoeGemvConfig cfg, bool pdl = false) {
        gemv::launch_moe_gemv_fp4_symmetric<T>(
            static_cast<const T*>(p_fc1_buf.get()),
            gemv_fc2_weight,
            static_cast<const T*>(gemv_fp4_fc2_scales_.get()), gemv_fc2_raw_block_scales,
            gemv_fc2_raw_global_scales,
            static_cast<const T*>(fc2_bias), static_cast<T*>(p_fc2_buf.get()),
            p_efto, p_exp, num_experts, expanded, hidden, inter, gemv_group_size, sm_, cfg,
            fc2_gemv_sm80_layout, use_raw_nvfp4_gemv, stream, nvfp4_weights_row_major_,
            nvfp4_fc2_decode_scale_eligible_, &fc2_used_decode, pdl);
      };

      if (do_tune) {
        constexpr MoeGemvConfig kCandidates[] = {
            MoeGemvConfig::kDefault, MoeGemvConfig::kCtaN16, MoeGemvConfig::kThreads64};
        constexpr int kWarmup = 3;
        constexpr int kIters = 20;

        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        CUDA_CALL_THROW(cudaEventCreate(&start_event));
        std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)> start_event_guard(start_event, cudaEventDestroy);
        CUDA_CALL_THROW(cudaEventCreate(&stop_event));
        std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)> stop_event_guard(stop_event, cudaEventDestroy);

        auto time_launch = [&](auto&& launch_fn) -> float {
          for (int i = 0; i < kWarmup; ++i) {
            launch_fn();
          }
          CUDA_CALL_THROW(cudaStreamSynchronize(stream));
          CUDA_CALL_THROW(cudaEventRecord(start_event, stream));
          for (int i = 0; i < kIters; ++i) {
            launch_fn();
          }
          CUDA_CALL_THROW(cudaEventRecord(stop_event, stream));
          CUDA_CALL_THROW(cudaEventSynchronize(stop_event));
          float elapsed_ms = 0.0f;
          CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
          return elapsed_ms;
        };

        // fc1 reads the original input through p_r2u when skip-expand is enabled; otherwise
        // it reads p_act_buf populated by the expand above.
        const bool log_tune = enable_fp4_gemv_autotune_log_;
        float best_fc1 = std::numeric_limits<float>::max();
        for (MoeGemvConfig cfg : kCandidates) {
          if (!gemv::is_moe_gemv_fp4_supported(sm_, expanded, fc1_n, hidden, gemv_group_size, cfg,
                                               fc1_gemv_sm80_layout, is_nvfp4, nvfp4_weights_row_major_)) {
            continue;
          }
          const float ms = time_launch([&] { launch_fc1(cfg); });
          if (log_tune) {
            LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune candidate: fc1 expanded=" << expanded
                                  << " n=" << fc1_n << " k=" << hidden << " cfg=" << QMoEGemvConfigName(cfg)
                                  << " " << ms << "ms/" << kIters << "it";
          }
          if (ms < best_fc1) {
            best_fc1 = ms;
            fc1_config = cfg;
          }
        }
        // fc2 reads p_fc1_buf; populate it once with the chosen fc1 config before timing fc2.
        launch_fc1(fc1_config);
        float best_fc2 = std::numeric_limits<float>::max();
        for (MoeGemvConfig cfg : kCandidates) {
          if (!gemv::is_moe_gemv_fp4_supported(sm_, expanded, hidden, inter, gemv_group_size, cfg,
                                               fc2_gemv_sm80_layout, is_nvfp4, nvfp4_weights_row_major_)) {
            continue;
          }
          const float ms = time_launch([&] { launch_fc2(cfg); });
          if (log_tune) {
            LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune candidate: fc2 expanded=" << expanded
                                  << " n=" << hidden << " k=" << inter << " cfg=" << QMoEGemvConfigName(cfg)
                                  << " " << ms << "ms/" << kIters << "it";
          }
          if (ms < best_fc2) {
            best_fc2 = ms;
            fc2_config = cfg;
          }
        }

        {
          std::lock_guard<std::mutex> lock(fp4_gemv_tune_cache_mutex_);
          fp4_gemv_tune_cache_.try_emplace(tune_key, Fp4GemvTuneResult{fc1_config, fc2_config});
        }
        if (log_tune) {
          LOGS_DEFAULT(WARNING) << "FP4 GEMV autotune: is_fp16=" << is_fp16_ << " expanded=" << expanded
                                << " hidden=" << hidden << " inter=" << inter << " fc1="
                                << QMoEGemvConfigName(fc1_config) << " (" << best_fc1 << "ms/" << kIters
                                << "it) fc2=" << QMoEGemvConfigName(fc2_config) << " (" << best_fc2 << "ms/"
                                << kIters << "it)";
        }
      }

      launch_fc1(fc1_config, !do_tune && fused_route_launched && skip_expand && fp4_gemv_fc1_pdl);
      launch_fc2(fc2_config, fc2_finalize_pdl && fc1_used_decode);
      ck::finalizeMoeRoutingKernelLauncher<T, T, T>(
          static_cast<const T*>(p_fc2_buf.get()), static_cast<T*>(output->MutableDataRaw()),
          nullptr, expert_scales, unpermuted_row_to_permuted_row, p_r2u, expert_indices,
          p_efto, num_rows, hidden, static_cast<int64_t>(k_), num_experts,
          parallelism_config, false, stream, fc2_finalize_pdl && fc2_used_decode);
    };

    if (is_fp16_) {
      run_fused(static_cast<half*>(nullptr));
    } else {
      run_fused(static_cast<__nv_bfloat16*>(nullptr));
    }
    if (enable_kernel_debug_info_) {
      PrintQMoEKernelDebugInfo(use_raw_nvfp4_gemv ? "fp4_gemv_raw" : "fp4_gemv_prepacked", moe_params.num_rows,
                               moe_params.num_rows, moe_params.num_rows,
                               expanded, expanded,
                               workspace_size, total_scratch_bytes);
      if (nvfp4_weights_row_major_) {
        std::cout << "QMoE NVFP4 DecodeImageAvailable="
                  << (gemv::is_moe_gemv_fp4_decode_image_available(sm_, is_fp16_) ? 1 : 0)
                  << " FC1ScaleEligible=" << (nvfp4_fc1_decode_scale_eligible_ ? 1 : 0)
                  << " FC2ScaleEligible=" << (nvfp4_fc2_decode_scale_eligible_ ? 1 : 0)
                  << " FC1Kernel=" << (fc1_used_decode ? "decode" : "kpacked")
                  << " FC2Kernel=" << (fc2_used_decode ? "decode" : "kpacked")
                  << std::endl;
      }
    }
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    if (routing_snapshot_) {
      ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
    }
#endif
    return Status::OK();
  }

  const void* fc1_weight_data = fc1_experts_weights ? fc1_experts_weights->DataRaw() : nullptr;
  const void* fc2_weight_data = fc2_experts_weights ? fc2_experts_weights->DataRaw() : nullptr;
  if (use_fp4_deep_gemm) {
    fc1_weight_data = fp4_deep_gemm_fc1_weights_.get();
    fc2_weight_data = fp4_deep_gemm_fc2_weights_.get();
  } else if (fp4_sm80_prefill) {
    // SM80 FP4 grouped GEMM: consume the e2m1 weights in the SM80 CUTLASS interleaved layout
    // that PrePack produced into the GEMV interleaved-layout buffers (same layout the INT4 SM80
    // grouped GEMM uses). The activation-dtype group scales are wired via quant_params above.
    fc1_weight_data = gemv_fp4_fc1_weights_.get();
    fc2_weight_data = gemv_fp4_fc2_weights_.get();
  } else if ((is_fp4_family && route_native_fp4) || (is_wfp4afp8 && !use_wfp4afp8_dequant_fallback_)) {
    // The native CUTLASS FP4 paths consume weights in the repacked FP4
    // layout produced by PrePack. If PrePack never ran (e.g.
    // ``session.disable_prepacking`` is set) the repacked buffers stay null and
    // falling through to the raw initializer bytes would feed a non-CUTLASS
    // layout to the runner, producing silently wrong output. Fail loudly.
    if (!nvfp4_weights_row_major_ && (packed_fp4_fc1_weights_ == nullptr || packed_fp4_fc2_weights_ == nullptr)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "QMoE native FP4 requires PrePack to run, but the repacked FP4 weight "
                             "buffers were not produced (is session.disable_prepacking set?). "
                             "Enable prepacking to use the native FP4 path.");
    }
    if (!nvfp4_weights_row_major_) {
      fc1_weight_data = packed_fp4_fc1_weights_.get();
      fc2_weight_data = packed_fp4_fc2_weights_.get();
    }
  } else if (int_weights_consumed_by_prepack) {
    // PrePack converted the raw int4/int8 weights to the CUTLASS fpA_intB
    // layout that the runner consumes and freed the source initializer
    // (``is_packed = true``). Gate on ``int_weights_consumed_by_prepack``
    // (which already requires both packed weight buffers) rather than
    // just ``is_int && !weights_prepacked_``: when prepacking is disabled at
    // the session level (``session.disable_prepacking``) PrePack never runs,
    // the prepack buffers stay null, and the raw initializer pointers read
    // above must be kept so the runner is not handed null weight pointers.
    fc1_weight_data = packed_fc1_weights_.get();
    fc2_weight_data = packed_fc2_weights_.get();
  }
  IAllocatorUniquePtr<void> dequant_fc1_weights;
  IAllocatorUniquePtr<void> dequant_fc2_weights;
  IAllocatorUniquePtr<void> compact_fc1_bias;
  IAllocatorUniquePtr<void> compact_fc2_bias;
  IAllocatorUniquePtr<int> compact_expert_indices;
  IAllocatorUniquePtr<int> expert_to_compact;
  IAllocatorUniquePtr<int> compact_to_expert;
  std::array<IAllocatorUniquePtr<float>, 3> fp8_block_scales;
  std::array<const float*, 3> fp8_scales{};
  size_t dequant_weight_bytes = 0;
  std::function<void()> dequantize_fp4;
  std::function<void(int)> dequantize_fp4_dma;
  bool use_prefill_dma = false;
  int prefill_dma_dev_id = 0;
  int64_t prefill_dma_layer = -1;
  QMoEPrefillPassDmaView prefill_dma_view{};
  // Held across every tile and pass so concurrent sessions cannot interleave on the per-device DMA state.
  std::optional<QMoEPrefillDmaLease> prefill_dma_lease;
  if (use_int_dequant_fallback) {
    ORT_ENFORCE(fc1_experts_weights != nullptr && fc2_experts_weights != nullptr &&
                    fc1_scales != nullptr && fc2_scales != nullptr,
                "INT2 or mixed-width CUDA QMoE requires raw weights and scales for FC1 and FC2.");

    const int fc1_n = static_cast<int>(is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size);
    const int fc1_k = static_cast<int>(moe_params.hidden_size);
    const int fc2_n = static_cast<int>(moe_params.hidden_size);
    const int fc2_k = static_cast<int>(moe_params.inter_size);
    const int num_experts = static_cast<int>(moe_params.num_experts);
    dequant_fc1_weights = GetScratchBuffer<void>(int_dequant_fc1_bytes, GetComputeStream(context));
    dequant_fc2_weights = GetScratchBuffer<void>(int_dequant_fc2_bytes, GetComputeStream(context));

    const auto* fc1_zero_data = fc1_zeros ? static_cast<const uint8_t*>(fc1_zeros->DataRaw()) : nullptr;
    const auto* fc2_zero_data = fc2_zeros ? static_cast<const uint8_t*>(fc2_zeros->DataRaw()) : nullptr;
    if (is_fp16_) {
      ORT_RETURN_IF_NOT(fc1_scales->IsDataType<MLFloat16>() && fc2_scales->IsDataType<MLFloat16>(),
                        "INT2 or mixed-width CUDA QMoE with FP16 activations requires FP16 scales.");
      ORT_RETURN_IF_ERROR((DequantizeNBits<half, uint8_t>(
          static_cast<int>(fc1_expert_weight_bits_), static_cast<half*>(dequant_fc1_weights.get()),
          static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
          reinterpret_cast<const half*>(fc1_scales->DataRaw()), fc1_zero_data, nullptr,
          fc1_k, num_experts * fc1_n, static_cast<int>(block_size_), stream)));
      ORT_RETURN_IF_ERROR((DequantizeNBits<half, uint8_t>(
          static_cast<int>(fc2_expert_weight_bits_), static_cast<half*>(dequant_fc2_weights.get()),
          static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
          reinterpret_cast<const half*>(fc2_scales->DataRaw()), fc2_zero_data, nullptr,
          fc2_k, num_experts * fc2_n, static_cast<int>(block_size_), stream)));
    } else {
      ORT_RETURN_IF_NOT(fc1_scales->IsDataType<BFloat16>() && fc2_scales->IsDataType<BFloat16>(),
                        "INT2 or mixed-width CUDA QMoE with BF16 activations requires BF16 scales.");
      ORT_RETURN_IF_ERROR((DequantizeNBits<__nv_bfloat16, uint8_t>(
          static_cast<int>(fc1_expert_weight_bits_), static_cast<__nv_bfloat16*>(dequant_fc1_weights.get()),
          static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
          reinterpret_cast<const __nv_bfloat16*>(fc1_scales->DataRaw()), fc1_zero_data, nullptr,
          fc1_k, num_experts * fc1_n, static_cast<int>(block_size_), stream)));
      ORT_RETURN_IF_ERROR((DequantizeNBits<__nv_bfloat16, uint8_t>(
          static_cast<int>(fc2_expert_weight_bits_), static_cast<__nv_bfloat16*>(dequant_fc2_weights.get()),
          static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
          reinterpret_cast<const __nv_bfloat16*>(fc2_scales->DataRaw()), fc2_zero_data, nullptr,
          fc2_k, num_experts * fc2_n, static_cast<int>(block_size_), stream)));
    }
    fc1_weight_data = dequant_fc1_weights.get();
    fc2_weight_data = dequant_fc2_weights.get();
  }
  // FP4 (W4A16) and WFP4AFP8 (W4A8) share the MXFP4 weight format (Float8E8M0 block scales, block 32).
  // NVFP4 (W4A16) uses Float8E4M3FN block scales with block 16.
  // When the native CUTLASS path is unavailable on the current SM, or when native
  // FP4 routes this call to the dense fallback for the large per-expert-M regime, dequantize the E2M1
  // weights to FP16/BF16 and run the dense A16 runner.
  if (!use_fp4_deep_gemm &&
      ((is_fp4_family && !route_native_fp4) || (is_wfp4afp8 && use_wfp4afp8_dequant_fallback_)) &&
      !fp4_sm80_prefill) {
    // The dequant kernel expects raw [E, n, k_blocks] block scales. When native FP4 is enabled
    // (this is the large per-expert-M fallback), packed_fp4_*_block_scales_ holds the TMA-swizzled
    // layout, so use the raw copy kept in gemv_fp4_*_block_raw_ instead. NVFP4 retains its raw
    // scale inputs; prefer those even when native prepacking also created swizzled scales.
    const void* p_fc1_block_scales =
        (is_nvfp4 && fp4_fc1_block_scales && fp4_fc1_block_scales->Location().device.Type() == OrtDevice::GPU)
            ? fp4_fc1_block_scales->DataRaw()
        : gemv_fp4_fc1_block_raw_ ? gemv_fp4_fc1_block_raw_.get()
                                  : (packed_fp4_fc1_block_scales_ ? packed_fp4_fc1_block_scales_.get()
                                                                  : (fp4_fc1_block_scales ? fp4_fc1_block_scales->DataRaw() : nullptr));
    const void* p_fc1_global_scale = packed_fc1_global_scale_ ? packed_fc1_global_scale_.get()
                                                              : (fc1_global_scale ? fc1_global_scale->DataRaw() : nullptr);
    const void* p_fc2_block_scales =
        (is_nvfp4 && fp4_fc2_block_scales && fp4_fc2_block_scales->Location().device.Type() == OrtDevice::GPU)
            ? fp4_fc2_block_scales->DataRaw()
        : gemv_fp4_fc2_block_raw_ ? gemv_fp4_fc2_block_raw_.get()
                                  : (packed_fp4_fc2_block_scales_ ? packed_fp4_fc2_block_scales_.get()
                                                                  : (fp4_fc2_block_scales ? fp4_fc2_block_scales->DataRaw() : nullptr));
    const void* p_fc2_global_scale = packed_fc2_global_scale_ ? packed_fc2_global_scale_.get()
                                                              : (fc2_global_scale ? fc2_global_scale->DataRaw() : nullptr);
    ORT_RETURN_IF_NOT(p_fc1_block_scales && p_fc1_global_scale && p_fc2_block_scales && p_fc2_global_scale,
                      "QMoE FP4 dequant fallback requires block and global scales for fc1 and fc2.");

    int fc1_n = static_cast<int>(is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size);
    int fc1_k = static_cast<int>(moe_params.hidden_size);
    int fc2_n = static_cast<int>(moe_params.hidden_size);
    int fc2_k = static_cast<int>(moe_params.inter_size);
    int num_experts = runner_num_experts;
    size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fc1_bytes = SafeInt<size_t>(num_experts) * fc1_n * fc1_k * element_size;
    size_t fc2_bytes = SafeInt<size_t>(num_experts) * fc2_n * fc2_k * element_size;
    dequant_weight_bytes = SafeInt<size_t>(fc1_bytes) + fc2_bytes;

    ORT_RETURN_IF_NOT(fc1_experts_weights != nullptr && fc2_experts_weights != nullptr,
                      "QMoE FP4/NVFP4 dequant fallback requires valid raw expert-weight tensors.");
    QMoEResidentLayerSlabView resident_view{};
    if (is_nvfp4 && nvfp4_weights_row_major_ && offload_layer_index_ >= 0) {
      const int64_t parsed_layer = offload_layer_index_;
      int dev_id = 0;
      static_cast<void>(cudaGetDevice(&dev_id));
      prefill_dma_dev_id = dev_id;
      prefill_dma_layer = parsed_layer;
      if (use_nvfp4_compaction && num_compact_passes > 1) {
        const float* host_fc1_g = (fc1_global_scale && fc1_global_scale->Location().device.Type() == OrtDevice::CPU)
                                      ? fc1_global_scale->Data<float>()
                                      : static_cast<const float*>(p_fc1_global_scale);
        const float* host_fc2_g = (fc2_global_scale && fc2_global_scale->Location().device.Type() == OrtDevice::CPU)
                                      ? fc2_global_scale->Data<float>()
                                      : static_cast<const float*>(p_fc2_global_scale);
        prefill_dma_lease.emplace(dev_id, stream);
        use_prefill_dma = TryPrepareQMoEPrefillDmaLayer(
            dev_id, parsed_layer, narrow<int>(moe_params.num_experts),
            narrow<int>(moe_params.hidden_size), narrow<int>(moe_params.inter_size),
            runner_num_experts, moe_params.num_rows, element_size,
            static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
            static_cast<const uint8_t*>(p_fc1_block_scales),
            host_fc1_g,
            static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
            static_cast<const uint8_t*>(p_fc2_block_scales),
            host_fc2_g,
            stream, &prefill_dma_view);
        if (!use_prefill_dma) prefill_dma_lease.reset();
      }
      if (!use_prefill_dma) {
        static_cast<void>(TryGetQMoEResidentLayerSlabView(
            dev_id, parsed_layer, narrow<int>(moe_params.num_experts),
            narrow<int>(moe_params.hidden_size), narrow<int>(moe_params.inter_size), &resident_view, stream,
            static_cast<const uint8_t*>(fc1_experts_weights->DataRaw())));
      }
    }
    const void* fc1_bias_gpu_src =
        offload_gpu_fc1_bias_
            ? offload_gpu_fc1_bias_.get()
            : (fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr);
    const void* fc2_bias_gpu_src =
        offload_gpu_fc2_bias_
            ? offload_gpu_fc2_bias_.get()
            : (fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr);
    if (use_prefill_dma) {
      if (!prefill_dma_view.dequant_fc1_weights || !prefill_dma_view.dequant_fc2_weights) {
        dequant_fc1_weights = dequant_transient_env
                                  ? GetTransientScratchBuffer<void>(fc1_bytes)
                                  : GetScratchBuffer<void>(fc1_bytes, GetComputeStream(context));
        dequant_fc2_weights = dequant_transient_env
                                  ? GetTransientScratchBuffer<void>(fc2_bytes)
                                  : GetScratchBuffer<void>(fc2_bytes, GetComputeStream(context));
        prefill_dma_view.dequant_fc1_weights = dequant_fc1_weights.get();
        prefill_dma_view.dequant_fc2_weights = dequant_fc2_weights.get();
      }
      fc1_weight_data = prefill_dma_view.dequant_fc1_weights;
      fc2_weight_data = prefill_dma_view.dequant_fc2_weights;
      dequantize_fp4_dma = [&, fc1_n, fc1_k, fc2_n, fc2_k, wave_experts = num_experts, fc1_bias_gpu_src, fc2_bias_gpu_src](int pass_start_expert) {
        // The last wave covers only the remaining experts; launching a full wave would read past the per-expert tables.
        const int num_experts = std::min(wave_experts, narrow<int>(moe_params.num_experts) - pass_start_expert);
        if (num_experts <= 0) return;
        auto dequant_pass = [&](const uint8_t* stg_w, const uint8_t* stg_s, const float* stg_g,
                                void* out, int n, int k, const void* bias_ptr, void* output_bias,
                                const uint8_t* res_w, const uint8_t* res_s,
                                const uint8_t* host_w, const uint8_t* host_s) {
          if (is_fp16_) {
            half* out_h = static_cast<half*>(out);
            LaunchQMoEDequantizeNvfp4Weights(
                stg_w, stg_s, stg_g, out_h, num_experts, n, k, stream,
                nullptr, static_cast<const half*>(bias_ptr),
                static_cast<half*>(output_bias), nvfp4_weights_row_major_,
                res_w, res_s, prefill_dma_view.slot_table_dev,
                pass_start_expert, expert_to_compact.get(),
                host_w, host_s, prefill_dma_view.host_expert_stride, prefill_dma_view.host_slot_table_dev);
          } else {
            __nv_bfloat16* out_b = static_cast<__nv_bfloat16*>(out);
            LaunchQMoEDequantizeNvfp4Weights(
                stg_w, stg_s, stg_g, out_b, num_experts, n, k, stream,
                nullptr, static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<__nv_bfloat16*>(output_bias), nvfp4_weights_row_major_,
                res_w, res_s, prefill_dma_view.slot_table_dev,
                pass_start_expert, expert_to_compact.get(),
                host_w, host_s, prefill_dma_view.host_expert_stride, prefill_dma_view.host_slot_table_dev);
          }
        };
        dequant_pass(prefill_dma_view.staging_fc1_weights,
                     prefill_dma_view.staging_fc1_scales,
                     prefill_dma_view.staging_fc1_global_scales,
                     prefill_dma_view.dequant_fc1_weights, fc1_n, fc1_k,
                     fc1_bias_gpu_src, compact_fc1_bias.get(),
                     prefill_dma_view.resident_fc1_weights, prefill_dma_view.resident_fc1_scales,
                     prefill_dma_view.host_fc1_weights, prefill_dma_view.host_fc1_scales);
        dequant_pass(prefill_dma_view.staging_fc2_weights,
                     prefill_dma_view.staging_fc2_scales,
                     prefill_dma_view.staging_fc2_global_scales,
                     prefill_dma_view.dequant_fc2_weights, fc2_n, fc2_k,
                     fc2_bias_gpu_src, compact_fc2_bias.get(),
                     prefill_dma_view.resident_fc2_weights, prefill_dma_view.resident_fc2_scales,
                     prefill_dma_view.host_fc2_weights, prefill_dma_view.host_fc2_scales);
      };
    } else {
      ORT_RETURN_IF(nvfp4_weights_row_major_ &&
                        (offload_plan_.use_file_backing ||
                         ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_MIB", 0) > 0) &&
                        !resident_view.all_experts_cached,
                    "Bounded prefill with incomplete GPU/host cache coverage requires prefill staging DMA. "
                    "Enable ORT_QMOE_ENABLE_PREFILL_STAGING_DMA and provide sufficient workspace; "
                    "pageable expert banks cannot be read directly by the GPU.");
      dequant_fc1_weights = dequant_transient_env
                                ? GetTransientScratchBuffer<void>(fc1_bytes)
                                : GetScratchBuffer<void>(fc1_bytes, GetComputeStream(context));
      dequant_fc2_weights = dequant_transient_env
                                ? GetTransientScratchBuffer<void>(fc2_bytes)
                                : GetScratchBuffer<void>(fc2_bytes, GetComputeStream(context));
      dequantize_fp4 = [&, fc1_n, fc1_k, fc2_n, fc2_k, num_experts,
                        p_fc1_block_scales, p_fc1_global_scale, p_fc2_block_scales, p_fc2_global_scale,
                        resident_view, fc1_bias_gpu_src, fc2_bias_gpu_src]() {
        // Choose the FP4 (MXFP4 / E8M0, block 32) or NVFP4 (E4M3, block 16) dequant launcher.
        auto dequant = [&](const uint8_t* weights, const uint8_t* block_scales, const float* global_scale,
                           void* out, int n, int k, const void* bias_ptr, void* output_bias,
                           const uint8_t* res_w, const uint8_t* res_s,
                           const uint8_t* host_w, const uint8_t* host_s) {
          if (is_fp16_) {
            half* out_h = static_cast<half*>(out);
            if (is_nvfp4) {
              LaunchQMoEDequantizeNvfp4Weights(weights, block_scales, global_scale, out_h, num_experts, n, k, stream,
                                               compact_to_expert.get(), static_cast<const half*>(bias_ptr),
                                               static_cast<half*>(output_bias), nvfp4_weights_row_major_,
                                               res_w, res_s, resident_view.slot_table_dev,
                                               -1, nullptr,
                                               host_w, host_s, resident_view.host_expert_stride, resident_view.host_slot_table_dev);
            } else {
              LaunchQMoEDequantizeFp4Weights(weights, block_scales, global_scale, out_h, num_experts, n, k, stream);
            }
          } else {
            __nv_bfloat16* out_b = static_cast<__nv_bfloat16*>(out);
            if (is_nvfp4) {
              LaunchQMoEDequantizeNvfp4Weights(weights, block_scales, global_scale, out_b, num_experts, n, k, stream,
                                               compact_to_expert.get(), static_cast<const __nv_bfloat16*>(bias_ptr),
                                               static_cast<__nv_bfloat16*>(output_bias), nvfp4_weights_row_major_,
                                               res_w, res_s, resident_view.slot_table_dev,
                                               -1, nullptr,
                                               host_w, host_s, resident_view.host_expert_stride, resident_view.host_slot_table_dev);
            } else {
              LaunchQMoEDequantizeFp4Weights(weights, block_scales, global_scale, out_b, num_experts, n, k, stream);
            }
          }
        };
        const float* fc1_g_ptr = static_cast<const float*>(p_fc1_global_scale);
        const float* fc2_g_ptr = static_cast<const float*>(p_fc2_global_scale);
        const uint8_t* fc1_w_ptr = packed_fp4_fc1_weights_
                                       ? static_cast<const uint8_t*>(packed_fp4_fc1_weights_.get())
                                       : static_cast<const uint8_t*>(fc1_experts_weights->DataRaw());
        const uint8_t* fc2_w_ptr = packed_fp4_fc2_weights_
                                       ? static_cast<const uint8_t*>(packed_fp4_fc2_weights_.get())
                                       : static_cast<const uint8_t*>(fc2_experts_weights->DataRaw());
        dequant(fc1_w_ptr,
                static_cast<const uint8_t*>(p_fc1_block_scales),
                fc1_g_ptr,
                dequant_fc1_weights.get(), fc1_n, fc1_k, fc1_bias_gpu_src, compact_fc1_bias.get(),
                resident_view.fc1_weights, resident_view.fc1_scales,
                resident_view.host_fc1_weights, resident_view.host_fc1_scales);
        dequant(fc2_w_ptr,
                static_cast<const uint8_t*>(p_fc2_block_scales),
                fc2_g_ptr,
                dequant_fc2_weights.get(), fc2_n, fc2_k, fc2_bias_gpu_src, compact_fc2_bias.get(),
                resident_view.fc2_weights, resident_view.fc2_scales,
                resident_view.host_fc2_weights, resident_view.host_fc2_scales);
      };
      fc1_weight_data = dequant_fc1_weights.get();
      fc2_weight_data = dequant_fc2_weights.get();
    }
  } else if (is_fp8 && use_fp8_dequant_fallback_) {
    auto get_block_scales = [&](int input_idx, size_t scratch_idx) -> const float* {
      const auto* tensor = context->Input<Tensor>(input_idx);
      if (tensor->IsDataType<float>()) {
        return tensor->Data<float>();
      }
      const size_t count = narrow<size_t>(tensor->Shape().Size());
      auto& converted = fp8_block_scales[scratch_idx];
      converted = GetScratchBuffer<float>(count, GetComputeStream(context));
      if (tensor->IsDataType<MLFloat16>()) {
        Impl_Cast<half, float>(stream, static_cast<const half*>(tensor->DataRaw()), converted.get(), count);
      } else {
        ORT_ENFORCE(tensor->IsDataType<BFloat16>(), "Unsupported FP8 block scale type.");
        Impl_Cast<BFloat16, float>(stream, tensor->Data<BFloat16>(), converted.get(), count);
      }
      CUDA_CALL_THROW(cudaGetLastError());
      return converted.get();
    };
    fp8_scales[0] = is_block_fp8 ? get_block_scales(3, 0)
                                 : (packed_fc1_global_scale_ ? static_cast<const float*>(packed_fc1_global_scale_.get())
                                                             : (fc1_global_scale ? fc1_global_scale->Data<float>() : nullptr));
    fp8_scales[1] = is_block_fp8 ? get_block_scales(6, 1)
                                 : (packed_fc2_global_scale_ ? static_cast<const float*>(packed_fc2_global_scale_.get())
                                                             : (fc2_global_scale ? fc2_global_scale->Data<float>() : nullptr));
    ORT_RETURN_IF_NOT(fp8_scales[0] && fp8_scales[1],
                      "QMoE FP8 dequant fallback requires fc1_global_scale and fc2_global_scale.");
    if (split_fp8_fc1) {
      fp8_scales[2] = get_block_scales(9, 2);
    }

    int fc1_n = static_cast<int>((is_fused_swiglu || split_fp8_fc1) ? moe_params.inter_size * 2 : moe_params.inter_size);
    int fc1_k = static_cast<int>(moe_params.hidden_size);
    int fc2_n = static_cast<int>(moe_params.hidden_size);
    int fc2_k = static_cast<int>(moe_params.inter_size);
    size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fc1_bytes = SafeInt<size_t>(runner_num_experts) * fc1_n * fc1_k * element_size;
    size_t fc2_bytes = SafeInt<size_t>(runner_num_experts) * fc2_n * fc2_k * element_size;
    dequant_weight_bytes = SafeInt<size_t>(fc1_bytes) + fc2_bytes;
    dequant_fc1_weights = GetScratchBuffer<void>(fc1_bytes, GetComputeStream(context));
    dequant_fc2_weights = GetScratchBuffer<void>(fc2_bytes, GetComputeStream(context));
    fc1_weight_data = dequant_fc1_weights.get();
    fc2_weight_data = dequant_fc2_weights.get();
  }

  if (use_expert_compaction) {
    const size_t num_routes = SafeInt<size_t>(row_tile_plan.rows_per_tile) * k_;
    compact_expert_indices = dequant_transient_env
                                 ? GetTransientScratchBuffer<int>(num_routes)
                                 : GetScratchBuffer<int>(num_routes, GetComputeStream(context));
    expert_to_compact = dequant_transient_env
                            ? GetTransientScratchBuffer<int>(moe_params.num_experts)
                            : GetScratchBuffer<int>(moe_params.num_experts, GetComputeStream(context));
    compact_to_expert = dequant_transient_env
                            ? GetTransientScratchBuffer<int>(runner_num_experts)
                            : GetScratchBuffer<int>(runner_num_experts, GetComputeStream(context));
    const size_t element_size = input->DataType()->Size();
    if (fc1_experts_bias_optional) {
      const size_t fc1_n = SafeInt<size_t>(moe_params.inter_size) * (is_fused_swiglu ? 2 : 1);
      compact_fc1_bias = dequant_transient_env
                             ? GetTransientScratchBuffer<void>(SafeInt<size_t>(runner_num_experts) * fc1_n * element_size)
                             : GetScratchBuffer<void>(
                                   SafeInt<size_t>(runner_num_experts) * fc1_n * element_size, GetComputeStream(context));
    }
    if (fc2_experts_bias_optional) {
      compact_fc2_bias = dequant_transient_env
                             ? GetTransientScratchBuffer<void>(SafeInt<size_t>(runner_num_experts) * moe_params.hidden_size * element_size)
                             : GetScratchBuffer<void>(
                                   SafeInt<size_t>(runner_num_experts) * moe_params.hidden_size * element_size, GetComputeStream(context));
    }
  }
  if (dequantize_fp4 && !use_nvfp4_compaction) {
    dequantize_fp4();
  }

  auto dequantize_fp8 = [&]() {
    const int fc1_n = narrow<int>(is_fused_swiglu ? moe_params.inter_size * 2 : moe_params.inter_size);
    const int fc1_k = narrow<int>(moe_params.hidden_size);
    const int fc2_n = narrow<int>(moe_params.hidden_size);
    const int fc2_k = narrow<int>(moe_params.inter_size);
    const int fp8_block_size = is_block_fp8 ? static_cast<int>(block_size_) : 0;
    auto dequantize = [&](auto* fc1_output, auto* fc2_output) {
      using T = std::remove_pointer_t<decltype(fc1_output)>;
      const auto* fc1_bias = compact_fc1_bias ? static_cast<const T*>(fc1_experts_bias_optional->DataRaw()) : nullptr;
      const auto* fc2_bias = compact_fc2_bias ? static_cast<const T*>(fc2_experts_bias_optional->DataRaw()) : nullptr;
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()),
                                     fp8_scales[0],
                                     fc1_output, runner_num_experts, fc1_n, fc1_k, stream, fp8_block_size, split_fp8_fc1,
                                     is_block_fp8 && swiglu_fusion_ == 2, compact_to_expert.get(),
                                     fc1_bias, static_cast<T*>(compact_fc1_bias.get()));
      LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()),
                                     fp8_scales[1],
                                     fc2_output, runner_num_experts, fc2_n, fc2_k, stream, fp8_block_size, false, false,
                                     compact_to_expert.get(), fc2_bias, static_cast<T*>(compact_fc2_bias.get()));
      if (split_fp8_fc1) {
        LaunchQMoEDequantizeFp8Weights(static_cast<const uint8_t*>(fc3_experts_weights->DataRaw()),
                                       fp8_scales[2],
                                       fc1_output + fc1_k,
                                       runner_num_experts, fc1_n, fc1_k, stream, fp8_block_size, true, false,
                                       compact_to_expert.get());
      }
    };
    if (is_fp16_) {
      dequantize(static_cast<half*>(dequant_fc1_weights.get()), static_cast<half*>(dequant_fc2_weights.get()));
    } else {
      dequantize(static_cast<__nv_bfloat16*>(dequant_fc1_weights.get()),
                 static_cast<__nv_bfloat16*>(dequant_fc2_weights.get()));
    }
  };
  if (is_fp8 && use_fp8_dequant_fallback_ && !is_block_fp8) {
    dequantize_fp8();
  }

  onnxruntime::llm::kernels::cutlass_kernels::ActivationParams activation_params(kernel_activation_type);
  activation_params.alpha = activation_alpha_;
  activation_params.beta = activation_beta_;
  activation_params.swiglu_fusion = swiglu_fusion;
  activation_params.limit = swiglu_limit_;
  if (split_fp8_fc1) {
    activation_params.alpha = 1.0f;
    activation_params.beta = 0.0f;
    activation_params.limit = std::numeric_limits<float>::infinity();
  }

  IAllocatorUniquePtr<void> multi_pass_output_buf;
  IAllocatorUniquePtr<float> multi_pass_accum_buf;
  if (num_compact_passes > 1) {
    const size_t max_tile_elems = SafeInt<size_t>(row_tile_plan.rows_per_tile) * moe_params.hidden_size;
    const size_t pass_out_bytes = SafeInt<size_t>(max_tile_elems) * input->DataType()->Size();
    multi_pass_output_buf = dequant_transient_env
                                ? GetTransientScratchBuffer<void>(pass_out_bytes)
                                : GetScratchBuffer<void>(pass_out_bytes, GetComputeStream(context));
    multi_pass_accum_buf = dequant_transient_env
                               ? GetTransientScratchBuffer<float>(max_tile_elems)
                               : GetScratchBuffer<float>(max_tile_elems, GetComputeStream(context));
  }

  // Route and execute each row tile in order on the same stream. The tile-local top-k metadata and
  // runner workspace are safe to reuse because the next tile's writes are ordered after all kernels
  // from the previous runMoe call. Tactic mutation remains serialized across concurrent QMoE calls.
  for (int64_t tile_index = 0; tile_index < row_tile_plan.TileCount(); ++tile_index) {
    const int64_t row_offset = row_tile_plan.RowOffset(tile_index);
    const int64_t tile_rows = row_tile_plan.RowsInTile(tile_index);
    const RunnerTileConfig* tile_config = nullptr;
    for (size_t i = 0; i < runner_tile_config_count; ++i) {
      if (runner_tile_configs[i].num_rows == tile_rows) {
        tile_config = &runner_tile_configs[i];
        break;
      }
    }
    ORT_ENFORCE(tile_config != nullptr,
                "QMoE did not prepare a runner configuration for row tile size ", tile_rows, ".");

    auto fused_routing = route_tile(row_offset, tile_rows);
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    if (routing_snapshot_) {
      if (fused_routing.router_logits != nullptr) {
        fused_routing.on_routing_ready = KernelPilotMoeExpertSelectionCuda::CaptureRouting;
        fused_routing.routing_context = &*routing_snapshot_;
      } else {
        ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(
            expert_indices, SafeInt<size_t>(tile_rows) * SafeInt<size_t>(k_), stream));
      }
    }
#endif
    const size_t input_element_offset =
        SafeInt<size_t>(row_offset) * SafeInt<size_t>(moe_params.hidden_size);
    const size_t input_byte_offset =
        SafeInt<size_t>(input_element_offset) * input->DataType()->Size();
    const size_t output_byte_offset =
        SafeInt<size_t>(input_element_offset) * output->DataType()->Size();
    const void* tile_input =
        reinterpret_cast<const char*>(input->DataRaw()) + input_byte_offset;
    void* tile_output =
        reinterpret_cast<char*>(output->MutableDataRaw()) + output_byte_offset;

    if (num_compact_passes > 1) {
      const int64_t tile_elems = SafeInt<int64_t>(tile_rows) * moe_params.hidden_size;
      const int64_t tile_routes = SafeInt<int64_t>(tile_rows) * k_;
      CUDA_CALL_THROW(cudaMemsetAsync(multi_pass_accum_buf.get(), 0,
                                      static_cast<size_t>(tile_elems) * sizeof(float), stream));
      if (use_prefill_dma) {
        LaunchQMoEMarkActiveExperts(expert_indices, expert_to_compact.get(),
                                    narrow<int>(moe_params.num_experts), tile_routes, stream);
      }
      if (prefill_dma_view.prefill_layer_hist_dev != nullptr) {
        LaunchQMoEAccumulateExpertHistogram(expert_indices, tile_routes,
                                            narrow<int>(moe_params.num_experts),
                                            prefill_dma_view.prefill_layer_hist_dev, stream);
      }
      for (int pass_idx = 0; pass_idx < num_compact_passes; ++pass_idx) {
        if (use_prefill_dma) {
          ORT_ENFORCE(BeginQMoEPrefillDmaPass(
              prefill_dma_dev_id, prefill_dma_layer, pass_idx, num_compact_passes,
              runner_num_experts, stream, &prefill_dma_view, expert_to_compact.get()));
          LaunchQMoERemapExpertRange(expert_indices, compact_expert_indices.get(),
                                     tile_routes, pass_idx * runner_num_experts,
                                     runner_num_experts, stream);
          dequantize_fp4_dma(pass_idx * runner_num_experts);
          EndQMoEPrefillDmaPassDequant(prefill_dma_dev_id, pass_idx, stream);
        } else {
          LaunchQMoECompactExperts(expert_indices, compact_expert_indices.get(),
                                   expert_to_compact.get(), compact_to_expert.get(),
                                   narrow<int>(moe_params.num_experts), runner_num_experts,
                                   tile_routes, stream,
                                   pass_idx * runner_num_experts);
          dequantize_fp4();
        }
        {
          std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);
          active_runner->setTactic(tile_config->config1, tile_config->config2);
          active_runner->runMoe(
              tile_input,
              nullptr,
              compact_expert_indices.get(),
              expert_scales,
              fc1_weight_data,
              compact_fc1_bias ? compact_fc1_bias.get()
                               : (offload_gpu_fc1_bias_
                                      ? offload_gpu_fc1_bias_.get()
                                      : (fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr)),
              kernel_activation_type,
              fc2_weight_data,
              compact_fc2_bias ? compact_fc2_bias.get()
                               : (offload_gpu_fc2_bias_
                                      ? offload_gpu_fc2_bias_.get()
                                      : (fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr)),
              quant_params,
              tile_rows,
              moe_params.hidden_size,
              moe_params.inter_size,
              runner_num_experts,
              k_,
              workspace_ptr,
              multi_pass_output_buf.get(),
              unpermuted_row_to_permuted_row,
              parallelism_config,
              activation_params,
              fused_routing,
              stream);
        }
        const bool is_last_pass = (pass_idx == num_compact_passes - 1);
        if (is_fp16_) {
          LaunchQMoEAccumulatePassOutput(multi_pass_accum_buf.get(),
                                         static_cast<const half*>(multi_pass_output_buf.get()),
                                         static_cast<half*>(tile_output),
                                         tile_elems, is_last_pass, stream);
        } else {
          LaunchQMoEAccumulatePassOutput(multi_pass_accum_buf.get(),
                                         static_cast<const __nv_bfloat16*>(multi_pass_output_buf.get()),
                                         static_cast<__nv_bfloat16*>(tile_output),
                                         tile_elems, is_last_pass, stream);
        }
        if (use_prefill_dma) {
          PrefetchNextQMoEPrefillDmaPass(
              prefill_dma_dev_id, prefill_dma_layer, pass_idx, num_compact_passes,
              runner_num_experts, stream);
        }
      }
    } else {
      if (use_expert_compaction) {
        LaunchQMoECompactExperts(expert_indices, compact_expert_indices.get(),
                                 expert_to_compact.get(), compact_to_expert.get(),
                                 narrow<int>(moe_params.num_experts), runner_num_experts,
                                 SafeInt<int64_t>(tile_rows) * k_, stream);
        if (use_nvfp4_compaction) {
          dequantize_fp4();
        } else {
          dequantize_fp8();
        }
      }

      std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);
      if (use_fp4_deep_gemm) {
        active_runner->setFp4DeepGemmWeightScales(
            static_cast<const float*>(fp4_deep_gemm_fc1_weight_scales_.get()),
            static_cast<const float*>(fp4_deep_gemm_fc2_weight_scales_.get()));
      }
      active_runner->setTactic(tile_config->config1, tile_config->config2);
      active_runner->runMoe(
          tile_input,
          nullptr,
          use_expert_compaction ? compact_expert_indices.get() : expert_indices,
          expert_scales,
          fc1_weight_data,
          compact_fc1_bias ? compact_fc1_bias.get()
                           : (offload_gpu_fc1_bias_
                                  ? offload_gpu_fc1_bias_.get()
                                  : (fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr)),
          kernel_activation_type,
          fc2_weight_data,
          compact_fc2_bias ? compact_fc2_bias.get()
                           : (offload_gpu_fc2_bias_
                                  ? offload_gpu_fc2_bias_.get()
                                  : (fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr)),
          quant_params,
          tile_rows,
          moe_params.hidden_size,
          moe_params.inter_size,
          runner_num_experts,
          k_,
          workspace_ptr,
          tile_output,
          unpermuted_row_to_permuted_row,
          parallelism_config,
          activation_params,
          fused_routing,
          stream);
    }

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    if (routing_snapshot_) {
      ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
    }
#endif
  }

  if (enable_kernel_debug_info_) {
    PrintQMoEKernelDebugInfo("grouped_moe", moe_params.num_rows,
                             row_tile_plan.rows_per_tile, final_tile_rows,
                             row_tile_plan.rows_per_tile, final_tile_rows,
                             workspace_size, total_scratch_bytes);
    if (use_expert_compaction) {
      std::cout << "QMoE " << (use_nvfp4_compaction ? "NVFP4" : "FP8") << " ExpertCapacity=" << runner_num_experts
                << " DequantWeightBytes=" << dequant_weight_bytes << std::endl;
    }
  }

  return Status::OK();
}

Status QMoE::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;
  if (offload_node_registered_ && !offload_prepack_bypass_) {
    return Status::OK();
  }
  if (nvfp4_weights_row_major_ && (input_idx == 2 || input_idx == 5 ||
                                   (use_fp4_dequant_fallback_ && (input_idx == 3 || input_idx == 6)))) {
    return Status::OK();
  }
  if (quant_type_ == "int" && (input_idx == 3 || input_idx == 6)) {
    ORT_RETURN_IF_NOT(is_fp16_ ? tensor.IsDataType<MLFloat16>() : tensor.IsDataType<BFloat16>(),
                      "QMoE integer fc", input_idx == 3 ? 1 : 2, "_scales must match the activation type.");
  }
  if (quant_type_ == "fp8" && (input_idx == 11 || input_idx == 12 || input_idx == 13)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QMoE FP8 does not support zero_points.");
  }

  const bool is_mixed_width = fc1_expert_weight_bits_ != expert_weight_bits_ ||
                              fc2_expert_weight_bits_ != expert_weight_bits_ ||
                              fc3_expert_weight_bits_ != expert_weight_bits_;
  if (is_mixed_width && quant_type_ != "int") {
    return Status::OK();
  }
  const bool uses_packed_int_decode = quant_type_ == "int" && (is_mixed_width || expert_weight_bits_ == 2);
  if (uses_packed_int_decode && input_idx != 2 && input_idx != 3 && input_idx != 5 && input_idx != 6) {
    return Status::OK();
  }
  if (quant_type_ == "int" && (is_mixed_width || expert_weight_bits_ == 2) &&
      !weights_prepacked_ && sm_ < 75 && (input_idx == 2 || input_idx == 5)) {
    return Status::OK();
  }

  const auto int_weight_supports_packed_execution = [&](int64_t weight_bits) {
    if ((!enable_int2_gemv_ && !enable_int2_prefill_) ||
        activation_type_ != onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu ||
        swiglu_fusion_ != 1 || (block_size_ != 32 && block_size_ != 64 && block_size_ != 128)) {
      return false;
    }

    const auto& shape = tensor.Shape();
    const int64_t pack_factor = 8 / weight_bits;
    if (shape.NumDimensions() != 3 || shape[2] > std::numeric_limits<int64_t>::max() / pack_factor) {
      return false;
    }

    const int64_t reduction_size = shape[2] * pack_factor;
    if (enable_int2_prefill_ && shape[0] > 0 && shape[0] <= 256 &&
        shape[1] > 0 && shape[1] <= std::numeric_limits<int>::max() && shape[1] % 64 == 0 &&
        reduction_size > 0 && reduction_size <= std::numeric_limits<int>::max() && reduction_size % 64 == 0) {
      return true;
    }

    // Block size 32 is served only by packed prefill; ComputeInternal's packed GEMV accepts 64 and 128.
    return enable_int2_gemv_ && (block_size_ == 64 || block_size_ == 128) &&
           onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_supported(
               sm_, /*expanded_num_rows=*/1, shape[1], shape[2] * pack_factor,
               static_cast<int>(weight_bits), static_cast<int>(block_size_));
  };

  cudaStream_t stream = 0;  // Use default stream for PrePack operations

  DUMP_TENSOR_INIT();

#if DUMP_TENSOR_LEVEL >= 1
  auto dump_tensor = [&](const char* name, const IAllocatorUniquePtr<void>& packed_scales, const Tensor& scales) {
    auto shape = scales.Shape();
    if (shape.NumDimensions() == 3 && is_fp16_) {
      size_t rows = shape[1];
      size_t cols = shape[2];
      size_t batch = shape[0];
      if (expert_weight_bits_ == 8 && block_size_ <= 0 && strstr(name, "bias") != nullptr) {
        DUMP_TENSOR(name, static_cast<const uint8_t*>(packed_scales.get()), int(batch), int(cols), int(rows));
      } else {
        DUMP_TENSOR(name, static_cast<const half*>(packed_scales.get()), int(batch), int(cols), int(rows));
      }
    }
  };
#define DUMP_PACK_TENSOR(name, packed_scales, scales) dump_tensor(name, packed_scales, scales)
#else
#define DUMP_PACK_TENSOR(name, packed_scales, scales)
#endif

#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
  if (enable_fp4_deep_gemm_ && (input_idx == 2 || input_idx == 5 || input_idx == 3 || input_idx == 6)) {
    const bool fc1 = input_idx == 2 || input_idx == 3;
    const bool weight = input_idx == 2 || input_idx == 5;
    const int64_t expected_k = fc1 ? onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize
                                   : onnxruntime::llm::kernels::deep_gemm_sm90::kInterSize;
    const int64_t expected_n = fc1 ? onnxruntime::llm::kernels::deep_gemm_sm90::kFc1OutputSize
                                   : onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize;
    const auto& shape = tensor.Shape();
    ORT_ENFORCE(shape.NumDimensions() == 3 &&
                    shape[0] == fp4_deep_gemm_num_experts_ &&
                    shape[1] == (weight ? expected_k : expected_n) &&
                    shape[2] == (weight ? expected_n / 2 : expected_k / 32),
                "QMoE FP4 DeepGEMM received an unexpected ", fc1 ? "fc1" : "fc2", weight ? " weight" : " scale",
                " shape: ", shape.ToString());
    IAllocatorUniquePtr<void>& staged =
        weight ? (fc1 ? fp4_deep_gemm_fc1_staged_weights_ : fp4_deep_gemm_fc2_staged_weights_)
               : (fc1 ? fp4_deep_gemm_fc1_staged_block_scales_ : fp4_deep_gemm_fc2_staged_block_scales_);
    bool staged_packed = false;
    PrePackCopyToGpu(tensor, stream, alloc, staged, staged_packed);
  }
#endif

  if (input_idx == 2 && ((quant_type_ == "fp4" && !use_fp4_dequant_fallback_) ||
                         (quant_type_ == "nvfp4" && !use_fp4_dequant_fallback_) ||
                         (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_))) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc1_weights_, is_packed);
    // Native CUTLASS + GEMV coexist. The optional raw NVFP4 route skips the decode repack.
    bool gemv_packed = false;
    if ((quant_type_ == "fp4" || (quant_type_ == "nvfp4" && !nvfp4_gemv_raw_layout_)) && enable_fp4_gemv_) {
      const bool use_interleave =
          quant_type_ == "fp4" && onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved();
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_, gemv_packed, use_interleave);
    }
    // All native FP4 modes still need the raw tensor for input validation and fallback.
    is_packed = false;
  } else if (input_idx == 5 && ((quant_type_ == "fp4" && !use_fp4_dequant_fallback_) ||
                                (quant_type_ == "nvfp4" && !use_fp4_dequant_fallback_) ||
                                (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_))) {
    PrePackRepackFP4Weights(tensor, stream, alloc, packed_fp4_fc2_weights_, is_packed);
    bool gemv_packed = false;
    if ((quant_type_ == "fp4" || (quant_type_ == "nvfp4" && !nvfp4_gemv_raw_layout_)) && enable_fp4_gemv_) {
      const bool use_interleave =
          quant_type_ == "fp4" && onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved();
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_, gemv_packed, use_interleave);
    }
    is_packed = false;
  } else if (input_idx == 2 &&
             (quant_type_ == "fp4" || (quant_type_ == "nvfp4" && !nvfp4_gemv_raw_layout_)) && enable_fp4_gemv_) {
    // Fused FP4 GEMV: lay out fc1 weights as [E, 2*inter, hidden/2] row-major.
    // MXFP4 keeps the raw initializer unless release_fp4_raw_weights_ is set. When the SM80 grouped-GEMM
    // port is enabled (MXFP4 only), force the SM80 CUTLASS ColumnMajorTileInterleave layout
    // so this buffer feeds the prefill grouped GEMM. The decode GEMV reads that same buffer and
    // inverts the nibble pair-interleave in-register (Fp4KernelDetailsSm80Pair), so one copy of
    // the e2m1 weights serves both regimes; only when the shape misses the interleaved GEMV rules
    // do we pack a dedicated GEMV-native copy into gemv_fp4_fc1_weights_decode_.
    const bool use_interleave =
        quant_type_ == "fp4" &&
        (onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved() || enable_fp4_sm80_gemm_);
    const bool sm80_pair = enable_fp4_sm80_gemm_;
    bool local_packed = false;
    PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_, local_packed,
                            use_interleave, /*sm80_pair_interleave=*/sm80_pair);
    if (sm80_pair) {
      // Raw initializer is [E, K, N/2]; the GEMV problem is n = N, k = K.
      const auto& raw_shape = tensor.Shape();
      gemv_fp4_fc1_reads_sm80_layout_ = onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_fp4_sm80_layout_supported(
          raw_shape[2] * 2, raw_shape[1], /*group_size=*/32);
      if (!gemv_fp4_fc1_reads_sm80_layout_) {
        LogQMoEFp4DualLayoutOnce();
      }
    }
    if (enable_fp4_sm80_gemm_ && !gemv_fp4_fc1_reads_sm80_layout_) {
      bool decode_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc1_weights_decode_, decode_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved(),
                              /*sm80_pair_interleave=*/false);
    }
    if (release_fp4_raw_weights_ && local_packed) {
      // Cache the source shape for CheckInputs and let ORT free the raw initializer.
      fc1_weights_shape_ = tensor.Shape();
      is_packed = true;
    }
  } else if (input_idx == 5 &&
             (quant_type_ == "fp4" || (quant_type_ == "nvfp4" && !nvfp4_gemv_raw_layout_)) && enable_fp4_gemv_) {
    const bool use_interleave =
        quant_type_ == "fp4" &&
        (onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved() || enable_fp4_sm80_gemm_);
    const bool sm80_pair = enable_fp4_sm80_gemm_;
    bool local_packed = false;
    PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_, local_packed,
                            use_interleave, /*sm80_pair_interleave=*/sm80_pair);
    if (sm80_pair) {
      const auto& raw_shape = tensor.Shape();
      gemv_fp4_fc2_reads_sm80_layout_ = onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_fp4_sm80_layout_supported(
          raw_shape[2] * 2, raw_shape[1], /*group_size=*/32);
      if (!gemv_fp4_fc2_reads_sm80_layout_) {
        LogQMoEFp4DualLayoutOnce();
      }
    }
    if (enable_fp4_sm80_gemm_ && !gemv_fp4_fc2_reads_sm80_layout_) {
      bool decode_packed = false;
      PrePackRepackFP4Weights(tensor, stream, alloc, gemv_fp4_fc2_weights_decode_, decode_packed,
                              onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvUseInterleaved(),
                              /*sm80_pair_interleave=*/false);
    }
    if (release_fp4_raw_weights_ && local_packed) {
      fc2_weights_shape_ = tensor.Shape();
      is_packed = true;
    }
  } else if (input_idx == 2 && quant_type_ == "int" && !weights_prepacked_) {
    // Caller opted in (``weights_prepacked=0`` attribute) to having ORT
    // do the CUTLASS fpA_intB layout transform internally, instead of
    // shipping pre-prepacked bytes. Mirrors ``MatMulNBits::PrePack_B``
    // looped over the E experts of ``[E, N, K/pack]``. We cache the
    // source shape in ``fc1_weights_shape_`` so ``CheckInputs`` can be
    // satisfied without holding the original initializer alive, then
    // set ``is_packed = true`` to let ORT free it.
    if (is_mixed_width || expert_weight_bits_ == 2) {
      if (int_weight_supports_packed_execution(fc1_expert_weight_bits_)) {
        bool local_packed = false;
        PrePackIntExpertWeights(tensor, stream, alloc, packed_fc1_weights_, local_packed,
                                fc1_expert_weight_bits_);
      }
      is_packed = false;
    } else {
      fc1_weights_shape_ = tensor.Shape();
      PrePackIntExpertWeights(tensor, stream, alloc, packed_fc1_weights_, is_packed,
                              fc1_expert_weight_bits_);
    }
  } else if (input_idx == 5 && quant_type_ == "int" && !weights_prepacked_) {
    if (is_mixed_width || expert_weight_bits_ == 2) {
      if (int_weight_supports_packed_execution(fc2_expert_weight_bits_)) {
        bool local_packed = false;
        PrePackIntExpertWeights(tensor, stream, alloc, packed_fc2_weights_, local_packed,
                                fc2_expert_weight_bits_);
      }
      is_packed = false;
    } else {
      fc2_weights_shape_ = tensor.Shape();
      PrePackIntExpertWeights(tensor, stream, alloc, packed_fc2_weights_, is_packed,
                              fc2_expert_weight_bits_);
    }
  } else if (input_idx == 3) {  // fc1_scales
    DUMP_TENSOR("fc1_scales", tensor);
    if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(tensor.IsDataType<Float8E4M3FN>() && tensor.Shape().NumDimensions() == 3,
                  "QMoE NVFP4 fc1_scales must be a 3-D float8e4m3fn tensor.");
    }
    if (quant_type_ == "nvfp4" && !use_fp4_dequant_fallback_) {
      // Native SM120 NVFP4 consumes the 128x4-swizzled block-scale layout (the same SF atom as the
      // SM100 WFP4AFP8 path; byte-level swizzle is format-agnostic). Keep a raw E4M3 copy in
      // gemv_fp4_fc1_block_raw_ for the GEMV decode and dequant fallback paths.
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
      if (enable_fp4_gemv_ && !nvfp4_weights_row_major_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc1_block_raw_, raw_packed);
        gemv_fp4_fc1_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc1_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc1_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(1, stream, alloc);
      }
    } else if (quant_type_ == "fp4" && !use_fp4_dequant_fallback_) {
      PrePackFp4ScalesForTmaWs(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
      // Native CUTLASS swizzles the block scales; keep a raw copy + dims so GEMV decode can
      // build its combined scale layout.
      if (enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc1_block_raw_, raw_packed);
        gemv_fp4_fc1_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc1_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc1_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(1, stream, alloc);
      }
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc1_block_scales_, is_packed);
      if ((quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        gemv_fp4_fc1_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc1_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc1_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(1, stream, alloc);
      }
    } else if (quant_type_ == "int") {
      if (uses_packed_int_decode) {
        if (packed_fc1_weights_) {
          bool local_packed = false;
          PrePackTransposeAndPack(tensor, stream, alloc, gemv_int_fc1_scales_, local_packed);
        }
        is_packed = false;
      } else {
        PrePackTransposeAndPack(tensor, stream, alloc, packed_fc1_scales_, is_packed);
        DUMP_PACK_TENSOR("packed_fc1_scales", packed_fc1_scales_, tensor);
      }
    }
    if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      is_packed = false;
    }
  } else if (input_idx == 6) {  // fc2_scales
    DUMP_TENSOR("fc2_scales", tensor);
    if (quant_type_ == "nvfp4") {
      ORT_ENFORCE(tensor.IsDataType<Float8E4M3FN>() && tensor.Shape().NumDimensions() == 3,
                  "QMoE NVFP4 fc2_scales must be a 3-D float8e4m3fn tensor.");
    }
    if (quant_type_ == "nvfp4" && !use_fp4_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
      if (enable_fp4_gemv_ && !nvfp4_weights_row_major_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc2_block_raw_, raw_packed);
        gemv_fp4_fc2_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc2_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc2_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(2, stream, alloc);
      }
    } else if (quant_type_ == "fp4" && !use_fp4_dequant_fallback_) {
      PrePackFp4ScalesForTmaWs(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
      if (enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        bool raw_packed = false;
        PrePackCopyToGpu(tensor, stream, alloc, gemv_fp4_fc2_block_raw_, raw_packed);
        gemv_fp4_fc2_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc2_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc2_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(2, stream, alloc);
      }
    } else if (quant_type_ == "wfp4afp8" && !use_wfp4afp8_dequant_fallback_) {
      PrePackSwizzleBlockScales(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
    } else if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fp4_fc2_block_scales_, is_packed);
      if ((quant_type_ == "fp4" || quant_type_ == "nvfp4") && enable_fp4_gemv_ && tensor.Shape().NumDimensions() == 3) {
        gemv_fp4_fc2_scale_e_ = tensor.Shape()[0];
        gemv_fp4_fc2_scale_n_ = tensor.Shape()[1];
        gemv_fp4_fc2_scale_kb_ = tensor.Shape()[2];
        TryBuildGemvFp4Scales(2, stream, alloc);
      }
    } else if (quant_type_ == "int") {
      if (uses_packed_int_decode) {
        if (packed_fc2_weights_) {
          bool local_packed = false;
          PrePackTransposeAndPack(tensor, stream, alloc, gemv_int_fc2_scales_, local_packed);
        }
        is_packed = false;
      } else {
        PrePackTransposeAndPack(tensor, stream, alloc, packed_fc2_scales_, is_packed);
        DUMP_PACK_TENSOR("packed_fc2_scales", packed_fc2_scales_, tensor);
      }
    }
    if (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "wfp4afp8") {
      is_packed = false;
    }
  } else if (input_idx == 11) {  // fc1_zeros
    DUMP_TENSOR("fc1_zeros", tensor);
    PrePackComputeBias(tensor, stream, alloc, packed_fc1_scales_, packed_fc1_bias_, is_packed);
    DUMP_PACK_TENSOR("packed_fc1_bias", packed_fc1_bias_, tensor);
  } else if (input_idx == 12) {  // fc2_zeros
    DUMP_TENSOR("fc2_zeros", tensor);
    PrePackComputeBias(tensor, stream, alloc, packed_fc2_scales_, packed_fc2_bias_, is_packed);
    DUMP_PACK_TENSOR("packed_fc2_bias", packed_fc2_bias_, tensor);
  } else if ((input_idx == 15 || input_idx == 16) &&
             (quant_type_ == "fp4" || quant_type_ == "nvfp4" || quant_type_ == "fp8" || quant_type_ == "wfp4afp8")) {
    ORT_ENFORCE(tensor.IsDataType<float>() && tensor.Shape().NumDimensions() == 1,
                "QMoE ", quant_type_, " global scales must be 1-D float tensors.");
    const auto is_nvfp4_decode_scale_eligible = [&](const Tensor& scale_tensor) -> bool {
      if (quant_type_ != "nvfp4") {
        return false;
      }
      const int64_t count = scale_tensor.Shape().Size();
      if (count <= 0) {
        return false;
      }
      const float* scales = scale_tensor.Data<float>();
      std::unique_ptr<float[]> host_scales;
      if (scale_tensor.Location().device.Type() != OrtDevice::CPU) {
        host_scales = std::make_unique<float[]>(static_cast<size_t>(count));
        CUDA_CALL_THROW(cudaMemcpyAsync(host_scales.get(), scales, static_cast<size_t>(count) * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
        CUDA_CALL_THROW(cudaStreamSynchronize(stream));
        scales = host_scales.get();
      }
      for (int64_t i = 0; i < count; ++i) {
        if (!onnxruntime::llm::kernels::moe_gemv::IsNvfp4DecodeFastScale(scales[i])) {
          return false;
        }
      }
      return true;
    };
    if (input_idx == 15) {
      nvfp4_fc1_decode_scale_eligible_ = is_nvfp4_decode_scale_eligible(tensor);
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc1_global_scale_, is_packed);
      fp4_fc1_global_scale_e_ = tensor.Shape()[0];
      TryBuildGemvFp4Scales(1, stream, alloc);
    } else {
      nvfp4_fc2_decode_scale_eligible_ = is_nvfp4_decode_scale_eligible(tensor);
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc2_global_scale_, is_packed);
      fp4_fc2_global_scale_e_ = tensor.Shape()[0];
      TryBuildGemvFp4Scales(2, stream, alloc);
    }
    is_packed = false;
  } else if ((input_idx == 17 || input_idx == 18) && quant_type_ == "wfp4afp8") {
    if (input_idx == 17) {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc1_act_scale_, is_packed);
    } else {
      PrePackCopyToGpu(tensor, stream, alloc, packed_fc2_act_scale_, is_packed);
    }
  }

  if (enable_fp4_deep_gemm_) {
    if (input_idx == 2 || input_idx == 3 || input_idx == 15) {
      TryBuildFp4DeepGemmWeights(1, stream, alloc);
    } else if (input_idx == 5 || input_idx == 6 || input_idx == 16) {
      TryBuildFp4DeepGemmWeights(2, stream, alloc);
    }
  }

  return Status::OK();
}

// ---------------------------------------------------------------------------
// PrePack helper: Transpose [E, N, Blocks] -> [E, Blocks, N] and copy to GPU.
// ---------------------------------------------------------------------------
void QMoE::PrePackTransposeAndPack(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  size_t bytes = tensor.SizeInBytes();
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyDefault, stream));
    p_src = temp_src_gpu.get();
  }

  if (shape.NumDimensions() == 3 && shape[2] > 1) {
    size_t rows = shape[1];   // N
    size_t cols = shape[2];   // Blocks
    size_t batch = shape[0];  // Experts
    const int32_t element_type = tensor.GetElementType();
    if (element_type == utils::ToTensorProtoElementType<MLFloat16>()) {
      LaunchQMoETranspose2D(static_cast<const half*>(p_src), static_cast<half*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (element_type == utils::ToTensorProtoElementType<BFloat16>()) {
      LaunchQMoETranspose2D(static_cast<const __nv_bfloat16*>(p_src), static_cast<__nv_bfloat16*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (element_type == utils::ToTensorProtoElementType<float>()) {
      LaunchQMoETranspose2D(static_cast<const float*>(p_src), static_cast<float*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (element_type == utils::ToTensorProtoElementType<uint8_t>()) {
      LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src), static_cast<uint8_t*>(packed_buf.get()), batch, rows, cols, stream);
    } else if (element_type == utils::ToTensorProtoElementType<Float8E8M0>()) {
      LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src), static_cast<uint8_t*>(packed_buf.get()), batch, rows, cols, stream);
    } else {
      ORT_THROW("Unsupported data type for scale transposition");
    }
  } else {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyDefault, stream));
  }

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Copy tensor to GPU without transformation.
// ---------------------------------------------------------------------------
void QMoE::PrePackCopyToGpu(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                            IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  size_t bytes = tensor.SizeInBytes();
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
  const void* p_src = tensor.DataRaw();
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
  } else {
    CUDA_CALL_THROW(cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyDeviceToDevice, stream));
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: int2/int4/int8 per-expert weights → CUTLASS fpA_intB layout.
// ---------------------------------------------------------------------------
// Mirrors ``MatMulNBits::PrePack_B`` but loops over the leading E (experts)
// dimension. Input ``tensor`` is the row-major 3-D ``[E, N, K/(8/bits)]``
// quantized weight initializer; output is a GPU buffer in the
// kernel-expected ``[E, K, N/(8/bits)]`` layout.
void QMoE::PrePackIntExpertWeights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed,
                                   int64_t weight_bits) {
  ORT_ENFORCE(weight_bits == 2 || weight_bits == 4 || weight_bits == 8,
              "PrePackIntExpertWeights: only 2, 4, and 8 bits are supported, got ", weight_bits);
  ORT_ENFORCE(sm_ >= 75,
              "PrePackIntExpertWeights: quant_type='int' with weights_prepacked=0 requires SM75+ CUDA hardware, got SM",
              sm_);
  const auto& shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3,
              "PrePackIntExpertWeights: expected 3-D weight tensor [E, N, K/pack], got ndim=",
              shape.NumDimensions());

  const int bits = static_cast<int>(weight_bits);
  const int pack_factor = 8 / bits;
  const int64_t num_experts = shape[0];
  const int64_t n = shape[1];
  const int64_t k_packed = shape[2];
  const int64_t k = k_packed * pack_factor;

  // The CUDA QMoE int4/int8 MoE GEMM always dispatches to the Ampere (SM80)
  // grouped-GEMM kernel -- even on SM90 -- because mixed int-weight + fp16/bf16
  // is not a valid Hopper TMA warp-specialized specialisation. The kernel thus
  // consumes the SM80 CUTLASS fpA_intB layout on every GPU, so the weights must
  // always be preprocessed for SM80 regardless of the runtime device SM.
  // (Using get_arch_for_mixed_gemm_weight_preprocess(sm_) here would emit the
  // SM90 layout on Hopper, which the SM80 kernel cannot consume -> wrong output.)
  const int packing_sm =
      onnxruntime::llm::kernels::weight_only::get_arch_for_mixed_gemm_weight_preprocess(80);

  // Per-expert sizes.
  const size_t per_expert_bytes = static_cast<size_t>(n) * static_cast<size_t>(k) / pack_factor;
  const size_t total_bytes = per_expert_bytes * static_cast<size_t>(num_experts);

  // Output buffer holds all E prepacked experts back-to-back in
  // [E, K, N/pack_factor] layout.
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, total_bytes, /*use_reserve=*/true);
  int8_t* dst_all = reinterpret_cast<int8_t*>(packed_buf.get());

  // Two transient per-expert scratch buffers reused across experts.
  IAllocatorUniquePtr<void> transposed_scratch =
      this->GetTransientScratchBuffer<void>(per_expert_bytes);
  int8_t* transposed_scratch_ptr = reinterpret_cast<int8_t*>(transposed_scratch.get());

  IAllocatorUniquePtr<void> src_gpu_scratch;
  const uint8_t* src_base_gpu = nullptr;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    src_gpu_scratch = this->GetTransientScratchBuffer<void>(total_bytes);
    CUDA_CALL_THROW(cudaMemcpyAsync(src_gpu_scratch.get(), tensor.DataRaw(), total_bytes,
                                    cudaMemcpyHostToDevice, stream));
    src_base_gpu = reinterpret_cast<const uint8_t*>(src_gpu_scratch.get());
  } else {
    src_base_gpu = reinterpret_cast<const uint8_t*>(tensor.DataRaw());
  }

  IAllocatorUniquePtr<int32_t> permutation_map = this->GetTransientScratchBuffer<int32_t>(64);

  using onnxruntime::llm::kernels::weight_only::QuantType;
  const QuantType quant_type = bits == 2 ? QuantType::W2_A16
                                         : (bits == 4 ? QuantType::W4_A16 : QuantType::W8_A16);

  for (int64_t e = 0; e < num_experts; ++e) {
    const uint8_t* src_e = src_base_gpu + static_cast<size_t>(e) * per_expert_bytes;
    int8_t* dst_e = dst_all + static_cast<size_t>(e) * per_expert_bytes;

    // Step 1: transpose + (for int4) unpack/zero-point bias into the
    // transposed-int8 scratch buffer. Mirrors MatMulNBits's PrePack_B.
    if (bits == 2) {
      onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint2_transposed_to_int8_direct_cuda(
          stream, transposed_scratch_ptr, src_e, static_cast<int>(n), static_cast<int>(k));
    } else if (bits == 4) {
      onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
          stream, transposed_scratch_ptr, src_e, static_cast<int>(n), static_cast<int>(k));
    } else {
      onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
          stream, transposed_scratch_ptr, src_e, static_cast<int>(n), static_cast<int>(k));
    }

    // Step 2: apply the CUTLASS fpA_intB row-permutation / column-interleave /
    // bias / pair-interleave transform into the per-expert output slot.
    // ``synchronize=false``: avoid one host-blocking ``cudaStreamSynchronize``
    // per expert (which would scale model-load time with ``num_experts``).
    // Stream ordering guarantees expert e's transform finishes before expert
    // e+1 reuses the shared transpose scratch, and a single sync after the loop
    // makes the whole batch complete before the scratch buffers are freed.
    onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
        stream,
        packing_sm,
        dst_e,
        transposed_scratch_ptr,
        permutation_map.get(),
        {static_cast<size_t>(k), static_cast<size_t>(n)},
        quant_type,
        /*synchronize=*/false);
  }

  // Single host-blocking sync after all experts: this guarantees every
  // per-expert transform (and the CPU->GPU staging copy above) is complete, so
  // the transient scratch buffers are safe to free on return.
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Swizzle MXFP block scales for SM120 TMA layout using GPU kernel.
// ---------------------------------------------------------------------------
void QMoE::PrePackSwizzleBlockScales(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                     IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 block scales for native block-scale swizzle");
  ORT_ENFORCE(tensor.IsDataType<Float8E8M0>() || tensor.IsDataType<Float8E4M3FN>(),
              "Expected Float8E8M0 (MXFP4) or Float8E4M3FN (NVFP4) FP4 block scales for native block-scale swizzle");

  const int64_t experts = shape[0];
  const int64_t rows = shape[1];
  const int64_t scale_cols = shape[2];
  ORT_ENFORCE(experts > 0 && rows > 0 && scale_cols > 0,
              "FP4 block scales must have positive dimensions, got ", shape.ToString());
  const int64_t rows_padded_i64 = ((rows + 127) / 128) * 128;
  const int64_t cols_padded_i64 = ((scale_cols + 3) / 4) * 4;
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max() && rows <= std::numeric_limits<int>::max() &&
                  scale_cols <= std::numeric_limits<int>::max() &&
                  rows_padded_i64 <= std::numeric_limits<int>::max() &&
                  cols_padded_i64 <= std::numeric_limits<int>::max(),
              "FP4 block-scale dimensions exceed CUDA launch int range, got ", shape.ToString());
  const int rows_padded = static_cast<int>(rows_padded_i64);
  const int cols_padded = static_cast<int>(cols_padded_i64);
  const size_t dst_bytes = SafeInt<size_t>(experts) * SafeInt<size_t>(rows_padded) *
                           SafeInt<size_t>(cols_padded) * sizeof(uint8_t);

  // Ensure input is on GPU
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, tensor.SizeInBytes(), cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  // QMoEBlockScaleInterleaveKernel writes every byte of the output buffer
  // (the (batch, row, col) -> offset map is a bijection over
  // [0, batch_size) x [0, rows_padded) x [0, cols_padded), and padded
  // source positions are written as 0), so no explicit memset is required.
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, dst_bytes, true);

  int multi_processor_count = 0;
  int device_id = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device_id));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id));

  LaunchQMoEBlockScaleInterleave(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      static_cast<int>(rows),
      static_cast<int>(scale_cols),
      rows_padded,
      cols_padded,
      multi_processor_count,
      stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

void QMoE::PrePackFp4ScalesForTmaWs(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                    IAllocatorUniquePtr<void>& packed_buf, bool& is_packed) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 block scales for WFP4A16 native prepack");
  ORT_ENFORCE(tensor.IsDataType<Float8E8M0>(), "Expected Float8E8M0 FP4 block scales for WFP4A16 native prepack");

  const int64_t experts = shape[0];
  const int64_t rows = shape[1];
  const int64_t k_blocks = shape[2];
  ORT_ENFORCE(experts > 0 && rows > 0 && k_blocks > 0,
              "FP4 block scales must have positive dimensions, got ", shape.ToString());
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max() && rows <= std::numeric_limits<int>::max() &&
                  k_blocks <= std::numeric_limits<int>::max(),
              "FP4 block-scale dimensions exceed CUDA launch int range, got ", shape.ToString());

  const size_t bytes = tensor.SizeInBytes();
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  // LaunchQMoEPackFp4ScalesForTmaWs pads the k-block count up to a multiple of the packed
  // K-tile (8). Allocate the output buffer for the padded count so the GEMM's last partial
  // CTA-K-tile can read a full packed scale group. The kernel writes every padded element
  // (real blocks copied, tail blocks zeroed), so the buffer is fully initialized.
  constexpr int64_t kPackedScalesPerKTile = 8;
  const int64_t k_blocks_padded =
      ((k_blocks + kPackedScalesPerKTile - 1) / kPackedScalesPerKTile) * kPackedScalesPerKTile;
  const size_t out_bytes =
      SafeInt<size_t>(experts) * SafeInt<size_t>(rows) * SafeInt<size_t>(k_blocks_padded);
  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, out_bytes, true);
  LaunchQMoEPackFp4ScalesForTmaWs(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      static_cast<int>(rows),
      static_cast<int>(k_blocks),
      stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

// ---------------------------------------------------------------------------
// PrePack helper: Repack column-major FP4 weights to row-major using GPU kernel.
// ---------------------------------------------------------------------------
void QMoE::PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                   IAllocatorUniquePtr<void>& packed_buf, bool& is_packed,
                                   bool gemv_interleaved, bool sm80_pair_interleave) {
  auto shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D FP4 weights for WFP4AFP8 native prepack");
  ORT_ENFORCE(tensor.IsDataType<uint8_t>(), "Expected uint8 FP4 weights for WFP4AFP8 native prepack");

  const int64_t experts = shape[0];
  const int64_t k = shape[1];
  const int64_t n = shape[2] * 2;  // Packed: n/2 bytes per row in source
  ORT_ENFORCE(experts > 0 && k > 0 && n > 0, "FP4 weights must have positive dimensions, got ", shape.ToString());
  ORT_ENFORCE(k % 2 == 0 && n % 2 == 0,
              "FP4 weight repack requires even k and n dimensions, got k=", k, ", n=", n);
  ORT_ENFORCE(experts <= std::numeric_limits<int>::max(),
              "FP4 weight expert count exceeds CUDA launch int range, got ", experts);
  const size_t bytes = tensor.SizeInBytes();

  // Ensure input is on GPU
  const void* p_src = tensor.DataRaw();
  IAllocatorUniquePtr<void> temp_src_gpu;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
    CUDA_CALL_THROW(cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyHostToDevice, stream));
    p_src = temp_src_gpu.get();
  }

  packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  if (gemv_interleaved) {
    // Interleaved layout: produce the INT4-style ColumnMajorInterleaved FP4 weight layout instead of the
    // [E, n, k/2] row-major ColToRow layout. The source per expert is [k, n/2] bytes == a
    // [K, N] row-major W4 (e2m1) tensor, which is exactly what the CUTLASS fpA_intB SM80 W4_A16
    // preprocessor consumes (shape {k, n}). The preprocessor's layout-only steps 1-3 (row-
    // permute + subbyte-transpose + column-interleave) apply to e2m1 unchanged; step 4 (the
    // integer +bias + pair-interleave) MUST be skipped (apply_bias_interleave=false) because it
    // would corrupt the floating-point nibbles. Block scales are unchanged (kStepK=32 == MXFP4
    // block size), so TryBuildGemvFp4Scales's [E,k/32,n] layout already maps one scale per
    // column per K-block.
    using onnxruntime::llm::kernels::weight_only::QuantType;
    // The fused FP4 GEMV always runs on the SM80 fpA_intB layout (mixed W4 + fp16/bf16 act is
    // not a Hopper TMA specialisation), so preprocess for SM80 regardless of the runtime device.
    const int packing_sm =
        onnxruntime::llm::kernels::weight_only::get_arch_for_mixed_gemm_weight_preprocess(80);
    const size_t per_expert_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) / 2;
    int8_t* dst_all = reinterpret_cast<int8_t*>(packed_buf.get());
    const int8_t* src_all = reinterpret_cast<const int8_t*>(p_src);

    // The preprocessor mutates its input buffers in place (it ping-pongs between the dst and a
    // writable src across the layout steps), so feed it a writable per-expert scratch copy of
    // the (const) source rather than the source initializer itself. Reused across experts.
    IAllocatorUniquePtr<void> src_scratch = this->GetTransientScratchBuffer<void>(per_expert_bytes);
    int8_t* src_scratch_ptr = reinterpret_cast<int8_t*>(src_scratch.get());
    IAllocatorUniquePtr<int32_t> permutation_map = this->GetTransientScratchBuffer<int32_t>(32);

    for (int64_t e = 0; e < experts; ++e) {
      CUDA_CALL_THROW(cudaMemcpyAsync(src_scratch_ptr, src_all + static_cast<size_t>(e) * per_expert_bytes,
                                      per_expert_bytes, cudaMemcpyDeviceToDevice, stream));
      // synchronize=false: one host-blocking sync after the loop (stream ordering guarantees
      // expert e finishes before e+1 reuses src_scratch). apply_bias_interleave=false for e2m1.
      // When this buffer feeds the SM80 MoE grouped GEMM (sm80_pair_interleave), additionally
      // apply step 4's nibble pair-interleave WITHOUT the +8 bias, which is the layout that
      // GEMM's e2m1 dequant converter inverts. The fused GEMV decode kernel consumes the plain
      // steps-1-3 layout, so it packs its own copy with sm80_pair_interleave=false.
      onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
          stream,
          packing_sm,
          dst_all + static_cast<size_t>(e) * per_expert_bytes,
          src_scratch_ptr,
          permutation_map.get(),
          {static_cast<size_t>(k), static_cast<size_t>(n)},
          QuantType::W4_A16,
          /*synchronize=*/false,
          /*apply_bias_interleave=*/false,
          /*interleave_without_bias=*/sm80_pair_interleave);
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    is_packed = true;
    return;
  }

  LaunchQMoERepackFP4ColToRow(
      static_cast<const uint8_t*>(p_src),
      static_cast<uint8_t*>(packed_buf.get()),
      static_cast<int>(experts),
      k, n, stream);

  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

void QMoE::TryBuildGemvFp4Scales(int fc, cudaStream_t stream, AllocatorPtr alloc) {
  const bool is_nvfp4 = (quant_type_ == "nvfp4");
  if (!enable_fp4_gemv_ || (quant_type_ != "fp4" && !is_nvfp4)) {
    return;
  }
  // Raw-layout NVFP4 GEMV decodes the original E4M3 block scales and applies the per-expert
  // global scale in-register. Avoid materializing a persistent activation-dtype scale bank.
  if (is_nvfp4 && nvfp4_gemv_raw_layout_) {
    return;
  }
  IAllocatorUniquePtr<void>& block = (fc == 1) ? packed_fp4_fc1_block_scales_ : packed_fp4_fc2_block_scales_;
  IAllocatorUniquePtr<void>& raw_block = (fc == 1) ? gemv_fp4_fc1_block_raw_ : gemv_fp4_fc2_block_raw_;
  // Native CUTLASS swizzles packed_fp4_*_block_scales_; in that regime read the raw e8m0 copy.
  IAllocatorUniquePtr<void>& src_block = (raw_block != nullptr) ? raw_block : block;
  IAllocatorUniquePtr<void>& global = (fc == 1) ? packed_fc1_global_scale_ : packed_fc2_global_scale_;
  IAllocatorUniquePtr<void>& out = (fc == 1) ? gemv_fp4_fc1_scales_ : gemv_fp4_fc2_scales_;
  const int64_t experts = (fc == 1) ? gemv_fp4_fc1_scale_e_ : gemv_fp4_fc2_scale_e_;
  const int64_t n = (fc == 1) ? gemv_fp4_fc1_scale_n_ : gemv_fp4_fc2_scale_n_;
  const int64_t k_blocks = (fc == 1) ? gemv_fp4_fc1_scale_kb_ : gemv_fp4_fc2_scale_kb_;
  const int64_t global_scale_e = (fc == 1) ? fp4_fc1_global_scale_e_ : fp4_fc2_global_scale_e_;

  // Build exactly once, and only when both inputs are present and dimensions are known.
  if (out != nullptr || src_block == nullptr || global == nullptr || experts <= 0 || n <= 0 || k_blocks <= 0 ||
      global_scale_e != experts) {
    return;
  }

  const size_t element_size = is_fp16_ ? sizeof(half) : sizeof(__nv_bfloat16);
  const size_t bytes = SafeInt<size_t>(experts) * SafeInt<size_t>(n) * SafeInt<size_t>(k_blocks) * element_size;
  out = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

  if (is_fp16_) {
    if (is_nvfp4) {
      LaunchQMoECombineNvfp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<half*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    } else {
      LaunchQMoECombineFp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<half*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    }
  } else {
    if (is_nvfp4) {
      LaunchQMoECombineNvfp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<__nv_bfloat16*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    } else {
      LaunchQMoECombineFp4ScalesForGemv(
          static_cast<const uint8_t*>(src_block.get()),
          static_cast<const float*>(global.get()),
          static_cast<__nv_bfloat16*>(out.get()),
          static_cast<int>(experts), static_cast<int>(n), static_cast<int>(k_blocks), stream);
    }
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));

  // In the SM80 grouped-GEMM regime the combined [E, k/32, n] activation-dtype scales built above
  // are the only scales any dispatch reads (prefill via QuantParams::GroupWise, decode via the
  // fused GEMV), and the dequant fallback that consumes the raw e8m0 block scales is unreachable.
  // Drop the staged e8m0 copy -- for a 20B-class MXFP4 MoE that is a few hundred MiB of dead
  // device memory. ``block`` is the buffer ``src_block`` aliases in this regime (raw_block is only
  // populated by the native CUTLASS paths, which do not set release_fp4_raw_weights_).
  if (release_fp4_raw_weights_ && raw_block == nullptr) {
    block.reset();
  }
}

void QMoE::TryBuildFp4DeepGemmWeights(int fc, cudaStream_t stream, AllocatorPtr alloc) {
#if defined(HAS_SM90_OR_LATER) && defined(USE_DEEP_GEMM)
  if (!enable_fp4_deep_gemm_) {
    return;
  }

  IAllocatorUniquePtr<void>& staged_weights =
      fc == 1 ? fp4_deep_gemm_fc1_staged_weights_ : fp4_deep_gemm_fc2_staged_weights_;
  IAllocatorUniquePtr<void>& staged_block_scales =
      fc == 1 ? fp4_deep_gemm_fc1_staged_block_scales_ : fp4_deep_gemm_fc2_staged_block_scales_;
  IAllocatorUniquePtr<void>& global_scale = fc == 1 ? packed_fc1_global_scale_ : packed_fc2_global_scale_;
  IAllocatorUniquePtr<void>& output = fc == 1 ? fp4_deep_gemm_fc1_weights_ : fp4_deep_gemm_fc2_weights_;
  IAllocatorUniquePtr<void>& output_scales =
      fc == 1 ? fp4_deep_gemm_fc1_weight_scales_ : fp4_deep_gemm_fc2_weight_scales_;
  if (output != nullptr || staged_weights == nullptr || staged_block_scales == nullptr || global_scale == nullptr) {
    return;
  }

  const int num_experts = fp4_deep_gemm_num_experts_;
  const int n = fc == 1 ? onnxruntime::llm::kernels::deep_gemm_sm90::kFc1OutputSize
                        : onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize;
  const int k = fc == 1 ? onnxruntime::llm::kernels::deep_gemm_sm90::kHiddenSize
                        : onnxruntime::llm::kernels::deep_gemm_sm90::kInterSize;
  const size_t weight_bytes = SafeInt<size_t>(num_experts) * SafeInt<size_t>(n) * SafeInt<size_t>(k);
  const size_t scale_bytes =
      onnxruntime::llm::kernels::deep_gemm_sm90::WeightScaleCount(num_experts, n, k) * sizeof(float);
  output = IAllocator::MakeUniquePtr<void>(alloc, weight_bytes, true);
  output_scales = IAllocator::MakeUniquePtr<void>(alloc, scale_bytes, true);

  // The conversion is bit-exact only while every block's MXFP4 group exponents fit in e4m3's
  // range. The kernel verifies every element and sets inexact_flag on any mismatch. The host
  // check below is the safety boundary for using the converted weights: never retain them or
  // silently continue on the DeepGEMM path when the flag is set.
  auto inexact_flag = IAllocator::MakeUniquePtr<void>(alloc, sizeof(int), true);
  CUDA_CALL_THROW(cudaMemsetAsync(inexact_flag.get(), 0, sizeof(int), stream));
  LaunchQMoEQuantizeFp4WeightsToFp8(
      static_cast<const uint8_t*>(staged_weights.get()),
      static_cast<const uint8_t*>(staged_block_scales.get()),
      static_cast<const float*>(global_scale.get()), static_cast<uint8_t*>(output.get()),
      static_cast<float*>(output_scales.get()), static_cast<int*>(inexact_flag.get()),
      num_experts, n, k, stream);
  int inexact = 0;
  CUDA_CALL_THROW(cudaMemcpyAsync(&inexact, inexact_flag.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  // Fail closed before releasing the staged MXFP4 inputs. ORT session creation then reports the
  // unsupported conversion instead of allowing inference with non-bit-exact FP8 weights.
  ORT_ENFORCE(inexact == 0,
              "QMoE FP4 DeepGEMM: MXFP4 fc", fc,
              " weights do not convert losslessly to FP8 e4m3 (a [128 N, 128 K] block spans more "
              "exponents than e4m3 can hold). Set ORT_QMOE_FP4_DEEPGEMM=0.");
  staged_weights.reset();
  staged_block_scales.reset();
#else
  ORT_UNUSED_PARAMETER(fc);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(alloc);
#endif
}

// ---------------------------------------------------------------------------
// PrePack helper: Compute bias from zero-points and scales.
// ---------------------------------------------------------------------------
void QMoE::PrePackComputeBias(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                              const IAllocatorUniquePtr<void>& packed_scale,
                              IAllocatorUniquePtr<void>& packed_bias, bool& is_packed) {
  if ((expert_weight_bits_ == 4) && !packed_scale) {
    return;
  }

  size_t num_elements = tensor.Shape().Size();
  auto shape = tensor.Shape();

  if (expert_weight_bits_ == 8) {
    if (block_size_ > 0) {
      bool is_fp16 = is_fp16_;
      bool is_bf16 = !is_fp16_;
      size_t bytes = num_elements * (is_fp16 || is_bf16 ? 2 : 4);
      packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

      const void* p_src_zp = tensor.DataRaw();
      IAllocatorUniquePtr<void> temp_zp_gpu;
      if (tensor.Location().device.Type() == OrtDevice::CPU) {
        temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
        p_src_zp = temp_zp_gpu.get();
      }

      const void* p_zp_for_calc = p_src_zp;
      IAllocatorUniquePtr<void> temp_zp_transposed;

      if (shape.NumDimensions() == 3 && shape[2] > 1) {
        size_t rows = shape[1];
        size_t cols = shape[2];
        size_t batch = shape[0];
        temp_zp_transposed = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
        p_zp_for_calc = temp_zp_transposed.get();
      }

      if (is_fp16) {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const half*>(packed_scale.get()), static_cast<half*>(packed_bias.get()), num_elements, 128.0f, stream);
      } else if (is_bf16) {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const __nv_bfloat16*>(packed_scale.get()), static_cast<__nv_bfloat16*>(packed_bias.get()), num_elements, 128.0f, stream);
      } else {
        LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const float*>(packed_scale.get()), static_cast<float*>(packed_bias.get()), num_elements, 128.0f, stream);
      }
    } else {
      size_t bytes = num_elements * sizeof(uint8_t);
      packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

      const void* p_src_zp = tensor.DataRaw();
      IAllocatorUniquePtr<void> temp_zp_gpu;
      if (tensor.Location().device.Type() == OrtDevice::CPU) {
        temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
        p_src_zp = temp_zp_gpu.get();
      }

      if (shape.NumDimensions() == 3 && shape[2] > 1) {
        size_t rows = shape[1];
        size_t cols = shape[2];
        size_t batch = shape[0];
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(packed_bias.get()), batch, rows, cols, stream);
      } else {
        CUDA_CALL_THROW(cudaMemcpyAsync(packed_bias.get(), p_src_zp, bytes, cudaMemcpyDefault, stream));
      }
    }
  } else {
    if (block_size_ <= 0) {
      return;
    }

    ORT_ENFORCE(shape.NumDimensions() == 3, "Expected 3D zeros for block-wise 4-bit");
    ORT_ENFORCE(shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                "4-bit block-wise zeros must have positive dimensions, got ", shape.ToString());
    // packed_k_blocks is doubled to k_blocks below; constrain it to half of INT_MAX to keep the
    // doubled value (and the int dims passed into LaunchQMoEScaledZP4BitBatched) within int range.
    constexpr int64_t kMaxPackedKBlocks = std::numeric_limits<int>::max() / 2;
    ORT_ENFORCE(shape[0] <= std::numeric_limits<int>::max() &&
                    shape[1] <= std::numeric_limits<int>::max() &&
                    shape[2] <= kMaxPackedKBlocks,
                "4-bit block-wise zeros dimensions exceed CUDA launch int range, got ", shape.ToString());
    const int experts = static_cast<int>(shape[0]);
    const int n = static_cast<int>(shape[1]);
    const int packed_k_blocks = static_cast<int>(shape[2]);
    const int k_blocks = packed_k_blocks * 2;
    // QMoE only supports FP16/BF16 inputs (is_fp16_ is set in the ctor), both of which are 2 bytes.
    size_t output_count = static_cast<size_t>(experts) * static_cast<size_t>(k_blocks) * static_cast<size_t>(n);
    size_t bytes = output_count * sizeof(uint16_t);
    packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

    const void* p_src_zp = tensor.DataRaw();
    IAllocatorUniquePtr<void> temp_zp_gpu;
    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
      CUDA_CALL_THROW(cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
      p_src_zp = temp_zp_gpu.get();
    }

    const uint8_t* zp_ptr = static_cast<const uint8_t*>(p_src_zp);
    constexpr float kDefaultZeroPoint4Bit = 8.0f;
    if (is_fp16_) {
      LaunchQMoEScaledZP4BitBatched(
          zp_ptr,
          static_cast<const half*>(packed_scale.get()),
          static_cast<half*>(packed_bias.get()),
          experts, n, k_blocks, kDefaultZeroPoint4Bit, stream);
    } else {
      LaunchQMoEScaledZP4BitBatched(
          zp_ptr,
          static_cast<const __nv_bfloat16*>(packed_scale.get()),
          static_cast<__nv_bfloat16*>(packed_bias.get()),
          experts, n, k_blocks, kDefaultZeroPoint4Bit, stream);
    }
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  is_packed = true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
