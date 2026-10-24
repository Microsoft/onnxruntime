// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/common.h"
#include "core/framework/moe_offload_plan.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/moe/qmoe_offload.h"
#include "contrib_ops/cuda/moe/qmoe_row_tiling.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

constexpr const char* kEnableQMoEKernelDebugInfo = "ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO";
constexpr const char* kQMoERowTileSize = "ORT_QMOE_ROW_TILE_SIZE";
constexpr const char* kQMoERowTileSizeConfig = "ep.cuda.qmoe_row_tile_size";
constexpr const char* kQMoEIntDequantMaxScratchBytes = "ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES";
constexpr const char* kQMoEIntDequantMaxScratchBytesConfig = "ep.cuda.qmoe_int_dequant_max_scratch_bytes";

class QMoE final : public CudaKernel, public MoEBase, public onnxruntime::moe_offload::IMoeOffloadNode {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  ~QMoE() override;
  Status ComputeInternal(OpKernelContext* ctx) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

  const onnxruntime::moe_offload::MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept override {
    return offload_source_desc_;
  }
  const onnxruntime::moe_offload::MoeBackendCapabilities& GetBackendCapabilities() const noexcept override {
    return offload_capabilities_;
  }
  Status InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) override;

 private:
  // PrePack helpers - each handles one category of input tensor.
  void PrePackTransposeAndPack(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackComputeBias(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                          const IAllocatorUniquePtr<void>& packed_scale,
                          IAllocatorUniquePtr<void>& packed_bias, bool& is_packed);
  void PrePackCopyToGpu(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                        IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackSwizzleBlockScales(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                 IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackFp4ScalesForTmaWs(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  // ``gemv_interleaved`` selects the CUTLASS fpA_intB interleaved layout (steps 1-3) instead of
  // the plain [E, n, k/2] row-major ColToRow layout. ``sm80_pair_interleave`` additionally applies
  // the SM80 grouped-GEMM nibble pair-interleave (step 4, no +8 bias) on top of the interleaved
  // layout; it is meaningful only when ``gemv_interleaved`` is true. The decode GEMV consumes the
  // ColToRow (or steps-1-3) layout, so it must pass ``sm80_pair_interleave = false``; only the SM80
  // grouped-GEMM prefill buffer sets it true.
  void PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed,
                               bool gemv_interleaved = false, bool sm80_pair_interleave = false);
  // Builds the fused MXFP4 GEMV scale buffer for fc (1 or 2) once both the e8m0 block
  // scales (inputs 3/6) and the per-expert global scale (inputs 15/16) have been staged
  // to GPU. Order-independent: invoked from both PrePack handlers; the call that completes
  // the pair performs the combine. No-op unless the fused FP4 GEMV path is enabled.
  void TryBuildGemvFp4Scales(int fc, cudaStream_t stream, AllocatorPtr alloc);
  void TryBuildFp4DeepGemmWeights(int fc, cudaStream_t stream, AllocatorPtr alloc);
  // Prepacks int2/int4/int8 expert weights into the CUTLASS fpA_intB layout. INT4/INT8 are
  // consumed by the QMoE runner; INT2 is consumed by the decode GEMV while the raw initializer
  // remains live for dense prefill fallback. Mirrors what MatMulNBits.PrePack
  // does, looped over the E expert dimension. ``tensor`` is the 3-D
  // ``[E, N, K / (8 / bits)]`` weight initializer; ``packed_buf`` receives a
  // GPU buffer in the kernel-expected ``[E, K, N / (8 / bits)]`` layout.
  void PrePackIntExpertWeights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed, int64_t weight_bits);
  int64_t expert_weight_bits_;
  int64_t fc1_expert_weight_bits_;
  int64_t fc2_expert_weight_bits_;
  int64_t fc3_expert_weight_bits_;
  bool is_fp16_;
  // When true, the int4/int8 fc1/fc2 weight initializers are already in a
  // CUTLASS fpA_intB layout — produced offline e.g. via
  // ``pack_weights_for_cuda_mixed_gemm`` — and the compute path reads them
  // as-is. When false, the raw schema-conformant ``[E, N, K/pack]`` layout
  // (as produced by ``quantize_matmul_{4,8}bits``) is rewritten inside the
  // PrePack hook via ``PrePackIntExpertWeights``, removing the offline
  // prepack dependency. Only meaningful when ``quant_type_ == "int"``.
  // Derived from the optional tri-state ``weights_prepacked`` attribute:
  // -1 (default) and 1 both map to true; 0 maps to false. The concrete
  // prepacked layouts selected by -1 and 1 are determined by the execution
  // provider. For the CUDA EP the int4/int8 MoE GEMM always dispatches to the
  // Ampere (SM80) grouped-GEMM kernel -- even on SM90 -- because mixed
  // int-weight + fp16/bf16 activation is not a valid Hopper TMA warp-specialized
  // specialisation (matches TensorRT-LLM, which also routes W4A16/W8A16 MoE to
  // the SM80 kernel on Hopper). The kernel therefore consumes the SM80 fpA_intB
  // layout on every GPU, so -1 and 1 are currently equivalent for the CUDA EP;
  // 1 is reserved for a possible future Hopper-specific layout (e.g. W4A8).
  bool weights_prepacked_ = true;
  bool nvfp4_weights_row_major_ = false;
  bool nvfp4_fc1_decode_scale_eligible_ = false;
  bool nvfp4_fc2_decode_scale_eligible_ = false;
  // Cached source weight shapes captured at PrePack time. When the
  // PrePack hook consumed and released the original int4/int8 or MXFP4
  // weight initializers (``is_packed = true``), ``context->Input<Tensor>(2)``
  // and ``(5)`` return nothing, so ``moe_helper::CheckInputs`` can no
  // longer read the shapes from the live tensors. We feed it these
  // cached shapes instead via the ``TensorShape*`` overload, matching
  // how ``MatMulNBits`` caches ``N_`` / ``K_`` in its constructor.
  TensorShape fc1_weights_shape_;
  TensorShape fc2_weights_shape_;
  bool use_fp4_dequant_fallback_ = false;
  // Dequantizes FP8 weights to FP16/BF16 scratch buffers before invoking the A16 MoE runner.
  bool use_fp8_dequant_fallback_ = false;
  bool enable_fp8_fused_ = false;
  // WFP4AFP8 (W4A8) requires SM100+ (Blackwell) block-scaled tensor ops. On older GPUs we
  // dequantize MXFP4 weights to FP16/BF16 and run the dense A16 MoE runner.
  bool use_wfp4afp8_dequant_fallback_ = false;
  std::string quant_type_;  // "int", "fp4", "nvfp4", "fp8", or "wfp4afp8"
  bool enable_kernel_debug_info_ = false;
  bool enable_int2_gemv_ = false;
  bool enable_int2_prefill_ = false;
  int64_t row_tile_size_ = qmoe::kDisabledRowTileSize;
  int64_t int_dequant_max_scratch_bytes_ = int64_t{1} << 30;

  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> m_moe_runner;

  // Dense A16 (FP16/BF16) fallback runner, constructed only when the native FP4 CUTLASS path is
  // enabled. The native FP4 grouped GEMM wins when each expert processes few tokens, but the dense
  // grouped GEMM (FP4 weights dequantized to FP16/BF16) is faster once the per-expert token count
  // is large (compute-bound regime). ComputeInternal routes per call based on average tokens per
  // expert vs fp4_native_max_tokens_per_expert_. Null when native FP4 is not enabled.
  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface>
      m_fp4_dense_fallback_runner_;

  // Pre-packed buffers
  // Note: For QMoE, we need both Scales (for dequant) and Bias (derived from ZP/Scale) during inference.
  // PrePack logic:
  // - Copies scales to GPU buffer (if in CPU) or just keeps them. For simplicity, we allocate and copy.
  // - Computes Bias from ZP and Scale using PrePack kernel.
  // - For ``quant_type == "int"``, also prepacks the per-expert int4/int8
  //   weight tensors into the CUTLASS fpA_intB layout, mirroring
  //   ``MatMulNBits.PrePack_B``. Without this, callers would have to
  //   pre-prepack the weights offline using ``pack_weights_for_cuda_mixed_gemm``,
  //   which is asymmetric with how ``MatMulNBits`` is consumed and forces
  //   a CUDA-enabled ORT build for any offline quantization tooling.
  IAllocatorUniquePtr<void> packed_fc1_weights_;
  IAllocatorUniquePtr<void> packed_fc2_weights_;
  IAllocatorUniquePtr<void> packed_fc1_scales_;
  IAllocatorUniquePtr<void> packed_fc1_bias_;
  IAllocatorUniquePtr<void> packed_fc2_scales_;
  IAllocatorUniquePtr<void> packed_fc2_bias_;
  // Decode-only [E, K_blocks, N] scale caches for INT2/mixed-width packed GEMV.
  // The canonical initializer inputs remain live for validation and dense prefill fallback.
  IAllocatorUniquePtr<void> gemv_int_fc1_scales_;
  IAllocatorUniquePtr<void> gemv_int_fc2_scales_;

  // FP4 pre-packed buffers
  IAllocatorUniquePtr<void> packed_fp4_fc1_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc1_block_scales_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_block_scales_;

  // Fused MXFP4 GEMV (W4A16) decode path. Default-on (opt-out via ORT_ENABLE_FP4_GEMV=0) on
  // the dequant-fallback regime. When enabled, PrePack additionally lays out the MXFP4
  // weights in the GEMV-consumed [E, n, k/2] row-major layout and combines the e8m0 block
  // scales with the per-expert global scale into the
  // [E, k/32, n] activation-dtype scale layout. ComputeInternal routes small-decode shapes
  // through a standalone fused GEMV pipeline (prologue -> expand -> fc1 SwiGLU GEMV ->
  // fc2 GEMV -> finalize) instead of dequantizing to dense weights. Falls back to the
  // dequant path for unsupported shapes (prefill / large batch).
  bool enable_fp4_gemv_ = false;
  // Read once during op construction so ORT_DISABLE_FP4_GEMV_SKIP_EXPAND follows the same
  // session-scoped configuration model as the other FP4 GEMV environment options.
  bool fp4_gemv_skip_expand_ = true;
  bool nvfp4_gemv_raw_layout_ = false;
  bool enable_fp4_cutlass_gemm_ = false;
  bool enable_fp4_deep_gemm_ = false;
  // Per-rank expert count DeepGEMM was built for (32); 0 when the static shapes rule it out.
  int fp4_deep_gemm_num_experts_ = 0;
  // Native block-scaled CUTLASS FP4xFP4 grouped GEMM for NVFP4 (e2m1 weight + e2m1 activation,
  // block size 16, E4M3 block scales). Blackwell SM120+. When enabled, prefill routes through the
  // native FP4xFP4 runner (m_moe_runner) and decode/oversized shapes fall back to the fused GEMV
  // and the dense A16 runner respectively, mirroring the MXFP4 (enable_fp4_cutlass_gemm_) split.
  bool enable_nvfp4_cutlass_gemm_ = false;
  // Default-on (set ORT_FP4_SM80_GEMM=0 to disable) port of the INT4 SM80 fused-dequant grouped
  // GEMM to MXFP4 (wfp4a16: e2m1 weight + FP16/BF16 activation). On H200 the native SM90
  // TMA FP4 path is ~50x slower than vLLM at prefill; routing prefill through the Ampere/SM80
  // DqMma grouped GEMM (the same kernel INT4 uses) closes most of the gap. When set, PrePack lays the e2m1
  // weights out in the SM80 CUTLASS ColumnMajorTileInterleave layout (reusing the GEMV
  // interleaved-layout buffers) and ComputeInternal routes prefill through the FP4 runner with
  // QuantParams::GroupWise(32, ...) activation-dtype group scales. Because that layout is
  // incompatible with the decode GEMV kernel, PrePack also packs a separate ColToRow copy of
  // the e2m1 weights (gemv_fp4_fc*_weights_decode_) that the fused GEMV decode path consumes.
  bool enable_fp4_sm80_gemm_ = false;
  // When true, PrePack reports ``is_packed = true`` for the e2m1 weight initializers (inputs 2/5)
  // so ORT releases them. Safe only in the ``enable_fp4_sm80_gemm_`` regime, where prefill reads
  // gemv_fp4_fc*_weights_ and decode reads either that same buffer (pair-interleaved un-permute)
  // or gemv_fp4_fc*_weights_decode_, and the dequant fallback -- the only consumer of the raw
  // [E, K, N/2] layout -- is unreachable. For a 20B-class MXFP4 MoE the retained initializers are
  // ~9 GiB of otherwise dead device memory. NOTE: the memory is returned to the *device* only when
  // initializers bypass the BFC arena, i.e. with the session option
  // ``session.use_device_allocator_for_initializers = 1``; otherwise it is merely recycled inside
  // the arena for later activation/KV allocations.
  bool release_fp4_raw_weights_ = false;
  // Set by PrePack (per weight tensor) when the decode GEMV will read the SM80 pair-interleaved
  // buffer in gemv_fp4_fc*_weights_ instead of a dedicated gemv_fp4_fc*_weights_decode_ copy.
  // False when SM80 GEMM is off (gemv_fp4_fc*_weights_ is already GEMV-native) or when the shape
  // misses the interleaved rules.
  bool gemv_fp4_fc1_reads_sm80_layout_ = false;
  bool gemv_fp4_fc2_reads_sm80_layout_ = false;
  // When native CUTLASS WFP4A16 is enabled, GEMV is also pre-packed and used for decode shapes
  // (M < this threshold); prefill (M >= threshold) runs the native grouped GEMM. 0 disables the
  // split (pure GEMV regime). Overridable via ORT_FP4_PREFILL_MIN_TOKENS.
  int64_t fp4_prefill_min_tokens_ = 0;
  // Upper bound on average tokens per expert (num_rows * top_k / num_experts) for the native FP4
  // grouped GEMM. Above it, the prefill routes to the dense A16 fallback runner, which is faster in
  // the compute-bound regime (measured crossover ~128 tokens/expert on H200). 0 disables the upper
  // bound (always native for prefill). Overridable via ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT.
  int64_t fp4_native_max_tokens_per_expert_ = 0;
  // Per-shape autotune of the fused FP4 GEMV tiling. Both knobs are read once in the constructor
  // (default-off autotune and logging) rather than fresh on every inference call, so the
  // decision cannot change underneath a session whose runner/layout was already built — matching
  // the constructor-plumbed pattern used by ORT_FP4_SM80_GEMM. Overridable via
  // ORT_FP4_GEMV_AUTOTUNE / ORT_FP4_GEMV_AUTOTUNE_LOG.
  bool enable_fp4_gemv_autotune_ = false;
  bool enable_fp4_gemv_autotune_log_ = false;
  IAllocatorUniquePtr<void> gemv_fp4_fc1_weights_;  // [E, 2*inter, hidden/2] row-major e2m1
  IAllocatorUniquePtr<void> gemv_fp4_fc2_weights_;  // [E, hidden, inter/2] row-major e2m1
  // When enable_fp4_sm80_gemm_ repurposes gemv_fp4_fc*_weights_ for the SM80 grouped-GEMM
  // prefill (SM80 pair-interleaved layout), these hold a dedicated decode-GEMV copy of the e2m1
  // weights in the GEMV-native layout (ColToRow, or the interleaved layout's preprocessor steps
  // 1-3). Only allocated when gemv_fp4_fc*_reads_sm80_layout_ is false, i.e. when the GEMV cannot
  // un-permute the pair-interleaved buffer itself. Null when SM80 GEMM is disabled -- then the
  // decode GEMV reads gemv_fp4_fc*_weights_ directly.
  IAllocatorUniquePtr<void> gemv_fp4_fc1_weights_decode_;
  IAllocatorUniquePtr<void> gemv_fp4_fc2_weights_decode_;
  // Combined activation-dtype scales for prepacked GEMV. Raw-layout NVFP4 skips this bank.
  IAllocatorUniquePtr<void> gemv_fp4_fc1_scales_;  // [E, hidden/32, 2*inter] activation dtype
  IAllocatorUniquePtr<void> gemv_fp4_fc2_scales_;  // [E, inter/32, hidden] activation dtype
  // Raw [E, n, k_blocks] e8m0 block scales kept for GEMV when the native CUTLASS path has
  // already swizzled packed_fp4_*_block_scales_ into the TMA layout. Empty in the pure-GEMV
  // regime, where TryBuildGemvFp4Scales reads packed_fp4_*_block_scales_ directly.
  IAllocatorUniquePtr<void> gemv_fp4_fc1_block_raw_;
  IAllocatorUniquePtr<void> gemv_fp4_fc2_block_raw_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc1_staged_weights_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc2_staged_weights_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc1_staged_block_scales_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc2_staged_block_scales_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc1_weights_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc2_weights_;
  // fp32 per-[128 N, 128 K] block scales for the e4m3 weights above, [E, N/128, K/128].
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc1_weight_scales_;
  IAllocatorUniquePtr<void> fp4_deep_gemm_fc2_weight_scales_;
  // Block-scale dimensions captured at PrePack time so TryBuildGemvFp4Scales can size and
  // launch the combine kernel once the global scale also arrives. [E, n, k_blocks].
  int64_t gemv_fp4_fc1_scale_e_ = 0;
  int64_t gemv_fp4_fc1_scale_n_ = 0;
  int64_t gemv_fp4_fc1_scale_kb_ = 0;
  int64_t gemv_fp4_fc2_scale_e_ = 0;
  int64_t gemv_fp4_fc2_scale_n_ = 0;
  int64_t gemv_fp4_fc2_scale_kb_ = 0;
  // Global-scale lengths captured during PrePack. TryBuildGemvFp4Scales must not
  // launch until they match the staged block-scale expert dimension.
  int64_t fp4_fc1_global_scale_e_ = 0;
  int64_t fp4_fc2_global_scale_e_ = 0;

  // Per-expert global weight scales used by FP4 and FP8 modes.
  IAllocatorUniquePtr<void> packed_fc1_global_scale_;
  IAllocatorUniquePtr<void> packed_fc2_global_scale_;

  // Per-tensor or per-expert FP8 activation global scales used by W4A8 (WFP4AFP8) Variant A.
  // Inputs 17/18 in the QMoE schema. Optional; absent for the MXFP8 block-scaled variant.
  IAllocatorUniquePtr<void> packed_fc1_act_scale_;
  IAllocatorUniquePtr<void> packed_fc2_act_scale_;

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;

  struct Fp4GemvTuneKey {
    bool is_fp16 = false;
    int64_t row_bucket = 0;
    int64_t hidden = 0;
    int64_t inter = 0;
    int sm = 0;

    bool operator==(const Fp4GemvTuneKey& other) const {
      return is_fp16 == other.is_fp16 && row_bucket == other.row_bucket && hidden == other.hidden &&
             inter == other.inter && sm == other.sm;
    }
  };

  struct Fp4GemvTuneKeyHash {
    size_t operator()(const Fp4GemvTuneKey& key) const {
      size_t hash = 1469598103934665603ULL;
      auto combine = [&hash](auto value) {
        hash ^= static_cast<size_t>(value);
        hash *= 1099511628211ULL;
      };
      combine(key.is_fp16);
      combine(key.row_bucket);
      combine(key.hidden);
      combine(key.inter);
      combine(key.sm);
      return hash;
    }
  };

  struct Fp4GemvTuneResult {
    onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig fc1_config =
        onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig::kDefault;
    onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig fc2_config =
        onnxruntime::llm::kernels::moe_gemv::MoeGemvConfig::kDefault;
  };

  mutable std::mutex fp4_gemv_tune_cache_mutex_;
  mutable std::unordered_map<Fp4GemvTuneKey, Fp4GemvTuneResult, Fp4GemvTuneKeyHash> fp4_gemv_tune_cache_;

  // v4 MoE Offload state
  int32_t offload_layer_index_{-1};
  int offload_registered_dev_id_{-1};
  const uint8_t* offload_registered_fc1_w_{nullptr};
  bool offload_node_registered_{false};
  bool offload_prepack_bypass_{false};
  IAllocatorUniquePtr<void> offload_gpu_fc1_bias_;
  IAllocatorUniquePtr<void> offload_gpu_fc2_bias_;
  onnxruntime::moe_offload::MoeOffloadSessionPolicy offload_policy_{};
  onnxruntime::moe_offload::MoeExpertSourceDescriptor offload_source_desc_{};
  onnxruntime::moe_offload::MoeBackendCapabilities offload_capabilities_{};
  onnxruntime::moe_offload::MoeNodeExecutionPlan offload_plan_{};
  std::unique_ptr<QMoEOffload> offload_backend_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
