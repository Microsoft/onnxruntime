#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/inlined_containers.h"
#include "core/framework/moe_offload_plan.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
#include "contrib_ops/cuda/moe/qmoe_callback_timing.h"

namespace onnxruntime::contrib::cuda {

class QMoE;
struct QMoEResidentLayerSlabView;
struct QMoEPrefillPassDmaView;

class QMoEOffload final : public onnxruntime::cuda::CudaKernel, public MoEBase {
 public:
  explicit QMoEOffload(const OpKernelInfo& info);
  QMoEOffload(const OpKernelInfo& info,
              const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
              const onnxruntime::moe_offload::MoeExpertSourceDescriptor& desc,
              const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan);
  ~QMoEOffload() override;
  Status ComputeInternal(OpKernelContext* context) const override;
  void SyncPrefillSlotTables(cudaStream_t stream = nullptr) const;
  struct SharedLayerCache;
  friend bool TryGetQMoEResidentLayerSlabView(int device_id, int64_t layer_index, int num_experts,
                                              int hidden_size, int inter_size, QMoEResidentLayerSlabView* out_view,
                                              cudaStream_t stream, const uint8_t* fc1_w_id);
  friend bool TryPrepareQMoEPrefillDmaLayer(
      int device_id, int64_t layer_index, int num_experts, int hidden_size, int inter_size,
      int runner_num_experts, int64_t tile_rows, size_t elem_size,
      const uint8_t* fc1_w, const uint8_t* fc1_s, const float* fc1_g,
      const uint8_t* fc2_w, const uint8_t* fc2_s, const float* fc2_g,
      cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view);
  friend bool BeginQMoEPrefillDmaPass(
      int device_id, int64_t layer_index, int pass_idx, int num_passes, int runner_num_experts,
      cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view,
      const int* expert_active_mask_dev);
  friend void PrefetchNextQMoEPrefillDmaPass(
      int device_id, int64_t layer_index, int pass_idx, int num_passes, int runner_num_experts,
      cudaStream_t exec_stream);
  friend void FinalizeQMoEPrefillAndSeedDecode(int device_id, cudaStream_t stream);
  friend void UnregisterQMoEDenseLayerHostWeights(int device_id, int64_t layer_index, const uint8_t* fc1_w);

 private:
  enum Buffer : size_t {
    Fc1Weights,
    Fc2Weights,
    Fc1Scales,
    Fc2Scales,
    Fc1Bias,
    Fc2Bias,
    RouteBias,
    RouterScales,
    RouterIndices,
    UnitScales,
    IdentityPermutation,
    RouteIds,
    CompactExperts,
    WavePermutation,
    GatheredInput,
    WaveOutput,
    Contributions,
    RunnerWorkspace,
    BufferCount
  };

  using PinnedBuffer = decltype(std::declval<const onnxruntime::cuda::CudaKernel&>()
                                    .AllocateBufferOnCPUPinned<uint8_t>(0));
  using ResidentSlab = decltype(std::declval<const onnxruntime::cuda::CudaKernel&>()
                                    .GetScratchBuffer<uint8_t>(0, nullptr));

  ResidentSlab AllocateReservedDeviceSlab(size_t bytes) const {
    return ::onnxruntime::IAllocator::MakeUniquePtr<uint8_t>(
        Info().GetAllocator(OrtMemType::OrtMemTypeDefault), bytes, /*use_reserve*/ true);
  }
  size_t LendResidentSlab() const;
  size_t ReclaimDecodeExtension() const;
  void RestoreDecodeExtensionConfig() const;
  void ReleasePersistentDecodePinned() const noexcept;
  void CopyExpertToResidentSlot(int expert, int slot, cudaStream_t stream) const;
  void BatchCopyExpertsToResidentSlots(const int* experts, const int* slots, int count, cudaStream_t stream) const;
  void RestoreAndSeedResidentSlab(const int* prompt_hist, int total_prompt_routes,
                                  bool enable_seeding, cudaStream_t stream) const;
  void StageExperts(const int* experts, int count) const;
  void InitializeShared(const OpKernelInfo& info);
  void InitializeNvfp4Offload(const OpKernelInfo& info);
  Status ComputeShared(OpKernelContext* context) const;
  Status PromoteResident(cudaStream_t stream, const InlinedVector<InlinedVector<int>>& cold_routes,
                         int max_promotions = 0, const int* streamed_experts = nullptr,
                         int streamed_count = 0) const;
  void RecordResidentRoute(int phase, int expert, bool hit, float weight = 1.0f) const;
  void CommitInvocationObservation(const std::array<int, 5>& phase_rows,
                                   const std::array<std::vector<int>, 5>& phase_expert_rows,
                                   const uint8_t* invocation_expert_rows = nullptr) const;

  struct CpuWork;
  struct AsyncTile;
  struct PersistentDecodeState;
  class DoorbellService;
  friend class DoorbellService;
  void InitializeComplementMode();
  void InitializeSsdStore();
  void InitializeBoundedHostCache();
  // Reuses the host cache slot of a just-promoted expert to hold the GPU victim copied back from the slab,
  // avoiding a storage re-read of the victim. Returns false if the victim must be fetched from storage instead.
  bool SwapHostCandidateForVictim(int candidate, int evicted, int victim_slab_slot, cudaStream_t stream) const;
  Status EnsurePersistentDecodeState(OpKernelContext* context, size_t min_gpu_bytes,
                                     size_t min_host_bytes, cudaStream_t stream) const;
  Status StepBetweenRuns(cudaStream_t stream) const;
  void ServiceDoorbell() const;
  struct CrossLayerPromoRequest {
    const QMoEOffload* node = nullptr;
    int device_id = 0;
    int candidate = -1;
    int target_slot = -1;
    int victim_slot = -1;
    int evicted_expert = -1;
  };
  bool TryPrepareCrossLayerPromotion(CrossLayerPromoRequest* out_req) const noexcept;
  bool ExecuteCrossLayerPromotionDma(const CrossLayerPromoRequest& req, uint8_t* pinned_bounce_buf,
                                     size_t bounce_cap) const noexcept;
  bool TryCommitCrossLayerPromotion(const CrossLayerPromoRequest& req) const noexcept;
  static int LaunchDeferredMergeStatic(void* ctx, cudaStream_t stream, void* shared_out_ptr, int trigger);
  int LaunchDeferredMerge(cudaStream_t stream, void* shared_out_ptr, int trigger) const;
  template <typename ActT, typename ParallelRunner>
  int RunColdExperts(CpuWork& work, int group_calls, const ActT* host_input, int tile_rows, ActT* host_result,
                     int* host_cpu_ids, bool compact, ParallelRunner&& run_parallel,
                     const char* skip_experts = nullptr) const;
  Status ComputeSharedAsync(OpKernelContext* context, const Tensor* input, const Tensor* router, const Tensor* bias,
                            const Tensor* phase, Tensor* output, int32_t* route_rows, int32_t* route_calls,
                            int32_t* counts, int tile_rows) const;
  void RunAsyncCpu(AsyncTile& tile) const;

  bool shared_mode_ = false;
  bool v3_compact_shared_ = false;
  bool v4_planned_mode_ = false;
  onnxruntime::moe_offload::MoeOffloadSessionPolicy v4_policy_{};
  onnxruntime::moe_offload::MoeNodeExecutionPlan v4_plan_{};
  bool nvfp4_mode_ = false;
  bool mxfp4_mode_ = false;
  bool is_bf16_ = false;
  bool nvfp4_fc1_decode_scale_eligible_ = false;
  bool nvfp4_fc2_decode_scale_eligible_ = false;
  const Tensor* expert_blocks_ = nullptr;
  // NVFP4 / MXFP4 multi-bank CPU tensors (when nvfp4_mode_ && !expert_blocks_)
  const Tensor* nvfp4_fc1_w_ = nullptr;
  const Tensor* nvfp4_fc1_s_ = nullptr;
  const Tensor* nvfp4_fc1_g_ = nullptr;
  const Tensor* nvfp4_fc1_b_ = nullptr;
  const Tensor* nvfp4_fc2_w_ = nullptr;
  const Tensor* nvfp4_fc2_s_ = nullptr;
  const Tensor* nvfp4_fc2_g_ = nullptr;
  const Tensor* nvfp4_fc2_b_ = nullptr;
  std::vector<float> default_global_scales_;
  ResidentSlab fc2_bias_gpu_;
  size_t fc2_bias_gpu_bytes_ = 0;
  mutable ResidentSlab resident_fc1_bias_slab_;
  mutable size_t resident_fc1_bias_bytes_ = 0;
  mutable std::array<ResidentSlab, 2> stream_fc1_bias_slab_{};
  int device_id_ = 0;
  mutable std::atomic<uint64_t> cpu_expert_invocations_{0};

  onnxruntime::contrib::qmoe_shared::ExpertBlockLayout shared_layout_;
  onnxruntime::contrib::qmoe_shared::Nvfp4ExpertBlockLayout nvfp4_layout_;
  onnxruntime::contrib::qmoe_shared::ExpertBlockLease expert_lease_;
  onnxruntime::contrib::qmoe_shared::TensorExpertStore expert_store_;
  mutable onnxruntime::contrib::qmoe_shared::FileExpertStore file_expert_store_;
  int ssd_cache_slots_ = 0;
  size_t host_cache_bytes_ = 0;
  std::string ssd_store_path_;
  std::string ssd_temp_file_;
  mutable uint64_t ssd_prefetch_gen_ = 0;
  bool host_complement_mode_ = false;
  int complement_slots_ = 0;
  size_t complement_stride_ = 0;
  mutable PinnedBuffer complement_host_slab_;
  mutable InlinedVector<int> expert_host_slot_;
  mutable InlinedVector<int> host_slot_expert_;
  mutable uint64_t complement_eviction_writebacks_ = 0;
  int64_t layer_index_ = 0;
  size_t shared_gpu_limit_ = 0;
  size_t shared_pinned_limit_ = 0;
  size_t stream_pinned_limit_ = 0;
  int group_calls_ = 1;
  bool deterministic_replay_ = false;
  bool allow_cuda_graph_ = false;
  bool doorbell_enabled_ = false;
  bool overlap_shared_expert_ = true;
  bool fuse_merge_finalize_ = true;
  std::string output0_name_;
  mutable std::unique_ptr<PersistentDecodeState> decode_state_;

  // Decoupled dispatch controls (v3 Section 8.1 / F7).
  int max_sync_tile_tokens_ = 32;
  int async_max_tokens_ = 32;
  int stream_min_tokens_ = 33;
  int prefill_chunk_tokens_ = 0;

  bool async_ = false;
  bool experts_registered_ = false;
  bool all_expert_banks_registered_ = false;
  mutable bool promo_budget_retained_ = false;
  std::vector<const void*> registered_host_ptrs_;
  bool async_aux_ = false;
  mutable cudaStream_t aux_stream_ = nullptr;
  mutable cudaEvent_t fork_event_ = nullptr;
  mutable cudaEvent_t join_event_ = nullptr;
  mutable cudaEvent_t final_device_done_ = nullptr;
  mutable std::atomic<bool> final_device_done_recorded_{false};
  mutable cudaEvent_t slot_reader_done_ = nullptr;
  mutable bool slot_reader_done_recorded_ = false;
  mutable cudaStream_t promo_stream_ = nullptr;
  mutable cudaEvent_t promo_ready_ = nullptr;
  struct PendingPromotion {
    int candidate = -1;
    int target_slot = -1;
    int victim_slot = -1;
    int evicted_expert = -1;
    int host_evict_slot = -1;
    int host_cand_slot = -1;
  };
  mutable std::vector<PendingPromotion> pending_promotions_;
  mutable bool slot_table_dirty_ = true;
  mutable bool host_slot_table_dirty_ = true;
  bool host_cache_mapped_ = false;
  bool swap_victim_to_host_cache_ = true;
  mutable uint64_t victim_swaps_ = 0;
  mutable uint64_t victim_swap_fallbacks_ = 0;
  mutable InlinedVector<int32_t> file_store_expert_slot_;
  int cpu_threads_ = 0;
  int pool_task_batch_ = 1;
  bool unified_cpu_service_ = false;
  std::unique_ptr<CpuWork> async_work_;
  mutable std::atomic<bool> async_in_flight_{false};
  mutable std::atomic<bool> async_failed_{false};
  mutable std::string async_error_message_;
  std::unique_ptr<qmoe::CallbackTiming> callback_timing_;
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> session_controller_;
  std::shared_ptr<SharedLayerCache> owned_layer_cache_;
  std::unique_ptr<QMoE> prefill_qmoe_;

  // Dynamic per-layer GPU expert cache with in-slab spare and dummy slots: P = C + S + D (v3 Section 7.4/7.5 / F3).
  mutable int resident_slots_ = 0;             // C: logical resident capacity (including decode_extension_slots_)
  int base_resident_slots_ = 0;                // Base static resident capacity from model attributes
  int configured_decode_extension_slots_ = 0;  // Configured extension slots for <= 8k decode
  mutable int decode_extension_slots_ = 0;     // Additional reclaimable decode resident slots from prefill headroom
  int spare_slots_ = 0;                        // S: spare physical slots for promotions
  int dummy_slots_ = 0;                        // D: 1 if CutlassMoeFCRunner async needs an immutable dummy slot, 0 for NVFP4 GEMV
  mutable int physical_slots_ = 0;             // P = C + S + D
  int dummy_slot_index_ = -1;                  // C + S when D == 1, -1 when D == 0
  int resident_promotions_ = 0;
  int resident_decay_routes_ = 0;
  int resident_margin_percent_ = 100;
  float marginal_row_weight_ = 0.80f;
  float presence_ema_weight_ = 2.0f;
  int step_promo_budget_ = 16;
  size_t step_promo_bytes_ = 64ULL << 20;
  bool resident_phase_scores_ = false;
  bool continuous_decay_ = false;
  bool bg_promo_stream_ = true;
  mutable bool resident_decode_active_ = false;
  std::array<size_t, 6> resident_bank_bytes_{};
  mutable std::array<size_t, 6> resident_bank_offsets_{};
  mutable size_t resident_slab_bytes_ = 0;
  std::vector<int> initial_experts_;
  std::vector<int> ranked_experts_;
  size_t resident_workspace_bytes_ = 0;
  size_t stream_workspace_bytes_ = 0;
  mutable ResidentSlab resident_slab_;
  mutable const uint8_t* last_slab_ptr_ = nullptr;
  ResidentSlab slot_table_dev_;
  int* CurrentSlotTableDev() const noexcept {
    if (!slot_table_dev_) return nullptr;
    return reinterpret_cast<int*>(slot_table_dev_.get());
  }
  mutable InlinedVector<int> slot_expert_;
  mutable InlinedVector<int> expert_slot_;
  mutable InlinedVector<float> expert_score_;
  mutable InlinedVector<float> decode_score_;
  mutable InlinedVector<float> expert_presence_ema_;
  mutable InlinedVector<InlinedVector<int>> last_cold_routes_;
  mutable bool sync_routes_pending_ = false;
  mutable std::array<int, 2> phase_routes_since_decay_{};
  mutable std::array<uint64_t, 5> phase_hit_routes_{};
  mutable std::array<uint64_t, 5> phase_total_routes_{};
  mutable int routes_since_decay_ = 0;
  mutable uint64_t resident_hit_routes_ = 0;
  mutable uint64_t resident_total_routes_ = 0;
  mutable uint64_t resident_promoted_ = 0;
  mutable uint64_t placement_epoch_ = 0;
  bool resident_log_ = false;

  // Prefill streaming via session-owned landing ring (v3 Section 7.6 / F5).
  int stream_rows_ = 0;
  int stream_min_rows_ = 16;
  int stream_wave_ = 32;
  int stream_promotions_ = 16;
  bool stream_promotion_reuse_ = false;
  size_t stream_gpu_limit_ = 0;
  std::array<std::array<size_t, 6>, 2> stream_bank_offsets_{};
  size_t stream_landing_bytes_ = 0;
  std::shared_ptr<onnxruntime::cuda::QMoESessionLandingRing> stream_shared_;
  mutable uint64_t stream_tiles_ = 0;
  mutable uint64_t stream_experts_ = 0;
  mutable uint64_t stream_routes_ = 0;
  mutable uint64_t stream_waves_ = 0;
  mutable uint64_t stream_cpu_routes_ = 0;
  mutable uint64_t stream_reused_promotions_ = 0;

  bool cpu_timing_ = false;
  bool cpu_fp16_boundaries_ = false;
  int fc1_cols_ = 64;
  int fc2_cols_ = 128;
  mutable uint64_t timing_calls_ = 0;
  mutable uint64_t timing_routes_ = 0;
  mutable uint64_t timing_cold_routes_ = 0;
  mutable uint64_t timing_cold_experts_ = 0;
  mutable uint64_t timing_cpu_ns_ = 0;
  mutable uint64_t timing_classify_ns_ = 0;
  mutable uint64_t timing_prep_ns_ = 0;
  mutable uint64_t timing_fc1_ns_ = 0;
  mutable uint64_t timing_fc2_ns_ = 0;
  mutable uint64_t timing_write_ns_ = 0;

  std::array<const Tensor*, 6> banks_{};
  std::array<size_t, 6> expert_bytes_{};
  std::array<size_t, 6> staging_offsets_{};
  std::array<size_t, BufferCount> offsets_{};
  int num_experts_ = 0;
  int hidden_size_ = 0;
  int inter_size_ = 0;
  int token_tile_size_ = 0;
  int expert_wave_size_ = 0;
  int max_routes_ = 0;
  int max_wave_routes_ = 0;
  int staging_optimization_ = 0;
  size_t scratch_bytes_ = 0;
  size_t expert_slab_bytes_ = 0;
  size_t workspace_bytes_ = 0;
  size_t host_indices_offset_ = 0;
  size_t host_routes_offset_ = 0;
  size_t host_compact_offset_ = 0;
  PinnedBuffer pinned_ring_;
  mutable InlinedVector<int> expert_to_slot_;
  mutable InlinedVector<int> wave_experts_;
  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> runner_;
  mutable std::mutex mutex_;
  mutable bool stream_failed_ = false;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QMoEOffload);
};

}  // namespace onnxruntime::contrib::cuda
