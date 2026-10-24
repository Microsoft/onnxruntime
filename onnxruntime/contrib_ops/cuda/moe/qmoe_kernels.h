// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSoftmaxTopK(
    const __nv_bfloat16* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

// Decode-only fused routing: softmax + top-k (bit-identical to LaunchSoftmaxTopK) and the (expert, token)-sorted expert
// maps of fusedBuildExpertMapsSortFirstToken in one single-CTA launch. Returns false (nothing launched) outside its
// envelope: 1..4 rows, exactly 512 experts, k in {1, 2, 4, 6, 8, 10}, 8-byte aligned logits.
bool LaunchFusedRouteMaps(const half* logits, float* topk_scales, int* topk_indices, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, int* permuted_expert, int64_t* expert_first_token_offset,
                          int num_rows, int num_experts, int k, bool normalize_scales, bool pdl, cudaStream_t stream);
bool LaunchFusedRouteMaps(const __nv_bfloat16* logits, float* topk_scales, int* topk_indices,
                          int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
                          int* permuted_expert, int64_t* expert_first_token_offset, int num_rows, int num_experts,
                          int k, bool normalize_scales, bool pdl, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadGather(
    const half* input, const int* route_ids, half* gathered,
    int route_count, int hidden_size, int k, cudaStream_t stream);
cudaError_t LaunchQMoEOffloadGather(
    const __nv_bfloat16* input, const int* route_ids, __nv_bfloat16* gathered,
    int route_count, int hidden_size, int k, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadScatter(
    const half* wave_output, const int* route_ids, half* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream);
cudaError_t LaunchQMoEOffloadScatter(
    const __nv_bfloat16* wave_output, const int* route_ids, __nv_bfloat16* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadMetadata(
    float* unit_scales, int* identity_permutation, int route_count, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadSourceRows(
    int* source_rows, int route_count, int k, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadScatterBias(
    const half* wave_bias, const int* compact_experts, const int* route_ids, half* route_bias,
    int route_count, int hidden_size, cudaStream_t stream);
cudaError_t LaunchQMoEOffloadScatterBias(
    const __nv_bfloat16* wave_bias, const int* compact_experts, const int* route_ids, __nv_bfloat16* route_bias,
    int route_count, int hidden_size, cudaStream_t stream);

cudaError_t LaunchQMoEOffloadClassify(
    const int* topk_indices, const int* expert_slot, int route_count, int row_count,
    int* hit_ids, int* hit_compact, int* hit_flag, cudaStream_t stream,
    int dummy_slot = 0, float* unit_scales = nullptr, int* identity_permutation = nullptr,
    int k_for_source_rows = 0, int* host_indices_out = nullptr,
    const void* input_dev = nullptr, void* host_input_out = nullptr, int input_vec4_count = 0,
    uint64_t* doorbell_seq_dev = nullptr, volatile uint64_t* doorbell_req_host = nullptr,
    volatile int32_t* doorbell_tile_rows_host = nullptr, int tile_rows = 0);

cudaError_t LaunchQMoEOffloadMerge(
    const half* wave_output, const half* cpu_rows, const int* hit_flag, half* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev = nullptr, const volatile uint64_t* doorbell_done_host = nullptr);
cudaError_t LaunchQMoEOffloadMerge(
    const __nv_bfloat16* wave_output, const __nv_bfloat16* cpu_rows, const int* hit_flag, __nv_bfloat16* contributions,
    int route_count, int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev = nullptr, const volatile uint64_t* doorbell_done_host = nullptr);

cudaError_t LaunchQMoEOffloadMergeFinalizeCombine(
    const __nv_bfloat16* wave_output, const __nv_bfloat16* cpu_rows, const int* hit_flag,
    const float* scales, __nv_bfloat16* routed_out, __nv_bfloat16* shared_out,
    int hidden_size, int tile_rows, int k, cudaStream_t stream,
    const uint64_t* doorbell_seq_dev = nullptr, const volatile uint64_t* doorbell_done_host = nullptr,
    int trigger = 0);

cudaError_t LaunchQMoESharedExpertCombine(
    __nv_bfloat16* shared_out, const __nv_bfloat16* routed_out,
    int count, int trigger, cudaStream_t stream);
void LaunchSparseMixerTop2(
    const float* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const half* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const __nv_bfloat16* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const float* scales,
    float* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const half* scales,
    half* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

// Batched 4-bit packed zero-point scaled bias computation.
// ZP layout: [experts, n, packed_k_blocks] (packed_k_blocks = (k_blocks+1)/2)
// Scale/Output layout: [experts, k_blocks, n]
// Computes: output[e][row][col] = scale[e][row][col] * (-unpack4(zp[e][col][row/2]) + default_zp)
void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const half* transposed_scale,
    half* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream);

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream);

void LaunchQMoEShiftWeights(
    const uint8_t* input,
    uint8_t* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const half* input,
    half* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoEBlockScaleInterleave(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded,
    int multi_processor_count,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

// Converts MXFP4 expert weights ([E, K, N/2] n-packed E2M1 nibbles, Float8E8M0 block scales of
// group size 32, per-expert fp32 global scale) into the e4m3 weights + fp32 per-[128 N, 128 K]
// block scales the QMoE DeepGEMM FP8 kernel consumes. ``output`` is [E, N, K] e4m3 bytes and
// ``output_scales`` is [E, N/128, K/128] fp32.
//
// The block scale retains the per-expert global scale and rounds the remaining factor up to a
// power of two, which makes the conversion bit-exact for E2M1 inputs. ``inexact_flag`` (one
// device int, zeroed by the caller) is set to 1 if any weight fails to round-trip, which can only
// happen if a block's MXFP4 group exponents span more than e4m3's range.
void LaunchQMoEQuantizeFp4WeightsToFp8(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    uint8_t* output,
    float* output_scales,
    int* inexact_flag,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

// Packs MXFP4 e8m0 block scales from [experts, n, k_blocks] into the SM90 TMA WS
// WFP4A16 layout. The currently dispatched native WFP4A16 K tile is 256, so one
// TMA scale element contains 8 adjacent k_blocks for one output row.
void LaunchQMoEPackFp4ScalesForTmaWs(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

// Keeps the original routing IDs intact. Unused compact_to_expert slots are -1.
void LaunchQMoECompactExperts(
    const int* expert_indices,
    int* compact_indices,
    int* expert_to_compact,
    int* compact_to_expert,
    int num_experts,
    int expert_capacity,
    int64_t num_routes,
    cudaStream_t stream,
    int pass_offset = 0);

void LaunchQMoEMarkActiveExperts(
    const int* expert_indices,
    int* expert_active_mask,
    int num_experts,
    int64_t num_routes,
    cudaStream_t stream);

void LaunchQMoERemapExpertRange(
    const int* expert_indices,
    int* compact_indices,
    int64_t num_routes,
    int pass_start,
    int expert_capacity,
    cudaStream_t stream);

void LaunchQMoEAccumulateExpertHistogram(
    const int* expert_indices,
    int64_t num_routes,
    int num_experts,
    int* global_hist,
    cudaStream_t stream);

void LaunchQMoEAccumulatePassOutput(
    float* accum,
    const half* pass_output,
    half* final_output,
    int64_t count,
    bool write_final,
    cudaStream_t stream);

void LaunchQMoEAccumulatePassOutput(
    float* accum,
    const __nv_bfloat16* pass_output,
    __nv_bfloat16* final_output,
    int64_t count,
    bool write_final,
    cudaStream_t stream);

struct QMoEFp8ProjectionParams {
  const uint8_t* weights = nullptr;
  const uint8_t* up_weights = nullptr;
  const void* scales = nullptr;
  const void* up_scales = nullptr;
  int scale_type = 0;
  int up_scale_type = 0;
  const int* row_to_unpermuted = nullptr;
  const int* experts = nullptr;
  const int64_t* expert_offsets = nullptr;
  const int* tile_offsets = nullptr;
  int num_experts = 0;
  int num_rows = 0;
  int expanded_rows = 0;
  int n = 0;
  int k = 0;
  int block_size = 0;
  int fusion = 0;
};

void LaunchQMoEFp8ExpertTiles(const int64_t* expert_offsets, int* tile_offsets,
                              int num_experts, cudaStream_t stream);

template <typename T, typename OutputT = T>
void LaunchQMoEFp8Projection(const QMoEFp8ProjectionParams& params,
                             const T* input, OutputT* output, cudaStream_t stream);

bool QMoEFp8GemvSupportsNativeDecode();
bool QMoEFp8GemvKernelUsesNativeDecode(const void* kernel);

template <typename T, typename InputT = T>
void LaunchQMoEFp8Activation(const InputT* input, const T* bias, T* output,
                             const int* experts, int rows, int inter_size, int fusion,
                             float alpha, float beta, float limit, cudaStream_t stream);

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    int block_size = 0,
    bool split_fc1 = false,
    bool block_fused_fc1 = false,
    const int* compact_to_expert = nullptr,
    const half* bias = nullptr,
    half* output_bias = nullptr);

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    int block_size = 0,
    bool split_fc1 = false,
    bool block_fused_fc1 = false,
    const int* compact_to_expert = nullptr,
    const __nv_bfloat16* bias = nullptr,
    __nv_bfloat16* output_bias = nullptr);

struct QMoEResidentLayerSlabView {
  const uint8_t* fc1_weights = nullptr;
  const uint8_t* fc1_scales = nullptr;
  const uint8_t* fc2_weights = nullptr;
  const uint8_t* fc2_scales = nullptr;
  const int* slot_table_dev = nullptr;
  const float* fc1_global_scales = nullptr;
  const float* fc2_global_scales = nullptr;
  const uint8_t* host_fc1_weights = nullptr;
  const uint8_t* host_fc1_scales = nullptr;
  const uint8_t* host_fc2_weights = nullptr;
  const uint8_t* host_fc2_scales = nullptr;
  size_t host_expert_stride = 0;
  const int* host_slot_table_dev = nullptr;
  bool all_experts_cached = false;
};

struct QMoEPrefillPassDmaView {
  void* dequant_fc1_weights = nullptr;
  void* dequant_fc2_weights = nullptr;
  const uint8_t* staging_fc1_weights = nullptr;
  const uint8_t* staging_fc1_scales = nullptr;
  const float* staging_fc1_global_scales = nullptr;
  const uint8_t* staging_fc2_weights = nullptr;
  const uint8_t* staging_fc2_scales = nullptr;
  const float* staging_fc2_global_scales = nullptr;
  const uint8_t* resident_fc1_weights = nullptr;
  const uint8_t* resident_fc1_scales = nullptr;
  const uint8_t* resident_fc2_weights = nullptr;
  const uint8_t* resident_fc2_scales = nullptr;
  const int* slot_table_dev = nullptr;
  const uint8_t* host_fc1_weights = nullptr;
  const uint8_t* host_fc1_scales = nullptr;
  const uint8_t* host_fc2_weights = nullptr;
  const uint8_t* host_fc2_scales = nullptr;
  size_t host_expert_stride = 0;
  const int* host_slot_table_dev = nullptr;
  int* prefill_layer_hist_dev = nullptr;
};

bool TryGetQMoEResidentLayerSlabView(
    int device_id,
    int64_t layer_index,
    int num_experts,
    int hidden_size,
    int inter_size,
    QMoEResidentLayerSlabView* out_view,
    cudaStream_t stream = nullptr,
    const uint8_t* fc1_w_id = nullptr);

void RegisterQMoEDenseLayerHostWeights(
    int device_id, int64_t layer_index, int num_experts, int hidden_size, int inter_size,
    const uint8_t* fc1_w, const uint8_t* fc1_s, const float* fc1_g,
    const uint8_t* fc2_w, const uint8_t* fc2_s, const float* fc2_g);

void UnregisterQMoEDenseLayerHostWeights(
    int device_id, int64_t layer_index, const uint8_t* fc1_w = nullptr);

// Exclusive per-device lease over the shared prefill DMA state (staging buffers, dense-layer table and
// workspace). Hold it from TryPrepareQMoEPrefillDmaLayer through the last pass of a layer so concurrent
// sessions cannot interleave their prepare/begin/dequant steps. Releasing the lease orders the next owner's
// work after everything this owner enqueued on its stream.
class QMoEPrefillDmaLease {
 public:
  QMoEPrefillDmaLease(int device_id, cudaStream_t exec_stream);
  ~QMoEPrefillDmaLease();
  QMoEPrefillDmaLease(const QMoEPrefillDmaLease&) = delete;
  QMoEPrefillDmaLease& operator=(const QMoEPrefillDmaLease&) = delete;

 private:
  int device_id_;
  cudaStream_t exec_stream_;
};

bool TryPrepareQMoEPrefillDmaLayer(
    int device_id, int64_t layer_index, int num_experts, int hidden_size, int inter_size,
    int runner_num_experts, int64_t tile_rows, size_t elem_size,
    const uint8_t* fc1_w, const uint8_t* fc1_s, const float* fc1_g,
    const uint8_t* fc2_w, const uint8_t* fc2_s, const float* fc2_g,
    cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view);

bool BeginQMoEPrefillDmaPass(
    int device_id, int64_t layer_index, int pass_idx, int num_passes, int runner_num_experts,
    cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view,
    const int* expert_active_mask_dev = nullptr);

void PrefetchNextQMoEPrefillDmaPass(
    int device_id, int64_t layer_index, int pass_idx, int num_passes, int runner_num_experts,
    cudaStream_t exec_stream = nullptr);

void EndQMoEPrefillDmaPassDequant(
    int device_id, int pass_idx, cudaStream_t exec_stream);

void OnQMoEDeviceArenasShrunk(int device_id) noexcept;
void FinalizeQMoEPrefillAndSeedDecode(int device_id, cudaStream_t stream = nullptr);

// NVFP4 weight dequantization: E2M1 4-bit weights with Float8E4M3FN block scales
// (block size 16) and per-expert float32 global scales. Weight layout [E, K, N/2],
// block-scale layout [E, N, K/16]. Mirrors LaunchQMoEDequantizeFp4Weights (MXFP4).
void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    const int* compact_to_expert = nullptr,
    const half* bias = nullptr,
    half* output_bias = nullptr,
    bool weights_row_major = false,
    const uint8_t* resident_packed_weights = nullptr,
    const uint8_t* resident_block_scales = nullptr,
    const int* resident_expert_to_slot = nullptr,
    int pass_start_expert = -1,
    const int* expert_active_mask = nullptr,
    const uint8_t* host_packed_weights = nullptr,
    const uint8_t* host_block_scales = nullptr,
    size_t host_expert_stride = 0,
    const int* host_expert_to_slot = nullptr);

void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream,
    const int* compact_to_expert = nullptr,
    const __nv_bfloat16* bias = nullptr,
    __nv_bfloat16* output_bias = nullptr,
    bool weights_row_major = false,
    const uint8_t* resident_packed_weights = nullptr,
    const uint8_t* resident_block_scales = nullptr,
    const int* resident_expert_to_slot = nullptr,
    int pass_start_expert = -1,
    const int* expert_active_mask = nullptr,
    const uint8_t* host_packed_weights = nullptr,
    const uint8_t* host_block_scales = nullptr,
    size_t host_expert_stride = 0,
    const int* host_expert_to_slot = nullptr);

// Repack column-major FP4 packed weights to row-major layout on GPU.
// Input shape interpretation: [experts, k, n/2] (col-major packed),
// output: [experts, n, k/2] (row-major packed).
// Each byte holds two 4-bit values. k and n must be even.
void LaunchQMoERepackFP4ColToRow(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int64_t k,
    int64_t n,
    cudaStream_t stream);

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

namespace onnxruntime::llm::kernels {
void LaunchBatchedTranspose(cudaStream_t stream, const void* input, void* output, int batch, int rows, int cols, int element_size);
}
