// SharedExpertBlockV1 and NVFP4 RowMajor CPU/GPU shared expert layouts and kernels.
// See dev/docs/qmoe_offload_design_v3.md (Sections 5, 6, 8, and F12).
// This header is intentionally independent of ORT internal headers so it can be unit-tested standalone.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {

constexpr int64_t kBlockSize = 32;          // K values per FP16 scale (SharedExpertBlockV1 INT4)
constexpr int64_t kNvfp4BlockSize = 16;     // K values per Float8E4M3FN scale (NVFP4)
constexpr int64_t kMxfp4BlockSize = 32;     // K values per Float8E8M0 scale (MXFP4)
constexpr int64_t kColumnsInterleaved = 4;  // ColumnMajorTileInterleave<64,4>
constexpr int64_t kRowsPerTile = 64;        // K values per interleave tile
constexpr size_t kBankAlignment = 256;
constexpr int kMaxRowsPerCall = 8;

constexpr int64_t kRequiredWeightsPrepacked = 1;
constexpr const char* kSharedLayoutId = "ort.qmoe.w4a16.block32.shared.v1";
constexpr const char* kNvfp4RowMajorLayoutId = "ort.qmoe.nvfp4.block16.rowmajor.v1";
constexpr const char* kNvfp4SharedLayoutId = "ort.qmoe.nvfp4.block16.shared.v1";
constexpr const char* kMxfp4RowMajorLayoutId = "ort.qmoe.mxfp4.block32.rowmajor.v1";
constexpr const char* kMxfp4SharedLayoutId = "ort.qmoe.mxfp4.block32.shared.v1";

bool IsAcceptedPrepackContract(int64_t weights_prepacked, int64_t block_size);
bool IsAcceptedNvfp4PrepackContract(int64_t weights_prepacked, int64_t block_size);
bool IsAcceptedMxfp4PrepackContract(int64_t weights_prepacked, int64_t block_size);

// Standalone checked arithmetic helpers (F12).
inline bool CheckedMulSize(size_t a, size_t b, size_t* out) noexcept {
  if (out == nullptr) return false;
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *out = a * b;
  return true;
}

inline bool CheckedAddSize(size_t a, size_t b, size_t* out) noexcept {
  if (out == nullptr) return false;
  if (b > std::numeric_limits<size_t>::max() - a) return false;
  *out = a + b;
  return true;
}

inline bool CheckedAlignUp(size_t v, size_t alignment, size_t* out) noexcept {
  if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
  const size_t mask = alignment - 1;
  if (v > std::numeric_limits<size_t>::max() - mask) return false;
  *out = (v + mask) & ~mask;
  return true;
}

// Validates that E * stride and external [offset, offset + length) cannot overflow (F12).
bool ValidateExpertStorageRange(int64_t num_experts, size_t stride, uint64_t external_offset,
                                uint64_t external_length, size_t* total_bytes = nullptr) noexcept;

// Returns true only when [data, data + bytes) is a verified read-only regular-file mapping
// whose pages can be safely discarded and reloaded from the underlying file.
bool IsFileBackedReadOnlyMapping(const void* data, size_t bytes) noexcept;

// Evicts whole OS pages inside [data, data + bytes) only when the range is a verified
// read-only regular-file mapping; returns the number of evicted bytes (0 for heap/anonymous memory).
size_t EvictFileBackedHostPages(const void* data, size_t bytes) noexcept;

// Lease descriptor recording actual pointer alignment alongside length and optional owner (F12 & Section 6).
struct ExpertBlockLease {
  const uint8_t* data = nullptr;
  size_t length = 0;
  size_t alignment = 0;
  uint64_t content_id = 0;
};

struct ExpertKey {
  uint64_t model_content_id = 0;
  int64_t layer_index = 0;
  int32_t expert_id = 0;
  int64_t layout_version = 1;
  uint64_t weight_revision = 0;
};

// Zero-copy tensor-backed ExpertStore seam (v3 Section 6).
class TensorExpertStore {
 public:
  TensorExpertStore() = default;
  bool Initialize(const void* base_ptr, size_t total_bytes, int64_t num_experts, size_t expert_stride,
                  size_t min_required_alignment = 2, uint64_t model_content_id = 0) noexcept;

  bool AcquireReady(int32_t expert_id, ExpertBlockLease* out_lease) const noexcept;
  bool AcquireReady(const ExpertKey& key, ExpertBlockLease* out_lease) const noexcept {
    if (key.model_content_id != 0 && model_content_id_ != 0 && key.model_content_id != model_content_id_) {
      return false;
    }
    return AcquireReady(key.expert_id, out_lease);
  }
  bool Prefetch(const int32_t* expert_ids, size_t count, uint64_t generation) const noexcept;

  const ExpertBlockLease& FullStoreLease() const noexcept { return store_lease_; }
  int64_t NumExperts() const noexcept { return num_experts_; }
  size_t ExpertStride() const noexcept { return expert_stride_; }
  uint64_t ContentId() const noexcept { return model_content_id_; }

 private:
  ExpertBlockLease store_lease_{};
  int64_t num_experts_ = 0;
  size_t expert_stride_ = 0;
  size_t min_alignment_ = 2;
  uint64_t model_content_id_ = 0;
};

// Asynchronous SSD-backed ExpertStore with bounded host RAM LRU cache, pin-counted leases,
// and background generation-tagged prefetch worker (v3 Section 6, Section 10 & P3).
class FileExpertStore {
 public:
  struct MappedBank {
    const void* data = nullptr;
    size_t expert_bytes = 0;
    size_t block_offset = 0;
    const char* file_path = nullptr;
    uint64_t file_offset = 0;
  };
  FileExpertStore() = default;
  ~FileExpertStore() noexcept;
  FileExpertStore(const FileExpertStore&) = delete;
  FileExpertStore& operator=(const FileExpertStore&) = delete;

  bool Initialize(const char* file_path, uint64_t file_offset, int64_t num_experts, size_t expert_stride,
                  size_t cache_slots, size_t min_required_alignment = 256,
                  uint64_t model_content_id = 0) noexcept;
  void Shutdown() noexcept;

  // Linux file-backed banks, read directly into the cache without repacking a second weight file.
  // Banks must remain immutable during initialization. Without an explicit file path, read-only
  // mappings are discovered via /proc/self/maps; heap banks up to 4 MiB are copied into owned backing.
  bool InitializeMapped(const MappedBank* banks, size_t bank_count, int64_t num_experts,
                        size_t expert_stride, size_t cache_slots,
                        uint64_t model_content_id = 0) noexcept;
  bool Evict(int32_t expert_id) noexcept;
  // Reassigns the Ready, unpinned cache slot of `old_id` to `new_id` without touching storage. Returns the slot
  // payload for the caller to fill (the slot is Loading, so readers of `new_id` wait), or nullptr if the slot
  // cannot be taken. Every successful call must be followed by FinishReplace(new_id, ok).
  uint8_t* BeginReplace(int32_t old_id, int32_t new_id) noexcept;
  void FinishReplace(int32_t new_id, bool ok) noexcept;
  bool TryAcquireCached(int32_t expert_id, ExpertBlockLease* out_lease) noexcept;
  bool ReadExpertDirect(int32_t expert_id, uint8_t* dst) noexcept;
  bool TryAcquireGroupCached(const int32_t* expert_ids, size_t count, ExpertBlockLease* out_leases) noexcept;
  bool AcquireGroupReady(const int32_t* expert_ids, size_t count, ExpertBlockLease* out_leases) noexcept;
  void ReleaseGroupLeases(const int32_t* expert_ids, size_t count) noexcept;
  void SnapshotExpertToSlot(int32_t* out_slots, int32_t max_experts) const noexcept;

  bool AcquireReady(int32_t expert_id, ExpertBlockLease* out_lease) noexcept;
  bool AcquireReady(const ExpertKey& key, ExpertBlockLease* out_lease) noexcept {
    if (key.model_content_id != 0 && model_content_id_ != 0 && key.model_content_id != model_content_id_) {
      return false;
    }
    return AcquireReady(key.expert_id, out_lease);
  }
  void ReleaseLease(int32_t expert_id) noexcept;
  bool Prefetch(const int32_t* expert_ids, size_t count, uint64_t generation) noexcept;
  void WaitPrefetchIdle() noexcept;

  bool IsInitialized() const noexcept { return initialized_; }
  int64_t NumExperts() const noexcept { return num_experts_; }
  size_t ExpertStride() const noexcept { return expert_stride_; }
  size_t CacheSlots() const noexcept { return cache_slots_; }
  uint8_t* CacheBase() const noexcept;
  size_t CacheBytes() const noexcept { return cache_slots_ * expert_stride_; }
  size_t ResidentBackingBytes() const noexcept;
  bool IsBankFileBacked(size_t bank_index) const noexcept;
  bool AreAllBanksFileBacked() const noexcept;
  uint64_t ContentId() const noexcept { return model_content_id_; }
  uint64_t CacheHits() const noexcept;
  uint64_t PrefetchHits() const noexcept;
  uint64_t SsdMisses() const noexcept;
  uint64_t BytesReadFromSsd() const noexcept;
  uint64_t BytesCopiedFromMemory() const noexcept;
  uint64_t ReadNanoseconds() const noexcept;
  void ResetStats() noexcept;

 private:
  bool InitializeStorage(const char* file_path, uint64_t file_offset, int64_t num_experts, size_t expert_stride,
                         size_t cache_slots, size_t min_required_alignment, uint64_t model_content_id,
                         const MappedBank* banks, size_t bank_count) noexcept;
  struct Impl;
  Impl* impl_ = nullptr;
  bool initialized_ = false;
  int64_t num_experts_ = 0;
  size_t expert_stride_ = 0;
  size_t cache_slots_ = 0;
  uint64_t model_content_id_ = 0;
};

size_t ComputePointerAlignment(const void* ptr) noexcept;
bool MakeExpertBlockLease(const void* ptr, size_t length, size_t min_required_alignment,
                          ExpertBlockLease* lease) noexcept;

// Byte offsets of the six banks inside one SharedExpertBlockV1 expert block.
// FC1 is the fused interleaved gate/up projection [2I,H]; FC2 is [H,I].
// Scales are FP16 block-major [K/32,N]; biases are FP16 [N].
struct ExpertBlockLayout {
  int64_t hidden = 0;
  int64_t inter = 0;
  size_t fc1_weights = 0;
  size_t fc1_scales = 0;
  size_t fc1_bias = 0;
  size_t fc2_weights = 0;
  size_t fc2_scales = 0;
  size_t fc2_bias = 0;
  size_t stride = 0;
};

// Returns false if the shape cannot use the layout or any intermediate product overflows (F12).
bool MakeExpertBlockLayout(int64_t hidden, int64_t inter, ExpertBlockLayout* layout);

// Byte offsets inside a contiguous NVFP4 row-major expert block (optional packed block mode).
struct Nvfp4ExpertBlockLayout {
  int64_t hidden = 0;
  int64_t inter = 0;
  size_t fc1_weights = 0;       // [2I, H/2] uint8 E2M1 row-major
  size_t fc1_scales = 0;        // [2I, H/16] uint8 Float8E4M3FN row-major
  size_t fc1_global_scale = 0;  // [1] float
  size_t fc2_weights = 0;       // [H, I/2] uint8 E2M1 row-major
  size_t fc2_scales = 0;        // [H, I/16] uint8 Float8E4M3FN row-major
  size_t fc2_global_scale = 0;  // [1] float
  size_t stride = 0;
};

bool MakeNvfp4ExpertBlockLayout(int64_t hidden, int64_t inter, Nvfp4ExpertBlockLayout* layout);
bool MakeMxfp4ExpertBlockLayout(int64_t hidden, int64_t inter, Nvfp4ExpertBlockLayout* layout);

// Zero-copy view of one INT4 expert (works with canonical com.microsoft::QMoE bank tensors
// or a contiguous ExpertBlockLayout block).
struct Int4ExpertView {
  const uint8_t* fc1_weights = nullptr;  // [2*inter, hidden/2] (SM80 prepacked or raw row-major)
  const uint16_t* fc1_scales = nullptr;  // [2*inter, hidden/32] row-major or [hidden/32, 2*inter] block-major
  const uint16_t* fc1_bias = nullptr;    // optional [2*inter] in FP16 or BF16
  const uint8_t* fc2_weights = nullptr;  // [hidden, inter/2] (SM80 prepacked or raw row-major)
  const uint16_t* fc2_scales = nullptr;  // [hidden, inter/32] row-major or [inter/32, hidden] block-major
  const uint16_t* fc2_bias = nullptr;    // optional [hidden] in FP16 or BF16
  bool scales_are_block_major = false;   // false = canonical QMoE [N, K/32]; true = SharedExpertBlockV1 [K/32, N]
  bool weights_are_prepacked = true;     // true = SM80 ColumnMajorTileInterleave<64,4>; false = raw ONNX [N, K/2]
  bool scale_is_bf16 = false;
  bool bias_is_bf16 = false;
};

// Zero-copy view of one NVFP4 or MXFP4 expert (works with either separate bank tensors or a contiguous block).
struct Nvfp4ExpertView {
  const uint8_t* fc1_weights = nullptr;  // [2*inter, hidden/2]
  const uint8_t* fc1_scales = nullptr;   // [2*inter, hidden/16] E4M3 (NVFP4) or [2*inter, hidden/32] E8M0 (MXFP4)
  float fc1_global_scale = 1.0f;
  const uint16_t* fc1_bias = nullptr;    // optional [2*inter] in BF16 or FP16
  const uint8_t* fc2_weights = nullptr;  // [hidden, inter/2]
  const uint8_t* fc2_scales = nullptr;   // [hidden, inter/16] E4M3 (NVFP4) or [hidden, inter/32] E8M0 (MXFP4)
  float fc2_global_scale = 1.0f;
  const uint16_t* fc2_bias = nullptr;  // optional [hidden] in BF16 or FP16
  bool bias_is_bf16 = true;
  bool is_mxfp4 = false;
};

// Original K offset (0..31) of the low nibble of byte b (0..15) of a 32-K block; the high nibble
// holds that offset plus 8.
inline int LowNibbleK(int b) { return (b >> 1) + 16 * (b & 1); }

// Natural-order index -> index in the [low16 | high16] arrangement used by the INT4 CPU kernels.
int ReorderedPosition(int k_offset_in_block);

// Reorders one activation row (length k, multiple of 32) into the per-block INT4 decode order.
void ReorderActivationRow(const float* src, float* dst, int64_t k);

// Natural-order index (0..31) -> index in the raw ONNX INT4 [even16 | odd16] arrangement.
inline int RawInt4ReorderedPosition(int k_offset_in_block) {
  return ((k_offset_in_block & 1) << 4) + (k_offset_in_block >> 1);
}

// Reorders one activation row (length k, multiple of 32) into the raw ONNX INT4 [even16 | odd16] decode order.
void ReorderRawInt4ActivationRow(const float* src, float* dst, int64_t k);

// Converts between raw ONNX row-major INT4 [n_total, k_total/2] and SM80 ColumnMajorTileInterleave<64,4> INT4.
void PackSm80Int4Weights(const uint8_t* raw_weights, uint8_t* sm80_weights, int64_t n_total, int64_t k_total);
void UnpackSm80Int4Weights(const uint8_t* sm80_weights, uint8_t* raw_weights, int64_t n_total, int64_t k_total);
void TransposeInt4Scales(const uint16_t* src, uint16_t* dst, int64_t rows, int64_t cols);

// Natural-order index (0..31) -> index in the NVFP4 32-element pair-of-16-blocks arrangement:
// lanes 0..7 = even K of block 0, lanes 8..15 = even K of block 1,
// lanes 16..23 = odd K of block 0, lanes 24..31 = odd K of block 1.
inline int Nvfp4ReorderedPosition(int k_offset_in_32) {
  const int blk = (k_offset_in_32 >> 4) & 1;
  const int pos = k_offset_in_32 & 15;
  return ((pos & 1) << 4) + (blk << 3) + (pos >> 1);
}

// Reorders one activation row (length k, multiple of 32) into the NVFP4 AVX-512 decode order.
void ReorderNvfp4ActivationRow(const float* src, float* dst, int64_t k);

struct SwiGluParams {
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
};

uint8_t WeightCodeAt(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n, int64_t k);

float HalfToFloat(uint16_t h);
uint16_t FloatToHalf(float f);
float BFloat16ToFloat(uint16_t b);
uint16_t FloatToBFloat16(float f);
float DecodeFp4E2M1(uint8_t code);
float DecodeFloat8E4M3FN(uint8_t code);
float DecodeFloat8E8M0(uint8_t code);

void HalfBitsToFloatRow(const uint16_t* src, float* dst, int64_t count, bool force_scalar = false);
void FloatToHalfBitsRow(const float* src, uint16_t* dst, int64_t count, bool force_scalar = false);
void BFloat16BitsToFloatRow(const uint16_t* src, float* dst, int64_t count, bool force_scalar = false);
void FloatToBFloat16BitsRow(const float* src, uint16_t* dst, int64_t count, bool force_scalar = false);

// Ranges use [n_begin, n_end); out points at the first requested column, with out_stride between rows.
void GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total,
               int64_t k, const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin,
               int64_t n_end, bool force_scalar = false);

void Fc1SwiGluRange(const ExpertBlockLayout& layout, const uint8_t* block, const float* x_reordered,
                    int64_t m, const SwiGluParams& swiglu, float* inter_reordered, int64_t n_begin,
                    int64_t n_end, bool force_scalar = false, bool fp16_boundaries = false);

void Fc2Range(const ExpertBlockLayout& layout, const uint8_t* block, const float* inter_reordered,
              int64_t m, float* out, int64_t n_begin, int64_t n_end, bool force_scalar = false,
              bool apply_bias = true);

// Convenience forward with owned scratch allocations. Inference callers should use the range
// kernels above with preallocated activation buffers.
bool ExpertForward(const ExpertBlockLayout& layout, const uint8_t* block, const float* x, int64_t m,
                   const SwiGluParams& swiglu, float* out, bool force_scalar = false, bool apply_fc2_bias = true,
                   bool fp16_boundaries = false);

// Canonical INT4 QMoE CPU kernels (supports both row-major [N, K/32] and block-major [K/32, N] scales,
// both SM80 prepacked and raw row-major [N, K/2] INT4 weights, and both FP16 and BF16 scales/biases/boundaries).
void Int4GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias,
                   bool scales_are_block_major, bool weights_are_prepacked,
                   bool scale_is_bf16, bool bias_is_bf16,
                   int64_t n_total, int64_t k, const float* a_reordered, int64_t m,
                   float* out, int64_t out_stride, int64_t n_begin, int64_t n_end,
                   bool force_scalar = false);

void Int4Fc1SwiGluRange(int64_t hidden, int64_t inter, const Int4ExpertView& view,
                        const float* x_reordered, int64_t m, const SwiGluParams& swiglu,
                        float* inter_reordered, int64_t n_begin, int64_t n_end,
                        bool force_scalar = false, bool bf16_boundaries = false,
                        bool fp16_boundaries = false);

void Int4Fc2Range(int64_t hidden, int64_t inter, const Int4ExpertView& view,
                  const float* inter_reordered, int64_t m, float* out,
                  int64_t n_begin, int64_t n_end, bool force_scalar = false,
                  bool apply_bias = false);

bool Int4ExpertForward(int64_t hidden, int64_t inter, const Int4ExpertView& view,
                       const float* x, int64_t m, const SwiGluParams& swiglu, float* out,
                       bool force_scalar = false, bool apply_fc2_bias = false,
                       bool bf16_boundaries = false, bool fp16_boundaries = false);

// NVFP4 and MXFP4 row-major CPU kernels (supports both BF16 and FP16 scale/boundary rounding).
void Nvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                    int64_t n_begin, int64_t n_end, bool force_scalar = false,
                    bool round_scale_bf16 = true);

void Mxfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                    int64_t n_begin, int64_t n_end, bool force_scalar = false,
                    bool round_scale_bf16 = true);

void Nvfp4Fc1SwiGluRange(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                         const float* x_reordered, int64_t m, const SwiGluParams& swiglu,
                         float* inter_reordered, int64_t n_begin, int64_t n_end,
                         bool force_scalar = false, bool bf16_boundaries = true,
                         bool fp16_boundaries = false);

void Nvfp4Fc2Range(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                   const float* inter_reordered, int64_t m, float* out,
                   int64_t n_begin, int64_t n_end, bool force_scalar = false,
                   bool apply_bias = false, bool round_scale_bf16 = true);

// Convenience forward with owned scratch allocations, not for the allocation-free inference path.
bool Nvfp4ExpertForward(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                        const float* x, int64_t m, const SwiGluParams& swiglu, float* out,
                        bool force_scalar = false, bool apply_fc2_bias = false,
                        bool bf16_boundaries = true, bool fp16_boundaries = false);

bool CompilerHasAvx512() noexcept;
bool CpuHasAvx512();

}  // namespace qmoe_shared
}  // namespace contrib
}  // namespace onnxruntime

extern "C" {
int qmoe_shared_expert_forward(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                               int64_t m, float* out, float alpha, float beta, float limit,
                               int force_scalar);
int qmoe_shared_expert_contribution(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                    int64_t m, float* out, float alpha, float beta, float limit,
                                    int force_scalar);
int qmoe_shared_expert_contribution_fp16(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                         int64_t m, float* out, float alpha, float beta, float limit,
                                         int force_scalar);
int64_t qmoe_shared_block_bytes(int64_t hidden, int64_t inter);
int qmoe_shared_cpu_has_avx512(void);
int qmoe_shared_accepts_prepack_contract(int64_t weights_prepacked, int64_t block_size);
int qmoe_shared_weight_code_at(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n,
                               int64_t k);
int qmoe_int4_expert_contribution(const uint8_t* fc1_w, const uint16_t* fc1_s, const uint16_t* fc1_b,
                                  const uint8_t* fc2_w, const uint16_t* fc2_s, const uint16_t* fc2_b,
                                  int64_t hidden, int64_t inter, const float* x, int64_t m,
                                  float* out, float alpha, float beta, float limit,
                                  int weights_prepacked, int is_bf16, int apply_fc2_bias, int force_scalar);
int qmoe_nvfp4_expert_contribution_bf16(const uint8_t* fc1_w, const uint8_t* fc1_s, float fc1_g,
                                        const uint8_t* fc2_w, const uint8_t* fc2_s, float fc2_g,
                                        int64_t hidden, int64_t inter, const float* x, int64_t m,
                                        float* out, float alpha, float beta, float limit,
                                        int force_scalar);
int64_t qmoe_nvfp4_block_bytes(int64_t hidden, int64_t inter);
int qmoe_mxfp4_expert_contribution_bf16(const uint8_t* fc1_w, const uint8_t* fc1_s, float fc1_g,
                                        const uint8_t* fc2_w, const uint8_t* fc2_s, float fc2_g,
                                        int64_t hidden, int64_t inter, const float* x, int64_t m,
                                        float* out, float alpha, float beta, float limit,
                                        int force_scalar);
int64_t qmoe_mxfp4_block_bytes(int64_t hidden, int64_t inter);
void qmoe_shared_half_bits_to_float_row(const uint16_t* src, float* dst, int64_t count, int force_scalar);
int qmoe_shared_is_file_backed_mapping(const void* data, size_t bytes);
size_t qmoe_shared_evict_file_backed_pages(const void* data, size_t bytes);
int qmoe_file_store_test_heap_eviction_safety(void);
}
