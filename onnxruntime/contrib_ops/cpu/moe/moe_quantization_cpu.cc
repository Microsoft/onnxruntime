// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_quantization_cpu.h"
#include "core/framework/allocator.h"
#include "core/common/float16.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "core/common/narrow.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/util/math.h"
#include "core/platform/env_var_utils.h"
#include "core/common/logging/logging.h"
#include "core/util/thread_utils.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <atomic>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {
inline uint8_t GetPackedZeroPointValue(int64_t num_bits, uint8_t zero_point) {
  ORT_ENFORCE(num_bits > 0 && num_bits <= 8 && (8 % num_bits) == 0,
              "num_bits must be a positive divisor of 8, got ", num_bits);
  const int64_t pack_size = 8 / num_bits;
  const uint8_t mask = static_cast<uint8_t>((1u << num_bits) - 1u);
  uint8_t packed_value = 0;

  for (int64_t i = 0; i < pack_size; ++i) {
    packed_value |= static_cast<uint8_t>((zero_point & mask) << (i * num_bits));
  }

  return packed_value;
}

inline int64_t GetOptimalBlockSize(int64_t total_elements, int num_threads) {
  if (total_elements <= 0 || num_threads <= 0) return 64;
  const int64_t l1_cache_elements = 8192;  // ~32KB / 4 bytes per float
  const int64_t divisor = std::max(1, num_threads > 1 ? 4 : 2);
  const int64_t base_block_size = l1_cache_elements / divisor;
  const int64_t max_block = std::max(int64_t{32}, total_elements / std::max(int64_t{1}, int64_t{4}));
  return std::clamp(base_block_size, int64_t{32}, std::min(int64_t{512}, max_block));
}

inline int64_t GetUnrollFactor(int64_t vector_size) {
  if (vector_size <= 0) return 2;
  if (vector_size >= 512) return 16;
  if (vector_size >= 128) return 8;
  if (vector_size >= 32) return 4;
  return 2;
}

inline bool ShouldUseMemcpy(int64_t size) {
  return size >= 64;
}

inline int64_t GetDequantBlockSize(int64_t features, int64_t total_work, int64_t alignment = 1) {
  if (features <= 0 || total_work <= 0) return 16;
  const int64_t target_block_size = std::max(int64_t{16}, features / std::max(int64_t{1}, int64_t{8}));
  const int64_t work_based_size = std::max(int64_t{16}, total_work / std::max(int64_t{1}, int64_t{4}));
  int64_t block_size = std::min(target_block_size, work_based_size);
  // Round up to alignment so that row-wise zero-point packed-byte offsets are correct
  // when sharding across parallel dequant blocks.
  if (alignment > 1) {
    block_size = ((block_size + alignment - 1) / alignment) * alignment;
  }
  return block_size;
}

bool CanUseMlasQ4Dequant(int64_t num_bits) {
  if (num_bits != 4) {
    return false;
  }

  return true;
}

bool CanUseMlasQ4Gemm(int64_t expert_weight_bits, int64_t block_size,
                      int64_t rows, int64_t cols, MLAS_BLK_QUANT_TYPE& out_qtype) {
  if (expert_weight_bits != 4) {
    return false;
  }

  if (block_size == 64) {
    out_qtype = BlkQ4Sym64;
  } else if (block_size == 128) {
    out_qtype = BlkQ4Sym128;
  } else if (block_size == 0 || block_size == 32) {
    out_qtype = BlkQ4Sym;
  } else {
    return false;
  }

  size_t expected_size = MlasQ4GemmPackBSize(out_qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  return expected_size > 0;
}

bool CanUseMlasLutGemm(int64_t expert_weight_bits, int64_t block_size,
                       int64_t rows, int64_t cols) {
  if (expert_weight_bits != 2 || block_size <= 0) {
    return false;
  }

  if ((cols % block_size) != 0) {
    return false;
  }

  return MlasIsLutGemmAvailable(static_cast<size_t>(rows), static_cast<size_t>(cols),
                                static_cast<size_t>(expert_weight_bits), static_cast<size_t>(block_size));
}

}  // namespace

namespace onnxruntime {
namespace contrib {

constexpr const char* kUseMlasQ4GemmMoe = "ORT_USE_MLAS_Q4_GEMM_MOE";

// Overrides the node's accuracy_level for the MLAS QNBit GEMM (MatMulNBits kernel) path of
// block-wise 4/8-bit experts. Unset: the attribute decides for 4-bit; 8-bit has no fp32 kernel
// and uses int8 activations at every level. "fp32": fp32 activations only, so 8-bit keeps the
// dequantize path. "int8": int8 activations (accuracy_level 4). "0": path disabled. Any other
// value is an error.
constexpr const char* kQMoEQNBitGemmEnv = "ORT_QMOE_CPU_QNBIT_GEMM";

// Tag appended to the prepacked shape buffer to mark the QNBit-packed layout (see PrePackQNBitExperts).
constexpr int64_t kQNBitPackedLayoutTag = 0x514E4249;  // 'QNBI'

// Prepacked shape buffer: rank, dims, then `trailer` (empty for the legacy layouts; the QNBit layout
// appends kQNBitPackedLayoutTag, the compute type it was packed for and whether zero points were folded in).
static void PushPrePackedShapeBuffer(const TensorShape& shape, gsl::span<const int64_t> trailer,
                                     const AllocatorPtr& alloc, PrePackedWeights& prepacked_weights) {
  const auto dims = shape.GetDims();
  const size_t shape_size = (1 + dims.size() + trailer.size()) * sizeof(int64_t);
  auto shape_buffer = IAllocator::MakeUniquePtr<void>(alloc, shape_size);
  int64_t* buffer_data = static_cast<int64_t*>(shape_buffer.get());
  buffer_data[0] = static_cast<int64_t>(dims.size());
  std::copy(dims.begin(), dims.end(), buffer_data + 1);
  std::copy(trailer.begin(), trailer.end(), buffer_data + 1 + dims.size());
  prepacked_weights.buffers_.push_back(std::move(shape_buffer));
  prepacked_weights.buffer_sizes_.push_back(shape_size);
}

template <typename TScale>
void DequantizeBlockWithMlas(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             float* dequantized_data,
                             MLAS_THREADPOOL* thread_pool);

template <typename TScale>
Status ConvertToMlasQ4Format(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             MLAS_BLK_QUANT_TYPE qtype,
                             AllocatorPtr allocator,
                             IAllocatorUniquePtr<uint8_t>& mlas_packed_buffer) {
  if (num_bits != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only 4-bit quantization supported for MLAS Q4 format conversion");
  }
  if (zero_points != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MLAS Q4 format conversion only supports symmetric quantization (zero_points must be null)");
  }

  auto temp_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(rows * cols));
  float* temp_float = temp_float_buffer.get();

  DequantizeBlockWithMlas(quantized_data, scales, zero_points, block_size, num_bits, rows, cols, temp_float, nullptr);

  // Transpose from N x K (weights) to K x N.
  // DirectQ4Gemm expects weights to be packed in a specific layout ([K, N] logically)
  auto transposed_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(rows * cols));
  float* transposed_float = transposed_float_buffer.get();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      transposed_float[c * rows + r] = temp_float[r * cols + c];
    }
  }

  size_t packed_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  if (packed_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MLAS Q4 packing not supported for this configuration");
  }

  mlas_packed_buffer = IAllocator::MakeUniquePtr<uint8_t>(allocator, packed_size);
  MlasQ4GemmPackB(qtype, mlas_packed_buffer.get(), transposed_float, static_cast<size_t>(rows), static_cast<size_t>(cols), static_cast<size_t>(rows));

  return Status::OK();
}

Status DirectQ4Gemm(const float* A,
                    const uint8_t* mlas_packed_B,
                    const float* bias,
                    float* C,
                    int64_t M,
                    int64_t N,
                    int64_t K,
                    MLAS_BLK_QUANT_TYPE qtype,
                    MLAS_THREADPOOL* thread_pool) {
  MLAS_Q4_GEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = static_cast<size_t>(K);
  params.B = mlas_packed_B;
  params.Bias = bias;
  params.C = C;
  params.ldc = static_cast<size_t>(N);
  params.OutputProcessor = nullptr;

  MlasQ4GemmBatch(qtype, static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K), 1, &params, thread_pool);
  return Status::OK();
}

template <typename TScale>
const float* GetFloatScaleData(const TScale* scales_data,
                               size_t scale_count,
                               float* converted_scales) {
  if constexpr (std::is_same_v<TScale, float>) {
    ORT_UNUSED_PARAMETER(scale_count);
    ORT_UNUSED_PARAMETER(converted_scales);
    return scales_data;
  } else {
    ORT_ENFORCE(converted_scales != nullptr, "converted_scales buffer must be provided for non-float scale data.");
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(scales_data), converted_scales, scale_count);
    return converted_scales;
  }
}

template <typename TScale>
bool TryRunLutGemm(const float* activations,
                   float* output,
                   const uint8_t* weights_data,
                   const void* direct_lut_cache_ptr,
                   const TScale* scales_ptr,
                   const uint8_t* zp_ptr,
                   int64_t expert_idx,
                   int64_t rows,
                   int64_t cols,
                   int64_t packed_cols,
                   int64_t block_size,
                   int64_t blocks_per_row,
                   std::byte* thread_lut_packed_buffer,
                   float* thread_lut_scale_buffer,
                   int64_t num_expert_tokens,
                   MLAS_THREADPOOL* thread_pool) {
  if (direct_lut_cache_ptr == nullptr && weights_data == nullptr) {
    return false;
  }

  const void* packed_lut_b = direct_lut_cache_ptr;
  if (packed_lut_b == nullptr) {
    // Caller must have already called MlasInitLutGemmKernelConfig before invoking this function.
    ORT_ENFORCE(thread_lut_packed_buffer != nullptr, "Thread-local LUT packed buffer is required.");
    const size_t scale_count = static_cast<size_t>(rows * blocks_per_row);
    const float* scales_fp32 = GetFloatScaleData(scales_ptr, scale_count, thread_lut_scale_buffer);
    MlasLutGemmPack(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                    static_cast<size_t>(block_size), zp_ptr != nullptr,
                    reinterpret_cast<const std::byte*>(weights_data + expert_idx * rows * packed_cols),
                    scales_fp32, zp_ptr, false, thread_lut_packed_buffer, thread_pool);
    packed_lut_b = thread_lut_packed_buffer;
  }

  MlasLutGemm(activations, static_cast<size_t>(block_size), packed_lut_b, output,
              static_cast<size_t>(cols), static_cast<size_t>(num_expert_tokens),
              static_cast<size_t>(rows), zp_ptr != nullptr, thread_pool);
  return true;
}

template <typename TScale>
Status BuildDirectLutPackedBCache(const uint8_t* quantized_data,
                                  const TScale* scales_data,
                                  const uint8_t* zero_points,
                                  int64_t num_experts,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t block_size,
                                  int64_t blocks_per_row,
                                  AllocatorPtr allocator,
                                  IAllocatorUniquePtr<void>& packed_b) {
  ORT_RETURN_IF_NOT(CanUseMlasLutGemm(2, block_size, rows, cols),
                    "LUT GEMM is not supported for rows=", rows, ", cols=", cols, ", block_size=", block_size, ".");

  const bool has_zero_points = (zero_points != nullptr);
  MlasInitLutGemmKernelConfig(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                              static_cast<size_t>(block_size), has_zero_points);
  const size_t packed_size_per_expert = MlasLutGemmPackedSize(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                                              static_cast<size_t>(block_size), has_zero_points);
  ORT_RETURN_IF(packed_size_per_expert == 0, "Failed to compute LUT GEMM packed size.");

  constexpr int64_t kPackSize2Bit = 4;
  const int64_t packed_cols = cols / kPackSize2Bit;
  const size_t quantized_stride = static_cast<size_t>(rows * packed_cols);
  const size_t scales_stride = static_cast<size_t>(rows * blocks_per_row);
  const size_t zp_stride = has_zero_points ? static_cast<size_t>(rows * ((blocks_per_row + kPackSize2Bit - 1) / kPackSize2Bit)) : 0;
  const size_t total_packed_size = SafeInt<size_t>(packed_size_per_expert) * static_cast<size_t>(num_experts);

  packed_b = IAllocator::MakeUniquePtr<void>(allocator, total_packed_size, true);
  auto* packed_b_ptr = static_cast<std::byte*>(packed_b.get());
  std::vector<float> scales_fp32(scales_stride);

  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const uint8_t* expert_quantized = quantized_data + static_cast<size_t>(expert_idx) * quantized_stride;
    const TScale* expert_scales = scales_data + static_cast<size_t>(expert_idx) * scales_stride;
    const uint8_t* expert_zero_points = has_zero_points ? zero_points + static_cast<size_t>(expert_idx) * zp_stride : nullptr;
    const float* expert_scales_fp32 = GetFloatScaleData(expert_scales, scales_stride, scales_fp32.data());

    MlasLutGemmPack(static_cast<size_t>(rows), static_cast<size_t>(cols), 2, static_cast<size_t>(block_size),
                    has_zero_points, reinterpret_cast<const std::byte*>(expert_quantized), expert_scales_fp32,
                    expert_zero_points, false, packed_b_ptr + static_cast<size_t>(expert_idx) * packed_size_per_expert, nullptr);
  }

  return Status::OK();
}

template <typename TScale>
void DequantizeBlockWithMlas(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             float* dequantized_data,
                             MLAS_THREADPOOL* thread_pool) {
  ORT_UNUSED_PARAMETER(thread_pool);
  const float default_zp_8bit = 128.0f;
  const int64_t zp_pack_size = 8 / num_bits;

  if (CanUseMlasQ4Dequant(num_bits) && zero_points == nullptr) {
    // Use optimized symmetric 4-bit dequantization
    const float default_zp_4bit = 8.0f;
    const int64_t packed_cols = (cols + 1) / 2;
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;

    if (block_size == 0) {
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * packed_cols;
        float* row_output = dequantized_data + r * cols;
        const float scale = static_cast<float>(scales[r]);

        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
          const uint8_t packed_val0 = row_data[(c + 0) / 2];
          const uint8_t packed_val1 = row_data[(c + 2) / 2];
          const uint8_t packed_val2 = row_data[(c + 4) / 2];
          const uint8_t packed_val3 = row_data[(c + 6) / 2];

          row_output[c + 0] = scale * (static_cast<float>(packed_val0 & 0x0F) - default_zp_4bit);
          row_output[c + 1] = scale * (static_cast<float>(packed_val0 >> 4) - default_zp_4bit);
          row_output[c + 2] = scale * (static_cast<float>(packed_val1 & 0x0F) - default_zp_4bit);
          row_output[c + 3] = scale * (static_cast<float>(packed_val1 >> 4) - default_zp_4bit);
          row_output[c + 4] = scale * (static_cast<float>(packed_val2 & 0x0F) - default_zp_4bit);
          row_output[c + 5] = scale * (static_cast<float>(packed_val2 >> 4) - default_zp_4bit);
          row_output[c + 6] = scale * (static_cast<float>(packed_val3 & 0x0F) - default_zp_4bit);
          row_output[c + 7] = scale * (static_cast<float>(packed_val3 >> 4) - default_zp_4bit);
        }

        for (; c < cols; c += 2) {
          const uint8_t packed_val = row_data[c / 2];
          const uint8_t val0 = packed_val & 0x0F;
          const uint8_t val1 = packed_val >> 4;

          row_output[c] = scale * (static_cast<float>(val0) - default_zp_4bit);
          if (c + 1 < cols) {
            row_output[c + 1] = scale * (static_cast<float>(val1) - default_zp_4bit);
          }
        }
      }
      return;
    } else {
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * packed_cols;
        float* row_output = dequantized_data + r * cols;

        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);

          int64_t c = block_start;
          for (; c + 4 <= block_end; c += 4) {
            const uint8_t packed_val0 = row_data[(c + 0) / 2];
            const uint8_t packed_val1 = row_data[(c + 2) / 2];

            row_output[c + 0] = scale * (static_cast<float>(packed_val0 & 0x0F) - default_zp_4bit);
            row_output[c + 1] = scale * (static_cast<float>(packed_val0 >> 4) - default_zp_4bit);
            row_output[c + 2] = scale * (static_cast<float>(packed_val1 & 0x0F) - default_zp_4bit);
            row_output[c + 3] = scale * (static_cast<float>(packed_val1 >> 4) - default_zp_4bit);
          }

          for (; c < block_end; c += 2) {
            const uint8_t packed_val = row_data[c / 2];
            const uint8_t val0 = packed_val & 0x0F;
            const uint8_t val1 = packed_val >> 4;

            row_output[c] = scale * (static_cast<float>(val0) - default_zp_4bit);
            if (c + 1 < block_end) {
              row_output[c + 1] = scale * (static_cast<float>(val1) - default_zp_4bit);
            }
          }
        }
      }
      return;
    }
  }

  // Generic dequantization logic for 8-bit (symmetric/asymmetric) and 4-bit (asymmetric)
  if (num_bits == 8) {
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
    if (block_size == 0) {
      // 8-bit, row-wise
      for (int64_t r = 0; r < rows; ++r) {
        const float scale = static_cast<float>(scales[r]);
        const uint8_t zero_pt = (zero_points == nullptr) ? static_cast<uint8_t>(default_zp_8bit) : zero_points[r];
        MlasDequantizeLinear(
            quantized_data + r * cols,
            dequantized_data + r * cols,
            static_cast<size_t>(cols),
            scale,
            zero_pt);
      }
    } else {
      // 8-bit, block-wise
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * cols;
        float* row_output = dequantized_data + r * cols;
        const uint8_t* row_zp_data = (zero_points == nullptr) ? nullptr : zero_points + r * blocks_per_row;

        int64_t c = 0;
        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);
          const float zp = (row_zp_data == nullptr) ? default_zp_8bit : static_cast<float>(row_zp_data[block_idx]);

          for (c = block_start; c + 4 <= block_end; c += 4) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zp);
            row_output[c + 1] = scale * (static_cast<float>(row_data[c + 1]) - zp);
            row_output[c + 2] = scale * (static_cast<float>(row_data[c + 2]) - zp);
            row_output[c + 3] = scale * (static_cast<float>(row_data[c + 3]) - zp);
          }
          for (; c < block_end; ++c) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zp);
          }
        }
      }
    }
  } else if (num_bits == 2 || num_bits == 4) {
    const uint8_t value_mask = static_cast<uint8_t>((1u << num_bits) - 1u);
    const uint8_t default_zero_point = static_cast<uint8_t>(1u << (num_bits - 1));
    const uint8_t default_zp_packed = GetPackedZeroPointValue(num_bits, default_zero_point);
    const int64_t pack_size = 8 / num_bits;
    const int64_t packed_cols = (cols + pack_size - 1) / pack_size;
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
    const int64_t blocks_per_row_packed = (blocks_per_row + zp_pack_size - 1) / zp_pack_size;

    for (int64_t r = 0; r < rows; ++r) {
      const uint8_t* row_data = quantized_data + r * packed_cols;
      float* row_output = dequantized_data + r * cols;

      if (block_size > 0) {
        const uint8_t* row_zp_data = (zero_points == nullptr) ? nullptr : zero_points + r * blocks_per_row_packed;
        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);

          const uint8_t packed_zp = (row_zp_data == nullptr) ? default_zp_packed : row_zp_data[block_idx / zp_pack_size];
          const int zp_shift = static_cast<int>((block_idx % zp_pack_size) * num_bits);
          const float zp = static_cast<float>((packed_zp >> zp_shift) & value_mask);

          for (int64_t c = block_start; c < block_end; ++c) {
            const uint8_t packed_val = row_data[c / pack_size];
            const int shift = static_cast<int>((c % pack_size) * num_bits);
            const uint8_t value = static_cast<uint8_t>((packed_val >> shift) & value_mask);
            row_output[c] = scale * (static_cast<float>(value) - zp);
          }
        }
      } else {
        const uint8_t packed_zp = (zero_points == nullptr) ? default_zp_packed : zero_points[r / zp_pack_size];
        const int zp_shift = static_cast<int>((r % zp_pack_size) * num_bits);
        const float zp = static_cast<float>((packed_zp >> zp_shift) & value_mask);
        const float scale = static_cast<float>(scales[r]);

        for (int64_t c = 0; c < cols; ++c) {
          const uint8_t packed_val = row_data[c / pack_size];
          const int shift = static_cast<int>((c % pack_size) * num_bits);
          const uint8_t value = static_cast<uint8_t>((packed_val >> shift) & value_mask);
          row_output[c] = scale * (static_cast<float>(value) - zp);
        }
      }
    }
  } else {
    ORT_THROW("Unsupported num_bits (", num_bits, ") in DequantizeBlockWithMlas");
  }
}

template <typename TScale>
void DequantizeBlock(const uint8_t* quantized_data,
                     const TScale* scales,
                     const uint8_t* zero_points,
                     int64_t block_size,
                     int64_t num_bits,
                     int64_t rows,
                     int64_t cols,
                     float* dequantized_data,
                     MLAS_THREADPOOL* thread_pool = nullptr) {
  DequantizeBlockWithMlas(quantized_data, scales, zero_points, block_size, num_bits, rows, cols, dequantized_data, thread_pool);
}

template <typename TScale>
void DequantizePrePacked(const uint8_t* prepacked_data,
                         const TScale* scales,
                         const uint8_t* zero_points,
                         int64_t block_size,
                         int64_t rows,
                         int64_t cols,
                         float* dequantized_data,
                         const gsl::span<const int64_t>& scale_dims) {
  // TODO(tlwu): Generalize this helper if we add prepacked 2-bit QMoE support.
  // The current prepack path is intentionally 4-bit-only.
  // prepacked_data is [cols, rows] (transposed, unpacked)
  // dequantized_data is [cols, rows] (transposed)
  // scales, zero_points correspond to original [rows, cols] layout

  const float default_zp_4bit = 8.0f;
  const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
  const int64_t zp_pack_size = 2;  // Always 2 for 4-bit

  // Iterate over Columns (K) then Rows (N) because prepacked_data is [K, N]
  for (int64_t c = 0; c < cols; ++c) {
    for (int64_t r = 0; r < rows; ++r) {
      uint8_t val = prepacked_data[c * rows + r];

      int64_t block_idx = (block_size > 0) ? (c / block_size) : 0;
      if (block_size > 0) block_idx = std::min(block_idx, blocks_per_row - 1);

      int64_t scale_idx;
      if (scale_dims.size() == 3) {  // block-wise
        scale_idx = r * blocks_per_row + block_idx;
      } else {  // per-channel
        scale_idx = r;
      }

      float scale = static_cast<float>(scales[scale_idx]);
      float zp = default_zp_4bit;

      if (zero_points != nullptr) {
        int64_t zp_idx;
        bool is_lower_nibble;

        if (scale_dims.size() == 3) {  // block-wise
          int64_t zp_blocks_packed = (blocks_per_row + zp_pack_size - 1) / zp_pack_size;
          zp_idx = r * zp_blocks_packed + block_idx / 2;
          is_lower_nibble = (block_idx % 2 == 0);
        } else {
          zp_idx = r / 2;
          is_lower_nibble = (r % 2 == 0);
        }

        uint8_t packed_zp = zero_points[zp_idx];
        zp = is_lower_nibble ? static_cast<float>(packed_zp & 0x0F) : static_cast<float>(packed_zp >> 4);
      }

      dequantized_data[c * rows + r] = scale * (static_cast<float>(val) - zp);
    }
  }
}

template <typename TScale>
Status BuildDirectQ4PackedBCache(const uint8_t* prepacked_weights,
                                 const TScale* scales_data,
                                 int64_t num_experts,
                                 int64_t rows,
                                 int64_t cols,
                                 int64_t block_size,
                                 const gsl::span<const int64_t>& scales_dims,
                                 MLAS_BLK_QUANT_TYPE qtype,
                                 AllocatorPtr allocator,
                                 IAllocatorUniquePtr<void>& packed_b) {
  const size_t packed_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  if (packed_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to compute MLAS Q4 packed size for cache");
  }

  const bool is_block_wise = scales_dims.size() == 3;
  const int64_t scales_expert_stride = is_block_wise ? (rows * scales_dims[2]) : rows;
  const size_t prepacked_expert_stride = static_cast<size_t>(rows * cols);
  const size_t total_packed_size = packed_size * static_cast<size_t>(num_experts);

  packed_b = IAllocator::MakeUniquePtr<void>(allocator, total_packed_size, true);
  uint8_t* packed_b_ptr = static_cast<uint8_t*>(packed_b.get());

  std::vector<float> dequantized_transposed(static_cast<size_t>(rows * cols));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const uint8_t* expert_prepacked = prepacked_weights + static_cast<size_t>(expert_idx) * prepacked_expert_stride;
    const TScale* expert_scales = scales_data + expert_idx * scales_expert_stride;

    DequantizePrePacked(expert_prepacked, expert_scales, nullptr, block_size, rows, cols,
                        dequantized_transposed.data(), scales_dims);

    MlasQ4GemmPackB(qtype, packed_b_ptr + expert_idx * packed_size, dequantized_transposed.data(),
                    static_cast<size_t>(rows), static_cast<size_t>(cols), static_cast<size_t>(rows));
  }

  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                           /*out*/ bool& is_packed,
                           /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  if (fc1_expert_weight_bits_ != expert_weight_bits_ ||
      fc2_expert_weight_bits_ != expert_weight_bits_ ||
      fc3_expert_weight_bits_ != expert_weight_bits_) {
    return Status::OK();
  }

  // If scales are prepacked, they are constant initializers.
  if (input_idx == 3) {
    return Status::OK();
  }
  if (input_idx == 6) {
    return Status::OK();
  }

  Tensor unpacked_sm80_tensor;
  const Tensor* weight_tensor = &tensor;
  if (weights_prepacked_ && (input_idx == 2 || input_idx == 5) && tensor.Shape().NumDimensions() == 3) {
    const auto& raw_shape = tensor.Shape();
    const int64_t num_experts = raw_shape[0];
    int64_t rows = raw_shape[1];
    int64_t cols = raw_shape[2] * 2;
    const int scales_idx = (input_idx == 2) ? 3 : 6;
    const Tensor* scales_tensor = nullptr;
    if (Info().TryGetConstantInput(scales_idx, &scales_tensor) && scales_tensor != nullptr &&
        scales_tensor->Shape().NumDimensions() == 3 && scales_tensor->Shape()[0] == num_experts) {
      rows = scales_tensor->Shape()[1];
      cols = scales_tensor->Shape()[2] * block_size_;
    }
    ORT_RETURN_IF_NOT(rows > 0 && cols > 0 &&
                          (rows % qmoe_shared::kColumnsInterleaved) == 0 &&
                          (cols % qmoe_shared::kRowsPerTile) == 0 &&
                          SafeInt<size_t>(num_experts) * rows * (cols / 2) == tensor.SizeInBytes(),
                      "QMoE CPU weights_prepacked=1 requires SM80 tile-aligned INT4 dimensions.");
    const size_t expert_bytes = static_cast<size_t>(rows * (cols / 2));
    TensorShape canonical_shape{num_experts, rows, cols / 2};
    unpacked_sm80_tensor = Tensor(tensor.DataType(), canonical_shape, alloc);
    const uint8_t* src_ptr = static_cast<const uint8_t*>(tensor.DataRaw());
    uint8_t* dst_ptr = unpacked_sm80_tensor.MutableData<uint8_t>();
    for (int64_t e = 0; e < num_experts; ++e) {
      qmoe_shared::UnpackSm80Int4Weights(src_ptr + static_cast<size_t>(e) * expert_bytes,
                                         dst_ptr + static_cast<size_t>(e) * expert_bytes,
                                         rows, cols);
    }
    weight_tensor = &unpacked_sm80_tensor;
  }

  if ((input_idx == 2 || input_idx == 5) && weight_tensor->Shape().NumDimensions() == 3) {
    ORT_RETURN_IF_ERROR(PrePackQNBitExperts(*weight_tensor, input_idx, alloc, is_packed, prepacked_weights));
    if (is_packed) {
      if (input_idx == 2) {
        fc1_shape_ = tensor.Shape();
      } else {
        fc2_shape_ = tensor.Shape();
      }
      return Status::OK();
    }
  }

  // Only support PrePack for FC1 (2) and FC2 (5) weights.
  // 4-bit uses the existing unpacked-transposed/Q4 cache path.
  // 2-bit block-wise uses a direct LUT GEMM packed cache when supported.
  if (expert_weight_bits_ != 4 && expert_weight_bits_ != 2) {
    return Status::OK();
  }

  if (input_idx == 2 || input_idx == 5) {
    const auto& shape = weight_tensor->Shape();
    const int64_t num_experts = shape[0];
    const int64_t rows = shape[1];
    const int64_t cols_packed = shape[2];
    const int64_t pack_size = 8 / expert_weight_bits_;
    const int64_t cols = cols_packed * pack_size;

    if (input_idx == 2) {
      fc1_shape_ = tensor.Shape();
    } else if (input_idx == 5) {
      fc2_shape_ = tensor.Shape();
    }

    if (expert_weight_bits_ == 2) {
      if (block_size_ <= 0 || (cols % block_size_) != 0 || !prepacked_weights) {
        return Status::OK();
      }

      const int scales_idx = (input_idx == 2) ? 3 : 6;
      const int zp_idx = (input_idx == 2) ? 11 : 12;
      const Tensor* scales_tensor = nullptr;
      if (!Info().TryGetConstantInput(scales_idx, &scales_tensor) || scales_tensor == nullptr) {
        return Status::OK();
      }

      const auto& scales_dims = scales_tensor->Shape().GetDims();
      if (scales_dims.size() != 3 || scales_dims[2] <= 1) {
        return Status::OK();
      }

      if (scales_dims[1] != rows || scales_dims[2] != (cols / block_size_)) {
        return Status::OK();
      }

      const bool has_zp_input = zp_idx < static_cast<int>(Info().node().InputDefs().size()) &&
                                Info().node().InputDefs()[zp_idx]->Exists();
      const Tensor* zp_tensor = nullptr;
      if (has_zp_input && !Info().TryGetConstantInput(zp_idx, &zp_tensor)) {
        return Status::OK();
      }

      if (!CanUseMlasLutGemm(expert_weight_bits_, block_size_, rows, cols)) {
        return Status::OK();
      }

      IAllocatorUniquePtr<void> lut_cache_buffer;
      const uint8_t* zp_data = zp_tensor ? zp_tensor->Data<uint8_t>() : nullptr;
      ORT_RETURN_IF_ERROR(BuildDirectLutPackedBCache(static_cast<const uint8_t*>(weight_tensor->DataRaw()),
                                                     scales_tensor->Data<T>(),
                                                     zp_data,
                                                     num_experts,
                                                     rows,
                                                     cols,
                                                     block_size_,
                                                     scales_dims[2],
                                                     alloc,
                                                     lut_cache_buffer));

      const size_t cache_size = MlasLutGemmPackedSize(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                                      static_cast<size_t>(block_size_), zp_data != nullptr) *
                                static_cast<size_t>(num_experts);
      prepacked_weights->buffers_.push_back(std::move(lut_cache_buffer));
      prepacked_weights->buffer_sizes_.push_back(cache_size);
      is_packed = true;
      PushPrePackedShapeBuffer(tensor.Shape(), {}, alloc, *prepacked_weights);
      return Status::OK();
    }

    size_t packed_size = static_cast<size_t>(num_experts * rows * cols);
    auto packed_buffer = IAllocator::MakeUniquePtr<void>(alloc, packed_size, true);
    uint8_t* dst_base = static_cast<uint8_t*>(packed_buffer.get());
    const uint8_t* src_base = static_cast<const uint8_t*>(weight_tensor->DataRaw());

    for (int64_t i = 0; i < num_experts; ++i) {
      const uint8_t* src = src_base + i * rows * cols_packed;
      uint8_t* dst = dst_base + i * rows * cols;

      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          uint8_t packed_val = src[r * cols_packed + (c / 2)];
          uint8_t val = (c % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);

          dst[c * rows + r] = val;
        }
      }
    }

    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_buffer));
      prepacked_weights->buffer_sizes_.push_back(packed_size);
      is_packed = true;
      PushPrePackedShapeBuffer(tensor.Shape(), {}, alloc, *prepacked_weights);

      // Try build MLAS Q4 cache if scales are available
      if (use_mlas_q4_gemm_) {
        const Tensor* scales_tensor = nullptr;
        MLAS_BLK_QUANT_TYPE qtype = BlkQ4Sym;
        int scales_idx = -1;
        int zp_idx = -1;

        if (input_idx == 2) {  // FC1
          scales_idx = 3;
          zp_idx = 11;
        } else if (input_idx == 5) {  // FC2
          scales_idx = 6;
          zp_idx = 12;
        }

        if (scales_idx != -1 &&
            (zp_idx >= static_cast<int>(Info().node().InputDefs().size()) || !Info().node().InputDefs()[zp_idx]->Exists()) &&
            Info().TryGetConstantInput(scales_idx, &scales_tensor) &&
            scales_tensor != nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, block_size_, rows, cols, qtype)) {
          IAllocatorUniquePtr<void> cache_buffer;
          const auto& scales_dims = scales_tensor->Shape().GetDims();
          const T* scales_data = scales_tensor->Data<T>();
          // Use the simple packed buffer we just created (buffer 0) as input
          const uint8_t* simple_packed = dst_base;

          if (BuildDirectQ4PackedBCache(simple_packed, scales_data, num_experts, rows, cols,
                                        block_size_, scales_dims, qtype,
                                        alloc, cache_buffer)
                  .IsOK()) {
            // Store the MLAS Q4 cache as buffer 2 (after unpacked weights and shape).
            size_t cache_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols)) * static_cast<size_t>(num_experts);
            prepacked_weights->buffers_.push_back(std::move(cache_buffer));
            prepacked_weights->buffer_sizes_.push_back(cache_size);
          }
        }
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                             gsl::span<const size_t> prepacked_buffer_sizes,
                                             int input_idx,
                                             /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  auto parse_shape = [&](TensorShape& shape) -> Status {
    if (prepacked_buffers.size() <= 1) {
      return Status::OK();
    }

    ORT_RETURN_IF_NOT(prepacked_buffer_sizes.size() > 1,
                      "Missing QMoE prepacked shape buffer size metadata.");
    ORT_RETURN_IF_NOT(prepacked_buffer_sizes[1] >= sizeof(int64_t),
                      "QMoE prepacked shape buffer is too small to contain a rank.");

    const int64_t* buffer_data = static_cast<const int64_t*>(prepacked_buffers[1].get());
    const int64_t rank = buffer_data[0];
    ORT_RETURN_IF_NOT(rank == 3, "Expected rank 3 for QMoE weight shape, got ", rank, ".");

    const size_t shape_buffer_size = SafeInt<size_t>(static_cast<size_t>(rank) + 1) * sizeof(int64_t);
    ORT_RETURN_IF_NOT(prepacked_buffer_sizes[1] >= shape_buffer_size,
                      "QMoE prepacked shape buffer is too small for rank ", rank, ".");

    std::vector<int64_t> dims(static_cast<size_t>(rank));
    memcpy(dims.data(), buffer_data + 1, static_cast<size_t>(rank) * sizeof(int64_t));
    shape = TensorShape(dims);
    return Status::OK();
  };

  // QNBit-packed layout: [packed experts, shape buffer = rank, dims, kQNBitPackedLayoutTag, compute type, has zero point].
  if ((input_idx == 2 || input_idx == 5) && prepacked_buffers.size() == 2 && prepacked_buffer_sizes.size() == 2 &&
      prepacked_buffer_sizes[1] >= 7 * sizeof(int64_t)) {
    const int64_t* buffer_data = static_cast<const int64_t*>(prepacked_buffers[1].get());
    if (buffer_data[0] == 3 && buffer_data[4] == kQNBitPackedLayoutTag) {
      ORT_RETURN_IF_NOT(use_qnbit_gemm_, "QMoE prepacked weights use the QNBit layout but this kernel has it disabled.");
      // The packed bytes depend on the compute type (block sums, folded scales), and equal sizes do
      // not imply equal layouts, so a buffer packed under another ORT_QMOE_CPU_QNBIT_GEMM setting or
      // saved by another build must be rejected rather than silently used.
      ORT_RETURN_IF_NOT(buffer_data[5] == static_cast<int64_t>(qnbit_compute_type_),
                        "QMoE prepacked weights were packed for QNBit compute type ", buffer_data[5],
                        " but this kernel uses ", static_cast<int64_t>(qnbit_compute_type_), ".");
      TensorShape& shape = (input_idx == 2) ? fc1_shape_ : fc2_shape_;
      ORT_RETURN_IF_ERROR(parse_shape(shape));
      const int64_t num_experts = shape[0];
      const int64_t rows = shape[1];
      const int64_t cols = shape[2] * (8 / expert_weight_bits_);
      QNBitEligibility eligibility;
      ORT_RETURN_IF_NOT(QNBitGemmEligible(input_idx, num_experts, rows, cols, eligibility),
                        "QMoE prepacked weights use the QNBit layout but the node is not eligible for it.");
      ORT_RETURN_IF_NOT(buffer_data[6] == (eligibility.zero_points != nullptr ? 1 : 0),
                        "QMoE prepacked weights were packed ", buffer_data[6] ? "with" : "without",
                        " zero points but this node has ", eligibility.zero_points != nullptr ? "them." : "none.");
      QNBitPackedExperts& packed = (input_idx == 2) ? qnbit_fc1_ : qnbit_fc2_;
      // Skip re-initialization (and the fp16 scales re-conversion) when this kernel already packed
      // the weights itself: the session-state prepack pass always runs PrePack before handing the
      // buffers back here.
      if (packed.packed_size_per_expert == 0) {
        ORT_RETURN_IF_ERROR(InitQNBitPacked(packed, eligibility, num_experts, rows, cols,
                                            Info().GetAllocator(OrtMemType::OrtMemTypeDefault)));
      }
      ORT_RETURN_IF_NOT(prepacked_buffer_sizes[0] == SafeInt<size_t>(packed.packed_size_per_expert) * static_cast<size_t>(num_experts),
                        "QMoE prepacked QNBit buffer size does not match the expert shape.");
      packed.packed = std::move(prepacked_buffers[0]);
      used_shared_buffers = true;
      return Status::OK();
    }
  }

  if (expert_weight_bits_ == 2) {
    if ((input_idx == 2 || input_idx == 5) && !prepacked_buffers.empty()) {
      if (input_idx == 2) {
        packed_fc1_lut_cache_ = std::move(prepacked_buffers[0]);
        ORT_RETURN_IF_ERROR(parse_shape(fc1_shape_));
      } else {
        packed_fc2_lut_cache_ = std::move(prepacked_buffers[0]);
        ORT_RETURN_IF_ERROR(parse_shape(fc2_shape_));
      }

      // Re-initialize MLAS LUT kernel config for the restored shape.
      // The global T-MAC param cache may not be populated in a fresh session sharing prepacked weights.
      const TensorShape& restored_shape = (input_idx == 2) ? fc1_shape_ : fc2_shape_;
      if (restored_shape.NumDimensions() == 3) {
        const int64_t rows = restored_shape[1];
        const int64_t cols = restored_shape[2] * (8 / expert_weight_bits_);
        const int zp_idx = (input_idx == 2) ? 11 : 12;
        const bool has_zp = zp_idx < static_cast<int>(Info().node().InputDefs().size()) &&
                            Info().node().InputDefs()[zp_idx]->Exists();
        MlasInitLutGemmKernelConfig(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                    static_cast<size_t>(block_size_), has_zp);
      }

      used_shared_buffers = true;
    }

    return Status::OK();
  }

  if (expert_weight_bits_ != 4) {
    return Status::OK();
  }

  if ((input_idx == 2 || input_idx == 5) && !prepacked_buffers.empty()) {
    if (input_idx == 2) {
      packed_fc1_ = std::move(prepacked_buffers[0]);
      ORT_RETURN_IF_ERROR(parse_shape(fc1_shape_));
      if (prepacked_buffers.size() > 2) {
        packed_fc1_mlas_cache_ = std::move(prepacked_buffers[2]);
      }
    } else if (input_idx == 5) {
      packed_fc2_ = std::move(prepacked_buffers[0]);
      ORT_RETURN_IF_ERROR(parse_shape(fc2_shape_));
      if (prepacked_buffers.size() > 2) {
        packed_fc2_mlas_cache_ = std::move(prepacked_buffers[2]);
      }
    }
    used_shared_buffers = true;
  }

  return Status::OK();
}

template <typename T>
QMoECPU<T>::QMoECPU(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(activation_type_ != ActivationType::SwiGLU || swiglu_fusion_ == 1,
              "CPU QMoE only supports interleaved SwiGLU format. Please set swiglu_fusion=1.");
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 2 || expert_weight_bits_ == 4 || expert_weight_bits_ == 8,
              "Attribute 'expert_weight_bits' must be 2, 4, or 8.");
  fc1_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc1_expert_weight_bits", expert_weight_bits_);
  fc2_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc2_expert_weight_bits", expert_weight_bits_);
  fc3_expert_weight_bits_ = op_kernel_info.GetAttrOrDefault<int64_t>("fc3_expert_weight_bits", expert_weight_bits_);
  ORT_ENFORCE((fc1_expert_weight_bits_ == 2 || fc1_expert_weight_bits_ == 4 || fc1_expert_weight_bits_ == 8) &&
                  (fc2_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 4 || fc2_expert_weight_bits_ == 8) &&
                  (fc3_expert_weight_bits_ == 2 || fc3_expert_weight_bits_ == 4 || fc3_expert_weight_bits_ == 8),
              "FC-specific expert weight bits must be 2, 4, or 8.");
  ORT_ENFORCE(swiglu_fusion_ == 0 || fc3_expert_weight_bits_ == fc1_expert_weight_bits_,
              "Fused SwiGLU requires FC1 and FC3 expert weight bits to match.");
  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", 0);
  ORT_ENFORCE(block_size_ >= 0);

  if (block_size_ > 0) {
    ORT_ENFORCE(block_size_ >= 16, "block_size must be >= 16 when provided.");
    ORT_ENFORCE((block_size_ & (block_size_ - 1)) == 0, "block_size must be a power of 2.");
  }

  const int64_t weights_prepacked_attr = op_kernel_info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
  ORT_ENFORCE(weights_prepacked_attr == -1 || weights_prepacked_attr == 0 || weights_prepacked_attr == 1,
              "weights_prepacked must be -1 (default), 0, or 1, got ", weights_prepacked_attr);
  weights_prepacked_ = (weights_prepacked_attr == 1);
  if (weights_prepacked_) {
    ORT_ENFORCE(expert_weight_bits_ == 4 && block_size_ == qmoe_shared::kBlockSize,
                "CPU QMoE with weights_prepacked=1 requires expert_weight_bits=4 and block_size=32.");
  }

  const auto use_mlas_q4_gemm = ParseEnvironmentVariable<bool>(kUseMlasQ4GemmMoe);
  if (use_mlas_q4_gemm.has_value()) {
    use_mlas_q4_gemm_ = *use_mlas_q4_gemm;
    use_mlas_q4_gemm_overridden_ = true;
  } else {
    // Default policy: enable fast path unless this run hits a known accuracy-loss configuration.
    use_mlas_q4_gemm_ = true;
    use_mlas_q4_gemm_overridden_ = false;
  }

  // QNBit GEMM path policy. The MatMulNBits kernels consume the block-wise QMoE encoding directly
  // (no re-quantization), have NEON/AVX2/AVX512 backends, and thread a single GEMM over N, so they
  // are preferred for block-wise 4/8-bit experts. Int8 activations are lossier than the
  // dequantize+SGEMM path they replace, so for 4-bit they need accuracy_level=4 as in MatMulNBits
  // and the default is fp32 activations. 8-bit has no fp32 kernel; unlike MatMulNBits (which
  // dequantizes at level 0) it uses int8 activations at every level, since that is the only way
  // onto these kernels. The environment variable overrides the attribute.
  accuracy_level_ = op_kernel_info.GetAttrOrDefault<int64_t>("accuracy_level", 0);
  bool allow_int8_compute = (accuracy_level_ == 4);
  bool allow_fp32_compute = true;
  bool fp32_compute_forced = false;
  const auto qnbit_gemm_env = ParseEnvironmentVariable<std::string>(kQMoEQNBitGemmEnv);
  if (qnbit_gemm_env.has_value()) {
    if (*qnbit_gemm_env == "0") {
      use_qnbit_gemm_ = false;
    } else if (*qnbit_gemm_env == "fp32") {
      allow_int8_compute = false;
      fp32_compute_forced = true;
    } else if (*qnbit_gemm_env == "int8") {
      allow_int8_compute = true;
      allow_fp32_compute = false;
    } else {
      ORT_THROW("Unsupported value for ", kQMoEQNBitGemmEnv, ": \"", *qnbit_gemm_env,
                "\" (expected \"0\", \"fp32\" or \"int8\").");
    }
  }
  if (use_qnbit_gemm_ && (expert_weight_bits_ == 4 || expert_weight_bits_ == 8) && block_size_ > 0) {
    const size_t nbits = static_cast<size_t>(expert_weight_bits_);
    const size_t blk = static_cast<size_t>(block_size_);
    const bool fp32_available = MlasIsQNBitGemmAvailable(nbits, blk, SQNBIT_CompFp32);
    const bool int8_available = MlasIsQNBitGemmAvailable(nbits, blk, SQNBIT_CompInt8);
    // int8 is taken when asked for (accuracy_level 4 or the env override), like MatMulNBits, and
    // also where MLAS has no fp32 kernel for the bit width (8-bit) unless fp32 was forced, since
    // int8 is then the only way onto these kernels. fp32 is the fallback whenever it is allowed.
    const bool want_int8 = allow_int8_compute || (!fp32_available && !fp32_compute_forced);
    if (want_int8 && int8_available) {
      qnbit_compute_type_ = SQNBIT_CompInt8;
    } else if (allow_fp32_compute && fp32_available) {
      qnbit_compute_type_ = SQNBIT_CompFp32;
    } else {
      use_qnbit_gemm_ = false;
    }
  } else {
    use_qnbit_gemm_ = false;
  }
}

template <typename T>
bool QMoECPU<T>::QNBitGemmEligible(int input_idx, int64_t num_experts, int64_t rows, int64_t cols,
                                   QNBitEligibility& out) const {
  out = QNBitEligibility{};
  if (!use_qnbit_gemm_ || (input_idx != 2 && input_idx != 5)) {
    return false;
  }
  if (block_size_ <= 0) {
    return false;
  }
  if ((cols % block_size_) != 0) {
    out.ineligible_reason = "the input feature size is not a multiple of block_size";
    return false;
  }

  const int scales_idx = (input_idx == 2) ? 3 : 6;
  const int zp_idx = (input_idx == 2) ? 11 : 12;
  const auto& input_defs = Info().node().InputDefs();
  const int64_t blocks_per_row = cols / block_size_;

  const Tensor* scales_tensor = nullptr;
  if (!Info().TryGetConstantInput(scales_idx, &scales_tensor) || scales_tensor == nullptr) {
    out.ineligible_reason = "the scales are not a constant initializer";
    return false;
  }
  const auto& scales_dims = scales_tensor->Shape().GetDims();
  if (scales_dims.size() != 3 || scales_dims[0] != num_experts || scales_dims[1] != rows ||
      scales_dims[2] != blocks_per_row) {
    return false;  // row-wise (per-channel) scales keep the existing paths
  }

  const bool has_zp_input = zp_idx < static_cast<int>(input_defs.size()) && input_defs[zp_idx]->Exists();
  const Tensor* zp_tensor = nullptr;
  if (has_zp_input) {
    if (!Info().TryGetConstantInput(zp_idx, &zp_tensor) || zp_tensor == nullptr) {
      out.ineligible_reason = "the zero points are not a constant initializer";
      return false;
    }
    // The kernels read the MatMulNBits zero point layout, [rows, ceil(blocks/pack)] per expert with
    // the even block in the low nibble, which is QMoE's rank-3 block-wise layout.
    const int64_t zp_pack = 8 / expert_weight_bits_;
    const auto& zp_dims = zp_tensor->Shape().GetDims();
    if (zp_dims.size() != 3 || zp_dims[0] != num_experts || zp_dims[1] != rows ||
        zp_dims[2] != (blocks_per_row + zp_pack - 1) / zp_pack) {
      out.ineligible_reason = "the zero points are not in the block-wise [num_experts, rows, blocks/pack] layout";
      return false;
    }
  }

  const size_t packed_size = MlasQNBitGemmPackQuantBDataSize(
      static_cast<size_t>(rows), static_cast<size_t>(cols), static_cast<size_t>(expert_weight_bits_),
      static_cast<size_t>(block_size_), has_zp_input, qnbit_compute_type_,
      &mlas_backend_kernel_selector_config_);
  if (packed_size == 0) {
    out.ineligible_reason = "MLAS has no QNBit packing for this shape on this platform";
    return false;
  }

  out.scales = scales_tensor;
  out.zero_points = zp_tensor;
  out.packed_size_per_expert = packed_size;
  return true;
}

template <typename T>
Status QMoECPU<T>::InitQNBitPacked(QNBitPackedExperts& packed, const QNBitEligibility& eligibility,
                                   int64_t num_experts, int64_t rows, int64_t cols, AllocatorPtr alloc) {
  packed.packed_size_per_expert = eligibility.packed_size_per_expert;
  packed.has_zero_point = (eligibility.zero_points != nullptr);
  packed.scales_packed = MlasQNBitGemmScalesPacked(
      static_cast<size_t>(cols), static_cast<size_t>(expert_weight_bits_), static_cast<size_t>(block_size_),
      qnbit_compute_type_, packed.has_zero_point, &mlas_backend_kernel_selector_config_);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    // The MLAS kernels take fp32 scales; convert the constant fp16 scales once.
    const size_t scales_count = static_cast<size_t>(num_experts * rows * (cols / block_size_));
    packed.scales_fp32 = IAllocator::MakeUniquePtr<float>(alloc, scales_count, true);
    MlasConvertHalfToFloatBuffer(eligibility.scales->template Data<MLFloat16>(), packed.scales_fp32.get(), scales_count);
  } else {
    ORT_UNUSED_PARAMETER(num_experts);
  }
  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::PrePackQNBitExperts(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                       /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) {
  const auto& shape = tensor.Shape();
  const int64_t num_experts = shape[0];
  const int64_t rows = shape[1];
  const int64_t pack_unit = 8 / expert_weight_bits_;
  const int64_t cols = shape[2] * pack_unit;

  QNBitEligibility eligibility;
  if (!QNBitGemmEligible(input_idx, num_experts, rows, cols, eligibility)) {
    // The dequantize+SGEMM fallback is an order of magnitude slower for decode, so say why once.
    if (eligibility.ineligible_reason != nullptr && !qnbit_fallback_logged_) {
      qnbit_fallback_logged_ = true;
      LOGS_DEFAULT(WARNING) << "QMoE node '" << Info().node().Name() << "': block-wise expert weights (input "
                            << input_idx << ") cannot use the MLAS QNBit GEMM kernels because "
                            << eligibility.ineligible_reason << "; falling back to dequantize + SGEMM.";
    }
    return Status::OK();
  }

  QNBitPackedExperts& packed = (input_idx == 2) ? qnbit_fc1_ : qnbit_fc2_;
  ORT_RETURN_IF_ERROR(InitQNBitPacked(packed, eligibility, num_experts, rows, cols, alloc));

  const float* scales_fp32 = nullptr;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    scales_fp32 = packed.scales_fp32.get();
  } else {
    scales_fp32 = eligibility.scales->template Data<float>();
  }
  const uint8_t* zp_data = packed.has_zero_point ? eligibility.zero_points->template Data<uint8_t>() : nullptr;
  const size_t zp_stride = packed.has_zero_point
                               ? static_cast<size_t>(eligibility.zero_points->Shape()[1] * eligibility.zero_points->Shape()[2])
                               : 0;

  const size_t nbits = static_cast<size_t>(expert_weight_bits_);
  const size_t blk = static_cast<size_t>(block_size_);
  const size_t per_expert = packed.packed_size_per_expert;
  const size_t total_packed_size = SafeInt<size_t>(per_expert) * static_cast<size_t>(num_experts);
  auto packed_buffer = IAllocator::MakeUniquePtr<void>(alloc, total_packed_size, true);
  // The pack routines need not write every byte; zero so the prepacked-weights hash is deterministic.
  std::memset(packed_buffer.get(), 0, total_packed_size);

  const size_t qdata_stride = static_cast<size_t>(rows * shape[2]);
  const size_t scale_stride = static_cast<size_t>(rows * (cols / block_size_));
  const std::byte* qdata = static_cast<const std::byte*>(tensor.DataRaw());
  std::byte* dst = static_cast<std::byte*>(packed_buffer.get());

  // Same two-step sequence as MatMulNBits::PrePack: pack the quantized data, then (where the
  // platform folds them into B) finalize the scales / block sums with a nullptr QuantBData call.
  bool finalize_scales = (qnbit_compute_type_ == SQNBIT_CompInt8);
#if !defined(MLAS_TARGET_AMD64_IX86)
  finalize_scales = finalize_scales && (nbits == 8 || packed.scales_packed);
#endif

  // Experts pack independently into disjoint regions, so spread them over a load-time pool
  // (the session pool is not reachable from PrePack; same approach as MatMulNBits::PrePack).
  std::unique_ptr<concurrency::ThreadPool> pack_tp;
  const int pack_threads = narrow<int>(std::min<int64_t>(num_experts, Env::Default().GetNumPhysicalCpuCores()));
  if (pack_threads > 1) {
    OrtThreadPoolParams pack_tp_params;
    pack_tp_params.thread_pool_size = pack_threads;
    pack_tp_params.allow_spinning = false;
    pack_tp_params.auto_set_affinity = false;
    pack_tp = concurrency::CreateThreadPool(&Env::Default(), pack_tp_params, concurrency::ThreadPoolType::INTRA_OP);
  }
  concurrency::ThreadPool::TrySimpleParallelFor(pack_tp.get(), narrow<int>(num_experts), [&](std::ptrdiff_t e) {
    std::byte* expert_dst = dst + static_cast<size_t>(e) * per_expert;
    const float* expert_scales = scales_fp32 + static_cast<size_t>(e) * scale_stride;
    const uint8_t* expert_zp = (zp_data == nullptr) ? nullptr : zp_data + static_cast<size_t>(e) * zp_stride;
    MlasQNBitGemmPackQuantBData(static_cast<size_t>(rows), static_cast<size_t>(cols), nbits, blk, qnbit_compute_type_,
                                qdata + static_cast<size_t>(e) * qdata_stride, expert_dst, expert_scales,
                                packed.has_zero_point, expert_zp, nullptr, &mlas_backend_kernel_selector_config_);
    if (finalize_scales) {
      MlasQNBitGemmPackQuantBData(static_cast<size_t>(rows), static_cast<size_t>(cols), nbits, blk, qnbit_compute_type_,
                                  nullptr, expert_dst, expert_scales,
                                  packed.has_zero_point, expert_zp, nullptr, &mlas_backend_kernel_selector_config_);
    }
  });

  if (input_idx == 2) {
    fc1_shape_ = shape;
  } else {
    fc2_shape_ = shape;
  }

  if (prepacked_weights != nullptr) {
    prepacked_weights->buffers_.push_back(std::move(packed_buffer));
    prepacked_weights->buffer_sizes_.push_back(total_packed_size);

    // The layout tag lets UseSharedPrePackedBuffers tell this layout apart from the legacy
    // unpacked-transposed one (rank + dims only); the compute type and zero point flag guard
    // against a buffer packed for different kernels.
    const int64_t trailer[] = {kQNBitPackedLayoutTag, static_cast<int64_t>(qnbit_compute_type_),
                               packed.has_zero_point ? int64_t{1} : int64_t{0}};
    PushPrePackedShapeBuffer(shape, trailer, alloc, *prepacked_weights);
  } else {
    packed.packed = std::move(packed_buffer);
  }

  is_packed = true;
  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::Compute(OpKernelContext* context) const {
  const ComputeInputs inputs{
      context->Input<Tensor>(0),
      context->Input<Tensor>(1),
      ((packed_fc1_ != nullptr) || (packed_fc1_lut_cache_ != nullptr) || (qnbit_fc1_.packed != nullptr)) ? nullptr : context->Input<Tensor>(2),
      context->Input<Tensor>(3),
      context->Input<Tensor>(4),
      ((packed_fc2_ != nullptr) || (packed_fc2_lut_cache_ != nullptr) || (qnbit_fc2_.packed != nullptr)) ? nullptr : context->Input<Tensor>(5),
      context->Input<Tensor>(6),
      context->Input<Tensor>(7),
      context->Input<Tensor>(8),
      context->Input<Tensor>(9),
      context->Input<Tensor>(10),
      context->Input<Tensor>(11),
      context->Input<Tensor>(12),
      context->Input<Tensor>(13),
      context->Input<Tensor>(14),
  };

  const bool has_fc1_prepacked = (packed_fc1_ != nullptr) || (packed_fc1_lut_cache_ != nullptr) || (qnbit_fc1_.packed != nullptr);
  const bool has_fc2_prepacked = (packed_fc2_ != nullptr) || (packed_fc2_lut_cache_ != nullptr) || (qnbit_fc2_.packed != nullptr);

  const TensorShape* fc1_shape_ptr = has_fc1_prepacked ? &fc1_shape_ : (inputs.fc1_experts_weights ? &inputs.fc1_experts_weights->Shape() : nullptr);
  const TensorShape* fc2_shape_ptr = has_fc2_prepacked ? &fc2_shape_ : (inputs.fc2_experts_weights ? &inputs.fc2_experts_weights->Shape() : nullptr);
  const TensorShape* fc3_shape_ptr = inputs.fc3_experts_weights ? &inputs.fc3_experts_weights->Shape() : nullptr;

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(moe_helper::CheckInputs<Tensor>(
      moe_params, inputs.input, inputs.router_probs,
      fc1_shape_ptr, inputs.fc1_experts_bias, inputs.fc1_scales, inputs.fc1_zero_points,
      fc2_shape_ptr, inputs.fc2_experts_bias, inputs.fc2_scales, inputs.fc2_zero_points,
      fc3_shape_ptr, inputs.fc3_experts_bias, inputs.fc3_scales, inputs.fc3_zero_points,
      moe_helper::MoEWeightBits{fc1_expert_weight_bits_,
                                fc2_expert_weight_bits_,
                                fc3_expert_weight_bits_},
      activation_type_ == ActivationType::SwiGLU,
      block_size_));

  if (fc1_expert_weight_bits_ != expert_weight_bits_ ||
      fc2_expert_weight_bits_ != expert_weight_bits_ ||
      fc3_expert_weight_bits_ != expert_weight_bits_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Mixed-width QMoE execution is not yet implemented on CPU.");
  }

  if (fc3_shape_ptr || inputs.fc3_experts_bias || inputs.fc3_scales || inputs.fc3_zero_points) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented on CPU for QMoE");
  }

  return ComputeCommon(context, inputs, moe_params);
}

template <typename T>
Status QMoECPU<T>::ComputeCommon(OpKernelContext* context, const ComputeInputs& inputs, const MoEParameters& moe_params) const {
  const auto* input = inputs.input;
  const auto* router_probs = inputs.router_probs;
  const auto* fc1_experts_weights = inputs.fc1_experts_weights;
  const auto* fc1_scales = inputs.fc1_scales;
  const auto* fc1_experts_bias = inputs.fc1_experts_bias;
  const auto* fc2_experts_weights = inputs.fc2_experts_weights;
  const auto* fc2_scales = inputs.fc2_scales;
  const auto* fc2_experts_bias = inputs.fc2_experts_bias;
  const auto* fc1_zero_points = inputs.fc1_zero_points;
  const auto* fc2_zero_points = inputs.fc2_zero_points;
  const auto* router_weights = inputs.router_weights;

  const auto& input_shape = input->Shape();
  const int64_t num_tokens = moe_params.num_rows;
  const int64_t hidden_size = moe_params.hidden_size;
  const int64_t inter_size = moe_params.inter_size;
  const int64_t num_experts = moe_params.num_experts;
#if !defined(ORT_MINIMAL_BUILD)
  const size_t routing_element_count = enable_moe_expert_tracking_
                                           ? static_cast<size_t>(SafeInt<size_t>(num_tokens) * SafeInt<size_t>(k_))
                                           : 0;
#endif

  ORT_RETURN_IF_NOT(k_ <= num_experts,
                    "QMoE attribute 'k' must be <= num_experts; got k=", k_,
                    ", num_experts=", num_experts);
  const int64_t fc1_out_features = inter_size * (swiglu_fusion_ > 0 ? 2 : 1);

  auto* output = context->Output(0, input_shape);
  auto* tp = context->GetOperatorThreadPool();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const size_t output_buffer_size = static_cast<size_t>(output->Shape().Size());

  const T* input_data = input->template Data<T>();

  IAllocatorUniquePtr<float> router_logits_float_buffer;
  const float* router_logits_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    router_logits_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
    router_logits_float = router_logits_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_probs->template Data<T>()),
                                 const_cast<float*>(router_logits_float),
                                 static_cast<size_t>(num_tokens * num_experts));
  } else {
    router_logits_float = reinterpret_cast<const float*>(router_probs->template Data<T>());
  }

  // Handle optional router_weights input for separate selection/aggregation tensors
  const bool has_router_weights = (router_weights != nullptr);
  IAllocatorUniquePtr<float> router_weights_float_buffer;
  const float* router_weights_float = nullptr;
  if (has_router_weights) {
    const auto& rw_shape = router_weights->Shape();
    if (rw_shape.NumDimensions() != 2 || rw_shape[0] != num_tokens || rw_shape[1] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'router_weights' is expected to have shape (",
                             num_tokens, ", ", num_experts, "), got ", rw_shape);
    }
    if constexpr (std::is_same_v<T, MLFloat16>) {
      router_weights_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
      router_weights_float = router_weights_float_buffer.get();
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_weights->template Data<T>()),
                                   const_cast<float*>(router_weights_float),
                                   static_cast<size_t>(num_tokens * num_experts));
    } else {
      router_weights_float = reinterpret_cast<const float*>(router_weights->template Data<T>());
    }
  }

  auto route_expert_ptr = IAllocator::MakeUniquePtr<int>(allocator, static_cast<size_t>(num_tokens * k_));
  int* route_expert = route_expert_ptr.get();
  auto route_scale_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * k_));
  float* route_scale = route_scale_ptr.get();

  const int max_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
  const int64_t thread_divisor = std::max(1, max_threads * 4);
  const int64_t min_work_per_thread = std::max(int64_t{32}, static_cast<int64_t>(num_tokens / thread_divisor));
  const int optimal_routing_threads = (tp == nullptr || num_tokens < min_work_per_thread) ? 1 : std::min(narrow<int>(num_tokens / std::max(int64_t{1}, min_work_per_thread)), max_threads);
  const int num_routing_threads = std::max(1, optimal_routing_threads);

  std::vector<std::vector<std::vector<int64_t>>> thread_local_expert_token_maps(num_routing_threads);
  for (auto& map : thread_local_expert_token_maps) {
    map.resize(static_cast<size_t>(num_experts));
    for (auto& expert_tokens : map) {
      expert_tokens.reserve(32);
    }
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_routing_threads, [&](std::ptrdiff_t thread_id) {
    auto work = concurrency::ThreadPool::PartitionWork(narrow<int>(thread_id), num_routing_threads, static_cast<std::ptrdiff_t>(num_tokens));
    auto& local_expert_token_map = thread_local_expert_token_maps[thread_id];

    std::vector<std::pair<float, int64_t>> sorted_logits(static_cast<size_t>(num_experts));
    std::vector<float> top_k_exp(static_cast<size_t>(k_));

    for (int64_t i = work.start; i < work.end; ++i) {
      const float* logits = router_logits_float + i * num_experts;

      for (size_t j = 0; j < narrow<size_t>(num_experts); ++j) {
        sorted_logits[j] = {logits[j], j};
      }
      std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + static_cast<std::ptrdiff_t>(k_),
                        sorted_logits.end(), std::greater<>());

      if (has_router_weights) {
        // When router_weights is provided, use it for aggregation weights instead of softmax of router_probs.
        // Gather weights from router_weights at the selected expert indices.
        // Note: top_k_exp is reused here as a scratch buffer for the gathered weights.
        const float* weights_row = router_weights_float + i * num_experts;
        if (normalize_routing_weights_) {
          float weight_sum = 0.0f;
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            top_k_exp[j] = weights_row[expert_idx];
            weight_sum += top_k_exp[j];
          }
          const float inv_weight_sum = (weight_sum == 0.0f) ? 0.0f : (1.0f / weight_sum);
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            int64_t route_idx = i * k_ + narrow<int64_t>(j);
            route_expert[route_idx] = narrow<int>(expert_idx);
            route_scale[route_idx] = top_k_exp[j] * inv_weight_sum;
            if (route_scale[route_idx] > 1e-8f) {
              local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
            }
          }
        } else {
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            int64_t route_idx = i * k_ + narrow<int64_t>(j);
            route_expert[route_idx] = narrow<int>(expert_idx);
            route_scale[route_idx] = weights_row[expert_idx];
            if (route_scale[route_idx] > 1e-8f) {
              local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
            }
          }
        }
      } else {
        // Default path: compute softmax weights from router_probs for aggregation.
        float max_logit = sorted_logits[0].first;

        float sum_exp = 0.0f;
        for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
          top_k_exp[j] = std::exp(sorted_logits[j].first - max_logit);
          sum_exp += top_k_exp[j];
        }

        const float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
        for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
          int64_t expert_idx = sorted_logits[j].second;
          int64_t route_idx = i * k_ + narrow<int64_t>(j);
          route_expert[route_idx] = narrow<int>(expert_idx);
          route_scale[route_idx] = top_k_exp[j] * inv_sum;
          if (route_scale[route_idx] > 1e-8f) {  // Use small threshold to avoid zero weights
            local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
          }
        }
      }
    }
  });

  std::vector<std::vector<int64_t>> expert_token_map(static_cast<size_t>(num_experts));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    size_t total_tokens_for_expert = 0;
    for (int t = 0; t < num_routing_threads; ++t) {
      total_tokens_for_expert += thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)].size();
    }
    expert_token_map[static_cast<size_t>(expert_idx)].reserve(total_tokens_for_expert);

    for (int t = 0; t < num_routing_threads; ++t) {
      auto& local_tokens = thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)];
      if (!local_tokens.empty()) {
        expert_token_map[static_cast<size_t>(expert_idx)].insert(
            expert_token_map[static_cast<size_t>(expert_idx)].end(),
            local_tokens.begin(), local_tokens.end());
      }
    }
  }

  IAllocatorUniquePtr<float> input_float_buffer;
  const float* input_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * hidden_size));
    input_float = input_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(input_data),
                                 const_cast<float*>(input_float),
                                 static_cast<size_t>(num_tokens * hidden_size));
  } else {
    input_float = reinterpret_cast<const float*>(input_data);
  }

  const int max_expert_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;

  // Number of experts that actually received tokens; only these do work in the expert loop.
  int num_active_experts = 0;
  for (const auto& tokens : expert_token_map) {
    if (!tokens.empty()) ++num_active_experts;
  }

  // Nested parallelism (outer expert loop + inner GEMM/dequant on the same pool) can livelock ORT's
  // Eigen pool (see PR #29081), so exactly one level uses the session pool tp: either the expert
  // loop is multi-threaded and the inner ops run serially (inner_tp == nullptr), or the expert loop
  // is serial and the inner ops get the full pool (inner_tp == tp). Which level to thread depends
  // on the GEMM path:
  //  - dequantize+SGEMM / LUT paths: one active expert per thread. For decode (M==1) the per-expert
  //    GEMM is a GEMV that MLAS does not thread internally, so this is the only way to keep the
  //    cores busy.
  //  - QNBit path: the MatMulNBits kernels thread a single GEMM over N (and M), so when fewer experts
  //    are active than there are threads (decode: top_k experts) the expert loop is serialized and the
  //    experts are instead batched into one MLAS dispatch (see the grouped path below), which keeps
  //    the whole pool busy without paying a thread-pool barrier per expert.
  // This must be decided BEFORE the per-thread workspaces below, which are sized by num_expert_threads.
  int num_expert_threads = std::max(1, std::min(num_active_experts, max_expert_threads));
  if (qnbit_fc1_.packed != nullptr && qnbit_fc2_.packed != nullptr && num_active_experts < max_expert_threads) {
    num_expert_threads = 1;
  }
  concurrency::ThreadPool* inner_tp = (num_expert_threads > 1) ? nullptr : tp;

  auto thread_local_outputs_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * output_buffer_size);
  float* thread_local_outputs = thread_local_outputs_ptr.get();
  std::memset(thread_local_outputs, 0, static_cast<size_t>(num_expert_threads) * output_buffer_size * sizeof(float));

  size_t max_tokens_per_expert = 0;
  for (const auto& tokens : expert_token_map) {
    max_tokens_per_expert = std::max(max_tokens_per_expert, tokens.size());
  }

  const auto align_size = [](size_t size) -> size_t {
    return (size + 63) & ~63;
  };

  const size_t A1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t C1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(fc1_out_features));
  const size_t A2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(inter_size));
  const size_t C2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t B1_dequant_size = (qnbit_fc1_.packed != nullptr) ? 0 : align_size(static_cast<size_t>(fc1_out_features) * static_cast<size_t>(hidden_size));
  const size_t B2_dequant_size = (qnbit_fc2_.packed != nullptr) ? 0 : align_size(static_cast<size_t>(hidden_size) * static_cast<size_t>(inter_size));

  const size_t workspace_elements_per_thread = A1_size + C1_size + A2_size + C2_size +
                                               B1_dequant_size + B2_dequant_size;

  auto workspace_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * workspace_elements_per_thread);
  float* workspace = workspace_ptr.get();

  auto bias_conversion_buffers_ptr = IAllocator::MakeUniquePtr<float>(allocator,
                                                                      static_cast<size_t>(num_expert_threads) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size)));
  float* bias_conversion_buffers = bias_conversion_buffers_ptr.get();

  const auto& fc1_scales_dims = fc1_scales->Shape().GetDims();
  const auto& fc2_scales_dims = fc2_scales->Shape().GetDims();
  const bool is_fc1_block_wise = block_size_ > 0 && fc1_scales_dims.size() == 3;
  const bool is_fc2_block_wise = block_size_ > 0 && fc2_scales_dims.size() == 3;

  const bool use_qnbit_fc1 = (qnbit_fc1_.packed != nullptr);
  const bool use_qnbit_fc2 = (qnbit_fc2_.packed != nullptr);
  const uint8_t* fc1_weights_data = (packed_fc1_ != nullptr || packed_fc1_lut_cache_ != nullptr || use_qnbit_fc1) ? nullptr : fc1_experts_weights->template Data<uint8_t>();
  const uint8_t* fc2_weights_data = (packed_fc2_ != nullptr || packed_fc2_lut_cache_ != nullptr || use_qnbit_fc2) ? nullptr : fc2_experts_weights->template Data<uint8_t>();
  IAllocatorUniquePtr<uint8_t> unpacked_fc1_weights_buf;
  IAllocatorUniquePtr<uint8_t> unpacked_fc2_weights_buf;
  if (weights_prepacked_) {
    if (fc1_weights_data != nullptr) {
      ORT_RETURN_IF_NOT((fc1_out_features % qmoe_shared::kColumnsInterleaved) == 0 &&
                            (hidden_size % qmoe_shared::kRowsPerTile) == 0,
                        "QMoE CPU weights_prepacked=1 requires SM80 tile-aligned FC1 dimensions.");
      const size_t expert_bytes = static_cast<size_t>(fc1_out_features * (hidden_size / 2));
      unpacked_fc1_weights_buf = IAllocator::MakeUniquePtr<uint8_t>(
          allocator, static_cast<size_t>(num_experts) * expert_bytes, true);
      for (int64_t e = 0; e < num_experts; ++e) {
        qmoe_shared::UnpackSm80Int4Weights(fc1_weights_data + static_cast<size_t>(e) * expert_bytes,
                                           unpacked_fc1_weights_buf.get() + static_cast<size_t>(e) * expert_bytes,
                                           fc1_out_features, hidden_size);
      }
      fc1_weights_data = unpacked_fc1_weights_buf.get();
    }
    if (fc2_weights_data != nullptr) {
      ORT_RETURN_IF_NOT((hidden_size % qmoe_shared::kColumnsInterleaved) == 0 &&
                            (inter_size % qmoe_shared::kRowsPerTile) == 0,
                        "QMoE CPU weights_prepacked=1 requires SM80 tile-aligned FC2 dimensions.");
      const size_t expert_bytes = static_cast<size_t>(hidden_size * (inter_size / 2));
      unpacked_fc2_weights_buf = IAllocator::MakeUniquePtr<uint8_t>(
          allocator, static_cast<size_t>(num_experts) * expert_bytes, true);
      for (int64_t e = 0; e < num_experts; ++e) {
        qmoe_shared::UnpackSm80Int4Weights(fc2_weights_data + static_cast<size_t>(e) * expert_bytes,
                                           unpacked_fc2_weights_buf.get() + static_cast<size_t>(e) * expert_bytes,
                                           hidden_size, inter_size);
      }
      fc2_weights_data = unpacked_fc2_weights_buf.get();
    }
  }
  const T* fc1_scales_data = fc1_scales->template Data<T>();
  const T* fc2_scales_data = fc2_scales->template Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->template Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->template Data<T>() : nullptr;
  const uint8_t* fc1_zp_data = fc1_zero_points ? fc1_zero_points->template Data<uint8_t>() : nullptr;
  const uint8_t* fc2_zp_data = fc2_zero_points ? fc2_zero_points->template Data<uint8_t>() : nullptr;

  // Known loss-prone case from parity testing: 4-bit symmetric path (row-wise and block-wise).
  const bool known_accuracy_loss_case = (expert_weight_bits_ == 4) &&
                                        (fc1_zp_data == nullptr) && (fc2_zp_data == nullptr);
  const bool use_mlas_q4_gemm_effective = use_mlas_q4_gemm_overridden_
                                              ? use_mlas_q4_gemm_
                                              : (use_mlas_q4_gemm_ && !known_accuracy_loss_case);
  const bool use_mlas_lut_gemm_effective = (expert_weight_bits_ == 2);

  const int64_t pack_unit = (8 / expert_weight_bits_);
  const int64_t fc1_packed_cols = (hidden_size + pack_unit - 1) / pack_unit;
  const int64_t fc2_packed_cols = (inter_size + pack_unit - 1) / pack_unit;
  const bool has_fc1_bias = (fc1_bias_data != nullptr);
  const bool has_fc2_bias = (fc2_bias_data != nullptr);

  // Calculate strides for zero-point tensors
  const int64_t zp_pack_size = 8 / expert_weight_bits_;
  int64_t fc1_zp_expert_stride = 0;
  int64_t fc2_zp_expert_stride = 0;

  if (is_fc1_block_wise) {
    const int64_t fc1_blocks_per_row = (hidden_size + block_size_ - 1) / block_size_;
    const int64_t fc1_zp_blocks_packed = (fc1_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    fc1_zp_expert_stride = fc1_out_features * fc1_zp_blocks_packed;
  } else {
    fc1_zp_expert_stride = (fc1_out_features + zp_pack_size - 1) / zp_pack_size;
  }

  if (is_fc2_block_wise) {
    const int64_t fc2_blocks_per_row = (inter_size + block_size_ - 1) / block_size_;
    const int64_t fc2_zp_blocks_packed = (fc2_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    fc2_zp_expert_stride = hidden_size * fc2_zp_blocks_packed;
  } else {
    fc2_zp_expert_stride = (hidden_size + zp_pack_size - 1) / zp_pack_size;
  }

  MLAS_BLK_QUANT_TYPE fc1_direct_qtype = BlkQ4Sym;
  MLAS_BLK_QUANT_TYPE fc2_direct_qtype = BlkQ4Sym;
  const bool can_use_fc1_lut_gemm = use_mlas_lut_gemm_effective &&
                                    is_fc1_block_wise &&
                                    CanUseMlasLutGemm(expert_weight_bits_, block_size_, fc1_out_features, hidden_size);
  const bool can_use_fc2_lut_gemm = use_mlas_lut_gemm_effective &&
                                    is_fc2_block_wise &&
                                    CanUseMlasLutGemm(expert_weight_bits_, block_size_, hidden_size, inter_size);

  if (can_use_fc1_lut_gemm) {
    MlasInitLutGemmKernelConfig(static_cast<size_t>(fc1_out_features), static_cast<size_t>(hidden_size), 2,
                                static_cast<size_t>(block_size_), fc1_zp_data != nullptr);
  }
  if (can_use_fc2_lut_gemm) {
    MlasInitLutGemmKernelConfig(static_cast<size_t>(hidden_size), static_cast<size_t>(inter_size), 2,
                                static_cast<size_t>(block_size_), fc2_zp_data != nullptr);
  }

  // Use pre-packed MLAS cache if available
  const void* fc1_direct_q4_cache_ptr = nullptr;
  if (use_mlas_q4_gemm_effective && packed_fc1_mlas_cache_ && fc1_zp_data == nullptr &&
      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0, fc1_out_features, hidden_size, fc1_direct_qtype)) {
    fc1_direct_q4_cache_ptr = packed_fc1_mlas_cache_.get();
  }

  const void* fc2_direct_q4_cache_ptr = nullptr;
  if (use_mlas_q4_gemm_effective && packed_fc2_mlas_cache_ && fc2_zp_data == nullptr &&
      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0, hidden_size, inter_size, fc2_direct_qtype)) {
    fc2_direct_q4_cache_ptr = packed_fc2_mlas_cache_.get();
  }

  const void* fc1_direct_lut_cache_ptr = can_use_fc1_lut_gemm ? packed_fc1_lut_cache_.get() : nullptr;
  const void* fc2_direct_lut_cache_ptr = can_use_fc2_lut_gemm ? packed_fc2_lut_cache_.get() : nullptr;
  const size_t fc1_lut_packed_size_per_expert = can_use_fc1_lut_gemm
                                                    ? MlasLutGemmPackedSize(static_cast<size_t>(fc1_out_features),
                                                                            static_cast<size_t>(hidden_size),
                                                                            2,
                                                                            static_cast<size_t>(block_size_),
                                                                            fc1_zp_data != nullptr)
                                                    : 0;
  const size_t fc2_lut_packed_size_per_expert = can_use_fc2_lut_gemm
                                                    ? MlasLutGemmPackedSize(static_cast<size_t>(hidden_size),
                                                                            static_cast<size_t>(inter_size),
                                                                            2,
                                                                            static_cast<size_t>(block_size_),
                                                                            fc2_zp_data != nullptr)
                                                    : 0;
  const size_t fc1_lut_packed_size = (fc1_direct_lut_cache_ptr == nullptr) ? fc1_lut_packed_size_per_expert : 0;
  const size_t fc2_lut_packed_size = (fc2_direct_lut_cache_ptr == nullptr) ? fc2_lut_packed_size_per_expert : 0;
  const size_t lut_packed_scratch_size_per_thread = std::max(fc1_lut_packed_size, fc2_lut_packed_size);
  IAllocatorUniquePtr<std::byte> lut_packed_buffers_ptr;
  std::byte* lut_packed_buffers = nullptr;
  if (lut_packed_scratch_size_per_thread > 0) {
    lut_packed_buffers_ptr = IAllocator::MakeUniquePtr<std::byte>(allocator,
                                                                  static_cast<size_t>(num_expert_threads) * lut_packed_scratch_size_per_thread,
                                                                  true);
    lut_packed_buffers = lut_packed_buffers_ptr.get();
  }

  const size_t lut_scale_count_per_thread = std::max(static_cast<size_t>(is_fc1_block_wise ? fc1_out_features * fc1_scales_dims[2] : 0),
                                                     static_cast<size_t>(is_fc2_block_wise ? hidden_size * fc2_scales_dims[2] : 0));
  IAllocatorUniquePtr<float> lut_scale_conversion_buffers_ptr;
  float* lut_scale_conversion_buffers = nullptr;
  if constexpr (!std::is_same_v<T, float>) {
    if (lut_scale_count_per_thread > 0) {
      lut_scale_conversion_buffers_ptr = IAllocator::MakeUniquePtr<float>(allocator,
                                                                          static_cast<size_t>(num_expert_threads) * lut_scale_count_per_thread,
                                                                          true);
      lut_scale_conversion_buffers = lut_scale_conversion_buffers_ptr.get();
    }
  }

  // One expert GEMM on the MLAS QNBit (MatMulNBits) kernels: C[M, N] = A[M, K] * dequant(B_expert)^T + bias.
  const size_t qnbit_bits = static_cast<size_t>(expert_weight_bits_);
  const size_t qnbit_blk = static_cast<size_t>(std::max<int64_t>(block_size_, 1));
  // The CompInt8 kernels need a per-GEMM workspace (quantized A); size it for the largest expert
  // batch once per thread rather than allocating per expert.
  size_t qnbit_workspace_per_thread = 0;
  if (use_qnbit_fc1) {
    const size_t fc1_workspace = MlasQNBitGemmBatchWorkspaceSize(
        static_cast<size_t>(max_tokens_per_expert), static_cast<size_t>(fc1_out_features), static_cast<size_t>(hidden_size),
        1, qnbit_bits, qnbit_blk, qnbit_fc1_.has_zero_point, qnbit_compute_type_, &mlas_backend_kernel_selector_config_);
    qnbit_workspace_per_thread = std::max(qnbit_workspace_per_thread, fc1_workspace);
  }
  if (use_qnbit_fc2) {
    const size_t fc2_workspace = MlasQNBitGemmBatchWorkspaceSize(
        static_cast<size_t>(max_tokens_per_expert), static_cast<size_t>(hidden_size), static_cast<size_t>(inter_size),
        1, qnbit_bits, qnbit_blk, qnbit_fc2_.has_zero_point, qnbit_compute_type_, &mlas_backend_kernel_selector_config_);
    qnbit_workspace_per_thread = std::max(qnbit_workspace_per_thread, fc2_workspace);
  }
  IAllocatorUniquePtr<std::byte> qnbit_workspace_ptr;
  std::byte* qnbit_workspace = nullptr;
  if (qnbit_workspace_per_thread > 0) {
    // Arena-backed (no reserve): this is per-call scratch that should be recycled between runs.
    qnbit_workspace_ptr = IAllocator::MakeUniquePtr<std::byte>(allocator, static_cast<size_t>(num_expert_threads) * qnbit_workspace_per_thread);
    qnbit_workspace = qnbit_workspace_ptr.get();
  }
  // fp32 [E, N, K/block] scales for the kernels: the scales input itself for T == float, the copy
  // converted at PrePack for T == MLFloat16.
  const float* qnbit_fc1_scales = nullptr;
  const float* qnbit_fc2_scales = nullptr;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    qnbit_fc1_scales = qnbit_fc1_.scales_fp32.get();
    qnbit_fc2_scales = qnbit_fc2_.scales_fp32.get();
  } else {
    qnbit_fc1_scales = fc1_scales_data;
    qnbit_fc2_scales = fc2_scales_data;
  }
  auto run_qnbit_gemm = [&](const QNBitPackedExperts& packed, const float* scales_base,
                            const uint8_t* zp_base, int64_t zp_expert_stride, int64_t expert_idx,
                            const float* A, size_t M, size_t N, size_t K,
                            const float* bias, float* C,
                            std::byte* gemm_workspace, concurrency::ThreadPool* gemm_tp) {
    const float* scales = packed.scales_packed
                              ? nullptr
                              : scales_base + static_cast<size_t>(expert_idx) * N * (K / qnbit_blk);
    const uint8_t* zero_points = packed.has_zero_point
                                     ? zp_base + static_cast<size_t>(expert_idx * zp_expert_stride)
                                     : nullptr;
    const std::byte* packed_b = static_cast<const std::byte*>(packed.packed.get()) +
                                static_cast<size_t>(expert_idx) * packed.packed_size_per_expert;

    MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
    params.A = A;
    params.lda = K;
    params.QuantBDataWorkspace = packed_b;
    params.PackedQuantBData = packed_b;
    params.QuantBScale = scales;
    params.QuantBZeroPoint = zero_points;
    params.Bias = bias;
    params.C = C;
    params.ldc = N;
    MlasQNBitGemmBatch<float>(M, N, K, 1, qnbit_bits, qnbit_blk, qnbit_compute_type_, &params,
                              gemm_workspace, gemm_tp, &mlas_backend_kernel_selector_config_);
  };

  // Grouped expert GEMMs: batch the active experts into as few MLAS dispatches as possible rather
  // than one per expert, removing the 2 * num_active_experts thread-pool barriers per layer. MLAS
  // takes a single M/N/K per batch, so experts are bucketed by token count; decode (one token per
  // active expert) collapses to a single bucket. This is only reached when the expert loop is
  // already serial: with at least as many active experts as threads, running one expert per thread
  // above is faster than batching.
  InlinedVector<int64_t> grouped_experts;
  bool use_grouped_qnbit = use_qnbit_fc1 && use_qnbit_fc2 && num_expert_threads == 1 &&
                           activation_type_ == ActivationType::SwiGLU;
  if (use_grouped_qnbit) {
    grouped_experts.reserve(static_cast<size_t>(num_active_experts));
    SafeInt<size_t> total_grouped_rows = 0;
    for (int64_t i = 0; i < num_experts; ++i) {
      const size_t expert_rows = expert_token_map[static_cast<size_t>(i)].size();
      if (expert_rows > 0) {
        grouped_experts.push_back(i);
        total_grouped_rows += expert_rows;
      }
    }

    // Bound the additional route-wide staging matrices. Large prefills retain the per-expert
    // path, whose scratch size is based on the largest expert rather than all routed rows.
    constexpr size_t kMaxGroupedStagingBytes = 8 * 1024 * 1024;
    const size_t staging_bytes_per_row = SafeInt<size_t>(hidden_size) * 2 * sizeof(float) +
                                         SafeInt<size_t>(inter_size) * 3 * sizeof(float);
    use_grouped_qnbit = grouped_experts.size() > 1 &&
                        total_grouped_rows <= kMaxGroupedStagingBytes / staging_bytes_per_row;
  }

  if (use_grouped_qnbit) {
    const size_t num_grouped = grouped_experts.size();
    const size_t n1 = static_cast<size_t>(fc1_out_features);
    const size_t k1 = static_cast<size_t>(hidden_size);
    const size_t n2 = static_cast<size_t>(hidden_size);
    const size_t k2 = static_cast<size_t>(inter_size);

    // Experts sharing a token count must be contiguous so each bucket is one batched call.
    std::sort(grouped_experts.begin(), grouped_experts.end(), [&](int64_t a, int64_t b) {
      return expert_token_map[static_cast<size_t>(a)].size() < expert_token_map[static_cast<size_t>(b)].size();
    });
    InlinedVector<size_t> row_offset(num_grouped + 1, 0);
    for (size_t g = 0; g < num_grouped; ++g) {
      row_offset[g + 1] = row_offset[g] + expert_token_map[static_cast<size_t>(grouped_experts[g])].size();
    }
    const size_t total_rows = row_offset[num_grouped];

    auto a1_all = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(total_rows) * k1);
    auto c1_all = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(total_rows) * n1);
    auto a2_all = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(total_rows) * k2);
    auto c2_all = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(total_rows) * n2);

    // fp16 bias needs one fp32 copy per active expert; fp32 bias is passed straight from the initializer.
    IAllocatorUniquePtr<float> grouped_bias;
    if constexpr (std::is_same_v<T, MLFloat16>) {
      if (has_fc1_bias || has_fc2_bias) {
        grouped_bias = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(num_grouped) * (n1 + n2));
        for (size_t g = 0; g < num_grouped; ++g) {
          const int64_t e = grouped_experts[g];
          float* dst = grouped_bias.get() + g * (n1 + n2);
          if (has_fc1_bias) {
            MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(fc1_bias_data + e * fc1_out_features), dst, n1);
          }
          if (has_fc2_bias) {
            MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(fc2_bias_data + e * hidden_size), dst + n1, n2);
          }
        }
      }
    }

    concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(num_grouped), [&](std::ptrdiff_t g_idx) {
      const size_t g = static_cast<size_t>(g_idx);
      const auto& routes = expert_token_map[static_cast<size_t>(grouped_experts[g])];
      float* dst = a1_all.get() + row_offset[g] * k1;
      for (size_t i = 0; i < routes.size(); ++i) {
        const int64_t token_idx = routes[i] / k_;
        std::memcpy(dst + i * k1, input_float + token_idx * hidden_size, k1 * sizeof(float));
      }
    });

    // [begin, end) ranges over grouped_experts that share a token count.
    InlinedVector<std::pair<size_t, size_t>> buckets;
    for (size_t b0 = 0; b0 < num_grouped;) {
      const size_t rows = expert_token_map[static_cast<size_t>(grouped_experts[b0])].size();
      size_t b1 = b0 + 1;
      while (b1 < num_grouped && expert_token_map[static_cast<size_t>(grouped_experts[b1])].size() == rows) {
        ++b1;
      }
      buckets.emplace_back(b0, b1);
      b0 = b1;
    }

    size_t grouped_ws_size = 0;
    for (const auto& bucket : buckets) {
      const size_t rows = expert_token_map[static_cast<size_t>(grouped_experts[bucket.first])].size();
      const size_t count = bucket.second - bucket.first;
      grouped_ws_size = std::max({grouped_ws_size,
                                  MlasQNBitGemmBatchWorkspaceSize(rows, n1, k1, count, qnbit_bits, qnbit_blk,
                                                                  qnbit_fc1_.has_zero_point, qnbit_compute_type_,
                                                                  &mlas_backend_kernel_selector_config_),
                                  MlasQNBitGemmBatchWorkspaceSize(rows, n2, k2, count, qnbit_bits, qnbit_blk,
                                                                  qnbit_fc2_.has_zero_point, qnbit_compute_type_,
                                                                  &mlas_backend_kernel_selector_config_)});
    }
    IAllocatorUniquePtr<std::byte> grouped_ws;
    if (grouped_ws_size > 0) {
      grouped_ws = IAllocator::MakeUniquePtr<std::byte>(allocator, grouped_ws_size);
    }

    InlinedVector<MLAS_QNBIT_GEMM_DATA_PARAMS<float>> gemm_params(num_grouped);
    auto run_buckets = [&](const QNBitPackedExperts& packed, const float* scales_base, const uint8_t* zp_base,
                           int64_t zp_expert_stride, size_t n, size_t k, const float* a_all, float* c_all,
                           bool is_fc1) {
      for (const auto& bucket : buckets) {
        const size_t rows = expert_token_map[static_cast<size_t>(grouped_experts[bucket.first])].size();
        const size_t count = bucket.second - bucket.first;
        for (size_t g = bucket.first; g < bucket.second; ++g) {
          const int64_t e = grouped_experts[g];
          auto& p = gemm_params[g - bucket.first];
          p = MLAS_QNBIT_GEMM_DATA_PARAMS<float>{};
          p.A = a_all + row_offset[g] * k;
          p.lda = k;
          const std::byte* b = static_cast<const std::byte*>(packed.packed.get()) +
                               static_cast<size_t>(e) * packed.packed_size_per_expert;
          p.QuantBDataWorkspace = b;
          p.PackedQuantBData = b;
          p.QuantBScale = packed.scales_packed ? nullptr : scales_base + static_cast<size_t>(e) * n * (k / qnbit_blk);
          p.QuantBZeroPoint = packed.has_zero_point ? zp_base + static_cast<size_t>(e * zp_expert_stride) : nullptr;
          if (is_fc1 ? has_fc1_bias : has_fc2_bias) {
            if constexpr (std::is_same_v<T, MLFloat16>) {
              p.Bias = grouped_bias.get() + g * (n1 + n2) + (is_fc1 ? 0 : n1);
            } else {
              const T* bias_base = is_fc1 ? fc1_bias_data : fc2_bias_data;
              p.Bias = reinterpret_cast<const float*>(bias_base) + static_cast<size_t>(e) * n;
            }
          }
          p.C = c_all + row_offset[g] * n;
          p.ldc = n;
        }
        MlasQNBitGemmBatch<float>(rows, n, k, count, qnbit_bits, qnbit_blk, qnbit_compute_type_,
                                  gemm_params.data(), grouped_ws.get(), tp,
                                  &mlas_backend_kernel_selector_config_);
      }
    };

    run_buckets(qnbit_fc1_, qnbit_fc1_scales, fc1_zp_data, fc1_zp_expert_stride, n1, k1,
                a1_all.get(), c1_all.get(), /*is_fc1*/ true);

    concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(total_rows), [&](std::ptrdiff_t idx) {
      const size_t row = static_cast<size_t>(idx);
      ApplySwiGLUActivation(c1_all.get() + row * n1, a2_all.get() + row * k2,
                            inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
    });

    run_buckets(qnbit_fc2_, qnbit_fc2_scales, fc2_zp_data, fc2_zp_expert_stride, n2, k2,
                a2_all.get(), c2_all.get(), /*is_fc1*/ false);

    for (size_t g = 0; g < num_grouped; ++g) {
      const auto& routes = expert_token_map[static_cast<size_t>(grouped_experts[g])];
      const float* src = c2_all.get() + row_offset[g] * n2;
      for (size_t i = 0; i < routes.size(); ++i) {
        const int64_t route_idx = routes[i];
        const int64_t token_idx = route_idx / k_;
        if (token_idx < 0 || token_idx >= num_tokens) continue;
        const size_t buffer_offset = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_size);
        if (buffer_offset + static_cast<size_t>(hidden_size) > output_buffer_size) continue;
        const float weight = route_scale[route_idx];
        float* dest = thread_local_outputs + buffer_offset;
        for (int64_t j = 0; j < hidden_size; ++j) {
          dest[j] += weight * src[i * n2 + static_cast<size_t>(j)];
        }
      }
    }
  }

  // The grouped path above already produced every active expert's contribution.
  std::vector<std::pair<int64_t, size_t>> expert_workload;
  if (!use_grouped_qnbit) {
    for (int64_t i = 0; i < num_experts; ++i) {
      const size_t token_count = expert_token_map[static_cast<size_t>(i)].size();
      if (token_count > 0) {
        expert_workload.emplace_back(i, token_count);
      }
    }
  }

  std::sort(expert_workload.begin(), expert_workload.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<std::vector<int64_t>> expert_batches(num_expert_threads);
  size_t thread_idx = 0;
  for (const auto& work : expert_workload) {
    expert_batches[thread_idx].push_back(work.first);
    thread_idx = (thread_idx + 1) % static_cast<size_t>(num_expert_threads);
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_expert_threads, [&](std::ptrdiff_t thread_id_pd) {
    const int thread_id = narrow<int>(thread_id_pd);
    const auto& expert_batch = expert_batches[static_cast<size_t>(thread_id)];

    float* thread_workspace = workspace + static_cast<size_t>(thread_id) * workspace_elements_per_thread;
    std::byte* thread_lut_packed_buffer = (lut_packed_buffers == nullptr)
                                              ? nullptr
                                              : (lut_packed_buffers + static_cast<size_t>(thread_id) * lut_packed_scratch_size_per_thread);
    float* thread_lut_scale_buffer = (lut_scale_conversion_buffers == nullptr)
                                         ? nullptr
                                         : (lut_scale_conversion_buffers + static_cast<size_t>(thread_id) * lut_scale_count_per_thread);

    float* thread_bias1_buffer = bias_conversion_buffers + static_cast<size_t>(thread_id) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size));
    float* thread_bias2_buffer = thread_bias1_buffer + static_cast<size_t>(fc1_out_features);
    std::byte* thread_qnbit_workspace = (qnbit_workspace == nullptr)
                                            ? nullptr
                                            : (qnbit_workspace + static_cast<size_t>(thread_id) * qnbit_workspace_per_thread);

    for (int64_t expert_idx : expert_batch) {
      bool fc2_bias_added_by_mlas = false;
      const auto& routes = expert_token_map[static_cast<size_t>(expert_idx)];
      if (routes.empty()) {
        continue;
      }

      const int64_t num_expert_tokens = static_cast<int64_t>(routes.size());

      float* A1 = thread_workspace;
      float* C1 = A1 + A1_size;
      float* A2 = C1 + C1_size;
      float* C2 = A2 + A2_size;
      float* B1_dequant = C2 + C2_size;
      float* B2_dequant = B1_dequant + B1_dequant_size;

      const int64_t dynamic_block_size = GetOptimalBlockSize(num_expert_tokens, inner_tp ? concurrency::ThreadPool::DegreeOfParallelism(inner_tp) : 1);
      const int64_t num_blocks = (num_expert_tokens + dynamic_block_size - 1) / dynamic_block_size;

      if (num_expert_tokens >= 8 && num_blocks > 1 && inner_tp != nullptr) {
        concurrency::ThreadPool::TrySimpleParallelFor(inner_tp, narrow<int>(num_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_idx = block_idx * dynamic_block_size;
          const int64_t end_idx = std::min(start_idx + dynamic_block_size, num_expert_tokens);

          for (int64_t i = start_idx; i < end_idx; ++i) {
            const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
            const float* src = input_float + token_idx * hidden_size;
            float* dst = A1 + i * hidden_size;

            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          }
        });
      } else {
        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
          const float* src = input_float + token_idx * hidden_size;
          float* dst = A1 + i * hidden_size;

          if (ShouldUseMemcpy(hidden_size)) {
            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
            size_t j = 0;
            for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
              for (size_t k = 0; k < unroll_factor; ++k) {
                dst[j + k] = src[j + k];
              }
            }
            for (; j < narrow<size_t>(hidden_size); ++j) {
              dst[j] = src[j];
            }
          }
        }
      }

      const T* fc1_scales_ptr;
      const uint8_t* fc1_zp_ptr;

      if (is_fc1_block_wise) {
        const int64_t fc1_blocks_per_row = fc1_scales_dims[2];
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features * fc1_blocks_per_row;
        fc1_zp_ptr = (fc1_zp_data == nullptr) ? nullptr : fc1_zp_data + expert_idx * fc1_zp_expert_stride;
      } else {
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features;
        fc1_zp_ptr = (fc1_zp_data == nullptr) ? nullptr : fc1_zp_data + expert_idx * fc1_zp_expert_stride;
      }

      // When row-wise ZP is present, align block size to zp_pack_size so that parallel
      // shards start at packed-byte boundaries for correct zero-point lane indexing.
      const int64_t fc1_dequant_alignment = (!is_fc1_block_wise && fc1_zp_ptr != nullptr) ? zp_pack_size : 1;
      const int64_t dequant_block_size = GetDequantBlockSize(fc1_out_features, num_expert_tokens, fc1_dequant_alignment);
      const int64_t num_dequant_blocks = (fc1_out_features + dequant_block_size - 1) / dequant_block_size;

      const size_t m = static_cast<size_t>(num_expert_tokens);
      const size_t n = static_cast<size_t>(fc1_out_features);
      const size_t k = static_cast<size_t>(hidden_size);

      MLAS_BLK_QUANT_TYPE q_type = BlkQ4Sym;  // Initialize to default
      bool use_direct_q4_gemm = use_mlas_q4_gemm_effective &&
                                ((fc1_direct_q4_cache_ptr != nullptr) ||
                                 ((packed_fc1_ == nullptr) && (fc1_zp_data == nullptr) &&
                                  CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0,
                                                   fc1_out_features, hidden_size, q_type)));

      if (use_qnbit_fc1) {
        float* fc1_bias_float = nullptr;
        if (has_fc1_bias) {
          const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
          if constexpr (std::is_same_v<T, MLFloat16>) {
            MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
          } else {
            std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
          }
          fc1_bias_float = thread_bias1_buffer;
        }
        run_qnbit_gemm(qnbit_fc1_, qnbit_fc1_scales, fc1_zp_data, fc1_zp_expert_stride, expert_idx, A1, m, n, k,
                       fc1_bias_float, C1, thread_qnbit_workspace, inner_tp);
        goto fc1_gemm_done;
      }

      if (can_use_fc1_lut_gemm &&
          TryRunLutGemm(A1, C1, fc1_weights_data,
                        fc1_direct_lut_cache_ptr != nullptr
                            ? static_cast<const void*>(static_cast<const std::byte*>(fc1_direct_lut_cache_ptr) + expert_idx * fc1_lut_packed_size_per_expert)
                            : nullptr,
                        fc1_scales_ptr, fc1_zp_ptr, expert_idx,
                        fc1_out_features, hidden_size, fc1_packed_cols,
                        block_size_, fc1_scales_dims[2],
                        thread_lut_packed_buffer, thread_lut_scale_buffer,
                        num_expert_tokens, inner_tp)) {
        goto fc1_bias_handling;
      }

      if (packed_fc1_ != nullptr) {
        if (use_mlas_q4_gemm_effective && fc1_zp_data == nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0,
                             fc1_out_features, hidden_size, q_type)) {
          if (fc1_direct_q4_cache_ptr != nullptr) {
            float* fc1_bias_float = nullptr;
            if (has_fc1_bias) {
              const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
              if constexpr (std::is_same_v<T, MLFloat16>) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
              } else {
                std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
              }
              fc1_bias_float = thread_bias1_buffer;
            }

            size_t packed_size = MlasQ4GemmPackBSize(q_type, static_cast<size_t>(fc1_out_features), static_cast<size_t>(hidden_size));
            const uint8_t* packed_b = static_cast<const uint8_t*>(fc1_direct_q4_cache_ptr) + expert_idx * packed_size;

            Status gemm_status = DirectQ4Gemm(A1, packed_b, fc1_bias_float, C1,
                                              num_expert_tokens, fc1_out_features, hidden_size, fc1_direct_qtype, inner_tp);
            if (gemm_status.IsOK()) {
              goto fc1_gemm_done;
            }
          }
        }

        // Fallback: Dequantize from PrePacked (transposed, unpacked) -> MlasGemm
        const uint8_t* current_packed_ptr = static_cast<const uint8_t*>(packed_fc1_.get()) + expert_idx * fc1_out_features * hidden_size;

        DequantizePrePacked(current_packed_ptr, fc1_scales_ptr, fc1_zp_ptr,
                            is_fc1_block_wise ? block_size_ : 0,
                            fc1_out_features, hidden_size,
                            B1_dequant, fc1_scales_dims);

        // Use MlasGemm with B1_dequant (which is already float transposed)
        MlasGemm(CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f, A1, k,
                 B1_dequant, n,
                 0.0f, C1, n,
                 inner_tp, &mlas_backend_kernel_selector_config_);

        goto fc1_bias_handling;
      }

      if (use_direct_q4_gemm) {
        IAllocatorUniquePtr<uint8_t> mlas_packed_fc1;
        Status convert_status = ConvertToMlasQ4Format(
            fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
            fc1_scales_ptr,
            fc1_zp_ptr,  // This will be nullptr
            is_fc1_block_wise ? block_size_ : 0,
            expert_weight_bits_,
            fc1_out_features,
            hidden_size,
            q_type,
            allocator,
            mlas_packed_fc1);

        if (convert_status.IsOK()) {
          float* fc1_bias_float = nullptr;

          if (has_fc1_bias) {
            const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
            fc1_bias_float = thread_bias1_buffer;

            if constexpr (std::is_same_v<T, MLFloat16>) {
              MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), fc1_bias_float, static_cast<size_t>(fc1_out_features));
            } else {
              for (int64_t i = 0; i < fc1_out_features; ++i) {
                fc1_bias_float[i] = static_cast<float>(B1_bias[i]);
              }
            }
          }

          Status gemm_status = DirectQ4Gemm(A1, mlas_packed_fc1.get(), fc1_bias_float, C1,
                                            num_expert_tokens, fc1_out_features, hidden_size, q_type, inner_tp);

          if (gemm_status.IsOK()) {
            goto fc1_gemm_done;
          }
        }
        // If direct Q4 GEMM failed, fall back to traditional approach
      }

      // Traditional approach: dequantize + regular GEMM
      if (num_dequant_blocks > 1 && fc1_out_features >= 32) {
        concurrency::ThreadPool::TrySimpleParallelFor(inner_tp, narrow<int>(num_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * dequant_block_size;
          const int64_t end_row = std::min(start_row + dequant_block_size, fc1_out_features);
          const auto offset = expert_idx * fc1_out_features * fc1_packed_cols + start_row * fc1_packed_cols;

          const T* current_scales_ptr = fc1_scales_ptr + (is_fc1_block_wise ? start_row * fc1_scales_dims[2] : start_row);
          const uint8_t* current_zp_ptr = nullptr;
          if (fc1_zp_ptr != nullptr) {
            if (is_fc1_block_wise) {
              const int64_t fc1_blocks_per_row = (hidden_size + block_size_ - 1) / block_size_;
              const int64_t fc1_zp_blocks_packed = (fc1_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
              current_zp_ptr = fc1_zp_ptr + start_row * fc1_zp_blocks_packed;
            } else {
              current_zp_ptr = fc1_zp_ptr + start_row / zp_pack_size;
            }
          }

          DequantizeBlock(fc1_weights_data + offset,
                          current_scales_ptr,
                          current_zp_ptr,
                          is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, hidden_size, B1_dequant + start_row * hidden_size, inner_tp);
        });
      } else {
        DequantizeBlock(fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
                        fc1_scales_ptr,
                        fc1_zp_ptr,
                        is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                        fc1_out_features, hidden_size, B1_dequant, inner_tp);
      }

      MlasGemm(CblasNoTrans, CblasTrans,
               m, n, k,
               1.0f, A1, k,
               B1_dequant, k,
               0.0f, C1, n,
               inner_tp, &mlas_backend_kernel_selector_config_);

    fc1_bias_handling:

      if (has_fc1_bias) {
        const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
        } else {
          if (ShouldUseMemcpy(fc1_out_features)) {
            std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
          } else {
            const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(fc1_out_features));
            size_t j = 0;
            for (; j + unroll_factor <= static_cast<size_t>(fc1_out_features); j += unroll_factor) {
              for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
                thread_bias1_buffer[j + loop_k] = static_cast<float>(B1_bias[j + loop_k]);
              }
            }
            for (; j < static_cast<size_t>(fc1_out_features); ++j) {
              thread_bias1_buffer[j] = static_cast<float>(B1_bias[j]);
            }
          }
        }

        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          float* C1_row = C1 + i * fc1_out_features;
          const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(fc1_out_features));

          size_t j = 0;
          for (; j + unroll_factor <= static_cast<size_t>(fc1_out_features); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              C1_row[j + loop_k] += thread_bias1_buffer[j + loop_k];
            }
          }
          for (; j < static_cast<size_t>(fc1_out_features); ++j) {
            C1_row[j] += thread_bias1_buffer[j];
          }
        }
      }

    fc1_gemm_done:

      if (activation_type_ == ActivationType::SwiGLU) {
        const int64_t activation_threshold = std::max(int64_t{4}, 256 / std::max(int64_t{1}, inter_size));
        if (num_expert_tokens >= activation_threshold && inner_tp != nullptr) {
          const int64_t activation_block_size = std::max(int64_t{1}, std::min(int64_t{64}, activation_threshold));
          const int64_t num_activation_blocks = (num_expert_tokens + activation_block_size - 1) / activation_block_size;

          if (num_activation_blocks > 1) {
            concurrency::ThreadPool::TrySimpleParallelFor(inner_tp, narrow<int>(num_activation_blocks), [&](std::ptrdiff_t block_idx) {
              const int64_t start_token = block_idx * activation_block_size;
              const int64_t end_token = std::min(start_token + activation_block_size, num_expert_tokens);

              for (int64_t i = start_token; i < end_token; ++i) {
                const float* C1_token = C1 + i * fc1_out_features;
                float* A2_token = A2 + i * inter_size;
                ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
              }
            });
          } else {
            for (int64_t i = 0; i < num_expert_tokens; ++i) {
              const float* C1_token = C1 + i * fc1_out_features;
              float* A2_token = A2 + i * inter_size;
              ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
            }
          }
        } else {
          for (int64_t i = 0; i < num_expert_tokens; ++i) {
            const float* C1_token = C1 + i * fc1_out_features;
            float* A2_token = A2 + i * inter_size;
            ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
          }
        }
      } else {
        ApplyActivationVectorized(C1, num_expert_tokens * fc1_out_features);
        std::copy(C1, C1 + (num_expert_tokens * fc1_out_features), A2);
      }

      const T* fc2_scales_ptr;
      const uint8_t* fc2_zp_ptr;

      if (is_fc2_block_wise) {
        const int64_t fc2_blocks_per_row = fc2_scales_dims[2];
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size * fc2_blocks_per_row;
        fc2_zp_ptr = (fc2_zp_data == nullptr) ? nullptr : fc2_zp_data + expert_idx * fc2_zp_expert_stride;
      } else {
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size;
        fc2_zp_ptr = (fc2_zp_data == nullptr) ? nullptr : fc2_zp_data + expert_idx * fc2_zp_expert_stride;
      }

      // When row-wise ZP is present, align block size to zp_pack_size for correct lane indexing.
      const int64_t fc2_dequant_alignment = (!is_fc2_block_wise && fc2_zp_ptr != nullptr) ? zp_pack_size : 1;
      const int64_t fc2_dequant_block_size = GetDequantBlockSize(hidden_size, num_expert_tokens, fc2_dequant_alignment);
      const int64_t num_fc2_dequant_blocks = (hidden_size + fc2_dequant_block_size - 1) / fc2_dequant_block_size;

      const size_t m2 = static_cast<size_t>(num_expert_tokens);
      const size_t n2 = static_cast<size_t>(hidden_size);
      const size_t k2 = static_cast<size_t>(inter_size);

      MLAS_BLK_QUANT_TYPE q_type2 = BlkQ4Sym;  // Initialize to default
      bool use_direct_q4_gemm_fc2 = use_mlas_q4_gemm_effective &&
                                    ((fc2_direct_q4_cache_ptr != nullptr) ||
                                     ((packed_fc2_ == nullptr) && (fc2_zp_data == nullptr) &&
                                      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0,
                                                       hidden_size, inter_size, q_type2)));

      if (use_qnbit_fc2) {
        float* fc2_bias_float = nullptr;
        if (has_fc2_bias) {
          const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
          if constexpr (std::is_same_v<T, MLFloat16>) {
            MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
          } else {
            std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
          }
          fc2_bias_float = thread_bias2_buffer;
        }
        run_qnbit_gemm(qnbit_fc2_, qnbit_fc2_scales, fc2_zp_data, fc2_zp_expert_stride, expert_idx, A2, m2, n2, k2,
                       fc2_bias_float, C2, thread_qnbit_workspace, inner_tp);
        fc2_bias_added_by_mlas = true;
        goto fc2_gemm_done;
      }

      if (can_use_fc2_lut_gemm &&
          TryRunLutGemm(A2, C2, fc2_weights_data,
                        fc2_direct_lut_cache_ptr != nullptr
                            ? static_cast<const void*>(static_cast<const std::byte*>(fc2_direct_lut_cache_ptr) + expert_idx * fc2_lut_packed_size_per_expert)
                            : nullptr,
                        fc2_scales_ptr, fc2_zp_ptr, expert_idx,
                        hidden_size, inter_size, fc2_packed_cols,
                        block_size_, fc2_scales_dims[2],
                        thread_lut_packed_buffer, thread_lut_scale_buffer,
                        num_expert_tokens, inner_tp)) {
        goto fc2_gemm_done;
      }

      if (packed_fc2_ != nullptr) {
        if (use_mlas_q4_gemm_effective && fc2_zp_data == nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0,
                             hidden_size, inter_size, q_type2)) {
          if (fc2_direct_q4_cache_ptr != nullptr) {
            float* fc2_bias_float = nullptr;
            if (has_fc2_bias) {
              const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
              if constexpr (std::is_same_v<T, MLFloat16>) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
              } else {
                std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
              }
              fc2_bias_float = thread_bias2_buffer;
            }

            size_t packed_size = MlasQ4GemmPackBSize(q_type2, static_cast<size_t>(hidden_size), static_cast<size_t>(inter_size));
            const uint8_t* packed_b = static_cast<const uint8_t*>(fc2_direct_q4_cache_ptr) + expert_idx * packed_size;

            Status gemm_status = DirectQ4Gemm(A2, packed_b, fc2_bias_float, C2,
                                              num_expert_tokens, hidden_size, inter_size, fc2_direct_qtype, inner_tp);
            if (gemm_status.IsOK()) {
              fc2_bias_added_by_mlas = true;
              goto fc2_gemm_done;
            }
          }
        }

        // Dequantize from PrePacked (transposed, unpacked)
        const uint8_t* current_packed_ptr = static_cast<const uint8_t*>(packed_fc2_.get()) + expert_idx * hidden_size * inter_size;

        DequantizePrePacked(current_packed_ptr, fc2_scales_ptr, fc2_zp_ptr,
                            is_fc2_block_wise ? block_size_ : 0,
                            hidden_size, inter_size,
                            B2_dequant, fc2_scales_dims);

        // Fallback
        MlasGemm(CblasNoTrans, CblasNoTrans,
                 m2, n2, k2,
                 1.0f, A2, k2,
                 B2_dequant, n2,
                 0.0f, C2, n2,
                 inner_tp, &mlas_backend_kernel_selector_config_);

        goto fc2_gemm_done;
      }

      if (use_direct_q4_gemm_fc2) {
        IAllocatorUniquePtr<uint8_t> mlas_packed_fc2;
        Status convert_status = ConvertToMlasQ4Format(
            fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
            fc2_scales_ptr,
            fc2_zp_ptr,  // This will be nullptr
            is_fc2_block_wise ? block_size_ : 0,
            expert_weight_bits_,
            hidden_size,
            inter_size,
            q_type2,
            allocator,
            mlas_packed_fc2);

        if (convert_status.IsOK()) {
          float* fc2_bias_float = nullptr;

          if (has_fc2_bias) {
            const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
            fc2_bias_float = thread_bias2_buffer;

            if constexpr (std::is_same_v<T, MLFloat16>) {
              MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), fc2_bias_float, static_cast<size_t>(hidden_size));
            } else {
              for (int64_t i = 0; i < hidden_size; ++i) {
                fc2_bias_float[i] = static_cast<float>(B2_bias[i]);
              }
            }
          }

          Status gemm_status = DirectQ4Gemm(A2, mlas_packed_fc2.get(), fc2_bias_float, C2,
                                            num_expert_tokens, hidden_size, inter_size, q_type2, inner_tp);

          if (gemm_status.IsOK()) {
            fc2_bias_added_by_mlas = true;
            goto fc2_gemm_done;
          }
        }

        // If direct Q4 GEMM failed, fall back to traditional approach
      }

      // Traditional approach: dequantize + regular GEMM
      if (num_fc2_dequant_blocks > 1 && hidden_size >= 32) {
        concurrency::ThreadPool::TrySimpleParallelFor(inner_tp, narrow<int>(num_fc2_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * fc2_dequant_block_size;
          const int64_t end_row = std::min(start_row + fc2_dequant_block_size, hidden_size);
          const auto offset = expert_idx * hidden_size * fc2_packed_cols + start_row * fc2_packed_cols;

          const T* current_scales_ptr = fc2_scales_ptr + (is_fc2_block_wise ? start_row * fc2_scales_dims[2] : start_row);
          const uint8_t* current_zp_ptr = nullptr;
          if (fc2_zp_ptr != nullptr) {
            if (is_fc2_block_wise) {
              const int64_t fc2_blocks_per_row = (inter_size + block_size_ - 1) / block_size_;
              const int64_t fc2_zp_blocks_packed = (fc2_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
              current_zp_ptr = fc2_zp_ptr + start_row * fc2_zp_blocks_packed;
            } else {
              current_zp_ptr = fc2_zp_ptr + start_row / zp_pack_size;
            }
          }

          DequantizeBlock(fc2_weights_data + offset,
                          current_scales_ptr,
                          current_zp_ptr,
                          is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, inter_size, B2_dequant + start_row * inter_size, inner_tp);
        });
      } else {
        DequantizeBlock(fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
                        fc2_scales_ptr,
                        fc2_zp_ptr,
                        is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                        hidden_size, inter_size, B2_dequant, inner_tp);
      }

      MlasGemm(CblasNoTrans, CblasTrans,
               m2, n2, k2,
               1.0f, A2, k2,
               B2_dequant, k2,
               0.0f, C2, n2,
               inner_tp, &mlas_backend_kernel_selector_config_);

    fc2_gemm_done:

      if (has_fc2_bias && !fc2_bias_added_by_mlas) {
        const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
        } else {
          if (ShouldUseMemcpy(hidden_size)) {
            std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
            size_t j = 0;
            for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
              for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
                thread_bias2_buffer[j + loop_k] = static_cast<float>(B2_bias[j + loop_k]);
              }
            }
            for (; j < narrow<size_t>(hidden_size); ++j) {
              thread_bias2_buffer[j] = static_cast<float>(B2_bias[j]);
            }
          }
        }
      }

      for (int64_t i = 0; i < num_expert_tokens; ++i) {
        const int64_t route_idx = routes[static_cast<size_t>(i)];
        const int64_t token_idx = route_idx / k_;
        const float weight = route_scale[route_idx];

        if (token_idx < 0 || token_idx >= num_tokens) continue;

        const size_t buffer_offset = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_size);
        if (buffer_offset + static_cast<size_t>(hidden_size) > output_buffer_size) continue;

        float* dest = thread_local_outputs + static_cast<size_t>(thread_id) * output_buffer_size + buffer_offset;
        const float* src = C2 + i * hidden_size;

        if (has_fc2_bias && !fc2_bias_added_by_mlas) {
          const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
          size_t j = 0;
          for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dest[j + loop_k] += weight * (src[j + loop_k] + thread_bias2_buffer[j + loop_k]);
            }
          }
          for (; j < narrow<size_t>(hidden_size); ++j) {
            dest[j] += weight * (src[j] + thread_bias2_buffer[j]);
          }
        } else {
          const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
          size_t j = 0;
          for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dest[j + loop_k] += weight * src[j + loop_k];
            }
          }
          for (; j < narrow<size_t>(hidden_size); ++j) {
            dest[j] += weight * src[j];
          }
        }
      }
    }
  });

  auto accumulate = [&](float* buffer) {
    std::memset(buffer, 0, output_buffer_size * sizeof(float));

    const int max_acc_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
    const size_t acc_thread_divisor = std::max(size_t{1}, static_cast<size_t>(max_acc_threads) * 8);
    const size_t min_elements_per_thread = std::max(size_t{32}, output_buffer_size / acc_thread_divisor);
    const int optimal_acc_threads = (tp == nullptr || output_buffer_size < min_elements_per_thread) ? 1 : std::min(narrow<int>(output_buffer_size / std::max(size_t{1}, min_elements_per_thread)), max_acc_threads);
    const int num_acc_threads = std::max(1, optimal_acc_threads);

    if (num_acc_threads > 1) {
      concurrency::ThreadPool::TrySimpleParallelFor(tp, num_acc_threads, [&](std::ptrdiff_t acc_thread_id) {
        const size_t elements_per_thread = output_buffer_size / static_cast<size_t>(num_acc_threads);
        const size_t start_idx = static_cast<size_t>(acc_thread_id) * elements_per_thread;
        const size_t end_idx = (acc_thread_id == num_acc_threads - 1) ? output_buffer_size : start_idx + elements_per_thread;

        for (int i = 0; i < num_expert_threads; ++i) {
          const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
          const float* src = thread_local_outputs + thread_offset + start_idx;
          float* dst = buffer + start_idx;

          size_t j = 0;
          const size_t chunk_size = end_idx - start_idx;
          const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(static_cast<int64_t>(chunk_size)));
          for (; j + unroll_factor <= chunk_size; j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dst[j + loop_k] += src[j + loop_k];
            }
          }
          for (; j < chunk_size; ++j) {
            dst[j] += src[j];
          }
        }
      });
    } else {
      for (int i = 0; i < num_expert_threads; ++i) {
        const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
        const float* src = thread_local_outputs + thread_offset;

        size_t j = 0;
        const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(narrow<int64_t>(output_buffer_size)));
        for (; j + unroll_factor <= output_buffer_size; j += unroll_factor) {
          for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
            buffer[j + loop_k] += src[j + loop_k];
          }
        }
        for (; j < output_buffer_size; ++j) {
          buffer[j] += src[j];
        }
      }
    }
  };

  if constexpr (std::is_same_v<T, MLFloat16>) {
    auto final_output_float_ptr = IAllocator::MakeUniquePtr<float>(allocator, output_buffer_size);
    float* final_output_float = final_output_float_ptr.get();
    accumulate(final_output_float);

    MlasConvertFloatToHalfBuffer(final_output_float,
                                 reinterpret_cast<MLFloat16*>(output->template MutableData<T>()),
                                 static_cast<size_t>(output_buffer_size));
  } else {
    accumulate(output->template MutableData<T>());
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (enable_moe_expert_tracking_) {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "MoE expert tracking is enabled but its collector is unavailable.");
    auto& usage = pilot->Moe();
    ORT_RETURN_IF_ERROR(usage.BeginInvocation(static_cast<size_t>(num_experts)));
    ORT_RETURN_IF_ERROR(usage.Collect(gsl::make_span(route_expert, routing_element_count)));
  }
#endif

  return Status::OK();
}

template <typename T>
void QMoECPU<T>::ApplyActivationVectorized(float* data, int64_t size) const {
  for (int64_t i = 0; i < size; ++i) {
    data[i] = ApplyActivation(data[i], activation_type_);
  }
}

template QMoECPU<float>::QMoECPU(const OpKernelInfo& op_kernel_info);

template Status QMoECPU<float>::Compute(OpKernelContext* context) const;
template Status QMoECPU<float>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed, PrePackedWeights* prepacked_weights);
template Status QMoECPU<float>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, gsl::span<const size_t> prepacked_buffer_sizes, int input_idx, bool& used_shared_buffers);
template QMoECPU<MLFloat16>::QMoECPU(const OpKernelInfo& op_kernel_info);
template Status QMoECPU<MLFloat16>::Compute(OpKernelContext* context) const;
template Status QMoECPU<MLFloat16>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed, PrePackedWeights* prepacked_weights);
template Status QMoECPU<MLFloat16>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, gsl::span<const size_t> prepacked_buffer_sizes, int input_idx, bool& used_shared_buffers);

// Kernel Registration
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, float, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    QMoECPU<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, MLFloat16, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()),
    QMoECPU<MLFloat16>);

}  // namespace contrib
}  // namespace onnxruntime
