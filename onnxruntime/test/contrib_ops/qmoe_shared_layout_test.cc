// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "gtest/gtest.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

namespace onnxruntime::test {
namespace {

namespace shared = contrib::qmoe_shared;

TEST(QMoESharedLayoutTest, StorageRangeChecksOffsetEvenWithoutExternalLength) {
  constexpr auto max_offset = std::numeric_limits<uint64_t>::max();
  size_t bytes = 123;
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 0, &bytes));
  EXPECT_EQ(bytes, 123U);
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 512));
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 0, &bytes));
  EXPECT_EQ(bytes, 512U);
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 512));
}

TEST(QMoESharedLayoutTest, TensorExpertStoreRejectsEmptyOrNullBuffers) {
  std::vector<uint8_t> raw(512 + shared::kBankAlignment, 0);
  const uintptr_t addr =
      (reinterpret_cast<uintptr_t>(raw.data()) + shared::kBankAlignment - 1) & ~(shared::kBankAlignment - 1);
  auto* aligned_ptr = reinterpret_cast<uint8_t*>(addr);
  shared::TensorExpertStore store;
  EXPECT_FALSE(store.Initialize(nullptr, 512, 2, 256, 256, 0));
  EXPECT_FALSE(store.Initialize(nullptr, 512, 2, 256, 256, 42));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 0, 2, 256, 256, 0));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 0, 2, 256, 256, 42));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 256, 2, 256, 256, 42));
  EXPECT_TRUE(store.Initialize(aligned_ptr, 512, 2, 256, 256, 0));
}

TEST(QMoESharedLayoutTest, LargeAlignmentRequirementsSupportedInTensorAndFileStores) {
  constexpr size_t kLargeAlign = 8192;
  constexpr size_t kNumExperts = 2;
  constexpr size_t kTotalBytes = kNumExperts * kLargeAlign;
  void* raw = std::malloc(kTotalBytes + kLargeAlign);
  ASSERT_NE(raw, nullptr);
  const uintptr_t addr = (reinterpret_cast<uintptr_t>(raw) + kLargeAlign - 1) & ~(kLargeAlign - 1);
  auto* aligned_ptr = reinterpret_cast<uint8_t*>(addr);
  std::fill_n(aligned_ptr, kLargeAlign, uint8_t{0x12});
  std::fill_n(aligned_ptr + kLargeAlign, kLargeAlign, uint8_t{0x34});

  EXPECT_GE(shared::ComputePointerAlignment(aligned_ptr), kLargeAlign);
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(shared::MakeExpertBlockLease(aligned_ptr, kLargeAlign, kLargeAlign, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);

  shared::TensorExpertStore tensor_store;
  ASSERT_TRUE(tensor_store.Initialize(aligned_ptr, kTotalBytes, kNumExperts, kLargeAlign, kLargeAlign));
  ASSERT_TRUE(tensor_store.AcquireReady(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x12);
  ASSERT_TRUE(tensor_store.AcquireReady(1, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x34);

  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_align8192_test.bin";
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(aligned_ptr), static_cast<std::streamsize>(kTotalBytes));
  }
  shared::FileExpertStore file_store;
  ASSERT_TRUE(file_store.Initialize(tmp_path.string().c_str(), 0, kNumExperts, kLargeAlign, 2, kLargeAlign));
  ASSERT_TRUE(file_store.AcquireReady(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x12);
  file_store.ReleaseLease(0);
  ASSERT_TRUE(file_store.TryAcquireCached(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  file_store.ReleaseLease(0);
  const int32_t group_id = 0;
  ASSERT_TRUE(file_store.TryAcquireGroupCached(&group_id, 1, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  file_store.ReleaseGroupLeases(&group_id, 1);
  file_store.Shutdown();
  std::filesystem::remove(tmp_path);
  std::free(raw);
}

TEST(QMoESharedLayoutTest, FileStoreRetainsOpenHandleAcrossPathReplacement) {
  constexpr size_t kStride = shared::kBankAlignment;
  constexpr int64_t kNumExperts = 2;
  const auto dir = std::filesystem::temp_directory_path();
  const auto orig_path = dir / "ort_qmoe_handle_orig.bin";
  const auto moved_path = dir / "ort_qmoe_handle_moved.bin";
  std::filesystem::remove(orig_path);
  std::filesystem::remove(moved_path);

  {
    std::vector<uint8_t> initial(kNumExperts * kStride);
    std::fill_n(initial.data(), kStride, uint8_t{0x11});
    std::fill_n(initial.data() + kStride, kStride, uint8_t{0x22});
    std::ofstream out(orig_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(initial.data()), static_cast<std::streamsize>(initial.size()));
  }

  shared::FileExpertStore store;
  ASSERT_TRUE(store.Initialize(orig_path.string().c_str(), 0, kNumExperts, kStride, /*cache_slots=*/1));
  EXPECT_TRUE(store.IsBankFileBacked(0));
  EXPECT_TRUE(store.AreAllBanksFileBacked());

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 0x11);
  store.ReleaseLease(0);

  // Rename the active file path and write a replacement file with different contents at orig_path.
  std::error_code ec;
  std::filesystem::rename(orig_path, moved_path, ec);
  ASSERT_FALSE(ec) << ec.message();
  {
    std::vector<uint8_t> replacement(kNumExperts * kStride, uint8_t{0x99});
    std::ofstream out(orig_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(replacement.data()), static_cast<std::streamsize>(replacement.size()));
  }

  // Subsequent cache miss on expert 1 must read from the retained handle (0x22), never the replacement (0x99).
  ASSERT_TRUE(store.AcquireReady(1, &lease));
  EXPECT_EQ(lease.data[0], 0x22);
  store.ReleaseLease(1);
  store.Shutdown();

  std::filesystem::remove(orig_path, ec);
  std::filesystem::remove(moved_path, ec);
}

TEST(QMoESharedLayoutTest, PrefetchCoalescesDuplicatesRefreshesGenerationAndBoundsQueue) {
  constexpr size_t kStride = shared::kBankAlignment;
  constexpr int64_t kNumExperts = 6;
  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_prefetch_coalesce.bin";
  {
    std::vector<uint8_t> data(static_cast<size_t>(kNumExperts) * kStride);
    for (int64_t e = 0; e < kNumExperts; ++e) {
      std::fill_n(data.data() + static_cast<size_t>(e) * kStride, kStride, static_cast<uint8_t>(e + 10));
    }
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  shared::FileExpertStore store;
  ASSERT_TRUE(store.Initialize(tmp_path.string().c_str(), 0, kNumExperts, kStride, /*cache_slots=*/2));
  const std::array<int32_t, 6> dup_batch{0, 0, 0, 1, 1, 0};
  ASSERT_TRUE(store.Prefetch(dup_batch.data(), dup_batch.size(), 1));
  const std::array<int32_t, 4> gen2_batch{2, 3, 2, 3};
  ASSERT_TRUE(store.Prefetch(gen2_batch.data(), gen2_batch.size(), 2));
  // Refresh expert 0 to generation 3 and enqueue expert 5 at generation 3.
  const std::array<int32_t, 4> gen3_batch{0, 5, 0, 5};
  ASSERT_TRUE(store.Prefetch(gen3_batch.data(), gen3_batch.size(), 3));
  store.WaitPrefetchIdle();

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 10);
  store.ReleaseLease(0);
  ASSERT_TRUE(store.AcquireReady(5, &lease));
  EXPECT_EQ(lease.data[0], 15);
  store.ReleaseLease(5);

  store.Shutdown();
  std::filesystem::remove(tmp_path);
}

TEST(QMoESharedLayoutTest, HalfConversionPreservesSignedZero) {
  const std::array<uint16_t, 2> bits{0x0000, 0x8000};
  for (bool scalar : {true, false}) {
    std::array<float, 2> output{};
    shared::HalfBitsToFloatRow(bits.data(), output.data(), bits.size(), scalar);
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_FALSE(std::signbit(output[0]));
    EXPECT_EQ(output[1], 0.0f);
    EXPECT_TRUE(std::signbit(output[1]));
  }
}

void CheckPartialInt4Ranges(bool scalar) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr float sentinel = 12345.0f;
  shared::ExpertBlockLayout layout;
  ASSERT_TRUE(shared::MakeExpertBlockLayout(hidden, inter, &layout));
  std::vector<uint8_t> block(layout.stride);
  auto* weights = block.data() + layout.fc2_weights;
  auto* scales = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_scales);
  auto* bias = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_bias);
  for (int64_t i = 0; i < hidden * inter / 2; ++i) weights[i] = static_cast<uint8_t>(i * 67 + i / 13);
  for (int64_t i = 0; i < hidden * inter / 32; ++i) scales[i] = shared::FloatToHalf((i % 7 + 1) / 64.0f);
  for (int64_t i = 0; i < hidden; ++i) bias[i] = shared::FloatToHalf((i % 11 - 5) / 32.0f);

  const std::array<std::pair<int64_t, int64_t>, 8> ranges{
      {{0, 0}, {0, 1}, {1, 4}, {1, 5}, {4, 7}, {124, 127}, {126, 128}, {0, 128}}};
  for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
    std::vector<float> input(static_cast<size_t>(rows * inter));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 17) - 8) / 64.0f;
    std::vector<float> expected(static_cast<size_t>(rows * hidden));
    shared::Fc2Range(layout, block.data(), input.data(), rows, expected.data(), 0, hidden, true);
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " range=[" << begin << "," << end << ")");
      std::vector<float> output(static_cast<size_t>(rows * hidden + 2), sentinel);
      shared::Fc2Range(layout, block.data(), input.data(), rows, output.data() + 1, begin, end, scalar);
      EXPECT_EQ(output.front(), sentinel);
      EXPECT_EQ(output.back(), sentinel);
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < hidden; ++col) {
          const size_t index = static_cast<size_t>(row * hidden + col);
          if (col >= begin && col < end) {
            EXPECT_NEAR(output[index + 1], expected[index], 1e-5f);
          } else {
            EXPECT_EQ(output[index + 1], sentinel);
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, ScalarPartialInt4RangesPreserveOtherColumns) {
  CheckPartialInt4Ranges(true);
}

TEST(QMoESharedLayoutTest, Avx512PartialInt4RangesPreserveOtherColumns) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  CheckPartialInt4Ranges(false);
}

TEST(QMoESharedLayoutTest, Avx512BFloat16ConversionMatchesScalarIncludingNaNs) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  const std::array<uint32_t, 18> patterns{
      0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000,
      0x00000001, 0x00008000, 0x00018000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
      0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00000, 0xFFC00000};
  std::array<float, 36> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const auto bits = patterns[i % patterns.size()];
    std::memcpy(&input[i], &bits, sizeof(bits));
  }
  std::array<uint16_t, 36> scalar{}, simd{};
  shared::FloatToBFloat16BitsRow(input.data(), scalar.data(), input.size(), true);
  shared::FloatToBFloat16BitsRow(input.data(), simd.data(), input.size(), false);
  EXPECT_EQ(simd, scalar);
  EXPECT_EQ(simd[12], 0x7FC1);
  EXPECT_EQ(simd[13], 0x7FC1);
}

TEST(QMoESharedLayoutTest, Avx512Fp4ScalesMatchScalarIncludingNonfiniteValues) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  constexpr int64_t k = 32;
  constexpr int64_t columns = 9;  // Exercise vector groups and the scalar tail.
  std::array<uint8_t, columns * k / 2> weights;
  weights.fill(0x22);  // E2M1 1.0 in both nibbles.
  std::array<uint8_t, columns * k / 16> scales{};
  std::array<float, shared::kMaxRowsPerCall * k> input;
  input.fill(1.0f);
  std::array<float, shared::kMaxRowsPerCall * columns> scalar{}, simd{};
  for (bool mxfp4 : {false, true}) {
    for (bool bf16 : {false, true}) {
      for (int code = 0; code < 256; ++code) {
        scales.fill(static_cast<uint8_t>(code));
        for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
          SCOPED_TRACE(testing::Message() << "mxfp4=" << mxfp4 << " bf16=" << bf16
                                          << " scale=" << code << " rows=" << rows);
          const auto gemm = mxfp4 ? shared::Mxfp4GemmRange : shared::Nvfp4GemmRange;
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               scalar.data(), columns, 0, columns, true, bf16);
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               simd.data(), columns, 0, columns, false, bf16);
          for (int64_t i = 0; i < rows * columns; ++i) {
            if (std::isnan(scalar[i])) {
              EXPECT_TRUE(std::isnan(simd[i])) << i;
            } else {
              EXPECT_EQ(simd[i], scalar[i]) << i;
            }
          }
        }
      }
    }
  }
}

#if defined(__linux__)
TEST(QMoESharedLayoutTest, MappedStoreHonorsPinsAndRollsBackFailedGroups) {
  constexpr size_t stride = shared::kBankAlignment;
  alignas(shared::kBankAlignment) std::array<uint8_t, 3 * stride> backing{};
  for (size_t i = 0; i < backing.size(); ++i) backing[i] = static_cast<uint8_t>(i / stride + 1);
  shared::FileExpertStore::MappedBank bank{backing.data(), stride, 0};
  shared::FileExpertStore store;
  ASSERT_TRUE(store.InitializeMapped(&bank, 1, 3, stride, 2));
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 1);
  EXPECT_FALSE(store.Evict(0));
  const std::array<int32_t, 2> group{1, 3};
  std::array<shared::ExpertBlockLease, 2> group_leases{};
  EXPECT_FALSE(store.AcquireGroupReady(group.data(), group.size(), group_leases.data()));
  EXPECT_TRUE(store.Evict(1));
  store.ReleaseLease(0);
  const int32_t prefetch = 2;
  ASSERT_TRUE(store.Prefetch(&prefetch, 1, 1));
  store.WaitPrefetchIdle();
  ASSERT_TRUE(store.TryAcquireCached(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
  auto* replacement = store.BeginReplace(2, 1);
  ASSERT_NE(replacement, nullptr);
  EXPECT_FALSE(store.TryAcquireCached(1, &lease));
  std::fill_n(replacement, stride, uint8_t{42});
  store.FinishReplace(1, true);
  ASSERT_TRUE(store.TryAcquireCached(1, &lease));
  EXPECT_EQ(lease.data[0], 42);
  store.ReleaseLease(1);
  ASSERT_NE(store.BeginReplace(1, 2), nullptr);
  store.FinishReplace(2, false);
  EXPECT_FALSE(store.TryAcquireCached(2, &lease));
  ASSERT_TRUE(store.AcquireReady(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
}

TEST(QMoESharedLayoutTest, ModifiedPrivateReadOnlyMappingExcludedFromReloadAndEviction) {
  const long page_raw = sysconf(_SC_PAGESIZE);
  const size_t page_size = (page_raw > 0) ? static_cast<size_t>(page_raw) : 4096u;
  const size_t expert_bytes = page_size * 2;
  constexpr int64_t num_experts = 2;
  const size_t total_bytes = static_cast<size_t>(num_experts) * expert_bytes;

  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_private_map_test.bin";
  {
    std::vector<uint8_t> file_bytes(total_bytes, uint8_t{0x11});
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
  }

  const int fd = open(tmp_path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);

  void* priv_map = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(priv_map, MAP_FAILED);
  auto* priv_bytes = static_cast<uint8_t*>(priv_map);
  std::fill_n(priv_bytes, expert_bytes, uint8_t{0x22});
  std::fill_n(priv_bytes + expert_bytes, expert_bytes, uint8_t{0x33});
  ASSERT_EQ(mprotect(priv_map, total_bytes, PROT_READ), 0);

  // A modified MAP_PRIVATE mapping frozen to PROT_READ (r--p) must not be treated as reloadable from file.
  EXPECT_FALSE(shared::IsFileBackedReadOnlyMapping(priv_map, total_bytes));
  EXPECT_EQ(shared::EvictFileBackedHostPages(priv_map, total_bytes), 0U);
  EXPECT_EQ(priv_bytes[0], 0x22);
  EXPECT_EQ(priv_bytes[expert_bytes], 0x33);

  shared::FileExpertStore store;
  shared::FileExpertStore::MappedBank bank{priv_map, expert_bytes, 0, nullptr, 0};
  ASSERT_TRUE(store.InitializeMapped(&bank, 1, num_experts, expert_bytes, /*cache_slots=*/1));
  EXPECT_FALSE(store.IsBankFileBacked(0));
  EXPECT_EQ(store.ResidentBackingBytes(), total_bytes);

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 0x22);
  store.ReleaseLease(0);
  ASSERT_TRUE(store.AcquireReady(1, &lease));
  EXPECT_EQ(lease.data[0], 0x33);
  store.ReleaseLease(1);
  store.Shutdown();

  // Verify that a genuine shared read-only mapping (r--s) is accepted.
  void* shared_map = mmap(nullptr, total_bytes, PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(shared_map, MAP_FAILED);
  EXPECT_TRUE(shared::IsFileBackedReadOnlyMapping(shared_map, total_bytes));
  EXPECT_EQ(shared::EvictFileBackedHostPages(shared_map, total_bytes), total_bytes);

  munmap(shared_map, total_bytes);
  munmap(priv_map, total_bytes);
  close(fd);
  std::filesystem::remove(tmp_path);
}
#endif

TEST(QMoESharedLayoutTest, CanonicalInt4ExpertViewMatchesReferenceAcrossPrepackedRawFp16AndBf16) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr int64_t n1 = 2 * inter;
  std::vector<uint8_t> fc1_raw(static_cast<size_t>(n1 * (hidden / 2)));
  std::vector<uint8_t> fc2_raw(static_cast<size_t>(hidden * (inter / 2)));
  for (size_t i = 0; i < fc1_raw.size(); ++i) fc1_raw[i] = static_cast<uint8_t>((i * 73 + 19) & 0xFF);
  for (size_t i = 0; i < fc2_raw.size(); ++i) fc2_raw[i] = static_cast<uint8_t>((i * 41 + 7) & 0xFF);

  std::vector<uint8_t> fc1_sm80(fc1_raw.size());
  std::vector<uint8_t> fc2_sm80(fc2_raw.size());
  shared::PackSm80Int4Weights(fc1_raw.data(), fc1_sm80.data(), n1, hidden);
  shared::PackSm80Int4Weights(fc2_raw.data(), fc2_sm80.data(), hidden, inter);

  std::vector<uint8_t> fc1_unpacked(fc1_raw.size());
  shared::UnpackSm80Int4Weights(fc1_sm80.data(), fc1_unpacked.data(), n1, hidden);
  EXPECT_EQ(fc1_unpacked, fc1_raw);
  for (int64_t n = 0; n < n1; ++n) {
    for (int64_t k = 0; k < hidden; ++k) {
      const uint8_t raw_byte = fc1_raw[static_cast<size_t>(n * (hidden / 2) + (k / 2))];
      const uint8_t expected_code = (k & 1) ? (raw_byte >> 4) : (raw_byte & 0x0F);
      EXPECT_EQ(shared::WeightCodeAt(fc1_sm80.data(), n1, hidden, n, k), expected_code);
    }
  }

  std::vector<uint16_t> fc1_s_fp16(static_cast<size_t>(n1 * (hidden / 32)));
  std::vector<uint16_t> fc2_s_fp16(static_cast<size_t>(hidden * (inter / 32)));
  std::vector<uint16_t> fc1_s_bf16(fc1_s_fp16.size());
  std::vector<uint16_t> fc2_s_bf16(fc2_s_fp16.size());
  for (size_t i = 0; i < fc1_s_fp16.size(); ++i) {
    const float v = static_cast<float>((i % 7) + 1) / 64.0f;
    fc1_s_fp16[i] = shared::FloatToHalf(v);
    fc1_s_bf16[i] = shared::FloatToBFloat16(v);
  }
  for (size_t i = 0; i < fc2_s_fp16.size(); ++i) {
    const float v = static_cast<float>((i % 5) + 1) / 64.0f;
    fc2_s_fp16[i] = shared::FloatToHalf(v);
    fc2_s_bf16[i] = shared::FloatToBFloat16(v);
  }
  std::vector<uint16_t> fc1_b_fp16(static_cast<size_t>(n1));
  std::vector<uint16_t> fc2_b_fp16(static_cast<size_t>(hidden));
  std::vector<uint16_t> fc1_b_bf16(static_cast<size_t>(n1));
  std::vector<uint16_t> fc2_b_bf16(static_cast<size_t>(hidden));
  for (int64_t i = 0; i < n1; ++i) {
    const float v = static_cast<float>((i % 9) - 4) / 32.0f;
    fc1_b_fp16[static_cast<size_t>(i)] = shared::FloatToHalf(v);
    fc1_b_bf16[static_cast<size_t>(i)] = shared::FloatToBFloat16(v);
  }
  for (int64_t i = 0; i < hidden; ++i) {
    const float v = static_cast<float>((i % 11) - 5) / 32.0f;
    fc2_b_fp16[static_cast<size_t>(i)] = shared::FloatToHalf(v);
    fc2_b_bf16[static_cast<size_t>(i)] = shared::FloatToBFloat16(v);
  }

  // Build legacy SharedExpertBlockV1 block to cross-check FP16 prepacked parity.
  shared::ExpertBlockLayout blk_layout;
  ASSERT_TRUE(shared::MakeExpertBlockLayout(hidden, inter, &blk_layout));
  std::vector<uint8_t> blk(blk_layout.stride, 0);
  std::memcpy(blk.data() + blk_layout.fc1_weights, fc1_sm80.data(), fc1_sm80.size());
  shared::TransposeInt4Scales(fc1_s_fp16.data(), reinterpret_cast<uint16_t*>(blk.data() + blk_layout.fc1_scales),
                              n1, hidden / 32);
  std::memcpy(blk.data() + blk_layout.fc1_bias, fc1_b_fp16.data(), fc1_b_fp16.size() * sizeof(uint16_t));
  std::memcpy(blk.data() + blk_layout.fc2_weights, fc2_sm80.data(), fc2_sm80.size());
  shared::TransposeInt4Scales(fc2_s_fp16.data(), reinterpret_cast<uint16_t*>(blk.data() + blk_layout.fc2_scales),
                              hidden, inter / 32);
  std::memcpy(blk.data() + blk_layout.fc2_bias, fc2_b_fp16.data(), fc2_b_fp16.size() * sizeof(uint16_t));

  constexpr int64_t rows = 5;
  std::vector<float> x(static_cast<size_t>(rows * hidden));
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 32.0f;
  const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};

  std::vector<float> ref_blk_out(static_cast<size_t>(rows * hidden));
  ASSERT_TRUE(shared::ExpertForward(blk_layout, blk.data(), x.data(), rows, swiglu, ref_blk_out.data(),
                                    /*force_scalar=*/true, /*apply_fc2_bias=*/true, /*fp16_boundaries=*/true));

  for (bool prepacked : {true, false}) {
    for (bool is_bf16 : {false, true}) {
      shared::Int4ExpertView view{};
      view.fc1_weights = prepacked ? fc1_sm80.data() : fc1_raw.data();
      view.fc1_scales = is_bf16 ? fc1_s_bf16.data() : fc1_s_fp16.data();
      view.fc1_bias = is_bf16 ? fc1_b_bf16.data() : fc1_b_fp16.data();
      view.fc2_weights = prepacked ? fc2_sm80.data() : fc2_raw.data();
      view.fc2_scales = is_bf16 ? fc2_s_bf16.data() : fc2_s_fp16.data();
      view.fc2_bias = is_bf16 ? fc2_b_bf16.data() : fc2_b_fp16.data();
      view.scales_are_block_major = false;
      view.weights_are_prepacked = prepacked;
      view.scale_is_bf16 = is_bf16;
      view.bias_is_bf16 = is_bf16;

      std::vector<float> out_scalar(static_cast<size_t>(rows * hidden));
      std::vector<float> out_simd(static_cast<size_t>(rows * hidden));
      ASSERT_TRUE(shared::Int4ExpertForward(hidden, inter, view, x.data(), rows, swiglu, out_scalar.data(),
                                            /*force_scalar=*/true, /*apply_fc2_bias=*/true,
                                            /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
      ASSERT_TRUE(shared::Int4ExpertForward(hidden, inter, view, x.data(), rows, swiglu, out_simd.data(),
                                            /*force_scalar=*/false, /*apply_fc2_bias=*/true,
                                            /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
      for (size_t i = 0; i < out_scalar.size(); ++i) {
        EXPECT_NEAR(out_simd[i], out_scalar[i], 1e-5f) << "prepacked=" << prepacked << " bf16=" << is_bf16 << " i=" << i;
        if (!is_bf16) {
          EXPECT_NEAR(out_scalar[i], ref_blk_out[i], 1e-5f) << "prepacked=" << prepacked << " i=" << i;
        }
      }
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test
