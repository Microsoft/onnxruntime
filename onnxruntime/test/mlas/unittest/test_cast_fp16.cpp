/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_cast_fp16.cpp

Abstract:

    Tests for MLAS FP16<->FP32 cast kernels.
    Verifies bit-exactness against MLAS_Half2Float / MLAS_Float2Half for all
    non-NaN values.  For NaN values only NaN-ness and sign are asserted,
    because the scalar reference canonicalizes payload while hardware preserves it.

--*/

#include "test_util.h"
#include "mlas.h"
#include "mlas_float16.h"
#include "core/mlas/lib/mlasi.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

class MlasCastFp16Test : public MlasTestBase {
 public:
  void TestF16ToF32(size_t count, size_t source_alignment_offset = 0) {
    std::vector<_mlas_fp16_> input(count + 3);
    std::vector<float> output_ref(count);
    std::vector<float> output_dispatch(count);
    const size_t input_offset =
        (source_alignment_offset + 4 - (reinterpret_cast<uintptr_t>(input.data()) / sizeof(input[0])) % 4) % 4;
    auto* input_data = input.data() + input_offset;

    for (size_t i = 0; i < count; i++) {
      float val = (static_cast<float>(i % 2048) / 1024.0f) - 1.0f;
      input_data[i] = MLAS_Float2Half(val);
      output_ref[i] = MLAS_Half2Float(input_data[i]);
    }

    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(input_data),
        output_dispatch.data(), count);

    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(output_dispatch[i], output_ref[i])
          << "F16->F32 mismatch at [" << i << "], count=" << count
          << ", source_alignment_offset=" << source_alignment_offset;
    }
  }

  void TestF32ToF16(size_t count, size_t source_alignment_offset = 0) {
    std::vector<float> input(count + 3);
    std::vector<_mlas_fp16_> output_ref(count);
    std::vector<_mlas_fp16_> output_dispatch(count);
    const size_t input_offset =
        (source_alignment_offset + 4 - (reinterpret_cast<uintptr_t>(input.data()) / sizeof(input[0])) % 4) % 4;
    auto* input_data = input.data() + input_offset;

    for (size_t i = 0; i < count; i++) {
      input_data[i] = (static_cast<float>(i % 2048) / 1024.0f) - 1.0f;
      output_ref[i] = MLAS_Float2Half(input_data[i]);
    }

    MlasConvertFloatToHalfBuffer(
        input_data,
        reinterpret_cast<MLAS_FP16*>(output_dispatch.data()), count);

    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(output_dispatch[i], output_ref[i])
          << "F32->F16 mismatch at [" << i << "], count=" << count
          << ", source_alignment_offset=" << source_alignment_offset;
    }
  }

  // Test special IEEE 754 values: ±0, ±Inf, NaN, denormals, and a value
  // requiring round-to-nearest-even in the f32→f16 direction.
  //
  // NaN handling: The MLAS scalar reference (MLAS_Float2Half) canonicalizes
  // every NaN to 0x7E00 (f16 canonical qNaN), discarding the payload.
  // Hardware FCVTN/FCVTL preserves (truncates) the payload.  Because payload
  // semantics differ, for NaN results we assert only:
  //   1. Both results are NaN (exponent all-ones, mantissa non-zero).
  //   2. Sign bits match.
  // Payload bits are deliberately unasserted — they are implementation-
  // defined and not portable across scalar vs. NEON paths.
  void TestSpecialValues() {
    // Raw fp16 bit patterns
    const uint16_t kPosZero = 0x0000;
    const uint16_t kNegZero = 0x8000;
    const uint16_t kPosInf = 0x7C00;
    const uint16_t kNegInf = 0xFC00;
    const uint16_t kQNaN = 0x7E00;
    const uint16_t kNegQNaN = 0xFE00;
    const uint16_t kSNaN = 0x7C01;
    const uint16_t kNegSNaN = 0xFC01;
    const uint16_t kDenormMin = 0x0001;
    const uint16_t kDenormMid = 0x0200;
    const uint16_t kNegDenorm = 0x8001;

    std::vector<uint16_t> special_bits = {
        kPosZero, kNegZero, kPosInf, kNegInf, kQNaN, kNegQNaN,
        kSNaN, kNegSNaN, kDenormMin, kDenormMid, kNegDenorm};
    const size_t n = special_bits.size();

    // F16 -> F32: convert via dispatch and via scalar reference
    std::vector<_mlas_fp16_> h_input(n);
    std::memcpy(h_input.data(), special_bits.data(), n * sizeof(uint16_t));

    std::vector<float> out_dispatch(n);
    std::vector<float> out_ref(n);
    for (size_t i = 0; i < n; i++) {
      out_ref[i] = MLAS_Half2Float(h_input[i]);
    }

    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(h_input.data()),
        out_dispatch.data(), n);

    for (size_t i = 0; i < n; i++) {
      uint32_t ref_bits, disp_bits;
      std::memcpy(&ref_bits, &out_ref[i], sizeof(float));
      std::memcpy(&disp_bits, &out_dispatch[i], sizeof(float));

      if (std::isnan(out_ref[i])) {
        // NaN: assert both NaN and same sign.  Payload is not asserted
        // because hardware preserves it while the scalar reference may not.
        ASSERT_TRUE(std::isnan(out_dispatch[i]))
            << "F16->F32 expected NaN at [" << i
            << "], h=0x" << std::hex << special_bits[i];
        constexpr uint32_t kF32SignBit = 0x80000000u;
        ASSERT_EQ(disp_bits & kF32SignBit, ref_bits & kF32SignBit)
            << "F16->F32 NaN sign mismatch at [" << i
            << "], h=0x" << std::hex << special_bits[i];
      } else {
        ASSERT_EQ(disp_bits, ref_bits)
            << "F16->F32 special mismatch at [" << i
            << "], h=0x" << std::hex << special_bits[i];
      }
    }

    // F32 -> F16 includes signed NaNs, exact half subnormals, a subnormal
    // underflow tie, and a normal round-to-nearest-even tie.
    const std::vector<uint32_t> f32_input_bits = {
        0x00000000u,  // +0
        0x80000000u,  // -0
        0x7F800000u,  // +Inf
        0xFF800000u,  // -Inf
        0x7FC00000u,  // +qNaN
        0xFFC00000u,  // -qNaN
        0x7F800001u,  // +sNaN
        0xFF800001u,  // -sNaN
        0x33800000u,  // 2^-24: smallest positive half subnormal
        0xB3800000u,  // -2^-24
        0x387FC000u,  // largest positive half subnormal
        0xB87FC000u,  // largest negative half subnormal
        0x33000000u,  // 2^-25: tie rounds to +0
        0xB3000000u,  // -2^-25: tie rounds to -0
        0x3F801000u,  // 1 + 2^-11: tie rounds to half 1.0
    };
    std::vector<float> f32_input(f32_input_bits.size());
    std::memcpy(f32_input.data(), f32_input_bits.data(), f32_input_bits.size() * sizeof(uint32_t));
    const size_t m = f32_input.size();

    std::vector<_mlas_fp16_> f16_out_dispatch(m);
    std::vector<_mlas_fp16_> f16_out_ref(m);
    for (size_t i = 0; i < m; i++) {
      f16_out_ref[i] = MLAS_Float2Half(f32_input[i]);
    }

    MlasConvertFloatToHalfBuffer(
        f32_input.data(),
        reinterpret_cast<MLAS_FP16*>(f16_out_dispatch.data()), m);

    for (size_t i = 0; i < m; i++) {
      uint16_t ref_bits, disp_bits;
      std::memcpy(&ref_bits, &f16_out_ref[i], sizeof(uint16_t));
      std::memcpy(&disp_bits, &f16_out_dispatch[i], sizeof(uint16_t));

      if (std::isnan(f32_input[i])) {
        // NaN: assert result is NaN and sign matches.  Payload is not
        // asserted because MLAS_Float2Half canonicalizes to 0x7E00 while
        // hardware FCVTN preserves/truncates the source payload.
        constexpr uint16_t kF16ExpMask = 0x7C00u;
        constexpr uint16_t kF16MantMask = 0x03FFu;
        ASSERT_EQ(disp_bits & kF16ExpMask, kF16ExpMask)
            << "F32->F16 expected NaN exponent at [" << i
            << "], f32=" << f32_input[i];
        ASSERT_NE(disp_bits & kF16MantMask, 0u)
            << "F32->F16 expected NaN mantissa non-zero at [" << i
            << "], f32=" << f32_input[i];
        constexpr uint16_t kF16SignBit = 0x8000u;
        ASSERT_EQ(disp_bits & kF16SignBit, ref_bits & kF16SignBit)
            << "F32->F16 NaN sign mismatch at [" << i
            << "], f32=" << f32_input[i];
      } else {
        ASSERT_EQ(disp_bits, ref_bits)
            << "F32->F16 special mismatch at [" << i
            << "], f32=" << f32_input[i];
      }
    }
  }

  void TestUnalignedSources() {
    constexpr size_t kCount = 17;
    for (size_t source_alignment_offset = 1; source_alignment_offset <= 3; ++source_alignment_offset) {
      TestF16ToF32(kCount, source_alignment_offset);
      TestF32ToF16(kCount, source_alignment_offset);
    }
  }

  // Verify the vectorised kernel is dispatched (non-null function pointer)
  // on macOS arm64 or other platforms with MLAS_F16VEC_INTRINSICS_SUPPORTED.
  // The macOS condition deliberately does not depend on
  // MLAS_CAST_F16_NEON_SUPPORTED, so a missing feature definition cannot turn
  // this test into a false-green no-op.  The dispatch table pointers are what
  // cast.cpp checks at runtime to select the NEON kernel.
  // This test compiles to nothing on platforms without a vectorised kernel.
  void TestKernelIsDispatched() {
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || \
    (defined(__APPLE__) && defined(MLAS_TARGET_ARM64) && TARGET_OS_OSX)
    // On macOS arm64 (or other platforms with vectorised cast support),
    // the dispatch pointers must be non-null — proving the NEON kernel
    // is selected rather than the scalar fallback.
    ASSERT_NE(GetMlasPlatform().CastF16ToF32Kernel, nullptr)
        << "Expected non-null CastF16ToF32Kernel on this platform";
    ASSERT_NE(GetMlasPlatform().CastF32ToF16Kernel, nullptr)
        << "Expected non-null CastF32ToF16Kernel on this platform";
#endif
  }
};

class CastFp16ShortExecuteTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  CastFp16ShortExecuteTest(size_t count, bool f16_to_f32)
      : count_(count), f16_to_f32_(f16_to_f32) {}

  void TestBody() override {
    if (f16_to_f32_) {
      MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestF16ToF32(count_);
    } else {
      MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestF32ToF16(count_);
    }
  }

  static size_t RegisterSingleTest(size_t count, bool f16_to_f32) {
    std::stringstream ss;
    ss << "/" << (f16_to_f32 ? "F16toF32" : "F32toF16")
       << "/count" << count;
    auto test_name = ss.str();

    testing::RegisterTest(
        "CastFp16",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16ShortExecuteTest(count, f16_to_f32);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t cnt = 0;
    // Various lengths including non-multiples of vector width (4/8)
    for (size_t n : {1, 2, 3, 5, 7, 9, 15, 16, 17, 31, 32, 63, 64, 128, 255, 256, 1024, 65536}) {
      cnt += RegisterSingleTest(n, true);
      cnt += RegisterSingleTest(n, false);
    }
    return cnt;
  }

 private:
  size_t count_;
  bool f16_to_f32_;
};

class CastFp16UnalignedSourcesTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  void TestBody() override {
    MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestUnalignedSources();
  }

  static size_t RegisterTests() {
    testing::RegisterTest(
        "CastFp16",
        "/UnalignedSources",
        nullptr,
        "/UnalignedSources",
        __FILE__,
        __LINE__,
        []() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16UnalignedSourcesTest();
        });
    return 1;
  }
};

class CastFp16SpecialValuesTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  void TestBody() override {
    MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestSpecialValues();
  }

  static size_t RegisterTests() {
    testing::RegisterTest(
        "CastFp16",
        "/SpecialValues",
        nullptr,
        "/SpecialValues",
        __FILE__,
        __LINE__,
        []() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16SpecialValuesTest();
        });
    return 1;
  }
};

class CastFp16KernelDispatchTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  void TestBody() override {
    MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestKernelIsDispatched();
  }

  static size_t RegisterTests() {
    testing::RegisterTest(
        "CastFp16",
        "/KernelDispatched",
        nullptr,
        "/KernelDispatched",
        __FILE__,
        __LINE__,
        []() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16KernelDispatchTest();
        });
    return 1;
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return CastFp16ShortExecuteTest::RegisterShortExecuteTests() +
               CastFp16UnalignedSourcesTest::RegisterTests() +
               CastFp16SpecialValuesTest::RegisterTests() +
               CastFp16KernelDispatchTest::RegisterTests();
      }
      return 0;
    });
