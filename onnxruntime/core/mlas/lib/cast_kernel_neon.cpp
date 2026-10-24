/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast_kernel_neon.cpp

Abstract:

    This module implements the common kernels for ARM NEON specific to float16.

--*/

#include "mlasi.h"

#include "arm_neon.h"

// This file is enabled in cmake for ARM64 targets: on non-Apple platforms when
// MLAS_F16VEC_INTRINSICS_SUPPORTED is set, and on macOS ARM64 via MLAS_CAST_F16_NEON_SUPPORTED.
// The conversion intrinsics (vcvt_f32_f16 / vcvt_f16_f32) are baseline ARMv8-A;
// no -march=armv8.2-a+fp16 is needed for conversions alone.

MLAS_FORCEINLINE
size_t
StoreFp32Lane(float* dest, float32x4_t src, size_t count)
{
    if (count == 3) {
        vst1q_lane_f32(dest + 0, src, 0);
        vst1q_lane_f32(dest + 1, src, 1);
        vst1q_lane_f32(dest + 2, src, 2);
        return 3;
    } else if (count == 2) {
        vst1q_lane_f32(dest + 0, src, 0);
        vst1q_lane_f32(dest + 1, src, 1);
        return 2;
    } else if (count == 1) {
        vst1q_lane_f32(dest + 0, src, 0);
        return 1;
    }

    return 0;
}

void
MlasCastF16ToF32KernelNeon(const unsigned short* src, float* dest, size_t count)
{
    // 4 float16 alignment
    auto* src_aligned = reinterpret_cast<const unsigned short*>((reinterpret_cast<uintptr_t>(src) + 7) & ~7);
    auto pre_count = std::min(static_cast<size_t>(src_aligned - src), count);
    size_t i = 0;

    // Handle leading unaligned src
    if (pre_count > 0) {
        float16x4_t fp16v4;
        std::memcpy(&fp16v4, src, pre_count * sizeof(unsigned short));
        float32x4_t fp32v4 = vcvt_f32_f16(fp16v4);

        i = StoreFp32Lane(dest, fp32v4, pre_count);
    }

    // aligned src
    for (; i + 7 < count; i += 8)
    {
        float16x4_t fp16v4_0 = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t fp32v4_0 = vcvt_f32_f16(fp16v4_0);
        vst1q_f32(dest + i, fp32v4_0);

        float16x4_t fp16v4_1 = vreinterpret_f16_u16(vld1_u16(src + i + 4));
        float32x4_t fp32v4_1 = vcvt_f32_f16(fp16v4_1);
        vst1q_f32(dest + i + 4, fp32v4_1);
    }

    if (i + 3 < count)
    {
        float16x4_t fp16v4_0 = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t fp32v4_0 = vcvt_f32_f16(fp16v4_0);
        vst1q_f32(dest + i, fp32v4_0);
        i += 4;
    }

    // Handle trailing unaligned src
    auto post_count = count - i;
    if (post_count > 0)
    {
        float16x4_t fp16v4;
        std::memcpy(&fp16v4, src + i, post_count * sizeof(unsigned short));
        float32x4_t fp32v4 = vcvt_f32_f16(fp16v4);

        StoreFp32Lane(dest + i, fp32v4, post_count);
    }
}

MLAS_FORCEINLINE
size_t
StoreU16Lane(unsigned short* dest, uint16x4_t src, size_t count)
{
    if (count == 3) {
        vst1_lane_u16(dest + 0, src, 0);
        vst1_lane_u16(dest + 1, src, 1);
        vst1_lane_u16(dest + 2, src, 2);
        return 3;
    } else if (count == 2) {
        vst1_lane_u16(dest + 0, src, 0);
        vst1_lane_u16(dest + 1, src, 1);
        return 2;
    } else if (count == 1) {
        vst1_lane_u16(dest + 0, src, 0);
        return 1;
    }

    return 0;
}

void
MlasCastF32ToF16KernelNeon(const float* src, unsigned short* dest, size_t count)
{
    // 4 float32 alignment
    auto* src_aligned = reinterpret_cast<const float*>((reinterpret_cast<uintptr_t>(src) + 15) & ~15);
    auto pre_count = std::min(static_cast<size_t>(src_aligned - src), count);
    size_t i = 0;

    // Handle leading unaligned src
    if (pre_count > 0)
    {
        float32x4_t fp32v4;
        std::memcpy(&fp32v4, src, pre_count * sizeof(float));
        uint16x4_t u16v4 = vreinterpret_u16_f16(vcvt_f16_f32(fp32v4));

        i = StoreU16Lane(dest, u16v4, pre_count);
    }

    // aligned src
    for (; i + 7 < count; i += 8)
    {
        float32x4_t fp32v4_0 = vld1q_f32(src + i);
        float16x4_t fp16v4_0 = vcvt_f16_f32(fp32v4_0);
        vst1_u16(dest + i, vreinterpret_u16_f16(fp16v4_0));

        float32x4_t fp32v4_1 = vld1q_f32(src + i + 4);
        float16x4_t fp16v4_1 = vcvt_f16_f32(fp32v4_1);
        vst1_u16(dest + i + 4, vreinterpret_u16_f16(fp16v4_1));
    }

    if (i + 3 < count)
    {
        float32x4_t fp32v4_0 = vld1q_f32(src + i);
        float16x4_t fp16v4_0 = vcvt_f16_f32(fp32v4_0);
        vst1_u16(dest + i, vreinterpret_u16_f16(fp16v4_0));
        i += 4;
    }

    // Handle trailing unaligned src
    auto post_count = count - i;
    if (post_count > 0)
    {
        float32x4_t fp32v4;
        std::memcpy(&fp32v4, src + i, post_count * sizeof(float));
        uint16x4_t u16v4 = vreinterpret_u16_f16(vcvt_f16_f32(fp32v4));

        StoreU16Lane(dest + i, u16v4, post_count);
    }
}
