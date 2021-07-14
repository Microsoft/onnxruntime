/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    platform.cpp

Abstract:

    This module implements logic to select the best configuration for the
    this platform.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_POWER) && defined(__linux__)
#include <sys/auxv.h>
#endif

#if defined(MLAS_TARGET_ARM64)
#if defined(_WIN32)
// N.B. Support building with downlevel versions of the Windows SDK.
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
// N.B. Support building with older versions of asm/hwcap.h that do not define
// this capability bit.
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#endif // MLAS_TARGET_ARM64

//
// Stores the platform information.
//

MLAS_PLATFORM MlasPlatform;

#ifdef MLAS_TARGET_AMD64_IX86

//
// Stores a vector to build a conditional load/store mask for vmaskmovps.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint32_t MlasMaskMoveAvx[8], 32) = { 0, 1, 2, 3, 4, 5, 6, 7 };

//
// Stores a table of AVX vmaskmovps/vmaskmovpd load/store masks.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint32_t MlasMaskMoveTableAvx[16], 32) = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
};

//
// Stores a table of AVX512 opmask register values.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const int16_t MlasOpmask16BitTableAvx512[16], 32) = {
    0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F,
    0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF,
};

//
// Reads the processor extended control register to determine platform
// capabilities.
//

#if !defined(_XCR_XFEATURE_ENABLED_MASK)
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

inline
uint64_t
MlasReadExtendedControlRegister(
    unsigned int ext_ctrl_reg
)
{
#if defined(_WIN32)
    return _xgetbv(ext_ctrl_reg);
#else
    uint32_t eax, edx;

    __asm__
    (
        "xgetbv"
        : "=a" (eax), "=d" (edx)
        : "c" (ext_ctrl_reg)
    );

    return ((uint64_t)edx << 32) | eax;
#endif
}

#endif

MLAS_PLATFORM::MLAS_PLATFORM(
    void
    )
/*++

Routine Description:

    This routine initializes the platform support for this library.

Arguments:

    None.

Return Value:

    None.

--*/
{

#if defined(MLAS_TARGET_AMD64_IX86)

    //
    // Default to the baseline SSE2 support.
    //

    this->GemmFloatKernel = MlasGemmFloatKernelSse;
    this->GemmU8S8Dispatch = &MlasGemmU8X8DispatchSse;
    this->GemmU8U8Dispatch = &MlasGemmU8X8DispatchSse;

#if defined(MLAS_TARGET_AMD64)

    this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Sse;
    this->GemmDoubleKernel = MlasGemmDoubleKernelSse;
    this->ConvNchwFloatKernel = MlasConvNchwFloatKernelSse;
    this->ConvNchwcFloatKernel = MlasConvNchwcFloatKernelSse;
    this->ConvDepthwiseFloatKernel = MlasConvDepthwiseFloatKernelSse;
    this->ConvPointwiseFloatKernel = MlasConvPointwiseFloatKernelSse;
    this->PoolFloatKernel[MlasMaximumPooling] = MlasPoolMaximumFloatKernelSse;
    this->PoolFloatKernel[MlasAveragePoolingExcludePad] = MlasPoolAverageExcludePadFloatKernelSse;
    this->PoolFloatKernel[MlasAveragePoolingIncludePad] = MlasPoolAverageIncludePadFloatKernelSse;
    this->ComputeExpF32Kernel = MlasComputeExpF32Kernel;
    this->LogisticKernelRoutine = MlasLogisticKernel;
    this->TanhKernelRoutine = MlasTanhKernel;
    this->ErfKernelRoutine = MlasErfKernel;
    this->ComputeSumExpF32Kernel = MlasComputeSumExpF32Kernel;
    this->ComputeSoftmaxOutputF32Kernel = MlasComputeSoftmaxOutputF32Kernel;
    this->ComputeLogSoftmaxOutputF32Kernel = MlasComputeLogSoftmaxOutputF32Kernel;
    this->ReduceMaximumF32Kernel = MlasReduceMaximumF32Kernel;
    this->ReduceMinimumMaximumF32Kernel = MlasReduceMinimumMaximumF32Kernel;
    this->QLinearAddS8Kernel = MlasQLinearAddS8Kernel;
    this->QLinearAddU8Kernel = MlasQLinearAddU8Kernel;
    this->QuantizeLinearS8Kernel = MlasQuantizeLinearS8Kernel;
    this->QuantizeLinearU8Kernel = MlasQuantizeLinearU8Kernel;
    this->ConvDepthwiseU8S8Kernel = MlasConvDepthwiseKernel<int8_t>;
    this->ConvDepthwiseU8U8Kernel = MlasConvDepthwiseKernel<uint8_t>;

    this->NchwcBlockSize = 8;
    this->PreferredBufferAlignment = MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT;

    this->MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT;

#endif

    unsigned Cpuid1[4];
#if defined(_WIN32)
    __cpuid((int*)Cpuid1, 1);
#else
    __cpuid(1, Cpuid1[0], Cpuid1[1], Cpuid1[2], Cpuid1[3]);
#endif

#if defined(_MSC_VER)

    //
    // Check if the processor supports SSE 4.1 instructions.
    //

    if ((Cpuid1[2] & 0x80000) != 0) {
        this->GemmU8S8Dispatch = &MlasGemmU8S8DispatchSse41;
    }

#endif

    //
    // Check if the processor supports the AVX and OSXSAVE features.
    //

    if ((Cpuid1[2] & 0x18000000) == 0x18000000) {

        //
        // Check if the operating system supports saving SSE and AVX states.
        //

        uint64_t xcr0 = MlasReadExtendedControlRegister(_XCR_XFEATURE_ENABLED_MASK);

        if ((xcr0 & 0x6) == 0x6) {

            this->GemmFloatKernel = MlasGemmFloatKernelAvx;

#if defined(MLAS_TARGET_AMD64)

            this->KernelM1Routine = MlasSgemmKernelM1Avx;
            this->KernelM1TransposeBRoutine = MlasSgemmKernelM1TransposeBAvx;
            this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Avx;
            this->GemmDoubleKernel = MlasGemmDoubleKernelAvx;
            this->ConvNchwFloatKernel = MlasConvNchwFloatKernelAvx;
            this->ConvNchwcFloatKernel = MlasConvNchwcFloatKernelAvx;
            this->ConvDepthwiseFloatKernel = MlasConvDepthwiseFloatKernelAvx;
            this->ConvPointwiseFloatKernel = MlasConvPointwiseFloatKernelAvx;
            this->PoolFloatKernel[MlasMaximumPooling] = MlasPoolMaximumFloatKernelAvx;
            this->PoolFloatKernel[MlasAveragePoolingExcludePad] = MlasPoolAverageExcludePadFloatKernelAvx;
            this->PoolFloatKernel[MlasAveragePoolingIncludePad] = MlasPoolAverageIncludePadFloatKernelAvx;
            this->ComputeSoftmaxOutputF32Kernel = MlasComputeSoftmaxOutputF32KernelAvx;
            this->ComputeLogSoftmaxOutputF32Kernel = MlasComputeLogSoftmaxOutputF32KernelAvx;
            this->ReduceMaximumF32Kernel = MlasReduceMaximumF32KernelAvx;
            this->ReduceMinimumMaximumF32Kernel = MlasReduceMinimumMaximumF32KernelAvx;

            //
            // Check if the processor supports AVX2/FMA3 features.
            //

            unsigned Cpuid7[4];
#if defined(_WIN32)
            __cpuidex((int*)Cpuid7, 7, 0);
#else
            __cpuid_count(7, 0, Cpuid7[0], Cpuid7[1], Cpuid7[2], Cpuid7[3]);
#endif

            if (((Cpuid1[2] & 0x1000) != 0) && ((Cpuid7[1] & 0x20) != 0)) {

                this->GemmU8S8Dispatch = &MlasGemmU8S8DispatchAvx2;
                this->GemmU8S8Kernel = MlasGemmU8S8KernelAvx2;
                this->GemvU8S8Kernel = MlasGemvU8S8KernelAvx2;
                this->GemmU8U8Dispatch = &MlasGemmU8U8DispatchAvx2;
                this->GemmU8U8Kernel = MlasGemmU8U8KernelAvx2;

                this->GemmFloatKernel = MlasGemmFloatKernelFma3;
                this->GemmDoubleKernel = MlasGemmDoubleKernelFma3;
                this->ConvNchwFloatKernel = MlasConvNchwFloatKernelFma3;
                this->ConvNchwcFloatKernel = MlasConvNchwcFloatKernelFma3;
                this->ConvDepthwiseFloatKernel = MlasConvDepthwiseFloatKernelFma3;
                this->ConvPointwiseFloatKernel = MlasConvPointwiseFloatKernelFma3;
                this->ComputeExpF32Kernel = MlasComputeExpF32KernelFma3;
                this->LogisticKernelRoutine = MlasComputeLogisticF32KernelFma3;
                this->TanhKernelRoutine = MlasComputeTanhF32KernelFma3;
                this->ErfKernelRoutine = MlasErfKernelFma3;
                this->QLinearAddS8Kernel = MlasQLinearAddS8KernelAvx2;
                this->QLinearAddU8Kernel = MlasQLinearAddU8KernelAvx2;
                this->ConvDepthwiseU8S8Kernel = MlasConvDepthwiseKernelAvx2<int8_t>;
                this->ConvDepthwiseU8U8Kernel = MlasConvDepthwiseKernelAvx2<uint8_t>;
                this->ComputeSumExpF32Kernel = MlasComputeSumExpF32KernelFma3;

                //
                // Check if the processor supports Hybrid core architecture.
                //

                if ((Cpuid7[3] & 0x8000) != 0) {
                    this->MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT * 4;
                }

                //
                // Check if the processor supports AVXVNNI features.
                //

                unsigned Cpuid7_1[4];
#if defined(_WIN32)
                __cpuidex((int*)Cpuid7_1, 7, 1);
#else
                __cpuid_count(7, 1, Cpuid7_1[0], Cpuid7_1[1], Cpuid7_1[2], Cpuid7_1[3]);
#endif

                if ((Cpuid7_1[0] & 0x10) != 0) {

                    this->GemmU8U8Dispatch = &MlasGemmU8S8DispatchAvx2;
                    this->GemmU8S8Kernel = MlasGemmU8S8KernelAvxVnni;
                    this->GemvU8S8Kernel = MlasGemvU8S8KernelAvxVnni;
                }

#if !defined(MLAS_AVX512F_UNSUPPORTED)

                //
                // Check if the processor supports AVX512F features and the
                // operating system supports saving AVX512F state.
                //

                if (((Cpuid7[1] & 0x10000) != 0) && ((xcr0 & 0xE0) == 0xE0)) {

                    this->GemmFloatKernel = MlasGemmFloatKernelAvx512F;
                    this->GemmDoubleKernel = MlasGemmDoubleKernelAvx512F;
                    this->ConvNchwFloatKernel = MlasConvNchwFloatKernelAvx512F;
                    this->ConvNchwcFloatKernel = MlasConvNchwcFloatKernelAvx512F;
                    this->ConvDepthwiseFloatKernel = MlasConvDepthwiseFloatKernelAvx512F;
                    this->ConvPointwiseFloatKernel = MlasConvPointwiseFloatKernelAvx512F;
                    this->PoolFloatKernel[MlasMaximumPooling] = MlasPoolMaximumFloatKernelAvx512F;
                    this->PoolFloatKernel[MlasAveragePoolingExcludePad] = MlasPoolAverageExcludePadFloatKernelAvx512F;
                    this->PoolFloatKernel[MlasAveragePoolingIncludePad] = MlasPoolAverageIncludePadFloatKernelAvx512F;
                    this->ComputeExpF32Kernel = MlasComputeExpF32KernelAvx512F;
                    this->ComputeSumExpF32Kernel = MlasComputeSumExpF32KernelAvx512F;
                    this->NchwcBlockSize = 16;
                    this->PreferredBufferAlignment = 64;

#if !defined(MLAS_AVX512F_INTRINSICS_UNSUPPORTED)
                    this->QuantizeLinearS8Kernel = MlasQuantizeLinearS8KernelAvx512F;
                    this->QuantizeLinearU8Kernel = MlasQuantizeLinearU8KernelAvx512F;
#endif

                    //
                    // Check if the processor supports AVX512 core features
                    // (AVX512BW/AVX512DQ/AVX512VL).
                    //

#if !defined(MLAS_AVX512CORE_UNSUPPORTED)

                    if ((Cpuid7[1] & 0xC0020000) == 0xC0020000) {

                        this->GemmU8S8Kernel = MlasGemmU8S8KernelAvx512Core;
                        this->GemvU8S8Kernel = MlasGemvU8S8KernelAvx512Core;
                        this->GemmU8U8Kernel = MlasGemmU8U8KernelAvx512Core;

                        //
                        // Check if the processor supports AVX512VNNI.
                        //

                        if ((Cpuid7[2] & 0x800) != 0) {

                            this->GemmU8U8Dispatch = &MlasGemmU8S8DispatchAvx2;
                            this->GemmU8S8Kernel = MlasGemmU8S8KernelAvx512Vnni;
                            this->GemvU8S8Kernel = MlasGemvU8S8KernelAvx512Vnni;
                        }
                    }

#endif // MLAS_AVX512CORE_UNSUPPORTED

                }

#endif // MLAS_AVX512F_UNSUPPORTED

            }

#endif // MLAS_TARGET_AMD64

        }
    }

#endif // MLAS_TARGET_AMD64_IX86

#if defined(MLAS_TARGET_ARM64)

    this->GemmU8X8Dispatch = &MlasGemmU8X8DispatchNeon;

    //
    // Check if the processor supports ASIMD dot product instructions.
    //

    bool HasDotProductInstructions;

#if defined(_WIN32)
    HasDotProductInstructions = (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0);
#elif defined(__linux__)
    HasDotProductInstructions = ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0);
#else
    HasDotProductInstructions = false;
#endif

    if (HasDotProductInstructions) {
        this->GemmU8X8Dispatch = &MlasGemmU8X8DispatchUdot;
    }

#endif // MLAS_TARGET_ARM64
#if defined(MLAS_TARGET_POWER)
  this->GemmFloatKernel = MlasSgemmKernel;
#if defined(__linux__)  && defined(POWER10)
#if (defined(__GNUC__) && ((__GNUC__ > 10) || (__GNUC__== 10 && __GNUC_MINOR__ >= 2))) || \
    (defined(__clang__) && (__clang_major__ >= 12))
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  bool HasP10Instructions = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
  if (HasP10Instructions) {
    this->GemmFloatKernel = MlasSgemmKernelPOWER10;
  }
#endif
#endif
#endif

}

size_t
MLASCALL
MlasGetPreferredBufferAlignment(
    void
    )
/*++

Routine Description:

    This routine returns the preferred byte alignment for buffers that are used
    with this library. Buffers that are not byte aligned to this value will
    function, but will not achieve best performance.

Arguments:

    None.

Return Value:

    Returns the preferred byte alignment for buffers.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    return MlasPlatform.PreferredBufferAlignment;
#else
    return MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT;
#endif
}
