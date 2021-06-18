/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"

//
// Quantized integer matrix/matrix dispatch structure.
//

//
// Define the default striding parameters used for the quantized integer
// matrix/matrix multiply operation.
//

struct MLAS_GEMM_U8X8_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

typedef
void
(MLAS_GEMM_U8X8_OPERATION)(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const MLAS_GEMM_U8X8_STRIDES& Strides,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

typedef
void
(MLAS_GEMM_U8X8_BLK_OPERATION)(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

typedef
void
(MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE)(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    );

struct MLAS_GEMM_U8X8_DISPATCH {
    MLAS_GEMM_U8X8_BLK_OPERATION* Operation;
    MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t PackedK;
    size_t StrideK;
};

const MLAS_GEMM_U8X8_DISPATCH*
MlasGemmU8X8GetDispatch(
    bool BIsSigned
    )
{
    const MLAS_GEMM_U8X8_DISPATCH* GemmU8X8Dispatch;

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

#if defined(MLAS_TARGET_AMD64)
    if (BIsSigned) {
        GemmU8X8Dispatch = MlasPlatform.GemmU8S8Dispatch;
    } else {
        GemmU8X8Dispatch = MlasPlatform.GemmU8U8Dispatch;
    }
#elif defined(MLAS_SSE2_INTRINSICS)
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchSse;
#elif defined(MLAS_NEON64_INTRINSICS)
    GemmU8X8Dispatch = MlasPlatform.GemmU8X8Dispatch;
#elif defined(MLAS_NEON32_INTRINSICS) && !defined(_MSC_VER)
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchNeon;
#else
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchDefault;
#endif

    return GemmU8X8Dispatch;
}

//
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_U8X8_WORK_BLOCK {
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;
};

void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* Output,
    const int32_t* Input,
    size_t N,
    int32_t Scale
    )
{
    for (size_t n = 0; n < N; n++) {
        Output[n] = Input[n] * Scale;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* SumBuffer,
    size_t N,
    int32_t Scale
    )
{
    return MlasGemmU8X8ScaleSumBuffer(SumBuffer, SumBuffer, N, Scale);
}

template<typename KernelType>
MLAS_FORCEINLINE
bool
MlasGemmU8X8TryGemvKernel(
    const uint8_t* A,
    const uint8_t* B,
    size_t ldb,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(A);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(C);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return false;
}

//
// TODO!! query L2 data cache size from system
// 
constexpr size_t CacheSize = 256 * 1024;

thread_local uint8_t PackPanels[CacheSize];

template <typename KernelType>
MLAS_FORCEINLINE
size_t
MlasQGemmPackPanelASize(const MLAS_GEMM_U8X8_STRIDES& Strides)
{
    const size_t K = (Strides.K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
    const size_t PackPanelASize =
        Strides.M * K * sizeof(typename KernelType::PackedAType);

    return PackPanelASize;
}

template <typename KernelType>
MLAS_FORCEINLINE size_t
MlasQGemmPackPanelBSize(const MLAS_GEMM_U8X8_STRIDES& Strides)
{
    const size_t K = (Strides.K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);
    const size_t N = (Strides.N + 15) & ~15;
    const size_t PackPanelBSize = K * N * sizeof(typename KernelType::PackedBType);

    return PackPanelBSize;
}


template<typename KernelType>
int32_t
MlasGemmU8X8FixupZeroPointB(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return ZeroPointB;
}

template<typename KernelType>
MLAS_FORCEINLINE
void
MlasGemmU8X8FixupZeroPointB(
    const uint8_t* PackedZeroPointB,
    int32_t* ZeroPointBBuffer,
    size_t N,
    bool BIsSigned
    )
{
    int32_t ZeroPointB;

    for (size_t n = 0; n < N; n++) {

        ZeroPointB = typename KernelType::OffsetBType(PackedZeroPointB[n]);
        ZeroPointB = MlasGemmU8X8FixupZeroPointB<KernelType>(ZeroPointB, BIsSigned);

        ZeroPointBBuffer[n] = -ZeroPointB;
    }

    //
    // Fill the misaligned slots of the zero point buffer with zeroes to guard
    // against tools that check for uninitialized data usage.
    //

    size_t AlignedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    for (size_t n = N; n < AlignedN; n++) {
        ZeroPointBBuffer[n] = 0;
    }
}

template<typename KernelType>
void
MlasGemmU8X8CopyPackA(
    typename KernelType::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    );

template<typename KernelType>
void
MlasGemmU8X8CopyPackB(
    typename KernelType::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    );

template<typename KernelType>
size_t
MlasGemmU8X8Kernel(
    const typename KernelType::PackedAType* A,
    const typename KernelType::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    );

template<typename KernelType>
void
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const MLAS_GEMM_U8X8_STRIDES& Strides,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{

    typename KernelType::PackedAType* PanelA =
        reinterpret_cast<typename KernelType::PackedAType*>(PackPanels);

    typename KernelType::PackedBType* PanelB = reinterpret_cast<typename KernelType::PackedBType*>(
        PackPanels + MlasQGemmPackPanelASize<KernelType>(Strides));


    MLAS_DECLSPEC_ALIGN(int32_t RowSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ZeroPointBBuffer[1024], 64);

    const size_t K = Shape->K;

    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* B = (const uint8_t*)Data->B + RangeStartN;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;

    int32_t ZeroPointA = Data->ZeroPointA;
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((RangeCountM == 1) &&
        (ZeroPointA == 0) && (PackedZeroPointB == nullptr) && (ZeroPointB == 0) &&
        (Data->OutputProcessor == nullptr)) {
        if (MlasGemmU8X8TryGemvKernel<KernelType>(A, B, ldb, C, K, RangeCountN, Shape->BIsSigned)) {
            return;
        }
    }

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmU8X8FixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmU8X8FixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            MlasGemmU8X8CopyPackB<KernelType>(
                PanelB,
                B + n,
                ldb,
                CountN,
                CountK,
                ColumnSumBuffer,
                Shape->BIsSigned);

            MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, CountN, -ZeroPointA);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmU8X8CopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmU8X8Kernel<KernelType>(
                        pa,
                        PanelB,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}

template<typename KernelType>
void
MlasGemmU8X8PackedOperation(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const MLAS_GEMM_U8X8_STRIDES& Strides,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{
    typename KernelType::PackedAType* PanelA =
        reinterpret_cast<typename KernelType::PackedAType*>(PackPanels);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[1024], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ZeroPointBBuffer[1024], 64);

    const size_t K = Shape->K;

    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)Data->B;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;

    int32_t ZeroPointA = Data->ZeroPointA;
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmU8X8FixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Extract the pointer to the column sum buffer from the packed matrix.
    //

    const size_t AlignedN =
        (Shape->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const uint8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        if (k > 0) {
            std::fill_n(ColumnSumBuffer, Strides.N, 0);
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            if (k == 0) {
                MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer + n,
                    CountN, -ZeroPointA);
            }

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmU8X8FixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const auto* b = reinterpret_cast<const KernelType::PackedBType*>(
                PackedB + (RangeStartN + n) * KernelType::PackedK * PackedCountK);
            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmU8X8CopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmU8X8Kernel<KernelType>(
                        pa,
                        b,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        PackedB = (const uint8_t*)PackedB + AlignedN * CountK;
    }
}


template <typename KernelType>
void
MLASCALL
MlasGemmBatchBlockPartition(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool)
{

    constexpr size_t unitwork = 2097152;
    //
    // Start with strides of 16, to reduce false sharing between threads.
    // increase the strides if K is too small.
    //
    size_t StrideN = 16;
    size_t StrideM = 48;

    size_t multipler = unitwork / (StrideM * StrideN * Shape.K);
    multipler = std::max(multipler, size_t(1));

    StrideN *= multipler;
    if (StrideN >= (Shape.N - (Shape.N / 4))) {
        StrideN = Shape.N;
    }

    multipler = unitwork / (StrideM * StrideN * Shape.K);
    multipler = std::max(multipler, size_t(1));

    StrideM *= multipler;
    if (StrideM >= (Shape.M - (Shape.M / 4))) {
        StrideM = Shape.M;
    }

    size_t numJobN = (Shape.N + StrideN - 1) / StrideN;
    size_t numJobM = (Shape.M + StrideM - 1) / StrideM;
    size_t NumJobPerGemm = numJobM * numJobN;

    constexpr size_t BPanelSize = CacheSize / 3;
    constexpr size_t AllPanelSize = CacheSize * 5 / 8;

    MLAS_GEMM_U8X8_STRIDES Strides;
    Strides.K = std::min(Shape.K, KernelType::Strides.K);

    // try to find a shape for panel B that fill half cache
    const size_t AlignKernelStrideK =
        (Strides.K + KernelType::PackedK - 1) & ~(KernelType::PackedK - 1);

    size_t BStrideLimit =
        std::min(size_t(1024), BPanelSize / (AlignKernelStrideK * sizeof(KernelType::PackedBType)));
    if (BStrideLimit >= StrideN) {
        Strides.N = StrideN;
    } else {
        Strides.N = BStrideLimit & ~15;
    }

    const size_t AlignKernelStrideN = (Strides.N + 15) & ~15;
    size_t AStrideLimit =
        std::min(size_t(1024),
        (AllPanelSize - MlasQGemmPackPanelBSize<KernelType>(Strides)) /
                     (AlignKernelStrideK * sizeof(KernelType::PackedAType)));
    if (AStrideLimit >= StrideM) {
        Strides.M = StrideM;
    } else {
        Strides.M = AStrideLimit - (AStrideLimit % 48);
    }

    MLAS_GEMM_U8X8_OPERATION* GemmU8X8Operation;

    if (DataParams->BIsPacked) {
        GemmU8X8Operation = MlasGemmU8X8PackedOperation<KernelType>;
    } else {
        GemmU8X8Operation = MlasGemmU8X8Operation<KernelType>;
    }

    MlasTrySimpleParallel(ThreadPool, BatchN * NumJobPerGemm, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / NumJobPerGemm;
        const auto blk_i = tid % NumJobPerGemm;

        const auto m_idx = blk_i % numJobM;
        const auto n_idx = blk_i / numJobM; 
        
        const auto StartM = m_idx * StrideM;
        const auto CountM = std::min(StrideM, Shape.M - StartM);
        const auto StartN = n_idx * StrideN;
        const auto CountN = std::min(StrideN, Shape.N - StartN);
        GemmU8X8Operation(&Shape, &(DataParams[gemm_i]), Strides, StartM, CountM,
            StartN, CountN);
    });
}

#if defined(MLAS_SSE2_INTRINSICS)

struct MLAS_GEMM_U8X8_KERNEL_SSE
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{12, 128, 128};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_SSE::Strides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_SSE>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_SSE::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_SSE>(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    uint8_t PaddedMatrixAData[8] = { 0 };

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i ReductionVector = ZeroVector;

        //
        // Zero extend the source bytes to 16-bits and write to the packed
        // buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 2 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //
        // These 16-bit values are also accumulated into an intermediate per-row
        // accumulator. CountK cannot be greater than 128 to avoid overflowing
        // these signed 16-bit accumulators.
        //

        while (k >= 8) {

            __m128i Bytes = _mm_loadl_epi64((const __m128i*)&a[0]);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            ReductionVector = _mm_add_epi16(ReductionVector, Words);

            _mm_storeu_si128((__m128i*)&D[0], Words);

            a += 8;
            D += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* padded = PaddedMatrixAData;
            uint8_t* padded_end = padded + k;

            do {
                padded[0] = a[0];
                padded++;
                a++;
            } while (padded < padded_end);

            __m128i Bytes = _mm_loadl_epi64((__m128i*)PaddedMatrixAData);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            ReductionVector = _mm_add_epi16(ReductionVector, Words);

            //
            // Copy pairs of 16-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t pairs = (k + 1) / 2; pairs > 0; pairs--) {
                *((int32_t*)D) = _mm_cvtsi128_si32(Words);
                D += 2;
                Words = _mm_shuffle_epi32(Words, _MM_SHUFFLE(0, 3, 2, 1));
            }
        }

        //
        // Reduce the partial accumulators.
        //

        ReductionVector = _mm_madd_epi16(ReductionVector, OnesWordBroadcast);
        ReductionVector = _mm_add_epi32(ReductionVector,
            _mm_shuffle_epi32(ReductionVector, _MM_SHUFFLE(3, 2, 3, 2)));
        ReductionVector = _mm_add_epi32(ReductionVector,
            _mm_shuffle_epi32(ReductionVector, _MM_SHUFFLE(0, 1, 0, 1)));

        *RowSumBuffer++ = _mm_cvtsi128_si32(ReductionVector);

        A += lda;
        CountM -= 1;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessSse(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* D,
    __m128i BytesRow0,
    __m128i BytesRow1,
    __m128i BitFlipVector,
    __m128i ColumnSums[2]
    )
{
    __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, BytesRow1);

    BytesInterleaved = _mm_xor_si128(BytesInterleaved, BitFlipVector);

    __m128i WordsInterleaved0 = _mm_srai_epi16(_mm_unpacklo_epi8(BytesInterleaved, BytesInterleaved), 8);
    __m128i WordsInterleaved1 = _mm_srai_epi16(_mm_unpackhi_epi8(BytesInterleaved, BytesInterleaved), 8);

    ColumnSums[0] = _mm_add_epi16(ColumnSums[0], WordsInterleaved0);
    ColumnSums[1] = _mm_add_epi16(ColumnSums[1], WordsInterleaved1);

    _mm_storeu_si128((__m128i*)&D[0], WordsInterleaved0);
    _mm_storeu_si128((__m128i*)&D[8], WordsInterleaved1);
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_SSE>(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    const __m128i BitFlipVector = _mm_set1_epi32(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_SSE::PackedK) {

            __m128i BytesRow0 = _mm_loadl_epi64((const __m128i*)&b[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((const __m128i*)&b[ldb]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            __m128i BytesRow0 = _mm_loadl_epi64((const __m128i*)&b[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);

            D += 16;
        }

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[16];

        _mm_storeu_si128((__m128i*)PaddedMatrixBData, BitFlipVector);

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_SSE::PackedK) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[8]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);
        }

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowSse(
    __m128i ABroadcast,
    const int16_t* B,
    __m128i Accumulators[2]
    )
{
    __m128i BElements0 = _mm_load_si128((__m128i*)&B[0]);
    __m128i BElements1 = _mm_load_si128((__m128i*)&B[8]);

    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_madd_epi16(BElements0, ABroadcast));
    Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_madd_epi16(BElements1, ABroadcast));
}

template<>
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_SSE>(
    const MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        int32_t RowSumValue = RowSumBuffer[0];

        if (ZeroPointB != nullptr) {

            int32_t ScaledRowSumBuffer[8];

            for (size_t i = 0; i < 8; i++) {
                ScaledRowSumBuffer[i] = RowSumValue * ZeroPointB[i];
            }

            ZeroPointB += 8;

            Accumulators[0] = _mm_loadu_si128((__m128i*)&ScaledRowSumBuffer[0]);
            Accumulators[1] = _mm_loadu_si128((__m128i*)&ScaledRowSumBuffer[4]);

        } else {

            Accumulators[0] = _mm_set1_epi32(RowSumValue);
            Accumulators[1] = Accumulators[0];
        }

        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*)&ColumnSumBuffer[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*)&ColumnSumBuffer[4]));
        ColumnSumBuffer += 8;

        //
        // Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the pair of 16-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const int16_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = _mm_loadu_si128((__m128i*)a);
            __m128i ABroadcast;

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(0, 0, 0, 0));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[0], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(1, 1, 1, 1));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[16], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(2, 2, 2, 2));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[32], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(3, 3, 3, 3));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[48], Accumulators);

            a += 4 * 2;
            B += 4 * 16;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = _mm_set1_epi32(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[0], Accumulators);

            a += 2;
            B += 16;
            k -= 1;
        }

        //
        // Output the accumulator block after optionally accumulating the values
        // from matrix C.
        //

        if (CountN >= 8) {

            if (!ZeroMode) {
                Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*)&C[4]));
            }

            _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
            _mm_storeu_si128((__m128i*)&C[4], Accumulators[1]);

            C += 8;
            CountN -= 8;

        } else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                }

                _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadl_epi64((__m128i*)&C[0]));
                }

                _mm_storel_epi64((__m128i*)&C[0], Accumulators[0]);
                C += 2;

                Accumulators[0] = _mm_shuffle_epi32(Accumulators[0], _MM_SHUFFLE(3, 2, 3, 2));
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = _mm_cvtsi128_si32(Accumulators[0]);

                if (!ZeroMode) {
                    AccumulatorValue += C[0];
                }

                C[0] = AccumulatorValue;
            }

            CountN = 0;
        }
    }

    return 1;
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchSse = {
    MlasGemmBatchBlockPartition<MLAS_GEMM_U8X8_KERNEL_SSE>,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedK,
    0,
};

#endif

// N.B. MSVC does not require turning on SSE 4.1 intrinsics and the current use
// for this code is Windows only, so restrict this kernel to that environment.
#if defined(MLAS_SSE2_INTRINSICS) && defined(_MSC_VER)

struct MLAS_GEMM_U8S8_KERNEL_SSE41
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 128, 128};
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_SSE41::Strides;

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i ReductionVector = ZeroVector;

        //
        // Copy the source bytes to the packed buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 4 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //

        while (k >= 8) {

            __m128i Bytes = _mm_loadl_epi64((const __m128i*)&a[0]);

            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);
            ReductionVector = _mm_add_epi32(ReductionVector, _mm_madd_epi16(Words, OnesWordBroadcast));

            _mm_storel_epi64((__m128i*)&D[0], Bytes);

            a += 8;
            D += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            _mm_storel_epi64((__m128i*)&D[0], ZeroVector);

            std::copy_n(&a[0], k, &D[0]);

            __m128i Bytes = _mm_loadl_epi64((__m128i*)&D[0]);
            D += (k + 3) & ~3;

            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);
            ReductionVector = _mm_add_epi32(ReductionVector, _mm_madd_epi16(Words, OnesWordBroadcast));
        }

        //
        // Reduce the partial accumulators.
        //

        ReductionVector = _mm_hadd_epi32(ReductionVector, ReductionVector);
        ReductionVector = _mm_hadd_epi32(ReductionVector, ReductionVector);

        *RowSumBuffer++ = _mm_cvtsi128_si32(ReductionVector);

        A += lda;
        CountM -= 1;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessSse41(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* D,
    __m128i BytesRows[4],
    __m128i OnesByteBroadcast,
    __m128i OnesWordBroadcast,
    __m128i ColumnSums[2]
    )
{
    __m128i PairsInterleaved0 = _mm_unpacklo_epi8(BytesRows[0], BytesRows[1]);
    __m128i PairsInterleaved1 = _mm_unpacklo_epi8(BytesRows[2], BytesRows[3]);

    __m128i QuadsInterleaved0 = _mm_unpacklo_epi16(PairsInterleaved0, PairsInterleaved1);
    __m128i QuadsInterleaved1 = _mm_unpackhi_epi16(PairsInterleaved0, PairsInterleaved1);

    __m128i PairwiseAdd0 = _mm_maddubs_epi16(OnesByteBroadcast, QuadsInterleaved0);
    __m128i PairwiseAdd1 = _mm_maddubs_epi16(OnesByteBroadcast, QuadsInterleaved1);

    PairwiseAdd0 = _mm_madd_epi16(PairwiseAdd0, OnesWordBroadcast);
    PairwiseAdd1 = _mm_madd_epi16(PairwiseAdd1, OnesWordBroadcast);

    ColumnSums[0] = _mm_add_epi32(ColumnSums[0], PairwiseAdd0);
    ColumnSums[1] = _mm_add_epi32(ColumnSums[1], PairwiseAdd1);

    _mm_storeu_si128((__m128i*)&D[0], QuadsInterleaved0);
    _mm_storeu_si128((__m128i*)&D[16], QuadsInterleaved1);
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const __m128i OnesByteBroadcast = _mm_set1_epi8(1);
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    __m128i BytesRows[4];

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK) {

            BytesRows[0] = _mm_loadl_epi64((const __m128i*)&b[ldb * 0]);
            BytesRows[1] = _mm_loadl_epi64((const __m128i*)&b[ldb * 1]);
            BytesRows[2] = _mm_loadl_epi64((const __m128i*)&b[ldb * 2]);
            BytesRows[3] = _mm_loadl_epi64((const __m128i*)&b[ldb * 3]);

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRows[0] = _mm_loadl_epi64((const __m128i*)&b[ldb * 0]);
            BytesRows[1] = _mm_setzero_si128();
            BytesRows[2] = _mm_setzero_si128();
            BytesRows[3] = _mm_setzero_si128();

            if (k >= 2) {
                BytesRows[1] = _mm_loadl_epi64((const __m128i*)&b[ldb * 1]);
            }

            if (k >= 3) {
                BytesRows[2] = _mm_loadl_epi64((const __m128i*)&b[ldb * 2]);
            }

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            D += 32;
        }

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const __m128i ZeroVector = _mm_setzero_si128();

        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[32];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        while (CountK > 0) {

            size_t k = std::min(CountK, MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK);
            CountK -= k;

            _mm_storeu_si128((__m128i*)&PaddedMatrixBData[0], ZeroVector);
            _mm_storeu_si128((__m128i*)&PaddedMatrixBData[16], ZeroVector);

            uint8_t* padded = PaddedMatrixBData;

            do {

                std::copy_n(B, CountN, padded);

                padded += 8;
                B += ldb;
                k -= 1;

            } while (k > 0);

            BytesRows[0] = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);
            BytesRows[1] = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[8]);
            BytesRows[2] = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[16]);
            BytesRows[3] = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            D += 32;
        }

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowSse41(
    __m128i ABroadcast,
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* B,
    __m128i OnesWordBroadcast,
    __m128i Accumulators[2]
    )
{
    __m128i BElements0 = _mm_load_si128((__m128i*)&B[0]);
    __m128i BElements1 = _mm_load_si128((__m128i*)&B[16]);

    __m128i Intermediate0 = _mm_maddubs_epi16(ABroadcast, BElements0);
    __m128i Intermediate1 = _mm_maddubs_epi16(ABroadcast, BElements1);

    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_madd_epi16(Intermediate0, OnesWordBroadcast));
    Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_madd_epi16(Intermediate1, OnesWordBroadcast));
}

template<>
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedAType* A,
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);

    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        Accumulators[0] = _mm_set1_epi32(RowSumBuffer[0]);
        Accumulators[1] = Accumulators[0];

        if (ZeroPointB != nullptr) {
            Accumulators[0] = _mm_mullo_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*)&ZeroPointB[0]));
            Accumulators[1] = _mm_mullo_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*)&ZeroPointB[4]));
            ZeroPointB += 8;
        }

        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*)&ColumnSumBuffer[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*)&ColumnSumBuffer[4]));
        ColumnSumBuffer += 8;

        //
        // Broadcast each quad of 8-bit values from the matrix A and multiply
        // with the quad of 8-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const uint8_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = _mm_loadu_si128((__m128i*)a);
            __m128i ABroadcast;

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(0, 0, 0, 0));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[0], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(1, 1, 1, 1));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[32], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(2, 2, 2, 2));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[64], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(3, 3, 3, 3));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[96], OnesWordBroadcast, Accumulators);

            a += 4 * 4;
            B += 4 * 32;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = _mm_set1_epi32(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[0], OnesWordBroadcast, Accumulators);

            a += 4;
            B += 32;
            k -= 1;
        }

        //
        // Output the accumulator block after optionally accumulating the values
        // from matrix C.
        //

        if (CountN >= 8) {

            if (!ZeroMode) {
                Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*)&C[4]));
            }

            _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
            _mm_storeu_si128((__m128i*)&C[4], Accumulators[1]);

            C += 8;
            CountN -= 8;

        } else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                }

                _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadl_epi64((__m128i*)&C[0]));
                }

                _mm_storel_epi64((__m128i*)&C[0], Accumulators[0]);
                C += 2;

                Accumulators[0] = _mm_shuffle_epi32(Accumulators[0], _MM_SHUFFLE(3, 2, 3, 2));
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = _mm_cvtsi128_si32(Accumulators[0]);

                if (!ZeroMode) {
                    AccumulatorValue += C[0];
                }

                C[0] = AccumulatorValue;
            }

            CountN = 0;
        }
    }

    return 1;
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8S8DispatchSse41 = {
    MlasGemmBatchBlockPartition<MLAS_GEMM_U8S8_KERNEL_SSE41>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_SSE41>,
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK,
    MLAS_GEMM_U8S8_KERNEL_SSE41::Strides.K,
};

#endif

#if defined(MLAS_TARGET_AMD64)

//
// Stores a vector to transpose a 4x4 byte vector using vpshufb.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint8_t MlasTranspose4x4BytesAvx[16], 16) =
    { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

//
// Define the prototypes of the AVX2/AVX512 routines written in assembly.
//

extern "C" {

    void
    MLASCALL
    MlasGemmU8S8CopyPackAAvx2(
        uint8_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8S8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackAAvx2(
        int16_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer
        );
}

struct MLAS_GEMM_U8S8_KERNEL_AVX2
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{48, 256, 768};
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::Strides;

template<>
MLAS_FORCEINLINE
bool
MlasGemmU8X8TryGemvKernel<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const uint8_t* A,
    const uint8_t* B,
    size_t ldb,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        MlasPlatform.GemvU8S8Kernel(A, B, C, CountK, CountN, ldb);
        return true;
    }

    return false;
}

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8S8_KERNEL_AVX2::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    MlasGemmU8S8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MlasGemmU8S8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const MLAS_GEMM_U8S8_KERNEL_AVX2::PackedAType* A,
    const MLAS_GEMM_U8S8_KERNEL_AVX2::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasPlatform.GemmU8S8Kernel(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8S8DispatchAvx2 = {
    MlasGemmBatchBlockPartition<MLAS_GEMM_U8S8_KERNEL_AVX2>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8S8_KERNEL_AVX2>,
    MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK,
    MLAS_GEMM_U8S8_KERNEL_AVX2::Strides.K,
};

struct MLAS_GEMM_U8U8_KERNEL_AVX2
{
    typedef int16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{48, 256, 768};
};

constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::Strides;

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    MlasGemmU8U8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
}

template<>
MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    MlasGemmU8U8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer);
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    const MLAS_GEMM_U8U8_KERNEL_AVX2::PackedAType* A,
    const MLAS_GEMM_U8U8_KERNEL_AVX2::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasPlatform.GemmU8U8Kernel(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8U8DispatchAvx2 = {
    MlasGemmBatchBlockPartition<MLAS_GEMM_U8U8_KERNEL_AVX2>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8U8_KERNEL_AVX2>,
    MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK,
    MLAS_GEMM_U8U8_KERNEL_AVX2::Strides.K,
};

#endif

#if defined(MLAS_NEON64_INTRINSICS) || (defined(MLAS_NEON32_INTRINSICS) && !defined(_MSC_VER))

//
// Define the prototypes of the NEON routines written in assembly.
//
// N.B. The kernel has not been ported to build with the Windows ARM32 toolset.
//

extern "C" {

    size_t
    MLASCALL
    MlasGemmU8X8KernelNeon(
        const uint8_t* A,
        const uint8_t* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumVector,
        const int32_t* ColumnSumVector,
        const int32_t* ZeroPointB,
        bool ZeroMode
        );
}

struct MLAS_GEMM_U8X8_KERNEL_NEON
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 128, 256};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::Strides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_NEON>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_NEON::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_NEON>(
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    uint8_t PaddedMatrixAData[16];

    //
    // Process four rows of matrix A in a loop.
    //
    // The buffer is packed as a series of 16 byte vectors where four rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    while (CountM >= 4) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;
        const uint8_t* a2 = a1 + lda;
        const uint8_t* a3 = a2 + lda;

        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint32x4_t v0 = vld1q_u32(reinterpret_cast<const uint32_t*>(a0));
            a0 += 16;
            uint32x4_t v1 = vld1q_u32(reinterpret_cast<const uint32_t*>(a1));
            a1 += 16;
            uint32x4_t v2 = vld1q_u32(reinterpret_cast<const uint32_t*>(a2));
            a2 += 16;
            uint32x4_t v3 = vld1q_u32(reinterpret_cast<const uint32_t*>(a3));
            a3 += 16;

#if defined(MLAS_NEON32_INTRINSICS)
            uint32x4x2_t z0 = vzipq_u32(v0, v2);
            uint32x4x2_t z1 = vzipq_u32(v1, v3);

            v0 = z0.val[0];
            v1 = z0.val[1];
            v2 = z1.val[0];
            v3 = z1.val[1];

            uint32x4x2_t z2 = vzipq_u32(v0, v2);
            uint32x4x2_t z3 = vzipq_u32(v1, v3);

            v0 = z2.val[0];
            v1 = z2.val[1];
            v2 = z3.val[0];
            v3 = z3.val[1];
#else
            uint32x4_t z0 = vzip1q_u32(v0, v2);
            uint32x4_t z1 = vzip2q_u32(v0, v2);
            uint32x4_t z2 = vzip1q_u32(v1, v3);
            uint32x4_t z3 = vzip2q_u32(v1, v3);

            v0 = vzip1q_u32(z0, z2);
            v1 = vzip2q_u32(z0, z2);
            v2 = vzip1q_u32(z1, z3);
            v3 = vzip2q_u32(z1, z3);
#endif

            vst1q_u8(&D[0], vreinterpretq_u8_u32(v0));
            vst1q_u8(&D[16], vreinterpretq_u8_u32(v1));
            vst1q_u8(&D[32], vreinterpretq_u8_u32(v2));
            vst1q_u8(&D[48], vreinterpretq_u8_u32(v3));

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v0)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v1)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v2)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v3)));

            D += 64;
            k -= 16;
        }

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;
            uint32_t v2 = *reinterpret_cast<const uint32_t*>(a2);
            a2 += 4;
            uint32_t v3 = *reinterpret_cast<const uint32_t*>(a3);
            a3 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;
            *reinterpret_cast<uint32_t*>(&D[8]) = v2;
            *reinterpret_cast<uint32_t*>(&D[12]) = v3;

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vld1q_u8(D)));

            D += 16;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;
                d[8] = *a2++;
                d[12] = *a3++;

                d += 1;
                k -= 1;
            }

            uint8x16_t PackedVector = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, PackedVector);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(PackedVector));

            D += 16;
        }

        vst1q_s32(RowSumBuffer, vreinterpretq_s32_u32(RowSums));
        RowSumBuffer += 4;

        A = A + lda * 4;
        CountM -= 4;
    }

    //
    // Process two rows of matrix A.
    //
    // The buffer is packed as a series of 8 byte vectors where two rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if ((CountM & 2) != 0) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;

        size_t k = CountK;
        uint32x2_t RowSums = vmov_n_u32(0);

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;

            RowSums = vpadal_u16(RowSums, vpaddl_u8(vld1_u8(D)));

            D += 8;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;

                d += 1;
                k -= 1;
            }

            uint8x8_t PackedVector = vld1_u8(PaddedMatrixAData);
            vst1_u8(D, PackedVector);

            RowSums = vpadal_u16(RowSums, vpaddl_u8(PackedVector));

            D += 8;
        }

        vst1_s32(RowSumBuffer, vreinterpret_s32_u32(RowSums));
        RowSumBuffer += 2;

        A = A + lda * 2;
        CountM -= 2;
    }

    //
    // Process one row of matrix A.
    //
    // The buffer is packed as a series of 4 byte with the following pattern:
    //
    //      [ A0 A1 A2 A3 ]
    //      [ A4 A5 A6 A7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if ((CountM & 1) != 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint8x16_t v = vld1q_u8(a);
            a += 16;

            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));

            D += 16;
            k -= 16;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            for (size_t kk = 0; kk < k; kk++) {
                PaddedMatrixAData[kk] = a[kk];
            }

            uint8x16_t v = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));
        }

#if defined(MLAS_NEON32_INTRINSICS)
        uint32x2_t RowSumsLow = vpadd_u32(vget_high_u32(RowSums), vget_low_u32(RowSums));
        RowSumsLow = vpadd_u32(RowSumsLow, RowSumsLow);
        vst1_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSumsLow, 0);
#elif defined(_M_ARM64)
        // N.B. The workaround of defining a local vaddvq_u32 doesn't work here
        // as VS2019 added new intrinsics to make the operation work. Also, not
        // all build environments using VS2019 have the up-to-date arm64_neon.h,
        // so fallback to pairwise addition.
        RowSums = vpaddq_u32(RowSums, RowSums);
        RowSums = vpaddq_u32(RowSums, RowSums);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSums, 0);
#else
        *RowSumBuffer = int32_t(vaddvq_u32(RowSums));
#endif
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessNeon(
    uint8_t* D,
    const uint8_t* B,
    uint8x8_t BitFlipVector,
    uint32x4_t ColumnSums[2]
    )
{
    uint8x8_t BytesRow = veor_u8(vld1_u8(B), BitFlipVector);
    vst1_u8(D, BytesRow);

    uint16x8_t WordsRow = vmovl_u8(BytesRow);
    ColumnSums[0] = vaddq_u32(ColumnSums[0], vmovl_u16(vget_low_u16(WordsRow)));
#if defined(MLAS_NEON32_INTRINSICS)
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_u16(vget_high_u16(WordsRow)));
#else
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_high_u16(WordsRow));
#endif
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_NEON>(
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8x8_t BitFlipVector = vdup_n_u8(BIsSigned ? 0x80 : 0);
    const uint8x8_t ZeroVector = vmov_n_u8(0);
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_NEON::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_NEON::PackedK - 1);

    //
    // Process 8 columns of matrix B in a loop.
    //
    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of four, then the packed buffer
    // is padded with zero vectors.
    //
    // If CountN is not aligned to a multiple of four, then the extra columns
    // are padded with zeroes.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        uint32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        for (size_t k = CountK; k > 0; k--) {

            MlasGemmU8X8CopyPackBProcessNeon(D, b, BitFlipVector, ColumnSums);

            b += ldb;
            D += 8;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            vst1_u8(D, ZeroVector);
            D += 8;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        uint8_t PaddedMatrixBData[8];
        uint32x4_t ColumnSums[2];

        vst1_u8(PaddedMatrixBData, ZeroVector);

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        for (size_t k = CountK; k > 0; k--) {

            for (size_t n = 0; n < CountN; n++) {
                PaddedMatrixBData[n] = b[n];
            }

            MlasGemmU8X8CopyPackBProcessNeon(D, PaddedMatrixBData, BitFlipVector, ColumnSums);

            b += ldb;
            D += 8;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            vst1_u8(D, ZeroVector);
            D += 8;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
    }
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_NEON>(
    const MLAS_GEMM_U8X8_KERNEL_NEON::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_NEON::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasGemmU8X8KernelNeon(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchNeon = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedK,
    MLAS_GEMM_U8X8_KERNEL_NEON::Strides.K,
};

#endif

#if defined(MLAS_NEON64_INTRINSICS)

//
// Define the prototypes of the NEON UDOT routines written in assembly.
//

extern "C" {

    size_t
    MLASCALL
    MlasGemmU8X8KernelUdot(
        const uint8_t* A,
        const uint8_t* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumVector,
        const int32_t* ColumnSumVector,
        const int32_t* ZeroPointB,
        bool ZeroMode
        );
}

struct MLAS_GEMM_U8X8_KERNEL_UDOT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 128, 384};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_UDOT::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_UDOT::Strides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_UDOT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    uint8_t PaddedMatrixAData[16];

    //
    // Process four rows of matrix A.
    //
    // The buffer is packed as a series of 16 byte vectors where four rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of eight, then the vector is padded
    // with zeroes.
    //

    while (CountM >= 4) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;
        const uint8_t* a2 = a1 + lda;
        const uint8_t* a3 = a2 + lda;

        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint32x4_t v0 = vld1q_u32(reinterpret_cast<const uint32_t*>(a0));
            a0 += 16;
            uint32x4_t v1 = vld1q_u32(reinterpret_cast<const uint32_t*>(a1));
            a1 += 16;
            uint32x4_t v2 = vld1q_u32(reinterpret_cast<const uint32_t*>(a2));
            a2 += 16;
            uint32x4_t v3 = vld1q_u32(reinterpret_cast<const uint32_t*>(a3));
            a3 += 16;

            uint32x4_t z0 = vzip1q_u32(v0, v2);
            uint32x4_t z1 = vzip2q_u32(v0, v2);
            uint32x4_t z2 = vzip1q_u32(v1, v3);
            uint32x4_t z3 = vzip2q_u32(v1, v3);

            v0 = vzip1q_u32(z0, z2);
            v1 = vzip2q_u32(z0, z2);
            v2 = vzip1q_u32(z1, z3);
            v3 = vzip2q_u32(z1, z3);

            vst1q_u8(&D[0], vreinterpretq_u8_u32(v0));
            vst1q_u8(&D[16], vreinterpretq_u8_u32(v1));
            vst1q_u8(&D[32], vreinterpretq_u8_u32(v2));
            vst1q_u8(&D[48], vreinterpretq_u8_u32(v3));

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v0)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v1)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v2)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v3)));

            D += 64;
            k -= 16;
        }

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;
            uint32_t v2 = *reinterpret_cast<const uint32_t*>(a2);
            a2 += 4;
            uint32_t v3 = *reinterpret_cast<const uint32_t*>(a3);
            a3 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;
            *reinterpret_cast<uint32_t*>(&D[8]) = v2;
            *reinterpret_cast<uint32_t*>(&D[12]) = v3;

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vld1q_u8(D)));

            D += 16;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;
                d[8] = *a2++;
                d[12] = *a3++;

                d += 1;
                k -= 1;
            }

            uint8x16_t PackedVector = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, PackedVector);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(PackedVector));

            D += 16;
        }

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(D, vmovq_n_u8(0));

            D += 16;
        }

        vst1q_s32(RowSumBuffer, vreinterpretq_s32_u32(RowSums));
        RowSumBuffer += 4;

        A = A + lda * 4;
        CountM -= 4;
    }

    //
    // Process two rows of matrix A.
    //
    // The buffer is packed as a series of 8 byte vectors where two rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if (CountM >= 2) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;

        size_t k = CountK;
        uint32x2_t RowSums = vmov_n_u32(0);

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;

            RowSums = vpadal_u16(RowSums, vpaddl_u8(vld1_u8(D)));

            D += 8;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1_u8(PaddedMatrixAData, vmov_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;

                d += 1;
                k -= 1;
            }

            uint8x8_t PackedVector = vld1_u8(PaddedMatrixAData);
            vst1_u8(D, PackedVector);

            RowSums = vpadal_u16(RowSums, vpaddl_u8(PackedVector));

            D += 8;
        }

        if (((CountK - 1) & 7) < 4) {

            vst1_u8(D, vmov_n_u8(0));

            D += 8;
        }

        vst1_s32(RowSumBuffer, vreinterpret_s32_u32(RowSums));
        RowSumBuffer += 2;

        A = A + lda * 2;
        CountM -= 2;
    }

    //
    // Process one row of matrix A.
    //
    // The buffer is packed as a series of 4 byte with the following pattern:
    //
    //      [ A0 A1 A2 A3 ]
    //      [ A4 A5 A6 A7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint8x16_t v = vld1q_u8(a);
            a += 16;

            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));

            D += 16;
            k -= 16;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            for (size_t kk = 0; kk < k; kk++) {
                PaddedMatrixAData[kk] = a[kk];
            }

            uint8x16_t v = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));
        }

#if defined(_M_ARM64)
        // N.B. The workaround of defining a local vaddvq_u32 doesn't work here
        // as VS2019 added new intrinsics to make the operation work. Also, not
        // all build environments using VS2019 have the up-to-date arm64_neon.h,
        // so fallback to pairwise addition.
        RowSums = vpaddq_u32(RowSums, RowSums);
        RowSums = vpaddq_u32(RowSums, RowSums);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSums, 0);
#else
        *RowSumBuffer = int32_t(vaddvq_u32(RowSums));
#endif
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessUdot(
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* D,
    uint8x8_t BytesRow[4],
    uint8x16_t BitFlipVector,
    uint32x4_t ColumnSums[2]
    )
{
    uint8x16_t v02 = veorq_u8(vcombine_u8(BytesRow[0], BytesRow[2]), BitFlipVector);
    uint8x16_t v13 = veorq_u8(vcombine_u8(BytesRow[1], BytesRow[3]), BitFlipVector);

    uint8x16x2_t zw = vzipq_u8(v02, v13);
    uint16x8x2_t zd = vzipq_u16(vreinterpretq_u16_u8(zw.val[0]), vreinterpretq_u16_u8(zw.val[1]));

    vst1q_u8(&D[0], vreinterpretq_u8_u16(zd.val[0]));
    vst1q_u8(&D[16], vreinterpretq_u8_u16(zd.val[1]));

    ColumnSums[0] = vpadalq_u16(ColumnSums[0], vpaddlq_u8(vreinterpretq_u8_u16(zd.val[0])));
    ColumnSums[1] = vpadalq_u16(ColumnSums[1], vpaddlq_u8(vreinterpretq_u8_u16(zd.val[1])));
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8x16_t ZeroVector = vmovq_n_u8(0);
    const uint8x16_t BitFlipVector = vdupq_n_u8(BIsSigned ? 0x80 : 0);
    uint8x8_t BytesRow[4];

    //
    // Process 8 columns of matrix B in a loop.
    //
    // The buffer is packed as a series of 16 byte vectors where eight rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ E0 E1 E2 E3 F0 F1 F2 F3 G0 G1 G2 G3 H0 H1 H2 H3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //      [ E4 E5 E6 E7 F4 F5 F6 F7 G4 G5 G6 G7 H4 H5 H6 H7 ]
    //
    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of eight, then the packed buffer
    // is padded with zero vectors.
    //
    // If CountN is not aligned to a multiple of four, then the extra columns
    // are padded with zeroes.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        uint32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //

        while (k >= 4) {

            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = vld1_u8(&b[ldb * 1]);
            BytesRow[2] = vld1_u8(&b[ldb * 2]);
            BytesRow[3] = vld1_u8(&b[ldb * 3]);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = (k >= 2) ? vld1_u8(&b[ldb * 1]) : vget_low_u8(BitFlipVector);
            BytesRow[2] = (k > 2) ? vld1_u8(&b[ldb * 2]) : vget_low_u8(BitFlipVector);
            BytesRow[3] = vget_low_u8(BitFlipVector);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(&D[0], ZeroVector);
            vst1q_u8(&D[16], ZeroVector);

            D += 32;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        size_t k = CountK;
        uint8_t PaddedMatrixBData[32];
        uint32x4_t ColumnSums[2];

        vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
        vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k > 0) {

            const uint8_t* bcopy0 = &b[ldb * 0];
            const uint8_t* bcopy1 = &b[ldb * 1];
            const uint8_t* bcopy2 = &b[ldb * 2];
            const uint8_t* bcopy3 = &b[ldb * 3];

            if (k >= 4) {

                b += ldb * 4;
                k -= 4;

            } else {

                vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
                vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);

                bcopy1 = (k >= 2) ? bcopy1 : &PaddedMatrixBData[24];
                bcopy2 = (k > 2) ? bcopy2 : &PaddedMatrixBData[24];
                bcopy3 = &PaddedMatrixBData[24];

                k = 0;
            }

            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = *bcopy0++;
                padded[8] = *bcopy1++;
                padded[16] = *bcopy2++;
                padded[24] = *bcopy3++;
            } while (++padded < padded_end);

            BytesRow[0] = vld1_u8(&PaddedMatrixBData[0]);
            BytesRow[1] = vld1_u8(&PaddedMatrixBData[8]);
            BytesRow[2] = vld1_u8(&PaddedMatrixBData[16]);
            BytesRow[3] = vld1_u8(&PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(&D[0], ZeroVector);
            vst1q_u8(&D[16], ZeroVector);

            D += 32;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
    }
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    const MLAS_GEMM_U8X8_KERNEL_UDOT::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    return MlasGemmU8X8KernelUdot(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchUdot = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedK,
    MLAS_GEMM_U8X8_KERNEL_UDOT::Strides.K,
};

#endif

struct MLAS_GEMM_U8X8_KERNEL_DEFAULT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{16, 128, 128};
};

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_DEFAULT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM-- > 0) {

        int32_t RowSum = 0;

        for (size_t k = 0; k < CountK; k++) {

            uint8_t a0 = A[k];
            D[k] = a0;

            RowSum += a0;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *RowSumBuffer++ = RowSum;

        A += lda;
        D += AlignedCountK;
    }
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK - 1);
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);

    //
    // Process a single column of matrix B in a loop.
    //

    while (CountN-- > 0) {

        const uint8_t* b = B;
        int32_t ColumnSum = 0;

        //
        // Transpose the data from matrix B to the packed buffer.
        //

        for (size_t k = 0; k < CountK; k++) {

            uint8_t b0 = b[0] ^ BitFlipValue;
            D[k] = b0;

            ColumnSum += b0;

            b += ldb;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            D[k] = 0;
        }

        *ColumnSumBuffer++ = ColumnSum;

        B += 1;
        D += AlignedCountK;
    }
}

template<>
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_DEFAULT>(
    const MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    //
    // Process a single column of matrix B in a loop.
    //

    while (CountN-- > 0) {

        int32_t Accumulator = *RowSumBuffer;

        if (ZeroPointB != nullptr) {
            Accumulator *= *ZeroPointB++;
        }

        Accumulator += *ColumnSumBuffer++;

        const auto* a = A;

        for (size_t k = 0; k < PackedCountK; k++) {

            Accumulator += a[0] * B[0];
            Accumulator += a[1] * B[1];
            Accumulator += a[2] * B[2];
            Accumulator += a[3] * B[3];

            a += 4;
            B += 4;
        }

        if (!ZeroMode) {
            Accumulator += C[0];
        }

        C[0] = Accumulator;
        C += 1;
    }

    return 1;
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchDefault = {
    MlasGemmBatchBlockPartition<MLAS_GEMM_U8X8_KERNEL_DEFAULT>,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_DEFAULT::PackedK,
    0,
};

void
MLASCALL
MlasGemm(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS &Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS &DataParams,
    MLAS_THREADPOOL *ThreadPool)
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MlasGemmBatch(Shape, &DataParams, 1, ThreadPool);
}

void
MLASCALL
MlasGemmBatch(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool)
{
    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(Shape.BIsSigned);
    GemmU8X8Dispatch->Operation(Shape, DataParams, BatchN, ThreadPool);

}


size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool BIsSigned
    )
/*++

Routine Description:

    This routine computes the number of bytes required to pack a matrix with
    the supplied shape and type.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

Return Value:

    Returns the number of bytes required to pack the matrix, else zero if the
        current implementation does not support packing.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(BIsSigned);

    size_t PackedK = GemmU8X8Dispatch->PackedK;
    size_t PackedStrideK = GemmU8X8Dispatch->StrideK;

    if (PackedStrideK == 0) {
        return 0;
    }

    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);

    const size_t BytesRequired =
        (AlignedN * sizeof(int32_t)) + (AlignedN * AlignedK * sizeof(uint8_t));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedB
    )
/*++

Routine Description:

    This routine packs the supplied matrix B to the supplied packed matrix B
    buffer. The size of the packed buffer was obtained from MlasGemmPackBSize.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmU8X8Dispatch = MlasGemmU8X8GetDispatch(BIsSigned);

    size_t PackedK = GemmU8X8Dispatch->PackedK;
    size_t PackedStrideK = GemmU8X8Dispatch->StrideK;

    //
    // Reserve and initialize storage for the column sum buffer to hold the sums
    // of the elements along each of the columns.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    int32_t* PackedColumnSumBuffer = (int32_t*)PackedB;
    std::fill_n(PackedColumnSumBuffer, AlignedN, 0);
    PackedB = PackedColumnSumBuffer + AlignedN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, PackedStrideK);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (CountK + PackedK - 1) & ~(PackedK - 1);
        uint8_t* pb = (uint8_t*)PackedB;
        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(N - n, BatchedN);

            GemmU8X8Dispatch->CopyPackBRoutine(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
            }

            pb += CountN * AlignedK;
        }

        PackedB = (uint8_t*)PackedB + AlignedN * AlignedK;
        B += ldb * CountK;
    }
}
