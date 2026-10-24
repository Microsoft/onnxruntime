# MatMulBlockQuantizedFp8Weight - CUDA Experiments

## Opt-in two-row per-column specialization

`ORT_FP8_GEMV_PER_COLUMN_S2=1` specializes the existing MMA GEMV body for
two-row weight-only projections with one scale per output column
(`block_size == K`, no activation scale, and no pinned-residency dispatch).
It removes per-block scale indexing and activation-QDQ bookkeeping while
retaining the MMA accumulation, K-split reduction, optional bias, and split
outputs. Unsupported shapes and activation-QDQ inputs retain existing dispatch.
The default is `0`; other integer values are rejected. Set the option before
the first applicable invocation.

FP16 and BF16 exact-output tests cover K=320/640/2560/6144, partial output
tiles, bias, and both normal and split outputs. Performance results from
experimental packages do not qualify model accuracy or arbitrary shapes.

This document records CUDA performance experiments for
**MatMulBlockQuantizedFp8Weight** (`com.microsoft::MatMulBlockQuantizedFp8Weight`).
It includes retained optimizations and their measured effect so future tuning
does not repeat work whose results are already known.

Related documentation:

- [matmul_block_scaled_fp8.md](matmul_block_scaled_fp8.md) - operator behavior and current dispatch chain.

---

## Table of Contents

1. [Test Environment](#1-test-environment)
2. [Baseline Latency Profile](#2-baseline-latency-profile)
3. [Prefill Bottleneck - Weight Dequantization](#3-prefill-bottleneck---weight-dequantization)
4. [Optimization - Vectorized Dequantization Kernel](#4-optimization---vectorized-dequantization-kernel)
5. [Decode GEMV - Memory-Level Parallelism](#5-decode-gemv---memory-level-parallelism)
6. [Decode GEMV - Tensor Cores](#6-decode-gemv---tensor-cores)
7. [Benchmark Commands](#7-benchmark-commands)
8. [Lessons](#8-lessons)

---

## 1. Test Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, SM120 (Blackwell), 36 SMs, about 448 GB/s memory bandwidth.
  Section 5 was measured separately on an NVIDIA H200, SM90 (Hopper), 132 SMs, about 4.8 TB/s.
- CUDA toolkit: 13.0.
- CUTLASS: 4.4.2.
- Build directory: `build/cu130/Release`.
- Benchmark shape unless stated otherwise: `N=4096`, `K=4096`, `block_size=128`.
- Data: FP16 activation `A`, FP8 E4M3 weight `B`, FP32 per-block scales, FP16 output.
- Timing: warmup then measured iterations (see commands in section 6); latency is the mean over measured iterations.
- Device selection: `CUDA_VISIBLE_DEVICES=0`.

The operator is weight-only: `A` stays FP16/BF16, `B` is dequantized to the
activation type, and the product runs either as a fused decode GEMV (small `M`)
or as a cuBLAS GEMM on the dequantized weight (larger `M`). There is no native
FP8 block-scaled tensor-core path; the design is architecture independent
(SM80+).

---

## 2. Baseline Latency Profile

Measured latency of the two dispatch paths at `N=4096, K=4096, fp16` before the
dequantization kernel was optimized:

| M | Path | Mean latency | Notes |
|----|------|--------------|-------|
| 1 | decode GEMV | 0.076 ms | one warp per output column, row group 1 |
| 2 | decode GEMV | 0.097 ms | row group 2 |
| 4 | decode GEMV | 0.123 ms | row group 4 |
| 8 | decode GEMV | 0.132 ms | row group 8 |
| 16 | dequant + cuBLAS | 0.396 ms | dequant dominated |
| 32 | dequant + cuBLAS | 0.396 ms | dequant dominated |
| 64 | dequant + cuBLAS | 0.399 ms | dequant dominated |
| 128 | dequant + cuBLAS | 0.453 ms | dequant + growing GEMM |
| 256 | dequant + cuBLAS | 0.596 ms | GEMM growing |
| 512 | dequant + cuBLAS | 0.732 ms | GEMM significant |

The flat 0.396-0.399 ms across `M = 16, 32, 64` is the tell: the prefill cost
does not depend on `M` there, so the GEMM is negligible and a fixed per-shape
cost dominates.

---

## 3. Prefill Bottleneck - Weight Dequantization

The default prefill path expands the whole `[N, K]` weight into an FP16/BF16
scratch buffer before the cuBLAS GEMM. That expansion moves `N*K` bytes in
(FP8) and `2*N*K` bytes out (FP16/BF16), independent of `M`. To confirm this is
the bottleneck, prefill latency at `M=16` was measured while varying `N*K`:

| N | K | N*K | Mean latency |
|------|------|--------|--------------|
| 4096 | 2048 | 8.4 M | 0.203 ms |
| 2048 | 4096 | 8.4 M | 0.194 ms |
| 4096 | 4096 | 16.8 M | 0.399 ms |
| 4096 | 8192 | 33.6 M | 0.730 ms |
| 8192 | 4096 | 33.6 M | 0.733 ms |

Latency scales linearly with `N*K` and is independent of whether `N` or `K`
grows, which matches a memory-bound dequantization and rules out the GEMM as the
prefill bottleneck. At 16.8 M elements in 0.40 ms the effective traffic is only
about 125 GB/s, roughly 28% of the 448 GB/s peak.

The original kernel `DequantizeBlockScaledFp8Kernel` mapped one thread to one
output element and computed, per element:

```cpp
const int row = static_cast<int>(idx / k);              // 64-bit division per element
const int col = static_cast<int>(idx - (long long)row * k);
const int blk = col / block_size;
out[idx] = FromFloat<T>(static_cast<float>(b_fp8[idx]) * weight_scale[row * k_blocks + blk]);
```

Two problems limited bandwidth: a 64-bit `idx / k` division on every element, and
scalar 1-byte loads / 2-byte stores that do not form wide coalesced memory
transactions.

---

## 4. Optimization - Vectorized Dequantization Kernel

`DequantizeBlockScaledFp8Vec16Kernel` replaces the scalar kernel when
`K % 16 == 0` (the common layout; the scalar kernel is kept for `K % 16 != 0`).
Each thread converts one aligned 16-element K chunk of a single row:

- **Coalesced wide memory access.** The 16 FP8 values load as one 16-byte
  `uint4`; the 16 FP16/BF16 results store as a 32-byte pair of `uint4`. Because
  `K % 16 == 0` every row begins 16-byte aligned, so both accesses are aligned
  and 32 threads of a warp cover 512 contiguous bytes.
- **No per-element division.** The row index comes from a 2D grid
  (`row = blockIdx.y`, with a grid-stride loop when `N` exceeds 65535), so the
  expensive `idx / k` division is gone entirely.
- **One scale load per chunk.** When `block_size % 16 == 0`, all 16 elements of a
  chunk fall in the same K block, so the kernel loads a single `b_scale` value
  per chunk instead of one per element. The general case still does a per-element
  block lookup.

Measured effect at `N=4096, K=4096, fp16`:

| M | Before | After | Speedup |
|----|--------|-------|---------|
| 16 | 0.396 ms | 0.262 ms | 1.51x |
| 32 | 0.396 ms | 0.260 ms | 1.52x |
| 64 | 0.399 ms | 0.261 ms | 1.53x |
| 128 | 0.453 ms | 0.303 ms | 1.50x |
| 256 | 0.596 ms | 0.392 ms | 1.52x |
| 512 | 0.732 ms | 0.584 ms | 1.25x |

Larger weight, `N=11008, K=4096, fp16`:

| M | Before | After | Speedup |
|----|--------|-------|---------|
| 32 | 0.970 ms | 0.662 ms | 1.47x |
| 128 | 1.005 ms | 0.713 ms | 1.41x |

Effective dequantization bandwidth (counting 3 bytes per element: 1 read + 2
write) after the change, from the `N*K` scaling sweep at `M=16`:

| N*K | Mean latency | Effective bandwidth |
|--------|--------------|---------------------|
| 8.4 M | 0.133 ms | about 189 GB/s |
| 16.8 M | 0.258 ms | about 195 GB/s |
| 33.6 M | 0.484 ms | about 208 GB/s |

Bandwidth rose from about 125 GB/s to about 195-208 GB/s (about 1.55x), and the
marginal throughput between the smallest and largest sweep points is about
215 GB/s. Accuracy is unchanged: the focused C++ tests and the Python harness
accuracy checks pass at every measured shape.

This optimization is kept.

---

## 5. Decode GEMV - Memory-Level Parallelism

An earlier round of this document concluded the decode GEMV was "already well
tuned and not the optimization target". That conclusion came from measurements
taken through the ORT Python API without CUDA graphs, which on a fast GPU are
dominated by per-node host overhead rather than by the kernel. Re-measuring on
H200 (SM90) with the launches captured in a CUDA graph changed the picture.

### 5.1 What the measurement was actually reporting

At `N = 8192, K = 2048, M = 1` the op-level measurement reported 21.0 us while
the kernel itself takes 7.2 us. The difference is ORT per-node host work; the
host, not the GPU, was the limiter in that harness. Two rules follow:

- Measure kernels standalone, or with the launches captured in a CUDA graph.
- On H200 an empty kernel costs **1.79 us** as a stream launch and **0.68 us**
  as a CUDA graph node. That 0.68 us is a floor no kernel optimization can go
  below, and it dominates any op whose useful work is smaller.

### 5.2 Nsight Compute diagnosis

`ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy
--section WarpStateStats` on the original kernel at `N = 8192, K = 2048, M = 1`:

| Metric | Value | Reading |
|---|---|---|
| DRAM throughput | 35.3% | not bandwidth bound |
| Compute (SM) throughput | 52.3% | not compute bound either |
| Waves | 1.0 | grid barely fills the GPU once |
| L1/TEX hit rate | 79.6% | the `A` row is already resident in L1 |
| L2 hit rate | 5.2% | `A` re-reads never reach L2 |
| Block limit (registers) | 8 blocks/SM | occupancy already register capped |
| Achieved occupancy | 71.0% | vs 100% theoretical |
| Warp cycles / issued instr | 16.7, of which 5.5 on L1TEX | latency bound |

The kernel is short of **outstanding loads**, not of bandwidth or instructions.
Each thread moves only `K / 32 = 64` bytes of `B`, and because `k` is a runtime
value the K loop does not unroll, so a thread has exactly one `B` load in flight
and pays full L1 latency every iteration.

Two hypotheses were tested and rejected:

- *Reduce conversion instructions.* Replacing the 16 scalar `static_cast<float>`
  FP8 converts with `__nv_cvt_fp8x2_to_halfraw2` (one `cvt.rn.f16x2.e4m3x2` per
  pair) is bit-exact and worth only 0-15%. Instruction count was not the limit.
- *Stage `A` in shared memory.* The concern was that all `N` warps re-read the
  whole `A` row. The 79.6% L1 hit rate shows L1 already absorbs this; a shared
  memory variant was neutral to slower except at very small `N`.

### 5.3 Change

`MatMulBlockScaledFp8GemvKernel` is now templated on
`<RowsPerWarp, ColsPerWarp, Unroll, AType>`, where `<R, 1, 1, A>` reproduces the
original geometry exactly:

- `Unroll` pre-issues `Unroll` independent `B`/`A` loads before consuming any of
  them, so several requests are in flight per thread.
- `ColsPerWarp` gives one warp several output columns, so each `A` load feeds
  several independent FMA chains.

Both trade occupancy - already register capped, and irrelevant at one wave - for
per-thread memory-level parallelism. The FP8 to FP16 conversion is also
vectorized. FP32 accumulation is unchanged, so results are **bit-identical** to
the previous kernel.

Dispatch (only `M == 1`, the batch-1 decode case, uses wide tiles):

| Condition | Config |
|---|---|
| `M == 1, N >= 8192` | `<1, 4, 2>` |
| `M == 1, N >= 4096` | `<1, 2, 2>` |
| otherwise | `<RowsPerWarp, 1, 1>` (unchanged) |

Below `N = 4096` the wider tiles leave too few warps to fill the GPU, and for
`M > 1` the extra live registers (accumulators plus pre-issued loads) cost more
than the added parallelism returns. Both measured slower, hence the guards.

> Superseded for `M > 1` by section 6.4, which re-tunes `ColsPerWarp` / `Unroll`
> for the speculative-decode tiles. Because `Unroll` changes the K chunk each
> lane accumulates first, the `M > 1` dispatch there is not bit-identical to
> `<RowsPerWarp, 1, 1>` (last-ulp only; the accumulation is still FP32).

### 5.4 Results (H200, `M = 1`, CUDA graph, us, includes 0.68 us node overhead)

| Shape (N x K) | cuBLAS FP16 | GEMV before | GEMV after | vs before | vs cuBLAS |
|---|---|---|---|---|---|
| 8192 x 2048 | 10.2 | 7.2 | **5.4** | 1.33x | 1.89x |
| 4096 x 2048 | 7.1 | 4.4 | **4.0** | 1.10x | 1.78x |
| 4096 x 4096 | 9.2 | 7.1 | **6.0** | 1.18x | 1.53x |
| 2048 x 4096 | 7.6 | 4.6 | 4.6 | 1.00x | 1.65x |
| 512 x 2048 | 5.4 | 2.7 | 2.7 | 1.00x | 2.00x |

At `8192 x 2048` this is 3.1 TB/s of the 4.8 TB/s HBM peak, up from 2.3 TB/s.

Note the last column: the weight-only FP8 GEMV is **1.5-2.0x faster than cuBLAS
FP16** at `M = 1`, so quantizing a projection to FP8 is a decode win on latency
as well as on footprint. At `M >= 4` cuBLAS wins and the GEMV path should not be
preferred on speed alone.

---

## 6. Decode GEMV - Tensor Cores

Section 5 tuned memory-level parallelism at `M = 1`. At `M = 4` - the width of a
speculative-decode / MTP verify forward - the kernel is limited by something
else. With `RowsPerWarp = 4` a lane executes roughly 240 instructions per 32
weight bytes, only 128 of which are the FMAs that do useful work, and effective
bandwidth falls from about 2.35 TB/s at `M = 1` to about 1.25 TB/s at `M = 4`.
More ILP cannot fix that; the dot products have to leave the FMA pipe.

### 6.1 Design

`MatMulBlockScaledFp8MmaGemvKernel` uses `mma.m16n8k16` with FP32 accumulation.
The operand assignment is the key decision:

| mma operand | fed from | why it fits |
|---|---|---|
| `A[16, 16]` row-major | weight `[16 output cols][16 k]` | `B` is `[N, K]` row-major |
| `B[16, 8]` col-major | activation `[16 k][8 rows]` | `A` is `[M, K]` row-major |
| `D[16, 8]` | `y[16 output cols][8 rows]` | |

So the mma "M" extent is the output column count and the mma "N" extent is `M`.
At `M = 4` half the mma N lanes are idle, which is irrelevant: the kernel is
bound by weight traffic and instruction issue, and both improve about 10x per
weight byte.

The naive fragment load is badly coalesced - a lane needs bytes
`{2t, 2t+1, 2t+8, 2t+9}` of a row, which spreads a warp across 16 rows x 16
bytes and over-fetches every 32-byte sector 2x. The fix is to **permute the K
axis**. K is a reduction axis, so any permutation applied to *both* operands
leaves the result unchanged. Inside a 64-element K window the permutation used is

```
mma k-slot (of step j)  ->  actual k
  2t,   2t+1                 16t + 4j,     16t + 4j + 1
  2t+8, 2t+9                 16t + 4j + 2, 16t + 4j + 3
```

so lane `(g = lane >> 2, t = lane & 3)` loads one contiguous `uint4` of weight
bytes `[16t, 16t + 16)` and the matching 32 activation bytes, four lanes cover 64
contiguous bytes of one weight row, and that single `uint4` feeds all four mma
steps.

16 columns per warp gives about 8x fewer warps than the FMA kernel, which alone
costs more in lost memory-level parallelism than the instruction saving is worth.
`KSplit` warps per block therefore take a strided share of the K windows and are
reduced through shared memory at the end. `KSplit = 8` for `N >= 8192` (the
column count already fills the grid) and 16 otherwise.

Preconditions: SM80+, `K % 64 == 0`, `K >= 256`, `block_size % 64 == 0`, `M <= 8`.
Otherwise the FMA kernel runs unchanged. `ORT_FP8_GEMV_MMA=0` forces the FMA
kernel for A/B testing in a single binary.

### 6.2 Accuracy

Not bit-identical to the FMA kernel (different summation order), but not less
accurate either. E4M3 to FP16 is lossless, E4M3 to BF16 is lossless, FP16 x FP16
products are exact in FP32, and the mma accumulates in FP32 exactly as the FMA
path does. Scored against an FP64 CPU reference on the shapes below, the maximum
error is *identical* for the two kernels (2-4e-4, i.e. pure FP16 output
rounding).

### 6.3 Results (H200, us, standalone, `fp8_gemv_m4_bench.cu`)

| Shape (N x K) | M | cuBLAS FP16 | FMA kernel | mma kernel | vs FMA | vs cuBLAS |
|---|---|---|---|---|---|---|
| 8192 x 2048 | 1 | 11.0 | 6.3 | **5.1** | 1.23x | 2.16x |
| 4096 x 2048 | 1 | 8.4 | 4.8 | **4.0** | 1.20x | 2.09x |
| 2048 x 4096 | 1 | 9.0 | 5.1 | **4.2** | 1.21x | 2.13x |
| 512 x 2048 | 1 | 6.7 | 3.3 | **3.1** | 1.06x | 2.12x |
| 8192 x 2048 | 4 | 11.0 | 9.8 | **5.2** | 1.87x | 2.10x |
| 4096 x 2048 | 4 | 8.5 | 6.9 | **4.1** | 1.69x | 2.09x |
| 2048 x 4096 | 4 | 8.5 | 8.0 | **4.4** | 1.82x | 1.92x |
| 512 x 2048 | 4 | 6.8 | 4.7 | **3.3** | 1.43x | 2.08x |
| 8192 x 2048 | 8 | 10.9 | 17.5 | **5.7** | 3.09x | 1.93x |
| 2048 x 4096 | 8 | 8.5 | 13.0 | **4.5** | 2.90x | 1.90x |

The mma kernel is faster at every measured `M`, so it is preferred whenever its
preconditions hold rather than only for `M > 1`. Note also that the FMA kernel
crosses over and loses to cuBLAS at `M = 8`, while the mma kernel stays about 1.9x
ahead.

End to end on a 40-layer Qwen3.6-35B-A3B NVFP4 MTP decode (130 FP8 matmul nodes
per step, `M = 4`), CUDA graphs on:

| | FMA kernel | mma kernel |
|---|---|---|
| FP8 GEMV kernel time | 1.052 ms/step | **0.713 ms/step** |
| total kernel time | 7.368 ms/step | **7.021 ms/step** |
| wall | 9.80 ms/step | **9.54 ms/step** |

No other kernel family moved. This optimization is kept.

### 6.4 FMA fallback re-tune for `M > 1`

The FMA kernel still runs when the tensor-core preconditions do not hold (pre-SM80,
`K < 256`, `K % 64 != 0` or `block_size % 64 != 0`), so the `M > 1` tiles were
re-tuned there as well. Widening `A` to FP32 is now hoisted out of the column loop
(one widening per row instead of one per row/column pair), which makes `ColsPerWarp`
profitable at `M > 1` for a second reason beyond memory-level parallelism:

| Condition | Config |
|---|---|
| `2 <= M <= 2, N >= 8192` | `<2, 4, 1>` |
| `2 <= M <= 2, N >= 2048` | `<2, 2, 1>` |
| `2 <= M <= 2` otherwise | `<2, 1, 2>` |
| `3 <= M <= 4, N >= 4096` | `<4, 4, 1>` |
| `3 <= M <= 4, N >= 2048` | `<4, 2, 2>` |
| `3 <= M <= 4` otherwise | `<4, 1, 2>` |
| `M > 4` | `<8, 1, 1>` |

Measured on H200 (us, `M = 4`, versus the previous `<R, 1, 1>` and cuBLAS FP16):

| Shape (N x K) | cuBLAS | `<4, 1, 1>` | tuned |
|---|---|---|---|
| 8192 x 2048 | 10.9 | 13.7 | **9.7** (`<4, 4, 1>`) |
| 4096 x 2048 | 8.3 | 8.1 | **6.9** (`<4, 4, 1>`) |
| 2048 x 4096 | 8.3 | 9.0 | **7.9** (`<4, 2, 2>`) |
| 512 x 2048 | 7.3 | 5.0 | **4.6** (`<4, 1, 2>`) |

The hoisting itself is bit-identical (the per-lane `fmaf` sequence is unchanged),
but a different `Unroll` changes which K chunk a lane accumulates first, so the
re-tuned dispatch is a last-ulp change relative to `<RowsPerWarp, 1, 1>`.

---

## 7. Benchmark Commands

The commands below use `ORT_REPO` and `ORT_BUILD` so they can be copied without
editing developer-specific paths. Set them once:

```bash
export ORT_REPO=$(git rev-parse --show-toplevel)
export ORT_BUILD="$ORT_REPO/build/cu130/Release"
```

Provider rebuild and Python-provider sync after editing the `.cu` kernel:

```bash
cmake --build "$ORT_BUILD" --target onnxruntime_providers_cuda --parallel
cp "$ORT_BUILD/libonnxruntime_providers_cuda.so" \
  "$ORT_BUILD/onnxruntime/capi/libonnxruntime_providers_cuda.so"
```

Decode GEMV benchmarks (small M):

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp8 --activation-dtype fp16 --m 1 --n 4096 --k 4096 --warmup 100 --repeat 500
```

Default prefill (dequantize + cuBLAS):

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp8 --activation-dtype fp16 --m 32 --n 4096 --k 4096 --warmup 50 --repeat 200
```

`N*K` scaling sweep used to isolate dequantization cost:

```bash
for shape in "4096 2048" "4096 4096" "4096 8192" "2048 4096" "8192 4096"; do
  set -- $shape
  cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
    python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
    --op fp8 --activation-dtype fp16 --m 16 --n $1 --k $2 --warmup 50 --repeat 300
done
```

Focused C++ tests:

```bash
CUDA_VISIBLE_DEVICES=0 "$ORT_BUILD/onnxruntime_provider_test" \
  --gtest_filter='MatMulBlockQuantizedFp8WeightOpTest.*'
```

---

## 8. Lessons

- The prefill path is memory bound on weight dequantization, not on the GEMM;
  latency there scales with `N*K` and is independent of `M`.
- Isolate a memory-bound helper by sweeping the dimension it depends on (`N*K`
  here) rather than the FLOP-bearing dimension (`M`). The flat latency across
  `M = 16..64` and the linear `N*K` scaling both pointed at dequantization.
- For a byte-to-halfword expansion, wide coalesced `uint4` loads/stores plus a
  2D grid that removes per-element integer division recovered about 1.55x and
  lifted effective bandwidth from about 28% to about 46% of peak.
- Keep the scalar dequant kernel as a correctness fallback for `K % 16 != 0`; the
  vectorized kernel requires the 16-element alignment that `K % 16 == 0`
  guarantees.
- Never benchmark a fast kernel through the ORT Python API without CUDA graphs.
  At `8192 x 2048, M = 1` that harness reported 21.0 us for a 7.2 us kernel; the
  measurement was host bound and led to the wrong conclusion that the decode
  GEMV was fine and that FP8 was slower than FP16.
- A decode GEMV runs one wave and is usually starved of *outstanding loads*, not
  of bandwidth or instructions. When SOL shows both DRAM and SM well under 60%
  with a large L1TEX stall share, add per-thread memory-level parallelism
  (unroll to pre-issue loads, widen the tile) rather than cutting instructions
  or adding shared memory staging. Trading register-capped occupancy for ILP is
  the right move at one wave.
- Do not pass an array by reference (`__half2 (&)[8]`) to a `__device__` helper.
  It is placed in local memory; inlining the same code via a macro was about 2x
  faster here and much more at `RowsPerWarp > 1`.
- Know the launch floor before optimizing: on H200 an empty kernel costs 0.68 us
  as a CUDA graph node. Ops cheaper than that are launch bound and should be
  fused, not tuned.
- The FP8 weight-only GEMV is 1.5-2.0x faster than cuBLAS FP16 at `M = 1`, so
  quantizing a projection is a decode latency win, not just a footprint win. The
  ordering reverses by `M = 4`.


## 9. Opt-in DeepGEMM W8A8 on SM90

Measured on NVIDIA H200, CUDA 13.0.48, driver 580.105.08, with the existing
DeepGEMM dependency introduced by PR #32122. Both configurations use the same
Release CUDA provider binary; only `ORT_FP8_MATMUL_DEEPGEMM` changes. Inputs use
`block_size=128`, a scalar activation scale, bias, and seed 123.

The profiler's `--cuda-graph` mode captures 16 warmed operator calls on a user
stream and measures each graph replay with CUDA events. The following numbers
are medians of 50 replays after 10 warmup calls, divided by 16. They include
activation quantization, scale preparation, output clearing, GEMM, conversion,
and bias. Model/session creation, host dispatch, and reference computation are
outside the timed interval. These are warm-cache, fixed-shape operator results,
not end-to-end model latency; the 80 MiB `16384 x 5120` weight also exceeds H200 L2.

| Activation | M | N | K | Flag off (us) | Flag on (us) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 64 | 4096 | 4096 | 27.79 | 17.33 | 1.60x |
| fp16 | 128 | 4096 | 4096 | 33.65 | 19.81 | 1.70x |
| fp16 | 64 | 11008 | 4096 | 68.08 | 31.74 | 2.15x |
| fp16 | 128 | 11008 | 4096 | 71.92 | 42.58 | 1.69x |
| fp16 | 128 | 16384 | 5120 | 122.85 | 63.85 | 1.92x |
| bf16 | 64 | 4096 | 4096 | 27.78 | 17.23 | 1.61x |
| bf16 | 128 | 4096 | 4096 | 33.37 | 19.77 | 1.69x |
| bf16 | 64 | 11008 | 4096 | 68.73 | 31.73 | 2.17x |
| bf16 | 128 | 11008 | 4096 | 71.96 | 42.72 | 1.68x |
| bf16 | 128 | 16384 | 5120 | 122.65 | 63.76 | 1.92x |

Decode cases M=4 and M=32 keep the existing GEMV and were within 1% between
flag settings. M=512 and M=2048 keep dequantization plus cuBLAS. Small weights
also keep that fallback: N=512 or 2048 with K=2048 showed helper overhead
outweighing savings when forced through the native kernel. The enabled range
is therefore limited to M<=128, N>=2048, at least 8 MiB of FP8 weights, and the
alignment and activation-scale requirements documented in
[the operator guide](matmul_block_scaled_fp8.md#61-opt-in-sm90-deepgemm-w8a8-path).
Larger M was deliberately excluded after measurements showed regressions for
some shapes, including M=512 and M=2048 with N=K=4096.

Reproduce an individual row from outside the source package directory:

```bash
cd /tmp
for enabled in 0 1; do
  PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 ORT_FP8_MATMUL_DEEPGEMM=$enabled \
    python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
    --op fp8 --m 128 --n 4096 --k 4096 --activation-dtype bf16 \
    --w8a8 --bias --cuda-graph --seed 123 --warmup 10 --repeat 50
done
```

### Kernel profiling

Nsight Systems 2026.1.3 with `--trace=cuda,nvtx --cuda-graph-trace=node` confirmed
69 native `sm90_fp8_gemm_1d1d_impl` launches for the M=128, N=K=4096 probe
(5 warmups plus four graph replays of 16 calls). It also showed the activation
quantization/scale-fill kernel, weight-scale packing, output clearing, and
conversion/bias. The separate activation-scale fill launch was fused into
quantization during tuning.

For BF16, average instrumented kernel durations were approximately 10.51 us
for DeepGEMM, 2.67 us for activation preparation, 2.34 us for scale packing,
and 3.29 us for conversion/bias. The flag-off trace instead showed weight
dequantization (14.47 us), cuBLAS GEMM (12.96 us), activation QDQ (3.25 us),
and bias (2.65 us). These instrumented durations explain where work is saved;
the uninstrumented complete-operator table is the performance comparison.

### Validation

- All 44 flag-off/on benchmark records passed the profiler's FP32-reference
  accuracy check for FP16 and BF16.
- Six focused integration tests passed on H200. They use an FP64 accumulation
  reference, dtype-aware rounding tolerances, independent row/K-block scales,
  changing activation and weight scales, zero scale, clipping, bias, partial M
  tiles, K=128, bounded output scratch, fallbacks, a nondefault stream, and ORT
  CUDA graph replay. A subprocess asserts the native dispatch log so a build
  without DeepGEMM cannot silently pass by using the fallback.
- Compute Sanitizer memcheck passed the tile/pipeline-boundary and output-scratch
  tests with zero errors.
- The CUDA provider and Python binding built successfully. The operator source
  also compiled with `USE_DEEP_GEMM` removed. Changed C++/CUDA files passed
  clang-format; changed Python files passed ruff.

Native FP8 applies scales to FP32 partial sums and has different intermediate
rounding from dequantizing operands to FP16/BF16. These tests establish operator
correctness within the stated tolerances; model-quality validation remains
necessary before deployment.
