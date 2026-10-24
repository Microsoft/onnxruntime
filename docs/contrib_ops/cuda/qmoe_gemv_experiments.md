# QMoE GEMV Profiling Experiments

## Small-decode fused routing and optional warp selection

The packed-INT and FP4 GEMV paths can fuse stable softmax/top-K routing with
expert-map construction for one or two rows, avoiding separate routing/map
launches. Unsupported configurations keep the existing routing and map builders.

`ORT_MOE_ROUTING_WARP_SELECT=1` additionally specializes normalized routing
with 512 experts and top-K=10. One warp per row holds 16 stable packed keys per
lane, selects ten maxima, and preserves the sequential normalization order.
A 128-thread block constructs the stable expert maps using the shared helper.
Other configurations retain the fused sort or generic fallback.

The warp-selection option defaults to `0` and accepts only `0` or `1`.
Set it before the first invocation. Tests cover FP32/FP16/BF16, ties,
infinities, expert-map stability, and CUDA graph replay.

This file records QMoE INT4/INT8 GEMV profiling results so future kernel and dispatch changes can be compared against a stable baseline.

> **Note**: These are **point-in-time** measurements captured on the specific GPU, driver, CUDA
> toolkit, and ORT build noted in each section header. Treat the numbers as a historical baseline
> for regression comparison, not as current performance guidance — re-run the benchmark script on
> your own hardware before drawing conclusions.

## 2026-06-12 Baseline: SM90, Warmup 5, Repeat 100

### Setup

- Machine/GPU: local CUDA machine, SM90 reported by `torch.cuda.get_device_capability()`.
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`.
- Python: `~/onnxruntime/.venv/bin/python`.
- Nsight Systems: `~/cuda13.0/bin/nsys`.
- Benchmark script: `onnxruntime/test/python/transformers/profile_qmoe_gemv.sh`.
- Warmup: 5 ORT runs before the measured `benchmark` NVTX range.
- Repeat: 100 measured ORT runs inside the `benchmark` NVTX range.
- Common shape settings: hidden size 128, intermediate size 256, 4 experts, top-k 2, INT4 symmetric per-channel weights, SwiGLU interleaved activation.
- Profile log: `/tmp/qmoe_gemv_warmup5_r100.log`.
- Nsight artifacts: `/tmp/qmoe_gemv_<case>_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

Command template:

```bash
pushd /tmp >/dev/null
PATH=~/cuda13.0/bin:$PATH \
PYTHONPATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/onnxruntime/test/python/transformers \
~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --python ~/onnxruntime/.venv/bin/python \
  --case <case-name> --warmup 5 --repeat 100 \
  -o /tmp/qmoe_gemv_<case-name>_warmup5_r100
popd >/dev/null
```

Parse command used for kernel-level comparisons:

```bash
python onnxruntime/test/python/transformers/parse_nsys.py \
  /tmp/qmoe_gemv_<case>_warmup5_r100_<mode>.sqlite \
  --nvtx-range benchmark --pattern '%'
```

`--pattern '%'` is important for fallback runs because the grouped-GEMM compute kernels are named as CUTLASS kernels and are not shown by the default parser pattern.

### End-To-End ORT Loop Timing

Lower is better. `GEMV/GEMM` compares the host-observed average latency printed by `profile_qmoe_gemv.py`.

| Case | Tokens | Expanded rows | DType | GEMV ms | GEMM fallback ms | GEMV/GEMM | Result |
|------|--------|---------------|-------|---------|------------------|-----------|--------|
| `m1_top2_fp16_128x256` | 1 | 2 | FP16 | 0.0967 | 0.0706 | 1.37x | GEMV slower |
| `m4_top2_fp16_128x256` | 4 | 8 | FP16 | 0.1372 | 0.0705 | 1.94x | GEMV slower |
| `m8_top2_fp16_128x256` | 8 | 16 | FP16 | 0.1381 | 0.0702 | 1.97x | GEMV slower |
| `m1_top2_bf16_128x256` | 1 | 2 | BF16 | 0.0972 | 0.0706 | 1.38x | GEMV slower |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX range. GEMV has two compute kernels per ORT run in these cases, corresponding to the two MoE FC stages. GEMM fallback has two grouped-GEMM kernels per ORT run, except `m8_top2_fp16_128x256` appears as one aggregated CUTLASS kernel row with 200 calls.

| Case | GEMV compute avg us | GEMM compute avg us | Notes |
|------|---------------------|---------------------|-------|
| `m1_top2_fp16_128x256` | 2.82, 2.56 | 3.98, 3.44 | GEMV compute kernels are faster, but total ORT loop is slower. |
| `m4_top2_fp16_128x256` | 3.01, 2.74 | 3.72, 3.42 | GEMV compute kernels are faster, but total ORT loop is slower. |
| `m8_top2_fp16_128x256` | 3.09, 2.75 | 3.55 aggregated over 200 calls | GEMV compute kernels are faster per call, but total ORT loop is slower. |
| `m1_top2_bf16_128x256` | 2.84, 2.61 | 4.22, 3.49 | GEMV compute kernels are faster, but total ORT loop is slower. |

### Observations

- With explicit warmup and `nsys` NVTX filtering, the host-observed ORT loop latency is not aligned with the earlier low-repeat smoke numbers that suggested a GEMV win. This warmed baseline shows grouped GEMM fallback faster for these small 128x256 cases.
- Kernel-only compute timing still shows the custom GEMV compute kernels faster than grouped GEMM compute kernels. The end-to-end loss likely comes from non-compute overhead around the GEMV path, dispatch/prologue behavior, extra launches, or measurement sensitivity at very small latencies.
- The default `parse_nsys.py` kernel filter can hide fallback CUTLASS grouped-GEMM kernels. Use `--pattern '%'` for GEMV-vs-GEMM profile comparisons.
- This baseline strengthens the P1/P2 work items: avoid repeated row-to-expert scans, tune tile/threshold decisions with data, and consider fusing FC1 activation or FC2 finalize/scatter before expanding GEMV coverage.

### Next Experiments

- Sweep larger realistic FC dimensions, especially model-like hidden/intermediate sizes, because this baseline only covers 128x256 test-sized shapes.
- Sweep expanded rows `{1, 2, 4, 8, 16, 32, 64}` while keeping shape and dtype fixed.
- Capture CUDA API timing inside the `benchmark` range to identify host-side synchronization or launch overhead differences.
- Record per-architecture results for SM80/SM89/SM90/SM100/SM120 when available.

## 2026-06-12 Larger Shape Sweep: SM90, 1024x4096, Warmup 5, Repeat 100

### Setup

- Same environment as the baseline section.
- Custom profiler arguments were used instead of named cases.
- Common shape settings: hidden size 1024, intermediate size 4096, 8 experts, top-k 2, INT4 symmetric per-channel weights, SwiGLU interleaved activation.
- Profile log: `/tmp/qmoe_gemv_1024x4096_e8_warmup5_r100.log`.
- Nsight artifacts: `/tmp/qmoe_gemv_custom_<case>_1024x4096_e8_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

Command template:

```bash
pushd /tmp >/dev/null
PATH=~/cuda13.0/bin:$PATH \
PYTHONPATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/onnxruntime/test/python/transformers \
~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --python ~/onnxruntime/.venv/bin/python \
  --batch-size 1 --sequence-length <m> \
  --hidden-size 1024 --intermediate-size 4096 \
  --num-experts 8 --top-k 2 --dtype FLOAT16 \
  --warmup 5 --repeat 100 \
  -o /tmp/qmoe_gemv_custom_m<m>_top2_float16_1024x4096_e8_warmup5_r100
popd >/dev/null
```

### End-To-End ORT Loop Timing

Lower is better. `GEMV/GEMM` compares the host-observed average latency printed by `profile_qmoe_gemv.py`.

| Case | Tokens | Expanded rows | DType | GEMV ms | GEMM fallback ms | GEMV/GEMM | Result |
|------|--------|---------------|-------|---------|------------------|-----------|--------|
| `custom_m1_top2_float16_1024x4096_e8` | 1 | 2 | FP16 | 0.0622 | 0.0811 | 0.77x | GEMV faster |
| `custom_m4_top2_float16_1024x4096_e8` | 4 | 8 | FP16 | 0.1542 | 0.0873 | 1.77x | GEMV slower |
| `custom_m8_top2_float16_1024x4096_e8` | 8 | 16 | FP16 | 0.2083 | 0.0976 | 2.13x | GEMV slower |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 1 | 2 | BF16 | 0.2144 | 0.0844 | 2.54x | GEMV slower |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX range.

| Case | GEMV compute avg us | GEMM compute avg us | Notes |
|------|---------------------|---------------------|-------|
| `custom_m1_top2_float16_1024x4096_e8` | 6.78, 4.70 | 14.61, 8.55 | GEMV wins both compute and end-to-end. |
| `custom_m4_top2_float16_1024x4096_e8` | 12.05, 10.00 | 16.56, 12.71 | GEMV compute is faster, but total ORT loop is slower. |
| `custom_m8_top2_float16_1024x4096_e8` | 23.19, 13.50 | 26.34, 14.29 | GEMV compute advantage narrows and end-to-end latency is worse. |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 6.99, 4.69 | 12.86 aggregated over 200 calls | BF16 GEMV is much slower end-to-end despite comparable compute-kernel timing. |

### Observations

- FP16 GEMV shows a real end-to-end win for single-token decode at 1024x4096, unlike the tiny 128x256 test shape.
- The previous broad `expanded_num_rows <= 64` dispatch threshold was too optimistic on SM90 for this shape. At expanded rows 8 and 16, grouped GEMM fallback is faster end-to-end.
- The initial P1 dispatch cutoff is therefore conservative: FP16 only, `expanded_num_rows <= 2`, and `N/K >= 1024`. Collect more model-size points around expanded rows 2, 4, 8, and 16 before enabling GEMV beyond true single-token decode.

## 2026-06-12 Post-Threshold Dispatch Smoke: SM90, Warmup 1, Repeat 2

### Setup

- Same build and Python environment as above, after syncing the rebuilt CUDA provider into the Python package `capi` directory.
- Nsight Systems profiles used the measured `benchmark` NVTX range and counted kernels matching `%moe_gemv%`.
- These runs validate routing only; repeat count is too small for performance comparison.

| Case | Expanded rows | DType | Expected route | `moe_gemv` calls in benchmark | Result |
|------|---------------|-------|----------------|--------------------------------|--------|
| `m1_top2_fp16_128x256` | 2 | FP16 | grouped GEMM fallback (`N/K < 1024`) | 0 | Passed |
| `custom_m1_top2_float16_1024x4096_e8` | 2 | FP16 | GEMV | 4 | Passed |
| `custom_m4_top2_float16_1024x4096_e8` | 8 | FP16 | grouped GEMM fallback (`expanded_num_rows > 2`) | 0 | Passed |
| `custom_m1_top2_bfloat16_1024x4096_e8` | 2 | BF16 | grouped GEMM fallback (FP16-only GEMV gate) | 0 | Passed |

## 2026-06-12 Actual Model Dimensions: SM90, FP16, Warmup 5, Repeat 20

### Setup

- Same build and Python environment as above.
- Nsight Systems profiles used the measured `benchmark` NVTX range.
- Enabled mode leaves `ORT_DISABLE_MOE_GEMV` unset. Fallback mode sets `ORT_DISABLE_MOE_GEMV=1`.
- Nsight artifacts: `/tmp/qmoe_actual_<case>_warmup5_r20_{enabled,gemm}.{nsys-rep,sqlite}`.
- Summary log: `/tmp/qmoe_actual_model_dims_fp16_warmup5_r20_summary.txt`.

Model dimensions came from the Hugging Face configs below. Qwen's shared expert
is ignored in this benchmark.

| Model case | Source dimensions | Tokens | Expanded rows | Enabled ms | Fallback ms | `moe_gemv` calls in enabled profile | Route |
|------------|-------------------|--------|---------------|------------|-------------|--------------------------------------|-------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | `hidden_size=2880`, `intermediate_size=2880`, `num_local_experts=32`, `top_k=4` | 1 | 4 | 0.0909 | 0.0944 | 0 | grouped GEMM fallback |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | `hidden_size=2048`, `moe_intermediate_size=512`, `num_experts=256`, `top_k=8` | 1 | 8 | 0.0744 | 0.0743 | 0 | grouped GEMM fallback |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | `hidden_size=2816`, `moe_intermediate_size=704`, `num_experts=128`, `top_k=8` | 1 | 8 | 0.0819 | 0.0820 | 0 | grouped GEMM fallback |

### Observations

- The actual single-token top-k values for these models produce expanded rows 4
  or 8, so they intentionally stay on grouped GEMM under the current
  `expanded_num_rows <= 2` GEMV gate.
- Enabled and forced-fallback timings are nearly identical, and the enabled
  profiles contain zero custom `moe_gemv` kernels, confirming the dispatch route.
- These profiles establish the current grouped-GEMM baseline for actual model
  dimensions. Future GEMV work for model-realistic top-k needs targeted row-count
  improvements before expanding the dispatch gate.

## 2026-06-12 Actual Model Dimensions With Relaxed GEMV Gate: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Dispatch was temporarily relaxed to `expanded_num_rows <= 8` and `N/K >= 512`
  to force GEMV coverage for GPT-OSS-20B, Qwen3.6-35B-A3B, and Gemma-4-26B-A4B
  FP16 model-size cases.
- Nsight artifacts: `/tmp/qmoe_actual_<case>_relaxed_warmup5_r100_{enabled,gemm}.{nsys-rep,sqlite}`.
- Summary log: `/tmp/qmoe_actual_model_dims_fp16_relaxed_warmup5_r100_summary.txt`.

| Model case | Expanded rows | Enabled route | Enabled ms | Fallback ms | GEMV/GEMM | GEMV calls | Primary enabled compute avg us | Primary fallback compute avg us | Decision |
|------------|---------------|---------------|------------|-------------|-----------|------------|--------------------------------|---------------------------------|----------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 4 | GEMV | 0.0760 | 0.0905 | 0.84x | 200 | 15.35, 11.94 | 20.11, 18.48 | Keep GEMV candidate |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | 8 | GEMV | 0.0865 | 0.0707 | 1.22x | 200 | 20.37, 18.77 | 10.33 aggregated over 200 calls | Reject current GEMV gate |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8 | GEMV | 0.0745 | 0.0801 | 0.93x | 200 | 14.61, 11.85 | 13.58 aggregated over 200 calls | Keep GEMV candidate |

### Observations

- GPT-OSS-20B benefits from GEMV at expanded rows 4 with square 2880x2880 FC
  dimensions.
- Gemma-4-26B-A4B shows a smaller but positive end-to-end win at expanded rows 8
  with 2816x704 / 704x2816 FC dimensions.
- Qwen3.6-35B-A3B regresses with the current GEMV kernel at expanded rows 8 and
  2048x512 / 512x2048 FC dimensions. The primary GEMV kernels are slower than
  grouped GEMM for this 512-wide MoE hidden size.
- Based on this pass, the dispatch gate should keep `expanded_num_rows <= 8` and
  `N/K >= 512`, but require the logical MoE intermediate size and each GEMV call
  dimension to be at least 704 when `expanded_num_rows > 4`. The logical
  intermediate-size check is needed because Qwen's gated FC1 has `N=1024` even
  though the MoE hidden size is 512. This keeps GPT-OSS and Gemma enabled while
  routing the measured Qwen regression to grouped GEMM. A future autotuner could
  replace this hand cutoff.

### Final Route Check After Logical Intermediate-Size Guard

After adding the logical intermediate-size guard, short Nsight profiles with
warmup 1 and repeat 2 confirmed the intended final routing:

| Model case | Expected route | `moe_gemv` calls in benchmark | Result |
|------------|----------------|--------------------------------|--------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | GEMV | 4 | Passed |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | grouped GEMM fallback | 0 | Passed |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | GEMV | 4 | Passed |

## 2026-06-13 P1 Row-To-Expert Map: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Change under test: the prologue now writes the local expert id for each
  permuted row into `permuted_token_selected_experts`, and the INT4
  per-channel MoE GEMV kernel uses that direct row-to-expert map instead of
  scanning `expert_first_token_offset` in every N-tile CTA. The prefix-offset
  scan remains as a fallback when no map is passed.
- Nsight artifacts:
  `/tmp/qmoe_actual_gpt_oss_20b_p1_row_expert_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`
  and
  `/tmp/qmoe_actual_gemma4_26b_p1_row_expert_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.
- Summary logs:
  `/tmp/qmoe_actual_gpt_oss_20b_p1_row_expert_warmup5_r100.log` and
  `/tmp/qmoe_actual_gemma4_26b_p1_row_expert_warmup5_r100.log`.

### End-To-End ORT Loop Timing

| Model case | Expanded rows | Enabled route | Enabled ms | Fallback ms | GEMV/GEMM | GEMV calls | Result |
|------------|---------------|---------------|------------|-------------|-----------|------------|--------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 4 | GEMV | 0.0721 | 0.0941 | 0.77x | 200 | GEMV faster |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8 | GEMV | 0.0610 | 0.0795 | 0.77x | 200 | GEMV faster |

### Primary Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX
range. The two GEMV compute rows correspond to FC1 and FC2.

| Model case | GEMV compute avg us | Fallback compute avg us | Previous GEMV compute avg us | Notes |
|------------|---------------------|-------------------------|------------------------------|-------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 13.69, 10.22 | 20.32, 18.55 | 15.35, 11.94 | Direct map reduces GEMV kernel time by about 11% and 14% versus the relaxed-gate baseline. |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 7.17, 4.56 | 18.82, 7.99 | 14.61, 11.85 | Direct map removes a large per-tile scan cost at top-k 8 for this 704-wide case. |

### Observations

- The P1 row-to-expert map improves both actual-model GEMV candidates without
  changing the dispatch policy.
- The improvement is largest for Gemma, where expanded rows 8 and many N tiles
  made the repeated prefix scan particularly visible.
- Qwen remains routed to grouped GEMM by the logical-intermediate-size guard. A
  quick enabled-mode smoke run reported valid output for
  `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256`.

## 2026-06-12 P1 Tile-Shape Probe: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Goal: test the next P1 tile-shape knobs against the actual model-size cases
  that route to GEMV after the row-to-expert map optimization.
- Variants tested:
  - `CtaN=16, Threads=128`: halves the number of N-tile CTAs.
  - `CtaN=8, Threads=64`: reduces threads per CTA while preserving the N tile.
  - `CtaN=8, Threads=128` with a map-specialized launch: splits direct-map and
    prefix-scan kernels so the common direct-map path avoids the runtime branch.
- Nsight artifacts:
  `/tmp/qmoe_actual_*_ctan16_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`,
  `/tmp/qmoe_actual_*_threads64_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`,
  and `/tmp/qmoe_actual_*_specialized_map_warmup5_r100_{gemv,gemm}.{nsys-rep,sqlite}`.

### End-To-End ORT Loop Timing

| Variant | Model case | Enabled ms | Fallback ms | Result |
|---------|------------|------------|-------------|--------|
| `CtaN=16, Threads=128` | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0852 | 0.0872 | Reject |
| `CtaN=16, Threads=128` | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0644 | 0.0791 | Reject |
| `CtaN=8, Threads=64` | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0832 | 0.0993 | Reject |
| `CtaN=8, Threads=64` | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0645 | 0.0799 | Reject |
| map-specialized launch | `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 0.0726 | 0.0909 | Reject, neutral/slightly worse kernels |
| map-specialized launch | `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 0.0600 | 0.0790 | Reject, neutral/slightly worse kernels |

### Primary GEMV Compute Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX
range. Previous best is the row-to-expert map build with `CtaN=8, Threads=128`:
GPT `13.69, 10.22` us and Gemma `7.17, 4.56` us.

| Variant | GPT GEMV avg us | Gemma GEMV avg us | Decision |
|---------|-----------------|-------------------|----------|
| `CtaN=16, Threads=128` | 19.48, 17.52 | 11.07, 5.65 | Wider N tile is slower for both models. |
| `CtaN=8, Threads=64` | 18.77, 16.78 | 11.09, 5.09 | Fewer threads are slower for both models. |
| map-specialized launch | 14.00, 10.10 | 7.25, 4.62 | No clear kernel win; keep the simpler launch. |

### Observations

- The existing `CtaN=8, Threads=128` tile remains the best measured choice for
  the current SM90 actual-model cases.
- Halving N tiles with `CtaN=16` reduced CTA count but lost too much per-CTA
  efficiency.
- `Threads=64` did not help the 704-wide Gemma case and significantly hurt the
  larger GPT-OSS case.
- The direct-map branch in the current kernel is not a visible bottleneck after
  the row-to-expert map optimization, so the extra specialized launch variants
  are not worth the additional code size.

## 2026-06-13 FC1 Interleaved SwiGLU Fusion: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Change under test: FC1 INT4 per-channel GEMV has an interleaved SwiGLU epilogue
  for the profiled FP16 path. It computes adjacent gate/linear FC1 columns,
  applies the existing `alpha=1.702`, `beta=1.0`, `limit=7.0` SwiGLU formula,
  and writes post-activation `[expanded_rows, inter_size]` directly to the FC2
  input buffer.
- The unfused GEMV plus `doGatedActivationKernel` path remains the fallback for
  non-FP16, non-INT4-per-channel, groupwise quantization, non-interleaved
  activations, and shapes rejected by the existing GEMV dispatch policy.
- Nsight artifacts:
  `/tmp/ort_qmoe_profile/qmoe_swiglu_fused_gpt_{gemv,gemm}.{nsys-rep,sqlite}`,
  `/tmp/ort_qmoe_profile/qmoe_swiglu_fused_gemma_{gemv,gemm}.{nsys-rep,sqlite}`,
  and `/tmp/ort_qmoe_profile/qmoe_swiglu_fused_qwen_{gemv,gemm}.{nsys-rep,sqlite}`.

### Correctness

- Focused QMoE GEMV smoke:
  `TestQMoEGemvBenchmark::test_decode_latency` passed.
- Large enough FP16 INT4 SwiGLU parity case to exercise the fused path
  (`hidden_size=1024`, `intermediate_size=512`, `num_experts=4`, `top_k=2`) had
  max absolute difference `0.0009766` before the helper's parity check and
  `0.000893` in the built-in parity check.

### End-To-End ORT Loop Timing

Lower is better. Previous best is the P1 row-to-expert map build with the same
`CtaN=8, Threads=128` tile shape.

| Model case | Expanded rows | Enabled route | Fused GEMV ms | Previous best GEMV ms | Fallback ms | Result |
|------------|---------------|---------------|---------------|-----------------------|-------------|--------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 4 | GEMV | 0.0681 | 0.0721 | 0.1013 | GEMV faster; fusion improves enabled path by about 5.5%. |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8 | GEMV | 0.0580 | 0.0610 | 0.0811 | GEMV faster; fusion improves enabled path by about 4.9%. |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | 8 | grouped GEMM fallback | 0.0706 | N/A | 0.0710 | No custom GEMV kernels; route unchanged by fusion. |

### Primary Kernel Timing

Values are average kernel duration in microseconds inside the measured NVTX
range. The fused GEMV compute rows correspond to fused FC1 and unfused FC2.

| Model case | Fused GEMV compute avg us | Previous best GEMV compute avg us | Removed activation avg us from fallback profile | Notes |
|------------|---------------------------|-----------------------------------|-----------------------------------------------|-------|
| `gpt_oss_20b_m1_top4_fp16_2880x2880_e32` | 13.93, 10.11 | 13.69, 10.22 | 3.23 | FC1 GEMV is slightly slower due to the fused epilogue, but removing the activation launch gives a net end-to-end win. |
| `gemma4_26b_a4b_m1_top8_fp16_2816x704_e128` | 8.30, 4.57 | 7.17, 4.56 | 1.82 | The fused epilogue costs more for the small FC1 tile, but still beats the separate activation launch end-to-end. |
| `qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256` | N/A | N/A | 1.76 | The 512-wide intermediate-size guard keeps Qwen on grouped GEMM. |

### Observations

- FC1 interleaved SwiGLU fusion is a modest but real win for the two actual-model
  cases that already route to GEMV.
- The win comes from launch and memory-traffic removal, not from faster FC1
  compute. The fused FC1 kernel is slightly slower than the prior FC1 GEMV
  kernel, so this optimization should stay narrowly gated to the measured path.
- The FC2 finalize/scatter fusion remains a larger but riskier opportunity. It
  can remove another launch, but it needs a design that preserves the current
  float accumulation behavior across top-k experts.

## 2026-06-13 GPT-OSS FC2 / Finalize Follow-Up: SM90, FP16, Warmup 5, Repeat 100

### Setup

- Same build and Python environment as above.
- Target case: `gpt_oss_20b_m1_top4_fp16_2880x2880_e32`.
- Goal: find remaining GPT-OSS-specific opportunity after FC1 interleaved SwiGLU
  fusion. The prior profile had FC1 GEMV around `13.93` us, FC2 GEMV around
  `10.11` us, and finalize around `5.65` us.
- Nsight artifacts:
  `/tmp/ort_qmoe_profile/qmoe_fc2_finalize_fused_gpt_{gemv,gemm}.{nsys-rep,sqlite}`
  for the rejected fused-FC2 prototype and
  `/tmp/ort_qmoe_profile/qmoe_finalize_onerow_gpt_{gemv,gemm}.{nsys-rep,sqlite}`
  for the retained one-row finalize specialization.

### Rejected Prototype: FC2 GEMV + Finalize Fusion

The prototype assigned one CTA to each original token and N tile, then looped
over the token's top-k experts inside that CTA. It avoided atomics and preserved
the top-k accumulation order, but it also serialized the four GPT-OSS FC2 GEMVs
that previously ran as separate expanded-row CTAs.

| Variant | Enabled ms | Fused FC2/finalize avg us | Decision |
|---------|------------|---------------------------|----------|
| FC2 GEMV + finalize fusion | 0.0848 | 33.21 | Reject; removed from code. |

### Retained Prototype: One-Row Finalize Specialization

The retained change keeps the existing FC2 GEMV kernel and only specializes
`finalizeMoeRoutingKernelLauncher` for `num_rows == 1` and `top_k <= 4`. The
specialized kernel caches `unpermuted_row_to_permuted_row`, expert ids, and
routing scales once in shared memory instead of reloading them for every output
vector element.

| Variant | Enabled ms | FC1 GEMV avg us | FC2 GEMV avg us | Finalize avg us | Result |
|---------|------------|-----------------|-----------------|-----------------|--------|
| FC1 fused baseline | 0.0681 | 13.93 | 10.11 | ~5.65 | Baseline for comparison. |
| one-row finalize specialization | 0.0681 | 13.98 | 10.09 | 5.50 | Kept as first step; small finalize-kernel win, end-to-end neutral within noise. |
| static top-k one-row specialization | 0.0671 | 13.93 | 10.01 | 5.00 | Keep; top-k 4 compile-time unroll improves finalize and gives the best GPT-OSS enabled latency so far. |
| static top-k with 128 threads | 0.0684 | 13.82 | 10.24 | 6.48 | Reject; smaller block underutilizes the 2880-wide output. |

### Correctness

- Large enough FP16 INT4 SwiGLU top-k 4 parity case
  (`hidden_size=1024`, `intermediate_size=1024`, `num_experts=8`, `top_k=4`) had
  max absolute difference `0.0006104` before the helper's parity check and
  `0.000732` in the built-in parity check.

### Observations

- GPT-OSS still has room around FC2 and finalize, but a simple single-CTA
  FC2/finalize fusion is the wrong shape because it gives up expanded-row
  parallelism.
- A viable future FC2/finalize design likely needs to preserve parallelism across
  top-k experts and N tiles, then reduce partial results without atomics or with
  carefully bounded numerical change.
- The current one-row finalize specialization is intentionally small: it is useful
  for GPT-style single-token decode and does not alter routing for larger batches
  or top-k 8 models. Dispatching exact top-k specializations is worthwhile for
  GPT-OSS top-k 4; reducing the one-row block from 256 to 128 threads is not.

## 2026-06-13 GenAI End-To-End Throughput: GPT-OSS-20B INT4, SM90 (H200), Batch 1

### Setup

- Measures full ONNX Runtime GenAI token-generation throughput, not isolated
  kernel time, so it captures the real end-to-end impact of the MoE GEMV path.
- GPU: single H200 (SM90), `CUDA_VISIBLE_DEVICES=1` on an otherwise idle GPU.
  GPU 0 was busy with another job during early runs and produced corrupted
  numbers, so all results below use an idle GPU.
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release` (branch
  `tlw/rel-1.27.0_qmoe_update`), CUDA 13.0, `CMAKE_CUDA_ARCHITECTURES=89;90`.
- ONNX Runtime GenAI: `0.14.0-dev`, venv `~/.venv_src_8f0278c` (Python 3.14).
- Model: `gpt-oss-20b` INT4 per-channel,
  `~/models/gpt-oss-20b/cuda/cuda-int4-kquant-block-32-mixed/`
  (`hidden=inter=2880`, 32 experts, top-k 4, SwiGLU interleaved). Single-token
  decode expands to 4 rows per step, which routes to the INT4 per-channel GEMV.
- Three configurations compared:
  - **Cutlass baseline (grouped GEMM)**: GEMV disabled via
    `ORT_DISABLE_MOE_GEMV=1`, so FC1/FC2 use the cutlass grouped-GEMM path.
  - **GEMV (final)**: default build with the INT4 per-channel MoE GEMV fast path,
    including the row-to-expert map, FC1 interleaved SwiGLU fusion, and static
    top-k one-row finalize specialization.
  - **FT baseline**: FasterTransformer MoE kernel used in ORT 1.26 (or ORT GenAI 0.14.1) reference token-generation throughput for the same model and prompt lengths.
- Correctness: both GEMV-enabled and `ORT_DISABLE_MOE_GEMV=1` produce identical
  correct output ("Paris is the capital of France.") for the sanity prompt, and
  the Nsight trace confirms `moe_gemv_kernel` executes for FC1 and FC2.

Command template (run once per configuration):

```bash
cd ~/onnxruntime-genai/benchmark/python
source ~/.venv_src_8f0278c/bin/activate
export LD_LIBRARY_PATH=~/cuda13.0/lib64:~/cudnn9.19_cuda13/lib:$LD_LIBRARY_PATH
# Add ORT_DISABLE_MOE_GEMV=1 for the cutlass baseline.
CUDA_VISIBLE_DEVICES=1 python benchmark_e2e.py \
  -i ~/models/gpt-oss-20b/cuda/cuda-int4-kquant-block-32-mixed/ \
  -b 1 -l 128,1024,2048 -g 256 -r 10 -w 2 \
  --use_random_tokens --chat_template '{input}' \
  -pm 1 -e cuda -mn gpt-oss-20b -pr int4 -o results_gptoss/<name>.csv
```

### Token-Generation Throughput

Higher is better. Values are average token-generation throughput in tokens per
second (tps) for batch size 1, 256 generated tokens, 10 repeats, 2 warmups.

| Prompt length | Cutlass baseline (gemm) tps | GEMV (final) tps | FT baseline tps | GEMV vs cutlass | GEMV vs FT |
|---------------|-----------------------------|------------------|-----------------|-----------------|------------|
| 128 | 248.9 | 288.0 | 265.2 | +15.7% | +8.6% |
| 1024 | 237.8 | 272.2 | 252.9 | +14.5% | +7.6% |
| 2048 | 231.3 | 265.0 | 245.6 | +14.6% | +7.9% |

### Observations

- The final MoE GEMV path beats both the cutlass grouped-GEMM baseline and the
  FasterTransformer 0.14.1 reference at every measured prompt length.
- Versus the cutlass grouped-GEMM baseline, GEMV improves end-to-end
  token-generation throughput by roughly 15% across prompt lengths.
- Versus the FT 0.14.1 baseline, GEMV is about 8% faster, meeting the goal of
  outperforming FasterTransformer for GPT-OSS-20B single-token decode.
- These end-to-end gains are consistent with the kernel-level wins recorded in
  the row-to-expert map, FC1 interleaved SwiGLU fusion, and static top-k finalize
  sections above.
- Always confirm the benchmark GPU is idle (`nvidia-smi`) before recording
  numbers; contention on a shared GPU inflates sampling latency and corrupts the
  throughput measurement.

## 2026-06-13 Block-Wise INT4/INT8 GEMV: SM90, `block_size=64`, Warmup 5, Repeat 100

### Setup

- Goal: extend the CUDA QMoE GEMV path from INT4 per-column quantization to
  symmetric block-wise integer quantization.
- GPU: H200, SM90.
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`.
- Python: `~/onnxruntime/.venv/bin/python`.
- Nsight Systems: `~/cuda13.0/bin/nsys`.
- All model-shape runs used `--warmup 5 --repeat 100 --block-size 64` and parsed
  the `benchmark` NVTX range.
- Kernel parsing used `parse_nsys.py --pattern '%'` so the CUTLASS fallback
  kernels and custom GEMV kernels both appear.

Implementation summary:

- The GEMV kernels now support symmetric INT4 and INT8 block-wise scales for
  group sizes 64 and 128.
- Block-wise scale inputs are `[E, N, K_blocks]`; QMoE prepack/runtime transposes
  them to `[E, K_blocks, N]`, and GEMV consumes that same layout.
- Block-wise GEMV is symmetric only. When zero-point compensation is present,
  dispatch rejects GEMV and falls back to grouped GEMM.
- The per-column INT4 path is unchanged and still uses `GroupSize == 0`.

Command template:

```bash
cd /tmp
source ~/onnxruntime/.venv/bin/activate
export CUDA_HOME=~/cuda13.0
export CUDNN_HOME=~/cudnn9.19_cuda13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/onnxruntime/test/python/transformers

~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
   --python ~/onnxruntime/.venv/bin/python \
   --case <case-name> \
   --block-size 64 --warmup 5 --repeat 100 \
   -o /tmp/qmoe_profile_20260613/<output-name>
```

### Correctness and Smoke Coverage

- Provider build passed after threading `MOEParallelismConfig` into the static
  `gemm1` helper.
- Python syntax passed for `test_qmoe_cuda.py` and `profile_qmoe_gemv.py`.
- Focused parity passed: `python -m pytest .../test_qmoe_cuda.py -k "blockwise"`
  reported `40 passed, 35 deselected`.
- Short benchmark smokes on SM90 produced finite output for INT4/INT8,
  `block_size` 64/128, with `expanded_num_rows=2`.

| Case | Quant Bits | Block Size | Latency (ms) | Status |
|------|------------|------------|--------------|--------|
| `blockwise_int4_b64_m1_top2_fp16_1024x4096_e8` | 4 | 64 | 0.074851 | passed |
| `blockwise_int4_b128_m1_top2_fp16_1024x4096_e8` | 4 | 128 | 0.129937 | passed |
| `blockwise_int8_b64_m1_top2_fp16_1024x4096_e8` | 8 | 64 | 0.078567 | passed |
| `blockwise_int8_b128_m1_top2_fp16_1024x4096_e8` | 8 | 128 | 0.081153 | passed |

### Route Investigation

- The symmetric INT4 block-wise ONNX graph has empty zero-point inputs 11/12 and
  no zero-point initializers.
- Temporary diagnostics showed early profiler/tactic invocations could reject
  GEMV, but the measured benchmark loop had null zero-point pointers and launched
  the custom kernels.
- Temporary diagnostics were removed before final profiling; the provider no
  longer contains `MOE_GEMV_DEBUG`, `QMOE_PREPACK_DEBUG`, or `QMOE_COMPUTE_DEBUG`
  strings.
- Route verification must come from Nsight kernels inside the `benchmark` NVTX
  range, not from the benchmark JSON alone.

### GPT-OSS-20B Shape, INT4, `block_size=64`

Shape: `M=1`, `top_k=4`, `expanded_num_rows=4`, `hidden_size=2880`,
`intermediate_size=2880`, `num_experts=32`, FP16.

Artifacts:

- GEMV enabled: `/tmp/qmoe_profile_20260613/gpt_oss_20b_b64_gemv.sqlite`
- GEMV disabled: `/tmp/qmoe_profile_20260613/gpt_oss_20b_b64_gemm.sqlite`

| Mode | Benchmark latency (ms) | Key FC kernels in benchmark range |
|------|------------------------|-----------------------------------|
| GEMV enabled | 0.068951 | `moe_gemv_interleaved_swiglu_kernel`: 100 calls, 14.37 us avg; `moe_gemv_kernel`: 100 calls, 10.91 us avg |
| GEMV disabled | 0.096503 | `MoeFCGemm`: 200 calls, 22.36 us avg |

Result: real GEMV route, about 1.40x faster than grouped GEMM fallback by
end-to-end benchmark latency.

### Qwen3-6-35B-A3B Shape, INT4, `block_size=64`

Shape: `M=1`, `top_k=8`, `expanded_num_rows=8`, `hidden_size=2048`,
`intermediate_size=512`, `num_experts=256`, FP16.

The old `kMinProfiledProblemDimForExpandedRowsAbove4 = 704` policy produced a
fallback-vs-fallback comparison:

| Mode | Benchmark latency (ms) | Route |
|------|------------------------|-------|
| GEMV enabled | 0.072524 | grouped GEMM fallback, `MoeFCGemm`: 200 calls, 10.90 us avg |
| GEMV disabled | 0.074525 | grouped GEMM fallback, `MoeFCGemm`: 200 calls, 10.93 us avg |

After lowering `kMinProfiledProblemDimForExpandedRowsAbove4` to 512:

Artifacts:

- GEMV enabled: `/tmp/qmoe_profile_20260613/qwen3_6_35b_a3b_b64_gate512_gemv.sqlite`
- GEMV disabled: `/tmp/qmoe_profile_20260613/qwen3_6_35b_a3b_b64_gate512_gemm.sqlite`

| Mode | Benchmark latency (ms) | Key FC kernels in benchmark range |
|------|------------------------|-----------------------------------|
| GEMV enabled | 0.052437 | `moe_gemv_interleaved_swiglu_kernel`: 100 calls, 5.07 us avg; `moe_gemv_kernel`: 100 calls, 3.28 us avg |
| GEMV disabled | 0.073088 | `MoeFCGemm`: 200 calls, 10.81 us avg |

Result: the 512 gate makes Qwen a true GEMV-vs-GEMM comparison and improves
end-to-end latency by about 1.39x. FC kernel time drops from about 2.162 ms
total per 100 iterations for grouped GEMM to about 0.835 ms total per 100
iterations for GEMV.

### Decision

- Keep `block_size=64` as the immediate model-shape profiling target.
- Keep symmetric block-wise INT4/INT8 support for both 64 and 128 in the
  implementation, but only report model-shape profiling for 64 in this round.
- Keep `kMinProfiledProblemDimForExpandedRowsAbove4 = 512` so Qwen-style
  `top_k=8`, `intermediate_size=512` decode runs can use the custom GEMV path.

## 2026-06-14 Block Size 32 vs 64: SM90, FP16 INT4, Warmup 5, Repeat 100

### Setup

- Goal: compare the newly enabled INT4 `block_size=32` path against the existing
  `block_size=64` path on real model-shaped QMoE decode workloads.
- GPU: single H200 (SM90).
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`, CUDA 13.0,
  after the `block_size=32` CUTLASS/GEMV changes were built and installed.
- Python: `~/onnxruntime/.venv/bin/python`.
- Nsight Systems: `nsys 2025.3.2.367`. The first `nsys` run hit a
  `/tmp/nvidia/nsight_systems` permission issue; rerunning with `TMPDIR` under
  the artifact directory fixed it.
- Artifacts: `/tmp/qmoe_profile_block32_vs64_20260614_024925/`.

The profiling CLI was updated to accept `--block-size 32`; prior to this run it
only allowed `0/64/128` even though the benchmark helper could construct custom
cases.

Command template:

```bash
cd ~/onnxruntime
source .venv/bin/activate
export CUDA_VERSION=13.0
export CUDA_HOME=~/cuda13.0
export CUDNN_HOME=~/cudnn_9.19_cuda13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/build/cu130/Release:$PWD/onnxruntime/test/python/transformers:${PYTHONPATH:-}

python onnxruntime/test/python/transformers/profile_qmoe_gemv.py \
  --case <case-name> --block-size <32|64> --warmup 5 --repeat 100

TMPDIR=/tmp/qmoe_profile_block32_vs64_20260614_024925/tmp \
bash onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --case <case-name> --block-size <32|64> --warmup 5 --repeat 100 \
  -o /tmp/qmoe_profile_block32_vs64_20260614_024925/<output-name>
```

### Standalone End-To-End Benchmark Latency

Lower is better. These numbers are from the plain benchmark loop, outside nsys,
with the default GEMV-enabled route. Every case reported
`has_invalid_output=false`.

| Model case | Expanded rows | Block size | Latency (ms) | Relative to b64 |
|------------|---------------|------------|--------------|-----------------|
| `gpt_oss_20b` (`2880x2880`, e32, top4) | 4 | 32 | 0.068118 | 1.006x |
| `gpt_oss_20b` (`2880x2880`, e32, top4) | 4 | 64 | 0.067712 | 1.000x |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | 8 | 32 | 0.050836 | 0.991x |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | 8 | 64 | 0.051312 | 1.000x |

Result: `block_size=32` and `block_size=64` are effectively tied on the default
GEMV path for these model-shaped decode cases. GPT-OSS is about 0.6% faster with
`block_size=64`; Qwen3.6-35B-A3B is about 0.9% faster with `block_size=32`. Both
differences are small enough to treat as measurement noise unless repeated
end-to-end model runs show the same direction.

### Nsight Route Confirmation And FC Kernel Timing

All rows below are restricted to the `benchmark` NVTX range. The GEMV-enabled
route uses `moe_gemv_interleaved_swiglu_kernel` for FC1 and `moe_gemv_kernel` for
FC2. The disabled route (`ORT_DISABLE_MOE_GEMV=1`) uses CUTLASS `MoeFCGemm`.

| Model case | Block size | Mode | Route kernel | Calls | Avg us |
|------------|------------|------|--------------|-------|--------|
| `gpt_oss_20b` | 32 | GEMV enabled | `moe_gemv_interleaved_swiglu_kernel` | 100 | 17.878 |
| `gpt_oss_20b` | 32 | GEMV enabled | `moe_gemv_kernel` | 100 | 12.978 |
| `gpt_oss_20b` | 32 | GEMV disabled | `MoeFCGemm` | 200 | 43.283 |
| `gpt_oss_20b` | 64 | GEMV enabled | `moe_gemv_interleaved_swiglu_kernel` | 100 | 17.655 |
| `gpt_oss_20b` | 64 | GEMV enabled | `moe_gemv_kernel` | 100 | 12.364 |
| `gpt_oss_20b` | 64 | GEMV disabled | `MoeFCGemm` | 200 | 41.113 |
| `qwen3_6_35b_a3b` | 32 | GEMV enabled | `moe_gemv_interleaved_swiglu_kernel` | 100 | 6.266 |
| `qwen3_6_35b_a3b` | 32 | GEMV enabled | `moe_gemv_kernel` | 100 | 4.000 |
| `qwen3_6_35b_a3b` | 32 | GEMV disabled | `MoeFCGemm` | 200 | 20.719 |
| `qwen3_6_35b_a3b` | 64 | GEMV enabled | `moe_gemv_interleaved_swiglu_kernel` | 100 | 6.201 |
| `qwen3_6_35b_a3b` | 64 | GEMV enabled | `moe_gemv_kernel` | 100 | 3.930 |
| `qwen3_6_35b_a3b` | 64 | GEMV disabled | `MoeFCGemm` | 200 | 20.000 |

The custom GEMV kernels are present for both `block_size=32` and 64. Kernel-level
timing is slightly lower for `block_size=64` in this profile, but the standalone
benchmark latency is essentially flat between the two block sizes.

### nsys Wrapper Benchmark Latency

These numbers are useful for route comparisons but include profiling overhead;
prefer the standalone table above for clean latency. Lower is better.

| Model case | Mode | Block size 32 (ms) | Block size 64 (ms) |
|------------|------|--------------------|--------------------|
| `gpt_oss_20b` | GEMV enabled | 0.075520 | 0.072995 |
| `gpt_oss_20b` | GEMV disabled | 0.140200 | 0.134682 |
| `qwen3_6_35b_a3b` | GEMV enabled | 0.055514 | 0.064243 |
| `qwen3_6_35b_a3b` | GEMV disabled | 0.093866 | 0.096064 |

The profiled fallback runs confirm GEMV remains substantially faster than the
grouped-GEMM fallback for both block sizes and both model shapes.

### Artifacts

Primary summaries:

- `/tmp/qmoe_profile_block32_vs64_20260614_024925/benchmark_results.jsonl`
- `/tmp/qmoe_profile_block32_vs64_20260614_024925/route_kernel_summary.tsv`
- `/tmp/qmoe_profile_block32_vs64_20260614_024925/artifacts.txt`

Nsight Systems produced one `.nsys-rep` and one `.sqlite` for each
`{model, block_size, mode}` tuple. Examples:

- `/tmp/qmoe_profile_block32_vs64_20260614_024925/nsys_gpt_oss_20b_m1_top4_fp16_2880x2880_e32_b32_gemv.sqlite`
- `/tmp/qmoe_profile_block32_vs64_20260614_024925/nsys_gpt_oss_20b_m1_top4_fp16_2880x2880_e32_b64_gemv.sqlite`
- `/tmp/qmoe_profile_block32_vs64_20260614_024925/nsys_qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256_b32_gemv.sqlite`
- `/tmp/qmoe_profile_block32_vs64_20260614_024925/nsys_qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256_b64_gemv.sqlite`

### Decision

- Keep `block_size=32` enabled for INT4 QMoE. It reaches the same custom GEMV
  route as `block_size=64` for the GPT-OSS-20B and Qwen3.6-35B-A3B decode
  shapes, with no meaningful end-to-end latency regression in the standalone
  benchmark loop.
- Do not tune the GEMV gate differently for 32 vs 64 based on this data. The
  current model-shape route is valid for both block sizes.
- Continue using nsys NVTX-range kernel evidence, not benchmark JSON alone, when
  verifying whether a block-wise QMoE case actually reached GEMV or fell back to
  grouped GEMM.

## 2026-06-13 BF16 GEMV Enablement: SM90, Warmup 5, Repeat 100

### Setup

- Goal: extend the CUDA QMoE GEMV fast path from FP16-only activations to BF16,
  reaching dispatch and kernel parity with FP16.
- GPU: single H200 (SM90), `CUDA_VISIBLE_DEVICES=1` on an otherwise idle GPU
  (`nvidia-smi` confirmed 0% before recording).
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`, CUDA 13.0.
- Python: `~/onnxruntime/.venv_cu130/bin/python` (Python 3.14).
- Nsight Systems: `~/cuda13.0/bin/nsys`. Kernel parsing used
  `parse_nsys.py --nvtx-range benchmark --pattern '%'` so both the CUTLASS
  grouped-GEMM fallback and the custom GEMV kernels appear.
- All model-shape runs used `--warmup 5 --repeat 100 --block-size 64`.

Implementation summary:

- The runtime gate in `moe_kernels.cu` relaxes from `std::is_same_v<T, half>` to
  `std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>` for both
  `tryLaunchMoeGemvIntSymmetric` and the interleaved-SwiGLU variant.
- `moe_gemv.cu` adds `__nv_bfloat16` `DetailsForTAndWeight` specializations for
  `cutlass::uint4b_t` and `uint8_t`, plus the matching `__nv_bfloat16` template
  instantiations (group sizes 0/64/128, INT4/INT8, bias on/off) under
  `ENABLE_BF16`.
- BF16 and FP16 now share one dispatch gate and one set of custom kernels.

Command template (run once per dtype):

```bash
cd /tmp
export CUDA_HOME=~/cuda13.0
export CUDNN_HOME=~/cudnn9.19_cuda13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/onnxruntime/test/python/transformers

CUDA_VISIBLE_DEVICES=1 \
~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
   --python ~/onnxruntime/.venv_cu130/bin/python \
   --case <case-name> --dtype <FLOAT16|BFLOAT16> \
   --block-size 64 --warmup 5 --repeat 100 \
   -o /tmp/qmoe_gemv_bf16_20260613/<output-name>
```

### Routing Parity

Both dtypes route to the same custom kernels for the same shape:
`moe_gemv_interleaved_swiglu_kernel` for FC1 and `moe_gemv_kernel` for FC2.

| Case | Expanded rows | DType | FC1 kernel | FC2 kernel | Route |
|------|---------------|-------|------------|------------|-------|
| `gpt_oss_20b` (`2880x2880`, e32, top4) | 4 | FP16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |
| `gpt_oss_20b` (`2880x2880`, e32, top4) | 4 | BF16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | 8 | FP16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | 8 | BF16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |
| `gemma4_26b_a4b` (`2816x704`, e128, top8) | 8 | FP16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |
| `gemma4_26b_a4b` (`2816x704`, e128, top8) | 8 | BF16 | `moe_gemv_interleaved_swiglu_kernel` | `moe_gemv_kernel` | GEMV |

### End-To-End Benchmark Latency (block_size=64, INT4)

Lower is better. `Enabled` is the default GEMV build; `Fallback` sets
`ORT_DISABLE_MOE_GEMV=1`. Values are the benchmark-loop latency in milliseconds.

| Model case | DType | Enabled ms | Fallback ms | Speedup |
|------------|-------|------------|-------------|---------|
| `gpt_oss_20b` (`2880x2880`, e32, top4) | FP16 | 0.0683 | 0.0969 | 1.42x |
| `gpt_oss_20b` (`2880x2880`, e32, top4) | BF16 | 0.0673 | 0.0995 | 1.48x |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | FP16 | 0.0512 | 0.0730 | 1.43x |
| `qwen3_6_35b_a3b` (`2048x512`, e256, top8) | BF16 | 0.0526 | 0.0724 | 1.38x |
| `gemma4_26b_a4b` (`2816x704`, e128, top8) | FP16 | 0.0595 | 0.0827 | 1.39x |
| `gemma4_26b_a4b` (`2816x704`, e128, top8) | BF16 | 0.0604 | 0.0771 | 1.28x |

### Primary FC Compute Kernel Timing

Average kernel duration in microseconds inside the measured `benchmark` NVTX
range. GEMV columns are FC1 (`moe_gemv_interleaved_swiglu_kernel`) and FC2
(`moe_gemv_kernel`); the fallback column is the CUTLASS `MoeFCGemm` average over
its 200 calls.

| Model case | DType | FC1 GEMV us | FC2 GEMV us | Fallback `MoeFCGemm` us |
|------------|-------|-------------|-------------|--------------------------|
| `gpt_oss_20b` | FP16 | 14.17 | 10.97 | 22.14 |
| `gpt_oss_20b` | BF16 | 14.58 | 11.00 | 22.71 |
| `qwen3_6_35b_a3b` | FP16 | 5.11 | 3.24 | 10.67 |
| `qwen3_6_35b_a3b` | BF16 | 5.29 | 3.35 | 10.94 |
| `gemma4_26b_a4b` | FP16 | 7.52 | 4.59 | 14.96 |
| `gemma4_26b_a4b` | BF16 | 9.45 | 5.29 | 12.28 |

### Standalone INT4/INT8 Synthetic Shape (`1024x4096`, e8, top2)

| Case | DType | Route | FC1 GEMV us | FC2 GEMV us |
|------|-------|-------|-------------|-------------|
| `blockwise_int4_b64` | FP16 | GEMV | 4.53 | 6.95 |
| `blockwise_int4_b64` | BF16 | GEMV | 4.62 | 7.01 |
| `blockwise_int8_b64` | FP16 | grouped GEMM fallback | — | — |
| `blockwise_int8_b64` | BF16 | grouped GEMM fallback | — | — |

The INT4 GEMV kernel times are within noise across dtypes. End-to-end latency on
this tiny `e8` shape is high-variance (the disabled path swung between 0.17 and
0.27 ms across repeats), so only the stable in-NVTX kernel times are reported
here. INT8 block-wise stays on grouped GEMM for this shape, and that fallback is
dtype-independent (both FP16 and BF16 fall back identically).

### Correctness

- Focused SwiGLU BF16 parity tests passed:
  `pytest test_qmoe_cuda.py -k "swiglu and bf16"` reported `16 passed, 59
  deselected`.
- Every benchmark case above reported `has_invalid_output=false`, so the
  GEMV-enabled BF16 output matched the reference within tolerance.

### Per-Channel Note

The default per-channel (`block_size=0`) `gpt_oss` benchmark case stays on
grouped GEMM in this build for both FP16 and BF16 (the trace shows two
`MoeFCGemm` calls plus a separate `doGatedActivationKernel`). The fallback is
dtype-independent, so BF16 still matches FP16 behavior. The current branch's
confirmed model-shape GEMV coverage is the block-wise (`block_size=64`) path used
in the tables above.

> Update (2026-06-13): the per-channel fallback observed here was a gate bug
> (`group_size = -1` rejected by `is_moe_gemv_supported`), not a fundamental
> limitation. It is fixed in the "INT8 Per-Column GEMV Enablement" section below,
> after which per-column INT4/INT8 route to GEMV for FP16 and BF16.


### Decision

- Keep the relaxed gate: BF16 and FP16 share one dispatch path and one set of
  custom GEMV kernels.
- For every profiled model shape, BF16 routes to GEMV exactly where FP16 does and
  matches FP16 latency within measurement noise.

## 2026-06-13 INT8 Per-Column GEMV Enablement: SM90, `block_size=0`, Warmup 5, Repeat 100

### Setup

- Goal: route per-column (`block_size <= 0`) symmetric INT8 W8A16 QMoE decode
  shapes to the custom GEMV fast path for FP16 and BF16 activations.
- GPU: single H200 (SM90), `CUDA_VISIBLE_DEVICES=1` on an otherwise idle GPU
  (`nvidia-smi` confirmed 0% before recording).
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`, CUDA 13.0,
  `CMAKE_CUDA_ARCHITECTURES=89;90`. Rebuilt with `--clean_moe`.
- Python: `~/onnxruntime/.venv_cu130/bin/python` (Python 3.14).
- Nsight Systems: `~/cuda13.0/bin/nsys`. Kernel parsing used
  `parse_nsys.py --nvtx-range benchmark --pattern '%'`.
- Artifacts: `/tmp/qmoe_gemv_int8pc_20260613/<case>_{gemv,gemm}.{nsys-rep,sqlite}`.

### Root Cause Of The Prior Per-Column Fallback

The earlier BF16 section's "Per-Channel Note" observed that per-channel
(`block_size=0`) cases stayed on grouped GEMM. The cause was in the GEMV gate:

- The QMoE runtime carries per-column scales through `QuantParams::Int`, which
  leaves `groupwise.group_size` at its struct default of `-1`
  (`moe_kernels.h`).
- `is_moe_gemv_supported` previously required `group_size == 0` exactly for the
  per-column case, so `group_size = -1` was rejected (`sup=0`) and dispatch fell
  back to grouped GEMM — for per-column INT4 *and* INT8.
- The GEMV launcher dispatch (`dispatch_moe_gemv_group_size` and the
  interleaved-SwiGLU variant) already maps any `group_size <= 0` to the
  `GroupSize == 0` per-column kernel.

The fix relaxes `is_moe_gemv_supported` to accept any `group_size <= 0` as the
per-column case (alongside block-wise 64/128). No new kernel instantiation is
needed: `(half, uint8_t)` and `(__nv_bfloat16, uint8_t)` GEMV details already
exist from the block-wise INT8 work.

### Routing Confirmation

Both FC stages now route to the custom kernels for per-column INT8. Values are
average kernel duration in microseconds inside the measured `benchmark` NVTX
range; FC1 is `moe_gemv_interleaved_swiglu_kernel`, FC2 is `moe_gemv_kernel`.

| Case | Expanded rows | DType | FC1 GEMV us | FC2 GEMV us | Route |
|------|---------------|-------|-------------|-------------|-------|
| `int8_per_column_m1_top2_*_1024x4096_e8` | 2 | FP16 | 5.15 | 5.26 | GEMV |
| `int8_per_column_m1_top2_*_1024x4096_e8` | 2 | BF16 | 6.24 | 5.75 | GEMV |
| `gpt_oss_20b_m1_top4_int8_2880x2880_e32` | 4 | FP16 | 22.57 | 11.90 | GEMV |
| `gpt_oss_20b_m1_top4_int8_2880x2880_e32` | 4 | BF16 | 22.71 | 12.13 | GEMV |

### End-To-End Benchmark Latency

Lower is better. `Enabled` is the default GEMV build; `Fallback` sets
`ORT_DISABLE_MOE_GEMV=1`. Values are the benchmark-loop latency in milliseconds.
Every case reported `has_invalid_output=false`.

| Model case | DType | Enabled ms | Fallback ms | Speedup |
|------------|-------|------------|-------------|---------|
| `int8_per_column_m1_top2_1024x4096_e8` | FP16 | 0.0566 | 0.0816 | 1.44x |
| `int8_per_column_m1_top2_1024x4096_e8` | BF16 | 0.0578 | 0.0862 | 1.49x |
| `gpt_oss_20b_m1_top4_int8_2880x2880_e32` | FP16 | 0.0785 | 0.0947 | 1.21x |
| `gpt_oss_20b_m1_top4_int8_2880x2880_e32` | BF16 | 0.0785 | 0.0989 | 1.26x |

### Correctness

- Focused INT8 per-column SwiGLU parity tests passed:
  `pytest test_qmoe_cuda.py -k "test_swiglu_qmoe_parity_1 or
  test_swiglu_qmoe_parity_3 or test_swiglu_qmoe_parity_bf16_1 or
  test_swiglu_qmoe_parity_bf16_3"` reported `4 passed`.
- Regression check: `pytest -k "TestSwigluQMoE or TestQMoEIntPrePackSmoke"`
  reported `34 passed, 4 skipped`.
- The per-column INT4 and block-wise INT4/INT8 routes are unchanged: block-wise
  cases still pass `group_size = 64/128` and per-column INT4 now also reaches
  GEMV via the same `group_size <= 0` relaxation.

### Decision

- Keep the relaxed gate: `is_moe_gemv_supported` accepts `group_size <= 0` as the
  per-column case for INT4 and INT8.
- Per-column INT8 W8A16 decode shapes route to GEMV for both FP16 and BF16 and
  beat the grouped-GEMM fallback at every profiled shape.

## 2026-06-19: Split-K2 Two-Pass SwiGLU GEMV Experiment

### Change Under Test

- Code commit: `f1d6718be719c1237be392c0389874b6a8926a3c`
  (`Experiment QMoE split-K SwiGLU GEMV`).
- Added default Split-K2 route with opt-out env knob:
  `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1`.
- Scope: FP16 INT4/interleaved-SwiGLU FC1 GEMV path for decode-shaped QMoE.
- Implementation:
  - First pass launches `moe_gemv_splitk_partials_kernel` with `SplitK=2` and
    writes FP32 partials into QMoE workspace.
  - Second pass launches `moe_gemv_splitk_reduce_swiglu_kernel` to reduce the
    partials, add optional bias, and apply SwiGLU.
  - FC2 remains on the existing `moe_gemv_kernel`.
  - Scratch is allocated only for the supported Split-K2 route. Setting
    `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1` restores the previous single-kernel
    FC1 SwiGLU GEMV path.

### Repro Notes

- Build: `cmake --build build/cu130/Release --target onnxruntime_providers_cuda --parallel $(nproc)`.
- Important provider sync: Python tests importing from
  `build/cu130/Release/onnxruntime` load
  `build/cu130/Release/onnxruntime/capi/libonnxruntime_providers_cuda.so`, not
  only the top-level `build/cu130/Release/libonnxruntime_providers_cuda.so` or
  the venv copy. Sync all relevant copies before measuring:

  ```bash
  cp build/cu130/Release/libonnxruntime_providers_cuda.so \
     build/cu130/Release/onnxruntime/capi/libonnxruntime_providers_cuda.so
  cp build/cu130/Release/libonnxruntime_providers_cuda.so \
     .venv_cu130/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
  ```

- Focused QMoE helper:

  ```bash
  cd ~
  CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH=~/onnxruntime/build/cu130/Release:~/cuda13.0/lib64:~/cudnn9.19_cuda13/lib:~/cudnn9.19_cuda13/lib64:${LD_LIBRARY_PATH:-} \
  PYTHONPATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/onnxruntime/test/python/transformers \
  ~/onnxruntime/.venv_cu130/bin/python \
  ~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.py \
    --case gpt_oss_20b_m1_top4_fp16_2880x2880_e32 --warmup 3 --repeat 20
  ```

### Focused QMoE Smoke

Both modes reported `has_invalid_output=false`.

| Mode | Env | Latency ms |
|------|-----|------------|
| Baseline | `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1` | 0.072344 |
| Split-K2 | none | 0.073816 |

The short helper was slightly slower with split-K2, so Nsight was required to
confirm route selection and isolate kernel time.

### Nsight Systems Kernel Results

Artifacts:

- Baseline: `/tmp/qmoe_gptoss_baseline_final.{nsys-rep,sqlite}`
- Split-K2: `/tmp/qmoe_gptoss_splitk_final.{nsys-rep,sqlite}`

Command shape:

```bash
~/cuda13.0/bin/nsys profile -t cuda,nvtx --force-overwrite true \
  -o /tmp/qmoe_gptoss_splitk_final --export=sqlite \
  ~/onnxruntime/.venv_cu130/bin/python \
  ~/onnxruntime/onnxruntime/test/python/transformers/profile_qmoe_gemv.py \
    --case gpt_oss_20b_m1_top4_fp16_2880x2880_e32 --warmup 3 --repeat 30 --nvtx
```

Parsed with `parse_nsys.py --nvtx-range benchmark --pattern '%'`.

| Mode | Kernel | Calls | Avg us |
|------|--------|-------|--------|
| Baseline | `moe_gemv_interleaved_swiglu_kernel` | 30 | 21.42 |
| Baseline | `moe_gemv_kernel` | 30 | 12.13 |
| Split-K2 | `moe_gemv_splitk_partials_kernel` | 30 | 17.59 |
| Split-K2 | `moe_gemv_splitk_reduce_swiglu_kernel` | 30 | 2.39 |
| Split-K2 | `moe_gemv_kernel` | 30 | 12.22 |

Split-K2 reduced FC1 kernel work from about `21.42 us` to `17.59 + 2.39 =
19.98 us`, a net FC1 reduction of about `1.44 us` per QMoE invocation. End-to-end
under Nsight was effectively tied:

| Mode | Helper latency ms |
|------|-------------------|
| Baseline | 0.079855 |
| Split-K2 | 0.079728 |

### Model-Level Decode Benchmark With CUDA Graph

The user requested model-level measurement assuming CUDA graph. Both runs used
the GPT-OSS-20B INT4 QMoE model package, CUDA graph enabled, XQA enabled, and
deterministic MoE tactic selection:

```bash
MODEL=models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
GPU=0 PROMPT_LEN=512 GEN_LEN=128 REPS=10 WARMUP=3 CUDA_GRAPH=1 XQA=1 SYNC_LIB=1 \
ORT_FORCE_DETERMINISTIC_MOE=1 \
bash scripts/bench_gpt_oss_ort_decode.sh
```

Baseline additionally set `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1`.

| Run | Mode | Decode latency ms/token | Decode throughput tok/s |
|-----|------|-------------------------|-------------------------|
| R1, `REPS=5`, `WARMUP=2` | Baseline | 2.869450 | 348.498901 |
| R1, `REPS=5`, `WARMUP=2` | Split-K2 | 2.823800 | 354.132707 |
| R2, `REPS=10`, `WARMUP=3` | Baseline | 2.865840 | 348.937861 |
| R2, `REPS=10`, `WARMUP=3` | Split-K2 | 2.839335 | 352.195107 |

The longer CUDA-graph pair showed about `+0.9%` decode throughput. The shorter
pair showed about `+1.6%`. Since the focused helper reported valid output and
the model-level gain repeated in the same direction, even this modest gain is
worth enabling for GPT-OSS-20B decode while keeping an opt-out for A/B checks.

After flipping Split-K2 to the default and adding
`ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1` as the opt-out, three more paired
CUDA-graph model runs were collected with `REPS=10`, `WARMUP=3`, prompt length
512, and generation length 128:

| Run | Mode | Decode latency ms/token | Decode throughput tok/s |
|-----|------|-------------------------|-------------------------|
| R3 | Default Split-K2 | 3.017252 | 331.427448 |
| R3 | Split-K2 disabled | 3.055736 | 327.253380 |
| R4 | Default Split-K2 | 3.006739 | 332.586260 |
| R4 | Split-K2 disabled | 3.047570 | 328.130314 |
| R5 | Default Split-K2 | 3.009466 | 332.284898 |
| R5 | Split-K2 disabled | 3.047015 | 328.190090 |
| Average | Default Split-K2 | 3.011152 | 332.099536 |
| Average | Split-K2 disabled | 3.050107 | 327.857928 |

The default Split-K2 route was faster in all three pairs, averaging `+1.29%`
decode throughput and `-1.28%` decode latency versus the opt-out fallback.

### Accuracy Smoke

A 1000-sample `match_mmlu` smoke was run with the local parallel eval harness on
all eight H200 GPUs, using the same GPT-OSS-20B INT4 QMoE model package and the
current ORT build package. The default Split-K2 run scored `0.8380` pooled
accuracy; the opt-out fallback with `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1`
scored `0.8350`. The small positive difference is within smoke-test noise, and
there is no accuracy regression signal from enabling Split-K2 by default.

### Decision

- Enable Split-K2 by default for its supported fp16 INT4 interleaved-SwiGLU GEMV
  scope.
- Keep `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1` as the fallback and A/B knob.
- The 1000-sample MMLU smoke matched the opt-out fallback within noise, so the
  default flip has an accuracy sanity check in addition to focused-helper valid
  output.
- Future work:
  - Add per-shape autotune so route selection is data-driven instead of a fixed
    default.
  - Try a launch-fused reduction strategy or cooperative approach to keep the
    FC1 parallelism benefit without the extra reduce launch.
## 2026-06-19 FP16 Accumulation Default: SM90, GPT-OSS Decode Shape

### Setup

- Goal: make fp16 accumulation the default for fp16 QMoE GEMV, while preserving
  the previous fp32 accumulation path behind `ORT_MOE_GEMV_FP32_ACCUM=1`.
- GPU: single H200 (SM90).
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release`, CUDA 13.0.
- QMoE helper case: `gpt_oss_20b_m1_top4_fp16_2880x2880_e32`, warmup 5,
  repeat 20.
- Full-model case: GPT-OSS-20B INT4 QMoE, batch 1, prompt 512, generation 128,
  warmup 2, repeat 5, CUDA graph enabled, XQA enabled, deterministic MoE tactic
  selection.

### Standalone QMoE Helper

Lower is better. Both modes reported `has_invalid_output=false`.

| Mode | Helper latency ms | FC1 SwiGLU GEMV avg us | FC2 GEMV avg us |
|------|------------------:|-----------------------:|----------------:|
| default fp16 accumulation | 0.0708 | 13.93 | 10.14 |
| `ORT_MOE_GEMV_FP32_ACCUM=1` | 0.0812 | 21.57 | 12.24 |

The new default is about 12.8% faster than the fp32 fallback in the isolated
GPT-OSS decode-shaped QMoE helper. The gain comes from the expected GEMV rows:
FC1 interleaved SwiGLU is about 35% faster and FC2 GEMV is about 17% faster.

### Full GPT-OSS Decode

| Mode | Decode latency ms/token | Decode throughput tok/s |
|------|-------------------------:|-------------------------:|
| default fp16 accumulation | 2.588930 | 386.259956 |
| `ORT_MOE_GEMV_FP32_ACCUM=1` | 2.827260 | 353.699315 |

The full-model A/B shows the default fp16 accumulation path is about 9.2% faster
in decode throughput than the fp32 fallback in this run.

### Accuracy Smoke

Prior 1000-sample MMLU runs found no pooled-accuracy difference between the old
fp32 default and the fp16-accumulation experiment:

| Mode | Output dir | Pooled accuracy |
|------|------------|-----------------|
| fp32 accumulation | `~/eval_runs/mmlu1000_default_20260619_001348` | 0.8260 |
| fp16 accumulation | `~/eval_runs/mmlu1000_fp16accum_20260619_001352` | 0.8260 |

### Decision

- Make fp16 accumulation the default for fp16 QMoE GEMV.
- Keep bf16 on fp32 accumulation.
- Keep `ORT_MOE_GEMV_FP32_ACCUM=1` as the opt-in numerical fallback and A/B knob.

## 2026-06-19: Exact-Shape GPT-OSS Router GEMV Specialization

### Change Under Test

- Scope: standalone `MatMulNBits` router projection used before each GPT-OSS
  `QMoE` node.
- Exact shape: `M=1`, `N=32`, `K=2880`, `block_size=32`, 4-bit weights, no
  zero points.
- Added CUDA provider kernel: `MatMulFloatInt4RouterKernel<T>`.
- Dispatch: `TryMatMul4Bits` selects the specialized kernel only for the exact
  shape above. For clean A/B measurement, set
  `ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION=1` to force the generic 4-bit
  M=1 route.
- Motivation: avoiding 32 router logits in global memory is small. The more
  direct win is reducing overhead in the special `N=32`, `K=2880` int4 router
  GEMV itself.

### Build and Sync

```bash
cd ~/onnxruntime
git diff --check
cmake --build build/cu130/Release --target onnxruntime_providers_cuda --parallel $(nproc)
cp build/cu130/Release/libonnxruntime_providers_cuda.so \
  build/cu130/Release/onnxruntime/capi/libonnxruntime_providers_cuda.so
cp build/cu130/Release/libonnxruntime_providers_cuda.so \
  .venv_cu130/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
```

### Nsight Route Check

Profile command:

```bash
cd ~/onnxruntime
PYTHONPATH=~/onnxruntime/build/cu130/Release \
MODEL=models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
GPU=0 PROMPT_LEN=512 GEN_LEN=8 REPS=1 WARMUP=0 CUDA_GRAPH=0 XQA=1 SYNC_LIB=0 \
ORT_FORCE_DETERMINISTIC_MOE=1 \
~/cuda13.0/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --export=sqlite -o /tmp/gptoss_routergemv_special_cg0 \
  bash scripts/h200_18/bench_gpt_oss_ort_decode.sh
```

Nsight artifacts:

- `/tmp/gptoss_routergemv_special_cg0.nsys-rep`
- `/tmp/gptoss_routergemv_special_cg0.sqlite`
- `/tmp/gptoss_routergemv_special_stats_cuda_gpu_kern_sum.csv`

Kernel summary rows from the no-CUDA-graph profile:

| Kernel | Total ns | Instances | Avg ns |
|--------|---------:|----------:|-------:|
| generic 4-bit M=1 GEMV (pre-split profile) | 3318180 | 343 | 9674.0 |
| specialized `MatMulFloatInt4RouterKernel<__half>` | 655863 | 168 | 3903.9 |
| `SoftmaxTopKWarpBitonicKernel<__half, 8>` | 439868 | 192 | 2291.0 |

The `168` specialized router calls correspond to decode router projections in
the measured `GEN_LEN=8` no-CUDA-graph run. The generic int4 GEMV row remains for
other `MatMulNBits` projections.

### Model-Level Decode Benchmark With CUDA Graph

Both comparisons used the same rebuilt provider, CUDA graph enabled, XQA enabled,
and deterministic MoE tactic selection:

```bash
PYTHONPATH=~/onnxruntime/build/cu130/Release \
MODEL=models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
GPU=0 PROMPT_LEN=512 GEN_LEN=128 REPS=10 WARMUP=3 CUDA_GRAPH=1 XQA=1 SYNC_LIB=0 \
ORT_FORCE_DETERMINISTIC_MOE=1 \
bash scripts/h200_18/bench_gpt_oss_ort_decode.sh
```

The disabled runs additionally set
`ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION=1`.

| Run order | Mode | Decode latency ms/token | Decode throughput tok/s | Output CSV |
|-----------|------|-------------------------:|-------------------------:|------------|
| off then on | disabled | 2.870915 | 348.321025 | `~/ort_gptoss_routergemv_special_off.csv` |
| off then on | enabled | 2.819178 | 354.713290 | `~/ort_gptoss_routergemv_special_on.csv` |
| on then off | enabled | 2.824963 | 353.986941 | `~/ort_gptoss_routergemv_special_on_2.csv` |
| on then off | disabled | 2.870436 | 348.379097 | `~/ort_gptoss_routergemv_special_off_2.csv` |

### Decision

- Keep this as the current positive router experiment. The two CUDA-graph A/B
  pairs show about `+1.6%` to `+1.8%` model-level decode throughput.
- This is more useful than the earlier node-only router graph fusion because it
  reduces actual router GEMV kernel time and is exercised by the active GenAI
  benchmark through provider dispatch.
- The specialization is deliberately exact-shape gated. Broader routing should
  wait for more model shapes and correctness coverage.
- The earlier fused-router QMoE graph rewrite remains a separate prototype. Its
  main value is future launch reduction if GEMV, bias, softmax, and top-k are
  fused into one kernel; avoiding only the 32-float logits write is not expected
  to be the dominant win.

### Quality Follow-Up

- The dispatch logic now uses named GPT-OSS router shape constants and a helper
  predicate instead of repeating raw `M/N/K/block_size` literals at the launch
  site.
- Added C++ CUDA coverage:
  `MatMulNBits.Fp16_Int4_GptOssRouterShapeNoZeroPoint`. The test runs the exact
  `M=1`, `N=32`, `K=2880`, `block_size=32`, no-zero-point FP16 shape with the
  specialization enabled and then with
  `ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION=1` to cover the generic fallback.

Validation command:

```bash
cd ~/onnxruntime/build/cu130/Release
./onnxruntime_provider_test --gtest_filter='MatMulNBits.Fp16_Int4_GptOssRouterShapeNoZeroPoint'
```

Result: `1 test passed`.

## 2026-06-19: Router Bias Fusion Into Exact-Shape GEMV

### Change Under Test

- Scope: GPT-OSS router chain `MatMulNBits(K=2880, N=32, bits=4, block_size=32)`
  followed by `Add([32] bias)` before `QMoE`.
- Reuses the existing `MatMulNBitsFusion` graph transformer to rewrite
  `MatMulNBits + Add -> MatMulNBits` with bias input slot 5.
- CPU behavior remains unchanged. CUDA fusion is gated to the exact GPT-OSS
  router shape with no zero-points or `g_idx` input.
- The exact router GEMV kernel now accepts an optional bias pointer and adds the
  `[32]` bias before storing the router logits.
- The only runtime control is `ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION=1`,
  which disables the exact router GEMV specialization; the fused bias then falls
  back to a separate bias-add in the generic path. The graph-level
  `MatMulNBits + Add` rewrite itself is performed by the `MatMulNBitsFusion`
  transformer and has no dedicated opt-out env var.

### Graph Validation

Optimized model export used the rebuilt Python runtime directly from
`build/cu130/Release`:

```bash
cd /tmp
source ~/onnxruntime/.venv_cu130/bin/activate
PYTHONPATH=~/onnxruntime/build/cu130/Release \
LD_LIBRARY_PATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/build/cu130/Release/onnxruntime/capi:$LD_LIBRARY_PATH \
python <export-and-count-script>
```

| Mode | Total `Add` nodes | Router `MatMulNBits` nodes with bias |
|------|------------------:|--------------------------------------:|
| enabled | 48 | 24 |
| router bias fusion disabled | 72 | 0 |

This confirms all 24 real GPT-OSS router bias Adds are fused and disabling the
fusion restores the unfused graph.

### Model-Level Decode Benchmark With CUDA Graph

Both runs used the rebuilt runtime, CUDA graph enabled, XQA enabled, and
deterministic MoE tactic selection:

```bash
PYTHONPATH=~/onnxruntime/build/cu130/Release \
LD_LIBRARY_PATH=~/onnxruntime/build/cu130/Release:~/onnxruntime/build/cu130/Release/onnxruntime/capi:$LD_LIBRARY_PATH \
MODEL=models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
GPU=0 PROMPT_LEN=512 GEN_LEN=128 REPS=10 WARMUP=3 CUDA_GRAPH=1 XQA=1 SYNC_LIB=0 \
ORT_FORCE_DETERMINISTIC_MOE=1 \
bash scripts/h200_18/bench_gpt_oss_ort_decode.sh
```

| Mode | Decode latency ms/token | Decode throughput tok/s |
|------|-------------------------:|-------------------------:|
| enabled | 2.822998 | 354.233302 |
| router bias fusion disabled | 2.829066 | 353.473500 |

### Decision

- Keep the change. The gain is intentionally small, about `+0.2%` in this
  CUDA-graph run, but the implementation is exact-shape gated, tested, and
  removes 24 router Add kernels from the optimized graph.
- This result also bounds the value of router-only graph fusion: after the exact
  GEMV specialization, the remaining separate router bias Add is measurable but
  much smaller than the GEMV and softmax/top-k work.
- The next router/QMoE task should fuse more of the router tail, especially
  GEMV plus softmax/top-k, so the 32 logits can remain in registers or shared
  memory and avoid a separate top-k launch.

### Validation

```bash
cmake --build ~/onnxruntime/build/cu130/Release \
  --target onnxruntime_test_all onnxruntime_pybind11_state onnxruntime_provider_test \
  --parallel $(nproc)

cd ~/onnxruntime/build/cu130/Release
./onnxruntime_test_all \
  --gtest_filter='GraphTransformationTests.MatMulNBitsBiasFusion:GraphTransformationTests.MatMulNBitsBiasFusionDoesNotFuseUnsupportedCudaShape:GraphTransformationTests.MatMulNBitsBiasFusionCanDisableCudaRouterFusion'
./onnxruntime_provider_test \
  --gtest_filter='MatMulNBits.Fp16_Int4_GptOssRouterShapeNoZeroPoint'
```

Result: graph transformer tests `3 passed`; provider test `1 passed`.

## 2026-06-21: NVFP4 GEMV Packed E2M1 Dequantize (`prmt` Quad Decode)

### Change Under Test

- Scope: `Fp4I2FConverter` in
  `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/details.h`, i.e. every MXFP4 /
  NVFP4 QMoE GEMV kernel instantiated from `moe_gemv_fp4.cu` (`moe_gemv_kernel`
  and `moe_gemv_interleaved_swiglu_kernel`). INT4/INT8 use `I2FConverter` and are
  untouched.
- `Fp4I2FConverter::convert<N>()` previously decoded one E2M1 code at a time.
  Even though `decode()` was already branchless (a `prmt.b32` byte-select into a
  packed magnitude table), the surrounding per-element mask / shift / or / pack
  sequence dominated: ~53 ALU instructions per 8 codes.
- New packed path (`decode_quad`) taken whenever `N % 8 == 0`, which covers both
  live FP4 configurations (`ColumnMajor` `StepK=8` used by NVFP4, and the opt-in
  `ColumnMajorInterleaved` `StepK=32` used by MXFP4):
  - Load the weights a whole 32-bit word (eight codes) at a time.
  - `mag = w & 0x77777777` keeps the three magnitude bits per nibble and clears
    bit 3 so `prmt` stays in byte-select mode instead of sign-replicate mode.
  - `sgn = (w >> 3) & 0x11111111` becomes `0x00`/`0x80` sign bytes with one
    `prmt`.
  - One further `prmt` performs **four** magnitude table lookups at once, and one
    or two more expand the bytes into two packed `half2` / `bfloat16x2` words.
- Numerically this is a pure instruction-count change: the magnitude constants
  are the same exact `half`/`bf16` encodings of `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`,
  and nibble `i` still maps to output element `i`, so `pack_to_vec2`/`mma` need
  no changes. There is no env var; the packed path is unconditional on device.

### Bit-Exactness

A standalone harness compared `decode_quad` against the per-element reference for
both `half` and `__nv_bfloat16`. The reachable input space is only 2^16 wide (a
`decode_quad` call is fully described by its 4x3-bit magnitude selector plus its
4x1-bit sign selector), so the sweep of 256x256x64 randomized 32-bit words covers
every reachable pattern -- this is an exhaustive check, not a sample:

```
cuda=no error mismatches=0
```

### SASS Instruction Count (sm_90, CUDA 13.0)

Same object (`moe_gemv_fp4.cu.o`), built with and without the change:

| Kernel (fp16, `ColumnMajor`, CtaN=8, Threads=128, GroupSize=16) | Before | After | Delta |
|---|---:|---:|---:|
| `moe_gemv_interleaved_swiglu_kernel` (fc1) | 1088 | 760 | -30.1% |
| `moe_gemv_kernel` (fc2) | 960 | 640 | -33.3% |
| whole object, all FP4 instantiations | 300552 | 180112 | -40.1% |

### Repro Notes

Model: `qwen3.6_nvfp4_fp8dense_statscale_mtp_int8head_fp8kv_v10_w4`
(hidden 2048, inter 512, 256 experts, top_k 8, NVFP4 block_size 16, 40 layers),
H200 SXM, single GPU.

```bash
source ~/git/venv/bin/activate
export CUDA_HOME=/home/tianlei/cuda13.0 CUDNN_HOME=/home/tianlei/cudnn_9.23_cuda13
export LD_LIBRARY_PATH=/home/tianlei/ort_home_cu130_fp4_bench/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 ORT_ENABLE_FP4_GEMV=1 ORT_MTP_DIRECT_ARENA_COMMIT=1 ORT_FP4_GEMV_AUTOTUNE=0
# per-kernel timing: CUDA graph OFF so nsys reports real per-launch durations
nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/qmoe --force-overwrite true --export=sqlite \
  python scripts/h200_18/profile_mtp_decode.py $MODEL $HEAD off 3 20 10
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/qmoe.sqlite | grep moe_gemv
# end-to-end: CUDA graph ON
python scripts/h200_18/profile_mtp_decode.py $MODEL $HEAD on 3 200 50
```

Caveat that cost time here: this workspace has **three** copies of
`libonnxruntime_providers_cuda.so` (`onnxruntime/capi`, `onnxruntime_genai`, and
the `ort_home` lib dir). A/B swaps must replace all three, otherwise both arms
silently load the same library and report identical numbers.

### Nsight Systems Per-Kernel Results (CUDA graph OFF, 800 launches each)

Interleaved base/new reps, `ORT_FP4_GEMV_AUTOTUNE=0` (shipping default tiling):

| Kernel | Before (us) | After (us) | Delta |
|---|---:|---:|---:|
| `moe_gemv_interleaved_swiglu_kernel` (fc1) | 33.14 / 33.24 | 26.34 / 26.13 | **-20.9%** |
| `moe_gemv_kernel` (fc2) | 30.21 / 30.19 | 22.22 / 22.20 | **-26.5%** |

40 launches of each per decode step, so about `-0.60 ms/step` of GPU time.

### Model-Level Decode Benchmark (CUDA graph ON, 200 steps, 50 warmup)

Interleaved reps; `ms/step` is the clean metric because `tok/s` also moves with
the MTP acceptance rate.

| Rep | Before ms/step | After ms/step |
|---|---:|---:|
| 1 | 11.663 | 11.061 |
| 2 | 11.572 | 11.032 |
| 3 | 11.594 | 11.035 |
| mean | 11.610 | 11.043 |

**-4.9% step time (+5.1% step rate).** Every "after" rep beat every "before" rep.

### Validation

```bash
cd onnxruntime/test/python/transformers
ORT_ENABLE_FP4_GEMV=1 python -m pytest -q test_qmoe_nvfp4_cuda.py   # 22 passed
ORT_ENABLE_FP4_GEMV=0 python -m pytest -q test_qmoe_nvfp4_cuda.py   # 22 passed
```

### Decision

- Keep the change. It is bit-exact, has no env gate, and is the largest single
  QMoE NVFP4 GEMV win measured so far.
- After this change the kernels are no longer dominated by dequantize. The next
  FP4 GEMV lever is **tiling**, not the converter.

### Follow-Up: Tiling Default Is Still Wrong For These Shapes

`moe_gemv_fp4.cu` defaults to `Threads=128` with `StepK=8`, so `CtaK = 1024`.
For fc2 (`k = 512`) half the threads in every block do no work at all. Enabling
the existing autotuner picks `CtaN=8, Threads=64` for **both** GEMVs:

| Config | fc1 (us) | fc2 (us) | ms/step |
|---|---:|---:|---:|
| `ORT_FP4_GEMV_AUTOTUNE=0` (default) | 26.2 | 22.2 | 11.028 / 11.004 |
| `ORT_FP4_GEMV_AUTOTUNE=1` | 23.9 | 17.5 | 10.787 / 10.751 |

Another `-0.24 ms/step` (`-2.2%`) is available. `ORT_FP4_GEMV_AUTOTUNE` is off by
default because it synchronizes the inference stream, and it is skipped during
CUDA-graph capture, so the fix should be an analytic default (at minimum, drop to
`Threads=64` when `StepK * Threads > k`) rather than relying on the autotuner.
Resolved by the next section.

## 2026-07-28: Analytic Default Tiling For The FP4 GEMV

### Change Under Test

Follow-up to the previous section, which showed the shipping FP4 GEMV tiling
(`CtaN=8, Threads=128`) is not the best choice for the NVFP4 decode shapes but
that the only way to get the better one was `ORT_FP4_GEMV_AUTOTUNE=1`. The
autotuner is off by default because it synchronizes the inference stream, and it
is skipped during CUDA-graph capture, so in practice it never runs in the
configuration that matters.

New `gemv::Fp4MoeGemvDefaultConfig(expanded_num_rows, n, k)` in
`moe_gemv_fp4.cu` derives the tiling from the shape instead. `moe_quantization.cc`
now seeds `fc1_config` / `fc2_config` from it rather than from `kDefault`; the
autotuner, when enabled, still overrides it and the per-shape cache is unchanged.
Only `Threads` is derived. `CtaN` stays at `kDefaultCtaN`, so the analytic config
never changes which shapes `is_moe_gemv_fp4_supported` accepts. Note that
`Threads` is a tiling knob but **not** a bit-exact one: a block walks K in strides
of `CtaK = StepK * Threads` and the epilogue reduces across `Threads / 32` warps,
so changing it changes the floating-point summation order and the low bits of the
result can move.

Two clauses, both about the fact that the grid is `(expanded_num_rows, n / CtaN)`
and therefore does **not** depend on `Threads`:

- **(a) Idle threads.** A block walks K in strides of `CtaK = StepK * Threads`
  (`StepK = 8` for the ColumnMajor FP4 layout). When `CtaK > k` the tail of every
  block never enters the K loop. NVFP4 fc2 has `k = inter_size = 512` against
  `CtaK = 1024`, i.e. half of every block does nothing. `k < 1024 -> Threads=64`.
- **(b) Epilogue width.** MAC work per block is fixed by `(CtaN, k)` regardless of
  `Threads`, but the epilogue reduces partials across `Threads/32` warps through
  shared memory. A narrower block does the same math with half the barriers and a
  shallower reduction tree. This is gated on the grid being large enough that the
  SM still fills: these kernels use ~72 registers/thread on sm_90 (~900 resident
  threads/SM), which a 128-thread block reaches with ~7 resident blocks and a
  64-thread block only with ~14, so the threshold is **16 blocks/SM**
  (`expanded_num_rows * (n / CtaN) >= 16 * multiProcessorCount`).

`ORT_FP4_GEMV_DEFAULT_TILING=0` restores the fixed `kDefault` tiling.

### Selected Configs

Qwen3.6-35B-A3B-NVFP4 MTP decode (hidden 2048, inter 512, 256 experts, top_k 8,
expanded 32), H200, 132 SMs:

| GEMV | n | k | blocks | Clause | Chosen |
|---|---:|---:|---:|---|---|
| fc1 (swiglu) | 1024 | 2048 | 4096 (31/SM) | (b) | `Threads=64` |
| fc2 | 2048 | 512 | 8192 (62/SM) | (a) | `Threads=64` |

Single-token decode (expanded 8) keeps `Threads=128` for fc1 (1024 blocks, 7.8/SM,
below the clause (b) threshold) and still takes `Threads=64` for fc2 via clause
(a), which is the intended behavior.

The analytic choice reproduces the autotuner's pick exactly:

| Kernel | `AUTOTUNE=1` (us) | Analytic default (us) |
|---|---:|---:|
| `moe_gemv_interleaved_swiglu_kernel` (fc1) | 23.93 | 23.89 / 24.05 |
| `moe_gemv_kernel` (fc2) | 17.46 | 17.48 / 17.52 |

### Model-Level Decode Benchmark (CUDA graph ON, 200 steps, 50 warmup)

Same binary for both arms, toggled with `ORT_FP4_GEMV_DEFAULT_TILING`, interleaved
reps. Final threshold (16 blocks/SM):

| Rep | Fixed default ms/step | Analytic default ms/step |
|---|---:|---:|
| 1 | 10.986 | 10.739 |
| 2 | 11.002 | 10.725 |
| 3 | 11.061 | 10.742 |
| mean | 11.016 | 10.735 |

**-2.6% step time.** Every analytic rep beat every fixed-default rep. An earlier
build with the threshold at 8 blocks/SM measured -2.3% on the same shapes; the
16 blocks/SM threshold is the more conservative choice and loses nothing here.

Combined with the packed E2M1 decode from the previous section, decode goes from
11.610 ms/step to 10.735 ms/step, **-7.5%**.

### Validation

```bash
cd onnxruntime/test/python/transformers
ORT_ENABLE_FP4_GEMV=1 python -m pytest -q test_qmoe_nvfp4_cuda.py   # 22 passed
ORT_ENABLE_FP4_GEMV=0 python -m pytest -q test_qmoe_nvfp4_cuda.py   # 22 passed
ORT_ENABLE_FP4_GEMV=1 ORT_FP4_GEMV_DEFAULT_TILING=0 \
  python -m pytest -q test_qmoe_nvfp4_cuda.py                       # 22 passed
```

### Decision

- Keep. Every config computes the same dot products with the same accumulation
  dtype, so this is purely about picking the faster launch shape, and it now
  happens without a stream sync and works under CUDA graph capture. The summation
  order does depend on `Threads`, so the choice is not bit-neutral; the tests
  cover both clauses of the heuristic and the `=0` opt-out.
- Clause (a) is unconditional and carries most of the win (fc2: 22.2 -> 17.5 us).
  Clause (b) adds the fc1 win (26.2 -> 23.9 us) and is the part that generalizes
  least, hence the deliberately conservative occupancy-derived threshold and the
  `ORT_FP4_GEMV_DEFAULT_TILING=0` opt-out.
- The autotuner is still the right tool for shapes the heuristic gets wrong; it
  now starts from a better default and overrides it only when it measures a win.
