# MoE and QMoE — CUDA Operator Documentation

This document describes the design, schema, kernel dispatch, weight formats, and current
implementation status of the **MoE** (`com.microsoft::MoE`) and **QMoE**
(`com.microsoft::QMoE`) operators on the CUDA execution provider.

The CUTLASS kernels are derived from TensorRT-LLM (CUTLASS 4.4.2, commit `346018db87`)
and have been significantly modified for ONNX Runtime — see
[§16 Differences vs. TensorRT-LLM](#16-differences-vs-tensorrt-llm).

---

## Table of Contents

1. [Overview & Operator Set](#1-overview--operator-set)
2. [Operator Schema](#2-operator-schema)
3. [Quantization Modes](#3-quantization-modes)
4. [Architecture Dispatch & Kernel Paths](#4-architecture-dispatch--kernel-paths)
5. [PrePack Transformations](#5-prepack-transformations)
6. [Weight Formats](#6-weight-formats)
7. [Cross-Architecture Packing Compatibility](#7-cross-architecture-packing-compatibility)
8. [SwiGLU Fusion](#8-swiglu-fusion)
9. [FP4 (MXFP4) Details](#9-fp4-mxfp4-details)
  - [9.9 Runtime environment variables](#99-runtime-environment-variables)
  - [9.10 Interleaved GEMV layout + dtype-conditional accumulation](#910-interleaved-gemv-layout--dtype-conditional-accumulation)
  - [9.11 Single-copy SM80 weights (prefill + decode share one buffer)](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer)
  - [9.12 Fixed-shape SM90 DeepGEMM path](#912-fixed-shape-sm90-deepgemm-path)
9b. [NVFP4 (W4A16, block-16) Details](#9b-nvfp4-w4a16-block-16-details)
  - [9b.2 Fused GEMV decode (group-16)](#9b2-fused-gemv-decode-group-16)
  - [9b.3 Fast E2M1 → half/bf16 decode](#9b3-fast-e2m1--halfbf16-decode)
  - [9b.5 Native prefill accuracy & routing](#9b5-native-prefill-accuracy--routing)
10. [FP8 (W8A16) Details](#10-fp8-w8a16-details)
11. [WFP4AFP8 Details](#11-wfp4afp8-details)
12. [Future / Deferred Modes](#12-future--deferred-modes)
  - [12.1 MoE GEMV Optimization Summary](#121-moe-gemv-optimization-summary)
13. [Testing](#13-testing)
14. [Build Configuration](#14-build-configuration)
  - [14.1 MSVC and TMA grouped MoE GEMM](#141-msvc-and-tma-grouped-moe-gemm)
  - [14.2 LLM object library and SM120 native SASS](#142-llm-object-library-and-sm120-native-sass)
15. [Limitations & Known Issues](#15-limitations--known-issues)
16. [Differences vs. TensorRT-LLM](#16-differences-vs-tensorrt-llm)

---

## 1. Overview & Operator Set

Two contrib ops are registered in the `com.microsoft` domain:

| Operator | Purpose | Source |
|----------|---------|--------|
| `MoE` | Standard (non-quantized) Mixture-of-Experts. FP16/BF16/FP32 weights. | [onnxruntime/contrib_ops/cuda/moe/moe.cc](onnxruntime/contrib_ops/cuda/moe/moe.cc) |
| `QMoE` | Quantized Mixture-of-Experts. INT4/INT8/FP8/MXFP4 weights, FP16/BF16/FP8 activations. | [onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc) |

Both ops share the same CUTLASS-based runner ([CutlassMoeFCRunner](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_kernels.h)), routing engine,
and sort/permute infrastructure. They differ only in how the weight tensors are
interpreted.

The execution pipeline is:

```
input tokens → router (top-k softmax) → permute by expert
  → GEMM1 (per-expert) → activation (SiLU/GeLU/ReLU/SwiGLU)
  → GEMM2 (per-expert) → un-permute → weighted sum
```

---

## 2. QMoE Operator Schema

### 2.1 Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 1 | Top-K experts selected per token. |
| `activation_type` | string | `"relu"` | One of `"relu"`, `"gelu"`, `"silu"`, `"swiglu"`, `"identity"`. These are the only values accepted end-to-end (attribute parsing); other kernel-internal types are not reachable from ONNX. |
| `normalize_routing_weights` | int | 0 | Re-normalize the top-k weights to sum to 1. |
| `use_sparse_mixer` | int | 0 | Enable sparse-mixer routing variant. |
| `swiglu_fusion` | int | 0 | 0=no fusion, 1=interleaved (Gate/Value), 2=block (Gate;Value). See [§8](#8-swiglu-fusion). |
| `activation_alpha` | float | `1.0` | SwiGLU alpha. Default `1.0` (Standard SwiGLU); GPT-OSS uses `1.702`. |
| `activation_beta` | float | `0.0` | SwiGLU beta. Default `0.0` (Standard SwiGLU); GPT-OSS uses `1.0`. |
| `swiglu_limit` | float | unset (`+inf`) | SwiGLU clamp limit. Unset means no clamp (Standard SwiGLU); GPT-OSS uses `7.0`. |
| `expert_weight_bits` (QMoE only) | int | 4 | 4 (INT4/MXFP4) or 8 (INT8/FP8). |
| `block_size` (QMoE only) | int | -1 | Group size for INT4/INT8 group-wise quantization. -1 = per-output-channel. FP4/WFP4AFP8 normalize an omitted value to 32 and require 32 when present; NVFP4 normalizes an omitted value to 16 and requires 16 when present. FP8 uses square blocks for positive values, per-expert globals otherwise. |
| `quant_type` (QMoE only) | string | `"int"` | `"int"`, `"fp4"`, `"nvfp4"`, `"fp8"`, `"wfp4afp8"`. See [§3](#3-quantization-modes). |
| `weights_prepacked` (QMoE only) | int | -1 | Layout selector for integer quantization and CUDA NVFP4. Integer semantics are unchanged: `-1` and `1` select EP-determined prepacked layouts (both SM80 `fpA_intB` on CUDA today); `0` selects raw `[E, N, K/pack]` weights transformed in `PrePack()`. For CUDA NVFP4, omitted/`-1` and `0` select legacy N-packed bytes; `1` selects offline K-packed row-major bytes. Mode `2` is unsupported for NVFP4. See [§5.1](#51-weights-input-2--5--8). |

### 2.2 Type Constraints

| Constraint | Allowed Types | Used By |
|------------|---------------|---------|
| `T`  | `float`, `float16`, `bfloat16` | input, output, biases, router |
| `T1` | `uint8`, `float8e4m3fn` | quantized weights and zero points: INT4/INT8/FP4/NVFP4 weights use `uint8`; FP8 weights use `float8e4m3fn` |
| `T2` | `float`, `float16`, `bfloat16`, `uint8`, `float8e4m3fn` | INT4/INT8 weight scales use floating-point tensors; MXFP block scales use `uint8` (`float8e8m0`) storage; **NVFP4 block scales use `float8e4m3fn`** |
| `T4` | `float` | per-expert global scales, FP8 activation scales |

### 2.3 Inputs

The schema is unified across all `quant_type` values. Inputs that are not relevant
to the selected `quant_type` are simply omitted (most are `Optional`).

| Idx | Name | Type | Shape | Used by `quant_type` |
|----:|------|------|-------|----------------------|
| 0 | `input` | T | packed `(total_tokens, hidden_size)` or padded `(batch, sequence, hidden_size)` | all |
| 1 | `router_probs` | T | `(total_tokens, num_experts)` | all |
| 2 | `fc1_experts_weights` | T1 | `(E, fusion×inter, hidden/pack)` | all |
| 3 | `fc1_scales` | T2 (Opt) | varies — see [§2.4](#24-input-369-interpretation-by-quant_type) | int, fp4, nvfp4, wfp4afp8, block-scaled fp8 |
| 4 | `fc1_experts_bias` | T (Opt) | `(E, fusion×inter)` | optional |
| 5 | `fc2_experts_weights` | T1 | `(E, hidden, inter/pack)` | all |
| 6 | `fc2_scales` | T2 (Opt) | varies | int, fp4, nvfp4, wfp4afp8, block-scaled fp8 |
| 7 | `fc2_experts_bias` | T (Opt) | `(E, hidden)` | optional |
| 8 | `fc3_experts_weights` | T1 (Opt) | `(E, inter, hidden/pack)` | optional (SwiGLU split-weight) |
| 9 | `fc3_scales` | T2 (Opt) | varies | optional |
| 10 | `fc3_experts_bias` | T (Opt) | `(E, inter)` | optional |
| 11 | `fc1_zero_points` | T1 (Opt) | matches `fc1_scales` | int only |
| 12 | `fc2_zero_points` | T1 (Opt) | matches `fc2_scales` | int only |
| 13 | `fc3_zero_points` | T1 (Opt) | matches `fc3_scales` | optional, int only |
| 14 | `router_weights` | T (Opt) | `(num_tokens, num_experts)` | optional (DeepSeek noaux_tc) |
| 15 | `fc1_global_scale` | T4 (Opt) | `(num_experts,)` | fp4, nvfp4, fp8, wfp4afp8 |
| 16 | `fc2_global_scale` | T4 (Opt) | `(num_experts,)` | fp4, nvfp4, fp8, wfp4afp8 |
| 17 | `fc1_act_scale` | T4 (Opt) | `(1,)` or `(num_experts,)` | wfp4afp8 (Variant A) |
| 18 | `fc2_act_scale` | T4 (Opt) | `(1,)` or `(num_experts,)` | wfp4afp8 (Variant A) |
| 19 | `fc1_act_block_scale` | T2 (Opt, float8e8m0) | `(E, M_pad, K/32)` | wfp4afp8 (Variant B) |
| 20 | `fc2_act_block_scale` | T2 (Opt, float8e8m0) | `(E, M_pad, inter/32)` | wfp4afp8 (Variant B) |

`E = num_experts`. `pack = 8 / expert_weight_bits` for INT/MXFP4 weights; `pack = 1`
for FP8 weights. `fusion = 2` for `swiglu_fusion=1`, otherwise `1`.

> **CUDA EP restriction for `quant_type="int"`:** `fc1_scales` and `fc2_scales`
> must use the same FP16 or BF16 type as the activation input. Although the
> operator schema also permits float scales, the CUDA INT kernel does not
> currently cast float scales internally. `fusion = 2` for `swiglu_fusion=1`, otherwise `1`.

For packed input, tokens from variable-length sequences may be concatenated
without padding. MoE routing is token-local, so no sequence-offset input is
required, and the output preserves `(total_tokens, hidden_size)`. Padded 3D
input is flattened internally to the same token-major execution path. This
behavior is shared by `MoE` and `QMoE` on the CPU and CUDA execution providers.

`router_weights` (input 14) enables DeepSeek-style routing where `router_probs`
is used only for top-K selection and `router_weights` provides the mixing
weights gathered at the selected expert indices. When omitted, `router_probs`
is used for both (backward compatible).

### 2.4 Input 3/6/9 interpretation by `quant_type`

| `quant_type` | dtype | Shape | Semantics |
|--------------|-------|-------|-----------|
| `"int"` (group-wise) | float / fp16 / bf16 | `(E, N, K/block_size)` | `w_float = w_int × scale (+ zero)` |
| `"int"` (per-channel) | float / fp16 / bf16 | `(E, N)` | per-output-channel scale |
| `"fp4"` | uint8 (`float_ue8m0_t`) | `(E, N, K/32)` | MXFP4 block scale, group=32 |
| `"nvfp4"` | uint8 (`float8e4m3fn` bytes) | `(E, N, K/16)` | NVFP4 block scale, group=16 (needs `fc*_global_scale`) |
| `"fp8"` (`block_size<=0`) | — | — | not used; per-expert global scales are inputs 15/16 |
| `"fp8"` (`block_size=B>0`) | float32/float16/bfloat16 | `(E, ceil(N/B), ceil(K/B))` | square `B×B` dequant scales; global scales omitted |
| `"wfp4afp8"` | uint8 (`float_ue8m0_t`) | `(E, N, K/32)` | MXFP4 block scale, group=32 |

Inputs 11/12/13 (`fc*_zero_points`) are valid only for `"int"`. FP8 e4m3 and
FP4 e2m1 (both MXFP4 and NVFP4) are symmetric formats with no zero-point.

---

## 3. Quantization Modes

| `quant_type` | Notation | Activation | Weight | Native SM | Fallback | Build gate |
|--------------|----------|-----------|--------|-----------|----------|------------|
| `"int"` (4-bit) | W4A16 | FP16/BF16 | INT4 group-wise | SM75+ (Ampere GemmGrouped) | — | always |
| `"int"` (8-bit) | W8A16 | FP16/BF16 | INT8 group-wise | SM75+ | — | always |
| `"fp8"` | W8A16-fp8 | BF16/FP16 | FP8 e4m3 (no packing) | **SM90+** native (globals only) | dequant→A16 on SM<90 or with block scales | `ENABLE_FP8` (CUDA ≥ 11.8) |
| `"fp4"` | W4A16-MXFP4 | BF16/FP16 | MXFP4 e2m1, group=32 | **SM80+** fused-dequant grouped GEMM prefill + fused GEMV decode (including SM120/SM121) | dequant→A16 on SM<80 or with `ORT_FP4_SM80_GEMM=0` | `ENABLE_FP4` + `USE_FP4_QMOE` (CUDA ≥ 12.8) |
| `"nvfp4"` | W4A16-NVFP4 | BF16/FP16 | NVFP4 e2m1, group=16, `float8e4m3fn` block scale + per-expert FP32 global scale | **SM120/SM121** native (block-scaled FP4×FP4 prefill) + **fused GEMV decode** | dequant→A16 on other SMs | `ENABLE_FP4` + `USE_FP4_QMOE` (CUDA ≥ 12.8) |
| `"wfp4afp8"` | W4A8-MXFP4×FP8 | FP8 e4m3 (quantized in-runner) | MXFP4 e2m1, group=32 | **SM100+** native | dequant→A16 on SM<100 | `ENABLE_FP4` + `USE_FP4_QMOE` + `ENABLE_FP8` |

Selection logic (see [moe_quantization.cc](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc)):

```cpp
if (quant_type_ == "fp4")      use_fp4_dequant_fallback_      = !enable_fp4_cutlass_gemm_;  // SM90-only opt-in native TMA WS; SM80+ fallback regime uses the SM80 grouped GEMM + fused GEMV
if (quant_type_ == "nvfp4")    use_fp4_dequant_fallback_      = !enable_nvfp4_cutlass_gemm_;  // native SM120+ block-scaled FP4xFP4 prefill (ORT_ENABLE_NVFP4_CUTLASS_GEMM, shape-gated); fused GEMV decode still covers small-decode shapes
if (quant_type_ == "wfp4afp8") use_wfp4afp8_dequant_fallback_ = (sm_ < 100);
if (quant_type_ == "fp8")      use_fp8_dequant_fallback_      = (sm_ < 90 || block_size_ > 0);
```

`expert_weight_bits` validation:
- `int` → 4 or 8
- `fp4`, `nvfp4`, `wfp4afp8` → must be 4
- `fp8` → must be 8

When the build is configured without the corresponding flags, `quant_type`
values that require them are rejected at construction time:

```cpp
#if !defined(ENABLE_FP4) || !defined(USE_FP4_QMOE)
  ORT_ENFORCE(quant_type_ != "fp4",
              "QMoE quant_type='fp4' requires USE_FP4_QMOE with CUDA 12.8 or newer.");
  ORT_ENFORCE(quant_type_ != "wfp4afp8", ...);
#endif
#if !defined(ENABLE_FP8)
  ORT_ENFORCE(quant_type_ != "wfp4afp8", "...");
#endif
```

---

## 4. Architecture Dispatch & Kernel Paths

The runner selects between three CUTLASS kernel families and one small-row GEMV
fast path at runtime. The choice is
made by `CutlassMoeFCRunner::supportsTmaWarpSpecialized()` and the dispatch headers
under [onnxruntime/contrib_ops/cuda/llm/moe_gemm/](onnxruntime/contrib_ops/cuda/llm/moe_gemm/).

| Path | CUTLASS class | Used for | SM range |
|------|---------------|----------|----------|
| **MoE GEMV fast path** | `fpA_intB_gemv`-based custom kernel | INT4/INT8 per-column W*A16 and symmetric INT4/INT8 block-wise W*A16, and **MXFP4 (group-32) / NVFP4 (group-16) W4A16** decode, with FP16 or BF16 activations and true decode row counts | SM80+ |
| **Ampere GemmGrouped** | `cutlass::gemm::kernel::GemmGrouped` | INT4/INT8 W*A16, MXFP4 W4A16 (fused dequant), FP8 W8A16 dequant fallback, FP32 | SM75–SM89, plus all mixed-input on SM90+ |
| **TMA Warp-Specialized (mixed-input)** | `CollectiveBuilderMixedInput` | Same-type FP16×FP16 / BF16×BF16, opt-in native MXFP4 W4A16 | SM90 only (sm_90a WGMMA) |
| **Block-Scaled Tensor Op** | `OpClassBlockScaledTensorOp` | Native FP8×MXFP4 (`wfp4afp8`) | SM100+ (Blackwell) |
| **QMoE FP4 DeepGEMM** | DeepGEMM `sm90_fp8_gemm_1d2d_impl`, `MGroupedMasked` | Opt-in, fixed-shape MXFP4→E4M3 small-token path with BF16 output | H200 only (SM90, 132 SMs, at least 120 GiB) |

The MoE GEMV fast path is selected before the Ampere grouped GEMM for integer
QMoE when all of the following are true:

- activation/output dtype is FP16 or BF16;
- scales and biases use the same dtype as the activation;
- weights are INT4 or INT8 for per-column scales, or INT4/INT8 for symmetric block-wise scales;
- `block_size <= 0` for per-column INT4/INT8, or `block_size` is 32, 64, or 128 for block-wise INT4/INT8;
- block-wise GEMV is symmetric only; if zero-point compensation is present, dispatch falls back to grouped GEMM;
- `expanded_num_rows = num_tokens * top_k` is in `(0, 8]`;
- `N >= 512` and `K >= 512`;
- if `expanded_num_rows > 4`, the logical MoE intermediate size is at least 512;
- `N` is divisible by the column-interleaved tile width (32 for INT4, 16 for INT8),
  and `K` satisfies the kernel step and, for block-wise scales, complete-block alignment.

Asymmetric block-wise quantization, broader row counts, and dimensions
outside the profiled gate stay on grouped GEMM until profile data shows an
end-to-end GEMV win. FP16 and BF16 share the same dispatch gate and custom
kernels; for a given shape, BF16 routes to GEMV exactly where FP16 does and
shows comparable latency.

Set `ORT_DISABLE_MOE_GEMV=1` before process start to force the grouped GEMM
fallback for debugging, benchmarking, or bisecting numerical differences. The
switch is cached on first use.

### 4.1 Per-mode dispatch matrix

| Mode | SM75-89 (Ampere/Ada) | SM90 (Hopper) | SM100 (Blackwell) | SM120 (RTX 5090) |
|------|----------------------|---------------|-------------------|------------------|
| INT4/INT8 W*A16 | Ampere GemmGrouped | Ampere GemmGrouped (TMA WS rejects mixed-type INT) | Ampere GemmGrouped | Ampere GemmGrouped |
| FP16/BF16 (no quant, MoE op) | Ampere GemmGrouped | TMA WS (same-type) | TMA WS / valid Blackwell spec | TMA WS / Ampere fallback |
| FP8 W8A16 native | dequant fallback | TMA WS | TMA WS | SM89 FP8 kernel redirect |
| FP4 W4A16 | SM80+: Ampere GemmGrouped (fused dequant) prefill + fused GEMV decode | same; optional fixed-shape QMoE FP8 DeepGEMM; opt-in TMA WS mixed-input FP4 | Ampere GemmGrouped + fused GEMV | Ampere GemmGrouped + fused GEMV |
| NVFP4 W4A16 (group-16) | dequant fallback + fused GEMV decode | dequant fallback + fused GEMV decode | dequant fallback + fused GEMV decode | TMA WS block-scaled FP4×FP4 prefill + fused GEMV decode |
| WFP4AFP8 native | dequant fallback | dequant fallback | Block-scaled tensor op | Block-scaled tensor op |
| FP32 | Ampere GemmGrouped (forced) | same | same | same |

### 4.2 Minimum dimension constraint (`min_dim`)

- Both `hidden_size` and `inter_size` must be ≥ 16.
- TMA WS path: smallest tile is 128×16×128B (N=16 for FP16). K residues handled by TMA.
- Ampere GemmGrouped path: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
- Alignment to 128 bits is enforced separately (e.g., dimensions must be multiples of 8 for FP16).

### 4.3 Dequant-to-A16 fallback

When the requested native path is not available on the running GPU, the QMoE op
decodes the quantized weights into FP16/BF16 once and feeds them to the dense
A16 runner. Helper kernels:

- `LaunchQMoEDequantizeFp4Weights` — MXFP4 → FP16/BF16
- `LaunchQMoEDequantizeFp8Weights` — FP8 e4m3 → FP16/BF16

The decoded buffers are owned by the QMoE op for the lifetime of the session.

> **MXFP4 exception.** With `ORT_FP4_SM80_GEMM=1` (default on SM80+) both prefill and
> decode are served by pre-packed e2m1 buffers, so `LaunchQMoEDequantizeFp4Weights` is never
> reached and the raw initializers are released in `PrePack`
> ([§9.11](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer)). Set
> `ORT_FP4_SM80_GEMM=0` to restore the dequant fallback.

### 4.4 Target hardware (developer matrix)

RTX 3090 (SM86), RTX 4090 (SM89), H200 (SM90), GB200/B200 (SM100), RTX 5090 (SM120).

---

## 5. PrePack Transformations

`QMoE::PrePack` ([moe_quantization.cc](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc))
copies constant inputs to GPU once and, for INT4/INT8, derives a pre-scaled bias
from the zero points so the kernel can apply asymmetric quantization with no
extra subtraction.

### 5.1 Weights (input 2 / 5 / 8)

**INT4/INT8** weight layout is controlled by the `weights_prepacked` attribute
([§2.1](#21-attributes)). The prepacked layouts selected by `-1` and `1` are
determined by the execution provider:

- **`weights_prepacked=-1` (default)** — the `fc1`/`fc2` weights are already in
  the EP's default prepacked layout (e.g. packed offline by
  `pack_weights_for_cuda_mixed_gemm` for the CUDA EP). They are copied to GPU
  and consumed as-is.
- **`weights_prepacked=1`** — the `fc1`/`fc2` weights are already in the EP's
  **SM90** (Hopper) prepacked layout (reserved; see the note below).
- **`weights_prepacked=0`** — the `fc1`/`fc2` weights are raw, schema-conformant
  `[E, N, K/pack]` tensors as produced by `quantize_matmul_{4,8}bits`. `PrePack`
  runs the CUTLASS layout transform itself via `PrePackIntExpertWeights`,
  removing the offline pre-pack dependency. This makes integer QMoE symmetric
  with `MatMulNBits::PrePack_B`.

> **Single layout on the CUDA EP.** The CUDA EP INT4/INT8 MoE GEMM always
> dispatches to the Ampere (**SM80**) grouped-GEMM kernel — even on SM90 —
> because mixed int-weight + fp16/bf16 activation is not a valid Hopper TMA
> warp-specialized specialisation (`isValidHopperMOESpecialisation` is `false`).
> This matches **TensorRT-LLM**, which likewise routes `W4A16`/`W8A16` MoE to the
> SM80 kernel on Hopper; its Hopper TMA-WS mixed-dtype MoE kernel is reserved for
> `W4A8` (FP8 activation) and `WFP4A16` (FP4 weight). Consequently the CUDA EP
> consumes the **SM80 `fpA_intB` layout on every GPU**, `PrePack` always packs
> for SM80, and `weights_prepacked=-1` and `=1` are equivalent today. `1` is
> accepted and reserved for a possible future Hopper-specific layout (e.g.
> `W4A8`). There is therefore no architecture-match constraint: SM80-format
> weights run correctly on SM90 via the SM80 kernel.

`PrePackIntExpertWeights` loops over the `E` experts and, per expert, applies the
same transpose + row-permutation / column-interleave / bias / pair-interleave
transform as `pack_weights_for_cuda_mixed_gemm` (see [§6.1](#61-int4-group-wise-quant_typeint-expert_weight_bits4)),
always targeting the SM80 layout. SM75+ is required. The source
`[E, N, K/pack]` initializers are released after their shapes are cached
(`fc1_weights_shape_` / `fc2_weights_shape_`), so peak weight memory stays ~1×.
The prepacked GPU buffers (`packed_fc1_weights_` / `packed_fc2_weights_`) are then
preferred by `ComputeInternal`. If prepacking is disabled at the session level
(`session.disable_prepacking`), the buffers stay null and the raw initializer
pointers are read at compute time instead.

> **Note**: `weights_prepacked=0` is the only path that triggers an in-`PrePack`
> layout transform for INT weights. FP4 / FP8 / WFP4AFP8 weight handling is
> unaffected.

**CUDA NVFP4** uses a separate `weights_prepacked` contract:

| `weights_prepacked` | Physical packed-byte storage |
|---------------------|------------------------------|
| Omitted / `-1` | Legacy N-packed `[E, K, N/2]` |
| `0` | Explicit legacy N-packed `[E, K, N/2]` |
| `1` | Offline K-packed row-major `[E, N, K/2]` |

Logical ONNX dimensions remain `[E, K, N/2]` in every mode. Mode `1` changes
only the physical weight-byte layout: E2M1 codes, raw E4M3 block scales
`[E, N, K/16]`, and FP32 global scales `[E]` retain their values. Mode `2` is
unsupported. Integer-quantization semantics are unchanged.

Mode `1` shares the offline payload across eligible native GEMV decode,
Blackwell native GEMM prefill, and dense dequantization fallback without a
decode-only weight repack or expanded GEMV scale bank. Blackwell native GEMM
still derives its TMA-swizzled block scales; the dense fallback reads the
retained raw scale inputs, not that swizzled buffer. Disabling prepacking with
`session.disable_prepacking` preserves correct offline-layout dequantization
and eligible GEMV execution; native prefill requires prepacking.

MXFP4 weights must be packed by `pack_fp4_weights_for_cuda_moe_gemm`. FP8 weights
are stored as raw e4m3 bytes (no packing).

> **MXFP4 on SM80+.** When `ORT_FP4_SM80_GEMM` is on (the default), `PrePack`
> produces a **single** pre-packed e2m1 buffer per FC that serves both the grouped-GEMM
> prefill and the fused GEMV decode, and reports `is_packed = true` so ORT releases the
> raw `[E, K, N/2]` initializers. Peak MXFP4 weight memory is ~1×; without this it was
> ~3× (raw initializer + prefill layout + decode layout). See
> [§9.11](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer).


### 5.2 INT4/INT8 scales + zero-point → bias

The kernels use a pre-calculated additive bias to avoid the per-element
zero-point subtraction.

- **8-bit**: weights are shifted `uint8 → int8` (− 128). Bias compensates:
  ```
  bias = (128 − ZP) × scale
  ```
  Effectively computes `(W_stored + 128 − ZP) × scale = (W_orig − ZP) × scale`.
- **4-bit**: zero points are unpacked from nibbles (2 per byte) and converted to
  scaled biases:
  ```
  bias = (8 − ZP) × scale
  ```
  Equivalent to `(W − (ZP − 8)) × scale`.
- **Symmetric** (no `fc*_zero_points`): bias = 0.

Kernels: `LaunchQMoEPrePackOffsetBias`, `LaunchQMoEPrePackPacked4BitZPKernel`.
Output buffer (`packed_bias`) has the scale dtype (`float16` / `bfloat16` / `float`).

### 5.3 FP8 / FP4 / WFP4AFP8 PrePack

For floating-point quantization modes, `PrePack` simply copies constant tensors
to GPU memory:

| Input idx | Member | Used by |
|-----------|--------|---------|
| 15/16/17 | `packed_fc{1,2,3}_global_scale_` | fp4, fp8, wfp4afp8 |
| 18/19    | `packed_fc{1,2}_act_scale_`      | wfp4afp8 (Variant A) |

Block scales (inputs 3/6/9 for fp4/wfp4afp8 and 20/21 for wfp4afp8 Variant B)
that are constant initializers are also copied to GPU; otherwise they are
read directly from `context->Input` at runtime.

---

## 6. Weight Formats

This section covers the five distinct weight encodings supported by QMoE.

### 6.1 INT4 group-wise (`quant_type="int"`, `expert_weight_bits=4`)

#### Logical and packed shapes

| Tensor | Logical | Packed storage |
|--------|---------|----------------|
| FC1 weight | `[E, N, K]` (`N = fusion × inter`) | `[E, N, K/2]` bytes |
| FC1 scales | — | `[E, N, K/group_size]` (T2) |
| FC1 zero-points (asymmetric) | — | `[E, N, K/group_size/2]` packed (T1) |

INT4 packing layout within a byte: `[high_nibble | low_nibble] = [elt_1 | elt_0]`.
Each INT4 element is in `[-8, 7]` (signed) before bias, `[0, 15]` after the +8 bias.

#### Preprocessing pipeline (offline `pack_weights_for_cuda_mixed_gemm`, or in-`PrePack` via `PrePackIntExpertWeights`)

This is the layout transform applied either offline by
`pack_weights_for_cuda_mixed_gemm`, or per-expert inside `PrePack` when
`weights_prepacked=0` (see [§5.1](#51-weights-input-2--5--8)).


INT MoE/QMoE CUDA kernels, including the small-row MoE GEMV path, consume the
SM80 `ColumnMajorTileInterleave<64, 4>` layout. Pack INT4/INT8 MoE weights with
the SM80 target layout even when the runtime GPU is Hopper or newer.

1. **Input layout**: `[N, K]` per expert (Out × In), 2 elements per byte for INT4.
2. **Transpose & signed conversion**:
   - Unpack `uint4 [0, 15]` → subtract 8 → `int8 [-8, 7]`.
   - Transpose row-major `[K, N]` → column-major `[N, K]` with nibble-level swaps.
3. **Row permutation (LDSM)** for SM75+ tensor cores. INT4 uses a 32-row pattern:
   ```
   {0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
    4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31}
   ```
   (`kPerm_W4_A16` in [fpA_intB_gemm_preprocessors_impl.h](onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.h)).
4. **Column interleaving**: `ColumnMajorTileInterleave<64, 4>` on Ampere/Ada/Blackwell.
   `RowsPerTile=64` (K direction), `ColumnsInterleaved=4` (N direction).
5. **Bias addition + register interleaving**: add 128 (so storage is `uint8`),
   reorder elements within a 32-bit word from `[7,6,5,4,3,2,1,0]` to
   `[7,5,3,1,6,4,2,0]` to minimize shift/mask cost in the kernel
   (`add_bias_and_interleave_int8s_inplace_kernel`).

#### Dequantization

```cpp
// Symmetric (no zero-point):  W_stored is uint8 in [0, 15]
float W = (float)(W_stored - 8) * scale;

// Asymmetric (with zero tensor):
float W = (float)W_stored * scale + zero;     // zero is the scaled bias from PrePack
```

### 6.2 INT8 group-wise (`quant_type="int"`, `expert_weight_bits=8`)

| Tensor | Logical | Packed storage |
|--------|---------|----------------|
| FC1 weight | `[E, N, K]` | `[E, N, K]` (no packing) |
| FC1 scales | — | `[E, N, K/group_size]` (T2) |

Preprocessing pipeline differences from INT4:

- **Row permutation**: 16-row pattern `{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}`
  (`kPerm_W8_A16`).
- **Column interleaving**: `ColumnMajorTileInterleave<64, 2>` (`RowsPerTile=64`,
  `ColumnsInterleaved=2`).
- **Bias**: +128 shift (signed `[-128,127]` → unsigned `[0,255]`).
- **Register interleaving**: `[3, 1, 2, 0]` per 32-bit word.

Dequantization (symmetric): `W = (W_stored - 128) * scale`.

### 6.3 INT4 vs INT8 comparison

| Aspect | INT4 | INT8 |
|--------|------|------|
| Elements per byte | 2 | 1 |
| Elements per int32 word | 8 | 4 |
| Value range (signed) | `[-8, 7]` | `[-128, 127]` |
| Bias offset | +8 | +128 |
| Row permutation size | 32 rows | 16 rows |
| Packed shape | `[E, N, K/2]` | `[E, N, K]` |
| Column interleave | `<64, 4>` | `<64, 2>` |

### 6.4 FP8 e4m3 (`quant_type="fp8"`)

- **Storage**: `[E, N, K]` `float8e4m3fn` (`Float8E4M3FN` in ORT; `__nv_fp8_e4m3` in CUDA), 1 byte per value.
- **Packing**: `pack_size = 1` — no offline packing required.
- **Scales**: nonpositive `block_size` uses per-expert float32 globals (inputs 15/16,
  shape `(E,)`), with inputs 3/6/9 omitted. Positive `block_size=B` instead uses
  square-block float32/float16/bfloat16 scales `(E,ceil(N/B),ceil(K/B))` in inputs 3/6/9 and omits globals.
- **Zero-points**: not applicable (FP8 is symmetric); inputs 11/12/13 must be absent.
- **Dequantization**: legacy globals are applied in the native GEMM epilogue.
  Square-block scales are applied by the dense fallback before GEMM; see §10.3.

### 6.5 MXFP4 e2m1 (`quant_type="fp4"` and `"wfp4afp8"`)

- **Storage**: `[E, N, K/2]` `uint8`, reinterpreted as `__nv_fp4_e2m1` (2 values per byte).
- **Packer**: `pack_fp4_weights_for_cuda_moe_gemm` (Python binding in
  [onnxruntime_pybind_quant.cc](onnxruntime/python/onnxruntime_pybind_quant.cc)).
  No Ampere-style row permutation or column interleaving — SM90+ TMA-based FP4
  kernels expect a simpler column-major packed layout:
  1. Input: `[N, K/2]` FP4 (2 per byte along K, row-major per expert)
  2. Nibble-level transpose `[N, K]` → `[K, N]`
  3. Output: `[K, N/2]` bytes (2 per byte along N, column-major packed)
- **Block scales**: `fc1_scales` (input 3) — `(E, N, K/32)` `uint8` storage,
  semantically `float_ue8m0_t` (8-bit power-of-2 exponent).
- **Global scale**: `fc1_global_scale` (input 15) — `(E,)` float32.
- **Dequantization** (applied during the GEMM in registers):
  `W_float ≈ fp4_to_float(W_fp4) × ue8m0_to_float(block_scale) × global_scale`.

### 6.6 Supported INT group sizes

| Architecture | Activation | Supported `block_size` |
|--------------|-----------|------------------------|
| SM75–89 (Turing/Ampere/Ada) | FP16/BF16 | 32, 64, 128 |
| SM90 (Hopper) | FP16/BF16 | falls back to Ampere — 32, 64, 128 |
| SM100/120 (Blackwell) | FP16/BF16 | falls back to Ampere — 32, 64, 128 |

For MXFP4, the block size is fixed at **32** by the format.

### 6.7 INT2, INT3, and Mixed-Width Integer QMoE (`quant_type="int"`)

`com.microsoft::QMoE` supports projection-specific integer bit-widths via `expert_weight_bits`,
`fc1_expert_weight_bits`, `fc2_expert_weight_bits`, and `fc3_expert_weight_bits` in `{2, 3, 4, 8}`:

- **INT3 and mixed-width INT3 projections (`(3, 3)`, `(3, 4)`, `(4, 3)`, `(2, 3)`)**:
  - Uses canonical row-local LSB-first unsigned codes (`[E, N, ceil(K * bits / 8)]`, zero-point offset `4` for `INT3`) with `weights_prepacked = 0` and `block_size` in `{32, 64, 128}`.
  - Executed directly in-place on CUDA via `LaunchInt3Moe` (`FusedFc1FastKernel` with aligned `uint32_t` `__ldg` + `__funnelshift_r` 8-element unpacking and shared-memory input staging, `Fc2RouteFastKernel` with parallel-route execution, and `FinalizeRoutesFastKernel`).
- **Offline SM80-prepacked INT2 and mixed INT2/INT4 (`(2, 2)`, `(2, 4)`, `(4, 2)`)**:
  - Supports both `weights_prepacked = 0` (runtime `PrePackIntExpertWeights`) and `weights_prepacked = 1` (offline SM80 `fpA_intB` `W2_A16` / `W4_A16` prepacked `[E, K, N / (8 / bits)]` tensors) across `block_size` in `{32, 64, 128}`, `num_experts <= 512`, and `k` in `{1, 2, 4, 6, 8, 10}`.
  - Decode ($M \cdot k \le 64$) dispatches to `launch_moe_gemv_int_symmetric_interleaved_swiglu` and `launch_moe_gemv_int_symmetric` (including 2-scale `GroupSize == 32` dequantization for `Int2DetailsW`), while prefill dispatches to `RunInt2MoePrefill` (SM80 CUTLASS Tensor Core grouped GEMM) using a single GPU weight copy.

---

## 7. Cross-Architecture Packing Compatibility

Weight packing is architecture-aware. The following table summarizes which packed
weights are interchangeable across SMs:

| Target SM | Compatible packed weights from… | Notes |
|-----------|--------------------------------|-------|
| SM70 (Volta) | — | Not supported (no INT8 LDSM). |
| SM75 (Turing) | SM75/80/86/89/100/120 | LDSM permutation + column interleaving. |
| SM80 (Ampere) | SM75/80/86/89/100/120 | Same. |
| SM86/89 (Ada/Lovelace) | SM75/80/86/89/100/120 | Same. |
| SM90 (Hopper) | **SM90 only** | Hopper skips column interleaving (uses Permuted-Linear). |
| SM100/120 (Blackwell) | SM75/80/86/89/100/120 | Falls back to SM80 packing for INT4/INT8. |

**Summary groups**

- **Group A (universal INT4/INT8)**: SM75, SM80, SM86, SM89, SM100, SM120.
- **Group B (Hopper INT4/INT8)**: SM90 only.
- **MXFP4**: separate format ([§6.5](#65-mxfp4-e2m1-quant_typefp4-and-wfp4afp8))
  — does not use `pack_weights_for_cuda_mixed_gemm`.
- **FP8**: no packing.

> **QMoE uses Group A on every GPU.** The table above describes the layouts the
> `pack_weights_for_cuda_mixed_gemm` *preprocessor* can emit. The QMoE INT4/INT8
> MoE GEMM, however, always dispatches to the Ampere (SM80) grouped-GEMM kernel —
> even on SM90 — because mixed int-weight + fp16/bf16 activation is not a valid
> Hopper TMA warp-specialized specialisation (the same is true in TensorRT-LLM).
> It therefore consumes the **Group A (SM80) layout on all architectures,
> including Hopper**. For QMoE, always pack INT4/INT8 weights for SM80 (`arch=80`),
> and `PrePackIntExpertWeights` (`weights_prepacked=0`) does exactly that
> regardless of the runtime device SM. Group B (SM90) layout is currently unused
> by QMoE.

---

## 8. SwiGLU Fusion

SwiGLU formula:

```
SwiGLU(x) = G × Sigmoid(alpha × G) × (L + beta)
    G = clamp(Gate,  max=limit)
    L = clamp(Value, min=-limit, max=limit)
```

`Gate` and `Value` are the two halves of the FC1 output. The behavior is controlled
by `activation_alpha` (alpha), `activation_beta` (beta) and `swiglu_limit` (limit).

| Parameter | Attribute | Default | Standard SwiGLU | GPT-OSS SwiGLU |
|-----------|-----------|---------|-----------------|----------------|
| alpha | `activation_alpha` | `1.0` | `1.0` | `1.702` |
| beta | `activation_beta` | `0.0` | `0.0` | `1.0` |
| limit | `swiglu_limit` | unset → `+inf` (no clamp) | unset / `+inf` | `7.0` |

The attribute **defaults are exactly Standard SwiGLU**, which reduces to:

```
SwiGLU(x) = Gate × Sigmoid(Gate) × Value = SiLU(Gate) × Value
```

This is the activation used by Llama- and Gemma-style MoE. GPT-OSS SwiGLU instead
uses `alpha=1.702`, `beta=1.0`, `limit=7.0`. Both variants run on the same CUDA
kernel; when `alpha=1.0`, `beta=0.0` and `limit=+inf`, the kernel takes the plain
`SiLU(Gate) × Value` path (no clamping).

The operator supports three fusion modes via the `swiglu_fusion` attribute:

| `swiglu_fusion` | Inputs | FC1 layout | Notes |
|----------------:|--------|------------|-------|
| 0 | `fc1`, `fc2`, `fc3` | separate Gate / Value / Up | Conceptually three GEMMs. |
| 1 (interleaved) | `fc1`, `fc2` | `[Gate_0, Value_0, Gate_1, Value_1, …]` — `[E, 2×inter, hidden]` | GPT-OSS layout. |
| 2 (block) | `fc1`, `fc2` | `[Gate_0…Gate_N | Value_0…Value_N]` — `[E, 2×inter, hidden]` | Concatenated halves; Llama/Gemma layout. |

> **Backward-compatibility remap (`swiglu_fusion=0` + SwiGLU)**: The published gpt-oss-20b
> model before June 2025 uses **interleaved** SwiGLU layout but `swiglu_fusion` attribute to `0`.
> To keep those models working, when `activation_type="swiglu"` and `swiglu_fusion=0`,
> the CUDA op treats the FC1 weights as interleaved (i.e. as if `swiglu_fusion=1`):
> for **QMoE** without a separate `fc3` (block-scaled FP8 can supply separate FC3).
> A one-time warning is logged. Consequently a SwiGLU model that genuinely intended the
> non-interleaved split must provide a separate `fc3` (standard MoE) rather than rely on
> `swiglu_fusion=0`. New exporters should set `swiglu_fusion` explicitly.

> **CPU note**: The CPU MoE/QMoE implementation only supports the **interleaved**
> SwiGLU layout (`swiglu_fusion=1`). The concatenated layout (`swiglu_fusion=2`)
> throws `ORT_NOT_IMPLEMENTED` on CPU; use the CUDA EP for concatenated SwiGLU.

### Standard MoE runtime fc3 fusion

The non-quantized **MoE** operator (not QMoE) accepts an optional `fc3_experts_weights`
input. When present, the op allocates a temporary buffer and concatenates `fc1` (Gate)
with `fc3` (Value) per expert at runtime, simulating `swiglu_fusion=2`. This makes it
easy to feed Mixtral-style models without offline fusion.

> **Note**: This runtime fusion is **only** in standard MoE. For **QMoE**, weights
> must be fused offline before quantization+packing.

---

## 9. FP4 (MXFP4) Details

The QMoE operator supports **MXFP4** quantized weights with FP16/BF16 activations
(W4A16) via `quant_type="fp4"`. The kernel path is the mixed-input TMA
warp-specialized CUTLASS kernel.

### 9.1 Why "mixed input"?

Block-scaled tensor ops (`OpClassBlockScaledTensorOp`) require **both** operands to use
block scaling (FP4×FP4 or FP8×FP4). W4A16 has full-precision activations paired with
narrow FP4 weights — that is the mixed-input configuration. The dispatch flips on the
`use_wfp4a16` flag in [moe_gemm_kernels.h](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h):

```cpp
static constexpr bool use_wfp4a16 = weight_fp4 &&
    (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
```

### 9.2 Dispatch flow

```
CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>::dispatchToArch()
  └─ use_wfp4a16 == true
     └─ select fusion from hopper_inputs.fusion (NONE or FINALIZE)
        └─ select K tile: inputs.k % 256 == 0 → PackedScalesNum=1 (K=256)
                           else              → PackedScalesNum=2 (K=128)
           └─ sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., FUSION, PackedScalesNum>()
              └─ Ktile = PackedScalesNum==2 ? 128 : 256
                 └─ dispatch on tile_config_sm90 enum (M×N from heuristic)
                    └─ sm90_dispatch_moe_mixed_dtype_gemm_config<..., FUSION, Shape<M, N, Ktile>>()
                       └─ dispatch on cluster_shape
                          └─ sm90_dispatch_mainloop_schedules<..., FUSION>()
                             └─ sm90_generic_mixed_moe_gemm_kernelLauncher()
                                ├─ ElementA = cutlass::half_t  (activation)
                                ├─ ElementB = cutlass::float_e2m1_t  (weight, stored as FP4)
                                ├─ group_size = 32 (mxfp4_group_size)
                                ├─ ElementScale = cutlass::float_ue8m0_t
                                ├─ CollectiveBuilderMixedInput (FP4→FP16 upconvert in registers)
                                └─ Epilogue: NONE (per-expert output) or FINALIZE (fused scatter+scale)
```

Note: H100/H200 (SM90) does **not** have native FP4 tensor core instructions. The kernel uses FP4 purely
as a **compressed storage format** — weights are loaded via TMA and upconverted to FP16/BF16 in shared
memory/registers by `CollectiveBuilderMixedInput` before the actual MMA runs on FP16 tensor cores. This
is a **memory bandwidth optimization** (4x compression), not a compute throughput feature. Native FP4 MMA
is available on Blackwell (SM100+) via the separate block-scaled tensor op path (see [§11](#11-wfp4afp8-details)).

This TMA WS mixed-input kernel is compiled for `sm_90a` (WGMMA) and runs only on SM90, where it is
an opt-in debugging path (`ORT_ENABLE_FP4_CUTLASS_GEMM=1` + `ORT_ENABLE_FP4_CUTLASS_UNSAFE=1`).
SM100 and SM120/SM121 have no WGMMA and cannot load `sm_90a` SASS. On every SM80+ GPU, including
SM120/SM121, the default MXFP4 path is the SM80 fused-dequant grouped GEMM for prefill plus the
fused MXFP4 GEMV for decode ([§9.11](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer)).
With `ORT_FP4_SM80_GEMM=0`, or below SM80, MXFP4 weights are decoded via
`LaunchQMoEDequantizeFp4Weights` and fed to the dense A16 runner.

### 9.3 W4A16 vs W4A8-INT4 differences

| Property | W4A16 (MXFP4) | W4A8-INT4 |
|----------|---------------|-----------|
| `ElementA` | `half_t` / `bfloat16_t` | `float_e4m3_t` |
| `ElementB` | `float_e2m1_t` | `int4b_t` |
| Group size | 32 (MXFP4) | 128 (INT4) |
| `ElementScale` | `float_ue8m0_t` | `__nv_bfloat16` (SFA) |
| Epilogue α | 1 (no per-group α) | 0 (uses `alpha_ptr_array`) |
| Epilogue fusion | NONE or FINALIZE | NONE or FINALIZE |
| M tiles | 64, 128 | 64, 128 |
| N tiles | 16, 32, 64, 128 | 16, 32, 64, 128 |
| K tiles | 128, 256 | 128 × PackedScalesNum / sizeof(T) |
| Cluster shapes | (1,1), (2,1), (1,2), (2,2) | (1,1), (2,1), (1,2), (2,2) |
| Mainloop schedules | Pingpong, Cooperative | Pingpong, Cooperative |

### 9.4 Mainloop modification

The CUTLASS collective mainloop uses a type-dependent group size:

```cpp
static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
static constexpr int  ScalingGroupSize = IsMXFP4 ? detail::mxfp4_group_size
                                                 : detail::int4_group_size;
```

This affects `scale_k = K / ScalingGroupSize`, `NumMMAsPerChunk`, and
`NumChunksPerTileK` calculations.

### 9.5 Key data structures

```cpp
// QuantParams::FP4Inputs (moe_kernels.h)
struct FP4Inputs {
  struct GemmInputs {
    bool use_per_expert_act_scale = false;
    float const* act_global_scale = nullptr;        // nullptr for W4A16
    NVFP4ElementSF const* weight_block_scale;       // (E, N, K/32) ue8m0 bytes
    float const* global_scale;                      // (E,) float
  };
  GemmInputs fc1, fc2;
};

// Block scaling type (moe_gemm_kernels.h)
enum class FpXBlockScalingType { MXFPX /*32*/, NVFP4 /*16*/, NONE };
```

### 9.6 Constructor and ComputeInternal

```cpp
// Constructor (SM80 grouped-GEMM regime or SM90 native opt-in, ENABLE_FP4 + USE_FP4_QMOE)
m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>>(
    sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);

// ComputeInternal
quant_params = QuantParams::FP4(
    /*fc1_act_global_scale*/ nullptr,
    fc1_block_scales, fc1_global_scale,
    /*fc2_act_global_scale*/ nullptr,
    fc2_block_scales, fc2_global_scale);
```

### 9.7 Kernel instantiation files

| File | Template |
|------|----------|
| `moe_gemm/moe_gemm_kernels_fp16_fp4.cu` | `MoeGemmRunner<half, __nv_fp4_e2m1, half>` |
| `moe_gemm/moe_gemm_kernels_bf16_fp4.cu` | `MoeGemmRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>` |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_instantiation.cuh` | Instantiation macros: `ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_{PP,CO}` (NONE fusion), `ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_{PP,CO}_FINALIZE` |
| `moe_gemm/launchers/generate_moe_gemm_tma_ws_sm90_fp4.py` | Python generator: produces 320 `.generated.cu` files across FP16/BF16, M={64,128}, N={16,32,64,128}, K={128,256}, 4 cluster shapes, PP/CO schedules, NONE/FINALIZE fusion |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_*.generated.cu` | 320 generated SM90 mixed-input FP4 launcher instantiations (built when `onnxruntime_USE_FP4_QMOE=ON`) |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm120_fp4_*.generated.cu` | SM120 mixed-input FP4 launcher |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm120_fp8_fp4.generated.cu` | SM120 block-scaled FP8×FP4 launcher (WFP4AFP8) |

> **Build note**: When `onnxruntime_USE_FP4_QMOE` is OFF, the stub is also
> excluded and all `moe_gemm_kernels_*_fp4.cu` / `moe_gemm_tma_ws_sm{90,120}_fp4_*.generated.cu`
> files are filtered out. CUDA 13 PTXAS does not complete the FP4 M=128/N=64
> pingpong specializations, so those specific generated units are also excluded
> (the dispatcher routes that tile through cooperative variants instead). See
> [§14](#14-build-configuration).

### 9.8 K=128, Epilogue Fusion & Expanded Tile Configs

This subsection documents the expanded SM90 W4A16 mixed-input FP4 MoE GEMM configuration
that closes the gap with TRT-LLM.

#### Changes Summary

| Gap | Before | After |
|-----|--------|-------|
| K tiles | 256 only | {128, 256} — selected at runtime based on `inputs.k % 256` |
| Epilogue fusion | NONE only | NONE + FINALIZE — routed from `hopper_inputs.fusion` |
| N tiles accessible | Only `CtaShape128x32x128B` + `ClusterShape_1x1x1` | All instantiated tiles (N={16,32,64,128}, clusters=(1,1),(2,1),(1,2),(2,2)) |
| Generated .cu files | ~80 | 320 |
| Mainloop schedules | Pingpong only (for most tiles) | Pingpong + Cooperative (for M=128 tiles) |

#### K Tile Dispatch Mechanism

The `CutlassTileConfigSM90` enum encodes K as "128B" (128 bytes), but for FP4 mixed-input the actual K tile
in elements differs. The dispatch uses a `PackedScalesNum` encoding trick:

- `PackedScalesNum = 1` → K = 256 elements (selected when `inputs.k % 256 == 0`)
- `PackedScalesNum = 2` → K = 128 elements (selected otherwise)

Inside `sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass`:
```cpp
constexpr int Ktile = is_wfp4a16 ? (PackedScalesNum == 2 ? 128 : 256) : 128 * PackedScalesNum / sizeof(T);
```

#### Epilogue Fusion

The mixed-input launcher now supports two epilogue modes, matching the same-type launcher pattern:

- **NONE**: Per-expert intermediate output (standard grouped GEMM epilogue)
- **FINALIZE**: Fused scatter + router-scale + bias epilogue using `EpilogueMoeFusedFinalizeBuilder`

The fusion is routed at runtime in `dispatchToArch`:
```cpp
switch (hopper_inputs.fusion) {
  case EpilogueFusion::FINALIZE:
    sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., FINALIZE, PackedScalesNum>(...);
    break;
  case EpilogueFusion::NONE:
  default:
    sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., NONE, PackedScalesNum>(...);
    break;
}
```

#### Files Modified

| File | Changes |
|------|---------|
| `launchers/moe_gemm_tma_ws_mixed_input_launcher.h` | Added `EpilogueFusion FUSION` template parameter |
| `launchers/moe_gemm_tma_ws_mixed_input_launcher.inl` | Added FINALIZE epilogue support (`CollectiveEpilogueFinalize`, `make_epilogue_scalars/args` lambdas) |
| `launchers/moe_gemm_tma_ws_sm90_fp4_instantiation.cuh` | Added `_PP_FINALIZE` and `_CO_FINALIZE` macros |
| `launchers/generate_moe_gemm_tma_ws_sm90_fp4.py` | Added `k` and `fusion` fields; generates K={128,256} × NONE/FINALIZE |
| `moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` | `FUSION` param throughout; `PackedScalesNum`-based K tile; direct N tile mapping; workspace calc with `Ntile=128` |
| `moe_gemm_template_dispatch.h` | FUSION routing in `dispatchToArch`; removed restrictive wfp4a16 config filter |

### 9.9 Runtime environment variables

The FP4 path is gated by several process-start environment variables (read once in the
QMoE constructor). Defaults give the validated production behavior; the rest are opt-in or
debug switches.

| Variable | Default | Effect |
|----------|---------|--------|
| `ORT_ENABLE_FP4_GEMV` | on | Fused MXFP4 GEMV decode kernel. Set to `0` to force the dequant-to-dense fallback (debugging/bisecting). Active in the SM80+ fallback regime, and as the decode arm when native CUTLASS prefill is enabled. |
| `ORT_FP4_GEMV_AUTOTUNE` | `0` | Opt-in per-shape autotune of the GEMV CtaN/Threads tiling. Enabling it synchronizes the first uncached inference for each shape. |
| `ORT_FP4_GEMV_AUTOTUNE_LOG` | `0` | Set to `1` to log the chosen GEMV configs per shape. |
| `ORT_FP4_GEMV_DECODE_KERNEL` | `1` | NVFP4 offline K-packed (`weights_prepacked=1`) decode GEMV mode: `1` (default) enables `MoeGemvFp4DecodeKernel` in the measured SM90 decode envelope (`expanded_num_rows <= 20`, `N, K <= 2560`, and per-projection `|global_scale| * 448 < 3.98`); `0` forces `MoeGemvFp4RawKPackedKernel`; `2` opts into the full kernel envelope (`SM90+`, `expanded_num_rows <= 80`, `K <= 8192`, including the generic scale decoder). Outputs are bit-identical across all modes. Read once per process. See [§9b.2](#9b2-fused-gemv-decode-group-16). |
| `ORT_FP4_GEMV_FUSED_ROUTE` | `1` | Single-CTA fused softmax + top-k + expert-map decode routing kernel for offline K-packed NVFP4 on SM90+ (`num_rows <= 4`, `num_experts == 512`, `k in {1, 2, 4, 6, 8, 10}`). Bit-identical to `LaunchSoftmaxTopK` + `fusedBuildExpertMapsSortFirstToken`; set to `0` to disable. |
| `ORT_FP4_GEMV_PDL` | `1` | Programmatic dependent launch (`cudaLaunchAttributeProgrammaticStreamSerialization`) overlapping FC1 -> FC2 -> finalize on the offline K-packed NVFP4 decode path on SM90+. Set to `0` to disable. |
| `ORT_FP4_GEMV_FC1_PDL` | `0` | Opt-in programmatic dependent launch of FC1 behind the fused routing kernel. Off by default. |
| `ORT_FP4_GEMV_INTERLEAVED` | `0` | **Experimental, opt-in.** Routes the MXFP4 decode GEMV through the `ColumnMajorInterleaved` weight layout (`kInterleave=4`, `kStepK=32`) with dtype-conditional accumulation. fp16 gets faster decode; bf16 stays accuracy-safe. Default off keeps the shipping `ColumnMajor` path byte-for-byte unchanged. See [§9.10](#910-interleaved-gemv-layout--dtype-conditional-accumulation). |
| `ORT_FP4_GEMV_INTERLEAVED_HALFACC` | `0` | **Override.** When `ORT_FP4_GEMV_INTERLEAVED=1`, forces 16-bit accumulation for *both* fp16 and bf16, overriding the dtype-conditional policy; regresses bf16 accuracy, so it is off by default. |
| `ORT_FP4_SM80_GEMM` | `1` | Routes SM80+ (including SM120/SM121) FP4 prefill through the fused-dequant grouped GEMM. Set to `0` to force dense fallback for debugging or comparison. Decode routes through the fused MXFP4 GEMV, reading the *same* pre-packed buffer as prefill. In this regime the raw e2m1 initializers are released after `PrePack`. See [§9.11](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer). |
| `ORT_QMOE_FP4_DEEPGEMM` | `0` | **Experimental, fixed-shape H200 path.** Set to `1` to convert supported MXFP4 expert weights to persistent E4M3 buffers with FP32 block scales and run FC1/FC2 with DeepGEMM masked grouped GEMMs for `num_tokens <= 8`. Unsupported hardware, shapes, dtypes, attributes, or larger calls use the normal FP4 dispatch. See [§9.12](#912-fixed-shape-sm90-deepgemm-path). |
| `ORT_ENABLE_FP4_CUTLASS_GEMM` | `0` | Opt-in native SM90 WFP4A16 CUTLASS GEMM (fast prefill). Requires FP16, SM90, and aligned shapes (`hidden`/`inter` divisible by 256). Must be combined with `ORT_ENABLE_FP4_CUTLASS_UNSAFE=1`. |
| `ORT_ENABLE_FP4_CUTLASS_UNSAFE` | `0` | Confirms use of the experimental native SM90 path. Without it, a request to enable native GEMM logs a warning and falls back to dequant/GEMV. |
| `ORT_FP4_PREFILL_MIN_TOKENS` | `64` | When native CUTLASS is enabled, the per-node decode threshold. Tokens with `M >= threshold` (prefill) route to native CUTLASS; `M < threshold` (decode) route to the fused GEMV. Both weight/scale layouts are pre-packed so one node serves both regimes. |

When native CUTLASS is enabled, weights and scales are dual-prepacked (native layout plus
raw e8m0 block scales for GEMV), so `ORT_FP4_PREFILL_MIN_TOKENS` selects per call between the
prefill and decode kernels with no extra conversion.

### 9.10 Interleaved GEMV layout + dtype-conditional accumulation

**Opt-in (`ORT_FP4_GEMV_INTERLEAVED=1`, default off).** The shipping MXFP4 decode GEMV uses a
non-interleaved `ColumnMajor` weight layout (`kInterleave=1`, `kStepK=8`). This experimental path
instead mirrors the INT4 GEMV's `ColumnMajorInterleaved` layout (`kInterleave=4`, `kStepK=32`, 4×
fewer K-loop trips) and pairs it with a **dtype-conditional accumulator** to balance bf16 accuracy
against occupancy:

```cpp
// onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu
template <typename T>
using Fp4GemvAccT = std::conditional_t<std::is_same<T, half>::value, half, float>;
```

| Activation | Accumulator | Rationale |
|-----------|-------------|-----------|
| **fp16** | fp16 (`half`) | fp16's 10-bit mantissa tolerates 16-bit accumulation over the longer `kStepK=32` chains; the cheaper accumulator keeps register use low so the interleaved layout's K-trip savings translate into a real speedup. |
| **bf16** | fp32 (`float`) | bf16's 7-bit mantissa loses too much precision under 16-bit accumulation (fails tolerance), so it must accumulate in fp32 — at the cost of the extra registers that erase the speedup. |

The interleaved weights are produced by the `gemv_interleaved` branch of
[`PrePackRepackFP4Weights`](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc), which routes the
raw e2m1 codes per-expert through the CUTLASS fpA_intB SM80 `W4_A16` preprocessor with
`apply_bias_interleave=false` (the integer-only `+8`/pair-interleave step would corrupt the
floating-point e2m1 codes; the layout-only steps apply unchanged). Block scales are unchanged
(`kStepK=32` equals the MXFP4 block size). The shape gate requires `n % (CtaN*4)==0` and
`k % 64==0`; `CtaN`/`Threads` are pinned (`4`/`128`) so the kernel always matches the prepacked
weights.

The interleaved layout speeds up fp16 decode; bf16 gets no speedup (the fp32 register cost cancels
it) but stays accurate. The path is opt-in so fp16 deployments can take the win without affecting
bf16 or the shipping default.

### 9.11 Single-copy SM80 weights (prefill + decode share one buffer)

**Default on SM80+ (`ORT_FP4_SM80_GEMM=1`).** The SM80 grouped GEMM and the fused decode
GEMV historically needed *different* e2m1 weight layouts, so a QMoE node kept up to three
persistent copies of the expert weights:

| copy | layout | consumer |
|------|--------|----------|
| raw initializer | `[E, K, N/2]` schema layout | dequant fallback — **unreachable** in this regime |
| `gemv_fp4_fc*_weights_` | SM80 `ColumnMajorTileInterleave` + nibble pair-interleave | grouped-GEMM prefill |
| `gemv_fp4_fc*_weights_decode_` | `ColToRow` (or interleave steps 1–3) | fused GEMV decode |

For a 20B-class MXFP4 MoE each copy is ~9 GiB, which put post-load VRAM out of reach of 24 GB
consumer cards. Two changes collapse this to a single copy:

1. **Release the raw initializers.** In this regime every dispatch is served by a pre-packed
   buffer, so the dequant fallback — the only consumer of the raw layout — is unreachable.
   `PrePack` now caches `fc*_weights_shape_` for `CheckInputs` and reports `is_packed = true`.
2. **Teach the GEMV to read the SM80 layout.** The two layouts differ by exactly one
   preprocessor step: the `[e0,e2,e4,e6,e1,e3,e5,e7]` nibble pair-interleave applied by
   `interleave_int4s_inplace_kernel`. The decoder restores linear order per 32-bit word with one
   `prmt` plus a nibble swap (`cutlass::detail::fp4_e2m1x8_uninterleave`) and then runs the same
   packed table-lookup decode as the linear layout. The SM80 grouped GEMM's FP4 converter
   (`FastInterleavedAndBiasedNumericArrayConverter<*, float_e2m1_t, 8>`) uses the same helpers:

```cpp
// onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/details.h
template <typename AType, bool PairInterleaved = false>
struct Fp4I2FConverter { ... };            // PairInterleaved un-permutes the SM80 nibble order

// onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu
template <typename T>
using Fp4KernelDetailsSm80Pair =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW,
                       fiv::ColumnMajorInterleaved, /*UseInterleavedConverter=*/true,
                       kTileSizeKFp4>;
```

`ConverterWrapper` now forwards `kUseInterleavedConverter` to the FP4 branch, so selecting
`Fp4KernelDetailsSm80Pair` is all that is needed to decode the prefill buffer directly.

**Shape gate.** Because this reuses the interleaved GEMV (`kInterleave=4`, `kStepK=32`), it
inherits the interleaved rules — MXFP4 `group_size == 32`, `n % 16 == 0`, `k % 64 == 0` —
checked at pack time by `is_moe_gemv_fp4_sm80_layout_supported`. Shapes that miss the gate
transparently fall back to a dedicated `gemv_fp4_fc*_weights_decode_` copy (logged at INFO);
NVFP4 (block 16) always uses the plain `ColToRow` layout. `gpt-oss-20b`
(`k=2880`, `n=5760`/`2880`) clears the gate.

**Measured** — `gpt-oss-20b` MXFP4, post-load device memory:

| configuration | post-load |
|---|---|
| before (3 copies) | 32908 MiB |
| release initializers only (2 copies) | 23164 MiB |
| **single copy (default)** | **13996 MiB** |

> **Returning the memory to the device.** Released initializers only shrink the process's
> device footprint when initializers bypass the BFC arena. Set the session option
> `session.use_device_allocator_for_initializers = 1`; otherwise the freed bytes are merely
> recycled inside the arena for later activation/KV allocations.

### 9.12 Fixed-shape SM90 DeepGEMM path

**Experimental, opt-in (`ORT_QMOE_FP4_DEEPGEMM=1`, default off).** This is a specialized
fixed-shape decode/verification path for H200, initially validated with DeepSeek V4. It is not
DeepGEMM's Blackwell FP8×FP4 kernel:
`PrePack` converts the MXFP4 expert weights once to E4M3 with power-of-two FP32 block scales,
and execution uses DeepGEMM's SM90 FP8 1D2D `MGroupedMasked` GEMM for FC1 and FC2. GEMM
outputs remain BF16.

The path is enabled only when every construction-time constraint holds:

| Constraint | Required value |
|------------|----------------|
| CUDA EP build | bundled or plugin CUDA EP with `HAS_SM90_OR_LATER` |
| GPU | SM90 with exactly 132 SMs and at least 120 GiB global memory (H200) |
| QMoE mode | `quant_type="fp4"`, BF16 activation/output, `k=6` |
| Activation | `activation_type="swiglu"`, `swiglu_fusion=1` (interleaved gate/value) |
| Local experts | 32 (the validated 8-rank split) |
| Hidden / local intermediate | 4096 / 2048 |
| FC1 packed weight shape | `[32, 4096, 2048]` (`[E, K, 2*inter/2]`) |
| FC2 packed weight shape | `[32, 2048, 2048]` (`[E, inter, hidden/2]`) |

The FC shapes must be statically available in the ONNX graph. At `PrePack`, FC1/FC2 weights,
E8M0 block scales, and per-expert global scales must all be present. The expected block-scale
shapes are `[32, 4096, 128]` for FC1 and `[32, 4096, 64]` for FC2. Once all three inputs for an
FC are available, `LaunchQMoEQuantizeFp4WeightsToFp8` creates the persistent E4M3 expert-weight
buffer and FP32 `[128 N, 128 K]` block scales, verifies element-wise round-trip exactness, and
releases that path's staged MXFP4 weight and scale copies.

The 4-rank split with 64 local experts is intentionally unsupported. Its doubled persistent
conversion footprint exhausts device memory during `PrePack` in the tested deployment. Such a
graph does not pass the static DeepGEMM gate and continues through the standard QMoE path even
when `ORT_QMOE_FP4_DEEPGEMM=1`.

Each invocation has additional runtime gates:

- `1 <= num_tokens <= 8`;
- no FC1 or FC2 expert bias;
- the local runner topology is TP=1, EP=1, cluster=1 (model-level TP/EP sharding must happen
  outside this QMoE node, as in the DeepSeek V4 per-rank graph);
- no AWQ, activation scale input, group-wise quantization parameters, or weight-only scales
  are passed to the DeepGEMM runner.

Calls that miss a runtime gate transparently continue through the normal MXFP4 dispatch
(fused GEMV or grouped-GEMM fallback). Setting the environment variable on unsupported hardware
or a node that misses the static gate likewise leaves the standard path selected. The variable is
read when the QMoE kernel is constructed, so set it before creating the ORT session.

The execution sequence is:

1. Quantize compact BF16 expert-major rows into `[32, 64, 4096]` E4M3 with per-row,
  per-128-K FP32 scales and build the per-expert `masked_m` row counts. Every expert has a
  physical 64-row stride because DeepGEMM's TMA output flattens `[expert, row]` and each
  stride must end on the `BLOCK_M=64` store boundary.
2. Run FC1 as E4M3 `[4096] × [4096,4096]` per expert with the SM90 FP8 1D2D masked-grouped
  kernel, producing BF16 output.
3. Apply the interleaved SwiGLU, including `activation_alpha`, `activation_beta`, and
  `swiglu_limit`, while quantizing its result into `[32,64,2048]` E4M3 with FP32 scales.
4. Run FC2 as E4M3 `[2048] × [2048,4096]`, producing BF16 output; unpack the valid rows,
  then use ORT's existing final routing kernel to apply router weights and combine experts.

**Memory cost.** The persistent E4M3 weights are 0.5 GiB for FC1 plus 0.25 GiB for FC2,
or 0.75 GiB per QMoE node/rank at this fixed shape; FP32 block scales add 48 KiB. The temporary
quantized activations, FP32 scales, BF16 GEMM outputs, and row-count workspace total 44.4 MiB
per invocation. Standard FP4
buffers may also remain available because calls outside the DeepGEMM row gate still need the
normal dispatch. Account for this explicitly before enabling the path across every model layer.

Implementation:

- [deep_gemm_sm90.h](onnxruntime/contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.h) — fixed
  dimensions, row limit, and entry points;
- [deep_gemm_sm90.cu](onnxruntime/contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.cu) — packing,
  activation quantization, masked grouped FP8 GEMMs, SwiGLU, and unpacking;
- [moe_quantization.cc](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc) — environment,
  static/runtime gates, MXFP4→E4M3 PrePack conversion, and dispatch.

---

## 9b. NVFP4 (W4A16, block-16) Details

`quant_type="nvfp4"` is a second FP4 weight-only mode, distinct from MXFP4 (`"fp4"`):

| | MXFP4 (`"fp4"`) | **NVFP4 (`"nvfp4"`)** |
|---|---|---|
| Weight codes | E2M1, 2 codes/byte (`uint8`) | E2M1, 2 codes/byte (`uint8`) — same |
| Block size | 32 | **16** |
| Block scale | `float8e8m0` (E8M0, pow-of-two), stored `uint8` | **`float8e4m3fn`** (E4M3) |
| Global scale | 1.0 (implicit) | **per-expert FP32** `fc*_global_scale` (`weight_scale_2`) |
| Reconstruct | `w = e2m1(code) · e8m0(block_scale)` | `w = e2m1(code) · e4m3(block_scale[n, k/16]) · global_scale[e]` |

NVFP4 is the format emitted by NVIDIA Model-Optimizer (e.g. `nvidia/Qwen3.6-35B-A3B-NVFP4`).

> **Execution behavior:** On **SM120/SM121 (Blackwell)** NVFP4 runs a **native block-scaled CUTLASS
> FP4×FP4 grouped-GEMM prefill path** (activations dynamically quantized to NVFP4 in-runner), gated
> by `ORT_ENABLE_NVFP4_CUTLASS_GEMM` (default on) and the CUTLASS shape requirements. On SM<120, or
> when the native path is disabled / shape-unsupported, NVFP4 falls back to the **dequant-to-A16**
> path. The **fused GEMV decode fast path** additionally handles supported small-decode shapes on all
> SMs. Details in [§9b.1](#9b1-dispatch).

### 9b.1 Dispatch

On **SM120/SM121** NVFP4 selects a **native block-scaled CUTLASS FP4×FP4 grouped-GEMM prefill path**
(`CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, ...>`, runner instantiated in
[moe_gemm_kernels_fp4_fp4.cu](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp4_fp4.cu)).
Activations are dynamically quantized to NVFP4 (block-16 E4M3 scale, global scale 1.0) inside the
runner (`expandInputRowsKernel`), and the prepacked weight block scales are consumed as the
Blackwell 128×4 swizzled SF atom. The path is enabled when `sm_` is 120 or 121,
`ORT_ENABLE_NVFP4_CUTLASS_GEMM` is on (default), and the CUTLASS shape requirements hold
(`hidden`/`inter` multiple of 64); it sets `use_fp4_dequant_fallback_ = false`.
`ORT_FP4_PREFILL_MIN_TOKENS` (default 64) splits the workload: prefill-shaped batches take the
native grouped GEMM while small-decode shapes still take the fused GEMV path; an A16 dense fallback
runner covers batches routed away from native.

When the native path is unavailable (SM<120, disabled, or shape-unsupported), NVFP4 uses the
dequant-to-A16 fallback ([§4.3](#43-dequant-to-a16-fallback)) for prefill / GEMV-unsupported shapes,
and the **fused GEMV decode fast path** ([§4](#4-architecture-dispatch--kernel-paths)) for
small-decode shapes. `enable_fp4_gemv_` is on by default for NVFP4 (opt-out `ORT_ENABLE_FP4_GEMV=0`);
`enable_fp4_sm80_gemm_` stays off (the SM80 grouped-GEMM FP4 prefill path is MXFP4-only).

The NVFP4 dense fallback uses routed-expert compaction, like block-scaled FP8. Each row tile
retains the original routing IDs for instrumentation, remaps the selected experts to compact IDs,
and dequantizes only those experts into FP16/BF16 scratch. Packed E2M1 weights, E4M3 block scales,
FP32 global scales, and optional expert biases are indexed by the original expert ID; outputs
and gathered biases use the compact ID. The map is rebuilt for every tile, including a short
final tile, without host synchronization. ONNX weights and scales remain in their native formats.

Weight scratch is bounded by `C * (N1*K1 + N2*K2) * sizeof(activation)`, where
`C = min(num_experts, rows_per_tile * top_k)`. For Qwen Flash with 512 experts, hidden size
2560, intermediate size 640, fused SwiGLU, and top-10 routing, single-token decode requires
93.75 MiB of dequantized weight scratch rather than 4.6875 GiB. Other routing/GEMM workspace
and persistent prepacked buffers are additional. Compaction does not change the native GEMV
or grouped-GEMM routes and does not eliminate persistent decode prepacking allocations.
The fallback also supports `session.disable_prepacking=1`; native routes still require their
existing prepacking where applicable.

### 9b.2 Fused GEMV decode (group-16)

The MXFP4 decode GEMV ([§9.10](#910-interleaved-gemv-layout--dtype-conditional-accumulation)) is
block-size generic: the kernel indexes scales as `scales[e][k/group_size][n]`, and `GroupSize` is a
compile-time template (`static_assert((CtaK/kInterleave) % GroupSize == 0)`). NVFP4 reuses it with
`group_size = 16`:

- `is_moe_gemv_fp4_supported` accepts `group_size ∈ {16, 32}`; the dispatch instantiates the
  `GroupSize=16` cases in `dispatch_moe_gemv_group_size` /
  `dispatch_moe_gemv_interleaved_swiglu_group_size` ([moe_gemv_device.cuh](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv_device.cuh)).
- Legacy prepacked NVFP4 uses the non-interleaved `ColumnMajor` layout; the opt-in interleaved path
  ([§9.10](#910-interleaved-gemv-layout--dtype-conditional-accumulation)) is MXFP4-only because its
  `kStepK=32` tile is tied to the block-32 scale layout.
- For legacy prepacked NVFP4, `QMoECombineNvfp4ScalesForGemv` ([qmoe_kernels.cu](onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu))
  decodes the `float8e4m3fn` block scales, folds in the per-expert FP32 global scale, and rewrites
  `[E, n, k/16] → [E, k/16, n]` in the activation dtype (`TypeA`) that the GEMV expects.
- The decode gate ([moe_quantization.cc](onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc)) fires
  when `expanded = num_tokens·top_k ∈ (0, 64]` for legacy N-packed storage or
  `(0, 80]` for offline K-packed storage, SwiGLU is fused, and both FC1
  (`n=2·inter`, `k=hidden`) and FC2 (`n=hidden`, `k=inter`) satisfy `n,k ≥ 512` and group-16 block
  alignment.
  The fused expert-map prologue supports `top_k ∈ {1, 2, 4, 6, 8, 10}` and up to 1022 experts.
  Qwen Flash (`hidden=2560`, `inter=640`, `top_k=10`) qualifies for one through six tokens
  with legacy storage, or one through eight tokens with offline K-packed storage.

#### Decode kernel for offline K-packed storage

For `weights_prepacked=1` (row-major `[E, N, K/2]`) the GEMV launcher first tries
`MoeGemvFp4DecodeKernel` ([moe_gemv_fp4.cu](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu)),
a decode-specialised replacement for `MoeGemvFp4RawKPackedKernel`. Each thread owns two adjacent
output columns and the K-lane pair `(2j, 2j + 1)`, so one 128-bit load covers two 16-element blocks;
several K tiles are kept in flight, FP4 is decoded in registers (table-free for scales below 4), and
the activation row is staged once per CTA in shared memory as FP32.

The kernel is **bit-identical** to `MoeGemvFp4RawKPackedKernel`: it keeps the same K-lane count
(`KLanes = 4` for `K <= 1024`, else `8`), the same lane assignment (`lane = block % KLanes`), the same
separate even/odd FP32 accumulators per lane, and the same `acc_even + acc_odd` followed by an xor tree
over `KLanes / 2 .. 1`, so outputs do not depend on which of the two kernels runs.

By default (`ORT_FP4_GEMV_DECODE_KERNEL=1`), dispatch is restricted to the H200-measured decode envelope:
- `SM90` (`sm == 90`) with a selected CUDA image compiled for `SM90` or newer (`attributes.ptxVersion >= 90`;
  forward-compatible builds using pre-SM90 PTX such as `CMAKE_CUDA_ARCHITECTURES=80` fall back to
  `MoeGemvFp4RawKPackedKernel`),
- FP16 or BF16, at most 20 expanded rows, `K % 32 == 0`, `K <= 2560`, even `N <= 2560`, and 16/2/16-byte aligned
  weight/scale/activation pointers,
- Per-projection table-free scale eligibility (`|global_scale| * 448 < 3.98` for every expert in the projection),
  recorded independently for FC1 and FC2 during `PrePack`. Projections whose scales exceed this bound stay on
  `MoeGemvFp4RawKPackedKernel` by default rather than taking the slower generic `Fp4I2FConverter` branch.
- When `ORT_NVFP4_GEMV_NATIVE_SCALE=1` is also enabled on `SM90` (see [§9b.3](#9b3-fast-e2m1--halfbf16-decode)),
  FP16 projections with `K <= 1024` (such as Qwen3.8 FC2, `K=640`) and any projection that stays on
  `MoeGemvFp4RawKPackedKernel` use the 4/8-lane native-scale `MoeGemvFp4RawKPackedKernel<..., NativeScale=true>`
  (`8.03 us` vs. `8.41 us` for FP16 FC2 at 10 expanded rows, and `17.50 / 9.90 us` vs. `18.54 / 10.30 us` on
  large-scale fallback), while remaining bit-identical.
- Steps aside when `ORT_NVFP4_FC1_EXPERT_REUSE=1` selects the two-row reuse kernel.

Set `ORT_FP4_GEMV_DECODE_KERNEL=0` before the first GEMV invocation (latched once per process) to force
`MoeGemvFp4RawKPackedKernel`, or `ORT_FP4_GEMV_DECODE_KERNEL=2` to opt into the full kernel envelope
(`SM90+`, up to 80 expanded rows, `K <= 8192`, any even `N`, and the generic `Fp4I2FConverter` fallback for
`|global_scale| * 448 >= 3.98`).

On H200 (`SM90`, CUDA 13.0) for the Qwen3.8 shape (`hidden=2560`, `inter=640`, 512 experts, `top_k=10`,
`weights_prepacked=1`), the FC1 (`N=1280`, `K=2560`, fused SwiGLU) and FC2 (`N=2560`, `K=640`) launches
average (over 300 launches per CUDA graph, 10 graph replays; repeated-replay variation within ±0.15 us):
- BF16, 10 expanded rows: `12.13` and `7.49 us` vs. `18.51` and `10.27 us` (`1.47x` faster FC1+FC2).
- BF16, 20 expanded rows: `20.79` and `12.20 us` vs. `36.05` and `17.23 us` (`1.62x` faster FC1+FC2).
- FP16, 10 expanded rows: `13.77` and `8.41 us` vs. `17.55` and `9.44 us` (`1.22x` faster FC1+FC2).
- FP16, 20 expanded rows: `24.71` and `14.28 us` vs. `34.01` and `16.80 us` (`1.30x` faster FC1+FC2).

To keep the 512-expert weights (`800 MiB` FC1 + `100 MiB` scales; `400 MiB` FC2 + `50 MiB` scales) cold in
H200's `50 MiB` L2 cache during CUDA-graph timing, consecutive launches in the captured graph advance
`permuted_row_to_expert` (or `router_probs` at the ORT node level) through 64 disjoint routing permutations
covering all 512 experts (`ceil(512 / 10) = 52` launches stream the full `900 MiB` FC1 / `450 MiB` FC2 bank
before any expert repeats). To reproduce the end-to-end ORT QMoE node comparison on H200:

```bash
python onnxruntime/test/python/transformers/benchmark_qwen38_mtp_kernels.py --baseline-stdin --kernel nvfp4_decode < onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu
ORT_FP4_GEMV_DECODE_KERNEL=0 pytest onnxruntime/test/python/transformers/test_qmoe_nvfp4_cuda.py -k test_nvfp4_offline_row_major_decode_gemv_bit_exact
ORT_FP4_GEMV_DECODE_KERNEL=1 pytest onnxruntime/test/python/transformers/test_qmoe_nvfp4_cuda.py -k test_nvfp4_offline_row_major_decode_gemv_bit_exact
```

#### Raw-layout memory option

Set `ORT_NVFP4_GEMV_RAW_LAYOUT=1` **before creating the session** to skip the decode-only
weight repack and combined activation-dtype scale bank. The default is `0` (prepacked),
preserving the existing latency-oriented path. The option applies only to NVFP4 GEMV;
MXFP4, native grouped GEMM, and the dense fallback retain their existing layouts.

Raw GEMV reads `[E, K, N/2]` packed E2M1 weights, `[E, N, K/16]` E4M3 block scales, and FP32
per-expert global scales directly. It transposes an N16/K1024 tile in shared memory and
reuses each scale for sixteen weights. Scales and scaled weights are rounded to the
activation dtype before FP32 accumulation. No persistent weight conversion is performed.
Raw initializers remain available for input validation and fallback in both modes.

For each weight matrix, raw mode avoids an additional `E*N*K/2` weight bytes and
`E*N*K/16*sizeof(activation)` combined-scale bytes. This describes persistent buffers, not
measured peak session memory; native prefill buffers and allocator retention still contribute.

The raw mode is a memory/latency tradeoff, not a universally faster replacement. A100
kernel-only measurements covering FP16/BF16 decode, multi-token prediction, and long K
found the tiled raw implementation approximately 1.8 to 4.5 times slower than prepacked
GEMV, although substantially faster than the original scalar raw kernel. These are not
full-model measurements and do not establish performance on other GPU architectures.
Disabling GEMV entirely is not equivalent: the dense fallback dequantizes every expert,
which can be much slower for large expert counts.

Raw mode ignores `ORT_FP4_GEMV_AUTOTUNE`: the prepacked tiling candidates do not change
the raw kernel, so profiling them only adds synchronization and repeated work. With
`ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO=1`, the routes are reported as `fp4_gemv_raw` and
`fp4_gemv_prepacked`.

### 9b.3 Fast E2M1 → half/bf16 decode

For offline K-packed NVFP4 GEMV, `ORT_NVFP4_GEMV_NATIVE_SCALE=1` optionally
uses native E4M3-to-half conversion on compiled SM90+ code for block scales.
It defaults to false. Set it before initializing the runtime; the launch
specializations cache its value. Older compiled architectures retain software
conversion. Existing K-lane selection and optional expert-row reuse are retained.
Global scaling, activation-dtype rounding, weight decoding, accumulation,
routing, and prefill are unchanged. This scalar conversion does not introduce
FP8 activations or native FP8 matrix multiplication.

CUDA internal-test builds with FP4 QMoE register
`onnxruntime_nvfp4_native_scale_0` and `onnxruntime_nvfp4_native_scale_1` in CTest.
These launch fresh processes with the flag explicitly disabled/enabled, because
the launcher caches it. The native-specific test skips unless FP8 helpers and the
loaded kernel's binary and PTX compilation targets support SM90+; a newer device
alone is insufficient. Software/fallback numerical coverage remains separate.
The all-scale-code regression exercises FP16/BF16 with K=16, 80, and 1040.
Nonzero contributions in every K block, varying scales, and an independent
full-reduction reference cover all four/eight K lanes and later iterations.

CUDA plugin internal-test builds compile the FP4 launcher implementation into
their test implementation target, with relaxed constexpr enabled for its CUTLASS
headers. The standalone kernel benchmark preserves device-function templates and
extracts only the global kernel body, excluding host-only inspection helpers.
Its source-generation unit tests run without CuPy or GPU execution.

For native K-packed NVFP4 verification, `ORT_NVFP4_FC1_EXPERT_REUSE=1` enables
experimental two-row FC1 weight and scale reuse. Set it before the first GEMV
invocation; it is read once per launcher specialization. The dispatch is limited
to fused SwiGLU with 1280 output columns, reduction size 2560, at least 40 expanded
rows, and available sorted expert routing maps. Within that path, only expert
groups with at least four rows reuse a decoded tile; other groups compute rows
individually. Stored FP4 codes/scales and activation-dtype rounding are unchanged.

The default is `0`: local A100 tests improved repeated-expert groups but regressed
sparse expert distributions. Enable it only after measuring the target model's
expert overlap and end-to-end latency on the target GPU. It does not enable native
FP4 tensor-core computation or dequantize the checkpoint into a dense weight bank.

Profiling the NVFP4 decode GEMV on H200 (SM90) at the Qwen shapes showed the FC1/FC2 kernels are
**ALU-pipeline bound** (ncu: ALU ≈ 79%, DRAM ≈ 7%), i.e. the per-element FP4 dequantization — not
memory or tiling — dominates. The original `Fp4I2FConverter::decode` used a `float` lookup table,
a sign branch, and a per-element `float → half/bf16` conversion. It is replaced by a **branchless
`prmt.b32` byte-select** that builds the 16-bit float bit pattern directly (E2M1 has only eight
magnitudes, so the half/bf16 encodings are packed into two 32-bit constants and selected by the
low three code bits, with the sign bit shifted into place):

```cpp
// onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/details.h — Fp4I2FConverter::decode (bit-identical to the LUT)
uint32_t sel = code & 0x7u, sign = uint32_t(code & 0x8u) << 12;  // sign -> bit 15
// half: low byte always 0 -> one prmt; bf16: two prmt (high + low byte).
```

This is numerically bit-identical to the previous LUT. Measured decode speedup (H200, `tokens=1`,
`top_k=8`, `hidden=2048`, `inter=512`, `E=256`, bf16, autotuned tiling):

| kernel | before | after | Δ |
|--------|-------:|------:|---|
| FC1 GEMV (SwiGLU-fused, `n=1024 k=2048`) | 20.6 µs | **13.25 µs** | **−36%** |
| FC2 GEMV (`n=2048 k=512`) | 10.7 µs | **7.55 µs** | **−29%** |

ncu confirms the mechanism: FC1 compute (SM) throughput dropped 68.8% → 55.9% and DRAM rose
7.3% → 10.2% (a more balanced kernel). Full-node latency (incl. expand / sort / finalize /
softmax-topk overhead) fell 66.8 µs → 56.1 µs, and end-to-end Qwen3.6 decode reaches ~131 tok/s on
H200.

### 9b.4 Testing & benchmarking

- Parity: [test_qmoe_nvfp4_cuda.py](onnxruntime/test/python/transformers/test_qmoe_nvfp4_cuda.py)
  (`-k gemv` compares `ORT_ENABLE_FP4_GEMV=1` vs `=0` vs a torch reference).
- Profiling: `profile_qmoe_gemv.py` builds the QMoE node at configurable decode shapes and compares
  GEMV with the grouped-GEMM path (with `ORT_FP4_GEMV_AUTOTUNE_LOG=1` for the chosen tiling).

### 9b.5 Native prefill accuracy & routing

The native FP4×FP4 prefill path quantizes **both** operands to 4-bit NVFP4: weights are
pre-quantized offline, and **activations are dynamically quantized to NVFP4** (block-16 E4M3 scale,
global scale 1.0) inside `expandInputRowsKernel`. This is a true W4A4 GEMM, so it carries more
quantization error than the weight-only (W4A16) dequant-fallback / GEMV paths, which keep activations
in FP16/BF16. Against a full-precision-activation reference the native prefill output therefore
differs by a small but non-negligible amount (order ~0.2 absolute on unit-scale outputs), coming
entirely from the activation quantization — **not** a kernel bug. Forcing the same shape through the
dequant fallback (`ORT_FP4_PREFILL_MIN_TOKENS` very large) reduces the difference to ~1e-3, which
confirms the native GEMM arithmetic is correct.

Routing between the native and weight-only paths is governed by `ORT_FP4_PREFILL_MIN_TOKENS`
(default 64):

- `num_rows >= ORT_FP4_PREFILL_MIN_TOKENS` → native block-scaled FP4×FP4 prefill (when enabled and
  shape-supported on SM120/SM121).
- `num_rows <  ORT_FP4_PREFILL_MIN_TOKENS` → small-decode shapes stay on the fused GEMV decode /
  dequant fallback (W4A16), keeping the latency-sensitive decode path off the extra activation-quant
  error.

`ORT_ENABLE_NVFP4_CUTLASS_GEMM=0` disables the native path entirely (all shapes route to
GEMV/fallback). The parity test
[test_qmoe_nvfp4_cuda.py](onnxruntime/test/python/transformers/test_qmoe_nvfp4_cuda.py) mirrors this
gate: it applies a looser tolerance **only** to shapes that route native (SM120/SM121, native enabled,
`num_rows >= ORT_FP4_PREFILL_MIN_TOKENS`) and keeps the strict bound for all decode/fallback shapes,
so a genuinely broken native kernel (error order ~1.0+) is still caught.

---

## 10. FP8 (W8A16) Details

`quant_type="fp8"` supplies FP8 e4m3 weights with BF16/FP16 activations. This was
added so H200 (SM90) has a working narrow-weight QMoE path that does not require
the FP4 launcher.

### 10.1 Native dispatch (SM90+, per-expert scales only)

```cpp
// Constructor — sm_ >= 90 with ENABLE_FP8
m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp8_e4m3, half>>(...);
// or BF16 variant
m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>(...);
```

The SM80 specialization in [moe_gemm_template_dispatch.h](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h)
is intentionally left uninstantiated for W8A16-FP8; the implementation comment
states that the SM80 path is not supported and the native path is SM90 TMA WS.
On Hopper, `use_wfp8a16` is routed through the TMA warp-specialized dispatcher,
which enforces `inputs.gemm_config.is_tma_warp_specialized` and applies the
per-expert global scale via `alpha_scale_ptr_array` in the epilogue. On SM120,
the code redirects W8A16-FP8 to the SM89 FP8 kernel implementations.

### 10.2 Scale wiring

```
QuantParams::FP8::dequant_fc1  (float*, num_experts)
       │
       ▼
computeFP8DequantScale()  →  alpha_scale_ptr_array[e] = &dequant_fc1[e]
       │
       ▼
GroupedGemm with EpilogueOpDefault:
   output[i] = fp8_to_bf16(gemm_accum[i]) * (*alpha_scale_ptr_array[expert])
```

`computeFP8DequantScale` and the Ampere FP8 epilogue already exist
([moe_kernels.cu](onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_kernels.cu)) — the
QMoE op only needs to construct `QuantParams::FP8(dequant_fc1, nullptr, dequant_fc2)`
from the per-expert global scales (inputs 15/16).

### 10.3 Block-scaled FP8 dispatch and dequant fallback

`LaunchQMoEDequantizeFp8Weights` decodes weights into BF16/FP16 and the dense
A16 runner is used.

Positive `block_size=B` selects square-block FP8, including the official
`Qwen/Qwen3.8-Flash-Next-FP8` `[128,128]` checkpoint format:

- `expert_weight_bits=8`, `quant_type="fp8"`.
- Weights are row-major `float8e4m3fn` `[E,N,K]`, not transposed or CUTLASS-prepacked.
- Inputs 3/6/9 hold float32, float16, or bfloat16 dequantization scales `[E,ceil(N/B),ceil(K/B)]`.
  Each weight becomes `float(weight[e,n,k]) * scale[e,n/B,k/B]`.
  The official checkpoint's `weight_scale_inv` tensors are **bfloat16** and are
  multiplicative dequantization scales despite their name; do not invert them.
  CUDA converts the small scale grids to float32 on the execution stream.
- Inputs 15/16 and all zero points must be omitted. Nonpositive `block_size`
  retains the legacy per-expert global-scale contract and native dispatch.
- On SM80-or-newer GPUs, SiLU/SwiGLU block FP8 uses fused weight-only GEMV/GEMM
  by default for supported routing sizes. These kernels decode FP8 tiles and
  apply square-block scales on chip, without full FP16/BF16 weight buffers.
  SiLU without FC3 retains FC1 accumulators in FP32 through bias and activation,
  then narrows to the activation dtype, matching the dense epilogue. Gated
  projections retain their existing FP16/BF16 intermediate rounding.
  Set `ORT_ENABLE_FP8_FUSED=0` before session creation to select the compact
  dense-dequant fallback. Other activation types and unsupported routing sizes
  also retain this fallback. The legacy native FP8 epilogue supports only one
  scale per expert and cannot apply these two-dimensional blocks.
- Separate FC1 gate / FC3 up projections are supported with `activation_type="silu"`
  and `swiglu_fusion=0`. FC1/FC3 biases are currently rejected in this mode.
  Fused SwiGLU accepts interleaved (`1`) or contiguous gate/up halves (`2`);
  layout 2 is converted to interleaved while dequantizing and rejects FC1 bias.
- Hidden and intermediate sizes must be multiples of 8 and at least 16 for the
  dense CUDA runner. They need not be divisible by `B`; partial scale blocks work.

The official shape is BF16, hidden size 2560, intermediate size 640, 512 experts,
top-10 routing, and SiLU with separate gate/up projections. The dense fallback routes
before dequantizing, compacts the selected expert IDs on the GPU, and decodes
each selected expert once per row tile. Repeated selections share one dense
slot. Biases follow the same mapping; routing records retain original expert IDs.

Dense weight scratch and the GEMM profiler's expert count are bounded by
`C = min(E, rows_per_tile * top_k)`, rather than always using `E`. For separate
gate/up projections, dense weight scratch is `C * 3 * hidden * inter * 2` bytes:
**93.75 MiB** for one-token/top-10 decode, versus **4.6875 GiB** for all 512
experts. Allocation uses this upper bound without copying the active count to
the CPU; unused slots are not dequantized. The small scale grids are still
converted in full. Quantized weight storage is unchanged.

The existing `ep.cuda.qmoe_row_tile_size` session option also bounds FP8 weight
scratch during prefill. Buffers are reused on the same stream between tiles;
experts selected in multiple tiles are decoded again. Without tiling, a large
prefill may still reserve all `E` slots. Legacy global-scale FP8 dispatch is
unchanged. This fallback remains dense-GEMM execution, not the fused block-FP8 path.
Router logits use the existing softmax/top-k semantics and
`normalize_routing_weights` controls renormalization.

With `ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO=1`, block FP8 additionally reports
`ExpertCapacity` and `DequantWeightBytes` (the dense weight buffers only, not
total allocator or profiler memory).

CUDA tests are in `test_qmoe_fp8_cuda.py`. The official 512-expert GPU test is
opt-in with `ORT_RUN_LARGE_FP8_QMOE_TEST=1` and uses runtime weight inputs to avoid
the protobuf 2 GiB initializer limit. It covers both single-token decode and
multi-token prefill with BF16 scales. Smaller tests also cover float32/float16
scales, initializer weights, runtime updates, sparse routing with biases,
partial final tiles, and the decode scratch bound. Internal CUDA tests in
`qmoe_fp8_compaction_test.cc` check expert remapping, unused slots, and selected
weight/bias decoding for separate and fused layouts.

### 10.4 Kernel instantiation files

| File | Template |
|------|----------|
| `moe_gemm/moe_gemm_kernels_fp16_fp8.cu` | `MoeGemmRunner<half, __nv_fp8_e4m3, half>` |
| `moe_gemm/moe_gemm_kernels_bf16_fp8.cu` | `MoeGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>` |

### 10.5 End-to-end data flow

```
Model input (BF16/FP16)
    │
    ▼
Router → top-k → permute
    │
    ▼  (still BF16/FP16 — no activation quantization)
GEMM1: bf16_act × fp8_weight → bf16_out  (×dequant_fc1 in epilogue)
    │
    ▼
Activation (SwiGLU / SiLU / ReLU / …)
    │
    ▼
GEMM2: bf16_act × fp8_weight → bf16_out  (×dequant_fc2 in epilogue)
    │
    ▼
Un-permute → weighted sum → output
```

---

### 10.6 Optional native E4M3 GEMV decoding

`ORT_QMOE_FP8_GEMV_NATIVE_DECODE=1` enables native E4M3-to-half conversion for
block-scaled FP8 weights in the fused FP16/BF16 GEMV path on compiled SM90+
code. The flag defaults to false. Set it before initializing the runtime;
launch specializations cache its value. Older compiled architectures retain
software conversion. GEMM/prefill continues to use the existing decoder.

CUDA internal-test builds register `onnxruntime_fp8_native_decode_0` and
`onnxruntime_fp8_native_decode_1` in CTest. Each flag value runs in a fresh process.
The graph cases inspect the captured kernel function to verify the requested
template specialization, and native-specific coverage additionally checks the
FP8 build gate and the loaded kernel's binary/PTX targets before running. It skips
explicitly when native instructions are unavailable instead of claiming fallback
execution as native coverage.
All K elements contribute to the numerical reference; the three K blocks have
different scales and the partial tail has a significant nonzero activation.
This detects dropped blocks, missing tails, and reuse of the first block's scale.

This changes only scalar byte decoding. Block scaling, activation-dtype
rounding, accumulation order, row mappings, and gate/up layouts are retained.
Finite E4M3FN values are exactly representable in FP16, so this conversion does
not add quantization. NaN propagation is preserved; NaN-payload equivalence
and full-model output identity are not promised.

CUDA kernel regressions cover all 256 bytes through GEMV with mapped rows,
separate and fused gate/up layouts, 16- and 128-element scale blocks, partial
K blocks, FP32/FP16/BF16 scale tensors, and CUDA graph capture/replay.
Run them in separate processes with the native flag disabled and enabled.

## 11. WFP4AFP8 Details

`quant_type="wfp4afp8"` pairs MXFP4 weights with FP8 e4m3 activations. Unlike
W4A16, both operands use block scaling, so this path uses CUTLASS's
**block-scaled tensor op** primitive (`OpClassBlockScaledTensorOp`) — natively
supported only on SM100+ (Blackwell).

### 11.1 Native dispatch (SM100+)

```cpp
// Constructor — sm_ >= 100 with ENABLE_FP4 + USE_FP4_QMOE + ENABLE_FP8
m_moe_runner = std::make_unique<
    CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>>(...);
// or BF16 output:
m_moe_runner = std::make_unique<
    CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>>(...);
```

The runner is constructed with `T = __nv_fp8_e4m3`, `WeightType = __nv_fp4_e2m1`,
`OutputType = half/bf16`, `InputType = half/bf16`. The user-facing input is
BF16/FP16; the runner quantizes it to MXFP8 (FP8 + per-block ue8m0 scales)
inside `expandInputRowsKernel` (MXFP8 branch). The MXFP8 branch is triggered
when `quant_params.mxfp8_mxfp4.fc{1,2}.weight_block_scale` is non-null.

### 11.2 Two variants

| Variant | `QuantParams` factory | Activation scaling | Used inputs |
|---------|----------------------|--------------------|-------------|
| **A — global-scaled FP8 act** | `QuantParams::FP8MXFP4` | per-tensor or per-expert float scale | weight side (3,15) + (6,16); act 18, 19 |
| **B — MXFP8 block-scaled act** | `QuantParams::MXFP8MXFP4` | per-block `ue8m0` scales | weight side (3,15) + (6,16); act 20, 21 |

The current build uses **Variant B** (`QuantParams::MXFP8MXFP4`) for the native
SM100+ path; activation block scales are produced **inside the runner** by
`expandInputRowsKernel`. The act_scale inputs (18/19) are validated and
pre-packed for forward compatibility with Variant A but are not consumed by the
current native plumbing.

### 11.3 Dequant fallback (SM<100)

```cpp
use_wfp4afp8_dequant_fallback_ = (sm_ < 100);
```

When the fallback is selected, MXFP4 weights are decoded with
`LaunchQMoEDequantizeFp4Weights` and fed into the dense BF16/FP16 MoE runner —
exactly the same path used by `quant_type="fp4"` with `ORT_FP4_SM80_GEMM=0`. Verified working
on SM90 (H200) using the bundled Python parity test.

### 11.4 Kernel instantiation files

| File | Template |
|------|----------|
| `moe_gemm/moe_gemm_kernels_fp8_fp4.cu` | `MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>` and BF16 variant |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm120_fp8_fp4.generated.cu` | SM120 block-scaled tensor op launcher (FP8×FP4, 128×128×128 tile) |

Built only when `onnxruntime_USE_FP4_QMOE=ON` (which implies
`ENABLE_FP4`+`ENABLE_FP8`). The SM120 launcher additionally requires
`COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS` (set by cmake when SM120 is
in `CMAKE_CUDA_ARCHITECTURES`).

### 11.5 Why two CUTLASS paths

```
W4A16 :  FP16 act (full precision) × FP4 weight (block scaled)
        → mixed-input  (CollectiveBuilderMixedInput, group=32, ue8m0 scales)

W4A8  :  FP8 act (block scaled)    × FP4 weight (block scaled)
        → block-scaled tensor op  (OpClassBlockScaledTensorOp — native FP4×FP8 in tensor cores)
```

The block-scaled tensor op path is fundamentally more efficient because the
hardware fuses dequantization with the matrix multiply, vs. the in-register
software dequant of the mixed-input path.

> **MSVC note**: Native SM90/SM120 TMA grouped MoE kernels are disabled in Windows/MSVC
> builds because CUDA 13 generates host stubs that MSVC rejects for over-aligned
> TMA parameters. See [§14.1](#141-msvc-and-tma-grouped-moe-gemm).

---

## 12. Future / Deferred Modes

| Mode | Notation | Activation | Weight | Status |
|------|----------|-----------|--------|--------|
| **W4AFP8** | INT4 weight + FP8 activation | FP8 e4m3 | INT4 (`uint4b_t`) | Deferred — fast path is gated to SM89 only in TRT-LLM (`moe_gemm_template_dispatch.h`); falls back to Ampere dequant on SM90+ which offers no advantage over W4A16-int. A proper SM90 W4AFP8 TMA WS kernel would be needed. |
| **W8A8-fp8** | FP8 act + FP8 weight | FP8 e4m3 | FP8 e4m3 | Not implemented. Targeted for SM89 (RTX 4090). |
| **WFP4AFP8 native validation** | (above) | — | — | Native path implemented; end-to-end validation requires SM100+ hardware. |
| **WFP4AFP8 Variant A** | global-scaled FP8 activation | — | — | Requires QMoE op to accept pre-quantized FP8 input or wire a separate global-scaled BF16→FP8 prologue. |

The schema reserves the necessary input slots (18–21) so adding these modes
will not change the operator interface.

### 12.1 MoE GEMV Optimization Summary

The MoE GEMV work targets decode-sized integer QMoE workloads where grouped GEMM
launch overhead, prologue overhead, and intermediate traffic dominate the FC
compute. Detailed measurements are recorded in
[qmoe_gemv_experiments.md](qmoe_gemv_experiments.md). This section is the
implementation summary and current backlog.

#### Completed per-column INT4 work

Per-column here means the original INT4 W4A16 path with one scale per output
column: scale tensors are `[E, N]`, and `block_size <= 0`.

| Area | What is implemented | Result |
|------|---------------------|--------|
| Benchmarking | Added `profile_qmoe_gemv.py` and `profile_qmoe_gemv.sh` to run GEMV-enabled and `ORT_DISABLE_MOE_GEMV=1` grouped-GEMM profiles in separate processes. | Stable A/B profiles with the `benchmark` NVTX range; use `parse_nsys.py --pattern '%'` so fallback CUTLASS kernels are visible. |
| Route policy | Dispatch is data-gated to FP16/BF16 integer QMoE decode shapes, `expanded_num_rows <= 8`, `N >= 512`, `K >= 512`, and profiled alignment constraints. | Keeps tiny shapes and unprofiled row counts on grouped GEMM. |
| Row-to-expert lookup | The prologue materializes the local expert id for each permuted row and passes it to the GEMV kernels. | Removes repeated prefix-offset scans inside each N-tile CTA; GPT-OSS and Gemma model-shape kernels improved. |
| FC1 interleaved SwiGLU | The FC1 GEMV path can apply interleaved SwiGLU in the GEMV epilogue for the profiled FP16/BF16 INT4 path. | Removes the separate activation launch and FC1 intermediate traffic; GPT-OSS and Gemma improved end-to-end. |
| One-row finalize | `num_rows == 1`, `top_k <= 4` has a static top-k finalize specialization. | Modest GPT-OSS finalize improvement while preserving the existing FC2 GEMV parallelism. |
| End-to-end GPT-OSS | The final per-column GEMV path was validated in ORT GenAI on GPT-OSS-20B INT4. | About 15% faster than the grouped-GEMM baseline and about 8% faster than the FasterTransformer reference at batch 1. |

#### Completed per-column INT8 work

Per-column INT8 means W8A16 with one symmetric scale per output column: scale
tensors are `[E, N]` and `block_size <= 0`. The per-column path previously
required `block_size == 0` exactly in `is_moe_gemv_supported`, but the QMoE
runtime carries per-column scales as `group_size = -1` (the `QuantParams::Int`
default), so per-column INT4 *and* INT8 silently fell back to grouped GEMM. The
gate now treats any `group_size <= 0` as the per-column case, matching the GEMV
launcher dispatch that already mapped `group_size <= 0` to the `GroupSize == 0`
kernel.

| Area | What is implemented | Result |
|------|---------------------|--------|
| Gate fix | `is_moe_gemv_supported` accepts `group_size <= 0` (per-column) alongside 32/64/128 block-wise group sizes. | Per-column INT4 and INT8 now reach the GEMV kernels, and block-wise INT4/INT8 includes `block_size=32`. |
| INT8 details | The existing `(half, uint8_t)` and `(__nv_bfloat16, uint8_t)` GEMV kernel details cover per-column INT8 with no new instantiation. | FC1 interleaved-SwiGLU and FC2 per-column INT8 GEMV run for FP16 and BF16. |
| Profiling | `int8_per_column_*_1024x4096_e8` and `gpt_oss_20b_*_int8_2880x2880_e32` cases profiled in GEMV and `ORT_DISABLE_MOE_GEMV=1` modes. | Real `moe_gemv_kernel` / `moe_gemv_interleaved_swiglu_kernel` confirmed; about 1.2x–1.4x lower benchmark latency than grouped GEMM with valid output. |

#### Completed block-wise INT4/INT8 work


Block-wise here means `quant_type="int"` with `block_size` 32, 64, or 128 and scales
provided as `[E, N, K / block_size]`. QMoE prepack/runtime transposes those
scales to `[E, K_blocks, N]`, and the GEMV kernels consume that same layout.

| Area | What is implemented | Result |
|------|---------------------|--------|
| INT4 and INT8 details | The GEMV kernel details support `(half, cutlass::uint4b_t)`, `(half, uint8_t)`, and the matching `__nv_bfloat16` weight-type pairs. | Both weight types use the SM80 column-interleaved `fpA_intB` layout on all GPUs, matching the grouped-GEMM path, for FP16 and BF16 activations. |
| Group-size dispatch | GEMV templates now cover `GroupSize == 0`, 32, 64, and 128. | Per-column and block-wise paths share the same kernel structure while preserving complete-block checks. |
| Scale indexing | Block-wise scale loads use `real_offset_k / GroupSize * n + real_offset_n` with a K-loop scale step. | Reuses QMoE's existing `[E, K_blocks, N]` runtime scale layout; no new scale pack format is needed. |
| Symmetric-only gate | Block-wise GEMV runs only when zero-point compensation is absent. | Asymmetric block-wise models stay on grouped GEMM until a zero-point GEMV path is implemented and profiled. |
| Model-shape b64 profile | GPT-OSS-20B and Qwen3.6-35B-A3B were profiled with `block_size=64`. | Both use real GEMV kernels under the current 512 threshold and show about 1.4x lower benchmark latency than grouped GEMM fallback. |

#### Completed BF16 enablement

BF16 activations now share the exact dispatch gate and custom GEMV kernels with
FP16. The runtime gate relaxes from `T == half` to `T == half || T ==
__nv_bfloat16`, and `__nv_bfloat16` template instantiations were added for the
per-column INT4, block-wise INT4/INT8, and interleaved-SwiGLU GEMV kernels.

| Area | What is implemented | Result |
|------|---------------------|--------|
| Gate relaxation | `tryLaunchMoeGemvIntSymmetric` and the interleaved-SwiGLU variant accept `__nv_bfloat16` activations with `ScaleBiasType == T`. | For a given shape, BF16 routes to GEMV exactly where FP16 does. |
| Kernel instantiation | `moe_gemv.cu` adds `__nv_bfloat16` details/instantiations (group sizes 0/32/64/128, INT4/INT8, bias on/off) under `ENABLE_BF16`. | The custom FC1/FC2 GEMV kernels run for BF16; no grouped-GEMM fallback when the FP16 gate would route. |
| Profiling | GPT-OSS-20B, Qwen3.6-35B-A3B, and Gemma model shapes profiled with `block_size=64` for both dtypes. | BF16 matches FP16 routing and latency within noise (about 1.3x–1.5x faster than grouped GEMM); SwiGLU BF16 parity tests pass. |

#### Split-K2 SwiGLU GEMV default path

The fp16 INT4 interleaved-SwiGLU GEMV path uses a two-pass Split-K2 FC1 kernel by
default for supported decode shapes. The first pass computes two K-split FP32
partials into QMoE workspace, and the second pass reduces those partials, adds
optional bias, and applies the interleaved SwiGLU epilogue. FC2 stays on the
regular `moe_gemv_kernel` path.

Set `ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1` before process start to force the
previous single-kernel FC1 SwiGLU GEMV path for debugging, A/B benchmarking, or
bisecting numerical differences. On GPT-OSS-20B, Split-K2 reduced FC1 kernel
work from about 21.42 us to 19.98 us and improved repeated CUDA-graph decode
throughput by about 0.9% to 1.6% with valid focused-helper output. A 1000-sample
MMLU smoke matched the opt-out fallback within noise. A future autotuner can
replace this hand-selected default with per-shape route selection.

```bash
onnxruntime/test/python/transformers/profile_qmoe_gemv.py \
  --case gpt_oss_20b_m1_top4_fp16_2880x2880_e32 \
  --disable-splitk2-swiglu --warmup 5 --repeat 100 --nvtx
```
#### Accumulation policy

The QMoE GEMV fast path accumulates fp16 activations in fp16 by default. Set
`ORT_MOE_GEMV_FP32_ACCUM=1` before process start to restore the previous fp32
accumulation path for fp16 activations. BF16 activations always use fp32
accumulation because bf16 accumulation is too lossy.

On the GPT-OSS-20B decode-shaped helper case
`gpt_oss_20b_m1_top4_fp16_2880x2880_e32`, the default fp16-accumulation path was
0.0708 ms versus 0.0812 ms with `ORT_MOE_GEMV_FP32_ACCUM=1`. In a full GPT-OSS
CUDA-graph decode run, default fp16 accumulation reached 386.26 tok/s versus
353.70 tok/s with the fp32 fallback. A 1000-sample MMLU smoke test matched pooled
accuracy at 0.8260 for both modes.

#### Experiments rejected after profiling

| Experiment | Why it was rejected |
|------------|---------------------|
| Broad GEMV enablement for tiny 128x256 cases | Raw GEMV compute kernels were faster, but end-to-end ORT loop latency was worse than grouped GEMM. |
| Expanded rows 8/16 for 1024x4096 before model-specific tuning | GEMV compute stayed competitive, but total latency regressed for larger expanded-row counts. |
| `CtaN=16, Threads=128` | Fewer N-tile CTAs did not offset lower per-CTA efficiency; GPT and Gemma kernels slowed down. |
| `CtaN=8, Threads=64` | Lower thread count slowed both profiled model-size cases. |
| Map-specialized launch | Avoiding the runtime map/prefix branch was neutral or slightly worse and added code size. |
| Naive FC2 GEMV + finalize fusion | Serialized top-k expert GEMVs inside one CTA; GPT-OSS FC2/finalize kernel time regressed sharply. |
| One-row finalize with 128 threads | Underutilized the 2880-wide GPT-OSS output; the 256-thread static top-k variant was better. |
| Asymmetric block-size-128 fallback stress cases | Existing grouped-GEMM parity tolerance was exceeded in this environment; they were not kept in the GEMV-focused matrix. |

#### Remaining ideas not tried

- A better FC2/finalize design that preserves parallelism across top-k experts
  and N tiles, then reduces partial results with bounded numerical change.
- Architecture-specific dispatch thresholds or a small autotuner for SM80, SM89,
  SM90, SM100, and SM120 rather than one broad hand-tuned gate.
- Asymmetric block-wise GEMV with zero-point compensation in the kernel, if model
  demand and grouped-GEMM baseline data justify the maintenance cost.
- More model-shape block-wise profiling, especially `block_size=128`, INT8, and
  end-to-end GenAI runs for models that ship block-wise QMoE weights.
- Native validation on SM100/SM120 for interactions between GEMV routing and the
  FP4 / WFP4AFP8 paths.

Keep `ORT_DISABLE_MOE_GEMV=1` available. It is useful for A/B testing, fallback
validation, and bisecting numerical or performance regressions. For quick
profiling, use
[profile_qmoe_gemv.sh](../../../onnxruntime/test/python/transformers/profile_qmoe_gemv.sh):

```bash
onnxruntime/test/python/transformers/profile_qmoe_gemv.sh \
  --case gpt_oss_20b_m1_top4_fp16_2880x2880_e32 \
  --block-size 64 --warmup 5 --repeat 100
```

---

## 13. Testing

| Test file | Coverage |
|-----------|----------|
| [test_moe_cuda.py](onnxruntime/test/python/transformers/test_moe_cuda.py) | Standard MoE on CUDA: FP16/BF16, SiLU/GeLU/SwiGLU, routing, GEMM parity. SwiGLU coverage includes both GPT-OSS (`TestSwigluMoE`: interleaved, alpha=1.702/beta=1.0/limit=7.0) and Standard/Llama-Gemma (`TestStandardSwigluMoE`: concatenated `swiglu_fusion=2`, alpha=1.0/beta=0.0/no limit → `SiLU(Gate)×Value`). |
| [test_moe_cpu.py](onnxruntime/test/python/transformers/test_moe_cpu.py) | Standard MoE on CPU (smoke). |
| [test_qmoe_cuda.py](onnxruntime/test/python/transformers/test_qmoe_cuda.py) | INT4/INT8 QMoE — primary regression signal for the production QMoE path. Exercises `pack_weights_for_cuda_mixed_gemm` and dequant-then-matmul reference. `TestQMoEIntPrePackSmoke` covers the raw-weight `weights_prepacked=0` in-`PrePack` layout transform (smoke test: asserts finite output, not bit-parity). |
| [test_qmoe_cpu.py](onnxruntime/test/python/transformers/test_qmoe_cpu.py) | INT4/INT8 QMoE on CPU (smoke). |
| [test_qmoe_fp4_cuda.py](onnxruntime/test/python/transformers/test_qmoe_fp4_cuda.py) | MXFP4 QMoE: quantization utilities, packing, FP16/BF16, SiLU/SwiGLU, top-k and expert-count variants. On SM80+ (including SM120) `TestQMoEFP4Sm80SingleWeightCopy` checks the SM80 grouped GEMM + fused GEMV regime against the dequant fallback. |
| [test_qmoe_fp8_cuda.py](onnxruntime/test/python/transformers/test_qmoe_fp8_cuda.py) | Legacy FP8 W8A16 native/dequant paths and square-block FP8 fallback, separate FC3, fused SwiGLU, partial blocks, top-10, validation errors, and opt-in official Qwen dimensions. |
| [test_qmoe_wfp4afp8_cuda.py](onnxruntime/test/python/transformers/test_qmoe_wfp4afp8_cuda.py) | WFP4AFP8 — native Blackwell path requires SM100+; SM<100 exercises the dequant fallback. |

### Reference computation

The "ground truth" is computed by dequantizing weights to FP16 in Python:

```python
dequantized = (q_weight - zero_point) * scale     # INT
# or
dequantized = fp4_to_float(W) * ue8m0_to_float(block_scale) * global_scale  # FP4
reference = input @ dequantized.T
```

This validates the numerical correctness of the dequantization fusion.

---

## 14. Build Configuration

CMake gates relevant to MoE/QMoE (see [cmake/CMakeLists.txt](cmake/CMakeLists.txt) and
[cmake/onnxruntime_providers_cpu.cmake](cmake/onnxruntime_providers_cpu.cmake)):

| Define | Set when | Effect |
|--------|----------|--------|
| `ENABLE_BF16` | CUDA ≥ 11.0 | BF16 weight/activation paths. |
| `ENABLE_FP8`  | CUDA ≥ 11.8 | FP8 e4m3 instantiations and `QuantParams::FP8`. |
| `ENABLE_FP4`  | CUDA ≥ 12.8 | FP4 e2m1 type (`__nv_fp4_e2m1`) and FP4 traits. |
| `onnxruntime_USE_FP4_QMOE` | ON by default for CUDA builds (requires `ENABLE_FP4`) | Enables FP4 / WFP4AFP8 kernel instantiations and CUTLASS launchers. Set to `OFF` to exclude them. |
| `EXCLUDE_SM_100`, `EXCLUDE_SM_120` | architecture exclusion | Drops the corresponding generated kernels. |

`onnxruntime_USE_FP4_QMOE` is enabled automatically when `onnxruntime_USE_CUDA=ON`.
It remains `OFF` for non-CUDA builds and can be explicitly disabled for CUDA builds with
`-Donnxruntime_USE_FP4_QMOE=OFF` when the additional FP4 kernel instantiations are not needed.

CUDA architecture defaults:
- CUDA 12.8+ : `60;70;75;80;86;89;90;100;120`
- CUDA 13.x  : `75;80;86;89;90;100;120`
- SM90+ gets `-a` suffix (enables WGMMA, TMA, `setmaxnreg`).

### CMake exclusion filters (current state)

[cmake/onnxruntime_cuda_source_filters.cmake](cmake/onnxruntime_cuda_source_filters.cmake):

```cmake
if(NOT onnxruntime_USE_FP4_QMOE)
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm90_fp4_.*\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_.*\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp4\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_kernels_fp4_fp4\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
else()
  # CUDA 13 PTXAS does not complete the FP4 M=128/N=64 pingpong specializations
  # in this build configuration. The dispatcher routes that tile through
  # cooperative mainloop variants instead, so exclude only those unused units.
  list(FILTER … EXCLUDE REGEX
       "moe_gemm_tma_ws_sm90_fp4_(fp16|bf16)_m128_n64_k[0-9]+_cm[12]_cn[12]_pp(_finalize)?\\.generated\\.cu")
endif()

if(NOT onnxruntime_USE_FP8_QMOE)
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm90_wfp8_.*\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_fp8_.*\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp8\\.cu")
  list(FILTER … EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
endif()
```

### 14.1 MSVC and TMA grouped MoE GEMM

Windows/MSVC builds normally do not define the grouped TMA MoE compile switches:

- `COMPILE_HOPPER_TMA_GROUPED_GEMMS`
- `COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS`

The generated grouped TMA launchers pass CUTLASS TMA descriptor types through
NVCC-generated host stubs. With CUDA 13 and MSVC, those stubs contain formal
parameters with 128-byte alignment requirements, which triggers MSVC `C2719`:
the requested alignment for a by-value formal parameter cannot be guaranteed.
This affects the generated SM90/SM120 grouped MoE TMA launcher translation units,
including the native SM120 QMoE FP4 / FP8×FP4 launchers.

The source files are still present in the build graph, but the generated launcher
bodies are guarded by the compile switches above, so they become empty units on
MSVC. Runtime dispatch mirrors this build-time choice:

- Standard FP16/BF16 MoE may skip TMA configs and use the existing SM80/Ampere
  grouped GEMM fallback.
- QMoE modes that require grouped TMA kernels do not silently fall back to SM80.
  This includes FP4/block-scaled modes such as native SM120 `fp4`, `wfp4afp8`,
  and other TMA-only mixed quantized paths. They fail with a clear error saying
  the required TMA grouped MoE GEMM was not compiled.
- `wfp4a16` (MXFP4 W4A16, `quant_type="fp4"`) does not need any TMA kernel by default: on
  every SM80+ GPU, including SM120/SM121, it uses the SM80 fused-dequant grouped GEMM and the
  fused MXFP4 GEMV. Only the SM90 opt-in native path (`ORT_ENABLE_FP4_CUTLASS_GEMM=1`) needs
  the Hopper grouped TMA switch, and it fails at session creation when that switch is off.

When FP4 QMoE is enabled for SM120, the build defines the SM120 grouped-TMA switch
and compiles that object library with the MSVC host option `/Zc:__cplusplus-`.
NVCC still compiles device code as C++20, while CUDA 13's `CUtensorMap` keeps its
128-byte payload with ordinary host alignment in the generated stub. This avoids
MSVC `C2719` without changing the device descriptor layout.

### 14.2 LLM object library and SM120 native SASS

The CUDA provider is split into per-architecture OBJECT libraries
([cmake/onnxruntime_providers_cuda.cmake](cmake/onnxruntime_providers_cuda.cmake)). The general
**LLM object library** (`onnxruntime_providers_cuda_llm`, SM75+) holds the ordinary LLM/MoE kernels
(MoE expand/finalize, `fpA_intB` GEMV/GEMM, NVFP4 dequant, ...). The SM120-specific TMA
warp-specialized grouped-GEMM kernels are compiled separately in
`onnxruntime_providers_cuda_sm120_tma` and their bodies are gated behind
`COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS`, so the LLM library itself contains **no SM120-only
device code** — building it at `120-real` simply produces native SASS for the ordinary kernels
instead of leaving them as `compute_120` PTX that must be JIT-compiled on first use.

Architecture filtering of the LLM library (`onnxruntime_filter_cuda_archs`):

| Platform / config | LLM library archs | Rationale |
|---|---|---|
| Linux (any) | includes `120-real` (native SASS) | LLM kernels run natively on SM120; no JIT warm-up; robust to real-only arch lists. |
| Windows/MSVC, general LLM objects | replaces `120-real` with virtual `compute_120` PTX | Native `sm_120a` pulls CCCL `tcgen05` PTX headers that fail to compile with the MSVC host toolchain; SM120 devices JIT ordinary LLM kernels from PTX. |
| any platform, FP4 QMoE objects | includes `120-real` in a dedicated object library | The NVFP4 native path emits `cvt.e2m1x2` in `expandInputRowsKernel`, valid only for real `sm_120a` and **not** expressible in virtual `compute_120` PTX. |

> **CUDA 209 gotcha (real-only arch list).** If the LLM library excludes `120-real` **and** the
> configured arch list carries no virtual `compute_120` PTX entry, the library can end up with no
> loadable SM120 image and fail with CUDA error 209. The MSVC architecture filter now replaces each
> requested SM120-family real architecture with its virtual counterpart for the general LLM target,
> while the FP4 QMoE target retains native SASS. Verify native SASS is present with
> `cuobjdump libonnxruntime_providers_cuda.so | grep 'arch ='` (use the toolkit-matched `cuobjdump`;
> an older one in `PATH` may misparse the multi-arch fatbin and abort).

---

## 15. Limitations & Known Issues

- **Row-wise INT quantization** (`block_size <= 0`): does not currently support
  zero points in the QMoE operator.
- **Asymmetric INT zero points** are supported only when `block_size >= 64`.
- **Minimum dimension**: `hidden_size` and `inter_size` must be ≥ 16 (and aligned
  to 128 bits — multiples of 8 for FP16). See [§4.2](#42-minimum-dimension-constraint-min_dim).
- **Float32 input**: always uses the SM80 (Ampere) kernel path regardless of
  the actual device SM.
- **MXFP4 W4A16 TMA WS path**: the CUTLASS mixed-input kernel uses `sm_90a` WGMMA, so it can run
  only on SM90 and is opt-in there. SM100 and SM120/SM121 have no WGMMA; they use the SM80
  fused-dequant grouped GEMM plus the fused GEMV. SM120 block-scaled FP4 tensor cores require both
  operands to be block-scaled, so they cannot execute W4A16 without quantizing activations.
- **Windows/MSVC SM120 split**: ordinary LLM kernels use virtual `compute_120`
  PTX to avoid CCCL `tcgen05` host-compile failures. FP4 QMoE conversion, runner,
  and grouped-TMA launchers are isolated into native `sm_120a` object libraries;
  the TMA target uses the host-only `CUtensorMap` alignment workaround described
  in [§14.1](#141-msvc-and-tma-grouped-moe-gemm).
- **WFP4AFP8 native** requires SM100+ hardware; only the dequant fallback path
  is validated end-to-end so far.
- **In-`PrePack` INT weight layout transform** (`weights_prepacked=0`) is
  currently covered only by a smoke test (`TestQMoEIntPrePackSmoke`), not a
  bit-parity check: the existing offline pre-pack harness hardcodes
  `force_arch=80` (the same SM80 layout consumed by the CUDA EP on all GPUs),
  so a separate parity harness for this path is still pending.
- **Hopper W4A8** (INT4 weight + FP8 activation) is not supported — TRT-LLM gates
  its fast path to SM89 only.
- **MXFP4 single-copy weights** require `group_size == 32`, `n % 16 == 0` and
  `k % 64 == 0`. Shapes outside that gate keep a second decode-layout copy
  (~2× weight memory); NVFP4 (block 16) always does.
  See [§9.11](#911-single-copy-sm80-weights-prefill--decode-share-one-buffer).

---

## 16. Differences vs. TensorRT-LLM

The CUTLASS kernels are derived from TensorRT-LLM (CUTLASS 4.4.2, commit
`346018db87`) but have been significantly modified.

### Modifications

1. **Pre-packed ZP/Bias optimization** — `PrePack` derives `(K − ZP) × scale`
   biases offline so the kernel handles asymmetric quantization with no extra
   subtraction. (See [§5.2](#52-int4int8-scales--zero-point--bias).)
2. **SwiGLU interleaving** — activation kernels support interleaved Gate/Value
   weights ([§8](#8-swiglu-fusion)).
3. **Sparse Mixer** support via the `use_sparse_mixer` attribute.
4. **`supportsTmaWarpSpecialized()`** exposed on `CutlassMoeFCRunnerInterface`
   to allow dynamic `min_dim` selection without knowing the concrete template
   type at the call site.
5. **MXFP4 in QMoE schema** — extended schema and runner to accept MXFP4 weights
   plus per-expert global scales and ue8m0 block scales ([§9](#9-fp4-mxfp4-details)).
6. **WFP4AFP8 (SM100+)** — added `MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, …>`
   with in-runner BF16/FP16→MXFP8 quantization in `expandInputRowsKernel`
   ([§11](#11-wfp4afp8-details)).
7. **Backported bug fix** (TRT-LLM `603ec03f`) — moved `griddepcontrol.launch_dependents`
   to after `computeTmaWarpSpecializedInputPointers` in
   `computeStridesTmaWarpSpecializedKernel` to fix a pre-exit race.

### Removed (not needed for ORT MoE/QMoE)

- LoRA parameters (`use_lora`, `LoraParams`)
- Min-latency mode (`MoeMinLatencyParams`)
- AllToAll MoE paths (`enable_alltoall`)
- DeepSeek FP8 block-scale GEMM mode (`use_deepseek_fp8_block_scale`,
  `BlockScaleParams`)
- `Deep Gemm`, standalone FP4 GEMM, FP8 block-scale GEMM, fused gated GEMM
  directories (the relevant pieces are inlined into the MoE runner).
