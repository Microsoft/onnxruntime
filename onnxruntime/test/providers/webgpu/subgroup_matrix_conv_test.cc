// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Reference NCHW 2D convolution (group == 1).
static void ComputeExpectedConv(const std::vector<float>& x, const std::vector<float>& w,
                                const std::vector<float>& b, std::vector<float>& y,
                                int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t kh, int64_t kw,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                int64_t dil_h, int64_t dil_w,
                                int64_t out_h, int64_t out_w) {
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < Cout; ++co) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          float sum = b.empty() ? 0.0f : b[co];
          for (int64_t ci = 0; ci < Cin; ++ci) {
            for (int64_t r = 0; r < kh; ++r) {
              for (int64_t s = 0; s < kw; ++s) {
                const int64_t ih = oh * stride_h - pad_h + r * dil_h;
                const int64_t iw = ow * stride_w - pad_w + s * dil_w;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                  continue;
                }
                const float xv = x[((n * Cin + ci) * H + ih) * W + iw];
                const float wv = w[((co * Cin + ci) * kh + r) * kw + s];
                sum += xv * wv;
              }
            }
          }
          y[((n * Cout + co) * out_h + oh) * out_w + ow] = sum;
        }
      }
    }
  }
}

#if !defined(DISABLE_CONTRIB_OPS)
// Reference for the fused activations; must stay in step with GetActivationSnippet
// (core/providers/webgpu/nn/fuse_utils.cc), which the kernel's epilogue builds from.
static void ApplyExpectedActivation(const std::string& kind, const std::vector<float>& params,
                                    std::vector<float>& y) {
  const float alpha = params.size() > 0 ? params[0] : 0.0f;
  const float beta = params.size() > 1 ? params[1] : 0.0f;
  for (float& v : y) {
    if (kind == "Relu") {
      v = std::max(v, 0.0f);
    } else if (kind == "Sigmoid") {
      v = 1.0f / (1.0f + std::exp(-v));
    } else if (kind == "Clip") {
      v = std::clamp(v, alpha, beta);
    } else if (kind == "HardSigmoid") {
      v = std::clamp(alpha * v + beta, 0.0f, 1.0f);
    } else if (kind == "LeakyRelu") {
      v = v >= 0.0f ? v : alpha * v;
    } else if (kind == "Tanh") {
      v = std::tanh(v);
    } else if (kind == "QuickGelu") {
      v = v * (1.0f / (1.0f + std::exp(-(alpha * v))));
    } else if (kind == "HardSwish") {
      v = v * std::clamp(v * (1.0f / 6.0f) + 0.5f, 0.0f, 1.0f);
    } else if (kind == "Elu") {
      v = v >= 0.0f ? v : alpha * (std::exp(v) - 1.0f);
    } else if (kind == "ThresholdedRelu") {
      v = v > alpha ? v : 0.0f;
    } else if (kind == "Erf") {
      v = std::erf(v);
    } else if (kind == "Gelu") {
      v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
    } else if (kind == "FastGelu") {
      v = v * (0.5f + 0.5f * std::tanh(v * (0.035677408136300125f * v * v + 0.79788456080286535f)));
    } else if (kind == "Softplus") {
      v = std::max(v, 0.0f) + std::log(1.0f + std::exp(-std::abs(v)));
    } else {
      FAIL() << "unhandled activation in the test reference: " << kind;
    }
  }
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

// Runs a Conv over the WebGPU EP and compares against the CPU reference above. A
// non-empty `activation` switches to FusedConv instead.
//
// Device-independent by construction (enabled, not DISABLED_ like the *_large tests
// here): where CanApplySubgroupMatrixConv() declines, the EP falls back to regular
// conv, which must match the same reference -- so a failure is a real defect
// either way, not a capability gap.
static void RunConvTest(int64_t N, int64_t Cin, int64_t H, int64_t W,
                        int64_t Cout, int64_t kh, int64_t kw,
                        int64_t stride = 1, int64_t pad = 0, int64_t dilation = 1,
                        bool has_bias = false,
                        const std::string& activation = "",
                        const std::vector<float>& activation_params = {},
                        float abs_err = 0.06f, float rel_err = 0.02f,
                        bool weight_is_initializer = true) {
#if defined(DISABLE_CONTRIB_OPS)
  if (!activation.empty()) {
    GTEST_SKIP() << "FusedConv requires contrib ops.";
  }
#endif
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const int64_t out_h = (H + 2 * pad - dilation * (kh - 1) - 1) / stride + 1;
  const int64_t out_w = (W + 2 * pad - dilation * (kw - 1) - 1) / stride + 1;
  ASSERT_GT(out_h, 0);
  ASSERT_GT(out_w, 0);

  const std::vector<int64_t> x_dims{N, Cin, H, W};
  const std::vector<int64_t> w_dims{Cout, Cin, kh, kw};
  const std::vector<int64_t> b_dims{Cout};
  const std::vector<int64_t> y_dims{N, Cout, out_h, out_w};

  RandomValueGenerator random{1234};
  std::vector<float> x_vals(random.Gaussian<float>(AsSpan(x_dims), 0.0f, 0.25f));
  std::vector<float> w_vals(random.Gaussian<float>(AsSpan(w_dims), 0.0f, 0.25f));
  std::vector<float> b_vals = has_bias ? random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f)
                                       : std::vector<float>{};

  std::vector<float> expected(static_cast<size_t>(N * Cout * out_h * out_w));
  ComputeExpectedConv(x_vals, w_vals, b_vals, expected, N, Cin, H, W, Cout, kh, kw,
                      stride, stride, pad, pad, dilation, dilation, out_h, out_w);

  std::unique_ptr<OpTester> test;
  if (activation.empty()) {
    test = std::make_unique<OpTester>("Conv", 11);
  } else {
#if !defined(DISABLE_CONTRIB_OPS)
    test = std::make_unique<OpTester>("FusedConv", 1, onnxruntime::kMSDomain);
    test->AddAttribute("activation", activation);
    // GetFusedActivationAttr requires the attribute for the parameterized kinds and
    // ignores it for the rest, so only set it when the case supplies parameters.
    if (!activation_params.empty()) {
      test->AddAttribute("activation_params", activation_params);
    }
    ApplyExpectedActivation(activation, activation_params, expected);
#endif
  }
  test->AddAttribute("group", static_cast<int64_t>(1));
  test->AddAttribute("kernel_shape", std::vector<int64_t>{kh, kw});
  test->AddAttribute("pads", std::vector<int64_t>{pad, pad, pad, pad});
  test->AddAttribute("strides", std::vector<int64_t>{stride, stride});
  test->AddAttribute("dilations", std::vector<int64_t>{dilation, dilation});

  test->AddInput<MLFloat16>("X", x_dims, FloatsToMLFloat16s(x_vals));
  // Default (constant) weight exercises the cached-transpose branch; RuntimeWeight
  // covers the non-constant (rebuilt every run) one.
  test->AddInput<MLFloat16>("W", w_dims, FloatsToMLFloat16s(w_vals), weight_is_initializer);
  if (has_bias) {
    test->AddInput<MLFloat16>("B", b_dims, FloatsToMLFloat16s(b_vals));
  }
  test->AddOutput<MLFloat16>("Y", y_dims, FloatsToMLFloat16s(expected));
  test->SetOutputAbsErr("Y", abs_err);
  test->SetOutputRelErr("Y", rel_err);

  test->ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// K = kh*kw*Cin is a multiple of 16: the aligned fast path. out_h*out_w and Cout are
// tile multiples here (no partial tiles).
TEST(SubgroupMatrixConv, Aligned) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/1, /*Cin=*/32, /*H=*/8, /*W=*/8, /*Cout=*/128, /*kh=*/3, /*kw=*/3);
}

// Partial output tiles: out_h*out_w (M) and/or Cout (N) are not tile multiples,
// exercising the kernel's bounds-checked M stores and its N tile shift.
TEST(SubgroupMatrixConv, PartialTiles) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/15, /*W=*/13, /*Cout=*/66, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/17, /*W=*/9, /*Cout=*/130, /*kh=*/3, /*kw=*/3);
}

// Cout (GEMM N) values that stress the trailing-N-tile shift keeping the unchecked
// weight load inside the buffer. tile_n clamps to the largest multiple of 16 in
// Cout, so Cout in (16, 32) overlaps the shifted tile with the previous one by all
// but (Cout - 16) columns -- n_skip near kTileN, the narrowest write-out this
// kernel does.
//
//   16  -> exactly one N block, num_n_tile == 1, no shift at all
//   17  -> odd, minimal overhang: last tile shifts back by 15, writes 1 column
//   18  -> n_skip == 14, writes 2 columns
//   31  -> odd, largest overhang below the next block
//   33  -> odd, just past two blocks
//   50  -> if the selector picks tile_n 64 the clamp cuts it to 48, i.e. three
//          subgroup matrices along N, a count the candidate list never produces
//          on its own
//
// Odd values also cover the removal of the old N % 2 == 0 restriction: unlike the
// MatMul kernel, this one needs no even N because its weight is column-major with
// stride K, so load alignment follows from K % 16 == 0 rather than from Cout.
TEST(SubgroupMatrixConv, ShiftedAndOddNTiles) {
  for (int64_t channels_out : {16, 17, 18, 31, 33, 50}) {
    RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/10, /*W=*/10, /*Cout=*/channels_out, /*kh=*/3, /*kw=*/3);
  }
  // Odd Cout together with a bias, which the epilogue indexes by the shifted column.
  RunConvTest(/*N=*/2, /*Cin=*/16, /*H=*/11, /*W=*/7, /*Cout=*/17, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true);
}

// Cin values that select each im2col gather width (vec_size 4 / 2 / 1). K must stay a
// multiple of 16, so the kernel size is chosen per case to make kh*kw*Cin align.
TEST(SubgroupMatrixConv, GatherVectorWidths) {
  // Cin % 4 == 0 -> vec_size 4.
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3);
  // Cin % 4 != 0, Cin % 2 == 0 -> vec_size 2. K = 2*4*6 = 48.
  RunConvTest(/*N=*/1, /*Cin=*/6, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/2, /*kw=*/4);
  // Cin odd -> vec_size 1. K = 2*8*3 = 48.
  RunConvTest(/*N=*/1, /*Cin=*/3, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/2, /*kw=*/8);
}

// Padding and strides: the im2col gather must return zero outside the input.
TEST(SubgroupMatrixConv, PadStride) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/2, /*pad=*/1);
  RunConvTest(/*N=*/1, /*Cin=*/32, /*H=*/14, /*W=*/14, /*Cout=*/64, /*kh=*/5, /*kw=*/5,
              /*stride=*/1, /*pad=*/2);
}

// Dilated convolution.
TEST(SubgroupMatrixConv, Dilation) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/20, /*W=*/20, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/2, /*dilation=*/2);
}

// Optional bias add.
TEST(SubgroupMatrixConv, Bias) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/0, /*dilation=*/1, /*has_bias=*/true);
}

// Non-constant weight: PrePackInternal never runs, IsWeightConstant() is false and the
// impl rebuilds the OIHW -> OHWI transpose on every run instead of caching it. Same
// numerics as the constant-weight cases above, so any divergence points at the cache.
TEST(SubgroupMatrixConv, RuntimeWeight) {
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"", /*activation_params=*/{},
              /*abs_err=*/0.06f, /*rel_err=*/0.02f, /*weight_is_initializer=*/false);
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/11, /*W=*/9, /*Cout=*/17, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/0, /*dilation=*/1, /*has_bias=*/false,
              /*activation=*/"", /*activation_params=*/{},
              /*abs_err=*/0.06f, /*rel_err=*/0.02f, /*weight_is_initializer=*/false);
}

// Batched conv: each batch slice is dispatched on z. Small per-slice M x N grids
// with a larger batch also stress the selector's split-K clamping.
TEST(SubgroupMatrixConv, Batched) {
  RunConvTest(/*N=*/4, /*Cin=*/16, /*H=*/16, /*W=*/16, /*Cout=*/64, /*kh=*/3, /*kw=*/3);
  RunConvTest(/*N=*/8, /*Cin=*/32, /*H=*/8, /*W=*/8, /*Cout=*/64, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1);
}

#if !defined(DISABLE_CONTRIB_OPS)
// Fused activations. The epilogue is generated from GetActivationSnippet, so every
// ActivationKind is expected to work -- including the ones a hand-written shader
// switch would have silently dropped. Each case below is a distinct branch of that
// snippet; the parameterized kinds also check that activation_params reach the
// uniforms in the right slots.
TEST(SubgroupMatrixConv, FusedActivationUnparameterized) {
  for (const char* act : {"Relu", "Sigmoid", "Tanh", "HardSwish", "Softplus"}) {
    RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
                /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true, /*activation=*/act);
  }
}

TEST(SubgroupMatrixConv, FusedActivationParameterized) {
  // alpha, beta land in activation_param_0 / activation_param_1.
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"Clip", /*activation_params=*/{-0.5f, 0.5f});
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"HardSigmoid", /*activation_params=*/{0.2f, 0.5f});
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"LeakyRelu", /*activation_params=*/{0.1f});
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"Elu", /*activation_params=*/{1.0f});
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"ThresholdedRelu", /*activation_params=*/{0.25f});
  // QuickGelu has two shader variants: alpha == 1 drops the multiply and the uniform.
  for (float alpha : {1.0f, 1.702f}) {
    RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
                /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
                /*activation=*/"QuickGelu", /*activation_params=*/{alpha});
  }
}

// The kinds that need GetActivationDeclaration to emit a module-scope helper
// (fused_act_erf for Erf/Gelu, fused_act_tanh for FastGelu) ahead of the epilogue --
// the part of fuse_utils a hand-written shader switch cannot express at all. Erf and
// Gelu go through the Abramowitz & Stegun 7.1.26 approximation evaluated in f16,
// which costs roughly another 1e-2 of absolute accuracy against the CPU's std::erf,
// hence the looser bound.
TEST(SubgroupMatrixConv, FusedActivationWithDeclaration) {
  for (const char* act : {"Erf", "Gelu", "FastGelu"}) {
    RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/12, /*W=*/12, /*Cout=*/32, /*kh=*/3, /*kw=*/3,
                /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true, /*activation=*/act,
                /*activation_params=*/{}, /*abs_err=*/0.08f, /*rel_err=*/0.03f);
  }
}

// A fused activation on a shifted/odd N tile: the epilogue, the bias index and the
// tile shift all interact in the same loop.
TEST(SubgroupMatrixConv, FusedActivationOddN) {
  RunConvTest(/*N=*/2, /*Cin=*/16, /*H=*/11, /*W=*/9, /*Cout=*/17, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"Relu");
  RunConvTest(/*N=*/1, /*Cin=*/16, /*H=*/11, /*W=*/9, /*Cout=*/33, /*kh=*/3, /*kw=*/3,
              /*stride=*/1, /*pad=*/1, /*dilation=*/1, /*has_bias=*/true,
              /*activation=*/"LeakyRelu", /*activation_params=*/{0.1f});
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
