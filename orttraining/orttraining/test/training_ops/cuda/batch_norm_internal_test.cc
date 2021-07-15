// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

#ifdef USE_CUDA
static void TestBatchNormInternal(bool test_double = false, bool T_is_half = false,
                                  bool T1_is_half = false, bool T2_is_half = false,
                                  const std::vector<int64_t>& input_output_dims = {2, 2, 2, 2}) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);

  std::vector<int64_t> channel_dims{2};

  std::vector<float> X = {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f,
                          -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f};
  std::vector<float> scale = {1.0f, 1.0f};
  std::vector<float> B = {0.0f, 0.0f};
  std::vector<float> mean = {1.0f, 2.0f};
  std::vector<float> var = {1.0f, 2.0f};

  std::vector<float> Y = {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f,
                          -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f};
  std::vector<float> running_mean = {-0.1754f, 0.303106f};
  std::vector<float> running_var = {0.7812f, 1.5865f};
  std::vector<float> saved_mean = {-0.306f, 0.114562f};
  std::vector<float> saved_inv_std = {1.2288f, 0.861317f};

  if (test_double) {
    std::vector<double> X_double (X.begin(), X.end());
    std::vector<double> scale_double (scale.begin(), scale.end());
    std::vector<double> B_double (B.begin(), B.end());
    std::vector<double> mean_double (mean.begin(), mean.end());
    std::vector<double> var_double (var.begin(), var.end());

    std::vector<double> Y_double (Y.begin(), Y.end());
    std::vector<double> running_mean_double (running_mean.begin(), running_mean.end());
    std::vector<double> running_var_double (running_var.begin(), running_var.end());
    std::vector<double> saved_mean_double (saved_mean.begin(), saved_mean.end());
    std::vector<double> saved_inv_std_double (saved_inv_std.begin(), saved_inv_std.end());

    test.AddInput<double>("X", input_output_dims, X_double);
    test.AddInput<double>("scale", channel_dims, scale_double);
    test.AddInput<double>("B", channel_dims, B_double);
    test.AddInput<double>("mean", channel_dims, mean_double);
    test.AddInput<double>("var", channel_dims, var_double);

    test.AddOutput<double>("Y", input_output_dims, Y_double);
    test.AddOutput<double>("running_mean", channel_dims, running_mean_double);
    test.AddOutput<double>("running_var", channel_dims, running_var_double);
    test.AddOutput<double>("saved_mean", channel_dims, saved_mean_double);
    test.AddOutput<double>("saved_inv_std", channel_dims, saved_inv_std_double);
  } else {
    if (T_is_half) {
      std::vector<MLFloat16> X_half(X.size());
      ConvertFloatToMLFloat16(X.data(), X_half.data(), int(X.size()));
      test.AddInput<MLFloat16>("X", input_output_dims, X_half);

      std::vector<MLFloat16> Y_half(Y.size());
      ConvertFloatToMLFloat16(Y.data(), Y_half.data(), int(Y.size()));
      test.AddOutput<MLFloat16>("Y", input_output_dims, Y_half);
    } else {
      test.AddInput<float>("X", input_output_dims, X);
      test.AddOutput<float>("Y", input_output_dims, Y);
    }

    if (T1_is_half) {
      std::vector<MLFloat16> scale_half(scale.size());
      ConvertFloatToMLFloat16(scale.data(), scale_half.data(), int(scale.size()));
      test.AddInput<MLFloat16>("scale", channel_dims, scale_half);

      std::vector<MLFloat16> B_half(B.size());
      ConvertFloatToMLFloat16(B.data(), B_half.data(), int(B.size()));
      test.AddInput<MLFloat16>("B", channel_dims, B_half);
    } else {
      test.AddInput<float>("scale", channel_dims, scale);
      test.AddInput<float>("B", channel_dims, B);
    }

    if (T2_is_half) {
      std::vector<MLFloat16> mean_half(mean.size());
      ConvertFloatToMLFloat16(mean.data(), mean_half.data(), int(mean.size()));
      test.AddInput<MLFloat16>("mean", channel_dims, mean_half);

      std::vector<MLFloat16> var_half(var.size());
      ConvertFloatToMLFloat16(var.data(), var_half.data(), int(var.size()));
      test.AddInput<MLFloat16>("var", channel_dims, var_half);

      std::vector<MLFloat16> running_mean_half(running_mean.size());
      ConvertFloatToMLFloat16(running_mean.data(), running_mean_half.data(), int(running_mean.size()));
      test.AddOutput<MLFloat16>("running_mean", channel_dims, running_mean_half);

      std::vector<MLFloat16> running_var_half(running_var.size());
      ConvertFloatToMLFloat16(running_var.data(), running_var_half.data(), int(running_var.size()));
      test.AddOutput<MLFloat16>("running_var", channel_dims, running_var_half);

      std::vector<MLFloat16> saved_mean_half(saved_mean.size());
      ConvertFloatToMLFloat16(saved_mean.data(), saved_mean_half.data(), int(saved_mean.size()));
      test.AddOutput<MLFloat16>("saved_mean", channel_dims, saved_mean_half);

      std::vector<MLFloat16> saved_inv_std_half(saved_inv_std.size());
      ConvertFloatToMLFloat16(saved_inv_std.data(), saved_inv_std_half.data(), int(saved_inv_std.size()));
      test.AddOutput<MLFloat16>("saved_inv_std", channel_dims, saved_inv_std_half);
    } else {
      test.AddInput<float>("mean", channel_dims, mean);
      test.AddInput<float>("var", channel_dims, var);
      test.AddOutput<float>("running_mean", channel_dims, running_mean);
      test.AddOutput<float>("running_var", channel_dims, running_var);
      test.AddOutput<float>("saved_mean", channel_dims, saved_mean);
      test.AddOutput<float>("saved_inv_std", channel_dims, saved_inv_std);
    }
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(CudaKernelTest, BNInternalBasic) { // float case
  TestBatchNormInternal();
}

TEST(CudaKernelTest, BNInternalDouble) { // double case
  TestBatchNormInternal(true);
}

TEST(CudaKernelTest, BNInternalHalf) { // half case
  TestBatchNormInternal(false, true, true, true);
}

TEST(CudaKernelTest, BNInternalHalfHalfFloat) { // half X/Y & scale/B, float mean/var
  TestBatchNormInternal(false, true, true);
}

TEST(CudaKernelTest, BNInternalHalfFloatFloat) { // half X/Y, float scale/B & mean/var
  TestBatchNormInternal(false, true);
}

TEST(CudaKernelTest, BNInternal3DInput) { // float case, 3d input
  TestBatchNormInternal(false, false, false, false, {2, 2, 4});
}

TEST(CudaKernelTest, BNInternal5DInput) { // float case, 5d input
  TestBatchNormInternal(false, false, false, false, {2, 2, 2, 1, 2});
}

TEST(CudaKernelTest, BNInternal1DInput) { // float case, 1d input
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);

  std::vector<int64_t> input_output_dims{16};
  std::vector<int64_t> channel_dims{1};

  test.AddInput<float>("X", input_output_dims,
                       {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f,
                        -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f});
  test.AddInput<float>("var", channel_dims, {1.0f});

  test.AddOutput<float>("Y", input_output_dims,
                        {-0.1948f, 0.2086f, 1.1646f, -0.0951f, -0.1017f, 0.0703f, 1.5754f, 0.1009f,
                         -0.9638f, -1.4131f, 0.5158f, -0.8645f, 0.8622f, -0.3049f, -2.1659f, 1.6059f});
  test.AddOutput<float>("running_mean", channel_dims, {0.0139f});
  test.AddOutput<float>("running_var", channel_dims, {1.1074f});
  test.AddOutput<float>("saved_mean", channel_dims, {-0.0957f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {0.9762f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
#endif // USE_CUDA

}  // namespace test
} // namespace contrib
}  // namespace onnxruntime
