// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <type_traits>

#include "activation_op_test.h"
#include <limits>
#include "core/providers/cpu/activation/activations.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if defined(ENABLE_TRAINING_OPS)
namespace {
void TestElementwiseGradientOp(
    const char* op,
    const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
    std::function<float(const std::vector<float>&)> expected_func,
    const std::unordered_map<std::string, float> attrs = {},
    int opset_version = 7, const char* domain = kOnnxDomain) {
  const auto first_input = inputs.begin();
  ASSERT_NE(first_input, inputs.end());
  for (auto input = first_input; input != inputs.end(); ++input) {
    if (input == first_input) continue;
    ASSERT_EQ(first_input->second.size(), input->second.size());
  }

  OpTester test(op, opset_version, domain);

  for (auto attr : attrs) {
    test.AddAttribute(attr.first, attr.second);
  }

  const auto input_size = first_input->second.size();
  std::vector<int64_t> dims{static_cast<int64_t>(input_size)};

  std::vector<float> expected_vals;
  for (size_t i = 0; i < input_size; i++) {
    std::vector<float> params(inputs.size());
    std::transform(
        inputs.begin(), inputs.end(), params.begin(),
        [i](const std::pair<std::string, std::vector<float>>& input) {
          return input.second[i];
        });
    expected_vals.push_back(expected_func(params));
  }

  for (const auto& input : inputs) {
    test.AddInput<float>(input.first.c_str(), dims, input.second);
  }
  test.AddOutput<float>("dX", dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

constexpr float ReluGrad(float dy, float x) {
  return x > 0 ? dy : 0;
}

constexpr float SigmoidGrad(float dy, float y) {
  return dy * y * (1 - y);
}

constexpr float TanhGrad(float dy, float y) {
  return dy * (1 - y * y);
}

constexpr float LeakyReluGrad(float dy, float y, float alpha) {
  return dy * (y > 0.0f ? 1.0f : alpha);
}
}  // namespace
#endif

TEST_F(ActivationOpTest, Sigmoid) {
  auto sigmoid_f32 = [](float x) {
    auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };
  auto sigmoid_f64 = [](double x) {
    auto y = 1. / (1. + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };
  // Test sigmoid using the default validator
  TestActivationOp<float>("Sigmoid", input_values, sigmoid_f32);
  // Test sigmoid using custom validator to check output range
  TestActivationOp<float>("Sigmoid", input_values, sigmoid_f32,
                          {}, {}, true, 7, kOnnxDomain,
                          [](const std::vector<OrtValue>& fetches, const std::string&) {
                            const auto& output = fetches[0].Get<Tensor>();
                            const float* output_data = output.Data<float>();
                            for (int64_t i = 0; i < output.Shape().Size(); ++i) {
                              EXPECT_TRUE(output_data[i] >= 0.f && output_data[i] <= 1.f)
                                  << "Output value out of range: " << output_data[i];
                            }
                          });
  // Test sigmoid using the default validator
  TestActivationOp<double>("Sigmoid", input_values_double, sigmoid_f64);
  // Test sigmoid using custom validator to check output range
  TestActivationOp<double>("Sigmoid", input_values_double, sigmoid_f64,
                           {}, {}, true, 7, kOnnxDomain,
                           [](const std::vector<OrtValue>& fetches, const std::string&) {
                             const auto& output = fetches[0].Get<Tensor>();
                             const double* output_data = output.Data<double>();
                             for (int64_t i = 0; i < output.Shape().Size(); ++i) {
                               EXPECT_TRUE(output_data[i] >= 0. && output_data[i] <= 1.)
                                   << "Output value out of range: " << output_data[i];
                             }
                           });
}

TEST_F(ActivationOpTest, HardSigmoid) {
  float alpha = 0.2f;
  float beta = 0.5f;
  TestActivationOp<float>("HardSigmoid",
                          input_values,
                          [alpha, beta](float x) {
                            return std::max(std::min((alpha * x + beta), 1.0f), 0.0f);
                          },
                          {{"alpha", alpha}, {"beta", beta}});
}

#if defined(USE_CUDA)
TEST_F(ActivationOpTest, HardSwish) {
  TestActivationOp<float>("HardSwish", input_values, [](float x) { return x * std::max(std::min(x / 6.0f + 0.5f, 1.0f), 0.0f); }, {}, {},
                          /*is_tensorrt_supported=*/false,
                          /*opset_version= */ 14);
  TestActivationOp<double>("HardSwish", input_values_double, [](double x) { return x * std::max(std::min(x / 6.0 + 0.5, 1.0), 0.0); }, {}, {},
                           /*is_tensorrt_supported=*/false,
                           /*opset_version= */ 14);
}
#endif  // USE_CUDA

TEST_F(ActivationOpTest, Tanh) {
  TestActivationOp<float>("Tanh",
                          input_values,
                          [](float x) { return std::tanh(x); });
  TestActivationOp<double>("Tanh",
                           input_values_double,
                           [](double x) { return std::tanh(x); });
}

TEST_F(ActivationOpTest, Relu) {
  TestActivationOp<float>("Relu",
                          input_values,
                          [](float x) { return std::max(x, 0.0f); });
  TestActivationOp<double>(
      "Relu",
      input_values_double,
      [](double x) { return std::max(x, 0.0); },
      {}, {},
      /*is_tensorrt_supported=*/false);
  TestActivationOp<int8_t>(
      "Relu",
      input_values_int8,
      [](int8_t x) { return std::max(x, static_cast<int8_t>(0)); },
      {}, {},
      /*is_tensorrt_supported=*/false,
      /*opset_version= */ 14);
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || defined(USE_COREML)
  TestActivationOp<MLFloat16>(
      "Relu",
      input_values_fp16,
      [](MLFloat16 x) {
        if (x.ToFloat() > 0.0f) return x;
        return MLFloat16();
      },
      {}, {},
      /*is_tensorrt_supported=*/false,
      /*opset_version= */ 11);
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
}

#if defined(USE_CUDA) || defined(USE_COREML)
TEST_F(ActivationOpTest, Sigmoid_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Sigmoid", 14);

  auto formula = [](float x) {
    auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(ActivationOpTest, Tanh_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Tanh", 14);

  auto formula = [](float x) { return std::tanh(x); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(ActivationOpTest, Relu_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Relu", 14);

  auto formula = [](float x) { return std::max(x, 0.0f); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

#if defined(USE_CUDA) || defined(USE_DNNL)
TEST_F(ActivationOpTest, Sigmoid_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Sigmoid", 14);

  auto formula = [](float x) {
    auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST_F(ActivationOpTest, Tanh_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Tanh", 14);

  auto formula = [](float x) { return std::tanh(x); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST_F(ActivationOpTest, Relu_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Relu", 14);

  auto formula = [](float x) { return std::max(x, 0.0f); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA || USE_DNNL

#if defined(USE_DNNL)
TEST_F(ActivationOpTest, LeakyRelu_bfloat16) {
#ifdef USE_DNNL
  DNNL_GTEST_SKIP();

  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("LeakyRelu", 16);
  float alpha = 0.01f;  // oneDNN set alpha equal to 0.01
  auto formula = [alpha](float x) { return (x >= 0) ? x : alpha * x; };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_DNNL

TEST_F(ActivationOpTest, Elu) {
  float alpha = 0.1f;
  TestActivationOp<float>("Elu",
                          input_values,
                          [alpha](float x) { return (x >= 0) ? x : alpha * (exp(x) - 1); },
                          {{"alpha", alpha}});
}

TEST_F(ActivationOpTest, Celu) {
  float alpha = -0.5f;
  TestActivationOp<float>(
      "Celu",
      input_values,
      // TODO: Investigate why gcc 4 fails to compile without the explicit cast
      [alpha](float x) { return std::max(0.0f, x) + std::min(0.0f, alpha * (static_cast<float>(exp(x / alpha)) - 1)); },
      // Disable on TensorRT as it seems like it doesn't yet support Celu
      {{"alpha", alpha}}, {}, false, 12);
}

TEST_F(ActivationOpTest, CeluOpset28NewTypes) {
  constexpr float alpha = 0.5f;
  const std::vector<float> input{-1.0f, -0.5f, 0.0f, 1.0f};
  const std::vector<float> expected{-0.43233237f, -0.31606027f, 0.0f, 1.0f};

  OpTester fp16_test("Celu", 28);
  fp16_test.AddAttribute("alpha", alpha);
  fp16_test.AddInput<MLFloat16>("X", {2, 2}, FloatsToMLFloat16s(input));
  fp16_test.AddOutput<MLFloat16>("Y", {2, 2}, FloatsToMLFloat16s(expected));
  fp16_test.Run();

  OpTester bf16_test("Celu", 28);
  bf16_test.AddAttribute("alpha", alpha);
  bf16_test.AddInput<BFloat16>("X", {2, 2}, FloatsToBFloat16s(input));
  bf16_test.AddOutput<BFloat16>("Y", {2, 2}, FloatsToBFloat16s(expected));
  bf16_test.Run();

  TestActivationOp<double>(
      "Celu", {{-1.0, -0.5, 0.0, 1.0}},
      [](double x) { return std::max(0.0, x) + std::min(0.0, alpha * (std::exp(x / alpha) - 1.0)); },
      {{"alpha", alpha}}, {}, false, 28);
}

TEST_F(ActivationOpTest, CeluOpset28Float16NaNPropagation) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  OpTester test("Celu", 28);
  test.AddInput<MLFloat16>("X", {5}, FloatsToMLFloat16s({-1.0f, nan, 0.0f, -nan, 1.0f}));
  test.AddOutput<MLFloat16>("Y", {5}, FloatsToMLFloat16s({std::exp(-1.0f) - 1.0f, nan, 0.0f, -nan, 1.0f}));
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST_F(ActivationOpTest, CeluOpset28BFloat16NaNPropagation) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  OpTester test("Celu", 28);
  test.AddInput<BFloat16>("X", {5}, FloatsToBFloat16s({-1.0f, nan, 0.0f, -nan, 1.0f}));
  test.AddOutput<BFloat16>("Y", {5}, FloatsToBFloat16s({std::exp(-1.0f) - 1.0f, nan, 0.0f, -nan, 1.0f}));
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST_F(ActivationOpTest, CeluOpset28BFloat16CastsAlpha) {
  OpTester test("Celu", 28);
  test.AddAttribute("alpha", 0.5001f);
  test.AddInput<BFloat16>("X", {1}, {BFloat16(-3.109375f)});
  test.AddOutput<BFloat16>("Y", {1}, {BFloat16(-0.498046875f)});
  test.Run();
}

TEST_F(ActivationOpTest, LeakyRelu) {
  float alpha = 0.1f;
  TestActivationOp<float>("LeakyRelu",
                          input_values,
                          [alpha](float x) { return (x >= 0) ? x : alpha * x; },
                          {{"alpha", alpha}}, {});
}

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || defined(USE_COREML)
TEST_F(ActivationOpTest, LeakyRelu_fp16) {
  OpTester test("LeakyRelu", 11);
  float alpha = 0.01f;  // oneDNN set alpha equal to 0.01
  auto formula = [alpha](float x) { return (x >= 0) ? x : alpha * x; };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> bf_X(X.size());
  ConvertFloatToMLFloat16(X.data(), bf_X.data(), (int)X.size());
  std::vector<MLFloat16> bf_Y(Y.size());
  ConvertFloatToMLFloat16(Y.data(), bf_Y.data(), (int)Y.size());

  test.AddInput<MLFloat16>("X", dims, bf_X);
  test.AddOutput<MLFloat16>("Y", dims, bf_Y);
  test.Run();
}
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

TEST_F(ActivationOpTest, ThresholdedRelu) {
  float alpha = 0.1f;
  TestActivationOp<float>(
      "ThresholdedRelu",
      input_values,
      [alpha](float x) { return (x >= alpha) ? x : 0; },
      {{"alpha", alpha}}, {}, true, 10);
}

TEST_F(ActivationOpTest, Selu) {
  static constexpr float alpha = 1.6732f;
  static constexpr float gamma = 1.0507f;

  TestActivationOp<float>("Selu",
                          input_values,
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, Selu_Attributes) {
  static constexpr float alpha = 1.8f;
  static constexpr float gamma = 0.5f;

  TestActivationOp<float>("Selu",
                          input_values,
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, Selu_GH10726) {
  static constexpr float alpha = -2.0f;
  static constexpr float gamma = -3.0f;

  TestActivationOp<float>("Selu",
                          {{1.f, -1.f}},
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, PRelu) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, -4.0f, 0.0f, -9.0f};
  std::vector<float> slopes{1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> outputs;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i]));

  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, inputs);
  test.AddInput<float>("slope", dims, slopes);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, PRelu_Infinity) {
  // Regression test: PRelu must not return NaN for +/-inf inputs.
  // For x > 0, output should be x (so +inf stays +inf).
  // For x <= 0, output should be x * slope (so -inf with positive slope stays -inf).
  OpTester test("PRelu");

  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> inputs{inf, -inf, 5e30f, -2.5f};
  std::vector<float> slopes{0.25f, 0.5f, 0.25f, 0.25f};
  std::vector<float> outputs{inf, -inf, 5e30f, -0.625f};

  std::vector<int64_t> dims{4};
  test.AddInput<float>("X", dims, inputs);
  test.AddInput<float>("slope", dims, slopes);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, PRelu_SingleSlope) {
  auto test = [](bool slope_is_initializer) {
    SCOPED_TRACE(MakeString("slope_is_initializer: ", slope_is_initializer));

    OpTester test("PRelu");

    auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

    auto inputs = {1.0f, 2.0f, -4.0f, 3.0f, 0.0f, 5.0f, -9.0f, 8.0f};
    auto slope = 1.5f;
    std::vector<float> outputs;
    for (auto& input : inputs)
      outputs.push_back(formula(input, slope));

    std::vector<int64_t> dims{2, 2, 2};
    test.AddInput<float>("X", dims, inputs);
    test.AddInput<float>("slope", {}, {slope}, slope_is_initializer);
    test.AddOutput<float>("Y", dims, outputs);
    test.Run();
  };

  test(true /* slope_is_initializer */);
  test(false /* slope_is_initializer */);
}

TEST_F(ActivationOpTest, PRelu_MultiChannel3D) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, 2.0f, -4.0f, 3.0f, 0.0f, 5.0f, -9.0f, 8.0f};
  std::vector<float> slopes{1.0f, -2.0f};
  std::vector<float> outputs;
  constexpr int64_t num_images = 2;
  constexpr int64_t num_channels = 2;
  constexpr int64_t num_pixels = 2;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i / num_pixels % num_channels]));

  std::vector<int64_t> x_dims{num_images, num_channels, num_pixels};
  std::vector<int64_t> slope_dims{num_channels, 1};
  test.AddInput<float>("X", x_dims, inputs);
  test.AddInput<float>("slope", slope_dims, slopes);
  test.AddOutput<float>("Y", x_dims, outputs);
  // QNN has some issue with the broadcast support
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST_F(ActivationOpTest, PRelu_MultiChannel4D) {
  RandomValueGenerator random{2345};

  auto test = [&](bool slope_is_initializer,
                  int64_t n, int64_t c, int64_t h, int64_t w) {
    SCOPED_TRACE(MakeString("slope_is_initializer: ", slope_is_initializer,
                            ", n: ", n, ", c: ", c, ", h: ", h, ", w: ", w));

    OpTester test("PRelu");

    auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

    const std::vector<int64_t> x_dims{n, c, h, w};
    const std::vector<int64_t> slope_dims{c, 1, 1};
    std::vector<float> inputs = random.Uniform<float>(x_dims, -16.0f, 16.0f);
    std::vector<float> slopes = random.Uniform<float>(slope_dims, -1.0f, 1.0f);
    std::vector<float> outputs;
    for (unsigned i = 0; i < inputs.size(); i++) {
      outputs.push_back(formula(inputs[i], slopes[i / (h * w) % c]));
    }

    test.AddInput<float>("X", x_dims, inputs);
    test.AddInput<float>("slope", slope_dims, slopes, slope_is_initializer);
    test.AddOutput<float>("Y", x_dims, outputs);
    // QNN has some issue with the broadcast support
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
  };

  test(true /* slope_is_initializer */, 5, 4, 3, 2);
  test(false, 5, 4, 3, 2);

  test(true, 3, 1, 1, 1);
  test(false, 3, 1, 1, 1);
}

// Edge case: PRelu with +inf and -inf inputs should not produce NaN
TEST_F(ActivationOpTest, PRelu_InfiniteInputs) {
  float pos_inf = std::numeric_limits<float>::infinity();
  float neg_inf = -std::numeric_limits<float>::infinity();

  OpTester test("PRelu");
  test.AddInput<float>("X", {6}, {pos_inf, neg_inf, pos_inf, neg_inf, 5e30f, -2.5f});
  test.AddInput<float>("slope", {6}, {0.25f, 0.5f, 0.0f, 0.25f, 0.25f, 0.25f});
  // +inf >= 0: return +inf
  // -inf < 0: return -inf * 0.5 = -inf
  // +inf >= 0: return +inf
  // -inf < 0: return -inf * 0.25 = -inf
  // 5e30 >= 0: return 5e30
  // -2.5 < 0: return -2.5 * 0.25 = -0.625
  test.AddOutput<float>("Y", {6}, {pos_inf, neg_inf, pos_inf, neg_inf, 5e30f, -0.625f});
  test.Run();
}

// Edge case: PRelu with scalar slope and infinite input
TEST_F(ActivationOpTest, PRelu_InfiniteInputs_ScalarSlope) {
  float pos_inf = std::numeric_limits<float>::infinity();
  float neg_inf = -std::numeric_limits<float>::infinity();

  OpTester test("PRelu");
  test.AddInput<float>("X", {4}, {pos_inf, neg_inf, 1.0f, -1.0f});
  test.AddInput<float>("slope", {1}, {0.5f});
  // +inf >= 0: return +inf
  // -inf < 0: return -inf * 0.5 = -inf
  // 1.0 >= 0: return 1.0
  // -1.0 < 0: return -1.0 * 0.5 = -0.5
  test.AddOutput<float>("Y", {4}, {pos_inf, neg_inf, 1.0f, -0.5f});
  test.Run();
}

// Edge case: PRelu with NaN input propagates NaN
TEST_F(ActivationOpTest, PRelu_NaNPropagation) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();

  OpTester test("PRelu");
  test.AddInput<float>("X", {3}, {nan_val, nan_val, 1.0f});
  test.AddInput<float>("slope", {3}, {0.5f, 0.5f, 0.5f});
  // NaN propagation: NaN in either branch should remain NaN
  test.AddOutput<float>("Y", {3}, {nan_val, nan_val, 1.0f});
  test.Run();
}

// Edge case: -inf * 0 = NaN per IEEE 754 (indeterminate form).
// PRelu spec says y = slope * x for x < 0, so -inf with slope=0 is 0 * -inf = NaN.
TEST_F(ActivationOpTest, PRelu_NegInf_ZeroSlope) {
  float neg_inf = -std::numeric_limits<float>::infinity();
  float nan_val = std::numeric_limits<float>::quiet_NaN();

  OpTester test("PRelu");
  test.AddInput<float>("X", {3}, {neg_inf, neg_inf, 1.0f});
  test.AddInput<float>("slope", {3}, {0.0f, 0.5f, 0.0f});
  // -inf * 0 = NaN (IEEE 754); -inf * 0.5 = -inf; 1.0 >= 0 → 1.0
  test.AddOutput<float>("Y", {3}, {nan_val, neg_inf, 1.0f});
  test.Run();
}

TEST_F(ActivationOpTest, Softplus) {
  TestActivationOp<float>("Softplus",
                          input_values,
                          [](float x) {
                            if (x > 0)
                              return x + log1pf(expf(-x));
                            else
                              return log1pf(expf(x));
                          });
}

TEST_F(ActivationOpTest, Softplus_Opset22) {
  TestActivationOp<float>("Softplus", {{-1.0f, 0.0f, 1.0f, -5.0f, 5.0f, -100.0f, 100.0f}}, [](float x) {
                            if (x > 0)
                              return x + log1pf(expf(-x));
                            else
                              return log1pf(expf(x)); }, {}, {}, /*is_tensorrt_supported=*/true, /*opset_version=*/22);
}

#if defined(USE_CUDA)
template <typename T, typename Reference>
static void CheckCudaActivationSmallPositiveTails(const char* op_name, const Reference& reference) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP unavailable";
  std::vector<T> input;
  std::vector<T> expected;
  for (float value : {-100.0f, -80.0f, -40.0f, -30.0f, -20.0f, -16.0f, -14.0f,
                      -12.0f, -8.0f, -5.0f, -1.0f, -0.125f, 0.0f, 0.125f,
                      1.0f, 5.0f, 8.0f, 14.0f, 20.0f, 40.0f, 100.0f}) {
    input.emplace_back(value);
    const double result = reference(static_cast<double>(value));
    if constexpr (std::is_same_v<T, double>) {
      expected.emplace_back(result);
    } else {
      expected.emplace_back(static_cast<float>(result));
    }
  }
  OpTester tester(op_name, 22);
  const std::vector<int64_t> dimensions{static_cast<int64_t>(input.size())};
  tester.AddInput<T>("X", dimensions, input);
  tester.AddOutput<T>("Y", dimensions, expected);
  tester.SetOutputAbsErr("Y", 0.0f);
  tester.SetOutputRelErr("Y", std::is_same_v<T, double> ? 2e-14f : 2e-6f);
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
static void CheckCudaSoftplusSmallPositiveTails() {
  CheckCudaActivationSmallPositiveTails<T>("Softplus", [](double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
  });
}

template <typename T>
static void CheckCudaSigmoidSmallPositiveTails() {
  CheckCudaActivationSmallPositiveTails<T>("Sigmoid", [](double value) {
    const double exponent = std::exp(-std::abs(value));
    return value > 0 ? 1.0 / (1.0 + exponent) : exponent / (1.0 + exponent);
  });
}

TEST_F(ActivationOpTest, CudaSigmoidFloatSmallPositiveTails) {
  CheckCudaSigmoidSmallPositiveTails<float>();
}

TEST_F(ActivationOpTest, CudaSigmoidDoubleSmallPositiveTails) {
  CheckCudaSigmoidSmallPositiveTails<double>();
}

TEST_F(ActivationOpTest, CudaSigmoidFloat16SmallPositiveTails) {
  CheckCudaSigmoidSmallPositiveTails<MLFloat16>();
}

TEST_F(ActivationOpTest, CudaSigmoidBFloat16SmallPositiveTails) {
  CheckCudaSigmoidSmallPositiveTails<BFloat16>();
}

TEST_F(ActivationOpTest, CudaSoftplusFloatSmallPositiveTails) {
  CheckCudaSoftplusSmallPositiveTails<float>();
}

TEST_F(ActivationOpTest, CudaSoftplusDoubleSmallPositiveTails) {
  CheckCudaSoftplusSmallPositiveTails<double>();
}

TEST_F(ActivationOpTest, CudaSoftplusFloat16SmallPositiveTails) {
  CheckCudaSoftplusSmallPositiveTails<MLFloat16>();
}

TEST_F(ActivationOpTest, CudaSoftplusBFloat16SmallPositiveTails) {
  CheckCudaSoftplusSmallPositiveTails<BFloat16>();
}

TEST_F(ActivationOpTest, Softplus_bfloat16_Opset22) {
  if (!HasCudaEnvironment(530)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }

  OpTester test("Softplus", 22);
  std::vector<float> X = {-1.0f, 0.0f, 1.0f, -5.0f, 5.0f, -100.0f, 100.0f};
  std::vector<float> Y;
  for (float x : X) {
    Y.push_back(x > 0 ? x + log1pf(expf(-x)) : log1pf(expf(x)));
  }
  std::vector<int64_t> dims{static_cast<int64_t>(X.size())};

  test.AddInput<BFloat16>("X", dims, FloatsToBFloat16s(X));
  test.AddOutput<BFloat16>("Y", dims, FloatsToBFloat16s(Y));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA

TEST_F(ActivationOpNoInfTest, Softsign) {
  if constexpr (!SessionOptions::DEFAULT_USE_PER_SESSION_THREADS) {
    GTEST_SKIP() << "Skipping the test";
  }
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1, which exceeds threshold";
  }

  TestActivationOp<float>(
      "Softsign",
      input_values,
      [](float x) { return x / (1 + std::abs(x)); },
      {}, {}, false);  // Disable TensorRT because result mismatches
}

TEST_F(ActivationOpNoInfTest, Softsign_Opset22) {
  if constexpr (!SessionOptions::DEFAULT_USE_PER_SESSION_THREADS) {
    GTEST_SKIP() << "Skipping the test";
  }

  TestActivationOp<float>(
      "Softsign",
      {{-1.0f, 0.0f, 1.0f, -5.0f, 5.0f, -100.0f, 100.0f}},
      [](float x) { return x / (1 + std::abs(x)); },
      {}, {}, /*is_tensorrt_supported=*/false, /*opset_version=*/22);
}

#if defined(USE_CUDA)
TEST_F(ActivationOpNoInfTest, Softsign_bfloat16_Opset22) {
  if (!HasCudaEnvironment(530)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }

  OpTester test("Softsign", 22);
  std::vector<float> X = {-1.0f, 0.0f, 1.0f, -5.0f, 5.0f, -100.0f, 100.0f};
  std::vector<float> Y;
  for (float x : X) {
    Y.push_back(x / (1 + std::abs(x)));
  }
  std::vector<int64_t> dims{static_cast<int64_t>(X.size())};

  test.AddInput<BFloat16>("X", dims, FloatsToBFloat16s(X));
  test.AddOutput<BFloat16>("Y", dims, FloatsToBFloat16s(Y));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA

#if defined(ENABLE_TRAINING_OPS)
TEST(ReluGradInferenceTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "ReluGrad",
      {{"dY", dY}, {"X", x_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], x = params[1];

        return ReluGrad(dy, x);
      },
      {}, 1, kMSDomain);
}

TEST(SigmoidGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "SigmoidGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return SigmoidGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(TanhGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "TanhGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return TanhGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(LeakyReluGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);
  float alpha = 0.5f;

  TestElementwiseGradientOp(
      "LeakyReluGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [alpha](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return LeakyReluGrad(dy, y, alpha);
      },
      {{"alpha", alpha}}, 1, kMSDomain);
}
#endif

// Remove DNNL from running this test because DNNL Gelu op seems not check domain for kernel implementation.
// It will run the DNNL Gelu op which only be part of standard of Gelu-20 op.
// [TODO] Temporarily ignore this test for OpenVINO to avoid an exception due to mishandling of the
// approximate parameter. Re-enable it later when the issue is fixed
#if !defined(USE_DNNL) && !defined(USE_QNN) && !defined(USE_OPENVINO)
TEST_F(ActivationOpTest, ONNX_Gelu) {
  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) { return static_cast<float>(0.5 * x * (1 + erf(x * M_SQRT1_2))); }, {},
      {{"approximate", "none"}}, true, 20);

  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) { return static_cast<float>(0.5 * x * (1 + erf(x * M_SQRT1_2))); },
      {},
      {/*default value of approximate attribute is none */}, true, 20);

  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) {
        return static_cast<float>(0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x))));
      },
      {},
      {{"approximate", "tanh"}}, true, 20);
}

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
TEST_F(ActivationOpTest, Gelu_fp16_tanh) {
  OpTester test("Gelu", 20);
  auto formula = [](float x) {
    return 0.5f * x * (1 + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  };
  const std::vector<float> X = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  std::vector<float> Y;
  Y.reserve(X.size());
  for (float x : X) {
    Y.push_back(formula(x));
  }
  std::vector<int64_t> dims{static_cast<int64_t>(X.size())};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.AddAttribute("approximate", "tanh");

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(ActivationOpTest, Gelu_fp16_erf) {
  OpTester test("Gelu", 20);
  auto formula = [](float x) {
    return static_cast<float>(0.5 * x * (1 + erf(x * M_SQRT1_2)));
  };
  const std::vector<float> X = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  std::vector<float> Y;
  Y.reserve(X.size());
  for (float x : X) {
    Y.push_back(formula(x));
  }
  std::vector<int64_t> dims{static_cast<int64_t>(X.size())};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.AddAttribute("approximate", "none");

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif
#endif

}  // namespace test
}  // namespace onnxruntime
