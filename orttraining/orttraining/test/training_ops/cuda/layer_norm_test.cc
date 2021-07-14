// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Eigen/Core"

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

constexpr auto k_epsilon_default = 1e-5f;
constexpr auto k_random_data_min = -10.0f;
constexpr auto k_random_data_max = 10.0f;
const std::string SIMPLIFIED_LAYER_NORM_GRAD_OP = "SimplifiedLayerNormalizationGrad";
const std::string LAYER_NORM_GRAD_OP = "LayerNormalizationGrad";

// The dimensions are split at the specified axis into N (before, exclusive) and M (after, inclusive).
static Status SplitDims(
    const std::vector<int64_t>& dims, int64_t axis,
    std::vector<int64_t>& n_dims, std::vector<int64_t>& m_dims) {
  if (axis < 0) axis += dims.size();
  ORT_RETURN_IF_NOT(0 <= axis && static_cast<decltype(dims.size())>(axis) <= dims.size(),
                    "0 <= axis && axis <= dims.size() was false");
  const auto boundary = dims.begin() + axis;
  n_dims.assign(dims.begin(), boundary);
  m_dims.assign(boundary, dims.end());
  return Status::OK();
}

static void TestLayerNormGrad(
    const std::vector<int64_t>& x_dims,
    const std::string& op,
    int64_t axis = -1,
    double error_tolerance = 1e-4) {
  const std::vector<int64_t>& n_x_m_dims = x_dims;
  std::vector<int64_t> n_dims, m_dims;
  ASSERT_TRUE(SplitDims(n_x_m_dims, axis, n_dims, m_dims).IsOK());

  const auto N = std::accumulate(n_dims.begin(), n_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});
  const auto M = std::accumulate(m_dims.begin(), m_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});

  CompareOpTester test{op.c_str(), 1, kMSDomain};

  test.AddAttribute("axis", axis);

  RandomValueGenerator random{};
  const auto Y_grad_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto X_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto scale_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);

  // these inputs are dependent on X_data
  std::vector<float> mean_data(N);         // mean(X)
  std::vector<float> inv_std_data(N);  // 1 / sqrt(mean(X^2) - mean(X)^2 + epsilon)
  {
    using ConstEigenArrayMap = Eigen::Map<const Eigen::ArrayXX<float>>;
    using EigenRowVectorArrayMap = Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>>;

    ConstEigenArrayMap X{X_data.data(), M, N};
    EigenRowVectorArrayMap mean{mean_data.data(), N};
    EigenRowVectorArrayMap inv_std{inv_std_data.data(), N};

    if (op.compare(SIMPLIFIED_LAYER_NORM_GRAD_OP) != 0) {
      mean = X.colwise().mean();
      inv_std = ((X.colwise().squaredNorm() / X.rows()) - mean.square() + k_epsilon_default).rsqrt();
    } else {
      inv_std = ((X.colwise().squaredNorm() / X.rows()) + k_epsilon_default).rsqrt();
    }
  }

  test.AddInput("Y_grad", n_x_m_dims, Y_grad_data);
  test.AddInput("X", n_x_m_dims, X_data);
  test.AddInput("scale", m_dims, scale_data, true);
  if (op.compare(SIMPLIFIED_LAYER_NORM_GRAD_OP) != 0) {
    test.AddInput("mean", n_dims, mean_data);
  }
  test.AddInput("inv_std", n_dims, inv_std_data);

  const auto X_grad_data = FillZeros<float>(n_x_m_dims);
  const auto scale_grad_data = FillZeros<float>(m_dims);
  const auto bias_grad_data = FillZeros<float>(m_dims);

  test.AddOutput("X_grad", n_x_m_dims, X_grad_data);
  test.AddOutput("scale_grad_data", m_dims, scale_grad_data);
  if (op.compare(SIMPLIFIED_LAYER_NORM_GRAD_OP) != 0) {
    test.AddOutput("bias_grad_data", m_dims, bias_grad_data);
  }

  test.CompareWithCPU(kGpuExecutionProvider, error_tolerance);
}

TEST(CudaKernelTest, LayerNormGrad_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNormGrad(X_dims, LAYER_NORM_GRAD_OP);
}

TEST(CudaKernelTest, LayerNormGrad_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 16, 8};
  const int64_t axis = -2;
  TestLayerNormGrad(X_dims, LAYER_NORM_GRAD_OP, axis);
}

TEST(CudaKernelTest, LayerNormGrad_MidSizeTensor) {
  const std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNormGrad(X_dims, LAYER_NORM_GRAD_OP, 1, 5e-3);
}

TEST(CudaKernelTest, LayerNormGrad_LargeSizeTensor) {
  const std::vector<int64_t> X_dims{16, 512, 1024};
  TestLayerNormGrad(X_dims, LAYER_NORM_GRAD_OP, -1, 5e-3);
}

TEST(CudaKernelTest, SimplifiedLayerNormGrad_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNormGrad(X_dims, SIMPLIFIED_LAYER_NORM_GRAD_OP);
}

TEST(CudaKernelTest, SimplifiedLayerNormGrad_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 16, 8};
  const int64_t axis = -2;
  TestLayerNormGrad(X_dims, SIMPLIFIED_LAYER_NORM_GRAD_OP, axis);
}

TEST(CudaKernelTest, SimplifiedLayerNormGrad_MidSizeTensor) {
  const std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNormGrad(X_dims, SIMPLIFIED_LAYER_NORM_GRAD_OP);
}

TEST(CudaKernelTest, SimplifiedLayerNormGrad_LargeSizeTensor) {
  const std::vector<int64_t> X_dims{16, 512, 1024};
  TestLayerNormGrad(X_dims, SIMPLIFIED_LAYER_NORM_GRAD_OP, -1, 5e-3);
}

static void TestInvertibleLayerNormGrad(
    const std::vector<int64_t>& x_dims,
    int64_t axis = -1,
    double error_tolerance = 1e-4,
    bool test_fp16 = false) {
  const std::vector<int64_t>& n_x_m_dims = x_dims;
  std::vector<int64_t> n_dims, m_dims;
  ASSERT_TRUE(SplitDims(n_x_m_dims, axis, n_dims, m_dims).IsOK());

  const auto N = std::accumulate(n_dims.begin(), n_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});
  const auto M = std::accumulate(m_dims.begin(), m_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});

  CompareOpTester test{"InvertibleLayerNormalizationGrad", 1, kMSDomain};

  test.AddAttribute("axis", axis);

  RandomValueGenerator random{};
  const auto Y_grad_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto X_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto scale_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);
  const auto bias_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);

  // these inputs are dependent on X_data
  std::vector<float> mean_data(N);         // mean(X)
  std::vector<float> inv_std_data(N);  // 1 / sqrt(mean(X^2) - mean(X)^2 + epsilon)
  std::vector<float> Y_data(N*M);
  {
    using ConstEigenArrayMap = Eigen::Map<const Eigen::ArrayXX<float>>;
    using EigenArrayMap = Eigen::Map<Eigen::ArrayXX<float>>;

    ConstEigenArrayMap X{X_data.data(), M, N};

    for (int i = 0; i < N; ++i) {
      mean_data[i] = X.col(i).mean();
      inv_std_data[i] = X.col(i).square().mean() - mean_data[i] * mean_data[i];
    }

    // Compute Y = ((x - mean) * (inv_std)) * scale + bias
    EigenArrayMap Y(Y_data.data(), M, N);

    using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
    using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
    ConstEigenVectorArrayMap mean(mean_data.data(), N);
    EigenVectorArrayMap inv_std(inv_std_data.data(), N);
    inv_std = (inv_std + k_epsilon_default).sqrt().inverse();

    Y = (X.rowwise() - mean.transpose()).rowwise() * inv_std.transpose();

    ConstEigenVectorArrayMap scale(scale_data.data(), M);
    ConstEigenVectorArrayMap bias(bias_data.data(), M);
    Y = (Y.colwise() * scale).colwise() + bias;
  }

  if (test_fp16) {
    std::vector<MLFloat16> Y_grad_data_half(Y_grad_data.size());
    std::vector<MLFloat16> Y_data_half(Y_data.size());
    std::vector<MLFloat16> scale_data_half(scale_data.size());
    std::vector<MLFloat16> bias_data_half(bias_data.size());
    ConvertFloatToMLFloat16(Y_grad_data.data(),Y_grad_data_half.data(), int(Y_grad_data.size()));
    ConvertFloatToMLFloat16(Y_data.data(),Y_data_half.data(), int(Y_data.size()));
    ConvertFloatToMLFloat16(scale_data.data(), scale_data_half.data(), int(scale_data.size()));
    ConvertFloatToMLFloat16(bias_data.data(), bias_data_half.data(), int(bias_data.size()));

    test.AddInput<MLFloat16>("Y_grad", n_x_m_dims, Y_grad_data_half);
    test.AddInput<MLFloat16>("Y", n_x_m_dims, Y_data_half);
    test.AddInput<MLFloat16>("scale", m_dims, scale_data_half, true);
    test.AddInput<MLFloat16>("bias", m_dims, bias_data_half);

    const auto X_grad_data = FillZeros<MLFloat16>(n_x_m_dims);
    const auto scale_grad_data = FillZeros<MLFloat16>(m_dims);
    const auto bias_grad_data = FillZeros<MLFloat16>(m_dims);
    test.AddOutput("X_grad", n_x_m_dims, X_grad_data);
    test.AddOutput("scale_grad_data", m_dims, scale_grad_data);
    test.AddOutput("bias_grad_data", m_dims, bias_grad_data);
  } else {
    test.AddInput("Y_grad", n_x_m_dims, Y_grad_data);
    test.AddInput("Y", n_x_m_dims, Y_data);
    test.AddInput("scale", m_dims, scale_data, true);
    test.AddInput("bias", m_dims, bias_data);

    const auto X_grad_data = FillZeros<float>(n_x_m_dims);
    const auto scale_grad_data = FillZeros<float>(m_dims);
    const auto bias_grad_data = FillZeros<float>(m_dims);
    test.AddOutput("X_grad", n_x_m_dims, X_grad_data);
    test.AddOutput("scale_grad_data", m_dims, scale_grad_data);
    test.AddOutput("bias_grad_data", m_dims, bias_grad_data);
  }
  test.AddInput<float>("inv_std", n_dims, inv_std_data);

  if (test_fp16) {
    test.CompareWithCPU(kGpuExecutionProvider, error_tolerance, error_tolerance);
  } else {
    test.CompareWithCPU(kGpuExecutionProvider, error_tolerance);
  }
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestInvertibleLayerNormGrad(X_dims);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 16, 8};
  const int64_t axis = -2;
  TestInvertibleLayerNormGrad(X_dims, axis);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_MidSizeTensor) {
  const std::vector<int64_t> X_dims{8, 80, 768};
  TestInvertibleLayerNormGrad(X_dims, 1, 5e-3);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_LargeSizeTensor) {
  const std::vector<int64_t> X_dims{16, 512, 1024};
  TestInvertibleLayerNormGrad(X_dims, -1, 5e-3);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_SmallSizeTensor_FP16) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestInvertibleLayerNormGrad(X_dims, -1, 2e-3, true);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_SmallSizeTensor_IntermediateAxis_FP16) {
  const std::vector<int64_t> X_dims{4, 20, 16, 8};
  const int64_t axis = -2;
  TestInvertibleLayerNormGrad(X_dims, axis, 2e-3, true);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_MidSizeTensor_FP16) {
  const std::vector<int64_t> X_dims{8, 80, 768};
  TestInvertibleLayerNormGrad(X_dims, -1, 2e-3, true);
}

TEST(CudaKernelTest, InvertibleLayerNormGrad_LargeSizeTensor_FP16) {
  const std::vector<int64_t> X_dims{16, 512, 1024};
  TestInvertibleLayerNormGrad(X_dims, -1, 2e-3, true);
}

}  // namespace test
}  // namespace onnxruntime
