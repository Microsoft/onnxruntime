// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TestMeanSquaredDistance(const std::vector<int64_t>& X_dims,
                        const std::vector<int64_t>& label_dims,
                        const std::vector<int64_t>& Y_dims,
                        const std::string& reduction,
                        const std::vector<int64_t>* weight_dims) {
  CompareOpTester test("MeanSquaredDistance", 12, kOnnxDomain);
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -100.0f, 100.0f);
  std::vector<float> label_data = random.OneHot<float>(label_dims, label_dims.back());
  test.AddInput<float>("scores", X_dims, X_data);
  test.AddInput<float>("labels", label_dims, label_data);

  if (weight_dims != nullptr) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    test.AddInput<float>("weights", *weight_dims, weight_data);
  }

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  test.AddOutput<float>("output", Y_dims, Y_data);

  test.CompareWithCPU(kCudaExecutionProvider);
}

TEST(CudaKernelTest, MeanSquaredDistance_TinyTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> label_dims{8, 2};
  std::vector<int64_t> weight_dims{8, 2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 2};

  // with weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", &weight_dims);

  // no weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", nullptr);
}

TEST(CudaKernelTest, MeanSquaredDistance_SmallTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> label_dims{8, 20, 10};
  std::vector<int64_t> weight_dims{8, 20, 10};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 20, 10};

  // with weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", &weight_dims);

  // no weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", nullptr);
}

TEST(CudaKernelTest, MeanSquaredDistance_MediumTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> label_dims{8, 1024};
  std::vector<int64_t> weight_dims{8, 1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 1024};

  // with weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", &weight_dims);

  // no weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", nullptr);
}

TEST(CudaKernelTest, MeanSquaredDistance_LargeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> label_dims{4, 512, 30528};
  std::vector<int64_t> weight_dims{4, 512, 30528};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{4, 512, 30528};

  // with weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", &weight_dims);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", &weight_dims);

  // no weights
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "mean", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims, "sum", nullptr);
  TestMeanSquaredDistance(X_dims, label_dims, Y_dims_none, "none", nullptr);
}

}  // namespace test
}  // namespace onnxruntime
