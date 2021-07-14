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

TEST(BatchNormInternalTest, ForwardTrainingTest) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<float>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<float>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<float>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<float>("running_mean", channel_dims, {-0.1754f, 0.303106f});
  test.AddOutput<float>("running_var", channel_dims, {0.696052f, 1.41316f});
  test.AddOutput<float>("saved_mean", channel_dims, {-0.306f, 0.114562f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.2288f, 0.861317f});

  // exclude CUDA Execution Provider due to flakiness
  // exclude TRT and OpenVINO for same reasons as seen in TestBatchNorm()
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
} // namespace contrib
}  // namespace onnxruntime
