// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MathOpTest, Clip_6) {
  OpTester test("Clip", 6);

  test.AddAttribute("min", -10.0f);
  test.AddAttribute("max", 10.0f);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {10.0f, 4.4f, 10.0f,
                         -1.3f, 3.5f, 10.0f,
                         -5.4f, 9.3f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Clip_Default) {
  OpTester test("Clip", 11);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {11.0f, 4.4f, 432.3f,
                         -1.3f, 3.5f, 64.0f,
                         -5.4f, 9.3f, 82.4f});

  // nGraph does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
}

TEST(MathOpTest, Clip) {
  OpTester test("Clip", 11);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {-1.0f, 0.0f, 1.0f,
                        -6.0f, 0.0f, 6.0f,
                        -5.4f, 2.0f, 6.0f});
  test.AddInput<float>("min", {}, {-5});
  test.AddInput<float>("max", {}, {5});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 0.0f, 1.0f,
                         -5.0f, 0.0f, 5.0f,
                         -5.0f, 2.0f, 5.0f});

  // nGraph and Tensorrt does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

TEST(MathOpTest, ClipDimWithZero) {
  std::vector<int64_t> dims{3, 0};  // dim with value of zero should be handled

  OpTester test("Clip"); 
  test.AddInput<float>("X", dims, {});
  test.AddInput<float>("min", {}, {-5});
  test.AddInput<float>("max", {}, {5});
  test.AddOutput<float>("Y", dims, {});

  // nGraph and Tensorrt does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});

  OpTester test6("Clip", 6);  // CUDA only has opset 6 support currently
  test6.AddInput<float>("X", dims, {});
  test6.AddAttribute("min", -10.0f);
  test6.AddAttribute("max", 10.0f);
  test6.AddOutput<float>("Y", dims, {});
  test6.Run();
}

}  // namespace test
}  // namespace onnxruntime
