// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types or limits
// in its parser: axis >=0 && axis < nbDims. Those Tests will fallback to other EPs

TEST(ConcatOpTest, Concat1D_string) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<std::string>("input1", {1}, {"1"});
  test.AddInput<std::string>("input2", {2}, {"2", "3"});
  test.AddInput<std::string>("input3", {4}, {"4", "5", "6", "7"});
  test.AddOutput<std::string>("concat_result", {7}, {"1", "2", "3", "4", "5", "6", "7"});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_int32) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddOutput<int32_t>("concat_result", {7}, {1, 2, 3, 4, 5, 6, 7});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_int32_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-1});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddOutput<int32_t>("concat_result", {7}, {1, 2, 3, 4, 5, 6, 7});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<float>("input1", {1}, {1.0f});
  test.AddInput<float>("input2", {2}, {2.0f, 3.0f});
  test.AddInput<float>("input3", {4}, {4.0f, 5.0f, 6.0f, 7.0f});
  test.AddOutput<float>("concat_result", {7}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<float>("input1", {1}, {1.0f});
  test.AddInput<float>("input2", {2}, {2.0f, 3.0f});
  test.AddInput<float>("input3", {0}, {});
  test.AddOutput<float>("concat_result", {3}, {1.0f, 2.0f, 3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,  //TensorRT: no support for dynamic shape tensor
            kNnapiExecutionProvider});   // NNAPI: concat does not support 0 size input
}

TEST(ConcatOpTest, Concat2D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 4};
  test.AddInput<float>("input1", dims, {11.0f, 12.0f, 13.0f, 14.0f});
  test.AddInput<float>("input2", dims, {21.0f, 22.0f, 23.0f, 24.0f});
  test.AddInput<float>("input3", dims, {31.0f, 32.0f, 33.0f, 34.0f});
  test.AddOutput<float>("concat_result", {3, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f});
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_3) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<float>("input1", {1, 0}, {});
  test.AddInput<float>("input2", {1, 0}, {});
  test.AddInput<float>("input3", {1, 0}, {});
  test.AddOutput<float>("concat_result", {1, 0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,  //TensorRT: no support for dynamic shape tensor
            kNnapiExecutionProvider});   // NNAPI: concat does not support 0 size input
}

// Test Concat of tensors when one of them has dynamic shape
// This is useful for testing EP's own shape inferencing, such as NNAPI EP
TEST(ConcatOpTest, Concat2D_4) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f}, false, &dim_params);
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider});  //TensorRT: no support for dynamic shape tensor
}

TEST(ConcatOpTest, Concat2D_5) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("input1", dims,
                        {111.0f, 112.0f,
                         121.0f, 122.0f});
  test.AddInput<double>("input2", dims,
                        {211.0f, 212.0f,
                         221.0f, 222.0f});
  test.AddOutput<double>("concat_result", {4, 2},
                         {111.0f, 112.0f,
                          121.0f, 122.0f,
                          211.0f, 212.0f,
                          221.0f, 222.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_1_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-3});

  std::vector<int64_t> dims{1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{3, 1, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        211.0f, 212.0f, 213.0f,
                        311.0f, 312.0f, 313.0f});
  test.AddInput<float>("input2", dims,
                       {121.0f, 122.0f, 123.0f,
                        221.0f, 222.0f, 223.0f,
                        321.0f, 322.0f, 323.0f});
  test.AddInput<float>("input3", dims,
                       {131.0f, 132.0f, 133.0f,
                        231.0f, 232.0f, 233.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_3) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("input1", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f});
  test.AddInput<float>("input2", dims,
                       {9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f});
  test.AddOutput<float>("concat_result", {2, 4, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,
                         9.0f, 10.0f,
                         11.0f, 12.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,
                         13.0f, 14.0f,
                         15.0f, 16.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_4) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{2});

  test.AddInput<float>("input1", {1, 3, 3},
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", {1, 3, 4},
                       {211.0f, 212.0f, 213.0f, 214.0f,
                        221.0f, 222.0f, 223.0f, 224.0f,
                        231.0f, 232.0f, 233.0f, 234.0f});
  test.AddInput<float>("input3", {1, 3, 5},
                       {311.0f, 312.0f, 313.0f, 314.0f, 315.0f,
                        321.0f, 322.0f, 323.0f, 324.0f, 325.0f,
                        331.0f, 332.0f, 333.0f, 334.0f, 335.0f});
  test.AddOutput<float>("concat_result", {1, 3, 12},
                        {111.0f, 112.0f, 113.0f, 211.0f, 212.0f, 213.0f, 214.0f, 311.0f, 312.0f, 313.0f, 314.0f, 315.0f,
                         121.0f, 122.0f, 123.0f, 221.0f, 222.0f, 223.0f, 224.0f, 321.0f, 322.0f, 323.0f, 324.0f, 325.0f,
                         131.0f, 132.0f, 133.0f, 231.0f, 232.0f, 233.0f, 234.0f, 331.0f, 332.0f, 333.0f, 334.0f, 335.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_5) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<float>("input1", {1, 3, 2},
                       {111.0f, 112.0f,
                        121.0f, 122.0f,
                        131.0f, 132.0f});
  test.AddInput<float>("input2", {1, 2, 2},
                       {211.0f, 212.0f,
                        221.0f, 222.0f});
  test.AddInput<float>("input3", {1, 4, 2},
                       {311.0f, 312.0f,
                        321.0f, 322.0f,
                        331.0f, 332.0f,
                        341.0f, 342.0f});
  test.AddOutput<float>("concat_result", {1, 9, 2},
                        {111.0f, 112.0f,
                         121.0f, 122.0f,
                         131.0f, 132.0f,
                         211.0f, 212.0f,
                         221.0f, 222.0f,
                         311.0f, 312.0f,
                         321.0f, 322.0f,
                         331.0f, 332.0f,
                         341.0f, 342.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{1, 1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_1_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-3});

  std::vector<int64_t> dims{1, 1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{2});

  std::vector<int64_t> dims{1, 3, 1, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        211.0f, 212.0f, 213.0f,
                        311.0f, 312.0f, 313.0f});
  test.AddInput<float>("input2", dims,
                       {121.0f, 122.0f, 123.0f,
                        221.0f, 222.0f, 223.0f,
                        321.0f, 322.0f, 323.0f});
  test.AddInput<float>("input3", dims,
                       {131.0f, 132.0f, 133.0f,
                        231.0f, 232.0f, 233.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
