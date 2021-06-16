// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on the tests because axis=0 is not supported

template <typename T>
void RunSliceTest(const std::vector<int64_t>& input_dims,
                  const std::vector<T>& input_vals,
                  const std::vector<int64_t>& starts,
                  const std::vector<int64_t>& ends,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& steps,
                  const std::vector<int64_t>& output_dims,
                  const std::vector<T>& output_vals,
                  bool v10_only = false) {
  // V1-9
  ORT_UNUSED_PARAMETER(steps);
  if (!v10_only) {
    OpTester testv9("Slice", 9);
    testv9.AddAttribute("starts", starts);
    testv9.AddAttribute("ends", ends);
    if (axes.size() != 0)
      testv9.AddAttribute("axes", axes);
    testv9.AddInput<T>("data", input_dims, input_vals);
    testv9.AddOutput<T>("output", output_dims, output_vals);
    testv9.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO EP: Disabled temporarily
  }

  // V10
  OpTester testv10("Slice", 10);
  testv10.AddInput<T>("data", input_dims, input_vals);
  testv10.AddInput<int64_t>("starts", {static_cast<int64_t>(starts.size())}, starts);
  testv10.AddInput<int64_t>("ends", {static_cast<int64_t>(ends.size())}, ends);
  if (axes.size() != 0)
    testv10.AddInput<int64_t>("axes", {static_cast<int64_t>(axes.size())}, axes);
  if (steps.size() != 0)
    testv10.AddInput<int64_t>("steps", {static_cast<int64_t>(steps.size())}, steps);
  testv10.AddOutput<T>("output", output_dims, output_vals);
  testv10.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Slice V1-9 & Slice V10 can both run the following tests
TEST(SliceTest, Slice1D_InvalidStartEndRange) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {3},
                      {2},
                      {0},
                      {},
                      {0},
                      {});
}

TEST(SliceTest, Slice1D_ValidStartEndRange_NoOutput) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {2},
                      {2},
                      {0},
                      {},
                      {0},
                      {});
}

TEST(SliceTest, Slice1D_Regular) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {2},
                      {4},
                      {0},
                      {},
                      {2},
                      {2.0f, 3.0f});
}

TEST(SliceTest, Slice1D_Perf) {
  std::vector<float> input(1000, 2.0f);
  std::vector<float> output(500, 2.0f);
  RunSliceTest<float>({1000},
                      input,
                      {2},
                      {502},
                      {0},
                      {},
                      {500},
                      output);
}

TEST(SliceTest, Slice1D_EndOutOfBounds) {
  RunSliceTest<float>({6},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {0},
                      {10},
                      {},
                      {},
                      {6},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(SliceTest, Slice1D_StartAndEndOutOfBounds) {
  RunSliceTest<float>({6},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {1000},
                      {1001},
                      {},
                      {},
                      {0},
                      {});
}

TEST(SliceTest, Slice2D_StartAndEndOutOfBounds) {
  RunSliceTest<float>({2, 3},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {0, 1000},
                      {10, 1000},
                      {0, 1},
                      {},
                      {2, 0},
                      {});
}

TEST(SliceTest, Slice2D_OneAxis) {
  RunSliceTest<float>({6, 4},
                      {00.0f, 01.0f, 02.0f, 03.0f,
                       10.0f, 11.0f, 12.0f, 13.0f,
                       20.0f, 21.0f, 22.0f, 23.0f,
                       30.0f, 31.0f, 32.0f, 33.0f,
                       40.0f, 41.0f, 42.0f, 43.0f,
                       50.0f, 51.0f, 52.0f, 53.0f},
                      {1},
                      {3},
                      {0},
                      {},
                      {2, 4},
                      {10.0f, 11.0f, 12.0f, 13.0f,
                       20.0f, 21.0f, 22.0f, 23.0f});
}

TEST(SliceTest, Slice2D_TwoAxes) {
  RunSliceTest<float>({6, 4},
                      {00.0f, 01.0f, 02.0f, 03.0f,
                       10.0f, 11.0f, 12.0f, 13.0f,
                       20.0f, 21.0f, 22.0f, 23.0f,
                       30.0f, 31.0f, 32.0f, 33.0f,
                       40.0f, 41.0f, 42.0f, 43.0f,
                       50.0f, 51.0f, 52.0f, 53.0f},
                      {2, 3},
                      {1000, -1},
                      {1, 0},
                      {},
                      {2, 2},
                      {32.0f, 33.0f,
                       42.0f, 43.0f});
}

TEST(SliceTest, Slice2D_TwoAxes_Many) {
  std::vector<uint32_t> input_val(1000);
  for (size_t i = 0; i < input_val.size(); ++i) {
    input_val[i] = (uint32_t)i;
  }
  std::vector<uint32_t> expected_val(98 * 8);
  size_t pos = 0;
  size_t ir, ic;
  for (size_t i = 0; i < input_val.size(); ++i) {
    ir = i / 10;
    ic = i % 10;
    if (ir >= 1 && ic >= 1 && ir < 99 && ic < 9) {
      expected_val[pos++] = input_val[i];
    }
  }

  RunSliceTest<uint32_t>({100, 10},
                         input_val,
                         {1, 1},
                         {99, 9},
                         {0, 1},
                         {},
                         {98, 8},
                         expected_val);
}

TEST(SliceTest, Slice2D_TwoAxes_Many10) {
  std::vector<uint32_t> input_val(1000);
  for (size_t i = 0; i < input_val.size(); ++i) {
    input_val[i] = (uint32_t)i;
  }
  std::vector<uint32_t> expected_val(98 * 8);
  size_t pos = 0;
  size_t ir, ic;
  for (size_t i = 0; i < input_val.size(); ++i) {
    ir = i / 10;
    ic = i % 10;
    if (ir >= 1 && ic >= 1 && ir < 99 && ic < 9) {
      expected_val[pos++] = input_val[i];
    }
  }

  RunSliceTest<uint32_t>({100, 10},
                         input_val,
                         {1, 1},
                         {99, 9},
                         {0, 1},
                         {},
                         {98, 8},
                         expected_val,
                         true);
}

TEST(SliceTest, Slice2D_TwoAxesEque) {
  RunSliceTest<float>({6, 4},
                      {00.0f, 01.0f, 02.0f, 03.0f,
                       10.0f, 11.0f, 12.0f, 13.0f,
                       20.0f, 21.0f, 22.0f, 23.0f,
                       30.0f, 31.0f, 32.0f, 33.0f,
                       40.0f, 41.0f, 42.0f, 43.0f,
                       50.0f, 51.0f, 52.0f, 53.0f},
                      {2, 3},
                      {1000, 3},
                      {1, 0},
                      {},
                      {0, 2},
                      {});
}

TEST(SliceTest, Slice3D) {
  RunSliceTest<float>({3, 3, 3},
                      {111.0f, 112.0f, 113.0f,
                       121.0f, 122.0f, 123.0f,
                       131.0f, 132.0f, 133.0f,

                       211.0f, 212.0f, 213.0f,
                       221.0f, 222.0f, 223.0f,
                       231.0f, 232.0f, 233.0f,

                       311.0f, 312.0f, 313.0f,
                       321.0f, 322.0f, 323.0f,
                       331.0f, 332.0f, 333.0f},
                      {0, 1, 1},
                      {1000, 1000, 1000},
                      {},
                      {},
                      {3, 2, 2},
                      {122.0f, 123.0f,
                       132.0f, 133.0f,

                       222.0f, 223.0f,
                       232.0f, 233.0f,

                       322.0f, 323.0f,
                       332.0f, 333.0f});
}

TEST(SliceTest, Slice1D_Int) {
  RunSliceTest<int32_t>({6},
                        {0L, 1L, 2L, 3L, 4L, 5L},
                        {2},
                        {4},
                        {0},
                        {},
                        {2},
                        {2L, 3L});
}

TEST(SliceTest, Slice1D_String) {
  RunSliceTest<std::string>({6},
                            {"0", "1", "2", "3", "4", "5"},
                            {2},
                            {4},
                            {0},
                            {},
                            {2},
                            {"2", "3"});
}

// Only Slice V10 can run the following tests
TEST(SliceTest, Slice1D_WithPositiveSteps) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {0},
                      {6},
                      {0},
                      {2},
                      {3},
                      {0.0f, 2.0f, 4.0f},
                      true);
}

// In numpy:
// x = np.array([1, 2, 3, 4])
// y = x[-1:-4:-1]
TEST(SliceTest, Slice1D_WithNegativeSteps_Regular) {
  RunSliceTest<float>({4},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {-4},
                      {0},
                      {-1},
                      {3},
                      {4.0f, 3.0f, 2.0f},
                      true);
}

TEST(SliceTest, Slice1D_WithNegativeSteps_EndOutOfBounds_1) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {0},
                      {6},
                      {0},
                      {-1},
                      {0},
                      {},
                      true);
}

TEST(SliceTest, Slice1D_WithNegativeSteps_EndOutOfBounds_2) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {0},
                      {-10},
                      {0},
                      {-1},
                      {1},
                      {0.0f},
                      true);
}

TEST(SliceTest, Slice1D_WithNegativeSteps_ValidStartEndRange) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {5},
                      {0},
                      {0},
                      {-1},
                      {5},
                      {5.0f, 4.0f, 3.0f, 2.0f, 1.0f},
                      true);
}

TEST(SliceTest, Slice1D_WithNegativeSteps_StartOutOfBounds) {
  RunSliceTest<float>({6},
                      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                      {7},
                      {0},
                      {0},
                      {-3},
                      {2},
                      {5.0f, 2.0f},
                      true);
}

TEST(SliceTest, Slice2D_WithPositiveSteps_1) {
  RunSliceTest<float>({2, 4},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {1, 0},
                      {2, 3},
                      {0, 1},
                      {1, 2},
                      {1, 2},
                      {5.0f, 7.0f},
                      true);
}

TEST(SliceTest, Slice2D_WithPositiveSteps_2) {
  RunSliceTest<float>({2, 4},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {0, 1},
                      {-1, 1000},
                      {},  // default axes
                      {},  // default steps
                      {1, 3},
                      {2.0f, 3.0f, 4.0f},
                      true);
}

TEST(SliceTest, Slice2D_WithNegativeSteps_1) {
  RunSliceTest<float>({2, 4},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {1, 0},
                      {2, 3},
                      {0, 1},
                      {-1, -2},
                      {0, 0},
                      {},
                      true);
}

TEST(SliceTest, Slice2D_WithNegativeSteps_2) {
  RunSliceTest<float>({2, 4},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                      {1, 3},
                      {0, 1},
                      {0, 1},
                      {-1, -2},
                      {1, 1},
                      {8.0f},
                      true);
}

TEST(SliceTest, Slice3D_WithPositiveSteps_AllAxes) {
  RunSliceTest<int32_t>({3, 3, 3},
                        {27, 20, 2,
                         4, 26, 11,
                         26, 5, 17,

                         0, 21, 6,
                         22, 13, 29,
                         19, 17, 27,

                         4, 20, 12,
                         3, 9, 24,
                         17, 6, 24},
                        {0, 1, 1},
                        {1000, 1000, 1000},
                        {0, 1, 2},
                        {2, 2, 2},
                        {2, 1, 1},
                        {26, 9},
                        true);
}

TEST(SliceTest, Slice3D_FlattenInnermostDimsIncopy) {
  RunSliceTest<int32_t>({3, 3, 3},
                        {27, 20, 2,
                         4, 26, 11,
                         26, 5, 17,

                         0, 21, 6,
                         22, 13, 29,
                         19, 17, 27,

                         4, 20, 12,
                         3, 9, 24,
                         17, 6, 24},
                        {0, 0, 1},
                        {1000, 1000, 1000},
                        {2, 1, 0},  // reverse the axes to test that's handled correctly by the flattening logic
                        {1, 1, 2},
                        {1, 3, 3},
                        {0, 21, 6,
                         22, 13, 29,
                         19, 17, 27},
                        true);
}

TEST(SliceTest, Slice3D_WithPositiveAndNegativeSteps_SubsetOfAxes_1) {
  RunSliceTest<int32_t>({3, 3, 3},
                        {27, 20, 2,
                         4, 26, 11,
                         26, 5, 17,

                         0, 21, 6,
                         22, 13, 29,
                         19, 17, 27,

                         4, 20, 12,
                         3, 9, 24,
                         17, 6, 24},
                        {1, 4},
                        {1000, 1},
                        {1, 2},
                        {3, -2},
                        {3, 1, 1},
                        {11, 29, 24},
                        true);
}

TEST(SliceTest, Slice3D_WithPositiveAndNegativeSteps_SubsetOfAxes_2) {
  RunSliceTest<int32_t>({3, 3, 3},
                        {27, 20, 2,
                         4, 26, 11,
                         26, 5, 17,

                         0, 21, 6,
                         22, 13, 29,
                         19, 17, 27,

                         4, 20, 12,
                         3, 9, 24,
                         17, 6, 24},
                        {1, 4},
                        {1000, 2},
                        {1, 2},
                        {3, -2},
                        {3, 1, 0},
                        {},
                        true);
}

// Slice for Reversing
// With numeric_limit_max, it means slice to the end of a dimension
// (whichever direction we are stepping)
TEST(SliceTest, Slice1D_ReverseAllAxes_1) {
  RunSliceTest<float>({4},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {std::numeric_limits<int32_t>::max()},
                      {0},
                      {-1},
                      {4},
                      {4.0f, 3.0f, 2.0f, 1.0f},
                      true);
}

// With numeric_limit_min, the end value should be clamped to -1
TEST(SliceTest, Slice1D_ReverseAllAxes_2) {
  RunSliceTest<float>({4},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {std::numeric_limits<int32_t>::min()},
                      {0},
                      {-1},
                      {4},
                      {4.0f, 3.0f, 2.0f, 1.0f},
                      true);
}

// giving an end value < -{dim_value} should also clamp it to -1
TEST(SliceTest, Slice1D_ReverseAllAxes_3) {
  RunSliceTest<float>({4},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {-5},
                      {0},
                      {-1},
                      {4},
                      {4.0f, 3.0f, 2.0f, 1.0f},
                      true);
}

TEST(SliceTest, Slice2D_ReverseAllAxes) {
  RunSliceTest<float>({2, 2},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1, -1},
                      {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()},
                      {0, 1},
                      {-1, -1},
                      {2, 2},
                      {4.0f, 3.0f, 2.0f, 1.0f},
                      true);
}

TEST(SliceTest, Slice2D_ReverseSubsetOfAxes_1) {
  RunSliceTest<float>({2, 2},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {std::numeric_limits<int64_t>::max()},
                      {1},  // axis = 1 only
                      {-1},
                      {2, 2},
                      {2.0f, 1.0f, 4.0f, 3.0f},
                      true);
}

TEST(SliceTest, Slice2D_ReverseSubsetOfAxes_2) {
  RunSliceTest<float>({2, 2},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {std::numeric_limits<int64_t>::max()},  // end of dimension
                      {0},                                    // axis = 0 only
                      {-1},
                      {2, 2},
                      {3.0f, 4.0f, 1.0f, 2.0f},
                      true);
}

// Slice for implicit copy
TEST(SliceTest, Slice2D_ImplicitCopyBySlicingADimensionFully) {
  RunSliceTest<float>({2, 2},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {0},
                      {std::numeric_limits<int64_t>::max()},  // end of dimension
                      {1},                                    // axis = 1 only
                      {1},
                      {2, 2},
                      {1.0f, 2.0, 3.0f, 4.0f},
                      true);
}

TEST(SliceTest, OptionalAxesInputAloneMissing) {
  std::vector<int64_t> input_dims = {6};
  auto input_vals = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::initializer_list<int64_t> starts = {2};
  std::initializer_list<int64_t> ends = {4};
  std::initializer_list<int64_t> steps = {1};
  std::vector<int64_t> output_dims = {2};
  auto output_vals = {2.0f, 3.0f};

  OpTester testv10("Slice", 10);
  testv10.AddInput<float>("data", input_dims, input_vals);
  testv10.AddInput<int64_t>("starts", {static_cast<int64_t>(starts.size())}, starts);
  testv10.AddInput<int64_t>("ends", {static_cast<int64_t>(ends.size())}, ends);
  testv10.AddMissingOptionalInput<int64_t>();
  testv10.AddInput<int64_t>("steps", {static_cast<int64_t>(steps.size())}, steps);
  testv10.AddOutput<float>("output", output_dims, output_vals);
  testv10.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SliceTest, Slice2D_ReverseSubsetOfNegAxes_1) {
  RunSliceTest<float>({2, 2},
                      {1.0f, 2.0f, 3.0f, 4.0f},
                      {-1},
                      {std::numeric_limits<int64_t>::max()},
                      {-1},  // axis = -1 only
                      {-1},
                      {2, 2},
                      {2.0f, 1.0f, 4.0f, 3.0f},
                      true);
}

// test where we provide a subset of axes, we can flatten some dimensions, and we need to skip some input before
// getting to the first value to output.
TEST(SliceTest, Slice5D_SubsetOfAxes_Flatten2Dims_OffsetInput) {
  RunSliceTest<float>({1, 2, 2, 2, 2},
                      {1.f, 2.f, 3.f, 4.f,
                       5.f, 6.f, 7.f, 8.f,
                       -1.f, -2.f, -3.f, -4.f,
                       -5.f, -6.f, -7.f, -8.f},
                      {0, 1, 1, 0},
                      {1, 2, std::numeric_limits<int64_t>::max(), 6},
                      {0, 1, 2, 3},
                      {},
                      {1, 1, 1, 2, 2},
                      {-5.f, -6.f, -7.f, -8.f},
                      true);
}
}  // namespace test
}  // namespace onnxruntime
