// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <cmath>
#include <limits>
#include <type_traits>
#include "gtest/gtest.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/reduction/reduction_test_cases.h"
#include "core/providers/cpu/reduction/reduction_ops.h"
#include "test/util/include/default_providers.h"

#if defined(USE_DML)
#include "core/framework/kernel_registry.h"
#include "test/test_environment.h"
#endif

namespace onnxruntime {
namespace test {

constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NINF = -std::numeric_limits<float>::infinity();
constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
constexpr double DOUBLE_NINF = -std::numeric_limits<double>::infinity();

// Disable TensorRT on some of the tests because the limit in its parser: axis >=0 && axis < nbDims
template <typename OutT>
void TestReduceOp(const std::string& op,
                  int opset_version,
                  const std::vector<int64_t>& input_dims,
                  const std::vector<float>& data,
                  const std::vector<int64_t>& axes,
                  int64_t keepdims,
                  const std::vector<int64_t>& expected_dims,
                  const std::vector<OutT>& expected_data)

{
  OpTester test(op.c_str(), opset_version);

  if (!axes.empty()) {
    if (op.compare("ArgMax") == 0 || op.compare("ArgMin") == 0)
      test.AddAttribute("axis", axes[0]);
    else
      test.AddAttribute("axes", axes);
  }
  test.AddAttribute("keepdims", keepdims);
  test.AddInput<float>("data", input_dims, data);
  test.AddOutput<OutT>("reduced", expected_dims, expected_data);
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kOpenVINOExecutionProvider, kTensorrtExecutionProvider});  // TensorRT,OpenVINO: result differs
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider});  // TensorRT: result differs
#endif
}

// TODO:investigate why it is so slow. It need 12 seconds on an Azure Standard F48s_v2 (48 vcpus, 96 GiB memory)
//  machine in RelWithDebInfo build mode, but only 2 seconds on my local dev machine(4 cores).
#ifdef NDEBUG
TEST(ReductionOpTest, ReductionVariationTest) {
  const std::vector<float>& input_data = testcases.input_data;
  const std::vector<int64_t>& input_dims = testcases.input_dims;
  OpAttributesResultMap& opAttributesResultMap = testcases.map_op_attribute_expected;

  for (auto a : opAttributesResultMap) {
    const ReductionAttribute& attributes = std::get<0>(a.second);
    const std::vector<int64_t> expected_dims = std::get<1>(a.second);
    if (a.first.compare("ArgMax") == 0 || a.first.compare("ArgMin") == 0) {
      std::vector<int64_t> expected_values;
      for (auto v : std::get<2>(a.second))
        expected_values.push_back(static_cast<int64_t>(v));
      TestReduceOp<int64_t>(a.first, 7, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
      TestReduceOp<int64_t>(a.first, 11, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
    } else {
      const std::vector<float> expected_values = std::get<2>(a.second);
      TestReduceOp<float>(a.first, 7, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
      TestReduceOp<float>(a.first, 11, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
    }
  }
}
#endif

TEST(ReductionOpTest, ReduceL1_default_axes_keepdims) {
  OpTester test("ReduceL1");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL1_default_axes_keepdims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL1", 13);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 1, 1}, MakeBFloat16({78.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceL1_do_not_keep_dims) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL1_do_not_keep_dims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL1", 13);
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 2}, MakeBFloat16({3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceL1_do_not_keep_dims_2) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL1_do_not_keep_dims_2_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL1", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3},
                          MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.AddOutput<BFloat16>("reduced", {}, MakeBFloat16({6.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceL1_keepdims) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2, 1}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run();
}
#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL1_keepdims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL1", 13);
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 2, 1}, MakeBFloat16({3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceL1) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_double) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_int32) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_int64) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL1_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL1", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({33.0f, 45.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceL10DTensor) {
  OpTester test("ReduceL1");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_0DTensor_negative_input) {
  OpTester test("ReduceL1");
  test.AddInput<float>("data", {}, {-3.0f});
  test.AddOutput<float>("reduced", {}, {3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_int32_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<int32_t>("data", {1, 1}, {-4});
  test.AddOutput<int32_t>("reduced", {1}, {4});
  // TensorRT does not support int32 ReduceL1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_int64_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<int64_t>("data", {1, 1}, {-4});
  test.AddOutput<int64_t>("reduced", {1}, {4});
  // TensorRT does not support int64 ReduceL1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_float_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {1, 1}, {-4.0f});
  test.AddOutput<float>("reduced", {1}, {4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_double_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {1, 1}, {-4.0});
  test.AddOutput<double>("reduced", {1}, {4.0});
  // TensorRT does not support double ReduceL1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceL1_half_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("data", {1, 1}, FloatsToMLFloat16s({-4.0f}));
  test.AddOutput<MLFloat16>("reduced", {1}, FloatsToMLFloat16s({4.0f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceL1_float_multi_element_all_negative) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2}, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
  test.AddOutput<float>("reduced", {2}, {9.0f, 12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_int32_keepdims_singleton_axis_negative_input) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<int32_t>("data", {1, 2}, {-3, -7});
  test.AddOutput<int32_t>("reduced", {1, 2}, {3, 7});
  // TensorRT does not support int32 ReduceL1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// INT_MIN abs is UB in two's complement; saturating_abs should clamp to INT_MAX.
TEST(ReductionOpTest, ReduceL1_int32_INT_MIN) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<int32_t>("data", {1}, {std::numeric_limits<int32_t>::min()});
  // abs(INT_MIN) overflows; we expect saturation to INT_MAX.
  test.AddOutput<int32_t>("reduced", {}, {std::numeric_limits<int32_t>::max()});
  // Exclude EPs that may not implement the CPU's saturating INT_MIN behavior.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

// Summation overflow: summing large positive values should saturate to INT_MAX rather than wrap.
TEST(ReductionOpTest, ReduceL1_int32_summation_overflow) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  // 3 * 1,000,000,000 = 3e9 > INT32_MAX (2,147,483,647)
  test.AddInput<int32_t>("data", {3}, {1000000000, 1000000000, 1000000000});
  test.AddOutput<int32_t>("reduced", {}, {std::numeric_limits<int32_t>::max()});
  // CUDA now uses double accumulation (same as CPU), so saturation matches.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_default_axes_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {25.49509757f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_default_axes_do_not_keep_dims) {
  OpTester test("ReduceL2");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {25.49509757f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_do_not_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {2.23606798f, 5.0f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_do_not_keepdims_2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.741657387f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {1, 1}, {-4.0f});
  test.AddOutput<float>("reduced", {1}, {4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_double_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {1, 1}, {-4.0});
  test.AddOutput<double>("reduced", {1}, {4.0});
  // TensorRT does not support double ReduceL2.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_int32_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<int32_t>("data", {1, 1}, {-4});
  test.AddOutput<int32_t>("reduced", {1}, {4});
  // TensorRT/OpenVINO do not support int32 ReduceL2.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceL2_half_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("data", {1, 1}, FloatsToMLFloat16s({-4.0f}));
  test.AddOutput<MLFloat16>("reduced", {1}, FloatsToMLFloat16s({4.0f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceL2_float_multi_element_all_negative) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  // sqrt((-1)^2 + (-3)^2 + (-5)^2) = sqrt(35) ≈ 5.916
  // sqrt((-2)^2 + (-4)^2 + (-6)^2) = sqrt(56) ≈ 7.483
  test.AddInput<float>("data", {3, 2}, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
  test.AddOutput<float>("reduced", {2}, {5.91607978f, 7.48331477f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_float_keepdims_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<float>("data", {1, 2}, {-3.0f, -7.0f});
  test.AddOutput<float>("reduced", {1, 2}, {3.0f, 7.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_int32_keepdims_singleton_axis_negative_input) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<int32_t>("data", {1, 2}, {-3, -7});
  test.AddOutput<int32_t>("reduced", {1, 2}, {3, 7});
  // TensorRT/OpenVINO do not support int32 ReduceL2.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// INT_MIN: sqrt(INT_MIN^2) overflows; should saturate to INT_MAX.
TEST(ReductionOpTest, ReduceL2_int32_INT_MIN) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<int32_t>("data", {1}, {std::numeric_limits<int32_t>::min()});
  // sqrt(INT_MIN^2) = |INT_MIN| which overflows int32; saturate to INT_MAX.
  test.AddOutput<int32_t>("reduced", {}, {std::numeric_limits<int32_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

// Squaring overflow: large values squared exceed int32 range; double accumulator avoids UB.
TEST(ReductionOpTest, ReduceL2_int32_squaring_overflow) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  // sqrt(50000^2 + 50000^2) = 50000*sqrt(2) ≈ 70710
  test.AddInput<int32_t>("data", {2}, {50000, 50000});
  test.AddOutput<int32_t>("reduced", {}, {70710});
  // Both CPU and CUDA use double accumulator to avoid integer overflow UB.
  // For these test values (50000^2 = 2.5e9 < 2^53), squaring is exact in double.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2, 1}, {2.23606798f, 5.0f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {15.71623325f, 20.07485962f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_double) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {2}, {15.71623325f, 20.07485962f});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceL2_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceL2", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {2}, MakeBFloat16({15.71623325f, 20.07485962f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceL2_int32) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {2}, {15, 20});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Int32 not allowed as input to this layer
}

TEST(ReductionOpTest, ReduceL2_int64) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2}, {15, 20});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Int64 not allowed as input to this layer
}

TEST(ReductionOpTest, ReduceL20DTensor) {
  OpTester test("ReduceL2");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_0DTensor_negative_input) {
  OpTester test("ReduceL2");
  test.AddInput<float>("data", {}, {-5.0f});
  test.AddOutput<float>("reduced", {}, {5.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceLogSum) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2},
                        {1.38629436f, 1.79175949f,
                         2.48490667f, 2.6390574f,
                         2.99573231f, 3.09104252f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogOps_Opset28_FloatAxesInput) {
  const std::vector<std::pair<std::string, std::vector<float>>> cases{
      {"ReduceLogSum", {1.38629436f, 1.79175949f}},
      {"ReduceLogSumExp", {3.12692801f, 4.12692801f}},
  };

  for (const auto& [op, expected] : cases) {
    OpTester test(op, 28);
    test.SetAllowUnreleasedOnnxOpset();
    test.AddInput<float>("data", {2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
    test.AddInput<int64_t>("axes", {1}, {1});
    test.AddAttribute<int64_t>("keepdims", 0);
    test.AddOutput<float>("reduced", {2}, expected);
    test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
  }
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceLogOps_Opset28_Cuda) {
  for (int opset : {27, 28}) {
    for (const auto& [op, expected] :
         std::vector<std::pair<std::string, std::vector<float>>>{
             {"ReduceLogSum", {1.38629436f, 1.79175949f}},
             {"ReduceLogSumExp", {3.12692801f, 4.12692801f}}}) {
      auto provider = DefaultCudaExecutionProvider();
      if (!provider) {
        GTEST_SKIP() << "CUDA execution provider is not available.";
      }
      OpTester test(op, opset);
      test.SetAllowUnreleasedOnnxOpset();
      test.AddInput<float>("data", {2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
      test.AddInput<int64_t>("axes", {1}, {1});
      test.AddAttribute<int64_t>("keepdims", 0);
      test.AddOutput<float>("reduced", {2}, expected);
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
    }
  }
}
#endif

TEST(ReductionOpTest, ReduceLogOps_Opset28_Scalar) {
  for (const auto& [op, expected] :
       std::vector<std::pair<std::string, float>>{{"ReduceLogSum", std::log(2.0f)}, {"ReduceLogSumExp", 2.0f}}) {
    OpTester test(op, 28);
    test.SetAllowUnreleasedOnnxOpset();
    test.AddInput<float>("data", {}, {2.0f});
    test.AddOptionalInputEdge<int64_t>();
    test.AddOutput<float>("reduced", {}, {expected});
    test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
  }
}

#if defined(USE_DML)
TEST(ReductionOpTest, ReduceLogOps_Opset28_Dml) {
  for (int opset : {27, 28}) {
    for (const auto& [op, expected] :
         std::vector<std::pair<std::string, std::vector<float>>>{
             {"ReduceLogSum", {1.38629436f, 1.79175949f}},
             {"ReduceLogSumExp", {3.12692801f, 4.12692801f}}}) {
      auto provider = DefaultDmlExecutionProvider();
      if (!provider) {
        GTEST_SKIP() << "DML execution provider is not available.";
      }
      const auto registry = provider->GetKernelRegistry();
      ASSERT_NE(registry, nullptr);
      const KernelCreateInfo* kernel_info = nullptr;
      const KernelRegistry::TypeConstraintMap type_constraints{{"T", DataTypeImpl::GetTensorType<float>()}};
      const int since_version = opset == 28 ? 28 : 18;
      ASSERT_STATUS_OK(registry->TryFindKernel(kDmlExecutionProvider, op, kOnnxDomain, since_version, type_constraints,
                                               DefaultLoggingManager().DefaultLogger(), &kernel_info));
      ASSERT_NE(kernel_info, nullptr);
      EXPECT_EQ(kernel_info->kernel_def->SinceVersion().first, since_version);
      if (opset == 28) {
        EXPECT_EQ(kernel_info->kernel_def->TypeConstraints().at("T"),
                  (std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(),
                                           DataTypeImpl::GetTensorType<MLFloat16>()}));
      }
      OpTester test(op, opset);
      test.SetAllowUnreleasedOnnxOpset();
      test.AddInput<float>("data", {2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
      test.AddInput<int64_t>("axes", {1}, {1});
      test.AddAttribute<int64_t>("keepdims", 0);
      test.AddOutput<float>("reduced", {2}, expected);
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
    }
  }
}
#endif

TEST(ReductionOpTest, ReduceLogOps_Opset28_RejectIntegerData) {
  for (const char* op : {"ReduceLogSum", "ReduceLogSumExp"}) {
    OpTester test(op, 28);
    test.SetAllowUnreleasedOnnxOpset();
    test.AddInput<int32_t>("data", {2, 2}, {1, 3, 2, 4});
    test.AddInput<int64_t>("axes", {1}, {1});
    test.AddAttribute<int64_t>("keepdims", 0);
    test.AddOutput<int32_t>("reduced", {2}, {0, 0});
    test.Config(OpTester::ExpectResult::kExpectFailure, "Type Error")
        .ConfigEp(DefaultCpuExecutionProvider())
        .RunWithConfig();
  }
}

TEST(ReductionOpTest, ReduceLogOps_Opset27_AcceptsIntegerData) {
  const std::vector<std::pair<std::string, std::vector<int32_t>>> cases{
      {"ReduceLogSum", {1, 1}},
      {"ReduceLogSumExp", {3, 4}},
  };

  for (const auto& [op, expected] : cases) {
    OpTester test(op, 27);
    test.AddInput<int32_t>("data", {2, 2}, {1, 3, 2, 4});
    test.AddInput<int64_t>("axes", {1}, {1});
    test.AddAttribute<int64_t>("keepdims", 0);
    test.AddOutput<int32_t>("reduced", {2}, expected);
    test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
  }
}

TEST(ReductionOpTest, ReduceLogSum_double) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,
                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {3, 1, 2},
                         {1.38629436f, 1.79175949f,
                          2.48490667f, 2.6390574f,
                          2.99573231f, 3.09104252f});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceLogSum_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceLogSum", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,
                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 1, 2},
                           MakeBFloat16({1.38629436f, 1.79175949f,
                                         2.48490667f, 2.6390574f,
                                         2.99573231f, 3.09104252f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  // USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceLogSum_samesize) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddOutput<float>("reduced", {3, 2, 1},
                        {0.0f, 0.6931471f,
                         1.09861230f, 1.38629436f,
                         1.60943794f, 1.79175949f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSum_do_not_keepdims_2) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {1.79175947f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumAxes01) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4, 5},
                       {0.5172141f, 0.36681905f, 0.7489675f, 0.21859895f, 0.6378839f,
                        0.6120873f, 0.74698675f, 0.87222993f, 0.23919299f, 0.4877085f,
                        0.58392614f, 0.56973755f, 0.28744474f, 0.56500393f, 0.13348383f,
                        0.06321382f, 0.20588198f, 0.08760026f, 0.9710815f, 0.6781033f,
                        0.38189054f, 0.9127731f, 0.21229997f, 0.7597165f, 0.36321816f,
                        0.18440539f, 0.44839138f, 0.888846f, 0.54862875f, 0.15642975f,
                        0.5046317f, 0.6035792f, 0.42172152f, 0.55201846f, 0.8684674f,
                        0.8725194f, 0.89469117f, 0.88513845f, 0.48750868f, 0.2512843f,
                        0.54381144f, 0.970685f, 0.44817686f, 0.7655562f, 0.64186585f,
                        0.8696393f, 0.91110307f, 0.12956737f, 0.9199235f, 0.26789218f,
                        0.25372583f, 0.6147827f, 0.67517287f, 0.74066293f, 0.6317299f,
                        0.70738846f, 0.27802366f, 0.15887405f, 0.95882577f, 0.23314993f});
  test.AddOutput<float>("reduced", {5},
                        {1.8073791f, 2.0180254f, 1.7606194f, 2.0446842f, 1.6773242f});

  test.Run();
}

TEST(ReductionOpTest, ReduceLogSum0DTensor) {
  OpTester test("ReduceLogSum");
  test.AddInput<float>("data", {}, {2.f});
  test.AddOutput<float>("reduced", {}, {0.693147f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceLogSumExp1DTensor) {
  OpTester test("ReduceLogSumExp");
  test.AddInput<float>("data", {24},
                       {-4.025670051574707f, -9.444348335266113f, -3.1193981170654297f,
                        -5.943697929382324f, -0.3701804578304291f, -4.397126197814941f,
                        -6.605968475341797f, -5.534277439117432f, -7.361471176147461f,
                        -1.9987547397613525f, -9.093968391418457f, -8.693618774414062f,
                        -8.416788101196289f, -1.010741114616394f, -9.814584732055664f,
                        -9.725259780883789f, -9.157071113586426f, -0.001698818989098072f,
                        -9.963415145874023f, -5.991659641265869f, -6.180599689483643f,
                        -1.2336505651474f, -0.44234341382980347f, -6.990072250366211f});
  test.AddOutput<float>("reduced", {1}, {1.1666961908340454f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {60.00671387f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {1, 1, 1}, {60.00671387});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_do_not_keep_dims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {60.00671387f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_do_not_keep_dims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {}, {60.00671387});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {20.0f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 2}, {20.0, 2.31326175, 40.00004578, 2.31326175, 60.00671387, 2.31326175});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_2) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.40760596f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_2_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3},
                        {1.0, 2.0, 3.0});
  test.AddOutput<double>("reduced", {}, {3.40760596});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {20.0f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 1, 2}, {20.0, 2.31326175, 40.00004578, 2.31326175, 60.00671387, 2.31326175});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {10.33174133f, 12.33174133f});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceLogSumExp_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceLogSumExp", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({10.33174133f, 12.33174133f}));
  test.Run();
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceLogSumExp_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {10.33174133, 12.33174133});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_float_no_reduction) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddOutput<float>("reduced", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_float_no_reduction_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.Run();
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceLogSumExp_half) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<MLFloat16>("data", {3, 2, 2},
                           FloatsToMLFloat16s({1.0f, 2.0f,
                                               3.0f, 4.0f,

                                               5.0f, 6.0f,
                                               7.0f, 8.0f,

                                               9.0f, 10.0f,
                                               11.0f, 12.0f}));
  test.AddOutput<MLFloat16>("reduced", {1, 2, 1}, FloatsToMLFloat16s({10.33174133f, 12.33174133f}));
  test.Run();
}
#endif  // defined(USE_CUDA)

TEST(ReductionOpTest, ReduceLogSumExp_int32) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {10, 12});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_int64) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {10, 12});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp0DTensor) {
  OpTester test("ReduceLogSumExp");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceLogSumExp0DTensor_double) {
  OpTester test("ReduceLogSumExp");
  test.AddInput<double>("data", {}, {2});
  test.AddOutput<double>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMax_default_axes_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {60.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMax");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {60.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch //TensorRT: axis must be 0
}

TEST(ReductionOpTest, test_bool_ReduceMax_0) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {-1, 1});
  test.AddOutput<bool>("reduced", {2}, {true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_1) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {-1, 1});
  test.AddOutput<bool>("reduced", {2}, {false, false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_2) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {-1, 1});
  test.AddOutput<bool>("reduced", {2, 1, 1}, {true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      }

  );
}

TEST(ReductionOpTest, test_bool_ReduceMin_3) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {-1, 1});
  test.AddOutput<bool>("reduced", {2, 1, 1}, {false, false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_4) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {2, 1});
  test.AddOutput<bool>("reduced", {2}, {true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_5) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {2, 1});
  test.AddOutput<bool>("reduced", {2}, {false, false});
  test.Run();
}

TEST(ReductionOpTest, test_bool_ReduceMax_6) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {2, 1});
  test.AddOutput<bool>("reduced", {2, 1, 1}, {true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_7) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {2}, {2, 1});
  test.AddOutput<bool>("reduced", {2, 1, 1}, {false, false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_8) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<bool>("reduced", {3, 2}, {false, true, true, true, false, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_9) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<bool>("reduced", {3, 2}, {false, false, false, true, false, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_10) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<bool>("reduced", {1, 3, 2}, {false, true, true, true, false, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_11) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<bool>("reduced", {1, 3, 2}, {false, false, false, true, false, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_12) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {2});
  test.AddOutput<bool>("reduced", {2, 3}, {false, true, true, true, true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_13) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {2});
  test.AddOutput<bool>("reduced", {2, 3}, {false, true, false, false, false, false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_14) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {2});
  test.AddOutput<bool>("reduced", {2, 3, 1}, {false, true, true, true, true, true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_15) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddInput<int64_t>("axes", {1}, {2});
  test.AddOutput<bool>("reduced", {2, 3, 1}, {false, true, false, false, false, false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_16) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddOutput<bool>("reduced", {}, {true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_17) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddOutput<bool>("reduced", {}, {false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMax_18) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddOutput<bool>("reduced", {1, 1, 1}, {true});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, test_bool_ReduceMin_19) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<bool>("data", {2, 3, 2}, {false, false, true, true, false, true, false, true, false, true, false, true});
  test.AddOutput<bool>("reduced", {1, 1, 1}, {false});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kOpenVINOExecutionProvider,
      });
}

TEST(ReductionOpTest, ReduceMax_do_not_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {20.0f, 2.0f, 40.0f, 2.0f, 60.0f, 2.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_do_not_keepdims_2) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {5.0f, 1.0f, 20.0f});
  test.AddOutput<float>("reduced", {}, {20.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMax_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {20.0f, 2.0f, 40.0f, 2.0f, 60.0f, 2.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMaxAxesInitializerOpset18) {
  OpTester test("ReduceMax", 18);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {1, 2}, true);
  test.AddOutput<float>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceMax_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMax", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 1, 1}, MakeBFloat16({4.0f, 8.0f, 12.0f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceMax_double) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_COREML)
TEST(ReductionOpTest, ReduceMax_half) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<MLFloat16>("data", {3, 2, 2},
                           FloatsToMLFloat16s({1.0f, 2.0f,
                                               3.0f, 4.0f,

                                               5.0f, 6.0f,
                                               7.0f, 8.0f,

                                               9.0f, 10.0f,
                                               11.0f, 12.0f}));
  test.AddOutput<MLFloat16>("reduced", {3, 1, 1}, FloatsToMLFloat16s({4.0f, 8.0f, 12.0f}));
  test.Run();
}
#endif  // defined(USE_CUDA)

TEST(ReductionOpTest, ReduceMax_int32) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {3, 1, 1}, {4, 8, 12});

#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
#endif
}

TEST(ReductionOpTest, ReduceMax_int64) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 1}, {4, 8, 12});
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
#endif
}

TEST(ReductionOpTest, ReduceMax_int8) {
  OpTester test("ReduceMax", 12);
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int8_t>("reduced", {3, 1, 1}, {4, 8, 12});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ReduceMax_uint8) {
  OpTester test("ReduceMax", 12);
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<uint8_t>("reduced", {3, 1, 1}, {4, 8, 12});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ReduceMax0DTensor) {
  OpTester test("ReduceMax");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // can't access out of range dim in OpenVINO-EP
}

TEST(ReductionOpTest, ReduceMean_default_axes_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {18.25f});
  test.Run();
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_default_axes_keepdims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif

  OpTester test("ReduceMean", 13);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({5.0f, 1.0f,
                                        20.0f, 2.0f,

                                        30.0f, 1.0f,
                                        40.0f, 2.0f,

                                        55.0f, 1.0f,
                                        60.0f, 2.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 1, 1}, MakeBFloat16({18.25f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceMean_default_axes_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {1, 1, 1}, {18.25});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {18.25f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_default_axes_do_not_keep_dims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({5.0f, 1.0f,
                                        20.0f, 2.0f,

                                        30.0f, 1.0f,
                                        40.0f, 2.0f,

                                        55.0f, 1.0f,
                                        60.0f, 2.0f}));
  test.AddOutput<BFloat16>("reduced", {}, MakeBFloat16({18.25f}));
  test.Run();
}
#endif
TEST(ReductionOpTest, ReduceMean_default_axes_do_not_keep_dims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {}, {18.25});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif
  test.Run();
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_do_not_keepdims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({5.0f, 1.0f,
                                        20.0f, 2.0f,

                                        30.0f, 1.0f,
                                        40.0f, 2.0f,

                                        55.0f, 1.0f,
                                        60.0f, 2.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 2}, MakeBFloat16({12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0f,
                         40.0, 2.0f,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 2}, {12.5, 1.5, 35.0, 1.5, 57.5, 1.5});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}
#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3},
                          MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.AddOutput<BFloat16>("reduced", {}, MakeBFloat16({2.0f}));
  test.Run();
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3},
                        {1.0, 2.0, 3.0});
  test.AddOutput<double>("reduced", {}, {2.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif
  test.Run();
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_keepdims_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({5.0f, 1.0f,
                                        20.0f, 2.0f,

                                        30.0f, 1.0f,
                                        40.0f, 2.0f,

                                        55.0f, 1.0f,
                                        60.0f, 2.0f}));
  test.AddOutput<BFloat16>("reduced", {3, 1, 2}, MakeBFloat16({12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f}));
  test.Run();
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceMean_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {5.0, 1.0,
                         20.0, 2.0,

                         30.0, 1.0,
                         40.0, 2.0,

                         55.0, 1.0,
                         60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 1, 2}, {12.5, 1.5, 35.0, 1.5, 57.5, 1.5});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif
  test.Run();
}

TEST(ReductionOpTest, ReduceMean) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5f, 7.5f});

  test.Run();
}

TEST(ReductionOpTest, ReduceMeanAxesInitializerOpset18) {
  OpTester test("ReduceMean", 18);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true);
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5f, 7.5f});

  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

#ifdef USE_CUDA
TEST(ReductionOpTest, ReduceMean_bfloat16_cuda_opset13) {
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,
                                        5.0f, 6.0f,
                                        7.0f, 8.0f,
                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({5.5f, 7.5f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMean_bfloat16_cuda_opset18) {
  OpTester test("ReduceMean", 18);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,
                                        5.0f, 6.0f,
                                        7.0f, 8.0f,
                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true);
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({5.5f, 7.5f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({5.5f, 7.5f}));

  test.Run();
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceMean_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {5.5, 7.5});

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_int32) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {10, 20,
                          30, 40,

                          50, 60,
                          70, 80,

                          90, 100,
                          110, 120});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {55, 75});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_int64) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {10, 20,
                          30, 40,

                          50, 60,
                          70, 80,

                          90, 100,
                          110, 120});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {55, 75});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_axes_input) {
  OpTester test("ReduceMean", 18, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1, 2,
                        3, 4,

                        5, 6,
                        7, 8,

                        9, 10,
                        11, 12});
  test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, 2}, true);
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5, 7.5});

  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_axes_input_initializer) {
  OpTester test("ReduceMean", 18, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, true);
  test.AddOutput<float>("reduced", {1, 2}, {2.0f, 3.0f});

  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean0DTensor) {
  OpTester test("ReduceMean");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean0DTensor_double) {
  OpTester test("ReduceMean");
  test.AddInput<double>("data", {}, {2});
  test.AddOutput<double>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean_keepdims_results_in_noop) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {1, 3}, {1.0, 2.0, 3.0});
  test.AddOutput<float>("reduced", {1, 3}, {1.0, 2.0, 3.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_keepdims_results_in_noop_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {1, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceMean_keepdims_results_in_shape_change) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 3}, {1.0, 2.0, 3.0});
  test.AddOutput<float>("reduced", {3}, {1.0, 2.0, 3.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#ifdef USE_DNNL
TEST(ReductionOpTest, ReduceMean_keepdims_results_in_shape_change_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {1, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.AddOutput<BFloat16>("reduced", {3}, MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.Run();
}
#endif

TEST(ReductionOpTest, ReduceMin_default_axes_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_default_axes_do_not_keep_dims_2D) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_do_not_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {5.0f, 1.0f, 30.0f, 1.0f, 55.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_do_not_keepdims_2) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {5.0f, 1.0f, 20.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {5.0f, 1.0f, 30.0f, 1.0f, 55.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {1.0f, 3.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMinAxesInitializerOpset18) {
  OpTester test("ReduceMin", 18);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true);
  test.AddOutput<float>("reduced", {1, 2, 1}, {1.0f, 3.0f});
  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceMin_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceMin", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({1.0f, 3.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  // USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceMin_double) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {1, 2, 1}, {1.0f, 3.0f});
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_COREML)
TEST(ReductionOpTest, ReduceMin_half) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<MLFloat16>("data", {3, 2, 2},
                           FloatsToMLFloat16s({1.0f, 2.0f,
                                               3.0f, 4.0f,

                                               5.0f, 6.0f,
                                               7.0f, 8.0f,

                                               9.0f, 10.0f,
                                               11.0f, 12.0f}));
  test.AddOutput<MLFloat16>("reduced", {1, 2, 1}, FloatsToMLFloat16s({1.0f, 3.0f}));
  test.Run();
}
#endif  // defined(USE_CUDA)

TEST(ReductionOpTest, ReduceMin_int32) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_int64) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_int8) {
  OpTester test("ReduceMin", 12);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int8_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin_uint8) {
  OpTester test("ReduceMin", 12);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<uint8_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin0DTensor) {
  OpTester test("ReduceMin");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

// Opset 13
TEST(ReductionOpTest, ReduceSumAxesInitializerOpset13) {
  OpTester test("ReduceSum", 13);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true);
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_double) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {33.0, 45.0});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_axes01) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_axes02) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_int32) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_COREML)
TEST(ReductionOpTest, ReduceSumHalfHalf) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});

  std::vector<float> data = {1.0f, 2.0f,
                             3.0f, 4.0f,

                             5.0f, 6.0f,
                             7.0f, 8.0f,

                             9.0f, 10.0f,
                             11.0f, 12.0f};
  std::vector<MLFloat16> data_half(12);
  ConvertFloatToMLFloat16(data.data(), data_half.data(), 12);

  std::vector<float> result = {36.0f, 42.0f};
  std::vector<MLFloat16> result_half(2);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 2);

  test.AddInput<MLFloat16>("data", {3, 2, 2}, data_half);
  test.AddOutput<MLFloat16>("reduced", {2}, result_half);
  test.Run();
}

TEST(ReductionOpTest, ReduceSumHalfHalf_2) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});

  std::vector<float> data = {1.0f, 2.0f,
                             3.0f, 4.0f,

                             5.0f, 6.0f,
                             7.0f, 8.0f,

                             9.0f, 10.0f,
                             11.0f, 12.0f};
  std::vector<MLFloat16> data_half(12);
  ConvertFloatToMLFloat16(data.data(), data_half.data(), 12);

  std::vector<float> result = {33.0f, 45.0f};
  std::vector<MLFloat16> result_half(2);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 2);

  test.AddInput<MLFloat16>("data", {3, 2, 2}, data_half);
  test.AddOutput<MLFloat16>("reduced", {2}, result_half);
  test.Run();
}

void test_half_reduce_sum(
    int64_t m, int64_t n) {
  OpTester test("ReduceSum");
  // Input tensor.
  std::vector<float> X(m * n, 0.0f);
  // Reduced tensor.
  std::vector<float> Y(n, 0.0f);
  // Random number generator.
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const float value = distribution(generator) / float(m);
      X[i * n + j] = value;
      Y[j] += value;
    }
  }

  std::vector<MLFloat16> X_half(m * n);
  ConvertFloatToMLFloat16(X.data(), X_half.data(), int(m * n));

  std::vector<MLFloat16> Y_half(n);
  ConvertFloatToMLFloat16(Y.data(), Y_half.data(), int(n));

  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<MLFloat16>("data", {m, n}, X_half);
  test.AddOutput<MLFloat16>("reduced", {n}, Y_half);
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_half_bert) {
  test_half_reduce_sum(6 * 128, 128);
  test_half_reduce_sum(8 * 128, 128);
  test_half_reduce_sum(6 * 384, 128);
  test_half_reduce_sum(8 * 384, 128);
}

// Add more UTs for half as needed
#endif

#if defined(USE_CUDA) || defined(USE_DNNL)
TEST(ReductionOpTest, ReduceSum_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceSum", 14);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, 1});
  test.AddOutput<BFloat16>("reduced", {2}, MakeBFloat16({36.0f, 42.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_CUDA USE_DNNL

// on CUDA - this UT, with axes {0,2}, will go thru cudnn lib only if ATenOp is not initialized
#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceSumBFloat16_2) {
  OpTester test("ReduceSum", 14);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, 2});
  test.AddOutput<BFloat16>("reduced", {2}, MakeBFloat16({33.0f, 45.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(ReductionOpTest, ReduceSum_apex_reduction) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {36.0f, 42.0f});
  test.Run();
}

void test_apex_reduce_sum(
    int64_t m, int64_t n, bool exclude_openvino = false) {
  OpTester test("ReduceSum");
  // Input tensor.
  std::vector<float> X(m * n, 0.0f);
  // Reduced tensor.
  std::vector<float> Y(n, 0.0f);
  // Random number generator.
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const float value = distribution(generator) / float(m);
      X[i * n + j] = value;
      Y[j] += value;
    }
  }

  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {m, n}, X);
  test.AddOutput<float>("reduced", {n}, Y);

  if (exclude_openvino) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  } else {
    test.Run();
  }
}

TEST(ReductionOpTest, ReduceSum_apex_matrix_large) {
#ifdef USE_TENSORRT
  // Reduction op takes much longer time for TRT 8.2, so we test smaller range of inputs.
  int64_t threshold = 4096;
#else
  int64_t threshold = 32768;
#endif
  for (int64_t m = 1; m < 2049; m *= 8) {
    for (int64_t n = 2; n < 2049; n *= 8) {
      if (m * n > threshold) {
        continue;
      }
      test_apex_reduce_sum(m, n);
      test_apex_reduce_sum(m + 1, n);
      test_apex_reduce_sum(m + 3, n);
      test_apex_reduce_sum(m + 5, n);
      test_apex_reduce_sum(m + 23, n);
      test_apex_reduce_sum(m + 47, n);
      test_apex_reduce_sum(m + 97, n);
    }
  }
}

TEST(ReductionOpTest, ReduceSum_apex_bert) {
  test_apex_reduce_sum(6 * 128, 128);
  test_apex_reduce_sum(8 * 128, 128);
  test_apex_reduce_sum(6 * 384, 128);
  test_apex_reduce_sum(8 * 384, 128);
}

TEST(ReductionOpTest, ReduceSum_batch_by_two) {
  for (int i = 1; i < 128; ++i) {
    test_apex_reduce_sum(i, 2);
  }
}

TEST(ReductionOpTest, ReduceSum_batch_by_seq_by_128) {
#ifdef USE_TENSORRT
  // Reduction op takes much longer time for TRT 8.2, so we test smaller range of inputs.
  int i_max = 8;
#else
  int i_max = 16;
#endif
  for (int i = 1; i < i_max; i += 1) {
    test_apex_reduce_sum(i * 128, 128);
    test_apex_reduce_sum(i * 512, 128);
    test_apex_reduce_sum(i * 128, 768);
    test_apex_reduce_sum(i * 512, 768);
    test_apex_reduce_sum(i * 128, 1024);
    test_apex_reduce_sum(i * 512, 1024);
  }
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceSum_batch_by_seq_by_30528) {
  test_apex_reduce_sum(4 * 128, 30528);
  test_apex_reduce_sum(4 * 512, 30528);
}
#endif

TEST(ReductionOpTest, ReduceSum_bert_selected_batch_size) {
  test_apex_reduce_sum(85 * 128, 768);
  test_apex_reduce_sum(86 * 128, 768);
}

TEST(ReductionOpTest, ReduceSum_apex_more) {
  std::srand(0);
#ifdef USE_TENSORRT
  // Reduction op takes much longer time for TRT 8.2, so we test smaller range of inputs.
  int64_t m_max = 8;
  int64_t n_max = 8;
#else
  int64_t m_max = 16;
  int64_t n_max = 16;
#endif
  for (int64_t m = 1; m < m_max; ++m) {
    for (int64_t n = 1; n < n_max; ++n) {
      const auto m_ = 2 * m;
      const auto n_ = 2 * n;
      test_apex_reduce_sum(m_, n_);
    }
  }
}

TEST(ReductionOpTest, ReduceSum_int64) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

#if defined(USE_CUDA)
TEST(ReductionOpTest, ReduceSum_int64_omitted_optional_axes) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int64_t>("data", {3}, {1, 2, 3});
  test.AddOptionalInputEdge<int64_t>();
  test.AddOutput<int64_t>("reduced", {}, {6});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceSum_int64_cancellation) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  const int64_t large = int64_t{1} << 53;
  test.AddInput<int64_t>("data", {3}, {large, 1, -large});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int64_t>("reduced", {}, {1});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(ReductionOpTest, ReduceSum_default_axes_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_default_axes_do_not_keep_dims) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {78.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSum_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_int32_axes_input) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, 2}, true);
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_axes_input_initializer) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, true);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_axes_input_not_initializer) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, false);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_noop_axes_input_initializer_opset_13) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_empty_axes_input_initializer_opset_13) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {10.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_missing_axes_input_noop_opset_13) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);  // missing axes input and noop. should be identity
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddOptionalInputEdge<int64_t>();  // missing optional axes input
  test.AddOutput<float>("reduced", {1, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f});

  // TODO: OpenVINO doesn't support "axes" input in opset 13
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum0DTensor) {
  OpTester test("ReduceSum");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceSumSquare) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {247.0f, 403.f});
  test.Run();
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceSumSquare_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceSumSquare", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({247.0f, 403.f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  // USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_DNNL

TEST(ReductionOpTest, ReduceSumSquare_double) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {247.0, 403.});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_int32) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {247, 403});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Int64 not allowed as input to this layer
}

TEST(ReductionOpTest, ReduceSumSquare_int64) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {247, 403});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Int64 not allowed as input to this layer
}

TEST(ReductionOpTest, ReduceSumSquare_default_axes_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {650.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_default_axes_do_not_keep_dims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {650.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSumSquare_do_not_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {10.0f, 20.0f, 74.0f, 100.0f, 202.0f, 244.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_do_not_keepdims_2) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {14.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSumSquare_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {10.0f, 20.0f, 74.0f, 100.0f, 202.0f, 244.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare0DTensor) {
  OpTester test("ReduceSumSquare");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceProd_default_axes_keepdims) {
  OpTester test("ReduceProd");
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {479001600.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_default_axes_do_not_keep_dims) {
  OpTester test("ReduceProd");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {479001600.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceProd_do_not_keepdims) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.f, 8.f, 35.f, 48.f, 99.f, 120.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_do_not_keepdims_2) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceProd_keepdims) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {3.f, 8.f, 35.f, 48.f, 99.f, 120.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5400.f, 88704.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_double) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.AddOutput<double>("reduced", {1, 2, 1}, {5400.f, 88704.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProdAxesInitializerOpset18) {
  OpTester test("ReduceProd", 18);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true);
  test.AddOutput<float>("reduced", {1, 2, 1}, {5400.f, 88704.f});
  // TODO: DNNL, TensorRT, and OpenVINO dont support "axes" input in opset 18, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDnnlExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider});
}

#if defined(USE_DNNL)
TEST(ReductionOpTest, ReduceProd_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("ReduceProd", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<BFloat16>("data", {3, 2, 2},
                          MakeBFloat16({1.0f, 2.0f,
                                        3.0f, 4.0f,

                                        5.0f, 6.0f,
                                        7.0f, 8.0f,

                                        9.0f, 10.0f,
                                        11.0f, 12.0f}));
  test.AddOutput<BFloat16>("reduced", {1, 2, 1}, MakeBFloat16({5400.f, 88704.f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  // USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(ReductionOpTest, ReduceProd_int32) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {5400, 88704});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_int64) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {5400, 88704});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd0DTensor) {
  OpTester test("ReduceProd");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

template <typename T>
class ReductionOpTest : public ::testing::Test {
};

using ReductionOpTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(ReductionOpTest, ReductionOpTestTypes);

TYPED_TEST(ReductionOpTest, ArgMax) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<TypeParam>("data", {3, 2, 2},
                           GetTypedArray<TypeParam>({1.0f, 2.0f,
                                                     3.0f, 4.0f,

                                                     5.0f, 6.0f,
                                                     7.0f, 8.0f,

                                                     9.0f, 10.0f,
                                                     11.0f, 12.0f}));
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_Double_Type) {
  OpTester test("ArgMax", 11);
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_do_not_keepdims) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {3, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_do_not_keepdims_2) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {},
                          {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: node1: at least 2 dimensions are required for input
}

TEST(ReductionOpTest, ArgMax_int64) {
  OpTester test("ArgMax", 13);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run();
}

TEST(ReductionOpTest, ArgMax_int32) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run();
}

TEST(ReductionOpTest, ArgMax_int32_last_index_nodups) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);

  test.AddInput<int32_t>("data", {3, 2, 2},
                         {2, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          10, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_int32_last_index_dups) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);

  test.AddInput<int32_t>("data", {3, 2, 2},
                         {2, 4,
                          3, 4,

                          8, 6,
                          7, 8,

                          9, 13,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           0, 1,
                           1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_float_first_index_random) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", static_cast<int64_t>(0));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));

  // Since select_last_index is 0 by default, this test should run on both CPU and CUDA
  test.AddAttribute("select_last_index", static_cast<int64_t>(0));

  constexpr size_t vector_size = 64 * 1024;
  constexpr float max_value = std::numeric_limits<float>::infinity();

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, static_cast<int>(vector_size) - 1);

  std::vector<float> data_vec(vector_size, 0.0f);

  int min_index = -1;

  // Try replace 8 elements with max_value. It is fine that some elements hit same index.
  for (int i = 0; i < 8; ++i) {
    int index = distribution(generator);
    data_vec[index] = max_value;
    if (i == 0 || index < min_index) {
      min_index = index;
    }
  }

  test.AddInput<float>("data", {vector_size}, data_vec);
  test.AddOutput<int64_t>("reduced", {1}, {min_index});

  // Exclude OpenVINO since it failed to handle this case.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Rows that are wide enough to be reduced by several cooperating CUDA blocks,
// with more than one row so the per row bookkeeping is exercised as well.
static std::vector<float> MakeWideArgReductionInput(int64_t num_rows, int64_t num_cols,
                                                    const std::vector<int64_t>& max_indices,
                                                    const std::vector<int64_t>& min_indices) {
  std::vector<float> data(static_cast<size_t>(num_rows * num_cols));
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t col = 0; col < num_cols; ++col) {
      // Repeating ramp: plenty of duplicated values, no accidental extremes.
      data[static_cast<size_t>(row * num_cols + col)] = static_cast<float>((col + row) % 97) * 0.5f;
    }
    // Duplicated extremes: the first occurrence has to win.
    data[static_cast<size_t>(row * num_cols + max_indices[static_cast<size_t>(row)])] = 1000.0f;
    data[static_cast<size_t>(row * num_cols + num_cols - 1)] = 1000.0f;
    data[static_cast<size_t>(row * num_cols + min_indices[static_cast<size_t>(row)])] = -1000.0f;
    data[static_cast<size_t>(row * num_cols + num_cols - 2)] = -1000.0f;
  }
  return data;
}

TEST(ReductionOpTest, ArgMax_float_wide_last_axis) {
  constexpr int64_t num_rows = 3;
  constexpr int64_t num_cols = 40000;
  const std::vector<int64_t> max_indices{0, 12345, num_cols - 3};
  const std::vector<int64_t> min_indices{7, 30000, 1};

  OpTester test("ArgMax", 13);
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {num_rows, num_cols},
                       MakeWideArgReductionInput(num_rows, num_cols, max_indices, min_indices));
  test.AddOutput<int64_t>("reduced", {num_rows}, max_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_float_wide_last_axis) {
  constexpr int64_t num_rows = 3;
  constexpr int64_t num_cols = 40000;
  const std::vector<int64_t> max_indices{0, 12345, num_cols - 3};
  const std::vector<int64_t> min_indices{7, 30000, 1};

  OpTester test("ArgMin", 13);
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {num_rows, num_cols},
                       MakeWideArgReductionInput(num_rows, num_cols, max_indices, min_indices));
  test.AddOutput<int64_t>("reduced", {num_rows}, min_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_float_wide_last_axis_keepdims) {
  constexpr int64_t num_rows = 2;
  constexpr int64_t num_cols = 202048;
  const std::vector<int64_t> max_indices{131072, 3};
  const std::vector<int64_t> min_indices{0, 200000};

  OpTester test("ArgMax", 13);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<float>("data", {num_rows, num_cols},
                       MakeWideArgReductionInput(num_rows, num_cols, max_indices, min_indices));
  test.AddOutput<int64_t>("reduced", {num_rows, 1}, max_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_int32_neg_axis) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)(-2));
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});

  test.Run();
}

TEST(ReductionOpTest, ArgMax_int8) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  // TensorRT: input/output with DataType Int8 in network without Q/DQ layers
  //           must have dynamic range set when no calibrator is used
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_uint8) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run();
}

TEST(ReductionOpTest, ArgMax2D) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2},
                       {1.0f, 2.0f,
                        6.0f, 5.0f,
                        9.0f, 10.0f});
  test.AddOutput<int64_t>("reduced", {3, 1},
                          {1, 0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax2D_select_last) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<float>("data", {3, 2},
                       {1.0f, 2.0f,
                        6.0f, 5.0f,
                        9.0f, 10.0f});
  test.AddOutput<int64_t>("reduced", {3, 1},
                          {1, 0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax2D_dim1) {
  OpTester test("ArgMax", 11);
  test.AddAttribute("axis", (int64_t)1);
  test.AddInput<float>("data", {3, 1},
                       {1.0f,
                        6.0f,
                        9.0f});
  test.AddOutput<int64_t>("reduced", {3, 1},
                          {0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {1, 2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_Double_Type) {
  OpTester test("ArgMin", 11);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {0, 0,
                           0, 0,
                           0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMin_Double_Precision) {
  OpTester test("ArgMin", 11);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0 + 1e-10, 3.0,
                         1.0, 3.0 - 0.5e-10,

                         1.0, 3.0 + 1e-10,
                         1 + 1e-10, 3 - 1e-10,

                         1.0 + 2e-10, 3.0 - 1e-10,
                         1 - 1e-10, 3.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           0, 1,
                           1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims_2) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: node1: at least 2 dimensions are required for input
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims_2_select_last) {
  OpTester test("ArgMin", 12);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_uint8) {
  OpTester test("ArgMin", 13);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_int8) {
  OpTester test("ArgMin", 13);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_int64) {
  OpTester test("ArgMin", 13);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_int32) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_int32_select_last) {
  OpTester test("ArgMin", 12);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          1, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {1, 0,
                           0, 0});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_int32_neg_axis) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)(-3));
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});

  test.Run();
}

TEST(ReductionOpTest, ArgMin_float_first_index_random) {
  OpTester test("ArgMin", 13);
  test.AddAttribute("axis", static_cast<int64_t>(0));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));

  // Since select_last_index is 0 by default, this test should run on both CPU and CUDA
  test.AddAttribute("select_last_index", static_cast<int64_t>(0));

  constexpr size_t vector_size = 64 * 1024;
  constexpr float min_value = -std::numeric_limits<float>::infinity();

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, static_cast<int>(vector_size) - 1);

  std::vector<float> data_vec(vector_size, 0.0f);

  int min_index = -1;

  // Try replace 8 elements with min_value. It is fine that some elements hit same index.
  for (int i = 0; i < 8; ++i) {
    int index = distribution(generator);
    data_vec[index] = min_value;
    if (i == 0 || index < min_index) {
      min_index = index;
    }
  }

  test.AddInput<float>("data", {vector_size}, data_vec);
  test.AddOutput<int64_t>("reduced", {1}, {min_index});

  // Exclude OpenVINO since it failed to handle this case.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_ReduceDimWithZero1) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{3, 0, 2}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {};
  expected_fast_axes = {};
  expected_fast_output_shape = {1, 0, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kEmpty);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_ReduceDimWithScalarInputAxesPresent) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      EmptySpan<int64_t>(), AsSpan<int64_t>({1, 2, 3}),
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {};
  expected_fast_axes = {};
  expected_fast_output_shape = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kEmpty);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_ReduceDimWithZero1b) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{3, 0, 2}, std::vector<int64_t>{1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {};
  expected_fast_axes = {};
  expected_fast_output_shape = {3, 0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kEmpty);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

// test that PrepareForReduce handles this case. Called by all reduction ops so any op can be used in the test
TEST(ReductionOpTest, ReduceDimWithZero1) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected output shape [{1,0,1}] did not match run output shape [{1,1,1}] for reduced";
  }

  auto run = [](OpTester& tester, const std::string& error_msg = "") {
    auto expect = error_msg.empty() ? OpTester::ExpectResult::kExpectSuccess
                                    : OpTester::ExpectResult::kExpectFailure;

    tester.Run(expect, error_msg,
               // exclude EPs that don't handle this
               // TODO: fix reduce kernel for zero set cases. see: https://github.com/microsoft/onnxruntime/issues/18588
               {
                   kCoreMLExecutionProvider,
                   kDnnlExecutionProvider,
                   kMIGraphXExecutionProvider,
                   kOpenVINOExecutionProvider,
                   kQnnExecutionProvider,
                   kTensorrtExecutionProvider,
               });
  };

  // reduce on all axes keeping dims. should allow the 0 to be the reduced value
  OpTester test("ReduceSum", 10);
  test.AddAttribute("keepdims", int64_t(1));
  test.AddInput<float>("data", {3, 0, 2}, {});
  test.AddOutput<float>("reduced", {1, 1, 1}, {0.0f});
  run(test);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_ReduceDimWithZero2) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=0 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{3, 0, 2}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {};
  expected_fast_axes = {};
  expected_fast_output_shape = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kEmpty);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, ReduceDimWithZero2) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Can't reduce on dim with value of 0 if 'keepdims' is false. Invalid output shape would be produced. input_shape:{?,0,?}";
  }

  auto run = [](OpTester& tester, const std::string& error_msg = "") {
    auto expect = error_msg.empty() ? OpTester::ExpectResult::kExpectSuccess
                                    : OpTester::ExpectResult::kExpectFailure;

    tester.Run(expect, error_msg,
               // exclude EPs that don't handle this
               // TODO: fix reduce kernel for zero set cases. see: https://github.com/microsoft/onnxruntime/issues/18588
               {
                   kCoreMLExecutionProvider,
                   kCudaExecutionProvider,
                   kCudaNHWCExecutionProvider,
                   kDnnlExecutionProvider,
                   kMIGraphXExecutionProvider,
                   kOpenVINOExecutionProvider,
                   kQnnExecutionProvider,
                   kTensorrtExecutionProvider,
               });
  };

  // reducing on all axes including one or more with 0 dimension, with keepdims=0, results a scalar of 0.
  OpTester test2("ReduceSum", 10);
  test2.AddAttribute("keepdims", int64_t(0));
  test2.AddShapeToTensorData(true, 1);
  test2.AddInput<float>("data", {3, 0, 2}, {});
  test2.AddOutput<float>("reduced", {}, {0.0f});
  run(test2);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_ReduceDimWithZero3) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=0 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{3, 0, 2}, std::vector<int64_t>{2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {0, 2};
  expected_fast_axes = {1};
  expected_fast_output_shape = {3, 0};
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
}

TEST(ReductionOpTest, ReduceSum_ReduceDimWithZero3) {
  auto run = [](OpTester& tester, const std::string& error_msg = "") {
    auto expect = error_msg.empty() ? OpTester::ExpectResult::kExpectSuccess
                                    : OpTester::ExpectResult::kExpectFailure;

    tester.Run(expect, error_msg,
               // exclude EPs that don't handle this
               {
                   kCoreMLExecutionProvider,
                   kTensorrtExecutionProvider,
                   kOpenVINOExecutionProvider,
                   kQnnExecutionProvider,
               });
  };

  // reduction is possible without keeping dims if we only reduce on non-zero dims
  OpTester test3("ReduceSum", 10);
  test3.AddAttribute("keepdims", int64_t(0));
  test3.AddAttribute("axes", std::vector<int64_t>{2});
  test3.AddShapeToTensorData(true, 1);
  test3.AddInput<float>("data", {3, 0, 2}, {});
  test3.AddOutput<float>("reduced", {3, 0}, {});
  run(test3);
}

// test if noop_with_empty_axes behaves correctly
TEST(ReductionOpTest, ReduceL1_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceL1", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceL1_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceL1", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {10.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceL2", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceL2_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceL2", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {5.47722558f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceMax", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceMax_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceMax", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {4.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceMean", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceMean", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {2.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceMin", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceMin", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceProd", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kTensorrtExecutionProvider,
       kOpenVINOExecutionProvider,
       kDnnlExecutionProvider,
       kDmlExecutionProvider});
}

TEST(ReductionOpTest, ReduceProd_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceProd", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {24.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceSum", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {kOpenVINOExecutionProvider}  // OpenVINO: Disabled temporarily
  );
}

TEST(ReductionOpTest, ReduceSum_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceSum", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {10.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_noop_axes_input_initializer_opset_18) {
  OpTester test("ReduceSumSquare", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 4.0f, 9.0f, 16.0f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceSumSquare_empty_axes_input_initializer_opset_18) {
  OpTester test("ReduceSumSquare", 18);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)0);  // Not NoOP, use default axes.
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {}, {30.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfMax) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {1.0f, 4.0f,
                         FLOAT_INF, FLOAT_INF,
                         FLOAT_INF, FLOAT_INF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfMax_double) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {6, 2},
                        {1.0f, DOUBLE_NINF,
                         DOUBLE_NINF, 4.0f,
                         DOUBLE_INF, DOUBLE_NINF,
                         DOUBLE_NINF, DOUBLE_INF,
                         1.0f, DOUBLE_INF,
                         DOUBLE_INF, 4.0f});
  test.AddOutput<double>("reduced", {6},
                         {1.0f, 4.0f,
                          DOUBLE_INF, DOUBLE_INF,
                          DOUBLE_INF, DOUBLE_INF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfMin) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {1.0f, 4.0f,
                         FLOAT_NINF, FLOAT_NINF,
                         FLOAT_NINF, FLOAT_NINF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfMin_double) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {6, 2},
                        {1.0f, DOUBLE_INF,
                         DOUBLE_INF, 4.0f,
                         DOUBLE_INF, DOUBLE_NINF,
                         DOUBLE_NINF, DOUBLE_INF,
                         1.0f, DOUBLE_NINF,
                         DOUBLE_NINF, 4.0f});
  test.AddOutput<double>("reduced", {6},
                         {1.0f, 4.0f,
                          DOUBLE_NINF, DOUBLE_NINF,
                          DOUBLE_NINF, DOUBLE_NINF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfSum) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {FLOAT_INF, FLOAT_INF,
                         std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                         FLOAT_NINF, FLOAT_NINF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSum) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 1.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 1.0f});
  test.AddOutput<float>("reduced", {6},
                        {FLOAT_INF, FLOAT_INF,
                         -std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSumExp) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 2}, {1.0f, FLOAT_NINF, FLOAT_NINF, 1.0f});
  test.AddOutput<float>("reduced", {2}, {1.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSumExp_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {2, 2}, {1.0, DOUBLE_NINF, DOUBLE_NINF, 1.0});
  test.AddOutput<double>("reduced", {2}, {1.0, 1.0});
  test.Run();
}

// Specific cases for Reduce.

TEST(ReductionOpTest, OptimizeShapeForFastReduce_R_K) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>{0},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {10};
  expected_fast_output_shape = {1};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>{0, 1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {110};
  expected_fast_output_shape = {1, 1};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // R - keep_dims=0
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>{0},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {10};
  expected_fast_output_shape = {};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>{0, 1},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {110};
  expected_fast_output_shape = {};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_R_empty) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_axes = {0};
  expected_fast_shape = {10};
  expected_fast_output_shape = {1};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {110};
  expected_fast_output_shape = {1, 1};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // R - keep_dims=0 - noop=false
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>{},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {10};
  expected_fast_output_shape = {};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>{},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {110};
  expected_fast_output_shape = {};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_K_empty) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // R - keep_dims=1 - noop=true
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, true, true);
  expected_fast_shape = {10};
  expected_fast_output_shape = {10};
  expected_fast_axes = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>(),
      fast_shape, fast_output_shape, fast_axes, true, true);
  expected_fast_shape = {110};
  expected_fast_output_shape = {10, 11};
  expected_fast_axes = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // R - keep_dims=0 - noop=true
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10}, std::vector<int64_t>{},
      fast_shape, fast_output_shape, fast_axes, false, true);
  expected_fast_shape = {10};
  expected_fast_output_shape = {10};
  expected_fast_axes = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>{},
      fast_shape, fast_output_shape, fast_axes, false, true);
  expected_fast_shape = {110};
  expected_fast_output_shape = {10, 11};
  expected_fast_axes = {};
  ASSERT_EQ(fast_kind, FastReduceKind::kK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_KR) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // KR - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{10, 11}, TensorShapeVector{1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {10, 11};
  expected_fast_output_shape = {10, 1};
  expected_fast_axes = {1};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{9, 10, 11}, TensorShapeVector{1, 2},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {9, 110};
  expected_fast_output_shape = {9, 1, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{9, 10, 11}, TensorShapeVector{2},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {90, 11};
  expected_fast_output_shape = {9, 10, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // KR - keep_dims=0
  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{10, 11}, TensorShapeVector{1},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {10, 11};
  expected_fast_output_shape = {10};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{9, 10, 11}, TensorShapeVector{1, 2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {9, 110};
  expected_fast_output_shape = {9};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {90, 11};
  expected_fast_output_shape = {9, 10};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_KR_neg) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;

  // KR - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{10, 11}, TensorShapeVector{-1},
      fast_shape, fast_output_shape, fast_axes, true);
  TensorShapeVector expected_fast_shape{10, 11};
  TensorShapeVector expected_fast_output_shape{10, 1};
  TensorShapeVector expected_fast_axes{1};
  ASSERT_EQ(fast_kind, FastReduceKind::kKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_RK) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // RK - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{10, 11}, TensorShapeVector{0},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {10, 11};
  expected_fast_output_shape = {1, 11};
  expected_fast_axes = {0};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{9, 10, 11}, TensorShapeVector{0, 1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {90, 11};
  expected_fast_output_shape = {1, 1, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{9, 10, 11}, TensorShapeVector{0},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {9, 110};
  expected_fast_output_shape = {1, 10, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // RK - keep_dims=0
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{10, 11}, std::vector<int64_t>{0},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {10, 11};
  expected_fast_output_shape = {11};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{0, 1},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {90, 11};
  expected_fast_output_shape = {11};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{0},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {9, 110};
  expected_fast_output_shape = {10, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_KRK) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // KRK - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {9, 10, 11};
  expected_fast_output_shape = {9, 1, 11};
  expected_fast_axes = {1};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{7, 9, 10, 11}, TensorShapeVector{1, 2},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {7, 90, 11};
  expected_fast_output_shape = {7, 1, 1, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{7, 9, 10, 11}, TensorShapeVector{1},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {7, 9, 110};
  expected_fast_output_shape = {7, 1, 10, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      TensorShapeVector{7, 9, 10, 11}, TensorShapeVector{2},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {63, 10, 11};
  expected_fast_output_shape = {7, 9, 1, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // KRK - keep_dims=0
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{1},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {9, 10, 11};
  expected_fast_output_shape = {9, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{1, 2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {7, 90, 11};
  expected_fast_output_shape = {7, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{1},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {7, 9, 110};
  expected_fast_output_shape = {7, 10, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {63, 10, 11};
  expected_fast_output_shape = {7, 9, 11};
  ASSERT_EQ(fast_kind, FastReduceKind::kKRK);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_RKR) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // RKR - keep_dims=1
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{0, 2},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = TensorShapeVector{9, 10, 11};
  expected_fast_output_shape = TensorShapeVector{1, 10, 1};
  expected_fast_axes = TensorShapeVector{0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 3},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = TensorShapeVector{7, 90, 11};
  expected_fast_output_shape = TensorShapeVector{1, 9, 10, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 2, 3},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = TensorShapeVector{7, 9, 110};
  expected_fast_output_shape = TensorShapeVector{1, 9, 1, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 1, 3},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = TensorShapeVector{63, 10, 11};
  expected_fast_output_shape = TensorShapeVector{1, 1, 10, 1};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // KRK - keep_dims=0
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{9, 10, 11}, std::vector<int64_t>{0, 2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = TensorShapeVector{9, 10, 11};
  expected_fast_output_shape = TensorShapeVector{10};
  expected_fast_axes = TensorShapeVector{0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 3},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = TensorShapeVector{7, 90, 11};
  expected_fast_output_shape = TensorShapeVector{9, 10};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 2, 3},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = TensorShapeVector{7, 9, 110};
  expected_fast_output_shape = TensorShapeVector{9};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 1, 3},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = TensorShapeVector{63, 10, 11};
  expected_fast_output_shape = TensorShapeVector{10};
  ASSERT_EQ(fast_kind, FastReduceKind::kRKR);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, OptimizeShapeForFastReduce_NONE) {
  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, fast_output_shape, fast_axes;
  TensorShapeVector expected_fast_shape, expected_fast_output_shape, expected_fast_axes;

  // RKRK
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{0, 2},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {7, 9, 10, 11};
  expected_fast_output_shape = {9, 11};
  expected_fast_axes = {0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kNone);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11}, std::vector<int64_t>{1, 3},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {7, 9, 10, 11};
  expected_fast_output_shape = {7, 1, 10, 1};
  expected_fast_axes = {1, 3};
  ASSERT_EQ(fast_kind, FastReduceKind::kNone);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  // RRKKRRKK
  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11, 2, 3, 4, 6}, std::vector<int64_t>{0, 1, 4, 5},
      fast_shape, fast_output_shape, fast_axes, false);
  expected_fast_shape = {63, 110, 6, 24};
  expected_fast_output_shape = {10, 11, 4, 6};
  expected_fast_axes = {0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kNone);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);

  fast_kind = OptimizeShapeForFastReduce(
      std::vector<int64_t>{7, 9, 10, 11, 2, 3, 4, 6}, std::vector<int64_t>{0, 1, 4, 5},
      fast_shape, fast_output_shape, fast_axes, true);
  expected_fast_shape = {63, 110, 6, 24};
  expected_fast_output_shape = {1, 1, 10, 11, 1, 1, 4, 6};
  expected_fast_axes = {0, 2};
  ASSERT_EQ(fast_kind, FastReduceKind::kNone);
  ASSERT_EQ(fast_shape, expected_fast_shape);
  ASSERT_EQ(fast_output_shape, expected_fast_output_shape);
  ASSERT_EQ(fast_axes, expected_fast_axes);
}

TEST(ReductionOpTest, EigenMax) {
  std::vector<float> mat{1, 2, 3, 4};

  auto res1 = ConstEigenMatrixMap<float>(mat.data(), 2, 2).rowwise().maxCoeff();
  std::vector<float> expected{3, 4};
  std::vector<float> out1(res1.begin(), res1.end());
  ASSERT_EQ(out1, expected);

  auto res2 = ConstEigenMatrixMap<float>(mat.data(), 2, 2).colwise().maxCoeff();
  expected = std::vector<float>{2, 4};
  std::vector<float> out2(res2.begin(), res2.end());
  ASSERT_EQ(out2, expected);

  mat = std::vector<float>{1, 2, 3, 4, 5, 6};

  auto res3 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).rowwise().maxCoeff();
  expected = std::vector<float>{5, 6};
  std::vector<float> out3(res3.begin(), res3.end());
  ASSERT_EQ(out3, expected);

  auto res4 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).colwise().maxCoeff();
  expected = std::vector<float>{2, 4, 6};
  std::vector<float> out4(res4.begin(), res4.end());
  ASSERT_EQ(out4, expected);

  auto res5 = ConstEigenMatrixMap<float>(mat.data(), 3, 2).rowwise().maxCoeff();
  expected = std::vector<float>{4, 5, 6};
  std::vector<float> out5(res5.begin(), res5.end());
  ASSERT_EQ(out5, expected);

  auto res6 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).colwise().maxCoeff();
  expected = std::vector<float>{2, 4, 6};
  std::vector<float> out6(res6.begin(), res6.end());
  ASSERT_EQ(out6, expected);
}

TEST(ReductionOpTest, EigenSum) {
  std::vector<float> mat{1, 10, 100, 1000};

  auto res1 = ConstEigenMatrixMap<float>(mat.data(), 2, 2).rowwise().sum();
  std::vector<float> expected{101, 1010};
  std::vector<float> out1(res1.begin(), res1.end());
  ASSERT_EQ(out1, expected);

  auto res2 = ConstEigenMatrixMap<float>(mat.data(), 2, 2).colwise().sum();
  expected = std::vector<float>{11, 1100};
  std::vector<float> out2(res2.begin(), res2.end());
  ASSERT_EQ(out2, expected);

  mat = std::vector<float>{1, 10, 100, 1000, 10000, 100000};

  auto res3 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).rowwise().sum();
  expected = std::vector<float>{10101, 101010};
  std::vector<float> out3(res3.begin(), res3.end());
  ASSERT_EQ(out3, expected);

  auto res4 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).colwise().sum();
  expected = std::vector<float>{11, 1100, 110000};
  std::vector<float> out4(res4.begin(), res4.end());
  ASSERT_EQ(out4, expected);

  auto res5 = ConstEigenMatrixMap<float>(mat.data(), 3, 2).rowwise().sum();
  expected = std::vector<float>{1001, 10010, 100100};
  std::vector<float> out5(res5.begin(), res5.end());
  ASSERT_EQ(out5, expected);

  auto res6 = ConstEigenMatrixMap<float>(mat.data(), 2, 3).colwise().sum();
  expected = std::vector<float>{11, 1100, 110000};
  std::vector<float> out6(res6.begin(), res6.end());
  ASSERT_EQ(out6, expected);
}

TEST(ReductionOpTest, ReduceMax_KR_parallel) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {4, 3},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {4}, {3.f, 6.f, 9.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_KR) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3}, {4.f, 8.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_KR_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1}, {4.f, 8.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RK) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {4}, {9.f, 10.f, 11.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RK_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 4}, {9.f, 10.f, 11.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RK_parallel) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  std::vector<float> in_data(65536);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = 1.f + (float)(i % 17) / 17.f;
  test.AddInput<float>("data", {2048, 32}, in_data);
  std::vector<float> expected(32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = 0;
    for (size_t j = 0; j < 2048; ++j) {
      expected[i] = std::max(expected[i], in_data[i + j * expected.size()]);
    }
  }
  test.AddOutput<float>("reduced", {32}, expected);
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_KRK) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.f, 4.f, 7.f, 8.f, 11.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_KRK_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {3.f, 4.f, 7.f, 8.f, 11.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RKR) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {10.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RKR_parallel) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 16, 2},
                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f});
  test.AddOutput<float>("reduced", {16}, {33.0f, 35.0f, 37.0f, 39.0f, 41.0f, 43.0f, 45.0f, 47.0f, 49.0f, 51.0f, 53.0f, 55.0f, 57.0f, 59.0f, 61.0f, 63.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RKR_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {10.f, 12.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RKRK) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {2, 2}, {19.f, 20.f, 23.f, 24.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_RKRK_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {1, 2, 1, 2}, {19.f, 20.f, 23.f, 24.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_KR) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3}, {2.5f, 6.5f, 10.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_KR_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1}, {2.5f, 6.5f, 10.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RK) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {4}, {5.f, 6.f, 7.f, 8.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RK_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 4}, {5.f, 6.f, 7.f, 8.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_KRK) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {2.f, 3.f, 6.f, 7.f, 10.f, 11.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_KRK_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {2.f, 3.f, 6.f, 7.f, 10.f, 11.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RKR) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {5.5f, 7.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RKR_parallel) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 16, 2},
                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f});
  test.AddOutput<float>("reduced", {16}, {16.5f, 18.5f, 20.5f, 22.5f, 24.5f, 26.5f, 28.5f, 30.5f, 32.5f, 34.5f, 36.5f, 38.5f, 40.5f, 42.5f, 44.5f, 46.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RKR_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5f, 7.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RKRK) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {2, 2}, {10.f, 11.f, 14.f, 15.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_RKRK_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {1, 2, 1, 2}, {10.f, 11.f, 14.f, 15.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_KR) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3}, {1.f, 5.f, 9.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_KR_parallel) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {4, 3},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {4}, {11.f, 14.f, 17.f, 20.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_KR_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {3, 1}, {11.f, 15.f, 19.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RK) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {4}, {11.f, 12.f, 13.f, 14.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RK_parallel) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  std::vector<float> in_data(65536);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = 1.f + (float)(i % 17) / 17.f;
  test.AddInput<float>("data", {2048, 32}, in_data);
  std::vector<float> expected(32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = 10.f;
    for (size_t j = 0; j < 2048; ++j) {
      expected[i] = std::min(expected[i], in_data[i + j * expected.size()]);
    }
  }
  test.AddOutput<float>("reduced", {32}, expected);
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RK_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {1, 4}, {11.f, 12.f, 13.f, 14.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_KRK) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {3, 2}, {11.f, 12.f, 15.f, 16.f, 19.f, 20.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_KRK_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {11.f, 12.f, 15.f, 16.f, 19.f, 20.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RKR) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {2}, {11.f, 13.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RKR_parallel) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 16, 2},
                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f});
  test.AddOutput<float>("reduced", {16}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RKR_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {11.0f, 12.0f,
                        13.0f, 14.0f,

                        15.0f, 16.0f,
                        17.0f, 18.0f,

                        19.0f, 20.0f,
                        21.0f, 22.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {11.f, 13.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RKRK) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {2, 2}, {1.f, 2.f, 5.f, 6.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_RKRK_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {1, 2, 1, 2}, {1.f, 2.f, 5.f, 6.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KR) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3}, {10.0f, 26.0f, 42.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KR_parallel) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {4, 3},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {4}, {6.0f, 15.0f, 24.0f, 33.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KR_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1}, {10.0f, 26.0f, 42.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KR2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3}, {10.0f, 26.0f, 42.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KR2_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 1}, {10.0f, 26.0f, 42.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RK) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {4}, {15.f, 18.f, 21.f, 24.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RK_parallel) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  std::vector<float> in_data(65536);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = 1.f + (float)(i % 17) / 17.f;
  test.AddInput<float>("data", {2048, 32}, in_data);
  std::vector<float> expected(32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = 0;
    for (size_t j = 0; j < 2048; ++j) {
      expected[i] += in_data[i + j * expected.size()];
    }
  }
  test.AddOutput<float>("reduced", {32}, expected);

  // CoreML does not provide 1e-5 precision here (it's off by 1e-4)
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ReductionOpTest, ReduceSum_RK_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 4}, {15.f, 18.f, 21.f, 24.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RK2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {36.f, 42.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RK2_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 2}, {36.f, 42.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KRK) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {4.f, 6.f, 12.f, 14.f, 20.f, 22.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KRK_parallel) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  std::vector<float> in_data(512);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = (float)i;
  test.AddInput<float>("data", {128, 2, 2}, in_data);
  std::vector<float> expected(256);
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      expected[i * 2 + j] = 0;
      for (size_t k = 0; k < 2; ++k) {
        expected[i * 2 + j] += in_data[i * 4 + k * 2 + j];
      }
    }
  }
  test.AddOutput<float>("reduced", {128, 2}, expected);
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KRK_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {4.f, 6.f, 12.f, 14.f, 20.f, 22.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KRK2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {3, 2}, {16.f, 20.f, 48.f, 52.f, 80.f, 84.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_KRK2_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {3, 1, 1, 2}, {16.f, 20.f, 48.f, 52.f, 80.f, 84.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {33.f, 45.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR_parallel) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 16, 2},
                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f});
  test.AddOutput<float>("reduced", {16}, {66.0f, 74.0f, 82.0f, 90.0f, 98.0f, 106.0f, 114.0f, 122.0f, 130.0f, 138.0f, 146.0f, 154.0f, 162.0f, 170.0f, 178.0f, 186.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR_parallel_bigger) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  std::vector<float> in_data(512);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = (float)i;
  test.AddInput<float>("data", {2, 128, 2}, in_data);
  std::vector<float> expected(128);
  for (size_t j = 0; j < 128; ++j) {
    expected[j] = 0;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t k = 0; k < 2; ++k) {
        expected[j] += in_data[i * 256 + j * 2 + k];
      }
    }
  }
  test.AddOutput<float>("reduced", {128}, expected);
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.f, 45.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 3});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {2, 2}, {57.0f, 69.0f, 81.0f, 93.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKR2_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 3});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {1, 2, 2, 1}, {57.0f, 69.0f, 81.0f, 93.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKRK) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {2, 2}, {60.f, 66.f, 84.f, 90.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_RKRK_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f,

                        17.0f, 18.0f,
                        19.0f, 20.0f,

                        21.0f, 22.0f,
                        23.0f, 24.0f});
  test.AddOutput<float>("reduced", {1, 2, 1, 2}, {60.f, 66.f, 84.f, 90.f});
  test.Run();
}

void test_empty_set(const std::string& op, int opset, bool axes_as_input, float empty_value) {
  OpTester test(op, opset);
  std::vector<int64_t> input_shape = {2, 0, 4};
  int64_t input_size = std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  std::vector<float> data(input_size);
  test.AddInput("data", input_shape, data);
  std::vector<int64_t> axes = {1};
  if (axes_as_input) {
    test.AddInput("axes", {(int64_t)(axes.size())}, axes);
  } else {
    test.AddAttribute("axes", axes);
  }

  std::vector<int64_t> output_shape = {2, 1, 4};
  int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  std::vector<float> reduced(output_size, empty_value);
  test.AddOutput<float>("reduced", output_shape, reduced);
  test.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      {
          kCoreMLExecutionProvider,
          kCudaExecutionProvider,
          kCudaNHWCExecutionProvider,
          kDmlExecutionProvider,
          kDnnlExecutionProvider,
          kMIGraphXExecutionProvider,
          kOpenVINOExecutionProvider,
          kQnnExecutionProvider,
          kTensorrtExecutionProvider,
          kWebGpuExecutionProvider,
      });
}

void TestEmptySetNegativeAxis(const char* op, int opset, bool axes_as_input,
                              int64_t keepdims, const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& output_shape) {
  OpTester test(op, opset);
  test.AddAttribute("keepdims", keepdims);
  test.AddInput<float>("data", input_shape, {});
  const std::vector<int64_t> axes{-1};
  if (axes_as_input) {
    test.AddInput<int64_t>("axes", {1}, axes);
  } else {
    test.AddAttribute("axes", axes);
  }
  test.AddOutput<float>("reduced", output_shape, {});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, EmptySetNegativeAxis) {
  for (const char* op : {"ReduceSum", "ReduceProd", "ReduceMean"}) {
    TestEmptySetNegativeAxis(op, 18, true, 1, {1, 0, 3}, {1, 0, 1});
    TestEmptySetNegativeAxis(op, 18, true, 0, {0, 3}, {0});
  }

  TestEmptySetNegativeAxis("ReduceProd", 13, false, 1, {1, 0, 3}, {1, 0, 1});
}

TEST(ReductionOpTest, EmptySetMissingOptionalAxesReducesAllDimensions) {
  OpTester test("ReduceSum", 20);
  test.AddInput<float>("data", {2, 0, 4}, {});
  test.AddOptionalInputEdge<int64_t>();
  test.AddOutput<float>("reduced", {1, 1, 1}, {0.0f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, EmptySetAxesMustBeVector) {
  const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> axes_cases{
      {{}, {1}},
      {{2, 0}, {}},
  };
  for (const char* const op : {"ReduceSum", "ReduceLogSumExp"}) {
    for (const auto& [axes_shape, axes_data] : axes_cases) {
      OpTester test(op, 20);
      test.AddInput<float>("data", {2, 0, 4}, {});
      test.AddInput<int64_t>("axes", axes_shape, axes_data);
      test.AddOutput<float>("reduced", {1, 1, 1}, {0.0f});
      test.Config(OpTester::ExpectResult::kExpectFailure, "An axes tensor must be a vector tensor.")
          .ConfigEp(DefaultCpuExecutionProvider())
          .RunWithConfig();
    }
  }
}

void TestEmptySetNoopWithEmptyAxes(const std::string& op, bool omit_axes) {
  OpTester test(op, 20);
  test.AddInput<float>("data", {2, 0, 4}, {});
  if (omit_axes) {
    test.AddOptionalInputEdge<int64_t>();
  } else {
    test.AddInput<int64_t>("axes", {0}, {}, true);
  }
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 0, 4}, {});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, EmptySetNoopWithMissingAxes) {
  TestEmptySetNoopWithEmptyAxes("ReduceSum", true);
  TestEmptySetNoopWithEmptyAxes("ReduceLogSumExp", true);
}

TEST(ReductionOpTest, EmptySetNoopWithEmptyAxes) {
  TestEmptySetNoopWithEmptyAxes("ReduceSum", false);
  TestEmptySetNoopWithEmptyAxes("ReduceLogSumExp", false);
}

TEST(ReductionOpTest, empty_set_ReduceL1) {
  test_empty_set("ReduceL1", 20, true, 0);
}

TEST(ReductionOpTest, empty_set_ReduceL1_13) {
  test_empty_set("ReduceL1", 13, false, 0);
}

TEST(ReductionOpTest, empty_set_ReduceL2) {
  test_empty_set("ReduceL2", 20, true, 0);
}

TEST(ReductionOpTest, empty_set_ReduceL2_13) {
  test_empty_set("ReduceL2", 13, false, 0);
}

TEST(ReductionOpTest, empty_set_ReduceLogSum) {
  test_empty_set("ReduceLogSum", 20, true, -std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceLogSum_13) {
  test_empty_set("ReduceLogSum", 13, false, -std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceLogSumExp) {
  test_empty_set("ReduceLogSumExp", 20, true, -std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceLogSumExp_13) {
  test_empty_set("ReduceLogSumExp", 13, false, -std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceMin) {
  test_empty_set("ReduceMin", 20, true, std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceMin_13) {
  test_empty_set("ReduceMin", 13, false, std::numeric_limits<float>::infinity());
}

TEST(ReductionOpTest, empty_set_ReduceProd) {
  test_empty_set("ReduceProd", 20, true, 1.0f);
}

TEST(ReductionOpTest, empty_set_ReduceProd_13) {
  test_empty_set("ReduceProd", 13, false, 1.0f);
}

TEST(ReductionOpTest, empty_set_ReduceSum) {
  test_empty_set("ReduceSum", 20, true, 0.0f);
}

TEST(ReductionOpTest, empty_set_ReduceSum_13) {
  test_empty_set("ReduceSum", 11, false, 0.0f);
}

TEST(ReductionOpTest, empty_set_ReduceSumSquare) {
  test_empty_set("ReduceSumSquare", 20, true, 0.0f);
}

TEST(ReductionOpTest, empty_set_ReduceSumSquare_13) {
  test_empty_set("ReduceSumSquare", 13, false, 0.0f);
}

TEST(ReductionOpTest, MissingOptionalAxes) {
  OpTester test("ReduceMax", 18);
  test.AddInput<float>("data", {2, 2},
                       {1.0f, 4.0f,
                        3.0f, 2.0f});
  test.AddOptionalInputEdge<int64_t>();
  test.AddOutput<float>("reduced", {1, 1}, {4.0f});

  // OpenVINO doesn't support "axes" input
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSumSquare_NoopWithEmptyAxes_Scalar) {
  OpTester test("ReduceSumSquare", 18);
  test.AddInput<float>("data", {}, {-3.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {}, {9.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceSumSquare_NoopWithEmptyAxes_ElementwiseSquare) {
  OpTester test("ReduceSumSquare", 18);
  test.AddInput<float>("data", {2}, {2.f, 3.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2}, {4.f, 9.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceSumSquare_NoopWithEmptyAxes_2D_ElementwiseSquare) {
  OpTester test("ReduceSumSquare", 18);
  test.AddInput<float>("data", {2, 3}, {-1.f, 2.f, -3.f, 0.5f, -0.5f, 4.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 3}, {1.f, 4.f, 9.f, 0.25f, 0.25f, 16.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceSumSquare_NoopWithEmptyAxes_3D_ElementwiseSquare) {
  OpTester test("ReduceSumSquare", 18);
  test.AddInput<float>("data", {2, 1, 3}, {-1.f, 2.f, -3.f, 0.5f, -0.5f, 4.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 1, 3}, {1.f, 4.f, 9.f, 0.25f, 0.25f, 16.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL1_NoopWithEmptyAxes_Scalar) {
  OpTester test("ReduceL1", 18);
  test.AddInput<float>("data", {}, {-3.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {}, {3.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL1_NoopWithEmptyAxes_ElementwiseAbs) {
  OpTester test("ReduceL1", 18);
  test.AddInput<float>("data", {4}, {-2.f, 0.f, 3.5f, -4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {4}, {2.f, 0.f, 3.5f, 4.0f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL1_NoopWithEmptyAxes_2D_ElementwiseAbs) {
  OpTester test("ReduceL1", 18);
  test.AddInput<float>("data", {2, 2}, {-2.f, 0.f, 3.5f, -4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 2}, {2.f, 0.f, 3.5f, 4.0f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL1_NoopWithEmptyAxes_3D_ElementwiseAbs) {
  OpTester test("ReduceL1", 18);
  test.AddInput<float>("data", {1, 2, 2}, {-2.f, 0.f, 3.5f, -4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {1, 2, 2}, {2.f, 0.f, 3.5f, 4.0f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL2_NoopWithEmptyAxes_Scalar) {
  OpTester test("ReduceL2", 18);
  test.AddInput<float>("data", {}, {-3.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {}, {3.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL2_NoopWithEmptyAxes_ElementwiseAbs) {
  OpTester test("ReduceL2", 18);
  test.AddInput<float>("data", {3}, {-3.f, 0.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {3}, {3.f, 0.f, 4.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL2_NoopWithEmptyAxes_2D_ElementwiseAbs) {
  OpTester test("ReduceL2", 18);
  test.AddInput<float>("data", {2, 2}, {-3.f, 0.f, 4.f, -1.5f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 2}, {3.f, 0.f, 4.f, 1.5f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceL2_NoopWithEmptyAxes_3D_ElementwiseAbs) {
  OpTester test("ReduceL2", 18);
  test.AddInput<float>("data", {2, 1, 3}, {-3.f, 0.f, 4.f, 1.5f, -2.5f, -1.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 1, 3}, {3.f, 0.f, 4.f, 1.5f, 2.5f, 1.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSum_NoopWithEmptyAxes_Scalar) {
  OpTester test("ReduceLogSum", 18);
  test.AddInput<float>("data", {}, {2.7182817f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {}, {1.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSum_NoopWithEmptyAxes_ElementwiseLog) {
  OpTester test("ReduceLogSum", 18);
  test.AddInput<float>("data", {2}, {2.7182817f, 7.389056f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2}, {1.f, 2.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSum_NoopWithEmptyAxes_2D_ElementwiseLog) {
  OpTester test("ReduceLogSum", 18);
  test.AddInput<float>("data", {2, 2}, {2.7182817f, 7.389056f, 1.6487213f, 20.085537f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 2}, {1.f, 2.f, 0.5f, 3.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSum_NoopWithEmptyAxes_3D_ElementwiseLog) {
  OpTester test("ReduceLogSum", 18);
  test.AddInput<float>("data", {2, 1, 3}, {2.7182817f, 7.389056f, 1.6487213f, 20.085537f, 54.59815f, 148.41316f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 1, 3}, {1.f, 2.f, 0.5f, 3.f, 4.f, 5.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_NoopWithEmptyAxes_Scalar) {
  OpTester test("ReduceLogSumExp", 18);
  test.AddInput<float>("data", {}, {2.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {}, {2.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_NoopWithEmptyAxes_Identity) {
  OpTester test("ReduceLogSumExp", 18);
  test.AddInput<float>("data", {3}, {2.f, -0.5f, 0.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {3}, {2.f, -0.5f, 0.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_NoopWithEmptyAxes_2D_Identity) {
  OpTester test("ReduceLogSumExp", 18);
  test.AddInput<float>("data", {2, 3}, {2.f, -0.5f, 0.f, 1.25f, -3.f, 4.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 3}, {2.f, -0.5f, 0.f, 1.25f, -3.f, 4.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_NoopWithEmptyAxes_3D_Identity) {
  OpTester test("ReduceLogSumExp", 18);
  test.AddInput<float>("data", {2, 1, 3}, {2.f, -0.5f, 0.f, 1.25f, -3.f, 4.f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2, 1, 3}, {2.f, -0.5f, 0.f, 1.25f, -3.f, 4.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

// Edge case: all -inf input should produce -inf (log(sum(exp(-inf))) = log(0) = -inf)
TEST(ReductionOpTest, ReduceLogSumExp_AllNegInf) {
  OpTester test("ReduceLogSumExp", 18);
  float neg_inf = -std::numeric_limits<float>::infinity();
  test.AddInput<float>("data", {5}, {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<float>("reduced", {1}, {neg_inf});
  test.Run();
}

// Edge case: all -inf input with keepdims=0
TEST(ReductionOpTest, ReduceLogSumExp_AllNegInf_NoKeepDims) {
  OpTester test("ReduceLogSumExp", 18);
  test.AddAttribute<int64_t>("keepdims", 0);
  float neg_inf = -std::numeric_limits<float>::infinity();
  test.AddInput<float>("data", {2, 3},
                       {neg_inf, neg_inf, neg_inf,
                        neg_inf, neg_inf, neg_inf});
  test.AddInput<int64_t>("axes", {1}, {1});
  test.AddOutput<float>("reduced", {2}, {neg_inf, neg_inf});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider});
}

// Edge case: mixed -inf and finite values
TEST(ReductionOpTest, ReduceLogSumExp_MixedNegInfAndFinite) {
  OpTester test("ReduceLogSumExp", 18);
  float neg_inf = -std::numeric_limits<float>::infinity();
  // log(exp(-inf) + exp(0)) = log(0 + 1) = 0
  test.AddInput<float>("data", {2}, {neg_inf, 0.f});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<float>("reduced", {1}, {0.f});
  test.Run();
}

// Edge case: +inf input
TEST(ReductionOpTest, ReduceLogSumExp_PosInf) {
  OpTester test("ReduceLogSumExp", 18);
  float pos_inf = std::numeric_limits<float>::infinity();
  // log(exp(+inf) + exp(0)) = log(+inf) = +inf
  test.AddInput<float>("data", {2}, {pos_inf, 0.f});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<float>("reduced", {1}, {pos_inf});
  test.Run();
}

// Edge case: all -inf double precision
TEST(ReductionOpTest, ReduceLogSumExp_AllNegInf_Double) {
  OpTester test("ReduceLogSumExp", 18);
  double neg_inf = -std::numeric_limits<double>::infinity();
  test.AddInput<double>("data", {4}, {neg_inf, neg_inf, neg_inf, neg_inf});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<double>("reduced", {1}, {neg_inf});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_NoopWithAxesNotProvided_ElementwiseSquare) {
  OpTester test("ReduceSumSquare", 18);
  test.AddInput<float>("data", {2}, {2.f, 3.f});
  test.AddAttribute<int64_t>("noop_with_empty_axes", 1);
  test.AddOutput<float>("reduced", {2}, {4.f, 9.f});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

// Opset 20 tests for ReduceMax and ReduceMin on CUDA.
// Verifies CUDA kernel registration at opset 20 works for all supported types.
#if defined(USE_CUDA)

TEST(ReductionOpTest, ReduceMax_float_Opset20_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<float>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMax_double_Opset20_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<double>("reduced", {3, 1, 1}, {4.0, 8.0, 12.0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMax_half_Opset20_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<MLFloat16>("data", {3, 2, 2},
                           FloatsToMLFloat16s({1.0f, 2.0f, 3.0f, 4.0f,
                                               5.0f, 6.0f, 7.0f, 8.0f,
                                               9.0f, 10.0f, 11.0f, 12.0f}));
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<MLFloat16>("reduced", {3, 1, 1}, FloatsToMLFloat16s({4.0f, 8.0f, 12.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMax_int32_Opset20_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<int32_t>("reduced", {3, 1, 1}, {4, 8, 12});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMax_int64_Opset20_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<int64_t>("reduced", {3, 1, 1}, {4, 8, 12});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_float_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<float>("reduced", {3, 1, 1}, {1.0f, 5.0f, 9.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_double_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<double>("reduced", {3, 1, 1}, {1.0, 5.0, 9.0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_half_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<MLFloat16>("data", {3, 2, 2},
                           FloatsToMLFloat16s({1.0f, 2.0f, 3.0f, 4.0f,
                                               5.0f, 6.0f, 7.0f, 8.0f,
                                               9.0f, 10.0f, 11.0f, 12.0f}));
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<MLFloat16>("reduced", {3, 1, 1}, FloatsToMLFloat16s({1.0f, 5.0f, 9.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_int32_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<int32_t>("reduced", {3, 1, 1}, {1, 5, 9});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_int64_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<int64_t>("reduced", {3, 1, 1}, {1, 5, 9});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_int8_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<int8_t>("reduced", {3, 1, 1}, {1, 5, 9});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ReductionOpTest, ReduceMin_uint8_Opset20_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<uint8_t>("reduced", {3, 1, 1}, {1, 5, 9});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Test ReduceMax at opset 20 with keepdims=0 on CUDA
TEST(ReductionOpTest, ReduceMax_float_Opset20_NoKeepdims_Cuda) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<float>("reduced", {3}, {4.0f, 8.0f, 12.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Test ReduceMin at opset 20 with keepdims=0 on CUDA
TEST(ReductionOpTest, ReduceMin_float_Opset20_NoKeepdims_Cuda) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {1, 2});
  test.AddOutput<float>("reduced", {3}, {1.0f, 5.0f, 9.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // defined(USE_CUDA)

// =============================================================================
// Integer overflow saturation tests.
// These test the double-accumulation + saturation paths added to prevent
// signed integer overflow UB in reduction aggregators. Most tests run on
// CPU and CUDA; some are CPU-only where CUDA uses a different code path
// (e.g., ReduceLogSumExp, ReduceSumSquare) or where int64 precision > 2^53
// makes cross-EP comparison unreliable. Other EPs are excluded per-test.
// =============================================================================

TEST(ReductionOpTest, ReduceSum_int32_Overflow_Saturates) {
  OpTester test("ReduceSum", 13);
  // 3 values of ~INT32_MAX/2 + 1 that sum to > INT32_MAX
  int32_t big = 1'100'000'000;
  test.AddInput<int32_t>("data", {3}, {big, big, big});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Expected: saturates to INT32_MAX (3.3B > 2.147B)
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_int32_NegativeOverflow_Saturates) {
  OpTester test("ReduceSum", 13);
  int32_t neg = -1'100'000'000;
  test.AddInput<int32_t>("data", {3}, {neg, neg, neg});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Expected: saturates to INT32_MIN (-3.3B < -2.147B)
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::min()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_int64_Overflow_Saturates) {
  OpTester test("ReduceSum", 13);
  int64_t big = 4'000'000'000'000'000'000LL;  // ~4e18
  test.AddInput<int64_t>("data", {3}, {big, big, big});
  test.AddInput<int64_t>("axes", {1}, {0});
  // 12e18 > INT64_MAX (~9.2e18)
  test.AddOutput<int64_t>("reduced", {1}, {std::numeric_limits<int64_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean_int32_LargeValues_NoOverflow) {
  OpTester test("ReduceMean", 18);
  // Values that would overflow if summed naively in int32, but mean fits in range.
  int32_t big = 2'000'000'000;
  test.AddInput<int32_t>("data", {3}, {big, big, big});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Mean = 2B (fits in int32). Sum would be 6B (overflow).
  test.AddOutput<int32_t>("reduced", {1}, {big});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMean_int64_LargeValues_NoOverflow) {
  OpTester test("ReduceMean", 18);
  int64_t big = 4'000'000'000'000'000'000LL;
  test.AddInput<int64_t>("data", {3}, {big, big, big});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Mean = 4e18 (fits in int64). Sum = 12e18 (overflows int64).
  test.AddOutput<int64_t>("reduced", {1}, {big});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceProd_int32_Overflow_Saturates) {
  OpTester test("ReduceProd", 18);
  // 100000 * 100000 * 100000 = 1e15 > INT32_MAX
  test.AddInput<int32_t>("data", {3}, {100'000, 100'000, 100'000});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceProd_int32_NegativeOverflow_Saturates) {
  OpTester test("ReduceProd", 18);
  // -100000 * 100000 * 100000 = -1e15 < INT32_MIN
  test.AddInput<int32_t>("data", {3}, {-100'000, 100'000, 100'000});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::min()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceSumSquare_int32_Overflow_Saturates) {
  OpTester test("ReduceSumSquare", 18);
  // 50000^2 = 2.5e9 > INT32_MAX for a single element
  test.AddInput<int32_t>("data", {2}, {50'000, 50'000});
  test.AddInput<int64_t>("axes", {1}, {0});
  // 50000^2 + 50000^2 = 5e9 > INT32_MAX → saturate
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::max()});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_int32_Saturation) {
  OpTester test("ReduceLogSumExp", 18);
  // For integers: log(count_of_max) + max. With all same values:
  // log(3) + INT32_MAX ≈ 1.1 + 2147483647 → saturates to INT32_MAX
  int32_t big = std::numeric_limits<int32_t>::max();
  test.AddInput<int32_t>("data", {3}, {big, big, big});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int32_t>("reduced", {1}, {std::numeric_limits<int32_t>::max()});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceLogSumExp_int32_NormalCase) {
  OpTester test("ReduceLogSumExp", 18);
  // For integers: exp(v - max) truncates to 0 unless v == max.
  // With input {5, 5, 3}: max=5, count_of_5=2, result=log(2)+5=5 (truncated)
  test.AddInput<int32_t>("data", {3}, {5, 5, 3});
  test.AddInput<int64_t>("axes", {1}, {0});
  // log(2) ≈ 0.693, truncates to 0 in int, so result = 0 + 5 = 5
  test.AddOutput<int32_t>("reduced", {1}, {5});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(ReductionOpTest, ReduceMax_int32_EmptySet) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {0, 3}, {});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Empty reduction should return lowest() for integers
  test.AddOutput<int32_t>("reduced", {1, 3},
                          {std::numeric_limits<int32_t>::lowest(),
                           std::numeric_limits<int32_t>::lowest(),
                           std::numeric_limits<int32_t>::lowest()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin_int32_EmptySet) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {0, 3}, {});
  test.AddInput<int64_t>("axes", {1}, {0});
  // Empty reduction should return max() for integers
  test.AddOutput<int32_t>("reduced", {1, 3},
                          {std::numeric_limits<int32_t>::max(),
                           std::numeric_limits<int32_t>::max(),
                           std::numeric_limits<int32_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMax_int64_EmptySet) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {0, 2}, {});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int64_t>("reduced", {1, 2},
                          {std::numeric_limits<int64_t>::lowest(),
                           std::numeric_limits<int64_t>::lowest()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDmlExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin_int64_EmptySet) {
  OpTester test("ReduceMin", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {0, 2}, {});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<int64_t>("reduced", {1, 2},
                          {std::numeric_limits<int64_t>::max(),
                           std::numeric_limits<int64_t>::max()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kDmlExecutionProvider, kWebGpuExecutionProvider});
}

// Test empty-set reduction with default axes (reduce all dims) and keepdims=1.
// Validates that CUDA PrepareForReduce produces output shape {1,1,1} (not {1,0,1})
// and fills identity values correctly.
TEST(ReductionOpTest, ReduceSum_EmptySet_DefaultAxes_KeepDims) {
  OpTester test("ReduceSum", 13);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 0, 2}, {});
  test.AddInput<int64_t>("axes", {0}, {}, true);  // empty axes = reduce all
  test.AddOutput<float>("reduced", {1, 1, 1}, {0.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCoreMLExecutionProvider, kDmlExecutionProvider, kDnnlExecutionProvider,
            kMIGraphXExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider,
            kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceMax_EmptySet_DefaultAxes_KeepDims) {
  OpTester test("ReduceMax", 20);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {2, 0, 3}, {});
  test.AddInput<int64_t>("axes", {0}, {}, true);  // empty axes = reduce all
  test.AddOutput<float>("reduced", {1, 1, 1}, {-std::numeric_limits<float>::infinity()});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCoreMLExecutionProvider, kDmlExecutionProvider, kDnnlExecutionProvider,
            kMIGraphXExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider,
            kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(ReductionOpTest, ReduceProd_EmptySet_DefaultAxes_KeepDims) {
  OpTester test("ReduceProd", 18);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {2, 0}, {});
  test.AddInput<int64_t>("axes", {0}, {}, true);  // empty axes = reduce all
  test.AddOutput<float>("reduced", {1, 1}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCoreMLExecutionProvider, kDmlExecutionProvider, kDnnlExecutionProvider,
            kMIGraphXExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider,
            kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
