// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include <cmath>
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/current_test_name.h"
#include "test/util/include/test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "core/util/math.h"
#include <algorithm>
#include <limits>
#include <math.h>
#include <type_traits>

#ifdef USE_DML
#include "core/providers/dml/dml_session_options_config_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#endif

namespace onnxruntime {
namespace test {

TEST(SwiGLUTest, Double) {
  OpTester test("SwiGLU", 28);
  test.SetAllowUnreleasedOnnxOpset();
  test.AddAttribute<float>("alpha", 0.5f);
  test.AddInput<double>("A", {2, 2}, {-2.0, -1.0, 0.0, 2.0});
  test.AddInput<double>("B", {2, 2}, {0.5, 2.0, -3.0, 4.0});
  test.AddOutput<double>("Y", {2, 2},
                         {-1.0 / (1.0 + std::exp(1.0)),
                          -2.0 / (1.0 + std::exp(0.5)),
                          0.0, 8.0 / (1.0 + std::exp(-1.0))});
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
}

TEST(SwiGLUTest, EmptyTensor) {
  for (const auto& dims : {std::vector<int64_t>{0}, std::vector<int64_t>{2, 0}}) {
    OpTester test("SwiGLU", 28);
    test.SetAllowUnreleasedOnnxOpset();
    test.AddInput<float>("A", dims, std::vector<float>{});
    test.AddInput<float>("B", dims, std::vector<float>{});
    test.AddOutput<float>("Y", dims, std::vector<float>{});
    test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
  }
}

std::vector<MLFloat16> MakeMLFloat16(const std::initializer_list<float>& input) {
  std::vector<MLFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 [](float fl) { return MLFloat16(fl); });
  return output;
}

void TestBinaryFloat16(const char* op_name,
                       const std::vector<int64_t>& lhs_dim,
                       const std::initializer_list<float>& lhs_values,
                       const std::vector<int64_t>& rhs_dim,
                       const std::initializer_list<float>& rhs_values,
                       const std::vector<int64_t>& out_dim,
                       const std::initializer_list<float>& out_values,
                       bool enable_bf16 = true) {
  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_COREML
    execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
#elif USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
    if (execution_providers.size() > 0) {
      OpTester tester(op_name, 14);
      tester.AddInput<MLFloat16>("A", lhs_dim, MakeMLFloat16(lhs_values));
      tester.AddInput<MLFloat16>("B", rhs_dim, MakeMLFloat16(rhs_values));
      tester.AddOutput<MLFloat16>("C", out_dim, MakeMLFloat16(out_values));

      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#endif

    if (enable_bf16 && execution_providers.size() > 0) {
      OpTester tester(op_name, 14);
      tester.AddInput<BFloat16>("A", lhs_dim, MakeBFloat16(lhs_values));
      tester.AddInput<BFloat16>("B", rhs_dim, MakeBFloat16(rhs_values));
      tester.AddOutput<BFloat16>("C", out_dim, MakeBFloat16(out_values));

      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

void TestUnaryFloat16(const char* op_name,
                      const std::vector<int64_t>& lhs_dim,
                      const std::initializer_list<float>& lhs_values,
                      const std::vector<int64_t>& out_dim,
                      const std::initializer_list<float>& out_values,
                      int opset = 14,
                      bool run_bf16 = true) {
  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_COREML
    execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
#elif USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
    if (execution_providers.size() > 0) {
      OpTester tester(op_name, opset);
      tester.AddInput<MLFloat16>("A", lhs_dim, MakeMLFloat16(lhs_values));
      tester.AddOutput<MLFloat16>("C", out_dim, MakeMLFloat16(out_values));

      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }

  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#endif

    if (run_bf16 && execution_providers.size() > 0) {
      OpTester tester(op_name, opset);
      tester.AddInput<BFloat16>("A", lhs_dim, MakeBFloat16(lhs_values));
      tester.AddOutput<BFloat16>("C", out_dim, MakeBFloat16(out_values));

      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

void TestBFloat16(const char* op_name, const std::vector<int64_t>& lhs_dim,
                  const std::initializer_list<float>& lhs_values, const std::vector<int64_t>& rhs_dim,
                  const std::initializer_list<float>& rhs_values, const std::vector<int64_t>& out_dim,
                  const std::initializer_list<float>& out_values) {
  {
#ifdef USE_DNNL
    if (!DnnlHasBF16Support()) {
      LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
      return;
    }
#endif
    OpTester tester(op_name, 14);
    tester.AddInput<BFloat16>("A", lhs_dim, MakeBFloat16(lhs_values));
    tester.AddInput<BFloat16>("B", rhs_dim, MakeBFloat16(rhs_values));
    tester.AddOutput<BFloat16>("C", out_dim, MakeBFloat16(out_values));
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
    execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, DimWithZeroHandling) {
  auto run = [](OpTester& tester) {
    // exclude QNN, TensorRT and NNAPI as this isn't handled by those EPs
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "",
               {kTensorrtExecutionProvider, kNnapiExecutionProvider, kQnnExecutionProvider});
  };

  // test binary element-wise op broadcasting when there's a dim with value of zero
  // equal rank
  OpTester test("Add");
  test.AddInput<int64_t>("A", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("B", {3, 0}, {});
  test.AddOutput<int64_t>("C", {3, 0}, {});
  run(test);

  // zero in shape with smaller rank
  OpTester test1("Add");
  test1.AddInput<int64_t>("A", {2, 1, 2}, {1, 2, 3, 4});
  test1.AddInput<int64_t>("B", {0, 2}, {});
  test1.AddOutput<int64_t>("C", {2, 0, 2}, {});
  run(test1);

  // zero in shape with larger rank
  OpTester test2("Add");
  test2.AddInput<int64_t>("A", {0, 2, 2}, {});
  test2.AddInput<int64_t>("B", {1, 2}, {1, 2});
  test2.AddOutput<int64_t>("C", {0, 2, 2}, {});
  run(test2);

  // scalar
  OpTester test3("Add");
  test3.AddInput<int64_t>("A", {}, {1});
  test3.AddInput<int64_t>("B", {0}, {});
  test3.AddOutput<int64_t>("C", {0}, {});
  run(test3);

  // test that BroadcastLoopSpan also works. Mod uses that
  OpTester test4("Mod", 10);
  test4.AddInput<int64_t>("A", {2, 2, 0}, {});
  test4.AddInput<int64_t>("B", {2, 1}, {1, 2});
  test4.AddOutput<int64_t>("C", {2, 2, 0}, {});
  run(test4);

  // test unary op handles it as well
  OpTester test5("Floor");
  test5.AddInput<float>("A", {0, 3}, {});
  test5.AddOutput<float>("B", {0, 3}, {});
  run(test5);
}

TEST(MathOpTest, Add_int8) {
  OpTester test("Add", 14);
  test.AddInput<int8_t>("A", {3}, {1, -2, 3});
  test.AddInput<int8_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int8_t>("C", {3}, {5, 3, 9});
  // Only run this test with CPU and Cuda EPs because on TensorRT EP engine creation fails.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Add_int16) {
  OpTester test("Add", 14);
  test.AddInput<int16_t>("A", {3}, {1, -2, 3});
  test.AddInput<int16_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int16_t>("C", {3}, {5, 3, 9});
  test.Run();
}

TEST(MathOpTest, Add_int32) {
  OpTester test("Add");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int32_t>("C", {3}, {5, 7, 9});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: elementwise inputs must not be Int32
}

TEST(MathOpTest, Add_int64) {
  OpTester test("Add");
  test.AddInput<int64_t>("A", {3}, {1, 2, 3});
  test.AddInput<int64_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int64_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_uint8) {
  OpTester test("Add", 14);
  test.AddInput<uint8_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint8_t>("B", {3}, {4, 5, 6});
  test.AddOutput<uint8_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_uint16) {
  OpTester test("Add", 14);
  test.AddInput<uint16_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint16_t>("B", {3}, {4, 5, 6});
  test.AddOutput<uint16_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_uint32) {
  OpTester test("Add");
  test.AddInput<uint32_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint32_t>("B", {3}, {4, 5, 6});
  test.AddOutput<uint32_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_uint64) {
  OpTester test("Add");
  test.AddInput<uint64_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint64_t>("B", {3}, {4, 5, 6});
  test.AddOutput<uint64_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_float) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  std::initializer_list<float> lhs_values{1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.4f, 9.3f, -10000.0f};
  std::initializer_list<float> rhs_values{-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.3f, 10000.0f};
  std::initializer_list<float> out_values{0.0f, 6.4f, 431.3f, 0.0f, 5.0f, -36.0f, -10.8f, 18.6f, 0.0f};
  test.AddInput<float>("A", dims, lhs_values);
  test.AddInput<float>("B", dims, rhs_values);
  test.AddOutput<float>("C", dims, out_values);

#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kOpenVINOExecutionProvider});  // OpenVINO: Disabled due to accuracy mismatch for FP16
#else
  test.Run();
#endif

  TestBinaryFloat16("Add", dims, lhs_values, dims, rhs_values, dims, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Add", dims, lhs_values, dims, rhs_values, dims, out_values);
#endif  // USE_DNNL
}

TEST(MathOpTest, Add_double) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<double>("A", dims,
                        {1.0, 2.0, -1.0,
                         0.0, 1.5, -100.0,
                         -5.4, 9.3, -10000.0});
  test.AddInput<double>("B", dims,
                        {-1.0, 4.4, 432.3,
                         0.0, 3.5, 64.0,
                         -5.4, 9.3, 10000.0});
  test.AddOutput<double>("C", dims,
                         {0.0, 6.4, 431.3,
                          0.0, 5.0, -36.0,
                          -10.8, 18.6, 0.0});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 3};
  std::initializer_list<float> lhs_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::initializer_list<float> rhs_values{3.0f, 2.0f, 1.0f};
  std::initializer_list<float> out_values{4.0f, 5.0f, 6.0f, 6.0f, 7.0f, 8.0f, 8.0f, 9.0f, 10.0f};
  test.AddInput<float>("A", dims, lhs_values);
  test.AddInput<float>("B", {3, 1}, rhs_values);
  test.AddOutput<float>("C", dims, out_values);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");

  TestBinaryFloat16("Add", dims, lhs_values, {3, 1}, rhs_values, dims, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Add", dims, lhs_values, {3, 1}, rhs_values, dims, out_values);
#endif  // USE_DNNL
}

TEST(MathOpTest, Add_Broadcast_MultidirectionalAB) {
  OpTester test("Add");
  std::initializer_list<float> lhs_values{3.0f, 2.0f, 1.0f};
  std::initializer_list<float> rhs_values{1.0f, 2.0f, 3.0f};
  std::initializer_list<float> out_values{4.0f, 5.0f, 6.0f, 3.0f, 4.0f, 5.0f, 2.0f, 3.0f, 4.0f};
  test.AddInput<float>("A", {3, 1}, lhs_values);
  test.AddInput<float>("B", {3}, rhs_values);
  test.AddOutput<float>("C", {3, 3}, out_values);
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,
            kOpenVINOExecutionProvider});  // OpenVINO: disabled temporarily due to accurarcy issues
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider});  // TensorRT: got C with shape [3, 1]
#endif

  TestBinaryFloat16("Add", {3, 1}, lhs_values, {3}, rhs_values, {3, 3}, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Add", {3, 1}, lhs_values, {3}, rhs_values, {3, 3}, out_values);
#endif  // USE_DNNL
}

TEST(MathOpTest, Add_Broadcast_MultidirectionalBA) {
  OpTester test("Add");
  std::initializer_list<float> lhs_values{1.0f, 2.0f, 3.0f};
  std::initializer_list<float> rhs_values{3.0f, 2.0f, 1.0f};
  std::initializer_list<float> out_values{4.0f, 5.0f, 6.0f, 3.0f, 4.0f, 5.0f, 2.0f, 3.0f, 4.0f};
  test.AddInput<float>("A", {3}, lhs_values);
  test.AddInput<float>("B", {3, 1}, rhs_values);
  test.AddOutput<float>("C", {3, 3}, out_values);
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,
            kOpenVINOExecutionProvider});  // OpenVINO: disabled temporarily due to accurarcy issues
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider});  // TensorRT: got C with shape [3, 1]
#endif

  TestBinaryFloat16("Add", {3}, lhs_values, {3, 1}, rhs_values, {3, 3}, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Add", {3}, lhs_values, {3, 1}, rhs_values, {3, 3}, out_values);
#endif  // USE_DNNL
}

TEST(MathOpTest, Add_Broadcast_0x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{10.0f};
  std::initializer_list<float> rhs_values{2.0f};
  std::initializer_list<float> out_values{12.0f};
  TestBFloat16("Add", {}, lhs_values, {}, rhs_values, {}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_0x1) {
  auto run = [](bool scalar_as_initializer) {
    OpTester test("Add");

    test.AddInput<float>("A", {}, {10.0f}, scalar_as_initializer);
    test.AddInput<float>("B", {1}, {2.0f});
    test.AddOutput<float>("C", {1}, {12.0f});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "");
  };

  run(false);
  run(true);

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{10.0f};
  std::initializer_list<float> rhs_values{2.0f};
  std::initializer_list<float> out_values{12.0f};
  TestBFloat16("Add", {}, lhs_values, {1}, rhs_values, {1}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_1x0) {
  auto run = [](bool scalar_as_initializer) {
    OpTester test("Add");

    test.AddInput<float>("A", {1}, {10.0f});
    test.AddInput<float>("B", {}, {2.0f}, scalar_as_initializer);
    test.AddOutput<float>("C", {1}, {12.0f});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "");
  };

  run(false);
  run(true);

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{10.0f};
  std::initializer_list<float> rhs_values{2.0f};
  std::initializer_list<float> out_values{12.0f};
  TestBFloat16("Add", {1}, lhs_values, {}, rhs_values, {1}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_1x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{10.0f};
  std::initializer_list<float> rhs_values{2.0f};
  std::initializer_list<float> out_values{12.0f};
  TestBFloat16("Add", {1}, lhs_values, {1}, rhs_values, {1}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_3x2_3x1) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 2};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, 3.0f,
                         5.0f, 6.0f,
                         8.0f, 9.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{1.0f, 2.0f,
                                          3.0f, 4.0f,
                                          5.0f, 6.0f};
  std::initializer_list<float> rhs_values{1.0f,
                                          2.0f,
                                          3.0f};
  std::initializer_list<float> out_values{2.0f, 3.0f,
                                          5.0f, 6.0f,
                                          8.0f, 9.0f};
  TestBFloat16("Add", {3, 2}, lhs_values, {3, 1}, rhs_values, {3, 2}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_2x2x2_1x2x2) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 2, 2},
                       {101.0f, 102.0f,
                        103.0f, 104.0f,

                        201.0f, 202.0f,
                        203.0f, 204.0f});
  test.AddInput<float>("B", {1, 2, 2},
                       {010.0f, 020.0f,
                        030.0f, 040.0f});
  test.AddOutput<float>("C", {2, 2, 2},
                        {111.0f, 122.0f,
                         133.0f, 144.0f,

                         211.0f, 222.0f,
                         233.0f, 244.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Add_Broadcast_2x1x4_1x3x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 4},
                       {101.0f, 102.0f, 103.0f, 104.0f,
                        201.0f, 202.0f, 203.0f, 204.0f});
  test.AddInput<float>("B", {1, 3, 1},
                       {010.0f, 020.0f, 030.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{101.0f, 102.0f, 103.0f, 104.0f,
                                          201.0f, 202.0f, 203.0f, 204.0f};
  std::initializer_list<float> rhs_values{010.0f, 020.0f, 030.0f};
  std::initializer_list<float> out_values{111.0f, 112.0f, 113.0f, 114.0f,
                                          121.0f, 122.0f, 123.0f, 124.0f,
                                          131.0f, 132.0f, 133.0f, 134.0f,

                                          211.0f, 212.0f, 213.0f, 214.0f,
                                          221.0f, 222.0f, 223.0f, 224.0f,
                                          231.0f, 232.0f, 233.0f, 234.0f};
  TestBFloat16("Add", {2, 1, 4}, lhs_values, {1, 3, 1}, rhs_values, {2, 3, 4}, out_values);
#endif
}

TEST(MathOpTest, Add_Broadcast_2x1x1_3x4) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 1},
                       {100.0f, 200.0f});
  test.AddInput<float>("B", {3, 4},
                       {011.0f, 012.0f, 013.0f, 014.0f,
                        021.0f, 022.0f, 023.0f, 024.0f,
                        031.0f, 032.0f, 033.0f, 034.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});

  std::unordered_set<std::string> excluded_providers;
  excluded_providers.insert(kTensorrtExecutionProvider);
#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO GPU: Disabled temporarily due to accuracy issues
  // OpenVINO VPU: Disabled due to software limitation
  excluded_providers.insert(kOpenVINOExecutionProvider);
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);  // TensorRT: Input batch size is inconsistent

#if defined(USE_DNNL)
  std::initializer_list<float> lhs_values{100.0f, 200.0f};
  std::initializer_list<float> rhs_values{011.0f, 012.0f, 013.0f, 014.0f,
                                          021.0f, 022.0f, 023.0f, 024.0f,
                                          031.0f, 032.0f, 033.0f, 034.0f};
  std::initializer_list<float> out_values{111.0f, 112.0f, 113.0f, 114.0f,
                                          121.0f, 122.0f, 123.0f, 124.0f,
                                          131.0f, 132.0f, 133.0f, 134.0f,

                                          211.0f, 212.0f, 213.0f, 214.0f,
                                          221.0f, 222.0f, 223.0f, 224.0f,
                                          231.0f, 232.0f, 233.0f, 234.0f};
  TestBFloat16("Add", {2, 1, 1}, lhs_values, {3, 4}, rhs_values, {2, 3, 4}, out_values);
#endif
}

// Validate runtime failure has useful error message when ORT_ENFORCE is used
TEST(MathOpTest, Add_Invalid_Broadcast) {
  OpTester test("Add");

  std::vector<int64_t> dims{2, 3};

  // Use symbolic dimension for first dim so it doesn't fail during shape inferencing
  test.AddShapeToTensorData(true, 0);

  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {0.0f, 0.0f,
                         0.0f, 0.0f,
                         0.0f, 0.0f});

  // Call Run twice to validate different parts of the error message.
  // Only test on CPU as it's that implementation that has the ORT_ENFORCE we're targeting
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Non-zero status code returned while running Add node. Name:'node1'",
           {}, nullptr, &execution_providers);

  // test.Run std::move's the EP from execution_providers into the per-Run session so need to re-create
  execution_providers[0] = DefaultCpuExecutionProvider();
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "axis == 1 || axis == largest was false. "
           "Attempting to broadcast an axis by a dimension other than 1. 2 by 3",
           {}, nullptr, &execution_providers);
}

// TEST(MathOpTest, Add_large_dimension) {
//   OpTester test("Add");

//   int64_t num_elem = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1000;
//   // int64_t num_elem = static_cast<int64_t>(200) + 1000;
//   float input_scalar{4.0f};
//   std::vector<float> input_sequence(num_elem, 0), output_sequence(num_elem, input_scalar);
//   test.AddInput<float>("A", {num_elem}, input_sequence);
//   test.AddInput<float>("B", {1}, {input_scalar});
//   test.AddOutput<float>("C", {num_elem}, output_sequence);

//   std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
//   execution_providers.push_back(DefaultCudaExecutionProvider());
//   test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
// }

TEST(MathOpTest, Sub_int8) {
  OpTester test("Sub", 14);
  test.AddInput<int8_t>("A", {3}, {1, 5, 6});
  test.AddInput<int8_t>("B", {3}, {4, 5, 3});
  test.AddOutput<int8_t>("C", {3}, {-3, 0, 3});
  // Only run this test with CPU and Cuda EPs because on TensorRT EP engine creation fails.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Sub_int16) {
  OpTester test("Sub", 14);
  test.AddInput<int16_t>("A", {3}, {1, 5, 6});
  test.AddInput<int16_t>("B", {3}, {4, 5, 3});
  test.AddOutput<int16_t>("C", {3}, {-3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_int32) {
  OpTester test("Sub");
  test.AddInput<int32_t>("A", {3}, {1, 4, 3});
  test.AddInput<int32_t>("B", {3}, {4, 2, 4});
  test.AddOutput<int32_t>("C", {3}, {-3, 2, -1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Sub_int64) {
  OpTester test("Sub");
  test.AddInput<int64_t>("A", {3}, {1, 5, 6});
  test.AddInput<int64_t>("B", {3}, {4, 5, 3});
  test.AddOutput<int64_t>("C", {3}, {-3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_uint8) {
  OpTester test("Sub", 14);
  test.AddInput<uint8_t>("A", {3}, {4, 5, 6});
  test.AddInput<uint8_t>("B", {3}, {1, 5, 3});
  test.AddOutput<uint8_t>("C", {3}, {3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_uint16) {
  OpTester test("Sub", 14);
  test.AddInput<uint16_t>("A", {3}, {4, 5, 6});
  test.AddInput<uint16_t>("B", {3}, {1, 5, 3});
  test.AddOutput<uint16_t>("C", {3}, {3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_uint32) {
  OpTester test("Sub");
  test.AddInput<uint32_t>("A", {3}, {4, 5, 6});
  test.AddInput<uint32_t>("B", {3}, {1, 5, 3});
  test.AddOutput<uint32_t>("C", {3}, {3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_uint64) {
  OpTester test("Sub");
  test.AddInput<uint64_t>("A", {3}, {4, 5, 6});
  test.AddInput<uint64_t>("B", {3}, {1, 5, 3});
  test.AddOutput<uint64_t>("C", {3}, {3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub_float) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  std::initializer_list<float> lhs_values{1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.4f, 9.3f, -10000.0f};
  std::initializer_list<float> rhs_values{-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.3f, 10000.0f};
  std::initializer_list<float> out_values{2.0f, -2.4f, -433.3f, 0.0f, -2.0f, -164.0f, 0.0f, 0.0f, -20000.0f};
  test.AddInput<float>("A", dims, lhs_values);
  test.AddInput<float>("B", dims, rhs_values);
  test.AddOutput<float>("C", dims, out_values);
  test.Run();

  TestBinaryFloat16("Sub", dims, lhs_values, dims, rhs_values, dims, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Sub", dims, lhs_values, dims, rhs_values, dims, out_values);
#endif
}

TEST(MathOpTest, Sub_double) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<double>("A", dims, {1.0, 2.0, -1.0, 0.0, 1.5, -100.0, -5.4, 9.3, -10000.0});
  test.AddInput<double>("B", dims, {-1.0, 4.4, 432.3, 0.0, 3.5, 64.0, -5.4, 9.3, 10000.0});
  test.AddOutput<double>("C", dims, {2.0, -2.4, -433.3, 0.0, -2.0, -164.0, 0.0, 0.0, -20000.0});
  test.Run();
}

TEST(MathOpTest, Sub_Broadcast_Scalar) {
  auto run = [](bool scalar_as_initializer) {
    OpTester test("Sub");
    std::vector<int64_t> dims{3, 3};
    test.AddInput<float>("A", dims,
                         {1.0f, 2.0f, -1.0f,
                          0.0f, 1.5f, -100.0f,
                          -5.4f, 9.3f, -10000.0f});
    test.AddInput<float>("B", {}, {5.0f}, scalar_as_initializer);
    test.AddOutput<float>("C", dims,
                          {-4.0f, -3.0f, -6.0f,
                           -5.0f, -3.5f, -105.0f,
                           -10.4f, 4.3f, -10005.0f});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "");
  };

  run(false);
  run(true);
}

TEST(MathOpTest, Mul_int8) {
  OpTester test("Mul", 14);
  test.AddInput<int8_t>("A", {3}, {1, 2, 3});
  test.AddInput<int8_t>("B", {3}, {4, -3, 6});
  test.AddOutput<int8_t>("C", {3}, {4, -6, 18});
  // Only run this test with CPU and Cuda EPs because on TensorRT EP engine creation fails.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Mul_int16) {
  OpTester test("Mul", 14);
  test.AddInput<int16_t>("A", {3}, {3, 6, -3});
  test.AddInput<int16_t>("B", {3}, {4, -3, -2});
  test.AddOutput<int16_t>("C", {3}, {12, -18, 6});
  test.Run();
}

TEST(MathOpTest, Mul_int32) {
  OpTester test("Mul");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, -3, 6});
  test.AddOutput<int32_t>("C", {3}, {4, -6, 18});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Mul_int64) {
  OpTester test("Mul");
  test.AddInput<int64_t>("A", {3}, {3, 6, -3});
  test.AddInput<int64_t>("B", {3}, {4, -3, -2});
  test.AddOutput<int64_t>("C", {3}, {12, -18, 6});
  test.Run();
}

TEST(MathOpTest, Mul_uint8) {
  OpTester test("Mul", 14);
  test.AddInput<uint8_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint8_t>("B", {3}, {4, 3, 6});
  test.AddOutput<uint8_t>("C", {3}, {4, 6, 18});
  test.Run();
}

TEST(MathOpTest, Mul_uint16) {
  OpTester test("Mul", 14);
  test.AddInput<uint16_t>("A", {3}, {3, 6, 3});
  test.AddInput<uint16_t>("B", {3}, {4, 3, 2});
  test.AddOutput<uint16_t>("C", {3}, {12, 18, 6});
  test.Run();
}

TEST(MathOpTest, Mul_uint32) {
  OpTester test("Mul");
  test.AddInput<uint32_t>("A", {3}, {1, 2, 3});
  test.AddInput<uint32_t>("B", {3}, {4, 3, 6});
  test.AddOutput<uint32_t>("C", {3}, {4, 6, 18});
  test.Run();
}

TEST(MathOpTest, Mul_uint64) {
  OpTester test("Mul");
  test.AddInput<uint64_t>("A", {3}, {3, 6, 3});
  test.AddInput<uint64_t>("B", {3}, {4, 3, 2});
  test.AddOutput<uint64_t>("C", {3}, {12, 18, 6});
  test.Run();
}

TEST(MathOpTest, Mul_float) {
  OpTester test("Mul");
  std::vector<int64_t> dims{3, 3};
  std::initializer_list<float> lhs_values{1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.0f, 9.3f, -10000.0f};
  std::initializer_list<float> rhs_values{-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.0f, 10000.0f};
  std::initializer_list<float> out_values{-1.0f, 8.8f, -432.3f, 0.0f, 5.25f, -6400.0f, 27.0f, 83.7f, -100000000.0f};
  test.AddInput<float>("A", dims, lhs_values);
  test.AddInput<float>("B", dims, rhs_values);
  test.AddOutput<float>("C", dims, out_values);

  test.Run();

  TestBinaryFloat16("Mul", dims, lhs_values, dims, rhs_values, dims, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Mul", dims, lhs_values, dims, rhs_values, dims, out_values);
#endif
}

TEST(MathOpTest, Mul_double) {
  OpTester test("Mul");
  test.AddInput<double>("A", {3, 3}, {1.0, 2.0, -1.0, 0.0, 1.5, -100.0, -5.0, 9.3, -10000.0});
  test.AddInput<double>("B", {3, 3}, {-1.0, 4.4, 432.3, 0.0, 3.5, 64.0, -5.4, 9.0, 10000.0});
  test.AddOutput<double>("C", {3, 3}, {-1.0, 8.8, -432.3, 0.0, 5.25, -6400.0, 27.0, 83.7, -100000000.0});
  test.Run();
}

TEST(MathOpTest, Div_int8) {
  OpTester test("Div", 14);
  test.AddInput<int8_t>("A", {3}, {4, 8, 8});
  test.AddInput<int8_t>("B", {3}, {1, -3, 2});
  test.AddOutput<int8_t>("C", {3}, {4, -2, 4});
  // Only run this test with CPU and Cuda EPs because on TensorRT EP engine creation fails and on
  // DNNL EP the result for (8 / -3) is -3 instead of -2.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Div_int16) {
  OpTester test("Div", 14);
  test.AddInput<int16_t>("A", {3}, {4, 8, -8});
  test.AddInput<int16_t>("B", {3}, {2, 3, 4});
  test.AddOutput<int16_t>("C", {3}, {2, 2, -2});
  test.Run();
}

TEST(MathOpTest, Div_int32) {
  OpTester test("Div");
  test.AddInput<int32_t>("A", {3}, {4, 8, 8});
  test.AddInput<int32_t>("B", {3}, {1, 3, 2});
  test.AddOutput<int32_t>("C", {3}, {4, 2, 4});
  // ov parser accuracy mismatch for div
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Div_int64) {
  OpTester test("Div");
  test.AddInput<int64_t>("A", {3}, {4, 8, 8});
  test.AddInput<int64_t>("B", {3}, {2, 3, 4});
  test.AddOutput<int64_t>("C", {3}, {2, 2, 2});
  // ov parser accuracy mismatch for div
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Div_uint8) {
  OpTester test("Div", 14);
  test.AddInput<uint8_t>("A", {3}, {4, 8, 8});
  test.AddInput<uint8_t>("B", {3}, {1, 3, 2});
  test.AddOutput<uint8_t>("C", {3}, {4, 2, 4});
  // Only run this test with CPU and Cuda EPs because on DNNL EP the result for (8 / 3) is 3 instead of -2.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Div_uint16) {
  OpTester test("Div", 14);
  test.AddInput<uint16_t>("A", {3}, {4, 8, 8});
  test.AddInput<uint16_t>("B", {3}, {2, 3, 4});
  test.AddOutput<uint16_t>("C", {3}, {2, 2, 2});
  test.Run();
}

TEST(MathOpTest, Div_uint32) {
  OpTester test("Div");
  test.AddInput<uint32_t>("A", {3}, {4, 8, 8});
  test.AddInput<uint32_t>("B", {3}, {1, 3, 2});
  test.AddOutput<uint32_t>("C", {3}, {4, 2, 4});
  test.Run();
}

TEST(MathOpTest, Div_uint64) {
  OpTester test("Div");
  test.AddInput<uint64_t>("A", {3}, {4, 8, 8});
  test.AddInput<uint64_t>("B", {3}, {2, 3, 4});
  test.AddOutput<uint64_t>("C", {3}, {2, 2, 2});
  test.Run();
}

TEST(MathOpTest, Div_int8_by_zero) {
  OpTester test("Div", 14);
  test.AddInput<int8_t>("A", {3}, {4, 8, 8});
  test.AddInput<int8_t>("B", {3}, {1, 0, 2});
  test.AddOutput<int8_t>("C", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer division by zero",
           {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Div_int32_by_zero) {
  OpTester test("Div");
  test.AddInput<int32_t>("A", {3}, {4, 8, 8});
  test.AddInput<int32_t>("B", {3}, {1, 0, 2});
  test.AddOutput<int32_t>("C", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer division by zero",
           {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Div_int64_by_zero_scalar) {
  // Scalar divisor of 0 (the exact scenario from the bug report)
  OpTester test("Div");
  test.AddInput<int64_t>("A", {3}, {4, 8, 8});
  test.AddInput<int64_t>("B", {}, {0});
  test.AddOutput<int64_t>("C", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer division by zero",
           {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Div_int32_by_zero_constant_initializer) {
  // Divisor is a constant initializer — validated once at kernel creation time
  OpTester test("Div");
  test.AddInput<int32_t>("A", {3}, {4, 8, 8});
  test.AddInput<int32_t>("B", {3}, {1, 0, 2}, true);  // is_initializer = true
  test.AddOutput<int32_t>("C", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer division by zero",
           {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Div_float) {
  OpTester test("Div");
  std::vector<int64_t> dims{2, 3};
  std::initializer_list<float> lhs_values{1000.0f, 1.0f, 6.0f, 0.0f, -10.0f, -1.0f};
  std::initializer_list<float> rhs_values{1000.0f, 2.0f, 3.0f, 1.0f, -1.0f, 4.0f};
  std::initializer_list<float> out_values{1.0f, 0.5f, 2.0f, 0.0f, 10.0f, -0.25f};
  test.AddInput<float>("A", dims, lhs_values);
  test.AddInput<float>("B", dims, rhs_values);
  test.AddOutput<float>("C", dims, out_values);
  test.Run();

  TestBinaryFloat16("Div", dims, lhs_values, dims, rhs_values, dims, out_values);

#if defined(USE_DNNL)
  TestBFloat16("Div", dims, lhs_values, dims, rhs_values, dims, out_values);
#endif
}

TEST(MathOpTest, Div_double) {
  OpTester test("Div");
  test.AddInput<double>("A", {2, 3}, {1000.0, 1.0, 6.0, 0.0, -10.0, -1.0});
  test.AddInput<double>("B", {2, 3}, {1000.0, 2.0, 3.0, 1.0, -1.0, 4.0});
  test.AddOutput<double>("C", {2, 3}, {1.0, 0.5, 2.0, 0.0, 10.0, -0.25});
  test.Run();
}

TEST(MathOpTest, Abs) {
  OpTester test("Abs");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 0.0f, 10.0f});
  test.Run();
}

#ifdef USE_CUDA
TEST(MathOpTest, Abs_Cuda_CanonicalizesSignedZero) {
  OpTester test("Abs");
  test.AddInput<float>("X", {2}, {0.0f, -0.0f});
  test.AddOutput<float>("Y", {2}, {0.0f, 0.0f});
  test.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches,
                                  const std::string& /*provider_type*/) {
    ASSERT_EQ(fetches.size(), 1u);
    ASSERT_TRUE(fetches[0].IsTensor());

    const auto* output = fetches[0].Get<Tensor>().Data<float>();
    EXPECT_FALSE(std::signbit(output[0])) << "Abs(+0.0f) must be +0.0f";
    EXPECT_FALSE(std::signbit(output[1])) << "Abs(-0.0f) must be +0.0f";
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA

#if defined(USE_CUDA) || defined(USE_DNNL)
TEST(MathOpTest, Abs_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test_bf16("Abs", 13);
  std::vector<int64_t> dims{2, 2};
  test_bf16.AddInput<BFloat16>("X", dims, MakeBFloat16({1.0f, -2.0f, -0.0f, -10.0f}));
  test_bf16.AddOutput<BFloat16>("Y", dims, MakeBFloat16({1.0f, 2.0f, 0.0f, 10.0f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_CUDA)
  execution_providers.push_back(DefaultCudaExecutionProvider());
#endif

#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif

  test_bf16.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_CUDA || USE_DNNL

TEST(MathOpTest, Abs_int8) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int8_t>("Y", dims, {1, 2, 1, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: INT8, Assertion `regionRanges != nullptr' failed
}

TEST(MathOpTest, Abs_int32) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int32_t>("Y", dims, {1, 2, 1, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Neg_float) {
  OpTester test("Neg");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, -2.0f,
                        0.0f, -10.0f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 2.0f,
                         -0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Neg_double) {
  OpTester test("Neg");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {1.0, -2.0,
                         0.0, -10.0});
  test.AddOutput<double>("Y", dims,
                         {-1.0, 2.0,
                          -0.0, 10.0});
  test.Run();
}

TEST(MathOpTest, Neg_int8) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int8_t>("Y", dims, {-1, 2, 0, 10});

  // OpenVINO EP: Disabled temporarily
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: INT8 is not supported
}

TEST(MathOpTest, Neg_int16) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int8_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Int16 not allowed as input to this layer
}

TEST(MathOpTest, Neg_int32) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int32_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Neg_int64) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int64_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: Int64 not allowed as input to this layer
}

TEST(MathOpTest, Floor) {
  OpTester test("Floor");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-2.0f, 0.0f,
                         -1.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Floor_double) {
  OpTester test("Floor");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {-1.5, 0.2,
                         -0.5, 10.3});
  test.AddOutput<double>("Y", dims,
                         {-2.0, 0.0,
                          -1.0, 10.0});
  test.Run();
}

TEST(MathOpTest, Ceil) {
  OpTester test("Ceil");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 1.0f,
                         0.0f, 11.0f});
#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled due to software limitation for GPU and VPU Plugins.
  // This test runs fine on CPU Plugin
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
#else
  test.Run();
#endif
}

TEST(MathOpTest, Ceil_double) {
  OpTester test("Ceil");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {-1.5, 0.2,
                         -0.5, 10.3});
  test.AddOutput<double>("Y", dims,
                         {-1.0, 1.0,
                          0.0, 11.0});
#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled due to software limitation for GPU and VPU Plugins.
  // This test runs fine on CPU Plugin
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
#else
  test.Run();
#endif
}

TEST(MathOpTest, Reciprocal) {
  OpTester test("Reciprocal");
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<float> inputs = {1.0f, 2.0f, -1.0f, -2.0f};
  std::initializer_list<float> outputs = {1.0f, 0.5f, -1.0f, -0.5f};
  test.AddInput<float>("X", dims, inputs);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
  TestUnaryFloat16("Reciprocal", dims, inputs, dims, outputs, 12, false);
}

TEST(MathOpTest, Reciprocal_double) {
  OpTester test("Reciprocal");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {1.0, 2.0,
                         -1.0, -2.0});
  test.AddOutput<double>("Y", dims,
                         {1.0, 0.5,
                          -1.0, -0.5});
  test.Run();
}

TEST(MathOpTest, Sqrt_Float) {
  OpTester test("Sqrt");
  std::initializer_list<float> inputs = {1.0f, 4.0f, 0.0f, 9.0f};
  std::initializer_list<float> outputs = {1.0f, 2.0f, 0.0f, 3.0f};
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, inputs);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
  TestUnaryFloat16("Sqrt", dims, inputs, dims, outputs);
}

#if defined(USE_DNNL) || defined(USE_CUDA)
TEST(MathOpTest, Sqrt_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BFP16";
    return;
  }
#endif
  OpTester test_bf16("Sqrt", 13);  // only version 13 support bf16 for sqrt
  test_bf16.AddInput<BFloat16>("X", {2, 3},
                               MakeBFloat16({1.0f, 4.0f,
                                             0.0f, 9.0f, 2.0f, 5.0f}));
  test_bf16.AddOutput<BFloat16>("Y", {2, 3},
                                MakeBFloat16({1.0f, 2.0f,
                                              0.0f, 3.0f, 1.414213562f, 2.236067977f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
  test_bf16.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(MathOpTest, Sqrt_Double) {
  OpTester test("Sqrt");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {1.0, 4.0,
                         0.0, 9.0});
  test.AddOutput<double>("Y", dims,
                         {1.0, 2.0,
                          0.0, 3.0});
  test.Run();
}

TEST(MathOpTest, Pow_Float) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        std::sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f});
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Double) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {2.0, 2.0,
                         std::sqrt(2.0), 1.0});
  test.AddInput<double>("Y", dims,
                        {0.0, 8.0,
                         2.0, 9.0});
  test.AddOutput<double>("Z", dims,
                         {1.0, 256.0,
                          2.0, 1.0});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar0) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", {}, {2.0f});
  test.AddInput<float>("Y", dims, {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Z", dims, {2.0f, 4.0f, 8.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar1) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Float_12) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        std::sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f});
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Float_15) {
  OpTester test("Pow", 15);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        std::sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f},
                       true);
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

#if defined(USE_DNNL) || defined(USE_CUDA)
TEST(MathOpTest, Pow_bfloat16_15) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#ifdef USE_CUDA
  if (!CudaHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test_bf16("Pow", 15);
  test_bf16.AddInput<BFloat16>("X", {2, 2},
                               MakeBFloat16({2.0f, 2.0f,
                                             std::sqrt(2.0f), 1.0f}));
  test_bf16.AddInput<BFloat16>("Y", {},
                               MakeBFloat16({0.0f}),
                               true);
  test_bf16.AddOutput<BFloat16>("Z", {2, 2},
                                MakeBFloat16({1.0f, 1.0f,
                                              1.0f, 1.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
#if defined(USE_CUDA)
  execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
  test_bf16.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(MathOpTest, Pow_Double_12) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {2.0, 2.0,
                         std::sqrt(2.0), 1.0});
  test.AddInput<double>("Y", dims,
                        {0.0, 8.0,
                         2.0, 9.0});
  test.AddOutput<double>("Z", dims,
                         {1.0, 256.0,
                          2.0, 1.0});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar0_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", {}, {2.0f});
  test.AddInput<float>("Y", dims, {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Z", dims, {2.0f, 4.0f, 8.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar1_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar1_float_int32_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int32_t>("Y", {}, {3});
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}

TEST(MathOpTest, Pow_float_int64) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("Y", dims, {4, 5, 6});
  test.AddOutput<float>("Z", dims, {1.f, 32.f, 729.f});
  test.Run();
}

TEST(MathOpTest, Pow_int64_float) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("X", dims, {1, 2, 3});
  test.AddInput<float>("Y", dims, {4.f, 5.f, 6.f});
  test.AddOutput<int64_t>("Z", dims, {1, 32, 729});
  test.Run();
}

TEST(MathOpTest, Pow_float_int32) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int32_t>("Y", dims, {4, 5, 6});
  test.AddOutput<float>("Z", dims, {1.f, 32.f, 729.f});
  test.Run();
}

TEST(MathOpTest, Pow_int32_float) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<int32_t>("X", dims, {1, 2, 3});
  test.AddInput<float>("Y", dims, {4.f, 5.f, 6.f});
  test.AddOutput<int32_t>("Z", dims, {1, 32, 729});
  test.Run();
}

TEST(MathOpTest, Pow_int64_double) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("X", dims, {1, 2, 3});
  test.AddInput<double>("Y", dims, {4.f, 5.f, 6.f});
  test.AddOutput<int64_t>("Z", dims, {1, 32, 729});
  test.Run();
}

TEST(MathOpTest, Pow_double_int64) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{3};
  test.AddInput<double>("X", dims, {1., 2., 3.});
  test.AddInput<int64_t>("Y", dims, {4, 5, 6});
  test.AddOutput<double>("Z", dims, {1., 32., 729.});
  test.Run();
}

TEST(MathOpTest, Pow_float16_float16) {
  std::vector<int64_t> dims{4};
  TestBinaryFloat16("Pow", dims, {2.0f, 2.0f, std::sqrt(2.0f), 1.0f}, dims, {0.0f, 8.0f, 2.0f, 9.0f},
                    dims, {1.0f, 256.0f, 2.0f, 1.0f}, false);
}

#if defined(USE_CUDA) || defined(USE_COREML)
TEST(MathOpTest, Pow_float_float16) {
  OpTester test("Pow", 12);
  std::vector<int64_t> dims{4};

  test.AddInput<MLFloat16>("X", dims, MakeMLFloat16({2.0f, 2.0f, std::sqrt(2.0f), 1.0f}));
  test.AddInput<float>("Y", dims, {0.0f, 8.0f, 2.0f, 9.0f});
  test.AddOutput<MLFloat16>("Z", dims, MakeMLFloat16({1.0f, 256.0f, 2.0f, 1.0f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_COREML
  execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

#if defined(USE_DNNL)
TEST(MathOpTest, Exp_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Exp", 13);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<BFloat16>("X", dims,
                          MakeBFloat16({0.0f, 1.0f,
                                        2.0f, 10.0f}));
  test.AddOutput<BFloat16>("Y", dims,
                           MakeBFloat16({1.0f, std::exp(1.0f),
                                         std::exp(2.0f), std::exp(10.0f)}));
  test.SetOutputRelErr("Y", 1e-7f);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);  // TensorRT: result differs
}
#endif
TEST(MathOpTest, Exp_float) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {0.0f, 1.0f,
                        2.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, std::exp(1.0f),
                         std::exp(2.0f), std::exp(10.0f)});
  test.SetOutputRelErr("Y", 2e-7f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: result differs
}

TEST(MathOpTest, Exp_double) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {0.0, 1.0,
                         2.0, 10.0});
  test.AddOutput<double>("Y", dims,
                         {1.0, std::exp(1.0),
                          std::exp(2.0), std::exp(10.0)});
  test.SetOutputRelErr("Y", 1e-7f);
  // TODO: Check if this test's result really differs for tensorRT
  // For now basing this exclusion based on this test's float counterpart - Exp_float
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Log) {
  OpTester test("Log");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        5.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {0.0f, std::log(2.0f),
                         std::log(5.0f), std::log(10.0f)});
  test.Run();
}

#if defined(USE_DNNL)
TEST(MathOpTest, Log_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Log", 13);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<BFloat16>("X", dims,
                          MakeBFloat16({1.0f, 2.0f,
                                        5.0f, 10.0f}));
  test.AddOutput<BFloat16>("Y", dims,
                           MakeBFloat16({0.0f, std::log(2.0f),
                                         std::log(5.0f), std::log(10.0f)}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL
TEST(MathOpTest, Log_double) {
  OpTester test("Log");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {1.0, 2.0,
                         5.0, 10.0});
  test.AddOutput<double>("Y", dims,
                         {0.0, std::log(2.0),
                          std::log(5.0), std::log(10.0)});
  test.SetOutputRelErr("Y", 1e-7f);
  test.Run();
}

TEST(MathOpTest, Sum_6) {
  OpTester test("Sum", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.25f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10000.0f});
  test.AddOutput<float>("sum", dims,
                        {3.0f, 0.0f, 6.0f,
                         -6.0f, 6.6f, 28.0f,
                         -1.0f, 0.06f, 0.25f});

#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO EP: Disabled due to accuracy mismatch for FP16
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TRT10: Disabled due to segfault caused by older opset 6
#endif
}

TEST(MathOpTest, Sum_6_double) {
  OpTester test("Sum", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<double>("data_0", dims,
                        {1.0, 0.0, 1.0,
                         -1.0, 1.1, -100.0,
                         -5.4, 0.01, -10000.0});
  test.AddInput<double>("data_1", dims,
                        {1.0, 0.0, 2.0,
                         -2.0, 2.2, 64.0,
                         -1.0, 0.02, 0.25});
  test.AddInput<double>("data_3", dims,
                        {1.0, 0.0, 3.0,
                         -3.0, 3.3, 64.0,
                         5.4, 0.03, 10000.0});
  test.AddOutput<double>("sum", dims,
                         {3.0, 0.0, 6.0,
                          -6.0, 6.6, 28.0,
                          -1.0, 0.06, 0.25});

#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO EP: Disabled due to accuracy mismatch for FP16
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TRT10: Disabled due to segfault caused by older opset 6
#endif
}

TEST(MathOpTest, Sum_8_Test1) {
  OpTester test("Sum", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 1}, {10.0f, 20.0f, 30.0f});
  test.AddInput<float>("data_2", {3, 1, 1}, {100.0f, 200.0f, 300.0f});
  test.AddOutput<float>("sum", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled due to software limitation for GPU and VPU Plugins.
  // This test runs fine on CPU Plugin
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Expected output shape [{3,3,3}] did not match run output shape [{3,1,1}] for sum
#endif
}

TEST(MathOpTest, Sum_8_Test1_double) {
  OpTester test("Sum", 8);
  test.AddInput<double>("data_0", {3}, {1.0, 2.0, 3.0});
  test.AddInput<double>("data_1", {3, 1}, {10.0, 20.0, 30.0});
  test.AddInput<double>("data_2", {3, 1, 1}, {100.0, 200.0, 300.0});
  test.AddOutput<double>("sum", {3, 3, 3},
                         {111.0, 112.0, 113.0,
                          121.0, 122.0, 123.0,
                          131.0, 132.0, 133.0,

                          211.0, 212.0, 213.0,
                          221.0, 222.0, 223.0,
                          231.0, 232.0, 233.0,

                          311.0, 312.0, 313.0,
                          321.0, 322.0, 323.0,
                          331.0, 332.0, 333.0});
#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled due to software limitation for GPU and VPU Plugins.
  // This test runs fine on CPU Plugin
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Expected output shape [{3,3,3}] did not match run output shape [{3,1,1}] for sum
#endif
}
TEST(MathOpTest, Sum_8_Test2) {
  OpTester test("Sum", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {
                           1.0f,
                           0.0f,
                           1.0f,
                           -1.0f,
                           1.1f,
                           -100.0f,
                           -5.4f,
                           0.01f,
                           -74.0f,
                       });
  std::vector<int64_t> dims_1{3};
  test.AddInput<float>("data_1", dims_1,
                       {1.0f, 0.0f, 2.0f});
  std::vector<int64_t> dims_2{3, 1};
  test.AddInput<float>("data_2", dims_2,
                       {-3.0f, 3.3f, 64.0f});
  test.AddOutput<float>("sum", dims,
                        {-1.0f, -3.0f, 0.0f,
                         3.3f, 4.4f, -94.7f,
                         59.6f, 64.01f, -8.0f});

#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled temporarily due to accuracy issues
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "Sum is not correct", {kTensorrtExecutionProvider});  // TensorRT: result differs
#endif
}

TEST(MathOpTest, Sum_8_Test2_double) {
  OpTester test("Sum", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<double>("data_0", dims,
                        {
                            1.0,
                            0.0,
                            1.0,
                            -1.0,
                            1.1,
                            -100.0,
                            -5.4,
                            0.01,
                            -74.0,
                        });
  std::vector<int64_t> dims_1{3};
  test.AddInput<double>("data_1", dims_1,
                        {1.0, 0.0, 2.0});
  std::vector<int64_t> dims_2{3, 1};
  test.AddInput<double>("data_2", dims_2,
                        {-3.0, 3.3, 64.0});
  test.AddOutput<double>("sum", dims,
                         {-1.0, -3.0, 0.0,
                          3.3, 4.4, -94.7,
                          59.6, 64.01, -8.0});

#if defined(OPENVINO_CONFIG_GPU)
  // OpenVINO: Disabled temporarily due to accuracy issues
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "Sum is not correct", {kTensorrtExecutionProvider});  // TensorRT: result differs
#endif
}

template <typename T>
static void TestSumMultipleInputsNoBroadcasting(size_t num_inputs, const TensorShape& shape) {
  using element_type = T;

  OpTester test{"Sum", 8};

  const auto dims = shape.AsShapeVector();
  const std::vector<element_type> input_data(shape.Size(), 1);

  for (size_t i = 0; i < num_inputs; ++i) {
    test.AddInput<element_type>(MakeString("data_", i).c_str(), dims, input_data);
  }

  const std::vector<element_type> expected_output_data =
      [&input_data, num_inputs]() {
        std::vector<element_type> result;
        std::transform(
            input_data.begin(), input_data.end(), std::back_inserter(result),
            [num_inputs](element_type value) { return num_inputs * value; });
        return result;
      }();

  test.AddOutput<element_type>("sum", dims, expected_output_data);

  // TRT EP segmentation fault in A100
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(MathOpTest, SumMultipleInputsNoBroadcasting) {
  const TensorShape shape{3, 3, 3};
  // Special case:
  //   2: BinaryImplDispatchTarget
  //   3-8: NoBroadcastBatchImplDispatchTarget(i)
  //   9: NoBroadcastBatchImplDispatchTarget(8) + BinaryImplDispatchTarget
  //   10: NoBroadcastBatchImplDispatchTarget(8) + NoBroadcastBatchImplDispatchTarget(3)
  //   15: NoBroadcastBatchImplDispatchTarget(8) + NoBroadcastBatchImplDispatchTarget(8)
  //   16: NoBroadcastBatchImplDispatchTarget(8) + NoBroadcastBatchImplDispatchTarget(8) + BinaryImplDispatchTarget
  for (size_t num_inputs = 2; num_inputs <= 10; ++num_inputs) {
    TestSumMultipleInputsNoBroadcasting<float>(num_inputs, shape);
  }
  TestSumMultipleInputsNoBroadcasting<float>(15, shape);
  TestSumMultipleInputsNoBroadcasting<float>(16, shape);
}

TEST(MathOpTest, SumMultipleInputsNoBroadcasting_double) {
  const TensorShape shape{3, 3, 3};
  for (size_t num_inputs = 2; num_inputs < 10; ++num_inputs) {
    TestSumMultipleInputsNoBroadcasting<double>(num_inputs, shape);
  }
}

#if defined(USE_DNNL)
TEST(MathOpTest, Sum_13_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Sum", 13);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<BFloat16>("data_0", dims,
                          MakeBFloat16({1.0f, 0.0f, 1.0f,
                                        -1.0f, 1.1f, -100.0f,
                                        -5.4f, 0.01f, -10000.0f}));
  test.AddInput<BFloat16>("data_1", dims,
                          MakeBFloat16({1.0f, 0.0f, 2.0f,
                                        -2.0f, 2.2f, 64.0f,
                                        -1.0f, 0.02f, 0.25f}));
  test.AddInput<BFloat16>("data_3", dims,
                          MakeBFloat16({1.0f, 0.0f, 3.0f,
                                        -3.0f, 3.3f, 64.0f,
                                        5.4f, 0.03f, 10000.0f}));
  test.AddOutput<BFloat16>("sum", dims,
                           MakeBFloat16({3.0f, 0.0f, 6.0f,
                                         -6.0f, 6.6f, 28.0f,
                                         -1.0f, 0.06f, 0.25f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, Min_6) {
  OpTester test("Min", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10000.0f});
  test.AddOutput<float>("sum", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10000.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TRT10: Disabled due to segfault caused by older opset 6
}

TEST(MathOpTest, Min_8) {
  OpTester test("Min", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10000.0f});
  test.AddOutput<float>("min", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10000.0f});
  test.Run();
}

TEST(MathOpTest, Min_12_Float) {
  OpTester test("Min", 12);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        -70.0f, -80.0f, -90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, 20.0f, 300.0f});
  test.AddOutput<float>("min", {3, 3},
                        {-1.0f, -1.0f, -1.0f,
                         1.0f, 2.0f, 3.0f,
                         -70.0f, -80.0f, -90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_Float_2_Input) {
  OpTester test("Min", 12);
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        -70.0f, -80.0f, -90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, 20.0f, 300.0f});
  test.AddOutput<float>("min", {3, 3},
                        {-1.0f, -1.0f, -1.0f,
                         20.0f, 20.0f, 20.0f,
                         -70.0f, -80.0f, -90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_Float_Nan) {
  OpTester test("Min", 12);
  test.AddInput<float>("data_2", {3, 3},
                       {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        -0.5f, 0.0f, -2.0f,
                        0.5f, 0.0f, 2.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {0.0f, -1.0f, 1.0f});
  test.AddOutput<float>("min", {3, 3},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -1.0f, -1.0f, -2.0f,
                         0.5f, 0.0f, 1.0f});
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Float_Nan_with_scalar) {
  OpTester test("Min", 12);
  test.AddInput<float>("data_1", {3, 1},
                       {std::numeric_limits<float>::quiet_NaN(), -0.5f, 0.5f});
  test.AddInput<float>("data_2", {1}, {0.25f});
  test.AddOutput<float>("min", {3, 1},
                        {std::numeric_limits<float>::quiet_NaN(), -0.5f, 0.25f});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Float_with_scalar_Nan) {
  OpTester test("Min", 12);
  test.AddInput<float>("data_1", {2, 2},
                       {0.25f, -0.25f, -0.5f, 0.5f});
  test.AddInput<float>("data_2", {1}, {std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<float>("min", {2, 2},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Double) {
  OpTester test("Min", 12);
  test.AddInput<double>("data_0", {1, 3},
                        {1.0f, 2.0f, 3.0f});
  test.AddInput<double>("data_2", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         -70.0f, -80.0f, -90.0f});
  test.AddInput<double>("data_1", {3, 1},
                        {-1.0f, 20.0f, 300.0f});
  test.AddOutput<double>("min", {3, 3},
                         {-1.0f, -1.0f, -1.0f,
                          1.0f, 2.0f, 3.0f,
                          -70.0f, -80.0f, -90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_Double_Nan) {
  OpTester test("Min", 12);
  test.AddInput<double>("data_2", {3, 3},
                        {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         -0.5, 0.0, -2.0,
                         0.5, 0.0, 2.0});
  test.AddInput<double>("data_1", {3, 1},
                        {0.0, -1.0, 1.0});
  test.AddOutput<double>("min", {3, 3},
                         {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          -1.0, -1.0, -2.0,
                          0.5, 0.0, 1.0});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Double_Nan_with_scalar) {
  OpTester test("Min", 12);
  test.AddInput<double>("data_1", {3, 1},
                        {std::numeric_limits<double>::quiet_NaN(), -0.5, 0.5});
  test.AddInput<double>("data_2", {1}, {0.25});
  test.AddOutput<double>("min", {3, 1},
                         {std::numeric_limits<double>::quiet_NaN(), -0.5, 0.25});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Double_with_scalar_Nan) {
  OpTester test("Min", 12);
  test.AddInput<double>("data_1", {2, 2},
                        {0.25, -0.25, -0.5, 0.5});
  test.AddInput<double>("data_2", {1}, {std::numeric_limits<double>::quiet_NaN()});
  test.AddOutput<double>("min", {2, 2},
                         {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_12_Int32) {
  OpTester test("Min", 12);
  test.AddInput<int32_t>("data_0", {1, 3},
                         {1, 2, 3});
  test.AddInput<int32_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          -70, -80, -90});
  test.AddInput<int32_t>("data_1", {3, 1},
                         {-1, 20, 300});
  test.AddOutput<int32_t>("min", {3, 3},
                          {-1, -1, -1,
                           1, 2, 3,
                           -70, -80, -90});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_Int64) {
  OpTester test("Min", 12);
  test.AddInput<int64_t>("data_0", {1, 3},
                         {1, 2, 3});
  test.AddInput<int64_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          -70, -80, -90});
  test.AddInput<int64_t>("data_1", {3, 1},
                         {-1, 20, 300});
  test.AddOutput<int64_t>("min", {3, 3},
                          {-1, -1, -1,
                           1, 2, 3,
                           -70, -80, -90});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_UInt32) {
  OpTester test("Min", 12);
  test.AddInput<uint32_t>("data_0", {1, 3},
                          {1, 20, 30});
  test.AddInput<uint32_t>("data_2", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           70, 80, 90});
  test.AddInput<uint32_t>("data_1", {3, 1},
                          {1, 20, 30});
  test.AddOutput<uint32_t>("min", {3, 3},
                           {1, 1, 1,
                            1, 20, 20,
                            1, 20, 30});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_UInt64) {
  OpTester test("Min", 12);
  test.AddInput<uint64_t>("data_0", {1, 3},
                          {1, 20, 30});
  test.AddInput<uint64_t>("data_2", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           70, 80, 90});
  test.AddInput<uint64_t>("data_1", {3, 1},
                          {1, 20, 30});
  test.AddOutput<uint64_t>("min", {3, 3},
                           {1, 1, 1,
                            1, 20, 20,
                            1, 20, 30});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_MLFloat16) {
  OpTester test("Min", 12);
  test.AddInput<MLFloat16>("data_0", {1, 3},
                           MakeMLFloat16({1.f, 1.f, 1.f}));
  test.AddInput<MLFloat16>("data_1", {1, 3},
                           MakeMLFloat16({2.f, -1.f, -2.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({3.f, 2.f, -3.f}));
  test.AddOutput<MLFloat16>("min", {1, 3},
                            MakeMLFloat16({1.f, -1.f, -3.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_MLFloat16_Scalar0) {
  OpTester test("Min", 12);
  test.AddInput<MLFloat16>("data_0", {},
                           MakeMLFloat16({-10.f}));
  test.AddInput<MLFloat16>("data_1", {1, 3},
                           MakeMLFloat16({2.f, -1.f, -2.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({3.f, 2.f, -3.f}));
  test.AddOutput<MLFloat16>("min", {1, 3},
                            MakeMLFloat16({-10.f, -10.f, -10.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Min_12_MLFloat16_Scalar1) {
  OpTester test("Min", 12);
  test.AddInput<MLFloat16>("data_0", {1, 3},
                           MakeMLFloat16({2.f, 3.f, 4.f}));
  test.AddInput<MLFloat16>("data_1", {},
                           MakeMLFloat16({-10.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({3.f, 2.f, -3.f}));
  test.AddOutput<MLFloat16>("min", {1, 3},
                            MakeMLFloat16({-10.f, -10.f, -10.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

void TestFloat16MinMax(
    const char* op_name,
    const std::vector<int64_t>& lhs_dim,
    const std::initializer_list<float>& lhs_values,
    const std::vector<int64_t>& rhs_dim,
    const std::initializer_list<float>& rhs_values,
    const std::vector<int64_t>& out_dim,
    const std::initializer_list<float>& out_values) {
  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (nullptr != DefaultCpuExecutionProvider()) {
      execution_providers.push_back(DefaultCpuExecutionProvider());
    }
    if (nullptr != DefaultCudaExecutionProvider()) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }
    OpTester test(op_name, 13);
    test.AddInput<MLFloat16>("data_0", lhs_dim, MakeMLFloat16(lhs_values));
    test.AddInput<MLFloat16>("data_1", rhs_dim, MakeMLFloat16(rhs_values));
    test.AddOutput<MLFloat16>("output", out_dim, MakeMLFloat16(out_values));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());

    OpTester test(op_name, 13);
    test.AddInput<BFloat16>("data_0", lhs_dim, MakeBFloat16(lhs_values));
    test.AddInput<BFloat16>("data_1", rhs_dim, MakeBFloat16(rhs_values));
    test.AddOutput<BFloat16>("output", out_dim, MakeBFloat16(out_values));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Min_13_Float16_MatrixVector) {
  TestFloat16MinMax("Min",
                    {3, 3},
                    {1.0f, 1.0f, 1.0f,
                     -0.5f, 0.0f, -2.0f,
                     0.5f, 0.0f, 2.0f},
                    {3, 1}, {0.0f, -1.0f, 1.0f},
                    {3, 3},
                    {0.0f, 0.0f, 0.0f,
                     -1.0f, -1.0f, -2.0f,
                     0.5f, 0.0f, 1.0f});
}

TEST(MathOpTest, Min_13_Float16_VectorMatrix) {
  TestFloat16MinMax("Min",
                    {3, 1}, {0.0f, -1.0f, 1.0f},
                    {3, 4},
                    {1.0f, 1.0f, 1.0f, -1.0f,
                     -0.5f, 0.0f, -2.0f, -1.25f,
                     0.5f, 0.0f, 2.0f, 1.5f},
                    {3, 4},
                    {0.0f, 0.0f, 0.0f, -1.0f,
                     -1.0f, -1.0f, -2.0f, -1.25f,
                     0.5f, 0.0f, 1.0f, 1.0f});
}

TEST(MathOpTest, Min_13_Float16_Nan) {
  TestFloat16MinMax("Min",
                    {4, 1}, {-1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.5f},
                    {4, 1}, {0.5f, 1.0f, 0.25f, std::numeric_limits<float>::quiet_NaN()},
                    {4, 1},
                    {-1.0f, std::numeric_limits<float>::quiet_NaN(), 0.25f, std::numeric_limits<float>::quiet_NaN()});
}

TEST(MathOpTest, Min_13_Float16_Nan_with_scalar) {
  TestFloat16MinMax("Min",
                    {3, 1}, {-1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f},
                    {1}, {0.25f},
                    {3, 1}, {-1.0f, std::numeric_limits<float>::quiet_NaN(), 0.25f});
}

TEST(MathOpTest, Min_13_Float16_with_scalar_Nan) {
  TestFloat16MinMax("Min",
                    {3, 1}, {-0.5f, 1.0f, 1.5f},
                    {1}, {std::numeric_limits<float>::quiet_NaN()},
                    {3, 1},
                    {std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN()});
}

// EP-agnostic int64 coverage for the binary ops this change newly gates on the WebGPU enableInt64
// option (Mul, Div, Greater, Less, GreaterOrEqual, LessOrEqual). Running through the default
// providers exercises every capable EP -- including WebGPU, for which the test harness enables
// int64 (see DefaultWebGpuExecutionProvider and the dynamic plugin EP config in test_main.cc), so
// no WebGPU-specific operator cases are needed. Values stay within the int32 range and include the
// -1 (all-ones) pattern, so EPs that implement int64 on 32-bit lanes (e.g. WebGPU) still produce
// correct results. Min/Max int64 are already covered by the generic Min_12_Int64/Max_12_Int64
// cases above. TensorRT/OpenVINO are excluded to match those existing int64 element-wise cases.
TEST(MathOpTest, Mul_Int64) {
  OpTester test("Mul", 14);
  test.AddInput<int64_t>("A", {3}, {-1, 100, 7});
  test.AddInput<int64_t>("B", {3}, {5, 3, -2});
  test.AddOutput<int64_t>("C", {3}, {-5, 300, -14});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Div_Int64) {
  OpTester test("Div", 14);
  test.AddInput<int64_t>("A", {3}, {-1, 100, -20});
  test.AddInput<int64_t>("B", {3}, {1, 5, 4});
  test.AddOutput<int64_t>("C", {3}, {-1, 20, -5});  // integer division truncates toward zero
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Greater_Int64) {
  OpTester test("Greater", 13);
  test.AddInput<int64_t>("A", {3}, {5, -1, 2147483647});
  test.AddInput<int64_t>("B", {3}, {3, 0, -100});
  test.AddOutput<bool>("C", {3}, {true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Less_Int64) {
  OpTester test("Less", 13);
  test.AddInput<int64_t>("A", {3}, {5, -1, 100});
  test.AddInput<int64_t>("B", {3}, {3, 0, 200});
  test.AddOutput<bool>("C", {3}, {false, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_Int64) {
  OpTester test("GreaterOrEqual", 16);
  test.AddInput<int64_t>("A", {3}, {5, 5, -1});
  test.AddInput<int64_t>("B", {3}, {5, 3, 0});
  test.AddOutput<bool>("C", {3}, {true, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_Int64) {
  OpTester test("LessOrEqual", 16);
  test.AddInput<int64_t>("A", {3}, {5, 5, -1});
  test.AddInput<int64_t>("B", {3}, {5, 3, 0});
  test.AddOutput<bool>("C", {3}, {true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Max_6) {
  OpTester test("Max", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_2", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10000.0f});
  test.AddOutput<float>("max", dims,
                        {1.0f, 0.0f, 3.0f,
                         -1.0f, 3.3f, 64.0f,
                         5.4f, 0.03f, 10000.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TRT10: Disabled due to segfault caused by older opset 6
}

TEST(MathOpTest, Max_8_Float) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, -2.0f, 300.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         300.0f, 300.0f, 300.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_8_Double) {
  OpTester test("Max", 8);
  test.AddInput<double>("data_0", {1, 3},
                        {1.0, 2.0, 3.0});
  test.AddInput<double>("data_2", {3, 3},
                        {10.0, 20.0, 30.0,
                         40.0, 50.0, 60.0,
                         70.0, 80.0, 90.0});
  test.AddInput<double>("data_1", {3, 1},
                        {-1.0, -2.0, 300.0});
  test.AddOutput<double>("max", {3, 3},
                         {10.0, 20.0, 30.0,
                          40.0, 50.0, 60.0,
                          300.0, 300.0, 300.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_8_2inputbroadcast) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         70.0f, 80.0f, 90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_Float) {
  OpTester test("Max", 12);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, -2.0f, 300.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         300.0f, 300.0f, 300.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_Float_Nan) {
  OpTester test("Max", 12);
  test.AddInput<float>("data_2", {3, 3},
                       {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        -0.5f, 0.0f, -2.0f,
                        0.5f, 0.0f, 2.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {0.0f, -1.0f, 1.0f});
  test.AddOutput<float>("max", {3, 3},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -0.5f, 0.0f, -1.0f,
                         1.0f, 1.0f, 2.0f});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Float_Nan_with_scalar) {
  OpTester test("Max", 12);
  test.AddInput<float>("data_1", {3, 1},
                       {std::numeric_limits<float>::quiet_NaN(), -0.5f, 0.5f});
  test.AddInput<float>("data_2", {1}, {0.25f});
  test.AddOutput<float>("max", {3, 1},
                        {std::numeric_limits<float>::quiet_NaN(), 0.25f, 0.5f});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Float_with_scalar_Nan) {
  OpTester test("Max", 12);
  test.AddInput<float>("data_1", {2, 2},
                       {0.25f, -0.25f, -0.5f, 0.5f});
  test.AddInput<float>("data_2", {1}, {std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<float>("max", {2, 2},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Double) {
  OpTester test("Max", 12);
  test.AddInput<double>("data_0", {1, 3},
                        {1.0, 2.0, 3.0});
  test.AddInput<double>("data_2", {3, 3},
                        {10.0, 20.0, 30.0,
                         40.0, 50.0, 60.0,
                         70.0, 80.0, 90.0});
  test.AddInput<double>("data_1", {3, 1},
                        {-1.0, -2.0, 300.0});
  test.AddOutput<double>("max", {3, 3},
                         {10.0, 20.0, 30.0,
                          40.0, 50.0, 60.0,
                          300.0, 300.0, 300.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_Double_Nan) {
  OpTester test("Max", 12);
  test.AddInput<double>("data_2", {3, 3},
                        {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         -0.5, 0.0, -2.0,
                         0.5, 0.0, 2.0});
  test.AddInput<double>("data_1", {3, 1},
                        {0.0, -1.0, 1.0});
  test.AddOutput<double>("max", {3, 3},
                         {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          -0.5, 0.0, -1.0,
                          1.0, 1.0, 2.0});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Double_Nan_with_scalar) {
  OpTester test("Max", 12);
  test.AddInput<double>("data_1", {3, 1},
                        {std::numeric_limits<double>::quiet_NaN(), -0.5, 0.5});
  test.AddInput<double>("data_2", {1}, {0.25});
  test.AddOutput<double>("max", {3, 1},
                         {std::numeric_limits<double>::quiet_NaN(), 0.25, 0.5});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Double_with_scalar_Nan) {
  OpTester test("Max", 12);
  test.AddInput<double>("data_1", {2, 2},
                        {0.25, -0.25, -0.5, 0.5});
  test.AddInput<double>("data_2", {1}, {std::numeric_limits<double>::quiet_NaN()});
  test.AddOutput<double>("max", {2, 2},
                         {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()});
  if (nullptr != DefaultCpuExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
  if (nullptr != DefaultCudaExecutionProvider()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MathOpTest, Max_12_Int32) {
  OpTester test("Max", 12);
  test.AddInput<int32_t>("data_0", {1, 3},
                         {1, 2, 3});
  test.AddInput<int32_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<int32_t>("data_1", {3, 1},
                         {-1, -2, 300});
  test.AddOutput<int32_t>("max", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           300, 300, 300});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_Int64) {
  OpTester test("Max", 12);
  test.AddInput<int64_t>("data_0", {1, 3},
                         {1, 2, 3});
  test.AddInput<int64_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<int64_t>("data_1", {3, 1},
                         {-1, -2, 300});
  test.AddOutput<int64_t>("max", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           300, 300, 300});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_UInt32) {
  OpTester test("Max", 12);
  test.AddInput<uint32_t>("data_0", {1, 3},
                          {1, 2, 3});
  test.AddInput<uint32_t>("data_2", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           70, 80, 90});
  test.AddInput<uint32_t>("data_1", {3, 1},
                          {1, 2, 300});
  test.AddOutput<uint32_t>("max", {3, 3},
                           {10, 20, 30,
                            40, 50, 60,
                            300, 300, 300});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_UInt64) {
  OpTester test("Max", 12);
  test.AddInput<uint64_t>("data_0", {1, 3},
                          {1, 2, 3});
  test.AddInput<uint64_t>("data_2", {3, 3},
                          {10, 20, 30,
                           40, 50, 60,
                           70, 80, 90});
  test.AddInput<uint64_t>("data_1", {3, 1},
                          {1, 2, 300});
  test.AddOutput<uint64_t>("max", {3, 3},
                           {10, 20, 30,
                            40, 50, 60,
                            300, 300, 300});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_MLFloat16) {
  OpTester test("Max", 12);
  test.AddInput<MLFloat16>("data_0", {1, 3},
                           MakeMLFloat16({-1.f, -1.f, -1.f}));
  test.AddInput<MLFloat16>("data_1", {1, 3},
                           MakeMLFloat16({-2.f, -1.f, -2.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({-3.f, -2.f, -3.f}));
  test.AddOutput<MLFloat16>("max", {1, 3},
                            MakeMLFloat16({-1.f, -1.f, -1.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_MLFloat16_Scalar0) {
  OpTester test("Max", 12);
  test.AddInput<MLFloat16>("data_0", {},
                           MakeMLFloat16({-1.f}));
  test.AddInput<MLFloat16>("data_1", {1, 3},
                           MakeMLFloat16({-11.f, -12.f, -22.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({-10.f, -11.f, -13.f}));
  test.AddOutput<MLFloat16>("max", {1, 3},
                            MakeMLFloat16({-1.f, -1.f, -1.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_MLFloat16_Scalar1) {
  OpTester test("Max", 12);
  test.AddInput<MLFloat16>("data_0", {1, 3},
                           MakeMLFloat16({-1.f, -2.f, -3.f}));
  test.AddInput<MLFloat16>("data_1", {},
                           MakeMLFloat16({2.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({-2.f, -3.f, -4.f}));
  test.AddOutput<MLFloat16>("max", {1, 3},
                            MakeMLFloat16({2.f, 2.f, 2.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_12_MLFloat16_Scalar2) {
  OpTester test("Max", 12);
  test.AddInput<MLFloat16>("data_0", {1},
                           MakeMLFloat16({-1.f}));
  test.AddInput<MLFloat16>("data_1", {},
                           MakeMLFloat16({2.f}));
  test.AddInput<MLFloat16>("data_2", {1, 3},
                           MakeMLFloat16({-2.f, -3.f, -4.f}));
  test.AddInput<MLFloat16>("data_3", {1, 1, 3},
                           MakeMLFloat16({-2.f, -3.f, -4.f}));
  test.AddOutput<MLFloat16>("max", {1, 1, 3},
                            MakeMLFloat16({2.f, 2.f, 2.f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_13_Float16_MatrixVector) {
  TestFloat16MinMax("Max",
                    {4, 3},
                    {1.0f, 1.0f, 1.0f,
                     -0.5f, 0.0f, -2.0f,
                     0.0f, 0.5f, 0.75f,
                     0.5f, 0.0f, 2.0f},
                    {4, 1}, {0.0f, -1.0f, 0.5f, 1.0f},
                    {4, 3},
                    {1.0f, 1.0f, 1.0f,
                     -0.5f, 0.0f, -1.0f,
                     0.5f, 0.5f, 0.75f,
                     1.0f, 1.0f, 2.0f});
}

TEST(MathOpTest, Max_13_Float16_VectorMatrix) {
  TestFloat16MinMax("Max",
                    {3, 1}, {0.0f, -1.0f, 1.0f},
                    {3, 3},
                    {1.0f, 1.0f, 1.0f,
                     -0.5f, 0.0f, -2.0f,
                     0.5f, 0.0f, 2.0f},
                    {3, 3},
                    {1.0f, 1.0f, 1.0f,
                     -0.5f, 0.0f, -1.0f,
                     1.0f, 1.0f, 2.0f});
}

TEST(MathOpTest, Max_13_Float16_Nan) {
  TestFloat16MinMax("Max",
                    {4, 1}, {-1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.5f},
                    {4, 1}, {0.5f, 1.0f, 0.25f, std::numeric_limits<float>::quiet_NaN()},
                    {4, 1},
                    {0.5f, std::numeric_limits<float>::quiet_NaN(), 1.0f, std::numeric_limits<float>::quiet_NaN()});
}

TEST(MathOpTest, Max_13_Float16_Nan_with_scalar) {
  TestFloat16MinMax("Max",
                    {3, 1}, {-1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f},
                    {1}, {0.25f},
                    {3, 1}, {0.25f, std::numeric_limits<float>::quiet_NaN(), 1.0f});
}

TEST(MathOpTest, Max_13_Float16_with_scalar_Nan) {
  TestFloat16MinMax("Max",
                    {3, 1}, {-0.5f, 1.0f, 1.5f},
                    {1}, {std::numeric_limits<float>::quiet_NaN()},
                    {3, 1},
                    {std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN()});
}

TEST(MathOpTest, Not) {
  OpTester test("Not");
  std::vector<int64_t> dims{2};
  test.AddInput<bool>("X", dims, {false, true});
  test.AddOutput<bool>("Y", dims, {true, false});
  test.Run();
}

TEST(MathOpTest, And) {
  OpTester test("And");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, false, false, true});
  test.Run();
}

TEST(MathOpTest, Or) {
  OpTester test("Or");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Xor) {
  OpTester test("Xor");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Xor_Issue8880) {
  OpTester test("Xor");
  test.AddInput<bool>("A", {1}, {true});
  test.AddInput<bool>("B", {1}, {false});
  test.AddOutput<bool>("C", {1}, {true});
  test.Run();
}

TEST(MathOpTest, Xor_bcast3v2d) {
  OpTester test("Xor");

  test.AddInput<bool>("A", {2, 3, 4},
                      {false, true, false, true,
                       false, true, false, true,
                       false, true, false, true,

                       false, true, false, true,
                       false, true, false, true,
                       false, true, false, true});
  test.AddInput<bool>("B", {3, 4},
                      {false, false, true, true,
                       false, false, true, true,
                       false, false, true, true});
  test.AddOutput<bool>("C", {2, 3, 4},
                       {false, true, true, false,
                        false, true, true, false,
                        false, true, true, false,

                        false, true, true, false,
                        false, true, true, false,
                        false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less) {
  OpTester test("Less");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

#if defined(USE_DNNL)
TEST(MathOpTest, Less_bfloat16) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("Less", 13);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("A", dims, MakeBFloat16({1.0f, 0.0f, -1.0f, -1.0f}));
  test.AddInput<BFloat16>("B", dims, MakeBFloat16({1.0f, 1.0f, 2.0f, -1.0f}));
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, Less_Scalar0) {
  OpTester test("Less");
  test.AddInput<float>("A", {1}, {1.0f});
  test.AddInput<float>("B", {4}, {1.0f, 1.5f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less_Scalar1) {
  OpTester test("Less");
  test.AddInput<float>("A", {4}, {1.0f, 0.5f, 2.0f, -1.0f});
  test.AddInput<float>("B", {1}, {1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Less_int8_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<int8_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int8_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}
TEST(MathOpTest, Less_int16_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<int16_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int16_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Less_int64_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<int64_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int64_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Less_uint8_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<uint8_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint8_t>("B", {1}, {2});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}
TEST(MathOpTest, Less_uint16_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<uint16_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint16_t>("B", {1}, {2});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Less_uint32_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<uint32_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint32_t>("B", {1}, {2});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Less_uint64_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<uint64_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint64_t>("B", {1}, {2});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Less_broadcastAB) {
  OpTester test("Less", 9);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, false, false});
  test.Run();
}

TEST(MathOpTest, Less_broadcastBA) {
  OpTester test("Less", 9);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Less_multidirectional_broadcastAB) {
  OpTester test("Less", 9);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, true, false});
  test.Run();
}

TEST(MathOpTest, Less_multidirectional_broadcastBA) {
  OpTester test("Less", 9);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, false, true});
  test.Run();
}

TEST(MathOpTest, LessOrEqual) {
  OpTester test("LessOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {true, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_Scalar0) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<float>("A", {1}, {1.0f});
  test.AddInput<float>("B", {4}, {1.0f, 1.5f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", {4}, {true, true, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<float>("A", {4}, {1.0f, 0.5f, 2.0f, -1.0f});
  test.AddInput<float>("B", {1}, {1.0f});
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_int8_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int8_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int8_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_int16_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int16_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int16_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_int64_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int64_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int64_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_uint8_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<uint8_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint8_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_uint16_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<uint16_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint16_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_uint32_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<uint32_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint32_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_uint64_Scalar1) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<uint64_t>("A", {4}, {1, 0, 2, 3});
  test.AddInput<uint64_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_broadcastAB) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_broadcastBA) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_multidirectional_broadcastAB) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, LessOrEqual_multidirectional_broadcastBA) {
  OpTester test("LessOrEqual", 12);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

#if defined(USE_DNNL)
TEST(MathOpTest, LessOrEqual_bfloat16) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("A", dims, MakeBFloat16({1.0f, 0.0f, -1.0f, -1.0f}));
  test.AddInput<BFloat16>("B", dims, MakeBFloat16({1.0f, 1.0f, 2.0f, -1.0f}));
  test.AddOutput<bool>("C", dims, {true, true, true, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}

TEST(MathOpTest, LessOrEqual_bfloat16_Scalar0) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {1}, MakeBFloat16({1.0f}));
  test.AddInput<BFloat16>("B", {4}, MakeBFloat16({1.0f, 1.5f, 2.0f, -1.0f}));
  test.AddOutput<bool>("C", {4}, {true, true, true, false});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}
TEST(MathOpTest, LessOrEqual_bfloat16_Scalar1) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {4}, MakeBFloat16({1.0f, 0.5f, 2.0f, -1.0f}));
  test.AddInput<BFloat16>("B", {1}, MakeBFloat16({1.0f}));
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}

TEST(MathOpTest, LessOrEqual_bfloat16_broadcastAB) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {4, 2}, MakeBFloat16({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f}));
  test.AddInput<BFloat16>("B", {2}, MakeBFloat16({15.0f, 7.0f}));
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, false, false});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}

TEST(MathOpTest, LessOrEqual_bfloat16_broadcastBA) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {2}, MakeBFloat16({15.0f, 7.0f}));
  test.AddInput<BFloat16>("B", {4, 2}, MakeBFloat16({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f}));
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, true, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}

TEST(MathOpTest, LessOrEqual_multidirectional_bfloat16_broadcastAB) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {4, 1}, MakeBFloat16({10.0f, 11.0f, 12.0f, 13.0f}));
  test.AddInput<BFloat16>("B", {2}, MakeBFloat16({15.0f, 7.0f}));
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, true, false});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}

TEST(MathOpTest, LessOrEqual_multidirectional_bfloat16_broadcastBA) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("LessOrEqual", 16);
  test.AddInput<BFloat16>("A", {2}, MakeBFloat16({15.0f, 7.0f}));
  test.AddInput<BFloat16>("B", {4, 1}, MakeBFloat16({10.0f, 11.0f, 12.0f, 13.0f}));
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, false, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, Greater_7) {
  OpTester test("Greater");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Greater_9_float) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Greater_9_double) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<double>("A", dims, {1.0, 0.0, 3.0, -1.0});
  test.AddInput<double>("B", dims, {1.0, 1.0, 2.0, -1.0});
  test.AddOutput<bool>("C", dims, {false, false, true, false});
  test.Run();
}

TEST(MathOpTest, Greater_9_int8) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int8_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Greater_9_int16) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int16_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Greater_9_int32) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater_9_int64) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater_9_uint8) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint8_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Greater_9_uint16) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint16_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Greater_9_uint32) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<uint32_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint32_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Greater_9_uint64) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}
#if defined(USE_DNNL)
TEST(MathOpTest, Greater_13_bfloat16) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("Greater", 13);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("A", dims, MakeBFloat16({10.0f, 11.0f, 12.0f, 13.0f}));
  test.AddInput<BFloat16>("B", dims, MakeBFloat16({15.0f, 7.0f, 12.0f, 9.0f}));
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, Greater_broadcastAB) {
  OpTester test("Greater", 9);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Greater_broadcastBA) {
  OpTester test("Greater", 9);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, false, false});
  test.Run();
}

TEST(MathOpTest, Greater_multidirectional_broadcastAB) {
  OpTester test("Greater", 9);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater_multidirectional_broadcastBA) {
  OpTester test("Greater", 9);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, true, false});
  test.Run();
}

TEST(MathOpTest, GreaterOrEqual_12_float) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_double) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<double>("A", dims, {1.0, 0.0, 3.0, -1.0});
  test.AddInput<double>("B", dims, {1.0, 1.0, 2.0, -1.0});
  test.AddOutput<bool>("C", dims, {true, false, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_int8) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int8_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_int16) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int16_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_int32) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_int64) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_uint8) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint8_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_uint16) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint16_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_uint32) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<uint32_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint32_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_12_uint64) {
  OpTester test("GreaterOrEqual", 12);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider});
}

#if defined(USE_DNNL)
TEST(MathOpTest, GreaterOrEqual_16_bfloat16) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("GreaterOrEqual", 16);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("A", dims, MakeBFloat16({1.0f, 0.0f, -1.0f, -1.0f}));
  test.AddInput<BFloat16>("B", dims, MakeBFloat16({1.0f, 1.0f, 2.0f, -1.0f}));
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, GreaterOrEqual_broadcastAB) {
  OpTester test("GreaterOrEqual", 12);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_broadcastBA) {
  OpTester test("GreaterOrEqual", 12);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, false, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_multidirectional_broadcastAB) {
  OpTester test("GreaterOrEqual", 12);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<bool>("C", {4, 2}, {false, true, false, true, false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, GreaterOrEqual_multidirectional_broadcastBA) {
  OpTester test("GreaterOrEqual", 12);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<bool>("C", {4, 2}, {true, false, true, false, true, false, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kNnapiExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Equal_bool) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_11_bool) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {true, true, true, true});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar0) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {1}, {false});
  test.AddInput<bool>("B", {4}, {false, false, true, true});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar1) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {4}, {false, false, true, true});
  test.AddInput<bool>("B", {1}, {false});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_int8) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int8_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_int16) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int16_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_int32) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_int64) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int64_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_uint8) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {1, 0, 1, 1});
  test.AddInput<uint8_t>("B", dims, {1, 1, 2, 1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_uint16) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("A", dims, {1, 0, 1, 1});
  test.AddInput<uint16_t>("B", dims, {1, 1, 2, 1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_uint32) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<uint32_t>("A", dims, {1, 0, 1, 1});
  test.AddInput<uint32_t>("B", dims, {1, 1, 2, 1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_uint64) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {1, 0, 1, 1});
  test.AddInput<uint64_t>("B", dims, {1, 1, 2, 1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});
}

TEST(MathOpTest, Equal_float) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_string) {
  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<std::string>("A", dims, {"1.0f", "0.0f", "-1.0f", "-1.0f"});
  test.AddInput<std::string>("B", dims, {"1.0f", "1.0f", "2.0f", "-1.0f"});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

#ifdef USE_CUDA
// Opset 19 tests for numeric types (CUDA EP)
TEST(MathOpTest, Equal_19_bool) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_int32) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_int64) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int64_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_float) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_double) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<double>("A", dims, {1.0, 0.0, -1.0, -1.0});
  test.AddInput<double>("B", dims, {1.0, 1.0, 2.0, -1.0});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_float16) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  std::vector<int64_t> dims{4};
  test.AddInput<MLFloat16>("A", dims, {MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(-1.0f), MLFloat16(-1.0f)});
  test.AddInput<MLFloat16>("B", dims, {MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(-1.0f)});
  test.AddOutput<bool>("C", dims, {true, false, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Equal_19_broadcastAB) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    return;
  }

  OpTester test("Equal", 19);
  test.AddInput<int32_t>("A", {4, 2}, {1, 0, -1, -1, 1, 1, -1, 0});
  test.AddInput<int32_t>("B", {2}, {1, 1});
  test.AddOutput<bool>("C", {4, 2}, {true, false, false, false, true, true, false, false});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

#if defined(USE_DNNL)
TEST(MathOpTest, Equal_bfloat16) {
#ifdef USE_DNNL
#ifdef DNNL_GPU_RUNTIME
  LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
  return;
#else
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#endif
  OpTester test("Equal", 13);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("A", dims, MakeBFloat16({1.0f, 0.0f, -1.0f, -1.0f}));
  test.AddInput<BFloat16>("B", dims, MakeBFloat16({1.0f, 1.0f, 2.0f, -1.0f}));
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(MathOpTest, Equal_broadcastAB) {
  OpTester test("Equal");
  test.AddInput<int32_t>("A", {4, 2}, {1, 0, -1, -1, 1, 1, -1, 0});
  test.AddInput<int32_t>("B", {2}, {1, 1});
  test.AddOutput<bool>("C", {4, 2}, {true, false, false, false, true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_broadcastBA) {
  OpTester test("Equal");
  test.AddInput<int32_t>("A", {2}, {1, 1});
  test.AddInput<int32_t>("B", {4, 2}, {1, 0, -1, -1, 1, 1, -1, 0});
  test.AddOutput<bool>("C", {4, 2}, {true, false, false, false, true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_multidirectional_broadcastAB) {
  OpTester test("Equal");
  test.AddInput<int32_t>("A", {4, 1}, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", {2}, {1, 1});
  test.AddOutput<bool>("C", {4, 2}, {true, true, false, false, false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_multidirectional_broadcastBA) {
  OpTester test("Equal");
  test.AddInput<int32_t>("A", {2}, {1, 1});
  test.AddInput<int32_t>("B", {4, 1}, {1, 0, -1, -1});
  test.AddOutput<bool>("C", {4, 2}, {true, true, false, false, false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_multidirectional_broadcastAB_bool) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {4, 1}, {true, false, false, false});
  test.AddInput<bool>("B", {2}, {true, true});
  test.AddOutput<bool>("C", {4, 2}, {true, true, false, false, false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Max_12_Int8) {
  OpTester test("Max", 12);
  test.AddInput<int8_t>("data_0", {1, 3},
                        {1, 2, 3});
  test.AddInput<int8_t>("data_2", {3, 3},
                        {10, 20, 30,
                         40, 50, 60,
                         70, 80, 90});
  test.AddInput<int8_t>("data_1", {3, 1},
                        {-1, -2, 127});
  test.AddOutput<int8_t>("max", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          127, 127, 127});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Max_12_UInt8) {
  OpTester test("Max", 12);
  test.AddInput<uint8_t>("data_0", {1, 3},
                         {1, 20, 30});
  test.AddInput<uint8_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<uint8_t>("data_1", {3, 1},
                         {100, 20, 30});
  test.AddOutput<uint8_t>("max", {3, 3},
                          {100, 100, 100,
                           40, 50, 60,
                           70, 80, 90});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Min_12_Int8) {
  OpTester test("Min", 12);
  test.AddInput<int8_t>("data_0", {1, 3},
                        {1, 2, 3});
  test.AddInput<int8_t>("data_2", {3, 3},
                        {10, 20, 30,
                         40, 50, 60,
                         -70, -80, -90});
  test.AddInput<int8_t>("data_1", {3, 1},
                        {-1, 20, 127});
  test.AddOutput<int8_t>("min", {3, 3},
                         {-1, -1, -1,
                          1, 2, 3,
                          -70, -80, -90});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Min_12_UInt8) {
  OpTester test("Min", 12);
  test.AddInput<uint8_t>("data_0", {1, 3},
                         {1, 20, 30});
  test.AddInput<uint8_t>("data_2", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<uint8_t>("data_1", {3, 1},
                         {1, 20, 30});
  test.AddOutput<uint8_t>("min", {3, 3},
                          {1, 1, 1,
                           1, 20, 20,
                           1, 20, 30});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Mean_6) {
  OpTester test("Mean", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.0f, 0.01f, -10.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 65.0f,
                        -1.0f, 0.02f, -1.0f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 65.0f,
                        -3.0f, 0.03f, -1.0f});
  test.AddOutput<float>("mean", dims,
                        {1.0f, 0.0f, 2.0f,
                         -2.0f, 2.2f, 10.0f,
                         -3.0f, 0.02f, -4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TRT10: Disabled due to segfault caused by older opset 6
}

TEST(MathOpTest, Mean_8) {
  OpTester test("Mean", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_3", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("mean", {3, 3},
                        {12.0f / 3.0f, 22.0f / 3.0f, 32.0f / 3.0f,
                         43.0f / 3.0f, 53.0f / 3.0f, 63.0f / 3.0f,
                         74.0f / 3.0f, 84.0f / 3.0f, 94.0f / 3.0f});
  // OpenVINO: Disabled due to accuracy mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: Input batch size is inconsistent
}

#ifdef _LIBCPP_VERSION
#define MATH_NO_EXCEPT
#else
#define MATH_NO_EXCEPT noexcept
#endif

template <float (&op)(float value) MATH_NO_EXCEPT>
void TrigFloatTest(OpTester& test, std::initializer_list<float> input, float abs_error = -1.0f) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<float> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  if (abs_error >= 0.0f) {
    test.SetOutputTolerance(abs_error);
  }

  test.Run();
}

template <double (&op)(double value) MATH_NO_EXCEPT>
void TrigDoubleTest(OpTester& test, std::initializer_list<double> input,
                    const std::unordered_set<std::string> excluded_provider_types = {}) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<double> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<double>("X", dims, input);
  test.AddOutput<double>("Y", dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_provider_types);
}

template <float (&op)(float value) MATH_NO_EXCEPT>
void TrigFloat16Test(OpTester& test, std::initializer_list<float> input) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<MLFloat16> float16_input;
  std::vector<MLFloat16> float16_output;
  for (auto v : input) {
    float16_input.push_back(MLFloat16(v));
    float16_output.push_back(MLFloat16(op(v)));
  }

  test.AddInput<MLFloat16>("X", dims, float16_input);
  test.AddOutput<MLFloat16>("Y", dims, float16_output);
  test.Run();
}
TEST(MathOpTest, SinFloat) {
  OpTester test("Sin");
  TrigFloatTest<::sinf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, SinDouble) {
  OpTester test("Sin");
  TrigDoubleTest<::sin>(test, {1.1, -1.1, 2.2, -2.2});
}

TEST(MathOpTest, SinFloat16) {
  if (DefaultCudaExecutionProvider().get() != nullptr) {  // MLFloat16 type not supported on CPU
    OpTester test("Sin");
    TrigFloat16Test<::sinf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
  }
}

TEST(MathOpTest, CosFloat) {
  OpTester test("Cos");
  TrigFloatTest<::cosf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, CosDouble) {
  OpTester test("Cos");
  TrigDoubleTest<::cos>(test, {1.1, -1.1, 2.2, -2.2});
}

TEST(MathOpTest, CosFloat16) {
  if (DefaultCudaExecutionProvider().get() != nullptr) {  // MLFloat16 type not supported on CPU
    OpTester test("Cos");
    TrigFloat16Test<::cosf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
  }
}

TEST(MathOpTest, Sin_Opset22) {
  OpTester test("Sin", 22);
  TrigFloatTest<::sinf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Cos_Opset22) {
  OpTester test("Cos", 22);
  TrigFloatTest<::cosf>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

#ifdef USE_CUDA
TEST(MathOpTest, Sin_BFloat16_Opset22_CUDA) {
  if (!HasCudaEnvironment(530)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test("Sin", 22);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("input", dims, MakeBFloat16({1.1f, -1.1f, 2.2f, -2.2f}));
  test.AddOutput<BFloat16>("output", dims, MakeBFloat16({std::sin(1.1f), std::sin(-1.1f), std::sin(2.2f), std::sin(-2.2f)}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Cos_BFloat16_Opset22_CUDA) {
  if (!HasCudaEnvironment(530)) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test("Cos", 22);
  std::vector<int64_t> dims{4};
  test.AddInput<BFloat16>("input", dims, MakeBFloat16({1.1f, -1.1f, 2.2f, -2.2f}));
  test.AddOutput<BFloat16>("output", dims, MakeBFloat16({std::cos(1.1f), std::cos(-1.1f), std::cos(2.2f), std::cos(-2.2f)}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(MathOpTest, Tan) {
  OpTester test("Tan");
  TrigFloatTest<::tanf>(test, {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f});
}

TEST(MathOpTest, Asin) {
  OpTester test("Asin");
  float abs_error =
#ifdef _WIN32
      // Set abs_error to 0.0001f for built-in function asin() in HLSL based EPs (DML and WebGPU)
      DefaultDmlExecutionProvider().get() != nullptr || DefaultWebGpuExecutionProvider().get() != nullptr
          ? 0.0001f
          :
#endif
          -1.0f;
  TrigFloatTest<::asinf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}, abs_error);
}

TEST(MathOpTest, Acos) {
  OpTester test("Acos");
  TrigFloatTest<::acosf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Atan) {
  OpTester test("Atan");
  TrigFloatTest<::atanf>(test, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});
}

TEST(MathOpTest, Sinh) {
  OpTester test("Sinh", 9);
  TrigFloatTest<::sinhf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Cosh) {
  OpTester test("Cosh", 9);
  TrigFloatTest<::coshf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Asinh) {
  OpTester test("Asinh", 9);
  TrigFloatTest<::asinhf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acosh) {
  OpTester test("Acosh", 9);
  TrigFloatTest<::acoshf>(test, {1.0f, 1.1f, 3.0f, 10.0f, 100.0f});
}

TEST(MathOpTest, Atanh) {
  OpTester test("Atanh", 9);
  TrigFloatTest<::atanhf>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Expand_8_3x3_string) {
  OpTester test("Expand", 8);
  test.AddInput<std::string>("data_0", {1}, {"1"});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<std::string>("result", {3, 3},
                              {"1", "1", "1",
                               "1", "1", "1",
                               "1", "1", "1"});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_string) {
  OpTester test("Expand", 8);
  test.AddInput<std::string>("data_0", {3}, {"1", "2", "3"});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<std::string>("result", {3, 3},
                              {"1", "2", "3",
                               "1", "2", "3",
                               "1", "2", "3"});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_string) {
  OpTester test("Expand", 8);
  test.AddInput<std::string>("data_0", {3, 1}, {"1", "2", "3"});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<std::string>("result", {3, 3},
                              {"1", "1", "1",
                               "2", "2", "2",
                               "3", "3", "3"});
  test.Run();
}

TEST(MathOpTest, Erf) {
  OpTester test("Erf", 9);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.5f, 1.0f, 0.7f, 2.0f});
  test.AddOutput<float>("B", dims, {0.5204999f, 0.8427008f, 0.6778012f, 0.9953223f});
  test.Run();
}

TEST(MathOpTest, Erf_f16) {
  OpTester test("Erf", 9);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<MLFloat16>("A", dims, MakeMLFloat16({0.5f, 1.0f, 0.7f, 2.0f}));
  test.AddOutput<MLFloat16>("B", dims, MakeMLFloat16({0.5204999f, 0.8427008f, 0.6778012f, 0.9953223f}));
  test.Run();
}

TEST(MathOpTest, ErfMoreData) {
  OpTester test("Erf", 9);
  std::vector<float> inputs{
      -3.625f, 3.375f, 0.0f, 0.00025f, 0.0005f, -0.00075f, -0.001f, 0.00125f,
      0.0015f, -3.125f, 0.00175f, 2.875f, 2.625f, 2.375f, 2.125f, 6.25e-05f,
      0.0003125f, 0.0005625f, -0.0008125f, 0.0010625f, 0.0013125f, 0.0015625f, 0.0018125f, 3.5625f,
      3.3125f, 3.0625f, 2.8125f, -2.5625f, 2.3125f, 2.0625f, 0.000125f, 0.000375f,
      -0.000625f, -0.000875f, -0.001125f, -0.001375f, -0.001625f, -0.001875f, -3.5f, -3.25f,
      3.0f, 2.75f, -2.5f, -2.25f, -2.0f, -0.0001875f, 0.0004375f, 0.0006875f,
      2.1875f, -1.9375f, 0.0014375f, -0.0016875f, -0.0019375f, 3.4375f, 3.1875f, -2.9375f,
      -2.4375f, -0.0009375f, 0.0011875f};
  std::vector<float> outputs{
      -1.0f, 0.999998f, 0.0f, 0.000282095f, 0.00056419f, -0.000846284f, -0.00112838f, 0.00141047f,
      0.00169257f, -0.99999f, 0.00197466f, 0.999952f, 0.999795f, 0.999217f, 0.997346f, 7.05237e-05f,
      0.000352618f, 0.000634713f, -0.000916808f, 0.0011989f, 0.001481f, 0.00176309f, 0.00204518f, 1.0f,
      0.999997f, 0.999985f, 0.99993f, -0.99971f, 0.998926f, 0.996464f, 0.000141047f, 0.000423142f,
      -0.000705237f, -0.000987331f, -0.00126943f, -0.00155152f, -0.00183361f, -0.00211571f, -0.999999f, -0.999996f,
      0.999978f, 0.999899f, -0.999593f, -0.998537f, -0.995322f, -0.000211571f, 0.000493666f, 0.000775761f,
      0.998022f, -0.993857f, 0.00162204f, -0.00190414f, -0.00218623f, 0.999999f, 0.999993f, -0.999967f,
      -0.999433f, -0.00105786f, 0.00133995f};
  std::vector<int64_t> dims{static_cast<int64_t>(inputs.size())};

  test.AddInput<float>("A", dims, inputs);
  test.AddOutput<float>("B", dims, outputs);
  test.Run();
}

TEST(MathOpTest, ErfNaN) {
  // Regression for #28462: NaN inputs were clamped to 1.0 on the FMA3 path.
  OpTester test("Erf", 9);
  constexpr float qnan = std::numeric_limits<float>::quiet_NaN();
  constexpr int64_t size = 36;
  std::vector<int64_t> dims{size};
  std::vector<float> inputs(size);
  std::vector<float> outputs(size);
  for (int64_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      inputs[i] = qnan;
      outputs[i] = qnan;
    } else {
      const float v = 0.1f * static_cast<float>(i - size / 2);
      inputs[i] = v;
      outputs[i] = std::erf(v);
    }
  }
  test.AddInput<float>("A", dims, inputs);
  test.AddOutput<float>("B", dims, outputs);
  test.Run();
}

TEST(MathOpTest, ErfCheckMultiThreadDataChunking) {
  OpTester test("Erf", 9);
  static constexpr int64_t size = 100;
  std::vector<int64_t> dims{size};
  RandomValueGenerator random(42);
  std::vector<float> inputs = random.Uniform<float>(dims, -5.0f, 5.0f);
  std::vector<float> outputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    outputs[i] = std::erf(inputs[i]);
  }
  test.AddInput<float>("A", dims, inputs);
  test.AddOutput<float>("B", dims, outputs);
  test.Run();
}

constexpr int ModOp_ver = 10;
constexpr int ModOp_ver28 = 28;

TEST(ModOpTest, Fmod_float_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<float>("X", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
  test.AddInput<float>("Y", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
  test.AddOutput<float>("Z", {6}, {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f});

  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, Fmod_double_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<double>("X", {6}, {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
  test.AddInput<double>("Y", {6}, {2.1f, -3.4, 8.0, -2.1, 3.4, 5.0});
  test.AddOutput<double>("Z", {6}, {-0.1, 0.4, 5., 0.1, -0.4, 3.});

  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, Fmod_float16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<MLFloat16>("X", {6}, MakeMLFloat16({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f}));
  test.AddInput<MLFloat16>("Y", {6}, MakeMLFloat16({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f}));
  // The output above is {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f} for float
  test.AddOutput<MLFloat16>("Z", {6}, MakeMLFloat16({-0.1015625f, 0.3984375f, 5.f, 0.1015625f, -0.3984375f, 3.f}));

  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

template <typename T>
void TestFloorModFloatingPointOpset28(const std::vector<T>& x,
                                      const std::vector<T>& y,
                                      const std::vector<T>& expected) {
  OpTester test("Mod", 28);
  test.AddAttribute<int64_t>("fmod", 0);
  const std::vector<int64_t> shape{static_cast<int64_t>(x.size())};
  test.AddInput<T>("X", shape, x);
  test.AddInput<T>("Y", shape, y);
  test.AddOutput<T>("Z", shape, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kQnnExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ModOpTest, FloorMod_float_mixed_sign_opset28) {
  TestFloorModFloatingPointOpset28<float>(
      {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f},
      {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f},
      {2.0f, -3.0f, 5.0f, -2.0f, 3.0f, 3.0f});
}

TEST(ModOpTest, FloorMod_double_mixed_sign_opset28) {
  TestFloorModFloatingPointOpset28<double>(
      {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0},
      {2.1, -3.4, 8.0, -2.1, 3.4, 5.0},
      {2.0, -3.0, 5.0, -2.0, 3.0, 3.0});
}

TEST(ModOpTest, FloorMod_float16_mixed_sign_opset28) {
  TestFloorModFloatingPointOpset28<MLFloat16>(
      MakeMLFloat16({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f}),
      MakeMLFloat16({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f}),
      MakeMLFloat16({2.0f, -3.0f, 5.0f, -2.0f, 3.0f, 3.0f}));
}

TEST(ModOpTest, FloorMod_float_special_values_opset28) {
  constexpr auto infinity = std::numeric_limits<float>::infinity();
  constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
  TestFloorModFloatingPointOpset28<float>(
      {6.0f, -6.0f, 5.0f, -5.0f, infinity, 1.0f, nan, 1.0f},
      {-3.0f, 3.0f, infinity, -infinity, 2.0f, 0.0f, 2.0f, nan},
      {-0.0f, 0.0f, 5.0f, -5.0f, nan, nan, nan, nan});
}

TEST(ModOpTest, FloorMod_opset28_provider_fallback) {
  InlinedVector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
#ifdef USE_CANN
  OrtCANNProviderOptions cann_options{};
  cann_options.enable_cann_graph = 1;
  auto cann_factory = CannProviderFactoryCreator::Create(&cann_options);
  ASSERT_NE(cann_factory, nullptr);
  providers.push_back(cann_factory->CreateProvider());
#endif
  providers.push_back(DefaultMIGraphXExecutionProvider());
  providers.push_back(DefaultOpenVINOExecutionProvider());
  for (auto& provider : providers) {
    if (!provider) {
      continue;
    }
    SCOPED_TRACE(provider->Type());
    const std::vector<int64_t> shape{4};
    const float infinity = std::numeric_limits<float>::infinity();
    const std::vector<float> x{-5.0f, 5.0f, -5.0f, 5.0f};
    const std::vector<float> y{3.0f, -3.0f, infinity, -infinity};
    OpTester test("Mod", 28);
    test.AddAttribute<int64_t>("fmod", 0);
    test.AddInput<float>("X", shape, x);
    test.AddInput<float>("Y", shape, y);
    test.AddOutput<float>("Z", shape, {1.0f, -1.0f, infinity, -infinity});
    test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
    std::string model_data;
    auto& model = test.BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    model.ToProto().SerializeToString(&model_data);
    EPVerificationParams params;
    params.ep_node_assignment = provider->Type() == kCpuExecutionProvider
                                    ? ExpectedEPNodeAssignment::All
                                    : ExpectedEPNodeAssignment::None;
    RunAndVerifyOutputsWithEP(
        AsByteSpan(model_data.data(), model_data.size()), CurrentTestName(), std::move(provider),
        {{"X", CreateInputOrtValueOnCPU<float>(shape, x)}, {"Y", CreateInputOrtValueOnCPU<float>(shape, y)}},
        params);
  }
}

#if defined(USE_CUDA)
TEST(ModOpTest, Fmod_bfloat16_mixed_sign) {
  OpTester test("Mod", 13);
  test.AddAttribute<int64_t>("fmod", 1);
  // Due to BFloat16's precision, if the result is too small, it's not easy get pass for both CUDA.
  test.AddInput<BFloat16>("X", {4}, MakeBFloat16({8.0f, 5.0f, -8.0f, 8.0f}));
  test.AddInput<BFloat16>("Y", {4}, MakeBFloat16({-3.4f, 8.0f, 3.4f, 5.0f}));
  test.AddOutput<BFloat16>("Z", {4}, MakeBFloat16({1.2f, 5.f, -1.2f, 3.f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(ModOpTest, FloorMod_float_mixed_sign) {
  OpTester test("Mod", ModOp_ver28);
  test.AddInput<float>("X", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
  test.AddInput<float>("Y", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
  test.AddOutput<float>("Z", {6}, {2.0f, -3.0f, 5.0f, -2.0f, 3.0f, 3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, FloorMod_double_mixed_sign) {
  OpTester test("Mod", ModOp_ver28);
  test.AddInput<double>("X", {6}, {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
  test.AddInput<double>("Y", {6}, {2.1, -3.4, 8.0, -2.1, 3.4, 5.0});
  test.AddOutput<double>("Z", {6}, {2.0, -3.0, 5.0, -2.0, 3.0, 3.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, FloorMod_float16_mixed_sign) {
  OpTester test("Mod", ModOp_ver28);
  test.AddInput<MLFloat16>("X", {6}, MakeMLFloat16({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f}));
  test.AddInput<MLFloat16>("Y", {6}, MakeMLFloat16({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f}));
  test.AddOutput<MLFloat16>("Z", {6}, MakeMLFloat16({2.0f, -3.0f, 5.0f, -2.0f, 3.0f, 3.0f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, FloorMod_float_edge_cases) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  OpTester test("Mod", ModOp_ver28);
  test.AddInput<float>("X", {14},
                       {0.0f, -0.0f, 0.0f, -0.0f, -3.0f, 3.0f, -1.0f, 1.0f,
                        inf, -inf, 1.0f, 1.0f, nan, 1.0f});
  test.AddInput<float>("Y", {14},
                       {-2.0f, 2.0f, 2.0f, -2.0f, inf, inf, -inf, -inf,
                        2.0f, 2.0f, 0.0f, -0.0f, 2.0f, nan});
  test.AddOutput<float>("Z", {14},
                        {-0.0f, 0.0f, 0.0f, -0.0f, inf, 3.0f, -1.0f, -inf,
                         nan, nan, nan, nan, nan, nan});
  test.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches,
                                  const std::string& /*provider_type*/) {
    ASSERT_EQ(fetches.size(), 1u);
    ASSERT_TRUE(fetches[0].IsTensor());
    const float* output = fetches[0].Get<Tensor>().Data<float>();

    EXPECT_EQ(output[0], 0.0f);
    EXPECT_TRUE(std::signbit(output[0]));
    EXPECT_EQ(output[1], 0.0f);
    EXPECT_FALSE(std::signbit(output[1]));
    EXPECT_EQ(output[2], 0.0f);
    EXPECT_FALSE(std::signbit(output[2]));
    EXPECT_EQ(output[3], 0.0f);
    EXPECT_TRUE(std::signbit(output[3]));
    EXPECT_EQ(output[4], std::numeric_limits<float>::infinity());
    EXPECT_EQ(output[5], 3.0f);
    EXPECT_EQ(output[6], -1.0f);
    EXPECT_EQ(output[7], -std::numeric_limits<float>::infinity());
    for (size_t i = 8; i < 14; ++i) {
      EXPECT_TRUE(std::isnan(output[i])) << "output index " << i;
    }
  });
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

#ifdef USE_DML
template <typename T>
void TestDirectMLFloorModBroadcast() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<T> x{T(0.0f), T(-0.0f), T(6.0f), T(-6.0f), T(3.0f),
                         T(-3.0f), T(inf), T(-inf), T(nan), T(1.0f)};
  const std::vector<T> y{T(2.0f), T(-2.0f), T(inf), T(-inf), T(0.0f), T(-0.0f), T(nan)};
  const std::vector<float> expected{
      0.0f, -0.0f, 0.0f, -0.0f, nan, nan, nan,
      0.0f, -0.0f, 0.0f, -0.0f, nan, nan, nan,
      0.0f, -0.0f, 6.0f, -inf, nan, nan, nan,
      0.0f, -0.0f, inf, -6.0f, nan, nan, nan,
      1.0f, -1.0f, 3.0f, -inf, nan, nan, nan,
      1.0f, -1.0f, inf, -3.0f, nan, nan, nan,
      nan, nan, nan, nan, nan, nan, nan,
      nan, nan, nan, nan, nan, nan, nan,
      nan, nan, nan, nan, nan, nan, nan,
      1.0f, -1.0f, 1.0f, -inf, nan, nan, nan};
  std::vector<T> typed_expected;
  typed_expected.reserve(expected.size());
  for (float value : expected) {
    typed_expected.emplace_back(value);
  }
  for (bool disable_fusion : {false, true}) {
    SCOPED_TRACE(disable_fusion ? "standalone kernel" : "fused graph");
    OpTester test("Mod", 28);
    test.AddAttribute<int64_t>("fmod", 0);
    test.AddInput<T>("X", {10, 1}, x);
    test.AddInput<T>("Y", {1, 7}, y);
    test.AddOutput<T>("Z", {10, 7}, typed_expected);
    test.SetCustomOutputVerifier([&expected](const std::vector<OrtValue>& fetches,
                                             const std::string& provider_type) {
      ASSERT_EQ(provider_type, kDmlExecutionProvider);
      ASSERT_EQ(fetches.size(), 1u);
      const auto& tensor = fetches[0].Get<Tensor>();
      ASSERT_EQ(tensor.Shape().Size(), static_cast<int64_t>(expected.size()));
      const T* output = tensor.Data<T>();
      for (size_t i = 0; i < expected.size(); ++i) {
        const float actual = static_cast<float>(output[i]);
        if (std::isnan(expected[i])) {
          EXPECT_TRUE(std::isnan(actual)) << "output index " << i;
        } else {
          EXPECT_EQ(actual, expected[i]) << "output index " << i;
          EXPECT_EQ(std::signbit(actual), std::signbit(expected[i])) << "output index " << i;
        }
      }
    });
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
        kOrtSessionOptionsConfigDisableDmlGraphFusion, disable_fusion ? "1" : "0"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(DefaultDmlExecutionProvider()).RunWithConfig();
  }
}

TEST(ModOpTest, FloorMod_DirectML_float_broadcast) {
  TestDirectMLFloorModBroadcast<float>();
}

TEST(ModOpTest, FloorMod_DirectML_float16_broadcast) {
  TestDirectMLFloorModBroadcast<MLFloat16>();
}
#endif

TEST(ModOpTest, Signed_integer_overflow_case) {
  OpTester test("Mod", ModOp_ver28);
  test.AddInput<int32_t>("X", {2}, {std::numeric_limits<int32_t>::min(), 7});
  test.AddInput<int32_t>("Y", {2}, {-1, -3});
  test.AddOutput<int32_t>("Z", {2}, {0, -2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, Signed_integer_overflow_case_fmod) {
  OpTester test("Mod", ModOp_ver28);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<int32_t>("X", {2}, {std::numeric_limits<int32_t>::min(), 7});
  test.AddInput<int32_t>("Y", {2}, {-1, -3});
  test.AddOutput<int32_t>("Z", {2}, {0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, Int64_fmod_precision) {
  OpTester test("Mod", ModOp_ver28);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<int64_t>("X", {1}, {9007199254740993LL});
  test.AddInput<int64_t>("Y", {1}, {10});
  test.AddOutput<int64_t>("Z", {1}, {3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, UInt64_fmod_precision) {
  OpTester test("Mod", ModOp_ver28);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<uint64_t>("X", {1}, {18446744073709551613ULL});
  test.AddInput<uint64_t>("Y", {1}, {10});
  test.AddOutput<uint64_t>("Z", {1}, {3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, Int8_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, -2, 5, 0, 2, 3});
  // For TensorRT running in these in INT8 quantization scales are needed, so skip it now
  // For QNN EP, when input is not fp32/fp16/ufxp16/ufxp8/sfxp8, Mod is lowered to Cast(a - b*floor(a/b)). And Int8 is not supported by Cast, so skip it now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, Int8_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  // For TensorRT running in these in INT8 quantization scales are needed, so skip it now
  // For QNN EP, when input is not fp32/fp16/ufxp16/ufxp8/sfxp8, Mod is lowered to Cast(a - b*floor(a/b)). And Int8 is not supported by Cast, so skip it now
  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(ModOpTest, UInt8_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint8_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint8_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint8_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  // For QNN EP, when input is not fp32/fp16/ufxp16/ufxp8/sfxp8, Mod is lowered to Cast(a - b*floor(a/b)). And Int16 is not supported by Cast, so skip it now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, Int16_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  // QNN EP doesn't support Fmod
  // For QNN EP, when input is not fp32, fp16, ufxp16, ufxp8, sfxp8 Mod is lowered to Cast(a - b*floor(a/b)). And Int16 is not supported by Cast, so skip it now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, UInt16_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint16_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint16_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint16_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  // For QNN EP, when input is not fp32, fp16, ufxp16, ufxp8, sfxp8 Mod is lowered to Cast(a - b*floor(a/b)). And Uint16 is not supported by Cast, so skip it now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, Int32_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, UInt32_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint32_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint32_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint32_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  // QNN EP doesn't support Fmod
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, UInt64_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint64_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint64_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint64_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  // For QNN EP, when input is not fp32/fp16/ufxp16/ufxp8/sfxp8, Mod is lowered to Cast(a - b*floor(a/b)). And Uint64 is not supported by Cast, so skip it now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST(ModOpTest, Int32_mod_bcast) {
  OpTester test("Mod", ModOp_ver);

  std::vector<int32_t> input_sequence;
  input_sequence.resize(30);
  std::generate(input_sequence.begin(), input_sequence.end(),
                [n = 0]() mutable { return n++; });

  // input [0..29]
  test.AddInput<int32_t>("X", {3, 2, 5}, input_sequence);
  test.AddInput<int32_t>("Y", {1}, {7});

  test.AddOutput<int32_t>("Z", {3, 2, 5},
                          {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1});

  test.Run();
}

TEST(ModOpTest, Mod_int8_by_zero) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int8_t>("X", {3}, {4, 8, 8});
  test.AddInput<int8_t>("Y", {3}, {1, 0, 2});
  test.AddOutput<int8_t>("Z", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer modulo by zero",
           {}, nullptr, &execution_providers);
}

TEST(ModOpTest, Mod_int32_by_zero) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int32_t>("X", {3}, {4, 8, 8});
  test.AddInput<int32_t>("Y", {3}, {1, 0, 2});
  test.AddOutput<int32_t>("Z", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer modulo by zero",
           {}, nullptr, &execution_providers);
}

TEST(ModOpTest, Mod_int64_by_zero_scalar) {
  // Scalar divisor of 0
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int64_t>("X", {3}, {4, 8, 8});
  test.AddInput<int64_t>("Y", {}, {0});
  test.AddOutput<int64_t>("Z", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer modulo by zero",
           {}, nullptr, &execution_providers);
}

TEST(ModOpTest, Mod_int32_by_zero_constant_initializer) {
  // Divisor is a constant initializer — validated once at kernel creation time
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int32_t>("X", {3}, {4, 8, 8});
  test.AddInput<int32_t>("Y", {3}, {1, 0, 2}, true);  // is_initializer = true
  test.AddOutput<int32_t>("Z", {3}, {0, 0, 0});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Integer modulo by zero",
           {}, nullptr, &execution_providers);
}

TEST(BitShiftOpTest, SimpleLeft) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint32_t>("X", {3}, {16, 4, 1});
  test.AddInput<uint32_t>("Y", {3}, {1, 2, 3});
  test.AddOutput<uint32_t>("Z", {3}, {32, 16, 8});
  test.Run();
}

TEST(BitShiftOpTest, SimpleRight) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint32_t>("X", {3}, {16, 4, 1});
  test.AddInput<uint32_t>("Y", {3}, {1, 2, 3});
  test.AddOutput<uint32_t>("Z", {3}, {8, 1, 0});
  test.Run();
}

TEST(BitShiftOpTest, ScalarLeftX) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint32_t>("X", {1}, {16});
  test.AddInput<uint32_t>("Y", {3}, {1, 2, 3});
  test.AddOutput<uint32_t>("Z", {3}, {32, 64, 128});
  test.Run();
}

TEST(BitShiftOpTest, ScalarLeftY) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint32_t>("X", {3}, {16, 4, 1});
  test.AddInput<uint32_t>("Y", {1}, {1});
  test.AddOutput<uint32_t>("Z", {3}, {32, 8, 2});
  test.Run();
}

TEST(BitShiftOpTest, ScalarRightX) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint32_t>("X", {1}, {16});
  test.AddInput<uint32_t>("Y", {3}, {1, 2, 3});
  test.AddOutput<uint32_t>("Z", {3}, {8, 4, 2});
  test.Run();
}

TEST(BitShiftOpTest, ScalarRightY) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint32_t>("X", {3}, {16, 4, 1});
  test.AddInput<uint32_t>("Y", {1}, {1});
  test.AddOutput<uint32_t>("Z", {3}, {8, 2, 0});
  test.Run();
}

TEST(BitShiftOpTest, BroadcastYLeft) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint64_t>("X", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<uint64_t>("Y", {2}, {1, 2});
  test.AddOutput<uint64_t>("Z", {3, 2}, {2, 8, 6, 16, 10, 24});
  test.Run();
}

TEST(BitShiftOpTest, BroadcastXRight) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint64_t>("X", {2}, {64, 32});
  test.AddInput<uint64_t>("Y", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<uint64_t>("Z", {3, 2}, {32, 8, 8, 2, 2, 0});
  test.Run();
}

TEST(BitShiftOpTest, BroadcastYLeft_Uint8) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint8_t>("X", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<uint8_t>("Y", {2}, {1, 2});
  test.AddOutput<uint8_t>("Z", {3, 2}, {2, 8, 6, 16, 10, 24});
  test.Run();
}

TEST(BitShiftOpTest, BroadcastXRight_Uint8) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint8_t>("X", {2}, {64, 32});
  test.AddInput<uint8_t>("Y", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<uint8_t>("Z", {3, 2}, {32, 8, 8, 2, 2, 0});
  test.Run();
}

template <typename T>
void TestSignedBitShiftOpset28() {
  constexpr T bit_width = static_cast<T>(std::numeric_limits<std::make_unsigned_t<T>>::digits);

  {
    OpTester test("BitShift", 28);
    test.AddAttribute("direction", "LEFT");
    test.AddInput<T>("X", {6}, {T{-8}, T{-3}, T{1}, std::numeric_limits<T>::max(), T{-2}, T{2}});
    test.AddInput<T>("Y", {6}, {T{1}, T{2}, T{3}, T{1}, T{-1}, bit_width});
    test.AddOutput<T>("Z", {6}, {T{-16}, T{-12}, T{8}, T{-2}, T{0}, T{0}});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  }

  {
    OpTester test("BitShift", 28);
    test.AddAttribute("direction", "RIGHT");
    test.AddInput<T>("X", {6}, {T{-8}, T{-3}, T{1}, T{-2}, T{-2}, T{2}});
    test.AddInput<T>("Y", {6}, {T{1}, T{2}, T{3}, T{-1}, bit_width, bit_width});
    test.AddOutput<T>("Z", {6}, {T{-4}, T{-1}, T{0}, T{-1}, T{-1}, T{0}});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  }

  {
    OpTester test("BitShift", 28);
    test.AddAttribute("direction", "RIGHT");
    test.AddInput<T>("X", {1}, {T{-8}});
    test.AddInput<T>("Y", {4}, {T{1}, T{2}, bit_width, T{-1}});
    test.AddOutput<T>("Z", {4}, {T{-4}, T{-2}, T{-1}, T{-1}});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  }

  {
    OpTester test("BitShift", 28);
    test.AddAttribute("direction", "LEFT");
    test.AddInput<T>("X", {2}, {std::numeric_limits<T>::max(), T{-8}});
    test.AddInput<T>("Y", {1}, {T{1}});
    test.AddOutput<T>("Z", {2}, {T{-2}, T{-16}});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  }
}

TEST(BitShiftOpTest, SignedInt8Opset28) {
  TestSignedBitShiftOpset28<int8_t>();
}

TEST(BitShiftOpTest, SignedInt16Opset28) {
  TestSignedBitShiftOpset28<int16_t>();
}

TEST(BitShiftOpTest, SignedInt32Opset28) {
  TestSignedBitShiftOpset28<int32_t>();
}

TEST(BitShiftOpTest, SignedInt64Opset28) {
  TestSignedBitShiftOpset28<int64_t>();
}

template <typename T>
void TestBitShiftOpset28ProviderFallback(const std::vector<T>& x, const std::vector<T>& y,
                                         const std::vector<T>& expected) {
  const std::vector<int64_t> shape{static_cast<int64_t>(x.size())};
  OpTester test("BitShift", 28);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<T>("X", shape, x);
  test.AddInput<T>("Y", shape, y);
  test.AddOutput<T>("Z", shape, expected);
  test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig();
  auto provider = DefaultOpenVINOExecutionProvider();
  if (!provider) {
    return;
  }

  auto& model = test.BuildModel();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));
  EPVerificationParams params;
  params.ep_node_assignment = ExpectedEPNodeAssignment::None;
  RunAndVerifyOutputsWithEP(
      AsByteSpan(model_data.data(), model_data.size()), CurrentTestName(), std::move(provider),
      {{"X", CreateInputOrtValueOnCPU<T>(shape, x)}, {"Y", CreateInputOrtValueOnCPU<T>(shape, y)}},
      params);
}

TEST(BitShiftOpTest, Opset28ProviderFallback) {
  TestBitShiftOpset28ProviderFallback<int32_t>({-8, -3, -2, -2}, {1, 2, -1, 32}, {-4, -1, -1, -1});
  TestBitShiftOpset28ProviderFallback<uint32_t>({8, 255, 1, 42}, {1, 32, 33, 64}, {4, 0, 0, 0});
}

template <typename T>
void TestUnsignedBitShiftRegistration(int opset) {
  OpTester test("BitShift", opset);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<T>("X", {2}, {T{1}, T{3}});
  test.AddInput<T>("Y", {2}, {T{1}, T{2}});
  test.AddOutput<T>("Z", {2}, {T{2}, T{12}});
  test.Run();
}

TEST(BitShiftOpTest, Uint16Opset11) {
  TestUnsignedBitShiftRegistration<uint16_t>(11);
}

TEST(BitShiftOpTest, UnsignedTypesOpset28) {
  TestUnsignedBitShiftRegistration<uint8_t>(28);
  TestUnsignedBitShiftRegistration<uint16_t>(28);
  TestUnsignedBitShiftRegistration<uint32_t>(28);
  TestUnsignedBitShiftRegistration<uint64_t>(28);
}

TEST(BitShiftOpTest, RightShiftByBitWidth_Uint64Opset28) {
  OpTester test("BitShift", 28);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint64_t>("X", {4}, {1000, 255, 1, 42});
  test.AddInput<uint64_t>("Y", {4}, {64, 64, 64, 64});
  test.AddOutput<uint64_t>("Z", {4}, {0, 0, 0, 0});
  test.Run();
}

// Test that shift amounts >= bit width produce 0 (not undefined behavior).
// DirectML EP has the same hardware-level shift masking behavior, so skip these tests for DML.
TEST(BitShiftOpTest, RightShiftByBitWidth_Uint64) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint64_t>("X", {4}, {1000, 255, 1, 42});
  test.AddInput<uint64_t>("Y", {4}, {64, 64, 64, 64});
  test.AddOutput<uint64_t>("Z", {4}, {0, 0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(BitShiftOpTest, LeftShiftByBitWidth_Uint64) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint64_t>("X", {4}, {1000, 255, 1, 42});
  test.AddInput<uint64_t>("Y", {4}, {64, 64, 64, 64});
  test.AddOutput<uint64_t>("Z", {4}, {0, 0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(BitShiftOpTest, RightShiftByBitWidth_Uint32) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint32_t>("X", {3}, {16, 4, 1});
  test.AddInput<uint32_t>("Y", {3}, {32, 32, 32});
  test.AddOutput<uint32_t>("Z", {3}, {0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(BitShiftOpTest, RightShiftByMoreThanBitWidth_Uint64) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint64_t>("X", {2}, {1000, 42});
  test.AddInput<uint64_t>("Y", {2}, {65, 128});
  test.AddOutput<uint64_t>("Z", {2}, {0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(BitShiftOpTest, ScalarRightShiftByBitWidth_Uint64) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "RIGHT");
  test.AddInput<uint64_t>("X", {1}, {1000});
  test.AddInput<uint64_t>("Y", {3}, {64, 65, 128});
  test.AddOutput<uint64_t>("Z", {3}, {0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(BitShiftOpTest, ScalarLeftShiftByBitWidth_Uint64) {
  OpTester test("BitShift", 11);
  test.AddAttribute("direction", "LEFT");
  test.AddInput<uint64_t>("X", {3}, {1000, 255, 42});
  test.AddInput<uint64_t>("Y", {1}, {64});
  test.AddOutput<uint64_t>("Z", {3}, {0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

TEST(MathOpTest, BitwiseAnd) {
  OpTester test("BitwiseAnd", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int32_t>("A", dims, {-1, -2, 3});
  test.AddInput<int32_t>("B", dims, {1, 0, 3});
  test.AddOutput<int32_t>("C", dims, {1, 0, 3});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_int64) {
  OpTester test("BitwiseAnd", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("A", dims, {-1, -2, 3});
  test.AddInput<int64_t>("B", dims, {1, 0, 3});
  test.AddOutput<int64_t>("C", dims, {1, 0, 3});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_int8) {
  OpTester test("BitwiseAnd", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int8_t>("A", dims, {-1, -2, 3});
  test.AddInput<int8_t>("B", dims, {1, 0, 3});
  test.AddOutput<int8_t>("C", dims, {1, 0, 3});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_uint8) {
  OpTester test("BitwiseAnd", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {1, 4, 5, 3});
  test.AddInput<uint8_t>("B", dims, {1, 2, 2, 1});
  test.AddOutput<uint8_t>("C", dims, {1, 0, 0, 1});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_uint64) {
  OpTester test("BitwiseAnd", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<uint64_t>("C", dims, {10, 3, 12, 9});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_broadcastAB) {
  OpTester test("BitwiseAnd", 18);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {10, 3, 12, 5, 14, 7, 0, 1});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_broadcastBA) {
  OpTester test("BitwiseAnd", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<int32_t>("C", {4, 2}, {10, 3, 12, 5, 14, 7, 0, 1});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_multidirectional_broadcastAB) {
  OpTester test("BitwiseAnd", 18);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {10, 2, 11, 3, 12, 4, 13, 5});
  test.Run();
}

TEST(MathOpTest, BitwiseAnd_multidirectional_broadcastBA) {
  OpTester test("BitwiseAnd", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<int32_t>("C", {4, 2}, {10, 2, 11, 3, 12, 4, 13, 5});
  test.Run();
}

TEST(MathOpTest, BitwiseOr) {
  OpTester test("BitwiseOr", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int32_t>("A", dims, {-1, -2, 3});
  test.AddInput<int32_t>("B", dims, {1, 0, 4});
  test.AddOutput<int32_t>("C", dims, {-1, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_int64) {
  OpTester test("BitwiseOr", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("A", dims, {-1, -2, 3});
  test.AddInput<int64_t>("B", dims, {1, 0, 4});
  test.AddOutput<int64_t>("C", dims, {-1, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_int8) {
  OpTester test("BitwiseOr", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int8_t>("A", dims, {-1, -2, 3});
  test.AddInput<int8_t>("B", dims, {1, 0, 4});
  test.AddOutput<int8_t>("C", dims, {-1, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_uint8) {
  OpTester test("BitwiseOr", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {1, 4, 5, 3});
  test.AddInput<uint8_t>("B", dims, {1, 2, 2, 1});
  test.AddOutput<uint8_t>("C", dims, {1, 6, 7, 3});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_uint64) {
  OpTester test("BitwiseOr", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<uint64_t>("C", dims, {15, 15, 12, 13});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_broadcastAB) {
  OpTester test("BitwiseOr", 18);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {15, 15, 15, 15, 15, 15, 31, 23});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_broadcastBA) {
  OpTester test("BitwiseOr", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<int32_t>("C", {4, 2}, {15, 15, 15, 15, 15, 15, 31, 23});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_multidirectional_broadcastAB) {
  OpTester test("BitwiseOr", 18);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {15, 15, 15, 15, 15, 15, 15, 15});
  test.Run();
}

TEST(MathOpTest, BitwiseOr_multidirectional_broadcastBA) {
  OpTester test("BitwiseOr", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<int32_t>("C", {4, 2}, {15, 15, 15, 15, 15, 15, 15, 15});
  test.Run();
}

TEST(MathOpTest, BitwiseXor) {
  OpTester test("BitwiseXor", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int32_t>("A", dims, {-1, -2, 3});
  test.AddInput<int32_t>("B", dims, {1, 0, 4});
  test.AddOutput<int32_t>("C", dims, {-2, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_int64) {
  OpTester test("BitwiseXor", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("A", dims, {-1, -2, 3});
  test.AddInput<int64_t>("B", dims, {1, 0, 4});
  test.AddOutput<int64_t>("C", dims, {-2, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_int8) {
  OpTester test("BitwiseXor", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int8_t>("A", dims, {-1, -2, 3});
  test.AddInput<int8_t>("B", dims, {1, 0, 4});
  test.AddOutput<int8_t>("C", dims, {-2, -2, 7});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_uint8) {
  OpTester test("BitwiseXor", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("A", dims, {1, 4, 5, 3});
  test.AddInput<uint8_t>("B", dims, {1, 2, 2, 1});
  test.AddOutput<uint8_t>("C", dims, {0, 6, 7, 2});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_uint64) {
  OpTester test("BitwiseXor", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<uint64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<uint64_t>("C", dims, {5, 12, 0, 4});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_broadcastAB) {
  OpTester test("BitwiseXor", 18);
  test.AddInput<int32_t>("A", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {5, 12, 3, 10, 1, 8, 31, 22});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_broadcastBA) {
  OpTester test("BitwiseXor", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 2}, {10, 11, 12, 13, 14, 15, 16, 17});
  test.AddOutput<int32_t>("C", {4, 2}, {5, 12, 3, 10, 1, 8, 31, 22});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_multidirectional_broadcastAB) {
  OpTester test("BitwiseXor", 18);
  test.AddInput<int32_t>("A", {4, 1}, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", {2}, {15, 7});
  test.AddOutput<int32_t>("C", {4, 2}, {5, 13, 4, 12, 3, 11, 2, 10});
  test.Run();
}

TEST(MathOpTest, BitwiseXor_multidirectional_broadcastBA) {
  OpTester test("BitwiseXor", 18);
  test.AddInput<int32_t>("A", {2}, {15, 7});
  test.AddInput<int32_t>("B", {4, 1}, {10, 11, 12, 13});
  test.AddOutput<int32_t>("C", {4, 2}, {5, 13, 4, 12, 3, 11, 2, 10});
  test.Run();
}

TEST(MathOpTest, BitwiseNot) {
  OpTester test("BitwiseNot", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int32_t>("X", dims, {-1, -2, 3});
  test.AddOutput<int32_t>("Y", dims, {0, 1, -4});
  test.Run();
}

TEST(MathOpTest, BitwiseNot_int64) {
  OpTester test("BitwiseNot", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int64_t>("X", dims, {-1, -2, 3});
  test.AddOutput<int64_t>("Y", dims, {0, 1, -4});
  test.Run();
}

TEST(MathOpTest, BitwiseNot_int8) {
  OpTester test("BitwiseNot", 18);
  std::vector<int64_t> dims{3};
  test.AddInput<int8_t>("X", dims, {-1, -2, 3});
  test.AddOutput<int8_t>("Y", dims, {0, 1, -4});
  test.Run();
}

TEST(MathOpTest, BitwiseNot_uint8) {
  OpTester test("BitwiseNot", 18);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("X", dims, {1, 4, 5, 3});
  test.AddOutput<uint8_t>("Y", dims, {254, 251, 250, 252});
  test.Run();
}

// EP-agnostic int64 coverage for Sub/Equal/Add. These run on every capable EP (including WebGPU,
// for which the test harness enables int64). The shapes are chosen to exercise WebGPU's int64
// shader paths (element-wise, broadcast, size divisible by 4, scalar operands); values stay within
// the int32 range so 32-bit-lane int64 implementations (e.g. WebGPU) stay correct.
TEST(MathOpTest, Sub_Int64) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {4}, {10, 5, -3, 0});
  test.AddInput<int64_t>("B", {4}, {3, 5, 1, -7});
  test.AddOutput<int64_t>("C", {4}, {7, 0, -4, 7});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// INT64 Sub with broadcast: A [1,3] broadcasts against B [2,3] -> output [2,3].
TEST(MathOpTest, Sub_Int64_broadcast) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {1, 3}, {10, 20, 30});
  test.AddInput<int64_t>("B", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<int64_t>("C", {2, 3}, {9, 18, 27, 6, 15, 24});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Size divisible by 4: catches future vec4 optimizations that might break INT64.
TEST(MathOpTest, Sub_Int64_size_div4) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {8}, {10, 20, 30, 40, 50, 60, 70, 80});
  test.AddInput<int64_t>("B", {8}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("C", {8}, {9, 18, 27, 36, 45, 54, 63, 72});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar LHS: exercises is_lhs_scalar_ path for INT64 Sub.
TEST(MathOpTest, Sub_Int64_lhs_scalar) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {1}, {100});
  test.AddInput<int64_t>("B", {5}, {1, 2, 3, 4, 5});
  test.AddOutput<int64_t>("C", {5}, {99, 98, 97, 96, 95});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar RHS: exercises is_rhs_scalar_ path for INT64 Sub.
TEST(MathOpTest, Sub_Int64_rhs_scalar) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {5}, {10, 20, 30, 40, 50});
  test.AddInput<int64_t>("B", {1}, {7});
  test.AddOutput<int64_t>("C", {5}, {3, 13, 23, 33, 43});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Shape broadcast, size divisible by 4: A [1,4] broadcasts against B [2,4] -> output [2,4].
TEST(MathOpTest, Sub_Int64_broadcast_size_div4) {
  OpTester test("Sub", 14);
  test.AddInput<int64_t>("A", {1, 4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("B", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("C", {2, 4}, {9, 18, 27, 36, 5, 14, 23, 32});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Equal_Int64) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {5}, {1, 0, -1, -1, 3});
  test.AddInput<int64_t>("B", {5}, {1, 1, 2, -1, 3});
  test.AddOutput<bool>("C", {5}, {true, false, false, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Size divisible by 4: exercises the packed-bool path without tail padding.
TEST(MathOpTest, Equal_Int64_size_div4) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {8}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int64_t>("B", {8}, {1, 0, 3, 0, 5, 0, 7, 0});
  test.AddOutput<bool>("C", {8}, {true, false, true, false, true, false, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar LHS: lhs is a scalar, rhs is a vector; exercises is_lhs_scalar_ path.
TEST(MathOpTest, Equal_Int64_lhs_scalar) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {1}, {3});
  test.AddInput<int64_t>("B", {5}, {1, 2, 3, 4, 3});
  test.AddOutput<bool>("C", {5}, {false, false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar RHS: rhs is a scalar, lhs is a vector; exercises is_rhs_scalar_ path.
TEST(MathOpTest, Equal_Int64_rhs_scalar) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {5}, {1, 2, 3, 4, 3});
  test.AddInput<int64_t>("B", {1}, {3});
  test.AddOutput<bool>("C", {5}, {false, false, true, false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Shape broadcast, size not divisible by 4: A [1,3] broadcasts against B [2,3] -> output [2,3].
// Exercises the INT64 broadcast path with tail padding in the packed-bool output.
TEST(MathOpTest, Equal_Int64_broadcast) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {1, 3}, {1, 2, 3});
  test.AddInput<int64_t>("B", {2, 3}, {1, 0, 3, 1, 2, 3});
  test.AddOutput<bool>("C", {2, 3}, {true, false, true, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Shape broadcast, size divisible by 4: A [1,4] broadcasts against B [2,4] -> output [2,4].
// Exercises the INT64 broadcast path without tail padding.
TEST(MathOpTest, Equal_Int64_broadcast_size_div4) {
  OpTester test("Equal", 13);
  test.AddInput<int64_t>("A", {1, 4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("B", {2, 4}, {1, 0, 3, 0, 1, 2, 3, 4});
  test.AddOutput<bool>("C", {2, 4}, {true, false, true, false, true, true, true, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Add_Int64) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {4}, {10, 5, -3, 0});
  test.AddInput<int64_t>("B", {4}, {3, 5, 1, -7});
  test.AddOutput<int64_t>("C", {4}, {13, 10, -2, -7});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// INT64 Add with broadcast: A [1,3] broadcasts against B [2,3] -> output [2,3].
TEST(MathOpTest, Add_Int64_broadcast) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {1, 3}, {10, 20, 30});
  test.AddInput<int64_t>("B", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<int64_t>("C", {2, 3}, {11, 22, 33, 14, 25, 36});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Size divisible by 4: catches future vec4 optimizations that might break INT64.
TEST(MathOpTest, Add_Int64_size_div4) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {8}, {10, 20, 30, 40, 50, 60, 70, 80});
  test.AddInput<int64_t>("B", {8}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("C", {8}, {11, 22, 33, 44, 55, 66, 77, 88});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar LHS: exercises is_lhs_scalar_ path for INT64 Add.
TEST(MathOpTest, Add_Int64_lhs_scalar) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {1}, {100});
  test.AddInput<int64_t>("B", {5}, {1, 2, 3, 4, 5});
  test.AddOutput<int64_t>("C", {5}, {101, 102, 103, 104, 105});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Scalar RHS: exercises is_rhs_scalar_ path for INT64 Add.
TEST(MathOpTest, Add_Int64_rhs_scalar) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {5}, {10, 20, 30, 40, 50});
  test.AddInput<int64_t>("B", {1}, {7});
  test.AddOutput<int64_t>("C", {5}, {17, 27, 37, 47, 57});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// Shape broadcast, size divisible by 4: A [1,4] broadcasts against B [2,4] -> output [2,4].
TEST(MathOpTest, Add_Int64_broadcast_size_div4) {
  OpTester test("Add", 14);
  test.AddInput<int64_t>("A", {1, 4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("B", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("C", {2, 4}, {11, 22, 33, 44, 15, 26, 37, 48});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
