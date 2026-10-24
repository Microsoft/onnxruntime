// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(ReductionOpTest, ReduceLogOps_Opset28_WebGpu) {
  for (int opset : {27, 28}) {
    for (const auto& [op, expected] :
         std::vector<std::pair<std::string, std::vector<float>>>{
             {"ReduceLogSum", {1.38629436f, 1.79175949f}},
             {"ReduceLogSumExp", {3.12692801f, 4.12692801f}}}) {
      auto provider = DefaultWebGpuExecutionProvider();
      if (!provider) {
        GTEST_SKIP() << "WebGPU execution provider is not available.";
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

TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {3}, {10, 20, 30});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {60});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Size divisible by 4: catches issues if the shader is ever accidentally vectorized for INT64.
TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64_SizeDiv4) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {100});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
