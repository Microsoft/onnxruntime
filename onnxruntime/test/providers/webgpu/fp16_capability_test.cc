// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

#if !defined(__wasm__)
namespace onnxruntime {
namespace test {
namespace {

class WebGpuFp16CapabilityTest : public testing::Test {
 protected:
  void SetUp() override {
    auto bootstrap = DefaultWebGpuExecutionProvider();
    if (!bootstrap) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }
    instance_ = webgpu::WebGpuContextFactory::GetContext(bootstrap->GetDeviceId()).Instance();
    bootstrap.reset();

    wgpu::RequestAdapterOptions adapter_options;
    adapter_options.backendType = static_cast<wgpu::BackendType>(webgpu::WebGpuContextConfig{}.backend_type);
    wgpu::Adapter adapter;
    std::string error;
    auto adapter_request = instance_.RequestAdapter(
        &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView message) noexcept {
          if (status == wgpu::RequestAdapterStatus::Success) adapter = std::move(result);
          error = std::string(message);
        });
    ASSERT_EQ(instance_.WaitAny(adapter_request, UINT64_MAX), wgpu::WaitStatus::Success);
    ASSERT_NE(adapter, nullptr) << error;

    // Request ORT's required native synchronization feature, leaving ShaderF16 disabled.
    const wgpu::FeatureName required_feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
    wgpu::DeviceDescriptor device_desc;
    device_desc.requiredFeatureCount = 1;
    device_desc.requiredFeatures = &required_feature;
    auto device_request = adapter.RequestDevice(
        &device_desc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView message) noexcept {
          if (status == wgpu::RequestDeviceStatus::Success) device_ = std::move(result);
          error = std::string(message);
        });
    ASSERT_EQ(instance_.WaitAny(device_request, UINT64_MAX), wgpu::WaitStatus::Success);
    ASSERT_NE(device_, nullptr) << error;
    ASSERT_FALSE(device_.HasFeature(wgpu::FeatureName::ShaderF16));

    ASSERT_STATUS_OK(options_.AddConfigEntry(webgpu::options::kDeviceId, "1"));
    ASSERT_STATUS_OK(options_.AddConfigEntry(webgpu::options::kWebGpuInstance,
                                             std::to_string(reinterpret_cast<uintptr_t>(instance_.Get())).c_str()));
    ASSERT_STATUS_OK(options_.AddConfigEntry(webgpu::options::kWebGpuDevice,
                                             std::to_string(reinterpret_cast<uintptr_t>(device_.Get())).c_str()));
    ep_ = CreateEp();
    ASSERT_NE(ep_, nullptr);
    ASSERT_FALSE(webgpu::WebGpuContextFactory::GetContext(ep_->GetDeviceId())
                     .DeviceHasFeature(wgpu::FeatureName::ShaderF16));
  }

  static void AddReluData(OpTester& test) {
    test.AddInput<MLFloat16>("X", {2}, {MLFloat16(-1.0f), MLFloat16(2.0f)});
    test.AddOutput<MLFloat16>("Y", {2}, {MLFloat16(0.0f), MLFloat16(2.0f)});
  }

  std::unique_ptr<IExecutionProvider> CreateEp() const {
    return WebGpuExecutionProviderWithOptions(options_);
  }

  void ExpectSkip(OpTester& test) {
    testing::TestPartResultArray results;
    {
      testing::ScopedFakeTestPartResultReporter reporter(
          testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
      test.ConfigSkipUnsupportedWebGpuFp16().ConfigEp(CreateEp()).RunWithConfig();
    }
    ASSERT_EQ(results.size(), 1);
    EXPECT_TRUE(results.GetTestPartResult(0).skipped()) << results.GetTestPartResult(0).message();
  }

 private:
  ConfigOptions options_;
  wgpu::Instance instance_;
  wgpu::Device device_;
  std::unique_ptr<IExecutionProvider> ep_;
};

TEST_F(WebGpuFp16CapabilityTest, SkipsUnsupportedExecution) {
  OpTester test("Relu", 14);
  AddReluData(test);
  ExpectSkip(test);
}

TEST_F(WebGpuFp16CapabilityTest, SkipsUnsupportedPrepacking) {
  OpTester test("Conv", 11);
  test.AddInput<MLFloat16>("X", {1, 1, 3, 3}, std::vector<MLFloat16>(9, MLFloat16(1.0f)));
  test.AddInput<MLFloat16>("W", {1, 1, 2, 2}, std::vector<MLFloat16>(4, MLFloat16(1.0f)), true);
  test.AddOutput<MLFloat16>("Y", {1, 1, 2, 2}, std::vector<MLFloat16>(4, MLFloat16(4.0f)));
  ExpectSkip(test);
}

TEST(WebGpuFp16CoverageTest, PreservesOtherProviderCoverage) {
  OpTester test("Cast", 13);
  test.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  test.AddInput<MLFloat16>("input", {2}, {MLFloat16(-1.0f), MLFloat16(2.0f)});
  test.AddOutput<float>("output", {2}, {-1.0f, 2.0f});
  bool cpu_ran = false;
  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string& provider) {
    cpu_ran |= provider == kCpuExecutionProvider;
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape().Size(), 2);
    EXPECT_EQ(output.Data<float>()[0], -1.0f);
    EXPECT_EQ(output.Data<float>()[1], 2.0f);
  });
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test.ConfigSkipUnsupportedWebGpuFp16().RunWithConfig();
  }
  ASSERT_EQ(results.size(), 0) << results.GetTestPartResult(0).message();
  EXPECT_TRUE(cpu_ran);
}

TEST_F(WebGpuFp16CapabilityTest, PreservesExpectedFailure) {
  OpTester test("Relu", 14);
  AddReluData(test);
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test.ConfigSkipUnsupportedWebGpuFp16()
        .Config(OpTester::ExpectResult::kExpectFailure, "requires f16 but the device does not support it.")
        .ConfigEp(CreateEp())
        .RunWithConfig();
  }
  ASSERT_EQ(results.size(), 0) << results.GetTestPartResult(0).message();
}

TEST_F(WebGpuFp16CapabilityTest, RequiresOptIn) {
  OpTester test("Relu", 14);
  AddReluData(test);
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test.ConfigEp(CreateEp()).RunWithConfig();
  }
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results.GetTestPartResult(0).fatally_failed());
}

TEST_F(WebGpuFp16CapabilityTest, PreservesOtherFailures) {
  OpTester test("Conv", 11);
  test.AddAttribute("group", int64_t{2});
  test.AddInput<float>("X", {1, 2, 1, 1, 1}, {1.0f, 1.0f});
  test.AddInput<float>("W", {2, 1, 1, 1, 1}, {1.0f, 1.0f});
  test.AddOutput<float>("Y", {1, 2, 1, 1, 1}, {1.0f, 1.0f});
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test.ConfigSkipUnsupportedWebGpuFp16().ConfigEp(CreateEp()).RunWithConfig();
  }
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results.GetTestPartResult(0).fatally_failed());
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(__wasm__)
