// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
#include "absl/flags/reflection.h"
#endif

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/temp_dir.h"

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
#include "core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"
#include "core/providers/webgpu/d3d12_external_data_loader.h"
#endif

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/DawnNative.h"
#include "test/providers/webgpu/webgpu_context_test_helper.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

using namespace webgpu::options;

ConfigOptions RobustnessOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kEnableRobustness, value));
  return options;
}

ConfigOptions KvCacheQuantizationOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kKvCacheQuantizationBits, value));
  return options;
}

template <typename TestBody>
void RunWithFreshDefaultContext(TestBody test_body, bool compile_only_parent = false) {
#if GTEST_HAS_DEATH_TEST
  // Context 0 can outlive an individual test. Re-exec instead of forking its
  // initialized Dawn device or clearing state that another EP still owns.
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  auto* death_test_style_flag = absl::FindCommandLineFlag("gtest_death_test_style");
  ASSERT_NE(death_test_style_flag, nullptr);
  const std::string previous_style = death_test_style_flag->CurrentValue();
  std::string flag_error;
  ASSERT_TRUE(death_test_style_flag->ParseFrom("threadsafe", &flag_error)) << flag_error;
#else
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif

  // Exercise isolation even when running only this test, and ensure the child
  // does not disturb a live provider in the parent process.
  std::unique_ptr<IExecutionProvider> existing_provider;
  WGPUDevice existing_device = nullptr;
  if (!testing::internal::InDeathTestChild()) {
    ConfigOptions options;
    if (compile_only_parent) {
      // Keep the compile-only test runnable on hosts without a GPU. An existing
      // device context, if any, will still be retained by this provider.
      ASSERT_STATUS_OK(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));
    }
    existing_provider = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(existing_provider, nullptr);
    existing_device = webgpu::WebGpuContextFactory::GetContext(0).Device().Get();
  }

  EXPECT_EXIT(
      {
        test_body();
        std::_Exit(testing::Test::HasFailure() || testing::Test::IsSkipped() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "");

  EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), existing_device);
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  flag_error.clear();
  EXPECT_TRUE(death_test_style_flag->ParseFrom(previous_style, &flag_error)) << flag_error;
#else
  GTEST_FLAG_SET(death_test_style, previous_style);
#endif
#else
  ORT_UNUSED_PARAMETER(compile_only_parent);
  test_body();
#endif
}

ConfigOptions WeightLoadAccelerationOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kWeightLoadAcceleration, value));
  return options;
}

ConfigOptions CompileOnlyWeightLoadAccelerationOptions(const char* value) {
  auto options = WeightLoadAccelerationOptions(value);
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));
  return options;
}

ConfigOptions EnvironmentAllocatorWeightLoadAccelerationOptions(const char* value) {
  auto options = CompileOnlyWeightLoadAccelerationOptions(value);
  ORT_THROW_IF_ERROR(
      options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1"));
  return options;
}

bool DeviceToggleIsEnabled(const webgpu::WebGpuContext& context, std::string_view toggle_name) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  const auto toggles = dawn::native::GetTogglesUsed(context.Device().Get());
  return std::any_of(toggles.begin(), toggles.end(), [toggle_name](const char* toggle) {
    return std::string_view{toggle} == toggle_name;
  });
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(toggle_name);
  return false;
#endif
}

bool DisableRobustnessToggleIsEnabled(const webgpu::WebGpuContext& context) {
  return DeviceToggleIsEnabled(context, "disable_robustness");
}

template <size_t ElementCount = 16>
std::array<uint32_t, ElementCount> ReadBufferWithExternalCommandEncoder(
    webgpu::WebGpuContext& context, WGPUBuffer buffer) {
  constexpr size_t kBufferSize =
      sizeof(std::array<uint32_t, ElementCount>);
  wgpu::BufferDescriptor readback_desc{};
  readback_desc.size = kBufferSize;
  readback_desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  auto readback_buffer = context.Device().CreateBuffer(&readback_desc);

  auto external_encoder = context.Device().CreateCommandEncoder();
  external_encoder.CopyBufferToBuffer(buffer, 0, readback_buffer, 0, kBufferSize);
  auto external_commands = external_encoder.Finish();
  context.Device().GetQueue().Submit(1, &external_commands);

  wgpu::MapAsyncStatus map_status{};
  ORT_THROW_IF_ERROR(context.Wait(readback_buffer.MapAsync(
      wgpu::MapMode::Read, 0, kBufferSize, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView /*message*/, wgpu::MapAsyncStatus* result) noexcept {
        *result = status;
      },
      &map_status)));
  ORT_ENFORCE(map_status == wgpu::MapAsyncStatus::Success);

  std::array<uint32_t, ElementCount> result;
  const auto* mapped_data = static_cast<const uint32_t*>(readback_buffer.GetConstMappedRange());
  ORT_ENFORCE(mapped_data != nullptr);
  std::copy_n(mapped_data, result.size(), result.begin());
  readback_buffer.Unmap();
  return result;
}

void TestCopyAfterDeferredDispatch(bool upload, bool checked_completion = false) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& recording = static_cast<WebGpuExecutionProvider*>(ep.get())->Recording();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& buffer_manager = context.BufferManager();

  std::array<uint32_t, 16> input_data;
  input_data.fill(7);
  wgpu::BufferDescriptor buffer_desc{};
  buffer_desc.size = sizeof(input_data);
  buffer_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto input = context.Device().CreateBuffer(&buffer_desc);
  auto output = context.Device().CreateBuffer(&buffer_desc);
  auto copy = context.Device().CreateBuffer(&buffer_desc);
  buffer_manager.Upload(recording, input_data.data(), input.Get(), sizeof(input_data));

  wgpu::ShaderSourceWGSL source{};
  source.code = R"(
    @group(0) @binding(0) var<storage, read> input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @compute @workgroup_size(16)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      output[id.x] = input[id.x] + 1u;
    }
  )";
  wgpu::ShaderModuleDescriptor shader_desc{};
  shader_desc.nextInChain = &source;
  wgpu::ComputePipelineDescriptor pipeline_desc{};
  pipeline_desc.compute.module = context.Device().CreateShaderModule(&shader_desc);
  pipeline_desc.compute.entryPoint = "main";
  auto pipeline = context.Device().CreateComputePipeline(&pipeline_desc);
  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].buffer = input;
  entries[0].size = sizeof(input_data);
  entries[1].binding = 1;
  entries[1].buffer = output;
  entries[1].size = sizeof(input_data);
  wgpu::BindGroupDescriptor bind_group_desc{};
  bind_group_desc.layout = pipeline.GetBindGroupLayout(0);
  bind_group_desc.entryCount = entries.size();
  bind_group_desc.entries = entries.data();
  webgpu::DispatchCommandInfo dispatch;
  dispatch.compute_pipeline = pipeline;
  dispatch.bind_group = context.Device().CreateBindGroup(&bind_group_desc);
  recording.deferred_dispatches.push_back(std::move(dispatch));
  recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);

  if (checked_completion) {
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(buffer_manager, recording));
  } else if (upload) {
    // The dispatch must consume the original input before Upload overwrites it.
    input_data.fill(42);
    buffer_manager.Upload(recording, input_data.data(), input.Get(), sizeof(input_data));
  } else {
    // MemCpy must read the dispatch result, not the output buffer's initial zeros.
    buffer_manager.MemCpy(recording, output.Get(), copy.Get(), sizeof(input_data));
  }
  EXPECT_TRUE(recording.deferred_dispatches.empty());

  std::array<uint32_t, 16> result{};
  buffer_manager.Download(recording, upload ? output.Get() : copy.Get(), result.data(), sizeof(result));
  std::array<uint32_t, 16> expected;
  expected.fill(8);
  EXPECT_EQ(result, expected);
}

TEST(WebGpuContextTest, UploadFollowsDeferredDispatch) {
  TestCopyAfterDeferredDispatch(true);
}

TEST(WebGpuContextTest, MemCpyFollowsDeferredDispatch) {
  TestCopyAfterDeferredDispatch(false);
}

void ExpectQueueCompleted(webgpu::WebGpuContext& context) {
  auto result = std::make_shared<wgpu::QueueWorkDoneStatus>(wgpu::QueueWorkDoneStatus::CallbackCancelled);
  const auto future = context.Device().GetQueue().OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [result](wgpu::QueueWorkDoneStatus status, wgpu::StringView /*message*/) noexcept {
        *result = status;
      });
  EXPECT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
  EXPECT_EQ(*result, wgpu::QueueWorkDoneStatus::Success);
}

TEST(WebGpuContextTest, CheckedCompletionHandlesEmptyAndPendingEncoders) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();

  for (bool pending_encoder : {false, true}) {
    SCOPED_TRACE(pending_encoder);
    if (pending_encoder) {
      context.GetCommandEncoder(recording);
      ASSERT_TRUE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
    }
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
    EXPECT_EQ(recording.command_encoder, nullptr);
    EXPECT_FALSE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
    ExpectQueueCompleted(context);
  }
}

TEST(WebGpuContextTest, CheckedCompletionSubmitsAndWaitsForCopies) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
  std::array<uint32_t, 16> expected;
  expected.fill(42);
  wgpu::BufferDescriptor desc{};
  desc.size = sizeof(expected);
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto source = context.Device().CreateBuffer(&desc);
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  auto destination = context.Device().CreateBuffer(&desc);
  context.Device().GetQueue().WriteBuffer(source, 0, expected.data(), sizeof(expected));

  for (bool already_submitted : {false, true}) {
    SCOPED_TRACE(already_submitted);
    context.GetCommandEncoder(recording).CopyBufferToBuffer(source, 0, destination, 0, sizeof(expected));
    if (already_submitted) {
      ASSERT_STATUS_OK(context.Flush(webgpu_ep.BufferManager(), recording));
    }
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
    ExpectQueueCompleted(context);

    auto map_status = std::make_shared<wgpu::MapAsyncStatus>(wgpu::MapAsyncStatus::Error);
    ASSERT_STATUS_OK(context.Wait(destination.MapAsync(
        wgpu::MapMode::Read, 0, sizeof(expected), wgpu::CallbackMode::WaitAnyOnly,
        [map_status](wgpu::MapAsyncStatus status, wgpu::StringView /*message*/) noexcept {
          *map_status = status;
        })));
    ASSERT_EQ(*map_status, wgpu::MapAsyncStatus::Success);
    std::array<uint32_t, 16> actual;
    std::copy_n(static_cast<const uint32_t*>(destination.GetConstMappedRange()), actual.size(), actual.begin());
    EXPECT_EQ(actual, expected);
    destination.Unmap();
  }
}

TEST(WebGpuContextTest, CheckedCompletionEncodesDeferredDispatch) {
  TestCopyAfterDeferredDispatch(true, true);
}

TEST(WebGpuContextTest, CheckedCompletionDrainsAfterDeferredDispatchFailure) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
  webgpu::DispatchCommandInfo invalid_dispatch;
  invalid_dispatch.program_key = "checked-completion-missing-pipeline";
  recording.deferred_dispatches.push_back(std::move(invalid_dispatch));
  recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);

  const auto status = context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("No cached or pending pipeline"));
  EXPECT_TRUE(recording.deferred_dispatches.empty());
  EXPECT_FALSE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
  ExpectQueueCompleted(context);
  ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
}

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
class PipelineCallbackTestProgram final : public webgpu::Program<PipelineCallbackTestProgram> {
 public:
  PipelineCallbackTestProgram() : Program{"PipelineCallbackTest"} {}

  Status GenerateShaderCode(webgpu::ShaderHelper&) const override {
    return Status::OK();
  }
};

Status StartPipelineCallbackTestBuild(webgpu::WebGpuContext& context, webgpu::PendingPipelineBuild& build,
                                      bool invalid_shader) {
  PipelineCallbackTestProgram program;
  program.SetWorkgroupSize(1);
  program.CacheHint(invalid_shader);
  auto metadata = program.Metadata();
  const webgpu::ProgramConstant invalid_constant{"invalid identifier", uint32_t{0}};
  if (invalid_shader) {
    metadata.constants = {&invalid_constant, 1};
  }
  webgpu::ProgramManager manager{context};
  const auto program_key = webgpu::CalculateProgramCacheKey(program, {}, {});
  build.name = program.Name();
  build.callback_context = std::make_shared<webgpu::PipelineCallbackContext>();
  return manager.Build(program, metadata, {}, {}, program_key, 1, 1, 1,
                       build.bind_group_layout, build.shape_uniform_ranks, build.future, build.callback_context);
}

#endif

TEST(WebGpuContextTest, SuccessfulPipelineBuildReleasesCallbackOwnership) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto&, const auto&, auto&) {
    webgpu::PendingPipelineBuild build;
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, false));
    const std::weak_ptr<webgpu::PipelineCallbackContext> weak_context = build.callback_context;
    ASSERT_STATUS_OK(context.Wait(build.future));
    ASSERT_STATUS_OK(build.callback_context->status);
    EXPECT_NE(build.callback_context->pipeline, nullptr);
    EXPECT_EQ(weak_context.use_count(), 1);
    build.callback_context.reset();
    EXPECT_TRUE(weak_context.expired());
  });
#endif
}

TEST(WebGpuContextTest, FailedPipelineWaitRetainsCallbackUntilDelivery) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    // Invalid WGSL makes the pipeline event ready synchronously, but WaitAnyOnly defers delivery.
    context.PushErrorScope();
    webgpu::DispatchCommandInfo dispatch;
    dispatch.program_key = "failed-wait-pipeline-callback";
    auto& build = dispatch.pending_build.emplace();
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, true));
    const auto future = build.future;
    const std::weak_ptr<webgpu::PipelineCallbackContext> weak_context = build.callback_context;
    auto& recording = ep.Recording();
    recording.deferred_dispatches.push_back(std::move(dispatch));

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(context.Flush(ep.BufferManager(), recording),
                                        "Failed to wait for the operation");
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    ASSERT_EQ(weak_context.use_count(), 1);

    auto callback_context = weak_context.lock();
    ASSERT_NE(callback_context, nullptr);
    ASSERT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
    EXPECT_FALSE(callback_context->status.IsOK());
    EXPECT_THAT(callback_context->status.ErrorMessage(),
                testing::HasSubstr("Failed to create a WebGPU compute pipeline"));
    EXPECT_EQ(callback_context->pipeline, nullptr);
    EXPECT_EQ(weak_context.use_count(), 1);
    callback_context.reset();
    EXPECT_TRUE(weak_context.expired());

    auto scope_error = std::make_shared<wgpu::ErrorType>(wgpu::ErrorType::NoError);
    const auto scope_future = context.Device().PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [scope_error](wgpu::PopErrorScopeStatus, wgpu::ErrorType error, wgpu::StringView) noexcept {
          *scope_error = error;
        });
    ASSERT_EQ(context.Instance().WaitAny(scope_future, 0), wgpu::WaitStatus::Success);
    EXPECT_EQ(*scope_error, wgpu::ErrorType::Validation);
  },
                                   false);
#endif
}

TEST(WebGpuContextTest, FailedPipelineWaitRetainsCallbackAfterContextDestruction) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  wgpu::Instance callback_instance;
  wgpu::Future future{};
  std::weak_ptr<webgpu::PipelineCallbackContext> weak_context;
  RunWithExternalCompletionContext([&](auto& context, auto& ep, const auto&, auto&) {
    context.PushErrorScope();
    webgpu::DispatchCommandInfo dispatch;
    dispatch.program_key = "post-context-pipeline-callback";
    auto& build = dispatch.pending_build.emplace();
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, true));
    callback_instance = context.Instance();
    future = build.future;
    weak_context = build.callback_context;
    auto& recording = ep.Recording();
    recording.deferred_dispatches.push_back(std::move(dispatch));

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(context.Flush(ep.BufferManager(), recording),
                                        "Failed to wait for the operation");
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    EXPECT_EQ(weak_context.use_count(), 1);
    // Leave the callback undelivered until after the ORT context and manager are destroyed.
  },
                                   false);
  EXPECT_THROW(webgpu::WebGpuContextFactory::GetContext(1), OnnxRuntimeException);
  ASSERT_EQ(weak_context.use_count(), 1);
  auto callback_context = weak_context.lock();
  ASSERT_NE(callback_context, nullptr);
  EXPECT_TRUE(callback_context->status.IsOK());
  ASSERT_EQ(callback_instance.WaitAny(future, 0), wgpu::WaitStatus::Success);
  EXPECT_EQ(callback_context.use_count(), 1);
  callback_context.reset();
  EXPECT_TRUE(weak_context.expired());
#endif
}

TEST(WebGpuContextTest, CheckedCompletionPropagatesValidationAndPreservesOuterScope) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    auto& recording = ep.Recording();
    wgpu::BufferDescriptor desc{};
    desc.size = 64;
    desc.usage = wgpu::BufferUsage::CopySrc;
    auto buffer = context.Device().CreateBuffer(&desc);

    context.PushErrorScope();
    context.Device().InjectError(wgpu::ErrorType::Validation, "checked-completion-outer-scope-error");
    // Poison the encoder before the helper's scopes; finishing it must report validation too.
    context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
    const auto status = context.FlushAndWaitChecked(ep.BufferManager(), recording);
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("WebGPU error scope failed"));
    const auto outer_status = context.PopErrorScope();
    EXPECT_FALSE(outer_status.IsOK());
    EXPECT_THAT(outer_status.ErrorMessage(), testing::HasSubstr("checked-completion-outer-scope-error"));
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
    ExpectQueueCompleted(context);
  });
#endif
}

TEST(WebGpuContextTest, ErrorScopesPropagateValidationAndOutOfMemory) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native error injection is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    for (const auto filter : {wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory}) {
      SCOPED_TRACE(static_cast<uint32_t>(filter));
      const auto type = filter == wgpu::ErrorFilter::Validation ? wgpu::ErrorType::Validation
                                                                : wgpu::ErrorType::OutOfMemory;
      context.PushErrorScope(filter);
      context.Device().InjectError(type, "checked-completion-scope-error");
      const auto status = context.PopErrorScope();
      EXPECT_FALSE(status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-completion-scope-error"));
      ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
    }
  });
#endif
}

TEST(WebGpuContextTest, CheckedCompletionCleansScopesAfterWaitFailure) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    const auto status = context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording());
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Failed to wait for the operation"));

    auto pop_status = std::make_shared<wgpu::PopErrorScopeStatus>(wgpu::PopErrorScopeStatus::Success);
    const auto future = context.Device().PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [pop_status](wgpu::PopErrorScopeStatus status, wgpu::ErrorType, wgpu::StringView) noexcept {
          *pop_status = status;
        });
    ASSERT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
    // No helper-owned scopes remain, even though its completion/scope waits failed.
    EXPECT_EQ(*pop_status, wgpu::PopErrorScopeStatus::Error);
  },
                                   false);
#endif
}

TEST(WebGpuContextTest, CheckedRegionReportsEarlyShaderErrorOnExternalDevice) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    wgpu::ShaderSourceWGSL source{};
    source.code = "checked_region_invalid_shader";
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &source;
    const auto status = context.RunAndWaitChecked(ep.BufferManager(), ep.Recording(), [&]() {
      context.Device().CreateShaderModule(&desc);
      return Status::OK();
    });
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("CreateShaderModule"));
    // No invalid command is recorded: a successful queue wait alone cannot detect this error.
    EXPECT_EQ(ep.Recording().command_encoder, nullptr);
    ExpectQueueCompleted(context);
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
  });
#endif
}

void TestCheckedRegionEnqueueFailure(bool throw_exception) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  ORT_UNUSED_PARAMETER(throw_exception);
  GTEST_SKIP() << "Dawn native error injection is unavailable.";
#else
  RunWithExternalCompletionContext([throw_exception](auto& context, auto& ep, const auto&, auto&) {
    context.PushErrorScope();
    context.Device().InjectError(wgpu::ErrorType::Validation, "checked-region-outer-error");
    size_t enqueue_count = 0;
    const auto status = context.RunAndWaitChecked(ep.BufferManager(), ep.Recording(), [&]() -> Status {
      ++enqueue_count;
      context.GetCommandEncoder(ep.Recording());
      context.Device().InjectError(wgpu::ErrorType::Validation, "checked-region-validation-error");
      context.Device().InjectError(wgpu::ErrorType::OutOfMemory, "checked-region-oom-error");
      if (throw_exception) {
        ORT_THROW("checked-region-enqueue-failure");
      }
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "checked-region-enqueue-failure");
    });
    const auto outer_status = context.PopErrorScope();
    EXPECT_EQ(enqueue_count, 1u);
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-region-enqueue-failure"));
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-region-validation-error"));
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-region-oom-error"));
    EXPECT_FALSE(outer_status.IsOK());
    EXPECT_THAT(outer_status.ErrorMessage(), testing::HasSubstr("checked-region-outer-error"));
    EXPECT_TRUE(ep.Recording().deferred_dispatches.empty());
    EXPECT_EQ(ep.Recording().command_encoder, nullptr);
    EXPECT_FALSE(ep.Recording().has_unsubmitted_work.load(std::memory_order_relaxed));
    ExpectQueueCompleted(context);
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
  });
#endif
}

TEST(WebGpuContextTest, CheckedRegionCleansScopesAfterEnqueueFailure) {
  TestCheckedRegionEnqueueFailure(false);
}

TEST(WebGpuContextTest, CheckedRegionCleansScopesAfterEnqueueException) {
#ifdef ORT_NO_EXCEPTIONS
  GTEST_SKIP() << "Exceptions are disabled.";
#else
  TestCheckedRegionEnqueueFailure(true);
#endif
}

void TestCheckedCompletionUncapturedError(wgpu::ErrorType type) {
#if !GTEST_HAS_DEATH_TEST || defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  ORT_UNUSED_PARAMETER(type);
  GTEST_SKIP() << "Uncaptured errors require an isolated ORT-owned Dawn device.";
#else
  RunWithFreshDefaultContext([type]() {
    ConfigOptions options;
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    context.Device().InjectError(type, "checked-completion-uncaptured-error");
    const auto status = context.FlushAndWaitChecked(webgpu_ep.BufferManager(), webgpu_ep.Recording());
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-completion-uncaptured-error"));
    EXPECT_FALSE(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), webgpu_ep.Recording()).IsOK());
    context.PushErrorScope();
    EXPECT_TRUE(context.PopErrorScope().IsOK());
  });
#endif
}

TEST(WebGpuContextTest, CheckedCompletionReportsEarlierUncapturedValidation) {
  TestCheckedCompletionUncapturedError(wgpu::ErrorType::Validation);
}

TEST(WebGpuContextTest, CheckedCompletionReportsEarlierUncapturedOutOfMemory) {
  TestCheckedCompletionUncapturedError(wgpu::ErrorType::OutOfMemory);
}

TEST(WebGpuContextTest, CheckedCompletionDetectsLostExternalDevice) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device loss injection is unavailable.";
#else
  for (bool pending_encoder : {false, true}) {
    RunWithExternalCompletionContext([pending_encoder](auto& context, auto& ep, const auto& lost, auto&) {
      auto& recording = ep.Recording();
      ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
      if (pending_encoder) {
        context.GetCommandEncoder(recording);
      }
      context.Device().ForceLoss(wgpu::DeviceLostReason::Unknown, "checked-completion-device-loss");

      // Dawn can report successful queue completion on a lost device.
      ExpectQueueCompleted(context);
      const auto status = context.FlushAndWaitChecked(ep.BufferManager(), recording);
      EXPECT_FALSE(status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("WebGPU device lost"));
      EXPECT_TRUE(*lost);
      EXPECT_FALSE(context.FlushAndWaitChecked(ep.BufferManager(), recording).IsOK());
    });
  }
#endif
}

TEST(WebGpuContextTest, BufferReuseFollowsCachePolicyAndRecordingSubmission) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;

  for (auto mode : {webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::LazyRelease,
                    webgpu::BufferCacheMode::Bucket, webgpu::BufferCacheMode::Simple,
                    webgpu::BufferCacheMode::Graph, webgpu::BufferCacheMode::GraphSimple}) {
    SCOPED_TRACE(static_cast<int>(mode));
    webgpu::BufferManager manager(context, mode, webgpu::BufferCacheMode::Disabled,
                                  webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled);
    webgpu::CommandRecordingState first;
    webgpu::CommandRecordingState second;
    auto buffer = manager.Create(first, 64, usage);
    wgpu::Buffer retained{buffer};
    context.GetCommandEncoder(first).ClearBuffer(buffer, 0, 64);
    manager.Release(buffer, &first);

    auto within_batch = manager.Create(first, 64, usage);
    EXPECT_NE(within_batch, buffer);
    auto other_buffer = manager.Create(second, 64, usage);
    EXPECT_NE(other_buffer, buffer);
    context.GetCommandEncoder(second).ClearBuffer(other_buffer, 0, 64);
    ASSERT_STATUS_OK(context.Flush(manager, second));
    // A submission on another recording cannot make the first buffer reusable.
    auto after_other_submission = manager.Create(second, 64, usage);
    EXPECT_NE(after_other_submission, buffer);

    ASSERT_STATUS_OK(context.Flush(manager, first));
    auto reused = manager.Create(second, 64, usage);
    if (mode == webgpu::BufferCacheMode::Disabled || mode == webgpu::BufferCacheMode::LazyRelease) {
      EXPECT_NE(reused, buffer);
    } else {
      EXPECT_EQ(reused, buffer);
    }
    manager.Release(reused);
    manager.Release(within_batch);
    manager.Release(other_buffer);
    manager.Release(after_other_submission);
  }
}

TEST(WebGpuContextTest, BufferReleasedAfterSubmissionIsReusableWithoutAnotherFlush) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  webgpu::CommandRecordingState other;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = manager.Create(recording, 64, usage);
  wgpu::Buffer retained{buffer};
  context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
  ASSERT_STATUS_OK(context.Flush(manager, recording));

  // An allocator can outlive its last Run. Free must not wait for another submission to this recording.
  manager.Release(buffer, &recording);
  auto reused = manager.Create(other, 64, usage);
  EXPECT_EQ(reused, buffer);
  manager.Release(reused);
}

TEST(WebGpuContextTest, EmptyFlushRefreshesIdleCacheWithoutReleasingAnotherRecording) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Simple,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState idle;
  webgpu::CommandRecordingState active;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto idle_buffer = manager.Create(idle, 64, usage);
  auto active_buffer = manager.Create(active, 64, usage);
  wgpu::Buffer retained_idle{idle_buffer};
  wgpu::Buffer retained_active{active_buffer};
  context.GetCommandEncoder(active).ClearBuffer(active_buffer, 0, 64);
  manager.Release(active_buffer, &active);
  manager.Release(idle_buffer, &idle);

  ASSERT_STATUS_OK(context.Flush(manager, idle));
  auto reused = manager.Create(idle, 64, usage);
  EXPECT_EQ(reused, idle_buffer);
  auto still_pending = manager.Create(idle, 64, usage);
  EXPECT_NE(still_pending, active_buffer);

  ASSERT_STATUS_OK(context.Flush(manager, active));
  auto after_submission = manager.Create(idle, 64, usage);
  EXPECT_EQ(after_submission, active_buffer);
  manager.Release(reused);
  manager.Release(still_pending);
  manager.Release(after_submission);
}

TEST(WebGpuContextTest, IndependentClearPreservesPendingAndCapturedUniformBuffers) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Graph,
                                webgpu::BufferCacheMode::GraphSimple, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState session_recording;
  webgpu::CommandRecordingState independent_recording;
  constexpr auto uniform_usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  constexpr auto storage_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                 wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  std::vector<webgpu::CapturedCommand> captured_commands;
  context.CaptureBegin(&captured_commands, manager, session_recording);
  auto uniform_buffer = manager.Create(session_recording, 64, uniform_usage);
  wgpu::Buffer retained_uniform{uniform_buffer};
  context.GetCommandEncoder(session_recording).ClearBuffer(uniform_buffer, 0, 64);
  manager.Release(uniform_buffer, &session_recording);

  const auto submit_independent_clear = [&]() {
    // Exercise the cached clear and immediate submission used by plugin streamless allocations.
    std::array<uint32_t, 16> dirty_data;
    dirty_data.fill(0xffffffffu);
    auto dirty_buffer = manager.Create(independent_recording, sizeof(dirty_data), storage_usage);
    manager.Upload(independent_recording, dirty_data.data(), dirty_buffer, sizeof(dirty_data));
    manager.Release(dirty_buffer, &independent_recording);
    auto cleared_buffer = wgpu::Buffer::Acquire(
        manager.Create(independent_recording, sizeof(dirty_data), storage_usage, true, true));
    EXPECT_EQ(cleared_buffer.Get(), dirty_buffer);
    const std::array<uint32_t, 16> zeros{};
    EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, cleared_buffer.Get()), zeros);
    manager.Release(cleared_buffer.MoveToCHandle(), &independent_recording);
  };

  submit_independent_clear();
  auto before_submission = wgpu::Buffer::Acquire(manager.Create(independent_recording, 64, uniform_usage));
  EXPECT_NE(before_submission.Get(), uniform_buffer);
  EXPECT_TRUE(session_recording.has_unsubmitted_work.load(std::memory_order_relaxed));

  ASSERT_STATUS_OK(context.Flush(manager, session_recording));
  context.CaptureEnd(session_recording);
  submit_independent_clear();
  auto after_submission = wgpu::Buffer::Acquire(manager.Create(independent_recording, 64, uniform_usage));
  EXPECT_NE(after_submission.Get(), uniform_buffer);
}

TEST(WebGpuContextTest, GraphFlushRetiresSharedBuffersForSubmittedRecording) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kStorageBufferCacheMode, "bucket"));
    ASSERT_STATUS_OK(options.AddConfigEntry(kUniformBufferCacheMode, "simple"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    webgpu::BufferManager graph_manager(context, webgpu::BufferCacheMode::Graph,
                                        webgpu::BufferCacheMode::GraphSimple, webgpu::BufferCacheMode::Disabled,
                                        webgpu::BufferCacheMode::Disabled);

    // Exercise both the Bucket and Simple shared caches during capture and replay.
    for (auto state : {webgpu::GraphCaptureState::Capturing, webgpu::GraphCaptureState::Replaying}) {
      for (auto usage : {wgpu::BufferUsage::Storage, wgpu::BufferUsage::Uniform}) {
        SCOPED_TRACE(static_cast<int>(state));
        SCOPED_TRACE(static_cast<uint64_t>(usage));
        usage |= wgpu::BufferUsage::CopyDst;
        webgpu::CommandRecordingState session;
        webgpu::CommandRecordingState other;
        webgpu::CommandRecordingState idle;
        session.graph_capture_state = state;
        auto released = shared_manager.Create(idle, 64, usage);
        auto other_released = shared_manager.Create(idle, 64, usage);
        wgpu::Buffer retained{released};
        wgpu::Buffer other_retained{other_released};
        context.GetCommandEncoder(session).ClearBuffer(released, 0, 64);
        context.GetCommandEncoder(other).ClearBuffer(other_released, 0, 64);
        shared_manager.Release(released, &session);
        shared_manager.Release(other_released, &other);

        ASSERT_STATUS_OK(context.Flush(graph_manager, idle));
        auto before_submission = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_NE(before_submission.Get(), released);
        EXPECT_NE(before_submission.Get(), other_released);

        ASSERT_STATUS_OK(context.Flush(graph_manager, session));
        EXPECT_EQ(session.graph_capture_state, state);
        auto reused = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_EQ(reused.Get(), released);
        auto still_pending = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_NE(still_pending.Get(), other_released);

        ASSERT_STATUS_OK(context.Flush(shared_manager, other));
        auto other_reused = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_EQ(other_reused.Get(), other_released);
      }
    }
  });
}

TEST(WebGpuContextTest, InitializerFlushRetiresSharedUniformBuffers) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kUniformBufferCacheMode, "simple"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    constexpr auto usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;

    // Prepacking flushes the initializer manager, while uniforms use the shared manager.
    // An empty flush must also promote idle buffers from SimpleCacheManager's pending list.
    for (bool encode_commands : {true, false}) {
      SCOPED_TRACE(encode_commands);
      webgpu::CommandRecordingState recording;
      auto released = shared_manager.Create(recording, 64, usage);
      wgpu::Buffer retained{released};
      if (encode_commands) {
        context.GetCommandEncoder(recording).ClearBuffer(released, 0, 64);
      }
      shared_manager.Release(released, &recording);
      ASSERT_STATUS_OK(context.Flush(context.InitializerBufferManager(), recording));
      auto reused = wgpu::Buffer::Acquire(shared_manager.Create(recording, 64, usage));
      EXPECT_EQ(reused.Get(), released);
    }
  });
}

TEST(WebGpuContextTest, FailedFlushDiscardsPendingBuffersFromBothManagers) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kStorageBufferCacheMode, "bucket"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                  webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                  webgpu::BufferCacheMode::Disabled);
    webgpu::CommandRecordingState recording;
    constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    auto released = manager.Create(recording, 64, usage);
    auto shared_released = shared_manager.Create(recording, 64, usage);
    wgpu::Buffer retained{released};
    wgpu::Buffer shared_retained{shared_released};
    webgpu::DispatchCommandInfo dispatch;
    dispatch.program_key = "missing_pipeline_for_failed_flush_test";
    recording.deferred_dispatches.push_back(std::move(dispatch));
    recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
    manager.Release(released, &recording);
    shared_manager.Release(shared_released, &recording);

    EXPECT_FALSE(context.Flush(manager, recording).IsOK());
    EXPECT_EQ(recording.command_encoder, nullptr);
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    // Reusing the recording after failure must not revive discarded buffers in either manager.
    ASSERT_STATUS_OK(context.Flush(manager, recording));
    auto replacement = wgpu::Buffer::Acquire(manager.Create(recording, 64, usage));
    auto shared_replacement = wgpu::Buffer::Acquire(shared_manager.Create(recording, 64, usage));
    EXPECT_NE(replacement.Get(), released);
    EXPECT_NE(shared_replacement.Get(), shared_released);
  });
}

TEST(WebGpuContextTest, AbandonedRecordingDoesNotReturnBuffersToPool) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = manager.Create(recording, 64, usage);
  // Keep the handle alive so a new device allocation cannot reuse its address.
  wgpu::Buffer retained{buffer};
  context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
  manager.Release(buffer, &recording);
  recording.command_encoder = nullptr;
  manager.DiscardPendingBuffers(recording);

  // Reuse the recording's address after abandoning its previous commands.
  context.GetCommandEncoder(recording);
  ASSERT_STATUS_OK(context.Flush(manager, recording));
  auto replacement = manager.Create(recording, 64, usage);
  EXPECT_NE(replacement, buffer);
  manager.Release(replacement);
}

TEST(WebGpuContextTest, SessionAllocatorSubmitsReusedBufferClearOutsideRun) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Bucket,
                                       webgpu::BufferCacheMode::Simple,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  webgpu::GpuBufferAllocator allocator(
      webgpu_ep->GetDeviceId(),
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);

  const auto downloaded_data = ReadBufferWithExternalCommandEncoder(context, reused_buffer);
  ASSERT_STATUS_OK(context.Flush(buffer_manager, webgpu_ep->Recording()));
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(downloaded_data, expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, ReplaysCopyOnlyGraphAndRetainsBuffersUntilRelease) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  std::array<uint32_t, 3> values{1, 2, 3};
  auto source = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  auto intermediate = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  auto destination = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  manager.Upload(recording, values.data(), source.Get(), sizeof(values));

  std::vector<webgpu::CapturedCommand> commands;
  context.CaptureBegin(&commands, manager, recording);
  manager.MemCpy(recording, source.Get(), intermediate.Get(), sizeof(values));
  manager.MemCpy(recording, intermediate.Get(), destination.Get(), sizeof(values));
  ASSERT_STATUS_OK(context.Flush(manager, recording));
  context.CaptureEnd(recording);
  ASSERT_EQ(commands.size(), 2u);
  EXPECT_EQ(recording.num_pending_dispatches, 0u);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<3>(context, destination.Get()), values);

  // Only the captured copies now own the intermediate; replay must keep it alive.
  intermediate = nullptr;
  values = {4, 5, 6};
  manager.Upload(recording, values.data(), source.Get(), sizeof(values));
  context.Replay(commands, manager, recording);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<3>(context, destination.Get()), values);
  EXPECT_EQ(recording.num_pending_dispatches, 0u);
  context.ReleaseGraphResources(commands);
  EXPECT_TRUE(commands.empty());
}

TEST(WebGpuContextTest, ReplaysPartialClearBetweenCopies) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  std::array<uint32_t, 3> values{1, 2, 3};
  auto source = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  auto intermediate = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  auto destination = wgpu::Buffer::Acquire(manager.Create(recording, sizeof(values), usage));
  manager.Upload(recording, values.data(), source.Get(), sizeof(values));

  std::vector<webgpu::CapturedCommand> commands;
  context.CaptureBegin(&commands, manager, recording);
  manager.MemCpy(recording, source.Get(), intermediate.Get(), sizeof(values));
  context.ClearBuffer(intermediate.Get(), 2 * sizeof(uint32_t), recording);
  manager.MemCpy(recording, intermediate.Get(), destination.Get(), sizeof(values));
  ASSERT_STATUS_OK(context.Flush(manager, recording));
  context.CaptureEnd(recording);
  ASSERT_EQ(commands.size(), 3u);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<3>(context, destination.Get()),
            (std::array<uint32_t, 3>{0, 0, values[2]}));

  for (uint32_t run = 1; run <= 3; ++run) {
    values.fill(run);
    manager.Upload(recording, values.data(), source.Get(), sizeof(values));
    context.Replay(commands, manager, recording);
    EXPECT_EQ(ReadBufferWithExternalCommandEncoder<3>(context, destination.Get()),
              (std::array<uint32_t, 3>{0, 0, values[2]}));
    EXPECT_EQ(recording.num_pending_dispatches, 0u);
  }
  context.ReleaseGraphResources(commands);
}

TEST(WebGpuContextTest, DoesNotCaptureDeviceAllocatorBufferClear) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Graph,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  std::vector<webgpu::CapturedCommand> captured_commands;
  webgpu::GpuBufferAllocator allocator(
      webgpu_ep->GetDeviceId(),
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false);

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  context.CaptureBegin(&captured_commands, buffer_manager, webgpu_ep->Recording());
  allocation = allocator.Alloc(sizeof(nonzero_data));
  if (allocation == nullptr) {
    context.CaptureEnd(webgpu_ep->Recording());
    FAIL() << "Failed to reacquire a device allocation during graph capture.";
  }
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);
  const Status flush_status = context.Flush(buffer_manager, webgpu_ep->Recording());
  context.CaptureEnd(webgpu_ep->Recording());
  if (!flush_status.IsOK()) {
    allocator.Free(allocation);
    FAIL() << flush_status.ErrorMessage();
  }

  EXPECT_TRUE(captured_commands.empty());

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, SessionAllocatorDefersReusedBufferClearDuringRun) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Bucket,
                                       webgpu::BufferCacheMode::Simple,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  webgpu::GpuBufferAllocator allocator(
      webgpu_ep->GetDeviceId(),
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  ASSERT_STATUS_OK(context.Flush(buffer_manager, webgpu_ep->Recording()));
  allocator.Free(allocation);

  RunOptions run_options;
  ASSERT_STATUS_OK(webgpu_ep->OnRunStart(run_options));
  allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), nonzero_data);

  ASSERT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, WebGpuExecutionProviderTracksRunActivity) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());
  RunOptions run_options;

  EXPECT_FALSE(webgpu_ep->IsRunActive());
  ASSERT_STATUS_OK(webgpu_ep->OnRunStart(run_options));
  EXPECT_TRUE(webgpu_ep->IsRunActive());
  ASSERT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
  EXPECT_FALSE(webgpu_ep->IsRunActive());
}

TEST(WebGpuContextTest, EnablesImplicitDeviceSynchronization) {
#if defined(__wasm__)
  GTEST_SKIP() << "ImplicitDeviceSynchronization is a Dawn native feature.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  EXPECT_TRUE(webgpu::WebGpuContextFactory::GetContext(0).DeviceHasFeature(
      wgpu::FeatureName::ImplicitDeviceSynchronization));
#endif
}

TEST(WebGpuContextTest, ExternalDeviceRequiresImplicitDeviceSynchronization) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  // Initialize ORT's Dawn proc table before using externally created devices.
  ConfigOptions options;
  auto owned_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(owned_ep, nullptr);

  dawn::native::Instance instance;
  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = static_cast<wgpu::BackendType>(webgpu::WebGpuContextConfig{}.backend_type);

  for (bool enable_synchronization : {false, true}) {
    SCOPED_TRACE(enable_synchronization);
    auto adapters = instance.EnumerateAdapters(&adapter_options);
    ASSERT_FALSE(adapters.empty());
    const auto feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
    wgpu::DeviceDescriptor device_desc{};
    device_desc.requiredFeatureCount = enable_synchronization ? 1 : 0;
    device_desc.requiredFeatures = enable_synchronization ? &feature : nullptr;
    auto device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_desc));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device.HasFeature(feature), enable_synchronization);

    ConfigOptions external_options;
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kDeviceId, "1"));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(instance.Get())).c_str()));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));

    if (enable_synchronization) {
      auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
      ASSERT_NE(external_ep, nullptr);
      EXPECT_TRUE(webgpu::WebGpuContextFactory::GetContext(1).DeviceHasFeature(feature));
    } else {
      EXPECT_THAT([&]() { WebGpuProviderFactoryCreator::Create(external_options); },
                  ::testing::ThrowsMessage<OnnxRuntimeException>(::testing::HasSubstr(
                      "an externally supplied native device must enable ImplicitDeviceSynchronization "
                      "in DeviceDescriptor.requiredFeatures when it is created.")));
    }
  }
#endif
}

TEST(WebGpuContextTest, EnablesLazyClearResourceOnFirstUse) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);

  EXPECT_TRUE(DeviceToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0),
                                    "lazy_clear_resource_on_first_use"));
#endif
}

TEST(WebGpuContextTest, EnableRobustnessControlsDawnToggle) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto enabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    ASSERT_NE(enabled_ep, nullptr);
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
    enabled_ep.reset();

    auto disabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
    ASSERT_NE(disabled_ep, nullptr);
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
  });
#endif
}

TEST(WebGpuContextTest, EnableRobustnessUsesBuildDefault) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(ep, nullptr);
#ifdef NDEBUG
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#else
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#endif
  });
#endif
}

TEST(WebGpuContextTest, EnableRobustnessRejectsInvalidValue) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(RobustnessOptions("true")), OnnxRuntimeException);
}

TEST(WebGpuContextTest, DisableSubgroupMatrixRejectsInvalidValue) {
  for (const char* value : {"true", "false", "2", "", "1x"}) {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kDisableSubgroupMatrix, value));
    EXPECT_THROW(WebGpuProviderFactoryCreator::Create(options), OnnxRuntimeException);
  }
}

TEST(WebGpuContextTest, DisableSubgroupMatrixControlsDeviceFeature) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kDisableSubgroupMatrix, "1"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    const auto feature = wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix;
    EXPECT_FALSE(context.Device().HasFeature(feature));
    EXPECT_FALSE(context.DeviceHasFeature(feature));
#if !defined(__wasm__)
    EXPECT_EQ(context.SubgroupMatrixConfigs().configCount, 0u);
#endif
    auto matching_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    // Omitted or conflicting values on a shared context warn and keep the feature disabled.
    auto omitted_ep = WebGpuProviderFactoryCreator::Create(ConfigOptions{})->CreateProvider();
    ASSERT_STATUS_OK(options.AddConfigEntry(kDisableSubgroupMatrix, "0"));
    auto conflicting_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    EXPECT_FALSE(context.DeviceHasFeature(feature));
  });
}

TEST(WebGpuContextTest, DisableSubgroupMatrixMasksExternalDeviceFeature) {
  ConfigOptions default_options;
  auto default_ep = WebGpuProviderFactoryCreator::Create(default_options)->CreateProvider();
  if (!webgpu::WebGpuContextFactory::GetContext(0).Device().HasFeature(
          wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    GTEST_SKIP() << "The adapter does not support subgroup matrices.";
  }
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kDisableSubgroupMatrix, "0"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    const auto feature = wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix;
    EXPECT_EQ(context.DeviceHasFeature(feature), context.Device().HasFeature(feature));
    ASSERT_TRUE(context.Device().HasFeature(feature));

    ConfigOptions external_options;
    ASSERT_STATUS_OK(external_options.AddConfigEntry(kDeviceId, "1"));
    ASSERT_STATUS_OK(external_options.AddConfigEntry(kDisableSubgroupMatrix, "1"));
    ASSERT_STATUS_OK(external_options.AddConfigEntry(
        kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(context.Instance().Get())).c_str()));
    ASSERT_STATUS_OK(external_options.AddConfigEntry(
        kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(context.Device().Get())).c_str()));
    auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
    auto& external_context = webgpu::WebGpuContextFactory::GetContext(1);
    EXPECT_TRUE(external_context.Device().HasFeature(feature));
    EXPECT_FALSE(external_context.DeviceHasFeature(feature));
#if !defined(__wasm__)
    EXPECT_EQ(external_context.SubgroupMatrixConfigs().configCount, 0u);
#endif
    EXPECT_TRUE(context.DeviceHasFeature(feature));
  });
}

TEST(WebGpuContextTest, KvCacheQuantizationAcceptsSupportedBitWidths) {
  for (const auto& [value, expected_bits] :
       std::array<std::pair<const char*, uint32_t>, 3>{{{"0", 0}, {"4", 4}, {"8", 8}}}) {
    auto ep = WebGpuProviderFactoryCreator::Create(KvCacheQuantizationOptions(value))->CreateProvider();
    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(static_cast<WebGpuExecutionProvider*>(ep.get())->KvCacheQuantizationBits(), expected_bits);
  }
}

TEST(WebGpuContextTest, KvCacheQuantizationRejectsInvalidValue) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(KvCacheQuantizationOptions("3")), OnnxRuntimeException);
}

TEST(WebGpuContextTest, DeviceIdRejectsInvalidValueBeforeContextCreation) {
  for (const char* value : {"-1", "-32768", "32768", "2147483647", "2147483648", "", "1x", "0x", "x"}) {
    SCOPED_TRACE(value);
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kDeviceId, value));
    try {
      WebGpuProviderFactoryCreator::Create(options);
      FAIL() << "Expected deviceId to be rejected.";
    } catch (const OnnxRuntimeException& ex) {
      EXPECT_NE(std::string_view{ex.what()}.find("Invalid deviceId value"), std::string_view::npos);
    }
  }
}

TEST(WebGpuContextTest, AdapterIndexRejectsInvalidValue) {
  for (const char* value : {"-1", "1x"}) {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, value));
    EXPECT_THROW(WebGpuProviderFactoryCreator::Create(options), OnnxRuntimeException);
  }
}

TEST(WebGpuContextTest, AdapterIndexAcceptsNonNegativeInteger) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  },
                             /*compile_only_parent=*/true);
#endif
}

TEST(WebGpuContextTest, AdapterIndexSelectsPhysicalAdapter) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_NE(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  });
#endif
}

TEST(WebGpuContextTest, AdapterIndexRejectsOutOfRangeValue) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  ConfigOptions options;
  const std::string adapter_index = std::to_string(std::numeric_limits<uint32_t>::max());
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, adapter_index.c_str()));

  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(options), OnnxRuntimeException);
#endif
}

TEST(WebGpuContextTest, AdapterIndexRejectsUnsupportedBuild) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "This build supports physical adapter enumeration.";
#else
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

  try {
    WebGpuProviderFactoryCreator::Create(options);
    FAIL() << "Expected adapterIndex to be rejected by this build.";
  } catch (const OnnxRuntimeException& ex) {
    EXPECT_NE(std::string_view{ex.what()}.find("requires a native Dawn build"), std::string_view::npos);
  }
#endif
}

TEST(WebGpuContextTest, WeightLoadAccelerationRejectsBooleanAndUnknownModes) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(
                   WeightLoadAccelerationOptions("1")),
               OnnxRuntimeException);
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(
                   WeightLoadAccelerationOptions("automatic")),
               OnnxRuntimeException);
}

TEST(WebGpuContextTest, WeightLoadAccelerationOffDoesNotRequireDeviceSupport) {
  auto ep = WebGpuProviderFactoryCreator::Create(
                CompileOnlyWeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Off))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
}

TEST(WebGpuContextTest, PreferredWeightLoadAccelerationFallsBackWithoutDeviceSupport) {
  auto ep = WebGpuProviderFactoryCreator::Create(
                CompileOnlyWeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Preferred))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
}

TEST(WebGpuContextTest, RequiredWeightLoadAccelerationDoesNotExposeLoaderWithoutDevice) {
  RunWithFreshDefaultContext([]() {
    auto factory = WebGpuProviderFactoryCreator::Create(
        CompileOnlyWeightLoadAccelerationOptions(
            kWeightLoadAcceleration_Required));
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
    auto ep = factory->CreateProvider();
    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(ep->GetExternalDataLoader(), nullptr);
#else
    EXPECT_THROW(factory->CreateProvider(), OnnxRuntimeException);
#endif
  },
                             true);
}

TEST(WebGpuContextTest, PreferredWeightLoadAccelerationIsDisabledWithEnvironmentAllocators) {
  auto ep = WebGpuProviderFactoryCreator::Create(
                EnvironmentAllocatorWeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Preferred))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
  EXPECT_EQ(ep->GetExternalDataLoader(), nullptr);
}

TEST(WebGpuContextTest, RequiredWeightLoadAccelerationRejectsEnvironmentAllocators) {
  EXPECT_THROW(
      WebGpuProviderFactoryCreator::Create(
          EnvironmentAllocatorWeightLoadAccelerationOptions(
              kWeightLoadAcceleration_Required)),
      OnnxRuntimeException);
}

TEST(WebGpuContextTest, DisabledWeightLoadAccelerationAllowsEnvironmentAllocators) {
  auto ep = WebGpuProviderFactoryCreator::Create(
                EnvironmentAllocatorWeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Off))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
  EXPECT_EQ(ep->GetExternalDataLoader(), nullptr);
}

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
void ExpectD3D12BufferContents(
    ID3D12Device* device,
    const windows::d3d12::D3D12FileBufferLoader::BufferCollection& batch,
    const std::vector<std::vector<uint8_t>>& expected_buffers) {
  ASSERT_EQ(batch.buffers.size(), expected_buffers.size());

  uint64_t total_size = 0;
  for (size_t index = 0; index < expected_buffers.size(); ++index) {
    ASSERT_NE(batch.buffers[index].Get(), nullptr);
    ASSERT_LE(expected_buffers[index].size(), batch.buffers[index]->GetDesc().Width);
    total_size += expected_buffers[index].size();
  }
  ASSERT_GT(total_size, 0u);

  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC queue_desc{};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
  ASSERT_TRUE(SUCCEEDED(device->CreateCommandQueue(
      &queue_desc, IID_PPV_ARGS(&queue))));

  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
  ASSERT_TRUE(SUCCEEDED(device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator))));
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list;
  ASSERT_TRUE(SUCCEEDED(device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr,
      IID_PPV_ARGS(&command_list))));

  D3D12_HEAP_PROPERTIES readback_heap{};
  readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC readback_desc{};
  readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  readback_desc.Width = total_size;
  readback_desc.Height = 1;
  readback_desc.DepthOrArraySize = 1;
  readback_desc.MipLevels = 1;
  readback_desc.SampleDesc.Count = 1;
  readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  Microsoft::WRL::ComPtr<ID3D12Resource> readback;
  ASSERT_TRUE(SUCCEEDED(device->CreateCommittedResource(
      &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&readback))));

  uint64_t destination_offset = 0;
  for (size_t index = 0; index < expected_buffers.size(); ++index) {
    command_list->CopyBufferRegion(
        readback.Get(), destination_offset,
        batch.buffers[index].Get(), 0,
        expected_buffers[index].size());
    destination_offset += expected_buffers[index].size();
  }
  ASSERT_TRUE(SUCCEEDED(command_list->Close()));
  ID3D12CommandList* command_lists[]{command_list.Get()};
  queue->ExecuteCommandLists(1, command_lists);

  Microsoft::WRL::ComPtr<ID3D12Fence> fence;
  ASSERT_TRUE(SUCCEEDED(device->CreateFence(
      0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
  ASSERT_TRUE(SUCCEEDED(queue->Signal(fence.Get(), 1)));
  if (fence->GetCompletedValue() < 1) {
    HANDLE event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(event_handle, nullptr);
    ASSERT_TRUE(SUCCEEDED(
        fence->SetEventOnCompletion(1, event_handle)));
    EXPECT_EQ(WaitForSingleObject(event_handle, INFINITE), WAIT_OBJECT_0);
    CloseHandle(event_handle);
  }

  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(total_size)};
  ASSERT_TRUE(SUCCEEDED(readback->Map(0, &read_range, &mapped)));
  size_t source_offset = 0;
  for (const auto& expected : expected_buffers) {
    EXPECT_EQ(
        std::memcmp(
            static_cast<const uint8_t*>(mapped) + source_offset,
            expected.data(), expected.size()),
        0);
    source_offset += expected.size();
  }
  const D3D12_RANGE written_range{0, 0};
  readback->Unmap(0, &written_range);
}

TEST(WebGpuContextTest, D3D12AcceleratedCanBeEnabledAfterInitialOffSession) {
  RunWithFreshDefaultContext([]() {
    auto off_factory = WebGpuProviderFactoryCreator::Create(
        WeightLoadAccelerationOptions(kWeightLoadAcceleration_Off));
    auto off_ep = off_factory->CreateProvider();
    ASSERT_NE(off_ep, nullptr);

    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    if (context.SelectedBackendType() != wgpu::BackendType::D3D12 ||
        !context.D3D12SharedResourceFeaturesAvailable()) {
      GTEST_SKIP() << "The selected adapter does not support D3D12 weight loading.";
    }
    ASSERT_NE(context.WeightLoadingD3D12Device(), nullptr);
    auto* native_device = context.WeightLoadingD3D12Device();
    auto* webgpu_device = context.Device().Get();
    EXPECT_EQ(off_ep->GetExternalDataLoader(), nullptr);

    auto required_factory = WebGpuProviderFactoryCreator::Create(
        WeightLoadAccelerationOptions(kWeightLoadAcceleration_Required));
    auto required_ep = required_factory->CreateProvider();
    ASSERT_NE(required_ep, nullptr);

    const auto support_status =
        webgpu::CheckD3D12AcceleratedExternalWeightsSupport(context);
    ASSERT_STATUS_OK(support_status);
    EXPECT_EQ(context.Device().Get(), webgpu_device);
    EXPECT_EQ(context.WeightLoadingD3D12Device(), native_device);
    EXPECT_TRUE(context.DeviceHasFeature(
        wgpu::FeatureName::SharedBufferMemoryD3D12Resource));
    EXPECT_TRUE(context.DeviceHasFeature(
        wgpu::FeatureName::SharedFenceDXGISharedHandle));

    auto loader = required_ep->GetExternalDataLoader();
    ASSERT_NE(loader, nullptr);
    EXPECT_STATUS_OK(loader->BeginLoad());
    loader->EndLoad();

    auto preferred_ep = WebGpuProviderFactoryCreator::Create(
                            WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred))
                            ->CreateProvider();
    ASSERT_NE(preferred_ep, nullptr);
    auto preferred_loader = preferred_ep->GetExternalDataLoader();
    ASSERT_NE(preferred_loader, nullptr);
    EXPECT_STATUS_OK(preferred_loader->BeginLoad());
    preferred_loader->EndLoad();
    EXPECT_EQ(context.Device().Get(), webgpu_device);
    EXPECT_EQ(context.WeightLoadingD3D12Device(), native_device);
  });
}

TEST(WebGpuContextTest, D3D12FileLoaderReusesUploadSlotAcrossChunks) {
  constexpr uint64_t kSlotSize = 64 * 1024;
  constexpr size_t kDataSize = 3 * kSlotSize;
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_upload_slot_reuse_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");

  std::vector<uint8_t> expected(kDataSize);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<uint8_t>((index * 37) & 0xff);
  }
  {
    std::ofstream stream{data_path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(expected.size()));
    ASSERT_TRUE(stream.good());
  }

  auto ep = WebGpuProviderFactoryCreator::Create(
                WeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Required))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto support_status =
      webgpu::CheckD3D12AcceleratedExternalWeightsSupport(context);
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  windows::d3d12::D3D12FileBufferLoader::Config config;
  config.upload_slot_count = 2;
  config.upload_slot_size = kSlotSize;
  std::unique_ptr<windows::d3d12::D3D12FileBufferLoader> loader;
  ASSERT_STATUS_OK(windows::d3d12::D3D12FileBufferLoader::Create(
      context.WeightLoadingD3D12Device(), loader, config));

  windows::d3d12::D3D12FileBufferLoader::BufferCollection batch;
  ASSERT_STATUS_OK(loader->Load(
      {{data_path.native(), 0, kDataSize}}, batch));
  ASSERT_EQ(batch.buffers.size(), 1u);

  auto* device = context.WeightLoadingD3D12Device();
  ExpectD3D12BufferContents(device, batch, {expected});
}

TEST(WebGpuContextTest, D3D12FileLoaderUsesMultipleHeapsAndCommittedResources) {
  constexpr uint64_t kHeapSize = 64 * 1024;
  const std::vector<std::vector<uint8_t>> expected_buffers{
      std::vector<uint8_t>(1, 0x19),
      std::vector<uint8_t>(17, 0x5a),
      std::vector<uint8_t>(kHeapSize + 1, 0xc3),
  };
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_destination_allocation_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");

  std::vector<windows::d3d12::D3D12FileBufferLoader::BufferSource> ranges;
  uint64_t file_offset = 0;
  {
    std::ofstream stream{data_path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    for (const auto& expected : expected_buffers) {
      ranges.push_back(
          {data_path.native(), file_offset, expected.size()});
      stream.write(
          reinterpret_cast<const char*>(expected.data()),
          static_cast<std::streamsize>(expected.size()));
      file_offset += expected.size();
    }
    ASSERT_TRUE(stream.good());
  }

  auto ep = WebGpuProviderFactoryCreator::Create(
                WeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Required))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto support_status =
      webgpu::CheckD3D12AcceleratedExternalWeightsSupport(context);
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  windows::d3d12::D3D12FileBufferLoader::Config config;
  config.max_destination_heap_size = kHeapSize;
  std::unique_ptr<windows::d3d12::D3D12FileBufferLoader> loader;
  ASSERT_STATUS_OK(windows::d3d12::D3D12FileBufferLoader::Create(
      context.WeightLoadingD3D12Device(), loader, config));

  windows::d3d12::D3D12FileBufferLoader::BufferCollection batch;
  ASSERT_STATUS_OK(loader->Load(ranges, batch));
  ASSERT_EQ(batch.buffers.size(), expected_buffers.size());
  EXPECT_EQ(batch.heaps.size(), 2u);
  ExpectD3D12BufferContents(
      context.WeightLoadingD3D12Device(), batch, expected_buffers);
}

TEST(WebGpuContextTest, D3D12AcceleratedAllocatorUsesProviderRecording) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_allocator_recording_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 4> expected{11, 22, 33, 44};
  {
    std::ofstream stream{data_path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(sizeof(expected)));
    ASSERT_TRUE(stream.good());
  }

  auto ep = WebGpuProviderFactoryCreator::Create(
                WeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Required))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto support_status =
      webgpu::CheckD3D12AcceleratedExternalWeightsSupport(context);
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto& allocator = allocators.front();
  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", 0, sizeof(expected)));
  ASSERT_STATUS_OK(loader->CommitLoadCandidates([]() { return false; }));

  auto& webgpu_ep = *static_cast<WebGpuExecutionProvider*>(ep.get());
  auto& recording = webgpu_ep.Recording();
  wgpu::BufferDescriptor destination_description{};
  destination_description.size = sizeof(expected);
  destination_description.usage =
      wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto destination = context.Device().CreateBuffer(&destination_description);
  ASSERT_TRUE(destination);
  {
    Tensor weights{DataTypeImpl::GetType<uint32_t>(), TensorShape({4}),
                   nullptr, allocator};
    ASSERT_STATUS_OK(loader->LoadTensor(
        Env::Default(), data_path, "weights", 0, sizeof(expected),
        allocator, weights));
    wgpu::Buffer source{
        reinterpret_cast<WGPUBuffer>(weights.MutableDataRaw())};
    context.GetCommandEncoder(recording).CopyBufferToBuffer(
        source, 0, destination, 0, sizeof(expected));
  }
  EXPECT_TRUE(recording.has_unsubmitted_work.load());
  ASSERT_EQ(recording.pending_release_callbacks.size(), 1u);

  auto& buffer_manager = webgpu_ep.InitializerBufferManager();
  webgpu::CommandRecordingState unrelated_recording;
  ASSERT_STATUS_OK(context.Flush(buffer_manager, unrelated_recording));
  EXPECT_TRUE(recording.has_unsubmitted_work.load());
  EXPECT_EQ(recording.pending_release_callbacks.size(), 1u);

  ASSERT_STATUS_OK(context.Flush(buffer_manager, recording));
  EXPECT_FALSE(recording.has_unsubmitted_work.load());
  EXPECT_TRUE(recording.pending_release_callbacks.empty());
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<4>(context, destination.Get()),
            expected);
}

TEST(WebGpuContextTest, DuplicateProviderDoesNotRegisterD3D12AcceleratedLoader) {
  InferenceSession session{SessionOptions{}, GetEnvironment()};
  auto off_provider = WebGpuProviderFactoryCreator::Create(
                          WeightLoadAccelerationOptions(
                              kWeightLoadAcceleration_Off))
                          ->CreateProvider();
  ASSERT_NE(off_provider, nullptr);
  auto allocators = off_provider->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto memory_info = allocators.front()->Info();
  ASSERT_STATUS_OK(
      session.RegisterExecutionProvider(std::move(off_provider)));

  auto accelerated_provider =
      WebGpuProviderFactoryCreator::Create(
          WeightLoadAccelerationOptions(kWeightLoadAcceleration_Required))
          ->CreateProvider();
  ASSERT_NE(accelerated_provider, nullptr);
  const auto status =
      session.RegisterExecutionProvider(std::move(accelerated_provider));
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("already been registered"),
            std::string::npos);
  EXPECT_EQ(session.GetExternalDataLoaderManager().GetExternalDataLoader(
                memory_info),
            nullptr);
}

TEST(WebGpuContextTest, D3D12AcceleratedLoadsExternalTensorsAcrossFilesAndRanges) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_external_tensor_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const auto other_data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("other_weights.bin");
  constexpr size_t kDataOffset = 32;
  constexpr size_t kSeparatedDataOffset = 128;
  const std::array<uint32_t, 15> expected{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  const std::array<uint32_t, 4> separated_expected{101, 102, 103, 104};
  const std::array<uint32_t, 3> other_expected{201, 202, 203};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    const std::array<char, kDataOffset> prefix{};
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    stream.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(sizeof(expected)));
    stream.seekp(kSeparatedDataOffset);
    stream.write(reinterpret_cast<const char*>(separated_expected.data()),
                 static_cast<std::streamsize>(sizeof(separated_expected)));
    ASSERT_TRUE(stream.good());
  }
  {
    std::ofstream stream{other_data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    const std::array<char, kDataOffset> prefix{};
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    stream.write(reinterpret_cast<const char*>(other_expected.data()),
                 static_cast<std::streamsize>(sizeof(other_expected)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Required);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  EXPECT_FALSE(loader->SupportsDataType(
      ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  EXPECT_TRUE(loader->SupportsDataType(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto& allocator = allocators.front();

  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", kDataOffset, sizeof(expected)));
  const auto duplicate_status = loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights",
      kDataOffset, sizeof(expected));
  EXPECT_FALSE(duplicate_status.IsOK());
  EXPECT_NE(duplicate_status.ErrorMessage().find(
                "Duplicate D3D12 accelerated initializer"),
            std::string::npos);
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "separated", kSeparatedDataOffset,
      sizeof(separated_expected)));
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), other_data_path, "other_weights", kDataOffset,
      sizeof(other_expected)));
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "empty",
      kSeparatedDataOffset + sizeof(separated_expected), 0));
  ASSERT_STATUS_OK(loader->CommitLoadCandidates([]() { return false; }));

  Tensor weights{DataTypeImpl::GetType<uint32_t>(), TensorShape({15}),
                 nullptr, allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), data_path, "weights", kDataOffset, sizeof(expected),
      allocator, weights));
  std::array<uint32_t, 16> aligned_expected{};
  std::copy(expected.begin(), expected.end(), aligned_expected.begin());
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<16>(
                webgpu::WebGpuContextFactory::GetContext(0),
                reinterpret_cast<WGPUBuffer>(weights.MutableDataRaw())),
            aligned_expected);

  Tensor separated_weights{DataTypeImpl::GetType<uint32_t>(), TensorShape({4}),
                           nullptr, allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), data_path, "separated", kSeparatedDataOffset,
      sizeof(separated_expected), allocator, separated_weights));
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<4>(
                webgpu::WebGpuContextFactory::GetContext(0),
                reinterpret_cast<WGPUBuffer>(
                    separated_weights.MutableDataRaw())),
            separated_expected);

  Tensor other_weights{DataTypeImpl::GetType<uint32_t>(), TensorShape({3}),
                       nullptr, allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), other_data_path, "other_weights", kDataOffset,
      sizeof(other_expected), allocator, other_weights));
  std::array<uint32_t, 4> aligned_other_expected{};
  std::copy(other_expected.begin(), other_expected.end(),
            aligned_other_expected.begin());
  aligned_other_expected.back() = other_expected.front();
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder<4>(
                webgpu::WebGpuContextFactory::GetContext(0),
                reinterpret_cast<WGPUBuffer>(other_weights.MutableDataRaw())),
            aligned_other_expected);

  Tensor empty{DataTypeImpl::GetType<uint32_t>(), TensorShape({0}), nullptr,
               allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), data_path, "empty",
      kSeparatedDataOffset + sizeof(separated_expected), 0, allocator, empty));
  EXPECT_EQ(empty.SizeInBytes(), 0u);
  EXPECT_EQ(empty.MutableDataRaw(), nullptr);
}

void RunD3D12AcceleratedExternalInitializerSessionTest(
    const char* mode, bool fail_late_registration = false) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_session_test")};
  const auto model_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("model.onnx");
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  constexpr size_t kDataOffset = 32;
  const std::array<float, 15> weights{
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
  {
    std::ofstream stream{data_path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    const std::array<char, kDataOffset> prefix{};
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    stream.write(reinterpret_cast<const char*>(weights.data()),
                 static_cast<std::streamsize>(sizeof(weights)));
    ASSERT_TRUE(stream.good());
  }

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.set_producer_name("onnxruntime-test");
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("webgpu_d3d12_accelerated_session");

  const auto set_tensor_type = [&weights](
                                   ONNX_NAMESPACE::ValueInfoProto* value_info,
                                   const char* name) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(weights.size());
  };
  set_tensor_type(graph->add_input(), "X");
  set_tensor_type(graph->add_output(), "Y");

  auto* initializer = graph->add_initializer();
  initializer->set_name("W");
  initializer->set_data_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  initializer->add_dims(weights.size());
  initializer->set_data_location(
      ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* location = initializer->add_external_data();
  location->set_key("location");
  location->set_value("weights.bin");
  auto* offset = initializer->add_external_data();
  offset->set_key("offset");
  offset->set_value(std::to_string(kDataOffset));
  auto* length = initializer->add_external_data();
  length->set_key("length");
  length->set_value(std::to_string(sizeof(weights)));

  auto* add = graph->add_node();
  add->set_op_type("Add");
  add->add_input("X");
  add->add_input("W");
  add->add_output("Y");
  {
    std::ofstream stream{model_path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    ASSERT_TRUE(model.SerializeToOstream(&stream));
  }

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
      kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSession session{session_options, GetEnvironment()};
  auto provider = WebGpuProviderFactoryCreator::Create(
                      WeightLoadAccelerationOptions(mode))
                      ->CreateProvider();
  ASSERT_NE(provider, nullptr);
  if (fail_late_registration) {
    auto allocators = provider->CreatePreferredAllocators();
    ASSERT_FALSE(allocators.empty());
    const auto memory_info = allocators.front()->Info();
    ASSERT_STATUS_OK(session.Load(model_path.native()));
    ASSERT_TRUE(std::filesystem::remove(data_path));
    auto status =
        session.RegisterExecutionProvider(std::move(provider));
    if (status.IsOK()) {
      // Some provider configurations defer external initializer loading until
      // initialization instead of attempting it during late registration.
      status = session.Initialize();
    }
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(session.GetExternalDataLoaderManager().GetExternalDataLoader(
                  memory_info),
              nullptr);
    return;
  }
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  ASSERT_STATUS_OK(session.Load(model_path.native()));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<float> input(weights.size(), 1.0f);
  OrtValue input_value;
  CreateMLValue<float>(
      TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
      {static_cast<int64_t>(weights.size())}, input, &input_value);
  NameMLValMap feeds{{"X", input_value}};
  std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(
      session.Run(RunOptions{}, feeds, output_names, &fetches));
  ASSERT_EQ(fetches.size(), 1u);
  const auto& output = fetches[0].Get<Tensor>();
  ASSERT_EQ(output.Shape(), TensorShape({static_cast<int64_t>(weights.size())}));
  const auto* output_data = output.Data<float>();
  for (size_t index = 0; index < weights.size(); ++index) {
    EXPECT_FLOAT_EQ(output_data[index], input[index] + weights[index]);
  }
}

TEST(WebGpuContextTest, D3D12AcceleratedLoadsExternalInitializerThroughSession) {
  auto probe_provider = WebGpuProviderFactoryCreator::Create(
                            WeightLoadAccelerationOptions(
                                kWeightLoadAcceleration_Required))
                            ->CreateProvider();
  ASSERT_NE(probe_provider, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }
  RunD3D12AcceleratedExternalInitializerSessionTest(
      kWeightLoadAcceleration_Required);
}

TEST(WebGpuContextTest, LateRegistrationFailureRollsBackD3D12AcceleratedLoader) {
  auto probe_provider = WebGpuProviderFactoryCreator::Create(
                            WeightLoadAccelerationOptions(
                                kWeightLoadAcceleration_Off))
                            ->CreateProvider();
  ASSERT_NE(probe_provider, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }
  RunD3D12AcceleratedExternalInitializerSessionTest(
      kWeightLoadAcceleration_Required, true);
}

TEST(WebGpuContextTest, PreferredD3D12AcceleratedPreservesCancellation) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_cancellation_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", 0, sizeof(data)));
  const auto status = loader->CommitLoadCandidates([]() { return true; });
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  loader->EndLoad();
}

TEST(WebGpuContextTest, D3D12AcceleratedZeroRequestFinalBatchPreservesCancellation) {
  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  const auto status = loader->CommitLoadCandidates([]() { return true; });
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  loader->EndLoad();
}

TEST(WebGpuContextTest, D3D12AcceleratedChecksCancellationDuringImport) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_import_cancellation_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", 0, sizeof(data)));

  size_t cancellation_checks = 0;
  const auto status = loader->CommitLoadCandidates([&cancellation_checks]() {
    return cancellation_checks++ != 0;
  });
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  loader->EndLoad();
}

TEST(WebGpuContextTest, PreferredD3D12AcceleratedFallsBackAfterOperationalFailure) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_fallback_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto& allocator = allocators.front();
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", 0, sizeof(data)));
  ASSERT_TRUE(std::filesystem::remove(data_path));
  EXPECT_STATUS_OK(loader->CommitLoadCandidates([]() { return false; }));
  EXPECT_FALSE(loader->CanLoad(allocator->Info()));
}

TEST(WebGpuContextTest, PreferredD3D12AcceleratedPreservesConcurrentCancellation) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_d3d12_accelerated_failure_cancellation_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto support_status = webgpu::CheckD3D12AcceleratedExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->RegisterLoadCandidate(
      Env::Default(), data_path, "weights", 0, sizeof(data)));
  ASSERT_TRUE(std::filesystem::remove(data_path));
  const auto status = loader->CommitLoadCandidates([]() { return true; });
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  loader->EndLoad();
}

TEST(WebGpuContextTest, WeightLoadAccelerationModeResolution) {
  const auto unsupported =
      ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "D3D12Accelerated unavailable");
  bool enabled = true;

  EXPECT_STATUS_OK(webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::Off, unsupported, enabled));
  EXPECT_FALSE(enabled);

  enabled = true;
  EXPECT_STATUS_OK(webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::Preferred, unsupported, enabled));
  EXPECT_FALSE(enabled);

  enabled = true;
  const auto required_status = webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::Required, unsupported,
      enabled);
  EXPECT_FALSE(required_status.IsOK());
  EXPECT_FALSE(enabled);
  EXPECT_EQ(required_status.ErrorMessage(), unsupported.ErrorMessage());
}

#endif

TEST(WebGpuContextTest, AdapterIndexRejectsConflictingSelectorOnReusedContext) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions first_options;
    ORT_THROW_IF_ERROR(first_options.AddConfigEntry(kAdapterIndex, "0"));
    auto first_ep = WebGpuProviderFactoryCreator::Create(first_options)->CreateProvider();
    ASSERT_NE(first_ep, nullptr);

    ConfigOptions power_options;
    ORT_THROW_IF_ERROR(power_options.AddConfigEntry(kAdapterIndex, "0"));
    ORT_THROW_IF_ERROR(power_options.AddConfigEntry(kPowerPreference, kPowerPreference_LowPower));
    EXPECT_THROW(WebGpuProviderFactoryCreator::Create(power_options), OnnxRuntimeException);

    webgpu::WebGpuContextConfig backend_config;
    backend_config.adapter_index = 0;
    backend_config.backend_type = std::numeric_limits<int>::max();
    EXPECT_THROW(webgpu::WebGpuContextFactory::CreateContext(backend_config), OnnxRuntimeException);
  });
#endif
}

TEST(WebGpuContextTest, CompileOnlyContextDoesNotCreateDevice) {
  RunWithFreshDefaultContext([]() {
    auto options = RobustnessOptions("0");
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  },
                             /*compile_only_parent=*/true);
}

TEST(WebGpuContextTest, EnableRobustnessIsIndependentFromValidationMode) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto robust_options = RobustnessOptions("1");
    ORT_THROW_IF_ERROR(robust_options.AddConfigEntry(kValidationMode, kValidationMode_Disabled));
    auto robust_ep = WebGpuProviderFactoryCreator::Create(robust_options)->CreateProvider();
    ASSERT_NE(robust_ep, nullptr);
    const auto& robust_context = webgpu::WebGpuContextFactory::GetContext(0);
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(robust_context));
    EXPECT_TRUE(DeviceToggleIsEnabled(robust_context, "skip_validation"));
    robust_ep.reset();

    auto non_robust_options = RobustnessOptions("0");
    ORT_THROW_IF_ERROR(non_robust_options.AddConfigEntry(kValidationMode, kValidationMode_full));
    auto non_robust_ep = WebGpuProviderFactoryCreator::Create(non_robust_options)->CreateProvider();
    ASSERT_NE(non_robust_ep, nullptr);
    const auto& non_robust_context = webgpu::WebGpuContextFactory::GetContext(0);
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(non_robust_context));
    EXPECT_FALSE(DeviceToggleIsEnabled(non_robust_context, "skip_validation"));
  });
#endif
}

TEST(WebGpuContextTest, ConflictingExplicitValueWarnsAndKeepsFirstValue) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto first_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    ASSERT_NE(first_ep, nullptr);

    testing::internal::CaptureStderr();
    auto second_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
    const std::string warning = testing::internal::GetCapturedStderr();

    ASSERT_NE(second_ep, nullptr);
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
    EXPECT_NE(warning.find("already initialized"), std::string::npos);
    EXPECT_NE(warning.find("will be ignored"), std::string::npos);
  });
#endif
}

TEST(WebGpuContextTest, OmittedAndMatchingValuesDoNotWarn) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto first_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    ASSERT_NE(first_ep, nullptr);

    ConfigOptions omitted_options;
    testing::internal::CaptureStderr();
    auto omitted_ep = WebGpuProviderFactoryCreator::Create(omitted_options)->CreateProvider();
    auto matching_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    const std::string warning = testing::internal::GetCapturedStderr();

    ASSERT_NE(omitted_ep, nullptr);
    ASSERT_NE(matching_ep, nullptr);
    EXPECT_EQ(warning.find("enableRobustness"), std::string::npos);
  });
#endif
}

TEST(WebGpuContextTest, ExternalDeviceValueWarnsAndIsIgnored) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto owned_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
    ASSERT_NE(owned_ep, nullptr);
    const auto& owned_context = webgpu::WebGpuContextFactory::GetContext(0);

    ConfigOptions external_options;
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kDeviceId, "1"));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuInstance,
        std::to_string(reinterpret_cast<uintptr_t>(owned_context.Instance().Get())).c_str()));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuDevice,
        std::to_string(reinterpret_cast<uintptr_t>(owned_context.Device().Get())).c_str()));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kEnableRobustness, "1"));

    testing::internal::CaptureStderr();
    auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
    const std::string warning = testing::internal::GetCapturedStderr();

    ASSERT_NE(external_ep, nullptr);
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(1)));
    EXPECT_NE(warning.find("externally supplied WebGPU device"), std::string::npos);
    EXPECT_NE(warning.find("will be ignored"), std::string::npos);
  });
#endif
}

#if !defined(__wasm__)
TEST(WebGpuContextTest, CanMapDeviceLocalMemory) {
  using webgpu::detail::CanMapDeviceLocalMemory;

  constexpr uint64_t kMiB = 1024 * 1024;
  constexpr auto kDeviceLocal = wgpu::HeapProperty::DeviceLocal;
  constexpr auto kHostMappable = wgpu::HeapProperty::HostVisible | wgpu::HeapProperty::HostCoherent;
  const std::array<wgpu::MemoryHeapInfo, 3> small_bar{{
      {kDeviceLocal, 20224 * kMiB},
      {kHostMappable, 64353 * kMiB},
      {kDeviceLocal | kHostMappable, 256 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 2> resizable_bar{{
      {kDeviceLocal | kHostMappable, 12216 * kMiB},
      {kHostMappable, 23781 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 1> unified_memory{{
      {kDeviceLocal | kHostMappable, 5461 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 2> larger_mappable_heap{{
      {kDeviceLocal, 256 * kMiB},
      {kDeviceLocal | kHostMappable, 16384 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 1> host_only{{
      {kHostMappable, 23781 * kMiB},
  }};

  EXPECT_FALSE(CanMapDeviceLocalMemory(small_bar));
  EXPECT_TRUE(CanMapDeviceLocalMemory(resizable_bar));
  EXPECT_TRUE(CanMapDeviceLocalMemory(unified_memory));
  EXPECT_TRUE(CanMapDeviceLocalMemory(larger_mappable_heap));
  EXPECT_FALSE(CanMapDeviceLocalMemory(host_only));
}
#endif  // !defined(__wasm__)

}  // namespace
}  // namespace test
}  // namespace onnxruntime
