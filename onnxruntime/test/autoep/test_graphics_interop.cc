// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for InitGraphicsInteropForEpDevice / DeinitGraphicsInteropForEpDevice using the example plugin EPs.
// The example_plugin_ep factory implements the graphics interop callbacks without a real graphics device; the
// kernel registry example EP does not implement them.

#include <string>

#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if !defined(ORT_MINIMAL_BUILD)

namespace {

OrtGraphicsInteropConfig MakeConfig(OrtGraphicsApi graphics_api) {
  OrtGraphicsInteropConfig config = {};
  config.version = ORT_API_VERSION;
  config.graphics_api = graphics_api;
  config.command_queue = nullptr;
  config.additional_options = nullptr;
  return config;
}

// Checks the error code without calling GetErrorCode on an OK (null) status.
void ExpectErrorCode(OrtStatus* ort_status, OrtErrorCode expected) {
  Ort::Status status(ort_status);
  ASSERT_FALSE(status.IsOK()) << "Expected error code " << expected;
  EXPECT_EQ(status.GetErrorCode(), expected) << status.GetErrorMessage();
}

}  // namespace

TEST(GraphicsInteropTest, InitAndDeinitAreForwardedToEpFactory) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  const OrtEpDevice* ep_device = example_ep.get();

  const OrtInteropApi& interop_api = Ort::GetInteropApi();
  const OrtGraphicsInteropConfig config = MakeConfig(ORT_GRAPHICS_API_D3D12);

  // The example EP rejects deinit before init, so this error shows the call reached the factory.
  {
    Ort::Status status(interop_api.DeinitGraphicsInteropForEpDevice(ep_device));
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
    EXPECT_NE(status.GetErrorMessage().find("not initialized"), std::string::npos) << status.GetErrorMessage();
  }

  {
    Ort::Status status(interop_api.InitGraphicsInteropForEpDevice(ep_device, &config));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }

  {
    Ort::Status status(interop_api.InitGraphicsInteropForEpDevice(ep_device, &config));
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find("already initialized"), std::string::npos) << status.GetErrorMessage();
  }

  {
    Ort::Status status(interop_api.DeinitGraphicsInteropForEpDevice(ep_device));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }
}

TEST(GraphicsInteropTest, EpFactoryErrorIsReturned) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));

  const OrtGraphicsInteropConfig config = MakeConfig(ORT_GRAPHICS_API_NONE);
  Ort::Status status(Ort::GetInteropApi().InitGraphicsInteropForEpDevice(example_ep.get(), &config));
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  EXPECT_NE(status.GetErrorMessage().find("unsupported graphics API"), std::string::npos)
      << status.GetErrorMessage();
}

TEST(GraphicsInteropTest, NotImplementedWhenEpFactoryDoesNotSupportIt) {
  RegisteredEpDeviceUniquePtr kernel_registry_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_kernel_registry_info,
                                                         kernel_registry_ep));
  const OrtEpDevice* ep_device = kernel_registry_ep.get();

  const OrtInteropApi& interop_api = Ort::GetInteropApi();
  const OrtGraphicsInteropConfig config = MakeConfig(ORT_GRAPHICS_API_D3D12);

  ExpectErrorCode(interop_api.InitGraphicsInteropForEpDevice(ep_device, &config), ORT_NOT_IMPLEMENTED);
  ExpectErrorCode(interop_api.DeinitGraphicsInteropForEpDevice(ep_device), ORT_NOT_IMPLEMENTED);
}

TEST(GraphicsInteropTest, NullArgumentsAreRejected) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));

  const OrtInteropApi& interop_api = Ort::GetInteropApi();
  const OrtGraphicsInteropConfig config = MakeConfig(ORT_GRAPHICS_API_D3D12);

  ExpectErrorCode(interop_api.InitGraphicsInteropForEpDevice(nullptr, &config), ORT_INVALID_ARGUMENT);
  ExpectErrorCode(interop_api.InitGraphicsInteropForEpDevice(example_ep.get(), nullptr), ORT_INVALID_ARGUMENT);
  ExpectErrorCode(interop_api.DeinitGraphicsInteropForEpDevice(nullptr), ORT_INVALID_ARGUMENT);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime
