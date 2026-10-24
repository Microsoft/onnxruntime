// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

#include <ILogConfiguration.hpp>
#include <NullObjects.hpp>

#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_1ds_platform.h"

namespace onnxruntime {
namespace {

Microsoft::Applications::Events::ILogManager* mock_manager = nullptr;
bool clean_shutdown = false;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s (Win32 error: %lu)\n", message, GetLastError());
    std::fflush(stderr);
    // Also fail reliably from a DLL process-detach destructor.
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE);
    std::abort();
  }
}

}  // namespace

void OrtTelemetryLifecycleInstall(
    uint32_t register_count, Microsoft::Applications::Events::ILogManager*& manager,
    std::unique_ptr<Microsoft::Applications::Events::ILogConfiguration>& config) {
  Check(register_count == 1, "Expected exactly one telemetry instance");
  Check(!manager && !config, "Real uploader must be disabled");
  Check(mock_manager != nullptr, "Mock uploader must be ready before construction");
  config = std::make_unique<Microsoft::Applications::Events::ILogConfiguration>();
  manager = mock_manager;
}

void OrtTelemetryLifecycleClean(
    uint32_t register_count, Microsoft::Applications::Events::ILogManager* manager,
    const Microsoft::Applications::Events::ILogConfiguration* config) {
  Check(register_count == 0 && !manager && !config, "Telemetry state was not cleaned up");
  clean_shutdown = true;
}

namespace {

class MockLogManager : public Microsoft::Applications::Events::NullLogManager {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MockLogManager);

  explicit MockLogManager(bool pending) : callback_(CreateEventW(nullptr, TRUE, !pending, nullptr)) {
    Check(callback_ != nullptr, "CreateEventW failed");
  }

  ~MockLogManager() override { CloseHandle(callback_); }

  Microsoft::Applications::Events::status_t Flush() override {
    ++flush_count;
    return Microsoft::Applications::Events::STATUS_SUCCESS;
  }

  void FlushAndTeardown() override {
    Check(flush_count == 1, "Expected Flush before FlushAndTeardown");
    ++teardown_count;
    // A pending callback cannot arrive after ExitProcess has killed the other
    // threads. Bound the mock's wait so a regression does not leave a stuck DLL.
    Check(WaitForSingleObject(callback_, 2000) == WAIT_OBJECT_0, "Timed out draining the pending uploader");
  }

  int flush_count = 0;
  int teardown_count = 0;

 private:
  HANDLE callback_;
};

class TelemetryLifetime {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TelemetryLifetime);

  TelemetryLifetime(bool process_exit, HANDLE verified)
      : manager_(process_exit), process_exit_(process_exit), verified_(verified) {
    mock_manager = &manager_;
    telemetry_.emplace();
  }

  ~TelemetryLifetime() {
    if (telemetry_) {
      Destroy();
    }
  }

  void Destroy() {
    Check(telemetry_internal::IsDllShutdownInProgress() == process_exit_, "Unexpected DLL shutdown state");
    telemetry_.reset();
    const int expected_calls = process_exit_ ? 0 : 1;
    Check(manager_.flush_count == expected_calls, "Unexpected Flush count");
    Check(manager_.teardown_count == expected_calls, "Unexpected FlushAndTeardown count");
    Check(clean_shutdown == !process_exit_, "Unexpected shared-state cleanup");
    Check(SetEvent(verified_) != FALSE, "Could not signal verified destruction");
  }

 private:
  MockLogManager manager_;
  std::optional<OneDsTelemetry> telemetry_;
  bool process_exit_;
  HANDLE verified_;
};

}  // namespace
}  // namespace onnxruntime

extern "C" void OrtTestTelemetryShutdown(bool process_exit, bool destroy_now, HANDLE verified) {
  // A DLL-owned static exercises CRT destruction during real Windows shutdown.
  static onnxruntime::TelemetryLifetime lifetime(process_exit, verified);
  if (destroy_now) {
    lifetime.Destroy();
  }
}
