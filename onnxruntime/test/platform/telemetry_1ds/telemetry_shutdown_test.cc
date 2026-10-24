// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>

#include "core/platform/env.h"
#include "DebugEvents.hpp"
#include "IHttpClient.hpp"
#include "ILogConfiguration.hpp"
#include "ILogManager.hpp"
#include "ITaskDispatcher.hpp"
#include "pal/PAL.hpp"
#include "telemetry_test_registry.h"

namespace {

namespace sdk = Microsoft::Applications::Events;
namespace pal = Microsoft::Applications::Events::PlatformAbstraction;
using namespace std::chrono_literals;

const auto main_thread = std::this_thread::get_id();
std::thread::id initialization_thread;
std::mutex state_mutex;
std::condition_variable state_changed;
std::weak_ptr<sdk::ITaskDispatcher> worker;
sdk::DebugEventSource debug_source;
void* worker_address = nullptr;
bool disabled = false;
bool env_ready = false;
bool mutex_initialized = false;
bool mutex_initialized_before_env = false;
bool mutex_retired = false;
bool callback_started = false;
bool callback_completed = false;
bool release_callback = false;
bool worker_joined = false;

[[noreturn]] void Fail(int exit_code, const char* message) {
  std::fprintf(stderr, "FAIL: %s\n", message);
  std::_Exit(exit_code);
}

class NoNetworkHttpClient final : public sdk::IHttpClient {
 public:
  sdk::IHttpRequest* CreateRequest() override { return new sdk::SimpleHttpRequest("no-network"); }

  void SendRequestAsync(sdk::IHttpRequest*, sdk::IHttpResponseCallback*) override {
    Fail(32, "mutex-order test unexpectedly attempted an upload");
  }

  void CancelRequestAsync(const std::string&) override {}
};

class MutexLifetime {
 public:
  MutexLifetime() {
    std::lock_guard lock(state_mutex);
    mutex_initialized = true;
    mutex_initialized_before_env = !env_ready;
    std::fprintf(stderr, "SDK mutex initialized %s Env::Default construction completed\n",
                 env_ready ? "after" : "before");
  }

  ~MutexLifetime() {
    std::lock_guard lock(state_mutex);
    // Registered immediately after the real mutex, so this check runs before libc++
    // destroys it. Failing exits without deliberately accessing a destroyed mutex.
    if (!callback_completed || !worker_joined || !worker.expired()) {
      Fail(21, "SDK mutex lifetime ended before Env::Default joined its worker");
    }
    mutex_retired = true;
    std::fputs("SDK callback completed and worker joined before mutex retirement\n", stderr);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MutexLifetime);
};

class DiagnosticCallback final : public sdk::Task {
 public:
  DiagnosticCallback() {
    Type = Call;
    TargetTime = 0;
    TypeName = "OrtTelemetryShutdownTest";
  }

  void operator()() override {
    if (std::this_thread::get_id() == main_thread) {
      Fail(32, "callback did not run on the SDK worker");
    }
    debug_source.DispatchEvent(sdk::DebugEvent{sdk::DebugEventType::EVT_UNKNOWN});
    {
      std::unique_lock lock(state_mutex);
      callback_started = true;
      state_changed.notify_all();
      if (!state_changed.wait_for(lock, 10s, [] { return release_callback; })) {
        Fail(32, "process exit did not release the SDK callback");
      }
    }
    debug_source.DispatchEvent(sdk::DebugEvent{sdk::DebugEventType::EVT_UNKNOWN});
    {
      std::lock_guard lock(state_mutex);
      callback_completed = true;
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DiagnosticCallback);
};

void ReleaseCallback() {
  std::lock_guard lock(state_mutex);
  release_callback = true;
  state_changed.notify_all();
}

void VerifyExit() {
  std::lock_guard lock(state_mutex);
  if (disabled) {
    if (mutex_initialized || callback_started || worker_joined) {
      Fail(32, "disabled telemetry initialized SDK state");
    }
    std::fputs("PASS: disabled telemetry created no SDK worker or mutex\n", stderr);
    return;
  }
  if (!mutex_initialized_before_env || !mutex_retired || !callback_started ||
      !callback_completed || !worker_joined || !worker.expired()) {
    Fail(32, "real singleton, SDK worker, and mutex teardown were not fully exercised");
  }
  std::fputs("PASS: Env::Default joined the SDK worker before mutex destruction\n", stderr);
}

void InitializeEnvironment() {
  {
    std::lock_guard lock(state_mutex);
    initialization_thread = std::this_thread::get_id();
  }
  const auto& telemetry = onnxruntime::Env::Default().GetTelemetryProvider();
  {
    std::lock_guard lock(state_mutex);
    env_ready = true;
    state_changed.notify_all();
  }
  if (disabled && telemetry.IsEnabled()) {
    Fail(32, "disabled provider was enabled");
  }
  if (!disabled && !telemetry.IsEnabled()) {
    Fail(32, "Env::Default did not initialize telemetry");
  }
}

}  // namespace

extern "C" void OrtTelemetryBeforeDebugLock() {
  std::unique_lock lock(state_mutex);
  // Delay background first use until the real owner's constructor has returned.
  // The eager diagnostic dispatch must still run synchronously during construction.
  if (std::this_thread::get_id() != initialization_thread &&
      !state_changed.wait_for(lock, 10s, [] { return env_ready; })) {
    Fail(32, "SDK initialization depended on the delayed background callback");
  }
  if (mutex_retired) {
    Fail(21, "SDK accessed its debug mutex after retirement");
  }
}

extern "C" void OrtTelemetryDebugLockInitialized() {
  static MutexLifetime lifetime;
}

extern "C" void OrtTelemetryWorkerJoined(void* value) {
  std::lock_guard lock(state_mutex);
  if (value == worker_address) {
    worker_joined = true;
  }
}

namespace onnxruntime {
void OrtTelemetryTestConfigure(sdk::ILogConfiguration& config) {
  config[sdk::CFG_BOOL_ENABLE_NET_DETECT] = false;
  config.AddModule(sdk::CFG_MODULE_HTTP_CLIENT, std::make_shared<NoNetworkHttpClient>());
}

void OrtTelemetryTestReady(sdk::ILogManager&) {}
}  // namespace onnxruntime

int main(int argc, char* argv[]) {
#if defined(_WIN32)
  static onnxruntime::test::TelemetryTestRegistry registry;
#endif
  if (argc != 2) {
    Fail(32, "expected a telemetry shutdown test mode");
  }
  const std::string_view mode = argv[1];
  if (mode != "enabled" && mode != "threaded" && mode != "disabled" && mode != "disabled-threaded") {
    Fail(32, "unknown telemetry shutdown test mode");
  }
  disabled = mode == "disabled" || mode == "disabled-threaded";
  if (std::atexit(VerifyExit) != 0) {
    Fail(32, "could not register final verification");
  }
  if (mode == "threaded" || mode == "disabled-threaded") {
    std::thread initializer(InitializeEnvironment);
    initializer.join();
  } else {
    InitializeEnvironment();
  }
  if (disabled) {
    return 0;
  }
  // Release the task before Env teardown, without shutting telemetry down in the
  // fixture. Env's existing destructor must perform the actual stop and native join.
  if (std::atexit(ReleaseCallback) != 0) {
    Fail(32, "could not register callback release");
  }
  {
    auto dispatcher = pal::getDefaultTaskDispatcher();
    if (!dispatcher) {
      Fail(32, "SDK dispatcher was unavailable");
    }
    {
      std::lock_guard lock(state_mutex);
      worker = dispatcher;
      worker_address = dispatcher.get();
    }
    dispatcher->Queue(new DiagnosticCallback());
  }
  {
    std::unique_lock lock(state_mutex);
    if (!state_changed.wait_for(lock, 10s, [] { return callback_started; })) {
      Fail(32, "callback did not reach the actual SDK worker");
    }
  }
  return 0;
}
