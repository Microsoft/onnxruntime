// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "core/common/inlined_containers.h"
#include "core/platform/env.h"
#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_environment.h"
#include "mat/config.h"
#include "http/HttpClientFactory.hpp"
#include "IHttpClient.hpp"
#include "ILogConfiguration.hpp"
#include "ILogManager.hpp"
#include "telemetry_test_registry.h"

#if defined(ORT_TELEMETRY_TRANSPORT_LIBRARY) && defined(HAVE_MAT_CURL_HTTP_CLIENT)
#include <curl/curl.h>
#endif

namespace {
namespace sdk = Microsoft::Applications::Events;
using namespace std::chrono_literals;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::fflush(stderr);
    std::_Exit(1);
  }
}

struct Observation {
  std::mutex mutex;
  std::condition_variable changed;
  size_t sent = 0;
  size_t completed = 0;
  size_t succeeded = 0;
  size_t failed = 0;
  size_t aborted = 0;
  size_t clients_destroyed = 0;
  bool configured = false;
  bool suppressed = false;
  bool pending = false;
  sdk::ILogManager* manager = nullptr;
};

Observation observation;
std::string collector_url;

class LoopbackHttpClient final : public sdk::IHttpClient {
 public:
  LoopbackHttpClient() : client_(sdk::HttpClientFactory::Create()) {
    Check(client_ != nullptr, "SDK did not create a native HTTP transport");
    auto request = std::unique_ptr<sdk::IHttpRequest>(client_->CreateRequest());
    Check(request != nullptr, "native transport did not create a request");
#if defined(_WIN32)
    Check(request->GetId().rfind("WH-", 0) == 0, "fixture requires the real WinHTTP transport");
    std::fputs("TRANSPORT WinHTTP\n", stdout);
#elif defined(HAVE_MAT_CURL_HTTP_CLIENT)
    Check(request->GetId().rfind("REQ-", 0) == 0, "fixture requires the real curl transport");
    std::fputs("TRANSPORT curl\n", stdout);
#else
    std::fputs("TRANSPORT platform\n", stdout);
#endif
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoopbackHttpClient);

  ~LoopbackHttpClient() override {
    client_->CancelAllRequests();
    client_.reset();
    std::lock_guard lock(observation.mutex);
    Check(observation.sent == observation.completed, "native client did not drain every terminal callback");
    ++observation.clients_destroyed;
  }

  sdk::IHttpRequest* CreateRequest() override { return client_->CreateRequest(); }

  void SendRequestAsync(sdk::IHttpRequest* request, sdk::IHttpResponseCallback* callback) override {
    // Every request goes to the parent-owned collector, including retries.
    request->SetUrl(collector_url);
    auto tracked = std::make_unique<Callback>(*callback);
    auto* tracked_ptr = tracked.get();
    {
      std::lock_guard lock(observation.mutex);
      ++observation.sent;
      callbacks_.push_back(std::move(tracked));
    }
    client_->SendRequestAsync(request, tracked_ptr);
  }

  void CancelRequestAsync(const std::string& id) override { client_->CancelRequestAsync(id); }
  void CancelAllRequests() override { client_->CancelAllRequests(); }
  void ApplySettings(sdk::ILogConfiguration& config) override { client_->ApplySettings(config); }

 private:
  class Callback final : public sdk::IHttpResponseCallback {
   public:
    explicit Callback(sdk::IHttpResponseCallback& callback) : callback_(callback) {}
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Callback);

    void OnHttpStateEvent(sdk::HttpStateEvent state, void* data, size_t size) override {
      callback_.OnHttpStateEvent(state, data, size);
    }

    void OnHttpResponse(sdk::IHttpResponse* response) override {
      const auto result = response->GetResult();
      const auto status = response->GetStatusCode();
      {
        std::lock_guard lock(observation.mutex);
        Check(!completed_, "native transport delivered a duplicate terminal callback");
        completed_ = true;
      }
      callback_.OnHttpResponse(response);
      {
        std::lock_guard lock(observation.mutex);
        ++observation.completed;
        if (result == sdk::HttpResult_Aborted) {
          ++observation.aborted;
        } else if (result == sdk::HttpResult_OK && status == 200) {
          ++observation.succeeded;
        } else {
          ++observation.failed;
        }
      }
      observation.changed.notify_all();
    }

   private:
    sdk::IHttpResponseCallback& callback_;
    bool completed_ = false;
  };

  std::shared_ptr<sdk::IHttpClient> client_;
  onnxruntime::InlinedVector<std::unique_ptr<Callback>> callbacks_;
};

void VerifyExit() {
  std::lock_guard lock(observation.mutex);
  if (observation.suppressed) {
    Check(!observation.configured && !observation.manager && observation.sent == 0,
          "suppressed telemetry initialized the SDK");
  } else {
    Check(observation.configured && observation.sent > 0, "test did not exercise an upload");
    Check(observation.clients_destroyed == 1 && observation.completed == observation.sent,
          "provider did not destroy and drain its native transport");
    if (observation.pending) {
      Check(observation.aborted > 0, "pending upload was not cancelled during provider teardown");
    }
  }
  std::fputs("PASS telemetry lifecycle\n", stdout);
  std::fflush(stdout);
}

void Run(std::string_view mode) {
  observation.suppressed = mode == "disabled" || mode == "unit-disabled" || mode == "ci-disabled";
  observation.pending = mode == "pending";
  const auto url = onnxruntime::telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_COLLECTOR");
  Check(url.has_value() && !url->empty(), "missing or unreadable loopback collector URL");
  collector_url = *url;
  constexpr std::string_view prefix = "http://127.0.0.1:";
  constexpr std::string_view suffix = "/collect";
  Check(collector_url.starts_with(prefix) && collector_url.ends_with(suffix),
        "collector must be the parent's IPv4 loopback endpoint");
  const auto port = std::string_view(collector_url).substr(prefix.size(), collector_url.size() - prefix.size() - suffix.size());
  Check(!port.empty() && port.size() <= 5 &&
            std::all_of(port.begin(), port.end(), [](char c) { return c >= '0' && c <= '9'; }) &&
            std::stoi(std::string(port)) > 0 && std::stoi(std::string(port)) <= 65535,
        "invalid loopback collector port");
#if defined(_WIN32)
  static onnxruntime::test::TelemetryTestRegistry registry;
#endif
#if defined(ORT_TELEMETRY_TRANSPORT_LIBRARY)
  // Stop the provider while worker threads can still exit, before entering the
  // loader lock. A live MSVC SDK worker can itself retain the DLL.
  auto owned_provider = std::make_unique<onnxruntime::OneDsTelemetry>();
  const onnxruntime::Telemetry* telemetry = owned_provider.get();
#else
  Check(std::atexit(VerifyExit) == 0, "could not register teardown verification");
  const onnxruntime::Telemetry* telemetry = nullptr;
  const auto initialize = [&] { telemetry = &onnxruntime::Env::Default().GetTelemetryProvider(); };
  if (mode == "threaded") {
    std::thread initializer(initialize);
    initializer.join();
  } else {
    initialize();
  }
#endif
  Check(telemetry->IsEnabled() != observation.suppressed, "unexpected provider enabled state");
  const auto log = [&](const char* message) {
    telemetry->LogRuntimeError(7, ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, message),
                               "telemetry-test.cc", "SyntheticFailure", 42);
  };
  log("synthetic-before-disable");
  if (observation.suppressed) {
    return;
  }
  Check(observation.manager != nullptr, "provider did not publish the SDK manager");
  observation.manager->UploadNow();
  if (observation.pending) {
    std::string command;
    Check(static_cast<bool>(std::getline(std::cin, command)) && command == "shutdown",
          "parent did not confirm an in-flight native request");
    return;
  }
  {
    std::unique_lock lock(observation.mutex);
    Check(observation.changed.wait_for(lock, 20s, [] { return observation.succeeded > 0; }),
          "real native transport never completed a successful upload");
    if (mode == "retry") {
      Check(observation.failed > 0 && observation.sent >= 2, "SDK did not retry the failed request");
    }
  }
  telemetry->DisableTelemetryEvents();
  Check(!telemetry->IsEnabled(), "DisableTelemetryEvents did not disable events");
  size_t previous_sent;
  {
    std::lock_guard lock(observation.mutex);
    previous_sent = observation.sent;
  }
  log("synthetic-while-disabled");
  telemetry->EnableTelemetryEvents();
  Check(telemetry->IsEnabled(), "EnableTelemetryEvents did not re-enable events");
  log("synthetic-after-enable");
  observation.manager->UploadNow();
  {
    std::unique_lock lock(observation.mutex);
    Check(observation.changed.wait_for(lock, 10s, [&] { return observation.succeeded >= 2; }),
          "re-enabled telemetry did not upload");
    Check(observation.sent >= previous_sent + 1, "re-enable did not reach the native transport");
  }
#if defined(ORT_TELEMETRY_TRANSPORT_LIBRARY)
  owned_provider.reset();
  VerifyExit();
#if defined(HAVE_MAT_CURL_HTTP_CLIENT)
  // This isolated host owns every curl user in the fixture. Release its global
  // allocations before unloading a private static curl image.
  curl_global_cleanup();
#endif
#endif
}
}  // namespace

namespace onnxruntime {
void OrtTelemetryTestConfigure(sdk::ILogConfiguration& config) {
  observation.configured = true;
  config[sdk::CFG_STR_COLLECTOR_URL] = collector_url;
  config[sdk::CFG_BOOL_ENABLE_NET_DETECT] = false;
  config[sdk::CFG_MAP_HTTP][sdk::CFG_BOOL_HTTP_COMPRESSION] = false;
  config[sdk::CFG_MAP_TPM][sdk::CFG_STR_TPM_BACKOFF] = "E,10,100,2,0";
  auto client = std::make_shared<LoopbackHttpClient>();
  client->ApplySettings(config);
  config.AddModule(sdk::CFG_MODULE_HTTP_CLIENT, client);
}

void OrtTelemetryTestReady(sdk::ILogManager& manager) {
  observation.manager = &manager;
  manager.SetTransmitProfile(sdk::TransmitProfile_RealTime);
}
}  // namespace onnxruntime

#if defined(ORT_TELEMETRY_TRANSPORT_LIBRARY)
#if defined(_WIN32)
#define ORT_TELEMETRY_TEST_EXPORT __declspec(dllexport)
#else
#define ORT_TELEMETRY_TEST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" ORT_TELEMETRY_TEST_EXPORT void OrtRunTelemetryTransportTest() {
  Run("upload");
}
#else
int main(int argc, char* argv[]) {
  Check(argc == 3, "expected test mode and fixture library path");
  const std::string_view mode = argv[1];
  if (mode != "reload") {
    Run(mode);
    return 0;
  }
  for (int iteration = 0; iteration != 2; ++iteration) {
#if defined(_WIN32)
    HMODULE library = LoadLibraryA(argv[2]);
    Check(library != nullptr, "could not load telemetry fixture DLL");
    auto run = reinterpret_cast<void (*)()>(GetProcAddress(library, "OrtRunTelemetryTransportTest"));
#else
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      std::fprintf(stderr, "%s\n", dlerror());
    }
    Check(library != nullptr, "could not load telemetry fixture library");
    auto run = reinterpret_cast<void (*)()>(dlsym(library, "OrtRunTelemetryTransportTest"));
#endif
    Check(run != nullptr, "missing telemetry fixture entry point");
    // Let caller-thread TLS destructors finish before unloading the fixture.
    std::thread(run).join();
#if defined(_WIN32)
    Check(FreeLibrary(library) != 0, "could not unload telemetry fixture DLL");
    Check(GetModuleHandleA(argv[2]) == nullptr, "fixture DLL remained loaded after FreeLibrary with an active SDK worker");
#else
    Check(dlclose(library) == 0, "could not unload telemetry fixture library");
#if defined(__linux__)
    Check(dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD) == nullptr, "fixture library remained loaded after dlclose");
#endif
#endif
  }
  return 0;
}
#endif
