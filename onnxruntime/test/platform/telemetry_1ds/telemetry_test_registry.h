// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_WIN32)
#include <Windows.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace onnxruntime::test {

class TelemetryTestRegistry {
 public:
  TelemetryTestRegistry()
      : path_("SOFTWARE\\Microsoft\\onnxruntime-tests\\telemetry-" + std::to_string(::GetCurrentProcessId())) {
    Check(::RegCreateKeyExA(HKEY_CURRENT_USER, path_.c_str(), 0, nullptr, 0, KEY_ALL_ACCESS,
                            nullptr, &key_, nullptr),
          "create isolated HKCU");
    Check(::RegOverridePredefKey(HKEY_CURRENT_USER, key_), "isolate HKCU");
  }

  ~TelemetryTestRegistry() {
    Check(::RegOverridePredefKey(HKEY_CURRENT_USER, nullptr), "restore HKCU");
    Check(::RegCloseKey(key_), "close isolated HKCU");
    Check(::RegDeleteTreeA(HKEY_CURRENT_USER, path_.c_str()), "remove isolated HKCU");
  }

  TelemetryTestRegistry(const TelemetryTestRegistry&) = delete;
  TelemetryTestRegistry& operator=(const TelemetryTestRegistry&) = delete;

 private:
  static void Check(LSTATUS status, const char* operation) {
    if (status != ERROR_SUCCESS) {
      std::fprintf(stderr, "FAIL: could not %s: %ld\n", operation, status);
      std::fflush(stderr);
      std::_Exit(32);
    }
  }

  std::string path_;
  HKEY key_ = nullptr;
};

}  // namespace onnxruntime::test
#endif
