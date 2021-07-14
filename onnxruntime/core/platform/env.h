/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/framework/callback.h"
#include "core/platform/env_time.h"
#include "core/platform/telemetry.h"

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif
namespace Eigen {
class ThreadPoolInterface;
}
namespace onnxruntime {

#ifdef _WIN32
using PIDType = unsigned long;
using FileOffsetType = int64_t;
#else
using PIDType = pid_t;
using FileOffsetType = off_t;
#endif

class EnvThread {
 public:
  virtual ~EnvThread() = default;
};

// Parameters that are required to create a set of threads for a thread pool
struct ThreadOptions {
  // Stack size for a new thread. If it is 0, the operating system uses the same value as the stack that's specified for
  // the main thread, which is usually set in the main executable(not controlled by onnxruntime.dll).
  unsigned int stack_size = 0;

  // Thread affinity means a thread can only run on the logical processors that the thread is allowed to run on.
  // If the vector is not empty, set the affinity of each thread to just one CPU.
  // Index is thread index, value is CPU ID, starting from zero. For example, the first thread in the pool will be bound
  // to the logical processor with id of affinity[0]. If the vector is empty, the thread can run on all the processors
  // its process can run on. NOTE: When hyperthreading is enabled, for example, on a 4 cores 8 physical threads CPU,
  // processor group [0,1,2,3] may only contain half of the physical cores.
  std::vector<size_t> affinity;

  // Set or unset denormal as zero.
  bool set_denormal_as_zero = false;
};

/// \brief An interface used by the onnxruntime implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  using EnvThread = onnxruntime::EnvThread;
  virtual ~Env() = default;
  // clang-format off
  /**
   * Start a new thread for a thread pool
   * \param name_prefix A human-readable string for debugging purpose, can be NULL
   * \param index The index value of the thread, for each thread pool instance, the index should start from 0 and be continuous.
   * \param start_address The entry point of thread
   * \param threadpool The thread pool that the new thread belongs to
   * \param thread_options options to create the thread
   *
   * Caller is responsible for deleting the returned value
   */
  // clang-format on
  virtual EnvThread* CreateThread(_In_opt_z_ const ORTCHAR_T* name_prefix, int index,
                                  _In_ unsigned (*start_address)(int id, Eigen::ThreadPoolInterface* param),
                                  Eigen::ThreadPoolInterface* threadpool, const ThreadOptions& thread_options) = 0;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env& Default();

  virtual int GetNumCpuCores() const = 0;

  // This function doesn't support systems with more than 64 logical processors
  virtual std::vector<size_t> GetThreadAffinityMasks() const = 0;

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const {
    return env_time_->NowMicros();
  }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const {
    return env_time_->NowSeconds();
  }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  /// On Windows, it's the min time to sleep, not the actual one.
  virtual void SleepForMicroseconds(int64_t micros) const = 0;

  /**
   * Gets the length of the specified file.
   */
  virtual common::Status GetFileLength(_In_z_ const ORTCHAR_T* file_path, size_t& length) const = 0;
  virtual common::Status GetFileLength(int fd, /*out*/ size_t& file_size) const = 0;

  /**
   * Copies the content of the file into the provided buffer.
   * @param file_path The path to the file.
   * @param offset The file offset from which to start reading.
   * @param length The length in bytes to read.
   * @param buffer The buffer in which to write.
   */
  virtual common::Status ReadFileIntoBuffer(_In_z_ const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                                            gsl::span<char> buffer) const = 0;

  using MappedMemoryPtr = std::unique_ptr<char[], OrtCallbackInvoker>;

  /**
   * Maps the content of the file into memory.
   * This is a copy-on-write mapping, so any changes are not written to the
   * actual file.
   * @param file_path The path to the file.
   * @param offset The file offset from which to start the mapping.
   * @param length The length in bytes of the mapping.
   * @param[out] mapped_memory A smart pointer to the mapped memory which
   *             unmaps the memory (unless release()'d) when destroyed.
   */
  virtual common::Status MapFileIntoMemory(_In_z_ const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
                                           MappedMemoryPtr& mapped_memory) const = 0;

#ifdef _WIN32
  /// \brief Returns true if the directory exists.
  virtual bool FolderExists(const std::wstring& path) const = 0;
  /// \brief Recursively creates the directory, if it doesn't exist.
  virtual common::Status CreateFolder(const std::wstring& path) const = 0;
  // Mainly for use with protobuf library
  virtual common::Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) const = 0;
  // Mainly for use with protobuf library
  virtual common::Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) const = 0;
#endif
  /// \brief Returns true if the directory exists.
  virtual bool FolderExists(const std::string& path) const = 0;
  /// \brief Recursively creates the directory, if it doesn't exist.
  virtual common::Status CreateFolder(const std::string& path) const = 0;
  // Recursively deletes the directory and its contents.
  // Note: This function is not thread safe!
  virtual common::Status DeleteFolder(const PathString& path) const = 0;
  // Mainly for use with protobuf library
  virtual common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const = 0;
  // Mainly for use with protobuf library
  virtual common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const = 0;
  // Mainly for use with protobuf library
  virtual common::Status FileClose(int fd) const = 0;

  /** Gets the canonical form of a file path (symlinks resolved). */
  virtual common::Status GetCanonicalPath(
      const PathString& path,
      PathString& canonical_path) const = 0;

  // This functions is always successful. It can't fail.
  virtual PIDType GetSelfPid() const = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // global_symbols only has an effect on unix, where a value of true means to load with RTLD_GLOBAL vs RTLD_LOCAL
  // 
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  virtual common::Status LoadDynamicLibrary(const std::string& library_filename, bool global_symbols, void** handle) const = 0;

  virtual common::Status UnloadDynamicLibrary(void* handle) const = 0;

  // \brief Gets the file path of the onnx runtime code
  //
  // Used to help load other shared libraries that live in the same folder as the core code, for example
  // The DNNL provider shared library. Without this path, the module won't be found on windows in all cases.
  virtual std::string GetRuntimePath() const { return ""; }

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadDynamicLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadDynamicLibrary() can use
  virtual std::string FormatLibraryFileName(const std::string& name, const std::string& version) const = 0;

  // \brief returns a provider that will handle telemetry on the current platform
  virtual const Telemetry& GetTelemetryProvider() const = 0;

  // \brief returns a value for the queried variable name (var_name)
  //
  // Returns the corresponding value stored in the environment variable if available
  // Returns empty string if there is no such environment variable available
  virtual std::string GetEnvironmentVar(const std::string& var_name) const = 0;

 protected:
  Env();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Env);
  EnvTime* env_time_ = EnvTime::Default();
};

}  // namespace onnxruntime
