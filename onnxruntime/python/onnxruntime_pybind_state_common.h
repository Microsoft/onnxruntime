// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"

namespace onnxruntime {
class InferenceSession;

namespace python {

using namespace onnxruntime;
using namespace onnxruntime::logging;

struct CustomOpLibrary {
  CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so);

  ~CustomOpLibrary();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpLibrary);

 private:
  void* library_handle_ = nullptr;
};

struct CustomOpLibraries {
  CustomOpLibraries() = default;

  void AddLibrary(std::unique_ptr<CustomOpLibrary> custom_op_library);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpLibraries);

 private:
  std::vector<std::unique_ptr<CustomOpLibrary>> custom_op_libraries_;
  std::mutex mutex_;
};

struct PySessionOptions : public SessionOptions {
  // Have the life cycle of the OrtCustomOpDomain pointers managed by a smart pointer
  std::vector<std::shared_ptr<OrtCustomOpDomain>> custom_op_domains_;
};

inline const PySessionOptions& GetDefaultCPUSessionOptions() {
  static PySessionOptions so;
  return so;
}

inline AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<TAllocator>();
  return alloc;
}

class SessionObjectInitializer {
 public:
  typedef const PySessionOptions& Arg1;
  // typedef logging::LoggingManager* Arg2;
  static const std::string default_logger_id;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  // operator Arg2() {
  //   static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
  //                                                 Severity::kWARNING, false, LoggingManager::InstanceType::Default,
  //                                                 &default_logger_id};
  //   return &default_logging_manager;
  // }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

Environment& get_env();

CustomOpLibraries& get_custom_op_libraries();

void InitializeSession(InferenceSession* sess, const std::vector<std::string>& provider_types);

}  // namespace python
}  // namespace onnxruntime
