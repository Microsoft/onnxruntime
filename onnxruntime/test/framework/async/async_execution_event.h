// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>

namespace onnxruntime {

class AsyncExecutionEvent {
 public:
  AsyncExecutionEvent() {
    // note: event are signaled in ctor, similar to cudaEventCreate
    signaled_.store(true);
  }

  ~AsyncExecutionEvent() {
    Signal();
  }

  void Reset() {
    signaled_.store(false);
  }

  void Wait() {
    while (!signaled_.load()) {
      std::this_thread::yield();
    }
  }

  void Signal() {
    signaled_.store(true);
  }

 private:
  std::atomic<bool> signaled_;
};

}  // namespace onnxruntime