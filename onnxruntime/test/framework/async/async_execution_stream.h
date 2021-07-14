// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <thread>

namespace onnxruntime {

class AsyncExecutionStream {
 public:
  AsyncExecutionStream(const std::string& name);
  ~AsyncExecutionStream();
  void Launch(std::function<void()> func);
  void Synchronize();

 protected:
  void ThreadProc();

 private:
  std::string name_;
  std::atomic<bool> stop_;
  std::condition_variable kick_off_;
  std::condition_variable drained_;
  std::mutex mutex_;
  std::deque<std::function<void()>> tasks_;
  std::unique_ptr<std::thread> worker_thread_;
};

}  // namespace onnxruntime