// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "async_execution_stream.h"
#include <thread>
#include <iostream>
#include "core/common/make_unique.h"

namespace onnxruntime {

AsyncExecutionStream::AsyncExecutionStream(const std::string& name)
    : name_(name),
      stop_(false) {
  worker_thread_ = onnxruntime::make_unique<std::thread>([&]() {
    this->ThreadProc();
  });
}

AsyncExecutionStream::~AsyncExecutionStream() {
  stop_ = true;
  kick_off_.notify_one();
  worker_thread_->join();
}

void AsyncExecutionStream::Launch(std::function<void()> func) {
  std::unique_lock<std::mutex> lock(mutex_);
  tasks_.push_back(func);
  kick_off_.notify_all();
}

void AsyncExecutionStream::Synchronize() {
  std::unique_lock<std::mutex> lock(mutex_);
  drained_.wait(lock, [this]() { return tasks_.empty() || stop_; });
}

void AsyncExecutionStream::ThreadProc() {
  while (!stop_) {
    std::unique_lock<std::mutex> lock(mutex_);
    kick_off_.wait(lock, [this] { return !tasks_.empty() || stop_; });
    while (!tasks_.empty() && !stop_) {
      auto func = tasks_.front();
      tasks_.pop_front();
      bool empty = tasks_.empty();
      lock.unlock();
      func();
      if (empty) {
        drained_.notify_one();
      }
      lock.lock();
    }
  }
}

}  // namespace onnxruntime