// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_c_api.h"
#include <memory>
#include <string>

struct OrtThreadPoolParams {
  //0: Use default setting. (All the physical cores or half of the logical cores)
  //1: Don't create thread pool
  //n: Create a thread pool with n threads.
  int thread_pool_size = 0;
  //If it is true and thread_pool_size = 0, populate the thread affinity information in ThreadOptions.
  //Otherwise if the thread_options has affinity information, we'll use it and set it.
  //In the other case, don't set affinity
  bool auto_set_affinity = false;
  //If it is true, the thread pool will spin a while after the queue became empty.
  bool allow_spinning = true;

  unsigned int stack_size = 0;
  //Index is thread id, value is processor ID
  //If the vector is empty, no explict affinity binding
  size_t* affinity_vec = nullptr;
  size_t affinity_vec_len = 0;
  const ORTCHAR_T* name = nullptr;

  // Set or unset denormal as zero
  bool set_denormal_as_zero = false;

  // Use lite version for intra ops
  bool use_tplite = false;
};

struct OrtThreadingOptions {
  // Params for creating the threads that parallelizes execution of an op
  OrtThreadPoolParams intra_op_thread_pool_params;

  // Params for creating the threads that parallelizes execution across ops
  OrtThreadPoolParams inter_op_thread_pool_params;
};

namespace onnxruntime {

namespace concurrency {
enum class ThreadPoolType : uint8_t {
  INTRA_OP,
  INTER_OP
};
std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, OrtThreadPoolParams options,
                                             ThreadPoolType tpool_type);
}  // namespace concurrency
}  // namespace onnxruntime
