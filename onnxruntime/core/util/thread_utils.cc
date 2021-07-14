#include "thread_utils.h"
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>
#include "core/session/ort_apis.h"
#include "core/platform/threadpoollite.h"

namespace onnxruntime {
namespace concurrency {
static std::unique_ptr<ThreadPool>
CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options, ThreadPoolType tpool_type) {
  if (options.thread_pool_size == 1)
    return nullptr;
  std::vector<size_t> cpu_list;
  ThreadOptions to;
  if (options.affinity_vec_len != 0) {
    to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
  }
  if (options.thread_pool_size <= 0) {  // default
    cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
  }
  to.set_denormal_as_zero = options.set_denormal_as_zero;

  if (tpool_type == ThreadPoolType::INTER_OP) {
    return std::make_unique<ThreadPool>(env, to, options.name,
                                        options.thread_pool_size,
                                        options.allow_spinning);
  } else {
    return options.use_tplite ? std::make_unique<ThreadPoolLite>(env, to, options.name,
                                                                 options.thread_pool_size,
                                                                 options.allow_spinning)
                              : std::make_unique<ThreadPool>(env, to, options.name,
                                                             options.thread_pool_size,
                                                             options.allow_spinning);
  }
}

std::unique_ptr<ThreadPool>
CreateThreadPool(Env* env, OrtThreadPoolParams options, ThreadPoolType tpool_type) {
// If openmp is enabled we don't want to create any additional threadpools for sequential execution.
// However, parallel execution relies on the existence of a separate threadpool. Hence we allow eigen threadpools
// to be created for parallel execution.
#ifdef _OPENMP
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(options);
  if (tpool_type != ThreadPoolType::INTER_OP) {
    return nullptr;
  } else {
    return CreateThreadPoolHelper(env, options);
  }
#else
  //ORT_UNUSED_PARAMETER(tpool_type);
  return CreateThreadPoolHelper(env, options, tpool_type);
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
namespace OrtApis {
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out) {
  *out = new OrtThreadingOptions();
  return nullptr;
}

ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions* p) {
  delete p;
}

ORT_API_STATUS_IMPL(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->intra_op_thread_pool_params.thread_pool_size = intra_op_num_threads;
  return nullptr;
}
ORT_API_STATUS_IMPL(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->inter_op_thread_pool_params.thread_pool_size = inter_op_num_threads;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!(allow_spinning == 1 || allow_spinning == 0)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received invalid value for allow_spinning. Valid values are 0 or 1");
  }
  tp_options->intra_op_thread_pool_params.allow_spinning = allow_spinning;
  tp_options->inter_op_thread_pool_params.allow_spinning = allow_spinning;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* tp_options) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->intra_op_thread_pool_params.set_denormal_as_zero = true;
  tp_options->inter_op_thread_pool_params.set_denormal_as_zero = true;
  return nullptr;
}

}  // namespace OrtApis
