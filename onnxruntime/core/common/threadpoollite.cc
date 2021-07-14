#include "core/platform/threadpoollite.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

namespace concurrency {

ThreadPoolLite::ThreadPoolLite(Env*,
                               const ThreadOptions& options,
                               const NAME_CHAR_TYPE*,
                               int num_threads,
                               bool) : profiler_(num_threads - 1, ORT_TSTR("ThreadPoolLite")) {
  num_sub_threads_ = num_threads - 1;
  slots_.assign(num_threads - 1, {});
  set_denormal_as_zero_ = options.set_denormal_as_zero;
#ifdef _WIN32
  size_t affinity_mask = 3;
  for (int i = 0; i < num_sub_threads_; ++i) {
    sub_threads_.emplace_back(&ThreadPoolLite::ThreadLoop, this, i);
    SetThreadAffinityMask(sub_threads_.back().native_handle(), affinity_mask);
    affinity_mask <<= 2;
  }
#else
  for (int i = 0; i < num_sub_threads_; ++i) {
    sub_threads_.emplace_back(&ThreadPoolLite::ThreadLoop, this, i);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    ORT_ENFORCE(0 == pthread_setaffinity_np(sub_threads_.back().native_handle(), sizeof(cpu_set_t), &cpuset),
                "Failed to set thread affinity in posix system.");
  }
#endif
}

ThreadPoolLite::~ThreadPoolLite() {
  exit_ = true;
  for (std::thread& t : sub_threads_) {
    t.join();
  }
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, double c, const Fn& fn) {
  TensorOpCost cost{0, 0, c};
  ParallelFor(total, cost, fn);
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost, const Fn& fn) {
  if (0 == total) {
    return;
  }
  std::ptrdiff_t block_size = GetBlockSize(total, cost, num_sub_threads_ + 1);
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<std::ptrdiff_t> iter{0};
  SchdFn schd_fn = [&]() {
    std::ptrdiff_t i{0};
    while ((i = iter.fetch_add(block_size, std::memory_order_relaxed)) < total) {
      fn(i, std::min(total, i + block_size));
    }
  };
  ParallelForImpl(schd_fn, block_size);
}

void ThreadPoolLite::SimpleParallelFor(std::ptrdiff_t total, const SimpleFn& fn) {
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<std::ptrdiff_t> iter{0};
  SchdFn schd_fn = [&]() {
    std::ptrdiff_t i{0};
    while ((i = iter.fetch_add(1, std::memory_order_relaxed)) < total) {
      fn(i);
    }
  };
  ParallelForImpl(schd_fn, 1);
}

void ThreadPoolLite::ParallelForImpl(const SchdFn& schd_fn, std::ptrdiff_t block_size) {
  profiler_.LogStartAndCoreAndBlock(block_size);
  std::vector<Slot*> engaged_slots;
  for (int i = 0; i < num_sub_threads_; ++i) {
    Stage stage = Stage::empty;
    if (slots_[i].stage_.compare_exchange_weak(stage, Stage::loading, std::memory_order_relaxed)) {
      slots_[i].schd_fn_ = schd_fn;
      slots_[i].stage_.store(Stage::ready, std::memory_order_release);
      engaged_slots.push_back(&slots_[i]);
    }
  }
  profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);
  schd_fn();
  profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);
  for (auto slot : engaged_slots) {
    Stage stage_ready = Stage::ready;
    Stage stage_done = Stage::done;
    while (!slot->stage_.compare_exchange_weak(stage_ready, Stage::empty, std::memory_order_relaxed) &&
           !slot->stage_.compare_exchange_weak(stage_done, Stage::empty, std::memory_order_relaxed)) {
      stage_ready = Stage::ready;
      stage_done = Stage::done;
      onnxruntime::concurrency::SpinPause();
    }
  }
  profiler_.LogEnd(ThreadPoolProfiler::WAIT);
}

void ThreadPoolLite::Schedule(SchdFn schd_fn) {
  schd_fn();
}

void ThreadPoolLite::StartProfiling() {
  profiler_.Start();
}

std::string ThreadPoolLite::StopProfiling() {
  return profiler_.Stop();
}

void ThreadPoolLite::ThreadLoop(int thread_id) {
  profiler_.LogThreadId(thread_id);
  auto& slot = slots_[thread_id];
  SetDenormalAsZero(set_denormal_as_zero_);
  while (!exit_) {
    Stage stage = Stage::ready;
    if (slot.stage_.compare_exchange_weak(stage, Stage::running, std::memory_order_acquire, std::memory_order_relaxed)) {
      slot.schd_fn_();
      profiler_.LogRun(thread_id);
      slot.stage_.store(Stage::done, std::memory_order_relaxed);
    } else {
      onnxruntime::concurrency::SpinPause();
    }
  }
}

}  // namespace concurrency
}  // namespace onnxruntime
