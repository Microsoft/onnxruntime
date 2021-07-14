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

#include <memory>

#include "core/platform/threadpool.h"
#include "core/common/common.h"
#include "core/common/cpuid_info.h"
#include "core/common/eigen_common_wrapper.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include "core/platform/ort_mutex.h"
#if !defined(ORT_MINIMAL_BUILD)
#ifdef _WIN32
#include "processthreadsapi.h"
#include <codecvt>
#include <locale>
#elif defined(__APPLE__)
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#elif defined(__wasm__)
#include <emscripten/threading.h>
#else
#include <sched.h>
#endif
#endif

namespace onnxruntime {

namespace concurrency {

#if !defined(ORT_MINIMAL_BUILD)
ThreadPoolProfiler::ThreadPoolProfiler(int num_threads, const CHAR_TYPE* thread_pool_name) : num_threads_(num_threads) {
  child_thread_stats_.assign(num_threads, {});
  if (thread_pool_name) {
#ifdef _WIN32
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;
    thread_pool_name_ = converter.to_bytes(thread_pool_name);
#else
    thread_pool_name_ = thread_pool_name;
#endif
  } else {
    thread_pool_name_ = "unnamed_thread_pool";
  }
}

ThreadPoolProfiler::~ThreadPoolProfiler() {
  enabled_ = false;
}

void ThreadPoolProfiler::Start() {
  enabled_ = true;
}

ThreadPoolProfiler::MainThreadStat& ThreadPoolProfiler::GetMainThreadStat() {
  static thread_local std::unique_ptr<MainThreadStat> stat;
  if (!stat) {
    stat.reset(new MainThreadStat());
  }
  return *stat;
}

std::string ThreadPoolProfiler::Stop() {
  ORT_ENFORCE(enabled_, "Profiler not started yet");
  std::ostringstream ss;
  ss << "{\"main_thread\": {"
     << "\"thread_pool_name\": \""
     << thread_pool_name_ << "\", "
     << GetMainThreadStat().Reset()
     << "}, \"sub_threads\": {"
     << DumpChildThreadStat()
     << "}}";
  return ss.str();
}

void ThreadPoolProfiler::LogStartAndCoreAndBlock(std::ptrdiff_t block_size) {
  if (enabled_) {
    MainThreadStat& stat = GetMainThreadStat();
    stat.LogCore();
    stat.LogBlockSize(block_size);
    stat.LogStart();
  }
}

void ThreadPoolProfiler::LogCoreAndBlock(std::ptrdiff_t block_size) {
  if (enabled_) {
    MainThreadStat& stat = GetMainThreadStat();
    stat.LogCore();
    stat.LogBlockSize(block_size);
  }
}

void ThreadPoolProfiler::LogStart() {
  if (enabled_) {
    GetMainThreadStat().LogStart();
  }
}

void ThreadPoolProfiler::LogEnd(ThreadPoolEvent evt) {
  if (enabled_) {
    GetMainThreadStat().LogEnd(evt);
  }
}

void ThreadPoolProfiler::LogEndAndStart(ThreadPoolEvent evt) {
  if (enabled_) {
    GetMainThreadStat().LogEndAndStart(evt);
  }
}

void ThreadPoolProfiler::MainThreadStat::LogCore() {
#ifdef _WIN32
  core_ = GetCurrentProcessorNumber();
#elif defined(__APPLE__)
#if defined(__x86_64__) || defined(__i386__)
  uint32_t CPUInfo[4];
  __cpuid_count(1, 0, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  if ((CPUInfo[3] & (1 << 9)) != 0) {
    core_ = (unsigned)CPUInfo[1] >> 24;
  }
#endif
#elif defined(__wasm__)
  core_ = emscripten_num_logical_cores();
#else
  core_ = sched_getcpu();
#endif
}

void ThreadPoolProfiler::MainThreadStat::LogBlockSize(std::ptrdiff_t block_size) {
  blocks_.emplace_back(block_size);
}

void ThreadPoolProfiler::MainThreadStat::LogStart() {
  points_.emplace_back(Clock::now());
}

void ThreadPoolProfiler::MainThreadStat::LogEnd(ThreadPoolEvent evt) {
  ORT_ENFORCE(!points_.empty(), "LogStart must pair with LogEnd");
  events_[evt] += TimeDiffMicroSeconds(points_.back(), Clock::now());
  points_.pop_back();
}

void ThreadPoolProfiler::MainThreadStat::LogEndAndStart(ThreadPoolEvent evt) {
  ORT_ENFORCE(!points_.empty(), "LogStart must pair with LogEnd");
  events_[evt] += TimeDiffMicroSeconds(points_.back(), Clock::now());
  points_.back() = Clock::now();
}

std::string ThreadPoolProfiler::MainThreadStat::Reset() {
  ORT_ENFORCE(points_.empty(), "LogStart must pair with LogEnd");
  std::stringstream ss;
  ss << "\"thread_id\": \"" << std::this_thread::get_id() << "\", \"block_size\": [";
  if (!blocks_.empty()) {
    std::copy(blocks_.begin(), blocks_.end() - 1, std::ostream_iterator<std::ptrdiff_t>(ss, ", "));
    ss << blocks_.back();
    blocks_.clear();
  }
  ss << "], \"core\": " << core_ << ", ";
  for (int i = 0; i < MAX_EVENT; ++i) {
    ss << "\"" << ThreadPoolProfiler::GetEventName(static_cast<ThreadPoolEvent>(i))
       << "\": " << events_[i] << ((i == MAX_EVENT - 1) ? std::string{} : ", ");
  }
  memset(events_, 0, sizeof(uint64_t) * MAX_EVENT);
  return ss.str();
}

const char* ThreadPoolProfiler::GetEventName(ThreadPoolEvent event) {
  switch (event) {
    case DISTRIBUTION:
      return "Distribution";
    case DISTRIBUTION_ENQUEUE:
      return "DistributionEnqueue";
    case RUN:
      return "Run";
    case WAIT:
      return "Wait";
    case WAIT_REVOKE:
      return "WaitRevoke";
    default:
      return "UnknownEvent";
  }
}

void ThreadPoolProfiler::LogThreadId(int thread_idx) {
  child_thread_stats_[thread_idx].thread_id_ = std::this_thread::get_id();
}

void ThreadPoolProfiler::LogRun(int thread_idx) {
  if (enabled_) {
    child_thread_stats_[thread_idx].num_run_++;
    auto now = Clock::now();
    if (child_thread_stats_[thread_idx].core_ < 0 ||
        TimeDiffMicroSeconds(child_thread_stats_[thread_idx].last_logged_point_, now) > 10000) {
#ifdef _WIN32
      child_thread_stats_[thread_idx].core_ = GetCurrentProcessorNumber();
#elif defined(__APPLE__)
#if defined(__x86_64__) || defined(__i386__)
      uint32_t CPUInfo[4];
      __cpuid_count(1, 0, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
      if ((CPUInfo[3] & (1 << 9)) != 0) {
        child_thread_stats_[thread_idx].core_ = (unsigned)CPUInfo[1] >> 24;
      }
#endif
#elif defined(__wasm__)
      child_thread_stats_[thread_idx].core_ = emscripten_num_logical_cores();
#else
      child_thread_stats_[thread_idx].core_ = sched_getcpu();
#endif
      child_thread_stats_[thread_idx].last_logged_point_ = now;
    }
  }
}

std::string ThreadPoolProfiler::DumpChildThreadStat() {
  std::stringstream ss;
  for (int i = 0; i < num_threads_; ++i) {
    ss << "\"" << child_thread_stats_[i].thread_id_ << "\": {"
       << "\"num_run\": " << child_thread_stats_[i].num_run_ << ", "
       << "\"core\": " << child_thread_stats_[i].core_ << "}"
       << (i == num_threads_ - 1 ? "" : ",");
  }
  return ss.str();
}
#endif

// A sharded loop counter distributes loop iterations between a set of worker threads.  The iteration space of
// the loop is divided (perhaps unevenly) between the shards.  Each thread has a home shard (perhaps not uniquely
// to it), and it claims iterations via atomic operations on its home shard.  It then proceeds through the other
// shards until all of the shards' iterations are complete.  This approach serves two purposes.  First, compared
// with atomic operations on a single counter, it reduces contention on the counter in the case of loops with
// large numbers of short-running iteration.  Second, by having a thread work on its home shard initially, it
// promotes affinity between the work that a thread performs in one loop and the work that it performs in the next.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324) /* Padding added to LoopCounterShard, LoopCounter for alignment */
#endif

static constexpr int CACHE_LINE_BYTES = 64;
static constexpr unsigned MAX_SHARDS = 8;

#ifndef _OPENMP
static constexpr int TaskGranularityFactor = 4;
#endif

struct alignas(CACHE_LINE_BYTES) LoopCounterShard {
  ::std::atomic<uint64_t> _next{0};
  uint64_t _end{0};
};

static_assert(sizeof(LoopCounterShard) == CACHE_LINE_BYTES, "Expected loop counter shards to match cache-line size");

class alignas(CACHE_LINE_BYTES) LoopCounter {
 public:
  LoopCounter(uint64_t num_iterations,
              uint64_t d_of_p,
              uint64_t block_size = 1) : _block_size(block_size),
                                         _num_shards(GetNumShards(num_iterations,
                                                                  d_of_p,
                                                                  block_size)) {
    // Divide the iteration space between the shards.  If the iteration
    // space does not divide evenly into shards of multiples of
    // block_size then the final shard is left uneven.

    auto num_blocks = num_iterations / block_size;
    auto blocks_per_shard = num_blocks / _num_shards;
    auto iterations_per_shard = blocks_per_shard * block_size;

    for (uint64_t shard = 0; shard < _num_shards; shard++) {
      // Initialize with a relaxed store; synchronization with worker
      // threads is provided via the thread pool
      _shards[shard]._next.store(shard * iterations_per_shard,
                                 ::std::memory_order_relaxed);

      bool is_last_shard = (shard == _num_shards - 1);
      _shards[shard]._end = is_last_shard ? num_iterations : ((shard + 1) * iterations_per_shard);
    }
  }

  // Allocate each thread to a home shard, from which it starts
  // claiming iterations.
  //
  // We use the worker ID provided by the thread pool as the basis of
  // this allocation.  Doing so promotes locality between successive
  // loops: the worker that runs a given iteration in one loop will
  // tend to run the same iterations in the next loop.  This helps
  // operators with a series of short loops, such as GRU.

  unsigned GetHomeShard(unsigned idx) const {
    return idx % _num_shards;
  }

  // Attempt to claim iterations from the sharded counter.  The function either
  // returns true, along with a block of exactly block_size iterations, or it returns false
  // if all of the iterations have been claimed.
  bool ClaimIterations(unsigned my_home_shard,
                       unsigned& my_shard,
                       uint64_t& my_start,
                       uint64_t& my_end) {
    do {
      if (_shards[my_shard]._next < _shards[my_shard]._end) {
        // Appears to be work in the current shard, try to claim with atomic fetch-and-add
        uint64_t temp_start = _shards[my_shard]._next.fetch_add(_block_size);
        if (temp_start < _shards[my_shard]._end) {
          my_start = temp_start;
          my_end = std::min(_shards[my_shard]._end, temp_start + _block_size);
          return true;
        }
      }
      // Work in the current shard is exhausted, move to the next shard, until
      // we are back at the home shard.
      my_shard = (my_shard + 1) % _num_shards;
    } while (my_shard != my_home_shard);
    return false;
  }

 private:
  // Derive the number of shards to use for a given loop.  We require
  // at least one block of work per shard, and subject to the
  // constraints:
  //
  // - We use no more than MAX_SHARDS (limiting the amount of space needed
  //   for the LoopCounter, and work needed to confirm that all shards have been
  //   completed at the end of a loop).
  //
  // - The number of shards is <= the number of threads (d_of_p).
  //   Hence, at low thread counts, each of N threads will get its own
  //   shard representing 1/N of the work.
  static unsigned GetNumShards(uint64_t num_iterations,
                               uint64_t d_of_p,
                               uint64_t block_size) {
    unsigned num_shards;
    auto num_blocks = num_iterations / block_size;
    if (num_blocks == 0) {
      num_shards = 1;
    } else if (num_blocks < MAX_SHARDS) {
      num_shards = static_cast<unsigned>(num_blocks);
    } else {
      num_shards = MAX_SHARDS;
    }
    if (num_shards > d_of_p) {
      num_shards = static_cast<unsigned>(d_of_p);
    }
    return num_shards;
  }

  alignas(CACHE_LINE_BYTES) LoopCounterShard _shards[MAX_SHARDS];
  const uint64_t _block_size;
  const unsigned _num_shards;
};

#ifdef _MSC_VER
#pragma warning(pop) /* Padding added in LoopCounterShard, LoopCounter */
#endif

ThreadPool::ThreadPool(Env* env,
                       const ThreadOptions& thread_options,
                       const NAME_CHAR_TYPE* name,
                       int degree_of_parallelism,
                       bool low_latency_hint)
    : thread_options_(thread_options) {
  // In the current implementation, a thread pool with degree_of_parallelism==1 uses
  // the caller as one of the threads for executing work.  Hence we only create
  // additional thread(s) for degree_of_parallelism>=2.
  assert(degree_of_parallelism >= 1);
  if (degree_of_parallelism >= 2) {
    int threads_to_create = degree_of_parallelism - 1;
    extended_eigen_threadpool_ =
        std::make_unique<ThreadPoolTempl<Env> >(name,
                                                        threads_to_create,
                                                        low_latency_hint,
                                                        *env,
                                                        thread_options_);
    underlying_threadpool_ = extended_eigen_threadpool_.get();
  }
}

ThreadPool::ThreadPool() {}

ThreadPool::~ThreadPool() = default;

// Base case for parallel loops, running iterations 0..total, divided into blocks
// of block_size iterations, and calling into a function that takes a start..end
// range of indices to run.
void ThreadPool::ParallelForFixedBlockSizeScheduling(const std::ptrdiff_t total,
                                                     const std::ptrdiff_t block_size,
                                                     const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
  if (total <= 0)
    return;

  if (total <= block_size) {
    fn(0, total);
    return;
  }

  // Split the work across threads in the pool.  Each work item will run a loop claiming iterations,
  // hence we need at most one for each thread, even if the numberof blocks of iterations is larger.
  auto d_of_p = DegreeOfParallelism(this);
  auto num_blocks = total / block_size;
  int num_work_items = static_cast<int>(std::min(static_cast<std::ptrdiff_t>(d_of_p), num_blocks));
  assert(num_work_items > 0);

  LoopCounter lc(total, d_of_p, block_size);
  std::function<void(unsigned)> run_work = [&](unsigned idx) {
    unsigned my_home_shard = lc.GetHomeShard(idx);
    unsigned my_shard = my_home_shard;
    uint64_t my_iter_start, my_iter_end;
    while (lc.ClaimIterations(my_home_shard, my_shard, my_iter_start, my_iter_end)) {
      fn(static_cast<std::ptrdiff_t>(my_iter_start),
         static_cast<std::ptrdiff_t>(my_iter_end));
    }
  };

  // Run the work in the thread pool (and in the current thread).  Synchronization with helping
  // threads is handled within RunInParallel, hence we can deallocate lc and other state captured by
  // run_work.
  RunInParallel(run_work, num_work_items, block_size);
}

void ThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
  ParallelForFixedBlockSizeScheduling(total, 1, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
    for (std::ptrdiff_t idx = first; idx < last; idx++) {
      fn(idx);
    }
  });
}

void ThreadPool::Schedule(std::function<void()> fn) {
  if (underlying_threadpool_) {
    underlying_threadpool_->Schedule(std::move(fn));
  } else {
    fn();
  }
}

void ThreadPool::StartProfiling() {
  if (underlying_threadpool_) {
    underlying_threadpool_->StartProfiling();
  }
}

std::string ThreadPool::StopProfiling() {
  if (underlying_threadpool_) {
    return underlying_threadpool_->StopProfiling();
  } else {
    return {};
  }
}

thread_local ThreadPool::ParallelSection* ThreadPool::ParallelSection::current_parallel_section{nullptr};

ThreadPool::ParallelSection::ParallelSection(ThreadPool* tp) {
#ifdef _OPENMP
  // Nothing
  ORT_UNUSED_PARAMETER(tp);
#else
  ORT_ENFORCE(!current_parallel_section, "Nested parallelism not supported");
  ORT_ENFORCE(!ps_.get());
  tp_ = tp;
  if (tp && tp->underlying_threadpool_) {
    ps_ = tp->underlying_threadpool_->AllocateParallelSection();
    tp_->underlying_threadpool_->StartParallelSection(*ps_.get());
    current_parallel_section = this;
  }
#endif
}

ThreadPool::ParallelSection::~ParallelSection() {
#ifdef _OPENMP
  // Nothing
#else
  if (current_parallel_section) {
    tp_->underlying_threadpool_->EndParallelSection(*ps_.get());
    ps_.reset();
    current_parallel_section = nullptr;
  }
#endif
}

void ThreadPool::RunInParallel(std::function<void(unsigned idx)> fn, unsigned n, std::ptrdiff_t block_size) {
  if (underlying_threadpool_) {
    if (ThreadPool::ParallelSection::current_parallel_section) {
      underlying_threadpool_->RunInParallelSection(*(ThreadPool::ParallelSection::current_parallel_section->ps_.get()),
                                                   std::move(fn),
                                                   n, block_size);
    } else {
      underlying_threadpool_->RunInParallel(std::move(fn),
                                            n, block_size);
    }
  } else {
    fn(0);
  }
}

bool ThreadPool::ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                                       const std::ptrdiff_t block_size) const {
  // Do not parallelize trivial loops, with only a single block of work
  if (block_size <= 0 || num_iterations <= block_size) {
    return false;
  }

  // Do not parallelize loops with only a single thread available.  If the
  // caller is outside the current pool (ID == -1) then we parallelize
  // if the pool has any threads.  If the caller is inside the current pool
  // (ID != -1) then we require at least one additional thread in the pool.
  if ((CurrentThreadId() == -1 && NumThreads() == 0) ||
      (CurrentThreadId() != -1 && NumThreads() == 1)) {
    return false;
  }

  return true;
}

using CostModel = Eigen::TensorCostModel<Eigen::ThreadPoolDevice>;

  // Calculates block size based on (1) the iteration cost and (2) parallel
// efficiency. We want blocks to be not too small to mitigate parallelization
// overheads; not too large to mitigate tail effect and potential load
// imbalance and we also want number of blocks to be evenly dividable across
// threads.
static ptrdiff_t CalculateParallelForBlock(const ptrdiff_t n, const Eigen::TensorOpCost& cost,
                                           std::function<ptrdiff_t(ptrdiff_t)> block_align, int num_threads) {
  const double block_size_f = 1.0 / CostModel::taskSize(1, cost);
  const ptrdiff_t max_oversharding_factor = 4;
  ptrdiff_t block_size = Eigen::numext::mini(
      n,
      Eigen::numext::maxi<ptrdiff_t>(Eigen::divup<ptrdiff_t>(n, max_oversharding_factor * num_threads), static_cast<ptrdiff_t>(block_size_f)));
  const ptrdiff_t max_block_size = Eigen::numext::mini(n, 2 * block_size);

  if (block_align) {
    ptrdiff_t new_block_size = block_align(block_size);
    assert(new_block_size >= block_size);
    block_size = Eigen::numext::mini(n, new_block_size);
  }

  ptrdiff_t block_count = Eigen::divup(n, block_size);

  // Calculate parallel efficiency as fraction of total CPU time used for
  // computations:
  double max_efficiency =
      static_cast<double>(block_count) / (Eigen::divup<ptrdiff_t>(block_count, num_threads) * num_threads);

  // Now try to increase block size up to max_block_size as long as it
  // doesn't decrease parallel efficiency.
  for (ptrdiff_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
    // This is the next block size that divides size into a smaller number
    // of blocks than the current block_size.
    ptrdiff_t coarser_block_size = Eigen::divup(n, prev_block_count - 1);
    if (block_align) {
      ptrdiff_t new_block_size = block_align(coarser_block_size);
      assert(new_block_size >= coarser_block_size);
      coarser_block_size = Eigen::numext::mini(n, new_block_size);
    }
    if (coarser_block_size > max_block_size) {
      break;  // Reached max block size. Stop.
    }
    // Recalculate parallel efficiency.
    const ptrdiff_t coarser_block_count = Eigen::divup(n, coarser_block_size);
    assert(coarser_block_count < prev_block_count);
    prev_block_count = coarser_block_count;
    const double coarser_efficiency =
        static_cast<double>(coarser_block_count) / (Eigen::divup<ptrdiff_t>(coarser_block_count, num_threads) * num_threads);
    if (coarser_efficiency + 0.01 >= max_efficiency) {
      // Taking it.
      block_size = coarser_block_size;
      if (max_efficiency < coarser_efficiency) {
        max_efficiency = coarser_efficiency;
      }
    }
  }

  return block_size;
}

ptrdiff_t GetBlockSize(ptrdiff_t n, const TensorOpCost& c, int num_threads) {
  Eigen::TensorOpCost cost{c.bytes_loaded, c.bytes_stored, c.compute_cycles};
  return CalculateParallelForBlock(n, cost, nullptr, num_threads);
}

void ThreadPool::ParallelFor(std::ptrdiff_t n, const TensorOpCost& c,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& f) {
  ORT_ENFORCE(n >= 0);
  Eigen::TensorOpCost cost{c.bytes_loaded, c.bytes_stored, c.compute_cycles};
  auto d_of_p = DegreeOfParallelism(this);
  // Compute small problems directly in the caller thread.
  if ((!ShouldParallelizeLoop(n)) ||
      CostModel::numThreads(static_cast<double>(n), cost, d_of_p) == 1) {
    f(0, n);
    return;
  }

  ptrdiff_t block = CalculateParallelForBlock(n, cost, nullptr, d_of_p);
  ParallelForFixedBlockSizeScheduling(n, block, f);
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

bool ThreadPool::ShouldParallelize(const concurrency::ThreadPool* tp) {
  return (DegreeOfParallelism(tp) != 1);
}

int ThreadPool::DegreeOfParallelism(const concurrency::ThreadPool* tp) {
#ifdef _OPENMP
  // When using OpenMP, omp_get_num_threads() returns the number of threads in the
  // current parallel region.  Hence if this is 1 then we aim to parallelise
  // across the number of threads configured.  Otherwise, given that we do not
  // use nested parallelism, we do not parallelise further.
  ORT_UNUSED_PARAMETER(tp);
  return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#else
  // When not using OpenMP, we parallelise over the N threads created by the pool
  // tp, plus 1 for the thread entering a loop.
  if (tp) {
    if (CPUIDInfo::GetCPUIDInfo().IsHybrid()) {
      return ((tp->NumThreads() + 1)) * TaskGranularityFactor;
    } else {
      return ((tp->NumThreads() + 1));
    }
  } else {
    return 1;
  }
#endif
}

void ThreadPool::StartProfiling(concurrency::ThreadPool* tp) {
  if (tp) {
    tp->StartProfiling();
  }
}

std::string ThreadPool::StopProfiling(concurrency::ThreadPool* tp) {
  if (tp) {
    return tp->StopProfiling();
  } else {
    return {};
  }
}

// Return the number of threads created by the pool.
int ThreadPool::NumThreads() const {
  if (underlying_threadpool_) {
    return underlying_threadpool_->NumThreads();
  } else {
    return 0;
  }
}

// Return ID of the current thread within this pool.  Returns -1 for a thread outside the
// current pool.
int ThreadPool::CurrentThreadId() const {
  if (underlying_threadpool_) {
    return underlying_threadpool_->CurrentThreadId();
  } else {
    return -1;
  }
}

void ThreadPool::TryParallelFor(concurrency::ThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                                const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
#ifdef _OPENMP
  ORT_ENFORCE(total >= 0);
  if (total == 0) {
    return;
  }

  if (total == 1) {
    fn(0, 1);
    return;
  }

  Eigen::TensorOpCost cost{cost_per_unit.bytes_loaded, cost_per_unit.bytes_stored, cost_per_unit.compute_cycles};
  auto d_of_p = DegreeOfParallelism(tp);
  std::ptrdiff_t num_threads = CostModel::numThreads(static_cast<double>(total), cost, d_of_p);

  if (total < num_threads) {
    num_threads = total;
  }

  if (num_threads == 1) {
    fn(0, total);
    return;
  }

  ptrdiff_t block_size = CalculateParallelForBlock(total, cost, nullptr, d_of_p);
  ptrdiff_t block_count = Eigen::divup(total, block_size);

  if (block_count == 1) {
    fn(0, total);
    return;
  }

#pragma omp parallel for schedule(dynamic, 1)
  for (std::ptrdiff_t i = 0; i < block_count; i++) {
    const auto start = i * block_size;
    fn(start, std::min(start + block_size, total));
  }
#else  //!_OPENMP
  if (tp == nullptr) {
    fn(0, total);
    return;
  }
  tp->ParallelFor(total, cost_per_unit, fn);
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
