// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if defined(__linux__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#endif

namespace onnxruntime::contrib::cuda::qmoe {

#if defined(__linux__)
inline void ParseLinuxCpuRangeList(const std::string& text, std::vector<int>* out) {
  size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == ',')) {
      ++pos;
    }
    if (pos >= text.size()) break;
    size_t end = pos;
    while (end < text.size() && text[end] != ',' && text[end] != '\n') ++end;
    const std::string token = text.substr(pos, end - pos);
    const size_t dash = token.find('-');
    try {
      if (dash == std::string::npos) {
        out->push_back(std::stoi(token));
      } else {
        const int lo = std::stoi(token.substr(0, dash));
        const int hi = std::stoi(token.substr(dash + 1));
        for (int v = lo; v <= hi; ++v) out->push_back(v);
      }
    } catch (...) {
    }
    pos = end + 1;
  }
}

inline std::vector<int> DiscoverLinuxMemoryNodes() {
  std::vector<int> nodes;
  std::ifstream in("/sys/devices/system/node/has_memory");
  if (!in) in.open("/sys/devices/system/node/online");
  if (in) {
    std::string line;
    std::getline(in, line);
    ParseLinuxCpuRangeList(line, &nodes);
  }
  return nodes;
}

inline std::vector<cpu_set_t> DiscoverAllowedNumaNodeCpuSets() {
  std::vector<cpu_set_t> node_sets;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed) != 0) {
    return node_sets;
  }
  const std::vector<int> nodes = DiscoverLinuxMemoryNodes();
  for (int node : nodes) {
    std::ifstream in("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    if (!in) continue;
    std::string line;
    std::getline(in, line);
    std::vector<int> cpus;
    ParseLinuxCpuRangeList(line, &cpus);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int count = 0;
    for (int cpu : cpus) {
      if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
        CPU_SET(cpu, &mask);
        ++count;
      }
    }
    if (count > 0) {
      node_sets.push_back(mask);
    }
  }
  return node_sets;
}

// Prefer one logical CPU per physical core, alternating NUMA nodes. SMT siblings
// are appended only after all permitted physical cores have been assigned.
inline std::vector<int> NumaBalancedCpuOrder(const std::vector<cpu_set_t>& node_sets) {
  std::vector<std::vector<int>> primary(node_sets.size()), siblings(node_sets.size());
  cpu_set_t seen;
  CPU_ZERO(&seen);
  for (size_t n = 0; n < node_sets.size(); ++n) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (!CPU_ISSET(cpu, &node_sets[n])) continue;
      if (CPU_ISSET(cpu, &seen)) {
        siblings[n].push_back(cpu);
        continue;
      }
      primary[n].push_back(cpu);
      CPU_SET(cpu, &seen);
      std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
      std::string line;
      std::vector<int> same_core;
      if (std::getline(in, line)) ParseLinuxCpuRangeList(line, &same_core);
      for (int sibling : same_core)
        if (sibling >= 0 && sibling < CPU_SETSIZE) CPU_SET(sibling, &seen);
    }
  }
  std::vector<int> order;
  for (const auto* sets : {&primary, &siblings}) {
    for (size_t i = 0;; ++i) {
      bool any = false;
      for (const auto& cpus : *sets) {
        if (i < cpus.size()) {
          order.push_back(cpus[i]);
          any = true;
        }
      }
      if (!any) break;
    }
  }
  return order;
}

class ScopedCpuAffinity {
 public:
  explicit ScopedCpuAffinity(int cpu) noexcept {
    if (cpu < 0 || cpu >= CPU_SETSIZE || pthread_getaffinity_np(pthread_self(), sizeof(saved_), &saved_) != 0 ||
        !CPU_ISSET(cpu, &saved_)) return;
    if (CPU_COUNT(&saved_) == 1) return;
    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_SET(cpu, &target);
    restore_ = pthread_setaffinity_np(pthread_self(), sizeof(target), &target) == 0;
  }
  ~ScopedCpuAffinity() {
    if (restore_) static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(saved_), &saved_));
  }
  ScopedCpuAffinity(const ScopedCpuAffinity&) = delete;
  ScopedCpuAffinity& operator=(const ScopedCpuAffinity&) = delete;

 private:
  cpu_set_t saved_{};
  bool restore_{};
};
#elif defined(_WIN32)
inline std::vector<int> DiscoverWindowsBalancedCpuOrder(int* out_l3_domains = nullptr) {
  DWORD_PTR proc_mask = 0, sys_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
    proc_mask = ~DWORD_PTR{0};
  }

  // Discover L3 cache domains (CCDs on AMD Ryzen / NUMA clusters) in processor group 0.
  std::vector<KAFFINITY> l3_masks;
  DWORD l3_bytes = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &l3_bytes);
  if (l3_bytes > 0) {
    std::vector<uint8_t> buf(l3_bytes);
    if (GetLogicalProcessorInformationEx(
            RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &l3_bytes)) {
      size_t offset = 0;
      while (offset < l3_bytes) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        if (info->Size == 0) break;
        if (info->Relationship == RelationCache && info->Cache.Level == 3 && info->Cache.GroupCount >= 1 &&
            info->Cache.GroupMasks[0].Group == 0) {
          const KAFFINITY m = info->Cache.GroupMasks[0].Mask & proc_mask;
          if (m != 0 && std::find(l3_masks.begin(), l3_masks.end(), m) == l3_masks.end()) {
            l3_masks.push_back(m);
          }
        }
        offset += info->Size;
      }
    }
  }
  if (l3_masks.empty()) {
    l3_masks.push_back(proc_mask);
  }
  if (out_l3_domains) *out_l3_domains = static_cast<int>(l3_masks.size());

  // Discover physical cores (RelationProcessorCore) and separate primary hardware threads from SMT siblings.
  std::vector<std::vector<int>> domain_primary(l3_masks.size());
  std::vector<std::vector<int>> domain_siblings(l3_masks.size());
  DWORD core_bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &core_bytes);
  if (core_bytes > 0) {
    std::vector<uint8_t> buf(core_bytes);
    if (GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &core_bytes)) {
      size_t offset = 0;
      while (offset < core_bytes) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        if (info->Size == 0) break;
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1 &&
            info->Processor.GroupMask[0].Group == 0) {
          const KAFFINITY mask = info->Processor.GroupMask[0].Mask & proc_mask;
          if (mask != 0) {
            size_t dom = 0;
            for (size_t d = 0; d < l3_masks.size(); ++d) {
              if ((l3_masks[d] & mask) != 0) {
                dom = d;
                break;
              }
            }
            bool first = true;
            for (int bit = 0; bit < static_cast<int>(sizeof(KAFFINITY) * 8); ++bit) {
              if ((mask & (KAFFINITY{1} << bit)) != 0) {
                if (first) {
                  domain_primary[dom].push_back(bit);
                  first = false;
                } else {
                  domain_siblings[dom].push_back(bit);
                }
              }
            }
          }
        }
        offset += info->Size;
      }
    }
  }

  // Rotate domain 0 so physical core 0 (typically used by the main/UI/WaitCycle thread) is assigned last among primary cores.
  if (!domain_primary.empty() && domain_primary[0].size() > 1) {
    std::rotate(domain_primary[0].begin(), domain_primary[0].begin() + 1, domain_primary[0].end());
  }

  std::vector<int> order;
  for (const auto* sets : {&domain_primary, &domain_siblings}) {
    for (size_t i = 0;; ++i) {
      bool any = false;
      for (const auto& cpus : *sets) {
        if (i < cpus.size()) {
          order.push_back(cpus[i]);
          any = true;
        }
      }
      if (!any) break;
    }
  }
  return order;
}
#endif

inline int DiscoverPhysicalCoreCount() {
#if defined(__linux__)
  const auto node_sets = DiscoverAllowedNumaNodeCpuSets();
  cpu_set_t seen;
  CPU_ZERO(&seen);
  int physical_cores = 0;
  for (size_t n = 0; n < node_sets.size(); ++n) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (!CPU_ISSET(cpu, &node_sets[n]) || CPU_ISSET(cpu, &seen)) continue;
      ++physical_cores;
      CPU_SET(cpu, &seen);
      std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
      std::string line;
      std::vector<int> same_core;
      if (std::getline(in, line)) ParseLinuxCpuRangeList(line, &same_core);
      for (int sibling : same_core) {
        if (sibling >= 0 && sibling < CPU_SETSIZE) CPU_SET(sibling, &seen);
      }
    }
  }
  if (physical_cores > 0) return physical_cores;
#elif defined(_WIN32)
  DWORD_PTR proc_mask = 0, sys_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
    proc_mask = ~DWORD_PTR{0};
  }
  DWORD core_bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &core_bytes);
  if (core_bytes > 0) {
    std::vector<uint8_t> buf(core_bytes);
    if (GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &core_bytes)) {
      int physical_cores = 0;
      size_t offset = 0;
      while (offset < core_bytes) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        if (info->Size == 0) break;
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1 &&
            info->Processor.GroupMask[0].Group == 0 && (info->Processor.GroupMask[0].Mask & proc_mask) != 0) {
          ++physical_cores;
        }
        offset += info->Size;
      }
      if (physical_cores > 0) {
        return (physical_cores >= 8) ? (physical_cores - 1) : physical_cores;
      }
    }
  }
#endif
  return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
}

inline size_t SpinPoolBatchSize(size_t count, size_t threads, size_t requested) {
  return std::min(requested, std::max(size_t{1}, count / threads));
}

// Unified bounded CPU expert service (v3 Section 8.3 / F9).
// Used by both synchronous and asynchronous (Option A stream callback) execution paths.
// Generation-tagged claims prevent late workers from executing a later job with stale metadata.
class SpinPool final {
 public:
  explicit SpinPool(int threads, int spin_us = 2000) : spin_us_(spin_us) {
    if (threads < 1) throw std::invalid_argument("QMoE pool threads must be positive.");
#if defined(__linux__)
    const char* pin_env = std::getenv("ORT_QMOE_NUMA_PIN_WORKERS");
    const bool enable_pin = (pin_env == nullptr || std::strcmp(pin_env, "0") != 0);
    if (enable_pin) {
      node_masks_ = DiscoverAllowedNumaNodeCpuSets();
      worker_cpus_ = NumaBalancedCpuOrder(node_masks_);
    }
    numa_node_count_ = static_cast<int>(node_masks_.size());
    if (const char* log = std::getenv("ORT_QMOE_NUMA_DIAGNOSTICS"); log && std::strcmp(log, "0") != 0) {
      std::string cpus;
      for (int i = 0; i < threads; ++i) {
        if (i > 0) cpus += ',';
        cpus += std::to_string(CpuForWorker(i));
      }
      std::fprintf(stderr, "[QMoE NUMA] pool_threads=%d nodes=%d caller_then_worker_cpus=%s\n",
                   threads, numa_node_count_, cpus.c_str());
    }
#elif defined(_WIN32)
    const char* pin_env = std::getenv("ORT_QMOE_NUMA_PIN_WORKERS");
    const bool enable_pin = (pin_env == nullptr || std::strcmp(pin_env, "0") != 0);
    if (enable_pin) {
      worker_cpus_ = DiscoverWindowsBalancedCpuOrder(&numa_node_count_);
    }
    if (const char* log = std::getenv("ORT_QMOE_NUMA_DIAGNOSTICS"); log && std::strcmp(log, "0") != 0) {
      std::string cpus;
      for (int i = 0; i < threads; ++i) {
        if (i > 0) cpus += ',';
        cpus += std::to_string(CpuForWorker(i));
      }
      std::fprintf(stderr, "[QMoE WinCCD] pool_threads=%d l3_domains=%d caller_then_worker_cpus=%s\n",
                   threads, numa_node_count_, cpus.c_str());
    }
#endif
    for (int worker = 1; worker < threads; ++worker) {
      workers_.emplace_back([this, worker] { Loop(worker); });
    }
  }

  ~SpinPool() {
    stop_.store(true);
    {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      wake_.notify_all();
    }
    for (auto& worker : workers_) worker.join();
  }

  void SetSpinDurationUs(int spin_us) noexcept {
    spin_us_.store(std::max(0, spin_us), std::memory_order_relaxed);
  }

  int SpinDurationUs() const noexcept {
    return spin_us_.load(std::memory_order_relaxed);
  }

  int ThreadCount() const noexcept {
    return static_cast<int>(workers_.size()) + 1;
  }

  int NumaNodeCount() const noexcept {
    return numa_node_count_;
  }

  int PinnedWorkerCount() const noexcept {
    return pinned_workers_.load(std::memory_order_acquire);
  }

  int CpuForWorker(int worker) const noexcept {
#if defined(__linux__) || defined(_WIN32)
    return worker >= 0 && static_cast<size_t>(worker) < worker_cpus_.size() ? worker_cpus_[worker] : -1;
#else
    (void)worker;
    return -1;
#endif
  }

  int AffinityFailures() const noexcept { return affinity_failures_.load(std::memory_order_acquire); }

  template <typename Function>
  void ParallelFor(ptrdiff_t count, Function& function, int batch = 1) {
    if (count < 0 || batch < 1 || batch > 16) throw std::invalid_argument("Invalid QMoE pool job/batch.");
    Run(static_cast<size_t>(count), [](void* data, size_t task) { (*static_cast<Function*>(data))(static_cast<ptrdiff_t>(task)); }, &function, static_cast<size_t>(batch));
  }

 private:
  static constexpr uint64_t kFieldBits = 20;
  static constexpr uint64_t kFieldMask = (uint64_t{1} << kFieldBits) - 1;

  static uint64_t Pack(uint64_t generation, uint64_t count, uint64_t next) {
    return (generation << (2 * kFieldBits)) | (count << kFieldBits) | next;
  }
  static bool HasWork(uint64_t ticket) { return (ticket & kFieldMask) < ((ticket >> kFieldBits) & kFieldMask); }

  void Run(size_t count, void (*function)(void*, size_t), void* data, size_t requested_batch) {
    if (count == 0) return;
#if defined(__linux__)
    // The submitting thread participates as worker zero; restore arbitrary ORT
    // callers' affinity when the job ends instead of permanently pinning them.
    ScopedCpuAffinity caller_affinity(CpuForWorker(0));
#endif
    if (workers_.empty() || count > kFieldMask) {
      for (size_t task = 0; task < count; ++task) function(data, task);
      return;
    }
    std::lock_guard<std::mutex> serial(call_mutex_);
    function_ = function;
    data_ = data;
    batch_atomic_.store(SpinPoolBatchSize(count, workers_.size() + 1, requested_batch), std::memory_order_relaxed);
    completed_.store(0, std::memory_order_relaxed);
    ++generation_;
    ticket_.store(Pack(generation_, count, 0), std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      wake_.notify_all();
    }
    RunTasks();
    while (completed_.load(std::memory_order_acquire) != count) Pause();
  }

  void RunTasks() {
    uint64_t ticket = ticket_.load(std::memory_order_acquire);
    while (HasWork(ticket)) {
      const uint64_t next = ticket & kFieldMask;
      const uint64_t count = (ticket >> kFieldBits) & kFieldMask;
      const uint64_t batch = std::min(count - next, batch_atomic_.load(std::memory_order_relaxed));
      if (ticket_.compare_exchange_weak(ticket, ticket + batch, std::memory_order_acq_rel, std::memory_order_acquire)) {
        for (uint64_t offset = 0; offset < batch; ++offset) function_(data_, static_cast<size_t>(next + offset));
        completed_.fetch_add(static_cast<size_t>(batch), std::memory_order_release);
        ticket = ticket_.load(std::memory_order_acquire);
      }
    }
  }

  static void Pause() {
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) && defined(__GNUC__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
  }

  void Loop(int worker_idx) {
#if defined(__linux__)
    const int cpu = CpuForWorker(worker_idx);
    if (cpu >= 0) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(cpu, &mask);
      if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0) {
        pinned_workers_.fetch_add(1, std::memory_order_release);
      } else {
        affinity_failures_.fetch_add(1, std::memory_order_release);
      }
    }
#elif defined(_WIN32)
    const int cpu = CpuForWorker(worker_idx);
    if (cpu >= 0 && cpu < static_cast<int>(sizeof(DWORD_PTR) * 8)) {
      if (SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) != 0) {
        pinned_workers_.fetch_add(1, std::memory_order_release);
      } else {
        affinity_failures_.fetch_add(1, std::memory_order_release);
      }
    }
#else
    (void)worker_idx;
#endif
    while (!stop_.load(std::memory_order_relaxed)) {
      RunTasks();
      auto idle_since = std::chrono::steady_clock::now();
      for (uint32_t spins = 1; !stop_.load(std::memory_order_relaxed); ++spins) {
        if (HasWork(ticket_.load(std::memory_order_acquire))) break;
        Pause();
        const int cur_spin_us = spin_us_.load(std::memory_order_relaxed);
        if ((spins & 0x3ff) == 0 &&
            std::chrono::steady_clock::now() - idle_since > std::chrono::microseconds(cur_spin_us)) {
          std::unique_lock<std::mutex> lock(sleep_mutex_);
          sleepers_.fetch_add(1, std::memory_order_seq_cst);
          wake_.wait(lock, [this] { return stop_.load() || HasWork(ticket_.load(std::memory_order_seq_cst)); });
          sleepers_.fetch_sub(1, std::memory_order_seq_cst);
          idle_since = std::chrono::steady_clock::now();
        }
      }
    }
  }

  std::atomic<int> spin_us_{2000};
  int numa_node_count_{0};
  std::atomic<int> pinned_workers_{0};
  std::atomic<int> affinity_failures_{0};
#if defined(__linux__)
  std::vector<cpu_set_t> node_masks_;
  std::vector<int> worker_cpus_;
#elif defined(_WIN32)
  std::vector<int> worker_cpus_;
#endif
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> ticket_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<int> sleepers_{0};
  std::atomic<uint64_t> batch_atomic_{1};
  std::mutex call_mutex_;
  std::mutex sleep_mutex_;
  std::condition_variable wake_;
  uint64_t generation_ = 0;
  void (*function_)(void*, size_t) = nullptr;
  void* data_ = nullptr;
};

// Host CPU budget group coordinating SpinPool spin policy across prefill/embedding/Engram and decode phases (v3 Section 8.3 / F9).
class CpuBudgetGroup final {
 public:
  enum class Phase : int {
    kDecode = 0,
    kPrefillOrAux = 1,
  };

  static CpuBudgetGroup& Default() {
    static CpuBudgetGroup instance;
    return instance;
  }

  void ConfigureDecodeSpinUs(int spin_us) noexcept {
    decode_spin_us_.store(std::max(0, spin_us), std::memory_order_relaxed);
  }

  void EnterPrefillOrAuxPhase(SpinPool* pool) noexcept {
    active_phase_.store(static_cast<int>(Phase::kPrefillOrAux), std::memory_order_release);
    if (pool) {
      pool->SetSpinDurationUs(0);
    }
  }

  void EnterDecodePhase(SpinPool* pool) noexcept {
    active_phase_.store(static_cast<int>(Phase::kDecode), std::memory_order_release);
    if (pool) {
      pool->SetSpinDurationUs(decode_spin_us_.load(std::memory_order_relaxed));
    }
  }

  Phase CurrentPhase() const noexcept {
    return static_cast<Phase>(active_phase_.load(std::memory_order_acquire));
  }

 private:
  std::atomic<int> decode_spin_us_{2000};
  std::atomic<int> active_phase_{static_cast<int>(Phase::kDecode)};
};

}  // namespace onnxruntime::contrib::cuda::qmoe
