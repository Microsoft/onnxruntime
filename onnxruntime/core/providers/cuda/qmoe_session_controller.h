// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Session-scoped QMoE Offload Controller (v3 Design Sections 5.1, 7.1, 7.6, 7.7, 8, 9).
// Owned per CUDA EP / session instance so plugin and built-in EPs share the exact same
// lifecycle, completion, landing-ring, CPU service, and observation commit contracts without
// crossing the C ABI with core-internal types.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")
#ifdef interface
#undef interface
#endif
#ifdef small
#undef small
#endif
#ifdef ERROR
#undef ERROR
#endif
#ifdef GetMessage
#undef GetMessage
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#endif
#endif

#include <cuda_runtime.h>

#ifndef SHARED_PROVIDER
#include "core/common/status.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/common/logging/logging.h"
#endif
#endif
#include "core/framework/moe_offload_plan.h"
#include "core/providers/cuda/bounded_pinned_pool.h"
#include "core/providers/cuda/cuda_device_memory_budget.h"

namespace onnxruntime::cuda {

using QMoEDeviceArenasShrunkHook = void (*)(int device_id) noexcept;

inline std::atomic<QMoEDeviceArenasShrunkHook>& GetQMoEDeviceArenasShrunkHook() noexcept {
  static std::atomic<QMoEDeviceArenasShrunkHook> hook{nullptr};
  return hook;
}

inline void SetQMoEDeviceArenasShrunkHook(QMoEDeviceArenasShrunkHook fn) noexcept {
  GetQMoEDeviceArenasShrunkHook().store(fn, std::memory_order_release);
}

inline void NotifyQMoEDeviceArenasShrunk(int device_id) noexcept {
  if (auto fn = GetQMoEDeviceArenasShrunkHook().load(std::memory_order_acquire)) {
    fn(device_id);
  }
}

// Per-invocation completion & observation record (v3 Section 7.1).
struct QMoEInvocationRecord {
  uint64_t run_id = 0;
  uint64_t invocation_id = 0;
  int64_t layer_index = -1;
  std::string node_name;

  std::mutex mutex;
  std::condition_variable cv;
  bool cpu_done = false;
  bool completed = false;
  bool failed = false;
  std::string error_message;

  // Optional callback invoked when the run commits its observations upon successful completion.
  std::function<void()> commit_observation;
  // Optional drain callback invoked during OnRunEnd if the invocation is still in flight.
  std::function<common::Status()> drain_device;
  cudaStream_t main_stream = nullptr;
  bool has_main_stream = false;

  void MarkFailed(std::string msg) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!failed) {
      failed = true;
      error_message = std::move(msg);
    }
    cpu_done = true;
    cv.notify_all();
  }

  void MarkCpuDone() {
    std::lock_guard<std::mutex> lock(mutex);
    cpu_done = true;
    cv.notify_all();
  }

  void MarkCompleted() {
    std::lock_guard<std::mutex> lock(mutex);
    cpu_done = true;
    completed = true;
    cv.notify_all();
  }
};

// Session-scoped landing ring with per-bank, per-generation event ordering (v3 Section 7.6 / F5 & F4).
struct QMoESessionLandingRing {
  std::mutex mutex;
  int device_id = -1;
  std::shared_ptr<SessionBudgetAccount> budget_account;
  std::shared_ptr<CudaDeviceMemoryBudget> session_pinned_budget;
  uint8_t* landing = nullptr;
  size_t landing_bytes = 0;
  uint8_t* pinned = nullptr;
  size_t pinned_bytes = 0;
  cudaStream_t copy_stream = nullptr;
  std::vector<cudaEvent_t> copy_done;
  std::vector<cudaEvent_t> gemm_done;

  // Per-bank (0, 1) cross-layer handoff events (F5):
  //   landing_ready[b]       recorded on copy_stream after H2D wave into bank b
  //   last_gemm_done[b]      recorded on compute stream after wave GEMM reads bank b
  //   promotion_read_done[b] recorded on promotion/compute stream after D2D promotion reads bank b
  cudaEvent_t landing_ready[2]{nullptr, nullptr};
  cudaEvent_t last_gemm_done[2]{nullptr, nullptr};
  cudaEvent_t promotion_read_done[2]{nullptr, nullptr};
  bool bank_has_gemm[2]{false, false};
  bool bank_has_promotion[2]{false, false};
  uint64_t bank_generation[2]{0, 0};

  ~QMoESessionLandingRing() {
    Shutdown();
  }

  void Shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    int prev_dev = -1;
    const bool restore_dev = (device_id >= 0 && cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != device_id);
    if (restore_dev) {
      static_cast<void>(cudaSetDevice(device_id));
    }
    if (copy_stream) {
      static_cast<void>(cudaStreamSynchronize(copy_stream));
    }
    for (int b = 0; b < 2; ++b) {
      if (landing_ready[b]) {
        static_cast<void>(cudaEventDestroy(landing_ready[b]));
        landing_ready[b] = nullptr;
      }
      if (last_gemm_done[b]) {
        static_cast<void>(cudaEventDestroy(last_gemm_done[b]));
        last_gemm_done[b] = nullptr;
      }
      if (promotion_read_done[b]) {
        static_cast<void>(cudaEventDestroy(promotion_read_done[b]));
        promotion_read_done[b] = nullptr;
      }
      bank_has_gemm[b] = false;
      bank_has_promotion[b] = false;
    }
    for (auto ev : copy_done) {
      if (ev) static_cast<void>(cudaEventDestroy(ev));
    }
    copy_done.clear();
    for (auto ev : gemm_done) {
      if (ev) static_cast<void>(cudaEventDestroy(ev));
    }
    gemm_done.clear();
    if (copy_stream) {
      static_cast<void>(cudaStreamDestroy(copy_stream));
      copy_stream = nullptr;
    }
    if (pinned) {
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      if (session_pinned_budget && pinned_bytes > 0) session_pinned_budget->BeginFree(pinned_bytes);
      if (pinned_budget && pinned_bytes > 0) pinned_budget->BeginFreeAllocated(pinned_bytes);
      const bool ok = (cudaFreeHost(pinned) == cudaSuccess);
      if (pinned_budget && pinned_bytes > 0) pinned_budget->FinishFreeAllocated(pinned_bytes, ok);
      if (session_pinned_budget && pinned_bytes > 0) session_pinned_budget->FinishFree(pinned_bytes, ok);
      pinned = nullptr;
      pinned_bytes = 0;
    }
    if (landing) {
      if (budget_account && landing_bytes > 0) {
        budget_account->BeginFree(landing_bytes);
        const bool ok = (cudaFree(landing) == cudaSuccess);
        budget_account->FinishFree(landing_bytes, ok);
      } else {
        auto dev_budget = (device_id >= 0) ? GetCudaDeviceMemoryBudgetByOrdinal(device_id) : nullptr;
        if (dev_budget && landing_bytes > 0) dev_budget->BeginFree(landing_bytes);
        const bool ok = (cudaFree(landing) == cudaSuccess);
        if (dev_budget && landing_bytes > 0) dev_budget->FinishFree(landing_bytes, ok);
      }
      landing = nullptr;
      landing_bytes = 0;
    }
    if (restore_dev) {
      static_cast<void>(cudaSetDevice(prev_dev));
    }
  }

  common::Status Ensure(int dev_id, size_t landing_needed, size_t pinned_needed, size_t waves) {
    device_id = dev_id;
    if (!copy_stream) {
      cudaError_t err = cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload failed to create copy_stream: ", cudaGetErrorString(err));
      }
    }
    for (int b = 0; b < 2; ++b) {
      if (!landing_ready[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&landing_ready[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
      if (!last_gemm_done[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&last_gemm_done[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
      if (!promotion_read_done[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&promotion_read_done[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
    }
    if (landing_needed > landing_bytes) {
      if (copy_stream) static_cast<void>(cudaStreamSynchronize(copy_stream));
      for (int b = 0; b < 2; ++b) {
        if (bank_has_gemm[b]) static_cast<void>(cudaEventSynchronize(last_gemm_done[b]));
        if (bank_has_promotion[b]) static_cast<void>(cudaEventSynchronize(promotion_read_done[b]));
        bank_has_gemm[b] = false;
        bank_has_promotion[b] = false;
      }
      auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(dev_id);
      SessionBudgetAccount::CombinedReservation acct_res;
      CudaDeviceMemoryBudget::Reservation res;
      try {
        if (budget_account) {
          acct_res = budget_account->Reserve(landing_needed);
        } else if (dev_budget) {
          res = dev_budget->Reserve(landing_needed);
        }
      } catch (const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing budget reservation failed: ", ex.what());
      }
      uint8_t* new_landing = nullptr;
      cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_landing), landing_needed);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing cudaMalloc failed: ", cudaGetErrorString(err));
      }
      if (budget_account) {
        acct_res.Commit();
      } else if (res.Active()) {
        res.Commit();
      }
      if (landing) {
        if (budget_account && landing_bytes > 0) {
          budget_account->BeginFree(landing_bytes);
          const bool ok = (cudaFree(landing) == cudaSuccess);
          budget_account->FinishFree(landing_bytes, ok);
        } else {
          if (dev_budget && landing_bytes > 0) dev_budget->BeginFree(landing_bytes);
          const bool ok = (cudaFree(landing) == cudaSuccess);
          if (dev_budget && landing_bytes > 0) dev_budget->FinishFree(landing_bytes, ok);
        }
      }
      landing = new_landing;
      landing_bytes = landing_needed;
    }
    if (pinned_needed > pinned_bytes) {
      if (copy_stream) static_cast<void>(cudaStreamSynchronize(copy_stream));
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      if (pinned) {
        if (session_pinned_budget && pinned_bytes > 0) session_pinned_budget->BeginFree(pinned_bytes);
        if (pinned_budget && pinned_bytes > 0) pinned_budget->BeginFreeAllocated(pinned_bytes);
        const bool ok = (cudaFreeHost(pinned) == cudaSuccess);
        if (pinned_budget && pinned_bytes > 0) pinned_budget->FinishFreeAllocated(pinned_bytes, ok);
        if (session_pinned_budget && pinned_bytes > 0) session_pinned_budget->FinishFree(pinned_bytes, ok);
        pinned = nullptr;
        pinned_bytes = 0;
      }
      CudaDeviceMemoryBudget::Reservation sess_res;
      CudaDeviceMemoryBudget::Reservation res;
      try {
        if (session_pinned_budget) sess_res = session_pinned_budget->Reserve(pinned_needed);
        if (pinned_budget) res = pinned_budget->ReserveAllocated(pinned_needed);
      } catch (const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing pinned budget reservation failed: ", ex.what());
      }
      uint8_t* new_pinned = nullptr;
      cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&new_pinned), pinned_needed, cudaHostAllocDefault);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing cudaHostAlloc failed: ", cudaGetErrorString(err));
      }
      if (sess_res.Active()) sess_res.Commit();
      if (res.Active()) res.Commit();
      pinned = new_pinned;
      pinned_bytes = pinned_needed;
    }
    while (copy_done.size() < waves) {
      cudaEvent_t copy_ev = nullptr;
      cudaEvent_t gemm_ev = nullptr;
      cudaError_t err1 = cudaEventCreateWithFlags(&copy_ev, cudaEventDisableTiming);
      if (err1 != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err1));
      cudaError_t err2 = cudaEventCreateWithFlags(&gemm_ev, cudaEventDisableTiming);
      if (err2 != cudaSuccess) {
        static_cast<void>(cudaEventDestroy(copy_ev));
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err2));
      }
      copy_done.push_back(copy_ev);
      gemm_done.push_back(gemm_ev);
    }
    return common::Status::OK();
  }
};

class QMoESessionController {
 public:
  struct GraphReplayNodeHook {
    const void* owner = nullptr;
    std::function<common::Status(cudaStream_t)> before_replay;
    std::function<common::Status(cudaStream_t, bool)> after_replay;
  };

  explicit QMoESessionController(int device_id = 0)
      : device_id_(device_id),
        landing_ring_(std::make_shared<QMoESessionLandingRing>()),
        pinned_budget_(std::make_shared<CudaDeviceMemoryBudget>()) {
    if (landing_ring_) {
      landing_ring_->session_pinned_budget = pinned_budget_;
    }
  }

  ~QMoESessionController() {
    Shutdown();
  }

  void RegisterOffloadNode() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++registered_nodes_;
  }

  void RegisterGraphReplayHook(const void* owner,
                               std::function<common::Status(cudaStream_t)> before_replay,
                               std::function<common::Status(cudaStream_t, bool)> after_replay = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& h : graph_replay_hooks_) {
      if (h.owner == owner) {
        h.before_replay = std::move(before_replay);
        h.after_replay = std::move(after_replay);
        return;
      }
    }
    graph_replay_hooks_.push_back({owner, std::move(before_replay), std::move(after_replay)});
  }

  void UnregisterOffloadNode(const void* owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = graph_replay_hooks_.begin(); it != graph_replay_hooks_.end();) {
      if (it->owner == owner) {
        it = graph_replay_hooks_.erase(it);
      } else {
        ++it;
      }
    }
    if (registered_nodes_ > 0) --registered_nodes_;
  }

  void BumpGraphGeneration() noexcept {
    graph_generation_.fetch_add(1, std::memory_order_acq_rel);
  }

  uint64_t GraphGeneration() const noexcept {
    return graph_generation_.load(std::memory_order_acquire);
  }

  common::Status OnBeforeGraphReplay(cudaStream_t stream) {
    std::vector<GraphReplayNodeHook> hooks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutting_down_.load(std::memory_order_acquire)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload session controller is shutting down.");
      }
      hooks = graph_replay_hooks_;
    }
    for (auto& h : hooks) {
      if (h.before_replay) {
        common::Status st = h.before_replay(stream);
        if (!st.IsOK()) return st;
      }
    }
    return common::Status::OK();
  }

  common::Status OnAfterGraphReplay(cudaStream_t stream, bool sync_stream) {
    std::vector<GraphReplayNodeHook> hooks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hooks = graph_replay_hooks_;
    }
    common::Status first_err = common::Status::OK();
    for (auto& h : hooks) {
      if (h.after_replay) {
        common::Status st = h.after_replay(stream, sync_stream);
        if (!st.IsOK() && first_err.IsOK()) first_err = st;
      }
    }
    if (sync_stream && stream != nullptr) {
      cudaError_t err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess && first_err.IsOK()) {
        first_err = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload graph replay stream sync failed: ", cudaGetErrorString(err));
      }
    }
    return first_err;
  }

  bool HasOffloadNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registered_nodes_ > 0;
  }

  bool IsShuttingDown() const noexcept {
    return shutting_down_.load(std::memory_order_acquire);
  }

  common::Status OnRunStart() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload session controller is shutting down.");
    }
    if (registered_nodes_ == 0) {
      return common::Status::OK();
    }
    if (run_in_flight_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "QMoEOffload does not support concurrent overlapping InferenceSession::Run calls on the same session.");
    }
    run_in_flight_ = true;
    ++current_run_id_;
    active_invocations_.clear();
    return common::Status::OK();
  }

  void RegisterInvocation(std::shared_ptr<QMoEInvocationRecord> inv) {
    std::lock_guard<std::mutex> lock(mutex_);
    inv->run_id = current_run_id_;
    inv->invocation_id = ++next_invocation_id_;
    active_invocations_.push_back(std::move(inv));
  }

  common::Status OnRunEnd(bool sync_stream, bool run_succeeded = true) {
    std::vector<std::shared_ptr<QMoEInvocationRecord>> invocations;
    bool had_offload_nodes = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      had_offload_nodes = (registered_nodes_ > 0);
      if (!had_offload_nodes && !run_in_flight_) {
        return common::Status::OK();
      }
      invocations = std::move(active_invocations_);
      active_invocations_.clear();
    }

    // Hold run_in_flight_ until all device/CPU completion draining and observation commits finish (F19).
    auto release_in_flight = [this](bool* active) noexcept {
      if (*active) {
        std::lock_guard<std::mutex> lock(mutex_);
        run_in_flight_ = false;
      }
    };
    bool hold_active = true;
    std::unique_ptr<bool, decltype(release_in_flight)> in_flight_guard(&hold_active, release_in_flight);

    common::Status first_error = common::Status::OK();
    if (had_offload_nodes && !sync_stream) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      const char* strict_env = std::getenv("ORT_QMOE_REJECT_DISABLE_SYNC");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      if (strict_env != nullptr && std::string(strict_env) == "1") {
        first_error = ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "QMoEOffload does not support disable_synchronize_execution_providers=1; required completion sync cannot be skipped.");
      }
    }
    cudaStream_t last_main_stream = nullptr;
    bool sync_last_main = false;
    for (auto& inv : invocations) {
      if (!inv) continue;
      if (inv->has_main_stream) {
        last_main_stream = inv->main_stream;
        sync_last_main = true;
      }
      if (inv->drain_device) {
        common::Status drain_st = inv->drain_device();
        if (!drain_st.IsOK() && first_error.IsOK()) {
          first_error = drain_st;
        }
      }
      std::lock_guard<std::mutex> inv_lock(inv->mutex);
      if (inv->failed && first_error.IsOK()) {
        first_error = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                      "QMoEOffload invocation failed on node '", inv->node_name,
                                      "' (layer ", inv->layer_index, "): ", inv->error_message);
      }
    }
    if (sync_last_main) {
      cudaError_t err = cudaStreamSynchronize(last_main_stream);
      if (err != cudaSuccess && first_error.IsOK()) {
        first_error = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                      "QMoEOffload compute stream sync failed: ", cudaGetErrorString(err));
      }
    }

    // Whole-run observation atomicity (v3 Section 5.1 & 7.1):
    // Commit staged observations only when the overall run and all invocations succeeded.
    if (run_succeeded && first_error.IsOK()) {
      for (auto& inv : invocations) {
        if (inv && inv->commit_observation) {
          inv->commit_observation();
        }
      }
    }
    return first_error;
  }

  void Shutdown() noexcept {
    shutting_down_.store(true, std::memory_order_release);
    std::vector<std::shared_ptr<QMoEInvocationRecord>> invocations;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      invocations = std::move(active_invocations_);
      active_invocations_.clear();
      run_in_flight_ = false;
    }
    for (auto& inv : invocations) {
      if (inv && inv->drain_device) {
        static_cast<void>(inv->drain_device());
      }
    }
    if (landing_ring_) {
      landing_ring_->Shutdown();
    }
    {
      std::lock_guard<std::mutex> lock(pinned_pool_mutex_);
      pinned_pool_.reset();
    }
  }

  std::shared_ptr<QMoESessionLandingRing> LandingRing() const { return landing_ring_; }
  const std::shared_ptr<CudaDeviceMemoryBudget>& PinnedBudget() const noexcept { return pinned_budget_; }
  int DeviceId() const noexcept { return device_id_; }

  std::shared_ptr<BoundedPinnedPool> GetOrCreatePinnedPool(size_t limit) {
    std::lock_guard<std::mutex> lock(pinned_pool_mutex_);
    if (!pinned_pool_) {
      auto sess_pinned = pinned_budget_;
      auto block_sizes = std::make_shared<std::pair<std::mutex, std::unordered_map<void*, size_t>>>();
      pinned_pool_ = std::make_shared<BoundedPinnedPool>(
          limit,
          [sess_pinned, block_sizes](size_t size) -> void* {
            auto host_ledger = GetCudaHostPinnedMemoryBudget();
            CudaDeviceMemoryBudget::Reservation sess_res;
            CudaDeviceMemoryBudget::Reservation host_res;
            try {
              if (sess_pinned) sess_res = sess_pinned->Reserve(size);
              if (host_ledger) host_res = host_ledger->ReserveAllocated(size);
            } catch (...) {
              return nullptr;
            }
            void* pointer = nullptr;
            if (cudaHostAlloc(&pointer, size, cudaHostAllocDefault) != cudaSuccess || !pointer) {
              static_cast<void>(cudaGetLastError());
              return nullptr;
            }
            {
              std::lock_guard<std::mutex> blk_lock(block_sizes->first);
              block_sizes->second[pointer] = size;
            }
            if (sess_res.Active()) sess_res.Commit();
            if (host_res.Active()) host_res.Commit();
            return pointer;
          },
          [sess_pinned, block_sizes](void* pointer) {
            if (!pointer) return;
            size_t size = 0;
            {
              std::lock_guard<std::mutex> blk_lock(block_sizes->first);
              auto it = block_sizes->second.find(pointer);
              if (it != block_sizes->second.end()) {
                size = it->second;
                block_sizes->second.erase(it);
              }
            }
            auto host_ledger = GetCudaHostPinnedMemoryBudget();
            if (sess_pinned && size > 0) sess_pinned->BeginFree(size);
            if (host_ledger && size > 0) host_ledger->BeginFreeAllocated(size);
            const bool ok = (cudaFreeHost(pointer) == cudaSuccess);
            if (host_ledger && size > 0) host_ledger->FinishFreeAllocated(size, ok);
            if (sess_pinned && size > 0) sess_pinned->FinishFree(size, ok);
          });
    }
    return pinned_pool_;
  }

  void SetBudgetAccount(std::shared_ptr<SessionBudgetAccount> account) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_account_ = std::move(account);
    if (landing_ring_) {
      landing_ring_->budget_account = budget_account_;
    }
  }

  std::shared_ptr<SessionBudgetAccount> BudgetAccount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return budget_account_;
  }

  void SetModelPath(std::filesystem::path model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_path_ = std::move(model_path);
  }

  std::filesystem::path GetModelPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_path_;
  }

  void RegisterMoeOffloadNode(onnxruntime::moe_offload::IMoeOffloadNode* node,
                              const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                              int32_t& inout_layer_index) {
    if (!node) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!policy_initialized_) {
      policy_ = policy;
      if (policy_.source_root.empty() && !model_path_.empty()) {
        std::error_code ec;
        policy_.source_root = std::filesystem::is_directory(model_path_, ec)
                                  ? model_path_
                                  : model_path_.parent_path();
      }
      if (!policy_.popularity_file.empty() && policy_.popularity_file.is_relative() &&
          !policy_.source_root.empty()) {
        policy_.popularity_file = policy_.source_root / policy_.popularity_file;
      }
      if (!policy_.budget_group.empty() && budget_account_) {
        const size_t acct_limit = budget_account_->GetAccountStats().limit;
        budget_account_->SetBudgetGroup(GetCudaBudgetGroupByName(
            device_id_, policy_.budget_group,
            acct_limit < std::numeric_limits<size_t>::max() ? acct_limit : 0));
      }
      if (pinned_budget_) {
        if (policy_.pinned_bytes_explicit && policy_.pinned_bytes == 0) {
          pinned_budget_->SetZeroLimit();
        } else if (policy_.pinned_bytes > 0) {
          pinned_budget_->SetExactLimit(static_cast<size_t>(policy_.pinned_bytes));
        }
      }
      policy_initialized_ = true;
    }
    if (inout_layer_index < 0) {
      int32_t next_id = static_cast<int32_t>(moe_nodes_.size());
      auto is_used = [&](int32_t id) {
        for (auto* existing : moe_nodes_) {
          if (existing && existing->GetSourceDescriptor().layer_index == id) return true;
        }
        return false;
      };
      while (is_used(next_id)) ++next_id;
      inout_layer_index = next_id;
    }
    for (auto* existing : moe_nodes_) {
      if (existing == node) return;
    }
    moe_nodes_.push_back(node);
  }

  void UnregisterMoeOffloadNode(onnxruntime::moe_offload::IMoeOffloadNode* node) {
    std::lock_guard<std::mutex> lock(mutex_);
    moe_nodes_.erase(std::remove(moe_nodes_.begin(), moe_nodes_.end(), node), moe_nodes_.end());
  }

  static void QueryPlatformUsableDeviceMemory(int device_id,
                                              uint64_t* out_usable_free_bytes,
                                              uint64_t* out_total_bytes = nullptr,
                                              uint64_t* out_non_local_usage_bytes = nullptr) noexcept {
    if (out_usable_free_bytes) *out_usable_free_bytes = 0;
    if (out_total_bytes) *out_total_bytes = 0;
    if (out_non_local_usage_bytes) *out_non_local_usage_bytes = 0;

    int prev_dev = 0;
    const bool restore_dev = (cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != device_id);
    if (restore_dev) {
      static_cast<void>(cudaSetDevice(device_id));
    }
    size_t cuda_free = 0;
    size_t cuda_total = 0;
    if (cudaMemGetInfo(&cuda_free, &cuda_total) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      if (restore_dev) static_cast<void>(cudaSetDevice(prev_dev));
      return;
    }

    uint64_t usable_free = static_cast<uint64_t>(cuda_free);
#if defined(_WIN32)
    cudaDeviceProp prop{};
    const uint64_t wddm_os_reserve =
        std::max<uint64_t>(1280ULL << 20, static_cast<uint64_t>(cuda_total) * 6 / 100);
    const uint64_t physical_ceiling =
        (static_cast<uint64_t>(cuda_total) > wddm_os_reserve)
            ? (static_cast<uint64_t>(cuda_total) - wddm_os_reserve)
            : 0ULL;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
      Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
      if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory) {
        LUID adapter_luid{};
        std::memcpy(&adapter_luid, prop.luid, sizeof(LUID));
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter3))) && adapter3) {
          DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
          DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info{};
          if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))) {
            const uint64_t dxgi_rem =
                (local_info.Budget > local_info.CurrentUsage)
                    ? (local_info.Budget - local_info.CurrentUsage)
                    : 0ULL;
            const uint64_t phys_rem =
                (physical_ceiling > local_info.CurrentUsage)
                    ? (physical_ceiling - local_info.CurrentUsage)
                    : 0ULL;
            usable_free = std::min({usable_free, dxgi_rem, phys_rem});
          }
          if (out_non_local_usage_bytes &&
              SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info))) {
            *out_non_local_usage_bytes = non_local_info.CurrentUsage;
          }
        }
      }
    } else {
      usable_free = (usable_free > wddm_os_reserve) ? (usable_free - wddm_os_reserve) : 0ULL;
    }
#else
    const uint64_t driver_reserve =
        std::max<uint64_t>(512ULL << 20, static_cast<uint64_t>(cuda_total) * 2 / 100);
    usable_free = (usable_free > driver_reserve) ? (usable_free - driver_reserve) : 0ULL;
#endif

    // Also respect CudaDeviceMemoryBudget limit (e.g. gpu_device_mem_limit) if configured.
    if (auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id)) {
      const auto stats = dev_budget->GetStats();
      if (stats.limit < std::numeric_limits<size_t>::max()) {
        const uint64_t cuda_used_now =
            (cuda_total > cuda_free) ? static_cast<uint64_t>(cuda_total - cuda_free) : 0ULL;
        const uint64_t tracked_used = static_cast<uint64_t>(stats.committed + stats.pending);
        const uint64_t effective_used = std::max(cuda_used_now, tracked_used);
        const uint64_t budget_rem =
            (static_cast<uint64_t>(stats.limit) > effective_used)
                ? (static_cast<uint64_t>(stats.limit) - effective_used)
                : 0ULL;
        usable_free = std::min(usable_free, budget_rem);
      }
    }

    if (restore_dev) {
      static_cast<void>(cudaSetDevice(prev_dev));
    }
    if (out_usable_free_bytes) *out_usable_free_bytes = usable_free;
    if (out_total_bytes) *out_total_bytes = static_cast<uint64_t>(cuda_total);
  }

  static void EvictUploadedHostPages(const onnxruntime::moe_offload::MoeExpertSourceDescriptor& desc,
                                     const std::vector<int32_t>& uploaded_experts) noexcept {
    if (uploaded_experts.empty() || desc.num_experts <= 0) return;
#if !defined(_WIN32)
    const long page_sz_raw = sysconf(_SC_PAGESIZE);
    const size_t page_size = (page_sz_raw > 0) ? static_cast<size_t>(page_sz_raw) : 4096u;
#else
    constexpr size_t page_size = 4096u;
#endif
    std::vector<uint8_t> is_uploaded(static_cast<size_t>(desc.num_experts), 0);
    for (int32_t e : uploaded_experts) {
      if (e >= 0 && e < desc.num_experts) is_uploaded[static_cast<size_t>(e)] = 1;
    }

    for (const auto& opt_bank : desc.banks) {
      if (!opt_bank.has_value()) continue;
      const auto& bank = *opt_bank;
      if (!bank.host_data || bank.bytes_per_expert < page_size) continue;
      const size_t total_bank_bytes =
          (bank.total_bytes > 0) ? bank.total_bytes
                                 : (static_cast<size_t>(desc.num_experts) * bank.bytes_per_expert);
      // Only evict verified read-only regular-file mappings; anonymous/heap memory would be zeroed by MADV_DONTNEED.
      if (!onnxruntime::moe_offload::IsReloadableFileMapping(bank.host_data, total_bank_bytes)) {
        continue;
      }
      const uintptr_t base_addr = reinterpret_cast<uintptr_t>(bank.host_data);
      int32_t run_start = -1;
      for (int32_t e = 0; e <= desc.num_experts; ++e) {
        const bool up = (e < desc.num_experts) && (is_uploaded[static_cast<size_t>(e)] != 0);
        if (up && run_start < 0) {
          run_start = e;
        } else if (!up && run_start >= 0) {
          const uintptr_t start_byte = base_addr + static_cast<size_t>(run_start) * bank.bytes_per_expert;
          const uintptr_t end_byte = base_addr + static_cast<size_t>(e) * bank.bytes_per_expert;
          // Align inward to page boundaries so we never evict a page shared with a cold CPU expert!
          const uintptr_t aligned_start = (start_byte + page_size - 1) & ~(page_size - 1);
          const uintptr_t aligned_end = end_byte & ~(page_size - 1);
          if (aligned_end > aligned_start) {
            void* ptr = reinterpret_cast<void*>(aligned_start);
            const size_t len = static_cast<size_t>(aligned_end - aligned_start);
#if defined(_WIN32)
            static_cast<void>(VirtualUnlock(ptr, len));
#else
            static_cast<void>(madvise(ptr, len, MADV_DONTNEED));
#endif
          }
          run_start = -1;
        }
      }
    }
  }

#if defined(__linux__)
  static void PrepareLinuxHostExpertMappings(
      const std::vector<onnxruntime::moe_offload::IMoeOffloadNode*>& nodes) noexcept {
    const char* interleave_env = std::getenv("ORT_QMOE_NUMA_INTERLEAVE");
    const bool interleave = (interleave_env == nullptr || std::strcmp(interleave_env, "0") != 0);
    const char* diag_env = std::getenv("ORT_QMOE_NUMA_DIAGNOSTICS");
    const bool diagnostics = (diag_env != nullptr && std::strcmp(diag_env, "0") != 0);
    const char* migrate_env = std::getenv("ORT_QMOE_NUMA_MIGRATE");
    const bool migrate = (migrate_env == nullptr || std::strcmp(migrate_env, "0") != 0);
    if (!interleave && !diagnostics) return;

    const long page_sz_raw = sysconf(_SC_PAGESIZE);
    const size_t page_size = (page_sz_raw > 0) ? static_cast<size_t>(page_sz_raw) : 4096u;

    constexpr size_t kMaxNodes = 1024;
    constexpr size_t kWordBits = sizeof(unsigned long) * 8;
    std::array<unsigned long, kMaxNodes / kWordBits> allowed{};
    if (syscall(SYS_get_mempolicy, nullptr, allowed.data(), kMaxNodes, nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
      return;
    }
    std::vector<int> numa_nodes;
    for (size_t n = 0; n < kMaxNodes; ++n) {
      if (allowed[n / kWordBits] & (1UL << (n % kWordBits))) {
        numa_nodes.push_back(static_cast<int>(n));
      }
    }
    if (numa_nodes.empty()) return;

    struct VmaEntry {
      uintptr_t start = 0;
      uintptr_t end = 0;
      uint64_t file_offset = 0;
      std::string filename;
    };
    std::vector<VmaEntry> vmas;
    if (std::ifstream maps("/proc/self/maps"); maps) {
      std::string line;
      while (std::getline(maps, line)) {
        std::istringstream iss(line);
        std::string range_str, perms, offset_str, dev, inode;
        if (!(iss >> range_str >> perms >> offset_str >> dev >> inode)) continue;
        std::string path_str;
        std::getline(iss >> std::ws, path_str);
        const auto dash = range_str.find('-');
        if (dash == std::string::npos) continue;
        try {
          VmaEntry v;
          v.start = static_cast<uintptr_t>(std::stoull(range_str.substr(0, dash), nullptr, 16));
          v.end = static_cast<uintptr_t>(std::stoull(range_str.substr(dash + 1), nullptr, 16));
          v.file_offset = std::stoull(offset_str, nullptr, 16);
          if (!path_str.empty()) {
            v.filename = std::filesystem::path(path_str).filename().string();
          }
          vmas.push_back(std::move(v));
        } catch (...) {
        }
      }
    }

    std::vector<std::pair<uintptr_t, uintptr_t>> raw_ranges;
    for (auto* node : nodes) {
      if (!node) continue;
      const auto& desc = node->GetSourceDescriptor();
      for (const auto& opt_bank : desc.banks) {
        if (!opt_bank.has_value()) continue;
        const auto& bank = *opt_bank;
        if (!bank.host_data || bank.total_bytes < (size_t{1} << 20)) continue;
        if (!onnxruntime::moe_offload::IsReloadableFileMapping(bank.host_data, bank.total_bytes)) continue;
        const uintptr_t lo = reinterpret_cast<uintptr_t>(bank.host_data) & ~(page_size - 1);
        const uintptr_t hi =
            (reinterpret_cast<uintptr_t>(bank.host_data) + bank.total_bytes + page_size - 1) & ~(page_size - 1);
        if (hi > lo) raw_ranges.emplace_back(lo, hi);
      }
    }
    if (raw_ranges.empty()) return;
    std::sort(raw_ranges.begin(), raw_ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto& iv : raw_ranges) {
      if (merged.empty() || iv.first > merged.back().second) {
        merged.push_back(iv);
      } else {
        merged.back().second = std::max(merged.back().second, iv.second);
      }
    }

    for (const auto& [lo, hi] : merged) {
      const size_t bytes = hi - lo;
      const size_t pages = bytes / page_size;
      uint64_t file_offset = 0;
      std::string fname = "expert_mmap";
      for (const auto& v : vmas) {
        if (lo >= v.start && lo < v.end) {
          file_offset = v.file_offset + static_cast<uint64_t>(lo - v.start);
          if (!v.filename.empty()) fname = v.filename;
          break;
        }
      }

      std::array<unsigned long, kMaxNodes / kWordBits> saved_mask{};
      int saved_mode = 0;
      bool policy_applied = false;
      if (interleave && numa_nodes.size() > 1) {
        if (syscall(SYS_get_mempolicy, &saved_mode, saved_mask.data(), kMaxNodes, nullptr, 0) == 0 &&
            syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, allowed.data(), kMaxNodes) == 0) {
          policy_applied = true;
        }
      }

      const volatile uint8_t* data = reinterpret_cast<const volatile uint8_t*>(lo);
      for (size_t p = 0; p < pages; ++p) {
        static_cast<void>(data[p * page_size]);
      }

      if (policy_applied) {
        const bool empty = std::all_of(saved_mask.begin(), saved_mask.end(), [](unsigned long v) { return v == 0; });
        static_cast<void>(syscall(SYS_set_mempolicy, saved_mode, empty ? nullptr : saved_mask.data(),
                                  empty ? 0 : kMaxNodes));
      }

      if ((interleave && migrate && numa_nodes.size() > 1) || diagnostics) {
        constexpr size_t kBatchPages = 4096;
        std::array<void*, kBatchPages> batch_ptrs{};
        std::array<void*, kBatchPages> move_ptrs{};
        std::array<int, kBatchPages> targets{};
        std::array<int, kBatchPages> status{};
        std::array<int, kBatchPages> move_status{};
        std::vector<size_t> pages_before(numa_nodes.size(), 0);
        std::vector<size_t> pages_after(numa_nodes.size(), 0);
        size_t misplaced_before = 0, misplaced_after = 0, move_attempts = 0, move_failures = 0;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(lo);

        for (size_t first = 0; first < pages; first += kBatchPages) {
          const size_t count = std::min(kBatchPages, pages - first);
          for (size_t i = 0; i < count; ++i) {
            batch_ptrs[i] = const_cast<uint8_t*>(base + (first + i) * page_size);
          }
          if (syscall(SYS_move_pages, 0, count, batch_ptrs.data(), nullptr, status.data(), 0) < 0) break;
          size_t to_move = 0;
          for (size_t i = 0; i < count; ++i) {
            const int target = numa_nodes[(file_offset / page_size + first + i) % numa_nodes.size()];
            auto it = std::find(numa_nodes.begin(), numa_nodes.end(), status[i]);
            if (it != numa_nodes.end()) ++pages_before[static_cast<size_t>(it - numa_nodes.begin())];
            if (status[i] >= 0 && status[i] != target) {
              ++misplaced_before;
              if (interleave && migrate && numa_nodes.size() > 1) {
                move_ptrs[to_move] = batch_ptrs[i];
                targets[to_move++] = target;
              }
            }
          }
          if (to_move > 0) {
            move_attempts += to_move;
            if (syscall(SYS_move_pages, 0, to_move, move_ptrs.data(), targets.data(),
                        move_status.data(), MPOL_MF_MOVE) == 0) {
              for (size_t i = 0; i < to_move; ++i) {
                if (move_status[i] != targets[i]) ++move_failures;
              }
            } else {
              move_failures += to_move;
            }
            if (syscall(SYS_move_pages, 0, count, batch_ptrs.data(), nullptr, status.data(), 0) < 0) break;
          }
          for (size_t i = 0; i < count; ++i) {
            const int target = numa_nodes[(file_offset / page_size + first + i) % numa_nodes.size()];
            auto it = std::find(numa_nodes.begin(), numa_nodes.end(), status[i]);
            if (it != numa_nodes.end()) ++pages_after[static_cast<size_t>(it - numa_nodes.begin())];
            if (status[i] >= 0 && status[i] != target) ++misplaced_after;
          }
        }
        if (diagnostics) {
          std::ostringstream oss;
          oss << "[QMoE NUMA] file=" << fname << " offset=" << file_offset << " bytes=" << bytes
              << " policy_applied=" << (policy_applied ? 1 : 0) << " pages=" << pages;
          for (size_t n = 0; n < numa_nodes.size(); ++n) {
            oss << " N" << numa_nodes[n] << "=" << pages_before[n] << "->" << pages_after[n];
          }
          oss << " misplaced=" << misplaced_before << "->" << misplaced_after
              << " move_attempts=" << move_attempts << " move_failures=" << move_failures;
          std::fprintf(stderr, "%s\n", oss.str().c_str());
        }
      }
    }
  }
#endif

  common::Status PlanAndAdmitSession() {
    std::vector<onnxruntime::moe_offload::IMoeOffloadNode*> nodes;
    onnxruntime::moe_offload::MoeOffloadSessionPolicy policy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (session_plan_finalized_ || moe_nodes_.empty()) {
        return common::Status::OK();
      }
      session_plan_finalized_ = true;
      nodes = moe_nodes_;
      policy = policy_;
    }

    auto rollback_guard = [this](bool* committed) {
      if (!*committed) {
        std::lock_guard<std::mutex> lock(mutex_);
        session_plan_finalized_ = false;
      }
    };
    bool plan_committed = false;
    std::unique_ptr<bool, decltype(rollback_guard)> guard(&plan_committed, rollback_guard);

    const size_t num_layers = nodes.size();
    uint64_t usable_free_vram = 0;
    uint64_t total_vram = 0;
    uint64_t non_local_before = 0;
    QueryPlatformUsableDeviceMemory(device_id_, &usable_free_vram, &total_vram, &non_local_before);
    if (auto acct = BudgetAccount()) {
      for (const auto& st : {acct->GetAccountStats(), acct->GetGroupStats(), acct->GetPhysicalStats()}) {
        if (st.limit < std::numeric_limits<size_t>::max()) {
          const uint64_t used = static_cast<uint64_t>(st.committed + st.pending);
          const uint64_t rem = (static_cast<uint64_t>(st.limit) > used) ? (static_cast<uint64_t>(st.limit) - used) : 0ULL;
          usable_free_vram = std::min(usable_free_vram, rem);
        }
      }
    }

    // Validate minimum host/device working budgets and check for duplicate layer indices (F09, F14).
    std::unordered_map<int32_t, size_t> layer_to_idx;
    uint64_t total_fixed_table_bytes = 0;
    uint64_t total_dummy_bytes = 0;
    uint64_t total_full_gpu_bytes = 0;
    for (size_t i = 0; i < num_layers; ++i) {
      auto* node = nodes[i];
      const auto& desc = node->GetSourceDescriptor();
      const auto& caps = node->GetBackendCapabilities();
      const auto [it, inserted] = layer_to_idx.emplace(desc.layer_index, i);
      if (!inserted) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MoE Offload session admission failed: duplicate layer_index=", desc.layer_index,
                               " on nodes '", nodes[it->second]->GetSourceDescriptor().stable_node_id, "' and '",
                               desc.stable_node_id, "'.");
      }
      const uint64_t min_node_host_work =
          static_cast<uint64_t>(std::max(1, desc.hidden_size + 2 * desc.inter_size)) * sizeof(float) * 4ULL;
      if (policy.host_working_bytes < min_node_host_work) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "MoE Offload session admission failed: host_working_bytes (", policy.host_working_bytes,
                               ") is smaller than minimum CPU working buffer (", min_node_host_work,
                               " bytes) for node '", desc.stable_node_id, "'.");
      }
      // The dummy slot is mandatory for every node whose decode kernel cannot skip negative slots, so it is
      // reserved before any resident capacity is distributed.
      const uint64_t dummy_bytes = caps.decode_kernel_skips_negative_slots ? 0ULL : caps.gpu_slot_bytes;
      total_fixed_table_bytes += caps.gpu_fixed_table_bytes;
      total_dummy_bytes += dummy_bytes;
      total_full_gpu_bytes += caps.gpu_fixed_table_bytes + dummy_bytes +
                              static_cast<uint64_t>(desc.num_experts) * caps.gpu_slot_bytes;
    }
    const uint64_t total_base_bytes = total_fixed_table_bytes + total_dummy_bytes;

    // Resolve optional per-node overrides (F14). Byte caps cover the complete node cost: fixed tables, dummy slot,
    // resident slots and spare slots.
    std::vector<int32_t> max_resident_slots_for_node(num_layers);
    std::vector<uint64_t> node_slot_budget_bytes(num_layers, std::numeric_limits<uint64_t>::max());
    for (size_t i = 0; i < num_layers; ++i) {
      max_resident_slots_for_node[i] = nodes[i]->GetSourceDescriptor().num_experts;
    }
    for (const auto& ov : policy.node_overrides) {
      bool matched = false;
      for (size_t i = 0; i < num_layers; ++i) {
        const auto& desc = nodes[i]->GetSourceDescriptor();
        if (desc.stable_node_id == ov.node_name ||
            std::to_string(desc.layer_index) == ov.node_name ||
            ("layer." + std::to_string(desc.layer_index)) == ov.node_name) {
          matched = true;
          if (ov.max_resident_slots.has_value()) {
            max_resident_slots_for_node[i] =
                std::clamp<int32_t>(*ov.max_resident_slots, 0, desc.num_experts);
          }
          if (ov.device_expert_bytes.has_value()) {
            const auto& caps = nodes[i]->GetBackendCapabilities();
            const uint64_t base = caps.gpu_fixed_table_bytes +
                                  (caps.decode_kernel_skips_negative_slots ? 0ULL : caps.gpu_slot_bytes);
            const uint64_t avail = (*ov.device_expert_bytes > base) ? (*ov.device_expert_bytes - base) : 0ULL;
            node_slot_budget_bytes[i] = std::min(node_slot_budget_bytes[i], avail);
          }
        }
      }
      if (!matched) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MoE Offload node_overrides references unknown node '", ov.node_name, "'.");
      }
    }

    uint64_t admitted_gpu_expert_budget = 0;
    if (!policy.device_expert_bytes.is_auto) {
      admitted_gpu_expert_budget = policy.device_expert_bytes.bytes;
    } else {
      const uint64_t reserves = policy.reserve_device_bytes + policy.device_working_bytes;
      admitted_gpu_expert_budget = (usable_free_vram > reserves) ? (usable_free_vram - reserves) : 0ULL;
    }

    if (admitted_gpu_expert_budget < total_base_bytes) {
      if (policy.mode == onnxruntime::moe_offload::OffloadMode::kRequired &&
          !policy.device_expert_bytes.is_auto && policy.device_expert_bytes.bytes == 0) {
        admitted_gpu_expert_budget = total_base_bytes;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "MoE Offload session admission failed: device_expert_bytes (",
                               admitted_gpu_expert_budget, ") is smaller than fixed lookup table and dummy slot overhead (",
                               total_base_bytes, " bytes across ", num_layers, " layers).");
      }
    }

    // Load optional popularity profile (STRP binary format 0x50525453) without hardcoded developer paths (F13, F18).
    std::filesystem::path prof_path = policy.popularity_file;
    if (prof_path.empty() && !policy.source_root.empty()) {
      const auto candidate1 = policy.source_root / "expert-profile.bin";
      std::error_code ec;
      if (std::filesystem::exists(candidate1, ec)) {
        prof_path = candidate1;
      }
    }

    std::vector<std::pair<int32_t, int32_t>> global_ranked_pairs;
    std::vector<std::vector<int32_t>> per_layer_ranked(num_layers);
    if (!prof_path.empty()) {
      std::ifstream pf(prof_path, std::ios::binary);
      uint32_t hdr[6]{};
      bool profile_valid = false;
      if (pf && pf.read(reinterpret_cast<char*>(hdr), sizeof(hdr))) {
        const uint32_t magic = hdr[0], ver = hdr[1], n_layers = hdr[2], n_exp = hdr[3], n_ranked = hdr[5];
        if (magic == 0x50525453u && ver == 1u && n_layers > 0 && n_exp > 0 && n_ranked <= n_layers * n_exp) {
          std::vector<uint16_t> raw_pairs(static_cast<size_t>(n_ranked) * 2);
          if (pf.read(reinterpret_cast<char*>(raw_pairs.data()),
                      static_cast<std::streamsize>(raw_pairs.size() * sizeof(uint16_t)))) {
            profile_valid = true;
            global_ranked_pairs.reserve(n_ranked);
            for (uint32_t i = 0; i < n_ranked; ++i) {
              global_ranked_pairs.emplace_back(static_cast<int32_t>(raw_pairs[2 * i]),
                                               static_cast<int32_t>(raw_pairs[2 * i + 1]));
            }
          }
        }
      }
      if (!profile_valid && policy.popularity_file_explicit) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MoE Offload failed to load or validate explicitly requested popularity_file: '",
                               prof_path.string(), "'.");
      }
    }

    std::vector<int32_t> resident_counts(num_layers, 0);
    std::vector<int32_t> spare_counts(num_layers, 0);
    uint64_t remaining_gpu = admitted_gpu_expert_budget - total_base_bytes;
    const bool is_adaptive = (policy.placement == onnxruntime::moe_offload::PlacementPolicy::kAdaptive);

    bool all_unconstrained = true;
    for (size_t i = 0; i < num_layers; ++i) {
      const auto& desc = nodes[i]->GetSourceDescriptor();
      const uint64_t full_slot_bytes =
          static_cast<uint64_t>(desc.num_experts) * nodes[i]->GetBackendCapabilities().gpu_slot_bytes;
      if (max_resident_slots_for_node[i] < desc.num_experts || node_slot_budget_bytes[i] < full_slot_bytes) {
        all_unconstrained = false;
        break;
      }
    }

    // Moves node `idx` to one more resident slot. Spare slots exist only while a node is adaptively and partially
    // resident, so the byte delta (and the node cap) is computed from the complete resident+spare cost before/after.
    auto spare_slots_for = [&](size_t idx, int32_t resident) -> int32_t {
      return (is_adaptive && resident > 0 && resident < nodes[idx]->GetSourceDescriptor().num_experts) ? 2 : 0;
    };
    auto try_add_slot = [&](size_t idx) -> bool {
      const auto& desc = nodes[idx]->GetSourceDescriptor();
      const uint64_t slot_cost = nodes[idx]->GetBackendCapabilities().gpu_slot_bytes;
      const int32_t cur = resident_counts[idx];
      if (cur >= max_resident_slots_for_node[idx] || cur >= desc.num_experts) return false;
      const uint64_t old_slots = static_cast<uint64_t>(cur + spare_slots_for(idx, cur));
      const uint64_t new_slots = static_cast<uint64_t>(cur + 1 + spare_slots_for(idx, cur + 1));
      if (new_slots * slot_cost > node_slot_budget_bytes[idx]) return false;
      if (new_slots >= old_slots) {
        const uint64_t delta = (new_slots - old_slots) * slot_cost;
        if (remaining_gpu < delta) return false;
        remaining_gpu -= delta;
      } else {
        remaining_gpu += (old_slots - new_slots) * slot_cost;
      }
      ++resident_counts[idx];
      spare_counts[idx] = spare_slots_for(idx, resident_counts[idx]);
      return true;
    };

    // Check if full GPU residency fits when mode == kAuto and no explicit smaller cap was set.
    if (policy.mode == onnxruntime::moe_offload::OffloadMode::kAuto &&
        all_unconstrained &&
        admitted_gpu_expert_budget >= total_full_gpu_bytes) {
      for (size_t i = 0; i < num_layers; ++i) {
        const int32_t ne = nodes[i]->GetSourceDescriptor().num_experts;
        resident_counts[i] = ne;
        spare_counts[i] = 0;
        per_layer_ranked[i].resize(static_cast<size_t>(ne));
        for (int32_t e = 0; e < ne; ++e) per_layer_ranked[i][static_cast<size_t>(e)] = e;
      }
      remaining_gpu = admitted_gpu_expert_budget - total_full_gpu_bytes;
    } else if (!global_ranked_pairs.empty()) {
      // Profile-guided byte-aware water-filling across globally ranked (layer, expert) pairs
      // with a per-layer floor (min_slots_per_layer = 4) before global ranking.
      std::vector<std::vector<uint8_t>> seen(num_layers);
      for (size_t i = 0; i < num_layers; ++i) {
        seen[i].assign(static_cast<size_t>(nodes[i]->GetSourceDescriptor().num_experts), 0);
      }
      for (const auto& [l_id, e_id] : global_ranked_pairs) {
        auto it = layer_to_idx.find(l_id);
        if (it == layer_to_idx.end()) continue;
        const size_t idx = it->second;
        const auto& desc = nodes[idx]->GetSourceDescriptor();
        if (e_id < 0 || e_id >= desc.num_experts || seen[idx][static_cast<size_t>(e_id)]) continue;
        seen[idx][static_cast<size_t>(e_id)] = 1;
        per_layer_ranked[idx].push_back(e_id);
      }
      for (size_t i = 0; i < num_layers; ++i) {
        const int32_t ne = nodes[i]->GetSourceDescriptor().num_experts;
        for (int32_t e = 0; e < ne; ++e) {
          if (!seen[i][static_cast<size_t>(e)]) per_layer_ranked[i].push_back(e);
        }
      }

      constexpr int32_t kMinFloorSlotsPerLayer = 4;
      std::vector<std::vector<uint8_t>> admitted(num_layers);
      for (size_t i = 0; i < num_layers; ++i) {
        const int32_t ne = nodes[i]->GetSourceDescriptor().num_experts;
        admitted[i].assign(static_cast<size_t>(ne), 0);
        const int32_t floor_slots = std::min({kMinFloorSlotsPerLayer, ne, max_resident_slots_for_node[i]});
        for (int32_t s = 0; s < floor_slots && static_cast<size_t>(s) < per_layer_ranked[i].size(); ++s) {
          if (try_add_slot(i)) {
            admitted[i][static_cast<size_t>(per_layer_ranked[i][static_cast<size_t>(s)])] = 1;
          }
        }
      }

      for (const auto& [l_id, e_id] : global_ranked_pairs) {
        auto it = layer_to_idx.find(l_id);
        if (it == layer_to_idx.end()) continue;
        const size_t idx = it->second;
        const auto& desc = nodes[idx]->GetSourceDescriptor();
        if (e_id < 0 || e_id >= desc.num_experts || admitted[idx][static_cast<size_t>(e_id)]) continue;
        if (try_add_slot(idx)) {
          admitted[idx][static_cast<size_t>(e_id)] = 1;
        }
      }

      // Remainder water-filling pass over completed per-layer rankings when a partial profile ends early (F18).
      std::vector<size_t> next_rank_cursor(num_layers, 0);
      bool rem_progress = true;
      while (rem_progress) {
        rem_progress = false;
        for (size_t i = 0; i < num_layers; ++i) {
          while (next_rank_cursor[i] < per_layer_ranked[i].size() &&
                 admitted[i][static_cast<size_t>(per_layer_ranked[i][next_rank_cursor[i]])]) {
            ++next_rank_cursor[i];
          }
          if (next_rank_cursor[i] >= per_layer_ranked[i].size()) continue;
          const int32_t cand_e = per_layer_ranked[i][next_rank_cursor[i]];
          if (try_add_slot(i)) {
            admitted[i][static_cast<size_t>(cand_e)] = 1;
            ++next_rank_cursor[i];
            rem_progress = true;
          }
        }
      }
    } else {
      // Uniform proportional water-filling pass across all layers.
      for (size_t i = 0; i < num_layers; ++i) {
        const int32_t ne = nodes[i]->GetSourceDescriptor().num_experts;
        per_layer_ranked[i].resize(static_cast<size_t>(ne));
        for (int32_t e = 0; e < ne; ++e) per_layer_ranked[i][static_cast<size_t>(e)] = e;
      }
      bool progress = true;
      while (progress) {
        progress = false;
        for (size_t i = 0; i < num_layers; ++i) {
          if (try_add_slot(i)) progress = true;
        }
      }
    }

    // Host expert cache budget distribution across layers with cold experts (F07, F10).
    uint64_t total_cold_bytes = 0;
    uint64_t total_full_cpu_bytes = 0;
    uint64_t min_bounded_host_bytes = 0;
    size_t partial_layers = 0;
    for (size_t i = 0; i < num_layers; ++i) {
      const auto& desc = nodes[i]->GetSourceDescriptor();
      const auto& caps = nodes[i]->GetBackendCapabilities();
      total_full_cpu_bytes += static_cast<uint64_t>(desc.num_experts) * caps.cpu_slot_bytes;
      if (resident_counts[i] < desc.num_experts) {
        ++partial_layers;
        total_cold_bytes += static_cast<uint64_t>(desc.num_experts - resident_counts[i]) * caps.cpu_slot_bytes;
        min_bounded_host_bytes += caps.cpu_slot_bytes;
      }
    }

    if (partial_layers > 0 && !policy.host_expert_bytes.is_auto) {
      if (!policy.allow_file_backing && policy.host_expert_bytes.bytes < total_cold_bytes) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "MoE Offload session admission failed: host_expert_bytes (", policy.host_expert_bytes.bytes,
            ") is smaller than cold expert complement (", total_cold_bytes,
            " bytes) while allow_file_backing=0.");
      }
      if (policy.allow_file_backing && policy.host_expert_bytes.bytes < total_cold_bytes &&
          policy.host_expert_bytes.bytes < min_bounded_host_bytes) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "MoE Offload session admission failed: host_expert_bytes (", policy.host_expert_bytes.bytes,
            ") is smaller than minimum bounded host cache capacity (", min_bounded_host_bytes,
            " bytes across ", partial_layers, " partially offloaded layers).");
      }
    }

    const bool use_bounded_file_cache =
        partial_layers > 0 &&
        !policy.host_expert_bytes.is_auto &&
        policy.host_expert_bytes.bytes < total_cold_bytes &&
        policy.allow_file_backing;

    std::vector<int32_t> host_cache_counts(num_layers, 0);
    if (use_bounded_file_cache && partial_layers > 0) {
      uint64_t rem_host = std::min(policy.host_expert_bytes.bytes, total_cold_bytes);
      // 1) Guarantee at least 1 bounded host cache slot for every partial layer first (F07).
      for (size_t i = 0; i < num_layers; ++i) {
        const auto& desc = nodes[i]->GetSourceDescriptor();
        const auto& caps = nodes[i]->GetBackendCapabilities();
        const int32_t max_cold = desc.num_experts - resident_counts[i];
        if (max_cold <= 0 || caps.cpu_slot_bytes == 0) continue;
        host_cache_counts[i] = 1;
        rem_host -= caps.cpu_slot_bytes;
      }
      // 2) Fast uniform base pass + water-filling remainder pass across layers with cold capacity.
      for (size_t i = 0; i < num_layers; ++i) {
        const auto& desc = nodes[i]->GetSourceDescriptor();
        const auto& caps = nodes[i]->GetBackendCapabilities();
        const int32_t max_cold = desc.num_experts - resident_counts[i];
        if (max_cold <= host_cache_counts[i] || caps.cpu_slot_bytes == 0) continue;
        const uint64_t base_share = policy.host_expert_bytes.bytes / partial_layers;
        const int32_t target_slots = static_cast<int32_t>(
            std::clamp<uint64_t>(base_share / caps.cpu_slot_bytes, 1ULL, static_cast<uint64_t>(max_cold)));
        const int32_t extra_slots = std::max<int32_t>(0, target_slots - host_cache_counts[i]);
        const uint64_t cost = static_cast<uint64_t>(extra_slots) * caps.cpu_slot_bytes;
        if (extra_slots > 0 && rem_host >= cost) {
          host_cache_counts[i] += extra_slots;
          rem_host -= cost;
        }
      }
      bool host_progress = true;
      while (host_progress) {
        host_progress = false;
        for (size_t i = 0; i < num_layers; ++i) {
          const auto& desc = nodes[i]->GetSourceDescriptor();
          const auto& caps = nodes[i]->GetBackendCapabilities();
          const int32_t max_cold = desc.num_experts - resident_counts[i];
          if (host_cache_counts[i] >= max_cold || caps.cpu_slot_bytes == 0) continue;
          if (rem_host >= caps.cpu_slot_bytes) {
            ++host_cache_counts[i];
            rem_host -= caps.cpu_slot_bytes;
            host_progress = true;
          }
        }
      }
    }

    uint64_t total_admitted_gpu_bytes = 0;
    int64_t total_resident_slots = 0;
    int64_t total_spare_slots = 0;

#if defined(__linux__)
    if (!use_bounded_file_cache && partial_layers > 0) {
      PrepareLinuxHostExpertMappings(nodes);
    }
#endif

    for (size_t i = 0; i < num_layers; ++i) {
      const auto& desc = nodes[i]->GetSourceDescriptor();
      const auto& node_caps = nodes[i]->GetBackendCapabilities();

      onnxruntime::moe_offload::MoeNodeExecutionPlan plan;
      plan.layer_index = desc.layer_index;
      plan.num_experts = desc.num_experts;
      plan.resident_slots = resident_counts[i];
      plan.spare_slots = (plan.resident_slots > 0 && plan.resident_slots < desc.num_experts && is_adaptive)
                             ? spare_counts[i]
                             : 0;
      plan.dummy_slots = node_caps.decode_kernel_skips_negative_slots ? 0 : 1;
      plan.enable_hybrid_cpu_decode =
          (plan.resident_slots < desc.num_experts) &&
          (policy.cpu_execution != onnxruntime::moe_offload::CpuExecutionMode::kDisabled) &&
          node_caps.supports_hybrid_doorbell_decode;
      plan.enable_streaming_dma_prefill =
          (plan.resident_slots < desc.num_experts) && node_caps.supports_gpu_streaming_prefill;
      plan.enable_background_promotion = (plan.spare_slots > 0) && is_adaptive;
      plan.admitted_gpu_expert_bytes =
          node_caps.gpu_fixed_table_bytes +
          static_cast<uint64_t>(plan.resident_slots + plan.spare_slots + plan.dummy_slots) * node_caps.gpu_slot_bytes;

      if (use_bounded_file_cache && plan.resident_slots < desc.num_experts && host_cache_counts[i] > 0 &&
          node_caps.cpu_slot_bytes > 0) {
        const int32_t cache_slots = host_cache_counts[i];
        plan.use_file_backing = true;
        plan.host_cache_slots = cache_slots;
        plan.host_cache_bytes = static_cast<uint64_t>(cache_slots) * node_caps.cpu_slot_bytes;
        plan.admitted_host_expert_bytes = plan.host_cache_bytes;
      } else {
        plan.use_file_backing = false;
        plan.host_cache_slots = 0;
        plan.host_cache_bytes = 0;
        plan.admitted_host_expert_bytes =
            static_cast<uint64_t>(desc.num_experts - plan.resident_slots) * node_caps.cpu_slot_bytes;
      }

      plan.ranked_experts = per_layer_ranked[i];
      plan.initial_resident_experts.assign(
          plan.ranked_experts.begin(),
          plan.ranked_experts.begin() + std::min<size_t>(static_cast<size_t>(plan.resident_slots),
                                                         plan.ranked_experts.size()));

      ORT_RETURN_IF_ERROR(nodes[i]->InitializeAdmittedPlan(policy, plan));
      if (!plan.initial_resident_experts.empty() || use_bounded_file_cache) {
#if defined(_WIN32)
        EvictUploadedHostPages(desc, plan.initial_resident_experts);
#else
        if (use_bounded_file_cache) {
          std::vector<int32_t> all_experts(static_cast<size_t>(desc.num_experts));
          for (int32_t e = 0; e < desc.num_experts; ++e) all_experts[static_cast<size_t>(e)] = e;
          EvictUploadedHostPages(desc, all_experts);
        } else if (!is_adaptive || plan.resident_slots >= desc.num_experts) {
          EvictUploadedHostPages(desc, plan.initial_resident_experts);
        }
#endif
      }

      total_admitted_gpu_bytes += plan.admitted_gpu_expert_bytes;
      total_resident_slots += plan.resident_slots;
      total_spare_slots += plan.spare_slots;
    }

#if defined(_WIN32)
    static_cast<void>(SetProcessWorkingSetSizeEx(
        GetCurrentProcess(), static_cast<SIZE_T>(-1), static_cast<SIZE_T>(-1), 0));
    uint64_t non_local_after = 0;
    QueryPlatformUsableDeviceMemory(device_id_, nullptr, nullptr, &non_local_after);
#ifdef LOGS_DEFAULT
    if (non_local_after > non_local_before + (64ULL << 20)) {
      LOGS_DEFAULT(WARNING) << "QMoE Offload v4 detected Windows WDDM Shared GPU Memory (non-local) paging delta of "
                            << ((non_local_after - non_local_before) >> 20)
                            << " MiB after session materialization. Increase session.moe.offload.reserve_device_bytes "
                               "or reduce session.moe.offload.device_expert_bytes.";
    }
#else
    (void)non_local_before;
    (void)non_local_after;
#endif
#endif

#ifdef LOGS_DEFAULT
    LOGS_DEFAULT(INFO) << "QMoE Offload v4 session plan committed across " << num_layers
                       << " layers: total_resident_slots=" << total_resident_slots
                       << ", total_spare_slots=" << total_spare_slots
                       << ", admitted_gpu_expert_mib=" << (total_admitted_gpu_bytes >> 20)
                       << ", bounded_file_cache=" << (use_bounded_file_cache ? 1 : 0);
#endif

    plan_committed = true;
    return common::Status::OK();
  }

 private:
  int device_id_ = 0;
  std::atomic<bool> shutting_down_{false};
  std::atomic<uint64_t> graph_generation_{0};
  mutable std::mutex mutex_;
  size_t registered_nodes_ = 0;
  bool run_in_flight_ = false;
  uint64_t current_run_id_ = 0;
  uint64_t next_invocation_id_ = 0;
  std::vector<std::shared_ptr<QMoEInvocationRecord>> active_invocations_;
  std::vector<GraphReplayNodeHook> graph_replay_hooks_;
  std::shared_ptr<QMoESessionLandingRing> landing_ring_;
  std::shared_ptr<SessionBudgetAccount> budget_account_;
  std::shared_ptr<CudaDeviceMemoryBudget> pinned_budget_;
  std::mutex pinned_pool_mutex_;
  std::shared_ptr<BoundedPinnedPool> pinned_pool_;
  std::filesystem::path model_path_;
  bool policy_initialized_ = false;
  bool session_plan_finalized_ = false;
  onnxruntime::moe_offload::MoeOffloadSessionPolicy policy_;
  std::vector<onnxruntime::moe_offload::IMoeOffloadNode*> moe_nodes_;
};

}  // namespace onnxruntime::cuda
