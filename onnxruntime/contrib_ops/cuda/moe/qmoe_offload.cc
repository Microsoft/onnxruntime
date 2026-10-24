#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/shared_library/provider_api.h"
#endif
#include "contrib_ops/cuda/moe/qmoe_offload.h"
#include "contrib_ops/cuda/moe/moe_quantization.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <nvtx3/nvToolsExt.h>
#include <sys/mman.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/moe/qmoe_offload_plan.h"
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/mlas/inc/mlas.h"
#if defined(_M_AMD64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// Satisfy MlasPlatformAvx512Supported() in the shared-provider bridge build so linking
// qmoe_shared_layout from onnxruntime_mlas does not pull in platform.cpp/threading.cpp
// (which reference onnxruntime_common's ThreadPool symbols not exported to shared providers).
bool MLASCALL MlasPlatformAvx512Supported(void) {
#if defined(_M_AMD64) || defined(__x86_64__)
  unsigned cpuid0[4]{0, 0, 0, 0};
  unsigned cpuid1[4]{0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuid(reinterpret_cast<int*>(cpuid0), 0);
  if (cpuid0[0] < 7u) return false;
  __cpuid(reinterpret_cast<int*>(cpuid1), 1);
#else
  __cpuid(0, cpuid0[0], cpuid0[1], cpuid0[2], cpuid0[3]);
  if (cpuid0[0] < 7u) return false;
  __cpuid(1, cpuid1[0], cpuid1[1], cpuid1[2], cpuid1[3]);
#endif
  constexpr unsigned kLeaf1EcxMask = (1u << 27) | (1u << 28) | (1u << 12) | (1u << 29);
  if ((cpuid1[2] & kLeaf1EcxMask) != kLeaf1EcxMask) return false;
  uint64_t xcr0 = 0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  uint32_t eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
#endif
  if ((xcr0 & 0xE6u) != 0xE6u) return false;
  unsigned cpuid7[4]{0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuidex(reinterpret_cast<int*>(cpuid7), 7, 0);
#else
  __cpuid_count(7, 0, cpuid7[0], cpuid7[1], cpuid7[2], cpuid7[3]);
#endif
  constexpr unsigned kLeaf7EbxMask = (1u << 5) | (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
  return (cpuid7[1] & kLeaf7EbxMask) == kLeaf7EbxMask;
#else
  return false;
#endif
}
#endif

namespace onnxruntime::contrib::cuda {

using namespace onnxruntime::cuda;
namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
namespace gemv = onnxruntime::llm::kernels::moe_gemv;

namespace {
std::atomic<int64_t> g_last_end_ns{0};
std::atomic<int64_t> g_step_start_ns{0}, g_step_end_ns{0};
std::atomic<uint64_t> g_gap_in_ns{0}, g_gap_in_n{0}, g_gap_out_ns{0}, g_gap_out_n{0};
std::atomic<uint64_t> g_t1_ns{0}, g_t2_ns{0}, g_t3_ns{0}, g_inter_n{0};
}  // namespace

namespace {

// Unified bounded CPU expert service (v3 Section 8.3 / F9, F16).
std::atomic<qmoe::SpinPool*> g_shared_spin_pool_ptr{nullptr};

qmoe::SpinPool& SharedSpinPool(int threads);

template <typename Function>
void OffloadParallelFor(OpKernelContext* context, ptrdiff_t count, Function& function) {
#ifdef ORT_USE_EP_API_ADAPTERS
  context->ParallelFor(
      [](void* data, size_t task) {
        (*static_cast<Function*>(data))(static_cast<ptrdiff_t>(task));
      },
      static_cast<size_t>(count), &function);
#else
  ORT_UNUSED_PARAMETER(context);
  SharedSpinPool(std::clamp(qmoe::DiscoverPhysicalCoreCount(), 1, 64)).ParallelFor(count, function, 1);
#endif
}

constexpr int kGroupedMoeMinRows =
    static_cast<int>(onnxruntime::llm::kernels::moe_gemv::kMaxProfiledExpandedRows) + 1;

size_t AddSlab(size_t bytes, size_t& total) {
  const size_t offset = (SafeInt<size_t>(total) + 255) / 256 * 256;
  total = SafeInt<size_t>(offset) + bytes;
  return offset;
}

qmoe::SpinPool& SharedSpinPool(int threads) {
  const int eff_threads = std::clamp(threads, 1, 256);
  qmoe::SpinPool* fast = g_shared_spin_pool_ptr.load(std::memory_order_acquire);
  if (fast != nullptr && fast->ThreadCount() == eff_threads) {
    return *fast;
  }
  static std::mutex pools_mutex;
  static std::unordered_map<int, std::unique_ptr<qmoe::SpinPool>> pools;
  std::lock_guard<std::mutex> lock(pools_mutex);
  auto& ptr = pools[eff_threads];
  if (!ptr) {
    const int default_spin_us = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_SPIN_US", 2000);
    qmoe::CpuBudgetGroup::Default().ConfigureDecodeSpinUs(default_spin_us);
    ptr = std::make_unique<qmoe::SpinPool>(eff_threads, default_spin_us);
  }
  g_shared_spin_pool_ptr.store(ptr.get(), std::memory_order_release);
  return *ptr;
}

bool IsStickyCudaContextError(cudaError_t err) noexcept {
  return err != cudaSuccess && err != cudaErrorNotReady && err != cudaErrorCudartUnloading;
}

inline bool IsInQMoEWarmup() noexcept {
  const char* v = std::getenv("ORT_QMOE_IN_WARMUP");
  return v != nullptr && v[0] == '1';
}

class StreamDrain final {
 public:
  StreamDrain(cudaStream_t stream, bool& failed, std::shared_ptr<bool> completed = {})
      : stream_(stream), failed_(failed), completed_(std::move(completed)) {}
  ~StreamDrain() {
    if (pending_ && cudaStreamSynchronize(stream_) != cudaSuccess) {
      failed_ = true;
    }
    if (completed_ && !failed_) *completed_ = true;
  }
  cudaError_t Synchronize() {
    const auto error = cudaStreamSynchronize(stream_);
    failed_ = failed_ || error != cudaSuccess;
    return error;
  }
  cudaError_t Wait() {
    const auto error = Synchronize();
    pending_ = error != cudaSuccess;
    return error;
  }
  void Disarm() {
    pending_ = false;
    completed_.reset();
  }

 private:
  cudaStream_t stream_;
  bool& failed_;
  bool pending_ = true;
  std::shared_ptr<bool> completed_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StreamDrain);
};

// Row conversions between FP16/BF16 and FP32 delegate to qmoe_shared_layout (AVX-512 when supported by compiler and CPU, scalar otherwise).
void ActToFloatRow(const MLFloat16* source, float* destination, int count) {
  qmoe_shared::HalfBitsToFloatRow(reinterpret_cast<const uint16_t*>(source), destination, count);
}

void FloatToActRow(const float* source, MLFloat16* destination, int count) {
  qmoe_shared::FloatToHalfBitsRow(source, reinterpret_cast<uint16_t*>(destination), count);
}

void ActToFloatRow(const BFloat16* source, float* destination, int count) {
  qmoe_shared::BFloat16BitsToFloatRow(reinterpret_cast<const uint16_t*>(source), destination, count);
}

void FloatToActRow(const float* source, BFloat16* destination, int count) {
  qmoe_shared::FloatToBFloat16BitsRow(source, reinterpret_cast<uint16_t*>(destination), count);
}

}  // namespace

struct QMoEOffload::CpuWork {
  struct ExpertCall {
    int expert;
    size_t first;
    int count;
  };
  struct alignas(64) PaddedAtomicInt {
    std::atomic<int> value{0};
  };
  static constexpr size_t kMaxGroupSlots = 64;
  InlinedVector<float> reordered;
  InlinedVector<float> intermediate;
  InlinedVector<float> result;
  InlinedVector<float> natural;
  InlinedVector<float> token_rows;
  InlinedVector<char> token_ready;
  InlinedVector<uint8_t> expert_invocation_rows;
  InlinedVector<InlinedVector<int>> expert_routes;
  InlinedVector<ExpertCall> calls;
  std::unique_ptr<PaddedAtomicInt[]> fc1_done = std::make_unique<PaddedAtomicInt[]>(kMaxGroupSlots);
  std::array<const float*, kMaxGroupSlots> slot_act_in{};
  std::array<qmoe_shared::Nvfp4ExpertView, kMaxGroupSlots> slot_view{};

  CpuWork(int group_calls, int num_experts, int hidden_size, int inter_size)
      : reordered(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * hidden_size)),
        intermediate(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * inter_size)),
        result(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * hidden_size)),
        natural(hidden_size),
        expert_invocation_rows(num_experts, 0),
        expert_routes(num_experts) {}
};

class SessionPromotionBudget {
 public:
  static SessionPromotionBudget& Instance() {
    static SessionPromotionBudget instance;
    return instance;
  }

  // Each session owns its step counters and pinned bounce buffers so interleaved sessions cannot
  // reset each other's step or overwrite DMA sources that are still in flight.
  void Retain(const void* session_key) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++sessions_[session_key].refs;
  }

  // Frees the session's bounce buffers when its last owner is gone. Callers must have drained their streams.
  void Release(const void* session_key) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_key);
    if (it == sessions_.end()) return;
    if (--it->second.refs <= 0) {
      it->second.FreeBounce();
      sessions_.erase(it);
    }
  }

  void EnsureBounceAllocated(const void* session_key, size_t max_bytes) noexcept {
    if (max_bytes == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    SessionState& st = sessions_[session_key];
    for (int b = 0; b < 2; ++b) st.EnsureCapacity(b, max_bytes);
  }

  bool TryAcquire(const void* session_key, uint64_t step_seq, int max_promotions,
                  size_t max_bytes, size_t promo_bytes,
                  bool need_pinned_bounce = false, uint8_t** out_pinned_bounce = nullptr) noexcept {
    if (out_pinned_bounce) *out_pinned_bounce = nullptr;
    if (max_promotions <= 0 && max_bytes == 0) return true;
    const size_t aligned_promo_bytes = (promo_bytes + 4095u) & ~size_t{4095u};
    std::lock_guard<std::mutex> lock(mutex_);
    SessionState& st = sessions_[session_key];
    if (!st.has_step || step_seq != st.current_step) {
      st.has_step = true;
      st.current_step = step_seq;
      st.promotions_in_step = 0;
      st.bytes_in_step = 0;
      st.step_epoch ^= 1;
    }
    if (max_promotions > 0 && st.promotions_in_step >= max_promotions) return false;
    if (max_bytes > 0 && st.promotions_in_step > 0 && st.bytes_in_step + aligned_promo_bytes > max_bytes) return false;
    if (need_pinned_bounce && out_pinned_bounce) {
      const int e = st.step_epoch;
      st.EnsureCapacity(e, std::max(max_bytes, st.bytes_in_step + aligned_promo_bytes));
      if (st.pinned_bounce[e] != nullptr && st.bytes_in_step + aligned_promo_bytes <= st.pinned_cap[e]) {
        *out_pinned_bounce = st.pinned_bounce[e] + st.bytes_in_step;
      }
    }
    ++st.promotions_in_step;
    st.bytes_in_step += aligned_promo_bytes;
    return true;
  }

 private:
  struct SessionState {
    int refs = 0;
    bool has_step = false;
    uint64_t current_step = 0;
    int promotions_in_step = 0;
    size_t bytes_in_step = 0;
    int step_epoch = 0;
    uint8_t* pinned_bounce[2]{nullptr, nullptr};
    size_t pinned_cap[2]{0, 0};

    void EnsureCapacity(int b, size_t bytes) noexcept {
      if (pinned_bounce[b] != nullptr && pinned_cap[b] >= bytes) return;
      if (pinned_bounce[b] != nullptr) {
        static_cast<void>(cudaFreeHost(pinned_bounce[b]));
        pinned_bounce[b] = nullptr;
        pinned_cap[b] = 0;
      }
      void* p = nullptr;
      if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) == cudaSuccess && p != nullptr) {
        pinned_bounce[b] = static_cast<uint8_t*>(p);
        pinned_cap[b] = bytes;
      } else {
        static_cast<void>(cudaGetLastError());
      }
    }

    void FreeBounce() noexcept {
      for (int b = 0; b < 2; ++b) {
        if (pinned_bounce[b] != nullptr) static_cast<void>(cudaFreeHost(pinned_bounce[b]));
        pinned_bounce[b] = nullptr;
        pinned_cap[b] = 0;
      }
    }
  };

  ~SessionPromotionBudget() {
    for (auto& kv : sessions_) kv.second.FreeBounce();
  }

  std::mutex mutex_;
  std::map<const void*, SessionState> sessions_;
};

struct QMoEOffload::SharedLayerCache {
  int device_id = -1;
  int64_t layer_index = -1;
  int num_experts = 0;
  int hidden_size = 0;
  int inter_size = 0;
  int resident_slots = 0;
  int physical_slots = 0;
  std::array<size_t, 6> bank_bytes{};
  std::array<size_t, 6> bank_offsets{};
  uint8_t* slab_ptr = nullptr;
  const int* slot_table_dev_ptr = nullptr;
  const InlinedVector<int>* expert_slot_ptr = nullptr;
  const InlinedVector<int32_t>* host_slot_ptr = nullptr;
  const QMoEOffload* owner = nullptr;
  const void* session_scope = nullptr;
  const uint8_t* fc1_w_id = nullptr;
  qmoe_shared::FileExpertStore* file_store_ptr = nullptr;
  qmoe_shared::Nvfp4ExpertBlockLayout nvfp4_layout{};
  bool host_cache_mapped = false;

  static std::mutex& Mutex() {
    static std::mutex m;
    return m;
  }
  static std::map<std::pair<int, int64_t>, std::weak_ptr<SharedLayerCache>>& Registry() {
    static std::map<std::pair<int, int64_t>, std::weak_ptr<SharedLayerCache>> reg;
    return reg;
  }
  static std::map<std::pair<int, int64_t>, std::vector<std::weak_ptr<SharedLayerCache>>>& MultiRegistry() {
    static std::map<std::pair<int, int64_t>, std::vector<std::weak_ptr<SharedLayerCache>>> reg;
    return reg;
  }
  static void Register(const std::shared_ptr<SharedLayerCache>& entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> lock(Mutex());
    Registry()[{entry->device_id, entry->layer_index}] = entry;
    auto& vec = MultiRegistry()[{entry->device_id, entry->layer_index}];
    vec.erase(std::remove_if(vec.begin(), vec.end(),
                             [&entry](const std::weak_ptr<SharedLayerCache>& w) {
                               auto sp = w.lock();
                               return !sp || sp->owner == nullptr || sp->owner == entry->owner;
                             }),
              vec.end());
    vec.push_back(entry);
  }
  static std::shared_ptr<SharedLayerCache> FindMatching(int dev_id, int64_t l_idx,
                                                        int num_e, int hid, int inter,
                                                        const uint8_t* fc1_w = nullptr,
                                                        const void* scope = nullptr) {
    std::lock_guard<std::mutex> lock(Mutex());
    auto& mreg = MultiRegistry();
    auto mit = mreg.find({dev_id, l_idx});
    if (mit != mreg.end()) {
      for (auto it = mit->second.rbegin(); it != mit->second.rend(); ++it) {
        auto sp = it->lock();
        if (!sp || sp->owner == nullptr) continue;
        if (sp->num_experts != num_e || sp->hidden_size != hid || sp->inter_size != inter) continue;
        if (fc1_w != nullptr && sp->fc1_w_id != nullptr && sp->fc1_w_id != fc1_w) continue;
        if (scope != nullptr && sp->session_scope != nullptr && sp->session_scope != scope) continue;
        return sp;
      }
    }
    auto& reg = Registry();
    auto it = reg.find({dev_id, l_idx});
    if (it != reg.end()) {
      auto sp = it->second.lock();
      if (sp && sp->owner != nullptr &&
          sp->num_experts == num_e && sp->hidden_size == hid && sp->inter_size == inter &&
          (fc1_w == nullptr || sp->fc1_w_id == nullptr || sp->fc1_w_id == fc1_w) &&
          (scope == nullptr || sp->session_scope == nullptr || sp->session_scope == scope)) {
        return sp;
      }
    }
    return nullptr;
  }
};

namespace {

std::atomic<bool> g_prefill_dma_active{false};
std::atomic<int64_t> g_step_prefill_remaining_tokens{0};

struct DenseLayerHostWeights {
  int num_experts = 0;
  int hidden_size = 0;
  int inter_size = 0;
  const uint8_t* fc1_w = nullptr;
  const uint8_t* fc1_s = nullptr;
  const float* fc1_g = nullptr;
  const uint8_t* fc2_w = nullptr;
  const uint8_t* fc2_s = nullptr;
  const float* fc2_g = nullptr;
};

struct DevicePrefillDmaState {
  std::mutex mutex;
  std::mutex lease_mutex;
  cudaEvent_t lease_done_event = nullptr;
  bool lease_done_recorded = false;
  std::map<int64_t, DenseLayerHostWeights> dense_layers;
  cudaStream_t dma_stream = nullptr;
  cudaEvent_t dma_done_event[2]{nullptr, nullptr};
  cudaEvent_t compute_done_event[2]{nullptr, nullptr};
  bool dma_done_recorded[2]{false, false};
  bool compute_done_recorded[2]{false, false};
  int64_t buffered_layer[2]{-1, -1};
  int buffered_pass[2]{-1, -1};
  int buffered_cold_count[2]{0, 0};
  bool buffered_compact_pack[2]{false, false};
  int64_t current_tile_rows = 0;

  uint8_t* host_staging[2]{nullptr, nullptr};
  size_t host_staging_bytes = 0;
  size_t host_staging_accounted_bytes = 0;
  std::shared_ptr<CudaDeviceMemoryBudget> host_staging_session_budget;
  int* host_active_mask = nullptr;
  int host_active_mask_cap = 0;
  size_t host_active_mask_accounted_bytes = 0;
  std::shared_ptr<CudaDeviceMemoryBudget> host_active_mask_session_budget;
  int64_t active_mask_layer = -1;
  uint8_t* workspace_dev = nullptr;
  size_t workspace_bytes = 0;
  int workspace_runner_experts = 0;
  int workspace_num_experts = 0;
  int workspace_hidden = 0;
  int workspace_inter = 0;
  size_t dequant_fc1_offset = 0;
  size_t dequant_fc2_offset = 0;
  size_t staging_fc1_w_offset[2]{0, 0};
  size_t staging_fc2_w_offset[2]{0, 0};
  size_t staging_fc1_s_offset[2]{0, 0};
  size_t staging_fc2_s_offset[2]{0, 0};
  size_t staging_fc1_g_offset[2]{0, 0};
  size_t staging_fc2_g_offset[2]{0, 0};
  size_t hist_offset = 0;
  int max_hist_layers = 64;

  bool prefill_active = false;
  int64_t total_prefill_tokens_accum = 0;
  int last_new_turn_prompt_tokens = 0;
  size_t total_lent_bytes = 0;
  std::vector<int64_t> lent_layers;
  const void* active_session_scope = nullptr;
  int* hist_dev = nullptr;
  size_t hist_bytes_cap = 0;
  int hist_expert_stride = 0;
  int hist_layer_cap = 0;
  bool use_uva_in_dma_pass = false;
  uint64_t mmap_touched_layer_mask = 0;

  void FreeHostStagingLocked() noexcept {
    bool all_ok = true;
    for (int b = 0; b < 2; ++b) {
      if (host_staging[b]) {
        if (cudaFreeHost(host_staging[b]) != cudaSuccess) all_ok = false;
        host_staging[b] = nullptr;
      }
      dma_done_recorded[b] = false;
    }
    if (host_staging_accounted_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (host_staging_session_budget) host_staging_session_budget->BeginFree(host_staging_accounted_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(host_staging_accounted_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(host_staging_accounted_bytes, all_ok);
      if (host_staging_session_budget) host_staging_session_budget->FinishFree(host_staging_accounted_bytes, all_ok);
      host_staging_accounted_bytes = 0;
    }
    host_staging_session_budget.reset();
    host_staging_bytes = 0;
  }

  void FreeHostActiveMaskLocked() noexcept {
    bool ok = true;
    if (host_active_mask) {
      ok = (cudaFreeHost(host_active_mask) == cudaSuccess);
      host_active_mask = nullptr;
      host_active_mask_cap = 0;
    }
    if (host_active_mask_accounted_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (host_active_mask_session_budget) host_active_mask_session_budget->BeginFree(host_active_mask_accounted_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(host_active_mask_accounted_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(host_active_mask_accounted_bytes, ok);
      if (host_active_mask_session_budget) {
        host_active_mask_session_budget->FinishFree(host_active_mask_accounted_bytes, ok);
      }
      host_active_mask_accounted_bytes = 0;
    }
    host_active_mask_session_budget.reset();
    active_mask_layer = -1;
  }

  void FreeHistDevLocked(int device_id) noexcept {
    if (hist_dev) {
      auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id);
      if (dev_budget && hist_bytes_cap > 0) dev_budget->BeginFree(hist_bytes_cap);
      const bool ok = (cudaFree(hist_dev) == cudaSuccess);
      if (dev_budget && hist_bytes_cap > 0) dev_budget->FinishFree(hist_bytes_cap, ok);
      hist_dev = nullptr;
    }
    hist_bytes_cap = 0;
    hist_expert_stride = 0;
    hist_layer_cap = 0;
  }

  static DevicePrefillDmaState& ForDevice(int device_id) {
    static std::mutex map_mutex;
    static std::map<int, std::unique_ptr<DevicePrefillDmaState>> states;
    std::lock_guard<std::mutex> lock(map_mutex);
    auto& ptr = states[device_id];
    if (!ptr) ptr = std::make_unique<DevicePrefillDmaState>();
    return *ptr;
  }
};

bool EnqueuePrefillPassDmaLocked(
    DevicePrefillDmaState& state,
    int device_id,
    int64_t target_layer,
    int target_pass,
    int runner_num_experts,
    int buf_idx) {
  if (state.buffered_layer[buf_idx] == target_layer && state.buffered_pass[buf_idx] == target_pass) {
    return true;
  }
  auto it = state.dense_layers.find(target_layer);
  if (it == state.dense_layers.end()) return false;
  const DenseLayerHostWeights& wl = it->second;
  if (!wl.fc1_w || !wl.fc1_s || !wl.fc1_g || !wl.fc2_w || !wl.fc2_s || !wl.fc2_g) return false;

  const int pass_start = target_pass * runner_num_experts;
  if (pass_start >= wl.num_experts) return false;
  const int count = std::min(runner_num_experts, wl.num_experts - pass_start);

  std::shared_ptr<QMoEOffload::SharedLayerCache> locked =
      QMoEOffload::SharedLayerCache::FindMatching(
          device_id, target_layer, wl.num_experts, wl.hidden_size, wl.inter_size,
          wl.fc1_w, state.active_session_scope);
  const InlinedVector<int>* expert_slot_ptr = nullptr;
  const InlinedVector<int32_t>* host_slot_ptr = nullptr;
  qmoe_shared::FileExpertStore* file_store_ptr = nullptr;
  qmoe_shared::Nvfp4ExpertBlockLayout nvfp4_layout{};
  bool host_cache_uva_mapped = false;
  if (locked) {
    if (locked->owner) {
      locked->owner->SyncPrefillSlotTables(state.dma_stream);
    }
    host_cache_uva_mapped = locked->host_cache_mapped;
    if (locked->slab_ptr != nullptr && locked->expert_slot_ptr != nullptr && locked->resident_slots > 0) {
      expert_slot_ptr = locked->expert_slot_ptr;
    }
    if (locked->file_store_ptr != nullptr) {
      file_store_ptr = locked->file_store_ptr;
      nvfp4_layout = locked->nvfp4_layout;
      host_slot_ptr = locked->host_slot_ptr;
    }
  }

  if (state.compute_done_recorded[buf_idx]) {
    if (cudaStreamWaitEvent(state.dma_stream, state.compute_done_event[buf_idx], 0) != cudaSuccess) return false;
  }

  const size_t fc1_n = static_cast<size_t>(wl.inter_size) * 2;
  const size_t fc1_k = static_cast<size_t>(wl.hidden_size);
  const size_t fc2_n = static_cast<size_t>(wl.hidden_size);
  const size_t fc2_k = static_cast<size_t>(wl.inter_size);
  const size_t fc1_w_stride = fc1_n * (fc1_k / 2);
  const size_t fc2_w_stride = fc2_n * (fc2_k / 2);
  const size_t fc1_s_stride = fc1_n * (fc1_k / 16);
  const size_t fc2_s_stride = fc2_n * (fc2_k / 16);

  uint8_t* dst_fc1_w = state.workspace_dev + state.staging_fc1_w_offset[buf_idx];
  uint8_t* dst_fc2_w = state.workspace_dev + state.staging_fc2_w_offset[buf_idx];
  uint8_t* dst_fc1_s = state.workspace_dev + state.staging_fc1_s_offset[buf_idx];
  uint8_t* dst_fc2_s = state.workspace_dev + state.staging_fc2_s_offset[buf_idx];
  uint8_t* dst_fc1_g = state.workspace_dev + state.staging_fc1_g_offset[buf_idx];
  uint8_t* dst_fc2_g = state.workspace_dev + state.staging_fc2_g_offset[buf_idx];

  const uint8_t* src_fc1_s = wl.fc1_s + static_cast<size_t>(pass_start) * fc1_s_stride;
  const uint8_t* src_fc2_s = wl.fc2_s + static_cast<size_t>(pass_start) * fc2_s_stride;
  const uint8_t* src_fc1_w_base = wl.fc1_w + static_cast<size_t>(pass_start) * fc1_w_stride;
  const uint8_t* src_fc2_w_base = wl.fc2_w + static_cast<size_t>(pass_start) * fc2_w_stride;

  if (state.host_staging[buf_idx] != nullptr) {
    const uint8_t* cache_base = (file_store_ptr != nullptr) ? file_store_ptr->CacheBase() : nullptr;
    const bool has_active_mask =
        (state.active_mask_layer == target_layer && state.host_active_mask != nullptr &&
         wl.num_experts <= state.host_active_mask_cap);
    if (state.dma_done_recorded[buf_idx]) {
      static_cast<void>(cudaEventSynchronize(state.dma_done_event[buf_idx]));
    }
    const size_t map_bytes = (static_cast<size_t>(runner_num_experts) * sizeof(int32_t) + 255) & ~size_t{255};
    const size_t g_bytes = (static_cast<size_t>(wl.num_experts) * sizeof(float) + 255) & ~size_t{255};
    const size_t hdr_bytes = map_bytes + 2 * g_bytes;

    uint8_t* hbuf = state.host_staging[buf_idx];
    auto* h_cold_map = reinterpret_cast<int32_t*>(hbuf);
    for (int i = 0; i < runner_num_experts; ++i) {
      h_cold_map[i] = -1;
    }
    InlinedVector<int> uncached_idx;
    for (int i = 0; i < count; ++i) {
      const int src_e = pass_start + i;
      if (has_active_mask && state.host_active_mask[src_e] < 0) {
        h_cold_map[i] = -1;
      } else if (expert_slot_ptr != nullptr && (*expert_slot_ptr)[src_e] >= 0) {
        h_cold_map[i] = -1;
      } else if (state.use_uva_in_dma_pass && host_cache_uva_mapped &&
                 host_slot_ptr != nullptr && (*host_slot_ptr)[src_e] >= 0) {
        h_cold_map[i] = -2 - (*host_slot_ptr)[src_e];
      } else {
        h_cold_map[i] = static_cast<int32_t>(uncached_idx.size());
        uncached_idx.push_back(i);
      }
    }
    std::memcpy(hbuf + map_bytes, wl.fc1_g, static_cast<size_t>(wl.num_experts) * sizeof(float));
    std::memcpy(hbuf + map_bytes + g_bytes, wl.fc2_g, static_cast<size_t>(wl.num_experts) * sizeof(float));

    const size_t M = uncached_idx.size();
    uint8_t* h_fc1_s = hbuf + hdr_bytes;
    uint8_t* h_fc2_s = h_fc1_s + M * fc1_s_stride;
    uint8_t* h_fc1_w = h_fc2_s + M * fc2_s_stride;
    uint8_t* h_fc2_w = h_fc1_w + M * fc1_w_stride;
    if (M > 0) {
      std::atomic<bool> touched_mmap{false};
      auto copy_expert_fn = [&](ptrdiff_t k_idx) noexcept {
        const size_t dst_k = static_cast<size_t>(k_idx);
        const size_t i = static_cast<size_t>(uncached_idx[dst_k]);
        const size_t src_e = static_cast<size_t>(pass_start) + i;
        if (cache_base != nullptr && host_slot_ptr != nullptr) {
          const int h_slot = (*host_slot_ptr)[src_e];
          if (h_slot >= 0) {
            const uint8_t* slot_data = cache_base + static_cast<size_t>(h_slot) * nvfp4_layout.stride;
            std::memcpy(h_fc1_s + dst_k * fc1_s_stride, slot_data + nvfp4_layout.fc1_scales, fc1_s_stride);
            std::memcpy(h_fc2_s + dst_k * fc2_s_stride, slot_data + nvfp4_layout.fc2_scales, fc2_s_stride);
            std::memcpy(h_fc1_w + dst_k * fc1_w_stride, slot_data + nvfp4_layout.fc1_weights, fc1_w_stride);
            std::memcpy(h_fc2_w + dst_k * fc2_w_stride, slot_data + nvfp4_layout.fc2_weights, fc2_w_stride);
            return;
          }
        }
        touched_mmap.store(true, std::memory_order_relaxed);
        std::memcpy(h_fc1_s + dst_k * fc1_s_stride, wl.fc1_s + src_e * fc1_s_stride, fc1_s_stride);
        std::memcpy(h_fc2_s + dst_k * fc2_s_stride, wl.fc2_s + src_e * fc2_s_stride, fc2_s_stride);
        std::memcpy(h_fc1_w + dst_k * fc1_w_stride, wl.fc1_w + src_e * fc1_w_stride, fc1_w_stride);
        std::memcpy(h_fc2_w + dst_k * fc2_w_stride, wl.fc2_w + src_e * fc2_w_stride, fc2_w_stride);
      };
      static const int pool_threads = []() {
        const int default_pool_threads =
            std::clamp(qmoe::DiscoverPhysicalCoreCount(), 1, 64);
        return std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", default_pool_threads), 1, 256);
      }();
      SharedSpinPool(pool_threads).ParallelFor(static_cast<ptrdiff_t>(M), copy_expert_fn, 1);
      if (touched_mmap.load(std::memory_order_relaxed) && target_layer >= 0 && target_layer < 64) {
        state.mmap_touched_layer_mask |= (uint64_t{1} << target_layer);
      }
    }
    const size_t total_dma_bytes =
        hdr_bytes + M * (fc1_s_stride + fc2_s_stride + fc1_w_stride + fc2_w_stride);
    if (cudaMemcpyAsync(dst_fc1_w, hbuf, total_dma_bytes, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
      return false;
    }
    state.buffered_cold_count[buf_idx] = static_cast<int>(M);
    state.buffered_compact_pack[buf_idx] = true;
  } else {
    state.buffered_cold_count[buf_idx] = count;
    state.buffered_compact_pack[buf_idx] = false;
    if (cudaMemcpyAsync(dst_fc1_g, wl.fc1_g, static_cast<size_t>(wl.num_experts) * sizeof(float),
                        cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpyAsync(dst_fc2_g, wl.fc2_g, static_cast<size_t>(wl.num_experts) * sizeof(float),
                        cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpyAsync(dst_fc1_s, src_fc1_s,
                        static_cast<size_t>(count) * fc1_s_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpyAsync(dst_fc2_s, src_fc2_s,
                        static_cast<size_t>(count) * fc2_s_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
      return false;
    }

    if (expert_slot_ptr == nullptr) {
      if (cudaMemcpyAsync(dst_fc1_w, src_fc1_w_base,
                          static_cast<size_t>(count) * fc1_w_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
        return false;
      }
      if (cudaMemcpyAsync(dst_fc2_w, src_fc2_w_base,
                          static_cast<size_t>(count) * fc2_w_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
        return false;
      }
    } else {
      // Coalesce cold expert runs; merge across single-expert resident gaps so PCIe DMA bursts stay >= 3-10 MiB.
      int i = 0;
      while (i < count) {
        if ((*expert_slot_ptr)[pass_start + i] >= 0) {
          ++i;
          continue;
        }
        const int r0 = i;
        ++i;
        while (i < count) {
          if ((*expert_slot_ptr)[pass_start + i] < 0) {
            ++i;
          } else if (i + 1 < count && (*expert_slot_ptr)[pass_start + i + 1] < 0) {
            i += 2;  // bridge single resident expert to avoid splitting DMA descriptor
          } else {
            break;
          }
        }
        const int r1 = i;
        const size_t run_len = static_cast<size_t>(r1 - r0);
        if (cudaMemcpyAsync(dst_fc1_w + static_cast<size_t>(r0) * fc1_w_stride,
                            src_fc1_w_base + static_cast<size_t>(r0) * fc1_w_stride,
                            run_len * fc1_w_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
          return false;
        }
        if (cudaMemcpyAsync(dst_fc2_w + static_cast<size_t>(r0) * fc2_w_stride,
                            src_fc2_w_base + static_cast<size_t>(r0) * fc2_w_stride,
                            run_len * fc2_w_stride, cudaMemcpyHostToDevice, state.dma_stream) != cudaSuccess) {
          return false;
        }
      }
    }
  }

  if (cudaEventRecord(state.dma_done_event[buf_idx], state.dma_stream) != cudaSuccess) {
    return false;
  }
  static_cast<void>(cudaStreamQuery(state.dma_stream));
  state.dma_done_recorded[buf_idx] = true;
  state.buffered_layer[buf_idx] = target_layer;
  state.buffered_pass[buf_idx] = target_pass;
  return true;
}

}  // namespace

bool TryGetQMoEResidentLayerSlabView(
    int device_id,
    int64_t layer_index,
    int num_experts,
    int hidden_size,
    int inter_size,
    QMoEResidentLayerSlabView* out_view,
    cudaStream_t stream,
    const uint8_t* fc1_w_id) {
  if (!out_view || layer_index < 0) return false;
  std::shared_ptr<QMoEOffload::SharedLayerCache> locked =
      QMoEOffload::SharedLayerCache::FindMatching(
          device_id, layer_index, num_experts, hidden_size, inter_size, fc1_w_id, nullptr);
  if (!locked || locked->num_experts != num_experts || locked->hidden_size != hidden_size ||
      locked->inter_size != inter_size || !locked->slot_table_dev_ptr ||
      locked->resident_slots <= 0) {
    return false;
  }
  const bool has_gpu_slab = (locked->slab_ptr != nullptr && locked->expert_slot_ptr != nullptr);
  if (has_gpu_slab) {
    out_view->fc1_weights = locked->slab_ptr + locked->bank_offsets[0];
    out_view->fc2_weights = locked->slab_ptr + locked->bank_offsets[1];
    out_view->fc1_scales = locked->slab_ptr + locked->bank_offsets[2];
    out_view->fc2_scales = locked->slab_ptr + locked->bank_offsets[3];
    out_view->slot_table_dev = locked->slot_table_dev_ptr;
  }
  if (locked->owner) {
    locked->host_cache_mapped = locked->owner->host_cache_mapped_;
  }
  if (locked->owner && locked->owner->host_cache_mapped_ && locked->file_store_ptr != nullptr) {
    locked->owner->SyncPrefillSlotTables(stream);
    if (const uint8_t* cache_base = locked->file_store_ptr->CacheBase()) {
      out_view->host_fc1_weights = cache_base + locked->nvfp4_layout.fc1_weights;
      out_view->host_fc2_weights = cache_base + locked->nvfp4_layout.fc2_weights;
      out_view->host_fc1_scales = cache_base + locked->nvfp4_layout.fc1_scales;
      out_view->host_fc2_scales = cache_base + locked->nvfp4_layout.fc2_scales;
      out_view->host_expert_stride = locked->nvfp4_layout.stride;
      out_view->host_slot_table_dev = locked->slot_table_dev_ptr + num_experts;
      out_view->all_experts_cached = true;
      for (int e = 0; e < num_experts; ++e) {
        const bool in_gpu = has_gpu_slab && ((*locked->expert_slot_ptr)[e] >= 0);
        const bool in_host = (locked->host_slot_ptr != nullptr) && ((*locked->host_slot_ptr)[e] >= 0);
        if (!in_gpu && !in_host) {
          out_view->all_experts_cached = false;
          break;
        }
      }
    }
  }
  return out_view->slot_table_dev != nullptr || out_view->host_slot_table_dev != nullptr;
}

void RegisterQMoEDenseLayerHostWeights(
    int device_id, int64_t layer_index, int num_experts, int hidden_size, int inter_size,
    const uint8_t* fc1_w, const uint8_t* fc1_s, const float* fc1_g,
    const uint8_t* fc2_w, const uint8_t* fc2_s, const float* fc2_g) {
  if (layer_index < 0 || !fc1_w || !fc1_s || !fc1_g || !fc2_w || !fc2_s || !fc2_g) return;
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.dense_layers[layer_index] = DenseLayerHostWeights{
      num_experts, hidden_size, inter_size, fc1_w, fc1_s, fc1_g, fc2_w, fc2_s, fc2_g};
}

void UnregisterQMoEDenseLayerHostWeights(
    int device_id, int64_t layer_index, const uint8_t* fc1_w) {
  if (device_id < 0 || layer_index < 0) return;
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.dense_layers.find(layer_index);
  if (it != state.dense_layers.end() && (fc1_w == nullptr || it->second.fc1_w == fc1_w)) {
    state.dense_layers.erase(it);
    for (int b = 0; b < 2; ++b) {
      if (state.buffered_layer[b] == layer_index) {
        state.buffered_layer[b] = -1;
        state.buffered_pass[b] = -1;
      }
    }
  }
  if (state.dense_layers.empty()) {
    int prev_dev = -1;
    const bool restore_dev = (cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != device_id);
    if (restore_dev) static_cast<void>(cudaSetDevice(device_id));
    if (state.dma_stream) {
      static_cast<void>(cudaStreamSynchronize(state.dma_stream));
    }
    state.FreeHostStagingLocked();
    state.FreeHostActiveMaskLocked();
    state.FreeHistDevLocked(device_id);
    if (state.workspace_dev) {
      auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id);
      if (dev_budget && state.workspace_bytes > 0) dev_budget->BeginFree(state.workspace_bytes);
      const bool ok = (cudaFree(state.workspace_dev) == cudaSuccess);
      if (dev_budget && state.workspace_bytes > 0) dev_budget->FinishFree(state.workspace_bytes, ok);
      state.workspace_dev = nullptr;
      state.workspace_bytes = 0;
      state.workspace_runner_experts = 0;
      state.workspace_num_experts = 0;
    }
    state.lent_layers.clear();
    state.total_lent_bytes = 0;
    state.prefill_active = false;
    state.total_prefill_tokens_accum = 0;
    state.active_session_scope = nullptr;
    state.mmap_touched_layer_mask = 0;
    if (restore_dev) static_cast<void>(cudaSetDevice(prev_dev));
  }
}

QMoEPrefillDmaLease::QMoEPrefillDmaLease(int device_id, cudaStream_t exec_stream)
    : device_id_(device_id), exec_stream_(exec_stream) {
  auto& state = DevicePrefillDmaState::ForDevice(device_id_);
  state.lease_mutex.lock();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.lease_done_event == nullptr) {
    if (cudaEventCreateWithFlags(&state.lease_done_event, cudaEventDisableTiming) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      state.lease_done_event = nullptr;
      return;
    }
  }
  if (state.lease_done_recorded) {
    static_cast<void>(cudaStreamWaitEvent(exec_stream_, state.lease_done_event, 0));
    if (state.dma_stream) static_cast<void>(cudaStreamWaitEvent(state.dma_stream, state.lease_done_event, 0));
  }
}

QMoEPrefillDmaLease::~QMoEPrefillDmaLease() {
  auto& state = DevicePrefillDmaState::ForDevice(device_id_);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.lease_done_event != nullptr &&
        cudaEventRecord(state.lease_done_event, exec_stream_) == cudaSuccess) {
      state.lease_done_recorded = true;
    } else {
      static_cast<void>(cudaGetLastError());
    }
  }
  state.lease_mutex.unlock();
}

bool TryPrepareQMoEPrefillDmaLayer(
    int device_id, int64_t layer_index, int num_experts, int hidden_size, int inter_size,
    int runner_num_experts, int64_t tile_rows, size_t elem_size,
    const uint8_t* fc1_w, const uint8_t* fc1_s, const float* fc1_g,
    const uint8_t* fc2_w, const uint8_t* fc2_s, const float* fc2_g,
    cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view) {
  if (!out_view || layer_index < 0 || runner_num_experts <= 0 || runner_num_experts >= num_experts) return false;
  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ENABLE_PREFILL_DMA", 1) == 0) return false;

  bool has_offload_cache = false;
  bool bounded_host_cache = false;
  (void)bounded_host_cache;
  bool has_uncached_cold_experts = false;
  const void* req_session_scope = nullptr;
  const QMoEOffload* matched_owner = nullptr;
  if (auto matched = QMoEOffload::SharedLayerCache::FindMatching(
          device_id, layer_index, num_experts, hidden_size, inter_size, fc1_w, nullptr)) {
    if (matched->resident_slots >= 0) {
      has_offload_cache = true;
      req_session_scope = matched->session_scope;
      matched_owner = matched->owner;
      if (matched->owner) {
        if (matched->owner->host_cache_bytes_ > 0) {
          bounded_host_cache = true;
          const size_t covered = matched->owner->host_cache_mapped_
                                     ? (matched->owner->file_expert_store_.CacheSlots() +
                                        static_cast<size_t>(matched->owner->base_resident_slots_))
                                     : 0;
          if (covered < static_cast<size_t>(num_experts)) {
            has_uncached_cold_experts = true;
          }
        } else if (!matched->owner->all_expert_banks_registered_ &&
                   matched->owner->base_resident_slots_ < num_experts) {
          has_uncached_cold_experts = true;
        }
      }
    }
  }
  if (!has_offload_cache) return false;

  if (layer_index == 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_NEW_TURN", 0) == 1) {
    g_step_prefill_remaining_tokens.store(0, std::memory_order_release);
#if defined(_WIN32)
    _putenv_s("ORT_QMOE_PREFILL_NEW_TURN", "0");
#else
    setenv("ORT_QMOE_PREFILL_NEW_TURN", "0", 1);
#endif
    FinalizeQMoEPrefillAndSeedDecode(device_id, exec_stream);
  }

  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.active_session_scope = req_session_scope;
  const auto prev_dl = state.dense_layers.find(layer_index);
  if (prev_dl == state.dense_layers.end() ||
      prev_dl->second.num_experts != num_experts ||
      prev_dl->second.hidden_size != hidden_size ||
      prev_dl->second.inter_size != inter_size ||
      prev_dl->second.fc1_w != fc1_w ||
      prev_dl->second.fc2_w != fc2_w ||
      prev_dl->second.fc1_s != fc1_s ||
      prev_dl->second.fc2_s != fc2_s ||
      prev_dl->second.fc1_g != fc1_g ||
      prev_dl->second.fc2_g != fc2_g) {
    for (int b = 0; b < 2; ++b) {
      if (state.buffered_layer[b] == layer_index) {
        state.buffered_layer[b] = -1;
        state.buffered_pass[b] = -1;
      }
    }
  }
  state.dense_layers[layer_index] = DenseLayerHostWeights{
      num_experts, hidden_size, inter_size, fc1_w, fc1_s, fc1_g, fc2_w, fc2_s, fc2_g};

  auto lend_tail_slabs_up_to = [&](size_t target_total_bytes) {
    if (state.total_lent_bytes >= target_total_bytes) return;
    std::vector<const QMoEOffload*> owners_to_lend;
    std::vector<int64_t> layers_to_lend;
    {
      std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
      auto& reg = QMoEOffload::SharedLayerCache::Registry();
      for (auto it = reg.rbegin(); it != reg.rend(); ++it) {
        if (it->first.first != device_id) continue;
        auto locked = it->second.lock();
        if (locked && locked->owner && locked->slab_ptr != nullptr &&
            (req_session_scope == nullptr || locked->session_scope == req_session_scope)) {
          owners_to_lend.push_back(locked->owner);
          layers_to_lend.push_back(it->first.second);
        }
      }
    }
    for (size_t idx = 0; idx < owners_to_lend.size() && state.total_lent_bytes < target_total_bytes; ++idx) {
      state.total_lent_bytes += owners_to_lend[idx]->LendResidentSlab();
      state.lent_layers.push_back(layers_to_lend[idx]);
    }
  };

  if (!state.dma_stream) {
    int least_prio = 0, greatest_prio = 0;
    if (cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess) {
      CUDA_CALL_THROW(cudaStreamCreateWithPriority(&state.dma_stream, cudaStreamNonBlocking, greatest_prio));
    } else {
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&state.dma_stream, cudaStreamNonBlocking));
    }
    for (int b = 0; b < 2; ++b) {
      CUDA_CALL_THROW(cudaEventCreateWithFlags(&state.dma_done_event[b], cudaEventDisableTiming));
      CUDA_CALL_THROW(cudaEventCreateWithFlags(&state.compute_done_event[b], cudaEventDisableTiming));
    }
  }
  int required_hist_stride = std::max(1, num_experts);
  int required_hist_layers = std::max<int>(state.max_hist_layers, static_cast<int>(layer_index) + 1);
  for (const auto& [l_idx, dl] : state.dense_layers) {
    required_hist_stride = std::max(required_hist_stride, dl.num_experts);
    if (l_idx >= 0) {
      required_hist_layers = std::max(required_hist_layers, static_cast<int>(l_idx) + 1);
    }
  }
  if (state.prefill_active && state.hist_dev != nullptr) {
    required_hist_stride = std::max(required_hist_stride, state.hist_expert_stride);
    required_hist_layers = std::max(required_hist_layers, state.hist_layer_cap);
  }
  const size_t required_hist_bytes =
      static_cast<size_t>(required_hist_layers) * static_cast<size_t>(required_hist_stride) * sizeof(int);
  if (!state.hist_dev || state.hist_bytes_cap < required_hist_bytes ||
      state.hist_expert_stride != required_hist_stride || state.hist_layer_cap < required_hist_layers) {
    auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id);
    CudaDeviceMemoryBudget::Reservation res;
    if (dev_budget) res = dev_budget->Reserve(required_hist_bytes);
    void* hptr = nullptr;
    CUDA_CALL_THROW(cudaMalloc(&hptr, required_hist_bytes));
    if (res.Active()) res.Commit();
    CUDA_CALL_THROW(cudaMemsetAsync(hptr, 0, required_hist_bytes, exec_stream));
    if (state.prefill_active && state.hist_dev != nullptr &&
        state.hist_layer_cap > 0 && state.hist_expert_stride > 0) {
      const size_t copy_width =
          static_cast<size_t>(std::min(state.hist_expert_stride, required_hist_stride)) * sizeof(int);
      const size_t copy_rows = static_cast<size_t>(std::min(state.hist_layer_cap, required_hist_layers));
      CUDA_CALL_THROW(cudaMemcpy2DAsync(
          hptr, static_cast<size_t>(required_hist_stride) * sizeof(int),
          state.hist_dev, static_cast<size_t>(state.hist_expert_stride) * sizeof(int),
          copy_width, copy_rows, cudaMemcpyDeviceToDevice, exec_stream));
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(exec_stream));
    state.FreeHistDevLocked(device_id);
    state.hist_dev = static_cast<int*>(hptr);
    state.hist_bytes_cap = required_hist_bytes;
    state.hist_expert_stride = required_hist_stride;
    state.hist_layer_cap = required_hist_layers;
  }

  if (layer_index == 0) {
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SUPPRESS_PREFILL_SPIN", 0) == 1) {
      qmoe::CpuBudgetGroup::Default().EnterPrefillOrAuxPhase(g_shared_spin_pool_ptr.load(std::memory_order_acquire));
    }
    const bool new_turn = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_NEW_TURN", 0) == 1;
    if (new_turn || !state.prefill_active) {
#if defined(_WIN32)
      _putenv_s("ORT_QMOE_PREFILL_NEW_TURN", "0");
#else
      setenv("ORT_QMOE_PREFILL_NEW_TURN", "0", 1);
#endif
      state.total_prefill_tokens_accum = 0;
      state.buffered_layer[0] = state.buffered_layer[1] = -1;
      state.buffered_pass[0] = state.buffered_pass[1] = -1;
      state.dma_done_recorded[0] = state.dma_done_recorded[1] = false;
      state.compute_done_recorded[0] = state.compute_done_recorded[1] = false;
    }
    const int64_t next_pos = state.total_prefill_tokens_accum + tile_rows;
    const int total_prompt_tokens =
        std::max<int>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_TOTAL_TOKENS", static_cast<int>(next_pos)),
                      static_cast<int>(next_pos));
    const int lend_threshold = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_LEND_THRESHOLD",
#if defined(_WIN32)
                                                                        1024
#else
                                                                        2048
#endif
    );
    const bool allow_proactive_lend =
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_PROACTIVE_LEND",
#if defined(_WIN32)
                                                 0
#else
                                                 1
#endif
                                                 ) == 1;
    int max_cfg_ext_slots = 0;
    if (allow_proactive_lend &&
        (next_pos > lend_threshold || total_prompt_tokens > lend_threshold || tile_rows > lend_threshold)) {
      // Reclaim decode extension slots across all layers before allocating prefill DMA workspace (> 2k prefill).
      std::vector<const QMoEOffload*> ext_owners;
      {
        std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
        auto& reg = QMoEOffload::SharedLayerCache::Registry();
        for (auto& [key, weak_cache] : reg) {
          if (key.first != device_id) continue;
          auto locked = weak_cache.lock();
          if (locked && locked->owner) ext_owners.push_back(locked->owner);
        }
      }
      for (const QMoEOffload* owner : ext_owners) {
        static_cast<void>(owner->ReclaimDecodeExtension());
      }
    } else {
      std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
      auto& reg = QMoEOffload::SharedLayerCache::Registry();
      for (auto& [key, weak_cache] : reg) {
        if (key.first != device_id) continue;
        auto locked = weak_cache.lock();
        if (locked && locked->owner) {
          max_cfg_ext_slots = std::max(max_cfg_ext_slots, locked->owner->configured_decode_extension_slots_);
        }
      }
    }
    if (allow_proactive_lend) {
      if (next_pos > 16384 || total_prompt_tokens > 16384) {
        const size_t high_lend_bytes =
            static_cast<size_t>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_LEND_HIGH_MIB", 7200)) << 20;
        lend_tail_slabs_up_to(high_lend_bytes);
      } else if (next_pos > 8192 || total_prompt_tokens > 8192) {
        const size_t mid_lend_bytes =
            static_cast<size_t>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_LEND_MIB", 5400)) << 20;
        lend_tail_slabs_up_to(mid_lend_bytes);
      } else if (total_prompt_tokens > lend_threshold || tile_rows > lend_threshold) {
        const size_t low_lend_bytes =
            static_cast<size_t>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_LEND_LOW_MIB", 4400)) << 20;
        lend_tail_slabs_up_to(low_lend_bytes);
      } else if (max_cfg_ext_slots > 4 && tile_rows >= 256) {
        // When a large decode extension (> 4 slots/layer, e.g. 24 slots/layer = +1,152 experts = +3.0 GiB) is configured,
        // temporarily lend the excess extension bytes during prefill and restore after prefill scratch is released.
        const size_t ext_lend_bytes =
            static_cast<size_t>(max_cfg_ext_slots - 4) * 48ULL * 2764808ULL;
        lend_tail_slabs_up_to(ext_lend_bytes);
      } else if (has_uncached_cold_experts) {
        const size_t min_ws_lend_bytes =
            static_cast<size_t>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_LEND_WS_MIB", 1024)) << 20;
        lend_tail_slabs_up_to(min_ws_lend_bytes);
      }
    }
    if (state.total_prefill_tokens_accum == 0 && state.hist_dev != nullptr && state.hist_bytes_cap > 0) {
      CUDA_CALL_THROW(cudaMemsetAsync(state.hist_dev, 0, state.hist_bytes_cap, exec_stream));
    }
    state.total_prefill_tokens_accum += tile_rows;
    state.prefill_active = true;
    g_prefill_dma_active.store(true, std::memory_order_release);
  }

  out_view->prefill_layer_hist_dev =
      (state.hist_dev != nullptr && layer_index >= 0 && layer_index < state.hist_layer_cap &&
       num_experts <= state.hist_expert_stride)
          ? (state.hist_dev + static_cast<size_t>(layer_index) * state.hist_expert_stride)
          : nullptr;

  const int enable_staging_dma = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ENABLE_PREFILL_STAGING_DMA", 1);
  if (enable_staging_dma == 0 ||
      (!has_uncached_cold_experts && tile_rows <= 2048 && state.workspace_dev == nullptr && state.total_lent_bytes == 0)) {
    return false;
  }

  const size_t fc1_n = static_cast<size_t>(inter_size) * 2;
  const size_t fc1_k = static_cast<size_t>(hidden_size);
  const size_t fc2_n = static_cast<size_t>(hidden_size);
  const size_t fc2_k = static_cast<size_t>(inter_size);

  const bool use_arena_dequant = (tile_rows <= 2048 && state.total_lent_bytes == 0);
  size_t total_ws_bytes = 0;
  const size_t deq_fc1_off = use_arena_dequant
                                 ? 0
                                 : AddSlab(static_cast<size_t>(runner_num_experts) * fc1_n * fc1_k * elem_size, total_ws_bytes);
  const size_t deq_fc2_off = use_arena_dequant
                                 ? 0
                                 : AddSlab(static_cast<size_t>(runner_num_experts) * fc2_n * fc2_k * elem_size, total_ws_bytes);
  size_t stg_fc1_w_off[2]{}, stg_fc2_w_off[2]{}, stg_fc1_s_off[2]{}, stg_fc2_s_off[2]{}, stg_fc1_g_off[2]{}, stg_fc2_g_off[2]{};
  for (int b = 0; b < 2; ++b) {
    stg_fc1_w_off[b] = AddSlab(static_cast<size_t>(runner_num_experts) * fc1_n * (fc1_k / 2), total_ws_bytes);
    stg_fc2_w_off[b] = AddSlab(static_cast<size_t>(runner_num_experts) * fc2_n * (fc2_k / 2), total_ws_bytes);
    stg_fc1_s_off[b] = AddSlab(static_cast<size_t>(runner_num_experts) * fc1_n * (fc1_k / 16), total_ws_bytes);
    stg_fc2_s_off[b] = AddSlab(static_cast<size_t>(runner_num_experts) * fc2_n * (fc2_k / 16), total_ws_bytes);
    stg_fc1_g_off[b] = AddSlab(static_cast<size_t>(num_experts) * sizeof(float), total_ws_bytes);
    stg_fc2_g_off[b] = AddSlab(static_cast<size_t>(num_experts) * sizeof(float) + 512, total_ws_bytes);
  }
  const size_t hist_off = AddSlab(static_cast<size_t>(state.max_hist_layers) * num_experts * sizeof(int), total_ws_bytes);
  state.current_tile_rows = tile_rows;

  if (!state.workspace_dev || state.workspace_bytes < total_ws_bytes) {
    auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id);
    if (state.workspace_dev) {
      if (dev_budget && state.workspace_bytes > 0) dev_budget->BeginFree(state.workspace_bytes);
      const bool ok = (cudaFree(state.workspace_dev) == cudaSuccess);
      if (dev_budget && state.workspace_bytes > 0) dev_budget->FinishFree(state.workspace_bytes, ok);
      state.workspace_dev = nullptr;
      state.workspace_bytes = 0;
    }
    CudaDeviceMemoryBudget::Reservation ws_res;
    try {
      if (dev_budget) ws_res = dev_budget->Reserve(total_ws_bytes);
    } catch (const std::exception&) {
      lend_tail_slabs_up_to(state.total_lent_bytes + total_ws_bytes + (size_t{512} << 20));
      try {
        if (dev_budget) ws_res = dev_budget->Reserve(total_ws_bytes);
      } catch (const std::exception&) {
        return false;
      }
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, total_ws_bytes);
    if (err != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      // Lend additional tail layers on demand and retry.
      lend_tail_slabs_up_to(state.total_lent_bytes + total_ws_bytes + (size_t{512} << 20));
      if (!ws_res.Active() && dev_budget) {
        try {
          ws_res = dev_budget->Reserve(total_ws_bytes);
        } catch (const std::exception&) {
          return false;
        }
      }
      err = cudaMalloc(&ptr, total_ws_bytes);
      if (err != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
      }
    }
    if (ws_res.Active()) ws_res.Commit();
    state.workspace_dev = static_cast<uint8_t*>(ptr);
    state.workspace_bytes = total_ws_bytes;
    state.buffered_layer[0] = state.buffered_layer[1] = -1;
    state.buffered_pass[0] = state.buffered_pass[1] = -1;
  }
  // Every offset depends on the expert count (global-scale and histogram slabs), so compare the complete layout.
  bool layout_changed = state.workspace_runner_experts != runner_num_experts ||
                        state.workspace_num_experts != num_experts ||
                        state.workspace_hidden != hidden_size ||
                        state.workspace_inter != inter_size ||
                        state.dequant_fc1_offset != deq_fc1_off ||
                        state.dequant_fc2_offset != deq_fc2_off ||
                        state.hist_offset != hist_off;
  for (int b = 0; b < 2 && !layout_changed; ++b) {
    layout_changed = state.staging_fc1_w_offset[b] != stg_fc1_w_off[b] ||
                     state.staging_fc2_w_offset[b] != stg_fc2_w_off[b] ||
                     state.staging_fc1_s_offset[b] != stg_fc1_s_off[b] ||
                     state.staging_fc2_s_offset[b] != stg_fc2_s_off[b] ||
                     state.staging_fc1_g_offset[b] != stg_fc1_g_off[b] ||
                     state.staging_fc2_g_offset[b] != stg_fc2_g_off[b];
  }
  if (layout_changed) {
    CUDA_CALL_THROW(cudaStreamSynchronize(exec_stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(state.dma_stream));
    state.workspace_runner_experts = runner_num_experts;
    state.workspace_num_experts = num_experts;
    state.workspace_hidden = hidden_size;
    state.workspace_inter = inter_size;
    state.dequant_fc1_offset = deq_fc1_off;
    state.dequant_fc2_offset = deq_fc2_off;
    for (int b = 0; b < 2; ++b) {
      state.staging_fc1_w_offset[b] = stg_fc1_w_off[b];
      state.staging_fc2_w_offset[b] = stg_fc2_w_off[b];
      state.staging_fc1_s_offset[b] = stg_fc1_s_off[b];
      state.staging_fc2_s_offset[b] = stg_fc2_s_off[b];
      state.staging_fc1_g_offset[b] = stg_fc1_g_off[b];
      state.staging_fc2_g_offset[b] = stg_fc2_g_off[b];
    }
    state.hist_offset = hist_off;
    state.buffered_layer[0] = state.buffered_layer[1] = -1;
    state.buffered_pass[0] = state.buffered_pass[1] = -1;
  }
#if defined(_WIN32)
  const bool use_host_staging_dma =
      (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_WIN_HOST_STAGING_DMA", 1) != 0) ||
      has_uncached_cold_experts || state.total_lent_bytes > 0;
#else
  const bool use_host_staging_dma = has_uncached_cold_experts || state.total_lent_bytes > 0;
#endif
  if (use_host_staging_dma) {
    const size_t hdr_bytes =
        ((static_cast<size_t>(runner_num_experts) * sizeof(int32_t) + 255) & ~size_t{255}) +
        2 * ((static_cast<size_t>(num_experts) * sizeof(float) + 255) & ~size_t{255});
    const size_t host_pass_bytes =
        hdr_bytes +
        static_cast<size_t>(runner_num_experts) *
            (fc1_n * (fc1_k / 2) + fc2_n * (fc2_k / 2) + fc1_n * (fc1_k / 16) + fc2_n * (fc2_k / 16));
    const size_t total_staging_bytes = SafeInt<size_t>(2) * host_pass_bytes;
    if (!state.host_staging[0] || !state.host_staging[1] || state.host_staging_bytes < host_pass_bytes) {
      if (state.dma_stream) {
        static_cast<void>(cudaStreamSynchronize(state.dma_stream));
      }
      state.FreeHostStagingLocked();
      const bool explicit_pinned_cap =
          (matched_owner != nullptr && matched_owner->v4_planned_mode_ && matched_owner->v4_policy_.pinned_bytes_explicit);
      if (matched_owner != nullptr && total_staging_bytes > matched_owner->shared_pinned_limit_) {
        if (explicit_pinned_cap) {
          ORT_THROW("QMoEOffload prefill host staging (2 * ", host_pass_bytes, " = ", total_staging_bytes,
                    " bytes) exceeds configured session.moe.offload.pinned_bytes limit (",
                    matched_owner->shared_pinned_limit_, " bytes).");
        }
        return false;
      }
      auto sess_pinned = (matched_owner && matched_owner->session_controller_)
                             ? matched_owner->session_controller_->PinnedBudget()
                             : nullptr;
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      CudaDeviceMemoryBudget::Reservation sess_res;
      CudaDeviceMemoryBudget::Reservation host_res;
      auto try_reserve_staging = [&]() -> bool {
        try {
          if (sess_pinned) sess_res = sess_pinned->Reserve(total_staging_bytes);
          if (host_ledger) host_res = host_ledger->ReserveAllocated(total_staging_bytes);
          return true;
        } catch (...) {
          sess_res = CudaDeviceMemoryBudget::Reservation{};
          host_res = CudaDeviceMemoryBudget::Reservation{};
          return false;
        }
      };
      if (!try_reserve_staging()) {
        // Reclaim any idle persistent decode pinned buffers in this session before failing prefill staging reservation.
        std::vector<const QMoEOffload*> session_owners;
        {
          std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
          auto& mreg = QMoEOffload::SharedLayerCache::MultiRegistry();
          for (auto& [key, vec] : mreg) {
            if (key.first != device_id) continue;
            for (const auto& weak_cache : vec) {
              auto locked = weak_cache.lock();
              if (locked && locked->owner &&
                  (req_session_scope == nullptr || locked->session_scope == req_session_scope)) {
                session_owners.push_back(locked->owner);
              }
            }
          }
        }
        for (const QMoEOffload* owner : session_owners) {
          if (owner) {
            owner->ReleasePersistentDecodePinned();
          }
        }
        if (!try_reserve_staging()) {
          if (explicit_pinned_cap) {
            ORT_THROW("QMoEOffload prefill host staging (2 * ", host_pass_bytes, " = ", total_staging_bytes,
                      " bytes) exceeds remaining session.moe.offload.pinned_bytes budget.");
          }
          return false;
        }
      }
      for (int b = 0; b < 2; ++b) {
        void* hptr = nullptr;
        if (cudaHostAlloc(&hptr, host_pass_bytes, cudaHostAllocDefault) == cudaSuccess) {
          state.host_staging[b] = static_cast<uint8_t*>(hptr);
        }
      }
      if (state.host_staging[0] && state.host_staging[1]) {
        if (sess_res.Active()) sess_res.Commit();
        if (host_res.Active()) host_res.Commit();
        state.host_staging_bytes = host_pass_bytes;
        state.host_staging_accounted_bytes = total_staging_bytes;
        state.host_staging_session_budget = std::move(sess_pinned);
      } else {
        for (int b = 0; b < 2; ++b) {
          if (state.host_staging[b]) {
            static_cast<void>(cudaFreeHost(state.host_staging[b]));
            state.host_staging[b] = nullptr;
          }
        }
        state.host_staging_bytes = 0;
        if (explicit_pinned_cap) {
          ORT_THROW("QMoEOffload failed to allocate prefill host staging buffers (", total_staging_bytes, " bytes).");
        }
        return false;
      }
      state.dma_done_recorded[0] = state.dma_done_recorded[1] = false;
    }
  }

  QMoEResidentLayerSlabView res_view{};
  static_cast<void>(TryGetQMoEResidentLayerSlabView(
      device_id, layer_index, num_experts, hidden_size, inter_size, &res_view, exec_stream, fc1_w));
  state.use_uva_in_dma_pass = (res_view.host_fc1_weights != nullptr);

  out_view->dequant_fc1_weights = use_arena_dequant ? nullptr : (state.workspace_dev + state.dequant_fc1_offset);
  out_view->dequant_fc2_weights = use_arena_dequant ? nullptr : (state.workspace_dev + state.dequant_fc2_offset);
  out_view->resident_fc1_weights = res_view.fc1_weights;
  out_view->resident_fc1_scales = res_view.fc1_scales;
  out_view->resident_fc2_weights = res_view.fc2_weights;
  out_view->resident_fc2_scales = res_view.fc2_scales;
  out_view->slot_table_dev = res_view.slot_table_dev;
  out_view->host_fc1_weights = res_view.host_fc1_weights;
  out_view->host_fc1_scales = res_view.host_fc1_scales;
  out_view->host_fc2_weights = res_view.host_fc2_weights;
  out_view->host_fc2_scales = res_view.host_fc2_scales;
  out_view->host_expert_stride = res_view.host_expert_stride;
  out_view->host_slot_table_dev = res_view.host_slot_table_dev;
  return true;
}

bool BeginQMoEPrefillDmaPass(
    int device_id, int64_t layer_index, int pass_idx, int /*num_passes*/, int runner_num_experts,
    cudaStream_t exec_stream, QMoEPrefillPassDmaView* out_view,
    const int* expert_active_mask_dev) {
  if (!out_view) return false;
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.workspace_dev || !state.dma_stream) return false;

  if (pass_idx == 0) {
    auto it = state.dense_layers.find(layer_index);
    const int ne = (it != state.dense_layers.end()) ? it->second.num_experts : 0;
    if (expert_active_mask_dev != nullptr && state.host_staging[0] != nullptr && ne > 0) {
      if (!state.host_active_mask || state.host_active_mask_cap < ne) {
        state.FreeHostActiveMaskLocked();
        const size_t mask_bytes = static_cast<size_t>(ne) * sizeof(int);
        auto sess_pinned = state.host_staging_session_budget;
        auto host_ledger = GetCudaHostPinnedMemoryBudget();
        CudaDeviceMemoryBudget::Reservation sess_res;
        CudaDeviceMemoryBudget::Reservation host_res;
        bool budget_ok = true;
        try {
          if (sess_pinned) sess_res = sess_pinned->Reserve(mask_bytes);
          if (host_ledger) host_res = host_ledger->ReserveAllocated(mask_bytes);
        } catch (...) {
          budget_ok = false;
        }
        void* mptr = nullptr;
        if (budget_ok && cudaHostAlloc(&mptr, mask_bytes, cudaHostAllocPortable) == cudaSuccess) {
          if (sess_res.Active()) sess_res.Commit();
          if (host_res.Active()) host_res.Commit();
          state.host_active_mask = static_cast<int*>(mptr);
          state.host_active_mask_cap = ne;
          state.host_active_mask_accounted_bytes = mask_bytes;
          state.host_active_mask_session_budget = std::move(sess_pinned);
        } else {
          static_cast<void>(cudaGetLastError());
        }
      }
      if (state.host_active_mask && state.host_active_mask_cap >= ne) {
        CUDA_CALL_THROW(cudaMemcpyAsync(state.host_active_mask, expert_active_mask_dev,
                                        static_cast<size_t>(ne) * sizeof(int),
                                        cudaMemcpyDeviceToHost, exec_stream));
        CUDA_CALL_THROW(cudaStreamSynchronize(exec_stream));
        state.active_mask_layer = layer_index;
      } else {
        state.active_mask_layer = -1;
      }
    } else {
      state.active_mask_layer = -1;
    }
  }

  const int cur_buf = pass_idx & 1;
  if (!EnqueuePrefillPassDmaLocked(state, device_id, layer_index, pass_idx, runner_num_experts, cur_buf)) {
    return false;
  }

  CUDA_CALL_THROW(cudaStreamWaitEvent(exec_stream, state.dma_done_event[cur_buf], 0));
  if (state.buffered_compact_pack[cur_buf]) {
    const auto& wl = state.dense_layers.at(layer_index);
    const size_t fc1_n = static_cast<size_t>(wl.inter_size) * 2;
    const size_t fc1_k = static_cast<size_t>(wl.hidden_size);
    const size_t fc2_n = static_cast<size_t>(wl.hidden_size);
    const size_t fc2_k = static_cast<size_t>(wl.inter_size);
    const size_t fc1_w_stride = fc1_n * (fc1_k / 2);
    const size_t fc1_s_stride = fc1_n * (fc1_k / 16);
    const size_t fc2_s_stride = fc2_n * (fc2_k / 16);
    const size_t map_bytes = (static_cast<size_t>(runner_num_experts) * sizeof(int32_t) + 255) & ~size_t{255};
    const size_t g_bytes = (static_cast<size_t>(wl.num_experts) * sizeof(float) + 255) & ~size_t{255};
    const size_t hdr_bytes = map_bytes + 2 * g_bytes;
    const size_t M = static_cast<size_t>(state.buffered_cold_count[cur_buf]);
    const uint8_t* base_dev = state.workspace_dev + state.staging_fc1_w_offset[cur_buf];

    if (!state.use_uva_in_dma_pass) {
      out_view->host_fc1_weights = nullptr;
      out_view->host_fc1_scales = nullptr;
      out_view->host_fc2_weights = nullptr;
      out_view->host_fc2_scales = nullptr;
      out_view->host_expert_stride = 0;
    }
    out_view->host_slot_table_dev = reinterpret_cast<const int*>(base_dev);
    out_view->staging_fc1_global_scales = reinterpret_cast<const float*>(base_dev + map_bytes);
    out_view->staging_fc2_global_scales = reinterpret_cast<const float*>(base_dev + map_bytes + g_bytes);
    out_view->staging_fc1_scales = base_dev + hdr_bytes;
    out_view->staging_fc2_scales = out_view->staging_fc1_scales + M * fc1_s_stride;
    out_view->staging_fc1_weights = out_view->staging_fc2_scales + M * fc2_s_stride;
    out_view->staging_fc2_weights = out_view->staging_fc1_weights + M * fc1_w_stride;
  } else {
    out_view->staging_fc1_weights = state.workspace_dev + state.staging_fc1_w_offset[cur_buf];
    out_view->staging_fc2_weights = state.workspace_dev + state.staging_fc2_w_offset[cur_buf];
    out_view->staging_fc1_scales = state.workspace_dev + state.staging_fc1_s_offset[cur_buf];
    out_view->staging_fc2_scales = state.workspace_dev + state.staging_fc2_s_offset[cur_buf];
    out_view->staging_fc1_global_scales =
        reinterpret_cast<const float*>(state.workspace_dev + state.staging_fc1_g_offset[cur_buf]);
    out_view->staging_fc2_global_scales =
        reinterpret_cast<const float*>(state.workspace_dev + state.staging_fc2_g_offset[cur_buf]);
  }
  return true;
}

void PrefetchNextQMoEPrefillDmaPass(
    int device_id, int64_t layer_index, int pass_idx, int num_passes, int runner_num_experts,
    cudaStream_t exec_stream) {
  if (exec_stream) {
    static_cast<void>(cudaStreamQuery(exec_stream));
  }
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.workspace_dev || !state.dma_stream) return;
  const int cur_buf = pass_idx & 1;
  int64_t next_layer = -1;
  int next_pass = -1;
  if (pass_idx + 1 < num_passes) {
    next_layer = layer_index;
    next_pass = pass_idx + 1;
  } else if ((num_passes & 1) == 0 && (state.active_mask_layer != layer_index || state.current_tile_rows >= 256)) {
    next_layer = layer_index + 1;
    next_pass = 0;
  }
  if (next_layer >= 0 && state.dense_layers.count(next_layer)) {
    static_cast<void>(EnqueuePrefillPassDmaLocked(
        state, device_id, next_layer, next_pass, runner_num_experts, 1 - cur_buf));
  }
}

void EndQMoEPrefillDmaPassDequant(int device_id, int pass_idx, cudaStream_t exec_stream) {
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.dma_stream) return;
  const int cur_buf = pass_idx & 1;
  CUDA_CALL_THROW(cudaEventRecord(state.compute_done_event[cur_buf], exec_stream));
  static_cast<void>(cudaStreamQuery(exec_stream));
  state.compute_done_recorded[cur_buf] = true;
  state.buffered_layer[cur_buf] = -1;
  state.buffered_pass[cur_buf] = -1;
}

void FinalizeQMoEPrefillAndSeedDecode(int device_id, cudaStream_t stream) {
  auto& state = DevicePrefillDmaState::ForDevice(device_id);
  std::unique_lock<std::mutex> lock(state.mutex);
  const bool free_prefill_ws =
      !state.lent_layers.empty() ||
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_FREE_PREFILL_WORKSPACE", 0) == 1;
  if (!state.prefill_active && state.lent_layers.empty() && (!free_prefill_ws || state.workspace_dev == nullptr)) {
    g_prefill_dma_active.store(false, std::memory_order_release);
  }

  const int64_t total_tokens = state.total_prefill_tokens_accum;
  const bool in_warmup = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_IN_WARMUP", 0) == 1;
  const bool handoff_done = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_HANDOFF_DONE", 0) == 1;
  if (handoff_done) {
#if defined(_WIN32)
    _putenv_s("ORT_QMOE_PREFILL_HANDOFF_DONE", "0");
#else
    setenv("ORT_QMOE_PREFILL_HANDOFF_DONE", "0", 1);
#endif
  }
  const bool aborted_previous_prefill =
      !handoff_done && (stream != nullptr) && (state.prefill_active || !state.lent_layers.empty());
  bool is_new_turn_reset = aborted_previous_prefill && !in_warmup;
  if (stream != nullptr && !in_warmup && !handoff_done) {
    const int next_prompt_tokens = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_TOTAL_TOKENS", 0);
#if defined(_WIN32)
    const int default_reset_on_short = 0;
#else
    const int default_reset_on_short = 1;
#endif
    if ((default_reset_on_short == 1 && next_prompt_tokens < 2048) ||
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_RESET_ON_NEW_TURN", 0) == 1) {
      is_new_turn_reset = true;
    }
    if (next_prompt_tokens > 0) {
      state.last_new_turn_prompt_tokens = next_prompt_tokens;
    }
  }
#if defined(_WIN32)
  const int default_seed_promos = 2;
#else
  const int default_seed_promos = 8;
#endif
  const bool seed_enabled =
      !in_warmup && !is_new_turn_reset && !aborted_previous_prefill && (total_tokens > 0) &&
      (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", default_seed_promos) > 0) &&
      (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_WARM_SEED", 1) == 1);

  if (!seed_enabled && (!is_new_turn_reset || in_warmup) &&
      state.lent_layers.empty() && (!free_prefill_ws || state.workspace_dev == nullptr)) {
    if (state.dma_stream) {
      static_cast<void>(cudaStreamSynchronize(state.dma_stream));
    }
    qmoe::CpuBudgetGroup::Default().EnterDecodePhase(g_shared_spin_pool_ptr.load(std::memory_order_acquire));
    state.dma_done_recorded[0] = state.dma_done_recorded[1] = false;
    state.compute_done_recorded[0] = state.compute_done_recorded[1] = false;
    state.buffered_layer[0] = state.buffered_layer[1] = -1;
    state.buffered_pass[0] = state.buffered_pass[1] = -1;
    state.active_mask_layer = -1;
    state.prefill_active = false;
    state.total_prefill_tokens_accum = 0;
    g_prefill_dma_active.store(false, std::memory_order_release);
    return;
  }

  if (state.dma_stream) {
    static_cast<void>(cudaStreamSynchronize(state.dma_stream));
  }
  if (stream) {
    static_cast<void>(cudaStreamSynchronize(stream));
  } else if (state.workspace_dev != nullptr || !state.lent_layers.empty()) {
    static_cast<void>(cudaDeviceSynchronize());
  }

  const int hist_stride = state.hist_expert_stride;
  const int hist_layers = state.hist_layer_cap;
  std::vector<int> host_hist;
  if (seed_enabled && state.hist_dev && hist_stride > 0 && hist_layers > 0 && total_tokens > 0 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_HIST_SEED", 1) == 1) {
    host_hist.resize(static_cast<size_t>(hist_layers) * static_cast<size_t>(hist_stride), 0);
    static_cast<void>(cudaMemcpy(host_hist.data(), state.hist_dev,
                                 host_hist.size() * sizeof(int), cudaMemcpyDeviceToHost));
  }

  // Free the prefill DMA workspace BEFORE restoring lent resident_slab_ allocations so peak HBM never overlaps!
  if (state.workspace_dev && free_prefill_ws) {
    auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(device_id);
    if (dev_budget && state.workspace_bytes > 0) dev_budget->BeginFree(state.workspace_bytes);
    const bool ok = (cudaFree(state.workspace_dev) == cudaSuccess);
    if (dev_budget && state.workspace_bytes > 0) dev_budget->FinishFree(state.workspace_bytes, ok);
    state.workspace_dev = nullptr;
    state.workspace_bytes = 0;
    state.workspace_runner_experts = 0;
  }
#if defined(__linux__)
  bool any_bounded_host_cache = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_MIB", 0) > 0;
  if (!any_bounded_host_cache) {
    std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
    auto& reg = QMoEOffload::SharedLayerCache::Registry();
    for (auto& [key, weak_cache] : reg) {
      if (key.first != device_id) continue;
      auto locked = weak_cache.lock();
      if (locked && locked->owner && locked->owner->host_cache_bytes_ > 0) {
        any_bounded_host_cache = true;
        break;
      }
    }
  }
#else
  const bool any_bounded_host_cache = false;
#endif
  if (any_bounded_host_cache || state.host_staging_session_budget != nullptr ||
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_FREE_HOST_STAGING", 0) == 1) {
    state.FreeHostStagingLocked();
    state.FreeHostActiveMaskLocked();
  } else {
    state.dma_done_recorded[0] = state.dma_done_recorded[1] = false;
  }
#if defined(__linux__)
  if (any_bounded_host_cache && state.mmap_touched_layer_mask != 0) {
    auto unmap_range = [](const uint8_t* ptr, size_t bytes) {
      if (!ptr || bytes == 0) return;
      static_cast<void>(qmoe_shared::EvictFileBackedHostPages(ptr, bytes));
    };
    for (const auto& [l_idx, wl] : state.dense_layers) {
      if (l_idx < 0 || l_idx >= 64 || ((state.mmap_touched_layer_mask >> l_idx) & 1u) == 0) continue;
      const size_t fc1_n = static_cast<size_t>(wl.inter_size) * 2;
      const size_t fc1_k = static_cast<size_t>(wl.hidden_size);
      const size_t fc2_n = static_cast<size_t>(wl.hidden_size);
      const size_t fc2_k = static_cast<size_t>(wl.inter_size);
      const size_t ne = static_cast<size_t>(wl.num_experts);
      unmap_range(wl.fc1_w, ne * fc1_n * (fc1_k / 2));
      unmap_range(wl.fc2_w, ne * fc2_n * (fc2_k / 2));
      unmap_range(wl.fc1_s, ne * fc1_n * (fc1_k / 16));
      unmap_range(wl.fc2_s, ne * fc2_n * (fc2_k / 16));
    }
    state.mmap_touched_layer_mask = 0;
  }
#endif
  qmoe::CpuBudgetGroup::Default().EnterDecodePhase(g_shared_spin_pool_ptr.load(std::memory_order_acquire));
  cudaStream_t restore_stream = stream ? stream : state.dma_stream;
  const void* target_scope = state.active_session_scope;
  state.buffered_layer[0] = state.buffered_layer[1] = -1;
  state.buffered_pass[0] = state.buffered_pass[1] = -1;
  state.compute_done_recorded[0] = state.compute_done_recorded[1] = false;
  state.lent_layers.clear();
  state.total_lent_bytes = 0;
  state.prefill_active = false;
  state.total_prefill_tokens_accum = 0;
  g_prefill_dma_active.store(false, std::memory_order_release);
  lock.unlock();

  std::vector<std::pair<int64_t, const QMoEOffload*>> layers;
  {
    std::lock_guard<std::mutex> cache_lock(QMoEOffload::SharedLayerCache::Mutex());
    auto& mreg = QMoEOffload::SharedLayerCache::MultiRegistry();
    for (auto& [key, vec] : mreg) {
      if (key.first != device_id) continue;
      for (const auto& weak_cache : vec) {
        auto locked = weak_cache.lock();
        if (locked && locked->owner &&
            (target_scope == nullptr || locked->session_scope == target_scope)) {
          layers.push_back({key.second, locked->owner});
        }
      }
    }
  }

  const int total_routes = static_cast<int>(total_tokens) * 10;
  for (const auto& [l_idx, owner] : layers) {
    if (total_tokens <= 8192) {
      owner->RestoreDecodeExtensionConfig();
    }
    const int* l_hist = (!host_hist.empty() && l_idx >= 0 && l_idx < hist_layers &&
                         owner->num_experts_ <= hist_stride)
                            ? (host_hist.data() + static_cast<size_t>(l_idx) * static_cast<size_t>(hist_stride))
                            : nullptr;
    owner->RestoreAndSeedResidentSlab(l_hist, total_routes, seed_enabled, restore_stream);
  }
  if (restore_stream) {
    static_cast<void>(cudaStreamSynchronize(restore_stream));
  }
}

void OnQMoEDeviceArenasShrunk(int device_id) noexcept {
  try {
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_HANDOFF_DONE", 0) == 1) {
#if defined(_WIN32)
      _putenv_s("ORT_QMOE_PREFILL_HANDOFF_DONE", "0");
#else
      setenv("ORT_QMOE_PREFILL_HANDOFF_DONE", "0", 1);
#endif
      FinalizeQMoEPrefillAndSeedDecode(device_id, nullptr);
    }
  } catch (...) {
  }
}

struct QMoEOffload::AsyncTile {
  const QMoEOffload* node = nullptr;
  std::shared_ptr<bool> completed;
  std::shared_ptr<onnxruntime::cuda::QMoEInvocationRecord> invocation;
  int tile_rows = 0;
  const int* host_indices = nullptr;
  const int8_t* host_phases = nullptr;
  bool has_phases = false;
  const void* host_input = nullptr;
  void* host_result = nullptr;
  int32_t* route_rows = nullptr;
  int32_t* route_calls = nullptr;
  int32_t* counts = nullptr;
  IAllocatorUniquePtr<uint8_t> pinned;
  qmoe::CallbackTiming* timing = nullptr;
  qmoe::CallbackTiming::Call trace_call;
  bool persistent = false;
  bool staged_for_run_end = false;
};

struct QMoESharedExpertBridge {
  static constexpr uint64_t kMagic = 0x514d4f4553484558ULL;
  static constexpr int kMaxRegisteredNames = 256;
  static constexpr int kMaxNameLen = 128;
  uint64_t magic = kMagic;
  std::atomic<int> lock{0};
  std::atomic<int> num_registered{0};
  char registered_routed_names[kMaxRegisteredNames][kMaxNameLen]{};
  bool pending = false;
  cudaStream_t stream = nullptr;
  const void* routed_output_ptr = nullptr;
  int tile_rows = 0;
  int hidden_size = 0;
  int (*launch_deferred)(void* ctx, cudaStream_t stream, void* shared_out_ptr, int trigger) = nullptr;
  void* ctx = nullptr;

  bool IsRoutedNameRegistered(const char* name) const noexcept {
    if (!name || !*name) return false;
    const int n = num_registered.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
      if (std::strncmp(registered_routed_names[i], name, kMaxNameLen - 1) == 0) {
        return true;
      }
    }
    return false;
  }
};

inline QMoESharedExpertBridge* GetQMoESharedExpertBridge() noexcept {
  static QMoESharedExpertBridge* const instance = []() -> QMoESharedExpertBridge* {
    constexpr const char* kEnvVar = "ORT_QMOE_SHARED_EXPERT_BRIDGE_PTR";
    if (const char* env = std::getenv(kEnvVar); env && *env) {
      unsigned long long raw_addr = 0;
      if (std::sscanf(env, "%llx", &raw_addr) == 1 && raw_addr != 0) {
        auto* p = reinterpret_cast<QMoESharedExpertBridge*>(static_cast<uintptr_t>(raw_addr));
        if (p->magic == QMoESharedExpertBridge::kMagic) return p;
      }
    }
    auto* p = new QMoESharedExpertBridge();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
#if defined(_WIN32)
    _putenv_s(kEnvVar, buf);
#else
    setenv(kEnvVar, buf, 1);
#endif
    return p;
  }();
  return instance;
}

struct QMoEOffload::PersistentDecodeState {
  ResidentSlab scratch;
  size_t scratch_bytes = 0;
  PinnedBuffer pinned;
  size_t pinned_bytes = 0;
  size_t pinned_accounted_bytes = 0;
  std::shared_ptr<CudaDeviceMemoryBudget> session_pinned_budget;
  uint64_t* doorbell_seq_dev = nullptr;
  volatile uint64_t* doorbell_req_host = nullptr;
  volatile uint64_t* doorbell_done_host = nullptr;
  volatile int32_t* doorbell_tile_rows_host = nullptr;
  int32_t* pinned_slot_table = nullptr;
  uint64_t last_serviced_seq = 0;
  uint64_t launched_seq = 0;
  uint64_t last_promoted_seq = 0;
  std::shared_ptr<bool> completed;
  std::shared_ptr<onnxruntime::cuda::QMoEInvocationRecord> invocation_rec;
  std::unique_ptr<AsyncTile> tile;
  bool registered_with_doorbell = false;
  const __nv_bfloat16* deferred_wave_output = nullptr;
  const __nv_bfloat16* deferred_host_result = nullptr;
  const int* deferred_hit_flag = nullptr;
  __nv_bfloat16* deferred_contributions = nullptr;
  const float* deferred_scales = nullptr;
  const int* deferred_identity = nullptr;
  __nv_bfloat16* deferred_routed_out = nullptr;
  int deferred_routes = 0;
  int deferred_tile_rows = 0;

  void ReleasePinned(bool context_ok = true) noexcept {
    if (pinned) {
      if (context_ok) {
        pinned.reset();
      } else {
        static_cast<void>(pinned.release());
      }
    }
    if (pinned_accounted_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (session_pinned_budget) session_pinned_budget->BeginFree(pinned_accounted_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(pinned_accounted_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(pinned_accounted_bytes, context_ok);
      if (session_pinned_budget) session_pinned_budget->FinishFree(pinned_accounted_bytes, context_ok);
      pinned_accounted_bytes = 0;
    }
    session_pinned_budget.reset();
    pinned_bytes = 0;
    doorbell_req_host = nullptr;
    doorbell_done_host = nullptr;
    doorbell_tile_rows_host = nullptr;
    pinned_slot_table = nullptr;
  }

  ~PersistentDecodeState() {
    ReleasePinned(true);
  }
};

void QMoEOffload::ReleasePersistentDecodePinned() const noexcept {
  if (decode_state_ && !async_in_flight_.load(std::memory_order_acquire)) {
    decode_state_->ReleasePinned(true);
  }
}

class QMoEOffload::DoorbellService {
 public:
  enum PromoState : int {
    kPromoIdle = 0,
    kPromoQueued = 1,
    kPromoEnqueued = 2,
  };

  static DoorbellService& Instance() {
    static DoorbellService instance;
    return instance;
  }

  void RegisterNode(const QMoEOffload* node) {
    if (!node) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
      nodes_.push_back(node);
    }
    if (!worker_started_) {
      worker_started_ = true;
      cross_layer_promo_enabled_ =
          ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CROSS_LAYER_PROMO", 0) == 1;
      if (cross_layer_promo_enabled_ && !promo_bounce_buf_) {
        constexpr size_t kBounceBytes = 3u * 1024u * 1024u;
        void* hptr = nullptr;
        if (cudaHostAlloc(&hptr, kBounceBytes, cudaHostAllocPortable) == cudaSuccess) {
          promo_bounce_buf_ = static_cast<uint8_t*>(hptr);
          promo_bounce_cap_ = kBounceBytes;
        } else {
          static_cast<void>(cudaGetLastError());
        }
      }
      worker_ = std::thread([this] { Loop(); });
      if (cross_layer_promo_enabled_) {
        promo_worker_ = std::thread([this] { PromoLoop(); });
      }
    }
  }

  void CancelPendingPromo(const QMoEOffload* node = nullptr) noexcept {
    while (promo_state_.load(std::memory_order_acquire) == kPromoQueued) {
      std::this_thread::yield();
    }
    if (promo_state_.load(std::memory_order_acquire) == kPromoEnqueued) {
      if (node == nullptr || promo_req_.node == node) {
        if (promo_req_.node && promo_req_.node->promo_stream_) {
          static_cast<void>(cudaStreamSynchronize(promo_req_.node->promo_stream_));
        }
        promo_req_ = CrossLayerPromoRequest{};
        promo_state_.store(kPromoIdle, std::memory_order_release);
      }
    }
  }

  void ResetForNewTurn() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    CancelPendingPromo(nullptr);
    cursor_ = 0;
  }

  void UnregisterNode(const QMoEOffload* node) noexcept {
    if (!node) return;
    std::unique_lock<std::mutex> lock(mutex_);
    CancelPendingPromo(node);
    service_done_cv_.wait(lock, [this, node] { return servicing_node_ != node; });
    nodes_.erase(std::remove(nodes_.begin(), nodes_.end(), node), nodes_.end());
    if (cursor_ >= nodes_.size()) cursor_ = 0;
  }

  void NotifyPending(size_t count = 1) noexcept {
    pending_.fetch_add(count, std::memory_order_release);
    if (sleeping_.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(cv_mutex_);
      cv_.notify_one();
    }
  }

  ~DoorbellService() {
    stop_.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(cv_mutex_);
      cv_.notify_all();
    }
    {
      std::lock_guard<std::mutex> plock(promo_cv_mutex_);
      promo_cv_.notify_all();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
    if (promo_worker_.joinable()) {
      promo_worker_.join();
    }
    if (promo_bounce_buf_) {
      static_cast<void>(cudaFreeHost(promo_bounce_buf_));
      promo_bounce_buf_ = nullptr;
    }
  }

 private:
  void PromoLoop() {
    constexpr size_t kBounceBytes = 3u * 1024u * 1024u;
    while (!stop_.load(std::memory_order_relaxed)) {
      if (promo_state_.load(std::memory_order_acquire) == kPromoQueued) {
        if (!promo_bounce_buf_) {
          void* hptr = nullptr;
          if (cudaHostAlloc(&hptr, kBounceBytes, cudaHostAllocPortable) == cudaSuccess) {
            promo_bounce_buf_ = static_cast<uint8_t*>(hptr);
            promo_bounce_cap_ = kBounceBytes;
          } else {
            static_cast<void>(cudaGetLastError());
          }
        }
        bool ok = false;
        if (promo_bounce_buf_ && promo_req_.node != nullptr) {
          ok = promo_req_.node->ExecuteCrossLayerPromotionDma(
              promo_req_, promo_bounce_buf_, promo_bounce_cap_);
        }
        promo_state_.store(ok ? kPromoEnqueued : kPromoIdle, std::memory_order_release);
        continue;
      }
      std::unique_lock<std::mutex> plock(promo_cv_mutex_);
      promo_cv_.wait(plock, [this] {
        return stop_.load(std::memory_order_relaxed) ||
               promo_state_.load(std::memory_order_acquire) == kPromoQueued;
      });
    }
  }

  void Loop() {
#if defined(_WIN32)
    {
      const int default_pool_threads =
          std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 40);
      const int threads = std::clamp(
          ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", default_pool_threads), 1, 256);
      const int cpu0 = SharedSpinPool(threads).CpuForWorker(0);
      if (cpu0 >= 0 && cpu0 < static_cast<int>(sizeof(DWORD_PTR) * 8)) {
        static_cast<void>(SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu0));
      }
    }
#endif
    auto idle_since = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      const QMoEOffload* ready_node = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = nodes_.size();
        for (size_t step = 0; step < n; ++step) {
          const size_t idx = (cursor_ + step) % n;
          const QMoEOffload* node = nodes_[idx];
          if (!node || !node->decode_state_ || !node->decode_state_->doorbell_req_host) continue;
          const uint64_t req = *node->decode_state_->doorbell_req_host;
          if (req > node->decode_state_->last_serviced_seq) {
            node->decode_state_->last_serviced_seq = req;
            cursor_ = (idx + 1) % n;
            // Cross-layer async cold-expert prefetch: while Layer `idx` is about to compute on CPU/GPU,
            // start an async H2D DMA copy on `promo_worker_` for a lookahead node 2..4 layers ahead (`idx + d`),
            // so cold expert data copy and compute execute in parallel.
            if (cross_layer_promo_enabled_ && n >= 4 &&
                promo_state_.load(std::memory_order_acquire) == kPromoIdle) {
              for (size_t d = 2; d <= 4 && d < n; ++d) {
                const QMoEOffload* lookahead = nodes_[(idx + d) % n];
                if (lookahead && lookahead->TryPrepareCrossLayerPromotion(&promo_req_)) {
                  promo_state_.store(kPromoQueued, std::memory_order_release);
                  {
                    std::lock_guard<std::mutex> plock(promo_cv_mutex_);
                    promo_cv_.notify_one();
                  }
                  break;
                }
              }
            }
            ready_node = node;
            servicing_node_ = node;
            break;
          }
        }
      }
      if (ready_node != nullptr) {
        // Execute CPU cold expert service outside DoorbellService::mutex_ so independent stages/sessions
        // never block on the registry mutex during CPU computation or storage reads (F16).
        ready_node->ServiceDoorbell();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          servicing_node_ = nullptr;
          service_done_cv_.notify_all();
        }
        const size_t prev = pending_.load(std::memory_order_relaxed);
        if (prev > 0) pending_.fetch_sub(1, std::memory_order_release);
        idle_since = std::chrono::steady_clock::now();
        spins = 0;
        continue;
      }
      if (pending_.load(std::memory_order_acquire) > 0) {
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
        _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
        continue;
      }
      ++spins;
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
      _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#endif
      if ((spins & 0x3ff) == 0) {
        const int spin_us =
            (qmoe::CpuBudgetGroup::Default().CurrentPhase() == qmoe::CpuBudgetGroup::Phase::kPrefillOrAux)
                ? 0
                : 2000;
        if (std::chrono::steady_clock::now() - idle_since > std::chrono::microseconds(spin_us)) {
          std::unique_lock<std::mutex> cv_lock(cv_mutex_);
          sleeping_.store(true, std::memory_order_seq_cst);
          cv_.wait(cv_lock, [this] {
            return stop_.load(std::memory_order_relaxed) || pending_.load(std::memory_order_acquire) > 0;
          });
          sleeping_.store(false, std::memory_order_seq_cst);
          idle_since = std::chrono::steady_clock::now();
          spins = 0;
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable service_done_cv_;
  const QMoEOffload* servicing_node_ = nullptr;
  std::vector<const QMoEOffload*> nodes_;
  size_t cursor_ = 0;
  std::atomic<size_t> pending_{0};
  std::atomic<bool> sleeping_{false};
  std::atomic<bool> stop_{false};
  bool worker_started_ = false;
  bool cross_layer_promo_enabled_ = true;
  std::atomic<int> promo_state_{kPromoIdle};
  CrossLayerPromoRequest promo_req_{};
  uint8_t* promo_bounce_buf_ = nullptr;
  size_t promo_bounce_cap_ = 0;
  std::mutex promo_cv_mutex_;
  std::condition_variable promo_cv_;
  std::thread promo_worker_;
  std::mutex cv_mutex_;
  std::condition_variable cv_;
  std::thread worker_;

 public:
  void CheckAndCommitBeforeDoorbellRelease(const QMoEOffload* current_node) noexcept {
    if (!cross_layer_promo_enabled_ || promo_state_.load(std::memory_order_acquire) != kPromoEnqueued) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t n = nodes_.size();
    if (n < 2) return;
    const QMoEOffload* next_node = nodes_[cursor_ % n];
    if (next_node && next_node != current_node && promo_req_.node == next_node) {
      if (next_node->TryCommitCrossLayerPromotion(promo_req_)) {
        promo_req_ = CrossLayerPromoRequest{};
        promo_state_.store(kPromoIdle, std::memory_order_release);
      }
    }
  }
};

QMoEOffload::QMoEOffload(const OpKernelInfo& info,
                         const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                         const onnxruntime::moe_offload::MoeExpertSourceDescriptor& desc,
                         const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan)
    : CudaKernel(info),
      MoEBase(info, GetDeviceProp()),
      v4_planned_mode_(true),
      v4_policy_(policy),
      v4_plan_(plan) {
  allow_cuda_graph_ = true;
  host_complement_mode_ = false;
  ssd_cache_slots_ = 0;
  host_cache_bytes_ = static_cast<size_t>(plan.host_cache_bytes);
  swap_victim_to_host_cache_ = (host_cache_bytes_ > 0);
  ::onnxruntime::cuda::SetQMoEDeviceArenasShrunkHook(&OnQMoEDeviceArenasShrunk);
  ORT_ENFORCE(sm_ >= 80, "QMoEOffload requires SM80 or later.");

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  auto input_elem_type = info.GetKernelInfo().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  is_bf16_ = (input_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
#else
  int32_t input_elem_type = info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  is_bf16_ = (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16);
#endif

  session_controller_ = GetQMoESessionController();
  if (session_controller_) {
    device_id_ = session_controller_->DeviceId();
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != device_id_) {
      CUDA_CALL_THROW(cudaSetDevice(device_id_));
    }
    session_controller_->RegisterOffloadNode();
    session_controller_->RegisterGraphReplayHook(
        this,
        [this](cudaStream_t s) -> common::Status { return StepBetweenRuns(s); },
        [this](cudaStream_t /*s*/, bool /*sync*/) -> common::Status {
          if (async_failed_.load(std::memory_order_acquire)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload async CPU expert path failed: ", async_error_message_);
          }
          return common::Status::OK();
        });
  } else {
    static_cast<void>(cudaGetDevice(&device_id_));
  }

  mxfp4_mode_ = (desc.quant_type == "mxfp4" || desc.quant_type == "fp4");
  nvfp4_mode_ = (desc.quant_type == "nvfp4") || mxfp4_mode_;
  doorbell_enabled_ = plan.enable_hybrid_cpu_decode && nvfp4_mode_;
  overlap_shared_expert_ = true;
  fuse_merge_finalize_ = true;
#ifdef ORT_USE_EP_API_ADAPTERS
  if (info.GetKernelInfo().GetOutputCount() > 0) {
    output0_name_ = info.GetKernelInfo().GetOutputName(0);
  }
#else
  if (!info.node().OutputDefs().empty() && info.node().OutputDefs()[0] != nullptr) {
    output0_name_ = info.node().OutputDefs()[0]->Name();
  }
#endif
  static_cast<void>(GetQMoESharedExpertBridge());

  ORT_ENFORCE(nvfp4_mode_, "QMoEOffload v4 planned backend currently requires quant_type in {'nvfp4', 'fp4'}.");
  InitializeNvfp4Offload(info);
}

QMoEOffload::QMoEOffload(const OpKernelInfo& info)
    : CudaKernel(info), MoEBase(info, GetDeviceProp()) {
  ::onnxruntime::cuda::SetQMoEDeviceArenasShrunkHook(&OnQMoEDeviceArenasShrunk);
  staging_optimization_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OFFLOAD_STAGING_OPT", 0);
  ORT_ENFORCE(staging_optimization_ >= 0 && staging_optimization_ <= 3,
              "ORT_QMOE_OFFLOAD_STAGING_OPT must be 0..3 (tiled scales=1, combined upload=2).");
  allow_cuda_graph_ =
      (info.GetAttrOrDefault<int64_t>("allow_cuda_graph", 0) == 1) ||
      (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ALLOW_CUDA_GRAPH", 0) == 1);
  host_complement_mode_ =
      (info.GetAttrOrDefault<int64_t>("host_complement_mode", 0) == 1) ||
      (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_COMPLEMENT_MODE", 0) == 1);
  ssd_cache_slots_ = std::max(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("ssd_cache_slots", 0)),
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SSD_CACHE_SLOTS", 0));
  const int host_cache_mib = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_MIB", 0);
  ORT_ENFORCE(host_cache_mib >= 0, "ORT_QMOE_HOST_CACHE_MIB must be nonnegative.");
  if (host_cache_mib > 0) {
    const int layers = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_LAYERS", 0);
    ORT_ENFORCE(layers > 0 && info.GetAttrOrDefault<int64_t>("layer_index", -1) >= 0 &&
                    info.GetAttrOrDefault<int64_t>("layer_index", -1) < layers,
                "Bounded host cache requires ORT_QMOE_HOST_CACHE_LAYERS and a valid layer_index.");
    ORT_ENFORCE(!host_complement_mode_ && ssd_cache_slots_ == 0 &&
                    ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 0 &&
                    ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_ROWS", 0) == 0,
                "Bounded host cache cannot be combined with complement, legacy SSD, whole-expert pinning, or shared streaming.");
    swap_victim_to_host_cache_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_VICTIM_SWAP", 1) != 0;
    host_cache_bytes_ = (SafeInt<size_t>(host_cache_mib) * (size_t{1} << 20)) / layers;
    ORT_ENFORCE(host_cache_bytes_ > 0, "Host cache budget is too small for the configured layer count.");
  }
  ssd_store_path_ = info.GetAttrOrDefault<std::string>("ssd_store_path", "");
  if (ssd_store_path_.empty()) {
    if (const char* env_path = std::getenv("ORT_QMOE_SSD_STORE_PATH")) {
      ssd_store_path_ = env_path;
    }
  }
  if (!allow_cuda_graph_) {
    const auto& options = info.GetConfigOptions();
    for (const auto* key : {"ep.cuda.enable_cuda_graph", "ep.CUDAExecutionProvider.enable_cuda_graph", "enable_cuda_graph"}) {
      const auto graph = options.GetConfigEntry(key);
      ORT_ENFORCE(!graph.has_value() || *graph == "0" || *graph == "false",
                  "QMoEOffload requires CUDA graphs off; enabled graph option ", key, ".");
    }
#ifndef BUILD_CUDA_EP_AS_PLUGIN
    ORT_ENFORCE(!info.GetExecutionProvider()->IsGraphCaptureEnabled(),
                "QMoEOffload requires CUDA graphs off; eager streaming cannot be captured.");
#endif
  }
  ORT_ENFORCE(sm_ >= 80, "QMoEOffload requires SM80 or later.");

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  auto input_elem_type = info.GetKernelInfo().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  is_bf16_ = (input_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
#else
  int32_t input_elem_type = info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  is_bf16_ = (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16);
#endif

  session_controller_ = GetQMoESessionController();
  if (session_controller_) {
    device_id_ = session_controller_->DeviceId();
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != device_id_) {
      CUDA_CALL_THROW(cudaSetDevice(device_id_));
    }
    session_controller_->RegisterOffloadNode();
    session_controller_->RegisterGraphReplayHook(
        this,
        [this](cudaStream_t s) -> common::Status { return StepBetweenRuns(s); },
        [this](cudaStream_t /*s*/, bool /*sync*/) -> common::Status {
          if (async_failed_.load(std::memory_order_acquire)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload async CPU expert path failed: ", async_error_message_);
          }
          return common::Status::OK();
        });
  } else {
    static_cast<void>(cudaGetDevice(&device_id_));
  }

  const std::string quant_type = info.GetAttrOrDefault<std::string>("quant_type", "int");
  ORT_ENFORCE(quant_type == "int" || quant_type == "nvfp4" || quant_type == "mxfp4" || quant_type == "fp4",
              "QMoEOffload supports quant_type='int', 'nvfp4', or 'mxfp4' ('fp4'), got '", quant_type, "'.");
  mxfp4_mode_ = (quant_type == "mxfp4" || quant_type == "fp4");
  nvfp4_mode_ = (quant_type == "nvfp4") || mxfp4_mode_;
  doorbell_enabled_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_DOORBELL", nvfp4_mode_ ? 1 : 0) == 1;
  overlap_shared_expert_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OVERLAP_SHARED_EXPERT", 1) == 1;
  fuse_merge_finalize_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_FUSE_MERGE_FINALIZE", 1) == 1;
#ifdef ORT_USE_EP_API_ADAPTERS
  if (info.GetKernelInfo().GetOutputCount() > 0) {
    output0_name_ = info.GetKernelInfo().GetOutputName(0);
  }
#else
  if (!info.node().OutputDefs().empty() && info.node().OutputDefs()[0] != nullptr) {
    output0_name_ = info.node().OutputDefs()[0]->Name();
  }
#endif
  static_cast<void>(GetQMoESharedExpertBridge());

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("expert_weight_bits", 4) == 4,
              "QMoEOffload requires expert_weight_bits=4.");
  for (const auto* attribute : {"fc1_expert_weight_bits", "fc2_expert_weight_bits", "fc3_expert_weight_bits"}) {
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>(attribute, 4) == 4,
                "QMoEOffload does not support mixed expert weight widths.");
  }

  ORT_ENFORCE(activation_type_ == ck::ActivationType::Swiglu && swiglu_fusion_ == 1 && !use_sparse_mixer_,
              "QMoEOffload requires fused interleaved SwiGLU and full softmax top-k routing.");
  const auto normalize = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0);
  ORT_ENFORCE((normalize == 0 || normalize == 1) &&
                  info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 0 &&
                  info.GetAttrOrDefault<int64_t>("accuracy_level", 0) == 0,
              "QMoEOffload does not support sparse mixer or CPU accuracy modes.");
  ORT_ENFORCE(std::isfinite(activation_alpha_) && std::isfinite(activation_beta_) &&
                  !std::isnan(swiglu_limit_) && swiglu_limit_ >= 0,
              "QMoEOffload has invalid SwiGLU parameters.");

  if (nvfp4_mode_) {
    InitializeNvfp4Offload(info);
    return;
  }

  ORT_ENFORCE(!is_bf16_, "QMoEOffload quant_type='int' requires FP16 activations.");
  ORT_ENFORCE(block_size_ == 32, "QMoEOffload symmetric INT4 requires block_size=32.");
  const auto prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
  ORT_ENFORCE((prepacked == -1 || prepacked == 1) &&
                  info.GetAttrOrDefault<std::string>("packing_layout", "cutlass_sm80_int4_v1") ==
                      "cutlass_sm80_int4_v1",
              "QMoEOffload requires immutable cutlass_sm80_int4_v1 prepacked weights; raw layouts are unsupported.");

  const bool legacy_shared = InputExists(info, 21);
  const bool compact_shared = !legacy_shared && info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                              InputExists(info, 2) && !InputExists(info, 5) && !InputExists(info, 6);
  shared_mode_ = legacy_shared || compact_shared;
  v3_compact_shared_ = compact_shared;
  if (shared_mode_) {
    InitializeShared(info);
    return;
  }

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("layout_version", -1) == -1,
              "QMoEOffload layout_version requires shared expert_blocks.");
  for (int index = 22; index <= 25; ++index) {
    ORT_ENFORCE(!InputExists(info, index), "QMoEOffload shared inputs require expert_blocks.");
  }
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("expert_cache_bytes", 0) == 0,
              "QMoEOffload eager streaming supports expert_cache_bytes=0 only; persistent cache is not implemented.");
  for (int index : {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}) {
    ORT_ENFORCE(!InputExists(info, index),
                "QMoEOffload does not support FC3, zero points, separate router weights or FP quantization inputs; slot ",
                index, " is present.");
  }

  constexpr std::array<int, 4> bank_slots{2, 5, 3, 6};
  for (size_t bank = 0; bank < bank_slots.size(); ++bank) {
    ORT_ENFORCE(Info().TryGetConstantInput(bank_slots[bank], &banks_[bank]) && banks_[bank] != nullptr,
                "QMoEOffload bank and scale inputs must be immutable CPU initializers; missing slot ", bank_slots[bank], ".");
    const auto& location = banks_[bank]->Location().device;
    ORT_ENFORCE(location.Type() == OrtDevice::CPU && location.MemType() == OrtDevice::MemType::DEFAULT,
                "QMoEOffload bank and scale inputs must use pageable CPU memory.");
    ORT_ENFORCE(banks_[bank]->Shape().NumDimensions() == 3 && banks_[bank]->DataRaw() != nullptr,
                "QMoEOffload banks and scales must be nonempty contiguous rank-three tensors.");
    ORT_ENFORCE(bank < 2 ? banks_[bank]->IsDataType<uint8_t>() : banks_[bank]->IsDataType<MLFloat16>(),
                "QMoEOffload requires uint8 packed banks and FP16 scales.");
  }
  const auto& fc1 = banks_[0]->Shape();
  const auto& fc2 = banks_[1]->Shape();
  ORT_ENFORCE(fc1[0] >= 2 && fc1[0] <= std::numeric_limits<int>::max() &&
                  fc1[1] > 0 && fc1[1] <= std::numeric_limits<int>::max() &&
                  fc2[1] > 0 && fc2[1] <= std::numeric_limits<int>::max() / 2,
              "QMoEOffload expert/feature dimensions are invalid or exceed the runner's integer limits.");
  num_experts_ = static_cast<int>(fc1[0]);
  hidden_size_ = static_cast<int>(fc1[1]);
  inter_size_ = static_cast<int>(fc2[1]);
  ORT_ENFORCE(hidden_size_ % 64 == 0 && inter_size_ % 64 == 0 && fc1[2] == inter_size_ &&
                  fc2[0] == num_experts_ && fc2[2] == hidden_size_ / 2,
              "QMoEOffload requires SM80 packed FC1 [E,H,I] and FC2 [E,I,H/2], with H and I multiples of 64.");
  const auto& scales1 = banks_[2]->Shape();
  const auto& scales2 = banks_[3]->Shape();
  ORT_ENFORCE(scales1[0] == num_experts_ && scales1[1] == int64_t{2} * inter_size_ &&
                  scales1[2] == hidden_size_ / 32 && scales2[0] == num_experts_ &&
                  scales2[1] == hidden_size_ && scales2[2] == inter_size_ / 32,
              "QMoEOffload requires FP16 scales [E,2I,H/32] and [E,H,I/32].");
  for (size_t bias = 0; bias < 2; ++bias) {
    const int slot = bias == 0 ? 4 : 7;
    if (!InputExists(info, slot)) continue;
    const size_t bank = 4 + bias;
    ORT_ENFORCE(Info().TryGetConstantInput(slot, &banks_[bank]) && banks_[bank] != nullptr &&
                    banks_[bank]->IsDataType<MLFloat16>() && banks_[bank]->Shape().NumDimensions() == 2 &&
                    banks_[bank]->Shape()[0] == num_experts_ &&
                    banks_[bank]->Shape()[1] == (bias == 0 ? int64_t{2} * inter_size_ : hidden_size_) &&
                    banks_[bank]->Location().device.Type() == OrtDevice::CPU &&
                    banks_[bank]->Location().device.MemType() == OrtDevice::MemType::DEFAULT,
                "QMoEOffload bias inputs must be immutable pageable CPU FP16 [E,2I] or [E,H] tensors.");
  }
  for (size_t bank = 2; bank < banks_.size(); ++bank) {
    if (!banks_[bank]) continue;
    ORT_ENFORCE(banks_[bank]->DataRaw() != nullptr &&
                    reinterpret_cast<uintptr_t>(banks_[bank]->DataRaw()) % alignof(MLFloat16) == 0,
                "QMoEOffload FP16 CPU scales and biases require non-null naturally aligned storage.");
  }
  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload k must be in [1,E].");
  ORT_ENFORCE(!onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_fused_finalize_supported(
                  sm_, 1, k_, k_, hidden_size_, inter_size_, 4, 32),
              "QMoEOffload requires the resident grouped FC2 FP16 contribution contract; "
              "decode shapes using fused FP32 GEMV finalization are unsupported. Use k >= ",
              kGroupedMoeMinRows, ".");

  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto wave = info.GetAttrOrDefault<int64_t>("expert_wave_size", 1);
  const auto gpu_limit = info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{64} << 20);
  const auto pinned_limit = info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{8} << 20);
  ORT_ENFORCE(tile > 0 && tile <= std::numeric_limits<int>::max() && wave > 0 && wave < num_experts_ &&
                  gpu_limit > 0 && pinned_limit > 0,
              "QMoEOffload needs positive tile/byte limits and expert_wave_size in [1,E-1]; full-bank staging is forbidden.");
  token_tile_size_ = static_cast<int>(tile);
  expert_wave_size_ = static_cast<int>(wave);
  max_routes_ = SafeInt<int>(tile) * k_;
  max_wave_routes_ = std::max(kGroupedMoeMinRows, static_cast<int>(SafeInt<int>(tile) * std::min(k_, wave)));
  ORT_ENFORCE(SafeInt<int64_t>(tile) * num_experts_ <= std::numeric_limits<int>::max() &&
                  SafeInt<int64_t>(max_wave_routes_) * expert_wave_size_ <= std::numeric_limits<int>::max(),
              "QMoEOffload tile routing exceeds the runner's integer indexing limits.");

  expert_bytes_[0] = SafeInt<size_t>(hidden_size_) * inter_size_;
  expert_bytes_[1] = SafeInt<size_t>(inter_size_) * hidden_size_ / 2;
  expert_bytes_[2] = SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 32) * sizeof(MLFloat16);
  expert_bytes_[3] = SafeInt<size_t>(hidden_size_) * (inter_size_ / 32) * sizeof(MLFloat16);
  expert_bytes_[4] = banks_[4] ? static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(MLFloat16)) : size_t{0};
  expert_bytes_[5] = banks_[5] ? static_cast<size_t>(SafeInt<size_t>(hidden_size_) * sizeof(MLFloat16)) : size_t{0};
  size_t staging_bytes = 0;
  for (size_t bank = 0; bank < banks_.size(); ++bank) {
    ORT_ENFORCE(!banks_[bank] || SafeInt<size_t>(expert_bytes_[bank]) * num_experts_ == banks_[bank]->SizeInBytes(),
                "QMoEOffload bank byte extent is inconsistent with its packed layout.");
    const size_t bytes = SafeInt<size_t>(expert_bytes_[bank]) * expert_wave_size_;
    staging_offsets_[bank] = AddSlab(bytes, staging_bytes);
    offsets_[bank] = AddSlab(bytes, scratch_bytes_);
  }
  expert_slab_bytes_ = scratch_bytes_;
  ORT_ENFORCE(staging_bytes == expert_slab_bytes_, "QMoEOffload CPU and CUDA bank slab layouts differ.");
  const int max_metadata_routes = std::max(max_routes_, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(max_metadata_routes) * sizeof(int);
  host_indices_offset_ = AddSlab(route_bytes, staging_bytes);
  host_routes_offset_ = AddSlab(route_bytes, staging_bytes);
  host_compact_offset_ = AddSlab(route_bytes, staging_bytes);
  ORT_ENFORCE(staging_bytes <= SafeInt<size_t>(pinned_limit),
              "QMoEOffload pinned ring requires ", staging_bytes, " bytes, exceeding pinned_staging_bytes=", pinned_limit, ".");
  ORT_ENFORCE(scratch_bytes_ <= SafeInt<size_t>(gpu_limit),
              "QMoEOffload expert wave alone requires ", scratch_bytes_, " bytes, exceeding gpu_working_bytes=", gpu_limit, ".");

  runner_ = std::make_unique<ck::CutlassMoeFCRunner<half, cutlass::uint4b_t>>(
      sm_, activation_type_, false, false);
  ORT_ENFORCE(!runner_->getTactics().empty(), "QMoEOffload has no compatible SM80 INT4 GEMM tactic.");
  const ck::MOEParallelismConfig parallelism;
  for (int experts = 1; experts <= expert_wave_size_; ++experts) {
    const int route_bound = std::max(kGroupedMoeMinRows,
                                     static_cast<int>(SafeInt<int>(token_tile_size_) * std::min<int64_t>(k_, experts)));
    for (int rows = kGroupedMoeMinRows; rows <= route_bound; ++rows) {
      workspace_bytes_ = std::max(workspace_bytes_, runner_->getWorkspaceSize(
                                                        rows, hidden_size_, inter_size_, experts, 1, activation_type_, parallelism, true, swiglu_fusion_));
    }
  }
  offsets_[RouterScales] = AddSlab(SafeInt<size_t>(max_routes_) * sizeof(float), scratch_bytes_);
  offsets_[RouterIndices] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[UnitScales] = AddSlab(SafeInt<size_t>(max_metadata_routes) * sizeof(float), scratch_bytes_);
  offsets_[IdentityPermutation] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[RouteIds] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[CompactExperts] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[WavePermutation] = AddSlab(route_bytes, scratch_bytes_);
  const size_t wave_activation_bytes = SafeInt<size_t>(max_wave_routes_) * hidden_size_ * sizeof(half);
  offsets_[GatheredInput] = AddSlab(wave_activation_bytes, scratch_bytes_);
  offsets_[WaveOutput] = AddSlab(wave_activation_bytes, scratch_bytes_);
  offsets_[Contributions] = AddSlab(SafeInt<size_t>(max_routes_) * hidden_size_ * sizeof(half), scratch_bytes_);
  offsets_[RouteBias] = AddSlab(banks_[5] ? static_cast<size_t>(SafeInt<size_t>(max_routes_) * hidden_size_ * sizeof(half)) : size_t{0},
                                scratch_bytes_);
  offsets_[RunnerWorkspace] = AddSlab(workspace_bytes_, scratch_bytes_);
  ORT_ENFORCE(scratch_bytes_ <= SafeInt<size_t>(gpu_limit),
              "QMoEOffload needs ", scratch_bytes_, " GPU scratch bytes (expert wave, route slabs, activations and ",
              workspace_bytes_, " runner workspace), exceeding gpu_working_bytes=", gpu_limit, ".");

  expert_to_slot_.reserve(num_experts_);
  for (int expert = 0; expert < num_experts_; ++expert) expert_to_slot_.emplace_back(-1);
  wave_experts_.reserve(expert_wave_size_);
  for (int expert = 0; expert < expert_wave_size_; ++expert) wave_experts_.emplace_back(0);
  pinned_ring_ = AllocateBufferOnCPUPinned<uint8_t>(staging_bytes);
  ORT_ENFORCE(pinned_ring_ != nullptr, "QMoEOffload failed to allocate its bounded pinned staging ring.");
  if (staging_optimization_ & 2) std::memset(pinned_ring_.get(), 0, expert_slab_bytes_);
}

QMoEOffload::~QMoEOffload() {
  int prev_dev = -1;
  const bool restore_dev = (device_id_ >= 0 && cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != device_id_);
  if (restore_dev) {
    static_cast<void>(cudaSetDevice(device_id_));
  }
  if (session_controller_) {
    session_controller_->UnregisterOffloadNode(this);
  }
  if (nvfp4_mode_ && nvfp4_fc1_w_) {
    UnregisterQMoEDenseLayerHostWeights(
        device_id_, layer_index_, static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()));
  }
  // 4-step teardown order (v3 Section 7.7 / F10):
  // 1) Drain participating main, auxiliary, promotion, prefill DMA, and landing copy streams
  //    while CPU expert service remains active.
  //    Wait with failed-context guard so a dead CUDA context that skipped host callbacks never hangs.
  bool context_failed = false;
  if (final_device_done_ != nullptr && final_device_done_recorded_.load(std::memory_order_acquire)) {
    const cudaError_t sync_err = cudaEventSynchronize(final_device_done_);
    if (IsStickyCudaContextError(sync_err)) context_failed = true;
  }
  if (aux_stream_) {
    const cudaError_t sync_err = cudaStreamSynchronize(aux_stream_);
    if (sync_err != cudaSuccess) {
      async_in_flight_.store(false, std::memory_order_release);
      async_failed_.store(true, std::memory_order_release);
      if (IsStickyCudaContextError(sync_err)) context_failed = true;
    }
  }
  if (promo_stream_) {
    const cudaError_t sync_err = cudaStreamSynchronize(promo_stream_);
    if (IsStickyCudaContextError(sync_err)) context_failed = true;
  }
  if (nvfp4_mode_) {
    auto& dma_state = DevicePrefillDmaState::ForDevice(device_id_);
    std::lock_guard<std::mutex> dma_lock(dma_state.mutex);
    if (dma_state.dma_stream) {
      const cudaError_t sync_err = cudaStreamSynchronize(dma_state.dma_stream);
      if (IsStickyCudaContextError(sync_err)) context_failed = true;
    }
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (async_in_flight_.load(std::memory_order_acquire)) {
    const cudaError_t peek_err = cudaPeekAtLastError();
    if (IsStickyCudaContextError(peek_err) || std::chrono::steady_clock::now() > deadline) {
      if (IsStickyCudaContextError(peek_err)) context_failed = true;
      break;
    }
    std::this_thread::yield();
  }
  if (stream_shared_) {
    std::lock_guard<std::mutex> lock(stream_shared_->mutex);
    if (stream_shared_->copy_stream) {
      const cudaError_t sync_err = cudaStreamSynchronize(stream_shared_->copy_stream);
      if (IsStickyCudaContextError(sync_err)) context_failed = true;
    }
  }
  // 2) Stop doorbell and SSD background services for this node after all device streams have drained.
  if (decode_state_ && decode_state_->registered_with_doorbell) {
    DoorbellService::Instance().UnregisterNode(this);
    decode_state_->registered_with_doorbell = false;
  }
  const uint64_t ssd_hits = file_expert_store_.CacheHits();
  const uint64_t ssd_pf_hits = file_expert_store_.PrefetchHits();
  const uint64_t ssd_misses = file_expert_store_.SsdMisses();
  const uint64_t ssd_read_ns = file_expert_store_.ReadNanoseconds();
  const uint64_t ssd_read_bytes = file_expert_store_.BytesReadFromSsd();
  if (callback_timing_ && callback_timing_->SelectedCalls() > 0 && callback_timing_->Quiescent()) {
    try {
      callback_timing_->Persist();
    } catch (...) {
    }
  }
  if (owned_layer_cache_) {
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    owned_layer_cache_->owner = nullptr;
    owned_layer_cache_->slab_ptr = nullptr;
    owned_layer_cache_->file_store_ptr = nullptr;
  }
  // If a sticky CUDA context error occurred and device quiescence cannot be guaranteed,
  // quarantine device allocations and registered host ranges rather than calling cudaHostUnregister/cudaFree
  // while DMA state is indeterminate (v3 Section 7.7 / F10).
  if (context_failed) {
    static_cast<void>(resident_slab_.release());
    static_cast<void>(resident_fc1_bias_slab_.release());
    static_cast<void>(stream_fc1_bias_slab_[0].release());
    static_cast<void>(stream_fc1_bias_slab_[1].release());
    static_cast<void>(slot_table_dev_.release());
    static_cast<void>(fc2_bias_gpu_.release());
    static_cast<void>(complement_host_slab_.release());
    if (decode_state_) {
      static_cast<void>(decode_state_->scratch.release());
      decode_state_->ReleasePinned(false);
    }
    if (experts_registered_) {
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      for (const void* ptr : registered_host_ptrs_) {
        if (ptr && pinned_budget) pinned_budget->QuarantineHostRange(ptr);
      }
      registered_host_ptrs_.clear();
    }
  } else {
    if (final_device_done_) cudaEventDestroy(final_device_done_);
    if (slot_reader_done_) cudaEventDestroy(slot_reader_done_);
    if (join_event_) cudaEventDestroy(join_event_);
    if (fork_event_) cudaEventDestroy(fork_event_);
    if (aux_stream_) cudaStreamDestroy(aux_stream_);
    if (promo_ready_) cudaEventDestroy(promo_ready_);
    if (promo_stream_) cudaStreamDestroy(promo_stream_);
    if (decode_state_) {
      decode_state_->ReleasePinned(true);
    }
    // 4) Unregister pinned host ranges only after all DMA and CPU readers have finished (F08).
    if (experts_registered_) {
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      for (const void* ptr : registered_host_ptrs_) {
        if (!ptr) continue;
        size_t reg_bytes = 0;
        if (!pinned_budget) {
          static_cast<void>(cudaHostUnregister(const_cast<void*>(ptr)));
        } else if (pinned_budget->ReleaseHostRangeOwner(ptr, &reg_bytes)) {
          const bool unreg_ok = (cudaHostUnregister(const_cast<void*>(ptr)) == cudaSuccess);
          if (!unreg_ok) static_cast<void>(cudaGetLastError());
          pinned_budget->FinishUnregisterHostRange(reg_bytes, unreg_ok);
        }
      }
      registered_host_ptrs_.clear();
    }
  }
  if (promo_budget_retained_) {
    SessionPromotionBudget::Instance().Release(session_controller_.get());
    promo_budget_retained_ = false;
  }
  file_expert_store_.Shutdown();
  if (restore_dev) {
    static_cast<void>(cudaSetDevice(prev_dev));
  }
  if (!ssd_temp_file_.empty()) {
    std::remove(ssd_temp_file_.c_str());
    ssd_temp_file_.clear();
  }
  if (cpu_timing_) {
    std::fprintf(stderr,
                 "QMoEOffload cpu layer=%lld calls=%llu routes=%llu cold_routes=%llu cold_experts=%llu cpu_us=%llu classify_us=%llu "
                 "gap_in_us=%llu gap_in_n=%llu gap_out_us=%llu gap_out_n=%llu t1_us=%llu t2_us=%llu t3_us=%llu inter_n=%llu prep_us=%llu fc1_us=%llu fc2_us=%llu write_us=%llu\n",
                 static_cast<long long>(layer_index_), static_cast<unsigned long long>(timing_calls_),
                 static_cast<unsigned long long>(timing_routes_), static_cast<unsigned long long>(timing_cold_routes_),
                 static_cast<unsigned long long>(timing_cold_experts_), static_cast<unsigned long long>(timing_cpu_ns_ / 1000),
                 static_cast<unsigned long long>(timing_classify_ns_ / 1000),
                 static_cast<unsigned long long>(g_gap_in_ns.load() / 1000), static_cast<unsigned long long>(g_gap_in_n.load()),
                 static_cast<unsigned long long>(g_gap_out_ns.load() / 1000), static_cast<unsigned long long>(g_gap_out_n.load()),
                 static_cast<unsigned long long>(g_t1_ns.load() / 1000), static_cast<unsigned long long>(g_t2_ns.load() / 1000),
                 static_cast<unsigned long long>(g_t3_ns.load() / 1000), static_cast<unsigned long long>(g_inter_n.load()),
                 static_cast<unsigned long long>(timing_prep_ns_ / 1000), static_cast<unsigned long long>(timing_fc1_ns_ / 1000),
                 static_cast<unsigned long long>(timing_fc2_ns_ / 1000), static_cast<unsigned long long>(timing_write_ns_ / 1000));
  }
  if (resident_log_) {
    std::fprintf(stderr,
                 "QMoEOffload resident layer=%lld slots=%d hit_routes=%llu routes=%llu promotions=%llu complement_writebacks=%llu ssd_hits=%llu ssd_pf_hits=%llu ssd_misses=%llu ssd_read_ms=%llu ssd_read_mib=%llu victim_swaps=%llu victim_swap_fallbacks=%llu\n",
                 static_cast<long long>(layer_index_), resident_slots_, static_cast<unsigned long long>(resident_hit_routes_),
                 static_cast<unsigned long long>(resident_total_routes_), static_cast<unsigned long long>(resident_promoted_),
                 static_cast<unsigned long long>(complement_eviction_writebacks_),
                 static_cast<unsigned long long>(ssd_hits), static_cast<unsigned long long>(ssd_pf_hits),
                 static_cast<unsigned long long>(ssd_misses),
                 static_cast<unsigned long long>(ssd_read_ns / 1000000),
                 static_cast<unsigned long long>(ssd_read_bytes >> 20),
                 static_cast<unsigned long long>(victim_swaps_), static_cast<unsigned long long>(victim_swap_fallbacks_));
    for (size_t phase = 0; phase < phase_total_routes_.size(); ++phase) {
      std::fprintf(stderr, "QMoEOffload resident_phase layer=%lld phase=%zu hit_routes=%llu routes=%llu\n",
                   static_cast<long long>(layer_index_), phase,
                   static_cast<unsigned long long>(phase_hit_routes_[phase]),
                   static_cast<unsigned long long>(phase_total_routes_[phase]));
    }
  }
  if (stream_rows_ > 0 && resident_log_) {
    std::fprintf(stderr,
                 "QMoEOffload stream layer=%lld tiles=%llu experts=%llu routes=%llu cpu_routes=%llu waves=%llu reused_promotions=%llu\n",
                 static_cast<long long>(layer_index_), static_cast<unsigned long long>(stream_tiles_),
                 static_cast<unsigned long long>(stream_experts_), static_cast<unsigned long long>(stream_routes_),
                 static_cast<unsigned long long>(stream_cpu_routes_), static_cast<unsigned long long>(stream_waves_),
                 static_cast<unsigned long long>(stream_reused_promotions_));
  }
}

void QMoEOffload::StageExperts(const int* experts, int count) const {
  uint8_t* const pinned = pinned_ring_.get();
  for (int compact = 0; compact < count; ++compact) {
    const size_t expert = static_cast<size_t>(experts[compact]);
    for (size_t bank = 0; bank < 2; ++bank) {
      const size_t bytes = expert_bytes_[bank];
      const auto* source = static_cast<const uint8_t*>(banks_[bank]->DataRaw()) + expert * bytes;
      std::memcpy(pinned + staging_offsets_[bank] + static_cast<size_t>(compact) * bytes, source, bytes);
    }
    for (size_t bank = 2; bank < 4; ++bank) {
      const int columns = bank == 2 ? 2 * inter_size_ : hidden_size_;
      const int blocks = (bank == 2 ? hidden_size_ : inter_size_) / 32;
      const size_t elements = static_cast<size_t>(columns) * blocks;
      const auto* source = banks_[bank]->Data<MLFloat16>() + expert * elements;
      auto* destination = reinterpret_cast<MLFloat16*>(pinned + staging_offsets_[bank] +
                                                       static_cast<size_t>(compact) * expert_bytes_[bank]);
      if (staging_optimization_ & 1) {
        qmoe::TransposeOffloadScales(source, destination, columns, blocks);
      } else {
        for (int column = 0; column < columns; ++column) {
          for (int block = 0; block < blocks; ++block) {
            destination[static_cast<size_t>(block) * columns + column] =
                source[static_cast<size_t>(column) * blocks + block];
          }
        }
      }
    }
    for (size_t bank = 4; bank < banks_.size(); ++bank) {
      if (!banks_[bank]) continue;
      const size_t bytes = expert_bytes_[bank];
      const auto* source = static_cast<const uint8_t*>(banks_[bank]->DataRaw()) + expert * bytes;
      std::memcpy(pinned + staging_offsets_[bank] + static_cast<size_t>(compact) * bytes, source, bytes);
    }
  }
}

Status QMoEOffload::ComputeInternal(OpKernelContext* context) const {
  if (shared_mode_ || nvfp4_mode_) return ComputeShared(context);
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_RETURN_IF(stream_failed_, "QMoEOffload stream synchronization failed on a previous call.");
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input != nullptr && router_probs != nullptr, "QMoEOffload requires input and router_probs.");
  ORT_RETURN_IF_NOT(input->IsDataType<MLFloat16>() && router_probs->IsDataType<MLFloat16>(),
                    "QMoEOffload requires FP16 input and router_probs.");
  ORT_RETURN_IF_NOT(input->Location().device.Type() == OrtDevice::GPU &&
                        router_probs->Location().device.Type() == OrtDevice::GPU,
                    "QMoEOffload activation and router_probs must be on CUDA.");
  const auto& input_shape = input->Shape();
  ORT_RETURN_IF_NOT((input_shape.NumDimensions() == 2 || input_shape.NumDimensions() == 3) &&
                        input_shape[input_shape.NumDimensions() - 1] == hidden_size_,
                    "QMoEOffload input must have shape [tokens,H] or [batch,sequence,H].");
  const int64_t total_rows = input_shape.NumDimensions() == 2
                                 ? input_shape[0]
                                 : static_cast<int64_t>(SafeInt<int64_t>(input_shape[0]) * input_shape[1]);
  const auto& router_shape = router_probs->Shape();
  ORT_RETURN_IF_NOT(router_shape.NumDimensions() == 2 && router_shape[0] == total_rows && router_shape[1] == num_experts_,
                    "QMoEOffload router shape must be [flattened_tokens,E].");
  Tensor* output = context->Output(0, input_shape);
  if (total_rows == 0) return Status::OK();

  cudaStream_t stream = Stream(context);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
  ORT_RETURN_IF(capture_status != cudaStreamCaptureStatusNone,
                "QMoEOffload cannot execute while the CUDA stream is capturing a graph.");

  auto scratch = GetScratchBuffer<uint8_t>(scratch_bytes_, GetComputeStream(context));
  ORT_RETURN_IF_NOT(scratch != nullptr, "QMoEOffload failed to allocate bounded GPU scratch.");
  StreamDrain drain(stream, stream_failed_);

  uint8_t* const base = scratch.get();
  uint8_t* const pinned = pinned_ring_.get();
  auto* router_scales = reinterpret_cast<float*>(base + offsets_[RouterScales]);
  auto* router_indices = reinterpret_cast<int*>(base + offsets_[RouterIndices]);
  auto* unit_scales = reinterpret_cast<float*>(base + offsets_[UnitScales]);
  auto* identity_permutation = reinterpret_cast<int*>(base + offsets_[IdentityPermutation]);
  auto* route_ids = reinterpret_cast<int*>(base + offsets_[RouteIds]);
  auto* compact_experts = reinterpret_cast<int*>(base + offsets_[CompactExperts]);
  auto* wave_permutation = reinterpret_cast<int*>(base + offsets_[WavePermutation]);
  auto* gathered_input = reinterpret_cast<half*>(base + offsets_[GatheredInput]);
  auto* wave_output = reinterpret_cast<half*>(base + offsets_[WaveOutput]);
  auto* contributions = reinterpret_cast<half*>(base + offsets_[Contributions]);
  auto* route_bias = banks_[5] ? reinterpret_cast<half*>(base + offsets_[RouteBias]) : nullptr;
  auto* workspace = reinterpret_cast<char*>(base + offsets_[RunnerWorkspace]);
  auto* host_indices = reinterpret_cast<int*>(pinned + host_indices_offset_);
  auto* host_routes = reinterpret_cast<int*>(pinned + host_routes_offset_);
  auto* host_compact = reinterpret_cast<int*>(pinned + host_compact_offset_);

  ck::ActivationParams activation_params(activation_type_);
  activation_params.alpha = activation_alpha_;
  activation_params.beta = activation_beta_;
  activation_params.limit = swiglu_limit_;
  activation_params.swiglu_fusion = swiglu_fusion_;
  const ck::MOEParallelismConfig parallelism;
  const auto quant_params = ck::QuantParams::GroupWise(32, base + offsets_[Fc1Scales], base + offsets_[Fc2Scales]);
  const void* fc1_bias = banks_[4] ? reinterpret_cast<const void*>(base + offsets_[Fc1Bias]) : nullptr;

  for (int64_t row_offset = 0; row_offset < total_rows;) {
    const int tile_rows = static_cast<int>(std::min<int64_t>(token_tile_size_, total_rows - row_offset));
    const int route_count = SafeInt<int>(tile_rows) * k_;
    const size_t activation_offset = SafeInt<size_t>(row_offset) * hidden_size_;
    const size_t router_offset = SafeInt<size_t>(row_offset) * num_experts_;
    const half* tile_input = reinterpret_cast<const half*>(input->DataRaw()) + activation_offset;
    half* tile_output = reinterpret_cast<half*>(output->MutableDataRaw()) + activation_offset;
    const half* tile_router = reinterpret_cast<const half*>(router_probs->DataRaw()) + router_offset;

    LaunchSoftmaxTopK(tile_router, router_scales, router_indices, tile_rows, num_experts_,
                      static_cast<int>(k_), normalize_routing_weights_, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        host_indices, router_indices, SafeInt<size_t>(route_count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    ORT_RETURN_IF_NOT(qmoe::InitializeOffloadRoutes(
                          host_indices, route_count, num_experts_, static_cast<int>(k_), expert_to_slot_.data()),
                      "QMoEOffload observed an out-of-range or duplicate top-k expert index.");
    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(
        unit_scales, identity_permutation, std::max(route_count, max_wave_routes_), stream));

    for (int first_expert = 0; first_expert < num_experts_;) {
      const qmoe::OffloadWave wave = qmoe::MakeOffloadWave(
          host_indices, route_count, num_experts_, first_expert, expert_wave_size_,
          expert_to_slot_.data(), wave_experts_.data(), host_routes, host_compact);
      first_expert = wave.next_expert;
      if (wave.expert_count == 0) break;

      const int compute_rows = std::max(wave.route_count, kGroupedMoeMinRows);
      for (int row = wave.route_count; row < compute_rows; ++row) {
        host_routes[row] = host_routes[0];
        host_compact[row] = host_compact[0];
      }
      StageExperts(wave_experts_.data(), wave.expert_count);
      if (staging_optimization_ & 2) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(base, pinned, expert_slab_bytes_, cudaMemcpyHostToDevice, stream));
      } else {
        for (size_t bank = 0; bank < banks_.size(); ++bank) {
          if (expert_bytes_[bank] == 0) continue;
          const size_t bytes = SafeInt<size_t>(wave.expert_count) * expert_bytes_[bank];
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
              base + offsets_[bank], pinned + staging_offsets_[bank], bytes, cudaMemcpyHostToDevice, stream));
        }
      }
      const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(route_ids, host_routes, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          compact_experts, host_compact, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
          tile_input, route_ids, gathered_input, compute_rows, hidden_size_, static_cast<int>(k_), stream));

      runner_->runMoe(gathered_input, nullptr, compact_experts, unit_scales,
                      base + offsets_[Fc1Weights], fc1_bias, activation_type_,
                      base + offsets_[Fc2Weights], nullptr, quant_params,
                      compute_rows, hidden_size_, inter_size_, wave.expert_count, 1,
                      workspace, wave_output, wave_permutation, parallelism, activation_params, {}, stream);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
          wave_output, route_ids, contributions, wave.route_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      if (route_bias != nullptr) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(base + offsets_[Fc2Bias]), compact_experts, route_ids,
            route_bias, wave.route_count, hidden_size_, stream));
      }
      CUDA_RETURN_IF_ERROR(drain.Synchronize());
    }

    ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
        contributions, tile_output, route_bias, router_scales, identity_permutation, nullptr,
        identity_permutation, nullptr, tile_rows, hidden_size_, static_cast<int>(k_),
        route_count, parallelism, false, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    row_offset += tile_rows;
  }
  CUDA_RETURN_IF_ERROR(drain.Wait());
  return Status::OK();
}

void QMoEOffload::InitializeNvfp4Offload(const OpKernelInfo& info) {
  const int64_t expected_block_size = mxfp4_mode_ ? 32 : 16;
  ORT_ENFORCE(block_size_ == -1 || block_size_ == expected_block_size,
              "QMoEOffload FP4 mode requires block_size=", expected_block_size, ", got ", block_size_, ".");
  block_size_ = expected_block_size;
  const auto prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
  ORT_ENFORCE(prepacked == 1, "QMoEOffload FP4 mode requires weights_prepacked=1 (row-major FP4).");
  ORT_ENFORCE(activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
                  (swiglu_fusion_ == 0 || swiglu_fusion_ == 1) && !use_sparse_mixer_,
              "QMoEOffload FP4 mode requires interleaved SwiGLU activation and use_sparse_mixer=0.");

  for (int index : {8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20}) {
    ORT_ENFORCE(!InputExists(info, index), "QMoEOffload FP4 unsupported input slot ", index, " is present.");
  }

  layer_index_ = v4_planned_mode_ ? static_cast<int64_t>(v4_plan_.layer_index)
                                  : info.GetAttrOrDefault<int64_t>("layer_index", 0);
  if (layer_index_ < 0) layer_index_ = 0;

  // Check if using contiguous expert_blocks or 6-input NVFP4/MXFP4 CPU bank tensors.
  const bool has_blocks_21 = !v4_planned_mode_ && InputExists(info, 21);
  const bool has_blocks_2 = !v4_planned_mode_ && !has_blocks_21 &&
                            info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                            InputExists(info, 2) && !InputExists(info, 5);
  if (has_blocks_21 || has_blocks_2) {
    const int blk_slot = has_blocks_21 ? 21 : 2;
    v3_compact_shared_ = has_blocks_2;
    const auto hidden = info.GetAttrOrDefault<int64_t>("hidden_size", 0);
    const auto inter = info.GetAttrOrDefault<int64_t>("inter_size", 0);
    const bool layout_ok = mxfp4_mode_ ? qmoe_shared::MakeMxfp4ExpertBlockLayout(hidden, inter, &nvfp4_layout_)
                                       : qmoe_shared::MakeNvfp4ExpertBlockLayout(hidden, inter, &nvfp4_layout_);
    ORT_ENFORCE(layout_ok, "QMoEOffload FP4 shared mode has invalid dimensions.");
    ORT_ENFORCE(info.TryGetConstantInput(blk_slot, &expert_blocks_) && expert_blocks_ &&
                    expert_blocks_->IsDataType<uint8_t>() &&
                    expert_blocks_->Location().device.Type() == OrtDevice::CPU &&
                    expert_blocks_->Shape().NumDimensions() == 2 &&
                    expert_blocks_->Shape()[0] > 0 &&
                    expert_blocks_->Shape()[1] == static_cast<int64_t>(nvfp4_layout_.stride),
                "QMoEOffload FP4 expert_blocks must be CPU uint8 [E, stride].");
    num_experts_ = static_cast<int>(expert_blocks_->Shape()[0]);
    hidden_size_ = static_cast<int>(hidden);
    inter_size_ = static_cast<int>(inter);
  } else {
    ORT_ENFORCE(info.TryGetConstantInput(2, &nvfp4_fc1_w_) && nvfp4_fc1_w_ &&
                    info.TryGetConstantInput(3, &nvfp4_fc1_s_) && nvfp4_fc1_s_ &&
                    info.TryGetConstantInput(5, &nvfp4_fc2_w_) && nvfp4_fc2_w_ &&
                    info.TryGetConstantInput(6, &nvfp4_fc2_s_) && nvfp4_fc2_s_,
                "QMoEOffload FP4 bank mode requires constant CPU initializers for slots 2, 3, 5, 6.");
    if (mxfp4_mode_) {
      ORT_ENFORCE(nvfp4_fc1_s_->IsDataType<Float8E8M0>() && nvfp4_fc2_s_->IsDataType<Float8E8M0>(),
                  "QMoEOffload MXFP4 requires Float8E8M0 block scales.");
    } else {
      ORT_ENFORCE(nvfp4_fc1_s_->IsDataType<Float8E4M3FN>() && nvfp4_fc2_s_->IsDataType<Float8E4M3FN>(),
                  "QMoEOffload NVFP4 requires Float8E4M3FN block scales.");
    }
    for (const Tensor* t : {nvfp4_fc1_w_, nvfp4_fc1_s_, nvfp4_fc2_w_, nvfp4_fc2_s_}) {
      ORT_ENFORCE(t->Location().device.Type() == OrtDevice::CPU && t->DataRaw() != nullptr,
                  "QMoEOffload FP4 expert weights/scales must be on CPU.");
    }
    const auto& fc1_shape = nvfp4_fc1_w_->Shape();
    const auto& fc2_shape = nvfp4_fc2_w_->Shape();
    ORT_ENFORCE(fc1_shape.NumDimensions() == 3 && fc2_shape.NumDimensions() == 3 &&
                    fc1_shape[0] == fc2_shape[0] && fc1_shape[0] > 0 &&
                    fc1_shape[1] == fc2_shape[2] * 2 && fc1_shape[2] == fc2_shape[1],
                "QMoEOffload FP4 weight shapes are inconsistent.");
    num_experts_ = static_cast<int>(fc1_shape[0]);
    hidden_size_ = static_cast<int>(fc1_shape[1]);
    inter_size_ = static_cast<int>(fc2_shape[1]);
    const bool layout_ok = mxfp4_mode_
                               ? qmoe_shared::MakeMxfp4ExpertBlockLayout(hidden_size_, inter_size_, &nvfp4_layout_)
                               : qmoe_shared::MakeNvfp4ExpertBlockLayout(hidden_size_, inter_size_, &nvfp4_layout_);
    ORT_ENFORCE(layout_ok, "QMoEOffload FP4 dimensions are invalid.");
    if (InputExists(info, 15)) {
      ORT_ENFORCE(info.TryGetConstantInput(15, &nvfp4_fc1_g_) && nvfp4_fc1_g_ &&
                      nvfp4_fc1_g_->Location().device.Type() == OrtDevice::CPU,
                  "QMoEOffload FP4 fc1_global_scale must be a constant CPU initializer.");
    } else {
      ORT_ENFORCE(mxfp4_mode_, "QMoEOffload NVFP4 requires fc1_global_scale at slot 15.");
    }
    if (InputExists(info, 16)) {
      ORT_ENFORCE(info.TryGetConstantInput(16, &nvfp4_fc2_g_) && nvfp4_fc2_g_ &&
                      nvfp4_fc2_g_->Location().device.Type() == OrtDevice::CPU,
                  "QMoEOffload FP4 fc2_global_scale must be a constant CPU initializer.");
    } else {
      ORT_ENFORCE(mxfp4_mode_, "QMoEOffload NVFP4 requires fc2_global_scale at slot 16.");
    }
    if (!nvfp4_fc1_g_ || !nvfp4_fc2_g_) {
      default_global_scales_.assign(static_cast<size_t>(num_experts_), 1.0f);
    }
    if (!mxfp4_mode_) {
      auto check_fast_scales = [this](const Tensor* g_tensor) -> bool {
        const float* scales = g_tensor ? g_tensor->Data<float>() : default_global_scales_.data();
        if (!scales) return false;
        for (int e = 0; e < num_experts_; ++e) {
          if (!onnxruntime::llm::kernels::moe_gemv::IsNvfp4DecodeFastScale(scales[e])) {
            return false;
          }
        }
        return true;
      };
      nvfp4_fc1_decode_scale_eligible_ = check_fast_scales(nvfp4_fc1_g_);
      nvfp4_fc2_decode_scale_eligible_ = check_fast_scales(nvfp4_fc2_g_);
    }
    if (InputExists(info, 4)) {
      ORT_ENFORCE(info.TryGetConstantInput(4, &nvfp4_fc1_b_) && nvfp4_fc1_b_ != nullptr,
                  "QMoEOffload NVFP4 fc1_bias must be constant.");
    }
    if (InputExists(info, 7)) {
      ORT_ENFORCE(info.TryGetConstantInput(7, &nvfp4_fc2_b_) && nvfp4_fc2_b_ != nullptr,
                  "QMoEOffload NVFP4 fc2_bias must be constant.");
      if (nvfp4_fc2_b_->Location().device.Type() == OrtDevice::CPU) {
        fc2_bias_gpu_bytes_ = nvfp4_fc2_b_->SizeInBytes();
        fc2_bias_gpu_ = GetTransientScratchBuffer<uint8_t>(fc2_bias_gpu_bytes_);
        CUDA_CALL_THROW(cudaMemcpy(fc2_bias_gpu_.get(), nvfp4_fc2_b_->DataRaw(), fc2_bias_gpu_bytes_,
                                   cudaMemcpyHostToDevice));
      }
    }
  }

  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload NVFP4 k must be in [1,E].");
  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto gpu_limit = v4_planned_mode_
                             ? static_cast<int64_t>(v4_policy_.device_working_bytes)
                             : info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{256} << 20);
  const auto pinned_limit = v4_planned_mode_
                                ? static_cast<int64_t>(v4_policy_.pinned_bytes)
                                : info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{32} << 20);
  token_tile_size_ = static_cast<int>(tile);
  max_sync_tile_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("max_sync_tile_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_SYNC_TILE_TOKENS", token_tile_size_)));
  async_max_tokens_ = v4_planned_mode_
                          ? static_cast<int>(v4_policy_.async_max_tokens)
                          : static_cast<int>(
                                info.GetAttrOrDefault<int64_t>("async_max_tokens",
                                                               ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ASYNC_MAX_TOKENS", 4)));
  stream_min_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("stream_min_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STREAM_MIN_TOKENS", token_tile_size_ + 1)));
  max_routes_ = SafeInt<int>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * k_;
  shared_gpu_limit_ = static_cast<size_t>(gpu_limit);
  shared_pinned_limit_ = static_cast<size_t>(pinned_limit);

  if (v4_planned_mode_) {
    deterministic_replay_ = false;
    resident_slots_ = std::clamp(static_cast<int>(v4_plan_.resident_slots), 0, num_experts_);
    base_resident_slots_ = resident_slots_;
    configured_decode_extension_slots_ = 0;
    decode_extension_slots_ = 0;
    spare_slots_ = std::clamp(static_cast<int>(v4_plan_.spare_slots), 0, 2);
    if (resident_slots_ == 0 || resident_slots_ == num_experts_) spare_slots_ = 0;
    dummy_slots_ = 0;
    physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
    dummy_slot_index_ = -1;
  } else {
    deterministic_replay_ =
        info.GetAttrOrDefault<int64_t>("deterministic_replay",
                                       ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_DETERMINISTIC_REPLAY", 0)) == 1;
    const int default_resident_slots = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_SLOTS", 0);
    resident_slots_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("resident_slots", default_resident_slots)), 0, num_experts_);
    const int resident_scale_pct = std::clamp(
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_RESIDENT_SCALE_PCT",
#if defined(_WIN32)
                                                 62
#else
                                                 100
#endif
                                                 ),
        10, 120);
    if (resident_scale_pct != 100 && resident_slots_ > 0 && resident_slots_ < num_experts_) {
      resident_slots_ = std::clamp((resident_slots_ * resident_scale_pct) / 100, 1, num_experts_);
    }
    const int max_resident_slots =
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_RESIDENT_SLOTS_PER_LAYER", num_experts_);
    if (max_resident_slots >= 0 && resident_slots_ > max_resident_slots) {
      resident_slots_ = max_resident_slots;
    }
    base_resident_slots_ = resident_slots_;
    const int default_ext_slots = 0;
    configured_decode_extension_slots_ = std::clamp(
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_DECODE_EXTENSION_SLOTS", default_ext_slots),
        0, std::max(0, num_experts_ - resident_slots_));
    decode_extension_slots_ = configured_decode_extension_slots_;
    resident_slots_ += decode_extension_slots_;
    const int default_spare_slots = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_SPARE_SLOTS", 0);
#if defined(_WIN32)
    if (std::getenv("ORT_QMOE_SHARED_SPARE_SLOTS") != nullptr) {
      spare_slots_ = std::clamp(default_spare_slots, 0, 16);
    } else {
      spare_slots_ = std::clamp(
          static_cast<int>(info.GetAttrOrDefault<int64_t>("spare_slots", 2)), 0, 2);
    }
#else
    if (std::getenv("ORT_QMOE_SHARED_SPARE_SLOTS") != nullptr) {
      spare_slots_ = std::clamp(default_spare_slots, 0, 16);
    } else {
      spare_slots_ = std::clamp(
          static_cast<int>(info.GetAttrOrDefault<int64_t>("spare_slots", default_spare_slots)), 0, 16);
    }
#endif
    if (resident_slots_ == 0 || resident_slots_ == num_experts_ || deterministic_replay_) spare_slots_ = 0;
    dummy_slots_ = 0;  // MoeGemvFp4DecodeKernel natively skips slot < 0 without reading weights!
    physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
    dummy_slot_index_ = -1;
  }
  last_cold_routes_.resize(num_experts_);
  expert_presence_ema_.assign(num_experts_, 0.0f);

  group_calls_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", 32), 1, 64);
  const int default_async = v4_planned_mode_
                                ? ((v4_plan_.enable_hybrid_cpu_decode || resident_slots_ == num_experts_) ? 1 : 0)
                                : ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_ASYNC", allow_cuda_graph_ ? 1 : 0);
  async_ = (default_async == 1) && (resident_slots_ > 0 || v4_planned_mode_ || info.GetAttrOrDefault<int64_t>("stream_rows", 0) == 0);
  pool_task_batch_ = deterministic_replay_ ? 1 : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_TASK_BATCH", 1), 1, 16);
  const int default_cpu_threads =
      std::clamp(qmoe::DiscoverPhysicalCoreCount(), 1, 64);
  cpu_threads_ = (v4_planned_mode_ && v4_policy_.cpu_threads > 0)
                     ? static_cast<int>(v4_policy_.cpu_threads)
                     : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", default_cpu_threads), 1, 256);
  unified_cpu_service_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_UNIFIED_CPU_SERVICE", 1) == 1;
  continuous_decay_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CONTINUOUS_DECAY", 1) == 1;
  bg_promo_stream_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_BG_PROMO_STREAM", 1) == 1;
  cpu_fp16_boundaries_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES", 0) == 1;
  const int default_stream_rows = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_ROWS", 0);
  if (v4_planned_mode_ &&
      v4_policy_.cpu_execution == onnxruntime::moe_offload::CpuExecutionMode::kDisabled &&
      resident_slots_ < num_experts_) {
    async_ = false;
    doorbell_enabled_ = false;
    stream_rows_ = std::max({token_tile_size_, max_sync_tile_tokens_, std::max(1, async_max_tokens_)});
    stream_min_tokens_ = 1;
    stream_min_rows_ = 1;
    stream_wave_ = std::clamp(static_cast<int>(v4_policy_.prefill_wave_experts), 1, num_experts_);
    stream_promotions_ = 0;
    stream_gpu_limit_ = shared_gpu_limit_;
    stream_pinned_limit_ = shared_pinned_limit_;
  } else {
    stream_rows_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_rows", default_stream_rows)), 0, 8192);
  }
  prefill_chunk_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("prefill_chunk_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_CHUNK_TOKENS", stream_rows_)));
  stream_promotion_reuse_ = !deterministic_replay_ && (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE", 0) == 1);
  if (stream_rows_ > 0 && !(v4_planned_mode_ &&
                            v4_policy_.cpu_execution == onnxruntime::moe_offload::CpuExecutionMode::kDisabled &&
                            resident_slots_ < num_experts_)) {
    const int default_min_rows = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_MIN_ROWS", 16);
    const int default_wave = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_WAVE", 32);
    stream_min_rows_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_min_rows", default_min_rows)), 1, 8192);
    stream_wave_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_wave", default_wave)), 1, num_experts_);
    stream_promotions_ = deterministic_replay_ ? 0 : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTIONS", 0), 0, 64);
    stream_gpu_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_GPU_MIB", 1024), 1)) << 20;
    stream_pinned_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PINNED_MIB", 128), 1)) << 20;
  }
  if (async_) {
    async_aux_ = resident_slots_ > 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_AUX_STREAM", 1) == 1;
    async_work_ = std::make_unique<CpuWork>(group_calls_, num_experts_, hidden_size_, inter_size_);
  }

  const size_t fp4_scale_group = mxfp4_mode_ ? 32 : 16;
  resident_bank_bytes_ = {
      static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 2)),
      static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / 2)),
      static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / fp4_scale_group)),
      static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / fp4_scale_group)),
      sizeof(float),
      sizeof(float)};

  if (host_cache_bytes_ > 0) InitializeBoundedHostCache();

#if defined(__linux__)
  const int default_pin_experts = (host_cache_bytes_ == 0 && resident_slots_ < num_experts_) ? 1 : 0;
#else
  const int default_pin_experts = 0;
#endif
  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", default_pin_experts) == 1) {
    auto pinned_budget = GetCudaHostPinnedMemoryBudget();
    size_t pinned_accum = 0;
    auto try_register = [this, &pinned_budget, &pinned_accum](const Tensor* t) {
      // Only register large (>= 1 MiB) tensors so small tensors sharing a 4 KiB OS page are never double-registered.
      if (!t || !t->DataRaw() || t->SizeInBytes() < (size_t{1} << 20)) return;
      if (v4_planned_mode_ && v4_policy_.pinned_bytes_explicit &&
          pinned_accum + t->SizeInBytes() > shared_pinned_limit_) {
        return;
      }
      for (const void* existing : registered_host_ptrs_) {
        if (existing == t->DataRaw()) return;
      }
      const bool first_owner = !pinned_budget || pinned_budget->RegisterHostRange(t->DataRaw(), t->SizeInBytes());
      if (!first_owner) {
        registered_host_ptrs_.push_back(t->DataRaw());
        experts_registered_ = true;
        pinned_accum += t->SizeInBytes();
        return;
      }
      cudaError_t reg_err = cudaHostRegister(
          const_cast<void*>(t->DataRaw()), t->SizeInBytes(),
          cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
      if (reg_err != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        reg_err = cudaHostRegister(
            const_cast<void*>(t->DataRaw()), t->SizeInBytes(),
            cudaHostRegisterPortable | cudaHostRegisterReadOnly);
      }
      if (reg_err == cudaSuccess) {
        registered_host_ptrs_.push_back(t->DataRaw());
        experts_registered_ = true;
        pinned_accum += t->SizeInBytes();
      } else {
        if (pinned_budget) static_cast<void>(pinned_budget->UnregisterHostRange(t->DataRaw(), true));
        static_cast<void>(cudaGetLastError());
      }
    };
    if (expert_blocks_) {
      try_register(expert_blocks_);
    } else {
      for (const Tensor* t : {nvfp4_fc1_w_, nvfp4_fc1_s_, nvfp4_fc2_w_, nvfp4_fc2_s_}) {
        try_register(t);
      }
    }
    // A single registered bank does not make every bank a valid pinned DMA source.
    const auto is_registered = [this](const Tensor* t) {
      return t != nullptr && std::find(registered_host_ptrs_.begin(), registered_host_ptrs_.end(), t->DataRaw()) !=
                                 registered_host_ptrs_.end();
    };
    all_expert_banks_registered_ =
        expert_blocks_ ? is_registered(expert_blocks_)
                       : (is_registered(nvfp4_fc1_w_) && is_registered(nvfp4_fc1_s_) &&
                          is_registered(nvfp4_fc2_w_) && is_registered(nvfp4_fc2_s_));
  }

  fc1_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC1_COLS", 64), 32, 1024) / 32 * 32;
  fc2_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC2_COLS", 128), 32, 1024) / 32 * 32;
  cpu_timing_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_TIMING", 0) == 1;

  if (resident_slots_ > 0) {
    if (v4_planned_mode_) {
      resident_promotions_ =
          (v4_plan_.enable_background_promotion && spare_slots_ > 0)
              ? std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", 2), 0, 64)
              : 0;
    } else {
#if defined(_WIN32)
      const int default_res_promos = (host_cache_bytes_ > 0 && allow_cuda_graph_) ? 8 : 0;
#else
      const int default_res_promos = allow_cuda_graph_ ? 8 : 0;
#endif
      resident_promotions_ = deterministic_replay_ ? 0 : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", default_res_promos), 0, 64);
    }
    if (resident_promotions_ == 0 && stream_promotions_ == 0) {
      spare_slots_ = 0;
      physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
    }
    resident_decay_routes_ = std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES", 4096), 1);
    resident_phase_scores_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES", 0) == 1;
    resident_margin_percent_ = std::clamp(
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT", 125), 100, 10000);
    marginal_row_weight_ = static_cast<float>(std::clamp(
                               ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MARGINAL_ROW_WEIGHT_PCT", 80), 0, 100)) /
                           100.0f;
    presence_ema_weight_ = static_cast<float>(std::clamp(
                               ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PRESENCE_EMA_WEIGHT_PCT", 200), 0, 10000)) /
                           100.0f;
    step_promo_budget_ = std::clamp(
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STEP_PROMOTION_BUDGET", 16), 0, 256);
    step_promo_bytes_ = static_cast<size_t>(std::clamp(
                            ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STEP_PROMOTION_MIB", 64), 0, 4096))
                        << 20;
    if (resident_promotions_ > 0 && !all_expert_banks_registered_ && !host_complement_mode_ &&
        !(file_expert_store_.IsInitialized() && host_cache_mapped_)) {
      if (!promo_budget_retained_) {
        SessionPromotionBudget::Instance().Retain(session_controller_.get());
        promo_budget_retained_ = true;
      }
      SessionPromotionBudget::Instance().EnsureBounceAllocated(session_controller_.get(), step_promo_bytes_);
    }
    resident_log_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_LOG", 0) == 1;

    size_t slab_bytes = 0;
    for (size_t bank = 0; bank < resident_bank_bytes_.size(); ++bank) {
      resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
    }
    resident_slab_bytes_ = slab_bytes;
    resident_slab_ = AllocateReservedDeviceSlab(slab_bytes);
    if (nvfp4_fc1_b_ != nullptr) {
      resident_fc1_bias_bytes_ = SafeInt<size_t>(physical_slots_) * 2 * inter_size_ * sizeof(uint16_t);
      resident_fc1_bias_slab_ = AllocateReservedDeviceSlab(resident_fc1_bias_bytes_);
      ORT_ENFORCE(resident_fc1_bias_slab_ != nullptr, "QMoEOffload failed to allocate resident FC1 bias slab.");
    }
    // Double-buffered device slot table: 2 epochs * num_experts_ (v3 Section 7.5 / F3).
    slot_table_dev_ = GetTransientScratchBuffer<uint8_t>(SafeInt<size_t>(2) * num_experts_ * sizeof(int));
    ORT_ENFORCE(resident_slab_ != nullptr && slot_table_dev_ != nullptr,
                "QMoEOffload NVFP4 failed to allocate resident slab of ", slab_bytes, " bytes.");
    slot_expert_.assign(physical_slots_, -1);
    expert_slot_.assign(num_experts_, -1);
    file_store_expert_slot_.assign(num_experts_, -1);
    expert_score_.assign(num_experts_, 0.0f);
    expert_presence_ema_.assign(num_experts_, 0.0f);
    if (resident_phase_scores_) decode_score_.assign(num_experts_, 0.0f);
    CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev() + num_experts_, file_store_expert_slot_.data(),
                               SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));

    if (v4_planned_mode_) {
      initial_experts_.resize(base_resident_slots_);
      for (int slot = 0; slot < base_resident_slots_; ++slot) {
        initial_experts_[slot] = (static_cast<size_t>(slot) < v4_plan_.initial_resident_experts.size())
                                     ? static_cast<int>(v4_plan_.initial_resident_experts[slot])
                                     : slot;
      }
      ranked_experts_.assign(v4_plan_.ranked_experts.begin(), v4_plan_.ranked_experts.end());
      if (ranked_experts_.size() < static_cast<size_t>(num_experts_)) {
        std::vector<uint8_t> seen_rank(static_cast<size_t>(num_experts_), 0);
        for (int e : ranked_experts_) {
          if (e >= 0 && e < num_experts_) seen_rank[static_cast<size_t>(e)] = 1;
        }
        for (int e = 0; e < num_experts_; ++e) {
          if (!seen_rank[static_cast<size_t>(e)]) ranked_experts_.push_back(e);
        }
      }
    } else {
      // Pre-populate initial resident slots [0, base_resident_slots_) from profile-seeded initial_experts or 0..C_base-1.
      std::vector<int64_t> initial_experts = info.GetAttrsOrDefault<int64_t>("initial_experts");
      initial_experts_.resize(base_resident_slots_);
      for (int slot = 0; slot < base_resident_slots_; ++slot) {
        initial_experts_[slot] = (static_cast<size_t>(slot) < initial_experts.size())
                                     ? static_cast<int>(initial_experts[slot])
                                     : slot;
      }
      ranked_experts_.clear();
      ranked_experts_.reserve(num_experts_);
      std::vector<uint8_t> seen_rank(static_cast<size_t>(num_experts_), 0);
      for (int64_t e64 : initial_experts) {
        const int e = static_cast<int>(e64);
        if (e >= 0 && e < num_experts_ && !seen_rank[static_cast<size_t>(e)]) {
          seen_rank[static_cast<size_t>(e)] = 1;
          ranked_experts_.push_back(e);
        }
      }
      if (ranked_experts_.size() < static_cast<size_t>(num_experts_) &&
          ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_USE_EXPERT_PROFILE", 1) == 1) {
        std::string prof_path = ParseEnvironmentVariableWithDefault<std::string>("ORT_QMOE_EXPERT_PROFILE", "");
        if (!prof_path.empty()) {
          std::ifstream pf(prof_path, std::ios::binary);
          uint32_t hdr[6]{};
          if (pf && pf.read(reinterpret_cast<char*>(hdr), sizeof(hdr))) {
            const uint32_t magic = hdr[0], ver = hdr[1], n_layers = hdr[2], n_exp = hdr[3], n_ranked = hdr[5];
            if (magic == 0x50525453u && ver == 1u && static_cast<int>(n_exp) == num_experts_ &&
                layer_index_ >= 0 && layer_index_ < static_cast<int64_t>(n_layers) &&
                n_ranked <= n_layers * n_exp) {
              std::vector<uint16_t> pairs(static_cast<size_t>(n_ranked) * 2);
              if (pf.read(reinterpret_cast<char*>(pairs.data()), static_cast<std::streamsize>(pairs.size() * sizeof(uint16_t)))) {
                for (uint32_t i = 0; i < n_ranked; ++i) {
                  const int l_id = static_cast<int>(pairs[static_cast<size_t>(i) * 2]);
                  const int e_id = static_cast<int>(pairs[static_cast<size_t>(i) * 2 + 1]);
                  if (l_id == static_cast<int>(layer_index_) && e_id >= 0 && e_id < num_experts_ &&
                      !seen_rank[static_cast<size_t>(e_id)]) {
                    seen_rank[static_cast<size_t>(e_id)] = 1;
                    ranked_experts_.push_back(e_id);
                  }
                }
              }
            }
          }
        }
      }
      for (int e = 0; e < num_experts_; ++e) {
        if (!seen_rank[static_cast<size_t>(e)]) {
          seen_rank[static_cast<size_t>(e)] = 1;
          ranked_experts_.push_back(e);
        }
      }
    }
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PREPOPULATE", 1) == 1) {
      const float init_ema = static_cast<float>(k_) / static_cast<float>(std::max(1, num_experts_));
      std::vector<int> init_slot_ids(base_resident_slots_);
      for (int slot = 0; slot < base_resident_slots_; ++slot) {
        const int expert = initial_experts_[slot];
        ORT_ENFORCE(expert >= 0 && expert < num_experts_ && expert_slot_[expert] < 0,
                    "QMoEOffload invalid or duplicate initial_experts entry ", expert, " at slot ", slot, ".");
        init_slot_ids[slot] = slot;
        slot_expert_[slot] = expert;
        expert_slot_[expert] = slot;
        const float prior = 4.0f + static_cast<float>(base_resident_slots_ - slot) * 0.25f;
        expert_score_[expert] = prior;
        expert_presence_ema_[expert] = init_ema;
        if (resident_phase_scores_) decode_score_[expert] = prior;
      }
      BatchCopyExpertsToResidentSlots(initial_experts_.data(), init_slot_ids.data(), base_resident_slots_, nullptr);
      CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev(), expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
      slot_table_dirty_ = false;
    } else {
      CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev(), expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
      slot_table_dirty_ = false;
    }
    if (host_cache_bytes_ > 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_PREWARM", 1) == 1) {
      std::vector<int32_t> cold_to_warm;
      const size_t target_warm = file_expert_store_.CacheSlots();
      cold_to_warm.reserve(target_warm);
      for (int e : ranked_experts_) {
        if (cold_to_warm.size() >= target_warm) break;
        if (expert_slot_[e] < 0) {
          cold_to_warm.push_back(static_cast<int32_t>(e));
        }
      }
      if (!cold_to_warm.empty()) {
        auto warm_fn = [this, &cold_to_warm](ptrdiff_t idx) noexcept {
          qmoe_shared::ExpertBlockLease lease{};
          const int32_t e = cold_to_warm[static_cast<size_t>(idx)];
          if (file_expert_store_.AcquireReady(e, &lease)) {
            file_expert_store_.ReleaseLease(e);
          }
        };
        SharedSpinPool(cpu_threads_).ParallelFor(static_cast<ptrdiff_t>(cold_to_warm.size()), warm_fn, 1);
        // Touch from coldest to hottest so FileExpertStore's LRU victim order evicts the coldest expert first.
        for (size_t i = cold_to_warm.size(); i-- > 0;) {
          qmoe_shared::ExpertBlockLease lease{};
          if (file_expert_store_.TryAcquireCached(cold_to_warm[i], &lease)) {
            file_expert_store_.ReleaseLease(cold_to_warm[i]);
          }
        }
      }
      file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
      CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev() + num_experts_, file_store_expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
      host_slot_table_dirty_ = false;
      file_expert_store_.ResetStats();
    }

    int dev_id = device_id_;
    owned_layer_cache_ = std::make_shared<SharedLayerCache>();
    owned_layer_cache_->device_id = dev_id;
    owned_layer_cache_->layer_index = layer_index_;
    owned_layer_cache_->num_experts = num_experts_;
    owned_layer_cache_->hidden_size = hidden_size_;
    owned_layer_cache_->inter_size = inter_size_;
    owned_layer_cache_->resident_slots = resident_slots_;
    owned_layer_cache_->physical_slots = physical_slots_;
    owned_layer_cache_->bank_bytes = resident_bank_bytes_;
    owned_layer_cache_->bank_offsets = resident_bank_offsets_;
    owned_layer_cache_->slab_ptr = resident_slab_.get();
    owned_layer_cache_->slot_table_dev_ptr = CurrentSlotTableDev();
    owned_layer_cache_->expert_slot_ptr = &expert_slot_;
    owned_layer_cache_->owner = this;
    owned_layer_cache_->session_scope = session_controller_.get();
    owned_layer_cache_->fc1_w_id = nvfp4_fc1_w_ ? static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) : nullptr;
    if (host_cache_bytes_ > 0 && file_expert_store_.IsInitialized()) {
      owned_layer_cache_->file_store_ptr = &file_expert_store_;
      owned_layer_cache_->nvfp4_layout = nvfp4_layout_;
      owned_layer_cache_->host_cache_mapped = host_cache_mapped_;
      if (file_expert_store_.CacheBase() != nullptr) {
        owned_layer_cache_->host_slot_ptr = &file_store_expert_slot_;
      }
    }
    SharedLayerCache::Register(owned_layer_cache_);
    if (nvfp4_mode_ && nvfp4_fc1_w_ && nvfp4_fc1_s_ && nvfp4_fc2_w_ && nvfp4_fc2_s_) {
      const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
      const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
      RegisterQMoEDenseLayerHostWeights(
          dev_id, layer_index_, num_experts_, hidden_size_, inter_size_,
          static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()),
          static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()),
          fc1_g_base,
          static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()),
          static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()),
          fc2_g_base);
      if (!v4_planned_mode_) {
        prefill_qmoe_ = std::make_unique<QMoE>(info);
      }
    }
  } else if (v4_planned_mode_ && nvfp4_mode_ && nvfp4_fc1_w_ && nvfp4_fc1_s_ && nvfp4_fc2_w_ && nvfp4_fc2_s_) {
    slot_table_dev_ = GetTransientScratchBuffer<uint8_t>(SafeInt<size_t>(2) * num_experts_ * sizeof(int));
    expert_slot_.assign(num_experts_, -1);
    file_store_expert_slot_.assign(num_experts_, -1);
    CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev(), expert_slot_.data(),
                               SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev() + num_experts_, file_store_expert_slot_.data(),
                               SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
    slot_table_dirty_ = false;
    int dev_id = device_id_;
    owned_layer_cache_ = std::make_shared<SharedLayerCache>();
    owned_layer_cache_->device_id = dev_id;
    owned_layer_cache_->layer_index = layer_index_;
    owned_layer_cache_->num_experts = num_experts_;
    owned_layer_cache_->hidden_size = hidden_size_;
    owned_layer_cache_->inter_size = inter_size_;
    owned_layer_cache_->resident_slots = 0;
    owned_layer_cache_->physical_slots = 0;
    owned_layer_cache_->bank_bytes = resident_bank_bytes_;
    owned_layer_cache_->bank_offsets = resident_bank_offsets_;
    owned_layer_cache_->slab_ptr = nullptr;
    owned_layer_cache_->slot_table_dev_ptr = CurrentSlotTableDev();
    owned_layer_cache_->expert_slot_ptr = &expert_slot_;
    owned_layer_cache_->owner = this;
    owned_layer_cache_->session_scope = session_controller_.get();
    owned_layer_cache_->fc1_w_id = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw());
    SharedLayerCache::Register(owned_layer_cache_);
    const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
    const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
    RegisterQMoEDenseLayerHostWeights(
        dev_id, layer_index_, num_experts_, hidden_size_, inter_size_,
        static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()),
        static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()),
        fc1_g_base,
        static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()),
        static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()),
        fc2_g_base);
  }

  if (stream_rows_ > 0) {
    for (auto& offsets : stream_bank_offsets_) {
      for (size_t bank = 0; bank < offsets.size(); ++bank) {
        offsets[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * stream_wave_, stream_landing_bytes_);
      }
    }
    if (nvfp4_fc1_b_ != nullptr) {
      const size_t wave_bias_bytes = SafeInt<size_t>(stream_wave_) * 2 * inter_size_ * sizeof(uint16_t);
      stream_fc1_bias_slab_[0] = AllocateReservedDeviceSlab(wave_bias_bytes);
      stream_fc1_bias_slab_[1] = AllocateReservedDeviceSlab(wave_bias_bytes);
      ORT_ENFORCE(stream_fc1_bias_slab_[0] != nullptr && stream_fc1_bias_slab_[1] != nullptr,
                  "QMoEOffload failed to allocate streaming FC1 bias slabs.");
    }
    stream_shared_ = session_controller_ ? session_controller_->LandingRing()
                                         : std::make_shared<onnxruntime::cuda::QMoESessionLandingRing>();
  }
  if (host_complement_mode_ && resident_slots_ > 0 && resident_slots_ < num_experts_) {
    InitializeComplementMode();
  }
  if (ssd_cache_slots_ > 0) {
    InitializeSsdStore();
  }
}

void QMoEOffload::InitializeShared(const OpKernelInfo& info) {
  ORT_ENFORCE(host_cache_bytes_ == 0, "Bounded host cache currently requires NVFP4 bank initializers.");
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                  info.GetAttrOrDefault<int64_t>("weights_prepacked", -1) == 1,
              "QMoEOffload shared mode requires layout_version=1 and weights_prepacked=1.");
  const auto layout_id = info.GetAttrOrDefault<std::string>("layout_id", qmoe_shared::kSharedLayoutId);
  ORT_ENFORCE(layout_id == qmoe_shared::kSharedLayoutId,
              "QMoEOffload INT4 shared mode requires layout_id='", qmoe_shared::kSharedLayoutId, "', got '", layout_id, "'.");

  int blk_slot = 21;
  if (v3_compact_shared_) {
    blk_slot = 2;
    for (int index = 5; index <= 25; ++index) {
      ORT_ENFORCE(!InputExists(info, index), "QMoEOffload v3 compact shared mode forbids input slot ", index, ".");
    }
  } else {
    for (int index = 2; index <= 20; ++index) {
      ORT_ENFORCE(!InputExists(info, index), "QMoEOffload shared and legacy expert inputs cannot be mixed.");
    }
    const bool optional_diag = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OPTIONAL_DIAGNOSTICS", 0) == 1;
    ORT_ENFORCE(InputExists(info, 24) && (optional_diag || InputExists(info, 25)),
                "QMoEOffload shared mode requires fc2_bias and token_phase.");
    ORT_ENFORCE(!InputExists(info, 22) && !InputExists(info, 23),
                "QMoEOffload shared zero-cache checkpoint does not yet support resident slabs.");
  }

  const auto hidden = info.GetAttrOrDefault<int64_t>("hidden_size", 0);
  const auto inter = info.GetAttrOrDefault<int64_t>("inter_size", 0);
  layer_index_ = info.GetAttrOrDefault<int64_t>("layer_index", v3_compact_shared_ ? 0 : -1);
  ORT_ENFORCE(layer_index_ >= 0 && hidden <= std::numeric_limits<int>::max() &&
                  inter <= std::numeric_limits<int>::max() / 2 &&
                  qmoe_shared::MakeExpertBlockLayout(hidden, inter, &shared_layout_),
              "QMoEOffload shared mode has invalid dimensions or layer_index.");
  ORT_ENFORCE(info.TryGetConstantInput(blk_slot, &expert_blocks_) && expert_blocks_ &&
                  expert_blocks_->IsDataType<uint8_t>() &&
                  expert_blocks_->Location().device.Type() == OrtDevice::CPU &&
                  expert_blocks_->Location().device.MemType() == OrtDevice::MemType::DEFAULT &&
                  expert_blocks_->Shape().NumDimensions() == 2 &&
                  expert_blocks_->Shape()[0] > 0 &&
                  expert_blocks_->Shape()[0] <= std::numeric_limits<int>::max() &&
                  expert_blocks_->Shape()[1] == static_cast<int64_t>(shared_layout_.stride) &&
                  expert_store_.Initialize(expert_blocks_->DataRaw(), expert_blocks_->SizeInBytes(),
                                           expert_blocks_->Shape()[0], shared_layout_.stride, alignof(uint16_t)) &&
                  qmoe_shared::MakeExpertBlockLease(expert_blocks_->DataRaw(), expert_blocks_->SizeInBytes(),
                                                    alignof(uint16_t), &expert_lease_),
              "QMoEOffload expert_blocks must be immutable pageable CPU uint8 [E,stride].");
  expert_lease_ = expert_store_.FullStoreLease();
  num_experts_ = static_cast<int>(expert_blocks_->Shape()[0]);
  hidden_size_ = static_cast<int>(hidden);
  inter_size_ = static_cast<int>(inter);
  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload shared k must be in [1,E].");

  const int bias_slot = v3_compact_shared_ ? 3 : 24;
  if (InputExists(info, bias_slot)) {
    const Tensor* cpu_fc2_bias = nullptr;
    if (info.TryGetConstantInput(bias_slot, &cpu_fc2_bias) && cpu_fc2_bias != nullptr &&
        cpu_fc2_bias->Location().device.Type() == OrtDevice::CPU) {
      const size_t row_bytes = SafeInt<size_t>(hidden_size_) * sizeof(uint16_t);
      ORT_ENFORCE(cpu_fc2_bias->SizeInBytes() == SafeInt<size_t>(num_experts_) * row_bytes,
                  "QMoEOffload shared fc2_bias size mismatch.");
      // Validate that in-block fc2_bias either matches separate fc2_bias or is zeroed (v3 Section 6).
      const uint8_t* blk_base = expert_blocks_->Data<uint8_t>();
      const uint8_t* bias_base = static_cast<const uint8_t*>(cpu_fc2_bias->DataRaw());
      for (int e = 0; e < num_experts_; ++e) {
        const size_t blk_off = static_cast<size_t>(SafeInt<size_t>(e) * shared_layout_.stride);
        const size_t sep_off = static_cast<size_t>(SafeInt<size_t>(e) * row_bytes);
        const uint8_t* in_blk = blk_base + blk_off + shared_layout_.fc2_bias;
        const uint8_t* sep_row = bias_base + sep_off;
        if (std::memcmp(in_blk, sep_row, row_bytes) != 0) {
          bool all_zero = true;
          for (size_t b = 0; b < row_bytes; ++b) {
            if (in_blk[b] != 0) {
              all_zero = false;
              break;
            }
          }
          ORT_ENFORCE(all_zero,
                      "QMoEOffload shared in-block fc2_bias disagrees with separate fc2_bias input at expert ", e, ".");
        }
      }
      if (v3_compact_shared_) {
        fc2_bias_gpu_bytes_ = cpu_fc2_bias->SizeInBytes();
        fc2_bias_gpu_ = GetTransientScratchBuffer<uint8_t>(fc2_bias_gpu_bytes_);
        CUDA_CALL_THROW(cudaMemcpy(fc2_bias_gpu_.get(), cpu_fc2_bias->DataRaw(), fc2_bias_gpu_bytes_,
                                   cudaMemcpyHostToDevice));
      }
    }
  }

  deterministic_replay_ =
      info.GetAttrOrDefault<int64_t>("deterministic_replay",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_DETERMINISTIC_REPLAY", 0)) == 1;
  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto gpu_limit = info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{64} << 20);
  const auto pinned_limit = info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{8} << 20);
  ORT_ENFORCE(tile > 0 && tile <= std::numeric_limits<int>::max() && gpu_limit > 0 && pinned_limit > 0,
              "QMoEOffload shared tile and memory limits must be positive.");
  token_tile_size_ = static_cast<int>(tile);
  max_sync_tile_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("max_sync_tile_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_SYNC_TILE_TOKENS", token_tile_size_)));
  async_max_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("async_max_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ASYNC_MAX_TOKENS", token_tile_size_)));
  stream_min_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("stream_min_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STREAM_MIN_TOKENS", token_tile_size_ + 1)));
  max_routes_ = SafeInt<int>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * k_;
  ORT_ENFORCE(SafeInt<int64_t>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * num_experts_ <= std::numeric_limits<int>::max(),
              "QMoEOffload shared tile routing exceeds integer limits.");
  shared_gpu_limit_ = static_cast<size_t>(gpu_limit);
  shared_pinned_limit_ = static_cast<size_t>(pinned_limit);

  const int default_res_slots = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_SLOTS", 0);
  resident_slots_ = std::clamp(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("resident_slots", default_res_slots)), 0, num_experts_);
  spare_slots_ = std::clamp(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("spare_slots",
                                                      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_SPARE_SLOTS", 0))),
      0, 16);
  if (resident_slots_ == 0 || resident_slots_ == num_experts_ || deterministic_replay_) spare_slots_ = 0;
  const bool use_dummy_slot = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_DUMMY_SLOT", 1) == 1;
  dummy_slots_ = (resident_slots_ > 0 && use_dummy_slot) ? 1 : 0;
  physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
  dummy_slot_index_ = dummy_slots_ > 0 ? (resident_slots_ + spare_slots_) : 0;
  last_cold_routes_.resize(num_experts_);
  expert_presence_ema_.assign(num_experts_, 0.0f);

  group_calls_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", 1), 1, 64);
  async_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_ASYNC", 0) == 1;
  pool_task_batch_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_TASK_BATCH", 1);
  ORT_ENFORCE(pool_task_batch_ >= 1 && pool_task_batch_ <= 16,
              "ORT_QMOE_SHARED_POOL_TASK_BATCH must be in [1,16].");
  ORT_ENFORCE(pool_task_batch_ == 1 || async_,
              "ORT_QMOE_SHARED_POOL_TASK_BATCH>1 requires shared async CPU experts.");
  if (deterministic_replay_) pool_task_batch_ = 1;
  cpu_fp16_boundaries_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES", 0) == 1;
  unified_cpu_service_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_UNIFIED_CPU_SERVICE", 0) == 1;
  continuous_decay_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CONTINUOUS_DECAY", 0) == 1;
  bg_promo_stream_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_BG_PROMO_STREAM", 1) == 1;
  stream_rows_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_ROWS", 0), 0, 8192);
  prefill_chunk_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("prefill_chunk_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_CHUNK_TOKENS", stream_rows_)));
  const int promotion_reuse = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE", 0);
  ORT_ENFORCE(promotion_reuse == 0 || promotion_reuse == 1,
              "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE must be 0 or 1.");
  stream_promotion_reuse_ = !deterministic_replay_ && (promotion_reuse == 1);
  ORT_ENFORCE(promotion_reuse == 0 || stream_rows_ > 0,
              "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE=1 requires shared prefill streaming.");
  if (stream_rows_ > 0) {
    ORT_ENFORCE(stream_rows_ > token_tile_size_ && resident_slots_ > 0 &&
                    ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 1,
                "ORT_QMOE_SHARED_STREAM_ROWS needs a value above the token tile, resident slots and PIN_EXPERTS=1.");
    stream_min_rows_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_MIN_ROWS", 16), 1, 8192);
    stream_wave_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_WAVE", 32), 1, num_experts_);
    stream_promotions_ = deterministic_replay_ ? 0 : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTIONS", 16), 1, 64);
    stream_gpu_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_GPU_MIB", 1024), 1)) << 20;
    stream_pinned_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PINNED_MIB", 128), 1)) << 20;
  }
  if (async_) {
    ORT_ENFORCE(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PINNED_REUSE", 0) == 1,
                "ORT_QMOE_SHARED_ASYNC=1 requires ORT_QMOE_SHARED_PINNED_REUSE=1.");
    cpu_threads_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", 40), 1, 256);
    async_aux_ = resident_slots_ > 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_AUX_STREAM", 0) == 1;
    async_work_ = std::make_unique<CpuWork>(group_calls_, num_experts_, hidden_size_, inter_size_);
  }
  callback_timing_ = qmoe::CallbackTiming::FromEnvironment(layer_index_);
  ORT_ENFORCE(!callback_timing_ || (async_ && async_aux_),
              "QMoE callback trace requires shared async CPU + resident auxiliary stream.");
  if (resident_slots_ == 0) {
    if (ssd_cache_slots_ > 0) InitializeSsdStore();
    return;
  }
  resident_promotions_ = deterministic_replay_ ? 0 : std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", 2), 1, 64);
  resident_decay_routes_ = std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES", 4096), 1);
  resident_phase_scores_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES", 0) == 1;
  resident_margin_percent_ = std::clamp(
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT", 100), 100, 10000);
  resident_log_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_LOG", 1) == 1;
  cpu_timing_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_TIMING", 0) == 1;
  fc1_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC1_COLS", 64), 32, 1024) / 32 * 32;
  fc2_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC2_COLS", 128), 32, 1024) / 32 * 32;
  resident_bank_bytes_ = {static_cast<size_t>(SafeInt<size_t>(hidden_size_) * inter_size_),
                          static_cast<size_t>(SafeInt<size_t>(inter_size_) * hidden_size_ / 2),
                          static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 32) * sizeof(MLFloat16)),
                          static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / 32) * sizeof(MLFloat16)),
                          static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(MLFloat16)),
                          0};
  ORT_ENFORCE(resident_bank_bytes_[0] <= shared_layout_.fc1_scales - shared_layout_.fc1_weights &&
                  resident_bank_bytes_[2] <= shared_layout_.fc1_bias - shared_layout_.fc1_scales &&
                  resident_bank_bytes_[1] <= shared_layout_.fc2_scales - shared_layout_.fc2_weights &&
                  resident_bank_bytes_[3] <= shared_layout_.fc2_bias - shared_layout_.fc2_scales &&
                  resident_bank_bytes_[4] <= shared_layout_.fc2_weights - shared_layout_.fc1_bias,
              "QMoEOffload shared block banks are smaller than the resident runner layout.");
  size_t slab_bytes = 0;
  for (size_t bank = 0; bank < 5; ++bank) {
    resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
  }
  runner_ = std::make_unique<ck::CutlassMoeFCRunner<half, cutlass::uint4b_t>>(sm_, activation_type_, false, false);
  ORT_ENFORCE(!runner_->getTactics().empty(), "QMoEOffload has no compatible SM80 INT4 GEMM tactic.");
  const ck::MOEParallelismConfig parallelism;
  const int route_bound = std::max(kGroupedMoeMinRows, max_routes_);
  for (int route_rows = kGroupedMoeMinRows; route_rows <= route_bound; ++route_rows) {
    resident_workspace_bytes_ = std::max(
        resident_workspace_bytes_,
        runner_->getWorkspaceSize(route_rows, hidden_size_, inter_size_, physical_slots_, 1, activation_type_,
                                  parallelism, true, swiglu_fusion_));
  }
  if (stream_rows_ > 0) {
    const int stream_route_bound = SafeInt<int>(stream_rows_) * static_cast<int>(k_);
    const int workspace_experts = std::max(physical_slots_, stream_wave_);
    stream_workspace_bytes_ = resident_workspace_bytes_;
    for (int route_rows = kGroupedMoeMinRows; route_rows <= stream_route_bound; route_rows += route_rows < 1024 ? 1 : 16) {
      stream_workspace_bytes_ = std::max(
          stream_workspace_bytes_,
          runner_->getWorkspaceSize(route_rows, hidden_size_, inter_size_, workspace_experts, 1, activation_type_,
                                    parallelism, true, swiglu_fusion_));
    }
    stream_workspace_bytes_ = std::max(
        stream_workspace_bytes_,
        runner_->getWorkspaceSize(stream_route_bound, hidden_size_, inter_size_, workspace_experts, 1, activation_type_,
                                  parallelism, true, swiglu_fusion_));
  }
  resident_slab_ = GetTransientScratchBuffer<uint8_t>(slab_bytes);
  if (dummy_slots_ > 0 && resident_slab_) {
    // Initialize the dedicated immutable dummy slot (C + S) to zero once so padding/miss reads never read uninitialized memory (F3).
    uint8_t* slab = resident_slab_.get();
    for (size_t bank = 0; bank < 5; ++bank) {
      const size_t off = resident_bank_offsets_[bank] + SafeInt<size_t>(dummy_slot_index_) * resident_bank_bytes_[bank];
      CUDA_CALL_THROW(cudaMemset(slab + off, 0, resident_bank_bytes_[bank]));
    }
  }
  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 1) {
    auto pinned_budget = GetCudaHostPinnedMemoryBudget();
    const bool first_owner =
        !pinned_budget || pinned_budget->RegisterHostRange(expert_blocks_->DataRaw(), expert_blocks_->SizeInBytes());
    if (first_owner) {
      const auto registered = cudaHostRegister(const_cast<void*>(expert_blocks_->DataRaw()), expert_blocks_->SizeInBytes(),
                                               cudaHostRegisterPortable | cudaHostRegisterReadOnly);
      if (registered != cudaSuccess) {
        if (pinned_budget) static_cast<void>(pinned_budget->UnregisterHostRange(expert_blocks_->DataRaw(), true));
        ORT_THROW("QMoEOffload cannot pin its CPU expert blocks: ", cudaGetErrorString(registered));
      }
    }
    registered_host_ptrs_.push_back(expert_blocks_->DataRaw());
    experts_registered_ = true;
    all_expert_banks_registered_ = true;
  }
  // Double-buffered device slot table: 2 epochs * num_experts_ (v3 Section 7.5 / F3).
  slot_table_dev_ = GetTransientScratchBuffer<uint8_t>(SafeInt<size_t>(2) * num_experts_ * sizeof(int));
  ORT_ENFORCE(slot_table_dev_ != nullptr, "QMoEOffload cannot allocate its device slot table.");
  ORT_ENFORCE(resident_slab_ != nullptr, "QMoEOffload cannot allocate its resident expert slab of ", slab_bytes, " bytes.");
  slot_expert_.assign(physical_slots_, -1);
  expert_slot_.assign(num_experts_, -1);
  expert_score_.assign(num_experts_, 0.0f);
  if (resident_phase_scores_) decode_score_.assign(num_experts_, 0.0f);

  std::vector<int64_t> initial_experts = info.GetAttrsOrDefault<int64_t>("initial_experts");
  if (!initial_experts.empty()) {
    const int populate_count = std::min(resident_slots_, static_cast<int>(initial_experts.size()));
    for (int slot = 0; slot < populate_count; ++slot) {
      const int expert = static_cast<int>(initial_experts[slot]);
      ORT_ENFORCE(expert >= 0 && expert < num_experts_ && expert_slot_[expert] < 0,
                  "QMoEOffload invalid or duplicate initial_experts entry ", expert, " at slot ", slot, ".");
      const size_t block_offset = SafeInt<size_t>(expert) * shared_layout_.stride;
      const uint8_t* block = expert_blocks_->Data<uint8_t>() + block_offset;
      const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                          shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
      for (size_t bank = 0; bank < sources.size(); ++bank) {
        const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(slot) * resident_bank_bytes_[bank];
        CUDA_CALL_THROW(cudaMemcpy(resident_slab_.get() + slot_offset, block + sources[bank],
                                   resident_bank_bytes_[bank], cudaMemcpyHostToDevice));
      }
      slot_expert_[slot] = expert;
      expert_slot_[expert] = slot;
      const float prior = 4.0f + static_cast<float>(resident_slots_ - slot) * 0.25f;
      expert_score_[expert] = prior;
      if (resident_phase_scores_) decode_score_[expert] = prior;
    }
    CUDA_CALL_THROW(cudaMemcpy(CurrentSlotTableDev(), expert_slot_.data(),
                               SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
    slot_table_dirty_ = false;
  }
  if (stream_rows_ > 0) {
    for (auto& offsets : stream_bank_offsets_) {
      for (size_t bank = 0; bank < 5; ++bank) {
        offsets[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * stream_wave_, stream_landing_bytes_);
      }
    }
    stream_shared_ = session_controller_ ? session_controller_->LandingRing()
                                         : std::make_shared<onnxruntime::cuda::QMoESessionLandingRing>();
  }
  if (host_complement_mode_ && resident_slots_ > 0 && resident_slots_ < num_experts_) {
    InitializeComplementMode();
  }
  if (ssd_cache_slots_ > 0) {
    InitializeSsdStore();
  }
}

void QMoEOffload::InitializeComplementMode() {
  complement_stride_ = nvfp4_mode_ ? nvfp4_layout_.stride : shared_layout_.stride;
  ORT_ENFORCE(complement_stride_ > 0, "QMoEOffload complement mode requires valid expert block stride.");
  complement_slots_ = (num_experts_ - resident_slots_) + spare_slots_ + 2;
  const size_t total_comp_bytes = SafeInt<size_t>(complement_slots_) * complement_stride_;
  complement_host_slab_ = AllocateBufferOnCPUPinned<uint8_t>(total_comp_bytes);
  ORT_ENFORCE(complement_host_slab_ != nullptr,
              "QMoEOffload failed to allocate complement host slab of ", total_comp_bytes, " bytes.");
  expert_host_slot_.assign(num_experts_, -1);
  host_slot_expert_.assign(complement_slots_, -1);
  int next_host_slot = 0;
  for (int e = 0; e < num_experts_; ++e) {
    if (expert_slot_[e] >= 0) continue;
    ORT_ENFORCE(next_host_slot < complement_slots_, "QMoEOffload complement host slots exhausted during init.");
    uint8_t* dst = complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(next_host_slot) * complement_stride_);
    if (expert_blocks_) {
      const uint8_t* src = expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(e) * complement_stride_);
      std::memcpy(dst, src, complement_stride_);
    } else if (nvfp4_mode_) {
      std::memset(dst, 0, complement_stride_);
      const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
      const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
      std::memcpy(dst + nvfp4_layout_.fc1_weights,
                  static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[0]),
                  resident_bank_bytes_[0]);
      std::memcpy(dst + nvfp4_layout_.fc2_weights,
                  static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[1]),
                  resident_bank_bytes_[1]);
      std::memcpy(dst + nvfp4_layout_.fc1_scales,
                  static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[2]),
                  resident_bank_bytes_[2]);
      std::memcpy(dst + nvfp4_layout_.fc2_scales,
                  static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[3]),
                  resident_bank_bytes_[3]);
      std::memcpy(dst + nvfp4_layout_.fc1_global_scale, fc1_g_base + e, sizeof(float));
      std::memcpy(dst + nvfp4_layout_.fc2_global_scale, fc2_g_base + e, sizeof(float));
    }
    expert_host_slot_[e] = next_host_slot;
    host_slot_expert_[next_host_slot] = e;
    ++next_host_slot;
  }
}

void QMoEOffload::InitializeBoundedHostCache() {
  ORT_ENFORCE(nvfp4_mode_ && !expert_blocks_ && (mxfp4_mode_ || (nvfp4_fc1_g_ && nvfp4_fc2_g_)) &&
                  (stream_rows_ == 0 || v4_planned_mode_),
              "Bounded host cache requires file-backed FP4 banks, valid global scales, and no shared streaming.");
  // A deterministic per-layer share is independent of construction order and other sessions.
  // Unused shares require a session-owned planner before they can safely be redistributed.
  const size_t max_cold_slots = static_cast<size_t>(std::max(1, num_experts_ - base_resident_slots_));
  const size_t slots = std::min(max_cold_slots, host_cache_bytes_ / nvfp4_layout_.stride);
  ORT_ENFORCE(slots >= 1, "Host cache budget is too small for one expert per layer (", nvfp4_layout_.stride, " bytes).");
  const void* fc1_g_ptr = nvfp4_fc1_g_ ? nvfp4_fc1_g_->DataRaw() : default_global_scales_.data();
  const void* fc2_g_ptr = nvfp4_fc2_g_ ? nvfp4_fc2_g_->DataRaw() : default_global_scales_.data();
  const qmoe_shared::FileExpertStore::MappedBank banks[]{
      {nvfp4_fc1_w_->DataRaw(), resident_bank_bytes_[0], nvfp4_layout_.fc1_weights},
      {nvfp4_fc2_w_->DataRaw(), resident_bank_bytes_[1], nvfp4_layout_.fc2_weights},
      {nvfp4_fc1_s_->DataRaw(), resident_bank_bytes_[2], nvfp4_layout_.fc1_scales},
      {nvfp4_fc2_s_->DataRaw(), resident_bank_bytes_[3], nvfp4_layout_.fc2_scales},
      {fc1_g_ptr, sizeof(float), nvfp4_layout_.fc1_global_scale},
      {fc2_g_ptr, sizeof(float), nvfp4_layout_.fc2_global_scale}};
  ORT_ENFORCE(file_expert_store_.InitializeMapped(banks, std::size(banks), num_experts_, nvfp4_layout_.stride, slots),
              "Bounded host cache requires immutable regular-file-backed initializers; cannot resolve source banks.");
  if (uint8_t* cache_base = file_expert_store_.CacheBase()) {
    const size_t cache_bytes = file_expert_store_.CacheBytes();
    const size_t reg_bytes = (cache_bytes + size_t{4095}) & ~size_t{4095};
    const bool allow_pin_by_budget =
        !(v4_planned_mode_ && v4_policy_.pinned_bytes_explicit);
    if (allow_pin_by_budget && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PIN_HOST_CACHE", 1) == 1) {
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      bool registered_in_ledger = false;
      try {
        registered_in_ledger = !pinned_budget || pinned_budget->RegisterHostRange(cache_base, reg_bytes);
      } catch (...) {
        registered_in_ledger = false;
      }
      if (cudaHostRegister(cache_base, reg_bytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
        registered_host_ptrs_.push_back(cache_base);
        experts_registered_ = true;
        host_cache_mapped_ = true;
      } else {
        if (pinned_budget && registered_in_ledger) {
          static_cast<void>(pinned_budget->UnregisterHostRange(cache_base, true));
        }
        static_cast<void>(cudaGetLastError());
      }
    }
  }
}

bool QMoEOffload::SwapHostCandidateForVictim(int candidate, int evicted, int victim_slab_slot,
                                             cudaStream_t stream) const {
  if (!nvfp4_mode_ || host_cache_bytes_ == 0 || !resident_slab_ || evicted < 0 || victim_slab_slot < 0 ||
      !swap_victim_to_host_cache_) {
    return false;
  }
  uint8_t* dst = file_expert_store_.BeginReplace(candidate, evicted);
  if (dst == nullptr) {
    ++victim_swap_fallbacks_;
    return false;
  }
  const std::array<size_t, 6> dst_offs{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                       nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                       nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
  const uint8_t* slab = resident_slab_.get();
  bool ok = true;
  for (size_t bank = 0; bank < dst_offs.size() && ok; ++bank) {
    const size_t src_off = resident_bank_offsets_[bank] +
                           static_cast<size_t>(SafeInt<size_t>(victim_slab_slot) * resident_bank_bytes_[bank]);
    ok = cudaMemcpyAsync(dst + dst_offs[bank], slab + src_off, resident_bank_bytes_[bank],
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  }
  ok = (cudaStreamSynchronize(stream) == cudaSuccess) && ok;
  file_expert_store_.FinishReplace(evicted, ok);
  host_slot_table_dirty_ = true;
  ++(ok ? victim_swaps_ : victim_swap_fallbacks_);
  return ok;
}

void QMoEOffload::InitializeSsdStore() {
  const size_t stride = nvfp4_mode_ ? nvfp4_layout_.stride : shared_layout_.stride;
  ORT_ENFORCE(stride > 0 && ssd_cache_slots_ > 0, "QMoEOffload SSD store requires positive stride and cache slots.");
  std::string path = ssd_store_path_;
  if ((path.empty() || path == "auto") && v4_planned_mode_ && !v4_policy_.derived_cache_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(v4_policy_.derived_cache_dir, ec);
    char fname[128];
    std::snprintf(fname, sizeof(fname), "ort_qmoe_ssd_%lld_%p.bin",
                  static_cast<long long>(layer_index_), static_cast<const void*>(this));
    path = (v4_policy_.derived_cache_dir / fname).string();
    ssd_temp_file_ = path;
  } else if (path.empty() || path == "auto") {
    char buf[256];
#if defined(_WIN32)
    const char* tmp_dir = std::getenv("TEMP");
    if (!tmp_dir || !*tmp_dir) tmp_dir = std::getenv("TMP");
    if (!tmp_dir || !*tmp_dir) tmp_dir = ".";
    std::snprintf(buf, sizeof(buf), "%s\\ort_qmoe_ssd_%lld_%p.bin",
                  tmp_dir, static_cast<long long>(layer_index_), static_cast<const void*>(this));
#else
    std::snprintf(buf, sizeof(buf), "/tmp/ort_qmoe_ssd_%lld_%p.bin",
                  static_cast<long long>(layer_index_), static_cast<const void*>(this));
#endif
    path = buf;
    ssd_temp_file_ = path;
  }
  bool need_write = false;
  {
    std::ifstream check_in(path, std::ios::binary | std::ios::ate);
    const size_t expected_bytes = SafeInt<size_t>(num_experts_) * stride;
    if (!check_in || static_cast<size_t>(check_in.tellg()) < expected_bytes) {
      need_write = true;
    }
  }
  if (need_write) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ORT_ENFORCE(static_cast<bool>(out), "QMoEOffload failed to create SSD backing file: ", path);
    if (expert_blocks_) {
      out.write(reinterpret_cast<const char*>(expert_blocks_->DataRaw()),
                static_cast<std::streamsize>(SafeInt<size_t>(num_experts_) * stride));
    } else if (nvfp4_mode_) {
      std::vector<uint8_t> blk(stride, 0);
      const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
      const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
      for (int e = 0; e < num_experts_; ++e) {
        std::memcpy(blk.data() + nvfp4_layout_.fc1_weights,
                    static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[0]),
                    resident_bank_bytes_[0]);
        std::memcpy(blk.data() + nvfp4_layout_.fc2_weights,
                    static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[1]),
                    resident_bank_bytes_[1]);
        std::memcpy(blk.data() + nvfp4_layout_.fc1_scales,
                    static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[2]),
                    resident_bank_bytes_[2]);
        std::memcpy(blk.data() + nvfp4_layout_.fc2_scales,
                    static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(e) * resident_bank_bytes_[3]),
                    resident_bank_bytes_[3]);
        std::memcpy(blk.data() + nvfp4_layout_.fc1_global_scale, fc1_g_base + e, sizeof(float));
        std::memcpy(blk.data() + nvfp4_layout_.fc2_global_scale, fc2_g_base + e, sizeof(float));
        out.write(reinterpret_cast<const char*>(blk.data()), static_cast<std::streamsize>(stride));
      }
    }
    out.flush();
    ORT_ENFORCE(static_cast<bool>(out), "QMoEOffload failed to write SSD backing file: ", path);
  }
  ORT_ENFORCE(file_expert_store_.Initialize(path.c_str(), 0, num_experts_, stride,
                                            static_cast<size_t>(ssd_cache_slots_), alignof(uint16_t)),
              "QMoEOffload failed to initialize FileExpertStore from ", path);
}

Status QMoEOffload::EnsurePersistentDecodeState(OpKernelContext* context, size_t min_gpu_bytes,
                                                size_t min_host_bytes, cudaStream_t stream) const {
  const size_t want_gpu = std::max(min_gpu_bytes, size_t{256} << 10) + 512;
  const size_t raw_slot_bytes = static_cast<size_t>(SafeInt<size_t>(num_experts_) * sizeof(int));
  const size_t slot_table_bytes = (raw_slot_bytes + 255) & ~size_t{255};
  const size_t want_host = std::max(min_host_bytes, size_t{256} << 10) + 512 + slot_table_bytes;
  if (!decode_state_) {
    decode_state_ = std::make_unique<PersistentDecodeState>();
  }
  if (!decode_state_->scratch || decode_state_->scratch_bytes < want_gpu) {
    decode_state_->scratch = GetScratchBuffer<uint8_t>(want_gpu, GetComputeStream(context));
    ORT_RETURN_IF_NOT(decode_state_->scratch != nullptr, "QMoEOffload failed to allocate persistent decode GPU scratch.");
    decode_state_->scratch_bytes = want_gpu;
    const size_t seq_off = ((want_gpu - 256) + 255) & ~size_t{255};
    decode_state_->doorbell_seq_dev = reinterpret_cast<uint64_t*>(decode_state_->scratch.get() + seq_off);
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(decode_state_->doorbell_seq_dev, 0, sizeof(uint64_t), stream));
  }
  if (!decode_state_->pinned || decode_state_->pinned_bytes < want_host) {
    ORT_RETURN_IF(want_host > shared_pinned_limit_,
                  "QMoEOffload persistent decode pinned buffer (", want_host,
                  " bytes) exceeds configured pinned_bytes limit (", shared_pinned_limit_, " bytes).");
    decode_state_->ReleasePinned(true);
    auto sess_pinned = session_controller_ ? session_controller_->PinnedBudget() : nullptr;
    auto host_ledger = GetCudaHostPinnedMemoryBudget();
    CudaDeviceMemoryBudget::Reservation sess_res;
    CudaDeviceMemoryBudget::Reservation host_res;
    try {
      if (sess_pinned) sess_res = sess_pinned->Reserve(want_host);
      if (host_ledger) host_res = host_ledger->ReserveAllocated(want_host);
    } catch (const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "QMoEOffload persistent decode pinned budget exhausted (requested ", want_host,
                             " bytes): ", ex.what());
    }
    decode_state_->pinned = AllocateBufferOnCPUPinned<uint8_t>(want_host);
    ORT_RETURN_IF_NOT(decode_state_->pinned != nullptr, "QMoEOffload failed to allocate persistent decode pinned buffer.");
    if (sess_res.Active()) sess_res.Commit();
    if (host_res.Active()) host_res.Commit();
    decode_state_->pinned_bytes = want_host;
    decode_state_->pinned_accounted_bytes = want_host;
    decode_state_->session_pinned_budget = std::move(sess_pinned);
    std::memset(decode_state_->pinned.get(), 0, want_host);
    const size_t meta_off = ((want_host - 256 - slot_table_bytes) + 255) & ~size_t{255};
    uint8_t* meta_ptr = decode_state_->pinned.get() + meta_off;
    decode_state_->doorbell_req_host = reinterpret_cast<volatile uint64_t*>(meta_ptr + 0);
    decode_state_->doorbell_done_host = reinterpret_cast<volatile uint64_t*>(meta_ptr + 8);
    decode_state_->doorbell_tile_rows_host = reinterpret_cast<volatile int32_t*>(meta_ptr + 16);
    decode_state_->pinned_slot_table = reinterpret_cast<int32_t*>(meta_ptr + 256);
    decode_state_->last_serviced_seq = 0;
  }
  if (!decode_state_->completed) {
    decode_state_->completed = std::make_shared<bool>(true);
  }
  if (!decode_state_->invocation_rec) {
    decode_state_->invocation_rec = std::make_shared<onnxruntime::cuda::QMoEInvocationRecord>();
    decode_state_->invocation_rec->layer_index = layer_index_;
    decode_state_->invocation_rec->node_name = Info().node().Name();
  }
  if (!decode_state_->tile) {
    decode_state_->tile = std::make_unique<AsyncTile>();
    decode_state_->tile->node = this;
    decode_state_->tile->completed = decode_state_->completed;
    decode_state_->tile->invocation = decode_state_->invocation_rec;
    decode_state_->tile->persistent = true;
  }
  if (doorbell_enabled_ && !decode_state_->registered_with_doorbell) {
    if (resident_promotions_ > 0 && spare_slots_ > 0) {
      if (!promo_stream_) {
        int least_prio = 0, greatest_prio = 0;
        if (cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess) {
          CUDA_RETURN_IF_ERROR(cudaStreamCreateWithPriority(&promo_stream_, cudaStreamNonBlocking, least_prio));
        } else {
          CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&promo_stream_, cudaStreamNonBlocking));
        }
      }
      if (!promo_ready_) {
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&promo_ready_, cudaEventDisableTiming));
      }
    }
    decode_state_->registered_with_doorbell = true;
    DoorbellService::Instance().RegisterNode(this);
  }
  return Status::OK();
}

void QMoEOffload::ServiceDoorbell() const {
  if (!decode_state_ || !decode_state_->tile) return;
  const int t_rows = std::max(1, static_cast<int>(*decode_state_->doorbell_tile_rows_host));
  decode_state_->tile->tile_rows = t_rows;
  try {
    RunAsyncCpu(*decode_state_->tile);
  } catch (const std::exception& ex) {
    async_error_message_ = ex.what();
    async_failed_.store(true, std::memory_order_release);
    if (decode_state_->invocation_rec) decode_state_->invocation_rec->MarkFailed(ex.what());
  } catch (...) {
    async_error_message_ = "Unknown exception in QMoEOffload doorbell CPU worker";
    async_failed_.store(true, std::memory_order_release);
    if (decode_state_->invocation_rec) decode_state_->invocation_rec->MarkFailed(async_error_message_);
  }
  // Commit any completed cross-layer async promotion for the NEXT layer (`nodes_[cursor_ % n]`)
  // BEFORE releasing `*doorbell_done_host` on the current layer, while the next layer is guaranteed
  // idle on both the GPU (`stream` is waiting inside current layer's MergeFinalizeCombineKernel) and CPU!
  DoorbellService::Instance().CheckAndCommitBeforeDoorbellRelease(this);
  std::atomic_thread_fence(std::memory_order_release);
  *decode_state_->doorbell_done_host = decode_state_->last_serviced_seq;
}

bool QMoEOffload::TryPrepareCrossLayerPromotion(CrossLayerPromoRequest* out_req) const noexcept {
  if (!out_req || !nvfp4_mode_ || !resident_slab_ || !slot_table_dev_ || !decode_state_ ||
      !decode_state_->pinned_slot_table || !async_work_ || resident_slots_ <= 0 ||
      spare_slots_ <= 0 || resident_promotions_ <= 0 || !pending_promotions_.empty() ||
      IsInQMoEWarmup()) {
    return false;
  }
  constexpr float kMinScore = 2.0f;
  constexpr float kMargin = 2.0f;
  const auto& scores = resident_phase_scores_ && resident_decode_active_ ? decode_score_ : expert_score_;
  auto effective_score = [&](int e) noexcept -> float {
    float s = scores[e];
    if (static_cast<size_t>(e) < expert_presence_ema_.size()) {
      s += presence_ema_weight_ * expert_presence_ema_[e];
    }
    return s;
  };

  const auto& cold_routes = sync_routes_pending_ ? last_cold_routes_ : async_work_->expert_routes;
  int candidate = -1;
  float best = kMinScore - 1.0f;
  for (int expert = 0; expert < num_experts_; ++expert) {
    if (cold_routes[expert].empty() || expert_slot_[expert] >= 0) continue;
    const float cand_score = effective_score(expert);
    if (cand_score > best) {
      candidate = expert;
      best = cand_score;
    }
  }
  if (candidate < 0) return false;

  // Protect the top 75% of static initial experts when running without a separate host RAM cache.
  const int min_evict_slot = (host_cache_bytes_ == 0 && !host_complement_mode_)
                                 ? ((base_resident_slots_ * 3) / 4)
                                 : 0;
  int active_residents = 0;
  int target_slot = -1;
  for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
    if (slot_expert_[idx] >= 0) {
      ++active_residents;
    } else if (target_slot < 0 && idx >= min_evict_slot) {
      target_slot = idx;
    }
  }
  if (target_slot < 0) return false;

  int victim_slot = -1;
  int evicted = -1;
  if (active_residents >= resident_slots_) {
    float lowest = std::numeric_limits<float>::infinity();
    for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
      if (slot_expert_[idx] >= 0) {
        const float res_score = effective_score(slot_expert_[idx]);
        if (res_score < lowest) {
          lowest = res_score;
          victim_slot = idx;
        }
      }
    }
    if (victim_slot < 0) return false;
    const float threshold = lowest * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
    if (best < threshold) return false;
    evicted = slot_expert_[victim_slot];
  } else if (min_evict_slot > 0) {
    const float threshold = 4.0f * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
    if (best < threshold) return false;
  }

  out_req->node = this;
  out_req->device_id = owned_layer_cache_ ? owned_layer_cache_->device_id : 0;
  out_req->candidate = candidate;
  out_req->target_slot = target_slot;
  out_req->victim_slot = victim_slot;
  out_req->evicted_expert = evicted;
  return true;
}

bool QMoEOffload::ExecuteCrossLayerPromotionDma(const CrossLayerPromoRequest& req, uint8_t* pinned_bounce_buf,
                                                size_t bounce_cap) const noexcept {
  if (!pinned_bounce_buf || !nvfp4_mode_ || !resident_slab_ || req.candidate < 0 ||
      req.candidate >= num_experts_ || req.target_slot < 0 || req.target_slot >= physical_slots_) {
    return false;
  }
  size_t total_expert_bytes = 0;
  std::array<size_t, 6> bounce_offs{};
  for (size_t b = 0; b < 6; ++b) {
    bounce_offs[b] = total_expert_bytes;
    total_expert_bytes += (resident_bank_bytes_[b] + 255) & ~size_t{255};
  }
  if (total_expert_bytes > bounce_cap) return false;

  int cur_dev = -1;
  if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != req.device_id) {
    if (cudaSetDevice(req.device_id) != cudaSuccess) return false;
  }
  if (!promo_stream_) {
    int least_prio = 0, greatest_prio = 0;
    if (cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess) {
      if (cudaStreamCreateWithPriority(&promo_stream_, cudaStreamNonBlocking, least_prio) != cudaSuccess) return false;
    } else {
      if (cudaStreamCreateWithFlags(&promo_stream_, cudaStreamNonBlocking) != cudaSuccess) return false;
    }
  }
  if (!promo_ready_) {
    if (cudaEventCreateWithFlags(&promo_ready_, cudaEventDisableTiming) != cudaSuccess) return false;
  }

  // Pre-touch evicted victim's CPU weight pages on this background worker so if the victim is later
  // routed on the CPU, its pages are already resident in the OS working set.
  if (req.evicted_expert >= 0 && req.evicted_expert < num_experts_ && !expert_blocks_ &&
      nvfp4_fc1_w_ && nvfp4_fc2_w_) {
    volatile uint8_t sink = 0;
    const uint8_t* v_fc1 = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) +
                           static_cast<size_t>(req.evicted_expert) * resident_bank_bytes_[0];
    const uint8_t* v_fc2 = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) +
                           static_cast<size_t>(req.evicted_expert) * resident_bank_bytes_[1];
    for (size_t off = 0; off < resident_bank_bytes_[0]; off += 65536) sink ^= v_fc1[off];
    for (size_t off = 0; off < resident_bank_bytes_[1]; off += 65536) sink ^= v_fc2[off];
    (void)sink;
  }

  const uint8_t* host_sources[6]{};
  if (expert_blocks_) {
    const uint8_t* blk = expert_blocks_->Data<uint8_t>() +
                         static_cast<size_t>(req.candidate) * nvfp4_layout_.stride;
    host_sources[0] = blk + nvfp4_layout_.fc1_weights;
    host_sources[1] = blk + nvfp4_layout_.fc2_weights;
    host_sources[2] = blk + nvfp4_layout_.fc1_scales;
    host_sources[3] = blk + nvfp4_layout_.fc2_scales;
    host_sources[4] = blk + nvfp4_layout_.fc1_global_scale;
    host_sources[5] = blk + nvfp4_layout_.fc2_global_scale;
  } else if (nvfp4_fc1_w_ && nvfp4_fc2_w_ && nvfp4_fc1_s_ && nvfp4_fc2_s_) {
    const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
    const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
    host_sources[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) +
                      static_cast<size_t>(req.candidate) * resident_bank_bytes_[0];
    host_sources[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) +
                      static_cast<size_t>(req.candidate) * resident_bank_bytes_[1];
    host_sources[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) +
                      static_cast<size_t>(req.candidate) * resident_bank_bytes_[2];
    host_sources[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) +
                      static_cast<size_t>(req.candidate) * resident_bank_bytes_[3];
    host_sources[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + req.candidate);
    host_sources[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + req.candidate);
  } else {
    return false;
  }

  for (size_t bank = 0; bank < 6; ++bank) {
    std::memcpy(pinned_bounce_buf + bounce_offs[bank], host_sources[bank], resident_bank_bytes_[bank]);
  }
  uint8_t* slab = resident_slab_.get();
  for (size_t bank = 0; bank < 6; ++bank) {
    const size_t slot_offset =
        resident_bank_offsets_[bank] + static_cast<size_t>(req.target_slot) * resident_bank_bytes_[bank];
    if (cudaMemcpyAsync(slab + slot_offset, pinned_bounce_buf + bounce_offs[bank], resident_bank_bytes_[bank],
                        cudaMemcpyHostToDevice, promo_stream_) != cudaSuccess) {
      return false;
    }
  }
  if (nvfp4_fc1_b_ && resident_fc1_bias_slab_) {
    const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
    const uint8_t* src_b =
        static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
        static_cast<size_t>(SafeInt<size_t>(req.candidate) * bias_row_bytes);
    uint8_t* dst_b =
        resident_fc1_bias_slab_.get() + static_cast<size_t>(SafeInt<size_t>(req.target_slot) * bias_row_bytes);
    if (cudaMemcpyAsync(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice, promo_stream_) != cudaSuccess) {
      return false;
    }
  }
  if (cudaEventRecord(promo_ready_, promo_stream_) != cudaSuccess) return false;
  static_cast<void>(cudaStreamQuery(promo_stream_));
  return true;
}

bool QMoEOffload::TryCommitCrossLayerPromotion(const CrossLayerPromoRequest& req) const noexcept {
  if (req.node != this || !promo_ready_ || !promo_stream_ || !decode_state_ || !decode_state_->pinned_slot_table) {
    return false;
  }
  if (cudaEventQuery(promo_ready_) != cudaSuccess) {
    return false;
  }
  if (req.evicted_expert >= 0 && req.evicted_expert < num_experts_ &&
      req.victim_slot >= 0 && req.victim_slot < physical_slots_) {
    expert_slot_[req.evicted_expert] = -1;
    slot_expert_[req.victim_slot] = -1;
  }
  slot_expert_[req.target_slot] = req.candidate;
  expert_slot_[req.candidate] = req.target_slot;
  if (async_work_ && req.candidate >= 0 && req.candidate < static_cast<int>(async_work_->expert_routes.size())) {
    async_work_->expert_routes[req.candidate].clear();
  }
  ++placement_epoch_;
  ++resident_promoted_;
  slot_table_dirty_ = false;

  std::memcpy(decode_state_->pinned_slot_table, expert_slot_.data(),
              static_cast<size_t>(num_experts_) * sizeof(int));
  int* dst_table = CurrentSlotTableDev();
  if (cudaMemcpyAsync(dst_table, decode_state_->pinned_slot_table,
                      static_cast<size_t>(num_experts_) * sizeof(int),
                      cudaMemcpyHostToDevice, promo_stream_) == cudaSuccess) {
    static_cast<void>(cudaStreamQuery(promo_stream_));
  }
  return true;
}

int QMoEOffload::LaunchDeferredMergeStatic(void* ctx, cudaStream_t stream, void* shared_out_ptr, int trigger) {
  if (!ctx) return -1;
  return static_cast<const QMoEOffload*>(ctx)->LaunchDeferredMerge(stream, shared_out_ptr, trigger);
}

int QMoEOffload::LaunchDeferredMerge(cudaStream_t stream, void* shared_out_ptr, int trigger) const {
  if (!decode_state_) return -1;
  auto* shared_out = reinterpret_cast<__nv_bfloat16*>(shared_out_ptr);
  const int tile_rows = decode_state_->deferred_tile_rows;
  const int routes = decode_state_->deferred_routes;
  if (fuse_merge_finalize_ && (hidden_size_ & 7) == 0 && k_ <= 16) {
    const cudaError_t err = LaunchQMoEOffloadMergeFinalizeCombine(
        decode_state_->deferred_wave_output,
        decode_state_->deferred_host_result,
        decode_state_->deferred_hit_flag,
        decode_state_->deferred_scales,
        decode_state_->deferred_routed_out,
        shared_out,
        hidden_size_,
        tile_rows,
        static_cast<int>(k_),
        stream,
        decode_state_->doorbell_seq_dev,
        decode_state_->doorbell_done_host,
        trigger);
    return err == cudaSuccess ? 0 : -1;
  }
  cudaError_t err = LaunchQMoEOffloadMerge(
      decode_state_->deferred_wave_output,
      decode_state_->deferred_host_result,
      decode_state_->deferred_hit_flag,
      decode_state_->deferred_contributions,
      routes,
      hidden_size_,
      tile_rows,
      static_cast<int>(k_),
      stream,
      decode_state_->doorbell_seq_dev,
      decode_state_->doorbell_done_host);
  if (err != cudaSuccess) return -1;
  const ck::MOEParallelismConfig parallelism;
  ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
      decode_state_->deferred_contributions,
      decode_state_->deferred_routed_out,
      nullptr,
      decode_state_->deferred_scales,
      decode_state_->deferred_identity,
      nullptr,
      decode_state_->deferred_identity,
      nullptr,
      tile_rows,
      hidden_size_,
      static_cast<int>(k_),
      routes,
      parallelism,
      false,
      stream);
  if (cudaGetLastError() != cudaSuccess) return -1;
  if (shared_out != nullptr) {
    err = LaunchQMoESharedExpertCombine(
        shared_out, decode_state_->deferred_routed_out, tile_rows * hidden_size_, trigger, stream);
    if (err != cudaSuccess) return -1;
  }
  return 0;
}

Status QMoEOffload::StepBetweenRuns(cudaStream_t stream) const {
  if (cpu_timing_ && layer_index_ == 0) {
    g_step_start_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
        std::memory_order_relaxed);
  }
  ORT_RETURN_IF(async_failed_.load(std::memory_order_acquire),
                "QMoEOffload async CPU expert path failed earlier: ", async_error_message_);
  const bool new_turn_env =
      (layer_index_ == 0 && !IsInQMoEWarmup() &&
       ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_NEW_TURN", 0) == 1);
  if (g_prefill_dma_active.load(std::memory_order_acquire) || new_turn_env) {
    if (new_turn_env) {
      const int next_prompt_tokens = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_TOTAL_TOKENS", 0);
      g_step_prefill_remaining_tokens.store(std::max(0, next_prompt_tokens), std::memory_order_release);
#if defined(_WIN32)
      _putenv_s("ORT_QMOE_PREFILL_NEW_TURN", "0");
#else
      setenv("ORT_QMOE_PREFILL_NEW_TURN", "0", 1);
#endif
    }
    FinalizeQMoEPrefillAndSeedDecode(device_id_, stream);
  }
  const int64_t rem_prefill = g_step_prefill_remaining_tokens.load(std::memory_order_acquire);
  const bool in_step_prefill = (rem_prefill > 0);
  if (in_step_prefill && layer_index_ == 47) {
    const int step_rows =
        (decode_state_ && decode_state_->doorbell_tile_rows_host && *decode_state_->doorbell_tile_rows_host > 0)
            ? static_cast<int>(*decode_state_->doorbell_tile_rows_host)
            : 4;
    g_step_prefill_remaining_tokens.store(std::max<int64_t>(0, rem_prefill - step_rows), std::memory_order_release);
  }
  const bool cross_layer_promo =
      doorbell_enabled_ && spare_slots_ > 0 && bg_promo_stream_ &&
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CROSS_LAYER_PROMO", 0) == 1;
  const bool prev_doorbell_in_flight =
      doorbell_enabled_ && decode_state_ && decode_state_->doorbell_done_host &&
      (*decode_state_->doorbell_done_host < decode_state_->launched_seq);
  const bool should_promote_step =
      !in_step_prefill && !prev_doorbell_in_flight &&
      (!doorbell_enabled_ || !decode_state_ || decode_state_->last_promoted_seq < decode_state_->launched_seq);
  if (resident_slots_ > 0 && !IsInQMoEWarmup() && async_work_ && !cross_layer_promo && should_promote_step) {
    if (decode_state_) {
      decode_state_->last_promoted_seq = decode_state_->launched_seq;
    }
    const auto& prev_routes = sync_routes_pending_ ? last_cold_routes_ : async_work_->expert_routes;
    ORT_RETURN_IF_ERROR(PromoteResident(stream, prev_routes));
    sync_routes_pending_ = false;
    if (slot_table_dirty_) {
      int* dst_table = CurrentSlotTableDev();
      if (owned_layer_cache_) {
        std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
        owned_layer_cache_->slot_table_dev_ptr = dst_table;
      }
      const int* src_table = expert_slot_.data();
      if (decode_state_ && decode_state_->pinned_slot_table) {
        std::memcpy(decode_state_->pinned_slot_table, expert_slot_.data(),
                    SafeInt<size_t>(num_experts_) * sizeof(int));
        src_table = decode_state_->pinned_slot_table;
      }
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_table, src_table,
                                           SafeInt<size_t>(num_experts_) * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
      slot_table_dirty_ = false;
    }
  }
  if (doorbell_enabled_ && decode_state_ && decode_state_->registered_with_doorbell) {
    ++decode_state_->launched_seq;
    DoorbellService::Instance().NotifyPending(1);
  }
  if (cpu_timing_ && layer_index_ == 47) {
    g_step_end_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
        std::memory_order_relaxed);
  }
  return Status::OK();
}

size_t QMoEOffload::LendResidentSlab() const {
  if (!nvfp4_mode_ || resident_slots_ <= 0 || resident_slab_bytes_ == 0 || !resident_slab_) {
    return 0;
  }
  if (promo_stream_) {
    static_cast<void>(cudaStreamSynchronize(promo_stream_));
  }
  if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
    for (const auto& p : pending_promotions_) {
      file_expert_store_.ReleaseLease(p.candidate);
    }
  }
  pending_promotions_.clear();
  if (owned_layer_cache_) {
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    owned_layer_cache_->slab_ptr = nullptr;
  }
  last_slab_ptr_ = resident_slab_.get();
  resident_slab_.reset();
  slot_expert_.assign(physical_slots_, -1);
  expert_slot_.assign(num_experts_, -1);
  slot_table_dirty_ = true;
  return resident_slab_bytes_;
}

size_t QMoEOffload::ReclaimDecodeExtension() const {
  if (!nvfp4_mode_ || decode_extension_slots_ <= 0 || resident_slots_ <= base_resident_slots_) {
    return 0;
  }
  if (promo_stream_) {
    static_cast<void>(cudaStreamSynchronize(promo_stream_));
  }
  if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
    for (const auto& p : pending_promotions_) {
      file_expert_store_.ReleaseLease(p.candidate);
    }
  }
  pending_promotions_.clear();
  const size_t old_bytes = resident_slab_bytes_;
  decode_extension_slots_ = 0;
  resident_slots_ = base_resident_slots_;
  physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
  size_t slab_bytes = 0;
  for (size_t bank = 0; bank < resident_bank_bytes_.size(); ++bank) {
    resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
  }
  resident_slab_bytes_ = slab_bytes;
  last_slab_ptr_ = resident_slab_.get();
  resident_slab_.reset();
  slot_expert_.assign(physical_slots_, -1);
  expert_slot_.assign(num_experts_, -1);
  slot_table_dirty_ = true;
  if (owned_layer_cache_) {
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    owned_layer_cache_->resident_slots = resident_slots_;
    owned_layer_cache_->physical_slots = physical_slots_;
    owned_layer_cache_->bank_offsets = resident_bank_offsets_;
    owned_layer_cache_->slab_ptr = nullptr;
  }
  return old_bytes > slab_bytes ? (old_bytes - slab_bytes) : 0;
}

void QMoEOffload::RestoreDecodeExtensionConfig() const {
  if (!nvfp4_mode_ || configured_decode_extension_slots_ <= 0 ||
      decode_extension_slots_ == configured_decode_extension_slots_) {
    return;
  }
  if (promo_stream_) {
    static_cast<void>(cudaStreamSynchronize(promo_stream_));
  }
  if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
    for (const auto& p : pending_promotions_) {
      file_expert_store_.ReleaseLease(p.candidate);
    }
  }
  pending_promotions_.clear();
  decode_extension_slots_ = configured_decode_extension_slots_;
  resident_slots_ = base_resident_slots_ + decode_extension_slots_;
  physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
  size_t slab_bytes = 0;
  for (size_t bank = 0; bank < resident_bank_bytes_.size(); ++bank) {
    resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
  }
  resident_slab_bytes_ = slab_bytes;
  last_slab_ptr_ = resident_slab_.get();
  resident_slab_.reset();
  slot_expert_.assign(physical_slots_, -1);
  expert_slot_.assign(num_experts_, -1);
  slot_table_dirty_ = true;
  if (owned_layer_cache_) {
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    owned_layer_cache_->resident_slots = resident_slots_;
    owned_layer_cache_->physical_slots = physical_slots_;
    owned_layer_cache_->bank_offsets = resident_bank_offsets_;
    owned_layer_cache_->slab_ptr = nullptr;
  }
}

void QMoEOffload::SyncPrefillSlotTables(cudaStream_t stream) const {
  if (!nvfp4_mode_ || !slot_table_dev_) return;
  if (!pending_promotions_.empty()) {
    if (promo_stream_) {
      static_cast<void>(cudaStreamSynchronize(promo_stream_));
    }
    InlinedVector<int32_t> evicted_to_prefetch;
    for (const auto& p : pending_promotions_) {
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
        file_expert_store_.ReleaseLease(p.candidate);
        if (host_cache_bytes_ > 0) {
          if (!SwapHostCandidateForVictim(p.candidate, p.evicted_expert, p.victim_slot,
                                          promo_stream_ ? promo_stream_ : stream)) {
            file_expert_store_.Evict(p.candidate);
            if (p.evicted_expert >= 0) {
              evicted_to_prefetch.push_back(static_cast<int32_t>(p.evicted_expert));
            }
          }
          host_slot_table_dirty_ = true;
        }
      }
    }
    if (!evicted_to_prefetch.empty()) {
      file_expert_store_.Prefetch(evicted_to_prefetch.data(), evicted_to_prefetch.size(), ++ssd_prefetch_gen_);
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
  }
  if (host_cache_bytes_ > 0 && file_expert_store_.IsInitialized()) {
    // Decode demand reads can replace host slots without changing GPU placement. Prefill is
    // serialized with decode; drain the worker before publishing a fresh, stable snapshot.
    file_expert_store_.WaitPrefetchIdle();
    bool table_changed = host_slot_table_dirty_;
    if (resident_slab_ != nullptr &&
        file_expert_store_.CacheSlots() + static_cast<size_t>(base_resident_slots_) >=
            static_cast<size_t>(num_experts_)) {
      for (int e = 0; e < num_experts_; ++e) {
        if (expert_slot_[e] >= 0) {
          file_expert_store_.Evict(e);
        }
      }
      file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
      bool any_missing = false;
      for (int e = 0; e < num_experts_; ++e) {
        if (expert_slot_[e] < 0 && file_store_expert_slot_[e] < 0) {
          qmoe_shared::ExpertBlockLease lease{};
          if (file_expert_store_.AcquireReady(e, &lease)) {
            file_expert_store_.ReleaseLease(e);
            any_missing = true;
          }
        }
      }
      if (any_missing) {
        file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
        table_changed = true;
      }
    } else if (table_changed) {
      file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
    }
    if (table_changed) {
      int* dst_host_table = CurrentSlotTableDev() + num_experts_;
      if (stream) {
        CUDA_CALL_THROW(cudaMemcpyAsync(dst_host_table, file_store_expert_slot_.data(),
                                        SafeInt<size_t>(num_experts_) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CALL_THROW(cudaMemcpy(dst_host_table, file_store_expert_slot_.data(),
                                   SafeInt<size_t>(num_experts_) * sizeof(int),
                                   cudaMemcpyHostToDevice));
      }
      host_slot_table_dirty_ = false;
    }
  }
  if (slot_table_dirty_) {
    int* dst_table = CurrentSlotTableDev();
    if (stream) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dst_table, expert_slot_.data(),
                                      SafeInt<size_t>(num_experts_) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CALL_THROW(cudaMemcpy(dst_table, expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int),
                                 cudaMemcpyHostToDevice));
    }
    slot_table_dirty_ = false;
  }
}

void QMoEOffload::CopyExpertToResidentSlot(int expert, int slot, cudaStream_t stream) const {
  uint8_t* slab = resident_slab_.get();
  if (!slab || expert < 0 || expert >= num_experts_ || slot < 0 || slot >= physical_slots_) return;
  const uint8_t* src_ptrs[6]{};
  qmoe_shared::ExpertBlockLease lease{};
  bool acquired_from_cache = false;
  thread_local std::vector<uint8_t> direct_staging;
  if (host_cache_bytes_ > 0) {
    const std::array<size_t, 6> offsets{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                        nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                        nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
    if (file_expert_store_.TryAcquireCached(expert, &lease)) {
      acquired_from_cache = true;
      for (size_t b = 0; b < offsets.size(); ++b) src_ptrs[b] = lease.data + offsets[b];
    } else {
      if (direct_staging.size() < nvfp4_layout_.stride) direct_staging.resize(nvfp4_layout_.stride);
      ORT_ENFORCE(file_expert_store_.ReadExpertDirect(expert, direct_staging.data()),
                  "Cannot read expert for resident upload: ", expert);
      for (size_t b = 0; b < offsets.size(); ++b) src_ptrs[b] = direct_staging.data() + offsets[b];
    }
  } else if (expert_blocks_) {
    const size_t blk_off = SafeInt<size_t>(expert) * nvfp4_layout_.stride;
    const uint8_t* blk = expert_blocks_->Data<uint8_t>() + blk_off;
    src_ptrs[0] = blk + nvfp4_layout_.fc1_weights;
    src_ptrs[1] = blk + nvfp4_layout_.fc2_weights;
    src_ptrs[2] = blk + nvfp4_layout_.fc1_scales;
    src_ptrs[3] = blk + nvfp4_layout_.fc2_scales;
    src_ptrs[4] = blk + nvfp4_layout_.fc1_global_scale;
    src_ptrs[5] = blk + nvfp4_layout_.fc2_global_scale;
  } else {
    const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
    const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
    src_ptrs[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[0]);
    src_ptrs[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[1]);
    src_ptrs[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[2]);
    src_ptrs[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[3]);
    src_ptrs[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + expert);
    src_ptrs[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + expert);
  }
  for (size_t bank = 0; bank < 6; ++bank) {
    const size_t dst_off = resident_bank_offsets_[bank] + SafeInt<size_t>(slot) * resident_bank_bytes_[bank];
    if (stream) {
      CUDA_CALL_THROW(cudaMemcpyAsync(slab + dst_off, src_ptrs[bank], resident_bank_bytes_[bank],
                                      cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CALL_THROW(cudaMemcpy(slab + dst_off, src_ptrs[bank], resident_bank_bytes_[bank],
                                 cudaMemcpyHostToDevice));
    }
  }
  if (nvfp4_fc1_b_ && resident_fc1_bias_slab_) {
    const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
    const uint8_t* src_b =
        static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
        static_cast<size_t>(SafeInt<size_t>(expert) * bias_row_bytes);
    uint8_t* dst_b =
        resident_fc1_bias_slab_.get() + static_cast<size_t>(SafeInt<size_t>(slot) * bias_row_bytes);
    if (stream) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CALL_THROW(cudaMemcpy(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice));
    }
  }
  if (host_cache_bytes_ > 0) {
    if (stream) {
      CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    }
    if (acquired_from_cache) {
      file_expert_store_.ReleaseLease(expert);
      file_expert_store_.Evict(expert);
      host_slot_table_dirty_ = true;
    }
  }
}

void QMoEOffload::BatchCopyExpertsToResidentSlots(
    const int* experts, const int* slots, int count, cudaStream_t stream) const {
  if (count <= 0) return;
  if (host_cache_bytes_ == 0 || !file_expert_store_.IsInitialized()) {
#if defined(_WIN32)
    MEMORY_PRIORITY_INFORMATION orig_prio{};
    const bool had_prio = (stream == nullptr) &&
                          (GetThreadInformation(GetCurrentThread(), ThreadMemoryPriority,
                                                &orig_prio, sizeof(orig_prio)) != FALSE);
    if (stream == nullptr) {
      MEMORY_PRIORITY_INFORMATION low_prio{};
      low_prio.MemoryPriority = MEMORY_PRIORITY_VERY_LOW;
      static_cast<void>(SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority,
                                             &low_prio, sizeof(low_prio)));
    }
#endif
    for (int i = 0; i < count; ++i) {
      CopyExpertToResidentSlot(experts[i], slots[i], stream);
    }
#if defined(_WIN32)
    if (stream == nullptr) {
      if (had_prio) {
        static_cast<void>(SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority,
                                               &orig_prio, sizeof(orig_prio)));
      } else {
        MEMORY_PRIORITY_INFORMATION norm_prio{};
        norm_prio.MemoryPriority = MEMORY_PRIORITY_NORMAL;
        static_cast<void>(SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority,
                                               &norm_prio, sizeof(norm_prio)));
      }
      static_cast<void>(SetProcessWorkingSetSize(
          GetCurrentProcess(), static_cast<SIZE_T>(-1), static_cast<SIZE_T>(-1)));
    }
#endif
    return;
  }
  uint8_t* slab = resident_slab_.get();
  if (!slab) return;
  int dev_id = 0;
  static_cast<void>(cudaGetDevice(&dev_id));
  auto& dma_state = DevicePrefillDmaState::ForDevice(dev_id);
  const size_t cap_per_buf = (dma_state.host_staging[0] && dma_state.host_staging[1] && nvfp4_layout_.stride > 0)
                                 ? (dma_state.host_staging_bytes / nvfp4_layout_.stride)
                                 : 0;
  if (cap_per_buf * 2 < static_cast<size_t>(count)) {
    for (int i = 0; i < count; ++i) {
      CopyExpertToResidentSlot(experts[i], slots[i], stream);
    }
    return;
  }
  const std::array<size_t, 6> offsets{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                      nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                      nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
  std::vector<qmoe_shared::ExpertBlockLease> leases(static_cast<size_t>(count));
  std::vector<uint8_t> acquired(static_cast<size_t>(count), 0);
  std::vector<int> miss_indices;
  miss_indices.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const int e = experts[i];
    if (e >= 0 && e < num_experts_ && file_expert_store_.TryAcquireCached(e, &leases[static_cast<size_t>(i)])) {
      acquired[static_cast<size_t>(i)] = 1;
    } else {
      miss_indices.push_back(i);
    }
  }
  std::vector<int> miss_pos(static_cast<size_t>(count), -1);
  if (!miss_indices.empty()) {
    std::vector<uint8_t> read_ok(miss_indices.size(), 0);
    for (size_t m = 0; m < miss_indices.size(); ++m) {
      miss_pos[static_cast<size_t>(miss_indices[m])] = static_cast<int>(m);
    }
    auto read_fn = [&](ptrdiff_t m_idx) noexcept {
      const size_t m = static_cast<size_t>(m_idx);
      const int e = experts[miss_indices[m]];
      uint8_t* dst = dma_state.host_staging[m & 1] + (m >> 1) * nvfp4_layout_.stride;
      read_ok[m] = file_expert_store_.ReadExpertDirect(e, dst) ? 1 : 0;
    };
    if (miss_indices.size() > 1) {
      SharedSpinPool(cpu_threads_).ParallelFor(static_cast<ptrdiff_t>(miss_indices.size()), read_fn, 1);
    } else {
      read_fn(0);
    }
    for (size_t m = 0; m < miss_indices.size(); ++m) {
      if (!read_ok[m]) {
        for (int i = 0; i < count; ++i) {
          if (acquired[static_cast<size_t>(i)]) {
            file_expert_store_.ReleaseLease(experts[i]);
          }
        }
        ORT_THROW("Cannot read expert for resident upload: ", experts[miss_indices[m]]);
      }
    }
  }
  for (int i = 0; i < count; ++i) {
    const int slot = slots[i];
    if (slot < 0 || slot >= physical_slots_) continue;
    const size_t m = static_cast<size_t>(miss_pos[static_cast<size_t>(i)]);
    const uint8_t* base = acquired[static_cast<size_t>(i)]
                              ? leases[static_cast<size_t>(i)].data
                              : (dma_state.host_staging[m & 1] + (m >> 1) * nvfp4_layout_.stride);
    for (size_t bank = 0; bank < 6; ++bank) {
      const size_t dst_off = resident_bank_offsets_[bank] + SafeInt<size_t>(slot) * resident_bank_bytes_[bank];
      if (stream) {
        CUDA_CALL_THROW(cudaMemcpyAsync(slab + dst_off, base + offsets[bank], resident_bank_bytes_[bank],
                                        cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CALL_THROW(cudaMemcpy(slab + dst_off, base + offsets[bank], resident_bank_bytes_[bank],
                                   cudaMemcpyHostToDevice));
      }
    }
    if (nvfp4_fc1_b_ && resident_fc1_bias_slab_) {
      const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
      const uint8_t* src_b =
          static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
          static_cast<size_t>(SafeInt<size_t>(experts[i]) * bias_row_bytes);
      uint8_t* dst_b =
          resident_fc1_bias_slab_.get() + static_cast<size_t>(SafeInt<size_t>(slot) * bias_row_bytes);
      if (stream) {
        CUDA_CALL_THROW(cudaMemcpyAsync(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CALL_THROW(cudaMemcpy(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice));
      }
    }
  }
  if (stream) {
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  }
  for (int i = 0; i < count; ++i) {
    if (acquired[static_cast<size_t>(i)]) {
      file_expert_store_.ReleaseLease(experts[i]);
      file_expert_store_.Evict(experts[i]);
      host_slot_table_dirty_ = true;
    }
  }
}

void QMoEOffload::RestoreAndSeedResidentSlab(
    const int* prompt_hist, int total_prompt_routes, bool enable_seeding, cudaStream_t stream) const {
  if (!nvfp4_mode_ || resident_slots_ <= 0 || resident_slab_bytes_ == 0) return;
  if (promo_stream_) {
    static_cast<void>(cudaStreamSynchronize(promo_stream_));
  }
  if (!pending_promotions_.empty()) {
    for (const auto& p : pending_promotions_) {
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
        file_expert_store_.ReleaseLease(p.candidate);
        if (host_cache_bytes_ > 0) {
          file_expert_store_.Evict(p.candidate);
          host_slot_table_dirty_ = true;
        }
      }
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
  }

  const bool was_lent = (resident_slab_ == nullptr);
  if (was_lent) {
    resident_slab_ = AllocateReservedDeviceSlab(resident_slab_bytes_);
    ORT_ENFORCE(resident_slab_ != nullptr,
                "QMoEOffload failed to restore lent resident slab of ", resident_slab_bytes_, " bytes.");
    slot_expert_.assign(physical_slots_, -1);
    expert_slot_.assign(num_experts_, -1);
    if (owned_layer_cache_) {
      std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
      owned_layer_cache_->slab_ptr = resident_slab_.get();
    }
    if (session_controller_ && resident_slab_.get() != last_slab_ptr_) {
      session_controller_->BumpGraphGeneration();
    }
    last_slab_ptr_ = resident_slab_.get();
  }

  bool modified = was_lent;
  const int init_slots = base_resident_slots_ > 0 ? base_resident_slots_ : resident_slots_;
  if (!enable_seeding) {
    if (decode_state_ && decode_state_->doorbell_done_host && decode_state_->doorbell_req_host) {
      if (stream) {
        CUDA_CALL_THROW(cudaStreamSynchronize(stream));
      }
      while (*decode_state_->doorbell_done_host < *decode_state_->doorbell_req_host) {
        std::this_thread::yield();
      }
      decode_state_->launched_seq = *decode_state_->doorbell_done_host;
      decode_state_->last_promoted_seq = decode_state_->launched_seq;
    }
    const float init_ema = static_cast<float>(k_) / static_cast<float>(std::max(1, num_experts_));
    for (int e = 0; e < num_experts_; ++e) {
      expert_score_[e] = 0.0f;
      if (static_cast<size_t>(e) < expert_presence_ema_.size()) expert_presence_ema_[e] = 0.0f;
      if (resident_phase_scores_) decode_score_[e] = 0.0f;
    }
    for (int slot = 0; slot < init_slots; ++slot) {
      const int e = (static_cast<size_t>(slot) < initial_experts_.size()) ? initial_experts_[slot] : slot;
      if (e >= 0 && e < num_experts_) {
        const float prior = 4.0f + static_cast<float>(init_slots - slot) * 0.25f;
        expert_score_[e] = prior;
        if (static_cast<size_t>(e) < expert_presence_ema_.size()) expert_presence_ema_[e] = init_ema;
        if (resident_phase_scores_) decode_score_[e] = prior;
      }
    }
    routes_since_decay_ = 0;
    phase_routes_since_decay_ = {0, 0};
    resident_decode_active_ = false;
    sync_routes_pending_ = false;
    for (auto& r : last_cold_routes_) r.clear();

    if (was_lent) {
      std::vector<int> restore_experts(init_slots);
      std::vector<int> restore_slots(init_slots);
      for (int slot = 0; slot < init_slots; ++slot) {
        const int e = (static_cast<size_t>(slot) < initial_experts_.size()) ? initial_experts_[slot] : slot;
        restore_experts[slot] = e;
        restore_slots[slot] = slot;
        slot_expert_[slot] = e;
        expert_slot_[e] = slot;
      }
      BatchCopyExpertsToResidentSlots(restore_experts.data(), restore_slots.data(), init_slots, stream);
    } else {
      DoorbellService::Instance().CancelPendingPromo(this);
      if (async_work_) {
        for (auto& r : async_work_->expert_routes) r.clear();
      }
      bool matches = true;
      for (int slot = 0; slot < init_slots; ++slot) {
        const int e = (static_cast<size_t>(slot) < initial_experts_.size()) ? initial_experts_[slot] : slot;
        if (slot_expert_[slot] != e) {
          matches = false;
          break;
        }
      }
      for (int slot = init_slots; slot < physical_slots_; ++slot) {
        if (slot_expert_[slot] >= 0) {
          expert_slot_[slot_expert_[slot]] = -1;
          slot_expert_[slot] = -1;
          modified = true;
        }
      }
      if (!matches) {
        std::vector<int> prev_slot_expert = std::vector<int>(slot_expert_.begin(), slot_expert_.end());
        slot_expert_.assign(physical_slots_, -1);
        expert_slot_.assign(num_experts_, -1);
        std::vector<int> restore_experts;
        std::vector<int> restore_slots;
        for (int slot = 0; slot < init_slots; ++slot) {
          const int e = (static_cast<size_t>(slot) < initial_experts_.size()) ? initial_experts_[slot] : slot;
          if (static_cast<size_t>(slot) >= prev_slot_expert.size() || prev_slot_expert[slot] != e) {
            restore_experts.push_back(e);
            restore_slots.push_back(slot);
          } else if (host_cache_bytes_ > 0) {
            file_expert_store_.Evict(e);
          }
          slot_expert_[slot] = e;
          expert_slot_[e] = slot;
        }
        if (!restore_experts.empty()) {
          BatchCopyExpertsToResidentSlots(
              restore_experts.data(), restore_slots.data(), static_cast<int>(restore_experts.size()), stream);
        }
        modified = true;
      }
    }
  } else {
    const bool seed_nonlent =
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_SEED_NONLENT", 0) == 1;
    if (!was_lent && !seed_nonlent) {
      if (prompt_hist != nullptr && total_prompt_routes > 0 && resident_promotions_ > 0) {
        const float avg_routes = static_cast<float>(total_prompt_routes) / static_cast<float>(num_experts_);
        const float bonus_weight =
            static_cast<float>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_SEED_WEIGHT", 150)) / 100.0f;
        for (int e = 0; e < num_experts_; ++e) {
          const int slot = expert_slot_[e];
          const float floor_score =
              (slot >= 0 && slot < init_slots) ? (4.0f + static_cast<float>(init_slots - slot) * 0.25f) : 0.0f;
          expert_score_[e] = floor_score;
          if (prompt_hist[e] > 0) {
            expert_score_[e] += bonus_weight * (static_cast<float>(prompt_hist[e]) / avg_routes);
          }
          if (resident_phase_scores_) decode_score_[e] = expert_score_[e];
        }
      }
      return;
    }
    // Apply continuous exponential decay for the prompt routes before adding the prompt bonus
    // so multi-turn workloads never inflate minimum resident scores and stall PromoteResident.
    std::vector<float> combined_scores(num_experts_);
    const auto& base_scores = (resident_phase_scores_ && !decode_score_.empty()) ? decode_score_ : expert_score_;
    for (int e = 0; e < num_experts_; ++e) combined_scores[e] = base_scores[e];
    if (prompt_hist != nullptr && total_prompt_routes > 0) {
      const float avg_routes = static_cast<float>(total_prompt_routes) / static_cast<float>(num_experts_);
      const float bonus_weight =
          static_cast<float>(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_SEED_WEIGHT", 150)) / 100.0f;
      for (int e = 0; e < num_experts_; ++e) {
        if (prompt_hist[e] > 0) {
          combined_scores[e] += bonus_weight * (static_cast<float>(prompt_hist[e]) / avg_routes);
        }
      }
    }
    const int max_seed_swaps = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_MAX_SEED_SWAPS", 0);
    if (was_lent || max_seed_swaps > 0) {
      std::vector<int> order(num_experts_);
      for (int e = 0; e < num_experts_; ++e) order[e] = e;
      std::partial_sort(order.begin(), order.begin() + init_slots, order.end(),
                        [&](int a, int b) {
                          if (combined_scores[a] != combined_scores[b]) return combined_scores[a] > combined_scores[b];
                          return a < b;
                        });
      if (was_lent) {
        std::vector<int> restore_experts(init_slots);
        std::vector<int> restore_slots(init_slots);
        for (int slot = 0; slot < init_slots; ++slot) {
          const int e = order[slot];
          restore_experts[slot] = e;
          restore_slots[slot] = slot;
          slot_expert_[slot] = e;
          expert_slot_[e] = slot;
          const float floor_score = 4.0f + static_cast<float>(init_slots - slot) * 0.25f;
          expert_score_[e] = std::max({expert_score_[e], combined_scores[e], floor_score});
          if (resident_phase_scores_) decode_score_[e] = expert_score_[e];
        }
        BatchCopyExpertsToResidentSlots(restore_experts.data(), restore_slots.data(), init_slots, stream);
      } else {
        const float margin_scale = static_cast<float>(resident_margin_percent_) / 100.0f;
        int swaps = 0;
        for (int idx = 0; idx < resident_slots_ && swaps < max_seed_swaps; ++idx) {
          const int cand = order[idx];
          if (expert_slot_[cand] >= 0) continue;
          int victim_slot = -1;
          float lowest = std::numeric_limits<float>::infinity();
          for (int s = 0; s < resident_slots_ + spare_slots_; ++s) {
            const int cur_e = slot_expert_[s];
            if (cur_e >= 0 && combined_scores[cur_e] < lowest) {
              lowest = combined_scores[cur_e];
              victim_slot = s;
            }
          }
          if (victim_slot < 0 || combined_scores[cand] < lowest * margin_scale + 2.0f) break;
          const int evicted = slot_expert_[victim_slot];
          expert_slot_[evicted] = -1;
          CopyExpertToResidentSlot(cand, victim_slot, stream);
          slot_expert_[victim_slot] = cand;
          expert_slot_[cand] = victim_slot;
          ++swaps;
          ++resident_promoted_;
          modified = true;
        }
      }
    }
  }

  if (modified || slot_table_dirty_) {
    if (host_cache_bytes_ > 0 && modified) {
      if (stream) {
        CUDA_CALL_THROW(cudaStreamSynchronize(stream));
      }
      file_expert_store_.WaitPrefetchIdle();
      for (int e = 0; e < num_experts_; ++e) {
        if (expert_slot_[e] >= 0) {
          file_expert_store_.Evict(e);
        }
      }
      file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
      if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_HOST_CACHE_PREWARM", 1) == 1) {
        const size_t target_warm = file_expert_store_.CacheSlots();
        size_t already_warm = 0;
        for (int e = 0; e < num_experts_; ++e) {
          if (expert_slot_[e] < 0 && file_store_expert_slot_[e] >= 0) {
            ++already_warm;
          }
        }
        std::vector<int32_t> cold_to_warm;
        if (already_warm < target_warm) {
          const size_t need_warm = target_warm - already_warm;
          cold_to_warm.reserve(need_warm);
          if (!ranked_experts_.empty()) {
            for (int e : ranked_experts_) {
              if (cold_to_warm.size() >= need_warm) break;
              if (expert_slot_[e] < 0 && file_store_expert_slot_[e] < 0) {
                cold_to_warm.push_back(static_cast<int32_t>(e));
              }
            }
          } else {
            for (int e = 0; e < num_experts_ && cold_to_warm.size() < need_warm; ++e) {
              if (expert_slot_[e] < 0 && file_store_expert_slot_[e] < 0) {
                cold_to_warm.push_back(static_cast<int32_t>(e));
              }
            }
          }
        }
        if (!cold_to_warm.empty()) {
          auto warm_fn = [this, &cold_to_warm](ptrdiff_t idx) {
            qmoe_shared::ExpertBlockLease lease{};
            const int32_t e = cold_to_warm[static_cast<size_t>(idx)];
            if (file_expert_store_.AcquireReady(e, &lease)) {
              file_expert_store_.ReleaseLease(e);
            }
          };
          SharedSpinPool(cpu_threads_).ParallelFor(static_cast<ptrdiff_t>(cold_to_warm.size()), warm_fn, 1);
          file_expert_store_.SnapshotExpertToSlot(file_store_expert_slot_.data(), num_experts_);
        }
      }
      int* dst_host_table = CurrentSlotTableDev() + num_experts_;
      if (stream) {
        CUDA_CALL_THROW(cudaMemcpyAsync(dst_host_table, file_store_expert_slot_.data(),
                                        SafeInt<size_t>(num_experts_) * sizeof(int),
                                        cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CALL_THROW(cudaMemcpy(dst_host_table, file_store_expert_slot_.data(),
                                   SafeInt<size_t>(num_experts_) * sizeof(int),
                                   cudaMemcpyHostToDevice));
      }
      host_slot_table_dirty_ = false;
    }
    ++placement_epoch_;
    int* dst_table = CurrentSlotTableDev();
    if (owned_layer_cache_) {
      std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
      owned_layer_cache_->slot_table_dev_ptr = dst_table;
    }
    const int* src_table = expert_slot_.data();
    if (decode_state_ && decode_state_->pinned_slot_table) {
      std::memcpy(decode_state_->pinned_slot_table, expert_slot_.data(),
                  SafeInt<size_t>(num_experts_) * sizeof(int));
      src_table = decode_state_->pinned_slot_table;
    }
    if (stream) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dst_table, src_table,
                                      SafeInt<size_t>(num_experts_) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CALL_THROW(cudaMemcpy(dst_table, src_table,
                                 SafeInt<size_t>(num_experts_) * sizeof(int),
                                 cudaMemcpyHostToDevice));
    }
    slot_table_dirty_ = false;
  }
}

void QMoEOffload::RecordResidentRoute(int phase, int expert, bool hit, float weight) const {
  ++resident_total_routes_;
  ++phase_total_routes_[phase];
  if (hit) {
    ++resident_hit_routes_;
    ++phase_hit_routes_[phase];
  }
  resident_decode_active_ = phase != 0;
  auto& scores = resident_phase_scores_ && resident_decode_active_ ? decode_score_ : expert_score_;
  scores[expert] += weight;
  if (resident_phase_scores_ && !continuous_decay_) {
    auto& routes = phase_routes_since_decay_[resident_decode_active_ ? 1 : 0];
    if (++routes >= resident_decay_routes_) {
      for (auto& score : scores) score *= 0.5f;
      routes = 0;
    }
  }
}

void QMoEOffload::CommitInvocationObservation(const std::array<int, 5>& phase_rows,
                                              const std::array<std::vector<int>, 5>& phase_expert_rows,
                                              const uint8_t* invocation_expert_rows) const {
  if (!continuous_decay_ || resident_slots_ == 0) return;
  // Continuous exponential decay per committed logical invocation (v3 Section 8.2 / F7):
  // d = exp2(-delta_routes / resident_decay_routes_).
  int total_delta_routes = 0;
  int decode_delta_routes = 0;
  for (int p = 0; p < 5; ++p) {
    const int r = phase_rows[p] * static_cast<int>(k_);
    total_delta_routes += r;
    if (p != 0) decode_delta_routes += r;
  }
  if (total_delta_routes > 0) {
    const float d = std::exp2(-static_cast<float>(total_delta_routes) / static_cast<float>(resident_decay_routes_));
    for (auto& s : expert_score_) s *= d;
  }
  if (resident_phase_scores_ && decode_delta_routes > 0) {
    const float d = std::exp2(-static_cast<float>(decode_delta_routes) / static_cast<float>(resident_decay_routes_));
    for (auto& s : decode_score_) s *= d;
  }
  // Update invocation-presence EMA (half-life = 32 invocations) from either fast invocation_expert_rows or phase_expert_rows.
  if (static_cast<int>(expert_presence_ema_.size()) == num_experts_ && total_delta_routes > 0) {
    const float d_inv = std::exp2(-1.0f / 32.0f);
    if (invocation_expert_rows != nullptr) {
      for (int e = 0; e < num_experts_; ++e) {
        const bool present = invocation_expert_rows[e] > 0;
        expert_presence_ema_[e] = d_inv * expert_presence_ema_[e] + (present ? (1.0f - d_inv) : 0.0f);
      }
    } else if (!phase_expert_rows[0].empty()) {
      for (int e = 0; e < num_experts_; ++e) {
        bool present = false;
        for (int p = 0; p < 5; ++p) {
          if (static_cast<int>(phase_expert_rows[p].size()) == num_experts_ && phase_expert_rows[p][e] > 0) {
            present = true;
            break;
          }
        }
        expert_presence_ema_[e] = d_inv * expert_presence_ema_[e] + (present ? (1.0f - d_inv) : 0.0f);
      }
    }
  }
}

Status QMoEOffload::PromoteResident(cudaStream_t stream, const InlinedVector<InlinedVector<int>>& cold_routes,
                                    int max_promotions, const int* streamed_experts, int streamed_count) const {
  const int promotion_limit = max_promotions > 0 ? max_promotions : resident_promotions_;
  if (pending_promotions_.empty() && promotion_limit <= 0) return Status::OK();
  constexpr float kMinScore = 2.0f;
  constexpr float kMargin = 2.0f;
  const auto& scores = resident_phase_scores_ && resident_decode_active_ ? decode_score_ : expert_score_;
  const bool use_bg_promo =
      async_ && spare_slots_ > 0 && max_promotions == 0 && streamed_experts == nullptr && bg_promo_stream_;

  // Step 1: Poll and publish any completed background promotions (v3 Section 7.5 / F3).
  if (!pending_promotions_.empty()) {
    if (use_bg_promo && promo_ready_ != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaEventSynchronize(promo_ready_));
    } else if (promo_stream_ != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(promo_stream_));
    }
    InlinedVector<int32_t> evicted_to_prefetch;
    for (const auto& p : pending_promotions_) {
      if (host_complement_mode_) {
        if (p.evicted_expert >= 0 && p.host_evict_slot >= 0) {
          expert_host_slot_[p.evicted_expert] = p.host_evict_slot;
          host_slot_expert_[p.host_evict_slot] = p.evicted_expert;
          ++complement_eviction_writebacks_;
        }
        if (p.host_cand_slot >= 0) {
          expert_host_slot_[p.candidate] = -1;
          host_slot_expert_[p.host_cand_slot] = -1;
        }
      }
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
        file_expert_store_.ReleaseLease(p.candidate);
        if (host_cache_bytes_ > 0) {
          host_slot_table_dirty_ = true;
          if (!SwapHostCandidateForVictim(p.candidate, p.evicted_expert, p.victim_slot,
                                          promo_stream_ ? promo_stream_ : stream)) {
            file_expert_store_.Evict(p.candidate);
            if (p.evicted_expert >= 0) {
              evicted_to_prefetch.push_back(static_cast<int32_t>(p.evicted_expert));
            }
          }
        }
      }
      ++placement_epoch_;
      ++resident_promoted_;
    }
    if (!evicted_to_prefetch.empty()) {
      file_expert_store_.Prefetch(evicted_to_prefetch.data(), evicted_to_prefetch.size(), ++ssd_prefetch_gen_);
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
    if (use_bg_promo) {
      // Allow at least one full invocation on the new slot_table_dev_ before reusing retired victim slots (v3 Section 7.5 / F3).
      return Status::OK();
    }
  }

  if (promotion_limit <= 0) return Status::OK();

  // Count currently active logical residents.
  int active_residents = 0;
  for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
    if (slot_expert_[idx] >= 0) ++active_residents;
  }

  auto effective_score = [&](int e) noexcept -> float {
    float s = scores[e];
    if (static_cast<size_t>(e) < expert_presence_ema_.size()) {
      s += presence_ema_weight_ * expert_presence_ema_[e];
    }
    return s;
  };
  size_t single_expert_bytes = 0;
  for (size_t b : resident_bank_bytes_) single_expert_bytes += b;

  if (use_bg_promo) {
    for (int promoted = 0; promoted < std::min(promotion_limit, spare_slots_); ++promoted) {
      int candidate = -1;
      float best = kMinScore - 1.0f;
      for (int expert = 0; expert < num_experts_; ++expert) {
        if (cold_routes[expert].empty() || expert_slot_[expert] >= 0) continue;
        bool already_pending = false;
        for (const auto& p : pending_promotions_) {
          if (p.candidate == expert) {
            already_pending = true;
            break;
          }
        }
        const float cand_score = effective_score(expert);
        if (!already_pending && cand_score > best) {
          candidate = expert;
          best = cand_score;
        }
      }
      if (candidate < 0) break;

      const int min_evict_slot = (host_cache_bytes_ == 0 && !host_complement_mode_ &&
                                  resident_slots_ > base_resident_slots_)
                                     ? base_resident_slots_
                                     : 0;
      int target_slot = -1;
      for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          bool used = false;
          for (const auto& p : pending_promotions_) {
            if (p.target_slot == idx) {
              used = true;
              break;
            }
          }
          if (!used) {
            target_slot = idx;
            break;
          }
        }
      }
      if (target_slot < 0) break;

      int victim_slot = -1;
      int evicted = -1;
      if (active_residents + static_cast<int>(pending_promotions_.size()) >= resident_slots_) {
        float lowest = std::numeric_limits<float>::infinity();
        for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
          if (slot_expert_[idx] >= 0) {
            bool used = false;
            for (const auto& p : pending_promotions_) {
              if (p.victim_slot == idx) {
                used = true;
                break;
              }
            }
            const float res_score = effective_score(slot_expert_[idx]);
            if (!used && res_score < lowest) {
              lowest = res_score;
              victim_slot = idx;
            }
          }
        }
        if (victim_slot < 0) break;
        const float eff_lowest = (min_evict_slot > 0) ? std::max(lowest, 4.0f) : lowest;
        const float threshold = eff_lowest * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
        if (best < threshold) break;
        evicted = slot_expert_[victim_slot];
      } else if (min_evict_slot > 0) {
        const float threshold = 4.0f * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
        if (best < threshold) break;
      }

      const uint64_t step_seq = (decode_state_ && decode_state_->last_serviced_seq > 0)
                                    ? decode_state_->last_serviced_seq
                                    : (resident_total_routes_ / static_cast<uint64_t>(std::max<int64_t>(1, k_)));
      const bool use_ssd = !host_complement_mode_ && file_expert_store_.IsInitialized();
      const bool source_is_pinned =
          all_expert_banks_registered_ || host_complement_mode_ || (use_ssd && host_cache_mapped_);
      uint8_t* pinned_bounce_slot = nullptr;
      if (!promo_budget_retained_) {
        SessionPromotionBudget::Instance().Retain(session_controller_.get());
        promo_budget_retained_ = true;
      }
      if (!SessionPromotionBudget::Instance().TryAcquire(session_controller_.get(), step_seq,
                                                         step_promo_budget_, step_promo_bytes_,
                                                         single_expert_bytes,
                                                         !source_is_pinned, &pinned_bounce_slot)) {
        break;
      }

      if (!promo_stream_) {
        int least_prio = 0, greatest_prio = 0;
        if (cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess) {
          CUDA_RETURN_IF_ERROR(cudaStreamCreateWithPriority(&promo_stream_, cudaStreamNonBlocking, least_prio));
        } else {
          CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&promo_stream_, cudaStreamNonBlocking));
        }
      }
      if (!promo_ready_) {
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&promo_ready_, cudaEventDisableTiming));
      }
      if (slot_reader_done_recorded_ && slot_reader_done_ != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(promo_stream_, slot_reader_done_, 0));
      }

      uint8_t* slab = resident_slab_.get();
      int host_evict_slot = -1;
      int host_cand_slot = -1;
      if (host_complement_mode_) {
        host_cand_slot = expert_host_slot_[candidate];
        if (host_cand_slot < 0) break;
        if (evicted >= 0 && victim_slot >= 0) {
          for (int h = 0; h < complement_slots_; ++h) {
            if (host_slot_expert_[h] < 0) {
              bool used = false;
              for (const auto& p : pending_promotions_) {
                if (p.host_evict_slot == h) {
                  used = true;
                  break;
                }
              }
              if (!used) {
                host_evict_slot = h;
                break;
              }
            }
          }
          if (host_evict_slot < 0) break;
          uint8_t* evict_dst = complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_evict_slot) * complement_stride_);
          if (nvfp4_mode_) {
            const std::array<size_t, 6> dst_offs{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                                 nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                                 nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
            for (size_t bank = 0; bank < 6; ++bank) {
              const size_t src_off = resident_bank_offsets_[bank] + static_cast<size_t>(SafeInt<size_t>(victim_slot) * resident_bank_bytes_[bank]);
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(evict_dst + dst_offs[bank], slab + src_off,
                                                   resident_bank_bytes_[bank], cudaMemcpyDeviceToHost, promo_stream_));
            }
          } else {
            const std::array<size_t, 5> dst_offs{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                                 shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
            for (size_t bank = 0; bank < 5; ++bank) {
              const size_t src_off = resident_bank_offsets_[bank] + static_cast<size_t>(SafeInt<size_t>(victim_slot) * resident_bank_bytes_[bank]);
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(evict_dst + dst_offs[bank], slab + src_off,
                                                   resident_bank_bytes_[bank], cudaMemcpyDeviceToHost, promo_stream_));
            }
          }
        }
      }
      qmoe_shared::ExpertBlockLease ssd_lease{};
      const uint8_t* ssd_promo_blk = nullptr;
      if (use_ssd) {
        ORT_RETURN_IF_NOT(file_expert_store_.AcquireReady(candidate, &ssd_lease),
                          "QMoEOffload FileExpertStore AcquireReady failed during background promotion.");
        ssd_promo_blk = ssd_lease.data;
      }
      if (nvfp4_mode_) {
        const uint8_t* host_sources[6]{};
        if (host_complement_mode_ || use_ssd || expert_blocks_) {
          const uint8_t* blk = host_complement_mode_
                                   ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_cand_slot) * complement_stride_))
                                   : (use_ssd ? ssd_promo_blk
                                              : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(candidate) * nvfp4_layout_.stride)));
          host_sources[0] = blk + nvfp4_layout_.fc1_weights;
          host_sources[1] = blk + nvfp4_layout_.fc2_weights;
          host_sources[2] = blk + nvfp4_layout_.fc1_scales;
          host_sources[3] = blk + nvfp4_layout_.fc2_scales;
          host_sources[4] = blk + nvfp4_layout_.fc1_global_scale;
          host_sources[5] = blk + nvfp4_layout_.fc2_global_scale;
        } else {
          const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
          const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
          host_sources[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[0]);
          host_sources[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[1]);
          host_sources[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[2]);
          host_sources[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[3]);
          host_sources[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + candidate);
          host_sources[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + candidate);
        }
        if (pinned_bounce_slot != nullptr) {
          std::array<size_t, 6> bounce_offs{};
          size_t bounce_off = 0;
          for (size_t bank = 0; bank < 6; ++bank) {
            bounce_offs[bank] = bounce_off;
            bounce_off += (resident_bank_bytes_[bank] + 255u) & ~size_t{255u};
          }
          constexpr size_t kChunkBytes = 64u << 10;
          std::array<size_t, 5> bank_chunk_prefix{};
          for (size_t b = 0; b < 4; ++b) {
            const size_t chunks = (resident_bank_bytes_[b] + kChunkBytes - 1) / kChunkBytes;
            bank_chunk_prefix[b + 1] = bank_chunk_prefix[b] + chunks;
          }
          const ptrdiff_t total_chunks = static_cast<ptrdiff_t>(bank_chunk_prefix[4]);
          auto copy_chunk = [&](ptrdiff_t c_idx) noexcept {
            const size_t u_idx = static_cast<size_t>(c_idx);
            size_t b = 0;
            while (b + 1 < 4 && u_idx >= bank_chunk_prefix[b + 1]) ++b;
            const size_t local_c = u_idx - bank_chunk_prefix[b];
            const size_t byte_off = local_c * kChunkBytes;
            const size_t copy_len = std::min(kChunkBytes, resident_bank_bytes_[b] - byte_off);
            std::memcpy(pinned_bounce_slot + bounce_offs[b] + byte_off,
                        host_sources[b] + byte_off, copy_len);
          };
          SharedSpinPool(cpu_threads_).ParallelFor(total_chunks, copy_chunk, 1);
          for (size_t bank = 4; bank < 6; ++bank) {
            std::memcpy(pinned_bounce_slot + bounce_offs[bank], host_sources[bank], resident_bank_bytes_[bank]);
          }
          for (size_t bank = 0; bank < 6; ++bank) {
            host_sources[bank] = pinned_bounce_slot + bounce_offs[bank];
          }
        }
        for (size_t bank = 0; bank < 6; ++bank) {
          const size_t slot_offset = resident_bank_offsets_[bank] + static_cast<size_t>(SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank]);
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, host_sources[bank], resident_bank_bytes_[bank],
                                               cudaMemcpyHostToDevice, promo_stream_));
        }
        if (nvfp4_fc1_b_ && resident_fc1_bias_slab_) {
          const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
          const uint8_t* src_b =
              static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
              static_cast<size_t>(SafeInt<size_t>(candidate) * bias_row_bytes);
          uint8_t* dst_b =
              resident_fc1_bias_slab_.get() + static_cast<size_t>(SafeInt<size_t>(target_slot) * bias_row_bytes);
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice, promo_stream_));
        }
      } else {
        const uint8_t* block = host_complement_mode_
                                   ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_cand_slot) * complement_stride_))
                                   : (use_ssd ? ssd_promo_blk
                                              : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(candidate) * shared_layout_.stride)));
        const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                            shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
        for (size_t bank = 0; bank < sources.size(); ++bank) {
          const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, block + sources[bank], resident_bank_bytes_[bank],
                                               cudaMemcpyHostToDevice, promo_stream_));
        }
      }
      // Retain the source lease until pending_promotions_ is published or cancelled,
      // including pageable legacy SSD storage. Each acquisition has one release.
      pending_promotions_.push_back({candidate, target_slot, victim_slot, evicted, host_evict_slot, host_cand_slot});
    }
    if (!pending_promotions_.empty()) {
      CUDA_RETURN_IF_ERROR(cudaEventRecord(promo_ready_, promo_stream_));
      static_cast<void>(cudaStreamQuery(promo_stream_));
    }
    return Status::OK();
  }

  for (int promoted = 0; promoted < promotion_limit; ++promoted) {
    int candidate = -1;
    float best = kMinScore - 1.0f;
    for (int expert = 0; expert < num_experts_; ++expert) {
      if (!cold_routes[expert].empty() && expert_slot_[expert] < 0 && scores[expert] > best) {
        candidate = expert;
        best = scores[expert];
      }
    }
    if (candidate < 0) break;

    const int min_evict_slot = (host_cache_bytes_ == 0 && !host_complement_mode_ &&
                                resident_slots_ > base_resident_slots_)
                                   ? base_resident_slots_
                                   : 0;
    int target_slot = -1;
    int evicted = -1;
    int intact_victim_slot = -1;
    if (active_residents < resident_slots_) {
      if (min_evict_slot > 0) {
        const float threshold = 4.0f * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
        if (best < threshold) break;
      }
      for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          target_slot = idx;
          break;
        }
      }
    } else {
      // Cache is at logical capacity C: find lowest-scoring resident victim, and if a spare physical slot
      // exists (P = C + S + D), write into the free spare physical slot first and unmap the victim!
      int victim_slot = -1;
      float lowest = std::numeric_limits<float>::infinity();
      for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] >= 0 && scores[slot_expert_[idx]] < lowest) {
          lowest = scores[slot_expert_[idx]];
          victim_slot = idx;
        }
      }
      if (victim_slot < 0) break;
      const float eff_lowest = (min_evict_slot > 0) ? std::max(lowest, 4.0f) : lowest;
      const float threshold = eff_lowest * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
      if (best < threshold) break;
      evicted = slot_expert_[victim_slot];
      for (int idx = min_evict_slot; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          target_slot = idx;
          break;
        }
      }
      if (target_slot < 0) {
        target_slot = victim_slot;
      } else {
        slot_expert_[victim_slot] = -1;
        intact_victim_slot = victim_slot;
      }
    }

    const auto landing = stream_promotion_reuse_
                             ? qmoe::FindLiveStreamExpert(candidate, streamed_experts, streamed_count, stream_wave_)
                             : qmoe::StreamExpertLocation{};
    const bool reuse = landing.buffer >= 0 && stream_shared_ != nullptr && stream_shared_->landing != nullptr;
    if (reuse && stream_shared_->landing_ready[landing.buffer] != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, stream_shared_->landing_ready[landing.buffer], 0));
    }

    uint8_t* slab = resident_slab_.get();
    int host_evict_slot = -1;
    int host_cand_slot = -1;
    if (host_complement_mode_) {
      host_cand_slot = expert_host_slot_[candidate];
      if (host_cand_slot < 0) break;
      if (evicted >= 0) {
        const int evict_gpu_slot = expert_slot_[evicted];
        for (int h = 0; h < complement_slots_; ++h) {
          if (host_slot_expert_[h] < 0) {
            host_evict_slot = h;
            break;
          }
        }
        if (host_evict_slot < 0) break;
        uint8_t* evict_dst = complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_evict_slot) * complement_stride_);
        if (nvfp4_mode_) {
          const std::array<size_t, 6> dst_offs{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                               nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                               nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
          for (size_t bank = 0; bank < 6; ++bank) {
            const size_t src_off = resident_bank_offsets_[bank] + static_cast<size_t>(SafeInt<size_t>(evict_gpu_slot) * resident_bank_bytes_[bank]);
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(evict_dst + dst_offs[bank], slab + src_off,
                                                 resident_bank_bytes_[bank], cudaMemcpyDeviceToHost, stream));
          }
        } else {
          const std::array<size_t, 5> dst_offs{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                               shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
          for (size_t bank = 0; bank < 5; ++bank) {
            const size_t src_off = resident_bank_offsets_[bank] + static_cast<size_t>(SafeInt<size_t>(evict_gpu_slot) * resident_bank_bytes_[bank]);
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(evict_dst + dst_offs[bank], slab + src_off,
                                                 resident_bank_bytes_[bank], cudaMemcpyDeviceToHost, stream));
          }
        }
      }
    }
    qmoe_shared::ExpertBlockLease ssd_lease{};
    const bool use_ssd = !host_complement_mode_ && !reuse && file_expert_store_.IsInitialized();
    if (use_ssd) {
      ORT_RETURN_IF_NOT(file_expert_store_.AcquireReady(candidate, &ssd_lease),
                        "QMoEOffload FileExpertStore AcquireReady failed during promotion.");
    }
    if (nvfp4_mode_) {
      const uint8_t* host_sources[6]{};
      if (host_complement_mode_ || use_ssd || expert_blocks_) {
        const uint8_t* blk = host_complement_mode_
                                 ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_cand_slot) * complement_stride_))
                                 : (use_ssd ? ssd_lease.data
                                            : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(candidate) * nvfp4_layout_.stride)));
        host_sources[0] = blk + nvfp4_layout_.fc1_weights;
        host_sources[1] = blk + nvfp4_layout_.fc2_weights;
        host_sources[2] = blk + nvfp4_layout_.fc1_scales;
        host_sources[3] = blk + nvfp4_layout_.fc2_scales;
        host_sources[4] = blk + nvfp4_layout_.fc1_global_scale;
        host_sources[5] = blk + nvfp4_layout_.fc2_global_scale;
      } else {
        const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
        const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
        host_sources[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[0]);
        host_sources[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[1]);
        host_sources[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[2]);
        host_sources[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[3]);
        host_sources[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + candidate);
        host_sources[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + candidate);
      }
      for (size_t bank = 0; bank < 6; ++bank) {
        const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
        const uint8_t* source = host_sources[bank];
        if (reuse) {
          const size_t landing_offset = SafeInt<size_t>(landing.slot) * resident_bank_bytes_[bank];
          source = stream_shared_->landing + stream_bank_offsets_[landing.buffer][bank] + landing_offset;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, source, resident_bank_bytes_[bank],
                                             reuse ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
      }
      if (nvfp4_fc1_b_ && resident_fc1_bias_slab_) {
        const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
        const uint8_t* src_b =
            static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
            static_cast<size_t>(SafeInt<size_t>(candidate) * bias_row_bytes);
        uint8_t* dst_b =
            resident_fc1_bias_slab_.get() + static_cast<size_t>(SafeInt<size_t>(target_slot) * bias_row_bytes);
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_b, src_b, bias_row_bytes, cudaMemcpyHostToDevice, stream));
      }
    } else {
      const uint8_t* block = host_complement_mode_
                                 ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(host_cand_slot) * complement_stride_))
                                 : (use_ssd ? ssd_lease.data
                                            : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(candidate) * shared_layout_.stride)));
      const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                          shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
      for (size_t bank = 0; bank < sources.size(); ++bank) {
        const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
        const uint8_t* source = block + sources[bank];
        if (reuse) {
          const size_t landing_offset = SafeInt<size_t>(landing.slot) * resident_bank_bytes_[bank];
          source = stream_shared_->landing + stream_bank_offsets_[landing.buffer][bank] + landing_offset;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, source, resident_bank_bytes_[bank],
                                             reuse ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
      }
    }
    if (use_ssd) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
      file_expert_store_.ReleaseLease(candidate);
      if (host_cache_bytes_ > 0) {
        host_slot_table_dirty_ = true;
        if (!SwapHostCandidateForVictim(candidate, evicted, intact_victim_slot, stream)) {
          file_expert_store_.Evict(candidate);
          if (evicted >= 0) {
            const int32_t ev_id = static_cast<int32_t>(evicted);
            file_expert_store_.Prefetch(&ev_id, 1, ++ssd_prefetch_gen_);
          }
        }
      }
    }
    if (host_complement_mode_) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
      if (evicted >= 0 && host_evict_slot >= 0) {
        expert_host_slot_[evicted] = host_evict_slot;
        host_slot_expert_[host_evict_slot] = evicted;
        ++complement_eviction_writebacks_;
      }
      if (host_cand_slot >= 0) {
        expert_host_slot_[candidate] = -1;
        host_slot_expert_[host_cand_slot] = -1;
      }
    }

    if (reuse && stream_shared_->promotion_read_done[landing.buffer] != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaEventRecord(stream_shared_->promotion_read_done[landing.buffer], stream));
      stream_shared_->bank_has_promotion[landing.buffer] = true;
    }

    if (evicted >= 0) {
      expert_slot_[evicted] = -1;
    } else {
      ++active_residents;
    }
    slot_expert_[target_slot] = candidate;
    expert_slot_[candidate] = target_slot;
    slot_table_dirty_ = true;
    ++placement_epoch_;
    ++resident_promoted_;
    if (reuse) ++stream_reused_promotions_;
  }
  return Status::OK();
}

template <typename ActT, typename ParallelRunner>
int QMoEOffload::RunColdExperts(CpuWork& work, int group_count, const ActT* host_input, int tile_rows, ActT* host_result,
                                int* host_cpu_ids, bool compact, ParallelRunner&& run_parallel,
                                const char* skip_experts) const {
  using ExpertCall = CpuWork::ExpertCall;
  const qmoe_shared::SwiGluParams activation{activation_alpha_, activation_beta_, swiglu_limit_};
  const size_t call_rows = qmoe_shared::kMaxRowsPerCall;
  const size_t group_calls = file_expert_store_.IsInitialized()
                                 ? std::min(static_cast<size_t>(group_count), file_expert_store_.CacheSlots())
                                 : static_cast<size_t>(group_count);
  auto& expert_routes = work.expert_routes;
  auto& calls = work.calls;
  int cpu_rows_written = 0;
  calls.clear();
  for (int expert = 0; expert < num_experts_; ++expert) {
    if (skip_experts && skip_experts[expert]) continue;
    const size_t assigned = expert_routes[expert].size();
    for (size_t first = 0; first < assigned; first += call_rows) {
      calls.push_back({expert, first, static_cast<int>(std::min(call_rows, assigned - first))});
    }
  }
  if (calls.empty()) {
    return 0;
  }
  ORT_ENFORCE(!(v4_planned_mode_ &&
                v4_policy_.cpu_execution == onnxruntime::moe_offload::CpuExecutionMode::kDisabled),
              "QMoEOffload CPU cold expert execution is forbidden when session.moe.offload.cpu_execution='disabled'.");
  cpu_expert_invocations_.fetch_add(calls.size(), std::memory_order_relaxed);
  const bool use_ssd = !host_complement_mode_ && file_expert_store_.IsInitialized();
  const int fc1_tasks = (2 * inter_size_ + fc1_cols_ - 1) / fc1_cols_;
  const int fc2_tasks = (hidden_size_ + fc2_cols_ - 1) / fc2_cols_;
  const bool timed = cpu_timing_ && tile_rows <= 4;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ns = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  const size_t token_Floats = SafeInt<size_t>(tile_rows) * hidden_size_;
  if (work.token_rows.size() < token_Floats) work.token_rows.resize(token_Floats);
  work.token_ready.assign(tile_rows, 0);

  auto make_nvfp4_view = [this](int expert, const uint8_t* override_blk = nullptr) -> qmoe_shared::Nvfp4ExpertView {
    qmoe_shared::Nvfp4ExpertView v{};
    v.bias_is_bf16 = is_bf16_;
    v.is_mxfp4 = mxfp4_mode_;
    const uint8_t* blk = override_blk;
    if (!blk && host_complement_mode_ && expert_host_slot_[expert] >= 0) {
      blk = complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(expert_host_slot_[expert]) * complement_stride_);
    } else if (!blk && expert_blocks_) {
      blk = expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(expert) * nvfp4_layout_.stride);
    }
    if (blk) {
      v.fc1_weights = blk + nvfp4_layout_.fc1_weights;
      v.fc1_scales = blk + nvfp4_layout_.fc1_scales;
      v.fc1_global_scale = *reinterpret_cast<const float*>(blk + nvfp4_layout_.fc1_global_scale);
      v.fc2_weights = blk + nvfp4_layout_.fc2_weights;
      v.fc2_scales = blk + nvfp4_layout_.fc2_scales;
      v.fc2_global_scale = *reinterpret_cast<const float*>(blk + nvfp4_layout_.fc2_global_scale);
      if (nvfp4_fc1_b_) {
        const size_t bias_off = SafeInt<size_t>(expert) * 2 * inter_size_;
        v.fc1_bias = reinterpret_cast<const uint16_t*>(nvfp4_fc1_b_->DataRaw()) + bias_off;
      }
    } else {
      const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
      const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
      v.fc1_weights = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[0]);
      v.fc2_weights = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[1]);
      v.fc1_scales = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[2]);
      v.fc2_scales = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[3]);
      v.fc1_global_scale = fc1_g_base[expert];
      v.fc2_global_scale = fc2_g_base[expert];
      if (nvfp4_fc1_b_) {
        const size_t bias_off = SafeInt<size_t>(expert) * 2 * inter_size_;
        v.fc1_bias = reinterpret_cast<const uint16_t*>(nvfp4_fc1_b_->DataRaw()) + bias_off;
      }
    }
    return v;
  };

  auto is_contiguous_token_range = [&](const InlinedVector<int>& assignments, size_t first_idx, int count, int t0) noexcept {
    for (int r = 1; r < count; ++r) {
      if ((assignments[first_idx + r] / static_cast<int>(k_)) != t0 + r) return false;
    }
    return true;
  };

  std::vector<qmoe_shared::ExpertBlockLease> group_ssd_leases;
  if (use_ssd) group_ssd_leases.resize(group_calls);
  static const bool per_expert_ready = []() {
    const char* env = std::getenv("ORT_QMOE_PER_EXPERT_READY");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();

  for (size_t first_call = 0; first_call < calls.size(); first_call += group_calls) {
    const auto t_prep = timed ? now() : std::chrono::steady_clock::time_point{};
    const ExpertCall* group = calls.data() + first_call;
    const int group_size = static_cast<int>(std::min(group_calls, calls.size() - first_call));
    InlinedVector<int32_t> group_expert_ids;
    if (use_ssd) {
      group_expert_ids.resize(group_size);
      for (int slot = 0; slot < group_size; ++slot) {
        group_expert_ids[slot] = static_cast<int32_t>(group[slot].expert);
      }
      if (!file_expert_store_.TryAcquireGroupCached(
              group_expert_ids.data(), static_cast<size_t>(group_size), group_ssd_leases.data())) {
        if (host_cache_bytes_ > 0) host_slot_table_dirty_ = true;
        if (group_size > 1 && static_cast<size_t>(group_size) <= file_expert_store_.CacheSlots()) {
          InlinedVector<int> miss_slots;
          InlinedVector<uint8_t> acquired(static_cast<size_t>(group_size), 0);
          for (int slot = 0; slot < group_size; ++slot) {
            if (file_expert_store_.TryAcquireCached(group_expert_ids[slot], &group_ssd_leases[slot])) {
              acquired[slot] = 1;
            } else {
              miss_slots.push_back(slot);
            }
          }
          auto acquire_task = [&](ptrdiff_t idx) noexcept {
            const int s = miss_slots[static_cast<size_t>(idx)];
            acquired[s] = file_expert_store_.AcquireReady(group_expert_ids[s], &group_ssd_leases[s]);
          };
          if (miss_slots.size() > 1) {
            run_parallel(static_cast<ptrdiff_t>(miss_slots.size()), acquire_task);
          } else if (!miss_slots.empty()) {
            acquire_task(0);
          }
          for (int slot = 0; slot < group_size; ++slot) {
            if (!acquired[slot]) {
              // Worker jobs must not throw: SpinPool has no exception boundary. Join first,
              // release only successful acquisitions, then report failure on the caller.
              for (int s = 0; s < group_size; ++s) {
                if (acquired[s]) file_expert_store_.ReleaseLease(group_expert_ids[s]);
              }
              ORT_THROW("QMoEOffload FileExpertStore AcquireReady failed for expert ", group_expert_ids[slot]);
            }
          }
        } else {
          ORT_ENFORCE(file_expert_store_.AcquireGroupReady(
                          group_expert_ids.data(), static_cast<size_t>(group_size), group_ssd_leases.data()),
                      "QMoEOffload FileExpertStore AcquireGroupReady failed.");
        }
      }
      const size_t next_first = first_call + static_cast<size_t>(group_size);
      if (next_first < calls.size()) {
        const size_t next_size = std::min(group_calls, calls.size() - next_first);
        InlinedVector<int32_t> next_ids;
        next_ids.reserve(next_size);
        for (size_t s = 0; s < next_size; ++s) {
          next_ids.push_back(static_cast<int32_t>(calls[next_first + s].expert));
        }
        file_expert_store_.Prefetch(next_ids.data(), next_ids.size(), ++ssd_prefetch_gen_);
      }
    }
    for (int slot = 0; slot < group_size; ++slot) {
      const auto& assignments = expert_routes[group[slot].expert];
      const int cnt = group[slot].count;
      const int t0 = assignments[group[slot].first] / static_cast<int>(k_);
      const bool contiguous_tokens = is_contiguous_token_range(assignments, group[slot].first, cnt, t0);
      for (int row = 0; row < cnt; ++row) {
        const int token = assignments[group[slot].first + row] / static_cast<int>(k_);
        float* token_row = work.token_rows.data() + static_cast<size_t>(token) * hidden_size_;
        if (!work.token_ready[token]) {
          ActToFloatRow(host_input + static_cast<size_t>(token) * hidden_size_, work.natural.data(), hidden_size_);
          if (nvfp4_mode_) {
            qmoe_shared::ReorderNvfp4ActivationRow(work.natural.data(), token_row, hidden_size_);
          } else {
            qmoe_shared::ReorderActivationRow(work.natural.data(), token_row, hidden_size_);
          }
          work.token_ready[token] = 1;
        }
        if (!contiguous_tokens) {
          std::memcpy(work.reordered.data() + (slot * call_rows + row) * hidden_size_, token_row,
                      static_cast<size_t>(hidden_size_) * sizeof(float));
        }
      }
      if (static_cast<size_t>(slot) < CpuWork::kMaxGroupSlots) {
        work.fc1_done[slot].value.store(0, std::memory_order_relaxed);
        work.slot_act_in[slot] = contiguous_tokens
                                     ? (work.token_rows.data() + static_cast<size_t>(t0) * hidden_size_)
                                     : (work.reordered.data() + slot * call_rows * hidden_size_);
        if (nvfp4_mode_) {
          work.slot_view[slot] = make_nvfp4_view(group[slot].expert, use_ssd ? group_ssd_leases[slot].data : nullptr);
        }
      }
    }
    const auto t_fc1 = timed ? now() : std::chrono::steady_clock::time_point{};
    auto fc1_task = [&](ptrdiff_t task) {
      const int slot = static_cast<int>(task) / fc1_tasks;
      const int begin = (static_cast<int>(task) % fc1_tasks) * fc1_cols_;
      const int end = std::min(begin + fc1_cols_, 2 * inter_size_);
      const int cnt = group[slot].count;
      const float* act_in = (static_cast<size_t>(slot) < CpuWork::kMaxGroupSlots)
                                ? work.slot_act_in[slot]
                                : (work.reordered.data() + slot * call_rows * hidden_size_);
      const uint8_t* ssd_blk = use_ssd ? group_ssd_leases[slot].data : nullptr;
      if (nvfp4_mode_) {
        const auto view = (static_cast<size_t>(slot) < CpuWork::kMaxGroupSlots)
                              ? work.slot_view[slot]
                              : make_nvfp4_view(group[slot].expert, ssd_blk);
        qmoe_shared::Nvfp4Fc1SwiGluRange(
            hidden_size_, inter_size_, view, act_in,
            cnt, activation, work.intermediate.data() + slot * call_rows * inter_size_,
            begin, end, false, /*bf16_boundaries=*/is_bf16_, /*fp16_boundaries=*/!is_bf16_);
      } else {
        const int exp = group[slot].expert;
        const uint8_t* block = ssd_blk ? ssd_blk
                                       : (host_complement_mode_
                                              ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(expert_host_slot_[exp]) * complement_stride_))
                                              : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(exp) * shared_layout_.stride)));
        qmoe_shared::Fc1SwiGluRange(shared_layout_, block, act_in,
                                    cnt, activation,
                                    work.intermediate.data() + slot * call_rows * inter_size_, begin, end,
                                    false, cpu_fp16_boundaries_);
      }
    };
    auto fc2_task = [&](ptrdiff_t task) {
      const int slot = static_cast<int>(task) / fc2_tasks;
      const int begin = (static_cast<int>(task) % fc2_tasks) * fc2_cols_;
      const int end = std::min(begin + fc2_cols_, hidden_size_);
      const uint8_t* ssd_blk = use_ssd ? group_ssd_leases[slot].data : nullptr;
      if (nvfp4_mode_) {
        const auto view = (static_cast<size_t>(slot) < CpuWork::kMaxGroupSlots)
                              ? work.slot_view[slot]
                              : make_nvfp4_view(group[slot].expert, ssd_blk);
        qmoe_shared::Nvfp4Fc2Range(
            hidden_size_, inter_size_, view, work.intermediate.data() + slot * call_rows * inter_size_,
            group[slot].count, work.result.data() + slot * call_rows * hidden_size_,
            begin, end, false, /*apply_bias=*/false, /*round_scale_bf16=*/is_bf16_);
      } else {
        const int exp = group[slot].expert;
        const uint8_t* block = ssd_blk ? ssd_blk
                                       : (host_complement_mode_
                                              ? (complement_host_slab_.get() + static_cast<size_t>(SafeInt<size_t>(expert_host_slot_[exp]) * complement_stride_))
                                              : (expert_blocks_->Data<uint8_t>() + static_cast<size_t>(SafeInt<size_t>(exp) * shared_layout_.stride)));
        qmoe_shared::Fc2Range(shared_layout_, block, work.intermediate.data() + slot * call_rows * inter_size_,
                              group[slot].count, work.result.data() + slot * call_rows * hidden_size_,
                              begin, end, false, false);
      }
      if (!compact) {
        const auto& assignments = expert_routes[group[slot].expert];
        const int col_count = end - begin;
        for (int row = 0; row < group[slot].count; ++row) {
          const int route = assignments[group[slot].first + row];
          const size_t expanded_row = SafeInt<size_t>(route % static_cast<int>(k_)) * tile_rows +
                                      route / static_cast<int>(k_);
          const size_t destination = expanded_row * hidden_size_ + begin;
          const float* source = work.result.data() + (slot * call_rows + row) * hidden_size_ + begin;
          FloatToActRow(source, host_result + destination, col_count);
        }
      }
    };
    std::chrono::steady_clock::time_point t_fc2{};
    const ptrdiff_t total_fc1 = static_cast<ptrdiff_t>(group_size) * fc1_tasks;
    const ptrdiff_t total_fc2 = static_cast<ptrdiff_t>(group_size) * fc2_tasks;
    if (per_expert_ready && static_cast<size_t>(group_size) <= CpuWork::kMaxGroupSlots) {
      auto fused_task = [&](ptrdiff_t task) {
        if (task < total_fc1) {
          const int slot = static_cast<int>(task) / fc1_tasks;
          fc1_task(task);
          work.fc1_done[slot].value.fetch_add(1, std::memory_order_release);
        } else {
          const ptrdiff_t fc2_idx = task - total_fc1;
          const int slot = static_cast<int>(fc2_idx) / fc2_tasks;
          while (work.fc1_done[slot].value.load(std::memory_order_acquire) < fc1_tasks) {
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
            _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
          }
          fc2_task(fc2_idx);
        }
      };
      run_parallel(total_fc1 + total_fc2, fused_task);
      t_fc2 = timed ? now() : std::chrono::steady_clock::time_point{};
    } else {
      run_parallel(total_fc1, fc1_task);
      t_fc2 = timed ? now() : std::chrono::steady_clock::time_point{};
      run_parallel(total_fc2, fc2_task);
    }
    if (use_ssd) {
      file_expert_store_.ReleaseGroupLeases(group_expert_ids.data(), static_cast<size_t>(group_size));
    }
    const auto t_write = timed ? now() : std::chrono::steady_clock::time_point{};
    if (compact) {
      for (int slot = 0; slot < group_size; ++slot) {
        const auto& assignments = expert_routes[group[slot].expert];
        for (int row = 0; row < group[slot].count; ++row) {
          const int route = assignments[group[slot].first + row];
          host_cpu_ids[cpu_rows_written] = route;
          const size_t destination = SafeInt<size_t>(static_cast<size_t>(cpu_rows_written++)) * hidden_size_;
          const float* source = work.result.data() + (slot * call_rows + row) * hidden_size_;
          FloatToActRow(source, host_result + destination, hidden_size_);
        }
      }
    }
    if (timed) {
      const auto t_end = now();
      timing_prep_ns_ += ns(t_prep, t_fc1);
      timing_fc1_ns_ += ns(t_fc1, t_fc2);
      timing_fc2_ns_ += ns(t_fc2, t_write);
      timing_write_ns_ += ns(t_write, t_end);
    }
  }
  return cpu_rows_written;
}

Status QMoEOffload::ComputeShared(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* router = context->Input<Tensor>(1);
  const Tensor* bias = v3_compact_shared_ ? context->Input<Tensor>(3)
                       : nvfp4_mode_      ? (context->Input<Tensor>(7) ? context->Input<Tensor>(7) : context->Input<Tensor>(24))
                                          : context->Input<Tensor>(24);
  const Tensor* phase = v3_compact_shared_ ? context->Input<Tensor>(4) : context->Input<Tensor>(25);

  const bool legacy_strict = shared_mode_ && !v3_compact_shared_ && !nvfp4_mode_ &&
                             ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OPTIONAL_DIAGNOSTICS", 0) == 0;
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(input && router && bias && phase && input->IsDataType<MLFloat16>() &&
                          router->IsDataType<MLFloat16>() && bias->IsDataType<MLFloat16>() && phase->IsDataType<int8_t>(),
                      "QMoEOffload shared mode requires FP16 input/router/bias and int8 token_phase.");
    for (const auto* tensor : {input, router, bias, phase}) {
      ORT_RETURN_IF_NOT(tensor->Location().device.Type() == OrtDevice::GPU,
                        "QMoEOffload shared activation/router/bias/phase must be on CUDA.");
    }
  } else {
    ORT_RETURN_IF_NOT(input && router, "QMoEOffload requires input and router_probs.");
    ORT_RETURN_IF_NOT(input->Location().device.Type() == OrtDevice::GPU &&
                          router->Location().device.Type() == OrtDevice::GPU,
                      "QMoEOffload input and router_probs must be on CUDA.");
    if (is_bf16_) {
      ORT_RETURN_IF_NOT(input->IsDataType<BFloat16>() && router->IsDataType<BFloat16>(),
                        "QMoEOffload BFloat16 kernel requires BFloat16 input and router_probs.");
    } else {
      ORT_RETURN_IF_NOT(input->IsDataType<MLFloat16>() && router->IsDataType<MLFloat16>(),
                        "QMoEOffload MLFloat16 kernel requires MLFloat16 input and router_probs.");
    }
    if (phase) {
      ORT_RETURN_IF_NOT(phase->IsDataType<int8_t>(), "QMoEOffload token_phase must be int8.");
    }
  }

  const auto& shape = input->Shape();
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[1] == hidden_size_,
                      "QMoEOffload shared input must be packed [T,H].");
  } else {
    ORT_RETURN_IF_NOT((shape.NumDimensions() == 2 || shape.NumDimensions() == 3) &&
                          shape[shape.NumDimensions() - 1] == hidden_size_,
                      "QMoEOffload input must be [T,H] or [B,S,H].");
  }
  const int64_t rows = shape.NumDimensions() == 2
                           ? shape[0]
                           : static_cast<int64_t>(SafeInt<int64_t>(shape[0]) * shape[1]);
  ORT_RETURN_IF_NOT(rows >= 0 && SafeInt<int64_t>(rows) * k_ <= std::numeric_limits<int>::max() &&
                        router->Shape() == TensorShape({rows, num_experts_}) &&
                        (!bias || bias->Shape() == TensorShape({num_experts_, hidden_size_})) &&
                        (!phase || phase->Shape() == TensorShape({rows})),
                    "QMoEOffload shared input shapes are inconsistent.");

  auto* output = context->Output(0, shape);
  auto* row_output = context->Output(1, TensorShape({5, num_experts_}));
  auto* call_output = context->Output(2, TensorShape({5, num_experts_}));
  auto* count_output = context->Output(3, TensorShape({6}));
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(row_output && call_output && count_output, "QMoEOffload shared statistics outputs are required.");
  }
  const size_t statistics_size = SafeInt<size_t>(5) * num_experts_;
  int32_t* route_rows = row_output ? row_output->MutableData<int32_t>() : nullptr;
  int32_t* route_calls = call_output ? call_output->MutableData<int32_t>() : nullptr;
  int32_t* counts = count_output ? count_output->MutableData<int32_t>() : nullptr;
  if (route_rows) std::fill_n(route_rows, statistics_size, 0);
  if (route_calls) std::fill_n(route_calls, statistics_size, 0);
  if (counts) std::fill_n(counts, 6, 0);

  const auto stream = Stream(context);
  cudaStreamCaptureStatus capture;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
  const bool allow_capture = allow_cuda_graph_ && async_ && rows > 0 && rows <= async_max_tokens_;
  ORT_RETURN_IF(capture != cudaStreamCaptureStatusNone && !allow_capture,
                "QMoEOffload shared CPU execution cannot be captured.");
  if (rows == 0) return Status::OK();
  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SUPPRESS_PREFILL_SPIN", 0) == 1) {
    if (rows > async_max_tokens_) {
      qmoe::CpuBudgetGroup::Default().EnterPrefillOrAuxPhase(g_shared_spin_pool_ptr.load(std::memory_order_acquire));
    } else {
      qmoe::CpuBudgetGroup::Default().EnterDecodePhase(g_shared_spin_pool_ptr.load(std::memory_order_acquire));
    }
  }
  const bool full_gpu_fast_path =
      nvfp4_mode_ && (resident_slots_ == num_experts_) && (resident_slab_ != nullptr) &&
      pending_promotions_.empty() && !slot_table_dirty_ &&
      (route_rows == nullptr) && (route_calls == nullptr) && (counts == nullptr) && (phase == nullptr) &&
      (rows <= max_sync_tile_tokens_);
  if ((async_ && rows <= async_max_tokens_) || full_gpu_fast_path) {
    return ComputeSharedAsync(context, input, router, bias, phase, output, route_rows, route_calls, counts,
                              static_cast<int>(rows));
  }
  if (nvfp4_mode_ && prefill_qmoe_ != nullptr) {
    if (async_in_flight_.load(std::memory_order_acquire)) {
      if (aux_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(aux_stream_));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }
    if (!pending_promotions_.empty()) {
      if (promo_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(promo_stream_));
      for (const auto& p : pending_promotions_) {
        if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
          expert_slot_[p.evicted_expert] = -1;
          slot_expert_[p.victim_slot] = -1;
        }
        slot_expert_[p.target_slot] = p.candidate;
        expert_slot_[p.candidate] = p.target_slot;
        if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
          file_expert_store_.ReleaseLease(p.candidate);
        }
        ++placement_epoch_;
        ++resident_promoted_;
      }
      pending_promotions_.clear();
      slot_table_dirty_ = true;
    }
    return prefill_qmoe_->ComputeInternal(context);
  }

  std::shared_ptr<SharedLayerCache> borrowed_cache;
  if (nvfp4_mode_ && resident_slots_ == 0 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARE_TARGET_SLAB", 1) == 1) {
    int dev_id = 0;
    static_cast<void>(cudaGetDevice(&dev_id));
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    auto& reg = SharedLayerCache::Registry();
    auto it = reg.find({dev_id, layer_index_});
    if (it != reg.end()) {
      auto locked = it->second.lock();
      if (locked && locked->num_experts == num_experts_ && locked->hidden_size == hidden_size_ &&
          locked->inter_size == inter_size_ && locked->slab_ptr != nullptr && locked->expert_slot_ptr != nullptr) {
        borrowed_cache = std::move(locked);
      }
    }
  }

  const bool resident = (resident_slots_ > 0) || (borrowed_cache != nullptr);
  const int eff_physical_slots = borrowed_cache ? borrowed_cache->physical_slots : physical_slots_;
  const auto& eff_bank_offsets = borrowed_cache ? borrowed_cache->bank_offsets : resident_bank_offsets_;
  const InlinedVector<int>& eff_expert_slot = borrowed_cache ? *borrowed_cache->expert_slot_ptr : expert_slot_;
  const bool streaming = stream_rows_ > 0 && rows >= stream_min_tokens_;
  const int tile_limit = streaming ? static_cast<int>(std::min<int64_t>(stream_rows_, rows)) : max_sync_tile_tokens_;
  const int route_cap = SafeInt<int>(tile_limit) * static_cast<int>(k_);
  std::unique_lock<std::mutex> resident_lock;
  if (resident) resident_lock = std::unique_lock<std::mutex>(mutex_);
  std::unique_lock<std::mutex> stream_lock;
  if (streaming) stream_lock = std::unique_lock<std::mutex>(stream_shared_->mutex);

  // Sync invocation also drains any prior async work on this node before touching shared score tables (F7).
  if (async_in_flight_.load(std::memory_order_acquire)) {
    if (aux_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(aux_stream_));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  }
  if (!pending_promotions_.empty()) {
    if (promo_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(promo_stream_));
    for (const auto& p : pending_promotions_) {
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      if (!host_complement_mode_ && file_expert_store_.IsInitialized()) {
        file_expert_store_.ReleaseLease(p.candidate);
        if (host_cache_bytes_ > 0) {
          host_slot_table_dirty_ = true;
          if (!SwapHostCandidateForVictim(p.candidate, p.evicted_expert, p.victim_slot,
                                          promo_stream_ ? promo_stream_ : stream)) {
            file_expert_store_.Evict(p.candidate);
            if (p.evicted_expert >= 0) {
              qmoe_shared::ExpertBlockLease ev_lease{};
              if (file_expert_store_.AcquireReady(p.evicted_expert, &ev_lease)) {
                file_expert_store_.ReleaseLease(p.evicted_expert);
              }
            }
          }
        }
      }
      ++placement_epoch_;
      ++resident_promoted_;
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
  }
  ORT_RETURN_IF(async_failed_.load(std::memory_order_acquire),
                "QMoEOffload async CPU expert path failed earlier: ", async_error_message_);

  const size_t elem_size = sizeof(uint16_t);
  size_t gpu_bytes = 0;
  const int route_slots = std::max(route_cap, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(route_slots) * sizeof(int);
  const int cpu_row_cap = streaming ? std::min(route_cap, (stream_min_rows_ - 1) * num_experts_) : route_cap;
  const size_t contribution_bytes = SafeInt<size_t>(route_cap) * hidden_size_ * elem_size;
  const size_t cpu_row_bytes = SafeInt<size_t>(std::max(cpu_row_cap, 1)) * hidden_size_ * elem_size;
  const size_t resident_row_bytes = SafeInt<size_t>(route_slots) * hidden_size_ * elem_size;
  const size_t nvfp4_inter_bytes = nvfp4_mode_ ? static_cast<size_t>(SafeInt<size_t>(route_slots) * inter_size_ * elem_size) : size_t{0};
  const int stream_waves_max = streaming ? ((num_experts_ + stream_wave_ - 1) / stream_wave_) : 0;
  const size_t stream_meta_bytes = streaming
                                       ? static_cast<size_t>(SafeInt<size_t>(route_slots + stream_waves_max * kGroupedMoeMinRows) * sizeof(int))
                                       : size_t{0};
  const bool has_bias = (bias != nullptr) || (fc2_bias_gpu_ != nullptr);

  const size_t scale_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t index_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t unit_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t identity_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t contribution_offset = AddSlab(contribution_bytes, gpu_bytes);
  const size_t bias_offset = has_bias ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t hit_ids_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_compact_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t cpu_ids_offset = (resident || streaming) ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t permutation_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t gathered_offset = (resident || streaming) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t wave_output_offset = (resident || streaming) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t nvfp4_inter_offset = (nvfp4_mode_ && (resident || streaming)) ? AddSlab(nvfp4_inter_bytes, gpu_bytes) : 0;
  const size_t cpu_rows_offset = (resident || streaming) ? AddSlab(streaming ? cpu_row_bytes : contribution_bytes, gpu_bytes) : 0;
  const size_t workspace_offset = (!nvfp4_mode_ && resident) ? AddSlab(streaming ? stream_workspace_bytes_ : resident_workspace_bytes_, gpu_bytes) : 0;
  const size_t stream_ids_offset = streaming ? AddSlab(stream_meta_bytes, gpu_bytes) : 0;
  const size_t stream_compact_offset = streaming ? AddSlab(stream_meta_bytes, gpu_bytes) : 0;

  size_t host_bytes = 0;
  const size_t host_index_offset = AddSlab(route_bytes, host_bytes);
  const size_t host_phase_offset = AddSlab(tile_limit, host_bytes);
  const size_t host_input_offset = AddSlab(SafeInt<size_t>(tile_limit) * hidden_size_ * elem_size, host_bytes);
  const size_t host_result_offset = AddSlab(streaming ? cpu_row_bytes : contribution_bytes, host_bytes);
  const size_t host_hit_ids_offset = resident ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_hit_compact_offset = resident ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_cpu_ids_offset = (resident || streaming) ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_stream_ids_offset = streaming ? AddSlab(stream_meta_bytes, host_bytes) : 0;
  const size_t host_stream_compact_offset = streaming ? AddSlab(stream_meta_bytes, host_bytes) : 0;

  ORT_RETURN_IF(gpu_bytes > (streaming ? std::max(shared_gpu_limit_, stream_gpu_limit_) : shared_gpu_limit_) ||
                    host_bytes > (streaming ? std::max(shared_pinned_limit_, stream_pinned_limit_) : shared_pinned_limit_),
                "QMoEOffload shared scratch exceeds configured GPU or pinned limits.");

  auto scratch = GetScratchBuffer<uint8_t>(gpu_bytes, GetComputeStream(context));
  auto completed = std::make_shared<bool>(false);
  IAllocatorUniquePtr<uint8_t> pinned;
  uint8_t* pinned_base = nullptr;
  if (streaming) {
    int dev_id = 0;
    static_cast<void>(cudaGetDevice(&dev_id));
    ORT_RETURN_IF_ERROR(stream_shared_->Ensure(dev_id, stream_landing_bytes_, host_bytes, static_cast<size_t>(stream_waves_max)));
    pinned_base = stream_shared_->pinned;
  } else {
    const bool reuse_pinned =
        (session_controller_ != nullptr) ||
        (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PINNED_REUSE", 0) == 1);
    pinned = reuse_pinned ? AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed)
                          : AllocateBufferOnCPUPinned<uint8_t>(host_bytes);
    pinned_base = pinned.get();
  }
  ORT_RETURN_IF_NOT(scratch && pinned_base, "QMoEOffload cannot allocate bounded shared scratch.");

  bool failed = false;
  StreamDrain drain(stream, failed, completed);
  auto* base = scratch.get();
  auto* scales = reinterpret_cast<float*>(base + scale_offset);
  auto* indices = reinterpret_cast<int*>(base + index_offset);
  auto* unit_scales = reinterpret_cast<float*>(base + unit_offset);
  auto* identity = reinterpret_cast<int*>(base + identity_offset);
  auto* contributions = base + contribution_offset;
  auto* route_bias = has_bias ? (base + bias_offset) : nullptr;
  auto* host_indices = reinterpret_cast<int*>(pinned_base + host_index_offset);
  auto* host_phases = reinterpret_cast<int8_t*>(pinned_base + host_phase_offset);
  auto* host_input = pinned_base + host_input_offset;
  auto* host_result = pinned_base + host_result_offset;
  auto* host_hit_ids = reinterpret_cast<int*>(pinned_base + host_hit_ids_offset);
  auto* host_hit_compact = reinterpret_cast<int*>(pinned_base + host_hit_compact_offset);
  auto* host_cpu_ids = reinterpret_cast<int*>(pinned_base + host_cpu_ids_offset);
  auto* host_stream_ids = reinterpret_cast<int*>(pinned_base + host_stream_ids_offset);
  auto* host_stream_compact = reinterpret_cast<int*>(pinned_base + host_stream_compact_offset);
  auto* stream_ids = reinterpret_cast<int*>(base + stream_ids_offset);
  auto* stream_compact = reinterpret_cast<int*>(base + stream_compact_offset);
  auto* hit_ids = reinterpret_cast<int*>(base + hit_ids_offset);
  auto* hit_compact = reinterpret_cast<int*>(base + hit_compact_offset);
  auto* cpu_ids = reinterpret_cast<int*>(base + cpu_ids_offset);
  auto* permutation = reinterpret_cast<int*>(base + permutation_offset);
  auto* gathered = base + gathered_offset;
  auto* wave_output = base + wave_output_offset;
  auto* nvfp4_inter_buf = base + nvfp4_inter_offset;
  auto* cpu_rows = base + cpu_rows_offset;
  auto* workspace = reinterpret_cast<char*>(base + workspace_offset);

  ck::ActivationParams runner_activation(activation_type_);
  runner_activation.alpha = activation_alpha_;
  runner_activation.beta = activation_beta_;
  runner_activation.limit = swiglu_limit_;
  runner_activation.swiglu_fusion = swiglu_fusion_;
  uint8_t* slab = borrowed_cache ? borrowed_cache->slab_ptr : (resident ? resident_slab_.get() : nullptr);
  const auto quantization = (!nvfp4_mode_ && resident)
                                ? ck::QuantParams::GroupWise(32, slab + eff_bank_offsets[2], slab + eff_bank_offsets[3])
                                : ck::QuantParams{};
  struct HitRoute {
    int slot;
    int route;
  };
  InlinedVector<HitRoute> hits;
  const ck::MOEParallelismConfig parallelism;
  const int group_calls = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", group_calls_), 1, 64);
  CpuWork cpu_work(group_calls, num_experts_, hidden_size_, inter_size_);
  auto& expert_routes = cpu_work.expert_routes;
  InlinedVector<char> stream_skip;
  InlinedVector<int> stream_list;

  std::array<int, 5> obs_phase_rows{};
  std::array<std::vector<int>, 5> obs_phase_expert_rows{};

  const void* bias_gpu_ptr = fc2_bias_gpu_ ? fc2_bias_gpu_.get()
                                           : (bias && bias->Location().device.Type() == OrtDevice::GPU ? bias->DataRaw() : nullptr);
  IAllocatorUniquePtr<uint8_t> transient_bias_gpu;
  if (!bias_gpu_ptr && bias && bias->Location().device.Type() == OrtDevice::CPU) {
    transient_bias_gpu = GetScratchBuffer<uint8_t>(bias->SizeInBytes(), GetComputeStream(context));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(transient_bias_gpu.get(), bias->DataRaw(), bias->SizeInBytes(),
                                         cudaMemcpyHostToDevice, stream));
    bias_gpu_ptr = transient_bias_gpu.get();
  }

  auto launch_nvfp4_gemv_pair = [&](const void* act_in, const uint8_t* w_base, const std::array<size_t, 6>& bank_offs,
                                    const void* fc1_bias_dev,
                                    const int* expert_map, int num_exp, int num_r, void* out_buf) -> Status {
    const uint8_t* fc1_w = w_base + bank_offs[0];
    const uint8_t* fc2_w = w_base + bank_offs[1];
    const uint8_t* fc1_s = w_base + bank_offs[2];
    const uint8_t* fc2_s = w_base + bank_offs[3];
    const float* fc1_g = reinterpret_cast<const float*>(w_base + bank_offs[4]);
    const float* fc2_g = reinterpret_cast<const float*>(w_base + bank_offs[5]);
    const int fp4_group = mxfp4_mode_ ? 32 : 16;
    if (is_bf16_) {
      gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(act_in), fc1_w,
          static_cast<const __nv_bfloat16*>(fc1_bias_dev), fc1_s, fc1_g, nullptr,
          reinterpret_cast<__nv_bfloat16*>(nvfp4_inter_buf), nullptr, expert_map, num_exp,
          num_r, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
          false, true, nullptr, num_r, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      gemv::launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
          static_cast<__nv_bfloat16*>(out_buf), nullptr, expert_map, num_exp,
          num_r, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
          false, true, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    } else {
      gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
          static_cast<const half*>(act_in), fc1_w,
          static_cast<const half*>(fc1_bias_dev), fc1_s, fc1_g, nullptr,
          reinterpret_cast<half*>(nvfp4_inter_buf), nullptr, expert_map, num_exp,
          num_r, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
          false, true, nullptr, num_r, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      gemv::launch_moe_gemv_fp4_symmetric<half>(
          reinterpret_cast<const half*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
          static_cast<half*>(out_buf), nullptr, expert_map, num_exp,
          num_r, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
          false, true, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
    return Status::OK();
  };

  struct StagedSyncRouteObs {
    int phase = 0;
    int expert = 0;
    bool hit = false;
  };
  std::vector<StagedSyncRouteObs> staged_sync_routes;
  const bool stage_obs_until_run_end = (session_controller_ != nullptr);

  for (int64_t start = 0; start < rows;) {
    const int tile_rows = static_cast<int>(std::min<int64_t>(tile_limit, rows - start));
    const int routes = SafeInt<int>(tile_rows) * k_;
    const size_t activation_start = SafeInt<size_t>(start) * hidden_size_;
    const size_t router_start = SafeInt<size_t>(start) * num_experts_;

    if (is_bf16_) {
      LaunchSoftmaxTopK(reinterpret_cast<const __nv_bfloat16*>(router->DataRaw()) + router_start,
                        scales, indices, tile_rows, num_experts_, static_cast<int>(k_), normalize_routing_weights_, stream);
    } else {
      LaunchSoftmaxTopK(reinterpret_cast<const half*>(router->DataRaw()) + router_start,
                        scales, indices, tile_rows, num_experts_, static_cast<int>(k_), normalize_routing_weights_, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices, indices, SafeInt<size_t>(routes) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (phase) {
      if (phase->Location().device.Type() == OrtDevice::GPU) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_phases, phase->Data<int8_t>() + start, tile_rows, cudaMemcpyDeviceToHost, stream));
      } else {
        std::memcpy(host_phases, phase->Data<int8_t>() + start, tile_rows);
      }
    } else {
      const int8_t inferred_phase = (rows > async_max_tokens_) ? 0 : 1;
      std::fill_n(host_phases, tile_rows, inferred_phase);
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input, static_cast<const uint8_t*>(input->DataRaw()) + activation_start * elem_size,
                                         SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size, cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    for (auto& assignments : expert_routes) assignments.clear();
    hits.clear();
    for (int token = 0; token < tile_rows; ++token) {
      const int token_phase = host_phases[token];
      ORT_RETURN_IF(token_phase < 0 || token_phase >= 5, "QMoEOffload invalid token_phase.");
      if (counts) ++counts[token_phase];
      ++obs_phase_rows[token_phase];
      for (int rank = 0; rank < k_; ++rank) {
        const int route = token * static_cast<int>(k_) + rank;
        const int expert = host_indices[route];
        ORT_RETURN_IF(expert < 0 || expert >= num_experts_, "QMoEOffload invalid routed expert.");
        for (int previous = 0; previous < rank; ++previous) {
          ORT_RETURN_IF(host_indices[token * static_cast<int>(k_) + previous] == expert,
                        "QMoEOffload duplicate routed expert.");
        }
        const size_t statistic = SafeInt<size_t>(token_phase) * num_experts_ + expert;
        if (route_rows) ++route_rows[statistic];
        if (route_calls) route_calls[statistic] = 1;
        const int resident_slot = resident ? eff_expert_slot[expert] : -1;
        if (resident_slot >= 0) {
          hits.push_back({resident_slot, route});
        } else {
          expert_routes[expert].push_back(route);
        }
        if (resident_slots_ > 0) {
          if (stage_obs_until_run_end) {
            staged_sync_routes.push_back({token_phase, expert, resident_slot >= 0});
          } else {
            RecordResidentRoute(token_phase, expert, resident_slot >= 0);
          }
        }
      }
    }

    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(unit_scales, identity, std::max(routes, kGroupedMoeMinRows), stream));
    if (!hits.empty()) {
      std::sort(hits.begin(), hits.end(), [](const HitRoute& left, const HitRoute& right) {
        return left.slot != right.slot ? left.slot < right.slot : left.route < right.route;
      });
      const int hit_count = static_cast<int>(hits.size());
      const int compute_rows = nvfp4_mode_ ? hit_count : std::max(hit_count, kGroupedMoeMinRows);
      for (int row = 0; row < compute_rows; ++row) {
        const auto& hit = hits[row < hit_count ? row : 0];
        host_hit_ids[row] = hit.route;
        host_hit_compact[row] = hit.slot;
      }
      const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(hit_ids, host_hit_ids, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(hit_compact, host_hit_compact, metadata_bytes, cudaMemcpyHostToDevice, stream));
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
            reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()) + activation_start,
            hit_ids, reinterpret_cast<__nv_bfloat16*>(gathered), compute_rows, hidden_size_, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
            reinterpret_cast<const half*>(input->DataRaw()) + activation_start,
            hit_ids, reinterpret_cast<half*>(gathered), compute_rows, hidden_size_, static_cast<int>(k_), stream));
      }
      if (nvfp4_mode_) {
        ORT_RETURN_IF_ERROR(launch_nvfp4_gemv_pair(
            gathered, slab, eff_bank_offsets,
            resident_fc1_bias_slab_ ? resident_fc1_bias_slab_.get() : nullptr,
            hit_compact, eff_physical_slots, compute_rows, wave_output));
      } else {
        runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, hit_compact, unit_scales,
                        slab + eff_bank_offsets[0], reinterpret_cast<const void*>(slab + eff_bank_offsets[4]), activation_type_,
                        slab + eff_bank_offsets[1], nullptr, quantization, compute_rows, hidden_size_, inter_size_,
                        eff_physical_slots, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      }
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
            reinterpret_cast<const __nv_bfloat16*>(wave_output), hit_ids,
            reinterpret_cast<__nv_bfloat16*>(contributions), hit_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
            reinterpret_cast<const half*>(wave_output), hit_ids,
            reinterpret_cast<half*>(contributions), hit_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      }
    }

    const char* skip_experts = nullptr;
    uint64_t tile_stream_routes = 0;
    if (streaming) {
      stream_skip.assign(num_experts_, 0);
      stream_list.clear();
      for (int expert = 0; expert < num_experts_; ++expert) {
        if (static_cast<int>(expert_routes[expert].size()) >= stream_min_rows_) {
          stream_list.push_back(expert);
          stream_skip[expert] = 1;
        }
      }
      skip_experts = stream_skip.data();
      auto& shared = *stream_shared_;
      const size_t stream_count = stream_list.size();
      const int waves = static_cast<int>((stream_count + stream_wave_ - 1) / stream_wave_);
      size_t meta_offset = 0;
      for (int wave = 0; wave < waves; ++wave) {
        const int bank_idx = wave & 1;
        const size_t first = static_cast<size_t>(wave) * stream_wave_;
        const int count = static_cast<int>(std::min<size_t>(stream_wave_, stream_count - first));
        const auto& offsets = stream_bank_offsets_[bank_idx];

        // Event-ordered landing ring reuse across waves and across layers (v3 Section 7.6 / F5).
        if (wave >= 2) {
          CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.gemm_done[wave - 2], 0));
        } else {
          if (shared.bank_has_gemm[bank_idx]) {
            CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.last_gemm_done[bank_idx], 0));
          }
          if (shared.bank_has_promotion[bank_idx]) {
            CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.promotion_read_done[bank_idx], 0));
            shared.bank_has_promotion[bank_idx] = false;
          }
        }

        if (nvfp4_mode_) {
          for (int begin = 0; begin < count;) {
            int end = begin + 1;
            while (end < count && stream_list[first + end] == stream_list[first + end - 1] + 1) ++end;
            const int run_len = end - begin;
            const int first_exp = stream_list[first + begin];
            if (expert_blocks_) {
              const uint8_t* blocks = expert_blocks_->Data<uint8_t>();
              const std::array<size_t, 6> sources{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                                  nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                                  nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
              for (size_t bank = 0; bank < 6; ++bank) {
                const size_t dst = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
                const size_t src = SafeInt<size_t>(first_exp) * nvfp4_layout_.stride + sources[bank];
                CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(shared.landing + dst, resident_bank_bytes_[bank],
                                                       blocks + src, nvfp4_layout_.stride,
                                                       resident_bank_bytes_[bank], static_cast<size_t>(run_len),
                                                       cudaMemcpyHostToDevice, shared.copy_stream));
              }
            } else if (host_cache_bytes_ > 0 && file_expert_store_.IsInitialized()) {
              const std::array<size_t, 6> sources{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                                  nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                                  nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
              for (int r = 0; r < run_len; ++r) {
                const int exp_id = first_exp + r;
                qmoe_shared::ExpertBlockLease lease{};
                ORT_RETURN_IF_NOT(file_expert_store_.AcquireReady(exp_id, &lease),
                                  "QMoEOffload FileExpertStore AcquireReady failed during GPU streaming.");
                for (size_t bank = 0; bank < 6; ++bank) {
                  const size_t dst = offsets[bank] + SafeInt<size_t>(begin + r) * resident_bank_bytes_[bank];
                  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shared.landing + dst, lease.data + sources[bank],
                                                       resident_bank_bytes_[bank],
                                                       cudaMemcpyHostToDevice, shared.copy_stream));
                }
                CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(shared.copy_stream));
                file_expert_store_.ReleaseLease(exp_id);
              }
            } else {
              const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
              const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
              const uint8_t* bank_bases[6] = {
                  static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()),
                  reinterpret_cast<const uint8_t*>(fc1_g_base),
                  reinterpret_cast<const uint8_t*>(fc2_g_base)};
              for (size_t bank = 0; bank < 6; ++bank) {
                const size_t dst = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
                const size_t src = SafeInt<size_t>(first_exp) * resident_bank_bytes_[bank];
                const size_t copy_bytes = SafeInt<size_t>(run_len) * resident_bank_bytes_[bank];
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shared.landing + dst, bank_bases[bank] + src, copy_bytes,
                                                     cudaMemcpyHostToDevice, shared.copy_stream));
              }
            }
            if (nvfp4_fc1_b_ && stream_fc1_bias_slab_[bank_idx]) {
              const size_t bias_row_bytes = static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(uint16_t));
              const uint8_t* src_b =
                  static_cast<const uint8_t*>(nvfp4_fc1_b_->DataRaw()) +
                  static_cast<size_t>(SafeInt<size_t>(first_exp) * bias_row_bytes);
              uint8_t* dst_b =
                  stream_fc1_bias_slab_[bank_idx].get() + static_cast<size_t>(SafeInt<size_t>(begin) * bias_row_bytes);
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
                  dst_b, src_b, static_cast<size_t>(SafeInt<size_t>(run_len) * bias_row_bytes),
                  cudaMemcpyHostToDevice, shared.copy_stream));
            }
            begin = end;
          }
        } else {
          const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                              shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
          const uint8_t* const blocks = expert_blocks_->Data<uint8_t>();
          for (int begin = 0; begin < count;) {
            int end = begin + 1;
            while (end < count && stream_list[first + end] == stream_list[first + end - 1] + 1) ++end;
            for (size_t bank = 0; bank < sources.size(); ++bank) {
              const size_t destination = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
              const size_t source = SafeInt<size_t>(stream_list[first + begin]) * shared_layout_.stride + sources[bank];
              CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(shared.landing + destination, resident_bank_bytes_[bank], blocks + source,
                                                     shared_layout_.stride, resident_bank_bytes_[bank],
                                                     static_cast<size_t>(end - begin), cudaMemcpyHostToDevice,
                                                     shared.copy_stream));
            }
            begin = end;
          }
        }

        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.copy_done[wave], shared.copy_stream));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.landing_ready[bank_idx], shared.copy_stream));
        ++shared.bank_generation[bank_idx];

        size_t wave_routes = 0;
        for (int local = 0; local < count; ++local) {
          for (const int route : expert_routes[stream_list[first + local]]) {
            host_stream_ids[meta_offset + wave_routes] = route;
            host_stream_compact[meta_offset + wave_routes] = local;
            ++wave_routes;
          }
        }
        const int compute_rows = nvfp4_mode_ ? static_cast<int>(wave_routes)
                                             : std::max(static_cast<int>(wave_routes), kGroupedMoeMinRows);
        for (int row = static_cast<int>(wave_routes); row < compute_rows; ++row) {
          host_stream_ids[meta_offset + row] = host_stream_ids[meta_offset];
          host_stream_compact[meta_offset + row] = host_stream_compact[meta_offset];
        }
        const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, shared.copy_done[wave], 0));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(stream_ids + meta_offset, host_stream_ids + meta_offset, metadata_bytes,
                                             cudaMemcpyHostToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(stream_compact + meta_offset, host_stream_compact + meta_offset, metadata_bytes,
                                             cudaMemcpyHostToDevice, stream));
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
              reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()) + activation_start,
              stream_ids + meta_offset, reinterpret_cast<__nv_bfloat16*>(gathered), compute_rows, hidden_size_,
              static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
              reinterpret_cast<const half*>(input->DataRaw()) + activation_start,
              stream_ids + meta_offset, reinterpret_cast<half*>(gathered), compute_rows, hidden_size_,
              static_cast<int>(k_), stream));
        }
        uint8_t* landing = shared.landing;
        if (nvfp4_mode_) {
          ORT_RETURN_IF_ERROR(launch_nvfp4_gemv_pair(
              gathered, landing, offsets,
              stream_fc1_bias_slab_[bank_idx] ? stream_fc1_bias_slab_[bank_idx].get() : nullptr,
              stream_compact + meta_offset, count, compute_rows, wave_output));
        } else {
          const auto wave_quantization = ck::QuantParams::GroupWise(32, landing + offsets[2], landing + offsets[3]);
          runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, stream_compact + meta_offset, unit_scales,
                          landing + offsets[0], reinterpret_cast<const void*>(landing + offsets[4]), activation_type_,
                          landing + offsets[1], nullptr, wave_quantization, compute_rows, hidden_size_, inter_size_,
                          count, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
          CUDA_RETURN_IF_ERROR(cudaGetLastError());
        }
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const __nv_bfloat16*>(wave_output), stream_ids + meta_offset,
              reinterpret_cast<__nv_bfloat16*>(contributions), static_cast<int>(wave_routes), hidden_size_,
              tile_rows, static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const half*>(wave_output), stream_ids + meta_offset,
              reinterpret_cast<half*>(contributions), static_cast<int>(wave_routes), hidden_size_,
              tile_rows, static_cast<int>(k_), stream));
        }
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.gemm_done[wave], stream));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.last_gemm_done[bank_idx], stream));
        shared.bank_has_gemm[bank_idx] = true;
        meta_offset += compute_rows;
        tile_stream_routes += wave_routes;
      }
      ++stream_tiles_;
      stream_experts_ += stream_count;
      stream_routes_ += tile_stream_routes;
      stream_waves_ += static_cast<uint64_t>(waves);
      stream_cpu_routes_ += static_cast<uint64_t>(routes) - hits.size() - tile_stream_routes;
    }

    auto parallel_runner = [&](ptrdiff_t count, auto& function) {
      if (unified_cpu_service_ && cpu_threads_ > 0) {
        SharedSpinPool(cpu_threads_).ParallelFor(count, function, pool_task_batch_);
      } else {
        OffloadParallelFor(context, count, function);
      }
    };
    const int cpu_rows_written = is_bf16_
                                     ? RunColdExperts(cpu_work, group_calls, reinterpret_cast<const BFloat16*>(host_input),
                                                      tile_rows, reinterpret_cast<BFloat16*>(host_result), host_cpu_ids,
                                                      resident || streaming, parallel_runner, skip_experts)
                                     : RunColdExperts(cpu_work, group_calls, reinterpret_cast<const MLFloat16*>(host_input),
                                                      tile_rows, reinterpret_cast<MLFloat16*>(host_result), host_cpu_ids,
                                                      resident || streaming, parallel_runner, skip_experts);
    if (resident || streaming) {
      if (cpu_rows_written > 0) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_ids, host_cpu_ids, SafeInt<size_t>(cpu_rows_written) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_rows, host_result, SafeInt<size_t>(cpu_rows_written) * hidden_size_ * elem_size, cudaMemcpyHostToDevice, stream));
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const __nv_bfloat16*>(cpu_rows), cpu_ids,
              reinterpret_cast<__nv_bfloat16*>(contributions), cpu_rows_written, hidden_size_, tile_rows, static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const half*>(cpu_rows), cpu_ids,
              reinterpret_cast<half*>(contributions), cpu_rows_written, hidden_size_, tile_rows, static_cast<int>(k_), stream));
        }
      }
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(contributions, host_result, SafeInt<size_t>(routes) * hidden_size_ * elem_size, cudaMemcpyHostToDevice, stream));
    }

    if (bias_gpu_ptr) {
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
      }
    }
    if (is_bf16_) {
      ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(contributions),
          reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()) + activation_start,
          bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    } else {
      ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
          reinterpret_cast<const half*>(contributions),
          reinterpret_cast<half*>(output->MutableDataRaw()) + activation_start,
          bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    if (resident_slots_ > 0) {
      last_cold_routes_ = expert_routes;
      sync_routes_pending_ = true;
      if (!resident_phase_scores_ && !continuous_decay_) {
        routes_since_decay_ += routes;
        if (routes_since_decay_ >= resident_decay_routes_) {
          for (auto& score : expert_score_) score *= 0.5f;
          routes_since_decay_ = 0;
        }
      }
      ORT_RETURN_IF_ERROR(PromoteResident(stream, expert_routes, streaming ? stream_promotions_ : 0,
                                          streaming ? stream_list.data() : nullptr,
                                          streaming ? static_cast<int>(stream_list.size()) : 0));
      if (slot_table_dirty_) {
        int* dst_table = CurrentSlotTableDev();
        if (owned_layer_cache_) {
          std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
          owned_layer_cache_->slot_table_dev_ptr = dst_table;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_table, expert_slot_.data(),
                                             SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice, stream));
        slot_table_dirty_ = false;
      }
      if (!streaming || !stream_promotion_reuse_) {
        CUDA_RETURN_IF_ERROR(drain.Synchronize());
      }
    }
    start += tile_rows;
  }

  if (stage_obs_until_run_end) {
    auto inv_rec = std::make_shared<onnxruntime::cuda::QMoEInvocationRecord>();
    inv_rec->layer_index = layer_index_;
    inv_rec->node_name = Info().node().Name();
    inv_rec->main_stream = stream;
    inv_rec->has_main_stream = true;
    inv_rec->commit_observation = [this, staged = std::move(staged_sync_routes), obs_phase_rows, obs_phase_expert_rows]() {
      for (const auto& r : staged) {
        RecordResidentRoute(r.phase, r.expert, r.hit);
      }
      CommitInvocationObservation(obs_phase_rows, obs_phase_expert_rows);
    };
    session_controller_->RegisterInvocation(std::move(inv_rec));
  } else {
    CommitInvocationObservation(obs_phase_rows, obs_phase_expert_rows);
  }
  if (!final_device_done_) {
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&final_device_done_, cudaEventDisableTiming));
  }
  CUDA_RETURN_IF_ERROR(cudaEventRecord(final_device_done_, stream));
  final_device_done_recorded_.store(true, std::memory_order_release);
  CUDA_RETURN_IF_ERROR(drain.Wait());
  return Status::OK();
}

void QMoEOffload::RunAsyncCpu(AsyncTile& tile) const {
  const bool timed = cpu_timing_ && tile.tile_rows <= 4;
  const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  if (timed) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    const int64_t previous = g_last_end_ns.load();
    if (previous > 0) {
      const int64_t gap = now_ns - previous;
      (gap < 5'000'000 ? g_gap_in_ns : g_gap_out_ns).fetch_add(static_cast<uint64_t>(gap));
      (gap < 5'000'000 ? g_gap_in_n : g_gap_out_n).fetch_add(1);
      if (layer_index_ == 0 && gap < 50'000'000) {
        const int64_t s0 = g_step_start_ns.load(std::memory_order_relaxed);
        const int64_t s1 = g_step_end_ns.load(std::memory_order_relaxed);
        if (s0 >= previous && s1 >= s0 && now_ns >= s1) {
          g_t1_ns.fetch_add(static_cast<uint64_t>(s0 - previous));
          g_t2_ns.fetch_add(static_cast<uint64_t>(s1 - s0));
          g_t3_ns.fetch_add(static_cast<uint64_t>(now_ns - s1));
          g_inter_n.fetch_add(1);
        }
      }
    }
  }
  auto& work = *async_work_;
  for (auto& assignments : work.expert_routes) assignments.clear();
  bool valid = true;
  std::string err_msg;
  std::array<int, 5> obs_phase_rows{};
  std::array<std::vector<int>, 5> obs_phase_expert_rows{};

  // Pass 1: Validate all token phases and routed expert indices before mutating any state or statistics.
  for (int token = 0; token < tile.tile_rows && valid; ++token) {
    const int token_phase = tile.has_phases ? static_cast<int>(tile.host_phases[token]) : 1;
    if (token_phase < 0 || token_phase >= 5) {
      valid = false;
      err_msg = "Invalid token_phase=" + std::to_string(token_phase) + " at token " + std::to_string(token);
      break;
    }
    for (int rank = 0; rank < k_ && valid; ++rank) {
      const int route = token * static_cast<int>(k_) + rank;
      const int expert = tile.host_indices[route];
      valid = expert >= 0 && expert < num_experts_;
      if (!valid) {
        err_msg = "Out-of-range expert=" + std::to_string(expert) + " at route " + std::to_string(route);
        break;
      }
      for (int previous = 0; valid && previous < rank; ++previous) {
        valid = tile.host_indices[token * static_cast<int>(k_) + previous] != expert;
      }
      if (!valid) {
        err_msg = "Duplicate routed expert=" + std::to_string(expert) + " at token " + std::to_string(token);
        break;
      }
    }
  }
  if (!valid) {
    async_error_message_ = err_msg;
    async_failed_.store(true, std::memory_order_release);
    if (tile.invocation) tile.invocation->MarkFailed(err_msg);
    return;
  }

  // Pass 2: Record validated routes and classify cold misses on the callback thread (overlapped with GPU GEMV).
  const bool in_warmup = IsInQMoEWarmup();
  const bool track_resident_scores = (resident_slots_ > 0 && resident_promotions_ > 0 && !in_warmup);
  if (track_resident_scores) {
    std::fill(work.expert_invocation_rows.begin(), work.expert_invocation_rows.end(), uint8_t{0});
  }
  for (int token = 0; token < tile.tile_rows; ++token) {
    const int token_phase = tile.has_phases ? static_cast<int>(tile.host_phases[token]) : 1;
    if (tile.counts) ++tile.counts[token_phase];
    if (track_resident_scores) ++obs_phase_rows[token_phase];
    for (int rank = 0; rank < k_; ++rank) {
      const int route = token * static_cast<int>(k_) + rank;
      const int expert = tile.host_indices[route];
      const size_t statistic = static_cast<size_t>(token_phase) * num_experts_ + expert;
      if (tile.route_rows) ++tile.route_rows[statistic];
      if (tile.route_calls) tile.route_calls[statistic] = 1;
      if (resident_slots_ > 0) {
        if (track_resident_scores) {
          const uint8_t prev_rows = work.expert_invocation_rows[expert]++;
          const float weight = (prev_rows == 0) ? 1.0f : marginal_row_weight_;
          RecordResidentRoute(token_phase, expert, expert_slot_[expert] >= 0, weight);
        }
        if (expert_slot_[expert] >= 0) {
          continue;
        }
      }
      work.expert_routes[expert].push_back(route);
    }
  }
  const int threads = cpu_threads_;
  std::chrono::steady_clock::time_point classified;
  if (timed) {
    classified = std::chrono::steady_clock::now();
    uint64_t cold_routes = 0;
    uint64_t cold_experts = 0;
    for (const auto& assignments : work.expert_routes) {
      cold_routes += assignments.size();
      cold_experts += assignments.empty() ? 0 : 1;
    }
    ++timing_calls_;
    timing_routes_ += static_cast<uint64_t>(tile.tile_rows) * k_;
    timing_cold_routes_ += cold_routes;
    timing_cold_experts_ += cold_experts;
  }
  if (tile.timing) tile.timing->Record(tile.trace_call, qmoe::CallbackEvent::ColdExpertsBegin);
  auto pool_runner = [this, threads](ptrdiff_t count, auto& function) {
    SharedSpinPool(threads).ParallelFor(count, function, pool_task_batch_);
  };
  if (is_bf16_) {
    RunColdExperts(work, group_calls_, static_cast<const BFloat16*>(tile.host_input), tile.tile_rows,
                   static_cast<BFloat16*>(tile.host_result), nullptr, false, pool_runner);
  } else {
    RunColdExperts(work, group_calls_, static_cast<const MLFloat16*>(tile.host_input), tile.tile_rows,
                   static_cast<MLFloat16*>(tile.host_result), nullptr, false, pool_runner);
  }
  if (tile.timing) tile.timing->Record(tile.trace_call, qmoe::CallbackEvent::ColdExpertsEnd);
  if (timed) {
    const auto end = std::chrono::steady_clock::now();
    timing_classify_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(classified - start).count());
    timing_cpu_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - classified).count());
    g_last_end_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count());
  }
  if (track_resident_scores && !async_failed_.load(std::memory_order_acquire)) {
    if (continuous_decay_) {
      CommitInvocationObservation(obs_phase_rows, obs_phase_expert_rows, work.expert_invocation_rows.data());
    } else if (!resident_phase_scores_) {
      routes_since_decay_ += tile.tile_rows * static_cast<int>(k_);
      if (routes_since_decay_ >= resident_decay_routes_) {
        for (auto& score : expert_score_) score *= 0.5f;
        routes_since_decay_ = 0;
      }
    }
  }
  if (tile.invocation) tile.invocation->MarkCpuDone();
}

namespace {
std::mutex g_pending_async_tile_releases_mutex;
std::vector<std::function<void()>> g_pending_async_tile_releases;

void RegisterPendingAsyncTileRelease(std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(g_pending_async_tile_releases_mutex);
  g_pending_async_tile_releases.push_back(std::move(fn));
}

void DrainPendingAsyncTileReleases() {
  std::vector<std::function<void()>> fns;
  {
    std::lock_guard<std::mutex> lock(g_pending_async_tile_releases_mutex);
    fns.swap(g_pending_async_tile_releases);
  }
  for (auto& fn : fns) {
    if (fn) fn();
  }
}
}  // namespace

Status QMoEOffload::ComputeSharedAsync(OpKernelContext* context, const Tensor* input, const Tensor* router,
                                       const Tensor* bias, const Tensor* phase, Tensor* output,
                                       int32_t* route_rows, int32_t* route_calls, int32_t* counts,
                                       int tile_rows) const {
  ORT_RETURN_IF(async_failed_.load(std::memory_order_acquire),
                "QMoEOffload async CPU expert path failed earlier: ", async_error_message_);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (async_in_flight_.load(std::memory_order_acquire)) {
    ORT_RETURN_IF(std::chrono::steady_clock::now() > deadline, "QMoEOffload async CPU scratch was never released.");
    std::this_thread::yield();
  }
  if (callback_timing_) callback_timing_->HostCheckpoint();
  const auto trace_call = callback_timing_
                              ? callback_timing_->Begin(static_cast<uint32_t>(tile_rows), qmoe::CallbackTiming::CaptureArmed())
                              : qmoe::CallbackTiming::Call{};
  auto trace = [&](qmoe::CallbackEvent kind, uint64_t detail = 0) {
    return trace_call ? callback_timing_->Record(trace_call, kind, detail) : uint64_t{0};
  };
#if defined(__linux__)
  if (trace_call) {
    char message[128];
    std::snprintf(message, sizeof(message), "qmoe_callback_anchor:%llu:%llu:%llu",
                  static_cast<unsigned long long>(callback_timing_->Pid()),
                  static_cast<unsigned long long>(callback_timing_->Instance()),
                  static_cast<unsigned long long>(trace_call.id));
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = message;
    attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attributes.payload.ullValue = trace(qmoe::CallbackEvent::AnchorBefore);
    nvtxMarkEx(&attributes);
    trace(qmoe::CallbackEvent::AnchorAfter);
  }
#endif
  const auto stream = Stream(context);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
  const bool is_capturing = (capture != cudaStreamCaptureStatusNone);

  if (!is_capturing) {
    if (g_prefill_dma_active.load(std::memory_order_acquire)) {
      FinalizeQMoEPrefillAndSeedDecode(device_id_, stream);
    } else if (layer_index_ == 0 && !IsInQMoEWarmup() &&
               ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_NEW_TURN", 0) == 1) {
#if defined(_WIN32)
      _putenv_s("ORT_QMOE_PREFILL_NEW_TURN", "0");
#else
      setenv("ORT_QMOE_PREFILL_NEW_TURN", "0", 1);
#endif
      FinalizeQMoEPrefillAndSeedDecode(device_id_, stream);
    }
    if (resident_slots_ > 0 && !IsInQMoEWarmup()) {
      const auto& prev_routes = sync_routes_pending_ ? last_cold_routes_ : async_work_->expert_routes;
      ORT_RETURN_IF_ERROR(PromoteResident(stream, prev_routes));
      sync_routes_pending_ = false;
      if (slot_table_dirty_) {
        int* dst_table = CurrentSlotTableDev();
        if (owned_layer_cache_) {
          std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
          owned_layer_cache_->slot_table_dev_ptr = dst_table;
        }
        const int* src_table = expert_slot_.data();
        if (decode_state_ && decode_state_->pinned_slot_table) {
          std::memcpy(decode_state_->pinned_slot_table, expert_slot_.data(),
                      SafeInt<size_t>(num_experts_) * sizeof(int));
          src_table = decode_state_->pinned_slot_table;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_table, src_table,
                                             SafeInt<size_t>(num_experts_) * sizeof(int),
                                             cudaMemcpyHostToDevice, stream));
        slot_table_dirty_ = false;
      }
    }
  }

  // O(1) check on common partial-resident decode path; full scan only when resident_slots_ == num_experts_ or slot 0 is empty.
  bool has_any_resident = false;
  bool all_experts_resident = false;
  if (resident_slots_ > 0 && resident_slab_ != nullptr) {
    if (resident_slots_ < num_experts_ && !slot_expert_.empty() && slot_expert_[0] >= 0) {
      has_any_resident = true;
    } else {
      int active_residents = 0;
      for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] >= 0) ++active_residents;
      }
      has_any_resident = (active_residents > 0);
      all_experts_resident = (active_residents == num_experts_);
    }
  }
  const bool has_diag_outputs = (route_rows != nullptr || route_calls != nullptr || counts != nullptr);
  const bool resident = has_any_resident;
  const bool static_full_resident = all_experts_resident &&
                                    pending_promotions_.empty() && !slot_table_dirty_ &&
                                    !has_diag_outputs && !trace_call && (phase == nullptr);
  const bool use_persistent_decode =
      !static_full_resident &&
      ((tile_rows <= 4 && !has_diag_outputs && !trace_call && (phase == nullptr) &&
        (!bias || fc2_bias_gpu_ != nullptr || bias->Location().device.Type() == OrtDevice::GPU)) ||
       is_capturing);

  const size_t elem_size = sizeof(uint16_t);
  const int routes = SafeInt<int>(tile_rows) * k_;
  // The persistent decode state is shared by every captured graph, so its layout must not depend on the shape being
  // captured; size it for the largest row count that can use it.
  const int persistent_cap_rows = std::max(4, async_max_tokens_);
  const int cap_rows = use_persistent_decode ? std::max(tile_rows, persistent_cap_rows) : tile_rows;
  const int cap_routes = SafeInt<int>(cap_rows) * k_;
  const int route_slots = nvfp4_mode_ ? cap_routes : std::max(max_routes_, kGroupedMoeMinRows);
  const int compute_rows = nvfp4_mode_ ? routes : std::max(routes, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(route_slots) * sizeof(int);
  const size_t contribution_bytes = SafeInt<size_t>(nvfp4_mode_ ? cap_routes : max_routes_) * hidden_size_ * elem_size;
  const size_t resident_row_bytes = SafeInt<size_t>(route_slots) * hidden_size_ * elem_size;
  const size_t nvfp4_inter_bytes = nvfp4_mode_ ? static_cast<size_t>(SafeInt<size_t>(route_slots) * inter_size_ * elem_size) : size_t{0};
  const bool has_bias = (bias != nullptr) || (fc2_bias_gpu_ != nullptr);
  const bool use_doorbell =
      doorbell_enabled_ && use_persistent_decode && resident && !static_full_resident && compute_rows <= 256;
  const bool aux = async_aux_ && resident && !static_full_resident && !use_doorbell;

  size_t gpu_bytes = 0;
  const size_t scale_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t index_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t unit_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t identity_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t contribution_offset = AddSlab(contribution_bytes, gpu_bytes);
  const size_t bias_offset = has_bias ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t hit_ids_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_compact_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_flag_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t permutation_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t gathered_offset = (!nvfp4_mode_ && resident) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t wave_output_offset = resident ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t nvfp4_inter_offset = (nvfp4_mode_ && resident) ? AddSlab(nvfp4_inter_bytes, gpu_bytes) : 0;
  const size_t cpu_rows_offset = resident ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t workspace_offset = (!nvfp4_mode_ && resident) ? AddSlab(resident_workspace_bytes_, gpu_bytes) : 0;

  size_t host_bytes = 0;
  const size_t host_index_offset = AddSlab(SafeInt<size_t>(cap_routes) * sizeof(int), host_bytes);
  const size_t host_phase_offset = AddSlab(cap_rows, host_bytes);
  const size_t host_input_offset = AddSlab(SafeInt<size_t>(cap_rows) * hidden_size_ * elem_size, host_bytes);
  const size_t host_result_offset = AddSlab(SafeInt<size_t>(cap_routes) * hidden_size_ * elem_size, host_bytes);
  ORT_RETURN_IF(gpu_bytes > shared_gpu_limit_ || (!static_full_resident && host_bytes > shared_pinned_limit_),
                "QMoEOffload shared scratch exceeds configured GPU or pinned limits.");

  IAllocatorUniquePtr<uint8_t> scratch;
  std::shared_ptr<bool> completed;
  IAllocatorUniquePtr<uint8_t> pinned;
  uint8_t* base = nullptr;
  uint8_t* pinned_base = nullptr;
  if (use_persistent_decode) {
    ORT_RETURN_IF_ERROR(EnsurePersistentDecodeState(context, gpu_bytes, host_bytes, stream));
    base = decode_state_->scratch.get();
    pinned_base = decode_state_->pinned.get();
    completed = decode_state_->completed;
  } else {
    scratch = GetScratchBuffer<uint8_t>(gpu_bytes, GetComputeStream(context));
    completed = std::make_shared<bool>(false);
    if (!static_full_resident) {
      try {
        pinned = AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed);
      } catch (const std::runtime_error&) {
        if (aux_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(aux_stream_));
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
        DrainPendingAsyncTileReleases();
        pinned = AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed);
      }
    }
    ORT_RETURN_IF_NOT(scratch && (static_full_resident || pinned), "QMoEOffload cannot allocate bounded shared scratch.");
    base = scratch.get();
    pinned_base = pinned ? pinned.get() : nullptr;
  }

  auto* scales = reinterpret_cast<float*>(base + scale_offset);
  auto* indices = reinterpret_cast<int*>(base + index_offset);
  auto* unit_scales = reinterpret_cast<float*>(base + unit_offset);
  auto* identity = reinterpret_cast<int*>(base + identity_offset);
  auto* contributions = base + contribution_offset;
  auto* route_bias = has_bias ? (base + bias_offset) : nullptr;
  auto* host_indices = pinned_base ? reinterpret_cast<int*>(pinned_base + host_index_offset) : nullptr;
  auto* host_phases = pinned_base ? reinterpret_cast<int8_t*>(pinned_base + host_phase_offset) : nullptr;
  auto* host_input = pinned_base ? (pinned_base + host_input_offset) : nullptr;
  auto* host_result = pinned_base ? (pinned_base + host_result_offset) : nullptr;
  auto* hit_ids = reinterpret_cast<int*>(base + hit_ids_offset);
  auto* hit_compact = reinterpret_cast<int*>(base + hit_compact_offset);
  auto* hit_flag = reinterpret_cast<int*>(base + hit_flag_offset);
  auto* permutation = reinterpret_cast<int*>(base + permutation_offset);
  auto* gathered = base + gathered_offset;
  auto* wave_output = base + wave_output_offset;
  auto* nvfp4_inter_buf = base + nvfp4_inter_offset;
  auto* cpu_rows = base + cpu_rows_offset;
  auto* workspace = reinterpret_cast<char*>(base + workspace_offset);

  const void* bias_gpu_ptr = fc2_bias_gpu_ ? fc2_bias_gpu_.get()
                                           : (bias && bias->Location().device.Type() == OrtDevice::GPU ? bias->DataRaw() : nullptr);
  IAllocatorUniquePtr<uint8_t> transient_bias_gpu;
  if (!bias_gpu_ptr && bias && bias->Location().device.Type() == OrtDevice::CPU) {
    transient_bias_gpu = GetScratchBuffer<uint8_t>(bias->SizeInBytes(), GetComputeStream(context));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(transient_bias_gpu.get(), bias->DataRaw(), bias->SizeInBytes(),
                                         cudaMemcpyHostToDevice, stream));
    bias_gpu_ptr = transient_bias_gpu.get();
  }

  std::unique_ptr<AsyncTile> owned;
  bool failed = false;
  bool aux_used = false;
  StreamDrain drain(stream, failed, completed);
  const ck::MOEParallelismConfig parallelism;

  std::shared_ptr<onnxruntime::cuda::QMoEInvocationRecord> invocation_rec;
  if (use_persistent_decode) {
    invocation_rec = decode_state_->invocation_rec;
    invocation_rec->cpu_done = false;
    invocation_rec->completed = false;
    invocation_rec->failed = false;
    invocation_rec->error_message.clear();
    invocation_rec->commit_observation = nullptr;
    invocation_rec->drain_device = nullptr;
    invocation_rec->main_stream = nullptr;
    invocation_rec->has_main_stream = false;
  } else {
    invocation_rec = std::make_shared<onnxruntime::cuda::QMoEInvocationRecord>();
    invocation_rec->layer_index = layer_index_;
    invocation_rec->node_name = Info().node().Name();
  }

  auto enqueue_hits = [&]() -> Status {
    ck::ActivationParams runner_activation(activation_type_);
    runner_activation.alpha = activation_alpha_;
    runner_activation.beta = activation_beta_;
    runner_activation.limit = swiglu_limit_;
    runner_activation.swiglu_fusion = swiglu_fusion_;
    uint8_t* slab = resident_slab_.get();
    if (nvfp4_mode_) {
      const uint8_t* fc1_w = slab + resident_bank_offsets_[0];
      const uint8_t* fc2_w = slab + resident_bank_offsets_[1];
      const uint8_t* fc1_s = slab + resident_bank_offsets_[2];
      const uint8_t* fc2_s = slab + resident_bank_offsets_[3];
      const float* fc1_g = reinterpret_cast<const float*>(slab + resident_bank_offsets_[4]);
      const float* fc2_g = reinterpret_cast<const float*>(slab + resident_bank_offsets_[5]);
      const int fp4_group = mxfp4_mode_ ? 32 : 16;
      const void* fc1_bias_dev = resident_fc1_bias_slab_ ? resident_fc1_bias_slab_.get() : nullptr;
      if (is_bf16_) {
        gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()), fc1_w,
            static_cast<const __nv_bfloat16*>(fc1_bias_dev), fc1_s, fc1_g, nullptr,
            reinterpret_cast<__nv_bfloat16*>(nvfp4_inter_buf), nullptr, hit_compact, physical_slots_,
            routes, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
            false, true, hit_ids, tile_rows, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        gemv::launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
            reinterpret_cast<__nv_bfloat16*>(wave_output), nullptr, hit_compact, physical_slots_,
            routes, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
            false, true, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      } else {
        gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
            reinterpret_cast<const half*>(input->DataRaw()), fc1_w,
            static_cast<const half*>(fc1_bias_dev), fc1_s, fc1_g, nullptr,
            reinterpret_cast<half*>(nvfp4_inter_buf), nullptr, hit_compact, physical_slots_,
            routes, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
            false, true, hit_ids, tile_rows, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        gemv::launch_moe_gemv_fp4_symmetric<half>(
            reinterpret_cast<const half*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
            reinterpret_cast<half*>(wave_output), nullptr, hit_compact, physical_slots_,
            routes, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
            false, true, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      }
      return Status::OK();
    }
    const auto quantization = ck::QuantParams::GroupWise(32, slab + resident_bank_offsets_[2], slab + resident_bank_offsets_[3]);
    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(reinterpret_cast<const half*>(input->DataRaw()), hit_ids,
                                                 reinterpret_cast<half*>(gathered), compute_rows, hidden_size_,
                                                 static_cast<int>(k_), stream));
    runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, hit_compact, unit_scales,
                    slab + resident_bank_offsets_[0], reinterpret_cast<const void*>(slab + resident_bank_offsets_[4]), activation_type_,
                    slab + resident_bank_offsets_[1], nullptr, quantization, compute_rows, hidden_size_, inter_size_,
                    physical_slots_, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  };

  auto record_slot_and_final_events = [&]() -> Status {
    if (!session_controller_ && !is_capturing) {
      if (resident && resident_promotions_ > 0 && spare_slots_ > 0) {
        if (!slot_reader_done_) {
          CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&slot_reader_done_, cudaEventDisableTiming));
        }
        CUDA_RETURN_IF_ERROR(cudaEventRecord(slot_reader_done_, stream));
        slot_reader_done_recorded_ = true;
      }
      if (!final_device_done_) {
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&final_device_done_, cudaEventDisableTiming));
      }
      CUDA_RETURN_IF_ERROR(cudaEventRecord(final_device_done_, stream));
      final_device_done_recorded_.store(true, std::memory_order_release);
    }
    return Status::OK();
  };

  auto enqueue = [&]() -> Status {
    if (is_bf16_) {
      LaunchSoftmaxTopK(reinterpret_cast<const __nv_bfloat16*>(router->DataRaw()), scales, indices, tile_rows, num_experts_,
                        static_cast<int>(k_), normalize_routing_weights_, stream);
    } else {
      LaunchSoftmaxTopK(reinterpret_cast<const half*>(router->DataRaw()), scales, indices, tile_rows, num_experts_,
                        static_cast<int>(k_), normalize_routing_weights_, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    // Statically full, populated, published placement fast path (v3 Section 7.3 / F6):
    // Skip CPU D2H, host callback, and H2D result upload when all E experts are resident on GPU.
    if (static_full_resident) {
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadClassify(
          indices, CurrentSlotTableDev(), routes,
          compute_rows, hit_ids, hit_compact, hit_flag, stream, dummy_slot_index_,
          unit_scales, identity, nvfp4_mode_ ? static_cast<int>(k_) : 0));
      ORT_RETURN_IF_ERROR(enqueue_hits());
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const __nv_bfloat16*>(wave_output),
            reinterpret_cast<const __nv_bfloat16*>(cpu_rows), hit_flag,
            reinterpret_cast<__nv_bfloat16*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const half*>(wave_output),
            reinterpret_cast<const half*>(cpu_rows), hit_flag,
            reinterpret_cast<half*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      }
      if (bias_gpu_ptr) {
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
        }
      }
      if (is_bf16_) {
        ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(contributions),
            reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()),
            bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
            scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
            static_cast<int>(k_), routes, parallelism, false, stream);
      } else {
        ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
            reinterpret_cast<const half*>(contributions),
            reinterpret_cast<half*>(output->MutableDataRaw()),
            bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
            scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
            static_cast<int>(k_), routes, parallelism, false, stream);
      }
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      ORT_RETURN_IF_ERROR(record_slot_and_final_events());
      if (!is_capturing) {
        resident_hit_routes_ += static_cast<uint64_t>(routes);
        resident_total_routes_ += static_cast<uint64_t>(routes);
        phase_hit_routes_[1] += static_cast<uint64_t>(routes);
        phase_total_routes_[1] += static_cast<uint64_t>(routes);
      }
      if (session_controller_) {
        invocation_rec->main_stream = stream;
        invocation_rec->has_main_stream = true;
        session_controller_->RegisterInvocation(invocation_rec);
      }
      return Status::OK();
    }

    if (use_doorbell) {
      AsyncTile& tile = *decode_state_->tile;
      tile.tile_rows = tile_rows;
      tile.host_indices = host_indices;
      tile.host_phases = host_phases;
      tile.has_phases = false;
      tile.host_input = host_input;
      tile.host_result = host_result;
      tile.route_rows = nullptr;
      tile.route_calls = nullptr;
      tile.counts = nullptr;
      tile.staged_for_run_end = (session_controller_ != nullptr && !is_capturing);
      if (!is_capturing) {
        ++decode_state_->launched_seq;
        DoorbellService::Instance().NotifyPending(1);
      }
      const int input_vec4_count =
          static_cast<int>((SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size) / sizeof(uint4));
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadClassify(
          indices, CurrentSlotTableDev(), routes,
          compute_rows, hit_ids, hit_compact, hit_flag, stream, dummy_slot_index_,
          unit_scales, identity, nvfp4_mode_ ? static_cast<int>(k_) : 0,
          host_indices, input->DataRaw(), host_input, input_vec4_count,
          decode_state_->doorbell_seq_dev, decode_state_->doorbell_req_host,
          decode_state_->doorbell_tile_rows_host, tile_rows));
      ORT_RETURN_IF_ERROR(enqueue_hits());
      if (bias_gpu_ptr) {
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
        }
      }
      if (is_bf16_) {
        decode_state_->deferred_wave_output = reinterpret_cast<const __nv_bfloat16*>(wave_output);
        decode_state_->deferred_host_result = reinterpret_cast<const __nv_bfloat16*>(host_result);
        decode_state_->deferred_hit_flag = hit_flag;
        decode_state_->deferred_contributions = reinterpret_cast<__nv_bfloat16*>(contributions);
        decode_state_->deferred_scales = scales;
        decode_state_->deferred_identity = identity;
        decode_state_->deferred_routed_out = reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw());
        decode_state_->deferred_routes = routes;
        decode_state_->deferred_tile_rows = tile_rows;
        auto* bridge = GetQMoESharedExpertBridge();
        const bool defer_to_shared_expert =
            overlap_shared_expert_ && !bias_gpu_ptr && (hidden_size_ & 7) == 0 && k_ <= 16 &&
            bridge != nullptr && bridge->IsRoutedNameRegistered(output0_name_.c_str());
        if (defer_to_shared_expert) {
          bridge->pending = true;
          bridge->stream = stream;
          bridge->routed_output_ptr = output->MutableDataRaw();
          bridge->tile_rows = tile_rows;
          bridge->hidden_size = hidden_size_;
          bridge->ctx = const_cast<QMoEOffload*>(this);
          bridge->launch_deferred = &QMoEOffload::LaunchDeferredMergeStatic;
        } else if (!bias_gpu_ptr && fuse_merge_finalize_ && (hidden_size_ & 7) == 0 && k_ <= 16) {
          ORT_RETURN_IF(LaunchDeferredMerge(stream, nullptr, 0) != 0,
                        "QMoEOffload fused merge+finalize launch failed.");
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
              reinterpret_cast<const __nv_bfloat16*>(wave_output),
              reinterpret_cast<const __nv_bfloat16*>(host_result), hit_flag,
              reinterpret_cast<__nv_bfloat16*>(contributions), routes, hidden_size_,
              tile_rows, static_cast<int>(k_), stream,
              decode_state_->doorbell_seq_dev, decode_state_->doorbell_done_host));
          ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
              reinterpret_cast<const __nv_bfloat16*>(contributions),
              reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()),
              bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
              scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
              static_cast<int>(k_), routes, parallelism, false, stream);
        }
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const half*>(wave_output),
            reinterpret_cast<const half*>(host_result), hit_flag,
            reinterpret_cast<half*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream,
            decode_state_->doorbell_seq_dev, decode_state_->doorbell_done_host));
        ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
            reinterpret_cast<const half*>(contributions),
            reinterpret_cast<half*>(output->MutableDataRaw()),
            bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
            scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
            static_cast<int>(k_), routes, parallelism, false, stream);
      }
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      ORT_RETURN_IF_ERROR(record_slot_and_final_events());
      if (session_controller_) {
        invocation_rec->main_stream = stream;
        invocation_rec->has_main_stream = true;
        session_controller_->RegisterInvocation(invocation_rec);
      }
      return Status::OK();
    }

    cudaStream_t callback_stream = stream;
    if (aux) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!aux_stream_) {
          int least_prio = 0, greatest_prio = 0;
          if (cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess) {
            CUDA_RETURN_IF_ERROR(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, greatest_prio));
          } else {
            CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
          }
          CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&fork_event_, cudaEventDisableTiming));
          CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&join_event_, cudaEventDisableTiming));
        }
      }
      trace(qmoe::CallbackEvent::ForkEventRecordBegin);
      CUDA_RETURN_IF_ERROR(cudaEventRecord(fork_event_, stream));
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(aux_stream_, fork_event_, 0));
      trace(qmoe::CallbackEvent::ForkEventQueued, reinterpret_cast<uintptr_t>(fork_event_));
      aux_used = true;
      callback_stream = aux_stream_;
    }
    trace(qmoe::CallbackEvent::SourceD2HBegin);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices, indices, SafeInt<size_t>(routes) * sizeof(int), cudaMemcpyDeviceToHost, callback_stream));
    if (phase) {
      if (phase->Location().device.Type() == OrtDevice::GPU) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_phases, phase->Data<int8_t>(), tile_rows, cudaMemcpyDeviceToHost, callback_stream));
      } else {
        std::memcpy(host_phases, phase->Data<int8_t>(), tile_rows);
      }
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input, input->DataRaw(),
                                         SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size, cudaMemcpyDeviceToHost, callback_stream));
    trace(qmoe::CallbackEvent::SourceD2HQueued, reinterpret_cast<uintptr_t>(callback_stream));
    if (resident) {
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadClassify(
          indices, CurrentSlotTableDev(), routes,
          compute_rows, hit_ids, hit_compact, hit_flag, stream, dummy_slot_index_,
          unit_scales, identity, nvfp4_mode_ ? static_cast<int>(k_) : 0));
      if (!aux) ORT_RETURN_IF_ERROR(enqueue_hits());
    } else {
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(unit_scales, identity, compute_rows, stream));
    }
    AsyncTile* tile_ptr = nullptr;
    if (use_persistent_decode) {
      tile_ptr = decode_state_->tile.get();
      tile_ptr->node = this;
      tile_ptr->completed = completed;
      tile_ptr->invocation = invocation_rec;
      tile_ptr->tile_rows = tile_rows;
      tile_ptr->host_indices = host_indices;
      tile_ptr->host_phases = host_phases;
      tile_ptr->has_phases = false;
      tile_ptr->host_input = host_input;
      tile_ptr->host_result = host_result;
      tile_ptr->route_rows = nullptr;
      tile_ptr->route_calls = nullptr;
      tile_ptr->counts = nullptr;
      tile_ptr->timing = nullptr;
      tile_ptr->persistent = true;
      tile_ptr->staged_for_run_end = (session_controller_ != nullptr && !is_capturing);
    } else {
      owned = std::make_unique<AsyncTile>();
      owned->node = this;
      owned->completed = completed;
      owned->invocation = invocation_rec;
      owned->tile_rows = tile_rows;
      owned->host_indices = host_indices;
      owned->host_phases = host_phases;
      owned->has_phases = (phase != nullptr);
      owned->host_input = host_input;
      owned->host_result = host_result;
      owned->route_rows = route_rows;
      owned->route_calls = route_calls;
      owned->counts = counts;
      owned->pinned = std::move(pinned);
      owned->timing = trace_call ? callback_timing_.get() : nullptr;
      owned->trace_call = trace_call;
      owned->persistent = false;
      owned->staged_for_run_end = (session_controller_ != nullptr && !is_capturing);
      if (owned->timing) owned->timing->CallbacksQueued(trace_call, 1);
      tile_ptr = owned.get();
    }
    if (!is_capturing) {
      async_in_flight_.store(true, std::memory_order_release);
    }

    if (session_controller_ && !is_capturing) {
      cudaStream_t main_s = stream;
      cudaStream_t aux_s = aux_used ? aux_stream_ : nullptr;
      invocation_rec->drain_device = [this, main_s, aux_s]() -> common::Status {
        if (aux_s) {
          cudaError_t err = cudaStreamSynchronize(aux_s);
          if (err != cudaSuccess) {
            async_in_flight_.store(false, std::memory_order_release);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload aux_stream sync failed: ", cudaGetErrorString(err));
          }
        }
        cudaError_t err = cudaStreamSynchronize(main_s);
        if (err != cudaSuccess) {
          async_in_flight_.store(false, std::memory_order_release);
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload compute stream sync failed: ", cudaGetErrorString(err));
        }
        return common::Status::OK();
      };
      session_controller_->RegisterInvocation(invocation_rec);
    }

    trace(qmoe::CallbackEvent::CallbackEnqueueBegin, reinterpret_cast<uintptr_t>(callback_stream));
    const auto cpu_status = cudaLaunchHostFunc(
        callback_stream,
        [](void* data) {
          auto* tile = static_cast<AsyncTile*>(data);
          const QMoEOffload* node = tile->node;
          if (tile->timing) {
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::CallbackEnter);
            uint64_t phase_mask = 0;
            for (int row = 0; row < tile->tile_rows; ++row) {
              const int p = tile->has_phases ? static_cast<int>(tile->host_phases[row]) : 1;
              if (p >= 0 && p < 5) phase_mask |= uint64_t{1} << p;
            }
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::InputReadyObserved, phase_mask);
          }
          try {
            node->RunAsyncCpu(*tile);
          } catch (const std::exception& ex) {
            node->async_error_message_ = ex.what();
            node->async_failed_.store(true, std::memory_order_release);
            if (tile->invocation) tile->invocation->MarkFailed(ex.what());
          } catch (...) {
            node->async_error_message_ = "Unknown exception in QMoEOffload CPU callback";
            node->async_failed_.store(true, std::memory_order_release);
            if (tile->invocation) tile->invocation->MarkFailed("Unknown exception in QMoEOffload CPU callback");
          }
          if (tile->timing) {
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::CallbackExit);
            tile->timing->CallbackDone(tile->trace_call);
          }
          node->async_in_flight_.store(false, std::memory_order_release);
        },
        tile_ptr);
    trace(qmoe::CallbackEvent::CallbackEnqueueEnd);
    if (cpu_status != cudaSuccess) {
      if (owned && owned->timing) owned->timing->CallbackDone(trace_call);
      async_in_flight_.store(false, std::memory_order_release);
    }
    CUDA_RETURN_IF_ERROR(cpu_status);
    trace(qmoe::CallbackEvent::ResultH2DBegin);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(resident ? static_cast<void*>(cpu_rows) : static_cast<void*>(contributions),
                                         host_result,
                                         SafeInt<size_t>(routes) * hidden_size_ * elem_size,
                                         cudaMemcpyHostToDevice, callback_stream));
    trace(qmoe::CallbackEvent::ResultH2DQueued);
    auto release = [](void* data) {
      std::unique_ptr<AsyncTile> tile(static_cast<AsyncTile*>(data));
      if (tile->timing) {
        tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::ResultCopyReadyObserved);
        tile->timing->CallbackDone(tile->trace_call);
      }
      if (tile->invocation) tile->invocation->MarkCompleted();
      *tile->completed = true;
    };
    if (aux) {
      trace(qmoe::CallbackEvent::JoinEventRecordBegin);
      CUDA_RETURN_IF_ERROR(cudaEventRecord(join_event_, aux_stream_));
      trace(qmoe::CallbackEvent::JoinEventQueued, reinterpret_cast<uintptr_t>(join_event_));
      if (use_persistent_decode) {
        if (session_controller_ && !is_capturing) {
          cudaStream_t aux_s = aux_stream_;
          invocation_rec->main_stream = stream;
          invocation_rec->has_main_stream = true;
          invocation_rec->drain_device = [this, aux_s, invocation_rec]() -> common::Status {
            cudaError_t err = cudaStreamSynchronize(aux_s);
            invocation_rec->MarkCompleted();
            if (err != cudaSuccess) {
              async_in_flight_.store(false, std::memory_order_release);
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload aux_stream sync failed: ", cudaGetErrorString(err));
            }
            return common::Status::OK();
          };
        }
      } else if (session_controller_ && !owned->timing) {
        auto tile_holder = std::make_shared<std::unique_ptr<AsyncTile>>(std::move(owned));
        cudaStream_t aux_s = aux_stream_;
        RegisterPendingAsyncTileRelease([tile_holder]() {
          if (tile_holder && *tile_holder) {
            if ((*tile_holder)->invocation) (*tile_holder)->invocation->MarkCompleted();
            *(*tile_holder)->completed = true;
            tile_holder->reset();
          }
        });
        invocation_rec->main_stream = stream;
        invocation_rec->has_main_stream = true;
        invocation_rec->drain_device = [this, aux_s, tile_holder]() -> common::Status {
          cudaError_t err = cudaStreamSynchronize(aux_s);
          if (tile_holder && *tile_holder) {
            if ((*tile_holder)->invocation) (*tile_holder)->invocation->MarkCompleted();
            *(*tile_holder)->completed = true;
            tile_holder->reset();
          }
          DrainPendingAsyncTileReleases();
          if (err != cudaSuccess) {
            async_in_flight_.store(false, std::memory_order_release);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload aux_stream sync failed: ", cudaGetErrorString(err));
          }
          return common::Status::OK();
        };
      } else {
        if (owned->timing) owned->timing->CallbacksQueued(trace_call, 1);
        const auto release_status = cudaLaunchHostFunc(aux_stream_, release, owned.get());
        if (release_status != cudaSuccess && owned->timing) owned->timing->CallbackDone(trace_call);
        CUDA_RETURN_IF_ERROR(release_status);
        owned.release();
      }
      ORT_RETURN_IF_ERROR(enqueue_hits());
      if (bias_gpu_ptr) {
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
        }
      }
      trace(qmoe::CallbackEvent::MergeWaitBegin);
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, join_event_, 0));
      trace(qmoe::CallbackEvent::MergeWaitQueued);
    }
    if (resident) {
      trace(qmoe::CallbackEvent::MergeEnqueueBegin);
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const __nv_bfloat16*>(wave_output),
            reinterpret_cast<const __nv_bfloat16*>(cpu_rows), hit_flag,
            reinterpret_cast<__nv_bfloat16*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const half*>(wave_output),
            reinterpret_cast<const half*>(cpu_rows), hit_flag,
            reinterpret_cast<half*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      }
      trace(qmoe::CallbackEvent::MergeEnqueueEnd);
    }
    if (bias_gpu_ptr && !aux) {
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
      }
    }
    if (is_bf16_) {
      ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(contributions),
          reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()),
          bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    } else {
      ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
          reinterpret_cast<const half*>(contributions),
          reinterpret_cast<half*>(output->MutableDataRaw()),
          bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    ORT_RETURN_IF_ERROR(record_slot_and_final_events());
    if (!aux && !use_persistent_decode) {
      CUDA_RETURN_IF_ERROR(cudaLaunchHostFunc(stream, release, owned.get()));
      owned.release();
    }
    trace(qmoe::CallbackEvent::CallEnqueueEnd);
    return Status::OK();
  };

  const Status status = enqueue();
  if (!status.IsOK()) {
    if (aux_used) cudaStreamSynchronize(aux_stream_);
    return status;
  }
  drain.Disarm();
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
