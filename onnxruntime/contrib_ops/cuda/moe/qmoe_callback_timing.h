// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#endif

namespace onnxruntime::contrib::cuda::qmoe {

enum class CallbackEvent : uint32_t {
  CallBegin,
  AnchorBefore,
  AnchorAfter,
  SourceD2HBegin,
  SourceD2HQueued,
  ForkEventRecordBegin,
  ForkEventQueued,
  CallbackEnqueueBegin,
  CallbackEnqueueEnd,
  CallbackEnter,
  InputReadyObserved,
  ColdExpertsBegin,
  ColdExpertsEnd,
  CallbackExit,
  ResultH2DBegin,
  ResultH2DQueued,
  ResultCopyReadyObserved,
  JoinEventRecordBegin,
  JoinEventQueued,
  MergeWaitBegin,
  MergeWaitQueued,
  MergeEnqueueBegin,
  MergeEnqueueEnd,
  CallEnqueueEnd,
};

inline const char* CallbackEventName(CallbackEvent kind) noexcept {
  static constexpr const char* names[] = {
      "call_begin", "anchor_before", "anchor_after", "source_d2h_begin", "source_d2h_queued",
      "fork_event_record_begin", "fork_event_queued", "callback_enqueue_begin", "callback_enqueue_end", "callback_enter",
      "input_ready_observed", "cold_experts_begin", "cold_experts_end", "callback_exit",
      "result_h2d_begin", "result_h2d_queued", "result_copy_ready_observed", "join_event_record_begin", "join_event_queued",
      "merge_wait_begin", "merge_wait_queued", "merge_enqueue_begin", "merge_enqueue_end", "call_enqueue_end"};
  const auto index = static_cast<uint32_t>(kind);
  return index < sizeof(names) / sizeof(names[0]) ? names[index] : "unknown";
}

// No CUDA, NVTX, allocation, file IO, locks or exceptions in Record/CallbackDone.
// Persist is an ordinary-host-only operation after both existing callbacks finish.
class CallbackTiming final {
 public:
  static constexpr uint64_t kMaxCalls = 4096;
  static constexpr uint64_t kEventsPerCall = 24;
  struct Config {
    int64_t layer = -1;
    uint64_t instance = 0;
    uint64_t first_call = 0;
    uint64_t calls = 32;
    uint32_t max_rows = 4;
    uint64_t capacity = 32 * kEventsPerCall;
    std::string output_prefix;
  };
  struct Call {
    uint64_t id = 0;
    uint64_t ordinal = 0;
    uint32_t rows = 0;
    explicit operator bool() const noexcept { return id != 0; }
  };
  enum class Error : uint32_t { None,
                                Overflow,
                                Clock,
                                Geometry,
                                Pending,
                                Incomplete };

  static void Validate(const Config& config) {
    if (config.layer < 0 || !config.calls || config.calls > kMaxCalls ||
        config.first_call > std::numeric_limits<uint64_t>::max() - config.calls ||
        !config.max_rows || config.max_rows > 4 ||
        !config.capacity || config.capacity > kMaxCalls * kEventsPerCall ||
        config.output_prefix.empty() || config.output_prefix[0] != '/' ||
        config.output_prefix.find_first_of("\"\n\r\\") != std::string::npos) {
      throw std::invalid_argument("Invalid bounded QMoE callback trace configuration");
    }
  }

  explicit CallbackTiming(Config config) : config_(std::move(config)) {
    Validate(config_);
#if defined(__linux__)
    static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                      std::atomic<uint32_t>::is_always_lock_free &&
                      std::atomic<Error>::is_always_lock_free,
                  "Callback tracing requires lock-free atomic slots");
    slots_ = std::make_unique<Slot[]>(config_.capacity);
    call_masks_ = std::make_unique<uint32_t[]>(config_.calls);
    pid_ = static_cast<uint64_t>(getpid());
    path_ = config_.output_prefix + ".pid" + std::to_string(pid_) + ".instance" +
            std::to_string(config_.instance) + ".json";
#else
    throw std::invalid_argument("QMoE callback tracing is Linux-only");
#endif
  }

  static uint64_t EnvironmentNumber(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    if (!*value) throw std::invalid_argument("Empty QMoE callback trace selector");
    uint64_t result = 0;
    for (const char* digit = value; *digit; ++digit) {
      if (*digit < '0' || *digit > '9' ||
          result > (std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(*digit - '0')) / 10) {
        throw std::invalid_argument("Invalid QMoE callback trace numeric selector");
      }
      result = result * 10 + static_cast<uint64_t>(*digit - '0');
    }
    return result;
  }

  static std::unique_ptr<CallbackTiming> FromEnvironment(int64_t layer) {
    if (EnvironmentNumber("ORT_QMOE_CALLBACK_TRACE", 0) == 0) return {};
    if (EnvironmentNumber("ORT_QMOE_CALLBACK_TRACE", 0) != 1)
      throw std::invalid_argument("ORT_QMOE_CALLBACK_TRACE must be 0 or 1");
    const auto selected_layer = EnvironmentNumber("ORT_QMOE_CALLBACK_LAYER", std::numeric_limits<uint64_t>::max());
    if (selected_layer > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      throw std::invalid_argument("ORT_QMOE_CALLBACK_LAYER is required");
    if (layer != static_cast<int64_t>(selected_layer)) return {};
    static std::atomic<uint64_t> next_instance{0};
    const auto instance = next_instance.fetch_add(1, std::memory_order_relaxed);
    if (instance != EnvironmentNumber("ORT_QMOE_CALLBACK_INSTANCE", 0)) return {};
    Config config;
    config.layer = layer;
    config.instance = instance;
    config.first_call = EnvironmentNumber("ORT_QMOE_CALLBACK_FIRST_CALL", 0);
    config.calls = EnvironmentNumber("ORT_QMOE_CALLBACK_CALLS", 32);
    config.max_rows = static_cast<uint32_t>(EnvironmentNumber("ORT_QMOE_CALLBACK_MAX_ROWS", 4));
    if (EnvironmentNumber("ORT_QMOE_CALLBACK_MAX_ROWS", 4) > 4)
      throw std::invalid_argument("QMoE callback trace row bound exceeds MTP decode bound");
    config.capacity = EnvironmentNumber("ORT_QMOE_CALLBACK_EVENTS", config.calls * kEventsPerCall);
    const char* output = std::getenv("ORT_QMOE_CALLBACK_OUTPUT_PREFIX");
    if (output) config.output_prefix = output;
    return std::make_unique<CallbackTiming>(std::move(config));
  }

  static bool CaptureArmed() {
    const auto armed = EnvironmentNumber("ORT_QMOE_CALLBACK_CAPTURE", 0);
    if (armed > 1) throw std::invalid_argument("ORT_QMOE_CALLBACK_CAPTURE must be 0 or 1");
    return armed == 1;
  }

  Call Begin(uint32_t rows, bool armed) noexcept {
    if (!armed || persisted_) return {};
    if (eligible_calls_ >= config_.first_call + config_.calls) return {};
    const auto ordinal = eligible_calls_++;
    if (ordinal < config_.first_call || ordinal - config_.first_call >= config_.calls) return {};
    if (!rows || rows > config_.max_rows) {
      Fail(Error::Geometry);
      return {};
    }
    ++selected_calls_;
    const Call call{ordinal + 1, ordinal, rows};
    Record(call, CallbackEvent::CallBegin);
    return call;
  }

  uint64_t Record(Call call, CallbackEvent kind, uint64_t detail = 0) noexcept {
    if (!call || error_.load(std::memory_order_relaxed) != Error::None) return 0;
    const auto index = used_.fetch_add(1, std::memory_order_relaxed);
    if (index >= config_.capacity) {
      Fail(Error::Overflow);
      return 0;
    }
    const auto now = MonotonicNs();
    if (!now) {
      Fail(Error::Clock);
      return 0;
    }
    auto& slot = slots_[index];
    slot.call = call;
    slot.kind = kind;
    slot.ns = now;
    slot.detail = detail;
    slot.ready.store(1, std::memory_order_release);
    return now;
  }

  void CallbacksQueued(Call call, uint32_t count) noexcept {
    if (call) pending_.fetch_add(count, std::memory_order_relaxed);
  }
  void CallbackDone(Call call) noexcept {
    if (call) pending_.fetch_sub(1, std::memory_order_release);
  }
  bool Quiescent() const noexcept { return pending_.load(std::memory_order_acquire) == 0; }
  Error Failure() const noexcept { return error_.load(std::memory_order_acquire); }
  uint64_t Used() const noexcept { return used_.load(std::memory_order_acquire); }
  uint64_t SelectedCalls() const noexcept { return selected_calls_; }
  uint64_t Pid() const noexcept { return pid_; }
  uint64_t Instance() const noexcept { return config_.instance; }
  const std::string& Path() const noexcept { return path_; }

  void HostCheckpoint() {
    if (Failure() != Error::None) {
      if (Quiescent()) Persist();
      throw std::runtime_error("QMoE callback trace failed; diagnostic is invalid");
    }
    if (!persisted_ && selected_calls_ == config_.calls && Quiescent()) Persist();
  }

  void Persist() {
    if (persisted_) return;
    if (!Quiescent()) throw std::runtime_error("QMoE callback trace callbacks are still pending");
#if defined(__linux__)
    const auto count = std::min(Used(), config_.capacity);
    std::fill_n(call_masks_.get(), config_.calls, uint32_t{0});
    uint64_t recorded = 0;
    for (uint64_t index = 0; index < count; ++index) {
      const auto& slot = slots_[index];
      if (!slot.ready.load(std::memory_order_acquire)) {
        Fail(Error::Incomplete);
        continue;
      }
      ++recorded;
      const auto kind = static_cast<uint32_t>(slot.kind);
      if (kind >= kEventsPerCall || slot.call.ordinal < config_.first_call ||
          slot.call.ordinal - config_.first_call >= config_.calls) {
        Fail(Error::Incomplete);
        continue;
      }
      auto& mask = call_masks_[slot.call.ordinal - config_.first_call];
      const auto bit = uint32_t{1} << kind;
      if (mask & bit) Fail(Error::Incomplete);
      mask |= bit;
    }
    uint64_t complete_calls = 0;
    for (uint64_t index = 0; index < config_.calls; ++index)
      complete_calls += call_masks_[index] == (uint32_t{1} << kEventsPerCall) - 1;
    if (!selected_calls_ || complete_calls != selected_calls_ || recorded != selected_calls_ * kEventsPerCall)
      Fail(Error::Incomplete);
    uint64_t checksum = 14695981039346656037ULL;
    for (uint64_t value : {uint64_t{2}, static_cast<uint64_t>(Failure()), pid_, static_cast<uint64_t>(config_.layer),
                           config_.instance, config_.first_call, config_.calls, selected_calls_,
                           static_cast<uint64_t>(config_.max_rows), config_.capacity, Used(), recorded, complete_calls}) {
      HashWord(checksum, value);
    }
    for (uint64_t index = 0; index < count; ++index) {
      const auto& slot = slots_[index];
      if (!slot.ready.load(std::memory_order_acquire)) continue;
      for (uint64_t value : {index, slot.call.id, slot.call.ordinal, static_cast<uint64_t>(slot.call.rows),
                             static_cast<uint64_t>(slot.kind), slot.ns, slot.detail}) {
        HashWord(checksum, value);
      }
    }
    const int fd = open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) throw std::runtime_error("Cannot exclusively create QMoE callback trace: " + path_);
    FILE* file = fdopen(fd, "w");
    if (!file) {
      close(fd);
      throw std::runtime_error("Cannot open QMoE callback trace stream");
    }
    bool good = std::fprintf(file,
                             "{\"schema\":2,\"clock\":\"CLOCK_MONOTONIC\",\"status\":\"%s\","
                             "\"error\":%u,\"pid\":%llu,\"layer\":%lld,\"instance\":%llu,"
                             "\"first_call\":%llu,\"requested_calls\":%llu,\"selected_calls\":%llu,"
                             "\"max_rows\":%u,\"capacity\":%llu,\"attempted_events\":%llu,"
                             "\"recorded_events\":%llu,\"complete_calls\":%llu,"
                             "\"checksum_algorithm\":\"fnv1a64-le-v1\",\"checksum\":\"%016llx\",\"events\":[",
                             Failure() == Error::None ? "CAPTURED" : "FAIL", static_cast<unsigned>(Failure()),
                             U(pid_), static_cast<long long>(config_.layer), U(config_.instance),
                             U(config_.first_call), U(config_.calls), U(selected_calls_), config_.max_rows,
                             U(config_.capacity), U(Used()), U(recorded), U(complete_calls), U(checksum)) >= 0;
    bool comma = false;
    for (uint64_t index = 0; index < count && good; ++index) {
      const auto& slot = slots_[index];
      if (!slot.ready.load(std::memory_order_acquire)) continue;
      good = std::fprintf(file,
                          "%s{\"sequence\":%llu,\"call\":%llu,\"ordinal\":%llu,\"rows\":%u,"
                          "\"kind\":\"%s\",\"ns\":%llu,\"detail\":%llu}",
                          comma ? "," : "", U(index), U(slot.call.id), U(slot.call.ordinal), slot.call.rows,
                          CallbackEventName(slot.kind), U(slot.ns), U(slot.detail)) >= 0;
      comma = true;
    }
    good = (std::fprintf(file, "]}\n") >= 0) && good;
    good = (std::fflush(file) == 0) && good;
    good = (fsync(fd) == 0) && good;
    good = (std::fclose(file) == 0) && good;
    if (!good) throw std::runtime_error("QMoE callback trace write/flush/close failed: " + path_);
    persisted_ = true;
    if (Failure() != Error::None) throw std::runtime_error("QMoE callback trace failed (failure file persisted)");
#else
    throw std::runtime_error("QMoE callback tracing is Linux-only");
#endif
  }

 private:
  struct Slot {
    std::atomic<uint32_t> ready{0};
    Call call;
    CallbackEvent kind = CallbackEvent::CallBegin;
    uint64_t ns = 0;
    uint64_t detail = 0;
  };
  [[maybe_unused]] static unsigned long long U(uint64_t value) noexcept {
    return static_cast<unsigned long long>(value);
  }
  [[maybe_unused]] static void HashWord(uint64_t& checksum, uint64_t value) noexcept {
    for (int byte = 0; byte < 8; ++byte) {
      checksum = (checksum ^ (value & 0xff)) * 1099511628211ULL;
      value >>= 8;
    }
  }
  static uint64_t MonotonicNs() noexcept {
#if defined(__linux__)
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return static_cast<uint64_t>(value.tv_sec) * 1'000'000'000 + static_cast<uint64_t>(value.tv_nsec);
#else
    return 0;
#endif
  }
  void Fail(Error error) noexcept {
    auto expected = Error::None;
    error_.compare_exchange_strong(expected, error, std::memory_order_release, std::memory_order_relaxed);
  }
  Config config_;
  std::unique_ptr<Slot[]> slots_;
  std::unique_ptr<uint32_t[]> call_masks_;
  std::atomic<uint64_t> used_{0};
  std::atomic<uint32_t> pending_{0};
  std::atomic<Error> error_{Error::None};
  uint64_t eligible_calls_ = 0;
  uint64_t selected_calls_ = 0;
  uint64_t pid_ = 0;
  std::string path_;
  bool persisted_ = false;
};

}  // namespace onnxruntime::contrib::cuda::qmoe
