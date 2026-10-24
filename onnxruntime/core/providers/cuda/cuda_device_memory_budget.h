#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace onnxruntime {

class SessionBudgetAccount;
class BudgetGroup;

// Physical device ledger and session/group budget accounts (v3 Section 9.1 / F4).
// Supports:
//  - Transactional Reserve -> Commit / Abort
//  - Lock-free Freeing state (BeginFree -> FinishFree) so cudaFree runs outside allocator locks
//    and failed driver frees retain their charge in quarantine rather than leaking accounting
//  - Safe reconfiguration when all prior allocations/pending/freeing bytes are discharged
//  - Independent per-session limits on the same physical device via SessionBudgetAccount
//  - Explicit BudgetGroup for target/head session sharing.
class CudaDeviceMemoryBudget {
 public:
  struct Stats {
    size_t limit;
    size_t committed;
    size_t pending;
    size_t peak;
    size_t freeing{0};
    size_t quarantined{0};
  };

  class Reservation {
   public:
    Reservation() noexcept : budget_(nullptr), bytes_(0) {}
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept
        : budget_(std::exchange(other.budget_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    Reservation& operator=(Reservation&& other) noexcept {
      if (this != &other) {
        if (budget_) budget_->Abort(bytes_);
        budget_ = std::exchange(other.budget_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
      }
      return *this;
    }
    ~Reservation() {
      if (budget_) budget_->Abort(bytes_);
    }

    void Commit() {
      if (!budget_) throw std::logic_error("CUDA budget reservation already consumed");
      budget_->Commit(bytes_);
      budget_ = nullptr;
      bytes_ = 0;
    }

    size_t ReleaseOwnership() noexcept {
      budget_ = nullptr;
      return std::exchange(bytes_, 0);
    }

    size_t Bytes() const noexcept { return bytes_; }
    bool Active() const noexcept { return budget_ != nullptr; }

   private:
    friend class CudaDeviceMemoryBudget;
    Reservation(CudaDeviceMemoryBudget& budget, size_t bytes) : budget_(&budget), bytes_(bytes) {}
    CudaDeviceMemoryBudget* budget_;
    size_t bytes_;
  };

  CudaDeviceMemoryBudget() = default;
  CudaDeviceMemoryBudget(const CudaDeviceMemoryBudget&) = delete;
  CudaDeviceMemoryBudget& operator=(const CudaDeviceMemoryBudget&) = delete;
  CudaDeviceMemoryBudget(CudaDeviceMemoryBudget&&) = delete;
  CudaDeviceMemoryBudget& operator=(CudaDeviceMemoryBudget&&) = delete;

  void Configure(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_session_limit_ = (limit > 0) ? limit : std::numeric_limits<size_t>::max();
    if (limit == 0) {
      if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0 &&
          !hard_physical_ceiling_) {
        limit_ = std::numeric_limits<size_t>::max();
      }
      return;
    }
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (in_use == 0 && active_owners_ <= 1 && !hard_physical_ceiling_) {
      limit_ = limit;
      return;
    }
    if (!hard_physical_ceiling_ && limit_ == std::numeric_limits<size_t>::max()) {
      limit_ = std::max(limit, in_use);
    }
  }

  void SetExactLimit(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t eff = (limit > 0) ? limit : std::numeric_limits<size_t>::max();
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (eff < in_use) {
      throw std::runtime_error("CUDA budget limit is smaller than currently committed bytes");
    }
    limit_ = eff;
    pending_session_limit_ = eff;
  }

  void SetZeroLimit() {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (in_use > 0) {
      throw std::runtime_error("CUDA budget limit (0) is smaller than currently committed bytes");
    }
    limit_ = 0;
    pending_session_limit_ = 0;
    hard_physical_ceiling_ = true;
  }

  void GetDetailedState(size_t* committed, size_t* pending, size_t* freeing, size_t* quarantined) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed) *committed = committed_;
    if (pending) *pending = pending_;
    if (freeing) *freeing = freeing_;
    if (quarantined) *quarantined = quarantined_;
  }

  void AdoptState(size_t committed, size_t pending, size_t freeing, size_t quarantined) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t add_total = committed + pending + freeing + quarantined;
    const size_t used = committed_ + pending_ + freeing_ + quarantined_;
    const size_t remaining = (limit_ >= used) ? (limit_ - used) : 0;
    if (add_total > remaining) {
      throw std::runtime_error("CUDA device memory budget group exhausted while attaching account: requested " +
                               std::to_string(add_total) + " bytes, remaining " + std::to_string(remaining) + " bytes");
    }
    committed_ += committed;
    pending_ += pending;
    freeing_ += freeing;
    quarantined_ += quarantined;
    peak_ = std::max(peak_, committed_ + freeing_ + quarantined_);
  }

  void DisownState(size_t committed, size_t pending, size_t freeing, size_t quarantined) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed > committed_ || pending > pending_ || freeing > freeing_ || quarantined > quarantined_) {
      std::terminate();
    }
    committed_ -= committed;
    pending_ -= pending;
    freeing_ -= freeing;
    quarantined_ -= quarantined;
  }

  void CommitPending(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > pending_) std::terminate();
    pending_ -= bytes;
    committed_ += bytes;
    peak_ = std::max(peak_, committed_ + freeing_ + quarantined_);
  }

  void AbortPending(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > pending_) std::terminate();
    pending_ -= bytes;
  }

  // Strictly configure a hard physical ceiling across all sessions on the device.
  void ConfigurePhysicalCeiling(size_t physical_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (physical_limit < in_use) {
      throw std::runtime_error("Physical device ceiling is smaller than currently committed bytes");
    }
    limit_ = physical_limit;
    hard_physical_ceiling_ = (physical_limit != std::numeric_limits<size_t>::max());
  }

  size_t ConsumePendingSessionLimit() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::exchange(pending_session_limit_, std::numeric_limits<size_t>::max());
  }

  void ClearPendingSessionLimit() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_session_limit_ = std::numeric_limits<size_t>::max();
  }

  size_t PeekPendingSessionLimit() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_session_limit_;
  }

  void ResetWhenDischarged() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0) {
      limit_ = std::numeric_limits<size_t>::max();
      pending_session_limit_ = std::numeric_limits<size_t>::max();
      hard_physical_ceiling_ = false;
      peak_ = 0;
    }
  }

  void AddOwner() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_owners_;
  }

  void RemoveOwner() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_owners_ > 0) --active_owners_;
    if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0) {
      limit_ = std::numeric_limits<size_t>::max();
      pending_session_limit_ = std::numeric_limits<size_t>::max();
      hard_physical_ceiling_ = false;
    }
  }

  Reservation Reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t used = committed_ + pending_ + freeing_ + quarantined_;
    const size_t remaining = (limit_ >= used) ? (limit_ - used) : 0;
    if (bytes > remaining) {
      throw std::runtime_error("CUDA device memory budget exhausted: requested " + std::to_string(bytes) +
                               " bytes, remaining " + std::to_string(remaining) + " bytes");
    }
    pending_ += bytes;
    return Reservation(*this, bytes);
  }

  // Transition committed bytes into Freeing state before calling cudaFree outside locks (F4).
  void BeginFree(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > committed_) std::terminate();
    committed_ -= bytes;
    freeing_ += bytes;
  }

  // Complete a BeginFree transition after cudaFree returns. If driver_ok is false, retain the charge
  // in quarantined_ rather than dropping accounting (F4).
  void FinishFree(size_t bytes, bool driver_ok) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > freeing_) std::terminate();
    freeing_ -= bytes;
    if (!driver_ok) {
      quarantined_ += bytes;
    }
  }

  void Release(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > committed_) std::terminate();
    committed_ -= bytes;
  }

  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {limit_, committed_ + freeing_ + quarantined_, pending_, peak_, freeing_, quarantined_};
  }

 private:
  void Abort(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ -= bytes;
  }

  void Commit(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ -= bytes;
    committed_ += bytes;
    peak_ = std::max(peak_, committed_ + freeing_ + quarantined_);
  }

  mutable std::mutex mutex_;
  size_t limit_{std::numeric_limits<size_t>::max()};
  size_t pending_session_limit_{std::numeric_limits<size_t>::max()};
  bool hard_physical_ceiling_{false};
  size_t committed_{};
  size_t pending_{};
  size_t freeing_{};
  size_t quarantined_{};
  size_t peak_{};
  size_t active_owners_{};
};

inline std::shared_ptr<CudaDeviceMemoryBudget> GetCudaDeviceMemoryBudgetByOrdinal(int device_id, size_t limit = 0) {
  static std::mutex registry_mutex;
  static std::unordered_map<int, std::shared_ptr<CudaDeviceMemoryBudget>> budgets;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto& budget = budgets[device_id];
  if (!budget) budget = std::make_shared<CudaDeviceMemoryBudget>();
  if (limit > 0) {
    budget->Configure(limit);
  }
  return budget;
}

// Explicit group budget account for related sessions such as target + MTP head (v3 Section 9.1 / F4).
class BudgetGroup {
 public:
  explicit BudgetGroup(std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger,
                       size_t group_limit = std::numeric_limits<size_t>::max())
      : physical_ledger_(std::move(physical_ledger)),
        group_budget_() {
    if (group_limit != 0 && group_limit != std::numeric_limits<size_t>::max()) {
      group_budget_.SetExactLimit(group_limit);
    }
    if (physical_ledger_) physical_ledger_->AddOwner();
  }

  ~BudgetGroup() {
    if (physical_ledger_) physical_ledger_->RemoveOwner();
  }

  BudgetGroup(const BudgetGroup&) = delete;
  BudgetGroup& operator=(const BudgetGroup&) = delete;

  // Repeated joins to an existing group must never enlarge its limit (F02).
  // A finite limit can only be set when the group is currently unbounded or discharged.
  void JoinWithLimit(size_t requested_limit) {
    if (requested_limit == 0 || requested_limit == std::numeric_limits<size_t>::max()) return;
    const auto stats = group_budget_.GetStats();
    if (stats.limit == std::numeric_limits<size_t>::max() || stats.committed + stats.pending == 0) {
      group_budget_.SetExactLimit(std::min(stats.limit, requested_limit));
    }
  }

  CudaDeviceMemoryBudget& GroupBudget() noexcept { return group_budget_; }
  const std::shared_ptr<CudaDeviceMemoryBudget>& PhysicalLedger() const noexcept { return physical_ledger_; }
  CudaDeviceMemoryBudget::Stats GetGroupStats() const { return group_budget_.GetStats(); }

 private:
  std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger_;
  CudaDeviceMemoryBudget group_budget_;
};

// Explicit session/group budget account layered over a physical ledger and optional BudgetGroup (v3 Section 9.1 / F4).
class SessionBudgetAccount {
 public:
  explicit SessionBudgetAccount(std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger,
                                size_t session_limit = std::numeric_limits<size_t>::max(),
                                std::shared_ptr<BudgetGroup> group = nullptr)
      : physical_ledger_(std::move(physical_ledger)),
        group_(std::move(group)),
        local_budget_() {
    if (session_limit != 0 && session_limit != std::numeric_limits<size_t>::max()) {
      local_budget_.SetExactLimit(session_limit);
    }
    if (physical_ledger_) physical_ledger_->AddOwner();
  }

  ~SessionBudgetAccount() {
    if (physical_ledger_) physical_ledger_->RemoveOwner();
  }

  SessionBudgetAccount(const SessionBudgetAccount&) = delete;
  SessionBudgetAccount& operator=(const SessionBudgetAccount&) = delete;

  void ConfigureSessionLimit(size_t session_limit) {
    std::lock_guard<std::mutex> lock(account_mutex_);
    if (session_limit != 0 && session_limit != std::numeric_limits<size_t>::max()) {
      local_budget_.SetExactLimit(session_limit);
    }
  }

  // Transactionally migrate all existing account charges (committed, pending, freeing, quarantined)
  // when attaching or switching a BudgetGroup so subsequent Commit/Abort/BeginFree/FinishFree calls
  // remain balanced across local, group, and physical ledgers.
  void SetBudgetGroup(std::shared_ptr<BudgetGroup> group) {
    std::lock_guard<std::mutex> lock(account_mutex_);
    if (group_ == group) return;
    size_t committed = 0;
    size_t pending = 0;
    size_t freeing = 0;
    size_t quarantined = 0;
    local_budget_.GetDetailedState(&committed, &pending, &freeing, &quarantined);
    if (group) {
      group->GroupBudget().AdoptState(committed, pending, freeing, quarantined);
    }
    if (group_) {
      group_->GroupBudget().DisownState(committed, pending, freeing, quarantined);
    }
    group_ = std::move(group);
  }

  std::shared_ptr<BudgetGroup> GetBudgetGroup() const noexcept {
    std::lock_guard<std::mutex> lock(account_mutex_);
    return group_;
  }
  const std::shared_ptr<CudaDeviceMemoryBudget>& PhysicalLedger() const noexcept { return physical_ledger_; }

  class CombinedReservation {
   public:
    CombinedReservation() noexcept = default;
    CombinedReservation(const CombinedReservation&) = delete;
    CombinedReservation& operator=(const CombinedReservation&) = delete;
    CombinedReservation(CombinedReservation&& other) noexcept
        : account_(std::exchange(other.account_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}
    CombinedReservation& operator=(CombinedReservation&& other) noexcept {
      if (this != &other) {
        if (account_) account_->AbortReservation(bytes_);
        account_ = std::exchange(other.account_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
      }
      return *this;
    }
    ~CombinedReservation() {
      if (account_) account_->AbortReservation(bytes_);
    }

    void Commit() {
      if (!account_) throw std::logic_error("CUDA session budget reservation already consumed");
      account_->CommitReservation(bytes_);
      account_ = nullptr;
      bytes_ = 0;
    }

    size_t Bytes() const noexcept { return bytes_; }
    bool Active() const noexcept { return account_ != nullptr; }

   private:
    friend class SessionBudgetAccount;
    CombinedReservation(SessionBudgetAccount* account, size_t bytes) noexcept
        : account_(account), bytes_(bytes) {}
    SessionBudgetAccount* account_{nullptr};
    size_t bytes_{0};
  };

  CombinedReservation Reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(account_mutex_);
    auto local_res = local_budget_.Reserve(bytes);
    auto group_res = group_ ? group_->GroupBudget().Reserve(bytes) : CudaDeviceMemoryBudget::Reservation{};
    auto phys_res = physical_ledger_ ? physical_ledger_->Reserve(bytes) : CudaDeviceMemoryBudget::Reservation{};
    local_res.ReleaseOwnership();
    if (group_res.Active()) group_res.ReleaseOwnership();
    if (phys_res.Active()) phys_res.ReleaseOwnership();
    return CombinedReservation(this, bytes);
  }

  void BeginFree(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(account_mutex_);
    local_budget_.BeginFree(bytes);
    if (group_) group_->GroupBudget().BeginFree(bytes);
    if (physical_ledger_) physical_ledger_->BeginFree(bytes);
  }

  void FinishFree(size_t bytes, bool driver_ok) noexcept {
    std::lock_guard<std::mutex> lock(account_mutex_);
    local_budget_.FinishFree(bytes, driver_ok);
    if (group_) group_->GroupBudget().FinishFree(bytes, driver_ok);
    if (physical_ledger_) physical_ledger_->FinishFree(bytes, driver_ok);
  }

  CudaDeviceMemoryBudget::Stats GetAccountStats() const {
    std::lock_guard<std::mutex> lock(account_mutex_);
    return local_budget_.GetStats();
  }
  CudaDeviceMemoryBudget::Stats GetGroupStats() const {
    std::lock_guard<std::mutex> lock(account_mutex_);
    return group_ ? group_->GetGroupStats() : local_budget_.GetStats();
  }
  CudaDeviceMemoryBudget::Stats GetPhysicalStats() const {
    std::lock_guard<std::mutex> lock(account_mutex_);
    return physical_ledger_ ? physical_ledger_->GetStats() : local_budget_.GetStats();
  }

 private:
  void CommitReservation(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(account_mutex_);
    local_budget_.CommitPending(bytes);
    if (group_) group_->GroupBudget().CommitPending(bytes);
    if (physical_ledger_) physical_ledger_->CommitPending(bytes);
  }

  void AbortReservation(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(account_mutex_);
    local_budget_.AbortPending(bytes);
    if (group_) group_->GroupBudget().AbortPending(bytes);
    if (physical_ledger_) physical_ledger_->AbortPending(bytes);
  }

  mutable std::mutex account_mutex_;
  std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger_;
  std::shared_ptr<BudgetGroup> group_;
  CudaDeviceMemoryBudget local_budget_;
};

// Host pinned and registered range ledger (v3 Section 9.2).
// Deduplicates registered host ranges across target/head sessions so shared initializers are charged once.
class HostPinnedBudgetLedger {
 public:
  CudaDeviceMemoryBudget::Reservation ReserveAllocated(size_t bytes) {
    return budget_.Reserve(bytes);
  }

  void BeginFreeAllocated(size_t bytes) noexcept {
    budget_.BeginFree(bytes);
  }

  void FinishFreeAllocated(size_t bytes, bool driver_ok = true) noexcept {
    budget_.FinishFree(bytes, driver_ok);
  }

  // Register a host pointer range; returns true if newly registered in the ledger, false if already tracked.
  bool RegisterHostRange(const void* base, size_t bytes) {
    if (!base || bytes == 0) return false;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto& entry = registered_ranges_[base];
    if (entry.ref_count > 0) {
      ++entry.ref_count;
      return false;
    }
    auto res = budget_.Reserve(bytes);
    res.Commit();
    entry.bytes = bytes;
    entry.ref_count = 1;
    return true;
  }

  // Decrements ref count for a registered range; returns true when the last owner releases and cudaHostUnregister should run.
  // Call FinishUnregisterHostRange(bytes, driver_ok) after cudaHostUnregister completes, or pass defer_finish=false.
  bool UnregisterHostRange(const void* base, bool driver_ok = true) noexcept {
    if (!base) return false;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = registered_ranges_.find(base);
    if (it == registered_ranges_.end()) return false;
    if (--it->second.ref_count > 0) {
      return false;
    }
    const size_t bytes = it->second.bytes;
    registered_ranges_.erase(it);
    budget_.BeginFree(bytes);
    budget_.FinishFree(bytes, driver_ok);
    return true;
  }

  bool ReleaseHostRangeOwner(const void* base, size_t* out_bytes = nullptr) noexcept {
    if (!base) return false;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = registered_ranges_.find(base);
    if (it == registered_ranges_.end()) return false;
    if (--it->second.ref_count > 0) {
      return false;
    }
    const size_t bytes = it->second.bytes;
    registered_ranges_.erase(it);
    budget_.BeginFree(bytes);
    if (out_bytes) *out_bytes = bytes;
    return true;
  }

  void FinishUnregisterHostRange(size_t bytes, bool driver_ok) noexcept {
    if (bytes == 0) return;
    budget_.FinishFree(bytes, driver_ok);
  }

  void QuarantineHostRange(const void* base) noexcept {
    if (!base) return;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = registered_ranges_.find(base);
    if (it == registered_ranges_.end()) return;
    const size_t bytes = it->second.bytes;
    registered_ranges_.erase(it);
    budget_.BeginFree(bytes);
    budget_.FinishFree(bytes, false);
  }

  CudaDeviceMemoryBudget::Stats GetStats() const { return budget_.GetStats(); }

 private:
  struct RangeEntry {
    size_t bytes = 0;
    size_t ref_count = 0;
  };
  mutable std::mutex reg_mutex_;
  std::unordered_map<const void*, RangeEntry> registered_ranges_;
  CudaDeviceMemoryBudget budget_;
};

inline std::shared_ptr<HostPinnedBudgetLedger> GetCudaHostPinnedMemoryBudget() {
  static auto ledger = std::make_shared<HostPinnedBudgetLedger>();
  return ledger;
}

inline std::shared_ptr<BudgetGroup> GetCudaBudgetGroupByName(int device_id,
                                                             const std::string& group_name,
                                                             size_t group_limit = 0) {
  static std::mutex group_mutex;
  static std::unordered_map<std::string, std::shared_ptr<BudgetGroup>> groups;
  const std::string key = std::to_string(device_id) + ":" + (group_name.empty() ? "default" : group_name);
  std::lock_guard<std::mutex> lock(group_mutex);
  auto& g = groups[key];
  if (!g) {
    g = std::make_shared<BudgetGroup>(
        GetCudaDeviceMemoryBudgetByOrdinal(device_id),
        group_limit > 0 ? group_limit : std::numeric_limits<size_t>::max());
  } else if (group_limit > 0) {
    g->JoinWithLimit(group_limit);
  }
  return g;
}

}  // namespace onnxruntime
