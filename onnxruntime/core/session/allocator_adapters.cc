// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "allocator_adapters.h"
#include "core/common/parse_string.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

namespace {
// The ORT API maintains ABI and backward compatibility, allowing applications to be built with an older version
// and run with a newer one. Users may call `RegisterAllocator` with a custom allocator. However, any new
// function pointers introduced in the newer version may contain invalid values, as the older application
// is unaware of them.
// Therefore, it's necessary to check the version value in `OrtAllocatorImplWrappingIAllocator` and
// `IAllocatorImplWrappingOrtAllocator` to ensure compatibility.
constexpr uint32_t kOrtAllocatorReserveMinVersion = 18;
constexpr uint32_t kOrtAllocatorStatsMinVersion = 23;
constexpr uint32_t kOrtAllocatorAllocOnStreamMinVersion = 23;
constexpr uint32_t kOrtAllocatorShrinkMinVersion = 25;

// Shared helper to parse OrtKeyValuePairs stats into AllocatorStats.
// Used by both IAllocatorImplWrappingOrtAllocator and IArenaImplWrappingOrtAllocator.
void GetStatsFromOrtAllocator(OrtAllocator* ort_allocator, AllocatorStats* stats) {
  if (ort_allocator->version >= kOrtAllocatorStatsMinVersion && ort_allocator->GetStats) {
    Ort::UnownedAllocator allocator(ort_allocator);
    Ort::KeyValuePairs kvps = allocator.GetStats();

    std::vector<const char*> keys, values;
    kvps.GetKeyValuePairs(keys, values);

    for (size_t i = 0; i < keys.size(); ++i) {
      int64_t val = 0;
      if (!TryParseStringWithClassicLocale(std::string_view(values[i]), val)) {
        continue;  // skip unparseable entries
      }
      stats->SetFromKeyValue(keys[i], val);
    }
  }
}

}  // namespace

OrtAllocatorImplWrappingIAllocator::OrtAllocatorImplWrappingIAllocator(onnxruntime::AllocatorPtr&& i_allocator)
    : i_allocator_(std::move(i_allocator)) {
  OrtAllocator::version = ORT_API_VERSION;

  OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) {
    return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Alloc(size);
  };

  OrtAllocator::Free = [](OrtAllocator* this_, void* p) {
    static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Free(p);
  };

  OrtAllocator::Info = [](const OrtAllocator* this_) {
    return static_cast<const OrtAllocatorImplWrappingIAllocator*>(this_)->Info();
  };

  if (OrtAllocator::version >= kOrtAllocatorReserveMinVersion) {
    OrtAllocator::Reserve = [](OrtAllocator* this_, size_t size) {
      return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->Reserve(size);
    };
  }

  if (OrtAllocator::version >= kOrtAllocatorStatsMinVersion) {
    OrtAllocator::GetStats = [](const OrtAllocator* this_, OrtKeyValuePairs** stats) noexcept -> OrtStatusPtr {
      API_IMPL_BEGIN
      auto kvp = std::make_unique<OrtKeyValuePairs>();
      const auto& stats_map = static_cast<const OrtAllocatorImplWrappingIAllocator*>(this_)->Stats();
      kvp->CopyFromMap(std::map<std::string, std::string>(stats_map.begin(), stats_map.end()));
      *stats = reinterpret_cast<OrtKeyValuePairs*>(kvp.release());
      return nullptr;
      API_IMPL_END
    };
  }

  if (OrtAllocator::version >= kOrtAllocatorAllocOnStreamMinVersion) {
    OrtAllocator::AllocOnStream = [](OrtAllocator* this_, size_t size, OrtSyncStream* stream) {
      return static_cast<OrtAllocatorImplWrappingIAllocator*>(this_)->AllocOnStream(size, stream);
    };
  }

  // Shrink is not forwarded through the generic adapter — only plugin allocators implement it directly.
  OrtAllocator::Shrink = nullptr;
}

void* OrtAllocatorImplWrappingIAllocator::Alloc(size_t size) {
  return i_allocator_->Alloc(size);
}

void* OrtAllocatorImplWrappingIAllocator::AllocOnStream(size_t size, OrtSyncStream* stream) {
  return i_allocator_->AllocOnStream(size, static_cast<Stream*>(stream));
}

void* OrtAllocatorImplWrappingIAllocator::Reserve(size_t size) {
  return i_allocator_->Reserve(size);
}

void OrtAllocatorImplWrappingIAllocator::Free(void* p) {
  i_allocator_->Free(p);
}

const OrtMemoryInfo* OrtAllocatorImplWrappingIAllocator::Info() const {
  return &i_allocator_->Info();
}

std::unordered_map<std::string, std::string> OrtAllocatorImplWrappingIAllocator::Stats() const {
  const auto* ort_allocator = i_allocator_->GetWrappedOrtAllocator();
  if (ort_allocator != nullptr && ort_allocator->version >= kOrtAllocatorStatsMinVersion && ort_allocator->GetStats) {
    Ort::UnownedAllocator allocator(const_cast<OrtAllocator*>(ort_allocator));
    Ort::KeyValuePairs kvps = allocator.GetStats();
    std::vector<const char*> keys, values;
    kvps.GetKeyValuePairs(keys, values);
    std::unordered_map<std::string, std::string> entries;
    for (size_t index = 0; index < keys.size(); ++index) {
      entries.emplace(keys[index], values[index]);
    }
    return entries;
  }

  AllocatorStats stats{};
  i_allocator_->GetStats(&stats);

  // Allocators which does not implement GetStats() will return empty stats
  std::unordered_map<std::string, std::string> entries;
  if (stats.num_allocs > 0 || stats.bytes_limit != 0) {
    entries.insert_or_assign("Limit", std::to_string(stats.bytes_limit));
    entries.insert_or_assign("InUse", std::to_string(stats.bytes_in_use));
    entries.insert_or_assign("RequestedInUse", std::to_string(stats.bytes_requested_in_use));
    entries.insert_or_assign("TotalAllocated", std::to_string(stats.total_allocated_bytes));
    entries.insert_or_assign("MaxInUse", std::to_string(stats.max_bytes_in_use));
    entries.insert_or_assign("NumAllocs", std::to_string(stats.num_allocs));
    entries.insert_or_assign("NumReserves", std::to_string(stats.num_reserves));
    entries.insert_or_assign("NumArenaExtensions", std::to_string(stats.num_arena_extensions));
    entries.insert_or_assign("NumArenaShrinkages", std::to_string(stats.num_arena_shrinkages));
    entries.insert_or_assign("MaxAllocSize", std::to_string(stats.max_alloc_size));
  }
  return entries;
}

onnxruntime::AllocatorPtr OrtAllocatorImplWrappingIAllocator::GetWrappedIAllocator() {
  return i_allocator_;
}

IAllocatorImplWrappingOrtAllocator::IAllocatorImplWrappingOrtAllocator(OrtAllocator* ort_allocator)
    : IAllocator(*ort_allocator->Info(ort_allocator)) {
  void (*noop_deleter)(OrtAllocator*) = [](OrtAllocator*) {};
  ort_allocator_ = OrtAllocatorUniquePtr(ort_allocator, noop_deleter);
}

IAllocatorImplWrappingOrtAllocator::IAllocatorImplWrappingOrtAllocator(OrtAllocatorUniquePtr ort_allocator)
    : IAllocator(*ort_allocator->Info(ort_allocator.get())), ort_allocator_(std::move(ort_allocator)) {
}

void* IAllocatorImplWrappingOrtAllocator::Alloc(size_t size) {
  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

bool IAllocatorImplWrappingOrtAllocator::IsStreamAware() const {
  return ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream != nullptr;
}

void* IAllocatorImplWrappingOrtAllocator::AllocOnStream(size_t size, Stream* stream) {
  if (ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream) {
    return ort_allocator_->AllocOnStream(ort_allocator_.get(), size, static_cast<OrtSyncStream*>(stream));
  }

  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

void* IAllocatorImplWrappingOrtAllocator::Reserve(size_t size) {
  if (ort_allocator_->version >= kOrtAllocatorReserveMinVersion && ort_allocator_->Reserve) {
    return ort_allocator_->Reserve(ort_allocator_.get(), size);
  }

  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

void IAllocatorImplWrappingOrtAllocator::Free(void* p) {
  return ort_allocator_->Free(ort_allocator_.get(), p);
}

void IAllocatorImplWrappingOrtAllocator::GetStats(AllocatorStats* stats) {
  *stats = {};
  GetStatsFromOrtAllocator(ort_allocator_.get(), stats);
}

// ---------------------------------------------------------------------------
// IArenaImplWrappingOrtAllocator
// ---------------------------------------------------------------------------

IArenaImplWrappingOrtAllocator::IArenaImplWrappingOrtAllocator(OrtAllocatorUniquePtr ort_allocator)
    : IArena(*ort_allocator->Info(ort_allocator.get())), ort_allocator_(std::move(ort_allocator)) {
}

void* IArenaImplWrappingOrtAllocator::Alloc(size_t size) {
  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

void IArenaImplWrappingOrtAllocator::Free(void* p) {
  return ort_allocator_->Free(ort_allocator_.get(), p);
}

void* IArenaImplWrappingOrtAllocator::Reserve(size_t size) {
  if (ort_allocator_->version >= kOrtAllocatorReserveMinVersion && ort_allocator_->Reserve) {
    return ort_allocator_->Reserve(ort_allocator_.get(), size);
  }

  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

bool IArenaImplWrappingOrtAllocator::IsStreamAware() const {
  return ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream != nullptr;
}

void* IArenaImplWrappingOrtAllocator::AllocOnStream(size_t size, Stream* stream) {
  if (ort_allocator_->version >= kOrtAllocatorAllocOnStreamMinVersion && ort_allocator_->AllocOnStream) {
    return ort_allocator_->AllocOnStream(ort_allocator_.get(), size, static_cast<OrtSyncStream*>(stream));
  }

  return ort_allocator_->Alloc(ort_allocator_.get(), size);
}

void IArenaImplWrappingOrtAllocator::GetStats(AllocatorStats* stats) {
  *stats = {};
  GetStatsFromOrtAllocator(ort_allocator_.get(), stats);
}

Status IArenaImplWrappingOrtAllocator::Shrink() {
  if (ort_allocator_->version >= kOrtAllocatorShrinkMinVersion && ort_allocator_->Shrink) {
    return ToStatusAndRelease(ort_allocator_->Shrink(ort_allocator_.get()));
  }
  return Status::OK();
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
ORT_API_STATUS_IMPL(OrtApis::CreateAllocator, const OrtSession* sess,
                    const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  auto* session = reinterpret_cast<const ::onnxruntime::InferenceSession*>(sess);
  auto allocator_ptr = session->GetAllocator(*mem_info);
  if (!allocator_ptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *out = new onnxruntime::OrtAllocatorImplWrappingIAllocator(std::move(allocator_ptr));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateAndRegisterAllocator, _Inout_ OrtEnv* ort_env,
                    _In_ const OrtMemoryInfo* mem_info,
                    _In_ const OrtArenaCfg* arena_cfg) {
  using namespace onnxruntime;
  if (!ort_env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtMemoryInfo is null");
  }

  auto& env = ort_env->GetEnvironment();
  auto st = env.CreateAndRegisterAllocator(*mem_info, arena_cfg);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::RegisterAllocator, _Inout_ OrtEnv* ort_env,
                    _In_ OrtAllocator* allocator) {
  using namespace onnxruntime;
  if (!ort_env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provided allocator is null");
  }

  if (allocator->Info(allocator)->alloc_type == OrtAllocatorType::OrtArenaAllocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Please register the allocator as OrtDeviceAllocator "
                                 "even if the provided allocator has arena logic built-in. "
                                 "OrtArenaAllocator is reserved for internal arena logic based "
                                 "allocators only.");
  }

  auto& env = ort_env->GetEnvironment();
  auto st = env.RegisterAllocator(allocator);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::UnregisterAllocator, _Inout_ OrtEnv* ort_env,
                    _In_ const OrtMemoryInfo* mem_info) {
  using namespace onnxruntime;
  if (!ort_env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provided OrtMemoryInfo is null");
  }

  auto& env = ort_env->GetEnvironment();
  auto st = env.UnregisterAllocator(*mem_info);

  if (!st.IsOK()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, st.ErrorMessage().c_str());
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator) {
  delete static_cast<onnxruntime::OrtAllocatorImpl*>(allocator);
}

ORT_API_STATUS_IMPL(OrtApis::CreateAndRegisterAllocatorV2, _Inout_ OrtEnv* ort_env, _In_ const char* provider_type,
                    _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys) {
  using namespace onnxruntime;
  std::unordered_map<std::string, std::string> options;
  for (size_t i = 0; i != num_keys; i++) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provider options key/value cannot be empty");
    }

    if (strlen(provider_options_keys[i]) > 1024 || strlen(provider_options_values[i]) > 1024) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Maximum string length for a provider options key/value is 1024.");
    }

    options[provider_options_keys[i]] = provider_options_values[i];
  }

  if (!ort_env) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Env is null");
  }

  if (!mem_info) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtMemoryInfo is null");
  }

  auto& env = ort_env->GetEnvironment();
  auto st = env.CreateAndRegisterAllocatorV2(provider_type, *mem_info, options, arena_cfg);
  return onnxruntime::ToOrtStatus(st);
}

ORT_API_STATUS_IMPL(OrtApis::GetSharedAllocator, _In_ OrtEnv* ort_env, _In_ const OrtMemoryInfo* mem_info,
                    _Outptr_result_maybenull_ OrtAllocator** allocator) {
  *allocator = nullptr;

  if (ort_env == nullptr || mem_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtEnv and OrtMemoryInfo must be provided");
  }

  auto& env = ort_env->GetEnvironment();
  auto st = env.GetSharedAllocator(*mem_info, *allocator);
  return onnxruntime::ToOrtStatus(st);
}

ORT_API_STATUS_IMPL(OrtApis::CreateSharedAllocator,
                    [[maybe_unused]] _In_ OrtEnv* ort_env,
                    [[maybe_unused]] _In_ const OrtEpDevice* ep_device,
                    [[maybe_unused]] _In_ OrtDeviceMemoryType mem_type,
                    [[maybe_unused]] _In_ OrtAllocatorType allocator_type,
                    [[maybe_unused]] _In_opt_ const OrtKeyValuePairs* allocator_options,
                    _Outptr_opt_ OrtAllocator** allocator) {
#if !defined(ORT_MINIMAL_BUILD)

  if (ort_env == nullptr || ep_device == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtEnv and OrtEpDevice must be provided");
  }

  auto& env = ort_env->GetEnvironment();
  ORT_API_RETURN_IF_STATUS_NOT_OK(env.CreateSharedAllocator(*ep_device, mem_type, allocator_type, allocator_options,
                                                            allocator));

  return nullptr;
#else
  // there's no support for plugin EPs in a minimal build so you can't get an OrtEpDevice
  *allocator = nullptr;
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "This API in not supported in a minimal build.");
#endif
}

ORT_API_STATUS_IMPL(OrtApis::ReleaseSharedAllocator,
                    [[maybe_unused]] _In_ OrtEnv* ort_env,
                    [[maybe_unused]] _In_ const OrtEpDevice* ep_device,
                    [[maybe_unused]] _In_ OrtDeviceMemoryType mem_type) {
#if !defined(ORT_MINIMAL_BUILD)
  if (ort_env == nullptr || ep_device == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtEnv and OrtEpDevice must be provided");
  }

  auto& env = ort_env->GetEnvironment();
  ORT_API_RETURN_IF_STATUS_NOT_OK(env.ReleaseSharedAllocator(*ep_device, mem_type));

  return nullptr;
#else
  // there's no support for plugin EPs in a minimal build so you can't get an OrtEpDevice
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "This API in not supported in a minimal build.");
#endif
}
