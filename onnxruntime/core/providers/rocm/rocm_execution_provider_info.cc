// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/rocm_common.h"

#include "core/common/make_string.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace rocm {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kMemLimit = "gpu_mem_limit";
constexpr const char* kArenaExtendStrategy = "arena_extend_strategy";
constexpr const char* kConvExhaustiveSearch = "conv_exhaustive_search";
constexpr const char* kGpuExternalAlloc = "gpu_external_alloc";
constexpr const char* kGpuExternalFree = "gpu_external_free";
}  // namespace provider_option_names
}  // namespace rocm

namespace {
const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};
}  // namespace

ROCMExecutionProviderInfo ROCMExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  ROCMExecutionProviderInfo info{};
  void* alloc = nullptr;
  void* free = nullptr;

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              rocm::provider_option_names::kGpuExternalAlloc,
              [&alloc](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kGpuExternalFree,
              [&free](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              rocm::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                ORT_RETURN_IF_NOT(
                    HIP_CALL(hipGetDeviceCount(&num_devices)),
                    "hipGetDeviceCount() failed.");
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddAssignmentToReference(rocm::provider_option_names::kMemLimit, info.gpu_mem_limit)
          .AddAssignmentToReference(rocm::provider_option_names::kConvExhaustiveSearch, info.miopen_conv_exhaustive_search)
          .AddAssignmentToEnumReference(
              rocm::provider_option_names::kArenaExtendStrategy,
              arena_extend_strategy_mapping, info.arena_extend_strategy)
          .Parse(options));

  ROCMExecutionProviderExternalAllocatorInfo alloc_info{alloc, free};
  info.external_allocator_info = alloc_info;

  return info;
}

ProviderOptions ROCMExecutionProviderInfo::ToProviderOptions(const ROCMExecutionProviderInfo& info) {
  const ProviderOptions options{
      {rocm::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {rocm::provider_option_names::kMemLimit, MakeStringWithClassicLocale(info.gpu_mem_limit)},
      {rocm::provider_option_names::kGpuExternalAlloc, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.alloc))},
      {rocm::provider_option_names::kGpuExternalFree, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.external_allocator_info.free))},
      {rocm::provider_option_names::kConvExhaustiveSearch, MakeStringWithClassicLocale(info.miopen_conv_exhaustive_search)},
      {rocm::provider_option_names::kArenaExtendStrategy,
       EnumToName(arena_extend_strategy_mapping, info.arena_extend_strategy)},
  };

  return options;
}

}  // namespace onnxruntime
