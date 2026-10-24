// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/providers/migraphx/migraphx_inc.h"
#include "core/providers/migraphx/migraphx_call.h"

namespace onnxruntime {

const EnumNameMapping<ArenaExtendStrategy> arena_extend_strategy_mapping{
    {ArenaExtendStrategy::kNextPowerOfTwo, "kNextPowerOfTwo"},
    {ArenaExtendStrategy::kSameAsRequested, "kSameAsRequested"},
};

MIGraphXExecutionProviderInfo::MIGraphXExecutionProviderInfo(const ProviderOptions& options) {
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              migraphx_provider_option::kDeviceId,
              [this](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, device_id));
                int num_devices{};
                ORT_RETURN_IF_ERROR(HIP_CALL(hipGetDeviceCount(&num_devices)));
                ORT_RETURN_IF_NOT(
                    0 <= device_id && device_id < num_devices,
                    "Invalid device ID: ", device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalAlloc,
              [this](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                external_alloc = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalFree,
              [this](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                external_free = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kGpuExternalEmptyCache,
              [this](const std::string& value_str) -> Status {
                std::uintptr_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                external_empty_cache = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddValueParser(
              migraphx_provider_option::kModelCacheDir,
              [this](const std::string& value_str) -> Status {
                model_cache_dir = ToPathString(value_str);
                return Status::OK();
              })
          .AddAssignmentToReference(migraphx_provider_option::kFp16Enable, fp16_enable)
          .AddAssignmentToReference(migraphx_provider_option::kBf16Enable, bf16_enable)
          .AddAssignmentToReference(migraphx_provider_option::kFp8Enable, fp8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kInt8Enable, int8_enable)
          .AddAssignmentToReference(migraphx_provider_option::kInt8UseNativeCalibTable, int8_use_native_calibration_table)
          .AddAssignmentToReference(migraphx_provider_option::kInt8CalibTable, int8_calibration_table_name)
          .AddAssignmentToReference(migraphx_provider_option::kExhaustiveTune, exhaustive_tune)
          .AddAssignmentToReference(migraphx_provider_option::kMemLimit, mem_limit)
          .AddAssignmentToEnumReference(migraphx_provider_option::kArenaExtendStrategy, arena_extend_strategy_mapping, arena_extend_strategy)
          .Parse(options));
}

MIGraphXExecutionProviderInfo::MIGraphXExecutionProviderInfo(const OrtMIGraphXProviderOptions& options) noexcept
    : device_id{static_cast<OrtDevice::DeviceId>(options.device_id)},
      fp16_enable{options.migraphx_fp16_enable != 0},
      fp8_enable{options.migraphx_fp8_enable != 0},
      int8_enable{options.migraphx_int8_enable != 0},
      exhaustive_tune{options.migraphx_exhaustive_tune != 0},
      mem_limit{options.migraphx_mem_limit},
      arena_extend_strategy{options.migraphx_arena_extend_strategy} {
}

ProviderOptions MIGraphXExecutionProviderInfo::ToProviderOptions() const {
  return {
      {std::string{migraphx_provider_option::kDeviceId}, MakeStringWithClassicLocale(device_id)},
      {std::string{migraphx_provider_option::kFp16Enable}, MakeStringWithClassicLocale(fp16_enable)},
      {std::string{migraphx_provider_option::kBf16Enable}, MakeStringWithClassicLocale(bf16_enable)},
      {std::string{migraphx_provider_option::kFp8Enable}, MakeStringWithClassicLocale(fp8_enable)},
      {std::string{migraphx_provider_option::kInt8Enable}, MakeStringWithClassicLocale(int8_enable)},
      {std::string{migraphx_provider_option::kInt8CalibTable}, MakeStringWithClassicLocale(int8_calibration_table_name)},
      {std::string{migraphx_provider_option::kInt8UseNativeCalibTable}, MakeStringWithClassicLocale(int8_use_native_calibration_table)},
      {std::string{migraphx_provider_option::kMemLimit}, MakeStringWithClassicLocale(mem_limit)},
      {std::string{migraphx_provider_option::kArenaExtendStrategy}, EnumToName(arena_extend_strategy_mapping, arena_extend_strategy)},
      {std::string{migraphx_provider_option::kExhaustiveTune}, MakeStringWithClassicLocale(exhaustive_tune)},
      {std::string{migraphx_provider_option::kGpuExternalAlloc}, MakeStringWithClassicLocale(external_alloc)},
      {std::string{migraphx_provider_option::kGpuExternalFree}, MakeStringWithClassicLocale(external_free)},
      {std::string{migraphx_provider_option::kGpuExternalEmptyCache}, MakeStringWithClassicLocale(external_empty_cache)},
      {std::string{migraphx_provider_option::kModelCacheDir}, MakeStringWithClassicLocale(model_cache_dir.string())},
  };
}

}  // namespace onnxruntime
