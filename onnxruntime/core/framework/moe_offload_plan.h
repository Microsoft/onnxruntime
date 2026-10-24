// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Portable MoE / QMoE Offload v4 Session Policy, Source Descriptors, Capability Reports, and Execution Plan.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

#ifdef SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"
#else
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/config_options.h"
#include "core/framework/tensor_shape.h"
#endif
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::moe_offload {

enum class OffloadMode : uint8_t {
  kDisabled = 0,
  kAuto = 1,
  kRequired = 2
};

enum class CpuExecutionMode : uint8_t {
  kAuto = 0,
  kEnabled = 1,
  kDisabled = 2
};

enum class PlacementPolicy : uint8_t {
  kStatic = 0,
  kAdaptive = 1
};

struct BudgetSpec {
  bool is_auto{true};
  uint64_t bytes{0};  // Valid when !is_auto; 0 means zero capacity (never "unlimited")
};

struct NodeOverrideSpec {
  std::string graph_scope{"main"};
  std::string node_name;
  std::optional<uint64_t> device_expert_bytes;
  std::optional<int32_t> max_resident_slots;
};

struct MoeExternalInitializerInfo {
  std::filesystem::path file_path;
  uint64_t file_offset{0};
  uint64_t byte_length{0};
};

struct MoeOffloadSessionPolicy {
  OffloadMode mode{OffloadMode::kDisabled};
  CpuExecutionMode cpu_execution{CpuExecutionMode::kAuto};
  PlacementPolicy placement{PlacementPolicy::kAdaptive};

  BudgetSpec device_expert_bytes{true, 0};
  BudgetSpec host_expert_bytes{true, 0};
  uint64_t device_working_bytes{256ULL << 20};  // Default 256 MiB transient GPU workspace cap
  bool device_working_bytes_explicit{false};
  uint64_t host_working_bytes{64ULL << 20};  // Default 64 MiB transient host workspace cap
  uint64_t pinned_bytes{64ULL << 20};        // Default 64 MiB unique pinned/registered host cap
  bool pinned_bytes_explicit{false};         // True when session.moe.offload.pinned_bytes was explicitly set
  uint64_t reserve_device_bytes{0};          // External KV/state/capture/child-session reserve
  std::string budget_group{};                // Optional cross-session budget group name (empty = per-session)

  bool allow_file_backing{true};
  bool popularity_file_explicit{false};
  int32_t cpu_threads{0};            // 0 = auto physical-core discovery
  int32_t async_max_tokens{4};       // Decode vs. prefill streaming token threshold
  int32_t prefill_wave_experts{32};  // Max active cold experts per DMA wave

  std::filesystem::path source_root;
  std::filesystem::path derived_cache_dir;
  std::filesystem::path popularity_file;
  std::vector<NodeOverrideSpec> node_overrides;

  static std::string ToLowerAscii(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
  }

  static std::string_view TrimAscii(std::string_view s) noexcept {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
  }

  static bool TryExtractLayerIndexFromName(std::string_view name, int32_t& out_layer) noexcept {
    constexpr std::array<std::string_view, 3> kPrefixes{"layers.", "layer.", "blocks."};
    for (std::string_view prefix : kPrefixes) {
      size_t search_from = 0;
      while (search_from < name.size()) {
        const size_t pos = name.find(prefix, search_from);
        if (pos == std::string_view::npos) break;
        size_t idx = pos + prefix.size();
        int64_t parsed = 0;
        bool has_digits = false;
        while (idx < name.size() && name[idx] >= '0' && name[idx] <= '9') {
          parsed = parsed * 10 + (name[idx] - '0');
          if (parsed > std::numeric_limits<int32_t>::max()) return false;
          has_digits = true;
          ++idx;
        }
        if (has_digits && (idx == name.size() || (!std::isalnum(static_cast<unsigned char>(name[idx]))))) {
          out_layer = static_cast<int32_t>(parsed);
          return true;
        }
        search_from = pos + prefix.size();
      }
    }
    return false;
  }

  static bool TryParseUint64(std::string_view s, uint64_t& value) noexcept {
    if (s.empty()) return false;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), value, 10);
    return res.ec == std::errc{} && res.ptr == s.data() + s.size();
  }

  static bool TryParseInt32(std::string_view s, int32_t& value) noexcept {
    if (s.empty()) return false;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), value, 10);
    return res.ec == std::errc{} && res.ptr == s.data() + s.size();
  }

  static bool TryParseBudgetSpec(std::string_view s, BudgetSpec& spec) noexcept {
    const std::string lower = ToLowerAscii(s);
    if (lower == "auto") {
      spec.is_auto = true;
      spec.bytes = 0;
      return true;
    }
    uint64_t parsed = 0;
    if (!TryParseUint64(s, parsed)) return false;
    spec.is_auto = false;
    spec.bytes = parsed;
    return true;
  }

  static bool TryParseNodeOverrides(std::string_view raw, std::vector<NodeOverrideSpec>& out_overrides) {
    out_overrides.clear();
    std::string_view trimmed = TrimAscii(raw);
    if (trimmed.empty() || trimmed == "{}") return true;
    // Support semicolon-separated entries: "node_name:max_resident_slots=8,device_expert_bytes=1048576"
    size_t pos = 0;
    while (pos < trimmed.size()) {
      const size_t next_semi = trimmed.find(';', pos);
      std::string_view entry = TrimAscii(trimmed.substr(pos, next_semi - pos));
      pos = (next_semi == std::string_view::npos) ? trimmed.size() : (next_semi + 1);
      if (entry.empty()) continue;
      const size_t colon = entry.find(':');
      if (colon == std::string_view::npos || colon == 0 || colon + 1 >= entry.size()) {
        return false;
      }
      NodeOverrideSpec spec;
      spec.node_name = std::string(TrimAscii(entry.substr(0, colon)));
      if (spec.node_name.empty()) return false;
      std::string_view kvs = TrimAscii(entry.substr(colon + 1));
      size_t kv_pos = 0;
      bool has_field = false;
      while (kv_pos < kvs.size()) {
        const size_t next_comma = kvs.find(',', kv_pos);
        std::string_view kv = TrimAscii(kvs.substr(kv_pos, next_comma - kv_pos));
        kv_pos = (next_comma == std::string_view::npos) ? kvs.size() : (next_comma + 1);
        if (kv.empty()) return false;
        const size_t eq = kv.find('=');
        if (eq == std::string_view::npos || eq == 0 || eq + 1 >= kv.size()) return false;
        const std::string_view k = TrimAscii(kv.substr(0, eq));
        const std::string_view v = TrimAscii(kv.substr(eq + 1));
        if (k == "max_resident_slots") {
          int32_t slots = -1;
          if (!TryParseInt32(v, slots) || slots < 0) return false;
          spec.max_resident_slots = slots;
          has_field = true;
        } else if (k == "device_expert_bytes") {
          uint64_t b = 0;
          if (!TryParseUint64(v, b)) return false;
          spec.device_expert_bytes = b;
          has_field = true;
        } else {
          return false;
        }
      }
      if (!has_field) return false;
      out_overrides.push_back(std::move(spec));
    }
    return true;
  }

  static std::filesystem::path ResolveRelativePath(const std::filesystem::path& base_dir,
                                                   const std::string& raw_path) {
    if (raw_path.empty()) return {};
    std::filesystem::path p(raw_path);
    if (p.is_relative() && !base_dir.empty()) {
      std::error_code ec;
      auto canon = std::filesystem::weakly_canonical(base_dir / p, ec);
      if (!ec && !canon.empty()) return canon;
      return base_dir / p;
    }
    return p;
  }

  static common::Status ParseFromConfigOptions(const ConfigOptions& config_options,
                                               const std::filesystem::path& model_location,
                                               MoeOffloadSessionPolicy& out) noexcept {
    out = MoeOffloadSessionPolicy{};
    std::filesystem::path base_dir;
    if (!model_location.empty()) {
      std::error_code ec;
      if (std::filesystem::is_directory(model_location, ec)) {
        base_dir = model_location;
      } else {
        base_dir = model_location.parent_path();
      }
    }

    const auto& kv_map = config_options.GetConfigOptionsMap();
    for (const auto& [key, val] : kv_map) {
      if (key.rfind("moe.cpu_offload.", 0) == 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Prototype configuration key '", key,
                               "' is discarded and not supported in QMoE Offload v4; use 'session.moe.offload.*'.");
      }
      if (key.rfind("session.moe.offload.", 0) != 0) {
        continue;
      }
      if (key == kOrtSessionOptionsMoeOffloadMode) {
        const std::string v = ToLowerAscii(val);
        if (v == "disabled" || v == "0" || v == "off") {
          out.mode = OffloadMode::kDisabled;
        } else if (v == "auto") {
          out.mode = OffloadMode::kAuto;
        } else if (v == "required" || v == "1" || v == "on") {
          out.mode = OffloadMode::kRequired;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadMode, " value '", val,
                                 "'; expected 'disabled', 'auto', or 'required'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadCpuExecution) {
        const std::string v = ToLowerAscii(val);
        if (v == "auto") {
          out.cpu_execution = CpuExecutionMode::kAuto;
        } else if (v == "enabled" || v == "1") {
          out.cpu_execution = CpuExecutionMode::kEnabled;
        } else if (v == "disabled" || v == "0") {
          out.cpu_execution = CpuExecutionMode::kDisabled;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadCpuExecution, " value '", val,
                                 "'; expected 'auto', 'enabled', or 'disabled'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPlacement) {
        const std::string v = ToLowerAscii(val);
        if (v == "static") {
          out.placement = PlacementPolicy::kStatic;
        } else if (v == "adaptive") {
          out.placement = PlacementPolicy::kAdaptive;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPlacement, " value '", val,
                                 "'; expected 'static' or 'adaptive' (prototype 'basic'/'layered' are not supported).");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadDeviceExpertBytes) {
        if (!TryParseBudgetSpec(val, out.device_expert_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadDeviceExpertBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadHostExpertBytes) {
        if (!TryParseBudgetSpec(val, out.host_expert_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadHostExpertBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadDeviceWorkingBytes) {
        if (!TryParseUint64(val, out.device_working_bytes) || out.device_working_bytes == 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadDeviceWorkingBytes, " value '", val,
                                 "'; expected positive integer bytes.");
        }
        out.device_working_bytes_explicit = true;
      } else if (key == kOrtSessionOptionsMoeOffloadHostWorkingBytes) {
        if (!TryParseUint64(val, out.host_working_bytes) || out.host_working_bytes == 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadHostWorkingBytes, " value '", val,
                                 "'; expected positive integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPinnedBytes) {
        if (ToLowerAscii(val) == "auto") {
          out.pinned_bytes = 64ULL << 20;
          out.pinned_bytes_explicit = false;
        } else if (!TryParseUint64(val, out.pinned_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPinnedBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        } else {
          out.pinned_bytes_explicit = true;
        }
      } else if (key == kOrtSessionOptionsMoeOffloadReserveDeviceBytes) {
        if (ToLowerAscii(val) == "auto") {
          out.reserve_device_bytes = 0;
        } else if (!TryParseUint64(val, out.reserve_device_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadReserveDeviceBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadBudgetGroup) {
        out.budget_group = std::string(TrimAscii(val));
      } else if (key == kOrtSessionOptionsMoeOffloadAllowFileBacking) {
        if (val == "1" || ToLowerAscii(val) == "true") {
          out.allow_file_backing = true;
        } else if (val == "0" || ToLowerAscii(val) == "false") {
          out.allow_file_backing = false;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadAllowFileBacking, " value '", val,
                                 "'; expected '0' or '1'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadCpuThreads) {
        if (!TryParseInt32(val, out.cpu_threads) || out.cpu_threads < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadCpuThreads, " value '", val,
                                 "'; expected non-negative integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadAsyncMaxTokens) {
        if (!TryParseInt32(val, out.async_max_tokens) || out.async_max_tokens < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadAsyncMaxTokens, " value '", val,
                                 "'; expected non-negative integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPrefillWaveExperts) {
        if (!TryParseInt32(val, out.prefill_wave_experts) || out.prefill_wave_experts <= 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPrefillWaveExperts, " value '", val,
                                 "'; expected positive integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPopularityFile) {
        out.popularity_file = ResolveRelativePath(base_dir, val);
        out.popularity_file_explicit = !val.empty();
      } else if (key == kOrtSessionOptionsMoeOffloadSourceRoot) {
        out.source_root = ResolveRelativePath(base_dir, val);
      } else if (key == kOrtSessionOptionsMoeOffloadDerivedCacheDir) {
        out.derived_cache_dir = ResolveRelativePath(base_dir, val);
      } else if (key == kOrtSessionOptionsMoeOffloadNodeOverrides) {
        if (!TryParseNodeOverrides(val, out.node_overrides)) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT,
              "Invalid ", kOrtSessionOptionsMoeOffloadNodeOverrides, " value '", val,
              "'; expected semicolon-separated 'node_name:max_resident_slots=<N>,device_expert_bytes=<B>'.");
        }
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Unknown session.moe.offload.* configuration key: ", key);
      }
    }

    if (out.source_root.empty() && !base_dir.empty()) {
      out.source_root = base_dir;
    }
    return common::Status::OK();
  }
};

enum class ExpertBankKind : uint8_t {
  kFc1Weights = 0,
  kFc2Weights = 1,
  kFc1Scales = 2,
  kFc2Scales = 3,
  kFc1GlobalScale = 4,
  kFc2GlobalScale = 5,
  kFc1Bias = 6,
  kFc2Bias = 7,
  kFc1ZeroPoints = 8,
  kFc2ZeroPoints = 9,
  kCount = 10
};

struct MoeExpertBankSlice {
  ExpertBankKind kind{};
  int input_index{-1};
  std::string initializer_name;
  ONNXTensorElementDataType elem_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  TensorShape full_shape;
  size_t bytes_per_expert{0};
  const void* host_data{nullptr};
  size_t total_bytes{0};
  bool is_file_mapped{false};
  std::filesystem::path external_file_path;
  uint64_t external_file_offset{0};
};

// Returns true only when [data, data + bytes) is a verified read-only regular-file mapping
// whose pages can be discarded with madvise(MADV_DONTNEED) and reloaded from the file on next fault.
inline bool IsReloadableFileMapping(const void* data, size_t bytes) noexcept {
  if (data == nullptr || bytes == 0) return false;
#if defined(__linux__)
  const uintptr_t address = reinterpret_cast<uintptr_t>(data);
  if (address > std::numeric_limits<uintptr_t>::max() - bytes) return false;
  std::ifstream maps("/proc/self/maps");
  if (!maps) return false;
  std::string line;
  while (std::getline(maps, line)) {
    unsigned long long lo = 0, hi = 0, offset = 0, inode = 0;
    unsigned int major_id = 0, minor_id = 0;
    char perms[5]{};
    int path_pos = 0;
    if (std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &lo, &hi, perms,
                    &offset, &major_id, &minor_id, &inode, &path_pos) != 7) {
      continue;
    }
    if (address < lo || address + bytes > hi) continue;
    if (perms[0] != 'r' || perms[1] != '-' || (perms[3] != 's' && perms[3] != 'p') || inode == 0) {
      return false;
    }
    const std::string path = line.substr(static_cast<size_t>(path_pos));
    if (path.empty() || path[0] != '/') return false;
    struct stat st{};
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) ||
        static_cast<uint64_t>(st.st_ino) != inode ||
        major(st.st_dev) != major_id || minor(st.st_dev) != minor_id) {
      return false;
    }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    if (offset > file_size || (address - lo) > file_size - offset ||
        bytes > file_size - offset - (address - lo)) {
      return false;
    }
    return true;
  }
  return false;
#else
  (void)data;
  (void)bytes;
  return false;
#endif
}

// Pipeline-parallel stage and tensor/expert-parallel shard descriptors (TP=1 / EP=1 default).
struct StageExecutionPlacement {
  int32_t stage_id{0};
  int32_t device_id{0};
  int32_t tp_rank{0};
  int32_t tp_world_size{1};
  int32_t ep_rank{0};
  int32_t ep_world_size{1};
};

struct ExpertShardSpec {
  int32_t tp_rank{0};
  int32_t tp_world_size{1};
  int32_t ep_rank{0};
  int32_t ep_world_size{1};
  int32_t global_num_experts{0};
  int32_t local_num_experts{0};
  int32_t expert_id_offset{0};
  int32_t global_hidden_size{0};
  int32_t global_inter_size{0};
  int32_t local_hidden_size{0};
  int32_t local_inter_size{0};
  int32_t inter_shard_offset{0};
  int32_t quant_block_size{0};
  int32_t fc2_bias_owner_rank{0};
};

struct MoeInvocationId {
  uint64_t request_id{0};
  uint64_t step{0};
  uint32_t microbatch_id{0};
  uint64_t generation{0};
};

struct MoeExpertSourceDescriptor {
  std::string graph_scope{"main"};
  std::string stable_node_id;
  std::string op_type;
  std::string quant_type;
  int32_t layer_index{-1};
  int32_t num_experts{0};
  int32_t top_k{0};
  int32_t hidden_size{0};
  int32_t inter_size{0};
  int32_t block_size{0};
  int32_t weight_bits{0};
  bool weights_prepacked{false};
  StageExecutionPlacement placement{};
  ExpertShardSpec shard_spec{};
  std::array<std::optional<MoeExpertBankSlice>, static_cast<size_t>(ExpertBankKind::kCount)> banks;
};

struct MoeBackendCapabilities {
  bool supports_full_gpu_resident{true};
  bool supports_hybrid_doorbell_decode{false};
  bool supports_gpu_streaming_prefill{false};
  bool supports_file_backed_bounded_host_cache{false};
  bool supports_cuda_graph_capture{false};
  bool decode_kernel_skips_negative_slots{true};

  size_t source_bytes_per_expert{0};
  size_t gpu_slot_bytes{0};
  size_t gpu_fixed_table_bytes{0};
  size_t cpu_slot_bytes{0};
  size_t min_host_cache_slots{2};
  size_t max_init_conversion_scratch_bytes{16ULL << 20};
};

struct MoeNodeExecutionPlan {
  int32_t layer_index{0};
  int32_t num_experts{0};
  int32_t resident_slots{0};
  int32_t spare_slots{0};
  int32_t dummy_slots{0};
  int32_t host_cache_slots{0};
  uint64_t host_cache_bytes{0};
  bool use_file_backing{false};
  bool enable_hybrid_cpu_decode{false};
  bool enable_streaming_dma_prefill{false};
  bool enable_background_promotion{false};
  uint64_t admitted_gpu_expert_bytes{0};
  uint64_t admitted_host_expert_bytes{0};
  std::vector<int32_t> initial_resident_experts;
  std::vector<int32_t> ranked_experts;
};

// Whether an offload-aware MoE/QMoE kernel backend exists. Expert initializers are only relocated to host memory
// when it does; the backend integration that implements IMoeOffloadNode enables this.
inline constexpr bool kExpertRelocationBackendAvailable = false;

class IMoeOffloadNode {
 public:
  virtual ~IMoeOffloadNode() = default;
  virtual const MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept = 0;
  virtual const MoeBackendCapabilities& GetBackendCapabilities() const noexcept = 0;
  virtual common::Status InitializeAdmittedPlan(const MoeOffloadSessionPolicy& policy,
                                                const MoeNodeExecutionPlan& plan) = 0;
};

}  // namespace onnxruntime::moe_offload
