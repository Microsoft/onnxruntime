// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <core/session/onnxruntime_cxx_api.h>
#include <optional>
#include <random>
#include "test_configuration.h"
#include "test_session.h"
#include "utils.h"

#if defined(USE_CUDA) || defined(USE_TENSORRT)
#include <cuda_runtime.h>
#endif

class TestModelInfo;
namespace onnxruntime {
namespace perftest {
class OnnxRuntimeTestSession : public TestSession {
 public:
  OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd, const PerformanceTestConfig& performance_test_config,
                         const TestModelInfo& m);

  void PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) override;

  bool PopulateGeneratedInputTestData(int32_t seed);
  bool PopulateGeneratedMultiShapeInputTestData(
      int32_t seed,
      const std::map<std::string, std::vector<std::vector<int64_t>>>& data_shape_groups);

  std::vector<int64_t> GetLoadedInputShape(size_t test_data_id, size_t input_id) const;
  void SelectTestDataSets(const std::vector<size_t>& selected_ids);
  void SetUseRoundRobin(bool v) { use_round_robin_ = v; }

  ~OnnxRuntimeTestSession();

  RunTiming Run() override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);

 private:
  void CreateAndStoreGeneratedInput(size_t test_data_id, size_t input_idx,
                                    const std::vector<int64_t>& dims,
                                    ONNXTensorElementDataType element_type, int32_t seed);

  // Copies value into allocator_'s memory if a plugin EP allocator was selected. Strings and
  // non-tensor values can't live in device memory, so those are returned unchanged.
  Ort::Value StageInputForPluginEpAllocator(Ort::Value&& value);

  // Stores an already-staged value; used to avoid double-staging in CreateAndStoreGeneratedInput.
  void StoreTestData(size_t test_data_id, size_t input_id, Ort::Value&& value);

  // True if allocator_ is for device-only (i.e., non CPU or host-accessible) memory.
  bool IsAllocatorDeviceOnly() const;

  Ort::Session session_{nullptr};
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  Ort::AllocatorWithDefaultOptions default_allocator_;
  // Note: custom_allocator_, if used, must outlive the `Ort::Value`s allocated with it in test_inputs_ and outputs_.
  // and must be declared before them to ensure it is destroyed after them.
  Ort::Allocator custom_allocator_{nullptr};
  Ort::UnownedAllocator allocator_{default_allocator_};
  std::vector<std::vector<Ort::Value>> test_inputs_;
  std::vector<Ort::Value> outputs_;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<const char*> input_names_;
  std::vector<std::string> input_names_str_;
  const int input_length_;
  std::string provider_name_;
  std::string device_memory_name_;  // Device memory type name to use from the list in allocator.h
  const std::unordered_map<std::string, std::string>& run_config_entries_;
  Ort::Env& env_;
  std::optional<perftest::utils::PluginEpAllocatorSelection> plugin_ep_allocator_selection_;
  bool has_dynamic_output_shapes_ = false;
  // Per-output dynamic-shape flag, aligned with outputs_. Lets Run() reset only the outputs that
  // actually have dynamic shapes instead of discarding every pre-allocated buffer.
  std::vector<bool> is_output_dynamic_;
  // Per-output string-type flag, used to bind string outputs to host memory under IO binding.
  std::vector<bool> is_output_string_;
  std::atomic<size_t> round_robin_counter_{0};
  bool use_round_robin_{false};
  // True when the selected plugin EP device's allocator is used for IO binding (-g).
  bool use_plugin_ep_io_binding_{false};
#if defined(USE_CUDA) || defined(USE_TENSORRT)
  cudaStream_t stream_{nullptr};  // Device stream if required by IO bindings
#endif
  Ort::ArenaCfg cuda_mempool_arena_cfg_{nullptr};
};

}  // namespace perftest
}  // namespace onnxruntime
