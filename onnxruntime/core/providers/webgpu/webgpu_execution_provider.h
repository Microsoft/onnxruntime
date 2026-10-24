// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/session_buffer_pool.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

struct pthreadpool;
namespace onnxruntime {
#if defined(ORT_USE_EP_API_ADAPTERS)
namespace ep::adapter {
struct Logger;
}
#endif

namespace webgpu {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class WebGpuContext;
class WebGpuProfiler;
class GpuBufferAllocator;
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
class D3D12ImportedBufferRegistry;
#endif

struct CapturedCommand;

// The actual implementation of kernel registration.
std::shared_ptr<KernelRegistry> GetKernelRegistry(bool enable_graph_capture, bool enable_int64);
}  // namespace webgpu

struct WebGpuExecutionProviderConfig {
  DataLayout data_layout{DataLayout::NHWC};  // preferred layout is NHWC by default
  bool enable_graph_capture{false};          // graph capture feature is disabled by default
  webgpu::WeightLoadAccelerationMode weight_load_acceleration_mode{
      webgpu::WeightLoadAccelerationMode::Off};
  bool enable_uma_weight_loading{true};
  bool enable_pix_capture{false};                // PIX capture is disabled by default
  bool enable_int64{false};                      // int64 ops are not enabled by default
  uint32_t multi_rotary_cache_concat_offset{0};  // offset for concatenated multi rotary cache (0 = disabled)
  // Number of generations of buffers to retain in the per-session pool for reuse
  // across captured-graph lifetimes. 0 disables pooling. Default 1 caches one
  // generator's worth of intermediate buffers.
  size_t session_buffer_pool_generations{1};
  uint32_t kv_cache_quantization_bits{0};  // KV cache quantization bits (0 = off, 4 = 4-bit)
  // Accumulate MatMul, Gemm and MatMulNBits dot products in f32 rather than in the output element type.
  // This is the single line that decides the shipped default for the
  // "enableMatmulFp32Accumulation" provider option.
  bool enable_matmul_fp32_accumulation{false};
  std::optional<webgpu::MatMulAlgorithm> forced_matmul_algorithm;
  std::vector<std::string> force_cpu_node_names{};
};

class WebGpuExecutionProvider : public IExecutionProvider {
 public:
  WebGpuExecutionProvider(int context_id, webgpu::WebGpuContext& context, WebGpuExecutionProviderConfig&& config);
  ~WebGpuExecutionProvider() override;

  inline auto GetKernelRegistryImpl() const {
    return webgpu::GetKernelRegistry(enable_graph_capture_, enable_int64_);
  }

#if !defined(ORT_USE_EP_API_ADAPTERS)
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& /*kernel_lookup*/,
      const GraphOptimizerRegistry& /* graph_optimizer_registry */,
      IResourceAccountant* /* resource_accountant */) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return GetKernelRegistryImpl();
  }
#endif
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
#if defined(__wasm__) || (defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING))
  std::unique_ptr<onnxruntime::IExternalDataLoader> GetExternalDataLoader() const override;
#endif

  DataLayout GetPreferredLayout() const override { return preferred_data_layout_; }

  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                   std::string_view node_op_type,
                                                   DataLayout target_data_layout) const override;

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // A Session owns one command recording timeline, so Run calls on that Session must be serialized.
  bool ConcurrentRunSupported() const override { return false; }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;
  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;
  bool IsRunActive() const { return run_active_.load(); }

  // WebGPU EP reuses the Device ID as the key to get the WebGpuContext instance.
  int GetDeviceId() const override { return context_id_; }

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(int graph_annotation_id) const override;
  Status ReplayGraph(int graph_annotation_id, bool sync = true) override;
  Status ReleaseCapturedGraph(int graph_annotation_id) override;
  OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const override {
    return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
  }
  webgpu::BufferManager& BufferManager() const;
  webgpu::BufferManager& InitializerBufferManager() const;
  webgpu::CommandRecordingState& Recording() const { return *recording_; }
  AllocatorPtr PrepackAllocator() const { return prepack_allocator_; }
  std::span<const std::string> GetForceCpuNodeNames() const { return force_cpu_node_names_; }
  uint32_t MultiRotaryCacheConcatOffset() const { return multi_rotary_cache_concat_offset_; }
  uint32_t KvCacheQuantizationBits() const { return kv_cache_quantization_bits_; }
  bool KvCacheQuantizationEnabled() const { return kv_cache_quantization_bits_ != 0; }
  bool EnableMatmulFp32Accumulation() const { return enable_matmul_fp32_accumulation_; }
  std::optional<webgpu::MatMulAlgorithm> ForcedMatMulAlgorithm() const { return forced_matmul_algorithm_; }

#if defined(ORT_USE_EP_API_ADAPTERS)
  onnxruntime::ep::adapter::Logger& GetEpLogger() const;
  void SetEpLogger(const OrtLogger* logger);
#endif

 private:
  bool IsGraphCaptureAllowed() const;
  void IncrementRegularRunCountBeforeGraphCapture();

  int context_id_;
  webgpu::WebGpuContext& context_;
  webgpu::WebGpuProfiler* session_profiler_{nullptr};
  DataLayout preferred_data_layout_;
  std::vector<std::string> force_cpu_node_names_;
  bool enable_graph_capture_ = false;
  bool graph_buffer_mgr_active_ = false;
  std::atomic<bool> run_active_{false};
  bool enable_int64_ = false;
  uint32_t multi_rotary_cache_concat_offset_ = 0;
  uint32_t kv_cache_quantization_bits_ = 0;
  bool enable_matmul_fp32_accumulation_ = false;
  std::optional<webgpu::MatMulAlgorithm> forced_matmul_algorithm_;
#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
  webgpu::WeightLoadAccelerationMode weight_load_acceleration_mode_{
      webgpu::WeightLoadAccelerationMode::Off};
  bool enable_uma_weight_loading_{true};
  std::shared_ptr<webgpu::D3D12ImportedBufferRegistry> accelerated_initializer_state_;
  AllocatorPtr accelerated_initializer_allocator_;
#endif
  std::unordered_map<int, int> graph_id_to_run_count_;
  // Required regular runs before graph capture for any necessary allocations.
  const int min_num_runs_before_graph_capture_ = 0;
  int current_graph_annotation_id_ = 0;

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  std::unique_ptr<WebGpuPIXFrameGenerator> pix_frame_generator_ = nullptr;
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

  // Command recording is per session and is passed separately from the context-level BufferManagers.
  std::unique_ptr<webgpu::CommandRecordingState> recording_;

  // Per-graph buffer managers keyed by annotation ID.
  // Each captured graph gets its own buffer manager so that buffer caches
  // are isolated between different generators.
  std::unordered_map<int, std::unique_ptr<webgpu::BufferManager>> per_graph_buffer_mgrs_;

  // Per-session pool of buffers donated by retired per-graph BufferManagers,
  // seeded into new per-graph BufferManagers to avoid device allocations for
  // identically-shaped intermediate tensors across generators.
  std::unique_ptr<webgpu::SessionBufferPool> session_buffer_pool_;

  // Store captured commands per graph annotation ID
  std::unordered_map<int, std::vector<webgpu::CapturedCommand>> captured_graphs_;
  // Track which graph annotation IDs have completed capture
  std::unordered_set<int> captured_graph_ids_;

  // Allocator for prepacked weights (uses buffers without mapping)
  AllocatorPtr prepack_allocator_;

#if defined(ORT_USE_EP_API_ADAPTERS)
  std::unique_ptr<onnxruntime::ep::adapter::Logger> ep_logger_;
#endif
};

}  // namespace onnxruntime
