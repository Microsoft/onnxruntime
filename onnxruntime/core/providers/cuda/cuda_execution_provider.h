// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include <filesystem>  // NOLINT(build/c++17)
#include <set>
#include <vector>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include <mutex>
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_graph.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/tunable/cuda_tuning_context.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

namespace onnxruntime {

void RunOnUnload(std::function<void()> function);

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  Status Sync() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  Status OnSessionInitializationEnd() override;

  DataLayout GetPreferredLayout() const override;

  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                   std::string_view node_op_type,
                                                   DataLayout target_data_layout) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t PerThreadDefaultCublasHandle() {
    return GetPerThreadContext().CublasHandle();
  }

  cublasLtHandle_t PerThreadCublasLtHandle() {
    return GetPerThreadContext().CublasLtHandle();
  }

  cudnnHandle_t PerThreadDefaultCudnnHandle() {
    return GetPerThreadContext().CudnnHandle();
  }

  cudaStream_t ComputeStream() {
    // this will return the CUDA EP level stream which can differ from the actual compute tasks stream
    // the compute task stream is supplied within OpKernelContext during inference
    return stream_;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  // Replaces the framework fallback used when an EP has no external-data loader. That path
  // materializes a pageable/mmap-backed CPU tensor before GPUDataTransfer copies it to CUDA.
  // This loader instead reads into pinned buffers and copies directly to the CUDA BFC Arena.
  std::unique_ptr<onnxruntime::IExternalDataLoader> GetExternalDataLoader() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& /* graph_optimizer_registry */,
      IResourceAccountant* resource_accountant) const override;

  int GetDeviceId() const override { return info_.device_id; }
  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; };
  int GetCudnnConvAlgo() const { return info_.cudnn_conv_algo_search; }
  bool DoCopyOnDefaultStream() const { return info_.do_copy_in_default_stream; }
  bool GetCudnnConvUseMaxWorkspace() const { return info_.cudnn_conv_use_max_workspace; }
  bool GetCudnnConv1dPadToNc1d() const { return info_.cudnn_conv1d_pad_to_nc1d; }
  bool IsNHWCPreferred() const { return info_.prefer_nhwc; }
  bool IsFuseConvBias() const { return info_.fuse_conv_bias; }
  bool UseTF32() const { return info_.use_tf32; }

  // Attention kernel options parsed from sdpa_kernel cuda provider option.
  const AttentionKernelOptions* GetAttentionKernelOptions() const {
    attention_kernel_options_.InitializeOnce(info_.sdpa_kernel, true, true);
    return &attention_kernel_options_;
  }

  ProviderOptions GetProviderOptions() const override {
    return CUDAExecutionProviderInfo::ToProviderOptions(info_);
  }

  struct CUDAAllocatorParams {
    OrtDevice::DeviceId device_id = 0;
    size_t cuda_mem_threshold = std::numeric_limits<size_t>::max();
    ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
    const CUDAExecutionProviderInfo* provider_info = nullptr;
    const CUDAExecutionProviderExternalAllocatorInfo* external_alloc_info = nullptr;
    const OrtArenaCfg* arena_cfg = nullptr;
    const logging::Logger* logger = nullptr;
    std::shared_ptr<SessionBudgetAccount> budget_account = nullptr;
  };

  static AllocatorPtr CreateCudaAllocator(const CUDAAllocatorParams& cuda_allocator_params);

  static AllocatorPtr CreateCudaPinnedAllocator(const CUDAAllocatorParams& cuda_allocator_params);

  ITuningContext* GetTuningContext() const override;

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  bool ConcurrentRunSupported() const override {
    return !(qmoe_controller_ && qmoe_controller_->HasOffloadNodes());
  }

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(CudaGraphAnnotation_t graph_annotation_id) const override;
  Status ReplayGraph(CudaGraphAnnotation_t graph_annotation_id, bool sync = true) override;
  OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const override {
    return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
  }
  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  // Takes ownership of a host buffer that a kernel staged a device copy from while the
  // stream was capturing.  The copy is a node of the captured graph and re-reads the
  // buffer on every replay, so it must not go back to the allocator and be overwritten
  // by a later run.  Held until the provider is destroyed.
  void RetainBufferForGraphCapture(std::shared_ptr<void> buffer) const;

  // Path of the model this provider partitioned, recorded in GetCapability. Empty for models loaded
  // from bytes, or before partitioning.
  void SetSessionModelPath(std::filesystem::path path) const;
  std::filesystem::path GetSessionModelPath() const;

  const std::shared_ptr<onnxruntime::cuda::QMoESessionController>& GetQMoESessionController() const {
    return qmoe_controller_;
  }

 private:
  mutable std::mutex captured_host_buffers_mutex_;
  mutable std::vector<std::shared_ptr<void>> captured_host_buffers_;
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> qmoe_controller_;
  std::shared_ptr<SessionBudgetAccount> session_budget_account_;

  mutable std::mutex model_path_mutex_;
  mutable std::filesystem::path model_path_;

  CUDAExecutionProviderInfo info_;
  cudaDeviceProp device_prop_;
  bool external_stream_ = false;
  // only used when set user external stream or cuda graph
  cudaStream_t stream_ = nullptr;

  bool use_ep_level_unified_stream_ = false;

  // the tuning context might be altered when calling into a TunableOp
  mutable cuda::tunable::CudaTuningContext tuning_context_;

  // Attention kernel options parsed from sdpa_kernel cuda provider option.
  mutable AttentionKernelOptions attention_kernel_options_;

  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, cudaStream_t stream, const CUDAExecutionProviderInfo& info);
    ~PerThreadContext();
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerThreadContext);

    cublasHandle_t CublasHandle() const {
      return cublas_handle_;
    }

    cudnnHandle_t CudnnHandle() const {
      return cudnn_handle_;
    }

    cublasLtHandle_t CublasLtHandle() const {
      return cublas_lt_handle_;
    }

    bool IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
    bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    CudaGraphAnnotation_t GetCudaGraphAnnotationId(const onnxruntime::RunOptions& run_options) const;
    Status ReplayGraph(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync = true);
    void IncrementRegularRunCountBeforeGraphCapture(CudaGraphAnnotation_t cuda_graph_annotation_id);
    uint64_t LastQMoEGraphGen() const noexcept { return last_qmoe_graph_gen_; }
    void ResetForQMoEGraphGen(uint64_t new_gen) {
      cuda_graph_.Reset();
      graph_id_to_run_count_.clear();
      last_qmoe_graph_gen_ = new_gen;
    }

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    cublasLtHandle_t cublas_lt_handle_ = nullptr;

    // Cuda graph with multi threads will be supported in the future, so cuda_graph_
    // is put under PerThreadContext.
    CUDAGraph cuda_graph_;
    // Map of graph id to regular_run_count_before_graph_capture
    std::unordered_map<CudaGraphAnnotation_t, int> graph_id_to_run_count_;
    uint64_t last_qmoe_graph_gen_ = 0;

    // There is chance that the second regular run allocates GPU memory for causes like:
    // (1) memory pattern is enabled. (2) arena allocation for stream.
    // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
    // to allocate enough memory in Arena before graph capturing.
    const int min_num_runs_before_cuda_graph_capture_ = 2;  // required min regular runs before graph capture for the necessary memory allocations.
  };

  using PerThreadContextMap = std::unordered_map<const CUDAExecutionProvider*, std::weak_ptr<PerThreadContext>>;
  // thread local PerThreadContext cache

  struct ContextCacheHolder {
    ContextCacheHolder() {
      // Keep a weak pointer to the object, if the weak pointer can be locked, then the shared pointer is still around, so we can reset it
      RunOnUnload([&, weak_p_ = std::weak_ptr<PerThreadContextMap>(p)] {
        if (auto lock = weak_p_.lock())
          p.reset();
      });
    }
    std::shared_ptr<PerThreadContextMap> p = std::make_shared<PerThreadContextMap>();
  };

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const ContextCacheHolder per_thread_context_cache;
    return per_thread_context_cache.p;
  }

  struct PerThreadContextState {
    // contexts that are currently active
    std::set<std::shared_ptr<PerThreadContext>, std::owner_less<std::shared_ptr<PerThreadContext>>> active_contexts;
    // contexts available for reuse
    std::vector<std::shared_ptr<PerThreadContext>> retired_context_pool;
    // weak references to thread local caches from which this CUDAExecutionProvider instance's entry should be removed
    // upon destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // synchronizes access to PerThreadContextState members
    std::mutex mutex;
  };

  // The execution provider maintains the PerThreadContexts in this structure.
  // Synchronization is required to update the contained structures.
  // On the other hand, access to an individual PerThreadContext is assumed to be from a single thread at a time,
  // so synchronization is not required for that.
  mutable PerThreadContextState context_state_;

  PerThreadContext& GetPerThreadContext() const;
  void ReleasePerThreadContext() const;
};

}  // namespace onnxruntime
