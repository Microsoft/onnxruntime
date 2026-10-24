// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"
#include "cuda_graph_plugin.h"
#include "cuda_profiler_plugin.h"
#include "ep/adapters.h"

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace cuda_plugin {

class CudaEpFactory;

/// CUDA execution provider implementation using public OrtEp interface.
class CudaEp : public onnxruntime::ep::adapter::Ep {
 public:
  /// Configuration parameters for the CUDA EP, parsed from session options.
  struct Config {
    bool prefer_nhwc = false;                        ///< Use NHWC data layout when available.
    bool use_tf32 = true;                            ///< Enable TF32 math on Ampere+ GPUs.
    int device_id = 0;                               ///< CUDA device ordinal.
    int cudnn_conv_algo = 0;                         ///< cuDNN convolution algorithm selection.
    bool cudnn_conv_use_max_workspace = true;        ///< Use maximum workspace for cuDNN conv algo search.
    bool cudnn_conv1d_pad_to_nc1d = false;           ///< Pad 1D convolutions to NC1D format.
    bool enable_cudnn = true;                        ///< Enable runtime loading and use of cuDNN-backed kernels.
    bool fuse_conv_bias = false;                     ///< Enable cuDNN frontend conv+bias fusion.
    int sdpa_kernel = 0;                             ///< Attention backend bitmask override.
    bool enable_cuda_graph = false;                  ///< Enable CUDA graph capture and replay.
    int min_num_runs_before_cuda_graph_capture = 2;  ///< Warm-up runs before graph capture begins.
    bool has_user_compute_stream = false;            ///< Whether user provided an external CUDA stream.
    void* user_compute_stream = nullptr;             ///< User-provided CUDA stream (cudaStream_t cast to void*).
    bool do_copy_in_default_stream = true;           ///< Use default stream for H2D/D2H copies.
    bool use_ep_level_unified_stream = false;        ///< Force all ops to share one stream (no concurrency).
    void* external_alloc = nullptr;                  ///< External GPU memory allocation function pointer.
    void* external_free = nullptr;                   ///< External GPU memory deallocation function pointer.
    void* external_empty_cache = nullptr;            ///< External GPU memory cache-clear function pointer.
    size_t device_mem_limit = 0;                     ///< Session/device allocation budget in bytes (0 = unlimited).
    std::shared_ptr<SessionBudgetAccount> budget_account;
  };

  CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~CudaEp();

  const char* GetEpName() const { return name_.c_str(); }
  const Config& GetConfig() const { return config_; }

 private:
  // OrtEp callback implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr, const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      OrtEp* this_ptr,
      const OrtKernelRegistry** kernel_registry) noexcept;

  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
      OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
      OrtEp* this_ptr, const char* domain, const char* op_type,
      OrtEpDataLayout target_data_layout, int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEp* this_ptr,
      const OrtMemoryInfo* memory_info,
      OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL SyncImpl(OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL IsConcurrentRunSupportedImpl(
      OrtEp* this_ptr, bool* is_supported) noexcept;

  // CUDA Graph callback implementations
  static OrtStatus* ORT_API_CALL OnSessionInitializationEndImpl(
      OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL OnRunStartImpl(
      OrtEp* this_ptr, const OrtRunOptions* run_options) noexcept;

  static OrtStatus* ORT_API_CALL OnRunEndImpl(
      OrtEp* this_ptr, const OrtRunOptions* run_options, bool sync_stream) noexcept;

  static bool ORT_API_CALL IsGraphCaptureEnabledImpl(const OrtEp* this_ptr) noexcept;

  static bool ORT_API_CALL IsGraphCapturedImpl(const OrtEp* this_ptr,
                                               int graph_annotation_id) noexcept;

  static OrtStatus* ORT_API_CALL ReplayGraphImpl(OrtEp* this_ptr,
                                                 int graph_annotation_id) noexcept;

  static OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL GetGraphCaptureNodeAssignmentPolicyImpl(
      const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetAvailableResourceImpl(
      const OrtEp* this_ptr, OrtResourceCount* available) noexcept;

#if defined(ENABLE_CUDA_PROFILING)
  static OrtStatus* ORT_API_CALL CreateProfilerImpl(
      OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept;
#endif

  /// Helper to parse the graph annotation ID from run options.
  CudaGraphAnnotation_t GetGraphAnnotationId(const OrtRunOptions* run_options) const;

  struct PerThreadContext;
  using PerThreadContextMap = std::unordered_map<const CudaEp*, std::shared_ptr<PerThreadContext>>;

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache();
  PerThreadContext& GetPerThreadContext() const;

  CudaEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;

  mutable std::mutex per_thread_contexts_mutex_;
  // The thread-local cache owns contexts so they are released when a thread exits.
  // The EP tracks live caches only to remove its entry when the EP is destroyed.
  mutable std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
      per_thread_context_caches_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
