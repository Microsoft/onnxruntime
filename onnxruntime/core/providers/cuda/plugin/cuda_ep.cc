// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

#include "cuda_ep.h"
#include "cuda_ep_factory.h"
#include "core/providers/cuda/cudnn_loader.h"
#include "cuda_stream_plugin.h"
#include "cuda_graph_plugin.h"
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"
#include "cuda_allocator_plugin.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/framework/allocator.h"
#include "ep/get_capability_utils.h"
#include "ep/api.h"  // onnxruntime::ep::CurrentOrtApiVersion()

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "core/graph/constants.h"
#include "core/providers/cuda/cuda_nhwc_ops.h"
#include "contrib_ops/cuda/quantization/matmul_nbits_tactic_cache.h"

namespace onnxruntime {
namespace cuda_plugin {

namespace {

std::unique_ptr<IExecutionProvider> CreateCudaPluginProvider(std::string_view ep_name, const OrtEp* ort_ep) {
  return std::make_unique<::onnxruntime::CUDAExecutionProvider>(std::string{ep_name}, ort_ep);
}

AllocatorPtr CreateCudaPluginTempSpaceAllocator(int device_id) {
  return std::make_shared<::onnxruntime::CUDAAllocator>(device_id, ::onnxruntime::CUDA);
}

AllocatorPtr CreateCudaPluginTempSpaceCpuAllocator() {
  return ::onnxruntime::CPUAllocator::DefaultInstance();
}

void DestroyCudaStreamForDevice(cudaStream_t stream, int device_id);

cudaStream_t CreateCudaStreamForDevice(int device_id) {
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  cudaStream_t stream = nullptr;

  try {
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
    PL_CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (restore_prev_device) {
      PL_CUDA_CALL_THROW(cudaSetDevice(prev_device));
    }
  } catch (...) {
    if (stream != nullptr) {
      DestroyCudaStreamForDevice(stream, device_id);
    }
    if (restore_prev_device) {
      static_cast<void>(cudaSetDevice(prev_device));
    }
    throw;
  }

  return stream;
}

void DestroyCudaStreamForDevice(cudaStream_t stream, int device_id) {
  if (stream == nullptr) {
    return;
  }

  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  static_cast<void>(cudaSetDevice(device_id));
  static_cast<void>(cudaStreamDestroy(stream));
  if (restore_prev_device) {
    static_cast<void>(cudaSetDevice(prev_device));
  }
}

}  // namespace

struct CudaEp::PerThreadContext {
  // When use_external_stream is true (user_compute_stream combined with CUDA graph), capture and
  // replay happen on that user-owned stream so they see the same stream as the kernels; the
  // context neither creates nor destroys it. Ownership is derived from the caller's intent rather
  // than from external_stream being non-null, because a user may legitimately select the CUDA
  // default stream (cudaStream_t(0), i.e. nullptr) as the compute stream — that is still an
  // external, user-owned stream and must not be destroyed by the context. When use_external_stream
  // is false the context creates and owns a dedicated graph stream.
  explicit PerThreadContext(int device_id, bool use_external_stream = false,
                            cudaStream_t external_stream = nullptr)
      : device_id(device_id),
        owns_graph_stream(!use_external_stream),
        graph_stream(use_external_stream ? external_stream
                                         : CreateCudaStreamForDevice(device_id)),
        cuda_graph(graph_stream) {
  }

  ~PerThreadContext() {
    // Destroy captured graph execs before destroying the stream they replay on.
    cuda_graph.Reset();
    if (owns_graph_stream) {
      DestroyCudaStreamForDevice(graph_stream, device_id);
    }
    graph_stream = nullptr;
  }

  int device_id;
  bool owns_graph_stream;
  cudaStream_t graph_stream = nullptr;
  CudaGraphManager cuda_graph;
  size_t pre_capture_free_mem = 0;
  uint64_t last_qmoe_graph_gen = 0;
};

CudaEp::CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger)
    : onnxruntime::ep::adapter::Ep{CreateCudaPluginProvider(factory.GetEpName(), static_cast<const OrtEp*>(this)),
                                   CreateCudaPluginTempSpaceCpuAllocator(),
                                   CreateCudaPluginTempSpaceAllocator(config.device_id)},
      factory_(factory),
      name_(factory.GetEpName()),
      config_(config),
      logger_(logger) {
  // ort_version_supported reports the ORT API version this plugin was compiled with (ORT_API_VERSION).
  // ORT uses it to avoid reading OrtEp struct fields that did not exist when the plugin was compiled.
  ort_version_supported = ORT_API_VERSION;

  // The plugin is compiled against the latest ORT headers (ORT_API_VERSION) but may be loaded by an
  // older ORT runtime, down to the floor declared in plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION. Some
  // OrtEp callbacks below — and the OrtEpApi functions their implementations call — only exist in
  // newer ORT versions. The guard against calling an OrtEpApi function the runtime does not provide
  // is the runtime API version, not ort_version_supported. We therefore gate such callbacks on the
  // version negotiated with the runtime (onnxruntime::ep::CurrentOrtApiVersion()): only advertise a
  // callback when the runtime is new enough to (a) know about the OrtEp struct field and (b) provide
  // every OrtEpApi function the callback relies on. Leaving the pointer null on older runtimes
  // disables only that optional capability while the EP stays fully functional.
  const uint32_t ort_version = ::onnxruntime::ep::CurrentOrtApiVersion();

  // Kernel-registry-based EP callbacks (all introduced in ORT <= 1.24).
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  CreateAllocator = (config_.external_alloc != nullptr && config_.external_free != nullptr)
                        ? CreateAllocatorImpl
                        : nullptr;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  OnSessionInitializationEnd = (ort_version >= 27) ? OnSessionInitializationEndImpl : nullptr;

  // OrtEp::Sync is \since ORT 1.25. A runtime older than 1.25 does not know about this OrtEp field and
  // ignores it regardless of its value. We still gate it on the runtime version to keep the
  // minimum-version dependency explicit at each assignment and consistent with the callbacks below.
  Sync = (ort_version >= 25) ? SyncImpl : nullptr;

  // Not a compile-based EP
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  // Graph capture/replay (OrtEp::IsGraphCaptureEnabled/IsGraphCaptured/ReplayGraph/
  // GetGraphCaptureNodeAssignmentPolicy) and device-resource accounting
  // (OrtEp::GetAvailableResource + OrtResourceCount) are \since ORT 1.26. Only advertise them when the
  // negotiated runtime is >= 1.26; older runtimes neither expose these OrtEp fields nor support
  // EP-driven graph capture, so leaving them null preserves the same behavior explicitly.
  if (ort_version >= 26) {
    IsGraphCaptureEnabled = IsGraphCaptureEnabledImpl;
    IsGraphCaptured = IsGraphCapturedImpl;
    ReplayGraph = ReplayGraphImpl;
    GetGraphCaptureNodeAssignmentPolicy = GetGraphCaptureNodeAssignmentPolicyImpl;
    GetAvailableResource = GetAvailableResourceImpl;
  } else {
    IsGraphCaptureEnabled = nullptr;
    IsGraphCaptured = nullptr;
    ReplayGraph = nullptr;
    GetGraphCaptureNodeAssignmentPolicy = nullptr;
    GetAvailableResource = nullptr;
  }

  // Profiling — CUPTI-based GPU activity tracing when profiling is enabled at build time.
  // The EP profiler API (OrtEp::CreateProfiler, OrtEpProfilerImpl, and the OrtEpApi
  // CreateProfilingEvent / ProfilingEventsContainer_AddEvents / ReleaseProfilingEvent functions that
  // CudaPluginEpProfiler calls) is \since ORT 1.25. Only advertise the profiler when the negotiated
  // runtime supports it; this single guard makes every 1.25 profiler API call unreachable on older
  // runtimes (the profiler is never created), so inference still runs without EP-level GPU profiling.
#if defined(ENABLE_CUDA_PROFILING)
  CreateProfiler = (ort_version >= 25) ? CreateProfilerImpl : nullptr;
#else
  CreateProfiler = nullptr;
#endif

  const OrtApi& ort_api = factory_.GetOrtApi();
  Ort::Status log_status(ort_api.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                                   "CUDA Plugin EP created",
                                                   ORT_FILE, __LINE__, __FUNCTION__));

  // Store per-EP runtime configuration inside the adapter-wrapped execution
  // provider itself. Migrated kernels retrieve a shared config object at
  // compute time via GetCudaKernelAdapterRuntimeConfigForProvider().
  // Adding a new config field only requires updating
  // CudaKernelAdapterRuntimeConfig, CudaEp::Config, and the struct-initializer
  // below — no function-signature change.
  onnxruntime::cuda::detail::CudaKernelAdapterRuntimeConfig adapter_config;
  adapter_config.use_tf32 = config_.use_tf32;
  adapter_config.cudnn_conv_algo = config_.cudnn_conv_algo;
  adapter_config.cudnn_conv_use_max_workspace = config_.cudnn_conv_use_max_workspace;
  adapter_config.cudnn_conv1d_pad_to_nc1d = config_.cudnn_conv1d_pad_to_nc1d;
  adapter_config.enable_cudnn = config_.enable_cudnn;
  adapter_config.fuse_conv_bias = config_.fuse_conv_bias;
  adapter_config.sdpa_kernel = config_.sdpa_kernel;
  adapter_config.device_id = config_.device_id;
  adapter_config.do_copy_in_default_stream = config_.do_copy_in_default_stream;
  adapter_config.enable_cuda_graph = config_.enable_cuda_graph;
  adapter_config.qmoe_controller = std::make_shared<onnxruntime::cuda::QMoESessionController>(config_.device_id);
  if (config_.budget_account) {
    adapter_config.qmoe_controller->SetBudgetAccount(config_.budget_account);
  }
  onnxruntime::cuda::SetCudaKernelAdapterRuntimeConfigForProvider(
      static_cast<const void*>(EpImpl()), adapter_config);

  // CUDA graph streams are created lazily per thread by PerThreadContext.
}

CudaEp::~CudaEp() {
  // The plugin build excludes cuda_execution_provider.cc, so persist staged fpA_intB tactics here.
  onnxruntime::contrib::cuda::FlushMatMulNBitsTacticCaches();

  if (EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller) {
      runtime_cfg->qmoe_controller->Shutdown();
    }
  }
  std::lock_guard<std::mutex> lock(per_thread_contexts_mutex_);
  for (const auto& cache_weak : per_thread_context_caches_) {
    auto cache = cache_weak.lock();
    if (!cache) {
      continue;
    }
    ORT_IGNORE_RETURN_VALUE(cache->erase(this));
  }
  per_thread_context_caches_.clear();
}

/*static*/
const char* ORT_API_CALL CudaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const CudaEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* ort_graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  Ort::ConstGraph graph{ort_graph};
  std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;
  }

  if (auto model_path = graph.GetModelPath(); !model_path.empty() && ep->EpImpl() != nullptr) {
    std::error_code ec;
    std::filesystem::path path(model_path);
    std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    std::filesystem::path resolved_path = (!ec && !abs_path.empty()) ? std::move(abs_path) : std::move(path);
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg) {
      if (runtime_cfg->qmoe_controller) {
        runtime_cfg->qmoe_controller->SetModelPath(resolved_path);
      }
      runtime_cfg->SetModelPath(std::move(resolved_path));
    }
  }

  // Three-phase filtering determines which graph nodes run on this EP:
  // Phase 1: Collect tentative nodes that have a registered CUDA kernel.
  // Phase 2: Filter out CPU-preferred nodes (cheap ops where device-to-host
  //          copy overhead would exceed the compute benefit).
  // Phase 3: Register remaining nodes as supported by this EP.

  // Phase 1: Collect tentative nodes — those for which we have a registered kernel.
  std::vector<const OrtNode*> candidate_nodes;
  candidate_nodes.reserve(all_nodes.size());
  std::vector<const OrtNode*> tentative_nodes;
  tentative_nodes.reserve(all_nodes.size());

  for (const auto& node : all_nodes) {
    const std::string& ep_name = node.GetEpName();
    if (!ep_name.empty()) {
      if (ep_name == ep->name_) {
        candidate_nodes.push_back(node);
      }
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
        graph_support_info, node, &kernel_def));

    if (kernel_def != nullptr) {
      candidate_nodes.push_back(node);
      tentative_nodes.push_back(node);
    } else {
      // Emit a diagnostic when an NHWC-domain node has no matching kernel.
      // This helps identify gaps between the layout conversion allowlist and
      // the actually-registered NHWC kernels in the plugin build.
      const std::string& node_domain = node.GetDomain();
      if (node_domain == kMSInternalNHWCDomain) {
        ORT_CXX_LOGF(Ort::Logger(&ep->logger_), ORT_LOGGING_LEVEL_WARNING,
                     "NHWC kernel miss: op=%s domain=%s version=%d node=%s - "
                     "no matching kernel registered in the CUDA plugin EP.",
                     node.GetOperatorType().c_str(), node_domain.c_str(),
                     node.GetSinceVersion(), node.GetName().c_str());
      }
    }
  }

  // Phase 2: Filter out CPU-preferred nodes (e.g., Shape, NonZero, small compute ops
  // that would be cheaper on CPU than incurring device-to-host copy overhead).
  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(ep::GetCpuPreferredNodes(
      *ort_graph, *graph_support_info, ep->logger_,
      gsl::span<const OrtNode* const>(tentative_nodes.data(), tentative_nodes.size()),
      cpu_preferred_nodes));

  // Phase 3: Add final supported nodes (tentative minus CPU-preferred).
  // Resource budget enforcement is handled by the host after GetCapability returns.

  for (const OrtNode* ort_node : candidate_nodes) {
    if (cpu_preferred_nodes.count(ort_node) != 0) {
      continue;
    }

    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_AddSingleNode(
        graph_support_info, ort_node));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetKernelRegistryImpl(
    OrtEp* this_ptr,
    const OrtKernelRegistry** kernel_registry) noexcept {
  auto* ep = static_cast<CudaEp*>(this_ptr);
  *kernel_registry = nullptr;

  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetPreferredDataLayoutImpl(
    OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
#ifdef ENABLE_CUDA_NHWC_OPS
  *preferred_data_layout = ep->config_.prefer_nhwc ? OrtEpDataLayout_NHWC : OrtEpDataLayout_NCHW;
#else
  ORT_UNUSED_PARAMETER(ep);
  *preferred_data_layout = OrtEpDataLayout_NCHW;
#endif
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::ShouldConvertDataLayoutForOpImpl(
    OrtEp* this_ptr, const char* domain, const char* op_type,
    OrtEpDataLayout target_data_layout, int* should_convert) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);

  if (should_convert == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "should_convert must not be null.");
  }

  const char* safe_domain = domain != nullptr ? domain : "";
  const char* safe_op_type = op_type != nullptr ? op_type : "";

#ifndef ENABLE_CUDA_NHWC_OPS
  ORT_UNUSED_PARAMETER(safe_domain);
  ORT_UNUSED_PARAMETER(safe_op_type);
  ORT_UNUSED_PARAMETER(target_data_layout);
  *should_convert = 0;  // NHWC kernels are not compiled into this plugin build.
  return nullptr;
#else

  // Only convert to NHWC; for any other target layout, let ORT decide.
  if (target_data_layout != OrtEpDataLayout_NHWC) {
    *should_convert = -1;  // Let ORT decide
    return nullptr;
  }

  if (cuda::IsNhwcEligible(safe_domain, safe_op_type)) {
    *should_convert = 1;  // Convert
  } else {
    *should_convert = 0;  // Explicitly decline conversion for unsupported NHWC ops.
  }
  return nullptr;
#endif
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  auto mem_type = ep_api.MemoryDevice_GetMemoryType(memory_device);
  if (mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
    error += std::to_string(mem_type);
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  int device_id = ep_api.MemoryDevice_GetDeviceId(memory_device);
  if (device_id != ep->config_.device_id) {
    std::string error = "Invalid OrtMemoryDevice. Expected CUDA device ordinal ";
    error += std::to_string(ep->config_.device_id);
    error += " for this EP instance. Got ";
    error += std::to_string(device_id);
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  auto cuda_stream = std::make_unique<CudaSyncStream>(ep->factory_, device_id, ep->config_.enable_cudnn, this_ptr);

  if (ep->config_.has_user_compute_stream) {
    // A user-provided compute stream is honored for kernels regardless of whether CUDA graph
    // capture is enabled - this branch is taken in both graph and non-graph runs. Use the caller's
    // intent flag rather than checking the handle for non-null: cudaStream_t(0) / nullptr is the
    // valid CUDA default stream and can be selected explicitly by the user. Wrap the external CUDA
    // stream with full cuBLAS/cuDNN handles. When CUDA graph capture is also enabled,
    // capture/replay run on this same user stream (see GetPerThreadContext), so kernels and graph
    // capture share one stream.
    RETURN_IF_ERROR(cuda_stream->InitHandlesWithUserStream(
        static_cast<cudaStream_t>(ep->config_.user_compute_stream)));
  } else if (ep->config_.enable_cuda_graph) {
    // When CUDA graph capture is enabled, all operations on this thread must go
    // through the thread's graph stream so capture/replay sees the same stream
    // as kernels.
    RETURN_IF_ERROR(cuda_stream->InitHandlesWithExternalStream(ep->GetPerThreadContext().graph_stream));
  } else {
    RETURN_IF_ERROR(cuda_stream->InitHandles());
  }

  *stream = cuda_stream.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::CreateAllocatorImpl(
    OrtEp* this_ptr,
    const OrtMemoryInfo* memory_info,
    OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto& ep = *static_cast<CudaEp*>(this_ptr);
  *allocator = nullptr;

  const OrtApi& ort_api = ep.factory_.GetOrtApi();
  const char* name = "";
  OrtStatus* status = ort_api.MemoryInfoGetName(memory_info, &name);
  if (status != nullptr) {
    return status;
  }

  int req_device_id = 0;
  status = ort_api.MemoryInfoGetId(memory_info, &req_device_id);
  if (status != nullptr) {
    return status;
  }

  if (name != nullptr && strcmp(name, "Cuda") == 0) {
    auto external_allocator = std::make_unique<CudaExternalDeviceAllocator>(
        memory_info, req_device_id,
        ep.config_.external_alloc, ep.config_.external_free, ep.config_.external_empty_cache);
    *allocator = external_allocator.release();
    return nullptr;
  }

  Ort::KeyValuePairs allocator_options;
  allocator_options.Add(kEpInstanceIdOptionKey,
                        std::to_string(reinterpret_cast<uintptr_t>(static_cast<const OrtEp*>(&ep))).c_str());
  return ep.factory_.CreateAllocator(&ep.factory_, memory_info, allocator_options, allocator);

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::SyncImpl(OrtEp* this_ptr) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);

  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);

  Ort::Status status = StatusFromCudaError(cudaSetDevice(ep->config_.device_id));
  if (status.IsOK()) {
    status = StatusFromCudaError(cudaDeviceSynchronize());
  }

  if (restore_prev_device) {
    Ort::Status restore_status = StatusFromCudaError(cudaSetDevice(prev_device));
    if (status.IsOK()) {
      status = std::move(restore_status);
    }
  }

  return status.release();

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::IsConcurrentRunSupportedImpl(
    OrtEp* this_ptr, bool* is_supported) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);

  if (is_supported == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "is_supported must not be null.");
  }

  auto* ep = static_cast<CudaEp*>(this_ptr);
  bool has_offload_nodes = false;
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller && runtime_cfg->qmoe_controller->HasOffloadNodes()) {
      has_offload_nodes = true;
    }
  }
  // Concurrent runs require stream-tagged scratch allocations. The plugin kernel adapter can tag
  // scratch chunks only when the hosting ORT runtime exposes KernelContext_GetSyncStream.
  static constexpr uint32_t kOrtKernelContextGetSyncStreamMinVersion = 28;
  *is_supported = !has_offload_nodes &&
                  !ep->config_.use_ep_level_unified_stream &&
                  ::onnxruntime::ep::CurrentOrtApiVersion() >= kOrtKernelContextGetSyncStreamMinVersion;
  return nullptr;
}

// --- CUDA Graph callback implementations ---

const std::shared_ptr<CudaEp::PerThreadContextMap>& CudaEp::PerThreadContextCache() {
  thread_local const std::shared_ptr<PerThreadContextMap> per_thread_context_cache =
      std::make_shared<PerThreadContextMap>();
  return per_thread_context_cache;
}

CudaEp::PerThreadContext& CudaEp::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    return *cached_context_it->second;
  }

  // NOTE: `enable_cuda_graph` in this condition does NOT restrict using a user compute stream to
  // the graph case. A user compute stream is honored for kernels in BOTH graph and non-graph runs
  // — that happens in CreateSyncStreamForDeviceImpl(), which wraps config_.user_compute_stream
  // independently of enable_cuda_graph. This flag only governs the PerThreadContext's *graph
  // stream*, and PerThreadContext is a graph-capture-only object: GetPerThreadContext() is reached
  // exclusively from the graph path (CreateSyncStreamForDeviceImpl's enable_cuda_graph branch,
  // OnRunStart/OnRunEnd, IsGraphCaptured, ReplayGraph). With graph disabled, no PerThreadContext is
  // ever constructed, so its stream ownership is irrelevant.
  //
  // When a user compute stream IS combined with CUDA graph capture, capture/replay must run on the
  // user's stream (the same stream the kernels are issued to) rather than a separate EP-owned
  // stream. The user owns the stream's lifetime, so the context must not destroy it. Derive this
  // from the caller's intent (has_user_compute_stream && enable_cuda_graph), not from whether the
  // handle is null: a user may explicitly choose the CUDA default stream (nullptr), which is still
  // an external stream that the context must not own/destroy.
  const bool use_external_stream = config_.has_user_compute_stream && config_.enable_cuda_graph;
  cudaStream_t external_stream =
      use_external_stream ? static_cast<cudaStream_t>(config_.user_compute_stream) : nullptr;
  auto context = std::make_shared<PerThreadContext>(config_.device_id, use_external_stream, external_stream);
  PerThreadContext& context_ref = *context;
  {
    std::lock_guard<std::mutex> lock(per_thread_contexts_mutex_);
    for (auto it = per_thread_context_caches_.begin(); it != per_thread_context_caches_.end();) {
      if (it->expired()) {
        it = per_thread_context_caches_.erase(it);
      } else {
        ++it;
      }
    }
    ORT_IGNORE_RETURN_VALUE(per_thread_context_caches_.insert(per_thread_context_cache));
  }

  auto insert_result = per_thread_context_cache->emplace(this, std::move(context));
  ORT_ENFORCE(insert_result.second);
  return context_ref;
}

CudaGraphAnnotation_t CudaEp::GetGraphAnnotationId(const OrtRunOptions* run_options) const {
  if (run_options == nullptr) {
    return kCudaGraphAnnotationDefault;
  }
  const char* value = factory_.GetOrtApi().GetRunConfigEntry(run_options, "gpu_graph_id");
  if (value == nullptr) {
    return kCudaGraphAnnotationDefault;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    return kCudaGraphAnnotationDefault;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnSessionInitializationEndImpl(OrtEp* this_ptr) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* ep = static_cast<CudaEp*>(this_ptr);
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller) {
      auto st = runtime_cfg->qmoe_controller->PlanAndAdmitSession();
      if (!st.IsOK()) {
        return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, st.ErrorMessage().c_str());
      }
    }
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

namespace {
// Ends a controller run on scope exit unless released. ORT does not call OnRunEnd for a provider whose OnRunStart
// failed, and an error or exception inside OnRunEnd must not leave the controller run active.
class QMoEControllerRunRollback {
 public:
  QMoEControllerRunRollback() = default;
  QMoEControllerRunRollback(const QMoEControllerRunRollback&) = delete;
  QMoEControllerRunRollback& operator=(const QMoEControllerRunRollback&) = delete;
  ~QMoEControllerRunRollback() {
    if (controller_) {
      try {
        static_cast<void>(controller_->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
      } catch (...) {
      }
    }
  }
  void Arm(std::shared_ptr<onnxruntime::cuda::QMoESessionController> controller) { controller_ = std::move(controller); }
  void Release() { controller_.reset(); }

 private:
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> controller_;
};
}  // namespace

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunStartImpl(
    OrtEp* this_ptr, const OrtRunOptions* run_options) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  QMoEControllerRunRollback controller_rollback;
  if (!ep->config_.enable_cuda_graph) {
    ep->factory_.ShrinkEligibleDeviceArenas(ep->config_.device_id, ep->config_.budget_account.get());
  }
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller && runtime_cfg->qmoe_controller->HasOffloadNodes()) {
      const char* strict_env = std::getenv("ORT_QMOE_REJECT_DISABLE_SYNC");
      if (strict_env != nullptr && std::strcmp(strict_env, "1") == 0 && run_options != nullptr) {
        const char* no_sync =
            ep->factory_.GetOrtApi().GetRunConfigEntry(run_options, "disable_synchronize_execution_providers");
        if (no_sync != nullptr && std::strcmp(no_sync, "1") == 0) {
          return ep->factory_.GetOrtApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "QMoEOffload does not support disable_synchronize_execution_providers=1; required completion sync cannot be skipped.");
        }
      }
      auto st = runtime_cfg->qmoe_controller->OnRunStart();
      if (!st.IsOK()) {
        return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, st.ErrorMessage().c_str());
      }
      controller_rollback.Arm(runtime_cfg->qmoe_controller);
    }
  }
  if (!ep->config_.enable_cuda_graph) {
    controller_rollback.Release();
    return nullptr;
  }

  // Recover from any previous failed run on this thread before deciding whether
  // this run will enter capture mode.
  IsThreadCapturingCudaGraph() = false;

  auto& context = ep->GetPerThreadContext();
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller) {
      const uint64_t cur_gen = runtime_cfg->qmoe_controller->GraphGeneration();
      if (cur_gen != context.last_qmoe_graph_gen) {
        context.cuda_graph.Reset();
        context.last_qmoe_graph_gen = cur_gen;
      }
    }
  }
  CudaGraphAnnotation_t id = ep->GetGraphAnnotationId(run_options);
  if (!context.cuda_graph.IsGraphCaptured(id) &&
      context.cuda_graph.IsGraphCaptureAllowed(id, ep->config_.min_num_runs_before_cuda_graph_capture)) {
    // Keep the current CUDA device aligned with the graph stream for the full
    // capture window. Kernel Compute() skips cudaSetDevice() while capturing.
    PL_CUDA_CALL_THROW(cudaSetDevice(ep->config_.device_id));

    // Record free GPU memory before capture for allocation-during-capture detection.
    context.pre_capture_free_mem = 0;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      context.pre_capture_free_mem = free_mem;
    }
    // Captured graphs embed arena addresses, so this session's arenas must not be shrunk from now on.
    ep->factory_.MarkSessionGraphCapture(ep->config_.device_id, ep->config_.budget_account.get());
    context.cuda_graph.CaptureBegin(id);
    IsThreadCapturingCudaGraph() = true;
  }

  controller_rollback.Release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunEndImpl(
    OrtEp* this_ptr, const OrtRunOptions* run_options, bool sync_stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> qmoe_controller;
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller && runtime_cfg->qmoe_controller->HasOffloadNodes()) {
      qmoe_controller = runtime_cfg->qmoe_controller;
    }
  }

  QMoEControllerRunRollback controller_rollback;
  controller_rollback.Arm(qmoe_controller);

  if (!ep->config_.enable_cuda_graph) {
    if (qmoe_controller) {
      controller_rollback.Release();
      bool run_succeeded = true;
      if (run_options != nullptr) {
        const char* outcome = ep->factory_.GetOrtApi().GetRunConfigEntry(run_options, "__ort.run_outcome");
        if (outcome != nullptr && std::strcmp(outcome, "ok") != 0) {
          run_succeeded = false;
        }
      }
      auto st = qmoe_controller->OnRunEnd(sync_stream, run_succeeded);
      if (!st.IsOK()) {
        return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, st.ErrorMessage().c_str());
      }
    }
    return nullptr;
  }

  // Always clear the flag before leaving this callback so a failed capture or
  // failed replay cannot poison later runs on the same thread.
  IsThreadCapturingCudaGraph() = false;

  auto& context = ep->GetPerThreadContext();
  CudaGraphAnnotation_t id = ep->GetGraphAnnotationId(run_options);
  bool replayed_graph = false;
  if (!context.cuda_graph.IsGraphCaptured(id)) {
    if (context.cuda_graph.IsGraphCaptureAllowed(id, ep->config_.min_num_runs_before_cuda_graph_capture)) {
      context.cuda_graph.CaptureEnd(id);

      // Check if GPU memory was allocated during capture (which would make the
      // captured graph invalid since CUDA graph replay cannot reproduce allocations).
      if (context.pre_capture_free_mem > 0) {
        size_t post_free_mem = 0;
        size_t total_mem = 0;
        if (cudaMemGetInfo(&post_free_mem, &total_mem) == cudaSuccess) {
          if (post_free_mem < context.pre_capture_free_mem) {
            Ort::Status log_status(ep->factory_.GetOrtApi().Logger_LogMessage(
                &ep->logger_, ORT_LOGGING_LEVEL_WARNING,
                "GPU memory was allocated during CUDA graph capture. "
                "Graph replay may produce incorrect results. Consider increasing "
                "min_num_runs_before_cuda_graph_capture to allow allocations to stabilize.",
                ORT_FILE, __LINE__, __FUNCTION__));
          }
        }
        context.pre_capture_free_mem = 0;
      }

      if (qmoe_controller) {
        auto prep_st = qmoe_controller->OnBeforeGraphReplay(context.graph_stream);
        if (!prep_st.IsOK()) {
          return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, prep_st.ErrorMessage().c_str());
        }
      }

      // CUDA work issued to a capturing stream doesn't actually run on the GPU,
      // so replay the captured graph to actually execute the work.
      OrtStatus* status = context.cuda_graph.Replay(id, sync_stream);
      if (status != nullptr) {
        return status;
      }
      replayed_graph = true;
      if (qmoe_controller) {
        auto after_st = qmoe_controller->OnAfterGraphReplay(context.graph_stream, sync_stream);
        if (!after_st.IsOK()) {
          return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, after_st.ErrorMessage().c_str());
        }
      }
    } else {
      context.cuda_graph.IncrementRegularRunCount(id);
    }
  }

  if (qmoe_controller) {
    controller_rollback.Release();
    bool run_succeeded = true;
    if (run_options != nullptr) {
      const char* outcome = ep->factory_.GetOrtApi().GetRunConfigEntry(run_options, "__ort.run_outcome");
      if (outcome != nullptr && std::strcmp(outcome, "ok") != 0) {
        run_succeeded = false;
      }
    }
    auto st = qmoe_controller->OnRunEnd(sync_stream, run_succeeded);
    if (!st.IsOK()) {
      return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, st.ErrorMessage().c_str());
    }
  }

  if (sync_stream && !replayed_graph) {
    PL_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(context.graph_stream));
  } else if (!sync_stream) {
    cudaError_t query_err = cudaStreamQuery(context.graph_stream);
    if (query_err != cudaSuccess && query_err != cudaErrorNotReady) {
      PL_CUDA_RETURN_IF_ERROR(query_err);
    }
  }
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
bool ORT_API_CALL CudaEp::IsGraphCaptureEnabledImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  return ep->config_.enable_cuda_graph;
}

/*static*/
bool ORT_API_CALL CudaEp::IsGraphCapturedImpl(const OrtEp* this_ptr, int graph_annotation_id) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return false;
  }

  try {
    if (const char* handoff_env = std::getenv("ORT_QMOE_PREFILL_HANDOFF_DONE");
        handoff_env != nullptr && std::strcmp(handoff_env, "1") == 0) {
      ep->factory_.ShrinkEligibleDeviceArenas(ep->config_.device_id, ep->config_.budget_account.get());
    }
    auto& context = ep->GetPerThreadContext();
    if (ep->EpImpl() != nullptr) {
      auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
          static_cast<const void*>(ep->EpImpl()));
      if (runtime_cfg && runtime_cfg->qmoe_controller) {
        const uint64_t cur_gen = runtime_cfg->qmoe_controller->GraphGeneration();
        if (cur_gen != context.last_qmoe_graph_gen) {
          context.cuda_graph.Reset();
          context.last_qmoe_graph_gen = cur_gen;
        }
      }
    }
    return context.cuda_graph.IsGraphCaptured(graph_annotation_id);
  } catch (...) {
    return false;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::ReplayGraphImpl(OrtEp* this_ptr, int graph_annotation_id) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "ReplayGraph called but CUDA graph manager is not initialized");
  }
  PL_CUDA_CALL_THROW(cudaSetDevice(ep->config_.device_id));
  auto& context = ep->GetPerThreadContext();
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> qmoe_controller;
  if (ep->EpImpl() != nullptr) {
    auto runtime_cfg = onnxruntime::cuda::detail::GetCudaKernelAdapterRuntimeConfigForProvider(
        static_cast<const void*>(ep->EpImpl()));
    if (runtime_cfg && runtime_cfg->qmoe_controller && runtime_cfg->qmoe_controller->HasOffloadNodes()) {
      qmoe_controller = runtime_cfg->qmoe_controller;
    }
  }
  if (qmoe_controller) {
    auto start_st = qmoe_controller->OnRunStart();
    if (!start_st.IsOK()) {
      return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, start_st.ErrorMessage().c_str());
    }
    auto prep_st = qmoe_controller->OnBeforeGraphReplay(context.graph_stream);
    if (!prep_st.IsOK()) {
      static_cast<void>(qmoe_controller->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
      return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, prep_st.ErrorMessage().c_str());
    }
  }
  // Launch graph without sync. The caller (PluginExecutionProvider::ReplayGraph)
  // handles synchronization based on disable_synchronize_execution_providers.
  // This function is only called from that bridge code path.
  OrtStatus* replay_st = context.cuda_graph.Replay(graph_annotation_id, /*sync=*/false);
  if (replay_st != nullptr) {
    if (qmoe_controller) {
      static_cast<void>(qmoe_controller->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
    }
    return replay_st;
  }
  if (qmoe_controller) {
    auto after_st = qmoe_controller->OnAfterGraphReplay(context.graph_stream, /*sync_stream=*/false);
    auto end_st = qmoe_controller->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/after_st.IsOK());
    if (!after_st.IsOK()) {
      return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, after_st.ErrorMessage().c_str());
    }
    if (!end_st.IsOK()) {
      return ep->factory_.GetOrtApi().CreateStatus(ORT_FAIL, end_st.ErrorMessage().c_str());
    }
  }
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL CudaEp::GetGraphCaptureNodeAssignmentPolicyImpl(
    const OrtEp* /*this_ptr*/) noexcept {
  return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetAvailableResourceImpl(
    const OrtEp* this_ptr, OrtResourceCount* available) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (available == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "`available` must not be null");
  }

  auto* ep = static_cast<const CudaEp*>(this_ptr);
  int current_device = 0;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err != cudaSuccess) {
    return Ort::GetApi().CreateStatus(
        ORT_RUNTIME_EXCEPTION,
        (std::string("cudaGetDevice failed: ") + cudaGetErrorString(cuda_err)).c_str());
  }

  // Switch to the EP's configured device if needed
  if (current_device != ep->config_.device_id) {
    cuda_err = cudaSetDevice(ep->config_.device_id);
    if (cuda_err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_RUNTIME_EXCEPTION,
          (std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_err)).c_str());
    }
  }

  size_t free_memory = 0;
  size_t total_memory = 0;
  cuda_err = cudaMemGetInfo(&free_memory, &total_memory);

  // Restore the original device
  if (current_device != ep->config_.device_id) {
    cudaSetDevice(current_device);  // best-effort restore
  }

  if (cuda_err != cudaSuccess) {
    return Ort::GetApi().CreateStatus(
        ORT_RUNTIME_EXCEPTION,
        (std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(cuda_err)).c_str());
  }

  *available = OrtResourceCount::FromTotalBytes(static_cast<uint64_t>(free_memory));
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

#if defined(ENABLE_CUDA_PROFILING)
/*static*/
OrtStatus* ORT_API_CALL CudaEp::CreateProfilerImpl(
    OrtEp* this_ptr, OrtEpProfilerImpl** profiler) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (profiler == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "`profiler` must not be null");
  }

  *profiler = nullptr;

  auto* ep = static_cast<CudaEp*>(this_ptr);
  auto profiler_impl = std::make_unique<CudaPluginEpProfiler>(ep->factory_.GetEpApi());
  *profiler = profiler_impl.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}
#endif  // defined(ENABLE_CUDA_PROFILING)

}  // namespace cuda_plugin
}  // namespace onnxruntime
