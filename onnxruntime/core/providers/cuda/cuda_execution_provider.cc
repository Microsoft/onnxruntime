// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#include "core/common/inlined_containers.h"
#include "core/common/parse_string.h"
#include "core/framework/int4.h"
#include "core/framework/kernel_dispatch_policy.h"
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "core/framework/node_shape_resolver.h"
#endif
#include "core/framework/resource_accountant.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_external_data_loader.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_nhwc_ops.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/cuda_profiler.h"
#include "core/providers/cuda/cuda_mempool_arena.h"
#include "core/providers/cuda/cudnn_loader.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#ifndef USE_CUDA_MINIMAL
#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cuda/cuda_contrib_kernels.h"
#endif

#ifdef ENABLE_CUDA_NHWC_OPS
#include "core/providers/cuda/cuda_nhwc_kernels.h"
#endif

#ifdef ENABLE_TRAINING_OPS
#include "orttraining/training_ops/cuda/cuda_training_kernels.h"
#endif

#ifdef USE_TRITON_KERNEL
#include "core/providers/cuda/triton_kernel.h"
#endif
#endif

#include "core/providers/cuda/cuda_stream_handle.h"

// Persistent fpA_intB tactic cache flush at EP teardown (no-op unless onnxruntime_USE_FPA_INTB_GEMM).
// The header is self-contained (no contrib/CUTLASS dependencies) and resolves to an inline no-op
// when the fpA_intB path is not compiled in.
#include "contrib_ops/cuda/quantization/matmul_nbits_tactic_cache.h"
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
#include "contrib_ops/cuda/quantization/matmul_nbits_workspace_estimate.h"
#endif

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#endif

using namespace onnxruntime::common;

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    auto X_type = ctx->InputType(0);
    if (X_type->IsTensorType()) {
      const auto* X = ctx->Input<Tensor>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
      Tensor* Y = ctx->Output(0, X->Shape());
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
      // The cudaMemCpyAsync will handle the pinned memory and non-pinned memory,
      // so we don't need the check here.
      auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
      ORT_RETURN_IF_ERROR(gpu_data_transfer->CopyTensorAsync(*X, *Y, *ctx->GetComputeStream()));
      return Status::OK();
    } else {
      if (X_type->IsSparseTensorType()) {
        // TODO: support aysnc copy for sparse tensor
        // sync the stream first, since it is a sync memory copy
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
        const auto* X = ctx->Input<SparseTensor>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
        SparseTensor* Y = ctx->OutputSparse(0, X->DenseShape());
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output sparse tensor.");
        return X->Copy(Info().GetDataTransferManager(), *Y);
      } else if (X_type->IsTensorSequenceType()) {
        const TensorSeq* X = ctx->Input<TensorSeq>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor sequence is nullptr.");
        TensorSeq* Y = ctx->Output<TensorSeq>(0);
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor sequence.");
        auto X_dtype = X->DataType();
        Y->SetType(X_dtype);
        AllocatorPtr alloc;

        // If we are copying contents to CUDA, the allocator to use
        // to allocate the buffers of the new tensors in the sequence
        // can be temp space allocator associated with the CUDA EP
        if (Node().OpType() == "MemcpyFromHost") {
          auto status = ctx->GetTempSpaceAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cuda: unable to get an allocator.");
          }
        } else {
          // If we are copying contents to CPU (op type is "MemcpyToHost"),
          // the allocator to use to allocate the buffers of the new tensors
          // in the sequence will be the allocator from the CPU EP
          auto status = ctx->GetTempSpaceCPUAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cuda: unable to get the CPU allocator.");
          }
        }
        auto X_size = X->Size();
        Y->Reserve(X_size);
        for (size_t i = 0; i < X_size; ++i) {
          const Tensor& source_tensor = X->Get(i);
          std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(), source_tensor.Shape(), alloc);
          auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(source_tensor.Location().device,
                                                                                    target_tensor->Location().device);
          ORT_RETURN_IF_ERROR(gpu_data_transfer->CopyTensorAsync(source_tensor, *target_tensor, *ctx->GetComputeStream()));
          Y->Add(std::move(*target_tensor));
        }
        return Status::OK();
      }
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    }
  }
};

namespace cuda {
ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()),
    Memcpy);

}  // namespace cuda

AllocatorPtr CUDAExecutionProvider::CreateCudaAllocator(const CUDAAllocatorParams& cuda_allocator_params) {
  ORT_ENFORCE(cuda_allocator_params.external_alloc_info != nullptr,
              "CUDAAllocatorParams.external_alloc_info is nullptr.");
  const auto& external_allocator_info = *(cuda_allocator_params.external_alloc_info);
  const bool device_budget_enabled =
      (cuda_allocator_params.provider_info != nullptr &&
       cuda_allocator_params.provider_info->gpu_device_mem_limit > 0 &&
       cuda_allocator_params.provider_info->gpu_device_mem_limit != std::numeric_limits<size_t>::max()) ||
      (cuda_allocator_params.budget_account != nullptr &&
       cuda_allocator_params.budget_account->GetAccountStats().limit != std::numeric_limits<size_t>::max());
  ORT_ENFORCE(!device_budget_enabled || !external_allocator_info.UseExternalAllocator(),
              "gpu_device_mem_limit does not support external CUDA allocators");
  ORT_ENFORCE(!device_budget_enabled || cuda_allocator_params.arena_cfg == nullptr ||
                  cuda_allocator_params.arena_cfg->use_cuda_mempool != 1,
              "gpu_device_mem_limit does not support CUDA memory pools");
  if (external_allocator_info.UseExternalAllocator()) {
    AllocatorCreationInfo default_memory_info(
        [external_allocator_info](OrtDevice::DeviceId id) {
          return std::make_unique<CUDAExternalAllocator>(id, CUDA,
                                                         external_allocator_info.alloc,
                                                         external_allocator_info.free,
                                                         external_allocator_info.empty_cache);
        },
        cuda_allocator_params.device_id,
        false);

    return CreateAllocator(default_memory_info);
  } else {
    const auto* arena_cfg = cuda_allocator_params.arena_cfg;
    const bool cuda_mempool_requested = arena_cfg != nullptr && arena_cfg->use_cuda_mempool == 1;
    bool use_cuda_mempool = cuda_mempool_requested && cuda::CudaMempoolArena::IsCudaVersionSupported();

    if (cuda_mempool_requested && !use_cuda_mempool) {
      LOGS_DEFAULT(WARNING)
          << "CUDA memory pool requested but not supported on this device/driver."
          << "Falling back to default BFCArena with CUDA allocator.";
    }

    if (use_cuda_mempool) {
      const bool cuda_graph_enabled = cuda_allocator_params.provider_info != nullptr &&
                                      cuda_allocator_params.provider_info->enable_cuda_graph;

      if (cuda_graph_enabled) {
        LOGS_DEFAULT(WARNING)
            << "CUDA Mempool Arena allocator is not compatible with requested CUDA Graph Capture"
            << "Falling back to default BFCArena with CUDA allocator.";
        use_cuda_mempool = false;
      }
    }

    if (use_cuda_mempool) {
      auto device = OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                              cuda_allocator_params.device_id);
      auto mem_info = OrtMemoryInfo("CUDAMemPoolArena", OrtAllocatorType::OrtArenaAllocator, device, OrtMemTypeDefault);

      auto mempool_allocator = std::make_shared<cuda::CudaMempoolArena>(mem_info,
                                                                        arena_cfg->cuda_mempool_release_threshold,
                                                                        arena_cfg->cuda_mempool_bytes_to_keep_on_shrink,
                                                                        cuda_allocator_params.logger);

      return mempool_allocator;
    } else {
      auto budget_account = cuda_allocator_params.budget_account;
      AllocatorCreationInfo default_memory_info(
          [budget_account](OrtDevice::DeviceId id) {
            return std::make_unique<CUDAAllocator>(id, CUDA, budget_account);
          },
          cuda_allocator_params.device_id,
          true,
          {arena_cfg ? *arena_cfg
                     : OrtArenaCfg(cuda_allocator_params.cuda_mem_threshold,
                                   static_cast<int>(cuda_allocator_params.arena_extend_strategy), -1, -1, -1, -1L)},
          // make it stream aware
          true);
      // CUDA malloc/free is expensive so always use an arena
      return CreateAllocator(default_memory_info);
    }
  }
}

AllocatorPtr CUDAExecutionProvider::CreateCudaPinnedAllocator(const CUDAAllocatorParams& cuda_allocator_params) {
  const auto* arena_cfg = cuda_allocator_params.arena_cfg;
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId id) {
        return std::make_unique<CUDAPinnedAllocator>(id, CUDA_PINNED);
      },
      cuda_allocator_params.device_id,
      true,
      {arena_cfg ? *arena_cfg
                 : OrtArenaCfg(cuda_allocator_params.cuda_mem_threshold,
                               static_cast<int>(cuda_allocator_params.arena_extend_strategy), -1, -1, -1, -1L)},
      // stream-aware flag (intentionally set to false for this allocator)
      false);

  return CreateAllocator(pinned_memory_info);
}

CUDAExecutionProvider::PerThreadContext::PerThreadContext(OrtDevice::DeviceId device_id, cudaStream_t stream,
                                                          const CUDAExecutionProviderInfo& info) {
  CUDA_CALL_THROW(cudaSetDevice(device_id));
#ifndef USE_CUDA_MINIMAL
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  CUBLAS_CALL_THROW(cublasLtCreate(&cublas_lt_handle_));
  CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, stream));

  if (info.enable_cudnn && cuda::CudnnLibrary::Get().Available()) {
    CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));
    CUDNN_CALL_THROW(cudnnSetStream(cudnn_handle_, stream));
    LOGS_DEFAULT(INFO) << "cuDNN version: " << cudnnGetVersion();
  }
#endif
  cuda_graph_.SetStream(stream);
}

CUDAExecutionProvider::PerThreadContext::~PerThreadContext() {
#ifndef USE_CUDA_MINIMAL
  ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasDestroy(cublas_handle_)));
  ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasLtDestroy(cublas_lt_handle_)));
  if (cudnn_handle_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnDestroy(cudnn_handle_)));
  }
#endif
}

bool CUDAExecutionProvider::PerThreadContext::IsGraphCaptureAllowed(
    CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  if (!IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id)) {
    return false;
  }
  if (graph_id_to_run_count_.find(cuda_graph_annotation_id) == graph_id_to_run_count_.end()) {
    return false;
  }
  return graph_id_to_run_count_.at(cuda_graph_annotation_id) >= min_num_runs_before_cuda_graph_capture_;
}

bool CUDAExecutionProvider::PerThreadContext::IsGraphCaptureAllowedOnRun(
    CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_.IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id);
}

CudaGraphAnnotation_t CUDAExecutionProvider::PerThreadContext::GetCudaGraphAnnotationId(
    const onnxruntime::RunOptions& run_options) const {
  auto graph_annotation_str =
      run_options.GetConfigOptions().GetConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation);
  // If graph annotation is not provided, fall back to the one cuda graph per session behavior
  CudaGraphAnnotation_t cuda_graph_annotation_id = 0;
  if (graph_annotation_str.has_value()) {
    ORT_ENFORCE(TryParseStringWithClassicLocale<int>(*graph_annotation_str, cuda_graph_annotation_id),
                "Failed to parse the cuda graph annotation id: ",
                *graph_annotation_str);
  }

  return cuda_graph_annotation_id;
}

void CUDAExecutionProvider::PerThreadContext::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  cuda_graph_.CaptureBegin(cuda_graph_annotation_id);
}

void CUDAExecutionProvider::PerThreadContext::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  cuda_graph_.CaptureEnd(cuda_graph_annotation_id);
}

bool CUDAExecutionProvider::PerThreadContext::IsGraphCaptured(CudaGraphAnnotation_t graph_annotation_id) const {
  return cuda_graph_.IsGraphCaptured(graph_annotation_id);
}

Status CUDAExecutionProvider::PerThreadContext::ReplayGraph(CudaGraphAnnotation_t graph_annotation_id, bool sync) {
  return cuda_graph_.Replay(graph_annotation_id, sync);
}

void CUDAExecutionProvider::PerThreadContext::IncrementRegularRunCountBeforeGraphCapture(
    CudaGraphAnnotation_t cuda_graph_annotation_id) {
  if (graph_id_to_run_count_.find(cuda_graph_annotation_id) == graph_id_to_run_count_.end()) {
    graph_id_to_run_count_[cuda_graph_annotation_id] = 1;
    return;
  }
  graph_id_to_run_count_[cuda_graph_annotation_id]++;
}

void OverrideTunableOpInfoByEnv(CUDAExecutionProviderInfo& info) {
  if (auto env_tunable_op_enable = onnxruntime::ParseTestOnlyEnvironmentVariable<bool>(
          "ORT_CUDA_TUNABLE_OP_ENABLE", {"0", "1"}, "Use provider_options \"tunable_op_enable\" instead.");
      env_tunable_op_enable.has_value() && env_tunable_op_enable != info.tunable_op.enable) {
    LOGS_DEFAULT(INFO) << "ORT_CUDA_TUNABLE_OP_ENABLE is set to " << *env_tunable_op_enable;
    info.tunable_op.enable = *env_tunable_op_enable;
  }

  if (auto env_tunable_op_tuning_enable = onnxruntime::ParseTestOnlyEnvironmentVariable<bool>(
          "ORT_CUDA_TUNABLE_OP_TUNING_ENABLE", {"0", "1"},
          "Use provider_options \"tunable_op_tuning_enable\" instead.");
      env_tunable_op_tuning_enable.has_value() && env_tunable_op_tuning_enable != info.tunable_op.tuning_enable) {
    LOGS_DEFAULT(INFO) << "ORT_CUDA_TUNABLE_OP_TUNING_ENABLE is set to " << *env_tunable_op_tuning_enable;
    info.tunable_op.tuning_enable = *env_tunable_op_tuning_enable;
  }

  if (info.tunable_op.tuning_enable && !info.tunable_op.enable) {
    LOGS_DEFAULT(WARNING) << "TunableOp is enabled for tuning but is not enabled for using. This will have no effect.";
  }
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCudaExecutionProvider,
                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                                   info.device_id)},
      qmoe_controller_{std::make_shared<onnxruntime::cuda::QMoESessionController>(info.device_id)},
      session_budget_account_{std::make_shared<SessionBudgetAccount>(
          GetCudaDeviceMemoryBudget(info.device_id),
          info.gpu_device_mem_limit > 0 ? info.gpu_device_mem_limit : std::numeric_limits<size_t>::max())},
      info_{info},
      tuning_context_(this, &info_.tunable_op) {
  if (qmoe_controller_) {
    qmoe_controller_->SetBudgetAccount(session_budget_account_);
  }
  ORT_ENFORCE(info_.gpu_device_mem_limit == 0 || !info_.external_allocator_info.UseExternalAllocator(),
              "gpu_device_mem_limit does not support external CUDA allocators");
#ifndef ENABLE_CUDA_NHWC_OPS
  ORT_ENFORCE(info_.prefer_nhwc == 0, "This build does not support NHWC layout");
#endif

  CUDA_CALL_THROW(cudaSetDevice(info_.device_id));

  // must wait GPU idle, otherwise cudaGetDeviceProperties might fail
  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, info_.device_id));

  // This scenario is not supported.
  ORT_ENFORCE(!(info.has_user_compute_stream && info.external_allocator_info.UseExternalAllocator()));

  if (info.has_user_compute_stream) {
    external_stream_ = true;
    use_ep_level_unified_stream_ = true;
    stream_ = static_cast<cudaStream_t>(info.user_compute_stream);
  } else {
    if (info.external_allocator_info.UseExternalAllocator()) {
      use_ep_level_unified_stream_ = true;
      stream_ = nullptr;
    } else if (info.enable_cuda_graph || info.use_ep_level_unified_stream) {
      // current cuda graph implementation only works with single stream
      // use EP level unified stream for all the reqeust
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
      use_ep_level_unified_stream_ = true;
    } else {
      stream_ = nullptr;
    }
  }

  size_t free = 0;
  size_t total = 0;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));

  OverrideTunableOpInfoByEnv(info_);

#ifdef USE_TRITON_KERNEL
  onnxruntime::cuda::LoadOrtTritonKernel();
#endif
}

DataLayout CUDAExecutionProvider::GetPreferredLayout() const {
  return this->IsNHWCPreferred() ? DataLayout::NHWC : DataLayout::NCHW;
}

std::optional<bool> CUDAExecutionProvider::ShouldConvertDataLayoutForOp([[maybe_unused]] std::string_view node_domain,
                                                                        [[maybe_unused]] std::string_view node_op_type,
                                                                        [[maybe_unused]] DataLayout target_data_layout) const {
#if defined(ENABLE_CUDA_NHWC_OPS)
  if (target_data_layout != DataLayout::NHWC) {
    return std::nullopt;
  }

  return cuda::IsNhwcEligible(node_domain, node_op_type);

#else  // defined(ENABLE_CUDA_NHWC_OPS)
  ORT_UNUSED_PARAMETER(node_domain);
  ORT_UNUSED_PARAMETER(node_op_type);
  ORT_UNUSED_PARAMETER(target_data_layout);
  return std::nullopt;
#endif
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  // Persist fpA_intB MatMulNBits tactics staged in memory to disk (best-effort, dirty-guarded).
  // No-op in builds without onnxruntime_USE_FPA_INTB_GEMM. The plugin EP flushes from ~CudaEp.
  onnxruntime::contrib::cuda::FlushMatMulNBitsTacticCaches();

  if (qmoe_controller_) {
    qmoe_controller_->Shutdown();
  }
  // clean up thread local context caches
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    for (const auto& cache_weak : context_state_.caches_to_update_on_destruction) {
      const auto cache = cache_weak.lock();
      if (!cache) continue;
      ORT_IGNORE_RETURN_VALUE(cache->erase(this));
    }
  }

  if (!external_stream_ && stream_) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream_)));
  }
}

ITuningContext* CUDAExecutionProvider::GetTuningContext() const {
  return const_cast<cuda::tunable::CudaTuningContext*>(&tuning_context_);
}

std::unique_ptr<profiling::EpProfiler> CUDAExecutionProvider::GetProfiler() {
  return std::make_unique<profiling::CudaProfiler>();
}

// Suppressing warning "C26816: The pointer points to memory allocated on the stack." for
// CUDAExecutionProvider::GetPerThreadContext().
// While CUDAExecutionProvider::GetPerThreadContext() does return the result of dereferencing a local
// std::shared_ptr<PerThreadContext>, the underlying PerThreadContext is owned by the CUDA EP and is not local to this
// function.
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 26816)
#endif

CUDAExecutionProvider::PerThreadContext& CUDAExecutionProvider::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    ORT_ENFORCE(cached_context);
    return *cached_context;
  }

  // get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);

    // get or create a context
    if (context_state_.retired_context_pool.empty()) {
      context = std::make_shared<PerThreadContext>(info_.device_id, stream_, info_);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // insert into active_contexts, should not already be present
    const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
    ORT_ENFORCE(active_contexts_insert_result.second);

    // insert into caches_to_update_on_destruction, may already be present
    ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

#ifdef _WIN32
#pragma warning(pop)
#endif

void CUDAExecutionProvider::ReleasePerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
  auto cached_context = cached_context_it->second.lock();
  ORT_ENFORCE(cached_context);

  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    context_state_.active_contexts.erase(cached_context);
    context_state_.retired_context_pool.push_back(cached_context);
  }

  per_thread_context_cache->erase(cached_context_it);
}

Status CUDAExecutionProvider::Sync() const {
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  return Status::OK();
}

Status CUDAExecutionProvider::OnSessionInitializationEnd() {
  if (qmoe_controller_) {
    ORT_RETURN_IF_ERROR(qmoe_controller_->PlanAndAdmitSession());
  }
  return Status::OK();
}

namespace {
// Ends a controller run on scope exit unless released; ORT does not call OnRunEnd for a provider whose OnRunStart failed.
class QMoEControllerRunRollback {
 public:
  QMoEControllerRunRollback() = default;
  QMoEControllerRunRollback(const QMoEControllerRunRollback&) = delete;
  QMoEControllerRunRollback& operator=(const QMoEControllerRunRollback&) = delete;
  ~QMoEControllerRunRollback() {
    if (controller_ != nullptr) {
      try {
        static_cast<void>(controller_->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
      } catch (...) {
      }
    }
  }
  void Arm(onnxruntime::cuda::QMoESessionController* controller) { controller_ = controller; }
  void Release() { controller_ = nullptr; }

 private:
  onnxruntime::cuda::QMoESessionController* controller_ = nullptr;
};
}  // namespace

Status CUDAExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  QMoEControllerRunRollback controller_rollback;
  if (qmoe_controller_ && qmoe_controller_->HasOffloadNodes()) {
    const char* strict_env = std::getenv("ORT_QMOE_REJECT_DISABLE_SYNC");
    if (strict_env != nullptr && std::strcmp(strict_env, "1") == 0 &&
        run_options.GetConfigOptions().GetConfigOrDefault(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "1") {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "QMoEOffload does not support disable_synchronize_execution_providers=1; required completion sync cannot be skipped.");
    }
    ORT_RETURN_IF_ERROR(qmoe_controller_->OnRunStart());
    controller_rollback.Arm(qmoe_controller_.get());
    if (IsGraphCaptureEnabled()) {
      const uint64_t cur_gen = qmoe_controller_->GraphGeneration();
      if (cur_gen != GetPerThreadContext().LastQMoEGraphGen()) {
        GetPerThreadContext().ResetForQMoEGraphGen(cur_gen);
      }
    }
  }
  CudaGraphAnnotation_t cuda_graph_annotation_id = GetPerThreadContext().GetCudaGraphAnnotationId(run_options);
  if (IsGraphCaptureEnabled() && !GetPerThreadContext().IsGraphCaptured(cuda_graph_annotation_id) &&
      GetPerThreadContext().IsGraphCaptureAllowed(cuda_graph_annotation_id)) {
    LOGS(*GetLogger(), INFO) << "Capturing the cuda graph for this model";
    GetPerThreadContext().CaptureBegin(cuda_graph_annotation_id);
  }
  controller_rollback.Release();
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) {
  const bool has_offload = qmoe_controller_ && qmoe_controller_->HasOffloadNodes();
  // Every exit below (error status or exception) must still end the controller run, otherwise the controller stays
  // active and rejects later runs as overlapping.
  QMoEControllerRunRollback controller_rollback;
  if (has_offload) controller_rollback.Arm(qmoe_controller_.get());

  Status capture_status = Status::OK();
  CudaGraphAnnotation_t cuda_graph_annotation_id = GetPerThreadContext().GetCudaGraphAnnotationId(run_options);
  if (IsGraphCaptureEnabled() && !GetPerThreadContext().IsGraphCaptured(cuda_graph_annotation_id)) {
    if (GetPerThreadContext().IsGraphCaptureAllowed(cuda_graph_annotation_id)) {
      GetPerThreadContext().CaptureEnd(cuda_graph_annotation_id);
      capture_status = [&]() -> Status {
        if (has_offload) {
          ORT_RETURN_IF_ERROR(qmoe_controller_->OnBeforeGraphReplay(static_cast<cudaStream_t>(stream_)));
        }
        // CUDA work issued to a capturing stream doesn’t actually run on the GPU,
        // so run the captured graph here to actually execute the work.
        ORT_RETURN_IF_ERROR(GetPerThreadContext().ReplayGraph(cuda_graph_annotation_id, sync_stream));
        if (has_offload) {
          ORT_RETURN_IF_ERROR(qmoe_controller_->OnAfterGraphReplay(static_cast<cudaStream_t>(stream_), sync_stream));
        }
        return Status::OK();
      }();
    } else {
      GetPerThreadContext().IncrementRegularRunCountBeforeGraphCapture(cuda_graph_annotation_id);
    }
  }
  if (has_offload) {
    controller_rollback.Release();
    const bool run_succeeded =
        capture_status.IsOK() &&
        run_options.GetConfigOptions().GetConfigOrDefault("__ort.run_outcome", "ok") == "ok";
    auto controller_status = qmoe_controller_->OnRunEnd(sync_stream, run_succeeded);
    ORT_RETURN_IF_ERROR(capture_status);
    ORT_RETURN_IF_ERROR(controller_status);
  } else {
    ORT_RETURN_IF_ERROR(capture_status);
  }

  if (sync_stream) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  }

  // The reason of !IsGraphCaptureEnabled():
  //  If cuda graph is enabled, the per thread context will not be released
  //  because the per thread cuda graph needs to be maintained and replayed for
  //  the next run.
  // The reason of PerThreadContextCache()->find(this) != PerThreadContextCache()->end():
  //  In extreme cases (e.g., 1-op graph and that op fallbacks to CPU),
  //  PerThreadContext won't be created and there isbe nothing to
  //  release. This didn't happen before because we always call
  //  GetPerThreadContext in OnRunStart.
  if (!IsGraphCaptureEnabled() &&
      PerThreadContextCache()->find(this) != PerThreadContextCache()->end()) {
    ReleasePerThreadContext();
  }

  return Status::OK();
}

bool CUDAExecutionProvider::IsGraphCaptureEnabled() const {
  return info_.enable_cuda_graph;
}

bool CUDAExecutionProvider::IsGraphCaptured(int graph_annotation_id) const {
  if (qmoe_controller_ && qmoe_controller_->HasOffloadNodes()) {
    const uint64_t cur_gen = qmoe_controller_->GraphGeneration();
    if (cur_gen != GetPerThreadContext().LastQMoEGraphGen()) {
      GetPerThreadContext().ResetForQMoEGraphGen(cur_gen);
    }
  }
  return GetPerThreadContext().IsGraphCaptured(graph_annotation_id);
}

Status CUDAExecutionProvider::ReplayGraph(int graph_annotation_id, bool sync) {
  if (qmoe_controller_ && qmoe_controller_->HasOffloadNodes()) {
    ORT_RETURN_IF_ERROR(qmoe_controller_->OnRunStart());
    auto prep_st = qmoe_controller_->OnBeforeGraphReplay(static_cast<cudaStream_t>(stream_));
    if (!prep_st.IsOK()) {
      static_cast<void>(qmoe_controller_->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
      return prep_st;
    }
    auto replay_st = GetPerThreadContext().ReplayGraph(graph_annotation_id, /*sync=*/true);
    if (!replay_st.IsOK()) {
      static_cast<void>(qmoe_controller_->OnRunEnd(/*sync_stream=*/false, /*run_succeeded=*/false));
      return replay_st;
    }
    auto after_st = qmoe_controller_->OnAfterGraphReplay(static_cast<cudaStream_t>(stream_), sync);
    auto end_st = qmoe_controller_->OnRunEnd(sync, after_st.IsOK());
    ORT_RETURN_IF_ERROR(after_st);
    ORT_RETURN_IF_ERROR(end_st);
    return Status::OK();
  }
  return GetPerThreadContext().ReplayGraph(graph_annotation_id, sync);
}

void CUDAExecutionProvider::RetainBufferForGraphCapture(std::shared_ptr<void> buffer) const {
  std::lock_guard<std::mutex> lock(captured_host_buffers_mutex_);
  captured_host_buffers_.push_back(std::move(buffer));
}

void CUDAExecutionProvider::SetSessionModelPath(std::filesystem::path path) const {
  if (!path.empty() && path.is_relative()) {
    std::error_code ec;
    std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    if (!ec && !abs_path.empty()) {
      path = std::move(abs_path);
    }
  }
  std::lock_guard<std::mutex> lock(model_path_mutex_);
  model_path_ = std::move(path);
  if (qmoe_controller_) {
    qmoe_controller_->SetModelPath(model_path_);
  }
}

std::filesystem::path CUDAExecutionProvider::GetSessionModelPath() const {
  std::lock_guard<std::mutex> lock(model_path_mutex_);
  return model_path_;
}

namespace cuda {

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  return {};
}

#ifndef DISABLE_ML_OPS
// LabelEncoder (ML domain) - numeric types only
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, float_int64, LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, int64_float, LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, float_float, LabelEncoder);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, int64_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_float, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, float_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, float_float, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, double_double, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, double_int64, LabelEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_double, LabelEncoder);

Status RegisterOnnxMLOperatorKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing

      // LabelEncoder (ML domain) - numeric types only
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, float_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, int64_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, float_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 2, 3, int64_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, float_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, float_float, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, double_double, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, double_int64, LabelEncoder)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMLDomain, 4, int64_double, LabelEncoder)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
#endif

// opset 1 to 9
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
#ifndef USE_CUDA_MINIMAL
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, float, Cos);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, double, Cos);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, MLFloat16, Cos);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, float, Sin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, double, Sin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, MLFloat16, Sin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, float, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, double, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, double, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, Softsign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, Softsign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, Softsign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, Softplus);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, Softplus);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, Softplus);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, float, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, double, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, Sum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Sum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Min);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 10, float, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, LpNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, LpNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, MLFloat16, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceL1);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, BFloat16, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, int32_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, int64_t, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, Tile);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, float, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, double, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, MLFloat16, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, int32_t, Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, uint8_t, Upsample);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 20, ConstantOfShape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Shrink);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, uint8_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, NonZero);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 8, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, SpaceToDepth);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomNormal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomNormalLike);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomUniform);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomUniformLike);

// opset 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 11, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 27, ReverseSequence);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 15, float, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 15, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, float, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, MLFloat16, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, Mod);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 19, IsInf);

// opset 11
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, Compress);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Gather);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, GatherElements);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Loop);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 26, Range);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 15, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterElements);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 23, TopK);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceAt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceConstruct);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceEmpty);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceLength);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ConcatFromSequence);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceErase);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceInsert);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 13, CumSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int32_t_float_int32_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_MLFloat16_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int32_t_MLFloat16_int32_t, OneHot);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace);

// OpSet 12
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Clip);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, double, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, int8_t, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, uint8_t, MaxPool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Pow);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 27, Einsum);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ArgMin);

// OpSet 13
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp);
// Add bf16 support for Exp in opset 13+ for phimm model
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf);
// Add bf16 support for Erf in opset 13+ for phimm model
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Erf);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Expand);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Sum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Max);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, float, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, double, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, bool, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 15, ScatterElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Unsqueeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, GatherElements);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 19, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceLogSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceLogSumExp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, BFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, Loop);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LRN);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 15, ScatterND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, bool, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27, SpaceToDepth);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27, DepthToSpace);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, DepthToSpace);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sign);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, BFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, ReduceSum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27, Mod);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Mod);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, DequantizeLinear);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMin);

// OpSet 14
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, CumSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, GRU);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, GRU);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 18, Identity);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 18, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, RNN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kCudaExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kCudaExecutionProvider, kOnnxDomain, 14, 14, double, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kCudaExecutionProvider, kOnnxDomain, 14, 14, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Trilu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, HardSwish);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, HardSwish);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, HardSwish);

// OpSet 15
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, double, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, 18, Shape);

// Opset 16
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 18, Scan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, BFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint8_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 17, ScatterElements);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 17, ScatterND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 19, float, GridSample);

// Opset 17
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, double, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization);

// Opset 18
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, Split);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, ScatterElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, ScatterND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, bool, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, double, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, int32_t, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, uint8_t, Resize);

// Opset 19
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, bool, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, float, DeformConv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, double, DeformConv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, MLFloat16, DeformConv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, BFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, bool, Cast);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, Cast);
#endif

class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, float, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, float, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, MLFloat16, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, MLFloat16, DequantizeLinear);
#endif

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Loop);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, float, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, float, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, MLFloat16, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, MLFloat16, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, double, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint8_t, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Shape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint32_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint64_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, float, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, double, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, MLFloat16, Equal);

// Opset 20
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, BFloat16, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, Gelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, IsInf);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, 21, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMin);

// Opset 21.
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, BFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, bool, Cast);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, Cast);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, ConstantOfShape);
// TODO(fajin): support other quantized types
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, MLFloat16, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, MLFloat16, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Loop);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, bool, Pad);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, MLFloat16, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, MLFloat16, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Unsqueeze);

// Opset 22.
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, LpNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, LpNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, LpNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, DeformConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, DeformConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, DeformConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, DeformConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GRU);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, HardSwish);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, HardSwish);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, HardSwish);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, HardSwish);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, LSTM);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomNormal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomNormalLike);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomUniform);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomUniformLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, RNN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, RoiAlign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Softsign);

// Opset 23.
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, float, Attention);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, MLFloat16, Attention);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, BFloat16, Attention);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, BFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, bool, Cast);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, Cast);
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float4E2M1x2, Cast);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, ConstantOfShape);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, MLFloat16, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, float, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, MLFloat16, DequantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Loop);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, bool, Pad);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, MLFloat16, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, float, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, MLFloat16, QuantizeLinear);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float_float, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, double_double, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16_MLFloat16, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, BFloat16_BFloat16, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float_MLFloat16, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16_float, RMSNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, BFloat16, RotaryEmbedding);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, Unsqueeze);

// Opset 24.
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, float, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, MLFloat16, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, BFloat16, Attention);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, double, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, bool, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, TensorScatter);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, TopK);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, Unsqueeze);

// Opset 25.
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, BFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, bool, Cast);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2, Cast);
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float4E2M1x2, Cast);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, ConstantOfShape);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, MLFloat16, DequantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, float, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, MLFloat16, DequantizeLinear);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Loop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, float, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, double, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, MLFloat16, Pad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, bool, Pad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, MLFloat16, QuantizeLinear);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, float, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, MLFloat16, QuantizeLinear);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Reshape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Scan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Size);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Transpose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Unsqueeze);

// Opset 27.
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 27, Range);

// Opset 28.
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, BFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, bool, Cast);
#if !defined(DISABLE_FLOAT8_TYPES)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float8E4M3FN, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float8E5M2, Cast);
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float4E2M1x2, Cast);
#endif
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Compress);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t_float_int32_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_MLFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t_MLFloat16_int32_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_BFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t_BFloat16_int32_t, OneHot);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, ReverseSequence);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Einsum);
#endif

static Status RegisterCudaKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
#ifndef USE_CUDA_MINIMAL
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, float, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, double, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, double, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 21, MLFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 10, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, float, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, double, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 21, MLFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, float, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, double, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, bool, Xor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 7, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 11, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 12, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, uint64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, double, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, bool, Not)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 13, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, double, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 11, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, BFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 4, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, 12, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 9, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 20, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int8_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int16_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int32_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, int64_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, double, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, EyeLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scatter)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, double_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 15, uint8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 9, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 8, 8, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, 10, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 12, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomNormal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomNormalLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomUniform)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 21, RandomUniformLike)>,

      // opset 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 11, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 27, ReverseSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 15, float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 15, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 21, MLFloat16, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 10, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 10, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, uint8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, 12, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10,
                                                                      19, IsInf)>,

      // opset 11
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, NonMaxSuppression)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 26, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 15, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 23, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceAt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceConstruct)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceEmpty)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceLength)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, ConcatFromSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceErase)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, SequenceInsert)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 18, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 11, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, float, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, double, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int64_t_MLFloat16_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 27, int32_t_MLFloat16_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace)>,

      // OpSet 12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Clip)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, double, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, int8_t, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, uint8_t, MaxPool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 27, Einsum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, ArgMin)>,

      // OpSet 13
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 19, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 15, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, BFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, float, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, double, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 20, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 15, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 17, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27,
                                                                      SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27,
                                                                      DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int8_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int16_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int32_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, int64_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint8_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint16_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint32_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, uint64_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sign)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, BFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 13, BFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, BFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 27, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, int8_t, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, DequantizeLinear)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, double, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMin)>,

      // OpSet 14
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int16_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint16_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 18, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 18, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kOnnxDomain, 14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kOnnxDomain, 14, 14, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kOnnxDomain, 14, 14, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, BFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, Trilu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, float, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, double, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 14, 21, MLFloat16, HardSwish)>,

      // OpSet 15
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, double, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 15, 18, Shape)>,

      // Opset 16
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 18, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, BFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint8_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, uint64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, double, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 17, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 17, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 16, 19, float, GridSample)>,

      // Opset 17
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, double, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization)>,

      // Opset 18
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 19, uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, ScatterElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 18, 18, uint8_t, Resize)>,

      // Opset 19-20
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, float, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, double, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 21, MLFloat16, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, BFloat16, Cast)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, Cast)>,
#endif

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, float, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, float, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, MLFloat16, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, MLFloat16, DequantizeLinear)>,
#endif

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Loop)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, float, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, float, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, uint8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, int8_t, MLFloat16, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E4M3FN, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Float8E5M2, MLFloat16, QuantizeLinear)>,
#endif

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, uint64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, float, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, double, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, MLFloat16, Equal)>,

      // Opset 20
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, Gelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, Gelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, BFloat16, Gelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, Gelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, IsInf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, 21, float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, float, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, double, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, MLFloat16, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int32_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, uint8_t, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 20, int64_t, ReduceMin)>,

      // Opset 21
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, BFloat16, Cast)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, Cast)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, ConstantOfShape)>,
      // TODO(fajin): support other quantized types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, MLFloat16, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, MLFloat16, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, uint8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, int8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, UInt4x2, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Int4x2, MLFloat16, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E4M3FN, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Float8E5M2, MLFloat16, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 21, 22, Unsqueeze)>,

      // Opset 22
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, LpNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, DeformConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, GRU)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, HardSwish)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, LSTM)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomNormal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomNormalLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomUniform)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, RandomUniformLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, RNN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, float, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, double, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, MLFloat16, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 22, BFloat16, Softsign)>,

      // Opset 23
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, float, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, MLFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, BFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, BFloat16, Cast)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, Cast)>,
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float4E2M1x2, Cast)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, MLFloat16, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, MLFloat16, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, uint8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, int8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, UInt4x2, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Int4x2, MLFloat16, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E4M3FN, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Float8E5M2, MLFloat16, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float_float, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, double_double, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16_MLFloat16, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, BFloat16_BFloat16, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float_MLFloat16, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16_float, RMSNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, float, RotaryEmbedding)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, MLFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, BFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 24, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 23, 23, Unsqueeze)>,

      // Opset 24
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, float, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, MLFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, BFloat16, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, TensorScatter)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 24, 24, Unsqueeze)>,

      // Opset 25
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, BFloat16, Cast)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float8E4M3FN, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float8E5M2, Cast)>,
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, 27, Float4E2M1x2, Cast)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, ConstantOfShape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, MLFloat16, DequantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, float, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, MLFloat16, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, If)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Loop)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, double, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, bool, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, uint8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, int8_t, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, UInt4x2, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Int4x2, MLFloat16, QuantizeLinear)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, float, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E4M3FN, MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Float8E5M2, MLFloat16, QuantizeLinear)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Scan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 25, Unsqueeze)>,
      // Opset 27
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 27, Range)>,
      // Opset 28
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, BFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, bool, Cast)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float8E4M3FN, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float8E5M2, Cast)>,
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Float4E2M1x2, Cast)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Compress)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_int64_t_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_float_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t_float_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int64_t_MLFloat16_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, int32_t_MLFloat16_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28,
                                                                  int64_t_BFloat16_int64_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28,
                                                                  int32_t_BFloat16_int32_t, OneHot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, ReverseSequence)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 28, Einsum)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

#ifndef USE_CUDA_MINIMAL
#ifndef DISABLE_CONTRIB_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::cuda::RegisterCudaContribKernels(kernel_registry));
#endif

#ifdef ENABLE_CUDA_NHWC_OPS
#ifndef DISABLE_CONTRIB_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::cuda::RegisterCudaNhwcContribKernels(kernel_registry));
#endif
  ORT_RETURN_IF_ERROR(::onnxruntime::cuda::RegisterCudaNhwcKernels(kernel_registry));
#endif

#ifdef ENABLE_TRAINING_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::cuda::RegisterCudaTrainingKernels(kernel_registry));
#endif
#endif

  return Status::OK();
}

}  // namespace cuda

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();
  ORT_THROW_IF_ERROR(cuda::RegisterCudaKernels(*s_kernel_registry));
#ifndef DISABLE_ML_OPS
  ORT_THROW_IF_ERROR(cuda::RegisterOnnxMLOperatorKernels(*s_kernel_registry));
#endif
}

void DeleteRegistry() {
  // Save any tactic caches of sessions still alive when the provider is shut down, before worker
  // threads are stopped or the provider DLL is unloaded.
  onnxruntime::contrib::cuda::FlushMatMulNBitsTacticCaches();
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> CUDAExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

static bool RNNNeedFallbackToCPU(const onnxruntime::Node& node,
                                 const std::vector<std::string> activations_supported,
                                 const std::string& op_type) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

    if ("activation_alpha" == attr_name || "activation_beta" == attr_name || "clip" == attr_name) {
      return true;
    }

    if ("activations" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS == attr_value.type()) {
      for (int i = 0; i < attr_value.strings_size(); ++i) {
        std::string activation_lowercase(attr_value.strings(i));
        std::transform(activation_lowercase.begin(), activation_lowercase.end(), activation_lowercase.begin(),
                       [](const unsigned char i) { return static_cast<char>(::tolower(i)); });
        if (activations_supported[i] != activation_lowercase) {
          return true;
        }
      }
    }

    if ("LSTM" == op_type &&
        "input_forget" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      if (0 != attr_value.i()) {
        return true;
      }
    }

    if ("GRU" == op_type &&
        "linear_before_reset" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      // cudnn GRU only support linear_before_reset = 1
      if (1 != attr_value.i()) {
        return true;
      }
    }
  }

  if ("LSTM" == op_type) {
    // cudnn LSTM not support peephole
    auto input_defs = node.InputDefs();
    if (8 == input_defs.size()) {
      auto peephole = input_defs.at(7);
      if (peephole->Exists()) {
        return true;
      }
    }
  }
  return false;
}

static bool ConvTransposeNeedFallbackToCPU([[maybe_unused]] const onnxruntime::Node& node,
                                           [[maybe_unused]] const logging::Logger& logger,
                                           [[maybe_unused]] const GraphViewer& graph_viewer,
                                           [[maybe_unused]] const bool prefer_nhwc) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

    // cudnn only supports symmetric padding, so drop the node down to CPU if the padding provided is asymmetric
    // TODO: Check if we can adopt a similar approach to deal with asymmetric pads in 'ConvTranspose'
    // as we did for 'Conv' to circumvent the cudnn limitation
    if ("pads" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS == attr_value.type()) {
      auto& pads = attr_value.ints();
      int pads_size = pads.size();
      ORT_ENFORCE(pads_size % 2 == 0);
      int rank = pads_size / 2;
      for (int i = 0; i < rank; i++) {
        if (pads.Get(i) != pads.Get(i + rank)) {
          LOGS(logger, WARNING) << "Dropping the ConvTranspose node: " << node.Name()
                                << " to CPU because it requires asymmetric padding which the CUDA EP"
                                << " currently does not support";
          return true;
        }
      }
    }

    if ("auto_pad" == attr_name &&
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING == attr_value.type()) {
      auto& auto_pad_attr = attr_value.s();
      ORT_ENFORCE(auto_pad_attr == "SAME_UPPER" || auto_pad_attr == "SAME_LOWER" ||
                      auto_pad_attr == "VALID" || auto_pad_attr == "NOTSET",
                  "auto_pad must be either NOTSET, VALID, SAME_UPPER, SAME_LOWER");

      // If auto_pad is SAME_UPPER or SAME_LOWER, pads will be computed dynamically at runtime
      // based on the provided input shape. This may or may not lead to symmetric padding.
      // If it turns out to be asymmetric padding, CuDNN will return a cryptic unfriendly error message.
      // So drop down the node to CPU if auto_pad is SAME_UPPER or SAME_LOWER even if it may lead to
      // symmetric padding.
      // TODO: Remove this after we have supported asymmetric padding in the CUDA ConvTranspose kernel
      if (auto_pad_attr == "SAME_UPPER" || auto_pad_attr == "SAME_LOWER") {
        LOGS(logger, WARNING) << "Dropping the ConvTranspose node: " << node.Name()
                              << " to CPU because it uses the auto_pad attribute which may lead to asymmetric padding which"
                              << " the CUDA EP currently does not support";
        return true;
      }
    }
  }

#ifdef ENABLE_CUDA_NHWC_OPS
  if (prefer_nhwc) {
    // NHWC implementation doesn't handle transpose of W if it's not an initializer
    if (!graph_viewer.IsConstantInitializer(node.InputDefs()[1]->Name(), true)) {
      return true;
    }
  }
#endif

  return false;
}

static bool NhwcConvNeedFallbackToCPU(const onnxruntime::Node& node, const logging::Logger& logger,
                                      [[maybe_unused]] const GraphViewer& graph_viewer,
                                      [[maybe_unused]] const bool prefer_nhwc) {
  // NHWC implementation doesn't handle W in NHWC layout if it's not an initializer
  if (!graph_viewer.IsConstantInitializer(node.InputDefs()[1]->Name(), true)) {
    LOGS(logger, WARNING) << "Dropping the NhwcConv node: " << node.Name()
                          << " to CPU because the Cuda EP requires W as initializer for NHWC operation.";
    return true;
  }

  return false;
}

static bool CastNeedFallbackToCPU(const onnxruntime::Node& node) {
  const auto& node_attributes = node.GetAttributes();
  // Check attributes
  for (auto& attr : node_attributes) {
    auto& attr_name = attr.first;
    auto& attr_value = attr.second;

    // string is not supported
    if ("to" == attr_name && ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT == attr_value.type()) {
      auto to_type = attr_value.i();
      if (to_type == ::ONNX_NAMESPACE::TensorProto_DataType_STRING)
        return true;
    }
  }

  return false;
}

static bool ArgMaxOrArgMinNeedFallbackToCPU(const onnxruntime::Node& node) {
  // Opset 12 introduced the attribute "select_last_index"
  if (node.SinceVersion() >= 12) {
    const auto& node_attributes = node.GetAttributes();

    for (auto& attr : node_attributes) {
      auto& attr_name = attr.first;
      auto& attr_value = attr.second;

      // CuDNN doesn't support picking the last index in case of encountering
      // duplicate max values.
      // CuDNN's API doc doesn't mention what happens in case duplicates are encountered,
      // but based on testing, the results seem to indicate a "stable" implementation
      // (i.e.) relative ordering is preserved which is the expected behavior when the
      // attribute takes on the default value (most common use-case for this operator).
      if ("select_last_index" == attr_name) {
        if (attr_value.i() != 0) {
          return true;
        }
      }
    }
  }

  return false;
}

std::unique_ptr<onnxruntime::IDataTransfer> CUDAExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::GPUDataTransfer>();
}

std::unique_ptr<onnxruntime::IExternalDataLoader> CUDAExecutionProvider::GetExternalDataLoader() const {
  if (info_.external_data_loader_reading_threads == 0) {
    return nullptr;
  }

  return std::make_unique<cuda::ExternalDataLoader>(
      info_.device_id, info_.external_data_loader_reading_threads);
}

std::vector<std::unique_ptr<ComputeCapability>>
CUDAExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const IKernelLookup& kernel_lookup,
                                     const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                     IResourceAccountant* resource_accountant) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  const logging::Logger& logger = *GetLogger();

  if (!graph.ModelPath().empty()) {
    SetSessionModelPath(graph.ModelPath());
  }

  // Figure out the memory limit if accountant is available
  size_t memory_threshold = std::numeric_limits<size_t>::max();
  SafeInt<size_t> consumed_memory = 0;
  if (resource_accountant != nullptr) {
    if (resource_accountant->IsStopIssued()) {
      LOGS(logger, WARNING) << "CUDA_EP returning due to Stop Set";
      return result;
    }

    auto threshold = resource_accountant->GetThreshold();
    if (!threshold) {
      // info_.gpu_mem_limit is for BFC arena
      size_t free_memory, total_memory;
      if (0 != cudaMemGetInfo(&free_memory, &total_memory)) {
        memory_threshold = info_.gpu_mem_limit;
        LOGS(logger, INFO)
            << "CUDA_EP failed to get available GPU memory info. Using info_.gpu_mem_limit instead: " << info_.gpu_mem_limit;
      } else {
        memory_threshold = std::min(free_memory, info_.gpu_mem_limit);
        LOGS(logger, VERBOSE)
            << "CUDA_EP Using threshold: " << memory_threshold << " Free memory reported: " << free_memory;
      }
    } else {
      memory_threshold = std::get<0>(*threshold);
    }

    consumed_memory = std::get<0>(resource_accountant->GetConsumedAmount());
  }

  InlinedHashSet<NodeIndex> previously_assigned_nodes;
  // On repeated calls to this function, we may have most of the nodes already
  // assigned to a CUDA EP capability. We'll skip accounting for these nodes.
  previously_assigned_nodes.reserve(graph.NumberOfNodes());
  InlinedVector<NodeIndex> candidates;
  // A subset of the above vector. A subset of the tentative_nodes might be moved to CPU.
  InlinedVector<NodeIndex> tentative_nodes;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      if (node.GetExecutionProviderType() == kCudaExecutionProvider) {
        candidates.push_back(node.Index());
        previously_assigned_nodes.insert(node.Index());
      }
      continue;
    }

    const KernelCreateInfo* cuda_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has a CUDA kernel for this node
    if (cuda_kernel_def == nullptr) {
      LOGS(logger, INFO) << "CUDA kernel not found in registries for Op type: " << node.OpType() << " node name: " << node.Name();
      continue;
    }

    bool not_supported = false;
    bool force_inside = false;  // for some compute heavy ops, we'll force it to run inside CUDA
    if ("LSTM" == node.OpType()) {
      // the supported activations covers the bidirectional mode
      std::vector<std::string> activations_supported{"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("RNN" == node.OpType()) {
      std::vector<std::string> activations_supported{"tanh", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("GRU" == node.OpType()) {
      std::vector<std::string> activations_supported{"sigmoid", "tanh", "sigmoid", "tanh"};
      not_supported = RNNNeedFallbackToCPU(node, activations_supported, node.OpType());
      force_inside = !not_supported;
    } else if ("ConvTranspose" == node.OpType()) {
      not_supported = ConvTransposeNeedFallbackToCPU(node, logger, graph, IsNHWCPreferred());
      force_inside = !not_supported;
    } else if ("ArgMax" == node.OpType() || "ArgMin" == node.OpType()) {
      not_supported = ArgMaxOrArgMinNeedFallbackToCPU(node);
      force_inside = !not_supported;
    } else if ("Cast" == node.OpType()) {
      not_supported = CastNeedFallbackToCPU(node);
      // cast is not compute heavy, and may be placed outside
    } else if ("NhwcConv" == node.OpType()) {
      not_supported = NhwcConvNeedFallbackToCPU(node, logger, graph, IsNHWCPreferred());
      force_inside = !not_supported;
    }

    if (!force_inside && not_supported) {
      if (not_supported) {
        LOGS(logger, WARNING) << "CUDA kernel not supported. Fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      }
    } else {
      tentative_nodes.push_back(node.Index());
      candidates.push_back(node.Index());
    }
  }

  // For CUDA EP, exclude the subgraph that is preferred to be placed in CPU
  // These are usually shape related computation subgraphs
  // Following logic can be extended for other EPs
  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, tentative_nodes, logger);
  for (auto& node_index : candidates) {
    if (cpu_nodes.count(node_index) > 0)
      continue;

    // Previously assigned nodes have been accounted before
    if (previously_assigned_nodes.count(node_index) > 0 || resource_accountant == nullptr) {
      auto sub_graph = IndexedSubGraph::Create();
      sub_graph->Nodes().push_back(node_index);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    } else {
      auto* node = graph.GetNode(node_index);
      std::optional<Level1MemoryEstimate> level1_memory_estimate;

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
      // Level 1 (Phase-A memory roadmap, issue microsoft/onnxruntime#29775): a partition-time,
      // kernel-independent memory estimate for MatMulNBits. Runtime workspace and prepack
      // allocations remain separate so their different lifetimes are visible in reporting.
      if (node != nullptr && node->OpType() == "MatMulNBits" && node->Domain() == kMSDomain) {
        const auto& estimator_config = resource_accountant->GetWorkspaceEstimatorConfig();
        const auto& fpa_intb_gemm = estimator_config.cuda_fpa_intb_gemm;
        const auto& profile_m = estimator_config.cuda_fpa_intb_profile_m;
        const auto& m_chunk_size = estimator_config.cuda_matmul_nbits_m_chunk_size;
        const auto& inferred_shapes = resource_accountant->GetMaxShapeInferenceResult();
        const auto& input_defs = node->InputDefs();
        const TensorShape* input_a_shape =
            inferred_shapes.Empty() || input_defs.empty() || input_defs[0] == nullptr
                ? nullptr
                : inferred_shapes.GetShape(&graph.GetGraph(), input_defs[0]->Name());
        const contrib::cuda::MatMulNBitsMemoryEstimateOptions estimate_options{
            fpa_intb_gemm.has_value()
                ? std::optional<std::string_view>{*fpa_intb_gemm}
                : std::nullopt,
            profile_m.has_value()
                ? std::optional<std::string_view>{*profile_m}
                : std::nullopt,
            /*input_shape_is_upper_bound=*/input_a_shape != nullptr,
            m_chunk_size.has_value()
                ? std::optional<std::string_view>{*m_chunk_size}
                : std::nullopt};
        level1_memory_estimate =
            input_a_shape != nullptr
                ? contrib::cuda::EstimateMatMulNBitsMemory(
                      *node, input_a_shape->GetDims(), GetDeviceProp(), estimate_options)
                : contrib::cuda::EstimateMatMulNBitsMemory(
                      *node, GetDeviceProp(), estimate_options);
        if (level1_memory_estimate.has_value() &&
            level1_memory_estimate->runtime_workspace_bytes.has_value()) {
          LOGS(logger, VERBOSE) << "Level-1 memory estimate for " << node->Name()
                                << ": runtime workspace="
                                << level1_memory_estimate->runtime_workspace_bytes.value_or(0)
                                << " bytes, runtime transient="
                                << level1_memory_estimate->runtime_transient_bytes
                                << " bytes, persistent prepack="
                                << level1_memory_estimate->persistent_prepack_bytes
                                << " bytes, initialization scratch="
                                << level1_memory_estimate->initialization_scratch_bytes << " bytes";
        } else {
          LOGS(logger, WARNING)
              << "CUDA_EP declining capacity-aware assignment for MatMulNBits node "
              << node->Name() << " because "
              << (level1_memory_estimate.has_value()
                      ? "ordinary runtime workspace is unknown for the admitted input shapes."
                      : "no proven route-complete memory estimate is available for this configuration.");
          continue;
        }
      }
#endif

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
      if (node != nullptr &&
          (node->OpType() == "PackedAttention" ||
           node->OpType() == "PackedMultiHeadAttention") &&
          node->Domain() == kMSDomain) {
        const auto input_shapes = ResolveNodeInputShapes(
            *node, &graph.GetGraph(),
            resource_accountant->GetMaxShapeInferenceResult());
        const auto ws = contrib::cuda::EstimatePackedAttentionWorkspace(
            *node, gsl::make_span(input_shapes), GetDeviceProp(),
            *GetAttentionKernelOptions());
        if (ws.has_value()) {
          Level1MemoryEstimate estimate;
          contrib::cuda::SetPackedAttentionLevel1MemoryEstimate(*ws, estimate);
          level1_memory_estimate = estimate;
          LOGS(logger, VERBOSE) << "Level-1 memory estimate for " << node->Name()
                                << ": runtime workspace="
                                << ws->total_workspace_bytes << " bytes";
        } else {
          LOGS(logger, WARNING)
              << "CUDA_EP declining capacity-aware assignment for " << node->OpType()
              << " node " << node->Name()
              << " because required shape metadata or a proven route-complete workspace estimate is unavailable.";
          continue;
        }
      }

      if (node != nullptr && node->OpType() == "GroupQueryAttention" &&
          node->Domain() == kMSDomain) {
        const auto input_shapes = ResolveNodeInputShapes(
            *node, &graph.GetGraph(),
            resource_accountant->GetMaxShapeInferenceResult());
        const auto& input_defs = node->InputDefs();
        const bool head_sink_is_constant_initializer =
            input_defs.size() > 11 && input_defs[11] != nullptr &&
            input_defs[11]->Exists() &&
            graph.IsConstantInitializer(input_defs[11]->Name(), true);
        // The factory validates the caller-declared envelope once.
        const auto& estimator_config = resource_accountant->GetWorkspaceEstimatorConfig();
        const auto gqa_workspace_max_total_sequence_length =
            estimator_config.cuda_gqa_workspace_max_total_sequence_length;
        // Reader-only intent; no value changes route selection or sizing.
        KernelDispatchPolicy kernel_dispatch_policy = KernelDispatchPolicy::Auto;
        if (const auto& policy = estimator_config.kernel_dispatch_policy;
            policy.has_value()) {
          ORT_ENFORCE(
              TryParseKernelDispatchPolicy(*policy, kernel_dispatch_policy),
              "Invalid ", kOrtSessionOptionsKernelDispatchPolicy,
              " in WorkspaceEstimatorConfig: '", *policy, "'");
        }
        const auto ws = contrib::cuda::EstimateGroupQueryAttentionWorkspace(
            *node, gsl::make_span(input_shapes), GetDeviceProp(),
            *GetAttentionKernelOptions(), head_sink_is_constant_initializer,
            gqa_workspace_max_total_sequence_length, kernel_dispatch_policy);
        if (ws.has_value()) {
          Level1MemoryEstimate estimate;
          contrib::cuda::SetGroupQueryAttentionLevel1MemoryEstimate(*ws, estimate);
          level1_memory_estimate = estimate;
          LOGS(logger, VERBOSE) << "Level-1 memory estimate for " << node->Name()
                                << ": runtime workspace="
                                << ws->total_workspace_bytes << " bytes";
        } else {
          // The generic safety margin is an ad-hoc fallback, not a proven GQA
          // workspace bound. Capacity-aware placement therefore fails closed.
          LOGS(logger, WARNING)
              << "CUDA_EP declining capacity-aware assignment for GroupQueryAttention node "
              << node->Name() << " because no proven route-aware workspace estimate is available.";
          continue;
        }
      }
#endif

      const auto resource_count_variant =
          resource_accountant->ComputeResourceCount(*node, level1_memory_estimate);
      const auto resource_count = std::get<size_t>(resource_count_variant);
      const auto would_be_consumed = resource_count + consumed_memory;
      LOGS(logger, INFO) << "CUDA_EP Node: " << node_index << " Memory usage : " << resource_count
                         << " would be consumed " << static_cast<size_t>(would_be_consumed)
                         << " threshold: " << memory_threshold;
      if (would_be_consumed < memory_threshold) {
        consumed_memory = would_be_consumed;
        auto sub_graph = IndexedSubGraph::Create();
        sub_graph->SetAccountant(resource_accountant);
        sub_graph->Nodes().push_back(node_index);
        sub_graph->AppendNodeCost(resource_count);
        result.push_back(ComputeCapability::Create(std::move(sub_graph)));
      } else {
        // We break here so we do not have patches of CUDA assigned nodes.
        if (node != nullptr) {
          LOGS(logger, WARNING) << "CUDA_EP Halting assignment due to capacity threshold at node: "
                                << node->Name() << " index: " << node_index;
        }

        resource_accountant->SetStopAssignment();
        break;
      }
    }
  }

  /*
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }*/
  return result;
}

void CUDAExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
                                                   AllocatorMap& allocators) const {
  // This allocator must be the same to the allocator
  // used in AllocateBufferOnCPUPinned.
  auto allocator = allocators[GetOrtDeviceByMemType(OrtMemTypeCPU)];
  RegisterCudaStreamHandles(stream_handle_registry,
                            OrtDevice::GPU,
                            allocator,
                            !IsGraphCaptureEnabled(),
                            stream_,
                            use_ep_level_unified_stream_,
                            GetPerThreadContext().CudnnHandle(),
                            GetPerThreadContext().CublasHandle(),
                            info_);
}

OrtDevice CUDAExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  // TODO(leca): For CpuInput, return default OrtDevice to make it consistent with previous logic, otherwise, it will fail GradientCheckerTest.TileGrad
  // in Windows training scenario. However, we need to figure out why PINNED memType fails
  if (mem_type == OrtMemTypeCPUInput)
    return OrtDevice();
  if (mem_type == OrtMemTypeCPUOutput)
    return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                     default_device_.Id());
  return default_device_;
}

std::vector<AllocatorPtr> CUDAExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
      },
      info_.device_id);

  CUDAExecutionProvider::CUDAAllocatorParams params{};
  params.device_id = info_.device_id;
  params.cuda_mem_threshold = info_.gpu_mem_limit;
  params.arena_extend_strategy = info_.arena_extend_strategy;
  params.provider_info = &info_;
  params.external_alloc_info = &info_.external_allocator_info;
  params.arena_cfg = info_.default_memory_arena_cfg;
  params.logger = GetLogger();
  params.budget_account = session_budget_account_;

  return std::vector<AllocatorPtr>{
      CreateCudaAllocator(params),
      CreateAllocator(pinned_memory_info),
  };
}

}  // namespace onnxruntime
