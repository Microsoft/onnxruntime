// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// cuda_kernel_adapter.h — Compatibility shim for migrating CUDA kernels to the
// plugin EP architecture.
//
// This header provides:
//   - CudaKernel base class (scratch buffers, CUDA handles, etc.)
//   - Error-return macros (CUDA_RETURN_IF_ERROR, etc.)
//   - Type mapping helpers (ToCudaType)
//   - Math/compute shims (HalfGemmOptions, CublasMathModeSetter)
//   - Self-registering BuildKernelCreateInfo<> macros via PluginKernelCollector
//   - CUDAExecutionProvider shim class
//   - CPU provider shims for the plugin build

#pragma once

#include <limits>
#include <unordered_map>

#include "core/common/status.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/framework/float4.h"
#include "core/framework/allocator.h"
#include "core/framework/stream_handles.h"
#include "core/framework/tensor_shape.h"
#include "core/util/math.h"
#include <gsl/gsl>

#include <cublas_v2.h>
#include <cudnn.h>
#include "core/providers/cuda/bounded_pinned_pool.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/cudnn_loader.h"
#include "core/providers/cuda/cufft_loader.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

// ===================================================================
// Macros will be defined later to override core definitions.
// ===================================================================

#include "core/providers/cuda/plugin/cuda_stream_plugin.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

// Forward-declare CudaStream so adapter overloads accepting CudaStream* compile
// without pulling in cuda_stream_handle.h (which depends on CUDAExecutionProvider).
namespace onnxruntime {
struct CudaStream;

namespace cuda_plugin {
namespace detail {
inline thread_local std::unordered_map<void*, onnxruntime::Stream*> stream_to_framework_stream;
inline thread_local void* current_cuda_stream = nullptr;
inline thread_local onnxruntime::Stream* current_framework_stream = nullptr;

inline void RegisterFrameworkStreamForCudaStream(void* cuda_stream, OrtSyncStream* framework_stream) {
  current_cuda_stream = cuda_stream;
  current_framework_stream = reinterpret_cast<onnxruntime::Stream*>(framework_stream);

  if (current_framework_stream == nullptr) {
    return;
  }

  // Map only from the raw cudaStream_t handle to the current framework stream. The framework
  // stream is already handled directly by GetFrameworkStreamForStreamArg, so we deliberately do
  // not insert a framework_stream -> framework_stream entry: it would be unused and would grow the
  // thread-local map without bound while retaining framework stream pointers past the
  // Session::Run() teardown lifetime documented for KernelContext_GetSyncStream.
  if (cuda_stream != nullptr) {
    stream_to_framework_stream[cuda_stream] = current_framework_stream;
  }
}

inline onnxruntime::Stream* GetFrameworkStreamForStreamArg(void* stream) {
  // A null stream argument means "the compute stream of the current Compute call". This is the
  // form used by GetTransientScratchBuffer and legacy GetScratchBuffer(..., nullptr). Map it to
  // the framework stream registered for this call so scratch chunks are still stream-tagged even
  // when the kernel runs on a non-default CUDA stream (where current_cuda_stream is non-null and a
  // nullptr arg would otherwise miss the map lookup and fall back to a null stream tag).
  //
  // current_framework_stream is scoped to a single CudaKernel::Compute invocation by
  // ComputeStreamScope (see below). Outside any Compute call it is nullptr, so allocations made
  // from kernel constructors (which also call GetScratchBuffer(..., nullptr)) fall back to the
  // non-stream-tagged path instead of inheriting a stale framework stream pointer whose lifetime
  // ended with a previous Session::Run().
  if (stream == nullptr || stream == current_cuda_stream || stream == current_framework_stream) {
    return current_framework_stream;
  }

  auto it = stream_to_framework_stream.find(stream);
  return it == stream_to_framework_stream.end() ? nullptr : it->second;
}

// RAII guard that scopes the thread-local "current Compute call" framework stream to the lifetime
// of a single CudaKernel::Compute invocation on a worker thread.
//
// On entry it clears current_cuda_stream/current_framework_stream so that scratch allocated before
// the kernel registers its stream (via Stream(ctx)/GetComputeStream(ctx)/GetOrtStream(ctx)), or via
// a nullptr stream argument, does not inherit a stale framework stream left over from a previous
// Compute call on this worker thread. On exit it restores the previous values, which keeps nested
// Compute calls (a kernel that invokes another kernel's Compute) correct and leaves the per-thread
// "current" stream cleared once the outermost Compute returns. The borrowed framework stream is
// only valid until its owning Session::Run() completes teardown, so it must not outlive the call.
struct ComputeStreamScope {
  ComputeStreamScope()
      : saved_cuda_stream_(current_cuda_stream),
        saved_framework_stream_(current_framework_stream) {
    current_cuda_stream = nullptr;
    current_framework_stream = nullptr;
  }
  ~ComputeStreamScope() {
    current_cuda_stream = saved_cuda_stream_;
    current_framework_stream = saved_framework_stream_;
  }

 private:
  void* saved_cuda_stream_;
  onnxruntime::Stream* saved_framework_stream_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ComputeStreamScope);
};
}  // namespace detail
}  // namespace cuda_plugin

// Lightweight Stream shim for plugin build: wraps a raw cudaStream_t as a
// framework-compatible Stream* that can be passed to _impl.cu functions which
// call stream->GetHandle().  Stack-allocated; does NOT own the stream.
// Only available in .cc translation units (not .cu) since Stream is incomplete in NVCC context.
#ifndef __CUDACC__
struct PluginStreamShim : public onnxruntime::Stream {
  explicit PluginStreamShim(void* cuda_stream_handle)
      : onnxruntime::Stream(cuda_stream_handle,
                            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                      OrtDevice::VendorIds::NVIDIA, 0)) {}
};

class OrtStreamAdapter {
 public:
  explicit OrtStreamAdapter(void* cuda_stream_handle)
      : plugin_stream_shim_(cuda_stream_handle), stream_(&plugin_stream_shim_) {}

  OrtStreamAdapter(void* cuda_stream_handle, OrtSyncStream* framework_stream)
      : plugin_stream_shim_(cuda_stream_handle),
        stream_(framework_stream == nullptr ? static_cast<onnxruntime::Stream*>(&plugin_stream_shim_)
                                            : reinterpret_cast<onnxruntime::Stream*>(framework_stream)) {}

  onnxruntime::Stream* get() const { return stream_; }
  operator onnxruntime::Stream*() const { return stream_; }

 private:
  PluginStreamShim plugin_stream_shim_;
  onnxruntime::Stream* stream_;
};
#else
class OrtStreamAdapter {
 public:
  explicit OrtStreamAdapter(void* cuda_stream_handle)
      : stream_(static_cast<onnxruntime::Stream*>(cuda_stream_handle)) {}

  OrtStreamAdapter(void* cuda_stream_handle, OrtSyncStream* framework_stream)
      : stream_(framework_stream == nullptr ? static_cast<onnxruntime::Stream*>(cuda_stream_handle)
                                            : reinterpret_cast<onnxruntime::Stream*>(framework_stream)) {}

  onnxruntime::Stream* get() const { return stream_; }
  operator onnxruntime::Stream*() const { return stream_; }

 private:
  onnxruntime::Stream* stream_;
};
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 1: Include path selection
// ===================================================================

#include "core/graph/constants.h"
#include "ep/adapters.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

// Forward declaration of GetEnvironmentVar for plugin builds on Windows.
// Defined in provider_api_shims.cc; mirrors the provider_api.h declaration
// which is a no-op when BUILD_CUDA_EP_AS_PLUGIN is set.
std::string GetEnvironmentVar(const std::string& var_name);
}  // namespace onnxruntime

namespace onnxruntime {
namespace cuda {

#ifndef CUDA_STREAM_FROM_CTX
// Helper for kernels that need a cudaStream_t from OpKernelContext in plugin build.
#define CUDA_STREAM_FROM_CTX(ctx) static_cast<cudaStream_t>(GetComputeStream(ctx))
#endif

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

// Tensor creation helper to replace deprecated Tensor::Create
inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace cuda {

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::contrib::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda
}  // namespace contrib
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 2: Error-return macros (redefined for all plugin paths)
// ===================================================================

// Redefine error macros for ported code to use our adapter-specific Status translation
#undef CUDA_RETURN_IF_ERROR
#define CUDA_RETURN_IF_ERROR(expr)                                                                                                                             \
  {                                                                                                                                                            \
    cudaError_t _err = (expr);                                                                                                                                 \
    if (_err != cudaSuccess) {                                                                                                                                 \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    }                                                                                                                                                          \
  }

#undef CUBLAS_RETURN_IF_ERROR
#define CUBLAS_RETURN_IF_ERROR(expr)                                                                                                                                              \
  {                                                                                                                                                                               \
    cublasStatus_t _status = (expr);                                                                                                                                              \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                                                                                                                       \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuBLAS error: ") + std::to_string(static_cast<int>(_status))); \
    }                                                                                                                                                                             \
  }

#undef CUDNN_RETURN_IF_ERROR
#define CUDNN_RETURN_IF_ERROR(expr)                                                                                                                                 \
  {                                                                                                                                                                 \
    cudnnStatus_t _status = (expr);                                                                                                                                 \
    if (_status != CUDNN_STATUS_SUCCESS) {                                                                                                                          \
      if (!onnxruntime::cuda::CudnnLibrary::Get().Available()) {                                                                                                    \
        return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,                                                  \
                                           std::string("cuDNN is unavailable for CUDA Plugin Execution Provider: ") +                                               \
                                               onnxruntime::cuda::CudnnLibrary::Get().Error());                                                                     \
      }                                                                                                                                                             \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuDNN error: ") + cudnnGetErrorString(_status)); \
    }                                                                                                                                                               \
  }

#undef CUFFT_RETURN_IF_ERROR
#define CUFFT_RETURN_IF_ERROR(expr)                                                                                                                                 \
  {                                                                                                                                                                 \
    cufftResult _status = (expr);                                                                                                                                   \
    if (_status != CUFFT_SUCCESS) {                                                                                                                                 \
      if (!onnxruntime::cuda::CufftLibrary::Get().Available()) {                                                                                                    \
        return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,                                                  \
                                           std::string("cuFFT is unavailable for CUDA Plugin Execution Provider: ") +                                               \
                                               onnxruntime::cuda::CufftLibrary::Get().Error());                                                                     \
      }                                                                                                                                                             \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuFFT error: ") + std::to_string((int)_status)); \
    }                                                                                                                                                               \
  }

// ===================================================================
// Section 3: Self-registering kernel collector using BuildKernelCreateInfo<>
//
// Each ONNX_OPERATOR_*_KERNEL_EX macro expansion both:
//   1. Creates a BuildKernelCreateInfo<CLASS>() template specialization
//      (identical to the framework's macro output, but using adapter types)
//   2. Auto-registers the BuildKernelCreateInfoFn pointer into a global
//      PluginKernelCollector singleton
//
// At registration time, the factory iterates the collector and calls
// adapter::KernelRegistry::Register(build_fn()) for each compiled kernel.
// Only ops whose .cc files are actually compiled get registered.
// ===================================================================

#include <vector>

namespace onnxruntime {
namespace cuda {

/// Singleton collector for BuildKernelCreateInfoFn pointers.
/// Each compiled kernel .cc file's macro expansion auto-registers here.
///
/// Thread-safety: Instance() uses a function-local static (C++11 §6.7/4:
/// constructed exactly once, even under concurrent first-access). Add()
/// is guarded by a mutex for formal correctness across translation units,
/// though in practice all calls occur during static initialization.
class PluginKernelCollector {
 public:
  static PluginKernelCollector& Instance() {
    static PluginKernelCollector instance;
    return instance;
  }

  void Add(BuildKernelCreateInfoFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(fn);
  }
  std::vector<BuildKernelCreateInfoFn> Entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
  }

 private:
  std::vector<BuildKernelCreateInfoFn> entries_;
  mutable std::mutex mutex_;
};

}  // namespace cuda
}  // namespace onnxruntime

// --- Macro overrides: produce BuildKernelCreateInfo<> AND auto-register ---
//
// These macros mirror the framework's ONNX_OPERATOR_*_KERNEL_EX definitions
// from core/framework/op_kernel.h, but additionally register each
// BuildKernelCreateInfoFn into PluginKernelCollector at static init time.
#define ORT_ADAPTER_CONCAT_IMPL(x, y) x##y
#define ORT_ADAPTER_CONCAT(x, y) ORT_ADAPTER_CONCAT_IMPL(x, y)

// The provider parameter are not used in below macros since we are hardcoding the provider to cuda plugin.
#define CUDA_PLUGIN_EP ::onnxruntime::kCudaExecutionProviderPluginAlias

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                           \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                \
  template <>                                                                                        \
  KernelCreateInfo                                                                                   \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {            \
    return KernelCreateInfo(                                                                         \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                              \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {   \
              out = std::make_unique<__VA_ARGS__>(info);                                             \
              return Status::OK();                                                                   \
            }));                                                                                     \
  }                                                                                                  \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_, __COUNTER__) =                 \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                   \
           &BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>),    \
       true);

#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, ...)                 \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name);                      \
  template <>                                                                                                     \
  KernelCreateInfo                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)>() {  \
    return KernelCreateInfo(                                                                                      \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                                           \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                \
              out = std::make_unique<__VA_ARGS__>(info);                                                          \
              return Status::OK();                                                                                \
            }));                                                                                                  \
  }                                                                                                               \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_, __COUNTER__) =                              \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(                                      \
               provider, domain, startver, endver, name)>),                                                       \
       true);

#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)                        \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);                             \
  template <>                                                                                                 \
  KernelCreateInfo                                                                                            \
  BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>() {         \
    return KernelCreateInfo(                                                                                  \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),          \
        static_cast<KernelCreatePtrFn>(                                                                       \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {            \
              out = std::make_unique<__VA_ARGS__>(info);                                                      \
              return Status::OK();                                                                            \
            }));                                                                                              \
  }                                                                                                           \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type##_, __COUNTER__) =                 \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                            \
           &BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>), \
       true);

#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...)     \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);          \
  template <>                                                                                                     \
  KernelCreateInfo                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                                          \
      provider, domain, startver, endver, type, name)>() {                                                        \
    return KernelCreateInfo(                                                                                      \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                                           \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                \
              out = std::make_unique<__VA_ARGS__>(info);                                                          \
              return Status::OK();                                                                                \
            }));                                                                                                  \
  }                                                                                                               \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type##_, __COUNTER__) =                     \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                                \
               provider, domain, startver, endver, type, name)>),                                                 \
       true);

#undef ONNX_OPERATOR_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(name, domain, ver, type1, type2, provider, builder, ...)                        \
  class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name);                             \
  template <>                                                                                                             \
  KernelCreateInfo                                                                                                        \
  BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name)>() {         \
    return KernelCreateInfo(                                                                                              \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),                      \
        static_cast<KernelCreatePtrFn>(                                                                                   \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                        \
              out = std::make_unique<__VA_ARGS__>(info);                                                                  \
              return Status::OK();                                                                                        \
            }));                                                                                                          \
  }                                                                                                                       \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_, __COUNTER__) =                  \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                        \
           &BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name)>), \
       true);

#undef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(name, domain, startver, endver, type1, type2,                    \
                                                    provider, builder, ...)                                          \
  class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type1, type2, name); \
  template <>                                                                                                        \
  KernelCreateInfo                                                                                                   \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(                                         \
      provider, domain, startver, endver, type1, type2, name)>() {                                                   \
    return KernelCreateInfo(                                                                                         \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(),    \
        static_cast<KernelCreatePtrFn>(                                                                              \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                   \
              out = std::make_unique<__VA_ARGS__>(info);                                                             \
              return Status::OK();                                                                                   \
            }));                                                                                                     \
  }                                                                                                                  \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_, __COUNTER__) =             \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                   \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(                               \
               provider, domain, startver, endver, type1, type2, name)>),                                            \
       true);

#undef ONNX_OPERATOR_THREE_TYPED_KERNEL_EX
#define ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(name, domain, ver, type1, type2, type3, provider, builder, ...)                \
  class ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, type3, name);                     \
  template <>                                                                                                              \
  KernelCreateInfo                                                                                                         \
  BuildKernelCreateInfo<ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, type3, name)>() { \
    return KernelCreateInfo(                                                                                               \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),                       \
        static_cast<KernelCreatePtrFn>(                                                                                    \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                         \
              out = std::make_unique<__VA_ARGS__>(info);                                                                   \
              return Status::OK();                                                                                         \
            }));                                                                                                           \
  }                                                                                                                        \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_##type3##_, __COUNTER__) =         \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                         \
           &BuildKernelCreateInfo<ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(                                             \
               provider, domain, ver, type1, type2, type3, name)>),                                                        \
       true);

// ===================================================================
// Section 4: Logging shim (adapter path only)
// LOGS_DEFAULT is re-routed through ep::adapter::LoggingManager, which
// holds the ORT default logger set up in CudaEpFactory::CudaEpFactory.
// All severity levels (including ERROR/WARNING) are forwarded to the
// ORT logger; no log output is suppressed.
// ===================================================================

// Explicit function instantiation — called once per unique class in each .cc file
#define ONNX_OPERATOR_TYPED_KERNEL_COMPUTE_INSTANTIATION(cls) template Status cls::ComputeInternal(OpKernelContext* context) const;

// The plugin utilizes ep::adapter::LoggingManager for LOGS_DEFAULT,
// which is initialized in CudaEpFactory::CudaEpFactory.

#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace cuda {

// ===================================================================
// Section 5: Runtime configuration for migrated kernels
// Fields are written once during CudaEp construction and owned by the
// shim CUDAExecutionProvider. Plugin kernels cache a shared_ptr to this
// config object during construction so Compute() does not need to rely on
// raw provider-pointer casts.
// ===================================================================

namespace detail {
struct CudaKernelAdapterRuntimeConfig {
  bool use_tf32 = true;
  int cudnn_conv_algo = 0;
  bool cudnn_conv_use_max_workspace = true;
  bool cudnn_conv1d_pad_to_nc1d = false;
  bool enable_cudnn = true;
  bool fuse_conv_bias = false;
  int sdpa_kernel = 0;
  int device_id = 0;
  bool do_copy_in_default_stream = true;
  bool enable_cuda_graph = false;
  cudaDeviceProp device_prop{};
  onnxruntime::AttentionKernelOptions attention_kernel_options;
  std::mutex captured_host_buffers_mutex;
  std::vector<std::shared_ptr<void>> captured_host_buffers;
  std::mutex qmoe_pinned_pool_mutex;
  std::shared_ptr<onnxruntime::cuda::BoundedPinnedPool> qmoe_pinned_pool;
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> qmoe_controller =
      std::make_shared<onnxruntime::cuda::QMoESessionController>();

  void RetainBufferForGraphCapture(std::shared_ptr<void> buffer) {
    std::lock_guard<std::mutex> lock(captured_host_buffers_mutex);
    captured_host_buffers.push_back(std::move(buffer));
  }

  // Path of the model the EP partitioned (recorded in GetCapability); empty if loaded from bytes.
  void SetModelPath(std::filesystem::path path) {
    if (!path.empty() && path.is_relative()) {
      std::error_code ec;
      std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
      if (!ec && !abs_path.empty()) {
        path = std::move(abs_path);
      }
    }
    std::lock_guard<std::mutex> lock(model_path_mutex);
    model_path = std::move(path);
  }
  std::filesystem::path GetModelPath() const {
    std::lock_guard<std::mutex> lock(model_path_mutex);
    return model_path;
  }

 private:
  mutable std::mutex model_path_mutex;
  std::filesystem::path model_path;
};
template <typename T>
struct SizeOf {
  static constexpr size_t value = sizeof(T);
};
template <>
struct SizeOf<void> {
  static constexpr size_t value = 0;
};

[[nodiscard]] inline bool TryBytesForCount(size_t count_or_bytes, size_t element_size, size_t& bytes) {
  if (element_size == 0) {
    // `element_size == 0` is the sentinel for the `T = void` path.
    // In that mode callers already pass a raw byte count to helpers like
    // GetScratchBuffer<void>(workspace_bytes, ...), so no multiplication is needed.
    bytes = count_or_bytes;
    return true;
  }

  if (count_or_bytes > (std::numeric_limits<size_t>::max() / element_size)) {
    return false;
  }

  bytes = count_or_bytes * element_size;
  return true;
}

struct DefaultCudaHandles {
  cublasHandle_t cublas = nullptr;
  cudnnHandle_t cudnn = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;

  ~DefaultCudaHandles() {
    if (cublas != nullptr) {
      cublasDestroy(cublas);
    }
    if (cudnn != nullptr) {
      cudnnDestroy(cudnn);
    }
    if (cublas_lt != nullptr) {
      cublasLtDestroy(cublas_lt);
    }
  }
};

inline DefaultCudaHandles& GetDefaultCudaHandlesForDevice(int device_id) {
  // Fallback handles are only used for code paths that need cuBLAS/cuDNN
  // without an active CudaSyncStream. Keep them thread-local so they are not
  // shared across callers that may use the libraries concurrently.
  //
  // Only cuBLAS/cuBLASLt are created here. The cuDNN fallback handle is created
  // lazily by GetDefaultCudnnHandleForDevice() so that cuBLAS-only paths (and
  // sessions with enable_cudnn=0) never trigger a cuDNN load.
  thread_local std::unordered_map<int, DefaultCudaHandles> handles_by_device;
  auto [it, inserted] = handles_by_device.try_emplace(device_id);
  if (inserted) {
    int prev_device = -1;
    const cudaError_t get_device_result = cudaGetDevice(&prev_device);
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
    if (cublasCreate(&it->second.cublas) != CUBLAS_STATUS_SUCCESS) {
      if (get_device_result == cudaSuccess) {
        cudaSetDevice(prev_device);
      }
      handles_by_device.erase(it);
      ORT_THROW("Failed to create default cuBLAS handle for CUDA plugin device ", device_id);
    }
    if (cublasLtCreate(&it->second.cublas_lt) != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(it->second.cublas);
      it->second.cublas = nullptr;
      if (get_device_result == cudaSuccess) {
        cudaSetDevice(prev_device);
      }
      handles_by_device.erase(it);
      ORT_THROW("Failed to create default cuBLASLt handle for CUDA plugin device ", device_id);
    }
    if (get_device_result == cudaSuccess) {
      PL_CUDA_CALL_THROW(cudaSetDevice(prev_device));
    }
  }

  return it->second;
}

// Lazily creates the thread-local fallback cuDNN handle for the device. Callers
// must check enable_cudnn and CudnnLibrary::Available() before invoking this so
// that cuBLAS-only paths never trigger a cuDNN load.
inline cudnnHandle_t GetDefaultCudnnHandleForDevice(int device_id) {
  DefaultCudaHandles& handles = GetDefaultCudaHandlesForDevice(device_id);
  if (handles.cudnn != nullptr) {
    return handles.cudnn;
  }

  int prev_device = -1;
  const cudaError_t get_device_result = cudaGetDevice(&prev_device);
  PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
  cudnnHandle_t cudnn = nullptr;
  const cudnnStatus_t status = cudnnCreate(&cudnn);
  if (get_device_result == cudaSuccess) {
    cudaSetDevice(prev_device);
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    ORT_THROW("Failed to create default cuDNN handle for CUDA plugin device ", device_id);
  }

  handles.cudnn = cudnn;
  return cudnn;
}

inline const cudaDeviceProp& GetDevicePropForDevice(int device_id) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<cudaDeviceProp>> props;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = props.find(device_id);
  if (it == props.end()) {
    auto prop = std::make_unique<cudaDeviceProp>();
    const cudaError_t result = cudaGetDeviceProperties(prop.get(), device_id);
    if (result != cudaSuccess) {
      ORT_THROW("Failed to query CUDA device properties for device ", device_id, ": ", cudaGetErrorString(result));
    }
    it = props.emplace(device_id, std::move(prop)).first;
  }
  return *it->second;
}
}  // namespace detail
}  // namespace cuda

// ===================================================================
// Section 6: CUDAExecutionProvider shim
// Provides the minimal API surface that migrated kernels expect
// (GetCudnnConvAlgo, UseTF32, GetDeviceProp, etc.) without the full
// CUDAExecutionProvider class from onnxruntime/core/providers/cuda/.
//
// In the plugin build this shim is wrapped by adapter::Ep. Plugin kernels
// should prefer the CudaKernel base-class accessors for runtime settings
// instead of re-casting info.GetExecutionProvider() inside Compute().
// ===================================================================

// Shim for CUDAExecutionProvider required by conv.cc, einsum, and others
class CUDAExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const std::string& name, const OrtEp* ort_ep = nullptr)
      : onnxruntime::IExecutionProvider{name}, ort_ep_{ort_ep} {}

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override {
    return std::make_unique<onnxruntime::GPUDataTransfer>();
  }

  const OrtEp* GetOrtEp() const override {
    return ort_ep_;
  }

  std::shared_ptr<cuda::detail::CudaKernelAdapterRuntimeConfig> GetRuntimeConfig() const {
    return config_;
  }

  int GetCudnnConvAlgo() const {
    return config_->cudnn_conv_algo;
  }
  bool GetCudnnConvUseMaxWorkspace() const {
    return config_->cudnn_conv_use_max_workspace;
  }
  bool GetCudnnConv1dPadToNc1d() const {
    return config_->cudnn_conv1d_pad_to_nc1d;
  }
  bool UseTF32() const {
    return config_->use_tf32;
  }
  bool IsFuseConvBias() const {
    return config_->fuse_conv_bias;
  }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    config_->attention_kernel_options.InitializeOnce(config_->sdpa_kernel, true, true);
    return &config_->attention_kernel_options;
  }
  const cudaDeviceProp& GetDeviceProp() const {
    return config_->device_prop;
  }
  bool DoCopyOnDefaultStream() const {
    return config_->do_copy_in_default_stream;
  }

 private:
  const OrtEp* ort_ep_ = nullptr;
  std::shared_ptr<cuda::detail::CudaKernelAdapterRuntimeConfig> config_ =
      std::make_shared<cuda::detail::CudaKernelAdapterRuntimeConfig>();
};

namespace cuda {
namespace detail {

inline std::shared_ptr<CudaKernelAdapterRuntimeConfig> GetCudaKernelAdapterRuntimeConfigForProvider(const void* provider) {
  return static_cast<const CUDAExecutionProvider*>(provider)->GetRuntimeConfig();
}

}  // namespace detail

// Populate the per-provider adapter config from a pre-filled initializer struct.
// Callers (e.g. CudaEp constructor) construct a detail::CudaKernelAdapterRuntimeConfig,
// fill every field they care about, then call this function. Adding a new config
// field only requires updating the struct and the call site — no signature change.
inline void SetCudaKernelAdapterRuntimeConfigForProvider(
    const void* provider, const detail::CudaKernelAdapterRuntimeConfig& init_config) {
  auto config = detail::GetCudaKernelAdapterRuntimeConfigForProvider(provider);
  // AttentionKernelOptions contains std::once_flag (not copyable), so assign
  // the plain-data fields individually rather than relying on operator=.
  config->use_tf32 = init_config.use_tf32;
  config->cudnn_conv_algo = init_config.cudnn_conv_algo;
  config->cudnn_conv_use_max_workspace = init_config.cudnn_conv_use_max_workspace;
  config->cudnn_conv1d_pad_to_nc1d = init_config.cudnn_conv1d_pad_to_nc1d;
  config->enable_cudnn = init_config.enable_cudnn;
  config->fuse_conv_bias = init_config.fuse_conv_bias;
  config->sdpa_kernel = init_config.sdpa_kernel;
  config->device_id = init_config.device_id;
  config->do_copy_in_default_stream = init_config.do_copy_in_default_stream;
  config->enable_cuda_graph = init_config.enable_cuda_graph;
  if (init_config.qmoe_controller && init_config.qmoe_controller->DeviceId() == init_config.device_id) {
    config->qmoe_controller = init_config.qmoe_controller;
  } else {
    config->qmoe_controller = std::make_shared<onnxruntime::cuda::QMoESessionController>(init_config.device_id);
    if (init_config.qmoe_controller && init_config.qmoe_controller->BudgetAccount()) {
      config->qmoe_controller->SetBudgetAccount(init_config.qmoe_controller->BudgetAccount());
    }
  }
  PL_CUDA_CALL_THROW(cudaGetDeviceProperties(&config->device_prop, config->device_id));
}

// Global aliases and shims
using Status = onnxruntime::common::Status;
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

// Type mapping for CUDA
template <typename T>
struct ToCudaType {
  typedef T MappedType;
  static MappedType FromFloat(float f) { return static_cast<MappedType>(f); }
};

template <>
struct ToCudaType<MLFloat16> {
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = onnxruntime::math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

#ifdef __CUDACC__
template <>
struct ToCudaType<BFloat16> {
  typedef nv_bfloat16 MappedType;
  static MappedType FromFloat(float f) {
    return nv_bfloat16(f);
  }
};

// Forward declare templates from common.cuh to allow specialization
// Match signatures from common.cuh exactly (no default parameters)
template <typename T, bool detect_positive, bool detect_negative>
struct _IsInf;
template <typename T>
struct _IsNan;

namespace bf16_isinf_nan {
template <typename T>
struct IsInfTyped;
template <>
struct IsInfTyped<nv_bfloat16> {
  static __device__ __inline__ bool IsInf(nv_bfloat16 a) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) == 0x0000;
  }
  static __device__ __inline__ bool IsInfPos(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0x7F80;
  }
  static __device__ __inline__ bool IsInfNeg(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0xFF80;
  }
};
}  // namespace bf16_isinf_nan

// Specialize for nv_bfloat16 to avoid ambiguity with isnan/isinf overloads
template <>
struct _IsNan<nv_bfloat16> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) != 0x0000;
  }
};

template <bool detect_positive, bool detect_negative>
struct _IsInf<nv_bfloat16, detect_positive, detect_negative> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    if constexpr (detect_positive && detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInf(a);
    } else if constexpr (detect_positive) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfPos(a);
    } else if constexpr (detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfNeg(a);
    } else {
      return false;
    }
  }
};

// cuda_utils.h only specializes NumericLimits for onnxruntime::BFloat16. Without this the plugin's
// nv_bfloat16 mapping falls back to std::numeric_limits, whose primary template returns 0, so
// kernels that pad with Lowest() (TopK, reductions) rank the padding above every negative input.
template <>
struct NumericLimits<nv_bfloat16> {
  __inline__ __host__ __device__ static nv_bfloat16 Lowest() {
    return __nv_bfloat16_raw{0xFF7FU};  // -3.38953139e38
  }

  __inline__ __host__ __device__ static nv_bfloat16 Max() {
    return __nv_bfloat16_raw{0x7F7FU};  // 3.38953139e38
  }
};
#endif

// ===================================================================
// Section 6b: CPU provider shims for the plugin build
// Inline implementations of CPU utility functions that CUDA kernels
// reference (e.g., OneHot validation, GatherElements shape checks).
// These are normally provided by onnxruntime_providers but the plugin
// does not link against it.
//
// We temporarily close namespace cuda so these shims live directly in
// namespace onnxruntime, where unqualified lookup from onnxruntime::cuda
// will find them, and where onnxruntime::GatherElements resolves correctly.
// ===================================================================

}  // namespace cuda

// Shim for ValidateInputs from core/providers/cpu/tensor/onehot.h
inline Status ValidateInputs(const Tensor* depth, const Tensor* values) {
  if (!depth->Shape().IsScalar()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for depth; it's not a scalar.");
  }
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }
  return Status::OK();
}

// Shim for PrepareOutputShape from core/providers/cpu/tensor/onehot.h
inline Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                                 int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                                 TensorShapeVector& output_shape) {
  const auto& indices_shape = indices->Shape();
  const auto indices_dims = indices_shape.GetDims();
  const auto indices_num_dims = indices_shape.NumDimensions();

  // ONNX spec requires indices to have rank >= 1.
  if (indices_num_dims == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "OneHot: indices tensor must have rank >= 1.");
  }

  output_shape = indices_shape.AsShapeVector();
  const auto output_rank = static_cast<int64_t>(indices_num_dims) + 1;
  auto true_axis = HandleNegativeAxis(axis, output_rank);
  output_shape.insert(output_shape.begin() + true_axis, depth_val);

  // Validate that the total output tensor element count does not overflow int64.
  {
    int64_t total_elements = 1;
    for (auto dim : output_shape) {
      if (dim > 0 && total_elements > std::numeric_limits<int64_t>::max() / dim) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "OneHot: output tensor size would overflow for the given indices shape "
                               "and depth value (",
                               depth_val, ").");
      }
      total_elements *= dim;
    }
  }

  // Use SafeInt for prefix_dim_size to guard against overflow.
  // SafeInt is defensive here -- the total-element overflow check above already covers this case,
  // so a SafeIntException should never fire in practice.
  SafeInt<int64_t> safe_prefix = 1;
  for (int64_t i = 0; i < true_axis; ++i) {
    safe_prefix *= indices_dims[narrow<size_t>(i)];
  }
  prefix_dim_size = safe_prefix;

  // Guard against division by zero when indices have a zero-sized dimension before the axis.
  suffix_dim_size = (prefix_dim_size > 0) ? (indices_shape.Size() / prefix_dim_size) : 0;
  return Status::OK();
}

// Shim for GatherElements::ValidateInputShapes from
// core/providers/cpu/tensor/gather_elements.h
class GatherElements {
 public:
  static Status ValidateInputShapes(const TensorShape& input_data_shape,
                                    const TensorShape& indices_shape,
                                    int64_t axis) {
    int64_t input_data_rank = static_cast<int64_t>(input_data_shape.NumDimensions());
    int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());
    if (input_data_rank < 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Cannot operate on scalar input");
    if (input_data_rank != indices_rank)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");
    for (int64_t i = 0; i < indices_rank; ++i) {
      if (i != axis) {
        if (indices_shape[narrow<size_t>(i)] < 0 ||
            indices_shape[narrow<size_t>(i)] > input_data_shape[narrow<size_t>(i)])
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
                                 "Invalid value in indices shape is: ",
                                 indices_shape[narrow<size_t>(i)]);
      }
    }
    return Status::OK();
  }
};

namespace cuda {  // re-open onnxruntime::cuda

// ===================================================================
// Section 7: CudaKernel base class
// Base class for all migrated CUDA kernels. Provides scratch-buffer
// management, CUDA handle access (cuBLAS, cuDNN), device property
// queries, and the CudaAsyncBuffer helper for host→device transfers.
// ===================================================================

// Additional adapter logic for CudaKernel

class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info) : OpKernel(info) {
    const auto* provider = info.GetExecutionProvider();
    runtime_config_ = detail::GetCudaKernelAdapterRuntimeConfigForProvider(provider);
    use_tf32_ = runtime_config_->use_tf32;
    device_id_ = runtime_config_->device_id;
    device_prop_ = runtime_config_->device_prop;
  }
  virtual ~CudaKernel() = default;
  Status Compute(OpKernelContext* ctx) const {
    // Scope the thread-local "current Compute call" framework stream to this invocation so that
    // scratch tagged via a nullptr stream argument never inherits a stale framework stream from a
    // previous Compute call (or leaks one to a later kernel constructor) on this worker thread.
    cuda_plugin::detail::ComputeStreamScope compute_stream_scope;

    // Ensure the correct CUDA device is active for this kernel.
    // Worker threads default to device 0; sessions on device > 0 need an
    // explicit cudaSetDevice. Skip during CUDA graph capture because
    // cudaSetDevice is not allowed on a capturing stream.
    if (!IsThreadCapturingCudaGraph()) {
      int current_device = -1;
      PL_CUDA_CALL_THROW(cudaGetDevice(&current_device));
      if (current_device != device_id_) {
        PL_CUDA_CALL_THROW(cudaSetDevice(device_id_));
      }
    }
    Status s = ComputeInternal(ctx);
    if (s.IsOK()) {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    return s;
  }
  virtual Status ComputeInternal(OpKernelContext* ctx) const = 0;

  inline cudaStream_t DefaultCudaStream() const { return Stream(static_cast<OpKernelContext*>(nullptr)); }
  inline cublasHandle_t DefaultCublasHandle() const { return detail::GetDefaultCudaHandlesForDevice(device_id_).cublas; }
  inline cudnnHandle_t DefaultCudnnHandle() const {
    if (!runtime_config_->enable_cudnn || !onnxruntime::cuda::CudnnLibrary::Get().Available()) {
      return nullptr;
    }
    return detail::GetDefaultCudnnHandleForDevice(device_id_);
  }
  inline cublasLtHandle_t DefaultCublasLtHandle() const { return detail::GetDefaultCudaHandlesForDevice(device_id_).cublas_lt; }

  inline Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst, onnxruntime::Stream& stream) const {
    if (src.Shape().Size() == 0) return Status::OK();
    if (cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream.GetHandle()) != cudaSuccess) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
    }
    return Status::OK();
  }

  cudaStream_t Stream(OpKernelContext* ctx) const {
    if (!ctx) return nullptr;
    // Register the framework sync stream for this Compute call so that scratch allocated via
    // GetTransientScratchBuffer()/GetScratchBuffer(..., nullptr) is still stream-tagged for kernels
    // that call Stream(ctx) before GetComputeStream()/GetOrtStream() (e.g. conv algo search).
    void* cuda_stream = ctx->GetGPUComputeStream();
    cuda_plugin::detail::RegisterFrameworkStreamForCudaStream(cuda_stream, ctx->GetSyncStream());
    return static_cast<cudaStream_t>(cuda_stream);
  }

  // Returns an opaque stream pointer for passing to GetScratchBuffer/AddDeferredReleaseCPUPtr/CopyToGpu.
  // Returns void* for dual-build compatibility: framework wraps Stream*, plugin wraps cudaStream_t.
  inline void* GetComputeStream(OpKernelContext* ctx) const {
    void* cuda_stream = ctx->GetGPUComputeStream();
    cuda_plugin::detail::RegisterFrameworkStreamForCudaStream(cuda_stream, ctx->GetSyncStream());
    return cuda_stream;
  }

  inline onnxruntime::OrtStreamAdapter GetOrtStream(OpKernelContext* ctx) const {
    void* cuda_stream = ctx->GetGPUComputeStream();
    OrtSyncStream* framework_stream = ctx->GetSyncStream();
    cuda_plugin::detail::RegisterFrameworkStreamForCudaStream(cuda_stream, framework_stream);
    return onnxruntime::OrtStreamAdapter(cuda_stream, framework_stream);
  }

  static cudnnHandle_t GetCudnnHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCudnnHandle() : nullptr;
  }
  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCudnnHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::Stream* stream) {
    return stream ? GetCudnnHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cudnnHandle_t GetCudnnHandle(OpKernelContext* ctx) const {
    auto stream = Stream(ctx);
    auto handle = GetCudnnHandle(stream);
    if (handle != nullptr) {
      return handle;
    }

    handle = DefaultCudnnHandle();
    if (handle == nullptr) {
      ORT_THROW_IF_ERROR(onnxruntime::common::Status(
          onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
          std::string("cuDNN is unavailable or disabled for CUDA Plugin Execution Provider: ") +
              onnxruntime::cuda::CudnnLibrary::Get().Error()));
    }
    // Bind the shared handle to the current compute stream. cudaStream_t 0/nullptr is the default
    // stream, which is still a valid stream to bind, so do this unconditionally to avoid leaving
    // the handle bound to a stale stream from a previous call.
    CUDNN_CALL_THROW(cudnnSetStream(handle, stream));
    return handle;
  }

  cudnnHandle_t TryGetCudnnHandle(OpKernelContext* ctx) const {
    auto stream = Stream(ctx);
    auto handle = GetCudnnHandle(stream);
    if (handle != nullptr) {
      return handle;
    }

    handle = DefaultCudnnHandle();
    if (handle != nullptr) {
      // Bind the shared handle to the current compute stream. cudaStream_t 0/nullptr is the default
      // stream, which is still a valid stream to bind, so do this unconditionally to avoid leaving
      // the handle bound to a stale stream from a previous call.
      // Keep this accessor non-throwing: if the stream cannot be bound, treat it as "no cuDNN handle"
      // so callers can fall back to a cuDNN-free path instead of failing.
      if (!CUDNN_CALL(cudnnSetStream(handle, stream)).IsOK()) {
        return nullptr;
      }
    }
    return handle;
  }

  static cublasHandle_t GetCublasHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCublasHandle() : nullptr;
  }
  static inline cublasHandle_t GetCublasHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCublasHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cublasHandle_t GetCublasHandle(onnxruntime::Stream* stream) {
    return stream ? GetCublasHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cublasHandle_t GetCublasHandle(OpKernelContext* ctx) const {
    auto stream = Stream(ctx);
    auto handle = GetCublasHandle(stream);
    if (handle != nullptr) {
      return handle;
    }

    handle = DefaultCublasHandle();
    if (stream != nullptr) {
      CUBLAS_CALL_THROW(cublasSetStream(handle, stream));
    }
    return handle;
  }

  static cublasLtHandle_t GetCublasLtHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCublasLtHandle() : nullptr;
  }
  static inline cublasLtHandle_t GetCublasLtHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCublasLtHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cublasLtHandle_t GetCublasLtHandle(onnxruntime::Stream* stream) {
    return stream ? GetCublasLtHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cublasLtHandle_t GetCublasLtHandle(OpKernelContext* ctx) const {
    auto handle = GetCublasLtHandle(Stream(ctx));
    return handle != nullptr ? handle : DefaultCublasLtHandle();
  }

  const cudaDeviceProp& GetDeviceProp() const {
    // Some migrated kernels size their launches from device properties. If the
    // per-provider cache was not populated for this kernel instance, fall back
    // to a direct lookup instead of returning an all-zero struct.
    if (device_prop_.maxThreadsPerMultiProcessor == 0 || device_prop_.multiProcessorCount == 0) {
      return detail::GetDevicePropForDevice(device_id_);
    }

    return device_prop_;
  }
  inline int GetDeviceId() const { return device_id_; }
  std::filesystem::path GetSessionModelPath() const { return runtime_config_->GetModelPath(); }
  // Identifies the session (execution provider instance) that owns this kernel.
  const void* GetSessionKey() const { return runtime_config_.get(); }
  int GetCudnnConvAlgo() const { return runtime_config_->cudnn_conv_algo; }
  bool GetCudnnConvUseMaxWorkspace() const { return runtime_config_->cudnn_conv_use_max_workspace; }
  bool GetCudnnConv1dPadToNc1d() const { return runtime_config_->cudnn_conv1d_pad_to_nc1d; }
  bool UseTF32() const { return use_tf32_; }
  bool IsFuseConvBias() const { return runtime_config_->fuse_conv_bias; }
  bool IsArchAvailable(int arch) const { return GetDeviceProp().major >= arch; }
  // Delegate to the base OpKernel::Info() which holds a safe copy of OpKernelInfo.
  // Do NOT store a reference to the constructor parameter — it becomes dangling.
  const OpKernelInfo& Info() const { return OpKernel::Info(); }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    runtime_config_->attention_kernel_options.InitializeOnce(runtime_config_->sdpa_kernel, true, true);
    return &runtime_config_->attention_kernel_options;
  }

  // Stub for GetTuningContext — tunable ops are not supported in the plugin.
  struct PluginTuningContextStub {
    bool IsTunableOpEnabled() const { return false; }
  };
  PluginTuningContextStub* GetTuningContext() const {
    static PluginTuningContextStub stub;
    return &stub;
  }

  template <typename T>
  using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t cnt, void* stream) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});

    // Route kernel scratch/workspace allocations through the EP allocator
    // (a BFC arena by default) instead of raw cudaMallocAsync/cudaMalloc.
    //
    // The arena pre-reserves device memory and reuses freed chunks across runs.
    // Once the model has executed `min_num_runs_before_cuda_graph_capture`
    // warmup runs, the arena has grown to its steady-state working set, so the
    // capture run serves every scratch allocation from an already-reserved chunk
    // without issuing a fresh cudaMalloc. This keeps the device free-memory
    // footprint stable across the capture window, which is required for correct
    // CUDA graph capture/replay.
    //
    // The previous behavior (cudaMallocAsync/cudaMalloc allocated-and-freed per
    // call) allocated new device memory on every run, including the capture run,
    // so no amount of warmup could stabilize it and the
    // "GPU memory was allocated during CUDA graph capture" detector would trip.
    // This now matches the built-in (non-plugin) CUDA EP, which also obtains
    // scratch from Info().GetAllocator() (see core/providers/cuda/cuda_kernel.h).
    // The overflow check that the previous hand-rolled path performed is still
    // enforced inside MakeUniquePtr via ValidatedCalcMemSizeForArray (it throws
    // on cnt * sizeof(T) overflow).
    //
    // The `stream` argument is the raw cudaStream_t used by migrated CUDA kernels, or a Stream*
    // from OrtStreamAdapter in code paths that need stream->GetHandle(). Stream-aware arena
    // allocation needs the stable framework Stream* wrapper instead, because the arena stores it
    // in each chunk and later queries sync ids through the EP stream API. Stream(ctx),
    // GetComputeStream(ctx) and GetOrtStream(ctx) record the mapping from both argument forms to
    // the framework stream for the current Compute call.
    // If the negotiated ORT API version does not include KernelContext_GetSyncStream, the lookup
    // returns null and allocation falls back to the non-stream-tagged path.
    auto* framework_stream = cuda_plugin::detail::GetFrameworkStreamForStreamArg(stream);
    return ::onnxruntime::IAllocator::MakeUniquePtr<T>(
        Info().GetAllocator(OrtMemType::OrtMemTypeDefault), cnt, /*use_reserve*/ false,
        framework_stream);
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t cnt) const {
    return GetScratchBuffer<T>(cnt, nullptr);
  }
  inline void AddDeferredReleaseCPUPtr(void* p, void* s) const {
    if (!p) return;
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(s));
    if (sync) {
      sync->EnqueueDeferredCPUBuffer(p);
      return;
    }

    if (s != nullptr) {
      cudaError_t sync_result = cudaStreamSynchronize(static_cast<cudaStream_t>(s));
      if (sync_result != cudaSuccess) {
        // If the raw stream handle is already invalid during teardown, prefer a
        // bounded leak over freeing pinned memory that could still be in use by
        // an in-flight async copy.
        LOGS_DEFAULT(WARNING) << "AddDeferredReleaseCPUPtr: cudaStreamSynchronize failed ("
                              << cudaGetErrorString(sync_result)
                              << "); leaking pinned buffer to avoid use-after-free";
        return;
      }
    }

    cudaFreeHost(p);
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t cnt) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = 0;
    if (!detail::TryBytesForCount(cnt, detail::SizeOf<T>::value, sz)) {
      ORT_THROW("CUDA pinned CPU buffer allocation size overflow for ", cnt, " elements");
    }
    void* p = nullptr;
    if (cudaHostAlloc(&p, sz, cudaHostAllocDefault) != cudaSuccess) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [](T* ptr) { if (ptr) cudaFreeHost(ptr); });
  }

  template <typename T>
  inline void RetainBufferForGraphCapture(IAllocatorUniquePtr<T> buffer) const {
    auto deleter = buffer.get_deleter();
    runtime_config_->RetainBufferForGraphCapture(
        std::shared_ptr<void>(buffer.release(),
                              [deleter](void* p) { deleter(static_cast<T*>(p)); }));
  }

  inline IAllocatorUniquePtr<uint8_t> AllocateReusableQMoEPinnedBuffer(
      size_t bytes, size_t limit, std::shared_ptr<bool> completed) const {
    std::shared_ptr<onnxruntime::cuda::BoundedPinnedPool> pool;
    if (runtime_config_->qmoe_controller) {
      pool = runtime_config_->qmoe_controller->GetOrCreatePinnedPool(limit);
      ORT_ENFORCE(pool != nullptr && pool->Limit() == limit, "QMoE pinned-pool limits conflict within one provider.");
    } else {
      std::lock_guard<std::mutex> lock(runtime_config_->qmoe_pinned_pool_mutex);
      if (!runtime_config_->qmoe_pinned_pool) {
        runtime_config_->qmoe_pinned_pool = std::make_shared<onnxruntime::cuda::BoundedPinnedPool>(
            limit, [](size_t size) -> void* {
              void* pointer = nullptr;
              return cudaHostAlloc(&pointer, size, cudaHostAllocDefault) == cudaSuccess ? pointer : nullptr; },
            [](void* pointer) { cudaFreeHost(pointer); });
      }
      pool = runtime_config_->qmoe_pinned_pool;
      ORT_ENFORCE(pool->Limit() == limit, "QMoE pinned-pool limits conflict within one provider.");
    }
    auto* pointer = static_cast<uint8_t*>(pool->Acquire(bytes));
    return IAllocatorUniquePtr<uint8_t>(pointer, [pool, completed = std::move(completed)](uint8_t* buffer) {
      pool->Release(buffer, *completed);
    });
  }

  inline std::shared_ptr<onnxruntime::cuda::QMoESessionController> GetQMoESessionController() const {
    return runtime_config_->qmoe_controller;
  }

  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* ok) : gpu_(nullptr, [](T*) {}), count_(0), op_kernel_(ok) {}
    CudaAsyncBuffer(const CudaKernel* ok, size_t n) : CudaAsyncBuffer(ok) { AllocCpuPtr(n); }
    CudaAsyncBuffer(const CudaKernel* ok, const T& v, size_t n) : CudaAsyncBuffer(ok, n) {
      T* p = CpuPtr();
      for (size_t i = 0; i != n; ++i) *p++ = v;
    }
    CudaAsyncBuffer(const CudaKernel* ok, gsl::span<T const> vec) : CudaAsyncBuffer(ok, vec.size()) {
      size_t bytes = 0;
      if (!detail::TryBytesForCount(vec.size(), sizeof(T), bytes)) {
        ORT_THROW("CUDA async buffer host copy size overflow for ", vec.size(), " elements");
      }
      memcpy(CpuPtr(), vec.data(), bytes);
    }
    void AllocCpuPtr(size_t n) {
      cpu_ = op_kernel_->AllocateBufferOnCPUPinned<T>(n);
      if (!cpu_) throw std::runtime_error("alloc fail");
      count_ = n;
    }
    Status CopyToGpu(void* s) {
      if (cpu_) {
        gpu_ = op_kernel_->GetScratchBuffer<T>(count_, s);
        size_t bytes = 0;
        if (!detail::TryBytesForCount(count_, sizeof(T), bytes)) {
          ORT_THROW("CUDA async buffer copy size overflow for ", count_, " elements");
        }
        if (cudaMemcpyAsync(gpu_.get(), cpu_.get(), bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(s)) != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (s != nullptr) {
          CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(static_cast<cudaStream_t>(s), &capture_status));
        }
        if (capture_status != cudaStreamCaptureStatusNone) {
          op_kernel_->RetainBufferForGraphCapture(std::move(cpu_));
        } else {
          op_kernel_->AddDeferredReleaseCPUPtr(cpu_.release(), s);
        }
      }
      return Status::OK();
    }
    T* CpuPtr() const { return cpu_.get(); }
    gsl::span<T> CpuSpan() const { return gsl::span<T>(CpuPtr(), count_); }
    T* GpuPtr() const { return gpu_.get(); }
    size_t count() const { return count_; }

   protected:
    IAllocatorUniquePtr<T> gpu_;
    std::unique_ptr<T, std::function<void(T*)>> cpu_{nullptr, [](T*) {}};
    size_t count_;
    const CudaKernel* op_kernel_;
  };

 private:
  std::shared_ptr<detail::CudaKernelAdapterRuntimeConfig> runtime_config_;
  cudaDeviceProp device_prop_{};
  bool use_tf32_ = true;
  int device_id_ = 0;
};

// ===================================================================
// Section 8: Compute helper shims (HalfGemmOptions, CublasMathModeSetter)
// ===================================================================

// Shims for HalfGemmOptions and CublasMathModeSetter required by fpgeneric.h
class HalfGemmOptions {
 public:
  static const HalfGemmOptions* GetInstance() {
    static HalfGemmOptions instance;
    return &instance;
  }
  cublasMath_t GetMathMode() const { return CUBLAS_DEFAULT_MATH; }
  bool IsCompute16F() const { return false; }
#if defined(CUBLAS_COMPUTE_32F)
  cublasComputeType_t GetComputeType() const { return CUBLAS_COMPUTE_32F; }
#else
  cudaDataType_t GetComputeType() const { return CUDA_R_32F; }
#endif
};

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle, cublasMath_t mode) : handle_(handle) {
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);
    if (enable_) {
      cublasGetMathMode(handle, &mode_);
      enable_ = (mode_ != mode);
      if (enable_) {
        cublasSetMathMode(handle, mode);
      }
    }
  }

  ~CublasMathModeSetter() {
    if (enable_) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  cublasHandle_t handle_;
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
  bool enable_;
};

}  // namespace cuda

// Global aliases for convenience
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

}  // namespace onnxruntime
