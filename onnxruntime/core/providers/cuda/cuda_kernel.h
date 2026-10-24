// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef BUILD_CUDA_EP_AS_PLUGIN

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cudnn_loader.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/providers/cuda/bounded_pinned_pool.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include <mutex>
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime {
namespace cuda {

class OrtStreamAdapter {
 public:
  explicit OrtStreamAdapter(onnxruntime::Stream* stream) : stream_(stream) {}
  explicit OrtStreamAdapter(void* stream) : stream_(static_cast<onnxruntime::Stream*>(stream)) {}

  onnxruntime::Stream* get() const { return stream_; }
  operator onnxruntime::Stream*() const { return stream_; }

 private:
  onnxruntime::Stream* stream_;
};

#ifndef CUDA_STREAM_FROM_CTX
// Helper for kernels that need a cudaStream_t from OpKernelContext in both framework and plugin builds.
#define CUDA_STREAM_FROM_CTX(ctx) static_cast<cudaStream_t>(GetComputeStream(ctx))
#endif

// -----------------------------------------------------------------------
// Base class for CUDA kernels
// -----------------------------------------------------------------------
class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<CUDAExecutionProvider*>(static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);
    // use this to precisely locate the node where CUDA failure comes from
    //  if (cudaSuccess != cudaDeviceSynchronize())
    //    __debugbreak();
    if (s.IsOK()) {
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
      }
    }
    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, onnxruntime::Stream* stream) const {
    if (count_or_bytes == 0) {
      return nullptr;
    }

    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault), count_or_bytes, false,
                                        stream);
  }

  // void* overload for dual-build compatibility with the plugin EP.
  // In the framework build, the void* is always a static_cast<void*>(onnxruntime::Stream*).
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, void* stream) const {
    return GetScratchBuffer<T>(count_or_bytes, static_cast<onnxruntime::Stream*>(stream));
  }

  // Resolve nullptr ambiguity between Stream* and void* overloads.
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, std::nullptr_t) const {
    return GetScratchBuffer<T>(count_or_bytes, static_cast<onnxruntime::Stream*>(nullptr));
  }

  // Different from GetScratchBuffer which use IAllocator::Alloc() to allocate memory,
  // this GetTransientScratchBuffer will call IAllocator::Reserve() to allocate memory.
  // IAllocator::Reserve() optionally implement some allocation logic that by-passes any arena-based
  // logic (or similar for different allocator) that may be housed in the Alloc() implementation.
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault), count_or_bytes, true);
  }

  inline void AddDeferredReleaseCPUPtr(void* p, onnxruntime::Stream* ort_stream) const {
    ORT_ENFORCE(ort_stream->GetDevice().Type() == OrtDevice::GPU);
    auto* cuda_ep_stream = static_cast<CudaStream*>(ort_stream);
    cuda_ep_stream->EnqueDeferredCPUBuffer(p);
  }

  // void* overload for dual-build compatibility with the plugin EP.
  inline void AddDeferredReleaseCPUPtr(void* p, void* stream) const {
    AddDeferredReleaseCPUPtr(p, static_cast<onnxruntime::Stream*>(stream));
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeCPU), count_or_bytes);
  }

  // Hands a host buffer to the provider to hold forever.  For buffers a captured
  // graph copies from: the graph re-reads them on every replay, so they must not be
  // recycled by the allocator.
  template <typename T>
  inline void RetainBufferForGraphCapture(IAllocatorUniquePtr<T> buffer) const {
    auto deleter = buffer.get_deleter();
    provider_->RetainBufferForGraphCapture(
        std::shared_ptr<void>(buffer.release(),
                              [deleter](void* p) { deleter(static_cast<T*>(p)); }));
  }

  inline std::shared_ptr<onnxruntime::cuda::QMoESessionController> GetQMoESessionController() const {
    return provider_ ? provider_->GetQMoESessionController() : nullptr;
  }

  inline IAllocatorUniquePtr<uint8_t> AllocateReusableQMoEPinnedBuffer(
      size_t bytes, size_t limit, std::shared_ptr<bool> completed) const {
    auto controller = GetQMoESessionController();
    ORT_ENFORCE(controller != nullptr, "QMoE pinned pool requires a valid QMoESessionController.");
    auto pool = controller->GetOrCreatePinnedPool(limit);
    ORT_ENFORCE(pool != nullptr && pool->Limit() == limit, "QMoE pinned-pool limits conflict within one provider.");
    auto* pointer = static_cast<uint8_t*>(pool->Acquire(bytes));
    return IAllocatorUniquePtr<uint8_t>(pointer, [pool, completed = std::move(completed)](uint8_t* buffer) {
      pool->Release(buffer, *completed);
    });
  }

  const cudaDeviceProp& GetDeviceProp() const { return provider_->GetDeviceProp(); }
  std::filesystem::path GetSessionModelPath() const { return provider_->GetSessionModelPath(); }
  // Identifies the session (execution provider instance) that owns this kernel.
  const void* GetSessionKey() const { return provider_; }
  int GetCudnnConvAlgo() const { return provider_->GetCudnnConvAlgo(); }
  bool GetCudnnConvUseMaxWorkspace() const { return provider_->GetCudnnConvUseMaxWorkspace(); }
  bool GetCudnnConv1dPadToNc1d() const { return provider_->GetCudnnConv1dPadToNc1d(); }
  bool IsFuseConvBias() const { return provider_->IsFuseConvBias(); }

  // Compatibility helper used by kernels that need the underlying ORT stream object.
  inline onnxruntime::Stream* GetComputeStream(OpKernelContext* ctx) const {
    return ctx ? ctx->GetComputeStream() : nullptr;
  }

  inline OrtStreamAdapter GetOrtStream(OpKernelContext* ctx) const {
    return OrtStreamAdapter(GetComputeStream(ctx));
  }

  inline cudaStream_t Stream(OpKernelContext* ctx) const {
    auto* stream = ctx->GetComputeStream();
    return stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
  }

  inline cudnnHandle_t GetCudnnHandle(OpKernelContext* ctx) const {
    return RequireCudnnHandle(GetCudnnHandle(static_cast<CudaStream*>(ctx->GetComputeStream())));
  }

  inline cudnnHandle_t TryGetCudnnHandle(OpKernelContext* ctx) const {
    return GetCudnnHandle(static_cast<CudaStream*>(ctx->GetComputeStream()));
  }

  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::CudaStream* stream) {
    return stream ? stream->cudnn_handle_ : nullptr;
  }

  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::Stream* stream) {
    auto* cuda_stream = dynamic_cast<CudaStream*>(stream);
    ORT_ENFORCE(cuda_stream != nullptr, "Stream is not a CudaStream.");
    return GetCudnnHandle(cuda_stream);
  }

  inline cudnnHandle_t GetCudnnHandleOrDefault(onnxruntime::Stream* stream) const {
    return stream ? RequireCudnnHandle(GetCudnnHandle(stream)) : DefaultCudnnHandle();
  }

  inline cublasHandle_t GetCublasHandle(OpKernelContext* ctx) const {
    return GetCublasHandle(static_cast<CudaStream*>(ctx->GetComputeStream()));
  }

  static inline cublasHandle_t GetCublasHandle(onnxruntime::CudaStream* stream) {
    return stream ? stream->cublas_handle_ : nullptr;
  }

  static inline cublasHandle_t GetCublasHandle(onnxruntime::Stream* stream) {
    auto* cuda_stream = dynamic_cast<CudaStream*>(stream);
    ORT_ENFORCE(cuda_stream != nullptr, "Stream is not a CudaStream.");
    return GetCublasHandle(cuda_stream);
  }

  inline cublasHandle_t GetCublasHandleOrDefault(onnxruntime::Stream* stream) const {
    return stream ? GetCublasHandle(stream) : DefaultCublasHandle();
  }

  inline cublasLtHandle_t GetCublasLtHandle(OpKernelContext* /*ctx*/) const {
    return provider_->PerThreadCublasLtHandle();
  }

  bool UseTF32() const {
    return provider_->UseTF32();
  }

  const AttentionKernelOptions* GetAttentionKernelOptions() const {
    return provider_->GetAttentionKernelOptions();
  }

  tunable::CudaTuningContext* GetTuningContext() const {
    return static_cast<tunable::CudaTuningContext*>(provider_->GetTuningContext());
  }

  // To support cudaMemcpyAsync, the cpu memory should be allocated in pinned memory
  // and it can only be released after the copy has finished
  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* op_kernel) : gpu_copy_(nullptr), count_(0), op_kernel_(op_kernel) {}

    CudaAsyncBuffer(const CudaKernel* op_kernel, size_t count) : CudaAsyncBuffer(op_kernel) {
      AllocCpuPtr(count);
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, const T& value, size_t count)
        : CudaAsyncBuffer(op_kernel, count) {
      T* p = CpuPtr();
      for (size_t i = 0; i != count; ++i) {
        *p++ = value;
      }
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, gsl::span<T const> vec) : CudaAsyncBuffer(op_kernel, vec.size()) {
      memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
    }

    void AllocCpuPtr(size_t count) {
      cpu_pinned_copy_ = op_kernel_->AllocateBufferOnCPUPinned<T>(count);
      if (cpu_pinned_copy_ == nullptr)
        throw std::runtime_error("alloc failed");
      count_ = count;
    }

    Status CopyToGpu(onnxruntime::Stream* stream) {
      if (cpu_pinned_copy_) {
        gpu_copy_ = op_kernel_->GetScratchBuffer<T>(count_, stream);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice,
                                             cuda_stream));
        // While the stream is capturing, the copy above becomes a node of the graph
        // and reads this host buffer again on every replay.  Releasing it here would
        // let a later run of a different shape take the block and overwrite it, and
        // the replay would then copy that run's bytes instead -- for the pointer
        // arrays this class mostly carries, addresses that belong to nothing.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cuda_stream != nullptr) {
          CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(cuda_stream, &capture_status));
        }
        if (capture_status != cudaStreamCaptureStatusNone) {
          op_kernel_->RetainBufferForGraphCapture(std::move(cpu_pinned_copy_));
        } else {
          op_kernel_->AddDeferredReleaseCPUPtr(cpu_pinned_copy_.release(), stream);
        }
      }
      return Status::OK();
    }

    // void* overload for dual-build compatibility with the plugin EP.
    Status CopyToGpu(void* stream) {
      return CopyToGpu(static_cast<onnxruntime::Stream*>(stream));
    }

    T* CpuPtr() const {
      return cpu_pinned_copy_.get();
    }

    gsl::span<T> CpuSpan() const {
      return gsl::span<T>(CpuPtr(), count_);
    }

    T* GpuPtr() const {
      return gpu_copy_.get();
    }

    size_t count() const {
      return count_;
    }

   protected:
    IAllocatorUniquePtr<T> gpu_copy_;
    IAllocatorUniquePtr<T> cpu_pinned_copy_;
    size_t count_;
    const CudaKernel* op_kernel_;
  };

  inline cublasHandle_t DefaultCublasHandle() const {
    return provider_->PerThreadDefaultCublasHandle();
  }

  inline cublasLtHandle_t CublasLtHandle() const {
    return provider_->PerThreadCublasLtHandle();
  }

  inline cudnnHandle_t DefaultCudnnHandle() const {
    return RequireCudnnHandle(provider_->PerThreadDefaultCudnnHandle());
  }

  static inline cudnnHandle_t RequireCudnnHandle(cudnnHandle_t handle) {
    if (handle == nullptr) {
#ifndef USE_CUDA_MINIMAL
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                                         "cuDNN is unavailable or disabled for CUDA Execution Provider: ",
                                         cuda::CudnnLibrary::Get().Error()));
#else
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                                         "cuDNN is unavailable for CUDA Execution Provider in a CUDA minimal build."));
#endif
    }
    return handle;
  }

  inline cudaStream_t DefaultCudaStream() const {
    // this will return the CUDA EP level stream which can differ from the actual compute tasks stream
    // the compute task stream is supplied within OpKernelContext during inference
    return provider_->ComputeStream();
  }

  inline Status CopyTensor(const Tensor& src, Tensor& dst, onnxruntime::Stream& stream) const {
    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(src.Location().device, dst.Location().device);
    return gpu_data_transfer->CopyTensorAsync(src, dst, stream);
  }

 protected:
  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  CUDAExecutionProvider* provider_;
};

}  // namespace cuda
}  // namespace onnxruntime

#else
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"
#endif

namespace onnxruntime {
namespace cuda {

inline bool InputExists(const OpKernelInfo& info, size_t input_index) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  const auto node = info.node();
  return node.InputExists(input_index);
#else
  const auto& input_defs = info.node().InputDefs();
  return input_index < input_defs.size() && input_defs[input_index]->Exists();
#endif
}

// Returns the tensor element type of the input at input_index. Out-of-range inputs and omitted
// optional inputs both return TensorProto_DataType_UNDEFINED.
inline int32_t GetInputElementType(const OpKernelInfo& info, size_t input_index) {
  if (!InputExists(info, input_index)) {
    return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  return static_cast<int32_t>(
      info.GetKernelInfo().GetInputTypeInfo(input_index).GetTensorTypeAndShapeInfo().GetElementType());
#else
  const auto& input_defs = info.node().InputDefs();
  return input_defs[input_index]->TypeAsProto()->tensor_type().elem_type();
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
