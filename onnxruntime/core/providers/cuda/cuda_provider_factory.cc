// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_options.h"

#include <chrono>
#include <memory>
#include <string>

#include <gsl/gsl>

#include "core/common/status.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

#ifdef ENABLE_NVTX_PROFILE
#include "nvtx_profile.h"
#endif

using namespace onnxruntime;

namespace onnxruntime {

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
namespace cuda {
cuda::INcclService& GetINcclService();
}
#endif

void InitializeRegistry();
void DeleteRegistry();

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(const CUDAExecutionProviderInfo& info)
      : info_{info} {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  CUDAExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  return std::make_unique<CUDAExecutionProvider>(info_);
}

struct ProviderInfo_CUDA_Impl final : ProviderInfo_CUDA {
  OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) override {
    int num_devices;
    auto cuda_err = ::cudaGetDeviceCount(&num_devices);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id since cudaGetDeviceCount failed.");
    }

    if (device_id >= num_devices) {
      std::ostringstream ostr;
      ostr << "Invalid device id. Device id should be less than total number of devices (" << num_devices << ")";
      return CreateStatus(ORT_INVALID_ARGUMENT, ostr.str().c_str());
    }

    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id.");
    }
    return nullptr;
  }

  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAPinnedAllocator>(device_id, name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override {
    return std::make_unique<GPUDataTransfer>();
  }

  void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }
  void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }

  void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) override {
    ORT_ENFORCE(dst_location.device.UsesCpuMemory(), "Copy destination is not CPU memory");

    // Current CUDA device.
    int device;
    CUDA_CALL_THROW(cudaGetDevice(&device));

    if (device != src_location.device.Id()) {
      // Need to switch to the allocating device.
      CUDA_CALL_THROW(cudaSetDevice(src_location.device.Id()));
      // Copy from GPU to CPU.
      CUDA_CALL_THROW(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
      // Switch back to current device.
      CUDA_CALL_THROW(cudaSetDevice(device));
    } else {
      // Copy from GPU to CPU.
      CUDA_CALL_THROW(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    }
  }

  // Used by slice_concatenate_test.cc and onnxruntime_pybind_state.cc

  void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // cudaMemcpy() operates on the default stream
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    CUDA_CALL_THROW(cudaStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  }

  int cudaGetDeviceCount() override {
    int num_devices = 0;
    CUDA_CALL_THROW(::cudaGetDeviceCount(&num_devices));
    return num_devices;
  }

  void CUDAExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options, CUDAExecutionProviderInfo& info) override {
    info = CUDAExecutionProviderInfo::FromProviderOptions(options);
  }

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
  cuda::INcclService& GetINcclService() override {
    return cuda::GetINcclService();
  }
#endif

#ifdef ENABLE_NVTX_PROFILE
  void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) override { p->BeginImpl(); }
  void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) override { p->EndImpl(); }
#endif

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const CUDAExecutionProviderInfo& info) override {
    return std::make_shared<CUDAProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) override {
    CUDAExecutionProvider::CUDAAllocatorParams params{};
    params.device_id = device_id;
    params.cuda_mem_threshold = gpu_mem_limit;
    params.arena_extend_strategy = arena_extend_strategy;
    params.external_alloc_info = &external_allocator_info;
    params.arena_cfg = default_memory_arena_cfg;
    return CUDAExecutionProvider::CreateCudaAllocator(params);
  }

  std::shared_ptr<IAllocator> CreateCudaPinnedAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, const OrtArenaCfg* default_memory_arena_cfg) override {
    CUDAExecutionProvider::CUDAAllocatorParams params{};
    params.device_id = device_id;
    params.cuda_mem_threshold = gpu_mem_limit;
    params.arena_extend_strategy = arena_extend_strategy;
    params.arena_cfg = default_memory_arena_cfg;
    return CUDAExecutionProvider::CreateCudaPinnedAllocator(params);
  }
} g_info;

struct CUDA_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto params = reinterpret_cast<const OrtCUDAProviderOptionsV2*>(void_params);
    ORT_ENFORCE(
        params->external_data_loader_reading_threads <=
            OrtCUDAProviderOptionsV2::kMaxExternalDataLoaderReadingThreadCount,
        "external_data_loader_reading_threads must be between 0 and ",
        OrtCUDAProviderOptionsV2::kMaxExternalDataLoaderReadingThreadCount, ".");

    // Calling a function like ::cudaDeviceSynchronize will cause CUDA to ensure there is binary code for the current GPU architecture
    // Ideally this will be already part of the binary, but if not, CUDA will JIT it during this call. This can take a very long time
    // (minutes even), so we want to detect when this happens and let the user know why so they can report it properly or even fix it.
    // See the linked issue in the warning message for more info
    {
      auto start_time = std::chrono::steady_clock::now();
      // Do a trivial cuda operation that will cause JIT to occur
      {
        void** cuda_memory{};
        ::cudaMalloc(&cuda_memory, 1);
        ::cudaFree(cuda_memory);
      }
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      if (duration > std::chrono::seconds{30}) {
        LOGS_DEFAULT(WARNING) << "CUDA took " << duration.count() << " seconds to start, please see this issue for how to fix it: https://github.com/microsoft/onnxruntime/issues/10746";
      }
    }

    CUDAExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(params->device_id);
    info.gpu_mem_limit = params->gpu_mem_limit;
    info.gpu_device_mem_limit = params->gpu_device_mem_limit;
    info.arena_extend_strategy = params->arena_extend_strategy;
    info.cudnn_conv_algo_search = params->cudnn_conv_algo_search;
    info.do_copy_in_default_stream = params->do_copy_in_default_stream != 0;
    info.has_user_compute_stream = params->has_user_compute_stream != 0;
    info.user_compute_stream = params->user_compute_stream;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;
    info.cudnn_conv_use_max_workspace = params->cudnn_conv_use_max_workspace != 0;
    info.enable_cuda_graph = params->enable_cuda_graph != 0;
    info.prefer_nhwc = params->prefer_nhwc;
    info.fuse_conv_bias = params->fuse_conv_bias;
    info.cudnn_conv1d_pad_to_nc1d = params->cudnn_conv1d_pad_to_nc1d != 0;
    info.tunable_op.enable = params->tunable_op_enable;
    info.tunable_op.tuning_enable = params->tunable_op_tuning_enable;
    info.tunable_op.max_tuning_duration_ms = params->tunable_op_max_tuning_duration_ms;
    info.use_ep_level_unified_stream = params->use_ep_level_unified_stream != 0;
    info.use_tf32 = params->use_tf32 != 0;
    info.sdpa_kernel = params->sdpa_kernel;
    info.external_data_loader_reading_threads = params->external_data_loader_reading_threads;

    return std::make_shared<CUDAProviderFactory>(info);
  }

  /**
   * This function will be called by the C API UpdateCUDAProviderOptions().
   *
   * What this function does is equivalent to resetting the OrtCUDAProviderOptionsV2 instance with
   * default CUDAExecutionProviderInf instance first and then set up the provided provider options.
   * See CUDAExecutionProviderInfo::FromProviderOptions() for more details.
   */
  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::CUDAExecutionProviderInfo::FromProviderOptions(options);
    auto& cuda_options = *reinterpret_cast<OrtCUDAProviderOptionsV2*>(provider_options);

    cuda_options.device_id = internal_options.device_id;
    cuda_options.cudnn_conv_algo_search = internal_options.cudnn_conv_algo_search;
    cuda_options.gpu_mem_limit = internal_options.gpu_mem_limit;
    cuda_options.gpu_device_mem_limit = internal_options.gpu_device_mem_limit;
    cuda_options.arena_extend_strategy = internal_options.arena_extend_strategy;
    cuda_options.do_copy_in_default_stream = internal_options.do_copy_in_default_stream;
    cuda_options.has_user_compute_stream = internal_options.has_user_compute_stream;
    // The 'has_user_compute_stream' of the OrtCUDAProviderOptionsV2 instance can be set by C API UpdateCUDAProviderOptionsWithValue() as well.
    // We only set the 'has_user_compute_stream' of the OrtCUDAProviderOptionsV2 instance if it is provided in options
    if (options.find("has_user_compute_stream") != options.end()) {
      cuda_options.user_compute_stream = internal_options.user_compute_stream;
    }
    cuda_options.default_memory_arena_cfg = internal_options.default_memory_arena_cfg;
    cuda_options.cudnn_conv_use_max_workspace = internal_options.cudnn_conv_use_max_workspace;
    cuda_options.enable_cuda_graph = internal_options.enable_cuda_graph;
    cuda_options.cudnn_conv1d_pad_to_nc1d = internal_options.cudnn_conv1d_pad_to_nc1d;
    cuda_options.prefer_nhwc = internal_options.prefer_nhwc;
    cuda_options.use_ep_level_unified_stream = internal_options.use_ep_level_unified_stream;
    cuda_options.use_tf32 = internal_options.use_tf32;
    cuda_options.sdpa_kernel = internal_options.sdpa_kernel;
    cuda_options.fuse_conv_bias = internal_options.fuse_conv_bias;
    cuda_options.external_data_loader_reading_threads =
        internal_options.external_data_loader_reading_threads;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtCUDAProviderOptionsV2*>(provider_options);
    return onnxruntime::CUDAExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (num_devices != 1) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "CUDA EP only supports one device.");
    }

    OrtCUDAProviderOptionsV2 options;
    UpdateProviderOptions(&options, provider_options);
    auto ep_factory = CreateExecutionProviderFactory(&options);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
  }

} g_provider;

CUDA_Provider* GetProvider() {
  return &g_provider;
}

}  // namespace onnxruntime

//
// Plug-in EP infrastructure
//

#include "core/session/abi_devices.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

struct ErrorHelper {
  static const OrtApi* ort_api;

  static OrtStatus* ToOrtStatus(const Status& status) {
    if (status.IsOK()) {
      return nullptr;  // no error
    }

    return ort_api->CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                 status.ErrorMessage().c_str());
  }
};

const OrtApi* ErrorHelper::ort_api = nullptr;

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF_STATUS_NOTOK(fn)              \
  do {                                          \
    Status _status = (fn);                      \
    if (!_status.IsOK()) {                      \
      return ErrorHelper::ToOrtStatus(_status); \
    }                                           \
  } while (0)

#define CUDA_RETURN_IF_ERROR(expr) RETURN_IF_STATUS_NOTOK(CUDA_CALL(expr))

struct CudaOrtAllocator : OrtAllocator {
  CudaOrtAllocator(const OrtMemoryInfo* mem_info, const OrtApi& api) : memory_info_{mem_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;      // no special behavior for Reserve so use AllocImpl
    GetStats = nullptr;       // GetStatsImpl. The CUDA allocators don't have stats currently so we can skip.
    AllocOnStream = nullptr;  // TODO. Plugin EP arena to provide this.
    Shrink = nullptr;

    const OrtEpApi& ep_api = *api.GetEpApi();
    const OrtMemoryDevice* mem_device = ep_api.MemoryInfo_GetMemoryDevice(mem_info);
    uint32_t device_id = ep_api.MemoryDevice_GetDeviceId(mem_device);
    const char* name = nullptr;
    auto* status = api.MemoryInfoGetName(mem_info, &name);
    static_cast<void>(status);  // GetName never fails

    if (ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
      allocator_ = std::make_unique<CUDAPinnedAllocator>(device_id, name);
    } else {
      allocator_ = std::make_unique<CUDAAllocator>(device_id, name);
    }
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    auto& impl = *static_cast<CudaOrtAllocator*>(this_);
    return impl.allocator_->Alloc(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    auto& impl = *static_cast<CudaOrtAllocator*>(this_);
    impl.allocator_->Free(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CudaOrtAllocator& impl = *static_cast<const CudaOrtAllocator*>(this_);
    return impl.memory_info_;
  }

 private:
  const OrtMemoryInfo* memory_info_;
  std::unique_ptr<IAllocator> allocator_;
};

struct CudaDataTransferImpl : OrtDataTransferImpl {
  CudaDataTransferImpl(const OrtApi& ort_api_in)
      : ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                          const OrtMemoryDevice* src_memory_device,
                          const OrtMemoryDevice* dst_memory_device) noexcept {
    const auto& impl = *static_cast<const CudaDataTransferImpl*>(this_ptr);

    // logic copied from GPUDataTransfer::CanCopy
    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
    auto src_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
    auto dst_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

    if ((src_type == OrtDevice::GPU && src_vendor_id != OrtDevice::VendorIds::NVIDIA) ||
        (dst_type == OrtDevice::GPU && dst_vendor_id != OrtDevice::VendorIds::NVIDIA)) {
      return false;
    }

    // copy must be GPU to GPU or between GPU and CPU
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
  }

  static OrtStatus* CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                    const OrtValue** src_tensors,
                                    OrtValue** dst_tensors,
                                    OrtSyncStream** streams,
                                    size_t num_tensors) noexcept {
    auto& impl = *static_cast<CudaDataTransferImpl*>(this_ptr);
    bool need_stream_sync = false;

    for (size_t idx = 0; idx < num_tensors; ++idx) {
      const OrtValue* src_tensor = src_tensors[idx];
      OrtValue* dst_tensor = dst_tensors[idx];
      OrtSyncStream* stream = streams ? streams[idx] : nullptr;

      const OrtMemoryDevice* src_device = impl.ep_api.Value_GetMemoryDevice(src_tensor);
      const OrtMemoryDevice* dst_device = impl.ep_api.Value_GetMemoryDevice(dst_tensor);

      size_t bytes;
      RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensor, &bytes));

      const void* src_data = nullptr;
      void* dst_data = nullptr;
      RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensor, &src_data));
      RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data));

      OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
      OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);
      OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_device);
      OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_device);

      const bool src_is_gpu_default = src_type == OrtMemoryInfoDeviceType_GPU &&
                                      src_mem_type == OrtDeviceMemoryType_DEFAULT;
      const bool dst_is_gpu_default = dst_type == OrtMemoryInfoDeviceType_GPU &&
                                      dst_mem_type == OrtDeviceMemoryType_DEFAULT;

      cudaStream_t cuda_stream = nullptr;
      if (stream) {
        cuda_stream = static_cast<cudaStream_t>(impl.ort_api.SyncStream_GetHandle(stream));
      }

      if (dst_is_gpu_default) {
        if (src_is_gpu_default) {
          // Copy only if the two addresses are different.
          if (dst_data != src_data) {
            if (cuda_stream) {
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, cuda_stream));

            } else {
              CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));

              // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
              // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
              need_stream_sync = true;
            }
          }
        } else {
          // copy from pinned or non-pinned CPU memory to GPU
          if (cuda_stream) {
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, cuda_stream));
          } else {
            CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));

            if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
              // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not
              // have completed.
              // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
              need_stream_sync = true;
            }
          }
        }
      } else if (src_is_gpu_default) {
        // copying from GPU to CPU memory, this is blocking

        if (cuda_stream) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, cuda_stream));

        } else {
          CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
        }
      } else {
        // copying between CPU accessible memory

        if (dst_data != src_data) {
          if (cuda_stream) {
            if (src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
              // sync the stream first to make sure the data arrived
              CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
            }
          }

          memcpy(dst_data, src_data, bytes);
        }
      }
    }

    if (need_stream_sync) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
    }

    return nullptr;
  }

  static void ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
    // no-op as we have a single shared instance in OrtEpFactory which is returned from CreateDataTransferImpl, and is
    // owned by and freed by the factory.
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct CudaSyncNotificationImpl : OrtSyncNotificationImpl {
  static OrtStatus* Create(cudaStream_t stream, const OrtApi& ort_api,
                           std::unique_ptr<CudaSyncNotificationImpl>& notification) {
    notification.reset(new CudaSyncNotificationImpl(stream, ort_api));  // can't use make_unique with private ctor
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&notification->event_, cudaEventDisableTiming));

    return nullptr;
  }

  static void ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    delete static_cast<CudaSyncNotificationImpl*>(this_ptr);
  }

  static OrtStatus* ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);
    CUDA_RETURN_IF_ERROR(cudaEventRecord(impl.event_, impl.stream_));

    return nullptr;
  }

  static OrtStatus* WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                     _In_ OrtSyncStream* consumer_stream) noexcept {
    auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);

    // setup the consumer stream to wait on our event.
    void* consumer_handle = impl.ort_api.SyncStream_GetHandle(consumer_stream);
    CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_handle), impl.event_));

    return nullptr;
  }

  static OrtStatus* WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);
    CUDA_RETURN_IF_ERROR(cudaEventSynchronize(impl.event_));

    return nullptr;
  }

  ~CudaSyncNotificationImpl() {
    cudaEventDestroy(event_);
  }

 private:
  CudaSyncNotificationImpl(cudaStream_t stream, const OrtApi& ort_api_in)
      : stream_{stream}, ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
    Release = ReleaseImpl;
  }

  cudaStream_t stream_;
  cudaEvent_t event_;

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct CudaSyncStreamImpl : OrtSyncStreamImpl {
  CudaSyncStreamImpl(cudaStream_t&& stream,
                     const OrtDevice& device,
                     AllocatorPtr cpu_allocator,
                     bool release_cpu_buffer_on_cuda_stream,
                     const OrtApi& ort_api_in)
      : stream_{
            stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, /*own*/ true,
            /*external_cudnn_handle*/ nullptr,
            /*external_cublas_handle*/ nullptr,
            // ep_info is used by GetResource which seems to be a somewhat ugly way to make arbitrary info that is
            // unrelated to the stream available to a custom op.
            // avoiding adding GetResource to OrtSyncStreamImpl as we should have a cleaner setup for custom ops,
            // so this argument value isn't used and doesn't matter.
            /*ep_info*/ CUDAExecutionProviderInfo{}},
        ort_api{ort_api_in} {
    ort_version_supported = ORT_API_VERSION;
    GetHandle = GetHandleImpl;
    CreateNotification = CreateNotificationImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
  }

  static void ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    delete static_cast<CudaSyncStreamImpl*>(this_ptr);
  }

  static void* GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
    return impl.stream_.GetHandle();
  }

  static OrtStatus* CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                           _Outptr_ OrtSyncNotificationImpl** notification_impl) noexcept {
    auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
    *notification_impl = nullptr;

    std::unique_ptr<CudaSyncNotificationImpl> notification;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(impl.stream_.GetHandle());

    RETURN_IF_ERROR(CudaSyncNotificationImpl::Create(cuda_stream, impl.ort_api, notification));
    *notification_impl = notification.release();

    return nullptr;
  }

  static OrtStatus* FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
    impl.stream_.Flush();

    return nullptr;
  }

  static OrtStatus* OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
    RETURN_IF_STATUS_NOTOK(impl.stream_.CleanUpOnRunEnd());

    return nullptr;
  }

 private:
  // this is a little onion-ish as CudaStream is a onnxruntime::Stream and this is an OrtSyncStreamImpl that will be
  // used via plugin_ep::Stream, which is also an onnxruntime::Stream. in a 'real' plugin EP implementation
  // CudaStream would go away and the logic it has would be implemented directly here.
  CudaStream stream_;
  const OrtApi& ort_api;
};

// OrtEpApi infrastructure to be able to use the CUDA EP as an OrtEpFactory for auto EP selection.
struct CudaEpFactory : OrtEpFactory {
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

  CudaEpFactory(const OrtApi& ort_api_in, const OrtLogger& default_logger_in) : OrtEpFactory{},
                                                                                ort_api{ort_api_in},
                                                                                ep_api{*ort_api_in.GetEpApi()},
                                                                                default_logger{default_logger_in},
                                                                                data_transfer_impl{ort_api_in} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;

    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;

    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  }

  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto& factory = *static_cast<const CudaEpFactory*>(this_ptr);
    return factory.ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto& factory = *static_cast<const CudaEpFactory*>(this_ptr);
    return factory.vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const CudaEpFactory*>(this_ptr);
    return factory->vendor_id;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto& factory = *static_cast<CudaEpFactory*>(this_ptr);

    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    RETURN_IF_ERROR(factory.CreateMemoryInfoForDevices(num_cuda_devices));

    /* in theory we can match on the LUID in the OrtHardwareDevice metadata, but that requires the CUDA Driver API
    std::vector<uint64_t> device_to_luid;
    device_to_luid.resize(num_cuda_devices);

    for (int i = 0; i < num_cuda_devices; ++i) {
      CUdevice device;
      cuDeviceGet(&device, i);

      char luid[8];
      unsigned int nodeMask;
      if (cuDeviceGetLuid(luid, &nodeMask, device) == CUDA_SUCCESS) {
        device_to_luid[i] = *reinterpret_cast<uint64_t*>(luid);
      }
    }
    */

    // guard against bad device discovery. max devices we expect to add is num_cuda_devices. if we're attempting
    // to add more than that we have duplicates in the `devices` array.
    max_ep_devices = std::min(max_ep_devices, static_cast<size_t>(num_cuda_devices));

    int16_t device_id = 0;
    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (factory.ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
          factory.ort_api.HardwareDevice_VendorId(&device) == 0x10de) {
        /* ideally we'd match on LUID here
           for now we use an incrementing device id. could be a mismatch if you have multiple different CUDA GPUs.
           alternative is to limit to one device only.

        // find the device id. On Windows we have the LUID in the OrtHardwareDevice metadata.
        const OrtKeyValuePairs* metadata = factory.ort_api.HardwareDevice_Metadata(&device);
        const char* luid_str = factory.ort_api.GetKeyValue(metadata, "LUID");

        if (!luid_str && num_devices > 1) {
          // if there's no LUID we can't match device
          return factory.ort_api.CreateStatus(ORT_EP_FAIL, "OrtHardwareDevice does not have LUID");
        }

        char* luid_end = nullptr;
        uint64_t luid = std::strtoull(luid_str, &luid_end, 10);
        for (; device_id < num_cuda_devices; ++device_id) {
          if (device_to_luid[device_id] == luid) {
            break;
          }
        }

        if (device_id == num_cuda_devices) {
          std::string msg("Could not match LUID to a CUDA device. LUID=");
          msg += luid_str;

          return factory.ort_api.CreateStatus(ORT_EP_FAIL, msg.c_str());
        }
        */

        // create the EP options and add the device id
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;
        factory.ort_api.CreateKeyValuePairs(&ep_options);
        factory.ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(device_id).c_str());

        // create the OrtEpDevice
        OrtEpDevice* ep_device = nullptr;
        RETURN_IF_ERROR(factory.ort_api.GetEpApi()->CreateEpDevice(&factory, &device, ep_metadata, ep_options,
                                                                   &ep_device));

        factory.ort_api.ReleaseKeyValuePairs(ep_options);

        const OrtMemoryInfo* gpu_mem_info = factory.gpu_memory_infos[device_id].get();
        const OrtMemoryInfo* host_accessible_mem_info = factory.host_accessible_memory_infos[device_id].get();

        RETURN_IF_ERROR(factory.ep_api.EpDevice_AddAllocatorInfo(ep_device, gpu_mem_info));
        RETURN_IF_ERROR(factory.ep_api.EpDevice_AddAllocatorInfo(ep_device, host_accessible_mem_info));

        ep_devices[num_ep_devices++] = ep_device;

        ++device_id;
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return CreateStatus(ORT_INVALID_ARGUMENT, "CUDA EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept {
    // this function is free to return the same allocator instance for all calls and make ReleaseAllocator a no-op
    // e.g. allocator instance is in unique_ptr in the OrtEpFactory instance.
    // ORT will create a shared allocator in the environment and the user can choose to use it in an inference session.
    // Otherwise ORT will create an allocator when adding the EP to an inference session.
    auto& factory = *static_cast<CudaEpFactory*>(this_ptr);

    auto cuda_allocator = std::make_unique<CudaOrtAllocator>(memory_info, factory.ort_api);
    *allocator = cuda_allocator.release();

    return nullptr;
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
    delete static_cast<CudaOrtAllocator*>(allocator);
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
    *data_transfer = &factory.data_transfer_impl;

    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return true;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* /*stream_options*/,
                                                               OrtSyncStreamImpl** ort_stream) noexcept {
    auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
    auto device_id = factory.ep_api.MemoryDevice_GetDeviceId(memory_device);

    // the OrtEpFactory could have a cache of stream instances if it wants to avoid creating a new one on every
    // call. the CudaStreamSyncImpl::Release could return the instance to the cache.
    cudaStream_t stream = nullptr;
    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
    CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Currently this API is only used for creating a stream that is used outside of a session, as we're using the
    // 'real' CUDA IExecutionProvider implementation for the EP. Due to that we need to connect it up to an internal
    // onnxruntime::Stream that has the correct settings for the session.
    // We do that externally by passing the cudaStream_t in via the "user_compute_stream" provider option.
    //
    // For use within an inference session in a completely plugin EP we'd need the session's CPU allocator to be
    // available, as well as for relevant EP instance specific options such as whether graph capture is enabled
    // to be applied.

    const OrtDevice* ort_device = static_cast<const OrtDevice*>(memory_device);
    // This OrtSyncStream isn't used for running the inference, so we don't need a CPU allocator for
    // CPU scratch buffers to be created by operator kernels.
    AllocatorPtr null_allocator;

    auto impl = std::make_unique<CudaSyncStreamImpl>(std::move(stream), *ort_device, nullptr,
                                                     /*release_cpu_buffer_on_cuda_stream*/ true,
                                                     factory.ort_api);
    *ort_stream = impl.release();

    return nullptr;
  }

  OrtStatus* CreateMemoryInfoForDevices(int num_devices) {
    gpu_memory_infos.reserve(num_devices);
    host_accessible_memory_infos.reserve(num_devices);

    for (int device_id = 0; device_id < num_devices; ++device_id) {
      OrtMemoryInfo* mem_info = nullptr;
      RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("CUDA", OrtMemoryInfoDeviceType_GPU,
                                                  /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                  /* device_id */ device_id,
                                                  OrtDeviceMemoryType_DEFAULT,
                                                  /*alignment*/ 0,
                                                  OrtAllocatorType::OrtDeviceAllocator,
                                                  &mem_info));

      gpu_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));

      // HOST_ACCESSIBLE memory should use the non-CPU device type
      mem_info = nullptr;
      RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("CUDA host accessible", OrtMemoryInfoDeviceType_GPU,
                                                  /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                  /* device_id */ device_id,
                                                  OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                  /*alignment*/ 0,
                                                  OrtAllocatorType::OrtDeviceAllocator,
                                                  &mem_info));

      host_accessible_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));
    }

    return nullptr;
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtLogger& default_logger;
  const std::string ep_name{kCudaExecutionProvider};  // EP name
  const std::string vendor{"Microsoft"};              // EP vendor name
  uint32_t vendor_id{0x1414};                         // Microsoft vendor ID

  // per-device memory info
  std::vector<MemoryInfoUniquePtr> gpu_memory_infos;
  std::vector<MemoryInfoUniquePtr> host_accessible_memory_infos;

  // we use a shared instance for the OrtDataTransferImpl instead of creating a new one on every call to
  // CreateDataTransferImpl.
  CudaDataTransferImpl data_transfer_impl;

  CudaEpFactory(const CudaEpFactory&) = delete;
  CudaEpFactory& operator=(const CudaEpFactory&) = delete;

  CudaEpFactory(CudaEpFactory&&) = delete;
  CudaEpFactory& operator=(CudaEpFactory&&) = delete;
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  ErrorHelper::ort_api = ort_api;  // setup our error helper

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<CudaEpFactory>(*ort_api, *default_logger);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<CudaEpFactory*>(factory);
  return nullptr;
}
}
