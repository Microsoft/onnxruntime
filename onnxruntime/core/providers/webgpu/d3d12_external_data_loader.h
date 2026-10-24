// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

#include "core/framework/allocator.h"
#include "core/framework/external_data_loader.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

namespace onnxruntime {
namespace windows {
namespace d3d12 {
class D3D12FileBufferLoader;
}  // namespace d3d12
}  // namespace windows

namespace webgpu {

class WebGpuContext;
struct CommandRecordingState;

common::Status CheckD3D12AcceleratedExternalWeightsSupport(const WebGpuContext& context);

common::Status ResolveWeightLoadAccelerationMode(
    WeightLoadAccelerationMode mode,
    const common::Status& support_status,
    bool& enabled);

class D3D12ImportedBufferRegistry;
struct D3D12AcceleratedLoadBatch;

AllocatorPtr CreateD3D12AcceleratedWebGpuAllocator(
    WebGpuContext& context,
    int context_id,
    std::function<CommandRecordingState&()> recording_getter,
    std::shared_ptr<D3D12ImportedBufferRegistry>& out_buffer_registry);

class D3D12AcceleratedExternalDataLoader final : public IExternalDataLoader {
 public:
  D3D12AcceleratedExternalDataLoader(
      WebGpuContext& context,
      int context_id,
      std::shared_ptr<D3D12ImportedBufferRegistry> buffer_registry,
      WeightLoadAccelerationMode mode,
      bool enable_uma_weight_loading);
  ~D3D12AcceleratedExternalDataLoader() override;

  // Actual path selected by the native loader after a batch has been loaded.
  bool UsesUMA() const noexcept;
  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;
  bool SupportsDataType(int32_t tensor_data_type) const override;
  bool CreatesTensorForDevice(const OrtDevice& target_device) const override;
  common::Status BeginLoad() const override;
  common::Status RegisterLoadCandidate(const Env& env,
                                       const std::filesystem::path& data_file_path,
                                       std::string_view tensor_name,
                                       FileOffsetType data_offset,
                                       SafeInt<size_t> data_length) const override;
  common::Status CommitLoadCandidates(const std::function<bool()>& is_canceled) const override;
  void EndLoad() const noexcept override;
  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
                            std::string_view tensor_name,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            const std::shared_ptr<IAllocator>& allocator,
                            Tensor& tensor) const override;

 private:
  void ResolveSupport() const;
  common::Status EnsureFileLoader() const;

  WebGpuContext& context_;
  int context_id_;
  std::shared_ptr<D3D12ImportedBufferRegistry> buffer_registry_;
  WeightLoadAccelerationMode acceleration_mode_;
  const bool enable_uma_weight_loading_;
  mutable std::once_flag device_support_resolution_once_;
  mutable common::Status device_support_resolution_status_;
  // Effective decision from mode and device support; cleared on fallback.
  mutable bool resolved_acceleration_enabled_ = false;
  mutable std::atomic<bool> load_end_requested_{false};
  mutable std::unique_ptr<windows::d3d12::D3D12FileBufferLoader> file_to_buffer_loader_;
  mutable std::unique_ptr<D3D12AcceleratedLoadBatch> accelerated_load_batch_;
};

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
