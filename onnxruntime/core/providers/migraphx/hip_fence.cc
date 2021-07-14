// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "migraphx_call.h"
#include "hip_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

HIPFence::HIPFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  HIP_CALL_THROW(hipEventCreate(&read_event_));
  HIP_CALL_THROW(hipEventCreate(&write_event_));
}

HIPFence::~HIPFence() {
  HIP_CALL_THROW(hipEventDestroy(read_event_));
  HIP_CALL_THROW(hipEventDestroy(write_event_));
}

void HIPFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  (void)provider_type;
  (void)async_queue_id;
  // sync on CPU for all other providers, this is blocking
  HIP_CALL_THROW(hipEventSynchronize(write_event_));
}

void HIPFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  (void)provider_type;
  (void)queue_id;
  
  // sync on CPU for all other providers, this is blocking
  HIP_CALL_THROW(hipEventSynchronize(read_event_));
  HIP_CALL_THROW(hipEventSynchronize(write_event_));
}

bool HIPFence::CanRelease() {
  return hipEventQuery(read_event_) == hipSuccess &&
         hipEventQuery(write_event_) == hipSuccess;
}

void HIPFence::AfterUsedAsInput(int queue_id) {
  // update read fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  HIP_CALL_THROW(hipEventRecord(read_event_, stream));
}

void HIPFence::AfterUsedAsOutput(int queue_id) {
  // update write fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  HIP_CALL_THROW(hipEventRecord(write_event_, stream));
}

}  // namespace onnxruntime
