// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/collective/nccl_kernels.h"

namespace onnxruntime {
namespace cuda {

NcclAllReduce::NcclAllReduce(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllReduce::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm(group_type_);

  const void* input_data = context->Input<Tensor>(0)->DataRaw();
  void* output_data = context->Output(0, context->Input<Tensor>(0)->Shape())->MutableDataRaw();
  MLDataType onnx_type = context->Input<Tensor>(0)->DataType();

  // Although we assumed the memory address is contiguous for the input, ORT pads activation tensors to 64 bytes aligned
  // and initializers to 256 bytes aligned. There are tiny padding gaps in the contiguous buffer space.
  // We have to AllReduce on the entire buffer, including the padding space.
  const Tensor* last_tensor = context->Input<Tensor>(context->InputCount() - 1);
  const char* end_address = reinterpret_cast<const char*>(last_tensor->DataRaw()) + last_tensor->SizeInBytes();
  size_t num_bytes = end_address - reinterpret_cast<const char*>(input_data);
  size_t input_count = num_bytes / onnx_type->Size();
  ORT_ENFORCE(num_bytes % onnx_type->Size() == 0);

  for (int i = 0; i < context->InputCount(); i++) {
    context->Output(i, context->Input<Tensor>(i)->Shape());
  }

  ncclDataType_t dtype = GetNcclDataType(onnx_type);
#ifdef ORT_USE_NCCL
  NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
#endif
  return Status::OK();
}

NcclAllGather::NcclAllGather(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclAllGather::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm(group_type_);
  const int rank = nccl_->Rank(group_type_);
  const int size = nccl_->Size(group_type_);

  auto onnx_type = context->Input<Tensor>(0)->DataType();
  const size_t element_size = onnx_type->Size();
  ncclDataType_t dtype = GetNcclDataType(onnx_type);

  // Count total number of elements to AllGather.
  const Tensor* first_tensor = context->Input<Tensor>(0);
  const Tensor* last_tensor = context->Input<Tensor>(context->InputCount() - 1);
  const char* start_address = reinterpret_cast<const char*>(first_tensor->DataRaw());
  const char* end_address = reinterpret_cast<const char*>(last_tensor->DataRaw()) + last_tensor->SizeInBytes();
  size_t buffer_size = end_address - start_address;

  // AllGather requires every rank to receive the same amount of data, and
  // slows down significantly if the data is not aligned.  Nvidia recommends 32-byte alignment,
  // so pad to multiple of 32 and world size.
  // Note: the alignment here needs to be kept in-sync with the alignment in zero_optimizer_graph_builder.cc
  const int64_t alignment = size * 32;
  ORT_ENFORCE(buffer_size % alignment == 0, "NcclAllGather's contiguous buffer is not padded to local_size * 32");

  // Calculate the range of inputs this rank will send.
  const int64_t rank_bytes = buffer_size / size;
  const int64_t rank_count = rank_bytes / element_size;
  const int64_t rank_start = rank * rank_bytes;

  // AllGather.
  Tensor* output_tensor = context->Output(0, first_tensor->Shape());
  const void* fusion_data_rank_offset = start_address + rank_start;
#ifdef ORT_USE_NCCL
  NCCL_RETURN_IF_ERROR(ncclAllGather(fusion_data_rank_offset, output_tensor->MutableDataRaw(), rank_count, dtype, comm, stream));
#endif

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    Tensor* output_tensor = context->Output(i, input_tensor->Shape());

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    output_tensor->SetByteOffset(input_tensor->ByteOffset());

    // Copy AllGather results to outputs if needed
    const void* input_data = input_tensor->DataRaw();
    void* output_data = output_tensor->MutableDataRaw();
    if (input_data != output_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice));
    }
  }

  return Status::OK();
}

NcclReduceScatter::NcclReduceScatter(const OpKernelInfo& info) : NcclKernel(info) {
}

Status NcclReduceScatter::ComputeInternal(OpKernelContext* context) const {
  cudaStream_t stream = nullptr;  // Default stream
  ncclComm_t comm = nccl_->Comm(group_type_);
  const int rank = nccl_->Rank(group_type_);
  const int size = nccl_->Size(group_type_);

  auto onnx_type = context->Input<Tensor>(0)->DataType();
  const size_t element_size = onnx_type->Size();
  ncclDataType_t dtype = GetNcclDataType(onnx_type);

  // Count total number of elements to ReduceScatter.
  const Tensor* first_tensor = context->Input<Tensor>(0);
  const Tensor* last_tensor = context->Input<Tensor>(context->InputCount() - 1);
  const char* start_address = reinterpret_cast<const char*>(first_tensor->DataRaw());
  const char* end_address = reinterpret_cast<const char*>(last_tensor->DataRaw()) + last_tensor->SizeInBytes();
  size_t buffer_size = end_address - start_address;

  // ReduceScatter requires every rank to receive the same amount of data, and significantly
  // slows down significantly if the data is not aligned.  Nvidia recommends 32-byte alignment,
  // so pad to multiple of 32 and world size.
  // Note: the alignment here needs to be kept in-sync with the alignment in zero_optimizer_graph_builder.cc
  const int64_t alignment = size * 32;
  ORT_ENFORCE(buffer_size % alignment == 0, "NcclReduceScatter's contiguous buffer is not padded to local_size * 32");

  // Calculate the range of outputs this rank will receive.
  const int64_t rank_bytes = buffer_size / size;
  const int64_t rank_count = rank_bytes / element_size;
  const int64_t rank_start = rank * rank_bytes;

  // ReduceScatter
  void* fusion_data_rank_offset = const_cast<char*>(start_address) + rank_start;
#ifdef ORT_USE_NCCL
  NCCL_RETURN_IF_ERROR(ncclReduceScatter(start_address, fusion_data_rank_offset, rank_count, dtype, ncclSum, comm, stream));
#endif

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    Tensor* output_tensor = context->Output(i, input_tensor->Shape());

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    output_tensor->SetByteOffset(input_tensor->ByteOffset());

    // Copy ReduceScatter results to outputs if needed
    const void* input_data = input_tensor->DataRaw();
    void* output_data = output_tensor->MutableDataRaw();
    if (input_data != output_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice));
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    NcclAllReduce,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .VariadicAlias(0, 0)  // outputs and inputs are mapped one to one
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    NcclAllReduce);

ONNX_OPERATOR_KERNEL_EX(
    NcclAllGather,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .VariadicAlias(0, 0)  // outputs and inputs are mapped one to one
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    NcclAllGather);

ONNX_OPERATOR_KERNEL_EX(
    NcclReduceScatter,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .VariadicAlias(0, 0)  // outputs and inputs are mapped one to one
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    NcclReduceScatter);

}  // namespace cuda
}  // namespace onnxruntime
