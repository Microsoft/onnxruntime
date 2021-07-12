// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {

/**
 * Copies a tensor to the provided byte span.
 *
 * @param data_transfer_manager The data transfer manager instance.
 * @param src_tensor The tensor to copy from.
 * @param dst_alloc_info An OrtMemoryInfo instance corresponding to the destination span memory.
 * @param dst_span The span to copy to.
 * @return The status of the operation.
 */
inline Status CopyTensorDataToByteSpan(
    const DataTransferManager& data_transfer_manager,
    const Tensor& src_tensor,
    const OrtMemoryInfo& dst_alloc_info, gsl::span<char> dst_span) {
  ORT_RETURN_IF_NOT(src_tensor.SizeInBytes() == static_cast<size_t>(dst_span.size_bytes()), "src size != dst size");
  Tensor dst_tensor{src_tensor.DataType(), src_tensor.Shape(), dst_span.data(), dst_alloc_info};
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(src_tensor, dst_tensor));
  return Status::OK();
}

/**
 * Copies a tensor to the provided span.
 *
 * @param data_transfer_manager The data transfer manager instance.
 * @param src_tensor The tensor to copy from.
 * @param dst_alloc_info An OrtMemoryInfo instance corresponding to the destination span memory.
 * @param dst_span The span to copy to.
 * @return The status of the operation.
 */
template <typename TElement>
common::Status CopyTensorDataToSpan(
    const DataTransferManager& data_transfer_manager,
    const Tensor& src_tensor,
    const OrtMemoryInfo& dst_alloc_info, gsl::span<TElement> dst_span) {
  static_assert(std::is_trivially_copyable<TElement>::value, "Element type must be trivially copyable.");
  ORT_RETURN_IF_NOT(src_tensor.DataType() == DataTypeImpl::GetType<TElement>(), "Data type mismatch");
  ORT_RETURN_IF_NOT(src_tensor.SizeInBytes() == static_cast<size_t>(dst_span.size_bytes()), "src size != dst size");
  Tensor dst_tensor{src_tensor.DataType(), src_tensor.Shape(), dst_span.data(), dst_alloc_info};
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(src_tensor, dst_tensor));
  return Status::OK();
}

}  // namespace onnxruntime
