// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
class SparseTensor;
class DataTransferManager;
namespace common {
class Status;
}
#endif

namespace sparse_utils {
#if !defined(ORT_MINIMAL_BUILD)
/// <summary>
/// This function converts dense tensor into Csr format.
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy is made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src"></param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator"></param>
/// <param name="dst"></param>
/// <returns></returns>
Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, SparseTensor& dst);

/// <summary>
/// Converts Csr format to Dense matrix
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy is made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src"></param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator"></param>
/// <param name="dst"></param>
/// <returns></returns>
Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst);

/// <summary>
/// <summary>
/// Convert COO format to dense matrix
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy is made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src"></param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator"></param>
/// <param name="dst"></param>
/// <returns>Status instance</returns>
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src"></param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator"></param>
/// <param name="dst"></param>
/// <returns></returns>
Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst);
#endif  //ORT_MINIMAL_BUILD

/// <summary>
/// Convert Dense Tensor to COO format
/// Conversion takes place on CPU. Thus if the source
/// is not on CPU, a copy is made first. Likewise, if the destination
/// is not on CPU, the function would perform a copy.
/// std::string src and destination are assumed to be both on CPU
/// </summary>
/// <param name="data_manager"></param>
/// <param name="src"></param>
/// <param name="cpu_allocator"></param>
/// <param name="dst_allocator"></param>
/// <param name="linear_index"></param>
/// <param name="dst"></param>
/// <returns></returns>
Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_index, SparseTensor& dst);

}  // namespace sparse_utils
}  // namespace onnxruntime
