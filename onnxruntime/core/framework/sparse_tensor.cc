// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/utils.h"

#include <safeint/SafeInt.hpp>

using namespace onnxruntime::common;

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, SparseFormat flags) {
  return os << std::hex << static_cast<uint32_t>(flags);
}

namespace {
// Round up size to a multiple of align.
// Example:
//   Roundup(13, 5)   => 15
//   Roundup(201, 16) => 208
constexpr size_t kIndexAlignment = alignof(int64_t);
int64_t Roundup(int64_t size) {
  return ((SafeInt<int64_t>(size) + kIndexAlignment - 1) / kIndexAlignment) * kIndexAlignment;
}

/// <summary>
/// Calculate required buffer size. We will place indices
/// after data and make sure indices start at int64_t aligned place
/// </summary>
/// <returns></returns>
int64_t CalculateRequiredBufferSize(int64_t data_size, int64_t indices_size) {
  return SafeInt<int64_t>(Roundup(data_size)) + indices_size;
}
}  // namespace

const void* SparseTensor::IndicesStart(int64_t values_bytes) const {
  if (p_data_ != nullptr) {
    return reinterpret_cast<const uint8_t*>(p_data_) + Roundup(values_bytes);
  }
  return nullptr;
}

void* SparseTensor::IndicesStart(int64_t values_bytes) {
  if (p_data_ != nullptr) {
    return reinterpret_cast<uint8_t*>(p_data_) + Roundup(values_bytes);
  }
  return nullptr;
}

int64_t SparseTensor::RequiredAllocationSize() const noexcept {
  if (p_data_ != nullptr) {
    // Can be zero for zero matrices
    assert(buffer_size_ >= 0);
    return buffer_size_;
  } else {
    auto data_size = values_.SizeInBytes();
    int64_t index_size = 0;
    for (const auto& t : format_data_) {
      index_size += t.SizeInBytes();
    }
    return CalculateRequiredBufferSize(data_size, index_size);
  }
}

SparseTensor::SparseTensor(MLDataType elt_type, const TensorShape& dense_shape,
                           const TensorShape& values_shape, void* values_data,
                           const OrtMemoryInfo& location)
    : SparseTensor() {
  dense_shape_ = dense_shape;
  ml_data_type_ = elt_type->AsPrimitiveDataType();
  location_ = location;
  values_ = Tensor(elt_type, values_shape, values_data, location_);
}

SparseTensor::SparseTensor(MLDataType elt_type, const TensorShape& dense_shape,
                           std::shared_ptr<IAllocator> allocator)
    : SparseTensor() {
  dense_shape_ = dense_shape;
  ml_data_type_ = elt_type->AsPrimitiveDataType();
  allocator_ = std::move(allocator);
  location_ = allocator_->Info();
}

SparseTensor::SparseTensor() noexcept
    : format_(SparseFormat::kUndefined),
      dense_shape_(),
      ml_data_type_(nullptr),
      allocator_(),
      location_(),
      p_data_(nullptr),
      buffer_size_(0),
      values_(),
      format_data_() {
}

SparseTensor::SparseTensor(SparseTensor&& o) noexcept : SparseTensor() {
  *this = std::move(o);
}

SparseTensor& SparseTensor::operator=(SparseTensor&& o) noexcept {
  ReleaseBuffer();
  format_ = o.format_;
  dense_shape_ = std::move(o.dense_shape_);
  ml_data_type_ = o.ml_data_type_;
  allocator_ = std::move(o.allocator_);
  location_ = std::move(o.location_);
  std::swap(p_data_, o.p_data_);
  std::swap(buffer_size_, o.buffer_size_);
  values_ = std::move(o.values_);
  format_data_ = std::move(o.format_data_);
  return *this;
}

SparseTensor::~SparseTensor() {
  ReleaseBuffer();
}

Status SparseTensor::AllocateBuffer(int64_t buffer_size, size_t num_values) {
  if (buffer_size > 0) {
    SafeInt<size_t> b_size_t(buffer_size);
    ORT_ENFORCE(b_size_t.Ref() > (num_values * ml_data_type_->Size()),
                "Values size must be less than total buffer size");
    auto data_ptr = IAllocator::MakeUniquePtr<void>(allocator_, b_size_t);
    ORT_RETURN_IF_NOT(data_ptr != nullptr, "SparseTensor Allocation failed for size: ", buffer_size);
    if (IsDataTypeString()) {
      // We own the buffer, so we must properly construct strings. Neither of the Tensors
      // we construct on top of the buffer own it. We are constructing empty strings, hopefully
      // nothrow and no buffer allocation
      utils::ConstructStrings(data_ptr.get(), static_cast<int64_t>(num_values));
    }
    p_data_ = data_ptr.release();
  }
  buffer_size_ = buffer_size;
  return Status::OK();
}

void SparseTensor::ReleaseBuffer() {
  if (allocator_ && p_data_ != nullptr) {
    // if current tensor is responsible for deleting the buffer
    // and it is a string tensor, need to explicitly call string(s)
    // __dtor(s).
    utils::ReleaseTensorBuffer(allocator_, IsDataTypeString(), p_data_, values_.Shape().Size());
  }
  p_data_ = nullptr;
  buffer_size_ = 0;
}

void SparseTensor::CopyStrings(const Tensor& src, Tensor& dst) const {
  auto src_span = src.DataAsSpan<std::string>();
  auto* dst_iter = dst.MutableData<std::string>();
  std::copy(src_span.cbegin(), src_span.cend(), dst_iter);
}

Status SparseTensor::CopyData(const IDataTransfer& data_transfer,
                              const std::vector<std::reference_wrapper<const Tensor>>& src,
                              const std::vector<std::reference_wrapper<Tensor>>& dst) {
  ORT_RETURN_IF_NOT(src.size() == dst.size(), "Must have the same size");
  for (size_t i = 0; i < src.size(); ++i) {
    const Tensor& src_t = src[i];
    Tensor& dst_t = dst[i];
    if (src_t.IsDataTypeString()) {
      CopyStrings(src_t, dst_t);
    } else {
      ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(src_t, dst_t));
    }
  }
  return Status::OK();
}

SparseTensor::CooView SparseTensor::AsCoo() const {
  ORT_ENFORCE(Format() == SparseFormat::kCoo, "Must contain Coo format");
  ORT_ENFORCE(format_data_.size() == 1U, "Expecting one index");
  return CooView(format_data_[0]);
}

std::vector<int64_t> SparseTensor::GetCooIndexDims(size_t values_count, size_t index_size) const {
  std::vector<int64_t> index_dims{gsl::narrow<int64_t>(values_count)};
  if (values_count * 2 == index_size) {
    // Coor index
    index_dims.push_back(2);
  } else {
    ORT_ENFORCE(values_count == index_size,
                      "Index size must be equal to or twice the values size");
  }
  return index_dims;
}

void SparseTensor::InitCooIndex(const TensorShape& index_shape, const int64_t* index_data) {
  format_data_.resize(1);
  format_data_[0] = Tensor(DataTypeImpl::GetType<int64_t>(), index_shape, const_cast<int64_t*>(index_data), Location());
  format_ = SparseFormat::kCoo;
}

Status SparseTensor::UseCooIndex(gsl::span<const int64_t> index) {
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Not expecting an allocator set");
  TensorShape index_shape(GetCooIndexDims(NumValues(), index.size()));
  InitCooIndex(index_shape, index.data());
  return Status::OK();
}

Status SparseTensor::MakeCooData(const IDataTransfer& data_transfer,
                                 const OrtMemoryInfo& data_location,
                                 size_t values_count, const void* values_data,
                                 gsl::span<const int64_t> index) {
  auto mutator = MakeCooData(values_count, index.size());
  if (values_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_index = mutator.Index();

    Tensor src_values(dst_values.DataType(), dst_values.Shape(), const_cast<void*>(values_data), data_location);
    Tensor src_index(dst_index.DataType(), dst_index.Shape(), const_cast<int64_t*>(index.data()), data_location);
    ORT_RETURN_IF_ERROR(CopyData(data_transfer, MakeListConst(src_values, src_index), MakeListNonConst(dst_values, dst_index)));
  }
  return Status::OK();
}

SparseTensor::CooMutator SparseTensor::MakeCooData(size_t values_count, size_t index_count) {
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  const auto num_values = gsl::narrow<int64_t>(values_count);
  TensorShape index_shape(GetCooIndexDims(values_count, index_count));
  TensorShape values_shape{num_values};
  if (num_values > 0) {
    const auto data_size = values_count * ml_data_type_->Size();
    const auto index_size = index_count * sizeof(int64_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(gsl::narrow<int64_t>(data_size),
                                                                  gsl::narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, values_count));
  }
  values_ = Tensor(DataType(), values_shape, p_data_, Location());
  InitCooIndex(index_shape, reinterpret_cast<int64_t*>(IndicesStart(values_.SizeInBytes())));
  return CooMutator(values_, format_data_[0]);
}

SparseTensor::CsrView SparseTensor::AsCsr() const {
  ORT_ENFORCE(Format() == SparseFormat::kCsrc, "Must contain Csr format");
  ORT_ENFORCE(format_data_.size() == 2U, "Expecting two indices");
  return CsrView(format_data_[0], format_data_[1]);
}

void SparseTensor::ValidateCsrIndices(size_t values_count, size_t inner_size, size_t outer_size) const {
  ORT_ENFORCE(dense_shape_.NumDimensions() == 2U, "dense shape must 2-D");
  ORT_ENFORCE(inner_size == values_count, "Expecting inner index size the same as values size");
  const auto rows = dense_shape_.GetDims()[0];
  ORT_ENFORCE(outer_size == 0 || outer_size == static_cast<size_t>(rows + 1),
              "Outer index count must be rows + 1 or zero");
}

void SparseTensor::InitCsrIndices(size_t inner_size, const int64_t* inner, size_t outer_size, const int64_t* outer) {
  TensorShape inner_shape{static_cast<int64_t>(inner_size)};
  TensorShape outer_shape{static_cast<int64_t>(outer_size)};
  auto index_type = DataTypeImpl::GetType<int64_t>();
  format_data_.resize(2);
  format_data_[0] = Tensor(index_type, inner_shape, const_cast<int64_t*>(inner), Location());
  format_data_[1] = Tensor(index_type, outer_shape, const_cast<int64_t*>(outer), Location());

  format_ = SparseFormat::kCsrc;
}

Status SparseTensor::UseCsrIndices(gsl::span<int64_t> inner_index, gsl::span<int64_t> outer_index) {
  ORT_ENFORCE(allocator_ == nullptr, "This method does not expect allocator to be set");
  ValidateCsrIndices(NumValues(), inner_index.size(), outer_index.size());
  InitCsrIndices(inner_index.size(), inner_index.data(), outer_index.size(), outer_index.data());
  return Status::OK();
}

Status SparseTensor::MakeCsrData(const IDataTransfer& data_transfer, const OrtMemoryInfo& data_location,
                                 size_t values_count, void* values_data,
                                 gsl::span<const int64_t> inner_index, gsl::span<const int64_t> outer_index) {
  auto mutator = MakeCsrData(values_count, inner_index.size(), outer_index.size());
  if (values_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_inner = mutator.Inner();
    auto& dst_outer = mutator.Outer();

    Tensor src_values(dst_values.DataType(), dst_values.Shape(), values_data, data_location);
    Tensor src_inner(dst_inner.DataType(), dst_inner.Shape(), const_cast<int64_t*>(inner_index.data()), data_location);
    Tensor src_outer(dst_outer.DataType(), dst_outer.Shape(), const_cast<int64_t*>(outer_index.data()), data_location);
    ORT_RETURN_IF_ERROR(CopyData(data_transfer, MakeListConst(src_values, src_inner, src_outer),
                                 MakeListNonConst(dst_values, dst_inner, dst_outer)));
  }
  return Status::OK();
}

SparseTensor::CsrMutator SparseTensor::MakeCsrData(size_t values_count,
                                                   size_t inner_index_count,
                                                   size_t outer_index_count) {
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  ValidateCsrIndices(values_count, inner_index_count, outer_index_count);

  if (values_count > 0) {
    const auto data_size = values_count * ml_data_type_->Size();
    const auto index_size = (inner_index_count + outer_index_count) * sizeof(int64_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(gsl::narrow<int64_t>(data_size),
                                                                  gsl::narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, values_count));
  }

  const auto num_values = gsl::narrow<int64_t>(values_count);
  values_ = Tensor(DataType(), {num_values}, p_data_, Location());

  auto* inner_index_start = reinterpret_cast<int64_t*>(IndicesStart(values_.SizeInBytes()));
  InitCsrIndices(inner_index_count, inner_index_start, outer_index_count, inner_index_start + inner_index_count);
  return CsrMutator(values_, format_data_[0], format_data_[1]);
}

SparseTensor::BlockSparseView SparseTensor::AsBlockSparse() const {
  ORT_ENFORCE(Format() == SparseFormat::kBlockSparse, "Must contain BlockSparse format");
  ORT_ENFORCE(format_data_.size() == 1U, "Expecting one index");
  return BlockSparseView(format_data_[0]);
}

Status SparseTensor::ValidateBlockSparseShapes(const TensorShape& values_shape, const TensorShape& index_shape) const {
  ORT_RETURN_IF_NOT(values_shape.NumDimensions() >= 3, "Expecting values dimensions to be at least 3");
  ORT_RETURN_IF_NOT(index_shape.NumDimensions() == 2, "Expecting index dimensions to be 2");
  const auto values_blocks = values_shape.SizeFromDimension(2);
  const auto index_blocks = index_shape.Size() / 2;  // Two integers per block
  ORT_RETURN_IF_NOT(values_blocks == index_blocks, "Expecting index blocks to be equal to values blocks");
  return Status::OK();
}

Status SparseTensor::UseBlockSparseIndices(const TensorShape& index_shape, const int32_t* index_data) {
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Not expecting an allocator set");
  ORT_RETURN_IF_ERROR(ValidateBlockSparseShapes(Values().Shape(), index_shape));

  format_data_.resize(1);
  format_data_[0] = Tensor(DataTypeImpl::GetType<int32_t>(), index_shape,
                           const_cast<int32_t*>(index_data), Location());
  format_ = SparseFormat::kBlockSparse;
  return Status::OK();
}

Status SparseTensor::MakeBlockSparseData(const IDataTransfer& data_transfer, const OrtMemoryInfo& data_location,
                                         const TensorShape& values_shape, const void* values_data,
                                         const TensorShape& index_shape, const int32_t* index_data) {
  auto mutator = MakeBlockSparseData(values_shape, index_shape);
  if (values_shape.Size() > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_index = mutator.Index();
    Tensor src_values(dst_values.DataType(), dst_values.Shape(), const_cast<void*>(values_data), data_location);
    Tensor src_index(dst_index.DataType(), dst_index.Shape(), const_cast<int32_t*>(index_data), data_location);
    ORT_RETURN_IF_ERROR(CopyData(data_transfer, MakeListConst(src_values, src_index), MakeListNonConst(dst_values, dst_index)));
  }
  return Status::OK();
}

SparseTensor::BlockSparseMutator SparseTensor::MakeBlockSparseData(const TensorShape& values_shape, const TensorShape& index_shape) {
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  ORT_THROW_IF_ERROR(ValidateBlockSparseShapes(values_shape, index_shape));
  if (values_shape.Size() > 0) {
    const auto data_size = values_shape.Size() * ml_data_type_->Size();
    const auto index_size = index_shape.Size() * sizeof(int32_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(gsl::narrow<int64_t>(data_size),
                                                                  gsl::narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, static_cast<size_t>(data_size / ml_data_type_->Size())));
  }
  values_ = Tensor(DataType(), values_shape, p_data_, Location());
  format_data_.resize(1);
  format_data_[0] = Tensor(DataTypeImpl::GetType<int32_t>(), index_shape, IndicesStart(values_.SizeInBytes()), Location());
  format_ = SparseFormat::kBlockSparse;
  return BlockSparseMutator(values_, format_data_[0]);
}

Status SparseTensor::Copy(const DataTransferManager& data_transfer_manager, int exec_q_id, SparseTensor& dst_tensor) const {
  const IDataTransfer* data_transfer = data_transfer_manager.GetDataTransfer(Location().device,
                                                                             dst_tensor.Location().device);
  ORT_RETURN_IF_NOT(data_transfer != nullptr, "Unable to find a data transfer for copying from device type: ",
                    Location().device.Type(), " to device type: ", dst_tensor.Location().device.Type());

  return Copy(*data_transfer, dst_tensor, exec_q_id);
}

Status SparseTensor::Copy(const IDataTransfer& data_transfer, SparseTensor& dst_tensor, int exec_q_id) const {
  // Do not copy same destination
  if (this == &dst_tensor) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(format_ != SparseFormat::kUndefined, "This instance should not be empty");
  const bool is_string = IsDataTypeString();

  ORT_RETURN_IF_NOT(dst_tensor.Format() == SparseFormat::kUndefined, "Destination should be empty");
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have a CPU allocator set");
  ORT_RETURN_IF_NOT((!is_string || dst_tensor.Location().device.Type() == OrtDevice::CPU),
                    "X-device copy of strings not supported");
  ORT_RETURN_IF_NOT(dst_tensor.DataType() == DataType(), "Src and Dst must be of the same type");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() == dense_shape_.Size(), "Must have the same shape");

  const auto required_buffer_size = RequiredAllocationSize();
  SparseTensor result(DataType(), Shape(), dst_tensor.allocator_);
  ORT_RETURN_IF_ERROR(result.AllocateBuffer(required_buffer_size, NumValues()));

  const auto values_bytes = values_.SizeInBytes();
  auto* const dst_index_start = reinterpret_cast<int8_t*>(result.IndicesStart(values_bytes));
  result.format_data_.resize(format_data_.size());
  int64_t index_bytes = 0;
  for (size_t i = 0; i < format_data_.size(); ++i) {
    const auto& src_idx = format_data_[i];
    result.format_data_[i] = Tensor(src_idx.DataType(), src_idx.Shape(),
                                    dst_index_start + index_bytes,
                                    result.Location());
    const auto size_in_bytes = src_idx.SizeInBytes();
    index_bytes += size_in_bytes;
    // We donot have a contiguous buffer, copy one by one
    if (p_data_ == nullptr && size_in_bytes > 0) {
      ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(src_idx, result.format_data_[i], exec_q_id));
    }
  }

  Tensor result_values(DataType(), Values().Shape(), result.p_data_, result.Location());
  if (Values().Shape().Size() > 0) {
    if (is_string) {
      CopyStrings(Values(), result_values);
      if (p_data_ != nullptr) {
        // We are on CPU
        memcpy(dst_index_start, IndicesStart(values_bytes), static_cast<size_t>(index_bytes));
      }
    } else {
      if (p_data_ == nullptr) {
        // No contiguous buffer, copy only values, indices copied above
        ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Values(), result_values, exec_q_id));
      } else {
        // Copy everything at once, setup artificial tensors
        // that describe the whole buffer
        auto bytes_type = DataTypeImpl::GetType<uint8_t>();
        TensorShape buffer_shape{required_buffer_size};
        Tensor src(bytes_type, buffer_shape, p_data_, Location());
        Tensor dst(bytes_type, buffer_shape, result.p_data_, result.Location());
        ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(src, dst, exec_q_id));
      }
    }
  }

  result.values_ = std::move(result_values);
  result.format_ = Format();

  dst_tensor = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime
