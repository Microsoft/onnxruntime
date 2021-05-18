// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/common/safeint.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/data_types.h"

namespace onnxruntime {

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
               ptrdiff_t offset)
    : alloc_info_(alloc) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, nullptr, offset);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  int64_t shape_size = shape.Size();  // value returned is checked for overflow by TensorShape::Size()
  if (shape_size < 0)
    ORT_THROW("shape.Size() must >=0");

  void* p_data = nullptr;
  if (shape_size > 0) {
    SafeInt<size_t> len = 0;
    if (!allocator->CalcMemSizeForArray(SafeInt<size_t>(shape_size), p_type->Size(), &len))
      ORT_THROW("tensor failed memory size calculation");

    p_data = allocator->Alloc(len);
  }

  Init(p_type, shape, p_data, allocator);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
               ptrdiff_t offset)
    : alloc_info_(deleter->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, deleter, offset);
}

size_t Tensor::SizeInBytes() const {
  size_t ret;
  if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_.Size()), dtype_->Size(), &ret)) {
    ORT_THROW("tensor size overflow");
  }
  return ret;
}

void Tensor::Init(MLDataType p_type, const TensorShape& shape, void* p_raw_data, AllocatorPtr deleter, ptrdiff_t offset) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");
  dtype_ = p_type->AsPrimitiveDataType();
  ORT_ENFORCE(dtype_ != nullptr, "Tensor is expected to contain one of the primitive data types. Got: ",
              DataTypeImpl::ToString(p_type));
  shape_ = shape;
  p_data_ = p_raw_data;
  // if caller passed in a deleter, that means this tensor own this buffer
  // we will release the buffer when this tensor is deconstructed.
  buffer_deleter_ = std::move(deleter);
  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (buffer_deleter_ && IsDataTypeString()) {
    auto* ptr = static_cast<std::string*>(p_data_);
    for (int64_t i = 0, n = shape_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  }
  byte_offset_ = offset;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    ReleaseBuffer();

    p_data_ = other.p_data_;
    buffer_deleter_ = std::move(other.buffer_deleter_);
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    alloc_info_ = std::move(other.alloc_info_);
    byte_offset_ = other.byte_offset_;
    sparse_meta_ = std::move(other.sparse_meta_);

    other.p_data_ = nullptr;
    assert(!other.buffer_deleter_);
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
    other.alloc_info_ = OrtMemoryInfo{};
    other.byte_offset_ = 0;
    assert(!other.sparse_meta_);
  }
  return *this;
}

Tensor::~Tensor() {
  ReleaseBuffer();
}

void Tensor::ReleaseBuffer() {
  if (buffer_deleter_) {
    // if current tensor is responsible for deleting the buffer
    // and it is a string tensor, need to explicitly call string(s)
    // __dtor(s).
    if (IsDataTypeString()) {
      using string = std::string;
      auto* ptr = static_cast<std::string*>(p_data_);
      const int64_t len = shape_.Size();
      for (int64_t i = 0; i < len; i++)
        ptr[i].~string();
    }
    buffer_deleter_->Free(p_data_);
    p_data_ = nullptr;
    buffer_deleter_.reset();
  }
}

}  // namespace onnxruntime
