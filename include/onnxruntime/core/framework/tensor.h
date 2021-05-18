// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <iostream>
#include <string>
#include <vector>

#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/buffer_deleter.h"
#include "onnxruntime_config.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"

namespace onnxruntime {

//TODO:ensure dtype_!=nullptr
#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_NULL_DEREFERENCE
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#endif

////////////////////////////////////////////////////////////////////////
/// This class holds optional Meta data describing sparse format
// when Tensor instance holds data in its sparse format.
/// When Tensor holds sparse data then:
/// - it holds only non-zero values in its buffer
/// - its shape is the shape or rather size of the data in the buffer.
/// - SparseMetadata must return a non nullptr const pointer to an instance of SparseMetadata
///
/// and SparseMetadata contains the enumeration member that indicates its format.
class SparseMetadata {
 public:
  virtual ~SparseMetadata() = default;
  /// <summary>
  /// Only one class
  /// </summary>
  enum SparseFormat {
    kNVidiaTwoByFour = 1
  };
  SparseFormat Format() const { return format_; }
  /// <summary>
  /// Safe way to obtain a ptr to a concrete derived class
  /// </summary>
  /// <typeparam name="T"></typeparam>
  /// <returns></returns>
  template <class T>
  const T* Get() const;

 protected:
  explicit SparseMetadata(SparseFormat format) noexcept : format_(format) {}

 private:
  SparseFormat format_;
};

/// <summary>
/// The only supported type for now that does not require any
/// additional metadata as the data format 2:4 is really dense.
/// So Tensor itself would hold the data as usual.
/// </summary>
class SparseNVidiaTwoByFour : public SparseMetadata {
 public:
  SparseNVidiaTwoByFour() : SparseMetadata(kNVidiaTwoByFour) {}
};

/// <summary>
/// Specialize for concrete types.
/// </summary>
/// <returns></returns>
template <>
inline const SparseNVidiaTwoByFour* SparseMetadata::Get<SparseNVidiaTwoByFour>() const {
  ORT_ENFORCE(format_ == kNVidiaTwoByFour, "Invalid sparse format meta requested");
  return static_cast<const SparseNVidiaTwoByFour*>(this);
}

/*
  We want to keep tensor as simple as possible, it is just a placeholder
  for a piece of memory, with additional shape information.
  Memory is owned and managed by Executor / Workspace, so Tensor just uses
  it, and won't do any allocation / release.
*/
class Tensor final {
 public:
  // to allow creating vector<Tensor> to support seq(tensor)
  Tensor() noexcept
      : p_data_(nullptr),
        buffer_deleter_(),
        shape_(),
        dtype_(nullptr),
        alloc_info_(),
        byte_offset_(0),
        sparse_meta_() {
  }

  /**
   * Create tensor with given type, shape, pre-allocated memory and allocator info.
   * This function won't check if the preallocated buffer(p_data) has enough room for the shape.
   * \param p_type Data type of the tensor
   * \param shape Shape of the tensor
   * \param p_data A preallocated buffer. Can be NULL if the shape is empty.
   *              Tensor does not own the data and will not delete it
   * \param alloc Where the buffer('p_data') was allocated from
   * \param offset Offset in bytes to start of Tensor within p_data. 
   */
  Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
         ptrdiff_t offset = 0);

  /**
   * Deprecated. The original design is this Tensor class won't do any allocation / release.
   * However, this function will allocate the buffer for the shape, and do placement new if p_type is string tensor.
   */
  Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator);

  /**
   * Create tensor with given type, shape, pre-allocated memory and allocator which will be used to free the pre-allocated memory.
   * This function won't check if the preallocated buffer(p_data) has enough room for the shape.
   * However, this function will de-allocate the buffer upon the tensor getting destructed.
   * \param p_type Data type of the tensor
   * \param shape Shape of the tensor
   * \param p_data A preallocated buffer. Can be NULL if the shape is empty.
   *              Tensor does not own the data and will not delete it
   * \param deleter Allocator used to free the pre-allocated memory
   * \param offset Offset in bytes to start of Tensor within p_data. 
   */
  Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
         ptrdiff_t offset = 0);

  ~Tensor();

  //Move is allowed
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Tensor);

  Tensor(Tensor&& other) noexcept
      : Tensor() {
    *this = std::move(other);
  }

  Tensor& operator=(Tensor&& other) noexcept;

  /**
     Returns the data type.
  */
  MLDataType DataType() const { return dtype_; }

  /**
     Returns the data type enum constant
     @remarks Use utils::ToTensorProtoElementType<T> for comparison.
  */
  int32_t GetElementType() const {
    return dtype_->GetDataType();
  }

  // Check if contains string data. This is a separate
  // interface bc it is frequently used.
  bool IsDataTypeString() const {
    return utils::IsPrimitiveDataType<std::string>(dtype_);
  }

  // Checks if the Tensor contains data type T
  template <class T>
  bool IsDataType() const {
    return utils::IsPrimitiveDataType<T>(dtype_);
  }

  /**
     Returns the shape of the tensor.
  */
  const TensorShape& Shape() const noexcept { return shape_; }

  /**
     Returns the location of the tensor's memory
  */
  const OrtMemoryInfo& Location() const { return alloc_info_; }

  /**
     May return nullptr if tensor size is zero
  */
  template <typename T>
  T* MutableData() {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset_);
  }

  /**
     May return nullptr if tensor size is zero
  */
  template <typename T>
  gsl::span<T> MutableDataAsSpan() {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset_);
    return gsl::make_span(data, static_cast<size_t>(shape_.Size()));
  }

  template <typename T>
  const T* Data() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<const T*>(static_cast<char*>(p_data_) + byte_offset_);
  }

  template <typename T>
  gsl::span<const T> DataAsSpan() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    const T* data = reinterpret_cast<const T*>(static_cast<char*>(p_data_) + byte_offset_);
    return gsl::make_span(data, static_cast<typename gsl::span<T>::index_type>(shape_.Size()));
  }

  void* MutableDataRaw(MLDataType type) {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return static_cast<char*>(p_data_) + byte_offset_;
  }

  const void* DataRaw(MLDataType type) const {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return static_cast<char*>(p_data_) + byte_offset_;
  }

  void* MutableDataRaw() noexcept {
    return static_cast<char*>(p_data_) + byte_offset_;
  }

  const void* DataRaw() const noexcept {
    return static_cast<char*>(p_data_) + byte_offset_;
  }

  bool OwnsBuffer() const noexcept {
    return buffer_deleter_ != nullptr;
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   * @warning this function is NOT thread-safe.
   */
  inline void Reshape(const TensorShape& new_shape) {
    ORT_ENFORCE(shape_.Size() == new_shape.Size(),
                "Tensor size (" + std::to_string(shape_.Size()) +
                    ") != new size (" + std::to_string(new_shape.Size()) + ")");
    shape_ = new_shape;
  }

  /**
   * Get the byte offset with respect to the p_data
   * @warning this is a temporary solution for reusing the buffer bigger than needed.
   * @warning use with caution - make sure you do boundary check before calling this method (see view.cc)
   */
  inline ptrdiff_t ByteOffset() const {
    return byte_offset_;
  }

  /**
   * Set the byte offset with respect to the p_data
   * @warning this is a temporary solution for reusing the buffer bigger than needed.
   */
  inline void SetByteOffset(ptrdiff_t byte_offset) {
    byte_offset_ = byte_offset;
  }

  /**
  The number of bytes of data.
  */
  size_t SizeInBytes() const;

  /**
  * Get SparseMetada. If it is a dense matrix, it returns nullptr
  */
  const SparseMetadata* SparseMeta() const {
    return sparse_meta_.get();
  }

  /**
  * Attach SparseMetadata
  */
  void AttachSparseMetadata(std::unique_ptr<SparseMetadata> sparse_meta) {
    sparse_meta_ = std::move(sparse_meta);
  }

  // More API methods.
 private:
  void Init(MLDataType p_type,
            const TensorShape& shape,
            void* p_raw_data,
            AllocatorPtr deleter,
            ptrdiff_t offset = 0);

  void ReleaseBuffer();

  void* p_data_;
  /**
     if buffer_deleter_ is null, it means tensor does not own the buffer.
     otherwise tensor will use the deleter to release the buffer when
     tensor is released.
  */
  AllocatorPtr buffer_deleter_;

  TensorShape shape_;
  const PrimitiveDataTypeBase* dtype_;
  OrtMemoryInfo alloc_info_;
  ptrdiff_t byte_offset_;
  std::unique_ptr<SparseMetadata> sparse_meta_;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime