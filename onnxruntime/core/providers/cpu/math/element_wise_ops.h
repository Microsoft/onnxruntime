// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
namespace functors {
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Do not use raw new/delete.
#pragma warning(disable : 26409)
#endif
template <typename T>
struct Log final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 15.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.log();
  }
};

template <typename T>
struct Abs final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseAbs();
  }
};

template <typename T>
struct Neg final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = -xm;
  }
};

template <typename T>
struct Floor final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.floor();
  }
};

template <typename T>
struct Ceil final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.ceil();
  }
};

template <typename T>
struct Reciprocal final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseInverse();
  }
};

template <typename T>
struct Sqrt final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 2.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseSqrt();
  }
};

template <typename T>
struct Exp final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 2.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.exp();
  }
};
}  // namespace functors

DEFINE_ELE_KERNEL(Log)
DEFINE_ELE_KERNEL(Abs)
DEFINE_ELE_KERNEL(Neg)
DEFINE_ELE_KERNEL(Floor)
DEFINE_ELE_KERNEL(Ceil)
DEFINE_ELE_KERNEL(Reciprocal)
DEFINE_ELE_KERNEL(Sqrt)
DEFINE_ELE_KERNEL(Exp)

template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public OpKernel {
 public:
  Sub(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public OpKernel {
 public:
  Mul(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public OpKernel {
 public:
  Div(const OpKernelInfo& info) : OpKernel(info) {
    // If the divisor is a constant initializer, validate for integer division by zero once
    // during kernel creation instead of on every Compute call.
    if constexpr (std::is_integral<T>::value) {
      const Tensor* constant_divisor = nullptr;
      if (info.TryGetConstantInput(1, &constant_divisor)) {
        const T* b_data = constant_divisor->Data<T>();
        const int64_t b_size = constant_divisor->Shape().Size();
        for (int64_t i = 0; i < b_size; ++i) {
          ORT_ENFORCE(b_data[i] != T{0}, "Integer division by zero");
        }
        divisor_is_validated_constant_ = true;
      }
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool divisor_is_validated_constant_{false};
};

class Pow final : public OpKernel {
 public:
  Pow(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_6 : public OpKernel {
 public:
  Sum_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_8 final : public OpKernel {
 public:
  Sum_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min_6 final : public OpKernel {
 public:
  Min_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// Max versions 8 - 12
// Version 8 added broadcast
// Version 12 added types support
class Min_8 final : public OpKernel {
 public:
  Min_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

template <typename T>
class Max_6 final : public OpKernel {
 public:
  Max_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// Max versions 8 - 12
// Version 8 added broadcast
// Version 12 added types support
class Max_8 final : public OpKernel {
 public:
  Max_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

class Not final : public OpKernel {
 public:
  Not(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class And final : public OpKernel {
 public:
  And(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Or final : public OpKernel {
 public:
  Or(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Xor final : public OpKernel {
 public:
  Xor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public OpKernel {
 public:
  Equal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public OpKernel {
 public:
  Less(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Greater final : public OpKernel {
 public:
  Greater(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class LessOrEqual final : public OpKernel {
 public:
  LessOrEqual(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class GreaterOrEqual final : public OpKernel {
 public:
  GreaterOrEqual(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_6 final : public OpKernel {
 public:
  Mean_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_8 final : public OpKernel {
 public:
  Mean_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class BitShift final : public OpKernel {
 public:
  explicit BitShift(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  bool shift_left_;
};

template <typename T>
class BitwiseAnd final : public OpKernel {
 public:
  explicit BitwiseAnd(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class BitwiseNot final : public OpKernel {
 public:
  explicit BitwiseNot(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class BitwiseOr final : public OpKernel {
 public:
  explicit BitwiseOr(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class BitwiseXor final : public OpKernel {
 public:
  explicit BitwiseXor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public OpKernel {
  // Currently only registered for float. Integer types would require addressing
  // overflow in multiplication and avoiding 0.0*inf=NaN in branchless formulations.
  static_assert(std::is_floating_point_v<T>,
                "PRelu is only supported for floating-point types.");

 public:
  PRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Expand_8 final : public OpKernel {
 public:
  Expand_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Erf final : public OpKernel {
 public:
  Erf(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
auto MakeEigenArrayMap(Tensor& t) -> EigenVectorArrayMap<T> {
  return EigenVectorArrayMap<T>(t.MutableData<T>(), narrow<ptrdiff_t>(t.Shape().Size()));
}

template <typename T>
auto MakeEigenArrayMap(const Tensor& t) -> ConstEigenVectorArrayMap<T> {
  return ConstEigenVectorArrayMap<T>(t.Data<T>(), narrow<ptrdiff_t>(t.Shape().Size()));
}

struct BroadcastIterator {
  size_t Current() const noexcept { return index_; }

  size_t AdvanceBy(size_t delta) {
    size_t index = index_;

    index_ += deltas_[0] * delta;
    counters_[0] += delta;
    if (counters_[0] == counts_[0]) {
      counters_[0] = 0;
      for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
        index_ += deltas_[counterIndex];
        if (++counters_[counterIndex] != counts_[counterIndex])
          break;
        counters_[counterIndex] = 0;
      }
    } else if (counters_[0] > counts_[0]) {  // Keep original logic above so that in most case it is faster
      delta = counters_[0] / counts_[0];
      counters_[0] = counters_[0] % counts_[0];
      for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
        index_ += delta * deltas_[counterIndex];
        counters_[counterIndex] += delta;
        if (counters_[counterIndex] < counts_[counterIndex]) break;
        delta = counters_[counterIndex] / counts_[counterIndex];
        counters_[counterIndex] = counters_[counterIndex] % counts_[counterIndex];
      }
    }
    return index;
  }

  void Reserve(ptrdiff_t max_dims) {
    deltas_.reserve(static_cast<size_t>(max_dims));
    counts_.reserve(static_cast<size_t>(max_dims));
  }

  void Init(ptrdiff_t axis, ptrdiff_t largest) {
    ORT_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    deltas_.push_back(axis > 1);
    counts_.push_back(largest);
    count_ *= axis;
  }

  void AllocateCounters() {
    counters_.resize(counts_.size(), 0);
  }

  ptrdiff_t GetCountsFront() const {
    return counts_.front();
  }

  ptrdiff_t GetDeltasFront() const {
    return deltas_.front();
  }

  void Append(ptrdiff_t axis, ptrdiff_t largest) {
    ORT_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    // If we're greater than 1, it doesn't matter what the other tensor does
    if (axis > 1) {
      if (deltas_.back() <= 0)  // Were we broadcasting
        StopBroadcasting();
    } else {  // We must be 1, at this point
      if (deltas_.back() > 0)
        StartBroadcasting();
    }

    counts_.back() *= largest;  // Just increase the last count
    count_ *= axis;
  }

  void StopBroadcasting() {
    deltas_.push_back(count_);
    counts_.push_back(1);
  }

  void StartBroadcasting() {
    deltas_.push_back(-count_);
    counts_.push_back(1);
  }

 private:
  using BroadCastVector = InlinedVector<ptrdiff_t, kTensorShapeSmallBufferElementsSize>;

  BroadCastVector counters_;
  BroadCastVector deltas_;
  BroadCastVector counts_;
  ptrdiff_t count_{1};  // Running total count of entries in tensor, used while building up the entries

  size_t index_{};
};

struct Broadcaster {
  Broadcaster(gsl::span<const int64_t> shape1, gsl::span<const int64_t> shape2) {
    size_t dimension_count_max = std::max(shape1.size(), shape2.size());
    size_t dimension_count_min = std::min(shape1.size(), shape2.size());
    output_shape_.resize(dimension_count_max);
    iterator1_.Reserve(dimension_count_max);
    iterator2_.Reserve(dimension_count_max);

    auto iter1 = shape1.end();
    auto iter2 = shape2.end();
    auto output_shape = output_shape_.end();

    // Scalars are a special case, as it's always a broadcast
    size_t index = 0;
    if (dimension_count_min == 0) {
      if (shape1.empty())  // Shape1 is a scalar
      {
        if (shape2.empty())  // Two scalars?
        {
          iterator1_.Init(1, 1);
          iterator2_.Init(1, 1);
        } else {
          ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter2);
          iterator1_.Init(1, axis);
          iterator2_.Init(axis, axis);
          *--output_shape = axis;
        }
      } else {  // Shape2 is a scalar
        ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter1);
        iterator1_.Init(axis, axis);
        iterator2_.Init(1, axis);
        *--output_shape = axis;
      }
      index++;  // Manually increment since we processed one axis
    } else {
      for (; index < dimension_count_min; index++) {
        ptrdiff_t axis1 = static_cast<ptrdiff_t>(*--iter1);
        ptrdiff_t axis2 = static_cast<ptrdiff_t>(*--iter2);

        ptrdiff_t largest = std::max<ptrdiff_t>(axis1, axis2);
        ptrdiff_t smallest = std::min<ptrdiff_t>(axis1, axis2);
        ptrdiff_t dim_to_use = largest;

        if (smallest == 0) {
          ORT_ENFORCE(largest <= 1, "Can broadcast 0 by 0 or 1. ", largest, " is invalid.");
          dim_to_use = smallest;
        }

        *--output_shape = dim_to_use;

        // if both 1, or a 1 and 0, and there are more dims, we can let the next iteration do the Init
        if (dim_to_use <= 1 && index + 1 < dimension_count_min)
          continue;

        iterator1_.Init(axis1, dim_to_use);
        iterator2_.Init(axis2, dim_to_use);
        index++;  // Manually increment since we processed one axis
        break;
      }
    }

    for (; index < dimension_count_min; index++) {
      ptrdiff_t axis1 = static_cast<ptrdiff_t>(*--iter1);
      ptrdiff_t axis2 = static_cast<ptrdiff_t>(*--iter2);

      ptrdiff_t largest = std::max(axis1, axis2);
      ptrdiff_t smallest = std::min(axis1, axis2);
      ptrdiff_t dim_to_use = largest;

      if (smallest == 0) {
        ORT_ENFORCE(largest <= 1, "Can broadcast 0 by 0 or 1. ", largest, " is invalid.");
        dim_to_use = smallest;
      }

      *--output_shape = dim_to_use;

      if (largest == 1)  // Nothing to do in this case
        continue;

      iterator1_.Append(axis1, dim_to_use);
      iterator2_.Append(axis2, dim_to_use);
    }

    // If one shape is bigger than another we need to broadcast the smaller onto the bigger from this point on
    for (; index < dimension_count_max; index++) {
      if (dimension_count_max == shape2.size()) {
        ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter2);
        iterator1_.Append(1, axis);
        iterator2_.Append(axis, axis);
        *--output_shape = axis;
      } else {
        ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter1);
        iterator1_.Append(axis, axis);
        iterator2_.Append(1, axis);
        *--output_shape = axis;
      }
    }

    // Allocate the counters
    iterator1_.AllocateCounters();
    iterator2_.AllocateCounters();
  }

  size_t GetSpanSize() const { return std::min(iterator1_.GetCountsFront(), iterator2_.GetCountsFront()); }

  BroadcastIterator iterator1_, iterator2_;
  TensorShapeVector output_shape_;
};

struct InputBroadcaster {
  InputBroadcaster(const Tensor& input0, const Tensor& input1)
      : input_tensor0_(input0),
        input_tensor1_(&input1),
        input_tensor1_shape_(input1.Shape()) {
  }

  InputBroadcaster(const Tensor& input0, const TensorShape& input1_shape)
      : input_tensor0_(input0),
        input_tensor1_(nullptr),
        input_tensor1_shape_(input1_shape) {
  }

  void AdvanceBy(size_t offset) {
    ORT_ENFORCE(offset % span_size_ == 0, "InputBroadcaster can only start at span boundary!");
    broadcaster_.iterator1_.AdvanceBy(offset);
    broadcaster_.iterator2_.AdvanceBy(offset);
  }

  TensorShape GetOutputShape() const { return TensorShape(broadcaster_.output_shape_); }
  size_t GetSpanSize() const { return span_size_; }

  // Check whether we have a tensor instance for input 1. Code using this class is required to validate this
  // before calling any methods that require input 1 to have data.
  bool HaveTwoTensors() const { return input_tensor1_ != nullptr; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.GetDeltasFront() == 0; }
  bool IsInput1Scalar() const { return broadcaster_.iterator2_.GetDeltasFront() == 0; }

  size_t Input0ElementSize() const { return input0_element_size_; }
  size_t Input1ElementSize() const { return input1_element_size_; }

  template <typename T>
  const T& Scalar0() { return *(static_cast<const T*>(input0_bytes_) + broadcaster_.iterator1_.Current()); }
  template <typename T>
  const T& Scalar1() { return *(static_cast<const T*>(input1_bytes_) + broadcaster_.iterator2_.Current()); }

  // general usage is to get a full span, but if we parallelize within a span we need intra-span pieces
  // which are specified via offset and num_elements
  template <typename T>
  ConstEigenVectorMap<T> Eigen0(size_t offset, size_t num_elements) {
    assert(offset < span_size_ && (offset + num_elements) <= span_size_);
    return ConstEigenVectorMap<T>(static_cast<const T*>(input0_bytes_) + broadcaster_.iterator1_.Current() + offset,
                                  num_elements);
  }
  template <typename T>
  ConstEigenVectorMap<T> Eigen1(size_t offset, size_t num_elements) {
    assert(offset < span_size_ && (offset + num_elements) <= span_size_);
    return ConstEigenVectorMap<T>(static_cast<const T*>(input1_bytes_) + broadcaster_.iterator2_.Current() + offset,
                                  num_elements);
  }

  template <typename T>
  gsl::span<const T> Span0(size_t offset, size_t num_elements) {
    return gsl::span<const T>(static_cast<const T*>(input0_bytes_) + broadcaster_.iterator1_.Current() + offset,
                              num_elements);
  }

  template <typename T>
  gsl::span<const T> Span1(size_t offset, size_t num_elements) {
    return gsl::span<const T>(static_cast<const T*>(input1_bytes_) + broadcaster_.iterator2_.Current() + offset,
                              num_elements);
  }

  void Next() {
    AdvanceBy(span_size_);
  }

 private:
  const Tensor& input_tensor0_;
  // need to support use case where input1 is just the shape for Expand op
  const Tensor* input_tensor1_{nullptr};
  const TensorShape& input_tensor1_shape_;
  const size_t input0_element_size_{input_tensor0_.DataType()->Size()};
  const size_t input1_element_size_{input_tensor1_ ? input_tensor1_->DataType()->Size() : 0};
  const void* input0_bytes_{input_tensor0_.DataRaw()};
  const void* input1_bytes_{input_tensor1_ ? input_tensor1_->DataRaw() : nullptr};

  Broadcaster broadcaster_{input_tensor0_.Shape().GetDims(), input_tensor1_shape_.GetDims()};
  size_t span_size_{broadcaster_.GetSpanSize()};
};

struct OutputBroadcaster {
  OutputBroadcaster(size_t span_size, Tensor& tensor, ptrdiff_t start_offset = 0, ptrdiff_t end_offset = 0)
      : element_size_(tensor.DataType()->Size()),
        span_size_(span_size) {
    ptrdiff_t len = narrow<ptrdiff_t>(tensor.Shape().Size());
    ptrdiff_t real_end = (end_offset <= 0) ? len : end_offset;
    if (start_offset != 0 || end_offset != 0) {  // Keep original semantic
      ORT_ENFORCE(start_offset >= 0 && real_end >= 0 && start_offset <= real_end && real_end <= len,
                  "Invalid start/ending offset [", start_offset, ",", real_end, ") for tensor of length:", len);
      ORT_ENFORCE(start_offset % span_size == 0 && real_end % span_size == 0,
                  "Broadcast Output range [", start_offset, ", ", real_end,
                  ") are not at boundary of span with size:", span_size);
    }

    output_elements_ = real_end - start_offset;
    output_bytes_ = static_cast<uint8_t*>(tensor.MutableDataRaw()) + (start_offset * element_size_);
    output_end_ = output_bytes_ + ((real_end - start_offset) * element_size_);
  }

  size_t OutputElementSize() const { return element_size_; }
  size_t NumOutputElements() const { return output_elements_; }

  operator bool() const {
    return output_bytes_ != output_end_;
  }

  template <typename T>
  EigenVectorMap<T> EigenOutput(size_t offset, size_t num_elements) {
    assert(offset < span_size_ && (offset + num_elements) <= span_size_);
    return EigenVectorMap<T>(reinterpret_cast<T*>(output_bytes_) + offset, num_elements);
  }

  template <typename T>
  gsl::span<T> SpanOutput(size_t offset, size_t num_elements) {
    assert(offset < span_size_ && (offset + num_elements) <= span_size_);
    return gsl::span<T>(reinterpret_cast<T*>(output_bytes_) + offset, num_elements);
  }

  void Next() {
    output_bytes_ += (span_size_ * element_size_);
  }

 private:
  const size_t element_size_;
  const size_t span_size_;
  size_t output_elements_;
  uint8_t* output_bytes_;
  const void* output_end_;
};

class BroadcastHelper {
 public:
  // general purpose ctor
  BroadcastHelper(InputBroadcaster& input_broadcaster,
                  OutputBroadcaster& output_broadcaster,
                  void* user_data = nullptr,
                  concurrency::ThreadPool* tp = nullptr,
                  double unit_cost = 0.0)
      : input_broadcaster_(input_broadcaster),
        output_broadcaster_(output_broadcaster),
        threadpool_(tp),
        unit_cost_(unit_cost),
        user_data_(user_data) {
  }

  // ctor for use when we parallelize within a span.
  BroadcastHelper(const BroadcastHelper& rhs, size_t offset, size_t num_elements)
      : input_broadcaster_(rhs.input_broadcaster_),
        output_broadcaster_(rhs.output_broadcaster_),
        input0_offset_(IsInput0Scalar() ? 0 : offset),
        input0_num_elements_(IsInput0Scalar() ? 1 : num_elements),
        input1_offset_(IsInput1Scalar() ? 0 : offset),
        input1_num_elements_(IsInput1Scalar() ? 1 : num_elements),
        output_offset_(offset),
        output_num_elements_(num_elements),
        user_data_(rhs.user_data_) {
  }

  // convenience accessors to simplify usage of this class. these will be optimized away in a release build

  bool HaveTwoTensorInputs() const { return input_broadcaster_.HaveTwoTensors(); }

  bool IsInput0Scalar() const { return input_broadcaster_.IsInput0Scalar(); }
  bool IsInput1Scalar() const { return input_broadcaster_.IsInput1Scalar(); }

  size_t Input0ElementSize() const { return input_broadcaster_.Input0ElementSize(); }
  size_t Input1ElementSize() const { return input_broadcaster_.Input1ElementSize(); }

  size_t OutputElementSize() const { return output_broadcaster_.OutputElementSize(); }
  size_t NumOutputElements() const { return output_broadcaster_.NumOutputElements(); }

  bool SingleSpanOutput() const { return input_broadcaster_.GetSpanSize() == output_broadcaster_.NumOutputElements(); }

  template <typename T>
  const T& ScalarInput0() { return input_broadcaster_.Scalar0<T>(); }

  template <typename T>
  const T& ScalarInput1() { return input_broadcaster_.Scalar1<T>(); }

  template <typename T>
  ConstEigenVectorMap<T> EigenInput0() { return input_broadcaster_.Eigen0<T>(input0_offset_, input0_num_elements_); }

  template <typename T>
  ConstEigenVectorMap<T> EigenInput1() { return input_broadcaster_.Eigen1<T>(input1_offset_, input1_num_elements_); }

  template <typename T>
  EigenVectorMap<T> OutputEigen() { return output_broadcaster_.EigenOutput<T>(output_offset_, output_num_elements_); }

  template <typename T>
  gsl::span<const T> SpanInput0() { return input_broadcaster_.Span0<T>(input0_offset_, input0_num_elements_); }

  template <typename T>
  gsl::span<const T> SpanInput1() { return input_broadcaster_.Span1<T>(input1_offset_, input1_num_elements_); }

  template <typename T>
  gsl::span<T> OutputSpan() { return output_broadcaster_.SpanOutput<T>(output_offset_, output_num_elements_); }

  void Next() {
    input_broadcaster_.Next();
    output_broadcaster_.Next();
  }

  bool NeedMoreOutput() const { return output_broadcaster_; }

  concurrency::ThreadPool* Threadpool() const { return threadpool_; }
  double UnitCost() const { return unit_cost_; }

  // user data is an opaque blob. there is no memory management provided by BroadcastHelper.
  // if the BroadcastHelper instance is copied during parallelization the pointer will be copied across
  void SetUserData(void* user_data) { user_data_ = user_data; }
  void* GetUserData() const { return user_data_; }

 private:
  InputBroadcaster& input_broadcaster_;
  OutputBroadcaster& output_broadcaster_;

  // info required if we parallelize within a span
  concurrency::ThreadPool* threadpool_{nullptr};
  double unit_cost_{0.0};
  size_t input0_offset_{0};
  size_t input0_num_elements_{input_broadcaster_.GetSpanSize()};  // default all to getting one full span
  size_t input1_offset_{0};
  size_t input1_num_elements_{input_broadcaster_.GetSpanSize()};
  size_t output_offset_{0};
  size_t output_num_elements_{input_broadcaster_.GetSpanSize()};

  // opaque user data that is passed through
  void* user_data_{nullptr};
};

// type agnostic functions to use in the low level broadcasting to process each span.
// type specific logic is applied within the functions.
// Raw function pointer is significantly cheaper in terms of binary size at the cost of no support for captures.
using ProcessSpanFunc = void (*)(BroadcastHelper&);
struct ProcessBroadcastSpanFuncs {
  ProcessSpanFunc input0scalar;
  ProcessSpanFunc input1scalar;
  ProcessSpanFunc general;
};

// Parallelize processing of data where all the output is covered by a single span
template <typename TBroadcastHelper>
static void ParallelizeSingleSpan(TBroadcastHelper& helper, const ProcessBroadcastSpanFuncs& functors) {
  TensorOpCost cost{static_cast<float>(std::max(helper.Input0ElementSize(), helper.Input1ElementSize())),
                    static_cast<float>(helper.OutputElementSize()),
                    helper.UnitCost()};

  if (helper.IsInput0Scalar()) {
    concurrency::ThreadPool::TryParallelFor(
        helper.Threadpool(), helper.NumOutputElements(), cost,
        [&helper, &functors](std::ptrdiff_t first, std::ptrdiff_t last) {
          size_t count = static_cast<size_t>(last - first);
          TBroadcastHelper segment_helper(helper, first, count);
          functors.input0scalar(segment_helper);
        });
  } else if (helper.IsInput1Scalar()) {
    concurrency::ThreadPool::TryParallelFor(
        helper.Threadpool(), helper.NumOutputElements(), cost,
        [&helper, &functors](std::ptrdiff_t first, std::ptrdiff_t last) {
          size_t count = static_cast<size_t>(last - first);
          TBroadcastHelper segment_helper(helper, first, count);
          functors.input1scalar(segment_helper);
        });

  } else {
    concurrency::ThreadPool::TryParallelFor(
        helper.Threadpool(), helper.NumOutputElements(), cost,
        [&helper, &functors](std::ptrdiff_t first, std::ptrdiff_t last) {
          size_t count = static_cast<size_t>(last - first);
          TBroadcastHelper segment_helper(helper, first, count);
          functors.general(segment_helper);
        });
  }
}

// Broadcast two inputs with no parallelization.
//
// This function is type agnostic, and uses function pointers instead of std::function, to minimize binary size.
// Type specific logic is plugged in via the functions in ProcessBroadcastSpanFuncs.
// Optional user_data can be provided, and will be available to the ProcessSpanFunc implementations
// via BroadcastHelper.GetUserData().
void UntypedBroadcastTwo(OpKernelContext& context, const ProcessBroadcastSpanFuncs& funcs, void* user_data = nullptr);

// Broadcast two inputs with parallelization.
//
// Operator usage is the same as the parallelization is opaque to the operator.
// unit_cost must be a valid cost value.
void UntypedBroadcastTwo(OpKernelContext& context, const ProcessBroadcastSpanFuncs& funcs, double unit_cost,
                         void* user_data = nullptr);

template <typename T>
class SwiGLU final : public OpKernel {
 public:
  explicit SwiGLU(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<float>("alpha", &alpha_, 1.0f);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
};

// Helper to provide the looping logic with optimization for parallelizing within a single span if the
// TBroadcastHelper instance was setup to enable that.
template <typename TBroadcastHelper>
void BroadcastLooper(TBroadcastHelper& helper, const ProcessBroadcastSpanFuncs& functors) {
  ORT_ENFORCE(helper.HaveTwoTensorInputs(), "BroadcastLooper requires two tensors as input.");

  bool par_available = concurrency::ThreadPool::ShouldParallelize(helper.Threadpool());
  if (par_available && helper.SingleSpanOutput()) {
    ParallelizeSingleSpan(helper, functors);
  } else {
    if (helper.IsInput0Scalar()) {
      while (helper.NeedMoreOutput()) {
        functors.input0scalar(helper);
        helper.Next();
      }
    } else if (helper.IsInput1Scalar()) {
      while (helper.NeedMoreOutput()) {
        functors.input1scalar(helper);
        helper.Next();
      }
    } else {
      while (helper.NeedMoreOutput()) {
        functors.general(helper);
        helper.Next();
      }
    }
  }
}

struct TensorAllocator {
  TensorAllocator(OpKernelContext& context) {
    auto status = context.GetTempSpaceAllocator(&allocator_);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  }

  template <typename T>
  std::unique_ptr<Tensor> Allocate(const TensorShape& shape) const {
    return std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                    shape,
                                    allocator_);
  }

 private:
  AllocatorPtr allocator_;
};
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
}  // namespace onnxruntime
