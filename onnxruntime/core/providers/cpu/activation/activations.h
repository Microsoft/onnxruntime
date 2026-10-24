// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>

#include "core/common/float16.h"
#include "core/common/common.h"
#include "core/platform/threadpool.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {

namespace functors {

template <typename T>
struct Celu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    // TODO: Tune the cost
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseMax(0.0f) + (((T)alpha * ((xm / (T)alpha).exp() - 1)).cwiseMin(0.0f));
  }
};

template <>
struct Celu<MLFloat16> : public ElementWiseRangedTransform<MLFloat16> {
  float alpha;
  Status Init(const onnxruntime::NodeAttributes& attributes) {
    return GetFloatParam("alpha", attributes, alpha);
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<MLFloat16>* Copy() const final {
    return new Celu<MLFloat16>(*this);
  }

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    const float typed_alpha = MLFloat16(alpha).ToFloat();
    for (auto i = first; i < last; ++i) {
      const float x = this->input[i].ToFloat();
      this->output[i] = std::isnan(x) ? this->input[i]
                                      : MLFloat16(std::max(0.0f, x) +
                                                  std::min(0.0f, typed_alpha * (std::exp(x / typed_alpha) - 1.0f)));
    }
  }
};

template <>
struct Celu<BFloat16> : public ElementWiseRangedTransform<BFloat16> {
  float alpha;
  Status Init(const onnxruntime::NodeAttributes& attributes) {
    return GetFloatParam("alpha", attributes, alpha);
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<BFloat16>* Copy() const final {
    return new Celu<BFloat16>(*this);
  }

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    const float typed_alpha = BFloat16(alpha).ToFloat();
    for (auto i = first; i < last; ++i) {
      const float x = this->input[i].ToFloat();
      this->output[i] = std::isnan(x) ? this->input[i]
                                      : BFloat16(std::max(0.0f, x) +
                                                 std::min(0.0f, typed_alpha * (std::exp(x / typed_alpha) - 1.0f)));
    }
  }
};

template <typename T>
struct Elu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 30.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * (xm.exp() - 1));
  }
};

template <typename T>
struct HardSigmoid : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 0.5f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (((T)alpha * xm + (T)beta).cwiseMin(1.0f)).cwiseMax(0.0f);
  }
};

template <typename T>
struct LeakyRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 25.0f;
  }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * xm);
  }
};

template <typename T>
struct Softplus : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 15.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > 0).select(xm + ((-xm).exp()).log1p(), ((xm).exp()).log1p());
  }
};

template <typename T>
struct Relu : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<T>* Copy() const {  // replace it with a macro. why this?
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;  // redundant?
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseMax(0);
  }
};

template <typename T>
struct Sigmoid : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 2.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
  }
};

template <>
void Sigmoid<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct Softsign : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    // Do not rewrite this as (1 + xm.abs()).inverse() * xm. Eigen's packet version of inverse() is
    // internal::preciprocal(), which is a reciprocal estimate instruction plus a Newton-Raphson step
    // rather than a real division on both x86 (rcp_ps, when built with FMA available) and ARM32
    // (NEON). Those instructions do not produce subnormal results, so for x = +/-FLT_MAX, where
    // 1 / (1 + |x|) ~= 2.94e-39 is subnormal, the estimate is 0 and Newton-Raphson cannot recover
    // from it. That yields +/-0 instead of +/-1. pdiv(), which this expression maps to, is a true
    // IEEE division on x86 and rescales its operands to avoid the same underflow on NEON.
    ym = xm / (1 + xm.abs());
  }
};

template <typename T>
struct Tanh : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  GSL_SUPPRESS(r.11)
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.tanh();
  }
};

template <>
void Tanh<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct ThresholdedRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > (T)alpha).select(xm, 0);
  }
};

template <typename T>
struct Selu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, gamma);

  float Cost() const final {
    return 4.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > 0).select((T)gamma * xm, (T)gamma * (T)alpha * (xm.exp() - 1.0f));
  }
};
}  // namespace functors

DEFINE_ELE_KERNEL(Celu);
DEFINE_ELE_KERNEL(Elu);
DEFINE_ELE_KERNEL(HardSigmoid);
DEFINE_ELE_KERNEL(LeakyRelu);
DEFINE_ELE_KERNEL(Softplus);
DEFINE_ELE_KERNEL(Relu);
DEFINE_ELE_KERNEL(Sigmoid);
DEFINE_ELE_KERNEL(Softsign);
DEFINE_ELE_KERNEL(Tanh);
DEFINE_ELE_KERNEL(ThresholdedRelu);
DEFINE_ELE_KERNEL(Selu);

}  // namespace onnxruntime
