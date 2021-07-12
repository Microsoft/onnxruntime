/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modifications Copyright (c) Microsoft.

#pragma once

#include "onnxruntime_config.h"
// external/eigen/Eigen/src/Core/AssignEvaluator.h:86:63:
// error: enum constant in boolean context [-Werror=int-in-bool-context]

#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if __GNUC__ >= 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#ifdef HAS_DEPRECATED_COPY
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif
#endif
#elif defined(_MSC_VER)
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence

// unsupported\eigen\cxx11\src\Tensor\TensorUInt128.h(150,0): Warning C4245: 'initializing': conversion from '__int64'
// to 'uint64_t', signed/unsigned mismatch
#pragma warning(push)
#pragma warning(disable : 4554)
#pragma warning(disable : 4245)
#pragma warning(disable : 4127)
#endif
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#endif
namespace onnxruntime {

// common Eigen types that we will often use
template <typename T>
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenMatrixMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <class T>
using ConstSparseMatrixMap = Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor, int64_t>>;

template <typename T>
using ConstEigenArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using ConstEigenVectorMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using EigenMatrixMapRowMajorOuterStride =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<>>;

template <typename T>
using ConstEigenMatrixMapRowMajorOuterStride =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<>>;

template <typename T>
auto EigenMap(Tensor& t) -> EigenVectorMap<T> {
  return EigenVectorMap<T>(t.template MutableData<T>(), gsl::narrow<ptrdiff_t>(t.Shape().Size()));
}
template <typename T>
auto EigenMap(const Tensor& t) -> ConstEigenVectorMap<T> {
  return ConstEigenVectorMap<T>(t.template Data<T>(), gsl::narrow<ptrdiff_t>(t.Shape().Size()));
}

class CPUMathUtil {
 public:
  /*CPUMathUtil contains some help method like generate a
        random seed. We only need a single instance for it.*/
  static CPUMathUtil& Instance() {
    static CPUMathUtil p;
    return p;
  }
  // todo: the random generate interface.
 private:
  CPUMathUtil() = default;
};

// cast TA and TB to TC, and do matrix multiply in Eigen
// note that inputs/outputs is row-major, while Eigen is col-major
// so (M, K) x (K, N) -> (M, N) becomes (N, K) x (K, M) -> (N, M) in Eigen
template <typename TA, typename TB, typename TY>
void EigenCastGEMM(const TA* A_data, const TB* B_data, TY* Y_data, int M, int N, int K) {
  auto A = ConstEigenMatrixMap<TA>(A_data, K, M);
  auto B = ConstEigenMatrixMap<TB>(B_data, N, K);
  EigenMatrixMap<TY>(Y_data, N, M) = B.template cast<TY>() * A.template cast<TY>();
}

}  // namespace onnxruntime
