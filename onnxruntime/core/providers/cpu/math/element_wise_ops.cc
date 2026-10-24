// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/common/narrow.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/math.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace onnxruntime {
// Supported types for operators that have type reduction enabled
namespace op_kernel_type_control {
// Max
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Max, 8, Input, 0, float, double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Max, 12, Input, 0,
                                        float, double, MLFloat16, int8_t, int32_t, uint32_t,
                                        int64_t, uint8_t, uint64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(kCpuExecutionProvider, kOnnxDomain, Max, 12, Input, 0,
                                         int32_t, int64_t);

// Min
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Min, 8, Input, 0, float, double);
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Min, 12, Input, 0,
                                        float, double, MLFloat16, int8_t, int32_t, uint32_t,
                                        int64_t, uint8_t, uint64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(kCpuExecutionProvider, kOnnxDomain, Min, 12, Input, 0,
                                         int32_t, int64_t);

// Mod
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain, Mod, Input, 0,
                                                   float, double, int64_t, uint64_t, int32_t, uint32_t,
                                                   int16_t, uint16_t, int8_t, uint8_t, MLFloat16);

// Pow
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Pow, 7, Input, 0, float, double);

// Pow 12 and later has separate Base and Exponent types.
// To reduce templatization we choose to support a subset of types for the base and exponent.
// This gives us 16 permutations.
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Pow, 12,
                                        Input, 0, int32_t, int64_t, float, double);
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(kCpuExecutionProvider, kOnnxDomain, Pow, 12,
                                        Input, 1, int32_t, int64_t, float, double);
}  // namespace op_kernel_type_control

//
// reduce the supported type lists to what's allowed in this build
//
using EnabledMax8Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain, Max, 8, Input, 0);
using EnabledMax12Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain, Max, 12, Input, 0);

using EnabledMin8Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain, Min, 8, Input, 0);
using EnabledMin12Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain, Min, 12, Input, 0);

using EnabledModTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Mod, Input, 0);

using EnabledPow7Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain, Pow, 7, Input, 0);
using EnabledPow12BaseTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain,
                                                                  Pow, 12, Input, 0);
using EnabledPow12ExpTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain,
                                                                 Pow, 12, Input, 1);

namespace functors {
template <>
void Exp<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  float* output_ptr = output + first;
  MlasComputeExp(input + first, output_ptr, static_cast<size_t>(len));
}
}  // namespace functors

#define REG_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      OP_TYPE,                                                                     \
      VERSION,                                                                     \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      OP_TYPE,                                                                       \
      VERSION,                                                                       \
      TYPE,                                                                          \
      KernelDefBuilder()                                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                           \
      OP_TYPE,                                                                                        \
      VERSION_FROM, VERSION_TO,                                                                       \
      TYPE,                                                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),                    \
      KERNEL_CLASS<TYPE>);

REG_ELEMENTWISE_TYPED_KERNEL(SwiGLU, 28, float, SwiGLU)
REG_ELEMENTWISE_TYPED_KERNEL(SwiGLU, 28, double, SwiGLU)

#define REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                                     \
      OP_TYPE,                                                                                                  \
      VERSION_FROM, VERSION_TO,                                                                                 \
      TYPE,                                                                                                     \
      KernelDefBuilder()                                                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                                             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                                           \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_KERNEL_NONT(OP_TYPE, VERSION, KERNEL_CLASS, CONSTRAINTS) \
  ONNX_CPU_OPERATOR_KERNEL(                                                      \
      OP_TYPE,                                                                   \
      VERSION,                                                                   \
      KernelDefBuilder()                                                         \
          .TypeConstraint("T", CONSTRAINTS),                                     \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, \
                                              CONSTRAINTS)                                     \
  ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                                          \
      OP_TYPE,                                                                                 \
      VERSION_FROM,                                                                            \
      VERSION_TO,                                                                              \
      KernelDefBuilder()                                                                       \
          .TypeConstraint("T", CONSTRAINTS),                                                   \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_KERNEL_NONT_2(OP_TYPE, VERSION, KERNEL_CLASS, \
                                      T1_CONSTRAINTS,                 \
                                      T2_CONSTRAINTS)                 \
  ONNX_CPU_OPERATOR_KERNEL(                                           \
      OP_TYPE,                                                        \
      VERSION,                                                        \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", T1_CONSTRAINTS)                        \
          .TypeConstraint("T1", T2_CONSTRAINTS),                      \
      KERNEL_CLASS);

#define REG_ELEMENTWISE_VERSIONED_KERNEL_NONT_2(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, \
                                                T1_CONSTRAINTS,                                  \
                                                T2_CONSTRAINTS)                                  \
  ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                                            \
      OP_TYPE,                                                                                   \
      VERSION_FROM,                                                                              \
      VERSION_TO,                                                                                \
      KernelDefBuilder()                                                                         \
          .TypeConstraint("T", T1_CONSTRAINTS)                                                   \
          .TypeConstraint("T1", T2_CONSTRAINTS),                                                 \
      KERNEL_CLASS);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, float, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, double, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, int32_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, int64_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, uint32_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, uint64_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, float, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, double, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, int32_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, int64_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, uint32_t, Add);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 13, 13, uint64_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, float, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, double, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, int8_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, int16_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, int32_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, int64_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, uint8_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, uint16_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, uint32_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, uint64_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 14, MLFloat16, Add);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, float, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, double, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, int32_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, int64_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, uint32_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, uint64_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, float, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, double, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, int32_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, int64_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, uint32_t, Sub);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 13, 13, uint64_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, float, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, double, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, int8_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, int16_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, int32_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, int64_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, uint8_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, uint16_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, uint32_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, uint64_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 14, MLFloat16, Sub);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, float, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, double, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, int32_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, int64_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, uint32_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, uint64_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, float, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, double, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, int32_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, int64_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, uint32_t, Mul);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 13, 13, uint64_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, float, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, double, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, int8_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, int16_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, int32_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, int64_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, uint8_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, uint16_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, uint32_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, uint64_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 14, MLFloat16, Mul);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, float, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, double, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, int32_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, int64_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, uint32_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, uint64_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, float, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, double, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, int32_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, int64_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, uint32_t, Div);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 13, 13, uint64_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, float, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, double, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, int8_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, int16_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, int32_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, int64_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, uint8_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, uint16_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, uint32_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, uint64_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 14, MLFloat16, Div);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, float, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, double, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, int8_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, int16_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, int32_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, int64_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, uint8_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, uint16_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, uint32_t, Abs);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, uint64_t, Abs);

REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, float, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, double, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, int8_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, int16_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, int32_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, int64_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, uint8_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, uint16_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, uint32_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, uint64_t, Abs);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, float, Neg);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, double, Neg);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, int8_t, Neg);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, int16_t, Neg);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, int32_t, Neg);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, int64_t, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, float, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, double, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, int8_t, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, int16_t, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, int32_t, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, int64_t, Neg);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Floor, 6, 12, float, Floor);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Floor, 6, 12, double, Floor);
REG_ELEMENTWISE_TYPED_KERNEL(Floor, 13, float, Floor);
REG_ELEMENTWISE_TYPED_KERNEL(Floor, 13, double, Floor);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Ceil, 6, 12, float, Ceil);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Ceil, 6, 12, double, Ceil);
REG_ELEMENTWISE_TYPED_KERNEL(Ceil, 13, float, Ceil);
REG_ELEMENTWISE_TYPED_KERNEL(Ceil, 13, double, Ceil);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Reciprocal, 6, 12, float, Reciprocal);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Reciprocal, 6, 12, double, Reciprocal);
REG_ELEMENTWISE_TYPED_KERNEL(Reciprocal, 13, float, Reciprocal);
REG_ELEMENTWISE_TYPED_KERNEL(Reciprocal, 13, double, Reciprocal);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sqrt, 6, 12, float, Sqrt);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sqrt, 6, 12, double, Sqrt);
REG_ELEMENTWISE_TYPED_KERNEL(Sqrt, 13, float, Sqrt);
REG_ELEMENTWISE_TYPED_KERNEL(Sqrt, 13, double, Sqrt);

REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Pow, 7, 11, Pow,
                                      BuildKernelDefConstraintsFromTypeList<EnabledPow7Types>());

REG_ELEMENTWISE_VERSIONED_KERNEL_NONT_2(Pow, 12, 12, Pow,
                                        BuildKernelDefConstraintsFromTypeList<EnabledPow12BaseTypes>(),
                                        BuildKernelDefConstraintsFromTypeList<EnabledPow12ExpTypes>());

REG_ELEMENTWISE_VERSIONED_KERNEL_NONT_2(Pow, 13, 14, Pow,
                                        BuildKernelDefConstraintsFromTypeList<EnabledPow12BaseTypes>(),
                                        BuildKernelDefConstraintsFromTypeList<EnabledPow12ExpTypes>());

REG_ELEMENTWISE_KERNEL_NONT_2(Pow, 15, Pow,
                              BuildKernelDefConstraintsFromTypeList<EnabledPow12BaseTypes>(),
                              BuildKernelDefConstraintsFromTypeList<EnabledPow12ExpTypes>());

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Exp, 6, 12, float, Exp);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Exp, 6, 12, double, Exp);
REG_ELEMENTWISE_TYPED_KERNEL(Exp, 13, float, Exp);
REG_ELEMENTWISE_TYPED_KERNEL(Exp, 13, double, Exp);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Log, 6, 12, float, Log);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Log, 6, 12, double, Log);
REG_ELEMENTWISE_TYPED_KERNEL(Log, 13, float, Log);
REG_ELEMENTWISE_TYPED_KERNEL(Log, 13, double, Log);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sum, 6, 7, float, Sum_6);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sum, 6, 7, double, Sum_6);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sum, 8, 12, float, Sum_8);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sum, 8, 12, double, Sum_8);
// Supposed to add BFloat16 but we are not supporting now, however, separate registration
REG_ELEMENTWISE_TYPED_KERNEL(Sum, 13, float, Sum_8);
REG_ELEMENTWISE_TYPED_KERNEL(Sum, 13, double, Sum_8);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Max, 6, 7, float, Max_6);

REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Max, 8, 11, Max_8, BuildKernelDefConstraintsFromTypeList<EnabledMax8Types>());
REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Max, 12, 12, Max_8, BuildKernelDefConstraintsFromTypeList<EnabledMax12Types>());
// Supposed to add BFloat16 but we are not supporting now, however, separate registration
REG_ELEMENTWISE_KERNEL_NONT(Max, 13, Max_8, BuildKernelDefConstraintsFromTypeList<EnabledMax12Types>());

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Min, 6, 7, float, Min_6);
REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Min, 8, 11, Min_8, BuildKernelDefConstraintsFromTypeList<EnabledMin8Types>());
REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Min, 12, 12, Min_8, BuildKernelDefConstraintsFromTypeList<EnabledMin12Types>());
// Supposed to add BFloat16 but we are not supporting now, however, separate registration
REG_ELEMENTWISE_KERNEL_NONT(Min, 13, Min_8, BuildKernelDefConstraintsFromTypeList<EnabledMin12Types>());

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 7, 8, float, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 7, 8, double, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, float, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, double, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, int8_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, int16_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, int32_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, int64_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, uint8_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, uint16_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, uint32_t, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 9, 12, uint64_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, float, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, double, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, int8_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, int16_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, int32_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, int64_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, uint8_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, uint16_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, uint32_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 13, uint64_t, Less);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 7, 8, float, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 7, 8, double, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, float, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, double, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, int8_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, int16_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, int32_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, int64_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, uint8_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, uint16_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, uint32_t, Greater);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 9, 12, uint64_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, float, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, double, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int8_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int16_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int32_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int64_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, uint8_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, uint16_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, uint32_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, uint64_t, Greater);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, float, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, double, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, int8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, int16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, uint8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, uint16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, uint32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, uint64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, float, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 11, 12, double, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, uint8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, uint16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, uint32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, uint64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, float, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, double, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, int8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, int16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, uint8_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, uint16_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, uint32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, uint64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, float, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, double, Equal);
using string = std::string;
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 19, string, Equal);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, float, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, double, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, int8_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, int16_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, int32_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, int64_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, uint8_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, uint16_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, uint32_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(LessOrEqual, 12, 15, uint64_t, LessOrEqual);

// Opset-16 adds BFloat16 to allowed types for the LessOrEqual operator
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, float, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, double, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, int8_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, int16_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, int32_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, int64_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, uint8_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, uint16_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, uint32_t, LessOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(LessOrEqual, 16, uint64_t, LessOrEqual);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, float, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, double, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, int8_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, int16_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, int32_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, int64_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, uint8_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, uint16_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, uint32_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(GreaterOrEqual, 12, 15, uint64_t, GreaterOrEqual);

// Opset-16 adds BFloat16 to allowed types for the GreaterOrEqual operator
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, float, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, double, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, int8_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, int16_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, int32_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, int64_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, uint8_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, uint16_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, uint32_t, GreaterOrEqual);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(GreaterOrEqual, 16, uint64_t, GreaterOrEqual);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mean, 6, 7, float, Mean_6);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mean, 8, 12, float, Mean_8);
// Supposed to add BFloat16 but we are not supporting now, however, separate registration
REG_ELEMENTWISE_TYPED_KERNEL(Mean, 13, float, Mean_8);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(BitShift, 11, 27, uint8_t, BitShift);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(BitShift, 11, 27, uint16_t, BitShift);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(BitShift, 11, 27, uint32_t, BitShift);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(BitShift, 11, 27, uint64_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, int8_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, int16_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, int32_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, int64_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, uint8_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, uint16_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, uint32_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 28, uint64_t, BitShift);

REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, int8_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, int16_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, int32_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, int64_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, uint8_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, uint16_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, uint32_t, BitwiseAnd);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseAnd, 18, uint64_t, BitwiseAnd);

REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, int8_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, int16_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, int32_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, int64_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, uint8_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, uint16_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, uint32_t, BitwiseNot);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseNot, 18, uint64_t, BitwiseNot);

REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, int8_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, int16_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, int32_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, int64_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, uint8_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, uint16_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, uint32_t, BitwiseOr);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseOr, 18, uint64_t, BitwiseOr);

REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, int8_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, int16_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, int32_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, int64_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, uint8_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, uint16_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, uint32_t, BitwiseXor);
REG_ELEMENTWISE_TYPED_KERNEL(BitwiseXor, 18, uint64_t, BitwiseXor);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Erf, 9, 12, float, Erf);
// Supposed to add BFloat16 but we are not supporting now, however, separate registration
REG_ELEMENTWISE_TYPED_KERNEL(Erf, 13, float, Erf);
REG_ELEMENTWISE_TYPED_KERNEL(Erf, 13, MLFloat16, Erf);

// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Not, 1, bool, Not);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(And, 7, bool, And);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Or, 7, bool, Or);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Xor, 7, bool, Xor);

ONNX_CPU_OPERATOR_KERNEL(
    Not,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Not);

ONNX_CPU_OPERATOR_KERNEL(
    And,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    And);

ONNX_CPU_OPERATOR_KERNEL(
    Or,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Or);

ONNX_CPU_OPERATOR_KERNEL(
    Xor,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Xor);

using AllocateTensorFunc = std::unique_ptr<Tensor> (*)(const TensorAllocator& tensor_allocator,
                                                       const TensorShape& shape);

static void UntypedBroadcastVariadic(int input_count, OpKernelContext& context,
                                     AllocateTensorFunc allocate_tensor,
                                     const ProcessBroadcastSpanFuncs& funcs);

template <typename T>
Status Add<T>::Compute(OpKernelContext* context) const {
  // BroadcastHelper received as argument may differ from 'helper' when parallelizing within a span
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() + per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() + per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>() + per_iter_bh.EigenInput1<T>();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0f);
  return Status::OK();
}

template <>
Status Add<MLFloat16>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.ScalarInput0<Eigen::half>() + per_iter_bh.EigenInput1<Eigen::half>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().array() + per_iter_bh.ScalarInput1<Eigen::half>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>() + per_iter_bh.EigenInput1<Eigen::half>();
      }};
  UntypedBroadcastTwo(*context, funcs, 1.0f);
  return Status::OK();
}

template <typename T>
Status Sub<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() - per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() - per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>() - per_iter_bh.EigenInput1<T>();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <>
Status Sub<MLFloat16>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.ScalarInput0<Eigen::half>() - per_iter_bh.EigenInput1<Eigen::half>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().array() - per_iter_bh.ScalarInput1<Eigen::half>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>() - per_iter_bh.EigenInput1<Eigen::half>();
      }};
  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status Mul<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() * per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() * per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().cwiseProduct(per_iter_bh.EigenInput1<T>());
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <>
Status Mul<MLFloat16>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.ScalarInput0<Eigen::half>() * per_iter_bh.EigenInput1<Eigen::half>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().array() * per_iter_bh.ScalarInput1<Eigen::half>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().cwiseProduct(per_iter_bh.EigenInput1<Eigen::half>());
      }};
  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status Div<T>::Compute(OpKernelContext* context) const {
  // Integer division by zero is undefined behavior in C++ and causes a hardware exception.
  // Check for zeros in the divisor before performing the division.
  // Skip the check if the divisor was already validated as a constant initializer during kernel creation.
  if constexpr (std::is_integral<T>::value) {
    if (!divisor_is_validated_constant_) {
      const Tensor& B = *context->Input<Tensor>(1);
      const T* b_data = B.Data<T>();
      const int64_t b_size = B.Shape().Size();
      for (int64_t i = 0; i < b_size; ++i) {
        ORT_RETURN_IF(b_data[i] == T{0}, "Integer division by zero");
      }
    }
  }

  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() / per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() / per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().cwiseQuotient(per_iter_bh.EigenInput1<T>());
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}
template <>
Status Div<MLFloat16>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.ScalarInput0<Eigen::half>() / per_iter_bh.EigenInput1<Eigen::half>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().array() / per_iter_bh.ScalarInput1<Eigen::half>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<Eigen::half>() = per_iter_bh.EigenInput0<Eigen::half>().cwiseQuotient(per_iter_bh.EigenInput1<Eigen::half>());
      }};
  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

namespace pow_internal {

template <typename T, typename E>
void PowImpl(OpKernelContext& context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<E>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](E y) {
                         return static_cast<T>(std::pow(X, y));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const E Y = per_iter_bh.ScalarInput1<E>();
        auto output = per_iter_bh.OutputSpan<T>();

        // optimize for X^2 and X^3
        if (Y == 2) {
          std::transform(X.begin(), X.end(), output.begin(),
                         [](T x) {
                           return static_cast<T>(x * x);
                         });

        } else if (Y == 3) {
          std::transform(X.begin(), X.end(), output.begin(),
                         [](T x) {
                           return static_cast<T>(x * x * x);
                         });
        } else {
          std::transform(X.begin(), X.end(), output.begin(),
                         [Y](T x) {
                           return static_cast<T>(std::pow(x, Y));
                         });
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<E>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](T x, E y) {
                         return static_cast<T>(std::pow(x, y));
                       });
      }};

  UntypedBroadcastTwo(context, funcs, 1.0);
}

template <typename B>
Status DispatchOnBase(OpKernelContext& context, const Tensor& Y) {
  namespace on = ONNX_NAMESPACE;
  Status s;
  switch (Y.GetElementType()) {
    case on::TensorProto_DataType_INT32:
      PowImpl<B, int32_t>(context);
      break;
    case on::TensorProto_DataType_INT64:
      PowImpl<B, int64_t>(context);
      break;
    case on::TensorProto_DataType_FLOAT:
      PowImpl<B, float>(context);
      break;
    case on::TensorProto_DataType_DOUBLE:
      PowImpl<B, double>(context);
      break;
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported Y type: ",
                          DataTypeImpl::ToString(Y.DataType()));
  }
  return s;
}

}  // namespace pow_internal

Status
Pow::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const Tensor& Y = *context->Input<Tensor>(1);

  namespace on = ONNX_NAMESPACE;
  using namespace pow_internal;

  Status s;
  // Switch on base type first
  switch (X.GetElementType()) {
    case on::TensorProto_DataType_INT32:
      s = DispatchOnBase<int32_t>(*context, Y);
      break;
    case on::TensorProto_DataType_INT64:
      s = DispatchOnBase<int64_t>(*context, Y);
      break;
    case on::TensorProto_DataType_FLOAT:
      s = DispatchOnBase<float>(*context, Y);
      break;
    case on::TensorProto_DataType_DOUBLE:
      s = DispatchOnBase<double>(*context, Y);
      break;
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported X type: ",
                          DataTypeImpl::ToString(X.DataType()));
  }
  return s;
}

template <typename T>
Status Sum_6<T>::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();
  ORT_ENFORCE(input_count >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<T>(*ctx->Output(0, shape));

  if (input_count == 1) {
    sum = EigenMap<T>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ORT_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<T>(data_0) + EigenMap<T>(data_1);
    for (int index = 2; index < input_count; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      sum += EigenMap<T>(data_n);
    }
  }

  return Status::OK();
}

template <typename T>
Status Sum_8<T>::Compute(OpKernelContext* context) const {
  const auto typed_allocator = [](const TensorAllocator& tensor_allocator, const TensorShape& shape) {
    return tensor_allocator.Allocate<T>(shape);
  };

  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() =
            per_iter_bh.ScalarInput0<T>() + per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() =
            per_iter_bh.EigenInput0<T>().array() + per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() =
            per_iter_bh.EigenInput0<T>() + per_iter_bh.EigenInput1<T>();
      }};

  int input_count = Node().InputArgCount().front();
  UntypedBroadcastVariadic(input_count, *context, typed_allocator, funcs);

  return Status::OK();
}

template <>
Status Min_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto min = EigenMap<float>(*ctx->Output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    min = min.array().template min<Eigen::PropagateNaN>(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <typename T>
struct Min_8::ComputeImpl {
  Status operator()(const Min_8& inst, OpKernelContext* context) const {
    const auto typed_allocator = [](const TensorAllocator& tensor_allocator, const TensorShape& shape) {
      return tensor_allocator.Allocate<T>(shape);
    };

    ProcessBroadcastSpanFuncs funcs{
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput1<T>().array().template min<Eigen::PropagateNaN>(per_iter_bh.ScalarInput0<T>());
        },
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput0<T>().array().template min<Eigen::PropagateNaN>(per_iter_bh.ScalarInput1<T>());
        },
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput0<T>().array().template min<Eigen::PropagateNaN>(
                  per_iter_bh.EigenInput1<T>().array());
        }};

    int input_count = inst.Node().InputArgCount().front();
    UntypedBroadcastVariadic(input_count, *context, typed_allocator, funcs);

    return Status::OK();
  }
};

template <bool is_min>
static Status MinMaxMLFloat16(const OpKernel& inst, OpKernelContext* context) {
  const auto typed_allocator = [](const TensorAllocator& tensor_allocator, const TensorShape& shape) {
    return tensor_allocator.Allocate<MLFloat16>(shape);
  };

  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        auto num_elements = per_iter_bh.EigenInput1<MLFloat16>().rows();

        const auto* input_1 = reinterpret_cast<const Eigen::half*>(per_iter_bh.EigenInput1<MLFloat16>().data());
        ConstEigenVectorArrayMap<Eigen::half> input_1_vec_map(input_1, num_elements);

        auto* output = reinterpret_cast<Eigen::half*>(per_iter_bh.OutputEigen<MLFloat16>().data());
        EigenVectorArrayMap<Eigen::half> output_vec_map(output, num_elements);

        if (is_min) {
          output_vec_map = input_1_vec_map.template min<Eigen::PropagateNaN>(
              static_cast<Eigen::half>(per_iter_bh.ScalarInput0<MLFloat16>()));
        } else {
          output_vec_map = input_1_vec_map.template max<Eigen::PropagateNaN>(
              static_cast<Eigen::half>(per_iter_bh.ScalarInput0<MLFloat16>()));
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        auto num_elements = per_iter_bh.EigenInput0<MLFloat16>().rows();

        const auto* input_0 = reinterpret_cast<const Eigen::half*>(per_iter_bh.EigenInput0<MLFloat16>().data());
        ConstEigenVectorArrayMap<Eigen::half> input_0_vec_map(input_0, num_elements);

        auto* output = reinterpret_cast<Eigen::half*>(per_iter_bh.OutputEigen<MLFloat16>().data());
        EigenVectorArrayMap<Eigen::half> output_vec_map(output, num_elements);

        if (is_min) {
          output_vec_map = input_0_vec_map.template min<Eigen::PropagateNaN>(
              static_cast<Eigen::half>(per_iter_bh.ScalarInput1<MLFloat16>()));
        } else {
          output_vec_map = input_0_vec_map.template max<Eigen::PropagateNaN>(
              static_cast<Eigen::half>(per_iter_bh.ScalarInput1<MLFloat16>()));
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        auto num_elements = per_iter_bh.EigenInput0<MLFloat16>().rows();

        const auto* input_0 = reinterpret_cast<const Eigen::half*>(per_iter_bh.EigenInput0<MLFloat16>().data());
        ConstEigenVectorArrayMap<Eigen::half> input_0_vec_map(input_0, num_elements);

        const auto* input_1 = reinterpret_cast<const Eigen::half*>(per_iter_bh.EigenInput1<MLFloat16>().data());
        ConstEigenVectorArrayMap<Eigen::half> input_1_vec_map(input_1, num_elements);

        auto* output = reinterpret_cast<Eigen::half*>(per_iter_bh.OutputEigen<MLFloat16>().data());
        EigenVectorArrayMap<Eigen::half> output_vec_map(output, num_elements);

        if (is_min) {
          output_vec_map = input_0_vec_map.template min<Eigen::PropagateNaN>(input_1_vec_map);
        } else {
          output_vec_map = input_0_vec_map.template max<Eigen::PropagateNaN>(input_1_vec_map);
        }
      }};

  int input_count = inst.Node().InputArgCount().front();
  UntypedBroadcastVariadic(input_count, *context, typed_allocator, funcs);

  return Status::OK();
}

Status Min_8::Compute(OpKernelContext* context) const {
  auto dt_type = context->Input<Tensor>(0)->GetElementType();

  switch (dt_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return MinMaxMLFloat16<true>(*this, context);
      break;
    default:
      utils::MLTypeCallDispatcher<float, double, int8_t, int32_t, uint32_t,
                                  int64_t, uint8_t, uint64_t>
          t_disp(dt_type);
      return t_disp.InvokeRet<Status, ComputeImpl>(*this, context);
  }
}

template <>
Status Max_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto max = EigenMap<float>(*ctx->Output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    max = max.array().template max<Eigen::PropagateNaN>(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <typename T>
struct Max_8::ComputeImpl {
  Status operator()(const Max_8& inst, OpKernelContext* context) const {
    const auto typed_allocator = [](const TensorAllocator& tensor_allocator, const TensorShape& shape) {
      return tensor_allocator.Allocate<T>(shape);
    };

    ProcessBroadcastSpanFuncs funcs{
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput1<T>().array().template max<Eigen::PropagateNaN>(per_iter_bh.ScalarInput0<T>());
        },
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput0<T>().array().template max<Eigen::PropagateNaN>(per_iter_bh.ScalarInput1<T>());
        },
        [](BroadcastHelper& per_iter_bh) {
          per_iter_bh.OutputEigen<T>() =
              per_iter_bh.EigenInput0<T>().array().template max<Eigen::PropagateNaN>(
                  per_iter_bh.EigenInput1<T>().array());
        }};

    int input_count = inst.Node().InputArgCount().front();
    // TODO: Parallelize across spans in UntypedBroadcastVariadic to avoid specific logic here
    if (input_count == 2) {
      UntypedBroadcastTwo(*context, funcs, 1.0);
    } else {
      UntypedBroadcastVariadic(input_count, *context, typed_allocator, funcs);
    }

    return Status::OK();
  }
};

Status Max_8::Compute(OpKernelContext* context) const {
  auto dt_type = context->Input<Tensor>(0)->GetElementType();

  switch (dt_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return MinMaxMLFloat16<false>(*this, context);
      break;
    default:
      utils::MLTypeCallDispatcher<float, double, int8_t, int32_t, uint32_t,
                                  int64_t, uint8_t, uint64_t>
          t_disp(dt_type);
      return t_disp.InvokeRet<Status, ComputeImpl>(*this, context);
  }
}

Status Not::Compute(OpKernelContext* context) const {
  auto& input = *context->Input<Tensor>(0);
  auto& output = *context->Output(0, input.Shape());

  EigenMap<bool>(output).array() = !EigenMap<bool>(input).array();
  return Status::OK();
}

Status And::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X && true = X' and 'X && false = false'
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        bool input0 = per_iter_bh.ScalarInput0<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input0)
          output = per_iter_bh.EigenInput1<bool>();
        else
          output.array() = false;
      },
      [](BroadcastHelper& per_iter_bh) {
        bool input1 = per_iter_bh.ScalarInput1<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input1)
          output = per_iter_bh.EigenInput0<bool>();
        else
          output.array() = false;
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<bool>().array() && per_iter_bh.EigenInput1<bool>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

Status Or::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X || true = true' and 'X || false = X'
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        bool input0 = per_iter_bh.ScalarInput0<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input0)
          output.array() = true;
        else
          output = per_iter_bh.EigenInput1<bool>();
      },
      [](BroadcastHelper& per_iter_bh) {
        bool input1 = per_iter_bh.ScalarInput1<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input1)
          output.array() = true;
        else
          output = per_iter_bh.EigenInput0<bool>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<bool>().array() || per_iter_bh.EigenInput1<bool>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

Status Xor::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X ^ true = !X' and 'X ^ false = X'
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        bool input0 = per_iter_bh.ScalarInput0<bool>();
        auto input1 = per_iter_bh.EigenInput1<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input0)
          output.array() = !input1.array();
        else
          output = input1;
      },
      [](BroadcastHelper& per_iter_bh) {
        auto input0 = per_iter_bh.EigenInput0<bool>();
        bool input1 = per_iter_bh.ScalarInput1<bool>();
        auto output = per_iter_bh.OutputEigen<bool>();
        if (input1)
          output.array() = !input0.array();
        else
          output = input0;
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<bool>().array() != per_iter_bh.EigenInput1<bool>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status Equal<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.ScalarInput0<T>() == per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() == per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<T>().array() == per_iter_bh.EigenInput1<T>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status Less<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput1<T>().array() > per_iter_bh.ScalarInput0<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() < per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() < per_iter_bh.EigenInput1<T>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status Greater<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput1<T>().array() < per_iter_bh.ScalarInput0<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() > per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<T>().array() > per_iter_bh.EigenInput1<T>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status LessOrEqual<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput1<T>().array() >= per_iter_bh.ScalarInput0<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() <= per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() <= per_iter_bh.EigenInput1<T>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <typename T>
Status GreaterOrEqual<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput1<T>().array() <= per_iter_bh.ScalarInput0<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() = per_iter_bh.EigenInput0<T>().array() >= per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<bool>() =
            per_iter_bh.EigenInput0<T>().array() >= per_iter_bh.EigenInput1<T>().array();
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

template <>
Status Mean_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto mean = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    mean = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ORT_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    mean = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      mean += EigenMap<float>(data_n);
    }
  }

  // Take the mean
  float weight = 1.0f / static_cast<float>(inputCount);
  mean = mean * weight;

  return Status::OK();
}

template <>
Status Mean_8<float>::Compute(OpKernelContext* context) const {
  const auto typed_allocator = [](const TensorAllocator& tensor_allocator, const TensorShape& shape) {
    return tensor_allocator.Allocate<float>(shape);
  };

  // Do a sum exactly the same as in Sum_8
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() =
            per_iter_bh.ScalarInput0<float>() + per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() =
            per_iter_bh.EigenInput0<float>().array() + per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() =
            per_iter_bh.EigenInput0<float>() + per_iter_bh.EigenInput1<float>();
      }};

  int input_count = Node().InputArgCount().front();
  UntypedBroadcastVariadic(input_count, *context, typed_allocator, funcs);

  // Now divide by the input count to get the mean
  EigenMap<float>(*context->Output<Tensor>(0)) *= 1.0f / static_cast<float>(input_count);

  return Status::OK();
}

template <typename T>
BitShift<T>::BitShift(const OpKernelInfo& info) : OpKernel(info) {
  std::string direction;
  auto status = info.GetAttr("direction", &direction);
  ORT_ENFORCE(status.IsOK(), status);

  if (direction == "LEFT")
    shift_left_ = true;
  else if (direction == "RIGHT")
    shift_left_ = false;
  else
    ORT_THROW("Invalid direction value of '", direction, "'. Valid values are 'LEFT' or 'RIGHT'.");
}

template <typename T>
T FromUnsignedBits(std::make_unsigned_t<T> bits) {
  static_assert(std::is_integral_v<T>);
  T value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
inline bool IsInvalidShift(T shift) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr UnsignedT bit_width = std::numeric_limits<UnsignedT>::digits;
  return (std::is_signed_v<T> && shift < 0) || static_cast<UnsignedT>(shift) >= bit_width;
}

template <typename T>
inline T SafeShiftLeft(T value, T shift) {
  if (IsInvalidShift(shift)) {
    return T{0};
  }

  using UnsignedT = std::make_unsigned_t<T>;
  const auto result = static_cast<UnsignedT>(
      static_cast<UnsignedT>(value) << static_cast<UnsignedT>(shift));
  return FromUnsignedBits<T>(result);
}

template <typename T>
inline T SafeShiftRight(T value, T shift) {
  if (IsInvalidShift(shift)) {
    if constexpr (std::is_signed_v<T>) {
      return value < 0 ? T{-1} : T{0};
    } else {
      return T{0};
    }
  }

  using UnsignedT = std::make_unsigned_t<T>;
  constexpr UnsignedT bit_width = std::numeric_limits<UnsignedT>::digits;
  const auto shift_amount = static_cast<UnsignedT>(shift);
  auto result = static_cast<UnsignedT>(static_cast<UnsignedT>(value) >> shift_amount);
  if constexpr (std::is_signed_v<T>) {
    if (value < 0 && shift_amount != 0) {
      result |= std::numeric_limits<UnsignedT>::max() << (bit_width - shift_amount);
    }
  }

  return FromUnsignedBits<T>(result);
}

template <typename T>
Status BitShift<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        bool shift_left = (per_iter_bh.GetUserData() != nullptr);
        const T& input0 = per_iter_bh.ScalarInput0<T>();
        ConstEigenVectorMap<T> input1 = per_iter_bh.EigenInput1<T>();
        EigenVectorMap<T> output = per_iter_bh.OutputEigen<T>();
        ptrdiff_t i = 0;
        if (shift_left) {
          for (const auto& input : input1.array()) {
            output[i++] = SafeShiftLeft(input0, input);
          }
        } else {
          for (const auto& input : input1.array()) {
            output[i++] = SafeShiftRight(input0, input);
          }
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        bool shift_left = (per_iter_bh.GetUserData() != nullptr);
        auto input0 = per_iter_bh.EigenInput0<T>();
        const T& input1 = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        ptrdiff_t i = 0;
        if (shift_left) {
          for (const auto& input : input0.array()) {
            output[i++] = SafeShiftLeft(input, input1);
          }
        } else {
          for (const auto& input : input0.array()) {
            output[i++] = SafeShiftRight(input, input1);
          }
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        bool shift_left = (per_iter_bh.GetUserData() != nullptr);
        auto input0 = per_iter_bh.EigenInput0<T>();
        auto input1 = per_iter_bh.EigenInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();

        auto cur0 = input0.begin(), end0 = input0.end();
        auto cur1 = input1.begin(), end1 = input1.end();
        auto cur_out = output.begin(), end_out = output.end();
        if (shift_left) {
          for (; cur0 != end0; ++cur0, ++cur1, ++cur_out) {
            *cur_out = SafeShiftLeft(*cur0, *cur1);
          }
        } else {
          for (; cur0 != end0; ++cur0, ++cur1, ++cur_out) {
            *cur_out = SafeShiftRight(*cur0, *cur1);
          }
        }

        ORT_ENFORCE(cur1 == end1);
        ORT_ENFORCE(cur_out == end_out);
      }};

  // set void* to value of bool (doesn't need to be address of) so it can be passed through to the lambdas via
  // BroadcastHelper::GetUserData. This is required as we use raw function pointers for the functors to reduce
  // the binary size, and doing so prevents using any captures in the lambdas.
  void* user_data = reinterpret_cast<void*>(shift_left_);

  UntypedBroadcastTwo(*context, funcs, 1.0, user_data);
  return Status::OK();
}

template <typename T>
Status BitwiseAnd<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return std::bit_and<T>()(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return static_cast<T>(std::bit_and<T>()(x, Y));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(), std::bit_and<T>());
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0f);
  return Status::OK();
}

template <typename T>
Status BitwiseNot<T>::Compute(OpKernelContext* context) const {
  auto& input = *context->Input<Tensor>(0);
  auto& output = *context->Output(0, input.Shape());

  std::transform(EigenMap<T>(input).array().begin(), EigenMap<T>(input).array().end(), EigenMap<T>(output).array().begin(), std::bit_not<T>());

  return Status::OK();
}

template <typename T>
Status BitwiseOr<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return std::bit_or<T>()(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return static_cast<T>(std::bit_or<T>()(x, Y));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(), std::bit_or<T>());
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0f);
  return Status::OK();
}

template <typename T>
Status BitwiseXor<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return std::bit_xor<T>()(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return static_cast<T>(std::bit_xor<T>()(x, Y));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(), std::bit_xor<T>());
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0f);
  return Status::OK();
}

template <typename T>
class Sin final : public OpKernel {
 public:
  Sin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<T>(Y) = MakeEigenArrayMap<T>(X).sin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Sin,
    7, 21,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Sin,
    7, 21,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Sin<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sin,
    22,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sin,
    22,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Sin<double>);

template <typename T>
class Cos final : public OpKernel {
 public:
  Cos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<T>(Y) = MakeEigenArrayMap<T>(X).cos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Cos,
    7, 21,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cos<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Cos,
    7, 21,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Cos<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Cos,
    22,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cos<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Cos,
    22,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Cos<double>);

template <typename T>
class Tan final : public OpKernel {
 public:
  Tan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).tan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Tan,
    7, 21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tan<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Tan,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tan<float>);

template <typename T>
class Asin final : public OpKernel {
 public:
  Asin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).asin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Asin,
    7,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asin<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Asin,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asin<float>);

template <typename T>
class Acos final : public OpKernel {
 public:
  Acos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).acos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Acos,
    7,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acos<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Acos,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acos<float>);

template <typename T>
class Atan final : public OpKernel {
 public:
  Atan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).atan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Atan,
    7,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atan<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Atan,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atan<float>);

template <typename T>
class Sinh final : public OpKernel {
 public:
  explicit Sinh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).sinh();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Sinh,
    9,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sinh<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Sinh,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sinh<float>);

template <typename T>
class Cosh final : public OpKernel {
 public:
  explicit Cosh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).cosh();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cosh,
    9,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cosh<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Cosh,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cosh<float>);

template <typename T>
class Asinh final : public OpKernel {
 public:
  explicit Asinh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.Data<float>();
    auto Y_data = Y.MutableData<float>();

    auto in = gsl::make_span(X_data, narrow<ptrdiff_t>(X.Shape().Size()));
    auto out = gsl::make_span(Y_data, narrow<ptrdiff_t>(Y.Shape().Size()));

    for (size_t index = 0; index < in.size(); ++index) {
      out[index] = std::asinh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Asinh);
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Asinh,
    9,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asinh<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Asinh,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asinh<float>);

template <typename T>
class Acosh final : public OpKernel {
 public:
  explicit Acosh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.Data<float>();
    auto Y_data = Y.MutableData<float>();

    auto in = gsl::make_span(X_data, narrow<ptrdiff_t>(X.Shape().Size()));
    auto out = gsl::make_span(Y_data, narrow<ptrdiff_t>(Y.Shape().Size()));

    for (size_t index = 0; index < in.size(); ++index) {
      out[index] = std::acosh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Acosh);
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Acosh,
    9,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acosh<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Acosh,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acosh<float>);

template <typename T>
class Atanh final : public OpKernel {
 public:
  explicit Atanh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.Data<float>();
    auto Y_data = Y.MutableData<float>();

    auto in = gsl::make_span(X_data, narrow<ptrdiff_t>(X.Shape().Size()));
    auto out = gsl::make_span(Y_data, narrow<ptrdiff_t>(Y.Shape().Size()));

    for (size_t index = 0; index < in.size(); ++index) {
      out[index] = std::atanh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Atanh);
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Atanh,
    9,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atanh<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Atanh,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atanh<float>);

template <>
Status PRelu<float>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        float input0 = per_iter_bh.ScalarInput0<float>();
        if (input0 >= 0)
          per_iter_bh.OutputEigen<float>().array() = input0;
        else
          per_iter_bh.OutputEigen<float>() = input0 * per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        const float* input0 = per_iter_bh.EigenInput0<float>().data();
        const float input1 = per_iter_bh.ScalarInput1<float>();
        float* output = per_iter_bh.OutputEigen<float>().data();
        size_t size = per_iter_bh.OutputEigen<float>().size();
        for (size_t i = 0; i < size; i++) {
          output[i] = (input0[i] >= 0) ? input0[i] : input0[i] * input1;
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        const float* input0 = per_iter_bh.EigenInput0<float>().data();
        const float* input1 = per_iter_bh.EigenInput1<float>().data();
        float* output = per_iter_bh.OutputEigen<float>().data();
        size_t size = per_iter_bh.OutputEigen<float>().size();
        for (size_t i = 0; i < size; i++) {
          output[i] = (input0[i] >= 0) ? input0[i] : input0[i] * input1[i];
        }
      }};

  UntypedBroadcastTwo(*context, funcs, 1.0);
  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    PRelu,
    7,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    PRelu,
    9,
    15,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

// Opset-16 adds BFloat16 to allowed types for the PRelu operator
ONNX_CPU_OPERATOR_KERNEL(
    PRelu,
    16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

static void ExpandBroadcastLooper(BroadcastHelper& helper, const ProcessBroadcastSpanFuncs& functors) {
  ORT_ENFORCE(!helper.HaveTwoTensorInputs(), "ExpandBroadcastLooper should only have a shape for the second input.");

  if (helper.IsInput0Scalar()) {
    while (helper.NeedMoreOutput()) {
      functors.input0scalar(helper);
      helper.Next();
    }
    /*
  } else if (helper.IsInput0Scalar()) {
    // not possible as we only have one tensor as input
  */
  } else {
    while (helper.NeedMoreOutput()) {
      functors.general(helper);
      helper.Next();
    }
  }
}

// Split out the untyped processing from the type specific work to minimize binary size
static void UntypedExpand(OpKernelContext& context, const ProcessBroadcastSpanFuncs& funcs) {
  // Input 1 is a 1-dimensional tensor containing the dimension values to exapnd to
  const auto& shape_data_tensor = *context.Input<Tensor>(1);
  ORT_ENFORCE(shape_data_tensor.Shape().GetDims().size() == 1,
              "Tensor with shape information must be 1 dimensional.");

  // Turn the shape tensor data into an actual shape
  const auto* p_dims = shape_data_tensor.Data<int64_t>();
  TensorShape shape(std::vector<int64_t>{p_dims, p_dims + shape_data_tensor.Shape().Size()});

  InputBroadcaster input_broadcaster(*context.Input<Tensor>(0), shape);
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape()));
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

  ExpandBroadcastLooper(broadcast_helper, funcs);
}

template <typename T>
Status Expand_8<T>::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>().array() = per_iter_bh.ScalarInput0<T>();
      },
      [](BroadcastHelper&) {
        ORT_THROW("Invalid usage. Input 1 is a shape with no data.");
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>();
      }};

  UntypedExpand(*context, funcs);
  return Status::OK();
}

template <typename T>
Status SwiGLU<T>::Compute(OpKernelContext* context) const {
  const auto* gate = context->Input<Tensor>(0);
  const auto* value = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(gate->Shape() == value->Shape(), "SwiGLU inputs must have identical shapes.");

  auto* output = context->Output(0, gate->Shape());
  const auto* gate_data = gate->Data<T>();
  const auto* value_data = value->Data<T>();
  auto* output_data = output->MutableData<T>();
  const auto alpha = static_cast<T>(alpha_);

  const auto element_count = onnxruntime::narrow<std::ptrdiff_t>(gate->Shape().Size());
  constexpr std::ptrdiff_t length_per_task = 4096;
  const auto task_count = (element_count + length_per_task - 1) / length_per_task;
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), task_count,
      [&](std::ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const auto end = std::min(start + length_per_task, element_count);
        const auto count = onnxruntime::narrow<size_t>(end - start);
        if constexpr (std::is_same_v<T, float>) {
          if (alpha_ == 1.0f) {
            MlasComputeSilu(gate_data + start, output_data + start, count);
            MlasEltwiseMul(output_data + start, value_data + start, output_data + start, count);
            return;
          }
        }

        for (auto i = start; i < end; ++i) {
          const auto a = gate_data[i];
          output_data[i] = a / (static_cast<T>(1) + std::exp(-alpha * a)) * value_data[i];
        }
      },
      0);

  return Status::OK();
}

#define REG_EXPAND_KERNEL_WITH_TYPE_NAME(TYPE, TYPE_NAME)                          \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                        \
      Expand,                                                                      \
      8,                                                                           \
      12,                                                                          \
      TYPE_NAME,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Expand_8<TYPE>);                                                             \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Expand,                                                                      \
      13,                                                                          \
      TYPE_NAME,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Expand_8<TYPE>);

#define REG_EXPAND_KERNEL(TYPE) REG_EXPAND_KERNEL_WITH_TYPE_NAME(TYPE, TYPE)

REG_EXPAND_KERNEL_WITH_TYPE_NAME(std::string, string)

template <>
Status Erf<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  auto* Y = context->Output(0, x_shape);
  const float* input_data = X->Data<float>();
  float* output_data = Y->MutableData<float>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  const std::ptrdiff_t elem_count = onnxruntime::narrow<std::ptrdiff_t>(X->Shape().Size());
  constexpr std::ptrdiff_t length_per_task = 4096;  // this number comes from FastGelu.
  const std::ptrdiff_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  concurrency::ThreadPool::TryBatchParallelFor(
      tp, task_count,
      [&](ptrdiff_t task_idx) {
        const std::ptrdiff_t start = task_idx * length_per_task;
        const float* p_input = input_data + start;
        float* p_output = output_data + start;
        const std::ptrdiff_t count = std::min(length_per_task, elem_count - start);
        MlasComputeErf(p_input, p_output, static_cast<size_t>(count));
      },
      0);

  return Status::OK();
}

template <>
Status Erf<MLFloat16>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  auto* Y = context->Output(0, x_shape);
  const auto* input_data = X->Data<MLFloat16>();
  auto* output_data = Y->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = X->Shape().Size();
  constexpr int64_t length_per_task = 4096;
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  const auto narrow_task_count = onnxruntime::narrow<std::ptrdiff_t>(task_count);

  // get allocator for temporary buffers
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  concurrency::ThreadPool::TryBatchParallelFor(
      tp, narrow_task_count,
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const int64_t count = std::min(length_per_task, elem_count - start);
        const MLFloat16* p_input = input_data + start;
        MLFloat16* p_output = output_data + start;
        // allocate temp buffers for fp32 input and output
        IAllocatorUniquePtr<float> input_tmp_fp32 = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(count));
        IAllocatorUniquePtr<float> output_tmp_fp32 = IAllocator::MakeUniquePtr<float>(alloc, static_cast<size_t>(count));
        MlasComputeFP16Erf(p_input, p_output, input_tmp_fp32.get(), output_tmp_fp32.get(), static_cast<size_t>(count));
      },
      0);

  return Status::OK();
}

class Mod final : public OpKernel {
 public:
  Mod(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  bool fmod_{false};
  bool supports_float_floor_mod_{false};
  bool divisor_is_validated_constant_{false};
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Mod,
    10,
    12,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledModTypes>()),
    Mod);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Mod,
    13,
    27,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledModTypes>()),
    Mod);

ONNX_CPU_OPERATOR_KERNEL(
    Mod,
    28,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledModTypes>()),
    Mod);

namespace mod_internal {

template <class T>
inline T TruncatingRemainder(T x, T y) {
  if constexpr (std::is_signed_v<T>) {
    if (x == std::numeric_limits<T>::min() && y == T{-1}) {
      return T{0};
    }
  }

  return x % y;
}

template <class T>
inline T FMod(T x, T y) {
  if constexpr (std::is_integral_v<T>) {
    return TruncatingRemainder(x, y);
  } else {
    return static_cast<T>(std::fmod(x, y));
  }
}

template <class T>
void BroadCastFMod(OpKernelContext* context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T& X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return FMod(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T& Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return FMod(x, Y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](T x, T y) {
                         return FMod(x, y);
                       });
      }};

  UntypedBroadcastTwo(*context, funcs);
}

template <class T>
inline T Modulus(T x, T y) {
  auto res = TruncatingRemainder(x, y);
  if ((res < 0 && y > 0) || (res > 0 && y < 0)) {
    res += y;
  }
  return res;
}

template <class T>
void BroadCastMod(OpKernelContext* context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T& X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return Modulus(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T& Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return Modulus(x, Y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](T x, T y) {
                         return Modulus(x, y);
                       });
      }};

  UntypedBroadcastTwo(*context, funcs);
}

template <class T>
inline T FloorMod(T x, T y) {
  auto res = std::fmod(x, y);
  if (res == T{0}) {
    return std::copysign(T{0}, y);
  }

  if ((res < T{0} && y > T{0}) || (res > T{0} && y < T{0})) {
    res += y;
  }
  return res;
}

template <class T>
void BroadCastFloorMod(OpKernelContext* context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const T& X = per_iter_bh.ScalarInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X](T y) {
                         return FloorMod(X, y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        const T& Y = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y](T x) {
                         return FloorMod(x, Y);
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<T>();
        auto Y = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](T x, T y) {
                         return FloorMod(x, y);
                       });
      }};

  UntypedBroadcastTwo(*context, funcs);
}

void BroadCastMLFloat16FMod(OpKernelContext* context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const auto X = per_iter_bh.ScalarInput0<MLFloat16>();
        auto Y = per_iter_bh.SpanInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X_fl = math::halfToFloat(X.val)](const MLFloat16& y) {
                         return MLFloat16(std::fmod(X_fl, y.ToFloat()));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<MLFloat16>();
        const MLFloat16 Y = per_iter_bh.ScalarInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y_fl = math::halfToFloat(Y.val)](const MLFloat16& x) {
                         return MLFloat16(std::fmod(x.ToFloat(), Y_fl));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<MLFloat16>();
        auto Y = per_iter_bh.SpanInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](const MLFloat16& x, const MLFloat16& y) {
                         auto x_fl = x.ToFloat();
                         auto y_fl = y.ToFloat();
                         return MLFloat16(std::fmod(x_fl, y_fl));
                       });
      }};

  UntypedBroadcastTwo(*context, funcs);
}

void BroadCastMLFloat16FloorMod(OpKernelContext* context) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        const auto X = per_iter_bh.ScalarInput0<MLFloat16>();
        auto Y = per_iter_bh.SpanInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(Y.begin(), Y.end(), output.begin(),
                       [X_fl = X.ToFloat()](const MLFloat16& y) {
                         return MLFloat16(FloorMod(X_fl, y.ToFloat()));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<MLFloat16>();
        const MLFloat16 Y = per_iter_bh.ScalarInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(X.begin(), X.end(), output.begin(),
                       [Y_fl = Y.ToFloat()](const MLFloat16& x) {
                         return MLFloat16(FloorMod(x.ToFloat(), Y_fl));
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X = per_iter_bh.SpanInput0<MLFloat16>();
        auto Y = per_iter_bh.SpanInput1<MLFloat16>();
        auto output = per_iter_bh.OutputSpan<MLFloat16>();

        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](const MLFloat16& x, const MLFloat16& y) {
                         return MLFloat16(FloorMod(x.ToFloat(), y.ToFloat()));
                       });
      }};

  UntypedBroadcastTwo(*context, funcs);
}

template <class T, typename Enable = void>
struct CallModImpl;

// Check for zero values in the divisor tensor for integral types.
template <class T>
struct CheckZeroDivisorImpl {
  Status operator()(const Tensor& B) const {
    if constexpr (std::is_integral<T>::value) {
      const T* b_data = B.Data<T>();
      const int64_t b_size = B.Shape().Size();
      for (int64_t i = 0; i < b_size; ++i) {
        ORT_RETURN_IF(b_data[i] == T{0}, "Integer modulo by zero");
      }
    }
    return Status::OK();
  }
};

// Generic implementation of Mod kernel, non-floating point types
template <class T>
struct CallModImpl<T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(bool fmod, bool /*supports_float_floor_mod*/, OpKernelContext* ctx) const {
    if (fmod) {
      BroadCastFMod<T>(ctx);
    } else {
      BroadCastMod<T>(ctx);
    }
  }
};

// Generic implementation of Mod kernel, floating point types
template <class T>
struct CallModImpl<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type> {
  void operator()(bool fmod, bool supports_float_floor_mod, OpKernelContext* ctx) const {
    ORT_ENFORCE(fmod || supports_float_floor_mod,
                "fmod attribute must be true for floating point types before opset 28");
    if (fmod) {
      BroadCastFMod<T>(ctx);
    } else {
      BroadCastFloorMod<T>(ctx);
    }
  }
};

// MLFloat16 implementation of Mod kernel
template <>
struct CallModImpl<MLFloat16> {
  void operator()(bool fmod, bool supports_float_floor_mod, OpKernelContext* ctx) const {
    ORT_ENFORCE(fmod || supports_float_floor_mod,
                "fmod attribute must be true for floating point types before opset 28");
    if (fmod) {
      BroadCastMLFloat16FMod(ctx);
    } else {
      BroadCastMLFloat16FloorMod(ctx);
    }
  }
};

}  // namespace mod_internal

Mod::Mod(const OpKernelInfo& info) : OpKernel(info) {
  supports_float_floor_mod_ = info.node().SinceVersion() >= 28;
  int64_t fmod = 0;
  Status s = info.GetAttr<int64_t>("fmod", &fmod);
  if (s.IsOK()) {
    ORT_ENFORCE((fmod == 0) || (fmod == 1), "fmod must have value either 0 or 1");
    fmod_ = (fmod == 1);
  }

  // If the divisor is a constant initializer, validate for integer modulo by zero once
  // during kernel creation instead of on every Compute call.
  const Tensor* constant_divisor = nullptr;
  if (info.TryGetConstantInput(1, &constant_divisor)) {
    const auto dt_type = constant_divisor->GetElementType();
    utils::MLTypeCallDispatcherFromTypeList<EnabledModTypes> check_disp(dt_type);
    Status check_status = check_disp.InvokeRet<Status, mod_internal::CheckZeroDivisorImpl>(*constant_divisor);
    ORT_THROW_IF_ERROR(check_status);
    divisor_is_validated_constant_ = true;
  }
}

Status Mod::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto dt_type = X.GetElementType();

  // Integer modulo by zero is undefined behavior in C++ and causes a hardware exception.
  // Check for zeros in the divisor before performing the mod.
  // Skip the check if the divisor was already validated as a constant initializer during kernel creation.
  if (!divisor_is_validated_constant_) {
    const Tensor& B = *context->Input<Tensor>(1);
    utils::MLTypeCallDispatcherFromTypeList<EnabledModTypes> check_disp(dt_type);
    Status check_status = check_disp.InvokeRet<Status, mod_internal::CheckZeroDivisorImpl>(B);
    ORT_RETURN_IF_ERROR(check_status);
  }

  utils::MLTypeCallDispatcherFromTypeList<EnabledModTypes> t_disp(dt_type);
  t_disp.Invoke<mod_internal::CallModImpl>(fmod_, supports_float_floor_mod_, context);

  return Status::OK();
}

// Broadcast two inputs with no parallelization.
//
// This function is type agnostic, and uses function pointers instead of std::function, to minimize binary size.
// Type specific logic is plugged in via the functions in ProcessBroadcastSpanFuncs.
// Optional user_data can be provided, and will be available to the ProcessSpanFunc implementations
// via BroadcastHelper.GetUserData().
void UntypedBroadcastTwo(OpKernelContext& context, const ProcessBroadcastSpanFuncs& funcs, void* user_data) {
  InputBroadcaster input_broadcaster(*context.Input<Tensor>(0), *context.Input<Tensor>(1));
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape()));
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster, user_data);

  BroadcastLooper(broadcast_helper, funcs);
}

// Variant of UntypedBroadcastTwo that will parallelize.
// Operator usage is the same as the parallelization is opaque to the operator.
// unit_cost must be a valid cost value.
void UntypedBroadcastTwo(OpKernelContext& context, const ProcessBroadcastSpanFuncs& funcs, double unit_cost,
                         void* user_data) {
  const Tensor& input0_tensor = *context.Input<Tensor>(0);
  const Tensor& input1_tensor = *context.Input<Tensor>(1);
  InputBroadcaster input_broadcaster(input0_tensor, input1_tensor);

  Tensor& output_tensor = *context.Output(0, input_broadcaster.GetOutputShape());

  size_t span_size = input_broadcaster.GetSpanSize();
  size_t output_size = static_cast<ptrdiff_t>(output_tensor.Shape().Size());

  // one or more zero dimensions so nothing more to do
  if (output_size == 0) {
    return;
  }

  concurrency::ThreadPool* tp = context.GetOperatorThreadPool();

  if (span_size == output_size) {  // Input data will be processed in a single span, so parallelize within the span
    OutputBroadcaster output_broadcaster(span_size, output_tensor);
    BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster, user_data, tp, unit_cost);
    BroadcastLooper(broadcast_helper, funcs);
  } else {
    // Input data will be processed in multiple spans, so parallelize across spans.

    // enforce const on input broadcaster we copy from
    const InputBroadcaster& const_input_broadcaster = input_broadcaster;

    concurrency::ThreadPool::TryParallelFor(
        tp, output_size / span_size,
        TensorOpCost{static_cast<double>(input_broadcaster.Input0ElementSize()) * span_size,
                     static_cast<double>(output_tensor.DataType()->Size()) * span_size,
                     unit_cost * span_size},
        [span_size, &const_input_broadcaster, &output_tensor, &funcs, user_data](std::ptrdiff_t first_span,
                                                                                 std::ptrdiff_t last_span) {
          // copy original input_broadcaster (which is at start of all input) and advance to this segment
          InputBroadcaster segment_input_broadcaster(const_input_broadcaster);
          segment_input_broadcaster.AdvanceBy(first_span * span_size);

          // create broadcaster for this segment of output
          OutputBroadcaster segment_output_broadcaster(span_size, output_tensor,
                                                       first_span * span_size, last_span * span_size);

          BroadcastHelper segment_helper(segment_input_broadcaster, segment_output_broadcaster, user_data);
          BroadcastLooper(segment_helper, funcs);
        });
  }
}

// allocate_tensor should allocate a tensor of the output type with the given shape
static void UntypedBroadcastVariadic(int input_count, OpKernelContext& context,
                                     AllocateTensorFunc allocate_tensor,
                                     const ProcessBroadcastSpanFuncs& funcs) {
  const auto& input0 = *context.Input<Tensor>(0);

  // One item is trivial, just copy and exit
  if (input_count == 1) {
    auto& output = *context.Output(0, input0.Shape());
    CopyCpuTensor(&input0, &output);
    return;
  }

  TensorAllocator tensor_allocator(context);
  std::unique_ptr<Tensor> temp_input;
  std::unique_ptr<Tensor> temp_output;

  // For more than 2 tensors, we combine the the current two inputs into a temporary tensor,
  // and combine the next input with that
  for (int i = 0; i < input_count - 1; i++) {
    auto& tensor0 = temp_input ? *temp_input : input0;
    auto& tensor1 = *context.Input<Tensor>(i + 1);

    InputBroadcaster input_broadcaster(tensor0, tensor1);

    // Create a temporary output for all but the last iteration, which goes to the real output
    Tensor* p_output = nullptr;
    if (i == input_count - 2) {
      p_output = context.Output(0, input_broadcaster.GetOutputShape());
    } else {
      temp_output = allocate_tensor(tensor_allocator, input_broadcaster.GetOutputShape());
      p_output = temp_output.get();
    }

    OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(), *p_output);
    BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

    BroadcastLooper(broadcast_helper, funcs);

    temp_input = std::move(temp_output);
  }
}

}  // namespace onnxruntime
