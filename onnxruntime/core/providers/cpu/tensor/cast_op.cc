// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/type_list.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_types.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/int2.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"

#include "Eigen/src/Core/arch/Default/BFloat16.h"
#include "Eigen/src/Core/arch/Default/Half.h"

#include "core/mlas/inc/mlas.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {

namespace {
// Define a type list that extends AllIRv10 with INT2 types, but without Float4.
// Float4E2M1x2 doesn't support all the casting operations that other types do,
// so we don't include it here for the Cast operator.
using AllIRv10WithInt2Base =
    boost::mp11::mp_push_back<
        element_type_lists::AllIRv10,
        UInt2x4,
        Int2x4>;

#if !defined(DISABLE_FLOAT8_TYPES)
// Float8E8M0 was added in opset 24 (IR v12), so include it in the full type list
// but use the base list (without Float8E8M0) for pre-opset-24 kernel registrations.
using AllIRv10WithInt2 =
    boost::mp11::mp_push_back<AllIRv10WithInt2Base, Float8E8M0>;
#else
using AllIRv10WithInt2 = AllIRv10WithInt2Base;
#endif

using AllIRv10WithInt2AndFloat6 =
    boost::mp11::mp_push_back<AllIRv10WithInt2, Float6E2M3, Float6E3M2>;
}  // namespace

namespace op_kernel_type_control {
// Type list for all opsets of Cast (includes Float8E8M0 and Float6 types for runtime dispatch).
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    AllIRv10WithInt2AndFloat6);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    bool, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    AllIRv10WithInt2AndFloat6);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    bool, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledSrcTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Input, 0);
using EnabledDstTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Output, 0);

// Pre-opset-24 type lists exclude types introduced after opset 23.
#if !defined(DISABLE_FLOAT8_TYPES)
using EnabledSrcTypesPreOpset24 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledSrcTypes, Float8E8M0>, Float6E2M3>,
                           Float6E3M2>;
using EnabledDstTypesPreOpset24 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledDstTypes, Float8E8M0>, Float6E2M3>,
                           Float6E3M2>;
#else
using EnabledSrcTypesPreOpset24 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledSrcTypes, Float6E2M3>, Float6E3M2>;
using EnabledDstTypesPreOpset24 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledDstTypes, Float6E2M3>, Float6E3M2>;
#endif

using EnabledSrcTypesPreOpset28 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledSrcTypes, Float6E2M3>, Float6E3M2>;
using EnabledDstTypesPreOpset28 =
    boost::mp11::mp_remove<boost::mp11::mp_remove<EnabledDstTypes, Float6E2M3>, Float6E3M2>;

template <typename T>
using IsOrtFloat16Type = boost::mp11::mp_contains<TypeList<BFloat16, MLFloat16>, T>;

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename T>
using IsOrtFloat8Type = boost::mp11::mp_contains<element_type_lists::AllFloat8, T>;
#else
template <typename T>
struct IsOrtFloat8Type : std::false_type {};
#endif

template <typename T>
using IsOrtFloat6Type = boost::mp11::mp_contains<TypeList<Float6E2M3, Float6E3M2>, T>;

template <typename T>
using IsOrtInt4Type = boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>;

template <typename T>
struct IsStandardIntegerType {
  static constexpr bool value =
      std::is_same_v<T, int8_t> ||
      std::is_same_v<T, uint8_t> ||
      std::is_same_v<T, int16_t> ||
      std::is_same_v<T, uint16_t> ||
      std::is_same_v<T, int32_t> ||
      std::is_same_v<T, uint32_t> ||
      std::is_same_v<T, int64_t> ||
      std::is_same_v<T, uint64_t>;
};

// Types that Int4x2 and UInt4x2 convert to and from, apart from string.
template <typename T>
struct IsOrtInt4NumericConversionType {
  static constexpr bool value =
      std::is_same_v<T, bool> ||
      IsStandardIntegerType<T>::value ||
      std::is_floating_point_v<T> ||
      IsOrtFloat16Type<T>::value ||
      IsOrtFloat8Type<T>::value ||
      IsOrtFloat6Type<T>::value;
};

template <typename T>
struct IsOrtInt4ConversionType {
  static constexpr bool value = IsOrtInt4NumericConversionType<T>::value || std::is_same_v<std::string, T>;
};

// INT2 type support helpers
template <typename T>
using IsOrtInt2Type = boost::mp11::mp_contains<TypeList<Int2x4, UInt2x4>, T>;

// Types that Int2x4 and UInt2x4 convert to and from, apart from string.
template <typename T>
struct IsOrtInt2NumericConversionType {
  static constexpr bool value =
      std::is_same_v<T, bool> ||
      IsStandardIntegerType<T>::value ||
      std::is_floating_point_v<T> ||
      IsOrtFloat16Type<T>::value ||
      IsOrtFloat8Type<T>::value ||
      IsOrtFloat6Type<T>::value;
};

template <typename T>
struct IsOrtInt2ConversionType {
  static constexpr bool value = IsOrtInt2NumericConversionType<T>::value || std::is_same_v<std::string, T>;
};

// string cast helpers
// Note: when C++17 is available, use <charconv> functions

// handle floating point output separately
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  static_assert(sizeof(SrcType) <= sizeof(double),
                "largest supported floating point type is double");
  if (std::isnan(input)) {
    output = "NaN";
  } else if (std::isinf(input)) {
    if (input < std::numeric_limits<SrcType>::lowest()) {
      output = "-INF";
    } else {
      output = "INF";
    }
  } else {
    // set precision to 8 to match numpy default behavior
    constexpr const char* format = "%.8g";
    const double value = static_cast<double>(input);

    char static_buffer[256];
    std::unique_ptr<char[]> dynamic_buffer{};

    gsl::span<char> buffer_span = gsl::make_span(static_buffer);

    auto snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
    ORT_ENFORCE(snprintf_result > 0, "snprintf() failed with return value: ", snprintf_result);

    // include trailing '\0'
    const size_t required_buffer_size = gsl::narrow_cast<size_t>(snprintf_result) + 1;

    if (required_buffer_size > buffer_span.size()) {
      // didn't get it all, allocate a bigger buffer and retry
      dynamic_buffer = std::make_unique<char[]>(required_buffer_size);
      buffer_span = gsl::make_span(dynamic_buffer.get(), required_buffer_size);
      snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
      ORT_ENFORCE(
          snprintf_result > 0 &&
              gsl::narrow_cast<size_t>(snprintf_result) == buffer_span.size() - 1,
          "Failed to write value with snprintf().");
    }

    output.assign(buffer_span.data(), required_buffer_size - 1);
  }
}

template <typename SrcType>
typename std::enable_if<std::is_integral<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  output = std::to_string(input);
}

template <typename SrcType>
typename std::enable_if<IsOrtFloat16Type<SrcType>::value || IsOrtFloat8Type<SrcType>::value ||
                            IsOrtFloat6Type<SrcType>::value,
                        void>::type
CastToString(const SrcType& input, std::string& output) {
  CastToString(static_cast<float>(input), output);
}

template <typename DstType>
typename std::enable_if<std::is_floating_point<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(double),
                "largest supported floating point type is double");
  output = gsl::narrow_cast<DstType>(std::stod(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_unsigned<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(unsigned long long),
                "largest supported unsigned integral type is unsigned long long");
  output = gsl::narrow_cast<DstType>(std::stoull(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_signed<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(long long),
                "largest supported signed integral type is long long");
  output = gsl::narrow_cast<DstType>(std::stoll(input));
}

template <typename DstType>
typename std::enable_if<IsOrtFloat16Type<DstType>::value || IsOrtFloat8Type<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  float intermediate;
  CastFromString(input, intermediate);
  output = DstType(intermediate);
}

template <typename DstType>
typename std::enable_if<IsOrtFloat6Type<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  double intermediate;
  CastFromString(input, intermediate);
  output = DstType(intermediate);
}

// type that is usable with Eigen cast
template <typename T>
struct EigenCastType {
  using type = T;
};

// ORT float16 types don't support Eigen cast, so map them to Eigen ones

template <>
struct EigenCastType<MLFloat16> {
  using type = Eigen::half;
};

template <>
struct EigenCastType<BFloat16> {
  using type = Eigen::bfloat16;
};

// Helper for converting (U)Int4x2 values to any destination type.
template <typename SrcType, typename DstType,
          typename Enable = std::enable_if_t<IsOrtInt4Type<SrcType>::value && IsOrtInt4ConversionType<DstType>::value>>
struct FromInt4Converter {
  // The input 'val' can be either an int8_t value coming from Int4x2.GetElem(pos),
  // or an uint8_t value coming from UInt4x2.GetElem(pos), where pos can be 0 or 1.
  static DstType Convert(typename SrcType::UnpackedType val) {
    if constexpr (IsOrtFloat16Type<DstType>::value) {
      return DstType(static_cast<float>(val));
    } else if constexpr (IsOrtFloat8Type<DstType>::value) {
      return DstType(static_cast<float>(val), true);
    } else if constexpr (IsOrtFloat6Type<DstType>::value) {
      return DstType(static_cast<float>(val));
    } else if constexpr (std::is_same_v<bool, DstType>) {
      return val != 0;
    } else if constexpr (std::is_same_v<std::string, DstType>) {
      // val has type (u)int8_t, so static_cast<int> is required in order for std::to_string
      // to interpret val as a number (65 -> "65"), instead of a char (65 -> "A").
      return std::to_string(static_cast<int>(val));
    } else {
      return static_cast<DstType>(val);
    }
  }
};

// Helper for converting any source type to (U)Int4x2::UnpackedType values (int8_t and uint8_t).
template <typename SrcType, typename DstType,
          typename Enable = std::enable_if_t<IsOrtInt4ConversionType<SrcType>::value && IsOrtInt4Type<DstType>::value>>
struct ToInt4Converter {
  static typename DstType::UnpackedType Convert(const SrcType& val);
};

// See https://onnx.ai/onnx/operators/onnx__Cast.html#summary for casting from
// fixed point to fixed point: when OOR, discard higher bits and reinterpret
// (with respect to two's complement representation for signed types).
// The following example is listed: 200 (int16) converts to -56 (int8).
// For our int4 conversion, 200 (int16) would convert to -8 (int4).
template <typename SrcType>
struct ToInt4Converter<SrcType, Int4x2,
                       std::enable_if_t<IsStandardIntegerType<SrcType>::value>> {
  static int8_t Convert(const SrcType& val) {
    // Example: int8_t(14) converts to int4 (-2)
    //   int8_t(14) is 0000_1110
    //   truncate: 0000_1110 & 0000_1111 = 0000_1110
    //   in 4-bit two's complement, 1110 = 1 * -8 + 1 * 4 + 1 * 2 + 1 * 0 = -2
    //   sign-extend: -2 in int8_t is 1111_0000 | 0000_1110 = 1111_1110

    // Truncate to 4 least significant bits
    uint8_t truncated = static_cast<uint8_t>(val) & 0x0F;

    // Sign-extend: if bit 3 is set, it's negative in 4-bit two's complement,
    // so set the 4 most significant bits to 1.
    return static_cast<int8_t>((truncated & 0x8) ? (truncated | 0xF0) : truncated);
  }
};

// See https://onnx.ai/onnx/operators/onnx__Cast.html#summary for casting from
// fixed point to fixed point: when OOR, discard higher bits and reinterpret
// (with respect to two's complement representation for signed types).

template <typename SrcType>
struct ToInt4Converter<SrcType, UInt4x2,
                       std::enable_if_t<IsStandardIntegerType<SrcType>::value>> {
  static uint8_t Convert(const SrcType& val) {
    // Truncate to 4 least significant bits
    return static_cast<uint8_t>(val) & 0x0F;
  }
};

// bool -> (U)Int4x2
template <typename DstType>
struct ToInt4Converter<bool, DstType,
                       std::enable_if_t<IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(bool val) {
    return static_cast<typename DstType::UnpackedType>(val ? 1 : 0);
  }
};

// float -> (U)Int4x2
// Per https://onnx.ai/onnx/operators/onnx__Cast.html#summary, casting from
// floating point to fixed point is undefined if OOR.
template <typename DstType>
struct ToInt4Converter<float, DstType,
                       std::enable_if_t<IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const float& val) {
    int result = static_cast<int>(std::roundf(val));
    return ToInt4Converter<int, DstType>::Convert(result);
  }
};

// double -> (U)Int4x2
template <typename DstType>
struct ToInt4Converter<double, DstType,
                       std::enable_if_t<IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const double& val) {
    int result = static_cast<int>(std::round(val));
    return ToInt4Converter<int, DstType>::Convert(result);
  }
};

// float 8 -> (U)Int4x2
template <typename SrcType, typename DstType>
struct ToInt4Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat8Type<SrcType>::value && IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    float result = val.ToFloat();
    return ToInt4Converter<float, DstType>::Convert(result);
  }
};

// float 16 -> (U)Int4x2
template <typename SrcType, typename DstType>
struct ToInt4Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat16Type<SrcType>::value && IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    float f_val = static_cast<float>(val);
    return ToInt4Converter<float, Int4x2>::Convert(f_val);
  }
};

template <typename SrcType, typename DstType>
struct ToInt4Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat6Type<SrcType>::value && IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    return ToInt4Converter<int, DstType>::Convert(static_cast<int>(static_cast<float>(val)));
  }
};

// string -> (U)Int4x2
template <typename DstType>
struct ToInt4Converter<std::string, DstType,
                       std::enable_if_t<IsOrtInt4Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const std::string& val) {
    double result = std::stod(val);
    return ToInt4Converter<double, DstType>::Convert(result);
  }
};

// Helper for converting (U)Int2x4 values to any destination type.
template <typename SrcType, typename DstType,
          typename Enable = std::enable_if_t<IsOrtInt2Type<SrcType>::value && IsOrtInt2ConversionType<DstType>::value>>
struct FromInt2Converter {
  // The input 'val' can be either an int8_t value coming from Int2x4.GetElem(pos),
  // or an uint8_t value coming from UInt2x4.GetElem(pos), where pos can be 0, 1, 2, or 3.
  static DstType Convert(typename SrcType::UnpackedType val) {
    if constexpr (IsOrtFloat16Type<DstType>::value) {
      return DstType(static_cast<float>(val));
    } else if constexpr (IsOrtFloat8Type<DstType>::value) {
      return DstType(static_cast<float>(val), true);
    } else if constexpr (IsOrtFloat6Type<DstType>::value) {
      return DstType(static_cast<float>(val));
    } else if constexpr (std::is_same_v<bool, DstType>) {
      return val != 0;
    } else if constexpr (std::is_same_v<std::string, DstType>) {
      // val has type (u)int8_t, so static_cast<int> is required in order for std::to_string
      // to interpret val as a number (1 -> "1"), instead of a char.
      return std::to_string(static_cast<int>(val));
    } else {
      return static_cast<DstType>(val);
    }
  }
};

// Helper for converting any source type to (U)Int2x4::UnpackedType values (int8_t and uint8_t).
template <typename SrcType, typename DstType,
          typename Enable = std::enable_if_t<IsOrtInt2ConversionType<SrcType>::value && IsOrtInt2Type<DstType>::value>>
struct ToInt2Converter {
  static typename DstType::UnpackedType Convert(const SrcType& val);
};

// Integer types -> Int2x4
// INT2 values range from -2 to 1 (2-bit signed two's complement)
template <typename SrcType>
struct ToInt2Converter<SrcType, Int2x4,
                       std::enable_if_t<IsStandardIntegerType<SrcType>::value>> {
  static int8_t Convert(const SrcType& val) {
    // Truncate to 2 least significant bits
    uint8_t truncated = static_cast<uint8_t>(val) & 0x03;

    // Sign-extend: if bit 1 is set, it's negative in 2-bit two's complement,
    // so set the 6 most significant bits to 1.
    return static_cast<int8_t>((truncated & 0x2) ? (truncated | 0xFC) : truncated);
  }
};

// Integer types -> UInt2x4
// UINT2 values range from 0 to 3 (2-bit unsigned)
template <typename SrcType>
struct ToInt2Converter<SrcType, UInt2x4,
                       std::enable_if_t<IsStandardIntegerType<SrcType>::value>> {
  static uint8_t Convert(const SrcType& val) {
    // Truncate to 2 least significant bits
    return static_cast<uint8_t>(val) & 0x03;
  }
};

// bool -> (U)Int2x4
template <typename DstType>
struct ToInt2Converter<bool, DstType,
                       std::enable_if_t<IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(bool val) {
    return static_cast<typename DstType::UnpackedType>(val ? 1 : 0);
  }
};

// float -> (U)Int2x4
template <typename DstType>
struct ToInt2Converter<float, DstType,
                       std::enable_if_t<IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const float& val) {
    int result = static_cast<int>(std::roundf(val));
    return ToInt2Converter<int, DstType>::Convert(result);
  }
};

// double -> (U)Int2x4
template <typename DstType>
struct ToInt2Converter<double, DstType,
                       std::enable_if_t<IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const double& val) {
    int result = static_cast<int>(std::round(val));
    return ToInt2Converter<int, DstType>::Convert(result);
  }
};

// float 8 -> (U)Int2x4
template <typename SrcType, typename DstType>
struct ToInt2Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat8Type<SrcType>::value && IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    float result = val.ToFloat();
    return ToInt2Converter<float, DstType>::Convert(result);
  }
};

// float 16 -> (U)Int2x4
template <typename SrcType, typename DstType>
struct ToInt2Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat16Type<SrcType>::value && IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    float f_val = static_cast<float>(val);
    return ToInt2Converter<float, DstType>::Convert(f_val);
  }
};

template <typename SrcType, typename DstType>
struct ToInt2Converter<SrcType, DstType,
                       std::enable_if_t<IsOrtFloat6Type<SrcType>::value && IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const SrcType& val) {
    return ToInt2Converter<int, DstType>::Convert(static_cast<int>(static_cast<float>(val)));
  }
};

// string -> (U)Int2x4
template <typename DstType>
struct ToInt2Converter<std::string, DstType,
                       std::enable_if_t<IsOrtInt2Type<DstType>::value>> {
  static typename DstType::UnpackedType Convert(const std::string& val) {
    double result = std::stod(val);
    return ToInt2Converter<double, DstType>::Convert(result);
  }
};

// generic tensor X -> Y
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCaster {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    using SrcEigenCastType = typename EigenCastType<SrcType>::type;
    using DstEigenCastType = typename EigenCastType<DstType>::type;

    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto in_vector =
        ConstEigenVectorMap<SrcEigenCastType>(reinterpret_cast<const SrcEigenCastType*>(in.Data<SrcType>()), shape_size);
    auto out_vector =
        EigenVectorMap<DstEigenCastType>(reinterpret_cast<DstEigenCastType*>(out.MutableData<DstType>()), shape_size);
    out_vector = in_vector.template cast<DstEigenCastType>();
  }
};

// FP6 values are logical elements stored in byte-sized runtime storage. Route their
// scalar conversions around Eigen, which cannot represent the custom FP6 formats.
template <typename SrcType, typename DstType>
struct TensorCaster<SrcType, DstType,
                    std::enable_if_t<(IsOrtFloat6Type<SrcType>::value || IsOrtFloat6Type<DstType>::value) &&
                                     !std::is_same_v<SrcType, MLFloat16> &&
                                     !std::is_same_v<DstType, MLFloat16> &&
                                     !IsOrtInt4Type<SrcType>::value && !IsOrtInt4Type<DstType>::value &&
                                     !IsOrtInt2Type<SrcType>::value && !IsOrtInt2Type<DstType>::value &&
                                     !std::is_same_v<SrcType, std::string> && !std::is_same_v<DstType, std::string>>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      if constexpr (std::is_same_v<SrcType, double> && IsOrtFloat6Type<DstType>::value) {
        out_data[i] = DstType(in_data[i]);
      } else {
        out_data[i] = DstType(static_cast<float>(in_data[i]));
      }
    }
  }
};

template <typename SrcType>
struct TensorCaster<SrcType, MLFloat16, std::enable_if_t<IsOrtFloat6Type<SrcType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<MLFloat16>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = MLFloat16(static_cast<float>(in_data[i]));
    }
  }
};

// tensor X -> string, if X != (U)Int4x2 and X != (U)Int2x4
template <typename SrcType>
struct TensorCaster<SrcType, std::string,
                    std::enable_if_t<!IsOrtInt4Type<SrcType>::value && !IsOrtInt2Type<SrcType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<std::string>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastToString(in_data[i], out_data[i]);
    }
  }
};

// tensor string -> X, if X != (U)Int4x2 and X != (U)Int2x4
template <typename DstType>
struct TensorCaster<std::string, DstType,
                    std::enable_if_t<!IsOrtInt4Type<DstType>::value && !IsOrtInt2Type<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], out_data[i]);
    }
  }
};

// tensor MLFloat16 -> float
template <>
struct TensorCaster<MLFloat16, float> {
  void Cast(const OpKernelContext& ctx, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    auto out_data = out.MutableData<float>();
    auto in_data = in.Data<MLFloat16>();
    const size_t shape_size = narrow<size_t>(shape.Size());
    MlasConvertHalfToFloatBufferInParallel(in_data, out_data, shape_size, ctx.GetOperatorThreadPool());
  }
};

// tensor float -> MLFloat16
template <>
struct TensorCaster<float, MLFloat16> {
  void Cast(const OpKernelContext& ctx, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    auto out_data = out.MutableData<MLFloat16>();
    auto in_data = in.Data<float>();
    const size_t shape_size = narrow<size_t>(shape.Size());
    MlasConvertFloatToHalfBufferInParallel(in_data, out_data, shape_size, ctx.GetOperatorThreadPool());
  }
};

template <>
struct TensorCaster<MLFloat16, Float6E2M3> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<MLFloat16>();
    auto* out_data = out.MutableData<Float6E2M3>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = Float6E2M3(static_cast<float>(in_data[i]));
    }
  }
};

template <>
struct TensorCaster<MLFloat16, Float6E3M2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<MLFloat16>();
    auto* out_data = out.MutableData<Float6E3M2>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = Float6E3M2(static_cast<float>(in_data[i]));
    }
  }
};

// (U)Int4x2 -> string or numeric types
template <typename SrcType, typename DstType>
struct TensorCaster<SrcType, DstType,
                    std::enable_if_t<IsOrtInt4Type<SrcType>::value && IsOrtInt4ConversionType<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      // elem 0 is the low nibble, 1 the high nibble
      auto val = in_data[i >> 1].GetElem(i & 0x1);
      out_data[i] = FromInt4Converter<SrcType, DstType>::Convert(val);
    }
  }
};

// string or numeric types -> (U)Int4x2
template <typename SrcType, typename DstType>
struct TensorCaster<SrcType, DstType,
                    std::enable_if_t<IsOrtInt4ConversionType<SrcType>::value && IsOrtInt4Type<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();

    ptrdiff_t i = 0;
    for (; i < shape_size - 1; i += 2) {
      auto low_val = ToInt4Converter<SrcType, DstType>::Convert(in_data[i]);
      auto high_val = ToInt4Converter<SrcType, DstType>::Convert(in_data[i + 1]);
      out_data[i >> 1] = DstType(low_val, high_val);
    }

    if (i < shape_size) {
      auto low_val = ToInt4Converter<SrcType, DstType>::Convert(in_data[i]);
      out_data[i >> 1] = DstType(low_val, 0);
    }
  }
};

// Int4x2 -> UInt4x2
template <>
struct TensorCaster<Int4x2, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size() + 1) >> 1;
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<UInt4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto low_nibble = in_data[i].GetElem(0);
      auto high_nibble = in_data[i].GetElem(1);

      uint8_t low_unsigned = static_cast<uint8_t>(low_nibble) & 0x0F;
      uint8_t high_unsigned = static_cast<uint8_t>(high_nibble) & 0x0F;

      out_data[i] = UInt4x2(low_unsigned, high_unsigned);
    }
  }
};

// UInt4x2 -> Int4x2
template <>
struct TensorCaster<UInt4x2, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size() + 1) >> 1;
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<Int4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto low_nibble = in_data[i].GetElem(0);
      auto high_nibble = in_data[i].GetElem(1);

      int8_t low_signed = static_cast<int8_t>((low_nibble & 0x0F) << 4) >> 4;
      int8_t high_signed = static_cast<int8_t>((high_nibble & 0x0F) << 4) >> 4;

      out_data[i] = Int4x2(low_signed, high_signed);
    }
  }
};

// (U)Int2x4 -> string or numeric types
template <typename SrcType, typename DstType>
struct TensorCaster<SrcType, DstType,
                    std::enable_if_t<IsOrtInt2Type<SrcType>::value && IsOrtInt2ConversionType<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      // 4 elements per byte, accessed by position 0-3
      auto val = in_data[i >> 2].GetElem(i & 0x3);
      out_data[i] = FromInt2Converter<SrcType, DstType>::Convert(val);
    }
  }
};

// string or numeric types -> (U)Int2x4
template <typename SrcType, typename DstType>
struct TensorCaster<SrcType, DstType,
                    std::enable_if_t<IsOrtInt2ConversionType<SrcType>::value && IsOrtInt2Type<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();

    ptrdiff_t i = 0;
    // Process 4 elements at a time
    for (; i + 3 < shape_size; i += 4) {
      auto val0 = ToInt2Converter<SrcType, DstType>::Convert(in_data[i]);
      auto val1 = ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 1]);
      auto val2 = ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 2]);
      auto val3 = ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 3]);
      out_data[i >> 2] = DstType(val0, val1, val2, val3);
    }

    // Handle remaining elements
    if (i < shape_size) {
      const auto zero = typename DstType::UnpackedType(0);
      auto val0 = ToInt2Converter<SrcType, DstType>::Convert(in_data[i]);
      auto val1 = (i + 1 < shape_size) ? ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 1]) : zero;
      auto val2 = (i + 2 < shape_size) ? ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 2]) : zero;
      auto val3 = (i + 3 < shape_size) ? ToInt2Converter<SrcType, DstType>::Convert(in_data[i + 3]) : zero;
      out_data[i >> 2] = DstType(val0, val1, val2, val3);
    }
  }
};

// Int2x4 -> UInt2x4
template <>
struct TensorCaster<Int2x4, UInt2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t num_quads = narrow<ptrdiff_t>((shape.Size() + 3) >> 2);
    const auto* in_data = in.Data<Int2x4>();
    auto* out_data = out.MutableData<UInt2x4>();

    for (ptrdiff_t i = 0; i < num_quads; ++i) {
      auto e0 = in_data[i].GetElem(0);
      auto e1 = in_data[i].GetElem(1);
      auto e2 = in_data[i].GetElem(2);
      auto e3 = in_data[i].GetElem(3);

      // Reinterpret: just mask to 2 bits
      uint8_t u0 = static_cast<uint8_t>(e0) & 0x03;
      uint8_t u1 = static_cast<uint8_t>(e1) & 0x03;
      uint8_t u2 = static_cast<uint8_t>(e2) & 0x03;
      uint8_t u3 = static_cast<uint8_t>(e3) & 0x03;

      out_data[i] = UInt2x4(u0, u1, u2, u3);
    }
  }
};

// UInt2x4 -> Int2x4
template <>
struct TensorCaster<UInt2x4, Int2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t num_quads = narrow<ptrdiff_t>((shape.Size() + 3) >> 2);
    const auto* in_data = in.Data<UInt2x4>();
    auto* out_data = out.MutableData<Int2x4>();

    for (ptrdiff_t i = 0; i < num_quads; ++i) {
      auto e0 = in_data[i].GetElem(0);
      auto e1 = in_data[i].GetElem(1);
      auto e2 = in_data[i].GetElem(2);
      auto e3 = in_data[i].GetElem(3);

      // Sign-extend: if bit 1 is set, the value is negative in 2-bit two's complement
      int8_t s0 = static_cast<int8_t>((e0 & 0x2) ? (e0 | 0xFC) : e0);
      int8_t s1 = static_cast<int8_t>((e1 & 0x2) ? (e1 | 0xFC) : e1);
      int8_t s2 = static_cast<int8_t>((e2 & 0x2) ? (e2 | 0xFC) : e2);
      int8_t s3 = static_cast<int8_t>((e3 & 0x2) ? (e3 | 0xFC) : e3);

      out_data[i] = Int2x4(s0, s1, s2, s3);
    }
  }
};

// Int4x2 -> Int2x4
template <>
struct TensorCaster<Int4x2, Int2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<Int2x4>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 1].GetElem(i & 0x1);
      int8_t truncated = static_cast<int8_t>((val & 0x03) << 6) >> 6;
      out_data[i >> 2].SetElem(i & 0x3, truncated);
    }
  }
};

// Int4x2 -> UInt2x4
template <>
struct TensorCaster<Int4x2, UInt2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<Int4x2>();
    auto* out_data = out.MutableData<UInt2x4>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 1].GetElem(i & 0x1);
      uint8_t truncated = static_cast<uint8_t>(val) & 0x03;
      out_data[i >> 2].SetElem(i & 0x3, truncated);
    }
  }
};

// UInt4x2 -> Int2x4
template <>
struct TensorCaster<UInt4x2, Int2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<Int2x4>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 1].GetElem(i & 0x1);
      int8_t truncated = static_cast<int8_t>((val & 0x03) << 6) >> 6;
      out_data[i >> 2].SetElem(i & 0x3, truncated);
    }
  }
};

// UInt4x2 -> UInt2x4
template <>
struct TensorCaster<UInt4x2, UInt2x4> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<UInt4x2>();
    auto* out_data = out.MutableData<UInt2x4>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 1].GetElem(i & 0x1);
      uint8_t truncated = val & 0x03;
      out_data[i >> 2].SetElem(i & 0x3, truncated);
    }
  }
};

// Int2x4 -> Int4x2
template <>
struct TensorCaster<Int2x4, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<Int2x4>();
    auto* out_data = out.MutableData<Int4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 2].GetElem(i & 0x3);
      out_data[i >> 1].SetElem(i & 0x1, val);
    }
  }
};

// Int2x4 -> UInt4x2
template <>
struct TensorCaster<Int2x4, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<Int2x4>();
    auto* out_data = out.MutableData<UInt4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 2].GetElem(i & 0x3);
      uint8_t masked = static_cast<uint8_t>(val) & 0x0F;
      out_data[i >> 1].SetElem(i & 0x1, masked);
    }
  }
};

// UInt2x4 -> Int4x2
template <>
struct TensorCaster<UInt2x4, Int4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<UInt2x4>();
    auto* out_data = out.MutableData<Int4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 2].GetElem(i & 0x3);
      out_data[i >> 1].SetElem(i & 0x1, static_cast<int8_t>(val));
    }
  }
};

// UInt2x4 -> UInt4x2
template <>
struct TensorCaster<UInt2x4, UInt4x2> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const ptrdiff_t shape_size = narrow<ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<UInt2x4>();
    auto* out_data = out.MutableData<UInt4x2>();

    for (ptrdiff_t i = 0; i < shape_size; ++i) {
      auto val = in_data[i >> 2].GetElem(i & 0x3);
      out_data[i >> 1].SetElem(i & 0x1, val);
    }
  }
};

#if defined(_M_AMD64) && !defined(_M_ARM64EC)
// specializations to use optimized and Windows x64-specific

Tensor GetIntermediateMLFloat16ToFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in) {
  AllocatorPtr allocator;
  ORT_THROW_IF_ERROR(context.GetTempSpaceAllocator(&allocator));
  Tensor out{DataTypeImpl::GetType<float>(), shape, allocator};
  TensorCaster<MLFloat16, float>{}.Cast(context, shape, in, out);
  return out;
}

template <typename DstType>
void CastMLFloat16ThroughFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) {
  // use optimized MLFloat16 -> float, then float -> DstType
  Tensor intermediate_tensor = GetIntermediateMLFloat16ToFloatTensor(context, shape, in);
  TensorCaster<float, DstType>{}.Cast(context, shape, intermediate_tensor, out);
}

// tensor MLFloat16 -> X
template <typename DstType>
struct TensorCaster<MLFloat16, DstType,
                    std::enable_if_t<!IsOrtInt4Type<DstType>::value && !IsOrtInt2Type<DstType>::value>> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<DstType>(context, shape, in, out);
  }
};

// tensor MLFloat16 -> string
template <>
struct TensorCaster<MLFloat16, std::string> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<std::string>(context, shape, in, out);
  }
};
#endif

#if !defined(DISABLE_FLOAT8_TYPES)
// TensorCasterNoSat is only called when all the below conditions are met (see Cast::Compute):
// - defined(DISABLE_FLOAT8_TYPES) == false
// - saturate_ == false
// - IsOrtFloat8Type<DstType>::value == true

// tensor X -> float 8
template <typename SrcType, typename DstType,
          typename Enable = std::enable_if_t<IsOrtFloat8Type<DstType>::value>>
struct TensorCasterNoSat {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = DstType(static_cast<float>(in_data[i]), false);
    }
  }
};

// tensor (U)Int4x2 -> float 8
template <typename SrcType, typename DstType>
struct TensorCasterNoSat<SrcType, DstType,
                         std::enable_if_t<IsOrtInt4Type<SrcType>::value && IsOrtFloat8Type<DstType>::value>> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) const {
    // Int4x2/UInt4x2 always fit inside any Float8 type, so we can reuse the saturate = true implementation.
    TensorCaster<SrcType, DstType>{}.Cast(context, shape, src, dst);
  }
};

// tensor (U)Int2x4 -> float 8
template <typename SrcType, typename DstType>
struct TensorCasterNoSat<SrcType, DstType,
                         std::enable_if_t<IsOrtInt2Type<SrcType>::value && IsOrtFloat8Type<DstType>::value>> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) const {
    // Int2x4/UInt2x4 always fit inside any Float8 type, so we can reuse the saturate = true implementation.
    TensorCaster<SrcType, DstType>{}.Cast(context, shape, src, dst);
  }
};

// tensor string -> float 8
template <typename DstType>
struct TensorCasterNoSat<std::string, DstType,
                         std::enable_if_t<IsOrtFloat8Type<DstType>::value>> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    float float_value;
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], float_value);
      out_data[i] = DstType(float_value, false);
    }
  }
};

#endif

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
#if !defined(DISABLE_FLOAT8_TYPES)
    if (saturate == 0 && (to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E8M0 &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT6E2M3 &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT6E3M2)) {
      ORT_THROW("Attribute saturate is only used for cast to low-precision floating-point types.");
    }
#else
    if (saturate == 0 && to != ONNX_NAMESPACE::TensorProto::FLOAT6E2M3 &&
        to != ONNX_NAMESPACE::TensorProto::FLOAT6E3M2) {
      ORT_THROW("Attribute saturate is only used for cast to low-precision floating-point types.");
    }
#endif
    saturate_ = saturate == 1;

    // round_mode only applies for casting to float8e8m0 (introduced in opset 24)
    std::string round_mode_str = info.GetAttrOrDefault("round_mode", std::string("up"));
#if !defined(DISABLE_FLOAT8_TYPES)
    if (round_mode_str == "up") {
      round_mode_ = Float8E8M0::RoundMode::Up;
    } else if (round_mode_str == "down") {
      round_mode_ = Float8E8M0::RoundMode::Down;
    } else if (round_mode_str == "nearest") {
      round_mode_ = Float8E8M0::RoundMode::Nearest;
    } else {
      ORT_THROW("Attribute round_mode must be 'up', 'down', or 'nearest'.");
    }
    if (round_mode_ != Float8E8M0::RoundMode::Up && to != ONNX_NAMESPACE::TensorProto::FLOAT8E8M0) {
      ORT_THROW("Attribute round_mode is only used for cast to float8e8m0.");
    }
#else
    if (round_mode_str != "up") {
      ORT_THROW("Attribute round_mode is only used for cast to float8e8m0.");
    }
#endif
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
#if !defined(DISABLE_FLOAT8_TYPES)
  Float8E8M0::RoundMode round_mode_{Float8E8M0::RoundMode::Up};
#endif
};

template <typename TSrc, typename TDst>
struct Dispatcher {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCaster<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc, typename TDst>
struct DispatcherNoSat {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCasterNoSat<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#endif

template <typename TSrc>
struct SrcDispatcher {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypes, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<Dispatcher, TypeList<TSrc>>(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc>
struct SrcDispatcherNoSat {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypeOnlyFloat8 = boost::mp11::mp_set_intersection<
        EnabledDstTypes, element_type_lists::AllFloat8>;
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypeOnlyFloat8, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<DispatcherNoSat, TypeList<TSrc>>(context, shape, src, dst);
  }
};

// Dispatcher for casting any source type to Float8E8M0 with round_mode and saturate support.
// This bypasses the generic TensorCaster/TensorCasterNoSat templates to thread round_mode through.
template <typename TSrc>
struct CastToE8M0Dispatcher {
  void operator()(const OpKernelContext&, const TensorShape& shape, const Tensor& src, Tensor& dst,
                  bool saturate, Float8E8M0::RoundMode round_mode) {
    const auto shape_size = narrow<std::ptrdiff_t>(shape.Size());
    auto* out_data = dst.MutableData<Float8E8M0>();

    if constexpr (IsOrtInt4Type<TSrc>::value) {
      const auto* in_data = src.Data<TSrc>();
      for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
        auto val = in_data[i >> 1].GetElem(i & 0x1);
        out_data[i] = Float8E8M0(static_cast<float>(val), saturate, round_mode);
      }
    } else if constexpr (IsOrtInt2Type<TSrc>::value) {
      const auto* in_data = src.Data<TSrc>();
      for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
        auto val = in_data[i >> 2].GetElem(i & 0x3);
        out_data[i] = Float8E8M0(static_cast<float>(val), saturate, round_mode);
      }
    } else if constexpr (std::is_same_v<TSrc, std::string>) {
      const auto* in_data = src.Data<std::string>();
      for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
        float float_val;
        CastFromString(in_data[i], float_val);
        out_data[i] = Float8E8M0(float_val, saturate, round_mode);
      }
    } else {
      const auto* in_data = src.Data<TSrc>();
      for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
        out_data[i] = Float8E8M0(static_cast<float>(in_data[i]), saturate, round_mode);
      }
    }
  }
};

#endif

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);

  if (shape.Size() == 0) {
    return Status::OK();
  }

  const auto from = X->GetElementType();

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return Status::OK();
  }

#if !defined(DISABLE_FLOAT8_TYPES)
  // Float8E8M0 destination needs special handling for round_mode support.
  // Dispatch directly to avoid threading round_mode through the TensorCaster templates.
  if (to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E8M0) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<CastToE8M0Dispatcher>(*context, shape, *X, *Y, saturate_, round_mode_);
    return Status::OK();
  }

  if (saturate_) {
#endif
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcher>(to_, *context, shape, *X, *Y);
#if !defined(DISABLE_FLOAT8_TYPES)
  } else if (to_ == ONNX_NAMESPACE::TensorProto::FLOAT6E2M3 ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT6E3M2) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcher>(to_, *context, shape, *X, *Y);
  } else if (to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcherNoSat>(to_, *context, shape, *X, *Y);
  }
#endif

  return Status::OK();
}
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    12,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset24>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset24>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    13,
    18,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset24>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset24>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    19,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset24>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset24>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// TODO(adrianlizarraga): Implement support for int4 and uint4.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    21,
    22,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset24>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset24>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// Opset 23 added support for float4e2m1.
// TODO: Implement support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    23,
    23,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset24>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset24>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

// Opset 24 added support for float8e8m0.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    24,
    24,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset28>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset28>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    25, 27,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypesPreOpset28>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypesPreOpset28>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    28,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

}  // namespace onnxruntime
