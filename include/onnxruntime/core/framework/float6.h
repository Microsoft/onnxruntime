// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace onnxruntime {

inline float Float6ToFloat(uint8_t bits, int exponent_bits, int mantissa_bits, int exponent_bias) {
  const uint8_t sign = bits >> 5;
  const uint8_t exponent = (bits >> mantissa_bits) & ((1u << exponent_bits) - 1);
  const uint8_t mantissa = bits & ((1u << mantissa_bits) - 1);
  const float significand = exponent == 0
                                ? static_cast<float>(mantissa)
                                : static_cast<float>((1u << mantissa_bits) | mantissa);
  const float value = std::ldexp(significand, exponent == 0
                                                  ? 1 - exponent_bias - mantissa_bits
                                                  : static_cast<int>(exponent) - exponent_bias - mantissa_bits);
  return sign == 0 ? value : -value;
}

template <int ExponentBits, int MantissaBits, int ExponentBias, typename FloatType>
inline uint8_t FloatToFloat6(FloatType value) {
  static const auto magnitudes = [] {
    std::array<float, 32> values{};
    for (uint8_t bits = 0; bits < values.size(); ++bits) {
      values[bits] = Float6ToFloat(bits, ExponentBits, MantissaBits, ExponentBias);
    }
    return values;
  }();

  if (std::isnan(value)) {
    return 0x20;
  }

  const uint8_t sign = std::signbit(value) ? 0x20 : 0;
  if (std::isinf(value)) {
    return static_cast<uint8_t>(sign | 0x1F);
  }
  const FloatType magnitude = std::abs(value);
  if (magnitude > magnitudes.back()) {
    return static_cast<uint8_t>(sign | 0x1F);
  }

  const auto upper = std::lower_bound(magnitudes.begin(), magnitudes.end(), magnitude);
  size_t index = static_cast<size_t>(upper - magnitudes.begin());
  if (index != 0) {
    const FloatType lower_distance = magnitude - magnitudes[index - 1];
    const FloatType upper_distance = magnitudes[index] - magnitude;
    if (lower_distance < upper_distance || (lower_distance == upper_distance && ((index - 1) & 1) == 0)) {
      --index;
    }
  }
  return static_cast<uint8_t>(sign | index);
}

struct Float6E2M3 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E2M3(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  explicit Float6E2M3(float value) : val(FloatToFloat6<2, 3, 1>(value)) {}
  explicit Float6E2M3(double value) : val(FloatToFloat6<2, 3, 1>(value)) {}
  constexpr uint8_t ToBits() const { return val; }
  operator float() const { return Float6ToFloat(val, 2, 3, 1); }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

struct Float6E3M2 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E3M2(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  explicit Float6E3M2(float value) : val(FloatToFloat6<3, 2, 3>(value)) {}
  explicit Float6E3M2(double value) : val(FloatToFloat6<3, 2, 3>(value)) {}
  constexpr uint8_t ToBits() const { return val; }
  operator float() const { return Float6ToFloat(val, 3, 2, 3); }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

static_assert(sizeof(Float6E2M3) == sizeof(uint8_t));
static_assert(sizeof(Float6E3M2) == sizeof(uint8_t));

}  // namespace onnxruntime
