// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include <core/common/safeint.h>
#include "core/framework/element_type_lists.h"
#include "core/common/float8.h"
#include "core/common/float16.h"
#include "core/framework/float6.h"
#include "core/framework/int4.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

namespace onnxruntime {

template <typename T>
class DequantizeLinear final : public OpKernel {
 public:
  explicit DequantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
};

template <typename T>
class QuantizeLinear final : public OpKernel {
 public:
  explicit QuantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
    precision_ = info.node().SinceVersion() < 23
                     ? static_cast<int64_t>(ONNX_NAMESPACE::TensorProto::FLOAT)
                     : info.GetAttrOrDefault<int64_t>("precision", 0);
    ORT_ENFORCE(precision_ == 0 || precision_ == ONNX_NAMESPACE::TensorProto::FLOAT ||
                    precision_ == ONNX_NAMESPACE::TensorProto::FLOAT16,
                "CPU QuantizeLinear only supports precision 0 (use scale type), FLOAT, and FLOAT16.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
  int64_t block_size_;
  int64_t precision_;
};

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t quant_block_size,
                          int64_t& process_block_count,
                          int64_t& broadcast_dim,
                          int64_t& process_block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    process_block_count = 1;
    broadcast_dim = 1;
    process_block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(quant_block_size == 0, "block_size must be 0 for per-tensor quantization.");
  } else {  // per-axis or blocked QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    process_block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    broadcast_dim = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    process_block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    if (quant_block_size) {  // blocked quantization
      ORT_ENFORCE(scale.Shape().NumDimensions() == input_shape.NumDimensions(),
                  "x_scale and x must have the same rank for blocked quantization");
      ORT_ENFORCE(zero_point_ptr == nullptr || zero_point_ptr->Shape().NumDimensions() == input_shape.NumDimensions(),
                  "x_zero_point must be null or have the same rank as x for blocked quantization");

      for (size_t i = 0, ndim = input_shape.NumDimensions(); i < ndim; ++i) {
        if (i == SafeInt<size_t>(axis_no_neg)) {
          ORT_ENFORCE(scale.Shape()[i] == (input_shape[i] + quant_block_size - 1) / quant_block_size,
                      "x_scale must be ceil(Di/block_size) on the quantize axis i for blocked quantization");
        } else {
          ORT_ENFORCE(scale.Shape()[i] == input_shape[i],
                      "x_scale and x must have the same shape despite the quantize axis for blocked quantization");
        }

        if (zero_point_ptr) {
          ORT_ENFORCE(zero_point_ptr->Shape()[i] == scale.Shape()[i],
                      "x_zero_point and x_scale must have the same shape for blocked quantization");
        }
      }
    } else {  // per-axis quantization
      ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                  "For per axis quantization, scale must be 1D tensor with size ",
                  broadcast_dim);
      ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 &&
                                                zero_point_ptr->Shape()[0] == broadcast_dim),
                  "For per axis quantization, x_zero_point must be null or 1D tensor with size ",
                  broadcast_dim);
    }
  }
}

#define REGISTER_DEQUANTIZELINEAR(T)                                         \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                  \
      DequantizeLinear,                                                      \
      25, 27,                                                                \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_28(T)                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                            \
      DequantizeLinear,                                                      \
      28,                                                                    \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T, start_version, end_version)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                  \
      DequantizeLinear,                                                      \
      start_version,                                                         \
      end_version,                                                           \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(T, start_version, end_version) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                       \
      DequantizeLinear,                                                           \
      start_version,                                                              \
      end_version,                                                                \
      T,                                                                          \
      KernelDefBuilder()                                                          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())                 \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),            \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}),      \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(T)             \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      13,                                                         \
      18,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);                                       \
                                                                  \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      10,                                                         \
      12,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

// Opset 25-27
REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int16_t)
REGISTER_DEQUANTIZELINEAR(uint16_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR(Int4x2)
REGISTER_DEQUANTIZELINEAR(UInt4x2)
REGISTER_DEQUANTIZELINEAR(Int2x4)
REGISTER_DEQUANTIZELINEAR(UInt2x4)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR(Float8E5M2)
REGISTER_DEQUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// Opset 28 adds float6 quantization types.
REGISTER_DEQUANTIZELINEAR_28(int8_t)
REGISTER_DEQUANTIZELINEAR_28(uint8_t)
REGISTER_DEQUANTIZELINEAR_28(int16_t)
REGISTER_DEQUANTIZELINEAR_28(uint16_t)
REGISTER_DEQUANTIZELINEAR_28(int32_t)
REGISTER_DEQUANTIZELINEAR_28(Int4x2)
REGISTER_DEQUANTIZELINEAR_28(UInt4x2)
REGISTER_DEQUANTIZELINEAR_28(Int2x4)
REGISTER_DEQUANTIZELINEAR_28(UInt2x4)
REGISTER_DEQUANTIZELINEAR_28(Float6E2M3)
REGISTER_DEQUANTIZELINEAR_28(Float6E3M2)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_28(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR_28(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR_28(Float8E5M2)
REGISTER_DEQUANTIZELINEAR_28(Float8E5M2FNUZ)
#endif

// opset24
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int16_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint16_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Int4x2, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(UInt4x2, 24, 24)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FN, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 24, 24)
#endif

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int16_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint16_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Int4x2, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(UInt4x2, 23, 23)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FN, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 23, 23)
#endif

// Opset 21 added 16-bit and 4-bit int to DQ.
// TODO(adrianlizarraga): Also support 4-bit int types and 'block' quantization.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int16_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint16_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int32_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Int4x2, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(UInt4x2, 21, 22)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 21, 22)
#endif

// Opset 19 added 8-bit float inputs and 16-bit float outputs to DQ.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int32_t, 19, 20)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 19, 20)
#endif

// Before opset 19, DQ only supported int8, uint8 and int32.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(int32_t)

#if !defined(DISABLE_CONTRIB_OPS)
namespace contrib {

// Register alternate MS domain versions of the DequantizeLinear kernel.
// The MS domain versions additionally support 16-bit integer quantization types.
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    uint16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int32_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    Int4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<Int4x2>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<Int4x2>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    UInt4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<UInt4x2>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<UInt4x2>);

}  // namespace contrib
#endif  // !defined(DISABLE_CONTRIB_OPS)

template <typename T, typename ScaleT, typename OutT, bool is_sub_byte, int elements_per_byte = 0>
struct DequantizeLinearApply;

// The dimensions before quantize axis and after quantize axis can be flattened.
// After flattening, the tensor can be represented by a rank-3 tensor.
// If the quantization happens on the first or last axis, the flattened tensor is
// effectively rank-2.
// For per tensor quantization, the tensor is effectively rank-1.
template <typename T, typename ScaleT, typename OutT, int elements_per_byte>
struct DequantizeLinearApply<T, ScaleT, OutT, false, elements_per_byte> {
  /**
   * @brief Calculate per-tensor/layer or per-axis quantization of DequantizeLinear on the
   *        flattened tensors.
   * @param[in]    M                      size of dimensions before the quantize axis
   * @param[in]    K                      dimension on the quantize axis
   * @param[in]    N                      size of dimensions after the quantize axis
   * @param[in]    input                  1D array of flattened [D0, ..., Di, ..., Dn]
   * @param[in]    scale                  scalar for per-tensor/layer quantization and 1D array [Di]
   *                                      for per-axis quantization. i is the quantize axis.
   * @param[out]   output                 same shape as input
   * @param[in]    zero_point             same shape as scale
   */
  void op(size_t M, size_t K, size_t N, const T* input,
          const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    for (size_t m = 0; m < M; m++) {
      for (size_t k = 0; k < K; k++) {
#if defined(ORT_CLIENT_PACKAGE_BUILD)
        // TODO: Only using multithreaded/SIMD DQ when ORT is built for client/on-device workloads.
        // Make this the default behavior after more testing.
        if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) &&
                      std::is_same_v<ScaleT, OutT>) {
          ParDequantizeLinearStd<T>(input, output, N, scale[k], zero_point ? zero_point[k] : 0, thread_pool);
          input += N;
          output += N;
        } else {
          auto zp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
          auto sc = static_cast<float>(scale[k]);
          for (size_t n = 0; n < N; n++) {
            *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
          }
        }
#else
        ORT_UNUSED_PARAMETER(thread_pool);
        auto zp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
        auto sc = static_cast<float>(scale[k]);
        for (size_t n = 0; n < N; n++) {
          *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
        }
#endif  // defined(ORT_CLIENT_PACKAGE_BUILD)
      }
    }
  }

  /**
   * @brief Calculate blocked quantization of DequantizeLinear on the flattened tensors.
   *        TODO(fajin): add mlas kernel to utilize multithreading, refer MlasDequantizeBlockwise.
   * @param[in]    M                      size of dimensions before the quantize axis
   * @param[in]    K                      dimension of the quantize axis
   * @param[in]    N                      size of dimensions after the quantize axis
   * @param[in]    quant_block_size       quantize block size along the quantize axis
   * @param[in]    input                  1D array of flattened [D0, ..., Di, ..., Dn]
   * @param[in]    scale                  1D array of flattened [D0, ..., ceil(Di/quant_block_size), ..., Dn].
   *                                      i is the quantize axis.
   * @param[out]   output                 same shape as input
   * @param[in]    zero_point             same shape as scale
   */
  void op(size_t M, size_t K, size_t N, size_t quant_block_size,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    ORT_UNUSED_PARAMETER(thread_pool);
    if (zero_point) {
      for (size_t m = 0; m < M; m++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            // within the quantize block, the zero point and scale are the same.
            for (size_t bs = 0; bs < N; bs++) {
              auto zp = static_cast<int32_t>(zero_point[bs]);
              auto sc = static_cast<float>(scale[bs]);
              *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
            }
          }

          // move to the next quantize block
          zero_point += N;
          scale += N;
        }
      }
    } else {
      for (size_t m = 0; m < M; m++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            // within the quantize block, the zero point and scale are the same.
            for (size_t bs = 0; bs < N; bs++) {
              auto sc = static_cast<float>(scale[bs]);
              *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++)) * sc);
            }
          }

          // move to the next quantize block
          scale += N;
        }
      }
    }
  }
};

template <typename T, typename ScaleT, typename OutT, int elements_per_byte>
struct DequantizeLinearApply<T, ScaleT, OutT, true, elements_per_byte> {
  // per-tensor/layer or per-axis quantization for sub-byte types
  void op(size_t M, size_t K, size_t N,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    ORT_UNUSED_PARAMETER(thread_pool);
    size_t input_index = 0;
    constexpr size_t shift_bits = (elements_per_byte == 2) ? 1 : 2;  // log2(elements_per_byte)
    constexpr size_t mask = elements_per_byte - 1;                   // For modulo operation

    for (size_t m = 0; m < M; m++) {
      for (size_t bd = 0; bd < K; bd++) {
        size_t bd_i = bd >> shift_bits;  // bd / elements_per_byte
        size_t bd_j = bd & mask;         // bd % elements_per_byte
        auto zp = zero_point ? static_cast<int32_t>(zero_point[bd_i].GetElem(bd_j)) : 0;
        auto sc = static_cast<float>(scale[bd]);

        for (size_t bs = 0; bs < N; bs++) {
          size_t input_i = input_index >> shift_bits;
          size_t input_j = input_index & mask;
          int32_t val = static_cast<int32_t>(input[input_i].GetElem(input_j));
          *output++ = static_cast<OutT>(static_cast<float>(val - zp) * sc);
          input_index += 1;
        }
      }
    }

    assert(input_index == M * K * N);
  }

  // Blocked quantization
  // TODO(fajin) : add mlas kernel to utilize multithreading, refer MlasDequantizeBlockwise.
  void op(size_t M, size_t K, size_t N, size_t quant_block_size,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    ORT_UNUSED_PARAMETER(thread_pool);
    size_t input_index = 0;
    constexpr size_t shift_bits = (elements_per_byte == 2) ? 1 : 2;  // log2(elements_per_byte)
    constexpr size_t mask = elements_per_byte - 1;                   // For modulo operation

    if (zero_point) {
      size_t zp_index = 0;

      for (size_t n = 0; n < M; n++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            auto q_zp_index = zp_index;
            for (size_t bs = 0; bs < N; ++bs, ++input_index, ++q_zp_index) {
              auto zp = static_cast<int32_t>(zero_point[q_zp_index >> shift_bits].GetElem(q_zp_index & mask));
              auto sc = static_cast<float>(scale[bs]);

              int32_t val = static_cast<int32_t>(input[input_index >> shift_bits].GetElem(input_index & mask));
              *output++ = static_cast<OutT>(static_cast<float>(val - zp) * sc);
            }
          }

          scale += N;
          zp_index += N;
        }
      }
    } else {
      for (size_t n = 0; n < M; n++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            for (size_t bs = 0; bs < N; ++bs, ++input_index) {
              auto sc = static_cast<float>(scale[bs]);

              int32_t val = static_cast<int32_t>(input[input_index >> shift_bits].GetElem(input_index & mask));
              *output++ = static_cast<OutT>(static_cast<float>(val) * sc);
            }
          }

          scale += N;
        }
      }
    }

    assert(input_index == M * K * N);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

#define DEQUANTIZE_LINEAR_APPLY_FLOAT8(T)                                                            \
  template <typename ScaleT, typename OutT, int elements_per_byte>                                   \
  struct DequantizeLinearApply<T, ScaleT, OutT, false, elements_per_byte> {                          \
    /* Per-tensor/layer or per-axis quantization */                                                  \
    void op(size_t M, size_t K, size_t N,                                                            \
            const T* input, const ScaleT* scale, OutT* output, const T*, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; m++) {                                                               \
        for (size_t bd = 0; bd < K; bd++) {                                                          \
          auto sc = scale[bd];                                                                       \
          for (size_t bs = 0; bs < N; bs++, input++) {                                               \
            *output++ = static_cast<OutT>(input->ToFloat() * sc);                                    \
          }                                                                                          \
        }                                                                                            \
      }                                                                                              \
    }                                                                                                \
    /* Blocked quantization */                                                                       \
    void op(size_t M, size_t K, size_t N, size_t quant_block_size,                                   \
            const T* input, const ScaleT* scale, OutT* output, const T*, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; m++) {                                                               \
        for (size_t bd = 0; bd < K; bd += quant_block_size) {                                        \
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {      \
            for (size_t bs = 0; bs < N; bs++, input++) {                                             \
              auto sc = static_cast<float>(scale[bs]);                                               \
              *output++ = static_cast<OutT>(input->ToFloat() * sc);                                  \
            }                                                                                        \
          }                                                                                          \
          scale += N;                                                                                \
        }                                                                                            \
      }                                                                                              \
    }                                                                                                \
  };

DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E4M3FN)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E4M3FNUZ)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E5M2)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E5M2FNUZ)

#endif

#define DEQUANTIZE_LINEAR_APPLY_FLOAT6(T)                                                                       \
  template <typename ScaleT, typename OutT, int elements_per_byte>                                              \
  struct DequantizeLinearApply<T, ScaleT, OutT, false, elements_per_byte> {                                     \
    void op(size_t M, size_t K, size_t N,                                                                       \
            const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; ++m) {                                                                          \
        for (size_t bd = 0; bd < K; ++bd) {                                                                     \
          const float sc = static_cast<float>(scale[bd]);                                                       \
          for (size_t bs = 0; bs < N; ++bs, ++input) {                                                          \
            const float zp = zero_point ? static_cast<float>(zero_point[bd]) : 0.0f;                            \
            *output++ = static_cast<OutT>((static_cast<float>(*input) - zp) * sc);                              \
          }                                                                                                     \
        }                                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
    void op(size_t M, size_t K, size_t N, size_t quant_block_size,                                              \
            const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; ++m) {                                                                          \
        for (size_t bd = 0; bd < K; bd += quant_block_size) {                                                   \
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {                 \
            for (size_t bs = 0; bs < N; ++bs, ++input) {                                                        \
              const float zp = zero_point ? static_cast<float>(zero_point[bs]) : 0.0f;                          \
              *output++ = static_cast<OutT>((static_cast<float>(*input) - zp) * static_cast<float>(scale[bs])); \
            }                                                                                                   \
          }                                                                                                     \
          scale += N;                                                                                           \
          if (zero_point) zero_point += N;                                                                      \
        }                                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
  };

DEQUANTIZE_LINEAR_APPLY_FLOAT6(Float6E2M3)
DEQUANTIZE_LINEAR_APPLY_FLOAT6(Float6E3M2)

// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t process_block_count;
  int64_t broadcast_dim;
  int64_t process_block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, block_size_,
                process_block_count, broadcast_dim, process_block_size);

  const T* zero_point = x_zero_point ? x_zero_point->Data<T>() : nullptr;

#if !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (boost::mp11::mp_contains<element_type_lists::AllFloat8, T>::value) {
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](T zp) { return zp == T{0}; }),
                "DequantizeLinear with type float8 should have no zero point or all zero points should be 0");
  }
#endif

  const auto to = y.GetElementType();
  const T* input = x.Data<T>();
  constexpr bool is_sub_byte = boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2, Int2x4, UInt2x4>, T>::value;
  // Determine elements_per_byte: Int4x2/UInt4x2 = 2, Int2x4/UInt2x4 = 4
  constexpr int elements_per_byte =
      boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value ? 2 : boost::mp11::mp_contains<TypeList<Int2x4, UInt2x4>, T>::value ? 4
                                                                                                                                        : 0;
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  auto dequantize = [&]<typename ScaleT, typename OutT>(const ScaleT* scale, OutT* output) {
    DequantizeLinearApply<T, ScaleT, OutT, is_sub_byte, elements_per_byte> apply;
    if (block_size_) {
      apply.op(static_cast<size_t>(process_block_count),
               static_cast<size_t>(broadcast_dim),
               static_cast<size_t>(process_block_size),
               static_cast<size_t>(block_size_),
               input, scale, output, zero_point, thread_pool);
    } else {
      apply.op(static_cast<size_t>(process_block_count),
               static_cast<size_t>(broadcast_dim),
               static_cast<size_t>(process_block_size),
               input, scale, output, zero_point, thread_pool);
    }
  };

  if (to == ONNX_NAMESPACE::TensorProto::FLOAT) {
    float* output = y.MutableData<float>();
    if (x_scale.IsDataType<float>()) {
      dequantize(x_scale.Data<float>(), output);
    } else {
      dequantize(x_scale.Data<MLFloat16>(), output);
    }
  } else if (to == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    MLFloat16* output = y.MutableData<MLFloat16>();
    if (x_scale.IsDataType<float>()) {
      dequantize(x_scale.Data<float>(), output);
    } else {
      dequantize(x_scale.Data<MLFloat16>(), output);
    }
  } else if (to == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    ORT_THROW("DequantizeLinear into BFLOAT16 is not implemented yet.");
  } else {
    ORT_THROW("DequantizeLinear only outputs FLOAT16, FLOAT or BFLOAT16.");
  }

  return Status::OK();
}

#define REGISTER_QUANTIZELINEAR(T)                                          \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                 \
      QuantizeLinear,                                                       \
      25, 27,                                                               \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_28(T)                                       \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                           \
      QuantizeLinear,                                                       \
      28,                                                                   \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T, start_version, end_version)    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                 \
      QuantizeLinear,                                                       \
      start_version,                                                        \
      end_version,                                                          \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(T, start_version, end_version) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                     \
      QuantizeLinear,                                                           \
      start_version,                                                            \
      end_version,                                                              \
      T,                                                                        \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),          \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),              \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(T)                   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      13,                                                             \
      18,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);                                             \
                                                                      \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      10,                                                             \
      12,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

// Opset 25-27
REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)
REGISTER_QUANTIZELINEAR(int16_t)
REGISTER_QUANTIZELINEAR(uint16_t)
REGISTER_QUANTIZELINEAR(Int4x2)
REGISTER_QUANTIZELINEAR(UInt4x2)
REGISTER_QUANTIZELINEAR(Int2x4)
REGISTER_QUANTIZELINEAR(UInt2x4)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR(Float8E4M3FN)
REGISTER_QUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR(Float8E5M2)
REGISTER_QUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// Opset 28 adds float6 quantization types.
REGISTER_QUANTIZELINEAR_28(int8_t)
REGISTER_QUANTIZELINEAR_28(uint8_t)
REGISTER_QUANTIZELINEAR_28(int16_t)
REGISTER_QUANTIZELINEAR_28(uint16_t)
REGISTER_QUANTIZELINEAR_28(Int4x2)
REGISTER_QUANTIZELINEAR_28(UInt4x2)
REGISTER_QUANTIZELINEAR_28(Int2x4)
REGISTER_QUANTIZELINEAR_28(UInt2x4)
REGISTER_QUANTIZELINEAR_28(Float6E2M3)
REGISTER_QUANTIZELINEAR_28(Float6E3M2)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_28(Float8E4M3FN)
REGISTER_QUANTIZELINEAR_28(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR_28(Float8E5M2)
REGISTER_QUANTIZELINEAR_28(Float8E5M2FNUZ)
#endif

// Opset 24
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(int16_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(uint16_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Int4x2, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(UInt4x2, 24, 24)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FN, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 24, 24)
#endif

// Opset 23 added support for float4e2m1.
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(int16_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(uint16_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Int4x2, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(UInt4x2, 23, 23)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FN, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 23, 23)
#endif

// Opset 21 added 16-bit and 4-bit int support to Q ops.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int16_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint16_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Int4x2, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(UInt4x2, 21, 22)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 21, 22)
#endif

// Opset 19 added 8-bit floats to Q ops.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 19, 20)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 19, 20)
#endif

// Before opset 19, Q only supported int8 and uint8.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(uint8_t)

#if !defined(DISABLE_CONTRIB_OPS)
namespace contrib {

// Register alternate MS domain versions of the QuantizeLinear kernel.
// The MS domain versions additionally support 16-bit integer quantization types.
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    QuantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()),
    QuantizeLinear<int8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    uint16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint16_t>()),
    QuantizeLinear<uint16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    int16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int16_t>()),
    QuantizeLinear<int16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    Int4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<Int4x2>()),
    QuantizeLinear<Int4x2>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    UInt4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<UInt4x2>()),
    QuantizeLinear<UInt4x2>);
}  // namespace contrib
#endif  // !defined(DISABLE_CONTRIB_OPS)

template <typename InputType, typename OutputType>
void ParQuantizeLinear(const InputType* Input,
                       OutputType* Output,
                       size_t N,
                       float Scale,
                       size_t bd,
                       const OutputType* ZeroPoint,
                       bool saturate,
                       concurrency::ThreadPool* thread_pool,
                       bool half_precision) {
  if constexpr (!IsFloatQuantizationType<OutputType>) {
    ORT_UNUSED_PARAMETER(saturate);
    ParQuantizeLinearStd(Input, Output, N, Scale, ZeroPoint != nullptr ? ZeroPoint[bd] : (OutputType)0,
                         thread_pool, half_precision);
  } else {
    const auto default_zero_point = []() {
      if constexpr (std::is_same_v<OutputType, Float6E2M3> || std::is_same_v<OutputType, Float6E3M2>) {
        return OutputType(0.0f);
      } else {
        return OutputType(0.0f, true);
      }
    }();
    ParQuantizeLinearSat(Input, Output, N, Scale,
                         ZeroPoint != nullptr ? ZeroPoint[bd] : default_zero_point,
                         saturate, thread_pool, half_precision);
  }
}

/**
 * @brief Compute per-tensor or per-axis quantization.
 */
template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx, const InT* input, QuantizeLinearScale scale, const T* zero_point, T* output,
                 int64_t process_block_count, int64_t broadcast_dim, int64_t process_block_size,
                 bool saturate, bool half_precision) {
  for (size_t n = 0; n < static_cast<size_t>(process_block_count); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      ParQuantizeLinear(input, output, static_cast<size_t>(process_block_size), scale[bd], bd, zero_point,
                        saturate, ctx->GetOperatorThreadPool(), half_precision);
      input += process_block_size;
      output += process_block_size;
    }
  }
}

// Helper macros to create zero point with correct number of constructor arguments
#define CREATE_SUB_BYTE_ZP_2(TYPE, zp) TYPE(zp, 0)
#define CREATE_SUB_BYTE_ZP_4(TYPE, zp) TYPE(zp, 0, 0, 0)
#define CREATE_SUB_BYTE_ZP(TYPE, zp, ELEMENTS_PER_BYTE) CREATE_SUB_BYTE_ZP_##ELEMENTS_PER_BYTE(TYPE, zp)

// Quantizes float32 to sub-byte types using MLAS kernel (4-bit) or generic quantization (2-bit).
#define DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(SUB_BYTE_TYPE, QUANT_FUNC, ELEMENTS_PER_BYTE)                               \
  template <>                                                                                                            \
  void ComputeLoop(OpKernelContext* ctx, const float* input, QuantizeLinearScale scale, const SUB_BYTE_TYPE* zero_point, \
                   SUB_BYTE_TYPE* output, int64_t M, int64_t K, int64_t N, bool saturate, bool half_precision) {         \
    ORT_UNUSED_PARAMETER(saturate);                                                                                      \
    size_t output_index = 0;                                                                                             \
    constexpr size_t shift_bits = (ELEMENTS_PER_BYTE == 2) ? 1 : 2; /* log2(ELEMENTS_PER_BYTE) */                        \
    constexpr size_t mask = ELEMENTS_PER_BYTE - 1;                  /* For modulo operation */                           \
    for (size_t m = 0; m < static_cast<size_t>(M); m++) {                                                                \
      for (size_t bd = 0; bd < static_cast<size_t>(K); bd++) {                                                           \
        size_t bd_i = bd >> shift_bits; /* bd / ELEMENTS_PER_BYTE */                                                     \
        size_t bd_j = bd & mask;        /* bd % ELEMENTS_PER_BYTE */                                                     \
        SUB_BYTE_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                                \
        QUANT_FUNC(input, output, output_index, output_index + static_cast<size_t>(N),                                   \
                   scale[bd], CREATE_SUB_BYTE_ZP(SUB_BYTE_TYPE, zp, ELEMENTS_PER_BYTE),                                  \
                   ctx->GetOperatorThreadPool(), half_precision);                                                        \
        input += N;                                                                                                      \
        output_index += static_cast<size_t>(N);                                                                          \
      }                                                                                                                  \
    }                                                                                                                    \
    assert(output_index == static_cast<size_t>(SafeInt<size_t>(M) * K * N));                                             \
  }

DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(Int4x2, ParQuantizeLinearStdS4, 2)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(UInt4x2, ParQuantizeLinearStdU4, 2)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(Int2x4, ParQuantizeLinearStdS2, 4)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(UInt2x4, ParQuantizeLinearStdU2, 4)

// Defines functions to quantize MLFloat16 to sub-byte types.
// This is not an efficient implementation: we allocate a buffer, quantize to the unpacked type, and then clamp/pack
// into output sub-byte buffer.
#define DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(SUB_BYTE_TYPE, ELEMENTS_PER_BYTE)                                         \
  template <>                                                                                                          \
  void ComputeLoop<SUB_BYTE_TYPE, MLFloat16>(OpKernelContext * ctx, const MLFloat16* input, QuantizeLinearScale scale, \
                                             const SUB_BYTE_TYPE* zero_point, SUB_BYTE_TYPE* output, int64_t M,        \
                                             int64_t K, int64_t N, bool saturate, bool half_precision) {               \
    ORT_UNUSED_PARAMETER(saturate);                                                                                    \
                                                                                                                       \
    size_t total_size = static_cast<size_t>(SafeInt<size_t>(M) * K * N);                                               \
    auto tmp_buf = std::make_unique<SUB_BYTE_TYPE::UnpackedType[]>(total_size);                                        \
    size_t tmp_buf_index = 0;                                                                                          \
    constexpr size_t shift_bits = (ELEMENTS_PER_BYTE == 2) ? 1 : 2; /* log2(ELEMENTS_PER_BYTE) */                      \
    constexpr size_t mask = ELEMENTS_PER_BYTE - 1;                  /* For modulo operation */                         \
                                                                                                                       \
    for (size_t m = 0; m < static_cast<size_t>(M); m++) {                                                              \
      for (size_t bd = 0; bd < static_cast<size_t>(K); bd++) {                                                         \
        size_t bd_i = bd >> shift_bits; /* bd / ELEMENTS_PER_BYTE */                                                   \
        size_t bd_j = bd & mask;        /* bd % ELEMENTS_PER_BYTE */                                                   \
        SUB_BYTE_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                              \
        ParQuantizeLinearStd<SUB_BYTE_TYPE::UnpackedType>(input, tmp_buf.get() + tmp_buf_index,                        \
                                                          static_cast<size_t>(N), scale[bd],                           \
                                                          zp, ctx->GetOperatorThreadPool(), half_precision);           \
        input += N;                                                                                                    \
        tmp_buf_index += static_cast<size_t>(N);                                                                       \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    for (size_t i = 0; i < total_size; i++) {                                                                          \
      tmp_buf[i] = std::min<SUB_BYTE_TYPE::UnpackedType>(SUB_BYTE_TYPE::max_val,                                       \
                                                         std::max<SUB_BYTE_TYPE::UnpackedType>(SUB_BYTE_TYPE::min_val, \
                                                                                               tmp_buf[i]));           \
    }                                                                                                                  \
                                                                                                                       \
    size_t num_packed = (total_size + ELEMENTS_PER_BYTE - 1) / ELEMENTS_PER_BYTE;                                      \
    auto dst = gsl::make_span<SUB_BYTE_TYPE>(output, num_packed);                                                      \
    auto src = gsl::make_span<const SUB_BYTE_TYPE::UnpackedType>(tmp_buf.get(), total_size);                           \
    SUB_BYTE_TYPE::Pack(dst, src);                                                                                     \
  }

DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(Int4x2, 2)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(UInt4x2, 2)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(Int2x4, 4)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(UInt2x4, 4)

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t process_block_count;
  int64_t broadcast_dim;
  int64_t process_block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, block_size_,
                process_block_count, broadcast_dim, process_block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
  T* output = y.MutableData<T>();

  const auto precision = precision_ == 0 ? y_scale.GetElementType() : precision_;
  const bool half_precision = precision == ONNX_NAMESPACE::TensorProto::FLOAT16;
  const QuantizeLinearScale scale = y_scale.IsDataType<float>()
                                        ? QuantizeLinearScale(y_scale.Data<float>())
                                        : QuantizeLinearScale(y_scale.Data<MLFloat16>());
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  if constexpr (boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2, Int2x4, UInt2x4>, T>::value) {
    // Every element is written below, so only the unused lanes of a partial last byte need zeroing.
    constexpr size_t elems_per_byte = boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value ? 2 : 4;
    if (static_cast<size_t>(x_shape.Size()) % elems_per_byte != 0) {
      output[y.SizeInBytes() / sizeof(T) - 1] = T{};
    }
  }

  constexpr int output_type_group_ =
      boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value   ? 2
      : boost::mp11::mp_contains<TypeList<Int2x4, UInt2x4>, T>::value ? 3
#if !defined(DISABLE_FLOAT8_TYPES)
      : boost::mp11::mp_contains<element_type_lists::AllFloat8, T>::value ? 1
#endif
      : (std::is_same_v<T, Float6E2M3> || std::is_same_v<T, Float6E3M2>) ? 1
                                                                         : 0;

  if (x.IsDataType<float>()) {
    if (block_size_) {
      if (process_block_size > 1) {
        BlockedQuantizeLinear<float, T, output_type_group_>::opNotLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<float>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(process_block_size),
            static_cast<std::ptrdiff_t>(block_size_),
            128,
            saturate_,
            half_precision);
      } else {
        BlockedQuantizeLinear<float, T, output_type_group_>::opLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<float>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(block_size_),
            saturate_,
            half_precision);
      }
    } else {
      ComputeLoop<T, float>(ctx, x.Data<float>(), scale, zero_point, output,
                            process_block_count, broadcast_dim, process_block_size, saturate_, half_precision);
    }
  } else if (x.IsDataType<MLFloat16>()) {
    if (block_size_) {
      if (process_block_size > 1) {
        BlockedQuantizeLinear<MLFloat16, T, output_type_group_>::opNotLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<MLFloat16>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(process_block_size),
            static_cast<std::ptrdiff_t>(block_size_),
            128,
            saturate_,
            half_precision);
      } else {
        BlockedQuantizeLinear<MLFloat16, T, output_type_group_>::opLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<MLFloat16>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(block_size_),
            saturate_,
            half_precision);
      }
    } else {
      ComputeLoop<T, MLFloat16>(ctx, x.Data<MLFloat16>(), scale, zero_point, output,
                                process_block_count, broadcast_dim, process_block_size, saturate_, half_precision);
    }
  } else {
    ORT_THROW("Unsupported input type.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
