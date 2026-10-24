// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/scatter_nd.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0,
    element_type_lists::All);
}

using EnabledScatterNDDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    13,
    15,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    16,
    17,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterND,
    18,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

template <typename TData>
struct Prepare {
  const TData* input_base;
  TData* output_base;
  uint64_t element_to_copy;
  std::vector<uint64_t> element_offsets;
  bool serialize_updates;

  Prepare() : input_base(nullptr),
              output_base(nullptr),
              element_to_copy(0),
              element_offsets(0),
              serialize_updates(false) {}
};  // struct Prepare

template <typename TData>
Status PrepareForCompute(OpKernelContext* context, Prepare<TData>& p) {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indice_tensor = context->Input<Tensor>(1);
  const auto* update_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indice_shape = indice_tensor->Shape();
  const auto& update_shape = update_tensor->Shape();

  ORT_RETURN_IF_ERROR(ScatterND::ValidateShapes(input_shape, indice_shape, update_shape));

  auto output_tensor = context->Output(0, input_shape);

  const auto* src_base = input_tensor->Data<TData>();
  auto* dst_base = output_tensor->MutableData<TData>();
  const bool is_string_type = input_tensor->IsDataTypeString();

  auto last_indice_dimension = indice_shape[indice_shape.NumDimensions() - 1];

  // Re-use input for output. If input/output Tensor* are the same, do not copy.
  if (src_base != dst_base && input_tensor->Shape().Size() > 0) {
    if (is_string_type) {
      const auto* str_begin = input_tensor->Data<std::string>();
      const std::string* str_end = str_begin + input_shape.Size();
      auto* dst = output_tensor->MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy((void*)dst_base, (const void*)src_base, input_tensor->SizeInBytes());
    }
  }

  std::vector<int64_t> element_counts(onnxruntime::narrow<size_t>(last_indice_dimension), 0LL);  // Number of elements for each input dimension

  TensorPitches input_strides(input_shape);
  for (int64_t i = 0; i < last_indice_dimension; ++i) {
    element_counts[onnxruntime::narrow<size_t>(i)] = input_strides[onnxruntime::narrow<size_t>(i)];
  }

  p.element_to_copy = input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(last_indice_dimension));
  p.serialize_updates = last_indice_dimension == 0;
  const int64_t* indice_offset = indice_tensor->Data<int64_t>();
  auto offset_count = indice_shape.SizeToDimension(indice_shape.NumDimensions() - 1);  // Times to copy
  p.element_offsets.assign(onnxruntime::narrow<size_t>(offset_count), 0LL);

  p.input_base = update_tensor->Data<TData>();
  p.output_base = output_tensor->MutableData<TData>();

  for (int64_t i = 0; i < offset_count; ++i) {
    for (int64_t j = 0; j < last_indice_dimension; ++j) {
      auto indice = *(indice_offset + i * last_indice_dimension + j);

      if (indice >= 0) {
        if (indice >= input_shape[onnxruntime::narrow<size_t>(j)]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", indice);
        }
      } else {
        if (indice < -input_shape[onnxruntime::narrow<size_t>(j)]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", indice);
        } else {
          indice += input_shape[onnxruntime::narrow<size_t>(j)];
        }
      }

      p.element_offsets[onnxruntime::narrow<size_t>(i)] += indice * element_counts[onnxruntime::narrow<size_t>(j)];
    }
  }
  return Status::OK();
}

template <class T>
struct Func_Copy_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    memcpy(a, b, SafeInt<size_t>(element_to_copy) * sizeof(T));
  }
};

template <>
struct Func_Copy_ND<std::string> {
  void operator()(std::string* a, const std::string* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) = (*b++);
  }
};

template <class T>
struct Func_Add_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) += (*b++);
  }
};

template <>
struct Func_Add_ND<bool> {
  void operator()(bool* a, const bool* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) |= (*b++);
  }
};

// MLFloat16 and BFloat16 expose conversions and comparisons but no arithmetic operators,
// so 'add' and 'mul' are evaluated in float and rounded back to half on every update. That
// matches the CUDA and WebGPU kernels, which likewise reduce one update at a time in half
// precision rather than carrying a float accumulator across updates.
// 'min' and 'max' need no specialization: both types have the comparison operators the
// generic functors use.
template <>
struct Func_Add_ND<MLFloat16> {
  void operator()(MLFloat16* a, const MLFloat16* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      *a = MLFloat16(a->ToFloat() + b->ToFloat());
      ++a;
      ++b;
    }
  }
};

template <>
struct Func_Add_ND<BFloat16> {
  void operator()(BFloat16* a, const BFloat16* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      *a = BFloat16(a->ToFloat() + b->ToFloat());
      ++a;
      ++b;
    }
  }
};

template <class T>
struct Func_Mul_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) *= (*b++);
  }
};

template <>
struct Func_Mul_ND<bool> {
  void operator()(bool* a, const bool* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) &= (*b++);
  }
};

template <>
struct Func_Mul_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul_ND<MLFloat16> {
  void operator()(MLFloat16* a, const MLFloat16* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      *a = MLFloat16(a->ToFloat() * b->ToFloat());
      ++a;
      ++b;
    }
  }
};

template <>
struct Func_Mul_ND<BFloat16> {
  void operator()(BFloat16* a, const BFloat16* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      *a = BFloat16(a->ToFloat() * b->ToFloat());
      ++a;
      ++b;
    }
  }
};

template <class T>
struct Func_Min_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      (*a) = (*a) < (*b) ? (*a) : (*b);
      a++;
      b++;
    }
  }
};

template <>
struct Func_Min_ND<bool> {
  void operator()(bool*, const bool*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <class T>
struct Func_Max_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      (*a) = (*a) > (*b) ? (*a) : (*b);
      a++;
      b++;
    }
  }
};

template <>
struct Func_Max_ND<bool> {
  void operator()(bool*, const bool*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <typename TData>
struct ScatterNDDispatchTarget {
  Status operator()(OpKernelContext* context, concurrency::ThreadPool* tp, ScatterND::Reduction reduction) const {
    Prepare<TData> prepare;
    ORT_RETURN_IF_ERROR(PrepareForCompute(context, prepare));
    if (prepare.element_to_copy == 0 || prepare.element_offsets.empty()) {
      return Status::OK();
    }
    if (prepare.serialize_updates) {
      tp = nullptr;
    }

    auto lambda = [&](ptrdiff_t i) {
      switch (reduction) {
        case ScatterND::Reduction::Add: {
          auto func = Func_Add_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + static_cast<int64_t>(i) * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Mul: {
          auto func = Func_Mul_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + static_cast<int64_t>(i) * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Min: {
          auto func = Func_Min_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + static_cast<int64_t>(i) * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Max: {
          auto func = Func_Max_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + static_cast<int64_t>(i) * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        default:
        case ScatterND::Reduction::None: {
          auto func = Func_Copy_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + static_cast<int64_t>(i) * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
      }
    };
    if (context->Input<Tensor>(0)->IsDataTypeString()) {
      tp = nullptr;
    }
    concurrency::ThreadPool::TryParallelFor(
        tp, prepare.element_offsets.size(), static_cast<double>(prepare.element_to_copy),
        [&lambda](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t i = first, end = last; i < end; ++i) {
            lambda(i);
          }
        });
    return Status::OK();
  }
};

Status ScatterND::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto data_type = context->Input<Tensor>(0)->GetElementType();
  utils::MLTypeCallDispatcherFromTypeList<EnabledScatterNDDataTypes> dispatcher{data_type};
  Status status = dispatcher.template InvokeRet<Status, ScatterNDDispatchTarget>(context, tp, this->reduction_);
  return status;
}
}  // namespace onnxruntime
