// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_utils.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/sparse_tensor.h"

#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace sparse_utils {

// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

template <typename T>
inline void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<T*>(dst)[dst_index] = reinterpret_cast<const T*>(src)[src_index];
}

void CopyString(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<std::string*>(dst)[dst_index] = reinterpret_cast<const std::string*>(src)[src_index];
}

template <typename T>
struct NotZero {
  bool operator()(T v) const {
    return v != T{0};
  }
};

template <>
struct NotZero<std::string> {
  bool operator()(const std::string& s) const {
    return !s.empty();
  }
};

#if !defined(ORT_MINIMAL_BUILD)
template <typename T, typename ValueRecorder>
void ScanAndRecordCsr(gsl::span<const T> src_span, int64_t cols,
                      std::vector<int64_t>& inner, std::vector<int64_t>& outer,
                      ValueRecorder recorder) {
  int64_t row = 0;
  int64_t index = 0;
  outer.push_back(0);
  NotZero<T> not_zero;
  for (const auto& v : src_span) {
    auto cur_row = index / cols;
    if (cur_row != row) {
      outer.push_back(static_cast<int64_t>(inner.size()));
      row = cur_row;
    }
    if (not_zero(v)) {
      auto cur_col = index - cur_row * cols;
      inner.push_back(cur_col);
      recorder(v);
    }
    ++index;
  }
  outer.push_back(static_cast<int64_t>(inner.size()));
}

Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src,
                              const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                              SparseTensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const bool is_string = src.IsDataTypeString();

  if (is_string && dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to convert strings tensor to a sparse tensor that not on CPU");
  }

  const IDataTransfer* data_transfer = data_manager.GetDataTransfer(cpu_allocator->Info().device,
                                                                    dst_allocator->Info().device);
  ORT_RETURN_IF_NOT(data_transfer != nullptr, "Unable to find a data transfer for copying from device type: ",
                    cpu_allocator->Info().device.Type(), " to device type: ", dst_allocator->Info().device.Type());

  const auto element_size = src.DataType()->Size();
  gsl::span<const uint8_t> src_span;
  Tensor src_cpu;
  if (src.Location().device.Type() != OrtDevice::CPU) {
    Tensor t(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
    src_cpu = std::move(t);
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src_cpu.DataRaw()), src_cpu.SizeInBytes());
  } else {
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src.DataRaw()), src.SizeInBytes());
  }

  const auto rows = src_dims[0];
  const auto cols = src_dims[1];

  std::vector<int64_t> inner_indices;
  std::vector<int64_t> outer_indices;
  outer_indices.reserve(static_cast<size_t>(rows) + 1);

  std::vector<uint8_t> values_8;
  std::vector<uint16_t> values_16;
  std::vector<uint32_t> values_32;
  std::vector<uint64_t> values_64;
  std::vector<std::reference_wrapper<const std::string>> values_str;
  Tensor nnz_tensor;

  if (is_string) {
    auto str_span = src.DataAsSpan<std::string>();
    ScanAndRecordCsr(str_span, cols, inner_indices, outer_indices,
                     [&](const std::string& s) { values_str.push_back(std::cref(s)); });
  } else {
    switch (element_size) {
      case sizeof(uint8_t): {
        ScanAndRecordCsr(src_span, cols, inner_indices, outer_indices, [&](uint8_t v) { values_8.push_back(v); });
        Tensor t(src.DataType(), {static_cast<int64_t>(values_8.size())}, values_8.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint16_t): {
        // MFFloat16 and BFloat16 are handled fine
        auto span16 = src_span.as_span<const uint16_t>();
        ScanAndRecordCsr(span16, cols, inner_indices, outer_indices, [&](uint16_t v) { values_16.push_back(v); });
        Tensor t(src.DataType(), {static_cast<int64_t>(values_16.size())}, values_16.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint32_t): {
        auto span32 = src_span.as_span<const uint32_t>();
        ScanAndRecordCsr(span32, cols, inner_indices, outer_indices, [&](uint32_t v) { values_32.push_back(v); });
        Tensor t(src.DataType(), {static_cast<int64_t>(values_32.size())}, values_32.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint64_t): {
        auto span64 = src_span.as_span<const uint64_t>();
        ScanAndRecordCsr(span64, cols, inner_indices, outer_indices, [&](uint64_t v) { values_64.push_back(v); });
        Tensor t(src.DataType(), {static_cast<int64_t>(values_64.size())}, values_64.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
    }
  }

  const auto nnz = inner_indices.size();
  const size_t outer_size = (nnz > 0) ? outer_indices.size() : 0U;

  SparseTensor dst_tensor(src.DataType(), src.Shape(), dst_allocator);
  auto mutator = dst_tensor.MakeCsrData(nnz, nnz, outer_size);
  if (nnz > 0) {
    if (is_string) {
      auto dst_span = mutator.Values().MutableDataAsSpan<std::string>();
      std::copy(values_str.cbegin(), values_str.cend(), dst_span.begin());
    } else {
      ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(nnz_tensor, mutator.Values()));
    }
    auto index_type = DataTypeImpl::GetType<int64_t>();
    Tensor inner(index_type, {static_cast<int64_t>(nnz)}, inner_indices.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(inner, mutator.Inner()));
    Tensor outer(index_type, {static_cast<int64_t>(outer_size)},
                 outer_indices.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(outer, mutator.Outer()));
  }

  dst = std::move(dst_tensor);
  return Status::OK();
}

Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                              const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                              Tensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Support 2-D matrices only");
  }

  if (!(src.Format() == SparseFormat::kCsrc)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input must be of CSR format");
  }

  const bool is_string = src.IsDataTypeString();

  if (is_string && dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to convert strings tensor to a sparse tensor that is not on CPU");
  }

  const AllocatorPtr& conversion_allocator = (dst_allocator->Info().device.Type() == OrtDevice::CPU)
                                                 ? dst_allocator
                                                 : cpu_allocator;

  Tensor cpu_result(src.DataType(), src.Shape(), conversion_allocator);
  if (!is_string) {
    memset(cpu_result.MutableDataRaw(), 0, cpu_result.SizeInBytes());
  }

  if (src.NumValues() > 0) {
    const auto rows = src_dims[0];
    const auto cols = src_dims[1];

    {
      auto csr_view = src.AsCsr();
      const auto inner_num = csr_view.Inner().Shape().Size();
      const auto outer_num = csr_view.Outer().Shape().Size();
      ORT_ENFORCE(inner_num == src.Values().Shape().Size(), "Expecting inner indices to be same as nnz. Got: ", inner_num);
      ORT_ENFORCE(outer_num == (rows + 1), "Outer indices must be M + 1. Got: ", outer_num);
    }

    CopyElementFunc copy_func;
    if (is_string) {
      copy_func = CopyString;
    } else {
      const auto element_size = src.DataType()->Size();
      switch (element_size) {
        case sizeof(uint8_t):
          copy_func = CopyElement<uint8_t>;
          break;
        case sizeof(uint16_t):
          copy_func = CopyElement<uint16_t>;
          break;
        case sizeof(uint32_t):
          copy_func = CopyElement<uint32_t>;
          break;
        case sizeof(uint64_t):
          copy_func = CopyElement<uint64_t>;
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
      }
    }

    SparseTensor cpu_src;
    const void* values = nullptr;
    gsl::span<const int64_t> inner_span;
    gsl::span<const int64_t> outer_span;
    if (src.Location().device.Type() != OrtDevice::CPU) {
      SparseTensor t(src.DataType(), src.Shape(), cpu_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
      cpu_src = std::move(t);
      values = cpu_src.Values().DataRaw();
      inner_span = cpu_src.AsCsr().Inner().DataAsSpan<int64_t>();
      outer_span = cpu_src.AsCsr().Outer().DataAsSpan<int64_t>();
    } else {
      values = src.Values().DataRaw();
      inner_span = src.AsCsr().Inner().DataAsSpan<int64_t>();
      outer_span = src.AsCsr().Outer().DataAsSpan<int64_t>();
    }

    void* output = cpu_result.MutableDataRaw();

    size_t src_idx = 0;
    size_t inner_idx = 0;
    for (size_t out_i = 1; out_i < outer_span.size(); ++out_i) {
      auto row_size = outer_span[out_i] - outer_span[out_i - 1];
      for (int64_t cnt = 0; cnt < row_size; ++cnt, ++inner_idx) {
        assert(inner_idx < inner_span.size());
        auto col = inner_span[inner_idx];
        auto dst_idx = (out_i - 1) * cols + col;
        copy_func(output, values, dst_idx, src_idx);
      }
    }
  }

  if (dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    Tensor dest_tensor(src.DataType(), src.Shape(), dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, dest_tensor));
    dst = std::move(dest_tensor);
  } else {
    dst = std::move(cpu_result);
  }

  return Status::OK();
}

Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                              const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator, Tensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  if (!(src.Format() == SparseFormat::kCoo)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input must be of COO format");
  }

  const bool is_string = src.IsDataTypeString();
  if (is_string && dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to convert strings tensor to a sparse tensor that is not on CPU");
  }

  const AllocatorPtr& conversion_allocator = (dst_allocator->Info().device.Type() == OrtDevice::CPU)
                                                 ? dst_allocator
                                                 : cpu_allocator;
  Tensor cpu_result(src.DataType(), src.Shape(), conversion_allocator);
  if (!is_string) {
    memset(cpu_result.MutableDataRaw(), 0, cpu_result.SizeInBytes());
  }

  if (src.NumValues() > 0) {
    const void* values = nullptr;
    const int64_t* indices = nullptr;
    const auto num_values = src.Values().Shape().Size();
    const auto num_indices = src.AsCoo().Index().Shape().Size();
    ORT_RETURN_IF_NOT((num_values == num_indices || 2 * num_values == num_indices), 
      "Expecting indices to be equal the number of values or be twice as many");

    SparseTensor src_cpu;
    if (src.Location().device.Type() != OrtDevice::CPU) {
      SparseTensor t(src.DataType(), src.Shape(), cpu_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
      src_cpu = std::move(t);
      values = src_cpu.Values().DataRaw();
      indices = src_cpu.AsCoo().Index().Data<int64_t>();
    } else {
      values = src.Values().DataRaw();
      indices = src.AsCoo().Index().Data<int64_t>();
    }

    const auto element_size = src.DataType()->Size();
    CopyElementFunc copy_func = nullptr;
    if (src.IsDataTypeString()) {
      copy_func = CopyString;
    } else {
      switch (element_size) {
        case sizeof(uint8_t):
          copy_func = CopyElement<uint8_t>;
          break;
        case sizeof(uint16_t): {
          copy_func = CopyElement<uint16_t>;
        } break;
        case sizeof(uint32_t): {
          copy_func = CopyElement<uint32_t>;
        } break;
        case sizeof(uint64_t): {
          copy_func = CopyElement<uint64_t>;
        } break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
      }
    }

    const auto dense_size = src.Shape().Size();
    void* output = cpu_result.MutableDataRaw();
    // Linear index
    if (num_indices == num_values) {
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto dst_idx = indices[src_idx];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    } else {
      const auto cols = src_dims[1];
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto tuple_idx = src_idx * 2;
        auto dst_idx = indices[tuple_idx] * cols + indices[tuple_idx + 1];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    }
  }

  if (dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    Tensor t(src.DataType(), src.Shape(), dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, t));
    dst = std::move(t);
  } else {
    dst = std::move(cpu_result);
  }

  return Status::OK();
}

#endif  //ORT_MINIMAL_BUILD

template <typename T, typename ValueRecorder>
void ScanAndRecordCoo(gsl::span<const T> src_span,
                      int64_t cols,
                      bool linear,
                      std::vector<int64_t>& indices,
                      ValueRecorder recorder) {
  int64_t index = 0;
  NotZero<T> not_zero;
  for (const auto& v : src_span) {
    if (not_zero(v)) {
      recorder(v);
      if (linear) {
        indices.push_back(index);
      } else {
        auto row = index / cols;
        auto col = index - row * cols;
        indices.push_back(row);
        indices.push_back(col);
      }
    }
    ++index;
   }
 }

Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src,
                              const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_index, SparseTensor& dst) {

  const IDataTransfer* data_transfer = data_manager.GetDataTransfer(cpu_allocator->Info().device,
                                                                    dst_allocator->Info().device);
  ORT_RETURN_IF_NOT(data_transfer != nullptr, "Unable to find a data transfer for copying from device type: ",
                    cpu_allocator->Info().device.Type(), " to device type: ", dst_allocator->Info().device.Type());

  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const bool is_string = src.IsDataTypeString();

  if (is_string && dst_allocator->Info().device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to convert strings tensor to a sparse tensor that is not on CPU");
  }

  gsl::span<const uint8_t> src_span;
  Tensor src_cpu;
  if (src.Location().device.Type() != OrtDevice::CPU) {
    Tensor t(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
    src_cpu = std::move(t);
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src_cpu.DataRaw()), src_cpu.SizeInBytes());
  } else {
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src.DataRaw()), src.SizeInBytes());
  }

  std::vector<int64_t> gathered_indices;
  const auto cols = src_dims[1];
  std::vector<uint8_t> values_8;
  std::vector<uint16_t> values_16;
  std::vector<uint32_t> values_32;
  std::vector<uint64_t> values_64;
  std::vector<std::reference_wrapper<const std::string>> values_str;
  Tensor nnz_tensor;

  if (is_string) {
    auto str_span = src.DataAsSpan<std::string>();
    ScanAndRecordCoo(str_span, cols, linear_index, gathered_indices,
                     [&](const std::string& s) { values_str.push_back(std::cref(s)); });
  } else {
    const auto element_size = src.DataType()->Size();
    switch (element_size) {
      case sizeof(uint8_t): {
        ScanAndRecordCoo(src_span, cols, linear_index, gathered_indices,
                         [&](int8_t v) { values_8.push_back(v); });
        Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_8.size())},
                 values_8.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint16_t): {
        // MFFloat16 and BFloat16 are handled fine
        auto span16 = src_span.as_span<const uint16_t>();
        ScanAndRecordCoo(span16, cols, linear_index, gathered_indices, [&](int16_t v) { values_16.push_back(v); });
        Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_16.size())},
                 values_16.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint32_t): {
        auto span32 = src_span.as_span<const uint32_t>();
        ScanAndRecordCoo(span32, cols, linear_index, gathered_indices, [&](int32_t v) { values_32.push_back(v); });
        Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_32.size())},
                 values_32.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      case sizeof(uint64_t): {
        auto span64 = src_span.as_span<const uint64_t>();
        ScanAndRecordCoo(span64, cols, linear_index, gathered_indices, [&](int64_t v) { values_64.push_back(v); });
        Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_64.size())},
                 values_64.data(), cpu_allocator->Info());
        nnz_tensor = std::move(t);
      } break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
    }
  }

  const auto nnz = (linear_index) ? gathered_indices.size() : gathered_indices.size() / 2;
  assert(static_cast<int64_t>(nnz) == nnz_tensor.Shape().Size() || nnz == values_str.size());

  SparseTensor dst_result(src.DataType(), src.Shape(), dst_allocator);
  auto mutator = dst_result.MakeCooData(nnz, gathered_indices.size());
  if (nnz > 0) {
    if (is_string) {
      auto dst_iter = mutator.Values().MutableData<std::string>();
      std::copy(values_str.cbegin(), values_str.cend(), dst_iter);
    } else {
      ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(nnz_tensor, mutator.Values()));
    }
    Tensor indices_tensor(DataTypeImpl::GetType<int64_t>(), mutator.Index().Shape(), gathered_indices.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(indices_tensor, mutator.Index()));
  }

  dst = std::move(dst_result);

  return Status::OK();
}

}  // namespace sparse_utils
}  // namespace onnxruntime