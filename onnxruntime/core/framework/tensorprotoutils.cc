// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include <memory>
#include <algorithm>
#include <limits>
#include <string>
#include <filesystem>
#if defined(__wasm__)
#include <emscripten.h>
#endif

#include <gsl/gsl>
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/endian_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/platform/path_lib.h"
#include "core/framework/to_tensor_proto_element_type.h"
#include "core/session/ort_apis.h"
#include "onnx/defs/tensor_proto_util.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::utils;

TensorProto ToTensorInitialize(TensorProto_DataType datatype) {
  TensorProto t;
  t.clear_int32_data();
  t.set_data_type(datatype);
  return t;
}

TensorProto ToScalarTensor(TensorProto_DataType datatype, int32_t value) {
  TensorProto t = ToTensorInitialize(datatype);
  t.add_int32_data(value);
  return t;
}

#define TO_TENSOR_ORT_TYPE(TYPE)                                                          \
  template <>                                                                             \
  TensorProto ToTensor<onnxruntime::TYPE>(const onnxruntime::TYPE& value) {               \
    return ToScalarTensor(ToTensorProtoElementType<onnxruntime::TYPE>(), value.val);      \
  }                                                                                       \
  template <>                                                                             \
  TensorProto ToTensor<onnxruntime::TYPE>(const std::vector<onnxruntime::TYPE>& values) { \
    TensorProto t = ToTensorInitialize(ToTensorProtoElementType<onnxruntime::TYPE>());    \
    for (const onnxruntime::TYPE& val : values) {                                         \
      t.add_int32_data(val.val);                                                          \
    }                                                                                     \
    return t;                                                                             \
  }

#define TO_TENSOR_ORT_TYPE_4BIT_TYPE(TYPE)                                                                      \
  template <>                                                                                                   \
  TensorProto ToTensor<onnxruntime::TYPE>(const onnxruntime::TYPE& value) {                                     \
    return ToScalarTensor(ToTensorProtoElementType<onnxruntime::TYPE>(), static_cast<int32_t>(value.ToBits())); \
  }                                                                                                             \
  template <>                                                                                                   \
  TensorProto ToTensor<onnxruntime::TYPE>(const std::vector<onnxruntime::TYPE>& values) {                       \
    TensorProto t = ToTensorInitialize(ToTensorProtoElementType<onnxruntime::TYPE>());                          \
    for (const onnxruntime::TYPE& val : values) {                                                               \
      t.add_int32_data(static_cast<int32_t>(val.ToBits()));                                                     \
    }                                                                                                           \
    return t;                                                                                                   \
  }

// 2-bit types use the same storage pattern as 4-bit types
#define TO_TENSOR_ORT_TYPE_2BIT_TYPE(TYPE)                                                                      \
  template <>                                                                                                   \
  TensorProto ToTensor<onnxruntime::TYPE>(const onnxruntime::TYPE& value) {                                     \
    return ToScalarTensor(ToTensorProtoElementType<onnxruntime::TYPE>(), static_cast<int32_t>(value.ToBits())); \
  }                                                                                                             \
  template <>                                                                                                   \
  TensorProto ToTensor<onnxruntime::TYPE>(const std::vector<onnxruntime::TYPE>& values) {                       \
    TensorProto t = ToTensorInitialize(ToTensorProtoElementType<onnxruntime::TYPE>());                          \
    for (const onnxruntime::TYPE& val : values) {                                                               \
      t.add_int32_data(static_cast<int32_t>(val.ToBits()));                                                     \
    }                                                                                                           \
    return t;                                                                                                   \
  }

namespace ONNX_NAMESPACE {

// Provide template specializations for onnxruntime-specific types.
TO_TENSOR_ORT_TYPE(MLFloat16)
TO_TENSOR_ORT_TYPE(BFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
TO_TENSOR_ORT_TYPE(Float8E4M3FN)
TO_TENSOR_ORT_TYPE(Float8E4M3FNUZ)
TO_TENSOR_ORT_TYPE(Float8E5M2)
TO_TENSOR_ORT_TYPE(Float8E5M2FNUZ)
TO_TENSOR_ORT_TYPE(Float8E8M0)
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
TO_TENSOR_ORT_TYPE_4BIT_TYPE(Float4E2M1x2)
#endif
TO_TENSOR_ORT_TYPE_4BIT_TYPE(Int4x2)
TO_TENSOR_ORT_TYPE_4BIT_TYPE(UInt4x2)

TO_TENSOR_ORT_TYPE_2BIT_TYPE(Int2x4)
TO_TENSOR_ORT_TYPE_2BIT_TYPE(UInt2x4)

bool operator==(const ONNX_NAMESPACE::TensorShapeProto_Dimension& l,
                const ONNX_NAMESPACE::TensorShapeProto_Dimension& r) {
  if (l.has_dim_value()) {
    return r.has_dim_value() && l.dim_value() == r.dim_value();
  } else if (l.has_dim_param()) {
    return r.has_dim_param() && l.dim_param() == r.dim_param() && !l.dim_param().empty();
  } else {
    // l is unknown - has neither dim_value nor dim_param
  }

  return false;
}

bool operator!=(const ONNX_NAMESPACE::TensorShapeProto_Dimension& l,
                const ONNX_NAMESPACE::TensorShapeProto_Dimension& r) {
  return !(l == r);
}

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace {

// This function doesn't support string tensors
static Status UnpackTensorWithRawDataImpl(const void* raw_data, size_t raw_data_len, size_t expected_num_elements,
                                          size_t element_size,
                                          /*out*/ unsigned char* p_data) {
  auto src = gsl::make_span<const unsigned char>(static_cast<const unsigned char*>(raw_data), raw_data_len);
  auto dst = gsl::make_span<unsigned char>(p_data, expected_num_elements * element_size);

  size_t expected_size_in_bytes;
  if (!onnxruntime::IAllocator::CalcMemSizeForArray(expected_num_elements, element_size, &expected_size_in_bytes)) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "size overflow");
  }

  if (dst.size_bytes() != expected_size_in_bytes) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "UnpackTensor: the pre-allocated size does not match the raw data size, expected ",
                           expected_size_in_bytes, ", got ", dst.size_bytes());
  }

  // ReadLittleEndian checks src and dst buffers are the same size
  return onnxruntime::utils::ReadLittleEndian(element_size, src, dst);
}

template <typename T>
Status UnpackTensorWithRawData(const void* raw_data, size_t raw_data_len, size_t expected_num_elements,
                               /*out*/ T* p_data) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

  return UnpackTensorWithRawDataImpl(raw_data, raw_data_len, expected_num_elements, sizeof(T),
                                     reinterpret_cast<unsigned char*>(p_data));
}

#define DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(FOUR_BIT_TYPE, CalcPairFun)                                         \
  template <>                                                                                                            \
  Status UnpackTensorWithRawData<FOUR_BIT_TYPE>(const void* raw_data, size_t raw_data_len, size_t expected_num_elements, \
                                                /*out*/ FOUR_BIT_TYPE* p_data) {                                         \
    static_assert(std::is_trivially_copyable<FOUR_BIT_TYPE>::value, "T must be trivially copyable");                     \
                                                                                                                         \
    ORT_RETURN_IF(nullptr == p_data, "nullptr == p_data");                                                               \
                                                                                                                         \
    size_t num_packed_pairs = FOUR_BIT_TYPE::CalcPairFun(expected_num_elements);                                         \
    ORT_RETURN_IF_NOT(num_packed_pairs == raw_data_len, "Unexpected number of packed int4 pairs");                       \
                                                                                                                         \
    gsl::span<const FOUR_BIT_TYPE> src_span =                                                                            \
        gsl::make_span(reinterpret_cast<const FOUR_BIT_TYPE*>(raw_data), num_packed_pairs);                              \
    gsl::span<FOUR_BIT_TYPE> dst_span = gsl::make_span(p_data, num_packed_pairs);                                        \
                                                                                                                         \
    std::memcpy(dst_span.data(), src_span.data(), num_packed_pairs);                                                     \
                                                                                                                         \
    return Status::OK();                                                                                                 \
  }

DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(Int4x2, CalcNumInt4Pairs)
DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(UInt4x2, CalcNumInt4Pairs)

// 2-bit types use the same pattern - CalcNumInt2Quads gives number of packed bytes
DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(Int2x4, CalcNumInt2Quads)
DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(UInt2x4, CalcNumInt2Quads)

#if !defined(DISABLE_FLOAT4_TYPES)
DEFINE_4BIT_UNPACK_TENSOR_WITH_RAW_DATA_IMPL(Float4E2M1x2, CalcNumFloat4Pairs)
#endif

template <typename FLOAT6_TYPE>
Status UnpackFloat6Tensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                          /*out*/ FLOAT6_TYPE* p_data, size_t expected_num_elems) {
  if (p_data == nullptr) {
    const size_t size = raw_data != nullptr ? raw_data_len : static_cast<size_t>(tensor.int32_data_size());
    return size == 0 ? Status::OK() : Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    const size_t expected_num_bytes = FLOAT6_TYPE::CalcNumFloat6Bytes(expected_num_elems);
    ORT_RETURN_IF_NOT(raw_data_len == expected_num_bytes, "Unexpected number of packed float6 bytes");
    const auto* packed_data = static_cast<const uint8_t*>(raw_data);
    for (size_t i = 0; i < expected_num_elems; ++i) {
      const uint8_t* packed_group = packed_data + (i / 4) * 3;
      uint8_t bits;
      switch (i % 4) {
        case 0:
          bits = packed_group[0] & 0x3F;
          break;
        case 1:
          bits = static_cast<uint8_t>((packed_group[0] >> 6) | ((packed_group[1] & 0x0F) << 2));
          break;
        case 2:
          bits = static_cast<uint8_t>((packed_group[1] >> 4) | ((packed_group[2] & 0x03) << 4));
          break;
        default:
          bits = packed_group[2] >> 2;
          break;
      }
      p_data[i] = FLOAT6_TYPE(bits, FLOAT6_TYPE::FromBits());
    }
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.int32_data_size()) == expected_num_elems,
                    "UnpackTensor: the pre-allocated size does not match the size in proto");
  for (int i = 0; i < tensor.int32_data_size(); ++i) {
    const int32_t value = tensor.int32_data(i);
    ORT_RETURN_IF(value < 0 || value > 0x3F, "Float6 int32_data values must use only bits 0-5");
    p_data[i] = FLOAT6_TYPE(static_cast<uint8_t>(value), FLOAT6_TYPE::FromBits());
  }
  return Status::OK();
}

template <typename FLOAT6_TYPE>
std::string PackFloat6Tensor(const FLOAT6_TYPE* data, size_t num_elements) {
  std::string packed_data(FLOAT6_TYPE::CalcNumFloat6Bytes(num_elements), '\0');
  auto* packed_bytes = reinterpret_cast<uint8_t*>(packed_data.data());
  for (size_t i = 0; i < num_elements; ++i) {
    const uint8_t bits = data[i].ToBits();
    uint8_t* packed_group = packed_bytes + (i / 4) * 3;
    switch (i % 4) {
      case 0:
        packed_group[0] |= bits;
        break;
      case 1:
        packed_group[0] |= bits << 6;
        packed_group[1] |= bits >> 2;
        break;
      case 2:
        packed_group[1] |= bits << 4;
        packed_group[2] |= bits >> 4;
        break;
      default:
        packed_group[2] |= bits << 2;
        break;
    }
  }
  return packed_data;
}

// Read external data for tensor in unint8_t* form and return Status::OK() if the data is read successfully.
// Uses the tensor_proto_dir to construct the full path for external data. If tensor_proto_dir == nullptr
// then uses the current directory instead.
// This function does not unpack string_data of an initializer tensor
Status ReadExternalDataForTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                 const std::filesystem::path& tensor_proto_dir,
                                 std::vector<uint8_t>& unpacked_tensor) {
  ORT_RETURN_IF(utils::HasString(tensor_proto), "This function does not support string data");
  PathString external_file_path;
  onnxruntime::FileOffsetType file_offset;
  SafeInt<size_t> tensor_byte_size;
  ORT_RETURN_IF_ERROR(
      GetExternalDataInfo(tensor_proto, tensor_proto_dir, external_file_path, file_offset, tensor_byte_size));

  if (external_file_path == kTensorProtoNativeEndianMemoryAddressTag) {
    // The external data is in the same memory as the tensor proto.
    // The offset is the address of the data.
    // NOTE: this is an exception: data in memory is already in native endian format
    unpacked_tensor.resize(tensor_byte_size);
    if (tensor_byte_size != 0) {
      std::memcpy(unpacked_tensor.data(), reinterpret_cast<const void*>(file_offset), tensor_byte_size);
    }
    return Status::OK();
  } else if (external_file_path == kTensorProtoLittleEndianMemoryAddressTag) {
    // The external data is in the same memory as the tensor proto.
    // The offset is the address of the data.
    unpacked_tensor.resize(tensor_byte_size);

    size_t element_size = onnxruntime::utils::GetElementSizeOfTensor(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_proto.data_type()));
    auto src_span = gsl::make_span(reinterpret_cast<const unsigned char*>(file_offset), tensor_byte_size);
    auto dst_span = gsl::make_span(reinterpret_cast<unsigned char*>(unpacked_tensor.data()), tensor_byte_size);

    return onnxruntime::utils::ReadLittleEndian(element_size, src_span, dst_span);
  }

  // Validate that the external file is large enough before allocating.
  // This protects against a model with a huge declared shape but a missing/short external file.
  std::error_code fs_error_code{};
  std::uintmax_t file_length = std::filesystem::file_size(external_file_path, fs_error_code);
  ORT_RETURN_IF(fs_error_code, "Failed to get file size for external initializer ", tensor_proto.name(),
                ". std::filesystem error: ", fs_error_code.message(), " (value: ", fs_error_code.value(), ")");
  SafeInt<onnxruntime::FileOffsetType> end_of_read(file_offset);
  end_of_read += tensor_byte_size;
  ORT_RETURN_IF(file_offset < 0 || static_cast<std::uintmax_t>(end_of_read) > file_length,
                "External initializer: ", tensor_proto.name(), " offset: ", file_offset,
                " size to read: ", static_cast<size_t>(tensor_byte_size), " given file_length: ", file_length,
                " are out of bounds or cannot be read in full.");

  unpacked_tensor.resize(tensor_byte_size);

  ORT_RETURN_IF_ERROR(onnxruntime::Env::Default().ReadFileIntoBuffer(
      external_file_path.c_str(),
      file_offset,
      tensor_byte_size,
      gsl::make_span(reinterpret_cast<char*>(unpacked_tensor.data()), tensor_byte_size)));

  if constexpr (endian::native != endian::little) {
    size_t element_size = onnxruntime::utils::GetElementSizeOfTensor(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_proto.data_type()));

    if (element_size > 1) {
      onnxruntime::utils::SwapByteOrderInplace(
          element_size,
          gsl::make_span(reinterpret_cast<std::byte*>(unpacked_tensor.data()), tensor_byte_size));
    }
  }

  return Status::OK();
}

Status TensorProtoToOrtValueImpl(const Env& env, const std::filesystem::path& model_path,
                                 const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer* m,
                                 AllocatorPtr alloc, OrtValue& value) {
  if (m && m->GetBuffer() == nullptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "MemBuffer has not been allocated.");
  }

  // to construct a Tensor with std::string we need to pass an allocator to the Tensor ctor
  // as the contents of each string needs to be allocated and freed separately.
  ONNXTensorElementDataType ele_type = utils::GetTensorElementType(tensor_proto);
  if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING && (m || !alloc)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "string tensor requires allocator to be provided.");
  }

  // Validate data consistency and enforce size limits before allocating memory.
  // This prevents a malicious model from triggering a massive allocation with a
  // large declared shape that has no/insufficient actual data.
  if (!utils::HasExternalData(tensor_proto)) {
    ORT_RETURN_IF_ERROR(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
  }

  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  TensorShape tensor_shape = GetTensorShapeFromTensorProto(tensor_proto);
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();

  std::unique_ptr<Tensor> tensor;

  if (m) {
    tensor = std::make_unique<Tensor>(type, tensor_shape, m->GetBuffer(), m->GetAllocInfo());

    if (tensor->SizeInBytes() > m->GetLen()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The preallocated buffer is too small. Requires ",
                             tensor->SizeInBytes(), ", Got ", m->GetLen());
    }
  } else {
    tensor = std::make_unique<Tensor>(type, tensor_shape, std::move(alloc));
  }

  ORT_RETURN_IF_ERROR(TensorProtoToTensor(env, model_path, tensor_proto, *tensor));

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value.Init(tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  return Status::OK();
}

}  // namespace

namespace utils {

bool HasExternalDataInMemory(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  if (HasExternalData(ten_proto)) {
    for (const auto& entry : ten_proto.external_data()) {
      if (entry.key() == "location") {
        const PathString location = ToWideString(entry.value());
        if (location == kTensorProtoLittleEndianMemoryAddressTag ||
            location == kTensorProtoNativeEndianMemoryAddressTag) {
          return true;
        }
      }
    }
  }

  return false;  // No external data in memory
}

bool HasExternalDataInFile(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return HasExternalData(tensor_proto) && !HasExternalDataInMemory(tensor_proto);
}

Status TensorProtoWithExternalDataToTensorProto(
    const ONNX_NAMESPACE::TensorProto& ten_proto,
    const std::filesystem::path& model_path,
    ONNX_NAMESPACE::TensorProto& new_tensor_proto) {
  // Check if the input tensor has external data
  ORT_RETURN_IF_NOT(HasExternalData(ten_proto), "Input tensor does not have external data.");

  // Copy the metadata from the source tensor to the new tensor
  ONNX_NAMESPACE::TensorProto result;
  result.set_name(ten_proto.name());
  result.set_data_type(ten_proto.data_type());
  result.mutable_dims()->CopyFrom(ten_proto.dims());

  // Strings can only be in memory
  if (utils::HasString(ten_proto)) {
    ORT_RETURN_IF_NOT(HasExternalDataInMemory(ten_proto),
                      "TensorProto with string data can only be in memory");

    std::unique_ptr<onnxruntime::ExternalDataInfo> external_data_info;
    ORT_RETURN_IF_ERROR(onnxruntime::ExternalDataInfo::Create(ten_proto.external_data(), external_data_info));

    // file_offset is address
    auto tensor_shape = utils::GetTensorShapeFromTensorProto(ten_proto);
    std::string* data = reinterpret_cast<std::string*>(external_data_info->GetOffset());
    for (size_t i = 0, lim = narrow<size_t>(tensor_shape.Size()); i < lim; ++i) {
      // set in raw data
      result.add_string_data(*data);
      ++data;
    }
  } else {
    // Load the external data into memory
    std::vector<uint8_t> unpacked_data;
    // ReadExternalDataForTensor expects a directory. Preserve existing behavior for callers that
    // already pass a directory, and only use parent_path() when model_path is a confirmed file.
    std::filesystem::path external_data_path = model_path;
    std::error_code ec;
    if (std::filesystem::is_regular_file(model_path, ec)) {
      external_data_path = model_path.parent_path();
    } else if (ec) {
      ec.clear();
    }

    ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(ten_proto, external_data_path, unpacked_data));
    // Set the raw data in the new tensor
    if (HasExternalDataInMemory(ten_proto) &&
        (ten_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3 ||
         ten_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2)) {
      std::unique_ptr<onnxruntime::ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(onnxruntime::ExternalDataInfo::Create(ten_proto.external_data(), external_data_info));
      if (external_data_info->GetRelPath() == kTensorProtoNativeEndianMemoryAddressTag) {
        const auto* bytes = unpacked_data.data();
        const std::string packed_data = ten_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3
                                            ? PackFloat6Tensor(reinterpret_cast<const Float6E2M3*>(bytes),
                                                               unpacked_data.size())
                                            : PackFloat6Tensor(reinterpret_cast<const Float6E3M2*>(bytes),
                                                               unpacked_data.size());
        onnxruntime::utils::SetRawDataInTensorProto(result, packed_data.data(), packed_data.size());
      } else {
        onnxruntime::utils::SetRawDataInTensorProto(result, unpacked_data.data(), unpacked_data.size());
      }
    } else {
      onnxruntime::utils::SetRawDataInTensorProto(result, unpacked_data.data(), unpacked_data.size());
    }
  }

  new_tensor_proto = std::move(result);

  return Status::OK();
}

// Wraps Env::GetWeaklyCanonicalPath for std::filesystem::path.
static Status WeaklyCanonicalPath(const std::filesystem::path& path, std::filesystem::path& result) {
  PathString canonical_str;
  ORT_RETURN_IF_ERROR(Env::Default().GetWeaklyCanonicalPath(path.native(), canonical_str));
  result = std::filesystem::path(std::move(canonical_str));
  return Status::OK();
}

// Wraps std::filesystem::exists with error_code handling.
static Status PathExists(const std::filesystem::path& path, bool& exists) {
  std::error_code ec;
  exists = std::filesystem::exists(path, ec);
  ORT_RETURN_IF(ec, "Failed to check existence of path: ", path, " - ", ec.message());
  return Status::OK();
}

// Checks whether `path` has the given path prefix.
static bool HasPathComponentPrefix(const std::filesystem::path& prefix, const std::filesystem::path& path) {
  auto [prefix_end, path_it] = std::mismatch(prefix.begin(), prefix.end(), path.begin(), path.end());
  return prefix_end == prefix.end();
}

/// Validates that `external_data_path` is a relative path contained within `model_dir` after symlink resolution,
/// and that the resolved file exists. This is the core security check for EP context cache paths.
///
/// Validation steps:
///   1. Reject empty paths
///   2. Reject internal in-memory reference tags
///   3. Reject absolute paths (including Unix-style '/...' on Windows)
///   4. Skip remaining checks on WASM if no filesystem is available
///   5. Resolve `model_dir / external_data_path` to a canonical path (resolving symlinks for existing segments)
///   6. Verify the canonical path is a prefix-child of the canonical model_dir (containment check)
///   7. Verify the resolved file exists on disk
///
/// This function does NOT handle the symlinked-model fallback — that is the responsibility of
/// ValidateExternalDataPath(), which calls this function as a first pass.
Status ValidateExternalDataPathFromDir(const std::filesystem::path& model_dir,
                                       const std::filesystem::path& external_data_path) {
  // Step 1: Reject empty external data paths.
  ORT_RETURN_IF(external_data_path.empty(), "Empty external data path not allowed");

  // Step 2: Reject internal in-memory reference tags.
  ORT_RETURN_IF(external_data_path.native() == kTensorProtoLittleEndianMemoryAddressTag ||
                    external_data_path.native() == kTensorProtoNativeEndianMemoryAddressTag,
                "In-memory external data reference tag is not a valid file path");

  // Step 3: Reject absolute paths.
  // Use !root_path().empty() to reject paths like '/some/path' even on Windows (where is_absolute()
  // requires a drive letter).
  ORT_RETURN_IF(!external_data_path.root_path().empty(), "Absolute path not allowed for external data location");

#if defined(__wasm__)
  // Step 4 (WASM only): If we can't access the current working directory, assume the WASM environment
  // does not have a virtual filesystem and defer validation to an ExternalDataLoader for the WASM EP.
  std::error_code error_code;
  std::filesystem::current_path(error_code);
  if (error_code) {
    return Status::OK();
  }
#endif

  // Step 5: Resolve both the model directory and the combined path to canonical forms.
  // WeaklyCanonicalPath resolves symlinks for existing path segments while lexically normalizing
  // non-existent trailing segments.
  std::filesystem::path resolved_dir = model_dir.empty() ? std::filesystem::path{"."} : model_dir;

  std::filesystem::path model_dir_canonical;
  std::filesystem::path external_data_path_canonical;
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(resolved_dir, model_dir_canonical));
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(model_dir_canonical / external_data_path, external_data_path_canonical));

  // Step 6: Containment check — verify the resolved external data path starts with the model directory.
  // Step 7: Existence check — verify the file actually exists on disk.
  if (HasPathComponentPrefix(model_dir_canonical, external_data_path_canonical)) {
    bool path_exists = false;
    ORT_RETURN_IF_ERROR(PathExists(external_data_path_canonical, path_exists));
    ORT_RETURN_IF(!path_exists, "External data path does not exist: ", external_data_path_canonical);
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "External data path escapes model directory. ",
                         "External data path: ", external_data_path, " resolved path: ",
                         external_data_path_canonical, " ", "allowed directory: ", resolved_dir);
}

/// Validates that `external_data_path` is a safe relative path under the model's directory.
/// This is the primary entry point for validating external data locations when loading ONNX models.
///
/// Validation flow:
///   1. Try ValidateExternalDataPathFromDir against the model file's parent directory.
///      If it passes, return success.
///   2. If it fails due to an empty/absolute path or an in-memory reference tag, return the error immediately
///      (these are input errors unrelated to the model location).
///   3. If model_path is empty (model loaded from bytes), wrap the error with context.
///   4. If model_path is a symlink, try the symlink fallback:
///      - Resolve the external data path from the *symlink* model directory to its canonical form.
///      - Check if that canonical target is under the *real* (resolved) model directory.
///      This supports Hugging Face Hub local cache layouts where both the model file and
///      external data files are symlinks into a shared blob store:
///        snapshots/v1/model.onnx -> ../../blobs/sha256-abc  (model symlink)
///        snapshots/v1/data.bin   -> ../../blobs/sha256-def  (data symlink)
///      Both symlink targets live under blobs/, which is the real model directory.
///   5. If none of the above succeed, return an error indicating directory escape.
Status ValidateExternalDataPath(const std::filesystem::path& model_path,
                                const std::filesystem::path& external_data_path) {
  // Derive the model directory from the model file path.
  // If model_path is empty (loaded from bytes) or has no directory component (bare filename),
  // fall back to "." (current working directory).
  std::filesystem::path model_dir = model_path.empty() || model_path.parent_path().empty()
                                        ? std::filesystem::path{"."}
                                        : model_path.parent_path();

  // --- Pass 1: Validate against the model file's parent directory ---
  Status status = ValidateExternalDataPathFromDir(model_dir, external_data_path);
  if (status.IsOK()) {
    return status;
  }

  // --- Guard: Don't retry for input-validation errors ---
  // Empty paths, absolute paths, and in-memory reference tags are always invalid regardless of model directory.
  // Return the error directly without misleading "escapes directory" context.
  if (external_data_path.empty() || !external_data_path.root_path().empty() ||
      external_data_path.native() == kTensorProtoLittleEndianMemoryAddressTag ||
      external_data_path.native() == kTensorProtoNativeEndianMemoryAddressTag) {
    return status;
  }

  // --- Empty model_path: model loaded from bytes ---
  // When there's no model file path, the working directory is used as the base.
  // Provide a specific error message indicating this context.
  if (model_path.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "External data path for model loaded from bytes escapes working directory. ",
                           status.ErrorMessage());
  }

  // --- Pass 2: Symlink fallback for symlinked model files ---
  //
  // When model_path is a symlink, the model's *apparent* parent directory (where the symlink lives)
  // differs from its *real* parent directory (where the symlink target lives). External data files
  // may also be symlinks in the apparent directory that resolve to the real directory tree.
  //
  // Example (Hugging Face Hub cache):
  //   apparent dir:  ~/.cache/huggingface/hub/models--foo/snapshots/abc123/
  //   real dir:      ~/.cache/huggingface/hub/models--foo/blobs/
  //   model.onnx -> ../../blobs/sha256-111  (symlink)
  //   weights.bin -> ../../blobs/sha256-222  (symlink)
  //
  // Pass 1 fails because "weights.bin" resolved from apparent_dir points to blobs/sha256-222,
  // which is not under apparent_dir. This fallback checks if it's under real_dir instead.
  std::error_code ec;
  if (!std::filesystem::is_symlink(model_path, ec)) {
    // model_path is not a symlink (or doesn't exist, or we lack permissions).
    // No fallback possible — return the original containment error.
    return status;
  }

  // Resolve the model symlink to get the real model directory.
  std::filesystem::path real_model_path;
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(model_path, real_model_path));
  auto real_model_dir = real_model_path.parent_path();

  // Resolve the external data path from the *apparent* (symlink) model directory.
  // This follows any symlinks in the external data file itself.
  std::filesystem::path external_data_full_path = model_dir / external_data_path;
  std::filesystem::path external_data_canonical;
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(external_data_full_path, external_data_canonical));

  // Check if the resolved external data target is under the real model directory.
  std::filesystem::path real_model_dir_canonical;
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(real_model_dir, real_model_dir_canonical));

  if (HasPathComponentPrefix(real_model_dir_canonical, external_data_canonical)) {
    bool path_exists = false;
    ORT_RETURN_IF_ERROR(PathExists(external_data_canonical, path_exists));
    ORT_RETURN_IF(!path_exists, "External data path does not exist: ", external_data_canonical);
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "External data path escapes model directory. ",
                         "External data path: ", external_data_path, " resolved path: ",
                         external_data_canonical, " ", "allowed directory: ", real_model_dir);
}

[[maybe_unused]] static Status ValidateOpenedExternalDataPath(const std::filesystem::path& model_path,
                                                              const std::filesystem::path& opened_path) {
  const std::filesystem::path model_dir = model_path.empty() || model_path.parent_path().empty()
                                              ? std::filesystem::path{"."}
                                              : model_path.parent_path();

  std::filesystem::path model_dir_canonical;
  ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(model_dir, model_dir_canonical));
  if (HasPathComponentPrefix(model_dir_canonical, opened_path)) {
    return Status::OK();
  }

  std::error_code ec;
  if (!model_path.empty() && std::filesystem::is_symlink(model_path, ec)) {
    std::filesystem::path real_model_path;
    ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(model_path, real_model_path));
    std::filesystem::path real_model_dir_canonical;
    ORT_RETURN_IF_ERROR(WeaklyCanonicalPath(real_model_path.parent_path(), real_model_dir_canonical));
    if (HasPathComponentPrefix(real_model_dir_canonical, opened_path)) {
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "Opened external data file escapes model directory. Opened path: ",
                         opened_path, " allowed directory: ", model_dir_canonical);
}

[[maybe_unused]] static Status OpenValidatedExternalDataFile(
    const Env& env,
    const std::filesystem::path& model_path,
    const PathString& external_data_file_path,
    std::unique_ptr<RandomAccessFile>& external_data_file) {
  ORT_RETURN_IF_ERROR(env.OpenRandomAccessFile(external_data_file_path.c_str(), external_data_file));
  PathString opened_path;
  ORT_RETURN_IF_ERROR(external_data_file->GetCanonicalPath(opened_path));
  return ValidateOpenedExternalDataPath(model_path, std::filesystem::path{opened_path});
}

Status GetExternalDataInfo(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                           const std::filesystem::path& tensor_proto_dir,
                           std::basic_string<ORTCHAR_T>& external_file_path,
                           onnxruntime::FileOffsetType& file_offset,
                           SafeInt<size_t>& tensor_byte_size,
                           ExternalDataInfo::PrepackedInfos* prepacked_infos) {
  ORT_RETURN_IF_NOT(HasExternalData(tensor_proto),
                    "Tensor does not have external data to read from.");

  ORT_RETURN_IF(!HasDataType(tensor_proto) || HasString(tensor_proto),
                "External data type cannot be UNDEFINED or STRING.");

  std::unique_ptr<onnxruntime::ExternalDataInfo> external_data_info;
  ORT_RETURN_IF_ERROR(onnxruntime::ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));

  const auto& location = external_data_info->GetRelPath();

  external_file_path = ((location == kTensorProtoLittleEndianMemoryAddressTag) || (location == kTensorProtoNativeEndianMemoryAddressTag))
                           ? std::filesystem::path(location)
                           : (tensor_proto_dir / location);

  if (tensor_proto.data_type() == TensorProto_DataType_FLOAT6E2M3 ||
      tensor_proto.data_type() == TensorProto_DataType_FLOAT6E3M2) {
    const TensorShape tensor_shape = GetTensorShapeFromTensorProto(tensor_proto);
    ORT_RETURN_IF(tensor_shape.Size() < 0, "External initializer has negative dimensions");
    ORT_RETURN_IF(static_cast<uint64_t>(tensor_shape.Size()) > std::numeric_limits<size_t>::max(),
                  "External initializer size exceeds addressable memory");
    const size_t num_elements = gsl::narrow_cast<size_t>(tensor_shape.Size());
    tensor_byte_size = location == kTensorProtoNativeEndianMemoryAddressTag
                           ? num_elements
                           : Float6E2M3::CalcNumFloat6Bytes(num_elements);
  } else {
    ORT_RETURN_IF_ERROR(GetSizeInBytesFromTensorProto<0>(tensor_proto, &tensor_byte_size));
  }
  const size_t external_data_length = external_data_info->GetLength();
  ORT_RETURN_IF_NOT(external_data_length == 0 || external_data_length == tensor_byte_size,
                    "TensorProto: ", tensor_proto.name(),
                    " external data size mismatch. Computed size: ", *&tensor_byte_size,
                    ", external_data.length: ", external_data_length);

  file_offset = external_data_info->GetOffset();

  if (prepacked_infos != nullptr && external_data_info->HasPrepackedInfo()) {
    *prepacked_infos = external_data_info->TakePrepackedInfos();
  }

  return Status::OK();
}

void SetRawDataInTensorProto(ONNX_NAMESPACE::TensorProto& tensor_proto, std::string&& param) {
  tensor_proto.set_raw_data(std::move(param));
  if constexpr (endian::native != endian::little) {
    utils::ConvertRawDataInTensorProto(tensor_proto);
  }
}

void ConvertRawDataInTensorProto(TensorProto& tensor) {
  size_t element_size = 1;
  void* bytes = NULL;
  size_t num_elements = 0;

  // For some data_type, element size differs for raw data vs
  // data set using the add_<data_type>data() API
  if (HasRawData(tensor)) {
    element_size = GetElementSizeOfTensor(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.data_type()));

    if (element_size <= 1) {
      return;
    }
    num_elements = tensor.raw_data().size() / element_size;
    bytes = tensor.mutable_raw_data()->data();
  } else {  // HasRawData(tensor)

    switch (tensor.data_type()) {
      case TensorProto_DataType_FLOAT:
        bytes = tensor.mutable_float_data()->mutable_data();
        num_elements = tensor.float_data_size();
        element_size = sizeof(float);
        break;

      case TensorProto_DataType_BOOL:
      case TensorProto_DataType_UINT4:
      case TensorProto_DataType_INT4:
      case TensorProto_DataType_UINT2:
      case TensorProto_DataType_INT2:
      case TensorProto_DataType_UINT8:
      case TensorProto_DataType_INT8:
      case TensorProto_DataType_FLOAT8E4M3FN:
      case TensorProto_DataType_FLOAT8E4M3FNUZ:
      case TensorProto_DataType_FLOAT8E5M2:
      case TensorProto_DataType_FLOAT8E5M2FNUZ:
        bytes = tensor.mutable_int32_data()->mutable_data();
        num_elements = tensor.int32_data_size() * (sizeof(int32_t) / sizeof(int8_t));
        element_size = sizeof(int8_t);
        break;

      case TensorProto_DataType_UINT16:
      case TensorProto_DataType_INT16:
      case TensorProto_DataType_FLOAT16:
      case TensorProto_DataType_BFLOAT16:
        bytes = tensor.mutable_int32_data()->mutable_data();
        num_elements = tensor.int32_data_size() * (sizeof(int32_t) / sizeof(int16_t));
        element_size = sizeof(int16_t);
        break;

      case TensorProto_DataType_INT32:
        bytes = tensor.mutable_int32_data()->mutable_data();
        num_elements = tensor.int32_data_size();
        element_size = sizeof(int32_t);
        break;

      // uint32_t is stored in uint64_t
      case TensorProto_DataType_UINT32:
        bytes = tensor.mutable_uint64_data()->mutable_data();
        num_elements = tensor.uint64_data_size() * (sizeof(uint64_t) / sizeof(uint32_t));
        element_size = sizeof(uint32_t);
        break;

      case TensorProto_DataType_UINT64:
        bytes = tensor.mutable_uint64_data()->mutable_data();
        num_elements = tensor.uint64_data_size();
        element_size = sizeof(uint64_t);
        break;

      case TensorProto_DataType_INT64:
        bytes = tensor.mutable_int64_data()->mutable_data();
        num_elements = tensor.int64_data_size();
        element_size = sizeof(int64_t);
        break;

      case TensorProto_DataType_DOUBLE:
        bytes = tensor.mutable_double_data()->mutable_data();
        num_elements = tensor.double_data_size();
        element_size = sizeof(double);
        break;
    }
  }

  gsl::span<std::byte> span = gsl::make_span(reinterpret_cast<std::byte*>(bytes), num_elements * element_size);
  SwapByteOrderInplace(element_size, span);
}

// Bool tensors must hold canonical {0, 1} byte values. Data sourced from raw_data or external
// files is copied verbatim and may contain other non-zero bytes; normalize any non-zero byte to 1
// so every consumer observes a single, consistent value. Operate on the byte representation to
// avoid loading a bool object that does not yet hold a valid value.
static void NormalizeBoolBytes(uint8_t* bool_bytes, size_t num_elements) {
  static_assert(sizeof(bool) == 1, "Normalization assumes 1 byte per bool element");
  for (size_t i = 0; i < num_elements; ++i) {
    bool_bytes[i] = bool_bytes[i] != 0 ? 1 : 0;
  }
}

Status ReadFloat6ExternalData(const Env& env, const ONNX_NAMESPACE::TensorProto& tensor,
                              const std::filesystem::path& tensor_proto_dir,
                              const std::filesystem::path& model_path,
                              std::vector<uint8_t>& data, bool& is_native) {
  PathString external_file_path;
  FileOffsetType file_offset;
  SafeInt<size_t> data_size;
  ORT_RETURN_IF_ERROR(GetExternalDataInfo(tensor, tensor_proto_dir, external_file_path, file_offset, data_size));
  is_native = external_file_path == kTensorProtoNativeEndianMemoryAddressTag;
#if defined(__wasm__)
  ORT_UNUSED_PARAMETER(model_path);
  if (external_file_path != kTensorProtoNativeEndianMemoryAddressTag &&
      external_file_path != kTensorProtoLittleEndianMemoryAddressTag) {
    SafeInt<FileOffsetType> end_of_read(file_offset);
    end_of_read += data_size;
    ORT_RETURN_IF(file_offset < 0 || end_of_read >= 4294967296LL,
                  "External initializer: ", tensor.name(), " offset: ", file_offset,
                  " size to read: ", static_cast<size_t>(data_size), " exceeds WASM file range.");
    ORT_RETURN_IF_ERROR(ValidateWebAssemblyExternalDataRange(external_file_path, file_offset, data_size));
    data.resize(data_size);
    return LoadWebAssemblyExternalData(env, external_file_path, file_offset, data_size,
                                       ExternalDataLoadType::CPU, reinterpret_cast<char*>(data.data()));
  }
#else
  if (external_file_path != kTensorProtoNativeEndianMemoryAddressTag &&
      external_file_path != kTensorProtoLittleEndianMemoryAddressTag) {
    std::unique_ptr<RandomAccessFile> external_data_file;
    ORT_RETURN_IF_ERROR(OpenValidatedExternalDataFile(env, model_path, external_file_path, external_data_file));
    uint64_t file_length = 0;
    ORT_RETURN_IF_ERROR(external_data_file->GetLength(file_length));
    SafeInt<FileOffsetType> end_of_read(file_offset);
    end_of_read += data_size;
    ORT_RETURN_IF(file_offset < 0 || static_cast<std::uintmax_t>(end_of_read) > file_length,
                  "External initializer: ", tensor.name(), " offset: ", file_offset,
                  " size to read: ", static_cast<size_t>(data_size), " given file_length: ", file_length,
                  " are out of bounds or cannot be read in full.");
    data.resize(data_size);
    ORT_RETURN_IF_ERROR(external_data_file->Read(
        file_offset, gsl::make_span(reinterpret_cast<char*>(data.data()), static_cast<size_t>(data_size))));
    return Status::OK();
  }
#endif
  return ReadExternalDataForTensor(tensor, tensor_proto_dir, data);
}

template <typename FLOAT6_TYPE>
Status UnpackFloat6TensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,
                                          const std::filesystem::path& tensor_proto_dir,
                                          size_t expected_num_elements, FLOAT6_TYPE* p_data) {
  std::vector<uint8_t> data;
  bool is_native = false;
  const auto model_path = tensor_proto_dir.empty() ? std::filesystem::path{} : tensor_proto_dir / "model.onnx";
  ORT_RETURN_IF_ERROR(ReadFloat6ExternalData(Env::Default(), tensor, tensor_proto_dir, model_path, data, is_native));
  ORT_RETURN_IF(p_data == nullptr && expected_num_elements != 0, "nullptr == p_data");
  if (is_native) {
    ORT_RETURN_IF(data.size() != expected_num_elements, "Unexpected number of native float6 bytes");
    if (expected_num_elements != 0) {
      std::memcpy(p_data, data.data(), data.size());
    }
    return Status::OK();
  }
  return UnpackFloat6Tensor(tensor, data.data(), data.size(), p_data, expected_num_elements);
}

#if !defined(ORT_MINIMAL_BUILD)

static Status UnpackTensorWithExternalDataImpl(const ONNX_NAMESPACE::TensorProto& tensor,
                                               const std::filesystem::path& tensor_proto_dir,
                                               size_t expected_num_elements, size_t element_size,
                                               /*out*/ unsigned char* p_data) {
  ORT_RETURN_IF(nullptr == p_data, "nullptr == p_data");
  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(tensor, tensor_proto_dir, unpacked_tensor));

  ORT_RETURN_IF_NOT(expected_num_elements * element_size == unpacked_tensor.size(), "Unexpected amount of data");
  // ReadExternalDataForTensor returns data in native endian, no need to byteswap here
  memcpy(p_data, unpacked_tensor.data(), unpacked_tensor.size());

  return Status::OK();
}

template <typename T>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,
                                    const std::filesystem::path& tensor_proto_dir, size_t expected_num_elements,
                                    /*out*/ T* p_data) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

  return UnpackTensorWithExternalDataImpl(tensor, tensor_proto_dir, expected_num_elements, sizeof(T),
                                          reinterpret_cast<unsigned char*>(p_data));
}

// UnpackTensorWithExternalData<bool>
// External data is copied verbatim and may contain bytes outside the canonical {0, 1} set, so
// normalize them (see NormalizeBoolBytes).
template <>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,
                                    const std::filesystem::path& tensor_proto_dir, size_t expected_num_elements,
                                    /*out*/ bool* p_data) {
  ORT_RETURN_IF_ERROR(UnpackTensorWithExternalDataImpl(tensor, tensor_proto_dir, expected_num_elements, sizeof(bool),
                                                       reinterpret_cast<unsigned char*>(p_data)));
  NormalizeBoolBytes(reinterpret_cast<uint8_t*>(p_data), expected_num_elements);
  return Status::OK();
}

#define DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(FOUR_BIT_TYPE, CalcPairFun)                                    \
  template <>                                                                                                       \
  Status UnpackTensorWithExternalData<FOUR_BIT_TYPE>(const ONNX_NAMESPACE::TensorProto& tensor,                     \
                                                     const std::filesystem::path& tensor_proto_dir,                 \
                                                     size_t expected_num_elements, /*out*/ FOUR_BIT_TYPE* p_data) { \
    static_assert(std::is_trivially_copyable<FOUR_BIT_TYPE>::value, "T must be trivially copyable");                \
                                                                                                                    \
    ORT_RETURN_IF(nullptr == p_data, "nullptr == p_data");                                                          \
    std::vector<uint8_t> unpacked_tensor;                                                                           \
    ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(tensor, tensor_proto_dir, unpacked_tensor));                      \
                                                                                                                    \
    size_t num_packed_pairs = FOUR_BIT_TYPE::CalcPairFun(expected_num_elements);                                    \
    ORT_RETURN_IF_NOT(num_packed_pairs == unpacked_tensor.size(), "Unexpected number of packed int4 pairs");        \
                                                                                                                    \
    gsl::span<const FOUR_BIT_TYPE> src_span =                                                                       \
        gsl::make_span(reinterpret_cast<const FOUR_BIT_TYPE*>(unpacked_tensor.data()), num_packed_pairs);           \
    gsl::span<FOUR_BIT_TYPE> dst_span = gsl::make_span(p_data, expected_num_elements);                              \
                                                                                                                    \
    std::memcpy(dst_span.data(), src_span.data(), num_packed_pairs);                                                \
                                                                                                                    \
    return Status::OK();                                                                                            \
  }

DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(Int4x2, CalcNumInt4Pairs)
DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(UInt4x2, CalcNumInt4Pairs)

// 2-bit types
DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(Int2x4, CalcNumInt2Quads)
DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(UInt2x4, CalcNumInt2Quads)

#if !defined(DISABLE_FLOAT4_TYPES)
DEFINE_4BIT_UNPACK_TENSOR_WITH_EXT_DATA_IMPL(Float4E2M1x2, CalcNumFloat4Pairs)
#endif

template <>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,
                                    const std::filesystem::path& tensor_proto_dir,
                                    size_t expected_num_elements, Float6E2M3* p_data) {
  return UnpackFloat6TensorWithExternalData(tensor, tensor_proto_dir, expected_num_elements, p_data);
}

template <>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,
                                    const std::filesystem::path& tensor_proto_dir,
                                    size_t expected_num_elements, Float6E3M2* p_data) {
  return UnpackFloat6TensorWithExternalData(tensor, tensor_proto_dir, expected_num_elements, p_data);
}

#define INSTANTIATE_UNPACK_EXTERNAL_TENSOR(type)                                                                 \
  template Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto&, const std::filesystem::path&, \
                                               size_t, type*);

INSTANTIATE_UNPACK_EXTERNAL_TENSOR(float)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(double)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(uint8_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(int8_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(int16_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(uint16_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(int32_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(int64_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(uint64_t)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(uint32_t)
// bool is intentionally omitted: UnpackTensorWithExternalData<bool> is explicitly specialized
// above (to normalize bytes to {0, 1}), so an explicit instantiation here would have no effect
// and triggers -Werror,-Winstantiation-after-specialization.
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(MLFloat16)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(BFloat16)

#if !defined(DISABLE_FLOAT8_TYPES)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(Float8E4M3FN)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(Float8E4M3FNUZ)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(Float8E5M2)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(Float8E5M2FNUZ)
INSTANTIATE_UNPACK_EXTERNAL_TENSOR(Float8E8M0)
#endif

template <>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& /*tensor*/,
                                    const std::filesystem::path& /*tensor_proto_dir*/, size_t /*expected_num_elements*/,
                                    /*out*/ std::string* /*p_data*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "External data type cannot be STRING.");
}
#endif  //! defined(ORT_MINIMAL_BUILD)

// implementation of type specific unpack of data contained within the TensorProto
template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ T* p_data, size_t expected_num_elements);

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float6E2M3* p_data, size_t expected_num_elems) {
  ORT_RETURN_IF(tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3,
                "Unexpected Float6E2M3 tensor type");
  return UnpackFloat6Tensor(tensor, raw_data, raw_data_len, p_data, expected_num_elems);
}

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float6E3M2* p_data, size_t expected_num_elems) {
  ORT_RETURN_IF(tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2,
                "Unexpected Float6E3M2 tensor type");
  return UnpackFloat6Tensor(tensor, raw_data, raw_data_len, p_data, expected_num_elems);
}

#define DEFINE_UNPACK_TENSOR_IMPL(T, Type, field_name, field_size)                                          \
  template <>                                                                                               \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, \
                      /*out*/ T* p_data, size_t expected_num_elements) {                                    \
    if (nullptr == p_data) {                                                                                \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.field_size();                         \
      if (size == 0)                                                                                        \
        return Status::OK();                                                                                \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                         \
    }                                                                                                       \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                  \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                         \
    }                                                                                                       \
    if (raw_data != nullptr) {                                                                              \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_num_elements, p_data);                \
    }                                                                                                       \
    if (static_cast<size_t>(tensor.field_size()) != expected_num_elements)                                  \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "corrupted protobuf data: tensor shape size(",  \
                             expected_num_elements, ") does not match the data size(", tensor.field_size(), \
                             ") in proto");                                                                 \
    auto& data = tensor.field_name();                                                                       \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                             \
      *p_data++ = static_cast<T>(*data_iter);                                                               \
    return Status::OK();                                                                                    \
  }

// TODO: complex64 complex128
DEFINE_UNPACK_TENSOR_IMPL(float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR_IMPL(double, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR_IMPL(uint8_t, ONNX_NAMESPACE::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR_IMPL(int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR_IMPL(int16_t, ONNX_NAMESPACE::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR_IMPL(uint16_t, ONNX_NAMESPACE::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR_IMPL(int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR_IMPL(int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR_IMPL(uint64_t, ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR_IMPL(uint32_t, ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

//
// Specializations of UnpackTensor that need custom handling for the input type
//

// UnpackTensor<std::string>. Note: doesn't support raw data
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* /*raw_data*/, size_t /*raw_data_len*/,
                    /*out*/ std::string* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0)
      return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_STRING != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (static_cast<size_t>(tensor.string_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  auto& string_data = tensor.string_data();
  for (const auto& iter : string_data) {
    *p_data++ = iter;
  }

  return Status::OK();
}

// UnpackTensor<bool>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ bool* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    ORT_RETURN_IF_ERROR(UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data));
    // raw_data is copied verbatim and may contain bytes outside the canonical {0, 1} set (see
    // NormalizeBoolBytes).
    NormalizeBoolBytes(reinterpret_cast<uint8_t*>(p_data), expected_size);
    return Status::OK();
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");
  for (int iter : tensor.int32_data()) {
    *p_data++ = static_cast<bool>(iter);
  }

  return Status::OK();
}

// UnpackTensor<MLFloat16>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ MLFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = MLFloat16::FromBits(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

// UnpackTensor<BFloat16>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ BFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = BFloat16(static_cast<uint16_t>(v), BFloat16::FromBits());
  }

  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)

// UnpackTensor<Float8E4M3FN>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float8E4M3FN* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float8E4M3FN(static_cast<uint8_t>(v), Float8E4M3FN::FromBits());
  }

  return Status::OK();
}

// UnpackTensor<Float8E4M3FNUZ>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float8E4M3FNUZ* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float8E4M3FNUZ(static_cast<uint8_t>(v), Float8E4M3FNUZ::FromBits());
  }

  return Status::OK();
}

// UnpackTensor<Float8E5M2>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float8E5M2* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float8E5M2(static_cast<uint8_t>(v), Float8E5M2::FromBits());
  }

  return Status::OK();
}

// UnpackTensor<Float8E5M2FNUZ>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float8E5M2FNUZ* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float8E5M2FNUZ(static_cast<uint8_t>(v), Float8E5M2FNUZ::FromBits());
  }

  return Status::OK();
}

// UnpackTensor<Float8E8M0>
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float8E8M0* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float8E8M0(static_cast<uint8_t>(v), Float8E8M0::FromBits());
  }

  return Status::OK();
}

#endif

#define DEFINE_INT4_UNPACK_TENSOR_IMPL(INT4_TYPE, ONNX_INT4_TYPE)                                           \
  template <>                                                                                               \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, \
                      /*out*/ INT4_TYPE* p_data, size_t expected_num_elems) {                               \
    if (nullptr == p_data) {                                                                                \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();                    \
      return size == 0 ? Status::OK() : Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);              \
    }                                                                                                       \
    if (ONNX_NAMESPACE::ONNX_INT4_TYPE != tensor.data_type()) {                                             \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                         \
    }                                                                                                       \
                                                                                                            \
    size_t expected_int4_pairs = INT4_TYPE::CalcNumInt4Pairs(expected_num_elems);                           \
                                                                                                            \
    if (raw_data != nullptr) {                                                                              \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_num_elems, p_data);                   \
    }                                                                                                       \
                                                                                                            \
    ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.int32_data_size()) == expected_int4_pairs,                 \
                      "UnpackTensor: the pre-allocated size does not match the size in proto");             \
                                                                                                            \
    for (int i = 0; i < static_cast<int>(tensor.int32_data_size()); i++) {                                  \
      p_data[i] = INT4_TYPE(static_cast<std::byte>(tensor.int32_data()[i]));                                \
    }                                                                                                       \
                                                                                                            \
    return Status::OK();                                                                                    \
  }

// UnpackTensor<Int4x2>
DEFINE_INT4_UNPACK_TENSOR_IMPL(Int4x2, TensorProto_DataType_INT4)

// UnpackTensor<UInt4x2>
DEFINE_INT4_UNPACK_TENSOR_IMPL(UInt4x2, TensorProto_DataType_UINT4)

// 2-bit type unpack implementation
#define DEFINE_INT2_UNPACK_TENSOR_IMPL(INT2_TYPE, ONNX_INT2_TYPE)                                           \
  template <>                                                                                               \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, \
                      /*out*/ INT2_TYPE* p_data, size_t expected_num_elems) {                               \
    if (nullptr == p_data) {                                                                                \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();                    \
      return size == 0 ? Status::OK() : Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);              \
    }                                                                                                       \
    if (ONNX_NAMESPACE::ONNX_INT2_TYPE != tensor.data_type()) {                                             \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                         \
    }                                                                                                       \
                                                                                                            \
    size_t expected_int2_quads = INT2_TYPE::CalcNumInt2Quads(expected_num_elems);                           \
                                                                                                            \
    if (raw_data != nullptr) {                                                                              \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_num_elems, p_data);                   \
    }                                                                                                       \
                                                                                                            \
    ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.int32_data_size()) == expected_int2_quads,                 \
                      "UnpackTensor: the pre-allocated size does not match the size in proto");             \
                                                                                                            \
    for (int i = 0; i < static_cast<int>(tensor.int32_data_size()); i++) {                                  \
      p_data[i] = INT2_TYPE(static_cast<std::byte>(tensor.int32_data()[i]));                                \
    }                                                                                                       \
                                                                                                            \
    return Status::OK();                                                                                    \
  }

// UnpackTensor<Int2x4>
DEFINE_INT2_UNPACK_TENSOR_IMPL(Int2x4, TensorProto_DataType_INT2)

// UnpackTensor<UInt2x4>
DEFINE_INT2_UNPACK_TENSOR_IMPL(UInt2x4, TensorProto_DataType_UINT2)

#if !defined(DISABLE_FLOAT4_TYPES)

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ Float4E2M1x2* p_data, size_t expected_num_elems) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    return size == 0 ? Status::OK() : Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  size_t expected_float4_pairs = Float4E2M1x2::CalcNumFloat4Pairs(expected_num_elems);

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_num_elems, p_data);
  }

  ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.int32_data_size()) == expected_float4_pairs,
                    "UnpackTensor: the pre-allocated size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint8_t>::max();

  for (int i = 0; i < static_cast<int>(tensor.int32_data_size()); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = Float4E2M1x2(static_cast<uint8_t>(v), Float4E2M1x2::FromBits());
  }

  return Status::OK();
}

#endif

// UnpackTensor from raw data, external data or the type specific data field.
// Uses the model path to construct the full path for loading external data. In case when model_path is empty
// it uses current directory.
template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const std::filesystem::path& model_path,
                    /*out*/ T* p_data, size_t expected_num_elements) {
#if !defined(ORT_MINIMAL_BUILD)
  if (HasExternalData(tensor)) {
    return UnpackTensorWithExternalData(tensor, model_path.parent_path(),
                                        expected_num_elements, p_data);
  }
#else
  ORT_UNUSED_PARAMETER(model_path);
  ORT_RETURN_IF(HasExternalData(tensor), "TensorProto with external data is not supported in ORT minimal build.");
#endif

  return HasRawData(tensor)
             ? UnpackTensor(tensor, tensor.raw_data().data(), tensor.raw_data().size(), p_data, expected_num_elements)
             : UnpackTensor(tensor, nullptr, 0, p_data, expected_num_elements);
}

// instantiate the UnpackTensor variant that supports external data
#define INSTANTIATE_UNPACK_TENSOR(type) \
  template Status UnpackTensor(const ONNX_NAMESPACE::TensorProto&, const std::filesystem::path&, type* p_data, size_t);

INSTANTIATE_UNPACK_TENSOR(float)
INSTANTIATE_UNPACK_TENSOR(double)
INSTANTIATE_UNPACK_TENSOR(uint8_t)
INSTANTIATE_UNPACK_TENSOR(int8_t)
INSTANTIATE_UNPACK_TENSOR(int16_t)
INSTANTIATE_UNPACK_TENSOR(uint16_t)
INSTANTIATE_UNPACK_TENSOR(int32_t)
INSTANTIATE_UNPACK_TENSOR(int64_t)
INSTANTIATE_UNPACK_TENSOR(uint64_t)
INSTANTIATE_UNPACK_TENSOR(uint32_t)
INSTANTIATE_UNPACK_TENSOR(bool)
INSTANTIATE_UNPACK_TENSOR(MLFloat16)
INSTANTIATE_UNPACK_TENSOR(BFloat16)
INSTANTIATE_UNPACK_TENSOR(std::string)

#if !defined(DISABLE_FLOAT8_TYPES)
INSTANTIATE_UNPACK_TENSOR(Float8E4M3FN)
INSTANTIATE_UNPACK_TENSOR(Float8E4M3FNUZ)
INSTANTIATE_UNPACK_TENSOR(Float8E5M2)
INSTANTIATE_UNPACK_TENSOR(Float8E5M2FNUZ)
INSTANTIATE_UNPACK_TENSOR(Float8E8M0)
#endif
INSTANTIATE_UNPACK_TENSOR(Int4x2)
INSTANTIATE_UNPACK_TENSOR(UInt4x2)

INSTANTIATE_UNPACK_TENSOR(Int2x4)
INSTANTIATE_UNPACK_TENSOR(UInt2x4)

#define CASE_PROTO_TRACE(X, Y)                                                                     \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                             \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) {          \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto"); \
    }                                                                                              \
    break;

#define CASE_PROTO_TRACE_INT4(X, Y)                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                         \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(Y::CalcNumInt4Pairs(size), sizeof(Y), out)) { \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");             \
    }                                                                                                          \
    break;

#if !defined(DISABLE_FLOAT4_TYPES)
#define CASE_PROTO_TRACE_FLOAT4(X, Y)                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                           \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(Y::CalcNumFloat4Pairs(size), sizeof(Y), out)) { \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");               \
    }                                                                                                            \
    break;
#endif

// 2-bit types
#define CASE_PROTO_TRACE_INT2(X, Y)                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                         \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(Y::CalcNumInt2Quads(size), sizeof(Y), out)) { \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");             \
    }                                                                                                          \
    break;

template <size_t alignment>
static common::Status GetSizeInBytesFromTensorElemCountAndType(size_t elem_count, int32_t element_type, size_t* out) {
  const size_t size = elem_count;  // Used by CASE_PROTO_TRACE macros
  switch (element_type) {
    CASE_PROTO_TRACE(FLOAT, float);
    CASE_PROTO_TRACE(DOUBLE, double);
    CASE_PROTO_TRACE(BOOL, bool);
    CASE_PROTO_TRACE(INT8, int8_t);
    CASE_PROTO_TRACE(INT16, int16_t);
    CASE_PROTO_TRACE(INT32, int32_t);
    CASE_PROTO_TRACE(INT64, int64_t);
    CASE_PROTO_TRACE(UINT8, uint8_t);
    CASE_PROTO_TRACE(UINT16, uint16_t);
    CASE_PROTO_TRACE(UINT32, uint32_t);
    CASE_PROTO_TRACE(UINT64, uint64_t);
    CASE_PROTO_TRACE(FLOAT16, MLFloat16);
    CASE_PROTO_TRACE(BFLOAT16, BFloat16);
    CASE_PROTO_TRACE(STRING, std::string);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_PROTO_TRACE(FLOAT8E4M3FN, Float8E4M3FN);
    CASE_PROTO_TRACE(FLOAT8E4M3FNUZ, Float8E4M3FNUZ);
    CASE_PROTO_TRACE(FLOAT8E5M2, Float8E5M2);
    CASE_PROTO_TRACE(FLOAT8E5M2FNUZ, Float8E5M2FNUZ);
    CASE_PROTO_TRACE(FLOAT8E8M0, Float8E8M0);
#endif
    CASE_PROTO_TRACE(FLOAT6E2M3, Float6E2M3);
    CASE_PROTO_TRACE(FLOAT6E3M2, Float6E3M2);
    CASE_PROTO_TRACE_INT4(UINT4, UInt4x2);
    CASE_PROTO_TRACE_INT4(INT4, Int4x2);
    CASE_PROTO_TRACE_INT2(UINT2, UInt2x4);
    CASE_PROTO_TRACE_INT2(INT2, Int2x4);

#if !defined(DISABLE_FLOAT4_TYPES)
    CASE_PROTO_TRACE_FLOAT4(FLOAT4E2M1, Float4E2M1x2);
#endif
    default:
      return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

template <size_t alignment>
static common::Status GetSizeInBytesFromTensorShapeAndType(const TensorShape& shape, int32_t element_type, size_t* out) {
  const auto size = narrow<size_t>(shape.Size());
  return GetSizeInBytesFromTensorElemCountAndType<alignment>(size, element_type, out);
}

template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  TensorShape tensor_shape = GetTensorShapeFromTensorProto(tensor_proto);

  bool any_out_of_bounds = std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(),
                                       [](int64_t dim) {
                                         if (dim < 0 ||
                                             static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max()) {
                                           return true;
                                         }
                                         return false;
                                       });

  ORT_RETURN_IF(any_out_of_bounds, "Out of bounds dimensions in TypeProto_Tensor");

  return GetSizeInBytesFromTensorShapeAndType<alignment>(tensor_shape, tensor_proto.data_type(), out);
}

template <size_t alignment>
common::Status GetSizeInBytesFromTensorTypeProto(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto, size_t* out) {
  ORT_RETURN_IF_NOT(HasShape(tensor_proto), "TypeProto_Tensor does not have shape");
  ORT_RETURN_IF_NOT(HasElemType(tensor_proto), "TypeProto_Tensor does not have element type");

  TensorShape tensor_shape = GetTensorShapeFromTensorShapeProto(tensor_proto.shape());

  bool any_out_of_bounds = std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(),
                                       [](int64_t dim) {
                                         return dim < 0 ||
                                                static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max();
                                       });
  ORT_RETURN_IF(any_out_of_bounds, "Out of bounds dimensions in TypeProto_Tensor");

  return GetSizeInBytesFromTensorShapeAndType<alignment>(tensor_shape, tensor_proto.elem_type(), out);
}

template Status GetSizeInBytesFromTensorTypeProto<0>(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto, size_t* out);

common::Status ValidateEmbeddedTensorProtoDataSizeAndShape(
    const ONNX_NAMESPACE::TensorProto& tensor_proto,
    size_t max_embedded_initializer_size_in_bytes) {
  ORT_RETURN_IF(HasExternalData(tensor_proto), "Expected to validate an embedded (non-external) TensorProto");

  TensorShape tensor_shape = GetTensorShapeFromTensorProto(tensor_proto);
  const int64_t num_elems_signed = tensor_shape.Size();  // returns -1 if any dim is negative.

  ORT_RETURN_IF(num_elems_signed < 0, "Initializer '", tensor_proto.name(), "' has negative dimensions");

  // Need to ensure num_elements < SIZE_MAX. This would be an issue in 32-bit platforms.
  ORT_RETURN_IF(static_cast<uint64_t>(num_elems_signed) > std::numeric_limits<size_t>::max(),
                "Initializer '", tensor_proto.name(), "' has a number of elements (", num_elems_signed,
                ") that exceeds SIZE_MAX (", std::numeric_limits<size_t>::max(), ")");

  const size_t num_elems_unsigned = gsl::narrow_cast<size_t>(num_elems_signed);
  size_t byte_size_from_shape = 0;
  if (tensor_proto.data_type() == TensorProto_DataType_COMPLEX64 ||
      tensor_proto.data_type() == TensorProto_DataType_COMPLEX128) {
    const size_t element_size =
        tensor_proto.data_type() == TensorProto_DataType_COMPLEX64 ? 2 * sizeof(float) : 2 * sizeof(double);
    ORT_RETURN_IF_NOT(IAllocator::CalcMemSizeForArray(num_elems_unsigned, element_size, &byte_size_from_shape),
                      "Invalid TensorProto");
  } else {
    ORT_RETURN_IF_ERROR(GetSizeInBytesFromTensorElemCountAndType<0>(num_elems_unsigned, tensor_proto.data_type(),
                                                                    &byte_size_from_shape));
  }
  ORT_RETURN_IF_NOT(byte_size_from_shape <= max_embedded_initializer_size_in_bytes,
                    "Initializer '", tensor_proto.name(), "' declares a size of ", byte_size_from_shape,
                    " bytes which exceeds the ", max_embedded_initializer_size_in_bytes,
                    " byte limit for embedded initializer data. Use external data for large initializers.");

  if (HasString(tensor_proto)) {
    ORT_RETURN_IF(HasRawData(tensor_proto),
                  "Initializer '", tensor_proto.name(), "': string tensor can not have raw data");
    ORT_RETURN_IF_NOT(tensor_proto.string_data_size() == num_elems_signed,
                      "Initializer '", tensor_proto.name(), "': string_data count (", tensor_proto.string_data_size(),
                      ") does not match expected count from shape (",
                      num_elems_signed, ")");

    size_t total_string_storage_size = byte_size_from_shape;
    for (const auto& string_data : tensor_proto.string_data()) {
      ORT_RETURN_IF(string_data.size() > max_embedded_initializer_size_in_bytes - total_string_storage_size,
                    "Initializer '", tensor_proto.name(), "': string_data shape bytes + payload exceeds the ",
                    max_embedded_initializer_size_in_bytes,
                    " byte limit for embedded initializer data. Use external data for large initializers.");
      total_string_storage_size += string_data.size();
    }
  } else if (HasRawData(tensor_proto)) {
    size_t expected_raw_data_size = byte_size_from_shape;
    if (tensor_proto.data_type() == TensorProto_DataType_FLOAT6E2M3 ||
        tensor_proto.data_type() == TensorProto_DataType_FLOAT6E3M2) {
      expected_raw_data_size = Float6E2M3::CalcNumFloat6Bytes(num_elems_unsigned);
    }
    ORT_RETURN_IF_NOT(tensor_proto.raw_data().size() == expected_raw_data_size,
                      "Initializer '", tensor_proto.name(), "': raw_data size (", tensor_proto.raw_data().size(),
                      " bytes) does not match expected size from shape and data type (",
                      expected_raw_data_size, " bytes)");
  } else {
    // Typed data fields. Each data type maps to a specific repeated field in the proto.
    int64_t expected_count = 0;
    int64_t actual_count = 0;

    switch (tensor_proto.data_type()) {
      case TensorProto_DataType_FLOAT:
        expected_count = num_elems_signed;
        actual_count = tensor_proto.float_data_size();
        break;
      case TensorProto_DataType_DOUBLE:
        expected_count = num_elems_signed;
        actual_count = tensor_proto.double_data_size();
        break;
      case TensorProto_DataType_COMPLEX64:
        expected_count = SafeInt<int64_t>(num_elems_signed) * 2;
        actual_count = tensor_proto.float_data_size();
        break;
      case TensorProto_DataType_COMPLEX128:
        expected_count = SafeInt<int64_t>(num_elems_signed) * 2;
        actual_count = tensor_proto.double_data_size();
        break;
      case TensorProto_DataType_INT64:
        expected_count = num_elems_signed;
        actual_count = tensor_proto.int64_data_size();
        break;
      case TensorProto_DataType_UINT64:
      case TensorProto_DataType_UINT32:
        expected_count = num_elems_signed;
        actual_count = tensor_proto.uint64_data_size();
        break;
      case TensorProto_DataType_INT4:
      case TensorProto_DataType_UINT4:
        expected_count = static_cast<int64_t>(Int4x2::CalcNumInt4Pairs(num_elems_unsigned));
        actual_count = tensor_proto.int32_data_size();
        break;
      case TensorProto_DataType_INT2:
      case TensorProto_DataType_UINT2:
        expected_count = static_cast<int64_t>(Int2x4::CalcNumInt2Quads(num_elems_unsigned));
        actual_count = tensor_proto.int32_data_size();
        break;
      case TensorProto_DataType_FLOAT6E2M3:
      case TensorProto_DataType_FLOAT6E3M2:
        expected_count = num_elems_signed;
        actual_count = tensor_proto.int32_data_size();
        break;
#if !defined(DISABLE_FLOAT4_TYPES)
      case TensorProto_DataType_FLOAT4E2M1:
        expected_count = static_cast<int64_t>(Float4E2M1x2::CalcNumFloat4Pairs(num_elems_unsigned));
        actual_count = tensor_proto.int32_data_size();
        break;
#endif
      case TensorProto_DataType_BOOL:
      case TensorProto_DataType_UINT8:
      case TensorProto_DataType_INT8:
      case TensorProto_DataType_UINT16:
      case TensorProto_DataType_INT16:
      case TensorProto_DataType_FLOAT16:
      case TensorProto_DataType_BFLOAT16:
#if !defined(DISABLE_FLOAT8_TYPES)
      case TensorProto_DataType_FLOAT8E4M3FN:
      case TensorProto_DataType_FLOAT8E4M3FNUZ:
      case TensorProto_DataType_FLOAT8E5M2:
      case TensorProto_DataType_FLOAT8E5M2FNUZ:
      case TensorProto_DataType_FLOAT8E8M0:
#endif
      case TensorProto_DataType_INT32:
        // BOOL, INT8, UINT8, INT16, UINT16, FLOAT16, BFLOAT16, INT32, FLOAT8* all use int32_data
        expected_count = num_elems_signed;
        actual_count = tensor_proto.int32_data_size();
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unhandled TensorProto_DataType (", tensor_proto.data_type(),
                               ") in ValidateEmbeddedTensorProtoDataSizeAndShape()");
    }

    ORT_RETURN_IF_NOT(actual_count == expected_count,
                      "Initializer '", tensor_proto.name(),
                      "': data field count (", actual_count,
                      ") does not match expected count from shape (",
                      expected_count, ")");
  }

  return Status::OK();
}

common::Status ValidateEmbeddedTensorProtoDataSizeAndShape(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto, kMaxEmbeddedInitializerSizeInBytes);
}

TensorShape GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  TensorShapeVector tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] =
        HasDimValue(dims[i]) ? dims[i].dim_value() : -1; /* symbolic dimensions are represented as -1 in onnxruntime*/
  }
  return TensorShape(tensor_shape_vec);
}

TensorShape GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  TensorShapeVector tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return TensorShape(tensor_shape_vec);
}

struct UnInitializeParam {
  void* preallocated;
  size_t preallocated_size;
  ONNXTensorElementDataType ele_type;
};

ORT_API_STATUS_IMPL(OrtInitializeBufferForTensor, _In_opt_ void* input, size_t input_len,
                    enum ONNXTensorElementDataType type) {
  OrtStatus* status = nullptr;
  ORT_TRY {
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr)
      return nullptr;
    size_t tensor_size = input_len / sizeof(std::string);
    std::string* ptr = reinterpret_cast<std::string*>(input);
    for (size_t i = 0, n = tensor_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { status = OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); });
  }

  return status;
}

ORT_API(void, OrtUninitializeBuffer, _In_opt_ void* input, size_t input_len, enum ONNXTensorElementDataType type) {
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr)
    return;
  size_t tensor_size = input_len / sizeof(std::string);
  std::string* ptr = reinterpret_cast<std::string*>(input);
  using std::string;
  for (size_t i = 0, n = tensor_size; i < n; ++i) {
    ptr[i].~string();
  }
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif

#if !defined(__wasm__)
static Status GetFileContent(const RandomAccessFile& file, FileOffsetType offset,
                             size_t length, IAllocatorUniquePtr<void>& external_data) {
  // query length if it is 0
  if (length == 0) {
    uint64_t file_length = 0;
    ORT_RETURN_IF_ERROR(file.GetLength(file_length));
    length = SafeInt<size_t>(file_length);
  }

  // first, try to map into memory
  {
    RandomAccessFile::MappedMemoryPtr mapped_memory{};
    auto status = file.Map(offset, length, mapped_memory);
    if (status.IsOK()) {
      const auto mapped_memory_deleter = mapped_memory.get_deleter();
      IAllocatorUniquePtr<void>::deleter_type raw_buffer_deleter =
          [mapped_memory_deleter](void* p) {
            mapped_memory_deleter(static_cast<char*>(p));
          };
      auto* mapped_data = mapped_memory.release();
      IAllocatorUniquePtr<void> raw_buffer(mapped_data, std::move(raw_buffer_deleter));
      external_data.swap(raw_buffer);
      return Status::OK();
    }
  }

  // if that fails, try to copy
  auto buffer = std::make_unique<char[]>(length);
  ORT_RETURN_IF_ERROR(file.Read(offset, gsl::make_span(buffer.get(), length)));

  IAllocatorUniquePtr<void> raw_buffer(buffer.release(), [](void* p) { delete[] reinterpret_cast<char*>(p); });
  external_data.swap(raw_buffer);
  return Status::OK();
}
#endif

// Backstop validation for callers that load external data outside Graph::Resolve (e.g. training
// checkpoints, custom-op initializers). Passes through ORT's in-memory address markers — those are
// validated at higher layers (Graph::ConvertInitializersIntoOrtValues for dense; markers on sparse
// sub-tensors are rejected outright in SparseTensorProtoToDenseTensorProto). For declared file paths,
// defers to ValidateExternalDataPath, which rejects absolute paths and paths that escape the model
// directory. Callers must have already verified the tensor has external data.
static Status ValidateExternalFilePathForTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                const std::filesystem::path& model_path) {
  if (HasExternalDataInMemory(tensor_proto)) {
    return Status::OK();
  }

  std::unique_ptr<ExternalDataInfo> external_data_info;
  ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));
  return utils::ValidateExternalDataPath(model_path, external_data_info->GetRelPath());
}

#if !defined(__wasm__)
static Status LoadPrepackedWeightsFromFile(const RandomAccessFile& file,
                                           std::uintmax_t file_length,
                                           const ExternalDataInfo::PrepackedInfos& prepacked_infos,
                                           PrepackedWeightsForGraph& prepacked_info) {
  for (const auto& [key, blobs] : prepacked_infos) {
    PrePackedWeights prepacked_weights;
    prepacked_weights.buffers_.reserve(blobs.size());
    prepacked_weights.buffer_sizes_.reserve(blobs.size());
    for (const auto& blob : blobs) {
      const auto blob_offset = std::get<0>(blob);
      const auto blob_length = std::get<1>(blob);
      SafeInt<FileOffsetType> end_of_blob{blob_offset};
      end_of_blob += blob_length;
      ORT_RETURN_IF(blob_offset < 0 || static_cast<uintmax_t>(end_of_blob) > file_length,
                    "Pre-packed blob: ", key, " offset: ", blob_offset, " file_length: ", file_length,
                    " is out of bounds and can not read in full");

      IAllocatorUniquePtr<void> data_ptr;
      ORT_RETURN_IF_ERROR(GetFileContent(file, blob_offset, blob_length, data_ptr));
      prepacked_weights.buffers_.push_back(std::move(data_ptr));
      prepacked_weights.buffer_sizes_.push_back(blob_length);
    }
    if (!blobs.empty()) {
      prepacked_info.InsertPrepackedWeights(key, std::move(prepacked_weights));
    }
  }

  return Status::OK();
}
#endif

Status GetExtDataFromTensorProto(const Env& env,
                                 const std::filesystem::path& model_path,
                                 const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                 OrtValue& ort_value, PrepackedWeightsForGraph* prepacked_info) {
  ORT_ENFORCE(HasExternalData(tensor_proto), "TensorProto for: ",
              tensor_proto.name(), "Expected to have external data");

  // Defense-in-depth: reject absolute or directory-escaping external data paths even when this
  // function is reached outside Graph::Resolve (e.g. training checkpoint load, custom-op init).
  // In-memory address markers are passed through; their validity is enforced upstream.
  ORT_RETURN_IF_ERROR(ValidateExternalFilePathForTensor(tensor_proto, model_path));

  if (tensor_proto.data_type() == TensorProto_DataType_FLOAT6E2M3 ||
      tensor_proto.data_type() == TensorProto_DataType_FLOAT6E3M2) {
    const TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
    ORT_RETURN_IF(tensor_shape.Size() < 0, "External initializer has negative dimensions");
    const auto tensor_proto_dir = model_path.empty() ? std::filesystem::path{} : model_path.parent_path();
    std::vector<uint8_t> data;
    bool is_native = false;
    ORT_RETURN_IF_ERROR(ReadFloat6ExternalData(env, tensor_proto, tensor_proto_dir, model_path, data, is_native));
    const size_t num_elements = gsl::narrow_cast<size_t>(tensor_shape.Size());
    const DataTypeImpl* type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
    Tensor tensor{type, tensor_shape, CPUAllocator::DefaultInstance()};
    if (is_native) {
      ORT_RETURN_IF(data.size() != num_elements, "Unexpected number of native float6 bytes");
      if (num_elements != 0) {
        std::memcpy(tensor.MutableDataRaw(), data.data(), data.size());
      }
    } else if (tensor_proto.data_type() == TensorProto_DataType_FLOAT6E2M3) {
      ORT_RETURN_IF_ERROR(UnpackFloat6Tensor(
          tensor_proto, data.data(), data.size(), tensor.MutableData<Float6E2M3>(), num_elements));
    } else {
      ORT_RETURN_IF_ERROR(UnpackFloat6Tensor(
          tensor_proto, data.data(), data.size(), tensor.MutableData<Float6E3M2>(), num_elements));
    }
    if (prepacked_info != nullptr) {
      ORT_RETURN_IF_ERROR(LoadPrepackedWeightsFromExternalData(env, model_path, tensor_proto, *prepacked_info));
    }
    Tensor::InitOrtValue(std::move(tensor), ort_value);
    return Status::OK();
  }

  std::basic_string<ORTCHAR_T> tensor_proto_dir;
  if (!model_path.empty()) {
    ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
  }

  std::basic_string<ORTCHAR_T> external_data_file_path;
  FileOffsetType file_offset;
  SafeInt<size_t> raw_data_safe_len = 0;
  std::optional<ExternalDataInfo::PrepackedInfos> prepacked_infos;
  if (prepacked_info != nullptr) {
    prepacked_infos.emplace();
  }

  ORT_RETURN_IF_ERROR(
      GetExternalDataInfo(tensor_proto, tensor_proto_dir, external_data_file_path, file_offset,
                          raw_data_safe_len, (prepacked_info != nullptr) ? &*prepacked_infos : nullptr));

  TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  MLDataType ml_tensor_type = DataTypeImpl::GetType<Tensor>();
  const auto& name = tensor_proto.name();

  if (external_data_file_path == onnxruntime::utils::kTensorProtoNativeEndianMemoryAddressTag) {
    // the value in location is the memory address of the data
    void* ext_data_buf = reinterpret_cast<void*>(file_offset);
    auto tensor = Tensor{type, tensor_shape, ext_data_buf, OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)};
    ORT_RETURN_IF(raw_data_safe_len != tensor.SizeInBytes(), "Weight: ", name,
                  " kTensorProtoNativeEndianMemoryAddressTag address points to length: ", static_cast<size_t>(raw_data_safe_len),
                  " while shape has bytes size: ", tensor.SizeInBytes());
    Tensor::InitOrtValue(std::move(tensor), ort_value);
  } else if (external_data_file_path == onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag) {
    // the value in location is the memory address of the data
    void* ext_data_buf = reinterpret_cast<void*>(file_offset);
    if constexpr (endian::native != endian::little) {
      auto allocator = CPUAllocator::DefaultInstance();

      auto deleter = [allocator](uint8_t* ptr) { allocator->Free(ptr); };
      std::unique_ptr<uint8_t[], decltype(deleter)> native_data{reinterpret_cast<uint8_t*>(allocator->Alloc(static_cast<size_t>(raw_data_safe_len))), deleter};

      size_t element_size = onnxruntime::utils::GetElementSizeOfTensor(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_proto.data_type()));
      auto src_span = gsl::make_span(reinterpret_cast<const unsigned char*>(ext_data_buf), static_cast<size_t>(raw_data_safe_len));
      auto dst_span = gsl::make_span(reinterpret_cast<unsigned char*>(native_data.get()), static_cast<size_t>(raw_data_safe_len));

      // If element size is unknown, set it to 1 to disable byteswapping
      if (element_size < 1) element_size = 1;

      ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(element_size, src_span, dst_span));

      auto tensor = Tensor{type, tensor_shape, native_data.release(), allocator};
      ORT_RETURN_IF(raw_data_safe_len != tensor.SizeInBytes(), "Weight: ", name,
                    " kTensorProtoLittleEndianMemoryAddressTag address points to length: ", static_cast<size_t>(raw_data_safe_len),
                    " while shape has bytes size: ", tensor.SizeInBytes());
      Tensor::InitOrtValue(std::move(tensor), ort_value);
    } else {
      auto tensor = Tensor{type, tensor_shape, ext_data_buf, OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)};
      ORT_RETURN_IF(raw_data_safe_len != tensor.SizeInBytes(), "Weight: ", name,
                    " kTensorProtoLittleEndianMemoryAddressTag address points to length: ", static_cast<size_t>(raw_data_safe_len),
                    " while shape has bytes size: ", tensor.SizeInBytes());
      Tensor::InitOrtValue(std::move(tensor), ort_value);
    }
  } else {
#if defined(__wasm__)
    ORT_RETURN_IF(file_offset < 0 || file_offset + raw_data_safe_len >= 4294967296,
                  "External initializer: ", tensor_proto.name(), " offset: ", file_offset,
                  " size to read: ", static_cast<size_t>(raw_data_safe_len),
                  " are out of bounds or can not be read in full (>4GB).");

    auto buffer = std::make_unique<char[]>(raw_data_safe_len);
    ORT_RETURN_IF_ERROR(LoadWebAssemblyExternalData(env,
                                                    external_data_file_path,
                                                    file_offset,
                                                    raw_data_safe_len,
                                                    ExternalDataLoadType::CPU,
                                                    buffer.get()));

    auto p_tensor = std::make_unique<Tensor>(type, tensor_shape, buffer.get(),
                                             OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));

    std::function<void(void*)> deleter = [ext_data = buffer.get()](void* t) {
      delete reinterpret_cast<Tensor*>(t);
      delete[] ext_data;
    };

    ort_value.Init(p_tensor.release(), ml_tensor_type, std::move(deleter));
    buffer.release();

#else
    std::unique_ptr<RandomAccessFile> external_data_file;
    ORT_RETURN_IF_ERROR(
        OpenValidatedExternalDataFile(env, model_path, external_data_file_path, external_data_file));

    uint64_t file_length = 0;
    ORT_RETURN_IF_ERROR(external_data_file->GetLength(file_length));

    SafeInt<FileOffsetType> end_of_read(file_offset);
    end_of_read += raw_data_safe_len;
    ORT_RETURN_IF(file_offset < 0 || static_cast<std::uintmax_t>(end_of_read) > file_length,
                  "External initializer: ", tensor_proto.name(), " offset: ", file_offset,
                  " size to read: ", static_cast<size_t>(raw_data_safe_len), " given file_length: ", file_length,
                  " are out of bounds or can not be read in full.");

    IAllocatorUniquePtr<void> ext_data_buf;
    ORT_RETURN_IF_ERROR(GetFileContent(*external_data_file, file_offset, raw_data_safe_len, ext_data_buf));

    // Data on disk is little endian
    if constexpr (endian::native != endian::little) {
      if (type->Size() > 1) {
        gsl::span<std::byte> data_span{reinterpret_cast<std::byte*>(ext_data_buf.get()), raw_data_safe_len};
        SwapByteOrderInplace(type->Size(), data_span);
      }
    }

    auto p_tensor = std::make_unique<Tensor>(type, tensor_shape, ext_data_buf.get(),
                                             OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
    ORT_RETURN_IF(raw_data_safe_len != p_tensor->SizeInBytes(), "Weight: ", name,
                  " External file content has length: ", static_cast<size_t>(raw_data_safe_len),
                  " while shape has bytes size: ", p_tensor->SizeInBytes());

    // Will destroy ext_data as a member of the functor
    // can not move the unique_ptr as it is not copyable
    std::function<void(void*)> deleter = [ext_data = ext_data_buf.get(),
                                          d = ext_data_buf.get_deleter()](void* t) {
      delete reinterpret_cast<Tensor*>(t);
      d(ext_data);
    };

    ort_value.Init(p_tensor.release(), ml_tensor_type, std::move(deleter));
    ext_data_buf.release();

    if (prepacked_info != nullptr && !prepacked_infos->empty()) {
      ORT_RETURN_IF_ERROR(LoadPrepackedWeightsFromFile(
          *external_data_file, file_length, *prepacked_infos, *prepacked_info));
    }
#endif
  }

  return Status::OK();
}

Status LoadPrepackedWeightsFromExternalData(const Env& env,
                                            const std::filesystem::path& model_path,
                                            const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                            PrepackedWeightsForGraph& prepacked_info) {
  ORT_ENFORCE(HasExternalData(tensor_proto), "TensorProto for: ",
              tensor_proto.name(), "Expected to have external data");
  ORT_RETURN_IF_ERROR(ValidateExternalFilePathForTensor(tensor_proto, model_path));

  std::basic_string<ORTCHAR_T> tensor_proto_dir;
  if (!model_path.empty()) {
    ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
  }

  std::basic_string<ORTCHAR_T> external_data_file_path;
  FileOffsetType file_offset;
  SafeInt<size_t> raw_data_safe_len = 0;
  ExternalDataInfo::PrepackedInfos prepacked_infos;
  ORT_RETURN_IF_ERROR(
      GetExternalDataInfo(tensor_proto, tensor_proto_dir, external_data_file_path, file_offset,
                          raw_data_safe_len, &prepacked_infos));
  if (prepacked_infos.empty()) {
    return Status::OK();
  }

#if defined(__wasm__)
  return Status::OK();
#else
  ORT_RETURN_IF(external_data_file_path == kTensorProtoNativeEndianMemoryAddressTag ||
                    external_data_file_path == kTensorProtoLittleEndianMemoryAddressTag,
                "Pre-packed blobs cannot be restored from an in-memory external tensor.");
  std::unique_ptr<RandomAccessFile> external_data_file;
  ORT_RETURN_IF_ERROR(
      OpenValidatedExternalDataFile(env, model_path, external_data_file_path, external_data_file));
  uint64_t file_length = 0;
  ORT_RETURN_IF_ERROR(external_data_file->GetLength(file_length));
  return LoadPrepackedWeightsFromFile(*external_data_file, file_length, prepacked_infos, prepacked_info);
#endif
}

Status LoadExtDataToTensorFromTensorProto(const Env& env, const std::filesystem::path& model_path,
                                          const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                          const IExternalDataLoader& ext_data_loader,
                                          const AllocatorPtr& allocator,
                                          Tensor& tensor) {
#if !defined(ENABLE_D3D12_FILE_LOADING)
  ORT_UNUSED_PARAMETER(allocator);
#endif

  ORT_ENFORCE(HasExternalData(tensor_proto));
  // Defense-in-depth path validation for callers reaching this function outside Graph::Resolve.
  // In-memory markers are passed through; rejected explicitly below as unsupported for this path.
  ORT_RETURN_IF_ERROR(ValidateExternalFilePathForTensor(tensor_proto, model_path));
  std::basic_string<ORTCHAR_T> tensor_proto_dir;
  if (!model_path.empty()) {
    ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
  }
  std::basic_string<ORTCHAR_T> external_data_file_path;
  FileOffsetType file_offset;
  SafeInt<size_t> raw_data_safe_len = 0;
  ORT_RETURN_IF_ERROR(
      GetExternalDataInfo(tensor_proto, tensor_proto_dir, external_data_file_path, file_offset, raw_data_safe_len));

  ORT_RETURN_IF(external_data_file_path == onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag ||
                    external_data_file_path == onnxruntime::utils::kTensorProtoNativeEndianMemoryAddressTag,
                "Memory address tag is not supported by custom external data loader.");
  ORT_RETURN_IF(file_offset < 0, "External initializer: ", tensor_proto.name(),
                " has a negative offset: ", file_offset);

  if (tensor_proto.data_type() == TensorProto_DataType_FLOAT6E2M3 ||
      tensor_proto.data_type() == TensorProto_DataType_FLOAT6E3M2) {
    const size_t expected_packed_size = Float6E2M3::CalcNumFloat6Bytes(static_cast<size_t>(tensor.Shape().Size()));
    ORT_RETURN_IF(static_cast<size_t>(raw_data_safe_len) != expected_packed_size,
                  "External initializer: ", tensor_proto.name(), " packed size does not match the tensor shape.");
    ORT_RETURN_IF(tensor.Location().device.Type() != OrtDevice::CPU,
                  "FP6 external data requires a CPU destination when loaded through IExternalDataLoader.");
    Tensor packed_tensor{DataTypeImpl::GetType<uint8_t>(), TensorShape({static_cast<int64_t>(raw_data_safe_len)}),
                         CPUAllocator::DefaultInstance()};
#if defined(__wasm__)
    ORT_RETURN_IF_ERROR(ext_data_loader.LoadTensor(env, external_data_file_path, file_offset, raw_data_safe_len,
                                                   packed_tensor));
#else
    std::unique_ptr<RandomAccessFile> external_data_file;
    ORT_RETURN_IF_ERROR(
        OpenValidatedExternalDataFile(env, model_path, external_data_file_path, external_data_file));
    ORT_RETURN_IF_ERROR(ext_data_loader.LoadTensor(*external_data_file, file_offset, raw_data_safe_len, packed_tensor));
#endif
    ONNX_NAMESPACE::TensorProto packed_proto = tensor_proto;
    packed_proto.clear_external_data();
    packed_proto.set_data_location(TensorProto_DataLocation_DEFAULT);
    packed_proto.set_raw_data(packed_tensor.DataRaw(), raw_data_safe_len);
    return TensorProtoToTensor(env, model_path, packed_proto, tensor);
  }

  ORT_RETURN_IF(file_offset < 0 || raw_data_safe_len != tensor.SizeInBytes(),
                "External initializer: ", tensor_proto.name(), " offset: ", file_offset,
                " size to read: ", static_cast<size_t>(raw_data_safe_len),
                " does not match the tensor size: ", tensor.SizeInBytes());
#if defined(ENABLE_D3D12_FILE_LOADING)
  if (ext_data_loader.CreatesTensorForDevice(tensor.Location().device)) {
    return ext_data_loader.LoadTensor(env, external_data_file_path, tensor_proto.name(), file_offset,
                                      raw_data_safe_len, allocator, tensor);
  }
#endif

#if defined(__wasm__)
  return ext_data_loader.LoadTensor(env, external_data_file_path, file_offset, raw_data_safe_len, tensor);
#else
  std::unique_ptr<RandomAccessFile> external_data_file;
  ORT_RETURN_IF_ERROR(
      OpenValidatedExternalDataFile(env, model_path, external_data_file_path, external_data_file));
  return ext_data_loader.LoadTensor(*external_data_file, file_offset, raw_data_safe_len, tensor);
#endif
}

#if defined(ENABLE_D3D12_FILE_LOADING)
Status RegisterExternalDataLoadCandidateFromTensorProto(
    const Env& env, const std::filesystem::path& model_path,
    const ONNX_NAMESPACE::TensorProto& tensor_proto,
    const IExternalDataLoader& ext_data_loader) {
  ORT_ENFORCE(HasExternalData(tensor_proto));

  std::basic_string<ORTCHAR_T> tensor_proto_dir;
  if (!model_path.empty()) {
    ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
  }

  std::basic_string<ORTCHAR_T> external_data_file_path;
  FileOffsetType file_offset;
  SafeInt<size_t> raw_data_safe_len = 0;
  ORT_RETURN_IF_ERROR(
      GetExternalDataInfo(tensor_proto, tensor_proto_dir, external_data_file_path, file_offset, raw_data_safe_len));
  ORT_RETURN_IF_ERROR(ValidateExternalFilePathForTensor(tensor_proto, model_path));

  size_t tensor_byte_size = 0;
  ORT_RETURN_IF_ERROR(GetSizeInBytesFromTensorProto<0>(tensor_proto, &tensor_byte_size));
  ORT_RETURN_IF(file_offset < 0 || raw_data_safe_len != tensor_byte_size,
                "External initializer: ", tensor_proto.name(), " offset: ", file_offset,
                " size to read: ", static_cast<size_t>(raw_data_safe_len),
                " does not match the tensor size: ", tensor_byte_size);
  ORT_RETURN_IF(external_data_file_path == onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag ||
                    external_data_file_path == onnxruntime::utils::kTensorProtoNativeEndianMemoryAddressTag,
                "Memory address tag is not supported by custom external data loader.");

  return ext_data_loader.RegisterLoadCandidate(
      env, external_data_file_path, tensor_proto.name(), file_offset,
      raw_data_safe_len);
}
#endif

#define CASE_PROTO(X, Y)                                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                              \
    ORT_RETURN_IF_ERROR(                                                                                            \
        UnpackTensor<Y>(tensor_proto, raw_data, raw_data_len, (Y*)preallocated, static_cast<size_t>(tensor_size))); \
    break;

/**
 * @brief Convert tensor_proto to tensor format and store it to pre-allocated tensor
 * @param env
 * @param model_path
 * @param tensor_proto  tensor data in protobuf format
 * @param tensor        pre-allocated tensor object, where we store the data
 * @return
 */
Status TensorProtoToTensor(const Env& env, const std::filesystem::path& model_path,
                           const ONNX_NAMESPACE::TensorProto& tensor_proto, Tensor& tensor) {
  // Validate tensor compatibility
  TensorShape tensor_shape = GetTensorShapeFromTensorProto(tensor_proto);

  if (std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(),
                  [](int64_t dim) {
                    return dim < 0;
                  })) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "tensor can't contain negative dims");
  }

  if (HasExternalData(tensor_proto)) {
    OrtValue ort_value;
    ORT_RETURN_IF_ERROR(GetExtDataFromTensorProto(env, model_path, tensor_proto, ort_value));
    const auto& ext_tensor = ort_value.Get<Tensor>();
    MakeCpuTensorCopy(ext_tensor, tensor);
    // MakeCpuTensorCopy memcpy's external bytes verbatim. Bool external initializers may carry
    // bytes outside the canonical {0, 1} set, so normalize them here as well (see NormalizeBoolBytes).
    NormalizeBoolTensorIfNeeded(tensor);
    return Status::OK();
  }

  if (tensor_shape != tensor.Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "TensorProtoToTensor() tensor shape mismatch!");
  }

  const DataTypeImpl* const source_type =
      DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  if (source_type->Size() > tensor.DataType()->Size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "TensorProto type ", DataTypeImpl::ToString(source_type),
                           " can not be written into Tensor type ", DataTypeImpl::ToString(tensor.DataType()));
  }

  //  Below we handle the case where TensorProto contains data in itself

  // find raw data in proto buf
  void* raw_data = nullptr;
  SafeInt<size_t> raw_data_len = 0;
  if (utils::HasRawData(tensor_proto)) {
    raw_data = const_cast<char*>(tensor_proto.raw_data().data());
    // TODO The line above has const-correctness issues. Below is a possible fix which copies the tensor_proto data
    //      into a writeable buffer. However, it requires extra memory which may exceed the limit for certain tests.
    // auto buffer = std::make_unique<char[]>(tensor_proto.raw_data().size());
    // std::memcpy(buffer.get(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
    // deleter_for_file_data.d = OrtCallback{DeleteCharArray, buffer.get()};
    // raw_data = buffer.release();
    raw_data_len = tensor_proto.raw_data().size();
  }

  if (nullptr != raw_data && utils::HasString(tensor_proto)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "string tensor can not have raw data");
  }

  // unpacking tensor_proto data to preallocated tensor
  void* preallocated = tensor.MutableDataRaw();
  const int64_t tensor_size = tensor_shape.Size();

  // tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
  if (narrow<uint64_t>(tensor_size) > SIZE_MAX) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
  }

  switch (tensor_proto.data_type()) {
    CASE_PROTO(FLOAT, float);
    CASE_PROTO(DOUBLE, double);
    CASE_PROTO(BOOL, bool);
    CASE_PROTO(INT8, int8_t);
    CASE_PROTO(INT16, int16_t);
    CASE_PROTO(INT32, int32_t);
    CASE_PROTO(INT64, int64_t);
    CASE_PROTO(UINT8, uint8_t);
    CASE_PROTO(UINT16, uint16_t);
    CASE_PROTO(UINT32, uint32_t);
    CASE_PROTO(UINT64, uint64_t);
    CASE_PROTO(FLOAT16, MLFloat16);
    CASE_PROTO(BFLOAT16, BFloat16);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_PROTO(FLOAT8E4M3FN, Float8E4M3FN);
    CASE_PROTO(FLOAT8E4M3FNUZ, Float8E4M3FNUZ);
    CASE_PROTO(FLOAT8E5M2, Float8E5M2);
    CASE_PROTO(FLOAT8E5M2FNUZ, Float8E5M2FNUZ);
    CASE_PROTO(FLOAT8E8M0, Float8E8M0);
#endif
    CASE_PROTO(INT4, Int4x2);
    CASE_PROTO(UINT4, UInt4x2);
    CASE_PROTO(INT2, Int2x4);
    CASE_PROTO(UINT2, UInt2x4);
    CASE_PROTO(FLOAT6E2M3, Float6E2M3);
    CASE_PROTO(FLOAT6E3M2, Float6E3M2);

#if !defined(DISABLE_FLOAT4_TYPES)
    CASE_PROTO(FLOAT4E2M1, Float4E2M1x2);
#endif

    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING:
      ORT_RETURN_IF_ERROR(UnpackTensor<std::string>(tensor_proto, raw_data, raw_data_len,
                                                    static_cast<std::string*>(preallocated),
                                                    static_cast<size_t>(tensor_size)));
      break;
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
  }

  return Status::OK();
}

common::Status CreateTensorFromTensorProto(const Env& env, const std::filesystem::path& model_path,
                                           const ONNX_NAMESPACE::TensorProto& tensor_proto, Tensor& tensor) {
  ORT_RETURN_IF_NOT(utils::HasDataType(tensor_proto), "Initializer must have a datatype");

  // Validate data consistency and enforce size limits before allocating memory.
  if (!utils::HasExternalData(tensor_proto)) {
    ORT_RETURN_IF_ERROR(ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
  }

  auto proto_data_type = tensor_proto.data_type();

  auto proto_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
  Tensor w(DataTypeImpl::TensorTypeFromONNXEnum(proto_data_type)->GetElementType(), proto_shape,
           CPUAllocator::DefaultInstance());
  ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(env, model_path, tensor_proto, w));

  tensor = std::move(w);
  return Status::OK();
}

Status GetTensorProtoWithDataIfInMemory(
    const ONNX_NAMESPACE::TensorProto& tensor_proto, std::unique_ptr<ONNX_NAMESPACE::TensorProto>& result) {
  if (HasExternalDataInMemory(tensor_proto)) {
    result = std::make_unique<ONNX_NAMESPACE::TensorProto>();
    return TensorProtoWithExternalDataToTensorProto(tensor_proto, {}, *result);
  }

  result.reset();
  return Status::OK();
}

Status TensorProtoToOrtValue(const Env& env, const std::filesystem::path& model_path,
                             const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer& m, OrtValue& value) {
  return TensorProtoToOrtValueImpl(env, model_path, tensor_proto, &m, nullptr, value);
}

Status TensorProtoToOrtValue(const Env& env, const std::filesystem::path& model_path,
                             const ONNX_NAMESPACE::TensorProto& tensor_proto, AllocatorPtr alloc, OrtValue& value) {
  return TensorProtoToOrtValueImpl(env, model_path, tensor_proto, nullptr, alloc, value);
}

#define CASE_TYPE(X)                             \
  case ONNX_NAMESPACE::TensorProto_DataType_##X: \
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_##X;

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type) {
  switch (type) {
    CASE_TYPE(FLOAT)
    CASE_TYPE(UINT8)
    CASE_TYPE(INT8)
    CASE_TYPE(UINT16)
    CASE_TYPE(INT16)
    CASE_TYPE(INT32)
    CASE_TYPE(INT64)
    CASE_TYPE(STRING)
    CASE_TYPE(BOOL)
    CASE_TYPE(FLOAT16)
    CASE_TYPE(DOUBLE)
    CASE_TYPE(UINT32)
    CASE_TYPE(UINT64)
    CASE_TYPE(COMPLEX64)
    CASE_TYPE(COMPLEX128)
    CASE_TYPE(BFLOAT16)
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_TYPE(FLOAT8E4M3FN)
    CASE_TYPE(FLOAT8E4M3FNUZ)
    CASE_TYPE(FLOAT8E5M2)
    CASE_TYPE(FLOAT8E5M2FNUZ)
    CASE_TYPE(FLOAT8E8M0)
#endif
    CASE_TYPE(UINT4)
    CASE_TYPE(INT4)
    CASE_TYPE(UINT2)
    CASE_TYPE(INT2)

#if !defined(DISABLE_FLOAT4_TYPES)
    CASE_TYPE(FLOAT4E2M1)
#endif
    CASE_TYPE(FLOAT6E2M3)
    CASE_TYPE(FLOAT6E3M2)

    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return CApiElementTypeFromProtoType(tensor_proto.data_type());
}

ONNX_NAMESPACE::TensorProto TensorToTensorProto(const Tensor& tensor,
                                                const std::string& tensor_proto_name,
                                                bool use_tensor_buffer) {
  // Set name, dimensions, type, and data of the TensorProto.
  ONNX_NAMESPACE::TensorProto tensor_proto;

  tensor_proto.set_name(tensor_proto_name);

  for (auto& dim : tensor.Shape().GetDims()) {
    tensor_proto.add_dims(dim);
  }

  tensor_proto.set_data_type(tensor.GetElementType());
  // String tensors cannot use the external data in-memory optimization because their raw buffer
  // contains std::string objects (with internal pointers), not serializable string content.
  if (use_tensor_buffer && !tensor.IsDataTypeString() &&
      tensor.GetElementType() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3 &&
      tensor.GetElementType() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2 &&
      tensor.SizeInBytes() > kSmallTensorExternalDataThreshold) {
    // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/graph_flatbuffers_utils.cc#L302
    const auto* raw_data = tensor.DataRaw();
    ORT_ENFORCE(raw_data, "Missing raw data for tensor proto. Invalid tensor.");
    static_assert(sizeof(void*) <= sizeof(ExternalDataInfo::OFFSET_TYPE));

    // we reinterpret_cast this back to void* in tensorprotoutils.cc:GetExtDataFromTensorProto.
    // use intptr_t as OFFSET_TYPE is signed. in theory you could get a weird looking value if the address uses the
    // high bit, but that should be unlikely in a scenario where we care about memory usage enough to use this path.
    auto offset = narrow<ExternalDataInfo::OFFSET_TYPE>(reinterpret_cast<intptr_t>(raw_data));

    ExternalDataInfo::SetExternalLocationToProto(onnxruntime::utils::kTensorProtoNativeEndianMemoryAddressTag,
                                                 offset, tensor.SizeInBytes(), tensor_proto);

  } else {
    if (tensor.IsDataTypeString()) {
      auto* mutable_string_data = tensor_proto.mutable_string_data();
      auto f = tensor.Data<std::string>();
      auto end = f + tensor.Shape().Size();
      for (; f < end; ++f) {
        *mutable_string_data->Add() = *f;
      }
    } else if (tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3) {
      *tensor_proto.mutable_raw_data() = PackFloat6Tensor(tensor.Data<Float6E2M3>(),
                                                          gsl::narrow_cast<size_t>(tensor.Shape().Size()));
    } else if (tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2) {
      *tensor_proto.mutable_raw_data() = PackFloat6Tensor(tensor.Data<Float6E3M2>(),
                                                          gsl::narrow_cast<size_t>(tensor.Shape().Size()));
    } else {
      SetRawDataInTensorProto(tensor_proto, tensor.DataRaw(), tensor.SizeInBytes());
    }
  }

  return tensor_proto;
}

ONNX_NAMESPACE::TypeProto TypeProtoFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  TypeProto type_proto;

  type_proto.mutable_tensor_type()->set_elem_type(tensor_proto.data_type());
  auto shape = type_proto.mutable_tensor_type()->mutable_shape();

  for (auto dim : tensor_proto.dims()) {
    shape->add_dim()->set_dim_value(dim);
  }

  return type_proto;
}

common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              const std::filesystem::path& model_path,
                                              ONNX_NAMESPACE::TensorProto& tensor, const std::string& tensor_name) {
  ORT_RETURN_IF_NOT(node.attribute_size() > 0, "Constant node: ", node.name(), " has no data attributes");

  const AttributeProto& constant_attribute = node.attribute(0);

  switch (constant_attribute.type()) {
    case AttributeProto_AttributeType_TENSOR:
      // Defense-in-depth: reject ORT in-memory address markers on a Constant node's dense tensor
      // attribute. These markers are an internal ORT sentinel for trusted in-memory buffers and must
      // never appear in a deserialized protobuf. The Graph constructor also checks all initializers
      // (including those converted from Constant nodes), but we block early here to prevent the
      // crafted tensor from propagating further.
      ORT_RETURN_IF(HasExternalDataInMemory(constant_attribute.t()),
                    "Constant node '", node.name(),
                    "' tensor attribute references an ORT in-memory address marker, "
                    "which is not allowed in a model protobuf.");
      tensor = constant_attribute.t();
      break;
    case AttributeProto_AttributeType_FLOAT:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      tensor.add_float_data(constant_attribute.f());
      break;
    case AttributeProto_AttributeType_FLOATS:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      *tensor.mutable_float_data() = constant_attribute.floats();
      tensor.add_dims(constant_attribute.floats().size());
      break;
    case AttributeProto_AttributeType_INT:
      tensor.set_data_type(TensorProto_DataType_INT64);
      tensor.add_int64_data(constant_attribute.i());
      break;
    case AttributeProto_AttributeType_INTS:
      tensor.set_data_type(TensorProto_DataType_INT64);
      *tensor.mutable_int64_data() = constant_attribute.ints();
      tensor.add_dims(constant_attribute.ints().size());
      break;
    case AttributeProto_AttributeType_STRING:
      tensor.set_data_type(TensorProto_DataType_STRING);
      tensor.add_string_data(constant_attribute.s());
      break;
    case AttributeProto_AttributeType_STRINGS: {
      tensor.set_data_type(TensorProto_DataType_STRING);
      *tensor.mutable_string_data() = constant_attribute.strings();
      tensor.add_dims(constant_attribute.strings().size());
      break;
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    case AttributeProto_AttributeType_SPARSE_TENSOR: {
      auto& s = constant_attribute.sparse_tensor();
      ORT_RETURN_IF_ERROR(SparseTensorProtoToDenseTensorProto(s, model_path, tensor));
      break;
    }
#else
      ORT_UNUSED_PARAMETER(model_path);
#endif
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Unsupported attribute value type of ",
                             constant_attribute.type(), " in 'Constant' node '", node.name(), "'");
  }

  // set name last in case attribute type was tensor (would copy over name)
  *(tensor.mutable_name()) = tensor_name;

  return Status::OK();
}

common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              const std::filesystem::path& model_path,
                                              ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_RETURN_IF_NOT(node.output_size() == 1, "NodeProto for Constant should have 1 output. Got:", node.output_size());
  return ConstantNodeProtoToTensorProto(node, model_path, tensor, node.output(0));
}

void MakeCpuTensorCopy(const Tensor& src_tensor, Tensor& dst_tensor) {
  if (src_tensor.IsDataTypeString()) {
    auto src_span = src_tensor.DataAsSpan<std::string>();
    std::copy(src_span.begin(), src_span.end(), dst_tensor.MutableDataAsSpan<std::string>().begin());
  } else {
    std::memcpy(dst_tensor.MutableDataRaw(), src_tensor.DataRaw(), src_tensor.SizeInBytes());
  }
}

void NormalizeBoolTensorIfNeeded(Tensor& tensor) {
  if (tensor.IsDataType<bool>()) {
    NormalizeBoolBytes(reinterpret_cast<uint8_t*>(tensor.MutableDataRaw()),
                       narrow<size_t>(tensor.Shape().Size()));
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)

// Validates the external data declaration on a sub-tensor of a SparseTensorProto (values or
// indices). Validates that any file path stays within the model directory.
//
// Gates on data_location == EXTERNAL (rather than HasExternalData()) so that path validation
// runs even when data_type is UNDEFINED. A malicious model could set data_location=EXTERNAL with
// data_type=UNDEFINED and an evil file path; downstream loading would also reject it, but we
// validate here for defense-in-depth.
//
// In-memory address markers must never appear on sparse sub-tensors. The trusted .ort loader
// materializes sparse sub-tensors as inline raw_data (see LoadSparseInitializerOrtFormat); the
// untrusted .onnx protobuf path rejects markers at the Graph constructor; and
// SparseTensorProtoToDenseTensorProto re-asserts the invariant before this function is reached.
// The HasExternalDataInMemory early-return below is a paranoid backstop.
static Status ValidateSparseSubTensorExternalDataPath(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                      const std::filesystem::path& model_path) {
  if (tensor_proto.data_location() != ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL ||
      HasExternalDataInMemory(tensor_proto)) {
    return Status::OK();
  }

  std::unique_ptr<ExternalDataInfo> external_data_info;
  ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));
  return utils::ValidateExternalDataPath(model_path, external_data_info->GetRelPath());
}

static Status CopySparseData(const std::string& name,
                             int64_t nnz_elements,
                             const ONNX_NAMESPACE::TensorProto& indices,
                             const std::filesystem::path& model_path,
                             gsl::span<const int64_t> dense_dims,
                             int64_t dense_elements,
                             std::function<void(size_t from_idx, size_t to_idx)> copier) {
  Status status = Status::OK();
  TensorShape indices_shape(indices.dims().data(), indices.dims().size());
  const int64_t indices_elements = indices_shape.Size();

  InlinedVector<int64_t> indices_values;  // used for conversion of smaller size indices
  std::vector<uint8_t> unpack_buffer;
  gsl::span<const int64_t> indices_data;
  const bool needs_unpack = utils::HasRawData(indices) || utils::HasExternalData(indices);
  const auto append_indices = [&indices_values, indices_elements](auto first, auto last) {
    indices_values.reserve(narrow<size_t>(indices_elements));
    for (; first != last; ++first) {
      indices_values.push_back(*first);
    }
  };
  switch (indices.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      if (needs_unpack) {
        // For inline raw_data, validate size before unpacking to avoid a large allocation from a
        // malformed tensor with small indices shape but oversized raw_data. For external data,
        // raw_data is empty so we can only validate after unpacking.
        if (!utils::HasExternalData(indices)) {
          ORT_RETURN_IF_NOT(indices.raw_data().size() == SafeInt<size_t>(indices_elements) * sizeof(int64_t),
                            "Sparse tensor: ", name, " indices raw data size does not match expected: ",
                            indices_elements * sizeof(int64_t));
        }
        ORT_RETURN_IF_ERROR(UnpackInitializerData(indices, model_path, unpack_buffer));
        ORT_RETURN_IF_NOT(unpack_buffer.size() == SafeInt<size_t>(indices_elements) * sizeof(int64_t),
                          "Sparse tensor: ", name, " indices data size does not match expected: ",
                          indices_elements * sizeof(int64_t));
        indices_data = ReinterpretAsSpan<const int64_t>(gsl::make_span(unpack_buffer));
      } else {
        ORT_RETURN_IF_NOT(indices.int64_data_size() == indices_elements,
                          "Sparse tensor: ", name, " indices int64 data size does not match expected: ",
                          indices_elements);
        indices_data = gsl::make_span(indices.int64_data().data(), narrow<size_t>(indices_elements));
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      if (needs_unpack) {
        if (!utils::HasExternalData(indices)) {
          ORT_RETURN_IF_NOT(indices.raw_data().size() == SafeInt<size_t>(indices_elements) * sizeof(int32_t),
                            "Sparse tensor: ", name, " indices raw data size does not match expected: ",
                            indices_elements * sizeof(int32_t));
        }
        ORT_RETURN_IF_ERROR(UnpackInitializerData(indices, model_path, unpack_buffer));
        ORT_RETURN_IF_NOT(unpack_buffer.size() == SafeInt<size_t>(indices_elements) * sizeof(int32_t),
                          "Sparse tensor: ", name, " indices data size does not match expected: ",
                          indices_elements * sizeof(int32_t));
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpack_buffer));
        append_indices(int32_span.begin(), int32_span.end());
        unpack_buffer.clear();
        unpack_buffer.shrink_to_fit();
      } else {
        ORT_RETURN_IF_NOT(indices.int32_data_size() == indices_elements,
                          "Sparse tensor: ", name, " indices int32 data size does not match expected: ",
                          indices_elements);
        append_indices(indices.int32_data().cbegin(), indices.int32_data().cend());
      }
      indices_data = gsl::make_span(indices_values);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      if (needs_unpack) {
        if (!utils::HasExternalData(indices)) {
          ORT_RETURN_IF_NOT(indices.raw_data().size() == SafeInt<size_t>(indices_elements) * sizeof(int16_t),
                            "Sparse tensor: ", name, " indices raw data size does not match expected: ",
                            indices_elements * sizeof(int16_t));
        }
        ORT_RETURN_IF_ERROR(UnpackInitializerData(indices, model_path, unpack_buffer));
        ORT_RETURN_IF_NOT(unpack_buffer.size() == SafeInt<size_t>(indices_elements) * sizeof(int16_t),
                          "Sparse tensor: ", name, " indices data size does not match expected: ",
                          indices_elements * sizeof(int16_t));
        auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpack_buffer));
        append_indices(int16_span.begin(), int16_span.end());
        unpack_buffer.clear();
        unpack_buffer.shrink_to_fit();
      } else {
        ORT_RETURN_IF_NOT(indices.int32_data_size() == indices_elements,
                          "Sparse tensor: ", name, " indices int16 data size does not match expected: ",
                          indices_elements);
        append_indices(indices.int32_data().cbegin(), indices.int32_data().cend());
      }
      indices_data = gsl::make_span(indices_values);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      if (needs_unpack) {
        if (!utils::HasExternalData(indices)) {
          ORT_RETURN_IF_NOT(indices.raw_data().size() == narrow<size_t>(indices_elements),
                            "Sparse tensor: ", name, " indices raw data size does not match expected: ",
                            indices_elements * sizeof(int8_t));
        }
        ORT_RETURN_IF_ERROR(UnpackInitializerData(indices, model_path, unpack_buffer));
        ORT_RETURN_IF_NOT(unpack_buffer.size() == narrow<size_t>(indices_elements),
                          "Sparse tensor: ", name, " indices data size does not match expected: ",
                          indices_elements * sizeof(int8_t));
        auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpack_buffer));
        append_indices(int8_span.begin(), int8_span.end());
        unpack_buffer.clear();
        unpack_buffer.shrink_to_fit();
      } else {
        ORT_RETURN_IF_NOT(indices.int32_data_size() == indices_elements,
                          "Sparse tensor: ", name, " indices int8 data size does not match expected: ",
                          indices_elements);
        append_indices(indices.int32_data().cbegin(), indices.int32_data().cend());
      }
      indices_data = gsl::make_span(indices_values);
      break;
    }
    default:
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_GRAPH,
          "Sparse tensor: ", name, " indices. Should be one of the following types: int8, int16, int32 or int64");
  }

  const auto indices_rank = indices_shape.NumDimensions();
  if (indices_rank == 1) {
    // flattened indexes
    for (size_t i = 0, lim = narrow<size_t>(nnz_elements); i < lim; ++i) {
      const auto idx = indices_data[i];
      ORT_RETURN_IF_NOT(idx >= 0 && idx < dense_elements,
                        "Sparse tensor: ", name, " index is out of bounds. Got:", idx,
                        " expected to be in [0, ", dense_elements, ")");

      copier(i, narrow<size_t>(idx));
    }
  } else if (indices_rank == 2) {
    // entries in format {NNZ, rank}
    ORT_ENFORCE(indices_shape[1] > 0 && static_cast<size_t>(indices_shape[1]) == dense_dims.size());
    const auto rank = static_cast<size_t>(indices_shape[1]);
    auto cur_index = indices_data.begin();
    InlinedVector<size_t> multipliers;
    multipliers.resize(rank);

    // calculate sum of inner dimension elements for each dimension.
    // e.g. if shape {2,3,4}, the result should be {3*4, 4, 1}
    multipliers[rank - 1] = 1;
    for (auto r = rank - 1; r > 0; --r) {
      multipliers[r - 1] = SafeInt<size_t>(dense_dims[r]) * multipliers[r];
    }

    // calculate the offset for the entry
    // e.g. if shape was {2,3,4} and entry was (1, 0, 2) the offset is 14
    // as there are 2 rows, each with 12 entries per row
    for (size_t i = 0, lim = narrow<size_t>(nnz_elements); i < lim; ++i) {
      SafeInt<int64_t> idx = 0;
      for (size_t j = 0; j < rank; ++j) {
        const auto dim_index = cur_index[j];
        ORT_RETURN_IF_NOT(dim_index >= 0 && dim_index < dense_dims[j],
                          "Sparse tensor: ", name, " index is out of bounds. Got:", dim_index,
                          " expected to be in [0, ", dense_dims[j], ")");
        idx += SafeInt<int64_t>(dim_index) * multipliers[j];
      }
      ORT_RETURN_IF_NOT(idx >= 0 && idx < dense_elements,
                        "Sparse tensor: ", name, " index is out of bounds. Got:", static_cast<int64_t>(idx),
                        " expected to be in [0, ", dense_elements, ")");

      copier(i, static_cast<size_t>(idx));
      cur_index += rank;
    }

    ORT_ENFORCE(cur_index == indices_data.end());
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                             "Sparse tensor: ", name, " indices shape. Expected to be rank 1 or 2. Got:", indices_shape);
  }

  return status;
}

common::Status SparseTensorProtoToDenseTensorProto(const ONNX_NAMESPACE::SparseTensorProto& sparse,
                                                   const std::filesystem::path& model_path,
                                                   ONNX_NAMESPACE::TensorProto& dense) {
  Status status;

  const auto& sparse_values = sparse.values();
  const auto& name = sparse_values.name();

  // In-memory address markers (pointing into mmap'd / heap buffers) are forbidden on sparse
  // sub-tensors. The trusted .ort loader is required to materialize sparse sub-tensors as inline
  // raw_data (see LoadSparseInitializerOrtFormat) so they never carry markers. Untrusted .onnx
  // protobuf input is rejected at the Graph constructor before reaching this function; this is
  // the function-level backstop. A marker here would otherwise trigger an arbitrary memory read
  // in UnpackInitializerData.
  if (HasExternalDataInMemory(sparse_values)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name,
                           " values use an in-memory address marker which is not permitted on sparse sub-tensors.");
  }
  if (HasExternalDataInMemory(sparse.indices())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name,
                           " indices use an in-memory address marker which is not permitted on sparse sub-tensors.");
  }

  const auto values_rank = sparse_values.dims_size();
  if (values_rank != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name, " values should be rank 1 for COO format. Got:", values_rank);
  }

  auto type = sparse_values.data_type();
  if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name,
                           " FLOAT8E8M0 values are unsupported because the type has no zero representation.");
  }

  dense.set_data_type(type);
  *dense.mutable_name() = name;
  SafeInt<int64_t> dense_elements = 1;

  for (auto dim : sparse.dims()) {
    if (dim < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                             "Sparse tensor: ", name, " dense dims expected to be non-negative. Got:", dim);
    }
    dense_elements *= dim;
    dense.add_dims(dim);
  }

  const auto dense_dims = gsl::make_span<const int64_t>(dense.dims().data(), dense.dims().size());

  SafeInt<int64_t> nnz_elements = 1;
  for (auto dim : sparse_values.dims()) {
    if (dim < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                             "Sparse tensor: ", name, " tensor dims expected to be non-negative. Got:", dim);
    }
    nnz_elements *= dim;
  }

  const auto& indices = sparse.indices();
  const auto indices_rank = indices.dims_size();
  if (indices_rank != 1 && indices_rank != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name, " indices should be rank 1 or 2 for supported COO format. Got:", indices_rank);
  }

  const auto indices_dims = gsl::make_span(indices.dims().data(), indices.dims().size());

  if (indices_dims[0] != nnz_elements) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name,
                           " indices outer dimension should match the number of non-zero values. Got:",
                           indices_dims[0], " expected: ", static_cast<int64_t>(nnz_elements));
  }

  if (indices_rank == 2 && dense_dims.size() != narrow<size_t>(indices_dims[1])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                           "Sparse tensor: ", name,
                           " indices is rank 2, its inner dimension should match the rank of the dense tensor. Got:",
                           indices_dims[1], " expected: ", dense_dims.size());
  }

  if (indices_rank == 2) {
    const auto num_indices = TensorShape(indices_dims).Size();
    const int64_t expected_indices_entries = SafeInt<int64_t>(nnz_elements) * indices_dims[1];
    if (num_indices != expected_indices_entries) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                             "Sparse tensor: ", name,
                             " indices is rank 2, it should have NNZ values * indices_dims[1] entries. Got:",
                             num_indices, " expected: ", expected_indices_entries);
    }
  }

  // Validate external data paths before any early returns or allocations.
  // This ensures malicious paths are rejected even for zero-element tensors,
  // and prevents large allocations before an invalid path is caught.
  ORT_RETURN_IF_ERROR(ValidateSparseSubTensorExternalDataPath(sparse_values, model_path));
  ORT_RETURN_IF_ERROR(ValidateSparseSubTensorExternalDataPath(indices, model_path));

  if (type != ONNX_NAMESPACE::TensorProto_DataType_STRING && !HasExternalData(sparse_values)) {
    ORT_RETURN_IF_ERROR(ValidateEmbeddedTensorProtoDataSizeAndShape(sparse_values));
  }

  if (type == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported sparse tensor data type of ",
                           ONNX_NAMESPACE::TensorProto_DataType_STRING);
  }

  // CopySparseData also validates the type, but it runs after allocating the dense buffer
  // and is skipped entirely for an empty dense tensor. Validate here so unsupported types
  // are rejected before either path.
  if (indices.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8 &&
      indices.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT16 &&
      indices.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      indices.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_GRAPH,
        "Sparse tensor: ", name,
        " indices. Should be one of the following types: int8, int16, int32 or int64");
  }

  // Validate embedded indices before allocating the dense zero-filled buffer so malformed
  // inline data is rejected before a large representable dense allocation is attempted.
  if (!HasExternalData(indices)) {
    ORT_RETURN_IF_ERROR(ValidateEmbeddedTensorProtoDataSizeAndShape(indices));
  }

  if (dense_elements == 0) {
    // if there are no elements in the dense tensor, we can return early with an empty tensor proto
    return status;
  }

  size_t element_size = GetElementSizeOfTensor(
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type));
  if (type == ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64 ||
      type == ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128) {
    element_size *= 2;
  }
  const size_t dense_data_size = SafeInt<size_t>(dense_elements) * element_size;
  ORT_RETURN_IF_NOT(dense_data_size <= kMaxEmbeddedInitializerSizeInBytes,
                    "Sparse tensor: ", name, " dense data size of ", dense_data_size,
                    " bytes exceeds the ", kMaxEmbeddedInitializerSizeInBytes,
                    " byte limit for embedded initializer data.");

  // by putting the data into a std::string we can avoid a copy as set_raw_data can do a std::move
  // into the TensorProto.
  std::string dense_data_storage(dense_data_size, 0);
  if (nnz_elements > 0) {
    // need to read in sparse data first as it could be in a type specific field, in raw data, or in external data
    std::vector<uint8_t> values_data;
    ORT_RETURN_IF_ERROR(UnpackInitializerData(sparse_values, model_path, values_data));
    ORT_RETURN_IF_NOT(values_data.size() == SafeInt<size_t>(nnz_elements) * element_size,
                      "Sparse tensor: ", name, " values data size does not match expected: ",
                      static_cast<size_t>(SafeInt<size_t>(nnz_elements) * element_size));
    void* sparse_data = values_data.data();
    void* dense_data = dense_data_storage.data();

    switch (element_size) {
      case 1: {
        status = CopySparseData(
            name, nnz_elements, indices, model_path, dense_dims, dense_elements,
            [sparse_data, dense_data](size_t from_idx, size_t to_idx) {
              static_cast<uint8_t*>(dense_data)[to_idx] = static_cast<const uint8_t*>(sparse_data)[from_idx];
            });

        break;
      }
      case 2: {
        status = CopySparseData(name, nnz_elements, indices, model_path, dense_dims, dense_elements,
                                [sparse_data, dense_data](size_t from_idx, size_t to_idx) {
                                  const auto* src = static_cast<const uint16_t*>(sparse_data) + from_idx;
                                  auto* dst = static_cast<uint16_t*>(dense_data) + to_idx;
                                  memcpy(dst, src, sizeof(uint16_t));
                                });

        break;
      }
      case 4: {
        status = CopySparseData(name, nnz_elements, indices, model_path, dense_dims, dense_elements,
                                [sparse_data, dense_data](size_t from_idx, size_t to_idx) {
                                  const auto* src = static_cast<const uint32_t*>(sparse_data) + from_idx;
                                  auto* dst = static_cast<uint32_t*>(dense_data) + to_idx;
                                  memcpy(dst, src, sizeof(uint32_t));
                                });

        break;
      }
      case 8: {
        status = CopySparseData(name, nnz_elements, indices, model_path, dense_dims, dense_elements,
                                [sparse_data, dense_data](size_t from_idx, size_t to_idx) {
                                  const auto* src = static_cast<const uint64_t*>(sparse_data) + from_idx;
                                  auto* dst = static_cast<uint64_t*>(dense_data) + to_idx;
                                  memcpy(dst, src, sizeof(uint64_t));
                                });
        break;
      }
      case 16: {
        status = CopySparseData(name, nnz_elements, indices, model_path, dense_dims, dense_elements,
                                [sparse_data, dense_data](size_t from_idx, size_t to_idx) {
                                  constexpr size_t element_size = 16;
                                  const auto* src = static_cast<const uint8_t*>(sparse_data) + from_idx * element_size;
                                  auto* dst = static_cast<uint8_t*>(dense_data) + to_idx * element_size;
                                  memcpy(dst, src, element_size);
                                });
        break;
      }

      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Element_size of: ", element_size, " is not supported.",
                               " type: ", type);
    }

    ORT_RETURN_IF_ERROR(status);
  }
  utils::SetRawDataInTensorProto(dense, std::move(dense_data_storage));
  return status;
}

#if !defined(ORT_MINIMAL_BUILD)
// Determines if this is a type specific zero
using IsZeroFunc = bool (*)(const void*);
// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

// Here we are not using tolerance for FP types since these dense tensors were
// created from sparse initializers where zeros were absolute
template <typename T>
inline bool IsZero(const void* p) {
  return (static_cast<T>(0) == *reinterpret_cast<const T*>(p));
}

template <typename T>
inline void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  const auto* src_p = reinterpret_cast<const T*>(src) + src_index;
  auto* dst_p = reinterpret_cast<T*>(dst) + dst_index;
  memcpy(dst_p, src_p, sizeof(T));
}

template <>
inline void CopyElement<uint8_t>(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<uint8_t*>(dst)[dst_index] = reinterpret_cast<const uint8_t*>(src)[src_index];
}

template <typename T>
static void SetIndices(gsl::span<int64_t> gathered_indices, std::string& raw_indices, TensorProto& indices) {
  raw_indices.resize(gathered_indices.size() * sizeof(T));
  auto* ind_dest = reinterpret_cast<T*>(raw_indices.data());
  size_t dest_index = 0;
  for (auto src_index : gathered_indices) {
    if constexpr (sizeof(T) == sizeof(int8_t)) {
      ind_dest[dest_index] = static_cast<T>(src_index);
    } else {
      auto* dst = ind_dest + dest_index;
      T v = static_cast<T>(src_index);
      if constexpr (endian::native != endian::little) {
        auto src = gsl::make_span<const unsigned char>(static_cast<const unsigned char*>(
                                                           reinterpret_cast<const unsigned char*>(&v)),
                                                       sizeof(T));
        auto dest = gsl::make_span<unsigned char>(static_cast<unsigned char*>(
                                                      reinterpret_cast<unsigned char*>(dst)),
                                                  sizeof(T));
        onnxruntime::utils::SwapByteOrderCopy(sizeof(T), src, dest);
      } else {
        memcpy(dst, &v, sizeof(T));
      }
    }
    ++dest_index;
  }
  indices.set_data_type(utils::ToTensorProtoElementType<T>());
}

static void SparsifyGeneric(const void* dense_raw_data, size_t n_dense_elements, size_t element_size,
                            IsZeroFunc is_zero, CopyElementFunc copy, TensorProto& values, TensorProto& indices,
                            size_t& nnz) {
  auto advance = [element_size](const void* start, size_t elements) -> const void* {
    return (reinterpret_cast<const uint8_t*>(start) + elements * element_size);
  };

  const auto* cbegin = dense_raw_data;
  const auto* const cend = advance(cbegin, n_dense_elements);
  std::vector<int64_t> gathered_indices;
  int64_t index = 0;
  while (cbegin != cend) {
    if (!is_zero(cbegin)) {
      gathered_indices.push_back(index);
    }
    ++index;
    cbegin = advance(cbegin, 1U);
  }

  if (!gathered_indices.empty()) {
    auto& raw_data = *values.mutable_raw_data();
    raw_data.resize(gathered_indices.size() * element_size);
    void* data_dest = raw_data.data();

    int64_t dest_index = 0;
    for (auto src_index : gathered_indices) {
      copy(data_dest, dense_raw_data, dest_index, src_index);
      ++dest_index;
    }

    auto gathered_span = gsl::make_span(gathered_indices);
    auto& raw_indices = *indices.mutable_raw_data();
    const auto max_index = gathered_indices.back();
    if (max_index <= std::numeric_limits<int8_t>::max()) {
      SetIndices<int8_t>(gathered_span, raw_indices, indices);
    } else if (max_index <= std::numeric_limits<int16_t>::max()) {
      SetIndices<int16_t>(gathered_span, raw_indices, indices);
    } else if (max_index <= std::numeric_limits<int32_t>::max()) {
      SetIndices<int32_t>(gathered_span, raw_indices, indices);
    } else {
      SetIndices<int64_t>(gathered_span, raw_indices, indices);
    }

    // Raw data is in little endian
    if constexpr (endian::native != endian::little) {
      gsl::span<std::byte> bytes(reinterpret_cast<std::byte*>(raw_data.data()), raw_data.size());
      onnxruntime::utils::SwapByteOrderInplace(element_size, bytes);
    }
  } else {
    indices.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
    utils::SetRawDataInTensorProto(indices, std::string());
  }
  nnz = gathered_indices.size();
}

common::Status DenseTensorToSparseTensorProto(const ONNX_NAMESPACE::TensorProto& dense_proto,
                                              const std::filesystem::path& model_path,
                                              ONNX_NAMESPACE::SparseTensorProto& result) {
  ORT_ENFORCE(HasDataType(dense_proto), "Must have a valid data type");

  if (dense_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported sparse tensor data type of ",
                           ONNX_NAMESPACE::TensorProto_DataType_STRING);
  }

  const auto data_type = dense_proto.data_type();
  SparseTensorProto sparse_proto;
  auto& values = *sparse_proto.mutable_values();
  values.set_name(dense_proto.name());
  values.set_data_type(data_type);

  auto& indices = *sparse_proto.mutable_indices();

  SafeInt<size_t> n_dense_elements = 1;
  for (auto dim : dense_proto.dims()) {
    n_dense_elements *= dim;
  }

  auto ml_data = DataTypeImpl::TensorTypeFromONNXEnum(data_type)->GetElementType();
  size_t element_size = ml_data->Size();

  std::vector<uint8_t> dense_raw_data;
  ORT_RETURN_IF_ERROR(UnpackInitializerData(dense_proto, model_path, dense_raw_data));

  size_t nnz = 0;
  void* dense_data = dense_raw_data.data();
  switch (element_size) {
    case 1: {
      SparsifyGeneric(dense_data, n_dense_elements, element_size, IsZero<uint8_t>, CopyElement<uint8_t>, values,
                      indices, nnz);
      break;
    }
    case 2: {
      SparsifyGeneric(dense_data, n_dense_elements, element_size, IsZero<uint16_t>, CopyElement<uint16_t>, values,
                      indices, nnz);
      break;
    }
    case 4: {
      SparsifyGeneric(dense_data, n_dense_elements, element_size, IsZero<uint32_t>, CopyElement<uint32_t>, values,
                      indices, nnz);
      break;
    }
    case 8: {
      SparsifyGeneric(dense_data, n_dense_elements, element_size, IsZero<uint64_t>, CopyElement<uint64_t>, values,
                      indices, nnz);
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Element_size of: ", element_size, " is not supported.",
                             " data_type: ", data_type);
  }

  // Fix up shapes
  values.add_dims(nnz);
  indices.add_dims(nnz);

  // Save dense shape
  *sparse_proto.mutable_dims() = dense_proto.dims();
  swap(result, sparse_proto);
  return Status::OK();
}

#endif  // !ORT_MINIMAL_BUILD
#endif  // !defined(DISABLE_SPARSE_TENSORS)

template common::Status GetSizeInBytesFromTensorProto<kAllocAlignment>(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                                       size_t* out);
template common::Status GetSizeInBytesFromTensorProto<0>(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

template <typename ComponentType, typename RepeatedField>
Status UnpackComplexInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                    const RepeatedField& typed_data,
                                    std::vector<uint8_t>& unpacked_tensor) {
  if (initializer.has_raw_data()) {
    const size_t tensor_byte_size = initializer.raw_data().size();
    const size_t component_count = tensor_byte_size / sizeof(ComponentType);
    unpacked_tensor.resize(tensor_byte_size);
    return UnpackTensorWithRawDataImpl(initializer.raw_data().data(), tensor_byte_size, component_count,
                                       sizeof(ComponentType), unpacked_tensor.data());
  }

  unpacked_tensor.resize(SafeInt<size_t>(typed_data.size()) * sizeof(ComponentType));
  auto* output = unpacked_tensor.data();
  for (const ComponentType component : typed_data) {
    memcpy(output, &component, sizeof(component));
    output += sizeof(component);
  }

  return Status::OK();
}

#define CASE_UNPACK(TYPE, ELEMENT_TYPE, DATA_SIZE)                                                                   \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##TYPE: {                                          \
    SafeInt<size_t> tensor_byte_size;                                                                                \
    size_t element_count = 0;                                                                                        \
    if (initializer.has_raw_data()) {                                                                                \
      tensor_byte_size = initializer.raw_data().size();                                                              \
      element_count = tensor_byte_size / sizeof(ELEMENT_TYPE);                                                       \
    } else {                                                                                                         \
      element_count = initializer.DATA_SIZE();                                                                       \
      tensor_byte_size = element_count * sizeof(ELEMENT_TYPE);                                                       \
    }                                                                                                                \
    unpacked_tensor.resize(tensor_byte_size);                                                                        \
    return onnxruntime::utils::UnpackTensor(initializer,                                                             \
                                            initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,    \
                                            initializer.has_raw_data() ? initializer.raw_data().size() : 0,          \
                                            reinterpret_cast<ELEMENT_TYPE*>(unpacked_tensor.data()), element_count); \
    break;                                                                                                           \
  }

#define CASE_UNPACK_COMPLEX(TYPE, COMPONENT_TYPE, DATA)                                  \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##TYPE: {              \
    return UnpackComplexInitializerData<COMPONENT_TYPE>(initializer, initializer.DATA(), \
                                                        unpacked_tensor);                \
  }

// Sub-byte types (2-bit and 4-bit) are stored in a packed format.
// This unpacking code is shared for INT4, UINT4, FLOAT4E2M1, INT2, and UINT2.
// CALC_PACKED_UNITS_FUN specifies the function to calculate packed byte count from element count.
#define CASE_UNPACK_SUBBYTE_TYPE(TYPE, ELEMENT_TYPE, DATA_SIZE, CALC_PACKED_UNITS_FUN)                               \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##TYPE: {                                          \
    TensorShape tensor_shape = GetTensorShapeFromTensorProto(initializer);                                           \
    size_t element_count = static_cast<size_t>(tensor_shape.Size());                                                 \
    size_t packed_element_count = ELEMENT_TYPE::CALC_PACKED_UNITS_FUN(element_count);                                \
    unpacked_tensor.resize(packed_element_count * sizeof(ELEMENT_TYPE));                                             \
    return onnxruntime::utils::UnpackTensor(initializer,                                                             \
                                            initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,    \
                                            initializer.has_raw_data() ? initializer.raw_data().size() : 0,          \
                                            reinterpret_cast<ELEMENT_TYPE*>(unpacked_tensor.data()), element_count); \
    break;                                                                                                           \
  }

Status UnpackInitializerData(const onnx::TensorProto& initializer,
                             const std::filesystem::path& model_path,
                             std::vector<uint8_t>& unpacked_tensor) {
  // TODO, if std::vector does not use a custom allocator, the default std::allocator will
  // allocation the memory aligned to std::max_align_t, need look into allocating
  // forced aligned memory (align as 16 or larger)for unpacked_tensor
  if (HasExternalData(initializer)) {
    ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(
        initializer,
        model_path.parent_path(),
        unpacked_tensor));
    if (initializer.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3 ||
        initializer.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2) {
      std::unique_ptr<ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(initializer.external_data(), external_data_info));
      if (external_data_info->GetRelPath() == kTensorProtoNativeEndianMemoryAddressTag) {
        return Status::OK();
      }

      ONNX_NAMESPACE::TensorProto packed_initializer = initializer;
      packed_initializer.clear_external_data();
      packed_initializer.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_DEFAULT);
      packed_initializer.set_raw_data(unpacked_tensor.data(), unpacked_tensor.size());
      return UnpackInitializerData(packed_initializer, {}, unpacked_tensor);
    }
    return Status::OK();
  }

  if (HasRawData(initializer)) {
    ORT_RETURN_IF_ERROR(ValidateEmbeddedTensorProtoDataSizeAndShape(initializer));
  }

  switch (initializer.data_type()) {
    CASE_UNPACK(FLOAT, float, float_data_size);
    CASE_UNPACK(DOUBLE, double, double_data_size);
    CASE_UNPACK_COMPLEX(COMPLEX64, float, float_data);
    CASE_UNPACK_COMPLEX(COMPLEX128, double, double_data);
    CASE_UNPACK(BOOL, bool, int32_data_size);
    CASE_UNPACK(INT8, int8_t, int32_data_size);
    CASE_UNPACK(INT16, int16_t, int32_data_size);
    CASE_UNPACK(INT32, int32_t, int32_data_size);
    CASE_UNPACK(INT64, int64_t, int64_data_size);
    CASE_UNPACK(UINT8, uint8_t, int32_data_size);
    CASE_UNPACK(UINT16, uint16_t, int32_data_size);
    CASE_UNPACK(UINT32, uint32_t, uint64_data_size);
    CASE_UNPACK(UINT64, uint64_t, uint64_data_size);
    CASE_UNPACK(FLOAT16, onnxruntime::MLFloat16, int32_data_size);
    CASE_UNPACK(BFLOAT16, onnxruntime::BFloat16, int32_data_size);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_UNPACK(FLOAT8E4M3FN, onnxruntime::Float8E4M3FN, int32_data_size);
    CASE_UNPACK(FLOAT8E4M3FNUZ, onnxruntime::Float8E4M3FNUZ, int32_data_size);
    CASE_UNPACK(FLOAT8E5M2, onnxruntime::Float8E5M2, int32_data_size);
    CASE_UNPACK(FLOAT8E5M2FNUZ, onnxruntime::Float8E5M2FNUZ, int32_data_size);
    CASE_UNPACK(FLOAT8E8M0, onnxruntime::Float8E8M0, int32_data_size);
#endif
    CASE_UNPACK_SUBBYTE_TYPE(INT4, Int4x2, int32_data_size, CalcNumInt4Pairs);
    CASE_UNPACK_SUBBYTE_TYPE(UINT4, UInt4x2, int32_data_size, CalcNumInt4Pairs);
    CASE_UNPACK_SUBBYTE_TYPE(INT2, Int2x4, int32_data_size, CalcNumInt2Quads);
    CASE_UNPACK_SUBBYTE_TYPE(UINT2, UInt2x4, int32_data_size, CalcNumInt2Quads);

#if !defined(DISABLE_FLOAT4_TYPES)
    CASE_UNPACK_SUBBYTE_TYPE(FLOAT4E2M1, Float4E2M1x2, int32_data_size, CalcNumFloat4Pairs);
#endif
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT6E2M3: {
      TensorShape tensor_shape = GetTensorShapeFromTensorProto(initializer);
      const size_t element_count = static_cast<size_t>(tensor_shape.Size());
      unpacked_tensor.resize(element_count * sizeof(Float6E2M3));
      return onnxruntime::utils::UnpackTensor(initializer,
                                              initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,
                                              initializer.has_raw_data() ? initializer.raw_data().size() : 0,
                                              reinterpret_cast<Float6E2M3*>(unpacked_tensor.data()), element_count);
    }
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT6E3M2: {
      TensorShape tensor_shape = GetTensorShapeFromTensorProto(initializer);
      const size_t element_count = static_cast<size_t>(tensor_shape.Size());
      unpacked_tensor.resize(element_count * sizeof(Float6E3M2));
      return onnxruntime::utils::UnpackTensor(initializer,
                                              initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,
                                              initializer.has_raw_data() ? initializer.raw_data().size() : 0,
                                              reinterpret_cast<Float6E3M2*>(unpacked_tensor.data()), element_count);
    }

    default:
      break;
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported type: ", initializer.data_type());
}
#undef CASE_UNPACK_COMPLEX
#undef CASE_UNPACK

Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer, std::vector<uint8_t>& unpacked_tensor) {
  ORT_RETURN_IF(initializer.data_location() == TensorProto_DataLocation_EXTERNAL,
                "The given initializer contains external data");
  return UnpackInitializerData(initializer, std::filesystem::path(), unpacked_tensor);
}

std::optional<std::string> GetNodeProtoLayeringAnnotation(const ONNX_NAMESPACE::NodeProto& node_proto) {
  std::optional<std::string> result;
  for (const auto& prop : node_proto.metadata_props()) {
    if (prop.key() == kNodeProtoLayerAnnotation) {
      if (!prop.value().empty()) {
        result = prop.value();
        break;
      }
    }
  }
  return result;
}

}  // namespace utils
}  // namespace onnxruntime
