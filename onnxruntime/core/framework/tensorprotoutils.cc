// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include <memory>
#include <algorithm>
#include <limits>
#include <gsl/gsl>

#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/endian_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/allocator.h"
#include "core/framework/callback.h"
#include "core/framework/data_types.h"
#include "core/platform/path_lib.h"
#include "core/session/ort_apis.h"
#include "onnx/defs/tensor_proto_util.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

// Provide template specializations for onnxruntime-specific types.
namespace ONNX_NAMESPACE {
template <>
TensorProto ToTensor<onnxruntime::MLFloat16>(const onnxruntime::MLFloat16& value) {
  TensorProto t;
  t.set_data_type(TensorProto_DataType_FLOAT16);
  t.add_int32_data(value.val);
  t.add_dims(1);
  return t;
}

template <>
TensorProto ToTensor<onnxruntime::MLFloat16>(const std::vector<onnxruntime::MLFloat16>& values) {
  TensorProto t;
  t.clear_int32_data();
  t.set_data_type(TensorProto_DataType_FLOAT16);
  for (const onnxruntime::MLFloat16& val : values) {
    t.add_int32_data(val.val);
  }
  return t;
}

template <>
TensorProto ToTensor<onnxruntime::BFloat16>(const onnxruntime::BFloat16& value) {
  TensorProto t;
  t.set_data_type(TensorProto_DataType_BFLOAT16);
  t.add_int32_data(value.val);
  t.add_dims(1);
  return t;
}

template <>
TensorProto ToTensor<onnxruntime::BFloat16>(const std::vector<onnxruntime::BFloat16>& values) {
  TensorProto t;
  t.clear_int32_data();
  t.set_data_type(TensorProto_DataType_BFLOAT16);
  for (const onnxruntime::BFloat16& val : values) {
    t.add_int32_data(val.val);
  }
  return t;
}

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

namespace {

std::vector<int64_t> GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}

// This function doesn't support string tensors
template <typename T>
static Status UnpackTensorWithRawData(const void* raw_data, size_t raw_data_length, size_t expected_size,
                                      /*out*/ T* p_data) {
  size_t expected_size_in_bytes;
  if (!onnxruntime::IAllocator::CalcMemSizeForArray(expected_size, sizeof(T), &expected_size_in_bytes)) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "size overflow");
  }
  if (raw_data_length != expected_size_in_bytes)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "UnpackTensor: the pre-allocated size does not match the raw data size, expected ",
                           expected_size_in_bytes, ", got ", raw_data_length);

  const char* const raw_data_bytes = reinterpret_cast<const char*>(raw_data);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(
      gsl::make_span(raw_data_bytes, raw_data_length), gsl::make_span(p_data, expected_size)));
  return Status::OK();
}

static Status GetExternalDataInfo(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                  const ORTCHAR_T* tensor_proto_dir,
                                  std::basic_string<ORTCHAR_T>& external_file_path,
                                  onnxruntime::FileOffsetType& file_offset,
                                  SafeInt<size_t>& tensor_data_length) {
  ORT_RETURN_IF_NOT(onnxruntime::utils::HasExternalData(tensor_proto),
                    "Tensor does not have external data to read from.");

  ORT_RETURN_IF_NOT(tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
                    "External data type cannot be UNDEFINED or STRING.");

  std::unique_ptr<onnxruntime::ExternalDataInfo> external_data_info;
  ORT_RETURN_IF_ERROR(onnxruntime::ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));

  if (tensor_proto_dir != nullptr) {
    external_file_path = onnxruntime::ConcatPathComponent<ORTCHAR_T>(tensor_proto_dir, external_data_info->GetRelPath());
  } else {
    external_file_path = external_data_info->GetRelPath();
  }

  file_offset = external_data_info->GetOffset();

  ORT_RETURN_IF_ERROR(onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(
      tensor_proto, &tensor_data_length));
  const size_t external_data_length = external_data_info->GetLength();

  ORT_RETURN_IF_NOT(
      external_data_length == 0 ||
          external_data_length == tensor_data_length,
      "TensorProto external data size mismatch. ",
      "Computed size: ", *&tensor_data_length,
      ", external_data.length: ", external_data_length);

  return Status::OK();
}

// Read external data for tensor in unint8_t* form and return Status::OK() if the data is read successfully.
// Uses the tensor_proto_dir to construct the full path for external data. If tensor_proto_dir == nullptr
// then uses the current directory instead.
// This function does not unpack string_data of an initializer tensor
static Status ReadExternalDataForTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                        const ORTCHAR_T* tensor_proto_dir,
                                        std::unique_ptr<uint8_t[]>& unpacked_tensor,
                                        SafeInt<size_t>& tensor_data_length) {
  std::basic_string<ORTCHAR_T> external_file_path;
  onnxruntime::FileOffsetType file_offset;
  ORT_RETURN_IF_ERROR(GetExternalDataInfo(
      tensor_proto,
      tensor_proto_dir,
      external_file_path,
      file_offset,
      tensor_data_length));

  unpacked_tensor.reset(new uint8_t[*&tensor_data_length]);
  ORT_RETURN_IF_ERROR(onnxruntime::Env::Default().ReadFileIntoBuffer(
      external_file_path.c_str(),
      file_offset,
      tensor_data_length,
      gsl::make_span(reinterpret_cast<char*>(unpacked_tensor.get()), tensor_data_length)));

  return Status::OK();
}
}  // namespace

namespace onnxruntime {
namespace utils {
#if !defined(ORT_MINIMAL_BUILD)
#define DEFINE_UNPACK_EXTERNAL_TENSOR(T)                                                                                   \
  template <>                                                                                                              \
  Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& tensor,                                           \
                                      const ORTCHAR_T* tensor_proto_dir, size_t expected_size,                             \
                                      /*out*/ T* p_data) {                                                                 \
    ORT_RETURN_IF(nullptr == p_data);                                                                                      \
                                                                                                                           \
    std::unique_ptr<uint8_t[]> unpacked_tensor;                                                                            \
    SafeInt<size_t> tensor_byte_size = 0;                                                                                  \
    ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(                                                                         \
        tensor,                                                                                                            \
        tensor_proto_dir,                                                                                                  \
        unpacked_tensor,                                                                                                   \
        tensor_byte_size));                                                                                                \
                                                                                                                           \
    size_t element_count = tensor_byte_size / sizeof(T);                                                                   \
    ORT_RETURN_IF_NOT(expected_size == element_count, "Expected data size does not match the actual external data size."); \
    ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(                                                              \
        gsl::make_span(reinterpret_cast<char*>(unpacked_tensor.get()), tensor_byte_size),                                  \
        gsl::make_span(p_data, expected_size)));                                                                           \
                                                                                                                           \
    return Status::OK();                                                                                                   \
  }

DEFINE_UNPACK_EXTERNAL_TENSOR(float)
DEFINE_UNPACK_EXTERNAL_TENSOR(double)
DEFINE_UNPACK_EXTERNAL_TENSOR(uint8_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(int8_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(int16_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(uint16_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(int32_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(int64_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(uint64_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(uint32_t)
DEFINE_UNPACK_EXTERNAL_TENSOR(bool)
DEFINE_UNPACK_EXTERNAL_TENSOR(MLFloat16)
DEFINE_UNPACK_EXTERNAL_TENSOR(BFloat16)

template <>
Status UnpackTensorWithExternalData(const ONNX_NAMESPACE::TensorProto& /*tensor*/,
                                    const ORTCHAR_T* /*tensor_proto_dir*/, size_t /*expected_size*/,
                                    /*out*/ std::string* /*p_data*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "External data type cannot be STRING.");
}
#endif  //!defined(ORT_MINIMAL_BUILD)

// This macro doesn't work for Float16/bool/string tensors
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                                      \
  template <>                                                                                                      \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,        \
                      /*out*/ T* p_data, size_t expected_size) {                                                   \
    if (nullptr == p_data) {                                                                                       \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.field_size();                                \
      if (size == 0) return Status::OK();                                                                          \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                         \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (raw_data != nullptr) {                                                                                     \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);                               \
    }                                                                                                              \
    if (static_cast<size_t>(tensor.field_size()) != expected_size)                                                 \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "corrupted protobuf data: tensor shape size(",         \
                             expected_size, ") does not match the data size(", tensor.field_size(), ") in proto"); \
    auto& data = tensor.field_name();                                                                              \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                                    \
      *p_data++ = *reinterpret_cast<const T*>(data_iter);                                                          \
    return Status::OK();                                                                                           \
  }

// TODO: complex64 complex128
DEFINE_UNPACK_TENSOR(float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR(double, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR(uint8_t, ONNX_NAMESPACE::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int16_t, ONNX_NAMESPACE::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(uint16_t, ONNX_NAMESPACE::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR(uint64_t, ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR(uint32_t, ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

// doesn't support raw data
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* /*raw_data*/, size_t /*raw_data_len*/,
                    /*out*/ std::string* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0) return Status::OK();
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
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ bool* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");
  for (int iter : tensor.int32_data()) {
    *p_data++ = static_cast<bool>(iter);
  }

  return Status::OK();
}
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ MLFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return Status::OK();
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
    p_data[i] = MLFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

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
    p_data[i] = BFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

#define CASE_PROTO_TRACE(X, Y)                                                                     \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                             \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) {          \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto"); \
    }                                                                                              \
    break;

template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (google::protobuf::int64 dim : dims) {
    if (dim < 0 || static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
    }
    if (!IAllocator::CalcMemSizeForArray(size, static_cast<size_t>(dim), &size)) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
    }
  }
  switch (tensor_proto.data_type()) {
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
    default:
      return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

TensorShape GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = HasDimValue(dims[i]) ? dims[i].dim_value()
                                               : -1; /* symbolic dimensions are represented as -1 in onnxruntime*/
  }
  return TensorShape(std::move(tensor_shape_vec));
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
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return nullptr;
    size_t tensor_size = input_len / sizeof(std::string);
    std::string* ptr = reinterpret_cast<std::string*>(input);
    for (size_t i = 0, n = tensor_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
  }

  return status;
}

ORT_API(void, OrtUninitializeBuffer, _In_opt_ void* input, size_t input_len, enum ONNXTensorElementDataType type) {
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return;
  size_t tensor_size = input_len / sizeof(std::string);
  std::string* ptr = reinterpret_cast<std::string*>(input);
  using std::string;
  for (size_t i = 0, n = tensor_size; i < n; ++i) {
    ptr[i].~string();
  }
}

static void UnInitTensor(void* param) noexcept {
  UnInitializeParam* p = reinterpret_cast<UnInitializeParam*>(param);
  OrtUninitializeBuffer(p->preallocated, p->preallocated_size, p->ele_type);
  delete p;
}

#define CASE_PROTO(X, Y)                                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                              \
    ORT_RETURN_IF_ERROR(                                                                                            \
        UnpackTensor<Y>(tensor_proto, raw_data, raw_data_len, (Y*)preallocated, static_cast<size_t>(tensor_size))); \
    break;

class AutoDelete {
 public:
  OrtCallback d{nullptr, nullptr};
  AutoDelete() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AutoDelete);
  ~AutoDelete() {
    if (d.f != nullptr) {
      d.f(d.param);
    }
  }
};

static void DeleteCharArray(void* param) noexcept {
  auto arr = reinterpret_cast<char*>(param);
  delete[] arr;
}

static Status GetFileContent(
    const Env& env, const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
    void*& raw_buffer, OrtCallback& deleter) {
  // query length if it is 0
  if (length == 0) {
    ORT_RETURN_IF_ERROR(env.GetFileLength(file_path, length));
  }

  // first, try to map into memory
  {
    Env::MappedMemoryPtr mapped_memory{};
    auto status = env.MapFileIntoMemory(file_path, offset, length, mapped_memory);
    if (status.IsOK()) {
      deleter = mapped_memory.get_deleter().callback;
      raw_buffer = mapped_memory.release();
      return Status::OK();
    }
  }

  // if that fails, try to copy
  auto buffer = onnxruntime::make_unique<char[]>(length);
  ORT_RETURN_IF_ERROR(env.ReadFileIntoBuffer(
      file_path, offset, length, gsl::make_span(buffer.get(), length)));

  deleter = OrtCallback{DeleteCharArray, buffer.get()};
  raw_buffer = buffer.release();
  return Status::OK();
}

static void MoveOrtCallback(OrtCallback& from, OrtCallback& to) {
  to.f = from.f;
  to.param = from.param;
  from.f = nullptr;
  from.param = nullptr;
}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6239)
#endif
// TODO: Change the current interface to take Path object for model path
// so that validating and manipulating path for reading external data becomes easy
Status TensorProtoToMLValue(const Env& env, const ORTCHAR_T* model_path,
                            const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer& m, OrtValue& value,
                            OrtCallback& deleter) {
  const OrtMemoryInfo& allocator = m.GetAllocInfo();
  ONNXTensorElementDataType ele_type = utils::GetTensorElementType(tensor_proto);
  deleter.f = nullptr;
  deleter.param = nullptr;
  void* raw_data = nullptr;
  SafeInt<size_t> raw_data_len = 0;
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  AutoDelete deleter_for_file_data;
  void* tensor_data;
  {
    if (utils::HasExternalData(tensor_proto)) {
      // Get the external data info
      std::basic_string<ORTCHAR_T> external_data_file_path;
      FileOffsetType file_offset;
      std::basic_string<ORTCHAR_T> tensor_proto_dir;
      if (model_path != nullptr) {
        ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
      }
      ORT_RETURN_IF_ERROR(GetExternalDataInfo(
          tensor_proto,
          tensor_proto_dir.size() == 0 ? nullptr : tensor_proto_dir.c_str(),
          external_data_file_path, file_offset, raw_data_len));

      // load the file
      ORT_RETURN_IF_ERROR(GetFileContent(
          env, external_data_file_path.c_str(), file_offset, raw_data_len,
          raw_data, deleter_for_file_data.d));
    } else if (utils::HasRawData(tensor_proto)) {
      if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "string tensor can not have raw data");
      raw_data = const_cast<char*>(tensor_proto.raw_data().data());
      // TODO The line above has const-correctness issues. Below is a possible fix which copies the tensor_proto data
      //      into a writeable buffer. However, it requires extra memory which may exceed the limit for certain tests.
      //auto buffer = onnxruntime::make_unique<char[]>(tensor_proto.raw_data().size());
      //std::memcpy(buffer.get(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
      //deleter_for_file_data.d = OrtCallback{DeleteCharArray, buffer.get()};
      //raw_data = buffer.release();
      raw_data_len = tensor_proto.raw_data().size();
    }
    if (endian::native == endian::little && raw_data != nullptr && deleter_for_file_data.d.f != nullptr) {
      tensor_data = raw_data;
      MoveOrtCallback(deleter_for_file_data.d, deleter);
    } else {
      void* preallocated = m.GetBuffer();
      size_t preallocated_size = m.GetLen();
      int64_t tensor_size = 1;
      {
        for (auto i : tensor_proto.dims()) {
          if (i < 0)
            return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "tensor can't contain negative dims");
          tensor_size *= i;
        }
      }
      // tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
      if (static_cast<uint64_t>(tensor_size) > SIZE_MAX) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
      }
      size_t size_to_allocate;
      if (!IAllocator::CalcMemSizeForArrayWithAlignment<0>(static_cast<size_t>(tensor_size), type->Size(),
                                                           &size_to_allocate)) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
      }

      if (preallocated && preallocated_size < size_to_allocate)
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The buffer planner is not consistent with tensor buffer size, expected ",
                               size_to_allocate, ", got ", preallocated_size);
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
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING:
          if (preallocated != nullptr) {
            OrtStatus* status = OrtInitializeBufferForTensor(preallocated, preallocated_size, ele_type);
            if (status != nullptr) {
              OrtApis::ReleaseStatus(status);
              return Status(common::ONNXRUNTIME, common::FAIL, "initialize preallocated buffer failed");
            }

            deleter.f = UnInitTensor;
            deleter.param = new UnInitializeParam{preallocated, preallocated_size, ele_type};
          }
          ORT_RETURN_IF_ERROR(UnpackTensor<std::string>(tensor_proto, raw_data, raw_data_len,
                                                        (std::string*)preallocated, static_cast<size_t>(tensor_size)));
          break;
        default: {
          std::ostringstream ostr;
          ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
        }
      }
      tensor_data = preallocated;
    }
  }
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  TensorShape tensor_shape{tensor_shape_vec};

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value.Init(new Tensor(type, tensor_shape, tensor_data, allocator), ml_tensor,
             ml_tensor->GetDeleteFunc());
  return Status::OK();
}
#ifdef _MSC_VER
#pragma warning(pop)
#pragma warning(disable : 6239)
#endif
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
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return CApiElementTypeFromProtoType(tensor_proto.data_type());
}

ONNX_NAMESPACE::TensorProto TensorToTensorProto(const Tensor& tensor, const std::string& tensor_proto_name) {
  // Given we are using the raw_data field in the protobuf, this will work only for little-endian format.
  ORT_ENFORCE(endian::native == endian::little);

  // Set name, dimensions, type, and data of the TensorProto.
  ONNX_NAMESPACE::TensorProto tensor_proto;

  tensor_proto.set_name(tensor_proto_name);

  for (auto& dim : tensor.Shape().GetDims()) {
    tensor_proto.add_dims(dim);
  }

  tensor_proto.set_data_type(tensor.GetElementType());
  if (tensor.IsDataTypeString()) {
    auto* mutable_string_data = tensor_proto.mutable_string_data();
    auto f = tensor.Data<std::string>();
    auto end = f + tensor.Shape().Size();
    for (; f < end; ++f) {
      *mutable_string_data->Add() = *f;
    }
  } else {
    tensor_proto.set_raw_data(tensor.DataRaw(), tensor.SizeInBytes());
  }

  return tensor_proto;
}

common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              const Path& model_path,
                                              ONNX_NAMESPACE::TensorProto& tensor) {
  const AttributeProto& constant_attribute = node.attribute(0);

  switch (constant_attribute.type()) {
    case AttributeProto_AttributeType_TENSOR:
      tensor = constant_attribute.t();
      break;
    case AttributeProto_AttributeType_FLOAT:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      tensor.add_float_data(constant_attribute.f());
      break;
    case AttributeProto_AttributeType_FLOATS:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      *tensor.mutable_float_data() = constant_attribute.floats();
      break;
    case AttributeProto_AttributeType_INT:
      tensor.set_data_type(TensorProto_DataType_INT64);
      tensor.add_int64_data(constant_attribute.i());
      break;
    case AttributeProto_AttributeType_INTS:
      tensor.set_data_type(TensorProto_DataType_INT64);
      *tensor.mutable_int64_data() = constant_attribute.ints();
      break;
    case AttributeProto_AttributeType_STRING:
      tensor.set_data_type(TensorProto_DataType_STRING);
      tensor.add_string_data(constant_attribute.s());
      break;
    case AttributeProto_AttributeType_STRINGS: {
      tensor.set_data_type(TensorProto_DataType_STRING);
      *tensor.mutable_string_data() = constant_attribute.strings();
      break;
    }
    case AttributeProto_AttributeType_SPARSE_TENSOR: {
      auto& s = constant_attribute.sparse_tensor();
      ORT_RETURN_IF_ERROR(SparseTensorProtoToDenseTensorProto(s, model_path, tensor));
      break;
    }
    default:
      ORT_THROW("Unsupported attribute value type of ", constant_attribute.type(),
                " in 'Constant' node '", node.name(), "'");
  }

  // set name last in case attribute type was tensor (would copy over name)
  *(tensor.mutable_name()) = node.output(0);

  return Status::OK();
}

template <typename T>
static Status CopySparseData(size_t n_sparse_elements,
                             const ONNX_NAMESPACE::TensorProto& indices,
                             gsl::span<const int64_t> dims,
                             std::function<void(size_t from_idx, size_t to_idx)> copier) {
  Status status = Status::OK();
  TensorShape indices_shape(indices.dims().data(), indices.dims().size());

  ORT_RETURN_IF_NOT(indices.data_type() == ONNX_NAMESPACE ::TensorProto_DataType_INT64, "Indicies expected to be INT64");

  gsl::span<const int64_t> indices_data;
  const auto elements = static_cast<size_t>(indices_shape.Size());
  if (indices.int64_data_size() > 0) {
    indices_data = gsl::make_span<const int64_t>(indices.int64_data().data(), elements);
  } else if (indices.has_raw_data()) {
    ORT_RETURN_IF_NOT(indices.raw_data().size() == (elements * sizeof(int64_t)),
                      "Sparse Indicies raw data size does not match expected.");
    indices_data = gsl::make_span<const int64_t>(reinterpret_cast<const int64_t*>(indices.raw_data().data()), elements);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Invalid SparseTensor indices. Should either have raw or int64 data");
  }

  if (indices_shape.NumDimensions() == 1) {
    // flattened indexes
    for (size_t i = 0; i < n_sparse_elements; ++i) {
      copier(i, static_cast<size_t>(indices_data[i]));
    }
  } else if (indices_shape.NumDimensions() == 2) {
    // entries in format {NNZ, rank}
    size_t rank = static_cast<size_t>(indices_shape[1]);
    ORT_ENFORCE(rank == dims.size() && rank > 0);
    const int64_t* cur_index = indices_data.data();
    std::vector<size_t> multipliers;
    multipliers.resize(rank);

    // calculate sum of inner dimension elements for each dimension.
    // e.g. if shape {2,3,4}, the result should be {3*4, 4, 1}
    multipliers[rank - 1] = 1;
    for (int32_t r = static_cast<int32_t>(rank) - 2; r >= 0; --r) {
      multipliers[r] = static_cast<size_t>(dims[r + 1]) * multipliers[r + 1];
    }

    // calculate the offset for the entry
    // e.g. if shape was {2,3,4} and entry was (1, 0, 2) the offset is 14
    // as there are 2 rows, each with 12 entries per row
    for (size_t i = 0; i < n_sparse_elements; ++i) {
      size_t idx = 0;
      for (size_t j = 0; j < rank; ++j) {
        idx += static_cast<size_t>(cur_index[j]) * multipliers[j];
      }

      copier(i, idx);
      cur_index += rank;
    }

    ORT_ENFORCE(cur_index == &*indices_data.cend());
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Invalid SparseTensor indices. Should be rank 0 or 1. Got:",
                             indices_shape);
  }

  return status;
}

struct UnsupportedSparseDataType {
  Status operator()(int32_t dt_type) const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported sparse tensor data type of ", dt_type);
  }
};

template <typename T>
struct GetElementSize {
  Status operator()(size_t& element_size) const {
    element_size = sizeof(T);
    return Status::OK();
  }
};

common::Status SparseTensorProtoToDenseTensorProto(const ONNX_NAMESPACE::SparseTensorProto& sparse,
                                                   const Path& model_path,
                                                   ONNX_NAMESPACE::TensorProto& dense) {
  Status status = Status::OK();

  const auto& sparse_values = sparse.values();
  auto type = sparse_values.data_type();
  dense.set_data_type(type);
  *dense.mutable_name() = sparse_values.name();

  SafeInt<size_t> n_sparse_elements = 1;
  for (auto dim : sparse_values.dims()) {
    n_sparse_elements *= dim;
  }

  SafeInt<size_t> n_dense_elements = 1;
  for (auto dim : sparse.dims()) {
    n_dense_elements *= dim;
    dense.add_dims(dim);
  }

  const auto& indices = sparse.indices();
  auto dims = gsl::make_span<const int64_t>(dense.dims().data(), dense.dims().size());

  if (type != TensorProto_DataType_STRING) {
    // need to read in sparse data first as it could be in a type specific field, in raw data, or in external data
    size_t sparse_bytes = 0;
    std::unique_ptr<uint8_t[]> sparse_data_storage;
    ORT_RETURN_IF_ERROR(UnpackInitializerData(sparse_values, model_path, sparse_data_storage, sparse_bytes));
    void* sparse_data = sparse_data_storage.get();
    size_t element_size = 0;
    // We want to this list to match the one used below in DenseTensorToSparseTensorProto()
    MLTypeCallDispatcherRet<Status, GetElementSize, float, int8_t, uint8_t> type_disp(type);
    ORT_RETURN_IF_ERROR(type_disp.InvokeWithUnsupportedPolicy<UnsupportedSparseDataType>(element_size));

    // by putting the data into a std::string we can avoid a copy as set_raw_data can do a std::move
    // into the TensorProto. however to actually write to the buffer we have created in the std::string we need
    // this somewhat dirty hack to get a mutable pointer. we could alternatively use &dense_data_storage.front()
    // but using const_cast makes it more obvious we're doing something ugly.
    // C++17 add non-const data() where we could remove const_cast
    std::string dense_data_storage(n_dense_elements * element_size, 0);
    void* dense_data = const_cast<char*>(dense_data_storage.data());

    switch (element_size) {
      case 1: {
        auto dense_data_span = gsl::make_span<uint8_t>(static_cast<uint8_t*>(dense_data), n_dense_elements);
        status = CopySparseData<uint8_t>(
            n_sparse_elements,
            indices, dims,
            [sparse_data, dense_data_span](size_t from_idx, size_t to_idx) {
              dense_data_span[to_idx] = static_cast<const uint8_t*>(sparse_data)[from_idx];
            });

        break;
      }
      case 4: {
        auto dense_data_span = gsl::make_span<uint32_t>(static_cast<uint32_t*>(dense_data), n_dense_elements);
        status = CopySparseData<uint32_t>(
            n_sparse_elements,
            indices, dims,
            [sparse_data, dense_data_span](size_t from_idx, size_t to_idx) {
              dense_data_span[to_idx] = static_cast<const uint32_t*>(sparse_data)[from_idx];
            });

        break;
      }
      default:
        ORT_THROW(false, "BUG! Report to onnxruntime team.");
    }

    ORT_RETURN_IF_ERROR(status);
    dense.set_raw_data(std::move(dense_data_storage));

  } else {
    // No request for std::string
    status = UnsupportedSparseDataType()(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  }
  return status;
}

#if !defined(ORT_MINIMAL_BUILD)
// Determines if this is a type specific zero
using IsZeroFunc = bool (*)(const void*);
// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

static void SparsifyGeneric(const void* dense_raw_data, size_t n_dense_elements, size_t element_size,
                            IsZeroFunc is_zero, CopyElementFunc copy,
                            TensorProto& values, TensorProto& indices) {
  auto advance = [element_size](const void* start, size_t elements) -> const void* {
    return (reinterpret_cast<const uint8_t*>(start) + elements * element_size);
  };

  const auto* cbegin = dense_raw_data;
  const auto* const cend = advance(cbegin, n_dense_elements);
  auto& indices_data = *indices.mutable_int64_data();
  int64_t index = 0;
  while (cbegin != cend) {
    if (!is_zero(cbegin)) {
      indices_data.Add(index);
    }
    ++index;
    cbegin = advance(cbegin, 1U);
  }

  auto& raw_data = *values.mutable_raw_data();
  raw_data.resize(indices.int64_data_size() * element_size);
  void* data_dest = const_cast<char*>(raw_data.data());

  int64_t dest_index = 0;
  for (auto src_index : indices.int64_data()) {
    copy(data_dest, dense_raw_data, dest_index, src_index);
    ++dest_index;
  }
}

template <typename T>
bool IsZero(const void* p) {
  return (static_cast<T>(0) == *reinterpret_cast<const T*>(p));
}

template <typename T>
void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<T*>(dst)[dst_index] = reinterpret_cast<const T*>(src)[src_index];
}

common::Status DenseTensorToSparseTensorProto(const ONNX_NAMESPACE::TensorProto& dense_proto,
                                              const Path& model_path,
                                              ONNX_NAMESPACE::SparseTensorProto& result) {
  ORT_ENFORCE(HasDataType(dense_proto), "Must have a valid data type");

  const bool is_string_data = dense_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  if (is_string_data) {
    return UnsupportedSparseDataType()(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  }

  const auto data_type = dense_proto.data_type();
  SparseTensorProto sparse_proto;
  auto& values = *sparse_proto.mutable_values();
  values.set_name(dense_proto.name());
  values.set_data_type(data_type);

  auto& indices = *sparse_proto.mutable_indices();
  indices.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  SafeInt<size_t> n_dense_elements = 1;
  for (auto dim : dense_proto.dims()) {
    n_dense_elements *= dim;
  }

  size_t tensor_bytes_size = 0;
  std::unique_ptr<uint8_t[]> dense_raw_data;
  ORT_RETURN_IF_ERROR(UnpackInitializerData(dense_proto, model_path, dense_raw_data, tensor_bytes_size));
  size_t element_size = 0;
  MLTypeCallDispatcherRet<Status, GetElementSize, float, int8_t, uint8_t> type_disp(data_type);
  ORT_RETURN_IF_ERROR(type_disp.InvokeWithUnsupportedPolicy<UnsupportedSparseDataType>(element_size));

  switch (element_size) {
    case 1: {
      // bytes
      SparsifyGeneric(dense_raw_data.get(), n_dense_elements, element_size,
                      IsZero<uint8_t>, CopyElement<uint8_t>, values, indices);
      break;
    }
    case 4: {
      // float
      SparsifyGeneric(dense_raw_data.get(), n_dense_elements, element_size,
                      IsZero<uint32_t>, CopyElement<uint32_t>, values, indices);
      break;
    }
    default:
      ORT_THROW(false, "BUG! Report to onnxruntime team.");
  }

  // Fix up shapes
  const auto nnz = indices.int64_data_size();
  values.add_dims(nnz);
  indices.add_dims(nnz);

  // Save dense shape
  *sparse_proto.mutable_dims() = dense_proto.dims();
  swap(result, sparse_proto);
  return Status::OK();
}

#endif  // !ORT_MINIMAL_BUILD

template common::Status GetSizeInBytesFromTensorProto<kAllocAlignment>(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                                       size_t* out);
template common::Status GetSizeInBytesFromTensorProto<0>(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

#define CASE_UNPACK(TYPE, ELEMENT_TYPE, DATA_SIZE)                                         \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##TYPE: {                \
    if (initializer.data_location() == TensorProto_DataLocation_EXTERNAL) {                \
      ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(                                       \
          initializer,                                                                     \
          model_path.IsEmpty() ? nullptr : model_path.ParentPath().ToPathString().c_str(), \
          unpacked_tensor,                                                                 \
          tensor_byte_size));                                                              \
      tensor_data_length = tensor_byte_size;                                               \
      return Status::OK();                                                                 \
    } else {                                                                               \
      size_t element_count = 0;                                                            \
      if (initializer.has_raw_data()) {                                                    \
        tensor_byte_size = initializer.raw_data().size();                                  \
        element_count = tensor_byte_size / sizeof(ELEMENT_TYPE);                           \
      } else {                                                                             \
        element_count = initializer.DATA_SIZE();                                           \
        tensor_byte_size = element_count * sizeof(ELEMENT_TYPE);                           \
      }                                                                                    \
      tensor_data_length = tensor_byte_size;                                               \
      unpacked_tensor.reset(new uint8_t[tensor_data_length]);                              \
      return onnxruntime::utils::UnpackTensor(                                             \
          initializer,                                                                     \
          initializer.has_raw_data() ? initializer.raw_data().data() : nullptr,            \
          initializer.has_raw_data() ? initializer.raw_data().size() : 0,                  \
          reinterpret_cast<ELEMENT_TYPE*>(unpacked_tensor.get()), element_count);          \
    }                                                                                      \
    break;                                                                                 \
  }

Status UnpackInitializerData(const onnx::TensorProto& initializer,
                             const Path& model_path,
                             std::unique_ptr<uint8_t[]>& unpacked_tensor,
                             size_t& tensor_data_length) {
  SafeInt<size_t> tensor_byte_size = tensor_data_length;
  switch (initializer.data_type()) {
    CASE_UNPACK(FLOAT, float, float_data_size);
    CASE_UNPACK(DOUBLE, double, double_data_size);
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
    default:
      break;
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Unsupported type: ", initializer.data_type());
}
#undef CASE_UNPACK

}  // namespace utils
}  // namespace onnxruntime
