// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"

#include "boost/mp11.hpp"

#include "core/framework/data_types_internal.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/data_type_utils.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

MLFloat16::MLFloat16(float f) : val{math::floatToHalf(f)} {}

float MLFloat16::ToFloat() const {
  return math::halfToFloat(val);
}

// Return the MLDataType used for a generic Tensor
template <>
MLDataType DataTypeImpl::GetType<Tensor>() {
  return TensorTypeBase::Type();
}

}  // namespace onnxruntime

// This conflics with the above GetType<>() specialization
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

// Return the MLDataType used for a generic SparseTensor
template <>
MLDataType DataTypeImpl::GetType<SparseTensor>() {
  return SparseTensorTypeBase::Type();
}

template <>
MLDataType DataTypeImpl::GetType<TensorSeq>() {
  return SequenceTensorTypeBase::Type();
}

//static bool IsTensorTypeScalar(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_type_proto) {
//  int sz = tensor_type_proto.shape().dim_size();
//  return sz == 0 || sz == 1;
//}

namespace data_types_internal {

template <typename T>
struct TensorElementTypeSetter<T> {
  static void SetTensorElementType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  }
  static void SetSparseTensorElementType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_sparse_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  }

#if !defined(DISABLE_ML_OPS)
  static void SetMapKeyType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_map_type()->set_key_type(utils::ToTensorProtoElementType<T>());
  }
#endif

  constexpr static int32_t GetElementType() {
    return utils::ToTensorProtoElementType<T>();
  }
};

// Pre-instantiate
template struct
    TensorElementTypeSetter<float>;
template struct
    TensorElementTypeSetter<uint8_t>;
template struct
    TensorElementTypeSetter<int8_t>;
template struct
    TensorElementTypeSetter<uint16_t>;
template struct
    TensorElementTypeSetter<int16_t>;
template struct
    TensorElementTypeSetter<int32_t>;
template struct
    TensorElementTypeSetter<int64_t>;
template struct
    TensorElementTypeSetter<std::string>;
template struct
    TensorElementTypeSetter<bool>;
template struct
    TensorElementTypeSetter<MLFloat16>;
template struct
    TensorElementTypeSetter<double>;
template struct
    TensorElementTypeSetter<uint32_t>;
template struct
    TensorElementTypeSetter<uint64_t>;
template struct
    TensorElementTypeSetter<BFloat16>;

#if !defined(DISABLE_ML_OPS)
void CopyMutableMapValue(const ONNX_NAMESPACE::TypeProto& value_proto,
                         ONNX_NAMESPACE::TypeProto& map_proto) {
  map_proto.mutable_map_type()->mutable_value_type()->CopyFrom(value_proto);
}
#endif

void CopyMutableSeqElement(const ONNX_NAMESPACE::TypeProto& elem_proto,
                           ONNX_NAMESPACE::TypeProto& proto) {
  proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(elem_proto);
}

void AssignOpaqueDomainName(const char* domain, const char* name,
                            ONNX_NAMESPACE::TypeProto& proto) {
  auto* mutable_opaque = proto.mutable_opaque_type();
  mutable_opaque->mutable_domain()->assign(domain);
  mutable_opaque->mutable_name()->assign(name);
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_Tensor& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto);

#if !defined(DISABLE_ML_OPS)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Map& map_proto,
                  const ONNX_NAMESPACE::TypeProto_Map& type_proto);
#endif

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Sequence& sequence_proto,
                  const ONNX_NAMESPACE::TypeProto_Sequence& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Opaque& opaque_proto,
                  const ONNX_NAMESPACE::TypeProto_Opaque& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_Tensor& type_proto) {
  return type_proto.elem_type() == tensor_proto.elem_type();
  /* Currently all Tensors with all kinds of shapes
     are mapped into the same MLDataType (same element type)
     so we omit shape from IsCompatible consideration
     */
}

#if !defined(DISABLE_ML_OPS)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Map& map_proto,
                  const ONNX_NAMESPACE::TypeProto_Map& type_proto) {
  const auto& lhs = map_proto;
  const auto& rhs = type_proto;
  bool result = true;
  if (lhs.key_type() == rhs.key_type() &&
      lhs.value_type().value_case() == rhs.value_type().value_case()) {
    switch (lhs.value_type().value_case()) {
      case TypeProto::ValueCase::kTensorType:
        result = IsCompatible(lhs.value_type().tensor_type(), rhs.value_type().tensor_type());
        break;
      case TypeProto::ValueCase::kSequenceType:
        result = IsCompatible(lhs.value_type().sequence_type(), rhs.value_type().sequence_type());
        break;
      case TypeProto::ValueCase::kMapType:
        result = IsCompatible(lhs.value_type().map_type(), rhs.value_type().map_type());
        break;
      case TypeProto::ValueCase::kOpaqueType:
        result = IsCompatible(lhs.value_type().opaque_type(), rhs.value_type().opaque_type());
        break;
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(lhs.value_type().sparse_tensor_type(), rhs.value_type().sparse_tensor_type());
        break;
      default:
        ORT_ENFORCE(false);
        break;
    }
  } else {
    result = false;
  }
  return result;
}
#endif

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Sequence& sequence_proto,
                  const ONNX_NAMESPACE::TypeProto_Sequence& type_proto) {
  bool result = true;
  const auto& lhs = sequence_proto;
  const auto& rhs = type_proto;
  if (lhs.elem_type().value_case() == rhs.elem_type().value_case()) {
    switch (lhs.elem_type().value_case()) {
      case TypeProto::ValueCase::kTensorType:
        result = IsCompatible(lhs.elem_type().tensor_type(), rhs.elem_type().tensor_type());
        break;
      case TypeProto::ValueCase::kSequenceType:
        result = IsCompatible(lhs.elem_type().sequence_type(), rhs.elem_type().sequence_type());
        break;
#if !defined(DISABLE_ML_OPS)
      case TypeProto::ValueCase::kMapType:
        result = IsCompatible(lhs.elem_type().map_type(), rhs.elem_type().map_type());
        break;
#endif
      case TypeProto::ValueCase::kOpaqueType:
        result = IsCompatible(lhs.elem_type().opaque_type(), rhs.elem_type().opaque_type());
        break;
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(lhs.elem_type().sparse_tensor_type(), rhs.elem_type().sparse_tensor_type());
        break;
      default:
        ORT_ENFORCE(false);
        break;
    }
  } else {
    result = false;
  }
  return result;
}
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Opaque& opaque_proto,
                  const ONNX_NAMESPACE::TypeProto_Opaque& type_proto) {
  const auto& lhs = opaque_proto;
  const auto& rhs = type_proto;
  bool lhs_domain = utils::HasDomain(lhs);
  bool rhs_domain = utils::HasDomain(rhs);

  if ((lhs_domain != rhs_domain) ||
      (lhs_domain && rhs_domain && lhs.domain() != lhs.domain())) {
    return false;
  }

  bool lhs_name = utils::HasName(lhs);
  bool rhs_name = utils::HasName(rhs);

  return !((lhs_name != rhs_name) ||
           (lhs_name && rhs_name && lhs.name() != rhs.name()));
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto) {
  return type_proto.elem_type() == tensor_proto.elem_type();
}

void RegisterAllProtos(const std::function<void(MLDataType)>& /*reg_fn*/);

class DataTypeRegistry {
  std::unordered_map<DataType, MLDataType> mapping_;

  DataTypeRegistry() {
    RegisterAllProtos([this](MLDataType mltype) { RegisterDataType(mltype); });
  }

  ~DataTypeRegistry() = default;

 public:
  DataTypeRegistry(const DataTypeRegistry&) = delete;
  DataTypeRegistry& operator=(const DataTypeRegistry&) = delete;

  static DataTypeRegistry& instance() {
    static DataTypeRegistry inst;
    return inst;
  }

  void RegisterDataType(MLDataType mltype) {
    using namespace ONNX_NAMESPACE;
    const auto* proto = mltype->GetTypeProto();
    ORT_ENFORCE(proto != nullptr, "Only ONNX MLDataType can be registered");
    DataType type = Utils::DataTypeUtils::ToType(*proto);
    auto p = mapping_.insert(std::make_pair(type, mltype));
    ORT_ENFORCE(p.second, "We do not expect duplicate registration of types for: ", type);
  }

  MLDataType GetMLDataType(const ONNX_NAMESPACE::TypeProto& proto) const {
    using namespace ONNX_NAMESPACE;
    DataType type = Utils::DataTypeUtils::ToType(proto);
    auto p = mapping_.find(type);
    if (p != mapping_.end()) {
      return p->second;
    }
    return nullptr;
  }

  MLDataType GetMLDataType(const std::string& data_type) const {
    using namespace ONNX_NAMESPACE;
    DataType dtype = Utils::DataTypeUtils::ToType(data_type);
    if (dtype == nullptr) {
      return nullptr;
    }
    auto hit = mapping_.find(dtype);
    if (hit == mapping_.end()) {
      return nullptr;
    }
    return hit->second;
  }
};

struct TypeProtoImpl {
  const TypeProto* GetProto() const {
    return &proto_;
  }
  TypeProto& mutable_type_proto() {
    return proto_;
  }

  TypeProto proto_;
};

}  // namespace data_types_internal

/// TensorTypeBase
struct TensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

const ONNX_NAMESPACE::TypeProto* TensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

TensorTypeBase::TensorTypeBase() : impl_(new Impl()) {}
TensorTypeBase::~TensorTypeBase() {
  delete impl_;
}

size_t TensorTypeBase::Size() const {
  return sizeof(Tensor);
}

template <typename T>
static void Delete(void* p) {
  delete static_cast<T*>(p);
}

DeleteFunc TensorTypeBase::GetDeleteFunc() const {
  return &Delete<Tensor>;
}

ONNX_NAMESPACE::TypeProto& TensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

bool TensorTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kTensorType);
  ORT_ENFORCE(utils::HasElemType(thisProto->tensor_type()));

  if (&type_proto == thisProto) {
    return true;
  }

  if (type_proto.value_case() != TypeProto::ValueCase::kTensorType) {
    return false;
  }

  return data_types_internal::IsCompatible(thisProto->tensor_type(), type_proto.tensor_type());
}

MLDataType TensorTypeBase::Type() {
  static TensorTypeBase tensor_base;
  return &tensor_base;
}

/// SparseTensor

struct SparseTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

SparseTensorTypeBase::SparseTensorTypeBase() : impl_(new Impl()) {}
SparseTensorTypeBase::~SparseTensorTypeBase() {
  delete impl_;
}

bool SparseTensorTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kSparseTensorType) {
    return false;
  }

  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kSparseTensorType);
  ORT_ENFORCE(utils::HasElemType(thisProto->sparse_tensor_type()));

  return data_types_internal::IsCompatible(thisProto->sparse_tensor_type(), type_proto.sparse_tensor_type());
}

size_t SparseTensorTypeBase::Size() const {
  return sizeof(SparseTensor);
}

DeleteFunc SparseTensorTypeBase::GetDeleteFunc() const {
  return &Delete<SparseTensor>;
}

const ONNX_NAMESPACE::TypeProto* SparseTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& SparseTensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

MLDataType SparseTensorTypeBase::Type() {
  static SparseTensorTypeBase sparse_tensor_base;
  return &sparse_tensor_base;
}

///// SequenceTensorTypeBase

struct SequenceTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

SequenceTensorTypeBase::SequenceTensorTypeBase() : impl_(new Impl()) {}

SequenceTensorTypeBase::~SequenceTensorTypeBase() {
  delete impl_;
}

bool SequenceTensorTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kSequenceType) {
    return false;
  }

  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kSequenceType);
  ORT_ENFORCE(utils::HasElemType(thisProto->sequence_type()));

  return data_types_internal::IsCompatible(thisProto->sequence_type(), type_proto.sequence_type());
}

size_t SequenceTensorTypeBase::Size() const {
  return sizeof(TensorSeq);
}

DeleteFunc SequenceTensorTypeBase::GetDeleteFunc() const {
  return &Delete<TensorSeq>;
}

const ONNX_NAMESPACE::TypeProto* SequenceTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& SequenceTensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

MLDataType SequenceTensorTypeBase::Type() {
  static SequenceTensorTypeBase sequence_tensor_base;
  return &sequence_tensor_base;
}

/// NoTensorTypeBase
struct NonTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {};

NonTensorTypeBase::NonTensorTypeBase() : impl_(new Impl()) {
}

NonTensorTypeBase::~NonTensorTypeBase() {
  delete impl_;
}

ONNX_NAMESPACE::TypeProto& NonTensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

const ONNX_NAMESPACE::TypeProto* NonTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

#if !defined(DISABLE_ML_OPS)
bool NonTensorTypeBase::IsMapCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kMapType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kMapType);
  ORT_ENFORCE(utils::HasKeyType(thisProto->map_type()));
  ORT_ENFORCE(utils::HasKeyType(thisProto->map_type()));
  return data_types_internal::IsCompatible(thisProto->map_type(), type_proto.map_type());
}
#endif

bool NonTensorTypeBase::IsSequenceCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kSequenceType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kSequenceType);
  ORT_ENFORCE(utils::HasElemType(thisProto->sequence_type()));
  return data_types_internal::IsCompatible(thisProto->sequence_type(), type_proto.sequence_type());
}

bool NonTensorTypeBase::IsOpaqueCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kOpaqueType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kOpaqueType);
  return data_types_internal::IsCompatible(thisProto->opaque_type(), type_proto.opaque_type());
}

// The below two APIs must be implemented in the derived types to be used
void NonTensorTypeBase::FromDataContainer(const void* /* data */, size_t /*data_size*/, OrtValue& /* output */) const {
  ORT_ENFORCE(false, "Not implemented");
}

void NonTensorTypeBase::ToDataContainer(const OrtValue& /* input */, size_t /*data_size */, void* /* data */) const {
  ORT_ENFORCE(false, "Not implemented");
}

ORT_REGISTER_TENSOR_TYPE(int32_t);
ORT_REGISTER_TENSOR_TYPE(float);
ORT_REGISTER_TENSOR_TYPE(bool);
ORT_REGISTER_TENSOR_TYPE(std::string);
ORT_REGISTER_TENSOR_TYPE(int8_t);
ORT_REGISTER_TENSOR_TYPE(uint8_t);
ORT_REGISTER_TENSOR_TYPE(uint16_t);
ORT_REGISTER_TENSOR_TYPE(int16_t);
ORT_REGISTER_TENSOR_TYPE(int64_t);
ORT_REGISTER_TENSOR_TYPE(double);
ORT_REGISTER_TENSOR_TYPE(uint32_t);
ORT_REGISTER_TENSOR_TYPE(uint64_t);
ORT_REGISTER_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_TENSOR_TYPE(BFloat16);

ORT_REGISTER_SPARSE_TENSOR_TYPE(int32_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(float);
ORT_REGISTER_SPARSE_TENSOR_TYPE(bool);
ORT_REGISTER_SPARSE_TENSOR_TYPE(std::string);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int8_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint8_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint16_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int16_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int64_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(double);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint32_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint64_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_SPARSE_TENSOR_TYPE(BFloat16);

#if !defined(DISABLE_ML_OPS)
ORT_REGISTER_MAP(MapStringToString);
ORT_REGISTER_MAP(MapStringToInt64);
ORT_REGISTER_MAP(MapStringToFloat);
ORT_REGISTER_MAP(MapStringToDouble);
ORT_REGISTER_MAP(MapInt64ToString);
ORT_REGISTER_MAP(MapInt64ToInt64);
ORT_REGISTER_MAP(MapInt64ToFloat);
ORT_REGISTER_MAP(MapInt64ToDouble);
#endif

ORT_REGISTER_SEQ_TENSOR_TYPE(float);
ORT_REGISTER_SEQ_TENSOR_TYPE(double);
ORT_REGISTER_SEQ_TENSOR_TYPE(int8_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint8_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(int16_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint16_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(int32_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint32_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(int64_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint64_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(bool);
ORT_REGISTER_SEQ_TENSOR_TYPE(std::string);
ORT_REGISTER_SEQ_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_SEQ_TENSOR_TYPE(BFloat16);

#if !defined(DISABLE_ML_OPS)
ORT_REGISTER_SEQ(VectorMapStringToFloat);
ORT_REGISTER_SEQ(VectorMapInt64ToFloat);
#endif

// Used for Tensor Proto registrations
#define REGISTER_TENSOR_PROTO(TYPE, reg_fn)                  \
  {                                                          \
    MLDataType mltype = DataTypeImpl::GetTensorType<TYPE>(); \
    reg_fn(mltype);                                          \
  }

#define REGISTER_SEQ_TENSOR_PROTO(TYPE, reg_fn)                      \
  {                                                                  \
    MLDataType mltype = DataTypeImpl::GetSequenceTensorType<TYPE>(); \
    reg_fn(mltype);                                                  \
  }

#define REGISTER_SPARSE_TENSOR_PROTO(TYPE, reg_fn)                 \
  {                                                                \
    MLDataType mltype = DataTypeImpl::GetSparseTensorType<TYPE>(); \
    reg_fn(mltype);                                                \
  }

#define REGISTER_ONNX_PROTO(TYPE, reg_fn)              \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    reg_fn(mltype);                                    \
  }

namespace data_types_internal {

void RegisterAllProtos(const std::function<void(MLDataType)>& reg_fn) {
  REGISTER_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_TENSOR_PROTO(float, reg_fn);
  REGISTER_TENSOR_PROTO(bool, reg_fn);
  REGISTER_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_TENSOR_PROTO(double, reg_fn);
  REGISTER_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_TENSOR_PROTO(BFloat16, reg_fn);

  REGISTER_SPARSE_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(float, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(bool, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(double, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(BFloat16, reg_fn);

#if !defined(DISABLE_ML_OPS)
  REGISTER_ONNX_PROTO(MapStringToString, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToInt64, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToFloat, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToDouble, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToString, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToInt64, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToFloat, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToDouble, reg_fn);
#endif

  REGISTER_SEQ_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(float, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(bool, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(double, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(BFloat16, reg_fn);

#if !defined(DISABLE_ML_OPS)
  REGISTER_ONNX_PROTO(VectorMapStringToFloat, reg_fn);
  REGISTER_ONNX_PROTO(VectorMapInt64ToFloat, reg_fn);
#endif
}
}  // namespace data_types_internal

void DataTypeImpl::RegisterDataType(MLDataType mltype) {
  data_types_internal::DataTypeRegistry::instance().RegisterDataType(mltype);
}

MLDataType DataTypeImpl::GetDataType(const std::string& data_type) {
  return data_types_internal::DataTypeRegistry::instance().GetMLDataType(data_type);
}

const char* DataTypeImpl::ToString(MLDataType type) {
  if (type == nullptr)
    return "(null)";

  auto prim_type = type->AsPrimitiveDataType();
  if (prim_type != nullptr) {
    switch (prim_type->GetDataType()) {
      case TensorProto_DataType_FLOAT:
        return "float";
      case TensorProto_DataType_BOOL:
        return "bool";
      case TensorProto_DataType_DOUBLE:
        return "double";
      case TensorProto_DataType_STRING:
        return "string";
      case TensorProto_DataType_INT8:
        return "int8";
      case TensorProto_DataType_UINT8:
        return "uint8";
      case TensorProto_DataType_INT16:
        return "int16";
      case TensorProto_DataType_UINT16:
        return "uint16";
      case TensorProto_DataType_INT32:
        return "int32";
      case TensorProto_DataType_UINT32:
        return "uint32";
      case TensorProto_DataType_INT64:
        return "int64";
      case TensorProto_DataType_UINT64:
        return "uint64";
      case TensorProto_DataType_FLOAT16:
        return "float16";
      case TensorProto_DataType_BFLOAT16:
        return "bfloat16";
      default:
        break;
    }
  }
  auto type_proto = type->GetTypeProto();
  if (type_proto != nullptr) {
    return ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto)->c_str();
  }
#ifdef ORT_NO_RTTI
  return "(unknown type)";
#else
  // TODO: name() method of `type_info` class is implementation dependent
  // and may return a mangled non-human readable string which may have to be unmangled
  return typeid(*type).name();
#endif
}

std::vector<std::string> DataTypeImpl::ToString(const std::vector<MLDataType>& types) {
  std::vector<std::string> type_strs;
  for (const auto& type : types) {
    type_strs.push_back(DataTypeImpl::ToString(type));
  }
  return type_strs;
}

const TensorTypeBase* DataTypeImpl::TensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetTensorType<float>()->AsTensorType();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetTensorType<bool>()->AsTensorType();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetTensorType<int32_t>()->AsTensorType();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetTensorType<double>()->AsTensorType();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetTensorType<std::string>()->AsTensorType();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetTensorType<uint8_t>()->AsTensorType();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetTensorType<uint16_t>()->AsTensorType();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetTensorType<int8_t>()->AsTensorType();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetTensorType<int16_t>()->AsTensorType();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetTensorType<int64_t>()->AsTensorType();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetTensorType<uint32_t>()->AsTensorType();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetTensorType<uint64_t>()->AsTensorType();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetTensorType<MLFloat16>()->AsTensorType();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetTensorType<BFloat16>()->AsTensorType();
    default:
      ORT_NOT_IMPLEMENTED("tensor type ", type, " is not supported");
  }
}

const NonTensorTypeBase* DataTypeImpl::SequenceTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetSequenceTensorType<float>()->AsNonTensorTypeBase();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetSequenceTensorType<bool>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetSequenceTensorType<int32_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetSequenceTensorType<double>()->AsNonTensorTypeBase();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetSequenceTensorType<std::string>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetSequenceTensorType<uint8_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetSequenceTensorType<uint16_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetSequenceTensorType<int8_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetSequenceTensorType<int16_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetSequenceTensorType<int64_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetSequenceTensorType<uint32_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetSequenceTensorType<uint64_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetSequenceTensorType<MLFloat16>()->AsNonTensorTypeBase();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetSequenceTensorType<BFloat16>()->AsNonTensorTypeBase();
    default:
      ORT_NOT_IMPLEMENTED("tensor type ", type, " is not supported");
  }
}

const SparseTensorTypeBase* DataTypeImpl::SparseTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<float>());
    case TensorProto_DataType_BOOL:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<bool>());
    case TensorProto_DataType_INT32:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int32_t>());
    case TensorProto_DataType_DOUBLE:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<double>());
    case TensorProto_DataType_STRING:
     return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<std::string>());
    case TensorProto_DataType_UINT8:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint8_t>());
    case TensorProto_DataType_UINT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint16_t>());
    case TensorProto_DataType_INT8:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int8_t>());
    case TensorProto_DataType_INT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int16_t>());
    case TensorProto_DataType_INT64:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int64_t>());
    case TensorProto_DataType_UINT32:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint32_t>());
    case TensorProto_DataType_UINT64:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint64_t>());
    case TensorProto_DataType_FLOAT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<MLFloat16>());
    case TensorProto_DataType_BFLOAT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<BFloat16>());
    default:
      ORT_NOT_IMPLEMENTED("sparse tensor type ", type, " is not supported");
  }
}

MLDataType DataTypeImpl::TypeFromProto(const ONNX_NAMESPACE::TypeProto& proto) {
  const auto& registry = data_types_internal::DataTypeRegistry::instance();

  MLDataType type = registry.GetMLDataType(proto);
  if (type == nullptr) {
    DataType str_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(proto);
    ORT_NOT_IMPLEMENTED("MLDataType for: ", *str_type, " is not currently registered or supported");
  }
  return type;
}

//Below are the types the we need to execute the runtime
//They are not compatible with TypeProto in ONNX.
ORT_REGISTER_PRIM_TYPE(int32_t);
ORT_REGISTER_PRIM_TYPE(float);
ORT_REGISTER_PRIM_TYPE(bool);
ORT_REGISTER_PRIM_TYPE(std::string);
ORT_REGISTER_PRIM_TYPE(int8_t);
ORT_REGISTER_PRIM_TYPE(uint8_t);
ORT_REGISTER_PRIM_TYPE(uint16_t);
ORT_REGISTER_PRIM_TYPE(int16_t);
ORT_REGISTER_PRIM_TYPE(int64_t);
ORT_REGISTER_PRIM_TYPE(double);
ORT_REGISTER_PRIM_TYPE(uint32_t);
ORT_REGISTER_PRIM_TYPE(uint64_t);
ORT_REGISTER_PRIM_TYPE(MLFloat16);
ORT_REGISTER_PRIM_TYPE(BFloat16);

namespace {
template <typename... ElementTypes>
struct GetTensorTypesImpl {
  std::vector<MLDataType> operator()() const {
    return {DataTypeImpl::GetTensorType<ElementTypes>()...};
  }
};

template <typename L>
std::vector<MLDataType> GetTensorTypesFromTypeList() {
  return boost::mp11::mp_apply<GetTensorTypesImpl, L>{}();
}

template <typename... ElementTypes>
struct GetSequenceTensorTypesImpl {
  std::vector<MLDataType> operator()() const {
    return {DataTypeImpl::GetSequenceTensorType<ElementTypes>()...};
  }
};

template <typename L>
std::vector<MLDataType> GetSequenceTensorTypesFromTypeList() {
  return boost::mp11::mp_apply<GetSequenceTensorTypesImpl, L>{}();
}
}  // namespace

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorExceptHalfTypes() {
  static std::vector<MLDataType> all_fixed_size_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllFixedSizeExceptHalf>();
  return all_fixed_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes() {
  static std::vector<MLDataType> all_IEEE_float_tensor_except_half_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIeeeFloatExceptHalf>();
  return all_IEEE_float_tensor_except_half_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllIEEEFloatTensorTypes() {
  static std::vector<MLDataType> all_IEEE_float_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIeeeFloat>();
  return all_IEEE_float_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypes() {
  static std::vector<MLDataType> all_fixed_size_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllFixedSize>();
  return all_fixed_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypes() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::All>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeSequenceTensorTypes() {
  static std::vector<MLDataType> all_fixed_size_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::AllFixedSize>();
  return all_fixed_size_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllSequenceTensorTypes() {
  static std::vector<MLDataType> all_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::All>();
  return all_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllNumericTensorTypes() {
  static std::vector<MLDataType> all_numeric_size_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllNumeric>();
  return all_numeric_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes() {
  static std::vector<MLDataType> all_fixed_size_tensor_and_sequence_tensor_types =
      []() {
        auto temp = AllFixedSizeTensorTypes();
        const auto& seq = AllFixedSizeSequenceTensorTypes();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_fixed_size_tensor_and_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorTypes() {
  static std::vector<MLDataType> all_tensor_and_sequence_types =
      []() {
        auto temp = AllTensorTypes();
        const auto& seq = AllSequenceTensorTypes();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_tensor_and_sequence_types;
}

// helper to stream. expected to only be used for error output, so any typeid lookup
// cost should be fine. alternative would be to add a static string field to DataTypeImpl
// that we set in the register macro to the type name, and output that instead.
std::ostream& operator<<(std::ostream& out, const DataTypeImpl* data_type) {
  if (data_type == nullptr)
    return out << "(null)";

#ifdef ORT_NO_RTTI
  return out << "(unknown type)";
#else
  return out << typeid(*data_type).name();
#endif
}

namespace utils {

ContainerChecker::ContainerChecker(MLDataType ml_type) {
  using namespace ONNX_NAMESPACE;
  using namespace data_types_internal;
  auto base_type = ml_type->AsNonTensorTypeBase();
  if (base_type == nullptr) {
    types_.emplace_back(ContainerType::kUndefined,
                        TensorProto_DataType_UNDEFINED);
  } else {
    auto type_proto = base_type->GetTypeProto();
    assert(type_proto != nullptr);
    while (type_proto != nullptr) {
      auto value_case = type_proto->value_case();
      switch (value_case) {
        // Terminal case
        case TypeProto::ValueCase::kTensorType:
          types_.emplace_back(ContainerType::kTensor, type_proto->tensor_type().elem_type());
          type_proto = nullptr;
          break;
#if !defined(DISABLE_ML_OPS)
        case TypeProto::ValueCase::kMapType: {
          const auto& map_type = type_proto->map_type();
          types_.emplace_back(ContainerType::kMap, map_type.key_type());
          // Move on handling the value
          type_proto = &map_type.value_type();
        } break;
#endif
        case TypeProto::ValueCase::kSequenceType:
          types_.emplace_back(ContainerType::kSequence, TensorProto_DataType_UNDEFINED);
          type_proto = &type_proto->sequence_type().elem_type();
          break;
        case TypeProto::ValueCase::kOpaqueType:
          // We do not handle this and terminate here
          types_.emplace_back(ContainerType::kOpaque,
                              TensorProto_DataType_UNDEFINED);
          type_proto = nullptr;
          break;
        default:
          ORT_ENFORCE(false, "Invalid DataTypeImpl TypeProto definition");
      }
    }
  }
}

bool IsOpaqueType(MLDataType ml_type, const char* domain, const char* name) {
  auto base_type = ml_type->AsNonTensorTypeBase();
  if (base_type == nullptr) {
    return false;
  }
  auto type_proto = base_type->GetTypeProto();
  assert(type_proto != nullptr);
  if (type_proto->value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kOpaqueType) {
    const auto& op_proto = type_proto->opaque_type();
    return (op_proto.domain() == domain &&
            op_proto.name() == name);
  }
  return false;
}

}  // namespace utils
}  // namespace onnxruntime
