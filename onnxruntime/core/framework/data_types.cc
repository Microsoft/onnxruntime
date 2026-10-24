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

// Return the MLDataType used for a generic Tensor
template <>
MLDataType DataTypeImpl::GetType<Tensor>() {
  return TensorTypeBase::Type();
}

const MLFloat16 MLFloat16::NaN(MLFloat16::FromBits(MLFloat16::kPositiveQNaNBits));
const MLFloat16 MLFloat16::NegativeNaN(MLFloat16::FromBits(MLFloat16::kNegativeQNaNBits));
const MLFloat16 MLFloat16::Infinity(MLFloat16::FromBits(MLFloat16::kPositiveInfinityBits));
const MLFloat16 MLFloat16::NegativeInfinity(MLFloat16::FromBits(MLFloat16::kNegativeInfinityBits));
const MLFloat16 MLFloat16::MaxValue(MLFloat16::FromBits(MLFloat16::kMaxValueBits));
const MLFloat16 MLFloat16::MinValue(MLFloat16::FromBits(MLFloat16::kMinValueBits));
const MLFloat16 MLFloat16::Zero(MLFloat16::FromBits(0));
const MLFloat16 MLFloat16::One(MLFloat16::FromBits(MLFloat16::kOneBits));
const MLFloat16 MLFloat16::MinusOne(MLFloat16::FromBits(MLFloat16::kMinusOneBits));

const BFloat16 BFloat16::NaN(BFloat16::FromBits(BFloat16::kPositiveQNaNBits));
const BFloat16 BFloat16::NegativeNaN(BFloat16::FromBits(BFloat16::kNegativeQNaNBits));
const BFloat16 BFloat16::Infinity(BFloat16::FromBits(BFloat16::kPositiveInfinityBits));
const BFloat16 BFloat16::NegativeInfinity(BFloat16::FromBits(BFloat16::kNegativeInfinityBits));
const BFloat16 BFloat16::MaxValue(BFloat16::FromBits(BFloat16::kMaxValueBits));
const BFloat16 BFloat16::MinValue(BFloat16::FromBits(BFloat16::kMinValueBits));
const BFloat16 BFloat16::Zero(BFloat16::FromBits(0));
const BFloat16 BFloat16::One(BFloat16::FromBits(BFloat16::kOneBits));
const BFloat16 BFloat16::MinusOne(BFloat16::FromBits(BFloat16::kMinusOneBits));

}  // namespace onnxruntime

// This conflicts with the above GetType<>() specialization
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

#if !defined(DISABLE_SPARSE_TENSORS)
// Return the MLDataType used for a generic SparseTensor
template <>
MLDataType DataTypeImpl::GetType<SparseTensor>() {
  return SparseTensorTypeBase::Type();
}
#endif

template <>
MLDataType DataTypeImpl::GetType<TensorSeq>() {
  return SequenceTensorTypeBase::Type();
}

// static bool IsTensorTypeScalar(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_type_proto) {
//   int sz = tensor_type_proto.shape().dim_size();
//   return sz == 0 || sz == 1;
// }

namespace data_types_internal {

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

void CopyMutableOptionalElement(const ONNX_NAMESPACE::TypeProto& elem_proto,
                                ONNX_NAMESPACE::TypeProto& proto) {
  proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(elem_proto);
}

void AssignOpaqueDomainName(const char* domain, const char* name,
                            ONNX_NAMESPACE::TypeProto& proto) {
  auto* mutable_opaque = proto.mutable_opaque_type();
  mutable_opaque->mutable_domain()->assign(domain);
  mutable_opaque->mutable_name()->assign(name);
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_Tensor& type_proto);

#if !defined(DISABLE_SPARSE_TENSORS)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto);
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Optional& optional_proto,
                  const ONNX_NAMESPACE::TypeProto_Optional& type_proto);
#endif

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
#if !defined(DISABLE_SPARSE_TENSORS)
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(lhs.value_type().sparse_tensor_type(), rhs.value_type().sparse_tensor_type());
        break;
#endif
#if !defined(DISABLE_OPTIONAL_TYPE)
      case TypeProto::ValueCase::kOptionalType:
        result = IsCompatible(lhs.value_type().optional_type(), rhs.value_type().optional_type());
        break;
#endif
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

static bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto_1,
                         const ONNX_NAMESPACE::TypeProto& type_proto_2) {
  bool result = true;
  if (type_proto_1.value_case() == type_proto_2.value_case()) {
    switch (type_proto_1.value_case()) {
      case TypeProto::ValueCase::kTensorType:
        result = IsCompatible(type_proto_1.tensor_type(), type_proto_2.tensor_type());
        break;
      case TypeProto::ValueCase::kSequenceType:
        result = IsCompatible(type_proto_1.sequence_type(), type_proto_2.sequence_type());
        break;
#if !defined(DISABLE_ML_OPS)
      case TypeProto::ValueCase::kMapType:
        result = IsCompatible(type_proto_1.map_type(), type_proto_2.map_type());
        break;
#endif
      case TypeProto::ValueCase::kOpaqueType:
        result = IsCompatible(type_proto_1.opaque_type(), type_proto_2.opaque_type());
        break;
#if !defined(DISABLE_SPARSE_TENSORS)
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(type_proto_1.sparse_tensor_type(), type_proto_2.sparse_tensor_type());
        break;
#endif
#if !defined(DISABLE_OPTIONAL_TYPE)
      case TypeProto::ValueCase::kOptionalType:
        result = IsCompatible(type_proto_1.optional_type(), type_proto_2.optional_type());
        break;
#endif
      default:
        ORT_ENFORCE(false);
        break;
    }
  } else {
    result = false;
  }
  return result;
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Sequence& sequence_proto,
                  const ONNX_NAMESPACE::TypeProto_Sequence& type_proto) {
  return IsCompatible(sequence_proto.elem_type(), type_proto.elem_type());
}

#if !defined(DISABLE_OPTIONAL_TYPE)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Optional& optional_proto,
                  const ONNX_NAMESPACE::TypeProto_Optional& type_proto) {
  return IsCompatible(optional_proto.elem_type(), type_proto.elem_type());
}
#endif

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
#if !defined(DISABLE_SPARSE_TENSORS)
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto) {
  return type_proto.elem_type() == tensor_proto.elem_type();
}
#endif

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
  TypeProto& MutableTypeProto() {
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
// TODO: Fix the warning
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
TensorTypeBase::TensorTypeBase()
    : DataTypeImpl{DataTypeImpl::GeneralType::kTensor, sizeof(Tensor)},
      impl_(new Impl()) {}
TensorTypeBase::~TensorTypeBase() {
  delete impl_;
}

template <typename T>
static void Delete(void* p) {
  delete static_cast<T*>(p);
}

DeleteFunc TensorTypeBase::GetDeleteFunc() const {
  return &Delete<Tensor>;
}

ONNX_NAMESPACE::TypeProto& TensorTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
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

#if !defined(DISABLE_SPARSE_TENSORS)

/// SparseTensor

struct SparseTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

SparseTensorTypeBase::SparseTensorTypeBase()
    : DataTypeImpl{DataTypeImpl::GeneralType::kSparseTensor, sizeof(SparseTensor)},
      impl_(new Impl()) {}

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

DeleteFunc SparseTensorTypeBase::GetDeleteFunc() const {
  return &Delete<SparseTensor>;
}

const ONNX_NAMESPACE::TypeProto* SparseTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& SparseTensorTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
}

MLDataType SparseTensorTypeBase::Type() {
  static SparseTensorTypeBase sparse_tensor_base;
  return &sparse_tensor_base;
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

///// SequenceTensorTypeBase

struct SequenceTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

SequenceTensorTypeBase::SequenceTensorTypeBase()
    : DataTypeImpl{DataTypeImpl::GeneralType::kTensorSequence, sizeof(TensorSeq)},
      impl_(new Impl()) {}

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

  if (type_proto.value_case() != TypeProto::ValueCase::kSequenceType) {
    return false;
  }

  return data_types_internal::IsCompatible(thisProto->sequence_type(), type_proto.sequence_type());
}

DeleteFunc SequenceTensorTypeBase::GetDeleteFunc() const {
  return &Delete<TensorSeq>;
}

const ONNX_NAMESPACE::TypeProto* SequenceTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& SequenceTensorTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
}

MLDataType SequenceTensorTypeBase::Type() {
  static SequenceTensorTypeBase sequence_tensor_base;
  return &sequence_tensor_base;
}

#if !defined(DISABLE_OPTIONAL_TYPE)
///// OptionalTypeBase

struct OptionalTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

OptionalTypeBase::OptionalTypeBase() : DataTypeImpl{DataTypeImpl::GeneralType::kOptional, 0},
                                       impl_(new Impl()) {}

OptionalTypeBase::~OptionalTypeBase() {
  delete impl_;
}

bool OptionalTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kOptionalType) {
    return false;
  }

  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kOptionalType);
  ORT_ENFORCE(utils::HasElemType(thisProto->optional_type()));

  return data_types_internal::IsCompatible(thisProto->optional_type(), type_proto.optional_type());
}

const ONNX_NAMESPACE::TypeProto* OptionalTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& OptionalTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
}

MLDataType OptionalTypeBase::Type() {
  static OptionalTypeBase optional_type_base;
  return &optional_type_base;
}
#endif

/// DisabledTypeBase

#if defined(DISABLE_OPTIONAL_TYPE)
struct DisabledTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

DisabledTypeBase::DisabledTypeBase(DataTypeImpl::GeneralType type, size_t size)
    : DataTypeImpl{type, size}, impl_(new Impl()) {}

DisabledTypeBase::~DisabledTypeBase() {
  delete impl_;
}

const ONNX_NAMESPACE::TypeProto* DisabledTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& DisabledTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
}

MLDataType DisabledTypeBase::Type() {
  static DisabledTypeBase disabled_base{GeneralType::kInvalid, 0};
  return &disabled_base;
}
#endif

/// NoTensorTypeBase
struct NonTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {};

NonTensorTypeBase::NonTensorTypeBase(size_t size)
    : DataTypeImpl{DataTypeImpl::GeneralType::kNonTensor, size},
      impl_(new Impl()) {
}

// The suppressed warning is: "The type with a virtual function needs either public virtual or protected nonvirtual destructor."
// However, we do not allocate this type on heap.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26436)
#endif
NonTensorTypeBase::~NonTensorTypeBase() {
  delete impl_;
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

ONNX_NAMESPACE::TypeProto& NonTensorTypeBase::MutableTypeProto() {
  return impl_->MutableTypeProto();
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
ORT_REGISTER_TENSOR_TYPE(Float6E2M3);
ORT_REGISTER_TENSOR_TYPE(Float6E3M2);

#if !defined(DISABLE_FLOAT8_TYPES)
ORT_REGISTER_TENSOR_TYPE(Float8E4M3FN);
ORT_REGISTER_TENSOR_TYPE(Float8E4M3FNUZ);
ORT_REGISTER_TENSOR_TYPE(Float8E5M2);
ORT_REGISTER_TENSOR_TYPE(Float8E5M2FNUZ);
ORT_REGISTER_TENSOR_TYPE(Float8E8M0);
#endif

#if !defined(DISABLE_FLOAT4_TYPES)
ORT_REGISTER_TENSOR_TYPE(Float4E2M1x2);
#endif

ORT_REGISTER_TENSOR_TYPE(Int4x2);
ORT_REGISTER_TENSOR_TYPE(UInt4x2);

ORT_REGISTER_TENSOR_TYPE(Int2x4);
ORT_REGISTER_TENSOR_TYPE(UInt2x4);

#if !defined(DISABLE_SPARSE_TENSORS)
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

#if !defined(DISABLE_FLOAT8_TYPES)
ORT_REGISTER_SPARSE_TENSOR_TYPE(Float8E4M3FN);
ORT_REGISTER_SPARSE_TENSOR_TYPE(Float8E4M3FNUZ);
ORT_REGISTER_SPARSE_TENSOR_TYPE(Float8E5M2);
ORT_REGISTER_SPARSE_TENSOR_TYPE(Float8E5M2FNUZ);
ORT_REGISTER_SPARSE_TENSOR_TYPE(Float8E8M0);
#endif

#endif

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
ORT_REGISTER_SEQ_TENSOR_TYPE(Float6E2M3);
ORT_REGISTER_SEQ_TENSOR_TYPE(Float6E3M2);

#if !defined(DISABLE_FLOAT8_TYPES)

ORT_REGISTER_SEQ_TENSOR_TYPE(Float8E4M3FN);
ORT_REGISTER_SEQ_TENSOR_TYPE(Float8E4M3FNUZ);
ORT_REGISTER_SEQ_TENSOR_TYPE(Float8E5M2);
ORT_REGISTER_SEQ_TENSOR_TYPE(Float8E5M2FNUZ);
ORT_REGISTER_SEQ_TENSOR_TYPE(Float8E8M0);

#endif

ORT_REGISTER_SEQ_TENSOR_TYPE(Int4x2);
ORT_REGISTER_SEQ_TENSOR_TYPE(UInt4x2);

ORT_REGISTER_SEQ_TENSOR_TYPE(Int2x4);
ORT_REGISTER_SEQ_TENSOR_TYPE(UInt2x4);

#if !defined(DISABLE_ML_OPS)
ORT_REGISTER_SEQ(VectorMapStringToFloat);
ORT_REGISTER_SEQ(VectorMapInt64ToFloat);
#endif

#if !defined(DISABLE_FLOAT8_TYPES)

#define ORT_REGISTER_OPTIONAL_ORT_TYPE(ORT_TYPE)        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int32_t);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, float);          \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, bool);           \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, std::string);    \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int8_t);         \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint8_t);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint16_t);       \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int16_t);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int64_t);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, double);         \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint32_t);       \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint64_t);       \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, MLFloat16);      \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, BFloat16);       \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float6E2M3);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float6E3M2);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float8E4M3FN);   \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float8E4M3FNUZ); \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float8E5M2);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float8E5M2FNUZ); \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float8E8M0);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Int4x2);         \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, UInt4x2);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Int2x4);         \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, UInt2x4);

#else

#define ORT_REGISTER_OPTIONAL_ORT_TYPE(ORT_TYPE)     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int32_t);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, float);       \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, bool);        \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, std::string); \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int8_t);      \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint8_t);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint16_t);    \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int16_t);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, int64_t);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, double);      \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint32_t);    \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, uint64_t);    \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, MLFloat16);   \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, BFloat16);    \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float6E2M3);  \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Float6E3M2);  \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Int4x2);      \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, UInt4x2);     \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, Int2x4);      \
  ORT_REGISTER_OPTIONAL_TYPE(ORT_TYPE, UInt2x4);

#endif

ORT_REGISTER_OPTIONAL_ORT_TYPE(Tensor)
ORT_REGISTER_OPTIONAL_ORT_TYPE(TensorSeq)

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

#if !defined(DISABLE_OPTIONAL_TYPE)
#define REGISTER_OPTIONAL_PROTO(ORT_TYPE, TYPE, reg_fn)                  \
  {                                                                      \
    MLDataType mltype = DataTypeImpl::GetOptionalType<ORT_TYPE, TYPE>(); \
    reg_fn(mltype);                                                      \
  }
#endif

#if !defined(DISABLE_SPARSE_TENSORS)
#define REGISTER_SPARSE_TENSOR_PROTO(TYPE, reg_fn)                 \
  {                                                                \
    MLDataType mltype = DataTypeImpl::GetSparseTensorType<TYPE>(); \
    reg_fn(mltype);                                                \
  }
#endif

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
  REGISTER_TENSOR_PROTO(Float6E2M3, reg_fn);
  REGISTER_TENSOR_PROTO(Float6E3M2, reg_fn);
#if !defined(DISABLE_FLOAT8_TYPES)
  REGISTER_TENSOR_PROTO(Float8E4M3FN, reg_fn);
  REGISTER_TENSOR_PROTO(Float8E4M3FNUZ, reg_fn);
  REGISTER_TENSOR_PROTO(Float8E5M2, reg_fn);
  REGISTER_TENSOR_PROTO(Float8E5M2FNUZ, reg_fn);
  REGISTER_TENSOR_PROTO(Float8E8M0, reg_fn);
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  REGISTER_TENSOR_PROTO(Float4E2M1x2, reg_fn);
#endif
  REGISTER_TENSOR_PROTO(Int4x2, reg_fn);
  REGISTER_TENSOR_PROTO(UInt4x2, reg_fn);
  REGISTER_TENSOR_PROTO(Int2x4, reg_fn);
  REGISTER_TENSOR_PROTO(UInt2x4, reg_fn);

#if !defined(DISABLE_SPARSE_TENSORS)
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
#if !defined(DISABLE_FLOAT8_TYPES)
  REGISTER_SPARSE_TENSOR_PROTO(Float8E4M3FN, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(Float8E4M3FNUZ, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(Float8E5M2, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(Float8E5M2FNUZ, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(Float8E8M0, reg_fn);
#endif
#endif

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
  REGISTER_SEQ_TENSOR_PROTO(Float6E2M3, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Float6E3M2, reg_fn);

#if !defined(DISABLE_FLOAT8_TYPES)

  REGISTER_SEQ_TENSOR_PROTO(Float8E4M3FN, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Float8E4M3FNUZ, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Float8E5M2, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Float8E5M2FNUZ, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Float8E8M0, reg_fn);

#endif

  REGISTER_SEQ_TENSOR_PROTO(Int4x2, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(UInt4x2, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(Int2x4, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(UInt2x4, reg_fn);

#if !defined(DISABLE_ML_OPS)
  REGISTER_ONNX_PROTO(VectorMapStringToFloat, reg_fn);
  REGISTER_ONNX_PROTO(VectorMapInt64ToFloat, reg_fn);
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)

#if !defined(DISABLE_FLOAT8_TYPES)

#define REGISTER_OPTIONAL_PROTO_ORT_TYPE(ORT_TYPE, reg_fn)   \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int32_t, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, float, reg_fn);          \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, bool, reg_fn);           \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, std::string, reg_fn);    \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int8_t, reg_fn);         \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint8_t, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint16_t, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int16_t, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int64_t, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, double, reg_fn);         \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint32_t, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint64_t, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, MLFloat16, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, BFloat16, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float6E2M3, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float6E3M2, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float8E4M3FN, reg_fn);   \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float8E4M3FNUZ, reg_fn); \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float8E5M2, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float8E5M2FNUZ, reg_fn); \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float8E8M0, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Int4x2, reg_fn);         \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, UInt4x2, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Int2x4, reg_fn);         \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, UInt2x4, reg_fn);

#else

#define REGISTER_OPTIONAL_PROTO_ORT_TYPE(ORT_TYPE, reg_fn) \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int32_t, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, float, reg_fn);        \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, bool, reg_fn);         \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, std::string, reg_fn);  \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int8_t, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint8_t, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint16_t, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int16_t, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, int64_t, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, double, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint32_t, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, uint64_t, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, MLFloat16, reg_fn);    \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, BFloat16, reg_fn);     \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float6E2M3, reg_fn);   \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Float6E3M2, reg_fn);   \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Int4x2, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, UInt4x2, reg_fn);      \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, Int2x4, reg_fn);       \
  REGISTER_OPTIONAL_PROTO(ORT_TYPE, UInt2x4, reg_fn);

#endif

  REGISTER_OPTIONAL_PROTO_ORT_TYPE(Tensor, reg_fn);
  REGISTER_OPTIONAL_PROTO_ORT_TYPE(TensorSeq, reg_fn);
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
      case TensorProto_DataType_FLOAT8E4M3FN:
        return "Float8E4M3FN";
      case TensorProto_DataType_FLOAT8E4M3FNUZ:
        return "Float8E4M3FNUZ";
      case TensorProto_DataType_FLOAT8E5M2:
        return "Float8E5M2";
      case TensorProto_DataType_FLOAT8E5M2FNUZ:
        return "Float8E5M2FNUZ";
      case TensorProto_DataType_FLOAT8E8M0:
        return "Float8E8M0";
      case TensorProto_DataType_FLOAT4E2M1:
        return "Float4E2M1";
      case TensorProto_DataType_FLOAT6E2M3:
        return "Float6E2M3";
      case TensorProto_DataType_FLOAT6E3M2:
        return "Float6E3M2";
      case TensorProto_DataType_INT4:
        return "Int4x2";
      case TensorProto_DataType_UINT4:
        return "UInt4x2";
      case TensorProto_DataType_INT2:
        return "Int2x4";
      case TensorProto_DataType_UINT2:
        return "UInt2x4";
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

#if !defined(DISABLE_FLOAT8_TYPES)
    case TensorProto_DataType_FLOAT8E4M3FN:
      return DataTypeImpl::GetTensorType<Float8E4M3FN>()->AsTensorType();
    case TensorProto_DataType_FLOAT8E4M3FNUZ:
      return DataTypeImpl::GetTensorType<Float8E4M3FNUZ>()->AsTensorType();
    case TensorProto_DataType_FLOAT8E5M2:
      return DataTypeImpl::GetTensorType<Float8E5M2>()->AsTensorType();
    case TensorProto_DataType_FLOAT8E5M2FNUZ:
      return DataTypeImpl::GetTensorType<Float8E5M2FNUZ>()->AsTensorType();
    case TensorProto_DataType_FLOAT8E8M0:
      return DataTypeImpl::GetTensorType<Float8E8M0>()->AsTensorType();
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
    case TensorProto_DataType_FLOAT4E2M1:
      return DataTypeImpl::GetTensorType<Float4E2M1x2>()->AsTensorType();
#endif
    case TensorProto_DataType_FLOAT6E2M3:
      return DataTypeImpl::GetTensorType<Float6E2M3>()->AsTensorType();
    case TensorProto_DataType_FLOAT6E3M2:
      return DataTypeImpl::GetTensorType<Float6E3M2>()->AsTensorType();
    case TensorProto_DataType_INT4:
      return DataTypeImpl::GetTensorType<Int4x2>()->AsTensorType();
    case TensorProto_DataType_UINT4:
      return DataTypeImpl::GetTensorType<UInt4x2>()->AsTensorType();
    case TensorProto_DataType_INT2:
      return DataTypeImpl::GetTensorType<Int2x4>()->AsTensorType();
    case TensorProto_DataType_UINT2:
      return DataTypeImpl::GetTensorType<UInt2x4>()->AsTensorType();

    default:
      ORT_NOT_IMPLEMENTED("tensor type ", type, " is not supported");
  }
}

const SequenceTensorTypeBase* DataTypeImpl::SequenceTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetSequenceTensorType<float>()->AsSequenceTensorType();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetSequenceTensorType<bool>()->AsSequenceTensorType();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetSequenceTensorType<int32_t>()->AsSequenceTensorType();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetSequenceTensorType<double>()->AsSequenceTensorType();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetSequenceTensorType<std::string>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetSequenceTensorType<uint8_t>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetSequenceTensorType<uint16_t>()->AsSequenceTensorType();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetSequenceTensorType<int8_t>()->AsSequenceTensorType();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetSequenceTensorType<int16_t>()->AsSequenceTensorType();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetSequenceTensorType<int64_t>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetSequenceTensorType<uint32_t>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetSequenceTensorType<uint64_t>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetSequenceTensorType<MLFloat16>()->AsSequenceTensorType();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetSequenceTensorType<BFloat16>()->AsSequenceTensorType();

#if !defined(DISABLE_FLOAT8_TYPES)

    case TensorProto_DataType_FLOAT8E4M3FN:
      return DataTypeImpl::GetSequenceTensorType<Float8E4M3FN>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT8E4M3FNUZ:
      return DataTypeImpl::GetSequenceTensorType<Float8E4M3FNUZ>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT8E5M2:
      return DataTypeImpl::GetSequenceTensorType<Float8E5M2>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT8E5M2FNUZ:
      return DataTypeImpl::GetSequenceTensorType<Float8E5M2FNUZ>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT8E8M0:
      return DataTypeImpl::GetSequenceTensorType<Float8E8M0>()->AsSequenceTensorType();

#endif
    case TensorProto_DataType_INT4:
      return DataTypeImpl::GetSequenceTensorType<Int4x2>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT4:
      return DataTypeImpl::GetSequenceTensorType<UInt4x2>()->AsSequenceTensorType();
    case TensorProto_DataType_INT2:
      return DataTypeImpl::GetSequenceTensorType<Int2x4>()->AsSequenceTensorType();
    case TensorProto_DataType_UINT2:
      return DataTypeImpl::GetSequenceTensorType<UInt2x4>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT6E2M3:
      return DataTypeImpl::GetSequenceTensorType<Float6E2M3>()->AsSequenceTensorType();
    case TensorProto_DataType_FLOAT6E3M2:
      return DataTypeImpl::GetSequenceTensorType<Float6E3M2>()->AsSequenceTensorType();

    default:
      ORT_NOT_IMPLEMENTED("sequence tensor type ", type, " is not supported");
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
const SparseTensorTypeBase* DataTypeImpl::SparseTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetSparseTensorType<float>()->AsSparseTensorType();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetSparseTensorType<bool>()->AsSparseTensorType();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetSparseTensorType<int32_t>()->AsSparseTensorType();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetSparseTensorType<double>()->AsSparseTensorType();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetSparseTensorType<std::string>()->AsSparseTensorType();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetSparseTensorType<uint8_t>()->AsSparseTensorType();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetSparseTensorType<uint16_t>()->AsSparseTensorType();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetSparseTensorType<int8_t>()->AsSparseTensorType();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetSparseTensorType<int16_t>()->AsSparseTensorType();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetSparseTensorType<int64_t>()->AsSparseTensorType();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetSparseTensorType<uint32_t>()->AsSparseTensorType();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetSparseTensorType<uint64_t>()->AsSparseTensorType();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetSparseTensorType<MLFloat16>()->AsSparseTensorType();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetSparseTensorType<BFloat16>()->AsSparseTensorType();

#if !defined(DISABLE_FLOAT8_TYPES)

    case TensorProto_DataType_FLOAT8E4M3FN:
      return DataTypeImpl::GetSparseTensorType<Float8E4M3FN>()->AsSparseTensorType();
    case TensorProto_DataType_FLOAT8E4M3FNUZ:
      return DataTypeImpl::GetSparseTensorType<Float8E4M3FNUZ>()->AsSparseTensorType();
    case TensorProto_DataType_FLOAT8E5M2:
      return DataTypeImpl::GetSparseTensorType<Float8E5M2>()->AsSparseTensorType();
    case TensorProto_DataType_FLOAT8E5M2FNUZ:
      return DataTypeImpl::GetSparseTensorType<Float8E5M2FNUZ>()->AsSparseTensorType();
    case TensorProto_DataType_FLOAT8E8M0:
      return DataTypeImpl::GetSparseTensorType<Float8E8M0>()->AsSparseTensorType();
#endif

    default:
      ORT_NOT_IMPLEMENTED("sparse tensor type ", type, " is not supported");
  }
}
#endif

MLDataType DataTypeImpl::TypeFromProto(const ONNX_NAMESPACE::TypeProto& proto) {
  const auto& registry = data_types_internal::DataTypeRegistry::instance();

  MLDataType type = registry.GetMLDataType(proto);
  if (type == nullptr) {
    DataType str_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(proto);
    ORT_NOT_IMPLEMENTED("MLDataType for: ", *str_type, " is not currently registered or supported");
  }
  return type;
}

// Below are the types the we need to execute the runtime
// They are not compatible with TypeProto in ONNX.
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
ORT_REGISTER_PRIM_TYPE(Float6E2M3);
ORT_REGISTER_PRIM_TYPE(Float6E3M2);

#if !defined(DISABLE_FLOAT8_TYPES)

ORT_REGISTER_PRIM_TYPE(Float8E4M3FN);
ORT_REGISTER_PRIM_TYPE(Float8E4M3FNUZ);
ORT_REGISTER_PRIM_TYPE(Float8E5M2);
ORT_REGISTER_PRIM_TYPE(Float8E5M2FNUZ);
ORT_REGISTER_PRIM_TYPE(Float8E8M0);

#endif

#if !defined(DISABLE_FLOAT4_TYPES)

ORT_REGISTER_PRIM_SUBBYTE_TYPE(Float4E2M1x2, 2);

#endif

ORT_REGISTER_PRIM_SUBBYTE_TYPE(Int4x2, 2);
ORT_REGISTER_PRIM_SUBBYTE_TYPE(UInt4x2, 2);
ORT_REGISTER_PRIM_SUBBYTE_TYPE(Int2x4, 4);
ORT_REGISTER_PRIM_SUBBYTE_TYPE(UInt2x4, 4);

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
struct GetOptionalTensorTypesImpl {
  std::vector<MLDataType> operator()() const {
    return {DataTypeImpl::GetOptionalType<Tensor, ElementTypes>()...};
  }
};

template <typename L>
std::vector<MLDataType> GetOptionalTensorTypesFromTypeList() {
  return boost::mp11::mp_apply<GetOptionalTensorTypesImpl, L>{}();
}

template <typename... ElementTypes>
struct GetOptionalSequenceTensorTypesImpl {
  std::vector<MLDataType> operator()() const {
    return {DataTypeImpl::GetOptionalType<TensorSeq, ElementTypes>()...};
  }
};

template <typename L>
std::vector<MLDataType> GetOptionalSequenceTensorTypesFromTypeList() {
  return boost::mp11::mp_apply<GetOptionalSequenceTensorTypesImpl, L>{}();
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

const std::vector<MLDataType>& DataTypeImpl::AllIEEEFloatTensorTypes() {
  static std::vector<MLDataType> all_IEEE_float_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIeeeFloat>();
  return all_IEEE_float_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypes() {
  return AllFixedSizeTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypesIRv4() {
  static std::vector<MLDataType> all_fixed_size_tensor_types_ir4 =
      GetTensorTypesFromTypeList<element_type_lists::AllFixedSizeIRv4>();
  return all_fixed_size_tensor_types_ir4;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypesIRv9() {
  static std::vector<MLDataType> all_fixed_size_tensor_types_ir9 =
      GetTensorTypesFromTypeList<element_type_lists::AllFixedSizeIRv9>();
  return all_fixed_size_tensor_types_ir9;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypes() {
  return AllTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv4() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv4>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv9() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv9>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv10() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv10>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv11() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv11>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv13() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv13>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypesIRv14() {
  static std::vector<MLDataType> all_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllIRv14>();
  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeSequenceTensorTypes() {
  return AllFixedSizeSequenceTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeSequenceTensorTypesIRv4() {
  static std::vector<MLDataType> all_fixed_size_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::AllFixedSizeIRv4>();
  return all_fixed_size_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeSequenceTensorTypesIRv9() {
  static std::vector<MLDataType> all_fixed_size_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::AllFixedSizeIRv9>();
  return all_fixed_size_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllSequenceTensorTypes() {
  return AllSequenceTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllSequenceTensorTypesIRv4() {
  static std::vector<MLDataType> all_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::AllIRv4>();
  return all_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllSequenceTensorTypesIRv9() {
  static std::vector<MLDataType> all_sequence_tensor_types =
      GetSequenceTensorTypesFromTypeList<element_type_lists::AllIRv9>();
  return all_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllNumericTensorTypes() {
  static std::vector<MLDataType> all_numeric_size_tensor_types =
      GetTensorTypesFromTypeList<element_type_lists::AllNumeric>();
  return all_numeric_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes() {
  return AllFixedSizeTensorAndSequenceTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv4() {
  static std::vector<MLDataType> all_fixed_size_tensor_and_sequence_tensor_types =
      []() {
        auto temp = AllFixedSizeTensorTypesIRv4();
        const auto& seq = AllFixedSizeSequenceTensorTypesIRv4();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_fixed_size_tensor_and_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9() {
  static std::vector<MLDataType> all_fixed_size_tensor_and_sequence_tensor_types =
      []() {
        auto temp = AllFixedSizeTensorTypesIRv9();
        const auto& seq = AllFixedSizeSequenceTensorTypesIRv9();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_fixed_size_tensor_and_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorTypes() {
  return AllTensorAndSequenceTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorTypesIRv4() {
  static std::vector<MLDataType> all_tensor_and_sequence_types_with_float8 =
      []() {
        auto temp = AllTensorTypesIRv4();
        const auto& seq = AllSequenceTensorTypesIRv4();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();
  return all_tensor_and_sequence_types_with_float8;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorTypesIRv9() {
  static std::vector<MLDataType> all_tensor_and_sequence_types_with_float8 =
      []() {
        auto temp = AllTensorTypesIRv9();
        const auto& seq = AllSequenceTensorTypesIRv9();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();
  return all_tensor_and_sequence_types_with_float8;
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypes() {
  return AllOptionalAndTensorAndSequenceTensorTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypesIRv4() {
  static std::vector<MLDataType> all_optional_and_tensor_and_sequence_types =
      []() {
        auto temp = AllOptionalTypesIRv4();
        const auto tensor = AllTensorTypesIRv4();
        temp.insert(temp.end(), tensor.begin(), tensor.end());
        const auto& seq = AllSequenceTensorTypesIRv4();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_optional_and_tensor_and_sequence_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypesIRv9() {
  static std::vector<MLDataType> all_optional_and_tensor_and_sequence_types =
      []() {
        auto temp = AllOptionalTypesIRv9();
        const auto tensor = AllTensorTypesIRv9();
        temp.insert(temp.end(), tensor.begin(), tensor.end());
        const auto& seq = AllSequenceTensorTypesIRv9();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_optional_and_tensor_and_sequence_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalTypes() {
  return AllOptionalTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalTypesIRv4() {
  static std::vector<MLDataType> all_optional_types =
      []() {
        auto temp = GetOptionalTensorTypesFromTypeList<element_type_lists::AllIRv4>();
        const auto& seq = GetOptionalSequenceTensorTypesFromTypeList<element_type_lists::AllIRv4>();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_optional_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllOptionalTypesIRv9() {
  static std::vector<MLDataType> all_optional_types =
      []() {
        auto temp = GetOptionalTensorTypesFromTypeList<element_type_lists::AllIRv9>();
        const auto& seq = GetOptionalSequenceTensorTypesFromTypeList<element_type_lists::AllIRv9>();
        temp.insert(temp.end(), seq.begin(), seq.end());
        return temp;
      }();

  return all_optional_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes() {
  return AllTensorAndSequenceTensorAndOptionalTypesIRv4();
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypesIRv4() {
  static std::vector<MLDataType> all_tensor_and_sequence_types_and_optional_types =
      []() {
        auto temp = AllTensorTypesIRv4();
        const auto& seq = AllSequenceTensorTypesIRv4();
        const auto& opt = AllOptionalTypes();
        temp.insert(temp.end(), seq.begin(), seq.end());
        temp.insert(temp.end(), opt.begin(), opt.end());
        return temp;
      }();

  return all_tensor_and_sequence_types_and_optional_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypesIRv9() {
  static std::vector<MLDataType> all_tensor_and_sequence_types_and_optional_types =
      []() {
        auto temp = AllTensorTypesIRv9();
        const auto& seq = AllSequenceTensorTypesIRv9();
        const auto& opt = AllOptionalTypes();
        temp.insert(temp.end(), seq.begin(), seq.end());
        temp.insert(temp.end(), opt.begin(), opt.end());
        return temp;
      }();

  return all_tensor_and_sequence_types_and_optional_types;
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
  auto base_type = ml_type->AsNonTensorType();
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
#if !defined(DISABLE_OPTIONAL_TYPE)
        case TypeProto::ValueCase::kOptionalType:
          types_.emplace_back(ContainerType::kOptional, TensorProto_DataType_UNDEFINED);
          type_proto = &type_proto->optional_type().elem_type();
          break;
#endif
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
  auto base_type = ml_type->AsNonTensorType();
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
