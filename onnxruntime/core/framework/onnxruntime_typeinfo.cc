// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//this file contains implementations of the C API

#include <cassert>
#include "onnxruntime_typeinfo.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_sequence_type_info.h"
#include "core/framework/TensorSeq.h"

using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::MLFloat16;
using onnxruntime::SparseTensor;
using onnxruntime::Tensor;
using onnxruntime::TensorShape;

namespace on = ONNX_NAMESPACE;

OrtTypeInfo::OrtTypeInfo(ONNXType type1) noexcept : type(type1) {
}

OrtTypeInfo::OrtTypeInfo(ONNXType type1, OrtTensorTypeAndShapeInfo* data1) noexcept : type(type1), data(data1) {
}

OrtTypeInfo::OrtTypeInfo(ONNXType type1, OrtMapTypeInfo* map_type_info1) noexcept : type(type1), map_type_info(map_type_info1) {
}

OrtTypeInfo::OrtTypeInfo(ONNXType type1, OrtSequenceTypeInfo* sequence_type_info1) noexcept : type(type1), sequence_type_info(sequence_type_info1) {
}

OrtTypeInfo::~OrtTypeInfo() {
  OrtApis::ReleaseTensorTypeAndShapeInfo(data);

  if (map_type_info) {
    OrtApis::ReleaseMapTypeInfo(map_type_info);
  }
  if (sequence_type_info) {
    OrtApis::ReleaseSequenceTypeInfo(sequence_type_info);
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetOnnxTypeFromTypeInfo, _In_ const struct OrtTypeInfo* input, _Out_ ONNXType* out) {
  *out = input->type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToTensorInfo, _In_ const struct OrtTypeInfo* input,
                    _Outptr_result_maybenull_ const struct OrtTensorTypeAndShapeInfo** out) {
  *out = input->type == ONNX_TYPE_TENSOR ? input->data : nullptr;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtMapTypeInfo** out) {
  API_IMPL_BEGIN
  *out = type_info->type == ONNX_TYPE_MAP ? type_info->map_type_info : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out) {
  API_IMPL_BEGIN
  *out = type_info->type == ONNX_TYPE_SEQUENCE ? type_info->sequence_type_info : nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetDenotationFromTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const char** const out,
                    _Out_ size_t* len) {
  API_IMPL_BEGIN
  *out = type_info->denotation.c_str();
  *len = type_info->denotation.size();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTypeInfo, _Frees_ptr_opt_ OrtTypeInfo* ptr) {
  delete ptr;
}

OrtStatus* GetTensorShapeAndType(const TensorShape& shape, const onnxruntime::DataTypeImpl& tensor_data_type,
                                 OrtTensorTypeAndShapeInfo** out);
OrtStatus* GetTensorShapeAndType(const TensorShape& shape, const std::vector<std::string>* dim_params,
                                 const ONNX_NAMESPACE::TypeProto& type_proto, OrtTensorTypeAndShapeInfo** out);

OrtStatus* OrtTypeInfo::FromOrtValue(const OrtValue& value, OrtTypeInfo** out) {
  onnxruntime::MLDataType type = value.Type();
  if (type == nullptr) {
    *out = new OrtTypeInfo(ONNX_TYPE_UNKNOWN);
    return nullptr;
  }

  // GetType<Tensor> and GetType<SparseTensor> do not have TypeProto populated because they return a static
  // TensorBase/SparseTensorBase instances, but other types are real MLDataTypes and they do have real protos
  // unless they are primitive data types, in which case we as before return them not implemented
  // however, this way we can support Opaque and we can avoid excessive calls to GetType()
  if (type->IsTensorType()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    const Tensor& tensor = value.Get<onnxruntime::Tensor>();
    const auto* tensor_data_type = tensor.DataType();
    if (tensor_data_type != nullptr) {
      OrtStatus* st = GetTensorShapeAndType(tensor.Shape(), *tensor_data_type, &info);
      if (st != nullptr)
        return st;
    }
    *out = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
    return nullptr;
  }

  if (type->IsSparseTensorType()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    const SparseTensor& tensor = value.Get<onnxruntime::SparseTensor>();
    const auto* tensor_data_type = tensor.DataType();
    if (tensor_data_type != nullptr) {
      OrtStatus* st = GetTensorShapeAndType(tensor.Shape(), *tensor_data_type, &info);
      if (st != nullptr) return st;
    }
    *out = new OrtTypeInfo(ONNX_TYPE_SPARSETENSOR, info);
    return nullptr;
  }

  if (type->IsTensorSequenceType()) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    const auto* tensor_data_type = value.Get<onnxruntime::TensorSeq>().DataType();
    if (tensor_data_type != nullptr) {
      TensorShape void_shape = {};
      OrtStatus* st = GetTensorShapeAndType(void_shape, *tensor_data_type, &info);
      if (st != nullptr) {
        return st;
      }

      auto element_type_info = new OrtTypeInfo(ONNX_TYPE_TENSOR, info);
      auto sequence_type_info = new OrtSequenceTypeInfo(element_type_info);
      *out = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, sequence_type_info);
      return nullptr;
    } else {
      return OrtApis::CreateStatus(ORT_FAIL, "OrtValue is TensorSequence type but has no element Tensor DataType.");
    }
  }

  const auto* type_proto = type->GetTypeProto();
  if (type_proto != nullptr) {
    // Place Opaque first as tensors will be mostly handled above and maps and sequences are not common
    switch (type_proto->value_case()) {
      case on::TypeProto::kOpaqueType: {
        *out = new OrtTypeInfo(ONNX_TYPE_OPAQUE);
        return nullptr;
      }
      case on::TypeProto::kMapType: {
        return OrtTypeInfo::FromTypeProto(type_proto, out);
      }
      case on::TypeProto::kSequenceType: {
        return OrtTypeInfo::FromTypeProto(type_proto, out);
      }
      // Real Tensor support
      case on::TypeProto::kTensorType:
      case on::TypeProto::kSparseTensorType: {
        return OrtApis::CreateStatus(ORT_FAIL, "Tensor types should have been handled already");
      }
      default:
        // NOT_IMPLEMENTED
        break;
    }
  }

  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}

const DataTypeImpl* OrtTypeInfo::ElementTypeFromProto(int type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetType<float>();
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return DataTypeImpl::GetType<bool>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return DataTypeImpl::GetType<int>();
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetType<double>();
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return DataTypeImpl::GetType<std::string>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return DataTypeImpl::GetType<int8_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return DataTypeImpl::GetType<uint8_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return DataTypeImpl::GetType<uint16_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return DataTypeImpl::GetType<int16_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return DataTypeImpl::GetType<int64_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return DataTypeImpl::GetType<uint32_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return DataTypeImpl::GetType<uint64_t>();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetType<MLFloat16>();
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetType<BFloat16>();

    default:
      ORT_NOT_IMPLEMENTED(__FUNCTION__, ":tensor type ", type, " is not supported");
  }
}

OrtStatus* OrtTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto* input, OrtTypeInfo** out) {
  auto value_case = input->value_case();
  switch (value_case) {
    case on::TypeProto::kTensorType:
    case on::TypeProto::kSparseTensorType: {
      ONNXType ten_type = ONNX_TYPE_UNKNOWN;
      const on::TypeProto_Tensor* tensor_type = nullptr;
      const on::TypeProto_SparseTensor* sparse_type = nullptr;
      const on::TensorShapeProto* sp = nullptr;
      if (value_case == on::TypeProto::kTensorType) {
        tensor_type = &input->tensor_type();
        ten_type = ONNX_TYPE_TENSOR;
        if (onnxruntime::utils::HasShape(*tensor_type)) {
          sp = &tensor_type->shape();
        }
      } else if (value_case == on::TypeProto::kSparseTensorType) {
        sparse_type = &input->sparse_tensor_type();
        ten_type = ONNX_TYPE_SPARSETENSOR;
        if (onnxruntime::utils::HasShape(*sparse_type)) {
          sp = &sparse_type->shape();
        }
      }

      OrtStatus* st = nullptr;
      OrtTensorTypeAndShapeInfo* info = nullptr;
      if (sp != nullptr) {
        const on::TensorShapeProto& s = *sp;
        std::vector<int64_t> dims(s.dim_size());
        std::vector<std::string> dim_params(s.dim_size());
        TensorShape shape_data(std::move(dims));
        for (int i = 0; i < s.dim_size(); ++i) {
          auto& t = s.dim(i);
          switch (t.value_case()) {
            case on::TensorShapeProto::Dimension::kDimValue:
              shape_data[i] = t.dim_value();
              break;
            case on::TensorShapeProto::Dimension::kDimParam:
              dim_params[i] = t.dim_param();
              // fall through
            case on::TensorShapeProto::Dimension::VALUE_NOT_SET:
              shape_data[i] = -1;
              break;
            default:
              assert(false);
          }
        }
        st = GetTensorShapeAndType(shape_data, &dim_params, *input, &info);
      } else {
        st = GetTensorShapeAndType(TensorShape(), nullptr, *input, &info);
      }
      if (st != nullptr) return st;
      auto type_info = new OrtTypeInfo(ten_type, info);
      type_info->denotation = input->denotation();
      *out = type_info;
      return nullptr;
    } break;
    case on::TypeProto::kSequenceType: {
      OrtSequenceTypeInfo* sequence_type_info = nullptr;

      if (auto status = OrtSequenceTypeInfo::FromTypeProto(input, &sequence_type_info)) {
        return status;
      }

      auto type_info = new OrtTypeInfo(ONNX_TYPE_SEQUENCE, sequence_type_info);
      type_info->denotation = input->denotation();
      *out = type_info;
      return nullptr;
    } break;
    case on::TypeProto::kMapType: {
      OrtMapTypeInfo* map_type_info = nullptr;

      if (auto status = OrtMapTypeInfo::FromTypeProto(input, &map_type_info)) {
        return status;
      }

      auto type_info = new OrtTypeInfo(ONNX_TYPE_MAP, map_type_info);
      type_info->denotation = input->denotation();
      *out = type_info;
      return nullptr;
    } break;
    case on::TypeProto::kOpaqueType: {
      auto type_info = new OrtTypeInfo(ONNX_TYPE_OPAQUE);
      type_info->denotation = input->denotation();
      *out = type_info;
      return nullptr;
    } break;
    case on::TypeProto::VALUE_NOT_SET:
      break;
    default:
      // Not implemented
      break;
  }
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}

OrtStatus* OrtTypeInfo::Clone(OrtTypeInfo** out) {
  switch (type) {
    case ONNX_TYPE_TENSOR:
    case ONNX_TYPE_SPARSETENSOR: {
      OrtTensorTypeAndShapeInfo* clone;
      if (auto status = data->Clone(&clone)) {
        return status;
      }
      *out = new OrtTypeInfo(type, clone);
      (*out)->denotation = denotation;
      return nullptr;
    }
    case ONNX_TYPE_SEQUENCE: {
      OrtSequenceTypeInfo* clone;
      if (auto status = sequence_type_info->Clone(&clone)) {
        return status;
      }
      *out = new OrtTypeInfo(type, clone);
      (*out)->denotation = denotation;
      return nullptr;
    }
    case ONNX_TYPE_MAP: {
      OrtMapTypeInfo* clone;
      if (auto status = map_type_info->Clone(&clone)) {
        return status;
      }
      *out = new OrtTypeInfo(type, clone);
      (*out)->denotation = denotation;
      return nullptr;
    }
    case ONNX_TYPE_OPAQUE: {
      *out = new OrtTypeInfo(type);
      (*out)->denotation = denotation;
      return nullptr;
    }
    default:
      // Not implemented
      break;
  }
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "not implemented");
}
