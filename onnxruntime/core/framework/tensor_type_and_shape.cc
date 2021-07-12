// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_type_and_shape.h"

#include <assert.h>
#include <atomic>
#include <stdexcept>

#include "core/common/safeint.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ml_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::MLFloat16;
using onnxruntime::SparseTensor;
using onnxruntime::Tensor;

ORT_API_STATUS_IMPL(OrtApis::CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  *out = new OrtTensorTypeAndShapeInfo();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTensorTypeAndShapeInfo, _Frees_ptr_opt_ OrtTensorTypeAndShapeInfo* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo* this_ptr,
                    enum ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
  this_ptr->type = type;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SetDimensions, OrtTensorTypeAndShapeInfo* this_ptr, _In_ const int64_t* dim_values, size_t dim_count) {
  API_IMPL_BEGIN
  this_ptr->shape = onnxruntime::TensorShape(dim_values, dim_count);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorElementType, _In_ const struct OrtTensorTypeAndShapeInfo* info, _Out_ ONNXTensorElementDataType* out) {
  *out = info->type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetDimensionsCount, _In_ const struct OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out) {
  *out = info->shape.NumDimensions();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetDimensions, _In_ const struct OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  info->shape.CopyDims(dim_values, dim_values_length);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetSymbolicDimensions, _In_ const struct OrtTensorTypeAndShapeInfo* info,
                    _Out_writes_all_(dim_params_length) const char** names, size_t dim_params_length) {
  for (size_t idx = 0, end = std::min(info->dim_params.size(), dim_params_length); idx < end; ++idx) {
    names[idx] = info->dim_params[idx].c_str();
  }

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* this_ptr, _Out_ size_t* out) {
  API_IMPL_BEGIN
  *out = SafeInt<size_t>{this_ptr->shape.Size()};
  return nullptr;
  API_IMPL_END
}

struct OrtValue;

ONNXTensorElementDataType TensorDataTypeToOnnxRuntimeTensorElementDataType(
    int32_t dtype) {
  namespace o = ONNX_NAMESPACE;
  ONNXTensorElementDataType type;
  switch (dtype) {
    case o::TensorProto_DataType_FLOAT:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    case o::TensorProto_DataType_DOUBLE:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      break;
    case o::TensorProto_DataType_FLOAT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      break;
    case o::TensorProto_DataType_BFLOAT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      break;
    case o::TensorProto_DataType_INT8:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      break;
    case o::TensorProto_DataType_UINT8:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      break;
    case o::TensorProto_DataType_INT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      break;
    case o::TensorProto_DataType_UINT16:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      break;
    case o::TensorProto_DataType_INT32:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      break;
    case o::TensorProto_DataType_UINT32:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
      break;
    case o::TensorProto_DataType_INT64:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      break;
    case o::TensorProto_DataType_UINT64:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
      break;
    case o::TensorProto_DataType_STRING:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
      break;
    case o::TensorProto_DataType_BOOL:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      break;
    default:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      break;
  }
  return type;
}

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(
    const onnxruntime::DataTypeImpl* cpp_type) {
  auto prim_type = cpp_type->AsPrimitiveDataType();
  if (prim_type == nullptr) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return TensorDataTypeToOnnxRuntimeTensorElementDataType(prim_type->GetDataType());
}

OrtStatus* GetTensorShapeAndTypeHelper(ONNXTensorElementDataType type, const onnxruntime::TensorShape shape,
                                       const std::vector<std::string>* dim_params, OrtTensorTypeAndShapeInfo** out) {
  OrtTensorTypeAndShapeInfo* ret;
  if (auto* status = OrtApis::CreateTensorTypeAndShapeInfo(&ret))
    return status;
  if (auto* status = OrtApis::SetTensorElementType(ret, type)) {
    OrtApis::ReleaseTensorTypeAndShapeInfo(ret);
    return status;
  }

  auto* status = OrtApis::SetDimensions(ret, shape.GetDims().data(), shape.GetDims().size());
  if (status != nullptr) {
    OrtApis::ReleaseTensorTypeAndShapeInfo(ret);
    return status;
  }

  if (dim_params != nullptr) {
    ret->dim_params = *dim_params;
  } else {
    // we expect to be being called with a concrete shape so validate that
    assert(shape.Size() >= 0);
    ret->dim_params.resize(shape.NumDimensions(), "");
  }

  *out = ret;
  return nullptr;
}

OrtStatus* GetTensorShapeAndType(const onnxruntime::TensorShape& shape,
                                 const onnxruntime::DataTypeImpl& tensor_data_type, OrtTensorTypeAndShapeInfo** out) {
  ONNXTensorElementDataType type = MLDataTypeToOnnxRuntimeTensorElementDataType(&tensor_data_type);
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Not implemented");
  }
  return GetTensorShapeAndTypeHelper(type, shape, nullptr, out);
}

OrtStatus* GetTensorShapeAndType(const onnxruntime::TensorShape& shape, const std::vector<std::string>* dim_params,
                                 const ONNX_NAMESPACE::TypeProto& type_proto, OrtTensorTypeAndShapeInfo** out) {
  auto value_case = type_proto.value_case();
  assert(value_case == ONNX_NAMESPACE::TypeProto::kTensorType ||
         value_case == ONNX_NAMESPACE::TypeProto::kSparseTensorType);

  auto dtype = (value_case == ONNX_NAMESPACE::TypeProto::kTensorType) ? type_proto.tensor_type().elem_type()
                                                                      : type_proto.sparse_tensor_type().elem_type();
  ONNXTensorElementDataType type = TensorDataTypeToOnnxRuntimeTensorElementDataType(dtype);
  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED == type) {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Not implemented");
  }
  return GetTensorShapeAndTypeHelper(type, shape, dim_params, out);
}

OrtStatus* OrtTensorTypeAndShapeInfo::Clone(OrtTensorTypeAndShapeInfo** out) {
  return GetTensorShapeAndTypeHelper(type, shape, &dim_params, out);
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorTypeAndShape, _In_ const OrtValue* v, _Outptr_ OrtTensorTypeAndShapeInfo** out) {
  API_IMPL_BEGIN
  onnxruntime::MLDataType type = v->Type();
  ORT_ENFORCE(type != nullptr, "OrtValue is not a Tensor");
  if (type->IsTensorType() || type->IsSparseTensorType()) {
    const onnxruntime::TensorShape* shape = nullptr;
    onnxruntime::MLDataType data_type = nullptr;
    if (type->IsTensorType()) {
      const Tensor& tensor = v->Get<onnxruntime::Tensor>();
      shape = &tensor.Shape();
      data_type = tensor.DataType();
    } else {
      const SparseTensor& tensor = v->Get<onnxruntime::SparseTensor>();
      shape = &tensor.Shape();
      data_type = tensor.DataType();
    }
    return GetTensorShapeAndType(*shape, *data_type, out);
  } else {
    ORT_THROW("Argument is not a tensor");
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetValueType, _In_ const OrtValue* v, _Out_ ONNXType* out) {
  API_IMPL_BEGIN
  OrtTypeInfo* type_info;
  auto status = OrtTypeInfo::FromOrtValue(*v, &type_info);
  if (status != nullptr)
    return status;

  *out = type_info->type;
  OrtApis::ReleaseTypeInfo(type_info);
  return nullptr;
  API_IMPL_END
}

/**
* Get the type information of an OrtValue
* \param value
* \return The returned value should be freed by OrtReleaseTypeInfo after use
*/
ORT_API_STATUS_IMPL(OrtApis::GetTypeInfo, _In_ const OrtValue* v, _Outptr_result_maybenull_ struct OrtTypeInfo** out) {
  API_IMPL_BEGIN
  // TODO: This is consistent with the previous implementation but inconsistent with GetValueType which returns
  // ONNX_TYPE_UNKNOWN if v->Type() is null. Should we instead just call OrtTypeInfo::FromOrtValue and
  // return an OrtTypeInfo value in 'out' with type set to ONNX_TYPE_UNKNOWN? Or is the inconsistency fine?
  if (v->Type() == nullptr) {
    *out = nullptr;
    return nullptr;
  }

  auto status = OrtTypeInfo::FromOrtValue(*v, out);
  return status;
  API_IMPL_END
}