// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
namespace onnxruntime {

extern ProviderHost* g_host;
using ProviderType = const std::string&;

struct CPUIDInfo final {
  static const CPUIDInfo& GetCPUIDInfo() { return g_host->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return g_host->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return g_host->CPUIDInfo__HasAVX512f(this); }
  bool HasAVX512_BF16() const { return g_host->CPUIDInfo__HasAVX512_BF16(this); }
  bool HasAMX_BF16() const { return g_host->CPUIDInfo__HasAMX_BF16(this); }
  bool HasAVX512Skylake() const { return g_host->CPUIDInfo__HasAVX512Skylake(this); }

  PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger final {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return g_host->logging__Logger__OutputIsEnabled(this, severity, data_type); }
  Severity GetSeverity() const noexcept {
    return g_host->logging__Logger__GetSeverity(this);
  }

  PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager final {
  static const Logger& DefaultLogger() { return g_host->logging__LoggingManager__DefaultLogger(); }
  static bool HasDefaultLogger() { return g_host->logging__LoggingManager__HasDefaultLogger(); }

  PROVIDER_DISALLOW_ALL(LoggingManager)
};

struct Capture final {
  static std::unique_ptr<Capture> Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType data_type, const CodeLocation& location) {
    return g_host->logging__Capture__construct(logger, severity, category, data_type, location);
  }
  static void operator delete(void* p) { g_host->logging__Capture__operator_delete(reinterpret_cast<Capture*>(p)); }

  std::ostream& Stream() noexcept { return g_host->logging__Capture__Stream(this); }
  void ProcessPrintf(const char* format, va_list args) { g_host->logging__Capture__ProcessPrintf(this, format, args); }

  Capture() = delete;
  Capture(const Capture&) = delete;
  void operator=(const Capture&) = delete;
};

#if defined(_WIN32)
struct EtwRegistrationManager final {
  using EtwInternalCallback = EtwRegistrationManager_EtwInternalCallback;
  static EtwRegistrationManager& Instance() { return g_host->logging__EtwRegistrationManager__Instance(); }
  static bool SupportsETW() { return g_host->logging__EtwRegistrationManager__SupportsETW(); }
  Severity MapLevelToSeverity() { return g_host->logging__EtwRegistrationManager__MapLevelToSeverity(this); }
  void RegisterInternalCallback(const EtwInternalCallback& callback) {
    g_host->logging__EtwRegistrationManager__RegisterInternalCallback(this, callback);
  }
  void UnregisterInternalCallback(const EtwInternalCallback& callback) {
    g_host->logging__EtwRegistrationManager__UnregisterInternalCallback(this, callback);
  }
};
#endif  // defined(_WIN32)

}  // namespace logging
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {

struct int64s final {
  int size() const { return g_host->int64s__size(this); }
  const int64_t& Get(int index) const { return g_host->int64s__Get(this, index); }
  const int64_t* data() const { return g_host->int64s__data(this); }
  const int64_t& operator[](int index) const { return Get(index); }
  void Reserve(int size) { g_host->int64s__Reserve(this, size); }
  const int64_t* begin() const { return data(); }
  const int64_t* end() const { return data() + size(); }
  PROVIDER_DISALLOW_ALL(int64s)
};

struct float32s final {
  void Reserve(int size) { g_host->float32s__Reserve(this, size); }
  const float* data() const { return g_host->float32s__data(this); }
  int size() const { return g_host->float32s__size(this); }
  PROVIDER_DISALLOW_ALL(float32s)
};

struct StringStringEntryProto final {
  std::string* mutable_key() { return g_host->StringStringEntryProto__mutable_key(this); }
  std::string* mutable_value() { return g_host->StringStringEntryProto__mutable_value(this); }

  PROVIDER_DISALLOW_ALL(StringStringEntryProto)
};

struct StringStringEntryProtos final {
  void Clear() { g_host->StringStringEntryProtos__Clear(this); }
  StringStringEntryProto* Add() { return g_host->StringStringEntryProtos__Add(this); }
  int size() { return g_host->StringStringEntryProtos__size(this); }
  StringStringEntryProto& at(int index) { return g_host->StringStringEntryProtos__at(this, index); }

  PROVIDER_DISALLOW_ALL(StringStringEntryProtos)
};

struct OperatorSetIdProto final {
  std::string* mutable_domain() { return g_host->OperatorSetIdProto__mutable_domain(this); }
  void set_version(int64_t version) { return g_host->OperatorSetIdProto__set_version(this, version); }
  int64_t version() { return g_host->OperatorSetIdProto__version(this); }

  PROVIDER_DISALLOW_ALL(OperatorSetIdProto)
};

struct AttributeProto final {
  static std::unique_ptr<AttributeProto> Create() { return g_host->AttributeProto__construct(); }
  void operator=(const AttributeProto& v) { g_host->AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->AttributeProto__operator_delete(reinterpret_cast<AttributeProto*>(p)); }

  const std::string& name() const { return g_host->AttributeProto__name(this); }
  AttributeProto_AttributeType type() const { return g_host->AttributeProto__type(this); }
  int ints_size() const { return g_host->AttributeProto__ints_size(this); }
  int floats_size() const { return g_host->AttributeProto__floats_size(this); }
  int strings_size() const { return g_host->AttributeProto__strings_size(this); }
  int64_t ints(int i) const { return g_host->AttributeProto__ints(this, i); }
  float floats(int i) const { return g_host->AttributeProto__floats(this, i); }
  const std::string& strings(int i) const { return g_host->AttributeProto__strings(this, i); }
  const int64s& ints() const { return g_host->AttributeProto__ints(this); }
  const float32s& floats() const { return g_host->AttributeProto__floats(this); }
  int64s* mutable_ints() { return g_host->AttributeProto__mutable_ints(this); }
  float32s* mutable_floats() { return g_host->AttributeProto__mutable_floats(this); }
  void add_ints(int64_t value) { g_host->AttributeProto__add_ints(this, value); }
  void add_floats(float value) { g_host->AttributeProto__add_floats(this, value); }
  void add_strings(const ::std::string& value) { g_host->AttributeProto__add_strings(this, value); }

  int64_t i() const { return g_host->AttributeProto__i(this); }
  float f() const { return g_host->AttributeProto__f(this); }
  const ONNX_NAMESPACE::TensorProto& t() const { return g_host->AttributeProto__t(this); }
  void set_s(const ::std::string& value) { return g_host->AttributeProto__set_s(this, value); }
  void set_s(::std::string&& value) { return g_host->AttributeProto__set_s(this, ::std::move(value)); }
  void set_f(const float& value) { return g_host->AttributeProto__set_f(this, value); }
  void set_i(int64_t value) { return g_host->AttributeProto__set_i(this, value); }
  void set_t(const TensorProto& value) { return g_host->AttributeProto__set_t(this, value); }
  const ::std::string& s() const { return g_host->AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return g_host->AttributeProto__set_name(this, value); }
  void set_type(AttributeProto_AttributeType value) { return g_host->AttributeProto__set_type(this, value); }
  TensorProto* add_tensors() { return g_host->AttributeProto__add_tensors(this); }
  std::string* release_s() { return g_host->AttributeProto__release_s(this); }

  typedef AttributeProto_AttributeType AttributeType;
  static constexpr AttributeType UNDEFINED = AttributeProto_AttributeType_UNDEFINED;
  static constexpr AttributeType FLOAT = AttributeProto_AttributeType_FLOAT;
  static constexpr AttributeType INT = AttributeProto_AttributeType_INT;
  static constexpr AttributeType STRING = AttributeProto_AttributeType_STRING;
  static constexpr AttributeType TENSOR = AttributeProto_AttributeType_TENSOR;
  static constexpr AttributeType GRAPH = AttributeProto_AttributeType_GRAPH;
#if !defined(DISABLE_SPARSE_TENSORS)
  static constexpr AttributeType SPARSE_TENSOR = AttributeProto_AttributeType_SPARSE_TENSOR;
#endif
  static constexpr AttributeType FLOATS = AttributeProto_AttributeType_FLOATS;
  static constexpr AttributeType INTS = AttributeProto_AttributeType_INTS;
  static constexpr AttributeType STRINGS = AttributeProto_AttributeType_STRINGS;
  static constexpr AttributeType TENSORS = AttributeProto_AttributeType_TENSORS;
  static constexpr AttributeType GRAPHS = AttributeProto_AttributeType_GRAPHS;
#if !defined(DISABLE_SPARSE_TENSORS)
  static constexpr AttributeType SPARSE_TENSORS = AttributeProto_AttributeType_SPARSE_TENSORS;
#endif

  AttributeProto() = delete;
  AttributeProto(const AttributeProto&) = delete;
};

struct GraphProto final {
  static std::unique_ptr<GraphProto> Create() { return g_host->GraphProto__construct(); }
  static void operator delete(void* p) { g_host->GraphProto__operator_delete(reinterpret_cast<GraphProto*>(p)); }
  GraphProto& operator=(const GraphProto& v) { return g_host->GraphProto__operator_assign(this, v); }
  GraphProto& operator=(GraphProto&& v) noexcept { return g_host->GraphProto__operator_move_assign(this, std::move(v)); }

  const ValueInfoProto& input(int index) const { return g_host->GraphProto__input(this, index); }
  ValueInfoProtos* mutable_input() { return g_host->GraphProto__mutable_input(this); }
  ValueInfoProto* mutable_input(int index) { return g_host->GraphProto__mutable_input(this, index); }
  int input_size() const { return g_host->GraphProto__input_size(this); }

  const ValueInfoProtos& output() const { return g_host->GraphProto__output(this); }
  const ValueInfoProto& output(int index) const { return g_host->GraphProto__output(this, index); }
  ValueInfoProtos* mutable_output() { return g_host->GraphProto__mutable_output(this); }

  ValueInfoProtos* mutable_value_info() { return g_host->GraphProto__mutable_value_info(this); }
  TensorProtos* mutable_initializer() { return g_host->GraphProto__mutable_initializer(this); }
  TensorProto* add_initializer() { return g_host->GraphProto__add_initializer(this); }
  NodeProto* add_node() { return g_host->GraphProto__add_node(this); }
  NodeProto* mutable_node(int index) { return g_host->GraphProto__mutable_node(this, index); }

  std::string* mutable_name() { return g_host->GraphProto__mutable_name(this); }

  void set_name(const std::string& name) { g_host->GraphProto__set_name(this, name); }
  void set_doc_string(const std::string& doc_str) { g_host->GraphProto__set_doc_string(this, doc_str); }

  GraphProto() = delete;
  GraphProto(const GraphProto&) = delete;
};

struct ModelProto final {
  static std::unique_ptr<ModelProto> Create() { return g_host->ModelProto__construct(); }
  static void operator delete(void* p) { g_host->ModelProto__operator_delete(reinterpret_cast<ModelProto*>(p)); }

  bool SerializeToString(std::string& string) const { return g_host->ModelProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return g_host->ModelProto__SerializeToOstream(this, output); }
  bool ParseFromString(const std::string& data) { return g_host->ModelProto__ParseFromString(this, data); }
  std::string SerializeAsString() const { return g_host->ModelProto__SerializeAsString(this); }
  StringStringEntryProtos* mutable_metadata_props() { return g_host->ModelProto__mutable_metadata_props(this); };
  const GraphProto& graph() const { return g_host->ModelProto__graph(this); }
  GraphProto* mutable_graph() { return g_host->ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return g_host->ModelProto__set_ir_version(this, value); }

  const OperatorSetIdProto& opset_import(int index) const { return g_host->ModelProto__opset_import(this, index); }
  OperatorSetIdProto* mutable_opset_import(int index) { return g_host->ModelProto__mutable_opset_import(this, index); }
  int opset_import_size() const { return g_host->ModelProto__opset_import_size(this); }
  OperatorSetIdProto* add_opset_import() { return g_host->ModelProto__add_opset_import(this); }

  ModelProto() = delete;
  ModelProto(const ModelProto&) = delete;
  void operator=(const ModelProto&) = delete;
};

struct NodeProto final {
  static std::unique_ptr<NodeProto> Create() { return g_host->NodeProto__construct(); }
  static void operator delete(void* p) { g_host->NodeProto__operator_delete(reinterpret_cast<NodeProto*>(p)); }
  void operator=(const NodeProto& v) { g_host->NodeProto__operator_assign(this, v); }
  int attribute_size() { return g_host->NodeProto__attribute_size(this); }
  const AttributeProto& attribute(int index) const { return g_host->NodeProto__attribute(this, index); }
  AttributeProto* mutable_attribute(int index) { return g_host->NodeProto__mutable_attribute(this, index); }
  AttributeProto* add_attribute() { return g_host->NodeProto__add_attribute(this); }

  NodeProto() = delete;
  NodeProto(const NodeProto&) = delete;
};

struct TensorProto final {
  static std::unique_ptr<TensorProto> Create() { return g_host->TensorProto__construct(); }
  static void operator delete(void* p) { g_host->TensorProto__operator_delete(reinterpret_cast<TensorProto*>(p)); }
  TensorProto& operator=(const TensorProto& v) {
    return g_host->TensorProto__operator_assign(this, v);
  }
  TensorProto& operator=(TensorProto&& v) noexcept { return g_host->TensorProto__operator_move_assign(this, std::move(v)); }

  bool has_name() const { return g_host->TensorProto__has_name(this); }
  void set_name(const ::std::string& name) { return g_host->TensorProto__set_name(this, name); }
  const ::std::string& name() const { return g_host->TensorProto__name(this); }

  int dims_size() const { return g_host->TensorProto__dims_size(this); }
  const int64s& dims() const { return g_host->TensorProto__dims(this); }
  void add_dims(int64_t value) { g_host->TensorProto__add_dims(this, value); }

  bool has_data_location() const { return g_host->TensorProto__has_data_location(this); }
  TensorProto_DataLocation data_location() const { return TensorProto_DataLocation(g_host->TensorProto__data_location(this)); }
  void set_data_location(TensorProto_DataLocation data_location) { return g_host->TensorProto__set_data_location(this, data_location); }

  bool has_raw_data() const { return g_host->TensorProto__has_raw_data(this); }
  const std::string& raw_data() const { return g_host->TensorProto__raw_data(this); }
  std::string* mutable_raw_data() { return g_host->TensorProto__mutable_raw_data(this); }

  bool has_data_type() const { return g_host->TensorProto__has_data_type(this); }
  int32_t data_type() const { return g_host->TensorProto__data_type(this); }
  void set_data_type(int32_t type) { return g_host->TensorProto__set_data_type(this, type); }

  typedef TensorProto_DataType DataType;
  static constexpr DataType UNDEFINED = TensorProto_DataType_UNDEFINED;

  static bool DataType_IsValid(int value) { return g_host->TensorProto_DataType_IsValid(value); }

  void copy_from(const TensorProto* other) { return g_host->TensorProto__CopyFrom(this, other); }
  StringStringEntryProtos* mutable_external_data() { return g_host->TensorProto__mutable_external_data(this); };
  void clear_float_data() { return g_host->TensorProto__clear_float_data(this); }
  void clear_int32_data() { return g_host->TensorProto__clear_int32_data(this); }
  void clear_string_data() { return g_host->TensorProto__clear_string_data(this); }
  void clear_int64_data() { return g_host->TensorProto__clear_int64_data(this); }
  void clear_double_data() { return g_host->TensorProto__clear_double_data(this); }
  void clear_uint64_data() { return g_host->TensorProto__clear_uint64_data(this); }

  TensorProto() = delete;
  TensorProto(const TensorProto&) = delete;
};

struct TensorProtos final {
  TensorProto* Add() { return g_host->TensorProtos__Add(this); }
  int size() const { return g_host->TensorProtos__size(this); }
  TensorProto& at(int index) { return g_host->TensorProtos__at(this, index); }
  IteratorHolder<TensorProto_ConstIterator, const TensorProto> begin() const { return g_host->TensorProtos__begin(this); }
  IteratorHolder<TensorProto_ConstIterator, const TensorProto> end() const { return g_host->TensorProtos__end(this); }
  IteratorHolder<TensorProto_Iterator, TensorProto> begin() { return g_host->TensorProtos__begin(this); }
  IteratorHolder<TensorProto_Iterator, TensorProto> end() { return g_host->TensorProtos__end(this); }

  PROVIDER_DISALLOW_ALL(TensorProtos)
};

struct TensorShapeProto_Dimension final {
  enum ValueCase {
    kDimValue = 1,
    kDimParam = 2,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(g_host->TensorShapeProto_Dimension__value_case(this)); }
  const std::string& dim_param() const { return g_host->TensorShapeProto_Dimension__dim_param(this); }
  int64_t dim_value() const { return g_host->TensorShapeProto_Dimension__dim_value(this); }
  void set_dim_value(int64_t value) { return g_host->TensorShapeProto_Dimension__set_dim_value(this, value); }
  bool has_dim_value() const { return g_host->TensorShapeProto_Dimension__has_dim_value(this); }
  bool has_dim_param() const { return g_host->TensorShapeProto_Dimension__has_dim_param(this); }
  void clear_dim_value() { return g_host->TensorShapeProto_Dimension__clear_dim_value(this); }
  const std::string& denotation() const { return g_host->TensorShapeProto_Dimension__denotation(this); }
  void set_denotation(const std::string& value) { g_host->TensorShapeProto_Dimension__set_denotation(this, value); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimension)
};

struct TensorShapeProto_Dimensions final {
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> begin() const { return g_host->TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> end() const { return g_host->TensorShapeProto_Dimensions__end(this); }
  size_t size() const { return g_host->TensorShapeProto_Dimensions__size(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimensions)
};

struct TensorShapeProto final {
  int dim_size() const { return g_host->TensorShapeProto__dim_size(this); }
  const TensorShapeProto_Dimensions& dim() const { return g_host->TensorShapeProto__dim(this); }
  const TensorShapeProto_Dimension& dim(int index) const { return g_host->TensorShapeProto__dim(this, index); }
  TensorShapeProto_Dimension* mutable_dim(int index) { return g_host->TensorShapeProto__mutable_dim(this, index); }
  void clear_dim() { return g_host->TensorShapeProto__clear_dim(this); }
  TensorShapeProto_Dimension* add_dim() { return g_host->TensorShapeProto__add_dim(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto)
};

struct TypeProto_Tensor final {
  bool has_shape() const { return g_host->TypeProto_Tensor__has_shape(this); }
  const TensorShapeProto& shape() const { return g_host->TypeProto_Tensor__shape(this); }
  TensorShapeProto* mutable_shape() { return g_host->TypeProto_Tensor__mutable_shape(this); }
  bool has_elem_type() const { return g_host->TypeProto_Tensor__has_elem_type(this); }
  int32_t elem_type() const { return g_host->TypeProto_Tensor__elem_type(this); }
  void set_elem_type(int32_t value) { g_host->TypeProto_Tensor__set_elem_type(this, value); }

  PROVIDER_DISALLOW_ALL(TypeProto_Tensor)
};

#if !defined(DISABLE_SPARSE_TENSORS)
struct TypeProto_SparseTensor final {
  bool has_shape() const { return g_host->TypeProto_SparseTensor__has_shape(this); }
  const TensorShapeProto& shape() const { return g_host->TypeProto_SparseTensor__shape(this); }
  TensorShapeProto* mutable_shape() { return g_host->TypeProto_SparseTensor__mutable_shape(this); }
  int32_t elem_type() const { return g_host->TypeProto_SparseTensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_SparseTensor)
};
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
struct TypeProto_Optional final {
  const TypeProto& elem_type() const { return g_host->TypeProto_Optional__elem_type(this); }
  TypeProto* mutable_elem_type() { return g_host->TypeProto_Optional__mutable_elem_type(this); }
  PROVIDER_DISALLOW_ALL(TypeProto_Optional)
};
#endif

struct TypeProto_Sequence final {
  const TypeProto& elem_type() const { return g_host->TypeProto_Sequence__elem_type(this); }
  TypeProto* mutable_elem_type() { return g_host->TypeProto_Sequence__mutable_elem_type(this); }
  PROVIDER_DISALLOW_ALL(TypeProto_Sequence)
};

struct TypeProto final {
  static std::unique_ptr<TypeProto> Create() { return g_host->TypeProto__construct(); }
  static void operator delete(void* p) { g_host->TypeProto__operator_delete(reinterpret_cast<TypeProto*>(p)); }

  bool has_tensor_type() const { return g_host->TypeProto__has_tensor_type(this); }
  const TypeProto_Tensor& tensor_type() const { return g_host->TypeProto__tensor_type(this); }
  TypeProto_Tensor* mutable_tensor_type() { return g_host->TypeProto__mutable_tensor_type(this); }

#if !defined(DISABLE_SPARSE_TENSORS)
  const TypeProto_SparseTensor& sparse_tensor_type() const { return g_host->TypeProto__sparse_tensor_type(this); }
  TypeProto_SparseTensor* mutable_sparse_tensor_type() { return g_host->TypeProto__mutable_sparse_tensor_type(this); }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  const TypeProto_Optional& optional_type() const { return g_host->TypeProto__optional_type(this); }
  TypeProto_Optional* mutable_optional_type() { return g_host->TypeProto__mutable_optional_type(this); }
#endif

  const TypeProto_Sequence& sequence_type() const { return g_host->TypeProto__sequence_type(this); }
  TypeProto_Sequence* mutable_sequence_type() { return g_host->TypeProto__mutable_sequence_type(this); }

  enum ValueCase {
    kTensorType = 1,
    kSequenceType = 4,
    kMapType = 5,
    kOptionalType = 9,
    kSparseTensorType = 8,
    kOpaqueType = 7,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(g_host->TypeProto__value_case(this)); }

  void copy_from(const TypeProto* other) { return g_host->TypeProto__CopyFrom(this, other); }

  TypeProto() = delete;
  TypeProto(const TypeProto&) = delete;
};

struct ValueInfoProto final {
  const TypeProto& type() const { return g_host->ValueInfoProto__type(this); }
  TypeProto* mutable_type() { return g_host->ValueInfoProto__mutable_type(this); }

  void operator=(const ValueInfoProto& v) { g_host->ValueInfoProto__operator_assign(this, v); }

  ValueInfoProto() = delete;
  ValueInfoProto(const ValueInfoProto&) = delete;
  static void operator delete(void*) = delete;
};

struct ValueInfoProtos final {
  ValueInfoProto* Add() { return g_host->ValueInfoProtos__Add(this); }
  const ValueInfoProto& operator[](int index) const { return g_host->ValueInfoProtos__operator_array(this, index); }

  PROVIDER_DISALLOW_ALL(ValueInfoProtos)
};

struct FunctionProto final {
  static std::unique_ptr<FunctionProto> Create() { return g_host->FunctionProto__construct(); }
  static void operator delete(void* p) { g_host->FunctionProto__operator_delete(reinterpret_cast<FunctionProto*>(p)); }

  bool SerializeToString(std::string& string) const { return g_host->FunctionProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return g_host->FunctionProto__SerializeToOstream(this, output); }
  bool ParseFromString(const std::string& data) { return g_host->FunctionProto__ParseFromString(this, data); }
  std::string SerializeAsString() const { return g_host->FunctionProto__SerializeAsString(this); }

  bool has_name() const { return g_host->FunctionProto__has_name(this); }
  const std::string& name() const { return g_host->FunctionProto__name(this); }
  void set_name(const std::string& name) { g_host->FunctionProto__set_name(this, name); }

  bool has_doc_string() const { return g_host->FunctionProto__has_doc_string(this); }
  const std::string& doc_string() const { return g_host->FunctionProto__doc_string(this); }
  void set_doc_string(const std::string& doc_string) { g_host->FunctionProto__set_doc_string(this, doc_string); }

  bool has_domain() const { return g_host->FunctionProto__has_domain(this); }
  const std::string& domain() const { return g_host->FunctionProto__domain(this); }
  void set_domain(const std::string& domain) { g_host->FunctionProto__set_domain(this, domain); }

  const std::string& input(int index) const { return g_host->FunctionProto__input(this, index); }
  std::string* mutable_input(int index) { return g_host->FunctionProto__mutable_input(this, index); }
  int input_size() const { return g_host->FunctionProto__input_size(this); }
  void add_input(const std::string& value) { g_host->FunctionProto__add_input(this, value); }

  const std::string& output(int index) const { return g_host->FunctionProto__output(this, index); }
  std::string* mutable_output(int index) { return g_host->FunctionProto__mutable_output(this, index); }
  int output_size() const { return g_host->FunctionProto__output_size(this); }
  void add_output(const std::string& value) { g_host->FunctionProto__add_output(this, value); }

  const std::string& attribute(int index) const { return g_host->FunctionProto__attribute(this, index); }
  std::string* mutable_attribute(int index) { return g_host->FunctionProto__mutable_attribute(this, index); }
  int attribute_size() const { return g_host->FunctionProto__attribute_size(this); }
  void add_attribute(const std::string& value) { g_host->FunctionProto__add_attribute(this, value); }

  const AttributeProto& attribute_proto(int index) const { return g_host->FunctionProto__attribute_proto(this, index); }
  AttributeProto* mutable_attribute_proto(int index) { return g_host->FunctionProto__mutable_attribute_proto(this, index); }
  int attribute_proto_size() const { return g_host->FunctionProto__attribute_proto_size(this); }
  AttributeProto* add_attribute_proto() { return g_host->FunctionProto__add_attribute_proto(this); }

  const NodeProto& node(int index) const { return g_host->FunctionProto__node(this, index); }
  NodeProto* mutable_node(int index) { return g_host->FunctionProto__mutable_node(this, index); }
  int node_size() const { return g_host->FunctionProto__node_size(this); }
  NodeProto* add_node() { return g_host->FunctionProto__add_node(this); }

  const ValueInfoProto& value_info(int index) const { return g_host->FunctionProto__value_info(this, index); }
  ValueInfoProtos* mutable_value_info() { return g_host->FunctionProto__mutable_value_info(this); }
  ValueInfoProto* mutable_value_info(int index) { return g_host->FunctionProto__mutable_value_info(this, index); }
  int value_info_size() const { return g_host->FunctionProto__value_info_size(this); }
  ValueInfoProto* add_value_info() { return g_host->FunctionProto__add_value_info(this); }

  const StringStringEntryProto& metadata_props(int index) const { return g_host->FunctionProto__metadata_props(this, index); }
  StringStringEntryProtos* mutable_metadata_props() { return g_host->FunctionProto__mutable_metadata_props(this); }
  StringStringEntryProto* mutable_metadata_props(int index) { return g_host->FunctionProto__mutable_metadata_props(this, index); }
  int metadata_props_size() const { return g_host->FunctionProto__metadata_props_size(this); }
  StringStringEntryProto* add_metadata_props() { return g_host->FunctionProto__add_metadata_props(this); }

  FunctionProto() = delete;
  FunctionProto(const FunctionProto&) = delete;
  void operator=(const FunctionProto&) = delete;
};

struct OpSchema final {
  const TypeConstraintMap& typeConstraintMap() const { return g_host->OpSchema__typeConstraintMap(this); }
  const std::string& inputs__GetName(const size_t i) const { return g_host->OpSchema__inputs__GetName(this, i); };
  const std::string& inputs__GetTypeStr(const size_t i) const { return g_host->OpSchema__inputs__GetTypeStr(this, i); };
  const std::string& outputs__GetName(const size_t i) const { return g_host->OpSchema__outputs__GetName(this, i); };
  const std::string& outputs__GetTypeStr(const size_t i) const { return g_host->OpSchema__outputs__GetTypeStr(this, i); };
  PROVIDER_DISALLOW_ALL(OpSchema)
};
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

namespace utils {
bool IsDataTypeString(MLDataType dt_type);

}  // namespace utils

namespace Utils {

struct DataTypeUtils final {
  static const std::string* ToType(const ONNX_NAMESPACE::TypeProto& type_proto) { return g_host->Utils__DataTypeUtils__ToType(type_proto); }
  static const std::string* ToType(const std::string& type_str) { return g_host->Utils__DataTypeUtils__ToType(type_str); }

  PROVIDER_DISALLOW_ALL(DataTypeUtils)
};

}  // namespace Utils

struct ConfigOptions final {
  std::optional<std::string> GetConfigEntry(const std::string& config_key) const {
    return g_host->ConfigOptions__GetConfigEntry(this, config_key);
  }

  std::string GetConfigOrDefault(const std::string& config_key, const std::string& default_value) const {
    return g_host->ConfigOptions__GetConfigOrDefault(this, config_key, default_value);
  }

  const std::unordered_map<std::string, std::string>& GetConfigOptionsMap() const {
    return g_host->ConfigOptions__GetConfigOptionsMap(this);
  }

  PROVIDER_DISALLOW_ALL(ConfigOptions)
};

struct ComputeCapability final {
  static std::unique_ptr<ComputeCapability> Create(std::unique_ptr<IndexedSubGraph> t_sub_graph) { return g_host->ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { g_host->ComputeCapability__operator_delete(reinterpret_cast<ComputeCapability*>(p)); }

  std::unique_ptr<IndexedSubGraph>& SubGraph() { return g_host->ComputeCapability__SubGraph(this); }

  void copy_optimization_func(ComputeCapability* selection_cc) { g_host->ComputeCapability__copy_optimization_func(this, selection_cc); }
  void add_nodes_to_optimize(std::unique_ptr<ComputeCapability> optimization_cc) { g_host->ComputeCapability__add_nodes_to_optimize(this, std::move(optimization_cc)); }

  ComputeCapability() = delete;
  ComputeCapability(const ComputeCapability&) = delete;
  void operator=(const ComputeCapability&) = delete;
};

struct DataTransferManager final {
  Status CopyTensor(const Tensor& src, Tensor& dst) const { return g_host->DataTransferManager__CopyTensor(this, src, dst); }
#if !defined(DISABLE_SPARSE_TENSORS)
  Status CopySparseTensor(const SparseTensor& src, SparseTensor& dst) const { return g_host->DataTransferManager__CopySparseTensor(this, src, dst); }
  Status CopySparseTensors(const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) const { return g_host->DataTransferManager__CopySparseTensors(this, src_dst_pairs); }
#endif
  const IDataTransfer* GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const { return g_host->DataTransferManager__GetDataTransfer(this, src_device, dst_device); }

  PROVIDER_DISALLOW_ALL(DataTransferManager)
};

struct IndexedSubGraph_MetaDef final {
  static std::unique_ptr<IndexedSubGraph_MetaDef> Create() { return g_host->IndexedSubGraph_MetaDef__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph_MetaDef__operator_delete(reinterpret_cast<IndexedSubGraph_MetaDef*>(p)); }

  const std::string& name() const { return g_host->IndexedSubGraph_MetaDef__name(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& name() { return g_host->IndexedSubGraph_MetaDef__name(this); }
  const std::string& domain() const { return g_host->IndexedSubGraph_MetaDef__domain(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& domain() { return g_host->IndexedSubGraph_MetaDef__domain(this); }
  int since_version() const { return g_host->IndexedSubGraph_MetaDef__since_version(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  int& since_version() { return g_host->IndexedSubGraph_MetaDef__since_version(this); }

  ONNX_NAMESPACE::OperatorStatus& status() { return g_host->IndexedSubGraph_MetaDef__status(this); }

  const std::vector<std::string>& inputs() const { return g_host->IndexedSubGraph_MetaDef__inputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& inputs() { return g_host->IndexedSubGraph_MetaDef__inputs(this); }
  const std::vector<std::string>& outputs() const { return g_host->IndexedSubGraph_MetaDef__outputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  const std::vector<std::string>& constant_initializers() const { return g_host->IndexedSubGraph_MetaDef__constant_initializers(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& constant_initializers() { return g_host->IndexedSubGraph_MetaDef__constant_initializers(this); }
  std::vector<std::string>& outputs() { return g_host->IndexedSubGraph_MetaDef__outputs(this); }
  NodeAttributes& attributes() { return g_host->IndexedSubGraph_MetaDef__attributes(this); }

  std::string& doc_string() { return g_host->IndexedSubGraph_MetaDef__doc_string(this); }

  IndexedSubGraph_MetaDef() = delete;
  IndexedSubGraph_MetaDef(const IndexedSubGraph_MetaDef&) = delete;
  void operator=(const IndexedSubGraph_MetaDef&) = delete;
};

enum class IndexedSubGraph_SourceOfSchema : uint8_t {
  CREATE,
  REUSE_OR_CREATE,
  EXISTING,
};

struct IndexedSubGraph final {
  static std::unique_ptr<IndexedSubGraph> Create() { return g_host->IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph__operator_delete(reinterpret_cast<IndexedSubGraph*>(p)); }

  gsl::span<const onnxruntime::NodeIndex> Nodes() const { return g_host->IndexedSubGraph__Nodes(this); }
  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) { return g_host->IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const IndexedSubGraph_MetaDef*>(g_host->IndexedSubGraph__GetMetaDef(this)); }

  void SetSchemaSource(IndexedSubGraph_SourceOfSchema schema_source) { return g_host->IndexedSubGraph__SetSchemaSource(this, schema_source); }
  IndexedSubGraph_SourceOfSchema GetSchemaSource() const { return g_host->IndexedSubGraph__GetSchemaSource(this); }
  void SetAccountant(IResourceAccountant* resource_accountant) {
    g_host->IndexedSubGraph__SetAccountant(this, resource_accountant);
  }
  void AppendNodeCost(const ResourceCount& resource_count) {
    g_host->IndexedSubGraph__AppendNodeCost(this, resource_count);
  }

  IndexedSubGraph() = delete;
  IndexedSubGraph(const IndexedSubGraph&) = delete;
  void operator=(const IndexedSubGraph&) = delete;
};

struct KernelDef final {
  static void operator delete(void* p) { g_host->KernelDef__operator_delete(reinterpret_cast<KernelDef*>(p)); }

  int ExecQueueId() const { return g_host->KernelDef__ExecQueueId(this); }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const { g_host->KernelDef__SinceVersion(this, start, end); }
  const std::string& Domain() const { return g_host->KernelDef__Domain(this); }
  const std::string& OpName() const { return g_host->KernelDef__OpName(this); }

  KernelDef() = delete;
  KernelDef(const KernelDef*) = delete;
  void operator=(const KernelDef&) = delete;
};

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

struct KernelDefBuilder final {
  static std::unique_ptr<KernelDefBuilder> Create() { return g_host->KernelDefBuilder__construct(); }
  static void operator delete(void* p) { g_host->KernelDefBuilder__operator_delete(reinterpret_cast<KernelDefBuilder*>(p)); }

  KernelDefBuilder& SetName(const char* op_name) {
    g_host->KernelDefBuilder__SetName(this, op_name);
    return *this;
  }
  KernelDefBuilder& SetDomain(const char* domain) {
    g_host->KernelDefBuilder__SetDomain(this, domain);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version) {
    g_host->KernelDefBuilder__SinceVersion(this, since_version);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    g_host->KernelDefBuilder__SinceVersion(this, since_version_start, since_version_end);
    return *this;
  }
  KernelDefBuilder& Provider(const char* provider_type) {
    g_host->KernelDefBuilder__Provider(this, provider_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) {
    g_host->KernelDefBuilder__TypeConstraint(this, arg_name, supported_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) {
    g_host->KernelDefBuilder__TypeConstraint(this, arg_name, supported_types);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    g_host->KernelDefBuilder__InputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, const std::vector<int>& input_indexes) {
    g_host->KernelDefBuilder__InputMemoryType(this, type, input_indexes);
    return *this;
  }
  KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) {
    g_host->KernelDefBuilder__OutputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& ExecQueueId(int queue_id) {
    g_host->KernelDefBuilder__ExecQueueId(this, queue_id);
    return *this;
  }
  KernelDefBuilder& MayInplace(int input_index, int output_index) {
    g_host->KernelDefBuilder__MayInplace(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    g_host->KernelDefBuilder__Alias(this, aliases);
    return *this;
  }
  KernelDefBuilder& Alias(int input_index, int output_index) {
    g_host->KernelDefBuilder__Alias(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& VariadicAlias(int input_offset, int output_offset) {
    g_host->KernelDefBuilder__VariadicAlias(this, input_offset, output_offset);
    return *this;
  }

  KernelDefBuilder& ExternalOutputs() {
    g_host->KernelDefBuilder__ExternalOutputs(this);
    return *this;
  }

  KernelDefBuilder& AllocateInputsContiguously() {
    g_host->KernelDefBuilder__AllocateInputsContiguously(this);
    return *this;
  }

#ifdef ENABLE_STRIDED_TENSORS
  KernelDefBuilder& MayStridedInput(int input_index) {
    g_host->KernelDefBuilder__MayStridedInput(this, input_index);
    return *this;
  }

  KernelDefBuilder& MayStridedOutput(int input_index, int output_index) {
    g_host->KernelDefBuilder__MayStridedOutput(this, input_index, output_index);
    return *this;
  }
#endif

  std::unique_ptr<KernelDef> Build() {
    return g_host->KernelDefBuilder__Build(this);
  }

  KernelDefBuilder() = delete;
  KernelDefBuilder(const KernelDefBuilder&) = delete;
  void operator=(const KernelDefBuilder&) = delete;
};

struct KernelRegistry final {
  static std::shared_ptr<KernelRegistry> Create() { return g_host->KernelRegistry__construct(); }
  static void operator delete(void* p) { g_host->KernelRegistry__operator_delete(reinterpret_cast<KernelRegistry*>(p)); }

  Status Register(KernelCreateInfo&& create_info) { return g_host->KernelRegistry__Register(this, std::move(create_info)); }

  KernelRegistry() = delete;
  KernelRegistry(const KernelRegistry&) = delete;
  void operator=(const KernelRegistry&) = delete;
};

struct PrimitiveDataTypeBase final {
  int32_t GetDataType() const { return g_host->PrimitiveDataTypeBase__GetDataType(this); }

  int32_t GetNumSubElems() const {
    return g_host->PrimitiveDataTypeBase__GetNumSubElems(this);
  }

  bool HasSubElems() const {
    return g_host->PrimitiveDataTypeBase__HasSubElems(this);
  }

  PROVIDER_DISALLOW_ALL(PrimitiveDataTypeBase)
};

class DataTypeImpl final {
 public:
  size_t Size() const { return g_host->DataTypeImpl__Size(this); }

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
#if !defined(DISABLE_SPARSE_TENSORS)
  template <typename elemT>
  static MLDataType GetSparseTensorType();
#endif

  static MLDataType GetTypeFromOnnxType(int);
  static MLDataType GetTensorTypeFromOnnxType(int);

  bool IsTensorType() const { return g_host->DataTypeImpl__IsTensorType(this); }
  bool IsTensorSequenceType() const { return g_host->DataTypeImpl__IsTensorSequenceType(this); }
#if !defined(DISABLE_SPARSE_TENSORS)
  bool IsSparseTensorType() const { return g_host->DataTypeImpl__IsSparseTensorType(this); }
#endif

  DeleteFunc GetDeleteFunc() const { return g_host->DataTypeImpl__GetDeleteFunc(this); }

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorTypesIRv4() { return g_host->DataTypeImpl__AllFixedSizeTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorTypesIRv9() { return g_host->DataTypeImpl__AllFixedSizeTensorTypesIRv9(); }

  static const std::vector<MLDataType>& AllTensorTypes() { return g_host->DataTypeImpl__AllTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorTypesIRv4() { return g_host->DataTypeImpl__AllTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllTensorTypesIRv9() { return g_host->DataTypeImpl__AllTensorTypesIRv9(); }
  static const std::vector<MLDataType>& AllTensorTypesIRv10() { return g_host->DataTypeImpl__AllTensorTypesIRv10(); }
  static const std::vector<MLDataType>& AllTensorTypesIRv11() { return g_host->DataTypeImpl__AllTensorTypesIRv11(); }

  static const std::vector<MLDataType>& AllIEEEFloatTensorTypes() { return g_host->DataTypeImpl__AllIEEEFloatTensorTypes(); }

  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypesIRv4() { return g_host->DataTypeImpl__AllTensorAndSequenceTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypesIRv9() { return g_host->DataTypeImpl__AllTensorAndSequenceTensorTypesIRv9(); }

  static const std::vector<MLDataType>& AllOptionalAndTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllOptionalAndTensorAndSequenceTensorTypesIRv4() { return g_host->DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllOptionalAndTensorAndSequenceTensorTypesIRv9() { return g_host->DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv9(); }

  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypesIRv4() { return g_host->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypesIRv9() { return g_host->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv9(); }

  static const std::vector<MLDataType>& AllSequenceTensorTypes() { return g_host->DataTypeImpl__AllSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllSequenceTensorTypesIRv4() { return g_host->DataTypeImpl__AllSequenceTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllSequenceTensorTypesIRv9() { return g_host->DataTypeImpl__AllSequenceTensorTypesIRv9(); }

  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypesIRv4() { return g_host->DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv4(); }
  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypesIRv9() { return g_host->DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv9(); }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const { return g_host->DataTypeImpl__AsPrimitiveDataType(this); }

  static const char* ToString(MLDataType type) { return g_host->DataTypeImpl__ToString(type); }

  PROVIDER_DISALLOW_ALL(DataTypeImpl)
};

struct Function final {
  const Graph& Body() const { return g_host->Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Function)
};

struct Node_EdgeEnd final {
  const Node& GetNode() const { return g_host->Node_EdgeEnd__GetNode(this); }
  int GetSrcArgIndex() const { return g_host->Node_EdgeEnd__GetSrcArgIndex(this); }
  int GetDstArgIndex() const { return g_host->Node_EdgeEnd__GetDstArgIndex(this); }

  PROVIDER_DISALLOW_ALL(Node_EdgeEnd)
};

struct Node final {
  enum class Type {
    Primitive = 0,
    Fused = 1,
  };
  const std::string& Name() const noexcept { return g_host->Node__Name(this); }
  const std::string& Description() const noexcept { return g_host->Node__Description(this); }
  const std::string& Domain() const noexcept { return g_host->Node__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->Node__OpType(this); }

  int SinceVersion() const noexcept { return g_host->Node__SinceVersion(this); }

  const Function* GetFunctionBody() const noexcept { return g_host->Node__GetFunctionBody(this); }
  ProviderType GetExecutionProviderType() const noexcept { return g_host->Node__GetExecutionProviderType(this); }

  ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const noexcept { return g_host->Node__ImplicitInputDefs(this); }

  const std::vector<int>& InputArgCount() const noexcept { return g_host->Node__InputArgCount(this); }

  ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept { return g_host->Node__InputDefs(this); }
  ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept { return g_host->Node__OutputDefs(this); }
  NodeIndex Index() const noexcept { return g_host->Node__Index(this); }

  std::vector<gsl::not_null<const Graph*>> GetSubgraphs() const noexcept { return g_host->Node__GetSubgraphs(this); }

  void ToProto(ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) const { return g_host->Node__ToProto(this, proto, update_subgraphs); }

  const NodeAttributes& GetAttributes() const noexcept { return g_host->Node__GetAttributes(this); }
  void AddAttribute(const ::std::string& attr_name, const ONNX_NAMESPACE::GraphProto& value) {
    g_host->Node__AddAttribute(this, attr_name, value);
  }
  void AddAttribute(const std::string& attr_name, const std::string& value) {
    g_host->Node__AddAttribute(this, attr_name, value);
  }
  void AddAttribute(const std::string& attr_name, int64_t value) {
    g_host->Node__AddAttribute(this, attr_name, value);
  }

  size_t GetInputEdgesCount() const noexcept { return g_host->Node__GetInputEdgesCount(this); }
  size_t GetOutputEdgesCount() const noexcept { return g_host->Node__GetOutputEdgesCount(this); }

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Node__NodeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const NodeConstIterator& p_other) const { return *impl_ != *p_other.impl_; }

    void operator++() { impl_->operator++(); }

    const Node& operator*() const { return impl_->operator*(); }
    const Node* operator->() const { return &impl_->operator*(); }

    std::unique_ptr<Node__NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return g_host->Node__InputNodesBegin(this); }
  NodeConstIterator InputNodesEnd() const noexcept { return g_host->Node__InputNodesEnd(this); }

  NodeConstIterator OutputNodesBegin() const noexcept { return g_host->Node__OutputNodesBegin(this); }
  NodeConstIterator OutputNodesEnd() const noexcept { return g_host->Node__OutputNodesEnd(this); }

  struct EdgeConstIterator {
    EdgeConstIterator(std::unique_ptr<Node__EdgeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const EdgeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() { impl_->operator++(); }
    const Node_EdgeEnd& operator*() { return impl_->operator*(); }
    const Node__EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Node__EdgeIterator> impl_;
  };

  EdgeConstIterator InputEdgesBegin() const noexcept { return g_host->Node__InputEdgesBegin(this); }
  EdgeConstIterator InputEdgesEnd() const noexcept { return g_host->Node__InputEdgesEnd(this); }
  EdgeConstIterator OutputEdgesBegin() const noexcept { return g_host->Node__OutputEdgesBegin(this); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return g_host->Node__OutputEdgesEnd(this); }

  void ForEachDef(std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs = false) const { g_host->Node__ForEachDef(this, func, std::move(include_missing_optional_defs)); }
  const std::unordered_map<std::string, gsl::not_null<Graph*>>& GetAttributeNameToMutableSubgraphMap() { return g_host->Node__GetAttributeNameToMutableSubgraphMap(this); }
  std::unordered_map<std::string, gsl::not_null<const Graph*>> GetAttributeNameToSubgraphMap() const { return g_host->Node__GetAttributeNameToSubgraphMap(this); }
  Type NodeType() const noexcept { return Type(g_host->Node__NodeType(this)); }

  PROVIDER_DISALLOW_ALL(Node)
};

struct NodeArg final {
  const std::string& Name() const noexcept { return g_host->NodeArg__Name(this); }
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const { return g_host->NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->NodeArg__Type(this); }
  const NodeArgInfo& ToProto() const noexcept { return g_host->NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->NodeArg__Exists(this); }
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept { return g_host->NodeArg__TypeAsProto(this); }
  Status OverrideTypesHelper(const ONNX_NAMESPACE::TypeProto& input_type, int32_t input_tensor_elem_type, int32_t current_tensor_elem_type, bool override_types) { return g_host->NodeArg__OverrideTypesHelper(this, input_type, input_tensor_elem_type, current_tensor_elem_type, override_types); }

  PROVIDER_DISALLOW_ALL(NodeArg)
};

struct NodeAttributes final {
  static std::unique_ptr<NodeAttributes> Create() { return g_host->NodeAttributes__construct(); }
  void operator=(const NodeAttributes& v) { return g_host->NodeAttributes__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->NodeAttributes__operator_delete(reinterpret_cast<NodeAttributes*>(p)); }

  size_t size() const { return g_host->NodeAttributes__size(this); }
  void clear() noexcept { g_host->NodeAttributes__clear(this); }
  size_t count(const std::string& keyval) const { return g_host->NodeAttributes__count(this, keyval); }
  ONNX_NAMESPACE::AttributeProto& operator[](const std::string& string) { return g_host->NodeAttributes__operator_array(this, string); }
  const ONNX_NAMESPACE::AttributeProto& at(const std::string& string) const { return g_host->NodeAttributes__at(this, string); }

  IteratorHolder<NodeAttributes_Iterator, const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> begin() const { return g_host->NodeAttributes__begin(this); }
  IteratorHolder<NodeAttributes_Iterator, const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> end() const { return g_host->NodeAttributes__end(this); }
  IteratorHolder<NodeAttributes_Iterator, const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> find(const std::string& key) const { return g_host->NodeAttributes__find(this, key); }
  void insert(const NodeAttributes& v) { return g_host->NodeAttributes__insert(this, v); }
  void emplace(const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) { g_host->NodeAttributes__emplace(this, k, v); }
  void emplace(const std::string& k, ONNX_NAMESPACE::AttributeProto&& v) { g_host->NodeAttributes__emplace(this, k, std::move(v)); }
  void insert_or_assign(const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) { g_host->NodeAttributes__insert_or_assign(this, k, v); }

  void reserve(size_t size) { g_host->NodeAttributes__reserve(this, size); }

  NodeAttributes() = delete;
  NodeAttributes(const NodeAttributes&) = delete;
};

struct NodeUnitIODef final {
  // The quantization parameter. Scale is mandatory. Zero-point and axis are optional.
  struct QuantParam {
    const NodeArg& scale;
    const NodeArg* zero_point{nullptr};
    std::optional<int64_t> axis{std::nullopt};
  };

  const NodeArg& node_arg;
  const std::optional<QuantParam> quant_param;
};

struct NodeUnit final {
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

  NodeUnit() = delete;
  NodeUnit(const NodeUnit&) = delete;
  void operator=(const NodeUnit& v) = delete;

  // Need delete because of APIs that return unique_ptr<NodeUnit>
  static void operator delete(void* p) { g_host->NodeUnit__operator_delete(reinterpret_cast<NodeUnit*>(p)); }

  Type UnitType() const noexcept { return static_cast<Type>(g_host->NodeUnit__UnitType(this)); }

  const std::vector<NodeUnitIODef>& Inputs() const noexcept { return g_host->NodeUnit__Inputs(this); }
  const std::vector<NodeUnitIODef>& Outputs() const noexcept { return g_host->NodeUnit__Outputs(this); }

  const std::string& Domain() const noexcept { return g_host->NodeUnit__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->NodeUnit__OpType(this); }
  const std::string& Name() const noexcept { return g_host->NodeUnit__Name(this); }
  int SinceVersion() const noexcept { return g_host->NodeUnit__SinceVersion(this); }
  NodeIndex Index() const noexcept { return g_host->NodeUnit__Index(this); }
  const std::filesystem::path& ModelPath() const noexcept { return g_host->NodeUnit__ModelPath(this); }
  ProviderType GetExecutionProviderType() const noexcept { return g_host->NodeUnit__GetExecutionProviderType(this); }

  const Node& GetNode() const noexcept { return g_host->NodeUnit__GetNode(this); }
  const std::vector<const Node*>& GetDQNodes() const noexcept { return g_host->NodeUnit__GetDQNodes(this); }
  const std::vector<const Node*>& GetQNodes() const noexcept { return g_host->NodeUnit__GetQNodes(this); }
  std::vector<const Node*> GetAllNodesInGroup() const noexcept { return g_host->NodeUnit__GetAllNodesInGroup(this); }

  /// Number of input edges to the logical node. For a QDQ node this is the count of input edges to the DQ nodes
  /// plus any other edges to the target node for inputs that are not via a DQ node.
  size_t InputEdgeCount() const { return g_host->NodeUnit__InputEdgeCount(this); }

  // output edges. src index is for outputs of the target node. dest index and node is for consumer of node unit
  // output. any Q nodes are hidden.
  Node::EdgeConstIterator OutputEdgesBegin() const { return g_host->NodeUnit__OutputEdgesBegin(this); }
  Node::EdgeConstIterator OutputEdgesEnd() const { return g_host->NodeUnit__OutputEdgesEnd(this); }
};

struct ModelSavingOptions;

struct Model final {
  static std::unique_ptr<Model> Create(ONNX_NAMESPACE::ModelProto&& model_proto, const PathString& model_path,
                                       const IOnnxRuntimeOpSchemaRegistryList* local_registries, const logging::Logger& logger) {
    return g_host->Model__construct(std::move(model_proto), model_path, local_registries, logger);
  }
  static std::unique_ptr<Model> Create(const std::string& graph_name, bool is_onnx_domain_only,
                                       const logging::Logger& logger) {
    return g_host->Model__construct(graph_name, is_onnx_domain_only, logger);
  }
  static void operator delete(void* p) { g_host->Model__operator_delete(reinterpret_cast<Model*>(p)); }
  static Status Load(const PathString& file_path, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) { return g_host->Model__Load(file_path, model_proto); }

  Graph& MainGraph() { return g_host->Model__MainGraph(this); }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToProto() { return g_host->Model__ToProto(this); }
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToGraphProtoWithExternalInitializers(
      const std::filesystem::path& external_file_name,
      const std::filesystem::path& file_path, const ModelSavingOptions& model_saving_options) {
    return g_host->Model__ToGraphProtoWithExternalInitializers(this, external_file_name, file_path,
                                                               model_saving_options);
  }
  const ModelMetaData& MetaData() const noexcept { return g_host->Model__MetaData(this); }

  Model() = delete;
  Model(const Model&) = delete;
  void operator=(const Model&) = delete;
};

struct Graph final {
  std::unique_ptr<GraphViewer> CreateGraphViewer() const { return g_host->Graph__CreateGraphViewer(this); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> ToGraphProto() const { return g_host->Graph__ToGraphProto(this); }

  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) { return g_host->Graph__GetOrCreateNodeArg(this, name, p_arg_type); }
  void AddOuterScopeNodeArg(const std::string& name) { g_host->Graph__AddOuterScopeNodeArg(this, name); }
  void SetInputs(gsl::span<const NodeArg* const> inputs) { g_host->Graph__SetInputs(this, inputs); }
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->Graph__DomainToVersionMap(this); }
  Status Resolve() { return g_host->Graph__Resolve(this); }
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor) { return g_host->Graph__AddInitializedTensor(this, tensor); }
  Status AddInitializedOrtValue(const ONNX_NAMESPACE::TensorProto& tensor, const OrtValue& ort_value) {
    return g_host->Graph__AddInitializedOrtValue(this, tensor, ort_value);
  }
  bool GetOrtValueInitializer(const std::string& tensor_name, OrtValue& ort_value,
                              bool check_outer_scope = false) const {
    return g_host->Graph__GetOrtValueInitializer(this, tensor_name, ort_value, check_outer_scope);
  }
  Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, gsl::span<NodeArg* const> input_args, gsl::span<NodeArg* const> output_args, const NodeAttributes* attributes, const std::string& domain) { return g_host->Graph__AddNode(this, name, op_type, description, input_args, output_args, attributes, domain); }
  Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, gsl::span<NodeArg* const> input_args, gsl::span<NodeArg* const> output_args, NodeAttributes&& attributes, const std::string& domain) { return g_host->Graph__AddNode(this, name, op_type, description, input_args, output_args, std::move(attributes), domain); }
  Node& AddNode(const Node& other) { return g_host->Graph__AddNode(this, other); }

  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->Graph__GetOutputs(this); }
  void SetOutputs(gsl::span<const NodeArg* const> outputs) { return g_host->Graph__SetOutputs(this, outputs); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->Graph__GetInputs(this); }
  std::vector<const Node*> Nodes() const noexcept { return g_host->Graph__Nodes(this); }

  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->Graph__GetInitializedTensor(this, tensor_name, value); }

  const Node* ParentNode() const { return g_host->Graph__ParentNode(this); }
  const Graph* ParentGraph() const { return g_host->Graph__ParentGraph(this); }
  Graph* MutableParentGraph() { return g_host->Graph__MutableParentGraph(this); }
  const std::string& Name() const noexcept { return g_host->Graph__Name(this); }
  void SetName(const std::string& name) noexcept { return g_host->Graph__SetName(this, name); }
  const std::filesystem::path& ModelPath() const { return g_host->Graph__ModelPath(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return g_host->Graph__GetInputsIncludingInitializers(this); }
  bool IsSubgraph() const { return g_host->Graph__IsSubgraph(this); }
  const Node* GetProducerNode(const std::string& node_arg_name) const { return g_host->Graph__GetProducerNode(this, node_arg_name); }
  const Model& GetModel() const { return g_host->Graph__GetModel(this); }
  void ReverseDFSFrom(gsl::span<const Node* const> from, const std::function<void(const Node*)>& enter,
                      const std::function<void(const Node*)>& leave,
                      const std::function<bool(const Node*, const Node*)>& comp,
                      const std::function<bool(const Node* from, const Node* to)>& stop) const {
    g_host->Graph__ReverseDFSFrom(this, from, enter, leave, comp, stop);
  }
  Graph& SetGraphResolveNeeded() { return g_host->Graph__SetGraphResolveNeeded(this); }
  void RemoveInitializedTensor(const std::string& tensor_name) { g_host->Graph__RemoveInitializedTensor(this, tensor_name); }

  std::vector<const Node*> GetConsumerNodes(const std::string& node_arg_name) const {
    return g_host->Graph__GetConsumerNodes(this, node_arg_name);
  }
  void AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index, int dst_arg_index) {
    g_host->Graph__AddEdge(this, src_node_index, dst_node_index, src_arg_index, dst_arg_index);
  }
  void RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index, int dst_arg_index) {
    g_host->Graph__RemoveEdge(this, src_node_index, dst_node_index, src_arg_index, dst_arg_index);
  }
  void RemoveNode(NodeIndex index) { g_host->Graph__RemoveNode(this, index); }
  Node& FuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name) {
    return g_host->Graph__FuseSubGraph(this, sub_graph, fused_node_name);
  }
  void UpdateProducerNode(const std::string& node_arg_name, NodeIndex node_index) {
    g_host->Graph__UpdateProducerNode(this, node_arg_name, node_index);
  }
  const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const std::string& name, bool check_outer_scope) const {
    return g_host->Graph__GetConstantInitializer(this, name, check_outer_scope);
  }
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->Graph__GetAllInitializedTensors(this); }
  int MaxNodeIndex() const noexcept { return g_host->Graph__MaxNodeIndex(this); }
  const Node* GetNode(NodeIndex node_index) const noexcept { return g_host->Graph__GetNode(this, node_index); }
  Node* GetNode(NodeIndex node_index) noexcept { return g_host->Graph__GetNode(this, node_index); }
  const NodeArg* GetNodeArg(const std::string& name) const { return g_host->Graph__GetNodeArg(this, name); }
  IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry() const { return g_host->Graph__GetSchemaRegistry(this); }
  bool SetOpSchemaFromRegistryForNode(Node& node) { return g_host->Graph__SetOpSchemaFromRegistryForNode(this, node); }

  PROVIDER_DISALLOW_ALL(Graph)
};

using ModelMetaData = std::unordered_map<std::string, std::string>;

class GraphViewer final {
 public:
  static void operator delete(void* p) { g_host->GraphViewer__operator_delete(reinterpret_cast<GraphViewer*>(p)); }

  std::unique_ptr<Model> CreateModel(const logging::Logger& logger, const ModelMetaData& metadata = ModelMetaData()) const { return g_host->GraphViewer__CreateModel(this, logger, metadata); }

  const std::string& Name() const noexcept { return g_host->GraphViewer__Name(this); }
  const std::filesystem::path& ModelPath() const noexcept { return g_host->GraphViewer__ModelPath(this); }

  const ConstGraphNodes& Nodes() const noexcept { return g_host->GraphViewer__Nodes(this); }
  const Node* GetNode(NodeIndex node_index) const { return g_host->GraphViewer__GetNode(this, node_index); }
  const NodeArg* GetNodeArg(const std::string& name) const { return g_host->GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return g_host->GraphViewer__IsSubgraph(this); }
  const Graph& GetGraph() const { return g_host->GraphViewer__GetGraph(this); }
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const { return g_host->GraphViewer__IsConstantInitializer(this, name, check_outer_scope); }
  const ONNX_NAMESPACE::TensorProto* GetConstantInitializer(const std::string& name,
                                                            bool check_outer_scope = true) const {
    return g_host->GraphViewer__GetConstantInitializer(this, name, check_outer_scope);
  }
  bool GetOrtValueInitializer(const std::string& tensor_name, OrtValue& ort_value) const {
    return g_host->GraphViewer__GetOrtValueInitializer(this, tensor_name, ort_value);
  }
  const Node* ParentNode() const { return g_host->GraphViewer__ParentNode(this); }

  int NumberOfNodes() const noexcept { return g_host->GraphViewer__NumberOfNodes(this); }
  int MaxNodeIndex() const noexcept { return g_host->GraphViewer__MaxNodeIndex(this); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->GraphViewer__GetInputs(this); }
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->GraphViewer__GetOutputs(this); }
  bool NodeProducesGraphOutput(const Node& node) const {
    return g_host->GraphViewer__NodeProducesGraphOutput(this, node);
  }
  const std::unordered_set<const NodeArg*>& GetValueInfo() const noexcept { return g_host->GraphViewer__GetValueInfo(this); }

  const InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder(int execution_order = 0) const { return g_host->GraphViewer__GetNodesInTopologicalOrder(this, execution_order); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return g_host->GraphViewer__GetInputsIncludingInitializers(this); }
  const std::unordered_set<std::string>& GetOuterScopeNodeArgNames() const noexcept {
    return g_host->GraphViewer__GetOuterScopeNodeArgNames(this);
  }

  void ToProto(ONNX_NAMESPACE::GraphProto& graph_proto,
               bool include_initializers,
               bool include_outer_scope_args,
               int execution_order = 0,
               bool include_initializer_data = true) const {
    g_host->GraphViewer__ToProto(this, graph_proto, include_initializers, include_outer_scope_args, execution_order, include_initializer_data);
  }
  const Node* GetProducerNode(const std::string& node_arg_name) const { return g_host->GraphViewer__GetProducerNode(this, node_arg_name); }
  IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry() const { return g_host->GraphViewer__GetSchemaRegistry(this); }

  GraphViewer() = delete;
  GraphViewer(const GraphViewer&) = delete;
  void operator=(const GraphViewer&) = delete;
};

struct ConstGraphNodes final {
  IteratorHolder<ConstGraphNodes_Iterator, const Node> begin() const {
    return g_host->ConstGraphNodes__begin(this);
  }
  IteratorHolder<ConstGraphNodes_Iterator, const Node> end() const {
    return g_host->ConstGraphNodes__end(this);
  }
  IteratorHolder<ConstGraphNodes_Iterator, const Node> cbegin() const {
    return g_host->ConstGraphNodes__cbegin(this);
  }
  IteratorHolder<ConstGraphNodes_Iterator, const Node> cend() const {
    return g_host->ConstGraphNodes__cend(this);
  }

  bool empty() const noexcept { return g_host->ConstGraphNodes__empty(this); }

  PROVIDER_DISALLOW_ALL(ConstGraphNodes)
};

class ExternalDataInfo {
 public:
  static void operator delete(void* p) {
    g_host->ExternalDataInfo__operator_delete(reinterpret_cast<ExternalDataInfo*>(p));
  }

  const PathString& GetRelPath() const {
    return g_host->ExternalDataInfo__GetRelPath(this);
  }

  int64_t GetOffset() const {
    return g_host->ExternalDataInfo__GetOffset(this);
  }

  size_t GetLength() const {
    return g_host->ExternalDataInfo__GetLength(this);
  }

  const std::string& GetChecksum() const {
    return g_host->ExternalDataInfo__GetChecksum(this);
  }

  static Status Create(
      const ONNX_NAMESPACE::StringStringEntryProtos& input,
      std::unique_ptr<ExternalDataInfo>& out) {
    return g_host->ExternalDataInfo__Create(input, out);
  }

  ExternalDataInfo() = delete;
  ExternalDataInfo(const ExternalDataInfo&) = delete;
  ExternalDataInfo& operator=(const ExternalDataInfo& v) = delete;
};

class Initializer {
 public:
  Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
              std::string_view name,
              gsl::span<const int64_t> dims) {
    this_ptr_ = g_host->Initializer__constructor(data_type, name, dims);
  }

  Initializer(const Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor_proto,
              const std::filesystem::path& model_path = {},
              bool check_outer_scope = false) {
    this_ptr_ = g_host->Initializer__constructor(graph, tensor_proto, model_path, check_outer_scope);
  }

  ~Initializer() {
    g_host->Initializer__destructor(this_ptr_);
  }

  PROVIDER_DISALLOW_ALL(Initializer);

  void ToProto(ONNX_NAMESPACE::TensorProto& tensor_proto) const {
    g_host->Initializer__ToProto(*this_ptr_, tensor_proto);
  }

  void ToProtoWithOrtValue(ONNX_NAMESPACE::TensorProto& tensor_proto, OrtValue& ort_value) const {
    g_host->Initializer__ToProtoWithOrtValue(*this_ptr_, tensor_proto, ort_value);
  }

  int data_type() const {
    return g_host->Initializer__data_type(*this_ptr_);
  }

  const std::string& name() const {
    return g_host->Initializer__name(*this_ptr_);
  }

  gsl::span<const int64_t> dims() const {
    return g_host->Initializer__dims(*this_ptr_);
  }

  size_t size() const {
    return g_host->Initializer__size(*this_ptr_);
  }

  // See definition for the below templates in provider_api.h
  template <class T>
  const T* data() const;

  template <class T>
  T* data();

  const void* data_raw() const {
    return g_host->Initializer__data_raw(*this_ptr_);
  }

  void* mutable_data_raw() {
    return g_host->Initializer__mutable_data_raw(*this_ptr_);
  }

 private:
  Initializer* this_ptr_;
};

// Retained for source and provider vtable ABI compatibility. New code should use KernelPilot.
class RunInstrumentationContext final {
 public:
  const std::string& RequestId() const { return g_host->RunInstrumentationContext__RequestId(this); }
  TimePoint StartProfiling() const { return g_host->RunInstrumentationContext__StartProfiling(this); }
  uint64_t ProfilerStartTimeNs() const { return g_host->RunInstrumentationContext__ProfilerStartTimeNs(this); }
  void AddDeferredRecord(std::unique_ptr<DeferredRunInstrumentationRecord> record) const {
    g_host->RunInstrumentationContext__AddDeferredRecord(this, std::move(record));
  }
  bool TryReserveMoeRoutingRecord(size_t element_count) const {
    return g_host->RunInstrumentationContext__TryReserveMoeRoutingRecord(this, element_count);
  }
  void RecordMoeRoutingEvent(const TimePoint& start_time,
                             const TimePoint& end_time,
                             const std::string& node_name,
                             NodeIndex node_index,
                             const std::string& node_type,
                             std::string expert_ids_json,
                             std::string router_weights_json,
                             int64_t num_rows,
                             int64_t top_k,
                             int execution_device_id,
                             int64_t completion_ns,
                             const std::string& completion_timestamp_source) const {
    g_host->RunInstrumentationContext__RecordMoeRoutingEvent(
        this, start_time, end_time, node_name, node_index, node_type,
        std::move(expert_ids_json), std::move(router_weights_json),
        num_rows, top_k, execution_device_id, completion_ns, completion_timestamp_source);
  }

  PROVIDER_DISALLOW_ALL(RunInstrumentationContext)
};

struct OpKernelContext final {
  template <typename T>
  const T& RequiredInput(int index) const;
  Tensor& RequiredOutput(int index, const TensorShape& shape) { return g_host->OpKernelContext__RequiredOutput(this, index, shape); }

  template <typename T>
  const T* Input(int index) const;
  int InputCount() const { return g_host->OpKernelContext__InputCount(this); }

  MLDataType InputType(int index) const { return g_host->OpKernelContext__InputType(this, index); }

  template <typename T>
  T* Output(int index);

  Tensor* Output(int index, const TensorShape& shape) { return g_host->OpKernelContext__Output(this, index, shape); }
#if !defined(DISABLE_SPARSE_TENSORS)
  SparseTensor* OutputSparse(int index, const TensorShape& shape) { return g_host->OpKernelContext__OutputSparse(this, index, shape); }
#endif
  int OutputCount() const { return g_host->OpKernelContext__OutputCount(this); }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const { return g_host->OpKernelContext__GetTempSpaceAllocator(this, output); }

  Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const { return g_host->OpKernelContext__GetTempSpaceCPUAllocator(this, output); }

  bool GetUseDeterministicCompute() const { return g_host->OpKernelContext__GetUseDeterministicCompute(this); }

  bool TryGetInferredOutputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredOutputShape(this, index, shape); }
  bool TryGetInferredInputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredInputShape(this, index, shape); }
  Stream* GetComputeStream() const { return g_host->OpKernelContext__GetComputeStream(this); }
  const RunInstrumentationContext* GetRunInstrumentationContext() const {
    return g_host->OpKernelContext__GetRunInstrumentationContext(this);
  }
  KernelPilot* GetKernelPilot() const {
    return g_host->OpKernelContext__GetKernelPilot(this);
  }

  PROVIDER_DISALLOW_ALL(OpKernelContext)
};

template <>
inline const Tensor* OpKernelContext::Input<Tensor>(int index) const {
  return g_host->OpKernelContext__Input_Tensor(this, index);
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <>
inline const SparseTensor* OpKernelContext::Input<SparseTensor>(int index) const {
  return g_host->OpKernelContext__Input_SparseTensor(this, index);
}
#endif

template <>
inline const TensorSeq* OpKernelContext::Input<TensorSeq>(int index) const {
  return g_host->OpKernelContext__Input_TensorSeq(this, index);
}

template <>
inline Tensor* OpKernelContext::Output<Tensor>(int index) {
  return g_host->OpKernelContext__Output_Tensor(this, index);
}

template <>
inline TensorSeq* OpKernelContext::Output<TensorSeq>(int index) {
  return g_host->OpKernelContext__Output_TensorSeq(this, index);
}

template <>
inline const Tensor& OpKernelContext::RequiredInput(int index) const {
  return g_host->OpKernelContext__RequiredInput_Tensor(this, index);
}

struct OpKernelInfo final {
  static void operator delete(void* p) { g_host->OpKernelInfo__operator_delete(reinterpret_cast<OpKernelInfo*>(p)); }

  const IExecutionProvider* GetExecutionProvider() const noexcept { return g_host->OpKernelInfo__GetExecutionProvider(this); }

  AllocatorPtr GetAllocator(OrtMemType mem_type) const { return g_host->OpKernelInfo__GetAllocator(this, mem_type); }

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }

  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }

  template <typename T>
  std::vector<T> GetAttrsOrDefault(const std::string& name, const std::vector<T>& default_value = std::vector<T>{}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }

  template <typename T>
  Status GetAttrsAsSpan(const std::string& name, gsl::span<const T>& out) const;

  Status GetAttrs(const std::string& name, TensorShapeVector& out) const;

  TensorShapeVector GetAttrsOrDefault(const std::string& name, const TensorShapeVector& default_value = TensorShapeVector{}) const;

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const { return g_host->OpKernelInfo__TryGetConstantInput(this, input_index, constant_input_value); }

  const DataTransferManager& GetDataTransferManager() const noexcept { return g_host->OpKernelInfo__GetDataTransferManager(this); }
  const KernelDef& GetKernelDef() const { return g_host->OpKernelInfo__GetKernelDef(this); }

  uint32_t GetInputCount() const { return g_host->OpKernelInfo__GetInputCount(this); }
  uint32_t GetOutputCount() const { return g_host->OpKernelInfo__GetOutputCount(this); }

  const Node& node() const noexcept { return g_host->OpKernelInfo__node(this); }

  const ConfigOptions& GetConfigOptions() const { return g_host->OpKernelInfo__GetConfigOptions(this); }

  OpKernelInfo() = delete;
  OpKernelInfo(const OpKernelInfo&) = delete;
  void operator=(const OpKernelInfo&) = delete;
};

template <>
inline Status OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const { return g_host->OpKernelInfo__GetAttr_int64(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const { return g_host->OpKernelInfo__GetAttr_float(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<std::string>(const std::string& name, std::string* value) const { return g_host->OpKernelInfo__GetAttr_string(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<ONNX_NAMESPACE::TensorProto>(const std::string& name, ONNX_NAMESPACE::TensorProto* value) const { return g_host->OpKernelInfo__GetAttr_TensorProto(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttrs<int64_t>(const std::string& name, std::vector<int64_t>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<float>(const std::string& name, std::vector<float>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<std::string>(const std::string& name, std::vector<std::string>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrsAsSpan<int64_t>(const std::string& name, gsl::span<const int64_t>& values) const { return g_host->OpKernelInfo__GetAttrsAsSpan(this, name, values); }

inline Status OpKernelInfo::GetAttrs(const std::string& name, TensorShapeVector& out) const {
  gsl::span<const int64_t> span;
  Status status = this->GetAttrsAsSpan<int64_t>(name, span);
  if (status.IsOK()) {
    out.reserve(span.size());
    out.assign(span.begin(), span.end());
  }
  return status;
}

inline TensorShapeVector OpKernelInfo::GetAttrsOrDefault(const std::string& name, const TensorShapeVector& default_value) const {
  TensorShapeVector tmp;
  return GetAttrs(name, tmp).IsOK() ? tmp : default_value;
}

class SessionState {
 public:
  const DataTransferManager& GetDataTransferMgr() const noexcept { return g_host->SessionState__GetDataTransferMgr(this); }

  PROVIDER_DISALLOW_ALL(SessionState)
};

struct Tensor final {
  static std::unique_ptr<Tensor> CreateDefault() { return g_host->Tensor__construct_default(); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) { return g_host->Tensor__construct(p_type, shape, std::move(allocator)); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset = 0) { return g_host->Tensor__construct(p_type, shape, p_data, alloc, offset); }

  static void operator delete(void* p) noexcept { g_host->Tensor__operator_delete(reinterpret_cast<Tensor*>(p)); }

  static void InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) {
    g_host->Tensor__InitOrtValue(elt_type, shape, std::move(allocator), ort_value);
  }

  static void InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location, OrtValue& ort_value) {
    g_host->Tensor__InitOrtValue(p_type, shape, p_data, location, ort_value);
  }

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  template <typename T>
  gsl::span<const T> DataAsSpan() const;

  void* MutableDataRaw(MLDataType type) { return g_host->Tensor__MutableDataRaw(this, type); }
  const void* DataRaw(MLDataType type) const { return g_host->Tensor__DataRaw(this, type); }

  void* MutableDataRaw() noexcept { return g_host->Tensor__MutableDataRaw(this); }
  const void* DataRaw() const noexcept { return g_host->Tensor__DataRaw(this); }

  const TensorShape& Shape() const { return g_host->Tensor__Shape(this); }
  void Reshape(const TensorShape& new_shape) { g_host->Tensor__Reshape(this, new_shape); }
  void SetByteOffset(ptrdiff_t byte_offset) { return g_host->Tensor__SetByteOffset(this, byte_offset); }
  ptrdiff_t ByteOffset() const { return g_host->Tensor__ByteOffset(this); }
  size_t SizeInBytes() const { return g_host->Tensor__SizeInBytes(this); }
  const OrtMemoryInfo& Location() const { return g_host->Tensor__Location(this); }

  int32_t GetElementType() const { return g_host->Tensor__GetElementType(this); }
  MLDataType DataType() const { return g_host->Tensor__DataType(this); }
  bool IsDataTypeString() const { return g_host->Tensor__IsDataTypeString(this); }

#ifdef ENABLE_STRIDED_TENSORS
  gsl::span<const int64_t> Strides() const noexcept { return g_host->Tensor__Strides(this); }
  bool IsContiguous() const { return g_host->Tensor__IsContiguous(this); }
  void SetShapeAndStrides(const TensorShape& new_shape, gsl::span<const int64_t> new_strides) {
    return g_host->Tensor__SetShapeAndStrides(this, new_shape, new_strides);
  }
#endif

  template <class T>
  bool IsDataType() const;

  Tensor() = delete;
  Tensor(const Tensor&) = delete;
  void operator=(const Tensor&) = delete;
  Tensor& operator=(Tensor&& o) noexcept {
    g_host->Tensor__move_assign(*this, std::move(o));
    return *this;
  }
};

template <>
inline bool Tensor::IsDataType<bool>() const { return g_host->Tensor__IsDataType_bool(this); }
template <>
inline bool Tensor::IsDataType<Int4x2>() const { return g_host->Tensor__IsDataType_Int4x2(this); }
template <>
inline bool Tensor::IsDataType<UInt4x2>() const { return g_host->Tensor__IsDataType_UInt4x2(this); }
template <>
inline bool Tensor::IsDataType<Int2x4>() const { return g_host->Tensor__IsDataType_Int2x4(this); }
template <>
inline bool Tensor::IsDataType<UInt2x4>() const { return g_host->Tensor__IsDataType_UInt2x4(this); }
template <>
inline bool Tensor::IsDataType<int8_t>() const { return g_host->Tensor__IsDataType_int8(this); }
template <>
inline bool Tensor::IsDataType<uint8_t>() const { return g_host->Tensor__IsDataType_uint8(this); }
template <>
inline bool Tensor::IsDataType<int16_t>() const { return g_host->Tensor__IsDataType_int16(this); }
template <>
inline bool Tensor::IsDataType<uint16_t>() const { return g_host->Tensor__IsDataType_uint16(this); }
template <>
inline bool Tensor::IsDataType<int32_t>() const { return g_host->Tensor__IsDataType_int32(this); }
template <>
inline bool Tensor::IsDataType<uint32_t>() const { return g_host->Tensor__IsDataType_uint32(this); }
template <>
inline bool Tensor::IsDataType<int64_t>() const { return g_host->Tensor__IsDataType_int64(this); }
template <>
inline bool Tensor::IsDataType<uint64_t>() const { return g_host->Tensor__IsDataType_uint64(this); }
template <>
inline bool Tensor::IsDataType<float>() const { return g_host->Tensor__IsDataType_float(this); }
template <>
inline bool Tensor::IsDataType<double>() const { return g_host->Tensor__IsDataType_double(this); }
template <>
inline bool Tensor::IsDataType<MLFloat16>() const { return g_host->Tensor__IsDataType_MLFloat16(this); }
template <>
inline bool Tensor::IsDataType<BFloat16>() const { return g_host->Tensor__IsDataType_BFloat16(this); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
inline bool Tensor::IsDataType<Float8E4M3FN>() const { return g_host->Tensor__IsDataType_Float8E4M3FN(this); }
template <>
inline bool Tensor::IsDataType<Float8E4M3FNUZ>() const { return g_host->Tensor__IsDataType_Float8E4M3FNUZ(this); }
template <>
inline bool Tensor::IsDataType<Float8E5M2>() const { return g_host->Tensor__IsDataType_Float8E5M2(this); }
template <>
inline bool Tensor::IsDataType<Float8E5M2FNUZ>() const { return g_host->Tensor__IsDataType_Float8E5M2FNUZ(this); }
template <>
inline bool Tensor::IsDataType<Float8E8M0>() const { return g_host->Tensor__IsDataType_Float8E8M0(this); }
#endif

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
inline bool Tensor::IsDataType<Float4E2M1x2>() const { return g_host->Tensor__IsDataType_Float4E2M1x2(this); }
#endif
template <>
inline bool Tensor::IsDataType<Float6E2M3>() const { return g_host->Tensor__IsDataType_Float6E2M3(this); }
template <>
inline bool Tensor::IsDataType<Float6E3M2>() const { return g_host->Tensor__IsDataType_Float6E3M2(this); }

template <>
inline bool* Tensor::MutableData<bool>() { return g_host->Tensor__MutableData_bool(this); }
template <>
inline Int4x2* Tensor::MutableData<Int4x2>() { return g_host->Tensor__MutableData_Int4x2(this); }
template <>
inline UInt4x2* Tensor::MutableData<UInt4x2>() { return g_host->Tensor__MutableData_UInt4x2(this); }
template <>
inline Int2x4* Tensor::MutableData<Int2x4>() { return g_host->Tensor__MutableData_Int2x4(this); }
template <>
inline UInt2x4* Tensor::MutableData<UInt2x4>() { return g_host->Tensor__MutableData_UInt2x4(this); }
template <>
inline int8_t* Tensor::MutableData<int8_t>() { return g_host->Tensor__MutableData_int8(this); }
template <>
inline uint8_t* Tensor::MutableData<uint8_t>() { return g_host->Tensor__MutableData_uint8(this); }
template <>
inline int16_t* Tensor::MutableData<int16_t>() { return g_host->Tensor__MutableData_int16(this); }
template <>
inline uint16_t* Tensor::MutableData<uint16_t>() { return g_host->Tensor__MutableData_uint16(this); }
template <>
inline int32_t* Tensor::MutableData<int32_t>() { return g_host->Tensor__MutableData_int32(this); }
template <>
inline uint32_t* Tensor::MutableData<uint32_t>() { return g_host->Tensor__MutableData_uint32(this); }
template <>
inline int64_t* Tensor::MutableData<int64_t>() { return g_host->Tensor__MutableData_int64(this); }
template <>
inline uint64_t* Tensor::MutableData<uint64_t>() { return g_host->Tensor__MutableData_uint64(this); }
template <>
inline float* Tensor::MutableData<float>() { return g_host->Tensor__MutableData_float(this); }
template <>
inline double* Tensor::MutableData<double>() { return g_host->Tensor__MutableData_double(this); }
template <>
inline BFloat16* Tensor::MutableData<BFloat16>() { return g_host->Tensor__MutableData_BFloat16(this); }
template <>
inline MLFloat16* Tensor::MutableData<MLFloat16>() { return g_host->Tensor__MutableData_MLFloat16(this); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
inline Float8E4M3FN* Tensor::MutableData<Float8E4M3FN>() { return g_host->Tensor__MutableData_Float8E4M3FN(this); }
template <>
inline Float8E4M3FNUZ* Tensor::MutableData<Float8E4M3FNUZ>() { return g_host->Tensor__MutableData_Float8E4M3FNUZ(this); }
template <>
inline Float8E5M2* Tensor::MutableData<Float8E5M2>() { return g_host->Tensor__MutableData_Float8E5M2(this); }
template <>
inline Float8E5M2FNUZ* Tensor::MutableData<Float8E5M2FNUZ>() { return g_host->Tensor__MutableData_Float8E5M2FNUZ(this); }
template <>
inline Float8E8M0* Tensor::MutableData<Float8E8M0>() { return g_host->Tensor__MutableData_Float8E8M0(this); }
#endif

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
inline Float4E2M1x2* Tensor::MutableData<Float4E2M1x2>() { return g_host->Tensor__MutableData_Float4E2M1x2(this); }
#endif
template <>
inline Float6E2M3* Tensor::MutableData<Float6E2M3>() { return g_host->Tensor__MutableData_Float6E2M3(this); }
template <>
inline Float6E3M2* Tensor::MutableData<Float6E3M2>() { return g_host->Tensor__MutableData_Float6E3M2(this); }

template <>
inline const bool* Tensor::Data<bool>() const { return g_host->Tensor__Data_bool(this); }
template <>
inline const Int4x2* Tensor::Data<Int4x2>() const { return g_host->Tensor__Data_Int4x2(this); }
template <>
inline const UInt4x2* Tensor::Data<UInt4x2>() const { return g_host->Tensor__Data_UInt4x2(this); }
template <>
inline const Int2x4* Tensor::Data<Int2x4>() const { return g_host->Tensor__Data_Int2x4(this); }
template <>
inline const UInt2x4* Tensor::Data<UInt2x4>() const { return g_host->Tensor__Data_UInt2x4(this); }
template <>
inline const Float6E2M3* Tensor::Data<Float6E2M3>() const { return g_host->Tensor__Data_Float6E2M3(this); }
template <>
inline const Float6E3M2* Tensor::Data<Float6E3M2>() const { return g_host->Tensor__Data_Float6E3M2(this); }
template <>
inline const int8_t* Tensor::Data<int8_t>() const { return g_host->Tensor__Data_int8(this); }
template <>
inline const uint8_t* Tensor::Data<uint8_t>() const { return g_host->Tensor__Data_uint8(this); }
template <>
inline const int16_t* Tensor::Data<int16_t>() const { return g_host->Tensor__Data_int16(this); }
template <>
inline const uint16_t* Tensor::Data<uint16_t>() const { return g_host->Tensor__Data_uint16(this); }
template <>
inline const int32_t* Tensor::Data<int32_t>() const { return g_host->Tensor__Data_int32(this); }
template <>
inline const uint32_t* Tensor::Data<uint32_t>() const { return g_host->Tensor__Data_uint32(this); }
template <>
inline const int64_t* Tensor::Data<int64_t>() const { return g_host->Tensor__Data_int64(this); }
template <>
inline const uint64_t* Tensor::Data<uint64_t>() const { return g_host->Tensor__Data_uint64(this); }
template <>
inline const float* Tensor::Data<float>() const { return g_host->Tensor__Data_float(this); }
template <>
inline const double* Tensor::Data<double>() const { return g_host->Tensor__Data_double(this); }
template <>
inline const BFloat16* Tensor::Data<BFloat16>() const { return g_host->Tensor__Data_BFloat16(this); }
template <>
inline const MLFloat16* Tensor::Data<MLFloat16>() const { return g_host->Tensor__Data_MLFloat16(this); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
inline const Float8E4M3FN* Tensor::Data<Float8E4M3FN>() const { return g_host->Tensor__Data_Float8E4M3FN(this); }
template <>
inline const Float8E4M3FNUZ* Tensor::Data<Float8E4M3FNUZ>() const { return g_host->Tensor__Data_Float8E4M3FNUZ(this); }
template <>
inline const Float8E5M2* Tensor::Data<Float8E5M2>() const { return g_host->Tensor__Data_Float8E5M2(this); }
template <>
inline const Float8E5M2FNUZ* Tensor::Data<Float8E5M2FNUZ>() const { return g_host->Tensor__Data_Float8E5M2FNUZ(this); }
template <>
inline const Float8E8M0* Tensor::Data<Float8E8M0>() const { return g_host->Tensor__Data_Float8E8M0(this); }
#endif

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
inline const Float4E2M1x2* Tensor::Data<Float4E2M1x2>() const { return g_host->Tensor__Data_Float4E2M1x2(this); }
#endif

// SparseTensor
#if !defined(DISABLE_SPARSE_TENSORS)
struct SparseTensor final {
  const TensorShape& DenseShape() const noexcept { return g_host->SparseTensor__DenseShape(this); }
  Status Copy(const DataTransferManager& dtm, SparseTensor& dst) const { return g_host->SparseTensor__Copy(this, dtm, dst); }
};
#endif

// TensorSeq
class TensorSeq final {
 public:
  MLDataType DataType() const noexcept { return g_host->TensorSeq__DataType(this); }
  void SetType(MLDataType elem_type) { g_host->TensorSeq__SetType(this, elem_type); }
  size_t Size() const noexcept { return g_host->TensorSeq__Size(this); }
  const Tensor& Get(size_t i) const { return g_host->TensorSeq__Get(this, i); }
  const OrtValue& GetAt(size_t i) const { return g_host->TensorSeq__GetAt(this, i); }
  void Add(const OrtValue& tensor) { g_host->TensorSeq__Add(this, tensor); }
  void Add(OrtValue&& tensor) { g_host->TensorSeq__Add(this, std::move(tensor)); }
  void Add(Tensor&& tensor) { g_host->TensorSeq__Add(this, std::move(tensor)); }
  void Reserve(size_t capacity) { g_host->TensorSeq__Reserve(this, capacity); }
};

class ModelMetadefIdGenerator {
 public:
  static std::unique_ptr<ModelMetadefIdGenerator> Create() { return g_host->ModelMetadefIdGenerator__construct(); }
  static void operator delete(void* p) { g_host->ModelMetadefIdGenerator__operator_delete(reinterpret_cast<ModelMetadefIdGenerator*>(p)); }
  int GenerateId(const GraphViewer& graph_viewer, HashValue& model_hash) const { return g_host->ModelMetadefIdGenerator__GenerateId(this, graph_viewer, model_hash); }
};

template <>
inline gsl::span<const int64_t> Tensor::DataAsSpan() const { return g_host->Tensor__DataAsSpan_int64(this); }
template <>
inline gsl::span<const int32_t> Tensor::DataAsSpan() const { return g_host->Tensor__DataAsSpan_int32(this); }

}  // namespace onnxruntime

struct OrtRunOptions final {
  const onnxruntime::ConfigOptions& GetConfigOptions() const {
    return onnxruntime::g_host->RunOptions__GetConfigOptions(this);
  }
  PROVIDER_DISALLOW_ALL(OrtRunOptions)
};

struct OrtSessionOptions final {
  const std::unordered_map<std::string, std::string>& GetConfigOptionsMap() const {
    return onnxruntime::g_host->SessionOptions__GetConfigOptionsMap(this);
  }

  const onnxruntime::ConfigOptions& GetConfigOptions() const {
    return onnxruntime::g_host->SessionOptions__GetConfigOptions(this);
  }

  bool GetEnableProfiling() const {
    return onnxruntime::g_host->SessionOptions__GetEnableProfiling(this);
  }

  void GetEpContextDataCallbacks(OrtReadNamedBufferFunc* read_func, void** read_state,
                                 OrtWriteNamedBufferFunc* write_func, void** write_state) const {
    onnxruntime::g_host->SessionOptions__GetEpContextDataCallbacks(this, read_func, read_state,
                                                                   write_func, write_state);
  }
  PROVIDER_DISALLOW_ALL(OrtSessionOptions)
};
