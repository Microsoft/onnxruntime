// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <optional>
#include <utility>
#include <vector>
#include <list>
#include <filesystem>

// Public wrappers around internal ort interfaces (currently)
#include "core/providers/shared_library/provider_host_api.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/run_instrumentation.h"
#include "core/framework/resource_accountant.h"
#include "core/providers/shared/common.h"

#define PROVIDER_DISALLOW_ALL(TypeName)     \
  TypeName() = delete;                      \
  TypeName(const TypeName&) = delete;       \
  void operator=(const TypeName&) = delete; \
  static void operator delete(void*) = delete;

namespace ONNX_NAMESPACE {
using namespace onnxruntime;

enum AttributeProto_AttributeType : int;
enum OperatorStatus : int;

// String pointer as unique TypeProto identifier.
using DataType = const std::string*;
using DataTypeSet = std::unordered_set<DataType>;
using TypeConstraintMap = std::unordered_map<std::string, std::pair<DataTypeSet, std::string>>;

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
// These types don't directly map to internal types
struct ProviderHost;
struct ProviderHostCPU;

class ExternalDataInfo;

class PhiloxGenerator;
using ProviderType = const std::string&;
class RandomGenerator;
class Initializer;
class IOnnxRuntimeOpSchemaCollection;
class KernelPilot;
class RunInstrumentationContext;

struct ModelSavingOptions;

#ifdef ENABLE_TRAINING_TORCH_INTEROP
namespace contrib {
class PythonOpBase;
class PythonOpGradBase;
}  // namespace contrib

namespace language_interop_ops {
namespace torch {
class RefCountTracker;
}  // namespace torch
}  // namespace language_interop_ops
#endif

namespace training {
class DistributedRunContext;
}

template <typename T, typename TResult>
struct IteratorHolder {
  IteratorHolder(std::unique_ptr<T>&& p) : p_{std::move(p)} {}

  bool operator!=(const IteratorHolder& p) const { return p_->operator!=(*p.p_); }

  void operator++() { p_->operator++(); }
  TResult& operator*() { return p_->operator*(); }
  T* operator->() { return p_.get(); }

 private:
  std::unique_ptr<T> p_;
};

struct TensorProto_ConstIterator {
  virtual ~TensorProto_ConstIterator() = default;
  virtual bool operator!=(const TensorProto_ConstIterator& p) const = 0;
  virtual void operator++() = 0;
  virtual const ONNX_NAMESPACE::TensorProto& operator*() const = 0;
};

struct TensorProto_Iterator {
  virtual ~TensorProto_Iterator() = default;
  virtual bool operator!=(const TensorProto_Iterator& p) const = 0;
  virtual void operator++() = 0;
  virtual ONNX_NAMESPACE::TensorProto& operator*() const = 0;
};

struct NodeAttributes_Iterator {
  virtual ~NodeAttributes_Iterator() {}

  virtual bool operator!=(const NodeAttributes_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>& operator*() const = 0;

  virtual const std::string& first() const = 0;
  virtual const ONNX_NAMESPACE::AttributeProto& second() const = 0;
};

struct TensorShapeProto_Dimension_Iterator {
  virtual ~TensorShapeProto_Dimension_Iterator() {}

  virtual bool operator!=(const TensorShapeProto_Dimension_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimension& operator*() = 0;
};

using HashValue = uint64_t;
using NodeIndex = size_t;
// We can't just reinterpret_cast this one, since it's an unordered_map of object BY VALUE (can't do anything by value on the real types)
// using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto_Copyable>;
using ModelMetaData = std::unordered_map<std::string, std::string>;

using IOnnxRuntimeOpSchemaCollectionPtr = std::shared_ptr<IOnnxRuntimeOpSchemaCollection>;
using IOnnxRuntimeOpSchemaRegistryList = std::list<IOnnxRuntimeOpSchemaCollectionPtr>;
using InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;
using KeyValueConfig = std::unordered_map<std::string, std::string>;
using SelectionFunc = std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&, const KeyValueConfig&, const GraphOptimizerRegistry&)>;

struct Node__NodeIterator {
  virtual ~Node__NodeIterator() {}

  virtual bool operator!=(const Node__NodeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Node& operator*() = 0;
};

struct Node__EdgeIterator {
  virtual ~Node__EdgeIterator() {}
  virtual bool operator!=(const Node__EdgeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Node_EdgeEnd& operator*() const = 0;
  virtual const Node& GetNode() const = 0;
  virtual int GetSrcArgIndex() const = 0;
  virtual int GetDstArgIndex() const = 0;
};

struct ConstGraphNodes_Iterator {
  virtual ~ConstGraphNodes_Iterator() {}

  virtual bool operator!=(const ConstGraphNodes_Iterator& other) const = 0;
  virtual void operator++() = 0;
  virtual const Node& operator*() = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to
// member function).
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases
// where we're calling a specific implementation of a virtual class member. Trying to get a pointer to member of a
// virtual function will return a thunk that calls the virtual function (which will lead to infinite recursion in the
// bridge). There is no known way to get the non virtual member function pointer implementation in this case.
// The suppressed warning is:
//  "The type with a virtual function needs either public virtual or protected nonvirtual destructor."
// However, we do not allocate this type on heap.
// Please do not new or delete this type(and subtypes).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26436)
#endif
struct ProviderHost {
  virtual const OrtApiBase* OrtGetApiBase() = 0;

  virtual Status GetOptimizerByName(const std::string& name,
                                    const GraphOptimizerRegistry& graph_optimizer_registry,
                                    SelectionFunc& selection_func) = 0;

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::string GetEnvironmentVar(const std::string& var_name) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status,
                               const char* file, const char* function, uint32_t line) = 0;

  virtual std::vector<std::string> GetStackTrace() = 0;

  virtual OrtStatus* CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept = 0;

  virtual AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) = 0;

  virtual std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) = 0;

  virtual void* CPUAllocator__Alloc(CPUAllocator* p, size_t size) = 0;
  virtual void CPUAllocator__Free(CPUAllocator* p, void* allocation) = 0;

  virtual unsigned int GetThreadId() = 0;
  virtual unsigned int GetProcessId() = 0;

  virtual std::string demangle(const char* name) = 0;
  virtual std::string demangle(const std::string& name) = 0;

  virtual std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) = 0;

  virtual Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;
  virtual void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;

  virtual std::unique_ptr<IAllocator> CreateMIGraphXAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(int16_t device_id, const char* name) = 0;

  virtual std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                             const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                             gsl::span<const NodeIndex> tentative_nodes,
                                                             const logging::Logger& logger) = 0;

  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ bool* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ float* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ double* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ MLFloat16* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ BFloat16* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int8_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint8_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int16_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint16_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int32_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint32_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int64_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint64_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& tensor, const std::filesystem::path& model_path,
                                       /*out*/ std::vector<uint8_t>& unpacked_tensor) = 0;

  virtual uint16_t math__floatToHalf(float f) = 0;
  virtual float math__halfToFloat(uint16_t h) = 0;

  // sparse_utils
#if !defined(DISABLE_SPARSE_TENSORS)
#if !defined(ORT_MINIMAL_BUILD)
  virtual Status sparse_utils__DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                                      const AllocatorPtr& dst_allocator, SparseTensor& dst) = 0;
  virtual Status sparse_utils__SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                                      const AllocatorPtr& dst_allocator, Tensor& dst) = 0;

  virtual Status sparse_utils__SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                                      const AllocatorPtr& dst_allocator, Tensor& dst) = 0;
#endif  // !ORT_MINIMAL_BUILD
  virtual Status sparse_utils__DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                                      const AllocatorPtr& dst_allocator, bool linear_indexs, SparseTensor& dst) = 0;
#endif  // !defined(DISABLE_SPARSE_TENSORS)

  // IAllocator
  virtual bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) = 0;

  // IExecutionProviderFactory
  virtual std::unique_ptr<IExecutionProvider> IExecutionProviderFactory__CreateProvider(
      IExecutionProviderFactory* p, const OrtSessionOptions& session_options, const OrtLogger& session_logger) = 0;

  // IExecutionProvider
  virtual std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
                                                                                            const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                                                            const GraphOptimizerRegistry& graph_optimizer_registry,
                                                                                            IResourceAccountant* resource_accountant) = 0;

  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  // Status
  virtual std::string Status__ToString(const Status* p) = 0;

  // TensorShape
  virtual void TensorShape__operator_assign(TensorShape* p, const TensorShape& other) = 0;
  virtual void TensorShape__operator_move_assign(TensorShape* p, TensorShape&& other) noexcept = 0;
  virtual void TensorShape__Allocate(TensorShape* p, size_t size) = 0;
  virtual int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) = 0;
  virtual std::string TensorShape__ToString(const TensorShape* p) = 0;
  virtual int64_t TensorShape__SizeToDimension(const TensorShape* p, size_t dimension) = 0;
  virtual int64_t TensorShape__SizeFromDimension(const TensorShape* p, size_t dimension) = 0;
  virtual std::ostream& operator_left_shift(std::ostream& out, const TensorShape& shape) = 0;

  // CPUIDInfo
  virtual const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() = 0;
  virtual bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAVX512_BF16(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAMX_BF16(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAVX512Skylake(const CPUIDInfo* p) = 0;

  // logging::Logger
  virtual bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) = 0;
  virtual logging::Severity logging__Logger__GetSeverity(const logging::Logger* p) = 0;

  // logging::LoggingManager
  virtual const logging::Logger& logging__LoggingManager__DefaultLogger() = 0;
  virtual bool logging__LoggingManager__HasDefaultLogger() = 0;

  // logging::Capture
  virtual std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger,
                                                                        logging::Severity severity,
                                                                        const char* category,
                                                                        logging::DataType data_type,
                                                                        const CodeLocation& location) = 0;
  virtual void logging__Capture__operator_delete(logging::Capture* p) noexcept = 0;
  virtual std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept = 0;
  virtual void logging__Capture__ProcessPrintf(logging::Capture* p, const char* format, va_list args) = 0;

#if defined(_WIN32)
  // logging::EtwRegistrationManager
  virtual logging::EtwRegistrationManager& logging__EtwRegistrationManager__Instance() = 0;
  virtual bool logging__EtwRegistrationManager__SupportsETW() = 0;
  virtual logging::Severity logging__EtwRegistrationManager__MapLevelToSeverity(logging::EtwRegistrationManager* p) = 0;
  virtual void logging__EtwRegistrationManager__RegisterInternalCallback(
      logging::EtwRegistrationManager* p,
      const logging::EtwRegistrationManager_EtwInternalCallback& callback) = 0;
  virtual void logging__EtwRegistrationManager__UnregisterInternalCallback(
      logging::EtwRegistrationManager* p,
      const logging::EtwRegistrationManager_EtwInternalCallback& callback) = 0;
#endif  // defined(_WIN32)

  // Env
  virtual Env& Env__Default() = 0;

  // Utils::DataTypeUtils
  virtual const std::string* Utils__DataTypeUtils__ToType(const ONNX_NAMESPACE::TypeProto& type_proto) = 0;
  virtual const std::string* Utils__DataTypeUtils__ToType(const std::string& type_str) = 0;

  // int64s
  virtual int int64s__size(const ONNX_NAMESPACE::int64s* p) = 0;
  virtual const int64_t& int64s__Get(const ONNX_NAMESPACE::int64s* p, int index) = 0;
  virtual void int64s__Reserve(ONNX_NAMESPACE::int64s* p, int size) = 0;
  virtual const int64_t* int64s__data(const ONNX_NAMESPACE::int64s* p) = 0;

  // float32s
  virtual void float32s__Reserve(ONNX_NAMESPACE::float32s* p, int size) = 0;
  virtual const float* float32s__data(const ONNX_NAMESPACE::float32s* p) = 0;
  virtual int float32s__size(const ONNX_NAMESPACE::float32s* p) = 0;

  // StringStringEntryProto
  virtual std::string* StringStringEntryProto__mutable_key(ONNX_NAMESPACE::StringStringEntryProto* p) = 0;
  virtual std::string* StringStringEntryProto__mutable_value(ONNX_NAMESPACE::StringStringEntryProto* p) = 0;

  // StringStringEntryProtos
  virtual void StringStringEntryProtos__Clear(ONNX_NAMESPACE::StringStringEntryProtos* p) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProto* StringStringEntryProtos__Add(ONNX_NAMESPACE::StringStringEntryProtos* p) = 0;
  virtual int StringStringEntryProtos__size(ONNX_NAMESPACE::StringStringEntryProtos* p) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProto& StringStringEntryProtos__at(ONNX_NAMESPACE::StringStringEntryProtos* p, int index) = 0;

  // OperatorSetIdProto
  virtual std::string* OperatorSetIdProto__mutable_domain(ONNX_NAMESPACE::OperatorSetIdProto* p) = 0;
  virtual void OperatorSetIdProto__set_version(ONNX_NAMESPACE::OperatorSetIdProto* p, int64_t version) = 0;
  virtual int64_t OperatorSetIdProto__version(const ONNX_NAMESPACE::OperatorSetIdProto* p) = 0;

#if !defined(DISABLE_OPTIONAL_TYPE)
  // TypeProto_Optional
  virtual const ONNX_NAMESPACE::TypeProto& TypeProto_Optional__elem_type(const ONNX_NAMESPACE::TypeProto_Optional* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto* TypeProto_Optional__mutable_elem_type(ONNX_NAMESPACE::TypeProto_Optional* p) = 0;
#endif

  // TypeProto_Sequence
  virtual const ONNX_NAMESPACE::TypeProto& TypeProto_Sequence__elem_type(const ONNX_NAMESPACE::TypeProto_Sequence* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto* TypeProto_Sequence__mutable_elem_type(ONNX_NAMESPACE::TypeProto_Sequence* p) = 0;

  // TypeProto_Tensor
  virtual bool TypeProto_Tensor__has_shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto& TypeProto_Tensor__shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto* TypeProto_Tensor__mutable_shape(ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual bool TypeProto_Tensor__has_elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual int32_t TypeProto_Tensor__elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual void TypeProto_Tensor__set_elem_type(ONNX_NAMESPACE::TypeProto_Tensor* p, int32_t value) = 0;

#if !defined(DISABLE_SPARSE_TENSORS)
  // TypeProto_SparseTensor
  virtual bool TypeProto_SparseTensor__has_shape(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto& TypeProto_SparseTensor__shape(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto* TypeProto_SparseTensor__mutable_shape(ONNX_NAMESPACE::TypeProto_SparseTensor* p) = 0;
  virtual int32_t TypeProto_SparseTensor__elem_type(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) = 0;
#endif

  // TypeProto
  virtual std::unique_ptr<ONNX_NAMESPACE::TypeProto> TypeProto__construct() = 0;
  virtual void TypeProto__operator_delete(ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual void TypeProto__CopyFrom(ONNX_NAMESPACE::TypeProto* p, const ONNX_NAMESPACE::TypeProto* other) = 0;
  virtual bool TypeProto__has_tensor_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TypeProto_Tensor& TypeProto__tensor_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_Tensor* TypeProto__mutable_tensor_type(ONNX_NAMESPACE::TypeProto* p) = 0;

#if !defined(DISABLE_SPARSE_TENSORS)
  virtual const ONNX_NAMESPACE::TypeProto_SparseTensor& TypeProto__sparse_tensor_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_SparseTensor* TypeProto__mutable_sparse_tensor_type(ONNX_NAMESPACE::TypeProto* p) = 0;
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  virtual const ONNX_NAMESPACE::TypeProto_Optional& TypeProto__optional_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_Optional* TypeProto__mutable_optional_type(ONNX_NAMESPACE::TypeProto* p) = 0;
#endif

  virtual const ONNX_NAMESPACE::TypeProto_Sequence& TypeProto__sequence_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_Sequence* TypeProto__mutable_sequence_type(ONNX_NAMESPACE::TypeProto* p) = 0;

  virtual int TypeProto__value_case(const ONNX_NAMESPACE::TypeProto* p) = 0;

  // AttributeProto
  virtual std::unique_ptr<ONNX_NAMESPACE::AttributeProto> AttributeProto__construct() = 0;
  virtual void AttributeProto__operator_delete(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__operator_assign(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::AttributeProto& v) = 0;

  virtual const std::string& AttributeProto__name(const ONNX_NAMESPACE::AttributeProto* p) const = 0;
  virtual ONNX_NAMESPACE::AttributeProto_AttributeType AttributeProto__type(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__ints_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__floats_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__strings_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int64_t AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual float AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual const ::std::string& AttributeProto__strings(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual const ONNX_NAMESPACE::int64s& AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual const ONNX_NAMESPACE::float32s& AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual ONNX_NAMESPACE::int64s* AttributeProto__mutable_ints(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual ONNX_NAMESPACE::float32s* AttributeProto__mutable_floats(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__add_ints(ONNX_NAMESPACE::AttributeProto* p, int64_t size) = 0;
  virtual void AttributeProto__add_floats(ONNX_NAMESPACE::AttributeProto* p, float size) = 0;
  virtual void AttributeProto__add_strings(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& size) = 0;
  virtual int64_t AttributeProto__i(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual float AttributeProto__f(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TensorProto& AttributeProto__t(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, ::std::string&& value) = 0;
  virtual void AttributeProto__set_f(ONNX_NAMESPACE::AttributeProto* p, const float& value) = 0;
  virtual void AttributeProto__set_i(ONNX_NAMESPACE::AttributeProto* p, int64_t value) = 0;
  virtual void AttributeProto__set_t(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::TensorProto& tensor) = 0;
  virtual const ::std::string& AttributeProto__s(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_name(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual void AttributeProto__set_type(ONNX_NAMESPACE::AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) = 0;
  virtual ONNX_NAMESPACE::TensorProto* AttributeProto__add_tensors(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual std::string* AttributeProto__release_s(ONNX_NAMESPACE::AttributeProto* p) = 0;

  // GraphProto
  virtual std::unique_ptr<ONNX_NAMESPACE::GraphProto> GraphProto__construct() = 0;
  virtual void GraphProto__operator_delete(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::GraphProto& GraphProto__operator_assign(ONNX_NAMESPACE::GraphProto* p, const ONNX_NAMESPACE::GraphProto& v) = 0;
  virtual ONNX_NAMESPACE::GraphProto& GraphProto__operator_move_assign(ONNX_NAMESPACE::GraphProto* p, ONNX_NAMESPACE::GraphProto&& v) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__input(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProto* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual int GraphProto__input_size(const ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProtos& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_output(ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_value_info(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorProtos* GraphProto__mutable_initializer(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorProto* GraphProto__add_initializer(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::NodeProto* GraphProto__add_node(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual std::string* GraphProto__mutable_name(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::NodeProto* GraphProto__mutable_node(ONNX_NAMESPACE::GraphProto* p, int index) = 0;

  virtual void GraphProto__set_name(ONNX_NAMESPACE::GraphProto* p, const std::string& name) = 0;
  virtual void GraphProto__set_doc_string(ONNX_NAMESPACE::GraphProto* p, const std::string& doc_str) = 0;

  // ModelProto
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> ModelProto__construct() = 0;
  virtual void ModelProto__operator_delete(ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual bool ModelProto__SerializeToString(const ONNX_NAMESPACE::ModelProto* p, std::string& string) = 0;
  virtual bool ModelProto__SerializeToOstream(const ONNX_NAMESPACE::ModelProto* p, std::ostream& output) = 0;
  virtual bool ModelProto__ParseFromString(ONNX_NAMESPACE::ModelProto* p, const std::string& data) = 0;
  virtual std::string ModelProto__SerializeAsString(const ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual const ONNX_NAMESPACE::GraphProto& ModelProto__graph(const ONNX_NAMESPACE::ModelProto* p) = 0;
  virtual ONNX_NAMESPACE::GraphProto* ModelProto__mutable_graph(ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual void ModelProto__set_ir_version(ONNX_NAMESPACE::ModelProto* p, int64_t value) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProtos* ModelProto__mutable_metadata_props(ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual const ONNX_NAMESPACE::OperatorSetIdProto& ModelProto__opset_import(const ONNX_NAMESPACE::ModelProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::OperatorSetIdProto* ModelProto__mutable_opset_import(ONNX_NAMESPACE::ModelProto* p, int index) = 0;
  virtual int ModelProto__opset_import_size(const ONNX_NAMESPACE::ModelProto* p) = 0;
  virtual ONNX_NAMESPACE::OperatorSetIdProto* ModelProto__add_opset_import(ONNX_NAMESPACE::ModelProto* p) = 0;

  // NodeProto
  virtual std::unique_ptr<ONNX_NAMESPACE::NodeProto> NodeProto__construct() = 0;
  virtual void NodeProto__operator_delete(ONNX_NAMESPACE::NodeProto* p) = 0;
  virtual void NodeProto__operator_assign(ONNX_NAMESPACE::NodeProto* p, const ONNX_NAMESPACE::NodeProto& v) = 0;
  virtual int NodeProto__attribute_size(ONNX_NAMESPACE::NodeProto* p) = 0;
  virtual const ONNX_NAMESPACE::AttributeProto& NodeProto__attribute(const ONNX_NAMESPACE::NodeProto* p, int index) const = 0;
  virtual ONNX_NAMESPACE::AttributeProto* NodeProto__mutable_attribute(ONNX_NAMESPACE::NodeProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::AttributeProto* NodeProto__add_attribute(ONNX_NAMESPACE::NodeProto* p) = 0;

  // TensorProto
  virtual std::unique_ptr<ONNX_NAMESPACE::TensorProto> TensorProto__construct() = 0;
  virtual void TensorProto__operator_delete(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorProto& TensorProto__operator_assign(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto& v) = 0;
  virtual ONNX_NAMESPACE::TensorProto& TensorProto__operator_move_assign(ONNX_NAMESPACE::TensorProto* p, ONNX_NAMESPACE::TensorProto&& v) = 0;
  virtual bool TensorProto__has_name(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__set_name(ONNX_NAMESPACE::TensorProto* p, const ::std::string& name) = 0;
  virtual const ::std::string& TensorProto__name(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int TensorProto__dims_size(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual const ONNX_NAMESPACE::int64s& TensorProto__dims(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__add_dims(ONNX_NAMESPACE::TensorProto* p, int64_t value) = 0;
  virtual bool TensorProto__has_data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int TensorProto__data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual bool TensorProto__has_raw_data(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual const std::string& TensorProto__raw_data(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual std::string* TensorProto__mutable_raw_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual bool TensorProto__has_data_type(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int32_t TensorProto__data_type(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__set_data_type(ONNX_NAMESPACE::TensorProto* p, int32_t type) = 0;
  virtual void TensorProto__CopyFrom(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto* other) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProtos* TensorProto__mutable_external_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_float_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_int32_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_string_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_int64_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_double_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__clear_uint64_data(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__set_data_location(ONNX_NAMESPACE::TensorProto* p, ONNX_NAMESPACE::TensorProto_DataLocation data_location) = 0;

  virtual bool TensorProto_DataType_IsValid(int value) = 0;

  // TensorProtos
  virtual ONNX_NAMESPACE::TensorProto* TensorProtos__Add(ONNX_NAMESPACE::TensorProtos* p) = 0;
  virtual int TensorProtos__size(const ONNX_NAMESPACE::TensorProtos* p) = 0;
  virtual ONNX_NAMESPACE::TensorProto& TensorProtos__at(ONNX_NAMESPACE::TensorProtos* p, int index) = 0;
  virtual std::unique_ptr<TensorProto_ConstIterator> TensorProtos__begin(const ONNX_NAMESPACE::TensorProtos* p) = 0;
  virtual std::unique_ptr<TensorProto_ConstIterator> TensorProtos__end(const ONNX_NAMESPACE::TensorProtos* p) = 0;
  virtual std::unique_ptr<TensorProto_Iterator> TensorProtos__begin(ONNX_NAMESPACE::TensorProtos* p) = 0;
  virtual std::unique_ptr<TensorProto_Iterator> TensorProtos__end(ONNX_NAMESPACE::TensorProtos* p) = 0;

  // TensorShapeProto_Dimension
  virtual int TensorShapeProto_Dimension__value_case(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual const std::string& TensorShapeProto_Dimension__dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual int64_t TensorShapeProto_Dimension__dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__set_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, int64_t value) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__clear_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual const std::string& TensorShapeProto_Dimension__denotation(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) const = 0;
  virtual void TensorShapeProto_Dimension__set_denotation(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, const std::string& value) = 0;

  // TensorShapeProto_Dimensions
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__begin(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__end(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;
  virtual size_t TensorShapeProto_Dimensions__size(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;

  // TensorShapeProto
  virtual int TensorShapeProto__dim_size(const ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimensions& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimension& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__mutable_dim(ONNX_NAMESPACE::TensorShapeProto* p, int index) = 0;
  virtual void TensorShapeProto__clear_dim(ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__add_dim(ONNX_NAMESPACE::TensorShapeProto* p) = 0;

  // ValueInfoProto
  virtual void ValueInfoProto__operator_assign(ONNX_NAMESPACE::ValueInfoProto* p, const ONNX_NAMESPACE::ValueInfoProto& v) = 0;
  virtual const ONNX_NAMESPACE::TypeProto& ValueInfoProto__type(const ONNX_NAMESPACE::ValueInfoProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto* ValueInfoProto__mutable_type(ONNX_NAMESPACE::ValueInfoProto* p) = 0;

  // ValueInfoProtos
  virtual ONNX_NAMESPACE::ValueInfoProto* ValueInfoProtos__Add(ONNX_NAMESPACE::ValueInfoProtos* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& ValueInfoProtos__operator_array(const ONNX_NAMESPACE::ValueInfoProtos* p, int index) = 0;

  // FunctionProto
  virtual std::unique_ptr<ONNX_NAMESPACE::FunctionProto> FunctionProto__construct() = 0;
  virtual void FunctionProto__operator_delete(ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual bool FunctionProto__SerializeToString(const ONNX_NAMESPACE::FunctionProto* p, std::string& string) = 0;
  virtual bool FunctionProto__SerializeToOstream(const ONNX_NAMESPACE::FunctionProto* p, std::ostream& output) = 0;
  virtual bool FunctionProto__ParseFromString(ONNX_NAMESPACE::FunctionProto* p, const std::string& data) = 0;
  virtual std::string FunctionProto__SerializeAsString(const ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual bool FunctionProto__has_name(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual const std::string& FunctionProto__name(const ONNX_NAMESPACE::FunctionProto* p) const = 0;
  virtual void FunctionProto__set_name(ONNX_NAMESPACE::FunctionProto* p, const ::std::string& name) = 0;

  virtual bool FunctionProto__has_doc_string(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual const std::string& FunctionProto__doc_string(const ONNX_NAMESPACE::FunctionProto* p) const = 0;
  virtual void FunctionProto__set_doc_string(ONNX_NAMESPACE::FunctionProto* p, const ::std::string& doc_string) = 0;

  virtual bool FunctionProto__has_domain(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual const std::string& FunctionProto__domain(const ONNX_NAMESPACE::FunctionProto* p) const = 0;
  virtual void FunctionProto__set_domain(ONNX_NAMESPACE::FunctionProto* p, const ::std::string& domain) = 0;

  virtual const std::string& FunctionProto__input(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual std::string* FunctionProto__mutable_input(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__input_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual void FunctionProto__add_input(ONNX_NAMESPACE::FunctionProto* p, const std::string& value) = 0;

  virtual const std::string& FunctionProto__output(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual std::string* FunctionProto__mutable_output(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__output_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual void FunctionProto__add_output(ONNX_NAMESPACE::FunctionProto* p, const std::string& value) = 0;

  virtual const std::string& FunctionProto__attribute(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual std::string* FunctionProto__mutable_attribute(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__attribute_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual void FunctionProto__add_attribute(ONNX_NAMESPACE::FunctionProto* p, const std::string& value) = 0;

  virtual const ONNX_NAMESPACE::AttributeProto& FunctionProto__attribute_proto(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::AttributeProto* FunctionProto__mutable_attribute_proto(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__attribute_proto_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::AttributeProto* FunctionProto__add_attribute_proto(ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual const ONNX_NAMESPACE::NodeProto& FunctionProto__node(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::NodeProto* FunctionProto__mutable_node(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__node_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::NodeProto* FunctionProto__add_node(ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& FunctionProto__value_info(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* FunctionProto__mutable_value_info(ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProto* FunctionProto__mutable_value_info(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__value_info_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProto* FunctionProto__add_value_info(ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual const ONNX_NAMESPACE::StringStringEntryProto& FunctionProto__metadata_props(const ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProtos* FunctionProto__mutable_metadata_props(ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProto* FunctionProto__mutable_metadata_props(ONNX_NAMESPACE::FunctionProto* p, int index) = 0;
  virtual int FunctionProto__metadata_props_size(const ONNX_NAMESPACE::FunctionProto* p) = 0;
  virtual ONNX_NAMESPACE::StringStringEntryProto* FunctionProto__add_metadata_props(ONNX_NAMESPACE::FunctionProto* p) = 0;

  virtual void InferShapes(const std::string& m, const std::string& save_path) = 0;
  virtual void InferShapes(ONNX_NAMESPACE::ModelProto& m) = 0;
  virtual void RegisterSchema(const std::string& domain, const OrtCustomOp* op) = 0;
  virtual void DeregisterSchema(const std::string& domain, const std::string& op_type, int version) = 0;
  virtual const ONNX_NAMESPACE::OpSchema* GetSchema(const std::string& name, const int maxInclusiveVersion, const std::string& domain) = 0;
  virtual const std::string& OpSchema__inputs__GetName(const ONNX_NAMESPACE::OpSchema* p, const size_t i) = 0;
  virtual const std::string& OpSchema__inputs__GetTypeStr(const ONNX_NAMESPACE::OpSchema* p, const size_t i) = 0;
  virtual const std::string& OpSchema__outputs__GetName(const ONNX_NAMESPACE::OpSchema* p, const size_t i) = 0;
  virtual const std::string& OpSchema__outputs__GetTypeStr(const ONNX_NAMESPACE::OpSchema* p, const size_t i) = 0;
  virtual const ONNX_NAMESPACE::TypeConstraintMap& OpSchema__typeConstraintMap(const ONNX_NAMESPACE::OpSchema* p) const = 0;

  // ConfigOptions
  virtual std::optional<std::string> ConfigOptions__GetConfigEntry(const ConfigOptions* p, const std::string& config_key) = 0;
  virtual std::string ConfigOptions__GetConfigOrDefault(const ConfigOptions* p, const std::string& config_key,
                                                        const std::string& default_value) = 0;
  virtual const std::unordered_map<std::string, std::string>& ConfigOptions__GetConfigOptionsMap(const ConfigOptions* p) = 0;

  // OrtRunOptions
  virtual const ConfigOptions& RunOptions__GetConfigOptions(const RunOptions* p) = 0;
  // OrtSessionOptions
  virtual const std::unordered_map<std::string, std::string>& SessionOptions__GetConfigOptionsMap(const OrtSessionOptions* p) = 0;
  virtual const ConfigOptions& SessionOptions__GetConfigOptions(const OrtSessionOptions* p) = 0;
  virtual bool SessionOptions__GetEnableProfiling(const OrtSessionOptions* p) = 0;
  // ComputeCapability
  virtual std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) = 0;
  virtual void ComputeCapability__operator_delete(ComputeCapability* p) = 0;
  virtual std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) = 0;
  virtual void ComputeCapability__copy_optimization_func(ComputeCapability* p, ComputeCapability* selection_cc) = 0;
  virtual void ComputeCapability__add_nodes_to_optimize(ComputeCapability* p, std::unique_ptr<ComputeCapability> optimization_cc) = 0;

  // DataTransferManager
  virtual Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst) = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual Status DataTransferManager__CopySparseTensor(const DataTransferManager* p, const SparseTensor& src, SparseTensor& dst) = 0;
  virtual Status DataTransferManager__CopySparseTensors(const DataTransferManager* p, const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) = 0;
#endif
  virtual const IDataTransfer* DataTransferManager__GetDataTransfer(const DataTransferManager* p, const OrtDevice& src_device, const OrtDevice& dst_device) = 0;

  // IDataTransfer
  virtual Status IDataTransfer__CopyTensor(const IDataTransfer* p, const Tensor& src, Tensor& dst) = 0;
  virtual Status IDataTransfer__CopyTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual Status IDataTransfer__CopySparseTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) = 0;
#endif

  // IndexedSubGraph_MetaDef
  virtual std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() = 0;
  virtual void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) = 0;

  virtual std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) = 0;
  virtual int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) = 0;
  virtual ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__constant_initializers(IndexedSubGraph_MetaDef* p) = 0;
  virtual NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) = 0;

  // IndexedSubGraph
  virtual std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() = 0;
  virtual void IndexedSubGraph__operator_delete(IndexedSubGraph* p) = 0;

  virtual const std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(const IndexedSubGraph* p) = 0;
  virtual std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) = 0;

  virtual void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) = 0;
  virtual const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) = 0;

  virtual void IndexedSubGraph__SetSchemaSource(IndexedSubGraph* p, IndexedSubGraph_SourceOfSchema schema_source) = 0;
  virtual IndexedSubGraph_SourceOfSchema IndexedSubGraph__GetSchemaSource(const IndexedSubGraph* p) = 0;
  virtual void IndexedSubGraph__SetAccountant(IndexedSubGraph* p, IResourceAccountant*) = 0;
  virtual void IndexedSubGraph__AppendNodeCost(IndexedSubGraph* p, const ResourceCount& count) = 0;

  // KernelDef
  virtual void KernelDef__operator_delete(KernelDef* p) = 0;
  virtual int KernelDef__ExecQueueId(const KernelDef* p) = 0;
  virtual void KernelDef__SinceVersion(const KernelDef* p, int* start, int* end) = 0;
  virtual const std::string& KernelDef__Domain(const KernelDef* p) = 0;
  virtual const std::string& KernelDef__OpName(const KernelDef* p) = 0;

  // KernelDefBuilder
  virtual std::unique_ptr<KernelDefBuilder> KernelDefBuilder__construct() = 0;
  virtual void KernelDefBuilder__operator_delete(KernelDefBuilder* p) = 0;

  virtual void KernelDefBuilder__SetName(KernelDefBuilder* p, const char* op_name) = 0;
  virtual void KernelDefBuilder__SetDomain(KernelDefBuilder* p, const char* domain) = 0;
  virtual void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version) = 0;
  virtual void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version_start, int since_version_end) = 0;
  virtual void KernelDefBuilder__Provider(KernelDefBuilder* p, const char* provider_type) = 0;
  virtual void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) = 0;
  virtual void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) = 0;
  virtual void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, const std::vector<int>& input_indexes) = 0;
  virtual void KernelDefBuilder__OutputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void KernelDefBuilder__ExecQueueId(KernelDefBuilder* p, int queue_id) = 0;
  virtual void KernelDefBuilder__MayInplace(KernelDefBuilder* p, int input_index, int output_index) = 0;
  virtual void KernelDefBuilder__Alias(KernelDefBuilder* p, int input_index, int output_index) = 0;
  virtual void KernelDefBuilder__Alias(KernelDefBuilder* p, const std::vector<std::pair<int, int>>& aliases) = 0;
  virtual void KernelDefBuilder__VariadicAlias(KernelDefBuilder* p, int input_offset, int output_offset) = 0;
  virtual void KernelDefBuilder__ExternalOutputs(KernelDefBuilder* p) = 0;
  virtual void KernelDefBuilder__AllocateInputsContiguously(KernelDefBuilder* p) = 0;
#ifdef ENABLE_STRIDED_TENSORS
  virtual void KernelDefBuilder__MayStridedInput(KernelDefBuilder* p, int input_index) = 0;
  virtual void KernelDefBuilder__MayStridedOutput(KernelDefBuilder* p, int input_index, int output_index) = 0;
#endif

  virtual std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) = 0;

  // KernelRegistry
  virtual std::shared_ptr<KernelRegistry> KernelRegistry__construct() = 0;
  virtual void KernelRegistry__operator_delete(KernelRegistry* p) = 0;
  virtual Status KernelRegistry__Register(KernelRegistry* p, KernelCreateInfo&& create_info) = 0;

  // PrimitiveDataTypeBase
  virtual int32_t PrimitiveDataTypeBase__GetDataType(const PrimitiveDataTypeBase* p) = 0;
  virtual int32_t PrimitiveDataTypeBase__GetNumSubElems(const PrimitiveDataTypeBase* p) = 0;
  virtual bool PrimitiveDataTypeBase__HasSubElems(const PrimitiveDataTypeBase* p) = 0;

  // DataTypeImpl
  virtual MLDataType DataTypeImpl__GetType_Tensor() = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual MLDataType DataTypeImpl__GetType_SparseTensor() = 0;
#endif
  virtual MLDataType DataTypeImpl__GetType_TensorSeq() = 0;
  virtual MLDataType DataTypeImpl__GetTypeFromOnnxType(int) = 0;
  virtual MLDataType DataTypeImpl__GetType_bool() = 0;
  virtual MLDataType DataTypeImpl__GetType_int8() = 0;
  virtual MLDataType DataTypeImpl__GetType_uint8() = 0;
  virtual MLDataType DataTypeImpl__GetType_int16() = 0;
  virtual MLDataType DataTypeImpl__GetType_uint16() = 0;
  virtual MLDataType DataTypeImpl__GetType_int32() = 0;
  virtual MLDataType DataTypeImpl__GetType_uint32() = 0;
  virtual MLDataType DataTypeImpl__GetType_int64() = 0;
  virtual MLDataType DataTypeImpl__GetType_uint64() = 0;
  virtual MLDataType DataTypeImpl__GetType_float() = 0;
  virtual MLDataType DataTypeImpl__GetType_double() = 0;
  virtual MLDataType DataTypeImpl__GetType_BFloat16() = 0;
  virtual MLDataType DataTypeImpl__GetType_MLFloat16() = 0;
  virtual MLDataType DataTypeImpl__GetType_string() = 0;
#if !defined(DISABLE_FLOAT8_TYPES)
  virtual MLDataType DataTypeImpl__GetType_Float8E4M3FN() = 0;
  virtual MLDataType DataTypeImpl__GetType_Float8E4M3FNUZ() = 0;
  virtual MLDataType DataTypeImpl__GetType_Float8E5M2() = 0;
  virtual MLDataType DataTypeImpl__GetType_Float8E5M2FNUZ() = 0;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  virtual MLDataType DataTypeImpl__GetType_Float4E2M1x2() = 0;
#endif
  virtual MLDataType DataTypeImpl__GetType_Int4x2() = 0;
  virtual MLDataType DataTypeImpl__GetType_UInt4x2() = 0;
  virtual MLDataType DataTypeImpl__GetType_Int2x4() = 0;
  virtual MLDataType DataTypeImpl__GetType_UInt2x4() = 0;

  virtual MLDataType DataTypeImpl__GetTensorTypeFromOnnxType(int) = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_bool() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_int8() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_uint8() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_int16() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_uint16() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_int32() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_uint32() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_int64() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_uint64() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_float() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_double() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_BFloat16() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_MLFloat16() = 0;
#if !defined(DISABLE_FLOAT8_TYPES)
  virtual MLDataType DataTypeImpl__GetTensorType_Float8E4M3FN() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float8E4M3FNUZ() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float8E5M2() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float8E5M2FNUZ() = 0;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  virtual MLDataType DataTypeImpl__GetTensorType_Float4E2M1x2() = 0;
#endif

  virtual MLDataType DataTypeImpl__GetTensorType_Int4x2() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_UInt4x2() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Int2x4() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_UInt2x4() = 0;

#if !defined(DISABLE_SPARSE_TENSORS)
  virtual MLDataType DataTypeImpl__GetSparseTensorType_bool() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_int8() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_uint8() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_int16() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_uint16() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_int32() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_uint32() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_int64() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_uint64() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_float() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_double() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_string() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_BFloat16() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_MLFloat16() = 0;
#if !defined(DISABLE_FLOAT8_TYPES)
  virtual MLDataType DataTypeImpl__GetSparseTensorType_Float8E4M3FN() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_Float8E4M3FNUZ() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_Float8E5M2() = 0;
  virtual MLDataType DataTypeImpl__GetSparseTensorType_Float8E5M2FNUZ() = 0;
#endif
#endif

  virtual const char* DataTypeImpl__ToString(MLDataType type) = 0;
  virtual bool DataTypeImpl__IsTensorType(const DataTypeImpl* p) = 0;
  virtual bool DataTypeImpl__IsTensorSequenceType(const DataTypeImpl* p) = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual bool DataTypeImpl__IsSparseTensorType(const DataTypeImpl* p) = 0;
#endif

  virtual DeleteFunc DataTypeImpl__GetDeleteFunc(const DataTypeImpl* p) = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypesIRv9() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv9() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv10() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv11() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllIEEEFloatTensorTypes() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypesIRv9() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv9() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv9() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypesIRv9() = 0;

  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv4() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv9() = 0;

  virtual size_t DataTypeImpl__Size(const DataTypeImpl* p) = 0;
  virtual const PrimitiveDataTypeBase* DataTypeImpl__AsPrimitiveDataType(const DataTypeImpl* p) = 0;

  // Function
  virtual const Graph& Function__Body(const Function* p) = 0;

  // Node
  virtual const std::string& Node__Name(const Node* p) noexcept = 0;
  virtual const std::string& Node__Description(const Node* p) noexcept = 0;
  virtual const std::string& Node__Domain(const Node* p) noexcept = 0;
  virtual const std::string& Node__OpType(const Node* p) noexcept = 0;
  virtual int Node__SinceVersion(const Node* p) = 0;

  virtual const Function* Node__GetFunctionBody(const Node* p) noexcept = 0;
  virtual ProviderType Node__GetExecutionProviderType(const Node* p) const noexcept = 0;

  virtual const std::vector<int>& Node__InputArgCount(const Node* p) = 0;
  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__ImplicitInputDefs(const Node* p) noexcept = 0;
  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__InputDefs(const Node* p) noexcept = 0;
  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__OutputDefs(const Node* p) noexcept = 0;
  virtual NodeIndex Node__Index(const Node* p) noexcept = 0;
  virtual std::vector<gsl::not_null<const Graph*>> Node__GetSubgraphs(const Node* p) const noexcept = 0;

  virtual void Node__ToProto(const Node* p, ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) = 0;

  virtual const NodeAttributes& Node__GetAttributes(const Node* p) noexcept = 0;
  virtual void Node__AddAttribute(Node* p, const ::std::string& attr_name, const ONNX_NAMESPACE::GraphProto& value) = 0;
  virtual void Node__AddAttribute(Node* p, const ::std::string& attr_name, const std::string& value) = 0;
  virtual void Node__AddAttribute(Node* p, const ::std::string& attr_name, int64_t value) = 0;
  virtual size_t Node__GetInputEdgesCount(const Node* p) noexcept = 0;
  virtual size_t Node__GetOutputEdgesCount(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__EdgeIterator> Node__InputEdgesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__EdgeIterator> Node__InputEdgesEnd(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept = 0;

  virtual void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) = 0;
  virtual int Node__NodeType(const Node* p) const noexcept = 0;
  virtual const std::unordered_map<std::string, gsl::not_null<Graph*>>& Node__GetAttributeNameToMutableSubgraphMap(Node* p) = 0;
  virtual std::unordered_map<std::string, gsl::not_null<const Graph*>> Node__GetAttributeNameToSubgraphMap(const Node* p) const = 0;

  // Node_EdgeEnd
  virtual const Node& Node_EdgeEnd__GetNode(const Node_EdgeEnd* p) = 0;
  virtual int Node_EdgeEnd__GetSrcArgIndex(const Node_EdgeEnd* p) = 0;
  virtual int Node_EdgeEnd__GetDstArgIndex(const Node_EdgeEnd* p) = 0;

  // NodeArg
  virtual const std::string& NodeArg__Name(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg__Shape(const NodeArg* p) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept = 0;
  virtual bool NodeArg__Exists(const NodeArg* p) const noexcept = 0;
  virtual const ONNX_NAMESPACE::TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept = 0;
  virtual Status NodeArg__OverrideTypesHelper(NodeArg* p, const ONNX_NAMESPACE::TypeProto& input_type, int32_t input_tensor_elem_type, int32_t current_tensor_elem_type, bool override_types) = 0;

  // NodeAttributes
  virtual std::unique_ptr<NodeAttributes> NodeAttributes__construct() = 0;
  virtual void NodeAttributes__operator_delete(NodeAttributes* p) noexcept = 0;
  virtual void NodeAttributes__operator_assign(NodeAttributes* p, const NodeAttributes& v) = 0;

  virtual size_t NodeAttributes__size(const NodeAttributes* p) = 0;
  virtual void NodeAttributes__clear(NodeAttributes* p) noexcept = 0;
  virtual size_t NodeAttributes__count(const NodeAttributes* p, const std::string& keyval) = 0;
  virtual ONNX_NAMESPACE::AttributeProto& NodeAttributes__operator_array(NodeAttributes* p, const std::string& string) = 0;
  virtual const ONNX_NAMESPACE::AttributeProto& NodeAttributes__at(const NodeAttributes* p, const std::string& string) = 0;

  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__begin(const NodeAttributes* p) = 0;
  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__end(const NodeAttributes* p) = 0;
  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__find(const NodeAttributes* p, const std::string& key) = 0;
  virtual void NodeAttributes__insert(NodeAttributes* p, const NodeAttributes& v) = 0;
  virtual void NodeAttributes__emplace(NodeAttributes* p, const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) = 0;
  virtual void NodeAttributes__emplace(NodeAttributes* p, const std::string& k, ONNX_NAMESPACE::AttributeProto&& v) = 0;
  virtual void NodeAttributes__insert_or_assign(NodeAttributes* p, const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) = 0;
  virtual void NodeAttributes__reserve(NodeAttributes* p, size_t size) = 0;

  // NodeUnit
  virtual void NodeUnit__operator_delete(NodeUnit* p) noexcept = 0;

  virtual int NodeUnit__UnitType(const NodeUnit* p) noexcept = 0;

  virtual const std::vector<NodeUnitIODef>& NodeUnit__Inputs(const NodeUnit* p) noexcept = 0;
  virtual const std::vector<NodeUnitIODef>& NodeUnit__Outputs(const NodeUnit* p) noexcept = 0;

  virtual const std::string& NodeUnit__Domain(const NodeUnit* p) noexcept = 0;
  virtual const std::string& NodeUnit__OpType(const NodeUnit* p) noexcept = 0;
  virtual const std::string& NodeUnit__Name(const NodeUnit* p) noexcept = 0;
  virtual int NodeUnit__SinceVersion(const NodeUnit* p) noexcept = 0;
  virtual NodeIndex NodeUnit__Index(const NodeUnit* p) noexcept = 0;
  virtual const std::filesystem::path& NodeUnit__ModelPath(const NodeUnit* p) noexcept = 0;
  virtual ProviderType NodeUnit__GetExecutionProviderType(const NodeUnit* p) noexcept = 0;

  virtual const Node& NodeUnit__GetNode(const NodeUnit* p) noexcept = 0;
  virtual const std::vector<const Node*>& NodeUnit__GetDQNodes(const NodeUnit* p) noexcept = 0;
  virtual const std::vector<const Node*>& NodeUnit__GetQNodes(const NodeUnit* p) noexcept = 0;
  virtual std::vector<const Node*> NodeUnit__GetAllNodesInGroup(const NodeUnit* p) noexcept = 0;

  virtual size_t NodeUnit__InputEdgeCount(const NodeUnit* p) = 0;
  virtual std::unique_ptr<Node__EdgeIterator> NodeUnit__OutputEdgesBegin(const NodeUnit* p) = 0;
  virtual std::unique_ptr<Node__EdgeIterator> NodeUnit__OutputEdgesEnd(const NodeUnit* p) = 0;

  virtual std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
  QDQ__GetAllNodeUnits(const GraphViewer* graph_viewer, const logging::Logger& logger) = 0;

  // Partitioning utils
  virtual std::vector<std::unique_ptr<ComputeCapability>>
  Utils__CreateSupportedPartitions(const GraphViewer& graph_viewer,
                                   const std::unordered_set<const Node*>& supported_nodes,
                                   const std::unordered_set<std::string>& stop_ops,
                                   const std::function<std::string()>& generate_metadef_name,
                                   const std::string& execution_provider_name,
                                   const std::string& execution_provider_type,
                                   const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
                                   bool drop_constant_initializers) = 0;

  virtual std::unique_ptr<ComputeCapability>
  Utils__MakeComputeCapability(const GraphViewer& graph_viewer,
                               const std::vector<const Node*>& group,
                               const std::function<std::string()>& generate_metadef_name,
                               const std::string& execution_provider_name,
                               bool drop_constant_initializers) = 0;

  virtual Status Utils__GetTensorProtoWithDataIfInMemory(
      const ONNX_NAMESPACE::TensorProto& tensor_proto, std::unique_ptr<ONNX_NAMESPACE::TensorProto>& result) = 0;

  virtual bool Utils__HasExternalDataInMemory(const ONNX_NAMESPACE::TensorProto& ten_proto) = 0;

  virtual Status Utils__ValidateExternalDataPath(const std::filesystem::path& model_path,
                                                 const std::filesystem::path& external_data_path) = 0;
  virtual Status Utils__ValidateExternalDataPathFromDir(const std::filesystem::path& model_dir,
                                                        const std::filesystem::path& external_data_path) = 0;

  // Model
  virtual std::unique_ptr<Model> Model__construct(ONNX_NAMESPACE::ModelProto&& model_proto, const PathString& model_path,
                                                  const IOnnxRuntimeOpSchemaRegistryList* local_registries,
                                                  const logging::Logger& logger) = 0;
  virtual std::unique_ptr<Model> Model__construct(const std::string& graph_name, bool is_onnx_domain_only,
                                                  const logging::Logger& logger) = 0;
  virtual void Model__operator_delete(Model* p) = 0;
  virtual Graph& Model__MainGraph(Model* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToProto(Model* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToGraphProtoWithExternalInitializers(
      Model* p,
      const std::filesystem::path& external_file_name,
      const std::filesystem::path& file_path,
      const ModelSavingOptions&) = 0;
  virtual const ModelMetaData& Model__MetaData(const Model* p) const noexcept = 0;
  virtual Status Model__Load(const PathString& file_path, /*out*/ ONNX_NAMESPACE::ModelProto& model_proto) = 0;

  // Graph
  virtual std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::GraphProto> Graph__ToGraphProto(const Graph* p) = 0;

  virtual NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) = 0;
  virtual void Graph__AddOuterScopeNodeArg(Graph* p, const std::string& name) = 0;
  virtual void Graph__SetInputs(Graph* p, gsl::span<const NodeArg* const> inputs) = 0;
  virtual const std::unordered_map<std::string, int>& Graph__DomainToVersionMap(const Graph* p) const noexcept = 0;
  virtual Status Graph__Resolve(Graph* p) = 0;
  virtual void Graph__AddInitializedTensor(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor) = 0;
  // We pass OrtValue by reference here (as opposed to the original Graph function) to avoid header inclusion
  virtual Status Graph__AddInitializedOrtValue(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor, const OrtValue& value) = 0;
  virtual bool Graph__GetOrtValueInitializer(const Graph* p, const std::string& tensor_name, OrtValue& value,
                                             bool check_outer_scope) = 0;
  virtual Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const gsl::span<NodeArg* const>& input_args, const gsl::span<NodeArg* const>& output_args, const NodeAttributes* attributes, const std::string& domain) = 0;
  virtual Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const gsl::span<NodeArg* const>& input_args, const gsl::span<NodeArg* const>& output_args, NodeAttributes&& attributes, const std::string& domain) = 0;
  virtual Node& Graph__AddNode(Graph* p, const Node& other) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept = 0;
  virtual void Graph__SetOutputs(Graph* p, gsl::span<const NodeArg* const> outputs) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept = 0;
  virtual std::vector<const Node*> Graph__Nodes(const Graph* p) = 0;
  virtual bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;

  virtual const Node* Graph__ParentNode(const Graph* p) const = 0;
  virtual const Graph* Graph__ParentGraph(const Graph* p) const = 0;
  virtual Graph* Graph__MutableParentGraph(Graph* p) = 0;
  virtual const std::string& Graph__Name(const Graph* p) const noexcept = 0;
  virtual void Graph__SetName(Graph* p, const std::string& name) const noexcept = 0;
  virtual const std::filesystem::path& Graph__ModelPath(const Graph* p) const = 0;
  virtual const std::vector<const NodeArg*>& Graph__GetInputsIncludingInitializers(const Graph* p) const noexcept = 0;
  virtual bool Graph__IsSubgraph(const Graph* p) = 0;
  virtual const Node* Graph__GetProducerNode(const Graph* p, const std::string& node_arg_name) const = 0;
  virtual const Model& Graph__GetModel(const Graph* p) = 0;
  virtual void Graph__ReverseDFSFrom(const Graph* p, gsl::span<const Node* const> from,
                                     const std::function<void(const Node*)>& enter,
                                     const std::function<void(const Node*)>& leave,
                                     const std::function<bool(const Node*, const Node*)>& comp,
                                     const std::function<bool(const Node* from, const Node* to)>& stop) const = 0;
  virtual Graph& Graph__SetGraphResolveNeeded(Graph* p) = 0;
  virtual void Graph__RemoveInitializedTensor(Graph* p, const std::string& tensor_name) = 0;

  virtual std::vector<const Node*> Graph__GetConsumerNodes(const Graph* p, const std::string& node_arg_name) const = 0;
  virtual void Graph__AddEdge(Graph* p, NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index,
                              int dst_arg_index) = 0;
  virtual void Graph__RemoveEdge(Graph* p, NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_index,
                                 int dst_arg_index) = 0;
  virtual void Graph__RemoveNode(Graph* p, NodeIndex index) = 0;
  virtual Node& Graph__FuseSubGraph(Graph* p, const IndexedSubGraph& sub_graph, const std::string& fused_node_name) = 0;
  virtual void Graph__UpdateProducerNode(Graph* p, const std::string& node_arg_name, NodeIndex node_index) = 0;
  virtual const ONNX_NAMESPACE::TensorProto* Graph__GetConstantInitializer(const Graph* p, const std::string& name, bool check_outer_scope) const = 0;
  virtual const InitializedTensorSet& Graph__GetAllInitializedTensors(const Graph* p) = 0;
  virtual int Graph__MaxNodeIndex(const Graph* p) const noexcept = 0;
  virtual Node* Graph__GetNode(Graph* p, NodeIndex node_index) noexcept = 0;
  virtual const Node* Graph__GetNode(const Graph* p, NodeIndex node_index) const = 0;
  virtual const NodeArg* Graph__GetNodeArg(const Graph* p, const std::string& name) const = 0;
  virtual IOnnxRuntimeOpSchemaCollectionPtr Graph__GetSchemaRegistry(const Graph* p) const = 0;
  virtual bool Graph__SetOpSchemaFromRegistryForNode(Graph* p, Node& node) = 0;

  // GraphViewer
  virtual void GraphViewer__operator_delete(GraphViewer* p) = 0;
  virtual std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* p, const logging::Logger& logger, const ModelMetaData&) = 0;

  virtual const std::string& GraphViewer__Name(const GraphViewer* p) noexcept = 0;
  virtual const std::filesystem::path& GraphViewer__ModelPath(const GraphViewer* p) noexcept = 0;

  virtual const ConstGraphNodes& GraphViewer__Nodes(const GraphViewer* p) noexcept = 0;
  virtual const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) = 0;
  virtual const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) = 0;

  virtual bool GraphViewer__IsSubgraph(const GraphViewer* p) = 0;
  virtual const Graph& GraphViewer__GetGraph(const GraphViewer* p) const = 0;
  virtual bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) = 0;
  virtual const ONNX_NAMESPACE::TensorProto* GraphViewer__GetConstantInitializer(const GraphViewer* p,
                                                                                 const std::string& name,
                                                                                 bool check_outer_scope) const = 0;
  virtual bool GraphViewer__GetOrtValueInitializer(const GraphViewer* p, const std::string& tensor_name,
                                                   OrtValue& value) = 0;
  virtual const Node* GraphViewer__ParentNode(const GraphViewer* p) = 0;
  virtual int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept = 0;
  virtual int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept = 0;

  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept = 0;
  virtual bool GraphViewer__NodeProducesGraphOutput(const GraphViewer* p, const Node& node) = 0;
  virtual const std::unordered_set<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept = 0;

  virtual const InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) = 0;
  virtual bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;
  virtual const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) = 0;

  virtual const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p, int execution_order) = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept = 0;
  virtual const std::unordered_set<std::string>& GraphViewer__GetOuterScopeNodeArgNames(const GraphViewer* p) const noexcept = 0;

  virtual void GraphViewer__ToProto(const GraphViewer* p,
                                    ONNX_NAMESPACE::GraphProto& graph_proto,
                                    bool include_initializers,
                                    bool include_outer_scope_args,
                                    int execution_order,
                                    bool include_initializer_data) noexcept = 0;
  virtual const Node* GraphViewer__GetProducerNode(const GraphViewer* p, const std::string& node_arg_name) const = 0;
  virtual IOnnxRuntimeOpSchemaCollectionPtr GraphViewer__GetSchemaRegistry(const GraphViewer* p) const = 0;

  // ConstGraphNodes
  virtual std::unique_ptr<ConstGraphNodes_Iterator> ConstGraphNodes__begin(const ConstGraphNodes* p) = 0;
  virtual std::unique_ptr<ConstGraphNodes_Iterator> ConstGraphNodes__end(const ConstGraphNodes* p) = 0;
  virtual std::unique_ptr<ConstGraphNodes_Iterator> ConstGraphNodes__cbegin(const ConstGraphNodes* p) = 0;
  virtual std::unique_ptr<ConstGraphNodes_Iterator> ConstGraphNodes__cend(const ConstGraphNodes* p) = 0;
  virtual bool ConstGraphNodes__empty(const ConstGraphNodes* p) noexcept = 0;

  // graph_util
  virtual NodeArg& GraphUtils__AddInitializerWithExternalData(Graph& graph,
                                                              const ONNX_NAMESPACE::TensorProto& new_initializer) = 0;
  virtual void GraphUtils__MakeInitializerCopyIfNotExist(const Graph& src_graph, Graph& dst_graph,
                                                         const std::string& name, bool load_inline) = 0;

  virtual Status GraphUtils__ConvertInMemoryDataToInline(Graph& graph, const std::string& name) = 0;

  // ExternalDataInfo
  virtual void ExternalDataInfo__operator_delete(ExternalDataInfo*) = 0;
  virtual const PathString& ExternalDataInfo__GetRelPath(const ExternalDataInfo*) const = 0;
  virtual int64_t ExternalDataInfo__GetOffset(const ExternalDataInfo*) const = 0;
  virtual size_t ExternalDataInfo__GetLength(const ExternalDataInfo*) const = 0;
  virtual const std::string& ExternalDataInfo__GetChecksum(const ExternalDataInfo*) const = 0;
  virtual Status ExternalDataInfo__Create(const ONNX_NAMESPACE::StringStringEntryProtos& input,
                                          std::unique_ptr<ExternalDataInfo>& out) = 0;

  // Initializer
  virtual Initializer* Initializer__constructor(ONNX_NAMESPACE::TensorProto_DataType data_type,
                                                std::string_view name,
                                                gsl::span<const int64_t> dims) = 0;
  virtual Initializer* Initializer__constructor(const Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                const std::filesystem::path& model_path = {},
                                                bool check_outer_scope = false) = 0;

  virtual void Initializer__destructor(Initializer*) = 0;
  virtual void Initializer__ToProto(const Initializer&,
                                    ONNX_NAMESPACE::TensorProto& tensor_proto) = 0;
  virtual void Initializer__ToProtoWithOrtValue(const Initializer&,
                                                ONNX_NAMESPACE::TensorProto& tensor_proto, OrtValue& ort_value) = 0;
  virtual int Initializer__data_type(const Initializer&) = 0;
  virtual const std::string& Initializer__name(const Initializer&) = 0;
  virtual gsl::span<const int64_t> Initializer__dims(const Initializer&) = 0;
  virtual size_t Initializer__size(const Initializer&) = 0;
  // data<T>() template helper
  virtual void* Initializer__mutable_data(Initializer&, int data_type) = 0;
  virtual const void* Initializer__data(const Initializer&, int data_type) = 0;
  virtual void* Initializer__mutable_data_raw(Initializer&) = 0;
  virtual const void* Initializer__data_raw(const Initializer&) = 0;

  // OpKernel
  virtual const Node& OpKernel__Node(const OpKernel* p) = 0;

  // OpKernelContext
  virtual const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual const SparseTensor* OpKernelContext__Input_SparseTensor(const OpKernelContext* p, int index) = 0;
#endif
  virtual const TensorSeq* OpKernelContext__Input_TensorSeq(const OpKernelContext* p, int index) = 0;
  virtual const Tensor& OpKernelContext__RequiredInput_Tensor(const OpKernelContext* p, int index) = 0;
  virtual Tensor* OpKernelContext__Output_Tensor(OpKernelContext* p, int index) = 0;
  virtual TensorSeq* OpKernelContext__Output_TensorSeq(OpKernelContext* p, int index) = 0;
  virtual Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual SparseTensor* OpKernelContext__OutputSparse(OpKernelContext* p, int index, const TensorShape& shape) = 0;
#endif
  virtual Tensor& OpKernelContext__RequiredOutput(OpKernelContext* p, int index, const TensorShape& shape) = 0;
  virtual MLDataType OpKernelContext__InputType(const OpKernelContext* p, int index) = 0;
  virtual int OpKernelContext__InputCount(const OpKernelContext* p) = 0;
  virtual int OpKernelContext__OutputCount(const OpKernelContext* p) = 0;
  virtual Status OpKernelContext__GetTempSpaceAllocator(const OpKernelContext* p, AllocatorPtr* output) = 0;
  virtual Status OpKernelContext__GetTempSpaceCPUAllocator(const OpKernelContext* p, AllocatorPtr* output) = 0;
  virtual bool OpKernelContext__GetUseDeterministicCompute(const OpKernelContext* p) = 0;
  virtual bool OpKernelContext__TryGetInferredOutputShape(const OpKernelContext* p, int index, TensorShape& shape) = 0;
  virtual bool OpKernelContext__TryGetInferredInputShape(const OpKernelContext* p, int index, TensorShape& shape) = 0;
  virtual Stream* OpKernelContext__GetComputeStream(const OpKernelContext* p) = 0;

  // OpKernelInfo
  virtual std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) = 0;
  virtual void OpKernelInfo__operator_delete(OpKernelInfo* p) = 0;
  virtual AllocatorPtr OpKernelInfo__GetAllocator(const OpKernelInfo* p, OrtMemType mem_type) = 0;
  virtual const IExecutionProvider* OpKernelInfo__GetExecutionProvider(const OpKernelInfo* p) = 0;
  virtual Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) = 0;
  virtual Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) = 0;
  virtual Status OpKernelInfo__GetAttr_string(const OpKernelInfo* p, const std::string& name, std::string* value) = 0;
  virtual Status OpKernelInfo__GetAttr_TensorProto(const OpKernelInfo* p, const std::string& name, ONNX_NAMESPACE::TensorProto* value) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<int64_t>& values) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<float>& values) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<std::string>& values) = 0;
  virtual Status OpKernelInfo__GetAttrsAsSpan(const OpKernelInfo* p, const std::string& name, gsl::span<const int64_t>& values) = 0;

  virtual const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept = 0;
  virtual const KernelDef& OpKernelInfo__GetKernelDef(const OpKernelInfo* p) = 0;
  virtual bool OpKernelInfo__TryGetConstantInput(const OpKernelInfo* p, int input_index, const Tensor** constant_input_value) = 0;

  virtual uint32_t OpKernelInfo__GetInputCount(const OpKernelInfo* p) = 0;
  virtual uint32_t OpKernelInfo__GetOutputCount(const OpKernelInfo* p) = 0;
  virtual const Node& OpKernelInfo__node(const OpKernelInfo* p) = 0;
  virtual const ConfigOptions& OpKernelInfo__GetConfigOptions(const OpKernelInfo* p) = 0;

  // SessionState
  virtual const DataTransferManager& SessionState__GetDataTransferMgr(const SessionState* p) = 0;

  // Tensor
  virtual std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) = 0;
  virtual std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset) = 0;
  virtual std::unique_ptr<Tensor> Tensor__construct_default() = 0;
  virtual void Tensor__move_assign(Tensor& lhs, Tensor&& rhs) noexcept = 0;
  virtual void Tensor__operator_delete(Tensor* p) noexcept = 0;

  virtual void Tensor__InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) = 0;
  virtual void Tensor__InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location, OrtValue& ort_value) = 0;

  virtual bool* Tensor__MutableData_bool(Tensor* p) = 0;
  virtual int8_t* Tensor__MutableData_int8(Tensor* p) = 0;
  virtual uint8_t* Tensor__MutableData_uint8(Tensor* p) = 0;
  virtual int16_t* Tensor__MutableData_int16(Tensor* p) = 0;
  virtual uint16_t* Tensor__MutableData_uint16(Tensor* p) = 0;
  virtual int32_t* Tensor__MutableData_int32(Tensor* p) = 0;
  virtual uint32_t* Tensor__MutableData_uint32(Tensor* p) = 0;
  virtual int64_t* Tensor__MutableData_int64(Tensor* p) = 0;
  virtual uint64_t* Tensor__MutableData_uint64(Tensor* p) = 0;
  virtual float* Tensor__MutableData_float(Tensor* p) = 0;
  virtual double* Tensor__MutableData_double(Tensor* p) = 0;
  virtual BFloat16* Tensor__MutableData_BFloat16(Tensor* p) = 0;
  virtual MLFloat16* Tensor__MutableData_MLFloat16(Tensor* p) = 0;

#if !defined(DISABLE_FLOAT8_TYPES)
  virtual Float8E4M3FN* Tensor__MutableData_Float8E4M3FN(Tensor* p) = 0;
  virtual Float8E4M3FNUZ* Tensor__MutableData_Float8E4M3FNUZ(Tensor* p) = 0;
  virtual Float8E5M2* Tensor__MutableData_Float8E5M2(Tensor* p) = 0;
  virtual Float8E5M2FNUZ* Tensor__MutableData_Float8E5M2FNUZ(Tensor* p) = 0;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  virtual Float4E2M1x2* Tensor__MutableData_Float4E2M1x2(Tensor* p) = 0;
#endif
  virtual Int4x2* Tensor__MutableData_Int4x2(Tensor* p) = 0;
  virtual UInt4x2* Tensor__MutableData_UInt4x2(Tensor* p) = 0;
  virtual Int2x4* Tensor__MutableData_Int2x4(Tensor* p) = 0;
  virtual UInt2x4* Tensor__MutableData_UInt2x4(Tensor* p) = 0;

  virtual const bool* Tensor__Data_bool(const Tensor* p) = 0;
  virtual const int8_t* Tensor__Data_int8(const Tensor* p) = 0;
  virtual const uint8_t* Tensor__Data_uint8(const Tensor* p) = 0;
  virtual const int16_t* Tensor__Data_int16(const Tensor* p) = 0;
  virtual const uint16_t* Tensor__Data_uint16(const Tensor* p) = 0;
  virtual const int32_t* Tensor__Data_int32(const Tensor* p) = 0;
  virtual const uint32_t* Tensor__Data_uint32(const Tensor* p) = 0;
  virtual const int64_t* Tensor__Data_int64(const Tensor* p) = 0;
  virtual const uint64_t* Tensor__Data_uint64(const Tensor* p) = 0;
  virtual const float* Tensor__Data_float(const Tensor* p) = 0;
  virtual const double* Tensor__Data_double(const Tensor* p) = 0;
  virtual const BFloat16* Tensor__Data_BFloat16(const Tensor* p) = 0;
  virtual const MLFloat16* Tensor__Data_MLFloat16(const Tensor* p) = 0;

#if !defined(DISABLE_FLOAT8_TYPES)
  virtual const Float8E4M3FN* Tensor__Data_Float8E4M3FN(const Tensor* p) = 0;
  virtual const Float8E4M3FNUZ* Tensor__Data_Float8E4M3FNUZ(const Tensor* p) = 0;
  virtual const Float8E5M2* Tensor__Data_Float8E5M2(const Tensor* p) = 0;
  virtual const Float8E5M2FNUZ* Tensor__Data_Float8E5M2FNUZ(const Tensor* p) = 0;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  virtual const Float4E2M1x2* Tensor__Data_Float4E2M1x2(const Tensor* p) = 0;
#endif
  virtual const Int4x2* Tensor__Data_Int4x2(const Tensor* p) = 0;
  virtual const UInt4x2* Tensor__Data_UInt4x2(const Tensor* p) = 0;
  virtual const Int2x4* Tensor__Data_Int2x4(const Tensor* p) = 0;
  virtual const UInt2x4* Tensor__Data_UInt2x4(const Tensor* p) = 0;

  virtual gsl::span<const int64_t> Tensor__DataAsSpan_int64(const Tensor* p) = 0;
  virtual gsl::span<const int32_t> Tensor__DataAsSpan_int32(const Tensor* p) = 0;

  virtual void* Allocator__AllocateBufferWithOptions(IAllocator& allocator, size_t size, bool use_reserve, Stream* stream, WaitNotificationFn wait_fn) = 0;

  virtual void* Tensor__MutableDataRaw(Tensor* p, MLDataType type) = 0;
  virtual const void* Tensor__DataRaw(const Tensor* p, MLDataType type) = 0;
  virtual void* Tensor__MutableDataRaw(Tensor* p) noexcept = 0;
  virtual const void* Tensor__DataRaw(const Tensor* p) noexcept = 0;

  virtual bool Tensor__IsDataType_bool(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_int8(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_uint8(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_int16(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_uint16(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_int32(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_uint32(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_int64(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_uint64(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_float(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_double(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_MLFloat16(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_BFloat16(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataTypeString(const Tensor* p) noexcept = 0;

#if !defined(DISABLE_FLOAT8_TYPES)
  virtual bool Tensor__IsDataType_Float8E4M3FN(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_Float8E4M3FNUZ(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_Float8E5M2(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_Float8E5M2FNUZ(const Tensor* p) noexcept = 0;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
  virtual bool Tensor__IsDataType_Float4E2M1x2(const Tensor* p) noexcept = 0;
#endif
  virtual bool Tensor__IsDataType_Int4x2(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_UInt4x2(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_Int2x4(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_UInt2x4(const Tensor* p) noexcept = 0;

  virtual const TensorShape& Tensor__Shape(const Tensor* p) = 0;
  virtual void Tensor__Reshape(Tensor* p, const TensorShape& new_shape) = 0;
  virtual void Tensor__SetByteOffset(Tensor* p, ptrdiff_t byte_offset) = 0;
  virtual ptrdiff_t Tensor__ByteOffset(const Tensor* p) = 0;
  virtual size_t Tensor__SizeInBytes(const Tensor* p) = 0;
  virtual const OrtMemoryInfo& Tensor__Location(const Tensor* p) = 0;
  virtual int32_t Tensor__GetElementType(const Tensor* p) = 0;
  virtual MLDataType Tensor__DataType(const Tensor* p) = 0;
#ifdef ENABLE_STRIDED_TENSORS
  virtual gsl::span<const int64_t> Tensor__Strides(const Tensor* p) = 0;
  virtual bool Tensor__IsContiguous(const Tensor* p) = 0;
  virtual void Tensor__SetShapeAndStrides(Tensor* p, const TensorShape& new_shape,
                                          gsl::span<const int64_t> new_strides) = 0;
#endif

#if !defined(DISABLE_SPARSE_TENSORS)
  // SparseTensor
  virtual const TensorShape& SparseTensor__DenseShape(const SparseTensor*) = 0;
  virtual Status SparseTensor__Copy(const SparseTensor*, const DataTransferManager&, SparseTensor&) = 0;
#endif

  // TensorSeq
  virtual MLDataType TensorSeq__DataType(const TensorSeq* p) noexcept = 0;
  virtual void TensorSeq__SetType(TensorSeq* p, MLDataType data_type) = 0;
  virtual size_t TensorSeq__Size(const TensorSeq* p) noexcept = 0;
  virtual const Tensor& TensorSeq__Get(const TensorSeq* p, size_t i) = 0;
  virtual const OrtValue& TensorSeq__GetAt(const TensorSeq* p, size_t i) = 0;
  virtual void TensorSeq__Add(TensorSeq* p, const OrtValue& tensor) = 0;
  virtual void TensorSeq__Add(TensorSeq* p, OrtValue&& tensor) = 0;
  virtual void TensorSeq__Add(TensorSeq* p, Tensor&& tensor) = 0;
  virtual void TensorSeq__Reserve(TensorSeq* p, size_t capacity) = 0;

#if defined(ENABLE_TRAINING) && defined(ORT_USE_NCCL)
  virtual training::DistributedRunContext& GetDistributedRunContextInstance() = 0;
#endif

  virtual PhiloxGenerator& PhiloxGenerator__Default() = 0;

#ifdef ENABLE_TRAINING_TORCH_INTEROP
  virtual void contrib__PythonOpBase__Init(contrib::PythonOpBase* p, const OpKernelInfo& info) = 0;
  virtual void contrib__PythonOpBase__Clear(contrib::PythonOpBase* p) = 0;
  virtual void contrib__PythonOpBase__RunForward(const contrib::PythonOpBase* p, OpKernelContext* context, void** diff_ctx, std::vector<OrtValue>& returned_ortvalues) = 0;
  virtual void contrib__PythonOpBase__SetOutputs(const contrib::PythonOpBase* p, OpKernelContext* context, void* diff_ctx, std::vector<OrtValue>& returned_args) = 0;

  virtual void contrib__PythonOpGradBase__Init(contrib::PythonOpGradBase* p, const OpKernelInfo& info) = 0;
  virtual void contrib__PythonOpGradBase__RunBackward(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) = 0;
  virtual void contrib__PythonOpGradBase__SetOutputs(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_args) = 0;

  virtual language_interop_ops::torch::RefCountTracker& GetRefCountTrackerInstance() = 0;
  virtual void RefCountTracker__DumpDetails(const language_interop_ops::torch::RefCountTracker* p, const std::string& phase_name) = 0;
#endif

#if defined(USE_CANN)
  virtual RandomGenerator& RandomGenerator__Default() = 0;
  virtual std::unique_ptr<Model> cann__CreateModel(const GraphViewer& graph_viewer, const logging::Logger& logger) = 0;
#endif

  virtual void MurmurHash3__x86_128(const void* key, size_t len, uint32_t seed, void* out) = 0;

#ifdef _WIN32
  virtual std::string ToUTF8String(const std::wstring& s) = 0;
  virtual std::wstring ToWideString(const std::string& s) = 0;
#endif

  virtual ProviderHostCPU& GetProviderHostCPU() = 0;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  virtual Status LoadDynamicLibrary(onnxruntime::PathString library_name) = 0;
#endif

  // ModelMetadefIdGenerator
  virtual std::unique_ptr<ModelMetadefIdGenerator> ModelMetadefIdGenerator__construct() = 0;
  virtual void ModelMetadefIdGenerator__operator_delete(ModelMetadefIdGenerator* p) = 0;
  virtual int ModelMetadefIdGenerator__GenerateId(const ModelMetadefIdGenerator* p, const GraphViewer& graph_viewer, HashValue& model_hash) = 0;

#if !defined(DISABLE_FLOAT8_TYPES)
  virtual MLDataType DataTypeImpl__GetType_Float8E8M0() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float8E8M0() = 0;
#if !defined(DISABLE_SPARSE_TENSORS)
  virtual MLDataType DataTypeImpl__GetSparseTensorType_Float8E8M0() = 0;
#endif
  virtual Float8E8M0* Tensor__MutableData_Float8E8M0(Tensor* p) = 0;
  virtual const Float8E8M0* Tensor__Data_Float8E8M0(const Tensor* p) = 0;
  virtual bool Tensor__IsDataType_Float8E8M0(const Tensor* p) noexcept = 0;
#endif

  virtual void SessionOptions__GetEpContextDataCallbacks(const OrtSessionOptions* p,
                                                         OrtReadNamedBufferFunc* read_func, void** read_state,
                                                         OrtWriteNamedBufferFunc* write_func, void** write_state) = 0;
  // Retained for provider vtable ABI compatibility. Run instrumentation is no longer active.
  virtual const RunInstrumentationContext* OpKernelContext__GetRunInstrumentationContext(
      const OpKernelContext* p) = 0;
  virtual const std::string& RunInstrumentationContext__RequestId(const RunInstrumentationContext* p) = 0;
  virtual TimePoint RunInstrumentationContext__StartProfiling(const RunInstrumentationContext* p) = 0;
  virtual uint64_t RunInstrumentationContext__ProfilerStartTimeNs(const RunInstrumentationContext* p) = 0;
  virtual void RunInstrumentationContext__AddDeferredRecord(
      const RunInstrumentationContext* p,
      std::unique_ptr<DeferredRunInstrumentationRecord> record) = 0;
  virtual bool RunInstrumentationContext__TryReserveMoeRoutingRecord(
      const RunInstrumentationContext* p, size_t element_count) = 0;
  virtual void RunInstrumentationContext__RecordMoeRoutingEvent(
      const RunInstrumentationContext* p,
      const TimePoint& start_time,
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
      const std::string& completion_timestamp_source) = 0;

  // Kernel pilot support — appended at end to preserve vtable ABI compatibility.
  virtual KernelPilot* OpKernelContext__GetKernelPilot(const OpKernelContext* p) = 0;

  // Float6 support — appended at end to preserve vtable ABI compatibility.
  virtual MLDataType DataTypeImpl__GetType_Float6E2M3() = 0;
  virtual MLDataType DataTypeImpl__GetType_Float6E3M2() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float6E2M3() = 0;
  virtual MLDataType DataTypeImpl__GetTensorType_Float6E3M2() = 0;
  virtual Float6E2M3* Tensor__MutableData_Float6E2M3(Tensor* p) = 0;
  virtual Float6E3M2* Tensor__MutableData_Float6E3M2(Tensor* p) = 0;
  virtual const Float6E2M3* Tensor__Data_Float6E2M3(const Tensor* p) = 0;
  virtual const Float6E3M2* Tensor__Data_Float6E3M2(const Tensor* p) = 0;
  virtual bool Tensor__IsDataType_Float6E2M3(const Tensor* p) noexcept = 0;
  virtual bool Tensor__IsDataType_Float6E3M2(const Tensor* p) noexcept = 0;
};

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
}  // namespace onnxruntime
