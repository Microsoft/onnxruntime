// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
#include "core/providers/shared_library/provider_host_api.h"

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

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
// These types don't directly map to internal types
struct ProviderHost;
struct ProviderHostCPU;

class PhiloxGenerator;

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
  const TResult& operator*() { return p_->operator*(); }
  T* operator->() { return p_.get(); }

 private:
  std::unique_ptr<T> p_;
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

using NodeIndex = size_t;
// We can't just reinterpret_cast this one, since it's an unordered_map of object BY VALUE (can't do anything by value on the real types)
// using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto_Copyable>;

using InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

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
  virtual const Node& GetNode() const = 0;
  virtual int GetSrcArgIndex() const = 0;
  virtual int GetDstArgIndex() const = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
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

#ifdef USE_CUDA
  virtual std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;

  virtual bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
  virtual bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
#endif

  virtual std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                             const std::string& provider_type,
                                                             const std::vector<const KernelRegistry*>& kernel_registries,
                                                             const std::vector<NodeIndex>& tentative_nodes) = 0;

  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ bool* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ float* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ double* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ MLFloat16* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int8_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint8_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int16_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint16_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int32_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint32_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int64_t* p_data, size_t expected_size) = 0;
  virtual Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint64_t* p_data, size_t expected_size) = 0;

  virtual uint16_t math__floatToHalf(float f) = 0;
  virtual float math__halfToFloat(uint16_t h) = 0;

  // IAllocator
  virtual bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) = 0;

  // IExecutionProvider
  virtual AllocatorPtr IExecutionProvider__GetAllocator(const IExecutionProvider* p, int id, OrtMemType mem_type) = 0;
  virtual void IExecutionProvider__InsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) = 0;
  virtual void IExecutionProvider__TryInsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) = 0;
  virtual std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
                                                                                            const std::vector<const KernelRegistry*>& kernel_registries) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::string& dll_path) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual int IExecutionProvider__GenerateMetaDefId(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) = 0;

  virtual void IExecutionProvider__RegisterAllocator(IExecutionProvider* p, std::shared_ptr<AllocatorManager> allocator_manager) = 0;
  // Status
  virtual std::string Status__ToString(const Status* p) = 0;

  // TensorShape
  virtual int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) = 0;
  virtual std::string TensorShape__ToString(const TensorShape* p) = 0;
  virtual int64_t TensorShape__SizeToDimension(const TensorShape* p, size_t dimension) = 0;
  virtual int64_t TensorShape__SizeFromDimension(const TensorShape* p, size_t dimension) = 0;
  virtual std::ostream& operator_left_shift(std::ostream& out, const TensorShape& shape) = 0;

  // CPUIDInfo
  virtual const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() = 0;
  virtual bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) = 0;

  // logging::Logger
  virtual bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) = 0;

  // logging::LoggingManager
  virtual const logging::Logger& logging__LoggingManager__DefaultLogger() = 0;

  // logging::Capture
  virtual std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) = 0;
  virtual void logging__Capture__operator_delete(logging::Capture* p) noexcept = 0;
  virtual std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept = 0;

  // Utils::DataTypeUtils
  virtual const std::string* Utils__DataTypeUtils__ToType(const ONNX_NAMESPACE::TypeProto& type_proto) = 0;

  // int64s
  virtual int int64s__size(const ONNX_NAMESPACE::int64s* p) = 0;
  virtual const int64_t& int64s__Get(const ONNX_NAMESPACE::int64s* p, int index) = 0;

  // TypeProto_Tensor
  virtual bool TypeProto_Tensor__has_shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto& TypeProto_Tensor__shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto* TypeProto_Tensor__mutable_shape(ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual int32_t TypeProto_Tensor__elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;

  // TypeProto
  virtual const ONNX_NAMESPACE::TypeProto_Tensor& TypeProto__tensor_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_Tensor* TypeProto__mutable_tensor_type(ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual int TypeProto__value_case(const ONNX_NAMESPACE::TypeProto* p) = 0;

  // AttributeProto
  virtual std::unique_ptr<ONNX_NAMESPACE::AttributeProto> AttributeProto__construct() = 0;
  virtual void AttributeProto__operator_delete(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__operator_assign(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::AttributeProto& v) = 0;

  virtual ONNX_NAMESPACE::AttributeProto_AttributeType AttributeProto__type(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__ints_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__floats_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__strings_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int64_t AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual float AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual const ::std::string& AttributeProto__strings(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual const ONNX_NAMESPACE::int64s& AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int64_t AttributeProto__i(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual float AttributeProto__f(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual const ::std::string& AttributeProto__s(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_name(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual void AttributeProto__set_type(ONNX_NAMESPACE::AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) = 0;
  virtual ONNX_NAMESPACE::TensorProto* AttributeProto__add_tensors(ONNX_NAMESPACE::AttributeProto* p) = 0;

  // GraphProto
  virtual void GraphProto__operator_delete(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual void GraphProto__operator_assign(ONNX_NAMESPACE::GraphProto* p, const ONNX_NAMESPACE::GraphProto& v) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__input(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProto* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual int GraphProto__input_size(const ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProtos& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_output(ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_value_info(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorProtos* GraphProto__mutable_initializer(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::NodeProto* GraphProto__add_node(ONNX_NAMESPACE::GraphProto* p) = 0;

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

  // TensorProto
  virtual std::unique_ptr<ONNX_NAMESPACE::TensorProto> TensorProto__construct() = 0;
  virtual void TensorProto__operator_delete(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__operator_assign(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto& v) = 0;
  virtual bool TensorProto__has_name(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int TensorProto__dims_size(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual const ONNX_NAMESPACE::int64s& TensorProto__dims(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual bool TensorProto__has_data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int TensorProto__data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual bool TensorProto__has_raw_data(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual const std::string& TensorProto__raw_data(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int32_t TensorProto__data_type(const ONNX_NAMESPACE::TensorProto* p) = 0;

  virtual bool TensorProto_DataType_IsValid(int value) = 0;

  // TensorProtos
  virtual ONNX_NAMESPACE::TensorProto* TensorProtos__Add(ONNX_NAMESPACE::TensorProtos* p) = 0;

  // TensorShapeProto_Dimension
  virtual int TensorShapeProto_Dimension__value_case(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual const std::string& TensorShapeProto_Dimension__dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual int64_t TensorShapeProto_Dimension__dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__set_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, int64_t value) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__clear_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;

  // TensorShapeProto_Dimensions
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__begin(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__end(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;

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

  // ComputeCapability
  virtual std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) = 0;
  virtual void ComputeCapability__operator_delete(ComputeCapability* p) = 0;
  virtual std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) = 0;

  // DataTransferManager
  virtual Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst, int exec_queue_id) = 0;
  virtual Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst) = 0;
  virtual const IDataTransfer* DataTransferManager__GetDataTransfer(const DataTransferManager* p, const OrtDevice& src_device, const OrtDevice& dst_device) = 0;

  // IDataTransfer
  virtual Status IDataTransfer__CopyTensor(const IDataTransfer* p, const Tensor& src, Tensor& dst) = 0;
  virtual Status IDataTransfer__CopyTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) = 0;

  // IndexedSubGraph_MetaDef
  virtual std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() = 0;
  virtual void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) = 0;

  virtual std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) = 0;
  virtual int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) = 0;
  virtual ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) = 0;

  // IndexedSubGraph
  virtual std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() = 0;
  virtual void IndexedSubGraph__operator_delete(IndexedSubGraph* p) = 0;

  virtual std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) = 0;

  virtual void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) = 0;
  virtual const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) = 0;

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

  virtual std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) = 0;

  // KernelRegistry
  virtual std::shared_ptr<KernelRegistry> KernelRegistry__construct() = 0;
  virtual void KernelRegistry__operator_delete(KernelRegistry* p) = 0;
  virtual Status KernelRegistry__Register(KernelRegistry* p, KernelCreateInfo&& create_info) = 0;
  virtual Status KernelRegistry__TryFindKernel(const KernelRegistry* p, const Node& node, ProviderType exec_provider, const KernelCreateInfo** out) = 0;

  // PrimitiveDataTypeBase
  virtual int32_t PrimitiveDataTypeBase__GetDataType(const PrimitiveDataTypeBase* p) = 0;

  // DataTypeImpl
  virtual MLDataType DataTypeImpl__GetType_Tensor() = 0;
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
  virtual const char* DataTypeImpl__ToString(MLDataType type) = 0;
  virtual bool DataTypeImpl__IsTensorType(const DataTypeImpl* p) = 0;
  virtual bool DataTypeImpl__IsTensorSequenceType(const DataTypeImpl* p) = 0;
  virtual bool DataTypeImpl__IsSparseTensorType(const DataTypeImpl* p) = 0;
  virtual DeleteFunc DataTypeImpl__GetDeleteFunc(const DataTypeImpl* p) = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllIEEEFloatTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypes() = 0;
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

  virtual void Node__ToProto(const Node* p, ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) = 0;

  virtual const NodeAttributes& Node__GetAttributes(const Node* p) noexcept = 0;
  virtual size_t Node__GetInputEdgesCount(const Node* p) noexcept = 0;
  virtual size_t Node__GetOutputEdgesCount(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept = 0;

  virtual void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) = 0;

  // NodeArg
  virtual const std::string& NodeArg__Name(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg__Shape(const NodeArg* p) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept = 0;
  virtual bool NodeArg__Exists(const NodeArg* p) const noexcept = 0;
  virtual const ONNX_NAMESPACE::TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept = 0;

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

  // Model
  virtual void Model__operator_delete(Model* p) = 0;
  virtual Graph& Model__MainGraph(Model* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToProto(Model* p) = 0;

  // Graph
  virtual std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::GraphProto> Graph__ToGraphProto(const Graph* p) = 0;

  virtual NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) = 0;

  virtual Status Graph__Resolve(Graph* p) = 0;
  virtual void Graph__AddInitializedTensor(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor) = 0;
  virtual Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept = 0;
  virtual void Graph__SetOutputs(Graph* p, const std::vector<const NodeArg*>& outputs) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept = 0;
  virtual bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;

  // GraphViewer
  virtual void GraphViewer__operator_delete(GraphViewer* p) = 0;
  virtual std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* p, const logging::Logger& logger) = 0;

  virtual const std::string& GraphViewer__Name(const GraphViewer* p) noexcept = 0;
  virtual const Path& GraphViewer__ModelPath(const GraphViewer* p) noexcept = 0;

  virtual const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) = 0;
  virtual const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) = 0;

  virtual bool GraphViewer__IsSubgraph(const GraphViewer* p) = 0;
  virtual bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) = 0;
  virtual int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept = 0;
  virtual int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept = 0;

  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept = 0;

  virtual const InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) = 0;
  virtual bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;
  virtual const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) = 0;

  virtual const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p) = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept = 0;

  // Path
  virtual PathString Path__ToPathString(const Path* p) noexcept = 0;

  // OpKernel
  virtual const Node& OpKernel__Node(const OpKernel* p) = 0;

  // OpKernelContext
  virtual const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) = 0;
  virtual const TensorSeq* OpKernelContext__Input_TensorSeq(const OpKernelContext* p, int index) = 0;
  virtual const Tensor& OpKernelContext__RequiredInput_Tensor(const OpKernelContext* p, int index) = 0;
  virtual Tensor* OpKernelContext__Output_Tensor(OpKernelContext* p, int index) = 0;
  virtual TensorSeq* OpKernelContext__Output_TensorSeq(OpKernelContext* p, int index) = 0;
  virtual Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) = 0;
  virtual Tensor& OpKernelContext__RequiredOutput(OpKernelContext* p, int index, const TensorShape& shape) = 0;
  virtual MLDataType OpKernelContext__InputType(const OpKernelContext* p, int index) = 0;
  virtual int OpKernelContext__InputCount(const OpKernelContext* p) = 0;
  virtual int OpKernelContext__OutputCount(const OpKernelContext* p) = 0;
  virtual Status OpKernelContext__GetTempSpaceAllocator(const OpKernelContext* p, AllocatorPtr* output) = 0;
  virtual bool OpKernelContext__GetUseDeterministicCompute(const OpKernelContext* p) = 0;
  virtual bool OpKernelContext__TryGetInferredOutputShape(const OpKernelContext* p, int index, TensorShape& shape) = 0;
  virtual bool OpKernelContext__TryGetInferredInputShape(const OpKernelContext* p, int index, TensorShape& shape) = 0;

  // OpKernelInfo
  virtual std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) = 0;
  virtual void OpKernelInfo__operator_delete(OpKernelInfo* p) = 0;
  virtual AllocatorPtr OpKernelInfo__GetAllocator(const OpKernelInfo* p, int device_id, OrtMemType mem_type) = 0;
  virtual const IExecutionProvider* OpKernelInfo__GetExecutionProvider(const OpKernelInfo* p) = 0;
  virtual Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) = 0;
  virtual Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) = 0;
  virtual Status OpKernelInfo__GetAttr_string(const OpKernelInfo* p, const std::string& name, std::string* value) = 0;
  virtual Status OpKernelInfo__GetAttr_TensorProto(const OpKernelInfo* p, const std::string& name, ONNX_NAMESPACE::TensorProto* value) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<int64_t>& values) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<float>& values) = 0;
  virtual Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<std::string>& values) = 0;

  virtual const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept = 0;
  virtual const KernelDef& OpKernelInfo__GetKernelDef(const OpKernelInfo* p) = 0;
  virtual bool OpKernelInfo__TryGetConstantInput(const OpKernelInfo* p, int input_index, const Tensor** constant_input_value) = 0;

  virtual uint32_t OpKernelInfo__GetInputCount(const OpKernelInfo* p) = 0;
  virtual uint32_t OpKernelInfo__GetOutputCount(const OpKernelInfo* p) = 0;
  virtual const Node& OpKernelInfo__node(const OpKernelInfo* p) = 0;

  // SessionState
  virtual const DataTransferManager& SessionState__GetDataTransferMgr(const SessionState* p) = 0;

  // Tensor
  virtual std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) = 0;
  virtual std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset) = 0;
  virtual void Tensor__operator_delete(Tensor* p) = 0;

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

  virtual gsl::span<const int64_t> Tensor__DataAsSpan_int64(const Tensor* p) = 0;

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
  virtual bool Tensor__IsDataTypeString(const Tensor* p) noexcept = 0;

  virtual const TensorShape& Tensor__Shape(const Tensor* p) = 0;
  virtual void Tensor__Reshape(Tensor* p, const TensorShape& new_shape) = 0;
  virtual void Tensor__SetByteOffset(Tensor* p, ptrdiff_t byte_offset) = 0;
  virtual ptrdiff_t Tensor__ByteOffset(const Tensor* p) = 0;
  virtual size_t Tensor__SizeInBytes(const Tensor* p) = 0;
  virtual const OrtMemoryInfo& Tensor__Location(const Tensor* p) = 0;
  virtual int32_t Tensor__GetElementType(const Tensor* p) = 0;
  virtual MLDataType Tensor__DataType(const Tensor* p) = 0;

  // TensorSeq
  virtual MLDataType TensorSeq__DataType(const TensorSeq* p) noexcept = 0;
  virtual void TensorSeq__SetType(TensorSeq* p, MLDataType data_type) = 0;
  virtual size_t TensorSeq__Size(const TensorSeq* p) noexcept = 0;
  virtual const Tensor& TensorSeq__Get(const TensorSeq* p, size_t i) = 0;
  virtual void TensorSeq__Add(TensorSeq* p, Tensor&& tensor) = 0;

  // AllocatorManager
  virtual void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator) = 0;
  virtual AllocatorPtr AllocatorManager__GetAllocator(const AllocatorManager* p, int id, OrtMemType mem_type) = 0;

#ifdef USE_CUDA

  virtual PhiloxGenerator& PhiloxGenerator__Default() = 0;

#if defined(ENABLE_TRAINING) && defined(ORT_USE_NCCL)
  virtual training::DistributedRunContext& GetDistributedRunContextInstance() = 0;
#endif

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
#endif

  virtual ProviderHostCPU& GetProviderHostCPU()=0;
};

#ifdef SHARED_PROVIDER

extern ProviderHost* g_host;

struct CPUIDInfo final {
  static const CPUIDInfo& GetCPUIDInfo() { return g_host->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return g_host->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return g_host->CPUIDInfo__HasAVX512f(this); }

  PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger final {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return g_host->logging__Logger__OutputIsEnabled(this, severity, data_type); }

  PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager final {
  static const Logger& DefaultLogger() { return g_host->logging__LoggingManager__DefaultLogger(); }

  PROVIDER_DISALLOW_ALL(LoggingManager)
};

struct Capture final {
  static std::unique_ptr<Capture> Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType dataType, const CodeLocation& location) { return g_host->logging__Capture__construct(logger, severity, category, dataType, location); }
  static void operator delete(void* p) { g_host->logging__Capture__operator_delete(reinterpret_cast<Capture*>(p)); }

  std::ostream& Stream() noexcept { return g_host->logging__Capture__Stream(this); }

  Capture() = delete;
  Capture(const Capture&) = delete;
  void operator=(const Capture&) = delete;
};
}  // namespace logging
}

namespace ONNX_NAMESPACE {

struct int64s final {
  int size() const { return g_host->int64s__size(this); }
  const int64_t& Get(int index) const { return g_host->int64s__Get(this, index); }
  const int64_t& operator[](int index) const { return Get(index); }

  PROVIDER_DISALLOW_ALL(int64s)
};

struct AttributeProto final {
  static std::unique_ptr<AttributeProto> Create() { return g_host->AttributeProto__construct(); }
  void operator=(const AttributeProto& v) { g_host->AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->AttributeProto__operator_delete(reinterpret_cast<AttributeProto*>(p)); }

  AttributeProto_AttributeType type() const { return g_host->AttributeProto__type(this); }
  int ints_size() const { return g_host->AttributeProto__ints_size(this); }
  int floats_size() const { return g_host->AttributeProto__floats_size(this); }
  int strings_size() const { return g_host->AttributeProto__strings_size(this); }
  int64_t ints(int i) const { return g_host->AttributeProto__ints(this, i); }
  float floats(int i) const { return g_host->AttributeProto__floats(this, i); }
  const std::string& strings(int i) const { return g_host->AttributeProto__strings(this, i); }
  const int64s& ints() const { return g_host->AttributeProto__ints(this); }
  int64_t i() const { return g_host->AttributeProto__i(this); }
  float f() const { return g_host->AttributeProto__f(this); }
  void set_s(const ::std::string& value) { return g_host->AttributeProto__set_s(this, value); }
  const ::std::string& s() const { return g_host->AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return g_host->AttributeProto__set_name(this, value); }
  void set_type(AttributeProto_AttributeType value) { return g_host->AttributeProto__set_type(this, value); }
  TensorProto* add_tensors() { return g_host->AttributeProto__add_tensors(this); }

  typedef AttributeProto_AttributeType AttributeType;
  static constexpr AttributeType UNDEFINED = AttributeProto_AttributeType_UNDEFINED;
  static constexpr AttributeType FLOAT = AttributeProto_AttributeType_FLOAT;
  static constexpr AttributeType INT = AttributeProto_AttributeType_INT;
  static constexpr AttributeType STRING = AttributeProto_AttributeType_STRING;
  static constexpr AttributeType TENSOR = AttributeProto_AttributeType_TENSOR;
  static constexpr AttributeType GRAPH = AttributeProto_AttributeType_GRAPH;
  static constexpr AttributeType SPARSE_TENSOR = AttributeProto_AttributeType_SPARSE_TENSOR;
  static constexpr AttributeType FLOATS = AttributeProto_AttributeType_FLOATS;
  static constexpr AttributeType INTS = AttributeProto_AttributeType_INTS;
  static constexpr AttributeType STRINGS = AttributeProto_AttributeType_STRINGS;
  static constexpr AttributeType TENSORS = AttributeProto_AttributeType_TENSORS;
  static constexpr AttributeType GRAPHS = AttributeProto_AttributeType_GRAPHS;
  static constexpr AttributeType SPARSE_TENSORS = AttributeProto_AttributeType_SPARSE_TENSORS;

  AttributeProto() = delete;
  AttributeProto(const AttributeProto&) = delete;
};

struct GraphProto final {
  static void operator delete(void* p) { g_host->GraphProto__operator_delete(reinterpret_cast<GraphProto*>(p)); }
  void operator=(const GraphProto& v) { return g_host->GraphProto__operator_assign(this, v); }

  const ValueInfoProto& input(int index) const { return g_host->GraphProto__input(this, index); }
  ValueInfoProtos* mutable_input() { return g_host->GraphProto__mutable_input(this); }
  ValueInfoProto* mutable_input(int index) { return g_host->GraphProto__mutable_input(this, index); }
  int input_size() const { return g_host->GraphProto__input_size(this); }

  const ValueInfoProtos& output() const { return g_host->GraphProto__output(this); }
  const ValueInfoProto& output(int index) const { return g_host->GraphProto__output(this, index); }
  ValueInfoProtos* mutable_output() { return g_host->GraphProto__mutable_output(this); }

  ValueInfoProtos* mutable_value_info() { return g_host->GraphProto__mutable_value_info(this); }
  TensorProtos* mutable_initializer() { return g_host->GraphProto__mutable_initializer(this); }
  NodeProto* add_node() { return g_host->GraphProto__add_node(this); }

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

  const GraphProto& graph() const { return g_host->ModelProto__graph(this); }
  GraphProto* mutable_graph() { return g_host->ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return g_host->ModelProto__set_ir_version(this, value); }

  ModelProto() = delete;
  ModelProto(const ModelProto&) = delete;
  void operator=(const ModelProto&) = delete;
};

struct TensorProto final {
  static std::unique_ptr<TensorProto> Create() { return g_host->TensorProto__construct(); }
  static void operator delete(void* p) { g_host->TensorProto__operator_delete(reinterpret_cast<TensorProto*>(p)); }
  void operator=(const TensorProto& v) { g_host->TensorProto__operator_assign(this, v); }

  bool has_name() const { return g_host->TensorProto__has_name(this); }

  int dims_size() const { return g_host->TensorProto__dims_size(this); }
  const int64s& dims() const { return g_host->TensorProto__dims(this); }

  bool has_data_location() const { return g_host->TensorProto__has_data_location(this); }
  TensorProto_DataLocation data_location() const { return TensorProto_DataLocation(g_host->TensorProto__data_location(this)); }

  bool has_raw_data() const { return g_host->TensorProto__has_raw_data(this); }
  const std::string& raw_data() const { return g_host->TensorProto__raw_data(this); }

  int32_t data_type() const { return g_host->TensorProto__data_type(this); }

  typedef TensorProto_DataType DataType;
  static constexpr DataType UNDEFINED = TensorProto_DataType_UNDEFINED;

  static bool DataType_IsValid(int value) { return g_host->TensorProto_DataType_IsValid(value); }

  TensorProto() = delete;
  TensorProto(const TensorProto&) = delete;
};

struct TensorProtos final {
  TensorProto* Add() { return g_host->TensorProtos__Add(this); }

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

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimension)
};

struct TensorShapeProto_Dimensions final {
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> begin() const { return g_host->TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> end() const { return g_host->TensorShapeProto_Dimensions__end(this); }

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
  int32_t elem_type() const { return g_host->TypeProto_Tensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_Tensor)
};

struct TypeProto final {
  const TypeProto_Tensor& tensor_type() const { return g_host->TypeProto__tensor_type(this); }
  TypeProto_Tensor* mutable_tensor_type() { return g_host->TypeProto__mutable_tensor_type(this); }

  enum ValueCase {
    kTensorType = 1,
    kSequenceType = 4,
    kMapType = 5,
    kSparseTensorType = 8,
    kOpaqueType = 7,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(g_host->TypeProto__value_case(this)); }

  PROVIDER_DISALLOW_ALL(TypeProto)
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

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

namespace utils {
bool IsDataTypeString(MLDataType dt_type);

}  // namespace utils

namespace Utils {

struct DataTypeUtils final {
  static const std::string* ToType(const ONNX_NAMESPACE::TypeProto& type_proto) { return g_host->Utils__DataTypeUtils__ToType(type_proto); }

  PROVIDER_DISALLOW_ALL(DataTypeUtils)
};

}  // namespace Utils

struct ComputeCapability final {
  static std::unique_ptr<ComputeCapability> Create(std::unique_ptr<IndexedSubGraph> t_sub_graph) { return g_host->ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { g_host->ComputeCapability__operator_delete(reinterpret_cast<ComputeCapability*>(p)); }

  std::unique_ptr<IndexedSubGraph>& SubGraph() { return g_host->ComputeCapability__SubGraph(this); }

  ComputeCapability() = delete;
  ComputeCapability(const ComputeCapability&) = delete;
  void operator=(const ComputeCapability&) = delete;
};

struct DataTransferManager final {
  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const { return g_host->DataTransferManager__CopyTensor(this, src, dst, exec_queue_id); }
  Status CopyTensor(const Tensor& src, Tensor& dst) const { return g_host->DataTransferManager__CopyTensor(this, src, dst); }

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
  std::vector<std::string>& outputs() { return g_host->IndexedSubGraph_MetaDef__outputs(this); }
  NodeAttributes& attributes() { return g_host->IndexedSubGraph_MetaDef__attributes(this); }

  std::string& doc_string() { return g_host->IndexedSubGraph_MetaDef__doc_string(this); }

  IndexedSubGraph_MetaDef() = delete;
  IndexedSubGraph_MetaDef(const IndexedSubGraph_MetaDef&) = delete;
  void operator=(const IndexedSubGraph_MetaDef&) = delete;
};

struct IndexedSubGraph final {
  static std::unique_ptr<IndexedSubGraph> Create() { return g_host->IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph__operator_delete(reinterpret_cast<IndexedSubGraph*>(p)); }

  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) { return g_host->IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const IndexedSubGraph_MetaDef*>(g_host->IndexedSubGraph__GetMetaDef(this)); }

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

  Status TryFindKernel(const Node& node, ProviderType exec_provider, const KernelCreateInfo** out) const { return g_host->KernelRegistry__TryFindKernel(this, node, exec_provider, out); }

  KernelRegistry() = delete;
  KernelRegistry(const KernelRegistry&) = delete;
  void operator=(const KernelRegistry&) = delete;
};

struct PrimitiveDataTypeBase final {
  int32_t GetDataType() const { return g_host->PrimitiveDataTypeBase__GetDataType(this); }

  PROVIDER_DISALLOW_ALL(PrimitiveDataTypeBase)
};

class DataTypeImpl final {
 public:
  size_t Size() const { return g_host->DataTypeImpl__Size(this); }

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
  static MLDataType GetTypeFromOnnxType(int);

  bool IsTensorType() const { return g_host->DataTypeImpl__IsTensorType(this); }
  bool IsTensorSequenceType() const { return g_host->DataTypeImpl__IsTensorSequenceType(this); }
  bool IsSparseTensorType() const { return g_host->DataTypeImpl__IsSparseTensorType(this); }
  DeleteFunc GetDeleteFunc() const { return g_host->DataTypeImpl__GetDeleteFunc(this); }

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorTypes() { return g_host->DataTypeImpl__AllTensorTypes(); }
  static const std::vector<MLDataType>& AllIEEEFloatTensorTypes() { return g_host->DataTypeImpl__AllIEEEFloatTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllSequenceTensorTypes() { return g_host->DataTypeImpl__AllSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeSequenceTensorTypes(); }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const { return g_host->DataTypeImpl__AsPrimitiveDataType(this); }

  static const char* ToString(MLDataType type) { return g_host->DataTypeImpl__ToString(type); }

  PROVIDER_DISALLOW_ALL(DataTypeImpl)
};

struct Function final {
  const Graph& Body() const { return g_host->Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Function)
};

struct Node final {
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

  void ToProto(ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) const { return g_host->Node__ToProto(this, proto, update_subgraphs); }

  const NodeAttributes& GetAttributes() const noexcept { return g_host->Node__GetAttributes(this); }
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
    const Node__EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Node__EdgeIterator> impl_;
  };

  EdgeConstIterator OutputEdgesBegin() const noexcept { return g_host->Node__OutputEdgesBegin(this); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return g_host->Node__OutputEdgesEnd(this); }

  void ForEachDef(std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs = false) const { g_host->Node__ForEachDef(this, func, std::move(include_missing_optional_defs)); }

  PROVIDER_DISALLOW_ALL(Node)
};

struct NodeArg final {
  const std::string& Name() const noexcept { return g_host->NodeArg__Name(this); }
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const { return g_host->NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->NodeArg__Type(this); }
  const NodeArgInfo& ToProto() const noexcept { return g_host->NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->NodeArg__Exists(this); }
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept { return g_host->NodeArg__TypeAsProto(this); }

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

  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> begin() const { return g_host->NodeAttributes__begin(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> end() const { return g_host->NodeAttributes__end(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> find(const std::string& key) const { return g_host->NodeAttributes__find(this, key); }
  void insert(const NodeAttributes& v) { return g_host->NodeAttributes__insert(this, v); }

  NodeAttributes() = delete;
  NodeAttributes(const NodeAttributes&) = delete;
};

struct Model final {
  static void operator delete(void* p) { g_host->Model__operator_delete(reinterpret_cast<Model*>(p)); }

  Graph& MainGraph() { return g_host->Model__MainGraph(this); }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToProto() { return g_host->Model__ToProto(this); }

  Model() = delete;
  Model(const Model&) = delete;
  void operator=(const Model&) = delete;
};

struct Graph final {
  std::unique_ptr<GraphViewer> CreateGraphViewer() const { return g_host->Graph__CreateGraphViewer(this); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> ToGraphProto() const { return g_host->Graph__ToGraphProto(this); }

  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) { return g_host->Graph__GetOrCreateNodeArg(this, name, p_arg_type); }

  Status Resolve() { return g_host->Graph__Resolve(this); }
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor) { return g_host->Graph__AddInitializedTensor(this, tensor); }
  Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) { return g_host->Graph__AddNode(this, name, op_type, description, input_args, output_args, attributes, domain); }

  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->Graph__GetOutputs(this); }
  void SetOutputs(const std::vector<const NodeArg*>& outputs) { return g_host->Graph__SetOutputs(this, outputs); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->Graph__GetInputs(this); }

  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->Graph__GetInitializedTensor(this, tensor_name, value); }

  PROVIDER_DISALLOW_ALL(Graph)
};

struct GraphViewer final {
  static void operator delete(void* p) { g_host->GraphViewer__operator_delete(reinterpret_cast<GraphViewer*>(p)); }

  std::unique_ptr<Model> CreateModel(const logging::Logger& logger) const { return g_host->GraphViewer__CreateModel(this, logger); }

  const std::string& Name() const noexcept { return g_host->GraphViewer__Name(this); }
  const Path& ModelPath() const noexcept { return g_host->GraphViewer__ModelPath(this); }

  const Node* GetNode(NodeIndex node_index) const { return g_host->GraphViewer__GetNode(this, node_index); }
  const NodeArg* GetNodeArg(const std::string& name) const { return g_host->GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return g_host->GraphViewer__IsSubgraph(this); }
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const { return g_host->GraphViewer__IsConstantInitializer(this, name, check_outer_scope); }

  int NumberOfNodes() const noexcept { return g_host->GraphViewer__NumberOfNodes(this); }
  int MaxNodeIndex() const noexcept { return g_host->GraphViewer__MaxNodeIndex(this); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->GraphViewer__GetInputs(this); }
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->GraphViewer__GetOutputs(this); }
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept { return g_host->GraphViewer__GetValueInfo(this); }

  const InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return g_host->GraphViewer__GetNodesInTopologicalOrder(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return g_host->GraphViewer__GetInputsIncludingInitializers(this); }

  GraphViewer() = delete;
  GraphViewer(const GraphViewer&) = delete;
  void operator=(const GraphViewer&) = delete;
};

struct Path final {
  PathString ToPathString() const noexcept { return g_host->Path__ToPathString(this); }

  PROVIDER_DISALLOW_ALL(Path)
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
  int OutputCount() const { return g_host->OpKernelContext__OutputCount(this); }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const { return g_host->OpKernelContext__GetTempSpaceAllocator(this, output); }

  bool GetUseDeterministicCompute() const { return g_host->OpKernelContext__GetUseDeterministicCompute(this); }

  bool TryGetInferredOutputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredOutputShape(this, index, shape); }
  bool TryGetInferredInputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredInputShape(this, index, shape); }

  PROVIDER_DISALLOW_ALL(OpKernelContext)
};

template <>
inline const Tensor* OpKernelContext::Input<Tensor>(int index) const {
  return g_host->OpKernelContext__Input_Tensor(this, index);
}

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

  AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const { return g_host->OpKernelInfo__GetAllocator(this, device_id, mem_type); }

  const IExecutionProvider* GetExecutionProvider() const noexcept { return g_host->OpKernelInfo__GetExecutionProvider(this); }

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

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const { return g_host->OpKernelInfo__TryGetConstantInput(this, input_index, constant_input_value); }

  const DataTransferManager& GetDataTransferManager() const noexcept { return g_host->OpKernelInfo__GetDataTransferManager(this); }
  const KernelDef& GetKernelDef() const { return g_host->OpKernelInfo__GetKernelDef(this); }

  uint32_t GetInputCount() const { return g_host->OpKernelInfo__GetInputCount(this); }
  uint32_t GetOutputCount() const { return g_host->OpKernelInfo__GetOutputCount(this); }

  const Node& node() const noexcept { return g_host->OpKernelInfo__node(this); }

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

class SessionState {
 public:
  const DataTransferManager& GetDataTransferMgr() const noexcept { return g_host->SessionState__GetDataTransferMgr(this); }

  PROVIDER_DISALLOW_ALL(SessionState)
};

struct Tensor final {
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) { return g_host->Tensor__construct(p_type, shape, allocator); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset = 0) { return g_host->Tensor__construct(p_type, shape, p_data, alloc, offset); }

  static void operator delete(void* p) { g_host->Tensor__operator_delete(reinterpret_cast<Tensor*>(p)); }

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

  template <class T>
  bool IsDataType() const;

  Tensor() = delete;
  Tensor(const Tensor&) = delete;
  void operator=(const Tensor&) = delete;
};

template <>
inline bool Tensor::IsDataType<bool>() const { return g_host->Tensor__IsDataType_bool(this); }
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
inline bool* Tensor::MutableData<bool>() { return g_host->Tensor__MutableData_bool(this); }
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

template <>
inline const bool* Tensor::Data<bool>() const { return g_host->Tensor__Data_bool(this); }
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

//TensorSeq
struct TensorSeq final {
  MLDataType DataType() const noexcept { return g_host->TensorSeq__DataType(this); }
  void SetType(MLDataType elem_type) { g_host->TensorSeq__SetType(this, elem_type); }
  size_t Size() const noexcept { return g_host->TensorSeq__Size(this); }
  const Tensor& Get(size_t i) const { return g_host->TensorSeq__Get(this, i); }
  void Add(Tensor&& tensor) { g_host->TensorSeq__Add(this, std::move(tensor)); }
};

template <>
inline gsl::span<const int64_t> Tensor::DataAsSpan() const { return g_host->Tensor__DataAsSpan_int64(this); }

#endif  // SHARED_PROVIDER

}  // namespace onnxruntime
