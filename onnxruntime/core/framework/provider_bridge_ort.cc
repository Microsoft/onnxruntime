// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_types.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_bridge_ort.h"
#include "core/framework/provider_shutdown.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/util/math.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/provider_options.h"
#include "core/common/string_helper.h"


#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/random_generator.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/providers/cpu/tensor/gather_elements.h"
#include "core/providers/cpu/tensor/unsqueeze.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/split.h"
#include "core/providers/cpu/tensor/size.h"
#include "core/providers/cpu/tensor/scatter_nd.h"
#include "core/providers/cpu/tensor/padbase.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/providers/cpu/tensor/onehot.h"
#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cpu/math/einsum.h"
#include "core/providers/cpu/math/cumsum.h"
#include "core/providers/cpu/object_detection/non_max_suppression.h"
#include "core/providers/cpu/object_detection/roialign.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/cpu/bert/embed_layer_norm_helper.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"
namespace onnxruntime {
namespace contrib {
Status LongformerAttentionBase__CheckInputs(const LongformerAttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const TensorShape& mask_shape, const TensorShape& global_weights_shape, const TensorShape& global_bias_shape, const TensorShape& global_shape);
}
}  // namespace onnxruntime
#include "contrib_ops/cpu/bert/attention_base.h"
#endif

#ifdef ENABLE_TRAINING
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"
#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "orttraining/training_ops/cpu/controlflow/record.h"
#include "orttraining/training_ops/cpu/controlflow/wait.h"
#include "orttraining/training_ops/cpu/controlflow/yield.h"
#include "orttraining/training_ops/cpu/loss/softmax_cross_entropy_loss.h"
#include "orttraining/training_ops/cpu/tensor/split.h"

#ifdef ENABLE_TRAINING_TORCH_INTEROP
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"
#endif

#endif

#if defined(USE_CUDA) && defined(ORT_USE_NCCL)
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "orttraining/core/framework/distributed_run_context.h"
#endif

namespace ONNX_NAMESPACE {
// We use these names in the provider API because we don't have the protobuf definitions of the RepeatedField* types
using int64s = google::protobuf::RepeatedField<int64_t>;
using TensorProtos = google::protobuf::RepeatedPtrField<TensorProto>;
using TensorShapeProto_Dimensions = google::protobuf::RepeatedPtrField<TensorShapeProto_Dimension>;
using ValueInfoProtos = google::protobuf::RepeatedPtrField<ValueInfoProto>;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
using IndexedSubGraph_MetaDef = IndexedSubGraph::MetaDef;
}  // namespace onnxruntime

#include "core/common/cpuid_info.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/providers/shared_library/provider_interfaces.h"

#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/platform/tensorrt_provider_options.h"

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ".dll"
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace onnxruntime {

ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
ProviderInfo_CUDA& GetProviderInfo_CUDA();

struct TensorShapeProto_Dimension_Iterator_Impl : TensorShapeProto_Dimension_Iterator {
  TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& operator*() override { return *v_; }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

struct NodeAttributes_Iterator_Impl : NodeAttributes_Iterator {
  NodeAttributes_Iterator_Impl(NodeAttributes::const_iterator&& v) : v_{std::move(v)} {}

  bool operator!=(const NodeAttributes_Iterator& p) const override { return v_ != static_cast<const NodeAttributes_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>& operator*() const override { return v_.operator*(); }

  const std::string& first() const override { return v_->first; }
  const ONNX_NAMESPACE::AttributeProto& second() const override { return v_->second; }

  NodeAttributes::const_iterator v_;
};

struct Node__NodeIterator_Impl : Node__NodeIterator {
  Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__NodeIterator& p) const override { return v_ != static_cast<const Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& operator*() override { return *v_; }

  Node::NodeConstIterator v_;
};

struct Node__EdgeIterator_Impl : Node__EdgeIterator {
  Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__EdgeIterator& p) const override { return v_ != static_cast<const Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& GetNode() const override { return v_->GetNode(); }
  int GetSrcArgIndex() const override { return v_->GetSrcArgIndex(); }
  int GetDstArgIndex() const override { return v_->GetDstArgIndex(); }

  Node::EdgeConstIterator v_;
};

// wrapped = The internal object is exposed as an opaque pointer, so we wrap it in a class that forwards every call to the real calls. No members are ever directly accessed
// direct = Same implementation is used for shared providers & core code, but some of the methods need to be routed through here to make the linker happy
struct ProviderHostImpl : ProviderHost {
  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete[] reinterpret_cast<uint8_t*>(p); }

  logging::Logger* LoggingManager_GetDefaultLogger() override {
    return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
  }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  std::vector<std::string> GetStackTrace() override { return onnxruntime::GetStackTrace(); }

  OrtStatus* CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept override { return OrtApis::CreateStatus(code, msg); }

  AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) override { return onnxruntime::CreateAllocator(info); }
  std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) override { return std::make_unique<CPUAllocator>(memory_info); };

  void* CPUAllocator__Alloc(CPUAllocator* p, size_t size) override { return p->CPUAllocator::Alloc(size); }
  void CPUAllocator__Free(CPUAllocator* p, void* allocation) override { return p->CPUAllocator::Free(allocation); }

#ifdef USE_CUDA
  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override { return GetProviderInfo_CUDA().CreateCUDAAllocator(device_id, name); }
  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override { return GetProviderInfo_CUDA().CreateCUDAPinnedAllocator(device_id, name); }
  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) override { return GetProviderInfo_CUDA().CreateGPUDataTransfer(stream); }

  void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }
  void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }

  bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return GetProviderInfo_CUDA().CudaCall_false(retCode, exprString, libName, successCode, msg); }
  bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return GetProviderInfo_CUDA().CudaCall_true(retCode, exprString, libName, successCode, msg); }
#endif

  std::string GetEnvironmentVar(const std::string& var_name) override { return Env::Default().GetEnvironmentVar(var_name); }

  std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                     const std::string& provider_type,
                                                     const std::vector<const KernelRegistry*>& kernel_registries,
                                                     const std::vector<NodeIndex>& tentative_nodes) override {
    return onnxruntime::GetCpuPreferredNodes(graph, provider_type, kernel_registries, tentative_nodes);
  }

  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ bool* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ float* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ double* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ MLFloat16* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int8_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint8_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int16_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint16_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int32_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint32_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int64_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint64_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }

  uint16_t math__floatToHalf(float f) override { return math::floatToHalf(f); }
  float math__halfToFloat(uint16_t h) override { return math::halfToFloat(h); }

  // IAllocator (direct)
  bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) override { return IAllocator::CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out); }

  // IExecutionProvider (direct)
  AllocatorPtr IExecutionProvider__GetAllocator(const IExecutionProvider* p, int id, OrtMemType mem_type) override { return p->IExecutionProvider::GetAllocator(id, mem_type); }
  void IExecutionProvider__InsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) override { return p->IExecutionProvider::InsertAllocator(allocator); }
  void IExecutionProvider__TryInsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) override { return p->IExecutionProvider::TryInsertAllocator(allocator); }
  std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
                                                                                    const std::vector<const KernelRegistry*>& kernel_registries) override { return p->IExecutionProvider::GetCapability(graph_viewer, kernel_registries); }
  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return p->IExecutionProvider::Compile(fused_nodes, node_compute_funcs);
  }

  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::string& dll_path) override {
    return p->IExecutionProvider::Compile(fused_nodes, dll_path);
  }

  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return p->IExecutionProvider::Compile(fused_nodes_and_graphs, node_compute_funcs);
  }

  int IExecutionProvider__GenerateMetaDefId(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) override {
    return p->IExecutionProvider::GenerateMetaDefId(graph_viewer, model_hash);
  }

  void IExecutionProvider__RegisterAllocator(IExecutionProvider* p, std::shared_ptr<AllocatorManager> allocator_manager) override {
    return p->IExecutionProvider::RegisterAllocator(allocator_manager);
  }

  // Status (direct)
  std::string Status__ToString(const Status* p) override { return p->Status::ToString(); }

  // TensorShape (direct)
  int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) override { return p->TensorShape::SizeHelper(start, end); }
  std::string TensorShape__ToString(const TensorShape* p) override { return p->TensorShape::ToString(); }
  int64_t TensorShape__SizeToDimension(const TensorShape* p, size_t dimension) override { return p->TensorShape::SizeToDimension(dimension); }
  int64_t TensorShape__SizeFromDimension(const TensorShape* p, size_t dimension) override { return p->TensorShape::SizeFromDimension(dimension); }
  std::ostream& operator_left_shift(std::ostream& out, const TensorShape& shape) override { return out << shape; }

  // CPUIDInfo (wrapped)
  const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() override { return CPUIDInfo::GetCPUIDInfo(); }
  bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) override { return p->HasAVX2(); }
  bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) override { return p->HasAVX512f(); }

  // logging::Logger (wrapped)
  bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) override { return p->OutputIsEnabled(severity, data_type); }

  // logging::LoggingManager (wrapped)
  const logging::Logger& logging__LoggingManager__DefaultLogger() override { return logging::LoggingManager::DefaultLogger(); }

  // logging::Capture (wrapped)
  std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) override {
    return std::make_unique<logging::Capture>(logger, severity, category, dataType, location);
  }
  void logging__Capture__operator_delete(logging::Capture* p) noexcept override { delete p; }
  std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept override { return p->Stream(); }

  // Utils::DataTypeUtils (wrapped)
  const std::string* Utils__DataTypeUtils__ToType(const ONNX_NAMESPACE::TypeProto& type_proto) override { return ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(type_proto); }

  // int64s (wrapped)
  int int64s__size(const ONNX_NAMESPACE::int64s* p) override { return p->size(); }
  const int64_t& int64s__Get(const ONNX_NAMESPACE::int64s* p, int index) override { return p->Get(index); }

  // TypeProto_Tensor (wrapped)
  bool TypeProto_Tensor__has_shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->has_shape(); }
  const ONNX_NAMESPACE::TensorShapeProto& TypeProto_Tensor__shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->shape(); }
  ONNX_NAMESPACE::TensorShapeProto* TypeProto_Tensor__mutable_shape(ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->mutable_shape(); }
  int32_t TypeProto_Tensor__elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->elem_type(); }

  // TypeProto (wrapped)
  const ONNX_NAMESPACE::TypeProto_Tensor& TypeProto__tensor_type(const ONNX_NAMESPACE::TypeProto* p) override { return p->tensor_type(); }
  ONNX_NAMESPACE::TypeProto_Tensor* TypeProto__mutable_tensor_type(ONNX_NAMESPACE::TypeProto* p) override { return p->mutable_tensor_type(); }
  int TypeProto__value_case(const ONNX_NAMESPACE::TypeProto* p) override { return p->value_case(); }

  // AttributeProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::AttributeProto> AttributeProto__construct() override { return std::make_unique<ONNX_NAMESPACE::AttributeProto>(); }
  void AttributeProto__operator_delete(ONNX_NAMESPACE::AttributeProto* p) override { delete p; }
  void AttributeProto__operator_assign(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::AttributeProto& v) override { *p = v; }

  ONNX_NAMESPACE::AttributeProto_AttributeType AttributeProto__type(const ONNX_NAMESPACE::AttributeProto* p) override { return p->type(); }
  int AttributeProto__ints_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->ints_size(); }
  int AttributeProto__floats_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->floats_size(); }
  int AttributeProto__strings_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->strings_size(); }
  int64_t AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->ints(i); }
  float AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->floats(i); }
  const std::string& AttributeProto__strings(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->strings(i); }
  const ONNX_NAMESPACE::int64s& AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p) override { return p->ints(); }
  int64_t AttributeProto__i(const ONNX_NAMESPACE::AttributeProto* p) override { return p->i(); }
  float AttributeProto__f(const ONNX_NAMESPACE::AttributeProto* p) override { return p->f(); }
  void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) override { return p->set_s(value); }
  const ::std::string& AttributeProto__s(const ONNX_NAMESPACE::AttributeProto* p) override { return p->s(); }
  void AttributeProto__set_name(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) override { return p->set_name(value); }
  void AttributeProto__set_type(ONNX_NAMESPACE::AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) override { return p->set_type(value); }
  ONNX_NAMESPACE::TensorProto* AttributeProto__add_tensors(ONNX_NAMESPACE::AttributeProto* p) override { return p->add_tensors(); }

  // GraphProto (wrapped)
  void GraphProto__operator_delete(ONNX_NAMESPACE::GraphProto* p) override { delete p; }

  const ONNX_NAMESPACE::ValueInfoProto& GraphProto__input(const ONNX_NAMESPACE::GraphProto* p, int index) override { return p->input(index); }
  ONNX_NAMESPACE::ValueInfoProto* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p, int index) override { return p->mutable_input(index); }
  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_input(); }
  int GraphProto__input_size(const ONNX_NAMESPACE::GraphProto* p) override { return p->input_size(); }

  const ONNX_NAMESPACE::ValueInfoProtos& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p) override { return p->output(); }
  const ONNX_NAMESPACE::ValueInfoProto& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p, int index) override { return p->output(index); }
  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_output(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_output(); }

  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_value_info(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_value_info(); }
  ONNX_NAMESPACE::TensorProtos* GraphProto__mutable_initializer(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_initializer(); }
  ONNX_NAMESPACE::NodeProto* GraphProto__add_node(ONNX_NAMESPACE::GraphProto* p) override { return p->add_node(); }

  void GraphProto__operator_assign(ONNX_NAMESPACE::GraphProto* p, const ONNX_NAMESPACE::GraphProto& v) override { *p = v; }

  // ModelProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ModelProto__construct() override { return std::make_unique<ONNX_NAMESPACE::ModelProto>(); }
  void ModelProto__operator_delete(ONNX_NAMESPACE::ModelProto* p) override { delete p; }

  bool ModelProto__SerializeToString(const ONNX_NAMESPACE::ModelProto* p, std::string& string) override { return p->SerializeToString(&string); }
  bool ModelProto__SerializeToOstream(const ONNX_NAMESPACE::ModelProto* p, std::ostream& output) override { return p->SerializeToOstream(&output); }
  bool ModelProto__ParseFromString(ONNX_NAMESPACE::ModelProto* p, const std::string& data) override { return p->ParseFromString(data); }
  std::string ModelProto__SerializeAsString(const ONNX_NAMESPACE::ModelProto* p) override { return p->SerializeAsString(); }

  const ONNX_NAMESPACE::GraphProto& ModelProto__graph(const ONNX_NAMESPACE::ModelProto* p) override { return p->graph(); }
  ONNX_NAMESPACE::GraphProto* ModelProto__mutable_graph(ONNX_NAMESPACE::ModelProto* p) override { return p->mutable_graph(); }

  void ModelProto__set_ir_version(ONNX_NAMESPACE::ModelProto* p, int64_t value) override { p->set_ir_version(value); }

  // TensorProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::TensorProto> TensorProto__construct() override { return std::make_unique<ONNX_NAMESPACE::TensorProto>(); }
  void TensorProto__operator_delete(ONNX_NAMESPACE::TensorProto* p) override { delete p; }
  void TensorProto__operator_assign(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto& v) override { *p = v; }
  bool TensorProto__has_name(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_name(); }
  int TensorProto__dims_size(const ONNX_NAMESPACE::TensorProto* p) override { return p->dims_size(); }
  const ONNX_NAMESPACE::int64s& TensorProto__dims(const ONNX_NAMESPACE::TensorProto* p) override { return p->dims(); }
  bool TensorProto__has_data_location(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_data_location(); }
  int TensorProto__data_location(const ONNX_NAMESPACE::TensorProto* p) override { return p->data_location(); }
  bool TensorProto__has_raw_data(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_raw_data(); }
  const std::string& TensorProto__raw_data(const ONNX_NAMESPACE::TensorProto* p) override { return p->raw_data(); }
  int32_t TensorProto__data_type(const ONNX_NAMESPACE::TensorProto* p) override { return p->data_type(); }

  bool TensorProto_DataType_IsValid(int value) override { return ONNX_NAMESPACE::TensorProto::DataType_IsValid(value); }

  // TensorProtos (wrapped)
  ONNX_NAMESPACE::TensorProto* TensorProtos__Add(ONNX_NAMESPACE::TensorProtos* p) override { return p->Add(); }

  // TensorShapeProto_Dimension (wrapped)
  int TensorShapeProto_Dimension__value_case(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->value_case(); }
  const std::string& TensorShapeProto_Dimension__dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->dim_param(); }
  int64_t TensorShapeProto_Dimension__dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->dim_value(); }
  void TensorShapeProto_Dimension__clear_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->clear_dim_value(); }
  void TensorShapeProto_Dimension__set_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, int64_t value) override { return p->set_dim_value(value); }
  bool TensorShapeProto_Dimension__has_dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->has_dim_value(); }
  bool TensorShapeProto_Dimension__has_dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->has_dim_param(); }

  // TensorShapeProto_Dimensions (wrapped)
  std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__begin(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) override {
    return std::make_unique<TensorShapeProto_Dimension_Iterator_Impl>(p->begin());
  }

  std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__end(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) override {
    return std::make_unique<TensorShapeProto_Dimension_Iterator_Impl>(p->end());
  }

  // TensorShapeProto (wrapped)
  int TensorShapeProto__dim_size(const ONNX_NAMESPACE::TensorShapeProto* p) override { return p->dim_size(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimensions& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p) override { return p->dim(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p, int index) override { return p->dim(index); }
  ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__mutable_dim(ONNX_NAMESPACE::TensorShapeProto* p, int index) override { return p->mutable_dim(index); }
  void TensorShapeProto__clear_dim(ONNX_NAMESPACE::TensorShapeProto* p) override { return p->clear_dim(); }
  ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__add_dim(ONNX_NAMESPACE::TensorShapeProto* p) override { return p->add_dim(); }

  // ValueInfoProto (wrapped)
  const ONNX_NAMESPACE::TypeProto& ValueInfoProto__type(const ONNX_NAMESPACE::ValueInfoProto* p) override { return p->type(); }
  ONNX_NAMESPACE::TypeProto* ValueInfoProto__mutable_type(ONNX_NAMESPACE::ValueInfoProto* p) override { return p->mutable_type(); }
  virtual void ValueInfoProto__operator_assign(ONNX_NAMESPACE::ValueInfoProto* p, const ONNX_NAMESPACE::ValueInfoProto& v) override { *p = v; }

  // ValueInfoProtos (wrapped)
  ONNX_NAMESPACE::ValueInfoProto* ValueInfoProtos__Add(ONNX_NAMESPACE::ValueInfoProtos* p) override { return p->Add(); }

  const ONNX_NAMESPACE::ValueInfoProto& ValueInfoProtos__operator_array(const ONNX_NAMESPACE::ValueInfoProtos* p, int index) override { return (*p)[index]; }

  // ComputeCapability (wrapped)
  std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) override { return std::make_unique<ComputeCapability>(std::move(t_sub_graph)); }
  void ComputeCapability__operator_delete(ComputeCapability* p) override { delete p; }
  std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) override { return p->sub_graph; }

  // DataTransferManager (wrapped)
  Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst, int exec_queue_id) override { return p->CopyTensor(src, dst, exec_queue_id); }
  Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst) override { return p->CopyTensor(src, dst); }
  const IDataTransfer* DataTransferManager__GetDataTransfer(const DataTransferManager* p, const OrtDevice& src_device, const OrtDevice& dst_device) override { return p->GetDataTransfer(src_device, dst_device); }

  // IDataTransfer (direct)
  Status IDataTransfer__CopyTensor(const IDataTransfer* p, const Tensor& src, Tensor& dst) override { return p->IDataTransfer::CopyTensor(src, dst); }
  Status IDataTransfer__CopyTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) override { return p->IDataTransfer::CopyTensors(src_dst_pairs); }

  // IndexedSubGraph_MetaDef (wrapped)
  std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() override { return std::make_unique<IndexedSubGraph::MetaDef>(); }
  void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) override { delete p; }

  std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) override { return p->name; }
  std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) override { return p->domain; }
  int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) override { return p->since_version; }
  ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) override { return p->status; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) override { return p->inputs; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) override { return p->outputs; }
  NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) override { return p->attributes; }
  std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) override { return p->doc_string; }

  // IndexedSubGraph (wrapped)
  std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() override { return std::make_unique<IndexedSubGraph>(); }
  void IndexedSubGraph__operator_delete(IndexedSubGraph* p) override { delete p; }

  std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) override { return p->nodes; }

  void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) override { return p->SetMetaDef(std::move(meta_def_)); }
  const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) override { return p->GetMetaDef(); }

  // KernelDef (wrapped)
  void KernelDef__operator_delete(KernelDef* p) override { delete p; }
  void KernelDef__SinceVersion(const KernelDef* p, int* start, int* end) override { return p->SinceVersion(start, end); }
  const std::string& KernelDef__Domain(const KernelDef* p) override { return p->Domain(); }
  const std::string& KernelDef__OpName(const KernelDef* p) override { return p->OpName(); }
  int KernelDef__ExecQueueId(const KernelDef* p) override { return p->ExecQueueId(); }

  // KernelDefBuilder (wrapped)
  std::unique_ptr<KernelDefBuilder> KernelDefBuilder__construct() override { return std::make_unique<KernelDefBuilder>(); }
  void KernelDefBuilder__operator_delete(KernelDefBuilder* p) override { delete p; }

  void KernelDefBuilder__SetName(KernelDefBuilder* p, const char* op_name) override { p->SetName(op_name); }
  void KernelDefBuilder__SetDomain(KernelDefBuilder* p, const char* domain) override { p->SetDomain(domain); }
  void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version) override { p->SinceVersion(since_version); }
  void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version_start, int since_version_end) override { p->SinceVersion(since_version_start, since_version_end); }
  void KernelDefBuilder__Provider(KernelDefBuilder* p, const char* provider_type) override { p->Provider(provider_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { p->TypeConstraint(arg_name, supported_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { p->TypeConstraint(arg_name, supported_types); }
  void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->InputMemoryType(type, input_index); }
  void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, const std::vector<int>& input_indexes) override { p->InputMemoryType(type, input_indexes); }
  void KernelDefBuilder__OutputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->OutputMemoryType(type, input_index); }
  void KernelDefBuilder__ExecQueueId(KernelDefBuilder* p, int queue_id) override { p->ExecQueueId(queue_id); }
  void KernelDefBuilder__MayInplace(KernelDefBuilder* p, int input_index, int output_index) override { p->MayInplace(input_index, output_index); }
  void KernelDefBuilder__Alias(KernelDefBuilder* p, int input_index, int output_index) override { p->Alias(input_index, output_index); }
  void KernelDefBuilder__Alias(KernelDefBuilder* p, const std::vector<std::pair<int, int>>& aliases) override { p->Alias(aliases); }
  void KernelDefBuilder__VariadicAlias(KernelDefBuilder* p, int input_offset, int output_offset) override { p->VariadicAlias(input_offset, output_offset); }
  void KernelDefBuilder__ExternalOutputs(KernelDefBuilder* p) override { p->ExternalOutputs(); }
  void KernelDefBuilder__AllocateInputsContiguously(KernelDefBuilder* p) override { p->AllocateInputsContiguously(); }

  std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) override { return p->Build(); }

  // KernelRegistry (wrapped)
  std::shared_ptr<KernelRegistry> KernelRegistry__construct() override { return std::make_shared<KernelRegistry>(); }
  void KernelRegistry__operator_delete(KernelRegistry* p) override { delete p; }
  Status KernelRegistry__Register(KernelRegistry* p, KernelCreateInfo&& create_info) override { return p->Register(std::move(create_info)); }

  Status KernelRegistry__TryFindKernel(const KernelRegistry* p, const Node& node, ProviderType exec_provider, const KernelCreateInfo** out) override {
    return p->TryFindKernel(node, exec_provider, out);
  }

  // PrimitiveDataTypeBase (wrapped)
  int32_t PrimitiveDataTypeBase__GetDataType(const PrimitiveDataTypeBase* p) override { return p->GetDataType(); }

  // DataTypeImpl (wrapped)
  MLDataType DataTypeImpl__GetType_Tensor() override { return DataTypeImpl::GetType<Tensor>(); }
  MLDataType DataTypeImpl__GetType_TensorSeq () override { return DataTypeImpl::GetType<TensorSeq>(); }
  MLDataType DataTypeImpl__GetTypeFromOnnxType (int onnx_type) override { return DataTypeImpl::TensorTypeFromONNXEnum(onnx_type)->GetElementType(); }
  MLDataType DataTypeImpl__GetType_bool() override { return DataTypeImpl::GetType<bool>(); }
  MLDataType DataTypeImpl__GetType_int8() override { return DataTypeImpl::GetType<int8_t>(); }
  MLDataType DataTypeImpl__GetType_uint8() override { return DataTypeImpl::GetType<uint8_t>(); }
  MLDataType DataTypeImpl__GetType_int16() override { return DataTypeImpl::GetType<int16_t>(); }
  MLDataType DataTypeImpl__GetType_uint16() override { return DataTypeImpl::GetType<uint16_t>(); }
  MLDataType DataTypeImpl__GetType_int32() override { return DataTypeImpl::GetType<int32_t>(); }
  MLDataType DataTypeImpl__GetType_uint32() override { return DataTypeImpl::GetType<uint32_t>(); }
  MLDataType DataTypeImpl__GetType_int64() override { return DataTypeImpl::GetType<int64_t>(); }
  MLDataType DataTypeImpl__GetType_uint64() override { return DataTypeImpl::GetType<uint64_t>(); }
  MLDataType DataTypeImpl__GetType_float() override { return DataTypeImpl::GetType<float>(); }
  MLDataType DataTypeImpl__GetType_double() override { return DataTypeImpl::GetType<double>(); }
  MLDataType DataTypeImpl__GetType_BFloat16() override { return DataTypeImpl::GetType<BFloat16>(); }
  MLDataType DataTypeImpl__GetType_MLFloat16() override { return DataTypeImpl::GetType<MLFloat16>(); }
  MLDataType DataTypeImpl__GetType_string() override { return DataTypeImpl::GetType<std::string>(); }
  MLDataType DataTypeImpl__GetTensorType_bool() override { return DataTypeImpl::GetTensorType<bool>(); }
  MLDataType DataTypeImpl__GetTensorType_int8() override { return DataTypeImpl::GetTensorType<int8_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint8() override { return DataTypeImpl::GetTensorType<uint8_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int16() override { return DataTypeImpl::GetTensorType<int16_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint16() override { return DataTypeImpl::GetTensorType<uint16_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int32() override { return DataTypeImpl::GetTensorType<int32_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint32() override { return DataTypeImpl::GetTensorType<uint32_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int64() override { return DataTypeImpl::GetTensorType<int64_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint64() override { return DataTypeImpl::GetTensorType<uint64_t>(); }
  MLDataType DataTypeImpl__GetTensorType_float() override { return DataTypeImpl::GetTensorType<float>(); }
  MLDataType DataTypeImpl__GetTensorType_double() override { return DataTypeImpl::GetTensorType<double>(); }
  MLDataType DataTypeImpl__GetTensorType_BFloat16() override { return DataTypeImpl::GetTensorType<BFloat16>(); }
  MLDataType DataTypeImpl__GetTensorType_MLFloat16() override { return DataTypeImpl::GetTensorType<MLFloat16>(); }
  const char* DataTypeImpl__ToString(MLDataType type) override { return DataTypeImpl::ToString(type); }
  bool DataTypeImpl__IsTensorType(const DataTypeImpl* p) override { return p->IsTensorType(); }
  bool DataTypeImpl__IsTensorSequenceType(const DataTypeImpl* p) override { return p->IsTensorSequenceType(); }
  bool DataTypeImpl__IsSparseTensorType(const DataTypeImpl* p) override { return p->IsSparseTensorType(); }
  DeleteFunc DataTypeImpl__GetDeleteFunc(const DataTypeImpl* p) override { return p->GetDeleteFunc(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypes() override { return DataTypeImpl::AllFixedSizeTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorTypes() override { return DataTypeImpl::AllTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllIEEEFloatTensorTypes() override { return DataTypeImpl::AllIEEEFloatTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypes() override { return DataTypeImpl::AllTensorAndSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes() override { return DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypes() override { return DataTypeImpl::AllSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypes() override { return DataTypeImpl::AllFixedSizeSequenceTensorTypes(); }
  size_t DataTypeImpl__Size(const DataTypeImpl* p) override { return p->Size(); }
  const PrimitiveDataTypeBase* DataTypeImpl__AsPrimitiveDataType(const DataTypeImpl* p) override { return p->AsPrimitiveDataType(); }

  // Function (wrapped)
  const Graph& Function__Body(const Function* p) override { return p->Body(); }

  // Node (wrapped)
  const std::string& Node__Name(const Node* p) noexcept override { return p->Name(); }
  const std::string& Node__Description(const Node* p) noexcept override { return p->Description(); }
  const std::string& Node__Domain(const Node* p) noexcept override { return p->Domain(); }
  const std::string& Node__OpType(const Node* p) noexcept override { return p->OpType(); }
  int Node__SinceVersion(const Node* p) override { return p->SinceVersion(); }

  const Function* Node__GetFunctionBody(const Node* p) noexcept override { return p->GetFunctionBody(); }
  ProviderType Node__GetExecutionProviderType(const Node* p) const noexcept override { return p->GetExecutionProviderType(); }

  const std::vector<int>& Node__InputArgCount(const Node* p) override { return p->InputArgCount(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__ImplicitInputDefs(const Node* p) noexcept override { return p->ImplicitInputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__InputDefs(const Node* p) noexcept override { return p->InputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__OutputDefs(const Node* p) noexcept override { return p->OutputDefs(); }
  NodeIndex Node__Index(const Node* p) noexcept override { return p->Index(); }

  void Node__ToProto(const Node* p, ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) override { p->ToProto(proto, update_subgraphs); }

  const NodeAttributes& Node__GetAttributes(const Node* p) noexcept override { return p->GetAttributes(); }
  size_t Node__GetInputEdgesCount(const Node* p) noexcept override { return p->GetInputEdgesCount(); }
  size_t Node__GetOutputEdgesCount(const Node* p) noexcept override { return p->GetOutputEdgesCount(); }

  std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->InputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->InputNodesEnd()); }

  std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->OutputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->OutputNodesEnd()); }

  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept override { return std::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesBegin()); }
  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept override { return std::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesEnd()); }

  void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) override { p->ForEachDef(func, std::move(include_missing_optional_defs)); }

  // NodeArg (wrapped)
  const std::string& NodeArg__Name(const NodeArg* p) noexcept override { return p->Name(); }
  const ONNX_NAMESPACE::TensorShapeProto* NodeArg__Shape(const NodeArg* p) override { return p->Shape(); }
  ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept override { return p->Type(); }
  const NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept override { return p->ToProto(); }
  bool NodeArg__Exists(const NodeArg* p) const noexcept override { return p->Exists(); }
  const ONNX_NAMESPACE::TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept override { return p->TypeAsProto(); }

  // NodeAttributes (wrapped)
  std::unique_ptr<NodeAttributes> NodeAttributes__construct() override { return std::make_unique<NodeAttributes>(); }
  void NodeAttributes__operator_delete(NodeAttributes* p) noexcept override { delete p; }
  size_t NodeAttributes__size(const NodeAttributes* p) override { return p->size(); }
  void NodeAttributes__clear(NodeAttributes* p) noexcept override { return p->clear(); }
  size_t NodeAttributes__count(const NodeAttributes* p, const std::string& keyval) override { return p->count(keyval); }
  ONNX_NAMESPACE::AttributeProto& NodeAttributes__operator_array(NodeAttributes* p, const std::string& string) override { return (*p)[string]; }
  const ONNX_NAMESPACE::AttributeProto& NodeAttributes__at(const NodeAttributes* p, const std::string& string) override { return p->at(string); }
  void NodeAttributes__operator_assign(NodeAttributes* p, const NodeAttributes& v) override { *p = v; }

  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__begin(const NodeAttributes* p) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->begin());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__end(const NodeAttributes* p) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->end());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__find(const NodeAttributes* p, const std::string& key) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->find(key));
  }
  void NodeAttributes__insert(NodeAttributes* p, const NodeAttributes& v) override { return p->insert(v.begin(), v.end()); }

  // Model (wrapped)
  void Model__operator_delete(Model* p) override { delete p; }
  Graph& Model__MainGraph(Model* p) override { return p->MainGraph(); }
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToProto(Model* p) override { return std::make_unique<ONNX_NAMESPACE::ModelProto>(p->ToProto()); }

  // Graph (wrapped)
  std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) override { return std::make_unique<GraphViewer>(*p); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> Graph__ToGraphProto(const Graph* p) override { return std::make_unique<ONNX_NAMESPACE::GraphProto>(p->ToGraphProto()); }

  NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) override { return p->GetOrCreateNodeArg(name, p_arg_type); }

  Status Graph__Resolve(Graph* p) override { return p->Resolve(); }
  void Graph__AddInitializedTensor(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor) override { p->AddInitializedTensor(tensor); }
  Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) override {
    return p->AddNode(name, op_type, description, input_args, output_args, attributes, domain);
  }

  const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept override { return p->GetOutputs(); }
  void Graph__SetOutputs(Graph* p, const std::vector<const NodeArg*>& outputs) override { p->SetOutputs(outputs); }

  const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept override { return p->GetInputs(); }
  bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  // GraphViewer (wrapped)
  void GraphViewer__operator_delete(GraphViewer* p) override { delete p; }
  std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* graph_viewer, const logging::Logger& logger) override {
    return std::make_unique<Model>(graph_viewer->Name(), true, ModelMetaData(), PathString(),
                                   IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
                                   std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
  }

  const std::string& GraphViewer__Name(const GraphViewer* p) noexcept override { return p->Name(); }
  const Path& GraphViewer__ModelPath(const GraphViewer* p) noexcept override { return p->ModelPath(); }

  const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) override { return p->GetNode(node_index); }
  const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) override { return p->GetNodeArg(name); }

  bool GraphViewer__IsSubgraph(const GraphViewer* p) override { return p->IsSubgraph(); }
  bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) override { return p->IsConstantInitializer(name, check_outer_scope); }
  int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept override { return p->NumberOfNodes(); }
  int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept override { return p->MaxNodeIndex(); }

  const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept override { return p->GetInputs(); }
  const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept override { return p->GetOutputs(); }
  const std::vector<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept override { return p->GetValueInfo(); }

  const InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) override { return p->GetAllInitializedTensors(); }
  bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) override { return p->DomainToVersionMap(); }

  const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p) override { return p->GetNodesInTopologicalOrder(); }
  const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept override { return p->GetInputsIncludingInitializers(); }

  // Path (wrapped)
  PathString Path__ToPathString(const Path* p) noexcept override { return p->ToPathString(); }

  // OpKernel (direct)
  const Node& OpKernel__Node(const OpKernel* p) override { return p->OpKernel::Node(); }

  // OpKernelContext (wrapped)
  const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) override { return p->Input<Tensor>(index); }
  const TensorSeq* OpKernelContext__Input_TensorSeq(const OpKernelContext* p, int index) override { return p->Input<TensorSeq>(index); }
  const Tensor& OpKernelContext__RequiredInput_Tensor(const OpKernelContext* p, int index) override { return p->RequiredInput<Tensor>(index); }
  MLDataType OpKernelContext__InputType(const OpKernelContext* p, int index) override { return p->InputType(index); }
  Tensor* OpKernelContext__Output_Tensor(OpKernelContext* p, int index) override { return p->Output<Tensor>(index); }
  TensorSeq* OpKernelContext__Output_TensorSeq(OpKernelContext* p, int index) override { return p->Output<TensorSeq>(index); }
  Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) override { return p->Output(index, shape); }
  Tensor& OpKernelContext__RequiredOutput(OpKernelContext* p, int index, const TensorShape& shape) override { return p->RequiredOutput(index, shape); }
  int OpKernelContext__InputCount(const OpKernelContext* p) override { return p->InputCount(); }
  int OpKernelContext__OutputCount(const OpKernelContext* p) override { return p->OutputCount(); }
  Status OpKernelContext__GetTempSpaceAllocator(const OpKernelContext* p, AllocatorPtr* output) override { return p->GetTempSpaceAllocator(output); }
  bool OpKernelContext__GetUseDeterministicCompute(const OpKernelContext* p) override { return p->GetUseDeterministicCompute(); }
  bool OpKernelContext__TryGetInferredOutputShape(const OpKernelContext* p, int index, TensorShape& shape) override { return p->TryGetInferredOutputShape(index, shape); }
  bool OpKernelContext__TryGetInferredInputShape(const OpKernelContext* p, int index, TensorShape& shape) override { return p->TryGetInferredInputShape(index, shape); }

  // OpKernelInfo (wrapped)
  std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) override { return onnxruntime::CopyOpKernelInfo(info); }
  void OpKernelInfo__operator_delete(OpKernelInfo* p) override { delete p; }
  AllocatorPtr OpKernelInfo__GetAllocator(const OpKernelInfo* p, int device_id, OrtMemType mem_type) override { return p->GetAllocator(device_id, mem_type); }
  const IExecutionProvider* OpKernelInfo__GetExecutionProvider(const OpKernelInfo* p) override { return p->GetExecutionProvider(); }
  Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_string(const OpKernelInfo* p, const std::string& name, std::string* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_TensorProto(const OpKernelInfo* p, const std::string& name, ONNX_NAMESPACE::TensorProto* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<int64_t>& values) override { return p->GetAttrs(name, values); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<float>& values) override { return p->GetAttrs(name, values); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<std::string>& values) override { return p->GetAttrs(name, values); }

  const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept override { return p->GetDataTransferManager(); }
  const KernelDef& OpKernelInfo__GetKernelDef(const OpKernelInfo* p) override { return p->GetKernelDef(); }
  bool OpKernelInfo__TryGetConstantInput(const OpKernelInfo* p, int input_index, const Tensor** constant_input_value) override { return p->TryGetConstantInput(input_index, constant_input_value); }

  uint32_t OpKernelInfo__GetInputCount(const OpKernelInfo* p) override { return p->GetInputCount(); }
  uint32_t OpKernelInfo__GetOutputCount(const OpKernelInfo* p) override { return p->GetOutputCount(); }
  const Node& OpKernelInfo__node(const OpKernelInfo* p) override { return p->node(); }

  // SessionState (wrapped)
  const DataTransferManager& SessionState__GetDataTransferMgr(const SessionState* p) override { return p->GetDataTransferMgr(); }

  // Tensor (wrapped)
  std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) override { return std::make_unique<Tensor>(p_type, shape, allocator); }
  std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset) override { return std::make_unique<Tensor>(p_type, shape, p_data, alloc, offset); }
  void Tensor__operator_delete(Tensor* p) override { delete p; }

  bool* Tensor__MutableData_bool(Tensor* p) override { return p->MutableData<bool>(); }
  int8_t* Tensor__MutableData_int8(Tensor* p) override { return p->MutableData<int8_t>(); }
  uint8_t* Tensor__MutableData_uint8(Tensor* p) override { return p->MutableData<uint8_t>(); }
  int16_t* Tensor__MutableData_int16(Tensor* p) override { return p->MutableData<int16_t>(); }
  uint16_t* Tensor__MutableData_uint16(Tensor* p) override { return p->MutableData<uint16_t>(); }
  int32_t* Tensor__MutableData_int32(Tensor* p) override { return p->MutableData<int32_t>(); }
  uint32_t* Tensor__MutableData_uint32(Tensor* p) override { return p->MutableData<uint32_t>(); }
  int64_t* Tensor__MutableData_int64(Tensor* p) override { return p->MutableData<int64_t>(); }
  uint64_t* Tensor__MutableData_uint64(Tensor* p) override { return p->MutableData<uint64_t>(); }
  float* Tensor__MutableData_float(Tensor* p) override { return p->MutableData<float>(); }
  double* Tensor__MutableData_double(Tensor* p) override { return p->MutableData<double>(); }
  BFloat16* Tensor__MutableData_BFloat16(Tensor* p) override { return p->MutableData<BFloat16>(); }
  MLFloat16* Tensor__MutableData_MLFloat16(Tensor* p) override { return p->MutableData<MLFloat16>(); }

  const bool* Tensor__Data_bool(const Tensor* p) override { return p->Data<bool>(); }
  const int8_t* Tensor__Data_int8(const Tensor* p) override { return p->Data<int8_t>(); }
  const uint8_t* Tensor__Data_uint8(const Tensor* p) override { return p->Data<uint8_t>(); }
  const int16_t* Tensor__Data_int16(const Tensor* p) override { return p->Data<int16_t>(); }
  const uint16_t* Tensor__Data_uint16(const Tensor* p) override { return p->Data<uint16_t>(); }
  const int32_t* Tensor__Data_int32(const Tensor* p) override { return p->Data<int32_t>(); }
  const uint32_t* Tensor__Data_uint32(const Tensor* p) override { return p->Data<uint32_t>(); }
  const int64_t* Tensor__Data_int64(const Tensor* p) override { return p->Data<int64_t>(); }
  const uint64_t* Tensor__Data_uint64(const Tensor* p) override { return p->Data<uint64_t>(); }
  const float* Tensor__Data_float(const Tensor* p) override { return p->Data<float>(); }
  const double* Tensor__Data_double(const Tensor* p) override { return p->Data<double>(); }
  const BFloat16* Tensor__Data_BFloat16(const Tensor* p) override { return p->Data<BFloat16>(); }
  const MLFloat16* Tensor__Data_MLFloat16(const Tensor* p) override { return p->Data<MLFloat16>(); }

  gsl::span<const int64_t> Tensor__DataAsSpan_int64(const Tensor* p) override { return p->DataAsSpan<int64_t>(); }

  void* Tensor__MutableDataRaw(Tensor* p, MLDataType type) override { return p->MutableDataRaw(type); }
  const void* Tensor__DataRaw(const Tensor* p, MLDataType type) override { return p->DataRaw(type); }
  void* Tensor__MutableDataRaw(Tensor* p) noexcept override { return p->MutableDataRaw(); }
  const void* Tensor__DataRaw(const Tensor* p) noexcept override { return p->DataRaw(); }

  bool Tensor__IsDataType_bool(const Tensor* p) noexcept override { return p->IsDataType<bool>(); }
  bool Tensor__IsDataType_int8(const Tensor* p) noexcept override { return p->IsDataType<int8_t>(); }
  bool Tensor__IsDataType_uint8(const Tensor* p) noexcept override { return p->IsDataType<uint8_t>(); }
  bool Tensor__IsDataType_int16(const Tensor* p) noexcept override { return p->IsDataType<int16_t>(); }
  bool Tensor__IsDataType_uint16(const Tensor* p) noexcept override { return p->IsDataType<uint16_t>(); }
  bool Tensor__IsDataType_int32(const Tensor* p) noexcept override { return p->IsDataType<int32_t>(); }
  bool Tensor__IsDataType_uint32(const Tensor* p) noexcept override { return p->IsDataType<uint32_t>(); }
  bool Tensor__IsDataType_int64(const Tensor* p) noexcept override { return p->IsDataType<int64_t>(); }
  bool Tensor__IsDataType_uint64(const Tensor* p) noexcept override { return p->IsDataType<uint64_t>(); }
  bool Tensor__IsDataType_float(const Tensor* p) noexcept override { return p->IsDataType<float>(); }
  bool Tensor__IsDataType_double(const Tensor* p) noexcept override { return p->IsDataType<double>(); }
  bool Tensor__IsDataType_MLFloat16(const Tensor* p) noexcept override { return p->IsDataType<MLFloat16>(); }
  bool Tensor__IsDataTypeString(const Tensor* p) noexcept override { return p->IsDataTypeString(); }

  const TensorShape& Tensor__Shape(const Tensor* p) override { return p->Shape(); }
  void Tensor__Reshape(Tensor* p, const TensorShape& new_shape) override { return p->Reshape(new_shape); }
  void Tensor__SetByteOffset(Tensor* p, ptrdiff_t byte_offset) override { p->SetByteOffset(byte_offset); }
  ptrdiff_t Tensor__ByteOffset(const Tensor* p) override { return p->ByteOffset(); }
  size_t Tensor__SizeInBytes(const Tensor* p) override { return p->SizeInBytes(); }
  const OrtMemoryInfo& Tensor__Location(const Tensor* p) override { return p->Location(); }
  int32_t Tensor__GetElementType(const Tensor* p) override { return p->GetElementType(); }
  MLDataType Tensor__DataType(const Tensor* p) override { return p->DataType(); }

  // TensorSeq(wrapped)
  MLDataType TensorSeq__DataType(const TensorSeq* p) noexcept override { return p->DataType(); }
  void TensorSeq__SetType(TensorSeq* p, MLDataType data_type) override { p->SetType(data_type); }
  size_t TensorSeq__Size(const TensorSeq* p) noexcept override { return p->Size(); }
  const Tensor& TensorSeq__Get(const TensorSeq* p, size_t i) override { return p->Get(i); }
  void TensorSeq__Add(TensorSeq* p, Tensor&& tensor) override { p->Add(std::move(tensor)); }

  // AllocatorManager (direct)
  void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator) override { p->AllocatorManager::InsertAllocator(allocator); }
  AllocatorPtr AllocatorManager__GetAllocator(const AllocatorManager* p, int id, OrtMemType mem_type) override { return p->AllocatorManager::GetAllocator(id, mem_type); };
  // From cpu/tensor/unsqueeze.h (direct)
  Status UnsqueezeBase__PrepareCompute(const UnsqueezeBase* p, OpKernelContext* ctx, UnsqueezeBase__Prepare& prepare) override { return p->UnsqueezeBase::PrepareCompute(ctx, reinterpret_cast<UnsqueezeBase::Prepare&>(prepare)); }
  // From cpu/tensor/gatherbase.h (direct)
  Status GatherBase__PrepareForCompute(const GatherBase* p, OpKernelContext* context, GatherBase__Prepare& prepare) override { return p->GatherBase::PrepareForCompute(context, reinterpret_cast<GatherBase::Prepare&>(prepare)); }

#ifdef USE_CUDA
  // GatherElements (direct)
  Status GatherElements__ValidateInputShapes(const TensorShape& input_data_shape, const TensorShape& indices_shape, int64_t axis) override { return GatherElements::ValidateInputShapes(input_data_shape, indices_shape, axis); }

  // cumsum (direct)
  Status cumsum_op__GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) override { return cumsum_op::GetAxis(axis_tensor, input_rank, axis_out); }

  // TileOp (direct)
  bool TileOp__IsTileMemcpy(const TensorShape& input_shape, const int64_t* repeats, size_t rank, bool& is_batched_memcpy, size_t& num_of_elements_per_batch, size_t& num_of_copies_per_batch, size_t& num_of_batch_copies) override { return TileOp::IsTileMemcpy(input_shape, repeats, rank, is_batched_memcpy, num_of_elements_per_batch, num_of_copies_per_batch, num_of_batch_copies); }

  // ROI (direct)
  Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) override { return onnxruntime::CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr); }

  // NonMaxSuppressionBase (direct)
  Status NonMaxSuppressionBase__PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) override { return NonMaxSuppressionBase::PrepareCompute(ctx, pc); }
  Status NonMaxSuppressionBase__GetThresholdsFromInputs(const PrepareContext& pc, int64_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold) override { return NonMaxSuppressionBase::GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold); }

  // From onehot.h (direct)
  Status ValidateInputs(const Tensor* depth, const Tensor* values) override { return onnxruntime::ValidateInputs(depth, values); }
  Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis, int64_t& prefix_dim_size, int64_t& suffix_dim_size, std::vector<int64_t>& output_shape) override { return onnxruntime::PrepareOutputShape(indices, depth_val, axis, prefix_dim_size, suffix_dim_size, output_shape); }

  // From cpu/tensor/slice.h (direct)
  Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      SliceOp__PrepareForComputeMetadata& compute_metadata) override { return SliceBase::PrepareForCompute(raw_starts, raw_ends, raw_axes, reinterpret_cast<SliceOp::PrepareForComputeMetadata&>(compute_metadata)); }

  Status SliceBase__PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      const std::vector<int64_t>& raw_steps,
                                      SliceOp__PrepareForComputeMetadata& compute_metadata) override { return SliceBase::PrepareForCompute(raw_starts, raw_ends, raw_axes, raw_steps, reinterpret_cast<SliceOp::PrepareForComputeMetadata&>(compute_metadata)); }

  Status SliceBase__FillVectorsFromInput(const Tensor& start_tensor,
                                         const Tensor& ends_tensor,
                                         const Tensor* axes_tensor,
                                         const Tensor* steps_tensor,
                                         std::vector<int64_t>& input_starts,
                                         std::vector<int64_t>& input_ends,
                                         std::vector<int64_t>& input_axes,
                                         std::vector<int64_t>& input_steps) override { return SliceBase::FillVectorsFromInput(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps); }
  // From cpu/tensor/size.h (direct)
  Status Size__Compute(const Size* p, OpKernelContext* context) override { return p->Size::Compute(context); }
  // From cpu/tensor/scatter_nd.h (direct)
  Status ScatterNDBase__ValidateShapes(const TensorShape& input_shape,
                                       const TensorShape& indice_shape,
                                       const TensorShape& update_shape) override { return ScatterNDBase::ValidateShapes(input_shape, indice_shape, update_shape); }
  // From cpu/tensor/padbase.h (direct)
  Status PadBase__HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, TensorShape& output_shape) override { return PadBase::HandleDimValueZero(mode, input_shape, output_shape); }
  // From cpu/tensor/split.h (direct)
  Status SplitBase__PrepareForCompute(const SplitBase* p, const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                      int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                      std::vector<int64_t>& split_sizes) override { return p->SplitBase::PrepareForCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }
  // From cpu/tensor/concatbase.h (direct)
  Status ConcatBase__PrepareForCompute(const ConcatBase* p, OpKernelContext* ctx, const std::vector<const Tensor*>& input_tensors, Prepare& prepare) override { return p->ConcatBase::PrepareForCompute(ctx, input_tensors, prepare); }
  
  PhiloxGenerator& PhiloxGenerator__Default() override { return PhiloxGenerator::Default(); }

  Status Einsum__Compute(const Einsum* p, OpKernelContext* context) override { return p->Einsum::Compute(context); }

  // EinsumComputePreprocessor (wrapped)
  void EinsumComputePreprocessor__operator_delete(EinsumComputePreprocessor* p) override { delete p; }
  virtual std::unique_ptr<EinsumComputePreprocessor> EinsumComputePreprocessor__Create(EinsumEquationPreprocessor& equation_preprocessor,
                                                                                       const std::vector<const Tensor*>& inputs,
                                                                                       AllocatorPtr allocator,
                                                                                       void* einsum_cuda_assets) override { return std::make_unique<EinsumComputePreprocessor>(equation_preprocessor, inputs, allocator, einsum_cuda_assets); }

  virtual Status EinsumComputePreprocessor__Run(EinsumComputePreprocessor* p) override { return p->Run(); }
  virtual void EinsumComputePreprocessor__SetDeviceHelpers(EinsumComputePreprocessor* p, const EinsumOp::DeviceHelpers::Diagonal& diagonal_func, const EinsumOp::DeviceHelpers::Transpose& transpose_func) override { return p->SetDeviceHelpers(diagonal_func, transpose_func); }

  // EinsumTypedComputeProcessor (wrapped)
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<float>* p) override { delete p; }
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<double>* p) override { delete p; }
  void EinsumTypedComputeProcessor__operator_delete(EinsumTypedComputeProcessor<MLFloat16>* p) override { delete p; }
  std::unique_ptr<EinsumTypedComputeProcessor<float>> EinsumTypedComputeProcessor_float__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<float>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  std::unique_ptr<EinsumTypedComputeProcessor<double>> EinsumTypedComputeProcessor_double__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<double>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  std::unique_ptr<EinsumTypedComputeProcessor<MLFloat16>> EinsumTypedComputeProcessor_MLFloat16__Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) override { return std::make_unique<EinsumTypedComputeProcessor<MLFloat16>>(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<float>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<float>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<float>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<double>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<double>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<double>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  void EinsumTypedComputeProcessor__SetDeviceHelpers(EinsumTypedComputeProcessor<MLFloat16>* p, const EinsumOp::DeviceHelpers::Transpose& device_transpose_func, const EinsumOp::DeviceHelpers::MatMul<MLFloat16>& device_matmul_func, const EinsumOp::DeviceHelpers::ReduceSum<MLFloat16>& device_reduce_sum_func, const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func) override { return p->SetDeviceHelpers(device_transpose_func, device_matmul_func, device_reduce_sum_func, device_data_copy_func); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<float>* p) override { return p->Run(); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<double>* p) override { return p->Run(); }
  Status EinsumTypedComputeProcessor__Run(EinsumTypedComputeProcessor<MLFloat16>* p) override { return p->Run(); }

  // If (direct)
  void If__Init(If* p, const OpKernelInfo& info) override { p->If::Init(info); }
  Status If__Compute(const If* p, OpKernelContext* ctx) override { return p->If::Compute(ctx); }
  Status If__SetupSubgraphExecutionInfo(If* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->If::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  // Loop (direct)
  void Loop__Init(Loop* p, const OpKernelInfo& info) override { p->Loop::Init(info); }
  Status Loop__Compute(const Loop* p, OpKernelContext* ctx) override { return p->Loop::Compute(ctx); }
  Status Loop__SetupSubgraphExecutionInfo(Loop* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Loop::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  // Scan (direct)
  void Scan__Init(Scan<8>* p, const OpKernelInfo& info) override { p->Scan::Init(info); }
  void Scan__Init(Scan<9>* p, const OpKernelInfo& info) override { p->Scan::Init(info); }
  Status Scan__Compute(const Scan<8>* p, OpKernelContext* ctx) override { return p->Scan<8>::Compute(ctx); }
  Status Scan__Compute(const Scan<9>* p, OpKernelContext* ctx) override { return p->Scan<9>::Compute(ctx); }
  Status Scan__SetupSubgraphExecutionInfo(Scan<8>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Scan<8>::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }
  Status Scan__SetupSubgraphExecutionInfo(Scan<9>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) override { return p->Scan<9>::SetupSubgraphExecutionInfo(session_state, attribute_name, subgraph_session_state); }

  // ContribOps (direct)
#ifndef DISABLE_CONTRIB_OPS
  Status embed_layer_norm__CheckInputs(const OpKernelContext* context) override { return contrib::embed_layer_norm::CheckInputs(context); }
  Status bias_gelu_helper__CheckInputs(const OpKernelContext* context) override { return contrib::bias_gelu_helper::CheckInputs(context); }
  Status LongformerAttentionBase__CheckInputs(const contrib::LongformerAttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const TensorShape& mask_shape, const TensorShape& global_weights_shape, const TensorShape& global_bias_shape, const TensorShape& global_shape) override {
    return contrib::LongformerAttentionBase__CheckInputs(p, input_shape, weights_shape, bias_shape, mask_shape, global_weights_shape, global_bias_shape, global_shape);
  }
  Status AttentionBase__CheckInputs(const contrib::AttentionBase* p, const TensorShape& input_shape, const TensorShape& weights_shape, const TensorShape& bias_shape, const Tensor*& mask_index, const Tensor* past, const int max_threads_per_block) override { return p->contrib::AttentionBase::CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past, max_threads_per_block); }
  Tensor* AttentionBase__GetPresent(const contrib::AttentionBase* p, OpKernelContext* context, const Tensor* past, int batch_size, int head_size, int sequence_length, int& past_sequence_length) override { return p->contrib::AttentionBase::GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length); }
#endif

#ifdef ENABLE_TRAINING
  Status ATenOp__Compute(const contrib::ATenOp* p, OpKernelContext* p_ctx) override { return p->ATenOp::Compute(p_ctx); }
  void contrib__record_event_in_tensor(const Tensor& event_id_tensor) override { return contrib::record_event_in_tensor(event_id_tensor); }
  void contrib__wait_event_in_tensor(const Tensor& event_id_tensor) override { return contrib::wait_event_in_tensor(event_id_tensor); }
  Status contrib__Group__Compute(const contrib::Group* p, OpKernelContext* context) override { return p->Group::Compute(context); }
  Status contrib__PassThrough__Compute(const contrib::PassThrough* p, OpKernelContext* context) override { return p->PassThrough::Compute(context); }
  void contrib__VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) override { contrib::VerifyLogitWeightAndLabelShape(logit_shape, label_shape, weight_shape); }
  void contrib__GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) override { contrib::GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C); }
  void contrib__GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, std::vector<int64_t>& new_shape, std::vector<size_t>& permutations) override { contrib::GetPermutationAndShape(ncd_to_ndc, tensor_shape, new_shape, permutations); }
  Status contrib__PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) override { return contrib::PrepareForTrainingCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }
  Status contrib__YieldOp__Compute(const contrib::YieldOp* p, OpKernelContext* context) override { return p->YieldOp::Compute(context); }

#if defined(ORT_USE_NCCL)
  training::DistributedRunContext& GetDistributedRunContextInstance() override { return training::DistributedRunContext::GetInstance(); }
#endif

#ifdef ENABLE_TRAINING_TORCH_INTEROP
  void contrib__PythonOpBase__Init(contrib::PythonOpBase* p, const OpKernelInfo& info) override { p->PythonOpBase::Init(info); }
  void contrib__PythonOpBase__Clear(contrib::PythonOpBase* p) override { p->PythonOpBase::Clear(); }
  void contrib__PythonOpBase__SetOutputs(const contrib::PythonOpBase* p, OpKernelContext* context, void* diff_ctx, std::vector<OrtValue>& returned_args) override {
    return p->PythonOpBase::SetOutputs(context, diff_ctx, returned_args);
  }
  void contrib__PythonOpBase__RunForward(const contrib::PythonOpBase* p, OpKernelContext* context, void** diff_ctx, std::vector<OrtValue>& returned_ortvalues) override {
    return p->PythonOpBase::RunForward(context, diff_ctx, returned_ortvalues);
  }

  void contrib__PythonOpGradBase__Init(contrib::PythonOpGradBase* p, const OpKernelInfo& info) override { return p->PythonOpGradBase::Init(info); }
  void contrib__PythonOpGradBase__RunBackward(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) {
    return p->PythonOpGradBase::RunBackward(context, returned_ortvalues);
  }
  void contrib__PythonOpGradBase__SetOutputs(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_args) override { p->PythonOpGradBase::SetOutputs(context, returned_args); }

  language_interop_ops::torch::RefCountTracker& GetRefCountTrackerInstance() override { return language_interop_ops::torch::RefCountTracker::GetInstance(); }
  void RefCountTracker__DumpDetails(const language_interop_ops::torch::RefCountTracker* p, const std::string& phase_name) override {
    return p->language_interop_ops::torch::RefCountTracker::DumpDetails(phase_name);
  }
#endif
#endif
#endif
} provider_host_;

struct ProviderSharedLibrary {
  bool Ensure() {
    if (handle_)
      return true;

    std::string full_path = Env::Default().GetRuntimePath() + std::string(LIBRARY_PREFIX "onnxruntime_providers_shared" LIBRARY_EXTENSION);
    auto error = Env::Default().LoadDynamicLibrary(full_path, true /*shared_globals on unix*/, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return false;
    }

    void (*PProvider_SetHost)(void*);
    Env::Default().GetSymbolFromLibrary(handle_, "Provider_SetHost", (void**)&PProvider_SetHost);

    PProvider_SetHost(&provider_host_);
    return true;
  }

  void Unload() {
    if (handle_) {
      Env::Default().UnloadDynamicLibrary(handle_);
      handle_ = nullptr;
    }
  }

  ProviderSharedLibrary() = default;
  ~ProviderSharedLibrary() {
    // assert(!handle_); // We should already be unloaded at this point (disabled until Python shuts down deterministically)
  }

 private:
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderSharedLibrary);
};

static ProviderSharedLibrary s_library_shared;

bool InitProvidersSharedLibrary() {
  return s_library_shared.Ensure();
}

struct ProviderLibrary {
  ProviderLibrary(const char* filename, bool unload = true) : filename_{filename}, unload_{unload} {}
  ~ProviderLibrary() {
    // assert(!handle_); // We should already be unloaded at this point (disabled until Python shuts down deterministically)
  }

  Provider* Get() {
    if (provider_)
      return provider_;

    if (!s_library_shared.Ensure())
      return nullptr;

    std::string full_path = Env::Default().GetRuntimePath() + std::string(filename_);
    auto error = Env::Default().LoadDynamicLibrary(full_path, false, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return nullptr;
    }

    Provider* (*PGetProvider)();
    Env::Default().GetSymbolFromLibrary(handle_, "GetProvider", (void**)&PGetProvider);

    provider_ = PGetProvider();
    return provider_;
  }

  void Unload() {
    if (handle_) {
      if (provider_)
        provider_->Shutdown();

      if (unload_)
        Env::Default().UnloadDynamicLibrary(handle_);

      handle_ = nullptr;
      provider_ = nullptr;
    }
  }

 private:
  const char* filename_;
  bool unload_;
  Provider* provider_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderLibrary);
};

static ProviderLibrary s_library_cuda(LIBRARY_PREFIX "onnxruntime_providers_cuda" LIBRARY_EXTENSION
#ifndef _WIN32
                                      ,
                                      false /* unload - On Linux if we unload the cuda shared provider we crash */
#endif
);
static ProviderLibrary s_library_dnnl(LIBRARY_PREFIX "onnxruntime_providers_dnnl" LIBRARY_EXTENSION);
static ProviderLibrary s_library_openvino(LIBRARY_PREFIX "onnxruntime_providers_openvino" LIBRARY_EXTENSION);
static ProviderLibrary s_library_tensorrt(LIBRARY_PREFIX "onnxruntime_providers_tensorrt" LIBRARY_EXTENSION);

void UnloadSharedProviders() {
  s_library_dnnl.Unload();
  s_library_openvino.Unload();
  s_library_tensorrt.Unload();
  s_library_cuda.Unload();
  s_library_shared.Unload();
}

// Used by test code
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->CreateCUDAPinnedAllocator(device_id, name);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options) {
  if (auto* provider = s_library_cuda.Get())
    return provider->CreateExecutionProviderFactory(provider_options);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena) {
  if (auto* provider = s_library_dnnl.Get())
    return provider->CreateExecutionProviderFactory(use_arena);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  if (auto* provider = s_library_tensorrt.Get())
    return provider->CreateExecutionProviderFactory(device_id);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* provider_options) {
  if (auto* provider = s_library_tensorrt.Get())
    return provider->CreateExecutionProviderFactory(provider_options);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* provider_options) {
  if (auto* provider = s_library_openvino.Get())
    return provider->CreateExecutionProviderFactory(provider_options);

  return nullptr;
}

ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO() {
  if (auto* provider = s_library_openvino.Get())
    return reinterpret_cast<ProviderInfo_OpenVINO*>(provider->GetInfo());
  return nullptr;
}

ProviderInfo_CUDA* TryGetProviderInfo_CUDA() {
  if (auto* provider = s_library_cuda.Get())
    return reinterpret_cast<ProviderInfo_CUDA*>(provider->GetInfo());

  return nullptr;
}

ProviderInfo_CUDA& GetProviderInfo_CUDA() {
  if(auto* info = TryGetProviderInfo_CUDA())
    return *info;

  ORT_THROW("CUDA Provider not available, can't get interface for it");
}

void CopyGpuToCpu(
    void* dst_ptr,
    const void* src_ptr,
    const size_t size,
    const OrtMemoryInfo& dst_location,
    const OrtMemoryInfo& src_location) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->CopyGpuToCpu(dst_ptr, src_ptr, size, dst_location, src_location);
  ORT_THROW("GPU-to-CPU copy is not implemented.");
}

void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->cudaMemcpy_HostToDevice(dst, src, count);
  ORT_THROW("cudaMemcpy_HostToDevice is not implemented.");
}

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
namespace cuda {
INcclService& INcclService::GetInstance() {
  return GetProviderInfo_CUDA().GetINcclService();
}
}  // namespace cuda
#endif

void UpdateProviderInfo_Tensorrt(OrtTensorRTProviderOptions* provider_options, const ProviderOptions& options) {
  if (auto provider = s_library_tensorrt.Get()) {
    provider->UpdateProviderOptions(reinterpret_cast<void*>(provider_options), options);
  }
}

ProviderOptions GetProviderInfo_Tensorrt(const OrtTensorRTProviderOptions* provider_options) {
  if (auto provider = s_library_tensorrt.Get()) {
    return provider->GetProviderOptions(reinterpret_cast<const void*>(provider_options));
  }

  return {};
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CreateExecutionProviderFactory_Tensorrt(device_id);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CreateExecutionProviderFactory_Tensorrt(tensorrt_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CreateExecutionProviderFactory_OpenVINO(provider_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_OpenVINO: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const char* device_type) {
  OrtOpenVINOProviderOptions provider_options{};
  provider_options.device_type = device_type;
  return OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO(options, &provider_options);
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id) {
  OrtCUDAProviderOptions provider_options{};
  provider_options.device_id = device_id;

  return OrtApis::SessionOptionsAppendExecutionProvider_CUDA(options, &provider_options);
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  API_IMPL_BEGIN
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->SetCurrentGpuDeviceId(device_id);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is either not enabled or not available.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  API_IMPL_BEGIN
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->GetCurrentGpuDeviceId(device_id);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is either not enabled or not available.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CreateExecutionProviderFactory_Cuda(cuda_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options) {
  return OrtApis::SessionOptionsAppendExecutionProvider_TensorRT(options, reinterpret_cast<const OrtTensorRTProviderOptions*>(tensorrt_options));
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  *out = new OrtTensorRTProviderOptionsV2();
  (*out)->device_id = 0;
  (*out)->has_user_compute_stream = 0;
  (*out)->user_compute_stream = nullptr;
  (*out)->trt_max_partition_iterations = 1000;
  (*out)->trt_min_subgraph_size = 1;
  (*out)->trt_max_workspace_size = 1 << 30;
  (*out)->trt_fp16_enable = false;
  (*out)->trt_int8_enable = false;
  (*out)->trt_int8_calibration_table_name = nullptr;
  (*out)->trt_int8_use_native_calibration_table = false;
  (*out)->trt_dla_enable = false;
  (*out)->trt_dla_core = false;
  (*out)->trt_dump_subgraphs = false;
  (*out)->trt_engine_cache_enable= false;
  (*out)->trt_engine_cache_path = nullptr;
  (*out)->trt_engine_decryption_enable = false;
  (*out)->trt_engine_decryption_lib_path = nullptr;
  (*out)->trt_force_sequential_engine_build = false;
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptions,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::UpdateProviderInfo_Tensorrt(reinterpret_cast<OrtTensorRTProviderOptions*>(tensorrt_options),
                                           reinterpret_cast<const onnxruntime::ProviderOptions&>(provider_options_map));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  onnxruntime::ProviderOptions options = onnxruntime::GetProviderInfo_Tensorrt(reinterpret_cast<const OrtTensorRTProviderOptions*>(tensorrt_options));
  onnxruntime::ProviderOptions::iterator it = options.begin();
  std::string options_str = "";

  while (it != options.end()) {
    if (options_str == "") {
      options_str += it->first;
      options_str += "=";
      options_str += it->second;
    } else {
      options_str += ";";
      options_str += it->first;
      options_str += "=";
      options_str += it->second;
    }
    it++;
  }

  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* ptr) {
#ifdef USE_TENSORRT
  if (ptr != nullptr) {
    if (ptr->trt_int8_calibration_table_name != nullptr) {
      delete ptr->trt_int8_calibration_table_name;
    }

    if (ptr->trt_engine_cache_path != nullptr) {
      delete ptr->trt_engine_cache_path;
    }

    if (ptr->trt_engine_decryption_lib_path != nullptr) {
      delete ptr->trt_engine_decryption_lib_path;
    }
  }

  delete ptr;
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}
