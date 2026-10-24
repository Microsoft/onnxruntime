// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built as a DLL

#include "provider_api.h"
#include <assert.h>
#include <mutex>
#include "core/providers/shared/common.h"

#include "core/common/inlined_containers.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/random_generator.h"
#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cpu/math/einsum.h"
#include "core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.h"
#include "core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h"
#include "core/providers/cpu/object_detection/non_max_suppression.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/cpu/tensor/padbase.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/split.h"
#include "core/providers/cpu/tensor/size.h"
#include "core/providers/cpu/tensor/scatter_nd.h"
#include "core/providers/cpu/tensor/unsqueeze.h"
#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/providers.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/cpu/bert/embed_layer_norm_helper.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"
#include "contrib_ops/cpu/transformers/beam_search.h"
#include "contrib_ops/cpu/transformers/greedy_search.h"
#include "contrib_ops/cpu/transformers/sampling.h"
#ifdef ENABLE_ATEN
#include "contrib_ops/cpu/aten_ops/aten_op.h"
#endif
#endif

#ifdef ENABLE_TRAINING_OPS
#include "orttraining/training_ops/cpu/controlflow/group.h"
#include "orttraining/training_ops/cpu/optimizer/adamw/adamwbase.h"
#include "orttraining/training_ops/cpu/optimizer/sgd/sgdbase.h"

// Should remove the include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.
#include "contrib_ops/cpu/tensor/shrunken_gather.h"
#endif

#ifdef ENABLE_TRAINING
#include "orttraining/training_ops/cpu/controlflow/yield.h"

#ifdef ENABLE_TRAINING_TORCH_INTEROP
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"
#include "orttraining/core/framework/torch/refcount_tracker.h"
#endif

#endif

#ifdef ENABLE_TRITON
#include "orttraining/training_ops/cpu/triton/triton_op.h"
#endif

#ifndef _Ret_notnull_
#define _Ret_notnull_
#endif

#ifndef _Post_writable_byte_size_
#define _Post_writable_byte_size_(n)
#endif

#ifdef _WIN32
// Override default new/delete so that we match the host's allocator
_Ret_notnull_ _Post_writable_byte_size_(n) void* operator new(size_t n) { return Provider_GetHost()->HeapAllocate(n); }
void operator delete(void* p) noexcept { return Provider_GetHost()->HeapFree(p); }
void operator delete(void* p, size_t /*size*/) noexcept { return Provider_GetHost()->HeapFree(p); }
#endif

namespace onnxruntime {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// "Global initializer calls a non-constexpr function."
#pragma warning(disable : 26426)
#endif
ProviderHost* g_host = Provider_GetHost();
ProviderHostCPU& g_host_cpu = g_host->GetProviderHostCPU();
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
static std::unique_ptr<std::vector<std::function<void()>>> s_run_on_unload_;

void RunOnUnload(std::function<void()> function) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard{mutex};
  if (!s_run_on_unload_)
    s_run_on_unload_ = std::make_unique<std::vector<std::function<void()>>>();
  s_run_on_unload_->push_back(std::move(function));
}

// This object is destroyed as part of the DLL unloading code and handles running all of the RunOnLoad functions
struct OnUnload {
  ~OnUnload() {
    if (!s_run_on_unload_)
      return;

    for (auto& function : *s_run_on_unload_)
      function();

    s_run_on_unload_.reset();
  }

} g_on_unload;

void* CPUAllocator::Alloc(size_t size) { return g_host->CPUAllocator__Alloc(this, size); }
void CPUAllocator::Free(void* p) { g_host->CPUAllocator__Free(this, p); }

AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) {
  return g_host->CreateAllocator(info);
}

template <>
MLDataType DataTypeImpl::GetType<Tensor>() { return Provider_GetHost()->DataTypeImpl__GetType_Tensor(); }
#if !defined(DISABLE_SPARSE_TENSORS)
template <>
MLDataType DataTypeImpl::GetType<SparseTensor>() { return Provider_GetHost()->DataTypeImpl__GetType_SparseTensor(); }
#endif
template <>
MLDataType DataTypeImpl::GetType<TensorSeq>() { return Provider_GetHost()->DataTypeImpl__GetType_TensorSeq(); }
MLDataType DataTypeImpl::GetTypeFromOnnxType(int onnx_type) { return Provider_GetHost()->DataTypeImpl__GetTypeFromOnnxType(onnx_type); }
template <>
MLDataType DataTypeImpl::GetType<bool>() { return Provider_GetHost()->DataTypeImpl__GetType_bool(); }
template <>
MLDataType DataTypeImpl::GetType<int8_t>() { return Provider_GetHost()->DataTypeImpl__GetType_int8(); }
template <>
MLDataType DataTypeImpl::GetType<uint8_t>() { return Provider_GetHost()->DataTypeImpl__GetType_uint8(); }
template <>
MLDataType DataTypeImpl::GetType<int16_t>() { return Provider_GetHost()->DataTypeImpl__GetType_int16(); }
template <>
MLDataType DataTypeImpl::GetType<uint16_t>() { return Provider_GetHost()->DataTypeImpl__GetType_uint16(); }
template <>
MLDataType DataTypeImpl::GetType<int32_t>() { return Provider_GetHost()->DataTypeImpl__GetType_int32(); }
template <>
MLDataType DataTypeImpl::GetType<uint32_t>() { return Provider_GetHost()->DataTypeImpl__GetType_uint32(); }
template <>
MLDataType DataTypeImpl::GetType<int64_t>() { return Provider_GetHost()->DataTypeImpl__GetType_int64(); }
template <>
MLDataType DataTypeImpl::GetType<uint64_t>() { return Provider_GetHost()->DataTypeImpl__GetType_uint64(); }
template <>
MLDataType DataTypeImpl::GetType<float>() { return Provider_GetHost()->DataTypeImpl__GetType_float(); }
template <>
MLDataType DataTypeImpl::GetType<double>() { return Provider_GetHost()->DataTypeImpl__GetType_double(); }
template <>
MLDataType DataTypeImpl::GetType<BFloat16>() { return Provider_GetHost()->DataTypeImpl__GetType_BFloat16(); }
template <>
MLDataType DataTypeImpl::GetType<MLFloat16>() { return Provider_GetHost()->DataTypeImpl__GetType_MLFloat16(); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
MLDataType DataTypeImpl::GetType<Float8E4M3FN>() { return Provider_GetHost()->DataTypeImpl__GetType_Float8E4M3FN(); }
template <>
MLDataType DataTypeImpl::GetType<Float8E4M3FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetType_Float8E4M3FNUZ(); }
template <>
MLDataType DataTypeImpl::GetType<Float8E5M2>() { return Provider_GetHost()->DataTypeImpl__GetType_Float8E5M2(); }
template <>
MLDataType DataTypeImpl::GetType<Float8E5M2FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetType_Float8E5M2FNUZ(); }
template <>
MLDataType DataTypeImpl::GetType<Float8E8M0>() { return Provider_GetHost()->DataTypeImpl__GetType_Float8E8M0(); }
#endif
template <>
MLDataType DataTypeImpl::GetType<Int4x2>() { return Provider_GetHost()->DataTypeImpl__GetType_Int4x2(); }
template <>
MLDataType DataTypeImpl::GetType<UInt4x2>() { return Provider_GetHost()->DataTypeImpl__GetType_UInt4x2(); }
template <>
MLDataType DataTypeImpl::GetType<Int2x4>() { return Provider_GetHost()->DataTypeImpl__GetType_Int2x4(); }
template <>
MLDataType DataTypeImpl::GetType<UInt2x4>() { return Provider_GetHost()->DataTypeImpl__GetType_UInt2x4(); }

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
MLDataType DataTypeImpl::GetType<Float4E2M1x2>() { return Provider_GetHost()->DataTypeImpl__GetType_Float4E2M1x2(); }
#endif
template <>
MLDataType DataTypeImpl::GetType<Float6E2M3>() { return Provider_GetHost()->DataTypeImpl__GetType_Float6E2M3(); }
template <>
MLDataType DataTypeImpl::GetType<Float6E3M2>() { return Provider_GetHost()->DataTypeImpl__GetType_Float6E3M2(); }

template <>
MLDataType DataTypeImpl::GetType<std::string>() { return Provider_GetHost()->DataTypeImpl__GetType_string(); }
MLDataType DataTypeImpl::GetTensorTypeFromOnnxType(int onnx_type) { return Provider_GetHost()->DataTypeImpl__GetTensorTypeFromOnnxType(onnx_type); }
template <>
MLDataType DataTypeImpl::GetTensorType<bool>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_bool(); }
template <>
MLDataType DataTypeImpl::GetTensorType<int8_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_int8(); }
template <>
MLDataType DataTypeImpl::GetTensorType<uint8_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_uint8(); }
template <>
MLDataType DataTypeImpl::GetTensorType<int16_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_int16(); }
template <>
MLDataType DataTypeImpl::GetTensorType<uint16_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_uint16(); }
template <>
MLDataType DataTypeImpl::GetTensorType<int32_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_int32(); }
template <>
MLDataType DataTypeImpl::GetTensorType<uint32_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_uint32(); }
template <>
MLDataType DataTypeImpl::GetTensorType<int64_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_int64(); }
template <>
MLDataType DataTypeImpl::GetTensorType<uint64_t>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_uint64(); }
template <>
MLDataType DataTypeImpl::GetTensorType<float>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_float(); }
template <>
MLDataType DataTypeImpl::GetTensorType<double>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_double(); }
template <>
MLDataType DataTypeImpl::GetTensorType<BFloat16>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_BFloat16(); }
template <>
MLDataType DataTypeImpl::GetTensorType<MLFloat16>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_MLFloat16(); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
MLDataType DataTypeImpl::GetTensorType<Float8E4M3FN>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float8E4M3FN(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Float8E4M3FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float8E4M3FNUZ(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Float8E5M2>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float8E5M2(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Float8E5M2FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float8E5M2FNUZ(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Float8E8M0>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float8E8M0(); }
#endif
template <>
MLDataType DataTypeImpl::GetTensorType<Int4x2>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Int4x2(); }
template <>
MLDataType DataTypeImpl::GetTensorType<UInt4x2>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_UInt4x2(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Int2x4>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Int2x4(); }
template <>
MLDataType DataTypeImpl::GetTensorType<UInt2x4>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_UInt2x4(); }

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
MLDataType DataTypeImpl::GetTensorType<Float4E2M1x2>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float4E2M1x2(); }
#endif
template <>
MLDataType DataTypeImpl::GetTensorType<Float6E2M3>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float6E2M3(); }
template <>
MLDataType DataTypeImpl::GetTensorType<Float6E3M2>() { return Provider_GetHost()->DataTypeImpl__GetTensorType_Float6E3M2(); }

#if !defined(DISABLE_SPARSE_TENSORS)
template <>
MLDataType DataTypeImpl::GetSparseTensorType<bool>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_bool(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<int8_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_int8(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<uint8_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_uint8(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<int16_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_int16(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<uint16_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_uint16(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<int32_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_int32(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<uint32_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_uint32(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<int64_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_int64(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<uint64_t>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_uint64(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<float>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_float(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<double>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_double(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<std::string>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_string(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<BFloat16>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_BFloat16(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<MLFloat16>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_MLFloat16(); }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
MLDataType DataTypeImpl::GetSparseTensorType<Float8E4M3FN>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_Float8E4M3FN(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<Float8E4M3FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_Float8E4M3FNUZ(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<Float8E5M2>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_Float8E5M2(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<Float8E5M2FNUZ>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_Float8E5M2FNUZ(); }
template <>
MLDataType DataTypeImpl::GetSparseTensorType<Float8E8M0>() { return Provider_GetHost()->DataTypeImpl__GetSparseTensorType_Float8E8M0(); }
#endif

#endif

Status IDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return g_host->IDataTransfer__CopyTensor(this, src, dst);
}

Status IDataTransfer::CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const {
  return g_host->IDataTransfer__CopyTensors(this, src_dst_pairs);
}
#if !defined(DISABLE_SPARSE_TENSORS)
Status IDataTransfer::CopySparseTensors(const std::vector<SparseSrcDstPair>& src_dst_pairs) const {
  return g_host->IDataTransfer__CopySparseTensors(this, src_dst_pairs);
}
#endif

const Node& OpKernel::Node() const { return g_host->OpKernel__Node(this); }

TensorShape::TensorShape(gsl::span<const int64_t> dims) {
  Allocate(dims.size());
  gsl::copy(dims, values_);
}

TensorShape& TensorShape::operator=(const TensorShape& other) {
  g_host->TensorShape__operator_assign(this, other);
  return *this;
}

TensorShape& TensorShape::operator=(TensorShape&& other) noexcept {
  g_host->TensorShape__operator_move_assign(this, std::move(other));
  return *this;
}

void TensorShape::Allocate(size_t size) {
  g_host->TensorShape__Allocate(this, size);
}

int64_t TensorShape::Size() const {
  int64_t size = SizeHelper(0, values_.size());
  // should we cache the size? as multiple operation may be expensive.
  return size;
}

int64_t TensorShape::SizeHelper(size_t start, size_t end) const {
  return g_host->TensorShape__SizeHelper(this, start, end);
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  assert(dimstart <= dimend && dimend <= values_.size());  // "Invalid tensor shape slice argument."
  return TensorShape(GetDims().subspan(dimstart, dimend - dimstart));
}

std::string TensorShape::ToString() const {
  return g_host->TensorShape__ToString(this);
}

int64_t TensorShape::SizeToDimension(size_t dimension) const { return g_host->TensorShape__SizeToDimension(this, dimension); }
int64_t TensorShape::SizeFromDimension(size_t dimension) const { return g_host->TensorShape__SizeFromDimension(this, dimension); }

std::ostream& operator<<(std::ostream& out, const TensorShape& shape) { return g_host->operator_left_shift(out, shape); }

AllocatorPtr CreateAllocator(AllocatorCreationInfo info) {
  return g_host->CreateAllocator(info);
}

std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& info) {
  return g_host->CreateCPUAllocator(info);
}

bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept {
  return g_host->IAllocator__CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out);
}

std::unique_ptr<IExecutionProvider> IExecutionProviderFactory::CreateProvider(
    const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
  return g_host->IExecutionProviderFactory__CreateProvider(this, session_options, session_logger);
}

std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                                  const IKernelLookup& kernel_lookup,
                                                                                  const GraphOptimizerRegistry& graph_optimizer_registry,
                                                                                  IResourceAccountant* resource_accountant) const {
  return g_host->IExecutionProvider__GetCapability(this, graph_viewer, kernel_lookup, graph_optimizer_registry, resource_accountant);
}
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  return g_host->IExecutionProvider__Compile(this, fused_nodes_and_graphs, node_compute_funcs);
}

#if defined(USE_TENSORRT)
std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAAllocator(device_id, name);
}

std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) {
  return g_host->CreateCUDAPinnedAllocator(device_id, name);
}

std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() {
  return g_host->CreateGPUDataTransfer();
}
#endif

std::unique_ptr<IAllocator> CreateMIGraphXAllocator(int16_t device_id, const char* name) {
  return g_host->CreateMIGraphXAllocator(device_id, name);
}

std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(int16_t device_id, const char* name) {
  return g_host->CreateMIGraphXPinnedAllocator(device_id, name);
}

std::string GetEnvironmentVar(const std::string& var_name) {
  return g_host->GetEnvironmentVar(var_name);
}

std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes,
                                                   const logging::Logger& logger) {
  return g_host->GetCpuPreferredNodes(graph, kernel_lookup, tentative_nodes, logger);
}

namespace profiling {

std::string demangle(const char* name) { return g_host->demangle(name); }
std::string demangle(const std::string& name) { return g_host->demangle(name); }

}  // namespace profiling

namespace logging {

unsigned int GetThreadId() { return g_host->GetThreadId(); }
unsigned int GetProcessId() { return g_host->GetProcessId(); }

const char* Category::onnxruntime = "onnxruntime";

}  // namespace logging

namespace common {

Status::Status(StatusCategory category, int code, const std::string& msg) {
  // state_ will be allocated here causing the status to be treated as a failure
  ORT_ENFORCE(code != static_cast<int>(common::OK));

  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code, const char* msg) {
  // state_ will be allocated here causing the status to be treated as a failure
  ORT_ENFORCE(code != static_cast<int>(common::OK));

  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code) : Status(category, code, "") {
}

StatusCategory Status::Category() const noexcept {
  return IsOK() ? StatusCategory::NONE : state_->category;
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(common::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const noexcept {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const { return g_host->Status__ToString(this); }

const std::string& Status::EmptyString() noexcept {
  static std::string s_empty;
  return s_empty;
}

}  // namespace common

namespace math {
uint16_t floatToHalf(float f) { return g_host->math__floatToHalf(f); }
float halfToFloat(uint16_t h) { return g_host->math__halfToFloat(h); }

}  // namespace math

namespace sparse_utils {
#if !defined(DISABLE_SPARSE_TENSORS)
#if !defined(ORT_MINIMAL_BUILD)
Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, SparseTensor& dst) {
  return g_host->sparse_utils__DenseTensorToSparseCsr(data_manager, src, cpu_allocator, dst_allocator, dst);
}

Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst) {
  return g_host->sparse_utils__SparseCsrToDenseTensor(data_manager, src, cpu_allocator, dst_allocator, dst);
}

Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, Tensor& dst) {
  return g_host->sparse_utils__SparseCooToDenseTensor(data_manager, src, cpu_allocator, dst_allocator, dst);
}
#endif  // !ORT_MINIMAL_BUILD

Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_indexs, SparseTensor& dst) {
  return g_host->sparse_utils__DenseTensorToSparseCoo(data_manager, src, cpu_allocator, dst_allocator, linear_indexs, dst);
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

}  // namespace sparse_utils

std::vector<std::string> GetStackTrace() { return g_host->GetStackTrace(); }

void LogRuntimeError(uint32_t session_id, const common::Status& status,
                     const char* file, const char* function, uint32_t line) {
  return g_host->LogRuntimeError(session_id, status, file, function, line);
}

std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) {
  return g_host->CopyOpKernelInfo(info);
}

namespace utils {
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ bool* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ float* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ double* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ MLFloat16* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ BFloat16* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int8_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint8_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int16_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint16_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int32_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint32_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int64_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint64_t* p_data, size_t expected_size) { return g_host->UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& tensor, const std::filesystem::path& model_path,
                             /*out*/ std::vector<uint8_t>& unpacked_tensor) {
  return g_host->UnpackInitializerData(tensor, model_path, unpacked_tensor);
}
Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ std::vector<uint8_t>& unpacked_tensor) {
  return g_host->UnpackInitializerData(tensor, std::filesystem::path(), unpacked_tensor);
}

}  // namespace utils

Status NonMaxSuppressionBase::PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) { return g_host_cpu.NonMaxSuppressionBase__PrepareCompute(ctx, pc); }
Status NonMaxSuppressionBase::GetThresholdsFromInputs(const PrepareContext& pc, int64_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold) { return g_host_cpu.NonMaxSuppressionBase__GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold); }

Status GatherBase::PrepareForCompute(OpKernelContext* context, GatherBase::Prepare& p) const { return g_host_cpu.GatherBase__PrepareForCompute(this, context, reinterpret_cast<GatherBase__Prepare&>(p)); }
Status UnsqueezeBase::PrepareCompute(OpKernelContext* ctx, UnsqueezeBase::Prepare& p) const { return g_host_cpu.UnsqueezeBase__PrepareCompute(this, ctx, reinterpret_cast<UnsqueezeBase__Prepare&>(p)); }

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
bool TileOp::IsTileMemcpy(const TensorShape& input_shape, const int64_t* repeats, size_t rank, bool& is_batched_memcpy, size_t& num_of_elements_per_batch, size_t& num_of_copies_per_batch, size_t& num_of_batch_copies) {
  return g_host_cpu.TileOp__IsTileMemcpy(input_shape, repeats, rank, is_batched_memcpy, num_of_elements_per_batch, num_of_copies_per_batch, num_of_batch_copies);
}

Status SliceBase::FlattenOutputDims(gsl::span<const int64_t> input_dimensions, gsl::span<const int64_t> output_dims,
                                    TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                    TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) {
  return g_host_cpu.SliceBase__FlattenOutputDims(
      input_dimensions, output_dims, starts, ends, steps, p_flattened_input_dims, p_flattened_output_dims);
}

Status SliceBase::PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                    gsl::span<const int64_t> raw_ends,
                                    gsl::span<const int64_t> raw_axes,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) { return g_host_cpu.SliceBase__PrepareForCompute(raw_starts, raw_ends, raw_axes, reinterpret_cast<SliceOp__PrepareForComputeMetadata&>(compute_metadata)); }

Status SliceBase::PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                    gsl::span<const int64_t> raw_ends,
                                    gsl::span<const int64_t> raw_axes,
                                    gsl::span<const int64_t> raw_steps,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) { return g_host_cpu.SliceBase__PrepareForCompute(raw_starts, raw_ends, raw_axes, raw_steps, reinterpret_cast<SliceOp__PrepareForComputeMetadata&>(compute_metadata)); }

Status SliceBase::FillVectorsFromInput(const Tensor& start_tensor,
                                       const Tensor& ends_tensor,
                                       const Tensor* axes_tensor,
                                       const Tensor* steps_tensor,
                                       TensorShapeVector& input_starts,
                                       TensorShapeVector& input_ends,
                                       TensorShapeVector& input_axes,
                                       TensorShapeVector& input_steps) { return g_host_cpu.SliceBase__FillVectorsFromInput(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps); }

Status SplitBase::PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                    int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                    std::vector<int64_t>& split_sizes) const { return g_host_cpu.SplitBase__PrepareForCompute(this, input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }

Status Size::Compute(OpKernelContext* context) const { return g_host_cpu.Size__Compute(this, context); }

// Inlined rather than delegated through g_host_cpu because ValidateShapes is
// pure shape arithmetic with no framework dependencies.
Status ScatterND::ValidateShapes(const TensorShape& input_shape,
                                 const TensorShape& indice_shape,
                                 const TensorShape& update_shape) { return scatter_nd_internal::ValidateShapes(input_shape, indice_shape, update_shape); }

Status PadBase::HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, const TensorShape& output_shape) {
  return g_host_cpu.PadBase__HandleDimValueZero(mode, input_shape, output_shape);
}

void PadBase::ComputePads(OpKernelContext& ctx, size_t data_rank, gsl::span<const int64_t> pads_data,
                          PadsVector& pads) {
  g_host_cpu.PadBase__ComputePads(ctx, data_rank, pads_data, pads);
}

Status ConcatBase::PrepareForCompute(OpKernelContext* ctx, const ConcatBase::InlinedTensorsVector& input_tensors,
                                     Prepare& p) const {
  return g_host_cpu.ConcatBase__PrepareForCompute(this, ctx, reinterpret_cast<const ConcatBase_InlinedTensorsVector&>(input_tensors), p);
}

PhiloxGenerator& PhiloxGenerator::Default() { return g_host->PhiloxGenerator__Default(); }

Status Einsum::Compute(OpKernelContext* context) const { return g_host_cpu.Einsum__Compute(this, context); }

template <>
Status EinsumTypedComputeProcessor<float>::Run() {
  return g_host_cpu.EinsumTypedComputeProcessor_float_Compute(
      context_, allocator_, tp_, mlas_backend_config_, einsum_compute_preprocessor_, einsum_ep_assets_,
      device_transpose_func_, device_matmul_func_, device_reduce_sum_func_,
      device_data_copy_func_, device_zero_buffer_func_, device_create_tensor_func_);
}

template <>
Status EinsumTypedComputeProcessor<double>::Run() {
  return g_host_cpu.EinsumTypedComputeProcessor_double_Compute(
      context_, allocator_, tp_, mlas_backend_config_, einsum_compute_preprocessor_, einsum_ep_assets_,
      device_transpose_func_, device_matmul_func_, device_reduce_sum_func_,
      device_data_copy_func_, device_zero_buffer_func_, device_create_tensor_func_);
}

template <>
Status EinsumTypedComputeProcessor<MLFloat16>::Run() {
  return g_host_cpu.EinsumTypedComputeProcessor_MLFloat16_Compute(
      context_, allocator_, tp_, mlas_backend_config_, einsum_compute_preprocessor_, einsum_ep_assets_,
      device_transpose_func_, device_matmul_func_, device_reduce_sum_func_,
      device_data_copy_func_, device_zero_buffer_func_, device_create_tensor_func_);
}

void UpsampleBase::AdjustOutputSizeAsPolicy(TensorShapeVector& output_dims, gsl::span<const int64_t> input_dims,
                                            InlinedVector<float>& scales) const {
  g_host_cpu.UpsampleBase__AdjustOutputSizeAsPolicy(this, output_dims, input_dims, scales);
}

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
Status embed_layer_norm::CheckInputs(const OpKernelContext* context, bool quantizedVersion) {
  return g_host_cpu.embed_layer_norm__CheckInputs(context, quantizedVersion);
}

Status bias_gelu_helper::CheckInputs(const OpKernelContext* context) { return g_host_cpu.bias_gelu_helper__CheckInputs(context); }

Status LongformerAttentionBase::CheckInputs(const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const TensorShape& mask_shape,
                                            const TensorShape& global_weights_shape,
                                            const TensorShape& global_bias_shape,
                                            const TensorShape& global_shape) const {
  return g_host_cpu.LongformerAttentionBase__CheckInputs(this, input_shape, weights_shape, bias_shape, mask_shape,
                                                         global_weights_shape, global_bias_shape, global_shape);
}

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape& weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const Tensor* attention_bias,
                                  void* parameters,
                                  const int max_threads_per_block,
                                  const Tensor* past_seq_len) const {
  return g_host_cpu.AttentionBase__CheckInputs(this, input_shape, weights_shape, bias_shape,
                                               mask_index, past, attention_bias, parameters,
                                               max_threads_per_block, past_seq_len);
}
Tensor* AttentionBase::GetPresent(OpKernelContext* context, const Tensor* past, int batch_size, int head_size,
                                  int sequence_length, int& past_sequence_length) const {
  return g_host_cpu.AttentionBase__GetPresent(this, context, past, batch_size, head_size,
                                              sequence_length, past_sequence_length);
}

namespace transformers {
#if !defined(DISABLE_GENERATION_OPS)
void BeamSearch::Init(const OpKernelInfo& info) { g_host_cpu.BeamSearch__Init(this, info); }

Status BeamSearch::Compute(OpKernelContext* ctx) const { return g_host_cpu.BeamSearch__Compute(this, ctx); }

Status BeamSearch::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name,
                                              const SessionState& subgraph_session_state) {
  return g_host_cpu.BeamSearch__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state);
}

Status WhisperBeamSearch::Compute(OpKernelContext* ctx) const { return g_host_cpu.WhisperBeamSearch__Compute(this, ctx); }

void BeamSearchParameters::ParseFromAttributes(const OpKernelInfo& info) { g_host_cpu.BeamSearchParameters__ParseFromAttributes(this, info); }

void GreedySearchParameters::ParseFromAttributes(const OpKernelInfo& info) { g_host_cpu.GreedySearchParameters__ParseFromAttributes(this, info); }

void SamplingParameters::ParseFromAttributes(const OpKernelInfo& info) { g_host_cpu.SamplingParameters__ParseFromAttributes(this, info); }

void WhisperBeamSearchParameters::ParseFromAttributes(const OpKernelInfo& info) { g_host_cpu.WhisperBeamSearchParameters__ParseFromAttributes(this, info); }

void GreedySearch::Init(const OpKernelInfo& info) { g_host_cpu.GreedySearch__Init(this, info); }

Status GreedySearch::Compute(OpKernelContext* ctx) const { return g_host_cpu.GreedySearch__Compute(this, ctx); }

Status GreedySearch::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name,
                                                const SessionState& subgraph_session_state) {
  return g_host_cpu.GreedySearch__SetupSubgraphExecutionInfo(this, session_state, attribute_name,
                                                             subgraph_session_state);
}

void Sampling::Init(const OpKernelInfo& info) { g_host_cpu.Sampling__Init(this, info); }

Status Sampling::Compute(OpKernelContext* ctx) const { return g_host_cpu.Sampling__Compute(this, ctx); }

Status Sampling::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name,
                                            const SessionState& subgraph_session_state) {
  return g_host_cpu.Sampling__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state);
}
#endif  // !defined(DISABLE_GENERATION_OPS)
}  // namespace transformers

#ifdef ENABLE_ATEN
Status ATen::Compute(OpKernelContext* p_ctx) const { return g_host_cpu.ATen__Compute(this, p_ctx); }
#endif
}  // namespace contrib
#endif

void If::Init(const OpKernelInfo& info) { g_host_cpu.If__Init(this, info); }
Status If::Compute(OpKernelContext* ctx) const { return g_host_cpu.If__Compute(this, ctx); }
Status If::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) { return g_host_cpu.If__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state); }

void Loop::Init(const OpKernelInfo& info) { g_host_cpu.Loop__Init(this, info); }
Status Loop::Compute(OpKernelContext* ctx) const { return g_host_cpu.Loop__Compute(this, ctx); }
Status Loop::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) { return g_host_cpu.Loop__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state); }

template <>
void Scan<8>::Init(const OpKernelInfo& info) { g_host_cpu.Scan__Init(this, info); }
template <>
void Scan<9>::Init(const OpKernelInfo& info) { g_host_cpu.Scan__Init(this, info); }
template <>
Status Scan<8>::Compute(OpKernelContext* ctx) const { return g_host_cpu.Scan__Compute(this, ctx); }
template <>
Status Scan<9>::Compute(OpKernelContext* ctx) const { return g_host_cpu.Scan__Compute(this, ctx); }
template <>
Status Scan<8>::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) { return g_host_cpu.Scan__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state); }
template <>
Status Scan<9>::SetupSubgraphExecutionInfo(const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) { return g_host_cpu.Scan__SetupSubgraphExecutionInfo(this, session_state, attribute_name, subgraph_session_state); }

void* AllocateBufferWithOptions(IAllocator& allocator, size_t size, bool use_reserve, Stream* stream, WaitNotificationFn wait_fn) { return g_host->Allocator__AllocateBufferWithOptions(allocator, size, use_reserve, stream, wait_fn); }

#ifdef ENABLE_TRAINING_OPS
namespace contrib {
Status Group::Compute(OpKernelContext* context) const { return g_host_cpu.contrib__Group__Compute(this, context); }
Status PassThrough::Compute(OpKernelContext* context) const { return g_host_cpu.contrib__PassThrough__Compute(this, context); }
Status AdamWOptimizerBase::PrepareForCompute(OpKernelContext* ctx, AdamWOptimizerBase::Prepare& prepare) const {
  return g_host_cpu.contrib__AdamWOptimizerBase__PrepareForCompute(this, ctx, reinterpret_cast<contrib__AdamWOptimizerBase__Prepare&>(prepare));
}
Status SGDOptimizerV2Base::PrepareForCompute(OpKernelContext* ctx, SGDOptimizerV2Base::Prepare& prepare) const {
  return g_host_cpu.contrib__SGDOptimizerV2Base__PrepareForCompute(this, ctx, reinterpret_cast<contrib__SGDOptimizerV2Base__Prepare&>(prepare));
}
void ShrunkenGatherCommon::CheckInput(const Tensor* input_tensor, const Tensor* indices_tensor, int64_t axis_in) const {
  return g_host_cpu.contrib__ShrunkenGatherCommon__CheckInput(this, input_tensor, indices_tensor, axis_in);
}
}  // namespace contrib
#endif

#ifdef ENABLE_TRAINING
namespace contrib {
Status YieldOp::Compute(OpKernelContext* context) const { return g_host_cpu.contrib__YieldOp__Compute(this, context); }
}  // namespace contrib

#ifdef ENABLE_TRAINING_TORCH_INTEROP
namespace contrib {
void PythonOpBase::Init(const OpKernelInfo& info) { return g_host->contrib__PythonOpBase__Init(this, info); }
void PythonOpBase::Clear() { return g_host->contrib__PythonOpBase__Clear(this); }
void PythonOpBase::RunForward(OpKernelContext* context, void** diff_ctx, std::vector<OrtValue>& returned_ortvalues) const {
  return g_host->contrib__PythonOpBase__RunForward(this, context, diff_ctx, returned_ortvalues);
}
void PythonOpBase::SetOutputs(OpKernelContext* context, void* diff_ctx, std::vector<OrtValue>& returned_args) const {
  return g_host->contrib__PythonOpBase__SetOutputs(this, context, diff_ctx, returned_args);
}

void PythonOpGradBase::Init(const OpKernelInfo& info) { return g_host->contrib__PythonOpGradBase__Init(this, info); }
void PythonOpGradBase::RunBackward(OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) const {
  return g_host->contrib__PythonOpGradBase__RunBackward(this, context, returned_ortvalues);
}
void PythonOpGradBase::SetOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_args) const {
  return g_host->contrib__PythonOpGradBase__SetOutputs(this, context, returned_args);
}
}  // namespace contrib

namespace language_interop_ops {
namespace torch {
void RefCountTracker::DumpDetails(const std::string& phase_name) const {
  return g_host->RefCountTracker__DumpDetails(this, phase_name);
}

}  // namespace torch
}  // namespace language_interop_ops
#endif

#endif

#ifdef ENABLE_TRITON
namespace contrib {
Status TritonOp::Compute(OpKernelContext* context) const {
  return g_host_cpu.contrib__TritonOp__Compute(this, context);
}
}  // namespace contrib
#endif

#endif

#if defined(USE_CANN)
RandomGenerator& RandomGenerator::Default() { return g_host->RandomGenerator__Default(); }
void* AllocateBufferWithOptions(IAllocator& allocator, size_t size, bool use_reserve, Stream* stream,
                                WaitNotificationFn wait_fn) {
  return g_host->Allocator__AllocateBufferWithOptions(allocator, size, use_reserve, stream, wait_fn);
}

namespace cann {
std::unique_ptr<Model> CreateModel(const GraphViewer& graph_viewer, const logging::Logger& logger) {
  return g_host->cann__CreateModel(graph_viewer, logger);
}
}  // namespace cann
#endif

void MurmurHash3::x86_128(const void* key, size_t len, uint32_t seed, void* out) {
  return g_host->MurmurHash3__x86_128(key, len, seed, out);
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
Status LoadDynamicLibrary(onnxruntime::PathString library_name) {
  return g_host->LoadDynamicLibrary(library_name);
}
#endif

#ifdef _WIN32
std::string ToUTF8String(std::wstring_view s) {
  return g_host->ToUTF8String(std::wstring{s});
}

std::wstring ToWideString(std::string_view s) {
  return g_host->ToWideString(std::string{s});
}
#endif  // _WIN32
}  // namespace onnxruntime
