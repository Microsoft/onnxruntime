// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api.h" instead.
// If interested in trying out features of the new experimental C++ API, include "experimental_onnxruntime_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

namespace Ort {

inline void ThrowOnError(const OrtApi& ort, OrtStatus* status) {
  if (status) {
    std::string error_message = ort.GetErrorMessage(status);
    OrtErrorCode error_code = ort.GetErrorCode(status);
    ort.ReleaseStatus(status);
    ORT_CXX_API_THROW(std::move(error_message), error_code);
  }
}

inline void ThrowOnError(OrtStatus* status) {
  ThrowOnError(GetApi(), status);
}

// This template converts a C++ type into it's ONNXTensorElementDataType
template <typename T>
struct TypeToTensorType;
template <>
struct TypeToTensorType<float> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
template <>
struct TypeToTensorType<Float16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
template <>
struct TypeToTensorType<BFloat16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; };
template <>
struct TypeToTensorType<double> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };
template <>
struct TypeToTensorType<int8_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; };
template <>
struct TypeToTensorType<int16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; };
template <>
struct TypeToTensorType<int32_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };
template <>
struct TypeToTensorType<int64_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; };
template <>
struct TypeToTensorType<uint8_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; };
template <>
struct TypeToTensorType<uint16_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; };
template <>
struct TypeToTensorType<uint32_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32; };
template <>
struct TypeToTensorType<uint64_t> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64; };
template <>
struct TypeToTensorType<bool> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL; };

inline MemoryAllocation::MemoryAllocation(OrtAllocator* allocator, void* p, size_t size)
    : allocator_(allocator), p_(p), size_(size) {
}

inline MemoryAllocation::~MemoryAllocation() {
  if (p_ != nullptr) {
    // We do not throw out of destructor
    auto ret = GetApi().AllocatorFree(allocator_, p_);
    static_cast<void>(ret);
  }
}

inline MemoryAllocation::MemoryAllocation(MemoryAllocation&& o) noexcept : allocator_(nullptr), p_(nullptr), size_(0) {
  *this = std::move(o);
}

inline MemoryAllocation& MemoryAllocation::operator=(MemoryAllocation&& o) noexcept {
  OrtAllocator* alloc = nullptr;
  void* p = nullptr;
  size_t sz = 0;

  // Swap out this
  std::swap(alloc, allocator_);
  std::swap(p, p_);
  std::swap(sz, size_);

  // Swap with incoming
  std::swap(allocator_, o.allocator_);
  std::swap(p_, o.p_);
  std::swap(size_, o.size_);

  // Destroy this instance if needed
  MemoryAllocation this_alloc(alloc, p, sz);
  return *this;
}

inline AllocatorWithDefaultOptions::AllocatorWithDefaultOptions() {
  ThrowOnError(GetApi().GetAllocatorWithDefaultOptions(&p_));
}

inline void* AllocatorWithDefaultOptions::Alloc(size_t size) {
  void* out;
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, &out));
  return out;
}

inline MemoryAllocation Ort::AllocatorWithDefaultOptions::GetAllocation(size_t size) {
  void* out;
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, &out));
  MemoryAllocation result(p_, out, size);
  return result;
}

inline void AllocatorWithDefaultOptions::Free(void* p) {
  ThrowOnError(GetApi().AllocatorFree(p_, p));
}

inline const OrtMemoryInfo* AllocatorWithDefaultOptions::GetInfo() const {
  const OrtMemoryInfo* out;
  ThrowOnError(GetApi().AllocatorGetInfo(p_, &out));
  return out;
}

template <typename B>
inline std::string BaseMemoryInfo<B>::GetAllocatorName() const {
  const char* name = nullptr;
  ThrowOnError(GetApi().MemoryInfoGetName(*this, &name));
  return std::string(name);
}

template <typename B>
inline OrtAllocatorType BaseMemoryInfo<B>::GetAllocatorType() const {
  OrtAllocatorType type;
  ThrowOnError(GetApi().MemoryInfoGetType(*this, &type));
  return type;
}

template <typename B>
int BaseMemoryInfo<B>::GetDeviceId() const {
  int id = 0;
  ThrowOnError(GetApi().MemoryInfoGetId(*this, &id));
  return id;
}

template <typename B>
inline OrtMemType BaseMemoryInfo<B>::GetMemoryType() const {
  OrtMemType type;
  ThrowOnError(GetApi().MemoryInfoGetMemType(*this, &type));
  return type;
}

template <typename B>
template <typename U>
inline bool BaseMemoryInfo<B>::operator==(const BaseMemoryInfo<U>& o) const {
  int comp_result = 0;
  ThrowOnError(Ort::GetApi().CompareMemoryInfo(*this, o, &comp_result));
  return comp_result == 0;
}

inline MemoryInfo MemoryInfo::CreateCpu(OrtAllocatorType type, OrtMemType mem_type) {
  OrtMemoryInfo* p;
  ThrowOnError(GetApi().CreateCpuMemoryInfo(type, mem_type, &p));
  return MemoryInfo(p);
}

inline MemoryInfo::MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type) {
  ThrowOnError(GetApi().CreateMemoryInfo(name, type, id, mem_type, &p_));
}

inline Allocator::Allocator(const Session& sess, const MemoryInfo& mem_info) {
  ThrowOnError(GetApi().CreateAllocator(sess, mem_info, &p_));
}

inline void* Allocator::Alloc(size_t size) const {
  void* out = nullptr;
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, &out));
  return out;
}

inline MemoryAllocation Ort::Allocator::GetAllocation(size_t size) {
  void* out = nullptr;
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, &out));
  MemoryAllocation result(p_, out, size);
  return result;
}

inline void Allocator::Free(void* p) const {
  ThrowOnError(GetApi().AllocatorFree(p_, p));
}

inline UnownedMemoryInfo Allocator::GetInfo() const {
  const OrtMemoryInfo* out = nullptr;
  ThrowOnError(GetApi().AllocatorGetInfo(p_, &out));
  return UnownedMemoryInfo(out);
}

inline IoBinding::IoBinding(Session& session) {
  ThrowOnError(GetApi().CreateIoBinding(session, &p_));
}

inline void IoBinding::BindInput(const char* name, const Value& value) {
  ThrowOnError(GetApi().BindInput(p_, name, value));
}

inline void IoBinding::BindOutput(const char* name, const Value& value) {
  ThrowOnError(GetApi().BindOutput(p_, name, value));
}

inline void IoBinding::BindOutput(const char* name, const MemoryInfo& mem_info) {
  ThrowOnError(GetApi().BindOutputToDevice(p_, name, mem_info));
}

inline std::vector<std::string> IoBinding::GetOutputNamesHelper(OrtAllocator* allocator) const {
  std::vector<std::string> result;
  auto free_fn = [allocator](void* p) { if (p) allocator->Free(allocator, p); };
  using Ptr = std::unique_ptr<void, decltype(free_fn)>;

  char* buffer = nullptr;
  size_t* lengths = nullptr;
  size_t count = 0;
  ThrowOnError(GetApi().GetBoundOutputNames(p_, allocator, &buffer, &lengths, &count));

  if (count == 0) {
    return result;
  }

  Ptr buffer_g(buffer, free_fn);
  Ptr lengths_g(lengths, free_fn);

  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto sz = *lengths;
    result.emplace_back(buffer, sz);
    buffer += sz;
    ++lengths;
  }
  return result;
}

inline std::vector<std::string> IoBinding::GetOutputNames() const {
  AllocatorWithDefaultOptions allocator;
  return GetOutputNamesHelper(allocator);
}

inline std::vector<std::string> IoBinding::GetOutputNames(Allocator& allocator) const {
  return GetOutputNamesHelper(allocator);
}

inline std::vector<Value> Ort::IoBinding::GetOutputValuesHelper(OrtAllocator* allocator) const {
  std::vector<Value> result;
  size_t owned = 0;
  size_t output_count = 0;
  // Lambda to release the buffer when no longer needed and
  // make sure that we destroy all instances on exception
  auto free_fn = [&owned, &output_count, allocator](OrtValue** buffer) {
    if (buffer) {
      while (owned < output_count) {
        auto* p = buffer + owned++;
        GetApi().ReleaseValue(*p);
      }
      allocator->Free(allocator, buffer);
    }
  };
  using Ptr = std::unique_ptr<OrtValue*, decltype(free_fn)>;

  OrtValue** output_buffer = nullptr;
  ThrowOnError(GetApi().GetBoundOutputValues(p_, allocator, &output_buffer, &output_count));
  if (output_count == 0) {
    return result;
  }

  Ptr buffer_g(output_buffer, free_fn);

  result.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    result.emplace_back(output_buffer[i]);
    ++owned;
  }
  return result;
}

inline std::vector<Value> Ort::IoBinding::GetOutputValues(Allocator& allocator) const {
  return GetOutputValuesHelper(allocator);
}

inline std::vector<Value> Ort::IoBinding::GetOutputValues() const {
  AllocatorWithDefaultOptions allocator;
  return GetOutputValuesHelper(allocator);
}

inline void IoBinding::ClearBoundInputs() {
  GetApi().ClearBoundInputs(p_);
}

inline void IoBinding::ClearBoundOutputs() {
  GetApi().ClearBoundOutputs(p_);
}

inline ArenaCfg::ArenaCfg(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) {
  ThrowOnError(GetApi().CreateArenaCfg(max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, &p_));
}

inline Env::Env(OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnv(logging_level, logid, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param) {
  ThrowOnError(GetApi().CreateEnvWithCustomLogger(logging_function, logger_param, logging_level, logid, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnvWithGlobalThreadPools(logging_level, logid, tp_options, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env::Env(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
                OrtLoggingLevel logging_level, _In_ const char* logid) {
  ThrowOnError(GetApi().CreateEnvWithCustomLoggerAndGlobalThreadPools(logging_function, logger_param, logging_level, logid, tp_options, &p_));
  if (strcmp(logid, "onnxruntime-node") == 0) {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_NODEJS));
  } else {
    ThrowOnError(GetApi().SetLanguageProjection(p_, OrtLanguageProjection::ORT_PROJECTION_CPLUSPLUS));
  }
}

inline Env& Env::EnableTelemetryEvents() {
  ThrowOnError(GetApi().EnableTelemetryEvents(p_));
  return *this;
}

inline Env& Env::DisableTelemetryEvents() {
  ThrowOnError(GetApi().DisableTelemetryEvents(p_));
  return *this;
}

inline Env& Env::CreateAndRegisterAllocator(const OrtMemoryInfo* mem_info, const OrtArenaCfg* arena_cfg) {
  ThrowOnError(GetApi().CreateAndRegisterAllocator(p_, mem_info, arena_cfg));
  return *this;
}

inline CustomOpDomain::CustomOpDomain(const char* domain) {
  ThrowOnError(GetApi().CreateCustomOpDomain(domain, &p_));
}

inline void CustomOpDomain::Add(OrtCustomOp* op) {
  ThrowOnError(GetApi().CustomOpDomain_Add(p_, op));
}

inline RunOptions::RunOptions() {
  ThrowOnError(GetApi().CreateRunOptions(&p_));
}

inline RunOptions& RunOptions::SetRunLogVerbosityLevel(int level) {
  ThrowOnError(GetApi().RunOptionsSetRunLogVerbosityLevel(p_, level));
  return *this;
}

inline RunOptions& RunOptions::SetRunLogSeverityLevel(int level) {
  ThrowOnError(GetApi().RunOptionsSetRunLogSeverityLevel(p_, level));
  return *this;
}

inline int RunOptions::GetRunLogVerbosityLevel() const {
  int out;
  ThrowOnError(GetApi().RunOptionsGetRunLogVerbosityLevel(p_, &out));
  return out;
}

inline RunOptions& RunOptions::SetRunTag(const char* run_tag) {
  ThrowOnError(GetApi().RunOptionsSetRunTag(p_, run_tag));
  return *this;
}

inline const char* RunOptions::GetRunTag() const {
  const char* out;
  ThrowOnError(GetApi().RunOptionsGetRunTag(p_, &out));
  return out;
}

inline RunOptions& RunOptions::AddConfigEntry(const char* config_key, const char* config_value) {
  ThrowOnError(GetApi().AddRunConfigEntry(p_, config_key, config_value));
  return *this;
}

inline RunOptions& RunOptions::SetTerminate() {
  ThrowOnError(GetApi().RunOptionsSetTerminate(p_));
  return *this;
}

inline RunOptions& RunOptions::UnsetTerminate() {
  ThrowOnError(GetApi().RunOptionsUnsetTerminate(p_));
  return *this;
}

inline SessionOptions::SessionOptions() {
  ThrowOnError(GetApi().CreateSessionOptions(&p_));
}

inline SessionOptions SessionOptions::Clone() const {
  OrtSessionOptions* out;
  ThrowOnError(GetApi().CloneSessionOptions(p_, &out));
  return SessionOptions{out};
}

inline SessionOptions& SessionOptions::SetIntraOpNumThreads(int intra_op_num_threads) {
  ThrowOnError(GetApi().SetIntraOpNumThreads(p_, intra_op_num_threads));
  return *this;
}

inline SessionOptions& SessionOptions::SetInterOpNumThreads(int inter_op_num_threads) {
  ThrowOnError(GetApi().SetInterOpNumThreads(p_, inter_op_num_threads));
  return *this;
}

inline SessionOptions& SessionOptions::SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level) {
  ThrowOnError(GetApi().SetSessionGraphOptimizationLevel(p_, graph_optimization_level));
  return *this;
}

inline SessionOptions& SessionOptions::SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_filepath) {
  ThrowOnError(GetApi().SetOptimizedModelFilePath(p_, optimized_model_filepath));
  return *this;
}

inline SessionOptions& SessionOptions::EnableProfiling(const ORTCHAR_T* profile_file_prefix) {
  ThrowOnError(GetApi().EnableProfiling(p_, profile_file_prefix));
  return *this;
}

inline SessionOptions& SessionOptions::DisableProfiling() {
  ThrowOnError(GetApi().DisableProfiling(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableOrtCustomOps() {
  ThrowOnError(GetApi().EnableOrtCustomOps(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableMemPattern() {
  ThrowOnError(GetApi().EnableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableMemPattern() {
  ThrowOnError(GetApi().DisableMemPattern(p_));
  return *this;
}

inline SessionOptions& SessionOptions::EnableCpuMemArena() {
  ThrowOnError(GetApi().EnableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::DisableCpuMemArena() {
  ThrowOnError(GetApi().DisableCpuMemArena(p_));
  return *this;
}

inline SessionOptions& SessionOptions::SetExecutionMode(ExecutionMode execution_mode) {
  ThrowOnError(GetApi().SetSessionExecutionMode(p_, execution_mode));
  return *this;
}

inline SessionOptions& SessionOptions::SetLogId(const char* logid) {
  ThrowOnError(GetApi().SetSessionLogId(p_, logid));
  return *this;
}

inline SessionOptions& SessionOptions::SetLogSeverityLevel(int level) {
  ThrowOnError(GetApi().SetSessionLogSeverityLevel(p_, level));
  return *this;
}

inline SessionOptions& SessionOptions::Add(OrtCustomOpDomain* custom_op_domain) {
  ThrowOnError(GetApi().AddCustomOpDomain(p_, custom_op_domain));
  return *this;
}

inline SessionOptions& SessionOptions::AddConfigEntry(const char* config_key, const char* config_value) {
  ThrowOnError(GetApi().AddSessionConfigEntry(p_, config_key, config_value));
  return *this;
}

inline SessionOptions& SessionOptions::AddInitializer(const char* name, const OrtValue* ort_val) {
  ThrowOnError(GetApi().AddInitializer(p_, name, ort_val));
  return *this;
}

inline SessionOptions& SessionOptions::AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CUDA(p_, &provider_options));
  return *this;
}

inline SessionOptions& SessionOptions::AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_ROCM(p_, &provider_options));
  return *this;
}

inline SessionOptions& SessionOptions::AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_TensorRT(p_, &provider_options));
  return *this;
}

inline SessionOptions& SessionOptions::AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options) {
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpenVINO(p_, &provider_options));
  return *this;
}

inline Session::Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
  ThrowOnError(GetApi().CreateSession(env, model_path, options, &p_));
}

inline Session::Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options,
                        OrtPrepackedWeightsContainer* prepacked_weights_container) {
  ThrowOnError(GetApi().CreateSessionWithPrepackedWeightsContainer(env, model_path, options, prepacked_weights_container, &p_));
}

inline Session::Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options) {
  ThrowOnError(GetApi().CreateSessionFromArray(env, model_data, model_data_length, options, &p_));
}

inline std::vector<Value> Session::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                                       const char* const* output_names, size_t output_names_count) {
  std::vector<Ort::Value> output_values;
  for (size_t i = 0; i < output_names_count; i++)
    output_values.emplace_back(nullptr);
  Run(run_options, input_names, input_values, input_count, output_names, output_values.data(), output_names_count);
  return output_values;
}

inline void Session::Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                         const char* const* output_names, Value* output_values, size_t output_count) {
  static_assert(sizeof(Value) == sizeof(OrtValue*), "Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely");
  auto ort_input_values = reinterpret_cast<const OrtValue**>(const_cast<Value*>(input_values));
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ThrowOnError(GetApi().Run(p_, run_options, input_names, ort_input_values, input_count, output_names, output_count, ort_output_values));
}

inline void Session::Run(const RunOptions& run_options, const IoBinding& io_binding) {
  ThrowOnError(GetApi().RunWithBinding(p_, run_options, io_binding));
}

inline size_t Session::GetInputCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetInputCount(p_, &out));
  return out;
}

inline size_t Session::GetOutputCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetOutputCount(p_, &out));
  return out;
}

inline size_t Session::GetOverridableInitializerCount() const {
  size_t out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerCount(p_, &out));
  return out;
}

inline char* Session::GetInputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetInputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOutputName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetOutputName(p_, index, allocator, &out));
  return out;
}

inline char* Session::GetOverridableInitializerName(size_t index, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerName(p_, index, allocator, &out));
  return out;
}

inline char* Session::EndProfiling(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().SessionEndProfiling(p_, allocator, &out));
  return out;
}

inline uint64_t Session::GetProfilingStartTimeNs() const {
  uint64_t out;
  ThrowOnError(GetApi().SessionGetProfilingStartTimeNs(p_, &out));
  return out;
}

inline ModelMetadata Session::GetModelMetadata() const {
  OrtModelMetadata* out;
  ThrowOnError(GetApi().SessionGetModelMetadata(p_, &out));
  return ModelMetadata{out};
}

inline char* ModelMetadata::GetProducerName(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetProducerName(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetGraphName(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetGraphName(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetDomain(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetDomain(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetDescription(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetDescription(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::GetGraphDescription(OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataGetGraphDescription(p_, allocator, &out));
  return out;
}

inline char* ModelMetadata::LookupCustomMetadataMap(const char* key, OrtAllocator* allocator) const {
  char* out;
  ThrowOnError(GetApi().ModelMetadataLookupCustomMetadataMap(p_, allocator, key, &out));
  return out;
}

inline char** ModelMetadata::GetCustomMetadataMapKeys(OrtAllocator* allocator, _Out_ int64_t& num_keys) const {
  char** out;
  ThrowOnError(GetApi().ModelMetadataGetCustomMetadataMapKeys(p_, allocator, &out, &num_keys));
  return out;
}

inline int64_t ModelMetadata::GetVersion() const {
  int64_t out;
  ThrowOnError(GetApi().ModelMetadataGetVersion(p_, &out));
  return out;
}

inline TypeInfo Session::GetInputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetInputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOutputTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetOutputTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline TypeInfo Session::GetOverridableInitializerTypeInfo(size_t index) const {
  OrtTypeInfo* out;
  ThrowOnError(GetApi().SessionGetOverridableInitializerTypeInfo(p_, index, &out));
  return TypeInfo{out};
}

inline ONNXTensorElementDataType TensorTypeAndShapeInfo::GetElementType() const {
  ONNXTensorElementDataType out;
  ThrowOnError(GetApi().GetTensorElementType(p_, &out));
  return out;
}

inline size_t TensorTypeAndShapeInfo::GetElementCount() const {
  size_t out;
  ThrowOnError(GetApi().GetTensorShapeElementCount(p_, &out));
  return static_cast<size_t>(out);
}

inline size_t TensorTypeAndShapeInfo::GetDimensionsCount() const {
  size_t out;
  ThrowOnError(GetApi().GetDimensionsCount(p_, &out));
  return out;
}

inline void TensorTypeAndShapeInfo::GetDimensions(int64_t* values, size_t values_count) const {
  ThrowOnError(GetApi().GetDimensions(p_, values, values_count));
}

inline void TensorTypeAndShapeInfo::GetSymbolicDimensions(const char** values, size_t values_count) const {
  ThrowOnError(GetApi().GetSymbolicDimensions(p_, values, values_count));
}

inline std::vector<int64_t> TensorTypeAndShapeInfo::GetShape() const {
  std::vector<int64_t> out(GetDimensionsCount(), 0);
  GetDimensions(out.data(), out.size());
  return out;
}

inline Unowned<TensorTypeAndShapeInfo> TypeInfo::GetTensorTypeAndShapeInfo() const {
  const OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToTensorInfo(p_, &out));
  return Unowned<TensorTypeAndShapeInfo>(const_cast<OrtTensorTypeAndShapeInfo*>(out));
}

inline Unowned<SequenceTypeInfo> TypeInfo::GetSequenceTypeInfo() const {
  const OrtSequenceTypeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToSequenceTypeInfo(p_, &out));
  return Unowned<SequenceTypeInfo>{const_cast<OrtSequenceTypeInfo*>(out)};
}

inline TypeInfo SequenceTypeInfo::GetSequenceElementType() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetSequenceElementType(p_, &output));
  return TypeInfo{output};
}

inline Unowned<MapTypeInfo> TypeInfo::GetMapTypeInfo() const {
  const OrtMapTypeInfo* out;
  ThrowOnError(GetApi().CastTypeInfoToMapTypeInfo(p_, &out));
  return Unowned<MapTypeInfo>{const_cast<OrtMapTypeInfo*>(out)};
}

inline ONNXTensorElementDataType MapTypeInfo::GetMapKeyType() const {
  ONNXTensorElementDataType out;
  ThrowOnError(GetApi().GetMapKeyType(p_, &out));
  return out;
}

inline TypeInfo MapTypeInfo::GetMapValueType() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetMapValueType(p_, &output));
  return TypeInfo{output};
}

inline ONNXType TypeInfo::GetONNXType() const {
  ONNXType out;
  ThrowOnError(GetApi().GetOnnxTypeFromTypeInfo(p_, &out));
  return out;
}

template <typename T>
inline Value Value::CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len) {
  return CreateTensor(info, p_data, p_data_element_count * sizeof(T), shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                                 ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, type, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len) {
  return CreateTensor(allocator, shape, shape_len, TypeToTensorType<T>::type);
}

inline Value Value::CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateTensorAsOrtValue(allocator, shape, shape_len, type, &out));
  return Value{out};
}

inline Value Value::CreateMap(Value& keys, Value& values) {
  OrtValue* out;
  OrtValue* inputs[2] = {keys, values};
  ThrowOnError(GetApi().CreateValue(inputs, 2, ONNX_TYPE_MAP, &out));
  return Value{out};
}

inline Value Value::CreateSequence(std::vector<Value>& values) {
  OrtValue* out;
  std::vector<OrtValue*> values_ort{values.data(), values.data() + values.size()};
  ThrowOnError(GetApi().CreateValue(values_ort.data(), values_ort.size(), ONNX_TYPE_SEQUENCE, &out));
  return Value{out};
}

template <typename T>
inline Value Value::CreateOpaque(const char* domain, const char* type_name, const T& data_container) {
  OrtValue* out;
  ThrowOnError(GetApi().CreateOpaqueValue(domain, type_name, &data_container, sizeof(T), &out));
  return Value{out};
}

template <typename T>
inline void Value::GetOpaqueData(const char* domain, const char* type_name, T& out) const {
  ThrowOnError(GetApi().GetOpaqueValue(domain, type_name, p_, &out, sizeof(T)));
}

inline bool Value::IsTensor() const {
  int out;
  ThrowOnError(GetApi().IsTensor(p_, &out));
  return out != 0;
}

inline bool Value::HasElement() const {
  int out;
  ThrowOnError(GetApi().HasElement(p_, &out));
  return out != 0;
}

inline size_t Value::GetCount() const {
  size_t out;
  ThrowOnError(GetApi().GetValueCount(p_, &out));
  return out;
}

inline Value Value::GetValue(int index, OrtAllocator* allocator) const {
  OrtValue* out;
  ThrowOnError(GetApi().GetValue(p_, index, allocator, &out));
  return Value{out};
}

inline size_t Value::GetStringTensorDataLength() const {
  size_t out;
  ThrowOnError(GetApi().GetStringTensorDataLength(p_, &out));
  return out;
}

inline size_t Value::GetStringTensorElementLength(size_t element_index) const {
  size_t out;
  ThrowOnError(GetApi().GetStringTensorElementLength(p_, element_index, &out));
  return out;
}

inline void Value::GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const {
  ThrowOnError(GetApi().GetStringTensorContent(p_, buffer, buffer_length, offsets, offsets_count));
}

inline void Value::GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const {
  ThrowOnError(GetApi().GetStringTensorElement(p_, buffer_length, element_index, buffer));
}

inline void Value::FillStringTensor(const char* const* s, size_t s_len) {
  ThrowOnError(GetApi().FillStringTensor(p_, s, s_len));
}

inline void Value::FillStringTensorElement(const char* s, size_t index) {
  ThrowOnError(GetApi().FillStringTensorElement(p_, s, index));
}

template <typename T>
T* Value::GetTensorMutableData() {
  T* out;
  ThrowOnError(GetApi().GetTensorMutableData(p_, (void**)&out));
  return out;
}

template <typename T>
const T* Value::GetTensorData() const {
  T* out;
  ThrowOnError(GetApi().GetTensorMutableData(p_, (void**)&out));
  return out;
}

template <typename T>
inline T& Value::At(const std::vector<int64_t>& location) {
  static_assert(!std::is_same<T, std::string>::value, "this api does not support std::string");
  T* out;
  ThrowOnError(GetApi().TensorAt(p_, location.data(), location.size(), (void**)&out));
  return *out;
}

inline TypeInfo Value::GetTypeInfo() const {
  OrtTypeInfo* output;
  ThrowOnError(GetApi().GetTypeInfo(p_, &output));
  return TypeInfo{output};
}

inline TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  OrtTensorTypeAndShapeInfo* output;
  ThrowOnError(GetApi().GetTensorTypeAndShape(p_, &output));
  return TensorTypeAndShapeInfo{output};
}

//
// Custom OP API Inlines
//
inline void CustomOpApi::ThrowOnError(OrtStatus* status) {
  Ort::ThrowOnError(api_, status);
}

template <>
inline float CustomOpApi::KernelInfoGetAttribute<float>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  float out;
  ThrowOnError(api_.KernelInfoGetAttribute_float(info, name, &out));
  return out;
}

template <>
inline int64_t CustomOpApi::KernelInfoGetAttribute<int64_t>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  int64_t out;
  ThrowOnError(api_.KernelInfoGetAttribute_int64(info, name, &out));
  return out;
}

template <>
inline std::string CustomOpApi::KernelInfoGetAttribute<std::string>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  size_t size = 0;
  std::string out;

  // Feed nullptr for the data buffer to query the true size of the string attribute
  OrtStatus* status = api_.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttribute_string(info, name, &out[0], &size));
    out.resize(size - 1);  // remove the terminating character '\0'
  } else {
    ThrowOnError(status);
  }
  return out;
}

template <>
inline std::vector<float> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  size_t size = 0;
  std::vector<float> out;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status = api_.KernelInfoGetAttributeArray_float(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttributeArray_float(info, name, out.data(), &size));
  } else {
    ThrowOnError(status);
  }
  return out;
}

template <>
inline std::vector<int64_t> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  size_t size = 0;
  std::vector<int64_t> out;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status = api_.KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttributeArray_int64(info, name, out.data(), &size));
  } else {
    ThrowOnError(status);
  }
  return out;
}
inline OrtTensorTypeAndShapeInfo* CustomOpApi::GetTensorTypeAndShape(_In_ const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(api_.GetTensorTypeAndShape(value, &out));
  return out;
}

inline size_t CustomOpApi::GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetTensorShapeElementCount(info, &out));
  return out;
}

inline ONNXTensorElementDataType CustomOpApi::GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) {
  ONNXTensorElementDataType out;
  ThrowOnError(api_.GetTensorElementType(info, &out));
  return out;
}

inline size_t CustomOpApi::GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetDimensionsCount(info, &out));
  return out;
}

inline void CustomOpApi::GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  ThrowOnError(api_.GetDimensions(info, dim_values, dim_values_length));
}

inline void CustomOpApi::SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) {
  ThrowOnError(api_.SetDimensions(info, dim_values, dim_count));
}

template <typename T>
inline T* CustomOpApi::GetTensorMutableData(_Inout_ OrtValue* value) {
  T* data;
  ThrowOnError(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
  return data;
}

template <typename T>
inline const T* CustomOpApi::GetTensorData(_Inout_ const OrtValue* value) {
  return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
}

inline std::vector<int64_t> CustomOpApi::GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
  std::vector<int64_t> output(GetDimensionsCount(info));
  GetDimensions(info, output.data(), output.size());
  return output;
}

inline void CustomOpApi::ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) {
  api_.ReleaseTensorTypeAndShapeInfo(input);
}

inline size_t CustomOpApi::KernelContext_GetInputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetInputCount(context, &out));
  return out;
}

inline const OrtValue* CustomOpApi::KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
  const OrtValue* out;
  ThrowOnError(api_.KernelContext_GetInput(context, index, &out));
  return out;
}

inline size_t CustomOpApi::KernelContext_GetOutputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetOutputCount(context, &out));
  return out;
}

inline OrtValue* CustomOpApi::KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index,
                                                      _In_ const int64_t* dim_values, size_t dim_count) {
  OrtValue* out;
  ThrowOnError(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, &out));
  return out;
}

inline SessionOptions& SessionOptions::DisablePerSessionThreads() {
  ThrowOnError(GetApi().DisablePerSessionThreads(p_));
  return *this;
}

inline std::vector<std::string> GetAvailableProviders() {
  int len;
  char** providers;
  const OrtApi& api = GetApi();
  ThrowOnError(api.GetAvailableProviders(&providers, &len));
  std::vector<std::string> available_providers(providers, providers + len);
  ThrowOnError(api.ReleaseAvailableProviders(providers, len));
  return available_providers;
}

SessionOptions& AddInitializer(const char* name, const OrtValue* ort_val);

}  // namespace Ort
