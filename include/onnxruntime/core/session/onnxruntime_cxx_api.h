// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The Ort C++ API is a header only wrapper around the Ort C API.
//
// The C++ API simplifies usage by returning values directly instead of error codes, throwing exceptions on errors
// and automatically releasing resources in the destructors.
//
// Each of the C++ wrapper classes holds only a pointer to the C internal object. Treat them like smart pointers.
// To create an empty object, pass 'nullptr' to the constructor (for example, Env e{nullptr};).
//
// Only move assignment between objects is allowed, there are no copy constructors. Some objects have explicit 'Clone'
// methods for this purpose.

#pragma once
#include "onnxruntime_c_api.h"
#include <cstddef>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

#ifdef ORT_NO_EXCEPTIONS
#include <iostream>
#endif

namespace Ort {

// All C++ methods that can fail will throw an exception of this type
struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

#ifdef ORT_NO_EXCEPTIONS
#define ORT_CXX_API_THROW(string, code)       \
  do {                                        \
    std::cerr << Ort::Exception(string, code) \
                     .what()                  \
              << std::endl;                   \
    abort();                                  \
  } while (false)
#else
#define ORT_CXX_API_THROW(string, code) \
  throw Ort::Exception(string, code)
#endif

// This is used internally by the C++ API. This class holds the global variable that points to the OrtApi, it's in a template so that we can define a global variable in a header and make
// it transparent to the users of the API.
template <typename T>
struct Global {
  static const OrtApi* api_;
};

// If macro ORT_API_MANUAL_INIT is defined, no static initialization will be performed. Instead, user must call InitApi() before using it.

template <typename T>
#ifdef ORT_API_MANUAL_INIT
const OrtApi* Global<T>::api_{};
inline void InitApi() { Global<void>::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION); }
#else
const OrtApi* Global<T>::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
#endif

// This returns a reference to the OrtApi interface in use, in case someone wants to use the C API functions
inline const OrtApi& GetApi() { return *Global<void>::api_; }

// This is a C++ wrapper for GetAvailableProviders() C API and returns
// a vector of strings representing the available execution providers.
std::vector<std::string> GetAvailableProviders();

// This is used internally by the C++ API. This macro is to make it easy to generate overloaded methods for all of the various OrtRelease* functions for every Ort* type
// This can't be done in the C API since C doesn't have function overloading.
#define ORT_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { GetApi().Release##NAME(ptr); }

ORT_DEFINE_RELEASE(Allocator);
ORT_DEFINE_RELEASE(MemoryInfo);
ORT_DEFINE_RELEASE(CustomOpDomain);
ORT_DEFINE_RELEASE(Env);
ORT_DEFINE_RELEASE(RunOptions);
ORT_DEFINE_RELEASE(Session);
ORT_DEFINE_RELEASE(SessionOptions);
ORT_DEFINE_RELEASE(TensorTypeAndShapeInfo);
ORT_DEFINE_RELEASE(SequenceTypeInfo);
ORT_DEFINE_RELEASE(MapTypeInfo);
ORT_DEFINE_RELEASE(TypeInfo);
ORT_DEFINE_RELEASE(Value);
ORT_DEFINE_RELEASE(ModelMetadata);
ORT_DEFINE_RELEASE(ThreadingOptions);
ORT_DEFINE_RELEASE(IoBinding);
ORT_DEFINE_RELEASE(ArenaCfg);

/*! \class Ort::Float16_t
  * \brief it is a structure that represents float16 data.
  * \details It is necessary for type dispatching to make use of C++ API
  * The type is implicitly convertible to/from uint16_t.
  * The size of the structure should align with uint16_t and one can freely cast
  * uint16_t buffers to/from Ort::Float16_t to feed and retrieve data.
  * 
  * Generally, you can feed any of your types as float16/blfoat16 data to create a tensor
  * on top of it, providing it can form a continuous buffer with 16-bit elements with no padding.
  * And you can also feed a array of uint16_t elements directly. For example,
  * 
  * \code{.unparsed}
  * uint16_t values[] = { 15360, 16384, 16896, 17408, 17664};
  * constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  * std::vector<int64_t> dims = {values_length};  // one dimensional example
  * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  * // Note we are passing bytes count in this api, not number of elements -> sizeof(values)
  * auto float16_tensor = Ort::Value::CreateTensor(info, values, sizeof(values), 
  *                                                dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  * \endcode
  * 
  * Here is another example, a little bit more elaborate. Let's assume that you use your own float16 type and you want to use
  * a templated version of the API above so the type is automatically set based on your type. You will need to supply an extra
  * template specialization.
  * 
  * \code{.unparsed}
  * namespace yours { struct half {}; } // assume this is your type, define this:
  * namespace Ort { 
  * template<>
  * struct TypeToTensorType<yours::half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
  * } //namespace Ort
  * 
  * std::vector<yours::half> values;
  * std::vector<int64_t> dims = {values.size()}; // one dimensional example
  * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  * // Here we are passing element count -> values.size()
  * auto float16_tensor = Ort::Value::CreateTensor<yours::half>(info, values.data(), values.size(), dims.data(), dims.size());
  * 
  *  \endcode
  */
struct Float16_t {
  uint16_t value;
  constexpr Float16_t() noexcept : value(0) {}
  constexpr Float16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const Float16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const Float16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(Float16_t) == sizeof(uint16_t), "Sizes must match");

/*! \class Ort::BFloat16_t
  * \brief is a structure that represents bfloat16 data.
  * \details It is necessary for type dispatching to make use of C++ API
  * The type is implicitly convertible to/from uint16_t.
  * The size of the structure should align with uint16_t and one can freely cast
  * uint16_t buffers to/from Ort::BFloat16_t to feed and retrieve data.
  * 
  * See also code examples for Float16_t above.
  */
struct BFloat16_t {
  uint16_t value;
  constexpr BFloat16_t() noexcept : value(0) {}
  constexpr BFloat16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const BFloat16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const BFloat16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(BFloat16_t) == sizeof(uint16_t), "Sizes must match");

// This is used internally by the C++ API. This is the common base class used by the wrapper objects.
template <typename T>
struct Base {
  using contained_type = T;

  Base() = default;
  Base(T* p) : p_{p} {
    if (!p)
      ORT_CXX_API_THROW("Allocation failure", ORT_FAIL);
  }
  ~Base() { OrtRelease(p_); }

  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* release() {
    T* p = p_;
    p_ = nullptr;
    return p;
  }

 protected:
  Base(const Base&) = delete;
  Base& operator=(const Base&) = delete;
  Base(Base&& v) noexcept : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) noexcept {
    OrtRelease(p_);
    p_ = v.p_;
    v.p_ = nullptr;
  }

  T* p_{};

  template <typename>
  friend struct Unowned;  // This friend line is needed to keep the centos C++ compiler from giving an error
};

template <typename T>
struct Base<const T> {
  using contained_type = const T;

  Base() = default;
  Base(const T* p) : p_{p} {
    if (!p)
      ORT_CXX_API_THROW("Invalid instance ptr", ORT_INVALID_ARGUMENT);
  }
  ~Base() = default;

  operator const T*() const { return p_; }

 protected:
  Base(const Base&) = delete;
  Base& operator=(const Base&) = delete;
  Base(Base&& v) noexcept : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) noexcept {
    p_ = v.p_;
    v.p_ = nullptr;
  }

  const T* p_{};
};

template <typename T>
struct Unowned : T {
  Unowned(decltype(T::p_) p) : T{p} {}
  Unowned(Unowned&& v) : T{v.p_} {}
  ~Unowned() { this->release(); }
};

struct AllocatorWithDefaultOptions;
struct MemoryInfo;
struct Env;
struct TypeInfo;
struct Value;
struct ModelMetadata;

struct Env : Base<OrtEnv> {
  Env(std::nullptr_t) {}
  Env(OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");
  Env(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");
  Env(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);
  Env(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
      OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");
  explicit Env(OrtEnv* p) : Base<OrtEnv>{p} {}

  Env& EnableTelemetryEvents();
  Env& DisableTelemetryEvents();

  Env& CreateAndRegisterAllocator(const OrtMemoryInfo* mem_info, const OrtArenaCfg* arena_cfg);

  static const OrtApi* s_api;
};

struct CustomOpDomain : Base<OrtCustomOpDomain> {
  explicit CustomOpDomain(std::nullptr_t) {}
  explicit CustomOpDomain(const char* domain);

  void Add(OrtCustomOp* op);
};

struct RunOptions : Base<OrtRunOptions> {
  RunOptions(std::nullptr_t) {}
  RunOptions();

  RunOptions& SetRunLogVerbosityLevel(int);
  int GetRunLogVerbosityLevel() const;

  RunOptions& SetRunLogSeverityLevel(int);
  int GetRunLogSeverityLevel() const;

  RunOptions& SetRunTag(const char* run_tag);
  const char* GetRunTag() const;

  RunOptions& AddConfigEntry(const char* config_key, const char* config_value);

  // terminate ALL currently executing Session::Run calls that were made using this RunOptions instance
  RunOptions& SetTerminate();
  // unset the terminate flag so this RunOptions instance can be used in a new Session::Run call
  RunOptions& UnsetTerminate();
};

struct SessionOptions : Base<OrtSessionOptions> {
  explicit SessionOptions(std::nullptr_t) {}
  SessionOptions();
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}

  SessionOptions Clone() const;

  SessionOptions& SetIntraOpNumThreads(int intra_op_num_threads);
  SessionOptions& SetInterOpNumThreads(int inter_op_num_threads);
  SessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level);

  SessionOptions& EnableCpuMemArena();
  SessionOptions& DisableCpuMemArena();

  SessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_file);

  SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);
  SessionOptions& DisableProfiling();

  SessionOptions& EnableOrtCustomOps();

  SessionOptions& EnableMemPattern();
  SessionOptions& DisableMemPattern();

  SessionOptions& SetExecutionMode(ExecutionMode execution_mode);

  SessionOptions& SetLogId(const char* logid);
  SessionOptions& SetLogSeverityLevel(int level);

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);

  SessionOptions& DisablePerSessionThreads();

  SessionOptions& AddConfigEntry(const char* config_key, const char* config_value);
  SessionOptions& AddInitializer(const char* name, const OrtValue* ort_val);

  SessionOptions& AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options);
  SessionOptions& AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options);
  SessionOptions& AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options);
  SessionOptions& AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options);
};

struct ModelMetadata : Base<OrtModelMetadata> {
  explicit ModelMetadata(std::nullptr_t) {}
  explicit ModelMetadata(OrtModelMetadata* p) : Base<OrtModelMetadata>{p} {}

  char* GetProducerName(OrtAllocator* allocator) const;
  char* GetGraphName(OrtAllocator* allocator) const;
  char* GetDomain(OrtAllocator* allocator) const;
  char* GetDescription(OrtAllocator* allocator) const;
  char* GetGraphDescription(OrtAllocator* allocator) const;
  char** GetCustomMetadataMapKeys(OrtAllocator* allocator, _Out_ int64_t& num_keys) const;
  char* LookupCustomMetadataMap(const char* key, OrtAllocator* allocator) const;
  int64_t GetVersion() const;
};

struct Session : Base<OrtSession> {
  explicit Session(std::nullptr_t) {}
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options);
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options, OrtPrepackedWeightsContainer* prepacked_weights_container);
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options);

  // Run that will allocate the output values
  std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
                         const char* const* output_names, size_t output_count);
  // Run for when there is a list of preallocated outputs
  void Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
           const char* const* output_names, Value* output_values, size_t output_count);

  void Run(const RunOptions& run_options, const struct IoBinding&);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  size_t GetOverridableInitializerCount() const;

  char* GetInputName(size_t index, OrtAllocator* allocator) const;
  char* GetOutputName(size_t index, OrtAllocator* allocator) const;
  char* GetOverridableInitializerName(size_t index, OrtAllocator* allocator) const;
  char* EndProfiling(OrtAllocator* allocator) const;
  uint64_t GetProfilingStartTimeNs() const;
  ModelMetadata GetModelMetadata() const;

  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
  TypeInfo GetOverridableInitializerTypeInfo(size_t index) const;
};

struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  explicit TensorTypeAndShapeInfo(std::nullptr_t) {}
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}

  ONNXTensorElementDataType GetElementType() const;
  size_t GetElementCount() const;

  size_t GetDimensionsCount() const;
  void GetDimensions(int64_t* values, size_t values_count) const;
  void GetSymbolicDimensions(const char** values, size_t values_count) const;

  std::vector<int64_t> GetShape() const;
};

struct SequenceTypeInfo : Base<OrtSequenceTypeInfo> {
  explicit SequenceTypeInfo(std::nullptr_t) {}
  explicit SequenceTypeInfo(OrtSequenceTypeInfo* p) : Base<OrtSequenceTypeInfo>{p} {}

  TypeInfo GetSequenceElementType() const;
};

struct MapTypeInfo : Base<OrtMapTypeInfo> {
  explicit MapTypeInfo(std::nullptr_t) {}
  explicit MapTypeInfo(OrtMapTypeInfo* p) : Base<OrtMapTypeInfo>{p} {}

  ONNXTensorElementDataType GetMapKeyType() const;
  TypeInfo GetMapValueType() const;
};

struct TypeInfo : Base<OrtTypeInfo> {
  explicit TypeInfo(std::nullptr_t) {}
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;
  Unowned<SequenceTypeInfo> GetSequenceTypeInfo() const;
  Unowned<MapTypeInfo> GetMapTypeInfo() const;

  ONNXType GetONNXType() const;
};

struct Value : Base<OrtValue> {
  template <typename T>
  static Value CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);
  template <typename T>
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type);

  static Value CreateMap(Value& keys, Value& values);
  static Value CreateSequence(std::vector<Value>& values);

  template <typename T>
  static Value CreateOpaque(const char* domain, const char* type_name, const T&);

  template <typename T>
  void GetOpaqueData(const char* domain, const char* type_name, T&) const;

  explicit Value(std::nullptr_t) {}
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

  bool IsTensor() const;
  bool HasElement() const;
  size_t GetCount() const;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
  Value GetValue(int index, OrtAllocator* allocator) const;

  size_t GetStringTensorDataLength() const;
  void GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const;

  template <typename T>
  T* GetTensorMutableData();

  template <typename T>
  const T* GetTensorData() const;

  template <typename T>
  T& At(const std::vector<int64_t>& location);

  TypeInfo GetTypeInfo() const;
  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;

  size_t GetStringTensorElementLength(size_t element_index) const;
  void GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const;

  void FillStringTensor(const char* const* s, size_t s_len);
  void FillStringTensorElement(const char* s, size_t index);
};

// Represents native memory allocation
struct MemoryAllocation {
  MemoryAllocation(OrtAllocator* allocator, void* p, size_t size);
  ~MemoryAllocation();
  MemoryAllocation(const MemoryAllocation&) = delete;
  MemoryAllocation& operator=(const MemoryAllocation&) = delete;
  MemoryAllocation(MemoryAllocation&&) noexcept;
  MemoryAllocation& operator=(MemoryAllocation&&) noexcept;

  void* get() { return p_; }
  size_t size() const { return size_; }

 private:
  OrtAllocator* allocator_;
  void* p_;
  size_t size_;
};

struct AllocatorWithDefaultOptions {
  AllocatorWithDefaultOptions();

  operator OrtAllocator*() { return p_; }
  operator const OrtAllocator*() const { return p_; }

  void* Alloc(size_t size);
  // The return value will own the allocation
  MemoryAllocation GetAllocation(size_t size);
  void Free(void* p);

  const OrtMemoryInfo* GetInfo() const;

 private:
  OrtAllocator* p_{};
};

template <typename B>
struct BaseMemoryInfo : B {
  BaseMemoryInfo() = default;
  explicit BaseMemoryInfo(typename B::contained_type* p) : B(p) {}
  ~BaseMemoryInfo() = default;
  BaseMemoryInfo(BaseMemoryInfo&&) = default;
  BaseMemoryInfo& operator=(BaseMemoryInfo&&) = default;

  std::string GetAllocatorName() const;
  OrtAllocatorType GetAllocatorType() const;
  int GetDeviceId() const;
  OrtMemType GetMemoryType() const;
  template <typename U>
  bool operator==(const BaseMemoryInfo<U>& o) const;
};

struct UnownedMemoryInfo : BaseMemoryInfo<Base<const OrtMemoryInfo> > {
  explicit UnownedMemoryInfo(std::nullptr_t) {}
  explicit UnownedMemoryInfo(const OrtMemoryInfo* p) : BaseMemoryInfo(p) {}
};

struct MemoryInfo : BaseMemoryInfo<Base<OrtMemoryInfo> > {
  static MemoryInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);

  explicit MemoryInfo(std::nullptr_t) {}
  explicit MemoryInfo(OrtMemoryInfo* p) : BaseMemoryInfo(p) {}
  MemoryInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);
};

struct Allocator : public Base<OrtAllocator> {
  Allocator(const Session& session, const MemoryInfo&);

  void* Alloc(size_t size) const;
  // The return value will own the allocation
  MemoryAllocation GetAllocation(size_t size);
  void Free(void* p) const;
  UnownedMemoryInfo GetInfo() const;
};

struct IoBinding : public Base<OrtIoBinding> {
 private:
  std::vector<std::string> GetOutputNamesHelper(OrtAllocator*) const;
  std::vector<Value> GetOutputValuesHelper(OrtAllocator*) const;

 public:
  explicit IoBinding(Session& session);
  void BindInput(const char* name, const Value&);
  void BindOutput(const char* name, const Value&);
  void BindOutput(const char* name, const MemoryInfo&);
  std::vector<std::string> GetOutputNames() const;
  std::vector<std::string> GetOutputNames(Allocator&) const;
  std::vector<Value> GetOutputValues() const;
  std::vector<Value> GetOutputValues(Allocator&) const;
  void ClearBoundInputs();
  void ClearBoundOutputs();
};

/*! \struct Ort::ArenaCfg
  * \brief it is a structure that represents the configuration of an arena based allocator
  * \details Please see docs/C_API.md for details
  */
struct ArenaCfg : Base<OrtArenaCfg> {
  explicit ArenaCfg(std::nullptr_t) {}
  /**
  * \param max_mem - use 0 to allow ORT to choose the default
  * \param arena_extend_strategy -  use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  * \param initial_chunk_size_bytes - use -1 to allow ORT to choose the default
  * \param max_dead_bytes_per_chunk - use -1 to allow ORT to choose the default
  * See docs/C_API.md for details on what the following parameters mean and how to choose these values
  */
  ArenaCfg(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk);
};

//
// Custom OPs (only needed to implement custom OPs)
//

struct CustomOpApi {
  CustomOpApi(const OrtApi& api) : api_(api) {}

  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>, float, int64_t, and string
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value);
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info);
  size_t GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value);
  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value);

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info);
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input);
  size_t KernelContext_GetInputCount(const OrtKernelContext* context);
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index);
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context);
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count);

  void ThrowOnError(OrtStatus* result);

 private:
  const OrtApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) { return static_cast<const TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetName(); };

    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetExecutionProviderType(); };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputCharacteristic(index); };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputCharacteristic(index); };
  }

  // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
  const char* GetExecutionProviderType() const { return nullptr; }

  // Default implementations of GetInputCharacteristic() and GetOutputCharacteristic() below
  // (inputs and outputs are required by default)
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
};

}  // namespace Ort

#include "onnxruntime_cxx_inline.h"
