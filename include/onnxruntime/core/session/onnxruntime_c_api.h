// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/** @file onnxruntime_c_api.h

  @brief ONNX Runtime C API.
*/

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// This value is used in structures passed to ORT so that a newer version of ORT will still work with them
#define ORT_API_VERSION 8

#ifdef __cplusplus
extern "C" {
#endif

// SAL2 Definitions
#ifndef _WIN32
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)
#define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#define ORT_ALL_ARGS_NONNULL
#endif

#ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
#ifdef ORT_DLL_IMPORT
#define ORT_EXPORT __declspec(dllimport)
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL _stdcall
#define ORT_MUST_USE_RESULT
#define ORTCHAR_T wchar_t
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define ORT_EXPORT __attribute__((visibility("default")))
#else
#define ORT_EXPORT
#endif
#define ORT_API_CALL
#define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
#define ORTCHAR_T char
#endif

#ifndef ORT_TSTR
#ifdef _WIN32
#define ORT_TSTR(X) L##X
#else
#define ORT_TSTR(X) X
#endif
#endif

// Any pointer marked with _In_ or _Out_, cannot be NULL.

// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.

#ifdef __cplusplus
// For any compiler with C++11 support, MSVC 2015 and greater, or Clang version supporting noexcept.
// Such complex condition is needed because compilers set __cplusplus value differently.
#ifndef __has_feature
#define __has_feature(x) 0
#endif
#if ((__cplusplus >= 201103L) || (_MSC_VER >= 1900) || (defined(__has_feature) && __has_feature(cxx_noexcept)))
#define NO_EXCEPTION noexcept
#else
#define NO_EXCEPTION throw()
#endif
#else
#define NO_EXCEPTION
#endif

// Copied from TensorProto::DataType
// Currently, Ort doesn't support complex64, complex128
typedef enum ONNXTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   // maps to c type float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,   // maps to c type uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,    // maps to c type int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,  // maps to c type uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,   // maps to c type int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,   // maps to c type int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // maps to c type int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,  // maps to c++ type std::string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,      // maps to c type double
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,      // maps to c type uint32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,      // maps to c type uint64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16     // Non-IEEE floating-point format based on IEEE754 single-precision
} ONNXTensorElementDataType;


// Synced with onnx TypeProto oneof
typedef enum ONNXType {
  ONNX_TYPE_UNKNOWN,
  ONNX_TYPE_TENSOR,
  ONNX_TYPE_SEQUENCE,
  ONNX_TYPE_MAP,
  ONNX_TYPE_OPAQUE,
  ONNX_TYPE_SPARSETENSOR,
} ONNXType;

// These types are synced with internal
// SparseFormatFlags but are not exposed
// as flags
typedef enum OrtSparseFormat {
  ORT_SPARSE_UNDEFINED = 0,
  ORT_SPARSE_COO = 0x1,
  ORT_SPARSE_CSRC = 0x2,
  ORT_SPARSE_BLOCK_SPARSE = 0x4
} OrtSparseFormat;


typedef enum OrtLoggingLevel {
  ORT_LOGGING_LEVEL_VERBOSE,
  ORT_LOGGING_LEVEL_INFO,
  ORT_LOGGING_LEVEL_WARNING,
  ORT_LOGGING_LEVEL_ERROR,
  ORT_LOGGING_LEVEL_FATAL,
} OrtLoggingLevel;

typedef enum OrtErrorCode {
  ORT_OK,
  ORT_FAIL,
  ORT_INVALID_ARGUMENT,
  ORT_NO_SUCHFILE,
  ORT_NO_MODEL,
  ORT_ENGINE_ERROR,
  ORT_RUNTIME_EXCEPTION,
  ORT_INVALID_PROTOBUF,
  ORT_MODEL_LOADED,
  ORT_NOT_IMPLEMENTED,
  ORT_INVALID_GRAPH,
  ORT_EP_FAIL,
} OrtErrorCode;

#define ORT_RUNTIME_CLASS(X) \
  struct Ort##X;             \
  typedef struct Ort##X Ort##X;

// The actual types defined have an Ort prefix
ORT_RUNTIME_CLASS(Env);
ORT_RUNTIME_CLASS(Status);  // nullptr for Status* indicates success
ORT_RUNTIME_CLASS(MemoryInfo);
ORT_RUNTIME_CLASS(IoBinding);
ORT_RUNTIME_CLASS(Session);  //Don't call ReleaseSession from Dllmain (because session owns a thread pool)
ORT_RUNTIME_CLASS(Value);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(SessionOptions);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(ModelMetadata);
ORT_RUNTIME_CLASS(ThreadPoolParams);
ORT_RUNTIME_CLASS(ThreadingOptions);
ORT_RUNTIME_CLASS(ArenaCfg);
ORT_RUNTIME_CLASS(PrepackedWeightsContainer);
ORT_RUNTIME_CLASS(TensorRTProviderOptionsV2);

#ifdef _WIN32
typedef _Return_type_success_(return == 0) OrtStatus* OrtStatusPtr;
#else
typedef OrtStatus* OrtStatusPtr;
#endif

// __VA_ARGS__ on Windows and Linux are different
#define ORT_API(RETURN_TYPE, NAME, ...) RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_API_STATUS(NAME, ...) \
  _Success_(return == 0) _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// XXX: Unfortunately, SAL annotations are known to not work with function pointers
#define ORT_API2_STATUS(NAME, ...) \
  _Check_return_ _Ret_maybenull_ OrtStatusPtr(ORT_API_CALL* NAME)(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT and ORT_EXPORT
#define ORT_API_STATUS_IMPL(NAME, ...) \
  _Success_(return == 0) _Check_return_ _Ret_maybenull_ OrtStatusPtr ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

#define ORT_CLASS_RELEASE(X) void(ORT_API_CALL * Release##X)(_Frees_ptr_opt_ Ort##X * input)
#define ORT_CLASS_RELEASE2(X) void(ORT_API_CALL * Release##X)(_Frees_ptr_opt_ Ort##X##V2 * input)

// When passing in an allocator to any ORT function, be sure that the allocator object
// is not destroyed until the last allocated object using it is freed.
typedef struct OrtAllocator {
  uint32_t version;  // Initialize to ORT_API_VERSION
  void*(ORT_API_CALL* Alloc)(struct OrtAllocator* this_, size_t size);
  void(ORT_API_CALL* Free)(struct OrtAllocator* this_, void* p);
  const struct OrtMemoryInfo*(ORT_API_CALL* Info)(const struct OrtAllocator* this_);
} OrtAllocator;

typedef void(ORT_API_CALL* OrtLoggingFunction)(
    void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message);

// Graph optimization level.
// Refer to https://www.onnxruntime.ai/docs/resources/graph-optimizations.html
// for an in-depth understanding of Graph Optimizations in ORT
typedef enum GraphOptimizationLevel {
  ORT_DISABLE_ALL = 0,
  ORT_ENABLE_BASIC = 1,
  ORT_ENABLE_EXTENDED = 2,
  ORT_ENABLE_ALL = 99
} GraphOptimizationLevel;

typedef enum ExecutionMode {
  ORT_SEQUENTIAL = 0,
  ORT_PARALLEL = 1,
} ExecutionMode;

// Set the language projection, default is C, which means it will classify the language not in the list to C also.
typedef enum OrtLanguageProjection {
  ORT_PROJECTION_C = 0,  // default
  ORT_PROJECTION_CPLUSPLUS = 1,
  ORT_PROJECTION_CSHARP = 2,
  ORT_PROJECTION_PYTHON = 3,
  ORT_PROJECTION_JAVA = 4,
  ORT_PROJECTION_WINML = 5,
  ORT_PROJECTION_NODEJS = 6,
} OrtLanguageProjection;

struct OrtKernelInfo;
typedef struct OrtKernelInfo OrtKernelInfo;
struct OrtKernelContext;
typedef struct OrtKernelContext OrtKernelContext;
struct OrtCustomOp;
typedef struct OrtCustomOp OrtCustomOp;

typedef enum OrtAllocatorType {
  Invalid = -1,
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1
} OrtAllocatorType;

/**
 * memory types for allocator, exec provider specific types should be extended in each provider
 * Whenever this struct is updated, please also update the MakeKey function in onnxruntime/core/framework/execution_provider.cc
*/
typedef enum OrtMemType {
  OrtMemTypeCPUInput = -2,              // Any CPU memory used by non-CPU execution provider
  OrtMemTypeCPUOutput = -1,             // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeCPU = OrtMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeDefault = 0,                // the default allocator for execution provider
} OrtMemType;

typedef enum OrtCudnnConvAlgoSearch {
  EXHAUSTIVE,  // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
  HEURISTIC,   // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
  DEFAULT,     // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
} OrtCudnnConvAlgoSearch;

/// <summary>
/// Options for the CUDA provider that are passed to SessionOptionsAppendExecutionProvider_CUDA
/// </summary>
typedef struct OrtCUDAProviderOptions {
  int device_id;                                  // cuda device with id=0 as default device.
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search;  // cudnn conv algo search option

  size_t gpu_mem_limit;  // default cuda memory limitation to maximum finite value of size_t.
                         // (will be overridden by "max_mem" value used while creating `arena_cfg` if `arena_cfg` is provided)

  int arena_extend_strategy;  // default area extend strategy to KNextPowerOfTwo.
                              // (will be overridden by "arena_extend_strategy" value used while creating `arena_cfg` if `arena_cfg` is provided)

  int do_copy_in_default_stream;
  int has_user_compute_stream;
  void* user_compute_stream;
  OrtArenaCfg* default_memory_arena_cfg;
} OrtCUDAProviderOptions;

/// <summary>
/// Options for the ROCM provider that are passed to SessionOptionsAppendExecutionProvider_ROCM
/// </summary>
typedef struct OrtROCMProviderOptions {
  int device_id;                      // hip device with id=0 as default device.
  int miopen_conv_exhaustive_search;  // miopen conv algo exhaustive search option
  size_t gpu_mem_limit;               // default hip memory limitation to maximum finite value of size_t.
  int arena_extend_strategy;          // default area extend strategy to KNextPowerOfTwo.
} OrtROCMProviderOptions;

/// <summary>
/// Options for the TensorRT provider that are passed to SessionOptionsAppendExecutionProvider_TensorRT
/// </summary>
typedef struct OrtTensorRTProviderOptions {
  int device_id;                                // cuda device id.
  int has_user_compute_stream;                  // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                    // user specified CUDA compute stream.
  int trt_max_partition_iterations;             // maximum iterations for TensorRT parser to get capability
  int trt_min_subgraph_size;                    // minimum size of TensorRT subgraphs
  size_t trt_max_workspace_size;                // maximum workspace size for TensorRT.
  int trt_fp16_enable;                          // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int trt_int8_enable;                          // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* trt_int8_calibration_table_name;  // TensorRT INT8 calibration table name.
  int trt_int8_use_native_calibration_table;    // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int trt_dla_enable;                           // enable DLA. Default 0 = false, nonzero = true
  int trt_dla_core;                             // DLA core number. Default 0
  int trt_dump_subgraphs;                       // dump TRT subgraph. Default 0 = false, nonzero = true
  int trt_engine_cache_enable;                  // enable engine caching. Default 0 = false, nonzero = true
  const char* trt_engine_cache_path;            // specify engine cache path
  int trt_engine_decryption_enable;             // enable engine decryption. Default 0 = false, nonzero = true
  const char* trt_engine_decryption_lib_path;   // specify engine decryption library path
  int trt_force_sequential_engine_build;        // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
} OrtTensorRTProviderOptions;

/// <summary>
/// Options for the OpenVINO provider that are passed to SessionOptionsAppendExecutionProvider_OpenVINO
/// </summary>
typedef struct OrtOpenVINOProviderOptions {
#ifdef __cplusplus
  OrtOpenVINOProviderOptions() : device_type{}, enable_vpu_fast_compile{}, device_id{}, num_of_threads{}, use_compiled_network{}, blob_dump_path{} {}
#endif
  const char* device_type;                // CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16 or VAD-F_FP32
  unsigned char enable_vpu_fast_compile;  // 0 = false, nonzero = true
  const char* device_id;
  size_t num_of_threads;               // 0 uses default number of threads
  unsigned char use_compiled_network;  // 0 = false, nonzero = true
  const char* blob_dump_path;          // path is set to empty by default
} OrtOpenVINOProviderOptions;

struct OrtApi;
typedef struct OrtApi OrtApi;

struct OrtApiBase {
  const OrtApi*(ORT_API_CALL* GetApi)(uint32_t version)NO_EXCEPTION;  // Pass in ORT_API_VERSION
  // nullptr will be returned if the version is unsupported, for example when using a runtime older than this header file

  const char*(ORT_API_CALL* GetVersionString)(void)NO_EXCEPTION;
};
typedef struct OrtApiBase OrtApiBase;

ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION;

struct OrtApi {
  /**
* \param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
*/
  OrtStatus*(ORT_API_CALL* CreateStatus)(OrtErrorCode code, _In_ const char* msg)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  OrtErrorCode(ORT_API_CALL* GetErrorCode)(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
 * \param status must not be NULL
 * \return The error message inside the `status`. Do not free the returned value.
 */
  const char*(ORT_API_CALL* GetErrorMessage)(_In_ const OrtStatus* status)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
     * \param out Should be freed by `ReleaseEnv` after use
     */
  ORT_API2_STATUS(CreateEnv, OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  /**
   * \param out Should be freed by `ReleaseEnv` after use
   */
  ORT_API2_STATUS(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                  OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

  // Platform telemetry events are on by default since they are lightweight.  You can manually turn them off.
  ORT_API2_STATUS(EnableTelemetryEvents, _In_ const OrtEnv* env);
  ORT_API2_STATUS(DisableTelemetryEvents, _In_ const OrtEnv* env);

  // TODO: document the path separator convention? '/' vs '\'
  // TODO: should specify the access characteristics of model_path. Is this read only during the
  // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
  // and continue to access throughout the OrtSession lifetime?
  //  What sort of access is needed to model_path : read or read/write?
  ORT_API2_STATUS(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                  _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

  ORT_API2_STATUS(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(input_len) const char* const* input_names,
                  _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                  _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                  _Inout_updates_all_(output_names_len) OrtValue** output);

  /**
    * \return A pointer of the newly created object. The pointer should be freed by ReleaseSessionOptions after use
    */
  ORT_API2_STATUS(CreateSessionOptions, _Outptr_ OrtSessionOptions** options);

  // Set filepath to save optimized model after graph level transformations.
  ORT_API2_STATUS(SetOptimizedModelFilePath, _Inout_ OrtSessionOptions* options,
                  _In_ const ORTCHAR_T* optimized_model_filepath);

  // create a copy of an existing OrtSessionOptions
  ORT_API2_STATUS(CloneSessionOptions, _In_ const OrtSessionOptions* in_options,
                  _Outptr_ OrtSessionOptions** out_options);

  // Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
  // has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
  // See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
  ORT_API2_STATUS(SetSessionExecutionMode, _Inout_ OrtSessionOptions* options, ExecutionMode execution_mode);

  // Enable profiling for this session.
  ORT_API2_STATUS(EnableProfiling, _Inout_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
  ORT_API2_STATUS(DisableProfiling, _Inout_ OrtSessionOptions* options);

  // Enable the memory pattern optimization.
  // The idea is if the input shapes are the same, we could trace the internal memory allocation
  // and generate a memory pattern for future request. So next time we could just do one allocation
  // with a big chunk for all the internal memory allocation.
  // Note: memory pattern optimization is only available when SequentialExecution enabled.
  ORT_API2_STATUS(EnableMemPattern, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableMemPattern, _Inout_ OrtSessionOptions* options);

  // Enable the memory arena on CPU
  // Arena may pre-allocate memory for future usage.
  // set this option to false if you don't want it.
  ORT_API2_STATUS(EnableCpuMemArena, _Inout_ OrtSessionOptions* options);
  ORT_API2_STATUS(DisableCpuMemArena, _Inout_ OrtSessionOptions* options);

  // < logger id to use for session output
  ORT_API2_STATUS(SetSessionLogId, _Inout_ OrtSessionOptions* options, const char* logid);

  // < applies to session load, initialization, etc
  ORT_API2_STATUS(SetSessionLogVerbosityLevel, _Inout_ OrtSessionOptions* options, int session_log_verbosity_level);
  ORT_API2_STATUS(SetSessionLogSeverityLevel, _Inout_ OrtSessionOptions* options, int session_log_severity_level);

  ORT_API2_STATUS(SetSessionGraphOptimizationLevel, _Inout_ OrtSessionOptions* options,
                  GraphOptimizationLevel graph_optimization_level);

  // Sets the number of threads used to parallelize the execution within nodes
  // A value of 0 means ORT will pick a default
  // Note: If you've built ORT with OpenMP, this API has no effect on the number of threads used. In this case
  // use the OpenMP env variables to configure the number of intra op num threads.
  ORT_API2_STATUS(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);

  // Sets the number of threads used to parallelize the execution of the graph (across nodes)
  // If sequential execution is enabled this value is ignored
  // A value of 0 means ORT will pick a default
  ORT_API2_STATUS(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

  /*
  Create a custom op domain. After all sessions using it are released, call ReleaseCustomOpDomain
  */
  ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

  /*
     * Add custom ops to the OrtCustomOpDomain
     *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
    */
  ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);

  /*
     * Add a custom op domain to the OrtSessionOptions
     *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
    */
  ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);

  /*
     * Loads a DLL named 'library_path' and looks for this entry point:
     *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
     * It then passes in the provided session options to this function along with the api base.
     * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
     * session options are destroyed, or if an error occurs and it is non null.
  */
  ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path,
                  void** library_handle);

  /**
    * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
    * functions to enable them in the session:
    *   OrtSessionOptionsAppendExecutionProvider_CPU
    *   OrtSessionOptionsAppendExecutionProvider_CUDA
    *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
    * The order they are called indicates the preference order as well. In other words call this method
    * on your most preferred execution provider first followed by the less preferred ones.
    * If none are called Ort will use its internal CPU execution provider.
    */

  ORT_API2_STATUS(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
  ORT_API2_STATUS(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);

  /**
   * \param out  should be freed by ReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param out  should be freed by ReleaseTypeInfo after use
   */
  ORT_API2_STATUS(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
 * \param out  should be freed by ReleaseTypeInfo after use
 */
  ORT_API2_STATUS(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index,
                  _Outptr_ OrtTypeInfo** type_info);

  /**
   * \param value is set to a null terminated UTF-8 encoded string allocated using 'allocator'.
   *              The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

  /**
   * \return A pointer to the newly created object. The pointer should be freed by ReleaseRunOptions after use
   */
  ORT_API2_STATUS(CreateRunOptions, _Outptr_ OrtRunOptions** out);

  ORT_API2_STATUS(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
  ORT_API2_STATUS(RunOptionsSetRunTag, _Inout_ OrtRunOptions*, _In_ const char* run_tag);

  ORT_API2_STATUS(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
  ORT_API2_STATUS(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

  // Set a flag so that ALL incomplete OrtRun calls that are using this instance of OrtRunOptions
  // will exit as soon as possible.
  ORT_API2_STATUS(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
  // Unset the terminate flag to enable this OrtRunOptions instance being used in new OrtRun calls.
  ORT_API2_STATUS(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

  /**
   * Create a tensor from an allocator. ReleaseValue will also release the buffer inside the output value
   * \param out Should be freed by calling ReleaseValue
   * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
   */
  ORT_API2_STATUS(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* shape, size_t shape_len,
                  ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

  /**
   * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
   * p_data is owned by caller. ReleaseValue won't release p_data.
   * \param out Should be freed by calling ReleaseValue
   */
  ORT_API2_STATUS(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                  size_t p_data_len, _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                  _Outptr_ OrtValue** out);

  /**
   * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
   */
  ORT_API2_STATUS(IsTensor, _In_ const OrtValue* value, _Out_ int* out);

  // This function doesn't work with string tensor
  // this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
  ORT_API2_STATUS(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s each A string array. Each string in this array must be null terminated.
     * \param s_len length of s
     */
  ORT_API2_STATUS(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);

  /**
     * \param value A tensor created from OrtCreateTensor... function.
     * \param len total data length, not including the trailing '\0' chars.
     */
  ORT_API2_STATUS(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);

  /**
     * \param s string contents. Each string is NOT null-terminated.
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s_len total data length, get it from OrtGetStringTensorDataLength
     */
  ORT_API2_STATUS(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                  size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);

  /**
     * Don't free the 'out' value
     */
  ORT_API2_STATUS(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*,
                  _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);

  /**
     * Return OnnxType from OrtTypeInfo
     */
  ORT_API2_STATUS(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);

  /**
     * The 'out' value should be released by calling ReleaseTensorTypeAndShapeInfo
     */
  ORT_API2_STATUS(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  ORT_API2_STATUS(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);

  /**
 * \param info Created from CreateTensorTypeAndShapeInfo() function
 * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
 * \param dim_count length of dim_values
 */
  ORT_API2_STATUS(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  ORT_API2_STATUS(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*,
                  _Out_ enum ONNXTensorElementDataType* out);
  ORT_API2_STATUS(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
  ORT_API2_STATUS(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values,
                  size_t dim_values_length);
  ORT_API2_STATUS(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                  _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);

  /**
 * Return the number of elements specified by the tensor shape.
 * Return a negative value if unknown (i.e., any dimension is negative.)
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
  ORT_API2_STATUS(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);

  /**
 * \param out Should be freed by ReleaseTensorTypeAndShapeInfo after use
 */
  ORT_API2_STATUS(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

  /**
 * Get the type information of an OrtValue
 * \param value
 * \param out The returned value should be freed by ReleaseTypeInfo after use
 */
  ORT_API2_STATUS(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);

  ORT_API2_STATUS(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);

  ORT_API2_STATUS(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                  enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);

  /**
 * Convenience function for special case of CreateMemoryInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
 */
  ORT_API2_STATUS(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1,
                  _Outptr_ OrtMemoryInfo** out);

  /**
 * Test if two memory info are equal
 * \Sets 'out' to 0 if equal, -1 if not equal
 */
  ORT_API2_STATUS(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out);

  /**
 * Do not free the returned value
 */
  ORT_API2_STATUS(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
  ORT_API2_STATUS(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

  ORT_API2_STATUS(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
  ORT_API2_STATUS(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
  ORT_API2_STATUS(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out);

  // The returned pointer doesn't have to be freed.
  // Always returns the same instance on every invocation.
  // Please note that this is a non-arena based allocator.
  ORT_API2_STATUS(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);

  // Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
  // optimizations that can take advantage of fixed values (such as memory planning, etc)
  ORT_API2_STATUS(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* dim_denotation,
                  _In_ int64_t dim_value);

  /**
   * APIs to support non-tensor types - map and sequence.
   * Currently only the following types are supported
   * Note: the following types should be kept in sync with data_types.h
   * Map types
   * =========
   * std::map<std::string, std::string>
   * std::map<std::string, int64_t>
   * std::map<std::string, float>
   * std::map<std::string, double>
   * std::map<int64_t, std::string>
   * std::map<int64_t, int64_t>
   * std::map<int64_t, float>
   * std::map<int64_t, double>
   *
   * Sequence types
   * ==============
   * std::vector<std::string>
   * std::vector<int64_t>
   * std::vector<float>
   * std::vector<double>
   * std::vector<std::map<std::string, float>>
   * std::vector<std::map<int64_t, float>
   */

  /**
   * If input OrtValue represents a map, you need to retrieve the keys and values
   * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
   * If input OrtValue represents a sequence, use index to retrieve the index'th element
   * of the sequence.
   */
  ORT_API2_STATUS(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator,
                  _Outptr_ OrtValue** out);

  /**
   * Returns 2 for type map and N for sequence where N is the number of elements
   * in the sequence.
   */
  ORT_API2_STATUS(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);

  /**
   * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
   * representing keys and values.
   * To construct a sequence, use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an arrary of N OrtValues.
   * \value_type should be either map or sequence.
   */
  ORT_API2_STATUS(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                  enum ONNXType value_type, _Outptr_ OrtValue** out);

  /**
     * Construct OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization. OrtValue in this case would contain
     * an internal representation of the Opaque type. Opaque types are distinguished between
     * each other by two strings 1) domain and 2) type name. The combination of the two
     * must be unique, so the type representation is properly identified internally. The combination
     * must be properly registered from within ORT at both compile/run time or by another API.
     *
     * To construct the OrtValue pass domain and type names, also a pointer to a data container
     * the type of which must be know to both ORT and the client program. That data container may or may
     * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
     * verification purposes.
     *
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */
  ORT_API2_STATUS(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                  _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);

  /**
     * Fetch data from an OrtValue that contains a value of non-standard type created for
     * experiments or while awaiting standardization.
     * \domain_name - domain name for the Opaque type, null terminated.
     * \type_name   - type name for the Opaque type, null terminated.
     * \data_contianer - data to populate OrtValue
     * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
     *                    data_container size internally.
     */

  ORT_API2_STATUS(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name, _In_ const OrtValue* in,
                  _Out_ void* data_container, size_t data_container_size);

  /**
     * Fetch a float stored as an attribute in the graph node
     * \info - OrtKernelInfo instance
     * \name - name of the attribute to be parsed
     * \out - pointer to memory where the attribute is to be stored
     */
  ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ float* out);

  /**
     * Fetch a 64-bit int stored as an attribute in the graph node
     * \info - OrtKernelInfo instance
     * \name - name of the attribute to be parsed
     * \out - pointer to memory where the attribute is to be stored
     */
  ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ int64_t* out);
  /**
     * Fetch a string stored as an attribute in the graph node
     * \info - OrtKernelInfo instance
     * \name - name of the attribute to be parsed
     * \out - pointer to memory where the attribute's contents are to be stored
     * \size - actual size of string attribute
     * (If `out` is nullptr, the value of `size` is set to the true size of the string
        attribute, and a success status is returned.

        If the `size` parameter is greater than or equal to the actual string attribute's size,
        the value of `size` is set to the true size of the string attribute, the provided memory
        is filled with the attribute's contents, and a success status is returned.

        If the `size` parameter is lesser than the actual string attribute's size and `out`
        is not nullptr, the value of `size` is set to the true size of the string attribute
        and a failure status is returned.)
     */
  ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out,
                  _Inout_ size_t* size);

  ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
  ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index,
                  _Out_ const OrtValue** out);
  ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index,
                  _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);

  ORT_CLASS_RELEASE(Env);
  ORT_CLASS_RELEASE(Status);  // nullptr for Status* indicates success
  ORT_CLASS_RELEASE(MemoryInfo);
  ORT_CLASS_RELEASE(Session);  //Don't call ReleaseSession from Dllmain (because session owns a thread pool)
  ORT_CLASS_RELEASE(Value);
  ORT_CLASS_RELEASE(RunOptions);
  ORT_CLASS_RELEASE(TypeInfo);
  ORT_CLASS_RELEASE(TensorTypeAndShapeInfo);
  ORT_CLASS_RELEASE(SessionOptions);
  ORT_CLASS_RELEASE(CustomOpDomain);

  // End of Version 1 - DO NOT MODIFY ABOVE (see above text for more information)

  // Version 2 - In development, feel free to add/remove/rearrange here

  /**
    * GetDenotationFromTypeInfo
	 * This api augments OrtTypeInfo to return denotations on the type.
	 * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
    */
  ORT_API2_STATUS(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation,
                  _Out_ size_t* len);

  // OrtTypeInfo Casting methods

  /**
    * CastTypeInfoToMapTypeInfo
	 * This api augments OrtTypeInfo to return an OrtMapTypeInfo when the type is a map.
	 * The OrtMapTypeInfo has additional information about the map's key type and value type.
	 * This is used by WinML to support model reflection APIs.
	 * This is used by WinML to support model reflection APIs.
	 *
	 * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtMapTypeInfo** out);

  /**
    * CastTypeInfoToSequenceTypeInfo
	 * This api augments OrtTypeInfo to return an OrtSequenceTypeInfo when the type is a sequence.
	 * The OrtSequenceTypeInfo has additional information about the sequence's element type.
    * This is used by WinML to support model reflection APIs.
	 *
	 * Don't free the 'out' value
    */
  ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                  _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

  // OrtMapTypeInfo Accessors

  /**
    * GetMapKeyType
	 * This api augments get the key type of a map. Key types are restricted to being scalar types and use ONNXTensorElementDataType.
	 * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);

  /**
    * GetMapValueType
	 * This api augments get the value type of a map.
    */
  ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

  // OrtSequenceTypeInfo Accessors

  /**
    * GetSequenceElementType
	 * This api augments get the element type of a sequence.
	 * This is used by WinML to support model reflection APIs.
    */
  ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info,
                  _Outptr_ OrtTypeInfo** type_info);

  ORT_CLASS_RELEASE(MapTypeInfo);
  ORT_CLASS_RELEASE(SequenceTypeInfo);

  /**
   * \param out is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * Profiling is turned ON automatically if enabled for the particular session by invoking EnableProfiling()
   * on the SessionOptions instance used to create the session.
   */
  ORT_API2_STATUS(SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator, _Outptr_ char** out);

  /**
   * \param out is a pointer to the newly created object. The pointer should be freed by calling ReleaseModelMetadata after use.
   */
  ORT_API2_STATUS(SessionGetModelMetadata, _In_ const OrtSession* sess, _Outptr_ OrtModelMetadata** out);

  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   */
  ORT_API2_STATUS(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator,
                  _Outptr_ char** value);
  ORT_API2_STATUS(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  /**
   * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
   * 'value' will be a nullptr if the given key is not found in the custom metadata map.
   */
  ORT_API2_STATUS(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

  ORT_API2_STATUS(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata, _Out_ int64_t* value);

  ORT_CLASS_RELEASE(ModelMetadata);

  /*
  * Creates an environment with global threadpools that will be shared across sessions.
  * Use this in conjunction with DisablePerSessionThreads API or else the session will use
  * its own thread pools.
  */
  ORT_API2_STATUS(CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level, _In_ const char* logid,
                  _In_ const OrtThreadingOptions* t_options, _Outptr_ OrtEnv** out);

  /*
  * Calling this API will make the session use the global threadpools shared across sessions.
  * This API should be used in conjunction with CreateEnvWithGlobalThreadPools API.
  */
  ORT_API2_STATUS(DisablePerSessionThreads, _Inout_ OrtSessionOptions* options);

  ORT_API2_STATUS(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);

  ORT_CLASS_RELEASE(ThreadingOptions);

  /**
   * \param num_keys contains the number of keys in the custom metadata map
   * \param keys is an array of null terminated strings (array count = num_keys) allocated using 'allocator'.
   * The caller is responsible for freeing each string and the pointer array.
   * 'keys' will be a nullptr if custom metadata map is empty.
   */
  ORT_API2_STATUS(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);

  // Override symbolic dimensions (by specific name strings) with actual values
  // if known at session initialization time to enable optimizations that can
  // take advantage of fixed values (such as memory planning, etc)
  ORT_API2_STATUS(AddFreeDimensionOverrideByName,
                  _Inout_ OrtSessionOptions* options, _In_ const char* dim_name,
                  _In_ int64_t dim_value);

  /**
   * \param out_ptr will hold a pointer to the array of char *
   * representing available providers.
   * \param provider_length is a pointer to an int variable where
   * the number of available providers will be added.
   * The caller is responsible for freeing each char * and the pointer
   * array by calling ReleaseAvailableProviders().
   */
  ORT_API2_STATUS(GetAvailableProviders, _Outptr_ char*** out_ptr,
                  _In_ int* provider_length);

  /**
   * \param ptr is the pointer to an array of available providers you
   * get after calling GetAvailableProviders().
   * \param providers_length is the number of available providers.
   */
  ORT_API2_STATUS(ReleaseAvailableProviders, _In_ char** ptr,
                  _In_ int providers_length);

  /**
     * \param value - A tensor created from OrtCreateTensor... function.
     * \param index - index of string tensor element, length of element at index will be returned.
     * \param out - number of UTF-8 bytes that the string contains
     */
  ORT_API2_STATUS(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);

  /**
     * \param s string element contents in UTF-8 encoding. The string is NOT null-terminated.
     * \param value A tensor created from OrtCreateTensor... function.
     * \param s_len element length, get it from OrtGetStringTensorElementLength.
     * \param index offset of element of tensor to return.
     */
  ORT_API2_STATUS(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);

  /**
     * \param value - A tensor created from OrtCreateTensor... function.
     * \param s - A null terminated UTF-8 encoded string.
     * \param index - index of string tensor element to fill
     */
  ORT_API2_STATUS(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);

  /**
     * Set a single session configuration entry as a pair of strings
     * If a configuration with same key exists, this will overwrite the configuration with the given config_value
     * \param config_key    A null terminated string representation of the config key
     * \param config_value  A null terminated string representation of the config value
     * The config_key and the format of config_value are defined in onnxruntime_session_options_config_keys.h
     */
  ORT_API2_STATUS(AddSessionConfigEntry, _Inout_ OrtSessionOptions* options,
                  _In_z_ const char* config_key, _In_z_ const char* config_value);

  /**
   * \param sess valid OrtSession instance
   * \param mem_info - valid OrtMemoryInfo instance
   * \param - out a ptr to a new instance of OrtAllocator according to the spec within mem_info
   *         if successful
   * \return OrtStatus or nullptr if successful
   */
  ORT_API2_STATUS(CreateAllocator, _In_ const OrtSession* sess, _In_ const OrtMemoryInfo* mem_info,
                  _Outptr_ OrtAllocator** out);

  // Release instance of OrtAllocator obtained from CreateAllocator API
  ORT_CLASS_RELEASE(Allocator);

  ORT_API2_STATUS(RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options, _In_ const OrtIoBinding* binding_ptr);

  // Creates an IoBinding instance that allows one to bind pre-allocated OrtValues
  // to input names. Thus if you want to use a raw on device buffer as input or output
  // you can avoid extra copy during runtime.
  ORT_API2_STATUS(CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out);

  // Release instance or OrtIoBinding obtained from CreateIoBinding API
  ORT_CLASS_RELEASE(IoBinding);

  /**
   * The function will bind the OrtValue to a specified input name.
   * The OrtValue must be a Tensor. ORT would use that value in place of input for the specified name.
   * \param binding_ptr - an instance of OrtIoBinding created by CreateIoBinding()
   * \param name - name for the model input
   * \param  val_ptr - OrtValue of Tensor type.
   * \return OrtStatus instance on error which the caller is responsible to free or nullptr on success
   */
  ORT_API2_STATUS(BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);

  /**
   * The function will bind the OrtValue to the specified output name.
   * The OrtValue must be a Tensor. ORT would use that value in place of output for the specified name.
   *
   * \param binding_ptr - an instance of OrtIoBinding created by CreateIoBinding()
   * \param name - name for the model output
   * \param  val_ptr - OrtValue of Tensor type.
   * \return OrtStatus instance on error which the caller is responsible to free or nullptr on success
   */
  ORT_API2_STATUS(BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);

  /**
   * The function will bind the OrtValue to a device which specification is contained within OrtMemoryInfo
   * You can either create an instance of OrtMemoryInfo with a device id or obtain one from the allocator that you are created/using
   * This is useful when one or more outputs have dynamic shapes and, it is hard to pre-allocated and bind a chunk of
   * memory within OrtValue ahead of time.
   *
   * \param binding_ptr - an instance of OrtIoBinding created by CreateIoBinding()
   * \param name - name for the model output
   * \param  mem_info_ptr - OrtMemoryInfo
   * \return OrtStatus instance on error which the caller is responsible to free or nullptr on success
   */
  ORT_API2_STATUS(BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* val_ptr);

  /**
    * The function returns the names of the outputs in the order they were bound. This is useful after running the model
    * with bound outputs because the returned names are in order in which output OrtValues are returned. This API is optional
    * to use. If you knew the order of outputs and its names you used for binding you would not need to use this API.
    *
    * \param  binding_ptr - a ptr to an instance of OrtIoBinding created obtained from CreateIoBinding()
    * \param  allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
    *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
    * \param buffer - pointer to a continuous buffer of non-zero terminated UTF-8 encoded strings. The number of strings stored is returned count parameter.
    *                 this buffer will be allocated with the specified allocator and must be freed after it is no longer needed.
    * \param lengths - a pointer to a continuous buffer of size_t lengths of strings returned in the buffer. The number of items is returned
    *                  in the count. This buffer is allocated with the specified allocator and must be freed after it is no longer needed.
    * \para count - is the number of strings returned. If the instance of OrtIoBiding has no bound outputs, zero is returned,
    *              no memory allocation is performed and buffer and lengths are nullptr on return.
    */
  ORT_API2_STATUS(GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                  _Out_ char** buffer, _Out_writes_all_(count) size_t** lengths, _Out_ size_t* count);

  /**
    * The function returns an array of pointers to individually allocated OrtValues that contain results of a model execution with RunWithBinding()
    * The array contains the same number of OrtValues and they are in the same order as they were bound with BindOutput()
    * or BindOutputToDevice().
    * The returned OrtValues must be individually released after they are no longer needed.
    * The array is allocated using the specified instance of the allocator and must be freed using the same allocator after
    * all the OrtValues contained therein are individually released.
    *
    * \param binding_ptr - instance of OrtIoBidning
    * \param allocator - instance of allocator to allocate output array
    * \param output - pointer to the allocated buffer. Returns nullptr if no outputs.
    * \param output_count - pointer to the number of OrtValues returned. Zero if no outputs.
    */
  ORT_API2_STATUS(GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                  _Out_writes_all_(output_count) OrtValue*** output, _Out_ size_t* output_count);

  /** Clears any previously specified bindings for inputs/outputs
   */
  void(ORT_API_CALL* ClearBoundInputs)(_Inout_ OrtIoBinding* binding_ptr) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;
  void(ORT_API_CALL* ClearBoundOutputs)(_Inout_ OrtIoBinding* binding_ptr) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

  /**
   * Provides element-level access into a tensor.
   * \param location_values a pointer to an array of index values that specify an element's location in the tensor data blob
   * \param location_values_count length of location_values
   * \param out a pointer to the element specified by location_values
   * e.g.
   * Given a tensor with overall shape [3,224,224], an element at
   * location [2,150,128] can be accessed directly.
   *
   * This function only works for numeric tensors.
   * This is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
   */
  ORT_API2_STATUS(TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count, _Outptr_ void** out);

  /**
   * Creates an allocator instance and registers it with the env to enable
   * sharing between multiple sessions that use the same env instance.
   * Lifetime of the created allocator will be valid for the duration of the environment.
   * Returns an error if an allocator with the same OrtMemoryInfo is already registered.
   * \param mem_info must be non-null.
   * \param arena_cfg if nullptr defaults will be used.
   * See docs/C_API.md for details.
  */
  ORT_API2_STATUS(CreateAndRegisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info,
                  _In_ const OrtArenaCfg* arena_cfg);

  /**
   * Set the language projection for collecting telemetry data when Env is created
   * \param projection the source projected language.
  */
  ORT_API2_STATUS(SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection);

  /**
   * On some platforms, this timer may not be as precise as nanoseconds
   * For instance, on Windows and MacOS, the precision will be ~100ns
   * \param out is set to the nanoseconds of profiling's start time
   */
  ORT_API2_STATUS(SessionGetProfilingStartTimeNs, _In_ const OrtSession* sess, _Outptr_ uint64_t* out);

  /**
   * Use this API to configure the global thread pool options to be used in the call to CreateEnvWithGlobalThreadPools.
   * A value of 0 means ORT will pick the default.
   * A value of 1 means the invoking thread will be used; no threads will be created in the thread pool.
   */
  ORT_API2_STATUS(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads);
  ORT_API2_STATUS(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads);

  /**
   * Use this API to configure the global thread pool options to be used in the call to CreateEnvWithGlobalThreadPools.
   * Allow spinning of thread pools when their queues are empty. This API will set the value for both
   * inter_op and intra_op threadpools.
   * \param allow_spinning valid values are 1 and 0.
   * 1: threadpool will spin to wait for queue to become non-empty, 0: it won't spin.
   * Prefer a value of 0 if your CPU usage is very high.
   */
  ORT_API2_STATUS(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning);

  /**
   * Add a pre-allocated initializer to a session. If a model contains an initializer with a name
   * that is same as the name passed to this API call, ORT will use this initializer instance
   * instead of deserializing one from the model file. This is useful when you want to share
   * the same initializer across sessions.
   * \param name name of the initializer
   * \param val OrtValue containing the initializer. Lifetime of 'val' and the underlying initializer buffer must be
   * managed by the user (created using the CreateTensorWithDataAsOrtValue API) and it must outlive the session object
   * to which it is added.
   */
  ORT_API2_STATUS(AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name,
                  _In_ const OrtValue* val);

  /**
   * Creates a custom environment with global threadpools and logger that will be shared across sessions.
   * Use this in conjunction with DisablePerSessionThreads API or else the session will use
   * its own thread pools.
   *
   * \param out should be freed by `ReleaseEnv` after use
   */
  ORT_API2_STATUS(CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel logging_level,
                  _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out);

  /**
   * Append CUDA execution provider to the session options
   * If CUDA is not available (due to a non cuda enabled build), this function will return failure.
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_CUDA,
                  _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);

  /**
   * Append ROCM execution provider to the session options
   * If ROCM is not available (due to a non rocm enabled build), this function will return failure.
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_ROCM,
                  _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options);

  /**
   * Append OpenVINO execution provider to the session options
   * If OpenVINO is not available (due to the OpenVINO provider shared library or its dependencies not being installed), this function will fail.
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_OpenVINO,
                  _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options);

  /**
   * Use this API to configure the global thread pool options to be used in the call to CreateEnvWithGlobalThreadPools.
   * When this API is called, flush-to-zero and denormal-as-zero are applied to threads in both intra and inter global thread pool.
   * Note that an alternative way not using this option at runtime is to train and export a model without denormals
   * and that's recommended because turning this option on may hurt model accuracy.
   */
  ORT_API2_STATUS(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* tp_options);

  /**
  * (Deprecated) Use `CreateArenaCfgV2` instead
  * Use this API to create the configuration of an arena that can eventually be used to define
  * an arena based allocator's behavior
  * \param max_mem - use 0 to allow ORT to choose the default
  * \param arena_extend_strategy -  use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  * \param initial_chunk_size_bytes - use -1 to allow ORT to choose the default
  * \param max_dead_bytes_per_chunk - use -1 to allow ORT to choose the default
  * \param out - a pointer to an OrtArenaCfg instance
  * \return a nullptr in case of success or a pointer to an OrtStatus instance in case of failure
  * See docs/C_API.md for details on what the following parameters mean and how to choose these values
  */
  ORT_API2_STATUS(CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
                  int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out);

  ORT_CLASS_RELEASE(ArenaCfg);

  /**
  * Use this API to obtain the description of the graph present in the model
  * (doc_string field of the GraphProto message within the ModelProto message).
  * If it doesn't exist, an empty string will be returned.
  * \param model_metadata - an instance of OrtModelMetadata
  * \param allocator - allocator used to allocate the string that will be returned back
  * \param value - is set to a null terminated string allocated using 'allocator'.
    The caller is responsible for freeing it.
  */
  ORT_API2_STATUS(ModelMetadataGetGraphDescription, _In_ const OrtModelMetadata* model_metadata,
                  _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
  /**
   * Append TensorRT execution provider to the session options with TensorRT provider options. 
   * If TensorRT is not available (due to a non TensorRT enabled build), this function will return failure.
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT,
                  _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);

  /**
  * Set the current device id of the GPU execution provider (cuda/tensorrt/rocm). The device id should be less
  * than the total number of devices available. Using this API makes sense only when doing multi-GPU inferencing.
  */
  ORT_API2_STATUS(SetCurrentGpuDeviceId, _In_ int device_id);

  /**
   * Get the current device id of the GPU execution provider (cuda/tensorrt/rocm).
   */
  ORT_API2_STATUS(GetCurrentGpuDeviceId, _In_ int* device_id);

  /**
     * Fetch an array of int64_t values stored as an attribute in the graph node
     * \info - OrtKernelInfo instance
     * \name - name of the attribute to be parsed
     * \out - pointer to memory where the attribute's contents are to be stored
     * \size - actual size of attribute array
     * (If `out` is nullptr, the value of `size` is set to the true size of the attribute
        array's size, and a success status is returned.

        If the `size` parameter is greater than or equal to the actual attribute array's size,
        the value of `size` is set to the true size of the attribute array's size,
        the provided memory is filled with the attribute's contents,
        and a success status is returned.

        If the `size` parameter is lesser than the actual attribute array's size and `out`
        is not nullptr, the value of `size` is set to the true size of the attribute array's size
        and a failure status is returned.)
     */
  ORT_API2_STATUS(KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ float* out, _Inout_ size_t* size);

  /**
     * Fetch an array of int64_t values stored as an attribute in the graph node
     * \info - OrtKernelInfo instance
     * \name - name of the attribute to be parsed
     * \out - pointer to memory where the attribute's contents are to be stored
     * \size - actual size of attribute array
     * (If `out` is nullptr, the value of `size` is set to the true size of the attribute
        array's size, and a success status is returned.

        If the `size` parameter is greater than or equal to the actual attribute array's size,
        the value of `size` is set to the true size of the attribute array's size,
        the provided memory is filled with the attribute's contents,
        and a success status is returned.

        If the `size` parameter is lesser than the actual attribute array's size and `out`
        is not nullptr, the value of `size` is set to the true size of the attribute array's size
        and a failure status is returned.)
     */
  ORT_API2_STATUS(KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name,
                  _Out_ int64_t* out, _Inout_ size_t* size);

  /**
  * Use this API to create the configuration of an arena that can eventually be used to define
  * an arena based allocator's behavior
  * \param arena_config_keys - keys to configure the arena
  * \param arena_config_values - values to configure the arena
  * \param num_keys - number of keys passed in
  * Supported keys are (See docs/C_API.md for details on what the following parameters mean and how to choose these values.):
  * "max_mem": Maximum memory that can be allocated by the arena based allocator.
     Use 0 for ORT to pick the best value. Default is 0.
  * "arena_extend_strategy": 0 = kNextPowerOfTwo, 1 = kSameAsRequested.
     Use -1 to allow ORT to choose the default.
  * "initial_chunk_size_bytes": (Possible) Size of the first allocation in the arena.
     Only relevant if arena strategy is `kNextPowerOfTwo`. Use -1 to allow ORT to choose the default.
     Ultimately, the first allocation size is determined by the allocation memory request.
  * "max_dead_bytes_per_chunk": Threshold of unused memory in an allocated chunk of arena memory after
     crossing which the current chunk is chunked into 2.
  * "initial_growth_chunk_size_bytes": (Possible) Size of the second allocation in the arena.
     Only relevant if arena strategy is `kNextPowerOfTwo`. Use -1 to allow ORT to choose the default.
     Ultimately, the allocation size is determined by the allocation memory request.
     Further allocation sizes are governed by the arena extend strategy.
  */
  ORT_API2_STATUS(CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys,
                  _In_reads_(num_keys) const size_t* arena_config_values, _In_ size_t num_keys,
                  _Outptr_ OrtArenaCfg** out);

  /**
     * Set a single run configuration entry as a pair of strings
     * If a configuration with same key exists, this will overwrite the configuration with the given config_value
     * \param config_key    A null terminated string representation of the config key
     * \param config_value  A null terminated string representation of the config value
     * The config_key and the format of config_value are defined in onnxruntime_run_options_config_keys.h
     */
  ORT_API2_STATUS(AddRunConfigEntry, _Inout_ OrtRunOptions* options,
                  _In_z_ const char* config_key, _In_z_ const char* config_value);

  /*
     * Creates an OrtPrepackedWeightsContainer instance.
     * This container will hold pre-packed buffers of shared initializers for sharing between sessions
     * (i.e.) if there are shared initializers that can be shared between sessions, the pre-packed buffers
     * of these (if any) may possibly be shared to provide memory footprint savings. Pass this container
     * to sessions that you would like to share pre-packed buffers of shared initializers at session
     * creation time.
     *  \out - created OrtPrepackedWeightsContainer instance
    */
  ORT_API2_STATUS(CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out);

  /*
     * Release OrtPrepackedWeightsContainer instance
     *  Note: The OrtPrepackedWeightsContainer instance must not be released until the sessions using it are released
    */
  ORT_CLASS_RELEASE(PrepackedWeightsContainer);

  /**
     * Same functionality offered by CreateSession() API except that a container that contains
     pre-packed weights' buffers is written into/read from by the created session.
     This is useful when used in conjunction with the AddInitializer() API which injects
     shared initializer info into sessions. Wherever possible, the pre-packed versions of these
     shared initializers are cached in this container so that multiple sessions can just re-use
     these instead of duplicating these in memory.
     * \env - OrtEnv instance instance
     * \model_path - model path
     * \options - OrtSessionOptions instance
     * \prepacked_weights_container - OrtPrepackedWeightsContainer instance
     * \out - created session instance
     */
  ORT_API2_STATUS(CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                  _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                  _Outptr_ OrtSession** out);

  /**
     * Same functionality offered by CreateSessionFromArray() API except that a container that contains
     pre-packed weights' buffers is written into/read from by the created session.
     This is useful when used in conjunction with the AddInitializer() API which injects
     shared initializer info into sessions. Wherever possible, the pre-packed versions of these
     shared initializers are cached in this container so that multiple sessions can just re-use
     these instead of duplicating these in memory.
     * \env - OrtEnv instance instance
     * \model_data - model byte array
     * \model_data_length - the size of the model byte array
     * \options - OrtSessionOptions instance
     * \prepacked_weights_container - OrtPrepackedWeightsContainer instance
     * \out - created session instance
     */
  ORT_API2_STATUS(CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env,
                  _In_ const void* model_data, size_t model_data_length,
                  _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                  _Outptr_ OrtSession** out);

  /**
   * Append TensorRT execution provider to the session options with TensorRT provider options.
   * If TensorRT is not available (due to a non TensorRT enabled build), this function will return failure.
   * Note: this API is slightly different than SessionOptionsAppendExecutionProvider_TensorRT.
   * SessionOptionsAppendExecutionProvider_TensorRT takes struct OrtTensorRTProviderOptions which is open to user as argument,
   * but this API takes opaque struct OrtTensorRTProviderOptionsV2 which must be created by CreateTensorRTProviderOptions.
   * User needs to instantiate OrtTensorRTProviderOptions as well as allocate/release buffers for some members of OrtTensorRTProviderOptions.
   * However, for using OrtTensorRTProviderOptionsV2, CreateTensorRTProviderOptions and ReleaseTensorRTProviderOptions will do the memory allocation and release for you. 
   *
   * \param options - OrtSessionOptions instance 
   * \param tensorrt_options - OrtTensorRTProviderOptionsV2 instance 
   */
  ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT_V2,
                  _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options);

  /**
   * Use this API to create the configuration of a TensorRT Execution Provider which is an instance of OrtTensorRTProviderOptionsV2.
   *
   * \param out - pointer to the pointer of TensorRT EP provider options instance.
   */
  ORT_API2_STATUS(CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out);

  /**
  * Use this API to set appropriate configuration knobs of a TensorRT Execution Provider.
  *
  * Please reference to https://www.onnxruntime.ai/docs/reference/execution-providers/TensorRT-ExecutionProvider.html#c-api-example
  * to know the available keys and values. Key should be in string format of the member of OrtTensorRTProviderOptions and value should be its related range.
  * For example, key="trt_max_workspace_size" and value="2147483648"
  *
  * \param tensorrt_options - OrtTensorRTProviderOptionsV2 instance
  * \param provider_options_keys - array of UTF-8 null-terminated string for provider options keys
  * \param provider_options_values - array of UTF-8 null-terminated string for provider options values
  * \param num_keys - number of keys
  */
  ORT_API2_STATUS(UpdateTensorRTProviderOptions, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                 _In_reads_(num_keys) const char* const* provider_options_keys,
                 _In_reads_(num_keys) const char* const* provider_options_values,
                 _In_ size_t num_keys);

  /**
  * Get serialized TensorRT provider options string.
  *
  * For example, "trt_max_workspace_size=2147483648;trt_max_partition_iterations=10;trt_int8_enable=1;......" 
  *
  * \param tensorrt_options - OrTensorRTProviderOptionsV2 instance 
  * \param allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
  *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
  * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
  */
  ORT_API2_STATUS(GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);

  /**
  * Use this API to release the instance of OrtTensorRTProviderV2.
  */
  ORT_CLASS_RELEASE2(TensorRTProviderOptions);

  /**
  * Enable custom operators in onnxruntime-extensions: https://github.com/microsoft/onnxruntime-extensions.git
  */
  ORT_API2_STATUS(EnableOrtCustomOps, _Inout_ OrtSessionOptions* options);
};

/*
 * Steps to use a custom op:
 *   1 Create an OrtCustomOpDomain with the domain name used by the custom ops
 *   2 Create an OrtCustomOp structure for each op and add them to the domain
 *   3 Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
*/
#define OrtCustomOpApi OrtApi

// Specifies some characteristics of inputs/outputs of custom ops:
// Specify if the inputs/outputs are one of:
// 1) Non-optional (input/output must be present in the node)
// 2) Optional (input/output may be absent in the node)
typedef enum OrtCustomOpInputOutputCharacteristic {
  // TODO: Support 'Variadic' inputs/outputs
  INPUT_OUTPUT_REQUIRED = 0,
  INPUT_OUTPUT_OPTIONAL,
} OrtCustomOpInputOutputCharacteristic;

/*
 * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
 * the implementor of the custom op.
*/
struct OrtCustomOp {
  uint32_t version;  // Initialize to ORT_API_VERSION

  // This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
  void*(ORT_API_CALL* CreateKernel)(_In_ const struct OrtCustomOp* op, _In_ const OrtApi* api,
                                    _In_ const OrtKernelInfo* info);

  // Returns the name of the op
  const char*(ORT_API_CALL* GetName)(_In_ const struct OrtCustomOp* op);

  // Returns the type of the execution provider, return nullptr to use CPU execution provider
  const char*(ORT_API_CALL* GetExecutionProviderType)(_In_ const struct OrtCustomOp* op);

  // Returns the count and types of the input & output tensors
  ONNXTensorElementDataType(ORT_API_CALL* GetInputType)(_In_ const struct OrtCustomOp* op, _In_ size_t index);
  size_t(ORT_API_CALL* GetInputTypeCount)(_In_ const struct OrtCustomOp* op);
  ONNXTensorElementDataType(ORT_API_CALL* GetOutputType)(_In_ const struct OrtCustomOp* op, _In_ size_t index);
  size_t(ORT_API_CALL* GetOutputTypeCount)(_In_ const struct OrtCustomOp* op);

  // Op kernel callbacks
  void(ORT_API_CALL* KernelCompute)(_In_ void* op_kernel, _In_ OrtKernelContext* context);
  void(ORT_API_CALL* KernelDestroy)(_In_ void* op_kernel);

  // Returns the characteristics of the input & output tensors
  OrtCustomOpInputOutputCharacteristic(ORT_API_CALL* GetInputCharacteristic)(_In_ const struct OrtCustomOp* op, _In_ size_t index);
  OrtCustomOpInputOutputCharacteristic(ORT_API_CALL* GetOutputCharacteristic)(_In_ const struct OrtCustomOp* op, _In_ size_t index);
};

#ifdef __cplusplus
}
#endif
