// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtApis {

ORT_API(const OrtApi*, GetApi, uint32_t version);
ORT_API(const char*, GetVersionString);

ORT_API(void, ReleaseEnv, OrtEnv*);
ORT_API(void, ReleaseStatus, _Frees_ptr_opt_ OrtStatus*);
ORT_API(void, ReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo*);
ORT_API(void, ReleaseSession, _Frees_ptr_opt_ OrtSession*);
ORT_API(void, ReleaseValue, _Frees_ptr_opt_ OrtValue*);
ORT_API(void, ReleaseRunOptions, _Frees_ptr_opt_ OrtRunOptions*);
ORT_API(void, ReleaseTypeInfo, _Frees_ptr_opt_ OrtTypeInfo*);
ORT_API(void, ReleaseTensorTypeAndShapeInfo, _Frees_ptr_opt_ OrtTensorTypeAndShapeInfo*);
ORT_API(void, ReleaseSessionOptions, _Frees_ptr_opt_ OrtSessionOptions*);
ORT_API(void, ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain*);
ORT_API(void, ReleaseMapTypeInfo, _Frees_ptr_opt_ OrtMapTypeInfo*);
ORT_API(void, ReleaseSequenceTypeInfo, _Frees_ptr_opt_ OrtSequenceTypeInfo*);
ORT_API(void, ReleaseModelMetadata, _Frees_ptr_opt_ OrtModelMetadata*);

_Check_return_ _Ret_notnull_ OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode code, _In_z_ const char* msg)
    NO_EXCEPTION ORT_MUST_USE_RESULT;

OrtErrorCode ORT_API_CALL GetErrorCode(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;
const char* ORT_API_CALL GetErrorMessage(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS_IMPL(CreateEnv, OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);
ORT_API_STATUS_IMPL(CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level, _In_ const char* logid,
                    _In_ const struct OrtThreadingOptions* t_options, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS_IMPL(EnableTelemetryEvents, _In_ const OrtEnv* env);
ORT_API_STATUS_IMPL(DisableTelemetryEvents, _In_ const OrtEnv* env);

ORT_API_STATUS_IMPL(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output);

ORT_API_STATUS_IMPL(CreateSessionOptions, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(CloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(SetSessionExecutionMode, _In_ OrtSessionOptions* options, ExecutionMode execution_mode);
ORT_API_STATUS_IMPL(SetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath);
ORT_API_STATUS_IMPL(EnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
ORT_API_STATUS_IMPL(DisableProfiling, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(SetSessionLogId, _In_ OrtSessionOptions* options, const char* logid);
ORT_API_STATUS_IMPL(SetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level);
ORT_API_STATUS_IMPL(SetSessionLogSeverityLevel, _In_ OrtSessionOptions* options, int session_log_severity_level);
ORT_API_STATUS_IMPL(SetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options,
                    GraphOptimizationLevel graph_optimization_level);
ORT_API_STATUS_IMPL(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);
ORT_API_STATUS_IMPL(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

ORT_API_STATUS_IMPL(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);
ORT_API_STATUS_IMPL(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);
ORT_API_STATUS_IMPL(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
ORT_API_STATUS_IMPL(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle);

ORT_API_STATUS_IMPL(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** out);
ORT_API_STATUS_IMPL(SessionGetModelMetadata, _In_ const OrtSession* sess,
                    _Outptr_ OrtModelMetadata** out);

ORT_API_STATUS_IMPL(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetGraphDescription, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

ORT_API_STATUS_IMPL(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata,
                    _Out_ int64_t* value);

ORT_API_STATUS_IMPL(CreateRunOptions, _Outptr_ OrtRunOptions** out);

ORT_API_STATUS_IMPL(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunTag, _Inout_ OrtRunOptions*, _In_ const char* run_tag);

ORT_API_STATUS_IMPL(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

ORT_API_STATUS_IMPL(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
ORT_API_STATUS_IMPL(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

ORT_API_STATUS_IMPL(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(IsTensor, _In_ const OrtValue* value, _Out_ int* out);
ORT_API_STATUS_IMPL(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);
ORT_API_STATUS_IMPL(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
ORT_API_STATUS_IMPL(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);
ORT_API_STATUS_IMPL(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);
ORT_API_STATUS_IMPL(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                    size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);
ORT_API_STATUS_IMPL(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);
ORT_API_STATUS_IMPL(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*,
                    _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);
ORT_API_STATUS_IMPL(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);
ORT_API_STATUS_IMPL(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
ORT_API_STATUS_IMPL(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                    _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);
ORT_API_STATUS_IMPL(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);
ORT_API_STATUS_IMPL(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* dim_denotation, _In_ int64_t dim_value);

ORT_API_STATUS_IMPL(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);
ORT_API_STATUS_IMPL(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
ORT_API_STATUS_IMPL(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
ORT_API_STATUS_IMPL(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
ORT_API_STATUS_IMPL(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

ORT_API_STATUS_IMPL(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
ORT_API_STATUS_IMPL(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
ORT_API_STATUS_IMPL(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out);
ORT_API_STATUS_IMPL(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);
ORT_API_STATUS_IMPL(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);
ORT_API_STATUS_IMPL(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                    enum ONNXType value_type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                    _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size);

ORT_API_STATUS_IMPL(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);

ORT_API_STATUS_IMPL(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
ORT_API_STATUS_IMPL(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out);

// OrtTypeInfo methods
ORT_API_STATUS_IMPL(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation, _Out_ size_t* len);
ORT_API_STATUS_IMPL(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtMapTypeInfo** out);
ORT_API_STATUS_IMPL(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

// OrtSequenceTypeInfo Accessors
ORT_API_STATUS_IMPL(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);

ORT_API_STATUS_IMPL(DisablePerSessionThreads, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);
ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions*);

ORT_API_STATUS_IMPL(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);

ORT_API_STATUS_IMPL(AddFreeDimensionOverrideByName, _Inout_ OrtSessionOptions* options, _In_ const char* dim_name, _In_ int64_t dim_value);

ORT_API_STATUS_IMPL(CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info,
                    _Outptr_ OrtAllocator** out);
ORT_API(void, ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator);

ORT_API_STATUS_IMPL(RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options, _In_ const OrtIoBinding* binding_ptr);

ORT_API_STATUS_IMPL(CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out);
ORT_API(void, ReleaseIoBinding, _Frees_ptr_opt_ OrtIoBinding* allocator);

ORT_API_STATUS_IMPL(BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);
ORT_API_STATUS_IMPL(BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);
ORT_API_STATUS_IMPL(BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* val_ptr);
ORT_API_STATUS_IMPL(GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Out_ char** buffer, _Outptr_result_maybenull_ size_t** lengths, _Out_ size_t* count);
ORT_API_STATUS_IMPL(GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Outptr_result_maybenull_ OrtValue*** output, _Out_ size_t* output_count);

ORT_API(void, ClearBoundInputs, _Inout_ OrtIoBinding* binding_ptr);
ORT_API(void, ClearBoundOutputs, _Inout_ OrtIoBinding* binding_ptr);

ORT_API_STATUS_IMPL(GetAvailableProviders, _Outptr_ char*** out_ptr,
                    _In_ int* providers_length);
ORT_API_STATUS_IMPL(ReleaseAvailableProviders, _In_ char** ptr,
                    _In_ int providers_length);

ORT_API_STATUS_IMPL(AddSessionConfigEntry, _Inout_ OrtSessionOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value);

ORT_API_STATUS_IMPL(TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count, _Outptr_ void** out);

ORT_API_STATUS_IMPL(CreateAndRegisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg);

ORT_API_STATUS_IMPL(SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection);
ORT_API_STATUS_IMPL(SessionGetProfilingStartTimeNs, _In_ const OrtSession* sess, _Out_ uint64_t* out);

ORT_API_STATUS_IMPL(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads);
ORT_API_STATUS_IMPL(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads);
ORT_API_STATUS_IMPL(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning);
ORT_API_STATUS_IMPL(AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name,
                    _In_ const OrtValue* val);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options);
ORT_API_STATUS_IMPL(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* options);

ORT_API_STATUS_IMPL(CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
                    int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out);
ORT_API(void, ReleaseArenaCfg, _Frees_ptr_opt_ OrtArenaCfg*);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);
ORT_API_STATUS_IMPL(SetCurrentGpuDeviceId, _In_ int device_id);
ORT_API_STATUS_IMPL(GetCurrentGpuDeviceId, _In_ int* device_id);
ORT_API_STATUS_IMPL(KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out, _Inout_ size_t* size);
ORT_API_STATUS_IMPL(KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out, _Inout_ size_t* size);
ORT_API_STATUS_IMPL(CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys, _In_reads_(num_keys) const size_t* arena_config_values,
                    _In_ size_t num_keys, _Outptr_ OrtArenaCfg** out);
ORT_API_STATUS_IMPL(AddRunConfigEntry, _Inout_ OrtRunOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value);
ORT_API_STATUS_IMPL(CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out);
ORT_API(void, ReleasePrepackedWeightsContainer, _Frees_ptr_opt_ OrtPrepackedWeightsContainer*);
ORT_API_STATUS_IMPL(CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out);
ORT_API_STATUS_IMPL(CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out);
ORT_API_STATUS_IMPL(SetSessionOnnxOpsetVersion, _In_ OrtSessionOptions* options, int session_onnx_opset_version);
ORT_API_STATUS_IMPL(SessionAddONNXOpDomain, int session_onnx_opset_version);
}  // namespace OrtApis
