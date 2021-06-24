// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OrtApiBase
    {
        public IntPtr GetApi;
        public IntPtr GetVersionString;
    };

    // NOTE: The order of the APIs in this struct should match exactly that in
    // OrtApi ort_api_1_to_4 (onnxruntime_c_api.cc)
    [StructLayout(LayoutKind.Sequential)]
    public struct OrtApi
    {
        public IntPtr CreateStatus;
        public IntPtr GetErrorCode;
        public IntPtr GetErrorMessage;
        public IntPtr CreateEnv;
        public IntPtr CreateEnvWithCustomLogger;
        public IntPtr EnableTelemetryEvents;
        public IntPtr DisableTelemetryEvents;
        public IntPtr CreateSession;
        public IntPtr CreateSessionFromArray;
        public IntPtr Run;

        public IntPtr CreateSessionOptions;
        public IntPtr SetOptimizedModelFilePath;
        public IntPtr CloneSessionOptions;
        public IntPtr SetSessionExecutionMode;
        public IntPtr EnableProfiling;
        public IntPtr DisableProfiling;
        public IntPtr EnableMemPattern;
        public IntPtr DisableMemPattern;
        public IntPtr EnableCpuMemArena;
        public IntPtr DisableCpuMemArena;
        public IntPtr SetSessionLogId;
        public IntPtr SetSessionLogVerbosityLevel;
        public IntPtr SetSessionLogSeverityLevel;
        public IntPtr SetSessionGraphOptimizationLevel;
        public IntPtr SetIntraOpNumThreads;
        public IntPtr SetInterOpNumThreads;

        public IntPtr CreateCustomOpDomain;
        public IntPtr CustomOpDomain_Add;
        public IntPtr AddCustomOpDomain;
        public IntPtr RegisterCustomOpsLibrary;

        public IntPtr SessionGetInputCount;
        public IntPtr SessionGetOutputCount;
        public IntPtr SessionGetOverridableInitializerCount;
        public IntPtr SessionGetInputTypeInfo;
        public IntPtr SessionGetOutputTypeInfo;
        public IntPtr SessionGetOverridableInitializerTypeInfo;
        public IntPtr SessionGetInputName;
        public IntPtr SessionGetOutputName;
        public IntPtr SessionGetOverridableInitializerName;
        public IntPtr CreateRunOptions;
        public IntPtr RunOptionsSetRunLogVerbosityLevel;
        public IntPtr RunOptionsSetRunLogSeverityLevel;
        public IntPtr RunOptionsSetRunTag;
        public IntPtr RunOptionsGetRunLogVerbosityLevel;
        public IntPtr RunOptionsGetRunLogSeverityLevel;
        public IntPtr RunOptionsGetRunTag;
        public IntPtr RunOptionsSetTerminate;
        public IntPtr RunOptionsUnsetTerminate;

        public IntPtr CreateTensorAsOrtValue;
        public IntPtr CreateTensorWithDataAsOrtValue;
        public IntPtr IsTensor;
        public IntPtr GetTensorMutableData;
        public IntPtr FillStringTensor;

        public IntPtr GetStringTensorDataLength;
        public IntPtr GetStringTensorContent;

        public IntPtr CastTypeInfoToTensorInfo;
        public IntPtr GetOnnxTypeFromTypeInfo;
        public IntPtr CreateTensorTypeAndShapeInfo;
        public IntPtr SetTensorElementType;

        public IntPtr SetDimensions;
        public IntPtr GetTensorElementType;
        public IntPtr GetDimensionsCount;
        public IntPtr GetDimensions;
        public IntPtr GetSymbolicDimensions;
        public IntPtr GetTensorShapeElementCount;
        public IntPtr GetTensorTypeAndShape;
        public IntPtr GetTypeInfo;
        public IntPtr GetValueType;
        public IntPtr CreateMemoryInfo;
        public IntPtr CreateCpuMemoryInfo;
        public IntPtr CompareMemoryInfo;
        public IntPtr MemoryInfoGetName;
        public IntPtr MemoryInfoGetId;
        public IntPtr MemoryInfoGetMemType;
        public IntPtr MemoryInfoGetType;
        public IntPtr AllocatorAlloc;
        public IntPtr AllocatorFree;
        public IntPtr AllocatorGetInfo;
        public IntPtr GetAllocatorWithDefaultOptions;
        public IntPtr AddFreeDimensionOverride;
        public IntPtr GetValue;
        public IntPtr GetValueCount;
        public IntPtr CreateValue;
        public IntPtr CreateOpaqueValue;
        public IntPtr GetOpaqueValue;

        public IntPtr KernelInfoGetAttribute_float;
        public IntPtr KernelInfoGetAttribute_int64;
        public IntPtr KernelInfoGetAttribute_string;
        public IntPtr KernelContext_GetInputCount;
        public IntPtr KernelContext_GetOutputCount;
        public IntPtr KernelContext_GetInput;
        public IntPtr KernelContext_GetOutput;

        public IntPtr ReleaseEnv;
        public IntPtr ReleaseStatus;
        public IntPtr ReleaseMemoryInfo;
        public IntPtr ReleaseSession;
        public IntPtr ReleaseValue;
        public IntPtr ReleaseRunOptions;
        public IntPtr ReleaseTypeInfo;
        public IntPtr ReleaseTensorTypeAndShapeInfo;
        public IntPtr ReleaseSessionOptions;
        public IntPtr ReleaseCustomOpDomain;
        public IntPtr GetDenotationFromTypeInfo;
        public IntPtr CastTypeInfoToMapTypeInfo;
        public IntPtr CastTypeInfoToSequenceTypeInfo;
        public IntPtr GetMapKeyType;
        public IntPtr GetMapValueType;
        public IntPtr GetSequenceElementType;
        public IntPtr ReleaseMapTypeInfo;
        public IntPtr ReleaseSequenceTypeInfo;
        public IntPtr SessionEndProfiling;

        public IntPtr SessionGetModelMetadata;
        public IntPtr ModelMetadataGetProducerName;
        public IntPtr ModelMetadataGetGraphName;
        public IntPtr ModelMetadataGetDomain;
        public IntPtr ModelMetadataGetDescription;
        public IntPtr ModelMetadataLookupCustomMetadataMap;
        public IntPtr ModelMetadataGetVersion;
        public IntPtr ReleaseModelMetadata;

        public IntPtr CreateEnvWithGlobalThreadPools;
        public IntPtr DisablePerSessionThreads;
        public IntPtr CreateThreadingOptions;
        public IntPtr ReleaseThreadingOptions;
        public IntPtr ModelMetadataGetCustomMetadataMapKeys;
        public IntPtr AddFreeDimensionOverrideByName;

        public IntPtr GetAvailableProviders;
        public IntPtr ReleaseAvailableProviders;
        public IntPtr GetStringTensorElementLength;
        public IntPtr GetStringTensorElement;
        public IntPtr FillStringTensorElement;
        public IntPtr AddSessionConfigEntry;

        public IntPtr CreateAllocator;
        public IntPtr ReleaseAllocator;
        public IntPtr RunWithBinding;
        public IntPtr CreateIoBinding;
        public IntPtr ReleaseIoBinding;
        public IntPtr BindInput;
        public IntPtr BindOutput;
        public IntPtr BindOutputToDevice;
        public IntPtr GetBoundOutputNames;
        public IntPtr GetBoundOutputValues;
        public IntPtr ClearBoundInputs;
        public IntPtr ClearBoundOutputs;
        public IntPtr TensorAt;
        public IntPtr CreateAndRegisterAllocator;
        public IntPtr SetLanguageProjection;
        public IntPtr SessionGetProfilingStartTimeNs;
        public IntPtr SetGlobalIntraOpNumThreads;
        public IntPtr SetGlobalInterOpNumThreads;
        public IntPtr SetGlobalSpinControl;
        public IntPtr AddInitializer;
        public IntPtr CreateEnvWithCustomLoggerAndGlobalThreadPools;
        public IntPtr SessionOptionsAppendExecutionProvider_CUDA;
        public IntPtr SessionOptionsAppendExecutionProvider_ROCM;
        public IntPtr SessionOptionsAppendExecutionProvider_OpenVINO;
        public IntPtr SetGlobalDenormalAsZero;
        public IntPtr CreateArenaCfg;
        public IntPtr ReleaseArenaCfg;
        public IntPtr ModelMetadataGetGraphDescription;
        public IntPtr SessionOptionsAppendExecutionProvider_TensorRT;
        public IntPtr SetCurrentGpuDeviceId;
        public IntPtr GetCurrentGpuDeviceId;
        public IntPtr KernelInfoGetAttributeArray_float;
        public IntPtr KernelInfoGetAttributeArray_int64;
        public IntPtr CreateArenaCfgV2;
        public IntPtr AddRunConfigEntry;
        public IntPtr CreatePrepackedWeightsContainer;
        public IntPtr ReleasePrepackedWeightsContainer;
        public IntPtr CreateSessionWithPrepackedWeightsContainer;
        public IntPtr CreateSessionFromArrayWithPrepackedWeightsContainer;
        public IntPtr SetSessionOnnxOpsetVersion;
        public IntPtr SessionAddONNXOpDomain;
    }

    internal static class NativeMethods
    {
        private const string nativeLib = "onnxruntime";
        internal const CharSet charSet = CharSet.Ansi;

        static OrtApi api_;

        public delegate ref OrtApi DOrtGetApi(UInt32 version);

        static NativeMethods()
        {
            DOrtGetApi OrtGetApi = (DOrtGetApi)Marshal.GetDelegateForFunctionPointer(OrtGetApiBase().GetApi, typeof(DOrtGetApi));

            // TODO: Make this save the pointer, and not copy the whole structure across
            api_ = (OrtApi)OrtGetApi(4 /*ORT_API_VERSION*/);

            OrtCreateEnv = (DOrtCreateEnv)Marshal.GetDelegateForFunctionPointer(api_.CreateEnv, typeof(DOrtCreateEnv));
            OrtReleaseEnv = (DOrtReleaseEnv)Marshal.GetDelegateForFunctionPointer(api_.ReleaseEnv, typeof(DOrtReleaseEnv));
            OrtEnableTelemetryEvents = (DOrtEnableTelemetryEvents)Marshal.GetDelegateForFunctionPointer(api_.EnableTelemetryEvents, typeof(DOrtEnableTelemetryEvents));
            OrtDisableTelemetryEvents = (DOrtDisableTelemetryEvents)Marshal.GetDelegateForFunctionPointer(api_.DisableTelemetryEvents, typeof(DOrtDisableTelemetryEvents));

            OrtGetErrorCode = (DOrtGetErrorCode)Marshal.GetDelegateForFunctionPointer(api_.GetErrorCode, typeof(DOrtGetErrorCode));
            OrtGetErrorMessage = (DOrtGetErrorMessage)Marshal.GetDelegateForFunctionPointer(api_.GetErrorMessage, typeof(DOrtGetErrorMessage));
            OrtReleaseStatus = (DOrtReleaseStatus)Marshal.GetDelegateForFunctionPointer(api_.ReleaseStatus, typeof(DOrtReleaseStatus));

            OrtCreateSession = (DOrtCreateSession)Marshal.GetDelegateForFunctionPointer(api_.CreateSession, typeof(DOrtCreateSession));
            OrtCreateSessionWithPrepackedWeightsContainer =
                (DOrtCreateSessionWithPrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionWithPrepackedWeightsContainer, typeof(DOrtCreateSessionWithPrepackedWeightsContainer));
            OrtCreateSessionFromArray = (DOrtCreateSessionFromArray)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionFromArray, typeof(DOrtCreateSessionFromArray));
            OrtCreateSessionFromArrayWithPrepackedWeightsContainer =
                (DOrtCreateSessionFromArrayWithPrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionFromArrayWithPrepackedWeightsContainer, typeof(DOrtCreateSessionFromArrayWithPrepackedWeightsContainer));
            OrtRun = (DOrtRun)Marshal.GetDelegateForFunctionPointer(api_.Run, typeof(DOrtRun));
            OrtRunWithBinding = (DOrtRunWithBinding)Marshal.GetDelegateForFunctionPointer(api_.RunWithBinding, typeof(DOrtRunWithBinding));
            OrtSessionGetInputCount = (DOrtSessionGetInputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputCount, typeof(DOrtSessionGetInputCount));
            OrtSessionGetOutputCount = (DOrtSessionGetOutputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputCount, typeof(DOrtSessionGetOutputCount));
            OrtSessionGetOverridableInitializerCount = (DOrtSessionGetOverridableInitializerCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerCount, typeof(DOrtSessionGetOverridableInitializerCount));

            OrtSessionGetInputName = (DOrtSessionGetInputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputName, typeof(DOrtSessionGetInputName));
            OrtSessionGetOutputName = (DOrtSessionGetOutputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputName, typeof(DOrtSessionGetOutputName));
            OrtSessionEndProfiling = (DOrtSessionEndProfiling)Marshal.GetDelegateForFunctionPointer(api_.SessionEndProfiling, typeof(DOrtSessionEndProfiling));
            OrtSessionGetOverridableInitializerName = (DOrtSessionGetOverridableInitializerName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerName, typeof(DOrtSessionGetOverridableInitializerName));
            OrtSessionGetInputTypeInfo = (DOrtSessionGetInputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputTypeInfo, typeof(DOrtSessionGetInputTypeInfo));
            OrtSessionGetOutputTypeInfo = (DOrtSessionGetOutputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputTypeInfo, typeof(DOrtSessionGetOutputTypeInfo));
            OrtSessionGetOverridableInitializerTypeInfo = (DOrtSessionGetOverridableInitializerTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerTypeInfo, typeof(DOrtSessionGetOverridableInitializerTypeInfo));
            OrtReleaseTypeInfo = (DOrtReleaseTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTypeInfo, typeof(DOrtReleaseTypeInfo));
            OrtReleaseSession = (DOrtReleaseSession)Marshal.GetDelegateForFunctionPointer(api_.ReleaseSession, typeof(DOrtReleaseSession));
            OrtSessionGetProfilingStartTimeNs = (DOrtSessionGetProfilingStartTimeNs)Marshal.GetDelegateForFunctionPointer(api_.SessionGetProfilingStartTimeNs, typeof(DOrtSessionGetProfilingStartTimeNs));

            OrtCreateSessionOptions = (DOrtCreateSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionOptions, typeof(DOrtCreateSessionOptions));
            OrtReleaseSessionOptions = (DOrtReleaseSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseSessionOptions, typeof(DOrtReleaseSessionOptions));
            OrtCloneSessionOptions = (DOrtCloneSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.CloneSessionOptions, typeof(DOrtCloneSessionOptions));
            OrtSetSessionExecutionMode = (DOrtSetSessionExecutionMode)Marshal.GetDelegateForFunctionPointer(api_.SetSessionExecutionMode, typeof(DOrtSetSessionExecutionMode));
            OrtSetOptimizedModelFilePath = (DOrtSetOptimizedModelFilePath)Marshal.GetDelegateForFunctionPointer(api_.SetOptimizedModelFilePath, typeof(DOrtSetOptimizedModelFilePath));
            OrtEnableProfiling = (DOrtEnableProfiling)Marshal.GetDelegateForFunctionPointer(api_.EnableProfiling, typeof(DOrtEnableProfiling));
            OrtDisableProfiling = (DOrtDisableProfiling)Marshal.GetDelegateForFunctionPointer(api_.DisableProfiling, typeof(DOrtDisableProfiling));
            OrtEnableMemPattern = (DOrtEnableMemPattern)Marshal.GetDelegateForFunctionPointer(api_.EnableMemPattern, typeof(DOrtEnableMemPattern));
            OrtDisableMemPattern = (DOrtDisableMemPattern)Marshal.GetDelegateForFunctionPointer(api_.DisableMemPattern, typeof(DOrtDisableMemPattern));
            OrtEnableCpuMemArena = (DOrtEnableCpuMemArena)Marshal.GetDelegateForFunctionPointer(api_.EnableCpuMemArena, typeof(DOrtEnableCpuMemArena));
            OrtDisableCpuMemArena = (DOrtDisableCpuMemArena)Marshal.GetDelegateForFunctionPointer(api_.DisableCpuMemArena, typeof(DOrtDisableCpuMemArena));
            OrtSetSessionLogId = (DOrtSetSessionLogId)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogId, typeof(DOrtSetSessionLogId));
            OrtSetSessionLogVerbosityLevel = (DOrtSetSessionLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogVerbosityLevel, typeof(DOrtSetSessionLogVerbosityLevel));
            OrtSetSessionLogSeverityLevel = (DOrtSetSessionLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogSeverityLevel, typeof(DOrtSetSessionLogSeverityLevel));
            OrtSetInterOpNumThreads = (DOrtSetInterOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetInterOpNumThreads, typeof(DOrtSetInterOpNumThreads));
            OrtSetIntraOpNumThreads = (DOrtSetIntraOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetIntraOpNumThreads, typeof(DOrtSetIntraOpNumThreads));
            OrtSetSessionGraphOptimizationLevel = (DOrtSetSessionGraphOptimizationLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionGraphOptimizationLevel, typeof(DOrtSetSessionGraphOptimizationLevel));
            OrtRegisterCustomOpsLibrary = (DOrtRegisterCustomOpsLibrary)Marshal.GetDelegateForFunctionPointer(api_.RegisterCustomOpsLibrary, typeof(DOrtRegisterCustomOpsLibrary));
            OrtAddSessionConfigEntry = (DOrtAddSessionConfigEntry)Marshal.GetDelegateForFunctionPointer(api_.AddSessionConfigEntry, typeof(DOrtAddSessionConfigEntry));
            OrtAddInitializer = (DOrtAddInitializer)Marshal.GetDelegateForFunctionPointer(api_.AddInitializer, typeof(DOrtAddInitializer));
            OrtSetSessionOnnxOpsetVersion = (DOrtSetSessionOnnxOpsetVersion)Marshal.GetDelegateForFunctionPointer(api_.SetSessionOnnxOpsetVersion, typeof(DOrtSetSessionOnnxOpsetVersion));

            OrtCreateRunOptions = (DOrtCreateRunOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateRunOptions, typeof(DOrtCreateRunOptions));
            OrtReleaseRunOptions = (DOrtReleaseRunOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseRunOptions, typeof(DOrtReleaseRunOptions));
            OrtRunOptionsSetRunLogVerbosityLevel = (DOrtRunOptionsSetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunLogVerbosityLevel, typeof(DOrtRunOptionsSetRunLogVerbosityLevel));
            OrtRunOptionsSetRunLogSeverityLevel = (DOrtRunOptionsSetRunLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunLogSeverityLevel, typeof(DOrtRunOptionsSetRunLogSeverityLevel));
            OrtRunOptionsSetRunTag = (DOrtRunOptionsSetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunTag, typeof(DOrtRunOptionsSetRunTag));
            OrtRunOptionsGetRunLogVerbosityLevel = (DOrtRunOptionsGetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunLogVerbosityLevel, typeof(DOrtRunOptionsGetRunLogVerbosityLevel));
            OrtRunOptionsGetRunLogSeverityLevel = (DOrtRunOptionsGetRunLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunLogSeverityLevel, typeof(DOrtRunOptionsGetRunLogSeverityLevel));
            OrtRunOptionsGetRunTag = (DOrtRunOptionsGetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunTag, typeof(DOrtRunOptionsGetRunTag));
            OrtRunOptionsSetTerminate = (DOrtRunOptionsSetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetTerminate, typeof(DOrtRunOptionsSetTerminate));
            OrtRunOptionsUnsetTerminate = (DOrtRunOptionsUnsetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsUnsetTerminate, typeof(DOrtRunOptionsUnsetTerminate));

            OrtCreateArenaCfg = (DOrtCreateArenaCfg)Marshal.GetDelegateForFunctionPointer(api_.CreateArenaCfg, typeof(DOrtCreateArenaCfg));
            OrtReleaseArenaCfg = (DOrtReleaseArenaCfg)Marshal.GetDelegateForFunctionPointer(api_.ReleaseArenaCfg, typeof(DOrtReleaseArenaCfg));
            OrtReleaseAllocator = (DOrtReleaseAllocator)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAllocator, typeof(DOrtReleaseAllocator));
            OrtCreateMemoryInfo = (DOrtCreateMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateMemoryInfo, typeof(DOrtCreateMemoryInfo));
            OrtCreateCpuMemoryInfo = (DOrtCreateCpuMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateCpuMemoryInfo, typeof(DOrtCreateCpuMemoryInfo));
            OrtReleaseMemoryInfo = (DOrtReleaseMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseMemoryInfo, typeof(DOrtReleaseMemoryInfo));
            OrtCompareMemoryInfo = (DOrtCompareMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CompareMemoryInfo, typeof(DOrtCompareMemoryInfo));
            OrtMemoryInfoGetName = (DOrtMemoryInfoGetName)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetName, typeof(DOrtMemoryInfoGetName));
            OrtMemoryInfoGetId = (DOrtMemoryInfoGetId)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetId, typeof(DOrtMemoryInfoGetId));
            OrtMemoryInfoGetMemType = (DOrtMemoryInfoGetMemType)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetMemType, typeof(DOrtMemoryInfoGetMemType));
            OrtMemoryInfoGetType = (DOrtMemoryInfoGetType)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetType, typeof(DOrtMemoryInfoGetType));
            OrtGetAllocatorWithDefaultOptions = (DOrtGetAllocatorWithDefaultOptions)Marshal.GetDelegateForFunctionPointer(api_.GetAllocatorWithDefaultOptions, typeof(DOrtGetAllocatorWithDefaultOptions));
            OrtCreateAllocator = (DOrtCreateAllocator)Marshal.GetDelegateForFunctionPointer(api_.CreateAllocator, typeof(DOrtCreateAllocator));
            OrtReleaseAllocator = (DOrtReleaseAllocator)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAllocator, typeof(DOrtReleaseAllocator));
            OrtAllocatorAlloc = (DOrtAllocatorAlloc)Marshal.GetDelegateForFunctionPointer(api_.AllocatorAlloc, typeof(DOrtAllocatorAlloc));
            OrtAllocatorFree = (DOrtAllocatorFree)Marshal.GetDelegateForFunctionPointer(api_.AllocatorFree, typeof(DOrtAllocatorFree));
            OrtAllocatorGetInfo = (DOrtAllocatorGetInfo)Marshal.GetDelegateForFunctionPointer(api_.AllocatorGetInfo, typeof(DOrtAllocatorGetInfo));
            OrtAddFreeDimensionOverride = (DOrtAddFreeDimensionOverride)Marshal.GetDelegateForFunctionPointer(api_.AddFreeDimensionOverride, typeof(DOrtAddFreeDimensionOverride));
            OrtAddFreeDimensionOverrideByName = (DOrtAddFreeDimensionOverrideByName)Marshal.GetDelegateForFunctionPointer(api_.AddFreeDimensionOverrideByName, typeof(DOrtAddFreeDimensionOverrideByName));

            OrtCreateIoBinding = (DOrtCreateIoBinding)Marshal.GetDelegateForFunctionPointer(api_.CreateIoBinding, typeof(DOrtCreateIoBinding));
            OrtReleaseIoBinding = (DOrtReleaseIoBinding)Marshal.GetDelegateForFunctionPointer(api_.ReleaseIoBinding, typeof(DOrtReleaseIoBinding));
            OrtBindInput = (DOrtBindInput)Marshal.GetDelegateForFunctionPointer(api_.BindInput, typeof(DOrtBindInput));
            OrtBindOutput = (DOrtBindOutput)Marshal.GetDelegateForFunctionPointer(api_.BindOutput, typeof(DOrtBindOutput));
            OrtBindOutputToDevice = (DOrtBindOutputToDevice)Marshal.GetDelegateForFunctionPointer(api_.BindOutputToDevice, typeof(DOrtBindOutputToDevice));
            OrtGetBoundOutputNames = (DOrtGetBoundOutputNames)Marshal.GetDelegateForFunctionPointer(api_.GetBoundOutputNames, typeof(DOrtGetBoundOutputNames));
            OrtGetBoundOutputValues = (DOrtGetBoundOutputValues)Marshal.GetDelegateForFunctionPointer(api_.GetBoundOutputValues, typeof(DOrtGetBoundOutputValues));
            OrtClearBoundInputs = (DOrtClearBoundInputs)Marshal.GetDelegateForFunctionPointer(api_.ClearBoundInputs, typeof(DOrtClearBoundInputs));
            OrtClearBoundOutputs = (DOrtClearBoundOutputs)Marshal.GetDelegateForFunctionPointer(api_.ClearBoundOutputs, typeof(DOrtClearBoundOutputs));
            OrtTensorAt = (DOrtTensorAt)Marshal.GetDelegateForFunctionPointer(api_.TensorAt, typeof(DOrtTensorAt));
            OrtCreateAndRegisterAllocator = (DOrtCreateAndRegisterAllocator)Marshal.GetDelegateForFunctionPointer(api_.CreateAndRegisterAllocator, typeof(DOrtCreateAndRegisterAllocator));
            OrtSetLanguageProjection = (DOrtSetLanguageProjection)Marshal.GetDelegateForFunctionPointer(api_.SetLanguageProjection, typeof(DOrtSetLanguageProjection));

            OrtGetValue = (DOrtGetValue)Marshal.GetDelegateForFunctionPointer(api_.GetValue, typeof(DOrtGetValue));
            OrtGetValueType = (DOrtGetValueType)Marshal.GetDelegateForFunctionPointer(api_.GetValueType, typeof(DOrtGetValueType));
            OrtGetOnnxTypeFromTypeInfo = (DOrtGetOnnxTypeFromTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetOnnxTypeFromTypeInfo, typeof(DOrtGetOnnxTypeFromTypeInfo));
            OrtGetValueCount = (DOrtGetValueCount)Marshal.GetDelegateForFunctionPointer(api_.GetValueCount, typeof(DOrtGetValueCount));
            OrtGetTypeInfo = (DOrtGetTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetTypeInfo, typeof(DOrtGetTypeInfo));
            OrtCreateTensorAsOrtValue = (DOrtCreateTensorAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorAsOrtValue, typeof(DOrtCreateTensorAsOrtValue));
            OrtCreateTensorWithDataAsOrtValue = (DOrtCreateTensorWithDataAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorWithDataAsOrtValue, typeof(DOrtCreateTensorWithDataAsOrtValue));
            OrtGetTensorMutableData = (DOrtGetTensorMutableData)Marshal.GetDelegateForFunctionPointer(api_.GetTensorMutableData, typeof(DOrtGetTensorMutableData));
            OrtFillStringTensor = (DOrtFillStringTensor)Marshal.GetDelegateForFunctionPointer(api_.FillStringTensor, typeof(DOrtFillStringTensor));
            OrtGetStringTensorContent = (DOrtGetStringTensorContent)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorContent, typeof(DOrtGetStringTensorContent));
            OrtGetStringTensorDataLength = (DOrtGetStringTensorDataLength)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorDataLength, typeof(DOrtGetStringTensorDataLength));
            OrtCastTypeInfoToTensorInfo = (DOrtCastTypeInfoToTensorInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToTensorInfo, typeof(DOrtCastTypeInfoToTensorInfo));
            OrtGetTensorTypeAndShape = (DOrtGetTensorTypeAndShape)Marshal.GetDelegateForFunctionPointer(api_.GetTensorTypeAndShape, typeof(DOrtGetTensorTypeAndShape));
            OrtReleaseTensorTypeAndShapeInfo = (DOrtReleaseTensorTypeAndShapeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTensorTypeAndShapeInfo, typeof(DOrtReleaseTensorTypeAndShapeInfo));
            OrtGetTensorElementType = (DOrtGetTensorElementType)Marshal.GetDelegateForFunctionPointer(api_.GetTensorElementType, typeof(DOrtGetTensorElementType));
            OrtGetDimensionsCount = (DOrtGetDimensionsCount)Marshal.GetDelegateForFunctionPointer(api_.GetDimensionsCount, typeof(DOrtGetDimensionsCount));
            OrtGetDimensions = (DOrtGetDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetDimensions, typeof(DOrtGetDimensions));
            OrtGetSymbolicDimensions = (DOrtGetSymbolicDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetSymbolicDimensions, typeof(DOrtGetSymbolicDimensions));
            OrtGetTensorShapeElementCount = (DOrtGetTensorShapeElementCount)Marshal.GetDelegateForFunctionPointer(api_.GetTensorShapeElementCount, typeof(DOrtGetTensorShapeElementCount));
            OrtReleaseValue = (DOrtReleaseValue)Marshal.GetDelegateForFunctionPointer(api_.ReleaseValue, typeof(DOrtReleaseValue));

            OrtSessionGetModelMetadata = (DOrtSessionGetModelMetadata)Marshal.GetDelegateForFunctionPointer(api_.SessionGetModelMetadata, typeof(DOrtSessionGetModelMetadata));
            OrtModelMetadataGetProducerName = (DOrtModelMetadataGetProducerName)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetProducerName, typeof(DOrtModelMetadataGetProducerName));
            OrtModelMetadataGetGraphName = (DOrtModelMetadataGetGraphName)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetGraphName, typeof(DOrtModelMetadataGetGraphName));
            OrtModelMetadataGetDomain = (DOrtModelMetadataGetDomain)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetDomain, typeof(DOrtModelMetadataGetDomain));
            OrtModelMetadataGetDescription = (DOrtModelMetadataGetDescription)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetDescription, typeof(DOrtModelMetadataGetDescription));
            OrtModelMetadataGetGraphDescription = (DOrtModelMetadataGetGraphDescription)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetGraphDescription, typeof(DOrtModelMetadataGetGraphDescription));
            OrtModelMetadataGetVersion = (DOrtModelMetadataGetVersion)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetVersion, typeof(DOrtModelMetadataGetVersion));
            OrtModelMetadataGetCustomMetadataMapKeys = (DOrtModelMetadataGetCustomMetadataMapKeys)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetCustomMetadataMapKeys, typeof(DOrtModelMetadataGetCustomMetadataMapKeys));
            OrtModelMetadataLookupCustomMetadataMap = (DOrtModelMetadataLookupCustomMetadataMap)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataLookupCustomMetadataMap, typeof(DOrtModelMetadataLookupCustomMetadataMap));
            OrtReleaseModelMetadata = (DOrtReleaseModelMetadata)Marshal.GetDelegateForFunctionPointer(api_.ReleaseModelMetadata, typeof(DOrtReleaseModelMetadata));

            OrtGetAvailableProviders = (DOrtGetAvailableProviders)Marshal.GetDelegateForFunctionPointer(api_.GetAvailableProviders, typeof(DOrtGetAvailableProviders));
            OrtReleaseAvailableProviders = (DOrtReleaseAvailableProviders)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAvailableProviders, typeof(DOrtReleaseAvailableProviders));

            OrtCreatePrepackedWeightsContainer = (DOrtCreatePrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreatePrepackedWeightsContainer, typeof(DOrtCreatePrepackedWeightsContainer));
            OrtReleasePrepackedWeightsContainer = (DOrtReleasePrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.ReleasePrepackedWeightsContainer, typeof(DOrtReleasePrepackedWeightsContainer));
            OrtSessionAddONNXOpDomain = (DOrtSessionAddONNXOpDomain)Marshal.GetDelegateForFunctionPointer(api_.SessionAddONNXOpDomain, typeof(DOrtSessionAddONNXOpDomain));
        }

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ref OrtApiBase OrtGetApiBase();

        #region Runtime/Environment API

        public delegate IntPtr /* OrtStatus* */DOrtCreateEnv(LogLevel default_warning_level, string logId, out IntPtr /*(OrtEnv*)*/ env);
        public static DOrtCreateEnv OrtCreateEnv;

        // OrtReleaseEnv should not be used
        public delegate void DOrtReleaseEnv(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtReleaseEnv OrtReleaseEnv;

        public delegate IntPtr /* OrtStatus* */DOrtEnableTelemetryEvents(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtEnableTelemetryEvents OrtEnableTelemetryEvents;

        public delegate IntPtr /* OrtStatus* */DOrtDisableTelemetryEvents(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtDisableTelemetryEvents OrtDisableTelemetryEvents;

        #endregion Runtime/Environment API

        #region Status API
        public delegate ErrorCode DOrtGetErrorCode(IntPtr /*(OrtStatus*)*/status);
        public static DOrtGetErrorCode OrtGetErrorCode;

        // returns char*, need to convert to string by the caller.
        // does not free the underlying OrtStatus*
        public delegate IntPtr /* char* */DOrtGetErrorMessage(IntPtr /* (OrtStatus*) */status);
        public static DOrtGetErrorMessage OrtGetErrorMessage;

        public delegate void DOrtReleaseStatus(IntPtr /*(OrtStatus*)*/ statusPtr);
        public static DOrtReleaseStatus OrtReleaseStatus;

        #endregion Status API

        #region InferenceSession API

        public delegate IntPtr /* OrtStatus* */DOrtCreateSession(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                //[MarshalAs(UnmanagedType.LPStr)]string modelPath
                                                byte[] modelPath,
                                                IntPtr /* (OrtSessionOptions*) */sessopnOptions,
                                                out IntPtr /**/ session);
        public static DOrtCreateSession OrtCreateSession;

        /// <summary>
        /// Creates an instance of OrtSession with provided parameters
        /// </summary>
        /// <param name="environment">Native OrtEnv instance</param>
        /// <param name="modelPath">UTF-8 bytes corresponding to model string path</param>
        /// <param name="sessionOptions">Native SessionOptions instance</param>         
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance</param>
        /// <param name="session">(Output) Created native OrtSession instance</param>
        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionWithPrepackedWeightsContainer(
                                        IntPtr /* (OrtEnv*) */ environment,
                                        byte[] modelPath,
                                        IntPtr /* (OrtSessionOptions*) */sessionOptions,
                                        IntPtr /* (OrtPrepackedWeightsContainer*) */prepackedWeightsContainer,
                                        out IntPtr /* (OrtSession**) */ session);
        public static DOrtCreateSessionWithPrepackedWeightsContainer OrtCreateSessionWithPrepackedWeightsContainer;

        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionFromArray(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                byte[] modelData,
                                                UIntPtr modelSize,
                                                IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                                out IntPtr /**/ session);
        public static DOrtCreateSessionFromArray OrtCreateSessionFromArray;

        /// <summary>
        /// Creates an instance of OrtSession with provided parameters
        /// </summary>
        /// <param name="environment">Native OrtEnv instance</param>
        /// <param name="modelData">Byte array correspoonding to the model</param>
        /// <param name="modelSize">Size of the model in bytes</param>
        /// <param name="sessionOptions">Native SessionOptions instance</param>         
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance</param>
        /// <param name="session">(Output) Created native OrtSession instance</param>
        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionFromArrayWithPrepackedWeightsContainer(
                                        IntPtr /* (OrtEnv*) */ environment,
                                        byte[] /* (void*) */ modelData,
                                        UIntPtr /* (size_t) */ modelSize,
                                        IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                        IntPtr /* (OrtPrepackedWeightsContainer*) */prepackedWeightsContainer,
                                        out IntPtr /* (OrtSession**) */ session);
        public static DOrtCreateSessionFromArrayWithPrepackedWeightsContainer OrtCreateSessionFromArrayWithPrepackedWeightsContainer;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionAddONNXOpDomain(int onnx_opset_version);
        public static DOrtSessionAddONNXOpDomain OrtSessionAddONNXOpDomain;

        public delegate IntPtr /*(ONNStatus*)*/ DOrtRun(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                IntPtr[] inputNames,
                                                IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                UIntPtr inputCount,
                                                IntPtr[] outputNames,
                                                UIntPtr outputCount,
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );
        public static DOrtRun OrtRun;

        public delegate IntPtr /*(ONNStatus*)*/ DOrtRunWithBinding(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions, // can not be null
                                                IntPtr /*(const OrtIoBinding*)*/ io_binding
                                                );
        public static DOrtRunWithBinding OrtRunWithBinding;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetInputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetInputCount OrtSessionGetInputCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOutputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetOutputCount OrtSessionGetOutputCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOverridableInitializerCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetOverridableInitializerCount OrtSessionGetOverridableInitializerCount;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetInputName OrtSessionGetInputName;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetOutputName OrtSessionGetOutputName;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionEndProfiling(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/profile_file);
        public static DOrtSessionEndProfiling OrtSessionEndProfiling;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetOverridableInitializerName OrtSessionGetOverridableInitializerName;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /*(struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetInputTypeInfo OrtSessionGetInputTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetOutputTypeInfo OrtSessionGetOutputTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetOverridableInitializerTypeInfo OrtSessionGetOverridableInitializerTypeInfo;

        // release the typeinfo using OrtReleaseTypeInfo
        public delegate void DOrtReleaseTypeInfo(IntPtr /*(OrtTypeInfo*)*/session);
        public static DOrtReleaseTypeInfo OrtReleaseTypeInfo;

        public delegate void DOrtReleaseSession(IntPtr /*(OrtSession*)*/session);
        public static DOrtReleaseSession OrtReleaseSession;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetProfilingStartTimeNs(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                out UIntPtr /*(ulong* out)*/ startTime);
        public static DOrtSessionGetProfilingStartTimeNs OrtSessionGetProfilingStartTimeNs;

        #endregion InferenceSession API

        #region SessionOptions API

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateSessionOptions(out IntPtr /*(OrtSessionOptions**)*/ sessionOptions);
        public static DOrtCreateSessionOptions OrtCreateSessionOptions;

        public delegate void DOrtReleaseSessionOptions(IntPtr /*(OrtSessionOptions*)*/session);
        public static DOrtReleaseSessionOptions OrtReleaseSessionOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCloneSessionOptions(IntPtr /*(OrtSessionOptions*)*/ sessionOptions, out IntPtr /*(OrtSessionOptions**)*/ output);
        public static DOrtCloneSessionOptions OrtCloneSessionOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionExecutionMode(IntPtr /*(OrtSessionOptions*)*/ options,
        ExecutionMode execution_mode);
        public static DOrtSetSessionExecutionMode OrtSetSessionExecutionMode;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetOptimizedModelFilePath(IntPtr /* OrtSessionOptions* */ options, byte[] optimizedModelFilepath);
        public static DOrtSetOptimizedModelFilePath OrtSetOptimizedModelFilePath;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableProfiling(IntPtr /* OrtSessionOptions* */ options, byte[] profilePathPrefix);
        public static DOrtEnableProfiling OrtEnableProfiling;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableProfiling(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableProfiling OrtDisableProfiling;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableMemPattern OrtEnableMemPattern;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableMemPattern OrtDisableMemPattern;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableCpuMemArena OrtEnableCpuMemArena;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableCpuMemArena OrtDisableCpuMemArena;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogId(IntPtr /* OrtSessionOptions* */ options, IntPtr /* const char* */logId);
        public static DOrtSetSessionLogId OrtSetSessionLogId;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogVerbosityLevel(IntPtr /* OrtSessionOptions* */ options, int sessionLogVerbosityLevel);
        public static DOrtSetSessionLogVerbosityLevel OrtSetSessionLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogSeverityLevel(IntPtr /* OrtSessionOptions* */ options, OrtLoggingLevel sessionLogSeverityLevel);
        public static DOrtSetSessionLogSeverityLevel OrtSetSessionLogSeverityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetIntraOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int intraOpNumThreads);
        public static DOrtSetIntraOpNumThreads OrtSetIntraOpNumThreads;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetInterOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int interOpNumThreads);
        public static DOrtSetInterOpNumThreads OrtSetInterOpNumThreads;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionGraphOptimizationLevel(IntPtr /* OrtSessionOptions* */ options, GraphOptimizationLevel graphOptimizationLevel);
        public static DOrtSetSessionGraphOptimizationLevel OrtSetSessionGraphOptimizationLevel;

        /// <summary>
        /// Add session config entry
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="configKey">Config key</param>
        /// <param name="configValue">Config value</param>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtAddSessionConfigEntry(IntPtr /* OrtSessionOptions* */ options,
                                                                          IntPtr /* const char* */configKey,
                                                                          IntPtr /* const char* */ configValue);
        public static DOrtAddSessionConfigEntry OrtAddSessionConfigEntry;

        ///**
        //  * The order of invocation indicates the preference order as well. In other words call this method
        //  * on your most preferred execution provider first followed by the less preferred ones.
        //  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
        //  */
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CPU(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Dnnl(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CUDA(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_ROCM(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_DML(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_OpenVINO(
                                                    IntPtr /*(OrtSessionOptions*)*/ options, IntPtr /*(const char*)*/ device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Tensorrt(IntPtr /*(OrtSessionOptions*)*/ options, int device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_MIGraphX(IntPtr /*(OrtSessionOptions*)*/ options, int device_id);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Nnapi(IntPtr /*(OrtSessionOptions*)*/ options, uint nnapi_flags);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Nuphar(IntPtr /*(OrtSessionOptions*) */ options,
                                                                                                     int allow_unaligned_buffers,
                                                                                                     IntPtr /*(char char*)*/ settings);

        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern void OrtAddCustomOp(IntPtr /*(OrtSessionOptions*)*/ options, string custom_op_path);

        /// <summary>
        /// Free Dimension override (by denotation)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="dimDenotation">Dimension denotation</param>
        /// <param name="dimValue">Dimension value</param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddFreeDimensionOverride(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                            IntPtr /*(const char*)*/ dimDenotation,
                                                                            long dimValue);
        public static DOrtAddFreeDimensionOverride OrtAddFreeDimensionOverride;

        /// <summary>
        /// Free Dimension override (by name)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="dimName">Dimension name</param>
        /// <param name="dimValue">Dimension value</param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddFreeDimensionOverrideByName(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                                  IntPtr /*(const char*)*/ dimName,
                                                                                  long dimValue);
        public static DOrtAddFreeDimensionOverrideByName OrtAddFreeDimensionOverrideByName;


        /// <summary>
        /// Register custom op library
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="libraryPath">Library path</param>
        /// <param name="libraryHandle">(out) Native library handle</param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtRegisterCustomOpsLibrary(IntPtr /*(OrtSessionOptions*) */ options,
                                                                            IntPtr /*(const char*)*/ libraryPath,
                                                                            out IntPtr /*(void**)*/ libraryHandle);
        public static DOrtRegisterCustomOpsLibrary OrtRegisterCustomOpsLibrary;

        /// <summary>
        /// Add initializer that is shared across Sessions using this SessionOptions (by denotation)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="name">Name of the initializer</param>
        /// <param name="ortValue">Native OrtValue instnce</param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddInitializer(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                  IntPtr /*(const char*)*/ name,
                                                                  IntPtr /*(OrtValue*)*/ ortValue);
        public static DOrtAddInitializer OrtAddInitializer;

        /// <summary>
        /// Set ONNX opset version to selectively load ONNX opset schema
        /// </summary>
        /// <param name="options">Set ONNX opset version</param>
        /// <param name="int">ONNX opset version</param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtSetSessionOnnxOpsetVersion(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                  int session_onnx_opset_version);
        public static DOrtSetSessionOnnxOpsetVersion OrtSetSessionOnnxOpsetVersion;

        
        #endregion

        #region RunOptions API
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateRunOptions(out IntPtr /* OrtRunOptions** */ runOptions);
        public static DOrtCreateRunOptions OrtCreateRunOptions;

        public delegate void DOrtReleaseRunOptions(IntPtr /*(OrtRunOptions*)*/options);
        public static DOrtReleaseRunOptions OrtReleaseRunOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, int value);
        public static DOrtRunOptionsSetRunLogVerbosityLevel OrtRunOptionsSetRunLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunLogSeverityLevel(IntPtr /* OrtRunOptions* */ options, OrtLoggingLevel value);
        public static DOrtRunOptionsSetRunLogSeverityLevel OrtRunOptionsSetRunLogSeverityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunTag(IntPtr /* OrtRunOptions* */ options, IntPtr /* const char* */ runTag);
        public static DOrtRunOptionsSetRunTag OrtRunOptionsSetRunTag;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, out int verbosityLevel);
        public static DOrtRunOptionsGetRunLogVerbosityLevel OrtRunOptionsGetRunLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunLogSeverityLevel(IntPtr /* OrtRunOptions* */ options, out OrtLoggingLevel severityLevel);
        public static DOrtRunOptionsGetRunLogSeverityLevel OrtRunOptionsGetRunLogSeverityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunTag(IntPtr /* const OrtRunOptions* */options, out IntPtr /* const char** */ runtag);
        public static DOrtRunOptionsGetRunTag OrtRunOptionsGetRunTag;

        // Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
        // will exit as soon as possible if the flag is true.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsSetTerminate OrtRunOptionsSetTerminate;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsUnsetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsUnsetTerminate OrtRunOptionsUnsetTerminate;



        #endregion

        #region Allocator/MemoryInfo API

        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateMemoryInfo(
                                                            IntPtr /*(const char*) */name,
                                                            OrtAllocatorType allocatorType,
                                                            int identifier,
                                                            OrtMemType memType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo    // memory ownership transfered to caller
                                                       );
        public static DOrtCreateMemoryInfo OrtCreateMemoryInfo;

        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateCpuMemoryInfo(
                                                            OrtAllocatorType allocatorType,
                                                            OrtMemType memoryType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo
                                                        );
        public static DOrtCreateCpuMemoryInfo OrtCreateCpuMemoryInfo;

        public delegate void DOrtReleaseMemoryInfo(IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo);
        public static DOrtReleaseMemoryInfo OrtReleaseMemoryInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCompareMemoryInfo(
                                               IntPtr /*(const OrtMemoryInfo*)*/ info1,
                                               IntPtr /*(const OrtMemoryInfo*)*/ info2,
                                               out int /*(int* out)*/ result);
        public static DOrtCompareMemoryInfo OrtCompareMemoryInfo;
        /**
        * Do not free the returned value
        */
        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetName(IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info, out IntPtr /*(const char**)*/ name);
        public static DOrtMemoryInfoGetName OrtMemoryInfoGetName;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetId(IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info, out int /*(int* out)*/ id);
        public static DOrtMemoryInfoGetId OrtMemoryInfoGetId;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetMemType(
                                                IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info,
                                                out OrtMemType /*(OrtMemType*)*/ mem_type);
        public static DOrtMemoryInfoGetMemType OrtMemoryInfoGetMemType;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetType(
                                                IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info,
                                                out OrtAllocatorType /*(OrtAllocatorType*)*/ alloc_type
                                                );
        public static DOrtMemoryInfoGetType OrtMemoryInfoGetType;

        public delegate IntPtr /*(OrtStatus*)*/DOrtGetAllocatorWithDefaultOptions(out IntPtr /*(OrtAllocator**)*/ allocator);
        public static DOrtGetAllocatorWithDefaultOptions OrtGetAllocatorWithDefaultOptions;

        public delegate IntPtr /*(OrtStatus*)*/DOrtAllocatorGetInfo(IntPtr /*(const OrtAllocator*)*/ ptr, out IntPtr /*(const struct OrtMemoryInfo**)*/info);
        public static DOrtAllocatorGetInfo OrtAllocatorGetInfo;

        /// <summary>
        /// Create an instance of arena configuration which will be used to create an arena based allocator
        /// See docs/C_API.md for details on what the following parameters mean and how to choose these values
        /// </summary>
        /// <param name="maxMemory">Maximum amount of memory the arena allocates</param>
        /// <param name="arenaExtendStrategy">Strategy for arena expansion</param>
        /// <param name="initialChunkSizeBytes">Size of the region that the arena allocates first</param>
        /// <param name="maxDeadBytesPerChunk">Maximum amount of fragmentation allowed per chunk</param>
        /// <returns>Pointer to a native OrtStatus instance indicating success/failure of config creation</returns>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateArenaCfg(UIntPtr /*(size_t)*/ maxMemory, int /*(int)*/ arenaExtendStrategy,
                                                                  int /*(int)*/ initialChunkSizeBytes, int /*(int)*/ maxDeadBytesPerChunk,
                                                                  out IntPtr /*(OrtArenaCfg**)*/ arenaCfg);
        public static DOrtCreateArenaCfg OrtCreateArenaCfg;

        /// <summary>
        /// Destroy an instance of an arena configuration instance
        /// </summary>
        /// <param name="arenaCfg">arena configuration instance to be destroyed</param>
        public delegate void DOrtReleaseArenaCfg(IntPtr /*(OrtArenaCfg*)*/ arenaCfg);
        public static DOrtReleaseArenaCfg OrtReleaseArenaCfg;

        /// <summary>
        /// Create an instance of allocator according to mem_info
        /// </summary>
        /// <param name="session">Session that this allocator should be used with</param>
        /// <param name="info">memory allocator specs</param>
        /// <param name="allocator">out pointer to a new allocator instance</param>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateAllocator(IntPtr /*(const OrtSession*)*/ session, IntPtr /*(const OrtMemoryInfo*)*/ info, out IntPtr /*(OrtAllocator**)*/ allocator);
        public static DOrtCreateAllocator OrtCreateAllocator;

        /// <summary>
        /// Destroy an instance of an allocator created by OrtCreateAllocator
        /// </summary>
        /// <param name="allocator">instance to be destroyed</param>
        public delegate void DOrtReleaseAllocator(IntPtr /*(OrtAllocator*)*/ allocator);
        public static DOrtReleaseAllocator OrtReleaseAllocator;

        /// <summary>
        /// Allocate  a chunk of native memory
        /// </summary>
        /// <param name="allocator">allocator instance</param>
        /// <param name="size">bytes to allocate</param>
        /// <param name="p">out pointer to the allocated memory. Must be freed by OrtAllocatorFree</param>
        public delegate IntPtr DOrtAllocatorAlloc(IntPtr /*(OrtAllocator*)*/ allocator, UIntPtr /*size_t*/ size, out IntPtr /*(void**)*/ p);
        public static DOrtAllocatorAlloc OrtAllocatorAlloc;

        /// <summary>
        /// Release native memory allocated by an allocator
        /// </summary>
        /// <param name="allocator">allocator instance</param>
        /// <param name="p">pointer to native memory allocated by the allocator instance</param>
        public delegate IntPtr DOrtAllocatorFree(IntPtr /*(OrtAllocator*)*/ allocator, IntPtr /*(void*)*/ p);
        public static DOrtAllocatorFree OrtAllocatorFree;

        #endregion Allocator/MemoryInfo API

        #region IoBinding API
        /// <summary>
        /// Create OrtIoBinding instance that is used to bind memory that is allocated
        /// either by a 3rd party allocator or an ORT device allocator. Such memory should be wrapped by
        /// a native OrtValue of Tensor type. By binding such named values you will direct ORT to read model inputs
        /// and write model outputs to the supplied memory.
        /// </summary>
        /// <param name="session">session to create OrtIoBinding instance</param>
        /// <param name="io_binding">out a new instance of OrtIoBinding</param>
        public delegate IntPtr /* OrtStatus*/ DOrtCreateIoBinding(IntPtr /*(const OrtSession*)*/ session, out IntPtr /*(OrtIoBinding)*/ io_binding);
        public static DOrtCreateIoBinding OrtCreateIoBinding;

        /// <summary>
        /// Destroy OrtIoBinding instance created by OrtCreateIoBinding
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        public delegate void DOrtReleaseIoBinding(IntPtr /*(OrtIoBinding)*/ io_binding);
        public static DOrtReleaseIoBinding OrtReleaseIoBinding;

        /// <summary>
        /// Bind OrtValue to the model input with the specified name
        /// If binding with the specified name already exists, it will be replaced
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        /// <param name="name">model input name (utf-8)</param>
        /// <param name="ort_value">OrtValue that is used for input (may wrap arbitrary memory).
        ///      The param instance is copied internally so this argument may be released.
        /// </param>
        public delegate IntPtr /* OrtStatus*/ DOrtBindInput(IntPtr /*(OrtIoBinding)*/ io_binding, IntPtr /*(const char*)*/ name, IntPtr /*const OrtValue**/ ort_value);
        public static DOrtBindInput OrtBindInput;

        /// <summary>
        /// Bind OrtValue to the model output with the specified name
        /// If binding with the specified name already exists, it will be replaced
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        /// <param name="name">model output name (utf-8)</param>
        /// <param name="ort_value">OrtValue that is used for output (may wrap arbitrary memory).
        ///      The param instance is copied internally so this argument may be released.
        /// </param>
        public delegate IntPtr /* OrtStatus*/ DOrtBindOutput(IntPtr /*(OrtIoBinding)*/ io_binding, IntPtr /*(const char*) */ name, IntPtr /*const OrtValue**/ ort_value);
        public static DOrtBindOutput OrtBindOutput;

        /// <summary>
        /// Bind a device to the model output with the specified name
        /// This is useful when the OrtValue can not be allocated ahead of time
        /// due to unknown dimensions.
        /// </summary>
        /// <param name="io_binding">Instance of OrtIoBinding</param>
        /// <param name="name">UTF-8 zero terminated name</param>
        /// <param name="mem_info">OrtMemoryInfo instance that contains device id. May be obtained from the device specific allocator instance</param>
        /// <returns></returns>
        public delegate IntPtr /* OrtStatus*/ DOrtBindOutputToDevice(IntPtr /*(OrtIoBinding)*/ io_binding, IntPtr /*(const char*) */ name, IntPtr /* const OrtMemoryInfo */ mem_info);
        public static DOrtBindOutputToDevice OrtBindOutputToDevice;

        /// <summary>
        /// The function will return all bound output names in the order they were bound.
        /// It is the same order that the output values will be returned after RunWithBinding() is used.
        /// The function will allocate two native allocations  using the allocator supplied.
        /// The caller is responsible for deallocating both of the buffers using the same allocator.
        /// You may use OrtMemoryAllocation disposable class to wrap those allocations.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        /// <param name="allocator">allocator to use for memory allocation</param>
        /// <param name="buffer">a continuous buffer that contains all output names.
        /// Names are not zero terminated use lengths to extract strings. This needs to be deallocated.</param>
        /// <param name="lengths">A buffer that contains lengths (size_t) for each of the returned strings in order.
        /// The buffer must be deallocated.</param>
        /// <param name="count">this contains the count of names returned which is the number of elements in lengths.</param>
        /// <returns></returns>
        public delegate IntPtr /* OrtStatus*/ DOrtGetBoundOutputNames(IntPtr /* (const OrtIoBinding*) */ io_binding, IntPtr /* OrtAllocator* */ allocator,
                                                                      out IntPtr /* char** */ buffer, out IntPtr /* size_t** */ lengths, out UIntPtr count);
        public static DOrtGetBoundOutputNames OrtGetBoundOutputNames;

        /// <summary>
        /// The function returns output values after the model has been run with RunWithBinding()
        /// It returns a natively allocated buffer of OrtValue pointers. All of the OrtValues must be individually
        /// released after no longer needed. You may use OrtValue disposable class to wrap the native handle and properly dispose it
        /// in connection with DisposableList<T>. All values are returned in the same order as they were bound.
        /// The buffer that contains OrtValues must deallocated using the same allocator that was specified as an argument.
        /// You may use an instance OrtMemoryAllocation to properly dispose of the native memory.
        /// </summary>
        /// <param name="io_binding">instance of OrtIOBinding</param>
        /// <param name="allocator">allocator to use to allocate output buffer</param>
        /// <param name="ortvalues">allocated buffer that contains pointers (IntPtr) to individual OrtValue instances</param>
        /// <param name="count">count of OrtValues returned</param>
        /// <returns></returns>
        public delegate IntPtr /* OrtStatus*/ DOrtGetBoundOutputValues(IntPtr /* (const OrtIoBinding*) */ io_binding, IntPtr /* OrtAllocator* */ allocator,
                                                                       out IntPtr /* OrtValue** */ ortvalues, out UIntPtr count);
        public static DOrtGetBoundOutputValues OrtGetBoundOutputValues;

        /// <summary>
        /// Clears Input bindings. This is a convenience method.
        /// Releasing OrtIoBinding instance would clear all bound inputs.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        public delegate void DOrtClearBoundInputs(IntPtr /*(OrtIoBinding)*/ io_binding);
        public static DOrtClearBoundInputs OrtClearBoundInputs;

        /// <summary>
        /// Clears Output bindings. This is a convenience method.
        /// Releasing OrtIoBinding instance would clear all bound outputs.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        public delegate void DOrtClearBoundOutputs(IntPtr /*(OrtIoBinding)*/ io_binding);
        public static DOrtClearBoundOutputs OrtClearBoundOutputs;

        /// <summary>
        /// Provides element-level access into a tensor.
        /// </summary>
        /// <param name="location_values">a pointer to an array of index values that specify an element's location in the tensor data blob</param>
        /// <param name="location_values_count">length of location_values</param>
        /// <param name="out">a pointer to the element specified by location_values</param>
        public delegate void DOrtTensorAt(IntPtr /*(OrtIoBinding)*/ io_binding);
        public static DOrtTensorAt OrtTensorAt;

        /// <summary>
        /// Creates an allocator instance and registers it with the env to enable
        /// sharing between multiple sessions that use the same env instance.
        /// Lifetime of the created allocator will be valid for the duration of the environment.
        /// Returns an error if an allocator with the same OrtMemoryInfo is already registered.
        /// <param name="env">Native OrtEnv instance</param>
        /// <param name="memInfo">Native OrtMemoryInfo instance</param>
        /// <param name="arenaCfg">Native OrtArenaCfg instance</param>
        /// <retruns>A pointer to native ortStatus indicating success/failure</retruns>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateAndRegisterAllocator(IntPtr /*(OrtEnv*)*/ env,
                                                                               IntPtr /*(const OrtMemoryInfo*)*/ memInfo,
                                                                               IntPtr/*(const OrtArenaCfg*)*/ arenaCfg);
        public static DOrtCreateAndRegisterAllocator OrtCreateAndRegisterAllocator;

        /// <summary>
        /// Set the language projection for collecting telemetry data when Env is created
        /// </summary>
        /// <param name="projection">the source projected language</param>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetLanguageProjection(IntPtr /* (OrtEnv*) */ environment, OrtLanguageProjection projection);
        public static DOrtSetLanguageProjection OrtSetLanguageProjection;

        #endregion IoBinding API

        #region ModelMetadata API

        /// <summary>
        /// Gets the ModelMetadata associated with an InferenceSession
        /// </summary>
        /// <param name="session">instance of OrtSession</param>
        /// <param name="modelMetadata">(output) instance of OrtModelMetadata</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtSessionGetModelMetadata(IntPtr /* (const OrtSession*) */ session, out IntPtr /* (OrtModelMetadata**) */ modelMetadata);
        public static DOrtSessionGetModelMetadata OrtSessionGetModelMetadata;

        /// <summary>
        /// Gets the producer name associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) producer name from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetProducerName(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataGetProducerName OrtModelMetadataGetProducerName;

        /// <summary>
        /// Gets the graph name associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) graph name from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetGraphName(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataGetGraphName OrtModelMetadataGetGraphName;

        /// <summary>
        /// Gets the domain associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) domain from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetDomain(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataGetDomain OrtModelMetadataGetDomain;

        /// <summary>
        /// Gets the description associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) description from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetDescription(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataGetDescription OrtModelMetadataGetDescription;

        /// <summary>
        /// Gets the description associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) graph description from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetGraphDescription(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataGetGraphDescription OrtModelMetadataGetGraphDescription;

        /// <summary>
        /// Gets the version associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="value">(output) version from the ModelMetadata instance</param>
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetVersion(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              out long /* (int64_t*) */ value);
        public static DOrtModelMetadataGetVersion OrtModelMetadataGetVersion;

        /// <summary>
        /// Gets all the keys in the custom metadata map in the ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="keys">(output) all keys in the custom metadata map</param>
        /// <param name="numKeys">(output) number of keys in the custom metadata map</param>

        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetCustomMetadataMapKeys(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
            IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char***) */ keys, out long /* (int64_t*) */ numKeys);
        public static DOrtModelMetadataGetCustomMetadataMapKeys OrtModelMetadataGetCustomMetadataMapKeys;

        /// <summary>
        /// Gets the value associated with the given key in custom metadata map in the ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="key">key in the custom metadata map</param>
        /// <param name="value">(output) value for the key in the custom metadata map</param>

        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataLookupCustomMetadataMap(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
            IntPtr /* (OrtAllocator*) */ allocator, IntPtr /* (const char*) */ key, out IntPtr /* (char**) */ value);
        public static DOrtModelMetadataLookupCustomMetadataMap OrtModelMetadataLookupCustomMetadataMap;


        /// <summary>
        /// Frees ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        public delegate void DOrtReleaseModelMetadata(IntPtr /*(OrtModelMetadata*)*/ modelMetadata);
        public static DOrtReleaseModelMetadata OrtReleaseModelMetadata;

        #endregion ModelMetadata API

        #region Tensor/OnnxValue API

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValue(IntPtr /*(OrtValue*)*/ value,
                                                                 int index,
                                                                 IntPtr /*(OrtAllocator*)*/ allocator,
                                                                 out IntPtr /*(OrtValue**)*/ outputValue);
        public static DOrtGetValue OrtGetValue;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueType(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(OnnxValueType*)*/ onnxtype);
        public static DOrtGetValueType OrtGetValueType;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetOnnxTypeFromTypeInfo(IntPtr /*(OrtTypeInfo*)*/ typeinfo, out IntPtr /*(OnnxValueType*)*/ onnxtype);
        public static DOrtGetOnnxTypeFromTypeInfo OrtGetOnnxTypeFromTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueCount(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(size_t*)*/ count);
        public static DOrtGetValueCount OrtGetValueCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTypeInfo(IntPtr /*(OrtValue*)*/ value, IntPtr /*(OrtValue**)*/ typeInfo);
        public static DOrtGetTypeInfo OrtGetTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateTensorAsOrtValue(
                        IntPtr /*_Inout_ OrtAllocator* */ allocator,
                        long[] /*_In_ const int64_t* */ shape,
                        UIntPtr /*size_t*/ shape_len,
                        Tensors.TensorElementType type,
                        out IntPtr /* OrtValue** */ outputValue);
        public static DOrtCreateTensorAsOrtValue OrtCreateTensorAsOrtValue;

        public delegate IntPtr /* OrtStatus */ DOrtCreateTensorWithDataAsOrtValue(
                                                        IntPtr /* (const OrtMemoryInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        UIntPtr dataLength,
                                                        long[] shape,
                                                        UIntPtr shapeLength,
                                                        Tensors.TensorElementType type,
                                                        out IntPtr /* OrtValue** */ outputValue);
        public static DOrtCreateTensorWithDataAsOrtValue OrtCreateTensorWithDataAsOrtValue;

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing OrtValue* is free'd.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorMutableData(IntPtr /*(OrtValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);
        public static DOrtGetTensorMutableData OrtGetTensorMutableData;

        /// \param value A tensor created from OrtCreateTensor... function.
        /// \param len total data length, not including the trailing '\0' chars.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtFillStringTensor(
                                                        IntPtr /* OrtValue */ value,
                                                        IntPtr[] /* const char* const* */s,
                                                        UIntPtr /* size_t */ s_len);
        public static DOrtFillStringTensor OrtFillStringTensor;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorContent(
                                                        IntPtr /*(OrtValue*)*/ value,
                                                        IntPtr /*(void*)*/  dst_buffer,
                                                        UIntPtr dst_buffer_len,
                                                        IntPtr offsets,
                                                        UIntPtr offsets_len);
        public static DOrtGetStringTensorContent OrtGetStringTensorContent;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorDataLength(IntPtr /*(OrtValue*)*/ value,
                                                        out UIntPtr /*(size_t*)*/ len);
        public static DOrtGetStringTensorDataLength OrtGetStringTensorDataLength;

        public delegate IntPtr /*(OrtStatus*)*/
                                DOrtCastTypeInfoToTensorInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /*(const struct OrtTensorTypeAndShapeInfo**)*/ typeAndShapeInfo);
        public static DOrtCastTypeInfoToTensorInfo OrtCastTypeInfoToTensorInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorTypeAndShape(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);
        public static DOrtGetTensorTypeAndShape OrtGetTensorTypeAndShape;


        public delegate void DOrtReleaseTensorTypeAndShapeInfo(IntPtr /*(OrtTensorTypeAndShapeInfo*)*/ value);
        public static DOrtReleaseTensorTypeAndShapeInfo OrtReleaseTensorTypeAndShapeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorElementType(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out IntPtr /*(TensorElementType*)*/ output);
        public static DOrtGetTensorElementType OrtGetTensorElementType;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetDimensionsCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out UIntPtr output);
        public static DOrtGetDimensionsCount OrtGetDimensionsCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetDimensions(
                            IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                            long[] dim_values,
                            UIntPtr dim_values_length);
        public static DOrtGetDimensions OrtGetDimensions;

        /**
        * Get the symbolic dimension names for dimensions with a value of -1.
        * Order and number of entries is the same as values returned by GetDimensions.
        * The name may be empty for an unnamed symbolic dimension.
        * e.g.
        * If OrtGetDimensions returns [-1, -1, 2], OrtGetSymbolicDimensions would return an array with 3 entries.
        * If the values returned were ['batch', '', ''] it would indicate that
        *  - the first dimension was a named symbolic dimension (-1 dim value and name in symbolic dimensions),
        *  - the second dimension was an unnamed symbolic dimension (-1 dim value and empty string),
        *  - the entry for the third dimension should be ignored as it is not a symbolic dimension (dim value >= 0).
        */
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetSymbolicDimensions(
                    IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                    IntPtr[] dim_params, /* const char* values, converted to string by caller */
                    UIntPtr dim_params_length);
        public static DOrtGetSymbolicDimensions OrtGetSymbolicDimensions;

        /**
         * How many elements does this tensor have.
         * May return a negative value
         * e.g.
         * [] -> 1
         * [1,3,4] -> 12
         * [2,0,4] -> 0
         * [-1,3,4] -> -1
         */
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorShapeElementCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out IntPtr /*(long*)*/ output);
        public static DOrtGetTensorShapeElementCount OrtGetTensorShapeElementCount;

        public delegate void DOrtReleaseValue(IntPtr /*(OrtValue*)*/ value);
        public static DOrtReleaseValue OrtReleaseValue;

        #endregion

        #region Misc API

        /// <summary>
        /// Queries all the execution providers supported in the native onnxruntime shared library
        /// </summary>
        /// <param name="providers">(output) all execution providers (strings) supported in the native onnxruntime shared library</param>
        /// <param name="numProviders">(output) number of execution providers (strings)</param>

        public delegate IntPtr /* (OrtStatus*) */ DOrtGetAvailableProviders(out IntPtr /* (char***) */ providers, out int /* (int*) */ numProviders);
        public static DOrtGetAvailableProviders OrtGetAvailableProviders;

        /// <summary>
        /// Releases all execution provider strings allocated and returned by OrtGetAvailableProviders
        /// </summary>
        /// <param name="providers">all execution providers (strings) returned by OrtGetAvailableProviders</param>
        /// <param name="numProviders">number of execution providers (strings)</param>

        public delegate IntPtr /* (OrtStatus*) */ DOrtReleaseAvailableProviders(IntPtr /* (char**) */ providers, int /* (int) */ numProviders);
        public static DOrtReleaseAvailableProviders OrtReleaseAvailableProviders;

        /// <summary>
        /// Create an instance of PrepackedWeightsContainer
        /// </summary>
        /// <param name="prepackedWeightsContainer">(output) Created native OrtPrepackedWeightsContainer instance</param>
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreatePrepackedWeightsContainer(out IntPtr /*(OrtPrepackedWeightsContainer**)*/ prepackedWeightsContainer);
        public static DOrtCreatePrepackedWeightsContainer OrtCreatePrepackedWeightsContainer;

        /// <summary>
        /// Destroy an instance of PrepackedWeightsContainer
        /// </summary>
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance to be destroyed</param>
        public delegate void DOrtReleasePrepackedWeightsContainer(IntPtr /*(OrtPrepackedWeightsContainer*)*/ prepackedWeightsContainer);
        public static DOrtReleasePrepackedWeightsContainer OrtReleasePrepackedWeightsContainer;
        #endregion

        public static byte[] GetPlatformSerializedString(string str)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return System.Text.Encoding.Unicode.GetBytes(str + Char.MinValue);
            else
                return System.Text.Encoding.UTF8.GetBytes(str + Char.MinValue);
        }
    } //class NativeMethods
} //namespace
