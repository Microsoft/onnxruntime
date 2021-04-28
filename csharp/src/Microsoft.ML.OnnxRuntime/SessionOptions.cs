// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Graph optimization level to use with SessionOptions
    ///  [https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md]
    /// </summary>
    public enum GraphOptimizationLevel
    {
        ORT_DISABLE_ALL = 0,
        ORT_ENABLE_BASIC = 1,
        ORT_ENABLE_EXTENDED = 2,
        ORT_ENABLE_ALL = 99
    }

    /// <summary>
    /// Controls whether you want to execute operators in the graph sequentially or in parallel.
    /// Usually when the model has many branches, setting this option to ExecutionMode.ORT_PARALLEL
    /// will give you better performance.
    /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
    /// </summary>
    public enum ExecutionMode
    {
        ORT_SEQUENTIAL = 0,
        ORT_PARALLEL = 1,
    }

    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions : SafeHandle
    {
        // Delay-loaded CUDA or cuDNN DLLs. Currently, delayload is disabled. See cmake/CMakeLists.txt for more information.
        private static string[] cudaDelayLoadedLibs = { };
        private static string[] trtDelayLoadedLibs = { };

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionOptions(out handle));
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <returns>A SessionsOptions() object configured for execution on deviceId=0</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider()
        {
            return MakeSessionOptionWithCudaProvider(0);
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider(int deviceId = 0)
        {
            CheckCudaExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, deviceId));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(int deviceId = 0)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(options.Handle, deviceId));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, deviceId));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trt_options">Provider Options for TensorRT EP.</param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(OrtTensorRTProviderOptions trt_options)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();

            OrtTensorRTProviderOptionsNative trt_options_native;
            trt_options_native = PrepareNativeTensorRTProviderOptions(trt_options);

            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_TensorRT(options.Handle, ref trt_options_native));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, trt_options.device_id));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for Nuphar execution.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">settings string, comprises of comma separated key:value pairs. default is empty</param>
        /// <returns>A SessionsOptions() object configured for execution with Nuphar</returns>
        public static SessionOptions MakeSessionOptionWithNupharProvider(String settings = "")
        {
            SessionOptions options = new SessionOptions();

            var settingsPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(settings), GCHandleType.Pinned);
            using (var pinnedSettingsName = new PinnedGCHandle(settingsPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(options.Handle, 1, pinnedSettingsName.Pointer));
            }

            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCM execution.
        /// Use only if ROCM is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <returns>A SessionsOptions() object configured for execution on deviceId=0</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider()
        {
            return MakeSessionOptionWithRocmProvider(0);
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCM execution.
        /// Use only if ROCM is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider(int deviceId = 0)
        {
            SessionOptions options = new SessionOptions();
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_ROCM(options.Handle, deviceId));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        #endregion

        #region ExecutionProviderAppends
        /// <summary>
        /// Appends CPU EP to a list of available execution providers for the session.
        /// </summary>
        /// <param name="useArena">1 - use arena, 0 - do not use arena</param>
        public void AppendExecutionProvider_CPU(int useArena)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(handle, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="useArena">1 - use allocation arena, 0 - otherwise</param>
        public void AppendExecutionProvider_Dnnl(int useArena)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Dnnl(handle, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">integer device ID</param>
        public void AppendExecutionProvider_CUDA(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(handle, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_DML(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_DML(handle, deviceId));
        }


        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification, default empty string</param>
        public void AppendExecutionProvider_OpenVINO(string deviceId = "")
        {
            var deviceIdPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(deviceId), GCHandleType.Pinned);
            using (var pinnedDeviceIdName = new PinnedGCHandle(deviceIdPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_OpenVINO(handle, pinnedDeviceIdName.Pointer));
            }
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_Tensorrt(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(handle, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trt_options">Provider Options for TensorRT EP.</param>
        public void AppendExecutionProvider_Tensorrt(OrtTensorRTProviderOptions trt_options)
        {
            OrtTensorRTProviderOptionsNative trt_options_native;
            trt_options_native = PrepareNativeTensorRTProviderOptions(trt_options);

            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_TensorRT(handle, ref trt_options_native));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">integer device ID</param>
        public void AppendExecutionProvider_ROCM(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_ROCM(handle, deviceId));
        }
       
        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_MIGraphX(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_MIGraphX(handle, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapi_flags">nnapi specific flag mask</param>
        public void AppendExecutionProvider_Nnapi(uint nnapi_flags)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, nnapi_flags));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">string with Nuphar specific settings</param>
        public void AppendExecutionProvider_Nuphar(string settings = "")
        {
            var settingsPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(settings), GCHandleType.Pinned);
            using (var pinnedSettingsName = new PinnedGCHandle(settingsPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(handle, 1, pinnedSettingsName.Pointer));
            }
        }
        #endregion //ExecutionProviderAppends

        #region Public Methods

        /// <summary>
        /// (Deprecated) Loads a DLL named 'libraryPath' and looks for this entry point:
        /// OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        /// Deprecated in favor of RegisterCustomOpLibraryV2() because it provides users with the library handle 
        /// to release when all sessions relying on it are destroyed
        /// </summary>
        /// <param name="libraryPath">path to the custom op library</param>
        [ObsoleteAttribute("RegisterCustomOpLibrary(...) is obsolete. Use RegisterCustomOpLibraryV2(...) instead.", false)]
        public void RegisterCustomOpLibrary(string libraryPath)
        {
            IntPtr libraryHandle = IntPtr.Zero;
            var libraryPathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(libraryPath), GCHandleType.Pinned);
            using (var pinnedlibraryPath = new PinnedGCHandle(libraryPathPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(handle, pinnedlibraryPath.Pointer, out libraryHandle));
            }
        }

        /// <summary>
        /// Loads a DLL named 'libraryPath' and looks for this entry point:
        /// OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        /// The handle to the loaded library is returned in 'libraryHandle'.
        /// It can be unloaded by the caller after all sessions using the passed in
        /// session options are destroyed, or if an error occurs and it is non null.
        /// Hint: .NET Core 3.1 has a 'NativeLibrary' class that can be used to free the library handle
        /// </summary>
        /// <param name="libraryPath">Custom op library path</param>
        /// <param name="libraryHandle">out parameter, library handle</param>
        public void RegisterCustomOpLibraryV2(string libraryPath, out IntPtr libraryHandle)
        {
            var libraryPathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(libraryPath), GCHandleType.Pinned);
            using (var pinnedlibraryPath = new PinnedGCHandle(libraryPathPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(handle, pinnedlibraryPath.Pointer, out libraryHandle));
            }
        }

        /// <summary>
        /// Add a pre-allocated initializer to a session. If a model contains an initializer with a name
        /// that is same as the name passed to this API call, ORT will use this initializer instance
        /// instead of deserializing one from the model file. This is useful when you want to share
        /// the same initializer across sessions.
        /// </summary>
        /// <param name="name">name of the initializer</param>
        /// <param name="ortValue">OrtValue containing the initializer. Lifetime of 'val' and the underlying initializer buffer must be
        /// managed by the user (created using the CreateTensorWithDataAsOrtValue API) and it must outlive the session object</param>
        public void AddInitializer(string name, OrtValue ortValue)
        {
            var utf8NamePinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name), GCHandleType.Pinned);
            using (var pinnedName = new PinnedGCHandle(utf8NamePinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAddInitializer(handle, pinnedName.Pointer, ortValue.Handle));
            }
        }

        /// <summary>
        /// Set a single session configuration entry as a pair of strings
        /// If a configuration with same key exists, this will overwrite the configuration with the given configValue
        /// </summary>
        /// <param name="configKey">config key name</param>
        /// <param name="configValue">config key value</param>
        public void AddSessionConfigEntry(string configKey, string configValue)
        {
            using (var pinnedConfigKeyName = new PinnedGCHandle(GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(configKey), GCHandleType.Pinned)))
            using (var pinnedConfigValueName = new PinnedGCHandle(GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(configValue), GCHandleType.Pinned)))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAddSessionConfigEntry(handle,
                                              pinnedConfigKeyName.Pointer, pinnedConfigValueName.Pointer));
            }
        }

        /// <summary>
        /// Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
        /// optimizations that can take advantage of fixed values (such as memory planning, etc)
        /// </summary>
        /// <param name="dimDenotation">denotation name</param>
        /// <param name="dimValue">denotation value</param>
        public void AddFreeDimensionOverride(string dimDenotation, long dimValue)
        {
            var utf8DimDenotationPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(dimDenotation), GCHandleType.Pinned);
            using (var pinnedDimDenotation = new PinnedGCHandle(utf8DimDenotationPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAddFreeDimensionOverride(handle, pinnedDimDenotation.Pointer, dimValue));
            }
        }

        /// <summary>
        /// Override symbolic dimensions (by specific name strings) with actual values if known at session initialization time to enable
        /// optimizations that can take advantage of fixed values (such as memory planning, etc)
        /// </summary>
        /// <param name="dimName">dimension name</param>
        /// <param name="dimValue">dimension value</param>
        public void AddFreeDimensionOverrideByName(string dimName, long dimValue)
        {
            var utf8DimNamePinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(dimName), GCHandleType.Pinned);
            using (var pinnedDimName = new PinnedGCHandle(utf8DimNamePinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAddFreeDimensionOverrideByName(handle, pinnedDimName.Pointer, dimValue));
            }
        }

        #endregion

        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }
        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
        /// </summary>
        /// <value>returns enableMemoryPattern flag value</value>
        public bool EnableMemoryPattern
        {
            get
            {
                return _enableMemoryPattern;
            }
            set
            {
                if (!_enableMemoryPattern && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableMemPattern(handle));
                    _enableMemoryPattern = true;
                }
                else if (_enableMemoryPattern && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableMemPattern(handle));
                    _enableMemoryPattern = false;
                }
            }
        }
        private bool _enableMemoryPattern = true;

        /// <summary>
        /// Path prefix to use for output of profiling data
        /// </summary>
        public string ProfileOutputPathPrefix
        {
            get; set;
        } = "onnxruntime_profile_";   // this is the same default in C++ implementation



        /// <summary>
        /// Enables profiling of InferenceSession.Run() calls. Default is false
        /// </summary>
        /// <value>returns _enableProfiling flag value</value>
        public bool EnableProfiling
        {
            get
            {
                return _enableProfiling;
            }
            set
            {
                if (!_enableProfiling && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableProfiling(handle, NativeMethods.GetPlatformSerializedString(ProfileOutputPathPrefix)));
                    _enableProfiling = true;
                }
                else if (_enableProfiling && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableProfiling(handle));
                    _enableProfiling = false;
                }
            }
        }
        private bool _enableProfiling = false;

        /// <summary>
        ///  Set filepath to save optimized model after graph level transformations. Default is empty, which implies saving is disabled.
        /// </summary>
        /// <value>returns _optimizedModelFilePath flag value</value>
        public string OptimizedModelFilePath
        {
            get
            {
                return _optimizedModelFilePath;
            }
            set
            {
                if (value != _optimizedModelFilePath)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtSetOptimizedModelFilePath(handle, NativeMethods.GetPlatformSerializedString(value)));
                    _optimizedModelFilePath = value;
                }
            }
        }
        private string _optimizedModelFilePath = "";



        /// <summary>
        /// Enables Arena allocator for the CPU memory allocations. Default is true.
        /// </summary>
        /// <value>returns _enableCpuMemArena flag value</value>
        public bool EnableCpuMemArena
        {
            get
            {
                return _enableCpuMemArena;
            }
            set
            {
                if (!_enableCpuMemArena && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableCpuMemArena(handle));
                    _enableCpuMemArena = true;
                }
                else if (_enableCpuMemArena && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableCpuMemArena(handle));
                    _enableCpuMemArena = false;
                }
            }
        }
        private bool _enableCpuMemArena = true;


        /// <summary>
        /// Log Id to be used for the session. Default is empty string.
        /// </summary>
        /// <value>returns _logId value</value>
        public string LogId
        {
            get
            {
                return _logId;
            }

            set
            {
                var logIdPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(value), GCHandleType.Pinned);
                using (var pinnedlogIdName = new PinnedGCHandle(logIdPinned))
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogId(handle, pinnedlogIdName.Pointer));
                }

                _logId = value;
            }
        }
        private string _logId = "";

        /// <summary>
        /// Log Severity Level for the session logs. Default = ORT_LOGGING_LEVEL_WARNING
        /// </summary>
        /// <value>returns _logSeverityLevel value</value>
        public OrtLoggingLevel LogSeverityLevel
        {
            get
            {
                return _logSeverityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogSeverityLevel(handle, value));
                _logSeverityLevel = value;
            }
        }
        private OrtLoggingLevel _logSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
        /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
        /// </summary>
        /// <value>returns _logVerbosityLevel value</value>
        public int LogVerbosityLevel
        {
            get
            {
                return _logVerbosityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogVerbosityLevel(handle, value));
                _logVerbosityLevel = value;
            }
        }
        private int _logVerbosityLevel = 0;


        /// <summary>
        // Sets the number of threads used to parallelize the execution within nodes
        // A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _intraOpNumThreads value</value>
        public int IntraOpNumThreads
        {
            get
            {
                return _intraOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetIntraOpNumThreads(handle, value));
                _intraOpNumThreads = value;
            }
        }
        private int _intraOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        // Sets the number of threads used to parallelize the execution of the graph (across nodes)
        // If sequential execution is enabled this value is ignored
        // A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _interOpNumThreads value</value>
        public int InterOpNumThreads
        {
            get
            {
                return _interOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetInterOpNumThreads(handle, value));
                _interOpNumThreads = value;
            }
        }
        private int _interOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_ALL.
        /// </summary>
        /// <value>returns _graphOptimizationLevel value</value>
        public GraphOptimizationLevel GraphOptimizationLevel
        {
            get
            {
                return _graphOptimizationLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionGraphOptimizationLevel(handle, value));
                _graphOptimizationLevel = value;
            }
        }
        private GraphOptimizationLevel _graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        /// <summary>
        /// Sets the execution mode for the session. Default is set to ORT_SEQUENTIAL.
        /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
        /// </summary>
        /// <value>returns _executionMode value</value>
        public ExecutionMode ExecutionMode
        {
            get
            {
                return _executionMode;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionExecutionMode(handle, value));
                _executionMode = value;
            }
        }
        private ExecutionMode _executionMode = ExecutionMode.ORT_SEQUENTIAL;

        #endregion

        #region Private Methods


        // Declared, but called only if OS = Windows.
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32.dll")]
        static extern uint GetSystemDirectory([Out] StringBuilder lpBuffer, uint uSize);
        private static bool CheckCudaExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in cudaDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile,
                        $"kernel32.LoadLibrary():'{dll}' not found. CUDA is required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );
                }
            }
            return true;
        }

        private static bool CheckTensorrtExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in trtDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile,
                        $"kernel32.LoadLibrary():'{dll}' not found. TensorRT/CUDA are required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );
                }
            }
            return true;
        }

        private static OrtTensorRTProviderOptionsNative PrepareNativeTensorRTProviderOptions(OrtTensorRTProviderOptions trt_options)
        {
            OrtTensorRTProviderOptionsNative trt_options_native;
            trt_options_native.device_id = trt_options.device_id;
            trt_options_native.has_user_compute_stream = 0;
            trt_options_native.user_compute_stream = IntPtr.Zero;
            trt_options_native.has_trt_options = trt_options.has_trt_options;
            if ((ulong)trt_options.trt_max_workspace_size > (1 << 30))
            {
                trt_options_native.trt_max_workspace_size = (UIntPtr)(1 << 30);
            }
            else
            {
                trt_options_native.trt_max_workspace_size = trt_options.trt_max_workspace_size;
            }
            trt_options_native.trt_fp16_enable = trt_options.trt_fp16_enable;
            trt_options_native.trt_int8_enable = trt_options.trt_int8_enable;
            var tableNamePinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(trt_options.trt_int8_calibration_table_name), GCHandleType.Pinned);
            using (var pinnedSettingsName = new PinnedGCHandle(tableNamePinned))
            {
                trt_options_native.trt_int8_calibration_table_name = pinnedSettingsName.Pointer;
            }
            trt_options_native.trt_int8_use_native_calibration_table = trt_options.trt_int8_use_native_calibration_table;
            trt_options_native.trt_max_partition_iterations = trt_options.trt_max_partition_iterations;
            trt_options_native.trt_min_subgraph_size = trt_options.trt_min_subgraph_size;
            trt_options_native.trt_dump_subgraphs = trt_options.trt_dump_subgraphs;
            trt_options_native.trt_engine_cache_enable = trt_options.trt_engine_cache_enable;
            var cachePathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(trt_options.trt_cache_path), GCHandleType.Pinned);
            using (var pinnedSettingsName2 = new PinnedGCHandle(cachePathPinned))
            {
                trt_options_native.trt_cache_path = pinnedSettingsName2.Pointer;
            }

            return trt_options_native;
        }


        #endregion
        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of SessionOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseSessionOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}
