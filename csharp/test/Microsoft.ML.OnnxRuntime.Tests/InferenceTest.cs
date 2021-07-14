// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

// This runs in a separate package built from EndToEndTests
// and for this reason it can not refer to non-public members
// of Onnxruntime package
namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class InferenceTest
    {
        private const string module = "onnxruntime.dll";
        private const string propertiesFile = "Properties.txt";
        private readonly ITestOutputHelper output;

        public InferenceTest(ITestOutputHelper o)
        {
            this.output = o;
        }

        [Fact]
        public void TestSessionOptions()
        {
            using (SessionOptions opt = new SessionOptions())
            {
                Assert.NotNull(opt);

                // check default values of the properties
                Assert.Equal(ExecutionMode.ORT_SEQUENTIAL, opt.ExecutionMode);
                Assert.True(opt.EnableMemoryPattern);
                Assert.False(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);
                Assert.True(opt.EnableCpuMemArena);
                Assert.Equal("", opt.LogId);
                Assert.Equal(0, opt.LogVerbosityLevel);
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);
                Assert.Equal(0, opt.IntraOpNumThreads);
                Assert.Equal(0, opt.InterOpNumThreads);
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_ALL, opt.GraphOptimizationLevel);

                // try setting options
                opt.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                Assert.Equal(ExecutionMode.ORT_PARALLEL, opt.ExecutionMode);

                opt.EnableMemoryPattern = false;
                Assert.False(opt.EnableMemoryPattern);

                opt.EnableProfiling = true;
                Assert.True(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);

                opt.ProfileOutputPathPrefix = "Ort_P_";
                Assert.Equal("Ort_P_", opt.ProfileOutputPathPrefix);

                opt.EnableCpuMemArena = false;
                Assert.False(opt.EnableCpuMemArena);

                opt.LogId = "MyLogId";
                Assert.Equal("MyLogId", opt.LogId);

                opt.LogVerbosityLevel = 1;
                Assert.Equal(1, opt.LogVerbosityLevel);

                opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR, opt.LogSeverityLevel);

                opt.IntraOpNumThreads = 4;
                Assert.Equal(4, opt.IntraOpNumThreads);

                opt.InterOpNumThreads = 4;
                Assert.Equal(4, opt.InterOpNumThreads);

                opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, opt.GraphOptimizationLevel);

                Assert.Throws<OnnxRuntimeException>(() => { opt.GraphOptimizationLevel = (GraphOptimizationLevel)10; });

                opt.AddSessionConfigEntry("key", "value");

                var ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AddSessionConfigEntry("", "invalid key"); });
                Assert.Contains("[ErrorCode:InvalidArgument] Config key is empty", ex.Message);

                opt.AppendExecutionProvider_CPU(1);
#if USE_DNNL
                opt.AppendExecutionProvider_Dnnl(0);
#endif
#if USE_CUDA
                opt.AppendExecutionProvider_CUDA(0);
#endif
#if USE_ROCM
                opt.AppendExecutionProvider_ROCM(0);
#endif
#if USE_DML
                // Explicitly set dll probe path so that the (potentially) stale system DirectML.dll
                // doesn't get loaded by the test process when it is eventually delay loaded by onnruntime.dll
                // The managed tests binary path already contains the right DirectML.dll, so use that

                var directml_dll_path = AppDomain.CurrentDomain.BaseDirectory;
                SetDllDirectory(directml_dll_path);
                opt.AppendExecutionProvider_DML(0);

                // Restore the default dll search order
                SetDllDirectory(null);

#endif
#if USE_OPENVINO
                opt.AppendExecutionProvider_OpenVINO();
#endif
#if USE_TENSORRT
                opt.AppendExecutionProvider_Tensorrt(0);
#endif
#if USE_MIGRAPHX
                opt.AppendExecutionProvider_MIGraphX(0);
#endif
#if USE_NNAPI
                opt.AppendExecutionProvider_Nnapi(0);
#endif


            }
        }

        // Use to set dll probe path so that the right dll(s) is loaded by the test process
        // Invoke only to specify Windows specific EPs' dll locations explicitly
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]

        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);

        [Fact]
        public void TestRunOptions()
        {
            using (var opt = new RunOptions())
            {
                Assert.NotNull(opt);

                //verify default options
                Assert.False(opt.Terminate);
                Assert.Equal(0, opt.LogVerbosityLevel);
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);
                Assert.Equal("", opt.LogId);

                // try setting options
                opt.Terminate = true;
                Assert.True(opt.Terminate);

                opt.LogVerbosityLevel = 1;
                Assert.Equal(1, opt.LogVerbosityLevel);

                opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR, opt.LogSeverityLevel);

                opt.LogId = "MyLogTag";
                Assert.Equal("MyLogTag", opt.LogId);
            }
        }

        [Fact]
        public void EnablingAndDisablingTelemetryEventCollection()
        {
            var ortEnvInstance = OrtEnv.Instance();
            ortEnvInstance.DisableTelemetryEvents();

            // no-op on non-Windows builds
            // may be no-op on certain Windows builds based on build configuration

            ortEnvInstance.EnableTelemetryEvents();
        }

        [Fact]
        public void GetAvailableProviders()
        {
            var ortEnvInstance = OrtEnv.Instance();
            string[] providers = ortEnvInstance.GetAvailableProviders();

            Assert.True(providers.Length > 0);
            Assert.Equal("CPUExecutionProvider", providers[providers.Length - 1]);

# if USE_CUDA
            Assert.True(Array.Exists(providers, provider => provider == "CUDAExecutionProvider"));
#endif
# if USE_ROCM
            Assert.True(Array.Exists(providers, provider => provider == "ROCMExecutionProvider"));
#endif

        }

        [Fact]
        public void CanCreateAndDisposeSessionWithModelPath()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.NotNull(session);
                Assert.NotNull(session.InputMetadata);
                Assert.Equal(1, session.InputMetadata.Count); // 1 input node
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input node name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].ElementType);
                Assert.True(session.OutputMetadata["softmaxout_1"].IsTensor);
                var expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                Assert.Equal(expectedOutputDimensions.Length, session.OutputMetadata["softmaxout_1"].Dimensions.Length);
                for (int i = 0; i < expectedOutputDimensions.Length; i++)
                {
                    Assert.Equal(expectedOutputDimensions[i], session.OutputMetadata["softmaxout_1"].Dimensions[i]);
                }
            }
        }

#if USE_TENSORRT
        [Fact]
        private void CanRunInferenceOnAModelWithTensorRT()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(0);
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }


                using (var results = session.Run(container))
                {
                    validateRunResults(results);
                }
            }
        }

        [Fact]
        private void TestTensorRTProviderOptions()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string calTablePath = "squeezenet_calibration.flatbuffers";
            string enginePath = "./";
            string engineDecrptLibPath = "engine_decryp";

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var trtProviderOptions = new OrtTensorRTProviderOptions();
                cleanUp.Add(trtProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = "0";
                providerOptionsDict["trt_fp16_enable"] = "1";
                providerOptionsDict["trt_int8_enable"] = "1";
                providerOptionsDict["trt_int8_calibration_table_name"] = calTablePath;
                providerOptionsDict["trt_engine_cache_enable"] = "1";
                providerOptionsDict["trt_engine_cache_path"] = enginePath;
                providerOptionsDict["trt_engine_decryption_enable"] = "0";
                providerOptionsDict["trt_engine_decryption_lib_path"] = engineDecrptLibPath;
                trtProviderOptions.UpdateOptions(providerOptionsDict);

                var resultProviderOptionsDict = new Dictionary<string, string>();
                ProviderOptionsValueHelper.StringToDict(trtProviderOptions.GetOptions(), resultProviderOptionsDict);

                // test provider options configuration
                string value;
                value = resultProviderOptionsDict["device_id"];
                Assert.Equal("0", value);
                value = resultProviderOptionsDict["trt_fp16_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_int8_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_int8_calibration_table_name"];
                Assert.Equal(calTablePath, value);
                value = resultProviderOptionsDict["trt_engine_cache_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_engine_cache_path"];
                Assert.Equal(enginePath, value);
                value = resultProviderOptionsDict["trt_engine_decryption_enable"];
                Assert.Equal("0", value);
                value = resultProviderOptionsDict["trt_engine_decryption_lib_path"];
                Assert.Equal(engineDecrptLibPath, value);

                // test correctness of provider options
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                session.Run(container);
            }
        }
#endif

        [Theory]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, true)]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, false)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, true)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, false)]
        private void CanRunInferenceOnAModel(GraphOptimizationLevel graphOptimizationLevel, bool enableParallelExecution)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                // Set the graph optimization level for this session.
                SessionOptions options = new SessionOptions();
                options.GraphOptimizationLevel = graphOptimizationLevel;
                if (enableParallelExecution) options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                ReadOnlySpan<int> expectedOutputDimensions = expectedDimensions;
                string[] expectedOutputNames = new string[] { "softmaxout_1" };

                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }


                // Run inference with named inputs and outputs created with in Run()
                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                {
                    validateRunResults(results);
                }

                // Run inference with named inputs, outputs created with in Run() and RunOptions
                using (var runOptions = new RunOptions())
                {
                    runOptions.LogId = "CsharpTest";
                    runOptions.Terminate = false;  // TODO: Test terminate = true, it currently crashes
                    runOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                    IReadOnlyCollection<string> outputNames = session.OutputMetadata.Keys.ToList();

                    using (var results = session.Run(container, outputNames, runOptions))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        validateRunResults(results);
                    }
                }

                // Run inference with pinned inputs and outputs created with in Run()
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    // output names not specified
                    using (var results = session.Run(inputNames, pinnedInputs))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        validateRunResults(results);
                    }

                    // output names specified explicitly
                    using (var results = session.Run(inputNames, pinnedInputs, expectedOutputNames))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        validateRunResults(results);
                    }
                }

                // Run inference with outputs pinned from buffers
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                using (var pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

                    // Create inputs
                    Assert.Single(inputMeta.Keys);
                    var inputNames = inputMeta.Keys.ToArray();
                    var inputName = inputNames[0];
                    Assert.Equal(typeof(float), inputMeta[inputName].ElementType);
                    Assert.True(inputMeta[inputName].IsTensor);
                    var longShape = Array.ConvertAll<int, long>(inputMeta[inputName].Dimensions, d => d);
                    var byteSize = longShape.Aggregate(1L, (a, b) => a * b) * sizeof(float);
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, inputData,
                        TensorElementType.Float, longShape, byteSize));


                    // Prepare output buffer
                    Assert.Single(outputMeta.Keys);
                    var outputNames = outputMeta.Keys.ToArray();
                    var outputName = outputNames[0];
                    Assert.Equal(typeof(float), outputMeta[outputName].ElementType);
                    Assert.True(outputMeta[outputName].IsTensor);
                    longShape = Array.ConvertAll<int, long>(outputMeta[outputName].Dimensions, d => d);
                    byteSize = longShape.Aggregate(1L, (a, b) => a * b) * sizeof(float);
                    float[] outputBuffer = new float[expectedOutput.Length];
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, outputBuffer,
                        TensorElementType.Float, longShape, byteSize));

                    session.Run(inputNames, pinnedInputs, outputNames, pinnedOutputs);
                    Assert.Equal(expectedOutput, outputBuffer, new floatComparer());
                }

                // Run inference with named inputs and named outputs
                {
                    // correct pre-allocated outputs
                    var expectedOutputValues = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor("softmaxout_1", new DenseTensor<float>(expectedOutputDimensions))
                    };
                    session.Run(container, expectedOutputValues);
                    validateRunResultData(expectedOutputValues[0].AsTensor<float>(), expectedOutput, expectedDimensions);
                }

                // Run inference with pinned inputs and named outputs
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    // expected inputs and outputs
                    var expectedOutputValues = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor("softmaxout_1", new DenseTensor<float>(expectedOutputDimensions))
                    };
                    session.Run(inputNames, pinnedInputs, expectedOutputValues);
                    validateRunResultData(expectedOutputValues[0].AsTensor<float>(), expectedOutput, expectedDimensions);
                }

                // Run inference with named inputs and pinned outputs
                {
                    // correct pre-allocated outputs
                    using (var pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                    {
                        var outputTensor = new DenseTensor<float>(expectedOutputDimensions);
                        pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromTensor(outputTensor));
                        session.Run(container, expectedOutputNames, pinnedOutputs);
                        validateRunResultData(outputTensor, expectedOutput, expectedDimensions);
                    }
                }

                // Run inference with pinned inputs and pinned outputs
                using (DisposableListTest<FixedBufferOnnxValue> pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
                                                            pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    var outputTensor = new DenseTensor<float>(expectedOutputDimensions);
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromTensor(outputTensor));

                    session.Run(inputNames, pinnedInputs, expectedOutputNames, pinnedOutputs);
                    validateRunResultData(outputTensor, expectedOutput, expectedDimensions);
                }
            }
        }

        [Fact]
        public void InferenceSessionManualDisposeAfterUse()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            // Set the graph optimization level for this session.
            SessionOptions options = new SessionOptions();
            options.ProfileOutputPathPrefix = "Ort_P_";
            options.EnableProfiling = true;
            var session = new InferenceSession(modelPath, options);


            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();

            float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

            foreach (var name in inputMeta.Keys)
            {
                Assert.Equal(typeof(float), inputMeta[name].ElementType);
                Assert.True(inputMeta[name].IsTensor);
                var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
            }

            // Run inference with named inputs and outputs created with in Run()
            using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
            {
                validateRunResults(results);
            }

            string profile_file = session.EndProfiling();

            // Profile file should have the output path prefix in it
            Assert.Contains("Ort_P_", profile_file);

            // Should be able to dispose the session manually
            session.Dispose();

        }

        [Fact]
        public void InferenceSessionGetProfilingStartTimeNs()
        {
            ulong getSingleSessionProfilingStartTime()
            {
                ulong startTime = 0;
                using (SessionOptions options = new SessionOptions())
                {
                    options.EnableProfiling = true;
                    string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
                    using (var session = new InferenceSession(modelPath, options))
                    {
                        startTime = session.ProfilingStartTimeNs;
                    }
                }
                return startTime;
            }

            // Get profiling's start time
            ulong ProfilingStartTime = getSingleSessionProfilingStartTime();

            // Check the profiling's start time has been updated
            Assert.True(ProfilingStartTime != 0);
        }

        [Fact]
        public void SessionOptionsFreeDimensionOverrides()
        {

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "abs_free_dimensions.onnx");

            // By Name
            using (SessionOptions options = new SessionOptions())
            {
                options.AddFreeDimensionOverrideByName("Dim1", 4);
                options.AddFreeDimensionOverrideByName("Dim2", 6);

                using (var session = new InferenceSession(modelPath, options))
                {
                    var inputMetadata = session.InputMetadata;
                    var dims = inputMetadata["x"].Dimensions;
                    Assert.Equal(3, dims.Length);
                    Assert.Equal(4, dims[0]);
                    Assert.Equal(6, dims[1]);
                    Assert.Equal(5, dims[2]);
                }
            }

            // By Denotation
            using (SessionOptions options = new SessionOptions())
            {
                options.AddFreeDimensionOverride("DATA_BATCH", 3);
                options.AddFreeDimensionOverride("DATA_CHANNEL", 5);

                using (var session = new InferenceSession(modelPath, options))
                {
                    var inputMetadata = session.InputMetadata;
                    var dims = inputMetadata["x"].Dimensions;
                    Assert.Equal(3, dims.Length);
                    Assert.Equal(3, dims[0]);
                    Assert.Equal(5, dims[1]);
                    Assert.Equal(5, dims[2]);
                }
            }

        }

        private void validateRunResults(IReadOnlyCollection<NamedOnnxValue> results)
        {
            // validate the results
            foreach (var r in results)
            {
                Assert.Equal(1, results.Count);
                Assert.Equal("softmaxout_1", r.Name);

                float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                validateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
            }
        }

        private void validateRunResultData(Tensor<float> resultTensor, float[] expectedOutput, int[] expectedDimensions)
        {
            Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

            var resultDimensions = resultTensor.Dimensions;
            for (int i = 0; i < expectedDimensions.Length; i++)
            {
                Assert.Equal(expectedDimensions[i], resultDimensions[i]);
            }

            var resultArray = resultTensor.ToArray();
            Assert.Equal(expectedOutput.Length, resultArray.Length);
            Assert.Equal(expectedOutput, resultArray, new floatComparer());
        }


        [Fact]
        private void ThrowWrongInputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("wrong_name", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Contains("Invalid Feed Input", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongInputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            int[] inputDataInt = inputData.Select(x => (int)x).ToArray();
            var tensor = new DenseTensor<int>(inputDataInt, inputMeta["data_0"].Dimensions);
            container.Add(NamedOnnxValue.CreateFromTensor<int>("data_0", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            var msg = ex.ToString().Substring(0, 101);
            // TODO: message is diff in LInux. Use substring match
            Assert.Equal("Microsoft.ML.OnnxRuntime.OnnxRuntimeException: [ErrorCode:InvalidArgument] Unexpected input data type", msg);
            session.Dispose();
        }

        [Fact]
        private void ThrowExtraInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var nov1 = NamedOnnxValue.CreateFromTensor<float>("data_0", tensor);
            var nov2 = NamedOnnxValue.CreateFromTensor<float>("extra", tensor);
            container.Add(nov1);
            container.Add(nov2);
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.StartsWith("[ErrorCode:InvalidArgument] Invalid Feed Input Name", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowInconsistentPinnedInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;

            using (var inputs = new DisposableListTest<FixedBufferOnnxValue>())
            {
                inputs.Add(FixedBufferOnnxValue.CreateFromTensor(tensor));
                var ex = Assert.Throws<ArgumentException>(() => session.Run(new string[0], inputs));
                Assert.StartsWith("Length of inputNames (0) must match that of inputValues (1).", ex.Message);
            }
        }

        [Fact]
        private void ThrowWrongOutputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 2 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("bad_output_name", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, outputs));
            Assert.Contains("Invalid Output Name", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongOutputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<int>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, outputs));
            // TODO: check exception message
            // InferenceSession::ValidateOutputs() does not check type so far. Currently this will finally trigger an error in Softmax.
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongOutputDimension()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1001, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, outputs));
            // TODO: check exception message
            // InferenceSession::ValidateOutputs() does not check dims so far. Currently this will finally trigger an error in Softmax.
            session.Dispose();
        }

        [Fact]
        private void ThrowNoOutput()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, new NamedOnnxValue[0]));
            Assert.Contains("[ErrorCode:InvalidArgument] At least one output should be requested.", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowInconsistentPinnedOutputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });

            using (var outputs = new DisposableListTest<FixedBufferOnnxValue>())
            {
                var ex = Assert.Throws<ArgumentException>(() => session.Run(inputs, new string[] { "softmaxout_1" }, outputs));
                Assert.StartsWith("Length of outputNames (1) must match that of outputValues (0).", ex.Message);
            }
        }

        [Fact]
        private void TestMultiThreads()
        {
            var numThreads = 10;
            var loop = 10;
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var expectedOut = tuple.Item4;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
            var tasks = new Task[numThreads];
            for (int i = 0; i < numThreads; i++)
            {
                tasks[i] = Task.Factory.StartNew(() =>
                {
                    for (int j = 0; j < loop; j++)
                    {
                        var resnov = session.Run(container);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                        Assert.Equal(res, expectedOut, new floatComparer());
                    }
                });
            };
            Task.WaitAll(tasks);
            session.Dispose();
        }

        private static Dictionary<string, string> GetSkippedModels(DirectoryInfo modelsDirInfo)
        {
            var skipModels = new Dictionary<string, string>() {
                { "mxnet_arcface", "Model is an invalid ONNX model"},
                { "tf_inception_v2", "TODO: Debug failing model, skipping for now" },
                { "fp16_tiny_yolov2", "Tolerance level for float16 is not known. We now support fp16." },
                { "fp16_test_tiny_yolov2", "ImageScaler is not a registered function/op"},
                { "fp16_coreml_FNS-Candy", "ImageScaler is not a registered function/op" },
                { "fp16_coreml_LinearRegression_NYCTaxi", "Error in Node:featureVectorizer : No Op registered for FeatureVectorizer with domain_version of 1"},
                { "test_bidaf", "Does not run in opset9, runs in other opsets. The model runs but I don't have a data set to debug output locally. Tensors of type ElementType not currently supported in the LoadTensorFromFile." },
                { "test_mnist", "Does not run in opset9, runs in other opsets. The model runs but I don't have a data set to debug output locally. Tensors of type ElementType not currently supported in the LoadTensorFromFile" },
                { "BERT_Squad", "Could not find an implementation for the node bert / embeddings / one_hot:OneHot(9)" },
                { "mlperf_ssd_mobilenet_300", "Could not find file output_0.pb" },
                { "tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied" },
                { "coreml_Imputer-LogisticRegression_sklearn_load_breast_cancer", "Can't determine model file name" },
                { "mask_rcnn_keras", "Model should be edited to remove the extra outputs" },
                { "cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose" },
                { "test_strnormalizer_export_monday_casesensintive_lower", "ElementType not currently supported"},
                { "test_max_float64", "node test error"},
                { "test_min_uint8", "node test error"},
                { "test_mod_mixed_sign_float64", "node test error"},
                { "test_einsum_transpose", "node test error"},
                { "test_momentum", "node test error"},
                { "test_max_uint16", "node test error"},
                { "test_resize_downsample_scales_linear_align_corners", "node test error"},
                { "test_strnormalizer_nostopwords_nochangecase", "node test error"},
                { "test_cast_STRING_to_FLOAT", "node test error"},
                { "test_cumsum_2d_negative_axis", "node test error"},
                { "test_cast_FLOAT16_to_DOUBLE", "node test error"},
                { "test_adagrad_multiple", "node test error"},
                { "test_einsum_inner_prod", "node test error"},
                { "test_clip_default_int8_min", "node test error"},
                { "test_max_int8", "node test error"},
                { "test_sequence_insert_at_back", "node test error"},
                { "test_mod_mixed_sign_int8", "node test error"},
                { "test_maxunpool_export_with_output_shape", "node test error"},
                { "test_strnormalizer_export_monday_empty_output", "node test error"},
                { "test_strnormalizer_export_monday_insensintive_upper_twodim", "ElementType not currently supported"},
                { "test_clip_default_int8_max", "node test error"},
                { "test_einsum_sum", "node test error"},
                { "test_min_int16", "node test error"},
                { "test_cast_FLOAT_to_DOUBLE", "node test error"},
                { "test_adagrad", "node test error"},
                { "test_min_float64", "node test error"},
                { "test_max_int16", "node test error"},
                { "test_einsum_batch_diagonal", "node test error"},
                { "test_sequence_insert_at_front", "node test error"},
                { "test_cumsum_1d_exclusive", "node test error"},
                { "test_training_dropout_default", "node test error"},
                { "test_cast_BFLOAT16_to_FLOAT", "node test error"},
                { "test_training_dropout", "node test error"},
                { "test_adam", "node test error"},
                { "test_training_dropout_mask", "node test error"},
                { "test_clip_default_int8_inbounds", "node test error"},
                { "test_eyelike_with_dtype", "node test error"},
                { "test_cumsum_1d", "node test error"},
                { "test_conv_with_autopad_same", "node test error"},
                { "test_cumsum_1d_reverse_exclusive", "node test error"},
                { "test_cast_FLOAT_to_BFLOAT16", "node test error"},
                { "test_bitshift_right_uint16", "node test error"},
                { "test_bitshift_left_uint16", "node test error"},
                { "test_pow_types_float32_uint64", "node test error"},
                { "test_cumsum_2d_axis_0", "node test error"},
                { "test_max_uint8", "node test error"},
                { "test_strnormalizer_export_monday_casesensintive_nochangecase", "ElementType not currently supported"},
                { "test_momentum_multiple", "node test error"},
                { "test_cumsum_1d_reverse", "node test error"},
                { "test_pow_types_float32_uint32", "node test error"},
                { "test_if_seq", "node test error"},
                { "test_resize_downsample_scales_cubic_align_corners", "node test error"},
                { "test_einsum_batch_matmul", "node test error"},
                { "test_nesterov_momentum", "node test error"},
                { "test_cumsum_2d_axis_1", "node test error"},
                { "test_strnormalizer_export_monday_casesensintive_upper", "node test error"},
                { "test_min_uint16", "node test error"},
                { "test_adam_multiple", "node test error"},
                { "test_loop13_seq", "node test error"},
                { "test_convtranspose_autopad_same", "node test error"},
                { "test_training_dropout_default_mask", "node test error"},
                { "test_min_int8", "node test error"},
                { "test_cast_FLOAT_to_STRING", "node test error"},
                { "test_identity_sequence", "data type not supported"},
                { "test_gru_batchwise", "batchwise operations not supported"},
                { "test_lstm_batchwise", "batchwise operations not supported"},
                { "test_simple_rnn_batchwise", "batchwise operations not supported"},
                { "test_sub_uint8", "data type not supported"},
                { "test_mul_uint8", "data type not supported"},
                { "test_add_uint8", "data type not supported"},
                { "test_div_uint8", "data type not supported"},
                { "test_batchnorm_epsilon", "opset14 version not implemented yet"},
                { "test_batchnorm_epsilon_training_mode", "opset14 version not implemented yet"},
                { "test_batchnorm_example", "opset14 version not implemented yet"},
                { "test_batchnorm_example_training_mode", "opset14 version not implemented yet"},
                { "test_bernoulli", "random generator"},
                { "test_bernoulli_seed", "random generator"},
                { "test_bernoulli_double", "random generator"},
            };

            // The following models fails on nocontribops win CI
            var disableContribOpsEnvVar = Environment.GetEnvironmentVariable("DisableContribOps");
            var isContribOpsDisabled = (disableContribOpsEnvVar != null) ? disableContribOpsEnvVar.Equals("ON") : false;
            if (isContribOpsDisabled)
            {
                skipModels["test_tiny_yolov2"] = "Fails when ContribOps is disabled";
                skipModels["mask_rcnn_keras"] = "Pad is not a registered function/op";
            }

            // Skip traditional ML models
            var disableMlOpsEnvVar = Environment.GetEnvironmentVariable("DisableMlOps");
            var isMlOpsDisabled = (disableMlOpsEnvVar != null) ? disableMlOpsEnvVar.Equals("ON") : false;
            if (isMlOpsDisabled)
            {
                foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
                {
                    foreach (var modelDir in opsetDir.EnumerateDirectories())
                    {
                        var modelDirName = modelDir.Name;
                        if (modelDirName.StartsWith("scikit_") ||
                        modelDirName.StartsWith("libsvm_") ||
                        modelDirName.StartsWith("coreml_") ||
                        modelDirName.StartsWith("keras2coreml_") ||
                        modelDirName.StartsWith("XGBoost_"))
                        {
                            skipModels[modelDirName] = "Fails when ML ops are disabled";
                        }
                    } //model
                } //opset
            }

            // This model fails on x86 Win CI
            if (System.Environment.Is64BitProcess == false)
            {
                skipModels["test_vgg19"] = "Get preallocated buffer for initializer conv4_4_b_0 failed";
                skipModels["GPT2_LM_HEAD"] = "System out of memory";
                skipModels["GPT2"] = "System out of memory";
                skipModels["test_GPT2"] = "System out of memory";
                skipModels["tf_pnasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_5/comb_iter_1/left/bn_sep_7x7_1/beta:0_203 failed";
                skipModels["tf_nasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_11/beginning_bn/beta:0_331 failed";
                skipModels["test_zfnet512"] = "System out of memory";
                skipModels["test_bvlc_reference_caffenet"] = "System out of memory";
                skipModels["coreml_VGG16_ImageNet"] = "System out of memory";
                skipModels["test_ssd"] = "System out of memory";
                skipModels["roberta_sequence_classification"] = "System out of memory";
            }

            return skipModels;
        }

        public static IEnumerable<object[]> GetModelsForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                //var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in opsetDir.EnumerateDirectories())
                {
                    if (!skipModels.ContainsKey(modelDir.Name))
                    {
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }
                } //model
            } //opset
        }

        public static IEnumerable<object[]> GetSkippedModelForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in modelRoot.EnumerateDirectories())
                {
                    if (skipModels.ContainsKey(modelDir.Name))
                    {
                        //Console.WriteLine("Model {0} is skipped due to the error: {1}", modelDir.FullName, skipModels[modelDir.Name]);
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }

                }
            }
        }

        [Theory]
        [MemberData(nameof(GetModelsForTest))]
        [MemberData(nameof(GetSkippedModelForTest), Skip = "Skipped due to Error, please fix the error and enable the test")]
        private void TestPreTrainedModels(string opset, string modelName)
        {
            var modelsDir = GetTestModelsDir();
            string onnxModelFileName = null;

            var modelDir = new DirectoryInfo(Path.Combine(modelsDir, opset, modelName));

            try
            {
                var onnxModelNames = modelDir.GetFiles("*.onnx");
                bool validModelFound = false;
                if (onnxModelNames.Length > 0)
                {
                    // TODO remove file "._resnet34v2.onnx" from test set
                    for (int i = 0; i < onnxModelNames.Length; i++)
                    {
                        if (onnxModelNames[i].Name != "._resnet34v2.onnx")
                        {
                            onnxModelNames[0] = onnxModelNames[i];
                            validModelFound = true;
                        }
                    }
                }

                if (validModelFound)
                {
                    onnxModelFileName = Path.Combine(modelDir.FullName, onnxModelNames[0].Name);
                }
                else
                {
                    var modelNamesList = string.Join(",", onnxModelNames.Select(x => x.ToString()));
                    throw new Exception($"Opset {opset} Model {modelName}. Can't determine model file name. Found these :{modelNamesList}");
                }

                using (var session = new InferenceSession(onnxModelFileName))
                {
                    var inMeta = session.InputMetadata;
                    string testDataDirNamePattern = "test_data*";
                    if (opset == "opset9" && modelName == "LSTM_Seq_lens_unpacked")
                    {
                        testDataDirNamePattern = "seq_lens*"; // discrepancy in data directory
                    }
                    foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
                    {
                        var inputContainer = new List<NamedOnnxValue>();
                        var outputContainer = new List<NamedOnnxValue>();
                        foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                        {
                            inputContainer.Add(LoadTensorFromFilePb(f.FullName, inMeta));
                        }
                        foreach (var f in testDataDir.EnumerateFiles("output_*.pb"))
                        {
                            outputContainer.Add(LoadTensorFromFilePb(f.FullName, session.OutputMetadata));
                        }

                        using (var resultCollection = session.Run(inputContainer))
                        {
                            foreach (var result in resultCollection)
                            {
                                Assert.True(session.OutputMetadata.ContainsKey(result.Name));
                                var outputMeta = session.OutputMetadata[result.Name];
                                NamedOnnxValue outputValue = null;
                                foreach (var o in outputContainer)
                                {
                                    if (o.Name == result.Name)
                                    {
                                        outputValue = o;
                                        break;
                                    }
                                }
                                if (outputValue == null)
                                {
                                    outputValue = outputContainer.First(); // in case the output data file does not contain the name
                                }
                                if (outputMeta.IsTensor)
                                {
                                    if (outputMeta.ElementType == typeof(float))
                                    {
                                        Assert.Equal(result.AsTensor<float>(), outputValue.AsTensor<float>(), new floatComparer());
                                    }
                                    else if (outputMeta.ElementType == typeof(int))
                                    {
                                        Assert.Equal(result.AsTensor<int>(), outputValue.AsTensor<int>(), new ExactComparer<int>());
                                    }
                                    else if (outputMeta.ElementType == typeof(uint))
                                    {
                                        Assert.Equal(result.AsTensor<uint>(), outputValue.AsTensor<uint>(), new ExactComparer<uint>());
                                    }
                                    else if (outputMeta.ElementType == typeof(short))
                                    {
                                        Assert.Equal(result.AsTensor<short>(), outputValue.AsTensor<short>(), new ExactComparer<short>());
                                    }
                                    else if (outputMeta.ElementType == typeof(ushort))
                                    {
                                        Assert.Equal(result.AsTensor<ushort>(), outputValue.AsTensor<ushort>(), new ExactComparer<ushort>());
                                    }
                                    else if (outputMeta.ElementType == typeof(long))
                                    {
                                        Assert.Equal(result.AsTensor<long>(), outputValue.AsTensor<long>(), new ExactComparer<long>());
                                    }
                                    else if (outputMeta.ElementType == typeof(ulong))
                                    {
                                        Assert.Equal(result.AsTensor<ulong>(), outputValue.AsTensor<ulong>(), new ExactComparer<ulong>());
                                    }
                                    else if (outputMeta.ElementType == typeof(byte))
                                    {
                                        Assert.Equal(result.AsTensor<byte>(), outputValue.AsTensor<byte>(), new ExactComparer<byte>());
                                    }
                                    else if (outputMeta.ElementType == typeof(bool))
                                    {
                                        Assert.Equal(result.AsTensor<bool>(), outputValue.AsTensor<bool>(), new ExactComparer<bool>());
                                    }
                                    else if (outputMeta.ElementType == typeof(Float16))
                                    {
                                        Assert.Equal(result.AsTensor<Float16>(), outputValue.AsTensor<Float16>(), new Float16Comparer { tolerance = 2 });
                                    }
                                    else if (outputMeta.ElementType == typeof(BFloat16))
                                    {
                                        Assert.Equal(result.AsTensor<BFloat16>(), outputValue.AsTensor<BFloat16>(), new BFloat16Comparer { tolerance = 2 });
                                    }
                                    else
                                    {
                                        Assert.True(false, "The TestPretrainedModels does not yet support output of type " + nameof(outputMeta.ElementType));
                                    }
                                }
                                else
                                {
                                    Assert.True(false, "TestPretrainedModel cannot handle non-tensor outputs yet");
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                var msg = $"Opset {opset}, Model {modelName}: ModelFile = {onnxModelFileName} error = {ex.Message}";
                if (ex.Message.Contains("ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions"))
                {
                    // If the exception is thrown because the opset version of the test model is
                    // not supported by ONNXRuntime yet, then ignore the test and proceed.
                    // ORT allows commits from ONNX master and in such cases we do come across new opsets which are
                    // not supported in ORT yet. In order to force these tests to run set env var ALLOW_RELEASED_ONNX_OPSET_ONLY=0
                    output.WriteLine("Skipping the model test as the latest ONNX opset is not supported yet. Error Message: " + msg);
                }
                else
                {
                    throw new Exception(msg + "\n" + ex.StackTrace);
                }
            }
        }

        [Fact]
        private void TestOverridableInitializerMetadata()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "overridable_initializer.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.Equal(2, session.InputMetadata.Count);
                Assert.True(session.InputMetadata.ContainsKey("Label"));
                Assert.True(session.InputMetadata.ContainsKey("F2"));

                Assert.Equal(1, session.OverridableInitializerMetadata.Count);
                Assert.True(session.OverridableInitializerMetadata.ContainsKey("F1"));
                Assert.True(session.OverridableInitializerMetadata["F1"].IsTensor);
                Assert.Equal(typeof(float), session.OverridableInitializerMetadata["F1"].ElementType);
                Assert.Equal(2, session.OverridableInitializerMetadata["F1"].Dimensions.Length);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[0]);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[1]);

                var container = new List<NamedOnnxValue>();
                var Label_input = new DenseTensor<bool>(new bool[] { true }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("Label", Label_input));

                var F2_input = new DenseTensor<string>(new string[] { "f2_string" }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F2", F2_input));

                var F1_initializer = new DenseTensor<float>(new float[] { 2.0f }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F1", F1_initializer));

                using (var result = session.Run(container))
                {
                    var resultMap = new Dictionary<string, NamedOnnxValue>();

                    foreach (var output in result)
                    {
                        resultMap[output.Name] = output;
                    }

                    Assert.True(resultMap.ContainsKey("Label0"));
                    Assert.True(resultMap.ContainsKey("F20"));
                    Assert.True(resultMap.ContainsKey("F11"));

                    var overriddenInitializer = resultMap["F11"].AsTensor<float>();
                    Assert.NotNull(overriddenInitializer);
                    Assert.True(overriddenInitializer.SequenceEqual(F1_initializer));
                }
            }
        }

        // Hint: .NET Core 3.1 has a 'NativeLibrary' class that can be used to free the library handle
        private void UnloadLibrary(IntPtr libraryHandle)
        {
            if (libraryHandle != IntPtr.Zero)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    if (!FreeLibrary(libraryHandle))
                    {
                        throw new Exception("Could not unload the provided shared library using its handle");
                    }
                }

                else
                {
                    // TODO: Deal with non-Windows platforms for the .NET Core use-case
                }
            }
        }

        [SkipNonPackageTests]
        private void TestRegisterCustomOpLibrary()
        {
            using (var option = new SessionOptions())
            {
                string libName = "custom_op_library.dll";
                string modelPath = "custom_op_test.onnx";
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    libName = "custom_op_library.dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    libName = "libcustom_op_library.so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    libName = "libcustom_op_library.dylib";
                }

                string libFullPath = Path.Combine(Directory.GetCurrentDirectory(), libName);
                Assert.True(File.Exists(libFullPath), $"Expected lib {libFullPath} does not exist.");

                IntPtr libraryHandle = IntPtr.Zero;
                try
                {

                    option.RegisterCustomOpLibraryV2(libFullPath, out libraryHandle);
                }
                catch (Exception ex)
                {
                    var msg = $"Failed to load custom op library {libFullPath}, error = {ex.Message}";
                    throw new Exception(msg + "\n" + ex.StackTrace);
                }


                using (var session = new InferenceSession(modelPath, option))
                {
                    var inputContainer = new List<NamedOnnxValue>();
                    inputContainer.Add(NamedOnnxValue.CreateFromTensor<float>("input_1",
                        new DenseTensor<float>(
                            new float[]
                            {
                                1.1f,   2.2f,   3.3f,   4.4f,   5.5f,
                                6.6f,   7.7f,   8.8f,   9.9f,   10.0f,
                                11.1f,  12.2f,  13.3f,  14.4f,  15.5f
                            },
                            new int[] { 3, 5 }
                            )));

                    inputContainer.Add(NamedOnnxValue.CreateFromTensor<float>("input_2",
                        new DenseTensor<float>(
                            new float[]
                            {
                                15.5f,   14.4f,   13.3f,   12.2f,   11.1f,
                                10.0f,   9.9f,    8.8f,    7.7f,    6.6f,
                                5.5f,    4.4f,    3.3f,    2.2f,    1.1f
                            },
                            new int[] { 3, 5 }
                            )));

                    using (var result = session.Run(inputContainer))
                    {
                        Assert.Equal("output", result.First().Name);
                        var tensorOut = result.First().AsTensor<int>();

                        var expectedOut = new DenseTensor<int>(
                            new int[]
                            {
                                17, 17, 17, 17, 17,
                                17, 18, 18, 18, 17,
                                17, 17, 17, 17, 17
                            },
                            new int[] { 3, 5 }
                            );
                        Assert.True(tensorOut.SequenceEqual(expectedOut));
                    }
                }

                // Safe to unload the custom op shared library now
                UnloadLibrary(libraryHandle);
            }
        }

        [Fact]
        private void TestSymbolicDimsMetadata()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "capi_symbolic_dims.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;

                Assert.Equal(2, inputs.Count);
                Assert.Equal(1, session.OutputMetadata.Count);
                Assert.True(inputs.ContainsKey("A"));
                Assert.True(inputs.ContainsKey("B"));
                Assert.True(outputs.ContainsKey("C"));

                var inputA = inputs["A"];
                var inputB = inputs["B"];
                var outputC = outputs["C"];

                // dimension values and any symbolic dimension info should have the same length
                Assert.Equal(inputA.Dimensions.Length, inputA.SymbolicDimensions.Length);
                Assert.Equal(inputB.Dimensions.Length, inputB.SymbolicDimensions.Length);
                Assert.Equal(outputC.Dimensions.Length, outputC.SymbolicDimensions.Length);

                Assert.Equal(inputA.Dimensions, new int[] { -1, 2 });
                Assert.Equal(inputA.SymbolicDimensions, new string[] { "n", "" });
                Assert.Equal(inputB.Dimensions, new int[] { -1 });
                Assert.Equal(inputB.SymbolicDimensions, new string[] { "m" });
                Assert.Equal(outputC.Dimensions, new int[] { -1 });
                Assert.Equal(outputC.SymbolicDimensions, new string[] { "" }); // unnamed symbolic dim
            }
        }


        [Fact]
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BOOL.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<bool>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestReusingRunOutputNonStringType()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BOOL.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                var res1 = session.Run(container);

                // change the name of the DisposableNamedOnnxValue
                res1.First().Name = "input";

                // Run inferencing 2 times using the output of the first Run()
                for (int i = 0; i < 2; ++i)
                {
                    using (var res2 = session.Run(res1))
                    {
                        var tensorOut = res2.First().AsTensor<bool>();
                        Assert.True(tensorOut.SequenceEqual(tensorIn));
                    }
                }
            }
        }

        [Fact]
        private void TestReusingRunOutputStringType()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] { "a", "b", "c", "d", "e" }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                var res1 = session.Run(container);

                // change the name of the DisposableNamedOnnxValue
                res1.First().Name = "input";

                // Run inferencing 2 times using the output of the first Run()
                for (int i = 0; i < 2; ++i)
                {
                    using (var res2 = session.Run(res1))
                    {
                        var tensorOut = res2.First().AsTensor<string>();
                        Assert.True(tensorOut.SequenceEqual(tensorIn));
                    }
                }
            }
        }

        [Fact]
        public void TestCreateFixedBufferOnnxValueFromStringTensor()
        {
            var tensor = new DenseTensor<string>(new string[] { "a", "b" }, new int[] { 1, 2 });
            using (var value = FixedBufferOnnxValue.CreateFromTensor(tensor)) { }
        }

        [Fact]
        public void TestReusingStringFixedBufferOnnxValue()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var tensorA = new DenseTensor<string>(new string[] { "a", "b", "c", "d", "e" }, new int[] { 1, 5 });
                var tensorB = new DenseTensor<string>(new string[] { "v", "w", "x", "y", "z" }, new int[] { 1, 5 });
                var tensorC = new DenseTensor<string>(new string[] { "i", "j", "k", "l", "m" }, new int[] { 1, 5 });
                var tensorD = new DenseTensor<string>(new string[] { "i", "j", "k", "l", "m" }, new int[] { 1, 5 });
                using (FixedBufferOnnxValue a = FixedBufferOnnxValue.CreateFromTensor(tensorA),
                                            b = FixedBufferOnnxValue.CreateFromTensor(tensorB),
                                            c = FixedBufferOnnxValue.CreateFromTensor(tensorC),
                                            d = FixedBufferOnnxValue.CreateFromTensor(tensorD))
                {
                    // OK to use string type FixedBufferOnnxValue only in input
                    session.Run(new[] { "input" }, new[] { a });

                    // Cannot use string type FixedBufferOnnxValue in output
                    Assert.Throws<NotSupportedException>(() =>
                    {
                        // NamedOnnxValue inputs
                        session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensorB) }, new[] { "output" }, new[] { b });
                    });
                    Assert.Throws<NotSupportedException>(() =>
                    {
                        // both FixedBufferOnnxValue for inputs and outputs
                        session.Run(new[] { "input" }, new[] { c }, new[] { "output" }, new[] { d });
                    });
                }

            }
        }

        [Fact]
        private void TestReusingFixedBufferOnnxValue()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BOOL.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var bufferA = new bool[] { true, false, true, false, true };
                var bufferB = new bool[bufferA.Length];
                var bufferC = new bool[bufferA.Length];
                var tensorA = new DenseTensor<bool>(bufferA, new int[] { 1, 5 });
                var tensorB = new DenseTensor<bool>(bufferB, new int[] { 1, 5 });
                var tensorC = new DenseTensor<bool>(bufferC, new int[] { 1, 5 });
                using (FixedBufferOnnxValue a = FixedBufferOnnxValue.CreateFromTensor(tensorA),
                                            b = FixedBufferOnnxValue.CreateFromTensor(tensorB),
                                            c = FixedBufferOnnxValue.CreateFromTensor(tensorC))
                {
                    session.Run(new[] { "input" }, new[] { a }, new[] { "output" }, new[] { b });
                    session.Run(new[] { "input" }, new[] { b }, new[] { "output" }, new[] { c });
                }

                Assert.True(tensorC.SequenceEqual(tensorA));
            }
        }

        [Fact]
        private void TestReusingFixedBufferOnnxValueMultiInferences()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var bufferInput = new int[5];
                var bufferOutput = new int[5];
                var tensorInput = new DenseTensor<int>(bufferInput, new int[] { 1, 5 });
                var tensorOutput = new DenseTensor<int>(bufferOutput, new int[] { 1, 5 });

                using (FixedBufferOnnxValue valueInput = FixedBufferOnnxValue.CreateFromTensor(tensorInput),
                                            valueOutput = FixedBufferOnnxValue.CreateFromTensor(tensorOutput))
                {
                    var inputNames = new[] { "input" };
                    var outputNames = new[] { "output" };
                    var inputValues = new[] { valueInput };
                    var outputValues = new[] { valueOutput };

                    var rand = new Random();

                    // run the model for multiple times
                    for (var i = 0; i < 1000; i++)
                    {
                        // feed inputs ( 5 random integers )
                        var inputs = Enumerable.Range(0, 5).Select(x => rand.Next()).ToArray();
                        inputs.CopyTo(bufferInput, 0);

                        // run inference
                        session.Run(inputNames, inputValues, outputNames, outputValues);

                        // use outputs
                        Assert.Equal(inputs, bufferOutput);
                    }
                }
            }
        }

        [Fact]
        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<int>(new int[] { 1, -2, -3, int.MinValue, int.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<int>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_DOUBLE.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, 5, 5 }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<double>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }

        }

        [Fact]
        private void TestModelInputSTRING()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] {
                 "hello",
                 "École élémentaire",
                 "mit freundlichen grüßen",
                 "Понедельник",
                 "最好的问候,"+
                 "नमस्ते," +
                 "こんにちは," +
                 "안녕하세요"
                }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<string>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputSTRING_ShouldFailWithNullInput()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[5], // null
                                                       new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                Assert.Throws<ArgumentNullException>(() => { session.Run(container); });
            }
        }

        [Fact]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<sbyte>(new sbyte[] { 1, 2, -3, sbyte.MinValue, sbyte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<sbyte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<byte>(new byte[] { 1, 2, 3, byte.MinValue, byte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<byte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt16>(new UInt16[] { 1, 2, 3, UInt16.MinValue, UInt16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int16>(new Int16[] { 1, 2, 3, Int16.MinValue, Int16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int64>(new Int64[] { 1, 2, -3, Int64.MinValue, Int64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt32>(new UInt32[] { 1, 2, 3, UInt32.MinValue, UInt32.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt32>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }
        [Fact]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt64>(new UInt64[] { 1, 2, 3, UInt64.MinValue, UInt64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT16.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Float16>(
                    new Float16[] { 15360, 16384, 16896, 17408, 17664 }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var valueOut = res.First();
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, valueOut.ValueType);
                    Assert.Equal(Tensors.TensorElementType.Float16, valueOut.ElementType);
                    var tensorOut = res.First().AsTensor<Float16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputBFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BFLOAT16.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<BFloat16>(
                    new BFloat16[] { 16256, 16384, 16448, 16512, 16544 }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var valueOut = res.First();
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, valueOut.ValueType);
                    Assert.Equal(Tensors.TensorElementType.BFloat16, valueOut.ElementType);
                    var tensorOut = res.First().AsTensor<BFloat16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        private class IgnoreWhenMlOpsDisabledFact : FactAttribute
        {
            public IgnoreWhenMlOpsDisabledFact()
            {
                var disableMlOpsEnvVar = Environment.GetEnvironmentVariable("DisableMlOps");
                var isMlOpsDisabled = (disableMlOpsEnvVar != null) ? disableMlOpsEnvVar.Equals("ON") : false;
                if (isMlOpsDisabled)
                {
                    Skip = "Skipping this test since Ml Ops are disabled.";
                }
            }
        }

        [IgnoreWhenMlOpsDisabledFact]
        private void TestModelSequenceOfMapIntFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_int_float.pb");
            using (var session = new InferenceSession(modelPath))
            {

                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode0 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode0.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outNode0.ValueType);
                    Assert.Equal(Tensors.TensorElementType.Int64, outNode0.ElementType);

                    // try-cast as a tensor
                    var outLabelTensor = outNode0.AsTensor<long>();
                    Assert.NotNull(outLabelTensor);

                    // Label 1 should have highest probability
                    Assert.Equal(1, outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode1 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode1.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode1.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode1.AsEnumerable<NamedOnnxValue>();
                    Assert.NotNull(seq);
                    // Try-cast into DisposableNov so we can control and check the process


                    // try-cast first element in sequence to map/dictionary type
                    if (System.Environment.Is64BitProcess)
                    {
                        var map = seq.First().AsDictionary<Int64, float>();
                        Assert.NotNull(map);
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                    else // 32-bit
                    {
                        var map = seq.First().AsDictionary<long, float>();
                        Assert.NotNull(map);
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                }
            }
        }

        [IgnoreWhenMlOpsDisabledFact]
        private void TestModelSequenceOfMapStringFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_string_float.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode0 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode0.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outNode0.ValueType);
                    Assert.Equal(TensorElementType.String, (TensorElementType)outNode0.ElementType);

                    // try-cast as a tensor
                    var outLabelTensor = outNode0.AsTensor<string>();
                    Assert.NotNull(outLabelTensor);

                    // Label 1 should have highest probability
                    Assert.Equal("1", outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode1 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode1.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode1.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode1.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    var map = seq.First().AsDictionary<string, float>();
                    Assert.NotNull(map);
                    //verify values are valid
                    Assert.Equal(0.25938290, map["0"], 6);
                    Assert.Equal(0.40904793, map["1"], 6);
                    Assert.Equal(0.33156919, map["2"], 6);
                }

            }
        }

        [Fact]
        private void TestModelSequenceOfTensors()
        {

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_tensors.onnx");

            using (var session = new InferenceSession(modelPath))
            {
                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["output_sequence"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var firstInputTensor = new DenseTensor<Int64>(new Int64[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
                var secondInputTensor = new DenseTensor<Int64>(new Int64[] { 7, 8, 9, 10, 11, 12 }, new int[] { 2, 3 });

                var firstNov = NamedOnnxValue.CreateFromTensor("tensor1", firstInputTensor);
                var secondNov = NamedOnnxValue.CreateFromTensor("tensor2", secondInputTensor);

                container.Add(firstNov);
                container.Add(secondNov);

                using (var outputs = session.Run(container))
                {
                    // output is a sequence<tensors>
                    // try-cast to an sequence of NOV
                    var outNode = outputs.ElementAtOrDefault(0);
                    Assert.Equal("output_sequence", outNode.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode.AsEnumerable<NamedOnnxValue>();

                    // make sure that the sequence holds only 2 elements (tensors)
                    Assert.True(seq.Count() == 2);

                    // try-cast the elements in sequence to tensor type
                    var firstTensorInOuputSequence = seq.First().AsTensor<Int64>();
                    var secondTensorInOuputSequence = seq.Last().AsTensor<Int64>();
                    Assert.NotNull(firstTensorInOuputSequence);
                    Assert.NotNull(secondTensorInOuputSequence);

                    // make sure the tensors in the output sequence hold the correct values
                    Assert.True(firstTensorInOuputSequence.GetValue(0) == 1);
                    Assert.True(firstTensorInOuputSequence.GetValue(1) == 2);
                    Assert.True(firstTensorInOuputSequence.GetValue(2) == 3);
                    Assert.True(firstTensorInOuputSequence.GetValue(3) == 4);
                    Assert.True(firstTensorInOuputSequence.GetValue(4) == 5);
                    Assert.True(firstTensorInOuputSequence.GetValue(5) == 6);

                    Assert.True(secondTensorInOuputSequence.GetValue(0) == 7);
                    Assert.True(secondTensorInOuputSequence.GetValue(1) == 8);
                    Assert.True(secondTensorInOuputSequence.GetValue(2) == 9);
                    Assert.True(secondTensorInOuputSequence.GetValue(3) == 10);
                    Assert.True(secondTensorInOuputSequence.GetValue(4) == 11);
                    Assert.True(secondTensorInOuputSequence.GetValue(5) == 12);


                }
            }
        }

        [Fact]
        private void TestModelMetadata()
        {

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "model_with_valid_ort_config_json.onnx");

            using (var session = new InferenceSession(modelPath))
            {
                var modelMetadata = session.ModelMetadata;

                Assert.Equal(1, modelMetadata.Version);

                Assert.Equal("Hari", modelMetadata.ProducerName);

                Assert.Equal("matmul test", modelMetadata.GraphName);

                Assert.Equal("", modelMetadata.Domain);

                Assert.Equal("This is a test model with a valid ORT config Json", modelMetadata.Description);

                Assert.Equal("graph description", modelMetadata.GraphDescription);

                Assert.Equal(2, modelMetadata.CustomMetadataMap.Keys.Count);
                Assert.Equal("dummy_value", modelMetadata.CustomMetadataMap["dummy_key"]);
                Assert.Equal("{\"session_options\": {\"inter_op_num_threads\": 5, \"intra_op_num_threads\": 2, \"graph_optimization_level\": 99, \"enable_profiling\": 1}}",
                              modelMetadata.CustomMetadataMap["ort_config"]);



            }
        }

        [Fact]
        private void TestModelSerialization()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string modelOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "optimized-squeezenet.onnx");
            // Set the optimized model file path to assert that no exception are thrown.
            using (SessionOptions options = new SessionOptions())
            {
                options.OptimizedModelFilePath = modelOutputPath;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
                using (var session = new InferenceSession(modelPath, options))
                {
                    Assert.NotNull(session);
                    Assert.True(File.Exists(modelOutputPath));
                }
            }
        }

        // TestGpu() will test the CUDA EP on CUDA enabled builds and
        // the DML EP on DML enabled builds
        [GpuFact]
        private void TestGpu()
        {
            var tuple = OpenSessionSqueezeNet(0); // run on deviceID 0
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");

            using (var session = tuple.Item1)
            {
                var inputData = tuple.Item2;
                var tensor = tuple.Item3;
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
                var res = session.Run(container);
                var resultArray = res.First().AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput, resultArray, new floatComparer());
            }
        }

        [Fact]
        private void TestInferenceSessionWithByteArray()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT.pb");
            byte[] modelData = File.ReadAllBytes(modelPath);

            using (var session = new InferenceSession(modelData))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        void TestCPUAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_cpu = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Cpu", info_cpu.Name);
                Assert.Equal(device_id, info_cpu.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_cpu.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_cpu.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_cpu))
                {
                    var alloc_info = allocator.Info;
                    // Allocator type returned may be different on x86 so we don't compare.
                    Assert.Equal(info_cpu.Name, alloc_info.Name);
                    Assert.Equal(info_cpu.GetMemoryType(), alloc_info.GetMemoryType());
                    Assert.Equal(info_cpu.Id, alloc_info.Id);

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    var chunk_info = chunk.Info;
                    // Allocator type returned may be different on x86 so we don't compare.
                    Assert.Equal(chunk_info.Name, alloc_info.Name);
                    Assert.Equal(chunk_info.GetMemoryType(), alloc_info.GetMemoryType());
                    Assert.Equal(chunk_info.Id, alloc_info.Id);
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }

#if USE_CUDA
        void TestCUDAAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_cuda = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Cuda", info_cuda.Name);
                Assert.Equal(device_id, info_cuda.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_cuda.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_cuda.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_cuda))
                {
                    var alloc_info = allocator.Info;
                    Assert.True(info_cuda.Equals(alloc_info));

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    Assert.True(chunk.Info.Equals(alloc_info));
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }
#endif

#if USE_ROCM
        void TestROCMAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_rocm = new OrtMemoryInfo(OrtMemoryInfo.allocatorROCM, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Rocm", info_rocm.Name);
                Assert.Equal(device_id, info_rocm.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_rocm.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_rocm.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_rocm))
                {
                    var alloc_info = allocator.Info;
                    Assert.True(info_rocm.Equals(alloc_info));

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    Assert.True(chunk.Info.Equals(alloc_info));
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }
#endif


        [Fact]
        private void TestAllocator()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            using (SessionOptions options = new SessionOptions())
            {
                options.AppendExecutionProvider_CPU(1);
#if USE_CUDA
                options.AppendExecutionProvider_CUDA(0);
#endif

#if USE_ROCM
                options.AppendExecutionProvider_ROCM(0);
#endif

                using (var session = new InferenceSession(modelPath, options))
                {
                    TestCPUAllocatorInternal(session);
#if USE_CUDA
                    TestCUDAAllocatorInternal(session);
#endif
#if USE_ROCM
                    TestROCMAllocatorInternal(session);
#endif

                }
            }
        }

        [Fact]
        private void TestIOBinding()
        {
            var inputName = "data_0";
            var outputName = "softmaxout_1";
            var allocator = OrtAllocator.DefaultInstance;
            // From the model
            using (var dispList = new DisposableListTest<IDisposable>())
            {
                var tuple = OpenSessionSqueezeNet();
                var session = tuple.Item1;
                var inputData = tuple.Item2;
                var inputTensor = tuple.Item3;
                var outputData = tuple.Item4;
                dispList.Add(session);
                var runOptions = new RunOptions();
                dispList.Add(runOptions);

                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var outputTensor = new DenseTensor<float>(outputData, outputMeta[outputName].Dimensions);

                var ioBinding = session.CreateIoBinding();
                dispList.Add(ioBinding);

                var ortAllocationOutput = allocator.Allocate((uint)outputData.Length * sizeof(float));
                dispList.Add(ortAllocationOutput);

                // Test GetOutputNames, bind two output names
                {
                    var cyrName = "несуществующийВыход";
                    var longShape = Array.ConvertAll<int, long>(outputMeta[outputName].Dimensions, i => i);
                    ioBinding.BindOutput(outputName, TensorElementType.Float, longShape, ortAllocationOutput);
                    ioBinding.BindOutput(cyrName, TensorElementType.Float, longShape, ortAllocationOutput);
                    string[] outputs = ioBinding.GetOutputNames();
                    Assert.Equal(2, outputs.Length);
                    Assert.Equal(outputName, outputs[0]);
                    Assert.Equal(cyrName, outputs[1]);
                    ioBinding.ClearBoundOutputs();
                }

                // Test 1. Bind input to fixed, Bind Output to Fixed.
                using (FixedBufferOnnxValue fixeInputBuffer = FixedBufferOnnxValue.CreateFromTensor(inputTensor),
                      fixedOutputBuffer = FixedBufferOnnxValue.CreateFromTensor(outputTensor))
                {
                    ioBinding.BindInput(inputName, fixeInputBuffer);
                    ioBinding.BindOutput(outputName, fixedOutputBuffer);
                    using (var outputs = session.RunWithBindingAndNames(runOptions, ioBinding))
                    {
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.First();
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new floatComparer());
                    }
                }

                // Test 2. Bind input to preallocated buffer. Output to a device so the allocation would happen
                // automatically
                using (FixedBufferOnnxValue fixedInputBuffer = FixedBufferOnnxValue.CreateFromTensor(inputTensor))
                {
                    ioBinding.BindInput(inputName, fixedInputBuffer);
                    ioBinding.BindOutputToDevice(outputName, allocator.Info);

                    using (var outputs = session.RunWithBindingAndNames(runOptions, ioBinding))
                    {
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.First();
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new floatComparer());
                    }
                }

                // Rebinding would happen without these but we want run them.
                ioBinding.ClearBoundInputs();
                ioBinding.ClearBoundOutputs();
            }
        }

        [Fact]
        private void TestSharingOfInitializerAndItsPrepackedVersion()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "matmul_1.onnx");

            // create initializer to share
            var ortCpuMemInfo = OrtMemoryInfo.DefaultInstance;
            var dims = new long[] { 2, 1 };
            var dataBuffer = new float[] { 2.0F, 1.0F };
            var dataHandle = GCHandle.Alloc(dataBuffer, GCHandleType.Pinned);

            try
            {
                unsafe
                {
                    float* p = (float*)dataHandle.AddrOfPinnedObject();
                    for (int i = 0; i < dataBuffer.Length; ++i)
                    {
                        *p++ = dataBuffer[i];
                    }
                }
                var dataBufferNumBytes = (uint)dataBuffer.Length * sizeof(float);

                using (var sharedInitializer = OrtValue.CreateTensorValueWithData(ortCpuMemInfo, Tensors.TensorElementType.Float,
                                        dims, dataHandle.AddrOfPinnedObject(), dataBufferNumBytes))
                {

                    using (var prepackedWeightsContainer = new PrePackedWeightsContainer())
                    {
                        using (var options = new SessionOptions())
                        {
                            // We want to share initializers between the two sessions
                            options.AddInitializer("W", sharedInitializer);

                            float[] expectedOutput = { 4.0F, 10.0F, 16.0F };
                            int[] expectedDimensions = { 3, 1 };

                            // We want the pre-packed weights of the shared initializer to be shared between sessions (memory savings)
                            // and hence we pass in the 'prepackedWeightsContainer' at session creation time
                            byte[] modelData = File.ReadAllBytes(modelPath);

                            // Test both InferenceSession ctors that take PrePackedWeightsContainer instances
                            using (var session = new InferenceSession(modelPath, options, prepackedWeightsContainer))
                            using (var session2 = new InferenceSession(modelData, options, prepackedWeightsContainer))
                            {
                                var inputMeta = session.InputMetadata;
                                var container = new List<NamedOnnxValue>();

                                foreach (var name in inputMeta.Keys)
                                {
                                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                                    Assert.True(inputMeta[name].IsTensor);
                                    var tensor = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, inputMeta[name].Dimensions);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                                }

                                ReadOnlySpan<int> expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                                string[] expectedOutputNames = new string[] { "Y" };

                                // Run inference with named inputs and outputs created with in Run()
                                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                                {
                                    foreach (var r in results)
                                    {
                                        validateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
                                    }
                                }

                                // Run inference with named inputs and outputs created with in Run()
                                using (var results2 = session2.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                                {
                                    foreach (var r in results2)
                                    {
                                        validateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            finally
            {
                dataHandle.Free();
            }
        }

        [Fact]
        private void TestSharedAllocatorUsingCreateAndRegisterAllocator()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "mul_1.onnx");

            using (var memInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU,
                                                   OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default))
            using (var arenaCfg = new OrtArenaCfg(0, -1, -1, -1))
            {
                var env = OrtEnv.Instance();
                // Create and register the arena based allocator
                env.CreateAndRegisterAllocator(memInfo, arenaCfg);

                using (var sessionOptions = new SessionOptions())
                {
                    // Key must match kOrtSessionOptionsConfigUseEnvAllocators in onnxruntime_session_options_config_keys.h
                    sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");

                    // Create two sessions to share the allocator
                    // Create a thrid session that DOES NOT use the allocator in the environment
                    using (var session1 = new InferenceSession(modelPath, sessionOptions))
                    using (var session2 = new InferenceSession(modelPath, sessionOptions))
                    using (var session3 = new InferenceSession(modelPath)) // Use the default SessionOptions instance
                    {
                        // Input data
                        var inputDims = new long[] { 3, 2 };
                        var input = new float[] { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };

                        // Output data
                        int[] outputDims = { 3, 2 };
                        float[] output = { 1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F };

                        // Run inference on all three models
                        var inputMeta = session1.InputMetadata;
                        var container = new List<NamedOnnxValue>();

                        foreach (var name in inputMeta.Keys)
                        {
                            Assert.Equal(typeof(float), inputMeta[name].ElementType);
                            Assert.True(inputMeta[name].IsTensor);
                            var tensor = new DenseTensor<float>(input, inputMeta[name].Dimensions);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session1.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                validateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session2.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                validateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session3.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                validateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }
                    }
                }
            }
        }

        [DllImport("kernel32", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Ansi)]
        static extern UIntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [Fact]
        private void VerifyNativeMethodsExist()
        {
            // Check for  external API changes
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            var entryPointNames = new[]{
            "OrtGetApiBase",
            "OrtSessionOptionsAppendExecutionProvider_CPU"
#if USE_DNNL
            ,"OrtSessionOptionsAppendExecutionProvider_Dnnl"
#endif
#if USE_CUDA
            ,"OrtSessionOptionsAppendExecutionProvider_CUDA"
#endif
#if USE_ROCM
            ,"OrtSessionOptionsAppendExecutionProvider_ROCM"
#endif
#if USE_DML
            ,"OrtSessionOptionsAppendExecutionProvider_DML"
#endif
#if USE_OPENVINO
            ,"OrtSessionOptionsAppendExecutionProvider_OpenVINO"
#endif
#if USE_TENSORRT
            ,"OrtSessionOptionsAppendExecutionProvider_Tensorrt"
#endif
#if USE_MIGRAPHX
            ,"OrtSessionOptionsAppendExecutionProvider_MIGraphX"
#endif
#if USE_NNAPI
            ,"OrtSessionOptionsAppendExecutionProvider_Nnapi"
#endif
    };
            IntPtr libraryHandle = IntPtr.Zero;
            try
            {
                libraryHandle = LoadLibrary(module);
                foreach (var ep in entryPointNames)
                {
                    var x = GetProcAddress(libraryHandle, ep);
                    Assert.False(x == UIntPtr.Zero, $"Entrypoint {ep} not found in module {module}");
                }
            }

            finally
            {
                UnloadLibrary(libraryHandle);
            }
        }

        static string GetTestModelsDir()
        {
            // get build directory, append downloaded models location
            var cwd = Directory.GetCurrentDirectory();
            var props = File.ReadAllLines(Path.Combine(cwd, propertiesFile));
            var modelsRelDir = Path.Combine(props[0].Split('=')[1].Trim());
            var modelsDir = Path.Combine(cwd, @"../../..", modelsRelDir, "models");
            return modelsDir;
        }

        static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        private static void GetTypeAndWidth(Tensors.TensorElementType elemType, out Type type, out int width)
        {
            TensorElementTypeInfo result = TensorBase.GetElementTypeInfo(elemType);
            if (result != null)
            {
                type = result.TensorType;
                width = result.TypeSize;
            }
            else
            {
                type = null;
                width = 0;
            }
        }

        static NamedOnnxValue LoadTensorFromFilePb(string filename, IReadOnlyDictionary<string, NodeMetadata> nodeMetaDict)
        {
            //Set buffer size to 4MB
            int readBufferSize = 4194304;
            Onnx.TensorProto tensor = null;
            using (var file = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.Read, readBufferSize))
            {
                tensor = Onnx.TensorProto.Parser.ParseFrom(file);
            }

            Type tensorElemType = null;
            int width = 0;
            GetTypeAndWidth((Tensors.TensorElementType)tensor.DataType, out tensorElemType, out width);
            var intDims = new int[tensor.Dims.Count];
            for (int i = 0; i < tensor.Dims.Count; i++)
            {
                intDims[i] = (int)tensor.Dims[i];
            }

            NodeMetadata nodeMeta = null;
            string nodeName = string.Empty;
            if (nodeMetaDict.Count == 1)
            {
                nodeMeta = nodeMetaDict.Values.First();
                nodeName = nodeMetaDict.Keys.First(); // valid for single node input
            }
            else if (nodeMetaDict.Count > 1)
            {
                if (tensor.Name.Length > 0)
                {
                    nodeMeta = nodeMetaDict[tensor.Name];
                    nodeName = tensor.Name;
                }
                else
                {
                    bool matchfound = false;
                    // try to find from matching type and shape
                    foreach (var key in nodeMetaDict.Keys)
                    {
                        var meta = nodeMetaDict[key];
                        if (tensorElemType == meta.ElementType && tensor.Dims.Count == meta.Dimensions.Length)
                        {
                            int i = 0;
                            for (; i < meta.Dimensions.Length; i++)
                            {
                                if (meta.Dimensions[i] != -1 && meta.Dimensions[i] != intDims[i])
                                {
                                    break;
                                }
                            }
                            if (i >= meta.Dimensions.Length)
                            {
                                matchfound = true;
                                nodeMeta = meta;
                                nodeName = key;
                                break;
                            }
                        }
                    }
                    if (!matchfound)
                    {
                        // throw error
                        throw new Exception("No Matching Tensor found in InputOutputMetadata corresponding to the serliazed tensor loaded from " + filename);
                    }
                }
            }
            else
            {
                // throw error
                throw new Exception("While reading the serliazed tensor loaded from " + filename + ", metaDataDict has 0 elements");
            }

            Assert.True(nodeMeta.IsTensor, "LoadTensorFromFile can load Tensor types only");
            //TODO: support other types when models are available in Onnx model zoo/ test data

            Assert.Equal(tensorElemType, nodeMeta.ElementType);
            Assert.Equal(nodeMeta.Dimensions.Length, tensor.Dims.Count);
            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                Assert.True((nodeMeta.Dimensions[i] == -1) || (nodeMeta.Dimensions[i] == intDims[i]));
            }

            if (nodeMeta.ElementType == typeof(float))
            {
                return CreateNamedOnnxValueFromRawData<float>(nodeName, tensor.RawData.ToArray(), sizeof(float), intDims);
            }
            else if (nodeMeta.ElementType == typeof(double))
            {
                return CreateNamedOnnxValueFromRawData<double>(nodeName, tensor.RawData.ToArray(), sizeof(double), intDims);
            }
            else if (nodeMeta.ElementType == typeof(int))
            {
                return CreateNamedOnnxValueFromRawData<int>(nodeName, tensor.RawData.ToArray(), sizeof(int), intDims);
            }
            else if (nodeMeta.ElementType == typeof(uint))
            {
                return CreateNamedOnnxValueFromRawData<uint>(nodeName, tensor.RawData.ToArray(), sizeof(uint), intDims);
            }
            else if (nodeMeta.ElementType == typeof(long))
            {
                return CreateNamedOnnxValueFromRawData<long>(nodeName, tensor.RawData.ToArray(), sizeof(long), intDims);
            }
            else if (nodeMeta.ElementType == typeof(ulong))
            {
                return CreateNamedOnnxValueFromRawData<ulong>(nodeName, tensor.RawData.ToArray(), sizeof(ulong), intDims);
            }
            else if (nodeMeta.ElementType == typeof(short))
            {
                return CreateNamedOnnxValueFromRawData<short>(nodeName, tensor.RawData.ToArray(), sizeof(short), intDims);
            }
            else if (nodeMeta.ElementType == typeof(ushort))
            {
                return CreateNamedOnnxValueFromRawData<ushort>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
            }
            else if (nodeMeta.ElementType == typeof(byte))
            {
                return CreateNamedOnnxValueFromRawData<byte>(nodeName, tensor.RawData.ToArray(), sizeof(byte), intDims);
            }
            else if (nodeMeta.ElementType == typeof(bool))
            {
                return CreateNamedOnnxValueFromRawData<bool>(nodeName, tensor.RawData.ToArray(), sizeof(bool), intDims);
            }
            else if (nodeMeta.ElementType == typeof(Float16))
            {
                return CreateNamedOnnxValueFromRawData<Float16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
            }
            else if (nodeMeta.ElementType == typeof(BFloat16))
            {
                return CreateNamedOnnxValueFromRawData<BFloat16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
            }
            else
            {
                //TODO: Add support for remaining types
                Assert.True(false, "Tensors of type " + nameof(nodeMeta.ElementType) + " not currently supported in the LoadTensorFromFile");
                return null;
            }
        }

        static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, int[] dimensions)
        {
            T[] typedArr = new T[rawData.Length / elemWidth];
            var typeOf = typeof(T);
            if (typeOf == typeof(Float16) || typeOf == typeof(BFloat16))
            {
                using (var memSrcHandle = new Memory<byte>(rawData).Pin())
                using (var memDstHandle = new Memory<T>(typedArr).Pin())
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(memSrcHandle.Pointer, memDstHandle.Pointer, typedArr.Length * elemWidth, rawData.Length);
                    }
                }
            }
            else
            {
                Buffer.BlockCopy(rawData, 0, typedArr, 0, rawData.Length);
            }
            var dt = new DenseTensor<T>(typedArr, dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        internal static Tuple<InferenceSession, float[], DenseTensor<float>, float[]> OpenSessionSqueezeNet(int? deviceId = null)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
#if USE_DML
            // Explicitly set dll probe path so that the (potentially) stale system DirectML.dll
            // doesn't get loaded by the test process when it is eventually delay loaded by onnruntime.dll
            // The managed tests binary path already contains the right DirectML.dll, so use that

            var directml_dll_path = AppDomain.CurrentDomain.BaseDirectory;
            SetDllDirectory(directml_dll_path);

            using (var option = new SessionOptions())
            {
                if (!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }

                else
                {
                    option.AppendExecutionProvider_DML(deviceId.Value);
                }

                 // Restore the default dll search order
                SetDllDirectory(null);
#elif USE_CUDA
            using (var option = (deviceId.HasValue) ?
                SessionOptions.MakeSessionOptionWithCudaProvider(deviceId.Value) :
                new SessionOptions())
            {
                if(!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }
#elif USE_ROCM
            using (var option = (deviceId.HasValue) ?
                SessionOptions.MakeSessionOptionWithRocmProvider(deviceId.Value) :
                new SessionOptions())
            {
                if(!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }
#else
            using (var option = new SessionOptions())
            {
                option.AppendExecutionProvider_CPU(1);
#endif
                var session = (deviceId.HasValue)
                    ? new InferenceSession(modelPath, option)
                    : new InferenceSession(modelPath);
                float[] inputData = LoadTensorFromFile(@"bench.in");
                float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
                var inputMeta = session.InputMetadata;
                var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
                return new Tuple<InferenceSession, float[], DenseTensor<float>, float[]>(session, inputData, tensor, expectedOutput);
            }
        }

        internal class floatComparer : IEqualityComparer<float>
        {
            private float atol = 1e-3f;
            private float rtol = 1.7e-2f;

            public bool Equals(float x, float y)
            {
                return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
            }
            public int GetHashCode(float x)
            {
                return x.GetHashCode();
            }
        }

        class ExactComparer<T> : IEqualityComparer<T>
        {
            public bool Equals(T x, T y)
            {
                return x.Equals(y);
            }
            public int GetHashCode(T x)
            {
                return x.GetHashCode();
            }
        }

        /// <summary>
        /// Use it to compare Float16 and BFloat16
        /// </summary>
        internal class Float16Comparer : IEqualityComparer<Float16>
        {
            public ushort tolerance;
            public bool Equals(Float16 x, Float16 y)
            {
                return Math.Abs(x - y) <= (tolerance + y);
            }
            public int GetHashCode(Float16 x)
            {
                return x.GetHashCode();
            }
        }

        internal class BFloat16Comparer : IEqualityComparer<BFloat16>
        {
            public ushort tolerance;
            public bool Equals(BFloat16 x, BFloat16 y)
            {
                return Math.Abs(x - y) <= (tolerance + y);
            }
            public int GetHashCode(BFloat16 x)
            {
                return x.GetHashCode();
            }
        }


        private class GpuFact : FactAttribute
        {
            public GpuFact()
            {
                var testOnGpu = System.Environment.GetEnvironmentVariable("TESTONGPU");
                if (testOnGpu == null || !testOnGpu.Equals("ON"))
                {
                    Skip = "GPU testing not enabled";
                }
            }
        }

        private class SkipNonPackageTests : FactAttribute
        {
            public SkipNonPackageTests()
            {
                var skipNonPackageTests = System.Environment.GetEnvironmentVariable("SKIPNONPACKAGETESTS");
                if (skipNonPackageTests != null && skipNonPackageTests.Equals("ON"))
                {
                    Skip = "Test skipped while testing the package as it is not within the scope";
                }
            }
        }
    }

    // Copy of the class that is internal in the main package
    internal class DisposableListTest<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T : IDisposable
    {
        public DisposableListTest() { }
        public DisposableListTest(int count) : base(count) { }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // Dispose in the reverse order.
                    // Objects should typically be destroyed/disposed
                    // in the reverse order of its creation
                    // especially if the objects created later refer to the
                    // objects created earlier. For homogeneous collections of objects
                    // it would not matter.
                    for (int i = this.Count - 1; i >= 0; --i)
                    {
                        this[i]?.Dispose();
                    }
                    this.Clear();
                }

                disposedValue = true;
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
