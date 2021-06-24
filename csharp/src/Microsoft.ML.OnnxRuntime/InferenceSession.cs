// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Inference Session on an ONNX Model.
    /// This is a IDisposable class and it must be disposed of
    /// using either a explicit call to Dispose() method or
    /// a pattern of using() block. If this is a member of another
    /// class that class must also become IDisposable and it must
    /// dispose of InferfenceSession in its Dispose() method.
    /// </summary>
    public class InferenceSession : IDisposable
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtSession
        /// </summary>
        protected IntPtr _nativeHandle;
        /// <summary>
        /// Dictionaries that represent input/output/overridableInitializers metadata
        /// </summary>
        protected Dictionary<string, NodeMetadata> _inputMetadata, _outputMetadata, _overridableInitializerMetadata;
        private SessionOptions _builtInSessionOptions = null;
        private RunOptions _builtInRunOptions = null;
        private ModelMetadata _modelMetadata = null;
        private bool _disposed = false;
        private ulong _profilingStartTimeNs = 0;

        #region Public API

        /// <summary>
        /// Constructs an InferenceSession from a model file
        /// </summary>
        /// <param name="modelPath"></param>
        public InferenceSession(string modelPath)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(modelPath, _builtInSessionOptions);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model file and it will use 
        /// the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="modelPath">Model path</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(string modelPath, PrePackedWeightsContainer prepackedWeightsContainer)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(modelPath, _builtInSessionOptions, prepackedWeightsContainer);
        }


        /// <summary>
        /// Constructs an InferenceSession from a model file, with some additional session options
        /// </summary>
        /// <param name="modelPath"></param>
        /// <param name="options"></param>
        public InferenceSession(string modelPath, SessionOptions options)
        {
            Init(modelPath, options);
        }


        /// <summary>
        /// Constructs an InferenceSession from a model file, with some additional session options
        /// and it will use the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="modelPath">Model path</param>
        /// <param name="options">Session options</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(string modelPath, SessionOptions options,
            PrePackedWeightsContainer prepackedWeightsContainer)
        {
            Init(modelPath, options, prepackedWeightsContainer);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data in byte array
        /// </summary>
        /// <param name="model"></param>
        public InferenceSession(byte[] model)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(model, _builtInSessionOptions);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data (in byte array) and it will use 
        /// the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="model">Model as byte array</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(byte[] model, PrePackedWeightsContainer prepackedWeightsContainer)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(model, _builtInSessionOptions, prepackedWeightsContainer);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data in byte array, with some additional session options
        /// </summary>
        /// <param name="model"></param>
        /// <param name="options"></param>
        public InferenceSession(byte[] model, SessionOptions options)
        {
            Init(model, options);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data (in byte array) with some additional
        /// session options and it will use the provided pre-packed weights container to store
        /// and share pre-packed buffers of shared initializers across sessions if any.
        /// </summary>
        /// <param name="model">Model as byte array</param>
        /// <param name="options">Session Options</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(byte[] model, SessionOptions options,
                                PrePackedWeightsContainer prepackedWeightsContainer)
        {
            Init(model, options, prepackedWeightsContainer);
        }
        /// <summary>
        /// Meta data regarding the input nodes, keyed by input names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> InputMetadata
        {
            get
            {
                return _inputMetadata;
            }
        }

        /// <summary>
        /// Metadata regarding the output nodes, keyed by output names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata
        {
            get
            {
                return _outputMetadata;
            }
        }

        /// <summary>
        /// Metadata regarding the overridable initializers, keyed by node names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> OverridableInitializerMetadata
        {
            get
            {
                return _overridableInitializerMetadata;
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches all the outputs.
        /// </summary>
        /// <param name="inputs">specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputs, outputNames);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the outputs specified in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames)
        {
            return Run(inputs, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames, RunOptions options)
        {
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputs, v => v.Name, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, cleanupList);
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, cleanupList);

                var ortValues = RunImpl(options, inputNamesArray, inputValuesArray, outputNamesArray, cleanupList);
                return CreateDisposableResult(ortValues, outputNames);
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches all the outputs.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputNames, inputValues, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the outputs specified in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames)
        {
            return Run(inputNames, inputValues, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, cleanupList);
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, cleanupList);


                var ortValues = RunImpl(options, inputNamesArray, inputValuesArray, outputNamesArray, cleanupList);
                return CreateDisposableResult(ortValues, outputNames);
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to accept the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            Run(inputNames, inputValues, outputNames, outputValues, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to accept the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }
            if (outputNames.Count != outputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(outputNames)} ({outputNames.Count}) must match that of {nameof(outputValues)} ({outputValues.Count}).");
            }

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, cleanupList);
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, cleanupList);
                IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, false);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputNames.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputNames.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputs, outputs, _builtInRunOptions);
        }

        /// <summary>
        ///
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs,
            RunOptions options)
        {
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputs, i => i.Name, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, cleanupList);

                var outputNamesArray = ConvertNamesToUtf8(outputs, o => o.Name, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputs, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            Run(inputs, outputNames, outputValues, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues,
            RunOptions options)
        {
            if (outputNames.Count != outputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(outputNames)} ({outputNames.Count}) must match that of {nameof(outputValues)} ({outputValues.Count}).");
            }

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputs, i => i.Name, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, cleanupList);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputValues, false);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputNames.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        ///
        /// Runs the loaded model for the given inputs and outputs.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputNames, inputValues, outputs, _builtInRunOptions);
        }

        /// <summary>
        ///
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<NamedOnnxValue> outputs,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputValues, true);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputs, o => o.Name, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputs, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputNames.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        /// Create OrtIoBinding instance to bind pre-allocated buffers
        /// to input/output
        /// </summary>
        /// <returns>A new instance of OrtIoBinding</returns>
        public OrtIoBinding CreateIoBinding()
        {
            return new OrtIoBinding(this);
        }

        /// <summary>
        /// This method runs inference on the OrtIoBinding instance
        /// The method does not return anything. This is a lightweight version of 
        /// RunWithBindingAndNames(). When you bind pre-allocated buffers to the output values
        /// you may not want to fetch the outputs since you already have access to them so you can spare
        /// the expense of fetching them and pairing with names.
        /// You can still fetch the outputs by calling OrtIOBinding.GetOutputValues()
        /// </summary>
        /// <param name="runOptions">runOptions</param>
        /// <param name="ioBinding">ioBinding instance to use</param>
        public void RunWithBinding(RunOptions runOptions, OrtIoBinding ioBinding)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRunWithBinding(Handle, runOptions.Handle, ioBinding.Handle));
        }

        /// <summary>
        ///  This method return a collection of DisposableNamedOnnxValue as in other interfaces
        ///  Query names from OrtIoBinding object and pair then with the array of OrtValues returned
        /// from OrtIoBinding.GetOutputValues()
        /// 
        /// </summary>
        /// <param name="runOptions">RunOptions</param>
        /// <param name="ioBinding">OrtIoBinding instance with bindings</param>
        /// <param name="names">optional parameter. If you already know the names of the outputs you can save a native
        /// call to retrieve output names. They will be paired with the returned OrtValues and combined into DisposbleNamedOnnxValues.
        /// Otherwise, the method will retrieve output names from the OrtIoBinding instance.
        /// It is an error if you supply a different number of names than the returned outputs</param>
        /// <returns>A disposable collection of DisposableNamedOnnxValue that encapsulate output OrtValues</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunWithBindingAndNames(RunOptions runOptions, OrtIoBinding ioBinding, string[] names = null)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRunWithBinding(Handle, runOptions.Handle, ioBinding.Handle));
            using (var ortValues = ioBinding.GetOutputValues())
            {
                string[] outputNames = names;
                if (outputNames == null)
                {
                    outputNames = ioBinding.GetOutputNames();
                }

                if (outputNames.Length != ortValues.Count)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        "Number of specified names: " + names.Length + " does not match the output number: " +
                        ortValues.Count);
                }

                var result = new DisposableList<DisposableNamedOnnxValue>(outputNames.Length);
                try
                {
                    for (int i = 0; i < outputNames.Length; ++i)
                    {
                        var ortValue = ortValues.ElementAt(i);
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(outputNames[i], ortValue));
                    }
                }
                catch (Exception e)
                {
                    result.Dispose();
                    throw e;
                }
                return result;
            }
        }

        /// <summary>
        /// Ends profiling for the session.
        /// </summary>
        /// <returns> Returns the profile file name.</returns>
        public string EndProfiling()
        {
            IntPtr nameHandle = IntPtr.Zero;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionEndProfiling(_nativeHandle,
                                                                   allocator.Pointer,
                                                                   out nameHandle));
            using (var allocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
        }

        // Delegate for string extraction from an arbitrary input/output object
        private delegate string NameExtractor<in TInput>(TInput input);

        /// <summary>
        /// Run helper
        /// </summary>
        /// <param name="names">names to convert to zero terminated utf8 and pin</param>
        /// <param name="cleanupList">list to add pinned memory to for later disposal</param>
        /// <returns></returns>
        private IntPtr[] ConvertNamesToUtf8<T>(IReadOnlyCollection<T> inputs, NameExtractor<T> extractor,
            DisposableList<IDisposable> cleanupList)
        {
            var result = new IntPtr[inputs.Count];
            for (int i = 0; i < inputs.Count; ++i)
            {
                var name = extractor(inputs.ElementAt(i));
                var utf8Name = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name);
                var pinnedHandle = new PinnedGCHandle(GCHandle.Alloc(utf8Name, GCHandleType.Pinned));
                result[i] = pinnedHandle.Pointer;
                cleanupList.Add(pinnedHandle);
            }
            return result;
        }

        /// <summary>
        /// This function obtains ortValues for NamedOnnxValue.
        /// The problem with NamedOnnxValue is that it does not contain any Onnx (OrtValue)
        /// so calling ToOrtValue creates a new instance of OrtValue that needs to be disposed.
        /// The deriving object DisposableNamedValue actually contains and owns OrtValue and it returns
        /// it.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="cleanupList"></param>
        /// <returns></returns>
        private IntPtr[] GetOrtValuesHandles(IReadOnlyCollection<NamedOnnxValue> values, DisposableList<IDisposable> cleanupList)
        {
            IntPtr[] result = new IntPtr[values.Count];
            for (int inputIndex = 0; inputIndex < values.Count; ++inputIndex)
            {
                var input = values.ElementAt(inputIndex);
                MemoryHandle? memHandle;
                var ortValue = input.ToOrtValue(out memHandle);
                if (memHandle.HasValue)
                {
                    cleanupList.Add(memHandle);
                }
                cleanupList.Add(ortValue);
                result[inputIndex] = ortValue.Handle;
            }
            return result;
        }

        private IntPtr[] GetOrtValuesHandles(IReadOnlyCollection<FixedBufferOnnxValue> values, bool input)
        {
            var valuesArray = new IntPtr[values.Count];
            for (int index = 0; index < values.Count; ++index)
            {
                var v = values.ElementAt(index);
                if (!input && v.ElementType == Tensors.TensorElementType.String)
                {
                    throw new NotSupportedException("Using string type FixedBufferOnnxValue in outputs is not supported.");
                }
                valuesArray[index] = v.Value.Handle;
            }
            return valuesArray;
        }


        private DisposableList<OrtValue> RunImpl(RunOptions options, IntPtr[] inputNames, IntPtr[] inputValues, IntPtr[] outputNames,
           DisposableList<IDisposable> cleanupList)
        {
            var ortValues = new DisposableList<OrtValue>(outputNames.Length);
            cleanupList.Add(ortValues);

            IntPtr[] outputValuesArray = new IntPtr[outputNames.Length];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                _nativeHandle,
                                                options.Handle,
                                                inputNames,
                                                inputValues,
                                                (UIntPtr)inputNames.Length,
                                                outputNames,
                                                (UIntPtr)outputNames.Length,
                                                outputValuesArray /* Empty array is passed in to receive output OrtValue pointers */
                                                ));

            foreach (var v in outputValuesArray)
            {
                ortValues.Add(new OrtValue(v));
            }
            return ortValues;
        }

        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> CreateDisposableResult(List<OrtValue> ortValues,
            IReadOnlyCollection<string> outputNames)
        {
            var result = new DisposableList<DisposableNamedOnnxValue>(outputNames.Count);
            try
            {
                for (int i = 0; i < ortValues.Count; i++)
                {
                    var ortValue = ortValues[i];
                    result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(outputNames.ElementAt(i), ortValue));
                }
            }
            catch (OnnxRuntimeException e)
            {
                result.Dispose();
                throw e;
            }
            return result;
        }

        /// <summary>
        /// This property queries model metadata, constructs
        /// an instance of ModelMetadata and caches it
        /// </summary>
        /// <returns>Instance of ModelMetdata</returns>
        public ModelMetadata ModelMetadata
        {
            get
            {
                if (_modelMetadata != null)
                {
                    return _modelMetadata;
                }

                _modelMetadata = new ModelMetadata(this);
                return _modelMetadata;
            }
        }

        /// <summary>
        /// Return the nanoseconds of profiling's start time
        /// On some platforms, this timer may not be as precise as nanoseconds
        /// For instance, on Windows and MacOS, the precision will be ~100ns
        /// </summary>
        public ulong ProfilingStartTimeNs
        {
            get
            {
                return _profilingStartTimeNs;
            }
        }

        #endregion

        #region private methods

        private void Init(string modelPath, SessionOptions options,
                          PrePackedWeightsContainer prepackedWeightsContainer = null)
        {
            var envHandle = OrtEnv.Handle;
            var session = IntPtr.Zero;
            // Register ONNX opset schema
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionAddONNXOpDomain(options.SessionOnnxOpsetVersion));
            if (prepackedWeightsContainer == null)
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSession(envHandle, NativeMethods.GetPlatformSerializedString(modelPath),
                    options.Handle, out session));
            }

            else
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionWithPrepackedWeightsContainer(
                    envHandle, NativeMethods.GetPlatformSerializedString(modelPath),
                    options.Handle, prepackedWeightsContainer.Pointer, out session));
            }

            InitWithSessionHandle(session, options);
        }

        private void Init(byte[] modelData, SessionOptions options,
                          PrePackedWeightsContainer prepackedWeightsContainer = null)
        {
            var envHandle = OrtEnv.Handle;
            var session = IntPtr.Zero;
            // Register ONNX opset schema
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionAddONNXOpDomain(options.SessionOnnxOpsetVersion));
            if (prepackedWeightsContainer == null)
            {

                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionFromArray(envHandle, modelData, (UIntPtr)modelData.Length, options.Handle, out session));
            }

            else
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionFromArrayWithPrepackedWeightsContainer(
                    envHandle, modelData, (UIntPtr)modelData.Length, options.Handle, prepackedWeightsContainer.Pointer,
                    out session));

            }

            InitWithSessionHandle(session, options);
        }

        /// <summary>
        /// Initializes the session object with a native session handle
        /// </summary>
        /// <param name="session">Value of a native session object</param>
        /// <param name="options">Session options</param>
        private void InitWithSessionHandle(IntPtr session, SessionOptions options)
        {
            _nativeHandle = session;
            try
            {
                // Initialize input/output metadata
                _inputMetadata = new Dictionary<string, NodeMetadata>();
                _outputMetadata = new Dictionary<string, NodeMetadata>();
                _overridableInitializerMetadata = new Dictionary<string, NodeMetadata>();

                // get input count
                UIntPtr inputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputCount(_nativeHandle, out inputCount));

                // get all the input names and metadata
                for (ulong i = 0; i < (ulong)inputCount; i++)
                {
                    var iname = GetInputName(i);
                    _inputMetadata[iname] = GetInputMetadata(i);
                }
                // get output count
                UIntPtr outputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputCount(_nativeHandle, out outputCount));

                // get all the output names and metadata
                for (ulong i = 0; i < (ulong)outputCount; i++)
                {
                    _outputMetadata[GetOutputName(i)] = GetOutputMetadata(i);
                }

                // get overridable initializer count
                UIntPtr initilaizerCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerCount(_nativeHandle, out initilaizerCount));

                // get all the overridable initializer names and metadata
                for (ulong i = 0; i < (ulong)initilaizerCount; i++)
                {
                    _overridableInitializerMetadata[GetOverridableInitializerName(i)] = GetOverridableInitializerMetadata(i);
                }
                // set profiling's start time
                UIntPtr startTime = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetProfilingStartTimeNs(_nativeHandle,
                                                                    out startTime));
                _profilingStartTimeNs = (ulong)startTime;
            }
            catch (OnnxRuntimeException e)
            {
                if (_nativeHandle != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseSession(_nativeHandle);
                    _nativeHandle = IntPtr.Zero;
                }
                throw e;
            }

            _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call  
        }


        private string GetOutputName(ulong index)
        {
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }

            return str;
        }

        private string GetInputName(ulong index)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
            return str;
        }

        private string GetOverridableInitializerName(ulong index)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerName(
                                            _nativeHandle,
                                            (UIntPtr)index,
                                            allocator.Pointer,
                                            out nameHandle));
            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
            return str;
        }

        private NodeMetadata GetInputMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        private NodeMetadata GetOutputMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        private NodeMetadata GetOverridableInitializerMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        internal static NodeMetadata GetMetadataFromTypeInfo(IntPtr typeInfo)
        {
            OnnxValueType valueType;
            {
                IntPtr valType;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOnnxTypeFromTypeInfo(typeInfo, out valType));
                valueType = (OnnxValueType)valType;
            }
            if (valueType != OnnxValueType.ONNX_TYPE_TENSOR && valueType != OnnxValueType.ONNX_TYPE_SPARSETENSOR)
            {
                return new NodeMetadata(valueType, new int[] { }, new string[] { }, typeof(NamedOnnxValue));
            }

            // This should not be released
            IntPtr tensorInfo;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToTensorInfo(typeInfo, out tensorInfo)); //(IntPtr)(int)(uint)
            // Convert the newly introduced OrtTypeInfo* to the older OrtTypeAndShapeInfo*

            if (tensorInfo == IntPtr.Zero)
                return null;

            TensorElementType type;
            {
                IntPtr el_type;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(tensorInfo, out el_type));
                type = (TensorElementType)el_type;
            }
            Type dotnetType = null;
            int width = 0;
            TensorElementTypeConverter.GetTypeAndWidth(type, out dotnetType, out width);
            UIntPtr numDimensions;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(tensorInfo, out numDimensions));

            long[] dimensions = new long[(int)numDimensions];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(tensorInfo, dimensions, numDimensions));
            int[] intDimensions = new int[(int)numDimensions];
            for (var i = 0; i < (long)numDimensions; i++)
            {
                intDimensions[i] = (int)dimensions[i];
            }

            IntPtr[] dimensionNamePtrs = new IntPtr[(int)numDimensions];
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtGetSymbolicDimensions(tensorInfo, dimensionNamePtrs, numDimensions));

            string[] symbolicDimensions = new string[(int)numDimensions];
            for (var i = 0; i < (int)numDimensions; i++)
            {
                symbolicDimensions[i] = NativeOnnxValueHelper.StringFromNativeUtf8(dimensionNamePtrs[i]);
            }

            return new NodeMetadata(valueType, intDimensions, symbolicDimensions, dotnetType);
        }

        /// <summary>
        /// Other classes access
        /// </summary>
        internal IntPtr Handle
        {
            get
            {
                return _nativeHandle;
            }
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Finalizer. to cleanup session in case it runs
        /// and the user forgets to Dispose() of the session
        /// </summary>
        ~InferenceSession()
        {
            Dispose(false);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                // cleanup managed resources
                if (_builtInSessionOptions != null)
                {
                    _builtInSessionOptions.Dispose();
                    _builtInSessionOptions = null;
                }

                if (_builtInRunOptions != null)
                {
                    _builtInRunOptions.Dispose();
                    _builtInRunOptions = null;
                }
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeMethods.OrtReleaseSession(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion
    }


    /// <summary>
    /// Resembles type and shape information of session-graph nodes, used for communicating the shape/type of input/output nodes
    /// </summary>
    public class NodeMetadata
    {
        internal NodeMetadata(OnnxValueType onnxValueType, int[] dimensions, string[] symbolicDimensions, Type type)
        {
            OnnxValueType = onnxValueType;
            Dimensions = dimensions;
            SymbolicDimensions = symbolicDimensions;
            ElementType = type;
        }

        /// <summary>
        /// Type value of the node
        /// </summary>
        /// <value>A value of OnnxValueType enum</value>
        public OnnxValueType OnnxValueType { get; }

        /// <summary>
        /// Shape
        /// </summary>
        /// <value>Array of dimensions</value>
        public int[] Dimensions { get; }

        /// <summary>
        /// Symbolic dimensions
        /// </summary>
        /// <value>Array of symbolic dimensions if present.</value>
        public string[] SymbolicDimensions { get; }

        /// <summary>
        /// .NET type that corresponds to this Node.
        /// </summary>
        /// <value>System.Type</value>
        public System.Type ElementType { get; }

        /// <summary>
        /// Whether it is a Tensor
        /// </summary>
        /// <value>currently always returns true</value>
        public bool IsTensor
        {
            get
            {
                return true; // currently only Tensor nodes are supported
            }
        }
    }


    /// <summary>
    /// A class that queries and caches model metadata and exposes
    /// it as properties
    /// </summary>
    public class ModelMetadata
    {
        private string _producerName;
        private string _graphName;
        private string _domain;
        private string _description;
        private string _graphDescription;
        private long _version;
        private Dictionary<string, string> _customMetadataMap = new Dictionary<string, string>();

        internal ModelMetadata(InferenceSession session)
        {
            IntPtr modelMetadataHandle = IntPtr.Zero;

            var allocator = OrtAllocator.DefaultInstance;

            // Get the native ModelMetadata instance associated with the InferenceSession

            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetModelMetadata(session.Handle, out modelMetadataHandle));

            try
            {

                // Process producer name
                IntPtr producerNameHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetProducerName(modelMetadataHandle, allocator.Pointer, out producerNameHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, producerNameHandle, 0))
                {
                    _producerName = NativeOnnxValueHelper.StringFromNativeUtf8(producerNameHandle);
                }

                // Process graph name
                IntPtr graphNameHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetGraphName(modelMetadataHandle, allocator.Pointer, out graphNameHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, graphNameHandle, 0))
                {
                    _graphName = NativeOnnxValueHelper.StringFromNativeUtf8(graphNameHandle);
                }


                // Process domain
                IntPtr domainHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetDomain(modelMetadataHandle, allocator.Pointer, out domainHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, domainHandle, 0))
                {
                    _domain = NativeOnnxValueHelper.StringFromNativeUtf8(domainHandle);
                }

                // Process description
                IntPtr descriptionHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetDescription(modelMetadataHandle, allocator.Pointer, out descriptionHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, descriptionHandle, 0))
                {
                    _description = NativeOnnxValueHelper.StringFromNativeUtf8(descriptionHandle);
                }

                // Process graph description
                IntPtr graphDescriptionHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetGraphDescription(modelMetadataHandle, allocator.Pointer, out graphDescriptionHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, graphDescriptionHandle, 0))
                {
                    _graphDescription = NativeOnnxValueHelper.StringFromNativeUtf8(graphDescriptionHandle);
                }

                // Process version
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetVersion(modelMetadataHandle, out _version));


                // Process CustomMetadata Map
                IntPtr customMetadataMapKeysHandle = IntPtr.Zero;
                long numKeys;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetCustomMetadataMapKeys(modelMetadataHandle, allocator.Pointer, out customMetadataMapKeysHandle, out numKeys));

                // We have received an array of null terminated C strings which are the keys that we can use to lookup the custom metadata map
                // The OrtAllocator will finally free the customMetadataMapKeysHandle
                using (var ortAllocationKeysArray = new OrtMemoryAllocation(allocator, customMetadataMapKeysHandle, 0))
                using (var ortAllocationKeys = new DisposableList<OrtMemoryAllocation>((int)numKeys))
                {
                    // Put all the handles to each key in the DisposableList to be disposed off in an exception-safe manner
                    for (int i = 0; i < (int)numKeys; ++i)
                    {
                        ortAllocationKeys.Add(new OrtMemoryAllocation(allocator, Marshal.ReadIntPtr(customMetadataMapKeysHandle, IntPtr.Size * i), 0));
                    }

                    // Process each key via the stored key handles
                    foreach (var allocation in ortAllocationKeys)
                    {
                        IntPtr keyHandle = allocation.Pointer;
                        IntPtr valueHandle = IntPtr.Zero;
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataLookupCustomMetadataMap(modelMetadataHandle, allocator.Pointer, keyHandle, out valueHandle));

                        using (var ortAllocationValue = new OrtMemoryAllocation(allocator, valueHandle, 0))
                        {
                            var key = NativeOnnxValueHelper.StringFromNativeUtf8(keyHandle);
                            var value = NativeOnnxValueHelper.StringFromNativeUtf8(valueHandle);

                            // Put the key/value pair into the dictionary
                            _customMetadataMap[key] = value;

                        }
                    }
                }
            }

            finally
            {

                // Free ModelMetadata handle
                NativeMethods.OrtReleaseModelMetadata(modelMetadataHandle);

            }

        }

        /// <summary>
        /// Producer name string
        /// </summary>
        /// <value>producer name string</value>
        public string ProducerName
        {
            get
            {
                return _producerName;
            }
        }

        /// <summary>
        /// Graph name for this model
        /// </summary>
        /// <value>graph name string</value>
        public string GraphName
        {
            get
            {
                return _graphName;
            }
        }

        /// <summary>
        /// Domain for this model
        /// </summary>
        /// <value>domain name string</value>
        public string Domain
        {
            get
            {
                return _domain;
            }
        }

        /// <summary>
        /// Unstructured model description
        /// </summary>
        /// <value>description string</value>
        public string Description
        {
            get
            {
                return _description;
            }
        }

        /// <summary>
        /// Unstructured graph description
        /// </summary>
        /// <value>description string</value>
        public string GraphDescription
        {
            get
            {
                return _graphDescription;
            }
        }

        /// <summary>
        /// Version number
        /// </summary>
        /// <value>long version integer</value>
        public long Version
        {
            get
            {
                return _version;
            }
        }

        /// <summary>
        /// Custom metadata key/value pairs
        /// </summary>
        /// <value>An instance of a Dictionary<string,string></value>
        public Dictionary<string, string> CustomMetadataMap
        {
            get
            {
                return _customMetadataMap;
            }
        }
    }


}
