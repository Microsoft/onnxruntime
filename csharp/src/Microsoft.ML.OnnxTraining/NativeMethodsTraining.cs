// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// Enumerations must match the same values as those in orttraining_c_api.h
    /// </summary>
    #region Enumreations

    public enum OrtTrainingOptimizer
    {
        ORT_TRAINING_OPTIMIZER_SGD = 0
    };

    public enum OrtTrainingLossFunction
    {
        ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY = 0
    };

    public enum OrtTrainingStringParameter
    {
        ORT_TRAINING_MODEL_PATH = 0,
        ORT_TRAINING_LOG_PATH = 1,
        ORT_TRAINING_INPUT_LABELS = 2,
        ORT_TRAINING_OUTPUT_PREDICTIONS = 3,
        ORT_TRAINING_OUTPUT_LOSS = 4
    };

    public enum OrtTrainingLongParameter
    {
        ORT_TRAINING_NUM_TRAIN_STEPS = 0,
        ORT_TRAINING_TRAIN_BATCH_SIZE = 1,
        ORT_TRAINING_EVAL_BATCH_SIZE = 2,
        ORT_TRAINING_EVAL_PERIOD = 3,
        ORT_TRAINING_DISPLAY_LOSS_STEPS = 4
    };

    public enum OrtTrainingNumericParameter
    {
        ORT_TRAINING_LEARNING_RATE = 0
    };

    public enum OrtTrainingBooleanParameter
    {
        ORT_TRAINING_USE_GIST = 0,
        ORT_TRAINING_USE_CUDA = 1,
        ORT_TRAINING_USE_PROFILER = 2,
        ORT_TRAINING_USE_TENSORBOARD = 3,
        ORT_TRAINING_IS_PERFTEST = 4,
        ORT_TRAINING_SHUFFLE_DATA = 5
    };

    #endregion

    [StructLayout(LayoutKind.Sequential)]
    public struct OrtTrainingApiBase
    {
        public IntPtr GetApi;
        public IntPtr GetVersionString;
    };

    /// <summary>
    /// Defines the error function callback delegate.
    /// </summary>
    /// <param name="nCount">Specifies the number of ORtValues in the OrtValueCollection.</param>
    /// <param name="colVal">Specifies the OrtValueCollection.  Use GetAt/SetAt funnctions with the collection.</param>
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate void OrtErrorFunctionCallback(IntPtr /* (OrtValueCollection*) */ colVal); 

    /// <summary>
    /// Defines the evaluation function callback delegate.
    /// </summary>
    /// <param name="num_samples">Specifies the number of samples run.</param>
    /// <param name="step">Specifies the current step.</param>
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate void OrtEvaluationFunctionCallback(long num_samples, long step);

    /// <summary>
    /// Defines the callback called to get a batch of data.  This callback is used to fill the colVal with OrtValues making up
    /// the entire batch.  All 'nCount' items should be set in the colVal using the SetAt method.
    /// </summary>
    /// <param name="nBatchSize">Specifies the batch size.</param>
    /// <param name="nCount">Specifies the number of items that can be set in the colVal.</param>
    /// <param name="colVal">Specifies the collection to be filled using the SetAt method.</param>
    /// <param name="hInputShape">Specifies the OrtShape containing all input dimensions 'after' the batch but not including the batch size.</param>
    /// <param name="hOutputShape">Specifies the OrtShape containing all output dimensions 'after' the batch but not including the batch size.</param>
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate void OrtDataGetBatchCallback(long nBatchSize, IntPtr /* (OrtValueCollection*) */ colVal, IntPtr /* (OrtShape*) */ hInputShape, IntPtr /* (OrtShape*) */ hOutputShape);

    // NOTE: The order of the APIs in this struct should match exactly that in
    // OrtTrainingApi ort_training_api_1_to_6 (orttraining_c_api.cc)
    [StructLayout(LayoutKind.Sequential)]
    public struct OrtTrainingApi
    {
        public IntPtr CreateTrainingParameters;
        public IntPtr CloneTrainingParameters;

        public IntPtr SetTrainingParameter_string;
        public IntPtr GetTrainingParameter_string;
        public IntPtr SetTrainingParameter_bool;
        public IntPtr GetTrainingParameter_bool;
        public IntPtr SetTrainingParameter_long;
        public IntPtr GetTrainingParameter_long;
        public IntPtr SetTrainingNumericParameter;
        public IntPtr GetTrainingNumericParameter;

        public IntPtr SetTrainingOptimizer;
        public IntPtr GetTrainingOptimizer;
        public IntPtr SetTrainingLossFunction;
        public IntPtr GetTrainingLossFunction;

        public IntPtr SetupTrainingParameters;
        public IntPtr SetupTrainingData;

        public IntPtr InitializeTraining;
        public IntPtr RunTraining;
        public IntPtr EndTraining;

        public IntPtr GetCount;
        public IntPtr GetCapacity;
        public IntPtr GetAt;
        public IntPtr SetAt;

        public IntPtr GetDimCount;
        public IntPtr GetDimAt;

        public IntPtr ReleaseTrainingParameters;
    };

    internal static class NativeMethodsTraining
    {
        private const string nativeLib = "onnxruntime";
        internal const CharSet charSet = CharSet.Ansi;

        static OrtTrainingApi api_;

        public delegate ref OrtTrainingApi DOrtGetTrainingApi(UInt32 version);

        static NativeMethodsTraining()
        {
            DOrtGetTrainingApi OrtGetTrainingApi = (DOrtGetTrainingApi)Marshal.GetDelegateForFunctionPointer(OrtGetTrainingApiBase().GetApi, typeof(DOrtGetTrainingApi));

            // TODO: Make this save the pointer, and not copy the whole structure across
            api_ = (OrtTrainingApi)OrtGetTrainingApi(6 /*ORT_API_VERSION*/);

            OrtCreateTrainingParameters = (DOrtCreateTrainingParameters)Marshal.GetDelegateForFunctionPointer(api_.CreateTrainingParameters, typeof(DOrtCreateTrainingParameters));
            OrtReleaseTrainingParameters = (DOrtReleaseTrainingParameters)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTrainingParameters, typeof(DOrtReleaseTrainingParameters));
            OrtCloneTrainingParameters = (DOrtCloneTrainingParameters)Marshal.GetDelegateForFunctionPointer(api_.CloneTrainingParameters, typeof(DOrtCloneTrainingParameters));

            OrtSetParameter_string = (DOrtSetParameter_string)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingParameter_string, typeof(DOrtSetParameter_string));
            OrtGetParameter_string = (DOrtGetParameter_string)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingParameter_string, typeof(DOrtGetParameter_string));
            OrtSetParameter_bool = (DOrtSetParameter_bool)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingParameter_bool, typeof(DOrtSetParameter_bool));
            OrtGetParameter_bool = (DOrtGetParameter_bool)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingParameter_bool, typeof(DOrtGetParameter_bool));
            OrtSetParameter_long = (DOrtSetParameter_long)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingParameter_long, typeof(DOrtSetParameter_long));
            OrtGetParameter_long = (DOrtGetParameter_long)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingParameter_long, typeof(DOrtGetParameter_long));
            OrtSetNumericParameter = (DOrtSetNumericParameter)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingNumericParameter, typeof(DOrtSetNumericParameter));
            OrtGetNumericParameter = (DOrtGetNumericParameter)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingNumericParameter, typeof(DOrtGetNumericParameter));

            OrtSetTrainingOptimizer = (DOrtSetTrainingOptimizer)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingOptimizer, typeof(DOrtSetTrainingOptimizer));
            OrtGetTrainingOptimizer = (DOrtGetTrainingOptimizer)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingOptimizer, typeof(DOrtGetTrainingOptimizer));
            OrtSetTrainingLossFunction = (DOrtSetTrainingLossFunction)Marshal.GetDelegateForFunctionPointer(api_.SetTrainingLossFunction, typeof(DOrtSetTrainingLossFunction));
            OrtGetTrainingLossFunction = (DOrtGetTrainingLossFunction)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingLossFunction, typeof(DOrtGetTrainingLossFunction));

            OrtSetupTrainingParameters = (DOrtSetupTrainingParameters)Marshal.GetDelegateForFunctionPointer(api_.SetupTrainingParameters, typeof(DOrtSetupTrainingParameters));
            OrtSetupTrainingData = (DOrtSetupTrainingData)Marshal.GetDelegateForFunctionPointer(api_.SetupTrainingData, typeof(DOrtSetupTrainingData));

            OrtInitializeTraining = (DOrtInitializeTraining)Marshal.GetDelegateForFunctionPointer(api_.InitializeTraining, typeof(DOrtInitializeTraining));
            OrtRunTraining = (DOrtRunTraining)Marshal.GetDelegateForFunctionPointer(api_.RunTraining, typeof(DOrtRunTraining));
            OrtEndTraining = (DOrtEndTraining)Marshal.GetDelegateForFunctionPointer(api_.EndTraining, typeof(DOrtEndTraining));

            OrtGetCount = (DOrtGetCount)Marshal.GetDelegateForFunctionPointer(api_.GetCount, typeof(DOrtGetCount));
            OrtGetCapacity = (DOrtGetCapacity)Marshal.GetDelegateForFunctionPointer(api_.GetCapacity, typeof(DOrtGetCapacity));
            OrtGetAt = (DOrtGetAt)Marshal.GetDelegateForFunctionPointer(api_.GetAt, typeof(DOrtGetAt));
            OrtSetAt = (DOrtSetAt)Marshal.GetDelegateForFunctionPointer(api_.SetAt, typeof(DOrtSetAt));

            OrtGetDimCount = (DOrtGetDimCount)Marshal.GetDelegateForFunctionPointer(api_.GetDimCount, typeof(DOrtGetDimCount));
            OrtGetDimAt = (DOrtGetDimAt)Marshal.GetDelegateForFunctionPointer(api_.GetDimAt, typeof(DOrtGetDimAt));
        }

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ref OrtTrainingApiBase OrtGetTrainingApiBase();

        #region Training Parameters

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateTrainingParameters(out IntPtr /*(OrtTrainingParameters**)*/ trainingParameters);
        public static DOrtCreateTrainingParameters OrtCreateTrainingParameters;

        public delegate void DOrtReleaseTrainingParameters(IntPtr /*(OrtTrainingParameters*)*/trainingParameters);
        public static DOrtReleaseTrainingParameters OrtReleaseTrainingParameters;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCloneTrainingParameters(IntPtr /*(OrtTrainingParameters*)*/ trainingParameters, out IntPtr /*(OrtTrainingParameters**)*/ output);
        public static DOrtCloneTrainingParameters OrtCloneTrainingParameters;

        public delegate IntPtr /* OrtStatus */DOrtSetParameter_string(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam, 
                                                OrtTrainingStringParameter key,
                                                //[MarshalAs(UnmanagedType.LPStr)]string strVal
                                                byte[] strVal);
        public static DOrtSetParameter_string OrtSetParameter_string;

        public delegate IntPtr /* OrtStatus */DOrtGetParameter_string(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingStringParameter key,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/ strVal);
        public static DOrtGetParameter_string OrtGetParameter_string;


        public delegate IntPtr /* OrtStatus */DOrtSetParameter_bool(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingBooleanParameter key,
                                                bool bVal);
        public static DOrtSetParameter_bool OrtSetParameter_bool;

        public delegate IntPtr /* OrtStatus */DOrtGetParameter_bool(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingBooleanParameter key,
                                                out UIntPtr val);
        public static DOrtGetParameter_bool OrtGetParameter_bool;


        public delegate IntPtr /* OrtStatus */DOrtSetParameter_long(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingLongParameter key,
                                                long lVal);
        public static DOrtSetParameter_long OrtSetParameter_long;

        public delegate IntPtr /* OrtStatus */DOrtGetParameter_long(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingLongParameter key,
                                                out UIntPtr val);
        public static DOrtGetParameter_long OrtGetParameter_long;


        public delegate IntPtr /* OrtStatus */DOrtSetNumericParameter(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingNumericParameter key,
                                                double dfVal);
        public static DOrtSetNumericParameter OrtSetNumericParameter;

        public delegate IntPtr /* OrtStatus */DOrtGetNumericParameter(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingNumericParameter key,
                                                IntPtr /* (OrtAllocator*) */ allocator,
                                                out IntPtr /* (char*) */ strVal);
        public static DOrtGetNumericParameter OrtGetNumericParameter;


        public delegate IntPtr /* OrtStatus */DOrtSetTrainingOptimizer(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingOptimizer opt);
        public static DOrtSetTrainingOptimizer OrtSetTrainingOptimizer;

        public delegate IntPtr /* OrtStatus */DOrtGetTrainingOptimizer(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                out UIntPtr val);
        public static DOrtGetTrainingOptimizer OrtGetTrainingOptimizer;

        public delegate IntPtr /* OrtStatus */DOrtSetTrainingLossFunction(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                OrtTrainingLossFunction loss);
        public static DOrtSetTrainingLossFunction OrtSetTrainingLossFunction;

        public delegate IntPtr /* OrtStatus */DOrtGetTrainingLossFunction(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                out UIntPtr val);
        public static DOrtGetTrainingLossFunction OrtGetTrainingLossFunction;


        public delegate IntPtr /* OrtStatus */DOrtSetupTrainingParameters(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                [MarshalAs(UnmanagedType.FunctionPtr)] OrtErrorFunctionCallback errorFn,
                                                [MarshalAs(UnmanagedType.FunctionPtr)] OrtEvaluationFunctionCallback evalFn);
        public static DOrtSetupTrainingParameters OrtSetupTrainingParameters;

        #endregion

        #region Training Data

        public delegate IntPtr /* OrtStatus */DOrtSetupTrainingData(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam,
                                                [MarshalAs(UnmanagedType.FunctionPtr)] OrtDataGetBatchCallback traininggetdataFn,
                                                //[MarshalAs(UnmanagedType.LPStr)]string strFeedNames
                                                [MarshalAs(UnmanagedType.FunctionPtr)] OrtDataGetBatchCallback testinggetdataFn,
                                                //[MarshalAs(UnmanagedType.LPStr)]string strFeedNames
                                                byte[] strFeedNames);
        public static DOrtSetupTrainingData OrtSetupTrainingData;

        #endregion

        #region Training

        public delegate IntPtr /* OrtStatus */DOrtInitializeTraining(
                                                IntPtr /* (OrtEnv*) */ env,
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam);
        public static DOrtInitializeTraining OrtInitializeTraining;

        public delegate IntPtr /* OrtStatus */DOrtRunTraining(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam);
        public static DOrtRunTraining OrtRunTraining;

        public delegate IntPtr /* OrtStatus */DOrtEndTraining(
                                                IntPtr /* (OrtTrainingParameters*) */ trainParam);
        public static DOrtEndTraining OrtEndTraining;

        #endregion

        #region OrtValueCollection

        public delegate IntPtr /* OrtStatus */DOrtGetCount(
                                                IntPtr /* (OrtValueCollection*) */ colVal,
                                                out UIntPtr count);
        public static DOrtGetCount OrtGetCount;

        public delegate IntPtr /* OrtStatus */DOrtGetCapacity(
                                                IntPtr /* (OrtValueCollection*) */ colVal,
                                                out UIntPtr count);
        public static DOrtGetCapacity OrtGetCapacity;

        public delegate IntPtr /* OrtStatus */DOrtGetAt(
                                                IntPtr /* (OrtValueCollection*) */ colVal,
                                                int nIdx,
                                                out IntPtr /* (OrtValue*) */ val,
                                                IntPtr /* (OrtAllocator*) */ allocator,
                                                out IntPtr /* (char*) */ strName);
        public static DOrtGetAt OrtGetAt;

        public delegate IntPtr /* OrtStatus */DOrtSetAt(
                                                IntPtr /* (OrtValueCollection*) */ colVal,
                                                int nIdx,
                                                IntPtr /* (OrtValue*) */ val,
                                                //[MarshalAs(UnmanagedType.LPStr)]string strName
                                                byte[] strName);
        public static DOrtSetAt OrtSetAt;

        #endregion

        #region OrtShape

        public delegate IntPtr /* OrtStatus */DOrtGetDimCount(
                                                IntPtr /* (OrtShape*) */ shape,
                                                out UIntPtr val);
        public static DOrtGetDimCount OrtGetDimCount;

        public delegate IntPtr /* OrtStatus */DOrtGetDimAt(
                                                IntPtr /* (OrtShape*) */ shape,
                                                int nIdx,
                                                out UIntPtr val);
        public static DOrtGetDimAt OrtGetDimAt;

        #endregion

        public static string GetPlatformSerializedString(byte[] rgBytes, int nLen)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return System.Text.Encoding.Unicode.GetString(rgBytes, 0, nLen);
            else
                return System.Text.Encoding.UTF8.GetString(rgBytes, 0, nLen);
        }
    };
}
