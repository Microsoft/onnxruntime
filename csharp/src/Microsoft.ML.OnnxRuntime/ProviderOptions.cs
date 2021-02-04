// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for configuring a CUDA Execution Provider instance
    /// </summary>
    public class OrtCUDAProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        #region Constructor

        /// <summary>
        /// Constructs an empty OrtCUDAProviderOptions instance
        /// </summary>
        public OrtCUDAProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCUDAProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Updates  the configuration knobs of OrtCUDAProviderOptions that will eventually be used to configure a CUDA EP
        /// Please refer to the following on different key/value pairs to configure a CUDA EP and their meaning:
        /// https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html
        /// </summary>
        /// <param name="providerOptions">key/value pairs used to configure a CUDA Execution Provider</param>

        public void UpdateOptions(Dictionary<string, string> providerOptions)
        {

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var keysArray = NativeOnnxValueHelper.ConvertNamesToUtf8(providerOptions.Keys.ToArray(), n => n, cleanupList);
                var valuesArray = NativeOnnxValueHelper.ConvertNamesToUtf8(providerOptions.Values.ToArray(), n => n, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtUpdateCUDAProviderOptions(handle, keysArray, valuesArray, (UIntPtr)providerOptions.Count));
            }
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region Private Methods


        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtCUDAProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseCUDAProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }

}
