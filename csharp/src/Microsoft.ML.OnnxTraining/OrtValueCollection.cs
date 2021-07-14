// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// The OrtValueCollection holds ort values returned by the error function, but does not actually own any of them
    /// and therefore does not release them.
    /// </summary>
    internal class OrtValueCollection
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtValueCollection
        /// </summary>
        protected IntPtr _nativeHandle;

        /// <summary>
        /// The OrtValueCollection is an object is not a native collection, but instead
        /// gives access to a group of native OrtValues via its GetAt and SetAt methods.
        /// </summary>
        /// <param name="h">Specifies the handle to the native OrtValueCollection.</param>
        /// <remarks>
        /// For efficiency, the OrtValue collection gives access to a set of OrtValues where
        /// each OrtValue does not actually own the memory but instead points to one or 
        /// more pre-allocated OrtValues. 
        /// </remarks>
        public OrtValueCollection(IntPtr h)
        {
            _nativeHandle = h;
        }

        #region Public Methods

        public int Count
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCount(_nativeHandle, out val));
                return (int)val;
            }
        }

        public int Capacity
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCapacity(_nativeHandle, out val));
                return (int)val;
            }
        }

        public OrtValue GetAt(int nIdx, out string strName)
        {
            IntPtr valData;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valName;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetAt(_nativeHandle, nIdx, out valData, allocator.Pointer, out valName));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valName, 0))
            {
                strName = NativeOnnxValueHelper.StringFromNativeUtf8(valName);
            }

            return new OrtValue(valData, false);
        }

        public void SetAt(int nIdx, OrtValue val, string strName = "")
        {
            byte[] rgName = (string.IsNullOrEmpty(strName)) ? null : NativeMethods.GetPlatformSerializedString(strName);
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetAt(_nativeHandle, nIdx, val.Handle, rgName));
        }

        #endregion
    }
}
