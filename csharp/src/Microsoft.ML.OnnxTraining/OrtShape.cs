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
    /// The OrtShape holds a set of dimensions defining a shape.
    /// </summary>
    // Copyright (c) Microsoft Corporation. All rights reserved.
    // Copyright (c) SignalPop LLC. All rights reserved.
    // Licensed under the MIT License.

    internal class OrtShape
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtShape
        /// </summary>
        protected IntPtr _nativeHandle;

        /// <summary>
        /// The OrtShape is an that contains a shape where the dim is queried using the GetShapeDimAt method.
        /// </summary>
        /// <param name="h">Specifies the handle to the native OrtShape.</param>
        public OrtShape(IntPtr h)
        {
            _nativeHandle = h;
        }

        #region Public Methods

        /// <summary>
        /// Get the number of dimensions in the shape.
        /// </summary>
        public int Count
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetDimCount(_nativeHandle, out val));
                return (int)val;
            }
        }

        /// <summary>
        /// Get the dimension at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index to query for the dimension.</param>
        /// <returns>The dimension at the index is returned.</returns>
        public int GetDimAt(int nIdx)
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetDimAt(_nativeHandle, nIdx, out val));
            return (int)val;
        }

        /// <summary>
        /// Return the shape list of dimensions.
        /// </summary>
        /// <returns>The list of dimensions is returned as an array of int.</returns>
        public List<int> GetShape()
        {
            int nCount = Count;

            List<int> rgShape = new List<int>();
            for (int i = 0; i < nCount; i++)
            {
                rgShape.Add(GetDimAt(i));
            }

            return rgShape;
        }

        #endregion
    }
}
