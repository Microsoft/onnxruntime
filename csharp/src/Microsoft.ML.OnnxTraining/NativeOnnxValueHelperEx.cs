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
    /// This helper class contains methods to create native OrtValue from a managed value object
    /// </summary>
    internal static class NativeOnnxValueHelperEx
    {
        /// <summary>
        /// Reads double from a byte array.
        /// </summary>
        /// <param name="byteArray">pointer to native or pinned memory where bytes resides</param>
        /// <returns></returns>
        internal static double DoubleFromByteArray(IntPtr byteArray, int nLen)
        {
            byte[] rgBuffer = new byte[nLen];
            Marshal.Copy(byteArray, rgBuffer, 0, nLen);
            return BitConverter.ToDouble(rgBuffer, 0);
        }
    }
}
