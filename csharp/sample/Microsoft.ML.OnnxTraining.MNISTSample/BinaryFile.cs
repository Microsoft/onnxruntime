// Copyright (c) 2018-2020 SignalPop LLC, and contributors. All rights reserved.
// License: Apache 2.0
// License: https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE
// Modified from Original Source: https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.data/MnistDataLoader.cs
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.data namespace contains dataset creators used to create common testing datasets such as MNIST, CIFAR-10 and VOC0712.
/// </summary>
namespace MyCaffe.data
{
    /// <summary>
    /// The BinaryFile class is used to manage binary files used by the MNIST dataset creator.
    /// </summary>
    public class BinaryFile : IDisposable
    {
        private FileStream _file;
        private BinaryReader _reader;
        private bool _disposed = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strFile">Specifies the filename.</param>
        public BinaryFile(string strFile)
        {
            _file = File.Open(strFile, FileMode.Open, FileAccess.Read, FileShare.Read);
            _reader = new BinaryReader(_file);
        }

        /// <summary>
        /// Finalizer. to cleanup session in case it runs
        /// and the user forgets to Dispose() of the session
        /// </summary>
        ~BinaryFile()
        {
            Dispose(false);
        }

        #region Disposable

        /// <summary>
        /// Dispose if disposing, or ignore if already disposed.
        /// </summary>
        /// <param name="disposing">Specifies whether or not we are disposing.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            // dispose managed state (managed objects).
            if (disposing)
            {
                _reader.Close();
                _disposed = true;
            }
        }

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // Suppress finalization.
            GC.SuppressFinalize(this);
        }

        #endregion

        /// <summary>
        /// Returns the binary reader used.
        /// </summary>
        /// <value></value>
        public BinaryReader Reader
        {
            get { return _reader; }
        }

        /// <summary>
        /// Reads in a UINT32 and performs an endian swap.
        /// </summary>
        /// <returns>The endian swapped UINT32 is returned.</returns>
        public UInt32 ReadUInt32()
        {
            UInt32 nVal = _reader.ReadUInt32();

            return swap_endian(nVal);
        }

        /// <summary>
        /// Reads bytes from the file.
        /// </summary>
        /// <param name="nCount">Specifies the number of bytes to read.</param>
        /// <returns>The bytes read are returned in an array.</returns>
        public byte[] ReadBytes(int nCount)
        {
            return _reader.ReadBytes(nCount);
        }

        private UInt32 swap_endian(UInt32 nVal)
        {
            nVal = ((nVal << 8) & 0xFF00FF00) | ((nVal >> 8) & 0x00FF00FF);
            return (nVal << 16) | (nVal >> 16);
        }
    }
}
