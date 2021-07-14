// Copyright (c) 2018-2020 SignalPop LLC, and contributors. All rights reserved.
// License: Apache 2.0
// License: https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE
// Original Source: https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.data/ProgressInfo.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// The ProgressInfo is used when reporting the overall progress of creating a dataset.
    /// </summary>
    public class ProgressInfo
    {
        int m_nIdx;
        int m_nTotal;
        string m_strMsg;
        Exception m_err;
        bool? m_bAlive = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIdx">Specifies the current index in the process.</param>
        /// <param name="nTotal">Specifies the total items to process.</param>
        /// <param name="str">Specifies the message to display.</param>
        /// <param name="err">Specifies an error if one occurred, or null.</param>
        /// <param name="bAlive">Specifies whether or not the process is alive.</param>
        public ProgressInfo(int nIdx, int nTotal, string str, Exception err = null, bool? bAlive = null)
        {
            m_nIdx = nIdx;
            m_nTotal = nTotal;
            m_strMsg = str;
            m_err = err;
            m_bAlive = bAlive;
        }

        /// <summary>
        /// Returns the percentage of the current process.
        /// </summary>
        public double Percentage
        {
            get { return (m_nTotal == 0) ? 0 : (double)m_nIdx / (double)m_nTotal; }
        }

        /// <summary>
        /// Returns the message as a string.
        /// </summary>
        public string Message
        {
            get { return m_strMsg; }
        }

        /// <summary>
        /// Returns the error if one occurred, or null.
        /// </summary>
        public Exception Error
        {
            get { return m_err; }
        }

        /// <summary>
        /// Returns whether or not the process is alive or not.
        /// </summary>
        public bool? Alive
        {
            get { return m_bAlive; }
            set { m_bAlive = value; }
        }
    }
}
