// Copyright (c) 2018-2020 SignalPop LLC, and contributors. All rights reserved.
// License: Apache 2.0
// License: https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE
// Original Source: https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.data/ProgressArgs.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// Defines the arguments sent to the OnProgress and OnError events.
    /// </summary>
    public class ProgressArgs : EventArgs
    {
        ProgressInfo m_pi;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="pi">Specifies the progress information.</param>
        public ProgressArgs(ProgressInfo pi)
        {
            m_pi = pi;
        }

        /// <summary>
        /// Returns the progress information.
        /// </summary>
        public ProgressInfo Progress
        {
            get { return m_pi; }
        }
    }
}
