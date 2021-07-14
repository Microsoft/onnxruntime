// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    public class TrainingSession : IDisposable
    {
        TrainingParameters m_param = new TrainingParameters();
        bool _disposed = false;

        public TrainingSession()
        {
        }

        /// <summary>
        /// Finalizer. to cleanup training session in case it runs
        /// and the user forgets to Dispose() of the training session.
        /// </summary>
        ~TrainingSession()
        {
            Dispose(false);
        }

        #region Disposable

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // Suppress finalization.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            // dispose managed state (managed objects).
            if (disposing)
            {
                m_param.Dispose();
            }

            _disposed = true;
        }

        #endregion


        public TrainingParameters Parameters
        {
            get { return m_param; }
        }

        public void Initialize(OrtEnv env)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtInitializeTraining(env.DangerousGetHandle(), m_param.Handle));
        }

        public void RunTraining()
        {           
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtRunTraining(m_param.Handle));
        }

        public void EndTraining()
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtEndTraining(m_param.Handle));
        }
    }
}
