// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Provider options for TensorRT.
    /// </summary>
    //  Example for setting:
    //    SessionOptions.OrtTensorRTProviderOptions trt_options;
    //    trt_options.device_id = 0;
    //    trt_options.has_trt_options = 1;
    //    trt_options.trt_max_workspace_size = (UIntPtr) (1<<30);
    //    trt_options.trt_fp16_enable = 1;
    //    trt_options.trt_int8_enable = 1;
    //    trt_options.trt_int8_calibration_table_name = "calibration.flatbuffers";
    //    trt_options.trt_int8_use_native_calibration_table = 0;
    public struct OrtTensorRTProviderOptions
    {
        public int device_id;                                  //!< cuda device id. Default is 0. </typeparam>
        public int has_trt_options;                            //!< override environment variables with following TensorRT settings at runtime. Default 0 = false, nonzero = true.
        public UIntPtr trt_max_workspace_size;                 //!< maximum workspace size for TensorRT. ORT C++ DLL has this field to be the type of size_t, hence using UIntPtr for conversion.
        public int trt_fp16_enable;                            //!< enable TensorRT FP16 precision. Default 0 = false, nonzero = true.
        public int trt_int8_enable;                            //!< enable TensorRT INT8 precision. Default 0 = false, nonzero = true.
        public String trt_int8_calibration_table_name;         //!< TensorRT INT8 calibration table name.
        public int trt_int8_use_native_calibration_table;      //!< use native TensorRT generated calibration table. Default 0 = false, nonzero = true
        public int trt_max_partition_iterations;               //!< maximum number of iterations allowed in model partitioning for TensorRT.
        public int trt_min_subgraph_size;                      //!< minimum node size in a subgraph after partitioning.
        public int trt_dump_subgraphs;                         //!< dump the subgraphs that are transformed into TRT engines in onnx format to the filesystem. Default 0 = false, nonzero = true
        public int trt_engine_cache_enable;                    //!< enable TensorRT engine caching. Default 0 = false, nonzero = true
        public String trt_cache_path;                          //!< specify path for TensorRT engine and profile files if engine_cache_enable is enabled, or INT8 calibration table file if trt_int8_enable is enabled.
    }

    /// <summary>
    /// Holds provider options configuration for creating an InferenceSession.
    /// </summary>
    public class ProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty ProviderOptions
        /// </summary>
        public ProviderOptions()
            : base(IntPtr.Zero, true)
        {
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get TensorRT provider options with default setting.
        /// </summary>
        /// <returns> TRT provider options instance.  </returns>
        public static OrtTensorRTProviderOptions GetDefaultTensorRTProviderOptions()
        {
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = 0;
            trt_options.has_trt_options = 0;
            trt_options.trt_max_workspace_size = (UIntPtr)(1 << 30);
            trt_options.trt_fp16_enable = 0;
            trt_options.trt_int8_enable = 0;
            trt_options.trt_int8_calibration_table_name = "";
            trt_options.trt_int8_use_native_calibration_table = 0;
            trt_options.trt_max_partition_iterations = 1000;
            trt_options.trt_min_subgraph_size = 1;
            trt_options.trt_dump_subgraphs = 0;
            trt_options.trt_engine_cache_enable = 0;
            trt_options.trt_cache_path = "";

            return trt_options;
        }
        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of SessionOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}