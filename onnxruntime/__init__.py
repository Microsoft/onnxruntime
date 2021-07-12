# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""
__version__ = "1.8.0"
__author__ = "Microsoft"

# we need to do device version validation (for example to check Cuda version for an onnxruntime-training package).
# in order to know whether the onnxruntime package is for training it needs
# to do import onnxruntime.training.ortmodule first.
# onnxruntime.capi._pybind_state is required before import onnxruntime.training.ortmodule.
# however, import onnxruntime.capi._pybind_state will already raise an exception if a required Cuda version
# is not found.
# here we need to save the exception and continue with Cuda version validation in order to post
# meaningful messages to the user.
# the saved exception is raised after device version validation.
try:
    from onnxruntime.capi._pybind_state import get_all_providers, get_available_providers, get_device, set_seed, \
        RunOptions, SessionOptions, set_default_logger_severity, enable_telemetry_events, disable_telemetry_events, \
        NodeArg, ModelMetadata, GraphOptimizationLevel, ExecutionMode, ExecutionOrder, SessionIOBinding, \
        OrtAllocatorType, OrtMemType, OrtArenaCfg, OrtMemoryInfo, create_and_register_allocator,  OrtSparseFormat
    import_capi_exception = None
except Exception as e:
    import_capi_exception = e

from onnxruntime.capi import onnxruntime_validation

if import_capi_exception:
    raise import_capi_exception

from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession, IOBinding, OrtValue, SparseTensor, \
    OrtDevice, numpy_array_to_cuda, numpy_array_to_cpu

from onnxruntime.capi.training import *  # noqa: F403

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:
    from . import experimental
except ImportError:
    pass

from onnxruntime.capi.onnxruntime_validation import package_name, version, cuda_version
if version:
    __version__ = version

onnxruntime_validation.check_distro_info()
