# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._training_manager import TrainingManager
from ._inference_manager import InferenceManager
from ._onnx_training_manager import OnnxTrainingManager
from ._onnx_inference_manager import OnnxInferenceManager


class GraphExecutionManagerFactory(object):
    def __init__(self, module):
        self._training_manager = TrainingManager(module)
        self._inference_manager = InferenceManager(module)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager


class OnnxGraphExecutionManagerFactory(object):
    def __init__(self, module, onnx_model_parameters=None, device=None):
        self._training_manager = OnnxTrainingManager(module, onnx_model_parameters, device)
        self._inference_manager = OnnxInferenceManager(module, onnx_model_parameters, device)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager