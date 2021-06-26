# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_manager.py

from . import _io
from ._graph_execution_manager_factory import OnnxGraphExecutionManagerFactory
from ._module_manager_interface import ModuleManagerInterface

import functools
import copy
import onnx
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable

T = TypeVar('T', bound='Module')

class OnnxTorchModuleManager(ModuleManagerInterface):
    def __init__(self, module: onnx.ModelProto, device, is_train: bool = True):
        super(OnnxTorchModuleManager, self).__init__(module)

        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''

            return self._execution_manager(self.is_training()).forward(*inputs, **kwargs)

        # Bind the forward method.
        self.forward = _forward.__get__(self)

        self._training = is_train
        self._onnx_model_parameters = [
                (initializer.name, torch.nn.Parameter(torch.as_tensor(copy.deepcopy(onnx.numpy_helper.to_array(initializer)))))
                        for initializer in self._original_module.graph.initializer]
        self._execution_manager = OnnxGraphExecutionManagerFactory(self._original_module, self._onnx_model_parameters, device)

    def is_training(self):
        return self._training and torch.is_grad_enabled()

    def train(self: T, mode: bool = True) -> T:
        self._training = mode
        return self

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        for _, parameter in self._onnx_model_parameters:
                yield parameter

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]: 
        yield from self._onnx_model_parameters


