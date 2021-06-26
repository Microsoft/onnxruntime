# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_manager.py

from . import _io
from ._graph_execution_manager_factory import GraphExecutionManagerFactory
from ._module_manager_interface import ModuleManagerInterface

import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable

T = TypeVar('T', bound='Module')

class TorchModuleManager(ModuleManagerInterface):
    def __init__(self, module: torch.nn.Module):
        super(TorchModuleManager, self).__init__(module)
        self._flattened_module = _io._FlattenedModule(module)

        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''

            return self._execution_manager(self.is_training()).forward(*inputs, **kwargs)

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the PyTorch module.
        functools.update_wrapper(
            self.forward.__func__, self._original_module.forward.__func__)

        self._execution_manager = GraphExecutionManagerFactory(self._flattened_module)

    def _apply(self, fn):
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Delegation must happen to _flattened_module since methods depend on
        # _apply to recursively apply the internal setting changes
        self._flattened_module._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Delegation must happen to _flattened_module since methods depend on
        # apply to recursively apply the internal setting changes
        self._flattened_module.apply(fn)
        return self

    def is_training(self):
        return self._flattened_module.training and torch.is_grad_enabled()

    def train(self: T, mode: bool = True) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Delegate the task to _module.flattened_module.train which will recursively
        # update the original_module
        self._flattened_module.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override original method to delegate execution to the original PyTorch user module"""

        # Override the state_dict() method so that the state dict key names
        # do not contain the flattened_module._original_module prefix
        return self._original_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override original method to delegate execution to the original PyTorch user module"""

        # Override the load_state_dict() method so that the loaded state dict
        # key names does not need to contain the _module.flattened_module._original_module prefix
        return self._original_module.load_state_dict(
            state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""

        self._original_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""

        self._original_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override original method to delegate execution to the original PyTorch user module"""

        return self._original_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override original method to delegate execution to the original PyTorch user module"""

        return self._original_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.named_buffers(prefix=prefix, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override original method to delegate execution to the original PyTorch user module"""

        # PyTorch load_state_dict implementation does not recursively call load_state_dict on its sub-modules. 
        # Instead, it creates a recursive function and invokes _load_from_state_dict on all child modules.
        # For the scenario where an ORTModule is a sub-module of another module, loading of the state
        # dictionary requires the _load_from_state_dict to be overridden to prevent an error.

        self._original_module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                    missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.named_children()

    def modules(self) -> Iterator['Module']:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._original_module.modules()

    def named_modules(self, *args, **kwargs):
        """Override original method to delegate execution to the original PyTorch user module"""

        # PyTorch >1.8.1 has an extra arg remove_duplicate that is not present in 1.8.1
        # To support both, use args and kwargs (since user can call the method with only positional args or kwargs)
        yield from self._original_module.named_modules(*args, **kwargs)
