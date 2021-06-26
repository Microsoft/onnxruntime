# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _io
from ._torch_module_manager import TorchModuleManager
from ._onnx_module_manager import OnnxTorchModuleManager
from onnxruntime.training import register_custom_ops_pytorch_exporter

import copy
import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable
import onnx

# Needed to override PyTorch methods
T = TypeVar('T', bound='Module')

def _validate_module(module):
    '''Raises a TypeError if the module is not a torch.nn.Module'''

    if not isinstance(module, torch.nn.Module) and not isinstance(module, onnx.ModelProto):
        raise TypeError(f"ORTModule only support torch.nn.Module or ONNX as input. {type(module)} is not supported.")

class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module`'s APIs.
    """

    def __init__(self, module, device):
        _validate_module(module)

        if isinstance(module, torch.nn.Module):
            self._module_manager = TorchModuleManager(module)
        else:
            self._module_manager = OnnxTorchModuleManager(module, device)

        # Create forward dynamically, so each ORTModule instance will have its own copy.
        # This is needed to be able to copy the forward signatures from the original PyTorch models
        # and possibly have different signatures for different instances.
        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`
            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''

            return self._module_manager.forward(*inputs, **kwargs)

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the _module_manager's forward signature.
        functools.update_wrapper(
            self.forward.__func__, self._module_manager.forward.__func__)

        super(ORTModule, self).__init__()

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op(is_ortmodule=True)

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        '''Dummy documentation for forward method'''
        ...

    def _apply(self, fn):
        """Override original method to delegate execution to the flattened PyTorch user module"""

        self._module_manager._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        self._module_manager.apply(fn)
        return self

    def _is_training(self):
        return self._module_manager.is_training()

    def train(self: T, mode: bool = True) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        self.training = mode
        # Since _modules is empty, the task needs to be delegated to _module_manager
        # which will recursively update the original module
        self._module_manager.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override original method to delegate execution to the base module"""

        return self._module_manager.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override original method to delegate execution to the original PyTorch user module"""

        return self._module_manager.load_state_dict(state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""

        self._module_manager.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""

        self._module_manager.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override original method to delegate execution to the original PyTorch user module"""

        return self._module_manager.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override original method to delegate execution to the original PyTorch user module"""

        return self._module_manager.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.named_buffers(prefix=prefix, recurse=recurse)

    def _replicate_for_data_parallel(self):
        """Raises a NotImplementedError exception since ORTModule is not compatible with torch.nn.DataParallel

        torch.nn.DataParallel requires the model to be replicated across multiple devices, and
        in this process, ORTModule tries to export the model to onnx on multiple devices with the same
        sample input. Because of this multiple device export with the same sample input, torch throws an
        exception that reads: "RuntimeError: Input, output and indices must be on the current device"
        which can be vague to the user since they might not be aware of what happens behind the scene.

        We therefore try to preemptively catch use of ORTModule with torch.nn.DataParallel and throw a
        more meaningful exception.

        Users must use torch.nn.parallel.DistributedDataParallel instead of torch.nn.DataParallel
        which does not need model replication and is also recommended by torch to use instead.
        """

        raise NotImplementedError("ORTModule is not compatible with torch.nn.DataParallel. "
                                  "Please use torch.nn.parallel.DistributedDataParallel instead.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        """Override original method to delegate execution to the original PyTorch user module"""

        self._module_manager._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                   missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.named_children()

    def modules(self) -> Iterator['Module']:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.modules()

    def named_modules(self, *args, **kwargs):
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_manager.named_modules(*args, **kwargs)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Raises a NotImplementedError exception since ORTModule does not support adding modules to it"""

        raise NotImplementedError("ORTModule does not support adding modules to it.")

    @property
    def module(self):
        """The original `torch.nn.Module` that this module wraps.

        This property provides access to methods and properties on the original module.
        """

        # HuggingFace Trainer `save_model` method checks to see if the input model is a HuggingFace PreTrainedModel
        # or if the model has an attribute called `module` which references a HuggingFace PreTrainedModel to save
        # the entire context of the model so that it can be loaded using HuggingFace `from_pretrained` method.
        # This `module` property enables HuggingFace Trainer to retrieve the underlying PreTrainedModel inside ORTModule
        # to save and load a complete checkpoint

        return self._module_manager.module
