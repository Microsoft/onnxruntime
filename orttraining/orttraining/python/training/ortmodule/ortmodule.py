# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._torch_module_factory import TorchModuleFactory

from onnxruntime.training import register_custom_ops_pytorch_exporter

import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable

# Needed to override PyTorch methods
T = TypeVar('T', bound='Module')

class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module`'s APIs.
    """

    def __init__(self, module):
        self._torch_module = TorchModuleFactory()(module)

        # Create forward dynamically, so each ORTModule instance will have its own copy.
        # This is needed to be able to copy the forward signatures from the original PyTorch models
        # and possibly have different signatures for different instances.
        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''

            return self._torch_module.forward(*inputs, **kwargs)

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the _torch_module's forward signature.
        functools.update_wrapper(
            self.forward.__func__, self._torch_module.forward.__func__)

        super(ORTModule, self).__init__()

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op(is_ortmodule=True)

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        '''Delegate the :meth:`~torch.nn.Module.forward` pass of PyTorch training to
        ONNX Runtime.

        The first call to forward performs setup and checking steps. During this call,
        ORTModule determines whether the module can be trained with ONNX Runtime. For
        this reason, the first forward call execution takes longer than subsequent calls.
        Execution is interupted if ONNX Runtime cannot process the model for training.

        Args:
            *inputs and **kwargs represent the positional, variable positional, keyword
            and variable keyword arguments defined in the user's PyTorch module's forward
            method. Values can be torch tensors and primitive types.

        Returns:
            The output as expected from the forward method defined by the user's
            PyTorch module. Output values supported include tensors, nested sequences
            of tensors and nested dictionaries of tensor values.
        '''

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

        return self._torch_module.module

    ################################################################################
    # The methods below are part of torch.nn.Module API that are encapsulated through
    # TorchModuleInterface
    ################################################################################

    def _apply(self, fn):
        """Override original method to delegate execution to the flattened PyTorch user module"""

        self._torch_module._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        """Override :meth:`~torch.nn.Module.apply` to delegate execution to ONNX Runtime"""

        self._torch_module.apply(fn)
        return self

    def _is_training(self):
        return self._torch_module.is_training()

    def train(self: T, mode: bool = True) -> T:
        """Override :meth:`~torch.nn.Module.train` to delegate execution to ONNX Runtime"""

        self.training = mode
        # In a torch.nn.Module, _modules stores all dependent modules (sub-modules) of the current module.
        # in a list so that torch.nn.Module can apply any changes to all sub-modules recursively.
        # Although the _flattened_module and _original_module are dependent modules for ORTModule,
        # they do not show up in _modules because they are abstracted away behind another class,
        # TorchModule. In order to apply changes to those sub-modules, delegate the task to _torch_module
        # which will recursively update the flattened_module and the original module.
        self._torch_module.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override :meth:`~torch.nn.Module.state_dict` to delegate execution to ONNX Runtime"""

        return self._torch_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override :meth:`~torch.nn.Module.load_state_dict` to delegate execution to ONNX Runtime"""

        return self._torch_module.load_state_dict(state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override :meth:`~torch.nn.Module.register_buffer`"""

        self._torch_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override :meth:`~torch.nn.Module.register_parameter`"""

        self._torch_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override :meth:`~torch.nn.Module.get_parameter`"""

        return self._torch_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override :meth:`~torch.nn.Module.get_buffer`"""

        return self._torch_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override :meth:`~torch.nn.Module.parameters`"""

        yield from self._torch_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override :meth:`~torch.nn.Module.named_parameters`"""

        yield from self._torch_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override :meth:`~torch.nn.Module.buffers`"""

        yield from self._torch_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override :meth:`~torch.nn.Module.named_buffers`"""

        yield from self._torch_module.named_buffers(prefix=prefix, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        """Override original method to delegate execution to the original PyTorch user module"""

        self._torch_module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                   missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Override :meth:`~torch.nn.Module.named_children`"""

        yield from self._torch_module.named_children()

    def modules(self) -> Iterator['Module']:
        """Override :meth:`~torch.nn.Module.modules`"""

        yield from self._torch_module.modules()

    def named_modules(self, *args, **kwargs):
        """Override :meth:`~torch.nn.Module.named_modules`"""

        yield from self._torch_module.named_modules(*args, **kwargs)
