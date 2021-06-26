# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _module_manager_interface.py

from abc import ABC, abstractmethod
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable

T = TypeVar('T', bound='Module')

class ModuleManagerInterface:
    """Abstract class that provides the function signatures for the torch.nn.Module

    Concrete implementations should inherit from this class and provide necessary executions.
    """
    def __init__(self, module):
        self._original_module = module

    @abstractmethod
    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        """
        pass

    @abstractmethod
    def _apply(self, fn):

        pass

    @abstractmethod
    def apply(self: T, fn: Callable[['Module'], None]) -> T:

        pass

    @abstractmethod
    def is_training(self):

        pass

    @abstractmethod
    def train(self: T, mode: bool = True) -> T:

        pass

    @abstractmethod
    def state_dict(self, destination=None, prefix='', keep_vars=False):

        pass

    @abstractmethod
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):

        pass

    @abstractmethod
    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:

        pass

    @abstractmethod
    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:

        pass

    @abstractmethod
    def get_parameter(self, target: str) -> torch.nn.Parameter:

        pass

    @abstractmethod
    def get_buffer(self, target: str) -> torch.Tensor:

        pass

    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:

        pass

    @abstractmethod
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:

        pass

    @abstractmethod
    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:

        pass

    @abstractmethod
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:

        pass

    @abstractmethod
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        pass

    @abstractmethod
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:

        pass

    @abstractmethod
    def modules(self) -> Iterator['Module']:

        yield from self._module_manager.modules()

    @abstractmethod
    def named_modules(self, *args, **kwargs):

        pass

    @property
    def module(self):
        """The original user provided module that this class manages.

        This property provides access to methods and properties on the original module.
        """

        return self._original_module
