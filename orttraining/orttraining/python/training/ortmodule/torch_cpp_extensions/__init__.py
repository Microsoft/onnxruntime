# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Support for PyTorch C++ extensions within ORTModule


TODO: Implement mechanism to register extensions and prevent issues with incorrect/missing flags
      for each :meth:`torch.utils.cpp_extension.*` call
"""

import threading
from functools import wraps
from onnxruntime.capi import _pybind_state as C
from torch.onnx import register_custom_op_symbolic
from ._aten_op_factory import ATenOpFactory


_onnx_opset_version = 1


def run_once_aten_op_executor(f):
    """
    Decorator to run a function only once.
    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """
    @wraps(f)
    def aten_op_executor_wrapper(*args, **kwargs):
        if not aten_op_executor_wrapper.has_run:
            with aten_op_executor_wrapper.lock:
                if not aten_op_executor_wrapper.has_run:
                    aten_op_executor_wrapper.has_run = True
                    return f(*args, **kwargs)

    aten_op_executor_wrapper.lock = threading.Lock()
    aten_op_executor_wrapper.has_run = False
    return aten_op_executor_wrapper


@run_once_aten_op_executor
def _load_aten_op_executor_cpp_extension():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import aten_op_executor
    C.register_aten_op_executor(str(aten_op_executor.is_tensor_argument_address()),
                                str(aten_op_executor.execute_aten_operator_address()),
                                str(aten_op_executor.get_gradient_definition_address()))

    for key, gradient_def_json in ATenOpFactory._GRADIENTS.items():
        aten_op_executor.set_gradient_definition(key[0], key[1], gradient_def_json)


def _load_aten_op_executor_cpp_extension_if_needed(onnx_model):
    for node in onnx_model.graph.node:
        if node.op_type == 'ATenOp' and node.domain == 'com.microsoft':
            _load_aten_op_executor_cpp_extension()
            break


def _register_custom_op_symbolic_for_aten_op():
    for key, fn in ATenOpFactory._SYMBOLICS.items():
        # Symbolic name is in format: domain::name
        register_custom_op_symbolic(key[1] + '::' + key[0], fn, _onnx_opset_version)
