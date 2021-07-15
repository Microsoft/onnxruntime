# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Support registration of ATenOp's symbolic (PyTorch exporter overriding) and gradient definition.

Each gradient definition of ATenOp is a list of node definitions: [node_def1, node_def2, ...].
    Each node definition is a tuple: (op_type, inputs, outputs, attributes), while attributes is optional.
        'op_type' is a string or a tuple of two strings with op_name and domain respectively.
            If it's string type, then the domain is the default ONNX domain: ''.
        'inputs' is a list of strings. Each input string can be GO(i), I(i), O(i),
            which means ith gradient output, ith input, ith output of forward node respectively,
            or any string for intermediate output of one of other nodes.
        'outputs' is a list of strings. Each input string can be GI(i),
            which means ith gradient input of forward node,
            or any string for intermediate output for some of other nodes.
        'attributes' (if present) is a dictionary. Each entry's key is the attribute name,
            the value is also a dictionary: {'value': v, 'dtype': t, 'is_tensor': b}.
                'v' can be string, number, or list of numbers, which is the value of the attribute.
                't' is a string to describe the element type. It can be 'int', 'float', 'bool', etc,
                    or 'IElemType(i)', 'OElemType(i)', which means the same type of ith input or
                    output of forward node.
                'b' is True or False, indicating if this attribute is tensor_proto ot not.
                    'is_tensor' is optional, if not present, the default is False.
"""

import json
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes


def gradient_definition_to_json(gradient_def):
    nodes = []
    for node_def in gradient_def:
        node_json = {}
        # Op type
        if isinstance(node_def[0], str):
            node_json['op_type'] = node_def[0]
        else:
            node_json['op_type'] = node_def[0][0]
            node_json['domain'] = node_def[0][1]
        # Inputs and outputs
        node_json['inputs'] = node_def[1]
        node_json['outputs'] = node_def[2]
        # Attributes
        if len(node_def) >= 4:
            node_json['attributes'] = node_def[3]
        nodes.append(node_json)
    return json.dumps(nodes)


class ATenOpFactory:
    _SYMBOLICS = {}
    _GRADIENTS = {}

    @classmethod
    def register_symbolic(cls, name, domain=''):
        def symbolic_wrapper(fn):
            cls._SYMBOLICS[(name, domain)] = fn
            return fn
        return symbolic_wrapper

    @classmethod
    def register_gradient(cls, op_name, overload_name=''):
        def gradient_wrapper(fn):
            cls._GRADIENTS[(op_name, overload_name)] = gradient_definition_to_json(fn())
            return fn
        return gradient_wrapper


@ATenOpFactory.register_symbolic('embedding')
def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    output = g.op("com.microsoft::ATenOp", weight, indices, padding_idx, scale_grad_by_freq, sparse,
                  name_s='aten::embedding')
    indices_shape = _get_tensor_sizes(indices)
    if indices_shape is not None and hasattr(weight.type(), 'with_sizes'):
        output_type = weight.type().with_sizes(
            indices_shape + [_get_tensor_dim_size(weight, 1)])
        output.setType(output_type)
    return output


@ATenOpFactory.register_gradient('aten::embedding')
def embedding_gradient():
    return [
        ('Constant', [], ['Const_0'], {'value': {'value': 0, 'dtype': 'int', 'is_tensor': True}}),
        ('Shape', ['I(0)'], ['Shape_X']),
        ('Gather', ['Shape_X', 'Const_0'], ['Gather_X_0'], {'axis': {'value': 0, 'dtype': 'int'}}),
        (('ATenOp', 'com.microsoft'), ['GO(0)', 'I(1)', 'Gather_X_0', 'I(2)', 'I(3)', 'I(4)'], [
         'GI(0)'], {'name': {'value': 'aten::embedding_backward', 'dtype': 'string'}}),
    ]


@ATenOpFactory.register_symbolic('max_pool2d')
def max_pool2d(g, self, kernel_size, stride, padding, dilation, ceil_mode):
    return g.op("com.microsoft::ATenOp", self, kernel_size, stride, padding, dilation, ceil_mode,
                name_s='aten::max_pool2d_with_indices', outputs=2)[0]


@ATenOpFactory.register_gradient('aten::max_pool2d_with_indices')
def max_pool2d_gradient():
    return [
        (('ATenOp', 'com.microsoft'), ['GO(0)', 'I(0)', 'I(1)', 'I(2)', 'I(3)', 'I(4)', 'I(5)', 'O(1)'], [
         'GI(0)'], {'name': {'value': 'aten::max_pool2d_with_indices_backward', 'dtype': 'string'}}),
    ]


@ATenOpFactory.register_symbolic('unfold')
def unfold(g, input, dimension, size, step):
    return g.op("com.microsoft::ATenOp", input, dimension, size, step, name_s='aten::unfold')


@ATenOpFactory.register_gradient('aten::unfold')
def unfold_gradient():
    return [
        ('Shape', ['I(0)'], ['Shape_X']),
        (('ATenOp', 'com.microsoft'), ['GO(0)', 'Shape_X', 'I(1)', 'I(2)', 'I(3)'], [
         'GI(0)'], {'name': {'value': 'aten::unfold_backward', 'dtype': 'string'}}),
    ]


@ATenOpFactory.register_symbolic('argmax')
def argmax(g, input, dim, keepdim):
    return g.op("com.microsoft::ATenOp", input, dim, keepdim, name_s='aten::argmax')
