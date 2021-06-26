# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import abc
import copy
import inspect
import torch
import warnings
import gc

class _OutputIdentityOp(torch.autograd.Function):
    '''Internal class used to prepend Identity ops in model's outputs

    This class is required to support ONNX models which passthrough [some of] the models's inputs
    directly to the graph output. This is an issue because ONNX Runtime cannot build proper
    gradient graph based on this pattern.

    Adding a direct Identity Op to the user model doesn't work as the ONNX exporter would optimize it away,
    resulting in the same issue.

    Therefore a custom Autograd function was introduced to add an Identity right before the output
    in a way the ONNX exporter will not optimize it away.

    Given the model below

    .. code-block:: python

        class PassthroughNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(PassthroughNet, self).__init__()
                self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
                self.relu1 = torch.nn.ReLU()
                self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)
            def forward(self, input1, passthrough_input):
                out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
                # use shape from passthrough_input
                out1 = out1.view(passthrough_input.size()[0], -1)
                return out1, passthrough_input

    We can see `passthrough_input` is part of both model input and output and the resulting
    ONNX subgraph would contain something like `output2 -> output2`.

    By prepending each model output to an :class:`_OutputIdentityOp` op, the resulting
    onnx subgraph for this example would be  `passthrough_input -> Identity -> output2`.

    TODO: Remove once PyTorch 1.8.2 or newer is released
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.nn.Identity()(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    @staticmethod
    def symbolic(g, self):
        return g.op("Identity", self)

class _PrimitiveType(object):
    _primitive_types = {int, bool, float}
    @staticmethod
    def is_primitive_type(value):
        return type(value) in _PrimitiveType._primitive_types

    @staticmethod
    def get_tensor(value, device):
        return torch.tensor(value, device=device)

    @staticmethod
    def get_primitive_dtype(value):
        # If `value` is a boolean, save the value of the boolean in dtype.
        # This way, if the value changes from one forward call to the next, the schema will mismatch,
        # and the model will be re-exported.
        return f"{str(type(value))}_{value}" if isinstance(value, bool) else str(type(value))

class _InputInfo(object):
    def __init__(self,
                 names,
                 shape,
                 require_grad_names=None,
                 dynamic_axes=None,
                 schema=None,
                 num_positionals=0,
                 num_positionals_non_none=0,
                 keyword_names=None):
        self.names = names
        self.shape = shape
        self.require_grad_names = require_grad_names if require_grad_names else []
        self.dynamic_axes = dynamic_axes if dynamic_axes else {}
        self.schema = schema if schema else []
        self.num_positionals = num_positionals
        self.num_positionals_non_none = num_positionals_non_none
        self.keyword_names = keyword_names

    def __repr__(self) -> str:
        return f'''_InputInfo class:
            \tNames:                   {self.names}
            \tShape:                   {self.shape}
            \tRequire gradient:        {self.require_grad_names}
            \tDynamic axes:            {self.dynamic_axes}
            \tSchema:                  {self.schema}
            \t#Positionals (total):    {self.num_positionals}
            \t#Positionals (non-None): {self.num_positionals_non_none}
            \tKeyword names:           {self.keyword_names}'''

    def flatten(self, args, kwargs, device):
        '''Flatten args and kwargs in a single tuple of tensors with strict ordering'''

        ret = [_PrimitiveType.get_tensor(arg, device) if _PrimitiveType.is_primitive_type(arg) else arg for arg in args]
        ret += [_PrimitiveType.get_tensor(kwargs[name], device) if _PrimitiveType.is_primitive_type(kwargs[name])
            else kwargs[name] for name in self.names if name in kwargs]

        return ret

    def unflatten(self, flat_args):
        '''Unflatten tuple of tensors into args and kwargs'''

        args = tuple(flat_args[:self.num_positionals])
        kwargs = {name: arg for name, arg in zip(self.names[self.num_positionals_non_none:], flat_args[self.num_positionals:]) \
            if name in self.keyword_names}
        return args, kwargs

def _combine_input_buffers_initializers(params, onnx_input_names, input_info, buffer_names, inputs, kwargs, device):
    '''Creates forward `*inputs` list from user input and PyTorch initializers

    ONNX Runtime forward requires an ordered list of:
        * User input: computed from forward InferenceSession
        * Initializers: computed from original PyTorch model parameters
    '''

    # User inputs
    non_none_inputs = [inp for inp in inputs if inp is not None]
    buffer_names_dict = {buffer_name: inp for buffer_name, inp in buffer_names}
    result = []

    for input_idx, name in enumerate(onnx_input_names):
        inp = None
        if name in kwargs and kwargs[name] is not None:
            # Only use keywords coming from user that are expected by ONNX model
            inp = kwargs[name]

        if inp is None:
            try:
                # Only use positionals coming from user that are expected by ONNX model
                # if input_idx >= len(input_info.names), IndexError will be thrown
                if name != input_info.names[input_idx]:
                    # When ONNX drops unused inputs, get correct index from user input
                    # if name is not in input_info.names, ValueError will be thrown
                    input_idx = input_info.names.index(name)
                inp = non_none_inputs[input_idx]
            except (IndexError, ValueError):
                # ONNX input name is not present in input_info.names.
                pass

        if inp is None:
            # Registered buffers are translated to user_input+initializer in ONNX
            try:
                inp = buffer_names_dict[name]
            except KeyError:
                # ONNX input name is not present in the registered buffer dict.
                pass

        if inp is not None:
            if _PrimitiveType.is_primitive_type(inp):
                inp = _PrimitiveType.get_tensor(inp, device)
            result.append(inp)
        else:
            raise RuntimeError(f'Input is present in ONNX graph but not provided: {name}.')

    # params is a list of all initializers known to the onnx graph
    result.extend(params)

    return result


def deepcopy_model_input(*inputs, **kwargs):
    def extract_tensor(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                return value.data.requires_grad_()
            else:
                return value.data
        else:
            return value
    sample_inputs_copy = [extract_tensor(value) for value in inputs]
    sample_inputs_copy = copy.deepcopy(tuple(sample_inputs_copy))

    sample_kwargs_copy = {}
    for name, value in kwargs.items():
        sample_kwargs_copy[name] = extract_tensor(value)
    sample_kwargs_copy = copy.deepcopy(sample_kwargs_copy)

    return sample_inputs_copy, sample_kwargs_copy


class _TensorStub(object):
    '''Tensor stub class used to represent model's input or output'''

    def __init__(self, name=None, dtype=None, shape=None, shape_dims=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shape_dims = shape_dims

    def __repr__(self) -> str:
        result = '_TensorStub('
        if self.name is not None:
            result += f'name={self.name}'
        if self.dtype is not None:
            if result[-1] != '(':
                result += ', '
            result += f'dtype={self.dtype}'
        if self.shape is not None:
            if result[-1] != '(':
                result += ', '
            result += f'shape={self.shape}'
        if self.shape_dims is not None:
            if result[-1] != '(':
                result += ', '
            result += f'shape_dims={self.shape_dims}'
        result += ')'
        return result

    def __eq__(self, other):
        if not other:
            return False
        elif not isinstance(other, _TensorStub):
            raise NotImplemented('_TensorStub must only be compared to another _TensorStub instance!')
        elif self.name != other.name:
            return False
        elif self.dtype != other.dtype:
            return False
        elif self.shape != other.shape:
            return False
        elif self.shape_dims != other.shape_dims:
            return False
        return True


def unflatten_user_output(output_schema, outputs):
    """Follows the schema to generate an output that is expected by the user"""

    def _replace_stub_with_tensor_value(user_output, outputs, output_idx):
        # Recursively traverse across user_output and replace all _TensorStub
        # with torch.Tensor values from outputs following output_idx

        # Output schema is None when ORTModule starts right with onnx model
        if isinstance(user_output, _TensorStub) or user_output is None:
            output_idx[0] += 1
            return outputs[output_idx[0]-1]

        if isinstance(user_output, abc.Sequence):
            sequence_type = type(user_output)
            user_output = list(user_output)
            for idx in range(len(user_output)):
                user_output[idx] = _replace_stub_with_tensor_value(user_output[idx], outputs, output_idx)
            try:
                # namedtuple can be created by passing the list sequence to method _make
                user_output = sequence_type._make(user_output)
            except AttributeError:
                # If attribute error encountered, create the sequence directly
                user_output = sequence_type(user_output)
        elif isinstance(user_output, abc.Mapping):
            for key in sorted(user_output):
                user_output[key] = _replace_stub_with_tensor_value(user_output[key], outputs, output_idx)
        else:
            raise TypeError(f'ORTModule does not support the following model output type {type(user_output)}.')

        return user_output

    # Replace every _TensorStub value in the schema with the torch.Tensor outputs calculated
    output_schema_copy = copy.deepcopy(output_schema)

    # It is expected that the outputs are ordered in the way defined in the exported onnx model
    # which is the order in which the output schema was saved.
    output_idx = [0]
    user_output = _replace_stub_with_tensor_value(output_schema_copy, outputs, output_idx)
    return user_output


def _extract_schema(data):
    """Extract the data schema by replacing every torch.Tensor value with _TensorStub"""

    if data is None:
        return None
    elif _PrimitiveType.is_primitive_type(data):
        return _TensorStub(dtype=_PrimitiveType.get_primitive_dtype(data), shape_dims=0)
    # Depth first traversal to iterate over the data to replace every tensor with a stub
    elif isinstance(data, torch.Tensor):
        return _TensorStub(dtype=str(data.dtype), shape_dims=len(data.size()))

    if isinstance(data, abc.Sequence) and not isinstance(data, str):
        sequence_type = type(data)
        data = list(data)
        for idx in range(len(data)):
            data[idx] = _extract_schema(data[idx])
        try:
            # namedtuple can be created by passing the list sequence to method _make
            data = sequence_type._make(data)
        except AttributeError:
            # If attribute error encountered, create the sequence directly
            data = sequence_type(data)
    elif isinstance(data, abc.Mapping):
        for key in sorted(data):
            data[key] = _extract_schema(data[key])
    else:
        raise TypeError(f'ORTModule does not support the following model data type {type(data)}')
    return data


def _parse_outputs_and_extract_names_and_dynamic_axes(module_output):
    """Parses through the module output and returns output names and dynamic axes"""

    def _populate_output_names_and_dynamic_axes(output, output_names, output_dynamic_axes, output_idx):
        # Depth first traversal to traverse through the entire output collecting output names and dynamic axes

        if output is None:
            return
        elif isinstance(output, torch.Tensor):
            # Naming the outputs with a hyphen ensures that there can be no input with the same
            # name, preventing collisions with other NodeArgs (for example an input to forward called output0)
            output_name = f'output-{output_idx[0]}'
            output_idx[0] += 1
            output_names.append(output_name)
            output_dynamic_axes[output_name] = {}
            for dim_idx in range(len(output.shape)):
                output_dynamic_axes[output_name].update({dim_idx: f'{output_name}_dim{dim_idx}'})
            return

        if isinstance(output, abc.Sequence):
            for value in output:
                _populate_output_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
        elif isinstance(output, abc.Mapping):
            for _, value in sorted(output.items()):
                _populate_output_names_and_dynamic_axes(value, output_names, output_dynamic_axes, output_idx)
        else:
            raise TypeError(f'ORTModule does not support the following model output type {type(output)}')

    output_names = []
    output_dynamic_axes = {}
    output_idx = [0]
    _populate_output_names_and_dynamic_axes(module_output, output_names, output_dynamic_axes, output_idx)

    return output_names, output_dynamic_axes


def _transform_output_to_flat_tuple(data):
    """Converts the data to a flat tuple by iterating over the entire data structure"""

    def _flatten_data(data, flat_data):
        # Recursively traverse over the data and populate the flat_data with torch.Tensors

        if data is None:
            return
        elif isinstance(data, torch.Tensor):
            identity = _OutputIdentityOp.apply
            flat_data.append(identity(data))
        elif isinstance(data, abc.Sequence):
            for value in data:
                _flatten_data(value, flat_data)
        elif isinstance(data, abc.Mapping):
            for _, value in sorted(data.items()):
                _flatten_data(value, flat_data)
        else:
            raise TypeError(f'ORTModule does not support the following data type {type(data)}.')

    flat_data = []
    _flatten_data(data, flat_data)
    return tuple(flat_data)


class _FlattenedModule(torch.nn.Module):
    def __init__(self, original_module):
        super(_FlattenedModule, self).__init__()
        self._original_module = original_module

        # Before `forward` is called, _ort_module must be assigned
        # Updated input info is needed to expand args into *args, **kwargs
        self._input_info = None

    def forward(self, *args):
        new_args, new_kwargs = self._input_info.unflatten(args)
        return _transform_output_to_flat_tuple(self._original_module(*new_args, **new_kwargs))


def parse_inputs_for_onnx_export(all_input_parameters, onnx_graph, inputs, kwargs):

    def _add_dynamic_shape(name, input):
        dynamic_axes[name] = {}
        for dim_idx in range(len(input.shape)):
            dynamic_axes[name].update({dim_idx: f'{name}_dim{dim_idx}'})
        return dynamic_axes

    def _add_input(name, input, onnx_graph, onnx_graph_input_names):
        if input is None:
            # Drop all None inputs.
            return

        # InputInfo should contain all the names irrespective of whether they are
        # a part of the onnx graph or not.
        input_names.append(name)

        if (onnx_graph is None or name in onnx_graph_input_names) and isinstance(input, torch.Tensor):
            if input.requires_grad:
                input_names_require_grad.append(name)
            dynamic_axes.update(_add_dynamic_shape(name, input))
            input_shape.append(list(input.size()))

    # Ignore optional inputs explicitly specified as None
    # ONNX exporter may remove unused inputs
    onnx_graph_input_names = []
    if onnx_graph is not None:
        onnx_graph_input_names = {inp.name for inp in onnx_graph.graph.input}

    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    input_shape = []
    var_positional_idx = 0

    for input_idx, input_parameter in enumerate(all_input_parameters):
        if input_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            # VAR_POSITIONAL parameter carries all *args parameters from original forward method

            for args_i in range(input_idx, len(inputs)):
                name = f'{input_parameter.name}_{var_positional_idx}'
                var_positional_idx += 1
                inp = inputs[args_i]
                _add_input(name, inp, onnx_graph, onnx_graph_input_names)
        elif input_parameter.kind == inspect.Parameter.POSITIONAL_ONLY or\
             input_parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or\
             input_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            # All positional non-*args and non-**kwargs are processed here
            name = input_parameter.name
            inp = None
            input_idx += var_positional_idx
            if input_idx < len(inputs) and inputs[input_idx] is not None:
                inp = inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
            _add_input(name, inp, onnx_graph, onnx_graph_input_names)
        elif input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs is always the last argument of forward()
            for name,inp in kwargs.items():
                if name not in input_names:
                    _add_input(name, inp, onnx_graph, onnx_graph_input_names)

    # Shallow copy is ok as we need the data structure, not the content
    schema = _extract_schema({'args': copy.copy(inputs), 'kwargs': copy.copy(kwargs)})

    return _InputInfo(names=input_names,
                      shape=input_shape,
                      require_grad_names=input_names_require_grad,
                      dynamic_axes=dynamic_axes,
                      schema=schema,
                      num_positionals=len(inputs),
                      num_positionals_non_none=len([i for i in inputs if i is not None]),
                      keyword_names=kwargs.keys())


def parse_outputs_for_onnx_export_and_extract_schema(module, inputs, kwargs):

    #   Do an inference to grab outputs
    is_train_mode = module.training
    module.eval()
    output_names = None
    output_dynamic_axes = None
    is_deepcopy = False
    with torch.no_grad():
        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy, sample_kwargs_copy = deepcopy_model_input(*inputs, **kwargs)
        try:
            # Deepcopy model, in case model is stateful and changes after model run.
            model_copy = copy.deepcopy(module)
            is_deepcopy = True
        except Exception:
            model_copy = module
            warnings.warn("This model cannot be deep copied (or pickled), "
                          "which is a required step for stateful models to be properly exported to ONNX."
                          " Compute will continue, but unexpected results may occur!")

        sample_outputs = model_copy(*sample_inputs_copy, **sample_kwargs_copy)

        # Parse the output and extract the output_names and output_dynamic_axes to be used for onnx export
        output_names, output_dynamic_axes = _parse_outputs_and_extract_names_and_dynamic_axes(sample_outputs)
    if is_train_mode:
        module.train()
    output_schema = _extract_schema(sample_outputs)
    if is_deepcopy:
        del model_copy
        gc.collect()
    # Return output names, output dynamic axes and output schema
    return output_names, output_dynamic_axes, output_schema
