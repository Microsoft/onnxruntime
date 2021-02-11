import copy
import io
import logging
import onnx
import onnxruntime
import os
import torch
import warnings
import numpy as np
from inspect import signature

from torch.utils.dlpack import from_dlpack

# Needed to re-implement PyTorch's cpu,cuda,to methods
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from onnxruntime.capi import _pybind_state as C
from . import _utils


ONNX_OPSET_VERSION = 12
__TEMP_ENABLE_METHOD_TIMING__ = False

# Needed to re-implement PyTorch's cpu,cuda,to methods
T = TypeVar('T', bound='Module')


def _get_device_index(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index

def _get_device_str(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        if device.find(':') == -1:
            device += ':' + str(torch.cuda.current_device())
    elif isinstance(device, int):
        device = 'cuda:' + str(device)
    elif isinstance(device, torch.device):
        if device.index is None:
            device = device.type + ':' + str(torch.cuda.current_device())
        else:
            device = device.type + ':' + str(device.index)
    else:
        raise ('Unsupported device type')
    return device

def _get_default_device_str(type):
    if type == 'cuda':
        return 'cuda:' + str(torch.cuda.current_device())
    else:
        return 'cpu'

def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_input(value_info.name, inputs[idx].device.type,
                              _get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype),
                              list(inputs[idx].size()),
                              inputs[idx].data_ptr())

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type, device_id=_get_device_index(device))

def _onnx_value_info_to_buffer_tensor(value_info, device):
    '''Create a torch zeroed tensor with the same shape and type of `value_info`'''

    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    dtype = _utils.dtype_onnx_to_torch(value_info.type.tensor_type.elem_type)
    return torch.zeros(shape, device=device, dtype=dtype)

# TODO: PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
# and convert the config to torch.uint8 tensor duing from_dlpack(). So a boolean tensor
# from forward graph outputs will be converted to torch.uint8 tensor. When this tensor
# is feeded to backward graph as input, it will cause data type mismatch issue during
# inference session running. We cannot change the from_dlpack() in PyTorch side, so we
# have to handle this specially, which will introduce a cast here and there is data copied.
# Always cast from torch.uint8 to torch.bool is not logically right, we need to check the
# real data type of the inputs in the backeard graph, and perform the cast only necessary.
def _ort_output_to_torch_tensor(ort_output):
    tensor = from_dlpack(ort_output.to_dlpack())
    return tensor.to(torch.bool) if tensor.dtype == torch.uint8 else tensor

def _extract_input_information(module, *inputs, **kwargs):
    '''Returns the all input names, dynamic_axes information and input names that require gradient'''

    # Ignore optional *inputs explicitly specified as None
    sig = signature(module.forward)
    all_input_names = sig.parameters.keys()
    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    for input_idx, name in enumerate(all_input_names):
        if input_idx < len(inputs) and inputs[input_idx] is not None:
            if inputs[input_idx].requires_grad:
                # input_names_require_grad holds all input tensors that have requires_grad
                input_names_require_grad.append(name)

            input_names.append(name)
            dynamic_axes[name] = {}
            for dim_idx in range(len(inputs[input_idx].shape)):
                dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})

    return input_names, dynamic_axes, input_names_require_grad

class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(module, torch.nn.Module), "'module' must be a torch.nn.Module"
        super(ORTModule, self).__init__()

        # TODO: This is incorrect when different layers may be in different devices
        self._device = next(module.parameters()).device
        self._device_changed = False

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._onnx_inference = None
        self._is_training = True

        # Related to training graph shape inference
        self._current_input_shape = None
        self._module_gradient_graph_builder = None
        self._input_names_require_grad = None

        # Training model
        self._onnx_training = None
        self._training_session = None
        self._training_io_binding = None
        self._run_options = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

    def _initialize_module_gradient_graph_builder(self):
        # TODO: PyTorch exporter bug: changes the initializer order
        initializer_names = [p[0] for p in self._original_module.named_parameters()]

        # Build full training graph
        grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
        grad_builder_config.initializer_names_to_train = initializer_names
        grad_builder_config.input_names_require_grad = self._input_names_require_grad
        self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
        self._module_gradient_graph_builder.initialize(self._onnx_inference.SerializeToString(), grad_builder_config)

    def _get_inference_graph_and_init_gradient_graph_builder(self, *inputs, **kwargs):
        input_names, dynamic_axes, self._input_names_require_grad = \
                _extract_input_information(self._original_module, *inputs, **kwargs)
        self._onnx_inference = self._get_forward_graph(input_names, dynamic_axes, *inputs, **kwargs)

        if self._save_onnx:
            onnx.save(self._onnx_inference, self._save_onnx_prefix + '_inference.onnx')

        self._initialize_module_gradient_graph_builder()

    def _create_training_session(self):
        providers = None
        provider_options = None
        if self._device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.use_deterministic_compute = False
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = 1

        self._training_session = onnxruntime.InferenceSession(
            self._onnx_training.SerializeToString(), session_options, providers=providers, provider_options=provider_options)
        
        # Use this global run_options for now
        self._run_options = C.RunOptions()

        # IO binding
        # TODO: we should try to reuse the output buffers as some of the output tensors are same sizes, expecially the backward graph outputs.
        self._training_io_binding = self._training_session.io_binding()

    def _build_training_graph(self, *inputs, **kwargs):
        self._module_gradient_graph_builder.build(self._current_input_shape)
        self._onnx_training = onnx.load_model_from_string(self._module_gradient_graph_builder.get_training_model())
        self._onnx_graphs_info = self._module_gradient_graph_builder.get_training_graph_info()

        if self._save_onnx:
            onnx.save(self._onnx_training, self._save_onnx_prefix + '_training.onnx')

    def cpu(self: T) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if self._device.type != 'cpu':
            self._device_changed = True
            self._device = torch.device('cpu')

        return super(ORTModule, self).cpu()

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if device is None:
            if _get_device_str(self._device) != _get_default_device_str('cuda'):
                self._device_changed = True
                self._device = torch.device(_get_default_device_str('cuda'))
        elif _get_device_str(self._device) != _get_device_str(device):
            self._device_changed = True
            self._device = torch.device(_get_device_str(device))

        return super(ORTModule, self).cuda(device)

    @overload
    def to(self: T, device: Optional[Union[int, torch.device]] = ...,
           dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[torch.dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        '''Thin layer to capture device for ORTModule IO bindings'''

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device:
            device_str = _get_device_str(device)
            if _get_device_str(self._device) != device_str:
                self._device_changed = True
                self._device = torch.device(device_str)
        return super(ORTModule, self).to(*args, **kwargs)

    def eval(self: T) -> T:
        self._is_training = False
        self._original_module.eval()

    def train(self: T, mode: bool = True) -> T:
        self._is_training = mode
        self._original_module.train(mode)

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, we build a full training graph with module_gradient_graph_builder. 
        Finally, we instantiate the ONNX Runtime InferenceSession.
        '''
        # TODO: using pytorch for evaluation for now. We will use ORT for evaluation latter. 
        if not self._is_training:
            return self._original_module(*inputs, **kwargs)

        # Exporting module to ONNX for the first time
        if not self._onnx_inference:
            self._get_inference_graph_and_init_gradient_graph_builder(*inputs, **kwargs)

        _, _, input_names_require_grad = _extract_input_information(self._original_module, *inputs, **kwargs)
        # If inputs requiring gradient change from one call to forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_names_require_grad != self._input_names_require_grad:
            self._input_names_require_grad = input_names_require_grad
            self._initialize_module_gradient_graph_builder()

        new_input_shape = [list(input.size()) for input in inputs if input is not None]
        if self._current_input_shape is None or self._current_input_shape != new_input_shape:
            self._current_input_shape = new_input_shape
            self._build_training_graph()
            self._create_training_session()
        # TODO: disabled for now, since it caused a bug in NVBert fp32 run
        # When creating a new InferenceSession, there is a bug for destructing the original InferenceSession 
        # elif self._device_changed:
        #     self._create_training_session()
        #     self._device_changed = False

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                '''Performs forward pass based on user input and PyTorch initializer

                TODO: **kwargs are not supported

                Module outputs are returned to the user
                '''

                # Use IO binding
                _create_iobinding(self._training_io_binding, inputs, self._onnx_training, self._device)

                # Run and return module outputs.
                user_outputs = tuple(_ort_output_to_torch_tensor(forward_output) \
                    for forward_output in self._training_session.run_forward(self._training_io_binding, self._run_options))
                return user_outputs[0] if len(user_outputs) == 1 else user_outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt module output
                '''

                # Use IO binding
                # Push user output grads to ONNX backend.
                backward_grad_output_ortvalue = []
                for grad_output in grad_output[:len(self._onnx_graphs_info.backward_output_grad_names)]:
                    backward_grad_output_ortvalue.append(onnxruntime.OrtValue.ortvalue_from_data_ptr(list(grad_output.size()), _utils.dtype_torch_to_numpy(
                        grad_output.dtype), grad_output.device.type, _get_device_index(grad_output.device), grad_output.data_ptr()))

                # Run and get results
                self._training_session.run_backward(backward_grad_output_ortvalue)
                backward_outputs = self._training_io_binding.get_outputs()

                # Return input and initializer gradients
                num_user_input_grads = len(self._input_names_require_grad)

                results = []
                for input_name in self._onnx_graphs_info.user_input_names:
                    try:
                        # Append to the results the backward output for each input that required grad
                        results.append(_ort_output_to_torch_tensor(
                            backward_outputs[self._input_names_require_grad.index(input_name)]))
                    except ValueError:
                        # input_name is not found in the self._input_names_require_grad list
                        # Append None to results for each input that did not require grad
                        results.append(None)
                # Append gradients of initializer to results
                results += [_ort_output_to_torch_tensor(backward_output) 
                            for backward_output in backward_outputs[num_user_input_grads:]]
                return tuple(results)

        proc_inputs = [data for data in inputs if data is not None]

        return _ORTModuleFunction.apply(*self._convert_training_graph_input_to_list(*proc_inputs, **kwargs))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_training_graph_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: **kwargs is not supported
        TODO: How IO binding model inputs and outputs affects initializer copies?

        ONNX Runtime forward requires an order list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters

        This codes assumes the exported model's inputs and initializers
            are the same as the original PyTorch model
        '''
        # User inputs
        result = list(inputs[:len(self._onnx_graphs_info.user_input_names)])

        # Initializers
        for param in self._original_module.named_parameters():
            result.append(param[1])

        return result

    def _get_forward_graph(self, input_names, dynamic_axes, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''
        # Export the model to memory
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy = []
        for model_input in inputs:
            sample_inputs_copy.append(model_input.data if isinstance(model_input, torch.Tensor) else model_input)
        sample_inputs_copy = copy.deepcopy(tuple(sample_inputs_copy))

        # TODO: Support contrib OPs support? user model has no hint
        # from onnxruntime.training import register_custom_ops_pytorch_exporter
        # register_custom_ops_pytorch_exporter.register_custom_op()

        with torch.no_grad():
            # Export torch.nn.Module to ONNX
            torch.onnx.export(self._original_module,
                              sample_inputs_copy,
                              f,
                              input_names=input_names,
                              opset_version=ONNX_OPSET_VERSION,
                              do_constant_folding=False,
                              training=torch.onnx.TrainingMode.TRAINING,
                              dynamic_axes=dynamic_axes)

        # TODO: this step might not be needed when we use the torch external allocator
        # clear cache after model export
        torch.cuda.empty_cache()

        return onnx.load_model_from_string(f.getvalue())
