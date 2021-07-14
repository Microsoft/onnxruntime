# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _io, _logger
from ._graph_execution_manager import GraphExecutionManager, _RunStateInfo, _FallbackLevel
from ._execution_agent import TrainingAgent

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import get_ort_device_type

import onnx
import torch
import warnings
from torch.utils.dlpack import from_dlpack, to_dlpack


class TrainingManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the training model

    TrainingManager is resposible for building and running the forward and backward graph of the training model
    """

    def __init__(self, model):
        super().__init__(model)
        self._export_mode = torch.onnx.TrainingMode.TRAINING

    @staticmethod
    def execution_session_run_forward(execution_session, onnx_model, device, *inputs):
        """Runs the forward graph on execution_session with given model inputs and device"""

        # Assert that the input and model device match
        _utils._check_same_device(device, "Input argument to forward", *inputs)

        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        state = C.PartialGraphExecutionState()
        forward_inputs = C.OrtValueVector()
        forward_inputs.reserve(len(inputs))
        for input in inputs:
            forward_inputs.push_back(to_dlpack(input), input.dtype == torch.bool)

        forward_outputs = C.OrtValueVector()
        # Run and return module outputs.
        execution_session.run_forward(forward_inputs, forward_outputs, state)
        user_outputs = tuple(_utils._ortvalue_to_torch_tensor(forward_output) for forward_output in forward_outputs)

        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = _RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, we build a full training graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession.
        '''

        # Fallback to PyTorch due to failures *external* to forward(),
        #  typically from initialization
        if self._pending_fallback():
            return self._apply_fallback(*inputs, **kwargs)

        try:
            # Exporting module to ONNX for the first time
            build_gradient_graph = self._export_model(*inputs, **kwargs)
            if build_gradient_graph:
                # If model was exported, then initialize the graph builder
                self._initialize_graph_builder(training=True)

            input_info = _io.parse_inputs_for_onnx_export(self._module_parameters,
                                                        self._onnx_model,
                                                        inputs,
                                                        kwargs)

            # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
            # Order of or operation is important here because we always need to call
            # _reinitialize_graph_builder irrespective of the value of build_gradient_graph.
            build_gradient_graph = self._reinitialize_graph_builder(input_info) or build_gradient_graph

            # Build the gradient graph
            if build_gradient_graph:
                self._build_graph()

            device = _utils.get_device_from_module(self._original_module) or \
                _utils.get_device_from_inputs(inputs, kwargs)
            # The _training_session/_inference_session should be created every time
            # the graph was built or if the device changed between calls to forward
            create_execution_session = build_gradient_graph or self._device != device
            if self._device != device:
                self._device = device
            if create_execution_session:
                # Create execution session creates the training_session
                self._create_execution_agent()

            class _ORTModuleFunction(torch.autograd.Function):
                '''Use a custom torch.autograd.Function to associate self.backward_graph as the
                gradient implementation for self.forward_graph.'''

                @staticmethod
                def forward(ctx, *inputs):
                    '''Performs forward pass based on user input and PyTorch initializer

                    Autograd Function's apply() doesn't support keyword arguments,
                    so `*inputs` has all the arguments - keyword arguments converted
                    to positional/keywords during `TrainingManager.forward`.

                    Module outputs are returned to the user
                    '''

                    user_outputs, ctx.run_info = TrainingManager.execution_session_run_forward(self._execution_agent,
                                                                                            self._optimized_onnx_model,
                                                                                            self._device,
                                                                                            *inputs)

                    # Disable materializing grads then None object will not be
                    # converted to a tensor filled with zeros prior to calling backward.
                    # Save shape, device and type info to ctx for materializing tensor in backward if output grad is None.
                    ctx.set_materialize_grads(False)

                    # Mark the outputs tensors needed in backward computation
                    # ORT is NOT relying on save_for_backward() to actually save the tensor,
                    # as this tensor is also kept in ORT's PartialGraphState
                    # This call is to invoke pytorch's version check to detect the potential inplace corruption
                    for idx in self._graph_info.module_output_indices_requires_save_for_backward:
                        ctx.save_for_backward(user_outputs[idx])

                    return user_outputs

                @staticmethod
                def backward(ctx, *grad_outputs):
                    '''Performs backward pass based on grad wrt module output'''

                    assert ctx.run_info is not None, 'forward() or __call__() methods must be called before backward()'
                    _utils._check_same_device(self._device, "Input argument to backward", *grad_outputs)

                    # Unpack saved_tensor to trigger version detection that catches inplace corruption
                    _ = ctx.saved_tensors

                    # Use IO binding
                    # Push user output grads to ONNX backend.
                    backward_inputs = C.OrtValueVector()
                    # Preallocate length of the vector. And then delete as required towards the end.
                    backward_inputs.reserve(len(grad_outputs))
                    for idx, grad_output in enumerate(grad_outputs):
                        if idx in self._graph_info.output_grad_indices_non_differentiable:
                            assert grad_output is None, "ORT found the {}-th module output '{}' is " \
                                                        "non-differentiable according to the onnx graph. " \
                                                        "However, the gradient value is still provided by " \
                                                        "PyTorch's autograd engine." \
                                                        .format(idx, self._graph_info.user_output_names[idx])
                            continue

                        if grad_output is None:
                            shape, device, dtype = ctx.run_info.output_info[idx]
                            if idx in self._graph_info.output_grad_indices_require_full_shape:
                                grad_output = torch.zeros(shape, device=device, dtype=dtype)
                            else:
                                grad_output = torch.tensor(0., device=device, dtype=dtype)
                        elif not grad_output.is_contiguous():
                            grad_output = grad_output.contiguous()
                        backward_inputs.push_back(to_dlpack(grad_output), grad_output.dtype == torch.bool)
                    backward_inputs.shrink_to_fit()

                    # Run and get results
                    backward_outputs = C.OrtValueVector()
                    self._execution_agent.run_backward(backward_inputs, backward_outputs, ctx.run_info.state)
                    # Destroy the state immediately (as opposed to be at the mercy of garbage collector) so it does not
                    # affect peak memory usage in a subsequent graph run.
                    del ctx.run_info.state
                    # Return input and initializer gradients
                    num_user_input_grads = len(self._input_info.require_grad_names)
                    results = []
                    require_grad_names_set = set(self._input_info.require_grad_names)
                    require_grad_names_index = 0
                    for input_name in self._graph_info.user_input_names:
                        # Append to the results the backward output for each input that required grad
                        if input_name in require_grad_names_set:
                            results.append(_utils._torch_tensor_from_dl_pack(
                                backward_outputs.dlpack_at(require_grad_names_index),
                                backward_outputs[require_grad_names_index]))
                            require_grad_names_index += 1
                        else:
                            # input_name is not found in the self._input_info.require_grad_names list
                            # Append None to results for each input that did not require grad
                            results.append(None)

                    # Append gradients of initializer to results
                    # Go over each initializer, check if it required grad and append to results accordingly
                    initializer_index = num_user_input_grads
                    for initializer_name in self._graph_info.initializer_names:
                        if initializer_name in self._graph_initializer_names_to_train:
                            results.append(_utils._torch_tensor_from_dl_pack(
                                backward_outputs.dlpack_at(initializer_index),
                                backward_outputs[initializer_index]))
                            initializer_index += 1
                        else:
                            results.append(None)

                    return tuple(results)

            return _io.unflatten_user_output(self._module_output_schema,
                                            _ORTModuleFunction.apply(
                                                *_io._combine_input_buffers_initializers(
                                                    self._graph_initializers,
                                                    self._graph_info.user_input_names,
                                                    self._input_info,
                                                    self._flattened_module.named_buffers(),
                                                    inputs,
                                                    kwargs,
                                                    self._device)))
        except Exception as e:
            self._check_fallback(e)

        # Fallback to PyTorch due to failures *during* forward(),
        #  (e.g. export, model/input post-processing, forward, output processing, etc)
        if self._pending_fallback():
            return self._apply_fallback(*inputs, **kwargs)

    def _build_graph(self):
        """Build an optimized gradient graph using the module_graph_builder"""

        super()._build_graph()

        if self._save_onnx:
            onnx.save(self._optimized_onnx_model, self._save_onnx_prefix + '_training.onnx')
            inference_optimized_model = onnx.load_model_from_string(self._graph_builder.get_inference_optimized_model())
            onnx.save(inference_optimized_model, self._save_onnx_prefix + '_inference_optimized.onnx')

    def _create_execution_agent(self):
        """Creates a TrainingAgent that can run the forward and backward graph on the training model"""

        session_options, providers, provider_options = self._get_session_config()
        fw_feed_names = [input.name for input in self._optimized_onnx_model.graph.input]
        fw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
            )] * len(self._graph_info.user_output_names)

        bw_fetches_names = [output.name for output in self._optimized_onnx_model.graph.output]
        bw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
            )] * len(bw_fetches_names)

        self._execution_agent = TrainingAgent(self._optimized_onnx_model.SerializeToString(),
                                              fw_feed_names,
                                              fw_outputs_device_info,
                                              bw_fetches_names,
                                              bw_outputs_device_info,
                                              session_options,
                                              providers,
                                              provider_options)

    def _reinitialize_graph_builder(self, input_info):
        """Return true if the module graph builder was reinitialized"""

        # Model could have unused parameters which are dropped after export and so not a part of self._graph_initializer_names_to_train.
        # To see if any trainable initializers changed, compare self._graph_initializer_names_to_train
        # with initializers in module named_parameters that are known to the onnx graph.
        initializer_names_to_train_set_user_model = {name for name, param in
                                                     self._flattened_module.named_parameters()
                                                     if param.requires_grad and name in self._graph_initializer_names}

        # If inputs requiring gradient change from forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_info.require_grad_names != self._input_info.require_grad_names or \
                initializer_names_to_train_set_user_model != self._graph_initializer_names_to_train:
            self._input_info = input_info
            self._initialize_graph_builder(training=True)
            return True
        return False
