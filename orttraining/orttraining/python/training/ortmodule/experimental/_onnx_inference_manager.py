# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _io
from ._graph_execution_manager import GraphExecutionManager, RunStateInfo
from ._execution_agent import InferenceAgent

from onnxruntime.capi import _pybind_state as C
import onnx
import torch


class OnnxInferenceManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the inference model

    InferenceManager is resposible for building and running the forward graph of the inference model
    """

    def __init__(self, model, onnx_model_parameters=None, device=None):
        super().__init__(model)
        self._device = device
        self._onnx_model_parameters = onnx_model_parameters
        self._export_mode = torch.onnx.TrainingMode.EVAL

    @staticmethod
    def execution_session_run_forward(execution_session, onnx_model, device, *inputs):
        """Runs the forward graph on execution_session with given model inputs and device"""

        # Assert that the input and model device match
        _utils._check_same_device(device, "Input argument to forward", *inputs)

        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        io_binding = execution_session.io_binding()
        run_options = C.RunOptions()

        # Use IO binding
        _utils._create_iobinding(io_binding, inputs, onnx_model, device)

        # Run and return module outputs.
        ort_output = execution_session.run_forward(io_binding, run_options)
        forward_outputs, run_id = ort_output.ortvalues, ort_output.run_id
        user_outputs = tuple(_utils._ortvalue_to_torch_tensor(forward_output._ortvalue) for forward_output in forward_outputs)
        state = None

        # Assert that the outputs and model device match
        _utils._check_same_device(device, "Output argument from forward", *user_outputs)

        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info

    def forward(self, *inputs, **kwargs):
        '''Forward pass of the inference model

        ONNX model is exported the first time this method is executed.
        Next, we build an optimized inference graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession through the InferenceAgent.
        '''

        # Exporting module to ONNX for the first time
        build_graph = self._export_model(*inputs, **kwargs)
        if build_graph:
            # If model was exported, then initialize the graph builder
            self._initialize_graph_builder(training=False)

            # Save the onnx model if the model was exported
            if self._save_onnx:
                onnx.save(self._onnx_model, self._save_onnx_prefix + '_exported_inference_model.onnx')

            # Build the inference graph
            self._build_graph()

        module_device = self._device
        if self._original_module:
            module_device = _utils.get_device_from_module(self._original_module)

        # The inference session should be created every time
        # the graph was built or if the device changed between calls to forward
        create_execution_session = build_graph or self._device != module_device
        if self._device != module_device:
            self._device = module_device
        if create_execution_session:
            # Create execution session creates the inference_session
            self._create_execution_agent()

        user_outputs, _ = OnnxInferenceManager.execution_session_run_forward(self._execution_agent,
                                                                             self._optimized_onnx_model,
                                                                             self._device,
                                                                             *_io._combine_input_buffers_initializers(
                                                                                 self._graph_initializers,
                                                                                 self._graph_info.user_input_names,
                                                                                 self._input_info,
                                                                                 self._flattened_module.named_buffers() if self._flattened_module else [],
                                                                                 inputs,
                                                                                 kwargs,
                                                                                 self._device))

        return _io.unflatten_user_output(self._module_output_schema,
                                         user_outputs)

    def _build_graph(self):
        """Build an optimized inference graph using the module_graph_builder"""

        super()._build_graph()
        if self._save_onnx:
            onnx.save(self._optimized_onnx_model, self._save_onnx_prefix + '_inference.onnx')

    def _create_execution_agent(self):
        """Creates an InferenceAgent that can run forward graph on an inference model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = InferenceAgent(self._optimized_onnx_model.SerializeToString(),
                                               session_options, providers, provider_options)
