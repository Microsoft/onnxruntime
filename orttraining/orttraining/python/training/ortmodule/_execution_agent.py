# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import IOBinding, OrtValue
from onnxruntime.capi._pybind_state import TrainingAgent as C_TrainingAgent


class ExecutionAgentOutput(object):
    def __init__(self, ortvalues, run_id=None):
        self.ortvalues = ortvalues
        self.run_id = run_id


class InferenceAgent(object):
    """
    This is the main class used to run an ORTModule model inferencing.
    """

    def __init__(self, path_or_bytes, session_options=None, providers=None, provider_options=None):
        """
        :param path_or_bytes: filename or serialized ONNX or ORT format model in a byte string
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """

        self._inference_session = None

        self.create_inference_agent(path_or_bytes, session_options, providers, provider_options)

    def create_inference_agent(self, path_or_bytes, session_options, providers, provider_options):
        self._inference_session = onnxruntime.InferenceSession(path_or_bytes, session_options,
                                                               providers, provider_options)

    def io_binding(self):
        """Return an onnxruntime.IOBinding object`."""

        return IOBinding(self._inference_session)

    def run_forward(self, iobinding, run_options):
        """
         Compute the forward graph.
         :param iobinding: the iobinding object that has graph inputs/outputs bind.
         :param run_options: See :class:`onnxruntime.RunOptions`.
        """

        self._inference_session.run_with_iobinding(iobinding, run_options)
        ortvalues = iobinding.get_outputs()
        return ExecutionAgentOutput(ortvalues)


class TrainingAgent(object):
    """
    This is the main class used to run an ORTModule model training.
    """

    def __init__(self, path_or_bytes, fw_feed_names, fw_outputs_device_info,
                 bw_fetches_names, bw_outputs_device_info, session_options=None,
                 providers=None, provider_options=None):
        """
        :param path_or_bytes: filename or serialized ONNX or ORT format model in a byte string
        :param fw_feed_names: Feed names for foward pass.
        :param fw_outputs_device_info: Device info for fetches in forward pass.
        :param bw_fetches_names: Fetch names for backward pass.
        :param bw_outputs_device_info: Device info for fetches in backward pass.
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """

        self._inference_session = onnxruntime.InferenceSession(path_or_bytes, session_options,
                                                               providers, provider_options,
                                                               disabled_optimizers=["CommonSubexpressionElimination"])

        self._training_agent = C_TrainingAgent(self._inference_session._sess, fw_feed_names, fw_outputs_device_info,
                                               bw_fetches_names, bw_outputs_device_info)

    def run_forward(self, feeds, fetches, state):
        """
         Compute the forward subgraph for given feeds and fetches.
         :param feeds: Inputs to the graph run.
         :param fetches: Outputs of the graph run.
         :param state: State of the graph that is used for executing partial graph runs.
        """
        self._training_agent.run_forward(feeds, fetches, state)

    def run_backward(self, feeds, fetches, state):
        """
         Compute the backward subgraph for given feeds and fetches.
         :param feeds: Inputs to the graph run.
         :param fetches: Outputs of the graph run.
         :param state: State of the graph that is used for executing partial graph runs.
        """
        self._training_agent.run_backward(feeds, fetches, state)
