# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _io, _logger, torch_cpp_extensions as _cpp_ext
from ._custom_autograd_function_exporter import _post_process_after_export
from ._graph_execution_interface import GraphExecutionInterface
from onnxruntime.training.ortmodule import ONNX_OPSET_VERSION

from onnxruntime.capi import _pybind_state as C
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from abc import ABC, abstractmethod
import copy
import io
import inspect
import onnx
import onnxruntime
import torch
import warnings

from torch.utils.cpp_extension import ROCM_HOME


class RunStateInfo(object):
    def __init__(self, state, output_info):
        """
        :param state: State of partial run that contains intermediate tensors needed to resume the run later.
        :param output_info: Output info.
        """
        self.state = state
        self.output_info = output_info

class GraphExecutionManager(GraphExecutionInterface):
    def __init__(self, module):
        """Manages building and execution of onnx graphs

        This class is an abstract class and should not directly be instantiated.
        Please use one of the concrete implementations of GraphExecutionManager.

        Interacts with OrtModuleGraphBuilder to build and optimize
        the onnx graph, and ExecutionAgent to run the onnx graph.
        """

        super(GraphExecutionManager, self).__init__(module._original_module)

        # Original and flattened (tranformed) output module
        self._flattened_module = module

        # Exported model
        self._onnx_model = None

        # Model after inference optimization or gradient building.
        self._optimized_onnx_model = None
        self._graph_builder = None
        self._graph_info = None
        self._graph_initializer_names = None
        self._graph_initializer_names_to_train = None
        self._graph_initializers = None

        # TrainingAgent or InferenceAgent
        self._execution_agent = None

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

        # Graph transformer config
        # Specify cast propagation strategy. Currently three strategies are available, NONE, INSERT-AND-REDUCE and FLOOD-FILL
        # The default is NONE, which implies the transformer does no cast-propagation transformation.
        self._propagate_cast_ops_strategy = C.PropagateCastOpsStrategy.NONE
        # Optimize by moving Cast operations if propagate_cast_ops_level is non-negative.
        # - If the _propagate_cast_ops_level is set to zero, then the transformation considers only the opcodes specified by _propagate_cast_ops_allow
        #   as "FP16 safe", in order to insert/(re)move cast operations before/after to perform such operations in reduced (16-bit) precision.
        # - If propagate_cast_ops_level is positive, 1 or 2, then in addition to opcode codes specified by propagate_cast_ops_allow use onnxruntime
        #   predetermined list of opcodes considered safe to move before/after cast operation.
        # - Onnxruntime Level 1 predetermind "FP16 safe" opcodes include only opcode that do not perform any computation such as Transpose, Split, Reshape, etc.
        #   whereas Level 2 perdetermined "FP16 safe" opcodes include opcodes that perform computation using contrib ops, GeLU, Dropout, LayerNormalization, etc.
        self._propagate_cast_ops_level = -1
        # List of opcodes to be considered safe to move before/after cast operation if propagate_cast_ops_level is zero.
        self._propagate_cast_ops_allow = []
        # Whether allow fusion of layer norm subgraph if doing so will cause modified precision.
        self._allow_layer_norm_mod_precision = False

        # Value can be either torch.onnx.TrainingMode.TRAINING or torch.onnx.TrainingMode.EVAL
        # To be instantiated in the concrete implementation of GraphExecutionManager
        self._export_mode = None

        # Related to training graph shape inference
        self._current_input_shape = None
        # default execution order is priority-based for both dynamic/static shape input for now
        # if we observe benefit of static shape, we can expose this flag to user
        self._use_static_shape = False

        # flag to enable symbolic shape inference for dynamic shape inputs to improve performance
        self._run_symbolic_shape_infer = True

        # A flag saying if custom autograd.Function should be allowed. True means yes and otherwise False.
        self._enable_custom_autograd_function = False

        self._input_info = None
        self._module_output_schema = None

        # Log level
        self._loglevel = _logger.LogLevel.WARNING

        # TODO: Single device support for now
        self._device = _utils.get_device_from_module(module)

        self._module_parameters = inspect.signature(self._original_module.forward).parameters.values()

        # TODO: remove after PyTorch ONNX exporter supports VAR_KEYWORD parameters.
        for input_parameter in self._module_parameters:
            if input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                if self._loglevel <= _logger.LogLevel.WARNING:
                    warnings.warn("The model's forward method has **kwargs parameter which has EXPERIMENTAL support!",
                                  UserWarning)

        self.is_rocm_pytorch = (True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False)

        self._use_external_gpu_allocator = True
        if self._use_external_gpu_allocator and torch.cuda.is_available():
            # CPP extension to get torch GPU allocator's alloc and free function addresses
            from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator
            self._torch_alloc = torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
            self._torch_free = torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()

        # WIP feature to enable caching in Gradient accumulation scenario.
        self._enable_grad_acc_optimization = False

    def _validate_module_type(self, module):
        """Raises a TypeError if the module is not a torch.nn.Module"""

        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"ORTModule only supports torch.nn.Module as input. {type(module)} is not supported.")

    @staticmethod
    def execution_session_run_forward(execution_session, onnx_model, device, *inputs):
        """Runs the forward pass on `execution_session` with given `onnx_model`, `device` and `inputs`

        This is a helper that can be called by the actual `GraphExecutionManager.forward` method

        Args:
            execution_session (InferenceAgent or InferenceAgent): Agent which runs either inference or train
            onnx_model (onnx.ModelProto): ONNX model
            device (torch.device): PyTorch device
            inputs: (torch.Tensor or a container of): User input

        Returns:
            Returns a tuple (user_outputs, run_info):
            user_outputs: The model output (either torch.Tensor or a container of torch.Tensor)
            run_info: A RunStateInfo which contains extra information about the execution of the graph
        """

        raise NotImplemented

    @abstractmethod
    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        This is the only method that the user should call on a concrete instance of the ExecutionManager
        All other methods are internal"""
        pass

    def _build_graph(self):
        if self._use_static_shape:
            self._graph_builder.build(self._input_info.shape)
        else:
            self._graph_builder.build()

        self._optimized_onnx_model = onnx.load_model_from_string(self._graph_builder.get_model())
        self._graph_info = self._graph_builder.get_graph_info()

    def _get_session_config(self):
        """Creates and returns the session configuration to be used for the ExecutionAgent"""
        providers = None
        provider_options = None
        if self._device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = (["ROCMExecutionProvider"] if self.is_rocm_pytorch else ["CUDAExecutionProvider"])
            providers.append("CPUExecutionProvider")
            if self._use_external_gpu_allocator:
                provider_options = [{"device_id": str(self._device.index),
                                     "gpu_external_alloc": str(self._torch_alloc),
                                     "gpu_external_free": str(self._torch_free)}, {}]
            else:
                provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.use_deterministic_compute = False
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._loglevel)

        # enable dumping optimized training graph
        if self._save_onnx:
            session_options.optimized_model_filepath = self._save_onnx_prefix + '_training_optimized.onnx'

        return session_options, providers, provider_options

    def _export_model(self, *inputs, **kwargs):
        # 1. Set the self._device from the user module
        # 2. Verify input schema matches schema used on previous model export
        # 3. Export the user model under self._export_training_flag mode
        # Return True if the model needed to be exported, False if no export was required.

        # Note: Model is only exported when:
        #       1. Model has never been exported before.
        #       2. Model input schema has changed (changes in inputs requiring gradient, shape, boolean inputs values change, etc)
        #       Model is not re-exported when the model parameters change. This can happen when the model is a stateful model,
        #       or the user explicitly changed model parameters after the onnx export.

        schema = _io._extract_schema({'args': copy.copy(inputs), 'kwargs': copy.copy(kwargs)})
        if self._onnx_model and schema == self._input_info.schema:
            # All required models have already been exported previously
            return False

        self._set_device_from_module(inputs, kwargs)
        self._onnx_model = self._get_exported_model(*inputs, **kwargs)
        _cpp_ext._load_aten_op_executor_cpp_extension_if_needed(self._onnx_model)
        if self._save_onnx:
            onnx.save(self._onnx_model, self._save_onnx_prefix + '_torch_exporter.onnx')

        if self._run_symbolic_shape_infer:
            self._onnx_model = SymbolicShapeInference.infer_shapes(self._onnx_model, auto_merge=True, guess_output_rank=True)

        return True

    def _get_exported_model(self, *inputs, **kwargs):
        '''Exports PyTorch `self._flattened_module` to ONNX for inferencing or training, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        '''

        # Setup dynamic axes for onnx model
        self._input_info = _io.parse_inputs_for_onnx_export(self._module_parameters,
                                                            None,
                                                            inputs,
                                                            kwargs)
        output_names, output_dynamic_axes, self._module_output_schema = \
            _io.parse_outputs_for_onnx_export_and_extract_schema(self._original_module, inputs, kwargs)
        self._input_info.dynamic_axes.update(output_dynamic_axes)

        # FlattenedModule needs _InputInfo to expand user input from *args to *args + **kwargs
        self._flattened_module._input_info = self._input_info

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy, sample_kwargs_copy = _io.deepcopy_model_input(*inputs, **kwargs)
        # NOTE: Flattening the input will change the 'input schema', resulting in a re-export
        sample_inputs_as_tuple = tuple(self._input_info.flatten(sample_inputs_copy, sample_kwargs_copy, self._device))
        # Ops behaving differently under train/eval mode need to exported with the
        # correct training flag to reflect the expected behavior.
        # For example, the Dropout node in a model is dropped under eval mode.
        assert self._export_mode is not None, "Please use a concrete instance of ExecutionManager"

        try:
            with torch.set_grad_enabled(self._enable_custom_autograd_function), \
                    _logger.suppress_os_stream_output(log_level=self._loglevel):
                torch.onnx.export(self._flattened_module,
                                  sample_inputs_as_tuple,
                                  f,
                                  input_names=self._input_info.names,
                                  output_names=output_names,
                                  opset_version=ONNX_OPSET_VERSION,
                                  do_constant_folding=False,
                                  training=self._export_mode,
                                  dynamic_axes=self._input_info.dynamic_axes,
                                  verbose=self._loglevel < _logger.LogLevel.WARNING,
                                  export_params=False,
                                  keep_initializers_as_inputs=True)
        except RuntimeError as e:
            raise RuntimeError('There was an error while exporting the PyTorch model to ONNX: {}'.format(e))
        exported_model = onnx.load_model_from_string(f.getvalue())

        exported_model = _post_process_after_export(exported_model, self._enable_custom_autograd_function)

        return exported_model

    def _set_device_from_module(self, inputs, kwargs):
        """Get the device from the module and save it to self._device"""

        device = _utils.get_device_from_module(self._original_module) or \
            _utils.get_device_from_inputs(inputs, kwargs)
        if not self._device or self._device != device:
            self._device = device
            if not self._device:
                raise RuntimeError('A device must be specified in the model or inputs!')

    def _get_graph_transformer_config(self):
        graph_transformer_config = C.TrainingGraphTransformerConfiguration()
        graph_transformer_config.propagate_cast_ops_config = C.PropagateCastOpsConfiguration()
        graph_transformer_config.propagate_cast_ops_config.level = self._propagate_cast_ops_level
        graph_transformer_config.propagate_cast_ops_config.allow = self._propagate_cast_ops_allow
        graph_transformer_config.propagate_cast_ops_config.strategy = self._propagate_cast_ops_strategy
        graph_transformer_config.allow_layer_norm_mod_precision = self._allow_layer_norm_mod_precision
        return graph_transformer_config

    def _initialize_graph_builder(self, training):
        """Creates a new OrtModuleGraphBuilder, initializes it and saves it to self._graph_builder"""

        # All initializer names along with user inputs are a part of the onnx graph inputs
        # since the onnx model was exported with the flag keep_initializers_as_inputs=True
        onnx_initializer_names = {p.name for p in self._onnx_model.graph.input}

        # TODO: PyTorch exporter bug: changes the initializer order in ONNX model
        initializer_names = [name for name, _ in self._flattened_module.named_parameters()
                             if name in onnx_initializer_names]
        initializer_names_to_train = [name for name, param in self._flattened_module.named_parameters()
                                      if param.requires_grad and name in onnx_initializer_names]

        # Build and optimize the full graph
        grad_builder_config = C.OrtModuleGraphBuilderConfiguration()
        grad_builder_config.initializer_names = initializer_names
        grad_builder_config.initializer_names_to_train = initializer_names_to_train
        grad_builder_config.input_names_require_grad = self._input_info.require_grad_names
        grad_builder_config.build_gradient_graph = training
        grad_builder_config.graph_transformer_config = self._get_graph_transformer_config()
        grad_builder_config.enable_caching = self._enable_grad_acc_optimization
        grad_builder_config.loglevel = _logger.ortmodule_loglevel_to_onnxruntime_c_loglevel(self._loglevel)
        self._graph_builder = C.OrtModuleGraphBuilder()

        # It is assumed here that the order and names of the inputs and outputs are not modified by the backend in any way
        # and are kept as they appear in the exported onnx model.
        self._graph_builder.initialize(self._onnx_model.SerializeToString(), grad_builder_config)

        # TODO: Explore ways to make self._graph_info.initializer_names and self._graph_info.initializer_names_to_train
        #       a set (unordered_set in the backend) that does not require a copy on each reference.
        self._graph_initializer_names = set(initializer_names)
        self._graph_initializer_names_to_train = set(initializer_names_to_train)

        # Initializers can be cached and used since they are expected not to be re-instantiated
        # between forward calls.
        self._graph_initializers = [param for name, param in self._flattened_module.named_parameters() 
                                    if name in self._graph_initializer_names]
