import copy
import io
import os
import onnx
import torch
from inspect import signature
import warnings
from functools import partial
import numpy as np

import onnxruntime as ort
from . import _utils, amp, checkpoint, optim, postprocess, ORTTrainerOptions, _checkpoint_storage
from .model_desc_validation import _ORTTrainerModelDesc

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

class TrainStepInfo(object):
    r"""Private class used to store runtime information from current train step.

    After every train step, :py:meth:`ORTTrainer.train_step` updates the internal instance of
    :py:class:`.TrainStepInfo` residing on :py:class:`.ORTTrainer` with relevant information
    from the forward pass.

    This class shouldn't be accessed directly by the user, unless they really know what they are doing.
    Instead, :py:class:`.ORTTrainer` passes it to relevant class methods automatically,
    such as :py:method:`._LRScheduler.get_lr` or :py:class:`.LossScaler.update`.

    Args:
        optimizer_config (optim._OptimizerConfig): reference to optimizer config
        all_finite (bool, default is True): flag that indicates whether all gradients are still finite after last step
        fetches (list of str, default is []): list of output names to fetch from train_step/eval_step. Set it to [] to reset normal behavior.
        optimization_step (int): indicates the number of optimizations performed. Used for learning rate scheduling
        step (int): indicates current training step. Used for gradient accumulation

    Example:

        .. code-block:: python

            info = TrainStepInfo(optimizer_config=optim.SGDConfig(lr=0.01))
            if info.all_finite:
                print(f'Yay, all gradients are finite at {step} step!')

    """

    def __init__(self, optimizer_config, all_finite=True, fetches=[], optimization_step=0, step=0):
        assert isinstance(optimizer_config, optim._OptimizerConfig),\
            "optimizer_config must be a optim._OptimizerConfig"
        assert isinstance(all_finite, bool),\
            "all_finite must be a bool"
        assert isinstance(fetches, list) and all([isinstance(item, str) for item in fetches]),\
            "fetches must be a list of str"
        assert isinstance(optimization_step, int) and optimization_step >= 0,\
            "optimization_step must be a positive int"
        assert (isinstance(step, int) and step >= 0),\
            "step must be a positive int"

        self.optimizer_config = optimizer_config
        self.all_finite = all_finite
        self.fetches = fetches
        self.optimization_step = optimization_step
        self.step = step


class ORTTrainer(object):
    r"""Pytorch frontend for ONNX Runtime training

    Entry point that exposes the C++ backend of ORT as a Pytorch frontend.

    Args:
        model (torch.nn.Module or onnx.ModelProto): either a PyTorch or ONNX model.
            When a PyTorch model and :py:attr:`loss_fn` are specified, :py:attr:`model` and :py:obj:`loss_fn` are combined.
            When a ONNX model is provided, the loss is identified by the flag :py:obj:`is_loss=True` in one of the :py:attr:`.model_desc.outputs` entries.
        model_desc (dict): model input and output description.
            This is used to identify inputs and outputs and their shapes, so that ORT can generate back propagation graph, plan memory allocation for
            training, and perform optimizations.
            :py:attr:`model_desc` must be consistent with the training :py:attr:`model` and have the following (:py:obj:`dict`) schema
            :py:obj:`{ 'inputs': [tuple(name, shape)], 'outputs': [tuple(name, shape, is_loss)]}`.
            :py:attr:`name` is a string representing the name of input or output of the model.
            For :py:obj:`model_desc['inputs']` entries, :py:attr:`name` must match input names of the original PyTorch model's :py:meth:`torch.nn.Module.forward` method.
            For ONNX models, both name and order of input names must match.
            For :py:obj:`model_desc['outputs']` entries, the order must match the original PyTorch's output as returned by :py:meth:`torch.nn.Module.forward` method.
            For ONNX models, both name and order of output names must match.
            :py:attr:`shape` is a list of string or integers that describes the shape of the input/output.
            Each dimension size can be either a string or an int. String means the dimension size is dynamic, while integers mean static dimensions.
            An empty list implies a scalar.
            Lastly, :py:attr:`is_loss` is a boolean (default is False) that flags if this output is considered a loss.
            ORT backend needs to know which output is loss in order to generate back propagation graph.
            Loss output must be specified when either :py:attr:`loss_fn` is specified or when loss is embedded in the model.
            Note that only one loss output is supported per model.
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.AdamConfig`, :py:class:`.optim.LambConfig` or :py:class:`.optim.SGDConfig`.
        loss_fn (callable, default is None): a PyTorch loss function.
            It takes two inputs [prediction, label] and outputs a scalar loss tensor.
            If provided, :py:attr:`loss_fn` is combined with the PyTorch :py:attr:`model` to form a combined PyTorch model.
            Inputs to the combined PyTorch model are concatenation of the :py:attr:`model`'s input and :py:attr:`loss_fn`'s label input.
            Outputs of the combined PyTorch model are concatenation of :py:attr:`loss_fn`'s loss output and :py:attr:`model`'s outputs.
        options (ORTTrainerOptions, default is None): options for additional features.
    Example:

        .. code-block:: python

            model = ...
            loss_fn = ...
            model_desc = {
                "inputs": [
                    ("input_ids", ["batch", "max_seq_len_in_batch"]),
                    ("attention_mask", ["batch", "max_seq_len_in_batch"]),
                    ("token_type_ids", ["batch", "max_seq_len_in_batch"]),
                    ("masked_lm_labels", ["batch", "max_seq_len_in_batch"]),
                    ("next_sentence_label", ["batch", 1])
                ],
                "outputs": [
                    ("loss", [], True),
                ],
            }
            optim_config = optim.LambConfig(param_groups = [ { 'params' : ['model_param0'], 'alpha' : 0.8, 'beta' : 0.7},
                                                             { 'params' : ['model_param1' , 'model_param_2'], 'alpha' : 0.0}
                                                           ],
                                            alpha=0.9, beta=0.999)
            ort_trainer = ORTTrainer(model, model_desc, optim_config, loss_fn)
    """

    def __init__(self, model, model_desc, optim_config, 
                 loss_fn=None, 
                 options=None):
        assert model is not None, "'model' is required and must be either a 'torch.nn.Module' or ONNX model"
        assert isinstance(model_desc, dict), "'model_desc' must be a 'dict'"
        assert isinstance(optim_config, optim._OptimizerConfig),\
            "'optim_config' is required and must be any of 'AdamConfig', 'LambConfig' or 'SGDConfig'"
        assert loss_fn is None or (callable(loss_fn) and len(signature(loss_fn).parameters) == 2),\
            "'loss_fn' must be either 'None' or a callable with two parameters"
        assert options is None or isinstance(options, ORTTrainerOptions),\
            "'options' must be either 'None' or 'ORTTrainerOptions'"

        #            Model + Loss validation
        #           Supported combinarios are
        #    ----------------------------------------
        #   |   | Model            | Loss            |
        #    ----------------------------------------
        #   | 1 | torch.nn.Module  | None            |
        #   | 2 | torch.nn.Module  | torch.nn.Module |
        #   | 3 | ONNX             | None            |
        #    ----------------------------------------
        self._torch_model = None
        self._onnx_model = None
        if isinstance(model, torch.nn.Module):
            assert loss_fn is None or isinstance(model, torch.nn.Module),\
                "'loss_fn' must be either 'None' or 'torch.nn.Module'"
            self._torch_model = model
            self.loss_fn = loss_fn
            # TODO: Remove when experimental checkpoint functions are removed.
            self._torch_state_dict_keys = list(model.state_dict().keys())
        elif isinstance(model, onnx.ModelProto):
            assert loss_fn is None, "'loss_fn' must not be specified when 'model' is an ONNX model"
            self._onnx_model = model
            self.loss_fn = None
        else:
            raise ValueError("'model' must be either 'torch.nn.Module' or 'onnx.ModelProto'")

        self.model_desc = _ORTTrainerModelDesc(model_desc)
        self.optim_config = optim_config

        # ORTTrainerOptions
        if not options:
            options = ORTTrainerOptions()
        self.options = options
        if self.options.mixed_precision.enabled and not self.options.mixed_precision.loss_scaler:
            # TODO: Move this to model_desc_validation.py
            self.options.mixed_precision.loss_scaler = amp.loss_scaler.DynamicLossScaler()
        # Post processing ONNX model given as input
        if self._onnx_model:
            if self.options._internal_use.enable_internal_postprocess:
                self._onnx_model = postprocess.run_postprocess(self._onnx_model)
            if self.options._internal_use.extra_postprocess:
                self._onnx_model = self.options._internal_use.extra_postprocess(self._onnx_model)
                assert isinstance(self._onnx_model, onnx.ModelProto), "'extra_postprocess' must return a ONNX model"

            # When input model is already ONNX (and not exported from Pytorch within ORTTrainer),
            # append 'dtype' from ONNX into model description's
            for idx_i, i_desc in enumerate(self.model_desc.inputs):
               dtype = None
               for onnx_input in self._onnx_model.graph.input:
                  if onnx_input.name == i_desc.name:
                     dtype = _utils.dtype_onnx_to_torch(onnx_input.type.tensor_type.elem_type)
                     self.model_desc.add_type_to_input_description(idx_i, dtype)
                     break
               assert dtype is not None, f"ONNX model with unknown input type ({i_desc.name})"
            for idx_o, o_desc in enumerate(self.model_desc.outputs):
                dtype = None
                for onnx_output in self._onnx_model.graph.output:
                    if onnx_output.name == o_desc.name:
                        dtype = _utils.dtype_onnx_to_torch(onnx_output.type.tensor_type.elem_type)
                        self.model_desc.add_type_to_output_description(idx_o, dtype)
                        break
                assert dtype is not None, f"ONNX model with unknown output type ({o_desc.name})"

        try:
            from torch.utils.cpp_extension import ROCM_HOME
            self.is_rocm_pytorch = (True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False)
        except ImportError:
            self.is_rocm_pytorch = False

        # TODO: Remove when experimental checkpoint functions are removed.
        self._state_dict = {}

        self._train_step_info = TrainStepInfo(self.optim_config)
        self._training_session = None
        self._load_state_dict = None
        self._init_session(provider_options=self.options._validated_opts['provider_options'],
                           session_options=self.options.session_options)

    def eval_step(self, *args, **kwargs):
        r"""Evaluation step method

        Args:
            *args: Arbitrary arguments that are used as model input (data only)
            **kwargs: Arbitrary keyword arguments that are used as model input (data only)

        Returns:
            ordered :py:obj:`list` with model outputs as described by :py:attr:`.ORTTrainer.model_desc`
        """
        # Get data. CombineTorchModelLossFn takes label as last input and outputs loss first
        sample_input = self._prepare_model_input(self.model_desc.inputs,
                                                 None, None, *args, **kwargs)

        # Export model to ONNX
        if self._onnx_model is None:
            if self._torch_model is not None:
                self._init_onnx_model(sample_input)
            else:
                raise RuntimeError("Model is uninitialized. Only ONNX and PyTorch models are supported")

        # Prepare input/output description
        inputs_desc = self.model_desc.inputs
        outputs_desc = self.model_desc.outputs
        if self._train_step_info.fetches:
            outputs_desc = [o_desc for o_desc in outputs_desc if o_desc.name in self._train_step_info.fetches]
            if len(outputs_desc) != len(self._train_step_info.fetches):
                raise RuntimeError("The specified fetches list contains invalid output names")

        # Normalize input
        if not isinstance(sample_input, (list, tuple)):
            sample_input = (sample_input,)

        # RunOptions
        run_options = ort.RunOptions()
        run_options.only_execute_path_to_fetches = True
        run_options.training_mode = False

        # Run a eval step and return
        session_run_results = self._training_session_run_helper(False,
                                                                sample_input,
                                                                inputs_desc,
                                                                outputs_desc,
                                                                run_options)

        # Output must be returned in the same order as defined in the model description
        results = [session_run_results[o_desc.name] for o_desc in outputs_desc]
        return results[0] if len (results) == 1 else results


    def save_as_onnx(self, path):
        r"""Persists ONNX model into :py:attr:`path`

        The model will be saved as a Google Protocol Buffers (aka protobuf) file as per ONNX standard.
        The graph includes full information, including inference and training metadata.

        Args:
            path (str): Full path, including filename, to save the ONNX model in the filesystem

        Raises:
            RuntimeWarning: raised when neither `train_step` or `eval_step` was called at least once
            ValueError: raised when `path` is not valid path
        """
        if not self._training_session:
            warnings.warn("Training session is not initialized yet. "
                                 "'train_step' or 'eval_step' methods must be executed at least once before calling 'save_as_onnx()'.")
            return
        state_tensors = self._training_session.get_state()
        self._update_onnx_model_initializers(state_tensors)

        assert isinstance(path, str), "'path' must be a valid path string"
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        if (dir_name and not os.path.exists(dir_name)) or not file_name:
            warnings.warn("'path' is not valid or does not exist")
            return

        with open(path, "wb") as f:
            f.write(self._onnx_model.SerializeToString())

    def _check_model_export(self, input):
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        from numpy.testing import assert_allclose
        onnx_model_copy = copy.deepcopy(self._onnx_model)

        # Mute the dropout nodes
        dropout_nodes = [n for n in onnx_model_copy.graph.node if n.op_type == 'Dropout']
        for node in dropout_nodes:
            ratio_node = [n for n in onnx_model_copy.graph.node if node.input[1] in n.output][0]
            training_mode_node = [n for n in onnx_model_copy.graph.node if node.input[2] in n.output][0]

            training_mode_node.attribute.pop()
            ratio_node.attribute.pop()
            new_training_mode_arr = np.array(False, dtype=bool)
            new_ratio_arr = np.array(0.0, dtype=np.float32)
            new_training_mode = numpy_helper.from_array(new_training_mode_arr)
            new_ratio = numpy_helper.from_array(new_ratio_arr)
            training_mode_node.attribute.add().t.CopyFrom(new_training_mode)
            ratio_node.attribute.add().t.CopyFrom(new_ratio)
            training_mode_node.attribute[0].type = 4
            ratio_node.attribute[0].type = 4
            training_mode_node.attribute[0].name = "value"
            ratio_node.attribute[0].name = "value"

        _inference_sess = ort.InferenceSession(onnx_model_copy.SerializeToString())
        inf_inputs = {}
        for i, input_elem in enumerate(input):
            inf_inputs[_inference_sess.get_inputs()[i].name] = input_elem.cpu().numpy()
        _inference_outs = _inference_sess.run(None, inf_inputs)
        for torch_item, ort_item in zip(self.torch_sample_outputs, _inference_outs):
            assert_allclose(torch_item, ort_item, rtol=1e-2, atol=1e-6,
                            err_msg="Mismatch between outputs of PyTorch model and exported ONNX model. "
                                    "Note that different backends may exhibit small computational differences."
                                    "If this is within acceptable margin, or if there is random generator "
                                    "in the model causing inevitable mismatch, you can proceed training by "
                                    "setting the flag debug.check_model_export to False.")

    def train_step(self, *args, **kwargs):
        r"""Train step method

        After forward pass, an ordered list with all outputs described at :py:attr:`ORTTrainer.model_desc` is returned.
        Additional information relevant to the train step is maintend by :py:attr:`ORTTrainer._train_step_info`.
        See :py:class:`.TrainStepInfo` for details.

        Args:
            *args: Arbitrary arguments that are used as model input (data only)
            **kwargs: Arbitrary keyword arguments that are used as model input (data only)

        Returns:
            ordered :py:obj:`list` with model outputs as described by :py:attr:`ORTTrainer.model_desc`
        """
        # Export model to ONNX
        if self._onnx_model is None:
            sample_input = self._prepare_model_input(self.model_desc.inputs, None, None, *args, **kwargs)
            self._init_onnx_model(sample_input)

            # Debug Model Export if indicated
            if self.options.debug.check_model_export:
                self._check_model_export(sample_input)


        # Prepare inputs+lr and output descriptions
        inputs_desc = self._model_desc_inputs_with_lr
        outputs_desc = self.model_desc.outputs

        # Train step must be incremented *before* gradient accumulation code
        # Gradients are accumulated when
        # self._train_step_info.step % self.options.batch.gradient_accumulation_steps != 0,
        # and they are updated otherwise
        self._train_step_info.step += 1

        # RunOptions
        run_options = None
        mixed_precision_without_fetches = False
        if self._train_step_info.fetches:
            outputs_desc = [o_desc for o_desc in outputs_desc if o_desc.name in self._train_step_info.fetches]
            if len(outputs_desc) != len(self._train_step_info.fetches):
                raise RuntimeError("The specified fetches list contains invalid output names")
        elif self._train_step_info.step % self.options.batch.gradient_accumulation_steps != 0:
            run_options = ort.RunOptions()
            run_options.only_execute_path_to_fetches = True
            outputs_desc = self._model_desc_outputs_with_gradient_accumulation
        elif self.options.mixed_precision.enabled:
            mixed_precision_without_fetches = True
            outputs_desc = self._model_desc_outputs_with_all_finite

        # Update Learning Rate if Necessary
        lr = self.optim_config.lr
        if self.options.lr_scheduler:
            lr = self.options.lr_scheduler._step(self._train_step_info)[0]

        # Loss Scale for mixed precision
        loss_scale = None
        if self.options.mixed_precision.enabled:
            loss_scaler = self.options.mixed_precision.loss_scaler
            assert loss_scaler, "Loss scaler is required when mixed precision is enabled"
            loss_scale = loss_scaler.loss_scale
            inputs_desc = self._model_desc_inputs_with_lr_and_loss_scale

        # Get data. CombineTorchModelLossFn takes label as last input and outputs loss first
        input = self._prepare_model_input(inputs_desc, lr, loss_scale, *args, **kwargs)

        # Normalize input
        if not isinstance(args, (list, tuple)):
            args = (args,)

        # Run a train step and return
        session_run_results = self._training_session_run_helper(True, input, inputs_desc,
                                                                outputs_desc, run_options)
        if mixed_precision_without_fetches:
            # After session run with all_fp32_gradients_finite, we need to clear the training I/O binding's output
            # Otherwise next run with only_execute_path_to_fetches will lead to gradient all reduce
            # because all_fp32_gradients_finite is still in the feed.
            self._train_io_binding.clear_binding_outputs()

            is_all_finite = session_run_results[self.model_desc.all_finite.name]
            self._train_step_info.all_finite = is_all_finite
            if loss_scaler:
                loss_scaler.update(self._train_step_info)
            if is_all_finite:
                # Optimization step must be incremented *after* optimization is successful
                self._train_step_info.optimization_step += 1
        elif self._train_step_info.step % self.options.batch.gradient_accumulation_steps == 0:
            # Optimization step must be incremented *after* optimization is successful
            self._train_step_info.optimization_step += 1

        # Output must be returned in the same order as defined in the model description
        # or in the order specified by TrainStepInfo.fetches, if applicable
        if self._train_step_info.fetches:
            results = [session_run_results[o_desc] for o_desc in self._train_step_info.fetches]
        else:
            results = [session_run_results[o_desc.name] for o_desc in self.model_desc.outputs]
        return results[0] if len (results) == 1 else results

    def _convert_torch_model_loss_fn_to_onnx(self, inputs, device):
        # Dynamic axes
        dynamic_axes = {}
        for input in self.model_desc.inputs:
            symbolic_axis = {}
            for i, axis in enumerate(input.shape):
                if isinstance(axis, str):
                    symbolic_axis[i] = axis
            if len(symbolic_axis):
                dynamic_axes[input.name] = symbolic_axis
        for output in self.model_desc.outputs:
            symbolic_axis = {}
            for i, axis in enumerate(output.shape):
                if isinstance(axis, str):
                    symbolic_axis[i] = axis
            if len(symbolic_axis):
                dynamic_axes[output.name] = symbolic_axis

        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if isinstance(inputs, dict):
            sample_inputs = [inputs[k.name_].to(device=device) for k in self.model_desc.inputs]
        elif isinstance(inputs, (list, tuple)):
            sample_inputs = [input.to(device=device) for i, input in enumerate(inputs) if i < len(self.model_desc.inputs)]
        else:
            raise RuntimeError("Unexpected input type. Only torch.Tensor, or dict/list/tuple of torch.Tensor is supported.")

        # PyTorch ONNX exporter does not match argument names
        # This is an issue because the ONNX graph depends on all inputs to be specified

        # Validate loss_fn
        if self.loss_fn:
            sig_loss = signature(self.loss_fn)
            if len(sig_loss.parameters) != 2:
                raise RuntimeError("loss function should take two arguments - predict and label.")

        # Basic input names from model
        input_names = [input.name for input in self.model_desc.inputs]
        sig = signature(self._torch_model.forward)
        ordered_input_list = list(sig.parameters.keys())

        # Label from loss_fn goes after model input
        if self.loss_fn:
            ordered_input_list = [*ordered_input_list,
                                list(sig_loss.parameters.keys())[1]]

        class CombineTorchModelLossFnWrapInput(torch.nn.Module):
            def __init__(self, model, loss_fn, input_names):
                super().__init__()
                self.model = model
                self.loss_fn = loss_fn
                self.input_names = input_names

            def forward(self, *inputs):
                sig = signature(self.model.forward)

                input_dict = {}
                for key in sig.parameters.keys():
                    if key in self.input_names:
                        input_dict[key] = inputs[self.input_names.index(key)]

                model_out = self.model(**input_dict)
                if self.loss_fn is None:
                    return model_out

                label = inputs[-1]
                preds = model_out
                return self.loss_fn(preds, label), preds

        model = CombineTorchModelLossFnWrapInput(self._torch_model, self.loss_fn, input_names)

        # Do an inference to grab output types
        model.eval()
        with torch.no_grad():
            # Deepcopy inputs, since input values may change after model run.
            sample_inputs_copy = copy.deepcopy(sample_inputs)
            try:
                # Deepcopy model, in case model is stateful and changes after model run.
                model_copy = copy.deepcopy(model)
            except Exception:
                model_copy = model
                warnings.warn("This model cannot be deep copied (or pickled), which is a required step for stateful models to be properly exported to ONNX."
                              " Compute will continue, but unexpected results may occur!")
            sample_outputs = model_copy(*sample_inputs_copy)
            self.torch_sample_outputs = sample_outputs
        model.train()

        if isinstance(sample_outputs, torch.Tensor):
            sample_outputs = [sample_outputs]

        # Append 'dtype' for model description's inputs/outputs
        for idx_i, sample_input in enumerate(sample_inputs):
            if idx_i < len(self.model_desc.inputs):
                self.model_desc.add_type_to_input_description(
                    idx_i, sample_input.dtype)
        for idx_o, sample_output in enumerate(sample_outputs):
            if idx_o < len(self.model_desc.outputs):
                self.model_desc.add_type_to_output_description(
                    idx_o, sample_output.dtype)

        # Export the model to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy = copy.deepcopy(sample_inputs)

        # Handle contrib OPs support
        from onnxruntime.training import register_custom_ops_pytorch_exporter
        if self.options._internal_use.enable_onnx_contrib_ops:
            # Enable contrib ops export from PyTorch
            register_custom_ops_pytorch_exporter.register_custom_op()
        else:
            # Unregister contrib ops, if they were registered in previous calls
            register_custom_ops_pytorch_exporter.unregister_custom_op()

        # Export torch.nn.Module to ONNX
        torch.onnx._export(model, tuple(sample_inputs_copy), f,
                           input_names=[input.name for input in self.model_desc.inputs],
                           output_names=[output.name for output in self.model_desc.outputs],
                           opset_version=self.options._internal_use.onnx_opset_version,
                           dynamic_axes=dynamic_axes,
                           _retain_param_name=True,
                           example_outputs=tuple(sample_outputs),
                           do_constant_folding=False,
                           training=torch.onnx.TrainingMode.TRAINING)
        onnx_model = onnx.load_model_from_string(f.getvalue())

        # Remove 'model.' prefix introduced by CombineTorchModelLossFn class
        if isinstance(model, CombineTorchModelLossFnWrapInput):
            replace_name_dict = {}
            for n in onnx_model.graph.initializer:
                if n.name.startswith('model.'):
                    replace_name_dict[n.name] = n.name[len('model.'):]
                    n.name = replace_name_dict[n.name]
            for n in onnx_model.graph.node:
                for i, name in enumerate(n.input):
                    if name in replace_name_dict:
                        n.input[i] = replace_name_dict[name]

        return onnx_model

    def _create_ort_training_session(self, 
                                     optimizer_state_dict={}, 
                                     session_options=None, 
                                     provider_options=None):
        # Validating frozen_weights names
        unused_frozen_weights = [n for n in self.options.utils.frozen_weights\
            if n not in [i.name for i in self._onnx_model.graph.initializer]]
        if unused_frozen_weights:
            raise RuntimeError("{} params from 'frozen_weights' not found in the ONNX model.".format(
                unused_frozen_weights))

        # Get loss name from model description
        loss_name = [item.name for item in self.model_desc.outputs if item.is_loss]
        assert len(loss_name) == 1, f"Only one loss output is supported ({len(loss_name)} were specified)"
        loss_name = loss_name[0]

        # Parse optimizer parameters
        optimizer_attributes_map = {}
        optimizer_int_attributes_map = {}
        trainable_params = set()
        for initializer in self._onnx_model.graph.initializer:
            if initializer.name in self.options.utils.frozen_weights:
                continue  # only trainable parameters are passed to the backend
            trainable_params.add(initializer.name)
            optimizer_attributes_map[initializer.name] = {}
            optimizer_int_attributes_map[initializer.name] = {}
            not_in_param_groups = True
            for param_group in self.optim_config.params:
                if initializer.name not in param_group['params']:
                    continue  # keep looking for a matching param_group
                not_in_param_groups = False
                for k, v in param_group.items():
                    # 'params' is not a hyper parameter, skip it. 'lr' per weight is not supported
                    if k == 'params' or k == 'lr':
                        continue
                    if isinstance(v, float):
                        optimizer_attributes_map[initializer.name][k] = v
                    elif isinstance(v, int):
                        optimizer_int_attributes_map[initializer.name][k] = v
                    else:
                        raise ValueError("Optimizer attributes must be either float or int.")

            # set default values for params not found in groups
            if not_in_param_groups:
                for k, v in self.optim_config.defaults.items():
                    if k == 'lr':
                        continue
                    if isinstance(v, float):
                        optimizer_attributes_map[initializer.name][k] = v
                    elif isinstance(v, int):
                        optimizer_int_attributes_map[initializer.name][k] = v
                    else:
                        raise ValueError("Optimizer attributes must be either float or int.")

        self.options.distributed.horizontal_parallel_size = max(self.options.distributed.horizontal_parallel_size, 1)
        self.options.distributed.data_parallel_size = self.options.distributed.world_size // self.options.distributed.horizontal_parallel_size
        
        # TrainingParameters
        ort_parameters = ort.TrainingParameters()
        ort_parameters.loss_output_name = loss_name
        ort_parameters.use_mixed_precision = self.options.mixed_precision.enabled
        ort_parameters.world_rank = self.options.distributed.world_rank
        ort_parameters.world_size = self.options.distributed.world_size
        ort_parameters.gradient_accumulation_steps = self.options.batch.gradient_accumulation_steps
        ort_parameters.allreduce_post_accumulation = self.options.distributed.allreduce_post_accumulation
        ort_parameters.enable_adasum = self.options.distributed.enable_adasum
        ort_parameters.deepspeed_zero_stage = self.options.distributed.deepspeed_zero_optimization.stage
        ort_parameters.enable_grad_norm_clip = self.options.utils.grad_norm_clip
        ort_parameters.set_gradients_as_graph_outputs = False
        ort_parameters.use_invertible_layernorm_grad = self.options.utils.invertible_layer_norm_gradient
        ort_parameters.training_optimizer_name = self.optim_config.name
        ort_parameters.lr_params_feed_name = self.model_desc.learning_rate.name
        ort_parameters.weights_to_train = trainable_params
        ort_parameters.optimizer_attributes_map = optimizer_attributes_map
        ort_parameters.optimizer_int_attributes_map = optimizer_int_attributes_map
        if bool(optimizer_state_dict):
            ort_parameters.set_optimizer_initial_state(optimizer_state_dict)

        ort_parameters.attn_dropout_recompute = self.options.graph_transformer.attn_dropout_recompute
        ort_parameters.gelu_recompute = self.options.graph_transformer.gelu_recompute
        ort_parameters.transformer_layer_recompute = self.options.graph_transformer.transformer_layer_recompute
        ort_parameters.number_recompute_layers = self.options.graph_transformer.number_recompute_layers

        ort_parameters.data_parallel_size = self.options.distributed.data_parallel_size
        ort_parameters.horizontal_parallel_size = self.options.distributed.horizontal_parallel_size
        ort_parameters.pipeline_parallel_size = self.options.distributed.pipeline_parallel.pipeline_parallel_size
        ort_parameters.num_pipeline_micro_batches = self.options.distributed.pipeline_parallel.num_pipeline_micro_batches
        ort_parameters.pipeline_cut_info_string = self.options.distributed.pipeline_parallel.pipeline_cut_info_string
        # We have special handling for dictionary-typed option.
        # sliced_schema._validated_opts is the original dictionary while sliced_schema is a _ORTTrainerOptionsInternal.
        ort_parameters.sliced_schema = self.options.distributed.pipeline_parallel.sliced_schema._validated_opts
        # We have special handling for dictionary-typed option.
        # sliced_axes._validated_opts is the original dictionary while sliced_schema is a _ORTTrainerOptionsInternal.
        ort_parameters.sliced_axes = self.options.distributed.pipeline_parallel.sliced_axes._validated_opts
        ort_parameters.sliced_tensor_names = self.options.distributed.pipeline_parallel.sliced_tensor_names

        ort_parameters.model_after_graph_transforms_path = self.options.debug.graph_save_paths.model_after_graph_transforms_path
        ort_parameters.model_with_gradient_graph_path = self.options.debug.graph_save_paths.model_with_gradient_graph_path
        ort_parameters.model_with_training_graph_path = self.options.debug.graph_save_paths.model_with_training_graph_path

        # SessionOptions
        session_options = ort.SessionOptions() if session_options is None else session_options
        session_options.use_deterministic_compute = self.options.debug.deterministic_compute
        if (self.options.graph_transformer.attn_dropout_recompute or
            self.options.graph_transformer.gelu_recompute or
            self.options.graph_transformer.transformer_layer_recompute):
            session_options.execution_order = ort.ExecutionOrder.PRIORITY_BASED
        if len(self.options.debug.graph_save_paths.model_with_training_graph_after_optimization_path) > 0:
            session_options.optimized_model_filepath = self.options.debug.graph_save_paths.model_with_training_graph_after_optimization_path

        # old ort session may already exists and occupies GPU memory when creating new session, this may cause OOM error.
        # for example, load_state_dict will be called before returing the function, and it calls _init_session again
        del self._training_session
        # Set provider-specific options if needed
        def get_providers(provider_options):
            providers = ort.get_available_providers()
            if provider_options:
                for provider_name in provider_options:
                    if provider_name in providers:
                        providers[providers.index(provider_name)] = (provider_name, provider_options[provider_name])
                    else:
                        providers.insert(0, (provider_name, provider_options[provider_name]))
            #default: using cuda
            elif 'cuda' in self.options.device.id.lower():
                gpu_ep_options = {"device_id": _utils.get_device_index(self.options.device.id)}
                gpu_ep_name = ("ROCMExecutionProvider" if self.is_rocm_pytorch else "CUDAExecutionProvider")
                if self.options.device.mem_limit > 0:
                    gpu_ep_options["gpu_mem_limit"] = self.options.device.mem_limit

                if gpu_ep_name not in providers:
                    raise RuntimeError(
                        "ORTTrainer options specify a CUDA device but the {} provider is unavailable.".format(
                            cuda_ep_name))

                providers[providers.index(gpu_ep_name)] = (gpu_ep_name, gpu_ep_options)

            return providers

        # TrainingSession
        self._training_session = ort.TrainingSession(self._onnx_model.SerializeToString(), ort_parameters,
                                                     session_options, get_providers(provider_options))

        # I/O bindings
        self._train_io_binding = self._training_session.io_binding()
        self._eval_io_binding = self._training_session.io_binding()

    def _init_onnx_model(self, inputs):
        if self._onnx_model is not None:
            return

        if self._torch_model is not None:
            # PyTorch model is moved to cpu to save GPU memory
            self._torch_model.cpu()

            # PyTorch buffers (created using 'register_buffer') shouldn't be trained
            torch_buffers = list(dict(self._torch_model.named_buffers()).keys())
            self.options.utils.frozen_weights.extend(torch_buffers)

            # Export to ONNX
            self._onnx_model = self._convert_torch_model_loss_fn_to_onnx(inputs, 'cpu')

            # Post processing for ONNX models expported from PyTorch
            if self.options._internal_use.enable_internal_postprocess:
                self._onnx_model = postprocess.run_postprocess(self._onnx_model)
            if self.options._internal_use.extra_postprocess:
                self._onnx_model = self.options._internal_use.extra_postprocess(self._onnx_model)

        optimizer_state_dict = {}
        if self._load_state_dict:
            optimizer_state_dict = self._load_state_dict()

        self._init_session(optimizer_state_dict,
                           session_options=self.options.session_options,
                           provider_options=self.options._validated_opts['provider_options'])

    def _init_session(self, optimizer_state_dict={},
                      session_options=None,
                      provider_options=None):
        if self._onnx_model is None:
            return

        if self.options.utils.run_symbolic_shape_infer:
            self._onnx_model = SymbolicShapeInference.infer_shapes(self._onnx_model, auto_merge=True, guess_output_rank=True)

        # Create training session used by train_step
        # pass all optimizer states to the backend
        self._create_ort_training_session(optimizer_state_dict,
                                          session_options=session_options, 
                                          provider_options=provider_options)

        # Update model description to update dtype when mixed precision is enabled
        # C++ backend modifies model's output dtype from float32 to float16 for mixed precision
        # Note that for training we must use float32 and for evaluation we must use float16
        for idx, o_desc in enumerate(self.model_desc.outputs):
            if (self.options.mixed_precision.enabled and o_desc.dtype == torch.float32 and
                    not self._training_session.is_output_fp32_node(o_desc.name)):
                self.model_desc.add_type_to_output_description(idx, o_desc.dtype, torch.float16)

        # Update model description
        self._model_desc_inputs_with_lr = [*self.model_desc.inputs, self.model_desc.learning_rate]

        # Update Mixed Precision, if applicable
        if self.options.mixed_precision.enabled:
            self.model_desc.loss_scale_input = self._training_session.loss_scale_input_name
            self._model_desc_inputs_with_lr_and_loss_scale = [
                *self._model_desc_inputs_with_lr, self.model_desc.loss_scale_input]
            self.model_desc.all_finite = _utils.get_all_gradients_finite_name_from_session(self._training_session)
            self._model_desc_outputs_with_all_finite = [*self.model_desc.outputs, self.model_desc.all_finite]
        elif self.options.mixed_precision.loss_scaler:
            raise ValueError("Loss Scaler cannot be specified when Mixed Precision is not enabled")

        # Update Loss Scaler Input Name, if applicable
        if self.options.mixed_precision.enabled and self.options.mixed_precision.loss_scaler:
            self.options.mixed_precision.loss_scaler.input_name = self.model_desc.loss_scale_input.name
        elif not self.options.mixed_precision.enabled and self.options.mixed_precision.loss_scaler:
            raise ValueError("Loss Scaler cannot be specified when Mixed Precision is not enabled")

        # Update Gradient Accumulation, if applicable
        if self.options.batch.gradient_accumulation_steps > 1:
            self.model_desc.gradient_accumulation = _utils.get_gradient_accumulation_name_from_session(self._training_session)
            self._model_desc_outputs_with_gradient_accumulation = [
                *self.model_desc.outputs, self.model_desc.gradient_accumulation]

        # TODO: Remove when experimental checkpoint functions are removed
        if self._state_dict:
            checkpoint.experimental_load_state_dict(self, self._state_dict, self._load_state_dict_strict)
            self._state_dict_debug = self._state_dict
        self._state_dict = {}

    def _prepare_model_input(self, inputs_desc, lr, loss_scale, *inputs, **kwargs):
        # Normalize input to tuple of samples
        if type(inputs) == tuple and len(inputs) == 1 and type(inputs[0]) == list:
            input = tuple(inputs[0])
        else:
            input = inputs

        # Append input from 'kwargs'
        for input_desc in inputs_desc:
            if input_desc.name in kwargs:
                input = input + (kwargs[input_desc.name],)

        # Append learning rate
        extra_inputs = 0
        if lr is not None:
            lr = torch.tensor([lr])
            input += (lr,)
            extra_inputs += 1

        # Append loss scale
        if loss_scale is not None:
            assert self.options.mixed_precision.enabled, "Loss scale cannot be used without mixed precision"
            loss_scale = torch.tensor([loss_scale])
            input += (loss_scale,)
            extra_inputs += 1

        # Only assert length of input when fetches is not used
        assert self._train_step_info.fetches or len(self.model_desc.inputs) + extra_inputs == len(input)
        return input

    def _resolve_symbolic_dimensions(self, inputs, inputs_desc, outputs_desc):
        outputs = copy.deepcopy(outputs_desc)
        resolved_dims = {}
        for input, i_desc in zip(inputs, inputs_desc):
            for i_idx, i_axis in enumerate(i_desc.shape):
                if isinstance(i_axis, str):
                    if i_axis not in resolved_dims:
                        resolved_dims[i_axis] = input.size()[i_idx]
                    else:
                        assert resolved_dims[i_axis] == input.size()[i_idx],\
                            f"Mismatch in dynamic shape {i_axis}"

        for o_desc in outputs:
            for idx_o, o_axis in enumerate(o_desc.shape):
                if isinstance(o_axis, str):
                    o_desc.shape[idx_o] = resolved_dims[o_axis]

        unknown_dim = [o_desc.name for dim in o_desc.shape for o_desc in outputs if isinstance(dim, str)]
        if unknown_dim:
            raise RuntimeError(f"Cannot execute model with unknown output dimensions ({unknown_dim}")

        return outputs

    def _training_session_run_helper(self, is_train, inputs, inputs_desc, outputs_desc, run_options=None):
        # Select IO binding
        if is_train:
            iobinding = self._train_io_binding
        else:
            iobinding = self._eval_io_binding

        # Get the list of the actual session inputs because unused inputs can be removed.
        input_nodes = self._training_session.get_inputs()
        input_node_names = [input_node.name for input_node in input_nodes]

        # Bind input tensors
        for input, input_desc in zip(inputs, inputs_desc):
            if input_desc.name in input_node_names:
                device_index = _utils.get_device_index_from_input(input)
                iobinding.bind_input(input_desc.name,
                                    input.device.type,
                                    device_index,
                                    _utils.dtype_torch_to_numpy(input.dtype),
                                    list(input.size()),
                                    input.data_ptr())

        # Bind output tensors
        outputs_desc_resolved = self._resolve_symbolic_dimensions(inputs, inputs_desc, outputs_desc)
        result = {}
        for output_desc in outputs_desc_resolved:
            target_device = self.options.device.id
            if self.options.mixed_precision.enabled and output_desc.name == self.model_desc.all_finite.name:
                # Keep all finite flag on CPU to match backend implementation
                # This prevents CPU -> GPU -> CPU copies between frontend and backend
                target_device = 'cpu'
            # the self.options.device may be a device that pytorch does not recognize.
            # in that case, we temporary prefer to leave the input/output on CPU and let ORT session 
            # to move the data between device and host. 
            # so output will be on the same device as input.
            try:
                test_pt_device = torch.device(target_device)
            except:
                #in this case, input/output must on CPU
                assert(input.device.type == 'cpu')
                target_device = 'cpu'
            
            torch_tensor = torch.zeros(output_desc.shape, device=target_device,
                                       dtype=output_desc.dtype_amp if output_desc.dtype_amp else output_desc.dtype)
            iobinding.bind_output(output_desc.name, torch_tensor.device.type, _utils.get_device_index(target_device),
                                  _utils.dtype_torch_to_numpy(torch_tensor.dtype),
                                  list(torch_tensor.size()), torch_tensor.data_ptr())
            result[output_desc.name] = torch_tensor

        # Run a train/eval step
        self._training_session.run_with_iobinding(iobinding, run_options)
        return result

    def _update_onnx_model_initializers(self, state_tensors):
        r""" Updates ONNX graph initializers with state_tensors's values

        Usually called to save or load an ONNX model.

        The tensors names of state_tensors are compared to all ONNX initializer tensors
        and when the name matches, the ONNX graph is updated with the new value.
        """
        assert isinstance(state_tensors, dict), "state_tensors must be a dict"

        new_weights = []
        replace_indices = []
        for i, w in enumerate(self._onnx_model.graph.initializer):
            if w.name in state_tensors:
                new_weights.append(onnx.numpy_helper.from_array(state_tensors[w.name], w.name))
                replace_indices.append(i)
        replace_indices.sort(reverse=True)
        for w_i in replace_indices:
            del self._onnx_model.graph.initializer[w_i]
        self._onnx_model.graph.initializer.extend(new_weights)

    def _extract_model_states(self, state_dict, pytorch_format):
        """Extract model states from the training session and load into the state_dict"""

        model_states = self._training_session.get_model_state(include_mixed_precision_weights=False)
        state_dict[_utils.state_dict_model_key()] = {}

        # extract trained model weights from the training session
        for precision in model_states:
            state_dict[_utils.state_dict_model_key()][precision] = {}
            for model_state_key in model_states[precision]:
                if pytorch_format:
                    state_dict[_utils.state_dict_model_key()][precision][model_state_key] = \
                        torch.from_numpy(model_states[precision][model_state_key])
                else:
                    state_dict[_utils.state_dict_model_key()][precision][model_state_key] = \
                        model_states[precision][model_state_key]

        # extract untrained (frozen) model weights
        for node in self._onnx_model.graph.initializer:
            if node.name not in state_dict[_utils.state_dict_model_key()][_utils.state_dict_full_precision_key()] and \
                node.name in self.options.utils.frozen_weights:
                if pytorch_format:
                    state_dict[_utils.state_dict_model_key()][_utils.state_dict_full_precision_key()][node.name] = \
                        torch.from_numpy(onnx.numpy_helper.to_array(node))
                else:
                    state_dict[_utils.state_dict_model_key()][_utils.state_dict_full_precision_key()][node.name] = \
                        onnx.numpy_helper.to_array(node)

    def _extract_trainer_options(self, state_dict):
        """Extract relevant trainer configuration and load it into the state_dict"""

        mixed_precision = _utils.state_dict_trainer_options_mixed_precision_key()
        zero_stage = _utils.state_dict_trainer_options_zero_stage_key()
        world_rank = _utils.state_dict_trainer_options_world_rank_key()
        world_size = _utils.state_dict_trainer_options_world_size_key()
        optimizer_name = _utils.state_dict_trainer_options_optimizer_name_key()
        D_size = _utils.state_dict_trainer_options_data_parallel_size_key()
        H_size = _utils.state_dict_trainer_options_horizontal_parallel_size_key()

        state_dict[_utils.state_dict_trainer_options_key()] = {}
        state_dict[_utils.state_dict_trainer_options_key()][mixed_precision] = self.options.mixed_precision.enabled
        state_dict[_utils.state_dict_trainer_options_key()][zero_stage] = \
            self.options.distributed.deepspeed_zero_optimization.stage
        state_dict[_utils.state_dict_trainer_options_key()][world_rank] = self.options.distributed.world_rank
        state_dict[_utils.state_dict_trainer_options_key()][world_size] = self.options.distributed.world_size
        state_dict[_utils.state_dict_trainer_options_key()][optimizer_name] = self.optim_config.name
        state_dict[_utils.state_dict_trainer_options_key()][D_size] = self.options.distributed.data_parallel_size
        state_dict[_utils.state_dict_trainer_options_key()][H_size] = self.options.distributed.horizontal_parallel_size

    def state_dict(self, pytorch_format=False):
        """Returns a dictionary with model, and optionally, optimizer states

        The returned dictionary contains the following information:
        - Model and optimizer states
        - Required ORTTrainerOptions settings
        - Distributed training information, such as but not limited to ZeRO

        Structure of the returned dictionary:
        - When `pytorch_format = False`
        schema:
        {
            "model":
            {
                type: dict,
                schema:
                {
                    "full_precision":
                    {
                        type: dict,
                        schema:
                        {
                            model_weight_name:
                            {
                                type: array
                            }
                        }
                    }
                }
            },
            "optimizer":
            {
                type: dict,
                schema:
                {
                    model_weight_name:
                    {
                        type: dict,
                        schema:
                        {
                            "Moment_1":
                            {
                                type: array
                            },
                            "Moment_2":
                            {
                                type: array
                            },
                            "Update_Count":
                            {
                                type: array,
                                optional: True # present if optimizer is adam, absent otherwise
                            }
                        }
                    },
                    "shared_optimizer_state":
                    {
                        type: dict,
                        optional: True, # present optimizer is shared, absent otherwise.
                        schema:
                        {
                            "step":
                            {
                                type: array,
                            }
                        }
                    }
                }
            },
            "trainer_options":
            {
                type: dict,
                schema:
                {
                    "mixed_precision":
                    {
                        type: bool
                    },
                    "zero_stage":
                    {
                        type: int
                    },
                    "world_rank":
                    {
                        type: int
                    },
                    "world_size":
                    {
                        type: int
                    },
                    "optimizer_name":
                    {
                        type: str
                    },
                    "data_parallel_size":
                    {
                        type: int
                    },
                    "horizontal_parallel_size":
                    {
                        type: int
                    }
                }
            },
            "partition_info":
            {
                type: dict,
                optional: True, # present if states partitioned, else absent
                schema:
                {
                    model_weight_name:
                    {
                        type: dict,
                        schema:
                        {
                            "original_dim":
                            {
                                type: array
                            },
                            "megatron_row_partition":
                            {
                                type: int
                            }
                        }
                    }
                }
            }
        }
        - When `pytorch_format = True`
        schema:
        {
            model_weight_name:
            {
                type: tensor
            }
        }

        Args:
            pytorch_format: boolean flag to select either ONNX Runtime or PyTorch state schema

        Returns:
            A dictionary with `ORTTrainer` state
        """
        if not self._training_session:
            warnings.warn("ONNX Runtime training session is not initialized yet. "
                          "Please run train_step or eval_step at least once before calling ORTTrainer.state_dict().",
                          UserWarning)
            return self._load_state_dict.args[0] if self._load_state_dict else {}

        state_dict = {}

        # load training session model states into the state_dict
        self._extract_model_states(state_dict, pytorch_format)
        if pytorch_format:
            if self.options.distributed.deepspeed_zero_optimization.stage > 0:
                warnings.warn("Incomplete state_dict: ZeRO enabled", UserWarning)
            if self.options.distributed.horizontal_parallel_size > 1:
                warnings.warn("Incomplete state_dict: Megatron enabled", UserWarning)
            # if pytorch_format is true, return a flat dictionary with only model states
            # which is compatible with a PyTorch model
            return state_dict[_utils.state_dict_model_key()][_utils.state_dict_full_precision_key()]

        # load training session optimizer states into the state_dict
        state_dict[_utils.state_dict_optimizer_key()] = self._training_session.get_optimizer_state()

        # extract the relevant training configuration from the trainer and load them into the state_dict
        self._extract_trainer_options(state_dict)

        # add partition information in case of a distributed run
        if self.options.distributed.deepspeed_zero_optimization.stage > 0 or self.options.distributed.horizontal_parallel_size > 1:
            state_dict[_utils.state_dict_partition_info_key()] = self._training_session.get_partition_info_map()

        return state_dict

    def _load_model_states(self, state_dict, strict):
        """Load the model states onto the onnx model graph"""

        if _utils.state_dict_model_key() not in state_dict:
            return

        # collect all initializer names from the current onnx graph
        assert self._onnx_model, "ONNX model graph is not exported"
        initializer_names = {node.name for node in self._onnx_model.graph.initializer}

        # loaded_initializers dict will be loaded with all the model states from the state dictionary
        # that are found in the initializer_names dictionary
        loaded_initializers = {}

        # copy over model states from the input state dict onto the onnx model
        for precision, precision_states in state_dict[_utils.state_dict_model_key()].items():
            for state_key, state_value in precision_states.items():
                if state_key in initializer_names:
                    loaded_initializers[state_key] = state_value
                elif strict:
                    raise RuntimeError("Unexpected key: {} in state_dict[model][{}]".format(state_key, precision))

        # update onnx model from loaded initializers
        self._update_onnx_model_initializers(loaded_initializers)

    def _load_optimizer_states(self, current_state_dict, state_dict):
        """Load the optimizer states onto the training session state dictionary"""

        def _check_optimizer_mismatch(state_dict):
            """Assert that the loaded optimizer has the same config as the current training session config"""

            # the state_dict optimizer_name can be a byte string (if coming from checkpoint file)
            # or can be a regular string (coming from user)
            optimizer_name = \
                state_dict[_utils.state_dict_trainer_options_key()][_utils.state_dict_trainer_options_optimizer_name_key()]

            # optimizer_name can be either a regular string or a byte string.
            # if it is a byte string, convert to regular string using decode()
            # if it is a regular string, do nothing to it
            try:
                optimizer_name = optimizer_name.decode()
            except AttributeError:
                pass
            assert self.optim_config.name == optimizer_name, \
                "Optimizer mismatch: expected {}, got {}".format(self.optim_config.name, optimizer_name)

        if _utils.state_dict_optimizer_key() not in state_dict:
            return

        # check optimizer config names are the same for current session and the sessino being loaded
        _check_optimizer_mismatch(state_dict)

        # create an entry for the optimizer in the training session state dictionary
        if _utils.state_dict_optimizer_key() not in current_state_dict:
            current_state_dict[_utils.state_dict_optimizer_key()] = {}

        # copy over optimizer states from the input state dict onto the training session state dict
        for model_state_key, optimizer_dict in state_dict[_utils.state_dict_optimizer_key()].items():
            if model_state_key not in current_state_dict[_utils.state_dict_optimizer_key()]:
                current_state_dict[_utils.state_dict_optimizer_key()][model_state_key] = {}
            for optimizer_state_key, optimizer_state_value in optimizer_dict.items():
                current_state_dict[_utils.state_dict_optimizer_key()][model_state_key][optimizer_state_key] = \
                    optimizer_state_value


    def _load_state_dict_impl(self, state_dict, strict=True):
        """Load the state dictionary onto the onnx model and on the training session graph"""

        # clear the callable partial
        self._load_state_dict = None

        def _mismatch_keys(keys1, keys2, in_error_str, allow_unexpected=False):
            """Find out the missing and the unexpected keys in two dictionaries

            Throws a runtime error if missing or unexpected keys are found
            - Keys in keys1 not in keys2 will be marked as missing
            - Keys in keys2 not in keys1 will be marked as unexpected
            """
            keys1 = set(keys1)
            keys2 = set(keys2)
            missing_keys = list(keys1 - keys2)
            unexpected_keys = list(keys2 - keys1)
            if len(missing_keys) > 0:
                raise RuntimeError("Missing keys: {} in {}".format(missing_keys, in_error_str))
            if len(unexpected_keys) > 0 and not allow_unexpected:
                raise RuntimeError("Unexpected keys: {} in {}".format(unexpected_keys, in_error_str))

        def _check_model_key_mismatch(current_state_dict, state_dict, allow_unexpected=False):
            """Check if there is any mismatch in the model sub state dictionary between the two state_dicts"""

            # check unxexpected and missing precision keys in the model state_dict compared to the training
            # session model state_dict
            _mismatch_keys(current_state_dict[_utils.state_dict_model_key()],
                                   state_dict[_utils.state_dict_model_key()], 'state_dict[model]', allow_unexpected)

            # check for model state key mismatch
            for precision_key in current_state_dict[_utils.state_dict_model_key()]:
                _mismatch_keys(current_state_dict[_utils.state_dict_model_key()][precision_key],
                                       state_dict[_utils.state_dict_model_key()][precision_key],
                                       'state_dict[model][{}]'.format(precision_key), allow_unexpected)

        def _check_optimizer_key_mismatch(current_state_dict, state_dict, allow_unexpected=False):
            """Check if there is any mismatch in the optimizer sub state dictionary between the two state_dicts"""

            # check for model state key mismatch for the optimizer state_dict
            _mismatch_keys(current_state_dict[_utils.state_dict_optimizer_key()],
                                   state_dict[_utils.state_dict_optimizer_key()],
                                   'state_dict[optimizer]', allow_unexpected)

            # check for optimizer state keys mismatch
            for model_state_key in current_state_dict[_utils.state_dict_optimizer_key()]:
                _mismatch_keys(current_state_dict[_utils.state_dict_optimizer_key()][model_state_key],
                                       state_dict[_utils.state_dict_optimizer_key()][model_state_key],
                                       'state_dict[optimizer][{}]'.format(model_state_key), allow_unexpected)

        def _check_key_mismatch(current_state_dict, state_dict, allow_unexpected=False):
            """Check if there is a mismatch in the keys (model and optimizer) in the two state_dicts"""

            # check presence of 'model' in the input state_dict
            if _utils.state_dict_model_key() in state_dict:
                _check_model_key_mismatch(current_state_dict, state_dict, allow_unexpected)
            else:
                warnings.warn("Missing key: model in state_dict", UserWarning)
            # check presence of 'optimizer' in the input state_dict
            if _utils.state_dict_optimizer_key() in state_dict:
                _check_optimizer_key_mismatch(current_state_dict, state_dict, allow_unexpected)
            else:
                warnings.warn("Missing key: optimizer in state_dict", UserWarning)

        # extract state dict from the current training session. this is to persist the states between
        # two training sessions.
        # for example, if user provided only the model states, the optimizer states from the current
        # training session must be persisted
        current_state_dict = {}
        if self._training_session:
            current_state_dict = self.state_dict()
            if strict:
                # for Zero enabled, the current trainer might not have the complete state, and we must allow 
                # extra keys to be present in the state dict
                allow_unexpected = True if self.options.distributed.deepspeed_zero_optimization.stage > 0 else False
                _check_key_mismatch(current_state_dict, state_dict, allow_unexpected)

        # load the model states from the input state dictionary into the onnx graph
        self._load_model_states(state_dict, strict)

        # load the optimizer states from the input state dictionary into the training session states
        # dictionary
        self._load_optimizer_states(current_state_dict, state_dict)

        return current_state_dict[_utils.state_dict_optimizer_key()] if \
            _utils.state_dict_optimizer_key() in current_state_dict else {}

    def load_state_dict(self, state_dict, strict=True):
        """Loads state_dict containing model/optimizer states into ORTTrainer

        The state_dict dictionary may contain the following information:
        - Model and optimizer states
        - Required ORTTrainerOptions settings
        - Distributed training information, such as but not limited to ZeRO

        Args:
            state_dict: state dictionary containing both model and optimizer states. The structure of this dictionary
                should be the same as the one that is returned by ORTTrainer.state_dict for the case when pytorch_format=False
            strict: boolean flag to strictly enforce that the input state_dict keys match the keys from ORTTrainer.state_dict
        """

        # if onnx graph has not been initialized, loading of states will be put on hold.
        # a copy of the state_dict and other arguments to the function will be stored until the onnx graph has
        # been initialized. Once the graph is initialized, the desired states will be loaded onto the grpah
        if not self._training_session:
            self._load_state_dict = partial(self._load_state_dict_impl, state_dict, strict=strict)
            return

        # load states onto the frontend onnx graph
        optimizer_state_dict = self._load_state_dict_impl(state_dict, strict=strict)

        # create a new training session after loading initializer states onto the onnx graph
        # pass the populated states to the training session to populate the backend graph
        self._init_session(optimizer_state_dict,
                           session_options=self.options.session_options,
                           provider_options=self.options._validated_opts['provider_options'])

    def save_checkpoint(self, path, user_dict={}, include_optimizer_states=True):
        """Persists ORTTrainer state dictionary on disk along with user_dict.

        Saves the state_dict along with the user_dict to a file specified by path.

        Args:
            path: string representation to a file path or a python file-like object.
                if file already exists at path, an exception is raised.
            user_dict: custom data to be saved along with the state_dict. This data will be returned
                to the user when load_checkpoint is called.
            include_optimizer_states: boolean flag indicating whether or not to persist the optimizer states.
                on load_checkpoint, only model states will be loaded if include_optimizer_states==True
        """

        # extract state_dict to be saved in the checkpoint
        state_dict = self.state_dict()

        # if user_dict is provided, serialize to bytes and convert to hex string.
        # this helps in loading the types as they are given by the user since hdf5
        # converts to numpy types otherwise
        if bool(user_dict):
            state_dict[_utils.state_dict_user_dict_key()] = _checkpoint_storage.to_serialized_hex(user_dict)

        # if include_optimizer_states is False, only save the model states in the checkpoint file
        if not include_optimizer_states:
            if _utils.state_dict_optimizer_key() in state_dict:
                del state_dict[_utils.state_dict_optimizer_key()]

        _checkpoint_storage.save(state_dict, path)

    def _aggregation_required(self, loaded_trainer_options):
        """Checks if aggregation is required for the loading the state_dict into the ORTTrainer"""

        # To load states in the backend, aggregation is required for every ZeRO 
        # or Megatron checkpoint
        return loaded_trainer_options[_utils.state_dict_trainer_options_zero_stage_key()] > 0 or \
                loaded_trainer_options[_utils.state_dict_trainer_options_horizontal_parallel_size_key()] > 1

    def load_checkpoint(self, *paths, strict=True):
        """Loads the saved checkpoint state dictionary into the ORTTrainer

        Reads the saved checkpoint files specified by paths from disk and loads the state dictionary
        onto the ORTTrainer.
        Aggregates the checkpoint files if aggregation is required.

        Args:
            paths: one or more files represented as strings where the checkpoint is saved
            strict: boolean flag to strictly enforce that the saved checkpoint state_dict
                keys match the keys from ORTTrainer.state_dict
        Returns:
            dictionary that the user had saved when calling save_checkpoint
        """
        state_dict = {}

        # check if aggregation is required
        loaded_trainer_options = _checkpoint_storage.load(paths[0], key=_utils.state_dict_trainer_options_key())
        if self._aggregation_required(loaded_trainer_options):
            # if aggregation is required, aggregation logic must be run on the saved checkpoints
            state_dict = checkpoint.aggregate_checkpoints(paths, pytorch_format=False)
        else:
            # if aggregation is not required, there must only be a single file that needs to be loaded
            assert len(paths) == 1, "Expected number of files to load: 1, got {}".format(len(paths))
            state_dict = _checkpoint_storage.load(paths[0])

        # extract user dict from the saved checkpoint
        user_dict = {}
        if _utils.state_dict_user_dict_key() in state_dict:
            user_dict = _checkpoint_storage.from_serialized_hex(state_dict[_utils.state_dict_user_dict_key()])
            del state_dict[_utils.state_dict_user_dict_key()]

        self.load_state_dict(state_dict, strict=strict)

        return user_dict
