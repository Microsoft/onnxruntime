import copy
import numpy as np
import os
import torch

from numpy.testing import assert_allclose
from onnxruntime.capi.ort_trainer import ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import orttrainer
from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule._graph_execution_manager_factory import GraphExecutionManagerFactory


def assert_model_outputs(output_a, output_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether output_a and output_b difference is within specified tolerance

    Args:
        output_a, output_b (list): Two list with of numeric values
        verbose (bool, default is False): if True, prints absolute difference for each weight
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(output_a, list) and isinstance(output_b, list),\
        "output_a and output_b must be list of numbers"
    assert len(output_a) == len(output_b), "output_a and output_b must have the same length"

    # for idx in range(len(output_a)):
    assert_allclose(output_a, output_b, rtol=rtol, atol=atol, err_msg=f"Model output value mismatch")

def assert_onnx_weights(model_a, model_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether weight difference between models a and b differences are within specified tolerance

    Compares the weights of two different ONNX models (model_a and model_b)
    and raises AssertError when they diverge by more than atol or rtol

    Args:
        model_a, model_b (ORTTrainer): Two instances of ORTTrainer with the same model structure
        verbose (bool, default is False): if True, prints absolute difference for each weight
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, orttrainer.ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b._training_session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol)


def assert_legacy_onnx_weights(model_a, model_b, verbose=False, rtol=1e-7, atol=0):
    r"""Asserts whether weight difference between models a and b differences are within specified tolerance

    Compares the weights of a legacy model model_a and experimental model_b model
    and raises AssertError when they diverge by more than atol or rtol.

    Args:
        model_a (ORTTrainer): Instance of legacy ORTTrainer
        model_b (ORTTrainer): Instance of experimental ORTTrainer
        verbose (bool, default is False): if True, prints absolute difference for each weight.
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """
    assert isinstance(model_a, orttrainer.ORTTrainer) and isinstance(model_b, Legacy_ORTTrainer)
    state_dict_a, state_dict_b = model_a._training_session.get_state(), model_b.session.get_state()
    assert len(state_dict_a.items()) == len(state_dict_b.items())
    _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol)


def _assert_state_dict_weights(state_dict_a, state_dict_b, verbose, rtol, atol):
    r"""Asserts whether dicts a and b value differences are within specified tolerance

    Compares the weights of two model's state_dict dicts and raises AssertError
    when they diverge by more than atol or rtol

    Args:
        model_a (ORTTrainer): Instance of legacy ORTTrainer
        model_b (ORTTrainer): Instance of experimental ORTTrainer
        verbose (bool, default is False): if True, prints absolute difference for each weight.
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 1e-4): Max absolute difference
    """

    for (a_name, a_val), (b_name, b_val) in zip(state_dict_a.items(), state_dict_b.items()):
        np_a_vals = np.array(a_val).flatten()
        np_b_vals = np.array(b_val).flatten()
        assert np_a_vals.shape == np_b_vals.shape
        if verbose:
            print(f'Weight name: {a_name}: absolute difference: {np.abs(np_a_vals-np_b_vals).max()}')
        assert_allclose(a_val, b_val, rtol=rtol, atol=atol, err_msg=f"Weight mismatch for {a_name}")

def assert_optim_state(expected_state, actual_state, rtol=1e-7, atol=0):
    r"""Asserts whether optimizer state differences are within specified tolerance

    Compares the expected and actual optimizer states of dicts and raises AssertError
    when they diverge by more than atol or rtol.
    The optimizer dict is of the form:
        model_weight_name:
            {
                "Moment_1": moment1_tensor,
                "Moment_2": moment2_tensor,
                "Update_Count": update_tensor # if optimizer is adam, absent otherwise
            },
        ...
        "shared_optimizer_state": # if optimizer is shared, absent otherwise. 
                                    So far, only lamb optimizer uses this.
        {
            "step": step_tensor # int array of size 1
        }

    Args:
        expected_state (dict(dict())): Expected optimizer state 
        actual_state (dict(dict())): Actual optimizer state
        rtol (float, default is 1e-7): Max relative difference
        atol (float, default is 0): Max absolute difference
    """
    assert expected_state.keys() == actual_state.keys()
    for param_name, a_state in actual_state.items():
        for k,v in a_state.items():
            assert_allclose(v, expected_state[param_name][k], rtol=rtol, atol=atol,
                            err_msg=f"Optimizer state mismatch for param {param_name}, key {k}")

def is_dynamic_axes(model):
    # Check inputs
    for inp in model._module_manager._execution_manager(model._is_training())._optimized_onnx_model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_param and not isinstance(dim.dim_param, str):
                    return False

    # Check outputs
    for out in model._module_manager._execution_manager(model._is_training())._optimized_onnx_model.graph.output:
        shape = out.type.tensor_type.shape
        if shape:
            for dim in shape.dim:
                if dim.dim_param and not isinstance(dim.dim_param, str):
                    return False
    return True

# TODO: thiagofc: Checkpoint related for redesign
def _get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError("Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

# Depending on calling backward() from which outputs, it's possible that grad of some weights are not calculated.
# none_pt_params is to tell what these weights are, so we will not compare the tensors.
def assert_gradients_match_and_reset_gradient(ort_model, pt_model, none_pt_params=[], reset_gradient=True, rtol=1e-05, atol=1e-06):
    ort_named_params = list(ort_model.named_parameters())
    pt_named_params = list(pt_model.named_parameters())
    assert len(ort_named_params) == len(pt_named_params)

    for ort_named_param, pt_named_param in zip(ort_named_params, pt_named_params):
        ort_name, ort_param = ort_named_param
        pt_name, pt_param = pt_named_param

        assert pt_name in ort_name
        if pt_name in none_pt_params:
            assert pt_param.grad is None
            assert ort_param.grad is None or not torch.is_nonzero(torch.count_nonzero(ort_param.grad))
        else:
            assert_values_are_close(ort_param.grad, pt_param.grad, rtol=rtol, atol=atol)

        if reset_gradient:
            ort_param.grad = None
            pt_param.grad = None

def assert_values_are_close(input, other, rtol=1e-05, atol=1e-06):
    are_close = torch.allclose(input, other, rtol=rtol, atol=atol)
    if not are_close:
        abs_diff = torch.abs(input - other)
        abs_other = torch.abs(other)
        max_atol = torch.max((abs_diff - rtol * abs_other))
        max_rtol = torch.max((abs_diff - atol) / abs_other)
        err_msg = "The maximum atol is {}, maximum rtol is {}".format(max_atol, max_rtol)
        assert False, err_msg

def enable_custom_autograd_function(module):
    for mode in [True, False]:
        module._execution_manager(mode)._enable_custom_autograd_function = True

def run_with_pytorch_on_device(device, model, input_list, label_input, is_eval_mode=False):
    model.to(device)
    if is_eval_mode:
        model.eval()
    else:
        model.train()

    inputs_on_device = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_device)
    forward_outputs = [output]
    grad_outputs = []

    if not is_eval_mode:
        criterion = torch.nn.MSELoss()
        target = label_input.to(device)
        loss = criterion(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_outputs.append(param.grad)
    return forward_outputs, grad_outputs

def run_with_ort_on_device(device, model, input_list, label_input, is_eval_mode=False):
    model = copy.deepcopy(model)
    model.to(device)
    model = ORTModule(model)
    enable_custom_autograd_function(model)
    if is_eval_mode:
        model.eval()
    else:
        model.train()

    inputs_on_device = [input_.to(device) for input_ in input_list]
    output = model(*inputs_on_device)
    forward_outputs = [output]
    grad_outputs = []

    if not is_eval_mode:
        criterion = torch.nn.MSELoss()
        target = label_input.to(device)
        loss = criterion(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_outputs.append(param.grad)
    return forward_outputs, grad_outputs

def compare_tensor_list(val_list_a, val_list_b):
    for val_a, val_b in zip(val_list_a, val_list_b):
       assert_values_are_close(val_a, val_b, atol=1e-7, rtol=1e-6)

def run_training_test_and_compare(pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, ignore_grad_compare=False):
    cpu = torch.device("cpu")

    def cpu_barrier_func():
        pass
    run_training_test_on_device_and_compare(
        cpu, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, cpu_barrier_func, ignore_grad_compare)

    def cuda_barrier_func():
        torch.cuda.synchronize()
    cuda = torch.device('cuda:0')
    run_training_test_on_device_and_compare(
        cuda, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, cuda_barrier_func, ignore_grad_compare)

def run_training_test_on_device_and_compare(device, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, barrier_func, ignore_grad_compare=False):
    repeats = 16
    for i in range(repeats):
        m = pt_model_builder_func()
        x = pt_model_inputs_generator()

        m_ort = copy.deepcopy(m)
        x_ort = copy.deepcopy(x)

        outputs, grads = run_with_pytorch_on_device(
            device, m, [x], pt_model_label_input)
        barrier_func()

        outputs_ort, grads_ort = run_with_ort_on_device(
            device, m_ort, [x_ort], pt_model_label_input)
        barrier_func()

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)

        # For some test, it is expected the diff might be big due to inconsistent computation orders.
        if ignore_grad_compare is False:
            val_list_a = [o.detach().cpu() for o in grads if o is not None]
            val_list_b = [o.detach().cpu() for o in grads_ort if o is not None]
            compare_tensor_list(val_list_a, val_list_b)

def run_evaluate_test_and_compare(pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input):
    cpu = torch.device("cpu")

    def cpu_barrier_func():
        pass

    run_evaluate_test_on_device_and_compare(
        cpu, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, cpu_barrier_func)

    def cuda_barrier_func():
        torch.cuda.synchronize()
        pass

    cuda = torch.device('cuda:0')
    run_evaluate_test_on_device_and_compare(
        cuda, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, cuda_barrier_func)

def run_evaluate_test_on_device_and_compare(device, pt_model_builder_func, pt_model_inputs_generator, pt_model_label_input, barrier_func):
    repeats = 16
    for i in range(repeats):
        m = pt_model_builder_func()
        x = pt_model_inputs_generator()

        m_ort = copy.deepcopy(m)
        x_ort = copy.deepcopy(x)

        outputs, grads = run_with_pytorch_on_device(
            device, m, [x], pt_model_label_input, is_eval_mode=True)
        barrier_func()

        outputs_ort, grads_ort = run_with_ort_on_device(
            device, m_ort, [x_ort], pt_model_label_input, is_eval_mode=True)
        barrier_func()

        val_list_a = [o.detach().cpu() for o in outputs if o is not None]
        val_list_b = [o.detach().cpu() for o in outputs_ort if o is not None]
        compare_tensor_list(val_list_a, val_list_b)
